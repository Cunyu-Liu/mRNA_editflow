"""Large source-conditioned latent-mixture XEditSetFlow V4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from core.route2_xeditcritic_v3 import EndpointConditionerV1
from core.route2_xeditsetflow_v3 import HybridSetFlowBlockV3


class XEditSetFlowV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditSetFlowV4Error(message)


@dataclass(frozen=True)
class MixtureSetFlowLossV4:
    total: torch.Tensor
    common_set_marginal: torch.Tensor
    source_candidate_coverage: torch.Tensor
    remaining_count: torch.Tensor
    mode_information: torch.Tensor
    active_state_count: int
    active_candidate_constraint_count: int


@dataclass(frozen=True)
class CommonSetMarginalLossV4:
    loss: torch.Tensor
    active_weight: torch.Tensor
    active_state_count: int


class XEditSetFlowV4(nn.Module):
    """18-block SetFlow with one trajectory-fixed source-level latent mode."""

    def __init__(
        self,
        *,
        assay_count: int,
        context_count: int,
        quantity_count: int,
        measurement_count: int,
        numerator_count: int,
        denominator_count: int,
        region_count: int = 2,
        pretrained_width: int = 768,
        model_width: int = 640,
        depth: int = 18,
        heads: int = 10,
        ffn_width: int = 2560,
        local_attention_window: int = 64,
        mode_count: int = 8,
        mode_residual_rank: int = 64,
        stop_bottleneck_width: int = 128,
        dropout: float = 0.10,
        support_floor: float = 1e-8,
        activation_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        _require(model_width % heads == 0, "SetFlow V4 width is not divisible by heads")
        _require(depth >= 1 and mode_count >= 1, "SetFlow V4 depth or mode count is invalid")
        _require(support_floor > 0.0, "SetFlow V4 support floor is nonpositive")
        self.model_width = int(model_width)
        self.mode_count = int(mode_count)
        self.support_floor = float(support_floor)
        self.activation_checkpointing = bool(activation_checkpointing)
        condition_width = min(320, model_width)
        category_width = max(16, condition_width // 8)
        self.condition_width = condition_width
        self.endpoint_conditioner = EndpointConditionerV1(
            quantity_count=quantity_count,
            measurement_count=measurement_count,
            numerator_count=numerator_count,
            denominator_count=denominator_count,
            assay_count=assay_count,
            context_count=context_count,
            region_count=region_count,
            output_width=condition_width,
            category_width=category_width,
        )
        nucleotide_width = max(32, model_width // 8)
        self.source_nucleotide = nn.Embedding(5, nucleotide_width, padding_idx=4)
        self.current_nucleotide = nn.Embedding(5, nucleotide_width, padding_idx=4)
        self.source_pretrained_projection = nn.Linear(pretrained_width, model_width)
        self.state_projection = nn.Linear(nucleotide_width * 2 + 4, model_width)
        self.input_norm = nn.LayerNorm(model_width)
        self.blocks = nn.ModuleList(
            HybridSetFlowBlockV3(
                width=model_width,
                heads=heads,
                ffn_width=ffn_width,
                window=local_attention_window,
                dilation=2 ** (index % 4),
                shifted=bool(index % 2),
                dropout=dropout,
                condition_width=condition_width,
            )
            for index in range(depth)
        )
        self.source_mode_projection = nn.Sequential(
            nn.Linear(pretrained_width + nucleotide_width, model_width),
            nn.GELU(),
            nn.LayerNorm(model_width),
        )
        self.mode_router = nn.Linear(model_width + condition_width, mode_count)
        self.mode_token_residuals = nn.ModuleList(
            nn.Sequential(
                nn.Linear(model_width, mode_residual_rank),
                nn.GELU(),
                nn.Linear(mode_residual_rank, model_width),
            )
            for _ in range(mode_count)
        )
        self.mode_substitution_heads = nn.ModuleList(
            nn.Linear(model_width, 4) for _ in range(mode_count)
        )
        self.mode_stop_attention = nn.ModuleList(
            nn.Linear(model_width, 1) for _ in range(mode_count)
        )
        self.mode_stop_heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(model_width + condition_width + 2, stop_bottleneck_width),
                nn.GELU(),
                nn.Linear(stop_bottleneck_width, 1),
            )
            for _ in range(mode_count)
        )
        self.remaining_count_head = nn.Sequential(
            nn.Linear(model_width + condition_width + 2, model_width),
            nn.GELU(),
            nn.Linear(model_width, 6),
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def parameter_counts_by_module(self) -> dict[str, int]:
        modules = {
            "endpoint": self.endpoint_conditioner,
            "state_input": nn.ModuleList(
                [
                    self.source_nucleotide,
                    self.current_nucleotide,
                    self.source_pretrained_projection,
                    self.state_projection,
                    self.input_norm,
                ]
            ),
            "hybrid_trunk": self.blocks,
            "mode_router": nn.ModuleList(
                [self.source_mode_projection, self.mode_router]
            ),
            "mode_token_residuals": self.mode_token_residuals,
            "mode_substitution_heads": self.mode_substitution_heads,
            "mode_stop_heads": nn.ModuleList(
                [self.mode_stop_attention, self.mode_stop_heads]
            ),
            "remaining_count_head": self.remaining_count_head,
        }
        counts = {
            name: sum(
                parameter.numel()
                for parameter in module.parameters()
                if parameter.requires_grad
            )
            for name, module in modules.items()
        }
        _require(
            sum(counts.values()) == self.trainable_parameter_count,
            "SetFlow V4 parameter module accounting is incomplete",
        )
        return counts

    def _condition(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.endpoint_conditioner(
            {
                "quantity": batch["quantity_ids"],
                "measurement": batch["measurement_ids"],
                "numerator": batch["numerator_ids"],
                "denominator": batch["denominator_ids"],
                "assay": batch["assay_ids"],
                "context": batch["context_ids"],
                "region": batch["region_ids"],
            }
        )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        source = batch["source_tokens"]
        current = batch["current_tokens"]
        padding = batch["padding_mask"]
        source_pretrained = batch["source_pretrained_tokens"]
        _require(source.shape == current.shape == padding.shape, "SetFlow V4 sequence geometry differs")
        _require(source_pretrained.shape[:2] == source.shape, "SetFlow V4 pretrained tokens do not align")
        valid = ~padding
        _require(bool(valid.any(dim=1).all().item()), "SetFlow V4 received an empty source sequence")
        edited = (source != current) & valid
        lengths = valid.sum(dim=1, keepdim=True)
        positions = torch.arange(source.shape[1], device=source.device)[None]
        normalized_position = positions / (lengths - 1).clamp_min(1)
        normalized_position = normalized_position.to(source_pretrained.dtype) * valid
        edited_count = edited.sum(dim=1).to(source_pretrained.dtype)
        remaining = batch["remaining_budget"].to(source_pretrained.dtype)
        assigned = edited_count + remaining
        progress = edited_count / assigned.clamp_min(1)
        condition = self._condition(batch)
        source_base = self.source_nucleotide(source)
        source_mode_tokens = self.source_mode_projection(
            torch.cat((source_pretrained, source_base), dim=-1)
        ) * valid.unsqueeze(-1)
        source_mode_pooled = source_mode_tokens.sum(dim=1) / lengths.to(
            source_mode_tokens.dtype
        )
        raw_mode_logits = self.mode_router(
            torch.cat((source_mode_pooled, condition), dim=-1)
        )
        mode_prior = 0.5 * torch.softmax(raw_mode_logits, dim=-1) + 0.5 / self.mode_count
        per_position_state = torch.cat(
            (
                source_base,
                self.current_nucleotide(current),
                edited.to(source_pretrained.dtype).unsqueeze(-1),
                normalized_position.unsqueeze(-1),
                remaining.log1p().view(-1, 1, 1).expand(-1, source.shape[1], -1),
                progress.view(-1, 1, 1).expand(-1, source.shape[1], -1),
            ),
            dim=-1,
        )
        hidden = self.input_norm(
            self.source_pretrained_projection(source_pretrained)
            + self.state_projection(per_position_state)
        ) * valid.unsqueeze(-1)
        for block in self.blocks:
            def block_forward(
                values: torch.Tensor,
                active_block: HybridSetFlowBlockV3 = block,
            ) -> torch.Tensor:
                return active_block(values, padding, condition)

            hidden = (
                checkpoint(
                    block_forward,
                    hidden,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
                if self.training
                and self.activation_checkpointing
                and torch.is_grad_enabled()
                else block_forward(hidden)
            )
        positions_legal = (
            valid
            & (~edited)
            & (batch["remaining_budget"][:, None] > 0)
        )
        alt_ids = torch.arange(4, device=source.device).view(1, 1, 4)
        legal_substitutions = positions_legal.unsqueeze(-1) & (
            alt_ids != source.unsqueeze(-1)
        )
        stop_legal = (batch["remaining_budget"] > 0).unsqueeze(-1)
        legal_mask = torch.cat(
            (legal_substitutions.reshape(source.shape[0], -1), stop_legal),
            dim=1,
        )
        mode_logits: list[torch.Tensor] = []
        mode_pooled: list[torch.Tensor] = []
        stop_features = (
            remaining.log1p().unsqueeze(-1),
            progress.unsqueeze(-1),
        )
        for mode_index in range(self.mode_count):
            mode_hidden = hidden + self.mode_token_residuals[mode_index](hidden)
            mode_hidden = mode_hidden * valid.unsqueeze(-1)
            substitution = self.mode_substitution_heads[mode_index](mode_hidden)
            substitution = substitution.masked_fill(
                ~legal_substitutions, -torch.inf
            )
            attention_logits = self.mode_stop_attention[mode_index](
                mode_hidden
            ).squeeze(-1).masked_fill(~valid, -torch.inf)
            attention = torch.softmax(attention_logits, dim=1)
            pooled = (mode_hidden * attention.unsqueeze(-1)).sum(dim=1)
            mode_pooled.append(pooled)
            stop = self.mode_stop_heads[mode_index](
                torch.cat((pooled, condition, *stop_features), dim=-1)
            ).masked_fill(~stop_legal, -torch.inf)
            mode_logits.append(
                torch.cat(
                    (substitution.reshape(source.shape[0], -1), stop), dim=1
                )
            )
        logits = torch.stack(mode_logits, dim=1)
        legal_by_mode = legal_mask[:, None, :].expand(-1, self.mode_count, -1)
        positive_rates = F.softplus(logits) + logits.new_tensor(self.support_floor)
        rates = torch.where(
            legal_by_mode, positive_rates, torch.zeros_like(positive_rates)
        )
        shared_pooled = torch.stack(mode_pooled, dim=1).mean(dim=1)
        remaining_count_logits = self.remaining_count_head(
            torch.cat((shared_pooled, condition, *stop_features), dim=-1)
        )
        return {
            "mode_rates": rates,
            "legal_action_mask": legal_mask,
            "mode_prior": mode_prior,
            "remaining_count_logits": remaining_count_logits,
        }


def select_trajectory_mode_rates_v4(
    mode_rates: torch.Tensor,
    mode_ids: torch.Tensor,
) -> torch.Tensor:
    """Select the one mode fixed at trajectory start without resampling it."""

    _require(mode_rates.ndim == 3, "SetFlow V4 mode rates are not batch x mode x action")
    _require(mode_ids.shape == (mode_rates.shape[0],), "SetFlow V4 trajectory mode ids do not align")
    _require(bool(((mode_ids >= 0) & (mode_ids < mode_rates.shape[1])).all().item()), "SetFlow V4 trajectory mode id is out of range")
    rows = torch.arange(mode_rates.shape[0], device=mode_rates.device)
    return mode_rates[rows, mode_ids]


def mixture_setflow_loss_v4(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    coverage_weight: float = 0.50,
    remaining_count_weight: float = 0.20,
    mode_information_weight: float = 0.05,
) -> MixtureSetFlowLossV4:
    """Common NLL + per-candidate coverage + count + mode-information loss."""

    rates = output["mode_rates"].float()
    legal = output["legal_action_mask"]
    prior = output["mode_prior"].float()
    common_positive = batch["common_positive_action_mask"]
    candidate_positive = batch["candidate_positive_action_mask"]
    candidate_valid = batch["candidate_valid_mask"]
    structural = batch["structural_budget_exhausted"]
    _require(rates.shape[:1] == legal.shape[:1] == prior.shape[:1], "SetFlow V4 loss batch geometry differs")
    _require(rates.shape[2] == legal.shape[1] == common_positive.shape[1] == candidate_positive.shape[2], "SetFlow V4 loss action geometry differs")
    _require(candidate_positive.shape[:2] == candidate_valid.shape, "SetFlow V4 candidate masks differ")
    _require(bool(torch.all(common_positive <= legal).item()), "SetFlow V4 common positive action is illegal")
    _require(bool(torch.all(candidate_positive <= legal[:, None, :]).item()), "SetFlow V4 candidate positive action is illegal")
    _require(bool(torch.allclose(prior.sum(dim=1), torch.ones_like(prior[:, 0]), atol=1e-6)), "SetFlow V4 mode prior is not normalized")
    _require(bool((prior >= 0.5 / rates.shape[1] - 1e-7).all().item()), "SetFlow V4 smoothed mode-prior floor changed")
    normalizer = rates.sum(dim=2, keepdim=True)
    mode_probabilities = rates / normalizer.clamp_min(1e-20)
    active = ~structural
    _require(bool(active.any().item()), "SetFlow V4 batch contains only structural states")
    common_mode_mass = (
        mode_probabilities
        * common_positive[:, None, :].to(mode_probabilities.dtype)
    ).sum(dim=2)
    common_mixture_mass = (common_mode_mass * prior).sum(dim=1)
    common_loss_rows = -torch.log(common_mixture_mass[active].clamp_min(1e-20))
    common_loss = common_loss_rows.mean()
    candidate_mode_mass = torch.einsum(
        "bma,bca->bcm",
        mode_probabilities,
        candidate_positive.to(mode_probabilities.dtype),
    )
    candidate_mixture_mass = (
        candidate_mode_mass * prior[:, None, :]
    ).sum(dim=2)
    active_candidates = candidate_valid & active[:, None]
    _require(bool(active_candidates.any(dim=1)[active].all().item()), "SetFlow V4 active state has no compatible candidate constraint")
    per_candidate_nll = -torch.log(candidate_mixture_mass.clamp_min(1e-20))
    per_source_coverage = (
        per_candidate_nll * active_candidates
    ).sum(dim=1) / active_candidates.sum(dim=1).clamp_min(1)
    coverage_loss = per_source_coverage[active].mean()
    count_target = batch["remaining_count_soft_target"].float()
    _require(count_target.shape == output["remaining_count_logits"].shape, "SetFlow V4 count target geometry changed")
    _require(bool(torch.allclose(count_target.sum(dim=1), torch.ones(count_target.shape[0], device=count_target.device), atol=1e-6)), "SetFlow V4 count target is not normalized")
    count_loss = -(
        count_target
        * torch.log_softmax(output["remaining_count_logits"].float(), dim=1)
    ).sum(dim=1).mean()
    mode_losses: list[torch.Tensor] = []
    uniform = 1.0 / rates.shape[1]
    for row in range(rates.shape[0]):
        valid_candidates = active_candidates[row]
        if not bool(valid_candidates.any().item()):
            continue
        joint = (
            candidate_mode_mass[row, valid_candidates]
            * prior[row][None, :]
        )
        posterior = joint / joint.sum(dim=1, keepdim=True).clamp_min(1e-20)
        aggregate = posterior.mean(dim=0)
        mutual_information = (
            posterior
            * (
                torch.log(posterior.clamp_min(1e-20))
                - torch.log(aggregate.clamp_min(1e-20))[None, :]
            )
        ).sum(dim=1).mean()
        aggregate_balance = (
            aggregate
            * (
                torch.log(aggregate.clamp_min(1e-20))
                - torch.log(aggregate.new_tensor(uniform))
            )
        ).sum()
        mode_losses.append(-mutual_information + aggregate_balance)
    mode_information = (
        rates.new_zeros(())
        if not mode_losses or rates.shape[1] == 1
        else torch.stack(mode_losses).mean()
    )
    total = (
        common_loss
        + float(coverage_weight) * coverage_loss
        + float(remaining_count_weight) * count_loss
        + float(mode_information_weight) * mode_information
    )
    _require(torch.isfinite(total).item(), "SetFlow V4 total loss is nonfinite")
    return MixtureSetFlowLossV4(
        total=total,
        common_set_marginal=common_loss,
        source_candidate_coverage=coverage_loss,
        remaining_count=count_loss,
        mode_information=mode_information,
        active_state_count=int(active.sum().item()),
        active_candidate_constraint_count=int(active_candidates.sum().item()),
    )


def common_set_marginal_loss_v4(
    output: Mapping[str, torch.Tensor],
    positive_action_mask: torch.Tensor,
    structural_budget_exhausted: torch.Tensor,
    sample_weight: torch.Tensor,
) -> CommonSetMarginalLossV4:
    """V3-comparable common-state NLL under the V4 latent-mode mixture."""

    rates = output["mode_rates"].float()
    legal = output["legal_action_mask"]
    prior = output["mode_prior"].float()
    _require(
        rates.shape[0] == positive_action_mask.shape[0] == legal.shape[0],
        "SetFlow V4 common NLL batch geometry differs",
    )
    _require(
        rates.shape[2] == positive_action_mask.shape[1] == legal.shape[1],
        "SetFlow V4 common NLL action geometry differs",
    )
    _require(
        structural_budget_exhausted.shape == sample_weight.shape == (rates.shape[0],),
        "SetFlow V4 common NLL state weights differ",
    )
    _require(
        bool(torch.all(positive_action_mask <= legal).item()),
        "SetFlow V4 common NLL positive action is illegal",
    )
    active = ~structural_budget_exhausted
    _require(
        bool(torch.all(positive_action_mask.any(dim=1) == active).item()),
        "SetFlow V4 common NLL target presence differs from structural state",
    )
    probabilities = rates / rates.sum(dim=2, keepdim=True).clamp_min(1e-20)
    per_mode_mass = (
        probabilities
        * positive_action_mask[:, None, :].to(probabilities.dtype)
    ).sum(dim=2)
    mixture_mass = (per_mode_mass * prior).sum(dim=1)
    active_weight = sample_weight.float()[active].sum()
    _require(
        torch.isfinite(active_weight).item() and float(active_weight.item()) > 0.0,
        "SetFlow V4 common NLL has no active weight",
    )
    loss = (
        -torch.log(mixture_mass[active].clamp_min(1e-20))
        * sample_weight.float()[active]
    ).sum() / active_weight
    _require(torch.isfinite(loss).item(), "SetFlow V4 common NLL is nonfinite")
    return CommonSetMarginalLossV4(
        loss=loss,
        active_weight=active_weight,
        active_state_count=int(active.sum().item()),
    )


def require_setflow_v4_trainable_parameter_range(
    model: XEditSetFlowV4,
    *,
    minimum: int = 80_000_000,
    maximum: int = 150_000_000,
    design_target_minimum: int = 95_000_000,
    design_target_maximum: int = 110_000_000,
) -> dict[str, Any]:
    count = model.trainable_parameter_count
    _require(minimum <= count <= maximum, "SetFlow V4 trainable count is outside 80–150M")
    _require(design_target_minimum <= count <= design_target_maximum, "SetFlow V4 missed the 95–110M design target")
    return {
        "trainable_parameter_count": count,
        "minimum": minimum,
        "maximum": maximum,
        "design_target_minimum": design_target_minimum,
        "design_target_maximum": design_target_maximum,
        "module_counts": model.parameter_counts_by_module(),
        "passed": True,
    }
