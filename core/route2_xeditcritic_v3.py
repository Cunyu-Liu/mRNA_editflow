"""XEditCritic V3 endpoint-aware antisymmetric effect predictor."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from core.route2_delta_predictor import ResidualConvBlock, normalized_position_channels


XEDITCRITIC_V3_ARMS = {"C0", "C1", "C2", "C3"}
XEDITCRITIC_V3_CONTROLS = {
    "NONE",
    "SOURCE_ONLY",
    "EDIT_METADATA_ONLY",
    "NO_CANDIDATE_SEQUENCE",
}


class XEditCriticV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticV3Error(message)


class EndpointConditionerV1(nn.Module):
    """Outcome-free endpoint/assay/context representation shared by all arms."""

    def __init__(
        self,
        *,
        quantity_count: int,
        measurement_count: int,
        numerator_count: int,
        denominator_count: int,
        assay_count: int,
        context_count: int,
        region_count: int,
        output_width: int,
        category_width: int = 32,
    ) -> None:
        super().__init__()
        counts = {
            "quantity": quantity_count,
            "measurement": measurement_count,
            "numerator": numerator_count,
            "denominator": denominator_count,
            "assay": assay_count,
            "context": context_count,
            "region": region_count,
        }
        _require(min(counts.values()) > 0, "endpoint categorical vocabulary is empty")
        self.embeddings = nn.ModuleDict(
            {name: nn.Embedding(count, category_width) for name, count in counts.items()}
        )
        self.projection = nn.Sequential(
            nn.Linear(category_width * len(counts), output_width),
            nn.GELU(),
            nn.LayerNorm(output_width),
        )

    def forward(self, category_ids: Mapping[str, torch.Tensor]) -> torch.Tensor:
        _require(set(category_ids) == set(self.embeddings), "endpoint category bundle is incomplete")
        return self.projection(
            torch.cat(
                [self.embeddings[name](category_ids[name]) for name in self.embeddings],
                dim=-1,
            )
        )


class LowRankCategoricalFiLM(nn.Module):
    """Low-rank region or endpoint-family adapter."""

    def __init__(self, counts: Mapping[str, int], width: int, rank: int = 8) -> None:
        super().__init__()
        _require(min(counts.values()) > 0 and rank > 0, "low-rank adapter geometry is invalid")
        self.codes = nn.ModuleDict(
            {name: nn.Embedding(count, rank) for name, count in counts.items()}
        )
        self.scale = nn.Linear(rank, width, bias=False)
        self.shift = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)

    def forward(self, values: torch.Tensor, category_ids: Mapping[str, torch.Tensor]) -> torch.Tensor:
        _require(set(category_ids) == set(self.codes), "adapter category bundle is incomplete")
        code = sum(self.codes[name](category_ids[name]) for name in self.codes)
        return values * (1.0 + self.scale(code)) + self.shift(code)


class StudyLogScaleCalibrationV3(nn.Module):
    """Multiplicative nuisance calibration with immutable unknown-study scale 1."""

    def __init__(self, study_count_including_unknown: int) -> None:
        super().__init__()
        _require(study_count_including_unknown >= 1, "study vocabulary lacks unknown")
        self.known_log_scale = nn.Parameter(
            torch.zeros(study_count_including_unknown - 1)
        )

    def scale(self, study_ids: torch.Tensor) -> torch.Tensor:
        _require(study_ids.dtype == torch.long, "study ids must be integer")
        _require(bool(torch.all(study_ids >= 0).item()), "study id is negative")
        _require(
            bool(torch.all(study_ids <= self.known_log_scale.numel()).item()),
            "study id is outside the frozen vocabulary",
        )
        padded = torch.cat(
            [self.known_log_scale.new_zeros(1), self.known_log_scale], dim=0
        )
        return torch.exp(padded[study_ids])

    def forward(self, prediction: torch.Tensor, study_ids: torch.Tensor) -> torch.Tensor:
        return prediction * self.scale(study_ids)


class RawAntisymmetricBranchV3(nn.Module):
    """Historical hidden-65/depth-2 full-context raw sequence branch."""

    def __init__(self, *, hidden_dim: int = 65, depth: int = 2, region_count: int = 2, condition_width: int = 128) -> None:
        super().__init__()
        _require(hidden_dim >= 16 and depth >= 1, "raw branch geometry is invalid")
        self.nucleotide = nn.Embedding(5, hidden_dim, padding_idx=4)
        self.input_projection = nn.Linear(hidden_dim * 3 + 3, hidden_dim)
        self.blocks = nn.ModuleList(
            ResidualConvBlock(hidden_dim, dilation=2**index) for index in range(depth)
        )
        self.region_adapter = LowRankCategoricalFiLM(
            {"region": region_count}, hidden_dim, rank=8
        )
        self.condition_projection = nn.Linear(condition_width, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.output_width = hidden_dim

    def forward(
        self,
        left_tokens: torch.Tensor,
        right_tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        region_ids: torch.Tensor,
        endpoint_condition: torch.Tensor,
    ) -> torch.Tensor:
        valid = ~padding_mask
        left = self.nucleotide(left_tokens)
        right = self.nucleotide(right_tokens)
        edited = ((left_tokens != right_tokens) & valid).to(left.dtype).unsqueeze(-1)
        position, edited_position = normalized_position_channels(padding_mask, edited)
        hidden = self.input_projection(
            torch.cat((left, right, right - left, edited, position, edited_position), dim=-1)
        ) * valid.unsqueeze(-1)
        for block in self.blocks:
            hidden = block(hidden, padding_mask)
        hidden = self.region_adapter(
            hidden,
            {"region": region_ids.unsqueeze(1).expand(-1, hidden.shape[1])},
        ) * valid.unsqueeze(-1)
        weights = valid.to(hidden.dtype).unsqueeze(-1)
        mean_pool = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        maximum_pool = hidden.masked_fill(padding_mask.unsqueeze(-1), -torch.inf).amax(dim=1)
        return self.fusion(
            torch.cat((mean_pool, maximum_pool, self.condition_projection(endpoint_condition)), dim=-1)
        )


class EditSiteTokenBranchV3(nn.Module):
    """Eight-layer edit-token transformer with separate global residual fusion."""

    def __init__(
        self,
        *,
        pretrained_width: int = 768,
        model_width: int = 512,
        depth: int = 8,
        heads: int = 8,
        ffn_width: int = 2048,
        dropout: float = 0.10,
        base_embedding_width: int = 32,
        condition_width: int = 128,
        region_count: int = 2,
        quantity_count: int,
        measurement_count: int,
    ) -> None:
        super().__init__()
        _require(model_width % heads == 0, "edit transformer width is not divisible by heads")
        _require(depth >= 1 and pretrained_width >= 1, "edit transformer geometry is invalid")
        # site delta/mean, local-mean delta/mean, and local-max delta/mean.
        self.feature_projection = nn.Sequential(
            nn.Linear(pretrained_width * 6, model_width // 2),
            nn.GELU(),
            nn.Linear(model_width // 2, model_width),
            nn.LayerNorm(model_width),
        )
        self.base_embedding = nn.Embedding(5, base_embedding_width, padding_idx=4)
        self.metadata_projection = nn.Linear(base_embedding_width * 2 + 1, model_width)
        layer = nn.TransformerEncoderLayer(
            d_model=model_width,
            nhead=heads,
            dim_feedforward=ffn_width,
            # Independent directional dropout masks would break exact
            # antisymmetry.  The parent applies one shared mask after both
            # deterministic directional representations have been computed.
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.configured_dropout = float(dropout)
        self.endpoint_adapter = LowRankCategoricalFiLM(
            {
                "region": region_count,
                "quantity": quantity_count,
                "measurement": measurement_count,
            },
            model_width,
            rank=8,
        )
        self.attention_pool = nn.Linear(model_width, 1)
        self.global_delta = nn.Linear(pretrained_width, model_width)
        self.global_background = nn.Linear(pretrained_width, model_width)
        self.condition_projection = nn.Linear(condition_width, model_width)
        self.fusion = nn.Sequential(
            nn.Linear(model_width * 5, model_width),
            nn.GELU(),
            nn.LayerNorm(model_width),
        )
        self.output_width = model_width

    def forward(
        self,
        *,
        left_site: torch.Tensor,
        right_site: torch.Tensor,
        left_window_mean: torch.Tensor,
        right_window_mean: torch.Tensor,
        left_window_max: torch.Tensor,
        right_window_max: torch.Tensor,
        left_global: torch.Tensor,
        right_global: torch.Tensor,
        left_base_ids: torch.Tensor,
        right_base_ids: torch.Tensor,
        normalized_edit_positions: torch.Tensor,
        edit_padding_mask: torch.Tensor,
        region_ids: torch.Tensor,
        quantity_ids: torch.Tensor,
        measurement_ids: torch.Tensor,
        endpoint_condition: torch.Tensor,
    ) -> torch.Tensor:
        _require(edit_padding_mask.ndim == 2, "edit mask must be batch x edits")
        features = torch.cat(
            (
                right_site - left_site,
                0.5 * (right_site + left_site),
                right_window_mean - left_window_mean,
                0.5 * (right_window_mean + left_window_mean),
                right_window_max - left_window_max,
                0.5 * (right_window_max + left_window_max),
            ),
            dim=-1,
        )
        metadata = torch.cat(
            (
                self.base_embedding(left_base_ids),
                self.base_embedding(right_base_ids),
                normalized_edit_positions.unsqueeze(-1),
            ),
            dim=-1,
        )
        hidden = self.feature_projection(features) + self.metadata_projection(metadata)
        valid = ~edit_padding_mask
        has_edit = valid.any(dim=1)
        safe_padding = edit_padding_mask.clone()
        safe_padding[~has_edit, 0] = False
        hidden = hidden.masked_fill(edit_padding_mask.unsqueeze(-1), 0.0)
        hidden = self.transformer(hidden, src_key_padding_mask=safe_padding)
        hidden = self.endpoint_adapter(
            hidden,
            {
                "region": region_ids.unsqueeze(1).expand_as(edit_padding_mask),
                "quantity": quantity_ids.unsqueeze(1).expand_as(edit_padding_mask),
                "measurement": measurement_ids.unsqueeze(1).expand_as(edit_padding_mask),
            },
        )
        hidden = hidden * valid.unsqueeze(-1)
        logits = self.attention_pool(hidden).squeeze(-1).masked_fill(~valid, -torch.inf)
        safe_logits = torch.where(has_edit.unsqueeze(1), logits, torch.zeros_like(logits))
        attention = torch.softmax(safe_logits, dim=1) * valid.to(hidden.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1.0)
        attention_value = (hidden * attention.unsqueeze(-1)).sum(dim=1)
        maximum = hidden.masked_fill(~valid.unsqueeze(-1), -torch.inf).amax(dim=1)
        maximum = torch.where(has_edit.unsqueeze(-1), maximum, torch.zeros_like(maximum))
        representation = self.fusion(
            torch.cat(
                (
                    attention_value,
                    maximum,
                    self.global_delta(right_global - left_global),
                    self.global_background(0.5 * (right_global + left_global)),
                    self.condition_projection(endpoint_condition),
                ),
                dim=-1,
            )
        )
        return representation


class XEditCriticV3(nn.Module):
    """C0/C1/C2/C3 critic family with exact primary-model antisymmetry."""

    def __init__(
        self,
        *,
        arm: str,
        control_mode: str = "NONE",
        study_count: int,
        assay_count: int,
        context_count: int,
        quantity_count: int,
        measurement_count: int,
        numerator_count: int,
        denominator_count: int,
        region_count: int = 2,
        pretrained_width: int = 768,
        condition_width: int = 128,
        raw_hidden_dim: int = 65,
        raw_depth: int = 2,
        model_width: int = 512,
        transformer_depth: int = 8,
        transformer_heads: int = 8,
        transformer_ffn_width: int = 2048,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        _require(arm in XEDITCRITIC_V3_ARMS, "unknown Critic V3 arm")
        _require(control_mode in XEDITCRITIC_V3_CONTROLS, "unknown Critic V3 control")
        if arm in {"C0", "C1"}:
            _require(control_mode == "NONE", "diagnostic arms do not accept full-model controls")
        self.arm = arm
        self.control_mode = control_mode
        self.paired_dropout_probability = float(dropout)
        self.endpoint_conditioner = EndpointConditionerV1(
            quantity_count=quantity_count,
            measurement_count=measurement_count,
            numerator_count=numerator_count,
            denominator_count=denominator_count,
            assay_count=assay_count,
            context_count=context_count,
            region_count=region_count,
            output_width=condition_width,
        )
        self.raw_branch = RawAntisymmetricBranchV3(
            hidden_dim=raw_hidden_dim,
            depth=raw_depth,
            region_count=region_count,
            condition_width=condition_width,
        )
        self.raw_projection = nn.Linear(raw_hidden_dim, model_width)
        self.global_delta = (
            nn.Linear(pretrained_width, model_width) if arm == "C1" else None
        )
        self.global_background = (
            nn.Linear(pretrained_width, model_width) if arm == "C1" else None
        )
        self.token_branch = (
            EditSiteTokenBranchV3(
                pretrained_width=pretrained_width,
                model_width=model_width,
                depth=transformer_depth,
                heads=transformer_heads,
                ffn_width=transformer_ffn_width,
                dropout=dropout,
                condition_width=condition_width,
                region_count=region_count,
                quantity_count=quantity_count,
                measurement_count=measurement_count,
            )
            if arm in {"C2", "C3"}
            else None
        )
        branch_count = 1 if arm == "C0" else 3 if arm == "C1" else 2
        self.effect_fusion = nn.Sequential(
            nn.Linear(model_width * branch_count + condition_width, model_width),
            nn.GELU(),
            nn.LayerNorm(model_width),
        )
        self.effect_head = nn.Linear(model_width, 1)
        self.study_calibration = StudyLogScaleCalibrationV3(study_count)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _directed_representation(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        reverse: bool,
        replace_candidate_sequence: bool = False,
        suppress_edits: bool = False,
    ) -> torch.Tensor:
        left_prefix, right_prefix = ("candidate", "source") if reverse else ("source", "candidate")
        if replace_candidate_sequence:
            left_prefix = "source"
            right_prefix = "source"
        endpoint_categories = {
            "quantity": batch["quantity_ids"],
            "measurement": batch["measurement_ids"],
            "numerator": batch["numerator_ids"],
            "denominator": batch["denominator_ids"],
            "assay": batch["assay_ids"],
            "context": batch["context_ids"],
            "region": batch["region_ids"],
        }
        condition = self.endpoint_conditioner(endpoint_categories)
        raw = self.raw_projection(
            self.raw_branch(
                batch[f"{left_prefix}_tokens"],
                batch[f"{right_prefix}_tokens"],
                batch["padding_mask"],
                batch["region_ids"],
                condition,
            )
        )
        branches = [raw]
        if self.arm == "C1":
            _require(
                self.global_delta is not None and self.global_background is not None,
                "global diagnostic branch is absent",
            )
            left_global = batch[f"{left_prefix}_global"]
            right_global = batch[f"{right_prefix}_global"]
            branches.extend(
                (
                    self.global_delta(right_global - left_global),
                    self.global_background(0.5 * (right_global + left_global)),
                )
            )
        elif self.arm in {"C2", "C3"}:
            _require(self.token_branch is not None, "token branch is absent")
            edit_padding_mask = (
                torch.ones_like(batch["edit_padding_mask"])
                if suppress_edits
                else batch["edit_padding_mask"]
            )
            branches.append(
                self.token_branch(
                    left_site=batch[f"{left_prefix}_site"],
                    right_site=batch[f"{right_prefix}_site"],
                    left_window_mean=batch[f"{left_prefix}_window_mean"],
                    right_window_mean=batch[f"{right_prefix}_window_mean"],
                    left_window_max=batch[f"{left_prefix}_window_max"],
                    right_window_max=batch[f"{right_prefix}_window_max"],
                    left_global=batch[f"{left_prefix}_global"],
                    right_global=batch[f"{right_prefix}_global"],
                    left_base_ids=batch[f"{left_prefix}_edit_base_ids"],
                    right_base_ids=batch[f"{right_prefix}_edit_base_ids"],
                    normalized_edit_positions=batch["normalized_edit_positions"],
                    edit_padding_mask=edit_padding_mask,
                    region_ids=batch["region_ids"],
                    quantity_ids=batch["quantity_ids"],
                    measurement_ids=batch["measurement_ids"],
                    endpoint_condition=condition,
                )
            )
        return self.effect_fusion(torch.cat((*branches, condition), dim=-1))

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.control_mode == "SOURCE_ONLY":
            representation = self._directed_representation(
                batch, reverse=False, replace_candidate_sequence=True, suppress_edits=True
            )
            representation = F.dropout(
                representation,
                p=self.paired_dropout_probability,
                training=self.training,
            )
            mean = self.effect_head(representation).squeeze(-1)
        elif self.control_mode == "NO_CANDIDATE_SEQUENCE":
            representation = self._directed_representation(
                batch, reverse=False, replace_candidate_sequence=True
            )
            representation = F.dropout(
                representation,
                p=self.paired_dropout_probability,
                training=self.training,
            )
            mean = self.effect_head(representation).squeeze(-1)
        elif self.control_mode == "EDIT_METADATA_ONLY":
            metadata_batch = dict(batch)
            for suffix in ("site", "window_mean", "window_max", "global"):
                metadata_batch[f"candidate_{suffix}"] = batch[f"source_{suffix}"]
            representation = self._directed_representation(
                metadata_batch, reverse=False
            )
            identity = self._directed_representation(
                metadata_batch, reverse=False, replace_candidate_sequence=True, suppress_edits=True
            )
            shared_mask = F.dropout(
                torch.ones_like(representation),
                p=self.paired_dropout_probability,
                training=self.training,
            )
            representation = representation * shared_mask
            identity = identity * shared_mask
            mean = (self.effect_head(representation) - self.effect_head(identity)).squeeze(-1)
        else:
            forward = self._directed_representation(batch, reverse=False)
            reverse = self._directed_representation(batch, reverse=True)
            shared_mask = F.dropout(
                torch.ones_like(forward),
                p=self.paired_dropout_probability,
                training=self.training,
            )
            forward = forward * shared_mask
            reverse = reverse * shared_mask
            mean = 0.5 * (self.effect_head(forward) - self.effect_head(reverse)).squeeze(-1)
        mean = self.study_calibration(mean, batch["study_ids"])
        return {"mean": mean, "log_variance": torch.zeros_like(mean)}
