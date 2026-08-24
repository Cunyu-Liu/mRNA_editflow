"""Large edit-local, endpoint-semantic, antisymmetric XEditCritic V4."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from core.route2_xeditcritic_v3 import (
    EndpointConditionerV1,
    RawAntisymmetricBranchV3,
    StudyLogScaleCalibrationV3,
)


XEDITCRITIC_V4_CONTROLS = {
    "NONE",
    "SOURCE_ONLY",
    "EDIT_METADATA_ONLY",
    "NO_CANDIDATE_SEQUENCE",
}
XEDITCRITIC_V4_MECHANISMS = {"FULL", "NO_CROSS", "NO_MOE"}


class XEditCriticV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticV4Error(message)


class PairedDropoutV4(nn.Module):
    """Use one dropout mask for source→candidate and candidate→source."""

    def __init__(self, probability: float) -> None:
        super().__init__()
        _require(0.0 <= probability < 1.0, "paired dropout probability is invalid")
        self.probability = float(probability)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _require(values.ndim >= 2 and values.shape[0] == 2, "paired dropout expects direction x batch x ...")
        if not self.training or self.probability == 0.0:
            return values
        keep_probability = 1.0 - self.probability
        mask = torch.empty(
            (1, *values.shape[1:]),
            dtype=values.dtype,
            device=values.device,
        ).bernoulli_(keep_probability)
        return values * mask / keep_probability


class TrainableUpperSixTransformerV4(nn.Module):
    """Geometry-compatible upper-six proxy; formal training loads pretrained blocks 6–11."""

    def __init__(
        self,
        *,
        width: int = 768,
        heads: int = 12,
        ffn_width: int = 3072,
        depth: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        _require(depth >= 1 and width % heads == 0, "upper-six transformer geometry is invalid")
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=ffn_width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        _require(hidden.ndim == 3 and attention_mask.shape == hidden.shape[:2], "upper-six input geometry changed")
        return self.layers(hidden, src_key_padding_mask=~attention_mask.to(torch.bool))


class EndpointSemanticRouterV4(nn.Module):
    """Top-two outcome-free semantic routing shared across V4 blocks."""

    def __init__(self, width: int, expert_count: int = 4, top_k: int = 2) -> None:
        super().__init__()
        _require(1 <= top_k <= expert_count, "semantic routing top-k is invalid")
        self.router = nn.Linear(width, expert_count)
        self.expert_count = int(expert_count)
        self.top_k = int(top_k)

    def forward(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.router(condition)
        top_values, top_indices = torch.topk(logits, self.top_k, dim=-1)
        selected = torch.softmax(top_values, dim=-1)
        weights = torch.zeros_like(logits).scatter(-1, top_indices, selected)
        aggregate = weights.mean(dim=0)
        balance = self.expert_count * torch.square(
            aggregate - 1.0 / self.expert_count
        ).sum()
        return weights, balance


class SemanticResidualExpertsV4(nn.Module):
    """Four bottleneck semantic experts or a parameter-matched generic adapter."""

    def __init__(
        self,
        *,
        width: int,
        bottleneck_width: int,
        expert_count: int,
        semantic_routing: bool,
    ) -> None:
        super().__init__()
        self.semantic_routing = bool(semantic_routing)
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(width, bottleneck_width),
                nn.GELU(),
                nn.Linear(bottleneck_width, width),
            )
            for _ in range(expert_count)
        )

    def forward(
        self, values: torch.Tensor, route_weights: torch.Tensor
    ) -> torch.Tensor:
        _require(values.ndim == 4 and values.shape[0] == 2, "semantic experts expect direction x batch x edits x width")
        _require(route_weights.shape == (values.shape[1], len(self.experts)), "semantic route geometry changed")
        outputs = torch.stack([expert(values) for expert in self.experts], dim=-2)
        if self.semantic_routing:
            weights = route_weights[None, :, None, :, None]
        else:
            weights = values.new_full(
                (1, values.shape[1], 1, len(self.experts), 1),
                1.0 / len(self.experts),
            )
        return (outputs * weights).sum(dim=-2)


class _SemanticBlockBaseV4(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        ffn_width: int,
        dropout: float,
        expert_count: int,
        expert_bottleneck_width: int,
        semantic_routing: bool,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.ffn_norm = nn.LayerNorm(width)
        self.expert_norm = nn.LayerNorm(width)
        self.shared_ffn = nn.Sequential(
            nn.Linear(width, ffn_width),
            nn.GELU(),
            nn.Linear(ffn_width, width),
        )
        self.experts = SemanticResidualExpertsV4(
            width=width,
            bottleneck_width=expert_bottleneck_width,
            expert_count=expert_count,
            semantic_routing=semantic_routing,
        )
        self.attention_dropout = PairedDropoutV4(dropout)
        self.ffn_dropout = PairedDropoutV4(dropout)
        self.expert_dropout = PairedDropoutV4(dropout)

    def _ffn_and_experts(
        self,
        values: torch.Tensor,
        *,
        edit_padding_mask: torch.Tensor,
        route_weights: torch.Tensor,
    ) -> torch.Tensor:
        valid = (~edit_padding_mask)[None, :, :, None]
        values = values + self.ffn_dropout(self.shared_ffn(self.ffn_norm(values)))
        values = values + self.expert_dropout(
            self.experts(self.expert_norm(values), route_weights)
        )
        return values * valid


class EditSetSelfAttentionBlockV4(_SemanticBlockBaseV4):
    def __init__(self, *, width: int, heads: int, **kwargs: Any) -> None:
        super().__init__(width=width, **kwargs)
        self.attention = nn.MultiheadAttention(
            width,
            heads,
            dropout=0.0,
            batch_first=True,
        )

    def forward(
        self,
        values: torch.Tensor,
        *,
        edit_padding_mask: torch.Tensor,
        route_weights: torch.Tensor,
    ) -> torch.Tensor:
        directions, batch_size, edit_count, width = values.shape
        _require(directions == 2, "edit self-attention lost its paired directions")
        safe_padding = edit_padding_mask.clone()
        no_edits = safe_padding.all(dim=1)
        safe_padding[no_edits, 0] = False
        normalized = self.attention_norm(values).reshape(directions * batch_size, edit_count, width)
        repeated_padding = safe_padding.unsqueeze(0).expand(directions, -1, -1).reshape(directions * batch_size, edit_count)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=repeated_padding,
            need_weights=False,
        )
        attended = attended.reshape(directions, batch_size, edit_count, width)
        values = values + self.attention_dropout(attended)
        values = values * (~edit_padding_mask)[None, :, :, None]
        return self._ffn_and_experts(
            values,
            edit_padding_mask=edit_padding_mask,
            route_weights=route_weights,
        )


class LocalCrossAttentionBlockV4(_SemanticBlockBaseV4):
    def __init__(self, *, width: int, heads: int, **kwargs: Any) -> None:
        super().__init__(width=width, **kwargs)
        self.attention = nn.MultiheadAttention(
            width,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.context_fusion = nn.Linear(width * 3, width)

    def forward(
        self,
        values: torch.Tensor,
        *,
        left_context: torch.Tensor,
        right_context: torch.Tensor,
        local_context_mask: torch.Tensor,
        edit_padding_mask: torch.Tensor,
        route_weights: torch.Tensor,
    ) -> torch.Tensor:
        directions, batch_size, edit_count, width = values.shape
        local_length = left_context.shape[-2]
        _require(left_context.shape == right_context.shape == (2, batch_size, edit_count, local_length, width), "local cross-attention context geometry changed")
        _require(local_context_mask.shape == (batch_size, edit_count, local_length), "local context mask geometry changed")
        query = self.attention_norm(values).reshape(directions * batch_size * edit_count, 1, width)
        left = left_context.reshape(directions * batch_size * edit_count, local_length, width)
        right = right_context.reshape(directions * batch_size * edit_count, local_length, width)
        context_padding = (~local_context_mask).unsqueeze(0).expand(directions, -1, -1, -1).reshape(directions * batch_size * edit_count, local_length)
        safe_padding = context_padding.clone()
        safe_padding[safe_padding.all(dim=1), 0] = False
        paired_query = torch.cat((query, query), dim=0)
        paired_context = torch.cat((left, right), dim=0)
        paired_padding = torch.cat((safe_padding, safe_padding), dim=0)
        attended, _ = self.attention(
            paired_query,
            paired_context,
            paired_context,
            key_padding_mask=paired_padding,
            need_weights=False,
        )
        left_value, right_value = attended.chunk(2, dim=0)
        left_value = left_value.squeeze(1).reshape(directions, batch_size, edit_count, width)
        right_value = right_value.squeeze(1).reshape(directions, batch_size, edit_count, width)
        fused = self.context_fusion(
            torch.cat(
                (
                    values,
                    right_value - left_value,
                    0.5 * (right_value + left_value),
                ),
                dim=-1,
            )
        )
        values = values + self.attention_dropout(F.gelu(fused))
        values = values * (~edit_padding_mask)[None, :, :, None]
        return self._ffn_and_experts(
            values,
            edit_padding_mask=edit_padding_mask,
            route_weights=route_weights,
        )


class LocalPooledResidualBlockV4(_SemanticBlockBaseV4):
    """NO-CROSS replacement with the exact MHA+fusion parameter count."""

    def __init__(self, *, width: int, heads: int, **kwargs: Any) -> None:
        del heads
        super().__init__(width=width, **kwargs)
        # Four width→width projections exactly match MultiheadAttention's Q/K/V/O
        # weights and biases; every projection participates in the pooled mixer.
        self.pooled_projections = nn.ModuleList(
            nn.Linear(width, width) for _ in range(4)
        )
        self.context_fusion = nn.Linear(width * 3, width)

    def forward(
        self,
        values: torch.Tensor,
        *,
        left_context: torch.Tensor,
        right_context: torch.Tensor,
        local_context_mask: torch.Tensor,
        edit_padding_mask: torch.Tensor,
        route_weights: torch.Tensor,
    ) -> torch.Tensor:
        weights = local_context_mask.to(values.dtype)[None, :, :, :, None]
        denominator = weights.sum(dim=-2).clamp_min(1.0)
        left = (left_context * weights).sum(dim=-2) / denominator
        right = (right_context * weights).sum(dim=-2) / denominator
        normalized = self.attention_norm(values)
        mixed = (
            self.pooled_projections[0](normalized)
            + self.pooled_projections[1](left)
            + self.pooled_projections[2](right)
            + self.pooled_projections[3](right - left)
        )
        fused = self.context_fusion(
            torch.cat((values, mixed, 0.5 * (right + left)), dim=-1)
        )
        values = values + self.attention_dropout(F.gelu(fused))
        values = values * (~edit_padding_mask)[None, :, :, None]
        return self._ffn_and_experts(
            values,
            edit_padding_mask=edit_padding_mask,
            route_weights=route_weights,
        )


def gather_ragged_local_contexts_v4(
    *,
    upper_chunk_hidden: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    edit_padding_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Gather only each edit's predeclared radius-32 nucleotide token window."""

    _require(upper_chunk_hidden.ndim == 3, "upper-six chunk hidden must be chunk x token x width")
    batch_size, maximum_edits = edit_padding_mask.shape
    width = upper_chunk_hidden.shape[-1]
    local_length = 65
    result: dict[str, torch.Tensor] = {}
    offsets = batch["record_edit_offsets"]
    _require(tuple(offsets.shape) == (batch_size + 1,), "record edit offsets differ from physical batch")
    flattened_edit_count = int(offsets[-1].item())
    _require(flattened_edit_count == int((~edit_padding_mask).sum().item()), "padded edit mask differs from ragged cache count")
    for side in ("source", "candidate"):
        contexts = upper_chunk_hidden.new_zeros((batch_size, maximum_edits, local_length, width))
        context_mask = torch.zeros((batch_size, maximum_edits, local_length), dtype=torch.bool, device=upper_chunk_hidden.device)
        sites = upper_chunk_hidden.new_zeros((batch_size, maximum_edits, width))
        chunk_indices = batch[f"edit_{side}_chunk_indices"]
        centers = batch[f"edit_{side}_token_centers"]
        starts = batch[f"edit_{side}_window_starts"]
        ends = batch[f"edit_{side}_window_ends"]
        _require(all(int(values.numel()) == flattened_edit_count for values in (chunk_indices, centers, starts, ends)), f"{side} flattened edit mapping changed")
        for record_index in range(batch_size):
            flat_start = int(offsets[record_index].item())
            flat_end = int(offsets[record_index + 1].item())
            _require(flat_end - flat_start <= maximum_edits, "ragged record exceeds padded edit dimension")
            for local_edit_index, flat_index in enumerate(range(flat_start, flat_end)):
                chunk_index = int(chunk_indices[flat_index].item())
                center = int(centers[flat_index].item())
                start = int(starts[flat_index].item())
                end = int(ends[flat_index].item())
                _require(0 <= start <= center < end <= upper_chunk_hidden.shape[1], f"{side} local window is outside the materialized chunk")
                length = end - start
                _require(length <= local_length, f"{side} local window exceeds radius 32")
                contexts[record_index, local_edit_index, :length] = upper_chunk_hidden[chunk_index, start:end]
                context_mask[record_index, local_edit_index, :length] = True
                sites[record_index, local_edit_index] = upper_chunk_hidden[chunk_index, center]
        result[f"{side}_context"] = contexts
        result[f"{side}_site"] = sites
        result[f"{side}_context_mask"] = context_mask
    _require(torch.equal(result["source_context_mask"], result["candidate_context_mask"]), "source/candidate local window lengths differ")
    result["local_context_mask"] = result.pop("source_context_mask")
    result.pop("candidate_context_mask")
    return result


class XEditCriticV4(nn.Module):
    """Full, control, and parameter-matched mechanism V4 critic family."""

    def __init__(
        self,
        *,
        upper_encoder: nn.Module,
        study_count: int,
        assay_count: int,
        context_count: int,
        quantity_count: int,
        measurement_count: int,
        numerator_count: int,
        denominator_count: int,
        region_count: int = 2,
        control_mode: str = "NONE",
        mechanism_mode: str = "FULL",
        pretrained_width: int = 768,
        model_width: int = 768,
        block_count: int = 12,
        heads: int = 12,
        ffn_width: int = 3072,
        expert_count: int = 4,
        expert_bottleneck_width: int = 256,
        expert_top_k: int = 2,
        base_embedding_width: int = 32,
        raw_hidden_dim: int = 65,
        raw_depth: int = 2,
        readout_hidden_width: int = 2560,
        dropout: float = 0.1,
        minimum_physical_batch: int = 4,
    ) -> None:
        super().__init__()
        _require(control_mode in XEDITCRITIC_V4_CONTROLS, "unknown Critic V4 control")
        _require(mechanism_mode in XEDITCRITIC_V4_MECHANISMS, "unknown Critic V4 mechanism")
        _require(block_count >= 2 and block_count % 2 == 0, "V4 edit blocks must alternate evenly")
        _require(model_width == pretrained_width and model_width % heads == 0, "V4 hidden/attention width changed")
        _require(minimum_physical_batch >= 1, "minimum physical batch is invalid")
        self.upper_encoder = upper_encoder
        self.control_mode = control_mode
        self.mechanism_mode = mechanism_mode
        self.minimum_physical_batch = int(minimum_physical_batch)
        self.endpoint_conditioner = EndpointConditionerV1(
            quantity_count=quantity_count,
            measurement_count=measurement_count,
            numerator_count=numerator_count,
            denominator_count=denominator_count,
            assay_count=assay_count,
            context_count=context_count,
            region_count=region_count,
            output_width=model_width,
        )
        self.router = EndpointSemanticRouterV4(
            model_width,
            expert_count=expert_count,
            top_k=expert_top_k,
        )
        self.base_embedding = nn.Embedding(5, base_embedding_width, padding_idx=4)
        self.edit_input = nn.Sequential(
            nn.Linear(model_width * 2 + base_embedding_width * 2 + 1, model_width),
            nn.GELU(),
            nn.LayerNorm(model_width),
        )
        common_block = {
            "width": model_width,
            "heads": heads,
            "ffn_width": ffn_width,
            "dropout": dropout,
            "expert_count": expert_count,
            "expert_bottleneck_width": expert_bottleneck_width,
            "semantic_routing": mechanism_mode != "NO_MOE",
        }
        blocks: list[nn.Module] = []
        for block_index in range(block_count):
            if block_index % 2 == 0:
                blocks.append(EditSetSelfAttentionBlockV4(**common_block))
            elif mechanism_mode == "NO_CROSS":
                blocks.append(LocalPooledResidualBlockV4(**common_block))
            else:
                blocks.append(LocalCrossAttentionBlockV4(**common_block))
        self.blocks = nn.ModuleList(blocks)
        self.raw_branch = RawAntisymmetricBranchV3(
            hidden_dim=raw_hidden_dim,
            depth=raw_depth,
            region_count=region_count,
            condition_width=model_width,
        )
        self.raw_projection = nn.Linear(raw_hidden_dim, model_width)
        self.global_delta = nn.Linear(pretrained_width, model_width)
        self.global_mean = nn.Linear(pretrained_width, model_width)
        self.edit_attention_pool = nn.Linear(model_width, 1)
        # Six width-768 branches -> 2560 -> 768 is a preflight-counted readout,
        # not an additional unreported transformer trunk.
        self.readout = nn.Sequential(
            nn.Linear(model_width * 6, readout_hidden_width),
            nn.GELU(),
            nn.Linear(readout_hidden_width, model_width),
            nn.LayerNorm(model_width),
        )
        self.readout_dropout = PairedDropoutV4(dropout)
        self.effect_head = nn.Linear(model_width, 1)
        self.study_calibration = StudyLogScaleCalibrationV3(study_count)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def parameter_counts_by_module(self) -> dict[str, int]:
        modules = {
            "upper_six": self.upper_encoder,
            "endpoint_conditioner": self.endpoint_conditioner,
            "semantic_router": self.router,
            "edit_input_metadata_and_pool": nn.ModuleList(
                [self.base_embedding, self.edit_input, self.edit_attention_pool]
            ),
            "edit_blocks": self.blocks,
            "raw_branch": nn.ModuleList([self.raw_branch, self.raw_projection]),
            "global_residual": nn.ModuleList([self.global_delta, self.global_mean]),
            "readout_and_head": nn.ModuleList([self.readout, self.effect_head]),
            "study_scale": self.study_calibration,
        }
        counts = {
            name: sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
            for name, module in modules.items()
        }
        _require(sum(counts.values()) == self.trainable_parameter_count, "module parameter accounting is incomplete")
        return counts

    def _endpoint_condition(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
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

    def _paired_representations(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        replace_candidate_sequence: bool = False,
        suppress_edits: bool = False,
        replace_candidate_edit_base: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(batch["source_tokens"].shape[0])
        _require(batch_size >= self.minimum_physical_batch, "Critic V4 physical batch is below the frozen minimum")
        condition = self._endpoint_condition(batch)
        route_weights, router_balance = self.router(condition)
        upper_hidden = self.upper_encoder(
            batch["chunk_hidden"],
            batch["chunk_attention_mask"],
        )
        _require(upper_hidden.shape == batch["chunk_hidden"].shape, "upper-six output geometry changed")
        edit_padding_mask = batch["edit_padding_mask"].clone()
        if suppress_edits:
            edit_padding_mask[:] = True
        local = gather_ragged_local_contexts_v4(
            upper_chunk_hidden=upper_hidden,
            batch=batch,
            edit_padding_mask=batch["edit_padding_mask"],
        )
        source_context = local["source_context"]
        candidate_context = local["candidate_context"]
        source_site = local["source_site"]
        candidate_site = local["candidate_site"]
        source_global = batch["record_source_global"]
        candidate_global = batch["record_candidate_global"]
        candidate_tokens = batch["candidate_tokens"]
        candidate_bases = batch["candidate_edit_base_ids"]
        if replace_candidate_sequence:
            candidate_context = source_context
            candidate_site = source_site
            candidate_global = source_global
            candidate_tokens = batch["source_tokens"]
        if replace_candidate_edit_base:
            candidate_bases = batch["source_edit_base_ids"]
        forward_features = torch.cat(
            (
                candidate_site - source_site,
                0.5 * (candidate_site + source_site),
                self.base_embedding(batch["source_edit_base_ids"]),
                self.base_embedding(candidate_bases),
                batch["normalized_edit_positions"].unsqueeze(-1),
            ),
            dim=-1,
        )
        reverse_features = torch.cat(
            (
                source_site - candidate_site,
                0.5 * (candidate_site + source_site),
                self.base_embedding(candidate_bases),
                self.base_embedding(batch["source_edit_base_ids"]),
                batch["normalized_edit_positions"].unsqueeze(-1),
            ),
            dim=-1,
        )
        edit_values = torch.stack(
            (self.edit_input(forward_features), self.edit_input(reverse_features)),
            dim=0,
        )
        edit_values = edit_values * (~edit_padding_mask)[None, :, :, None]
        left_context = torch.stack((source_context, candidate_context), dim=0)
        right_context = torch.stack((candidate_context, source_context), dim=0)
        for block in self.blocks:
            if isinstance(block, EditSetSelfAttentionBlockV4):
                edit_values = block(
                    edit_values,
                    edit_padding_mask=edit_padding_mask,
                    route_weights=route_weights,
                )
            else:
                edit_values = block(
                    edit_values,
                    left_context=left_context,
                    right_context=right_context,
                    local_context_mask=local["local_context_mask"],
                    edit_padding_mask=edit_padding_mask,
                    route_weights=route_weights,
                )
        valid = ~edit_padding_mask
        has_edit = valid.any(dim=1)
        logits = self.edit_attention_pool(edit_values).squeeze(-1).masked_fill(~valid[None], -torch.inf)
        safe_logits = torch.where(has_edit[None, :, None], logits, torch.zeros_like(logits))
        attention = torch.softmax(safe_logits, dim=-1) * valid[None].to(edit_values.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1.0)
        edit_attention = (edit_values * attention.unsqueeze(-1)).sum(dim=2)
        edit_maximum = edit_values.masked_fill(~valid[None, :, :, None], -torch.inf).amax(dim=2)
        edit_maximum = torch.where(has_edit[None, :, None], edit_maximum, torch.zeros_like(edit_maximum))
        raw_forward = self.raw_projection(
            self.raw_branch(
                batch["source_tokens"],
                candidate_tokens,
                batch["padding_mask"],
                batch["region_ids"],
                condition,
            )
        )
        raw_reverse = self.raw_projection(
            self.raw_branch(
                candidate_tokens,
                batch["source_tokens"],
                batch["padding_mask"],
                batch["region_ids"],
                condition,
            )
        )
        raw = torch.stack((raw_forward, raw_reverse), dim=0)
        global_delta = torch.stack(
            (
                self.global_delta(candidate_global - source_global),
                self.global_delta(source_global - candidate_global),
            ),
            dim=0,
        )
        global_mean = self.global_mean(0.5 * (candidate_global + source_global))[None].expand(2, -1, -1)
        endpoint = condition[None].expand(2, -1, -1)
        representation = self.readout(
            torch.cat(
                (
                    edit_attention,
                    edit_maximum,
                    raw,
                    global_delta,
                    global_mean,
                    endpoint,
                ),
                dim=-1,
            )
        )
        return self.readout_dropout(representation), router_balance, route_weights

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.control_mode == "SOURCE_ONLY":
            representations, balance, route = self._paired_representations(
                batch,
                replace_candidate_sequence=True,
                replace_candidate_edit_base=True,
                suppress_edits=True,
            )
            mean = self.effect_head(representations[0]).squeeze(-1)
        elif self.control_mode == "NO_CANDIDATE_SEQUENCE":
            representations, balance, route = self._paired_representations(
                batch,
                replace_candidate_sequence=True,
            )
            mean = self.effect_head(representations[0]).squeeze(-1)
        elif self.control_mode == "EDIT_METADATA_ONLY":
            representations, balance, route = self._paired_representations(
                batch,
                replace_candidate_sequence=True,
            )
            identity, _, _ = self._paired_representations(
                batch,
                replace_candidate_sequence=True,
                replace_candidate_edit_base=True,
                suppress_edits=True,
            )
            mean = (self.effect_head(representations[0]) - self.effect_head(identity[0])).squeeze(-1)
        else:
            representations, balance, route = self._paired_representations(batch)
            directed = self.effect_head(representations).squeeze(-1)
            mean = 0.5 * (directed[0] - directed[1])
        identity = torch.all(
            batch["source_tokens"] == batch["candidate_tokens"], dim=1
        )
        mean = torch.where(identity, torch.zeros_like(mean), mean)
        mean = self.study_calibration(mean, batch["study_ids"])
        return {
            "mean": mean,
            "router_balance_loss": balance,
            "route_weights": route,
        }


def require_v4_trainable_parameter_range(
    model: XEditCriticV4,
    *,
    minimum: int = 120_000_000,
    maximum: int = 180_000_000,
    design_target_minimum: int = 165_000_000,
    design_target_maximum: int = 175_000_000,
) -> dict[str, Any]:
    """Hard-fail a capacity drift before optimizer construction."""

    count = model.trainable_parameter_count
    _require(minimum <= count <= maximum, "Critic V4 trainable parameter count is outside 120–180M")
    _require(design_target_minimum <= count <= design_target_maximum, "Critic V4 trainable parameter count missed the frozen 165–175M design target")
    return {
        "trainable_parameter_count": count,
        "minimum": minimum,
        "maximum": maximum,
        "design_target_minimum": design_target_minimum,
        "design_target_maximum": design_target_maximum,
        "module_counts": model.parameter_counts_by_module(),
        "passed": True,
    }
