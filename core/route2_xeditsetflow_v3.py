"""Order-invariant XEditSetFlow V3 over the hard-legal SUB+STOP space."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from core.route2_xeditcritic_v3 import EndpointConditionerV1


class XEditSetFlowV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditSetFlowV3Error(message)


class ConditionalLowRankFiLMV3(nn.Module):
    def __init__(self, condition_width: int, model_width: int, rank: int = 16) -> None:
        super().__init__()
        self.code = nn.Linear(condition_width, rank)
        self.scale = nn.Linear(rank, model_width, bias=False)
        self.shift = nn.Linear(rank, model_width, bias=False)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)

    def forward(self, values: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        code = F.gelu(self.code(condition))
        return values * (1.0 + self.scale(code).unsqueeze(1)) + self.shift(code).unsqueeze(1)


class BlockLocalSelfAttentionV3(nn.Module):
    """Alternating shifted block attention with O(length * window) geometry."""

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        window: int = 64,
        shifted: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        _require(width % heads == 0, "local attention width is not divisible by heads")
        _require(window > 1 and window % 2 == 0, "local attention window must be positive and even")
        self.window = int(window)
        self.shift = self.window // 2 if shifted else 0
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )

    def forward(
        self, values: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        _require(values.ndim == 3 and padding_mask.shape == values.shape[:2], "local attention input geometry differs")
        batch, length, width = values.shape
        left = self.shift
        padded_length = left + length
        right = (-padded_length) % self.window
        padded_values = F.pad(values, (0, 0, left, right))
        padded_mask = F.pad(padding_mask, (left, right), value=True)
        block_count = padded_values.shape[1] // self.window
        blocks = padded_values.reshape(batch * block_count, self.window, width)
        masks = padded_mask.reshape(batch * block_count, self.window)
        # MultiheadAttention cannot accept an all-masked row.  A temporary zero
        # token makes the row numerically defined; the original mask zeros the
        # entire block immediately after attention.
        all_padding = masks.all(dim=1)
        safe_masks = masks.clone()
        safe_masks[all_padding, 0] = False
        attended, _ = self.attention(
            blocks, blocks, blocks, key_padding_mask=safe_masks, need_weights=False
        )
        attended = attended.masked_fill(masks.unsqueeze(-1), 0.0)
        restored = attended.reshape(batch, block_count * self.window, width)
        return restored[:, left : left + length]


class HybridSetFlowBlockV3(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        ffn_width: int,
        window: int,
        dilation: int,
        shifted: bool,
        dropout: float,
        condition_width: int,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = BlockLocalSelfAttentionV3(
            width=width,
            heads=heads,
            window=window,
            shifted=shifted,
            dropout=dropout,
        )
        self.convolution_norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(
            width,
            width,
            kernel_size=5,
            dilation=dilation,
            padding=2 * dilation,
            groups=width,
        )
        self.convolution_pointwise = nn.Sequential(
            nn.Conv1d(width, width, 1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, ffn_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_width, width),
        )
        self.dropout = nn.Dropout(dropout)
        self.condition_adapter = ConditionalLowRankFiLMV3(
            condition_width, width, rank=16
        )

    def forward(
        self,
        values: torch.Tensor,
        padding_mask: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        valid = (~padding_mask).unsqueeze(-1)
        values = values + self.dropout(
            self.attention(self.attention_norm(values), padding_mask)
        )
        convolution = self.depthwise(
            self.convolution_norm(values).transpose(1, 2)
        )
        convolution = self.convolution_pointwise(convolution).transpose(1, 2)
        values = values + self.dropout(convolution)
        values = values + self.dropout(self.ffn(self.ffn_norm(values)))
        values = self.condition_adapter(values, condition)
        return values * valid


class XEditSetFlowV3(nn.Module):
    """F2/F3 state model with per-position SUB rates and an independent STOP."""

    def __init__(
        self,
        *,
        model_width: int,
        depth: int,
        heads: int,
        ffn_width: int,
        assay_count: int,
        context_count: int,
        quantity_count: int,
        measurement_count: int,
        numerator_count: int,
        denominator_count: int,
        region_count: int = 2,
        pretrained_width: int = 768,
        local_attention_window: int = 64,
        dropout: float = 0.10,
        support_floor: float = 1e-8,
    ) -> None:
        super().__init__()
        _require(model_width >= 128 and depth >= 1, "SetFlow capacity is invalid")
        _require(support_floor > 0.0, "SetFlow support floor must be positive")
        self.support_floor = float(support_floor)
        condition_width = min(256, model_width)
        category_width = max(16, condition_width // 8)
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
        self.substitution_head = nn.Linear(model_width, 4)
        self.stop_attention = nn.Linear(model_width, 1)
        self.stop_head = nn.Sequential(
            nn.Linear(model_width + condition_width + 2, model_width),
            nn.GELU(),
            nn.Linear(model_width, 1),
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        source = batch["source_tokens"]
        current = batch["current_tokens"]
        padding = batch["padding_mask"]
        source_pretrained = batch["source_pretrained_tokens"]
        _require(source.shape == current.shape == padding.shape, "SetFlow sequence tensor geometry differs")
        _require(source_pretrained.shape[:2] == source.shape, "source pretrained tokens do not align")
        valid = ~padding
        edited = (source != current) & valid
        length = valid.sum(dim=1, keepdim=True)
        position_index = torch.arange(source.shape[1], device=source.device).unsqueeze(0)
        position = position_index / (length - 1).clamp_min(1)
        position = position.to(source_pretrained.dtype) * valid
        edited_count = edited.sum(dim=1).to(source_pretrained.dtype)
        remaining = batch["remaining_budget"].to(source_pretrained.dtype)
        assigned = edited_count + remaining
        progress = edited_count / assigned.clamp_min(1)
        per_position_state = torch.cat(
            (
                self.source_nucleotide(source),
                self.current_nucleotide(current),
                edited.to(source_pretrained.dtype).unsqueeze(-1),
                position.unsqueeze(-1),
                remaining.log1p().view(-1, 1, 1).expand(-1, source.shape[1], -1),
                progress.view(-1, 1, 1).expand(-1, source.shape[1], -1),
            ),
            dim=-1,
        )
        condition = self.endpoint_conditioner(
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
        hidden = self.input_norm(
            self.source_pretrained_projection(source_pretrained)
            + self.state_projection(per_position_state)
        ) * valid.unsqueeze(-1)
        for block in self.blocks:
            hidden = block(hidden, padding, condition)

        substitution_logits = self.substitution_head(hidden)
        positions_legal = valid & (~edited) & (batch["remaining_budget"][:, None] > 0)
        alt_ids = torch.arange(4, device=source.device).view(1, 1, 4)
        legal_substitutions = positions_legal.unsqueeze(-1) & (
            alt_ids != source.unsqueeze(-1)
        )
        substitution_logits = substitution_logits.masked_fill(
            ~legal_substitutions, -torch.inf
        )
        attention_logits = self.stop_attention(hidden).squeeze(-1).masked_fill(
            ~valid, -torch.inf
        )
        attention = torch.softmax(attention_logits, dim=1)
        pooled = (hidden * attention.unsqueeze(-1)).sum(dim=1)
        stop_logits = self.stop_head(
            torch.cat(
                (
                    pooled,
                    condition,
                    remaining.log1p().unsqueeze(-1),
                    progress.unsqueeze(-1),
                ),
                dim=-1,
            )
        )
        stop_legal = positions_legal.any(dim=1, keepdim=True)
        stop_logits = stop_logits.masked_fill(~stop_legal, -torch.inf)
        logits = torch.cat(
            (substitution_logits.reshape(source.shape[0], -1), stop_logits), dim=1
        )
        legal_mask = torch.cat(
            (legal_substitutions.reshape(source.shape[0], -1), stop_legal), dim=1
        )
        return logits, legal_mask

    def rates(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        logits, legal_mask = self.forward(batch)
        positive = F.softplus(logits) + torch.as_tensor(
            self.support_floor, device=logits.device, dtype=logits.dtype
        )
        return torch.where(legal_mask, positive, torch.zeros_like(positive)), legal_mask


def set_marginal_negative_log_likelihood_v1(
    rates: torch.Tensor,
    legal_mask: torch.Tensor,
    positive_action_mask: torch.Tensor,
    structural_budget_exhausted: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Negative log total probability assigned to every correct next edit."""

    _require(rates.shape == legal_mask.shape == positive_action_mask.shape, "set-marginal tensors differ")
    _require(structural_budget_exhausted.shape == (rates.shape[0],), "structural terminal flags differ")
    active = ~structural_budget_exhausted
    _require(bool(torch.all(positive_action_mask <= legal_mask).item()), "positive action is not hard-legal")
    _require(
        bool(torch.all(positive_action_mask.any(dim=1) == active).item()),
        "positive-set presence does not match structural terminal state",
    )
    log_rates = torch.full_like(rates, -torch.inf)
    log_rates[legal_mask] = torch.log(rates[legal_mask])
    log_normalizer = torch.logsumexp(log_rates, dim=1)
    positive_log_mass = torch.logsumexp(
        log_rates.masked_fill(~positive_action_mask, -torch.inf), dim=1
    )
    per_record = -(positive_log_mass[active] - log_normalizer[active])
    _require(per_record.numel() > 0, "batch contains only structural terminal states")
    if sample_weight is None:
        return per_record.mean()
    weights = sample_weight[active]
    return (per_record * weights).sum() / weights.sum().clamp_min(1e-12)
