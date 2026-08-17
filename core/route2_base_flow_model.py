"""Small learned base-flow model for the Route 2 SUB+STOP action space."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def normalized_generation_state_channels(
    padding_mask: torch.Tensor,
    edited: torch.Tensor,
    remaining_budget: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expose absolute position, edited position, and trajectory progress.

    The progress channel is the fraction of the assigned edit budget already
    consumed.  It is derived from the current state, so it remains available at
    generation time without consulting a measured candidate or an evaluator.
    """

    valid = ~padding_mask
    length = valid.sum(dim=1, keepdim=True)
    denominator = (length - 1).clamp_min(1)
    index = torch.arange(padding_mask.shape[1], device=padding_mask.device).unsqueeze(0)
    position = (index / denominator).to(dtype=torch.float32) * valid
    position = position.unsqueeze(-1)
    edited_float = edited.to(dtype=torch.float32).unsqueeze(-1)
    edited_count = edited.sum(dim=1).to(dtype=torch.float32)
    assigned_budget = edited_count + remaining_budget.to(dtype=torch.float32).clamp_min(0)
    progress = (edited_count / assigned_budget.clamp_min(1.0)).view(-1, 1, 1)
    progress = progress.expand(-1, padding_mask.shape[1], -1) * valid.unsqueeze(-1)
    return position, position * edited_float, progress


class Route2BaseFlowModel(nn.Module):
    """Shared sequence trunk with 5UTR/3UTR adapters and hard legal masking."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        assay_count: int,
        context_count: int,
        region_count: int = 2,
        support_floor: float = 1e-8,
        position_progress_features: bool = False,
    ):
        super().__init__()
        if hidden_dim < 8:
            raise ValueError("hidden_dim must be at least 8")
        if min(assay_count, context_count, region_count) <= 0:
            raise ValueError("categorical vocabularies must be non-empty")
        if support_floor <= 0.0:
            raise ValueError("support_floor must be strictly positive")
        self.support_floor = float(support_floor)
        self.position_progress_features = bool(position_progress_features)
        embed_dim = hidden_dim // 2
        category_dim = max(4, hidden_dim // 8)
        self.nucleotide = nn.Embedding(5, embed_dim, padding_idx=4)
        self.assay = nn.Embedding(assay_count, category_dim)
        self.context = nn.Embedding(context_count, category_dim)
        self.region = nn.Embedding(region_count, category_dim)
        input_dim = embed_dim * 2 + category_dim * 3 + 2
        if self.position_progress_features:
            input_dim += 3
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.shared_trunk = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.region_scale = nn.Embedding(region_count, hidden_dim)
        self.region_shift = nn.Embedding(region_count, hidden_dim)
        self.substitution_head = nn.Linear(hidden_dim, 4)
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim + 1 + int(self.position_progress_features), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        source_tokens: torch.Tensor,
        current_tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        region_ids: torch.Tensor,
        assay_ids: torch.Tensor,
        context_ids: torch.Tensor,
        remaining_budget: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return concatenated legal logits and their boolean mask.

        The final column is STOP.  Position/alt columns use
        ``position * 4 + alt_token``.  Illegal entries are negative infinity
        before normalization; callers never score a post-normalization mask.
        """

        if source_tokens.shape != current_tokens.shape or source_tokens.shape != padding_mask.shape:
            raise ValueError("source/current/padding shapes differ")
        if source_tokens.ndim != 2:
            raise ValueError("token tensors must have shape [batch, length]")
        batch, length = source_tokens.shape
        for values, label in (
            (region_ids, "region"),
            (assay_ids, "assay"),
            (context_ids, "context"),
            (remaining_budget, "remaining budget"),
        ):
            if values.shape != (batch,):
                raise ValueError(f"{label} tensor must have shape [batch]")

        edited = (source_tokens != current_tokens) & ~padding_mask
        categorical = torch.cat(
            [self.region(region_ids), self.assay(assay_ids), self.context(context_ids)], dim=-1
        ).unsqueeze(1).expand(-1, length, -1)
        budget_feature = remaining_budget.float().clamp_min(0).log1p().view(batch, 1, 1).expand(-1, length, -1)
        feature_parts = [
            self.nucleotide(source_tokens),
            self.nucleotide(current_tokens),
            categorical,
            edited.float().unsqueeze(-1),
            budget_feature,
        ]
        progress = None
        if self.position_progress_features:
            position, edited_position, progress = normalized_generation_state_channels(
                padding_mask, edited, remaining_budget
            )
            feature_parts.extend((position, edited_position, progress))
        features = torch.cat(feature_parts, dim=-1)
        hidden = self.input_projection(features)
        hidden = self.shared_trunk(hidden.transpose(1, 2)).transpose(1, 2)
        scale = 1.0 + self.region_scale(region_ids).unsqueeze(1)
        shift = self.region_shift(region_ids).unsqueeze(1)
        hidden = hidden * scale + shift

        substitution_logits = self.substitution_head(hidden)
        positions_legal = (~padding_mask) & (~edited) & (remaining_budget[:, None] > 0)
        alt_ids = torch.arange(4, device=source_tokens.device).view(1, 1, 4)
        legal_substitutions = positions_legal.unsqueeze(-1) & (alt_ids != source_tokens.unsqueeze(-1))
        substitution_logits = substitution_logits.masked_fill(~legal_substitutions, -torch.inf)

        valid = (~padding_mask).float().unsqueeze(-1)
        pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        stop_parts = [pooled, remaining_budget.float().log1p().unsqueeze(-1)]
        if progress is not None:
            stop_parts.append(progress[:, 0])
        stop_input = torch.cat(stop_parts, dim=-1)
        stop_logits = self.stop_head(stop_input)
        stop_legal = positions_legal.any(dim=1, keepdim=True)
        stop_logits = stop_logits.masked_fill(~stop_legal, -torch.inf)

        logits = torch.cat([substitution_logits.reshape(batch, length * 4), stop_logits], dim=1)
        legal_mask = torch.cat([legal_substitutions.reshape(batch, length * 4), stop_legal], dim=1)
        return logits, legal_mask

    def rates(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        logits, legal_mask = self.forward(*args, **kwargs)
        rates = torch.zeros_like(logits)
        rates[legal_mask] = F.softplus(logits[legal_mask]) + self.support_floor
        return rates, legal_mask
