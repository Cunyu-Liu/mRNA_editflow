"""Length-scalable source-relative Delta predictor for Route 2."""

from __future__ import annotations

import torch
from torch import nn


ROUTE2_DELTA_MODEL_KIND = "delta_anchored_position_aware_antisymmetric"


def normalized_position_channels(
    padding_mask: torch.Tensor,
    edited: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return absolute-in-sequence position and its edit-gated counterpart.

    Positions are normalized independently by each sequence's unpadded length so
    that 0 denotes the first base and 1 denotes the last base.  The second
    channel makes edit position directly observable without forcing the network
    to first learn a multiplication between the edit indicator and position.
    """

    valid = ~padding_mask
    length = valid.sum(dim=1, keepdim=True)
    denominator = (length - 1).clamp_min(1)
    index = torch.arange(padding_mask.shape[1], device=padding_mask.device).unsqueeze(0)
    position = (index / denominator).to(dtype=edited.dtype) * valid
    position = position.unsqueeze(-1)
    return position, position * edited


class ResidualConvBlock(nn.Module):
    def __init__(self, hidden_dim: int, dilation: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.depthwise = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=5, padding=2 * dilation,
            dilation=dilation, groups=hidden_dim,
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1),
        )

    def forward(self, values: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        residual = values
        hidden = self.norm(values).transpose(1, 2)
        hidden = self.pointwise(self.depthwise(hidden)).transpose(1, 2)
        return (residual + hidden) * (~padding_mask).unsqueeze(-1)


class Route2DeltaPredictor(nn.Module):
    """Anchored pair encoder with exact swap antisymmetry and identity zero."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        depth: int,
        study_count: int,
        assay_count: int,
        context_count: int,
        endpoint_count: int,
        region_count: int = 2,
        study_specific_scale_calibration: bool = False,
    ):
        super().__init__()
        if hidden_dim < 16 or depth < 1:
            raise ValueError("hidden_dim/depth are too small")
        if min(study_count, assay_count, context_count, endpoint_count, region_count) <= 0:
            raise ValueError("categorical vocabularies must be non-empty")
        category_dim = max(4, hidden_dim // 8)
        self.nucleotide = nn.Embedding(5, hidden_dim, padding_idx=4)
        self.study = nn.Embedding(study_count, category_dim)
        self.assay = nn.Embedding(assay_count, category_dim)
        self.context = nn.Embedding(context_count, category_dim)
        self.endpoint = nn.Embedding(endpoint_count, category_dim)
        self.region = nn.Embedding(region_count, category_dim)
        self.study_specific_scale_calibration = study_specific_scale_calibration
        if study_specific_scale_calibration:
            self.study_log_scale = nn.Embedding(study_count, 1)
            nn.init.zeros_(self.study_log_scale.weight)
        # Both the main critic and the same-information full-pair CNN receive
        # source, candidate, signed nucleotide delta, edit identity, normalized
        # absolute position, and edit-gated position.
        self.input_projection = nn.Linear(hidden_dim * 3 + 3, hidden_dim)
        self.blocks = nn.ModuleList(
            ResidualConvBlock(hidden_dim, dilation=2 ** (index % 4)) for index in range(depth)
        )
        self.region_scale = nn.Embedding(region_count, hidden_dim)
        self.region_shift = nn.Embedding(region_count, hidden_dim)
        pair_width = hidden_dim * 2 + category_dim * 5
        self.pair_fusion = nn.Sequential(
            nn.Linear(pair_width, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.score_head = nn.Linear(hidden_dim, 1)

    def _encode_pair(
        self,
        left_tokens: torch.Tensor,
        right_tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        study_ids: torch.Tensor,
        assay_ids: torch.Tensor,
        context_ids: torch.Tensor,
        endpoint_ids: torch.Tensor,
        region_ids: torch.Tensor,
    ) -> torch.Tensor:
        left = self.nucleotide(left_tokens)
        right = self.nucleotide(right_tokens)
        edited = ((left_tokens != right_tokens) & ~padding_mask).float().unsqueeze(-1)
        position, edited_position = normalized_position_channels(padding_mask, edited)
        hidden = self.input_projection(torch.cat([
            left, right, right - left, edited, position, edited_position,
        ], dim=-1))
        hidden = hidden * (~padding_mask).unsqueeze(-1)
        for block in self.blocks:
            hidden = block(hidden, padding_mask)
        hidden = hidden * (1.0 + self.region_scale(region_ids).unsqueeze(1))
        hidden = hidden + self.region_shift(region_ids).unsqueeze(1)
        valid = (~padding_mask).float().unsqueeze(-1)
        mean_pool = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        maximum_pool = hidden.masked_fill(padding_mask.unsqueeze(-1), -torch.inf).amax(dim=1)
        categories = torch.cat([
            self.study(study_ids), self.assay(assay_ids), self.context(context_ids),
            self.endpoint(endpoint_ids), self.region(region_ids),
        ], dim=-1)
        return self.pair_fusion(torch.cat([mean_pool, maximum_pool, categories], dim=-1))

    def forward(
        self,
        source_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        study_ids: torch.Tensor,
        assay_ids: torch.Tensor,
        context_ids: torch.Tensor,
        endpoint_ids: torch.Tensor,
        region_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        forward = self._encode_pair(
            source_tokens, candidate_tokens, padding_mask,
            study_ids, assay_ids, context_ids, endpoint_ids, region_ids,
        )
        reverse = self._encode_pair(
            candidate_tokens, source_tokens, padding_mask,
            study_ids, assay_ids, context_ids, endpoint_ids, region_ids,
        )
        mean = 0.5 * (self.score_head(forward) - self.score_head(reverse)).squeeze(-1)
        if self.study_specific_scale_calibration:
            mean = mean * torch.exp(self.study_log_scale(study_ids).squeeze(-1))
        log_variance = torch.zeros_like(mean)
        return {"mean": mean, "log_variance": log_variance}


class SequenceCNNEncoder(nn.Module):
    def __init__(self, hidden_dim: int, depth: int):
        super().__init__()
        self.embedding = nn.Embedding(5, hidden_dim, padding_idx=4)
        self.blocks = nn.ModuleList(
            ResidualConvBlock(hidden_dim, dilation=2 ** (index % 4)) for index in range(depth)
        )

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(tokens) * (~padding_mask).unsqueeze(-1)
        for block in self.blocks:
            hidden = block(hidden, padding_mask)
        valid = (~padding_mask).float().unsqueeze(-1)
        mean_pool = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        maximum_pool = hidden.masked_fill(padding_mask.unsqueeze(-1), -torch.inf).amax(dim=1)
        return torch.cat([mean_pool, maximum_pool], dim=-1)


class SequenceTransformerEncoder(nn.Module):
    def __init__(self, hidden_dim: int, depth: int, max_length: int):
        super().__init__()
        if hidden_dim % 4:
            raise ValueError("Transformer hidden_dim must be divisible by four")
        self.embedding = nn.Embedding(5, hidden_dim, padding_idx=4)
        self.position = nn.Embedding(max_length, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            hidden_dim, 4, hidden_dim * 4, dropout=0.1, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, depth)

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] > self.position.num_embeddings:
            raise ValueError("sequence exceeds configured Transformer maximum")
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        hidden = self.embedding(tokens) + self.position(positions)
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        valid = (~padding_mask).float().unsqueeze(-1)
        mean_pool = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        maximum_pool = hidden.masked_fill(padding_mask.unsqueeze(-1), -torch.inf).amax(dim=1)
        return torch.cat([mean_pool, maximum_pool], dim=-1)


class Route2NeuralBaseline(nn.Module):
    """Same-information neural baselines with a common prediction interface."""

    MODES = {"candidate_cnn", "siamese_cnn", "full_pair_cnn", "small_transformer"}

    def __init__(
        self,
        *,
        mode: str,
        hidden_dim: int,
        depth: int,
        study_count: int,
        assay_count: int,
        context_count: int,
        endpoint_count: int,
        region_count: int = 2,
        max_length: int = 2048,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown neural baseline mode: {mode}")
        self.mode = mode
        category_dim = max(4, hidden_dim // 8)
        if mode == "small_transformer":
            self.sequence_encoder = SequenceTransformerEncoder(hidden_dim, depth, max_length)
        else:
            self.sequence_encoder = SequenceCNNEncoder(hidden_dim, depth)
        if mode == "full_pair_cnn":
            self.pair_input = nn.Linear(hidden_dim * 3 + 3, hidden_dim)
            self.pair_blocks = nn.ModuleList(
                ResidualConvBlock(hidden_dim, dilation=2 ** (index % 4)) for index in range(depth)
            )
            representation_width = hidden_dim * 2
        elif mode == "candidate_cnn":
            representation_width = hidden_dim * 2
        else:
            representation_width = hidden_dim * 8
        self.study = nn.Embedding(study_count, category_dim)
        self.assay = nn.Embedding(assay_count, category_dim)
        self.context = nn.Embedding(context_count, category_dim)
        self.endpoint = nn.Embedding(endpoint_count, category_dim)
        self.region = nn.Embedding(region_count, category_dim)
        input_width = representation_width + category_dim * 5
        self.head = nn.Sequential(
            nn.Linear(input_width, hidden_dim * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
        )
        self.mean_head = nn.Linear(hidden_dim, 1)

    def _full_pair(self, source, candidate, padding_mask):
        left = self.sequence_encoder.embedding(source)
        right = self.sequence_encoder.embedding(candidate)
        edited = ((source != candidate) & ~padding_mask).float().unsqueeze(-1)
        position, edited_position = normalized_position_channels(padding_mask, edited)
        hidden = self.pair_input(torch.cat([
            left, right, right - left, edited, position, edited_position,
        ], dim=-1))
        hidden = hidden * (~padding_mask).unsqueeze(-1)
        for block in self.pair_blocks:
            hidden = block(hidden, padding_mask)
        valid = (~padding_mask).float().unsqueeze(-1)
        mean_pool = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        maximum_pool = hidden.masked_fill(padding_mask.unsqueeze(-1), -torch.inf).amax(dim=1)
        return torch.cat([mean_pool, maximum_pool], dim=-1)

    def forward(
        self,
        source_tokens,
        candidate_tokens,
        padding_mask,
        study_ids,
        assay_ids,
        context_ids,
        endpoint_ids,
        region_ids,
    ):
        if self.mode == "candidate_cnn":
            representation = self.sequence_encoder(candidate_tokens, padding_mask)
        elif self.mode == "full_pair_cnn":
            representation = self._full_pair(source_tokens, candidate_tokens, padding_mask)
        else:
            source = self.sequence_encoder(source_tokens, padding_mask)
            candidate = self.sequence_encoder(candidate_tokens, padding_mask)
            representation = torch.cat([source, candidate, candidate - source, torch.abs(candidate - source)], dim=-1)
        categories = torch.cat([
            self.study(study_ids), self.assay(assay_ids), self.context(context_ids),
            self.endpoint(endpoint_ids), self.region(region_ids),
        ], dim=-1)
        hidden = self.head(torch.cat([representation, categories], dim=-1))
        mean = self.mean_head(hidden).squeeze(-1)
        return {"mean": mean, "log_variance": torch.zeros_like(mean)}
