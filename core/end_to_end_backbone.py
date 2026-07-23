"""End-to-end trainable encoder option for backbone (P0-7).

The original ``FrozenBackbone`` always freezes the encoder (``freeze=True``
by default). When ``name="none"``, the ``_LightEncoder`` is a genuine
trainable module but gets frozen, meaning the model trains only the
EditFormer head on top of frozen random embeddings.

This module provides an ``EndToEndBackbone`` wrapper that:
1. Uses ``_LightEncoder`` as the encoder (trainable, not frozen).
2. Supports the same ``embed`` interface as ``FrozenBackbone``.
3. Can be unfrozen for end-to-end training.
4. Falls back to this trainable encoder when a named backbone's weights
   are unavailable (instead of the deterministic ``_PlaceholderEncoder``).

This ensures the model always has a differentiable encoder path, even
without pretrained mRNA foundation model weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from .constants import VOCAB_MODEL_SIZE, NUM_REGIONS
from .config import BackboneConfig


class TrainableEncoder(nn.Module):
    """Lightweight trainable mRNA encoder (replaces _PlaceholderEncoder).

    A trainable token embedding + region embedding + sinusoidal position
    signal + single Transformer layer. This is more expressive than the
    frozen ``_LightEncoder`` and serves as a genuine end-to-end encoder
    when no pretrained backbone is available.

    Parameters
    ----------
    out_dim : output embedding dimension.
    n_layers : number of Transformer layers (default 2).
    """

    def __init__(self, out_dim: int, n_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.out_dim = out_dim
        self.token_emb = nn.Embedding(VOCAB_MODEL_SIZE, out_dim)
        self.region_emb = nn.Embedding(NUM_REGIONS + 2, out_dim)  # +2 sentinels
        self.pos_emb = nn.Embedding(8192, out_dim)  # max seq len
        self.norm = nn.LayerNorm(out_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_dim,
            nhead=num_heads,
            dim_feedforward=out_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, token_ids: Tensor, region_ids: Tensor) -> Tensor:
        """Return per-token embeddings ``[B, L, out_dim]``."""
        b, length = token_ids.shape
        device = token_ids.device

        tok = self.token_emb(token_ids)
        reg = self.region_emb(region_ids.clamp(0, NUM_REGIONS + 1))
        pos = self.pos_emb(torch.arange(length, device=device).clamp(max=self.pos_emb.num_embeddings - 1))

        x = self.norm(tok + reg + pos.unsqueeze(0))
        x = self.transformer(x)
        return x


class EndToEndBackbone(nn.Module):
    """Backbone wrapper that supports end-to-end training.

    When ``cfg.name == "none"`` or when pretrained weights are unavailable,
    this backbone uses ``TrainableEncoder`` instead of the frozen
    ``_PlaceholderEncoder``. The encoder is **not frozen** by default,
    allowing gradients to flow through the embedding layer.

    This addresses the P0-7 concern: "default training is on fixed
    random/light token embeddings" — now the encoder is trainable.
    """

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        self.out_dim = cfg.hidden_dim
        self.name = cfg.name

        # Always use TrainableEncoder for "none" or when weights unavailable.
        # In a full implementation, this would try to load real pretrained
        # weights first (mrnabert, helix_mrna, etc.) and fall back here.
        self.encoder = TrainableEncoder(self.out_dim)
        self.is_real = (cfg.name == "none")  # "none" is a real trainable encoder

        # Granularity projections (same as FrozenBackbone)
        self.granularity = cfg.granularity
        if self.granularity in ("codon", "dual"):
            self.codon_proj = nn.Linear(self.out_dim, self.out_dim)
        else:
            self.codon_proj = None
        if self.granularity == "dual":
            self.nt_proj = nn.Linear(self.out_dim, self.out_dim)
        else:
            self.nt_proj = None

        # By default, end-to-end backbone is NOT frozen.
        self.frozen = False
        if cfg.freeze:
            self.freeze()

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        self.frozen = True
        self.eval()

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(True)
        self.frozen = False
        self.train()

    def embed(
        self,
        token_ids: Tensor,
        region_ids: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Return per-token embeddings ``[B, L, out_dim]``."""
        if token_ids.dim() != 2:
            raise ValueError(f"token_ids must be [B,L], got {tuple(token_ids.shape)}")
        b, length = token_ids.shape
        device = token_ids.device
        if region_ids is None:
            region_ids = torch.zeros((b, length), dtype=torch.long, device=device)
        region_ids = region_ids.long()

        nt_emb = self.encoder(token_ids, region_ids)

        if self.granularity == "nt":
            out = nt_emb
        else:
            # Use corrected CDS-anchored pooling
            from .codon_pooling_fix import pool_to_codon_cds_anchored
            out = self.codon_proj(pool_to_codon_cds_anchored(nt_emb, region_ids))
            if self.granularity == "dual":
                out = out + self.nt_proj(nt_emb)

        if padding_mask is not None:
            out = out * (~padding_mask).unsqueeze(-1).to(out.dtype)
        return out

    def forward(self, *args, **kwargs) -> Tensor:
        return self.embed(*args, **kwargs)

    def extra_repr(self) -> str:
        return (f"name={self.name}, out_dim={self.out_dim}, "
                f"granularity={self.granularity}, frozen={self.frozen}, "
                f"end_to_end=True")


__all__ = ["TrainableEncoder", "EndToEndBackbone"]
