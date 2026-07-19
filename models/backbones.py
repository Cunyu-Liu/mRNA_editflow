"""Frozen foundation-model backbones for mRNA-EditFlow (MEF).

This module provides :class:`FrozenBackbone`, a uniform wrapper that turns any
supported sequence encoder into a per-token embedding function

    ``embed(token_ids, region_ids, padding_mask) -> Tensor[B, L, out_dim]``

so the downstream Edit-Flow head (:mod:`mrna_editflow.models.mrna_editformer`)
can be written once and stay agnostic to which encoder produced the features.

Backbone catalogue
------------------
* ``none`` -- **REAL, from-scratch light embedding** (token + region + fixed
  sinusoidal position). Fully functional on CPU offline; this is the default for
  smoke tests and the ``model_dim=32`` unit tests.
* mRNA-native encoders -- ``mrnabert``, ``helix_mrna``, ``orthrus``,
  ``orthrus_mlm``, ``lamar``: **ADAPTER-STUB**. The adapter documents exactly how
  the real HF / torch checkpoint would be loaded from ``weights_path`` and then
  gracefully falls back to a *deterministic placeholder* encoder (identical
  output shape, reproducible across runs) whenever the weights or the third
  party library are unavailable -- i.e. always, in this offline environment.
* ncRNA controls -- ``rna_fm``, ``rinalmo``: same ADAPTER-STUB contract; used as
  negative-control encoders in ablations.

Granularity alignment
----------------------
Foundation models tokenise mRNA at different granularities. ``embed`` always
returns *nucleotide-resolution* features:

* ``nt``     -- 1:1, encoder output is used directly.
* ``codon``  -- codon-level reps are index-expanded across their 3 constituent
  nt positions (see :func:`upsample_codon_to_nt`) then linearly projected.
* ``dual``   -- both nt and codon tracks are combined (sum of two projections).

Freezing
--------
When ``BackboneConfig.freeze`` is ``True`` every parameter owned by the wrapper
(encoder *and* granularity projections) has ``requires_grad_(False)``; autograd
therefore never populates their ``.grad``. The downstream head keeps its own
trainable input projection so gradients still flow into the trainable part.

Complexity
----------
``embed`` is O(B * L * out_dim) for the light/placeholder encoders. Codon
pooling/upsampling adds an O(B * L * out_dim) pass. No quadratic cost is
incurred here (self-attention lives in the head).
"""
from __future__ import annotations

import math
import zlib
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..core.constants import (
    NUM_REGIONS,
    PAD_TOKEN,
    VOCAB_MODEL_SIZE,
)
from ..core.config import BackboneConfig

# Which names are genuinely mRNA-native foundation models vs ncRNA controls.
MRNA_NATIVE_BACKBONES = frozenset(
    {"mrnabert", "helix_mrna", "orthrus", "orthrus_mlm", "lamar"}
)
NCRNA_CONTROL_BACKBONES = frozenset({"rna_fm", "rinalmo"})
EXTERNAL_BACKBONES = MRNA_NATIVE_BACKBONES | NCRNA_CONTROL_BACKBONES
SUPPORTED_BACKBONES = EXTERNAL_BACKBONES | {"none"}

# Region embedding table has one extra sentinel row for BOS/PAD/out-of-range.
_REGION_TABLE_SIZE = NUM_REGIONS + 1
_REGION_SENTINEL = NUM_REGIONS


def _stable_seed(name: str) -> int:
    """Deterministic, process-independent seed derived from a string.

    ``hash()`` on ``str`` is salted per interpreter run, so we use CRC32 to make
    placeholder encoders reproducible across runs (important for the embedding
    cache).
    """
    return int(zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF)


def _sinusoidal_positions(length: int, dim: int, device: torch.device) -> Tensor:
    """Standard fixed sinusoidal position encoding ``[length, dim]``."""
    if length == 0:
        return torch.zeros((0, dim), device=device)
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    half = max(dim // 2, 1)
    div = torch.exp(
        torch.arange(half, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(half - 1, 1))
    )
    out = torch.zeros((length, dim), device=device)
    out[:, 0:half] = torch.sin(pos * div)
    # cos block; guard the odd-dim tail.
    cos_block = torch.cos(pos * div)
    out[:, half : 2 * half] = cos_block[:, : dim - half]
    return out


def upsample_codon_to_nt(codon_emb: Tensor, region_ids: Tensor) -> Tensor:
    """Expand codon-level embeddings to nucleotide resolution.

    Each codon representation is repeated across its 3 constituent nt positions
    (index expansion via :func:`torch.repeat_interleave`) and then cropped or
    zero-padded to the nt length ``L = region_ids.shape[1]``.

    Parameters
    ----------
    codon_emb : Tensor[B, n_codons, D]
        Codon-level features from a codon-tokenised backbone.
    region_ids : Tensor[B, L]
        Provides the target nt length ``L`` (and device); values are otherwise
        unused by the pure index-expansion mapping.

    Returns
    -------
    Tensor[B, L, D]

    Complexity: O(B * L * D).
    """
    if codon_emb.dim() != 3:
        raise ValueError(f"codon_emb must be [B,n_codons,D], got {tuple(codon_emb.shape)}")
    b, _, d = codon_emb.shape
    length = int(region_ids.shape[1])
    expanded = codon_emb.repeat_interleave(3, dim=1)  # [B, 3*n_codons, D]
    cur = expanded.shape[1]
    if cur < length:
        pad = torch.zeros((b, length - cur, d), device=expanded.device, dtype=expanded.dtype)
        expanded = torch.cat([expanded, pad], dim=1)
    return expanded[:, :length]


class _LightEncoder(nn.Module):
    """Real from-scratch nucleotide encoder (the ``none`` backbone).

    A trainable token embedding over the full model vocabulary plus a region
    embedding, augmented with a fixed sinusoidal positional signal. This is a
    genuine, differentiable module -- not a placeholder -- and is the reference
    encoder used by the offline tests.
    """

    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self.token_emb = nn.Embedding(VOCAB_MODEL_SIZE, out_dim)
        self.region_emb = nn.Embedding(_REGION_TABLE_SIZE, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, token_ids: Tensor, region_ids: Tensor) -> Tensor:
        b, length = token_ids.shape
        tok = token_ids.clamp(0, VOCAB_MODEL_SIZE - 1)
        reg = region_ids.clamp(0, _REGION_SENTINEL)
        emb = self.token_emb(tok) + self.region_emb(reg)
        pos = _sinusoidal_positions(length, self.out_dim, token_ids.device)
        emb = emb + pos.unsqueeze(0)
        return self.norm(emb)


class _PlaceholderEncoder(nn.Module):
    """Deterministic stand-in for an unavailable pretrained backbone.

    Produces embeddings with the *same shape* a real encoder would, seeded
    reproducibly from the backbone name so that (a) different backbones give
    different features and (b) a given backbone gives identical features across
    runs -- a hard requirement for the offline embedding cache.

    It is intentionally *not* trainable-quality; it exists only so the whole
    pipeline runs end-to-end offline. All weights are initialised from a fixed
    generator and then frozen.
    """

    def __init__(self, out_dim: int, name: str):
        super().__init__()
        self.out_dim = out_dim
        self.name = name
        self.token_emb = nn.Embedding(VOCAB_MODEL_SIZE, out_dim)
        self.region_emb = nn.Embedding(_REGION_TABLE_SIZE, out_dim)
        gen = torch.Generator().manual_seed(_stable_seed(name))
        with torch.no_grad():
            self.token_emb.weight.copy_(
                torch.randn(self.token_emb.weight.shape, generator=gen) * 0.02
            )
            self.region_emb.weight.copy_(
                torch.randn(self.region_emb.weight.shape, generator=gen) * 0.02
            )

    def forward(self, token_ids: Tensor, region_ids: Tensor) -> Tensor:
        b, length = token_ids.shape
        tok = token_ids.clamp(0, VOCAB_MODEL_SIZE - 1)
        reg = region_ids.clamp(0, _REGION_SENTINEL)
        emb = self.token_emb(tok) + self.region_emb(reg)
        pos = _sinusoidal_positions(length, self.out_dim, token_ids.device)
        return emb + pos.unsqueeze(0)


def _try_load_pretrained(name: str, weights_path: Optional[str], out_dim: int):
    """Attempt to load a real pretrained encoder; return ``None`` to fall back.

    This is the single seam where real checkpoint loading would live. For each
    supported external backbone the real integration would look like::

        # mrnabert / lamar (HF):
        #   from transformers import AutoModel
        #   m = AutoModel.from_pretrained(weights_path); return _HFAdapter(m)
        # orthrus / orthrus_mlm (torch .pt):
        #   sd = torch.load(weights_path, map_location="cpu"); ...
        # rna_fm / rinalmo (torch hub / .ckpt):
        #   ...

    In this offline environment neither the libraries nor the weights are
    present, so we always return ``None`` and the caller substitutes a
    deterministic :class:`_PlaceholderEncoder`. The function never raises.
    """
    if weights_path is None:
        return None
    try:  # pragma: no cover - exercised only when real weights exist.
        import os

        if not os.path.exists(weights_path):
            return None
        # Real loaders would be dispatched here on ``name``. Absent the third
        # party deps we conservatively decline and let the caller fall back.
        return None
    except Exception:
        return None


class FrozenBackbone(nn.Module):
    """Uniform, freezable per-token embedding wrapper.

    Parameters
    ----------
    cfg : BackboneConfig
        Selects the encoder (``cfg.name``), output width (``cfg.hidden_dim``),
        freezing (``cfg.freeze``) and token granularity (``cfg.granularity``).

    Attributes
    ----------
    out_dim : int
        Width of the returned embeddings (== ``cfg.hidden_dim``).
    is_real : bool
        ``True`` only for ``none`` (a genuine learnable encoder) or when a real
        pretrained checkpoint was successfully loaded; ``False`` for stubs.
    """

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        if cfg.name not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"unknown backbone {cfg.name!r}; supported: {sorted(SUPPORTED_BACKBONES)}"
            )
        self.cfg = cfg
        self.name = cfg.name
        self.out_dim = cfg.hidden_dim
        self.granularity = cfg.granularity
        if self.granularity not in ("nt", "dual", "codon"):
            raise ValueError(f"granularity must be nt|dual|codon, got {self.granularity!r}")

        # --- build the encoder ---
        self.is_real = False
        if cfg.name == "none":
            self.encoder = _LightEncoder(self.out_dim)
            self.is_real = True
        else:
            loaded = _try_load_pretrained(cfg.name, cfg.weights_path, self.out_dim)
            if loaded is not None:  # pragma: no cover - needs real weights.
                self.encoder = loaded
                self.is_real = True
            else:
                self.encoder = _PlaceholderEncoder(self.out_dim, cfg.name)
                self.is_real = False

        # --- granularity projections (learnable, but frozen with the backbone) ---
        # ``codon`` uses a codon-track projection; ``dual`` adds an nt-track too.
        if self.granularity in ("codon", "dual"):
            self.codon_proj = nn.Linear(self.out_dim, self.out_dim)
        else:
            self.codon_proj = None
        if self.granularity == "dual":
            self.nt_proj = nn.Linear(self.out_dim, self.out_dim)
        else:
            self.nt_proj = None

        self.frozen = bool(cfg.freeze)
        if self.frozen:
            self.freeze()

    # ------------------------------------------------------------------
    def freeze(self) -> None:
        """Set ``requires_grad=False`` on every backbone parameter."""
        for p in self.parameters():
            p.requires_grad_(False)
        self.frozen = True
        self.eval()

    def unfreeze(self) -> None:  # pragma: no cover - convenience for fine-tuning.
        for p in self.parameters():
            p.requires_grad_(True)
        self.frozen = False

    # ------------------------------------------------------------------
    def _pool_to_codon(self, nt_emb: Tensor) -> Tensor:
        """Average consecutive nt triplets into codon reps ``[B, ceil(L/3), D]``."""
        b, length, d = nt_emb.shape
        pad = (3 - length % 3) % 3
        if pad:
            nt_emb = torch.cat(
                [nt_emb, torch.zeros((b, pad, d), device=nt_emb.device, dtype=nt_emb.dtype)],
                dim=1,
            )
        n_codons = nt_emb.shape[1] // 3
        return nt_emb.view(b, n_codons, 3, d).mean(dim=2)

    def embed(
        self,
        token_ids: Tensor,
        region_ids: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Return per-token embeddings ``[B, L, out_dim]``.

        Parameters
        ----------
        token_ids : Tensor[B, L] long
            Nucleotide/BOS/PAD ids in x-space (GAP-free).
        region_ids : Tensor[B, L] long, optional
            Region label per position; defaults to zeros (all 5'UTR).
        padding_mask : Tensor[B, L] bool, optional
            ``True`` at padded positions; those rows are zeroed in the output.
        """
        if token_ids.dim() != 2:
            raise ValueError(f"token_ids must be [B,L], got {tuple(token_ids.shape)}")
        b, length = token_ids.shape
        device = token_ids.device
        if region_ids is None:
            region_ids = torch.zeros((b, length), dtype=torch.long, device=device)
        region_ids = region_ids.long()

        nt_emb = self.encoder(token_ids, region_ids)  # [B, L, out_dim]

        if self.granularity == "nt":
            out = nt_emb
        else:
            codon_emb = self._pool_to_codon(nt_emb)                     # [B, n_codons, D]
            up = upsample_codon_to_nt(codon_emb, region_ids)            # [B, L, D]
            out = self.codon_proj(up)
            if self.granularity == "dual":
                out = out + self.nt_proj(nt_emb)

        if padding_mask is not None:
            out = out * (~padding_mask).unsqueeze(-1).to(out.dtype)
        return out

    def forward(self, *args, **kwargs) -> Tensor:  # convenience alias
        return self.embed(*args, **kwargs)

    def extra_repr(self) -> str:
        kind = "real" if self.is_real else "adapter-stub"
        return f"name={self.name}, out_dim={self.out_dim}, granularity={self.granularity}, kind={kind}, frozen={self.frozen}"


__all__ = [
    "FrozenBackbone",
    "upsample_codon_to_nt",
    "MRNA_NATIVE_BACKBONES",
    "NCRNA_CONTROL_BACKBONES",
    "SUPPORTED_BACKBONES",
]
