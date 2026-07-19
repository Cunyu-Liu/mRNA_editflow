"""MRNA Edit-Flow generation head (MEF core model).

:class:`MRNAEditFormer` sits on top of frozen backbone embeddings and predicts,
for every position of the current (gap-free) sequence ``x_t``, the CTMC edit
rates and token distributions used by the Edit-Flow loss:

* ``rates``     : ``[B, L, 3]`` softplus non-negative rates ``(ins, sub, del)``.
* ``ins_probs`` : ``[B, L, V]`` insertion-token distribution (softmax).
* ``sub_probs`` : ``[B, L, V]`` substitution-token distribution (softmax).
* ``aux``       : ``[B, L, 2]`` optional structural head (MFE proxy,
  start-accessibility), gated by ``config.use_aux_struct``.

Here ``V == VOCAB_MODEL_SIZE`` (nucleotides + BOS + PAD) matching the reference
Edit-Flow head, so the reused z-space loss reprojection keeps working.

Region-conditioned codon-lattice operators (the core novelty)
-------------------------------------------------------------
Given per-position ``region_ids`` / ``phase_ids`` the head enforces the mRNA edit
grammar, each switch independently ablatable via :class:`ModelConfig`:

* **CDS substitution** (``use_codon_constraint``): ``sub_probs`` are masked to the
  nucleotides that keep the current codon *synonymous* (frame-safe, protein
  invariant) using :func:`synonymous_nt_sub_mask`.
* **CDS indels** (``codon_indel``): when ``False`` the ins/del *rates* are zeroed
  inside CDS (frame lock). When ``True`` only whole-codon indels are permitted,
  so ins/del rates survive only at codon-start (phase 0) CDS positions.
* **UTR**: unconstrained nt-level ins/sub/del.

Architecture
------------
backbone embedding -> input projection to ``model_dim`` -> (RoPE or absolute)
multi-head self-attention stack with FiLM(region+phase+time) modulation ->
LayerNorm -> heads. ``use_region_film`` toggles FiLM; ``use_rope`` toggles rotary
vs learned absolute positions.

Complexity: O(B * n_layers * L^2 * model_dim) (dense self-attention).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core.constants import (
    NUM_PHASES,
    NUM_REGIONS,
    PHASE_NONE,
    REGION_CDS,
    V as NUC_V,
    VOCAB_MODEL_SIZE,
)
from ..core.config import ModelConfig
from ..core.mrna_flow_utils import (
    REGION_SENTINEL,
    synonymous_nt_sub_mask,
)

_NEG_INF = -1e9  # float32/float64 additive logit mask value.


def _finite_neg_mask_value(dtype: torch.dtype) -> float:
    """Return a finite softmax mask value representable by ``dtype``.

    AMP may cast logits to ``float16``; filling such tensors with ``-1e9``
    overflows before softmax. ``-1e4`` is finite in fp16/bf16 and still makes
    ``exp(mask)`` numerically zero. Complexity: O(1).
    """
    if dtype in (torch.float16, torch.bfloat16):
        return -1e4
    return _NEG_INF


# ===========================================================================
# Time embedding (reused from Edit Flow)
# ===========================================================================
class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding of the continuous time ``t`` (Edit-Flow reuse)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, t: Tensor) -> Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        half_dim = self.hidden_dim // 2
        emb = math.log(10000.0) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float() * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.hidden_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb  # [B, hidden_dim]


# ===========================================================================
# Rotary position embeddings
# ===========================================================================
def _build_rope_cache(length: int, dim: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """Return ``(cos, sin)`` each ``[length, dim]`` for rotary embeddings."""
    half = dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, device=device).float() / max(half, 1)))
    pos = torch.arange(length, device=device).float()
    freqs = torch.outer(pos, inv_freq)  # [length, half]
    emb = torch.cat([freqs, freqs], dim=-1)  # [length, dim]
    return emb.cos(), emb.sin()


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to ``x`` of shape ``[B, H, L, Dh]``; cos/sin ``[L, Dh]``."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


# ===========================================================================
# FiLM modulation
# ===========================================================================
class FiLM(nn.Module):
    """Feature-wise linear modulation: ``y = (1 + gamma) * x + beta``.

    Conditioning vector (region + phase + time) produces per-channel gamma/beta.
    """

    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, feat_dim)
        self.to_beta = nn.Linear(cond_dim, feat_dim)
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        gamma = self.to_gamma(cond)
        beta = self.to_beta(cond)
        return (1 + gamma) * x + beta


# ===========================================================================
# Transformer block with optional RoPE + FiLM
# ===========================================================================
class EditFlowBlock(nn.Module):
    """Pre-norm self-attention + FFN block with optional RoPE and FiLM."""

    def __init__(self, cfg: ModelConfig, cond_dim: int):
        super().__init__()
        self.dim = cfg.model_dim
        self.num_heads = cfg.num_heads
        if self.dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.head_dim = self.dim // self.num_heads
        self.use_rope = cfg.use_rope
        self.use_film = cfg.use_region_film

        self.norm1 = nn.LayerNorm(self.dim)
        self.qkv = nn.Linear(self.dim, 3 * self.dim)
        self.attn_out = nn.Linear(self.dim, self.dim)
        self.norm2 = nn.LayerNorm(self.dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, self.dim * cfg.ffn_mult),
            nn.GELU(),
            nn.Linear(self.dim * cfg.ffn_mult, self.dim),
        )
        self.dropout = nn.Dropout(cfg.dropout)
        if self.use_film:
            self.film1 = FiLM(cond_dim, self.dim)
            self.film2 = FiLM(cond_dim, self.dim)

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        padding_mask: Optional[Tensor],
        rope: Optional[tuple[Tensor, Tensor]],
    ) -> Tensor:
        b, length, _ = x.shape
        h = self.norm1(x)
        if self.use_film:
            h = self.film1(h, cond)
        qkv = self.qkv(h).reshape(b, length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, Dh]
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.use_rope and rope is not None:
            cos, sin = rope
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

        attn_mask = None
        if padding_mask is not None:
            # [B, 1, 1, L] boolean mask over keys.
            attn_mask = padding_mask[:, None, None, :]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, _finite_neg_mask_value(scores.dtype))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        ctx = attn @ v  # [B, H, L, Dh]
        ctx = ctx.transpose(1, 2).reshape(b, length, self.dim)
        x = x + self.dropout(self.attn_out(ctx))

        h = self.norm2(x)
        if self.use_film:
            h = self.film2(h, cond)
        x = x + self.dropout(self.ffn(h))
        return x


# ===========================================================================
# The head
# ===========================================================================
class MRNAEditFormer(nn.Module):
    """Region-aware Edit-Flow transformer head over backbone embeddings.

    Parameters
    ----------
    cfg : ModelConfig
        Architecture + ablation switches.
    backbone_dim : int
        Width of the frozen backbone embeddings fed to :meth:`forward`.
    vocab_size : int
        Token vocabulary for the ins/sub heads (defaults to VOCAB_MODEL_SIZE).
    """

    def __init__(self, cfg: ModelConfig, backbone_dim: int, vocab_size: int = VOCAB_MODEL_SIZE):
        super().__init__()
        self.cfg = cfg
        self.dim = cfg.model_dim
        self.vocab_size = vocab_size

        # Trainable projection from (frozen) backbone features to model width.
        self.in_proj = nn.Linear(backbone_dim, self.dim)

        # Conditioning embeddings: region, phase, time -> cond_dim.
        self.region_emb = nn.Embedding(NUM_REGIONS + 1, self.dim)   # +1 sentinel
        self.phase_emb = nn.Embedding(NUM_PHASES + 1, self.dim)     # +1 PHASE_NONE
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(self.dim), nn.Linear(self.dim, self.dim), nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        self.cond_dim = self.dim

        # Absolute position embedding (used when RoPE is off).
        self.use_rope = cfg.use_rope
        if not self.use_rope:
            self.pos_emb = nn.Embedding(cfg.max_seq_len, self.dim)

        self.blocks = nn.ModuleList(
            [EditFlowBlock(cfg, self.cond_dim) for _ in range(cfg.num_layers)]
        )
        self.final_norm = nn.LayerNorm(self.dim)

        # Output heads.
        self.rates_out = nn.Sequential(
            nn.Linear(self.dim, self.dim), nn.SiLU(), nn.Linear(self.dim, 3)
        )
        self.ins_logits = nn.Sequential(
            nn.Linear(self.dim, self.dim), nn.SiLU(), nn.Linear(self.dim, vocab_size)
        )
        self.sub_logits = nn.Sequential(
            nn.Linear(self.dim, self.dim), nn.SiLU(), nn.Linear(self.dim, vocab_size)
        )
        self.use_aux = cfg.use_aux_struct
        if self.use_aux:
            self.aux_struct = nn.Sequential(
                nn.Linear(self.dim, self.dim), nn.SiLU(), nn.Linear(self.dim, 2)
            )

    # ------------------------------------------------------------------
    def _conditioning(self, region_ids: Tensor, phase_ids: Tensor, time_step: Tensor, length: int) -> Tensor:
        """Build the per-position FiLM conditioning tensor ``[B, L, cond_dim]``."""
        reg = self.region_emb(region_ids.clamp(0, NUM_REGIONS))
        pha = self.phase_emb(phase_ids.clamp(0, PHASE_NONE))
        tim = self.time_emb(time_step).unsqueeze(1).expand(-1, length, -1)
        return reg + pha + tim

    def _apply_codon_constraints(
        self,
        rates: Tensor,
        sub_logits: Tensor,
        token_ids: Tensor,
        region_ids: Tensor,
        phase_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Enforce region-conditioned codon-lattice grammar (in-place-safe).

        Returns ``(rates, sub_logits)`` after masking. See module docstring.
        """
        b, length = token_ids.shape
        is_cds = region_ids == REGION_CDS

        # --- CDS substitution: mask to synonymous nucleotides ---
        if self.cfg.use_codon_constraint:
            allowed_nt = synonymous_nt_sub_mask(token_ids, region_ids, phase_ids)  # [B,L,4]
            # Extend to full vocab: BOS/PAD channels are disallowed for sub anyway.
            allowed_full = torch.zeros(
                (b, length, self.vocab_size), dtype=torch.bool, device=token_ids.device
            )
            allowed_full[:, :, :NUC_V] = allowed_nt
            mask_value = _finite_neg_mask_value(sub_logits.dtype)
            sub_logits = sub_logits.masked_fill(~allowed_full, mask_value)

        # --- CDS indels: frame lock / whole-codon-only ---
        if self.cfg.use_codon_constraint:
            ins_rate, sub_rate, del_rate = rates[..., 0], rates[..., 1], rates[..., 2]
            if not self.cfg.codon_indel:
                # Forbid all nt-level ins/del inside CDS.
                keep = (~is_cds).to(dtype=rates.dtype)
                ins_rate = ins_rate * keep
                del_rate = del_rate * keep
            else:
                # Whole-codon indels: allow only at codon start (phase 0) in CDS.
                codon_start = is_cds & (phase_ids == 0)
                allow = (~is_cds | codon_start).to(dtype=rates.dtype)
                ins_rate = ins_rate * allow
                del_rate = del_rate * allow
            rates = torch.stack([ins_rate, sub_rate, del_rate], dim=-1)
        return rates, sub_logits

    # ------------------------------------------------------------------
    def encode(
        self,
        token_ids: Tensor,
        region_ids: Tensor,
        phase_ids: Tensor,
        time_step: Tensor,
        padding_mask: Tensor,
        backbone,
    ) -> Tensor:
        """Run backbone + conditioned transformer trunk and return ``[B, L, D]``.

        This is the representation consumed by :meth:`heads`. Splitting the trunk
        from the output heads lets region-specialized adapters insert a residual
        transform between them without changing this module's parameters or
        state dict. Complexity matches the dense self-attention stack,
        ``O(B * n_layers * L^2 * D)``.
        """
        b, length = token_ids.shape
        device = token_ids.device

        feats = backbone.embed(token_ids, region_ids, padding_mask)  # [B, L, backbone_dim]
        x = self.in_proj(feats)

        cond = self._conditioning(region_ids, phase_ids, time_step, length)

        if self.use_rope:
            rope = _build_rope_cache(length, self.dim // self.cfg.num_heads, device)
        else:
            pos = torch.arange(length, device=device).clamp(max=self.pos_emb.num_embeddings - 1)
            x = x + self.pos_emb(pos).unsqueeze(0)
            rope = None

        # Fold time into the residual stream too (helps when FiLM is off).
        x = x + self.time_emb(time_step).unsqueeze(1).expand(-1, length, -1)

        for block in self.blocks:
            x = block(x, cond, padding_mask, rope)

        return self.final_norm(x)

    def heads(
        self,
        x: Tensor,
        token_ids: Tensor,
        region_ids: Tensor,
        phase_ids: Tensor,
        padding_mask: Tensor,
    ) -> Dict[str, Tensor]:
        """Map trunk features ``x`` to CTMC rates + token distributions.

        Applies the region-conditioned codon-lattice grammar, zeroes padded
        positions and guards against non-finite values, exactly as the original
        monolithic ``forward``. Complexity is ``O(B * L * (D + V))``.
        """
        rates = F.softplus(self.rates_out(x))          # [B, L, 3] >= 0
        ins_logits = self.ins_logits(x)
        sub_logits = self.sub_logits(x)

        rates, sub_logits = self._apply_codon_constraints(
            rates, sub_logits, token_ids, region_ids, phase_ids
        )

        ins_probs = F.softmax(ins_logits, dim=-1)
        sub_probs = F.softmax(sub_logits, dim=-1)

        # Zero padded positions for all outputs.
        valid = (~padding_mask).unsqueeze(-1).to(x.dtype)
        rates = rates * valid
        ins_probs = ins_probs * valid
        sub_probs = sub_probs * valid

        aux = None
        if self.use_aux:
            aux = self.aux_struct(x) * valid

        # Finite guards (defensive; masks above are finite by construction).
        rates = torch.nan_to_num(rates, nan=0.0, posinf=1e4, neginf=0.0)
        ins_probs = torch.nan_to_num(ins_probs, nan=0.0)
        sub_probs = torch.nan_to_num(sub_probs, nan=0.0)
        if aux is not None:
            aux = torch.nan_to_num(aux, nan=0.0, posinf=1e4, neginf=-1e4)

        return {"rates": rates, "ins_probs": ins_probs, "sub_probs": sub_probs, "aux": aux}

    # ------------------------------------------------------------------
    def forward(
        self,
        token_ids: Tensor,
        region_ids: Tensor,
        phase_ids: Tensor,
        time_step: Tensor,
        padding_mask: Tensor,
        backbone,
    ) -> Dict[str, Tensor]:
        """Predict CTMC edit rates + token distributions for ``x_t``.

        Parameters
        ----------
        token_ids : Tensor[B, L] long
            Current gap-free sequence (BOS prepended, PAD elsewhere).
        region_ids : Tensor[B, L] long
            Per-position region label (``REGION_SENTINEL`` at BOS/PAD).
        phase_ids : Tensor[B, L] long
            Per-position codon phase (``PHASE_NONE`` outside CDS).
        time_step : Tensor[B, 1] float
            Flow time in ``[0, 1]``.
        padding_mask : Tensor[B, L] bool
            ``True`` at padded positions (excluded from attention + outputs).
        backbone : FrozenBackbone
            Frozen encoder supplying per-token features via ``.embed``.

        Returns
        -------
        dict with keys ``rates`` ``[B,L,3]``, ``ins_probs`` ``[B,L,V]``,
        ``sub_probs`` ``[B,L,V]``, ``aux`` ``[B,L,2]`` or ``None``. All PAD rows
        are zeroed; every value is finite.
        """
        x = self.encode(token_ids, region_ids, phase_ids, time_step, padding_mask, backbone)
        return self.heads(x, token_ids, region_ids, phase_ids, padding_mask)

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["MRNAEditFormer", "SinusoidalTimeEmbedding", "FiLM", "EditFlowBlock"]
