"""AtomicCodonHead: atomic synonymous codon substitution head (P0-5).

The current model (:class:`mrna_editflow.models.mrna_editformer.MRNAEditFormer`)
predicts nucleotide-level substitution rates and masks ``sub_probs`` to keep CDS
edits synonymous. Because each edit changes a single nucleotide, only codon
pairs that differ by exactly one nucleotide are reachable (e.g. ``CGU -> CGC``
for Arg). Some synonymous transitions require **two or more** nucleotide
changes (e.g. Arg ``CGU -> AGA``), which are unreachable through any
single-nucleotide path that stays synonymous at every intermediate step.

:class:`AtomicCodonHead` removes that limitation by operating directly at codon
resolution:

* it takes per-codon hidden states (pooled from the per-nt trunk output);
* it predicts a scalar per-codon substitution rate ``lambda_sub_codon``;
* it predicts a distribution ``Q_synonymous_codon`` over the synonymous codons
  of the current codon's amino acid;
* a single transition moves **atomically** from the original codon to the
  target synonymous codon, never passing through a nonsynonymous (or
  frameshifted) intermediate.

The head is a plain :class:`torch.nn.Module` and can be attached to
``MRNAEditFormer`` (e.g. ``model.codon_head = AtomicCodonHead(cfg.model_dim)``)
and invoked from the model's ``heads`` method on the pooled codon
representation. It introduces no GPU-only ops and is fully testable on CPU with
random tensors.

Complexity: ``O(B * L * (D + 64))`` per forward (linear heads + a ``[64, 64]``
synonym lookup); pooling is ``O(B * L * D)``.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .constants import (
    AA_TO_CODON_INDICES,
    CODON_INDEX_TO_AA,
    REGION_CDS,
    V,
)

# A finite logit mask value that survives fp16/bf16 softmax without overflow.
_NEG_INF_FP32 = -1e9
_NEG_INF_FP16 = -1e4


def _finite_neg_mask_value(dtype: torch.dtype) -> float:
    """Return a finite softmax mask value representable by ``dtype``.

    AMP may cast logits to ``float16``; ``-1e9`` overflows there. ``-1e4`` is
    finite in fp16/bf16 and still makes ``exp(mask)`` numerically zero.
    Complexity: O(1).
    """
    if dtype in (torch.float16, torch.bfloat16):
        return _NEG_INF_FP16
    return _NEG_INF_FP32


def _build_synonymous_matrix() -> Tensor:
    """Precompute the ``[64, 64]`` synonymous-codon bool matrix.

    ``M[i, j]`` is ``True`` iff codon ``j`` encodes the same amino acid as
    codon ``i`` (the stop group is included). Built once from
    :data:`~mrna_editflow.core.constants.CODON_INDEX_TO_AA`.
    Complexity: ``O(64 * #synonyms)`` one-time.
    """
    mat = torch.zeros((64, 64), dtype=torch.bool)
    for c in range(64):
        aa = CODON_INDEX_TO_AA[c]
        for j in AA_TO_CODON_INDICES[aa]:
            mat[c, j] = True
    return mat


def pool_per_codon(
    nt_hidden: Tensor,
    token_ids: Tensor,
    region_ids: Tensor,
    phase_ids: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pool per-nucleotide representations into per-codon representations.

    For every complete in-frame CDS codon (a phase-0 position whose following
    two positions are also CDS), the three nucleotide hidden states are
    mean-pooled into a single codon vector placed at the codon-start position.
    Non-codon positions are zeroed.

    Parameters
    ----------
    nt_hidden : Tensor[B, L, D]
        Per-nucleotide trunk features (e.g. output of
        :meth:`MRNAEditFormer.encode`).
    token_ids : Tensor[B, L] long
        Per-nucleotide token ids (used to compute the current codon index).
    region_ids : Tensor[B, L] long
        Per-nucleotide region ids.
    phase_ids : Tensor[B, L] long
        Per-nucleotide codon phases (``PHASE_NONE`` outside CDS).

    Returns
    -------
    codon_hidden : Tensor[B, L, D]
        Pooled codon features at codon-start positions, zero elsewhere.
    codon_indices : Tensor[B, L] long
        Current codon index in ``[0, 64)`` at codon-start positions (zero
        elsewhere; use ``codon_mask`` to select valid entries).
    codon_mask : Tensor[B, L] bool
        ``True`` at valid complete in-frame CDS codon-start positions.

    Complexity: ``O(B * L * D)`` (a few shifted gathers + a mean).
    """
    b, length, _ = nt_hidden.shape
    device = nt_hidden.device
    is_cds = region_ids == REGION_CDS  # [B, L]
    phase = phase_ids.clamp(0, 2)
    starts = is_cds & (phase == 0)  # [B, L]

    # Shifted CDS membership so that ``valid[s]`` requires s, s+1, s+2 all CDS.
    cds_next = torch.zeros_like(is_cds)
    cds_next[:, :-1] = is_cds[:, 1:]
    cds_next2 = torch.zeros_like(is_cds)
    cds_next2[:, :-2] = is_cds[:, 2:]
    valid = starts & cds_next & cds_next2  # [B, L]

    # Codon index at each position (meaningful only at codon starts).
    nt = token_ids.clamp(0, V - 1)
    nt1 = torch.zeros_like(nt)
    nt1[:, :-1] = nt[:, 1:]
    nt2 = torch.zeros_like(nt)
    nt2[:, :-2] = nt[:, 2:]
    codon_idx = nt * 16 + nt1 * 4 + nt2  # [B, L], base-4 place value

    # Mean-pool the three nucleotide hidden states onto the codon-start slot.
    h1 = torch.zeros_like(nt_hidden)
    h1[:, :-1] = nt_hidden[:, 1:]
    h2 = torch.zeros_like(nt_hidden)
    h2[:, :-2] = nt_hidden[:, 2:]
    pooled = (nt_hidden + h1 + h2) / 3.0  # [B, L, D]

    keep = valid.unsqueeze(-1).to(pooled.dtype)
    codon_hidden = pooled * keep
    codon_indices = codon_idx.clamp(0, 63)
    return codon_hidden, codon_indices, valid


class AtomicCodonHead(nn.Module):
    """Atomic synonymous-codon substitution head.

    Produces, for every codon-start position:

    * ``lambda_sub_codon`` : ``[B, L]`` non-negative substitution rate
      (softplus), the total rate of leaving the current codon for a synonymous
      alternative.
    * ``Q_synonymous_codon`` : ``[B, L, 64]`` distribution over synonymous
      target codons, **excluding the current codon** (a no-op self-transition is
      not a real substitution). For amino acids with a single codon (Met, Trp)
      there are no synonymous alternatives, so the distribution collapses back
      to the current codon and the rate naturally governs whether any move
      happens.

    Each non-zero ``lambda_sub_codon[b, s] * Q_synonymous_codon[b, s, j]`` is
    the rate of an **atomic** transition from the current codon to codon ``j``.
    Because both endpoints are synonymous by construction and the transition is
    direct, the path never visits a nonsynonymous or frameshifted intermediate.

    Parameters
    ----------
    model_dim : int
        Width of the pooled per-codon hidden states (``D``).
    n_codons : int, optional
        Codon vocabulary size (always 64 for the standard genetic code).

    Attributes
    ----------
    rate_head : nn.Module
        MLP producing a scalar rate (softplus on output).
    codon_logits : nn.Module
        MLP producing logits over the 64 codons.
    synonymous_mask : Tensor[64, 64] bool
        Registered (non-persistent) buffer marking synonymous codon pairs.
    """

    def __init__(self, model_dim: int, n_codons: int = 64) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.n_codons = int(n_codons)
        self.rate_head = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, 1),
        )
        self.codon_logits = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.n_codons),
        )
        self.register_buffer(
            "synonymous_mask", _build_synonymous_matrix(), persistent=False
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        codon_hidden: Tensor,
        codon_indices: Tensor,
        codon_mask: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        """Predict the atomic codon substitution field.

        Parameters
        ----------
        codon_hidden : Tensor[B, L, D]
            Pooled per-codon hidden states (e.g. from :func:`pool_per_codon`).
        codon_indices : Tensor[B, L] long
            Current codon index in ``[0, 64)`` at each position.
        codon_mask : Tensor[B, L] bool, optional
            ``True`` at valid codon-start positions. Outputs are zeroed where
            this is ``False``.

        Returns
        -------
        dict with keys
            * ``lambda_sub_codon`` : ``[B, L]`` non-negative rate.
            * ``Q_synonymous_codon`` : ``[B, L, 64]`` distribution.
            * ``synonymous_mask`` : ``[B, L, 64]`` bool mask of allowed target
              codons actually used (useful for diagnostics / loss masking).

        Complexity: ``O(B * L * (D + 64))``.
        """
        lam = F.softplus(self.rate_head(codon_hidden)).squeeze(-1)  # [B, L]
        logits = self.codon_logits(codon_hidden)  # [B, L, 64]

        idx = codon_indices.clamp(0, self.n_codons - 1).long()
        syn = self.synonymous_mask.to(logits.device)[idx]  # [B, L, 64] synonymous incl. self
        self_mask = F.one_hot(idx, num_classes=self.n_codons).bool()  # [B, L, 64]

        # Default targets: synonymous codons OTHER than the current one (a real
        # substitution). For codons with no synonymous alternatives (Met/Trp)
        # fall back to self so the softmax is well-defined (no move occurs).
        alternatives = syn & ~self_mask
        has_alt = alternatives.any(dim=-1, keepdim=True)  # [B, L, 1]
        allowed = torch.where(has_alt, alternatives, syn)

        mask_value = _finite_neg_mask_value(logits.dtype)
        logits = logits.masked_fill(~allowed, mask_value)
        q = F.softmax(logits, dim=-1)  # [B, L, 64]

        if codon_mask is not None:
            keep = codon_mask.to(lam.dtype)
            lam = lam * keep
            q = q * keep.unsqueeze(-1).to(q.dtype)
            allowed = allowed & codon_mask.unsqueeze(-1)

        # Finite guards (defensive; masked softmax is finite by construction).
        lam = torch.nan_to_num(lam, nan=0.0, posinf=1e4, neginf=0.0)
        q = torch.nan_to_num(q, nan=0.0)

        return {
            "lambda_sub_codon": lam,
            "Q_synonymous_codon": q,
            "synonymous_mask": allowed,
        }

    # ------------------------------------------------------------------
    def atomic_rates(self, codon_hidden: Tensor, codon_indices: Tensor, codon_mask: Tensor | None = None) -> Tensor:
        """Return the per-codon ``[B, L, 64]`` rate matrix ``lambda * Q``.

        Entry ``[b, s, j]`` is the rate of the atomic transition
        ``codon_indices[b, s] -> j``. Non-synonymous and self entries are zero.
        Convenience wrapper around :meth:`forward`. Complexity: ``O(B*L*64)``.
        """
        out = self.forward(codon_hidden, codon_indices, codon_mask)
        return out["lambda_sub_codon"].unsqueeze(-1) * out["Q_synonymous_codon"]


__all__ = [
    "AtomicCodonHead",
    "pool_per_codon",
]
