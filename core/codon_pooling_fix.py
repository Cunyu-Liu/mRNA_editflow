"""Fix for codon pooling offset (P0-6).

The original ``_pool_to_codon`` in ``backbones.py`` pools every 3 consecutive
nucleotides starting from position 0. But position 0 is BOS, and CDS starts
at ``cds_start`` (which varies per sequence and includes a BOS token + 5'UTR).
This causes codon embeddings to be systematically misaligned with real codons.

This module provides corrected versions that:
1. Find the CDS start from ``region_ids`` (first position where region == REGION_CDS).
2. Pool only CDS positions into codons, starting from the correct offset.
3. UTR/BOS positions pass through without codon pooling.
4. Upsample correctly reverses the mapping.

The corrected functions are drop-in replacements for the original ``_pool_to_codon``
and ``upsample_codon_to_nt``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .constants import REGION_CDS


def pool_to_codon_cds_anchored(
    nt_emb: Tensor,
    region_ids: Tensor,
) -> Tensor:
    """Pool nt embeddings into codon reps, anchored to CDS start.

    Unlike the original ``_pool_to_codon``, this function:
    - Finds the CDS start from ``region_ids`` (first REGION_CDS position).
    - Pools only CDS positions into codons (starting from the correct phase 0).
    - Non-CDS positions (BOS, 5'UTR, 3'UTR) are passed through unchanged at
      nt resolution.

    This means codon ``i`` in the output corresponds to CDS codon ``i``
    (i.e., amino acid ``i`` in the protein), not an arbitrary window of 3 nt
    starting from position 0.

    Parameters
    ----------
    nt_emb : Tensor[B, L, D]
        Per-nucleotide embeddings.
    region_ids : Tensor[B, L]
        Per-position region ids.

    Returns
    -------
    Tensor[B, L, D]
        Codon-pooled embeddings at CDS positions, nt-level elsewhere.
        The output length matches the input length (CDS positions are
        replaced with their codon-pooled representation, upsampled back).

    Complexity: O(B * L * D).
    """
    b, length, d = nt_emb.shape
    device = nt_emb.device
    out = nt_emb.clone()

    for batch_idx in range(b):
        rids = region_ids[batch_idx]
        # Find CDS region boundaries
        cds_mask = rids == REGION_CDS
        if not cds_mask.any():
            continue  # no CDS in this sequence

        cds_start = int(cds_mask.nonzero()[0].item())
        cds_end = length
        for i in range(cds_start, length):
            if rids[i] != REGION_CDS:
                cds_end = i
                break
        cds_len = cds_end - cds_start

        if cds_len < 3:
            continue  # CDS too short for codon pooling

        # Align to codon boundary (trim to multiple of 3)
        n_codons = cds_len // 3
        cds_trimmed = cds_len - (cds_len % 3)
        if cds_trimmed == 0:
            continue

        # Extract CDS embeddings and pool
        cds_emb = nt_emb[batch_idx, cds_start:cds_start + cds_trimmed]  # [cds_trimmed, D]
        cds_emb = cds_emb.view(n_codons, 3, d).mean(dim=1)  # [n_codons, D]

        # Upsample back to nt level (repeat each codon 3 times)
        upsampled = cds_emb.repeat_interleave(3, dim=0)  # [cds_trimmed, D]
        out[batch_idx, cds_start:cds_start + cds_trimmed] = upsampled

    return out


def upsample_codon_to_nt_cds_anchored(
    codon_emb: Tensor,
    region_ids: Tensor,
    cds_start_offset: int = 0,
) -> Tensor:
    """Expand codon-level embeddings to nt resolution, CDS-anchored.

    Corrected version of ``upsample_codon_to_nt`` that places codon
    embeddings at the correct CDS positions instead of starting from 0.

    Parameters
    ----------
    codon_emb : Tensor[B, n_codons, D]
        Codon-level features.
    region_ids : Tensor[B, L]
        Region ids (used to find CDS start).
    cds_start_offset : int
        Additional offset from the first REGION_CDS position (default 0).

    Returns
    -------
    Tensor[B, L, D]
    """
    b, n_codons, d = codon_emb.shape
    length = region_ids.shape[1]
    device = codon_emb.device
    out = torch.zeros((b, length, d), device=device, dtype=codon_emb.dtype)

    for batch_idx in range(b):
        rids = region_ids[batch_idx]
        cds_mask = rids == REGION_CDS
        if not cds_mask.any():
            continue

        cds_start = int(cds_mask.nonzero()[0].item()) + cds_start_offset
        # Upsample: each codon -> 3 nt
        upsampled = codon_emb[batch_idx].repeat_interleave(3, dim=0)  # [3*n_codons, D]
        end = min(cds_start + upsampled.shape[0], length)
        out[batch_idx, cds_start:end] = upsampled[:end - cds_start]

    return out


__all__ = [
    "pool_to_codon_cds_anchored",
    "upsample_codon_to_nt_cds_anchored",
]
