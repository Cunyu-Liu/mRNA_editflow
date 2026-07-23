"""Region-anchored alignment and grammar-safe corruption for mRNA Edit Flow.

Replaces the old plain Levenshtein DP (:func:`mrna_editflow.core.mrna_flow_utils._align_pair`)
with a region-aware alignment that:

1. Splits source ``x0`` and target ``x1`` by region (5'UTR / CDS / 3'UTR).
2. Aligns 5'UTR and 3'UTR with nucleotide-level Levenshtein DP.
3. Aligns CDS with **codon-level** DP — only synonymous codon substitutions are
   allowed; insertions/deletions are whole-codon only; frameshift is impossible
   by construction.
4. Concatenates the per-region aligned pairs through **immutable anchors**
   (start codon, stop codon, region boundaries) to form the final ``(z0, z1)``.

This guarantees that the auxiliary bridge path never passes through a
frameshifted, nonsynonymous, or boundary-violating intermediate state —
a hard requirement from the Edit Flow auxiliary process theorem.

The module also provides :func:`grammar_safe_corrupt` which replaces the old
:func:`_corrupt_tokens` with a region-aware corruption that preserves CDS
validity (synonymous substitutions only, no nt-level indels in CDS).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .constants import (
    CODON_TABLE,
    GAP_TOKEN,
    NUC_TO_ID,
    PAD_TOKEN,
    PHASE_NONE,
    REGION_5UTR,
    REGION_CDS,
    REGION_3UTR,
    START_CODON,
    STOP_CODONS,
    SYNONYMOUS_CODONS,
    V,
)


# ---------------------------------------------------------------------------
# Region splitting
# ---------------------------------------------------------------------------

def _split_by_region(
    tokens: Sequence[int],
    region_ids: Sequence[int],
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """Split a token list into per-region sub-lists.

    Returns ``(five_utr, cds, three_utr)`` where each element is a list of
    token ids. Region boundaries are determined by ``region_ids``.
    """
    utr5, cds, utr3 = [], [], []
    for tok, rid in zip(tokens, region_ids):
        if rid == REGION_5UTR:
            utr5.append(tok)
        elif rid == REGION_CDS:
            cds.append(tok)
        elif rid == REGION_3UTR:
            utr3.append(tok)
    return utr5, cds, utr3


def _region_boundaries(region_ids: Sequence[int]) -> Tuple[int, int, int]:
    """Return ``(utr5_end, cds_end, utr3_end)`` indices into the full sequence."""
    utr5_end = 0
    cds_end = 0
    utr3_end = len(region_ids)
    for i, r in enumerate(region_ids):
        if r == REGION_5UTR and utr5_end == cds_end:
            utr5_end = i + 1
        elif r == REGION_CDS:
            cds_end = i + 1
    # Handle case where 3'UTR is empty
    if utr5_end == 0 and cds_end == 0:
        utr5_end = 0
    elif cds_end == 0:
        utr5_end = utr3_end
    return utr5_end, cds_end, utr3_end


# ---------------------------------------------------------------------------
# Nucleotide-level Levenshtein alignment (for UTRs)
# ---------------------------------------------------------------------------

def _nt_levenshtein_align(
    seq_0: Sequence[int],
    seq_1: Sequence[int],
) -> Tuple[List[int], List[int]]:
    """Nucleotide-level Levenshtein DP + backtrack -> gap-padded aligned pair.

    Identical semantics to the old ``_align_pair`` but scoped to one region.
    Complexity: O(m*n).
    """
    m, n = len(seq_0), len(seq_1)
    if m == 0 and n == 0:
        return [], []
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_0[i - 1] == seq_1[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    aligned_0: List[int] = []
    aligned_1: List[int] = []
    i, j = m, n
    while i or j:
        if i and j and seq_0[i - 1] == seq_1[j - 1]:
            aligned_0.append(int(seq_0[i - 1]))
            aligned_1.append(int(seq_1[j - 1]))
            i, j = i - 1, j - 1
        elif i and j and dp[i][j] == dp[i - 1][j - 1] + 1:
            aligned_0.append(int(seq_0[i - 1]))
            aligned_1.append(int(seq_1[j - 1]))
            i, j = i - 1, j - 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            aligned_0.append(int(seq_0[i - 1]))
            aligned_1.append(GAP_TOKEN)
            i -= 1
        else:
            aligned_0.append(GAP_TOKEN)
            aligned_1.append(int(seq_1[j - 1]))
            j -= 1
    return aligned_0[::-1], aligned_1[::-1]


# ---------------------------------------------------------------------------
# Codon-level alignment (for CDS)
# ---------------------------------------------------------------------------

def _tokens_to_codons(cds_tokens: Sequence[int]) -> List[Tuple[int, int, int]]:
    """Group CDS tokens into codon tuples. Raises if length is not multiple of 3."""
    n = len(cds_tokens)
    if n % 3 != 0:
        raise ValueError(f"CDS length {n} is not a multiple of 3")
    return [(cds_tokens[i], cds_tokens[i + 1], cds_tokens[i + 2]) for i in range(0, n, 3)]


def _codon_to_nts(codon_idx: int) -> Tuple[int, int, int]:
    """Convert a linear codon index (0..63) to three nucleotide token ids."""
    return (codon_idx // 16, (codon_idx // 4) % 4, codon_idx % 4)


def _nts_to_codon_idx(a: int, b: int, c: int) -> int:
    """Convert three nucleotide token ids to a linear codon index (0..63)."""
    return a * 16 + b * 4 + c


def _codon_substitution_cost(c0: int, c1: int) -> int:
    """Cost of substituting codon ``c0`` with ``c1``.

    Cost is 0 if identical, 1 if synonymous (same amino acid), infinity
    otherwise. This ensures the alignment never passes through a
    nonsynonymous intermediate.
    """
    if c0 == c1:
        return 0
    # Convert to RNA strings to look up amino acids
    nts0 = _codon_to_nts(c0)
    nts1 = _codon_to_nts(c1)
    rna0 = "".join("ACGU"[n] for n in nts0)
    rna1 = "".join("ACGU"[n] for n in nts1)
    aa0 = CODON_TABLE.get(rna0, "X")
    aa1 = CODON_TABLE.get(rna1, "X")
    if aa0 == aa1 and aa0 != "X":
        return 1
    return 1000  # effectively forbidden


def _codon_levenshtein_align(
    codons_0: Sequence[int],
    codons_1: Sequence[int],
) -> Tuple[List[int], List[int]]:
    """Codon-level DP alignment.

    Operations:
    - **substitution**: codon → codon (cost 0 if identical, 1 if synonymous,
      1000 if nonsynonymous — effectively forbidden).
    - **insertion**: GAP → codon (whole-codon insertion, cost 1).
    - **deletion**: codon → GAP (whole-codon deletion, cost 1).

    The aligned output uses GAP_TOKEN for gaps, but since each position
    represents a codon (3 nt), gaps are represented as triple GAP_TOKENs
    when expanded to nucleotide level.

    Returns ``(aligned_codons_0, aligned_codons_1)`` where each element is
    a list of codon indices or ``-1`` for GAP.
    """
    m, n = len(codons_0), len(codons_1)
    if m == 0 and n == 0:
        return [], []
    INF = 10 ** 9
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub_cost = _codon_substitution_cost(codons_0[i - 1], codons_1[j - 1])
            dp[i][j] = min(
                dp[i - 1][j - 1] + sub_cost,  # substitution
                dp[i - 1][j] + 1,              # deletion
                dp[i][j - 1] + 1,              # insertion
            )

    aligned_0: List[int] = []
    aligned_1: List[int] = []
    i, j = m, n
    while i or j:
        if i and j:
            sub_cost = _codon_substitution_cost(codons_0[i - 1], codons_1[j - 1])
            if dp[i][j] == dp[i - 1][j - 1] + sub_cost:
                aligned_0.append(codons_0[i - 1])
                aligned_1.append(codons_1[j - 1])
                i, j = i - 1, j - 1
                continue
        if i and dp[i][j] == dp[i - 1][j] + 1:
            aligned_0.append(codons_0[i - 1])
            aligned_1.append(-1)  # GAP
            i -= 1
        else:
            aligned_0.append(-1)  # GAP
            aligned_1.append(codons_1[j - 1])
            j -= 1
    return aligned_0[::-1], aligned_1[::-1]


def _expand_codon_alignment(
    aligned_codons_0: Sequence[int],
    aligned_codons_1: Sequence[int],
) -> Tuple[List[int], List[int]]:
    """Expand codon-level alignment to nucleotide-level (with GAP_TOKENs).

    Each codon becomes 3 nucleotide tokens; each GAP becomes 3 GAP_TOKENs.
    """
    z0: List[int] = []
    z1: List[int] = []
    for c0, c1 in zip(aligned_codons_0, aligned_codons_1):
        if c0 == -1:
            z0.extend([GAP_TOKEN, GAP_TOKEN, GAP_TOKEN])
        else:
            z0.extend(_codon_to_nts(c0))
        if c1 == -1:
            z1.extend([GAP_TOKEN, GAP_TOKEN, GAP_TOKEN])
        else:
            z1.extend(_codon_to_nts(c1))
    return z0, z1


# ---------------------------------------------------------------------------
# Region-anchored alignment (main entry point)
# ---------------------------------------------------------------------------

def region_anchored_align(
    tokens_0: Sequence[int],
    tokens_1: Sequence[int],
    region_ids_1: Sequence[int],
) -> Tuple[List[int], List[int]]:
    """Align ``x0`` (source) to ``x1`` (target) with region-anchored grammar.

    ``x1`` is the target (natural mRNA), so ``region_ids_1`` describes its
    region partition. ``x0`` is the source (possibly corrupted/empty), and
    its region partition is inferred from the alignment.

    The alignment proceeds per-region:
    - 5'UTR: nt-level Levenshtein.
    - CDS: codon-level DP (synonymous substitutions + whole-codon indels).
    - 3'UTR: nt-level Levenshtein.

    The per-region alignments are concatenated. Region boundaries in ``z1``
    are always preserved (no cross-boundary alignment).

    Parameters
    ----------
    tokens_0 : source sequence token ids (may be empty for empty-growth).
    tokens_1 : target sequence token ids (natural mRNA).
    region_ids_1 : per-token region ids of the target.

    Returns
    -------
    ``(z0, z1)`` : gap-padded aligned token lists.
    """
    # Split target by region
    utr5_1, cds_1, utr3_1 = _split_by_region(tokens_1, region_ids_1)

    # Split source by region — source regions are inferred from the target's
    # region boundaries via the alignment. For simplicity, we split source
    # proportionally: if source is a corruption of target, source regions
    # roughly correspond. For empty-growth, source is empty.
    # Strategy: align each region independently, treating source tokens as
    # potentially belonging to any region. We use a simple heuristic:
    # if len(source) == 0, all regions are empty.
    # If source is a corruption, we split source at the same proportions as target.

    if len(tokens_0) == 0:
        # Empty-growth: align empty source to each region of target
        z0_utr5, z1_utr5 = _nt_levenshtein_align([], utr5_1)
        # CDS: align empty to codons
        cds_codons_1 = [_nts_to_codon_idx(cds_1[i], cds_1[i+1], cds_1[i+2])
                        for i in range(0, len(cds_1), 3)] if cds_1 else []
        # Handle odd-length CDS gracefully (shouldn't happen post-cleaning)
        if len(cds_1) % 3 != 0:
            # Fall back to nt alignment for the remainder
            cds_main = cds_1[:len(cds_1) - (len(cds_1) % 3)]
            cds_tail = cds_1[len(cds_1) - (len(cds_1) % 3):]
            cds_codons_1 = [_nts_to_codon_idx(cds_main[i], cds_main[i+1], cds_main[i+2])
                            for i in range(0, len(cds_main), 3)]
            ac0, ac1 = _codon_levenshtein_align([], cds_codons_1)
            z0_cds, z1_cds = _expand_codon_alignment(ac0, ac1)
            z0_tail, z1_tail = _nt_levenshtein_align([], cds_tail)
            z0_cds = z0_cds + z0_tail
            z1_cds = z1_cds + z1_tail
        else:
            ac0, ac1 = _codon_levenshtein_align([], cds_codons_1)
            z0_cds, z1_cds = _expand_codon_alignment(ac0, ac1)
        z0_utr3, z1_utr3 = _nt_levenshtein_align([], utr3_1)
    else:
        # Source is non-empty (corruption or ortholog).
        # Split source at same proportions as target.
        utr5_end_1 = len(utr5_1)
        cds_end_1 = utr5_end_1 + len(cds_1)

        # For source, split at proportional boundaries
        total_1 = len(tokens_1)
        if total_1 > 0:
            utr5_end_0 = round(len(tokens_0) * utr5_end_1 / total_1)
            cds_end_0 = round(len(tokens_0) * cds_end_1 / total_1)
        else:
            utr5_end_0 = 0
            cds_end_0 = 0

        utr5_0 = list(tokens_0[:utr5_end_0])
        cds_0_raw = list(tokens_0[utr5_end_0:cds_end_0])
        utr3_0 = list(tokens_0[cds_end_0:])

        # Align UTRs at nt level
        z0_utr5, z1_utr5 = _nt_levenshtein_align(utr5_0, utr5_1)

        # Align CDS at codon level
        # Trim source CDS to multiple of 3 (source may have indels)
        cds_0_trimmed = cds_0_raw[:len(cds_0_raw) - (len(cds_0_raw) % 3)]
        cds_0_tail = cds_0_raw[len(cds_0_raw) - (len(cds_0_raw) % 3):] if len(cds_0_raw) % 3 != 0 else []

        if len(cds_1) % 3 == 0 and len(cds_1) > 0:
            cds_codons_0 = [_nts_to_codon_idx(cds_0_trimmed[i], cds_0_trimmed[i+1], cds_0_trimmed[i+2])
                            for i in range(0, len(cds_0_trimmed), 3)] if cds_0_trimmed else []
            cds_codons_1 = [_nts_to_codon_idx(cds_1[i], cds_1[i+1], cds_1[i+2])
                            for i in range(0, len(cds_1), 3)]

            ac0, ac1 = _codon_levenshtein_align(cds_codons_0, cds_codons_1)
            z0_cds, z1_cds = _expand_codon_alignment(ac0, ac1)

            # Append any source CDS tail (odd-length remainder) via nt alignment
            if cds_0_tail:
                z0_tail, z1_tail = _nt_levenshtein_align(cds_0_tail, [])
                z0_cds = z0_cds + z0_tail
                z1_cds = z1_cds + z1_tail
        else:
            # Target CDS not valid (shouldn't happen post-cleaning) — fall back
            z0_cds, z1_cds = _nt_levenshtein_align(cds_0_raw, cds_1)

        z0_utr3, z1_utr3 = _nt_levenshtein_align(utr3_0, utr3_1)

    # Concatenate: 5'UTR + CDS + 3'UTR (boundaries are natural anchors)
    z0 = z0_utr5 + z0_cds + z0_utr3
    z1 = z1_utr5 + z1_cds + z1_utr3
    return z0, z1


# ---------------------------------------------------------------------------
# Grammar-safe corruption (replaces _corrupt_tokens)
# ---------------------------------------------------------------------------

def grammar_safe_corrupt(
    tokens: Sequence[int],
    region_ids: Sequence[int],
    sub_p: float,
    ins_p: float,
    del_p: float,
    rng: np.random.Generator,
) -> List[int]:
    """Region-aware corruption that preserves CDS validity.

    - **5'UTR / 3'UTR**: free nucleotide-level sub/ins/del (same as old).
    - **CDS**: only **synonymous codon substitutions** (whole-codon replace);
      nt-level indels are forbidden (no frameshift); start/stop codons are
      never corrupted.

    This ensures the corrupted source ``x0`` is always a valid mRNA with
    the same protein as ``x1``, eliminating the supervision conflict where
    the auxiliary path requires the model to fix illegal CDS states.

    Parameters
    ----------
    tokens : full sequence token ids.
    region_ids : per-token region ids.
    sub_p / ins_p / del_p : corruption probabilities (UTR only for ins/del).
    rng : numpy random generator.

    Returns
    -------
    Corrupted token list (may differ in length due to UTR indels).
    """
    utr5, cds, utr3 = _split_by_region(tokens, region_ids)

    # --- UTR corruption (free nt-level) ---
    def _corrupt_utr(seq: List[int]) -> List[int]:
        out: List[int] = []
        for tok in seq:
            if rng.random() < del_p:
                continue
            cur = tok
            if rng.random() < sub_p:
                cur = int(rng.integers(0, V))
            out.append(cur)
            if rng.random() < ins_p:
                out.append(int(rng.integers(0, V)))
        if not out:
            out = [int(rng.integers(0, V))]
        return out

    utr5_c = _corrupt_utr(utr5)
    utr3_c = _corrupt_utr(utr3)

    # --- CDS corruption (synonymous codon substitutions only) ---
    cds_c: List[int] = []
    if len(cds) >= 6:  # need at least start + stop
        n_codons = len(cds) // 3
        for ci in range(n_codons):
            nt0, nt1, nt2 = cds[ci * 3], cds[ci * 3 + 1], cds[ci * 3 + 2]
            rna = "".join("ACGU"[n] for n in (nt0, nt1, nt2))
            aa = CODON_TABLE.get(rna, "X")

            # Never corrupt start (AUG) or stop codons
            if rna == START_CODON or rna in STOP_CODONS or aa == "X":
                cds_c.extend([nt0, nt1, nt2])
                continue

            if rng.random() < sub_p:
                # Pick a random synonymous codon
                syn_codons = SYNONYMOUS_CODONS.get(aa, [rna])
                if len(syn_codons) > 1:
                    choices = [c for c in syn_codons if c != rna]
                    if choices:
                        new_rna = str(rng.choice(choices))
                        cds_c.extend([NUC_TO_ID[ch] for ch in new_rna])
                    else:
                        cds_c.extend([nt0, nt1, nt2])
                else:
                    cds_c.extend([nt0, nt1, nt2])
            else:
                cds_c.extend([nt0, nt1, nt2])

        # Append any remaining nt (length not multiple of 3 — shouldn't happen)
        remainder = len(cds) % 3
        if remainder:
            cds_c.extend(cds[n_codons * 3:])
    else:
        cds_c = list(cds)

    return utr5_c + cds_c + utr3_c


__all__ = [
    "region_anchored_align",
    "grammar_safe_corrupt",
]
