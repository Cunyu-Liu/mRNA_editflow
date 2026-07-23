"""P3-01: controlled-neighborhood generation (5 edit types x 5 regions).

For each eligible source we generate, per region scope:

    all_legal_single          exhaustive legal single edits (capped only for
                              the cds_remaining census scope, documented)
    random_double             seeded random pairs of legal singles (distinct pos)
    structure_guided_double   pairs maximizing local-structure disruption proxy
    topranked_double          pairs of top-|delta| singles under the region's
                              internal ranking feature (CNN-50mer proxy for the
                              5'UTR proxy window; deterministic CAI / structure
                              features for unlabeled scopes)
    matched_negative_single   singles with near-zero predicted |delta|

Region scopes (p3_task_v2 cds_scope_analysis + active task):
    five_utr        task_a_active        (5'UTR substitution)
    cds_first30     task_b_frozen_fallback (codons 1..29; start codon never edited)
    cds_first50     task_b_frozen_fallback (codons 1..49)
    cds_remaining   task_b_frozen_fallback (codons 50..end-1)
    joint_5utr_cds  task_c_locked_extension (one 5'UTR edit + one synonymous CDS edit)

All sequences use the RNA alphabet. Generation is deterministic given the seed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from mrna_editflow.data.p3_legality import (
    legal_cds_synonymous_single_subs,
    legal_five_utr_single_subs,
    normalize_rna,
)
from mrna_editflow.data.p3_local_edit_schema import Edit

# Turner-2004-style nearest-neighbor dinucleotide energies (kcal/mol), same
# simplified set used by eval/multi_region_oracle.py (deterministic feature).
NN_ENERGY: Dict[str, float] = {
    "AA": -1.0, "AC": -2.4, "AG": -2.1, "AU": -1.2,
    "CA": -2.1, "CC": -3.3, "CG": -2.4, "CU": -2.1,
    "GA": -2.2, "GC": -3.6, "GG": -3.3, "GU": -2.5,
    "UA": -1.3, "UC": -2.4, "UG": -2.1, "UU": -1.0,
}

REGION_FIVE_UTR = "five_utr"
REGION_CDS_FIRST30 = "cds_first30"
REGION_CDS_FIRST50 = "cds_first50"
REGION_CDS_REMAINING = "cds_remaining"
REGION_JOINT = "joint_5utr_cds"

CDS_REGION_CODON_WINDOWS: Dict[str, Tuple[int, int]] = {
    REGION_CDS_FIRST30: (0, 30),
    REGION_CDS_FIRST50: (0, 50),
    # cds_remaining upper bound is source-dependent -> handled by caller.
}

_CDS_REGIONS: frozenset = frozenset(
    {REGION_CDS_FIRST30, REGION_CDS_FIRST50, REGION_CDS_REMAINING}
)


@dataclass(frozen=True)
class NeighborhoodConfig:
    n_random_double: int = 32
    n_structure_double: int = 32
    n_topranked_double: int = 16
    n_matched_negative: int = 16
    topranked_pool: int = 8      # C(8,2)=28 candidate pairs, take first 16
    structure_pool: int = 8
    seed: int = 20260723
    cds_remaining_max_singles: int = 400  # documented census cap
    structure_window: int = 6


def apply_edits(region_seq: str, edits: Sequence[Edit]) -> str:
    """Apply substitution edits to a region-scope sequence."""
    seq = list(region_seq)
    for e in edits:
        assert seq[e.pos] == e.ref, f"ref mismatch at {e.pos}: {seq[e.pos]} != {e.ref}"
        seq[e.pos] = e.alt
    return "".join(seq)


def local_nn_energy(seq: str) -> float:
    return sum(NN_ENERGY.get(seq[i:i + 2], 0.0) for i in range(len(seq) - 1))


def structure_disruption(region_seq: str, edit: Edit, window: int = 6) -> float:
    """|delta local NN-energy| in a +/-window neighborhood around the edit."""
    lo = max(0, edit.pos - window)
    hi = min(len(region_seq), edit.pos + window + 1)
    before = local_nn_energy(region_seq[lo:hi])
    after = local_nn_energy(apply_edits(region_seq, [edit])[lo:hi])
    return abs(after - before)


def _distinct_position_pairs(items: Sequence[Edit]) -> List[Tuple[int, int]]:
    """Deterministic (i, j) index pairs, i < j, with distinct edit positions.

    CDS-scoped pairs must additionally lie in DIFFERENT codons: two
    individually-synonymous substitutions inside one codon can combine into a
    nonsynonymous codon (e.g. CUG->UUG and CUG->CUU are both Leu, but the
    combined CUG->UUU is Phe). Protein identity is thus guaranteed by
    construction; the benchmark builder re-verifies it fail-closed.
    """
    pairs: List[Tuple[int, int]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if a.pos == b.pos:
                continue
            if (
                a.region in _CDS_REGIONS
                and b.region in _CDS_REGIONS
                and a.pos // 3 == b.pos // 3
            ):
                continue
            pairs.append((i, j))
    return pairs


def assemble_doubles(
    singles: Sequence[Edit],
    toprank_scores: Dict[Edit, float],
    region_seq: str,
    cfg: NeighborhoodConfig,
    rng: random.Random,
) -> Dict[str, List[Tuple[Edit, ...]]]:
    """Build the 4 non-exhaustive edit types from scored legal singles.

    ``toprank_scores`` maps each single edit to its |delta| under the region's
    internal ranking feature (already absolute-valued by the caller).
    """
    out: Dict[str, List[Tuple[Edit, ...]]] = {
        "random_double": [],
        "structure_guided_double": [],
        "topranked_double": [],
        "matched_negative_single": [],
    }
    if not singles:
        return out

    # ---- random_double: seeded sample of distinct-position pairs ----
    all_pairs = _distinct_position_pairs(singles)
    pairs = list(all_pairs)
    rng.shuffle(pairs)
    for i, j in pairs[: cfg.n_random_double]:
        out["random_double"].append((singles[i], singles[j]))

    # ---- structure_guided_double: top-pool by local-structure disruption ----
    struct_scored = sorted(
        singles,
        key=lambda e: (-structure_disruption(region_seq, e, cfg.structure_window),
                       e.pos, e.alt),
    )
    spool = struct_scored[: cfg.structure_pool]
    for i, j in _distinct_position_pairs(spool)[: cfg.n_structure_double]:
        out["structure_guided_double"].append((spool[i], spool[j]))

    # ---- topranked_double: top-pool by |delta| ranking feature ----
    top_scored = sorted(
        singles,
        key=lambda e: (-toprank_scores.get(e, 0.0), e.pos, e.alt),
    )
    tpool = top_scored[: cfg.topranked_pool]
    for i, j in _distinct_position_pairs(tpool)[: cfg.n_topranked_double]:
        out["topranked_double"].append((tpool[i], tpool[j]))

    # ---- matched_negative_single: smallest |delta| ----
    for e in sorted(singles, key=lambda e: (toprank_scores.get(e, 0.0), e.pos, e.alt))[
        : cfg.n_matched_negative
    ]:
        out["matched_negative_single"].append((e,))

    return out


# ---------------------------------------------------------------------------
# Region-scoped single-edit enumeration (legal per p3_task_v2)
# ---------------------------------------------------------------------------

def five_utr_singles(five_utr: str, pos_range: Optional[Tuple[int, int]] = None) -> List[Edit]:
    seq = normalize_rna(five_utr)
    return [
        Edit(region=REGION_FIVE_UTR, pos=p, ref=r, alt=a)
        for (p, r, a) in legal_five_utr_single_subs(seq, pos_range)
    ]


def cds_singles(
    cds: str,
    region: str,
    codon_window: Tuple[int, int],
    max_singles: Optional[int] = None,
) -> Tuple[List[Edit], str]:
    """Legal synonymous singles for a CDS scope.

    Returns (edits, region_scope_sequence). Edit positions are relative to the
    RETURNED scope sequence (cds[3*lo : 3*hi_capped]), not the full CDS, so that
    benchmark source_sequence/candidate_sequence stay scope-sized.
    """
    seq = normalize_rna(cds)
    n_codons = len(seq) // 3
    lo = max(1, codon_window[0])              # never touch start codon
    hi = min(n_codons - 1, codon_window[1])   # never touch stop codon
    scope = seq[3 * lo: 3 * hi]
    raw = legal_cds_synonymous_single_subs(
        seq, (lo, hi), check_region_seq=scope)
    edits = [
        Edit(region=region, pos=nt_pos - 3 * lo, ref=ref, alt=alt)
        for (nt_pos, ref, alt, _ci, _nc) in raw
    ]
    if max_singles is not None and len(edits) > max_singles:
        # Deterministic census cap (documented): uniform stride subsample.
        stride = len(edits) / max_singles
        edits = [edits[int(i * stride)] for i in range(max_singles)]
    return edits, scope


def joint_doubles(
    five_utr_singles_seq: Sequence[Edit],
    cds_singles_seq: Sequence[Edit],
    five_utr: str,
    cds_scope_seq: str,
    cfg: NeighborhoodConfig,
    rng: random.Random,
) -> Dict[str, List[Tuple[Edit, ...]]]:
    """Joint 5'UTR + CDS doubles (task_c_locked_extension), unlabeled.

    One 5'UTR edit (positions relative to the 5'UTR) + one synonymous CDS edit
    (positions relative to the CDS scope). The caller stores a concatenated
    joint scope sequence (five_utr + cds_scope) and shifts CDS positions by
    len(five_utr) — handled here.
    """
    shift = len(normalize_rna(five_utr))
    shifted_cds = [
        Edit(region=REGION_JOINT, pos=e.pos + shift, ref=e.ref, alt=e.alt)
        for e in cds_singles_seq
    ]
    base_utr = [
        Edit(region=REGION_JOINT, pos=e.pos, ref=e.ref, alt=e.alt)
        for e in five_utr_singles_seq
    ]
    out: Dict[str, List[Tuple[Edit, ...]]] = {
        "random_double": [],
        "structure_guided_double": [],
        "topranked_double": [],
        "matched_negative_single": [],
    }
    if not base_utr or not shifted_cds:
        return out
    pairs = [(u, c) for u in base_utr for c in shifted_cds]
    rng.shuffle(pairs)
    for u, c in pairs[: cfg.n_random_double]:
        out["random_double"].append((u, c))
    # structure-guided: top-disruption 5'UTR edit x top-disruption CDS edit
    joint_scope = normalize_rna(five_utr) + cds_scope_seq
    u_top = sorted(base_utr, key=lambda e: (-structure_disruption(joint_scope, e, cfg.structure_window), e.pos, e.alt))[: cfg.structure_pool]
    c_top = sorted(shifted_cds, key=lambda e: (-structure_disruption(joint_scope, e, cfg.structure_window), e.pos, e.alt))[: cfg.structure_pool]
    for u, c in list(zip(u_top, c_top))[: cfg.n_structure_double]:
        out["structure_guided_double"].append((u, c))
    return out
