"""P3-01: legality rules for local-edit generation (p3_task_v2 + motif_policy_v1).

Implements the frozen contract's hard constraints at DATA-GENERATION time
(legality is guaranteed by construction, never by reward penalty):

hard_constraints (p3_task_v2):
    identical_protein, identical_transcript_length, preserved_start_codon,
    preserved_stop_codon, valid_reading_frame, motif_policy_v1_hard_forbidden

motif_policy_v1 hard_forbidden tier (action-space exclusion):
    creates_upstream_in_frame_start_codon
    creates_premature_stop_codon
    creates_cryptic_splice_donor_or_acceptor
    creates_ivt_blocking_homopolymer_ge_6nt

guarded_risk tier is TRACKED (recorded in ``motif_flags``), never auto-illegal.

Deterministic splice proxy (documented; full MaxEnt scoring deferred):
    donor    : regex [AC]AGGU[AG]AGU  (canonical MAG|GTRAGU 5' donor, 9 nt)
    acceptor : regex [CU]{10,}AG      (polypyrimidine tract >=10 + AG)

All sequences here use the RNA alphabet (A, C, G, U).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

BASES: Tuple[str, ...] = ("A", "C", "G", "U")
STOP_CODONS: frozenset = frozenset({"UAA", "UAG", "UGA"})

# ---------------------------------------------------------------------------
# Standard genetic code (RNA codons)
# ---------------------------------------------------------------------------

_CODON_AA: Dict[str, str] = {
    "UUU": "F", "UUC": "F", "UUA": "L", "UUG": "L",
    "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
    "AUU": "I", "AUC": "I", "AUA": "I", "AUG": "M",
    "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V",
    "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S",
    "CCU": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCU": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "UAU": "Y", "UAC": "Y", "UAA": "*", "UAG": "*",
    "UGU": "C", "UGC": "C", "UGA": "*", "UGG": "W",
    "CAU": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAU": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAU": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "AGU": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGU": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# Synonymous codon alternatives (excluding the codon itself), deterministic order.
SYNONYMOUS_CODONS: Dict[str, Tuple[str, ...]] = {}
for _codon, _aa in sorted(_CODON_AA.items()):
    if _aa == "*":
        continue
    SYNONYMOUS_CODONS[_codon] = tuple(
        c for c, a in sorted(_CODON_AA.items()) if a == _aa and c != _codon
    )


def normalize_rna(seq: str) -> str:
    """Uppercase RNA alphabet (DNA input tolerated: T -> U)."""
    return seq.upper().replace("T", "U")


def is_valid_rna(seq: str) -> bool:
    return all(c in BASES for c in seq)


def translate(cds: str) -> str:
    """Translate RNA CDS to protein ('*' for stop). Trailing partial codon dropped."""
    seq = normalize_rna(cds)
    trim = len(seq) - (len(seq) % 3)
    return "".join(_CODON_AA.get(seq[i:i + 3], "X") for i in range(0, trim, 3))


def protein_identical(cds_a: str, cds_b: str) -> bool:
    """True iff both CDS translate to the identical protein."""
    return translate(cds_a) == translate(cds_b)


def is_valid_cds(cds: str) -> bool:
    """Valid CDS: starts AUG, ends with stop, len % 3 == 0, no internal stop,
    RNA alphabet, min length 9 (AUG + 1 sense codon + stop)."""
    seq = normalize_rna(cds)
    if len(seq) < 9 or len(seq) % 3 != 0:
        return False
    if not is_valid_rna(seq):
        return False
    if seq[:3] != "AUG":
        return False
    if seq[-3:] not in STOP_CODONS:
        return False
    protein = translate(seq)
    if "*" in protein[:-1]:
        return False
    return True


# ---------------------------------------------------------------------------
# Motif policy v1 — hard_forbidden tier
# ---------------------------------------------------------------------------

_DONOR_RE = re.compile(r"[AC]AGGU[AG]AGU")
_ACCEPTOR_RE = re.compile(r"[CU]{10,}AG")
_HOMOPOLYMER_RE = re.compile(r"(A{6,}|C{6,}|G{6,}|U{6,})")


def has_homopolymer_ge6(seq: str) -> bool:
    return bool(_HOMOPOLYMER_RE.search(seq))


def creates_homopolymer_ge6(old: str, new: str) -> bool:
    return (not has_homopolymer_ge6(old)) and has_homopolymer_ge6(new)


def has_cryptic_splice(seq: str) -> bool:
    return bool(_DONOR_RE.search(seq) or _ACCEPTOR_RE.search(seq))


def creates_cryptic_splice(old: str, new: str) -> bool:
    return (not has_cryptic_splice(old)) and has_cryptic_splice(new)


def _in_frame_positions_five_utr(len5: int) -> range:
    """5'UTR offsets that are in-frame with the CDS AUG (frame 0 at CDS start)."""
    # CDS AUG sits at offset len5. Position q is in-frame iff (len5 - q) % 3 == 0.
    start = len5 % 3
    return range(start, len5, 3)


def _has_stop_between(seq: str, q: int, len5: int) -> bool:
    """True if an in-frame stop codon sits strictly between offset q and the CDS."""
    for p in range(q + 3, len5 - 2, 3):
        if seq[p:p + 3] in STOP_CODONS:
            return True
    return False


def has_upstream_in_frame_start_codon(five_utr: str) -> bool:
    """True if the 5'UTR contains an AUG in-frame with the CDS start codon and
    with NO intervening in-frame stop codon (would extend the ORF upstream)."""
    seq = normalize_rna(five_utr)
    len5 = len(seq)
    for q in _in_frame_positions_five_utr(len5):
        if seq[q:q + 3] == "AUG" and not _has_stop_between(seq, q, len5):
            return True
    return False


def creates_upstream_in_frame_start_codon(old_five_utr: str, new_five_utr: str) -> bool:
    return (not has_upstream_in_frame_start_codon(old_five_utr)) and \
        has_upstream_in_frame_start_codon(new_five_utr)


def creates_premature_stop_codon(old_cds: str, new_cds: str) -> bool:
    """CDS-scoped check: new in-frame stop codon before the natural stop.

    Synonymous-only generation makes this impossible by construction; the
    explicit check is kept as a fail-safe (and for validating external edits).
    """
    old_prot = translate(old_cds)
    new_prot = translate(new_cds)
    if len(old_prot) != len(new_prot):
        return True  # length/frame changed -> treated as premature-stop risk
    for i, (a, b) in enumerate(zip(old_prot, new_prot)):
        if a != "*" and b == "*":
            return True
    return False


def motif_policy_v1_hard_forbidden_triggered(
    region: str,
    old_region_seq: str,
    new_region_seq: str,
    *,
    full_cds_old: Optional[str] = None,
    full_cds_new: Optional[str] = None,
) -> List[str]:
    """Return the list of triggered hard_forbidden rules (empty == legal).

    ``region`` is one of the benchmark EDIT_REGIONS. For CDS-scoped regions,
    pass the full CDS (old/new) so the premature-stop check sees the whole ORF.
    """
    triggered: List[str] = []
    if region == "five_utr":
        if creates_upstream_in_frame_start_codon(old_region_seq, new_region_seq):
            triggered.append("creates_upstream_in_frame_start_codon")
    if full_cds_old is not None and full_cds_new is not None:
        if creates_premature_stop_codon(full_cds_old, full_cds_new):
            triggered.append("creates_premature_stop_codon")
    if creates_cryptic_splice(old_region_seq, new_region_seq):
        triggered.append("creates_cryptic_splice_donor_or_acceptor")
    if creates_homopolymer_ge6(old_region_seq, new_region_seq):
        triggered.append("creates_ivt_blocking_homopolymer_ge_6nt")
    return triggered


def motif_policy_v1_guarded_risk_flags(old_seq: str, new_seq: str) -> List[str]:
    """Tracked (never auto-illegal) guarded_risk tier flags.

    Implemented as deterministic proxies, documented in p3_01_data_limitations:
      - m6a_motif_gain_or_loss          : DRACH motif count changed
      - homopolymer_run_4_to_5nt        : (soft_objective; tracked alongside)
    """
    flags: List[str] = []
    drach = re.compile(r"[AGU][GA]AC[ACU]")
    if len(drach.findall(old_seq)) != len(drach.findall(new_seq)):
        flags.append("m6a_motif_gain_or_loss")
    run45 = re.compile(r"(A{4,5}|C{4,5}|G{4,5}|U{4,5})")
    if len(run45.findall(old_seq)) != len(run45.findall(new_seq)):
        flags.append("homopolymer_run_4_to_5nt")
    return flags


# ---------------------------------------------------------------------------
# Legal single-edit enumeration
# ---------------------------------------------------------------------------

def legal_five_utr_single_subs(
    five_utr: str,
    pos_range: Optional[Tuple[int, int]] = None,
) -> List[Tuple[int, str, str]]:
    """All legal (pos, ref, alt) single substitutions in a 5'UTR window.

    Legality (motif_policy_v1 hard_forbidden tier, action-space exclusion):
      - no new upstream in-frame start codon
      - no new cryptic splice donor/acceptor
      - no new homopolymer run >= 6 nt
    """
    seq = normalize_rna(five_utr)
    lo = 0 if pos_range is None else max(0, pos_range[0])
    hi = len(seq) if pos_range is None else min(len(seq), pos_range[1])
    out: List[Tuple[int, str, str]] = []
    for pos in range(lo, hi):
        ref = seq[pos]
        for alt in BASES:
            if alt == ref:
                continue
            cand = seq[:pos] + alt + seq[pos + 1:]
            if creates_upstream_in_frame_start_codon(seq, cand):
                continue
            if creates_cryptic_splice(seq, cand):
                continue
            if creates_homopolymer_ge6(seq, cand):
                continue
            out.append((pos, ref, alt))
    return out


def legal_cds_synonymous_single_subs(
    cds: str,
    codon_range: Tuple[int, int],
    *,
    check_region_seq: Optional[str] = None,
) -> List[Tuple[int, str, str, int, str]]:
    """All legal synonymous single substitutions within a CDS codon window.

    Returns (nt_pos, ref, alt, codon_index, new_codon) tuples where nt_pos is
    the 0-based offset WITHIN THE FULL CDS. Legality:
      - synonymous by construction (same amino acid; protein identical)
      - no new premature stop (fail-safe via full-CDS translation check)
      - no new cryptic splice donor/acceptor (checked within ``check_region_seq``
        scope when provided, else within the full CDS)
      - no new homopolymer run >= 6 nt
    Start codon (codon 0) and stop codon are never touched (codon_range callers
    must exclude them; enforced here as a fail-safe).
    """
    seq = normalize_rna(cds)
    n_codons = len(seq) // 3
    lo = max(1, codon_range[0])           # never edit start codon
    hi = min(n_codons - 1, codon_range[1])  # never edit stop codon
    scope = normalize_rna(check_region_seq) if check_region_seq is not None else seq
    out: List[Tuple[int, str, str, int, str]] = []
    for ci in range(lo, hi):
        codon = seq[3 * ci:3 * ci + 3]
        if codon not in SYNONYMOUS_CODONS:
            continue
        for new_codon in SYNONYMOUS_CODONS[codon]:
            # find the single nt difference (only single-substitution synonyms)
            diffs = [(k, codon[k], new_codon[k]) for k in range(3) if codon[k] != new_codon[k]]
            if len(diffs) != 1:
                continue
            k, ref, alt = diffs[0]
            nt_pos = 3 * ci + k
            cand_cds = seq[:nt_pos] + alt + seq[nt_pos + 1:]
            if creates_premature_stop_codon(seq, cand_cds):
                continue
            # Motif checks inside the region scope. Map nt_pos into scope
            # coordinates: callers pass scope = cds[3*window_lo:3*window_hi].
            if check_region_seq is not None:
                off = nt_pos - 3 * codon_range[0]
                if not (0 <= off < len(scope)):
                    continue
                cand_scope = scope[:off] + alt + scope[off + 1:]
            else:
                cand_scope = cand_cds
            if creates_cryptic_splice(scope, cand_scope):
                continue
            if creates_homopolymer_ge6(scope, cand_scope):
                continue
            out.append((nt_pos, ref, alt, ci, new_codon))
    return out
