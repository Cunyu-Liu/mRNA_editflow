"""X0-X CDS pure-development scaffolding: synonymous-codon state machine.

Phase X0-X (3'UTR & CDS transfer) — PURE DEVELOPMENT PREPARATION ONLY.  This
module does NOT touch the frozen 5' primary model, does NOT access sealed
labels, and does NOT trigger the formal X0-X gate (which the contract gates on
frozen 5' primary model + threshold + sealed results).

It implements the CDS-side design invariants required by §16 (X0-X):

* **state = synonymous codon** : a CDS is represented as an ordered tuple of
  codons (frame-locked triplets from an ATG start).
* **atomic codon substitution** : the only CDS edit primitive is replacing ONE
  codon with a SYNONYMOUS codon (same amino acid).
* **protein identity / frame / start / stop hard invariant** : any accepted
  edit must keep the translated protein identical, keep the frame aligned to
  the start codon, and keep the first (start) and last (stop) codon unchanged.
  These are enforced by construction, not by reward penalty.
* **family/listwise metric** : CDS evaluation uses protein-family listwise
  ranking (a family = one protein; candidates are synonymous variants ranked by
  a measured/ predicted score).
* **protein-family split** : CDS train/test splits are protein-family-disjoint.

The module is pure Python (no torch/GPU) so it is fully unit-testable and can
run anywhere.  GSE207584 is NOT used here (it is a PENDING_BLOCKED legacy CDS
liability until sequence/family/label rebuild); this is only the reusable
codon-state machinery.

NOTE on honesty: synonymous-codon machinery guarantees *protein identity* by
construction.  It does NOT by itself guarantee that a synonymous edit has a
measured expression/stability effect (that is an empirical question handled by
the benchmark/metric layer once qualified B1 data exists).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Standard genetic code (RNA alphabet: U replaces T)
# ---------------------------------------------------------------------------

#: codon -> amino-acid single letter; "*" = stop.
GENETIC_CODE: Dict[str, str] = {
    "UUU": "F", "UUC": "F", "UUA": "L", "UUG": "L",
    "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S",
    "UAU": "Y", "UAC": "Y", "UAA": "*", "UAG": "*",
    "UGU": "C", "UGC": "C", "UGA": "*", "UGG": "W",
    "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
    "CCU": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAU": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGU": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AUU": "I", "AUC": "I", "AUA": "I", "AUG": "M",
    "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAU": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGU": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V",
    "GCU": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAU": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGU": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

_STOP = "*"
_START = "M"
_START_CODON = "AUG"


def translate(cds: str) -> Optional[str]:
    """Translate a CDS nucleotide string to a protein (None if invalid).

    cds must be an uppercase ACGU string of length % 3 == 0.  Returns the
    amino-acid string (with "*" only at the terminal stop).  Any in-frame
    internal stop or non-ACGU symbol returns None (invalid CDS).
    """
    cds = cds.upper()
    if len(cds) % 3 != 0:
        return None
    if any(ch not in "ACGU" for ch in cds):
        return None
    protein = []
    for i in range(0, len(cds), 3):
        codon = cds[i:i + 3]
        aa = GENETIC_CODE.get(codon)
        if aa is None:
            return None
        if aa == _STOP:
            # only a single terminal stop is valid
            if i != len(cds) - 3:
                return None
            return "".join(protein)
        protein.append(aa)
    # no stop codon found -> incomplete CDS (invalid for our purposes)
    return None


def synonymous_codons(aa: str) -> List[str]:
    """All RNA codons that code for the amino acid `aa` ([] if stop/unknown)."""
    if aa == _STOP:
        return []
    return sorted(c for c, a in GENETIC_CODE.items() if a == aa)


def build_synonymous_classes() -> Dict[str, List[str]]:
    """aa -> list of synonymous codons (excluding stop)."""
    out: Dict[str, List[str]] = {}
    for c, a in GENETIC_CODE.items():
        if a == _STOP:
            continue
        out.setdefault(a, []).append(c)
    return {a: sorted(v) for a, v in out.items()}


# ---------------------------------------------------------------------------
# CDS state + atomic synonymous-codon substitution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CodonEdit:
    """A single atomic synonymous-codon substitution at a codon index."""
    codon_idx: int          # 0-based codon position within the CDS
    new_codon: str          # the synonymous replacement codon (3 nt)
    old_codon: str          # the original codon at that position

    @property
    def is_synonymous(self) -> bool:
        return GENETIC_CODE.get(self.old_codon) == GENETIC_CODE.get(self.new_codon)


@dataclass(frozen=True)
class CDSState:
    """A frame-locked CDS anchored to an ATG start.

    Invariants (enforced by construction on accepted edits):
      * the first codon is the start codon AUG,
      * the last codon is a stop codon,
      * the sequence is a multiple of 3 (frame preserved),
      * the translated protein is unchanged by a synonymous edit.
    """
    seq: str
    start_pos: int = 0          # nucleotide index of the ATG start
    protein: Optional[str] = None

    def __post_init__(self):
        p = translate(self.seq)
        if p is None:
            raise ValueError("CDSState requires a valid in-frame CDS with a "
                             "terminal stop (got %r)" % (self.seq,))
        object.__setattr__(self, "protein", p)

    @property
    def length(self) -> int:
        return len(self.seq)

    @property
    def n_codons(self) -> int:
        return len(self.seq) // 3

    def codons(self) -> List[str]:
        return [self.seq[i:i + 3] for i in range(0, len(self.seq), 3)]


def build_cds_state(seq: str, start_pos: int = 0) -> CDSState:
    """Build a CDSState, verifying start (AUG) and a terminal stop codon."""
    p = translate(seq)
    if p is None:
        raise ValueError("not a valid CDS (must be in-frame with a terminal stop)")
    if seq[start_pos:start_pos + 3] != _START_CODON:
        raise ValueError("CDS must start at an AUG start codon")
    if not seq.endswith(("UAA", "UAG", "UGA")):
        raise ValueError("CDS must end in a terminal stop codon")
    return CDSState(seq, start_pos, p)


def enumerate_synonymous_edits(state: CDSState,
                               editable_codons: Optional[Sequence[int]] = None
                               ) -> List[CodonEdit]:
    """All atomic synonymous-codon edits that preserve the protein and keep the
    start and stop codons intact.

    editable_codons: optional list of codon indices that may be edited (default
    = all internal codons excluding start [0] and stop [n_codons-1]).
    """
    codons = state.codons()
    n = len(codons)
    if editable_codons is None:
        editable_codons = list(range(1, n - 1))  # exclude start & stop
    edits: List[CodonEdit] = []
    for idx in editable_codons:
        old = codons[idx]
        aa = GENETIC_CODE.get(old)
        if aa is None or aa == _STOP:
            continue
        for new in synonymous_codons(aa):
            if new != old:
                edits.append(CodonEdit(idx, new, old))
    return edits


def apply_edit(state: CDSState, edit: CodonEdit) -> CDSState:
    """Apply a CodonEdit, preserving the protein / frame / start / stop.

    Raises ValueError if the edit is not synonymous, or touches the start or
    stop codon, or would break invariants.  Returns a NEW CDSState.
    """
    codons = state.codons()
    n = len(codons)
    if not (0 <= edit.codon_idx < n):
        raise ValueError("codon index out of range")
    if edit.codon_idx == 0 or edit.codon_idx == n - 1:
        raise ValueError("cannot edit the start or stop codon")
    if edit.old_codon != codons[edit.codon_idx]:
        raise ValueError("old_codon does not match the current codon")
    if not edit.is_synonymous:
        raise ValueError("non-synonymous edit is not allowed (protein change)")
    new_codons = list(codons)
    new_codons[edit.codon_idx] = edit.new_codon
    new_seq = "".join(new_codons)
    new_state = build_cds_state(new_seq, state.start_pos)
    if new_state.protein != state.protein:
        raise ValueError("invariant violation: protein identity changed")
    return new_state


# ---------------------------------------------------------------------------
# Protein-family helpers (family/listwise metric + protein-family split)
# ---------------------------------------------------------------------------

def protein_family_id(protein: str) -> str:
    """Stable family id from a protein string (identity-based)."""
    return "prot_%s" % protein


def family_members(records: Sequence[Dict]) -> Dict[str, List[int]]:
    """Group record indices by protein family (from each record's protein).

    Each record is a dict with a "protein" field.  Returns family_id ->
    list of record indices, families with a single member included (a family of
    one has no ranking headroom and is flagged, not silently dropped).
    """
    fam: Dict[str, List[int]] = {}
    for i, rec in enumerate(records):
        fam.setdefault(protein_family_id(rec["protein"]), []).append(i)
    return fam


def listwise_ndcg(score: Sequence[float], gain: Sequence[float],
                  k: Optional[int] = None) -> float:
    """Listwise NDCG@k over one protein family.

    score  : model scores (higher = predicted better).
    gain   : true gains (higher = actually better).  Gains may be signed/negative.
    Returns NDCG@k in [0,1] (1 = perfect ranking).  Ignores DCG normalization
    degeneracy: if the ideal DCG is 0, returns 1.0 (trivially all-equal family).
    """
    if len(score) == 0:
        return 0.0
    if k is None:
        k = len(score)
    k = max(1, min(int(k), len(score)))
    order = sorted(range(len(score)), key=lambda i: score[i], reverse=True)
    # min-max normalize gains to [0,1] so negative/signed gains rank correctly
    g = list(gain)
    gmin, gmax = min(g), max(g)
    if gmax == gmin:
        return 1.0
    norm = [(x - gmin) / (gmax - gmin) for x in g]
    dcg = sum((2 ** norm[order[j]] - 1) / (j + 2)
              for j in range(k))
    ideal = sorted(norm, reverse=True)
    idcg = sum((2 ** ideal[j] - 1) / (j + 2)
               for j in range(k))
    if idcg <= 0:
        return 1.0
    return dcg / idcg


def macro_listwise_ndcg_by_family(records: Sequence[Dict],
                                  score: Sequence[float],
                                  gain: Sequence[float],
                                  k: Optional[int] = None) -> Dict:
    """Study-macro listwise NDCG across protein families.

    Returns per-family NDCG, the macro mean, and the number of families with
    >1 member (rankable).  Families of size 1 are reported but excluded from
    the macro mean (no ranking headroom) with a note.
    """
    fam = family_members(records)
    per_family: Dict[str, float] = {}
    rankable = 0
    for fid, idxs in fam.items():
        if len(idxs) < 2:
            continue
        per_family[fid] = listwise_ndcg(
            [score[i] for i in idxs], [gain[i] for i in idxs], k=k)
        rankable += 1
    macro = sum(per_family.values()) / rankable if rankable else 0.0
    return {
        "per_family": per_family,
        "macro_ndcg": macro,
        "n_families": len(fam),
        "n_rankable_families": rankable,
        "n_singleton_families": len(fam) - rankable,
    }
