"""Shared constants and biological tables for mRNA-EditFlow (MEF).

This module is the single source of truth for the token vocabulary, region
labels, the standard genetic code, and derived synonymous-codon groupings.
Every other module (data pipeline, model heads, evaluation) MUST import these
symbols rather than re-defining them, to guarantee a consistent interface.

Design notes
------------
* mRNA sequences use the RNA alphabet ``A C G U``. Input ``T`` is normalised to
  ``U`` upstream in the data pipeline; this module only knows about ``U``.
* Special tokens follow the original Edit Flow convention (BOS/PAD/GAP) so the
  reused CTMC alignment kernel keeps working unchanged.
* Region ids encode the heterogeneous "edit grammar": UTRs are free-length,
  CDS is a frame-locked codon lattice.

Complexity: all builders below are O(64) one-time table constructions.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Nucleotide vocabulary (RNA alphabet)
# ---------------------------------------------------------------------------
NUC_VOCAB: str = "ACGU"
NUC_TO_ID: Dict[str, int] = {c: i for i, c in enumerate(NUC_VOCAB)}
ID_TO_NUC: Dict[int, str] = {i: c for i, c in enumerate(NUC_VOCAB)}

V: int = len(NUC_VOCAB)  # 4 valid nucleotides
BOS_TOKEN: int = V       # 4
PAD_TOKEN: int = V + 1   # 5
GAP_TOKEN: int = V + 2   # 6  (used only inside aligned z-space)
# Full model vocabulary size for embedding tables that must also see BOS/PAD.
VOCAB_MODEL_SIZE: int = V + 2  # 6 (A,C,G,U,BOS,PAD)
# Alignment vocabulary also needs the GAP symbol.
VOCAB_ALIGN_SIZE: int = V + 3  # 7

# ---------------------------------------------------------------------------
# Region labels (the "edit grammar" partition)
# ---------------------------------------------------------------------------
REGION_5UTR: int = 0
REGION_CDS: int = 1
REGION_3UTR: int = 2
REGION_NAMES: Dict[int, str] = {
    REGION_5UTR: "5UTR",
    REGION_CDS: "CDS",
    REGION_3UTR: "3UTR",
}
NUM_REGIONS: int = 3

# Codon phase inside CDS (0,1,2); non-CDS positions use PHASE_NONE.
NUM_PHASES: int = 3
PHASE_NONE: int = 3  # sentinel embedding index for non-CDS positions

# ---------------------------------------------------------------------------
# Standard genetic code (RNA codons -> amino acid single letter; '*' = stop)
# ---------------------------------------------------------------------------
CODON_TABLE: Dict[str, str] = {
    "UUU": "F", "UUC": "F", "UUA": "L", "UUG": "L",
    "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
    "AUU": "I", "AUC": "I", "AUA": "I", "AUG": "M",
    "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V",
    "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S",
    "CCU": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCU": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "UAU": "Y", "UAC": "Y", "UAA": "*", "UAG": "*",
    "CAU": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAU": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAU": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "UGU": "C", "UGC": "C", "UGA": "*", "UGG": "W",
    "CGU": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGU": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGU": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

START_CODON: str = "AUG"
STOP_CODONS: Tuple[str, ...] = ("UAA", "UAG", "UGA")


def _build_synonymous_groups() -> Dict[str, List[str]]:
    """Group codons by the amino acid they encode (stop codons grouped too).

    Returns a mapping ``aa -> sorted list of synonymous codons``.
    """
    groups: Dict[str, List[str]] = {}
    for codon, aa in CODON_TABLE.items():
        groups.setdefault(aa, []).append(codon)
    for aa in groups:
        groups[aa].sort()
    return groups


SYNONYMOUS_CODONS: Dict[str, List[str]] = _build_synonymous_groups()


def codon_to_index(codon: str) -> int:
    """Map a 3-nt RNA codon to a dense index in ``[0, 64)``.

    Ordering matches ``A C G U`` place value (base-4), giving a stable,
    reproducible codon vocabulary independent of dict iteration order.

    Complexity: O(3).
    """
    if len(codon) != 3:
        raise ValueError(f"codon must be length 3, got {codon!r}")
    idx = 0
    for ch in codon:
        if ch not in NUC_TO_ID:
            raise ValueError(f"illegal nucleotide in codon {codon!r}")
        idx = idx * 4 + NUC_TO_ID[ch]
    return idx


def index_to_codon(idx: int) -> str:
    """Inverse of :func:`codon_to_index`. Complexity: O(3)."""
    if not 0 <= idx < 64:
        raise ValueError(f"codon index out of range: {idx}")
    chars = []
    for _ in range(3):
        chars.append(ID_TO_NUC[idx % 4])
        idx //= 4
    return "".join(reversed(chars))


# Amino acid -> list of synonymous codon indices (for logit masking in the
# codon-lattice constrained substitution head).
AA_TO_CODON_INDICES: Dict[str, List[int]] = {
    aa: sorted(codon_to_index(c) for c in codons)
    for aa, codons in SYNONYMOUS_CODONS.items()
}

# codon index -> amino acid (fast lookup for the constrained operator).
CODON_INDEX_TO_AA: Dict[int, str] = {
    codon_to_index(c): aa for c, aa in CODON_TABLE.items()
}


def translate(cds: str) -> str:
    """Translate an in-frame RNA CDS string to a protein string.

    Trailing incomplete codons are ignored. A terminal stop is rendered as
    ``*``; internal stops are also rendered (callers validate separately).

    Complexity: O(len(cds)).
    """
    prot = []
    for i in range(0, len(cds) - 2, 3):
        prot.append(CODON_TABLE.get(cds[i:i + 3], "X"))
    return "".join(prot)


def is_valid_cds(cds: str) -> bool:
    """Check ATG start, in-frame length, single terminal stop, no internal stop.

    Complexity: O(len(cds)).
    """
    if len(cds) < 6 or len(cds) % 3 != 0:
        return False
    if cds[:3] != START_CODON:
        return False
    prot = translate(cds)
    if not prot.endswith("*"):
        return False
    if "*" in prot[:-1]:
        return False
    if "X" in prot:
        return False
    return True


__all__ = [
    "NUC_VOCAB", "NUC_TO_ID", "ID_TO_NUC", "V",
    "BOS_TOKEN", "PAD_TOKEN", "GAP_TOKEN",
    "VOCAB_MODEL_SIZE", "VOCAB_ALIGN_SIZE",
    "REGION_5UTR", "REGION_CDS", "REGION_3UTR", "REGION_NAMES", "NUM_REGIONS",
    "NUM_PHASES", "PHASE_NONE",
    "CODON_TABLE", "START_CODON", "STOP_CODONS",
    "SYNONYMOUS_CODONS", "AA_TO_CODON_INDICES", "CODON_INDEX_TO_AA",
    "codon_to_index", "index_to_codon", "translate", "is_valid_cds",
]
