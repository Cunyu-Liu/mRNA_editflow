"""Sequence augmentations that respect mRNA region constraints.

CDS augmentation is restricted to synonymous codon substitutions and asserts
100% protein identity after every perturbation. UTR augmentation is local
single-nucleotide noise with protected functional motifs. Reverse-complement
augmentation is explicitly forbidden because mRNAs are strand-oriented
transcripts, not double-stranded sequence examples.
"""
from __future__ import annotations

import random
from typing import Iterable, List, Optional, Sequence, Set

from mrna_editflow.core.constants import (
    CODON_TABLE,
    NUC_VOCAB,
    SYNONYMOUS_CODONS,
    is_valid_cds,
    translate,
)
from mrna_editflow.core.schema import MRNARecord

_DEFAULT_PROTECTED_MOTIFS = (
    "AUG",       # uORF/start-like signal
    "GCCACC",    # Kozak core context
    "AUUUA",     # ARE
    "UGCACUU",   # miRNA seed-like site
    "AAUAAA",    # polyA signal
    "CCUCUCC",   # compact IRES-like pyrimidine-rich core
)


def _normalise(seq: str) -> str:
    s = "".join(str(seq).split()).upper().replace("T", "U")
    bad = set(s) - set(NUC_VOCAB)
    if bad:
        raise ValueError(f"sequence contains non-ACGU characters: {sorted(bad)}")
    return s


def protein_identity(a: str, b: str) -> float:
    """Fractional identity for two translated protein strings."""
    if len(a) != len(b):
        return 0.0
    if not a:
        return 1.0
    return sum(x == y for x, y in zip(a, b)) / len(a)


def synonymously_perturb_cds(
    cds: str,
    edit_fraction: float = 0.1,
    seed: Optional[int] = None,
    min_edits: int = 1,
) -> str:
    """Apply synonymous codon substitutions while preserving the protein.

    Start and terminal stop codons are never modified. If a CDS has no
    synonymous alternatives, it is returned unchanged.
    """
    cds = _normalise(cds)
    if len(cds) % 3 != 0:
        raise ValueError("CDS length must be a multiple of 3")
    original_protein = translate(cds)
    codons = [cds[i:i + 3] for i in range(0, len(cds), 3)]
    candidates: List[int] = []
    for idx in range(1, max(1, len(codons) - 1)):
        codon = codons[idx]
        aa = CODON_TABLE.get(codon)
        if aa is None or aa == "*":
            continue
        alternatives = [c for c in SYNONYMOUS_CODONS[aa] if c != codon]
        if alternatives:
            candidates.append(idx)
    if not candidates or edit_fraction <= 0.0:
        return cds

    rng = random.Random(seed)
    rng.shuffle(candidates)
    n_target = int(round(len(candidates) * edit_fraction))
    n_target = max(min_edits, n_target)
    n_target = min(len(candidates), n_target)
    for idx in candidates[:n_target]:
        aa = CODON_TABLE[codons[idx]]
        alternatives = [c for c in SYNONYMOUS_CODONS[aa] if c != codons[idx]]
        codons[idx] = rng.choice(alternatives)

    mutated = "".join(codons)
    mutated_protein = translate(mutated)
    assert mutated_protein == original_protein, "synonymous perturbation changed protein"
    if is_valid_cds(cds):
        assert is_valid_cds(mutated), "synonymous perturbation broke CDS validity"
    return mutated


def _protected_positions(seq: str, motifs: Iterable[str]) -> Set[int]:
    protected: Set[int] = set()
    for motif in motifs:
        m = _normalise(motif)
        if not m:
            continue
        start = seq.find(m)
        while start != -1:
            protected.update(range(start, start + len(m)))
            start = seq.find(m, start + 1)
    return protected


def motif_preserving_utr_perturb(
    utr: str,
    protected_motifs: Optional[Sequence[str]] = None,
    edit_fraction: float = 0.08,
    seed: Optional[int] = None,
    min_edits: int = 1,
) -> str:
    """Locally perturb UTR nucleotides without changing protected motifs."""
    seq = _normalise(utr)
    if not seq or edit_fraction <= 0.0:
        return seq
    motifs = protected_motifs if protected_motifs is not None else _DEFAULT_PROTECTED_MOTIFS
    protected = _protected_positions(seq, motifs)
    editable = [i for i in range(len(seq)) if i not in protected]
    if not editable:
        return seq

    rng = random.Random(seed)
    window_len = max(1, min(len(seq), int(round(len(seq) * max(edit_fraction, 0.05) * 4))))
    window_start = rng.randint(0, max(0, len(seq) - window_len))
    window = {i for i in range(window_start, window_start + window_len)}
    local_editable = [i for i in editable if i in window] or editable
    rng.shuffle(local_editable)
    n_target = max(min_edits, int(round(len(seq) * edit_fraction)))
    n_target = min(len(local_editable), n_target)

    chars = list(seq)
    for idx in local_editable[:n_target]:
        chars[idx] = rng.choice([c for c in NUC_VOCAB if c != chars[idx]])
    mutated = "".join(chars)

    for motif in motifs:
        m = _normalise(motif)
        if m and seq.count(m) > mutated.count(m):
            raise AssertionError(f"protected motif {m!r} was removed")
    return mutated


def augment_record(
    record: MRNARecord,
    seed: Optional[int] = None,
    cds_edit_fraction: float = 0.1,
    utr_edit_fraction: float = 0.08,
) -> MRNARecord:
    """Return a region-aware augmented transcript."""
    rng = random.Random(seed)
    cds = synonymously_perturb_cds(record.cds, cds_edit_fraction, seed=rng.randint(0, 10**9))
    five = motif_preserving_utr_perturb(record.five_utr, edit_fraction=utr_edit_fraction,
                                        seed=rng.randint(0, 10**9))
    three = motif_preserving_utr_perturb(record.three_utr, edit_fraction=utr_edit_fraction,
                                         seed=rng.randint(0, 10**9))
    return MRNARecord(
        transcript_id=f"{record.transcript_id}_aug",
        five_utr=five,
        cds=cds,
        three_utr=three,
        species=record.species,
    )


def reverse_complement_augment(_record: MRNARecord) -> MRNARecord:
    """Reject reverse-complement augmentation for strand-oriented mRNA data."""
    raise ValueError("reverse-complement augmentation is forbidden for mRNA tasks")


__all__ = [
    "protein_identity",
    "synonymously_perturb_cds",
    "motif_preserving_utr_perturb",
    "augment_record",
    "reverse_complement_augment",
]
