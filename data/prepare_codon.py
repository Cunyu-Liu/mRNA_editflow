"""Protein<->CDS downstream task preparation.

This module turns cleaned :class:`MRNARecord` objects into codon-optimisation
samples. It computes lightweight, dependency-free sequence features and assigns
splits at the protein level so synonymous CDS variants do not leak between
train/validation/test.
"""
from __future__ import annotations

import json
import math
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from mrna_editflow.core.constants import CODON_TABLE, SYNONYMOUS_CODONS, is_valid_cds, translate
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.download_mrna import synthesize_corpus

_PAIR_SCORE = {
    ("G", "C"): 3.0,
    ("C", "G"): 3.0,
    ("A", "U"): 2.0,
    ("U", "A"): 2.0,
    ("G", "U"): 1.0,
    ("U", "G"): 1.0,
}


def _codons(cds: str) -> List[str]:
    return [cds[i:i + 3] for i in range(0, len(cds) - 2, 3)]


def gc_fraction(seq: str) -> float:
    """GC fraction in ``[0, 1]`` with an empty-sequence fallback."""
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


def gc3_fraction(cds: str) -> float:
    """Third-position GC fraction over sense codons."""
    codons = [c for c in _codons(cds)[:-1] if CODON_TABLE.get(c) != "*"]
    if not codons:
        return 0.0
    return sum(c[2] in "GC" for c in codons) / len(codons)


def _relative_adaptiveness() -> Dict[str, float]:
    """Small deterministic codon-weight table used by :func:`calculate_cai`."""
    weights: Dict[str, float] = {}
    for aa, codons in SYNONYMOUS_CODONS.items():
        if aa == "*":
            continue
        scores = {}
        for codon in codons:
            # GC3 and moderate total GC are common in highly expressed human CDS.
            scores[codon] = 1.0 + 0.35 * (codon[2] in "GC") + 0.05 * (
                codon.count("G") + codon.count("C")
            )
        max_score = max(scores.values())
        for codon, score in scores.items():
            weights[codon] = max(0.05, score / max_score)
    return weights


_CODON_WEIGHTS = _relative_adaptiveness()


def calculate_cai(cds: str) -> float:
    """Compute a lightweight CAI proxy as a geometric mean of codon weights."""
    logs: List[float] = []
    for codon in _codons(cds)[:-1]:  # skip terminal stop
        aa = CODON_TABLE.get(codon)
        if aa is None or aa == "*":
            continue
        logs.append(math.log(_CODON_WEIGHTS.get(codon, 0.05)))
    if not logs:
        return 0.0
    return float(math.exp(sum(logs) / len(logs)))


def mfe_proxy(seq: str) -> float:
    """Dependency-free MFE proxy based on long-range complementarity."""
    if not seq:
        return 0.0
    score = 0.0
    n_pairs = min(len(seq) // 2, 256)
    for i in range(n_pairs):
        score += _PAIR_SCORE.get((seq[i], seq[-i - 1]), 0.0)
    return float(-score)


def make_codon_sample(record: MRNARecord) -> dict:
    """Create one protein<->CDS sample and assert translation consistency."""
    if not is_valid_cds(record.cds):
        raise ValueError(f"record {record.transcript_id!r} has invalid CDS")
    protein_with_stop = translate(record.cds)
    if not protein_with_stop.endswith("*"):
        raise ValueError("valid CDS did not translate to a terminal stop")
    protein = protein_with_stop[:-1]
    assert translate(record.cds) == protein + "*", "CDS/protein mismatch"
    return {
        "sample_id": record.transcript_id,
        "species": record.species,
        "protein": protein,
        "protein_with_stop": protein_with_stop,
        "cds": record.cds,
        "cai": calculate_cai(record.cds),
        "gc": gc_fraction(record.cds),
        "gc3": gc3_fraction(record.cds),
        "mfe_proxy": mfe_proxy(record.cds),
    }


def _assign_group_splits(
    groups: Sequence[str],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Dict[str, str]:
    rng = random.Random(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    n = len(shuffled)
    split_by_group: Dict[str, str] = {}
    for i, group in enumerate(shuffled):
        frac = (i + 0.5) / max(1, n)
        if frac < train_frac:
            split = "train"
        elif frac < train_frac + val_frac:
            split = "val"
        else:
            split = "test"
        split_by_group[group] = split
    return split_by_group


def protein_level_split(
    samples: Sequence[Mapping[str, object]],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 0,
) -> List[dict]:
    """Assign splits so identical protein strings never straddle splits."""
    proteins = sorted({str(s["protein"]) for s in samples})
    split_by_protein = _assign_group_splits(proteins, train_frac, val_frac, seed)
    out: List[dict] = []
    for sample in samples:
        item = dict(sample)
        item["split"] = split_by_protein[str(item["protein"])]
        out.append(item)
    return out


def prepare_codon_dataset(
    records: Optional[Iterable[MRNARecord]] = None,
    n_synthetic: int = 64,
    seed: int = 0,
    output_jsonl: Optional[str] = None,
) -> List[dict]:
    """Build protein<->CDS samples from records or synthetic fallback."""
    if records is None:
        records = synthesize_corpus(n_synthetic, seed=seed)
    samples = [make_codon_sample(r) for r in records]
    samples = protein_level_split(samples, seed=seed)
    if output_jsonl is not None:
        write_jsonl(samples, output_jsonl)
    return samples


def write_jsonl(samples: Sequence[Mapping[str, object]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(dict(sample), sort_keys=True) + "\n")


__all__ = [
    "gc_fraction",
    "gc3_fraction",
    "calculate_cai",
    "mfe_proxy",
    "make_codon_sample",
    "protein_level_split",
    "prepare_codon_dataset",
    "write_jsonl",
]
