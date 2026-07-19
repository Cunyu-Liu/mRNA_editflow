"""Small offline functional-element library for mRNA design tasks.

The built-in library is intentionally compact but covers the element families
needed for downstream task construction: uORF, IRES, Kozak, ARE, miRNA seed,
and polyA signal. Optional CSV/JSONL loading is provided for local Rfam or
literature-derived motif files, but no network access is required.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from mrna_editflow.core.constants import STOP_CODONS

_VALID_RNA = frozenset("ACGU")


@dataclass(frozen=True)
class FunctionalElement:
    name: str
    family: str
    sequence: str
    region: str
    description: str
    effect: str

    def to_dict(self) -> dict:
        return asdict(self)


BUILTIN_ELEMENTS: Tuple[FunctionalElement, ...] = (
    FunctionalElement(
        name="uORF_minimal",
        family="uORF",
        sequence="AUGGCUUAA",
        region="5UTR",
        description="Minimal upstream ORF with AUG, one alanine codon, and UAA stop.",
        effect="often decreases main-ORF translation by ribosome diversion",
    ),
    FunctionalElement(
        name="IRES_pyrimidine_core",
        family="IRES",
        sequence="CCUCUCC",
        region="5UTR",
        description="Compact pyrimidine-rich IRES-like core motif for offline tasks.",
        effect="supports cap-independent initiation in motif-insertion tasks",
    ),
    FunctionalElement(
        name="Kozak_strong_core",
        family="Kozak",
        sequence="GCCACC",
        region="5UTR",
        description="Strong Kozak context immediately upstream of AUG.",
        effect="increases initiation efficiency when placed before CDS start",
    ),
    FunctionalElement(
        name="ARE_core",
        family="ARE",
        sequence="AUUUA",
        region="3UTR",
        description="AU-rich element core.",
        effect="can decrease stability through RNA-binding protein recruitment",
    ),
    FunctionalElement(
        name="miRNA_seed_site",
        family="miRNA",
        sequence="UGCACUU",
        region="3UTR",
        description="Small seed-match-like miRNA target site.",
        effect="can repress translation or reduce stability",
    ),
    FunctionalElement(
        name="polyA_signal_hexamer",
        family="polyA",
        sequence="AAUAAA",
        region="3UTR",
        description="Canonical cleavage/polyadenylation signal.",
        effect="supports 3' end processing and mRNA stability",
    ),
)


def _normalise(seq: str) -> str:
    s = "".join(str(seq).split()).upper().replace("T", "U")
    bad = set(s) - _VALID_RNA
    if bad:
        raise ValueError(f"sequence contains non-ACGU characters: {sorted(bad)}")
    return s


def get_element_library(families: Optional[Iterable[str]] = None) -> List[FunctionalElement]:
    """Return built-in elements, optionally filtered by family."""
    if families is None:
        return list(BUILTIN_ELEMENTS)
    wanted = {f.lower() for f in families}
    return [e for e in BUILTIN_ELEMENTS if e.family.lower() in wanted]


def get_element(name_or_family: str) -> FunctionalElement:
    """Find an element by exact name first, then by family."""
    key = name_or_family.lower()
    for element in BUILTIN_ELEMENTS:
        if element.name.lower() == key:
            return element
    for element in BUILTIN_ELEMENTS:
        if element.family.lower() == key:
            return element
    raise KeyError(f"unknown functional element {name_or_family!r}")


def _find_literal(seq: str, element: FunctionalElement) -> List[dict]:
    motif = _normalise(element.sequence)
    hits: List[dict] = []
    start = seq.find(motif)
    while start != -1:
        hits.append(
            {
                "name": element.name,
                "family": element.family,
                "start": start,
                "end": start + len(motif),
                "sequence": motif,
                "region": element.region,
            }
        )
        start = seq.find(motif, start + 1)
    return hits


def _find_uorfs(seq: str) -> List[dict]:
    hits: List[dict] = []
    start = seq.find("AUG")
    while start != -1:
        for j in range(start + 3, min(len(seq) - 2, start + 90), 3):
            codon = seq[j:j + 3]
            if codon in STOP_CODONS:
                hits.append(
                    {
                        "name": "uORF_detected",
                        "family": "uORF",
                        "start": start,
                        "end": j + 3,
                        "sequence": seq[start:j + 3],
                        "region": "5UTR",
                    }
                )
                break
        start = seq.find("AUG", start + 1)
    return hits


def find_motifs(
    sequence: str,
    library: Optional[Sequence[FunctionalElement]] = None,
    include_uorf: bool = True,
) -> List[dict]:
    """Detect built-in motifs in ``sequence`` using exact offline matching."""
    seq = _normalise(sequence)
    elements = list(library) if library is not None else list(BUILTIN_ELEMENTS)
    hits: List[dict] = []
    for element in elements:
        if element.family == "uORF":
            continue
        hits.extend(_find_literal(seq, element))
    if include_uorf:
        hits.extend(_find_uorfs(seq))
    hits.sort(key=lambda h: (h["start"], h["end"], h["family"], h["name"]))
    return hits


def insert_element(
    sequence: str,
    name_or_family: str,
    position: Optional[int] = None,
) -> Tuple[str, dict]:
    """Insert a built-in element and return ``(new_sequence, insertion_info)``."""
    seq = _normalise(sequence)
    element = get_element(name_or_family)
    motif = _normalise(element.sequence)
    if position is None:
        position = len(seq)
    if not 0 <= position <= len(seq):
        raise ValueError("position out of range")
    new_seq = seq[:position] + motif + seq[position:]
    info = {
        "name": element.name,
        "family": element.family,
        "start": position,
        "end": position + len(motif),
        "sequence": motif,
        "region": element.region,
    }
    return new_seq, info


def remove_motifs(
    sequence: str,
    name_or_family: str,
    max_remove: int = 1,
) -> Tuple[str, List[dict]]:
    """Remove up to ``max_remove`` detected motifs matching name or family."""
    seq = _normalise(sequence)
    key = name_or_family.lower()
    matches = [
        h for h in find_motifs(seq)
        if h["name"].lower() == key or h["family"].lower() == key
    ][:max_remove]
    chars = seq
    removed: List[dict] = []
    for hit in sorted(matches, key=lambda h: h["start"], reverse=True):
        chars = chars[:hit["start"]] + chars[hit["end"]:]
        removed.append(hit)
    removed.reverse()
    return chars, removed


def load_element_library(path: Optional[str] = None) -> List[FunctionalElement]:
    """Load optional local CSV/JSONL motifs, or return the built-in library."""
    if path is None:
        return get_element_library()
    fp = Path(path)
    rows: List[Mapping[str, object]] = []
    if fp.suffix.lower() == ".jsonl":
        with open(fp, "r", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    else:
        delim = "\t" if fp.suffix.lower() in {".tsv", ".tab"} else ","
        with open(fp, "r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter=delim))
    elements: List[FunctionalElement] = []
    for row in rows:
        elements.append(
            FunctionalElement(
                name=str(row["name"]),
                family=str(row["family"]),
                sequence=_normalise(str(row["sequence"])),
                region=str(row.get("region", "")),
                description=str(row.get("description", "")),
                effect=str(row.get("effect", "")),
            )
        )
    return elements


__all__ = [
    "FunctionalElement",
    "BUILTIN_ELEMENTS",
    "get_element_library",
    "get_element",
    "find_motifs",
    "insert_element",
    "remove_motifs",
    "load_element_library",
]
