"""Alignment helpers: nucleotide UTR positions and codon lattice positions."""
from __future__ import annotations

from dataclasses import dataclass

from mrna_editflow.core.mixed_resolution_state import MixedResolutionState


@dataclass(frozen=True)
class AlignmentEntry:
    token_index: int
    region: str
    local_index: int
    span: int


def build_alignment(state: MixedResolutionState) -> tuple[AlignmentEntry, ...]:
    entries = [AlignmentEntry(i, "5UTR", i, 1) for i in range(len(state.five_utr))]
    offset = len(entries)
    entries.extend(AlignmentEntry(offset + i, "CDS", i // 3, 3) for i in range(len(state.cds)))
    offset = len(entries)
    entries.extend(AlignmentEntry(offset + i, "3UTR", i, 1) for i in range(len(state.three_utr)))
    return tuple(entries)


def assert_codon_atomicity(state: MixedResolutionState) -> None:
    alignment = build_alignment(state)
    cds = [x for x in alignment if x.region == "CDS"]
    if len(cds) != len(state.cds) or any(x.span != 3 for x in cds):
        raise AssertionError("CDS alignment must expose codon-span tokens")
