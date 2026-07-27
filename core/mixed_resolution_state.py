"""Typed mixed-resolution state and atomic legal action graph."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from mrna_editflow.core.constants import NUC_VOCAB, SYNONYMOUS_CODONS, translate


@dataclass(frozen=True)
class MixedResolutionState:
    five_utr: str
    cds: str
    three_utr: str = ""
    cargo_id: str = ""
    cell_context: str = ""
    transcript_id: str = ""

    def __post_init__(self) -> None:
        for name in ("five_utr", "cds", "three_utr"):
            value = getattr(self, name).upper().replace("T", "U")
            if any(ch not in NUC_VOCAB for ch in value):
                raise ValueError(f"{name} contains non-RNA symbols")
            object.__setattr__(self, name, value)
        if len(self.cds) % 3:
            raise ValueError("CDS must be codon-aligned")

    @property
    def protein(self) -> str:
        return translate(self.cds)

    @property
    def codons(self) -> tuple[str, ...]:
        return tuple(self.cds[i:i + 3] for i in range(0, len(self.cds), 3))

    @property
    def key(self) -> tuple:
        return (self.five_utr, self.cds, self.three_utr, self.cargo_id, self.cell_context)


@dataclass(frozen=True)
class MixedAction:
    kind: str
    index: int = -1
    value: str = ""

    @classmethod
    def stop(cls) -> "MixedAction":
        return cls("STOP")

    def is_stop(self) -> bool:
        return self.kind == "STOP"

    def to_tuple(self) -> tuple:
        return (self.kind, self.index, self.value)


def legal_actions(
    state: MixedResolutionState,
    *,
    include_three_utr: bool = True,
    include_cds: bool = True,
) -> List[MixedAction]:
    actions = [MixedAction.stop()]
    for pos, ref in enumerate(state.five_utr):
        actions.extend(MixedAction("UTR_SUB", pos, nt) for nt in NUC_VOCAB if nt != ref)
    if include_three_utr:
        offset = len(state.five_utr)
        for pos, ref in enumerate(state.three_utr):
            actions.extend(MixedAction("UTR3_SUB", offset + pos, nt) for nt in NUC_VOCAB if nt != ref)
    if include_cds:
        for codon_pos, codon in enumerate(state.codons):
            aa = next((aa for aa, codons in SYNONYMOUS_CODONS.items() if codon in codons), None)
            if aa is None:
                continue
            actions.extend(MixedAction("CDS_SYN_SWAP", codon_pos, target)
                           for target in SYNONYMOUS_CODONS[aa] if target != codon)
    return actions


def apply_action(state: MixedResolutionState, action: MixedAction) -> MixedResolutionState:
    if action.is_stop():
        return state
    if action.kind == "UTR_SUB":
        if not (0 <= action.index < len(state.five_utr)):
            raise ValueError("UTR_SUB index out of range")
        if action.value not in NUC_VOCAB or action.value == state.five_utr[action.index]:
            raise ValueError("illegal UTR_SUB target")
        seq = state.five_utr[:action.index] + action.value + state.five_utr[action.index + 1:]
        return MixedResolutionState(seq, state.cds, state.three_utr, state.cargo_id, state.cell_context, state.transcript_id)
    if action.kind == "UTR3_SUB":
        pos = action.index - len(state.five_utr)
        if not (0 <= pos < len(state.three_utr)):
            raise ValueError("UTR3_SUB index out of range")
        if action.value not in NUC_VOCAB or action.value == state.three_utr[pos]:
            raise ValueError("illegal UTR3_SUB target")
        seq = state.three_utr[:pos] + action.value + state.three_utr[pos + 1:]
        return MixedResolutionState(state.five_utr, state.cds, seq, state.cargo_id, state.cell_context, state.transcript_id)
    if action.kind == "CDS_SYN_SWAP":
        if not (0 <= action.index < len(state.codons)):
            raise ValueError("CDS_SYN_SWAP index out of range")
        old = state.codons[action.index]
        valid = SYNONYMOUS_CODONS.get(next((aa for aa, cs in SYNONYMOUS_CODONS.items() if old in cs), ""), [])
        if action.value not in valid or action.value == old:
            raise ValueError("non-synonymous or identity codon swap")
        codons = list(state.codons); codons[action.index] = action.value
        out = MixedResolutionState(state.five_utr, "".join(codons), state.three_utr, state.cargo_id, state.cell_context, state.transcript_id)
        if out.protein != state.protein:
            raise AssertionError("atomic codon swap changed protein identity")
        return out
    raise ValueError(f"unknown mixed-resolution action {action.kind!r}")


__all__ = ["MixedResolutionState", "MixedAction", "legal_actions", "apply_action"]
