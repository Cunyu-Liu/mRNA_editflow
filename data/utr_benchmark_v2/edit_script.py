"""Canonical UTR edit scripts with fail-closed dynamic-coordinate semantics.

Coordinates are zero-based offsets into the *current* sequence immediately
before an action is applied.  Canonical endpoint scripts contain only
mutating actions; ``STOP`` is supported for trajectory execution, is terminal,
and is never inserted by :func:`canonicalize_edit_script`.

The canonical alignment is a minimum-character-edit Levenshtein alignment.
Ties are resolved deterministically in this order:

``MATCH`` (right-anchor repeated indels), ``SUB``, ``DEL``, ``INS``.

The reported equivalent-script count is the exact count of minimum-cost
character alignments.  It intentionally does not claim to enumerate arbitrary
non-minimal cycles or every permutation of independent actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


RNA_ALPHABET = frozenset("ACGU")
ACTION_TYPES = frozenset({"SUB", "INS", "DEL", "STOP"})
_TIE_BREAK = ("MATCH", "SUB", "DEL", "INS")


class EditScriptError(ValueError):
    """Raised when an edit script is malformed or cannot be applied exactly."""


@dataclass(frozen=True)
class EditAction:
    """One dynamic-coordinate edit action.

    ``SUB`` uses one-base ``ref`` and ``alt`` alleles.  ``INS`` uses an empty
    ``ref`` and a non-empty ``alt``.  ``DEL`` uses a non-empty ``ref`` and an
    empty ``alt``.  ``STOP`` has no position or alleles.
    """

    op: str
    pos: int | None = None
    ref: str = ""
    alt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "pos": self.pos,
            "ref": self.ref,
            "alt": self.alt,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EditAction":
        if not isinstance(payload, Mapping):
            raise EditScriptError("edit action must be a mapping")
        if "op" not in payload:
            raise EditScriptError("edit action is missing op")
        return cls(
            op=str(payload["op"]),
            pos=payload.get("pos"),
            ref=str(payload.get("ref", "")),
            alt=str(payload.get("alt", "")),
        )


def _validate_rna(sequence: str, *, name: str, allow_empty: bool = True) -> None:
    if not isinstance(sequence, str):
        raise EditScriptError(f"{name} must be a string")
    if not allow_empty and not sequence:
        raise EditScriptError(f"{name} must be non-empty")
    if sequence != sequence.upper() or any(base not in RNA_ALPHABET for base in sequence):
        raise EditScriptError(
            f"{name} must use the uppercase RNA alphabet A/C/G/U"
        )


def _coerce_action(action: EditAction | Mapping[str, Any]) -> EditAction:
    if isinstance(action, EditAction):
        return action
    return EditAction.from_dict(action)


def _coerce_actions(
    script: Sequence[EditAction | Mapping[str, Any]] | Mapping[str, Any],
) -> list[EditAction]:
    if isinstance(script, Mapping):
        if "actions" not in script:
            raise EditScriptError("edit-script mapping is missing actions")
        script = script["actions"]
    if isinstance(script, (str, bytes)) or not isinstance(script, Sequence):
        raise EditScriptError("edit script must be an action sequence")
    return [_coerce_action(action) for action in script]


def _position(action: EditAction, *, upper: int, insertion: bool) -> int:
    pos = action.pos
    if isinstance(pos, bool) or not isinstance(pos, int):
        raise EditScriptError(f"{action.op} position must be an integer")
    valid = 0 <= pos <= upper if insertion else 0 <= pos < upper
    if not valid:
        raise EditScriptError(
            f"{action.op} position {pos} is out of bounds for length {upper}"
        )
    return pos


def _validate_allele(allele: str, *, name: str, allow_empty: bool) -> None:
    _validate_rna(allele, name=name, allow_empty=allow_empty)


def apply_edit_script(
    source: str,
    script: Sequence[EditAction | Mapping[str, Any]] | Mapping[str, Any],
) -> str:
    """Apply ``script`` to ``source`` using dynamic state coordinates.

    Every allele is checked against the current state.  No-op edits, repeated
    states (cycles), out-of-range positions, malformed alleles, and any action
    after ``STOP`` raise :class:`EditScriptError`.
    """

    _validate_rna(source, name="source")
    actions = _coerce_actions(script)
    current = source
    seen_states = {source}
    stopped = False

    for index, action in enumerate(actions):
        if stopped:
            raise EditScriptError(f"action {index} occurs after STOP")
        op = action.op
        if op not in ACTION_TYPES:
            raise EditScriptError(f"unsupported edit operation: {op!r}")

        if op == "STOP":
            if action.pos is not None:
                raise EditScriptError("STOP position must be null")
            if action.ref or action.alt:
                raise EditScriptError("STOP alleles must be empty")
            stopped = True
            continue

        if op == "SUB":
            pos = _position(action, upper=len(current), insertion=False)
            _validate_allele(action.ref, name="SUB reference allele", allow_empty=False)
            _validate_allele(action.alt, name="SUB alternate allele", allow_empty=False)
            if len(action.ref) != 1 or len(action.alt) != 1:
                raise EditScriptError("SUB alleles must each contain exactly one base")
            if action.ref == action.alt:
                raise EditScriptError("SUB no-op is forbidden")
            if current[pos] != action.ref:
                raise EditScriptError(
                    f"SUB reference allele mismatch at {pos}: "
                    f"state has {current[pos]!r}, action has {action.ref!r}"
                )
            updated = current[:pos] + action.alt + current[pos + 1 :]

        elif op == "INS":
            pos = _position(action, upper=len(current), insertion=True)
            if action.ref:
                raise EditScriptError("INS requires an empty reference allele")
            _validate_allele(action.alt, name="INS alternate allele", allow_empty=False)
            updated = current[:pos] + action.alt + current[pos:]

        else:  # DEL
            pos = _position(action, upper=len(current), insertion=False)
            _validate_allele(action.ref, name="DEL reference allele", allow_empty=False)
            if action.alt:
                raise EditScriptError("DEL requires an empty alternate allele")
            end = pos + len(action.ref)
            if end > len(current):
                raise EditScriptError(
                    f"DEL span [{pos}, {end}) is out of bounds for length {len(current)}"
                )
            observed = current[pos:end]
            if observed != action.ref:
                raise EditScriptError(
                    f"DEL reference allele mismatch at {pos}: "
                    f"state has {observed!r}, action has {action.ref!r}"
                )
            updated = current[:pos] + current[end:]

        if updated == current:
            raise EditScriptError(f"{op} no-op is forbidden")
        if updated in seen_states:
            raise EditScriptError(
                f"{op} creates a previously visited state (cycle detected)"
            )
        current = updated
        seen_states.add(current)

    return current


def _levenshtein_tables(
    source: str, candidate: str
) -> tuple[list[list[int]], list[list[int]]]:
    """Return suffix minimum costs and exact optimal-alignment path counts."""

    n, m = len(source), len(candidate)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    count = [[0] * (m + 1) for _ in range(n + 1)]
    count[n][m] = 1

    for i in range(n, -1, -1):
        for j in range(m, -1, -1):
            if i == n and j == m:
                continue
            choices: list[tuple[int, int]] = []
            if i < n and j < m and source[i] == candidate[j]:
                choices.append((cost[i + 1][j + 1], count[i + 1][j + 1]))
            if i < n and j < m and source[i] != candidate[j]:
                choices.append((1 + cost[i + 1][j + 1], count[i + 1][j + 1]))
            if i < n:
                choices.append((1 + cost[i + 1][j], count[i + 1][j]))
            if j < m:
                choices.append((1 + cost[i][j + 1], count[i][j + 1]))
            best = min(value for value, _ in choices)
            cost[i][j] = best
            count[i][j] = sum(paths for value, paths in choices if value == best)
    return cost, count


def _optimal_transitions(
    source: str,
    candidate: str,
    cost: list[list[int]],
    i: int,
    j: int,
) -> list[str]:
    best = cost[i][j]
    transitions: list[str] = []
    if (
        i < len(source)
        and j < len(candidate)
        and source[i] == candidate[j]
        and cost[i + 1][j + 1] == best
    ):
        transitions.append("MATCH")
    if (
        i < len(source)
        and j < len(candidate)
        and source[i] != candidate[j]
        and 1 + cost[i + 1][j + 1] == best
    ):
        transitions.append("SUB")
    if i < len(source) and 1 + cost[i + 1][j] == best:
        transitions.append("DEL")
    if j < len(candidate) and 1 + cost[i][j + 1] == best:
        transitions.append("INS")
    return transitions


def _merge_adjacent(actions: list[EditAction]) -> list[EditAction]:
    merged: list[EditAction] = []
    for action in actions:
        if merged and action.op == "DEL" and merged[-1].op == "DEL":
            previous = merged[-1]
            if action.pos == previous.pos:
                merged[-1] = EditAction(
                    "DEL",
                    previous.pos,
                    ref=previous.ref + action.ref,
                )
                continue
        if merged and action.op == "INS" and merged[-1].op == "INS":
            previous = merged[-1]
            if action.pos == previous.pos + len(previous.alt):
                merged[-1] = EditAction(
                    "INS",
                    previous.pos,
                    alt=previous.alt + action.alt,
                )
                continue
        merged.append(action)
    return merged


def _canonical_actions(
    source: str, candidate: str, cost: list[list[int]]
) -> list[EditAction]:
    primitive: list[EditAction] = []
    i = j = 0
    while i < len(source) or j < len(candidate):
        transitions = _optimal_transitions(source, candidate, cost, i, j)
        if not transitions:
            raise AssertionError("minimum-edit dynamic program has no transition")
        operation = next(op for op in _TIE_BREAK if op in transitions)
        if operation == "MATCH":
            i += 1
            j += 1
        elif operation == "SUB":
            primitive.append(
                EditAction("SUB", j, ref=source[i], alt=candidate[j])
            )
            i += 1
            j += 1
        elif operation == "DEL":
            primitive.append(EditAction("DEL", j, ref=source[i]))
            i += 1
        else:
            primitive.append(EditAction("INS", j, alt=candidate[j]))
            j += 1
    return _merge_adjacent(primitive)


def _single_indel_anchor_count(source: str, candidate: str, distance: int) -> int:
    """Count exact coordinate anchors for a pure contiguous minimal indel."""

    delta = len(source) - len(candidate)
    if delta > 0 and delta == distance:
        return sum(
            source[:position] + source[position + delta :] == candidate
            for position in range(len(source) - delta + 1)
        )
    if delta < 0 and -delta == distance:
        width = -delta
        return sum(
            candidate[:position] + candidate[position + width :] == source
            for position in range(len(candidate) - width + 1)
        )
    return 0


def _optimal_graph_has_indel(
    source: str, candidate: str, cost: list[list[int]]
) -> bool:
    pending = [(0, 0)]
    visited: set[tuple[int, int]] = set()
    while pending:
        i, j = pending.pop()
        if (i, j) in visited:
            continue
        visited.add((i, j))
        for operation in _optimal_transitions(source, candidate, cost, i, j):
            if operation in {"INS", "DEL"}:
                return True
            if operation in {"MATCH", "SUB"}:
                pending.append((i + 1, j + 1))
    return False


def canonical_edit_script(source: str, candidate: str) -> list[EditAction]:
    """Return the deterministic minimum-edit canonical endpoint script."""

    _validate_rna(source, name="source")
    _validate_rna(candidate, name="candidate")
    cost, _ = _levenshtein_tables(source, candidate)
    actions = _canonical_actions(source, candidate, cost)
    if apply_edit_script(source, actions) != candidate:
        raise AssertionError("internal error: canonical edit script does not roundtrip")
    return actions


def canonicalize_edit_script(source: str, candidate: str) -> dict[str, Any]:
    """Canonicalize an endpoint pair and quantify minimum-alignment ambiguity."""

    _validate_rna(source, name="source")
    _validate_rna(candidate, name="candidate")
    cost, count = _levenshtein_tables(source, candidate)
    actions = _canonical_actions(source, candidate, cost)
    if apply_edit_script(source, actions) != candidate:
        raise AssertionError("internal error: canonical edit script does not roundtrip")

    equivalent_count = count[0][0]
    anchor_count = _single_indel_anchor_count(
        source, candidate, cost[0][0]
    )
    has_indel = _optimal_graph_has_indel(source, candidate, cost)
    if equivalent_count == 1:
        category = "unique"
    elif anchor_count > 1:
        category = "repeat_indel_coordinate"
    elif has_indel:
        category = "alignment_tie_with_indel"
    else:
        category = "multiple_optimal_alignments"

    return {
        "actions": [action.to_dict() for action in actions],
        "minimal_edit_count": cost[0][0],
        "canonical_action_count": len(actions),
        "equivalent_minimal_script_count": equivalent_count,
        "path_ambiguity": equivalent_count > 1,
        "ambiguity_category": category,
        "ambiguity_categories": [] if category == "unique" else [category],
        "indel_coordinate_anchor_count": anchor_count,
        "coordinate_system": "0_based_dynamic_state",
        "indel_anchor": "rightmost_under_match_first_tie_break",
        "canonical_tie_break": list(_TIE_BREAK),
        "count_scope": "minimum_cost_character_alignments",
    }


def analyze_edit_script_ambiguity(source: str, candidate: str) -> dict[str, Any]:
    """Compatibility alias returning the complete canonicalization audit."""

    return canonicalize_edit_script(source, candidate)


__all__ = [
    "ACTION_TYPES",
    "RNA_ALPHABET",
    "EditAction",
    "EditScriptError",
    "analyze_edit_script_ambiguity",
    "apply_edit_script",
    "canonical_edit_script",
    "canonicalize_edit_script",
]
