"""Latent optimal-alignment coupling and independent switch-clock path.

The objects in this module are training-only auxiliary variables.  They are
never accepted by a rate-field or sampler interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Iterable, Optional, Sequence

from .schedule import cubic_schedule, linear_schedule
from .types import ActionType, AtomicAction

BLANK = "ε"


@dataclass(frozen=True)
class AlignmentColumn:
    source_token: str
    target_token: str
    source_index: Optional[int]
    target_index: Optional[int]


@dataclass(frozen=True)
class CouplingAlignment:
    source: str
    target: str
    columns: tuple[AlignmentColumn, ...]
    cost: int
    coupling_type: str
    tie_break_rule: str
    path_is_observed: bool = False
    path_semantics: str = "latent_algorithmic"
    algorithm_version: str = "unit-levenshtein-mk0-v1"

    @property
    def alignment_hash(self) -> str:
        payload = [
            (c.source_token, c.target_token, c.source_index, c.target_index)
            for c in self.columns
        ]
        return hashlib.sha256(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def _distance_table(source: str, target: str) -> list[list[int]]:
    if any(token not in "ACGU" for token in source + target):
        raise ValueError("alignment inputs must use only A,C,G,U")
    n, m = len(source), len(target)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            substitution = dp[i - 1][j - 1] + (source[i - 1] != target[j - 1])
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, substitution)
    return dp


def _optimal_choices(
    source: str, target: str, dp: list[list[int]], i: int, j: int
) -> list[str]:
    choices: list[str] = []
    if i and j and source[i - 1] == target[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
        choices.append("MATCH")
    if i and dp[i][j] == dp[i - 1][j] + 1:
        choices.append("DEL")
    if j and dp[i][j] == dp[i][j - 1] + 1:
        choices.append("INS")
    if i and j and source[i - 1] != target[j - 1] and dp[i][j] == dp[i - 1][j - 1] + 1:
        choices.append("SUB")
    return choices


def _trace(
    source: str,
    target: str,
    choices: Sequence[str],
) -> CouplingAlignment:
    i, j = len(source), len(target)
    reverse_columns: list[AlignmentColumn] = []
    for choice in choices:
        if choice in ("MATCH", "SUB"):
            reverse_columns.append(
                AlignmentColumn(source[i - 1], target[j - 1], i - 1, j - 1)
            )
            i -= 1
            j -= 1
        elif choice == "DEL":
            reverse_columns.append(AlignmentColumn(source[i - 1], BLANK, i - 1, None))
            i -= 1
        elif choice == "INS":
            reverse_columns.append(AlignmentColumn(BLANK, target[j - 1], None, j - 1))
            j -= 1
        else:  # pragma: no cover
            raise RuntimeError(f"unknown traceback choice: {choice}")
    if i or j:
        raise RuntimeError("traceback did not reach alignment origin")
    columns = tuple(reversed(reverse_columns))
    cost = sum(c.source_token != c.target_token for c in columns)
    return CouplingAlignment(
        source=source,
        target=target,
        columns=columns,
        cost=cost,
        coupling_type="canonical_optimal",
        tie_break_rule="MATCH>DEL>INS>SUB (backtrace)",
    )


def build_alignment(source: str, target: str) -> CouplingAlignment:
    """Build the deterministic unit-cost optimal alignment."""

    dp = _distance_table(source, target)
    i, j = len(source), len(target)
    trace: list[str] = []
    while i or j:
        choices = _optimal_choices(source, target, dp, i, j)
        if not choices:
            raise RuntimeError(f"alignment traceback dead end at {(i, j)}")
        choice = choices[0]
        trace.append(choice)
        if choice in ("MATCH", "SUB"):
            i -= 1
            j -= 1
        elif choice == "DEL":
            i -= 1
        else:
            j -= 1
    return _trace(source, target, trace)


def _path_counts(source: str, target: str, dp: list[list[int]]) -> list[list[int]]:
    counts = [[0] * (len(target) + 1) for _ in range(len(source) + 1)]
    counts[0][0] = 1
    for i in range(len(source) + 1):
        for j in range(len(target) + 1):
            if i == 0 and j == 0:
                continue
            counts[i][j] = sum(
                counts[i - (choice in ("MATCH", "SUB", "DEL"))][
                    j - (choice in ("MATCH", "SUB", "INS"))
                ]
                for choice in _optimal_choices(source, target, dp, i, j)
            )
    return counts


def sample_optimal_alignment(
    source: str, target: str, *, rng: random.Random
) -> CouplingAlignment:
    """Uniformly sample one optimal traceback using exact suffix counts."""

    dp = _distance_table(source, target)
    counts = _path_counts(source, target, dp)
    i, j = len(source), len(target)
    trace: list[str] = []
    while i or j:
        choices = _optimal_choices(source, target, dp, i, j)
        weights = []
        for choice in choices:
            pi = i - int(choice in ("MATCH", "SUB", "DEL"))
            pj = j - int(choice in ("MATCH", "SUB", "INS"))
            weights.append(counts[pi][pj])
        choice = rng.choices(choices, weights=weights, k=1)[0]
        trace.append(choice)
        i -= int(choice in ("MATCH", "SUB", "DEL"))
        j -= int(choice in ("MATCH", "SUB", "INS"))
    result = _trace(source, target, trace)
    return CouplingAlignment(
        **{
            **result.__dict__,
            "coupling_type": "sampled_optimal_sensitivity",
            "tie_break_rule": "uniform-by-optimal-path-count",
        }
    )


def reconstruct_alignment(alignment: CouplingAlignment) -> tuple[str, str]:
    return (
        "".join(c.source_token for c in alignment.columns if c.source_token != BLANK),
        "".join(c.target_token for c in alignment.columns if c.target_token != BLANK),
    )


def alignment_actions(alignment: CouplingAlignment) -> tuple[AtomicAction, ...]:
    """Map an augmented alignment to deterministic current-coordinate actions."""

    actions: list[AtomicAction] = []
    cursor = 0
    for column in alignment.columns:
        if column.source_token == column.target_token:
            cursor += 1
        elif column.source_token == BLANK:
            actions.append(AtomicAction(ActionType.INS, cursor, column.target_token))
            cursor += 1
        elif column.target_token == BLANK:
            actions.append(AtomicAction(ActionType.DEL, cursor))
        else:
            actions.append(AtomicAction(ActionType.SUB, cursor, column.target_token))
            cursor += 1
    return tuple(actions)


def changed_indices(alignment: CouplingAlignment) -> tuple[int, ...]:
    return tuple(
        i
        for i, column in enumerate(alignment.columns)
        if column.source_token != column.target_token
    )


def sample_switch_clocks(
    alignment: CouplingAlignment,
    *,
    rng: random.Random,
    schedule: str = "cubic",
) -> dict[int, float]:
    clocks: dict[int, float] = {}
    for index in changed_indices(alignment):
        uniform = rng.random()
        clocks[index] = uniform ** (1.0 / 3.0) if schedule == "cubic" else uniform
    if schedule not in ("cubic", "linear"):
        raise ValueError(f"unknown schedule: {schedule}")
    return clocks


def remaining_switches(
    alignment: CouplingAlignment, clocks: dict[int, float], t: float
) -> tuple[int, ...]:
    expected = set(changed_indices(alignment))
    if set(clocks) != expected:
        raise ValueError("clock set must equal changed alignment coordinates")
    return tuple(index for index in sorted(expected) if clocks[index] > t)


def switched_alignment_state(
    alignment: CouplingAlignment, clocks: dict[int, float], t: float
) -> tuple[str, ...]:
    return tuple(
        (
            column.target_token
            if index in clocks and clocks[index] <= t
            else column.source_token
        )
        for index, column in enumerate(alignment.columns)
    )


def joint_path_probability(
    alignment: CouplingAlignment, z: Sequence[str], t: float, *, schedule: str = "cubic"
) -> float:
    """Evaluate the frozen product path p_t(z | z_src, z_tar)."""

    if len(z) != len(alignment.columns):
        raise ValueError("augmented state length differs from alignment")
    if schedule == "cubic":
        kappa, _ = cubic_schedule(t)
    elif schedule == "linear":
        kappa, _ = linear_schedule(t)
    else:
        raise ValueError(f"unknown schedule: {schedule}")
    probability = 1.0
    for value, column in zip(z, alignment.columns):
        if column.source_token == column.target_token:
            factor = 1.0 if value == column.source_token else 0.0
        else:
            factor = (
                1.0 - kappa
                if value == column.source_token
                else kappa if value == column.target_token else 0.0
            )
        probability *= factor
    return probability


def coupling_manifest_record(
    alignment: CouplingAlignment, *, source_id: str, target_id: str
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "target_id": target_id,
        "coupling_type": alignment.coupling_type,
        "alignment_algorithm_version": alignment.algorithm_version,
        "alignment_cost": alignment.cost,
        "tie_break_rule": alignment.tie_break_rule,
        "alignment_hash": alignment.alignment_hash,
        "path_is_observed": False,
        "path_semantics": "latent_algorithmic",
    }
