"""Structured edit coupling prior for SetFlow V5 (Direction D prep).

The V4 common set-marginal objective weights every legal action equally in
its coupling, but measured edit statistics are strongly non-uniform over
(position, alt_base).  FlexFlow's structured coupling encodes domain
preferences into the source distribution without touching the flow
objective.  This module computes an outcome-free empirical edit prior from
substitutions-only (source, candidate) pairs and exposes a normalized
log-prior over per-position substitution actions plus STOP, suitable as an
additive rate bias or a coupling weight in a V5 family.  CPU-testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

BASES_V5 = ("A", "C", "G", "U")
DEFAULT_STOP_PROBABILITY_V5 = 0.1
DEFAULT_SMOOTHING_COUNT_V5 = 1.0


class XEditSetFlowEditCouplingV5Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditSetFlowEditCouplingV5Error(message)


def _sanitize(sequence: str) -> str:
    cleaned = "".join(str(sequence).split()).upper().replace("T", "U")
    _require(
        bool(cleaned) and all(base in "ACGU" for base in cleaned),
        "edit coupling sequence contains non-ACGU characters",
    )
    return cleaned


@dataclass(frozen=True)
class EmpiricalEditPriorV5:
    """Per-position, per-alt-base substitution counts plus STOP statistics."""

    sequence_length: int
    substitution_counts: tuple[tuple[float, ...], ...]
    total_edits: float
    total_pairs: float
    stop_probability: float

    @property
    def total_substitution_mass(self) -> float:
        return sum(
            sum(row) for row in self.substitution_counts
        )


def empirical_edit_prior_v5(
    pairs: Sequence[tuple[str, str]],
    *,
    stop_probability: float = DEFAULT_STOP_PROBABILITY_V5,
    smoothing_count: float = DEFAULT_SMOOTHING_COUNT_V5,
) -> EmpiricalEditPriorV5:
    """Count (position, alt_base) substitutions over equal-length pairs."""

    _require(bool(pairs), "edit prior batch is empty")
    _require(
        math.isfinite(stop_probability) and 0.0 < stop_probability < 1.0,
        "STOP probability must be inside (0, 1)",
    )
    _require(
        math.isfinite(smoothing_count) and smoothing_count > 0.0,
        "smoothing count must be positive",
    )
    lengths = {len(_sanitize(source)) for source, _ in pairs}
    _require(
        len(lengths) == 1,
        "edit prior requires a single frozen sequence length",
    )
    length = lengths.pop()
    counts = [[smoothing_count for _ in BASES_V5] for _ in range(length)]
    total_edits = 0.0
    for source, candidate in pairs:
        left = _sanitize(source)
        right = _sanitize(candidate)
        _require(
            len(left) == len(right),
            "edit prior pairs must be substitutions-only and equal length",
        )
        for position in range(length):
            if left[position] != right[position]:
                counts[position][BASES_V5.index(right[position])] += 1.0
                total_edits += 1.0
    return EmpiricalEditPriorV5(
        sequence_length=length,
        substitution_counts=tuple(tuple(row) for row in counts),
        total_edits=total_edits,
        total_pairs=float(len(pairs)),
        stop_probability=stop_probability,
    )


def edit_prior_log_rates_v5(prior: EmpiricalEditPriorV5) -> tuple[tuple[float, ...], ...]:
    """Unnormalized log-counts per (position, alt_base); STOP handled separately."""

    rows: list[tuple[float, ...]] = []
    for row in prior.substitution_counts:
        rows.append(tuple(math.log(count) for count in row))
    _require(
        all(math.isfinite(value) for row in rows for value in row),
        "edit prior log rates are nonfinite",
    )
    return tuple(rows)


def prior_scaled_substitution_rates_v5(
    prior: EmpiricalEditPriorV5,
) -> tuple[tuple[float, ...], ...]:
    """Renormalized per-position substitution probabilities scaled to (1 - p_stop)."""

    substitution_mass = 1.0 - prior.stop_probability
    rows: list[tuple[float, ...]] = []
    for row in prior.substitution_counts:
        total = sum(row)
        rows.append(tuple(substitution_mass * count / total for count in row))
    _require(
        all(
            abs(sum(row) - substitution_mass) <= 1e-9 for row in rows
        ),
        "scaled substitution rows must each sum to (1 - p_stop)",
    )
    return tuple(rows)


def prior_stop_rate_v5(prior: EmpiricalEditPriorV5) -> float:
    """The frozen STOP probability carried by the coupling."""

    return prior.stop_probability
