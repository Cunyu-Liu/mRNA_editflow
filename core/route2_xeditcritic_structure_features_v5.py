"""ViennaRNA structure-differential features for the Critic V5 hypothesis.

Direction B of the 2026-08-29 architecture review: edit effects on UTR
function are largely mediated by local secondary-structure changes, but the
V4 critic consumes no structure signal (UTR-LM shows structure supervision
helps).  This module computes an outcome-free, sequence-only fixed-width
feature vector per (source, candidate) pair:

- length-normalized MFE of source / candidate and their delta;
- ensemble free-energy delta;
- edit-site pairing-probability change summary (mean/max absolute delta at
  edited positions, plus a ±4 local window);
- GC-content delta.

Features are deterministic and CPU-only; the V5 family caches them per record
under the /mnt root exactly like the token caches.  No frozen V4 artifact is
touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

STRUCTURE_FEATURE_WIDTH_V5 = 12
LOCAL_PAIRING_WINDOW_V5 = 4


class XEditCriticStructureFeaturesV5Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticStructureFeaturesV5Error(message)


def _sanitize_sequence(sequence: str) -> str:
    cleaned = "".join(str(sequence).split()).upper().replace("T", "U")
    _require(
        bool(cleaned) and all(base in "ACGU" for base in cleaned),
        "structure feature sequence contains non-ACGU characters",
    )
    return cleaned


def gc_content(sequence: str) -> float:
    cleaned = _sanitize_sequence(sequence)
    gc = sum(1 for base in cleaned if base in "GC")
    return gc / len(cleaned)


def _fold_pairing_probabilities(sequence: str) -> tuple[float, float, list[float]]:
    """Return (MFE, ensemble free energy, per-position pairing probability)."""

    import RNA

    compound = RNA.fold_compound(sequence)
    structure, mfe = compound.mfe()
    ensemble_structure, ensemble_energy = compound.pf()
    _require(
        ensemble_structure is not None,
        "ViennaRNA partition function failed",
    )
    probabilities = list(compound.bpp())
    pairing = [0.0] * len(sequence)
    for left in range(1, len(sequence) + 1):
        row = probabilities[left]
        for right in range(left + 1, len(sequence) + 1):
            probability = row[right]
            if probability > 0.0:
                pairing[left - 1] += probability
                pairing[right - 1] += probability
    return float(mfe), float(ensemble_energy), pairing


@dataclass(frozen=True)
class StructureDifferentialFeaturesV5:
    """Fixed-width outcome-free structure feature record."""

    mfe_source: float
    mfe_candidate: float
    delta_mfe: float
    delta_mfe_per_nt: float
    delta_ensemble_energy: float
    delta_ensemble_per_nt: float
    edit_site_pairing_delta_mean: float
    edit_site_pairing_delta_max: float
    local_pairing_delta_mean: float
    local_pairing_delta_max: float
    delta_gc: float
    edit_count: float

    def to_vector(self) -> tuple[float, ...]:
        return (
            self.mfe_source,
            self.mfe_candidate,
            self.delta_mfe,
            self.delta_mfe_per_nt,
            self.delta_ensemble_energy,
            self.delta_ensemble_per_nt,
            self.edit_site_pairing_delta_mean,
            self.edit_site_pairing_delta_max,
            self.local_pairing_delta_mean,
            self.local_pairing_delta_max,
            self.delta_gc,
            self.edit_count,
        )


def _validate_lengths(source: str, candidate: str) -> tuple[str, str]:
    left = _sanitize_sequence(source)
    right = _sanitize_sequence(candidate)
    _require(
        len(left) == len(right),
        "structure differential features require equal-length substitutions-only pairs",
    )
    return left, right


def structure_differential_features_v5(
    source: str,
    candidate: str,
) -> StructureDifferentialFeaturesV5:
    """Compute the Direction-B feature vector for one substitutions-only pair."""

    left, right = _validate_lengths(source, candidate)
    length = len(left)
    edited_positions = [
        index for index in range(length) if left[index] != right[index]
    ]
    mfe_source, ensemble_source, pairing_source = _fold_pairing_probabilities(left)
    mfe_candidate, ensemble_candidate, pairing_candidate = _fold_pairing_probabilities(right)
    pairing_delta = [pairing_candidate[i] - pairing_source[i] for i in range(length)]
    if edited_positions:
        edit_deltas = [abs(pairing_delta[index]) for index in edited_positions]
        local_positions = set()
        for index in edited_positions:
            for offset in range(-LOCAL_PAIRING_WINDOW_V5, LOCAL_PAIRING_WINDOW_V5 + 1):
                neighbor = index + offset
                if 0 <= neighbor < length:
                    local_positions.add(neighbor)
        local_deltas = [abs(pairing_delta[index]) for index in sorted(local_positions)]
    else:
        edit_deltas = [0.0]
        local_deltas = [0.0]
    _require(
        all(delta == 0.0 for delta in pairing_delta) if not edited_positions else True,
        "identical sequences produced nonzero pairing delta",
    )
    return StructureDifferentialFeaturesV5(
        mfe_source=mfe_source / length,
        mfe_candidate=mfe_candidate / length,
        delta_mfe=(mfe_candidate - mfe_source),
        delta_mfe_per_nt=(mfe_candidate - mfe_source) / length,
        delta_ensemble_energy=(ensemble_candidate - ensemble_source),
        delta_ensemble_per_nt=(ensemble_candidate - ensemble_source) / length,
        edit_site_pairing_delta_mean=sum(edit_deltas) / len(edit_deltas),
        edit_site_pairing_delta_max=max(edit_deltas),
        local_pairing_delta_mean=sum(local_deltas) / len(local_deltas),
        local_pairing_delta_max=max(local_deltas),
        delta_gc=gc_content(right) - gc_content(left),
        edit_count=float(len(edited_positions)),
    )


def structure_feature_matrix_v5(
    pairs: Sequence[tuple[str, str]],
) -> tuple[tuple[float, ...], ...]:
    """Row-wise feature matrix for a batch of (source, candidate) pairs."""

    _require(bool(pairs), "structure feature batch is empty")
    matrix = tuple(
        structure_differential_features_v5(source, candidate).to_vector()
        for source, candidate in pairs
    )
    _require(
        all(len(row) == STRUCTURE_FEATURE_WIDTH_V5 for row in matrix),
        "structure feature row width drifted",
    )
    return matrix
