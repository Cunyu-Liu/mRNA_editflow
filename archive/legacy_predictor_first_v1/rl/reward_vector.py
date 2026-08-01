"""Serializable, auditable vector rewards for offline proposal teachers.

Hard constraints are represented for audit but are deliberately excluded from
preference scalarization.  They must be enforced by action masks/rejection
gates before a reward vector is considered.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Sequence


REWARD_SCHEMA_VERSION = 1
HARD_CONSTRAINT_CATEGORY = "hard_constraint"
SOFT_CATEGORIES = {
    "functional", "stability", "structure", "manufacturability", "safety", "edit_cost",
}


@dataclass(frozen=True)
class RewardComponent:
    name: str
    value: float
    source_model: str
    category: str
    independent: bool
    uncertainty: Optional[float]
    valid: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RewardComponent":
        return cls(
            name=str(payload["name"]),
            value=float(payload.get("value", 0.0)),
            source_model=str(payload.get("source_model", "unknown")),
            category=str(payload.get("category", "functional")),
            independent=bool(payload.get("independent", True)),
            uncertainty=(None if payload.get("uncertainty") is None else float(payload["uncertainty"])),
            valid=bool(payload.get("valid", True)),
        )


@dataclass(frozen=True)
class RewardVector:
    """Absolute levels and deltas for one legal action, including STOP."""

    raw_absolute_level: Mapping[str, float]
    raw_delta_from_source: Mapping[str, float]
    normalized_within_group: Mapping[str, float]
    components: Sequence[RewardComponent]
    validity: Mapping[str, bool]
    redundancy_groups: Sequence[str] = ()
    schema_version: int = REWARD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": int(self.schema_version),
            "raw_absolute_level": {key: float(value) for key, value in self.raw_absolute_level.items()},
            "raw_delta_from_source": {key: float(value) for key, value in self.raw_delta_from_source.items()},
            "normalized_within_group": {key: float(value) for key, value in self.normalized_within_group.items()},
            "components": [component.to_dict() for component in self.components],
            "validity": {key: bool(value) for key, value in self.validity.items()},
            "redundancy_groups": list(self.redundancy_groups),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RewardVector":
        raw_components = payload.get("components", [])
        components = [
            RewardComponent.from_dict(item)
            for item in raw_components
            if isinstance(item, Mapping)
        ]
        return cls(
            raw_absolute_level={str(k): float(v) for k, v in dict(payload.get("raw_absolute_level", {})).items()},
            raw_delta_from_source={str(k): float(v) for k, v in dict(payload.get("raw_delta_from_source", {})).items()},
            normalized_within_group={str(k): float(v) for k, v in dict(payload.get("normalized_within_group", {})).items()},
            components=components,
            validity={str(k): bool(v) for k, v in dict(payload.get("validity", {})).items()},
            redundancy_groups=tuple(str(item) for item in payload.get("redundancy_groups", ())),
            schema_version=int(payload.get("schema_version", REWARD_SCHEMA_VERSION)),
        )

    @classmethod
    def from_legacy(
        cls,
        objective_deltas: Mapping[str, float],
        source_scores: Mapping[str, float],
    ) -> "RewardVector":
        """Read legacy scalar-teacher fields without inventing independence."""
        components = [
            RewardComponent(
                name=str(name), value=float(value), source_model="legacy_teacher",
                category="functional", independent=True, uncertainty=None, valid=True,
            )
            for name, value in objective_deltas.items()
        ]
        return cls(
            raw_absolute_level={}, raw_delta_from_source=dict(objective_deltas),
            normalized_within_group=dict(source_scores), components=components,
            validity={str(name): True for name in objective_deltas},
        )


def hard_constraints_valid(vector: RewardVector) -> bool:
    """Return false when any hard constraint is invalid; never scalarize it."""
    return all(
        component.valid
        for component in vector.components
        if component.category == HARD_CONSTRAINT_CATEGORY
    )


def stop_reward_vector(component_names: Sequence[str]) -> RewardVector:
    """Construct STOP with mandated zero raw deltas and valid hard constraints."""
    zeros = {str(name): 0.0 for name in component_names}
    return RewardVector(
        raw_absolute_level={}, raw_delta_from_source=zeros,
        normalized_within_group=zeros,
        components=[
            RewardComponent(str(name), 0.0, "stop_baseline", "edit_cost", True, None, True)
            for name in component_names
        ],
        validity={str(name): True for name in component_names},
    )


__all__ = [
    "REWARD_SCHEMA_VERSION", "HARD_CONSTRAINT_CATEGORY", "RewardComponent", "RewardVector",
    "hard_constraints_valid", "stop_reward_vector",
]
