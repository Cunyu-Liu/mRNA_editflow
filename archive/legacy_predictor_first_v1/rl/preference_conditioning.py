"""Preference-conditioned vector-score combination without hard-constraint tradeoffs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch
from torch import nn

from mrna_editflow.rl.reward_vector import HARD_CONSTRAINT_CATEGORY, RewardVector, hard_constraints_valid


PREFERENCE_PRESETS: dict[str, dict[str, float]] = {
    "balanced": {"te": 1.0, "access": 1.0, "gc_constraint": 1.0, "uaug": 1.0, "cai": 0.0, "edit_cost": 0.5, "uncertainty": 0.5},
    "translation_focused": {"te": 3.0, "access": 1.5, "gc_constraint": 0.5, "uaug": 1.0, "cai": 0.0, "edit_cost": 0.25, "uncertainty": 0.5},
    "stability_focused": {"te": 1.0, "access": 2.0, "gc_constraint": 2.0, "uaug": 1.0, "cai": 0.0, "edit_cost": 0.5, "uncertainty": 1.0},
    "manufacturing_focused": {"te": 1.0, "access": 0.5, "gc_constraint": 3.0, "uaug": 1.0, "cai": 1.0, "edit_cost": 1.0, "uncertainty": 0.5},
}


def normalized_preference(weights: Mapping[str, float]) -> dict[str, float]:
    raw = {str(name): max(0.0, float(value)) for name, value in weights.items()}
    total = sum(raw.values())
    if total <= 0.0:
        raise ValueError("preference weights must contain a positive value")
    return {name: value / total for name, value in raw.items()}


def preference_from_name(name: str, custom_weights: Optional[Mapping[str, float]] = None) -> dict[str, float]:
    if str(name) == "custom":
        if custom_weights is None:
            raise ValueError("custom preference requires custom_weights")
        return normalized_preference(custom_weights)
    if name not in PREFERENCE_PRESETS:
        raise ValueError("unknown preference profile: " + str(name))
    return normalized_preference(PREFERENCE_PRESETS[name])


def sample_dirichlet_preference(alpha: Mapping[str, float], *, generator: Optional[torch.Generator] = None) -> dict[str, float]:
    names = list(alpha)
    concentration = torch.tensor([max(float(alpha[name]), 1e-8) for name in names])
    # torch.distributions does not consistently accept a generator, so seed
    # control remains the caller's torch global-generator responsibility.
    sample = torch.distributions.Dirichlet(concentration).sample()
    return {name: float(sample[index]) for index, name in enumerate(names)}


def combine_reward_vector(
    vector: RewardVector,
    preference: Mapping[str, float],
    *,
    uncertainty_penalty: float = 0.0,
    minimum_oracle_agreement: Optional[float] = None,
    oracle_agreement: Optional[float] = None,
) -> float:
    """Combine valid soft components; invalid hard constraints reject action."""
    if not hard_constraints_valid(vector):
        return float("-inf")
    if minimum_oracle_agreement is not None and oracle_agreement is not None:
        if float(oracle_agreement) < float(minimum_oracle_agreement):
            return float("-inf")
    weights = normalized_preference(preference)
    values = {component.name: component for component in vector.components}
    total = 0.0
    for name, weight in weights.items():
        component = values.get(name)
        if component is None or not component.valid or component.category == HARD_CONSTRAINT_CATEGORY:
            continue
        risk_adjusted = float(component.value) - float(uncertainty_penalty) * float(component.uncertainty or 0.0)
        total += float(weight) * risk_adjusted
    return float(total)


def combine_objective_scores(
    scores: Mapping[str, torch.Tensor], preference: Mapping[str, float]
) -> torch.Tensor:
    """Combine inference-time Q heads for a named or custom preference."""
    weights = normalized_preference(preference)
    available = [name for name in weights if name in scores]
    if not available:
        raise ValueError("preference has no scoreable objectives")
    return sum(float(weights[name]) * scores[name] for name in available)


def select_with_absolute_stop(
    vectors: Sequence[RewardVector],
    preference: Mapping[str, float],
    *,
    uncertainty_penalty: float = 0.0,
    minimum_oracle_agreement: Optional[float] = None,
    oracle_agreements: Optional[Sequence[Optional[float]]] = None,
) -> Optional[int]:
    """Return best candidate index, or ``None`` for STOP if all raw deltas < 0."""
    if not vectors:
        return None
    if all(
        all(float(value) < 0.0 for value in vector.raw_delta_from_source.values())
        for vector in vectors
    ):
        return None
    scores = [
        combine_reward_vector(
            vector, preference,
            uncertainty_penalty=uncertainty_penalty,
            minimum_oracle_agreement=minimum_oracle_agreement,
            oracle_agreement=(oracle_agreements[index] if oracle_agreements else None),
        )
        for index, vector in enumerate(vectors)
    ]
    best = max(range(len(scores)), key=lambda index: (scores[index], -index))
    return None if scores[best] <= 0.0 else best


class PreferenceConditionedObjectiveHead(nn.Module):
    """Independent affine heads over a base action score and operation class."""

    def __init__(self, objective_names: Sequence[str]) -> None:
        super().__init__()
        self.objective_names = tuple(str(name) for name in objective_names)
        self.heads = nn.ModuleDict({name: nn.Linear(4, 1) for name in self.objective_names})

    def forward(self, base_scores: torch.Tensor, operation_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        if base_scores.ndim != 1 or operation_ids.shape != base_scores.shape:
            raise ValueError("base_scores and operation_ids must be aligned 1-D tensors")
        one_hot = torch.nn.functional.one_hot(operation_ids.long().clamp(0, 2), num_classes=3).to(base_scores.dtype)
        features = torch.cat([base_scores.unsqueeze(-1), one_hot], dim=-1)
        return {name: head(features).squeeze(-1) for name, head in self.heads.items()}

    def schema(self) -> dict[str, object]:
        return {"objective_names": list(self.objective_names), "head_type": "independent_affine_action_heads"}


__all__ = [
    "PREFERENCE_PRESETS", "normalized_preference", "preference_from_name", "sample_dirichlet_preference",
    "combine_reward_vector", "combine_objective_scores", "select_with_absolute_stop", "PreferenceConditionedObjectiveHead",
]
