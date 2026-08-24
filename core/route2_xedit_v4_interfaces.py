"""Frozen cross-module behavior interfaces for the Route 2 V4 method."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, TypedDict

import torch


class XEditV4InterfaceError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditV4InterfaceError(message)


class CriticStateBatchV4(TypedDict):
    """One paired source/candidate physical batch consumed by Critic V4."""

    record_ids: list[str]
    cache_record_ids: list[str]
    source_groups: list[str]
    task_ids: list[str]
    source_tokens: torch.Tensor
    candidate_tokens: torch.Tensor
    padding_mask: torch.Tensor
    edit_padding_mask: torch.Tensor
    source_edit_base_ids: torch.Tensor
    candidate_edit_base_ids: torch.Tensor
    normalized_edit_positions: torch.Tensor
    edit_positions: torch.Tensor
    study_ids: torch.Tensor
    assay_ids: torch.Tensor
    context_ids: torch.Tensor
    quantity_ids: torch.Tensor
    measurement_ids: torch.Tensor
    numerator_ids: torch.Tensor
    denominator_ids: torch.Tensor
    region_ids: torch.Tensor
    target: torch.Tensor
    scaled_target: torch.Tensor
    target_scale: torch.Tensor
    sample_weight: torch.Tensor
    chunk_hidden: torch.Tensor
    chunk_attention_mask: torch.Tensor
    record_edit_offsets: torch.Tensor
    record_source_global: torch.Tensor
    record_candidate_global: torch.Tensor
    edit_source_chunk_indices: torch.Tensor
    edit_candidate_chunk_indices: torch.Tensor
    edit_source_token_centers: torch.Tensor
    edit_candidate_token_centers: torch.Tensor
    edit_source_window_starts: torch.Tensor
    edit_source_window_ends: torch.Tensor
    edit_candidate_window_starts: torch.Tensor
    edit_candidate_window_ends: torch.Tensor


@dataclass(frozen=True)
class CriticPredictionV4:
    """Study-neutral three-member standardized prediction and reward."""

    per_seed_predictions: tuple[tuple[int, float], ...]
    ensemble_mean: float
    ensemble_sd: float
    standardized_reward: float
    study_neutral: bool

    @classmethod
    def from_seed_predictions(
        cls,
        predictions: Mapping[int, float],
        *,
        uncertainty_penalty_kappa: float,
        required_seeds: tuple[int, int, int] = (20260908, 20260909, 20260910),
    ) -> "CriticPredictionV4":
        _require(
            tuple(sorted(predictions)) == required_seeds,
            "CriticPredictionV4 does not contain the exact frozen three seeds",
        )
        values = tuple(float(predictions[seed]) for seed in required_seeds)
        _require(
            all(math.isfinite(value) for value in values),
            "CriticPredictionV4 contains a nonfinite member prediction",
        )
        kappa = float(uncertainty_penalty_kappa)
        _require(
            math.isfinite(kappa) and kappa >= 0.0,
            "CriticPredictionV4 uncertainty penalty is invalid",
        )
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        sd = math.sqrt(variance)
        return cls(
            per_seed_predictions=tuple(zip(required_seeds, values, strict=True)),
            ensemble_mean=mean,
            ensemble_sd=sd,
            standardized_reward=mean - kappa * sd,
            study_neutral=True,
        )

    def to_artifact(self) -> dict[str, Any]:
        _require(self.study_neutral is True, "CriticPredictionV4 is not study-neutral")
        return {
            "per_seed_predictions": {
                str(seed): prediction
                for seed, prediction in self.per_seed_predictions
            },
            "ensemble_mean": self.ensemble_mean,
            "ensemble_sd": self.ensemble_sd,
            "standardized_reward": self.standardized_reward,
            "study_neutral": True,
        }


class SetFlowSourceBatchV4(TypedDict):
    """One source-level 32-state batch consumed by XEditSetFlow V4."""

    source_ids: list[str]
    record_ids: list[str]
    task_ids: list[str]
    state_kinds: list[str]
    source_tokens: torch.Tensor
    current_tokens: torch.Tensor
    padding_mask: torch.Tensor
    source_pretrained_tokens: torch.Tensor
    remaining_budget: torch.Tensor
    quantity_ids: torch.Tensor
    measurement_ids: torch.Tensor
    numerator_ids: torch.Tensor
    denominator_ids: torch.Tensor
    assay_ids: torch.Tensor
    context_ids: torch.Tensor
    region_ids: torch.Tensor
    common_positive_action_mask: torch.Tensor
    candidate_positive_action_mask: torch.Tensor
    candidate_valid_mask: torch.Tensor
    remaining_count_soft_target: torch.Tensor
    structural_budget_exhausted: torch.Tensor
    sample_weight: torch.Tensor


@dataclass(frozen=True)
class MixtureSetMarginalTargetV4:
    """Per-source union, per-candidate, count, and terminal target bundle."""

    common_positive_action_mask: torch.Tensor
    candidate_positive_action_mask: torch.Tensor
    candidate_valid_mask: torch.Tensor
    remaining_count_soft_target: torch.Tensor
    structural_budget_exhausted: torch.Tensor

    @classmethod
    def from_source_batch(
        cls, batch: Mapping[str, Any]
    ) -> "MixtureSetMarginalTargetV4":
        target = cls(
            common_positive_action_mask=batch["common_positive_action_mask"],
            candidate_positive_action_mask=batch["candidate_positive_action_mask"],
            candidate_valid_mask=batch["candidate_valid_mask"],
            remaining_count_soft_target=batch["remaining_count_soft_target"],
            structural_budget_exhausted=batch["structural_budget_exhausted"],
        )
        batch_size, action_count = target.common_positive_action_mask.shape
        _require(
            target.candidate_positive_action_mask.ndim == 3
            and target.candidate_positive_action_mask.shape[0] == batch_size
            and target.candidate_positive_action_mask.shape[2] == action_count,
            "MixtureSetMarginalTargetV4 candidate action geometry changed",
        )
        _require(
            target.candidate_valid_mask.shape
            == target.candidate_positive_action_mask.shape[:2],
            "MixtureSetMarginalTargetV4 candidate validity geometry changed",
        )
        _require(
            target.remaining_count_soft_target.shape == (batch_size, 6),
            "MixtureSetMarginalTargetV4 remaining-count geometry changed",
        )
        _require(
            target.structural_budget_exhausted.shape == (batch_size,),
            "MixtureSetMarginalTargetV4 structural-terminal geometry changed",
        )
        return target


class SetFlowCheckpointDecisionV4(TypedDict):
    """The sole pass-4/6/8/10 outcome-free checkpoint decision."""

    eligible_checkpoint_passes: list[int]
    generation_constrained_selected_checkpoint: dict[str, Any] | None
    nll_selected_checkpoint: dict[str, Any]
    nll_only_selection_differs: bool
