"""Mode-conditioned TRAIN-only soft-value targets for XEditFlow V4."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

from core.route2_source_token_cache_v3 import SourceTokenCacheIndexV3
from core.route2_xeditflow_gate_v4 import authorize_xeditflow_guidance_v4
from core.route2_xeditflow_guidance_v3 import (
    soft_value_target_v3,
    uncertainty_penalized_reward_v3,
)
from core.route2_xeditflow_guidance_v4 import XEditValueV4


class XEditFlowValueTrainingV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowValueTrainingV4Error(message)


VALUE_TARGET_SCHEMA_V4 = "route_a_v3_route2_xeditflow_value_targets.v4"
VALUE_CHECKPOINT_SCHEMA_V4 = "route_a_v3_route2_xeditflow_value_checkpoint.v4"
CRITIC_SEEDS_V4 = (20260908, 20260909, 20260910)
BASE_FLOW_SEEDS_V4 = (20260912, 20260913, 20260914)
MODE_IDS_V4 = tuple(range(8))


def require_value_training_authorization_v4(
    critic_readiness: Mapping[str, Any],
    setflow_confirmation: Mapping[str, Any],
) -> dict[str, Any]:
    authorization = authorize_xeditflow_guidance_v4(
        critic_readiness, setflow_confirmation
    )
    _require(
        authorization["guidance_authorized"] is True,
        "V4 value targets remain blocked until both frozen readiness gates",
    )
    return authorization


def _validate_state_row_v4(
    row: Mapping[str, Any], *, base_flow_seed: int
) -> None:
    _require(row.get("split") == "TRAIN", "non-TRAIN state entered V4 value targets")
    _require(
        int(row.get("base_flow_training_seed", -1)) == base_flow_seed,
        "V4 value state base-flow seed differs",
    )
    _require(
        int(row.get("trajectory_mode_id", -1)) in MODE_IDS_V4,
        "V4 value state trajectory mode differs",
    )
    source = str(row.get("source_sequence", ""))
    current = str(row.get("current_sequence", ""))
    _require(
        bool(source) and len(source) == len(current),
        "V4 value state sequence geometry differs",
    )
    _require(
        set(source) <= set("ACGU") and set(current) <= set("ACGU"),
        "V4 value state alphabet differs",
    )
    edited = sum(left != right for left, right in zip(source, current))
    assigned = int(row.get("assigned_budget", -1))
    remaining = int(row.get("remaining_budget", -1))
    _require(assigned in {1, 3, 5}, "V4 value state assigned budget differs")
    _require(
        0 <= remaining <= assigned and edited + remaining == assigned,
        "V4 value state progress differs",
    )
    _require(
        bool(str(row.get("cache_record_id", ""))),
        "V4 value state cache identity is absent",
    )
    for field in (
        "quantity_id",
        "measurement_id",
        "numerator_id",
        "denominator_id",
        "assay_id",
        "context_id",
        "region_id",
    ):
        _require(
            int(row.get(field, -1)) >= 0,
            f"V4 value state endpoint field is invalid: {field}",
        )
    _require(
        row.get("independent_evaluator_used") is False,
        "independent evaluator entered a V4 value state",
    )
    _require(
        row.get("development_test_outcomes_accessed_after_atomic_test") is False,
        "V4 value state reopened Development TEST outcomes",
    )
    _require(
        row.get("new_final_evaluation_outcomes_accessed") is False,
        "V4 value state accessed Evaluation outcome",
    )


def assemble_value_targets_v4(
    states: Sequence[Mapping[str, Any]],
    rollouts: Sequence[Mapping[str, Any]],
    *,
    critic_readiness: Mapping[str, Any],
    setflow_confirmation: Mapping[str, Any],
    base_flow_training_seed: int,
    kappa: float,
    temperature: float,
) -> dict[str, Any]:
    """Build K=8 targets for explicit TRAIN state-mode pairs."""

    require_value_training_authorization_v4(
        critic_readiness, setflow_confirmation
    )
    _require(
        base_flow_training_seed in BASE_FLOW_SEEDS_V4,
        "undeclared V4 base-flow training seed",
    )
    _require(bool(states), "V4 value target state cohort is empty")
    by_state: dict[str, Mapping[str, Any]] = {}
    for row in states:
        _validate_state_row_v4(row, base_flow_seed=base_flow_training_seed)
        state_id = str(row.get("state_id", ""))
        _require(
            bool(state_id) and state_id not in by_state,
            "V4 value state identity is empty or duplicated",
        )
        by_state[state_id] = row
    rollout_groups: dict[str, dict[int, Mapping[str, Any]]] = {
        state_id: {} for state_id in by_state
    }
    for row in rollouts:
        state_id = str(row.get("state_id", ""))
        _require(
            state_id in by_state,
            "V4 value rollout is outside the TRAIN state-mode cohort",
        )
        rollout_index = int(row.get("rollout_index", -1))
        _require(
            0 <= rollout_index < 8
            and rollout_index not in rollout_groups[state_id],
            "V4 value rollout index is invalid or duplicated",
        )
        _require(
            int(row.get("base_flow_training_seed", -1))
            == base_flow_training_seed,
            "V4 value rollout base-flow seed differs",
        )
        _require(
            int(row.get("trajectory_mode_id", -1))
            == int(by_state[state_id]["trajectory_mode_id"]),
            "V4 value rollout changed the state trajectory mode",
        )
        _require(
            tuple(int(value) for value in row.get("critic_seeds", ()))
            == CRITIC_SEEDS_V4,
            "V4 value rollout critic ensemble differs",
        )
        predictions = row.get("calibrated_seed_predictions")
        _require(
            isinstance(predictions, Sequence) and len(predictions) == 3,
            "V4 value rollout lacks three critic predictions",
        )
        _require(
            all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in predictions
            ),
            "V4 value rollout critic prediction is invalid",
        )
        _require(
            row.get("study_neutral") is True,
            "V4 value rollout is not study-neutral",
        )
        _require(
            row.get("independent_evaluator_used") is False,
            "independent evaluator entered V4 value targets",
        )
        _require(
            row.get("development_test_outcomes_accessed_after_atomic_test")
            is False,
            "V4 value rollout reopened Development TEST outcomes",
        )
        _require(
            row.get("new_final_evaluation_outcomes_accessed") is False,
            "V4 value rollout accessed Evaluation outcome",
        )
        rollout_groups[state_id][rollout_index] = row
    _require(
        all(set(group) == set(range(8)) for group in rollout_groups.values()),
        "every V4 state-mode must have exactly rollout indices 0..7",
    )
    ordered_ids = sorted(by_state)
    prediction_tensor = torch.tensor(
        [
            [
                rollout_groups[state_id][index]["calibrated_seed_predictions"]
                for index in range(8)
            ]
            for state_id in ordered_ids
        ],
        dtype=torch.float64,
    )
    rewards = uncertainty_penalized_reward_v3(
        prediction_tensor.reshape(-1, 3), kappa=float(kappa)
    ).reshape(len(ordered_ids), 8)
    targets = soft_value_target_v3(rewards, temperature=float(temperature))
    records: list[dict[str, Any]] = []
    for row_index, state_id in enumerate(ordered_ids):
        state = dict(by_state[state_id])
        state.pop("development_test_outcomes_accessed_after_atomic_test", None)
        state.pop("new_final_evaluation_outcomes_accessed", None)
        records.append(
            {
                **state,
                "rollout_count": 8,
                "critic_seeds": list(CRITIC_SEEDS_V4),
                "rollout_rewards": rewards[row_index].tolist(),
                "soft_value_target": float(targets[row_index]),
            }
        )
    return {
        "schema_version": VALUE_TARGET_SCHEMA_V4,
        "status": "XEDITFLOW_V4_VALUE_TARGETS_COMPLETE",
        "split": "TRAIN",
        "base_flow_training_seed": base_flow_training_seed,
        "critic_seeds": list(CRITIC_SEEDS_V4),
        "kappa": float(kappa),
        "temperature": float(temperature),
        "rollouts_per_state_mode": 8,
        "state_mode_count": len(records),
        "records": records,
        "setflow_mode_is_fixed_trajectory_state": True,
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


@dataclass(frozen=True)
class ValueTargetRecordV4:
    state_id: str
    source_sequence: str
    current_sequence: str
    cache_record_id: str
    remaining_budget: int
    assigned_budget: int
    trajectory_mode_id: int
    quantity_id: int
    measurement_id: int
    numerator_id: int
    denominator_id: int
    assay_id: int
    context_id: int
    region_id: int
    target: float


def value_target_records_v4(
    payload: Mapping[str, Any],
) -> list[ValueTargetRecordV4]:
    _require(
        payload.get("schema_version") == VALUE_TARGET_SCHEMA_V4,
        "unexpected V4 value target schema",
    )
    _require(
        payload.get("status") == "XEDITFLOW_V4_VALUE_TARGETS_COMPLETE",
        "V4 value targets are incomplete",
    )
    _require(payload.get("split") == "TRAIN", "V4 value target split differs")
    _require(
        int(payload.get("rollouts_per_state_mode", -1)) == 8,
        "V4 value target rollout count differs",
    )
    _require(
        payload.get("setflow_mode_is_fixed_trajectory_state") is True,
        "V4 value target artifact did not preserve mode state",
    )
    _require(
        payload.get("independent_evaluator_used") is False,
        "independent evaluator entered V4 value target artifact",
    )
    _require(
        payload.get("development_test_outcomes_accessed_after_atomic_test")
        is False,
        "V4 value target artifact reopened Development TEST",
    )
    _require(
        payload.get("new_final_evaluation_outcomes_accessed") is False,
        "V4 value target artifact accessed Evaluation",
    )
    records: list[ValueTargetRecordV4] = []
    seen: set[str] = set()
    for row in payload.get("records", ()):
        state_id = str(row["state_id"])
        _require(state_id not in seen, "V4 value target record is duplicated")
        seen.add(state_id)
        target = float(row["soft_value_target"])
        mode_id = int(row["trajectory_mode_id"])
        _require(math.isfinite(target), "V4 value target is nonfinite")
        _require(mode_id in MODE_IDS_V4, "V4 value target mode differs")
        records.append(
            ValueTargetRecordV4(
                state_id=state_id,
                source_sequence=str(row["source_sequence"]),
                current_sequence=str(row["current_sequence"]),
                cache_record_id=str(row["cache_record_id"]),
                remaining_budget=int(row["remaining_budget"]),
                assigned_budget=int(row["assigned_budget"]),
                trajectory_mode_id=mode_id,
                quantity_id=int(row["quantity_id"]),
                measurement_id=int(row["measurement_id"]),
                numerator_id=int(row["numerator_id"]),
                denominator_id=int(row["denominator_id"]),
                assay_id=int(row["assay_id"]),
                context_id=int(row["context_id"]),
                region_id=int(row["region_id"]),
                target=target,
            )
        )
    _require(
        len(records) == int(payload.get("state_mode_count", -1)) > 0,
        "V4 value target count differs",
    )
    return records


TOKEN_V4 = {"A": 0, "C": 1, "G": 2, "U": 3}
PAD_V4 = 4


def collate_value_targets_v4(
    records: Sequence[ValueTargetRecordV4],
    *,
    source_cache: SourceTokenCacheIndexV3,
) -> dict[str, torch.Tensor]:
    _require(bool(records), "V4 value target batch is empty")
    maximum = max(len(row.source_sequence) for row in records)
    width = int(source_cache.payload["embedding_width"])
    source = torch.full((len(records), maximum), PAD_V4, dtype=torch.long)
    current = torch.full_like(source, PAD_V4)
    padding = torch.ones_like(source, dtype=torch.bool)
    pretrained = torch.zeros(
        (len(records), maximum, width), dtype=torch.float16
    )
    for index, row in enumerate(records):
        length = len(row.source_sequence)
        _require(
            length == len(row.current_sequence),
            "V4 value target sequence length differs",
        )
        source[index, :length] = torch.tensor(
            [TOKEN_V4[base] for base in row.source_sequence]
        )
        current[index, :length] = torch.tensor(
            [TOKEN_V4[base] for base in row.current_sequence]
        )
        padding[index, :length] = False
        cached = source_cache.tokens_for_record(row.cache_record_id)
        _require(
            cached.shape == (length, width),
            "V4 value target source cache differs",
        )
        pretrained[index, :length] = cached
    return {
        "source_tokens": source,
        "current_tokens": current,
        "padding_mask": padding,
        "source_pretrained_tokens": pretrained,
        "remaining_budget": torch.tensor(
            [row.remaining_budget for row in records]
        ),
        "trajectory_mode_ids": torch.tensor(
            [row.trajectory_mode_id for row in records]
        ),
        "quantity_ids": torch.tensor([row.quantity_id for row in records]),
        "measurement_ids": torch.tensor(
            [row.measurement_id for row in records]
        ),
        "numerator_ids": torch.tensor([row.numerator_id for row in records]),
        "denominator_ids": torch.tensor(
            [row.denominator_id for row in records]
        ),
        "assay_ids": torch.tensor([row.assay_id for row in records]),
        "context_ids": torch.tensor([row.context_id for row in records]),
        "region_ids": torch.tensor([row.region_id for row in records]),
        "soft_value_targets": torch.tensor(
            [row.target for row in records], dtype=torch.float32
        ),
    }


def value_distillation_loss_v4(
    model: XEditValueV4, batch: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    predictions = model(batch)
    targets = batch["soft_value_targets"].to(predictions)
    _require(
        predictions.shape == targets.shape,
        "V4 value prediction and target differ",
    )
    loss = F.huber_loss(predictions, targets, delta=1.0, reduction="mean")
    _require(
        bool(torch.isfinite(loss).item()),
        "V4 value distillation loss is nonfinite",
    )
    return loss
