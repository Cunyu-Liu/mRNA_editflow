"""Source-level state-mode rows and replay contracts for XEditFlow V4."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from core.route2_legal_xeditflow import FlowState, validate_state
from core.route2_xeditflow_value_training_v4 import (
    BASE_FLOW_SEEDS_V4,
    CRITIC_SEEDS_V4,
    MODE_IDS_V4,
)
from core.route2_xeditsetflow_sampling_v3 import SetFlowGenerationMetadataV3
from core.route2_xeditsetflow_training_v4 import (
    SetFlowSourceRecordV4,
    SetFlowSourceStateDatasetV4,
)


class XEditFlowValueRolloutV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowValueRolloutV4Error(message)


def value_state_mode_id_v4(source_index: int, state_slot: int) -> int:
    """Frozen balanced assignment for four state slots over eight modes."""

    _require(source_index >= 0, "V4 value source index is negative")
    _require(0 <= state_slot < 4, "V4 value state slot is outside 0..3")
    return (int(source_index) * 4 + int(state_slot)) % 8


def build_value_train_state_rows_v4(
    records: Sequence[SetFlowSourceRecordV4],
    vocabs: Mapping[str, Mapping[str, int]],
    *,
    base_flow_training_seed: int,
    state_pass_index: int = 0,
) -> list[dict[str, Any]]:
    """Build four deterministic TRAIN state-mode rows per unique source."""

    _require(
        base_flow_training_seed in BASE_FLOW_SEEDS_V4,
        "undeclared V4 value base-flow seed",
    )
    _require(state_pass_index == 0, "V4 value state pass changed")
    _require(
        bool(records) and all(record.split == "TRAIN" for record in records),
        "non-TRAIN source entered V4 value states",
    )
    _require(
        [record.source_id for record in records]
        == sorted(record.source_id for record in records),
        "V4 value sources are not in frozen lexicographic source-key order",
    )
    dataset = SetFlowSourceStateDatasetV4(
        records, vocabs, seed=base_flow_training_seed
    )
    dataset.set_pass(state_pass_index)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_index, record in enumerate(records):
        for state_slot in range(4):
            sampled = dataset.state(source_index, state_slot)
            mode_id = value_state_mode_id_v4(source_index, state_slot)
            state_id = (
                f"{record.source_id}:pass{state_pass_index}:slot{state_slot}:"
                f"mode{mode_id}"
            )
            _require(state_id not in seen, "V4 value state identity is duplicated")
            seen.add(state_id)
            selected = tuple(
                (int(position), str(alt))
                for position, alt in sampled["selected_edit_set"]
            )
            rows.append(
                {
                    "schema_version": (
                        "route_a_v3_route2_xeditflow_value_train_state.v4"
                    ),
                    "state_id": state_id,
                    "split": "TRAIN",
                    "base_flow_training_seed": base_flow_training_seed,
                    "state_pass_index": state_pass_index,
                    "source_index": source_index,
                    "state_slot": state_slot,
                    "state_kind": str(sampled["state_kind"]),
                    "source_id": record.source_id,
                    "cache_record_id": record.cache_record_id,
                    "source_group_id": record.source_group,
                    "task_id": record.task,
                    "source_sequence": record.source,
                    "current_sequence": str(sampled["current_sequence"]),
                    "source_relative_edits": [
                        {
                            "position": position,
                            "source_base": record.source[position],
                            "candidate_base": alt,
                        }
                        for position, alt in selected
                    ],
                    "assigned_budget": int(sampled["assigned_budget"]),
                    "remaining_budget": int(sampled["remaining_budget"]),
                    "structural_budget_exhausted": bool(
                        sampled["structural_budget_exhausted"]
                    ),
                    "trajectory_mode_id": mode_id,
                    "quantity_id": int(sampled["quantity"]),
                    "measurement_id": int(sampled["measurement"]),
                    "numerator_id": int(sampled["numerator"]),
                    "denominator_id": int(sampled["denominator"]),
                    "assay_id": int(sampled["assay"]),
                    "context_id": int(sampled["context"]),
                    "region_id": int(sampled["region"]),
                    "endpoint_descriptor": {
                        "quantity_family": record.quantity,
                        "measurement_form": record.measurement,
                        "numerator_family": record.numerator,
                        "denominator_family": record.denominator,
                    },
                    "assay_category": record.assay,
                    "context_category": record.context,
                    "setflow_mode_is_fixed_trajectory_state": True,
                    "independent_evaluator_used": False,
                    "development_test_outcomes_accessed_after_atomic_test": False,
                    "new_final_evaluation_outcomes_accessed": False,
                }
            )
    _require(
        len(rows) == len(records) * 4,
        "V4 value state-mode count changed",
    )
    mode_counts = {mode: 0 for mode in MODE_IDS_V4}
    for row in rows:
        mode_counts[int(row["trajectory_mode_id"])] += 1
    _require(
        max(mode_counts.values()) - min(mode_counts.values()) <= 1,
        "V4 value state-mode assignment is not globally balanced",
    )
    return rows


def flow_state_from_value_row_v4(row: Mapping[str, Any]) -> FlowState:
    _require(row.get("split") == "TRAIN", "non-TRAIN V4 value state entered rollout")
    _require(
        int(row.get("base_flow_training_seed", -1)) in BASE_FLOW_SEEDS_V4,
        "V4 value rollout seed differs",
    )
    _require(
        int(row.get("trajectory_mode_id", -1)) in MODE_IDS_V4,
        "V4 value rollout mode differs",
    )
    edits = tuple(
        (int(edit["position"]), str(edit["candidate_base"]))
        for edit in row["source_relative_edits"]
    )
    remaining = int(row["remaining_budget"])
    structural = bool(row["structural_budget_exhausted"])
    _require(
        structural == (remaining == 0),
        "V4 value structural terminal flag differs from budget",
    )
    state = FlowState(
        source_sequence=str(row["source_sequence"]),
        current_sequence=str(row["current_sequence"]),
        source_relative_edits=edits,
        remaining_budget=remaining,
        assay_id=str(row["assay_category"]),
        context_id=str(row["context_category"]),
        terminal_cause="BUDGET_EXHAUSTED" if structural else None,
    )
    validate_state(state)
    _require(
        state.edit_count + state.remaining_budget == int(row["assigned_budget"]),
        "V4 value rollout state budget geometry changed",
    )
    return state


def generation_metadata_from_value_row_v4(
    row: Mapping[str, Any],
) -> SetFlowGenerationMetadataV3:
    return SetFlowGenerationMetadataV3(
        cache_record_id=str(row["cache_record_id"]),
        quantity=int(row["quantity_id"]),
        measurement=int(row["measurement_id"]),
        numerator=int(row["numerator_id"]),
        denominator=int(row["denominator_id"]),
        assay=int(row["assay_id"]),
        context=int(row["context_id"]),
        region=int(row["region_id"]),
    )


def value_rollout_seed_v4(
    state_index: int,
    rollout_index: int,
    *,
    base_flow_training_seed: int,
) -> int:
    _require(
        base_flow_training_seed in BASE_FLOW_SEEDS_V4,
        "undeclared V4 rollout base-flow seed",
    )
    _require(state_index >= 0, "V4 rollout state index is negative")
    _require(0 <= rollout_index < 8, "V4 rollout index is outside 0..7")
    return base_flow_training_seed + state_index * 1_000_003 + rollout_index


def terminal_rollout_row_v4(
    state_row: Mapping[str, Any],
    *,
    state_index: int,
    rollout_index: int,
    terminal_state: FlowState,
    trajectory_actions: Sequence[str],
    trunk_forwards: int,
    mode_forwards: int,
) -> dict[str, Any]:
    _require(
        terminal_state.terminal_cause is not None,
        "V4 value rollout did not terminate",
    )
    _require(
        terminal_state.source_sequence == state_row["source_sequence"],
        "V4 value rollout source changed",
    )
    _require(
        terminal_state.edit_count <= int(state_row["assigned_budget"]),
        "V4 value rollout exceeded edit budget",
    )
    base_flow_seed = int(state_row["base_flow_training_seed"])
    mode_id = int(state_row["trajectory_mode_id"])
    _require(
        base_flow_seed in BASE_FLOW_SEEDS_V4 and mode_id in MODE_IDS_V4,
        "V4 terminal rollout provenance differs",
    )
    _require(
        trunk_forwards >= 0 and mode_forwards >= trunk_forwards,
        "V4 terminal rollout compute count differs",
    )
    return {
        "schema_version": "route_a_v3_route2_xeditflow_terminal_rollout.v4",
        "state_id": str(state_row["state_id"]),
        "state_index": int(state_index),
        "rollout_index": int(rollout_index),
        "rollout_seed": value_rollout_seed_v4(
            state_index,
            rollout_index,
            base_flow_training_seed=base_flow_seed,
        ),
        "base_flow_training_seed": base_flow_seed,
        "trajectory_mode_id": mode_id,
        "source_sequence": terminal_state.source_sequence,
        "candidate_sequence": terminal_state.current_sequence,
        "source_relative_edits": [
            {
                "position": int(position),
                "source_base": terminal_state.source_sequence[int(position)],
                "candidate_base": str(alt),
            }
            for position, alt in terminal_state.source_relative_edits
        ],
        "task_id": str(state_row["task_id"]),
        "source_group_id": str(state_row["source_group_id"]),
        "endpoint_descriptor": dict(state_row["endpoint_descriptor"]),
        "assay_category": str(state_row["assay_category"]),
        "context_category": str(state_row["context_category"]),
        "region_id": int(state_row["region_id"]),
        "terminal_cause": str(terminal_state.terminal_cause),
        "edit_count": int(terminal_state.edit_count),
        "trajectory_actions": [str(value) for value in trajectory_actions],
        "trunk_forwards": int(trunk_forwards),
        "mode_forwards": int(mode_forwards),
        "setflow_mode_is_fixed_trajectory_state": True,
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def frozen_rollout_score_row_v4(
    terminal_row: Mapping[str, Any],
    member_rows: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    _require(
        tuple(sorted(member_rows)) == CRITIC_SEEDS_V4,
        "V4 critic rollout member seeds changed",
    )
    base_flow_seed = int(terminal_row.get("base_flow_training_seed", -1))
    _require(
        base_flow_seed in BASE_FLOW_SEEDS_V4,
        "V4 critic rollout base-flow seed differs",
    )
    mode_id = int(terminal_row.get("trajectory_mode_id", -1))
    _require(mode_id in MODE_IDS_V4, "V4 critic rollout mode differs")
    predictions: list[float] = []
    for seed in CRITIC_SEEDS_V4:
        member = member_rows[seed]
        _require(
            str(member.get("state_id")) == str(terminal_row.get("state_id")),
            "V4 critic member state differs",
        )
        _require(
            int(member.get("rollout_index", -1))
            == int(terminal_row.get("rollout_index", -2)),
            "V4 critic member rollout differs",
        )
        _require(
            int(member.get("trajectory_mode_id", -1)) == mode_id,
            "V4 critic member changed rollout mode",
        )
        _require(
            int(member.get("critic_seed", -1)) == seed,
            "V4 critic member seed differs",
        )
        prediction = float(member.get("standardized_prediction"))
        _require(math.isfinite(prediction), "V4 critic member prediction is nonfinite")
        _require(
            member.get("study_neutral") is True,
            "V4 critic member was not study-neutral",
        )
        predictions.append(prediction)
    return {
        "schema_version": "route_a_v3_route2_xeditflow_frozen_rollout_score.v4",
        "state_id": str(terminal_row["state_id"]),
        "rollout_index": int(terminal_row["rollout_index"]),
        "base_flow_training_seed": base_flow_seed,
        "trajectory_mode_id": mode_id,
        "critic_seeds": list(CRITIC_SEEDS_V4),
        "calibrated_seed_predictions": predictions,
        "study_neutral": True,
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
