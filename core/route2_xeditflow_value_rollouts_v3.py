"""Outcome-free TRAIN states and frozen rollout-score records for XEditFlow V3."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from core.route2_legal_xeditflow import FlowState, validate_state
from core.route2_xeditflow_value_training_v3 import BASE_FLOW_SEEDS_V3, CRITIC_SEEDS_V3
from core.route2_xeditsetflow_sampling_v3 import SetFlowGenerationMetadataV3
from core.route2_xeditsetflow_training_v3 import (
    SetMarginalStateDatasetV3,
    XEditSetFlowRecordV3,
)


class XEditFlowValueRolloutV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowValueRolloutV3Error(message)


def build_value_train_state_rows_v3(
    records: Sequence[XEditSetFlowRecordV3],
    vocabs: Mapping[str, Mapping[str, int]],
    *,
    base_flow_training_seed: int,
    state_pass_index: int = 0,
    states_per_record: int = 2,
) -> list[dict[str, Any]]:
    """Freeze two deterministic, outcome-free TRAIN states per measured record."""

    _require(base_flow_training_seed in BASE_FLOW_SEEDS_V3, "undeclared value base-flow seed")
    _require(state_pass_index == 0, "value state pass changed")
    _require(states_per_record == 2, "value state multiplicity changed")
    _require(bool(records) and all(record.split == "TRAIN" for record in records), "non-TRAIN record entered value states")
    dataset = SetMarginalStateDatasetV3(
        records,
        vocabs,
        seed=base_flow_training_seed,
        states_per_record=states_per_record,
    )
    dataset.set_pass(state_pass_index)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record_index, record in enumerate(records):
        for state_slot in range(states_per_record):
            sampled = dataset.state(record_index, state_slot)
            state_id = f"{record.record_id}:pass{state_pass_index}:slot{state_slot}"
            _require(state_id not in seen, "value state identity is duplicated")
            seen.add(state_id)
            selected = tuple(
                (int(position), str(alt))
                for position, alt in sampled["selected_edit_set"]
            )
            rows.append(
                {
                    "schema_version": "route_a_v3_route2_xeditflow_value_train_state.v3",
                    "state_id": state_id,
                    "split": "TRAIN",
                    "base_flow_training_seed": base_flow_training_seed,
                    "state_pass_index": state_pass_index,
                    "state_slot": state_slot,
                    "record_id": record.record_id,
                    "cache_record_id": record.record_id,
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
                    "independent_evaluator_used": False,
                    "development_test_outcomes_accessed": False,
                    "new_final_evaluation_outcomes_accessed": False,
                }
            )
    _require(len(rows) == len(records) * states_per_record, "value state count changed")
    return rows


def flow_state_from_value_row_v3(row: Mapping[str, Any]) -> FlowState:
    _require(row.get("split") == "TRAIN", "non-TRAIN value state entered rollout")
    _require(int(row.get("base_flow_training_seed", -1)) in BASE_FLOW_SEEDS_V3, "value rollout seed differs")
    edits = tuple(
        (int(edit["position"]), str(edit["candidate_base"]))
        for edit in row["source_relative_edits"]
    )
    remaining = int(row["remaining_budget"])
    terminal_cause = "BUDGET_EXHAUSTED" if remaining == 0 else None
    state = FlowState(
        source_sequence=str(row["source_sequence"]),
        current_sequence=str(row["current_sequence"]),
        source_relative_edits=edits,
        remaining_budget=remaining,
        assay_id=str(row["assay_category"]),
        context_id=str(row["context_category"]),
        terminal_cause=terminal_cause,
    )
    validate_state(state)
    _require(
        state.edit_count + state.remaining_budget == int(row["assigned_budget"]),
        "value rollout state budget geometry changed",
    )
    return state


def generation_metadata_from_value_row_v3(
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


def value_rollout_seed_v3(
    state_index: int,
    rollout_index: int,
    *,
    base_flow_training_seed: int,
) -> int:
    _require(base_flow_training_seed in BASE_FLOW_SEEDS_V3, "undeclared rollout base-flow seed")
    _require(state_index >= 0, "value rollout state index is negative")
    _require(0 <= rollout_index < 8, "value rollout index is outside 0..7")
    return base_flow_training_seed + state_index * 1_000_003 + rollout_index


def terminal_rollout_row_v3(
    state_row: Mapping[str, Any],
    *,
    state_index: int,
    rollout_index: int,
    terminal_state: FlowState,
    trajectory_actions: Sequence[str],
    base_flow_forwards: int,
) -> dict[str, Any]:
    _require(terminal_state.terminal_cause is not None, "value rollout did not terminate")
    _require(terminal_state.source_sequence == state_row["source_sequence"], "value rollout source changed")
    _require(terminal_state.edit_count <= int(state_row["assigned_budget"]), "value rollout exceeded edit budget")
    base_flow_seed = int(state_row["base_flow_training_seed"])
    _require(base_flow_seed in BASE_FLOW_SEEDS_V3, "terminal rollout base-flow seed differs")
    return {
        "schema_version": "route_a_v3_route2_xeditflow_terminal_rollout.v3",
        "state_id": str(state_row["state_id"]),
        "state_index": int(state_index),
        "rollout_index": int(rollout_index),
        "rollout_seed": value_rollout_seed_v3(
            state_index,
            rollout_index,
            base_flow_training_seed=base_flow_seed,
        ),
        "base_flow_training_seed": base_flow_seed,
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
        "base_flow_forwards": int(base_flow_forwards),
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def frozen_rollout_score_row_v3(
    terminal_row: Mapping[str, Any],
    member_rows: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    _require(tuple(sorted(member_rows)) == CRITIC_SEEDS_V3, "critic rollout member seeds changed")
    base_flow_seed = int(terminal_row.get("base_flow_training_seed", -1))
    _require(base_flow_seed in BASE_FLOW_SEEDS_V3, "critic rollout base-flow seed differs")
    predictions = []
    for seed in CRITIC_SEEDS_V3:
        member = member_rows[seed]
        _require(str(member.get("state_id")) == str(terminal_row.get("state_id")), "critic member state differs")
        _require(int(member.get("rollout_index", -1)) == int(terminal_row.get("rollout_index", -2)), "critic member rollout differs")
        _require(int(member.get("critic_seed", -1)) == seed, "critic member seed differs")
        prediction = float(member.get("standardized_prediction"))
        _require(math.isfinite(prediction), "critic member prediction is nonfinite")
        _require(member.get("study_neutral") is True, "critic member was not study-neutral")
        predictions.append(prediction)
    return {
        "schema_version": "route_a_v3_route2_xeditflow_frozen_rollout_score.v3",
        "state_id": str(terminal_row["state_id"]),
        "rollout_index": int(terminal_row["rollout_index"]),
        "base_flow_training_seed": base_flow_seed,
        "critic_seeds": list(CRITIC_SEEDS_V3),
        "calibrated_seed_predictions": predictions,
        "study_neutral": True,
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def attach_candidate_critic_rewards_v3(
    candidate_rows: Sequence[Mapping[str, Any]],
    member_rows: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    kappa: float,
) -> list[dict[str, Any]]:
    """Attach study-neutral ensemble diagnostics without changing generation rank."""

    _require(float(kappa) in {0.0, 0.5, 1.0}, "candidate Critic uncertainty penalty differs")
    _require(tuple(sorted(member_rows)) == CRITIC_SEEDS_V3, "candidate Critic seeds differ")
    _require(
        all(len(member_rows[seed]) == len(candidate_rows) for seed in CRITIC_SEEDS_V3),
        "candidate Critic prediction counts differ",
    )
    result = []
    billed_sources: set[str] = set()
    for index, candidate in enumerate(candidate_rows):
        predictions = []
        for seed in CRITIC_SEEDS_V3:
            member = member_rows[seed][index]
            _require(str(member.get("state_id")) == str(candidate.get("source_key")), "candidate Critic source differs")
            _require(int(member.get("rollout_index", -1)) == int(candidate.get("generation_rank", 0)) - 1, "candidate Critic rank differs")
            _require(int(member.get("critic_seed", -1)) == seed and member.get("study_neutral") is True, "candidate Critic member provenance differs")
            prediction = float(member["standardized_prediction"])
            _require(math.isfinite(prediction), "candidate Critic prediction is nonfinite")
            predictions.append(prediction)
        mean = math.fsum(predictions) / 3.0
        sd = math.sqrt(math.fsum((value - mean) ** 2 for value in predictions) / 3.0)
        reward = mean - float(kappa) * sd
        source_key = str(candidate["source_key"])
        critic_forwards = 0 if source_key in billed_sources else 3
        billed_sources.add(source_key)
        result.append(
            {
                **dict(candidate),
                "critic_seed_predictions": predictions,
                "critic_ensemble_mean": mean,
                "critic_ensemble_sd": sd,
                "critic_self_score": reward,
                "critic_forwards": critic_forwards,
                "critic_self_score_used_for_candidate_selection": False,
                "study_neutral": True,
            }
        )
    return result
