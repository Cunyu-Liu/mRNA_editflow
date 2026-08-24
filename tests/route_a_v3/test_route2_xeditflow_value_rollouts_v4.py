from __future__ import annotations

import copy

import pytest

from core.route2_xeditflow_value_rollouts_v4 import (
    build_value_train_state_rows_v4,
    flow_state_from_value_row_v4,
    frozen_rollout_score_row_v4,
    generation_metadata_from_value_row_v4,
    terminal_rollout_row_v4,
    value_rollout_seed_v4,
    value_state_mode_id_v4,
)
from core.route2_xeditsetflow_training_v4 import (
    setflow_source_records_from_projection_rows_v4,
    setflow_source_vocabs_v4,
)


def _row(record_id: str, source: str, position: int, alt: str) -> dict:
    candidate = list(source)
    candidate[position] = alt
    return {
        "canonical_record_id": record_id,
        "split": "TRAIN",
        "source_sequence": source,
        "candidate_sequence": "".join(candidate),
        "source_relative_edits": [
            {
                "position": position,
                "source_base": source[position],
                "candidate_base": alt,
            }
        ],
        "task_id": "task",
        "endpoint_id": "endpoint",
        "source_group_id": "legacy",
        "study_unit_id": "study",
        "assay_id": "assay",
        "biological_context_id": "context",
        "region_id": 0,
        "endpoint_descriptor": {
            "quantity_family": "quantity",
            "measurement_form": "measurement",
            "numerator_family": None,
            "denominator_family": None,
        },
    }


def _records():
    records, _ = setflow_source_records_from_projection_rows_v4(
        [
            _row("r0", "AAAAAA", 0, "C"),
            _row("r1", "CCCCCC", 1, "G"),
        ]
    )
    return records, setflow_source_vocabs_v4(records)


def test_v4_value_states_use_four_source_states_and_balanced_fixed_modes() -> None:
    records, vocabs = _records()
    rows = build_value_train_state_rows_v4(
        records, vocabs, base_flow_training_seed=20260912
    )
    assert len(rows) == 8
    assert [row["trajectory_mode_id"] for row in rows] == list(range(8))
    assert [row["state_slot"] for row in rows[:4]] == [0, 1, 2, 3]
    assert all(row["split"] == "TRAIN" for row in rows)
    assert all(row["setflow_mode_is_fixed_trajectory_state"] for row in rows)
    assert all("direction_normalized_delta" not in row for row in rows)
    assert value_state_mode_id_v4(2, 3) == 3


def test_v4_value_state_rows_are_replayable_and_require_lexicographic_sources() -> None:
    records, vocabs = _records()
    first = build_value_train_state_rows_v4(
        records, vocabs, base_flow_training_seed=20260912
    )
    second = build_value_train_state_rows_v4(
        records, vocabs, base_flow_training_seed=20260912
    )
    assert first == second
    with pytest.raises(Exception, match="lexicographic"):
        build_value_train_state_rows_v4(
            list(reversed(records)), vocabs, base_flow_training_seed=20260912
        )


def test_v4_terminal_rollout_and_critic_score_preserve_mode() -> None:
    records, vocabs = _records()
    rows = build_value_train_state_rows_v4(
        records, vocabs, base_flow_training_seed=20260912
    )
    state_row = rows[3]
    terminal_state = flow_state_from_value_row_v4(state_row)
    assert terminal_state.terminal_cause == "BUDGET_EXHAUSTED"
    metadata = generation_metadata_from_value_row_v4(state_row)
    assert metadata.cache_record_id == state_row["cache_record_id"]
    terminal = terminal_rollout_row_v4(
        state_row,
        state_index=3,
        rollout_index=2,
        terminal_state=terminal_state,
        trajectory_actions=(),
        trunk_forwards=0,
        mode_forwards=0,
    )
    assert terminal["trajectory_mode_id"] == 3
    members = {
        seed: {
            "state_id": terminal["state_id"],
            "rollout_index": 2,
            "trajectory_mode_id": 3,
            "critic_seed": seed,
            "standardized_prediction": float(index),
            "study_neutral": True,
        }
        for index, seed in enumerate((20260908, 20260909, 20260910))
    }
    score = frozen_rollout_score_row_v4(terminal, members)
    assert score["calibrated_seed_predictions"] == [0.0, 1.0, 2.0]
    assert score["trajectory_mode_id"] == 3
    assert value_rollout_seed_v4(
        3, 2, base_flow_training_seed=20260912
    ) == 20260912 + 3 * 1_000_003 + 2

    changed = copy.deepcopy(members)
    changed[20260908]["trajectory_mode_id"] = 4
    with pytest.raises(Exception, match="changed rollout mode"):
        frozen_rollout_score_row_v4(terminal, changed)
