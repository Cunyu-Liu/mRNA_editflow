from __future__ import annotations

import pytest

from core.route2_xeditflow_value_rollouts_v3 import (
    XEditFlowValueRolloutV3Error,
    attach_candidate_critic_rewards_v3,
    build_value_train_state_rows_v3,
    flow_state_from_value_row_v3,
    frozen_rollout_score_row_v3,
    terminal_rollout_row_v3,
)
from core.route2_xeditsetflow_training_v3 import XEditSetFlowRecordV3, setflow_vocabs


def _record() -> XEditSetFlowRecordV3:
    return XEditSetFlowRecordV3(
        record_id="r1",
        split="TRAIN",
        source="ACGU",
        terminal_edits=((1, "A"), (3, "C")),
        assigned_budget=3,
        task="task",
        study="study",
        source_group="group",
        assay="assay",
        context="context",
        region=0,
        quantity="RNA abundance",
        measurement="log2 fold",
        numerator="__NONE__",
        denominator="__NONE__",
    )


def test_value_states_are_two_deterministic_outcome_free_train_states() -> None:
    records = [_record()]
    vocabs = setflow_vocabs(records)
    for seed in (20260904, 20260905, 20260906):
        first = build_value_train_state_rows_v3(records, vocabs, base_flow_training_seed=seed)
        second = build_value_train_state_rows_v3(records, vocabs, base_flow_training_seed=seed)
        assert first == second
        assert len(first) == 2
        assert all(row["split"] == "TRAIN" for row in first)
        assert all(row["base_flow_training_seed"] == seed for row in first)
        assert all("direction_normalized_delta" not in row for row in first)
        assert flow_state_from_value_row_v3(first[-1]).terminal_cause is None


def test_terminal_rollout_and_three_member_score_are_identity_bound() -> None:
    record = _record()
    state = build_value_train_state_rows_v3(
        [record], setflow_vocabs([record]), base_flow_training_seed=20260904
    )[0]
    root = flow_state_from_value_row_v3(state)
    terminal = root.__class__(
        source_sequence=root.source_sequence,
        current_sequence=root.current_sequence,
        source_relative_edits=root.source_relative_edits,
        remaining_budget=root.remaining_budget,
        assay_id=root.assay_id,
        context_id=root.context_id,
        terminal_cause="EXPLICIT_STOP",
    )
    rollout = terminal_rollout_row_v3(
        state,
        state_index=0,
        rollout_index=0,
        terminal_state=terminal,
        trajectory_actions=("STOP",),
        base_flow_forwards=1,
    )
    members = {
        seed: {
            "state_id": state["state_id"],
            "rollout_index": 0,
            "critic_seed": seed,
            "standardized_prediction": value,
            "study_neutral": True,
        }
        for seed, value in zip((20260831, 20260901, 20260902), (0.1, 0.2, 0.3))
    }
    score = frozen_rollout_score_row_v3(rollout, members)
    assert score["base_flow_training_seed"] == 20260904
    assert score["calibrated_seed_predictions"] == [0.1, 0.2, 0.3]
    assert score["independent_evaluator_used"] is False
    members[20260831] = {**members[20260831], "study_neutral": False}
    with pytest.raises(XEditFlowValueRolloutV3Error):
        frozen_rollout_score_row_v3(rollout, members)


def test_candidate_reward_bills_three_members_once_per_source() -> None:
    candidates = [
        {"source_key": "s", "generation_rank": 1, "generation_score": 2.0},
        {"source_key": "s", "generation_rank": 2, "generation_score": 1.0},
    ]
    members = {
        seed: [
            {"state_id": "s", "rollout_index": index, "critic_seed": seed, "study_neutral": True, "standardized_prediction": float(seed_index + index)}
            for index in range(2)
        ]
        for seed_index, seed in enumerate((20260831, 20260901, 20260902))
    }
    rows = attach_candidate_critic_rewards_v3(candidates, members, kappa=0.5)
    assert [row["critic_forwards"] for row in rows] == [3, 0]
    assert [row["generation_score"] for row in rows] == [2.0, 1.0]
    assert all(row["critic_self_score_used_for_candidate_selection"] is False for row in rows)
