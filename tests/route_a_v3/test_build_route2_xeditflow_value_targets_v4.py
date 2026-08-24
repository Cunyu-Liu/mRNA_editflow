from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.route_a_v3.build_route2_xeditflow_value_targets_v4 import (
    VALUE_TARGET_GRID_V4,
    build,
)


def _critic_ready():
    return {
        "status": "CRITIC_V4_READY_FOR_GUIDANCE",
        "three_seed_passed": True,
        "frozen_test_passed": True,
        "all_development_refit_complete": True,
        "loso_readiness_passed": True,
        "development_test_access_event_count": 1,
        "general_test_projection_persisted": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
        "guidance_authorized": True,
    }


def _flow_ready():
    return {
        "status": "XEDITSETFLOW_V4_G0_READY",
        "required_seeds": [20260912, 20260913, 20260914],
        "critic_used": False,
        "independent_evaluator_used": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def test_v4_target_builder_writes_exact_six_packages_without_beta(tmp_path) -> None:
    state = {
        "state_id": "s0",
        "split": "TRAIN",
        "base_flow_training_seed": 20260912,
        "trajectory_mode_id": 0,
        "source_sequence": "AC",
        "current_sequence": "UC",
        "cache_record_id": "r",
        "assigned_budget": 3,
        "remaining_budget": 2,
        "quantity_id": 0,
        "measurement_id": 0,
        "numerator_id": 0,
        "denominator_id": 0,
        "assay_id": 0,
        "context_id": 0,
        "region_id": 0,
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    rollouts = [
        {
            "state_id": "s0",
            "rollout_index": index,
            "base_flow_training_seed": 20260912,
            "trajectory_mode_id": 0,
            "critic_seeds": [20260908, 20260909, 20260910],
            "calibrated_seed_predictions": [
                float(index),
                float(index + 1),
                float(index + 2),
            ],
            "study_neutral": True,
            "independent_evaluator_used": False,
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
        for index in range(8)
    ]
    artifacts = {
        "critic.json": _critic_ready(),
        "flow.json": _flow_ready(),
        "rollout_summary.json": {
            "status": (
                "XEDITFLOW_V4_VALUE_ROLLOUTS_COMPLETE_PENDING_CRITIC_SCORING"
            ),
            "base_flow_training_seed": 20260912,
            "fixed_seed_replayable": True,
            "fixed_seed_replay_failure_count": 0,
            "state_mode_count": 1,
            "terminal_rollout_count": 8,
        },
        "score_summary.json": {
            "status": "XEDITFLOW_V4_VALUE_CRITIC_SCORING_COMPLETE",
            "terminal_rollout_count": 8,
            "critic_seeds": [20260908, 20260909, 20260910],
            "study_policy": "UNKNOWN_STUDY_SCALE_FIXED_1",
            "trajectory_mode_used_as_critic_input": False,
        },
    }
    for name, payload in artifacts.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "states.jsonl").write_text(
        json.dumps(state) + "\n", encoding="utf-8"
    )
    (tmp_path / "scores.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rollouts),
        encoding="utf-8",
    )
    config = {
        "schema_version": (
            "route_a_v3_route2_xeditflow_value_target_grid_build_config.v4"
        ),
        "stage": "GUIDANCE_SCREEN",
        "base_flow_training_seed": 20260912,
        "grid": [list(row) for row in VALUE_TARGET_GRID_V4],
        "train_state_path": str(tmp_path / "states.jsonl"),
        "frozen_rollout_score_path": str(tmp_path / "scores.jsonl"),
        "critic_readiness_path": str(tmp_path / "critic.json"),
        "setflow_confirmation_path": str(tmp_path / "flow.json"),
        "rollout_summary_path": str(tmp_path / "rollout_summary.json"),
        "critic_score_summary_path": str(tmp_path / "score_summary.json"),
        "output_root": str(tmp_path / "targets"),
    }
    result = build(config, output_root=tmp_path / "targets")
    assert result["package_count"] == 6
    assert result["beta_max_used_in_target"] is False
    assert {(row["kappa"], row["temperature"]) for row in result["packages"]} == set(
        VALUE_TARGET_GRID_V4
    )
    assert all(Path(row["value_target_path"]).is_file() for row in result["packages"])
    payload = torch.load(
        result["packages"][0]["value_target_path"], weights_only=False
    )
    assert payload["state_mode_count"] == 1
    assert payload["records"][0]["rollout_count"] == 8
    assert "records" not in json.loads(
        Path(result["packages"][0]["summary_path"]).read_text()
    )


def test_v4_target_builder_rejects_grid_or_score_count_drift(tmp_path) -> None:
    with pytest.raises(Exception, match="grid changed"):
        build(
            {
                "schema_version": (
                    "route_a_v3_route2_xeditflow_value_target_grid_build_config.v4"
                ),
                "stage": "GUIDANCE_SCREEN",
                "base_flow_training_seed": 20260912,
                "grid": [[0.0, 0.5]],
            },
            output_root=tmp_path / "out",
        )
