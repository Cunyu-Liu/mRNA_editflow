from __future__ import annotations

import json

import pytest
import torch

from scripts.route_a_v3.build_route2_xeditflow_value_targets_v3 import build


def test_value_target_builder_writes_private_tensor_and_outcomefree_summary(tmp_path) -> None:
    critic = {
        "status": "CRITIC_READY_FOR_GUIDANCE",
        "frozen_test_passed": True,
        "all_development_refit_complete": True,
        "loso_readiness_passed": True,
    }
    flow = {"status": "XEDITSETFLOW_V3_CONFIRMATION_PASS", "flow_status": "FLOW_G0_READY"}
    state = {
        "state_id": "s", "split": "TRAIN", "base_flow_training_seed": 20260904,
        "source_sequence": "AC", "current_sequence": "UC", "cache_record_id": "r",
        "assigned_budget": 3, "remaining_budget": 2,
        "quantity_id": 0, "measurement_id": 0, "numerator_id": 0,
        "denominator_id": 0, "assay_id": 0, "context_id": 0, "region_id": 0,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    rollouts = [
        {
            "state_id": "s", "rollout_index": index, "base_flow_training_seed": 20260904,
            "critic_seeds": [20260831, 20260901, 20260902],
            "calibrated_seed_predictions": [0.1, 0.2, 0.3], "study_neutral": True,
            "independent_evaluator_used": False,
            "development_test_outcomes_accessed": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
        for index in range(8)
    ]
    for name, value in (("critic.json", critic), ("flow.json", flow)):
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    (tmp_path / "states.jsonl").write_text(json.dumps(state) + "\n", encoding="utf-8")
    (tmp_path / "rollouts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rollouts), encoding="utf-8"
    )
    output = tmp_path / "out"
    summary = build(
        {
            "schema_version": "route_a_v3_route2_xeditflow_value_target_build_config.v1",
            "train_state_path": str(tmp_path / "states.jsonl"),
            "frozen_rollout_score_path": str(tmp_path / "rollouts.jsonl"),
            "critic_readiness_path": str(tmp_path / "critic.json"),
            "setflow_confirmation_path": str(tmp_path / "flow.json"),
            "base_flow_training_seed": 20260904,
            "kappa": 0.5,
            "temperature": 1.0,
        },
        output_dir=output,
    )
    assert summary["state_count"] == 1
    assert summary["raw_outcome_values_persisted"] is False
    assert "records" not in json.loads((output / "summary.json").read_text())
    payload = torch.load(output / "value_targets.pt", weights_only=False)
    assert payload["records"][0]["soft_value_target"] == pytest.approx(
        payload["records"][0]["rollout_rewards"][0]
    )
