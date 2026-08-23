from __future__ import annotations

import json

import pytest

from core.route2_xeditsetflow_gate_v3 import (
    adjudicate_setflow_confirmation_v3,
    adjudicate_setflow_screen_v3,
    require_setflow_confirmation_authorization_v3,
)


def _artifacts(f2_recovery=0.26, f3_recovery=0.26, f2_top=0.16, f3_top=0.16, f2_nll=8.0, f3_nll=8.0):
    f0 = {
        "status": "FROZEN_BASE_FLOW_V2_COMMON_SET_NLL_REPLAY_COMPLETE",
        "selected_epoch": 1,
        "trainable_parameter_count": 817_957,
        "parameter_update_count": 0,
        "parameter_changed_during_replay": False,
        "validation_record_count": 15924,
        "validation_states_per_record": 2,
        "common_validation_set_marginal_nll": 10.0,
        "development_test_outcomes_accessed": False,
        "evaluation_outcomes_accessed": False,
        "critic_score_used": False,
        "independent_evaluator_used": False,
    }
    training = {}
    validation = {}
    for arm, nll, recovery, top in (
        ("f1", 8.0, 0.3, 0.2), ("f2", f2_nll, f2_recovery, f2_top), ("f3", f3_nll, f3_recovery, f3_top)
    ):
        parameter_counts = {"f1": 818_021, "f2": 16_178_790, "f3": 42_196_934}
        training[arm] = {
            "status": "XEDITSETFLOW_V3_GPU_TRAINING_COMPLETE", "seed": 20260903, "arm": arm,
            "run_stage": "SCREEN", "selectable": arm in {"f2", "f3"},
            "train_record_count": 68294, "validation_record_count": 15924,
            "states_per_record_per_pass": 2, "effective_batch_size": 32,
            "maximum_passes": 12, "training_precision": "BF16",
            "parameter_changed": True, "cpu_fallback_used": False,
            "trainable_parameter_count": parameter_counts[arm],
            "development_test_record_count_withheld": 18292,
            "critic_score_used": False, "independent_evaluator_used": False,
            "best_validation_common_set_marginal_nll": nll,
            "development_test_outcomes_accessed": False, "evaluation_outcomes_accessed": False,
        }
        validation[arm] = {
            "status": "FLOW_G0_READY", "seed": 20260903, "arm": arm,
            "method_id": f"unguided_xeditsetflow_v3_{arm}_seed20260903",
            "source_count": 891, "candidate_count": 28512,
            "trajectory_forward_batch_size": 64,
            "cpu_fallback_used": False, "parameter_update_count": 0,
            "guided_critic_used": False, "independent_evaluator_used": False,
            "source_macro_candidate_recovery_rate": recovery,
            "source_macro_measured_top_k_recovery_at_k": top,
            "source_macro_unique_candidate_rate": 0.91, "hard_legality_rate": 1.0,
            "edit_budget_violation_count": 0, "candidate_budget_violation_count": 0,
            "trajectory_replay_failure_count": 0, "numerical_failure_count": 0,
            "small_graph_reference": {"status": "PASS", "total_variation": 0.0, "tolerance": 1e-12},
            "development_test_outcomes_accessed": False, "evaluation_records_read": 0,
            "evaluation_outcomes_accessed": False,
            "generated_candidates_grant_canonical_credit": False,
            "biological_optimization_established": False,
        }
    return f0, training, validation


def test_f1_is_never_selectable_and_complete_tie_selects_smaller_f2() -> None:
    result = adjudicate_setflow_screen_v3(*_artifacts())
    assert result["status"] == "XEDITSETFLOW_V3_SCREEN_PASS"
    assert result["selected_arm"] == "f2"
    assert result["arms"]["f1"]["passes_screen_gate"] is False


def test_recovery_difference_must_strictly_exceed_0_01_before_primary_tie_break() -> None:
    result = adjudicate_setflow_screen_v3(*_artifacts(f2_recovery=0.26, f3_recovery=0.27, f2_top=0.17, f3_top=0.16))
    assert result["selected_arm"] == "f2"
    assert result["selection_reason"] == "TOP_K_RECOVERY_TIE_BREAK"
    result = adjudicate_setflow_screen_v3(*_artifacts(f2_recovery=0.26, f3_recovery=0.271, f2_top=0.17, f3_top=0.16))
    assert result["selected_arm"] == "f3"
    assert result["selection_reason"] == "RECOVERY_DIFFERENCE_EXCEEDS_0_01"


def test_both_selectable_fail_is_terminal_no_go() -> None:
    result = adjudicate_setflow_screen_v3(*_artifacts(f2_recovery=0.249, f3_recovery=0.249))
    assert result["status"] == "XEDITSETFLOW_V3_SCREEN_NO_GO"
    assert result["confirmation_authorized"] is False
    assert result["additional_seed_authorized"] is False


def test_screen_accepts_exact_terminal_v3_launch_head_summary_without_stage() -> None:
    f0, training, validation = _artifacts()
    for summary in training.values():
        summary.pop("run_stage")
        summary["schema_version"] = (
            "route_a_v3_route2_xeditsetflow_training_summary.v3"
        )
        summary["history_is_terminal"] = True
    result = adjudicate_setflow_screen_v3(f0, training, validation)
    assert result["status"] == "XEDITSETFLOW_V3_SCREEN_PASS"
    assert {
        row["run_stage_identity_source"] for row in result["arms"].values()
    } == {"DERIVED_FROM_TERMINAL_V3_SCREEN_SCHEMA_AND_SEED"}


def test_screen_missing_stage_fails_outside_exact_terminal_v3_summary() -> None:
    f0, training, validation = _artifacts()
    training["f2"].pop("run_stage")
    with pytest.raises(Exception, match="run stage is absent"):
        adjudicate_setflow_screen_v3(f0, training, validation)
    f0, training, validation = _artifacts()
    training["f2"]["run_stage"] = "CONFIRMATION"
    with pytest.raises(Exception, match="arm/role identity"):
        adjudicate_setflow_screen_v3(f0, training, validation)


def _confirmation_artifacts(arm: str = "f2"):
    training = {}
    validation = {}
    for seed in (20260904, 20260905, 20260906):
        training[seed] = {
            "status": "XEDITSETFLOW_V3_GPU_TRAINING_COMPLETE",
            "arm": arm,
            "seed": seed,
            "run_stage": "CONFIRMATION",
            "selectable": True,
            "train_record_count": 68294,
            "validation_record_count": 15924,
            "states_per_record_per_pass": 2,
            "effective_batch_size": 32,
            "maximum_passes": 12,
            "training_precision": "BF16",
            "parameter_changed": True,
            "trainable_parameter_count": 16_178_790 if arm == "f2" else 42_196_934,
            "cpu_fallback_used": False,
            "critic_score_used": False,
            "independent_evaluator_used": False,
            "development_test_outcomes_accessed": False,
            "evaluation_outcomes_accessed": False,
        }
        validation[seed] = {
            "status": "FLOW_G0_READY",
            "arm": arm,
            "seed": seed,
            "method_id": f"unguided_xeditsetflow_v3_{arm}_seed{seed}",
            "source_count": 891,
            "candidate_count": 28512,
            "trajectory_forward_batch_size": 64,
            "cpu_fallback_used": False,
            "parameter_update_count": 0,
            "guided_critic_used": False,
            "independent_evaluator_used": False,
            "source_macro_candidate_recovery_rate": 0.26,
            "source_macro_measured_top_k_recovery_at_k": 0.16,
            "source_macro_unique_candidate_rate": 0.91,
            "hard_legality_rate": 1.0,
            "edit_budget_violation_count": 0,
            "candidate_budget_violation_count": 0,
            "trajectory_replay_failure_count": 0,
            "numerical_failure_count": 0,
            "small_graph_reference": {
                "status": "PASS", "total_variation": 0.0, "tolerance": 1e-12,
            },
            "development_test_outcomes_accessed": False,
            "evaluation_records_read": 0,
            "evaluation_outcomes_accessed": False,
            "generated_candidates_grant_canonical_credit": False,
            "biological_optimization_established": False,
        }
    return training, validation


def test_confirmation_requires_all_three_seeds_to_pass() -> None:
    training, validation = _confirmation_artifacts()
    result = adjudicate_setflow_confirmation_v3(
        training, validation, selected_arm="f2"
    )
    assert result["status"] == "XEDITSETFLOW_V3_CONFIRMATION_PASS"
    assert result["flow_status"] == "FLOW_G0_READY"
    assert result["guidance_authorized"] is False
    validation[20260905]["source_macro_unique_candidate_rate"] = 0.89
    result = adjudicate_setflow_confirmation_v3(
        training, validation, selected_arm="f2"
    )
    assert result["status"] == "XEDITSETFLOW_V3_CONFIRMATION_NO_GO"
    assert result["flow_status"] == "FLOW_G0_NOT_READY"


def test_confirmation_rejects_missing_seed_and_unselected_arm(tmp_path) -> None:
    training, validation = _confirmation_artifacts()
    validation.pop(20260906)
    with pytest.raises(Exception, match="exactly the three frozen seeds"):
        adjudicate_setflow_confirmation_v3(
            training, validation, selected_arm="f2"
        )
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({
        "status": "XEDITSETFLOW_V3_SCREEN_PASS",
        "confirmation_authorized": True,
        "selected_arm": "f3",
    }))
    config = {
        "seed": 20260904,
        "run_stage": "CONFIRMATION",
        "screen_gate_path": str(gate),
        "selected_arm": "f3",
    }
    require_setflow_confirmation_authorization_v3(config, arm="f3")
    with pytest.raises(Exception, match="differs"):
        require_setflow_confirmation_authorization_v3(config, arm="f2")


def test_screen_and_confirmation_reject_misidentified_artifacts() -> None:
    f0, training, validation = _artifacts()
    training["f2"]["arm"] = "f3"
    with pytest.raises(Exception, match="arm/role identity"):
        adjudicate_setflow_screen_v3(f0, training, validation)
    training, validation = _confirmation_artifacts()
    validation[20260905]["guided_critic_used"] = True
    with pytest.raises(Exception, match="validation provenance"):
        adjudicate_setflow_confirmation_v3(
            training, validation, selected_arm="f2"
        )


def test_screen_rejects_wrong_capacity_cohort_or_small_graph_evidence() -> None:
    f0, training, validation = _artifacts()
    training["f2"]["trainable_parameter_count"] -= 1
    with pytest.raises(Exception, match="frozen arm"):
        adjudicate_setflow_screen_v3(f0, training, validation)
    f0, training, validation = _artifacts()
    validation["f2"]["candidate_count"] -= 1
    with pytest.raises(Exception, match="cohort or execution budget"):
        adjudicate_setflow_screen_v3(f0, training, validation)
    f0, training, validation = _artifacts()
    validation["f3"]["small_graph_reference"]["total_variation"] = 1e-6
    with pytest.raises(Exception, match="small-graph distribution exactness"):
        adjudicate_setflow_screen_v3(f0, training, validation)
