from __future__ import annotations

from core.route2_xeditsetflow_gate_v3 import adjudicate_setflow_screen_v3


def _artifacts(f2_recovery=0.26, f3_recovery=0.26, f2_top=0.16, f3_top=0.16, f2_nll=8.0, f3_nll=8.0):
    f0 = {"status": "FROZEN_BASE_FLOW_V2_COMMON_SET_NLL_REPLAY_COMPLETE", "common_validation_set_marginal_nll": 10.0}
    training = {}
    validation = {}
    for arm, nll, recovery, top in (
        ("f1", 8.0, 0.3, 0.2), ("f2", f2_nll, f2_recovery, f2_top), ("f3", f3_nll, f3_recovery, f3_top)
    ):
        training[arm] = {
            "status": "XEDITSETFLOW_V3_GPU_TRAINING_COMPLETE", "seed": 20260903,
            "best_validation_common_set_marginal_nll": nll,
            "development_test_outcomes_accessed": False, "evaluation_outcomes_accessed": False,
        }
        validation[arm] = {
            "status": "FLOW_G0_READY", "seed": 20260903,
            "source_macro_candidate_recovery_rate": recovery,
            "source_macro_measured_top_k_recovery_at_k": top,
            "source_macro_unique_candidate_rate": 0.91, "hard_legality_rate": 1.0,
            "edit_budget_violation_count": 0, "candidate_budget_violation_count": 0,
            "trajectory_replay_failure_count": 0, "numerical_failure_count": 0,
            "development_test_outcomes_accessed": False, "evaluation_outcomes_accessed": False,
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
