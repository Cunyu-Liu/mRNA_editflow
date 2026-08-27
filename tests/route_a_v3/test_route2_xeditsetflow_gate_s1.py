from __future__ import annotations

import pytest

from core.route2_xeditsetflow_gate_s1 import (
    OBJECTIVE_IDENTITY,
    OBJECTIVE_WEIGHT,
    XEditSetFlowGateS1Error,
    adjudicate_setflow_screen_s1,
    select_checkpoint_s1,
    validate_checkpoint_summary_identity_s1,
)


CONFIG = {
    "terminal_f2_reference": {
        "source_macro_recovery": 0.2924616535727647,
        "source_macro_top_k_recovery": 0.168278220268518,
        "source_macro_unique_candidate_rate": 0.6793630751964085,
    }
}


def _summary(run_id: str, checkpoint_pass: int, *, recovery=.40, top_k=.25, unique=.96, nll=2.0) -> dict:
    modes = 8 if run_id == "v4_s1_full" else 1
    compute = {
        "trunk_forward_batch_count": 1,
        "trunk_forward_state_count": 100,
        "mode_head_forward_state_count": 100 * modes,
    }
    return {
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_s1_checkpoint_validation.v1",
        "status": "TERMINAL_XEDITSETFLOW_V4_S1_CHECKPOINT_VALIDATION_COMPLETE",
        "g0_status": "FLOW_G0_READY",
        "run_id": run_id,
        "run_stage": "SCREEN",
        "selectable": run_id == "v4_s1_full",
        "mode_count": modes,
        "seed": 20260911,
        "checkpoint_pass": checkpoint_pass,
        "objective_identity": OBJECTIVE_IDENTITY,
        "cross_state_candidate_mode_responsibility_weight": OBJECTIVE_WEIGHT,
        "active_responsibility_constraint_count": 10,
        "training_summary_status": "TERMINAL_XEDITSETFLOW_V4_S1_TRAINING_COMPLETE_PENDING_VALIDATION",
        "common_validation_set_marginal_nll": nll,
        "common_validation": {"common_validation_set_marginal_nll": nll, "validation_candidate_record_count": 15924, "validation_states_per_record": 2},
        "source_count": 891,
        "trajectory_count": 28512,
        "candidate_count": 28512,
        "candidate_cap_per_source": 32,
        "duplicate_retry_or_rejection_count": 0,
        "aggregate_mode_allocations": ({str(i): 3564 for i in range(8)} if modes == 8 else {"0": 28512}),
        "hard_legality_rate": 1.0,
        "edit_budget_violation_count": 0,
        "candidate_budget_violation_count": 0,
        "trajectory_replay_failure_count": 0,
        "numerical_failure_count": 0,
        "source_macro_unique_candidate_rate": unique,
        "source_macro_candidate_recovery_rate": recovery,
        "source_macro_measured_top_k_recovery_at_k": top_k,
        "small_graph_reference": {"status": "PASS", "mode_count": modes, "total_variation": 0.0, "tolerance": 1e-12},
        "compute": {
            "common_nll_trunk_forward_batch_count": 996,
            "common_nll_trunk_forward_state_count": 31848,
            "common_nll_mode_head_forward_state_count": 31848 * modes,
            "trajectory_count": 28512,
            "candidate_count": 28512,
            "critic_forward_count": 0,
            "independent_evaluator_forward_count": 0,
            "root_prior": {"trunk_forward_batch_count": 14, "trunk_forward_state_count": 891, "mode_head_forward_state_count": 891 * modes},
            "primary_generation": compute,
            "replay_generation": compute,
        },
        "precision": "BF16",
        "wall_time_seconds": 1.0,
        "peak_vram_bytes": 1,
        "cpu_fallback_used": False,
        "parameter_update_count": 0,
        "critic_used": False,
        "independent_evaluator_used": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def test_s1_identity_and_selection_are_exact() -> None:
    rows = {
        p: validate_checkpoint_summary_identity_s1(
            _summary("v4_s1_full", p, recovery=.41 if p == 8 else .40),
            run_id="v4_s1_full",
            checkpoint_pass=p,
        )
        for p in (4, 6, 8, 10)
    }
    assert select_checkpoint_s1(rows)["generation_constrained_selected_checkpoint"]["checkpoint_pass"] == 8
    bad = _summary("v4_s1_full", 4)
    bad["cross_state_candidate_mode_responsibility_weight"] = .04
    with pytest.raises(XEditSetFlowGateS1Error):
        validate_checkpoint_summary_identity_s1(bad, run_id="v4_s1_full", checkpoint_pass=4)


def test_s1_pass_uses_old_absolute_and_relative_gates_but_no_legacy_successor() -> None:
    summaries = {
        "v4_s1_full": {p: _summary("v4_s1_full", p) for p in (4, 6, 8, 10)},
        "v4_s1_single_mode": {p: _summary("v4_s1_single_mode", p, recovery=.36, top_k=.21, unique=.90) for p in (4, 6, 8, 10)},
    }
    gate = adjudicate_setflow_screen_s1(CONFIG, summaries)
    assert gate["status"] == "XEDITSETFLOW_V4_S1_SCREEN_PASS"
    assert gate["successor_protocol_required"] is True
    assert gate["confirmation_authorized"] is False
    assert gate["guidance_authorized"] is False
    summaries["v4_s1_full"][4]["source_macro_candidate_recovery_rate"] = .34
    summaries["v4_s1_full"][6]["source_macro_candidate_recovery_rate"] = .34
    summaries["v4_s1_full"][8]["source_macro_candidate_recovery_rate"] = .34
    summaries["v4_s1_full"][10]["source_macro_candidate_recovery_rate"] = .34
    assert adjudicate_setflow_screen_s1(CONFIG, summaries)["status"] == "XEDITSETFLOW_V4_S1_SCREEN_NO_GO"


def test_s1_terminal_g0_failure_is_a_scientific_ineligible_checkpoint() -> None:
    summary = _summary("v4_s1_full", 4)
    summary["hard_legality_rate"] = 0.99
    summary["g0_status"] = "FLOW_G0_VALIDATION_FAIL"
    row = validate_checkpoint_summary_identity_s1(
        summary,
        run_id="v4_s1_full",
        checkpoint_pass=4,
    )
    assert row["checks"]["hard_legality_100pct"] is False
    assert row["eligible"] is False


def test_s1_g0_status_must_match_terminal_correctness_checks() -> None:
    summary = _summary("v4_s1_full", 4)
    summary["g0_status"] = "FLOW_G0_VALIDATION_FAIL"
    with pytest.raises(XEditSetFlowGateS1Error, match="disagrees"):
        validate_checkpoint_summary_identity_s1(
            summary,
            run_id="v4_s1_full",
            checkpoint_pass=4,
        )
