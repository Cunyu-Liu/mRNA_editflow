from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.route2_xeditsetflow_gate_v4 import (
    XEditSetFlowGateV4Error,
    adjudicate_setflow_confirmation_v4,
    adjudicate_setflow_screen_v4,
    confirmation_technical_failure_gate_v4,
    paired_bootstrap_recovery_improvement_v4,
    select_checkpoint_v4,
    technical_failure_gate_v4,
    validate_checkpoint_summary_identity_v4,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads(
    (ROOT / "configs/route_a_v3_route2_xeditsetflow_v4_screen_v1.json").read_text(
        encoding="utf-8"
    )
)


def _summary(
    run_id: str,
    checkpoint_pass: int,
    *,
    nll: float = 2.0,
    recovery: float = 0.40,
    top_k: float = 0.25,
    unique: float = 0.96,
) -> dict:
    modes = 8 if run_id == "v4_full" else 1
    allocations = (
        {str(index): 3564 for index in range(8)}
        if modes == 8
        else {"0": 28512}
    )
    compute_row = {
        "trunk_forward_batch_count": 1,
        "trunk_forward_state_count": 100,
        "mode_head_forward_state_count": 100 * modes,
    }
    return {
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_checkpoint_validation.v1",
        "status": "TERMINAL_XEDITSETFLOW_V4_CHECKPOINT_VALIDATION_COMPLETE",
        "g0_status": "FLOW_G0_READY",
        "run_id": run_id,
        "run_stage": "SCREEN",
        "selectable": run_id == "v4_full",
        "mode_count": modes,
        "seed": 20260911,
        "checkpoint_pass": checkpoint_pass,
        "training_summary_status": "TERMINAL_XEDITSETFLOW_V4_TRAINING_COMPLETE_PENDING_VALIDATION",
        "common_validation_set_marginal_nll": nll,
        "common_validation": {
            "common_validation_set_marginal_nll": nll,
            "validation_candidate_record_count": 15924,
            "validation_states_per_record": 2,
        },
        "source_count": 891,
        "trajectory_count": 28512,
        "candidate_count": 28512,
        "candidate_cap_per_source": 32,
        "duplicate_retry_or_rejection_count": 0,
        "aggregate_mode_allocations": allocations,
        "hard_legality_rate": 1.0,
        "edit_budget_violation_count": 0,
        "candidate_budget_violation_count": 0,
        "trajectory_replay_failure_count": 0,
        "numerical_failure_count": 0,
        "source_macro_unique_candidate_rate": unique,
        "source_macro_candidate_recovery_rate": recovery,
        "source_macro_measured_top_k_recovery_at_k": top_k,
        "small_graph_reference": {
            "status": "PASS",
            "mode_count": modes,
            "total_variation": 0.0,
            "tolerance": 1e-12,
        },
        "compute": {
            "common_nll_trunk_forward_batch_count": 996,
            "common_nll_trunk_forward_state_count": 31848,
            "common_nll_mode_head_forward_state_count": 31848 * modes,
            "trajectory_count": 28512,
            "candidate_count": 28512,
            "critic_forward_count": 0,
            "independent_evaluator_forward_count": 0,
            "root_prior": {
                "trunk_forward_batch_count": 14,
                "trunk_forward_state_count": 891,
                "mode_head_forward_state_count": 891 * modes,
            },
            "primary_generation": compute_row,
            "replay_generation": compute_row,
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


def test_identity_and_absolute_gate_accept_only_exact_budgeted_artifact() -> None:
    row = validate_checkpoint_summary_identity_v4(
        _summary("v4_full", 4), run_id="v4_full", checkpoint_pass=4
    )
    assert row["eligible"] is True
    bad = _summary("v4_full", 4)
    bad["development_test_outcome_reads"] = 1
    with pytest.raises(XEditSetFlowGateV4Error):
        validate_checkpoint_summary_identity_v4(
            bad, run_id="v4_full", checkpoint_pass=4
        )


def test_checkpoint_selection_uses_recovery_topk_nll_then_earlier_pass() -> None:
    rows = {
        checkpoint_pass: validate_checkpoint_summary_identity_v4(
            _summary(
                "v4_full",
                checkpoint_pass,
                nll=1.9 if checkpoint_pass == 4 else 2.0,
                recovery=0.40 if checkpoint_pass != 8 else 0.41,
                top_k=0.25,
            ),
            run_id="v4_full",
            checkpoint_pass=checkpoint_pass,
        )
        for checkpoint_pass in (4, 6, 8, 10)
    }
    decision = select_checkpoint_v4(rows)
    assert decision["generation_constrained_selected_checkpoint"]["checkpoint_pass"] == 8
    assert decision["nll_selected_checkpoint"]["checkpoint_pass"] == 4
    assert decision["nll_only_selection_differs"] is True


def test_no_eligible_checkpoint_is_no_go_not_nearest_selection() -> None:
    summaries = {
        run_id: {
            checkpoint_pass: _summary(
                run_id, checkpoint_pass, recovery=0.34
            )
            for checkpoint_pass in (4, 6, 8, 10)
        }
        for run_id in ("v4_full", "v4_single_mode")
    }
    gate = adjudicate_setflow_screen_v4(CONFIG, summaries)
    assert gate["status"] == "XEDITSETFLOW_V4_SCREEN_NO_GO"
    assert gate["selected_checkpoint_pass"] is None
    assert gate["confirmation_authorized"] is False


def test_full_pass_requires_absolute_f2_and_single_mode_margins() -> None:
    summaries = {
        "v4_full": {
            checkpoint_pass: _summary(
                "v4_full", checkpoint_pass, recovery=0.40, top_k=0.25, unique=0.96
            )
            for checkpoint_pass in (4, 6, 8, 10)
        },
        "v4_single_mode": {
            checkpoint_pass: _summary(
                "v4_single_mode",
                checkpoint_pass,
                recovery=0.36,
                top_k=0.21,
                unique=0.90,
            )
            for checkpoint_pass in (4, 6, 8, 10)
        },
    }
    gate = adjudicate_setflow_screen_v4(CONFIG, summaries)
    assert gate["status"] == "XEDITSETFLOW_V4_SCREEN_PASS"
    assert gate["selected_checkpoint_pass"] == 4
    assert gate["confirmation_authorized"] is True
    summaries["v4_single_mode"][4]["source_macro_unique_candidate_rate"] = 0.92
    summaries["v4_single_mode"][6]["source_macro_unique_candidate_rate"] = 0.92
    summaries["v4_single_mode"][8]["source_macro_unique_candidate_rate"] = 0.92
    summaries["v4_single_mode"][10]["source_macro_unique_candidate_rate"] = 0.92
    gate = adjudicate_setflow_screen_v4(CONFIG, summaries)
    assert gate["status"] == "XEDITSETFLOW_V4_SCREEN_NO_GO"


def test_technical_failure_is_terminal_no_go_and_authorizes_nothing() -> None:
    gate = technical_failure_gate_v4(
        [
            {
                "run_id": "v4_full",
                "error_type": "RuntimeError",
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            }
        ]
    )
    assert gate["status"] == "XEDITSETFLOW_V4_SCREEN_NO_GO"
    assert gate["confirmation_authorized"] is False
    assert gate["development_test_authorized"] is False
    assert gate["guidance_authorized"] is False
    with pytest.raises(XEditSetFlowGateV4Error):
        technical_failure_gate_v4(
            [
                {
                    "run_id": "v4_full",
                    "development_test_outcome_reads": 1,
                    "new_final_evaluation_outcome_reads": 0,
                }
            ]
        )


def _confirmation_config(seed: int) -> dict:
    return {
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_confirmation_runtime.v1",
        "run_stage": "CONFIRMATION",
        "training_seed": seed,
        "selected_model": "v4_full",
        "required_confirmation_seeds": [20260912, 20260913, 20260914],
        "additional_seed_authorized": False,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def _with_per_source(summary: dict, values: list[float]) -> dict:
    assert len(values) == 891
    result = dict(summary)
    result["source_macro_candidate_recovery_rate"] = sum(values) / len(values)
    result["measured_neighborhood_metrics"] = {
        "per_source": {
            f"source-{index:04d}": {"candidate_recovery_rate": value}
            for index, value in enumerate(values)
        }
    }
    return result


def _f2_summary(values: list[float]) -> dict:
    recovery = sum(values) / len(values)
    return {
        "schema_version": "route_a_v3_route2_xeditsetflow_unguided_validation.v3",
        "status": "FLOW_G0_READY",
        "arm": "f2",
        "seed": 20260903,
        "source_count": 891,
        "candidate_count": 28512,
        "source_macro_candidate_recovery_rate": recovery,
        "source_macro_measured_top_k_recovery_at_k": 0.16,
        "source_macro_unique_candidate_rate": 0.67,
        "measured_neighborhood_metrics": {
            "per_source": {
                f"source-{index:04d}": {"candidate_recovery_rate": value}
                for index, value in enumerate(values)
            }
        },
        "development_test_outcomes_accessed": False,
        "evaluation_records_read": 0,
        "evaluation_outcomes_accessed": False,
        "guided_critic_used": False,
        "independent_evaluator_used": False,
    }


def _confirmation_summaries(recovery: float = 0.40) -> dict:
    summaries = {}
    for seed in (20260912, 20260913, 20260914):
        summaries[seed] = {}
        for checkpoint_pass in (4, 6, 8, 10):
            row = _summary(
                "v4_full",
                checkpoint_pass,
                recovery=recovery,
                top_k=0.25,
                unique=0.96,
            )
            row["run_stage"] = "CONFIRMATION"
            row["seed"] = seed
            summaries[seed][checkpoint_pass] = _with_per_source(
                row, [recovery] * 891
            )
    return summaries


def test_three_seed_confirmation_requires_every_seed_and_positive_bootstrap_ci() -> None:
    configs = {seed: _confirmation_config(seed) for seed in (20260912, 20260913, 20260914)}
    summaries = _confirmation_summaries()
    gate = adjudicate_setflow_confirmation_v4(
        configs, summaries, _f2_summary([0.29] * 891)
    )
    assert gate["status"] == "XEDITSETFLOW_V4_G0_READY"
    assert gate["guidance_authorized"] is False
    assert all(
        row["paired_bootstrap_recovery_improvement"][
            "ci_lower_bound_strictly_greater_than_zero"
        ]
        for row in gate["seed_results"].values()
    )
    summaries.pop(20260914)
    with pytest.raises(XEditSetFlowGateV4Error):
        adjudicate_setflow_confirmation_v4(
            configs, summaries, _f2_summary([0.29] * 891)
        )


def test_bootstrap_can_reject_positive_point_margin_with_crossing_ci() -> None:
    v4_values = [1.0] * 468 + [0.0] * 423
    f2_values = [0.0] * 468 + [1.0] * 423
    summary = _with_per_source(
        {
            "source_macro_candidate_recovery_rate": sum(v4_values) / 891,
        },
        v4_values,
    )
    bootstrap = paired_bootstrap_recovery_improvement_v4(
        summary, _f2_summary(f2_values)
    )
    assert bootstrap["point_difference"] > 0.05
    assert bootstrap["ci_lower_bound_strictly_greater_than_zero"] is False


def test_confirmation_technical_failure_is_terminal_and_adds_no_seed() -> None:
    gate = confirmation_technical_failure_gate_v4(
        [
            {
                "training_seed": 20260912,
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            }
        ]
    )
    assert gate["status"] == "XEDITSETFLOW_V4_CONFIRMATION_NO_GO"
    assert gate["additional_seed_authorized"] is False
