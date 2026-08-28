from __future__ import annotations

import pytest

from core.route2_xeditsetflow_gate_s1 import (
    CONFIRMATION_BOOTSTRAP_REPLICATES,
    CONFIRMATION_BOOTSTRAP_SEED,
    CONFIRMATION_BOOTSTRAP_STATISTIC,
    OBJECTIVE_IDENTITY,
    OBJECTIVE_WEIGHT,
    XEditSetFlowGateS1Error,
    adjudicate_setflow_confirmation_s1,
    adjudicate_setflow_screen_s1,
    select_checkpoint_s1,
    validate_checkpoint_summary_identity_s1,
)


TRAINING_HEAD = "a" * 40
VALIDATION_HEAD = "b" * 40


CONFIG = {
    "terminal_f2_reference": {
        "source_macro_recovery": 0.2924616535727647,
        "source_macro_top_k_recovery": 0.168278220268518,
        "source_macro_unique_candidate_rate": 0.6793630751964085,
    }
}


def _summary(
    run_id: str,
    checkpoint_pass: int,
    *,
    recovery=.40,
    top_k=.25,
    unique=.96,
    nll=2.0,
    run_stage="SCREEN",
    seed=20260911,
) -> dict:
    modes = 8 if run_id == "v4_s1_full" else 1
    compute = {
        "trunk_forward_batch_count": 1,
        "trunk_forward_state_count": 100,
        "mode_head_forward_state_count": 100 * modes,
    }
    summary = {
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_s1_checkpoint_validation.v1",
        "status": "TERMINAL_XEDITSETFLOW_V4_S1_CHECKPOINT_VALIDATION_COMPLETE",
        "g0_status": "FLOW_G0_READY",
        "run_id": run_id,
        "run_stage": run_stage,
        "selectable": run_id == "v4_s1_full",
        "mode_count": modes,
        "seed": seed,
        "parameter_initialization_seed": seed,
        "parameter_initialization_seed_applied_before_model_construction": True,
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
        "physical_gpu_index": 0,
        "torch_device": "cuda:0",
        "device_name": "NVIDIA A100-SXM4-80GB",
        "cuda_available": True,
        "bf16_supported": True,
        "cuda_device_index": 0,
        "cuda_device_name": "NVIDIA A100-SXM4-80GB",
        "cuda_device_uuid": "GPU-validation",
        "declared_physical_gpu_uuid": "validation",
        "cuda_parent_uuid_matches_declared_physical_index": True,
        "training_torch_device": "cuda:1",
        "training_device_name": "NVIDIA A100-SXM4-80GB",
        "training_precision": "BF16",
        "training_cuda_available": True,
        "training_bf16_supported": True,
        "training_cpu_fallback_used": False,
        "wall_time_seconds": 1.0,
        "peak_vram_bytes": 1,
        "cpu_fallback_used": False,
        "parameter_update_count": 0,
        "critic_used": False,
        "independent_evaluator_used": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
        "checkpoint_path": (
            f"/tmp/s1_confirmation/seed_{seed}/v4_s1_full/pass_{checkpoint_pass}.pt"
        ),
        "training_summary_path": (
            f"/tmp/s1_confirmation/seed_{seed}/v4_s1_full/training_summary.json"
        ),
        "validation_summary_path": (
            f"/tmp/s1_confirmation/seed_{seed}/outcome_free_validation_generation/"
            f"v4_s1_full/pass_{checkpoint_pass}/validation_summary.json"
        ),
        "training_git_head": TRAINING_HEAD,
        "validation_git_head": VALIDATION_HEAD,
        "training_and_validation_git_heads_differ": True,
    }
    if run_stage == "CONFIRMATION":
        summary["selected_model"] = "v4_s1_full"
    return summary


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


def _confirmation_config(seed: int) -> dict:
    return {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v4_s1_confirmation_runtime.v1"
        ),
        "status": "FROZEN_S1_CONFIRMATION_CONFIG_NOT_STARTED",
        "run_stage": "CONFIRMATION",
        "training_seed": seed,
        "selected_model": "v4_s1_full",
        "required_confirmation_seeds": [20260912, 20260913, 20260914],
        "additional_seed_authorized": False,
        "objective_identity": OBJECTIVE_IDENTITY,
        "cross_state_candidate_mode_responsibility_weight": OBJECTIVE_WEIGHT,
        "confirmation_runner_git_head": TRAINING_HEAD,
        "output_root": f"/tmp/s1_confirmation/seed_{seed}",
        "validation_output_root": (
            f"/tmp/s1_confirmation/seed_{seed}/outcome_free_validation_generation"
        ),
        "screen_gate_path": "/tmp/s1_screen_gate.json",
        "screen_runner_git_head": (
            "930fccf468c14378b3dd2fd2caf3aaa3cc2eb3c8"
        ),
        "screen_selected_checkpoint_pass": 8,
        "screen_provenance": {
            "screen_runner_git_head": (
                "930fccf468c14378b3dd2fd2caf3aaa3cc2eb3c8"
            ),
            "screen_gate_path": "/tmp/s1_screen_gate.json",
            "screen_selected_checkpoint_pass": 8,
            "checkpoint_decisions": {
                "v4_s1_full": {"selected_checkpoint_pass": 8},
                "v4_s1_single_mode": {"selected_checkpoint_pass": 6},
            },
            "role": "BOUND_SCREEN_FULL_AND_SINGLE_MODE_PROVENANCE_ONLY",
        },
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
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
        summaries[seed] = {
            checkpoint_pass: _with_per_source(
                _summary(
                    "v4_s1_full",
                    checkpoint_pass,
                    recovery=recovery,
                    top_k=0.25,
                    unique=0.96,
                    run_stage="CONFIRMATION",
                    seed=seed,
                ),
                [recovery] * 891,
            )
            for checkpoint_pass in (4, 6, 8, 10)
        }
    return summaries


def test_s1_three_seed_confirmation_requires_exact_full_only_package() -> None:
    configs = {
        seed: _confirmation_config(seed)
        for seed in (20260912, 20260913, 20260914)
    }
    summaries = _confirmation_summaries()
    gate = adjudicate_setflow_confirmation_s1(
        configs, summaries, _f2_summary([0.29] * 891)
    )
    assert gate["status"] == "XEDITSETFLOW_V4_G0_READY"
    assert gate["selected_model"] == "v4_s1_full"
    assert gate["objective_identity"] == OBJECTIVE_IDENTITY
    assert gate["guidance_authorized"] is False
    assert gate["development_test_outcome_reads"] == 0
    assert gate["screen_selected_checkpoint_pass"] == 8
    assert gate["screen_provenance"] == configs[20260912]["screen_provenance"]
    assert gate["confirmation_runner_git_head"] == TRAINING_HEAD
    assert gate["training_git_head"] == TRAINING_HEAD
    assert gate["validation_git_head"] == VALIDATION_HEAD
    assert gate["training_and_validation_git_heads_differ"] is True
    assert all(
        result["selected_checkpoint_path"]
        == f"/tmp/s1_confirmation/seed_{seed}/v4_s1_full/pass_4.pt"
        for seed, result in (
            (int(seed), result) for seed, result in gate["seed_results"].items()
        )
    )
    assert all(
        result["paired_bootstrap_recovery_improvement"]["statistic"]
        == CONFIRMATION_BOOTSTRAP_STATISTIC
        and result["paired_bootstrap_recovery_improvement"]["replicates"]
        == CONFIRMATION_BOOTSTRAP_REPLICATES
        and result["paired_bootstrap_recovery_improvement"]["seed"]
        == CONFIRMATION_BOOTSTRAP_SEED
        for result in gate["seed_results"].values()
    )
    summaries.pop(20260914)
    with pytest.raises(XEditSetFlowGateS1Error, match="seed cohort"):
        adjudicate_setflow_confirmation_s1(
            configs, summaries, _f2_summary([0.29] * 891)
        )


def test_s1_confirmation_scientific_failure_is_no_go_not_technical() -> None:
    configs = {
        seed: _confirmation_config(seed)
        for seed in (20260912, 20260913, 20260914)
    }
    gate = adjudicate_setflow_confirmation_s1(
        configs,
        _confirmation_summaries(recovery=0.34),
        _f2_summary([0.29] * 891),
    )
    assert gate["status"] == "XEDITSETFLOW_V4_CONFIRMATION_NO_GO"
    assert "technical_failures" not in gate
    assert gate["additional_seed_authorized"] is False


def test_s1_confirmation_rejects_protected_or_checkpoint_lineage_drift() -> None:
    configs = {
        seed: _confirmation_config(seed)
        for seed in (20260912, 20260913, 20260914)
    }
    summaries = _confirmation_summaries()
    configs[20260912]["development_test_outcome_reads"] = 1
    with pytest.raises(XEditSetFlowGateS1Error, match="protected read"):
        adjudicate_setflow_confirmation_s1(
            configs, summaries, _f2_summary([0.29] * 891)
        )
    configs[20260912]["development_test_outcome_reads"] = 0
    configs[20260912]["screen_selected_checkpoint_pass"] = 6
    with pytest.raises(XEditSetFlowGateS1Error, match="screen lineage"):
        adjudicate_setflow_confirmation_s1(
            configs, summaries, _f2_summary([0.29] * 891)
        )
    configs[20260912]["screen_selected_checkpoint_pass"] = 8
    summaries[20260912][4]["checkpoint_path"] = "/tmp/legacy/v4_full/pass_4.pt"
    with pytest.raises(XEditSetFlowGateS1Error, match="artifact path"):
        adjudicate_setflow_confirmation_s1(
            configs, summaries, _f2_summary([0.29] * 891)
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("parameter_initialization_seed", 20260913, "initialization"),
        (
            "parameter_initialization_seed_applied_before_model_construction",
            False,
            "initialization",
        ),
        ("cuda_available", False, "Validation CUDA/A100/BF16"),
        ("bf16_supported", False, "Validation CUDA/A100/BF16"),
        ("cpu_fallback_used", True, "provenance changed"),
        ("precision", "FP32", "provenance changed"),
        ("cuda_parent_uuid_matches_declared_physical_index", False, "Validation CUDA/A100/BF16"),
        ("training_cuda_available", False, "training CUDA/A100/BF16"),
        ("training_bf16_supported", False, "training CUDA/A100/BF16"),
        ("training_cpu_fallback_used", True, "training CUDA/A100/BF16"),
    ),
)
def test_s1_confirmation_gate_rejects_initialization_or_cuda_evidence_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    configs = {
        seed: _confirmation_config(seed)
        for seed in (20260912, 20260913, 20260914)
    }
    summaries = _confirmation_summaries()
    summaries[20260912][4][field] = value
    with pytest.raises(XEditSetFlowGateS1Error, match=message):
        adjudicate_setflow_confirmation_s1(
            configs, summaries, _f2_summary([0.29] * 891)
        )


@pytest.mark.parametrize(
    ("field", "wrong_path"),
    (
        (
            "checkpoint_path",
            "/tmp/s1_confirmation/seed_20260913/v4_s1_full/pass_4.pt",
        ),
        (
            "training_summary_path",
            "/tmp/s1_confirmation/seed_20260913/v4_s1_full/training_summary.json",
        ),
        (
            "validation_summary_path",
            "/tmp/s1_confirmation/seed_20260913/outcome_free_validation_generation/"
            "v4_s1_full/pass_4/validation_summary.json",
        ),
    ),
)
def test_s1_confirmation_gate_rejects_wrong_seed_canonical_artifact_paths(
    field: str,
    wrong_path: str,
) -> None:
    configs = {
        seed: _confirmation_config(seed)
        for seed in (20260912, 20260913, 20260914)
    }
    summaries = _confirmation_summaries()
    summaries[20260912][4][field] = wrong_path
    with pytest.raises(XEditSetFlowGateS1Error, match="artifact path"):
        adjudicate_setflow_confirmation_s1(
            configs, summaries, _f2_summary([0.29] * 891)
        )


def test_s1_confirmation_gate_rejects_training_or_validation_head_drift() -> None:
    configs = {
        seed: _confirmation_config(seed)
        for seed in (20260912, 20260913, 20260914)
    }
    summaries = _confirmation_summaries()
    summaries[20260912][4]["training_git_head"] = "c" * 40
    with pytest.raises(XEditSetFlowGateS1Error, match="Git lineage"):
        adjudicate_setflow_confirmation_s1(
            configs, summaries, _f2_summary([0.29] * 891)
        )
    summaries = _confirmation_summaries()
    summaries[20260912][4]["validation_git_head"] = "not-a-head"
    with pytest.raises(XEditSetFlowGateS1Error, match="Git lineage"):
        adjudicate_setflow_confirmation_s1(
            configs, summaries, _f2_summary([0.29] * 891)
        )
    summaries = _confirmation_summaries()
    summaries[20260912][4]["validation_git_head"] = "c" * 40
    summaries[20260912][4]["training_and_validation_git_heads_differ"] = True
    with pytest.raises(XEditSetFlowGateS1Error, match="across 12 validations"):
        adjudicate_setflow_confirmation_s1(
            configs, summaries, _f2_summary([0.29] * 891)
        )
