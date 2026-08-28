from __future__ import annotations

import json

import pytest

from core.route2_xeditflow_equal_wall_time_v3 import EQUAL_WALL_TIME_SCOPE_V3
from scripts.route_a_v3.adapt_route2_xeditflow_strongest_baseline_v4 import (
    adapt_strongest_baseline_v4,
)
from scripts.route_a_v3.adjudicate_route2_xeditflow_final_v4 import (
    adjudicate_final_manifest_v4,
)
from scripts.route_a_v3.assemble_route2_xeditflow_final_seed_evidence_v4 import (
    METHODS_V4,
    V4_GENERATED_METHODS,
    assemble_final_seed_evidence_v4,
    write_final_seed_evidence_v4,
)
from scripts.route_a_v3.build_route2_xeditflow_equal_wall_time_sensitivity_v4 import (
    MATCHED_COMPUTE_SCORED_JSONL,
    SEARCH_CANDIDATE_JSONL,
    build_equal_wall_time_sensitivity_v4,
)
from scripts.route_a_v3.compose_route2_xeditflow_final_comparison_manifest_v4 import (
    compose_final_comparison_manifest_v4,
)
from scripts.route_a_v3.run_route2_xeditflow_strongest_timing_v4 import (
    strongest_timing_command_v4,
    validate_strongest_timing_config_v4,
)


def _value_checkpoint_path(seed: int) -> str:
    return (
        "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"
        f"final/value_seed_{seed}.pt"
    )


def _value_provenance(seed: int) -> dict:
    path = _value_checkpoint_path(seed)
    return {
        "parameter_initialization_seed": seed,
        "parameter_initialization_seed_applied_before_model_construction": True,
        "optimizer_steps": 10,
        "parameter_changed": True,
        "cuda_available": True,
        "bf16_supported": True,
        "training_precision": "BF16",
        "cpu_fallback_used": False,
        "torch_device": "cuda:0",
        "physical_gpu_index": 0,
        "cuda_device_index": 0,
        "cuda_device_name": "NVIDIA A100-SXM4-80GB",
        "cuda_device_uuid": "GPU-actual",
        "declared_physical_gpu_uuid": "actual",
        "cuda_parent_uuid_matches_declared_physical_index": True,
        "value_checkpoint_path": path,
    }


def _strongest_inputs():
    strongest = {
        "status": "DEVELOPMENT_STRONGEST_GENERATION_BASELINE_FROZEN_INDEPENDENT_EVALUATOR_ONLY",
        "strongest_generation_baseline_id": "genetic",
        "evaluation_outcomes_accessed": False,
        "forward_equivalent_budget_per_source": 320,
        "critic_forward_budget_per_source": 320,
        "guiding_checkpoint_path": "/critic.pt",
        "independent_evaluator_checkpoint_path": "/evaluator.pt",
    }
    selection = {
        "selection_pool": "DEVELOPMENT_MEASURED_NEIGHBORHOOD",
        "evaluation_release_state": "CLOSED",
        "baseline_evaluations": [
            {
                "method_id": "genetic",
                "evaluation": {
                    "generation": {
                        "method_id": "genetic",
                        "source_count": 891,
                        "source_macro_unique_candidate_rate": 0.92,
                        "hard_legality_rate": 1.0,
                        "edit_budget_violation_count": 0,
                        "candidate_budget_violation_count": 0,
                    },
                    "measured_neighborhood": {
                        "source_macro_candidate_recovery_rate": 0.30,
                        "source_macro_measured_top_k_recovery_at_k": 0.20,
                    },
                },
            }
        ],
    }
    return strongest, selection


def test_v4_strongest_adapter_is_read_only_and_seed_bound() -> None:
    strongest, selection = _strongest_inputs()
    result = adapt_strongest_baseline_v4(
        strongest, selection, base_flow_training_seed=20260914
    )
    assert result["generation"]["base_flow_training_seed"] == 20260914
    assert result["generation"]["frozen_baseline_reselected_for_v4"] is False
    assert result["open"]["historical_open_ndcg_used_as_new_closed_ndcg"] is False
    with pytest.raises(Exception, match="seed differs"):
        adapt_strongest_baseline_v4(
            strongest, selection, base_flow_training_seed=20260915
        )


def test_v4_strongest_timing_is_the_frozen_search_only() -> None:
    strongest, selection = _strongest_inputs()
    config = {
        "schema_version": "route_a_v3_route2_xeditflow_strongest_timing_config.v4",
        "method_id": "genetic",
        "strongest_generation_baseline_path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/strongest.json",
        "baseline_selection_input_path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/selection.json",
        "source_manifest_path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/sources.jsonl",
        "guiding_checkpoint_path": "/critic.pt",
        "critic_forward_budget_per_source": 320,
        "beam_width": 16,
        "genetic_population_size": 32,
        "oversample_factor": 8,
        "exhaustive_space_limit": 4096,
        "seed": 20260816,
        "physical_gpu_index": 2,
        "device": "cuda:2",
        "output_dir": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/timing",
        "timing_only_no_baseline_reselection": True,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    validate_strongest_timing_config_v4(config, strongest, selection)
    command = strongest_timing_command_v4(config, config["output_dir"] + ".jsonl")
    assert command[command.index("--method") + 1] == "genetic"
    assert command[command.index("--seed") + 1] == "20260816"
    assert command[command.index("--max-critic-forwards") + 1] == "320"
    changed = dict(config)
    changed["genetic_population_size"] = 64
    with pytest.raises(Exception, match="hyperparameters differ"):
        validate_strongest_timing_config_v4(changed, strongest, selection)


def _closed(method: str, seed: int, value: float, *, source_count: int = 891):
    per_source = {
        f"s{index:03d}": {
            "status": "DEFINED",
            "ndcg": value,
            "normalized_regret": 0.4,
            "top_1_recall": 0.5,
        }
        for index in range(source_count)
    }
    result = {
        "schema_version": "route_a_v3_route2_xeditflow_closed_neighborhood.v4",
        "status": "XEDITFLOW_V4_CLOSED_NEIGHBORHOOD_COMPLETE",
        "method_id": method,
        "base_flow_training_seed": seed,
        "source_macro_ndcg": value,
        "source_macro_normalized_regret": 0.4,
        "source_macro_top_1_recall": 0.5,
        "undefined_sources_are_not_filled_with_zero": True,
        "per_source": per_source,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    if method in V4_GENERATED_METHODS:
        result.update({"kappa": 0.5, "temperature": 1.0, "beta_max": 2.0})
    return result


def _compute(source_key: str):
    return {
        "schema_version": "MatchedComputeRecordV4",
        "source_key": source_key,
        "trunk_forwards": 10,
        "mode_forwards": 80,
        "value_forwards": 10,
        "critic_forwards_by_member": [2, 2, 2],
        "total_forward_equivalents": 106,
        "failure_counters": {
            "edit_budget_violation_count": 0,
            "candidate_budget_violation_count": 0,
            "replay_failure_count": 0,
            "numerical_failure_count": 0,
        },
        "terminal_critic_reservation_reconciled": True,
        "terminal_critic_forwards_are_reserved_pending_scoring": False,
        "trajectory_critic_forwards_preserved_during_reconciliation": True,
        "source_equal_wall_time_seconds": 1.0,
        "source_equal_wall_time_scope": EQUAL_WALL_TIME_SCOPE_V3,
        "source_equal_wall_peak_vram_mb": 2000.0,
        "source_cuda_device_name": "NVIDIA A100-SXM4-40GB",
    }


def test_v4_equal_wall_requires_terminal_reconciled_compute() -> None:
    seed = 20260912
    source_rows = [{"source_key": f"s{index:03d}"} for index in range(891)]
    closed = {method: _closed(method, seed, 0.7) for method in METHODS_V4}
    timing = {
        method: [_compute(str(row["source_key"])) for row in source_rows]
        for method in V4_GENERATED_METHODS
    }
    timing["strongest_matched_baseline"] = [
        {
            "source_key": row["source_key"],
            "source_equal_wall_time_seconds": 1.0,
            "source_equal_wall_time_scope": EQUAL_WALL_TIME_SCOPE_V3,
            "source_equal_wall_peak_vram_mb": 1000.0,
            "cuda_device_name": "NVIDIA A100-SXM4-40GB",
        }
        for row in source_rows
    ]
    config = {
        "schema_version": "route_a_v3_route2_xeditflow_equal_wall_time_config.v4",
        "base_flow_training_seed": seed,
        "source_manifest_path": "/unused/source.jsonl",
        "methods": {
            method: {
                "timing_path": f"/unused/{method}.jsonl",
                "timing_format": (
                    SEARCH_CANDIDATE_JSONL
                    if method == "strongest_matched_baseline"
                    else MATCHED_COMPUTE_SCORED_JSONL
                ),
                "closed_summary_path": f"/unused/{method}.json",
            }
            for method in METHODS_V4
        },
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    result = build_equal_wall_time_sensitivity_v4(
        config,
        source_rows=source_rows,
        timing_rows=timing,
        closed_results=closed,
    )
    assert result["status"] == "XEDITFLOW_V4_EQUAL_WALL_TIME_SENSITIVITY_COMPLETE"
    assert result["common_source_prefix_count"] == 891
    assert result["all_network_forwards_separately_charged"] is True
    timing["full_soft_value_smc"][0]["terminal_critic_reservation_reconciled"] = False
    with pytest.raises(Exception, match="reconciled compute accounting"):
        build_equal_wall_time_sensitivity_v4(
            config,
            source_rows=source_rows,
            timing_rows=timing,
            closed_results=closed,
        )


def _open(method: str, seed: int):
    result = {
        "status": "XEDITFLOW_V4_OPEN_GENERATION_METRICS_COMPLETE",
        "method_id": method,
        "base_flow_training_seed": seed,
        "source_macro_candidate_recovery": 0.30,
        "source_macro_top_k_recovery": 0.20,
        "source_macro_unique_candidate_rate": 0.95,
        "hard_legality_rate": 1.0,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    if method in V4_GENERATED_METHODS:
        result.update({"kappa": 0.5, "temperature": 1.0, "beta_max": 2.0})
    return result


def _generation(method: str, seed: int):
    if method == "strongest_matched_baseline":
        return {
            "status": "XEDITFLOW_V4_STRONGEST_BASELINE_ADAPTER_COMPLETE",
            "method_id": method,
            "base_flow_training_seed": seed,
            "maximum_forward_equivalents_per_source": 320,
            "hard_legality_rate": 1.0,
            "edit_budget_violation_count": 0,
            "candidate_budget_violation_count": 0,
            "trajectory_replay_failure_count": 0,
            "numerical_failure_count": 0,
            "frozen_baseline_reselected_for_v4": False,
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
    result = {
        "status": "XEDITFLOW_V4_SMC_GENERATION_COMPLETE_PENDING_TERMINAL_CRITIC_SCORING",
        "method_id": method,
        "base_flow_training_seed": seed,
        "kappa": 0.5,
        "temperature": 1.0,
        "beta_max": 2.0,
        "setflow_mode_is_fixed_trajectory_state": True,
        "free_action_ratio_head_used": False,
        "hard_legality_rate": 1.0,
        "edit_budget_violation_count": 0,
        "candidate_budget_violation_count": 0,
        "replay_failure_count": 0,
        "numerical_failure_count": 0,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    if method == "full_soft_value_smc":
        result.update(
            {
                "value_checkpoint_path": _value_checkpoint_path(seed),
                "value_training_provenance": _value_provenance(seed),
            }
        )
    return result


def _terminal(method: str, seed: int):
    return {
        "status": "XEDITFLOW_V4_CANDIDATE_CRITIC_SCORING_COMPLETE",
        "method_id": method,
        "base_flow_training_seed": seed,
        "kappa": 0.5,
        "temperature": 1.0,
        "beta_max": 2.0,
        "maximum_total_forward_equivalents_per_source": 300,
        "reservation_reconciled_for_every_source": True,
        "candidate_support_unchanged_by_terminal_rerank": True,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def _equal_wall(seed: int):
    return {
        "status": "XEDITFLOW_V4_EQUAL_WALL_TIME_SENSITIVITY_COMPLETE",
        "base_flow_training_seed": seed,
        "common_source_prefix_count": 2,
        "methods": {
            method: {
                "accelerator_name": "NVIDIA A100-SXM4-40GB",
                "full_cohort_generation_wall_time_seconds": 100.0,
                "common_prefix_generation_wall_time_seconds": 2.0,
                "peak_vram_mb": 2000.0,
                "source_macro_ndcg": 0.7,
                "source_macro_normalized_regret": 0.3,
                "source_macro_top_1_recall": 0.5,
            }
            for method in METHODS_V4
        },
        "five_v4_methods_use_terminal_scoring_reconciled_compute": True,
        "all_network_forwards_separately_charged": True,
        "matched_compute_schema": "MatchedComputeRecordV4",
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def _assembled(seed: int, evidence_mutator=None):
    values = {method: 0.70 for method in METHODS_V4}
    values["full_soft_value_smc"] = 0.85
    values["unguided_setflow"] = 0.65
    values["strongest_matched_baseline"] = 0.65
    evidence = {}
    for method in METHODS_V4:
        closed = _closed(method, seed, values[method], source_count=3)
        closed["per_source"]["s002"] = {
            "status": "UNDEFINED_ZERO_MEASURED_GAIN",
            "ndcg": None,
            "normalized_regret": 0.4,
            "top_1_recall": 0.5,
        }
        closed["source_macro_ndcg"] = values[method]
        regret = 0.30 if method == "full_soft_value_smc" else 0.40
        top_1 = 0.60 if method == "full_soft_value_smc" else 0.50
        for row in closed["per_source"].values():
            row["normalized_regret"] = regret
            row["top_1_recall"] = top_1
        closed["source_macro_normalized_regret"] = regret
        closed["source_macro_top_1_recall"] = top_1
        bundle = {
            "closed": closed,
            "open": _open(method, seed),
            "generation": _generation(method, seed),
        }
        if method in V4_GENERATED_METHODS:
            bundle["terminal_critic"] = _terminal(method, seed)
        evidence[method] = bundle
    if evidence_mutator is not None:
        evidence_mutator(evidence)
    evaluator = {
        "status": "XEDITFLOW_V4_INDEPENDENT_EVALUATOR_COMPARISON_COMPLETE",
        "analysis_unit": "SOURCE",
        "base_flow_training_seed": seed,
        "combination": [0.5, 1.0, 2.0],
        "paired_margin_over_strongest_baseline": 0.20,
        "per_source_paired_margin": {"s000": 0.10, "s001": 0.30},
        "independent_evaluator_used_for_gradient": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcome_reads": 0,
    }
    candidate = lambda method, value: [
        {
            "method_id": method,
            "base_flow_training_seed": seed,
            "source_key": source,
            "critic_self_score": value,
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
        for source in ("s000", "s001")
    ]
    return assemble_final_seed_evidence_v4(
        evidence,
        base_flow_training_seed=seed,
        selected_combination=[0.5, 1.0, 2.0],
        value_checkpoint_path=_value_checkpoint_path(seed),
        equal_wall_time_sensitivity=_equal_wall(seed),
        full_independent_evaluator=evaluator,
        full_candidate_rows=candidate("full_soft_value_smc", 2.0),
        unguided_candidate_rows=candidate("unguided_setflow", 1.0),
        bootstrap_iterations=10_000,
        bootstrap_seed=20261001 + seed,
    )


@pytest.mark.parametrize(
    ("field", "message"),
    (
        (
            "source_macro_normalized_regret",
            "macro regret is not the source-defined mean",
        ),
        (
            "source_macro_top_1_recall",
            "macro top-1 is not the source-defined mean",
        ),
    ),
)
def test_v4_final_evidence_recomputes_all_closed_source_macros(
    field: str, message: str
) -> None:
    def tamper(evidence) -> None:
        evidence["full_soft_value_smc"]["closed"][field] += 0.01

    with pytest.raises(Exception, match=message):
        _assembled(20260912, evidence_mutator=tamper)


@pytest.mark.parametrize("bundle_name", ("closed", "open"))
def test_v4_final_evidence_binds_generated_metric_combination(
    bundle_name: str,
) -> None:
    def tamper(evidence) -> None:
        evidence["simple_rate_guidance"][bundle_name]["beta_max"] = 1.0

    with pytest.raises(Exception, match="generation combination differs"):
        _assembled(20260912, evidence_mutator=tamper)


def test_v4_final_evidence_and_three_seed_gate_are_exact_and_terminal(tmp_path) -> None:
    rows = []
    for seed in (20260912, 20260913, 20260914):
        row = write_final_seed_evidence_v4(
            _assembled(seed), output_dir=tmp_path / str(seed)
        )
        rows.append(row)
    result = adjudicate_final_manifest_v4(
        {
            "schema_version": "route_a_v3_route2_xeditflow_final_comparison_manifest.v4",
            "status": "XEDITFLOW_V4_FINAL_COMPARISON_RESULTS_COMPLETE",
            "guidance_screen_status": "XEDITFLOW_V4_GUIDANCE_SCREEN_FROZEN",
            "selected_combination": [0.5, 1.0, 2.0],
            "value_checkpoint_paths": {
                str(seed): _value_checkpoint_path(seed)
                for seed in (20260912, 20260913, 20260914)
            },
            "guidance_screen_value_checkpoint_path": _value_checkpoint_path(
                20260912
            ),
            "guidance_screen_value_training_provenance": _value_provenance(
                20260912
            ),
            "seeds": rows,
        }
    )
    assert result["gate"]["status"] == "XEDITFLOW_V4_PASS"
    assert result["new_final_evaluation_authorized"] is True
    assert result["submission_ready"] is False
    assert result["additional_training_seed_authorized"] is False


def test_v4_final_gate_rejects_non_v4_compute_evidence(tmp_path) -> None:
    rows = []
    for seed in (20260912, 20260913, 20260914):
        rows.append(
            write_final_seed_evidence_v4(
                _assembled(seed), output_dir=tmp_path / str(seed)
            )
        )
    bootstrap_path = tmp_path / "20260912" / "paired_bootstrap.json"
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    bootstrap["matched_compute_schema"] = "V3"
    bootstrap_path.write_text(json.dumps(bootstrap), encoding="utf-8")
    with pytest.raises(Exception, match="paired-bootstrap evidence differs"):
        adjudicate_final_manifest_v4(
            {
                "schema_version": (
                    "route_a_v3_route2_xeditflow_final_comparison_manifest.v4"
                ),
                "status": "XEDITFLOW_V4_FINAL_COMPARISON_RESULTS_COMPLETE",
                "guidance_screen_status": "XEDITFLOW_V4_GUIDANCE_SCREEN_FROZEN",
                "selected_combination": [0.5, 1.0, 2.0],
                "value_checkpoint_paths": {
                    str(seed): _value_checkpoint_path(seed)
                    for seed in (20260912, 20260913, 20260914)
                },
                "guidance_screen_value_checkpoint_path": _value_checkpoint_path(
                    20260912
                ),
                "guidance_screen_value_training_provenance": _value_provenance(
                    20260912
                ),
                "seeds": rows,
            }
        )


def test_v4_final_gate_rejects_value_training_provenance_drift(tmp_path) -> None:
    rows = [
        write_final_seed_evidence_v4(_assembled(seed), output_dir=tmp_path / str(seed))
        for seed in (20260912, 20260913, 20260914)
    ]
    rows[1]["value_training_provenance"]["bf16_supported"] = False
    manifest = {
        "schema_version": "route_a_v3_route2_xeditflow_final_comparison_manifest.v4",
        "status": "XEDITFLOW_V4_FINAL_COMPARISON_RESULTS_COMPLETE",
        "guidance_screen_status": "XEDITFLOW_V4_GUIDANCE_SCREEN_FROZEN",
        "selected_combination": [0.5, 1.0, 2.0],
        "value_checkpoint_paths": {
            str(seed): _value_checkpoint_path(seed)
            for seed in (20260912, 20260913, 20260914)
        },
        "guidance_screen_value_checkpoint_path": _value_checkpoint_path(20260912),
        "guidance_screen_value_training_provenance": _value_provenance(20260912),
        "seeds": rows,
    }
    with pytest.raises(Exception, match="BF16/no-CPU"):
        adjudicate_final_manifest_v4(manifest)


def test_v4_final_composer_accepts_only_exact_three_route2_rows() -> None:
    prefix = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/final"
    rows = {
        seed: {
            "base_flow_training_seed": seed,
            "methods": {
                method: f"{prefix}/{seed}/{method}.json" for method in METHODS_V4
            },
            "paired_bootstrap_path": f"{prefix}/{seed}/bootstrap.json",
            "equal_wall_time_sensitivity_path": f"{prefix}/{seed}/equal.json",
            "value_checkpoint_path": _value_checkpoint_path(seed),
            "value_training_provenance": _value_provenance(seed),
        }
        for seed in (20260912, 20260913, 20260914)
    }
    config = {
        "schema_version": "route_a_v3_route2_xeditflow_final_comparison_compose_config.v4",
        "expected_value_checkpoint_paths": {
            str(seed): _value_checkpoint_path(seed)
            for seed in (20260912, 20260913, 20260914)
        },
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    gate = {
        "schema_version": "route_a_v3_route2_xeditflow_v4_guidance_screen_gate.v1",
        "status": "XEDITFLOW_V4_GUIDANCE_SCREEN_FROZEN",
        "base_flow_training_seed": 20260912,
        "combination_count": 18,
        "selected_kappa": 0.5,
        "selected_temperature": 1.0,
        "selected_beta_max": 2.0,
        "selected_value_checkpoint_path": _value_checkpoint_path(20260912),
        "selected_value_training_provenance": _value_provenance(20260912),
    }
    result = compose_final_comparison_manifest_v4(config, gate, rows)
    assert result["status"] == "XEDITFLOW_V4_FINAL_COMPARISON_RESULTS_COMPLETE"
    assert result["selected_combination"] == [0.5, 1.0, 2.0]
    broken = dict(rows)
    broken.pop(20260914)
    with pytest.raises(Exception, match="seed-row inventory differs"):
        compose_final_comparison_manifest_v4(config, gate, broken)
