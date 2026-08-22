#!/usr/bin/env python3
"""Prepare frozen per-seed value and matched-generation jobs after guidance screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_xeditflow_gate_v3 import authorize_xeditflow_guidance_v3
from scripts.route_a_v3.prepare_route2_xeditflow_guidance_screen_configs_v3 import combination_id_v3
from scripts.route_a_v3.run_route2_xeditflow_matched_controls_v3 import METHODS


SEEDS = (20260904, 20260905, 20260906)
FROZEN_SEARCH_BASELINES = (
    "random_legal",
    "greedy",
    "beam",
    "genetic",
    "local_search",
)


class XEditFlowFinalGenerationPrepareV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowFinalGenerationPrepareV3Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def build_final_generation_configs_v3(
    config: Mapping[str, Any],
    *,
    critic_readiness: Mapping[str, Any],
    setflow_confirmation: Mapping[str, Any],
    critic_refit_manifest: Mapping[str, Any],
    guidance_screen_gate: Mapping[str, Any],
    strongest_generation_baseline: Mapping[str, Any],
    setflow_runtimes: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    _require(config.get("schema_version") == "route_a_v3_route2_xeditflow_final_generation_prepare.v1", "unexpected final-generation prepare schema")
    authorization = authorize_xeditflow_guidance_v3(critic_readiness, setflow_confirmation)
    _require(authorization["guidance_authorized"] is True, "final generation remains blocked before readiness")
    _require(
        critic_refit_manifest.get("status") == "XEDITCRITIC_V3_ALL_DEVELOPMENT_REFIT_COMPLETE"
        and len(critic_refit_manifest.get("checkpoints", ())) == 3,
        "final generation Critic refit differs",
    )
    ordered_critic_checkpoints = sorted(
        critic_refit_manifest["checkpoints"], key=lambda row: int(row["seed"])
    )
    _require(
        tuple(int(row["seed"]) for row in ordered_critic_checkpoints)
        == (20260831, 20260901, 20260902),
        "final generation Critic refit seeds differ",
    )
    critic_checkpoint_paths = [
        str(row["checkpoint_path"])
        for row in ordered_critic_checkpoints
    ]
    _require(len(set(critic_checkpoint_paths)) == 3, "final generation Critic checkpoint paths differ")
    _require(
        strongest_generation_baseline.get("status")
        == "DEVELOPMENT_STRONGEST_GENERATION_BASELINE_FROZEN_INDEPENDENT_EVALUATOR_ONLY"
        and strongest_generation_baseline.get("strongest_generation_baseline_id") == "genetic"
        and strongest_generation_baseline.get("evaluation_outcomes_accessed") is False
        and int(strongest_generation_baseline.get("forward_equivalent_budget_per_source", -1)) == 320,
        "final generation strongest baseline differs",
    )
    _require(
        str(config["independent_evaluator_checkpoint_path"])
        == str(strongest_generation_baseline["independent_evaluator_checkpoint_path"]),
        "final generation independent evaluator differs from frozen strongest selection",
    )
    _require(
        guidance_screen_gate.get("status") == "XEDITFLOW_V3_GUIDANCE_SCREEN_FROZEN"
        and int(guidance_screen_gate.get("base_flow_training_seed", -1)) == 20260904,
        "final generation guidance screen is not frozen",
    )
    kappa = float(guidance_screen_gate["selected_kappa"])
    temperature = float(guidance_screen_gate["selected_temperature"])
    beta_max = float(guidance_screen_gate["selected_beta_max"])
    _require(kappa in {0.0, 0.5, 1.0} and temperature in {0.5, 1.0} and beta_max in {0.5, 1.0, 2.0}, "final generation selected combination differs")
    selected_arm = str(setflow_confirmation.get("selected_arm"))
    _require(selected_arm in {"f2", "f3"}, "final generation SetFlow arm differs")
    _require(tuple(sorted(setflow_runtimes)) == SEEDS, "final generation SetFlow runtime seeds differ")
    gpu = int(config.get("physical_gpu_index", -1))
    _require(gpu in set(range(6)), "final generation GPU is outside 0-5")
    output_root = Path(str(config["output_root"]))
    _require(str(output_root).startswith("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"), "final generation outputs left Route 2 /mnt")
    timed_strongest_root = output_root / "benchmark_resources" / "strongest_matched_baseline_timing"
    timed_strongest_candidate_path = timed_strongest_root / "timed_genetic_candidates.private.jsonl"
    value_id = combination_id_v3(kappa, temperature)
    screen_value_checkpoint = Path(str(config["guidance_screen_output_root"])) / "value_models" / value_id / "value_checkpoint.pt"
    seed_jobs = []
    for seed in SEEDS:
        runtime = setflow_runtimes[seed]
        _require(int(runtime.get("seed", -1)) == seed and str(runtime.get("selected_arm")) == selected_arm, f"final SetFlow runtime identity differs: {seed}")
        seed_root = output_root / f"seed{seed}"
        setflow_checkpoint = Path(str(runtime["output_root"])) / selected_arm / "best.pt"
        value_rollout_config = None
        value_target_config = None
        value_training_config = None
        if seed == 20260904:
            value_checkpoint = screen_value_checkpoint
        else:
            rollout_root = seed_root / "value_rollouts"
            target_root = seed_root / "value_targets"
            model_root = seed_root / "value_model"
            value_checkpoint = model_root / "value_checkpoint.pt"
            value_rollout_config = {
                "schema_version": "route_a_v3_route2_xeditflow_value_rollout_config.v1",
                "status": "FROZEN_FINAL_VALUE_ROLLOUT_CONFIG_NOT_STARTED",
                "critic_readiness_path": str(config["critic_readiness_path"]),
                "setflow_confirmation_path": str(config["setflow_confirmation_path"]),
                "critic_refit_manifest_path": str(config["critic_refit_manifest_path"]),
                "train_projection_path": str(runtime["train_projection_path"]),
                "source_token_cache_path": str(runtime["source_token_cache_path"]),
                "setflow_checkpoint_path": str(setflow_checkpoint),
                "setflow_arm": selected_arm,
                "mrnabert_model_path": str(config["mrnabert_model_path"]),
                "expected_train_record_count": 68294,
                "base_flow_training_seed": seed,
                "states_per_record": 2,
                "state_pass_index": 0,
                "rollouts_per_state": 8,
                "sampling_state_batch_size": 128,
                "trajectory_forward_batch_size": 64,
                "critic_batch_size": 256,
                "critic_online_microbatch_size": 4,
                "physical_gpu_index": gpu,
                "device": f"cuda:{gpu}",
                "output_dir": str(rollout_root),
                "independent_evaluator_used": False,
                "development_test_outcomes_accessed": False,
                "new_final_evaluation_outcomes_accessed": False,
            }
            value_target_config = {
                "schema_version": "route_a_v3_route2_xeditflow_value_target_build_config.v1",
                "train_state_path": str(rollout_root / "train_states.jsonl"),
                "frozen_rollout_score_path": str(rollout_root / "frozen_rollout_scores.private.jsonl"),
                "critic_readiness_path": str(config["critic_readiness_path"]),
                "setflow_confirmation_path": str(config["setflow_confirmation_path"]),
                "base_flow_training_seed": seed,
                "kappa": kappa,
                "temperature": temperature,
                "output_dir": str(target_root),
            }
            value_training_config = {
                "schema_version": "route_a_v3_route2_xeditflow_value_training_config.v1",
                "value_target_path": str(target_root / "value_targets.pt"),
                "critic_readiness_path": str(config["critic_readiness_path"]),
                "setflow_confirmation_path": str(config["setflow_confirmation_path"]),
                "source_token_cache_path": str(runtime["source_token_cache_path"]),
                "experiment_ledger_path": str(runtime["experiment_ledger_path"]),
                "base_flow_training_seed": seed,
                "kappa": kappa,
                "temperature": temperature,
                "passes": 8,
                "batch_size": 32,
                "precision": "BF16",
                "learning_rate": 3e-4,
                "weight_decay": 1e-4,
                "gradient_clip_norm": 1.0,
                "dropout": 0.10,
                "checkpoint_selection": "FINAL_PASS_NO_EPOCH_RESELECTION",
                "physical_gpu_index": gpu,
                "device": f"cuda:{gpu}",
                "output_dir": str(model_root),
            }
        common = {
            "critic_readiness_path": str(config["critic_readiness_path"]),
            "setflow_confirmation_path": str(config["setflow_confirmation_path"]),
            "critic_refit_manifest_path": str(config["critic_refit_manifest_path"]),
            "mrnabert_model_path": str(config["mrnabert_model_path"]),
            "setflow_arm": selected_arm,
            "setflow_checkpoint_path": str(setflow_checkpoint),
            "source_token_cache_path": str(runtime["source_token_cache_path"]),
            "source_eligibility_manifest": str(runtime["source_eligibility_manifest"]),
            "validation_projection_path": str(runtime["validation_projection_path"]),
            "expected_source_count": 891,
            "base_flow_training_seed": seed,
            "kappa": kappa,
            "beta_max": beta_max,
            "particle_count": 32,
            "candidate_cap": 32,
            "ess_threshold": 16.0,
            "resampling": "STRATIFIED",
            "forward_equivalent_ceiling_per_source": 320,
            "reserved_terminal_critic_forwards": 3,
            "maximum_sampling_rounds": 32,
            "action_space": "SUB+STOP",
            "replay_check": True,
            "decoder_seed_base": int(config["decoder_seed_base"]),
            "physical_gpu_index": gpu,
            "device": f"cuda:{gpu}",
        }
        full_smc_config = {
            **common,
            "schema_version": "route_a_v3_route2_xeditflow_smc_run_config.v1",
            "method_id": "full_soft_value_smc",
            "temperature": temperature,
            "value_checkpoint_path": str(value_checkpoint),
            "output_dir": str(seed_root / "generation" / "full_soft_value_smc"),
        }
        controls = {}
        for method in sorted(METHODS):
            controls[method] = {
                **common,
                "schema_version": "route_a_v3_route2_xeditflow_matched_control_run_config.v1",
                "method_id": method,
                "critic_online_microbatch_size": 4,
                "output_dir": str(seed_root / "generation" / method),
            }
        closed_common = {
            "critic_readiness_path": str(config["critic_readiness_path"]),
            "setflow_confirmation_path": str(config["setflow_confirmation_path"]),
            "setflow_arm": selected_arm,
            "setflow_checkpoint_path": str(setflow_checkpoint),
            "source_token_cache_path": str(runtime["source_token_cache_path"]),
            "source_eligibility_manifest": str(runtime["source_eligibility_manifest"]),
            "validation_projection_path": str(runtime["validation_projection_path"]),
            "measured_neighborhood_path": str(config["measured_neighborhood_path"]),
            "expected_source_count": 891,
            "base_flow_training_seed": seed,
            "kappa": kappa,
            "temperature": temperature,
            "beta_max": beta_max,
            "pool_assignment": "DEVELOPMENT",
            "split": "VALIDATION",
            "maximum_enumerated_edits": 5,
            "maximum_permutation_paths": 120,
            "enumeration": "ALL_EDIT_PERMUTATIONS_EXACT_SUM",
            "analysis_unit": "SOURCE",
            "undefined_source_policy": "EXCLUDE_NOT_ZERO_FILL",
            "physical_gpu_index": gpu,
            "device": f"cuda:{gpu}",
        }
        closed_trajectory_configs = {
            "full_soft_value_smc": {
                **closed_common,
                "schema_version": "route_a_v3_route2_xeditflow_closed_neighborhood_config.v1",
                "method_id": "full_soft_value_smc",
                "potential_kind": "SOFT_VALUE",
                "value_checkpoint_path": str(value_checkpoint),
                "output_dir": str(seed_root / "closed" / "full_soft_value_smc"),
            },
            "unguided_setflow": {
                **closed_common,
                "schema_version": "route_a_v3_route2_xeditflow_closed_neighborhood_config.v1",
                "method_id": "unguided_setflow",
                "potential_kind": "ZERO",
                "output_dir": str(seed_root / "closed" / "unguided_setflow"),
            },
        }
        closed_frozen_score_configs = {}
        closed_score_metric_configs = {}
        for method in (
            "first_order_guidance",
            "simple_rate_guidance",
            "generate_then_rerank",
            "strongest_matched_baseline",
        ):
            score_root = seed_root / "closed_scores" / method
            score_config = {
                "schema_version": "route_a_v3_route2_xeditflow_closed_frozen_score_config.v1",
                "method_id": method,
                "critic_readiness_path": str(config["critic_readiness_path"]),
                "setflow_confirmation_path": str(config["setflow_confirmation_path"]),
                "source_eligibility_manifest": str(runtime["source_eligibility_manifest"]),
                "measured_neighborhood_path": str(config["measured_neighborhood_path"]),
                "pool_assignment": "DEVELOPMENT",
                "split": "VALIDATION",
                "expected_source_count": 891,
                "base_flow_training_seed": seed,
                "physical_gpu_index": gpu,
                "device": f"cuda:{gpu}",
                "output_dir": str(score_root),
            }
            if method != "strongest_matched_baseline":
                score_config.update(
                    {
                        "critic_refit_manifest_path": str(config["critic_refit_manifest_path"]),
                        "mrnabert_model_path": str(config["mrnabert_model_path"]),
                        "validation_projection_path": str(runtime["validation_projection_path"]),
                        "kappa": kappa,
                        "critic_online_microbatch_size": 4,
                    }
                )
            else:
                score_config.update(
                    {
                        "strongest_generation_baseline_path": str(config["strongest_generation_baseline_path"]),
                        "baseline_selection_input_path": str(config["baseline_selection_input_path"]),
                    }
                )
            closed_frozen_score_configs[method] = score_config
            closed_score_metric_configs[method] = {
                "schema_version": "route_a_v3_route2_xeditflow_closed_score_config.v1",
                "method_id": method,
                "score_table_method_id": method,
                "base_flow_training_seed": seed,
                "pool_assignment": "DEVELOPMENT",
                "split": "VALIDATION",
                "analysis_unit": "SOURCE",
                "undefined_source_policy": "EXCLUDE_NOT_ZERO_FILL",
                "score_transform": "SOURCEWISE_EXP_SHIFTED_MAX",
                "measured_neighborhood_path": str(config["measured_neighborhood_path"]),
                "score_table_path": str(score_root / "frozen_method_scores.private.jsonl"),
            }
        strongest_adapter_job = {
            "strongest_generation_baseline_path": str(config["strongest_generation_baseline_path"]),
            "baseline_selection_input_path": str(config["baseline_selection_input_path"]),
            "base_flow_training_seed": seed,
            "output_dir": str(seed_root / "generation" / "strongest_matched_baseline"),
        }
        full_generation_root = seed_root / "generation" / "full_soft_value_smc"
        full_raw_candidates = full_generation_root / "generated_candidates.private.jsonl"
        full_critic_root = full_generation_root / "critic_ensemble"
        full_scored_candidates = full_critic_root / "critic_scored_candidates.private.jsonl"
        full_independent_scored = full_generation_root / "independent_evaluator_scored_candidates.private.jsonl"
        metric_root = seed_root / "metrics"
        full_critic_score_config = {
            "schema_version": "route_a_v3_route2_xeditflow_critic_ensemble_score_config.v1",
            "critic_readiness_path": str(config["critic_readiness_path"]),
            "setflow_confirmation_path": str(config["setflow_confirmation_path"]),
            "critic_refit_manifest_path": str(config["critic_refit_manifest_path"]),
            "mrnabert_model_path": str(config["mrnabert_model_path"]),
            "source_eligibility_manifest": str(runtime["source_eligibility_manifest"]),
            "validation_projection_path": str(runtime["validation_projection_path"]),
            "candidate_path": str(full_raw_candidates),
            "method_id": "full_soft_value_smc",
            "base_flow_training_seed": seed,
            "kappa": kappa,
            "critic_batch_size": 256,
            "critic_online_microbatch_size": 4,
            "physical_gpu_index": gpu,
            "device": f"cuda:{gpu}",
            "output_dir": str(full_critic_root),
        }
        open_metric_configs = {}
        open_metric_output_paths = {}
        for method in (
            "full_soft_value_smc",
            "unguided_setflow",
            "first_order_guidance",
            "simple_rate_guidance",
            "generate_then_rerank",
        ):
            candidate_path = (
                full_scored_candidates
                if method == "full_soft_value_smc"
                else seed_root / "generation" / method / "generated_candidates.private.jsonl"
            )
            open_metric_configs[method] = {
                "schema_version": "route_a_v3_route2_xeditflow_open_generation_config.v1",
                "pool_assignment": "DEVELOPMENT",
                "candidate_support_mode": "OPEN_GENERATED_SUPPORT",
                "undefined_outcome_policy": "UNKNOWN_NOT_ZERO",
                "source_eligibility_manifest": str(runtime["source_eligibility_manifest"]),
                "candidate_path": str(candidate_path),
                "measured_neighborhood_path": str(config["measured_neighborhood_path"]),
                "measured_top_k": 10,
            }
            open_metric_output_paths[method] = str(metric_root / f"open_{method}.json")
        independent_evaluator_config = {
            "schema_version": "route_a_v3_route2_generation_independent_evaluator_job.v1",
            "method_id": "full_soft_value_smc",
            "evaluator_checkpoint_path": str(config["independent_evaluator_checkpoint_path"]),
            "guiding_checkpoint_paths": critic_checkpoint_paths,
            "source_manifest_path": str(runtime["source_eligibility_manifest"]),
            "evaluator_frozen_before_candidate_generation": True,
            "evaluation_outcomes_used_to_select_evaluator": 0,
            "device": f"cuda:{gpu}",
            "physical_gpu_index": gpu,
            "candidate_path": str(full_scored_candidates),
        }
        independent_evaluator_comparison_config = {
            "schema_version": "route_a_v3_route2_xeditflow_independent_evaluator_comparison_config.v1",
            "strongest_baseline_path": str(config["strongest_generation_baseline_path"]),
            "baseline_selection_input_path": str(config["baseline_selection_input_path"]),
            "source_eligibility_manifest": str(runtime["source_eligibility_manifest"]),
            "guided_scored_candidate_path": str(full_independent_scored),
            "bootstrap_iterations": 10_000,
            "bootstrap_seed": int(config["decoder_seed_base"]) + seed,
        }
        closed_summary_paths = {
            method: str(seed_root / "closed" / method / "run_summary.json")
            for method in closed_trajectory_configs
        }
        closed_score_metric_output_paths = {
            method: str(metric_root / f"closed_{method}.json")
            for method in closed_score_metric_configs
        }
        closed_summary_paths.update(closed_score_metric_output_paths)
        generation_summary_paths = {
            "full_soft_value_smc": str(full_generation_root / "run_summary.json"),
            **{
                method: str(seed_root / "generation" / method / "run_summary.json")
                for method in controls
            },
            "strongest_matched_baseline": str(
                seed_root / "generation" / "strongest_matched_baseline" / "generation.json"
            ),
        }
        open_summary_paths = {
            **open_metric_output_paths,
            "strongest_matched_baseline": str(
                seed_root / "generation" / "strongest_matched_baseline" / "open.json"
            ),
        }
        equal_wall_time_output_path = metric_root / "equal_wall_time_sensitivity.json"
        equal_wall_time_config = {
            "schema_version": "route_a_v3_route2_xeditflow_equal_wall_time_config.v1",
            "base_flow_training_seed": seed,
            "source_manifest_path": str(runtime["source_eligibility_manifest"]),
            "methods": {
                method: {
                    "timing_path": (
                        str(timed_strongest_candidate_path)
                        if method == "strongest_matched_baseline"
                        else str(seed_root / "generation" / method / "matched_compute.jsonl")
                    ),
                    "timing_format": (
                        "SEARCH_CANDIDATE_JSONL"
                        if method == "strongest_matched_baseline"
                        else "MATCHED_COMPUTE_JSONL"
                    ),
                    "closed_summary_path": closed_summary_paths[method],
                }
                for method in (
                    "full_soft_value_smc",
                    "unguided_setflow",
                    "first_order_guidance",
                    "simple_rate_guidance",
                    "generate_then_rerank",
                    "strongest_matched_baseline",
                )
            },
            "development_test_outcomes_accessed": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
        final_seed_evidence_config = {
            "schema_version": "route_a_v3_route2_xeditflow_final_seed_evidence_config.v1",
            "base_flow_training_seed": seed,
            "methods": {
                method: {
                    "closed_summary_path": closed_summary_paths[method],
                    "open_summary_path": open_summary_paths[method],
                    "generation_summary_path": generation_summary_paths[method],
                }
                for method in (
                    "full_soft_value_smc",
                    "unguided_setflow",
                    "first_order_guidance",
                    "simple_rate_guidance",
                    "generate_then_rerank",
                    "strongest_matched_baseline",
                )
            },
            "full_independent_evaluator_path": str(
                metric_root / "independent_evaluator_full_vs_strongest.json"
            ),
            "full_candidate_path": str(full_scored_candidates),
            "unguided_candidate_path": str(
                seed_root / "generation" / "unguided_setflow" / "generated_candidates.private.jsonl"
            ),
            "bootstrap_iterations": 10_000,
            "bootstrap_seed": int(config["decoder_seed_base"]) + seed + 100_000,
            "equal_wall_time_sensitivity_path": str(equal_wall_time_output_path),
        }
        seed_jobs.append(
            {
                "base_flow_training_seed": seed,
                "setflow_checkpoint_path": str(setflow_checkpoint),
                "value_checkpoint_path": str(value_checkpoint),
                "value_rollout_config": value_rollout_config,
                "value_target_config": value_target_config,
                "value_training_config": value_training_config,
                "full_smc_config": full_smc_config,
                "matched_control_configs": controls,
                "closed_trajectory_configs": closed_trajectory_configs,
                "closed_frozen_score_configs": closed_frozen_score_configs,
                "closed_score_metric_configs": closed_score_metric_configs,
                "strongest_adapter_job": strongest_adapter_job,
                "full_critic_score_config": full_critic_score_config,
                "open_metric_configs": open_metric_configs,
                "open_metric_output_paths": open_metric_output_paths,
                "independent_evaluator_config": independent_evaluator_config,
                "independent_evaluator_scored_candidate_path": str(full_independent_scored),
                "independent_evaluator_comparison_config": independent_evaluator_comparison_config,
                "independent_evaluator_comparison_output_path": str(
                    metric_root / "independent_evaluator_full_vs_strongest.json"
                ),
                "closed_score_metric_output_paths": closed_score_metric_output_paths,
                "equal_wall_time_config": equal_wall_time_config,
                "equal_wall_time_output_path": str(equal_wall_time_output_path),
                "final_seed_evidence_config": final_seed_evidence_config,
                "final_seed_evidence_output_dir": str(seed_root / "final_evidence"),
            }
        )
    _require(len(seed_jobs) == 3 and sum(job["value_rollout_config"] is not None for job in seed_jobs) == 2, "final generation job inventory differs")
    timed_strongest_baseline_config = {
        "schema_version": "route_a_v3_route2_xeditflow_strongest_timing_config.v1",
        "method_id": "genetic",
        "strongest_generation_baseline_path": str(config["strongest_generation_baseline_path"]),
        "baseline_selection_input_path": str(config["baseline_selection_input_path"]),
        "source_manifest_path": str(
            setflow_runtimes[20260904]["source_eligibility_manifest"]
        ),
        "guiding_checkpoint_path": str(
            strongest_generation_baseline["guiding_checkpoint_path"]
        ),
        "critic_forward_budget_per_source": int(
            strongest_generation_baseline["critic_forward_budget_per_source"]
        ),
        "beam_width": 16,
        "genetic_population_size": 32,
        "oversample_factor": 8,
        "exhaustive_space_limit": 4096,
        "seed": 20260816,
        "physical_gpu_index": gpu,
        "device": f"cuda:{gpu}",
        "output_dir": str(timed_strongest_root),
        "timing_only_no_baseline_reselection": True,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    shared_search_score_path = (
        output_root
        / "seed20260904"
        / "closed_scores"
        / "strongest_matched_baseline"
        / "frozen_method_scores.private.jsonl"
    )
    closed_search_baseline_benchmark = {
        "schema_version": "route_a_v3_route2_xeditflow_closed_search_baseline_benchmark.v1",
        "status": "CONFIGURED_NOT_STARTED",
        "benchmark_reference_seed": 20260904,
        "common_frozen_score_provider": "MATCHED_SEARCH_GUIDING_CHECKPOINT",
        "common_score_table_method_id": "strongest_matched_baseline",
        "common_score_table_path": str(shared_search_score_path),
        "methods": {
            method: {
                "schema_version": "route_a_v3_route2_xeditflow_closed_score_config.v1",
                "method_id": method,
                "score_table_method_id": "strongest_matched_baseline",
                "base_flow_training_seed": 20260904,
                "pool_assignment": "DEVELOPMENT",
                "split": "VALIDATION",
                "analysis_unit": "SOURCE",
                "undefined_source_policy": "EXCLUDE_NOT_ZERO_FILL",
                "score_transform": "SOURCEWISE_EXP_SHIFTED_MAX",
                "measured_neighborhood_path": str(config["measured_neighborhood_path"]),
                "score_table_path": str(shared_search_score_path),
            }
            for method in FROZEN_SEARCH_BASELINES
        },
        "output_paths": {
            method: str(
                output_root
                / "benchmark_closed_search_baselines"
                / f"{method}.json"
            )
            for method in FROZEN_SEARCH_BASELINES
        },
        "shared_score_reason": (
            "ALL_FIVE_FROZEN_SEARCH_METHODS_USE_THE_SAME_MATCHED_GUIDING_"
            "CHECKPOINT_TO_RANK_TERMINAL_CANDIDATES;_OPEN_SUPPORT_DIFFERS_BY_SEARCH"
        ),
        "historical_open_support_results_reused_as_closed": False,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    final_comparison_path = output_root / "final_comparison" / "manifest.json"
    final_adjudication_path = output_root / "final_comparison" / "adjudication.json"
    return {
        "schema_version": "route_a_v3_route2_xeditflow_final_generation_manifest.v1",
        "status": "XEDITFLOW_V3_FINAL_GENERATION_CONFIGS_PREPARED_NOT_STARTED",
        "selected_kappa": kappa,
        "selected_temperature": temperature,
        "selected_beta_max": beta_max,
        "selected_setflow_arm": selected_arm,
        "required_base_flow_training_seeds": list(SEEDS),
        "seed_jobs": seed_jobs,
        "timed_strongest_baseline_config": timed_strongest_baseline_config,
        "closed_search_baseline_benchmark": closed_search_baseline_benchmark,
        "three_seed_finalization": {
            "guidance_screen_gate_path": str(config["guidance_screen_gate_path"]),
            "seed_manifest_row_paths": [
                str(output_root / f"seed{seed}" / "final_evidence" / "seed_manifest_row.json")
                for seed in SEEDS
            ],
            "final_comparison_manifest_path": str(final_comparison_path),
            "final_adjudication_path": str(final_adjudication_path),
            "replacement_evaluation_authorized_only_by_final_adjudication": True,
        },
        "additional_seed_authorized": False,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def write_manifest_v3(payload: Mapping[str, Any], output_dir: Path) -> None:
    _require(not output_dir.exists(), f"final generation config root exists: {output_dir}")
    output_dir.mkdir(parents=True)
    for job in payload["seed_jobs"]:
        seed = int(job["base_flow_training_seed"])
        for name in ("value_rollout", "value_target", "value_training"):
            config = job[f"{name}_config"]
            if config is not None:
                (output_dir / f"{name}_seed{seed}.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_dir / f"full_soft_value_smc_seed{seed}.json").write_text(json.dumps(job["full_smc_config"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for method, config in job["matched_control_configs"].items():
            (output_dir / f"{method}_seed{seed}.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for method, config in job["closed_trajectory_configs"].items():
            (output_dir / f"closed_trajectory_{method}_seed{seed}.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for method, config in job["closed_frozen_score_configs"].items():
            (output_dir / f"closed_frozen_score_{method}_seed{seed}.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for method, config in job["closed_score_metric_configs"].items():
            (output_dir / f"closed_score_metric_{method}_seed{seed}.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_dir / f"strongest_adapter_seed{seed}.json").write_text(json.dumps(job["strongest_adapter_job"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_dir / f"full_critic_score_seed{seed}.json").write_text(json.dumps(job["full_critic_score_config"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for method, config in job["open_metric_configs"].items():
            (output_dir / f"open_metric_{method}_seed{seed}.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_dir / f"independent_evaluator_seed{seed}.json").write_text(json.dumps(job["independent_evaluator_config"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_dir / f"independent_evaluator_comparison_seed{seed}.json").write_text(json.dumps(job["independent_evaluator_comparison_config"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_dir / f"equal_wall_time_seed{seed}.json").write_text(json.dumps(job["equal_wall_time_config"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_dir / f"final_seed_evidence_seed{seed}.json").write_text(json.dumps(job["final_seed_evidence_config"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "timed_strongest_baseline.json").write_text(
        json.dumps(payload["timed_strongest_baseline_config"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    benchmark = payload["closed_search_baseline_benchmark"]
    for method, config in benchmark["methods"].items():
        (output_dir / f"closed_search_baseline_{method}.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (output_dir / "closed_search_baseline_benchmark.json").write_text(
        json.dumps(benchmark, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "three_seed_finalization.json").write_text(
        json.dumps(payload["three_seed_finalization"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = _json(args.config)
    runtime_paths = {int(seed): Path(path) for seed, path in config["setflow_runtime_config_paths"].items()}
    payload = build_final_generation_configs_v3(
        config,
        critic_readiness=_json(Path(config["critic_readiness_path"])),
        setflow_confirmation=_json(Path(config["setflow_confirmation_path"])),
        critic_refit_manifest=_json(Path(config["critic_refit_manifest_path"])),
        guidance_screen_gate=_json(Path(config["guidance_screen_gate_path"])),
        strongest_generation_baseline=_json(Path(config["strongest_generation_baseline_path"])),
        setflow_runtimes={seed: _json(path) for seed, path in runtime_paths.items()},
    )
    write_manifest_v3(payload, args.output_dir)
    print(json.dumps({key: value for key, value in payload.items() if key != "seed_jobs"}, sort_keys=True))


if __name__ == "__main__":
    main()
