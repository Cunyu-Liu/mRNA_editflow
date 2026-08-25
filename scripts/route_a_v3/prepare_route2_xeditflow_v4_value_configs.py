#!/usr/bin/env python3
"""Prepare the exact V4 rollout, scoring, target, and six value jobs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_xeditflow_gate_v4 import (
    GUIDANCE_GRID_V4,
    authorize_xeditflow_guidance_v4,
)
from core.route2_xeditflow_value_training_v4 import CRITIC_SEEDS_V4
from core.route2_xeditsetflow_training_v4 import (
    EXPECTED_VALIDATION_SOURCE_RECORD_COUNT_V4,
)
from scripts.route_a_v3.build_route2_xeditflow_value_targets_v4 import (
    VALUE_TARGET_GRID_V4,
)
from scripts.route_a_v3.generate_route2_xeditflow_value_rollouts_v4 import (
    _selected_checkpoint_pass_v4,
    validate_value_rollout_config_v4,
)
from scripts.route_a_v3.score_route2_xeditflow_value_rollouts_v4 import (
    validate_value_critic_score_config_v4,
)
from scripts.route_a_v3.score_route2_xeditflow_candidates_v4 import (
    validate_candidate_score_config_v4,
)
from scripts.route_a_v3.evaluate_route2_xeditflow_closed_neighborhood_v4 import (
    validate_closed_run_config_v4,
)
from scripts.route_a_v3.evaluate_route2_xeditflow_open_generation_v4 import (
    validate_open_generation_config_v4,
)
from scripts.route_a_v3.run_route2_xeditflow_smc_v4 import (
    terminal_critic_forward_reservation_v4,
    validate_smc_run_config_v4,
)


class XEditFlowValueConfigPrepareV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowValueConfigPrepareV4Error(message)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON root is not an object: {path}")
    return payload


def _component(value: float) -> str:
    return str(float(value)).replace(".", "p")


def _value_id(kappa: float, temperature: float) -> str:
    return (
        f"kappa_{_component(kappa)}_temperature_{_component(temperature)}"
    )


def _require_route2_path(value: Any, label: str) -> str:
    path = str(value)
    _require(
        path.startswith("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"),
        f"{label} left Route 2 /mnt",
    )
    return path


def build_value_configs_v4(
    protocol: Mapping[str, Any],
    authorization: Mapping[str, Any],
    critic_readiness: Mapping[str, Any],
    setflow_confirmation: Mapping[str, Any],
    critic_refit_manifest: Mapping[str, Any],
    source_data_audit: Mapping[str, Any],
    *,
    rollout_gpu: int,
    critic_gpu: int,
    value_gpus: Sequence[int],
    guidance_gpus: Sequence[int],
) -> dict[str, Any]:
    _require(
        protocol.get("schema_version")
        == "route_a_v3_route2_xeditflow_v4_guidance_protocol.v1"
        and protocol.get("status")
        == "FROZEN_PROSPECTIVE_BEFORE_V4_GUIDANCE_AUTHORIZATION_OR_OUTCOME_READ",
        "unexpected or unfrozen V4 guidance protocol",
    )
    frozen_grid = protocol.get("guidance_grid", {})
    _require(
        frozen_grid
        == {
            "kappa": [0.0, 0.5, 1.0],
            "temperature": [0.5, 1.0],
            "beta_max": [0.5, 1.0, 2.0],
            "combination_count": 18,
            "additional_combination_authorized": False,
        }
        and len(GUIDANCE_GRID_V4) == 18,
        "V4 guidance grid changed",
    )
    expected_authorization = authorize_xeditflow_guidance_v4(
        critic_readiness, setflow_confirmation
    )
    _require(
        dict(authorization) == expected_authorization
        and authorization.get("guidance_authorized") is True,
        "V4 value configs remain blocked before exact joint authorization",
    )
    _require(
        critic_refit_manifest.get("status")
        == "XEDITCRITIC_V4_ALL_DEVELOPMENT_REFIT_COMPLETE"
        and critic_refit_manifest.get("required_seeds")
        == list(CRITIC_SEEDS_V4)
        and int(critic_refit_manifest.get("completed_refit_count", -1)) == 3
        and int(critic_refit_manifest.get("refit_pass_count", -1)) == 8
        and critic_refit_manifest.get("loso_authorized") is True
        and critic_refit_manifest.get(
            "development_test_outcomes_accessed_during_refit"
        )
        is False
        and critic_refit_manifest.get("new_final_evaluation_outcomes_accessed")
        is False,
        "V4 value configs require the exact three frozen critic refits",
    )
    checkpoints = critic_refit_manifest.get("checkpoints", ())
    _require(
        sorted(int(row["seed"]) for row in checkpoints)
        == list(CRITIC_SEEDS_V4),
        "V4 critic refit checkpoint seeds differ",
    )
    checkpoint_by_seed = {
        int(row["seed"]): _require_route2_path(
            row["checkpoint_path"], f"V4 critic refit checkpoint {row['seed']}"
        )
        for row in checkpoints
    }
    guiding_checkpoint_paths = [
        checkpoint_by_seed[seed] for seed in CRITIC_SEEDS_V4
    ]
    _require(
        len(set(guiding_checkpoint_paths)) == 3,
        "V4 guiding critic checkpoint paths are not distinct",
    )
    critic_forwards_by_member = list(
        terminal_critic_forward_reservation_v4(critic_refit_manifest)
    )
    _selected_checkpoint_pass_v4(setflow_confirmation, seed=20260912)
    _require(
        source_data_audit.get("status")
        == "XEDITSETFLOW_V4_SOURCE_LEVEL_DATA_AUDIT_PASS"
        and int(source_data_audit.get("train_source_count", 0)) >= 8
        and int(source_data_audit.get("validation_source_count", -1))
        == EXPECTED_VALIDATION_SOURCE_RECORD_COUNT_V4
        and int(source_data_audit.get("development_test_outcome_reads", -1))
        == 0
        and int(source_data_audit.get("new_final_evaluation_outcome_reads", -1))
        == 0,
        "V4 value configs require the frozen source-level data audit",
    )
    _require(
        rollout_gpu in range(6) and critic_gpu in range(6),
        "V4 rollout or critic GPU is outside 0-5",
    )
    value_gpu_tuple = tuple(int(value) for value in value_gpus)
    _require(
        len(value_gpu_tuple) == 6
        and set(value_gpu_tuple) == set(range(6)),
        "V4 six value jobs must be assigned once each to GPUs 0-5",
    )
    guidance_gpu_tuple = tuple(int(value) for value in guidance_gpus)
    _require(
        len(guidance_gpu_tuple) == len(GUIDANCE_GRID_V4)
        and all(value in range(6) for value in guidance_gpu_tuple)
        and all(guidance_gpu_tuple.count(value) == 3 for value in range(6)),
        "V4 18 guidance combinations must assign exactly three chains to each GPU 0-5",
    )
    runtime = protocol.get("value_runtime")
    _require(
        runtime
        == {
            "sampling_state_batch_size": 32,
            "trajectory_forward_batch_size": 64,
            "critic_candidate_batch_size": 128,
            "bottom_six_maximum_sequences_per_batch": 8,
            "bottom_six_batch_token_budget": 4096,
            "attention_backend": "PYTORCH_SDPA_AUTO",
            "fixed_seed_replay_check": True,
            "value_training_gpu_assignment": (
                "ROUND_ROBIN_ACROSS_DECLARED_GPU_0_TO_5"
            ),
        },
        "V4 value runtime policy changed",
    )
    evaluator_checkpoint = _require_route2_path(
        protocol["independent_evaluator_checkpoint_path"],
        "V4 independent evaluator checkpoint",
    )
    evaluator_adjudication = _require_route2_path(
        protocol["independent_evaluator_adjudication_path"],
        "V4 independent evaluator adjudication",
    )
    strongest_baseline = _require_route2_path(
        protocol["strongest_generation_baseline_path"],
        "V4 strongest generation baseline",
    )
    baseline_selection = _require_route2_path(
        protocol["baseline_selection_input_path"],
        "V4 baseline selection input",
    )
    _require(
        protocol.get("independent_evaluator_frozen_before_v4_candidate_generation")
        is True
        and protocol.get("independent_evaluator_in_gradient") is False
        and int(
            protocol.get(
                "evaluation_outcomes_used_to_select_independent_evaluator", -1
            )
        )
        == 0
        and int(protocol.get("independent_evaluator_bootstrap_iterations", -1))
        == 10_000
        and protocol.get("strongest_baseline_frozen_before_v4_candidate_generation")
        is True
        and evaluator_checkpoint not in guiding_checkpoint_paths,
        "V4 independent evaluator or strongest-baseline freeze differs",
    )
    output_root = Path(
        _require_route2_path(
            protocol["guidance_screen_output_root"],
            "V4 guidance screen output",
        )
    )
    train_source_count = int(source_data_audit["train_source_count"])
    state_mode_count = train_source_count * 4
    terminal_rollout_count = state_mode_count * 8
    rollout_output = output_root / "value_rollouts"
    score_output = output_root / "value_critic_scores"
    target_output = output_root / "value_targets"
    value_output = output_root / "value_models"
    rollout_config = {
        "schema_version": (
            "route_a_v3_route2_xeditflow_value_rollout_config.v4"
        ),
        "critic_readiness_path": str(protocol["critic_readiness_path"]),
        "setflow_confirmation_path": str(
            protocol["setflow_confirmation_path"]
        ),
        "setflow_runtime_config_path": str(
            protocol["setflow_confirmation_runtime_config_paths"]["20260912"]
        ),
        "train_projection_path": str(protocol["train_projection_path"]),
        "source_token_cache_path": str(protocol["source_token_cache_path"]),
        "expected_train_source_count": train_source_count,
        "base_flow_training_seed": 20260912,
        "states_per_source": 4,
        "state_pass_index": 0,
        "rollouts_per_state_mode": 8,
        "sampling_state_batch_size": 32,
        "trajectory_forward_batch_size": 64,
        "fixed_seed_replay_check": True,
        "physical_gpu_index": rollout_gpu,
        "device": f"cuda:{rollout_gpu}",
        "output_dir": str(rollout_output),
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    validate_value_rollout_config_v4(rollout_config)
    score_config = {
        "schema_version": (
            "route_a_v3_route2_xeditflow_value_critic_score_config.v4"
        ),
        "critic_readiness_path": str(protocol["critic_readiness_path"]),
        "setflow_confirmation_path": str(
            protocol["setflow_confirmation_path"]
        ),
        "critic_refit_manifest_path": str(
            protocol["critic_refit_manifest_path"]
        ),
        "critic_refit_runtime_config_paths": dict(
            protocol["critic_refit_runtime_config_paths"]
        ),
        "critic_seeds": list(CRITIC_SEEDS_V4),
        "base_flow_training_seed": 20260912,
        "rollout_summary_path": str(rollout_output / "run_summary.json"),
        "terminal_rollout_path": str(
            rollout_output / "terminal_rollouts.private.jsonl"
        ),
        "expected_terminal_rollout_count": terminal_rollout_count,
        "mrnabert_model_path": str(protocol["mrnabert_model_path"]),
        "candidate_batch_size": 128,
        "bottom_six_maximum_sequences_per_batch": 8,
        "bottom_six_batch_token_budget": 4096,
        "attention_backend": "PYTORCH_SDPA_AUTO",
        "study_policy": "UNKNOWN_STUDY_SCALE_FIXED_1",
        "prediction_scale": "TASK_ROBUST_STANDARDIZED_EFFECT",
        "trajectory_mode_used_as_critic_input": False,
        "physical_gpu_index": critic_gpu,
        "device": f"cuda:{critic_gpu}",
        "output_dir": str(score_output),
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    validate_value_critic_score_config_v4(score_config)
    target_config = {
        "schema_version": (
            "route_a_v3_route2_xeditflow_value_target_grid_build_config.v4"
        ),
        "stage": "GUIDANCE_SCREEN",
        "base_flow_training_seed": 20260912,
        "grid": [list(row) for row in VALUE_TARGET_GRID_V4],
        "train_state_path": str(rollout_output / "train_state_modes.jsonl"),
        "frozen_rollout_score_path": str(
            score_output / "critic_scored_rollouts.private.jsonl"
        ),
        "critic_readiness_path": str(protocol["critic_readiness_path"]),
        "setflow_confirmation_path": str(
            protocol["setflow_confirmation_path"]
        ),
        "rollout_summary_path": str(rollout_output / "run_summary.json"),
        "critic_score_summary_path": str(score_output / "run_summary.json"),
        "output_root": str(target_output),
    }
    value_jobs: list[dict[str, Any]] = []
    for job_index, (kappa, temperature) in enumerate(VALUE_TARGET_GRID_V4):
        value_id = _value_id(kappa, temperature)
        gpu = value_gpu_tuple[job_index]
        training_config = {
            "schema_version": (
                "route_a_v3_route2_xeditflow_value_training_config.v4"
            ),
            "value_target_path": str(
                target_output / value_id / "value_targets.pt"
            ),
            "critic_readiness_path": str(protocol["critic_readiness_path"]),
            "setflow_confirmation_path": str(
                protocol["setflow_confirmation_path"]
            ),
            "source_token_cache_path": str(protocol["source_token_cache_path"]),
            "experiment_ledger_path": str(protocol["experiment_ledger_path"]),
            "base_flow_training_seed": 20260912,
            "kappa": kappa,
            "temperature": temperature,
            "passes": 8,
            "batch_size": 32,
            "precision": "BF16",
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "gradient_clip_norm": 1.0,
            "dropout": 0.10,
            "checkpoint_selection": "FINAL_PASS_8_NO_EPOCH_RESELECTION",
            "physical_gpu_index": gpu,
            "device": f"cuda:{gpu}",
            "output_dir": str(value_output / value_id),
            "independent_evaluator_used": False,
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
        value_jobs.append(
            {
                "value_id": value_id,
                "kappa": kappa,
                "temperature": temperature,
                "physical_gpu_index": gpu,
                "config": training_config,
            }
        )
    _require(len(value_jobs) == 6, "V4 value training job count differs")
    smc_policy = protocol.get("smc")
    _require(
        smc_policy
        == {
            "particles": 32,
            "candidate_cap_per_source": 32,
            "ess_resample_threshold": 16,
            "resampling": "STRATIFIED",
            "edit_budgets": [1, 3, 5],
            "decoder_seed_base": 20261001,
            "maximum_sampling_rounds": 32,
            "fixed_seed_replay_check": True,
            "same_decoder_seed_streams_across_all_18_combinations": True,
            "forward_equivalent_ceiling_per_source": 320,
            "trunk_mode_value_and_each_critic_member_forwards_separately_charged": True,
        },
        "V4 SMC runtime policy changed",
    )
    guidance_jobs: list[dict[str, Any]] = []
    for guidance_index, (kappa, temperature, beta_max) in enumerate(
        GUIDANCE_GRID_V4
    ):
        guidance_gpu = guidance_gpu_tuple[guidance_index]
        value_id = _value_id(kappa, temperature)
        combination_id = (
            f"{value_id}_beta_{_component(beta_max)}"
        )
        smc_output = output_root / "guidance_screen" / combination_id / "open_smc"
        smc_config = {
            "schema_version": "route_a_v3_route2_xeditflow_smc_run_config.v4",
            "critic_readiness_path": str(protocol["critic_readiness_path"]),
            "critic_refit_manifest_path": str(protocol["critic_refit_manifest_path"]),
            "setflow_confirmation_path": str(protocol["setflow_confirmation_path"]),
            "setflow_runtime_config_path": str(
                protocol["setflow_confirmation_runtime_config_paths"]["20260912"]
            ),
            "value_checkpoint_path": str(
                value_output / value_id / "value_checkpoint.pt"
            ),
            "source_token_cache_path": str(protocol["source_token_cache_path"]),
            "source_eligibility_manifest": str(protocol["source_eligibility_manifest"]),
            "validation_projection_path": str(protocol["validation_projection_path"]),
            "measured_neighborhood_path": str(protocol["measured_neighborhood_path"]),
            "expected_source_count": 891,
            "base_flow_training_seed": 20260912,
            "kappa": kappa,
            "temperature": temperature,
            "beta_max": beta_max,
            "particle_count": 32,
            "candidate_cap": 32,
            "ess_threshold": 16.0,
            "resampling": "STRATIFIED",
            "forward_equivalent_ceiling_per_source": 320,
            "terminal_critic_forwards_by_member": critic_forwards_by_member,
            "maximum_sampling_rounds": 32,
            "action_space": "SUB+STOP",
            "replay_check": True,
            "decoder_seed_base": 20261001,
            "physical_gpu_index": guidance_gpu,
            "device": f"cuda:{guidance_gpu}",
            "method_id": f"xeditflow_v4_guidance_screen_{combination_id}",
            "output_dir": str(smc_output),
            "independent_evaluator_used": False,
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
        validate_smc_run_config_v4(smc_config)
        critic_output = smc_output / "terminal_critic"
        critic_config = {
            "schema_version": (
                "route_a_v3_route2_xeditflow_candidate_critic_score_config.v4"
            ),
            "critic_readiness_path": str(protocol["critic_readiness_path"]),
            "setflow_confirmation_path": str(protocol["setflow_confirmation_path"]),
            "critic_refit_manifest_path": str(protocol["critic_refit_manifest_path"]),
            "critic_refit_runtime_config_paths": dict(
                protocol["critic_refit_runtime_config_paths"]
            ),
            "critic_seeds": list(CRITIC_SEEDS_V4),
            "generation_summary_path": str(smc_output / "run_summary.json"),
            "candidate_path": str(
                smc_output / "generated_candidates.private.jsonl"
            ),
            "generation_compute_path": str(smc_output / "matched_compute.jsonl"),
            "source_eligibility_manifest": str(protocol["source_eligibility_manifest"]),
            "validation_projection_path": str(protocol["validation_projection_path"]),
            "mrnabert_model_path": str(protocol["mrnabert_model_path"]),
            "expected_source_count": 891,
            "candidate_cap_per_source": 32,
            "base_flow_training_seed": 20260912,
            "kappa": kappa,
            "temperature": temperature,
            "beta_max": beta_max,
            "method_id": smc_config["method_id"],
            "study_policy": "UNKNOWN_STUDY_SCALE_FIXED_1",
            "prediction_scale": "TASK_ROBUST_STANDARDIZED_EFFECT",
            "bottom_six_maximum_sequences_per_batch": 8,
            "bottom_six_batch_token_budget": 4096,
            "attention_backend": "PYTORCH_SDPA_AUTO",
            "physical_gpu_index": guidance_gpu,
            "device": f"cuda:{guidance_gpu}",
            "output_dir": str(critic_output),
            "critic_self_score_used_for_generation_or_selection": False,
            "independent_evaluator_used": False,
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
        validate_candidate_score_config_v4(critic_config)
        closed_output = output_root / "guidance_screen" / combination_id / "closed"
        closed_config = {
            "schema_version": (
                "route_a_v3_route2_xeditflow_closed_neighborhood_config.v4"
            ),
            "critic_readiness_path": str(protocol["critic_readiness_path"]),
            "setflow_confirmation_path": str(protocol["setflow_confirmation_path"]),
            "setflow_runtime_config_path": str(
                protocol["setflow_confirmation_runtime_config_paths"]["20260912"]
            ),
            "value_checkpoint_path": str(
                value_output / value_id / "value_checkpoint.pt"
            ),
            "source_token_cache_path": str(protocol["source_token_cache_path"]),
            "source_eligibility_manifest": str(protocol["source_eligibility_manifest"]),
            "validation_projection_path": str(protocol["validation_projection_path"]),
            "measured_neighborhood_path": str(protocol["measured_neighborhood_path"]),
            "expected_source_count": 891,
            "base_flow_training_seed": 20260912,
            "kappa": kappa,
            "temperature": temperature,
            "beta_max": beta_max,
            "method_id": smc_config["method_id"],
            "potential_kind": "SOFT_VALUE",
            "latent_mode_policy": (
                "ROOT_PRIOR_WEIGHTED_SUM_OF_EIGHT_FIXED_MODE_TERMINAL_PROBABILITIES"
            ),
            "root_prior_forward_batch_size": 32,
            "value_child_forward_batch_size": 32,
            "pool_assignment": "DEVELOPMENT",
            "split": "VALIDATION",
            "maximum_enumerated_edits": 5,
            "maximum_permutation_paths": 120,
            "enumeration": "ALL_EDIT_PERMUTATIONS_EXACT_SUM",
            "analysis_unit": "SOURCE",
            "undefined_source_policy": "EXCLUDE_NOT_ZERO_FILL",
            "physical_gpu_index": guidance_gpu,
            "device": f"cuda:{guidance_gpu}",
            "output_dir": str(closed_output),
            "independent_evaluator_used": False,
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
        validate_closed_run_config_v4(closed_config)
        open_metric_output = smc_output / "generation_metrics.json"
        open_metric_config = {
            "schema_version": (
                "route_a_v3_route2_xeditflow_open_generation_config.v4"
            ),
            "pool_assignment": "DEVELOPMENT",
            "candidate_support_mode": "OPEN_GENERATED_SUPPORT",
            "undefined_outcome_policy": "UNKNOWN_NOT_ZERO",
            "source_eligibility_manifest": str(protocol["source_eligibility_manifest"]),
            "candidate_path": str(
                critic_output / "critic_scored_candidates.private.jsonl"
            ),
            "measured_neighborhood_path": str(protocol["measured_neighborhood_path"]),
            "measured_top_k": 10,
            "base_flow_training_seed": 20260912,
            "kappa": kappa,
            "temperature": temperature,
            "beta_max": beta_max,
            "method_id": smc_config["method_id"],
            "critic_self_score_used_for_ranking": False,
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcomes_accessed": False,
            "output_path": str(open_metric_output),
        }
        validate_open_generation_config_v4(open_metric_config)
        evaluator_output = (
            smc_output / "independent_evaluator_scored_candidates.private.jsonl"
        )
        evaluator_summary = Path(str(evaluator_output) + ".summary.json")
        evaluator_metric_output = (
            smc_output / "independent_evaluator_metrics.json"
        )
        evaluator_config = {
            "schema_version": (
                "route_a_v3_route2_xeditflow_independent_evaluator_job.v4"
            ),
            "method_id": smc_config["method_id"],
            "evaluator_checkpoint_path": evaluator_checkpoint,
            "evaluator_adjudication_path": evaluator_adjudication,
            "guiding_checkpoint_paths": list(guiding_checkpoint_paths),
            "source_manifest_path": str(protocol["source_eligibility_manifest"]),
            "candidate_path": str(
                critic_output / "critic_scored_candidates.private.jsonl"
            ),
            "output_path": str(evaluator_output),
            "expected_source_count": 891,
            "evaluator_frozen_before_candidate_generation": True,
            "independent_evaluator_in_gradient": False,
            "evaluation_outcomes_used_to_select_evaluator": 0,
            "physical_gpu_index": guidance_gpu,
            "device": f"cuda:{guidance_gpu}",
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcome_reads": 0,
        }
        evaluator_comparison_config = {
            "schema_version": (
                "route_a_v3_route2_xeditflow_independent_evaluator_comparison_config.v4"
            ),
            "method_id": smc_config["method_id"],
            "base_flow_training_seed": 20260912,
            "combination": [kappa, temperature, beta_max],
            "strongest_baseline_path": strongest_baseline,
            "baseline_selection_input_path": baseline_selection,
            "source_eligibility_manifest": str(
                protocol["source_eligibility_manifest"]
            ),
            "guided_scored_candidate_path": str(evaluator_output),
            "guided_scoring_summary_path": str(evaluator_summary),
            "bootstrap_iterations": 10_000,
            "bootstrap_seed": 20261001 + len(guidance_jobs),
            "independent_evaluator_in_gradient": False,
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcome_reads": 0,
            "output_path": str(evaluator_metric_output),
        }
        guidance_jobs.append(
            {
                "combination_id": combination_id,
                "combination": [kappa, temperature, beta_max],
                "value_id": value_id,
                "physical_gpu_index": guidance_gpu,
                "smc_config": smc_config,
                "critic_ensemble_config": critic_config,
                "closed_config": closed_config,
                "open_metric_config": open_metric_config,
                "independent_evaluator_config": evaluator_config,
                "independent_evaluator_comparison_config": (
                    evaluator_comparison_config
                ),
                "independent_evaluator_scored_candidate_path": str(
                    evaluator_output
                ),
                "independent_evaluator_metric_path": str(
                    evaluator_metric_output
                ),
            }
        )
    _require(len(guidance_jobs) == 18, "V4 guidance SMC job count differs")
    return {
        "schema_version": (
            "route_a_v3_route2_xeditflow_v4_value_config_manifest.v1"
        ),
        "status": "XEDITFLOW_V4_VALUE_CONFIGS_PREPARED_NOT_STARTED",
        "base_flow_training_seed": 20260912,
        "state_mode_count": state_mode_count,
        "terminal_rollout_count": terminal_rollout_count,
        "rollout_job_count": 1,
        "critic_score_job_count": 1,
        "value_target_package_count": 6,
        "value_training_job_count": 6,
        "later_guidance_combination_count": 18,
        "guidance_gpu_assignment": list(guidance_gpu_tuple),
        "rollout_config": rollout_config,
        "critic_score_config": score_config,
        "target_grid_config": target_config,
        "value_jobs": value_jobs,
        "guidance_jobs": guidance_jobs,
        "beta_max_used_in_value_target_or_training": False,
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def write_value_configs_v4(payload: Mapping[str, Any], output_dir: Path) -> None:
    staging_dir = output_dir.with_name(output_dir.name + ".staging")
    _require(
        not output_dir.exists() and not staging_dir.exists(),
        f"V4 value runtime config root exists: {output_dir}",
    )
    staging_dir.mkdir(parents=True)
    fixed = {
        "value_rollout.json": payload["rollout_config"],
        "value_critic_score.json": payload["critic_score_config"],
        "value_target_grid.json": payload["target_grid_config"],
    }
    paths: dict[str, str] = {}
    for name, config in fixed.items():
        path = staging_dir / name
        path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        paths[name] = str(output_dir / name)
    value_paths = []
    for job in payload["value_jobs"]:
        path = staging_dir / f"value_train_{job['value_id']}.json"
        path.write_text(
            json.dumps(job["config"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        value_paths.append(str(output_dir / path.name))
    guidance_paths = []
    critic_paths = []
    closed_paths = []
    open_metric_paths = []
    evaluator_paths = []
    evaluator_comparison_paths = []
    result_paths = []
    for job in payload["guidance_jobs"]:
        path = staging_dir / f"smc_{job['combination_id']}.json"
        path.write_text(
            json.dumps(job["smc_config"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        guidance_paths.append(str(output_dir / path.name))
        critic_path = staging_dir / f"critic_{job['combination_id']}.json"
        critic_path.write_text(
            json.dumps(
                job["critic_ensemble_config"], indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        critic_paths.append(str(output_dir / critic_path.name))
        closed_path = staging_dir / f"closed_{job['combination_id']}.json"
        closed_path.write_text(
            json.dumps(job["closed_config"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        closed_paths.append(str(output_dir / closed_path.name))
        open_path = staging_dir / f"open_metric_{job['combination_id']}.json"
        open_path.write_text(
            json.dumps(job["open_metric_config"], indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        open_metric_paths.append(str(output_dir / open_path.name))
        evaluator_path = (
            staging_dir / f"independent_evaluator_{job['combination_id']}.json"
        )
        evaluator_path.write_text(
            json.dumps(
                job["independent_evaluator_config"], indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        evaluator_paths.append(str(output_dir / evaluator_path.name))
        comparison_path = (
            staging_dir
            / f"independent_evaluator_comparison_{job['combination_id']}.json"
        )
        comparison_path.write_text(
            json.dumps(
                job["independent_evaluator_comparison_config"],
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        evaluator_comparison_paths.append(str(output_dir / comparison_path.name))
        result_paths.append(
            {
                "combination_id": job["combination_id"],
                "combination": list(job["combination"]),
                "smc_summary_path": str(
                    Path(job["smc_config"]["output_dir"]) / "run_summary.json"
                ),
                "critic_summary_path": str(
                    Path(job["critic_ensemble_config"]["output_dir"])
                    / "run_summary.json"
                ),
                "matched_compute_path": str(
                    Path(job["critic_ensemble_config"]["output_dir"])
                    / "matched_compute.scored.jsonl"
                ),
                "closed_summary_path": str(
                    Path(job["closed_config"]["output_dir"])
                    / "run_summary.json"
                ),
                "open_metric_path": str(
                    job["open_metric_config"]["output_path"]
                ),
                "independent_evaluator_summary_path": str(
                    job["independent_evaluator_scored_candidate_path"]
                )
                + ".summary.json",
                "independent_evaluator_metric_path": str(
                    job["independent_evaluator_metric_path"]
                ),
            }
        )
    manifest = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "rollout_config",
            "critic_score_config",
            "target_grid_config",
            "value_jobs",
            "guidance_jobs",
        }
    }
    manifest["config_paths"] = paths
    manifest["value_training_config_paths"] = value_paths
    manifest["guidance_smc_config_paths"] = guidance_paths
    manifest["guidance_critic_config_paths"] = critic_paths
    manifest["guidance_closed_config_paths"] = closed_paths
    manifest["guidance_open_metric_config_paths"] = open_metric_paths
    manifest["guidance_independent_evaluator_config_paths"] = evaluator_paths
    manifest["guidance_independent_evaluator_comparison_config_paths"] = (
        evaluator_comparison_paths
    )
    manifest["guidance_result_paths"] = result_paths
    (staging_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(staging_dir, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--rollout-gpu", required=True, type=int)
    parser.add_argument("--critic-gpu", required=True, type=int)
    parser.add_argument(
        "--value-gpus", required=True, type=int, nargs=6, metavar="GPU"
    )
    parser.add_argument(
        "--guidance-gpus", required=True, type=int, nargs=18, metavar="GPU"
    )
    arguments = parser.parse_args()
    protocol = _read(arguments.protocol)
    payload = build_value_configs_v4(
        protocol,
        _read(Path(protocol["authorization_output"])),
        _read(Path(protocol["critic_readiness_path"])),
        _read(Path(protocol["setflow_confirmation_path"])),
        _read(Path(protocol["critic_refit_manifest_path"])),
        _read(Path(protocol["source_level_data_audit_path"])),
        rollout_gpu=arguments.rollout_gpu,
        critic_gpu=arguments.critic_gpu,
        value_gpus=arguments.value_gpus,
        guidance_gpus=arguments.guidance_gpus,
    )
    output_dir = Path(protocol["runtime_config_root"])
    write_value_configs_v4(payload, output_dir)
    print(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key
                not in {
                    "rollout_config",
                    "critic_score_config",
                    "target_grid_config",
                    "value_jobs",
                    "guidance_jobs",
                }
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
