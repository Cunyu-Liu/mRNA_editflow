#!/usr/bin/env python3
"""Prepare the frozen V4 three-seed value and matched-generation job graph."""

from __future__ import annotations

import argparse
import json
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
from core.route2_xeditflow_value_training_v4 import (
    BASE_FLOW_SEEDS_V4,
    CRITIC_SEEDS_V4,
    validate_value_training_provenance_v4,
)
from core.route2_xeditsetflow_training_v4 import (
    EXPECTED_VALIDATION_SOURCE_RECORD_COUNT_V4,
)
from scripts.route_a_v3.build_route2_xeditflow_final_value_target_v4 import (
    validate_final_value_target_config_v4,
)
from scripts.route_a_v3.adapt_route2_xeditflow_strongest_baseline_v4 import (
    validate_strongest_adapter_config_v4,
)
from scripts.route_a_v3.evaluate_route2_xeditflow_closed_neighborhood_v4 import (
    validate_closed_run_config_v4,
)
from scripts.route_a_v3.evaluate_route2_xeditflow_open_generation_v4 import (
    validate_open_generation_config_v4,
)
from scripts.route_a_v3.generate_route2_xeditflow_value_rollouts_v4 import (
    _selected_checkpoint_pass_v4,
    validate_value_rollout_config_v4,
)
from scripts.route_a_v3.run_route2_xeditflow_matched_controls_v4 import (
    METHODS_V4,
    validate_matched_control_config_v4,
)
from scripts.route_a_v3.run_route2_xeditflow_smc_v4 import (
    terminal_critic_forward_reservation_v4,
    validate_smc_run_config_v4,
)
from scripts.route_a_v3.score_route2_xeditflow_candidates_v4 import (
    validate_candidate_score_config_v4,
)
from scripts.route_a_v3.score_route2_xeditflow_closed_controls_v4 import (
    CLOSED_CRITIC_METHODS_V4,
    validate_closed_control_score_config_v4,
)
from scripts.route_a_v3.score_route2_xeditflow_value_rollouts_v4 import (
    validate_value_critic_score_config_v4,
)
from scripts.route_a_v3.run_route2_xeditflow_strongest_timing_v4 import (
    validate_strongest_timing_config_v4,
)


FINAL_METHODS_V4 = (
    "full_soft_value_smc",
    "unguided_setflow",
    "first_order_guidance",
    "simple_rate_guidance",
    "generate_then_rerank",
)


class XEditFlowFinalGenerationPrepareV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowFinalGenerationPrepareV4Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _route2(value: Any, label: str) -> str:
    path = str(value)
    _require(
        path.startswith("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"),
        f"{label} left Route 2 /mnt",
    )
    return path


def _component(value: float) -> str:
    return str(float(value)).replace(".", "p")


def _value_id(kappa: float, temperature: float) -> str:
    return f"kappa_{_component(kappa)}_temperature_{_component(temperature)}"


def _gpu_map(
    values: Mapping[int, int], *, seeds: Sequence[int], label: str
) -> dict[int, int]:
    normalized = {int(seed): int(gpu) for seed, gpu in values.items()}
    _require(set(normalized) == set(seeds), f"V4 {label} seed inventory differs")
    _require(
        all(gpu in range(6) for gpu in normalized.values()),
        f"V4 {label} GPU is outside 0-5",
    )
    return normalized


def _validate_protocol_v4(protocol: Mapping[str, Any]) -> None:
    _require(
        protocol.get("schema_version")
        == "route_a_v3_route2_xeditflow_v4_guidance_protocol.v1"
        and protocol.get("status")
        == "FROZEN_PROSPECTIVE_BEFORE_V4_GUIDANCE_AUTHORIZATION_OR_OUTCOME_READ",
        "unexpected or unfrozen V4 guidance protocol",
    )
    _require(
        protocol.get("base_flow_screen_seed") == 20260912
        and tuple(protocol.get("base_flow_confirmation_seeds", ()))
        == BASE_FLOW_SEEDS_V4,
        "V4 final SetFlow seed protocol differs",
    )
    _require(
        protocol.get("guidance_grid")
        == {
            "kappa": [0.0, 0.5, 1.0],
            "temperature": [0.5, 1.0],
            "beta_max": [0.5, 1.0, 2.0],
            "combination_count": 18,
            "additional_combination_authorized": False,
        }
        and len(GUIDANCE_GRID_V4) == 18,
        "V4 final guidance grid differs",
    )
    _require(
        protocol.get("strongest_baseline_frozen_before_v4_candidate_generation")
        is True
        and protocol.get(
            "independent_evaluator_frozen_before_v4_candidate_generation"
        )
        is True
        and protocol.get("independent_evaluator_in_gradient") is False
        and int(protocol.get("evaluation_outcomes_used_to_select_independent_evaluator", -1))
        == 0,
        "V4 final evaluator or strongest baseline is not prospectively frozen",
    )
    protected = protocol.get("protected_outcomes", {})
    _require(
        protected.get("development_test_reopened") is False
        and protected.get("development_test_outcomes_accessed_after_atomic_test")
        is False
        and int(protected.get("new_final_evaluation_outcome_reads", -1)) == 0
        and protected.get("external_evaluation_authorized") is False,
        "V4 final generation protocol opened a protected outcome",
    )
    for field in (
        "critic_readiness_path",
        "setflow_confirmation_path",
        "critic_refit_manifest_path",
        "train_projection_path",
        "validation_projection_path",
        "source_token_cache_path",
        "source_level_data_audit_path",
        "source_eligibility_manifest",
        "measured_neighborhood_path",
        "mrnabert_model_path",
        "experiment_ledger_path",
        "guidance_screen_output_root",
        "runtime_config_root",
        "independent_evaluator_checkpoint_path",
        "independent_evaluator_adjudication_path",
        "strongest_generation_baseline_path",
        "baseline_selection_input_path",
        "strongest_closed_score_summary_path",
        "strongest_closed_score_table_path",
    ):
        _route2(protocol.get(field, ""), f"V4 protocol {field}")


def _validate_final_inputs_v4(
    protocol: Mapping[str, Any],
    critic_readiness: Mapping[str, Any],
    setflow_confirmation: Mapping[str, Any],
    critic_refit_manifest: Mapping[str, Any],
    source_data_audit: Mapping[str, Any],
    guidance_gate: Mapping[str, Any],
) -> tuple[float, float, float]:
    _validate_protocol_v4(protocol)
    _require(
        authorize_xeditflow_guidance_v4(
            critic_readiness, setflow_confirmation
        )["guidance_authorized"]
        is True,
        "V4 final generation remains blocked before joint readiness",
    )
    _require(
        setflow_confirmation.get("required_seeds") == list(BASE_FLOW_SEEDS_V4),
        "V4 final SetFlow confirmation seed inventory differs",
    )
    for seed in BASE_FLOW_SEEDS_V4:
        _selected_checkpoint_pass_v4(setflow_confirmation, seed=seed)
    _require(
        critic_refit_manifest.get("status")
        == "XEDITCRITIC_V4_ALL_DEVELOPMENT_REFIT_COMPLETE"
        and critic_refit_manifest.get("required_seeds") == list(CRITIC_SEEDS_V4)
        and int(critic_refit_manifest.get("completed_refit_count", -1)) == 3
        and int(critic_refit_manifest.get("refit_pass_count", -1)) == 8
        and critic_refit_manifest.get(
            "development_test_outcomes_accessed_during_refit"
        )
        is False
        and critic_refit_manifest.get("new_final_evaluation_outcomes_accessed")
        is False,
        "V4 final generation critic refit differs",
    )
    terminal_critic_forward_reservation_v4(critic_refit_manifest)
    _require(
        source_data_audit.get("status")
        == "XEDITSETFLOW_V4_SOURCE_LEVEL_DATA_AUDIT_PASS"
        and int(source_data_audit.get("train_source_count", 0)) >= 8
        and int(source_data_audit.get("validation_source_count", -1))
        == EXPECTED_VALIDATION_SOURCE_RECORD_COUNT_V4
        and int(source_data_audit.get("development_test_outcome_reads", -1)) == 0
        and int(source_data_audit.get("new_final_evaluation_outcome_reads", -1))
        == 0,
        "V4 final generation source-level data audit differs",
    )
    _require(
        guidance_gate.get("schema_version")
        == "route_a_v3_route2_xeditflow_v4_guidance_screen_gate.v1"
        and guidance_gate.get("status") == "XEDITFLOW_V4_GUIDANCE_SCREEN_FROZEN"
        and int(guidance_gate.get("base_flow_training_seed", -1)) == 20260912
        and int(guidance_gate.get("combination_count", -1)) == 18,
        "V4 final generation guidance screen is not frozen",
    )
    combination = (
        float(guidance_gate.get("selected_kappa", -1)),
        float(guidance_gate.get("selected_temperature", -1)),
        float(guidance_gate.get("selected_beta_max", -1)),
    )
    _require(
        combination in GUIDANCE_GRID_V4,
        "V4 final selected guidance combination left the frozen grid",
    )
    return combination


def build_final_generation_configs_v4(
    protocol: Mapping[str, Any],
    critic_readiness: Mapping[str, Any],
    setflow_confirmation: Mapping[str, Any],
    critic_refit_manifest: Mapping[str, Any],
    source_data_audit: Mapping[str, Any],
    guidance_gate: Mapping[str, Any],
    strongest_generation_baseline: Mapping[str, Any],
    baseline_selection_input: Mapping[str, Any],
    *,
    generation_gpus: Mapping[int, int],
    critic_gpus: Mapping[int, int],
    value_gpus: Mapping[int, int],
    strongest_timing_gpu: int,
    guidance_screen_gate_path: str,
    strongest_closed_score_table_path: str,
) -> dict[str, Any]:
    kappa, temperature, beta_max = _validate_final_inputs_v4(
        protocol,
        critic_readiness,
        setflow_confirmation,
        critic_refit_manifest,
        source_data_audit,
        guidance_gate,
    )
    generation_gpu = _gpu_map(
        generation_gpus, seeds=BASE_FLOW_SEEDS_V4, label="generation"
    )
    critic_gpu = _gpu_map(
        critic_gpus, seeds=BASE_FLOW_SEEDS_V4, label="critic scoring"
    )
    value_gpu = _gpu_map(
        value_gpus, seeds=(20260913, 20260914), label="final value training"
    )
    _require(
        int(strongest_timing_gpu) in range(6),
        "V4 strongest timing GPU is outside 0-5",
    )
    strongest_score_path = _route2(
        strongest_closed_score_table_path,
        "V4 pre-frozen strongest closed score table",
    )
    strongest_summary_path = _route2(
        protocol.get("strongest_closed_score_summary_path", ""),
        "V4 pre-frozen strongest closed score summary",
    )
    _require(
        strongest_score_path
        == _route2(
            protocol.get("strongest_closed_score_table_path", ""),
            "V4 protocol pre-frozen strongest closed score table",
        ),
        "V4 strongest closed score table differs from the frozen protocol",
    )
    frozen_screen_gate_path = _route2(
        guidance_screen_gate_path, "V4 frozen guidance screen gate"
    )
    _require(
        strongest_generation_baseline.get("status")
        == "DEVELOPMENT_STRONGEST_GENERATION_BASELINE_FROZEN_INDEPENDENT_EVALUATOR_ONLY"
        and strongest_generation_baseline.get("strongest_generation_baseline_id")
        == "genetic"
        and strongest_generation_baseline.get("evaluation_outcomes_accessed")
        is False
        and int(
            strongest_generation_baseline.get(
                "forward_equivalent_budget_per_source", -1
            )
        )
        == 320
        and baseline_selection_input.get("selection_pool")
        == "DEVELOPMENT_MEASURED_NEIGHBORHOOD"
        and baseline_selection_input.get("evaluation_release_state") == "CLOSED"
        and strongest_generation_baseline.get(
            "independent_evaluator_checkpoint_path"
        )
        == protocol.get("independent_evaluator_checkpoint_path"),
        "V4 final strongest baseline inputs differ",
    )
    runtime_paths = protocol.get("setflow_confirmation_runtime_config_paths")
    _require(
        isinstance(runtime_paths, Mapping)
        and set(runtime_paths) == {str(seed) for seed in BASE_FLOW_SEEDS_V4},
        "V4 final SetFlow runtime inventory differs",
    )
    critic_runtime_paths = protocol.get("critic_refit_runtime_config_paths")
    _require(
        isinstance(critic_runtime_paths, Mapping)
        and set(critic_runtime_paths) == {str(seed) for seed in CRITIC_SEEDS_V4},
        "V4 final Critic runtime inventory differs",
    )
    reservations = list(
        terminal_critic_forward_reservation_v4(critic_refit_manifest)
    )
    guiding_checkpoints = [
        _route2(row["checkpoint_path"], f"V4 refit checkpoint {row['seed']}")
        for row in sorted(
            critic_refit_manifest["checkpoints"], key=lambda item: int(item["seed"])
        )
    ]
    _require(
        len(guiding_checkpoints) == 3 and len(set(guiding_checkpoints)) == 3,
        "V4 final guiding checkpoint inventory differs",
    )
    output_root = (
        Path(str(protocol["guidance_screen_output_root"])).parent
        / "final_three_seed"
    )
    config_root = (
        Path(str(protocol["runtime_config_root"])).parent
        / "final_three_seed_v1"
    )
    train_source_count = int(source_data_audit["train_source_count"])
    terminal_rollout_count = train_source_count * 4 * 8
    value_id = _value_id(kappa, temperature)
    screen_value_checkpoint = (
        Path(str(protocol["guidance_screen_output_root"]))
        / "value_models"
        / value_id
        / "value_checkpoint.pt"
    )
    _require(
        guidance_gate.get("selected_value_checkpoint_path")
        == str(screen_value_checkpoint),
        "V4 final screen value checkpoint path differs from the frozen gate",
    )
    validate_value_training_provenance_v4(
        guidance_gate.get("selected_value_training_provenance", {}),
        base_flow_training_seed=20260912,
        value_checkpoint_path=str(screen_value_checkpoint),
    )
    screen_gate_path = Path(frozen_screen_gate_path)
    strongest_timing_root = (
        output_root / "benchmark_resources" / "strongest_matched_baseline_timing"
    )
    strongest_timing_candidate_path = (
        strongest_timing_root / "timed_genetic_candidates.private.jsonl"
    )
    strongest_timing_config = {
        "schema_version": "route_a_v3_route2_xeditflow_strongest_timing_config.v4",
        "method_id": "genetic",
        "strongest_generation_baseline_path": str(
            protocol["strongest_generation_baseline_path"]
        ),
        "baseline_selection_input_path": str(protocol["baseline_selection_input_path"]),
        "source_manifest_path": str(protocol["source_eligibility_manifest"]),
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
        "physical_gpu_index": int(strongest_timing_gpu),
        "device": f"cuda:{int(strongest_timing_gpu)}",
        "output_dir": str(strongest_timing_root),
        "timing_only_no_baseline_reselection": True,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    validate_strongest_timing_config_v4(
        strongest_timing_config,
        strongest_generation_baseline,
        baseline_selection_input,
    )
    seed_jobs: list[dict[str, Any]] = []
    for seed in BASE_FLOW_SEEDS_V4:
        seed_root = output_root / f"seed_{seed}"
        runtime_path = str(runtime_paths[str(seed)])
        rollout_config: dict[str, Any] | None = None
        score_config: dict[str, Any] | None = None
        target_config: dict[str, Any] | None = None
        training_config: dict[str, Any] | None = None
        if seed == 20260912:
            value_checkpoint = screen_value_checkpoint
        else:
            rollout_output = seed_root / "value_rollouts"
            score_output = seed_root / "value_critic_scores"
            target_output = seed_root / "value_target"
            model_output = seed_root / "value_model"
            value_checkpoint = model_output / "value_checkpoint.pt"
            rollout_config = {
                "schema_version": "route_a_v3_route2_xeditflow_value_rollout_config.v4",
                "critic_readiness_path": str(protocol["critic_readiness_path"]),
                "setflow_confirmation_path": str(protocol["setflow_confirmation_path"]),
                "setflow_runtime_config_path": runtime_path,
                "train_projection_path": str(protocol["train_projection_path"]),
                "source_token_cache_path": str(protocol["source_token_cache_path"]),
                "expected_train_source_count": train_source_count,
                "base_flow_training_seed": seed,
                "states_per_source": 4,
                "state_pass_index": 0,
                "rollouts_per_state_mode": 8,
                "sampling_state_batch_size": 32,
                "trajectory_forward_batch_size": 64,
                "fixed_seed_replay_check": True,
                "physical_gpu_index": value_gpu[seed],
                "device": f"cuda:{value_gpu[seed]}",
                "output_dir": str(rollout_output),
                "independent_evaluator_used": False,
                "development_test_outcomes_accessed_after_atomic_test": False,
                "new_final_evaluation_outcomes_accessed": False,
            }
            validate_value_rollout_config_v4(rollout_config)
            score_config = {
                "schema_version": "route_a_v3_route2_xeditflow_value_critic_score_config.v4",
                "critic_readiness_path": str(protocol["critic_readiness_path"]),
                "setflow_confirmation_path": str(protocol["setflow_confirmation_path"]),
                "critic_refit_manifest_path": str(protocol["critic_refit_manifest_path"]),
                "critic_refit_runtime_config_paths": dict(critic_runtime_paths),
                "critic_seeds": list(CRITIC_SEEDS_V4),
                "base_flow_training_seed": seed,
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
                "physical_gpu_index": critic_gpu[seed],
                "device": f"cuda:{critic_gpu[seed]}",
                "output_dir": str(score_output),
                "independent_evaluator_used": False,
                "development_test_outcomes_accessed_after_atomic_test": False,
                "new_final_evaluation_outcomes_accessed": False,
            }
            validate_value_critic_score_config_v4(score_config)
            target_config = {
                "schema_version": "route_a_v3_route2_xeditflow_final_value_target_config.v4",
                "base_flow_training_seed": seed,
                "kappa": kappa,
                "temperature": temperature,
                "train_state_path": str(rollout_output / "train_state_modes.jsonl"),
                "frozen_rollout_score_path": str(
                    score_output / "critic_scored_rollouts.private.jsonl"
                ),
                "rollout_summary_path": str(rollout_output / "run_summary.json"),
                "critic_score_summary_path": str(score_output / "run_summary.json"),
                "critic_readiness_path": str(protocol["critic_readiness_path"]),
                "setflow_confirmation_path": str(protocol["setflow_confirmation_path"]),
                "guidance_screen_gate_path": str(screen_gate_path),
                "output_dir": str(target_output),
                "independent_evaluator_used": False,
                "development_test_outcomes_accessed_after_atomic_test": False,
                "new_final_evaluation_outcomes_accessed": False,
            }
            validate_final_value_target_config_v4(target_config, guidance_gate)
            training_config = {
                "schema_version": "route_a_v3_route2_xeditflow_value_training_config.v4",
                "value_target_path": str(target_output / "value_targets.pt"),
                "critic_readiness_path": str(protocol["critic_readiness_path"]),
                "setflow_confirmation_path": str(protocol["setflow_confirmation_path"]),
                "source_token_cache_path": str(protocol["source_token_cache_path"]),
                "experiment_ledger_path": str(protocol["experiment_ledger_path"]),
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
                "checkpoint_selection": "FINAL_PASS_8_NO_EPOCH_RESELECTION",
                "physical_gpu_index": value_gpu[seed],
                "device": f"cuda:{value_gpu[seed]}",
                "output_dir": str(model_output),
                "independent_evaluator_used": False,
                "development_test_outcomes_accessed_after_atomic_test": False,
                "new_final_evaluation_outcomes_accessed": False,
            }

        generation_common = {
            "critic_readiness_path": str(protocol["critic_readiness_path"]),
            "critic_refit_manifest_path": str(protocol["critic_refit_manifest_path"]),
            "setflow_confirmation_path": str(protocol["setflow_confirmation_path"]),
            "setflow_runtime_config_path": runtime_path,
            "source_token_cache_path": str(protocol["source_token_cache_path"]),
            "source_eligibility_manifest": str(protocol["source_eligibility_manifest"]),
            "validation_projection_path": str(protocol["validation_projection_path"]),
            "expected_source_count": 891,
            "base_flow_training_seed": seed,
            "kappa": kappa,
            "temperature": temperature,
            "beta_max": beta_max,
            "particle_count": 32,
            "candidate_cap": 32,
            "ess_threshold": 16.0,
            "resampling": "STRATIFIED",
            "forward_equivalent_ceiling_per_source": 320,
            "terminal_critic_forwards_by_member": list(reservations),
            "maximum_sampling_rounds": 32,
            "action_space": "SUB+STOP",
            "replay_check": True,
            "decoder_seed_base": 20261001,
            "physical_gpu_index": generation_gpu[seed],
            "device": f"cuda:{generation_gpu[seed]}",
            "independent_evaluator_used": False,
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
        full_output = seed_root / "generation" / "full_soft_value_smc"
        full_smc_config = {
            **generation_common,
            "schema_version": "route_a_v3_route2_xeditflow_smc_run_config.v4",
            "method_id": "full_soft_value_smc",
            "value_checkpoint_path": str(value_checkpoint),
            "guidance_screen_gate_path": str(screen_gate_path),
            "output_dir": str(full_output),
        }
        validate_smc_run_config_v4(full_smc_config)
        controls: dict[str, dict[str, Any]] = {}
        for method in sorted(METHODS_V4):
            controls[method] = {
                **generation_common,
                "schema_version": "route_a_v3_route2_xeditflow_matched_control_run_config.v4",
                "method_id": method,
                "guidance_screen_gate_path": str(screen_gate_path),
                "critic_refit_runtime_config_paths": dict(critic_runtime_paths),
                "mrnabert_model_path": str(protocol["mrnabert_model_path"]),
                "bottom_six_maximum_sequences_per_batch": 8,
                "bottom_six_batch_token_budget": 4096,
                "attention_backend": "PYTORCH_SDPA_AUTO",
                "output_dir": str(seed_root / "generation" / method),
            }
            validate_matched_control_config_v4(controls[method])

        critic_score_configs: dict[str, dict[str, Any]] = {}
        open_metric_configs: dict[str, dict[str, Any]] = {}
        for method in FINAL_METHODS_V4:
            generation_output = seed_root / "generation" / method
            critic_output = generation_output / "terminal_critic"
            scorer = {
                "schema_version": "route_a_v3_route2_xeditflow_candidate_critic_score_config.v4",
                "critic_readiness_path": str(protocol["critic_readiness_path"]),
                "setflow_confirmation_path": str(protocol["setflow_confirmation_path"]),
                "critic_refit_manifest_path": str(protocol["critic_refit_manifest_path"]),
                "critic_refit_runtime_config_paths": dict(critic_runtime_paths),
                "critic_seeds": list(CRITIC_SEEDS_V4),
                "generation_summary_path": str(generation_output / "run_summary.json"),
                "candidate_path": str(
                    generation_output / "generated_candidates.private.jsonl"
                ),
                "generation_compute_path": str(
                    generation_output / "matched_compute.jsonl"
                ),
                "source_eligibility_manifest": str(protocol["source_eligibility_manifest"]),
                "validation_projection_path": str(protocol["validation_projection_path"]),
                "mrnabert_model_path": str(protocol["mrnabert_model_path"]),
                "expected_source_count": 891,
                "candidate_cap_per_source": 32,
                "base_flow_training_seed": seed,
                "kappa": kappa,
                "temperature": temperature,
                "beta_max": beta_max,
                "method_id": method,
                "guidance_screen_gate_path": str(screen_gate_path),
                "study_policy": "UNKNOWN_STUDY_SCALE_FIXED_1",
                "prediction_scale": "TASK_ROBUST_STANDARDIZED_EFFECT",
                "bottom_six_maximum_sequences_per_batch": 8,
                "bottom_six_batch_token_budget": 4096,
                "attention_backend": "PYTORCH_SDPA_AUTO",
                "physical_gpu_index": critic_gpu[seed],
                "device": f"cuda:{critic_gpu[seed]}",
                "output_dir": str(critic_output),
                "critic_self_score_used_for_generation_or_selection": (
                    method == "generate_then_rerank"
                ),
                "independent_evaluator_used": False,
                "development_test_outcomes_accessed_after_atomic_test": False,
                "new_final_evaluation_outcomes_accessed": False,
            }
            validate_candidate_score_config_v4(scorer)
            critic_score_configs[method] = scorer
            metric = {
                "schema_version": "route_a_v3_route2_xeditflow_open_generation_config.v4",
                "pool_assignment": "DEVELOPMENT",
                "candidate_support_mode": "OPEN_GENERATED_SUPPORT",
                "undefined_outcome_policy": "UNKNOWN_NOT_ZERO",
                "source_eligibility_manifest": str(protocol["source_eligibility_manifest"]),
                "candidate_path": str(
                    critic_output / "critic_scored_candidates.private.jsonl"
                ),
                "measured_neighborhood_path": str(protocol["measured_neighborhood_path"]),
                "measured_top_k": 10,
                "base_flow_training_seed": seed,
                "kappa": kappa,
                "temperature": temperature,
                "beta_max": beta_max,
                "method_id": method,
                "critic_self_score_used_for_ranking": (
                    method == "generate_then_rerank"
                ),
                "development_test_outcomes_accessed_after_atomic_test": False,
                "new_final_evaluation_outcomes_accessed": False,
                "output_path": str(seed_root / "metrics" / f"open_{method}.json"),
            }
            validate_open_generation_config_v4(metric)
            open_metric_configs[method] = metric

        closed_common = {
            "critic_readiness_path": str(protocol["critic_readiness_path"]),
            "setflow_confirmation_path": str(protocol["setflow_confirmation_path"]),
            "setflow_runtime_config_path": runtime_path,
            "source_token_cache_path": str(protocol["source_token_cache_path"]),
            "source_eligibility_manifest": str(protocol["source_eligibility_manifest"]),
            "validation_projection_path": str(protocol["validation_projection_path"]),
            "measured_neighborhood_path": str(protocol["measured_neighborhood_path"]),
            "expected_source_count": 891,
            "base_flow_training_seed": seed,
            "kappa": kappa,
            "temperature": temperature,
            "beta_max": beta_max,
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
            "physical_gpu_index": generation_gpu[seed],
            "device": f"cuda:{generation_gpu[seed]}",
            "guidance_screen_gate_path": str(screen_gate_path),
            "independent_evaluator_used": False,
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
        closed_exact_configs = {
            "full_soft_value_smc": {
                **closed_common,
                "schema_version": "route_a_v3_route2_xeditflow_closed_neighborhood_config.v4",
                "method_id": "full_soft_value_smc",
                "potential_kind": "SOFT_VALUE",
                "value_checkpoint_path": str(value_checkpoint),
                "output_dir": str(seed_root / "closed" / "full_soft_value_smc"),
            },
            "unguided_setflow": {
                **closed_common,
                "schema_version": "route_a_v3_route2_xeditflow_closed_neighborhood_config.v4",
                "method_id": "unguided_setflow",
                "potential_kind": "ZERO",
                "output_dir": str(seed_root / "closed" / "unguided_setflow"),
            },
        }
        for closed_config in closed_exact_configs.values():
            validate_closed_run_config_v4(closed_config)

        closed_score_configs: dict[str, dict[str, Any]] = {}
        closed_metric_configs: dict[str, dict[str, Any]] = {}
        for method in sorted(CLOSED_CRITIC_METHODS_V4):
            score_output = seed_root / "closed_scores" / method
            closed_scorer = {
                "schema_version": "route_a_v3_route2_xeditflow_closed_control_score_config.v4",
                "method_id": method,
                "base_flow_training_seed": seed,
                "kappa": kappa,
                "temperature": temperature,
                "beta_max": beta_max,
                "critic_seeds": list(CRITIC_SEEDS_V4),
                "critic_readiness_path": str(protocol["critic_readiness_path"]),
                "setflow_confirmation_path": str(protocol["setflow_confirmation_path"]),
                "critic_refit_manifest_path": str(protocol["critic_refit_manifest_path"]),
                "critic_refit_runtime_config_paths": dict(critic_runtime_paths),
                "mrnabert_model_path": str(protocol["mrnabert_model_path"]),
                "source_eligibility_manifest": str(protocol["source_eligibility_manifest"]),
                "validation_projection_path": str(protocol["validation_projection_path"]),
                "measured_neighborhood_path": str(protocol["measured_neighborhood_path"]),
                "guidance_screen_gate_path": str(screen_gate_path),
                "pool_assignment": "DEVELOPMENT",
                "split": "VALIDATION",
                "expected_source_count": 891,
                "study_policy": "UNKNOWN_STUDY_SCALE_FIXED_1",
                "bottom_six_maximum_sequences_per_batch": 8,
                "bottom_six_batch_token_budget": 4096,
                "attention_backend": "PYTORCH_SDPA_AUTO",
                "physical_gpu_index": critic_gpu[seed],
                "device": f"cuda:{critic_gpu[seed]}",
                "output_dir": str(score_output),
                "independent_evaluator_used": False,
                "measured_outcome_used_to_construct_score": False,
                "development_test_outcomes_accessed_after_atomic_test": False,
                "new_final_evaluation_outcomes_accessed": False,
            }
            validate_closed_control_score_config_v4(closed_scorer)
            closed_score_configs[method] = closed_scorer
            closed_metric_configs[method] = {
                "schema_version": "route_a_v3_route2_xeditflow_closed_score_config.v4",
                "method_id": method,
                "base_flow_training_seed": seed,
                "kappa": kappa,
                "temperature": temperature,
                "beta_max": beta_max,
                "pool_assignment": "DEVELOPMENT",
                "split": "VALIDATION",
                "analysis_unit": "SOURCE",
                "undefined_source_policy": "EXCLUDE_NOT_ZERO_FILL",
                "score_transform": "SOURCEWISE_EXP_SHIFTED_MAX",
                "measured_neighborhood_path": str(protocol["measured_neighborhood_path"]),
                "score_table_path": str(
                    score_output / "frozen_method_scores.private.jsonl"
                ),
                "score_summary_path": str(score_output / "run_summary.json"),
                "development_test_outcomes_accessed_after_atomic_test": False,
                "new_final_evaluation_outcomes_accessed": False,
                "output_path": str(seed_root / "metrics" / f"closed_{method}.json"),
            }
        closed_metric_configs["strongest_matched_baseline"] = {
            "schema_version": "route_a_v3_route2_xeditflow_closed_score_config.v4",
            "method_id": "strongest_matched_baseline",
            "score_table_method_id": "strongest_matched_baseline",
            "base_flow_training_seed": seed,
            "pool_assignment": "DEVELOPMENT",
            "split": "VALIDATION",
            "analysis_unit": "SOURCE",
            "undefined_source_policy": "EXCLUDE_NOT_ZERO_FILL",
            "score_transform": "SOURCEWISE_EXP_SHIFTED_MAX",
            "measured_neighborhood_path": str(protocol["measured_neighborhood_path"]),
            "score_table_path": strongest_score_path,
            "score_summary_path": strongest_summary_path,
            "strongest_baseline_frozen_before_v4_candidate_generation": True,
            "baseline_reselected_for_v4": False,
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcomes_accessed": False,
            "output_path": str(
                seed_root / "metrics" / "closed_strongest_matched_baseline.json"
            ),
        }

        evaluator_output = (
            full_output / "independent_evaluator_scored_candidates.private.jsonl"
        )
        evaluator_config = {
            "schema_version": "route_a_v3_route2_xeditflow_independent_evaluator_job.v4",
            "method_id": "full_soft_value_smc",
            "evaluator_checkpoint_path": str(
                protocol["independent_evaluator_checkpoint_path"]
            ),
            "evaluator_adjudication_path": str(
                protocol["independent_evaluator_adjudication_path"]
            ),
            "guiding_checkpoint_paths": list(guiding_checkpoints),
            "source_manifest_path": str(protocol["source_eligibility_manifest"]),
            "candidate_path": str(
                full_output
                / "terminal_critic"
                / "critic_scored_candidates.private.jsonl"
            ),
            "output_path": str(evaluator_output),
            "expected_source_count": 891,
            "evaluator_frozen_before_candidate_generation": True,
            "independent_evaluator_in_gradient": False,
            "evaluation_outcomes_used_to_select_evaluator": 0,
            "physical_gpu_index": critic_gpu[seed],
            "device": f"cuda:{critic_gpu[seed]}",
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcome_reads": 0,
        }
        evaluator_comparison_config = {
            "schema_version": "route_a_v3_route2_xeditflow_independent_evaluator_comparison_config.v4",
            "method_id": "full_soft_value_smc",
            "base_flow_training_seed": seed,
            "combination": [kappa, temperature, beta_max],
            "strongest_baseline_path": str(protocol["strongest_generation_baseline_path"]),
            "baseline_selection_input_path": str(protocol["baseline_selection_input_path"]),
            "source_eligibility_manifest": str(protocol["source_eligibility_manifest"]),
            "guided_scored_candidate_path": str(evaluator_output),
            "guided_scoring_summary_path": str(Path(str(evaluator_output) + ".summary.json")),
            "bootstrap_iterations": 10_000,
            "bootstrap_seed": 20261001 + seed,
            "independent_evaluator_in_gradient": False,
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcome_reads": 0,
            "output_path": str(
                seed_root / "metrics" / "independent_evaluator_full_vs_strongest.json"
            ),
        }
        strongest_adapter_root = (
            seed_root / "generation" / "strongest_matched_baseline"
        )
        strongest_adapter_config = {
            "schema_version": "route_a_v3_route2_xeditflow_strongest_baseline_adapter_config.v4",
            "strongest_generation_baseline_path": str(
                protocol["strongest_generation_baseline_path"]
            ),
            "baseline_selection_input_path": str(
                protocol["baseline_selection_input_path"]
            ),
            "base_flow_training_seed": seed,
            "output_dir": str(strongest_adapter_root),
            "strongest_baseline_reselected_for_v4": False,
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
        validate_strongest_adapter_config_v4(strongest_adapter_config)
        closed_summary_paths = {
            "full_soft_value_smc": str(
                seed_root / "closed" / "full_soft_value_smc" / "run_summary.json"
            ),
            "unguided_setflow": str(
                seed_root / "closed" / "unguided_setflow" / "run_summary.json"
            ),
            **{
                method: str(closed_metric_configs[method]["output_path"])
                for method in (
                    "first_order_guidance",
                    "simple_rate_guidance",
                    "generate_then_rerank",
                    "strongest_matched_baseline",
                )
            },
        }
        equal_wall_output = seed_root / "metrics" / "equal_wall_time_sensitivity.json"
        equal_wall_config = {
            "schema_version": "route_a_v3_route2_xeditflow_equal_wall_time_config.v4",
            "base_flow_training_seed": seed,
            "source_manifest_path": str(protocol["source_eligibility_manifest"]),
            "methods": {
                method: {
                    "timing_path": (
                        str(strongest_timing_candidate_path)
                        if method == "strongest_matched_baseline"
                        else str(
                            seed_root
                            / "generation"
                            / method
                            / "terminal_critic"
                            / "matched_compute.scored.jsonl"
                        )
                    ),
                    "timing_format": (
                        "SEARCH_CANDIDATE_JSONL"
                        if method == "strongest_matched_baseline"
                        else "MATCHED_COMPUTE_SCORED_JSONL"
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
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcomes_accessed": False,
            "output_path": str(equal_wall_output),
        }
        method_evidence_paths = {
            method: {
                "closed_summary_path": closed_summary_paths[method],
                "open_summary_path": (
                    str(strongest_adapter_root / "open.json")
                    if method == "strongest_matched_baseline"
                    else str(open_metric_configs[method]["output_path"])
                ),
                "generation_summary_path": (
                    str(strongest_adapter_root / "generation.json")
                    if method == "strongest_matched_baseline"
                    else str(seed_root / "generation" / method / "run_summary.json")
                ),
                **(
                    {}
                    if method == "strongest_matched_baseline"
                    else {
                        "terminal_critic_summary_path": str(
                            seed_root
                            / "generation"
                            / method
                            / "terminal_critic"
                            / "run_summary.json"
                        )
                    }
                ),
            }
            for method in (
                "full_soft_value_smc",
                "unguided_setflow",
                "first_order_guidance",
                "simple_rate_guidance",
                "generate_then_rerank",
                "strongest_matched_baseline",
            )
        }
        final_evidence_output = seed_root / "final_evidence"
        final_evidence_config = {
            "schema_version": "route_a_v3_route2_xeditflow_final_seed_evidence_config.v4",
            "base_flow_training_seed": seed,
            "selected_combination": [kappa, temperature, beta_max],
            "value_checkpoint_path": str(value_checkpoint),
            "methods": method_evidence_paths,
            "equal_wall_time_sensitivity_path": str(equal_wall_output),
            "full_independent_evaluator_path": str(
                evaluator_comparison_config["output_path"]
            ),
            "full_candidate_path": str(
                full_output
                / "terminal_critic"
                / "critic_scored_candidates.private.jsonl"
            ),
            "unguided_candidate_path": str(
                seed_root
                / "generation"
                / "unguided_setflow"
                / "terminal_critic"
                / "critic_scored_candidates.private.jsonl"
            ),
            "bootstrap_iterations": 10_000,
            "bootstrap_seed": 20261001 + seed + 100_000,
            "output_dir": str(final_evidence_output),
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
        seed_jobs.append(
            {
                "base_flow_training_seed": seed,
                "setflow_runtime_config_path": runtime_path,
                "value_checkpoint_path": str(value_checkpoint),
                "screen_value_checkpoint_reused": seed == 20260912,
                "value_rollout_config": rollout_config,
                "value_critic_score_config": score_config,
                "value_target_config": target_config,
                "value_training_config": training_config,
                "full_smc_config": full_smc_config,
                "matched_control_configs": controls,
                "terminal_critic_score_configs": critic_score_configs,
                "open_metric_configs": open_metric_configs,
                "closed_exact_configs": closed_exact_configs,
                "closed_control_score_configs": closed_score_configs,
                "closed_metric_configs": closed_metric_configs,
                "independent_evaluator_config": evaluator_config,
                "independent_evaluator_comparison_config": (
                    evaluator_comparison_config
                ),
                "strongest_adapter_config": strongest_adapter_config,
                "equal_wall_time_config": equal_wall_config,
                "equal_wall_time_output_path": str(equal_wall_output),
                "final_seed_evidence_config": final_evidence_config,
                "final_seed_evidence_output_dir": str(final_evidence_output),
            }
        )
    _require(
        len(seed_jobs) == 3
        and sum(job["screen_value_checkpoint_reused"] for job in seed_jobs) == 1
        and sum(job["value_training_config"] is not None for job in seed_jobs) == 2,
        "V4 final seed job inventory differs",
    )
    final_comparison_manifest_path = output_root / "final_comparison_manifest.json"
    final_comparison_compose_config = {
        "schema_version": "route_a_v3_route2_xeditflow_final_comparison_compose_config.v4",
        "guidance_screen_gate_path": str(screen_gate_path),
        "expected_value_checkpoint_paths": {
            str(job["base_flow_training_seed"]): job["value_checkpoint_path"]
            for job in seed_jobs
        },
        "seed_manifest_row_paths": {
            str(seed): str(
                output_root
                / f"seed_{seed}"
                / "final_evidence"
                / "seed_manifest_row.json"
            )
            for seed in BASE_FLOW_SEEDS_V4
        },
        "output_path": str(final_comparison_manifest_path),
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    return {
        "schema_version": "route_a_v3_route2_xeditflow_final_generation_manifest.v4",
        "status": "XEDITFLOW_V4_FINAL_GENERATION_CONFIGS_PREPARED_NOT_STARTED",
        "required_base_flow_training_seeds": list(BASE_FLOW_SEEDS_V4),
        "selected_combination": [kappa, temperature, beta_max],
        "screen_value_seed": 20260912,
        "screen_value_checkpoint_path": str(screen_value_checkpoint),
        "value_checkpoint_paths": {
            str(job["base_flow_training_seed"]): job["value_checkpoint_path"]
            for job in seed_jobs
        },
        "non_screen_value_training_seeds": [20260913, 20260914],
        "decoder_seed_base": 20261001,
        "same_decoder_seed_streams_across_methods_and_seeds": True,
        "candidate_cap_per_source": 32,
        "forward_equivalent_ceiling_per_source": 320,
        "terminal_critic_forwards_by_member": reservations,
        "strongest_closed_score_table_path": strongest_score_path,
        "strongest_closed_score_summary_path": strongest_summary_path,
        "strongest_baseline_reselected_for_v4": False,
        "strongest_timing_config": strongest_timing_config,
        "strongest_timing_candidate_path": str(strongest_timing_candidate_path),
        "final_comparison_compose_config": final_comparison_compose_config,
        "final_comparison_manifest_path": str(final_comparison_manifest_path),
        "final_adjudication_output_path": str(
            output_root / "final_adjudication.json"
        ),
        "output_root": str(output_root),
        "runtime_config_root": str(config_root),
        "seed_jobs": seed_jobs,
        "final_three_seed_gate_may_run_only_after_all_seed_jobs_terminal": True,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def write_final_generation_configs_v4(
    payload: Mapping[str, Any], output_dir: Path
) -> None:
    _require(
        output_dir == Path(str(payload["runtime_config_root"])),
        "V4 final runtime config output differs from manifest",
    )
    _require(not output_dir.exists(), f"V4 final config root exists: {output_dir}")
    output_dir.mkdir(parents=True)
    written: list[str] = []
    for job in payload["seed_jobs"]:
        seed = int(job["base_flow_training_seed"])
        fixed = {
            "full_smc": job["full_smc_config"],
            "independent_evaluator": job["independent_evaluator_config"],
            "independent_evaluator_comparison": job[
                "independent_evaluator_comparison_config"
            ],
            "strongest_adapter": job["strongest_adapter_config"],
            "equal_wall_time": job["equal_wall_time_config"],
            "final_seed_evidence": job["final_seed_evidence_config"],
        }
        for name, config in fixed.items():
            path = output_dir / f"seed_{seed}_{name}.json"
            path.write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            written.append(str(path))
        for name in (
            "value_rollout_config",
            "value_critic_score_config",
            "value_target_config",
            "value_training_config",
        ):
            config = job[name]
            if config is None:
                continue
            path = output_dir / f"seed_{seed}_{name.removesuffix('_config')}.json"
            path.write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            written.append(str(path))
        collections = {
            "control": job["matched_control_configs"],
            "terminal_critic": job["terminal_critic_score_configs"],
            "open_metric": job["open_metric_configs"],
            "closed_exact": job["closed_exact_configs"],
            "closed_control_score": job["closed_control_score_configs"],
            "closed_metric": job["closed_metric_configs"],
        }
        for prefix, configs in collections.items():
            for method, config in sorted(configs.items()):
                path = output_dir / f"seed_{seed}_{prefix}_{method}.json"
                path.write_text(
                    json.dumps(config, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                written.append(str(path))
    timing_path = output_dir / "strongest_matched_baseline_timing.json"
    timing_path.write_text(
        json.dumps(payload["strongest_timing_config"], indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    written.append(str(timing_path))
    compose_path = output_dir / "final_comparison_compose.json"
    compose_path.write_text(
        json.dumps(
            payload["final_comparison_compose_config"], indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(str(compose_path))
    manifest = dict(payload)
    manifest["written_runtime_config_paths"] = written
    manifest["written_runtime_config_count"] = len(written)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _three_gpu_map(values: Sequence[int]) -> dict[int, int]:
    _require(len(values) == 3, "V4 final GPU list must contain three entries")
    return dict(zip(BASE_FLOW_SEEDS_V4, (int(value) for value in values), strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--critic-readiness", required=True, type=Path)
    parser.add_argument("--setflow-confirmation", required=True, type=Path)
    parser.add_argument("--critic-refit-manifest", required=True, type=Path)
    parser.add_argument("--source-data-audit", required=True, type=Path)
    parser.add_argument("--guidance-screen-gate", required=True, type=Path)
    parser.add_argument("--strongest-closed-score-table", required=True)
    parser.add_argument("--generation-gpus", nargs=3, required=True, type=int)
    parser.add_argument("--critic-gpus", nargs=3, required=True, type=int)
    parser.add_argument("--value-gpus", nargs=2, required=True, type=int)
    parser.add_argument("--strongest-timing-gpu", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    protocol = _json(arguments.protocol)
    strongest = _json(Path(str(protocol["strongest_generation_baseline_path"])))
    baseline_selection = _json(Path(str(protocol["baseline_selection_input_path"])))
    payload = build_final_generation_configs_v4(
        protocol,
        _json(arguments.critic_readiness),
        _json(arguments.setflow_confirmation),
        _json(arguments.critic_refit_manifest),
        _json(arguments.source_data_audit),
        _json(arguments.guidance_screen_gate),
        strongest,
        baseline_selection,
        generation_gpus=_three_gpu_map(arguments.generation_gpus),
        critic_gpus=_three_gpu_map(arguments.critic_gpus),
        value_gpus={
            20260913: int(arguments.value_gpus[0]),
            20260914: int(arguments.value_gpus[1]),
        },
        strongest_timing_gpu=int(arguments.strongest_timing_gpu),
        guidance_screen_gate_path=str(arguments.guidance_screen_gate),
        strongest_closed_score_table_path=arguments.strongest_closed_score_table,
    )
    write_final_generation_configs_v4(payload, arguments.output_dir)
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
