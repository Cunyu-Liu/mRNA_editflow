#!/usr/bin/env python3
"""Launch Critic V4 confirmation only from the fixed V4.0.3 cross-root PASS."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


WORKTREE = Path(__file__).resolve().parents[2]
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from scripts.route_a_v3.prepare_route2_xeditcritic_v4_confirmation_configs import (
    build_critic_confirmation_configs_v4,
)


PYTHON = Path("/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10")
ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
# Historical screen provenance only. New controls and confirmation train with
# the exact current runner HEAD passed to this launcher.
REPAIRED_SCREEN_PROVENANCE_GIT_HEAD = (
    "f34ab7d865bb2477bfe24c1d0a7c9f5301a24cea"
)
HISTORICAL_C0_GIT_HEAD = "93703adec7a4c76b4466d3aaae8684620bee985a"
CONTROL_RETRY_IDENTITY = "v403_control_recovery_retry1"
TRAINING_SEMANTICS_PREVIOUS_AUDITED_BASELINE_HEAD = (
    "708e2843b4b4a6f36796db5c21b6e99469138f3b"
)
TRAINING_SEMANTICS_PREVIOUS_SUCCESSOR_BASELINE_HEAD = (
    "eba5b17431cb8e19202e5ea788fd419338da2d66"
)
TRAINING_SEMANTICS_AUDITED_SUCCESSOR_BASELINE_HEAD = (
    "f1a2328db57e1bd20fcc5cd5e6a23abcf4c62b66"
)
TRAINING_SEMANTICS_BASELINE_AUDIT = (
    WORKTREE
    / "audits/route_a_v3_route2_xeditcritic_v403_confirmation_"
    "training_semantics_reaudit_f1a2328db57e1bd20fcc5cd5e6a23abcf4c62b66.json"
)
CONTROLS_OOM_RETRY_TECHNICAL_BASELINE_HEAD = (
    "793eedfb4b84e8c0dbd5a30bdf79c8923ddf8110"
)
CONTROLS_OOM_RETRY_TECHNICAL_BASELINE_AUDIT = (
    WORKTREE
    / "audits/route_a_v3_route2_xeditcritic_v403_controls_oom_retry_"
    "training_semantics_793eedfb4b84e8c0dbd5a30bdf79c8923ddf8110.json"
)
CONTROLS_OOM_RETRY_TECHNICAL_BASELINE_AUDIT_SCHEMA = (
    "route_a_v3_route2_xeditcritic_v403_controls_oom_retry_"
    "training_semantics.v1"
)
CONTROLS_OOM_RETRY_TECHNICAL_BASELINE_AUDIT_STATUS = (
    "XEDITCRITIC_V403_CONTROLS_OOM_RETRY_TRAINING_SEMANTICS_PASS"
)
CONTROLS_OOM_RETRY_CHANGED_TRAINING_SEMANTIC_PATHS = (
    "scripts/route_a_v3/launch_route2_xeditcritic_v403_controls_after_full.py",
    "scripts/route_a_v3/run_route2_xeditcritic_v403_control_recovery_scheduler.py",
)
CONTROLS_OOM_RETRY_PATH_CLASSIFICATION = {
    CONTROLS_OOM_RETRY_CHANGED_TRAINING_SEMANTIC_PATHS[0]: (
        "IMMUTABLE_OOM_LINEAGE_NEW_INDEPENDENT_RETRY_NO_VRAM_GATE"
    ),
    CONTROLS_OOM_RETRY_CHANGED_TRAINING_SEMANTIC_PATHS[1]: (
        "FIXED_TWO_WAVE_ALLOCATOR_AND_PACKAGE_FIRST_FAILURE"
    ),
}
CONTROLS_OOM_RETRY_UNCHANGED_FLAGS = {
    "critic_model_implementation_changed": False,
    "critic_trainer_code_changed": False,
    "critic_model_architecture_changed": False,
    "critic_model_forward_changed": False,
    "critic_loss_changed": False,
    "critic_scientific_config_changed": False,
    "critic_training_data_changed": False,
    "critic_sampler_changed": False,
    "critic_physical_or_effective_batch_changed": False,
    "critic_seed_cohort_changed": False,
    "critic_optimizer_objective_changed": False,
    "critic_optimizer_update_budget_changed": False,
    "scientific_thresholds_changed": False,
    "control_physical_gpu_mapping_changed": False,
    "free_memory_gate_added": False,
}
PREVIOUS_CODE_BASELINE_HEAD = "26fdbcb38090cf98e68425bebabd084a374447c4"
CORRECTED_SCREEN_CONFIRMATION_PROVENANCE_BASELINE_HEAD = (
    "19bc3ed4dd3ee5647e3d3304c10dc9914f885e68"
)
CORRECTED_SCREEN_CONFIRMATION_PROVENANCE_AUDIT = (
    WORKTREE
    / "audits/route_a_v3_route2_xeditsetflow_s1_corrected_screen_"
    "confirmation_provenance_19bc3ed4dd3ee5647e3d3304c10dc9914f885e68.json"
)
CORRECTED_SCREEN_CONFIRMATION_PROVENANCE_AUDIT_SCHEMA = (
    "route_a_v3_route2_xeditsetflow_s1_corrected_screen_"
    "confirmation_provenance.v1"
)
CORRECTED_SCREEN_CONFIRMATION_PROVENANCE_AUDIT_STATUS = (
    "XEDITSETFLOW_V4_S1_CORRECTED_SCREEN_CONFIRMATION_PROVENANCE_PASS"
)
INVALIDATED_SCREEN_RUNNER_GIT_HEAD = (
    "930fccf468c14378b3dd2fd2caf3aaa3cc2eb3c8"
)
CORRECTED_SCREEN_RUNNER_GIT_HEAD = (
    "ebf99ebf8a253ad27e311e555121d328df8fae10"
)
CORRECTED_SCREEN_PROVENANCE_CHANGED_TRAINING_SEMANTIC_PATHS = (
    "core/route2_xeditsetflow_confirmation_s1.py",
    "core/route2_xeditsetflow_gate_s1.py",
)
CORRECTED_SCREEN_PROVENANCE_PATH_CLASSIFICATION = {
    path: "SETFLOW_PROVENANCE_ONLY_CRITIC_OBJECTIVE_NEUTRAL"
    for path in CORRECTED_SCREEN_PROVENANCE_CHANGED_TRAINING_SEMANTIC_PATHS
}
CORRECTED_SCREEN_PROVENANCE_UNCHANGED_FLAGS = {
    "setflow_model_architecture_forward_loss_changed": False,
    "setflow_trainer_parameter_update_semantics_changed": False,
    "objective_identity_changed": False,
    "objective_weight_changed": False,
    "screen_seed_changed": False,
    "confirmation_seed_cohort_changed": False,
    "pass_count_changed": False,
    "physical_or_effective_batch_changed": False,
    "checkpoint_passes_changed": False,
    "paired_bootstrap_changed": False,
    "scientific_thresholds_changed": False,
    "gpu_scope_or_fixed_mapping_changed": False,
    "free_or_estimated_memory_gate_added": False,
    "free_or_estimated_memory_sorting_added": False,
    "package_failure_policy_changed": False,
    "protected_outcome_policy_changed": False,
}
TRAINING_SEMANTICS_REAUDIT_CHANGED_PATHS = (
    "core/route2_experiment_ledger.py",
    "core/route2_xeditcritic_gate_v4.py",
    "core/route2_xeditflow_value_training_v4.py",
    "core/route2_xeditsetflow_confirmation_s1.py",
    "core/route2_xeditsetflow_gate_s1.py",
    "scripts/route_a_v3/adjudicate_route2_xeditcritic_v4_confirmation.py",
    "scripts/route_a_v3/adjudicate_route2_xeditcritic_v4_posttest.py",
    "scripts/route_a_v3/adjudicate_route2_xeditcritic_v4_screen.py",
    "scripts/route_a_v3/launch_route2_xedit_v4_postscreen_after_screen_terminal.py",
    "scripts/route_a_v3/launch_route2_xeditcritic_v403_controls_after_full.py",
    "scripts/route_a_v3/launch_route2_xeditcritic_v4_loso_after_refits.py",
    "scripts/route_a_v3/launch_route2_xeditcritic_v4_refit_after_atomic_test.py",
    "scripts/route_a_v3/prepare_route2_xeditcritic_v4_confirmation_configs.py",
    "scripts/route_a_v3/prepare_route2_xeditcritic_v4_posttest_configs.py",
    "scripts/route_a_v3/run_route2_xedit_v4_confirmation_training_scheduler.py",
    "scripts/route_a_v3/run_route2_xeditcritic_v403_control_recovery_scheduler.py",
    "scripts/route_a_v3/run_route2_xeditcritic_v4_loso_scheduler.py",
    "scripts/route_a_v3/run_route2_xeditcritic_v4_refit_scheduler.py",
    "scripts/route_a_v3/train_route2_xeditcritic_v4.py",
    "scripts/route_a_v3/transition_adjudicate_route2_xeditcritic_v403_cross_root_screen.py",
)
TRAINING_SEMANTICS_REAUDIT_PATH_CLASSIFICATION = {
    "core/route2_experiment_ledger.py": (
        "SHARED_TERMINAL_TRAINING_LEDGER_V2_PROVENANCE_SCHEMA"
    ),
    "core/route2_xeditcritic_gate_v4.py": (
        "CRITIC_V2_TERMINAL_PROVENANCE_AND_EXACT_HEAD_GATE_HARDENING"
    ),
    "core/route2_xeditflow_value_training_v4.py": (
        "GUIDANCE_VALUE_CHECKPOINT_PROVENANCE_ONLY_CRITIC_OBJECTIVE_NEUTRAL"
    ),
    "core/route2_xeditsetflow_confirmation_s1.py": (
        "SETFLOW_S1_CONFIRMATION_ONLY_CRITIC_OBJECTIVE_NEUTRAL"
    ),
    "core/route2_xeditsetflow_gate_s1.py": (
        "SETFLOW_S1_MATCHED_INITIALIZATION_GATE_ONLY_CRITIC_OBJECTIVE_NEUTRAL"
    ),
    "scripts/route_a_v3/adjudicate_route2_xeditcritic_v4_confirmation.py": (
        "CRITIC_CONFIRMATION_CONFIG_HEAD_AND_TERMINAL_PROVENANCE_BINDING"
    ),
    "scripts/route_a_v3/adjudicate_route2_xeditcritic_v4_posttest.py": (
        "CRITIC_REFIT_LOSO_EXPLICIT_MANIFEST_AND_V2_TERMINAL_BINDING"
    ),
    "scripts/route_a_v3/adjudicate_route2_xeditcritic_v4_screen.py": (
        "CRITIC_SCREEN_EXACT_RUNNER_HEAD_AND_TECHNICAL_FAILURE_SEPARATION"
    ),
    "scripts/route_a_v3/launch_route2_xedit_v4_postscreen_after_screen_terminal.py": (
        "CRITIC_SCREEN_RUNNER_HEAD_PROPAGATION_ONLY"
    ),
    "scripts/route_a_v3/launch_route2_xeditcritic_v403_controls_after_full.py": (
        "CRITIC_CONTROL_EXACT_HEAD_NO_VRAM_GATE_AND_SCHEDULER_LAUNCH_FAILURE_EVIDENCE"
    ),
    "scripts/route_a_v3/launch_route2_xeditcritic_v4_loso_after_refits.py": (
        "CRITIC_LOSO_MANIFEST_EXACT_HEAD_AND_SCHEDULER_LAUNCH_FAILURE_EVIDENCE"
    ),
    "scripts/route_a_v3/launch_route2_xeditcritic_v4_refit_after_atomic_test.py": (
        "CRITIC_REFIT_MANIFEST_EXACT_HEAD_AND_SCHEDULER_LAUNCH_FAILURE_EVIDENCE"
    ),
    "scripts/route_a_v3/prepare_route2_xeditcritic_v4_confirmation_configs.py": (
        "CRITIC_CONFIRMATION_CONFIG_RUNNER_HEAD_LINEAGE"
    ),
    "scripts/route_a_v3/prepare_route2_xeditcritic_v4_posttest_configs.py": (
        "CRITIC_REFIT_LOSO_CONFIG_RUNNER_HEAD_LINEAGE"
    ),
    "scripts/route_a_v3/run_route2_xedit_v4_confirmation_training_scheduler.py": (
        "CRITIC_CONFIRMATION_RETURN_CODE_AND_UNIQUE_SUMMARY_FAIL_CLOSED"
    ),
    "scripts/route_a_v3/run_route2_xeditcritic_v403_control_recovery_scheduler.py": (
        "CRITIC_CONTROL_PACKAGE_FIRST_FAILURE_AND_CLEAN_HEAD_FAIL_CLOSED"
    ),
    "scripts/route_a_v3/run_route2_xeditcritic_v4_loso_scheduler.py": (
        "CRITIC_LOSO_FIRST_FAILURE_JOB_BARRIER_SPAWN_AND_WORKTREE_IDENTITY_FAIL_CLOSED"
    ),
    "scripts/route_a_v3/run_route2_xeditcritic_v4_refit_scheduler.py": (
        "CRITIC_REFIT_FIRST_FAILURE_JOB_ADJUDICATION_SPAWN_AND_WORKTREE_IDENTITY_FAIL_CLOSED"
    ),
    "scripts/route_a_v3/train_route2_xeditcritic_v4.py": (
        "CRITIC_SEED_BEFORE_MODEL_CUDA_A100_BF16_AND_V2_TERMINAL_PROVENANCE"
    ),
    "scripts/route_a_v3/transition_adjudicate_route2_xeditcritic_v403_cross_root_screen.py": (
        "CRITIC_CROSS_ROOT_PROVENANCE_WORKTREE_IDENTITY_AND_ONE_SHOT_SIBLING_FAILURE"
    ),
}
TRAINING_SEMANTICS_INCREMENTAL_CHANGED_PATHS = (
    "scripts/route_a_v3/launch_route2_xeditcritic_v403_controls_after_full.py",
    "scripts/route_a_v3/launch_route2_xeditcritic_v4_loso_after_refits.py",
    "scripts/route_a_v3/launch_route2_xeditcritic_v4_refit_after_atomic_test.py",
    "scripts/route_a_v3/run_route2_xeditcritic_v4_loso_scheduler.py",
    "scripts/route_a_v3/run_route2_xeditcritic_v4_refit_scheduler.py",
    "scripts/route_a_v3/transition_adjudicate_route2_xeditcritic_v403_cross_root_screen.py",
)
TRAINING_SEMANTICS_INCREMENTAL_PATH_CLASSIFICATION = {
    path: TRAINING_SEMANTICS_REAUDIT_PATH_CLASSIFICATION[path]
    for path in TRAINING_SEMANTICS_INCREMENTAL_CHANGED_PATHS
}
TRAINING_SEMANTICS_CHANGED_FLAGS = {
    "critic_trainer_code_changed": True,
    "critic_parameter_initialization_semantics_changed": True,
    "critic_confirmation_config_lineage_changed": True,
    "critic_gate_and_terminal_acceptance_changed": True,
    "critic_scheduler_fail_closed_semantics_changed": True,
    "critic_scheduler_process_launch_failure_evidence_changed": True,
    "critic_job_and_barrier_spawn_fail_closed_semantics_changed": True,
    "critic_cross_root_worktree_identity_and_one_shot_failure_changed": True,
    "shared_training_ledger_schema_changed": True,
}
TRAINING_SEMANTICS_UNCHANGED_FLAGS = {
    "critic_screen_config_changed": False,
    "optimizer_objective_sampler_data_budget_changed": False,
    "model_architecture_forward_loss_changed": False,
    "critic_data_geometry_changed": False,
    "critic_sampler_semantics_changed": False,
    "critic_physical_or_effective_batch_changed": False,
    "critic_seed_cohort_changed": False,
    "critic_optimizer_update_budget_changed": False,
    "scientific_thresholds_changed": False,
}
BASE_CONFIG = (
    WORKTREE / "configs/route_a_v3_route2_xeditcritic_v4_screen_v1.json"
)
BASE_PROTOCOL = (
    WORKTREE
    / "configs/route_a_v3_route2_xeditcritic_v4_confirmation_protocol_v1.json"
)
TRAINER = WORKTREE / "scripts/route_a_v3/train_route2_xeditcritic_v4.py"
SCHEDULER = (
    WORKTREE
    / "scripts/route_a_v3/run_route2_xedit_v4_confirmation_training_scheduler.py"
)
LEGACY_GATE = (
    ROOT / "experiments/xeditcritic_v4/screen_seed_20260907/screen_gate.json"
)
FULL_RUNTIME = (
    ROOT
    / "experiments/xeditcritic_v4/"
    f"v403_rng_replay_fix_runner_{REPAIRED_SCREEN_PROVENANCE_GIT_HEAD}/runtime.json"
)
ATTEMPT_RECEIPT = (
    ROOT
    / "authorizations/xeditcritic_v4/"
    "v403_cross_root_confirmation_attempt.json"
)
RUNNER_VERIFICATION_RECEIPT_SCHEMA = (
    "route_a_v3_route2_xedit_v403_successor_runner_verification_receipt.v1"
)
RUNNER_VERIFICATION_RECEIPT_PASS = (
    "XEDIT_V403_SUCCESSOR_RUNNER_VERIFICATION_PASS"
)
MIN_FOCUSED_TESTS = 203
FOCUSED_PROCESS_GROUP_COUNT = 8
FOCUSED_GROUP_REQUIRED_TEST_MARKERS = (
    (
        "test_score_route2_xeditflow_closed_frozen_methods_v3.py",
        "test_launch_route2_xeditflow_v403_guidance_after_dual_readiness.py",
        "test_launch_route2_xeditflow_v4_guidance_authorization_after_dual_readiness.py",
        "test_launch_route2_xeditflow_v4_guidance_screen_after_authorization.py",
        "test_launch_route2_xeditflow_v4_final_after_guidance_screen.py",
        "test_launch_route2_xeditcritic_v403_confirmation_after_cross_root_screen.py",
        "test_launch_route2_xeditsetflow_v403_recovered_confirmation_posttraining.py",
        "test_authorize_route2_xeditsetflow_v403_recovered_confirmation.py",
    ),
    (
        "test_transition_adjudicate_route2_xeditcritic_v403_cross_root_screen.py",
        "test_prepare_route2_xeditcritic_v4_confirmation_configs.py",
        "test_route2_xeditcritic_v4_confirmation_runtime.py",
    ),
    (
        "test_run_route2_xeditsetflow_v402_terminal_validation_scheduler.py",
        "test_adjudicate_route2_xeditsetflow_v4_confirmation.py",
        "test_route2_xeditsetflow_training_v4.py",
        "test_route2_xeditsetflow_s1_protocol.py",
        "test_route2_xeditsetflow_s1.py",
        "test_train_route2_xeditsetflow_s1.py",
        "test_validate_route2_xeditsetflow_s1_checkpoint.py",
        "test_route2_xeditsetflow_gate_s1.py",
        "test_run_route2_xeditsetflow_s1_screen_scheduler.py",
        "test_launch_route2_xeditsetflow_s1_screen_after_v403_terminal.py",
        "test_transition_record_route2_xeditsetflow_s1_930_terminal_invalidation.py",
        "test_route2_xeditsetflow_confirmation_s1.py",
        "test_launch_route2_xeditsetflow_s1_confirmation_after_screen_pass.py",
        "test_launch_route2_xeditsetflow_s1_confirmation_posttraining.py",
        "test_adjudicate_route2_xeditsetflow_s1_confirmation.py",
    ),
    (
        "test_run_route2_xeditflow_v4_guidance_screen_scheduler.py",
        "test_adjudicate_route2_xeditflow_guidance_screen_v4.py",
        "test_route2_xeditflow_guidance_v4.py",
    ),
    (
        "test_train_route2_xeditcritic_v4.py",
        "test_run_route2_xeditcritic_v4_loso_scheduler.py",
    ),
    (
        "test_run_route2_xedit_v4_confirmation_training_scheduler.py",
        "test_run_route2_xedit_v4_confirmation_posttraining_scheduler.py",
    ),
    (
        "test_launch_route2_xedit_v4_confirmation_training_after_screen_pass.py",
        "test_launch_route2_xedit_v4_confirmation_posttraining_after_terminal.py",
        "test_launch_route2_xeditsetflow_v403_recovered_confirmation.py",
        "test_launch_route2_xeditcritic_v403_controls_after_full.py",
        "test_transition_record_route2_xeditcritic_v403_controls_oom_terminal.py",
        "test_launch_route2_xeditcritic_v4_atomic_frozen_test_after_confirmation.py",
        "test_launch_route2_xeditcritic_v4_refit_after_atomic_test.py",
        "test_launch_route2_xeditcritic_v4_loso_after_refits.py",
    ),
    (
        "test_prepare_route2_xeditflow_final_generation_configs_v4.py",
        "test_evaluate_route2_xeditflow_closed_scores_v4.py",
        "test_compare_route2_xeditflow_independent_evaluator_v4.py",
        "test_xeditflow_v4_final_evidence_chain.py",
        "test_run_route2_xeditflow_strongest_timing_v4.py",
        "test_reproduce_route2_base_flow_v2_handover_validation.py",
        "test_export_route2_xeditflow_v4_terminal_training_ledger.py",
    ),
)
V332_TEST_GLOB_MARKER = "*v332*.py"
CONFIRMATION_SEEDS = (20260908, 20260909, 20260910)
PHYSICAL_GPUS = (0, 1, 2, 3, 4, 5)
ARM_ORDER = (
    "c0_v4",
    "v4_full",
    "v4_source_only",
    "v4_edit_metadata_only",
    "v4_no_candidate_sequence",
    "v4_candidate_bundle_permutation",
    "v4_no_cross",
    "v4_no_moe",
)
CONTROL_RUN_IDS = ARM_ORDER[2:]
SCREEN_CHECKS = {
    "minimum_spearman_formula",
    "standardized_mae_ceiling",
    "standardized_mae_not_worse_than_c0",
    "positive_task_breadth",
    "minimum_tasks_won_over_c0",
    "beats_source_only",
    "beats_edit_metadata_only",
    "beats_no_candidate_sequence",
    "permutation_aggregate_margin",
    "permutation_five_of_six_tasks",
    "no_cross_margin",
    "no_moe_margin",
    "protected_reads_zero",
    "formal_parameter_batch_memory_update_identity",
}
SOURCE_BARRIERS = (
    "a100_current_head_focused_tests_passed",
    "a100_current_head_v332_tests_passed",
    "bottom_six_cache_terminal_complete",
    "formal_parameter_preflight_passed",
    "formal_memory_preflight_passed",
)
TRAINING_SEMANTIC_PATHS = (
    "configs/route_a_v3_route2_xeditcritic_v4_screen_v1.json",
    "core",
    "scripts/route_a_v3/preflight_route2_xeditcritic_v4.py",
    "scripts/route_a_v3/smoke_route2_xeditcritic_v402_recovery.py",
    "scripts/route_a_v3/train_route2_xeditcritic_v3.py",
    "scripts/route_a_v3/train_route2_xeditcritic_v4.py",
    "scripts/route_a_v3/prepare_route2_xeditcritic_v4_confirmation_configs.py",
    "scripts/route_a_v3/adjudicate_route2_xeditcritic_v4_confirmation.py",
    "scripts/route_a_v3/run_route2_xedit_v4_confirmation_training_scheduler.py",
    "scripts/route_a_v3/launch_route2_xeditcritic_v403_controls_after_full.py",
    "scripts/route_a_v3/run_route2_xeditcritic_v403_control_recovery_scheduler.py",
    "scripts/route_a_v3/transition_adjudicate_route2_xeditcritic_v403_cross_root_screen.py",
    "scripts/route_a_v3/adjudicate_route2_xeditcritic_v4_screen.py",
    "scripts/route_a_v3/prepare_route2_xeditcritic_v4_posttest_configs.py",
    "scripts/route_a_v3/adjudicate_route2_xeditcritic_v4_posttest.py",
    "scripts/route_a_v3/launch_route2_xeditcritic_v4_refit_after_atomic_test.py",
    "scripts/route_a_v3/run_route2_xeditcritic_v4_refit_scheduler.py",
    "scripts/route_a_v3/launch_route2_xeditcritic_v4_loso_after_refits.py",
    "scripts/route_a_v3/run_route2_xeditcritic_v4_loso_scheduler.py",
    "scripts/route_a_v3/launch_route2_xedit_v4_postscreen_after_screen_terminal.py",
    ":(glob)scripts/route_a_v3/route2_mrnabert_*.py",
)


class XEditCriticV403ConfirmationLaunchError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticV403ConfirmationLaunchError(message)


def command(
    arguments: Sequence[str], *, cwd: Path = WORKTREE
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(arguments),
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def require_zero_protected_reads(
    payload: Mapping[str, Any], *, label: str
) -> None:
    require(
        int(payload.get("development_test_outcome_reads", -1)) == 0
        and int(payload.get("new_final_evaluation_outcome_reads", -1)) == 0,
        f"{label} reports a protected outcome read",
    )


def write_new_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    require(not path.exists(), f"artifact already exists: {path}")
    partial = path.with_suffix(path.suffix + ".partial")
    require(not partial.exists(), f"partial artifact already exists: {partial}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def write_scheduler_launch_failure_evidence(
    path: Path,
    *,
    expected_head: str,
    command_line: Sequence[str],
    schedule_path: Path,
    runtime_path: Path,
    created_artifacts: Mapping[str, Path],
    error: Exception,
) -> None:
    write_new_atomic(
        path,
        {
            "schema_version": (
                "route_a_v3_route2_xeditcritic_v403_cross_root_confirmation_"
                "scheduler_launch_failure.v1"
            ),
            "status": (
                "XEDITCRITIC_V403_CONFIRMATION_SCHEDULER_LAUNCH_"
                "TECHNICAL_FAILURE"
            ),
            "failure_stage": "SCHEDULER_PROCESS_LAUNCH",
            "expected_git_head": expected_head,
            "worktree": str(WORKTREE),
            "scheduler_command": list(command_line),
            "schedule_path": str(schedule_path),
            "intended_runtime_manifest": str(runtime_path),
            "created_artifact_paths": {
                key: str(value) for key, value in created_artifacts.items()
            },
            "error_type": type(error).__name__,
            "error": str(error),
            "scheduler_started": False,
            "gpu_job_started": False,
            "automatic_retry_attempted": False,
            "free_memory_gate_applied": False,
            "cpu_fallback_used": False,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )


def spawn_scheduler_with_failure_evidence(
    *,
    failure_path: Path,
    expected_head: str,
    command_line: Sequence[str],
    schedule_path: Path,
    runtime_path: Path,
    scheduler_log: Path,
    created_artifacts: Mapping[str, Path],
) -> subprocess.Popen[str]:
    stream = scheduler_log.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            list(command_line),
            cwd=WORKTREE,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception as error:
        stream.close()
        write_scheduler_launch_failure_evidence(
            failure_path,
            expected_head=expected_head,
            command_line=command_line,
            schedule_path=schedule_path,
            runtime_path=runtime_path,
            created_artifacts=created_artifacts,
            error=error,
        )
        raise XEditCriticV403ConfirmationLaunchError(
            "Critic confirmation scheduler process could not start; durable "
            f"technical failure evidence: {failure_path}"
        ) from error
    stream.close()
    return process


def control_output_root(control_runner_head: str) -> Path:
    return (
        ROOT
        / "experiments/xeditcritic_v4/"
        f"screen_seed_20260907_{CONTROL_RETRY_IDENTITY}_{control_runner_head}"
    )


def control_runtime_path(control_runner_head: str) -> Path:
    return (
        ROOT
        / "experiments/xeditcritic_v4/"
        f"{CONTROL_RETRY_IDENTITY}_runner_{control_runner_head}/runtime.json"
    )


def cross_root_gate_path(control_runner_head: str) -> Path:
    return (
        ROOT
        / "experiments/xeditcritic_v4/"
        "screen_seed_20260907_v403_cross_root_controls_"
        f"retry1_{control_runner_head}/"
        "screen_gate.json"
    )


def expected_summary_paths(
    control_runner_head: str,
    *,
    actual_control_output_root: Path | None = None,
) -> dict[str, Path]:
    historical = (
        ROOT
        / "experiments/xeditcritic_v4/"
        f"screen_seed_20260907_v402_recovery_runner_{HISTORICAL_C0_GIT_HEAD}"
    )
    full = (
        ROOT
        / "experiments/xeditcritic_v4/"
        "screen_seed_20260907_v403_rng_replay_fix_"
        f"{REPAIRED_SCREEN_PROVENANCE_GIT_HEAD}"
    )
    controls = (
        control_output_root(control_runner_head)
        if actual_control_output_root is None
        else actual_control_output_root
    )
    result = {
        "c0_v4": historical / "c0_v4/run_summary.json",
        "v4_full": full / "v4_full/run_summary.json",
    }
    result.update(
        {
            run_id: controls / run_id / "run_summary.json"
            for run_id in CONTROL_RUN_IDS
        }
    )
    return result


def expected_source_role(run_id: str) -> str:
    if run_id == "c0_v4":
        return "HISTORICAL_MATCHED_C0_TERMINAL_SUMMARY"
    if run_id == "v4_full":
        return "CURRENT_V403_REPAIRED_FULL_TERMINAL_SUMMARY"
    return "CURRENT_HEAD_CONTROL_TERMINAL_SUMMARY"


def expected_source_head(run_id: str, control_runner_head: str) -> str:
    return (
        HISTORICAL_C0_GIT_HEAD
        if run_id == "c0_v4"
        else REPAIRED_SCREEN_PROVENANCE_GIT_HEAD
        if run_id == "v4_full"
        else control_runner_head
    )


def control_runner_head_from_gate(payload: Mapping[str, Any]) -> str:
    transition = payload.get("cross_root_transition")
    head = str(
        transition.get("control_runner_git_head", "")
        if isinstance(transition, Mapping)
        else ""
    )
    require(
        re.fullmatch(r"[0-9a-f]{40}", head) is not None
        and head
        not in {
            REPAIRED_SCREEN_PROVENANCE_GIT_HEAD,
            HISTORICAL_C0_GIT_HEAD,
        },
        "cross-root gate lacks a new exact controls runner HEAD",
    )
    return head


def control_lineage_paths_from_gate(
    payload: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Consume the frozen retry1 runtime and output paths from the gate."""

    control_runner_head = control_runner_head_from_gate(payload)
    transition = payload.get("cross_root_transition")
    require(
        isinstance(transition, Mapping),
        "cross-root gate lacks transition provenance",
    )
    runtime_value = transition.get("control_runtime_path")
    require(
        isinstance(runtime_value, str) and bool(runtime_value.strip()),
        "cross-root gate lacks the frozen control runtime path",
    )
    runtime_path = Path(runtime_value)
    require(
        runtime_path.is_absolute()
        and runtime_path == control_runtime_path(control_runner_head),
        "cross-root gate control runtime path is not the frozen retry1 family",
    )
    arm_sources = transition.get("arm_sources")
    require(
        isinstance(arm_sources, Mapping),
        "cross-root gate lacks frozen arm-source paths",
    )
    output_roots: set[Path] = set()
    for run_id in CONTROL_RUN_IDS:
        row = arm_sources.get(run_id)
        summary_value = (
            row.get("summary_path") if isinstance(row, Mapping) else None
        )
        require(
            isinstance(summary_value, str) and bool(summary_value.strip()),
            f"cross-root gate lacks the frozen summary path for {run_id}",
        )
        summary_path = Path(summary_value)
        require(
            summary_path.is_absolute()
            and summary_path.name == "run_summary.json"
            and summary_path.parent.name == run_id,
            f"cross-root gate summary path shape changed for {run_id}",
        )
        output_roots.add(summary_path.parent.parent)
    require(
        output_roots == {control_output_root(control_runner_head)},
        "cross-root gate control summary paths do not bind the frozen retry1 output root",
    )
    return runtime_path, next(iter(output_roots))


def validate_cross_root_gate(payload: Mapping[str, Any]) -> None:
    control_runner_head = control_runner_head_from_gate(payload)
    actual_control_runtime, actual_control_output_root = (
        control_lineage_paths_from_gate(payload)
    )
    require(
        payload.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_screen_gate.v1"
        and payload.get("status") == "XEDITCRITIC_V4_SCREEN_PASS"
        and payload.get("passed") is True
        and payload.get("confirmation_authorized") is True
        and payload.get("development_test_authorized") is False
        and payload.get("selectable_model") == "V4-FULL"
        and int(payload.get("screen_seed", -1)) == 20260907,
        "fixed cross-root Critic V4 screen gate is not terminal PASS",
    )
    checks = payload.get("checks")
    require(
        isinstance(checks, Mapping)
        and set(checks) == SCREEN_CHECKS
        and all(value is True for value in checks.values()),
        "fixed cross-root Critic V4 screen checks are incomplete or non-PASS",
    )
    require_zero_protected_reads(payload, label="cross-root screen gate")
    transition = payload.get("cross_root_transition")
    require(
        isinstance(transition, Mapping)
        and transition.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v403_cross_root_input.v1"
        and transition.get("ordered_run_ids") == list(ARM_ORDER)
        and transition.get("historical_c0_git_head")
        == HISTORICAL_C0_GIT_HEAD
        and transition.get("historical_full_git_head")
        == REPAIRED_SCREEN_PROVENANCE_GIT_HEAD
        and transition.get("control_runner_git_head") == control_runner_head
        and transition.get("full_runtime_path") == str(FULL_RUNTIME)
        and transition.get("full_runtime_status")
        == "XEDITCRITIC_V403_FULL_RECOVERY_TERMINAL"
        and transition.get("control_runtime_path")
        == str(actual_control_runtime)
        and transition.get("control_runtime_status")
        == (
            "XEDITCRITIC_V403_CONTROL_RECOVERY_"
            "ALL_SIX_SUMMARIES_TERMINAL"
        )
        and isinstance(transition.get("frozen_config_path"), str)
        and bool(str(transition.get("frozen_config_path")).strip())
        and transition.get("legacy_gate_path") == str(LEGACY_GATE)
        and transition.get("legacy_gate_preserved") is True
        and int(transition.get("terminal_summary_payloads_read", -1)) == 8
        and transition.get("full_retrained") is False
        and transition.get("c0_retrained") is False
        and transition.get("old_v402_stopped_process_resumed") is False
        and transition.get("scientific_thresholds_changed") is False,
        "fixed cross-root Critic V4 transition provenance changed",
    )
    require_zero_protected_reads(
        transition, label="cross-root transition provenance"
    )
    arm_sources = transition.get("arm_sources")
    summary_paths = expected_summary_paths(
        control_runner_head,
        actual_control_output_root=actual_control_output_root,
    )
    require(
        isinstance(arm_sources, Mapping)
        and set(arm_sources.keys()) == set(ARM_ORDER),
        "cross-root gate does not contain the exact eight-arm key set",
    )
    for run_id in ARM_ORDER:
        row = arm_sources[run_id]
        expected_head = expected_source_head(run_id, control_runner_head)
        require(
            isinstance(row, Mapping)
            and row.get("summary_path") == str(summary_paths[run_id])
            and row.get("source_role") == expected_source_role(run_id)
            and row.get("training_git_head") == expected_head
            and row.get("launch_authorization_schema_version")
            == (
                "route_a_v3_route2_xeditcritic_v4_"
                "screen_launch_authorization.v1"
            )
            and row.get("launch_authorization_status")
            == "XEDITCRITIC_V4_SCREEN_LAUNCH_AUTHORIZED"
            and row.get("authorized_git_head") == expected_head
            and row.get("run_id_authorization_verified") is True
            and row.get(
                "authorization_protected_outcome_reads_verified_zero"
            )
            is True
            and isinstance(row.get("launch_authorization_path"), str)
            and bool(str(row["launch_authorization_path"]).strip()),
            f"cross-root provenance changed for {run_id}",
        )


def load_and_validate_source_authorizations(
    gate: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    control_runner_head = control_runner_head_from_gate(gate)
    arm_sources = gate["cross_root_transition"]["arm_sources"]
    authorizations: dict[str, dict[str, Any]] = {}
    for run_id in ARM_ORDER:
        path = Path(str(arm_sources[run_id]["launch_authorization_path"]))
        require(path.is_file(), f"source authorization is absent for {run_id}: {path}")
        authorization = read_json(path)
        expected_head = expected_source_head(run_id, control_runner_head)
        authorized_run_ids = authorization.get("authorized_run_ids")
        barriers = authorization.get("barriers")
        require(
            authorization.get("schema_version")
            == (
                "route_a_v3_route2_xeditcritic_v4_"
                "screen_launch_authorization.v1"
            )
            and authorization.get("status")
            == "XEDITCRITIC_V4_SCREEN_LAUNCH_AUTHORIZED"
            and authorization.get("authorized_git_head") == expected_head
            and isinstance(authorized_run_ids, list)
            and run_id in authorized_run_ids
            and isinstance(barriers, Mapping)
            and all(barriers.get(key) is True for key in SOURCE_BARRIERS),
            f"source authorization identity or barriers changed for {run_id}",
        )
        require_zero_protected_reads(
            authorization, label=f"{run_id} source authorization"
        )
        authorizations[run_id] = authorization
    return authorizations


def derive_confirmation_protocol(
    base_protocol: Mapping[str, Any],
    *,
    control_runner_head: str,
) -> dict[str, Any]:
    require(
        base_protocol.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_confirmation_protocol.v1"
        and base_protocol.get("status")
        == "FROZEN_PROSPECTIVE_BEFORE_SCREEN_OR_CONFIRMATION_RESULT"
        and base_protocol.get("screen_gate_path") == str(LEGACY_GATE)
        and base_protocol.get("selected_model_run_id") == "v4_full"
        and base_protocol.get("matched_baseline_run_id") == "c0_v4"
        and base_protocol.get("required_seeds") == list(CONFIRMATION_SEEDS)
        and base_protocol.get("additional_seed_authorized") is False
        and base_protocol.get("development_test_authorized") is False,
        "base Critic V4 confirmation protocol changed",
    )
    require_zero_protected_reads(
        base_protocol, label="base confirmation protocol"
    )
    derived = dict(base_protocol)
    derived["screen_gate_path"] = str(
        cross_root_gate_path(control_runner_head)
    )
    require(
        all(
            derived[key] == value
            for key, value in base_protocol.items()
            if key != "screen_gate_path"
        ),
        "Critic V4 confirmation science changed during gate derivation",
    )
    return derived


def runner_verification_receipt_path(runner_head: str) -> Path:
    return (
        ROOT
        / "audits/xedit_v4/"
        f"v403_successor_runner_verification_{runner_head}.json"
    )


def validate_runner_verification_receipt(
    receipt: Mapping[str, Any],
    *,
    runner_head: str,
    receipt_path: Path,
) -> None:
    require(
        receipt_path == runner_verification_receipt_path(runner_head),
        "Critic successor runner verification receipt path changed",
    )
    require(
        receipt.get("schema_version") == RUNNER_VERIFICATION_RECEIPT_SCHEMA
        and receipt.get("status") == RUNNER_VERIFICATION_RECEIPT_PASS
        and receipt.get("runner_git_head") == runner_head
        and receipt.get("worktree_clean") is True,
        "Critic successor runner verification is not exact-HEAD clean PASS",
    )
    focused = receipt.get("focused_tests")
    require(
        isinstance(focused, Mapping),
        "Critic successor runner receipt lacks focused tests",
    )
    focused_commands = focused.get("command")
    group_passed_counts = focused.get("group_passed_counts")
    focused_passed_count = focused.get("passed_count")
    focused_failed_count = focused.get("failed_count")
    require(
        focused.get("isolated_process_groups") is True
        and isinstance(focused_commands, list)
        and len(focused_commands) == FOCUSED_PROCESS_GROUP_COUNT
        and all(
            isinstance(value, str) and bool(value)
            for value in focused_commands
        )
        and isinstance(group_passed_counts, list)
        and len(group_passed_counts) == FOCUSED_PROCESS_GROUP_COUNT
        and all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
            for value in group_passed_counts
        )
        and focused.get("passed") is True
        and isinstance(focused_passed_count, int)
        and not isinstance(focused_passed_count, bool)
        and focused_passed_count >= MIN_FOCUSED_TESTS
        and sum(group_passed_counts) == focused_passed_count
        and isinstance(focused_failed_count, int)
        and not isinstance(focused_failed_count, bool)
        and focused_failed_count == 0,
        "Critic successor runner receipt reports failed or incomplete focused tests",
    )
    for group_index, required_markers in enumerate(
        FOCUSED_GROUP_REQUIRED_TEST_MARKERS
    ):
        require(
            all(
                marker in focused_commands[group_index]
                for marker in required_markers
            ),
            "Critic successor runner receipt focused test group "
            f"{group_index + 1} lacks required test-module coverage",
        )

    v332 = receipt.get("v332_tests")
    require(
        isinstance(v332, Mapping),
        "Critic successor runner receipt lacks V3.3.2 tests",
    )
    v332_command = v332.get("command")
    v332_passed_count = v332.get("passed_count")
    v332_failed_count = v332.get("failed_count")
    require(
        isinstance(v332_command, list)
        and bool(v332_command)
        and all(
            isinstance(value, str) and bool(value) for value in v332_command
        )
        and any(V332_TEST_GLOB_MARKER in value for value in v332_command)
        and v332.get("passed") is True
        and isinstance(v332_passed_count, int)
        and not isinstance(v332_passed_count, bool)
        and v332_passed_count == 96
        and isinstance(v332_failed_count, int)
        and not isinstance(v332_failed_count, bool)
        and v332_failed_count == 0,
        "Critic successor runner receipt reports failed or incomplete V3.3.2 tests",
    )
    require_zero_protected_reads(
        receipt, label="successor runner verification receipt"
    )


def validate_training_semantics_baseline_audit(
    audit: Mapping[str, Any],
) -> None:
    consumer_review = audit.get("audit_consumer_launcher_review")
    previous_successor_review = audit.get("previous_successor_audit_review")
    changed_flags_match = all(
        audit.get(name) is expected
        for name, expected in TRAINING_SEMANTICS_CHANGED_FLAGS.items()
    )
    unchanged_flags_match = all(
        audit.get(name) is expected
        for name, expected in TRAINING_SEMANTICS_UNCHANGED_FLAGS.items()
    )
    require(
        audit.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v403_confirmation_training_semantics_reaudit.v2"
        and audit.get("status")
        == "XEDITCRITIC_V403_CONFIRMATION_TRAINING_SEMANTICS_REAUDIT_V2_PASS"
        and audit.get("previous_audited_baseline_git_head")
        == TRAINING_SEMANTICS_PREVIOUS_AUDITED_BASELINE_HEAD
        and audit.get("previous_successor_semantic_baseline_git_head")
        == TRAINING_SEMANTICS_PREVIOUS_SUCCESSOR_BASELINE_HEAD
        and audit.get("audited_successor_semantic_baseline_git_head")
        == TRAINING_SEMANTICS_AUDITED_SUCCESSOR_BASELINE_HEAD
        and audit.get("critic_repaired_screen_provenance_git_head")
        == REPAIRED_SCREEN_PROVENANCE_GIT_HEAD
        and audit.get("historical_c0_git_head") == HISTORICAL_C0_GIT_HEAD
        and audit.get("frozen_training_semantic_pathspec")
        == list(TRAINING_SEMANTIC_PATHS)
        and tuple(audit.get("changed_training_semantic_paths", []))
        == TRAINING_SEMANTICS_REAUDIT_CHANGED_PATHS
        and audit.get("changed_training_semantic_paths")
        == sorted(audit.get("changed_training_semantic_paths", []))
        and audit.get("path_classification")
        == TRAINING_SEMANTICS_REAUDIT_PATH_CLASSIFICATION
        and tuple(
            audit.get(
                "incremental_changed_training_semantic_paths_since_previous_successor",
                [],
            )
        )
        == TRAINING_SEMANTICS_INCREMENTAL_CHANGED_PATHS
        and audit.get("incremental_path_classification")
        == TRAINING_SEMANTICS_INCREMENTAL_PATH_CLASSIFICATION
        and changed_flags_match
        and unchanged_flags_match
        and audit.get(
            "changes_accepted_as_successor_safety_and_provenance_hardening"
        )
        is True
        and audit.get("repaired_screen_is_historical_provenance_only") is True
        and audit.get("historical_c0_is_provenance_only") is True
        and audit.get(
            "audited_successor_is_confirmation_training_semantic_baseline"
        )
        is True
        and isinstance(consumer_review, Mapping)
        and consumer_review.get("path")
        == (
            "scripts/route_a_v3/"
            "launch_route2_xeditcritic_v403_confirmation_after_cross_root_screen.py"
        )
        and consumer_review.get("included_in_frozen_training_semantic_pathspec")
        is False
        and consumer_review.get("expected_semantic_diff_from_baseline_to_runner")
        == []
        and consumer_review.get("runner_role")
        == "EXACT_CLEAN_AUDIT_CONSUMER_AND_CONFIRMATION_TRAINING_HEAD"
        and isinstance(previous_successor_review, Mapping)
        and previous_successor_review.get("path")
        == (
            "audits/route_a_v3_route2_xeditcritic_v403_confirmation_"
            "training_semantics_reaudit_"
            "eba5b17431cb8e19202e5ea788fd419338da2d66.json"
        )
        and previous_successor_review.get("preserved_as_history") is True
        and previous_successor_review.get("consumed_by_current_launcher")
        is False
        and audit.get("model_result_claimed") is False
        and audit.get("submission_ready") is False,
        "Critic confirmation training-semantics re-audit is absent or invalid",
    )
    require(
        len(
            {
                audit["critic_repaired_screen_provenance_git_head"],
                audit["historical_c0_git_head"],
                audit["previous_audited_baseline_git_head"],
                audit["previous_successor_semantic_baseline_git_head"],
                audit["audited_successor_semantic_baseline_git_head"],
            }
        )
        == 5,
        "Critic confirmation training-semantics provenance roles are mixed",
    )
    require_zero_protected_reads(
        audit, label="Critic confirmation training-semantics re-audit"
    )


def validate_controls_oom_retry_technical_baseline_audit(
    audit: Mapping[str, Any],
) -> None:
    consumer_review = audit.get("confirmation_consumer_review")
    gpu_assignment = audit.get("control_run_id_to_physical_gpu_index")
    unchanged_flags_match = all(
        audit.get(name) is expected
        for name, expected in CONTROLS_OOM_RETRY_UNCHANGED_FLAGS.items()
    )
    require(
        audit.get("schema_version")
        == CONTROLS_OOM_RETRY_TECHNICAL_BASELINE_AUDIT_SCHEMA
        and audit.get("status")
        == CONTROLS_OOM_RETRY_TECHNICAL_BASELINE_AUDIT_STATUS
        and audit.get("previous_audited_successor_semantic_baseline_git_head")
        == TRAINING_SEMANTICS_AUDITED_SUCCESSOR_BASELINE_HEAD
        and audit.get("previous_audited_successor_semantic_baseline_audit")
        == str(TRAINING_SEMANTICS_BASELINE_AUDIT.relative_to(WORKTREE))
        and audit.get("controls_oom_retry_technical_baseline_git_head")
        == CONTROLS_OOM_RETRY_TECHNICAL_BASELINE_HEAD
        and audit.get("frozen_training_semantic_pathspec")
        == list(TRAINING_SEMANTIC_PATHS)
        and tuple(
            audit.get(
                "changed_training_semantic_paths_from_f1a_to_technical_baseline",
                [],
            )
        )
        == CONTROLS_OOM_RETRY_CHANGED_TRAINING_SEMANTIC_PATHS
        and audit.get(
            "changed_training_semantic_paths_from_f1a_to_technical_baseline"
        )
        == sorted(
            audit.get(
                "changed_training_semantic_paths_from_f1a_to_technical_baseline",
                [],
            )
        )
        and audit.get("path_classification")
        == CONTROLS_OOM_RETRY_PATH_CLASSIFICATION
        and audit.get("immutable_oom_terminal_lineage_added") is True
        and audit.get("new_independent_retry_family_added") is True
        and audit.get("same_family_retry_authorized") is False
        and audit.get("successor_authorized_by_this_audit") is False
        and audit.get("fixed_two_wave_scheduler_added") is True
        and audit.get("pytorch_allocator_environment_added") is True
        and audit.get("package_first_failure_semantics_preserved") is True
        and audit.get("configured_gpu_existence_only_policy_preserved") is True
        and audit.get("physical_gpu_indices") == list(PHYSICAL_GPUS)
        and gpu_assignment
        == {
            run_id: gpu
            for gpu, run_id in enumerate(CONTROL_RUN_IDS)
        }
        and unchanged_flags_match
        and audit.get("baseline_scope")
        == "TECHNICAL_EXECUTION_SEMANTICS_ONLY"
        and audit.get("old_f1a_audit_preserved_and_validated") is True
        and isinstance(consumer_review, Mapping)
        and consumer_review.get("path")
        == (
            "scripts/route_a_v3/"
            "launch_route2_xeditcritic_v403_confirmation_after_cross_root_screen.py"
        )
        and consumer_review.get("included_in_frozen_training_semantic_pathspec")
        is False
        and consumer_review.get(
            "expected_semantic_diff_from_technical_baseline_to_runner"
        )
        == []
        and audit.get("model_result_claimed") is False
        and audit.get("submission_ready") is False,
        "Critic controls OOM-retry technical training-semantics audit is absent or invalid",
    )
    require_zero_protected_reads(
        audit, label="Critic controls OOM-retry technical semantics audit"
    )


def controls_oom_retry_technical_baseline_binding() -> dict[str, str]:
    return {
        "controls_oom_retry_technical_baseline_git_head": (
            CONTROLS_OOM_RETRY_TECHNICAL_BASELINE_HEAD
        ),
        "controls_oom_retry_technical_baseline_audit": str(
            CONTROLS_OOM_RETRY_TECHNICAL_BASELINE_AUDIT
        ),
        "controls_oom_retry_technical_baseline_audit_status": (
            CONTROLS_OOM_RETRY_TECHNICAL_BASELINE_AUDIT_STATUS
        ),
    }


def validate_corrected_screen_confirmation_provenance_audit(
    audit: Mapping[str, Any],
) -> None:
    consumer_review = audit.get("critic_confirmation_consumer_review")
    unchanged_flags_match = all(
        audit.get(name) is expected
        for name, expected in CORRECTED_SCREEN_PROVENANCE_UNCHANGED_FLAGS.items()
    )
    require(
        audit.get("schema_version")
        == CORRECTED_SCREEN_CONFIRMATION_PROVENANCE_AUDIT_SCHEMA
        and audit.get("status")
        == CORRECTED_SCREEN_CONFIRMATION_PROVENANCE_AUDIT_STATUS
        and audit.get("previous_code_baseline_git_head")
        == PREVIOUS_CODE_BASELINE_HEAD
        and audit.get("critic_controls_oom_retry_technical_baseline_git_head")
        == CONTROLS_OOM_RETRY_TECHNICAL_BASELINE_HEAD
        and audit.get(
            "corrected_screen_confirmation_provenance_baseline_git_head"
        )
        == CORRECTED_SCREEN_CONFIRMATION_PROVENANCE_BASELINE_HEAD
        and tuple(
            audit.get(
                "critic_training_semantic_paths_from_z1_to_corrected_screen_baseline",
                [],
            )
        )
        == CORRECTED_SCREEN_PROVENANCE_CHANGED_TRAINING_SEMANTIC_PATHS
        and audit.get(
            "critic_training_semantic_paths_from_z1_to_corrected_screen_baseline"
        )
        == sorted(
            audit.get(
                "critic_training_semantic_paths_from_z1_to_corrected_screen_baseline",
                [],
            )
        )
        and audit.get("critic_path_classification")
        == CORRECTED_SCREEN_PROVENANCE_PATH_CLASSIFICATION
        and audit.get("invalidated_screen_runner_git_head")
        == INVALIDATED_SCREEN_RUNNER_GIT_HEAD
        and audit.get("corrected_screen_runner_git_head")
        == CORRECTED_SCREEN_RUNNER_GIT_HEAD
        and unchanged_flags_match
        and audit.get("old_930_terminal_invalidation_preserved") is True
        and audit.get("old_930_seed_initialization_repair_preserved") is True
        and audit.get("invalidated_930_family_can_authorize_successor") is False
        and audit.get(
            "current_confirmation_runner_may_differ_from_corrected_screen_runner"
        )
        is True
        and audit.get("successor_authorized_by_this_audit") is False
        and audit.get("confirmation_authorized_by_this_audit") is False
        and isinstance(consumer_review, Mapping)
        and consumer_review.get("path")
        == (
            "scripts/route_a_v3/"
            "launch_route2_xeditcritic_v403_confirmation_after_cross_root_screen.py"
        )
        and consumer_review.get("included_in_critic_training_semantic_pathspec")
        is False
        and consumer_review.get(
            "expected_critic_training_semantic_paths_from_baseline_to_runner"
        )
        == []
        and audit.get("model_result_claimed") is False
        and audit.get("submission_ready") is False,
        "corrected SetFlow screen confirmation-provenance audit is absent or invalid",
    )
    require_zero_protected_reads(
        audit, label="corrected SetFlow screen confirmation-provenance audit"
    )


def corrected_screen_confirmation_provenance_binding() -> dict[str, str]:
    return {
        "corrected_screen_confirmation_provenance_baseline_git_head": (
            CORRECTED_SCREEN_CONFIRMATION_PROVENANCE_BASELINE_HEAD
        ),
        "corrected_screen_confirmation_provenance_audit": str(
            CORRECTED_SCREEN_CONFIRMATION_PROVENANCE_AUDIT
        ),
        "corrected_screen_confirmation_provenance_audit_status": (
            CORRECTED_SCREEN_CONFIRMATION_PROVENANCE_AUDIT_STATUS
        ),
    }


def validate_runner_training_semantics(current_head: str) -> dict[str, Any]:
    require(
        re.fullmatch(r"[0-9a-f]{40}", current_head) is not None
        and current_head
        not in {
            REPAIRED_SCREEN_PROVENANCE_GIT_HEAD,
            HISTORICAL_C0_GIT_HEAD,
        },
        "Critic confirmation runner HEAD aliases historical screen provenance",
    )
    baseline_audit = read_json(TRAINING_SEMANTICS_BASELINE_AUDIT)
    validate_training_semantics_baseline_audit(baseline_audit)
    technical_baseline_audit = read_json(
        CONTROLS_OOM_RETRY_TECHNICAL_BASELINE_AUDIT
    )
    validate_controls_oom_retry_technical_baseline_audit(
        technical_baseline_audit
    )
    corrected_screen_provenance_audit = read_json(
        CORRECTED_SCREEN_CONFIRMATION_PROVENANCE_AUDIT
    )
    validate_corrected_screen_confirmation_provenance_audit(
        corrected_screen_provenance_audit
    )
    audited_changed = command(
        [
            "git",
            "diff",
            "--name-only",
            TRAINING_SEMANTICS_PREVIOUS_AUDITED_BASELINE_HEAD,
            TRAINING_SEMANTICS_AUDITED_SUCCESSOR_BASELINE_HEAD,
            "--",
            *TRAINING_SEMANTIC_PATHS,
        ]
    ).stdout.splitlines()
    require(
        tuple(audited_changed) == TRAINING_SEMANTICS_REAUDIT_CHANGED_PATHS,
        "Critic successor semantic re-audit paths differ from the exact Git diff: "
        + ", ".join(audited_changed),
    )
    incremental_changed = command(
        [
            "git",
            "diff",
            "--name-only",
            TRAINING_SEMANTICS_PREVIOUS_SUCCESSOR_BASELINE_HEAD,
            TRAINING_SEMANTICS_AUDITED_SUCCESSOR_BASELINE_HEAD,
            "--",
            *TRAINING_SEMANTIC_PATHS,
        ]
    ).stdout.splitlines()
    require(
        tuple(incremental_changed)
        == TRAINING_SEMANTICS_INCREMENTAL_CHANGED_PATHS,
        "Critic incremental successor hardening paths differ from the exact Git diff: "
        + ", ".join(incremental_changed),
    )
    technical_baseline_changed = command(
        [
            "git",
            "diff",
            "--name-only",
            TRAINING_SEMANTICS_AUDITED_SUCCESSOR_BASELINE_HEAD,
            CONTROLS_OOM_RETRY_TECHNICAL_BASELINE_HEAD,
            "--",
            *TRAINING_SEMANTIC_PATHS,
        ]
    ).stdout.splitlines()
    require(
        tuple(technical_baseline_changed)
        == CONTROLS_OOM_RETRY_CHANGED_TRAINING_SEMANTIC_PATHS,
        "Critic controls OOM-retry technical semantic paths differ from the "
        "exact Git diff: " + ", ".join(technical_baseline_changed),
    )
    corrected_screen_provenance_changed = command(
        [
            "git",
            "diff",
            "--name-only",
            CONTROLS_OOM_RETRY_TECHNICAL_BASELINE_HEAD,
            CORRECTED_SCREEN_CONFIRMATION_PROVENANCE_BASELINE_HEAD,
            "--",
            *TRAINING_SEMANTIC_PATHS,
        ]
    ).stdout.splitlines()
    require(
        tuple(corrected_screen_provenance_changed)
        == CORRECTED_SCREEN_PROVENANCE_CHANGED_TRAINING_SEMANTIC_PATHS,
        "SetFlow corrected-screen provenance paths differ from the exact "
        "Critic training-semantic Git diff: "
        + ", ".join(corrected_screen_provenance_changed),
    )
    changed = command(
        [
            "git",
            "diff",
            "--name-only",
            CORRECTED_SCREEN_CONFIRMATION_PROVENANCE_BASELINE_HEAD,
            current_head,
            "--",
            *TRAINING_SEMANTIC_PATHS,
        ]
    ).stdout.splitlines()
    require(
        not changed,
        "Critic confirmation training semantics changed after the corrected "
        "SetFlow screen provenance baseline: " + ", ".join(changed),
    )
    return {
        "historical_repaired_screen_provenance_git_head": (
            REPAIRED_SCREEN_PROVENANCE_GIT_HEAD
        ),
        "historical_c0_git_head": HISTORICAL_C0_GIT_HEAD,
        "previous_successor_semantic_baseline_git_head": (
            TRAINING_SEMANTICS_PREVIOUS_SUCCESSOR_BASELINE_HEAD
        ),
        "audited_successor_semantic_baseline_git_head": (
            TRAINING_SEMANTICS_AUDITED_SUCCESSOR_BASELINE_HEAD
        ),
        "audited_successor_semantic_baseline_audit": str(
            TRAINING_SEMANTICS_BASELINE_AUDIT
        ),
        "audited_successor_semantic_baseline_audit_status": baseline_audit[
            "status"
        ],
        **controls_oom_retry_technical_baseline_binding(),
        **corrected_screen_confirmation_provenance_binding(),
        "runner_git_head": current_head,
        "training_git_head": current_head,
        "training_semantic_paths": list(TRAINING_SEMANTIC_PATHS),
        "audited_successor_changed_training_semantic_paths": audited_changed,
        "incremental_changed_training_semantic_paths_since_previous_successor": (
            incremental_changed
        ),
        "training_semantic_diff_paths_since_audited_successor_baseline": (
            technical_baseline_changed
        ),
        "training_semantics_unchanged_since_audited_successor_baseline": False,
        "training_semantic_diff_paths_since_controls_oom_retry_technical_baseline": (
            corrected_screen_provenance_changed
        ),
        "training_semantics_unchanged_since_controls_oom_retry_technical_baseline": (
            False
        ),
        "training_semantic_diff_paths_since_corrected_screen_confirmation_provenance_baseline": (
            changed
        ),
        "training_semantics_unchanged_since_corrected_screen_confirmation_provenance_baseline": (
            True
        ),
        "repaired_screen_is_historical_provenance_only": True,
    }


def build_confirmation_configs(
    base_config: Mapping[str, Any],
    base_protocol: Mapping[str, Any],
    gate: Mapping[str, Any],
    *,
    runner_head: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validate_cross_root_gate(gate)
    require(
        control_runner_head_from_gate(gate) == runner_head,
        "confirmation runner HEAD differs from the controls runner HEAD",
    )
    screen_gate_path = cross_root_gate_path(runner_head)
    protocol = derive_confirmation_protocol(
        base_protocol, control_runner_head=runner_head
    )
    configs = build_critic_confirmation_configs_v4(
        base_config,
        protocol,
        gate,
        confirmation_runner_git_head=runner_head,
    )
    require(
        [int(config["training_seed"]) for config in configs]
        == list(CONFIRMATION_SEEDS),
        "Critic V4 confirmation seed cohort changed during derivation",
    )
    for config in configs:
        seed = int(config["training_seed"])
        require(
            config.get("schema_version")
            == "route_a_v3_route2_xeditcritic_v4_confirmation_runtime.v1"
            and config.get("status")
            == "FROZEN_CONFIRMATION_CONFIG_NOT_STARTED"
            and config.get("run_stage") == "CONFIRMATION"
            and config.get("confirmation_runner_git_head") == runner_head
            and config.get("required_confirmation_run_ids")
            == ["v4_full", "c0_v4"]
            and config.get("screen_gate_path") == str(screen_gate_path)
            and config.get("output_root")
            == str(Path(str(protocol["run_root"])) / f"seed_{seed}")
            and config.get("required_confirmation_seeds")
            == list(CONFIRMATION_SEEDS)
            and config.get("additional_seed_authorized") is False
            and config.get("development_test_outcomes_accessed") is False
            and config.get("new_final_evaluation_outcomes_accessed") is False,
            f"derived Critic V4 confirmation config changed for seed {seed}",
        )
    return protocol, configs


def materialize_config_package(
    configs: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    runner_head: str,
) -> dict[str, Any]:
    config_root = Path(str(protocol["runtime_config_root"]))
    run_root = Path(str(protocol["run_root"]))
    staging = config_root.with_name(config_root.name + ".partial")
    require(not config_root.exists(), f"Critic confirmation config root exists: {config_root}")
    require(not staging.exists(), f"Critic confirmation partial config root exists: {staging}")
    require(not run_root.exists(), f"Critic confirmation run root exists: {run_root}")
    staging.mkdir(parents=True)
    paths: list[str] = []
    for config in configs:
        seed = int(config["training_seed"])
        filename = f"seed_{seed}.json"
        path = staging / filename
        path.write_text(
            json.dumps(dict(config), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        paths.append(str(config_root / filename))
    manifest = {
        "schema_version": (
            "route_a_v3_route2_xeditcritic_v4_"
            "confirmation_config_manifest.v1"
        ),
        "status": "THREE_MATCHED_CONFIRMATION_CONFIGS_PREPARED_NOT_STARTED",
        "required_run_ids": ["v4_full", "c0_v4"],
        "required_seeds": list(CONFIRMATION_SEEDS),
        "config_paths": paths,
        "confirmation_runner_git_head": runner_head,
        "training_git_head": runner_head,
        "audited_successor_semantic_baseline_git_head": (
            TRAINING_SEMANTICS_AUDITED_SUCCESSOR_BASELINE_HEAD
        ),
        "screen_gate_path": str(cross_root_gate_path(runner_head)),
        "historical_c0_git_head": HISTORICAL_C0_GIT_HEAD,
        "historical_full_git_head": REPAIRED_SCREEN_PROVENANCE_GIT_HEAD,
        "control_runner_git_head": runner_head,
        "legacy_screen_gate_preserved": True,
        "launch_attempt_receipt": str(ATTEMPT_RECEIPT),
        "development_test_authorized": False,
        "guidance_authorized": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(staging, config_root)
    return manifest


def validate_manifest(
    manifest: Mapping[str, Any], *, runner_head: str
) -> dict[int, Path]:
    require(
        manifest.get("schema_version")
        == (
            "route_a_v3_route2_xeditcritic_v4_"
            "confirmation_config_manifest.v1"
        )
        and manifest.get("status")
        == "THREE_MATCHED_CONFIRMATION_CONFIGS_PREPARED_NOT_STARTED"
        and manifest.get("required_run_ids") == ["v4_full", "c0_v4"]
        and manifest.get("required_seeds") == list(CONFIRMATION_SEEDS)
        and len(manifest.get("config_paths", [])) == 3
        and manifest.get("confirmation_runner_git_head") == runner_head
        and manifest.get("training_git_head") == runner_head
        and manifest.get("audited_successor_semantic_baseline_git_head")
        == TRAINING_SEMANTICS_AUDITED_SUCCESSOR_BASELINE_HEAD
        and manifest.get("screen_gate_path")
        == str(cross_root_gate_path(runner_head))
        and manifest.get("historical_c0_git_head") == HISTORICAL_C0_GIT_HEAD
        and manifest.get("historical_full_git_head")
        == REPAIRED_SCREEN_PROVENANCE_GIT_HEAD
        and manifest.get("control_runner_git_head") == runner_head
        and manifest.get("legacy_screen_gate_preserved") is True
        and manifest.get("launch_attempt_receipt") == str(ATTEMPT_RECEIPT),
        "Critic V4.0.3 confirmation manifest changed",
    )
    require_zero_protected_reads(
        manifest, label="Critic confirmation manifest"
    )
    configs: dict[int, Path] = {}
    for value in manifest["config_paths"]:
        path = Path(str(value))
        config = read_json(path)
        seed = int(config.get("training_seed", -1))
        require(
            config.get("schema_version")
            == "route_a_v3_route2_xeditcritic_v4_confirmation_runtime.v1"
            and config.get("run_stage") == "CONFIRMATION"
            and config.get("confirmation_runner_git_head") == runner_head
            and config.get("required_confirmation_run_ids")
            == ["v4_full", "c0_v4"]
            and config.get("screen_gate_path")
            == str(cross_root_gate_path(runner_head))
            and config.get("required_confirmation_seeds")
            == list(CONFIRMATION_SEEDS)
            and config.get("additional_seed_authorized") is False
            and config.get("development_test_outcomes_accessed") is False
            and config.get("new_final_evaluation_outcomes_accessed") is False,
            f"Critic V4.0.3 confirmation config changed: {path}",
        )
        configs[seed] = path
    require(
        set(configs) == set(CONFIRMATION_SEEDS),
        "Critic V4.0.3 confirmation config seeds changed",
    )
    return configs


def build_confirmation_authorization(
    gate: Mapping[str, Any],
    source_authorizations: Mapping[str, Mapping[str, Any]],
    runner_verification_receipt: Mapping[str, Any],
    training_semantics: Mapping[str, Any],
    *,
    runner_head: str,
    runner_verification_receipt_path_value: Path,
) -> dict[str, Any]:
    require(
        re.fullmatch(r"[0-9a-f]{40}", runner_head) is not None,
        "Critic confirmation runner Git HEAD is invalid",
    )
    validate_cross_root_gate(gate)
    require(
        control_runner_head_from_gate(gate) == runner_head,
        "confirmation runner HEAD differs from the controls runner HEAD",
    )
    validate_runner_verification_receipt(
        runner_verification_receipt,
        runner_head=runner_head,
        receipt_path=runner_verification_receipt_path_value,
    )
    require(
        training_semantics.get("runner_git_head") == runner_head
        and training_semantics.get("training_git_head") == runner_head
        and training_semantics.get(
            "historical_repaired_screen_provenance_git_head"
        )
        == REPAIRED_SCREEN_PROVENANCE_GIT_HEAD
        and training_semantics.get("historical_c0_git_head")
        == HISTORICAL_C0_GIT_HEAD
        and training_semantics.get(
            "previous_successor_semantic_baseline_git_head"
        )
        == TRAINING_SEMANTICS_PREVIOUS_SUCCESSOR_BASELINE_HEAD
        and training_semantics.get(
            "audited_successor_semantic_baseline_git_head"
        )
        == TRAINING_SEMANTICS_AUDITED_SUCCESSOR_BASELINE_HEAD
        and training_semantics.get(
            "audited_successor_semantic_baseline_audit"
        )
        == str(TRAINING_SEMANTICS_BASELINE_AUDIT)
        and training_semantics.get(
            "audited_successor_semantic_baseline_audit_status"
        )
        == "XEDITCRITIC_V403_CONFIRMATION_TRAINING_SEMANTICS_REAUDIT_V2_PASS"
        and training_semantics.get(
            "incremental_changed_training_semantic_paths_since_previous_successor"
        )
        == list(TRAINING_SEMANTICS_INCREMENTAL_CHANGED_PATHS)
        and training_semantics.get(
            "controls_oom_retry_technical_baseline_git_head"
        )
        == CONTROLS_OOM_RETRY_TECHNICAL_BASELINE_HEAD
        and training_semantics.get(
            "controls_oom_retry_technical_baseline_audit"
        )
        == str(CONTROLS_OOM_RETRY_TECHNICAL_BASELINE_AUDIT)
        and training_semantics.get(
            "controls_oom_retry_technical_baseline_audit_status"
        )
        == CONTROLS_OOM_RETRY_TECHNICAL_BASELINE_AUDIT_STATUS
        and training_semantics.get(
            "corrected_screen_confirmation_provenance_baseline_git_head"
        )
        == CORRECTED_SCREEN_CONFIRMATION_PROVENANCE_BASELINE_HEAD
        and training_semantics.get(
            "corrected_screen_confirmation_provenance_audit"
        )
        == str(CORRECTED_SCREEN_CONFIRMATION_PROVENANCE_AUDIT)
        and training_semantics.get(
            "corrected_screen_confirmation_provenance_audit_status"
        )
        == CORRECTED_SCREEN_CONFIRMATION_PROVENANCE_AUDIT_STATUS
        and training_semantics.get(
            "training_semantic_diff_paths_since_audited_successor_baseline"
        )
        == list(CONTROLS_OOM_RETRY_CHANGED_TRAINING_SEMANTIC_PATHS)
        and training_semantics.get(
            "training_semantics_unchanged_since_audited_successor_baseline"
        )
        is False
        and training_semantics.get(
            "training_semantic_diff_paths_since_controls_oom_retry_technical_baseline"
        )
        == list(CORRECTED_SCREEN_PROVENANCE_CHANGED_TRAINING_SEMANTIC_PATHS)
        and training_semantics.get(
            "training_semantics_unchanged_since_controls_oom_retry_technical_baseline"
        )
        is False
        and training_semantics.get(
            "training_semantic_diff_paths_since_corrected_screen_confirmation_provenance_baseline"
        )
        == []
        and training_semantics.get(
            "training_semantics_unchanged_since_corrected_screen_confirmation_provenance_baseline"
        )
        is True
        and training_semantics.get(
            "repaired_screen_is_historical_provenance_only"
        )
        is True,
        "Critic confirmation training-semantics baseline or runner roles changed",
    )
    require(
        tuple(source_authorizations) == ARM_ORDER,
        "Critic confirmation source authorization package changed",
    )
    for run_id in ARM_ORDER:
        authorization = source_authorizations[run_id]
        barriers = authorization.get("barriers")
        require(
            authorization.get("authorized_git_head")
            == expected_source_head(run_id, runner_head)
            and isinstance(barriers, Mapping)
            and all(barriers.get(key) is True for key in SOURCE_BARRIERS),
            f"Critic confirmation source barriers changed for {run_id}",
        )
        require_zero_protected_reads(
            authorization, label=f"{run_id} source authorization"
        )
    return {
        "schema_version": (
            "route_a_v3_route2_xeditcritic_v4_"
            "confirmation_launch_authorization.v1"
        ),
        "status": "XEDITCRITIC_V4_CONFIRMATION_LAUNCH_AUTHORIZED",
        "authorization_derivation": (
            "V403_FIXED_CROSS_ROOT_EIGHT_ARM_SCREEN_PASS"
        ),
        "authorized_git_head": runner_head,
        "authorized_seeds": list(CONFIRMATION_SEEDS),
        "authorized_run_ids": ["v4_full", "c0_v4"],
        "cross_root_screen_gate_path": str(cross_root_gate_path(runner_head)),
        "legacy_screen_gate_path": str(LEGACY_GATE),
        "legacy_screen_gate_preserved": True,
        "ordered_screen_run_ids": list(ARM_ORDER),
        "historical_c0_git_head": HISTORICAL_C0_GIT_HEAD,
        "historical_full_git_head": REPAIRED_SCREEN_PROVENANCE_GIT_HEAD,
        "control_runner_git_head": runner_head,
        "training_git_head": runner_head,
        "audited_successor_semantic_baseline_git_head": (
            TRAINING_SEMANTICS_AUDITED_SUCCESSOR_BASELINE_HEAD
        ),
        "audited_successor_semantic_baseline_audit": str(
            TRAINING_SEMANTICS_BASELINE_AUDIT
        ),
        **controls_oom_retry_technical_baseline_binding(),
        **corrected_screen_confirmation_provenance_binding(),
        "training_semantics": dict(training_semantics),
        "source_authorization_paths": {
            run_id: gate["cross_root_transition"]["arm_sources"][run_id][
                "launch_authorization_path"
            ]
            for run_id in ARM_ORDER
        },
        "runner_current_head_verification": {
            "receipt_path": str(runner_verification_receipt_path_value),
            "schema_version": runner_verification_receipt["schema_version"],
            "status": runner_verification_receipt["status"],
            "runner_git_head": runner_verification_receipt[
                "runner_git_head"
            ],
            "worktree_clean": runner_verification_receipt["worktree_clean"],
            "focused_tests": dict(
                runner_verification_receipt["focused_tests"]
            ),
            "v332_tests": dict(runner_verification_receipt["v332_tests"]),
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
        "barriers": {
            "screen_gate_passed": True,
            "a100_current_head_focused_tests_passed": (
                runner_verification_receipt["focused_tests"]["passed"]
            ),
            "a100_current_head_v332_tests_passed": (
                runner_verification_receipt["v332_tests"]["passed"]
            ),
            "bottom_six_cache_terminal_complete": True,
            "formal_parameter_preflight_passed": True,
            "formal_memory_preflight_passed": True,
            "cross_root_eight_arm_provenance_verified": True,
            "source_screen_barriers_all_eight_verified": True,
            "exact_runner_head_verified": True,
        },
        "scientific_thresholds_changed": False,
        "training_semantics_unchanged_since_audited_successor_baseline": False,
        "training_semantics_unchanged_since_controls_oom_retry_technical_baseline": (
            False
        ),
        "training_semantics_unchanged_since_corrected_screen_confirmation_provenance_baseline": (
            True
        ),
        "repaired_screen_is_historical_provenance_only": True,
        "additional_seed_authorized": False,
        "development_test_authorized": False,
        "guidance_authorized": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def materialize_authorization(
    authorization: Mapping[str, Any], authorization_root: Path
) -> Path:
    staging = authorization_root.with_name(authorization_root.name + ".partial")
    require(not authorization_root.exists(), f"confirmation authorization root exists: {authorization_root}")
    require(not staging.exists(), f"partial confirmation authorization root exists: {staging}")
    staging.mkdir(parents=True)
    path = staging / "critic.json"
    path.write_text(
        json.dumps(dict(authorization), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(staging, authorization_root)
    return authorization_root / "critic.json"


def gpu_memory_diagnostics() -> dict[int, dict[str, Any]]:
    result = command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    values: dict[int, dict[str, Any]] = {}
    for line in result.stdout.splitlines():
        index, name, free, total = (
            part.strip() for part in line.split(",", maxsplit=3)
        )
        values[int(index)] = {
            "name": name,
            "free_memory_mib": int(free),
            "total_memory_mib": int(total),
        }
    require(
        set(values).issuperset(PHYSICAL_GPUS),
        "physical GPU inventory 0-5 is incomplete",
    )
    return values


def cuda_bf16_probe(physical_gpu_index: int) -> dict[str, Any]:
    source = """
import json
import sys
import torch

index = int(sys.argv[1])
if not torch.cuda.is_available():
    raise RuntimeError("CUDA_UNAVAILABLE_CPU_FALLBACK_FORBIDDEN")
if index < 0 or index >= torch.cuda.device_count():
    raise RuntimeError("PHYSICAL_GPU_INDEX_UNAVAILABLE")
torch.cuda.set_device(index)
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("BF16_UNAVAILABLE_ON_SELECTED_GPU")
tensor = torch.ones((8,), device=f"cuda:{index}", dtype=torch.bfloat16)
if tensor.device.type != "cuda" or tensor.dtype != torch.bfloat16:
    raise RuntimeError("CUDA_BF16_PROBE_SILENT_CPU_FALLBACK")
print(json.dumps({
    "physical_gpu_index": index,
    "device_name": torch.cuda.get_device_name(index),
    "device_type": tensor.device.type,
    "dtype": str(tensor.dtype).replace("torch.", "").upper(),
    "cuda_available": True,
    "bf16_supported": True,
    "cpu_fallback_used": False,
}))
"""
    result = command([str(PYTHON), "-c", source, str(physical_gpu_index)])
    payload = json.loads(result.stdout)
    require(
        payload.get("physical_gpu_index") == physical_gpu_index
        and payload.get("device_type") == "cuda"
        and payload.get("dtype") == "BFLOAT16"
        and payload.get("cuda_available") is True
        and payload.get("bf16_supported") is True
        and payload.get("cpu_fallback_used") is False,
        f"GPU {physical_gpu_index} failed CUDA/BF16 identity probe",
    )
    return payload


def ensure_one_shot_targets_absent(
    *,
    protocol: Mapping[str, Any],
    authorization_root: Path,
    runtime_root: Path,
    log_root: Path,
) -> None:
    config_root = Path(str(protocol["runtime_config_root"]))
    run_root = Path(str(protocol["run_root"]))
    targets = (
        (ATTEMPT_RECEIPT, "canonical confirmation attempt receipt"),
        (config_root, "confirmation config root"),
        (config_root.with_name(config_root.name + ".partial"), "partial confirmation config root"),
        (run_root, "confirmation run root"),
        (authorization_root, "confirmation authorization root"),
        (authorization_root.with_name(authorization_root.name + ".partial"), "partial confirmation authorization root"),
        (runtime_root, "confirmation training runtime root"),
        (log_root, "confirmation training log root"),
    )
    for path, label in targets:
        require(not path.exists(), f"{label} already exists: {path}")


def build_schedule(
    configs: Mapping[int, Path],
    authorization_path: Path,
    manifest_path: Path,
    diagnostics: Mapping[int, Mapping[str, Any]],
    cuda_probes: Mapping[int, Mapping[str, Any]],
    *,
    runner_head: str,
    runtime_manifest: Path,
    log_root: Path,
    preflight_peak_allocated_gib: float,
) -> dict[str, Any]:
    require(
        set(configs) == set(CONFIRMATION_SEEDS),
        "Critic confirmation schedule config seeds changed",
    )
    require(
        all(gpu in diagnostics and gpu in cuda_probes for gpu in PHYSICAL_GPUS),
        "Critic confirmation GPU diagnostics or probes are incomplete",
    )
    assignments = (
        (0, CONFIRMATION_SEEDS[0], "v4_full"),
        (1, CONFIRMATION_SEEDS[0], "c0_v4"),
        (2, CONFIRMATION_SEEDS[1], "v4_full"),
        (3, CONFIRMATION_SEEDS[1], "c0_v4"),
        (4, CONFIRMATION_SEEDS[2], "v4_full"),
        (5, CONFIRMATION_SEEDS[2], "c0_v4"),
    )
    queues: list[dict[str, Any]] = []
    for gpu, seed, run_id in assignments:
        config_path = configs[seed]
        config = read_json(config_path)
        output = Path(str(config["output_root"])) / run_id
        queues.append(
            {
                "physical_gpu_index": gpu,
                "jobs": [
                    {
                        "job_key": f"critic:{seed}:{run_id}",
                        "component": "critic",
                        "training_seed": seed,
                        "run_id": run_id,
                        "output_directory": str(output),
                        "log_path": str(
                            log_root / f"critic_{seed}_{run_id}.log"
                        ),
                        "command": [
                            str(PYTHON),
                            str(TRAINER),
                            "--config",
                            str(config_path),
                            "--run-id",
                            run_id,
                            "--physical-gpu-index",
                            str(gpu),
                            "--launch-authorization",
                            str(authorization_path),
                        ],
                    }
                ],
            }
        )
    return {
        "schema_version": (
            "route_a_v3_route2_xedit_v4_confirmation_training_schedule.v1"
        ),
        "status": "FROZEN_CONFIRMATION_TRAINING_SCHEDULE",
        "git_head": runner_head,
        "experiment_head": runner_head,
        "training_git_head": runner_head,
        "audited_successor_semantic_baseline_git_head": (
            TRAINING_SEMANTICS_AUDITED_SUCCESSOR_BASELINE_HEAD
        ),
        "audited_successor_semantic_baseline_audit": str(
            TRAINING_SEMANTICS_BASELINE_AUDIT
        ),
        **controls_oom_retry_technical_baseline_binding(),
        **corrected_screen_confirmation_provenance_binding(),
        "worktree": str(WORKTREE),
        "runtime_manifest": str(runtime_manifest),
        "eligible_components": ["critic"],
        "config_manifest": str(manifest_path),
        "confirmation_authorization": str(authorization_path),
        "cross_root_screen_gate": str(cross_root_gate_path(runner_head)),
        "legacy_screen_gate": str(LEGACY_GATE),
        "legacy_screen_gate_preserved": True,
        "launch_attempt_receipt": str(ATTEMPT_RECEIPT),
        "historical_c0_git_head": HISTORICAL_C0_GIT_HEAD,
        "historical_full_git_head": REPAIRED_SCREEN_PROVENANCE_GIT_HEAD,
        "control_runner_git_head": runner_head,
        "required_seeds": list(CONFIRMATION_SEEDS),
        "required_run_ids": ["v4_full", "c0_v4"],
        "gpu_memory_diagnostics_before_launch": {
            str(gpu): dict(diagnostics[gpu]) for gpu in PHYSICAL_GPUS
        },
        "cuda_bf16_probes": {
            str(gpu): dict(cuda_probes[gpu]) for gpu in PHYSICAL_GPUS
        },
        "preflight_peak_allocated_gib_diagnostic": (
            preflight_peak_allocated_gib
        ),
        "free_memory_gate_applied": False,
        "gpu_sort_applied": False,
        "gpu_selection_policy": "FROZEN_PHYSICAL_GPU_0_TO_5_ASSIGNMENT",
        "gpu_queues": queues,
        "active_performance_output_read": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def run(current_head: str) -> dict[str, Any]:
    require(
        re.fullmatch(r"[0-9a-f]{40}", current_head) is not None,
        "expected current Git HEAD is invalid",
    )
    cross_root_gate = cross_root_gate_path(current_head)
    require(cross_root_gate != LEGACY_GATE, "cross-root gate aliases legacy gate")
    for path, label in (
        (PYTHON, "formal Python"),
        (BASE_CONFIG, "Critic screen config"),
        (BASE_PROTOCOL, "Critic confirmation protocol"),
        (TRAINER, "Critic trainer"),
        (SCHEDULER, "confirmation scheduler"),
        (cross_root_gate, "fixed cross-root PASS gate"),
    ):
        require(path.is_file(), f"{label} is absent: {path}")
    require(
        not os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES remapping is forbidden",
    )
    require(
        command(["git", "rev-parse", "HEAD"]).stdout.strip()
        == current_head,
        "A100 worktree is not at expected runner HEAD",
    )
    require(
        not command(["git", "status", "--porcelain"]).stdout.strip(),
        "A100 runner worktree is dirty",
    )

    gate = read_json(cross_root_gate)
    runner_receipt_path = runner_verification_receipt_path(current_head)
    require(
        runner_receipt_path.is_file(),
        "exact-runner-HEAD successor verification receipt is absent: "
        f"{runner_receipt_path}",
    )
    runner_receipt = read_json(runner_receipt_path)
    validate_runner_verification_receipt(
        runner_receipt,
        runner_head=current_head,
        receipt_path=runner_receipt_path,
    )
    training_semantics = validate_runner_training_semantics(current_head)
    validate_cross_root_gate(gate)
    source_authorizations = load_and_validate_source_authorizations(gate)
    base_config = read_json(BASE_CONFIG)
    base_protocol = read_json(BASE_PROTOCOL)
    preflight = read_json(Path(str(base_config["preflight_output"])))
    require(
        preflight.get("status") == "XEDITCRITIC_V4_PREFLIGHT_PASS"
        and preflight.get("passed") is True,
        "frozen Critic V4 preflight is not PASS",
    )
    require_zero_protected_reads(preflight, label="frozen Critic preflight")
    protocol, configs = build_confirmation_configs(
        base_config,
        base_protocol,
        gate,
        runner_head=current_head,
    )
    authorization = build_confirmation_authorization(
        gate,
        source_authorizations,
        runner_receipt,
        training_semantics,
        runner_head=current_head,
        runner_verification_receipt_path_value=runner_receipt_path,
    )

    authorization_root = (
        ROOT / f"authorizations/xedit_v4/confirmation_{current_head}"
    )
    runtime_root = (
        ROOT / f"experiments/xedit_v4/confirmation_training_{current_head}"
    )
    log_root = ROOT / f"logs/xedit_v4/confirmation_training_{current_head}"
    ensure_one_shot_targets_absent(
        protocol=protocol,
        authorization_root=authorization_root,
        runtime_root=runtime_root,
        log_root=log_root,
    )

    try:
        diagnostics = gpu_memory_diagnostics()
        cuda_probes = {
            gpu: cuda_bf16_probe(gpu) for gpu in PHYSICAL_GPUS
        }
    except Exception as error:
        write_new_atomic(
            ATTEMPT_RECEIPT,
            {
                "schema_version": (
                    "route_a_v3_route2_xeditcritic_v403_"
                    "confirmation_attempt.v1"
                ),
                "status": "STOPPED_BEFORE_CONFIRMATION_LAUNCH_CUDA_FAILURE",
                "runner_git_head": current_head,
                "cross_root_screen_gate": str(cross_root_gate),
                "training_semantics": training_semantics,
                "runner_verification_receipt": str(runner_receipt_path),
                "error_type": type(error).__name__,
                "error": str(error),
                "cpu_fallback_used": False,
                "training_jobs_started": 0,
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )
        raise

    write_new_atomic(
        ATTEMPT_RECEIPT,
        {
            "schema_version": (
                "route_a_v3_route2_xeditcritic_v403_confirmation_attempt.v1"
            ),
            "status": "XEDITCRITIC_V403_CONFIRMATION_ATTEMPT_CONSUMED",
            "runner_git_head": current_head,
            "cross_root_screen_gate": str(cross_root_gate),
            "legacy_screen_gate": str(LEGACY_GATE),
            "legacy_screen_gate_preserved": True,
            "historical_c0_git_head": HISTORICAL_C0_GIT_HEAD,
            "historical_full_git_head": REPAIRED_SCREEN_PROVENANCE_GIT_HEAD,
            "control_runner_git_head": current_head,
            "training_git_head": current_head,
            "audited_successor_semantic_baseline_git_head": (
                TRAINING_SEMANTICS_AUDITED_SUCCESSOR_BASELINE_HEAD
            ),
            "training_semantics": training_semantics,
            "runner_verification_receipt": str(runner_receipt_path),
            "required_seeds": list(CONFIRMATION_SEEDS),
            "required_run_ids": ["v4_full", "c0_v4"],
            "physical_gpu_assignment": list(PHYSICAL_GPUS),
            "gpu_memory_diagnostics": {
                str(gpu): dict(diagnostics[gpu]) for gpu in PHYSICAL_GPUS
            },
            "cuda_bf16_probes": {
                str(gpu): dict(cuda_probes[gpu]) for gpu in PHYSICAL_GPUS
            },
            "free_memory_gate_applied": False,
            "cpu_fallback_used": False,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )

    manifest = materialize_config_package(
        configs, protocol, runner_head=current_head
    )
    manifest_path = Path(str(protocol["runtime_config_root"])) / "manifest.json"
    require(read_json(manifest_path) == manifest, "materialized Critic manifest changed")
    config_paths = validate_manifest(manifest, runner_head=current_head)
    authorization_path = materialize_authorization(
        authorization, authorization_root
    )
    require(
        read_json(authorization_path) == authorization,
        "materialized Critic confirmation authorization changed",
    )

    runtime_root.mkdir(parents=True)
    log_root.mkdir(parents=True)
    runtime_manifest = runtime_root / "runtime.json"
    schedule_path = runtime_root / "schedule.json"
    schedule = build_schedule(
        config_paths,
        authorization_path,
        manifest_path,
        diagnostics,
        cuda_probes,
        runner_head=current_head,
        runtime_manifest=runtime_manifest,
        log_root=log_root,
        preflight_peak_allocated_gib=float(
            preflight["selected_peak_allocated_gib"]
        ),
    )
    write_new_atomic(schedule_path, schedule)
    scheduler_log = log_root / "scheduler.log"
    scheduler_command = [
        str(PYTHON),
        str(SCHEDULER),
        "--schedule",
        str(schedule_path),
    ]
    process = spawn_scheduler_with_failure_evidence(
        failure_path=runtime_root / "scheduler_launch.failed.json",
        expected_head=current_head,
        command_line=scheduler_command,
        schedule_path=schedule_path,
        runtime_path=runtime_manifest,
        scheduler_log=scheduler_log,
        created_artifacts={
            "launch_attempt_receipt": ATTEMPT_RECEIPT,
            "config_manifest": manifest_path,
            "confirmation_authorization": authorization_path,
            "schedule": schedule_path,
            "scheduler_log": scheduler_log,
        },
    )
    launch = {
        "schema_version": (
            "route_a_v3_route2_xeditcritic_v403_"
            "cross_root_confirmation_launch.v1"
        ),
        "status": "XEDITCRITIC_V403_CONFIRMATION_SCHEDULER_LAUNCHED",
        "runner_git_head": current_head,
        "training_git_head": current_head,
        "audited_successor_semantic_baseline_git_head": (
            TRAINING_SEMANTICS_AUDITED_SUCCESSOR_BASELINE_HEAD
        ),
        "audited_successor_semantic_baseline_audit": str(
            TRAINING_SEMANTICS_BASELINE_AUDIT
        ),
        **controls_oom_retry_technical_baseline_binding(),
        **corrected_screen_confirmation_provenance_binding(),
        "historical_c0_git_head": HISTORICAL_C0_GIT_HEAD,
        "historical_full_git_head": REPAIRED_SCREEN_PROVENANCE_GIT_HEAD,
        "control_runner_git_head": current_head,
        "scheduler_pid": process.pid,
        "schedule_path": str(schedule_path),
        "runtime_manifest": str(runtime_manifest),
        "scheduler_log": str(scheduler_log),
        "config_manifest": str(manifest_path),
        "confirmation_authorization": str(authorization_path),
        "cross_root_screen_gate": str(cross_root_gate),
        "launch_attempt_receipt": str(ATTEMPT_RECEIPT),
        "runner_verification_receipt": str(runner_receipt_path),
        "required_seeds": list(CONFIRMATION_SEEDS),
        "required_run_ids": ["v4_full", "c0_v4"],
        "selected_physical_gpus": list(PHYSICAL_GPUS),
        "free_memory_gate_applied": False,
        "cpu_fallback_used": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_new_atomic(runtime_root / "launch.json", launch)
    return launch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(arguments.expected_head), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
