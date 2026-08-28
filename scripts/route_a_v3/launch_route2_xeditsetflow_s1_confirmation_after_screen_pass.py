#!/usr/bin/env python3
"""Launch the fresh three-seed SetFlow S1 confirmation after exact screen PASS."""

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

from core.route2_xeditsetflow_confirmation_s1 import (
    CONFIRMATION_RUN_ID,
    CONFIRMATION_RUNTIME_SCHEMA,
    CONFIRMATION_RUNTIME_STATUS,
    CONFIRMATION_SEEDS,
    OBJECTIVE_IDENTITY,
    OBJECTIVE_WEIGHT,
    SCREEN_HEAD,
    build_confirmation_configs_s1,
    materialize_confirmation_configs_s1,
    validate_screen_pass_barrier_s1,
)
from scripts.route_a_v3.authorize_route2_xeditsetflow_v403_recovered_confirmation import (
    require_runner_verification_receipt_v403,
)
from scripts.route_a_v3.launch_route2_xeditcritic_v403_confirmation_after_cross_root_screen import (
    validate_runner_verification_receipt as validate_shared_runner_verification_receipt,
)
from scripts.route_a_v3.launch_route2_xeditsetflow_s1_screen_after_v403_terminal import (
    CONFIG as SCREEN_CONFIG,
    GPU_INVENTORY_COMMAND,
    XEditSetFlowS1GpuError,
    cuda_bf16_probe,
    expected_receipt_paths,
    gpu_diagnostics,
    validate_repo_fact_audits,
)


PYTHON = Path("/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10")
BRANCH = "route-a-v3-v403-no-vram-gate-20260827"
PROTOCOL = (
    WORKTREE
    / "configs/route_a_v3_route2_xeditsetflow_v4_s1_confirmation_protocol_v1.json"
)
CORRECTED_SCREEN_PROVENANCE_PREVIOUS_CODE_BASELINE_HEAD = (
    "26fdbcb38090cf98e68425bebabd084a374447c4"
)
CORRECTED_SCREEN_PROVENANCE_BASELINE_HEAD = (
    "19bc3ed4dd3ee5647e3d3304c10dc9914f885e68"
)
CRITIC_CONTROLS_OOM_RETRY_TECHNICAL_BASELINE_HEAD = (
    "793eedfb4b84e8c0dbd5a30bdf79c8923ddf8110"
)
INVALIDATED_SCREEN_HEAD = "930fccf468c14378b3dd2fd2caf3aaa3cc2eb3c8"
CORRECTED_SCREEN_PROVENANCE_AUDIT = (
    WORKTREE
    / "audits/route_a_v3_route2_xeditsetflow_s1_corrected_screen_"
    "confirmation_provenance_19bc3ed4dd3ee5647e3d3304c10dc9914f885e68.json"
)
CORRECTED_SCREEN_PROVENANCE_AUDIT_SCHEMA = (
    "route_a_v3_route2_xeditsetflow_s1_corrected_screen_"
    "confirmation_provenance.v1"
)
CORRECTED_SCREEN_PROVENANCE_AUDIT_STATUS = (
    "XEDITSETFLOW_V4_S1_CORRECTED_SCREEN_CONFIRMATION_PROVENANCE_PASS"
)
CORRECTED_SCREEN_PROVENANCE_PRODUCTION_PATHSPEC = (
    "configs",
    "core",
    "scripts/route_a_v3/train_route2_xeditsetflow_s1.py",
)
CORRECTED_SCREEN_PROVENANCE_CHANGED_PRODUCTION_PATHS = (
    "configs/route_a_v3_route2_xeditsetflow_v4_s1_confirmation_protocol_v1.json",
    "core/route2_xeditsetflow_confirmation_s1.py",
    "core/route2_xeditsetflow_gate_s1.py",
    "scripts/route_a_v3/train_route2_xeditsetflow_s1.py",
)
CORRECTED_SCREEN_PROVENANCE_PATH_CLASSIFICATION = {
    CORRECTED_SCREEN_PROVENANCE_CHANGED_PRODUCTION_PATHS[0]: (
        "CORRECTED_SCREEN_PROTOCOL_PROVENANCE_SIX_FIELDS_ONLY"
    ),
    CORRECTED_SCREEN_PROVENANCE_CHANGED_PRODUCTION_PATHS[1]: (
        "CORRECTED_SCREEN_CONFIRMATION_PROVENANCE_BINDING_ONLY"
    ),
    CORRECTED_SCREEN_PROVENANCE_CHANGED_PRODUCTION_PATHS[2]: (
        "CORRECTED_SCREEN_CONFIRMATION_GATE_LINEAGE_ONLY"
    ),
    CORRECTED_SCREEN_PROVENANCE_CHANGED_PRODUCTION_PATHS[3]: (
        "CORRECTED_SCREEN_CONFIRMATION_AUTHORIZATION_LINEAGE_ONLY"
    ),
}
CORRECTED_SCREEN_PROVENANCE_PROTOCOL_FIELDS = (
    "authorization_path",
    "runtime_config_path",
    "runtime_path",
    "schedule_path",
    "screen_gate_path",
    "screen_runner_git_head",
)
CORRECTED_SCREEN_PROVENANCE_CRITIC_PATHS = (
    "core/route2_xeditsetflow_confirmation_s1.py",
    "core/route2_xeditsetflow_gate_s1.py",
)
CORRECTED_SCREEN_PROVENANCE_CRITIC_PATH_CLASSIFICATION = {
    path: "SETFLOW_PROVENANCE_ONLY_CRITIC_OBJECTIVE_NEUTRAL"
    for path in CORRECTED_SCREEN_PROVENANCE_CRITIC_PATHS
}
CORRECTED_SCREEN_PROVENANCE_UNCHANGED_FLAGS = (
    "setflow_model_architecture_forward_loss_changed",
    "setflow_trainer_parameter_update_semantics_changed",
    "objective_identity_changed",
    "objective_weight_changed",
    "screen_seed_changed",
    "confirmation_seed_cohort_changed",
    "pass_count_changed",
    "physical_or_effective_batch_changed",
    "checkpoint_passes_changed",
    "paired_bootstrap_changed",
    "scientific_thresholds_changed",
    "gpu_scope_or_fixed_mapping_changed",
    "free_or_estimated_memory_gate_added",
    "free_or_estimated_memory_sorting_added",
    "package_failure_policy_changed",
    "protected_outcome_policy_changed",
)
TRAINER = WORKTREE / "scripts/route_a_v3/train_route2_xeditsetflow_s1.py"
SCHEDULER = (
    WORKTREE
    / "scripts/route_a_v3/run_route2_xedit_v4_confirmation_training_scheduler.py"
)

S1_FOCUSED_TEST_MARKERS = (
    "test_transition_record_route2_xeditsetflow_s1_930_terminal_invalidation.py",
    "test_route2_xeditsetflow_confirmation_s1.py",
    "test_train_route2_xeditsetflow_s1.py",
    "test_validate_route2_xeditsetflow_s1_checkpoint.py",
    "test_route2_xeditsetflow_gate_s1.py",
    "test_run_route2_xedit_v4_confirmation_training_scheduler.py",
    "test_run_route2_xedit_v4_confirmation_posttraining_scheduler.py",
    "test_launch_route2_xeditsetflow_s1_confirmation_after_screen_pass.py",
    "test_launch_route2_xeditsetflow_s1_confirmation_posttraining.py",
    "test_adjudicate_route2_xeditsetflow_s1_confirmation.py",
)


def require_seed_valid_screen_head_s1(
    repair_audit: Mapping[str, Any], *, screen_head: str
) -> None:
    affected = repair_audit.get("affected_family")
    defect = repair_audit.get("defect")
    require(
        repair_audit.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_s1_seed_initialization_repair.v1"
        and repair_audit.get("status")
        == "XEDITSETFLOW_V4_S1_SEED_INITIALIZATION_REPAIR_FROZEN_BEFORE_INDEPENDENT_RETRY"
        and isinstance(affected, Mapping)
        and isinstance(defect, Mapping),
        "S1 seed-initialization repair audit is absent",
    )
    require(
        not (
            affected.get("runner_git_head") == screen_head
            and defect.get("affected_family_can_authorize_successor") is False
        ),
        "S1 screen HEAD has uncontrolled parameter initialization and cannot authorize confirmation",
    )


class XEditSetFlowS1ConfirmationLaunchError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditSetFlowS1ConfirmationLaunchError(message)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


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


def command(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(arguments),
        cwd=WORKTREE,
        text=True,
        capture_output=True,
        check=True,
    )


def protected_reads_zero(payload: Mapping[str, Any], *, label: str) -> None:
    require(
        int(payload.get("development_test_outcome_reads", -1)) == 0
        and int(payload.get("new_final_evaluation_outcome_reads", -1)) == 0,
        f"SetFlow S1 {label} reports a protected outcome read",
    )


def validate_corrected_screen_provenance_audit_s1(
    audit: Mapping[str, Any],
) -> None:
    replacement = audit.get("protocol_screen_provenance_replacement")
    setflow_consumer = audit.get("setflow_confirmation_consumer_review")
    critic_consumer = audit.get("critic_confirmation_consumer_review")
    require(
        audit.get("schema_version")
        == CORRECTED_SCREEN_PROVENANCE_AUDIT_SCHEMA
        and audit.get("status") == CORRECTED_SCREEN_PROVENANCE_AUDIT_STATUS
        and audit.get("previous_code_baseline_git_head")
        == CORRECTED_SCREEN_PROVENANCE_PREVIOUS_CODE_BASELINE_HEAD
        and audit.get(
            "corrected_screen_confirmation_provenance_baseline_git_head"
        )
        == CORRECTED_SCREEN_PROVENANCE_BASELINE_HEAD
        and audit.get("critic_controls_oom_retry_technical_baseline_git_head")
        == CRITIC_CONTROLS_OOM_RETRY_TECHNICAL_BASELINE_HEAD
        and audit.get("invalidated_screen_runner_git_head")
        == INVALIDATED_SCREEN_HEAD
        and audit.get("corrected_screen_runner_git_head") == SCREEN_HEAD
        and audit.get("production_pathspec")
        == list(CORRECTED_SCREEN_PROVENANCE_PRODUCTION_PATHSPEC)
        and tuple(
            audit.get("changed_production_paths_from_previous_code_baseline", [])
        )
        == CORRECTED_SCREEN_PROVENANCE_CHANGED_PRODUCTION_PATHS
        and audit.get("changed_production_paths_from_previous_code_baseline")
        == sorted(
            audit.get("changed_production_paths_from_previous_code_baseline", [])
        )
        and audit.get("production_path_classification")
        == CORRECTED_SCREEN_PROVENANCE_PATH_CLASSIFICATION
        and tuple(audit.get("protocol_changed_screen_provenance_fields", []))
        == CORRECTED_SCREEN_PROVENANCE_PROTOCOL_FIELDS
        and isinstance(replacement, Mapping)
        and replacement.get("from_screen_runner_git_head")
        == INVALIDATED_SCREEN_HEAD
        and replacement.get("to_screen_runner_git_head") == SCREEN_HEAD
        and replacement.get("all_six_fields_replace_only_the_family_head")
        is True
        and replacement.get("all_other_protocol_fields_changed") is False
        and tuple(
            audit.get(
                "critic_training_semantic_paths_from_z1_to_corrected_screen_baseline",
                [],
            )
        )
        == CORRECTED_SCREEN_PROVENANCE_CRITIC_PATHS
        and audit.get("critic_path_classification")
        == CORRECTED_SCREEN_PROVENANCE_CRITIC_PATH_CLASSIFICATION
        and all(
            audit.get(field) is False
            for field in CORRECTED_SCREEN_PROVENANCE_UNCHANGED_FLAGS
        )
        and audit.get("old_930_terminal_invalidation_preserved") is True
        and audit.get("old_930_seed_initialization_repair_preserved") is True
        and audit.get("invalidated_930_family_can_authorize_successor") is False
        and audit.get(
            "current_confirmation_runner_may_differ_from_corrected_screen_runner"
        )
        is True
        and audit.get("successor_authorized_by_this_audit") is False
        and audit.get("confirmation_authorized_by_this_audit") is False
        and isinstance(setflow_consumer, Mapping)
        and setflow_consumer.get("path")
        == (
            "scripts/route_a_v3/"
            "launch_route2_xeditsetflow_s1_confirmation_after_screen_pass.py"
        )
        and setflow_consumer.get("included_in_changed_production_paths") is False
        and setflow_consumer.get(
            "expected_changed_production_paths_from_baseline_to_runner"
        )
        == []
        and isinstance(critic_consumer, Mapping)
        and critic_consumer.get("path")
        == (
            "scripts/route_a_v3/"
            "launch_route2_xeditcritic_v403_confirmation_after_cross_root_screen.py"
        )
        and critic_consumer.get("included_in_critic_training_semantic_pathspec")
        is False
        and critic_consumer.get(
            "expected_critic_training_semantic_paths_from_baseline_to_runner"
        )
        == []
        and audit.get("model_result_claimed") is False
        and audit.get("submission_ready") is False,
        "S1 corrected-screen confirmation provenance audit is absent or invalid",
    )
    protected_reads_zero(
        audit, label="corrected-screen confirmation provenance audit"
    )


def corrected_screen_provenance_baseline_binding_s1() -> dict[str, str]:
    return {
        "corrected_screen_confirmation_provenance_baseline_git_head": (
            CORRECTED_SCREEN_PROVENANCE_BASELINE_HEAD
        ),
        "corrected_screen_confirmation_provenance_audit": str(
            CORRECTED_SCREEN_PROVENANCE_AUDIT
        ),
        "corrected_screen_confirmation_provenance_audit_status": (
            CORRECTED_SCREEN_PROVENANCE_AUDIT_STATUS
        ),
    }


def validate_corrected_screen_provenance_baseline_s1(
    current_head: str,
) -> dict[str, Any]:
    audit = read_json(CORRECTED_SCREEN_PROVENANCE_AUDIT)
    validate_corrected_screen_provenance_audit_s1(audit)
    baseline_changed = command(
        [
            "git",
            "diff",
            "--name-only",
            CORRECTED_SCREEN_PROVENANCE_PREVIOUS_CODE_BASELINE_HEAD,
            CORRECTED_SCREEN_PROVENANCE_BASELINE_HEAD,
            "--",
            *CORRECTED_SCREEN_PROVENANCE_PRODUCTION_PATHSPEC,
        ]
    ).stdout.splitlines()
    require(
        tuple(baseline_changed)
        == CORRECTED_SCREEN_PROVENANCE_CHANGED_PRODUCTION_PATHS,
        "S1 corrected-screen provenance production paths differ from the exact "
        "Git diff: " + ", ".join(baseline_changed),
    )
    changed = command(
        [
            "git",
            "diff",
            "--name-only",
            CORRECTED_SCREEN_PROVENANCE_BASELINE_HEAD,
            current_head,
            "--",
            *CORRECTED_SCREEN_PROVENANCE_PRODUCTION_PATHSPEC,
        ]
    ).stdout.splitlines()
    require(
        not changed,
        "S1 corrected-screen provenance changed after its code baseline: "
        + ", ".join(changed),
    )
    return {
        **corrected_screen_provenance_baseline_binding_s1(),
        "changed_production_paths_from_previous_code_baseline": baseline_changed,
        "changed_production_paths_since_corrected_screen_provenance_baseline": (
            changed
        ),
        "corrected_screen_provenance_unchanged_since_baseline": True,
    }


def format_output_path(
    protocol: Mapping[str, Any], key: str, runner_head: str
) -> Path:
    outputs = protocol.get("runner_outputs")
    require(isinstance(outputs, Mapping), "S1 confirmation output contract is absent")
    require(key in outputs, f"S1 confirmation output path is absent: {key}")
    return Path(str(outputs[key]).format(runner_git_head=runner_head))


def validate_s1_receipt_marker_coverage(
    receipt: Mapping[str, Any], *, label: str
) -> None:
    focused = receipt.get("focused_tests")
    require(isinstance(focused, Mapping), f"{label} receipt lacks focused tests")
    commands = focused.get("command")
    require(
        isinstance(commands, list)
        and len(commands) == 8
        and all(isinstance(value, str) and value for value in commands),
        f"{label} receipt does not contain eight focused command groups",
    )
    missing = [
        marker
        for marker in S1_FOCUSED_TEST_MARKERS
        if not any(marker in value for value in commands)
    ]
    require(
        not missing,
        f"{label} receipt lacks S1-specific focused coverage: {missing}",
    )


def consume_confirmation_receipts_s1(
    runner_head: str, shared_path: Path, setflow_path: Path
) -> dict[str, Any]:
    expected_shared, expected_setflow = expected_receipt_paths(runner_head)
    require(shared_path == expected_shared, "shared receipt path is not canonical")
    require(setflow_path == expected_setflow, "SetFlow receipt path is not canonical")
    shared = read_json(shared_path)
    setflow = read_json(setflow_path)
    validate_shared_runner_verification_receipt(
        shared, runner_head=runner_head, receipt_path=shared_path
    )
    require_runner_verification_receipt_v403(
        setflow, current_runner_head=runner_head
    )
    validate_s1_receipt_marker_coverage(shared, label="shared")
    validate_s1_receipt_marker_coverage(setflow, label="SetFlow")
    return {
        "shared": {
            "path": str(shared_path),
            "schema_version": shared["schema_version"],
            "status": shared["status"],
            "runner_git_head": shared["runner_git_head"],
            "s1_marker_coverage": list(S1_FOCUSED_TEST_MARKERS),
        },
        "setflow": {
            "path": str(setflow_path),
            "schema_version": setflow["schema_version"],
            "status": setflow["status"],
            "runner_git_head": setflow["runner_git_head"],
            "s1_marker_coverage": list(S1_FOCUSED_TEST_MARKERS),
        },
    }


def validate_screen_runtime_terminal_s1(
    schedule: Mapping[str, Any],
    runtime: Mapping[str, Any],
    gate: Mapping[str, Any],
    *,
    runtime_path: Path,
    gate_path: Path,
) -> None:
    require(
        Path(str(schedule.get("runtime_manifest"))) == runtime_path,
        "S1 screen runtime path is not bound by the frozen schedule",
    )
    adjudication_spec = schedule.get("adjudication")
    require(
        isinstance(adjudication_spec, Mapping)
        and Path(str(adjudication_spec.get("gate_path"))) == gate_path,
        "S1 screen gate path is not bound by the frozen schedule",
    )
    require(
        runtime.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_s1_screen_runtime.v1"
        and runtime.get("status")
        == "XEDITSETFLOW_V4_S1_SCREEN_AND_GATE_TERMINAL"
        and runtime.get("git_head") == SCREEN_HEAD
        and runtime.get("objective_identity") == OBJECTIVE_IDENTITY
        and float(
            runtime.get("cross_state_candidate_mode_responsibility_weight", -1.0)
        )
        == OBJECTIVE_WEIGHT
        and runtime.get("first_terminal_failure") is None
        and runtime.get("free_memory_gate_applied") is False
        and runtime.get("active_performance_output_read") is False,
        "S1 screen runtime is not the exact terminal successful package",
    )
    training = runtime.get("training_jobs")
    validation = runtime.get("validation_jobs")
    require(
        isinstance(training, Mapping)
        and len(training) == 2
        and isinstance(validation, Mapping)
        and len(validation) == 8,
        "S1 screen runtime job inventory changed",
    )
    for label, rows in (("training", training), ("Validation", validation)):
        require(
            all(
                isinstance(row, Mapping)
                and row.get("status") == "TERMINAL_COMPLETE"
                and row.get("terminal_artifact_kind") == "SUMMARY"
                and int(row.get("return_code", -1)) == 0
                for row in rows.values()
            ),
            f"S1 screen {label} package is not exact zero-exit SUMMARY terminal",
        )
    schedule_groups = (
        (schedule.get("training_queues"), training, "training"),
        (schedule.get("validation_queues"), validation, "Validation"),
    )
    for queues, runtime_rows, label in schedule_groups:
        require(isinstance(queues, list), f"S1 screen {label} queues are absent")
        jobs = [job for queue in queues for job in queue.get("jobs", [])]
        require(
            len(jobs) == len(runtime_rows)
            and {str(job.get("job_key")) for job in jobs} == set(runtime_rows),
            f"S1 screen {label} schedule/runtime inventory differs",
        )
        for job in jobs:
            key = str(job.get("job_key"))
            summary = Path(str(job.get("terminal_summary")))
            failure = Path(str(job.get("terminal_failure")))
            runtime_row = runtime_rows[key]
            require(
                Path(str(runtime_row.get("terminal_summary"))) == summary
                and Path(str(runtime_row.get("terminal_failure"))) == failure
                and runtime_row.get("run_id") == job.get("run_id")
                and int(runtime_row.get("physical_gpu_index", -1))
                == int(job.get("physical_gpu_index", -2))
                and (
                    "checkpoint_pass" not in job
                    or int(runtime_row.get("checkpoint_pass", -1))
                    == int(job.get("checkpoint_pass", -2))
                )
                and summary.is_file()
                and not failure.exists()
                and not summary.with_suffix(summary.suffix + ".partial").exists()
                and not failure.with_suffix(failure.suffix + ".partial").exists(),
                f"S1 screen {label} job is not uniquely SUMMARY-terminal: {key}",
            )
    adjudication = runtime.get("adjudication")
    adjudication_failure = Path(str(adjudication_spec.get("failure_path")))
    require(
        isinstance(adjudication, Mapping)
        and adjudication.get("status") == "TERMINAL_COMPLETE"
        and adjudication.get("terminal_artifact_kind") == "GATE"
        and int(adjudication.get("return_code", -1)) == 0
        and adjudication.get("gate_present") is True
        and adjudication.get("failure_present") is False,
        "S1 screen adjudication is not exact terminal",
    )
    require(
        Path(str(adjudication.get("gate_path"))) == gate_path
        and Path(str(adjudication.get("failure_path"))) == adjudication_failure
        and gate_path.is_file()
        and not adjudication_failure.exists()
        and not gate_path.with_suffix(gate_path.suffix + ".partial").exists()
        and not adjudication_failure.with_suffix(
            adjudication_failure.suffix + ".partial"
        ).exists(),
        "S1 screen adjudication terminal paths are not uniquely bound",
    )
    require(
        gate.get("status") == "XEDITSETFLOW_V4_S1_SCREEN_PASS",
        "S1 screen scientific gate is not PASS",
    )
    protected_reads_zero(runtime, label="screen runtime")
    protected_reads_zero(gate, label="screen gate")


def validate_screen_bundle_s1(
    base: Mapping[str, Any],
    protocol: Mapping[str, Any],
    screen_schedule: Mapping[str, Any],
    screen_runtime: Mapping[str, Any],
    screen_runtime_config: Mapping[str, Any],
    screen_authorization: Mapping[str, Any],
    screen_gate: Mapping[str, Any],
    *,
    screen_schedule_path: Path,
    screen_runtime_path: Path,
    screen_runtime_config_path: Path,
    screen_authorization_path: Path,
    screen_gate_path: Path,
) -> dict[str, Any]:
    """Close the helper's frozen gate lineage and the scheduler runtime together."""

    barrier = validate_screen_pass_barrier_s1(
        base,
        protocol,
        screen_schedule,
        screen_runtime_config,
        screen_authorization,
        screen_gate,
        screen_schedule_path=screen_schedule_path,
        screen_runtime_config_path=screen_runtime_config_path,
        screen_authorization_path=screen_authorization_path,
        screen_gate_path=screen_gate_path,
    )
    validate_screen_runtime_terminal_s1(
        screen_schedule,
        screen_runtime,
        screen_gate,
        runtime_path=screen_runtime_path,
        gate_path=screen_gate_path,
    )
    return barrier


def require_fresh_targets(paths: Sequence[tuple[Path, str]]) -> None:
    for path, label in paths:
        require(not path.exists(), f"{label} already exists: {path}")
        partial = (
            path.with_name(path.name + ".partial")
            if path.suffix == ""
            else path.with_suffix(path.suffix + ".partial")
        )
        require(not partial.exists(), f"partial {label} already exists: {partial}")


def validate_manifest_s1(
    manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    runner_head: str,
) -> dict[int, Path]:
    require(
        manifest.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_s1_confirmation_config_manifest.v1"
        and manifest.get("status")
        == "THREE_S1_CONFIRMATION_CONFIGS_PREPARED_NOT_STARTED"
        and manifest.get("confirmation_runner_git_head") == runner_head
        and manifest.get("selected_model") == CONFIRMATION_RUN_ID
        and manifest.get("required_seeds") == list(CONFIRMATION_SEEDS)
        and int(manifest.get("training_job_count", -1)) == 3
        and int(manifest.get("single_mode_training_job_count", -1)) == 0
        and int(manifest.get("checkpoint_validation_job_count", -1)) == 12,
        "S1 confirmation config manifest identity changed",
    )
    protected_reads_zero(manifest, label="config manifest")
    config_root = format_output_path(
        protocol, "runtime_config_root_template", runner_head
    )
    paths = [Path(str(value)) for value in manifest.get("config_paths", [])]
    require(len(paths) == 3, "S1 confirmation config path count changed")
    result: dict[int, Path] = {}
    for path in paths:
        config = read_json(path)
        seed = int(config.get("training_seed", -1))
        require(
            path == config_root / f"seed_{seed}.json"
            and config.get("schema_version") == CONFIRMATION_RUNTIME_SCHEMA
            and config.get("status") == CONFIRMATION_RUNTIME_STATUS
            and config.get("run_stage") == "CONFIRMATION"
            and config.get("selected_model") == CONFIRMATION_RUN_ID
            and config.get("confirmation_runner_git_head") == runner_head
            and seed in CONFIRMATION_SEEDS
            and seed not in result,
            f"S1 confirmation config identity changed: {path}",
        )
        protected_reads_zero(config, label=f"seed {seed} config")
        result[seed] = path
    require(
        tuple(result) == CONFIRMATION_SEEDS,
        "S1 confirmation config seed order or cohort changed",
    )
    return result


def build_authorization_s1(
    *,
    runner_head: str,
    configs: Mapping[int, Path],
    receipts: Mapping[str, Any],
    diagnostics: Mapping[int, Mapping[str, Any]],
    probes: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    first = read_json(configs[CONFIRMATION_SEEDS[0]])
    for seed in CONFIRMATION_SEEDS[1:]:
        config = read_json(configs[seed])
        require(
            config.get("screen_provenance") == first.get("screen_provenance")
            and config.get("screen_gate_path") == first.get("screen_gate_path")
            and config.get("screen_selected_checkpoint_pass")
            == first.get("screen_selected_checkpoint_pass"),
            "S1 confirmation configs disagree on frozen screen provenance",
        )
    return {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v4_s1_confirmation_launch_authorization.v1"
        ),
        "status": "XEDITSETFLOW_V4_S1_CONFIRMATION_LAUNCH_AUTHORIZED",
        **corrected_screen_provenance_baseline_binding_s1(),
        "authorized_git_head": runner_head,
        "authorized_run_ids": [CONFIRMATION_RUN_ID],
        "authorized_seeds": list(CONFIRMATION_SEEDS),
        "objective_identity": OBJECTIVE_IDENTITY,
        "cross_state_candidate_mode_responsibility_weight": OBJECTIVE_WEIGHT,
        "screen_runner_git_head": SCREEN_HEAD,
        "screen_gate_path": first["screen_gate_path"],
        "screen_selected_checkpoint_pass": first[
            "screen_selected_checkpoint_pass"
        ],
        "screen_provenance": first["screen_provenance"],
        "runner_verification_receipts": dict(receipts),
        "configured_physical_gpus": list(range(6)),
        "selected_physical_gpus": list(probes),
        "gpu_diagnostics": {
            str(gpu): dict(row) for gpu, row in diagnostics.items()
        },
        "cuda_bf16_probes": {
            str(gpu): dict(row) for gpu, row in probes.items()
        },
        "free_memory_gate_applied": False,
        "additional_seed_authorized": False,
        "development_test_authorized": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def validate_authorization_s1(
    authorization: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    runner_head: str,
) -> None:
    require(
        authorization.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_s1_confirmation_launch_authorization.v1"
        and authorization.get("status")
        == "XEDITSETFLOW_V4_S1_CONFIRMATION_LAUNCH_AUTHORIZED"
        and authorization.get(
            "corrected_screen_confirmation_provenance_baseline_git_head"
        )
        == CORRECTED_SCREEN_PROVENANCE_BASELINE_HEAD
        and authorization.get("corrected_screen_confirmation_provenance_audit")
        == str(CORRECTED_SCREEN_PROVENANCE_AUDIT)
        and authorization.get(
            "corrected_screen_confirmation_provenance_audit_status"
        )
        == CORRECTED_SCREEN_PROVENANCE_AUDIT_STATUS
        and authorization.get("authorized_git_head") == runner_head
        and authorization.get("authorized_run_ids") == [CONFIRMATION_RUN_ID]
        and authorization.get("authorized_seeds") == list(CONFIRMATION_SEEDS)
        and authorization.get("objective_identity") == OBJECTIVE_IDENTITY
        and float(
            authorization.get(
                "cross_state_candidate_mode_responsibility_weight", -1.0
            )
        )
        == OBJECTIVE_WEIGHT
        and authorization.get("screen_runner_git_head") == SCREEN_HEAD
        and authorization.get("screen_gate_path") == config.get("screen_gate_path")
        and authorization.get("screen_selected_checkpoint_pass")
        == config.get("screen_selected_checkpoint_pass")
        and authorization.get("screen_provenance")
        == config.get("screen_provenance")
        and authorization.get("free_memory_gate_applied") is False
        and authorization.get("additional_seed_authorized") is False,
        "S1 confirmation authorization identity changed",
    )
    protected_reads_zero(authorization, label="authorization")


def build_training_schedule_s1(
    protocol: Mapping[str, Any],
    manifest_path: Path,
    authorization_path: Path,
    configs: Mapping[int, Path],
    selected_gpus: Sequence[int],
    diagnostics: Mapping[int, Mapping[str, Any]],
    probes: Mapping[int, Mapping[str, Any]],
    receipts: Mapping[str, Any],
    *,
    runner_head: str,
) -> dict[str, Any]:
    gpus = tuple(int(gpu) for gpu in selected_gpus)
    require(
        gpus == (0, 1, 2),
        "S1 confirmation training GPU order changed from configured first three",
    )
    require(
        all(gpu in diagnostics and gpu in probes for gpu in gpus),
        "S1 confirmation training GPU evidence is incomplete",
    )
    runtime_root = format_output_path(
        protocol, "training_runtime_root_template", runner_head
    )
    log_root = format_output_path(
        protocol, "training_log_root_template", runner_head
    )
    queues: list[dict[str, Any]] = []
    for gpu, seed in zip(gpus, CONFIRMATION_SEEDS, strict=True):
        config_path = configs[seed]
        config = read_json(config_path)
        output = Path(str(config["output_root"])) / CONFIRMATION_RUN_ID
        queues.append(
            {
                "physical_gpu_index": gpu,
                "jobs": [
                    {
                        "job_key": f"setflow:{seed}:{CONFIRMATION_RUN_ID}",
                        "component": "setflow",
                        "training_seed": seed,
                        "run_id": CONFIRMATION_RUN_ID,
                        "physical_gpu_index": gpu,
                        "config_path": str(config_path),
                        "output_directory": str(output),
                        "terminal_summary": str(output / "training_summary.json"),
                        "terminal_failure": str(output / "failure.json"),
                        "log_path": str(
                            log_root / f"seed_{seed}_{CONFIRMATION_RUN_ID}.log"
                        ),
                        "command": [
                            str(PYTHON),
                            str(TRAINER),
                            "--config",
                            str(config_path),
                            "--run-id",
                            CONFIRMATION_RUN_ID,
                            "--authorization",
                            str(authorization_path),
                            "--physical-gpu-index",
                            str(gpu),
                            "--output-dir",
                            str(output),
                        ],
                    }
                ],
            }
        )
    posttraining_root = format_output_path(
        protocol, "posttraining_runtime_root_template", runner_head
    )
    gate = format_output_path(
        protocol, "confirmation_gate_output_template", runner_head
    )
    schedule = {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v4_s1_confirmation_training_schedule.v1"
        ),
        "status": "FROZEN_S1_CONFIRMATION_TRAINING_SCHEDULE",
        **corrected_screen_provenance_baseline_binding_s1(),
        "git_head": runner_head,
        "experiment_head": SCREEN_HEAD,
        "worktree": str(WORKTREE),
        "runtime_manifest": str(runtime_root / "runtime.json"),
        "eligible_components": ["setflow"],
        "confirmation_protocol": str(PROTOCOL),
        "config_manifest": str(manifest_path),
        "confirmation_authorization": str(authorization_path),
        "required_seeds": list(CONFIRMATION_SEEDS),
        "selected_model": CONFIRMATION_RUN_ID,
        "training_job_count": 3,
        "single_mode_training_job_count": 0,
        "runner_verification_receipts": dict(receipts),
        "gpu_diagnostics_before_launch": {
            str(gpu): dict(diagnostics[gpu]) for gpu in range(6)
        },
        "cuda_bf16_probes": {
            str(gpu): dict(probes[gpu]) for gpu in gpus
        },
        "gpu_queues": queues,
        "posttraining_bindings": {
            "protocol_path": str(PROTOCOL),
            "runner_git_head": runner_head,
            "config_manifest_path": str(manifest_path),
            "confirmation_authorization_path": str(authorization_path),
            "training_runtime_path": str(runtime_root / "runtime.json"),
            "posttraining_runtime_root": str(posttraining_root),
            "posttraining_log_root": str(
                format_output_path(
                    protocol, "posttraining_log_root_template", runner_head
                )
            ),
            "confirmation_gate_output": str(gate),
        },
        "gpu_selection_policy": "CONFIG_ORDER_FIRST_THREE_WITHOUT_MEMORY_SORTING_OR_GATE",
        "free_memory_gate_applied": False,
        "cpu_fallback_used": False,
        "active_performance_output_read": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    jobs = [job for queue in queues for job in queue["jobs"]]
    require(
        len(jobs) == 3
        and {job["run_id"] for job in jobs} == {CONFIRMATION_RUN_ID}
        and {int(job["training_seed"]) for job in jobs}
        == set(CONFIRMATION_SEEDS),
        "S1 confirmation schedule is not exactly three full-only trainings",
    )
    return schedule


def _gpu_error_details(error: Exception) -> dict[str, Any]:
    if isinstance(error, XEditSetFlowS1GpuError):
        return {
            "gpu_prelaunch_failure_reason": error.reason,
            "return_code": error.return_code,
            "stdout": error.stdout,
            "stderr": error.stderr,
            "missing_physical_gpus": list(error.missing_physical_gpus),
            "failed_physical_gpu_index": error.failed_physical_gpu_index,
            "probe_command": list(error.probe_command),
        }
    return {}


def write_prelaunch_failure_s1(
    path: Path,
    *,
    runner_head: str,
    family_roots: Sequence[Path],
    configured_gpus: Sequence[int],
    selected_gpus: Sequence[int],
    diagnostics: Mapping[int, Mapping[str, Any]],
    completed_probes: Mapping[int, Mapping[str, Any]],
    error: Exception,
) -> None:
    write_new_atomic(
        path,
        {
            "schema_version": (
                "route_a_v3_route2_xeditsetflow_v4_s1_confirmation_prelaunch_failure.v1"
            ),
            "status": "XEDITSETFLOW_V4_S1_CONFIRMATION_STOPPED_BEFORE_FAMILY_MATERIALIZATION",
            "runner_git_head": runner_head,
            "failure_stage": (
                "A100_CUDA_BF16_PROBE"
                if isinstance(error, XEditSetFlowS1GpuError)
                and error.failed_physical_gpu_index is not None
                else "GPU0_5_INVENTORY"
            ),
            "inventory_command": list(GPU_INVENTORY_COMMAND),
            "configured_physical_gpus": [int(gpu) for gpu in configured_gpus],
            "selected_physical_gpus": [int(gpu) for gpu in selected_gpus],
            "gpu_diagnostics": {
                str(gpu): dict(row) for gpu, row in diagnostics.items()
            },
            "completed_cuda_bf16_probes": {
                str(gpu): dict(row) for gpu, row in completed_probes.items()
            },
            **_gpu_error_details(error),
            "error_type": type(error).__name__,
            "error": str(error),
            "intended_family_roots": [str(path) for path in family_roots],
            "family_roots_created": [path.exists() for path in family_roots],
            "scheduler_started": False,
            "gpu_job_started": False,
            "automatic_retry_attempted": False,
            "free_memory_gate_applied": False,
            "cpu_fallback_used": False,
            "parameter_update_count": 0,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )


def perform_gpu_preflight_s1(
    *,
    runner_head: str,
    configured_gpus: Sequence[int],
    selected_gpus: Sequence[int],
    failure_path: Path,
    family_roots: Sequence[Path],
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    diagnostics: dict[int, dict[str, Any]] = {}
    probes: dict[int, dict[str, Any]] = {}
    try:
        diagnostics = gpu_diagnostics(configured_gpus)
        for gpu in selected_gpus:
            probes[int(gpu)] = cuda_bf16_probe(int(gpu))
    except Exception as error:
        write_prelaunch_failure_s1(
            failure_path,
            runner_head=runner_head,
            family_roots=family_roots,
            configured_gpus=configured_gpus,
            selected_gpus=selected_gpus,
            diagnostics=diagnostics,
            completed_probes=probes,
            error=error,
        )
        raise
    return diagnostics, probes


def write_scheduler_launch_failure_s1(
    path: Path,
    *,
    runner_head: str,
    command_line: Sequence[str],
    schedule_path: Path,
    runtime_path: Path,
    error: Exception,
) -> None:
    write_new_atomic(
        path,
        {
            "schema_version": (
                "route_a_v3_route2_xeditsetflow_v4_s1_confirmation_scheduler_launch_failure.v1"
            ),
            "status": "XEDITSETFLOW_V4_S1_CONFIRMATION_SCHEDULER_LAUNCH_TECHNICAL_FAILURE",
            "runner_git_head": runner_head,
            "failure_stage": "CONFIRMATION_TRAINING_SCHEDULER_PROCESS_LAUNCH",
            "scheduler_command": list(command_line),
            "schedule_path": str(schedule_path),
            "runtime_manifest": str(runtime_path),
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


def require_exact_pushed_clean_head(expected_head: str) -> None:
    require(
        re.fullmatch(r"[0-9a-f]{40}", expected_head) is not None,
        "expected Git HEAD is invalid",
    )
    require(
        command(["git", "rev-parse", "HEAD"]).stdout.strip() == expected_head,
        "S1 confirmation worktree is not at expected HEAD",
    )
    require(
        not command(["git", "status", "--porcelain"]).stdout.strip(),
        "S1 confirmation worktree is dirty",
    )
    require(
        command(["git", "branch", "--show-current"]).stdout.strip() == BRANCH,
        "S1 confirmation worktree is on the wrong branch",
    )
    require(
        command(["git", "rev-parse", f"origin/{BRANCH}"]).stdout.strip()
        == expected_head,
        "S1 confirmation exact HEAD has not been pushed to GitHub",
    )


def run(
    expected_head: str,
    screen_runtime_path: Path,
    *,
    shared_receipt_path: Path | None = None,
    setflow_receipt_path: Path | None = None,
) -> dict[str, Any]:
    for path, label in (
        (PYTHON, "formal Python"),
        (PROTOCOL, "S1 confirmation protocol"),
        (TRAINER, "S1 trainer"),
        (SCHEDULER, "generic confirmation training scheduler"),
    ):
        require(path.is_file(), f"{label} is absent: {path}")
    require_exact_pushed_clean_head(expected_head)
    validate_corrected_screen_provenance_baseline_s1(expected_head)

    protocol = read_json(PROTOCOL)
    base_path = WORKTREE / str(protocol.get("base_screen_config", ""))
    require(
        base_path.resolve().is_relative_to(WORKTREE.resolve())
        and base_path.is_file(),
        "S1 base screen config path is not a tracked worktree file",
    )
    base = read_json(base_path)
    require(
        base_path.resolve() == SCREEN_CONFIG.resolve(),
        "S1 confirmation base config is not the canonical screen config",
    )
    repo_audits = validate_repo_fact_audits(base)
    require_seed_valid_screen_head_s1(
        repo_audits["s1_seed_initialization_repair"], screen_head=SCREEN_HEAD
    )
    provenance = protocol.get("screen_provenance")
    require(isinstance(provenance, Mapping), "S1 screen provenance is absent")
    screen_schedule_path = Path(str(provenance["schedule_path"]))
    require(
        Path(str(provenance.get("runtime_path"))) == screen_runtime_path,
        "explicit S1 screen runtime differs from frozen protocol provenance",
    )
    screen_runtime_config_path = Path(str(provenance["runtime_config_path"]))
    screen_authorization_path = Path(str(provenance["authorization_path"]))
    screen_gate_path = Path(str(provenance["screen_gate_path"]))

    canonical_shared, canonical_setflow = expected_receipt_paths(expected_head)
    require(
        format_output_path(
            protocol, "shared_runner_verification_receipt_template", expected_head
        )
        == canonical_shared
        and format_output_path(
            protocol, "setflow_runner_verification_receipt_template", expected_head
        )
        == canonical_setflow,
        "S1 confirmation protocol receipt paths are not canonical",
    )
    receipts = consume_confirmation_receipts_s1(
        expected_head,
        shared_receipt_path or canonical_shared,
        setflow_receipt_path or canonical_setflow,
    )

    config_root = format_output_path(
        protocol, "runtime_config_root_template", expected_head
    )
    training_root = format_output_path(
        protocol, "training_runtime_root_template", expected_head
    )
    training_log_root = format_output_path(
        protocol, "training_log_root_template", expected_head
    )
    posttraining_root = format_output_path(
        protocol, "posttraining_runtime_root_template", expected_head
    )
    posttraining_log_root = format_output_path(
        protocol, "posttraining_log_root_template", expected_head
    )
    authorization_path = format_output_path(
        protocol, "authorization_output_template", expected_head
    )
    gate_path = format_output_path(
        protocol, "confirmation_gate_output_template", expected_head
    )
    failure_path = format_output_path(
        protocol, "prelaunch_failure_template", expected_head
    )
    require_fresh_targets(
        (
            (config_root, "S1 confirmation config root"),
            (training_root, "S1 confirmation training family"),
            (training_log_root, "S1 confirmation training log family"),
            (posttraining_root, "S1 confirmation posttraining family"),
            (posttraining_log_root, "S1 confirmation posttraining log family"),
            (authorization_path, "S1 confirmation authorization"),
            (gate_path, "S1 confirmation gate"),
            (gate_path.with_name(gate_path.name + ".failed.json"), "S1 confirmation adjudication failure"),
            (failure_path, "S1 confirmation prelaunch failure"),
        )
    )

    gpu_policy = protocol.get("gpu_policy")
    require(
        isinstance(gpu_policy, Mapping)
        and gpu_policy.get("physical_gpu_scope") == list(range(6))
        and gpu_policy.get("cuda_bf16_only") is True
        and gpu_policy.get("cpu_fallback") is False
        and gpu_policy.get("free_or_estimated_memory_gate") is False
        and gpu_policy.get("free_or_estimated_memory_sorting") is False,
        "S1 confirmation GPU policy changed",
    )
    configured_gpus = tuple(int(gpu) for gpu in gpu_policy["physical_gpu_scope"])
    selected_gpus = configured_gpus[:3]
    diagnostics, probes = perform_gpu_preflight_s1(
        runner_head=expected_head,
        configured_gpus=configured_gpus,
        selected_gpus=selected_gpus,
        failure_path=failure_path,
        family_roots=(config_root, training_root, posttraining_root),
    )

    for path, label in (
        (screen_schedule_path, "screen schedule"),
        (screen_runtime_path, "screen runtime"),
        (screen_runtime_config_path, "screen runtime config"),
        (screen_authorization_path, "screen authorization"),
        (screen_gate_path, "screen gate"),
    ):
        require(path.is_file(), f"S1 {label} is absent: {path}")
    screen_schedule = read_json(screen_schedule_path)
    require(
        Path(str(screen_schedule.get("runtime_manifest", "")))
        == screen_runtime_path,
        "explicit S1 screen runtime differs from the frozen schedule",
    )
    screen_runtime = read_json(screen_runtime_path)
    screen_runtime_config = read_json(screen_runtime_config_path)
    screen_authorization = read_json(screen_authorization_path)
    screen_gate = read_json(screen_gate_path)
    screen_barrier = validate_screen_bundle_s1(
        base,
        protocol,
        screen_schedule,
        screen_runtime,
        screen_runtime_config,
        screen_authorization,
        screen_gate,
        screen_schedule_path=screen_schedule_path,
        screen_runtime_path=screen_runtime_path,
        screen_runtime_config_path=screen_runtime_config_path,
        screen_authorization_path=screen_authorization_path,
        screen_gate_path=screen_gate_path,
    )

    configs_payload = build_confirmation_configs_s1(
        base,
        protocol,
        screen_schedule,
        screen_runtime_config,
        screen_authorization,
        screen_gate,
        screen_schedule_path=screen_schedule_path,
        screen_runtime_config_path=screen_runtime_config_path,
        screen_authorization_path=screen_authorization_path,
        screen_gate_path=screen_gate_path,
        confirmation_runner_git_head=expected_head,
    )
    manifest = materialize_confirmation_configs_s1(
        configs_payload,
        protocol,
        confirmation_runner_git_head=expected_head,
    )
    manifest_path = config_root / "manifest.json"
    configs = validate_manifest_s1(manifest, protocol, runner_head=expected_head)
    authorization = build_authorization_s1(
        runner_head=expected_head,
        configs=configs,
        receipts=receipts,
        diagnostics=diagnostics,
        probes=probes,
    )
    require(
        authorization["screen_provenance"] == screen_barrier,
        "S1 confirmation authorization differs from the validated screen barrier",
    )
    write_new_atomic(authorization_path, authorization)
    validate_authorization_s1(
        authorization,
        read_json(configs[CONFIRMATION_SEEDS[0]]),
        runner_head=expected_head,
    )

    training_root.mkdir(parents=True)
    training_log_root.mkdir(parents=True)
    schedule_path = training_root / "schedule.json"
    runtime_path = training_root / "runtime.json"
    schedule = build_training_schedule_s1(
        protocol,
        manifest_path,
        authorization_path,
        configs,
        selected_gpus,
        diagnostics,
        probes,
        receipts,
        runner_head=expected_head,
    )
    require(
        Path(str(schedule["runtime_manifest"])) == runtime_path,
        "S1 confirmation runtime path differs from the frozen family",
    )
    write_new_atomic(schedule_path, schedule)
    scheduler_log = training_log_root / "scheduler.log"
    scheduler_command = [
        str(PYTHON),
        str(SCHEDULER),
        "--schedule",
        str(schedule_path),
    ]
    stream = scheduler_log.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            scheduler_command,
            cwd=WORKTREE,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception as error:
        stream.close()
        write_scheduler_launch_failure_s1(
            training_root / "scheduler_launch.failed.json",
            runner_head=expected_head,
            command_line=scheduler_command,
            schedule_path=schedule_path,
            runtime_path=runtime_path,
            error=error,
        )
        raise
    stream.close()
    launch = {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v4_s1_confirmation_training_launch.v1"
        ),
        "status": "XEDITSETFLOW_V4_S1_CONFIRMATION_TRAINING_SCHEDULER_LAUNCHED",
        **corrected_screen_provenance_baseline_binding_s1(),
        "runner_git_head": expected_head,
        "scheduler_pid": process.pid,
        "schedule_path": str(schedule_path),
        "runtime_manifest": str(runtime_path),
        "config_manifest": str(manifest_path),
        "authorization": str(authorization_path),
        "screen_runtime": str(screen_runtime_path),
        "screen_gate": str(screen_gate_path),
        "scheduler_log": str(scheduler_log),
        "training_job_count": 3,
        "single_mode_training_job_count": 0,
        "selected_physical_gpus": list(selected_gpus),
        "free_memory_gate_applied": False,
        "cpu_fallback_used": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_new_atomic(training_root / "launch.json", launch)
    return launch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--screen-runtime", required=True, type=Path)
    parser.add_argument("--runner-verification-receipt", type=Path)
    parser.add_argument("--setflow-runner-verification-receipt", type=Path)
    arguments = parser.parse_args()
    result = run(
        arguments.expected_head,
        arguments.screen_runtime,
        shared_receipt_path=arguments.runner_verification_receipt,
        setflow_receipt_path=arguments.setflow_runner_verification_receipt,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
