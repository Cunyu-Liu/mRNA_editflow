#!/usr/bin/env python3
"""Authorize SetFlow V4 confirmation from the V4.0.3 recovered screen gate."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_xeditsetflow_runtime_v4 import (
    require_setflow_v4_screen_launch_authorization,
)


TRAINING_HEAD = "edad89392077a0cf56e84dfcf94335606dd2b05a"
VALIDATION_HEAD = "37c5901000cf6bef1606f05af242512f1342ceb6"
SCREEN_EXPERIMENT_HEAD = "a7ef72fac23cd5b25dcc6c8d560236b97fa8b09d"
CONFIRMATION_SEEDS = (20260912, 20260913, 20260914)
SCIENCE_PROTOCOL_KEYS = (
    "required_seeds",
    "additional_seed_authorized",
    "selected_model",
    "training_policy",
    "checkpoint_gate",
    "paired_bootstrap",
    "terminal_f2_validation_summary",
    "development_test_authorized",
    "guidance_authorized",
    "development_test_outcome_reads",
    "new_final_evaluation_outcome_reads",
)
RUNNER_VERIFICATION_RECEIPT_SCHEMA = (
    "route_a_v3_route2_xeditsetflow_v403_runner_verification_receipt.v1"
)
RUNNER_VERIFICATION_RECEIPT_PASS = (
    "XEDITSETFLOW_V403_RUNNER_VERIFICATION_PASS"
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


class XEditSetFlowV403ConfirmationAuthorizationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditSetFlowV403ConfirmationAuthorizationError(message)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"artifact is not an object: {path}")
    return payload


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require_zero_protected_reads(
    payload: Mapping[str, Any], *, label: str
) -> None:
    require(
        int(payload.get("development_test_outcome_reads", -1)) == 0
        and int(payload.get("new_final_evaluation_outcome_reads", -1)) == 0,
        f"SetFlow V4.0.3 {label} reports a protected outcome read",
    )


def require_runner_verification_receipt_v403(
    receipt: Mapping[str, Any],
    *,
    current_runner_head: str,
) -> None:
    require(
        receipt.get("schema_version") == RUNNER_VERIFICATION_RECEIPT_SCHEMA
        and receipt.get("status") == RUNNER_VERIFICATION_RECEIPT_PASS
        and receipt.get("runner_git_head") == current_runner_head
        and receipt.get("worktree_clean") is True,
        "SetFlow V4.0.3 runner verification receipt is not exact-HEAD PASS",
    )
    focused = receipt.get("focused_tests")
    require(
        isinstance(focused, Mapping),
        "SetFlow V4.0.3 runner receipt lacks focused tests",
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
        "SetFlow V4.0.3 runner receipt reports failed or incomplete focused tests",
    )
    for group_index, required_markers in enumerate(
        FOCUSED_GROUP_REQUIRED_TEST_MARKERS
    ):
        require(
            all(
                marker in focused_commands[group_index]
                for marker in required_markers
            ),
            "SetFlow V4.0.3 runner receipt focused test group "
            f"{group_index + 1} lacks required test-module coverage",
        )

    v332 = receipt.get("v332_tests")
    require(
        isinstance(v332, Mapping),
        "SetFlow V4.0.3 runner receipt lacks V3.3.2 tests",
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
        "SetFlow V4.0.3 runner receipt reports failed or incomplete V3.3.2 tests",
    )
    require_zero_protected_reads(receipt, label="runner verification receipt")


def require_science_protocol_unchanged_v403(
    base_protocol: Mapping[str, Any],
    derived_protocol: Mapping[str, Any],
) -> None:
    require(
        base_protocol.get("schema_version")
        == derived_protocol.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_confirmation_protocol.v1",
        "SetFlow V4.0.3 confirmation protocol schema changed",
    )
    require(
        base_protocol.get("status")
        == derived_protocol.get("status")
        == "FROZEN_PROSPECTIVE_BEFORE_SCREEN_OR_CONFIRMATION_RESULT",
        "SetFlow V4.0.3 confirmation protocol is not prospectively frozen",
    )
    for key in SCIENCE_PROTOCOL_KEYS:
        require(
            derived_protocol.get(key) == base_protocol.get(key),
            f"SetFlow V4.0.3 confirmation science changed: {key}",
        )
    require(
        derived_protocol.get("required_seeds") == list(CONFIRMATION_SEEDS),
        "SetFlow V4.0.3 confirmation seed cohort changed",
    )


def require_recovery_config_derivation_v403(
    screen_config: Mapping[str, Any],
    recovery_config: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    provenance = protocol.get("validation_recovery_provenance")
    require(isinstance(provenance, Mapping), "recovery provenance is absent")
    require(
        provenance.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v403_confirmation_derivation.v1"
        and provenance.get("derivation")
        == "VALIDATION_ONLY_RECOVERY_FROM_TERMINAL_V4_CHECKPOINTS"
        and provenance.get("training_git_head") == TRAINING_HEAD
        and provenance.get("validation_git_head") == VALIDATION_HEAD
        and provenance.get("screen_experiment_head") == SCREEN_EXPERIMENT_HEAD,
        "SetFlow V4.0.3 recovery provenance identity changed",
    )
    allowed_overrides = {
        "status",
        "validation_output_root",
        "screen_gate_output_path",
    }
    require(
        set(recovery_config) == set(screen_config) | {"validation_recovery"},
        "SetFlow V4.0.3 recovery config field inventory changed",
    )
    for key, value in screen_config.items():
        if key not in allowed_overrides:
            require(
                recovery_config.get(key) == value,
                f"SetFlow V4.0.3 recovery config changed frozen field: {key}",
            )
    recovery = recovery_config.get("validation_recovery")
    require(isinstance(recovery, Mapping), "validation recovery record is absent")
    require(
        recovery_config.get("status")
        == "VALIDATION_ONLY_RECOVERY_FROM_TERMINAL_V4_CHECKPOINTS"
        and recovery.get("training_git_head") == TRAINING_HEAD
        and recovery.get("validation_git_head") == VALIDATION_HEAD
        and recovery.get("training_reused") is True
        and int(recovery.get("parameter_updates", -1)) == 0
        and recovery.get("scientific_thresholds_changed") is False,
        "SetFlow V4.0.3 recovery is not a zero-update dual-HEAD derivation",
    )
    require(
        recovery_config.get("screen_gate_output_path")
        == protocol.get("screen_gate_path")
        == provenance.get("recovered_screen_gate_path"),
        "SetFlow V4.0.3 recovered gate binding changed",
    )
    require(
        recovery.get("original_technical_gate")
        == provenance.get("original_technical_gate_path"),
        "SetFlow V4.0.3 original technical gate binding changed",
    )
    require_zero_protected_reads(recovery, label="recovery config provenance")
    require(
        recovery_config.get("development_test_outcomes_accessed") is False
        and recovery_config.get("new_final_evaluation_outcomes_accessed") is False,
        "SetFlow V4.0.3 recovery config authorizes protected outcomes",
    )


def require_recovery_terminal_v403(
    protocol: Mapping[str, Any],
    recovery_runtime: Mapping[str, Any],
    recovered_gate: Mapping[str, Any],
) -> None:
    provenance = protocol["validation_recovery_provenance"]
    require(
        recovery_runtime.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v403_validation_recovery_runtime.v1"
        and recovery_runtime.get("status")
        == "XEDITSETFLOW_V403_VALIDATION_RECOVERY_AND_GATE_TERMINAL"
        and recovery_runtime.get("git_head") == VALIDATION_HEAD
        and recovery_runtime.get("source_screen_head") == TRAINING_HEAD
        and recovery_runtime.get("experiment_head") == SCREEN_EXPERIMENT_HEAD,
        "SetFlow V4.0.3 validation recovery is not exact dual-HEAD terminal",
    )
    adjudication = recovery_runtime.get("setflow_adjudication")
    require(
        isinstance(adjudication, Mapping)
        and adjudication.get("status") == "TERMINAL_COMPLETE"
        and adjudication.get("gate_present") is True
        and adjudication.get("gate_path")
        == provenance.get("recovered_screen_gate_path"),
        "SetFlow V4.0.3 recovered adjudication is not terminal",
    )
    jobs = recovery_runtime.get("validation_jobs")
    require(
        isinstance(jobs, Mapping)
        and len(jobs) == 8
        and all(
            isinstance(row, Mapping)
            and row.get("status") == "TERMINAL_COMPLETE"
            and row.get("terminal_artifact_kind") == "SUMMARY"
            for row in jobs.values()
        ),
        "SetFlow V4.0.3 recovered validation jobs are not eight summaries",
    )
    require(
        recovered_gate.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_screen_gate.v1"
        and recovered_gate.get("status") == "XEDITSETFLOW_V4_SCREEN_PASS"
        and recovered_gate.get("confirmation_authorized") is True
        and recovered_gate.get("confirmation_seeds")
        == list(CONFIRMATION_SEEDS)
        and recovered_gate.get("additional_seed_authorized") is False
        and recovered_gate.get("development_test_authorized") is False
        and recovered_gate.get("guidance_authorized") is False,
        "SetFlow V4.0.3 recovered gate does not authorize exact confirmation",
    )
    selected_pass = recovered_gate.get("selected_checkpoint_pass")
    require(
        isinstance(selected_pass, int)
        and not isinstance(selected_pass, bool)
        and selected_pass in {4, 6, 8, 10},
        "SetFlow V4.0.3 recovered gate selected no frozen checkpoint",
    )
    for payload, label in (
        (protocol, "derived protocol"),
        (protocol["validation_recovery_provenance"], "recovery provenance"),
        (recovery_runtime, "recovery runtime"),
        (recovered_gate, "recovered gate"),
    ):
        require_zero_protected_reads(payload, label=label)
    require(
        recovery_runtime.get("active_performance_output_read") is False
        and int(recovery_runtime.get("critic_failure_payload_reads", -1)) == 0,
        "SetFlow V4.0.3 recovery consumed active or Critic performance output",
    )


def build_recovered_confirmation_authorization_v403(
    base_protocol: Mapping[str, Any],
    protocol: Mapping[str, Any],
    screen_config: Mapping[str, Any],
    screen_authorization: Mapping[str, Any],
    preflight: Mapping[str, Any],
    source_data_audit: Mapping[str, Any],
    recovery_config: Mapping[str, Any],
    recovery_runtime: Mapping[str, Any],
    recovered_gate: Mapping[str, Any],
    runner_verification_receipt: Mapping[str, Any],
    *,
    current_runner_head: str,
    runner_verification_receipt_path: str,
) -> dict[str, Any]:
    require(
        re.fullmatch(r"[0-9a-f]{40}", current_runner_head) is not None,
        "SetFlow V4.0.3 runner Git HEAD is invalid",
    )
    expected_receipt_path = protocol["runner_outputs"][
        "runner_verification_receipt_template"
    ].format(runner_git_head=current_runner_head)
    require(
        runner_verification_receipt_path == expected_receipt_path,
        "SetFlow V4.0.3 runner verification receipt path changed",
    )
    require_runner_verification_receipt_v403(
        runner_verification_receipt,
        current_runner_head=current_runner_head,
    )
    require_science_protocol_unchanged_v403(base_protocol, protocol)
    require_recovery_config_derivation_v403(
        screen_config, recovery_config, protocol
    )
    for run_id in ("v4_full", "v4_single_mode"):
        require_setflow_v4_screen_launch_authorization(
            screen_config,
            screen_authorization,
            preflight,
            source_data_audit,
            run_id=run_id,
            current_git_head=TRAINING_HEAD,
        )
    require_recovery_terminal_v403(protocol, recovery_runtime, recovered_gate)
    for payload, label in (
        (screen_authorization, "original screen authorization"),
        (preflight, "original preflight"),
        (source_data_audit, "source data audit"),
    ):
        require_zero_protected_reads(payload, label=label)
    screen_barriers = screen_authorization["barriers"]
    provenance = protocol["validation_recovery_provenance"]
    return {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v4_confirmation_launch_authorization.v1"
        ),
        "status": "XEDITSETFLOW_V4_CONFIRMATION_LAUNCH_AUTHORIZED",
        "authorization_derivation": (
            "V403_RECOVERED_SCREEN_PASS_WITH_DISTINCT_TRAINING_VALIDATION_AND_RUNNER_HEADS"
        ),
        "authorized_git_head": current_runner_head,
        "training_git_head": TRAINING_HEAD,
        "validation_git_head": VALIDATION_HEAD,
        "screen_experiment_head": SCREEN_EXPERIMENT_HEAD,
        "screen_authorization_path": provenance["original_screen_authorization"],
        "recovery_config_path": provenance["recovery_config_path"],
        "recovery_runtime_path": provenance["recovery_runtime_path"],
        "recovered_screen_gate_path": provenance["recovered_screen_gate_path"],
        "source_screen_head_test_evidence": {
            "source_screen_git_head": TRAINING_HEAD,
            "source_screen_head_focused_tests_passed": screen_barriers[
                "a100_current_head_focused_tests_passed"
            ],
            "source_screen_head_v332_tests_passed": screen_barriers[
                "a100_current_head_v332_tests_passed"
            ],
        },
        "runner_current_head_verification": {
            "receipt_path": runner_verification_receipt_path,
            "schema_version": runner_verification_receipt["schema_version"],
            "status": runner_verification_receipt["status"],
            "runner_git_head": runner_verification_receipt["runner_git_head"],
            "worktree_clean": runner_verification_receipt["worktree_clean"],
            "focused_tests": dict(runner_verification_receipt["focused_tests"]),
            "v332_tests": dict(runner_verification_receipt["v332_tests"]),
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
        "authorized_seeds": list(CONFIRMATION_SEEDS),
        "authorized_run_id": "v4_full",
        "barriers": {
            "screen_gate_passed": True,
            "a100_current_head_focused_tests_passed": (
                runner_verification_receipt["focused_tests"]["passed"]
            ),
            "a100_current_head_v332_tests_passed": (
                runner_verification_receipt["v332_tests"]["passed"]
            ),
            "source_token_cache_terminal_complete": screen_barriers[
                "source_token_cache_terminal_complete"
            ],
            "source_level_data_audit_passed": screen_barriers[
                "source_level_data_audit_passed"
            ],
            "formal_parameter_preflight_passed": screen_barriers[
                "formal_parameter_preflight_passed"
            ],
            "validation_only_recovery_terminal": True,
            "recovered_gate_passed": True,
            "zero_recovery_parameter_updates": True,
            "scientific_thresholds_unchanged": True,
        },
        "screen_selected_checkpoint_pass": recovered_gate[
            "selected_checkpoint_pass"
        ],
        "training_reused": True,
        "recovery_parameter_update_count": 0,
        "scientific_thresholds_changed": False,
        "additional_seed_authorized": False,
        "development_test_authorized": False,
        "guidance_authorized": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-confirmation-protocol", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--screen-config", required=True, type=Path)
    parser.add_argument("--screen-authorization", required=True, type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--source-data-audit", required=True, type=Path)
    parser.add_argument("--recovery-config", required=True, type=Path)
    parser.add_argument("--recovery-runtime", required=True, type=Path)
    parser.add_argument("--recovered-screen-gate", required=True, type=Path)
    parser.add_argument(
        "--runner-verification-receipt", required=True, type=Path
    )
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    protocol = read_json(arguments.protocol)
    provenance = protocol["validation_recovery_provenance"]
    require(
        str(arguments.recovery_config) == provenance["recovery_config_path"]
        and str(arguments.recovery_runtime) == provenance["recovery_runtime_path"]
        and str(arguments.recovered_screen_gate)
        == provenance["recovered_screen_gate_path"]
        and str(arguments.screen_authorization)
        == provenance["original_screen_authorization"],
        "SetFlow V4.0.3 recovery artifact paths differ from frozen protocol",
    )
    current_head = git_head()
    expected_output = protocol["runner_outputs"][
        "authorization_output_template"
    ].format(runner_git_head=current_head)
    expected_receipt = protocol["runner_outputs"][
        "runner_verification_receipt_template"
    ].format(runner_git_head=current_head)
    require(
        str(arguments.output) == expected_output,
        "SetFlow V4.0.3 confirmation authorization output path changed",
    )
    require(
        str(arguments.runner_verification_receipt) == expected_receipt,
        "SetFlow V4.0.3 runner verification receipt path changed",
    )
    require(
        not arguments.output.exists(),
        f"confirmation authorization exists: {arguments.output}",
    )
    result = build_recovered_confirmation_authorization_v403(
        read_json(arguments.base_confirmation_protocol),
        protocol,
        read_json(arguments.screen_config),
        read_json(arguments.screen_authorization),
        read_json(arguments.preflight),
        read_json(arguments.source_data_audit),
        read_json(arguments.recovery_config),
        read_json(arguments.recovery_runtime),
        read_json(arguments.recovered_screen_gate),
        read_json(arguments.runner_verification_receipt),
        current_runner_head=current_head,
        runner_verification_receipt_path=str(
            arguments.runner_verification_receipt
        ),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    partial = arguments.output.with_suffix(arguments.output.suffix + ".partial")
    require(not partial.exists(), f"partial authorization exists: {partial}")
    partial.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, arguments.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
