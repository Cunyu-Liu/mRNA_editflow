"""Focused coverage for the retry2 Critic controls launcher/scheduler."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import scripts.route_a_v3.launch_route2_xeditcritic_v403_controls_retry2_after_full as retry2_launcher
import scripts.route_a_v3.run_route2_xeditcritic_v403_control_recovery_retry2_scheduler as retry2_scheduler


def test_retry2_launcher_identity_is_new_independent_family() -> None:
    assert retry2_launcher.CONTROL_RETRY_ORDINAL == 2
    assert (
        retry2_launcher.CONTROL_RETRY_IDENTITY
        == "v403_control_recovery_retry2"
    )
    assert retry2_launcher.PRIOR_FAILED_CONTROL_GIT_HEAD == (
        "697043fdbfb904dc98adc74095a1bcaa8d62b0f3"
    )
    assert retry2_launcher.PRIOR_FAILED_CONTROL_RUNTIME.name == "runtime.json"
    assert "retry1_runner" in str(retry2_launcher.PRIOR_FAILED_CONTROL_RUNTIME)
    assert retry2_launcher.PRIOR_CONTROL_OOM_TERMINAL_RECEIPT.name.endswith(
        "retry1_runner_697043fdbfb904dc98adc74095a1bcaa8d62b0f3_terminal.json"
    )
    assert retry2_launcher.CONTROL_RETRY_IDENTITY in str(
        retry2_launcher.control_family_paths("a" * 40)["output_root"]
    )


def test_retry2_scheduler_identity_matches_launcher() -> None:
    assert retry2_scheduler.CONTROL_RETRY_ORDINAL == (
        retry2_launcher.CONTROL_RETRY_ORDINAL
    )
    assert retry2_scheduler.CONTROL_RETRY_IDENTITY == (
        retry2_launcher.CONTROL_RETRY_IDENTITY
    )
    assert retry2_scheduler.PRIOR_FAILED_CONTROL_GIT_HEAD == (
        retry2_launcher.PRIOR_FAILED_CONTROL_GIT_HEAD
    )
    assert retry2_scheduler.PHYSICAL_GPU_INDICES == (
        retry2_launcher.PHYSICAL_GPU_INDICES
    )


def test_retry2_prior_validator_accepts_not_run_terminal_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retry1 receipt with NOT_RUN_AFTER_TERMINAL_FAILURE arms is accepted."""
    run_ids = list(retry2_launcher.CONTROL_RUN_IDS)
    receipt = {
        "schema_version": retry2_launcher.PRIOR_CONTROL_OOM_TERMINAL_RECEIPT_SCHEMA,
        "status": retry2_launcher.PRIOR_CONTROL_OOM_TERMINAL_RECEIPT_STATUS,
        "terminal_class": "TECHNICAL_FAILURE_TERMINAL",
        "old_current_git_head": retry2_launcher.PRIOR_FAILED_CONTROL_GIT_HEAD,
        "old_runner_git_head": retry2_launcher.PRIOR_FAILED_CONTROL_GIT_HEAD,
        "old_orchestration_git_head": retry2_launcher.PRIOR_FAILED_CONTROL_GIT_HEAD,
        "old_training_code_git_head": retry2_launcher.PRIOR_FAILED_CONTROL_GIT_HEAD,
        "old_runtime_path": str(retry2_launcher.PRIOR_FAILED_CONTROL_RUNTIME),
        "old_runtime_status": "XEDITCRITIC_V403_CONTROL_RECOVERY_TECHNICAL_FAILURE",
        "scheduler_pid": 1755575,
        "scheduler_process_gone": True,
        "ordered_control_run_ids": run_ids,
        "terminal_jobs": [
            {"run_id": run_id, "status": "x"} for run_id in run_ids
        ],
        "technical_failure_run_ids": [run_ids[0]],
        "terminal_summary_run_ids": [run_ids[1], run_ids[2]],
        "not_run_after_terminal_failure_run_ids": list(run_ids[3:]),
        "first_terminal_failure": {
            "run_id": run_ids[0],
            "reason": "JOB_TERMINAL_FAILURE_ARTIFACT",
            "return_code": 1,
            "terminal_artifact_kind": "FAILURE",
        },
        "cross_root_adjudication_run": False,
        "cross_root_gate_path": str(
            tmp_path / "cross_root" / "screen_gate.json"
        ),
        "cross_root_gate_absent": True,
        "free_memory_gate_applied": False,
        "terminal_artifact_payloads_read_by_transition": 0,
        "historical_terminal_payloads_read_before_cross_root": 0,
        "successor_authorized": False,
        "same_family_retry_authorized": False,
        "new_independent_retry_eligible": True,
        "old_family_artifacts_read_only": True,
        "old_runtime_read_count_this_transition": 1,
        "gpu_inventory_or_probe_executed": False,
        "gpu_or_model_execution_started": False,
        "protected_outcome_payload_read": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    receipt_path = tmp_path / "retry1_terminal_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        retry2_launcher,
        "PRIOR_CONTROL_OOM_TERMINAL_RECEIPT",
        receipt_path,
    )
    retry2_launcher.validate_prior_control_oom_terminal_receipt(
        receipt, receipt_path=receipt_path
    )


def test_retry2_control_family_paths_are_retry2_scoped(tmp_path: Path) -> None:
    head = "c" * 40
    paths = retry2_launcher.control_family_paths(head)
    assert paths["transition_gate"].name == "screen_gate.json"
    assert "retry2_" in str(paths["transition_gate"])
    assert retry2_launcher.CONTROL_RETRY_IDENTITY in str(paths["output_root"])
    assert retry2_launcher.CONTROL_RETRY_IDENTITY in str(paths["runtime_root"])
    assert "retry1" not in str(paths["output_root"])
    assert "retry1" not in str(paths["runtime_root"])