from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import scripts.route_a_v3.launch_route2_xedit_v4_confirmation_posttraining_after_terminal as posttraining
import scripts.route_a_v3.launch_route2_xeditcritic_v403_confirmation_after_cross_root_screen as launcher
from scripts.route_a_v3.adjudicate_route2_xeditcritic_v4_confirmation import (
    load_critic_confirmation_configs_v4,
)

CONTROL_HEAD = "c" * 40


def test_confirmation_scheduler_popen_failure_is_durable_and_one_shot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    family = tmp_path / "confirmation_family"
    family.mkdir()
    schedule = family / "schedule.json"
    attempt = family / "attempt.json"
    authorization = family / "authorization.json"
    for path in (schedule, attempt, authorization):
        path.write_text("{}\n", encoding="utf-8")
    failure = family / "scheduler_launch.failed.json"

    def fail(*args, **kwargs):
        raise OSError("confirmation scheduler spawn failed")

    monkeypatch.setattr(launcher.subprocess, "Popen", fail)
    arguments = {
        "failure_path": failure,
        "expected_head": "a" * 40,
        "command_line": ["python", "scheduler.py", "--schedule", str(schedule)],
        "schedule_path": schedule,
        "runtime_path": family / "runtime.json",
        "scheduler_log": family / "scheduler.log",
        "created_artifacts": {
            "launch_attempt_receipt": attempt,
            "confirmation_authorization": authorization,
            "schedule": schedule,
        },
    }
    with pytest.raises(
        launcher.XEditCriticV403ConfirmationLaunchError,
        match="durable technical failure",
    ):
        launcher.spawn_scheduler_with_failure_evidence(**arguments)

    payload = json.loads(failure.read_text(encoding="utf-8"))
    assert payload["status"] == (
        "XEDITCRITIC_V403_CONFIRMATION_SCHEDULER_LAUNCH_TECHNICAL_FAILURE"
    )
    assert payload["scheduler_started"] is False
    assert payload["gpu_job_started"] is False
    assert payload["created_artifact_paths"]["launch_attempt_receipt"] == str(
        attempt
    )
    assert "scheduler_pid" not in payload
    assert not (family / "launch.json").exists()

    with pytest.raises(
        launcher.XEditCriticV403ConfirmationLaunchError,
        match="already exists",
    ):
        launcher.spawn_scheduler_with_failure_evidence(**arguments)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _source_authorization(run_id: str, head: str) -> dict:
    return {
        "schema_version": (
            "route_a_v3_route2_xeditcritic_v4_"
            "screen_launch_authorization.v1"
        ),
        "status": "XEDITCRITIC_V4_SCREEN_LAUNCH_AUTHORIZED",
        "authorized_git_head": head,
        "authorized_run_ids": list(launcher.ARM_ORDER),
        "barriers": {key: True for key in launcher.SOURCE_BARRIERS},
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def _gate(tmp_path: Path, *, control_head: str = CONTROL_HEAD) -> dict:
    arm_sources = {}
    for run_id in launcher.ARM_ORDER:
        head = launcher.expected_source_head(run_id, control_head)
        authorization_path = tmp_path / "authorizations" / f"{run_id}.json"
        _write_json(
            authorization_path, _source_authorization(run_id, head)
        )
        arm_sources[run_id] = {
            "summary_path": str(
                launcher.expected_summary_paths(control_head)[run_id]
            ),
            "training_git_head": head,
            "source_role": launcher.expected_source_role(run_id),
            "launch_authorization_path": str(authorization_path),
            "launch_authorization_schema_version": (
                "route_a_v3_route2_xeditcritic_v4_"
                "screen_launch_authorization.v1"
            ),
            "launch_authorization_status": (
                "XEDITCRITIC_V4_SCREEN_LAUNCH_AUTHORIZED"
            ),
            "authorized_git_head": head,
            "run_id_authorization_verified": True,
            "authorization_protected_outcome_reads_verified_zero": True,
        }
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_screen_gate.v1",
        "status": "XEDITCRITIC_V4_SCREEN_PASS",
        "passed": True,
        "selectable_model": "V4-FULL",
        "screen_seed": 20260907,
        "checks": {key: True for key in launcher.SCREEN_CHECKS},
        "confirmation_authorized": True,
        "development_test_authorized": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
        "cross_root_transition": {
            "schema_version": (
                "route_a_v3_route2_xeditcritic_v403_cross_root_input.v1"
            ),
            "ordered_run_ids": list(launcher.ARM_ORDER),
            "arm_sources": arm_sources,
            "historical_c0_git_head": launcher.HISTORICAL_C0_GIT_HEAD,
            "historical_full_git_head": (
                launcher.REPAIRED_SCREEN_PROVENANCE_GIT_HEAD
            ),
            "control_runner_git_head": control_head,
            "full_runtime_path": str(launcher.FULL_RUNTIME),
            "full_runtime_status": (
                "XEDITCRITIC_V403_FULL_RECOVERY_TERMINAL"
            ),
            "control_runtime_path": str(
                launcher.control_runtime_path(control_head)
            ),
            "control_runtime_status": (
                "XEDITCRITIC_V403_CONTROL_RECOVERY_"
                "ALL_SIX_SUMMARIES_TERMINAL"
            ),
            "frozen_config_path": str(launcher.BASE_CONFIG),
            "legacy_gate_path": str(launcher.LEGACY_GATE),
            "legacy_gate_preserved": True,
            "terminal_summary_payloads_read": 8,
            "full_retrained": False,
            "c0_retrained": False,
            "old_v402_stopped_process_resumed": False,
            "scientific_thresholds_changed": False,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    }


def _base_inputs() -> tuple[dict, dict]:
    return (
        json.loads(launcher.BASE_CONFIG.read_text(encoding="utf-8")),
        json.loads(launcher.BASE_PROTOCOL.read_text(encoding="utf-8")),
    )


def _diagnostics() -> dict[int, dict]:
    return {
        gpu: {
            "name": f"GPU-{gpu}",
            "free_memory_mib": 0,
            "total_memory_mib": 40960,
        }
        for gpu in launcher.PHYSICAL_GPUS
    }


def _probes() -> dict[int, dict]:
    return {
        gpu: {
            "physical_gpu_index": gpu,
            "device_name": f"GPU-{gpu}",
            "device_type": "cuda",
            "dtype": "BFLOAT16",
            "cuda_available": True,
            "bf16_supported": True,
            "cpu_fallback_used": False,
        }
        for gpu in launcher.PHYSICAL_GPUS
    }


def _focused_test_commands() -> list[str]:
    return [
        "python -m pytest -q "
        "test_score_route2_xeditflow_closed_frozen_methods_v3.py "
        "test_launch_route2_xeditflow_v403_guidance_after_dual_readiness.py "
        "test_launch_route2_xeditflow_v4_guidance_authorization_after_dual_readiness.py "
        "test_launch_route2_xeditflow_v4_guidance_screen_after_authorization.py "
        "test_launch_route2_xeditflow_v4_final_after_guidance_screen.py "
        "test_launch_route2_xeditcritic_v403_confirmation_after_cross_root_screen.py "
        "test_launch_route2_xeditsetflow_v403_recovered_confirmation_posttraining.py "
        "test_authorize_route2_xeditsetflow_v403_recovered_confirmation.py",
        "python -m pytest -q "
        "test_transition_adjudicate_route2_xeditcritic_v403_cross_root_screen.py "
        "test_prepare_route2_xeditcritic_v4_confirmation_configs.py "
        "test_route2_xeditcritic_v4_confirmation_runtime.py",
        "python -m pytest -q "
        "test_run_route2_xeditsetflow_v402_terminal_validation_scheduler.py "
        "test_adjudicate_route2_xeditsetflow_v4_confirmation.py "
        "test_route2_xeditsetflow_training_v4.py "
        "test_route2_xeditsetflow_s1_protocol.py "
        "test_route2_xeditsetflow_s1.py "
        "test_train_route2_xeditsetflow_s1.py "
        "test_validate_route2_xeditsetflow_s1_checkpoint.py "
        "test_route2_xeditsetflow_gate_s1.py "
        "test_run_route2_xeditsetflow_s1_screen_scheduler.py "
        "test_launch_route2_xeditsetflow_s1_screen_after_v403_terminal.py "
        "test_transition_record_route2_xeditsetflow_s1_930_terminal_invalidation.py "
        "test_route2_xeditsetflow_confirmation_s1.py "
        "test_launch_route2_xeditsetflow_s1_confirmation_after_screen_pass.py "
        "test_launch_route2_xeditsetflow_s1_confirmation_posttraining.py "
        "test_adjudicate_route2_xeditsetflow_s1_confirmation.py",
        "python -m pytest -q "
        "test_run_route2_xeditflow_v4_guidance_screen_scheduler.py "
        "test_adjudicate_route2_xeditflow_guidance_screen_v4.py "
        "test_route2_xeditflow_guidance_v4.py",
        "python -m pytest -q "
        "test_train_route2_xeditcritic_v4.py "
        "test_run_route2_xeditcritic_v4_loso_scheduler.py",
        "python -m pytest -q "
        "test_run_route2_xedit_v4_confirmation_training_scheduler.py "
        "test_run_route2_xedit_v4_confirmation_posttraining_scheduler.py",
        "python -m pytest -q "
        "test_launch_route2_xedit_v4_confirmation_training_after_screen_pass.py "
        "test_launch_route2_xedit_v4_confirmation_posttraining_after_terminal.py "
        "test_launch_route2_xeditsetflow_v403_recovered_confirmation.py "
        "test_launch_route2_xeditcritic_v403_controls_after_full.py "
        "test_launch_route2_xeditcritic_v4_atomic_frozen_test_after_confirmation.py "
        "test_launch_route2_xeditcritic_v4_refit_after_atomic_test.py "
        "test_launch_route2_xeditcritic_v4_loso_after_refits.py",
        "python -m pytest -q "
        "test_prepare_route2_xeditflow_final_generation_configs_v4.py "
        "test_evaluate_route2_xeditflow_closed_scores_v4.py "
        "test_compare_route2_xeditflow_independent_evaluator_v4.py "
        "test_xeditflow_v4_final_evidence_chain.py "
        "test_run_route2_xeditflow_strongest_timing_v4.py "
        "test_reproduce_route2_base_flow_v2_handover_validation.py "
        "test_export_route2_xeditflow_v4_terminal_training_ledger.py",
    ]


def _runner_receipt(runner_head: str) -> dict:
    return {
        "schema_version": launcher.RUNNER_VERIFICATION_RECEIPT_SCHEMA,
        "status": launcher.RUNNER_VERIFICATION_RECEIPT_PASS,
        "runner_git_head": runner_head,
        "worktree_clean": True,
        "focused_tests": {
            "command": _focused_test_commands(),
            "passed": True,
            "passed_count": 203,
            "failed_count": 0,
            "isolated_process_groups": True,
            "group_passed_counts": [75, 14, 4, 26, 14, 8, 36, 26],
        },
        "v332_tests": {
            "command": ["python", "-m", "pytest", "tests/*v332*.py"],
            "passed": True,
            "passed_count": 96,
            "failed_count": 0,
        },
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def _training_semantics(runner_head: str) -> dict:
    return {
        "historical_repaired_screen_provenance_git_head": (
            launcher.REPAIRED_SCREEN_PROVENANCE_GIT_HEAD
        ),
        "historical_c0_git_head": launcher.HISTORICAL_C0_GIT_HEAD,
        "previous_successor_semantic_baseline_git_head": (
            launcher.TRAINING_SEMANTICS_PREVIOUS_SUCCESSOR_BASELINE_HEAD
        ),
        "audited_successor_semantic_baseline_git_head": (
            launcher.TRAINING_SEMANTICS_AUDITED_SUCCESSOR_BASELINE_HEAD
        ),
        "audited_successor_semantic_baseline_audit": str(
            launcher.TRAINING_SEMANTICS_BASELINE_AUDIT
        ),
        "audited_successor_semantic_baseline_audit_status": (
            "XEDITCRITIC_V403_CONFIRMATION_"
            "TRAINING_SEMANTICS_REAUDIT_V2_PASS"
        ),
        "incremental_changed_training_semantic_paths_since_previous_successor": (
            list(launcher.TRAINING_SEMANTICS_INCREMENTAL_CHANGED_PATHS)
        ),
        "runner_git_head": runner_head,
        "training_git_head": runner_head,
        "training_semantic_diff_paths_since_audited_successor_baseline": [],
        "training_semantics_unchanged_since_audited_successor_baseline": True,
        "repaired_screen_is_historical_provenance_only": True,
    }


def test_fixed_cross_root_pass_and_all_eight_authorizations_are_accepted(
    tmp_path: Path,
) -> None:
    gate = _gate(tmp_path)
    launcher.validate_cross_root_gate(gate)
    authorizations = launcher.load_and_validate_source_authorizations(gate)
    assert tuple(authorizations) == launcher.ARM_ORDER
    assert authorizations["c0_v4"]["authorized_git_head"] == (
        launcher.HISTORICAL_C0_GIT_HEAD
    )
    assert all(
        authorizations[run_id]["authorized_git_head"]
        == CONTROL_HEAD
        for run_id in launcher.CONTROL_RUN_IDS
    )
    assert authorizations["v4_full"]["authorized_git_head"] == (
        launcher.REPAIRED_SCREEN_PROVENANCE_GIT_HEAD
    )


def test_sort_keys_persisted_pass_gate_round_trip_preserves_semantic_arm_order(
    tmp_path: Path,
) -> None:
    gate_path = tmp_path / "persisted_cross_root_gate.json"
    gate_path.write_text(
        json.dumps(_gate(tmp_path), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    persisted = json.loads(gate_path.read_text(encoding="utf-8"))
    transition = persisted["cross_root_transition"]
    assert tuple(transition["arm_sources"]) != launcher.ARM_ORDER
    assert set(transition["arm_sources"]) == set(launcher.ARM_ORDER)
    assert transition["ordered_run_ids"] == list(launcher.ARM_ORDER)

    launcher.validate_cross_root_gate(persisted)
    authorizations = launcher.load_and_validate_source_authorizations(
        persisted
    )
    assert tuple(authorizations) == launcher.ARM_ORDER


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda gate: gate.update(
                status="XEDITCRITIC_V4_SCREEN_NO_GO",
                passed=False,
                confirmation_authorized=False,
            ),
            "not terminal PASS",
        ),
        (
            lambda gate: gate.update(development_test_outcome_reads=1),
            "protected outcome read",
        ),
        (
            lambda gate: gate["cross_root_transition"].update(
                legacy_gate_preserved=False
            ),
            "transition provenance changed",
        ),
        (
            lambda gate: gate["cross_root_transition"]["arm_sources"][
                "c0_v4"
            ].update(
                authorized_git_head=(
                    launcher.REPAIRED_SCREEN_PROVENANCE_GIT_HEAD
                )
            ),
            "provenance changed for c0_v4",
        ),
        (
            lambda gate: gate["cross_root_transition"]["arm_sources"][
                "v4_no_moe"
            ].update(authorized_git_head=launcher.HISTORICAL_C0_GIT_HEAD),
            "provenance changed for v4_no_moe",
        ),
    ],
)
def test_gate_fails_closed_on_nonpass_protected_or_wrong_provenance(
    tmp_path: Path, mutation, error: str
) -> None:
    gate = _gate(tmp_path)
    mutation(gate)
    with pytest.raises(Exception, match=error):
        launcher.validate_cross_root_gate(gate)


def test_source_authorization_artifact_must_match_embedded_provenance(
    tmp_path: Path,
) -> None:
    gate = _gate(tmp_path)
    path = Path(
        gate["cross_root_transition"]["arm_sources"]["v4_no_cross"][
            "launch_authorization_path"
        ]
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["authorized_git_head"] = launcher.HISTORICAL_C0_GIT_HEAD
    _write_json(path, payload)
    with pytest.raises(Exception, match="v4_no_cross"):
        launcher.load_and_validate_source_authorizations(gate)


def test_confirmation_configs_change_only_the_screen_gate_binding(
    tmp_path: Path,
) -> None:
    base_config, base_protocol = _base_inputs()
    gate = _gate(tmp_path)
    runner_head = "c" * 40
    derived, configs = launcher.build_confirmation_configs(
        base_config, base_protocol, gate, runner_head=runner_head
    )
    assert base_protocol["screen_gate_path"] == str(launcher.LEGACY_GATE)
    assert derived["screen_gate_path"] == str(
        launcher.cross_root_gate_path(runner_head)
    )
    assert all(
        derived[key] == value
        for key, value in base_protocol.items()
        if key != "screen_gate_path"
    )
    assert [config["training_seed"] for config in configs] == list(
        launcher.CONFIRMATION_SEEDS
    )
    assert all(
        config["screen_gate_path"]
        == str(launcher.cross_root_gate_path(runner_head))
        and config["required_confirmation_run_ids"]
        == ["v4_full", "c0_v4"]
        for config in configs
    )


def test_exact_runner_accepts_audit_then_rejects_postbaseline_semantic_diff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def exact_audit_then_unchanged_runner(command_args, **kwargs):
        commands.append(list(command_args))
        if command_args[3] == (
            launcher.TRAINING_SEMANTICS_PREVIOUS_AUDITED_BASELINE_HEAD
        ):
            return SimpleNamespace(
                stdout=(
                    "\n".join(
                        launcher.TRAINING_SEMANTICS_REAUDIT_CHANGED_PATHS
                    )
                    + "\n"
                )
            )
        if command_args[3] == (
            launcher.TRAINING_SEMANTICS_PREVIOUS_SUCCESSOR_BASELINE_HEAD
        ):
            return SimpleNamespace(
                stdout=(
                    "\n".join(
                        launcher.TRAINING_SEMANTICS_INCREMENTAL_CHANGED_PATHS
                    )
                    + "\n"
                )
            )
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(
        launcher,
        "command",
        exact_audit_then_unchanged_runner,
    )
    receipt = launcher.validate_runner_training_semantics("c" * 40)
    assert (
        receipt[
            "training_semantics_unchanged_since_audited_successor_baseline"
        ]
        is True
    )
    assert "training_semantics_unchanged" not in receipt
    assert receipt["historical_repaired_screen_provenance_git_head"] == (
        launcher.REPAIRED_SCREEN_PROVENANCE_GIT_HEAD
    )
    assert receipt["audited_successor_semantic_baseline_git_head"] == (
        launcher.TRAINING_SEMANTICS_AUDITED_SUCCESSOR_BASELINE_HEAD
    )
    assert commands == [
        [
            "git",
            "diff",
            "--name-only",
            launcher.TRAINING_SEMANTICS_PREVIOUS_AUDITED_BASELINE_HEAD,
            launcher.TRAINING_SEMANTICS_AUDITED_SUCCESSOR_BASELINE_HEAD,
            "--",
            *launcher.TRAINING_SEMANTIC_PATHS,
        ],
        [
            "git",
            "diff",
            "--name-only",
            launcher.TRAINING_SEMANTICS_PREVIOUS_SUCCESSOR_BASELINE_HEAD,
            launcher.TRAINING_SEMANTICS_AUDITED_SUCCESSOR_BASELINE_HEAD,
            "--",
            *launcher.TRAINING_SEMANTIC_PATHS,
        ],
        [
            "git",
            "diff",
            "--name-only",
            launcher.TRAINING_SEMANTICS_AUDITED_SUCCESSOR_BASELINE_HEAD,
            "c" * 40,
            "--",
            *launcher.TRAINING_SEMANTIC_PATHS,
        ]
    ]

    def exact_audit_then_changed_runner(command_args, **kwargs):
        if command_args[3] == (
            launcher.TRAINING_SEMANTICS_PREVIOUS_AUDITED_BASELINE_HEAD
        ):
            return SimpleNamespace(
                stdout=(
                    "\n".join(
                        launcher.TRAINING_SEMANTICS_REAUDIT_CHANGED_PATHS
                    )
                    + "\n"
                )
            )
        if command_args[3] == (
            launcher.TRAINING_SEMANTICS_PREVIOUS_SUCCESSOR_BASELINE_HEAD
        ):
            return SimpleNamespace(
                stdout=(
                    "\n".join(
                        launcher.TRAINING_SEMANTICS_INCREMENTAL_CHANGED_PATHS
                    )
                    + "\n"
                )
            )
        return SimpleNamespace(
            stdout="scripts/route_a_v3/train_route2_xeditcritic_v4.py\n"
        )

    monkeypatch.setattr(
        launcher,
        "command",
        exact_audit_then_changed_runner,
    )
    with pytest.raises(Exception, match="training semantics changed"):
        launcher.validate_runner_training_semantics("d" * 40)


def test_committed_successor_head_preserves_audited_training_semantics(
) -> None:
    current_head = launcher.command(
        ["git", "rev-parse", "HEAD"]
    ).stdout.strip()
    receipt = launcher.validate_runner_training_semantics(current_head)
    assert receipt["audited_successor_semantic_baseline_git_head"] == (
        launcher.TRAINING_SEMANTICS_AUDITED_SUCCESSOR_BASELINE_HEAD
    )
    assert receipt[
        "training_semantic_diff_paths_since_audited_successor_baseline"
    ] == []
    assert receipt[
        "training_semantics_unchanged_since_audited_successor_baseline"
    ] is True
    assert receipt["audited_successor_changed_training_semantic_paths"] == (
        list(launcher.TRAINING_SEMANTICS_REAUDIT_CHANGED_PATHS)
    )
    assert len(receipt["audited_successor_changed_training_semantic_paths"]) == 20
    assert receipt[
        "incremental_changed_training_semantic_paths_since_previous_successor"
    ] == list(launcher.TRAINING_SEMANTICS_INCREMENTAL_CHANGED_PATHS)
    assert len(
        receipt[
            "incremental_changed_training_semantic_paths_since_previous_successor"
        ]
    ) == 6
    assert receipt["audited_successor_semantic_baseline_audit"] == str(
        launcher.TRAINING_SEMANTICS_BASELINE_AUDIT
    )


def test_training_semantics_reaudit_accepts_exact_v2_audit() -> None:
    audit = launcher.read_json(launcher.TRAINING_SEMANTICS_BASELINE_AUDIT)
    launcher.validate_training_semantics_baseline_audit(audit)
    assert audit["changed_training_semantic_paths"] == list(
        launcher.TRAINING_SEMANTICS_REAUDIT_CHANGED_PATHS
    )
    assert audit["path_classification"] == (
        launcher.TRAINING_SEMANTICS_REAUDIT_PATH_CLASSIFICATION
    )
    assert audit[
        "incremental_changed_training_semantic_paths_since_previous_successor"
    ] == list(launcher.TRAINING_SEMANTICS_INCREMENTAL_CHANGED_PATHS)
    assert audit["incremental_path_classification"] == (
        launcher.TRAINING_SEMANTICS_INCREMENTAL_PATH_CLASSIFICATION
    )
    assert audit["previous_successor_audit_review"] == {
        "path": (
            "audits/route_a_v3_route2_xeditcritic_v403_confirmation_"
            "training_semantics_reaudit_"
            "eba5b17431cb8e19202e5ea788fd419338da2d66.json"
        ),
        "preserved_as_history": True,
        "consumed_by_current_launcher": False,
    }
    assert launcher.TRAINING_SEMANTICS_BASELINE_AUDIT.name.endswith(
        "f1a2328db57e1bd20fcc5cd5e6a23abcf4c62b66.json"
    )


def test_training_semantics_reaudit_rejects_path_or_classification_drift() -> None:
    original = launcher.read_json(launcher.TRAINING_SEMANTICS_BASELINE_AUDIT)
    path_drift = copy.deepcopy(original)
    path_drift["changed_training_semantic_paths"] = path_drift[
        "changed_training_semantic_paths"
    ][:-1]
    with pytest.raises(Exception, match="re-audit"):
        launcher.validate_training_semantics_baseline_audit(path_drift)

    classification_drift = copy.deepcopy(original)
    classification_drift["path_classification"][
        "scripts/route_a_v3/train_route2_xeditcritic_v4.py"
    ] = "MODEL_OBJECTIVE_CHANGED"
    with pytest.raises(Exception, match="re-audit"):
        launcher.validate_training_semantics_baseline_audit(
            classification_drift
        )

    incremental_drift = copy.deepcopy(original)
    incremental_drift[
        "incremental_changed_training_semantic_paths_since_previous_successor"
    ] = incremental_drift[
        "incremental_changed_training_semantic_paths_since_previous_successor"
    ][:-1]
    with pytest.raises(Exception, match="re-audit"):
        launcher.validate_training_semantics_baseline_audit(
            incremental_drift
        )


@pytest.mark.parametrize(
    "field", tuple(launcher.TRAINING_SEMANTICS_CHANGED_FLAGS)
)
def test_training_semantics_reaudit_rejects_changed_flag_drift(
    field: str,
) -> None:
    audit = launcher.read_json(launcher.TRAINING_SEMANTICS_BASELINE_AUDIT)
    audit[field] = False
    with pytest.raises(Exception, match="re-audit"):
        launcher.validate_training_semantics_baseline_audit(audit)


@pytest.mark.parametrize(
    "field", tuple(launcher.TRAINING_SEMANTICS_UNCHANGED_FLAGS)
)
def test_training_semantics_reaudit_rejects_narrow_unchanged_flag_drift(
    field: str,
) -> None:
    audit = launcher.read_json(launcher.TRAINING_SEMANTICS_BASELINE_AUDIT)
    audit[field] = True
    with pytest.raises(Exception, match="re-audit"):
        launcher.validate_training_semantics_baseline_audit(audit)


@pytest.mark.parametrize(
    "historical_head",
    (
        launcher.REPAIRED_SCREEN_PROVENANCE_GIT_HEAD,
        launcher.HISTORICAL_C0_GIT_HEAD,
    ),
)
def test_runner_semantics_rejects_historical_provenance_as_current_runner(
    historical_head: str,
) -> None:
    with pytest.raises(Exception, match="historical screen provenance"):
        launcher.validate_runner_training_semantics(historical_head)


def test_training_semantics_reaudit_rejects_f34_x_role_mixing() -> None:
    original = launcher.read_json(launcher.TRAINING_SEMANTICS_BASELINE_AUDIT)
    repaired_as_successor = copy.deepcopy(original)
    repaired_as_successor["audited_successor_semantic_baseline_git_head"] = (
        launcher.REPAIRED_SCREEN_PROVENANCE_GIT_HEAD
    )
    with pytest.raises(Exception, match="re-audit"):
        launcher.validate_training_semantics_baseline_audit(
            repaired_as_successor
        )

    successor_as_repaired = copy.deepcopy(original)
    successor_as_repaired["critic_repaired_screen_provenance_git_head"] = (
        launcher.TRAINING_SEMANTICS_AUDITED_SUCCESSOR_BASELINE_HEAD
    )
    with pytest.raises(Exception, match="re-audit"):
        launcher.validate_training_semantics_baseline_audit(
            successor_as_repaired
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda receipt: receipt.update(runner_git_head="d" * 40),
            "exact-HEAD clean PASS",
        ),
        (
            lambda receipt: receipt.update(worktree_clean=False),
            "exact-HEAD clean PASS",
        ),
        (
            lambda receipt: receipt["focused_tests"].update(
                passed=False, failed_count=1
            ),
            "focused tests",
        ),
        (
            lambda receipt: receipt["v332_tests"].update(
                passed=False, failed_count=1
            ),
            "V3.3.2 tests",
        ),
        (
            lambda receipt: receipt.update(development_test_outcome_reads=1),
            "protected outcome read",
        ),
    ],
)
def test_runner_receipt_fails_closed_on_wrong_head_dirty_or_failed_tests(
    mutation, error: str
) -> None:
    runner_head = "c" * 40
    receipt = _runner_receipt(runner_head)
    mutation(receipt)
    with pytest.raises(Exception, match=error):
        launcher.validate_runner_verification_receipt(
            receipt,
            runner_head=runner_head,
            receipt_path=launcher.runner_verification_receipt_path(
                runner_head
            ),
        )


def test_runner_receipt_accepts_complete_eight_group_coverage() -> None:
    runner_head = "c" * 40
    launcher.validate_runner_verification_receipt(
        _runner_receipt(runner_head),
        runner_head=runner_head,
        receipt_path=launcher.runner_verification_receipt_path(runner_head),
    )


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("six_groups", "focused tests"),
        (
            "c5db_group7_missing_new_markers",
            "lacks required test-module coverage",
        ),
        (
            "missing_setflow_authorization_marker",
            "lacks required test-module coverage",
        ),
        (
            "missing_handover_validation_marker",
            "lacks required test-module coverage",
        ),
        (
            "missing_terminal_training_ledger_marker",
            "lacks required test-module coverage",
        ),
        ("group_sum_mismatch", "focused tests"),
        ("focused_below_c5db_floor", "focused tests"),
        ("not_isolated", "focused tests"),
        ("nonpositive_group", "focused tests"),
        ("v332_not_96", "V3.3.2 tests"),
        ("v332_glob_absent", "V3.3.2 tests"),
    ],
)
def test_runner_receipt_coverage_fails_closed(
    case: str, error: str
) -> None:
    runner_head = "c" * 40
    receipt = _runner_receipt(runner_head)
    focused = receipt["focused_tests"]
    if case == "six_groups":
        focused["command"] = focused["command"][:6]
        focused["group_passed_counts"] = focused[
            "group_passed_counts"
        ][:6]
        focused["passed_count"] = sum(focused["group_passed_counts"])
    elif case == "c5db_group7_missing_new_markers":
        focused["command"][6] = (
            "python -m pytest -q "
            "test_launch_route2_xedit_v4_confirmation_training_after_screen_pass.py "
            "test_launch_route2_xedit_v4_confirmation_posttraining_after_terminal.py "
            "test_launch_route2_xeditsetflow_v403_recovered_confirmation.py"
        )
    elif case == "missing_setflow_authorization_marker":
        focused["command"][0] = focused["command"][0].replace(
            "test_authorize_route2_xeditsetflow_v403_recovered_confirmation.py",
            "",
        )
    elif case == "missing_handover_validation_marker":
        focused["command"][7] = focused["command"][7].replace(
            "test_reproduce_route2_base_flow_v2_handover_validation.py",
            "",
        )
    elif case == "missing_terminal_training_ledger_marker":
        focused["command"][7] = focused["command"][7].replace(
            "test_export_route2_xeditflow_v4_terminal_training_ledger.py",
            "",
        )
    elif case == "group_sum_mismatch":
        focused["group_passed_counts"][0] += 1
    elif case == "focused_below_c5db_floor":
        focused["group_passed_counts"][0] -= 1
        focused["passed_count"] = 202
    elif case == "not_isolated":
        focused["isolated_process_groups"] = False
    elif case == "nonpositive_group":
        focused["group_passed_counts"][1] += focused[
            "group_passed_counts"
        ][0]
        focused["group_passed_counts"][0] = 0
    elif case == "v332_not_96":
        receipt["v332_tests"]["passed_count"] = 95
    elif case == "v332_glob_absent":
        receipt["v332_tests"]["command"][-1] = "tests/v332.py"
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(case)

    with pytest.raises(Exception, match=error):
        launcher.validate_runner_verification_receipt(
            receipt,
            runner_head=runner_head,
            receipt_path=launcher.runner_verification_receipt_path(
                runner_head
            ),
        )


@pytest.mark.parametrize(
    "marker", launcher.FOCUSED_GROUP_REQUIRED_TEST_MARKERS[2][2:]
)
def test_runner_receipt_requires_each_s1_focused_marker(
    marker: str,
) -> None:
    runner_head = "c" * 40
    receipt = _runner_receipt(runner_head)
    receipt["focused_tests"]["command"][2] = receipt["focused_tests"][
        "command"
    ][2].replace(marker, "")

    with pytest.raises(
        Exception, match="lacks required test-module coverage"
    ):
        launcher.validate_runner_verification_receipt(
            receipt,
            runner_head=runner_head,
            receipt_path=launcher.runner_verification_receipt_path(
                runner_head
            ),
        )


def test_manifest_is_standard_consumer_shape_and_binds_exact_runner(
    tmp_path: Path,
) -> None:
    base_config, base_protocol = _base_inputs()
    runner_head = "c" * 40
    protocol, configs = launcher.build_confirmation_configs(
        base_config,
        base_protocol,
        _gate(tmp_path),
        runner_head=runner_head,
    )
    protocol = copy.deepcopy(protocol)
    protocol["runtime_config_root"] = str(tmp_path / "configs")
    protocol["run_root"] = str(tmp_path / "runs")
    for config in configs:
        seed = int(config["training_seed"])
        config["output_root"] = str(tmp_path / "runs" / f"seed_{seed}")
    manifest = launcher.materialize_config_package(
        configs, protocol, runner_head=runner_head
    )
    paths = launcher.validate_manifest(manifest, runner_head=runner_head)
    assert set(paths) == set(launcher.CONFIRMATION_SEEDS)
    assert manifest["schema_version"] == (
        "route_a_v3_route2_xeditcritic_v4_confirmation_config_manifest.v1"
    )
    assert manifest["required_run_ids"] == ["v4_full", "c0_v4"]
    assert manifest["confirmation_runner_git_head"] == runner_head
    assert manifest["training_git_head"] == runner_head
    assert manifest["historical_full_git_head"] == (
        launcher.REPAIRED_SCREEN_PROVENANCE_GIT_HEAD
    )
    assert manifest["audited_successor_semantic_baseline_git_head"] == (
        launcher.TRAINING_SEMANTICS_AUDITED_SUCCESSOR_BASELINE_HEAD
    )
    assert manifest["screen_gate_path"] == str(
        launcher.cross_root_gate_path(runner_head)
    )
    loaded = load_critic_confirmation_configs_v4(
        manifest,
        runtime_config_root=Path(protocol["runtime_config_root"]),
        run_root=Path(protocol["run_root"]),
    )
    assert set(loaded) == set(launcher.CONFIRMATION_SEEDS)


def test_authorization_is_trainer_and_posttraining_compatible(
    tmp_path: Path,
) -> None:
    gate = _gate(tmp_path, control_head="d" * 40)
    source = launcher.load_and_validate_source_authorizations(gate)
    runner_head = "d" * 40
    receipt = _runner_receipt(runner_head)
    authorization = launcher.build_confirmation_authorization(
        gate,
        source,
        receipt,
        _training_semantics(runner_head),
        runner_head=runner_head,
        runner_verification_receipt_path_value=(
            launcher.runner_verification_receipt_path(runner_head)
        ),
    )
    assert authorization["schema_version"] == (
        "route_a_v3_route2_xeditcritic_v4_"
        "confirmation_launch_authorization.v1"
    )
    assert authorization["status"] == (
        "XEDITCRITIC_V4_CONFIRMATION_LAUNCH_AUTHORIZED"
    )
    assert authorization["authorized_git_head"] == runner_head
    assert authorization["authorized_seeds"] == list(
        launcher.CONFIRMATION_SEEDS
    )
    assert authorization["authorized_run_ids"] == ["v4_full", "c0_v4"]
    assert authorization["training_git_head"] == runner_head
    assert authorization["historical_full_git_head"] == (
        launcher.REPAIRED_SCREEN_PROVENANCE_GIT_HEAD
    )
    assert authorization[
        "training_semantics_unchanged_since_audited_successor_baseline"
    ] is True
    assert "training_semantics_unchanged_from_repaired_screen" not in (
        authorization
    )
    assert authorization["runner_current_head_verification"][
        "runner_git_head"
    ] == runner_head
    assert all(
        authorization["barriers"].get(key) is True
        for key in (
            "screen_gate_passed",
            "a100_current_head_focused_tests_passed",
            "a100_current_head_v332_tests_passed",
            "bottom_six_cache_terminal_complete",
            "formal_parameter_preflight_passed",
            "formal_memory_preflight_passed",
        )
    )


def test_authorization_rejects_historical_full_as_new_training_head(
    tmp_path: Path,
) -> None:
    runner_head = "d" * 40
    gate = _gate(tmp_path, control_head=runner_head)
    semantics = _training_semantics(runner_head)
    semantics["training_git_head"] = (
        launcher.REPAIRED_SCREEN_PROVENANCE_GIT_HEAD
    )
    with pytest.raises(Exception, match="baseline or runner roles changed"):
        launcher.build_confirmation_authorization(
            gate,
            launcher.load_and_validate_source_authorizations(gate),
            _runner_receipt(runner_head),
            semantics,
            runner_head=runner_head,
            runner_verification_receipt_path_value=(
                launcher.runner_verification_receipt_path(runner_head)
            ),
        )

    old_audit_semantics = _training_semantics(runner_head)
    old_audit_semantics["audited_successor_semantic_baseline_audit"] = str(
        launcher.WORKTREE
        / (
            "audits/route_a_v3_route2_xeditcritic_v403_confirmation_"
            "training_semantics_reaudit_"
            "eba5b17431cb8e19202e5ea788fd419338da2d66.json"
        )
    )
    with pytest.raises(Exception, match="baseline or runner roles changed"):
        launcher.build_confirmation_authorization(
            gate,
            launcher.load_and_validate_source_authorizations(gate),
            _runner_receipt(runner_head),
            old_audit_semantics,
            runner_head=runner_head,
            runner_verification_receipt_path_value=(
                launcher.runner_verification_receipt_path(runner_head)
            ),
        )


def test_schedule_is_exact_three_seeds_two_arms_on_fixed_gpu_zero_to_five(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = {}
    for seed in launcher.CONFIRMATION_SEEDS:
        path = tmp_path / f"seed_{seed}.json"
        _write_json(path, {"output_root": str(tmp_path / f"run_{seed}")})
        configs[seed] = path
    runner_head = "e" * 40
    schedule = launcher.build_schedule(
        configs,
        tmp_path / "authorization.json",
        tmp_path / "manifest.json",
        _diagnostics(),
        _probes(),
        runner_head=runner_head,
        runtime_manifest=tmp_path / "runtime.json",
        log_root=tmp_path / "logs",
        preflight_peak_allocated_gib=14.9,
    )
    jobs = [queue["jobs"][0] for queue in schedule["gpu_queues"]]
    assert schedule["training_git_head"] == runner_head
    assert schedule["historical_full_git_head"] == (
        launcher.REPAIRED_SCREEN_PROVENANCE_GIT_HEAD
    )
    assert schedule["audited_successor_semantic_baseline_git_head"] == (
        launcher.TRAINING_SEMANTICS_AUDITED_SUCCESSOR_BASELINE_HEAD
    )
    assert [queue["physical_gpu_index"] for queue in schedule["gpu_queues"]] == list(
        launcher.PHYSICAL_GPUS
    )
    assert [
        (job["training_seed"], job["run_id"]) for job in jobs
    ] == [
        (20260908, "v4_full"),
        (20260908, "c0_v4"),
        (20260909, "v4_full"),
        (20260909, "c0_v4"),
        (20260910, "v4_full"),
        (20260910, "c0_v4"),
    ]
    for queue, job in zip(schedule["gpu_queues"], jobs, strict=True):
        command = job["command"]
        gpu_flag = command.index("--physical-gpu-index")
        assert command.count("--physical-gpu-index") == 1
        assert command[gpu_flag + 1] == str(queue["physical_gpu_index"])
        assert command[gpu_flag + 2] == "--launch-authorization"
        assert len(command) == 10
    assert schedule["free_memory_gate_applied"] is False
    assert schedule["gpu_sort_applied"] is False
    assert all(
        row["free_memory_mib"] == 0
        for row in schedule["gpu_memory_diagnostics_before_launch"].values()
    )

    runtime = {
        "status": "V4_CONFIRMATION_TRAINING_ALL_JOBS_TERMINAL",
        "git_head": runner_head,
        "eligible_components": ["critic"],
        "jobs": {
            job["job_key"]: {"terminal_artifact_kind": "SUMMARY"}
            for job in jobs
        },
        "active_performance_output_read": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    assert posttraining.validate_confirmation_runtime(
        runtime, head=runner_head
    ) == ("critic",)

    import scripts.route_a_v3.train_route2_xeditcritic_v4 as trainer

    parsed: list[dict] = []

    def fake_run(config, **kwargs):
        parsed.append({"config": config, **kwargs})
        return {"status": "ARGPARSE_ACCEPTED"}

    monkeypatch.setattr(trainer, "run", fake_run)
    for job in jobs:
        monkeypatch.setattr(
            sys, "argv", [str(launcher.TRAINER), *job["command"][2:]]
        )
        trainer.main()
    assert len(parsed) == 6
    assert [row["physical_gpu_index"] for row in parsed] == list(
        launcher.PHYSICAL_GPUS
    )


def test_one_shot_receipt_rejects_duplicate_before_any_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "attempt.json"
    attempt.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "ATTEMPT_RECEIPT", attempt)
    protocol = {
        "runtime_config_root": str(tmp_path / "configs"),
        "run_root": str(tmp_path / "runs"),
    }
    with pytest.raises(Exception, match="canonical confirmation attempt"):
        launcher.ensure_one_shot_targets_absent(
            protocol=protocol,
            authorization_root=tmp_path / "authorization",
            runtime_root=tmp_path / "runtime",
            log_root=tmp_path / "logs",
        )
    assert not (tmp_path / "configs").exists()
    assert not (tmp_path / "runtime").exists()


def test_launcher_records_memory_but_never_gates_or_sorts_by_it() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    assert '"free_memory_gate_applied": False' in source
    assert '"gpu_sort_applied": False' in source
    assert "required_free_memory" not in source
    assert "free_memory_mib >=" not in source
    assert "sorted(diagnostics" not in source
    assert "torch.cuda.is_available()" in source
    assert "torch.cuda.is_bf16_supported()" in source
    assert "CUDA_BF16_PROBE_SILENT_CPU_FALLBACK" in source
    assert "STOPPED_BEFORE_CONFIRMATION_LAUNCH_CUDA_FAILURE" in source


def test_gate_and_provenance_validation_precede_one_shot_write_and_launch() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    run_source = source[source.index("def run(current_head: str)") :]
    gate_validation = run_source.index("validate_cross_root_gate(gate)")
    provenance_validation = run_source.index(
        "load_and_validate_source_authorizations(gate)"
    )
    attempt_write = run_source.index(
        "write_new_atomic(\n        ATTEMPT_RECEIPT"
    )
    scheduler_launch = run_source.index(
        "process = spawn_scheduler_with_failure_evidence("
    )
    assert gate_validation < provenance_validation < attempt_write < scheduler_launch
