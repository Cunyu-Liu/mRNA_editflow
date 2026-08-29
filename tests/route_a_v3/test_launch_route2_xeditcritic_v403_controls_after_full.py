from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import scripts.route_a_v3.launch_route2_xeditcritic_v403_controls_after_full as launcher
import scripts.route_a_v3.run_route2_xeditcritic_v403_control_recovery_scheduler as scheduler
from scripts.route_a_v3 import (
    transition_adjudicate_route2_xeditcritic_v403_cross_root_screen
    as cross_root_transition,
)


def test_control_scheduler_popen_failure_is_durable_and_one_shot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    family = tmp_path / "control_family"
    family.mkdir()
    schedule = family / "schedule.json"
    authorization = family / "authorization.json"
    schedule.write_text("{}\n", encoding="utf-8")
    authorization.write_text("{}\n", encoding="utf-8")
    failure = family / "scheduler_launch.failed.json"
    launch = family / "launch.json"

    def fail(*args, **kwargs):
        raise OSError("control scheduler spawn failed")

    monkeypatch.setattr(launcher.subprocess, "Popen", fail)
    arguments = {
        "failure_path": failure,
        "expected_head": "a" * 40,
        "command_line": ["python", "scheduler.py", "--schedule", str(schedule)],
        "schedule_path": schedule,
        "runtime_path": family / "runtime.json",
        "worker_log_path": family / "scheduler.log",
        "created_artifacts": {
            "launch_authorization": authorization,
            "schedule": schedule,
        },
    }
    with pytest.raises(
        launcher.XEditCriticV403ControlRecoveryLaunchError,
        match="durable technical failure",
    ):
        launcher.spawn_scheduler_with_failure_evidence(**arguments)

    payload = json.loads(failure.read_text(encoding="utf-8"))
    assert payload["status"] == (
        "XEDITCRITIC_V403_CONTROL_SCHEDULER_LAUNCH_TECHNICAL_FAILURE"
    )
    assert payload["error_type"] == "OSError"
    assert payload["scheduler_started"] is False
    assert payload["gpu_job_started"] is False
    assert payload["development_test_outcome_reads"] == 0
    assert payload["created_artifact_paths"]["schedule"] == str(schedule)
    assert "scheduler_pid" not in payload
    assert not launch.exists()

    with pytest.raises(
        launcher.XEditCriticV403ControlRecoveryLaunchError,
        match="already exists",
    ):
        launcher.spawn_scheduler_with_failure_evidence(**arguments)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _full_summary(
    authorization_path: Path,
    *,
    physical_gpu_index: int = 3,
    protected_reads: int = 0,
) -> dict:
    summary = {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_screen_run.v1",
        "status": "TERMINAL_XEDITCRITIC_V4_SCREEN_RUN_COMPLETE",
        "run_id": "v4_full",
        "model_kind": "V4-FULL",
        "precision": "BF16_FORWARD_FP32_EFFECTIVE_OBJECTIVE",
        "cpu_fallback_used": False,
        "parameter_changed": True,
        "physical_gpu_index": physical_gpu_index,
        "launch_authorization_path": str(authorization_path),
        "development_test_outcome_reads": protected_reads,
        "new_final_evaluation_outcome_reads": 0,
    }
    summary.update(launcher.FROZEN_FULL_SUMMARY_IDENTITY)
    return summary


def _full_runtime(*, physical_gpu_index: int = 3) -> dict:
    return {
        "schema_version": (
            "route_a_v3_route2_xeditcritic_v403_full_recovery_runtime.v1"
        ),
        "status": "XEDITCRITIC_V403_FULL_RECOVERY_TERMINAL",
        "terminal_artifact_kind": "SUMMARY",
        "return_code": 0,
        "run_id": "v4_full",
        "git_head": launcher.HISTORICAL_FULL_GIT_HEAD,
        "physical_gpu_index": physical_gpu_index,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def _full_authorization(
    *, physical_gpu_index: int = 3, protected_reads: int = 0
) -> dict:
    return {
        "schema_version": (
            "route_a_v3_route2_xeditcritic_v4_screen_launch_authorization.v1"
        ),
        "status": "XEDITCRITIC_V4_SCREEN_LAUNCH_AUTHORIZED",
        "authorized_git_head": launcher.HISTORICAL_FULL_GIT_HEAD,
        "authorized_run_ids": list(launcher.ALL_RUN_IDS),
        "v403_rng_replay_recovery": {
            "run_id": "v4_full",
            "physical_gpu_index": physical_gpu_index,
        },
        "development_test_outcome_reads": protected_reads,
        "new_final_evaluation_outcome_reads": 0,
    }


def _inventory() -> list[dict]:
    return [
        {
            "physical_gpu_index": index,
            "device_name": "NVIDIA A100-PCIE-40GB",
            "bf16_supported": True,
            "bf16_tensor_probe": True,
        }
        for index in launcher.PHYSICAL_GPU_INDICES
    ]


def _runner_receipt(head: str) -> dict:
    group_counts = [25, 25, 25, 25, 25, 25, 25, 28]
    return {
        "schema_version": launcher.RUNNER_VERIFICATION_RECEIPT_SCHEMA,
        "status": launcher.RUNNER_VERIFICATION_RECEIPT_PASS,
        "runner_git_head": head,
        "worktree_clean": True,
        "focused_tests": {
            "isolated_process_groups": True,
            "command": [
                "python -m pytest " + " ".join(markers)
                for markers in launcher.FOCUSED_GROUP_REQUIRED_TEST_MARKERS
            ],
            "group_passed_counts": group_counts,
            "passed": True,
            "passed_count": sum(group_counts),
            "failed_count": 0,
        },
        "v332_tests": {
            "command": ["python -m pytest tests/route_a_v3/*v332*.py"],
            "passed": True,
            "passed_count": 96,
            "failed_count": 0,
        },
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def _prior_oom_terminal_receipt() -> dict:
    return {
        "schema_version": launcher.PRIOR_CONTROL_OOM_TERMINAL_RECEIPT_SCHEMA,
        "status": launcher.PRIOR_CONTROL_OOM_TERMINAL_RECEIPT_STATUS,
        "terminal_class": "TECHNICAL_FAILURE_TERMINAL",
        "old_current_git_head": launcher.PRIOR_FAILED_CONTROL_GIT_HEAD,
        "old_runner_git_head": launcher.PRIOR_FAILED_CONTROL_GIT_HEAD,
        "old_orchestration_git_head": launcher.PRIOR_FAILED_CONTROL_GIT_HEAD,
        "old_training_code_git_head": launcher.PRIOR_FAILED_CONTROL_GIT_HEAD,
        "old_runtime_path": str(launcher.PRIOR_FAILED_CONTROL_RUNTIME),
        "old_runtime_status": (
            "XEDITCRITIC_V403_CONTROL_RECOVERY_TECHNICAL_FAILURE"
        ),
        "scheduler_pid": 123,
        "scheduler_process_gone": True,
        "ordered_control_run_ids": list(launcher.CONTROL_RUN_IDS),
        "terminal_jobs": list(launcher.CONTROL_RUN_IDS),
        "technical_failure_run_ids": list(launcher.CONTROL_RUN_IDS[:3]),
        "terminal_summary_run_ids": list(launcher.CONTROL_RUN_IDS[3:]),
        "first_terminal_failure": {
            "run_id": launcher.CONTROL_RUN_IDS[0],
            "reason": "JOB_TERMINAL_FAILURE_ARTIFACT",
        },
        "cross_root_adjudication_run": False,
        "cross_root_gate_path": "/absent/screen_gate.json",
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


def _schedule(tmp_path: Path, *, head: str = "a" * 40) -> dict:
    return launcher.build_control_schedule(
        current_head=head,
        config_path=tmp_path / "screen_config.json",
        authorization_path=tmp_path / "authorization.json",
        cuda_bf16_inventory=_inventory(),
        output_root=tmp_path / "fresh_controls",
        runtime_manifest=tmp_path / "runtime.json",
        log_root=tmp_path / "logs",
        transition_gate=tmp_path / "cross_root/screen_gate.json",
        prior_control_oom_terminal_receipt_path=(
            launcher.PRIOR_CONTROL_OOM_TERMINAL_RECEIPT
        ),
    )


def test_invalid_tracked_full_audit_stops_without_historical_payload_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"probe": 0, "spawn": 0}
    prior_receipt_path = tmp_path / "prior_oom_terminal_receipt.json"
    _write_json(prior_receipt_path, _prior_oom_terminal_receipt())
    monkeypatch.setattr(
        launcher, "PRIOR_CONTROL_OOM_TERMINAL_RECEIPT", prior_receipt_path
    )

    monkeypatch.setattr(
        launcher,
        "validate_historical_full_terminal_audit",
        lambda: (_ for _ in ()).throw(
            launcher.XEditCriticV403ControlRecoveryLaunchError(
                "tracked historical full terminal audit identity is invalid"
            )
        ),
    )
    monkeypatch.setattr(
        launcher,
        "validate_current_full_terminal",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("canonical launch must not read historical full payload")
        ),
    )
    monkeypatch.setattr(
        launcher,
        "validate_historical_c0_terminal",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("canonical launch must not read historical C0 payload")
        ),
    )

    def forbidden_probe() -> list[dict]:
        calls["probe"] += 1
        raise AssertionError("CUDA probe must not run before full terminal")

    def forbidden_spawn(*args, **kwargs):
        calls["spawn"] += 1
        raise AssertionError("worker must not spawn before full terminal")

    monkeypatch.setattr(launcher, "probe_cuda_bf16", forbidden_probe)
    monkeypatch.setattr(launcher.subprocess, "Popen", forbidden_spawn)

    with pytest.raises(Exception, match="tracked historical full terminal audit"):
        launcher.launch(
            "a" * 40,
            prior_terminal_receipt_path=prior_receipt_path,
        )

    assert calls == {"probe": 0, "spawn": 0}
    assert not (tmp_path / "controls").exists()
    assert not (tmp_path / "runner").exists()
    assert not (tmp_path / "auth").exists()
    assert not (tmp_path / "gate").exists()


@pytest.mark.parametrize("receipt_exists", [False, True])
def test_prior_oom_receipt_is_first_admission_evidence_and_has_no_side_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt_exists: bool,
) -> None:
    receipt_path = tmp_path / "canonical_oom_terminal_receipt.json"
    if receipt_exists:
        receipt = _prior_oom_terminal_receipt()
        receipt["new_independent_retry_eligible"] = False
        _write_json(receipt_path, receipt)
    monkeypatch.setattr(
        launcher, "PRIOR_CONTROL_OOM_TERMINAL_RECEIPT", receipt_path
    )

    def forbidden(label: str):
        def fail(*args, **kwargs):
            raise AssertionError(f"{label} ran before prior OOM receipt admission")

        return fail

    monkeypatch.setattr(
        launcher,
        "validate_historical_full_terminal_audit",
        forbidden("historical full audit"),
    )
    monkeypatch.setattr(
        launcher, "control_family_paths", forbidden("family path creation")
    )
    monkeypatch.setattr(
        launcher, "validate_training_source", forbidden("current-HEAD preflight")
    )
    monkeypatch.setattr(
        launcher, "physical_gpu_inventory", forbidden("GPU inventory")
    )
    monkeypatch.setattr(
        launcher, "probe_cuda_bf16", forbidden("CUDA/BF16 probe")
    )
    monkeypatch.setattr(
        launcher.subprocess, "Popen", forbidden("scheduler launch")
    )
    before = set(tmp_path.iterdir())

    with pytest.raises(Exception):
        launcher.launch(
            "a" * 40,
            prior_terminal_receipt_path=receipt_path,
        )

    assert set(tmp_path.iterdir()) == before


def test_tracked_full_terminal_audit_is_strict_and_payload_closed(
    tmp_path: Path,
) -> None:
    payload = json.loads(
        launcher.FULL_TERMINAL_AUDIT.read_text(encoding="utf-8")
    )
    audit_path = tmp_path / "full_terminal_audit.json"
    _write_json(audit_path, payload)
    assert launcher.validate_historical_full_terminal_audit(audit_path)[
        "status"
    ] == "XEDITCRITIC_V403_FULL_TERMINAL_SUMMARY_RECORDED"

    payload["terminal_facts"]["cuda_used"] = False
    _write_json(audit_path, payload)
    with pytest.raises(Exception, match="terminal facts are invalid"):
        launcher.validate_historical_full_terminal_audit(audit_path)


def test_full_terminal_requires_zero_protected_reads(tmp_path: Path) -> None:
    output_root = tmp_path / "full"
    authorization = tmp_path / "authorization.json"
    _write_json(
        authorization,
        _full_authorization(),
    )
    _write_json(
        output_root / "v4_full/run_summary.json",
        _full_summary(authorization, protected_reads=1),
    )
    runtime = tmp_path / "runtime.json"
    _write_json(runtime, _full_runtime())

    with pytest.raises(Exception, match="Development TEST read"):
        launcher.validate_current_full_terminal(output_root, runtime)


def test_full_terminal_accepts_consistent_non_gpu5_frozen_identity(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "full"
    authorization_path = tmp_path / "authorization.json"
    _write_json(
        authorization_path,
        _full_authorization(physical_gpu_index=3),
    )
    _write_json(
        output_root / "v4_full/run_summary.json",
        _full_summary(authorization_path, physical_gpu_index=3),
    )
    runtime_path = tmp_path / "runtime.json"
    _write_json(runtime_path, _full_runtime(physical_gpu_index=3))

    result = launcher.validate_current_full_terminal(output_root, runtime_path)

    assert result["physical_gpu_index"] == 3


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("seed", 20260908),
        ("pass_count", 7),
        ("selected_pass", 7),
        ("update_count", 22415),
        ("selection_policy", "VALIDATION_PEAK_RESELECTION"),
        ("train_record_count", 89579),
        ("validation_record_count", 18292),
        ("effective_batch_size", 16),
        ("physical_batch_size", 16),
    ],
)
def test_full_terminal_rejects_non_frozen_training_identity(
    tmp_path: Path, field: str, invalid_value: object
) -> None:
    output_root = tmp_path / "full"
    authorization_path = tmp_path / "authorization.json"
    _write_json(authorization_path, _full_authorization())
    summary = _full_summary(authorization_path)
    summary[field] = invalid_value
    _write_json(output_root / "v4_full/run_summary.json", summary)
    runtime_path = tmp_path / "runtime.json"
    _write_json(runtime_path, _full_runtime())

    with pytest.raises(Exception, match="frozen training identity"):
        launcher.validate_current_full_terminal(output_root, runtime_path)


def test_full_terminal_rejects_gpu_disagreement_or_unauthorized_run(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "full"
    authorization_path = tmp_path / "authorization.json"
    authorization = _full_authorization(physical_gpu_index=3)
    _write_json(authorization_path, authorization)
    _write_json(
        output_root / "v4_full/run_summary.json",
        _full_summary(authorization_path, physical_gpu_index=5),
    )
    runtime_path = tmp_path / "runtime.json"
    _write_json(runtime_path, _full_runtime(physical_gpu_index=5))
    with pytest.raises(Exception, match="physical GPUs disagree"):
        launcher.validate_current_full_terminal(output_root, runtime_path)

    authorization["v403_rng_replay_recovery"]["physical_gpu_index"] = 5
    authorization["authorized_run_ids"].remove("v4_full")
    _write_json(authorization_path, authorization)
    with pytest.raises(Exception, match="launch authorization identity"):
        launcher.validate_current_full_terminal(output_root, runtime_path)


def test_schedule_is_exactly_six_fresh_controls_from_current_head(
    tmp_path: Path,
) -> None:
    head = "a" * 40
    schedule = _schedule(tmp_path, head=head)

    assert [job["run_id"] for job in schedule["jobs"]] == list(
        launcher.CONTROL_RUN_IDS
    )
    assert not {"c0_v4", "v4_full"} & {
        job["run_id"] for job in schedule["jobs"]
    }
    assert [job["physical_gpu_index"] for job in schedule["jobs"]] == list(
        launcher.PHYSICAL_GPU_INDICES
    )
    assert [job["wave_index"] for job in schedule["jobs"]] == [0, 0, 0, 1, 1, 1]
    assert schedule["control_waves"] == [
        list(launcher.CONTROL_RUN_IDS[:3]),
        list(launcher.CONTROL_RUN_IDS[3:]),
    ]
    assert schedule["wave1_requires_wave0_all_summaries"] is True
    assert schedule["retry_ordinal"] == 1
    assert schedule["retry_identity"] == launcher.CONTROL_RETRY_IDENTITY
    assert schedule["prior_family_reused"] is False
    assert schedule["all_six_controls_retrained"] is True
    assert all(
        job["process_environment"]
        == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        for job in schedule["jobs"]
    )
    assert all(
        launcher.CONTROL_RETRY_IDENTITY in job["training_attempt_id"]
        for job in schedule["jobs"]
    )
    assert all(
        job["command"][:2] == [str(launcher.PYTHON), str(launcher.TRAINER)]
        for job in schedule["jobs"]
    )
    assert all(
        str(tmp_path / "fresh_controls")
        in str(job["output_directory"])
        for job in schedule["jobs"]
    )
    assert schedule["historical_full_git_head"] == launcher.HISTORICAL_FULL_GIT_HEAD
    assert schedule["historical_c0_git_head"] == launcher.HISTORICAL_C0_GIT_HEAD
    assert schedule["current_git_head"] == head
    assert schedule["runner_git_head"] == head
    assert schedule["training_code_git_head"] == head
    assert schedule["orchestration_git_head"] == head
    assert all(job["training_git_head"] == head for job in schedule["jobs"])
    assert schedule["cross_root_gate"] == str(
        tmp_path / "cross_root/screen_gate.json"
    )
    assert schedule["full_retrained"] is False
    assert schedule["c0_retrained"] is False
    assert schedule["old_v402_stopped_process_resumed"] is False
    assert schedule["development_test_outcome_reads"] == 0
    assert schedule["new_final_evaluation_outcome_reads"] == 0
    scheduler.validate_schedule(schedule)


def test_schedule_rejects_missing_control_or_full_injection(tmp_path: Path) -> None:
    schedule = _schedule(tmp_path)
    schedule["jobs"] = schedule["jobs"][:-1]
    with pytest.raises(Exception, match="exact six-control package"):
        scheduler.validate_schedule(schedule)

    schedule = _schedule(tmp_path)
    schedule["jobs"][0]["run_id"] = "v4_full"
    with pytest.raises(Exception, match="six-control package|retrain"):
        scheduler.validate_schedule(schedule)


def test_schedule_rejects_wave_or_allocator_drift(tmp_path: Path) -> None:
    schedule = _schedule(tmp_path)
    schedule["control_waves"] = [list(launcher.CONTROL_RUN_IDS)]
    with pytest.raises(Exception, match="wave policy"):
        scheduler.validate_schedule(schedule)

    schedule = _schedule(tmp_path)
    schedule["jobs"][0]["process_environment"] = {}
    with pytest.raises(Exception, match="allocator binding"):
        scheduler.validate_schedule(schedule)


def test_current_head_family_paths_and_v2_config_are_exact(tmp_path: Path) -> None:
    head = "b" * 40
    paths = launcher.control_family_paths(head)
    assert paths["output_root"].name == (
        f"screen_seed_20260907_v403_control_recovery_retry1_{head}"
    )
    assert paths["runtime_root"].name == (
        f"v403_control_recovery_retry1_runner_{head}"
    )
    assert paths["transition_gate"].parent.name == (
        f"screen_seed_20260907_v403_cross_root_controls_retry1_{head}"
    )
    base = {
        "required_screen_runs": [
            {"run_id": run_id} for run_id in launcher.ALL_RUN_IDS
        ],
        "output_root": "old-output",
        "screen_gate_output": "old-gate",
        "scientific_field": "unchanged",
    }
    config = launcher.build_recovery_config(
        base,
        current_head=head,
        output_root=tmp_path / "controls",
        screen_gate_output=tmp_path / "gate.json",
    )
    assert config["runner_git_head"] == head
    assert config["scientific_field"] == "unchanged"


def test_authorization_consumes_exact_current_head_runner_receipt() -> None:
    head = "c" * 40
    receipt = _runner_receipt(head)
    receipt_path = launcher.runner_verification_receipt_path(head)
    authorization = launcher.build_launch_authorization(
        {"git_head": "preflight-head"},
        current_head=head,
        prior_control_oom_terminal_receipt=_prior_oom_terminal_receipt(),
        prior_control_oom_terminal_receipt_path=(
            launcher.PRIOR_CONTROL_OOM_TERMINAL_RECEIPT
        ),
        historical_full_terminal_audit={
            "status": "XEDITCRITIC_V403_FULL_TERMINAL_SUMMARY_RECORDED"
        },
        historical_full_terminal_audit_path=launcher.FULL_TERMINAL_AUDIT,
        runner_verification_receipt=receipt,
        runner_verification_receipt_path=receipt_path,
    )
    assert authorization["authorized_git_head"] == head
    assert authorization["runner_verification_receipt"]["path"] == str(
        receipt_path
    )
    assert authorization["runner_verification_receipt"]["focused_tests"][
        "group_passed_counts"
    ] == receipt["focused_tests"]["group_passed_counts"]
    assert authorization["v403_control_recovery"]["historical_full_git_head"] == (
        launcher.HISTORICAL_FULL_GIT_HEAD
    )
    assert authorization["v403_control_recovery"]["current_git_head"] == head
    assert authorization["v403_control_recovery"]["retry_ordinal"] == 1
    assert authorization["v403_control_recovery"]["prior_family_reused"] is False
    assert authorization["prior_control_oom_terminal_receipt"]["path"] == str(
        launcher.PRIOR_CONTROL_OOM_TERMINAL_RECEIPT
    )

    wrong = _runner_receipt("d" * 40)
    with pytest.raises(Exception, match="exact-HEAD clean PASS"):
        launcher.validate_runner_verification_receipt(
            wrong, current_head=head, receipt_path=receipt_path
        )


def test_retry_requires_canonical_non_authorizing_old_oom_terminal_receipt(
    tmp_path: Path,
) -> None:
    receipt = _prior_oom_terminal_receipt()
    launcher.validate_prior_control_oom_terminal_receipt(
        receipt,
        receipt_path=launcher.PRIOR_CONTROL_OOM_TERMINAL_RECEIPT,
    )

    receipt["same_family_retry_authorized"] = True
    with pytest.raises(Exception, match="independent retry"):
        launcher.validate_prior_control_oom_terminal_receipt(
            receipt,
            receipt_path=launcher.PRIOR_CONTROL_OOM_TERMINAL_RECEIPT,
        )

    receipt = _prior_oom_terminal_receipt()
    with pytest.raises(Exception, match="not canonical"):
        launcher.validate_prior_control_oom_terminal_receipt(
            receipt,
            receipt_path=tmp_path / "copied_receipt.json",
        )


def test_launcher_and_scheduler_have_no_free_memory_launch_gate() -> None:
    sources = [
        Path(launcher.__file__).read_text(encoding="utf-8"),
        Path(scheduler.__file__).read_text(encoding="utf-8"),
    ]
    forbidden = (
        "memory.free",
        "required_free_memory",
        "launch_required_free_memory",
        "peak plus 2 GiB",
    )
    for source in sources:
        assert '"free_memory_gate_applied": False' in source
        assert not any(text in source for text in forbidden)
    assert "SIGCONT" not in sources[0] + sources[1]
    assert "os.kill" not in sources[0] + sources[1]


@pytest.mark.parametrize(
    ("result", "reason", "missing"),
    [
        (
            subprocess.CompletedProcess(
                launcher.GPU_INVENTORY_COMMAND,
                6,
                stdout="partial",
                stderr="driver",
            ),
            "NONZERO_RETURN_CODE",
            (),
        ),
        (
            subprocess.CompletedProcess(
                launcher.GPU_INVENTORY_COMMAND,
                0,
                stdout="not-an-index\n",
                stderr="",
            ),
            "OUTPUT_PARSE_FAILED",
            (),
        ),
        (
            subprocess.CompletedProcess(
                launcher.GPU_INVENTORY_COMMAND,
                0,
                stdout="0\n1\n2\n3\n4\n",
                stderr="",
            ),
            "PHYSICAL_GPU_INVENTORY_INCOMPLETE",
            (5,),
        ),
    ],
)
def test_controls_inventory_failures_are_structured(
    monkeypatch: pytest.MonkeyPatch,
    result: subprocess.CompletedProcess[str],
    reason: str,
    missing: tuple[int, ...],
) -> None:
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: result)
    with pytest.raises(launcher.XEditCriticV403GpuInventoryError) as captured:
        launcher.physical_gpu_inventory()
    assert captured.value.reason == reason
    assert captured.value.return_code == result.returncode
    assert captured.value.stdout == result.stdout
    assert captured.value.stderr == result.stderr
    assert captured.value.missing_physical_gpus == missing


def test_controls_inventory_execution_failure_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args, **kwargs):
        raise OSError("nvidia-smi absent")

    monkeypatch.setattr(launcher.subprocess, "run", fail)
    with pytest.raises(launcher.XEditCriticV403GpuInventoryError) as captured:
        launcher.physical_gpu_inventory()
    assert captured.value.reason == "COMMAND_EXECUTION_FAILED"
    assert captured.value.return_code is None
    assert captured.value.command_line == launcher.GPU_INVENTORY_COMMAND


def _patch_controls_prelaunch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    prior_receipt_path = tmp_path / "prior_oom_terminal_receipt.json"
    _write_json(prior_receipt_path, _prior_oom_terminal_receipt())
    monkeypatch.setattr(
        launcher, "PRIOR_CONTROL_OOM_TERMINAL_RECEIPT", prior_receipt_path
    )
    monkeypatch.setattr(
        launcher,
        "control_family_paths",
        lambda head: {
            "output_root": tmp_path / "controls",
            "runtime_root": tmp_path / "runner",
            "authorization_root": tmp_path / "authorization",
            "log_root": tmp_path / "logs",
            "transition_gate": tmp_path / "gate/screen_gate.json",
        },
    )
    monkeypatch.setattr(
        launcher,
        "validate_historical_full_terminal_audit",
        lambda: {"status": "XEDITCRITIC_V403_FULL_TERMINAL_SUMMARY_RECORDED"},
    )
    monkeypatch.setattr(
        launcher,
        "validate_current_full_terminal",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("canonical launch must not read historical full payload")
        ),
    )
    monkeypatch.setattr(
        launcher,
        "validate_historical_c0_terminal",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("canonical launch must not read historical C0 payload")
        ),
    )
    monkeypatch.setattr(
        launcher,
        "validate_training_source",
        lambda expected_head: {"preflight": {}},
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Popen must not run after prelaunch failure")
        ),
    )
    return prior_receipt_path


def test_controls_inventory_failure_stops_before_probe_runtime_or_popen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prior_receipt_path = _patch_controls_prelaunch(tmp_path, monkeypatch)
    error = launcher.XEditCriticV403GpuInventoryError(
        "missing configured GPU",
        reason="PHYSICAL_GPU_INVENTORY_INCOMPLETE",
        return_code=0,
        stdout="0\n1\n2\n3\n4\n",
        missing_physical_gpus=(5,),
    )
    monkeypatch.setattr(
        launcher,
        "physical_gpu_inventory",
        lambda: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(
        launcher,
        "probe_cuda_bf16",
        lambda: (_ for _ in ()).throw(
            AssertionError("CUDA/BF16 probe must not run after inventory failure")
        ),
    )

    with pytest.raises(launcher.XEditCriticV403GpuInventoryError):
        launcher.launch(
            "d" * 40,
            prior_terminal_receipt_path=prior_receipt_path,
        )

    runtime_root = tmp_path / "runner"
    evidence_path = launcher.sibling_failure_path(runtime_root)
    assert not runtime_root.exists()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["schema_version"] == (
        "route_a_v3_route2_xeditcritic_prelaunch_failure.v1"
    )
    assert evidence["status"] == "XEDITCRITIC_PRELAUNCH_GPU_OR_CUDA_FAILURE"
    assert evidence["launcher"] == "controls"
    assert evidence["failure_stage"] == "INVENTORY"
    assert evidence["expected_head"] == "d" * 40
    assert evidence["runtime_root_created"] is False
    assert evidence["jobs_started"] == 0
    assert evidence["cpu_fallback_used"] is False
    assert evidence["free_memory_gate_applied"] is False
    assert evidence["automatic_retry_attempted"] is False
    assert evidence["development_test_outcome_reads"] == 0
    assert evidence["new_final_evaluation_outcome_reads"] == 0


def test_controls_cuda_probe_failure_preserves_observed_cpu_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prior_receipt_path = _patch_controls_prelaunch(tmp_path, monkeypatch)
    monkeypatch.setattr(
        launcher,
        "physical_gpu_inventory",
        lambda: launcher.PHYSICAL_GPU_INDICES,
    )
    error = launcher.XEditCriticV403CudaBf16ProbeError(
        "CPU fallback observed", cpu_fallback_used=True
    )
    monkeypatch.setattr(
        launcher,
        "probe_cuda_bf16",
        lambda: (_ for _ in ()).throw(error),
    )

    with pytest.raises(launcher.XEditCriticV403CudaBf16ProbeError):
        launcher.launch(
            "e" * 40,
            prior_terminal_receipt_path=prior_receipt_path,
        )

    runtime_root = tmp_path / "runner"
    evidence = json.loads(
        launcher.sibling_failure_path(runtime_root).read_text(encoding="utf-8")
    )
    assert not runtime_root.exists()
    assert evidence["failure_stage"] == "CUDA_BF16_PROBE"
    assert evidence["reason"] == "CUDA_BF16_PROBE_OUTPUT_INVALID"
    assert evidence["cpu_fallback_used"] is True
    assert evidence["jobs_started"] == 0


@pytest.mark.parametrize("partial", [False, True])
def test_controls_existing_failure_evidence_requires_new_family(
    tmp_path: Path, partial: bool
) -> None:
    runtime_root = tmp_path / "controls_runtime"
    failure = launcher.sibling_failure_path(runtime_root)
    path = failure.with_suffix(failure.suffix + ".partial") if partial else failure
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(Exception, match="new retry family"):
        launcher.require_fresh_prelaunch_family(runtime_root)


class _FakeProcess:
    def __init__(self, *, pid: int, return_code: int) -> None:
        self.pid = pid
        self._return_code = return_code

    def wait(self) -> int:
        return self._return_code


def _patch_scheduler_processes(
    schedule: dict,
    monkeypatch: pytest.MonkeyPatch,
    *,
    first_return_code: int = 0,
    double_terminal_first: bool = False,
    observed_environments: list[dict[str, str]] | None = None,
) -> list[str]:
    outputs = {
        job["run_id"]: Path(job["output_directory"])
        for job in schedule["jobs"]
    }
    launched: list[str] = []

    def popen(command, **kwargs):
        run_id = command[command.index("--run-id") + 1]
        launched.append(run_id)
        if observed_environments is not None:
            observed_environments.append(dict(kwargs["env"]))
        _write_json(outputs[run_id] / "run_summary.json", {"run_id": run_id})
        if double_terminal_first and len(launched) == 1:
            _write_json(outputs[run_id] / "failure.json", {"run_id": run_id})
        return _FakeProcess(
            pid=1000 + len(launched),
            return_code=first_return_code if len(launched) == 1 else 0,
        )

    monkeypatch.setattr(scheduler.subprocess, "Popen", popen)
    monkeypatch.setattr(scheduler, "inspect_worktree_identity", lambda *a, **k: None)
    return launched


def test_scheduler_all_six_require_zero_return_and_unique_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedule = _schedule(tmp_path)
    environments: list[dict[str, str]] = []
    launched = _patch_scheduler_processes(
        schedule,
        monkeypatch,
        observed_environments=environments,
    )

    scheduler.run(schedule)

    runtime = json.loads((tmp_path / "runtime.json").read_text(encoding="utf-8"))
    assert launched == list(launcher.CONTROL_RUN_IDS)
    assert runtime["status"] == (
        "XEDITCRITIC_V403_CONTROL_RECOVERY_ALL_SIX_SUMMARIES_TERMINAL"
    )
    assert runtime["first_terminal_failure"] is None
    assert runtime["cross_root_adjudication_run"] is False
    assert runtime["retry_ordinal"] == 1
    assert runtime["control_waves"] == [
        list(launcher.CONTROL_RUN_IDS[:3]),
        list(launcher.CONTROL_RUN_IDS[3:]),
    ]
    assert len(environments) == 6
    assert all(
        environment["PYTORCH_CUDA_ALLOC_CONF"]
        == "expandable_segments:True"
        for environment in environments
    )
    assert all(
        row["status"] == "TERMINAL_SUMMARY"
        and row["return_code"] == 0
        and row["terminal_artifact_kind"] == "SUMMARY"
        for row in runtime["jobs"].values()
    )
    accepted = cross_root_transition.validate_control_runtime(
        tmp_path / "runtime.json",
        expected_control_runner_head="a" * 40,
    )
    assert accepted["status"] == (
        "XEDITCRITIC_V403_CONTROL_RECOVERY_ALL_SIX_SUMMARIES_TERMINAL"
    )


@pytest.mark.parametrize(
    ("return_code", "double_terminal", "reason"),
    [
        (9, False, "JOB_NONZERO_RETURN_CODE"),
        (0, True, "JOB_DOUBLE_TERMINAL_ARTIFACT"),
    ],
)
def test_scheduler_summary_nonzero_or_double_terminal_is_technical_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    return_code: int,
    double_terminal: bool,
    reason: str,
) -> None:
    schedule = _schedule(tmp_path)
    _patch_scheduler_processes(
        schedule,
        monkeypatch,
        first_return_code=return_code,
        double_terminal_first=double_terminal,
    )

    scheduler.run(schedule)

    runtime = json.loads((tmp_path / "runtime.json").read_text(encoding="utf-8"))
    first = launcher.CONTROL_RUN_IDS[0]
    assert [
        run_id
        for run_id in launcher.CONTROL_RUN_IDS
        if runtime["jobs"][run_id].get("training_pid") is not None
    ] == list(launcher.CONTROL_RUN_IDS[:3])
    assert runtime["status"] == "XEDITCRITIC_V403_CONTROL_RECOVERY_TECHNICAL_FAILURE"
    assert runtime["jobs"][first]["status"] == "TECHNICAL_FAILURE"
    assert runtime["first_terminal_failure"]["run_id"] == first
    assert runtime["first_terminal_failure"]["reason"] == reason
    assert runtime["cross_root_adjudication_run"] is False
    assert all(
        runtime["jobs"][run_id]["status"]
        == "NOT_RUN_AFTER_TERMINAL_FAILURE"
        for run_id in launcher.CONTROL_RUN_IDS[3:]
    )


def test_scheduler_waits_for_wave0_before_launching_wave1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedule = _schedule(tmp_path)
    outputs = {
        job["run_id"]: Path(job["output_directory"])
        for job in schedule["jobs"]
    }
    events: list[str] = []

    class Process:
        def __init__(self, run_id: str, pid: int) -> None:
            self.run_id = run_id
            self.pid = pid

        def wait(self) -> int:
            events.append(f"wait:{self.run_id}")
            return 0

    def popen(command, **kwargs):
        run_id = command[command.index("--run-id") + 1]
        events.append(f"launch:{run_id}")
        _write_json(outputs[run_id] / "run_summary.json", {"run_id": run_id})
        return Process(run_id, 2000 + len(events))

    monkeypatch.setattr(scheduler.subprocess, "Popen", popen)
    monkeypatch.setattr(
        scheduler, "inspect_worktree_identity", lambda *a, **k: None
    )

    scheduler.run(schedule)

    first_wave = list(launcher.CONTROL_RUN_IDS[:3])
    second_wave = list(launcher.CONTROL_RUN_IDS[3:])
    assert events[:3] == [f"launch:{run_id}" for run_id in first_wave]
    assert events[3:6] == [f"wait:{run_id}" for run_id in first_wave]
    assert events[6:9] == [f"launch:{run_id}" for run_id in second_wave]
    assert events[9:12] == [f"wait:{run_id}" for run_id in second_wave]


def test_scheduler_worktree_drift_stops_later_popen_and_marks_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedule = _schedule(tmp_path)
    launched = _patch_scheduler_processes(schedule, monkeypatch)
    inspections = {"count": 0}

    def inspect(*args, **kwargs):
        inspections["count"] += 1
        if inspections["count"] == 1:
            return None
        return {
            "reason": "WORKTREE_HEAD_MISMATCH",
            "expected_git_head": "a" * 40,
            "observed_git_head": "b" * 40,
        }

    monkeypatch.setattr(scheduler, "inspect_worktree_identity", inspect)

    scheduler.run(schedule)

    runtime = json.loads((tmp_path / "runtime.json").read_text(encoding="utf-8"))
    assert launched == [launcher.CONTROL_RUN_IDS[0]]
    assert runtime["first_terminal_failure"]["run_id"] == (
        launcher.CONTROL_RUN_IDS[1]
    )
    assert runtime["first_terminal_failure"]["reason"] == "WORKTREE_HEAD_MISMATCH"
    assert runtime["jobs"][launcher.CONTROL_RUN_IDS[0]]["status"] == (
        "TERMINAL_SUMMARY"
    )
    assert runtime["jobs"][launcher.CONTROL_RUN_IDS[1]]["status"] == (
        "TECHNICAL_FAILURE"
    )
    assert all(
        runtime["jobs"][run_id]["status"]
        == "NOT_RUN_AFTER_TERMINAL_FAILURE"
        for run_id in launcher.CONTROL_RUN_IDS[2:]
    )
