from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.route_a_v3.transition_record_route2_xeditsetflow_s1_930_terminal_invalidation as transition


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _repair_audit(runtime: Path) -> dict:
    return {
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_s1_seed_initialization_repair.v1",
        "status": "XEDITSETFLOW_V4_S1_SEED_INITIALIZATION_REPAIR_FROZEN_BEFORE_INDEPENDENT_RETRY",
        "affected_family": {
            "runner_git_head": transition.OLD_HEAD,
            "screen_seed": transition.SCREEN_SEED,
            "run_ids": list(transition.RUN_IDS),
            "runtime_path": str(runtime),
            "launcher_consumed_once": True,
            "artifacts_immutable": True,
            "same_family_retry_authorized": False,
        },
        "defect": {
            "identity": transition.DEFECT_IDENTITY,
            "model_construction_consumes_cpu_rng": True,
            "cpu_manual_seed_was_after_model_construction": True,
            "cuda_manual_seed_all_was_after_model_construction": True,
            "full_and_single_mode_used_independent_processes": True,
            "nominal_seed_controlled_parameter_initialization": False,
            "matched_full_single_initialization_established": False,
            "exact_seed_reproducibility_established": False,
            "affected_family_can_authorize_successor": False,
        },
        "claim_boundary": {
            "affected_nominal_gate_is_scientific_successor_authority": False,
        },
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def _terminal_payload(
    *, stage: str, run_id: str, checkpoint_pass: int | None = None
) -> dict:
    if stage == "TRAINING":
        return {
            "schema_version": "route_a_v3_route2_xeditsetflow_v4_s1_training_summary.v1",
            "status": "TERMINAL_XEDITSETFLOW_V4_S1_TRAINING_COMPLETE_PENDING_VALIDATION",
            "run_stage": "SCREEN",
            "run_id": run_id,
            "seed": transition.SCREEN_SEED,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        }
    return {
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_s1_checkpoint_validation.v1",
        "status": "TERMINAL_XEDITSETFLOW_V4_S1_CHECKPOINT_VALIDATION_COMPLETE",
        "run_stage": "SCREEN",
        "run_id": run_id,
        "seed": transition.SCREEN_SEED,
        "checkpoint_pass": checkpoint_pass,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def _runtime_base(tmp_path: Path) -> tuple[Path, dict]:
    runtime = tmp_path / "family" / "runtime.json"
    training_jobs = {}
    for run_id in transition.RUN_IDS:
        root = tmp_path / "family" / "training" / run_id
        summary = root / "training_summary.json"
        failure = root / "failure.json"
        _write(summary, _terminal_payload(stage="TRAINING", run_id=run_id))
        training_jobs[f"training:{run_id}"] = {
            "run_id": run_id,
            "physical_gpu_index": 0,
            "status": "TERMINAL_COMPLETE",
            "return_code": 0,
            "terminal_artifact_kind": "SUMMARY",
            "terminal_summary": str(summary),
            "terminal_failure": str(failure),
        }
    validation_jobs = {}
    for run_id in transition.RUN_IDS:
        for checkpoint_pass in transition.CHECKPOINT_PASSES:
            root = (
                tmp_path
                / "family"
                / "validation"
                / run_id
                / f"pass_{checkpoint_pass}"
            )
            summary = root / "validation_summary.json"
            failure = root.with_name(root.name + ".failed.json")
            _write(
                summary,
                _terminal_payload(
                    stage="VALIDATION",
                    run_id=run_id,
                    checkpoint_pass=checkpoint_pass,
                ),
            )
            key = f"validation:{run_id}:pass_{checkpoint_pass}"
            validation_jobs[key] = {
                "run_id": run_id,
                "checkpoint_pass": checkpoint_pass,
                "physical_gpu_index": checkpoint_pass % 6,
                "status": "TERMINAL_COMPLETE",
                "return_code": 0,
                "terminal_artifact_kind": "SUMMARY",
                "terminal_summary": str(summary),
                "terminal_failure": str(failure),
            }
    gate = tmp_path / "family" / "screen_gate.json"
    gate_failure = gate.with_name(gate.name + ".failed.json")
    payload = {
        "schema_version": transition.RUNTIME_SCHEMA,
        "status": transition.SCIENTIFIC_TERMINAL_STATUS,
        "scheduler_pid": 4321,
        "git_head": transition.OLD_HEAD,
        "objective_identity": transition.OBJECTIVE_IDENTITY,
        "cross_state_candidate_mode_responsibility_weight": transition.OBJECTIVE_WEIGHT,
        "training_jobs": training_jobs,
        "validation_jobs": validation_jobs,
        "adjudication": {
            "status": "TERMINAL_COMPLETE",
            "return_code": 0,
            "terminal_artifact_kind": "GATE",
            "gate_path": str(gate),
            "failure_path": str(gate_failure),
        },
        "first_terminal_failure": None,
        "free_memory_gate_applied": False,
        "active_performance_output_read": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    return runtime, payload


def _bind_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, runtime: Path
) -> tuple[Path, Path]:
    receipt = tmp_path / "audits" / "terminal_invalidation.json"
    repair = tmp_path / "repair.json"
    monkeypatch.setattr(transition, "OLD_RUNTIME", runtime)
    monkeypatch.setattr(transition, "CANONICAL_RECEIPT", receipt)
    monkeypatch.setattr(transition, "REPAIR_AUDIT", repair)
    _write(repair, _repair_audit(runtime))
    return receipt, repair


def _run_bound(
    runtime: Path,
    receipt: Path,
    repair: Path,
    *,
    process_alive: bool = False,
) -> dict:
    return transition.run(
        runtime_path=runtime,
        receipt_path=receipt,
        repair_audit_path=repair,
        process_is_alive=lambda _pid: process_alive,
    )


def test_running_runtime_is_read_once_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, payload = _runtime_base(tmp_path)
    payload["status"] = transition.RUNNING_STATUS
    _write(runtime, payload)
    receipt, repair = _bind_paths(monkeypatch, tmp_path, runtime)
    original_read = transition.read_json
    reads = []

    def counted_read(path: Path) -> dict:
        reads.append(path)
        return original_read(path)

    monkeypatch.setattr(transition, "read_json", counted_read)
    result = _run_bound(runtime, receipt, repair)
    assert result == {
        "status": transition.RUNNING_OBSERVED_STATUS,
        "runtime_status": transition.RUNNING_STATUS,
        "receipt_written": False,
    }
    assert reads == [runtime]
    assert not receipt.exists() and not receipt.with_suffix(".json.partial").exists()


@pytest.mark.parametrize(
    "gate_status",
    ["XEDITSETFLOW_V4_S1_SCREEN_PASS", "XEDITSETFLOW_V4_S1_SCREEN_NO_GO"],
)
def test_scientific_terminal_is_frozen_but_never_authorizes_successor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gate_status: str
) -> None:
    runtime, payload = _runtime_base(tmp_path)
    gate = Path(payload["adjudication"]["gate_path"])
    passed = gate_status == "XEDITSETFLOW_V4_S1_SCREEN_PASS"
    _write(
        gate,
        {
            "schema_version": "route_a_v3_route2_xeditsetflow_v4_s1_screen_gate.v1",
            "status": gate_status,
            "screen_seed": transition.SCREEN_SEED,
            "legacy_v4_confirmation_authorized": False,
            "confirmation_authorized": False,
            "successor_protocol_required": passed,
            "s1_mechanics_screen_passed": passed,
            "additional_seed_authorized": False,
            "development_test_authorized": False,
            "guidance_authorized": False,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )
    _write(runtime, payload)
    receipt, repair = _bind_paths(monkeypatch, tmp_path, runtime)
    result = _run_bound(runtime, receipt, repair)
    assert result["status"] == transition.SCIENTIFIC_INVALIDATION_STATUS
    assert result["terminal_adjudication"]["nominal_gate_status"] == gate_status
    assert result["known_defect"]["identity"] == transition.DEFECT_IDENTITY
    assert result["successor_authorized"] is False
    assert result["same_family_retry_authorized"] is False
    assert result["old_family_artifacts_read_only"] is True
    assert result["gpu_inventory_or_probe_executed"] is False
    assert json.loads(receipt.read_text()) == result


def test_exact_technical_terminal_freezes_first_failure_and_pending_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, payload = _runtime_base(tmp_path)
    payload["status"] = transition.TECHNICAL_TERMINAL_STATUS
    failed_key = "training:v4_s1_full"
    failed = payload["training_jobs"][failed_key]
    Path(failed["terminal_summary"]).unlink()
    _write(
        Path(failed["terminal_failure"]),
        {
            "schema_version": "route_a_v3_route2_xeditsetflow_v4_s1_training_failure.v1",
            "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
            "run_id": "v4_s1_full",
            "run_stage": "SCREEN",
            "seed": transition.SCREEN_SEED,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )
    failed.update(
        {
            "return_code": 1,
            "terminal_artifact_kind": "FAILURE",
        }
    )
    for row in payload["validation_jobs"].values():
        Path(row["terminal_summary"]).unlink()
        row.update(
            {
                "status": "NOT_RUN_AFTER_TERMINAL_FAILURE",
                "terminal_artifact_kind": None,
            }
        )
    payload["adjudication"]["status"] = "NOT_RUN_AFTER_TERMINAL_FAILURE"
    payload["adjudication"].pop("return_code")
    payload["adjudication"].pop("terminal_artifact_kind")
    payload["first_terminal_failure"] = {
        "stage": "TRAINING",
        "job_key": failed_key,
        "run_id": "v4_s1_full",
        "reason": "JOB_TERMINAL_FAILURE_ARTIFACT",
        "return_code": 1,
        "terminal_artifact_kind": "FAILURE",
    }
    _write(runtime, payload)
    receipt, repair = _bind_paths(monkeypatch, tmp_path, runtime)
    result = _run_bound(runtime, receipt, repair)
    assert result["status"] == transition.TECHNICAL_INVALIDATION_STATUS
    terminal = result["terminal_adjudication"]
    assert terminal["first_terminal_failure"]["job_key"] == failed_key
    assert terminal["not_run_after_terminal_failure_count"] == 8
    assert result["successor_authorized"] is False


@pytest.mark.parametrize("existing_partial", [False, True])
def test_existing_receipt_or_partial_refuses_before_old_runtime_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_partial: bool,
) -> None:
    runtime = tmp_path / "family" / "runtime.json"
    receipt, repair = _bind_paths(monkeypatch, tmp_path, runtime)
    occupied = receipt.with_suffix(".json.partial") if existing_partial else receipt
    _write(occupied, {"status": "IMMUTABLE_EXISTING_EVIDENCE"})
    monkeypatch.setattr(
        transition,
        "read_json",
        lambda _path: (_ for _ in ()).throw(AssertionError("runtime was read")),
    )
    with pytest.raises(transition.S1TerminalInvalidationError, match="already exists"):
        _run_bound(runtime, receipt, repair)


def test_terminal_refuses_while_recorded_scheduler_process_is_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, payload = _runtime_base(tmp_path)
    _write(runtime, payload)
    receipt, repair = _bind_paths(monkeypatch, tmp_path, runtime)
    with pytest.raises(transition.S1TerminalInvalidationError, match="still alive"):
        _run_bound(runtime, receipt, repair, process_alive=True)
    assert not receipt.exists()


@pytest.mark.parametrize(
    ("command_line", "expected"),
    [
        (
            "/python scripts/route_a_v3/run_route2_xeditsetflow_s1_screen_scheduler.py "
            f"--schedule /family_{transition.OLD_HEAD}/schedule.json",
            True,
        ),
        ("/python unrelated_reused_pid.py", False),
        ("", False),
    ],
)
def test_scheduler_process_check_distinguishes_scheduler_from_reused_pid(
    monkeypatch: pytest.MonkeyPatch, command_line: str, expected: bool
) -> None:
    monkeypatch.setattr(
        transition.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(
            returncode=0 if command_line else 1,
            stdout=command_line,
            stderr="",
        ),
    )
    assert transition.scheduler_process_is_alive(4321) is expected


def test_scheduler_process_inspection_failure_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        transition.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(
            returncode=2,
            stdout="",
            stderr="ps inspection failed",
        ),
    )
    with pytest.raises(
        transition.S1TerminalInvalidationError,
        match="process inspection failed",
    ):
        transition.scheduler_process_is_alive(4321)


def test_double_or_partial_job_terminal_is_not_exact_terminal_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, payload = _runtime_base(tmp_path)
    gate = Path(payload["adjudication"]["gate_path"])
    _write(
        gate,
        {
            "schema_version": "route_a_v3_route2_xeditsetflow_v4_s1_screen_gate.v1",
            "status": "XEDITSETFLOW_V4_S1_SCREEN_NO_GO",
            "screen_seed": transition.SCREEN_SEED,
            "legacy_v4_confirmation_authorized": False,
            "confirmation_authorized": False,
            "successor_protocol_required": False,
            "s1_mechanics_screen_passed": False,
            "additional_seed_authorized": False,
            "development_test_authorized": False,
            "guidance_authorized": False,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )
    row = payload["training_jobs"]["training:v4_s1_full"]
    _write(
        Path(row["terminal_failure"]),
        {
            "run_id": "v4_s1_full",
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )
    _write(runtime, payload)
    receipt, repair = _bind_paths(monkeypatch, tmp_path, runtime)
    with pytest.raises(transition.S1TerminalInvalidationError, match="unique terminal"):
        _run_bound(runtime, receipt, repair)
    assert not receipt.exists()


def test_transition_has_no_gpu_model_or_protected_outcome_reader() -> None:
    source = Path(transition.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "nvidia-smi",
        "import torch",
        "torch.load",
        "model.forward",
        "trajectories.private",
        "development_test_path",
        "evaluation_outcomes_accessed",
    ):
        assert forbidden not in source
