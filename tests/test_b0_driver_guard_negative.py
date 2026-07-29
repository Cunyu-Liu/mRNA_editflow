from __future__ import annotations

import copy
import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from scripts.execution import b0_driver_guard


SOURCE_ROOT = Path(__file__).resolve().parents[1]
GUARD = SOURCE_ROOT / "scripts/execution/b0_driver_guard.py"
EXPECTED_HEAD = "a" * 40
EXPECTED_DIRTY_STATE_SHA256 = "b" * 64
EXPECTED_D1_ACCEPTANCE = "/approved/d1-acceptance.json"
EXPECTED_CANONICAL_VALIDATION = "/approved/canonical-validation.json"
DELETE = object()


@dataclass(frozen=True)
class GateCase:
    gate: str
    core_path: tuple[str, ...]
    invalid_value: object


GATE_CASES = (
    GateCase("audit-completion", ("observed_process_exit_code",), "0"),
    GateCase(
        "audit-git-binding",
        ("git_prelaunch_snapshot", "dirty_state_sha256"),
        "c" * 64,
    ),
    GateCase("preflight", ("formal_neural_activity",), "false"),
    GateCase("canonical-validation", ("d1_binding",), DELETE),
    GateCase("split-common", ("partitions",), []),
    GateCase("split-5utr-source-role", ("region",), "three_utr"),
    GateCase("split-5utr-study-role", ("split_kind",), "source_disjoint"),
    GateCase("split-3utr-source-role", ("region",), "five_utr"),
    GateCase("split-3utr-study-role", ("split_kind",), "source_disjoint"),
    GateCase(
        "split-cross-region-role",
        ("target_region",),
        "five_utr",
    ),
    GateCase("leakage", ("counts", "near_neighbor_leakage_count"), "0"),
    GateCase("evaluation-bundle", ("scientific_result_claimed",), True),
    GateCase("final-acceptance", ("failed_gates",), ["unexpected_gate"]),
)


def _valid_gate_payloads() -> dict[str, dict[str, Any]]:
    acceptance = {
        "schema_version": "utr_b0_acceptance.v2",
        "b0_gate_passed": True,
        "failed_gates": [],
        "allowed_claim": "NONE",
        "requires_fm0_reaudit": True,
        "re_audit_required_before_foundation_use": True,
        "supplied_leakage_reports_match_recomputation": True,
        "exposure_ledger": {"coverage": 1, "gate_passed": True},
        "track_role_audit": {"gate_passed": True},
        "track_a_label_seal_audit": {
            "gate_passed": True,
            "role_policy_exact_binding_passed": True,
            "current_d1_chain_binding_passed": True,
        },
        "required_artifact_audit": {"gate_passed": True},
        "d1_exposure_ledger_binding": {"gate_passed": True},
    }
    bundle = {
        "schema_version": "utr_b0_evaluation_artifact_build.v2",
        "status": "PASS",
        "acceptance_preview": {
            "b0_gate_passed": True,
            "failed_gates": [],
        },
        "leakage_evidence_binding": {
            "supplied_reports_exactly_match_recomputation": True,
        },
        "track_a_label_seal_audit": {
            "gate_passed": True,
            "role_policy_exact_binding_passed": True,
            "current_d1_chain_binding_passed": True,
        },
        "required_artifact_audit": {"gate_passed": True},
        "d1_exposure_ledger_binding": {"gate_passed": True},
        "full_d1_binding": {"passed": True},
        "scientific_result_claimed": False,
        "foundation_status": "UNKNOWN_PENDING_FM0",
    }
    leakage = {
        "gate_passed": True,
        "recomputed_from_bound_structural_records": True,
        "canonical_manifest_exact_recomputation": True,
        "foundation_pretraining_overlap": {
            "status": "UNKNOWN_PENDING_FM0",
            "foundation_selected": False,
            "allowed_claim": "NONE",
            "re_audit_required": True,
        },
        "acceptance_gates": {
            "unexplained_overlap_zero": True,
            "exact_source_overlap_zero": True,
            "exact_candidate_overlap_zero": True,
            "reverse_edge_leakage_zero": True,
            "path_leakage_zero": True,
            "near_neighbor_leakage_zero": True,
            "final_endpoint_as_train_intermediate_zero": True,
            "required_axis_overlap_zero": True,
            "foundation_overlap_gate": True,
        },
        "counts": {
            "unexplained_overlap_count": 0,
            "exact_source_leakage_count": 0,
            "exact_candidate_leakage_count": 0,
            "reverse_edge_leakage_count": 0,
            "path_leakage_count": 0,
            "near_neighbor_leakage_count": 0,
            "final_endpoint_as_train_intermediate_count": 0,
            "required_axis_overlap_count": 0,
        },
    }
    return {
        "audit-completion": {
            "state": "COMMAND_COMPLETED",
            "observed_process_exit_code": 0,
            "wrapper_exit_code": 0,
            "stop_reason": None,
        },
        "audit-git-binding": {
            "state": "COMMAND_COMPLETED",
            "git_prelaunch_snapshot": {
                "head": EXPECTED_HEAD,
                "dirty_state_sha256": EXPECTED_DIRTY_STATE_SHA256,
            },
        },
        "preflight": {
            "schema_version": "b0_preflight.v1",
            "status": "PASS",
            "workload_class": "NON_NEURAL_DATA_BENCHMARK",
            "formal_neural_activity": False,
            "path_topology": {
                "attempt_parent_is_exact_approved_parent": True,
                "checks": [{"label": "fixture", "overlap": False}],
            },
            "disk": {"passed": True},
            "d1": {"exposure_ledger_path_is_absolute": True},
            "claim_boundary": {"stage_completion_claimed": False},
        },
        "canonical-validation": {
            "schema_version": "utr_b0_canonical_schema_validation.v2",
            "status": "PASS",
            "invalid_record_count": 0,
            "d1_acceptance_bound": True,
            "d1_binding": {"passed": True},
            "legacy_schema_only_validation": False,
        },
        "split-common": {
            "status": "READY",
            "d1_phase_gate_passed": True,
            "d1_acceptance_path": EXPECTED_D1_ACCEPTANCE,
            "canonical_validation_report_path": EXPECTED_CANONICAL_VALIDATION,
            "partitions": [{"status": "READY"}],
            "partitions_sha256": "d" * 64,
        },
        "split-5utr-source-role": {
            "split_kind": "source_disjoint",
            "region": "five_utr",
        },
        "split-5utr-study-role": {
            "split_kind": "study_disjoint",
            "region": "five_utr",
        },
        "split-3utr-source-role": {
            "split_kind": "source_disjoint",
            "region": "three_utr",
        },
        "split-3utr-study-role": {
            "split_kind": "study_disjoint",
            "region": "three_utr",
        },
        "split-cross-region-role": {
            "split_kind": "cross_region_transfer",
            "source_region": "five_utr",
            "target_region": "three_utr",
        },
        "leakage": leakage,
        "evaluation-bundle": bundle,
        "final-acceptance": acceptance,
    }


def _gate_context_args(gate: str) -> list[str]:
    if gate == "audit-git-binding":
        return [
            "--expected-head",
            EXPECTED_HEAD,
            "--expected-dirty-state-sha256",
            EXPECTED_DIRTY_STATE_SHA256,
        ]
    if gate == "split-common":
        return [
            "--expected-d1-acceptance",
            EXPECTED_D1_ACCEPTANCE,
            "--expected-canonical-validation",
            EXPECTED_CANONICAL_VALIDATION,
        ]
    return []


def _mutate(payload: dict[str, Any], case: GateCase) -> dict[str, Any]:
    mutated = copy.deepcopy(payload)
    parent: dict[str, Any] = mutated
    for key in case.core_path[:-1]:
        child = parent[key]
        assert isinstance(child, dict)
        parent = child
    final_key = case.core_path[-1]
    if case.invalid_value is DELETE:
        del parent[final_key]
    else:
        parent[final_key] = case.invalid_value
    return mutated


def _validate_gate_command(
    *,
    gate: str,
    artifact: Path,
    evidence: Path,
    label: str,
) -> list[str]:
    return [
        sys.executable,
        str(GUARD),
        "validate-gate",
        "--gate",
        gate,
        "--label",
        label,
        "--artifact",
        str(artifact.resolve()),
        "--evidence-output",
        str(evidence.resolve()),
        *_gate_context_args(gate),
    ]


def _assert_rejected_with_exclusive_evidence(
    *,
    gate: str,
    artifact: Path,
    evidence: Path,
    label: str,
) -> None:
    assert not evidence.exists()
    command = _validate_gate_command(
        gate=gate,
        artifact=artifact,
        evidence=evidence,
        label=label,
    )
    completed = subprocess.run(
        command,
        cwd=SOURCE_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode != 0, completed.stdout
    assert evidence.is_file()
    evidence_bytes = evidence.read_bytes()
    payload = json.loads(evidence_bytes)
    assert payload["schema_version"] == "b0_named_gate_evidence.v1"
    assert payload["label"] == label
    assert payload["gate"] == gate
    assert payload["artifact_path"] == str(artifact.resolve())
    assert payload["passed"] is False
    assert isinstance(payload.get("error"), str) and payload["error"]

    repeated = subprocess.run(
        command,
        cwd=SOURCE_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert repeated.returncode != 0
    assert evidence.read_bytes() == evidence_bytes


@pytest.mark.parametrize("case", GATE_CASES, ids=lambda case: case.gate)
def test_validate_gate_rejects_malformed_json_with_failed_exclusive_evidence(
    tmp_path: Path,
    case: GateCase,
) -> None:
    artifact = tmp_path / f"{case.gate}.malformed.json"
    evidence = tmp_path / f"{case.gate}.malformed.evidence.json"
    artifact.write_text("{not-valid-json\n", encoding="utf-8")

    _assert_rejected_with_exclusive_evidence(
        gate=case.gate,
        artifact=artifact,
        evidence=evidence,
        label=f"negative_malformed_{case.gate}",
    )


@pytest.mark.parametrize("case", GATE_CASES, ids=lambda case: case.gate)
def test_validate_gate_rejects_invalid_core_field_with_failed_exclusive_evidence(
    tmp_path: Path,
    case: GateCase,
) -> None:
    artifact = tmp_path / f"{case.gate}.invalid-field.json"
    evidence = tmp_path / f"{case.gate}.invalid-field.evidence.json"
    payload = _mutate(_valid_gate_payloads()[case.gate], case)
    artifact.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _assert_rejected_with_exclusive_evidence(
        gate=case.gate,
        artifact=artifact,
        evidence=evidence,
        label=f"negative_core_field_{case.gate}",
    )


@pytest.mark.parametrize(
    "case",
    (
        GateCase("audit-completion", ("observed_process_exit_code",), 0.0),
        GateCase("audit-completion", ("wrapper_exit_code",), False),
        GateCase("canonical-validation", ("invalid_record_count",), 0.0),
        GateCase("leakage", ("counts", "near_neighbor_leakage_count"), False),
    ),
    ids=(
        "process-exit-float",
        "wrapper-exit-bool",
        "invalid-record-count-float",
        "leakage-count-bool",
    ),
)
def test_validate_gate_rejects_non_integer_count_or_exit_code(
    tmp_path: Path,
    case: GateCase,
) -> None:
    artifact = tmp_path / f"{case.gate}.non-integer.json"
    evidence = tmp_path / f"{case.gate}.non-integer.evidence.json"
    payload = _mutate(_valid_gate_payloads()[case.gate], case)
    artifact.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _assert_rejected_with_exclusive_evidence(
        gate=case.gate,
        artifact=artifact,
        evidence=evidence,
        label=f"negative_non_integer_{case.gate}_{case.core_path[-1]}",
    )


@pytest.mark.parametrize(
    ("mutation", "semantic_error"),
    (
        ("shallow-document", "top-level 16-field evidence set is incomplete"),
        ("missing-observed-evidence", "observed evidence is incomplete"),
    ),
)
def test_named_final_acceptance_uses_full_semantics_for_incomplete_documents(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    semantic_error: str,
) -> None:
    payload = copy.deepcopy(_valid_gate_payloads()["final-acceptance"])
    if mutation == "missing-observed-evidence":
        payload["observed"] = {"leakage_report_count": 5}
    elif mutation != "shallow-document":
        raise AssertionError(f"unknown final-acceptance mutation: {mutation}")
    b0_driver_guard._validate_b0_acceptance_hard_gate(payload)
    calls: list[tuple[str, bool, dict[str, Any]]] = []

    def reject_incomplete_semantics(
        phase: str,
        observed: dict[str, Any],
        require_pass: bool = False,
    ) -> list[str]:
        calls.append((phase, require_pass, observed))
        return [semantic_error]

    monkeypatch.setattr(
        b0_driver_guard,
        "validate_phase_acceptance",
        reject_incomplete_semantics,
    )
    with pytest.raises(
        b0_driver_guard.GuardError,
        match="B0 acceptance semantic validation failed",
    ):
        b0_driver_guard._validate_named_gate(
            "final-acceptance",
            payload,
            expected_head=None,
            expected_dirty_state_sha256=None,
            expected_d1_acceptance=None,
            expected_canonical_validation=None,
        )

    assert calls == [("B0", True, payload)]


@pytest.mark.parametrize(
    "invalid_bytes", ("1", 1.0, True), ids=("string", "float", "bool")
)
def test_artifact_ref_rejects_non_integer_bytes(
    tmp_path: Path,
    invalid_bytes: object,
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"x")
    reference = b0_driver_guard._path_ref(artifact)
    reference["bytes"] = invalid_bytes

    with pytest.raises(b0_driver_guard.GuardError, match=r"\.bytes is invalid"):
        b0_driver_guard._validate_ref(
            artifact,
            reference,
            label="strict artifact ref",
        )


@pytest.mark.parametrize("invalid_sha", (True, 1), ids=("bool", "int"))
def test_sha_validator_rejects_non_string_json_scalars(invalid_sha: object) -> None:
    with pytest.raises(b0_driver_guard.GuardError, match="lowercase hexadecimal"):
        b0_driver_guard._require_hex(invalid_sha, 64, "strict JSON SHA256")


@pytest.mark.parametrize(
    "mutation",
    (
        "top-level-field",
        "entry-field",
        "exclusions",
        "entries-sha256",
        "entry-count-float",
        "entry-bytes-string",
        "entry-live-hash",
    ),
)
def test_success_index_rejects_schema_type_or_live_integrity_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    attempt_root = tmp_path / "attempt"
    nested_artifact = attempt_root / "artifacts/bundle/nested/output.txt"
    nested_artifact.parent.mkdir(parents=True)
    nested_artifact.write_text("sealed\n", encoding="utf-8")
    index_path = attempt_root / "artifact_checksums.json"
    valid_index = b0_driver_guard._success_index(attempt_root)
    index_path.write_text(
        json.dumps(valid_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert b0_driver_guard._validate_success_index(attempt_root)["passed"] is True

    mutated = copy.deepcopy(valid_index)
    if mutation == "top-level-field":
        mutated["unexpected"] = True
    elif mutation == "entry-field":
        mutated["entries"][0]["unexpected"] = True
    elif mutation == "exclusions":
        mutated["excluded_mutable_or_self_referential_paths"].append(
            "artifacts/bundle/nested/output.txt"
        )
    elif mutation == "entries-sha256":
        mutated["entries_sha256"] = "0" * 64
    elif mutation == "entry-count-float":
        mutated["entry_count"] = float(mutated["entry_count"])
    elif mutation == "entry-bytes-string":
        mutated["entries"][0]["bytes"] = str(mutated["entries"][0]["bytes"])
        mutated["entries_sha256"] = b0_driver_guard._canonical_json_sha(
            {"entries": mutated["entries"]}
        )
    elif mutation == "entry-live-hash":
        mutated["entries"][0]["sha256"] = "0" * 64
        mutated["entries_sha256"] = b0_driver_guard._canonical_json_sha(
            {"entries": mutated["entries"]}
        )
    else:
        raise AssertionError(f"unknown mutation fixture: {mutation}")
    index_path.write_text(
        json.dumps(mutated, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(b0_driver_guard.GuardError):
        b0_driver_guard._validate_success_index(attempt_root)


@pytest.mark.parametrize(
    "entry_kind",
    ("file-symlink", "directory-symlink", "dangling-symlink", "fifo"),
)
def test_success_inventory_rejects_symlinks_and_nonregular_entries(
    tmp_path: Path,
    entry_kind: str,
) -> None:
    attempt_root = tmp_path / "attempt"
    artifacts = attempt_root / "artifacts"
    artifacts.mkdir(parents=True)
    if entry_kind == "file-symlink":
        target = artifacts / "target.txt"
        target.write_text("sealed\n", encoding="utf-8")
        (artifacts / "file-link.txt").symlink_to(target.name)
    elif entry_kind == "directory-symlink":
        target = artifacts / "real-directory"
        target.mkdir()
        (artifacts / "directory-link").symlink_to(target.name, target_is_directory=True)
    elif entry_kind == "dangling-symlink":
        (artifacts / "dangling-link").symlink_to("missing-target")
    elif entry_kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO creation is not supported on this platform")
        os.mkfifo(artifacts / "named-pipe")
    else:
        raise AssertionError(f"unknown inventory fixture: {entry_kind}")

    with pytest.raises(
        b0_driver_guard.GuardError,
        match="symbolic links|non-regular",
    ):
        b0_driver_guard._success_index(attempt_root)


def test_success_index_validation_rejects_symlinked_index_file(
    tmp_path: Path,
) -> None:
    attempt_root = tmp_path / "attempt"
    artifact = attempt_root / "artifacts/output.txt"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("sealed\n", encoding="utf-8")
    valid_index = b0_driver_guard._success_index(attempt_root)
    external_index = tmp_path / "external-index.json"
    external_index.write_text(
        json.dumps(valid_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (attempt_root / "artifact_checksums.json").symlink_to(external_index)

    with pytest.raises(b0_driver_guard.GuardError, match="symbolic-link"):
        b0_driver_guard._validate_success_index(attempt_root)


FAILURE_AT_UTC = "2026-07-29T00:00:00Z"


def _valid_failure_payload(
    *,
    state: str = "FAILED_WITH_EVIDENCE",
    exit_code: int = 74,
    reason: str = "STRICT_FAILURE_FIXTURE",
    current_node: str | None = None,
    signal_name: str | None = None,
    line: int | None = None,
    command: str | None = None,
    wrapper_pid: int | None = None,
) -> dict[str, Any]:
    core = {
        "schema_version": "b0_driver_failure.v1",
        "state": state,
        "failed_at_utc": FAILURE_AT_UTC,
        "exit_code": exit_code,
        "reason": reason,
        "current_node": current_node,
        "signal": signal_name,
        "line": line,
        "command": command,
        "wrapper_pid": wrapper_pid,
        "evidence_preserved": True,
        "unrelated_processes_terminated": 0,
    }
    return {
        **core,
        "failure_id": b0_driver_guard._canonical_json_sha(core),
    }


def _write_failure_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _nonterminal_status(state: str) -> dict[str, Any]:
    status: dict[str, Any] = {
        "schema_version": "b0_driver_status.v1",
        "state": state,
        "updated_at_utc": FAILURE_AT_UTC,
        "current_node": None,
        "wrapper_pid": None,
        "terminal": False,
    }
    if state == "RUNNING":
        status.update(
            {
                "current_node": "01_canonical_validation",
                "wrapper_pid": 4242,
                "previous_state": "RUNNING",
                "reason": "WRAPPER_ACTIVE",
            }
        )
    elif state == "PREFLIGHT_PASSED":
        status["previous_state"] = "RUNNING"
    elif state != "REGISTERED":
        raise AssertionError(f"unsupported nonterminal fixture: {state}")
    return status


def _failure_attempt(tmp_path: Path, *, state: str = "REGISTERED") -> Path:
    attempt = tmp_path / "failure-attempt"
    (attempt / "failure").mkdir(parents=True)
    (attempt / "logs").mkdir()
    (attempt / "terminal.lock").write_bytes(b"")
    _write_failure_json(attempt / "status.json", _nonterminal_status(state))
    registered_event = {
        "schema_version": "b0_driver_event.v1",
        "at_utc": FAILURE_AT_UTC,
        "event": "ATTEMPT_REGISTERED",
        "attempt_id": attempt.name,
    }
    (attempt / "logs/events.jsonl").write_text(
        json.dumps(
            registered_event,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return attempt


def _run_failure_command(
    attempt: Path,
    *,
    exit_code: int = 74,
    reason: str = "STRICT_FAILURE_FIXTURE",
    extra_args: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(GUARD),
            "failure",
            "--attempt-root",
            str(attempt),
            "--exit-code",
            str(exit_code),
            "--reason",
            reason,
            *extra_args,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _run_event_command(
    attempt: Path,
    *,
    event: str = "STRICT_EVENT_FIXTURE",
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(GUARD),
            "event",
            "--attempt-root",
            str(attempt),
            "--event",
            event,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("exit_code", True),
        ("exit_code", 1.0),
        ("exit_code", "74"),
        ("exit_code", -1),
        ("exit_code", 256),
        ("reason", True),
        ("reason", ""),
        ("current_node", True),
        ("current_node", "unknown_node"),
        ("signal", True),
        ("signal", "KILL"),
        ("state", "SAFE_PAUSED"),
        ("line", True),
        ("line", 0),
        ("line", 1.0),
        ("command", True),
        ("command", ""),
        ("wrapper_pid", True),
        ("wrapper_pid", 0),
        ("wrapper_pid", 1.0),
        ("evidence_preserved", 1),
        ("unrelated_processes_terminated", False),
        ("unrelated_processes_terminated", 0.0),
        ("unrelated_processes_terminated", "0"),
        ("failure_id", True),
        ("failure_id", "0" * 64),
    ),
    ids=lambda value: repr(value),
)
def test_failure_payload_rejects_schema_type_range_or_canonical_id_drift(
    field: str,
    invalid_value: object,
) -> None:
    payload = _valid_failure_payload()
    payload[field] = invalid_value
    if field != "failure_id":
        core = {key: value for key, value in payload.items() if key != "failure_id"}
        payload["failure_id"] = b0_driver_guard._canonical_json_sha(core)

    with pytest.raises(b0_driver_guard.GuardError):
        b0_driver_guard._validate_failure_payload(payload)


def test_failure_payload_accepts_on_exit_zero_and_strict_safe_pause() -> None:
    on_exit_zero = _valid_failure_payload(exit_code=0)
    safe_pause = _valid_failure_payload(
        state="SAFE_PAUSED",
        exit_code=143,
        current_node="01_canonical_validation",
        signal_name="TERM",
        line=42,
        command='wait "$CURRENT_WRAPPER_PID"',
        wrapper_pid=4242,
    )

    assert b0_driver_guard._validate_failure_payload(on_exit_zero) == on_exit_zero
    assert b0_driver_guard._validate_failure_payload(safe_pause) == safe_pause


def test_failure_creation_and_terminal_retry_are_byte_stable_for_on_exit_zero(
    tmp_path: Path,
) -> None:
    attempt = _failure_attempt(tmp_path)
    first = _run_failure_command(
        attempt,
        exit_code=0,
        reason="ON_EXIT_ZERO",
    )
    assert first.returncode == 0, first.stderr

    failure_path = attempt / "failure/failure.json"
    status_path = attempt / "status.json"
    events_path = attempt / "logs/events.jsonl"
    failure = b0_driver_guard._validate_failure_payload(
        json.loads(failure_path.read_text(encoding="utf-8"))
    )
    assert failure["exit_code"] == 0
    assert failure["current_node"] is None
    assert failure["signal"] is None
    assert failure["line"] is None
    assert failure["command"] is None
    assert failure["wrapper_pid"] is None
    b0_driver_guard._validate_terminal_failure_state(attempt)
    frozen_bytes = {
        path: path.read_bytes() for path in (failure_path, status_path, events_path)
    }

    retry = _run_failure_command(
        attempt,
        exit_code=255,
        reason="IGNORED_AFTER_VALID_TERMINAL_FAILURE",
        extra_args=("--signal", "HUP"),
    )
    assert retry.returncode == 0, retry.stderr
    assert {
        path: path.read_bytes() for path in (failure_path, status_path, events_path)
    } == frozen_bytes


def test_regular_event_append_and_failure_recovery_remain_supported(
    tmp_path: Path,
) -> None:
    attempt = _failure_attempt(tmp_path)
    event_result = _run_event_command(attempt)
    assert event_result.returncode == 0, event_result.stderr
    events = [
        json.loads(line)
        for line in (attempt / "logs/events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert events[-1]["event"] == "STRICT_EVENT_FIXTURE"

    failure_result = _run_failure_command(attempt)
    assert failure_result.returncode == 0, failure_result.stderr
    b0_driver_guard._validate_terminal_failure_state(attempt)


def test_event_append_rejects_symlink_without_external_write(tmp_path: Path) -> None:
    attempt = _failure_attempt(tmp_path)
    events_path = attempt / "logs/events.jsonl"
    original_attempt_events = events_path.read_bytes()
    events_path.unlink()
    external_events = tmp_path / "external-events.jsonl"
    external_events.write_bytes(b'{"external":true}\n')
    external_before = external_events.read_bytes()
    events_path.symlink_to(external_events)

    completed = _run_event_command(attempt)
    assert completed.returncode == 74
    assert external_events.read_bytes() == external_before
    assert events_path.is_symlink()
    assert events_path.read_bytes() != original_attempt_events


def test_symlinked_attempt_root_failure_rejects_before_target_mutation(
    tmp_path: Path,
) -> None:
    target_attempt = _failure_attempt(tmp_path / "target")
    symlinked_attempt = tmp_path / "attempt-root-link"
    symlinked_attempt.symlink_to(target_attempt, target_is_directory=True)
    target_status_before = (target_attempt / "status.json").read_bytes()
    target_events_before = (target_attempt / "logs/events.jsonl").read_bytes()
    assert not (target_attempt / "failure/failure.json").exists()

    completed = _run_failure_command(symlinked_attempt)
    assert completed.returncode == 74
    assert (target_attempt / "status.json").read_bytes() == target_status_before
    assert (target_attempt / "logs/events.jsonl").read_bytes() == target_events_before
    assert not (target_attempt / "failure/failure.json").exists()


@pytest.mark.parametrize(
    "handler",
    (
        b0_driver_guard._event_command,
        b0_driver_guard._status_command,
        b0_driver_guard._terminal_success_command,
        b0_driver_guard._failure_command,
        b0_driver_guard._watchdog_once_command,
        b0_driver_guard._watchdog_command,
        b0_driver_guard._seal_command,
    ),
    ids=lambda handler: handler.__name__,
)
def test_existing_attempt_mutators_reject_symlinked_root_before_other_arguments(
    tmp_path: Path,
    handler: object,
) -> None:
    target_attempt = _failure_attempt(tmp_path / "target")
    symlinked_attempt = tmp_path / "mutator-attempt-link"
    symlinked_attempt.symlink_to(target_attempt, target_is_directory=True)
    status_before = (target_attempt / "status.json").read_bytes()
    events_before = (target_attempt / "logs/events.jsonl").read_bytes()

    with pytest.raises(b0_driver_guard.GuardError, match="non-symlink directory"):
        handler(argparse.Namespace(attempt_root=str(symlinked_attempt)))

    assert (target_attempt / "status.json").read_bytes() == status_before
    assert (target_attempt / "logs/events.jsonl").read_bytes() == events_before
    assert not (target_attempt / "failure/failure.json").exists()


@pytest.mark.parametrize("parent_pid", (None, True, False, 0, -1, 1.0, "1"))
def test_watchdog_parent_pid_requires_strict_positive_non_boolean_integer(
    parent_pid: object,
) -> None:
    with pytest.raises(
        b0_driver_guard.GuardError,
        match="watchdog parent PID must be a positive integer",
    ):
        b0_driver_guard._require_positive_int(
            parent_pid,
            "watchdog parent PID",
        )

    assert b0_driver_guard._require_positive_int(1, "watchdog parent PID") == 1


def _watchdog_attempt(tmp_path: Path) -> tuple[Path, Path, Path]:
    attempt = _failure_attempt(tmp_path)
    metrics_path = attempt / "logs/system_metrics.jsonl"
    metrics_path.write_text('{"seed":true}\n', encoding="utf-8")
    d1_acceptance = tmp_path / "d1-acceptance.json"
    d1_acceptance.write_text("{}\n", encoding="utf-8")
    return attempt, d1_acceptance, metrics_path


def test_watchdog_exits_at_start_without_pollution_when_exact_parent_is_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt, d1_acceptance, metrics_path = _watchdog_attempt(tmp_path)
    status_before = (attempt / "status.json").read_bytes()
    events_before = (attempt / "logs/events.jsonl").read_bytes()
    metrics_before = metrics_path.read_bytes()
    observed_parent_pids: list[int] = []

    def parent_is_absent(parent_pid: int) -> bool:
        observed_parent_pids.append(parent_pid)
        return False

    monkeypatch.setattr(
        b0_driver_guard,
        "_watchdog_parent_is_alive",
        parent_is_absent,
    )
    monkeypatch.setattr(
        b0_driver_guard.time,
        "sleep",
        lambda _: pytest.fail("orphan watchdog must not sleep at startup"),
    )

    result = b0_driver_guard._watchdog_command(
        argparse.Namespace(
            attempt_root=str(attempt),
            d1_acceptance=str(d1_acceptance),
            parent_pid=4242,
        )
    )

    assert result == 0
    assert observed_parent_pids == [4242]
    assert (attempt / "status.json").read_bytes() == status_before
    assert (attempt / "logs/events.jsonl").read_bytes() == events_before
    assert metrics_path.read_bytes() == metrics_before


def test_watchdog_parent_death_after_300_second_sleep_exits_without_pollution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt, d1_acceptance, metrics_path = _watchdog_attempt(tmp_path)
    status_before = (attempt / "status.json").read_bytes()
    events_before = (attempt / "logs/events.jsonl").read_bytes()
    metrics_before = metrics_path.read_bytes()
    parent_states = [True, False]
    observed_parent_pids: list[int] = []
    sleep_intervals: list[int] = []

    def parent_is_alive(parent_pid: int) -> bool:
        observed_parent_pids.append(parent_pid)
        return parent_states.pop(0)

    monkeypatch.setattr(
        b0_driver_guard,
        "_watchdog_parent_is_alive",
        parent_is_alive,
    )
    monkeypatch.setattr(
        b0_driver_guard.time,
        "sleep",
        sleep_intervals.append,
    )
    monkeypatch.setattr(
        b0_driver_guard,
        "_watchdog_sample",
        lambda *_: pytest.fail("orphan watchdog must not append a sample"),
    )

    result = b0_driver_guard._watchdog_command(
        argparse.Namespace(
            attempt_root=str(attempt),
            d1_acceptance=str(d1_acceptance),
            parent_pid=4242,
        )
    )

    assert result == 0
    assert observed_parent_pids == [4242, 4242]
    assert sleep_intervals == [300]
    assert parent_states == []
    assert (attempt / "status.json").read_bytes() == status_before
    assert (attempt / "logs/events.jsonl").read_bytes() == events_before
    assert metrics_path.read_bytes() == metrics_before


def test_init_rejects_preexisting_symlink_attempt_root(tmp_path: Path) -> None:
    target_attempt = tmp_path / "target-attempt"
    target_attempt.mkdir()
    symlinked_attempt = tmp_path / "new-attempt-link"
    symlinked_attempt.symlink_to(target_attempt, target_is_directory=True)

    with pytest.raises(b0_driver_guard.GuardError, match="refusing existing"):
        b0_driver_guard._init_command(
            argparse.Namespace(
                attempt_root=str(symlinked_attempt),
                approved_b0_parent=str(tmp_path),
            )
        )

    assert list(target_attempt.iterdir()) == []


@pytest.mark.parametrize(
    ("state", "preexisting_event"),
    (
        ("REGISTERED", False),
        ("REGISTERED", True),
        ("RUNNING", False),
        ("RUNNING", True),
        ("PREFLIGHT_PASSED", False),
        ("PREFLIGHT_PASSED", True),
    ),
)
def test_failure_recovers_generated_nonterminal_status_from_failure_only_or_event(
    tmp_path: Path,
    state: str,
    preexisting_event: bool,
) -> None:
    attempt = _failure_attempt(tmp_path, state=state)
    failure = _valid_failure_payload(
        current_node=("01_canonical_validation" if state == "RUNNING" else None),
        wrapper_pid=4242 if state == "RUNNING" else None,
    )
    _write_failure_json(attempt / "failure/failure.json", failure)
    if preexisting_event:
        event = b0_driver_guard._failure_event_payload(
            failure,
            at_utc=FAILURE_AT_UTC,
        )
        with (attempt / "logs/events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    event,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )

    completed = _run_failure_command(
        attempt,
        exit_code=1,
        reason="RECOVERY_ARGUMENTS_ARE_NOT_THE_FROZEN_EVIDENCE",
    )
    assert completed.returncode == 0, completed.stderr
    b0_driver_guard._validate_terminal_failure_state(attempt)


@pytest.mark.parametrize(
    "mutation",
    ("bad-json", "duplicate", "conflict", "terminal-not-final"),
)
def test_failure_recovery_rejects_bad_duplicate_conflicting_or_nonfinal_event_log(
    tmp_path: Path,
    mutation: str,
) -> None:
    attempt = _failure_attempt(tmp_path)
    failure = _valid_failure_payload()
    _write_failure_json(attempt / "failure/failure.json", failure)
    valid_event = b0_driver_guard._failure_event_payload(
        failure,
        at_utc=FAILURE_AT_UTC,
    )
    events_path = attempt / "logs/events.jsonl"
    if mutation == "bad-json":
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write("{not-json\n")
    else:
        events = [valid_event]
        if mutation == "duplicate":
            events.append(copy.deepcopy(valid_event))
        elif mutation == "conflict":
            events[0]["failure_id"] = "0" * 64
        elif mutation == "terminal-not-final":
            events.append(
                {
                    "schema_version": "b0_driver_event.v1",
                    "at_utc": FAILURE_AT_UTC,
                    "event": "POST_FAILURE_EVENT",
                }
            )
        else:
            raise AssertionError(f"unknown failure-event mutation: {mutation}")
        with events_path.open("a", encoding="utf-8") as handle:
            for event in events:
                handle.write(
                    json.dumps(
                        event,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )

    completed = _run_failure_command(attempt)
    assert completed.returncode == 74


@pytest.mark.parametrize(
    "mutation",
    ("status-extra-field", "failure-type", "event-type", "failure-ref-path"),
)
def test_existing_terminal_failure_retry_fully_revalidates_every_document(
    tmp_path: Path,
    mutation: str,
) -> None:
    attempt = _failure_attempt(tmp_path)
    created = _run_failure_command(attempt)
    assert created.returncode == 0, created.stderr
    status_path = attempt / "status.json"
    failure_path = attempt / "failure/failure.json"
    events_path = attempt / "logs/events.jsonl"
    if mutation == "status-extra-field":
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["unexpected"] = False
        _write_failure_json(status_path, status)
    elif mutation == "failure-type":
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        failure["exit_code"] = False
        core = {key: value for key, value in failure.items() if key != "failure_id"}
        failure["failure_id"] = b0_driver_guard._canonical_json_sha(core)
        _write_failure_json(failure_path, failure)
    elif mutation == "event-type":
        events = [
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
        ]
        events[-1]["exit_code"] = False
        events_path.write_text(
            "".join(
                json.dumps(
                    event,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
                for event in events
            ),
            encoding="utf-8",
        )
    elif mutation == "failure-ref-path":
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["failure"]["path"] = str(failure_path)
        _write_failure_json(status_path, status)
    else:
        raise AssertionError(f"unknown terminal failure mutation: {mutation}")

    retry = _run_failure_command(attempt)
    assert retry.returncode == 74


def test_failure_recovery_rejects_bad_json_and_symlinked_failure_evidence(
    tmp_path: Path,
) -> None:
    bad_json_attempt = _failure_attempt(tmp_path / "bad-json")
    (bad_json_attempt / "failure/failure.json").write_text(
        "{not-json\n",
        encoding="utf-8",
    )
    assert _run_failure_command(bad_json_attempt).returncode == 74

    symlink_attempt = _failure_attempt(tmp_path / "symlink")
    external_failure = tmp_path / "external-failure.json"
    _write_failure_json(external_failure, _valid_failure_payload())
    (symlink_attempt / "failure/failure.json").symlink_to(external_failure)
    assert _run_failure_command(symlink_attempt).returncode == 74

    symlink_dir_attempt = _failure_attempt(tmp_path / "symlink-directory")
    (symlink_dir_attempt / "failure").rmdir()
    external_failure_dir = tmp_path / "external-failure-directory"
    external_failure_dir.mkdir()
    (symlink_dir_attempt / "failure").symlink_to(
        external_failure_dir,
        target_is_directory=True,
    )
    assert _run_failure_command(symlink_dir_attempt).returncode == 74
    assert not (external_failure_dir / "failure.json").exists()


def _expected_gate_evidence_labels() -> frozenset[str]:
    split_basenames = (
        "5utr_source_disjoint.json",
        "5utr_study_disjoint.json",
        "3utr_source_disjoint.json",
        "3utr_study_disjoint.json",
        "cross_region_transfer.json",
    )
    labels = {
        "preflight",
        "canonical_validation",
        "split_5utr_source_role",
        "split_5utr_study_role",
        "split_3utr_source_role",
        "split_3utr_study_role",
        "split_cross_region_role",
        "evaluation_bundle",
        "final_acceptance",
    }
    for node in b0_driver_guard.EXPECTED_AUDIT_NODES:
        labels.add(f"audit_completion_{node}")
        labels.add(f"audit_git_binding_{node}")
    labels.update(f"split_common_{name}" for name in split_basenames)
    labels.update(f"leakage_{name}" for name in split_basenames)
    return frozenset(labels)


@pytest.fixture
def complete_gate_evidence_dir(tmp_path: Path) -> tuple[Path, frozenset[str]]:
    attempt_root = tmp_path / "attempt"
    gate_dir = attempt_root / "provenance/gates"
    gate_dir.mkdir(parents=True)
    expected_labels = _expected_gate_evidence_labels()
    for label in expected_labels:
        (gate_dir / f"{label}.json").write_text(
            json.dumps({"label": label, "passed": True}) + "\n",
            encoding="utf-8",
        )
    return attempt_root, expected_labels


def test_exact_gate_evidence_fixture_matches_driver_contract() -> None:
    labels = _expected_gate_evidence_labels()

    assert len(labels) == 47
    assert sum(label.startswith("audit_completion_") for label in labels) == 14
    assert sum(label.startswith("audit_git_binding_") for label in labels) == 14
    assert sum(label.startswith("split_common_") for label in labels) == 5
    assert sum(label.startswith("leakage_") for label in labels) == 5


def test_named_gate_set_rejects_unexpected_extra_evidence(
    complete_gate_evidence_dir: tuple[Path, frozenset[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt_root, expected_labels = complete_gate_evidence_dir
    unexpected = attempt_root / "provenance/gates/unexpected_extra.json"
    unexpected.write_text('{"passed":true}\n', encoding="utf-8")

    monkeypatch.setattr(
        b0_driver_guard,
        "_expected_gate_specs",
        lambda **_: {label: {} for label in expected_labels},
    )
    with pytest.raises(
        b0_driver_guard.GuardError,
        match="named-gate evidence set mismatch",
    ):
        b0_driver_guard._validate_named_gate_evidence_set(
            attempt_root=attempt_root,
            expected_head=EXPECTED_HEAD,
            expected_dirty_state_sha256=EXPECTED_DIRTY_STATE_SHA256,
            d1_acceptance=attempt_root / "unused-d1-acceptance.json",
            parsed_events=[],
        )
