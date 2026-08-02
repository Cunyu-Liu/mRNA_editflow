"""Adversarial tests for the MK0 section 19--21 terminal protocol."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pytest

from mrna_editflow.core.mk0 import run_contract as RUN_CONTRACT
from mrna_editflow.core.mk0.run_contract import (
    create_contract_tree,
    resume_failure_closure_if_present,
    sha256_file,
    update_status,
    validate_terminal_chain,
    write_bytes_exclusive_atomic,
    write_failed_sentinel,
    write_json_exclusive_atomic,
    write_whole_run_checksum_ledger,
)


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "MK0_utrlm_mathkernel_tiny_20260802T030000Z_abcdef0_s20260802"


def _load_script(name: str, module_name: str) -> Any:
    path = ROOT / "scripts" / "mk0" / name
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    write_json_exclusive_atomic(path, payload)


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _base_run(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    run_root = tmp_path / RUN_ID
    run_root.mkdir()
    create_contract_tree(run_root)
    (run_root / "artifacts" / "mk0").mkdir(parents=True)
    _write_json(
        run_root / ".mk0_run_owner.json",
        {"schema_version": "mk0_run_owner_v1", "run_id": RUN_ID},
    )
    _write_json(
        run_root / "run_manifest.json",
        {
            "schema_version": "mk0_run_manifest_v3",
            "run_id": RUN_ID,
            "run_root": str(run_root),
            "timing": {"start_utc": "2026-08-02T03:00:01Z", "end_utc": None},
            "process_identity": {"cpu_pid": 12345},
            "known_deviations": ["TEST_FIXTURE"],
        },
    )
    update_status(
        run_root,
        run_id=RUN_ID,
        state="CPU_VERIFIED_PENDING_GPU",
        terminal=False,
        stop_reason="TEST_FIXTURE",
    )
    return run_root


def _seal_failure(run_root: Path) -> None:
    write_failed_sentinel(
        run_root,
        run_id=RUN_ID,
        stage="GPU_SMOKE",
        reason="INJECTED_FAILURE",
        exit_code=17,
    )
    assert validate_terminal_chain(run_root, run_id=RUN_ID) == "FAILED"


def _seal_success(run_root: Path) -> None:
    freeze = run_root / "artifacts" / "mk0" / "mk0_freeze_manifest.json"
    completion = run_root / "summary" / "run_completion_manifest.json"
    _write_json(freeze, {"run_id": RUN_ID, "status": "PASS"})
    _write_json(completion, {"run_id": RUN_ID, "status": "DONE"})
    update_status(
        run_root,
        run_id=RUN_ID,
        state="CLOSED_ACCEPTED",
        terminal=True,
        stop_reason="ALL_M01_M35_PASSED",
        exit_code=0,
    )
    mk0_status = run_root / "mk0_status.json"
    _write_json(mk0_status, {"run_id": RUN_ID, "status": "DONE"})
    ledger = write_whole_run_checksum_ledger(run_root)
    write_bytes_exclusive_atomic(
        run_root / "DONE",
        (
            f"{RUN_ID}\n{sha256_file(freeze)}\n{sha256_file(completion)}\n"
            f"{ledger['sha256']}\n{sha256_file(run_root / 'status.json')}\n"
            f"{sha256_file(mk0_status)}\n"
        ).encode("ascii"),
    )
    assert validate_terminal_chain(run_root, run_id=RUN_ID) == "DONE"


def test_failed_five_line_chain_is_exact_and_duplicate_call_is_immutable(
    tmp_path: Path,
) -> None:
    run_root = _base_run(tmp_path)
    _seal_failure(run_root)
    lines = (run_root / "FAILED").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    assert lines[0] == RUN_ID
    assert lines[1] == "GPU_SMOKE"
    assert lines[2] == sha256_file(
        run_root / "failure" / "run_failure_completion_manifest.json"
    )
    assert lines[3] == sha256_file(
        run_root / "failure" / "artifact_checksums_at_failure.sha256"
    )
    assert lines[4] == sha256_file(run_root / "mk0_status.json")
    before = _file_hashes(run_root)
    _seal_failure(run_root)
    assert _file_hashes(run_root) == before


def test_done_six_line_chain_has_exact_ledger_coverage(tmp_path: Path) -> None:
    run_root = _base_run(tmp_path)
    _seal_success(run_root)
    lines = (run_root / "DONE").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    assert lines[0] == RUN_ID
    ledger_paths = {
        line.partition("  ")[2]
        for line in (run_root / "artifact_checksums.sha256")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    actual_paths = {
        path.relative_to(run_root).as_posix()
        for path in run_root.rglob("*")
        if path.is_file()
        and path.name != "DONE"
        and path != run_root / "artifact_checksums.sha256"
    }
    assert ledger_paths == actual_paths
    before = _file_hashes(run_root)
    assert validate_terminal_chain(run_root, run_id=RUN_ID) == "DONE"
    assert _file_hashes(run_root) == before


@pytest.mark.parametrize("terminal", ["DONE", "FAILED"])
def test_unledgered_file_invalidates_sealed_terminal(
    tmp_path: Path, terminal: str
) -> None:
    run_root = _base_run(tmp_path)
    if terminal == "DONE":
        _seal_success(run_root)
    else:
        _seal_failure(run_root)
    (run_root / "late_unledgered.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not exactly cover"):
        validate_terminal_chain(run_root, run_id=RUN_ID)


def test_interrupted_failure_after_ledger_resumes_without_changing_bound_bytes(
    tmp_path: Path,
) -> None:
    run_root = _base_run(tmp_path)
    _seal_failure(run_root)
    (run_root / "FAILED").unlink()
    before = _file_hashes(run_root)
    assert resume_failure_closure_if_present(run_root, run_id=RUN_ID) == "FAILED"
    after = _file_hashes(run_root)
    assert {key: value for key, value in after.items() if key != "FAILED"} == before
    assert validate_terminal_chain(run_root, run_id=RUN_ID) == "FAILED"


def test_interrupted_failure_after_completion_rebuilds_ledger_then_sentinel(
    tmp_path: Path,
) -> None:
    run_root = _base_run(tmp_path)
    _seal_failure(run_root)
    (run_root / "FAILED").unlink()
    (run_root / "failure" / "artifact_checksums_at_failure.sha256").unlink()
    before = _file_hashes(run_root)
    assert resume_failure_closure_if_present(run_root, run_id=RUN_ID) == "FAILED"
    after = _file_hashes(run_root)
    for relative, digest in before.items():
        assert after[relative] == digest
    assert validate_terminal_chain(run_root, run_id=RUN_ID) == "FAILED"


@pytest.mark.parametrize(
    "interrupt_after", ["intent", "status", "summary", "mk0_status"]
)
def test_early_failure_publication_interrupt_resumes_original_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_after: str,
) -> None:
    run_root = _base_run(tmp_path)
    original_update_status = RUN_CONTRACT.update_status
    original_atomic_replace = RUN_CONTRACT._atomic_json_replace

    if interrupt_after == "intent":

        def fail_before_status(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("injected interruption after intent")

        monkeypatch.setattr(RUN_CONTRACT, "update_status", fail_before_status)
    else:
        target_name = {
            "status": "status.json",
            "summary": "summary.json",
            "mk0_status": "mk0_status.json",
        }[interrupt_after]

        def fail_after_replace(path: Path, payload: dict[str, Any]) -> None:
            original_atomic_replace(path, payload)
            if path.name == target_name:
                raise RuntimeError(f"injected interruption after {interrupt_after}")

        monkeypatch.setattr(RUN_CONTRACT, "_atomic_json_replace", fail_after_replace)

    with pytest.raises(RuntimeError, match="injected interruption"):
        write_failed_sentinel(
            run_root,
            run_id=RUN_ID,
            stage="GPU_SMOKE",
            reason="ORIGINAL_FAILURE",
            exit_code=23,
        )
    intent_path = run_root / "failure" / "failure_closure_intent.json"
    intent_sha256 = sha256_file(intent_path)
    assert not (run_root / "FAILED").exists()
    assert not (run_root / "failure" / "run_failure_completion_manifest.json").exists()

    monkeypatch.setattr(RUN_CONTRACT, "update_status", original_update_status)
    monkeypatch.setattr(RUN_CONTRACT, "_atomic_json_replace", original_atomic_replace)
    assert resume_failure_closure_if_present(run_root, run_id=RUN_ID) == "FAILED"
    assert sha256_file(intent_path) == intent_sha256
    completion = json.loads(
        (run_root / "failure" / "run_failure_completion_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert completion["stage"] == "GPU_SMOKE"
    assert completion["stop_reason"] == "ORIGINAL_FAILURE"
    assert completion["exit_code"] == 23
    assert completion["failure_intent_sha256"] == intent_sha256
    assert validate_terminal_chain(run_root, run_id=RUN_ID) == "FAILED"


def test_stale_partial_failure_ledger_is_not_sealed_or_mutated(tmp_path: Path) -> None:
    run_root = _base_run(tmp_path)
    _seal_failure(run_root)
    (run_root / "FAILED").unlink()
    (run_root / "late_unledgered.json").write_text("{}\n", encoding="utf-8")
    before = _file_hashes(run_root)
    with pytest.raises(RuntimeError, match="does not exactly cover"):
        resume_failure_closure_if_present(run_root, run_id=RUN_ID)
    assert not (run_root / "FAILED").exists()
    assert _file_hashes(run_root) == before


@pytest.mark.parametrize("missing_relative", ["status.json", "logs/events.jsonl"])
def test_missing_canonical_failure_base_cannot_be_legalized(
    tmp_path: Path, missing_relative: str
) -> None:
    run_root = _base_run(tmp_path)
    _seal_failure(run_root)
    (run_root / "FAILED").unlink()
    (run_root / "failure" / "artifact_checksums_at_failure.sha256").unlink()
    (run_root / missing_relative).unlink()
    before = _file_hashes(run_root)
    with pytest.raises(
        RuntimeError, match="failure (status record|closure lacks canonical log)"
    ):
        resume_failure_closure_if_present(run_root, run_id=RUN_ID)
    assert not (run_root / "FAILED").exists()
    assert not (run_root / "failure" / "artifact_checksums_at_failure.sha256").exists()
    assert _file_hashes(run_root) == before


def test_corrupt_structured_log_cannot_be_legalized(tmp_path: Path) -> None:
    run_root = _base_run(tmp_path)
    _seal_failure(run_root)
    (run_root / "FAILED").unlink()
    (run_root / "failure" / "artifact_checksums_at_failure.sha256").unlink()
    events = run_root / "logs" / "events.jsonl"
    events.write_bytes(events.read_bytes() + b"{truncated\n")
    before = _file_hashes(run_root)
    with pytest.raises(RuntimeError, match="structured log is invalid"):
        resume_failure_closure_if_present(run_root, run_id=RUN_ID)
    assert not (run_root / "FAILED").exists()
    assert _file_hashes(run_root) == before


@pytest.mark.parametrize(
    "node_kind",
    [
        "fifo",
        "internal_hardlink",
        "external_status_hardlink",
        "external_events_hardlink",
    ],
)
def test_noncanonical_preclosure_node_is_rejected_before_failure_intent(
    tmp_path: Path, node_kind: str
) -> None:
    run_root = _base_run(tmp_path)
    if node_kind == "fifo":
        os.mkfifo(run_root / "preclosure_fifo")
    elif node_kind == "internal_hardlink":
        os.link(run_root / "status.json", run_root / "status_alias.json")
    elif node_kind == "external_status_hardlink":
        target = run_root / "status.json"
        outside_alias = tmp_path / "outside_status_alias.json"
        os.link(target, outside_alias)
    else:
        target = run_root / "logs" / "events.jsonl"
        outside_alias = tmp_path / "outside_events_alias.jsonl"
        os.link(target, outside_alias)
    before = _file_hashes(run_root)
    outside_before = (
        (outside_alias.stat().st_ino, sha256_file(outside_alias))
        if node_kind.startswith("external_")
        else None
    )
    status_identity = (run_root / "status.json").stat().st_ino
    with pytest.raises(RuntimeError, match="(special node|hardlink)"):
        write_failed_sentinel(
            run_root,
            run_id=RUN_ID,
            stage="GPU_SMOKE",
            reason="PRECONTRACT_NODE_FAILURE",
            exit_code=31,
        )
    assert not (run_root / "failure" / "failure_closure_intent.json").exists()
    assert not (run_root / "summary.json").exists()
    assert not (run_root / "mk0_status.json").exists()
    assert not (run_root / "FAILED").exists()
    assert (run_root / "status.json").stat().st_ino == status_identity
    assert _file_hashes(run_root) == before
    if outside_before is not None:
        assert (
            outside_alias.stat().st_ino,
            sha256_file(outside_alias),
        ) == outside_before


def test_symlink_is_rejected_from_terminal_inventory(tmp_path: Path) -> None:
    run_root = _base_run(tmp_path)
    _seal_success(run_root)
    (run_root / "alias").symlink_to(run_root / "status.json")
    with pytest.raises(RuntimeError, match="symlink"):
        validate_terminal_chain(run_root, run_id=RUN_ID)


def test_fifo_is_rejected_from_terminal_inventory(tmp_path: Path) -> None:
    run_root = _base_run(tmp_path)
    _seal_success(run_root)
    os.mkfifo(run_root / "unexpected_fifo")
    with pytest.raises(RuntimeError, match="special node"):
        validate_terminal_chain(run_root, run_id=RUN_ID)


def test_internal_hardlink_alias_is_rejected(tmp_path: Path) -> None:
    run_root = _base_run(tmp_path)
    _seal_success(run_root)
    os.link(run_root / "status.json", run_root / "status_alias.json")
    with pytest.raises(RuntimeError, match="hardlink"):
        validate_terminal_chain(run_root, run_id=RUN_ID)


def test_external_hardlink_alias_is_rejected(tmp_path: Path) -> None:
    run_root = _base_run(tmp_path)
    _seal_success(run_root)
    os.link(run_root / "status.json", tmp_path / "outside_status_alias.json")
    with pytest.raises(RuntimeError, match="hardlink"):
        validate_terminal_chain(run_root, run_id=RUN_ID)


def test_sealed_gpu_wrong_requested_run_id_is_nonzero_and_immutable(
    tmp_path: Path,
) -> None:
    run_root = _base_run(tmp_path)
    _seal_success(run_root)
    before = _file_hashes(run_root)
    gpu = _load_script("run_mk0_gpu_smoke.py", "mk0_gpu_terminal_test")
    exit_code = gpu.main(
        [
            "--output-dir",
            str(run_root / "artifacts" / "mk0"),
            "--run-id",
            "MK0_wrong_request",
            "--goal-sha256",
            "0" * 64,
            "--implementation-commit",
            "0" * 40,
            "--run-manifest",
            str(run_root / "run_manifest.json"),
            "--preflight-record",
            str(run_root / "unused-preflight.json"),
            "--snapshot-dir",
            str(run_root / "unused-snapshot"),
            "--device",
            "cuda:0",
        ]
    )
    assert exit_code == 2
    assert _file_hashes(run_root) == before


def test_sealed_finalizer_wrong_requested_run_id_is_nonzero_and_immutable(
    tmp_path: Path,
) -> None:
    run_root = _base_run(tmp_path)
    _seal_success(run_root)
    before = _file_hashes(run_root)
    finalizer = _load_script(
        "finalize_mk0_acceptance.py", "mk0_finalizer_terminal_test"
    )
    exit_code = finalizer.main(
        [
            "--run-root",
            str(run_root),
            "--run-id",
            "MK0_wrong_request",
            "--goal-sha256",
            "0" * 64,
            "--implementation-commit",
            "0" * 40,
            "--fm0-closure-root",
            str(run_root / "unused-fm0"),
            "--d1-data",
            str(run_root / "unused-d1"),
            "--d1-ledger",
            str(run_root / "unused-ledger"),
            "--preflight-record",
            str(run_root / "unused-preflight.json"),
        ]
    )
    assert exit_code == 2
    assert _file_hashes(run_root) == before


def test_bound_terminal_hash_tamper_fails_closed(tmp_path: Path) -> None:
    run_root = _base_run(tmp_path)
    _seal_success(run_root)
    status = run_root / "status.json"
    status.write_bytes(status.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="DONE sentinel/hash chain is invalid"):
        validate_terminal_chain(run_root, run_id=RUN_ID)


def test_truncated_terminal_and_ledger_fail_closed(tmp_path: Path) -> None:
    run_root = _base_run(tmp_path)
    _seal_failure(run_root)
    (run_root / "FAILED").write_text(f"{RUN_ID}\nGPU_SMOKE\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="FAILED sentinel/hash chain is invalid"):
        validate_terminal_chain(run_root, run_id=RUN_ID)

    run_root_2 = _base_run(tmp_path / "second")
    _seal_success(run_root_2)
    ledger = run_root_2 / "artifact_checksums.sha256"
    ledger.write_bytes(ledger.read_bytes()[:-1])
    with pytest.raises(RuntimeError):
        validate_terminal_chain(run_root_2, run_id=RUN_ID)
