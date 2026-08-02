"""Regression tests for fail-closed MK0 repair-run lineage binding."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PARENT_RUN_ID = "MK0_utrlm_mathkernel_tiny_20260802T083317Z_1d879e0_s20260802"
CHILD_RUN_ID = "MK0_utrlm_mathkernel_tiny_20260802T093317Z_abcdef0_s20260802"
GOAL_SHA256 = "3a3a654ca5c10a988eca897bff40be2e0b45c841f744f7423fdfd60b298b5791"
PARENT_COMMIT = "1d879e030c6ab2f86fa28c6bb491d1fb77059b9a"


def _load_cpu_runner():
    path = ROOT / "scripts" / "mk0" / "run_mk0_cpu_acceptance.py"
    spec = importlib.util.spec_from_file_location("mk0_cpu_repair_lineage_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CPU = _load_cpu_runner()


def _load_finalizer():
    path = ROOT / "scripts" / "mk0" / "finalize_mk0_acceptance.py"
    spec = importlib.util.spec_from_file_location("mk0_finalizer_lineage_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FINALIZER = _load_finalizer()


def _parent_manifest(root: Path, *, run_id: str = PARENT_RUN_ID) -> dict:
    return {
        "schema_version": "mk0_run_manifest_v3",
        "run_id": run_id,
        "task_id": "MK0-01",
        "phase": "MK0",
        "run_root": str(root),
        "goal_sha256": GOAL_SHA256,
        "contract": {"sha256": GOAL_SHA256},
        "implementation_commit": PARENT_COMMIT,
        "code": {"commit": PARENT_COMMIT},
        "source_binding": {"git_commit": PARENT_COMMIT},
    }


def _write_manifest(root: Path, payload: dict) -> None:
    (root / "run_manifest.json").write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parent_root(tmp_path: Path, *, run_id: str = PARENT_RUN_ID) -> Path:
    root = tmp_path / run_id
    root.mkdir()
    _write_manifest(root, _parent_manifest(root, run_id=run_id))
    return root


def _validate(
    canonical_parent: Path,
    *,
    child_run_id: str = CHILD_RUN_ID,
    parent_run_id: str | None = PARENT_RUN_ID,
):
    return CPU.validate_parent_run_lineage(
        child_run_id,
        parent_run_id,
        goal_sha256=GOAL_SHA256,
        canonical_parent=canonical_parent,
    )


def test_repair_child_binds_unsealed_parent_failure_evidence(tmp_path: Path) -> None:
    parent_root = _parent_root(tmp_path)
    failure = parent_root / "artifacts" / "mk0" / "mk0_gpu_smoke_failure.json"
    failure.parent.mkdir(parents=True)
    failure.write_text('{"failure": "observed"}\n', encoding="utf-8")

    binding = _validate(tmp_path)

    assert binding is not None
    assert binding["schema_version"] == "mk0_parent_run_binding_v1"
    assert binding["run_id"] == PARENT_RUN_ID
    assert binding["run_root"] == str(parent_root)
    assert binding["observed_classification"] == "UNSEALED_FAILED_EVIDENCE"
    manifest_binding = binding["registration_manifest"]
    assert manifest_binding["path"] == str(parent_root / "run_manifest.json")
    assert manifest_binding["sha256"] == CPU.sha256_file(
        parent_root / "run_manifest.json"
    )
    assert (
        manifest_binding["size_bytes"]
        == (parent_root / "run_manifest.json").stat().st_size
    )
    evidence = binding["failure_evidence"]
    assert evidence["file_count"] == 1
    assert evidence["total_size_bytes"] == failure.stat().st_size
    assert evidence["files"] == [
        {
            "path": "artifacts/mk0/mk0_gpu_smoke_failure.json",
            "size_bytes": failure.stat().st_size,
            "sha256": CPU.sha256_file(failure),
        }
    ]
    assert evidence["files_sha256"] == CPU.sha_record(evidence["files"])
    assert (
        FINALIZER.verify_parent_run_binding(
            binding,
            parent_run_id=PARENT_RUN_ID,
            goal_sha256=GOAL_SHA256,
            canonical_parent=tmp_path,
        )
        == binding
    )


def test_finalizer_rejects_parent_failure_bytes_changed_after_cpu_binding(
    tmp_path: Path,
) -> None:
    parent_root = _parent_root(tmp_path)
    failure = parent_root / "failure" / "gpu_smoke_failure.json"
    failure.parent.mkdir()
    failure.write_text('{"failure":"first"}\n', encoding="utf-8")
    binding = _validate(tmp_path)
    assert binding is not None

    failure.write_text('{"failure":"other"}\n', encoding="utf-8")
    with pytest.raises(FINALIZER.FinalizeFailure, match="live parent"):
        FINALIZER.verify_parent_run_binding(
            binding,
            parent_run_id=PARENT_RUN_ID,
            goal_sha256=GOAL_SHA256,
            canonical_parent=tmp_path,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda binding: binding.pop("failure_evidence"),
        lambda binding: binding.update(observed_classification="DONE"),
        lambda binding: binding["failure_evidence"].update(file_count=0),
        lambda binding: binding["failure_evidence"].update(file_count=True),
        lambda binding: binding["failure_evidence"].update(file_count=1.0),
        lambda binding: binding["registration_manifest"].update(size_bytes=1),
    ],
)
def test_finalizer_rejects_forged_parent_binding(tmp_path: Path, mutation) -> None:
    parent_root = _parent_root(tmp_path)
    (parent_root / "FAILED").write_text("FAILED\n", encoding="utf-8")
    binding = _validate(tmp_path)
    assert binding is not None
    mutation(binding)

    with pytest.raises(FINALIZER.FinalizeFailure, match="live parent"):
        FINALIZER.verify_parent_run_binding(
            binding,
            parent_run_id=PARENT_RUN_ID,
            goal_sha256=GOAL_SHA256,
            canonical_parent=tmp_path,
        )


@pytest.mark.parametrize("reader", ["cpu", "finalizer"])
def test_parent_snapshot_rejects_path_replacement_during_fd_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reader: str,
) -> None:
    target = tmp_path / f"{reader}.json"
    replacement = tmp_path / f"{reader}.replacement.json"
    target.write_text('{"phase":"MK0"}\n', encoding="utf-8")
    replacement.write_text('{"phase":"XX0"}\n', encoding="utf-8")
    original_read = os.read
    replaced = False

    def swapping_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        data = original_read(descriptor, size)
        if data and not replaced:
            replaced = True
            os.replace(replacement, target)
        return data

    monkeypatch.setattr(os, "read", swapping_read)
    if reader == "cpu":
        with pytest.raises(CPU.AcceptanceFailure, match="changed while it was read"):
            CPU._read_ordinary_unlinked_bytes(target, label="race fixture")
    else:
        with pytest.raises(
            FINALIZER.FinalizeFailure, match="changed while it was read"
        ):
            FINALIZER._ordinary_unlinked_file_snapshot(
                target,
                label="race fixture",
            )
    assert replaced is True


def test_repair_child_binds_terminal_failed_parent(tmp_path: Path) -> None:
    parent_root = _parent_root(tmp_path)
    (parent_root / "FAILED").write_text("FAILED\n", encoding="utf-8")

    binding = _validate(tmp_path)

    assert binding is not None
    assert binding["observed_classification"] == "FAILED"
    assert binding["failure_evidence"]["files"][0]["path"] == "FAILED"


def test_nonrepair_run_has_no_synthetic_parent_binding(tmp_path: Path) -> None:
    assert (
        CPU.validate_parent_run_lineage(
            "not-consulted-without-parent",
            None,
            goal_sha256=None,
            canonical_parent=tmp_path,
        )
        is None
    )


@pytest.mark.parametrize(
    ("child_run_id", "parent_run_id", "message"),
    [
        (CHILD_RUN_ID, "not-a-formal-run", "parent run ID"),
        ("not-a-formal-run", PARENT_RUN_ID, "child run ID"),
        (PARENT_RUN_ID, CHILD_RUN_ID, "must precede"),
        (CHILD_RUN_ID, CHILD_RUN_ID, "must precede"),
    ],
)
def test_repair_lineage_rejects_invalid_or_nonchronological_ids(
    tmp_path: Path,
    child_run_id: str,
    parent_run_id: str,
    message: str,
) -> None:
    with pytest.raises(CPU.AcceptanceFailure, match=message):
        CPU.validate_parent_run_lineage(
            child_run_id,
            parent_run_id,
            goal_sha256=GOAL_SHA256,
            canonical_parent=tmp_path,
        )


def test_repair_lineage_rejects_missing_or_substituted_parent(tmp_path: Path) -> None:
    with pytest.raises(CPU.AcceptanceFailure, match="parent run root is absent"):
        CPU.validate_parent_run_lineage(
            CHILD_RUN_ID,
            PARENT_RUN_ID,
            goal_sha256=GOAL_SHA256,
            canonical_parent=tmp_path,
        )

    parent_root = _parent_root(tmp_path)
    manifest = _parent_manifest(parent_root)
    manifest["run_id"] = CHILD_RUN_ID
    _write_manifest(parent_root, manifest)
    with pytest.raises(CPU.AcceptanceFailure, match="manifest ID drift"):
        _validate(tmp_path)


def test_repair_lineage_rejects_contradictory_terminal_sentinels(
    tmp_path: Path,
) -> None:
    parent_root = _parent_root(tmp_path)
    (parent_root / "DONE").write_text("DONE\n", encoding="utf-8")
    (parent_root / "FAILED").write_text("FAILED\n", encoding="utf-8")

    with pytest.raises(CPU.AcceptanceFailure, match="contradictory"):
        _validate(tmp_path)


def test_repair_lineage_rejects_done_and_nonterminal_parents(tmp_path: Path) -> None:
    nonterminal_parent = tmp_path / "nonterminal"
    nonterminal_parent.mkdir()
    _parent_root(nonterminal_parent)
    with pytest.raises(CPU.AcceptanceFailure, match="no failure evidence"):
        _validate(nonterminal_parent)

    done_parent = tmp_path / "done"
    done_parent.mkdir()
    root = _parent_root(done_parent)
    (root / "DONE").write_text("DONE\n", encoding="utf-8")
    with pytest.raises(CPU.AcceptanceFailure, match="terminal DONE"):
        _validate(done_parent)


def test_repair_lineage_failure_evidence_digest_detects_drift(tmp_path: Path) -> None:
    parent_root = _parent_root(tmp_path)
    failure = parent_root / "failure" / "cpu_acceptance_failure.json"
    failure.parent.mkdir()
    failure.write_text('{"failure":"first"}\n', encoding="utf-8")
    first = _validate(tmp_path)
    assert first is not None

    failure.write_text('{"failure":"other"}\n', encoding="utf-8")
    second = _validate(tmp_path)
    assert second is not None
    assert first["registration_manifest"] == second["registration_manifest"]
    assert (
        first["failure_evidence"]["files_sha256"]
        != second["failure_evidence"]["files_sha256"]
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(schema_version="mk0_run_manifest_v2"), "schema"),
        (lambda value: value.update(task_id="EF0-01"), "task"),
        (lambda value: value.update(phase="EF0"), "phase"),
        (lambda value: value.update(run_root="/tmp/substituted"), "root"),
        (lambda value: value.update(goal_sha256="0" * 64), "Goal"),
        (lambda value: value["contract"].update(sha256="0" * 64), "Goal"),
        (lambda value: value.update(implementation_commit="invalid"), "commit"),
        (lambda value: value["code"].update(commit="0" * 40), "code"),
        (
            lambda value: value["source_binding"].update(git_commit="0" * 40),
            "source",
        ),
        (
            lambda value: (
                value.update(implementation_commit="0" * 40),
                value["code"].update(commit="0" * 40),
                value["source_binding"].update(git_commit="0" * 40),
            ),
            "short SHA",
        ),
    ],
)
def test_repair_lineage_rejects_malformed_parent_manifest(
    tmp_path: Path, mutation, message: str
) -> None:
    parent_root = _parent_root(tmp_path)
    (parent_root / "FAILED").write_text("FAILED\n", encoding="utf-8")
    manifest = _parent_manifest(parent_root)
    mutation(manifest)
    _write_manifest(parent_root, manifest)

    with pytest.raises(CPU.AcceptanceFailure, match=message):
        _validate(tmp_path)


def test_repair_lineage_rejects_symlink_parent_root(tmp_path: Path) -> None:
    actual_parent = tmp_path / "actual-parent"
    actual_parent.mkdir()
    _write_manifest(actual_parent, _parent_manifest(actual_parent))
    (actual_parent / "FAILED").write_text("FAILED\n", encoding="utf-8")
    (tmp_path / PARENT_RUN_ID).symlink_to(actual_parent, target_is_directory=True)

    with pytest.raises(CPU.AcceptanceFailure, match="ordinary directory"):
        _validate(tmp_path)


def test_repair_lineage_rejects_symlink_or_hardlinked_manifest(tmp_path: Path) -> None:
    symlink_case = tmp_path / "symlink"
    symlink_case.mkdir()
    parent_root = symlink_case / PARENT_RUN_ID
    parent_root.mkdir()
    manifest_target = symlink_case / "manifest-target.json"
    _write_manifest(symlink_case, _parent_manifest(parent_root))
    (symlink_case / "run_manifest.json").replace(manifest_target)
    (parent_root / "run_manifest.json").symlink_to(manifest_target)
    (parent_root / "FAILED").write_text("FAILED\n", encoding="utf-8")
    with pytest.raises(CPU.AcceptanceFailure, match="ordinary unlinked file"):
        _validate(symlink_case)

    hardlink_case = tmp_path / "hardlink"
    hardlink_case.mkdir()
    parent_root = _parent_root(hardlink_case)
    alias = hardlink_case / "manifest-alias.json"
    os.link(parent_root / "run_manifest.json", alias)
    (parent_root / "FAILED").write_text("FAILED\n", encoding="utf-8")
    with pytest.raises(CPU.AcceptanceFailure, match="ordinary unlinked file"):
        _validate(hardlink_case)


def test_repair_lineage_rejects_nonregular_failure_evidence(tmp_path: Path) -> None:
    symlink_case = tmp_path / "symlink"
    symlink_case.mkdir()
    parent_root = _parent_root(symlink_case)
    failure_root = parent_root / "failure"
    failure_root.mkdir()
    target = symlink_case / "failure-target.json"
    target.write_text('{"failure":true}\n', encoding="utf-8")
    (failure_root / "linked.json").symlink_to(target)
    with pytest.raises(CPU.AcceptanceFailure, match="contains a symlink"):
        _validate(symlink_case)

    hardlink_case = tmp_path / "hardlink"
    hardlink_case.mkdir()
    parent_root = _parent_root(hardlink_case)
    failure_root = parent_root / "failure"
    failure_root.mkdir()
    evidence = failure_root / "failure.json"
    evidence.write_text('{"failure":true}\n', encoding="utf-8")
    os.link(evidence, failure_root / "failure-alias.json")
    with pytest.raises(CPU.AcceptanceFailure, match="ordinary unlinked file"):
        _validate(hardlink_case)

    special_case = tmp_path / "special"
    special_case.mkdir()
    parent_root = _parent_root(special_case)
    failure_root = parent_root / "failure"
    failure_root.mkdir()
    os.mkfifo(failure_root / "failure.fifo")
    with pytest.raises(CPU.AcceptanceFailure, match="special file"):
        _validate(special_case)


def test_repair_lineage_rejects_empty_failure_evidence(tmp_path: Path) -> None:
    parent_root = _parent_root(tmp_path)
    failure = parent_root / "failure" / "empty.json"
    failure.parent.mkdir()
    failure.touch()

    with pytest.raises(CPU.AcceptanceFailure, match="file is empty"):
        _validate(tmp_path)


def test_failure_reason_is_nonempty_for_keyboard_interrupt() -> None:
    assert CPU.failure_reason(KeyboardInterrupt()) == "KeyboardInterrupt"
    assert CPU.failure_reason(RuntimeError("closure failed")) == (
        "RuntimeError: closure failed"
    )
    assert CPU.failure_reason(ValueError("   ")) == "ValueError"
