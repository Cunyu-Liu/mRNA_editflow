from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.execution import run_audited_command
from scripts.data.validate_d1_acceptance import (
    _prelaunch_file_binding,
    _validate_builder_audit,
)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Audit Test")
    _git(repo, "config", "user.email", "audit@example.invalid")
    (repo / "tracked.txt").write_text("committed\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "initial")
    return repo


def _passed_cuda_health() -> dict:
    return {
        **{field: True for field in run_audited_command.CUDA_HEALTH_FIELDS},
        "passed": True,
        "device": "cuda",
        "error": None,
    }


def _failed_cuda_health() -> dict:
    return {
        **{field: False for field in run_audited_command.CUDA_HEALTH_FIELDS},
        "passed": False,
        "device": None,
        "error": "RuntimeError: CUDA_UNAVAILABLE",
    }


def test_d1_builder_audit_binds_stdout_to_live_build_manifest(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path)
    script = repo / "scripts/data/build_d1_utr_benchmark.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "\n".join(
            (
                "import argparse, json",
                "from pathlib import Path",
                "p=argparse.ArgumentParser()",
                "p.add_argument('--config', required=True)",
                "p.add_argument('--output-root', required=True)",
                "p.add_argument('--artifact-root', required=True)",
                "a=p.parse_args()",
                "o=Path(a.output_root); o.mkdir(parents=True)",
                "m={'schema_version':'d1_build_manifest_v2',"
                "'config_path':str(Path(a.config).resolve())}",
                "(o/'build_manifest.json').write_text("
                "json.dumps(m,sort_keys=True)+'\\n')",
                "print(json.dumps(m,sort_keys=True))",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    config = repo / "configs/d1.json"
    config.parent.mkdir()
    config.write_text("{}\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "builder fixture")
    config.write_text('{"tracked_dirty":true}\n', encoding="utf-8")
    output_root = tmp_path / "D1"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    audit_root = tmp_path / "audit/attempt_001"
    audit_root.parent.mkdir()
    command = [
        sys.executable,
        "scripts/data/build_d1_utr_benchmark.py",
        "--config",
        str(config),
        "--output-root",
        str(output_root),
        "--artifact-root",
        str(artifact_root),
    ]
    assert (
        run_audited_command.run_audited_command(
            run_root=audit_root,
            project_root=repo,
            working_directory=repo,
            command=command,
            workload_class=run_audited_command.NON_NEURAL,
        )
        == 0
    )

    valid = _validate_builder_audit(
        output_root,
        artifact_root,
        audit_root,
        required=True,
    )
    assert valid["passed"] is True
    prelaunch_config = _prelaunch_file_binding(
        audit_root,
        valid["git_prelaunch_snapshot"],
        "configs/d1.json",
        expected_bytes=config.stat().st_size,
        expected_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
    )
    assert prelaunch_config["passed"] is True
    assert prelaunch_config["source"] == "captured_head_plus_binary_diff"
    stdout = audit_root / "logs/stdout.log"
    stdout.write_text('{"forged":true}\n', encoding="utf-8")
    invalid = _validate_builder_audit(
        output_root,
        artifact_root,
        audit_root,
        required=True,
    )
    assert invalid["passed"] is False
    assert invalid["checks"]["evidence_path_bytes_sha_recursive"] is False
    assert invalid["checks"]["stdout_one_json_line_equals_build_manifest"] is False


def test_non_neural_run_captures_exact_process_git_and_logs(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path)
    (repo / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    untracked = repo / "untracked.txt"
    untracked.write_bytes(b"untracked-content\n")
    run_root = tmp_path / "audit" / "attempt_002"
    run_root.parent.mkdir()
    command = [
        sys.executable,
        "-c",
        (
            "import json, os, sys; "
            "print(json.dumps({'cuda': os.environ.get('CUDA_VISIBLE_DEVICES'), "
            "'require': os.environ.get('EDITFLOW_REQUIRE_CUDA')})); "
            "print('captured-stderr', file=sys.stderr)"
        ),
    ]

    def forbidden_probe() -> dict:
        raise AssertionError("non-neural work must not probe CUDA")

    rc = run_audited_command.run_audited_command(
        run_root=run_root,
        project_root=repo,
        command=command,
        workload_class=run_audited_command.NON_NEURAL,
        cuda_probe=forbidden_probe,
    )

    assert rc == 0
    manifest = json.loads(
        (run_root / "audit_manifest.json").read_text(encoding="utf-8")
    )
    completion = json.loads((run_root / "completion.json").read_text(encoding="utf-8"))
    invocation = json.loads((run_root / "invocation.json").read_text(encoding="utf-8"))
    process = json.loads((run_root / "process.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "COMMAND_COMPLETED"
    assert manifest["argv"] == command
    assert manifest["shell_used"] is False
    assert manifest["observed_process_exit_code"] == 0
    assert manifest["claim_boundary"]["command_exit_zero_is_phase_gate"] is False
    assert completion["observed_process_exit_code"] == 0
    assert completion["wrapper_exit_code"] == 0
    assert process["child_pid"] == manifest["child_pid"]
    assert process["wrapper_pid"] == os.getpid()
    assert invocation["started_at_utc"] == manifest["started_at_utc"]
    assert manifest["cuda"]["applicability"] == "NOT_APPLICABLE_NON_NEURAL_WORKLOAD"
    assert manifest["cuda"]["probe_executed"] is False
    stdout_payload = json.loads(
        (run_root / "logs/stdout.log").read_text(encoding="utf-8")
    )
    assert stdout_payload == {"cuda": "", "require": "0"}
    assert (run_root / "logs/stderr.log").read_text(
        encoding="utf-8"
    ).strip() == "captured-stderr"

    git_snapshot = manifest["git_prelaunch_snapshot"]
    assert git_snapshot["head"] == _git(repo, "rev-parse", "HEAD")
    assert git_snapshot["clean"] is False
    assert (
        git_snapshot["component_hashes"]["diff_head_binary"]
        == hashlib.sha256(
            (run_root / "git/diff.head.binary.patch").read_bytes()
        ).hexdigest()
    )
    untracked_manifest = json.loads(
        (run_root / "git/untracked_content_hashes.json").read_text(encoding="utf-8")
    )
    entry = next(
        item
        for item in untracked_manifest["entries"]
        if item["path"] == "untracked.txt"
    )
    assert entry["sha256"] == hashlib.sha256(untracked.read_bytes()).hexdigest()
    assert (
        manifest["evidence"]["stdout"]["sha256"]
        == hashlib.sha256((run_root / "logs/stdout.log").read_bytes()).hexdigest()
    )


def test_existing_run_root_is_refused_without_overwrite(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    run_root = tmp_path / "existing"
    run_root.mkdir()
    sentinel = run_root / "sentinel.txt"
    sentinel.write_text("preserve me\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_audited_command.run_audited_command(
            run_root=run_root,
            project_root=repo,
            command=[sys.executable, "-c", "raise SystemExit(0)"],
            workload_class=run_audited_command.NON_NEURAL,
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve me\n"
    assert sorted(path.name for path in run_root.iterdir()) == ["sentinel.txt"]


def test_git_snapshot_failure_prevents_command_and_keeps_failure_bundle(
    tmp_path: Path,
) -> None:
    not_a_repo = tmp_path / "not-a-repository"
    not_a_repo.mkdir()
    run_root = tmp_path / "runs" / "git-failed"
    run_root.parent.mkdir()
    marker = tmp_path / "must-not-exist"

    rc = run_audited_command.run_audited_command(
        run_root=run_root,
        project_root=not_a_repo,
        command=[
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ],
        workload_class=run_audited_command.NON_NEURAL,
    )

    assert rc == run_audited_command.AUDIT_FAILURE_EXIT
    assert not marker.exists()
    manifest = json.loads(
        (run_root / "audit_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["state"] == "FAILED_WITH_EVIDENCE"
    assert manifest["stop_reason"].startswith("GIT_PRELAUNCH_SNAPSHOT_FAILED_")
    assert manifest["child_pid"] is None
    assert manifest["git_prelaunch_snapshot"] is None
    assert (run_root / "git/snapshot_error.json").is_file()
    assert (run_root / "logs/stdout.log").read_bytes() == b""
    assert (run_root / "logs/stderr.log").read_bytes() == b""


def test_nonzero_child_exit_is_preserved_not_reclassified_as_gate(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path)
    run_root = tmp_path / "runs" / "attempt"
    run_root.parent.mkdir()

    rc = run_audited_command.run_audited_command(
        run_root=run_root,
        project_root=repo,
        command=[
            sys.executable,
            "-c",
            "import sys; print('builder failed', file=sys.stderr); sys.exit(2)",
        ],
        workload_class=run_audited_command.NON_NEURAL,
    )

    assert rc == 2
    manifest = json.loads(
        (run_root / "audit_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["state"] == "FAILED_WITH_EVIDENCE"
    assert manifest["stop_reason"] == "COMMAND_EXIT_2"
    assert manifest["observed_process_exit_code"] == 2
    assert manifest["wrapper_exit_code"] == 2
    assert "builder failed" in (run_root / "logs/stderr.log").read_text(
        encoding="utf-8"
    )


def test_neural_cuda_failure_prevents_command_and_keeps_evidence(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path)
    run_root = tmp_path / "runs" / "cuda-failed"
    run_root.parent.mkdir()
    marker = tmp_path / "must-not-exist"

    rc = run_audited_command.run_audited_command(
        run_root=run_root,
        project_root=repo,
        command=[
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ],
        workload_class=run_audited_command.NEURAL,
        cuda_probe=_failed_cuda_health,
    )

    assert rc == run_audited_command.CUDA_FAILURE_EXIT
    assert not marker.exists()
    manifest = json.loads(
        (run_root / "audit_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["state"] == "FAILED_WITH_EVIDENCE"
    assert manifest["stop_reason"] == "CUDA_PREFLIGHT_FAILED"
    assert manifest["child_pid"] is None
    assert manifest["observed_process_exit_code"] is None
    assert manifest["cuda"]["preflight"]["passed"] is False
    assert (run_root / "logs/cuda_preflight.json").is_file()
    assert (run_root / "logs/stdout.log").read_bytes() == b""
    assert (run_root / "logs/stderr.log").read_bytes() == b""


def test_neural_zero_exit_without_actual_cuda_health_fails_closed(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path)
    run_root = tmp_path / "runs" / "missing-health"
    run_root.parent.mkdir()

    rc = run_audited_command.run_audited_command(
        run_root=run_root,
        project_root=repo,
        command=[sys.executable, "-c", "print('no health emitted')"],
        workload_class=run_audited_command.NEURAL,
        cuda_probe=_passed_cuda_health,
    )

    assert rc == run_audited_command.CUDA_FAILURE_EXIT
    manifest = json.loads(
        (run_root / "audit_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["observed_process_exit_code"] == 0
    assert manifest["stop_reason"] == "ACTUAL_COMMAND_CUDA_HEALTH_FAILED"
    assert (
        manifest["cuda"]["actual_command_health"]["error"]
        == "MISSING_ACTUAL_COMMAND_CUDA_HEALTH"
    )


def test_neural_success_requires_actual_cuda_health_from_child(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path)
    run_root = tmp_path / "runs" / "healthy"
    run_root.parent.mkdir()
    health = _passed_cuda_health()
    command = [
        sys.executable,
        "-c",
        (
            "import json, os; "
            "open(os.environ['EDITFLOW_CUDA_HEALTH_FILE'], 'x').write("
            f"json.dumps({health!r}) + '\\n')"
        ),
    ]

    rc = run_audited_command.run_audited_command(
        run_root=run_root,
        project_root=repo,
        command=command,
        workload_class=run_audited_command.NEURAL,
        cuda_probe=_passed_cuda_health,
    )

    assert rc == 0
    manifest = json.loads(
        (run_root / "audit_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["state"] == "COMMAND_COMPLETED"
    assert manifest["cuda"]["preflight"]["passed"] is True
    assert manifest["cuda"]["actual_command_health"]["passed"] is True
