from __future__ import annotations

import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import time
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
    index_flags = audit_root / "git/index_flags.json"
    index_flags_original = index_flags.read_bytes()
    index_flags.write_text(
        '{"schema_version":"git_index_flags.v1","entry_count":0,'
        '"entries":[],"unsafe_entries":[],"all_entries_normal":true}\n',
        encoding="utf-8",
    )
    invalid_index = _validate_builder_audit(
        output_root,
        artifact_root,
        audit_root,
        required=True,
    )
    assert invalid_index["passed"] is False
    assert (
        invalid_index["checks"]["git_prelaunch_artifacts_path_bytes_sha_recursive"]
        is False
    )
    assert invalid_index["checks"]["git_index_flags_manifest_semantics"] is False
    index_flags.write_bytes(index_flags_original)

    explicit_manifest = audit_root / "git/explicit_prelaunch_files.json"
    explicit_original = explicit_manifest.read_bytes()
    explicit_manifest.write_text(
        '{"schema_version":"git_explicit_prelaunch_files.v1",'
        '"entry_count":1,"entries":[{"kind":"symbolic_link_target",'
        '"path":"configs/d1.json","bytes":1,"sha256":"' + ("0" * 64) + '"}]}\n',
        encoding="utf-8",
    )
    invalid_explicit = _validate_builder_audit(
        output_root,
        artifact_root,
        audit_root,
        required=True,
    )
    assert invalid_explicit["passed"] is False
    assert (
        invalid_explicit["checks"]["git_prelaunch_artifacts_path_bytes_sha_recursive"]
        is False
    )
    assert (
        invalid_explicit["checks"]["git_explicit_prelaunch_manifest_semantics"] is False
    )
    explicit_manifest.write_bytes(explicit_original)

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


def test_expected_git_binding_mismatch_prevents_child_launch(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path)
    approved = run_audited_command._capture_git_snapshot(repo)
    (repo / "tracked.txt").write_text("drifted after approval\n", encoding="utf-8")
    run_root = tmp_path / "runs" / "binding-mismatch"
    run_root.parent.mkdir()
    marker = tmp_path / "child-must-not-start"

    rc = run_audited_command.run_audited_command(
        run_root=run_root,
        project_root=repo,
        command=[
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ],
        workload_class=run_audited_command.NON_NEURAL,
        expected_git_head=approved["head"],
        expected_git_dirty_state_sha256=approved["dirty_state_sha256"],
    )

    assert rc == run_audited_command.AUDIT_FAILURE_EXIT
    assert not marker.exists()
    manifest = json.loads(
        (run_root / "audit_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["state"] == "FAILED_WITH_EVIDENCE"
    assert manifest["stop_reason"] == "GIT_PRELAUNCH_BINDING_MISMATCH"
    assert manifest["child_pid"] is None
    failure = json.loads(
        (run_root / "git/prelaunch_binding_failure.json").read_text(encoding="utf-8")
    )
    assert failure["command_started"] is False
    assert failure["checks"]["head"]["passed"] is True
    assert failure["checks"]["dirty_state_sha256"]["passed"] is False
    invocation = json.loads((run_root / "invocation.json").read_text(encoding="utf-8"))
    assert invocation["command_started"] is False


@pytest.mark.parametrize(
    ("update_index_flag", "expected_tag"),
    (
        ("--assume-unchanged", "h"),
        ("--skip-worktree", "S"),
    ),
)
def test_nonstandard_index_flag_prevents_child_launch(
    tmp_path: Path,
    update_index_flag: str,
    expected_tag: str,
) -> None:
    repo = _repository(tmp_path)
    _git(repo, "update-index", update_index_flag, "tracked.txt")
    (repo / "tracked.txt").write_text("hidden modified bytes\n", encoding="utf-8")
    run_root = tmp_path / "runs" / update_index_flag.removeprefix("--")
    run_root.parent.mkdir()
    marker = tmp_path / "child-must-not-start"

    rc = run_audited_command.run_audited_command(
        run_root=run_root,
        project_root=repo,
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
    assert manifest["stop_reason"] == "GIT_PRELAUNCH_BINDING_MISMATCH"
    failure = json.loads(
        (run_root / "git/prelaunch_binding_failure.json").read_text(encoding="utf-8")
    )
    assert failure["checks"]["index_flags_safe"] == {
        "applicable": True,
        "expected": True,
        "observed": False,
        "passed": False,
    }
    flags = json.loads((run_root / "git/index_flags.json").read_text(encoding="utf-8"))
    assert flags["all_entries_normal"] is False
    assert flags["unsafe_entries"][0]["tag"] == expected_tag


def test_matching_expected_git_binding_allows_child_launch(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    approved = run_audited_command._capture_git_snapshot(repo)
    run_root = tmp_path / "runs" / "binding-match"
    run_root.parent.mkdir()
    marker = tmp_path / "child-started"

    rc = run_audited_command.run_audited_command(
        run_root=run_root,
        project_root=repo,
        command=[
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ],
        workload_class=run_audited_command.NON_NEURAL,
        expected_git_head=approved["head"],
        expected_git_dirty_state_sha256=approved["dirty_state_sha256"],
    )

    assert rc == 0
    assert marker.is_file()
    completion = json.loads((run_root / "completion.json").read_text(encoding="utf-8"))
    assert completion["state"] == "COMMAND_COMPLETED"


def test_explicit_prelaunch_binding_captures_ignored_control_file(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path)
    (repo / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-m", "ignore generated artifacts")
    relative = "artifacts/stages/STAGE/D1/input_inventory.json"
    inventory = repo / relative
    inventory.parent.mkdir(parents=True)
    inventory.write_text('{"selection_is_label_independent":true}\n', encoding="utf-8")
    assert relative not in _git(repo, "ls-files", "--others", "--exclude-standard")
    run_root = tmp_path / "runs" / "explicit-prelaunch"
    run_root.parent.mkdir()

    rc = run_audited_command.run_audited_command(
        run_root=run_root,
        project_root=repo,
        command=[sys.executable, "-c", "raise SystemExit(0)"],
        workload_class=run_audited_command.NON_NEURAL,
        prelaunch_bind_files=[relative],
    )

    assert rc == 0
    manifest = json.loads(
        (run_root / "audit_manifest.json").read_text(encoding="utf-8")
    )
    binding = _prelaunch_file_binding(
        run_root,
        manifest["git_prelaunch_snapshot"],
        relative,
        expected_bytes=inventory.stat().st_size,
        expected_sha256=hashlib.sha256(inventory.read_bytes()).hexdigest(),
    )
    assert binding["passed"] is True
    assert binding["source"] == "explicit_prelaunch_file_manifest"
    explicit = json.loads(
        (run_root / "git/explicit_prelaunch_files.json").read_text(encoding="utf-8")
    )
    assert explicit["entries"] == [
            {
                "path": relative,
                "kind": "regular_file",
                "mode": stat.S_IMODE(inventory.stat().st_mode),
                "bytes": inventory.stat().st_size,
            "sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(),
        }
    ]


def test_explicit_prelaunch_binding_rejects_symlink_before_child(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path)
    target = repo / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    link = repo / "linked.json"
    link.symlink_to(target.name)
    run_root = tmp_path / "runs" / "explicit-symlink"
    run_root.parent.mkdir()
    marker = tmp_path / "child-must-not-start"

    rc = run_audited_command.run_audited_command(
        run_root=run_root,
        project_root=repo,
        command=[
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ],
        workload_class=run_audited_command.NON_NEURAL,
        prelaunch_bind_files=["linked.json"],
    )

    assert rc == run_audited_command.AUDIT_FAILURE_EXIT
    assert not marker.exists()
    manifest = json.loads(
        (run_root / "audit_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["state"] == "FAILED_WITH_EVIDENCE"
    assert manifest["stop_reason"].startswith("GIT_PRELAUNCH_SNAPSHOT_FAILED_")


def test_explicit_prelaunch_recheck_rejects_replacement_before_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repository(tmp_path)
    relative = "controls/input_inventory.json"
    control = repo / relative
    control.parent.mkdir()
    control.write_text('{"version":1}\n', encoding="utf-8")
    run_root = tmp_path / "runs" / "explicit-replaced"
    run_root.parent.mkdir()
    marker = tmp_path / "child-must-not-start"
    original_recheck = run_audited_command._recheck_explicit_prelaunch_files

    def replace_then_recheck(
        git_root: Path,
        captured_manifest: dict,
    ) -> dict:
        control.write_text('{"version":2}\n', encoding="utf-8")
        return original_recheck(git_root, captured_manifest)

    monkeypatch.setattr(
        run_audited_command,
        "_recheck_explicit_prelaunch_files",
        replace_then_recheck,
    )
    rc = run_audited_command.run_audited_command(
        run_root=run_root,
        project_root=repo,
        command=[
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ],
        workload_class=run_audited_command.NON_NEURAL,
        prelaunch_bind_files=[relative],
    )

    assert rc == run_audited_command.AUDIT_FAILURE_EXIT
    assert not marker.exists()
    manifest = json.loads(
        (run_root / "audit_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["stop_reason"] == "EXPLICIT_PRELAUNCH_FILE_RECHECK_FAILED"
    recheck = json.loads(
        (run_root / "git/explicit_prelaunch_recheck.json").read_text(encoding="utf-8")
    )
    assert recheck["matches"] is False


def test_wrapper_signal_forwards_only_to_exact_child_and_keeps_evidence(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path)
    approved = run_audited_command._capture_git_snapshot(repo)
    run_root = tmp_path / "runs" / "signal-forwarding"
    run_root.parent.mkdir()
    wrapper_script = Path(run_audited_command.__file__).resolve()
    sentinel = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    wrapper = subprocess.Popen(
        [
            sys.executable,
            str(wrapper_script),
            "--run-root",
            str(run_root),
            "--project-root",
            str(repo),
            "--workload-class",
            run_audited_command.NON_NEURAL,
            "--expected-git-head",
            approved["head"],
            "--expected-git-dirty-state-sha256",
            approved["dirty_state_sha256"],
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        process_path = run_root / "process.json"
        while not process_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert process_path.is_file()
        child_pid = json.loads(process_path.read_text(encoding="utf-8"))["child_pid"]

        wrapper.send_signal(signal.SIGTERM)
        stdout, stderr = wrapper.communicate(timeout=10)

        assert wrapper.returncode == 128 + signal.SIGTERM, (stdout, stderr)
        assert sentinel.poll() is None
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        completion = json.loads(
            (run_root / "completion.json").read_text(encoding="utf-8")
        )
        assert completion["state"] == "FAILED_WITH_EVIDENCE"
        assert completion["interrupted_by_signal"] == signal.SIGTERM
        assert completion["stop_reason"] == (f"INTERRUPTED_BY_SIGNAL_{signal.SIGTERM}")
        manifest = json.loads(
            (run_root / "audit_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["protection"]["unrelated_processes_terminated"] == 0
        assert (
            manifest["protection"]["only_exact_child_pid_may_receive_interrupt"] is True
        )
    finally:
        if wrapper.poll() is None:
            wrapper.terminate()
            wrapper.wait(timeout=10)
        if sentinel.poll() is None:
            sentinel.terminate()
            sentinel.wait(timeout=10)


def test_signal_during_prelaunch_snapshot_stops_before_child_with_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repository(tmp_path)
    run_root = tmp_path / "runs" / "signal-during-prelaunch"
    run_root.parent.mkdir()
    marker = tmp_path / "child-must-not-start"
    original_capture = run_audited_command._capture_git_snapshot

    def signal_then_capture(
        project_root: Path,
        *,
        prelaunch_bind_files: tuple[str, ...] = (),
    ) -> dict:
        os.kill(os.getpid(), signal.SIGTERM)
        return original_capture(
            project_root,
            prelaunch_bind_files=prelaunch_bind_files,
        )

    monkeypatch.setattr(
        run_audited_command,
        "_capture_git_snapshot",
        signal_then_capture,
    )
    rc = run_audited_command.run_audited_command(
        run_root=run_root,
        project_root=repo,
        command=[
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ],
        workload_class=run_audited_command.NON_NEURAL,
    )

    assert rc == 128 + signal.SIGTERM
    assert not marker.exists()
    completion = json.loads((run_root / "completion.json").read_text(encoding="utf-8"))
    assert completion["state"] == "FAILED_WITH_EVIDENCE"
    assert (
        completion["stop_reason"]
        == f"INTERRUPTED_BEFORE_COMMAND_SIGNAL_{signal.SIGTERM}"
    )
    assert completion["interrupted_by_signal"] == signal.SIGTERM


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


@pytest.mark.parametrize("forged_value", ("true", 1))
def test_neural_cuda_preflight_rejects_non_boolean_truthy_values(
    tmp_path: Path,
    forged_value: object,
) -> None:
    repo = _repository(tmp_path)
    run_root = tmp_path / "runs" / f"truthy-preflight-{type(forged_value).__name__}"
    run_root.parent.mkdir()
    marker = tmp_path / "must-not-exist"

    def forged_probe() -> dict:
        return {
            **{field: forged_value for field in run_audited_command.CUDA_HEALTH_FIELDS},
            "passed": True,
        }

    rc = run_audited_command.run_audited_command(
        run_root=run_root,
        project_root=repo,
        command=[
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ],
        workload_class=run_audited_command.NEURAL,
        cuda_probe=forged_probe,
    )

    assert rc == run_audited_command.CUDA_FAILURE_EXIT
    assert not marker.exists()
    manifest = json.loads(
        (run_root / "audit_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["stop_reason"] == "CUDA_PREFLIGHT_FAILED"
    assert manifest["cuda"]["preflight"]["passed"] is False


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


@pytest.mark.parametrize("forged_value", ("false", 1))
def test_neural_actual_health_rejects_non_boolean_truthy_values(
    tmp_path: Path,
    forged_value: object,
) -> None:
    repo = _repository(tmp_path)
    run_root = tmp_path / "runs" / f"truthy-actual-{type(forged_value).__name__}"
    run_root.parent.mkdir()
    forged = {field: forged_value for field in run_audited_command.CUDA_HEALTH_FIELDS}
    command = [
        sys.executable,
        "-c",
        (
            "import json, os; "
            "open(os.environ['EDITFLOW_CUDA_HEALTH_FILE'], 'x').write("
            f"json.dumps({forged!r}) + '\\n')"
        ),
    ]

    rc = run_audited_command.run_audited_command(
        run_root=run_root,
        project_root=repo,
        command=command,
        workload_class=run_audited_command.NEURAL,
        cuda_probe=_passed_cuda_health,
    )

    assert rc == run_audited_command.CUDA_FAILURE_EXIT
    manifest = json.loads(
        (run_root / "audit_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["stop_reason"] == "ACTUAL_COMMAND_CUDA_HEALTH_FAILED"
    assert manifest["cuda"]["actual_command_health"]["passed"] is False


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
