"""Adversarial local-fixture tests for the remote read-only preflight."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import scripts.ops.r3x_remote_readonly_preflight as preflight  # noqa: E402
from scripts.ops.r3x_remote_readonly_preflight import (  # noqa: E402
    CommandResult,
    PreflightConfig,
    PreflightError,
    canonical_json_bytes,
    collect_preflight,
)


HEAD = "a" * 40


class FixtureRunner:
    """Deterministic command fixture; it never starts a child process."""

    def __init__(
        self,
        repo_root: Path,
        *,
        tracked: bytes = b"scripts/model.py\x00docs/contract.md\x00",
        status: bytes | None = None,
        worktrees: bytes | None = None,
        ps: bytes | None = None,
        gpu_available: bool = True,
    ) -> None:
        self.repo_root = repo_root
        self.tracked = tracked
        self.status = status or (
            f"# branch.oid {HEAD}\n# branch.head main\n".encode("utf-8")
        )
        self.worktrees = worktrees or (
            f"worktree {repo_root}\nHEAD {HEAD}\nbranch refs/heads/main\n\n".encode(
                "utf-8"
            )
        )
        self.ps = ps or (
            b"1 0 root S 100 systemd\n"
            b"101 1 alice Sl 25 python3\n"
            b"202 1 bob S 7 torchrun\n"
            b"303 1 carol S 4 bash\n"
        )
        self.gpu_available = gpu_available
        self.calls: list[tuple[str, tuple[str, ...], Path]] = []

    def run(
        self,
        tool: str,
        arguments: Sequence[str],
        *,
        cwd: Path,
    ) -> CommandResult:
        args = tuple(arguments)
        self.calls.append((tool, args, cwd))
        if tool == "git":
            tail = args[args.index("-C") + 2 :]
            mapping = {
                ("rev-parse", "--show-toplevel"): (
                    f"{self.repo_root}\n".encode("utf-8")
                ),
                ("rev-parse", "--absolute-git-dir"): (
                    f"{self.repo_root / '.git'}\n".encode("utf-8")
                ),
                ("rev-parse", "--verify", "HEAD"): f"{HEAD}\n".encode("ascii"),
                ("symbolic-ref", "--short", "-q", "HEAD"): b"main\n",
                ("ls-files", "-z"): self.tracked,
                (
                    "status",
                    "--porcelain=v2",
                    "--branch",
                    "--untracked-files=no",
                ): self.status,
                ("worktree", "list", "--porcelain"): self.worktrees,
            }
            return CommandResult(0, mapping[tail])
        if tool == "ps":
            return CommandResult(0, self.ps)
        if tool == "nvidia-smi":
            if not self.gpu_available:
                return CommandResult(127, b"")
            if args[0].startswith("--query-gpu="):
                return CommandResult(
                    0,
                    b"0, GPU-fixture, NVIDIA A100, 81920, 1024, 80896, 4, 31\n",
                )
            return CommandResult(
                0,
                b"202, /usr/bin/python3, GPU-fixture, 1024\n",
            )
        raise AssertionError(f"unexpected fixture command: {tool} {args}")


@pytest.fixture
def safe_fixture(tmp_path: Path) -> dict[str, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    contract = tmp_path / "contract_v2.md"
    contract.write_bytes(b"contract-v2\n")
    ordinary = tmp_path / "ordinary_development_rows.jsonl"
    ordinary.write_bytes(b'{"row_id":"r1"}\n')
    missing = tmp_path / "ordinary_missing.jsonl"
    artifact = tmp_path / "r3x_report.json"
    artifact.write_bytes(b'{"status":"HOLD"}\n')
    artifact_directory = tmp_path / "artifact_root"
    artifact_directory.mkdir()
    # A protected-looking child proves that directory metadata is not walked.
    (artifact_directory / "sealed_child_should_not_be_seen.bin").write_bytes(b"x")
    return {
        "root": tmp_path,
        "repo": repo,
        "contract": contract,
        "ordinary": ordinary,
        "missing": missing,
        "artifact": artifact,
        "artifact_directory": artifact_directory,
    }


def _config(paths: dict[str, Path]) -> PreflightConfig:
    return PreflightConfig(
        repo_root=paths["repo"],
        read_roots=(paths["root"],),
        authority_contract_paths=(paths["contract"],),
        ordinary_data_paths=(paths["ordinary"], paths["missing"]),
        artifact_paths=(paths["artifact"], paths["artifact_directory"]),
    )


def test_fixture_collects_required_snapshot_as_canonical_stdout_payload(
    safe_fixture: dict[str, Path],
) -> None:
    runner = FixtureRunner(safe_fixture["repo"])
    times = iter(("2026-08-08T00:00:00Z", "2026-08-08T00:00:01Z"))
    payload = collect_preflight(
        _config(safe_fixture),
        runner=runner,
        clock=lambda: next(times),
        hostname="fixture-host",
    )

    assert payload["schema_version"] == preflight.SCHEMA_VERSION
    assert payload["status"] == preflight.COMPLETE_STATUS
    assert payload["hostname"] == "fixture-host"
    assert payload["started_utc"] == "2026-08-08T00:00:00Z"
    assert payload["finished_utc"] == "2026-08-08T00:00:01Z"
    assert payload["git"]["repo_root"] == str(safe_fixture["repo"])
    assert payload["git"]["head"] == HEAD
    assert payload["git"]["branch"] == "main"
    assert payload["git"]["status"]["clean_ignoring_untracked"] is True
    assert payload["git"]["status"]["untracked_files_scanned"] is False
    assert payload["git"]["tracked_path_count"] == 2
    assert payload["git"]["worktrees"] == [
        {
            "branch": "refs/heads/main",
            "head": HEAD,
            "path": str(safe_fixture["repo"]),
        }
    ]

    contract = payload["authority_contracts"][0]
    assert contract["exists"] is True
    assert contract["size_bytes"] == len(b"contract-v2\n")
    assert contract["sha256"] == hashlib.sha256(b"contract-v2\n").hexdigest()
    ordinary, missing = payload["ordinary_data"]
    assert ordinary["exists"] is True
    assert ordinary["size_bytes"] == len(b'{"row_id":"r1"}\n')
    assert ordinary["sha256"] == hashlib.sha256(
        b'{"row_id":"r1"}\n'
    ).hexdigest()
    assert missing == {
        "entry_type": "MISSING",
        "exists": False,
        "index": 1,
        "path": str(safe_fixture["missing"]),
        "sha256": None,
        "size_bytes": None,
    }
    assert payload["artifacts"][0]["sha256"] == hashlib.sha256(
        b'{"status":"HOLD"}\n'
    ).hexdigest()
    assert payload["artifacts"][1]["directory_traversed"] is False
    assert payload["artifacts"][1]["sha256"] is None
    assert payload["jobs"]["matched_count"] == 2
    assert [record["executable_basename"] for record in payload["jobs"]["records"]] == [
        "python3",
        "torchrun",
    ]
    assert all("arguments" not in record for record in payload["jobs"]["records"])
    assert payload["gpu"]["query_only"] is True
    assert payload["gpu"]["devices"][0]["memory_free_mib"] == 80896
    assert payload["disk"]
    assert payload["boundaries"] == preflight.BOUNDARIES

    raw = canonical_json_bytes(payload)
    assert raw.endswith(b"\n")
    assert b"\n" not in raw[:-1]
    assert json.loads(raw) == payload
    assert raw == (
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


@pytest.mark.parametrize(
    "relative",
    [
        "sealed/final.jsonl",
        "restricted/member.jsonl",
        "GSE246381_member.jsonl",
        "raw_outcome.bin",
        "internal-test-outcomes.jsonl",
        "ordinary_outcomes.jsonl",
        "outcome_authority.jsonl",
        "outcome-release/rows.jsonl",
        "test_labels.tsv",
        "final-labels.tsv",
        "protected_outcome_data/rows.jsonl",
        "ＳＥＡＬＥＤ/fullwidth-normalized.bin",
    ],
)
def test_protected_paths_fail_before_any_command_or_content_access(
    safe_fixture: dict[str, Path],
    relative: str,
) -> None:
    runner = FixtureRunner(safe_fixture["repo"])
    target = safe_fixture["root"] / relative
    with pytest.raises(PreflightError) as caught:
        collect_preflight(
            PreflightConfig(
                repo_root=safe_fixture["repo"],
                read_roots=(safe_fixture["root"],),
                ordinary_data_paths=(target,),
            ),
            runner=runner,
            hostname="fixture-host",
        )
    assert caught.value.code == "HOLD_PROTECTED_PATH_OR_REFERENCE"
    assert runner.calls == []
    assert str(target) not in caught.value.message
    assert str(target) not in json.dumps(caught.value.details)


def test_symlink_ancestor_and_read_root_escape_fail_before_commands(
    safe_fixture: dict[str, Path],
    tmp_path: Path,
) -> None:
    runner = FixtureRunner(safe_fixture["repo"])
    real_directory = safe_fixture["root"] / "real_directory"
    real_directory.mkdir()
    (real_directory / "rows.jsonl").write_bytes(b"row\n")
    alias = safe_fixture["root"] / "alias_directory"
    alias.symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(PreflightError) as symlink_caught:
        collect_preflight(
            PreflightConfig(
                repo_root=safe_fixture["repo"],
                read_roots=(safe_fixture["root"],),
                ordinary_data_paths=(alias / "rows.jsonl",),
            ),
            runner=runner,
        )
    assert symlink_caught.value.code == "HOLD_SYMLINK_PATH_REJECTED"
    assert runner.calls == []

    outside = tmp_path.parent / f"outside-{tmp_path.name}.jsonl"
    with pytest.raises(PreflightError) as escape_caught:
        collect_preflight(
            PreflightConfig(
                repo_root=safe_fixture["repo"],
                read_roots=(safe_fixture["repo"],),
                ordinary_data_paths=(outside,),
            ),
            runner=runner,
        )
    assert escape_caught.value.code == "HOLD_PATH_OUTSIDE_EXPLICIT_READ_ROOT"
    assert runner.calls == []


def test_hardlink_and_special_file_are_rejected(
    safe_fixture: dict[str, Path],
) -> None:
    original = safe_fixture["root"] / "ordinary_source.jsonl"
    alias = safe_fixture["root"] / "ordinary_alias.jsonl"
    original.write_bytes(b"ordinary\n")
    os.link(original, alias)
    runner = FixtureRunner(safe_fixture["repo"])
    with pytest.raises(PreflightError) as caught:
        collect_preflight(
            PreflightConfig(
                repo_root=safe_fixture["repo"],
                read_roots=(safe_fixture["root"],),
                ordinary_data_paths=(original,),
            ),
            runner=runner,
        )
    assert caught.value.code == "HOLD_HARDLINK_ALIAS_REJECTED"

    fifo = safe_fixture["root"] / "ordinary_pipe"
    os.mkfifo(fifo)
    with pytest.raises(PreflightError) as fifo_caught:
        collect_preflight(
            PreflightConfig(
                repo_root=safe_fixture["repo"],
                read_roots=(safe_fixture["root"],),
                ordinary_data_paths=(fifo,),
            ),
            runner=FixtureRunner(safe_fixture["repo"]),
        )
    assert fifo_caught.value.code == "HOLD_NONREGULAR_FILE_REJECTED"


def test_git_index_protected_path_stops_before_status_and_file_hashing(
    safe_fixture: dict[str, Path],
) -> None:
    runner = FixtureRunner(
        safe_fixture["repo"],
        tracked=b"scripts/model.py\x00data/GSE246381_member.jsonl\x00",
    )
    with pytest.raises(PreflightError) as caught:
        collect_preflight(_config(safe_fixture), runner=runner)
    assert caught.value.code == "HOLD_PROTECTED_PATH_OR_REFERENCE"
    git_tails = [
        args[args.index("-C") + 2 :]
        for tool, args, _cwd in runner.calls
        if tool == "git"
    ]
    assert (
        "status",
        "--porcelain=v2",
        "--branch",
        "--untracked-files=no",
    ) not in git_tails
    assert not any(tool in {"ps", "nvidia-smi"} for tool, _args, _cwd in runner.calls)


def test_protected_worktree_reference_is_not_emitted(
    safe_fixture: dict[str, Path],
) -> None:
    protected_worktree = (
        f"worktree {safe_fixture['root']}/sealed-evaluation\n"
        f"HEAD {HEAD}\nbranch refs/heads/main\n\n"
    ).encode("utf-8")
    runner = FixtureRunner(safe_fixture["repo"], worktrees=protected_worktree)
    with pytest.raises(PreflightError) as caught:
        collect_preflight(_config(safe_fixture), runner=runner)
    assert caught.value.code == "HOLD_PROTECTED_PATH_OR_REFERENCE"
    assert "sealed-evaluation" not in caught.value.message
    assert "sealed-evaluation" not in json.dumps(caught.value.details)


def _filesystem_snapshot(root: Path) -> list[tuple[str, str, int, int, str | None]]:
    records: list[tuple[str, str, int, int, str | None]] = []
    for path in sorted((root, *root.rglob("*"))):
        metadata = os.lstat(path)
        relative = "." if path == root else str(path.relative_to(root))
        if stat.S_ISREG(metadata.st_mode):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            kind = "file"
        elif stat.S_ISDIR(metadata.st_mode):
            digest = None
            kind = "directory"
        elif stat.S_ISLNK(metadata.st_mode):
            digest = os.readlink(path)
            kind = "symlink"
        else:
            digest = None
            kind = "other"
        records.append(
            (relative, kind, metadata.st_mode, metadata.st_mtime_ns, digest)
        )
    return records


def test_collection_performs_no_explicit_filesystem_mutation(
    safe_fixture: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _filesystem_snapshot(safe_fixture["root"])

    def forbidden(*_args, **_kwargs):
        raise AssertionError("collector attempted a mutating filesystem operation")

    for name in (
        "mkdir",
        "remove",
        "rename",
        "replace",
        "rmdir",
        "symlink",
        "truncate",
        "unlink",
    ):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, forbidden)
    runner = FixtureRunner(safe_fixture["repo"])
    collect_preflight(_config(safe_fixture), runner=runner, hostname="fixture")
    after = _filesystem_snapshot(safe_fixture["root"])
    assert after == before


def test_gpu_unavailable_is_metadata_not_a_fallback_workload(
    safe_fixture: dict[str, Path],
) -> None:
    runner = FixtureRunner(safe_fixture["repo"], gpu_available=False)
    payload = collect_preflight(_config(safe_fixture), runner=runner)
    assert payload["gpu"] == {
        "compute_processes": [],
        "devices": [],
        "query_only": True,
        "status": "NVIDIA_SMI_NOT_AVAILABLE",
    }
    gpu_calls = [call for call in runner.calls if call[0] == "nvidia-smi"]
    assert len(gpu_calls) == 1
    assert gpu_calls[0][1][0].startswith("--query-gpu=")


def test_exact_command_allowlist_rejects_mutating_or_expansive_forms(
    safe_fixture: dict[str, Path],
) -> None:
    prefix = preflight._git_args(safe_fixture["repo"])
    with pytest.raises(PreflightError) as gc_caught:
        preflight._validate_command_arguments(
            "git",
            (*prefix, "gc"),
            cwd=safe_fixture["repo"],
        )
    assert gc_caught.value.code == "HOLD_GIT_COMMAND_NOT_READ_ONLY"
    with pytest.raises(PreflightError) as untracked_caught:
        preflight._validate_command_arguments(
            "git",
            (*prefix, "status", "--untracked-files=all"),
            cwd=safe_fixture["repo"],
        )
    assert untracked_caught.value.code == "HOLD_GIT_COMMAND_NOT_READ_ONLY"
    with pytest.raises(PreflightError) as gpu_caught:
        preflight._validate_command_arguments(
            "nvidia-smi",
            ("--gpu-reset",),
            cwd=safe_fixture["repo"],
        )
    assert gpu_caught.value.code == "HOLD_GPU_COMMAND_NOT_QUERY_ONLY"


def test_runtime_audit_hook_blocks_writes_and_unknown_children() -> None:
    hook = preflight._make_read_only_audit_hook({"/usr/bin/git"})
    hook("open", ("/tmp/input", "r", os.O_RDONLY))
    hook("subprocess.Popen", ("/usr/bin/git", [], None, None))
    with pytest.raises(PermissionError, match="writable"):
        hook("open", ("/tmp/output", "w", os.O_WRONLY | os.O_CREAT))
    with pytest.raises(PermissionError, match="os.mkdir"):
        hook("os.mkdir", ("/tmp/new", 0o700, -1))
    with pytest.raises(PermissionError, match="non-allowlisted"):
        hook("subprocess.Popen", ("/usr/bin/python3", [], None, None))


def test_cli_v1_is_disabled_before_any_path_or_command_access(
    safe_fixture: dict[str, Path],
) -> None:
    script = REPO_ROOT / "scripts" / "ops" / "r3x_remote_readonly_preflight.py"
    protected = safe_fixture["repo"] / "sealed-member.jsonl"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(safe_fixture["repo"]),
            "--ordinary-data",
            str(protected),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 2
    assert completed.stderr == b""
    assert completed.stdout.endswith(b"\n")
    assert b"\n" not in completed.stdout[:-1]
    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == preflight.SCHEMA_VERSION
    assert payload["status"] == preflight.HOLD_STATUS
    assert payload["error"]["code"] == preflight.REMOTE_EXECUTION_HOLD_CODE
    assert payload["error"]["details"]["remote_run_approval"] is False
    assert (
        payload["error"]["details"]["sealed_zero_access_guarantee_established"]
        is False
    )
    assert str(protected) not in completed.stdout.decode("utf-8")
    assert completed.stdout == canonical_json_bytes(payload)

    missing_repo = safe_fixture["root"] / "ordinary_repo_that_does_not_exist"
    missing = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(missing_repo),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=10,
    )
    assert missing.returncode == 2
    assert missing.stderr == b""
    assert json.loads(missing.stdout)["error"]["code"] == (
        preflight.REMOTE_EXECUTION_HOLD_CODE
    )


def test_hash_bound_is_full_hash_or_hold_never_partial(
    safe_fixture: dict[str, Path],
) -> None:
    runner = FixtureRunner(safe_fixture["repo"])
    with pytest.raises(PreflightError) as caught:
        collect_preflight(
            PreflightConfig(
                repo_root=safe_fixture["repo"],
                read_roots=(safe_fixture["root"],),
                ordinary_data_paths=(safe_fixture["ordinary"],),
                max_hash_bytes=1,
            ),
            runner=runner,
        )
    assert caught.value.code == "HOLD_FILE_EXCEEDS_EXPLICIT_HASH_BOUND"


def test_argument_parser_failure_is_canonical_hold_payload() -> None:
    error = PreflightError(
        "HOLD_CLI_ARGUMENTS",
        "command-line arguments do not match the exact preflight interface",
    )
    payload = preflight._hold_payload(error)
    assert payload["status"] == preflight.HOLD_STATUS
    assert canonical_json_bytes(payload).count(b"\n") == 1
