from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts.execution.build_code_manifest import (
    CodeManifestError,
    build_manifest,
    verify_manifest,
)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "keep.txt").write_text("base\n", encoding="utf-8")
    (repo / "delete.txt").write_text("remove\n", encoding="utf-8")
    _git(repo, "add", "keep.txt", "delete.txt")
    _git(repo, "commit", "-m", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_build_and_verify_exact_staged_boundary(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    (repo / "keep.txt").write_text("changed\n", encoding="utf-8")
    (repo / "new.bin").write_bytes(b"\x00\xffpayload")
    (repo / "delete.txt").unlink()
    _git(repo, "add", "keep.txt", "new.bin", "delete.txt")

    output = repo / "artifacts" / "code_manifest.json"
    manifest = build_manifest(repo, base, output)
    assert [item["path"] for item in manifest["files"]] == [
        "keep.txt",
        "new.bin",
    ]
    assert manifest["files"][1]["bytes"] == len(b"\x00\xffpayload")
    assert (
        manifest["files"][1]["sha256"] == hashlib.sha256(b"\x00\xffpayload").hexdigest()
    )
    assert manifest["deleted_paths"] == ["delete.txt"]

    _git(repo, "add", output.relative_to(repo).as_posix())
    assert verify_manifest(repo, output) == manifest


def test_build_refuses_overwrite_and_empty_boundary(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    output = repo / "manifest.json"
    with pytest.raises(CodeManifestError, match="empty"):
        build_manifest(repo, base, output)
    output.write_text("{}\n", encoding="utf-8")
    (repo / "keep.txt").write_text("changed\n", encoding="utf-8")
    _git(repo, "add", "keep.txt")
    with pytest.raises(CodeManifestError, match="overwrite"):
        build_manifest(repo, base, output)


def test_verify_rejects_unlisted_and_worktree_drift(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    (repo / "keep.txt").write_text("changed\n", encoding="utf-8")
    _git(repo, "add", "keep.txt")
    output = repo / "release" / "code_manifest.json"
    build_manifest(repo, base, output)
    _git(repo, "add", output.relative_to(repo).as_posix())

    (repo / "extra.txt").write_text("extra\n", encoding="utf-8")
    _git(repo, "add", "extra.txt")
    with pytest.raises(CodeManifestError, match="staged paths differ"):
        verify_manifest(repo, output)

    _git(repo, "reset", "extra.txt")
    (repo / "keep.txt").write_text("unstaged drift\n", encoding="utf-8")
    with pytest.raises(CodeManifestError, match="worktree content differs"):
        verify_manifest(repo, output)


def test_verify_rejects_tampered_reference(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    (repo / "keep.txt").write_text("changed\n", encoding="utf-8")
    _git(repo, "add", "keep.txt")
    output = repo / "release" / "code_manifest.json"
    manifest = build_manifest(repo, base, output)
    manifest["files"][0]["sha256"] = "0" * 64
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _git(repo, "add", output.relative_to(repo).as_posix())
    with pytest.raises(CodeManifestError, match="reference changed"):
        verify_manifest(repo, output)
