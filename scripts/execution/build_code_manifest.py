#!/usr/bin/env python3
"""Build or verify the exact staged-file manifest for a D1/B0 code commit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = "d1_b0_code_manifest.v1"


class CodeManifestError(RuntimeError):
    """Raised when the staged release boundary is not exact and auditable."""


def _git(repo: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise CodeManifestError(
            f"git {' '.join(args)} failed with exit {completed.returncode}: {detail}"
        )
    return completed.stdout


def _safe_relative(repo: Path, path: Path) -> str:
    root = repo.resolve(strict=True)
    candidate = path if path.is_absolute() else root / path
    try:
        relative = candidate.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise CodeManifestError("output must remain inside the repository") from exc
    if not relative.parts or ".." in relative.parts:
        raise CodeManifestError("output path is not a safe repository-relative path")
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise CodeManifestError("symlinked output paths are forbidden")
    return relative.as_posix()


def _decode_paths(raw: bytes) -> list[str]:
    values: list[str] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        try:
            text = item.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise CodeManifestError("Git diff contains a non-UTF-8 path") from exc
        pure = PurePosixPath(text)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            raise CodeManifestError(f"Git returned an unsafe path: {text!r}")
        values.append(pure.as_posix())
    if len(values) != len(set(values)):
        raise CodeManifestError("Git diff returned duplicate paths")
    return sorted(values)


def _staged_paths(repo: Path, base_commit: str) -> tuple[list[str], list[str]]:
    present = _decode_paths(
        _git(
            repo,
            "diff",
            "--cached",
            "--name-only",
            "-z",
            "--no-renames",
            "--diff-filter=ACMRTUXB",
            base_commit,
            "--",
        )
    )
    deleted = _decode_paths(
        _git(
            repo,
            "diff",
            "--cached",
            "--name-only",
            "-z",
            "--no-renames",
            "--diff-filter=D",
            base_commit,
            "--",
        )
    )
    overlap = set(present) & set(deleted)
    if overlap:
        raise CodeManifestError(
            f"paths cannot be both present and deleted: {sorted(overlap)}"
        )
    return present, deleted


def _index_blob(repo: Path, relative: str) -> bytes:
    return _git(repo, "show", f":{relative}")


def _reference(repo: Path, relative: str) -> dict[str, Any]:
    payload = _index_blob(repo, relative)
    path = repo / relative
    if not path.is_file() or path.is_symlink():
        raise CodeManifestError(
            f"staged path is not a regular non-symlink file: {relative}"
        )
    try:
        live = path.read_bytes()
    except OSError as exc:
        raise CodeManifestError(f"cannot read staged path: {relative}") from exc
    if live != payload:
        raise CodeManifestError(
            f"worktree content differs from staged content: {relative}"
        )
    return {
        "path": relative,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _validate_base(repo: Path, base_commit: str) -> str:
    try:
        base = (
            _git(repo, "rev-parse", "--verify", f"{base_commit}^{{commit}}")
            .decode("ascii", errors="strict")
            .strip()
        )
    except UnicodeDecodeError as exc:
        raise CodeManifestError("base commit is not ASCII") from exc
    if len(base) != 40:
        raise CodeManifestError("base commit did not resolve to a full SHA")
    return base


def build_manifest(repo: Path, base_commit: str, output: Path) -> dict[str, Any]:
    repo = repo.resolve(strict=True)
    if _git(repo, "rev-parse", "--is-inside-work-tree").strip() != b"true":
        raise CodeManifestError("repo-root is not a Git worktree")
    base = _validate_base(repo, base_commit)
    output_relative = _safe_relative(repo, output)
    output_path = repo / output_relative
    if output_path.exists():
        raise CodeManifestError("refusing to overwrite an existing code manifest")
    present, deleted = _staged_paths(repo, base)
    if output_relative in present or output_relative in deleted:
        raise CodeManifestError(
            "output must not be staged until after the manifest is built"
        )
    if not present and not deleted:
        raise CodeManifestError("the staged code boundary is empty")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "base_commit_sha": base,
        "files": [_reference(repo, relative) for relative in present],
        "deleted_paths": deleted,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(output_path, flags, 0o644)
    try:
        payload = (
            json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n"
        ).encode("utf-8")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise CodeManifestError("failed to write the complete code manifest")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return manifest


def verify_manifest(repo: Path, output: Path) -> dict[str, Any]:
    repo = repo.resolve(strict=True)
    output_relative = _safe_relative(repo, output)
    output_path = repo / output_relative
    try:
        manifest = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CodeManifestError("code manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "base_commit_sha",
        "files",
        "deleted_paths",
    }:
        raise CodeManifestError("code manifest keys are not sealed")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise CodeManifestError("code manifest schema version is invalid")
    base = _validate_base(repo, str(manifest["base_commit_sha"]))
    present, deleted = _staged_paths(repo, base)
    references = manifest.get("files")
    if not isinstance(references, list) or not references:
        raise CodeManifestError("code manifest files must be a non-empty list")
    declared: list[str] = []
    for reference in references:
        if not isinstance(reference, dict) or set(reference) != {
            "path",
            "bytes",
            "sha256",
        }:
            raise CodeManifestError("code manifest contains an invalid file reference")
        relative = str(reference["path"])
        if _reference(repo, relative) != reference:
            raise CodeManifestError(f"staged blob reference changed: {relative}")
        declared.append(relative)
    if len(declared) != len(set(declared)):
        raise CodeManifestError("code manifest contains duplicate file paths")
    if manifest.get("deleted_paths") != deleted:
        raise CodeManifestError("deleted path inventory differs from the index")
    expected_present = sorted(set(declared) | {output_relative})
    if present != expected_present:
        raise CodeManifestError(
            "staged paths differ from the manifest plus the manifest itself"
        )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--base-commit", help="required in build mode")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.verify:
            manifest = verify_manifest(args.repo_root, args.output)
            action = "verified"
        else:
            if not args.base_commit:
                raise CodeManifestError("--base-commit is required in build mode")
            manifest = build_manifest(
                args.repo_root,
                args.base_commit,
                args.output,
            )
            action = "built"
    except CodeManifestError as exc:
        print(json.dumps({"result": "FAIL", "error": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "result": "PASS",
                "action": action,
                "file_count": len(manifest["files"]),
                "deleted_count": len(manifest["deleted_paths"]),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
