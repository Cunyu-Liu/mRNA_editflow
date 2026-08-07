"""Build a hash-bound R0-R2 repair manifest without sealed/restricted inputs.

The builder hashes the tracked repair scope and explicitly supplied ordinary
artifacts.  It rejects any external input whose resolved path contains a
configured forbidden token *before opening that path*.  It never imports or
invokes the E0 sealed runner and never writes an access-intent/reservation log.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.m4_sparse.ordinary_io import (
    assert_ordinary_path,
    load_ordinary_jsonl,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_output(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _tracked_scope(repo_root: Path, patterns: Iterable[str]) -> list[str]:
    tracked = git_output(repo_root, "ls-files").splitlines()
    selected = sorted(
        rel for rel in tracked if any(fnmatch.fnmatch(rel, pattern) for pattern in patterns)
    )
    if not selected:
        raise ValueError("tracked repair scope is empty")
    missing_patterns = [
        pattern for pattern in patterns
        if not any(fnmatch.fnmatch(rel, pattern) for rel in tracked)
    ]
    # Some optional test patterns are intentionally future-facing.  Exact paths
    # and non-test globs, however, must resolve so a typo cannot silently weaken
    # lineage coverage.
    required_missing = [
        pattern for pattern in missing_patterns
        if "*" not in pattern or not pattern.startswith("tests/")
    ]
    if required_missing:
        raise ValueError(f"tracked scope patterns matched nothing: {required_missing}")
    return selected


def _reject_forbidden_path(path: Path, forbidden_tokens: Iterable[str]) -> None:
    normalized = str(path.resolve(strict=False)).lower()
    hit = [token for token in forbidden_tokens if token.lower() in normalized]
    if hit:
        raise ValueError(
            f"external input rejected before open; forbidden path token(s): {hit}"
        )


def build_manifest(
    *,
    repo_root: Path,
    scope_path: Path,
    run_id: str,
    ordinary_artifacts: Iterable[Path] = (),
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    scope_path = scope_path.resolve()
    scope = yaml.safe_load(scope_path.read_text(encoding="utf-8"))
    patterns = list(scope["tracked_scope_globs"])
    forbidden_tokens = list(scope["forbidden_input_tokens"])
    base_commit = str(scope["base_commit"])

    head = git_output(repo_root, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", base_commit, head],
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise ValueError(f"HEAD {head} is not descended from frozen base {base_commit}")

    tracked_files = _tracked_scope(repo_root, patterns)
    tracked_records = []
    for rel in tracked_files:
        path = repo_root / rel
        tracked_records.append(
            {"path": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )

    ordinary_records = []
    for raw_path in ordinary_artifacts:
        path = raw_path.expanduser().resolve(strict=False)
        _reject_forbidden_path(path, forbidden_tokens)
        if not path.is_file():
            raise FileNotFoundError(path)
        assert_ordinary_path(path, purpose="lineage ordinary artifact")
        row_validation = None
        if path.suffix.lower() == ".jsonl":
            row_validation = {
                "validator": "ordinary_io.load_ordinary_jsonl",
                "n_rows": len(load_ordinary_jsonl(path, require_delta=False)),
                "status": "PASSED",
            }
        ordinary_records.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "data_role": "ORDINARY_EXPLICIT_INPUT",
                "ordinary_boundary_validation": row_validation,
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_kind": "xeditflow_r0_r2_lineage_manifest",
        "repair_goal": scope["repair_goal"],
        "run_id": run_id,
        "base_commit": base_commit,
        "head_commit": head,
        "snapshot_kind": "WORKING_TREE_BYTES_AT_HEAD_CONTEXT",
        "git_status_porcelain": git_output(repo_root, "status", "--short"),
        "source_binding": scope["source_binding"],
        "repair_decision_log": scope["repair_decision_log"],
        "scientific_status": scope["target_state"],
        "tracked_files": tracked_records,
        "ordinary_external_inputs": ordinary_records,
        "sealed_safety": {
            "formal_evaluator_invoked": "UNKNOWN_NOT_ASSERTED",
            "access_intent_written": "UNKNOWN_NOT_ASSERTED",
            "reservation_written": "UNKNOWN_NOT_ASSERTED",
            "restricted_or_sealed_member_input_count": "UNKNOWN_NOT_ASSERTED",
            "enforcement": (
                "ordinary inputs passed path and JSONL row guards; runtime "
                "zero-access requires independent before/after access-log evidence"),
        },
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    return manifest


def write_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--scope",
        type=Path,
        default=Path("configs/repair/r0_r2_repair_scope.yaml"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ordinary-artifact", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scope_path = args.scope
    if not scope_path.is_absolute():
        scope_path = args.repo_root / scope_path
    manifest = build_manifest(
        repo_root=args.repo_root,
        scope_path=scope_path,
        run_id=args.run_id,
        ordinary_artifacts=args.ordinary_artifact,
    )
    write_manifest(manifest, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "tracked_file_count": len(manifest["tracked_files"]),
        "ordinary_external_input_count": len(manifest["ordinary_external_inputs"]),
        "sealed_safety": manifest["sealed_safety"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
