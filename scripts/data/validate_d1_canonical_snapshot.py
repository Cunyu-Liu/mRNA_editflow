#!/usr/bin/env python3
"""Recompute and validate a D1 canonical structural-data snapshot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import jsonschema

from scripts.data.build_d1_canonical_snapshot import build_snapshot_payload


SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "schemas/d1_canonical_snapshot.schema.json"
)


def validate_snapshot(
    snapshot_path: Path,
    *,
    repo_root: Path,
) -> list[str]:
    errors: list[str] = []
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [f"snapshot_parse_failure:{type(exc).__name__}:{exc}"]
    if not isinstance(snapshot, dict):
        return ["snapshot must be a JSON object"]
    try:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        validator = jsonschema.Draft202012Validator(
            schema,
            format_checker=jsonschema.FormatChecker(),
        )
        for error in sorted(
            validator.iter_errors(snapshot),
            key=lambda item: tuple(str(part) for part in item.path),
        ):
            location = ".".join(str(part) for part in error.path) or "<root>"
            errors.append(f"schema:{location}:{error.message}")
    except Exception as exc:
        errors.append(f"snapshot_schema_unavailable:{type(exc).__name__}:{exc}")
    if errors:
        return errors

    acceptance = snapshot["acceptance"]
    acceptance_path = repo_root / str(acceptance["path"])
    try:
        recomputed = build_snapshot_payload(
            acceptance_path=acceptance_path,
            repo_root=repo_root,
            code_commit=str(snapshot["code_provenance"]["code_commit_sha"]),
            generated_at_utc=str(snapshot["generated_at_utc"]),
        )
    except Exception as exc:
        return [f"snapshot_recompute_failure:{type(exc).__name__}:{exc}"]
    if snapshot != recomputed:
        errors.append("snapshot_differs_from_exact_live_recomputation")
    return errors


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    errors = validate_snapshot(args.snapshot, repo_root=args.repo_root)
    payload: dict[str, Any] = {
        "snapshot": str(args.snapshot),
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "scientific_result_claimed": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
