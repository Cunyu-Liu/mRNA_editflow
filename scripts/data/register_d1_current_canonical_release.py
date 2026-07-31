#!/usr/bin/env python3
"""Register existing D1 evidence against a confirmed current canonical store."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.execution.acceptance_semantics import D1_REQUIRED_KEYS
from scripts.execution.acceptance_semantics import validate_phase_acceptance


STAGE_ID_RE = re.compile(r"^D1_B0_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{7}(?:_A[0-9]+)?$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _repo_child(repo_root: Path, path: Path) -> Path:
    candidate = path.resolve(strict=False)
    candidate.relative_to(repo_root.resolve(strict=True))
    return candidate


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _link_or_copy_exclusive(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        with source.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)


def _require_true(mapping: Mapping[str, Any], key: str, label: str) -> None:
    if mapping.get(key) is not True:
        raise ValueError(f"{label}.{key} must be true")


def _release_acceptance(
    *, rebind_root: Path, validation_report: Path, stage_id: str
) -> tuple[dict[str, Any], Path, str]:
    manifest_path = rebind_root / "rebind_manifest.json"
    source_acceptance_path = rebind_root / "acceptance.json"
    build_manifest_path = rebind_root / "build_manifest.json"
    manifest = _load_object(manifest_path)
    source = _load_object(source_acceptance_path)
    build = _load_object(build_manifest_path)
    validation = _load_object(validation_report)
    if manifest.get("artifact_type") != "d1_current_canonical_rebind.v1":
        raise ValueError("rebind artifact type is invalid")
    if manifest.get("status") != "PASS" or manifest.get("scientific_result_claimed") is not False:
        raise ValueError("rebind must be a passing non-scientific artifact")
    if validation.get("status") != "PASS":
        raise ValueError("B0 canonical validation report is not PASS")
    _require_true(validation, "d1_acceptance_bound", "B0 validation")
    binding = validation.get("d1_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("B0 validation has no D1 binding")
    _require_true(binding, "passed", "B0 D1 binding")
    canonical, structural = binding.get("canonical"), binding.get("structural")
    current, candidate = manifest.get("current_canonical"), manifest.get("candidate_store")
    if not all(isinstance(value, Mapping) for value in (canonical, structural, current, candidate)):
        raise ValueError("rebind/B0 binding lacks store metadata")
    if canonical.get("sha256") != current.get("sha256"):
        raise ValueError("B0 binding canonical SHA differs from rebind manifest")
    if structural.get("sha256") != candidate.get("sha256"):
        raise ValueError("B0 binding candidate SHA differs from rebind manifest")
    if not STAGE_ID_RE.fullmatch(stage_id):
        raise ValueError(f"invalid D1 release stage_id: {stage_id}")

    release = {key: source[key] for key in D1_REQUIRED_KEYS}
    release["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    required = dict(release["required_artifact_validation"])
    required["build_manifest"] = _file_ref(build_manifest_path)
    release["required_artifact_validation"] = required
    old_global = source.get("global_store_validation")
    if not isinstance(old_global, Mapping):
        raise ValueError("source acceptance global validation is missing")
    checks = dict(old_global.get("checks") or {})
    checks.update(
        {
            "current_canonical_rebind_manifest_passed": True,
            "current_canonical_schema_passed": True,
            "current_canonical_b0_loader_binding_passed": True,
            "current_canonical_projection_passed": True,
            "current_canonical_ambiguity_binding_passed": True,
        }
    )
    release["global_store_validation"] = {
        "passed": True,
        "label_store": str(canonical["path"]),
        "candidate_store": str(structural["path"]),
        "records": int(canonical["record_count"]),
        "candidate_label_leaks": [],
        "strict_validation_failures": [],
        "checks": checks,
    }
    release["note"] = (
        "CURRENT_CANONICAL_D1_RELEASE_REGISTRATION; "
        f"rebind_manifest_sha256={_sha256(manifest_path)}; "
        f"b0_validation_sha256={_sha256(validation_report)}; "
        "scientific_result_claimed=false"
    )
    errors = validate_phase_acceptance("D1", release, require_pass=True)
    if errors:
        raise ValueError("release acceptance is not a semantic PASS: " + "; ".join(errors))
    inventory = build.get("input_inventory")
    if not isinstance(inventory, Mapping):
        raise ValueError("rebind build manifest input inventory is missing")
    source_inventory = Path(str(inventory.get("path") or "")).resolve(strict=True)
    observed = _file_ref(source_inventory)
    if observed["bytes"] != inventory.get("bytes") or observed["sha256"] != inventory.get("sha256"):
        raise ValueError("rebind input inventory binding is stale")
    return release, source_inventory, str(validation["records_sha256"])


def register_release(
    *, repo_root: Path, rebind_root: Path, validation_report: Path, stage_id: str,
    acceptance_output: Path, input_inventory_output: Path, archive_snapshot_output: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    acceptance_output = _repo_child(repo_root, acceptance_output)
    input_inventory_output = _repo_child(repo_root, input_inventory_output)
    archive_snapshot_output = _repo_child(repo_root, archive_snapshot_output)
    prefix = repo_root / "artifacts" / "stages" / stage_id / "D1"
    if acceptance_output != prefix / "acceptance.json" or input_inventory_output != prefix / "input_inventory.json":
        raise ValueError("release outputs must use the canonical D1 stage path")
    active_snapshot = repo_root / "data/d1/manifests/d1_canonical_snapshot.json"
    if not active_snapshot.is_file():
        raise FileNotFoundError(f"active historical snapshot is missing: {active_snapshot}")
    if any(path.exists() for path in (acceptance_output, input_inventory_output, archive_snapshot_output)):
        raise FileExistsError("release registration outputs must all be new")
    release, inventory_source, canonical_sha = _release_acceptance(
        rebind_root=rebind_root.resolve(strict=True),
        validation_report=validation_report.resolve(strict=True),
        stage_id=stage_id,
    )
    _link_or_copy_exclusive(active_snapshot, archive_snapshot_output)
    _link_or_copy_exclusive(inventory_source, input_inventory_output)
    _write_json_exclusive(acceptance_output, release)
    return {
        "status": "PASS",
        "scientific_result_claimed": False,
        "acceptance": _file_ref(acceptance_output),
        "input_inventory": _file_ref(input_inventory_output),
        "archived_historical_snapshot": _file_ref(archive_snapshot_output),
        "current_canonical_sha256": canonical_sha,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--rebind-root", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--stage-id", required=True)
    parser.add_argument("--acceptance-output", type=Path, required=True)
    parser.add_argument("--input-inventory-output", type=Path, required=True)
    parser.add_argument("--archive-snapshot-output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = register_release(
            repo_root=args.repo_root, rebind_root=args.rebind_root,
            validation_report=args.validation_report, stage_id=args.stage_id,
            acceptance_output=args.acceptance_output,
            input_inventory_output=args.input_inventory_output,
            archive_snapshot_output=args.archive_snapshot_output,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAILED_WITH_EVIDENCE", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
