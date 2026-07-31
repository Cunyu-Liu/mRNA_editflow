#!/usr/bin/env python3
"""Create a complete hard-link D1 release view without rebuilding data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.execution.acceptance_semantics import validate_phase_acceptance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _hardlink_tree(source: Path, destination: Path) -> int:
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise NotADirectoryError(source)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")
    count = 0
    for current, directories, filenames in os.walk(source):
        relative = Path(current).relative_to(source)
        target_directory = destination / relative
        target_directory.mkdir(parents=True, exist_ok=False)
        for directory in directories:
            if (Path(current) / directory).is_symlink():
                raise ValueError(f"symlinked dataset directory is forbidden: {directory}")
        for name in filenames:
            original = Path(current) / name
            if original.is_symlink() or not original.is_file():
                raise ValueError(f"non-regular dataset artifact is forbidden: {original}")
            os.link(original, target_directory / name)
            count += 1
    return count


def materialize(
    *,
    historical_stage_root: Path,
    rebind_root: Path,
    release_acceptance_template: Path,
    output_root: Path,
    repository_acceptance_next: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite release root: {output_root}")
    if repository_acceptance_next.exists():
        raise FileExistsError(f"refusing to overwrite repository acceptance next: {repository_acceptance_next}")
    historical_stage_root = historical_stage_root.resolve(strict=True)
    rebind_root = rebind_root.resolve(strict=True)
    template = _load(release_acceptance_template.resolve(strict=True))
    rebind_build = _load(rebind_root / "build_manifest.json")
    original_datasets = historical_stage_root / "datasets"
    canonical_source = rebind_root / "canonical" / "records_with_labels.jsonl"
    candidate_source = rebind_root / "candidate_store" / "candidates.jsonl"
    if not canonical_source.is_file() or not candidate_source.is_file():
        raise FileNotFoundError("rebind global stores are missing")
    output_root.mkdir(parents=True)
    linked_files = _hardlink_tree(original_datasets, output_root / "datasets")
    (output_root / "canonical").mkdir()
    (output_root / "candidate_store").mkdir()
    os.link(canonical_source, output_root / "canonical" / "records_with_labels.jsonl")
    os.link(candidate_source, output_root / "candidate_store" / "candidates.jsonl")
    build = dict(rebind_build)
    stores = dict(build["global_stores"])
    label = _file_ref(output_root / "canonical" / "records_with_labels.jsonl")
    candidate = _file_ref(output_root / "candidate_store" / "candidates.jsonl")
    old_label = dict(stores["canonical_label_store"])
    old_candidate = dict(stores["sealed_label_free_candidate_store"])
    old_label.update(label)
    old_label["path"] = "canonical/records_with_labels.jsonl"
    old_candidate.update(candidate)
    old_candidate["path"] = "candidate_store/candidates.jsonl"
    stores["canonical_label_store"] = old_label
    stores["sealed_label_free_candidate_store"] = old_candidate
    build["global_stores"] = stores
    build["current_canonical_release_view"] = {
        "historical_stage_root": str(historical_stage_root),
        "rebind_root": str(rebind_root),
        "dataset_files_hardlinked": linked_files,
        "data_rebuilt": False,
        "scientific_result_claimed": False,
    }
    build_path = output_root / "build_manifest.json"
    _write_json_exclusive(build_path, build)
    acceptance = dict(template)
    acceptance["stage_d1_root"] = str(output_root.resolve())
    required = dict(acceptance["required_artifact_validation"])
    required["build_manifest"] = _file_ref(build_path)
    acceptance["required_artifact_validation"] = required
    global_validation = dict(acceptance["global_store_validation"])
    global_validation["label_store"] = str((output_root / "canonical" / "records_with_labels.jsonl").resolve())
    global_validation["candidate_store"] = str((output_root / "candidate_store" / "candidates.jsonl").resolve())
    checks = dict(global_validation["checks"])
    checks["complete_historical_dataset_view_hardlinked"] = True
    global_validation["checks"] = checks
    acceptance["global_store_validation"] = global_validation
    acceptance["note"] = str(acceptance["note"]) + "; complete_hardlink_release_view=true"
    errors = validate_phase_acceptance("D1", acceptance, require_pass=True)
    if errors:
        raise ValueError("materialized release acceptance is not a semantic PASS: " + "; ".join(errors))
    external_acceptance = output_root / "acceptance.json"
    _write_json_exclusive(external_acceptance, acceptance)
    repository_acceptance_next.parent.mkdir(parents=True, exist_ok=True)
    os.link(external_acceptance, repository_acceptance_next)
    manifest = {
        "artifact_type": "d1_current_canonical_release_view.v1",
        "status": "PASS",
        "scientific_result_claimed": False,
        "release_root": str(output_root.resolve()),
        "acceptance": _file_ref(external_acceptance),
        "build_manifest": _file_ref(build_path),
        "canonical": _file_ref(output_root / "canonical" / "records_with_labels.jsonl"),
        "candidate_store": _file_ref(output_root / "candidate_store" / "candidates.jsonl"),
        "dataset_files_hardlinked": linked_files,
        "data_rebuilt": False,
    }
    _write_json_exclusive(output_root / "release_view_manifest.json", manifest)
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-stage-root", type=Path, required=True)
    parser.add_argument("--rebind-root", type=Path, required=True)
    parser.add_argument("--release-acceptance-template", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repository-acceptance-next", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = materialize(
            historical_stage_root=args.historical_stage_root,
            rebind_root=args.rebind_root,
            release_acceptance_template=args.release_acceptance_template,
            output_root=args.output_root,
            repository_acceptance_next=args.repository_acceptance_next,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAILED_WITH_EVIDENCE", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "PASS", "release_view_manifest": result["release_root"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
