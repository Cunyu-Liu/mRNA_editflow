#!/usr/bin/env python3
"""Build an immutable D1 UTR EditBench snapshot from an explicit JSON config."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from data.utr_benchmark_v2.d1_artifacts import (
    REPRODUCTION_COLUMNS,
    build_required_artifact_payloads,
)
from data.utr_benchmark_v2.d1_builder import (
    D1_SCOPE_DATASETS,
    build_dataset_from_config,
    write_dataset_result,
)

EXPECTED_GOAL_CONTRACT_SHA256 = (
    "c3dc5875868d847b8519fee40b14c43b65e4c5948dc5c3b98101ca61a5671dd5"
)
EXPECTED_SCOPE_REPOSITORY_PATH = "data_registry/d1_dataset_scope_manifest.yaml"
EXPECTED_CONFIG_REPOSITORY_PATH = "configs/d1_build_20260729.json"


def _parse_config(source_bytes: bytes) -> dict[str, Any]:
    payload = json.loads(source_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("D1 build config must be a JSON object")
    if payload.get("schema_version") != "d1_build_config_v2":
        raise ValueError("D1 build config schema_version must be d1_build_config_v2")
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("D1 build config requires a non-empty datasets list")
    identifiers = [str(item.get("dataset_id", "")) for item in datasets]
    if any(not identifier for identifier in identifiers):
        raise ValueError("every D1 dataset config requires dataset_id")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("D1 build config contains duplicate dataset_id")
    if set(identifiers) != D1_SCOPE_DATASETS or len(identifiers) != len(
        D1_SCOPE_DATASETS
    ):
        raise ValueError(
            "D1 build config dataset scope must equal the frozen exact-12 scope"
        )
    selection = payload.get("selection_policy")
    if not isinstance(selection, dict):
        raise ValueError("D1 build config requires selection_policy")
    if (
        selection.get("candidate_final_labels_used_for_dataset_role_selection")
        is not False
    ):
        raise ValueError(
            "candidate final labels must not be used for dataset role selection"
        )
    goal_sha256 = str(selection.get("goal_contract_sha256", ""))
    if goal_sha256 != EXPECTED_GOAL_CONTRACT_SHA256:
        raise ValueError("D1 build config goal contract hash mismatch")
    scope_binding = selection.get("dataset_scope_manifest")
    if not isinstance(scope_binding, dict):
        raise ValueError(
            "selection_policy.dataset_scope_manifest must bind path/bytes/sha256"
        )
    scope_path = Path(str(scope_binding.get("path", "")))
    declared_bytes = scope_binding.get("bytes")
    declared_sha256 = str(scope_binding.get("sha256", ""))
    if not scope_path.is_absolute() or not scope_path.is_file():
        raise ValueError("frozen D1 scope manifest path must be absolute and exist")
    if (
        not isinstance(declared_bytes, int)
        or declared_bytes < 1
        or scope_path.stat().st_size != declared_bytes
    ):
        raise ValueError("frozen D1 scope manifest byte binding mismatch")
    if len(declared_sha256) != 64 or _sha256(scope_path) != declared_sha256:
        raise ValueError("frozen D1 scope manifest SHA-256 binding mismatch")
    scope = yaml.safe_load(scope_path.read_text(encoding="utf-8"))
    if not isinstance(scope, dict):
        raise ValueError("frozen D1 scope manifest must be a mapping")
    scope_rows = scope.get("datasets")
    if not isinstance(scope_rows, list):
        raise ValueError("frozen D1 scope manifest datasets must be a list")
    scope_ids = [str(row.get("dataset_id", "")) for row in scope_rows]
    if (
        len(scope_ids) != len(D1_SCOPE_DATASETS)
        or len(scope_ids) != len(set(scope_ids))
        or set(scope_ids) != D1_SCOPE_DATASETS
    ):
        raise ValueError("frozen D1 scope manifest is not the exact-12 scope")
    if scope.get("goal_contract_sha256") != goal_sha256:
        raise ValueError("scope manifest and config contract hashes differ")
    if scope.get("candidate_final_labels_used_for_role_selection") is not False:
        raise ValueError("scope manifest dataset-role selection is not label-free")
    if payload.get("stage_id") != scope.get("stage_id"):
        raise ValueError("config and scope manifest stage_id differ")
    scope_repository_path = scope_binding.get("repository_path")
    if (
        scope_repository_path is not None
        and scope_repository_path != EXPECTED_SCOPE_REPOSITORY_PATH
    ):
        raise ValueError("D1 scope manifest repository path mismatch")

    config_repository_path = payload.get("config_repository_path")
    if (
        config_repository_path is not None
        and config_repository_path != EXPECTED_CONFIG_REPOSITORY_PATH
    ):
        raise ValueError("D1 config repository path mismatch")

    inventory_binding = payload.get("input_inventory")
    if inventory_binding is not None:
        if not isinstance(inventory_binding, dict):
            raise ValueError("input_inventory must bind path/bytes/sha256")
        inventory_path = Path(str(inventory_binding.get("path", "")))
        inventory_bytes = inventory_binding.get("bytes")
        inventory_sha256 = str(inventory_binding.get("sha256", ""))
        expected_inventory_repository_path = (
            f"artifacts/stages/{payload.get('stage_id')}/D1/input_inventory.json"
        )
        if (
            inventory_binding.get("repository_path")
            != expected_inventory_repository_path
        ):
            raise ValueError("D1 input inventory repository path mismatch")
        if not inventory_path.is_absolute() or not inventory_path.is_file():
            raise ValueError(
                "frozen D1 input inventory path must be absolute and exist"
            )
        if (
            not isinstance(inventory_bytes, int)
            or inventory_bytes < 1
            or inventory_path.stat().st_size != inventory_bytes
        ):
            raise ValueError("frozen D1 input inventory byte binding mismatch")
        if (
            not re.fullmatch(r"[0-9a-f]{64}", inventory_sha256)
            or _sha256(inventory_path) != inventory_sha256
        ):
            raise ValueError("frozen D1 input inventory SHA-256 binding mismatch")
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        if (
            not isinstance(inventory, dict)
            or inventory.get("schema_version") != "d1_input_inventory.v1"
            or inventory.get("stage_id") != payload.get("stage_id")
            or inventory.get("selection_is_label_independent") is not True
        ):
            raise ValueError("frozen D1 input inventory semantics mismatch")
    return payload


def _load_config(path: Path) -> dict[str, Any]:
    return _parse_config(path.read_bytes())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_fresh(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite D1 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _install_exclusive(path: Path, temporary: Path) -> None:
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    _ensure_fresh(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        handle.write(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        )
    _install_exclusive(path, temporary)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_fresh(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    _install_exclusive(path, temporary)


def _write_required_artifacts(
    results: list[dict[str, Any]],
    artifact_root: Path,
) -> dict[str, dict[str, Any]]:
    """Write the five contract paths from deterministic frozen-store projections."""
    payloads = build_required_artifact_payloads(results)
    for relative, payload in payloads.items():
        path = artifact_root / relative
        if relative.endswith(".jsonl"):
            _write_jsonl(path, payload)
        elif relative.endswith(".json"):
            _write_json(path, payload)
        elif relative.endswith(".csv"):
            _ensure_fresh(path)
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            with temporary.open("x", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=REPRODUCTION_COLUMNS,
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(payload)
            _install_exclusive(path, temporary)
        else:  # pragma: no cover - fixed contract paths above
            raise AssertionError(f"unsupported D1 contract artifact: {relative}")
    return {
        relative: {
            "path": str((artifact_root / relative).resolve()),
            "bytes": (artifact_root / relative).stat().st_size,
            "sha256": _sha256(artifact_root / relative),
        }
        for relative in payloads
    }


def _write_global_stores(
    results: list[dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    blocked_records = sum(
        len(result["label_records"])
        for result in results
        if result["status"] == "blocked"
    )
    if blocked_records:
        raise ValueError("blocked D1 datasets cannot enter global stores")
    labels = sorted(
        [record for result in results for record in result["label_records"]],
        key=lambda record: record["record_id"],
    )
    candidates = sorted(
        [record for result in results for record in result["candidate_records"]],
        key=lambda record: record["record_id"],
    )
    label_ids = [record["record_id"] for record in labels]
    candidate_ids = [record["record_id"] for record in candidates]
    if len(label_ids) != len(set(label_ids)):
        raise ValueError("global canonical store has duplicate record_id")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("global candidate store has duplicate record_id")
    if label_ids != candidate_ids:
        raise ValueError(
            "global canonical/candidate stores are not a record_id bijection"
        )

    label_path = output_root / "canonical" / "records_with_labels.jsonl"
    candidate_path = output_root / "candidate_store" / "candidates.jsonl"
    _write_jsonl(label_path, labels)
    _write_jsonl(candidate_path, candidates)
    return {
        "canonical_label_store": {
            "path": label_path.relative_to(output_root).as_posix(),
            "bytes": label_path.stat().st_size,
            "sha256": _sha256(label_path),
            "records": len(labels),
            "record_ids_sha256": hashlib.sha256(
                ("\n".join(label_ids) + ("\n" if label_ids else "")).encode("utf-8")
            ).hexdigest(),
        },
        "sealed_label_free_candidate_store": {
            "path": candidate_path.relative_to(output_root).as_posix(),
            "bytes": candidate_path.stat().st_size,
            "sha256": _sha256(candidate_path),
            "records": len(candidates),
            "record_ids_sha256": hashlib.sha256(
                ("\n".join(candidate_ids) + ("\n" if candidate_ids else "")).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "label_bearing_fields": 0,
        },
        "record_id_bijection": True,
        "blocked_dataset_records": blocked_records,
    }


def build_snapshot(
    config_path: Path,
    output_root: Path,
    *,
    fixture_mode: bool = False,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    config_source_bytes = config_path.read_bytes()
    config = _parse_config(config_source_bytes)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty D1 output root: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_summaries: list[dict[str, Any]] = []
    built_results: list[dict[str, Any]] = []
    for dataset_config in config["datasets"]:
        result = build_dataset_from_config(
            dataset_config,
            fixture_mode=fixture_mode,
        )
        built_results.append(result)
        dataset_root = write_dataset_result(result, output_root)
        dataset_summaries.append(
            {
                "dataset_id": result["dataset_id"],
                "status": result["status"],
                "reason_code": result.get("reason_code"),
                "paper_eligible": result["paper_eligible"],
                "fixture_mode": result["fixture_mode"],
                "accounting": result["accounting"],
                "roundtrip_audit": result["roundtrip_audit"],
                "manifest": {
                    "path": (dataset_root / "manifest.json")
                    .relative_to(output_root)
                    .as_posix(),
                    "bytes": (dataset_root / "manifest.json").stat().st_size,
                    "sha256": _sha256(dataset_root / "manifest.json"),
                },
            }
        )

    required_supported = {
        "GSE114002",
        "GSE200304",
        "GSE246381",
        "GSE217518",
    }
    present = {summary["dataset_id"] for summary in dataset_summaries}
    missing_required = sorted(required_supported - present)
    missing_scope = sorted(D1_SCOPE_DATASETS - present)
    accepted_supported = sorted(
        summary["dataset_id"]
        for summary in dataset_summaries
        if summary["dataset_id"] in required_supported
        and summary["status"] in {"accepted", "accepted_fixture"}
    )
    global_stores = _write_global_stores(built_results, output_root)
    required_artifacts = _write_required_artifacts(
        built_results,
        artifact_root or output_root,
    )
    manifest = {
        "schema_version": "d1_build_snapshot_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixture_mode": fixture_mode,
        "config_path": str(config_path.resolve()),
        "config_bytes": len(config_source_bytes),
        "config_sha256": hashlib.sha256(config_source_bytes).hexdigest(),
        "config_repository_path": config.get("config_repository_path"),
        "goal_contract_sha256": config["selection_policy"]["goal_contract_sha256"],
        "dataset_scope_manifest": dict(
            config["selection_policy"]["dataset_scope_manifest"]
        ),
        "input_inventory": (
            dict(config["input_inventory"])
            if isinstance(config.get("input_inventory"), dict)
            else None
        ),
        "candidate_final_labels_used_for_dataset_role_selection": False,
        "datasets": dataset_summaries,
        "required_supported_datasets": sorted(required_supported),
        "accepted_supported_datasets": accepted_supported,
        "missing_required_supported_datasets": missing_required,
        "expected_d1_scope_datasets": sorted(D1_SCOPE_DATASETS),
        "missing_d1_scope_datasets": missing_scope,
        "required_artifacts": required_artifacts,
        "global_stores": global_stores,
        "structural_build_complete": (
            not missing_required
            and not missing_scope
            and set(accepted_supported) == required_supported
        ),
        "scientific_gate_claimed": False,
        "note": (
            "Build success is a D1 reconstruction fact, not a final scientific "
            "result. Fixture mode can never satisfy the D1 phase gate."
        ),
    }
    manifest_path = output_root / "build_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help=(
            "root under which exact contract paths data/... and reports/... "
            "are created; defaults to --output-root"
        ),
    )
    parser.add_argument(
        "--fixture-mode",
        action="store_true",
        help="mark every output as fixture-only and ineligible for the D1 gate",
    )
    args = parser.parse_args(argv)
    try:
        manifest = build_snapshot(
            args.config,
            args.output_root,
            fixture_mode=args.fixture_mode,
            artifact_root=args.artifact_root,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "FAILED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["structural_build_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
