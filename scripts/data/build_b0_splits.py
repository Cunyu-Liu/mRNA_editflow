#!/usr/bin/env python3
"""Validate frozen D1 evidence and build one deterministic B0 split manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.utr_benchmark_v2.d1_builder import CANDIDATE_STORE_FIELDS
from data.utr_benchmark_v2.d1_builder import candidate_store_label_paths
from data.utr_benchmark_v2.path_states import MINIMUM_ALIGNMENT_COUNT_SCOPE
from data.utr_benchmark_v2.path_states import minimum_alignment_statistics
from data.utr_benchmark_v2.split_graph import REGIONS
from data.utr_benchmark_v2.split_graph import SPLIT_KINDS
from data.utr_benchmark_v2.split_graph import build_split_manifest
from data.utr_benchmark_v2.split_graph import label_free_structural_projection
from data.utr_benchmark_v2.split_graph import partition_sha256
from data.utr_benchmark_v2.split_graph import record_ids_sha256
from data.utr_benchmark_v2.split_graph import record_universe_sha256
from data.utr_benchmark_v2.track_loader import assert_candidate_store_label_free


SCHEMA_NAMES = (
    "utr_edit_record.schema.json",
    "edit_script.schema.json",
    "generation_task.schema.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {line_number} is invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}: line {line_number} must be an object")
            records.append(value)
    if not records:
        raise ValueError(f"{path}: no canonical records")
    return records


def load_structural_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load the dedicated D1 label-free store, rejecting nested escape hatches."""

    allowed_fields = set(CANDIDATE_STORE_FIELDS)
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {line_number} is invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}: line {line_number} must be an object")
            unknown = set(value) - allowed_fields
            if unknown:
                raise ValueError(
                    f"{path}: line {line_number} contains unsealed fields: "
                    + ", ".join(sorted(unknown))
                )
            assert_candidate_store_label_free([value])
            records.append(value)
    if not records:
        raise ValueError(f"{path}: no sealed structural records")
    return records


def _record_index(
    records: Sequence[Mapping[str, Any]],
    *,
    name: str,
) -> Dict[str, Mapping[str, Any]]:
    index: Dict[str, Mapping[str, Any]] = {}
    for record in records:
        record_id = str(record.get("record_id") or "").strip()
        if not record_id:
            raise ValueError(f"{name} record lacks record_id")
        if record_id in index:
            raise ValueError(f"{name} has duplicate record_id: {record_id}")
        index[record_id] = record
    return index


def _projection_comparison(
    canonical_records: Sequence[Mapping[str, Any]],
    structural_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Compare the actual D1 store to a fresh exact D1 label-free projection."""

    expected = [
        label_free_structural_projection(record) for record in canonical_records
    ]
    for record in expected:
        leaks = candidate_store_label_paths(record)
        if leaks:
            raise ValueError(
                "D1 projection unexpectedly contains label-bearing paths: "
                + ",".join(leaks)
            )
    expected_index = _record_index(expected, name="expected projection")
    actual_index = _record_index(structural_records, name="structural store")
    missing = sorted(set(expected_index) - set(actual_index))
    extra = sorted(set(actual_index) - set(expected_index))
    mismatched = sorted(
        record_id
        for record_id in set(expected_index) & set(actual_index)
        if expected_index[record_id] != actual_index[record_id]
    )
    expected_ordered = [
        expected_index[record_id] for record_id in sorted(expected_index)
    ]
    actual_ordered = [actual_index[record_id] for record_id in sorted(actual_index)]
    return {
        "passed": not missing and not extra and not mismatched,
        "expected_record_count": len(expected_ordered),
        "actual_record_count": len(actual_ordered),
        "expected_record_ids_sha256": record_ids_sha256(expected_ordered),
        "actual_record_ids_sha256": record_ids_sha256(actual_ordered),
        "expected_structural_content_sha256": record_universe_sha256(expected_ordered),
        "actual_structural_content_sha256": record_universe_sha256(actual_ordered),
        "missing_record_ids": missing[:100],
        "extra_record_ids": extra[:100],
        "mismatched_record_ids": mismatched[:100],
        "mismatched_record_count": len(mismatched),
    }


def _ambiguity_audit(
    canonical_records: Sequence[Mapping[str, Any]],
    ambiguity_report: Mapping[str, Any],
) -> Dict[str, Any]:
    by_dataset: Dict[str, List[Mapping[str, Any]]] = {}
    for record in canonical_records:
        if (
            record.get("source_sequence") is None
            or record.get("candidate_sequence") is None
            or record.get("edit_script") is None
        ):
            continue
        dataset_id = str(record.get("dataset_id") or "")
        by_dataset.setdefault(dataset_id, []).append(record)

    failures: List[Dict[str, Any]] = []
    report_datasets = ambiguity_report.get("datasets")
    if not isinstance(report_datasets, Mapping):
        report_datasets = {}
        failures.append({"kind": "ambiguity_datasets_missing"})
    for dataset_id, report_row in report_datasets.items():
        records = by_dataset.get(str(dataset_id), [])
        statistics = [
            minimum_alignment_statistics(
                str(record["source_sequence"]),
                str(record["candidate_sequence"]),
                known_minimum_edit_count=(
                    record.get("edit_distance")
                    if isinstance(record.get("edit_distance"), int)
                    and not isinstance(record.get("edit_distance"), bool)
                    else None
                ),
            )
            for record in records
        ]
        expected = {
            "records": len(statistics),
            "ambiguous_records": sum(
                summary.minimum_alignment_count > 1 for summary in statistics
            ),
            "max_equivalent_minimal_script_count": max(
                (summary.minimum_alignment_count for summary in statistics),
                default=0,
            ),
            "constructed_paths_marked_observed": sum(
                record.get("trajectory_observed") is True for record in records
            ),
            "count_scopes": ([MINIMUM_ALIGNMENT_COUNT_SCOPE] if records else []),
        }
        if not isinstance(report_row, Mapping):
            failures.append(
                {
                    "kind": "ambiguity_dataset_row_invalid",
                    "dataset_id": dataset_id,
                }
            )
            continue
        for field, expected_value in expected.items():
            if report_row.get(field) != expected_value:
                failures.append(
                    {
                        "kind": "ambiguity_dataset_binding_mismatch",
                        "dataset_id": dataset_id,
                        "field": field,
                        "expected": expected_value,
                        "observed": report_row.get(field),
                    }
                )
    missing_report_datasets = sorted(
        set(by_dataset) - set(str(value) for value in report_datasets)
    )
    if missing_report_datasets:
        failures.append(
            {
                "kind": "ambiguity_report_missing_canonical_datasets",
                "datasets": missing_report_datasets,
            }
        )
    if ambiguity_report.get("count_scope") != [MINIMUM_ALIGNMENT_COUNT_SCOPE]:
        failures.append({"kind": "ambiguity_count_scope_mismatch"})
    if ambiguity_report.get("constructed_paths_marked_observed") != 0:
        failures.append({"kind": "constructed_paths_marked_observed_nonzero"})
    return {
        "passed": not failures,
        "count_scope": MINIMUM_ALIGNMENT_COUNT_SCOPE,
        "canonical_intervention_record_count": sum(
            len(records) for records in by_dataset.values()
        ),
        "failures": failures[:100],
    }


def _load_d1_acceptance_binding(
    canonical_records_path: Path,
    d1_acceptance_path: Path,
) -> Dict[str, Any]:
    acceptance = json.loads(d1_acceptance_path.read_text(encoding="utf-8"))
    checks: Dict[str, bool] = {
        "acceptance_object": isinstance(acceptance, dict),
    }
    if not isinstance(acceptance, dict):
        return {
            "passed": False,
            "checks": checks,
            "failures": ["D1 acceptance is not an object"],
        }
    checks.update(
        {
            "phase_gate_passed": (acceptance.get("phase_gate_passed") is True),
            "not_fixture_mode": acceptance.get("fixture_mode") is False,
            "structural_validation_passed": (
                acceptance.get("structural_validation_passed") is True
            ),
            "global_store_validation_passed": (
                isinstance(acceptance.get("global_store_validation"), Mapping)
                and acceptance["global_store_validation"].get("passed") is True
            ),
            "required_artifacts_passed": (
                isinstance(
                    acceptance.get("required_artifact_validation"),
                    Mapping,
                )
                and acceptance["required_artifact_validation"].get("passed") is True
            ),
        }
    )
    stage_root = Path(str(acceptance.get("stage_d1_root") or ""))
    build_manifest_path = stage_root / "build_manifest.json"
    checks["stage_root_absolute"] = stage_root.is_absolute()
    checks["build_manifest_exists"] = build_manifest_path.is_file()
    if not build_manifest_path.is_file():
        return {
            "passed": False,
            "checks": checks,
            "failures": ["D1 build_manifest.json is missing"],
        }
    build_manifest = json.loads(build_manifest_path.read_text(encoding="utf-8"))
    stores = build_manifest.get("global_stores", {})
    canonical_meta = stores.get("canonical_label_store", {})
    structural_meta = stores.get("sealed_label_free_candidate_store", {})
    actual_canonical_path = stage_root / str(canonical_meta.get("path") or "")
    structural_path = stage_root / str(structural_meta.get("path") or "")
    global_validation = acceptance.get("global_store_validation", {})
    checks.update(
        {
            "canonical_path_matches_cli": (
                actual_canonical_path.resolve() == canonical_records_path.resolve()
            ),
            "canonical_path_matches_acceptance": (
                Path(str(global_validation.get("label_store") or "")).resolve()
                == actual_canonical_path.resolve()
            ),
            "structural_path_matches_acceptance": (
                Path(str(global_validation.get("candidate_store") or "")).resolve()
                == structural_path.resolve()
            ),
            "canonical_exists": actual_canonical_path.is_file(),
            "structural_exists": structural_path.is_file(),
        }
    )
    if not actual_canonical_path.is_file() or not structural_path.is_file():
        return {
            "passed": False,
            "checks": checks,
            "failures": ["D1 aggregate canonical/structural store is missing"],
        }
    canonical_records = load_jsonl(actual_canonical_path)
    structural_records = load_structural_jsonl(structural_path)
    canonical_id_hash = record_ids_sha256(canonical_records)
    structural_id_hash = record_ids_sha256(structural_records)
    checks.update(
        {
            "canonical_sha256_matches_manifest": (
                sha256_file(actual_canonical_path) == canonical_meta.get("sha256")
            ),
            "canonical_bytes_match_manifest": (
                actual_canonical_path.stat().st_size == canonical_meta.get("bytes")
            ),
            "canonical_count_matches_manifest": (
                len(canonical_records) == canonical_meta.get("records")
            ),
            "canonical_ids_match_manifest": (
                canonical_id_hash == canonical_meta.get("record_ids_sha256")
            ),
            "structural_sha256_matches_manifest": (
                sha256_file(structural_path) == structural_meta.get("sha256")
            ),
            "structural_bytes_match_manifest": (
                structural_path.stat().st_size == structural_meta.get("bytes")
            ),
            "structural_count_matches_manifest": (
                len(structural_records) == structural_meta.get("records")
            ),
            "structural_ids_match_manifest": (
                structural_id_hash == structural_meta.get("record_ids_sha256")
            ),
            "canonical_structural_id_bijection": (
                canonical_id_hash == structural_id_hash
                and len(canonical_records) == len(structural_records)
            ),
        }
    )
    projection = _projection_comparison(canonical_records, structural_records)
    checks["fresh_d1_projection_matches_structural_store"] = projection["passed"]

    artifact_validation = acceptance.get("required_artifact_validation", {})
    artifact_map = artifact_validation.get("artifacts", {})
    ambiguity_meta = artifact_map.get("data/edit_script_ambiguity_report.json", {})
    ambiguity_path = Path(str(ambiguity_meta.get("path") or ""))
    checks["ambiguity_report_exists"] = ambiguity_path.is_file()
    checks[
        "ambiguity_report_sha256_matches_acceptance"
    ] = ambiguity_path.is_file() and sha256_file(ambiguity_path) == ambiguity_meta.get(
        "sha256"
    )
    ambiguity_report: Dict[str, Any] = {}
    ambiguity_audit = {
        "passed": False,
        "failures": ["ambiguity report unavailable"],
    }
    if ambiguity_path.is_file():
        ambiguity_report = json.loads(ambiguity_path.read_text(encoding="utf-8"))
        ambiguity_audit = _ambiguity_audit(canonical_records, ambiguity_report)
    checks["ambiguity_scope_and_counts_reproduced"] = ambiguity_audit["passed"]
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
        "d1_acceptance_path": str(d1_acceptance_path.resolve()),
        "d1_acceptance_sha256": sha256_file(d1_acceptance_path),
        "d1_phase_gate_passed": (acceptance.get("phase_gate_passed") is True),
        "build_manifest_path": str(build_manifest_path.resolve()),
        "build_manifest_sha256": sha256_file(build_manifest_path),
        "canonical": {
            "path": str(actual_canonical_path.resolve()),
            "sha256": sha256_file(actual_canonical_path),
            "bytes": actual_canonical_path.stat().st_size,
            "record_count": len(canonical_records),
            "record_ids_sha256": canonical_id_hash,
        },
        "structural": {
            "path": str(structural_path.resolve()),
            "sha256": sha256_file(structural_path),
            "bytes": structural_path.stat().st_size,
            "record_count": len(structural_records),
            "record_ids_sha256": structural_id_hash,
            "structural_content_sha256": record_universe_sha256(structural_records),
        },
        "projection_comparison": projection,
        "ambiguity": {
            "path": str(ambiguity_path.resolve()),
            "sha256": (
                sha256_file(ambiguity_path) if ambiguity_path.is_file() else None
            ),
            "count_scope": MINIMUM_ALIGNMENT_COUNT_SCOPE,
            "audit": ambiguity_audit,
        },
    }


def validate_canonical_records_schema(
    records_path: Path,
    schema_dir: Path,
    *,
    d1_acceptance_path: Path | None = None,
) -> Dict[str, Any]:
    """Validate canonical records; optionally bind them to a passing D1 gate."""

    try:
        from jsonschema import Draft202012Validator, RefResolver
    except ImportError as exc:
        raise RuntimeError(
            "jsonschema>=4.18 is required for B0 production validation"
        ) from exc

    schemas = {
        name: json.loads((schema_dir / name).read_text(encoding="utf-8"))
        for name in SCHEMA_NAMES
    }
    store = {schema["$id"]: schema for schema in schemas.values()}
    canonical_schema = schemas["utr_edit_record.schema.json"]
    Draft202012Validator.check_schema(canonical_schema)
    resolver = RefResolver.from_schema(canonical_schema, store=store)
    validator = Draft202012Validator(canonical_schema, resolver=resolver)

    record_count = 0
    invalid_count = 0
    errors: List[Dict[str, Any]] = []
    record_ids: List[str] = []
    seen_record_ids = set()
    with records_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record_count += 1
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid_count += 1
                if len(errors) < 100:
                    errors.append(
                        {
                            "line": line_number,
                            "path": "",
                            "message": f"invalid JSON: {exc}",
                        }
                    )
                continue
            record_errors = sorted(
                validator.iter_errors(value),
                key=lambda error: tuple(str(part) for part in error.path),
            )
            record_invalid = bool(record_errors)
            for error in record_errors:
                if len(errors) >= 100:
                    break
                errors.append(
                    {
                        "line": line_number,
                        "path": ".".join(str(part) for part in error.path),
                        "message": error.message,
                    }
                )
            if isinstance(value, dict):
                record_id = str(value.get("record_id") or "").strip()
                if record_id:
                    record_ids.append(record_id)
                    if record_id in seen_record_ids:
                        record_invalid = True
                        if len(errors) < 100:
                            errors.append(
                                {
                                    "line": line_number,
                                    "path": "record_id",
                                    "message": ("duplicate canonical record_id"),
                                }
                            )
                    seen_record_ids.add(record_id)
            if record_invalid:
                invalid_count += 1
    if record_count == 0:
        invalid_count = 1
        errors.append({"line": 0, "path": "", "message": "empty canonical store"})

    d1_binding: Dict[str, Any] | None = None
    if d1_acceptance_path is not None:
        d1_binding = _load_d1_acceptance_binding(records_path, d1_acceptance_path)
    schema_passed = invalid_count == 0
    binding_passed = d1_binding is None or d1_binding["passed"]
    return {
        "schema_version": "utr_b0_canonical_schema_validation.v2",
        "status": "PASS" if schema_passed and binding_passed else "FAIL",
        "records_path": str(records_path.resolve()),
        "records_sha256": sha256_file(records_path),
        "record_count": record_count,
        "record_ids_sha256": hashlib.sha256(
            (("\n".join(sorted(record_ids)) + "\n") if record_ids else "").encode(
                "utf-8"
            )
        ).hexdigest(),
        "invalid_record_count": invalid_count,
        "schema_draft": "2020-12",
        "schema_sha256": {
            name: sha256_file(schema_dir / name) for name in SCHEMA_NAMES
        },
        "errors": errors,
        "d1_acceptance_bound": d1_binding is not None,
        "d1_binding": d1_binding,
        "legacy_schema_only_validation": d1_binding is None,
    }


def write_json_exclusive(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        handle.write("\n")


def _revalidate_split_inputs(
    structural_path: Path,
    validation_report: Mapping[str, Any],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if (
        validation_report.get("status") != "PASS"
        or validation_report.get("schema_draft") != "2020-12"
        or validation_report.get("invalid_record_count") != 0
        or validation_report.get("d1_acceptance_bound") is not True
    ):
        raise ValueError(
            "canonical validation report is absent, invalid, unbound, or not PASS"
        )
    binding = validation_report.get("d1_binding")
    if not isinstance(binding, Mapping) or binding.get("passed") is not True:
        raise ValueError("D1 acceptance binding is absent or failed")
    d1_acceptance_path = Path(str(binding.get("d1_acceptance_path") or ""))
    if not d1_acceptance_path.is_file() or sha256_file(
        d1_acceptance_path
    ) != binding.get("d1_acceptance_sha256"):
        raise ValueError("D1 acceptance artifact changed after canonical validation")
    current_acceptance = json.loads(d1_acceptance_path.read_text(encoding="utf-8"))
    if (
        current_acceptance.get("phase_gate_passed") is not True
        or current_acceptance.get("fixture_mode") is not False
        or current_acceptance.get("structural_validation_passed") is not True
    ):
        raise ValueError("D1 acceptance gate is no longer valid")
    build_manifest_path = Path(str(binding.get("build_manifest_path") or ""))
    if not build_manifest_path.is_file() or sha256_file(
        build_manifest_path
    ) != binding.get("build_manifest_sha256"):
        raise ValueError("D1 build manifest changed after canonical validation")
    canonical = binding.get("canonical")
    structural = binding.get("structural")
    ambiguity = binding.get("ambiguity")
    if not all(
        isinstance(value, Mapping) for value in (canonical, structural, ambiguity)
    ):
        raise ValueError("D1 aggregate bindings are incomplete")
    canonical_path = Path(str(canonical["path"]))
    expected_structural_path = Path(str(structural["path"]))
    if structural_path.resolve() != expected_structural_path.resolve():
        raise ValueError("split input must be the exact D1 sealed structural store")
    if sha256_file(canonical_path) != canonical.get("sha256"):
        raise ValueError("canonical store changed after D1-bound validation")
    if sha256_file(structural_path) != structural.get("sha256"):
        raise ValueError("structural store changed after D1-bound validation")
    ambiguity_path = Path(str(ambiguity["path"]))
    if (
        sha256_file(ambiguity_path) != ambiguity.get("sha256")
        or ambiguity.get("count_scope") != MINIMUM_ALIGNMENT_COUNT_SCOPE
    ):
        raise ValueError("D1 ambiguity evidence changed or has the wrong scope")
    canonical_records = load_jsonl(canonical_path)
    structural_records = load_structural_jsonl(structural_path)
    projection = _projection_comparison(canonical_records, structural_records)
    if not projection["passed"]:
        raise ValueError("fresh D1 label-free projection differs from structural store")
    if (
        len(canonical_records) != canonical.get("record_count")
        or record_ids_sha256(canonical_records) != canonical.get("record_ids_sha256")
        or len(structural_records) != structural.get("record_count")
        or record_ids_sha256(structural_records) != structural.get("record_ids_sha256")
        or record_universe_sha256(structural_records)
        != structural.get("structural_content_sha256")
    ):
        raise ValueError(
            "aggregate canonical/structural count, ID, or content binding changed"
        )
    return structural_records, {
        "d1_acceptance_path": binding["d1_acceptance_path"],
        "d1_acceptance_sha256": binding["d1_acceptance_sha256"],
        "d1_phase_gate_passed": True,
        "canonical_records_path": str(canonical_path.resolve()),
        "canonical_records_sha256": canonical["sha256"],
        "canonical_record_count": canonical["record_count"],
        "canonical_record_ids_sha256": canonical["record_ids_sha256"],
        "structural_records_path": str(structural_path.resolve()),
        "structural_records_sha256": structural["sha256"],
        "structural_records_bytes": structural["bytes"],
        "structural_record_count": structural["record_count"],
        "structural_record_ids_sha256": structural["record_ids_sha256"],
        "structural_content_sha256": structural["structural_content_sha256"],
        "d1_ambiguity_report_path": str(ambiguity_path.resolve()),
        "d1_ambiguity_report_sha256": ambiguity["sha256"],
        "ambiguity_count_scope": MINIMUM_ALIGNMENT_COUNT_SCOPE,
        "fresh_projection_comparison": projection,
    }


def _bind_manifest(
    manifest: Dict[str, Any],
    binding: Mapping[str, Any],
) -> None:
    manifest.update(binding)
    partitions = manifest.get("partitions")
    if not isinstance(partitions, list):
        raise ValueError("split manifest did not emit partitions[]")
    for partition in partitions:
        if not isinstance(partition, dict):
            raise ValueError("split partition is not an object")
        partition.update(binding)
        partition["partition_sha256"] = partition_sha256(partition)
    if "folds" in manifest:
        manifest["folds"] = partitions
        manifest["folds_sha256"] = _stable_sha256(
            [
                {
                    "fold_id": partition.get("fold_id"),
                    "status": partition.get("status"),
                    "partition_sha256": partition.get("partition_sha256"),
                    "blocked_reasons": partition.get("blocked_reasons", []),
                }
                for partition in partitions
            ]
        )
    if "strata" in manifest:
        manifest["strata"] = partitions
        manifest["strata_sha256"] = _stable_sha256(
            [
                {
                    "stratum_id": partition.get("stratum_id"),
                    "status": partition.get("status"),
                    "partition_sha256": partition.get("partition_sha256"),
                    "blocked_reasons": partition.get("blocked_reasons", []),
                }
                for partition in partitions
            ]
        )
    manifest["partitions_sha256"] = _stable_sha256(
        [
            {
                "partition_id": partition["partition_id"],
                "partition_sha256": partition["partition_sha256"],
            }
            for partition in partitions
        ]
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--split-kind", choices=SPLIT_KINDS)
    parser.add_argument("--region", choices=REGIONS)
    parser.add_argument("--source-region", choices=REGIONS, default="five_utr")
    parser.add_argument("--target-region", choices=REGIONS, default="three_utr")
    parser.add_argument("--validate-canonical-only", action="store_true")
    parser.add_argument(
        "--schema-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "schemas",
    )
    parser.add_argument("--d1-acceptance", type=Path)
    parser.add_argument("--canonical-validation-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.validate_canonical_only:
        if args.split_kind is not None or args.region is not None:
            parser.error("split arguments are forbidden in canonical-validation mode")
        if args.d1_acceptance is None:
            parser.error(
                "--d1-acceptance is required for production canonical validation"
            )
        report = validate_canonical_records_schema(
            args.records,
            args.schema_dir,
            d1_acceptance_path=args.d1_acceptance,
        )
        write_json_exclusive(args.output, report)
        return 0 if report["status"] == "PASS" else 2
    if args.split_kind is None:
        parser.error("--split-kind is required in split-build mode")
    if args.canonical_validation_report is None:
        parser.error("--canonical-validation-report is required in split-build mode")
    if args.split_kind != "cross_region_transfer" and args.region is None:
        parser.error("--region is required for source/study-disjoint manifests")
    if args.split_kind == "cross_region_transfer" and args.region is not None:
        parser.error("--region must be omitted for cross-region transfer")

    validation_report = json.loads(
        args.canonical_validation_report.read_text(encoding="utf-8")
    )
    expected_schema_hashes = {
        name: sha256_file(args.schema_dir / name) for name in SCHEMA_NAMES
    }
    if validation_report.get("schema_sha256") != expected_schema_hashes:
        raise ValueError("canonical validation report schema hashes are stale")
    records, binding = _revalidate_split_inputs(args.records, validation_report)
    binding["canonical_validation_report_path"] = str(
        args.canonical_validation_report.resolve()
    )
    binding["canonical_validation_report_sha256"] = sha256_file(
        args.canonical_validation_report
    )
    manifest = build_split_manifest(
        records,
        region=args.region,
        split_kind=args.split_kind,
        source_region=args.source_region,
        target_region=args.target_region,
    )
    _bind_manifest(manifest, binding)
    write_json_exclusive(args.output, manifest)
    return 0 if manifest["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
