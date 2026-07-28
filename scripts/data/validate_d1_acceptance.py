#!/usr/bin/env python3
"""Validate D1 artifact semantics without upgrading fixture evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from data.utr_benchmark_v2.d1_artifacts import (
    LIBRARY_REQUIRED_AUDITS,
    LIBRARY_REQUIRED_FIELDS,
    REPRODUCTION_COLUMNS,
    build_required_artifact_payloads,
)
from data.utr_benchmark_v2.d1_builder import (
    CANONICAL_FIELDS,
    CANDIDATE_STORE_FORBIDDEN_FIELDS,
    D1_SCOPE_DATASETS,
    _candidate_record,
    _row_fingerprint,
    _stable_id,
    candidate_store_label_paths,
    iter_table_rows,
)
from data.utr_benchmark_v2.edit_script import (
    apply_edit_script,
    canonicalize_edit_script,
)
from data.utr_benchmark_v2.records import (
    ABSOLUTE_PAIR_TYPES,
    validate_canonical_record,
)
from scripts.data.build_d1_utr_benchmark import (
    EXPECTED_CONFIG_REPOSITORY_PATH,
    EXPECTED_GOAL_CONTRACT_SHA256,
    EXPECTED_SCOPE_REPOSITORY_PATH,
    _parse_config,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON payload is not an object")
    return value


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolved_child(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError(f"unsafe relative evidence path: {relative!r}")
    root_resolved = root.resolve()
    path = (root / relative).resolve()
    path.relative_to(root_resolved)
    return path


def _artifact_binding_valid(
    root: Path,
    binding: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    try:
        path = _resolved_child(root, str(binding.get("path", "")))
    except Exception as exc:
        return False, {"error": f"{type(exc).__name__}:{exc}"}
    exists = path.is_file()
    actual = {
        "path": str(path),
        "exists": exists,
        "bytes": path.stat().st_size if exists else None,
        "sha256": _sha256(path) if exists else None,
    }
    passed = (
        exists
        and actual["bytes"] == binding.get("bytes")
        and actual["sha256"] == binding.get("sha256")
    )
    return passed, actual


def _git_blob_bytes(repo: Path, revision: str, relative: str) -> bytes | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{relative}"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def _prelaunch_file_binding(
    audit_root: Path,
    git_snapshot: Mapping[str, Any],
    repository_relative_path: str,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> dict[str, Any]:
    """Bind a file to the exact pre-launch Git snapshot.

    Clean tracked files are read from the captured HEAD.  Untracked files are
    bound by the wrapper's content manifest.  A tracked, dirty file without a
    pre-launch content hash fails closed instead of trusting the live worktree.
    """
    repository = Path(str(git_snapshot.get("repository", "")))
    head = str(git_snapshot.get("head", ""))
    explicit_binding = git_snapshot.get("artifacts", {}).get("explicit_prelaunch_files")
    explicit_manifest: dict[str, Any] = {}
    if isinstance(explicit_binding, Mapping):
        passed, actual = _artifact_binding_valid(audit_root, explicit_binding)
        if passed:
            explicit_manifest = _load_json(Path(actual["path"]))
    explicit_entries = {
        str(item.get("path")): item
        for item in explicit_manifest.get("entries", [])
        if isinstance(item, Mapping)
    }
    if repository_relative_path in explicit_entries:
        entry = explicit_entries[repository_relative_path]
        checks = {
            "captured_as_explicit_regular_file": (entry.get("kind") == "regular_file"),
            "bytes_match": entry.get("bytes") == expected_bytes,
            "sha256_match": entry.get("sha256") == expected_sha256,
        }
        return {
            "passed": all(checks.values()),
            "source": "explicit_prelaunch_file_manifest",
            "checks": checks,
        }
    untracked_binding = git_snapshot.get("artifacts", {}).get(
        "untracked_content_hashes"
    )
    untracked_manifest: dict[str, Any] = {}
    if isinstance(untracked_binding, Mapping):
        passed, actual = _artifact_binding_valid(audit_root, untracked_binding)
        if passed:
            untracked_manifest = _load_json(Path(actual["path"]))
    entries = {
        str(item.get("path")): item
        for item in untracked_manifest.get("entries", [])
        if isinstance(item, Mapping)
    }
    if repository_relative_path in entries:
        entry = entries[repository_relative_path]
        checks = {
            "captured_as_untracked_regular_file": (entry.get("kind") == "regular_file"),
            "bytes_match": entry.get("bytes") == expected_bytes,
            "sha256_match": entry.get("sha256") == expected_sha256,
        }
        return {
            "passed": all(checks.values()),
            "source": "prelaunch_untracked_content_manifest",
            "checks": checks,
        }

    blob = (
        _git_blob_bytes(repository, head, repository_relative_path)
        if repository.is_dir() and re.fullmatch(r"[0-9a-f]{40}", head)
        else None
    )
    checks = {
        "captured_head_blob_exists": blob is not None,
        "bytes_match": blob is not None and len(blob) == expected_bytes,
        "sha256_match": (
            blob is not None and hashlib.sha256(blob).hexdigest() == expected_sha256
        ),
    }
    if blob is not None and not all(checks.values()):
        diff_binding = git_snapshot.get("artifacts", {}).get("diff_head_binary")
        if isinstance(diff_binding, Mapping):
            diff_valid, diff_actual = _artifact_binding_valid(audit_root, diff_binding)
            if diff_valid:
                try:
                    with tempfile.TemporaryDirectory(
                        prefix="d1-prelaunch-reconstruct-"
                    ) as temporary:
                        temporary_root = Path(temporary)
                        target = _resolved_child(
                            temporary_root, repository_relative_path
                        )
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(blob)
                        applied = subprocess.run(
                            [
                                "git",
                                "apply",
                                "--whitespace=nowarn",
                                "--include",
                                repository_relative_path,
                                str(diff_actual["path"]),
                            ],
                            cwd=temporary_root,
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            check=False,
                        )
                        reconstructed = (
                            target.read_bytes()
                            if applied.returncode == 0 and target.is_file()
                            else None
                        )
                    reconstructed_checks = {
                        "captured_head_blob_exists": True,
                        "captured_diff_integrity_valid": True,
                        "diff_applied": applied.returncode == 0,
                        "bytes_match": (
                            reconstructed is not None
                            and len(reconstructed) == expected_bytes
                        ),
                        "sha256_match": (
                            reconstructed is not None
                            and hashlib.sha256(reconstructed).hexdigest()
                            == expected_sha256
                        ),
                    }
                    if all(reconstructed_checks.values()):
                        return {
                            "passed": True,
                            "source": "captured_head_plus_binary_diff",
                            "checks": reconstructed_checks,
                        }
                except Exception:
                    pass
    return {
        "passed": all(checks.values()),
        "source": "captured_head_blob",
        "checks": checks,
        "reason": (
            None
            if all(checks.values())
            else "TRACKED_DIRTY_FILE_LACKS_PRELAUNCH_CONTENT_HASH"
        ),
    }


def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: Any,
) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _record_missing_fields(record: dict[str, Any]) -> list[str]:
    return [field for field in CANONICAL_FIELDS if field not in record]


PAPER_CLEAN_REQUIRED_FIELDS = frozenset(
    {
        "row_id",
        "record_id",
        "dataset_id",
        "source_sequence",
        "candidate_sequence",
        "endpoint",
        "source_value_raw",
        "candidate_value_raw",
        "delta_raw",
        "label_reproduction_status",
        "pair_type",
    }
)
PAPER_CLEAN_OPTIONAL_FIELDS = frozenset(
    {
        "raw_source_sequence",
        "raw_candidate_sequence",
        "source_seqName",
        "candidate_seqName",
        "canonicalization_provenance",
    }
)


def _identity_binding(
    records: list[dict[str, Any]], identity_field: str
) -> dict[str, Any]:
    identities = [str(record.get(identity_field, "")) for record in records]
    return {
        "identity_field": identity_field,
        "records": len(records),
        "identities_unique": len(identities) == len(set(identities)),
        "identities_sha256": hashlib.sha256(
            ("\n".join(identities) + ("\n" if identities else "")).encode("utf-8")
        ).hexdigest(),
    }


def _paper_clean_projection(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": record["record_id"],
        "dataset_id": record["dataset_id"],
        "source_sequence": record["source_sequence"],
        "candidate_sequence": record["candidate_sequence"],
        "endpoint": record["endpoint"],
        "source_value_raw": record["source_value_raw"],
        "candidate_value_raw": record["candidate_value_raw"],
        "delta_raw": record["delta_raw"],
        "label_reproduction_status": record["label_provenance"]["status"],
        "pair_type": record["pair_type"],
    }


def _paper_clean_optional_projection(
    record: dict[str, Any],
) -> dict[str, Any]:
    projection: dict[str, Any] = {}
    for field in (
        "raw_source_sequence",
        "raw_candidate_sequence",
        "canonicalization_provenance",
    ):
        if record.get(field) not in (None, {}):
            projection[field] = record[field]
    processed = record.get("sequence_provenance", {}).get("processed_artifact", {})
    if isinstance(processed, Mapping):
        if "reference_seqName" in processed:
            projection["source_seqName"] = processed["reference_seqName"]
        if "mutant_seqName" in processed:
            projection["candidate_seqName"] = processed["mutant_seqName"]
    return projection


def _validate_paper_clean(
    paper_clean: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    auxiliary: list[dict[str, Any]],
) -> dict[str, Any]:
    schema_failures: dict[str, list[str]] = {}
    for index, record in enumerate(paper_clean):
        key = str(record.get("record_id", f"index:{index}"))
        failures: list[str] = []
        missing = sorted(PAPER_CLEAN_REQUIRED_FIELDS - set(record))
        extra = sorted(
            set(record) - PAPER_CLEAN_REQUIRED_FIELDS - PAPER_CLEAN_OPTIONAL_FIELDS
        )
        if missing:
            failures.append("missing:" + ",".join(missing))
        if extra:
            failures.append("extra:" + ",".join(extra))
        if not str(record.get("row_id", "")).strip():
            failures.append("empty_row_id")
        if not re.fullmatch(
            r"[A-Za-z0-9_.-]+:(?:record|absolute):[0-9a-f]{24}",
            str(record.get("record_id", "")),
        ):
            failures.append("invalid_record_id")
        candidate = record.get("candidate_sequence")
        source = record.get("source_sequence")
        if (
            not isinstance(candidate, str)
            or not candidate
            or set(candidate) - set("ACGU")
        ):
            failures.append("invalid_candidate_sequence")
        if source is not None and (
            not isinstance(source, str) or not source or set(source) - set("ACGU")
        ):
            failures.append("invalid_source_sequence")
        for field in ("source_value_raw", "candidate_value_raw", "delta_raw"):
            value = record.get(field)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                failures.append(f"nonfinite_or_non_numeric:{field}")
        if record.get("pair_type") in ABSOLUTE_PAIR_TYPES:
            if source is not None:
                failures.append("absolute_record_has_source")
        elif source is None:
            failures.append("intervention_record_missing_source")
        if failures:
            schema_failures[key] = failures

    paper_ids = [str(record.get("record_id", "")) for record in paper_clean]
    canonical_by_id = {str(record.get("record_id", "")): record for record in labels}
    content_mismatches: list[str] = []
    for paper in paper_clean:
        record_id = str(paper.get("record_id", ""))
        canonical = canonical_by_id.get(record_id)
        if canonical is None:
            content_mismatches.append(record_id)
            continue
        projection = _paper_clean_projection(canonical)
        optional_projection = _paper_clean_optional_projection(canonical)
        present_optional = set(paper) & PAPER_CLEAN_OPTIONAL_FIELDS
        expected_keys = set(projection)
        if (
            any(paper.get(key) != value for key, value in projection.items())
            or any(
                paper.get(key) != optional_projection.get(key)
                for key in present_optional
            )
            or set(paper) != expected_keys | {"row_id"} | present_optional
        ):
            content_mismatches.append(record_id)

    paper_row_ids = {str(record.get("row_id", "")) for record in paper_clean}
    rejected_row_ids = {str(record.get("row_id", "")) for record in rejected}
    auxiliary_row_ids = {str(record.get("row_id", "")) for record in auxiliary}
    overlaps = {
        "paper_clean_rejected": sorted(paper_row_ids & rejected_row_ids),
        "paper_clean_auxiliary": sorted(paper_row_ids & auxiliary_row_ids),
        "rejected_auxiliary": sorted(rejected_row_ids & auxiliary_row_ids),
    }
    checks = {
        "strict_content_schema": not schema_failures,
        "record_ids_unique": len(paper_ids) == len(set(paper_ids)),
        "record_id_bijection_with_canonical": (
            paper_ids == [str(record.get("record_id", "")) for record in labels]
        ),
        "canonical_projection_exact": not content_mismatches,
        "row_roles_disjoint": not any(overlaps.values()),
        "paper_clean_count_equals_canonical_count": (len(paper_clean) == len(labels)),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "schema_failures": schema_failures,
        "content_mismatches": content_mismatches,
        "row_role_overlaps": overlaps,
    }


def _recompute_gse200304_raw_fingerprints(
    manifest: dict[str, Any],
) -> tuple[dict[str, str], list[str]]:
    fingerprints: dict[str, str] = {}
    errors: list[str] = []
    for input_spec in (
        manifest.get("input_provenance", {})
        .get("provenance_audit", {})
        .get("raw_files", [])
    ):
        role = str(input_spec.get("role", ""))
        if role not in {"construct_table", "processed_label_table"}:
            continue
        path = Path(str(input_spec.get("path", "")))
        try:
            rows = iter_table_rows(
                path,
                file_format=input_spec.get("format"),
                delimiter=input_spec.get("delimiter"),
                sheet_name=input_spec.get("sheet_name"),
                defaults=input_spec.get("defaults"),
            )
            for index, row in enumerate(rows):
                fingerprints[f"{role}:{index}"] = _row_fingerprint(row)
        except Exception as exc:
            errors.append(f"{path}:{type(exc).__name__}:{exc}")
    return fingerprints, errors


def _validate_gse200304_lineage(
    lineage: list[dict[str, Any]],
    manifest: dict[str, Any],
    paper_clean: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    *,
    fixture_mode: bool,
) -> dict[str, Any]:
    if fixture_mode and not lineage:
        return {
            "passed": True,
            "status": "NOT_APPLICABLE",
            "reason_code": "FIXTURE_WITHOUT_EXACT_TWO_RAW_TABLE_JOIN",
            "checks": {},
        }
    required = {
        "schema_version",
        "dataset_id",
        "raw_table_role",
        "raw_row_index",
        "raw_row_key",
        "raw_row_id",
        "raw_row_fingerprint_sha256",
        "lineage_id",
        "disposition",
        "normalized_target",
    }
    schema_failures = [
        index
        for index, record in enumerate(lineage)
        if not required <= set(record)
        or record.get("schema_version") != "gse200304_raw_row_lineage_v1"
        or record.get("dataset_id") != "GSE200304"
        or record.get("raw_table_role")
        not in {"construct_table", "processed_label_table"}
        or not isinstance(record.get("raw_row_index"), int)
        or record.get("raw_row_index", -1) < 0
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(record.get("raw_row_fingerprint_sha256", "")),
        )
        or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("lineage_id", "")))
        or not re.fullmatch(r"[A-Z][A-Z0-9_]+", str(record.get("disposition", "")))
    ]
    raw_keys = [str(record.get("raw_row_key", "")) for record in lineage]
    lineage_ids = [str(record.get("lineage_id", "")) for record in lineage]
    table_counts = {
        role: sum(record.get("raw_table_role") == role for record in lineage)
        for role in ("construct_table", "processed_label_table")
    }
    target_ids = {str(record.get("row_id", "")) for record in paper_clean + rejected}
    invalid_targets = []
    for record in lineage:
        target = record.get("normalized_target")
        if target is None:
            continue
        if (
            not isinstance(target, dict)
            or not str(target.get("row_id", ""))
            or str(target.get("row_id")) not in target_ids
        ):
            invalid_targets.append(str(record.get("raw_row_key")))

    recomputed, raw_errors = _recompute_gse200304_raw_fingerprints(manifest)
    stored = {
        str(record.get("raw_row_key")): str(record.get("raw_row_fingerprint_sha256"))
        for record in lineage
    }
    summary = manifest.get("extraction_audit", {}).get("raw_row_lineage_summary", {})
    dispositions = Counter(str(record.get("disposition", "")) for record in lineage)
    expected_summary = {
        "schema_version": "gse200304_raw_row_lineage_summary_v1",
        "row_counts_by_table": table_counts,
        "total_raw_rows": len(lineage),
        "unique_raw_row_keys": len(set(raw_keys)),
        "duplicate_raw_row_keys": len(raw_keys) - len(set(raw_keys)),
        "unique_lineage_ids": len(set(lineage_ids)),
        "duplicate_lineage_ids": len(lineage_ids) - len(set(lineage_ids)),
        "unique_raw_row_fingerprints": len(
            {str(record.get("raw_row_fingerprint_sha256", "")) for record in lineage}
        ),
        "duplicate_raw_row_fingerprints": len(lineage)
        - len(
            {str(record.get("raw_row_fingerprint_sha256", "")) for record in lineage}
        ),
        "rows_with_normalized_target": sum(
            record.get("normalized_target") is not None for record in lineage
        ),
        "rows_without_normalized_target": sum(
            record.get("normalized_target") is None for record in lineage
        ),
        "disposition_counts": dict(sorted(dispositions.items())),
    }
    extraction = manifest.get("extraction_audit", {})
    expected_table_counts = {
        "construct_table": extraction.get("construct_rows"),
        "processed_label_table": extraction.get("label_rows"),
    }
    checks = {
        "strict_schema": not schema_failures,
        "raw_row_keys_unique": len(raw_keys) == len(set(raw_keys)),
        "lineage_ids_unique": len(lineage_ids) == len(set(lineage_ids)),
        "table_coverage_matches_extraction": table_counts == expected_table_counts,
        "normalized_targets_resolve": not invalid_targets,
        "summary_exactly_recomputed": summary == expected_summary,
        "raw_tables_reopened_without_error": not raw_errors,
        "raw_fingerprints_exactly_recomputed": stored == recomputed,
        "production_exact_row_counts": fixture_mode
        or table_counts
        == {
            "construct_table": 13836,
            "processed_label_table": 12704,
        },
    }
    return {
        "passed": all(checks.values()),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "schema_failure_indices": schema_failures,
        "invalid_normalized_targets": invalid_targets,
        "raw_recompute_errors": raw_errors,
        "stored_rows": len(stored),
        "recomputed_rows": len(recomputed),
    }


def _validate_dataset(
    dataset_root: Path,
    *,
    fixture_mode: bool,
) -> dict[str, Any]:
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_id = str(manifest.get("dataset_id", dataset_root.name))
    checks: list[dict[str, Any]] = []
    outputs = manifest.get("outputs", {})
    required_outputs = {
        "paper_clean",
        "label_store",
        "candidate_store",
        "rejected",
        "auxiliary",
        "raw_row_lineage",
        "accounting",
    }
    _check(
        checks,
        "required_outputs_declared",
        required_outputs <= set(outputs),
        sorted(set(outputs)),
    )

    resolved: dict[str, Path] = {}
    for output_name in required_outputs & set(outputs):
        output = outputs[output_name]
        path = dataset_root / output["path"]
        resolved[output_name] = path
        valid = (
            path.is_file()
            and path.stat().st_size == output["bytes"]
            and _sha256(path) == output["sha256"]
        )
        _check(
            checks,
            f"output_integrity:{output_name}",
            valid,
            str(path),
        )

    if not required_outputs <= set(resolved):
        dataset_id = str(manifest.get("dataset_id", dataset_root.name))
        return {
            "dataset_id": dataset_id,
            "status": manifest.get("status"),
            "paper_eligible": manifest.get("paper_eligible"),
            "fixture_mode": manifest.get("fixture_mode"),
            "passed": False,
            "checks": checks,
            "counts": {
                "paper_clean": 0,
                "labels": 0,
                "candidates": 0,
                "rejected": 0,
                "auxiliary": 0,
                "raw_row_lineage": 0,
            },
            "paper_clean_validation": {
                "passed": False,
                "reason": "required_outputs_missing",
            },
            "raw_row_lineage_validation": {
                "passed": False,
                "reason": "required_outputs_missing",
            },
        }

    paper_clean = _load_jsonl(resolved["paper_clean"])
    labels = _load_jsonl(resolved["label_store"])
    candidates = _load_jsonl(resolved["candidate_store"])
    rejected = _load_jsonl(resolved["rejected"])
    auxiliary = _load_jsonl(resolved["auxiliary"])
    raw_row_lineage = _load_jsonl(resolved["raw_row_lineage"])
    accounting = json.loads(resolved["accounting"].read_text(encoding="utf-8"))
    output_counts = {
        "paper_clean": len(paper_clean),
        "label_store": len(labels),
        "candidate_store": len(candidates),
        "rejected": len(rejected),
        "auxiliary": len(auxiliary),
        "raw_row_lineage": len(raw_row_lineage),
        "accounting": 1,
    }
    _check(
        checks,
        "output_record_counts_bound",
        all(
            outputs[name].get("records") == count
            for name, count in output_counts.items()
        ),
        {
            "declared": {name: outputs[name].get("records") for name in output_counts},
            "recomputed": output_counts,
        },
    )

    _check(
        checks,
        "candidate_and_label_paths_distinct",
        len(
            {
                resolved["paper_clean"].resolve(),
                resolved["candidate_store"].resolve(),
                resolved["label_store"].resolve(),
            }
        )
        == 3,
        {
            "candidate": str(resolved["candidate_store"]),
            "label": str(resolved["label_store"]),
        },
    )
    candidate_forbidden = sorted(
        {
            field
            for record in candidates
            for field in set(record) & CANDIDATE_STORE_FORBIDDEN_FIELDS
        }
    )
    candidate_nested_forbidden = sorted(
        {path for record in candidates for path in candidate_store_label_paths(record)}
    )
    _check(
        checks,
        "candidate_store_label_free",
        not candidate_forbidden and not candidate_nested_forbidden,
        {
            "top_level": candidate_forbidden,
            "recursive": candidate_nested_forbidden,
        },
    )
    _check(
        checks,
        "candidate_label_record_id_bijection",
        {record["record_id"] for record in candidates}
        == {record["record_id"] for record in labels},
        {
            "candidate_records": len(candidates),
            "label_records": len(labels),
        },
    )
    expected_candidates = [_candidate_record(record) for record in labels]
    _check(
        checks,
        "candidate_store_exact_canonical_projection",
        candidates == expected_candidates,
        {
            "records": len(candidates),
            "expected_records": len(expected_candidates),
            "actual_sha256": _canonical_json_sha256(candidates),
            "expected_sha256": _canonical_json_sha256(expected_candidates),
        },
    )
    paper_clean_validation = _validate_paper_clean(
        paper_clean,
        labels,
        rejected,
        auxiliary,
    )
    _check(
        checks,
        "paper_clean_strict_content_and_layer_binding",
        paper_clean_validation["passed"],
        paper_clean_validation,
    )
    expected_content_bindings = {
        "paper_clean": _identity_binding(paper_clean, "record_id"),
        "canonical": _identity_binding(labels, "record_id"),
        "candidate_store": _identity_binding(candidates, "record_id"),
        "rejected": _identity_binding(rejected, "row_id"),
        "auxiliary": _identity_binding(auxiliary, "row_id"),
        "raw_row_lineage": _identity_binding(raw_row_lineage, "lineage_id"),
    }
    _check(
        checks,
        "manifest_content_identities_exact",
        manifest.get("content_bindings") == expected_content_bindings,
        {
            "declared": manifest.get("content_bindings"),
            "recomputed": expected_content_bindings,
        },
    )

    missing_fields = {
        record.get("record_id", f"index:{index}"): _record_missing_fields(record)
        for index, record in enumerate(labels)
        if _record_missing_fields(record)
    }
    _check(
        checks,
        "canonical_required_fields",
        not missing_fields,
        missing_fields,
    )
    strict_validation_failures: dict[str, str] = {}
    for index, record in enumerate(labels):
        record_id = str(record.get("record_id", f"index:{index}"))
        try:
            validate_canonical_record(record)
        except Exception as exc:
            strict_validation_failures[record_id] = f"{type(exc).__name__}: {exc}"
    _check(
        checks,
        "strict_canonical_record_validator",
        not strict_validation_failures,
        strict_validation_failures,
    )

    roundtrip_failures: list[str] = []
    identity_failures: list[str] = []
    canonicalization_failures: list[str] = []
    trajectory_claims: list[str] = []
    absolute_mislabels: list[str] = []
    provenance_failures: list[str] = []
    paper_by_record_id = {
        str(record.get("record_id", "")): record for record in paper_clean
    }
    for record in labels:
        record_id = str(record.get("record_id", "UNKNOWN"))
        row_id = paper_by_record_id.get(record_id, {}).get("row_id")
        candidate = str(record.get("candidate_sequence", ""))
        candidate_hash = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
        expected_candidate_id = f"{dataset_id}:candidate:{candidate_hash[:20]}"
        expected_record_id: str
        if record.get("pair_type") not in ABSOLUTE_PAIR_TYPES:
            source = str(record.get("source_sequence", ""))
            source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
            identity_suffix = _stable_id(
                row_id,
                source_hash,
                candidate_hash,
                record.get("endpoint"),
                record.get("context_id"),
            )
            expected_record_id = f"{dataset_id}:record:{identity_suffix}"
            expected_source_id = f"{dataset_id}:source:{source_hash[:20]}"
            if record.get("source_id") != expected_source_id:
                identity_failures.append(record_id)
            try:
                applied = apply_edit_script(
                    record["source_sequence"], record["edit_script"]
                )
            except Exception:
                roundtrip_failures.append(record_id)
            else:
                if applied != record.get("candidate_sequence"):
                    roundtrip_failures.append(record_id)
            try:
                canonical = canonicalize_edit_script(source, candidate)
                expected_ambiguity = {
                    "equivalent_minimal_script_count": canonical[
                        "equivalent_minimal_script_count"
                    ],
                    "path_ambiguous": canonical["path_ambiguity"],
                    "canonicalization": "deterministic_minimal_script",
                    "ambiguity_category": canonical["ambiguity_category"],
                    "count_scope": canonical["count_scope"],
                    "trajectory_observed": False,
                }
                if (
                    record.get("edit_script") != canonical["actions"]
                    or record.get("edit_script_ambiguity") != expected_ambiguity
                ):
                    canonicalization_failures.append(record_id)
            except Exception:
                canonicalization_failures.append(record_id)
            if record.get("trajectory_observed") is not False:
                trajectory_claims.append(record_id)
            if record.get("intermediate_sequences"):
                trajectory_claims.append(record_id)
        elif record.get("pair_type") in ABSOLUTE_PAIR_TYPES:
            identity_suffix = _stable_id(
                row_id,
                candidate_hash,
                record.get("endpoint"),
                record.get("context_id"),
            )
            expected_record_id = f"{dataset_id}:absolute:{identity_suffix}"
            if (
                record.get("source_sequence") is not None
                or record.get("edit_script") is not None
                or "ABSOLUTE_SEQUENCE_NOT_INTERVENTION"
                not in record.get("quality_flags", [])
            ):
                absolute_mislabels.append(record_id)
        else:
            absolute_mislabels.append(record_id)
            expected_record_id = ""

        if (
            record_id != expected_record_id
            or record.get("candidate_id") != expected_candidate_id
        ):
            identity_failures.append(record_id)

        sequence_provenance = record.get("sequence_provenance")
        if not isinstance(sequence_provenance, dict):
            provenance_failures.append(record_id)
        if not record.get("download_manifest") or not record.get("license"):
            provenance_failures.append(record_id)

    _check(
        checks,
        "edit_script_roundtrip_100_percent",
        not roundtrip_failures,
        roundtrip_failures,
    )
    _check(
        checks,
        "record_identity_recomputed_from_endpoints",
        not identity_failures,
        sorted(set(identity_failures)),
    )
    _check(
        checks,
        "canonical_edit_script_and_ambiguity_exactly_recomputed",
        not canonicalization_failures,
        sorted(set(canonicalization_failures)),
    )
    _check(
        checks,
        "constructed_paths_never_observed",
        not trajectory_claims,
        sorted(set(trajectory_claims)),
    )
    _check(
        checks,
        "absolute_sequences_not_interventions",
        not absolute_mislabels,
        absolute_mislabels,
    )
    _check(
        checks,
        "record_provenance_complete",
        not provenance_failures,
        sorted(set(provenance_failures)),
    )

    bad_rejections = [
        record.get("row_id", f"index:{index}")
        for index, record in enumerate(rejected)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]+", str(record.get("reason_code", "")))
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(record.get("row_fingerprint_sha256", "")),
        )
    ]
    _check(
        checks,
        "rejected_rows_have_stable_reason_and_fingerprint",
        not bad_rejections,
        bad_rejections,
    )

    accounted = (
        int(accounting.get("accepted_input_rows", -1)) + len(auxiliary) + len(rejected)
    )
    accounting_valid = (
        accounting.get("accounted_rows") == accounted
        and accounting.get("total_input_rows") == accounted
        and accounting.get("accepted_intervention_rows")
        == sum(record.get("pair_type") not in ABSOLUTE_PAIR_TYPES for record in labels)
        and accounting.get("accepted_absolute_rows")
        == sum(record.get("pair_type") in ABSOLUTE_PAIR_TYPES for record in labels)
        and accounting.get("auxiliary_source_anchor_rows") == len(auxiliary)
        and accounting.get("rejected_rows") == len(rejected)
    )
    _check(
        checks,
        "accepted_rejected_accounting_exact",
        accounting_valid,
        {"manifest": accounting, "recomputed": accounted},
    )

    provenance_audit = manifest.get("input_provenance", {}).get("provenance_audit", {})
    provenance_integrity_failures: list[str] = []
    for item in provenance_audit.get("raw_files", []):
        path = Path(str(item.get("path", "")))
        if (
            not path.is_file()
            or path.stat().st_size != item.get("bytes")
            or _sha256(path) != item.get("sha256")
        ):
            provenance_integrity_failures.append(str(path))
    download_manifest = manifest.get("input_provenance", {}).get("download_manifest")
    if isinstance(download_manifest, dict):
        path = Path(str(download_manifest.get("path", "")))
        if (
            not path.is_file()
            or path.stat().st_size != download_manifest.get("bytes")
            or _sha256(path) != download_manifest.get("sha256")
        ):
            provenance_integrity_failures.append(str(path))
    accepted_provenance_passed = (
        bool(provenance_audit.get("complete")) and not provenance_integrity_failures
    )
    blocked_or_excluded = manifest.get("status") == "blocked"
    input_provenance = manifest.get("input_provenance", {})
    license_text = str(input_provenance.get("license") or "").strip()
    metadata_only_provenance_passed = (
        blocked_or_excluded
        and manifest.get("read_final_labels") is False
        and bool(
            re.fullmatch(
                r"[A-Z][A-Z0-9_]+",
                str(manifest.get("reason_code", "")),
            )
        )
        and input_provenance.get("input_access")
        == "not_opened_due_to_fail_closed_dataset_policy"
        and isinstance(download_manifest, dict)
        and not provenance_integrity_failures
        and bool(license_text)
        and license_text.upper() not in {"UNKNOWN", "UNRESOLVED", "TBD", "MISSING"}
    )
    _check(
        checks,
        "production_input_provenance_complete",
        fixture_mode or accepted_provenance_passed or metadata_only_provenance_passed,
        {
            "fixture_exemption": fixture_mode,
            "blocked_or_excluded": blocked_or_excluded,
            "accepted_provenance_passed": accepted_provenance_passed,
            "metadata_only_provenance_passed": metadata_only_provenance_passed,
            "audit": provenance_audit,
            "integrity_failures": provenance_integrity_failures,
        },
    )
    if dataset_id == "GSE246381":
        bad_exposure = [
            record["record_id"]
            for record in labels
            if record.get("historical_exposure")
            != "historically_exposed_retrospective_E4"
            or record.get("paper_split") != "retrospective_only"
            or record.get("canonical_split") != "retrospective_only"
            or "NO_TRAINING_OR_SELECTION" not in record.get("quality_flags", [])
        ]
        _check(
            checks,
            "gse246381_retrospective_E4_only",
            not bad_exposure,
            bad_exposure,
        )
    if dataset_id == "GSE200304" and not fixture_mode:
        reconciliation = manifest.get("paper_count_reconciliation", {})
        extraction = manifest.get("extraction_audit", {})
        _check(
            checks,
            "gse200304_exact_construct_label_join",
            extraction.get("construct_rows") == 13836
            and extraction.get("construct_unique_merged_ids") == 13836
            and extraction.get("label_unique_barcodes") == 12704
            and extraction.get("label_barcodes_joined_to_construct") == 12704
            and extraction.get("unjoined_label_barcodes") == 0
            and extraction.get("sequence_pair_groups") == 6885
            and extraction.get("pair_201nt_count") == 6885
            and extraction.get("pair_hamming_distribution") == {"1": 6885}
            and extraction.get("control_constructs") == 66
            and extraction.get("labeled_control_constructs") == 47
            and extraction.get("malformed_pair_constructs") == 0
            and extraction.get("production_gate_exact") is True,
            extraction,
        )
        _check(
            checks,
            "gse200304_sequence_and_label_coverage_disclosed",
            reconciliation.get("paper_reported_pairs") == 6892
            and reconciliation.get("production_expected_sequence_pairs") == 6885
            and reconciliation.get("production_observed_sequence_pairs") == 6885
            and reconciliation.get("observed_label_coverage")
            == {
                "both_labeled": 6120,
                "source_only": 192,
                "candidate_only": 225,
                "neither_labeled": 348,
            }
            and reconciliation.get("known_discrepancy") == 7
            and reconciliation.get("status")
            == "sequence_and_label_coverage_reconciled",
            reconciliation,
        )
    lineage_validation: dict[str, Any] | None = None
    if dataset_id == "GSE200304":
        lineage_validation = _validate_gse200304_lineage(
            raw_row_lineage,
            manifest,
            paper_clean,
            rejected,
            fixture_mode=fixture_mode,
        )
        _check(
            checks,
            "gse200304_raw_row_lineage_exact",
            lineage_validation["passed"],
            lineage_validation,
        )
    if dataset_id == "GSE217518" and not fixture_mode:
        reconciliation = manifest.get("paper_count_reconciliation", {})
        _check(
            checks,
            "gse217518_official_csv_counts_and_paper_gap_disclosed",
            reconciliation.get("paper_reported_total_sequences") == 12472
            and reconciliation.get("paper_reported_pairs") == 6555
            and reconciliation.get("official_csv_rows")
            == {"three_utr": 3275, "five_utr": 4601}
            and reconciliation.get("official_csv_unique_pairs")
            == {"three_utr": 1124, "five_utr": 1756}
            and reconciliation.get("status")
            == "official_csv_counts_reconciled_and_paper_difference_disclosed",
            reconciliation,
        )

    if manifest.get("status") == "blocked":
        _check(
            checks,
            "blocked_dataset_has_no_admitted_records",
            not labels and not candidates and manifest.get("paper_eligible") is False,
            {
                "label_records": len(labels),
                "candidate_records": len(candidates),
                "reason_code": manifest.get("reason_code"),
            },
        )

    return {
        "dataset_id": dataset_id,
        "status": manifest.get("status"),
        "paper_eligible": manifest.get("paper_eligible"),
        "fixture_mode": manifest.get("fixture_mode"),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "counts": {
            "paper_clean": len(paper_clean),
            "labels": len(labels),
            "candidates": len(candidates),
            "rejected": len(rejected),
            "auxiliary": len(auxiliary),
            "raw_row_lineage": len(raw_row_lineage),
        },
        "paper_clean_validation": paper_clean_validation,
        "raw_row_lineage_validation": lineage_validation,
    }


def validate_d1_root(
    stage_d1_root: Path,
    *,
    fixture_mode: bool = False,
    artifact_root: Path | None = None,
    builder_audit_root: Path | None = None,
    require_global_stores: bool = True,
) -> dict[str, Any]:
    datasets_root = stage_d1_root / "datasets"
    dataset_roots = (
        sorted(
            path
            for path in datasets_root.iterdir()
            if path.is_dir() and (path / "manifest.json").is_file()
        )
        if datasets_root.is_dir()
        else []
    )
    dataset_results = [
        _validate_dataset(path, fixture_mode=fixture_mode) for path in dataset_roots
    ]
    present = {result["dataset_id"] for result in dataset_results}
    required = {"GSE114002", "GSE200304", "GSE246381", "GSE217518"}
    required_accepted = {
        result["dataset_id"]
        for result in dataset_results
        if result["dataset_id"] in required
        and result["status"] in {"accepted", "accepted_fixture"}
    }
    required_paper_eligible = {
        result["dataset_id"]
        for result in dataset_results
        if result["dataset_id"] in required and result["paper_eligible"] is True
    }
    global_store_validation = _validate_global_stores(
        stage_d1_root,
        required=require_global_stores,
    )
    builder_audit_validation = _validate_builder_audit(
        stage_d1_root,
        artifact_root,
        builder_audit_root,
        required=not fixture_mode,
    )
    config_binding_validation = _validate_build_config_binding(
        stage_d1_root,
        required=require_global_stores,
        builder_audit_validation=builder_audit_validation,
    )
    dataset_manifest_binding_validation = _validate_dataset_manifest_bindings(
        stage_d1_root,
        required=require_global_stores,
    )
    structural = (
        bool(dataset_results)
        and all(result["passed"] for result in dataset_results)
        and required <= present
        and D1_SCOPE_DATASETS == present
        and required_accepted == required
        and (fixture_mode or required_paper_eligible == required)
        and global_store_validation["passed"]
        and config_binding_validation["passed"]
        and dataset_manifest_binding_validation["passed"]
        and builder_audit_validation["passed"]
    )
    artifact_validation = _validate_required_artifacts(
        stage_d1_root,
        artifact_root,
    )
    phase_gate = structural and not fixture_mode and artifact_validation["passed"]
    return {
        "schema_version": "d1_acceptance_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage_d1_root": str(stage_d1_root),
        "fixture_mode": fixture_mode,
        "evidence_level": (
            "fixture_only" if fixture_mode else "production_reconstruction"
        ),
        "dataset_results": dataset_results,
        "required_supported_datasets": sorted(required),
        "missing_required_datasets": sorted(required - present),
        "expected_d1_scope_datasets": sorted(D1_SCOPE_DATASETS),
        "missing_d1_scope_datasets": sorted(D1_SCOPE_DATASETS - present),
        "structural_validation_passed": structural,
        "required_artifact_validation": artifact_validation,
        "global_store_validation": global_store_validation,
        "config_binding_validation": config_binding_validation,
        "dataset_manifest_binding_validation": (dataset_manifest_binding_validation),
        "builder_audit_validation": builder_audit_validation,
        "phase_gate_passed": phase_gate,
        "scientific_result_claimed": False,
        "note": (
            "A green structural validator is not a final scientific result. "
            "Fixture validation can never freeze D1."
        ),
    }


def _validate_builder_audit(
    stage_d1_root: Path,
    artifact_root: Path | None,
    audit_root: Path | None,
    *,
    required: bool,
) -> dict[str, Any]:
    if not required:
        return {"passed": True, "waived_for_unit_fixture": True}
    if audit_root is None:
        return {"passed": False, "reason": "builder_audit_root_not_supplied"}
    audit_root = audit_root.resolve()
    manifest_path = audit_root / "audit_manifest.json"
    build_manifest_path = stage_d1_root.resolve() / "build_manifest.json"
    try:
        audit = _load_json(manifest_path)
        evidence = audit["evidence"]
        verified_evidence: dict[str, Any] = {}
        evidence_valid = True
        for name in (
            "invocation",
            "completion",
            "process",
            "stdout",
            "stderr",
            "explicit_prelaunch_recheck",
        ):
            binding = evidence.get(name)
            if not isinstance(binding, Mapping):
                evidence_valid = False
                verified_evidence[name] = {"error": "missing_binding"}
                continue
            passed, actual = _artifact_binding_valid(audit_root, binding)
            evidence_valid &= passed
            verified_evidence[name] = {
                "passed": passed,
                "declared": dict(binding),
                "actual": actual,
            }
        invocation = _load_json(audit_root / "invocation.json")
        completion = _load_json(audit_root / "completion.json")
        process = _load_json(audit_root / "process.json")
        explicit_prelaunch_recheck = _load_json(
            audit_root / "git/explicit_prelaunch_recheck.json"
        )
        stdout_bytes = (audit_root / "logs/stdout.log").read_bytes()
        stderr_bytes = (audit_root / "logs/stderr.log").read_bytes()
        build_manifest = _load_json(build_manifest_path)
    except Exception as exc:
        return {
            "passed": False,
            "reason": f"builder_audit_parse_failure:{type(exc).__name__}:{exc}",
            "audit_root": str(audit_root),
        }

    git_snapshot = audit.get("git_prelaunch_snapshot")
    git_evidence_valid = isinstance(git_snapshot, Mapping)
    verified_git_evidence: dict[str, Any] = {}
    explicit_prelaunch_manifest_valid = False
    index_flags_manifest_valid = False
    if isinstance(git_snapshot, Mapping):
        git_artifacts = dict(git_snapshot.get("artifacts", {}))
        git_artifacts["snapshot_manifest"] = git_snapshot.get("snapshot_manifest")
        for name in (
            "head",
            "status_porcelain_v2_z",
            "diff_head_binary",
            "untracked_paths_z",
            "untracked_content_hashes",
            "explicit_prelaunch_files",
            "index_flags",
            "snapshot_manifest",
        ):
            binding = git_artifacts.get(name)
            if not isinstance(binding, Mapping):
                git_evidence_valid = False
                verified_git_evidence[name] = {"error": "missing_binding"}
                continue
            passed, actual = _artifact_binding_valid(audit_root, binding)
            git_evidence_valid &= passed
            verified_git_evidence[name] = {
                "passed": passed,
                "declared": dict(binding),
                "actual": actual,
            }
        explicit_verification = verified_git_evidence.get(
            "explicit_prelaunch_files", {}
        )
        index_verification = verified_git_evidence.get("index_flags", {})
        try:
            explicit_manifest = _load_json(
                Path(str(explicit_verification["actual"]["path"]))
            )
            explicit_entries = explicit_manifest.get("entries")
            explicit_prelaunch_manifest_valid = (
                explicit_verification.get("passed") is True
                and explicit_manifest.get("schema_version")
                == "git_explicit_prelaunch_files.v1"
                and isinstance(explicit_entries, list)
                and explicit_manifest.get("entry_count") == len(explicit_entries)
                and all(
                    isinstance(item, Mapping)
                    and item.get("kind") == "regular_file"
                    and isinstance(item.get("path"), str)
                    and isinstance(item.get("bytes"), int)
                    and item.get("bytes") >= 0
                    and bool(re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))))
                    for item in explicit_entries
                )
            )
        except (KeyError, OSError, TypeError, ValueError):
            explicit_prelaunch_manifest_valid = False
        try:
            index_manifest = _load_json(Path(str(index_verification["actual"]["path"])))
            index_entries = index_manifest.get("entries")
            index_flags_manifest_valid = (
                index_verification.get("passed") is True
                and index_manifest.get("schema_version") == "git_index_flags.v1"
                and isinstance(index_entries, list)
                and index_manifest.get("entry_count") == len(index_entries)
                and index_manifest.get("all_entries_normal") is True
                and index_manifest.get("unsafe_entries") == []
                and all(
                    isinstance(item, Mapping)
                    and item.get("tag") == "H"
                    and item.get("safe_normal_index_entry") is True
                    for item in index_entries
                )
            )
        except (KeyError, OSError, TypeError, ValueError):
            index_flags_manifest_valid = False

    argv = audit.get("argv")
    argv_paths: dict[str, str] = {}
    argv_shape = (
        isinstance(argv, list)
        and len(argv) == 8
        and Path(str(argv[1])).name == "build_d1_utr_benchmark.py"
        and argv[2::2] == ["--config", "--output-root", "--artifact-root"]
    )
    if argv_shape:
        argv_paths = {
            "config": str(Path(str(argv[3])).resolve()),
            "output_root": str(Path(str(argv[5])).resolve()),
            "artifact_root": str(Path(str(argv[7])).resolve()),
        }
    nonblank_stdout = [
        line for line in stdout_bytes.decode("utf-8").splitlines() if line.strip()
    ]
    stdout_payload: Any = None
    if len(nonblank_stdout) == 1:
        try:
            stdout_payload = json.loads(nonblank_stdout[0])
        except json.JSONDecodeError:
            pass
    cuda = audit.get("cuda", {})
    exact_cuda_statement = (
        "This workload is non-neural. The wrapper intentionally did not "
        "probe or launch CUDA and masked CUDA_VISIBLE_DEVICES for the child."
    )
    project_root = Path(str(audit.get("project_root", "")))
    checks = {
        "audit_manifest_exact_location": manifest_path
        == audit_root / "audit_manifest.json",
        "audit_run_root_exact": audit.get("run_root") == str(audit_root),
        "evidence_path_bytes_sha_recursive": evidence_valid,
        "command_completed": (
            audit.get("state") == "COMMAND_COMPLETED"
            and audit.get("stop_reason") is None
            and audit.get("observed_process_exit_code") == 0
            and audit.get("wrapper_exit_code") == 0
            and completion.get("state") == "COMMAND_COMPLETED"
            and completion.get("observed_process_exit_code") == 0
            and completion.get("wrapper_exit_code") == 0
        ),
        "argv_exact_builder_shape": argv_shape,
        "argv_identical_across_evidence": (
            argv == invocation.get("argv") == process.get("argv")
        ),
        "working_directory_is_project_root": (
            audit.get("working_directory") == audit.get("project_root")
            and invocation.get("working_directory") == audit.get("project_root")
        ),
        "builder_script_resolves_in_project": (
            argv_shape
            and project_root.is_absolute()
            and (
                Path(str(argv[1]))
                if Path(str(argv[1])).is_absolute()
                else project_root / str(argv[1])
            ).resolve()
            == (project_root / "scripts/data/build_d1_utr_benchmark.py").resolve()
        ),
        "output_root_exact": (
            argv_shape and argv_paths["output_root"] == str(stage_d1_root.resolve())
        ),
        "artifact_root_exact": (
            artifact_root is not None
            and argv_shape
            and argv_paths["artifact_root"] == str(artifact_root.resolve())
        ),
        "config_path_exact": (
            argv_shape
            and argv_paths["config"]
            == str(Path(str(build_manifest.get("config_path", ""))).resolve())
        ),
        "stdout_one_json_line_equals_build_manifest": (
            stdout_payload == build_manifest and len(nonblank_stdout) == 1
        ),
        "stderr_empty": stderr_bytes == b"",
        "non_neural_workload_exact": (
            audit.get("workload_class") == "NON_NEURAL_DATA_BENCHMARK"
            and invocation.get("workload_class") == "NON_NEURAL_DATA_BENCHMARK"
            and cuda.get("statement") == exact_cuda_statement
            and cuda.get("formal_neural_activity") is False
            and cuda.get("gpu_validation_started") is False
            and cuda.get("automatic_cpu_fallback") is False
        ),
        "protection_claims_exact": audit.get("protection")
        == {
            "exclusive_new_run_root": True,
            "existing_results_overwritten": 0,
            "unrelated_processes_terminated": 0,
            "only_exact_child_pid_may_receive_interrupt": True,
        },
        "git_prelaunch_snapshot_present": isinstance(git_snapshot, Mapping),
        "git_prelaunch_snapshot_before_command": (
            isinstance(git_snapshot, Mapping)
            and git_snapshot.get("captured_before_command") is True
        ),
        "git_index_flags_safe": (
            isinstance(git_snapshot, Mapping)
            and git_snapshot.get("index_flags_safe") is True
        ),
        "git_explicit_prelaunch_manifest_semantics": (
            explicit_prelaunch_manifest_valid
        ),
        "git_index_flags_manifest_semantics": index_flags_manifest_valid,
        "git_explicit_prelaunch_recheck_passed": (
            explicit_prelaunch_recheck.get("schema_version")
            == "git_explicit_prelaunch_recheck.v1"
            and explicit_prelaunch_recheck.get(
                "checked_immediately_before_child_launch"
            )
            is True
            and explicit_prelaunch_recheck.get("matches") is True
            and explicit_prelaunch_recheck.get("captured_manifest_sha256")
            == (
                git_snapshot.get("artifacts", {})
                .get("explicit_prelaunch_files", {})
                .get("sha256")
                if isinstance(git_snapshot, Mapping)
                else None
            )
            and explicit_prelaunch_recheck.get("observed_manifest_sha256")
            == explicit_prelaunch_recheck.get("captured_manifest_sha256")
        ),
        "git_prelaunch_artifacts_path_bytes_sha_recursive": git_evidence_valid,
    }
    return {
        "passed": all(checks.values()),
        "reason": None if all(checks.values()) else "builder_audit_gate_failed",
        "checks": checks,
        "audit_root": str(audit_root),
        "audit_manifest": {
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256": _sha256(manifest_path),
        },
        "git_prelaunch_snapshot": git_snapshot,
        "verified_evidence": verified_evidence,
        "verified_git_evidence": verified_git_evidence,
    }


def _validate_build_config_binding(
    stage_d1_root: Path,
    *,
    required: bool,
    builder_audit_validation: Mapping[str, Any],
) -> dict[str, Any]:
    if not required:
        return {"passed": True, "waived_for_unit_fixture": True}
    build_manifest_path = stage_d1_root / "build_manifest.json"
    if not build_manifest_path.is_file():
        return {
            "passed": False,
            "reason": "build_manifest_missing",
            "path": str(build_manifest_path),
        }
    try:
        build_manifest = json.loads(build_manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "passed": False,
            "reason": f"build_manifest_parse_failure:{type(exc).__name__}:{exc}",
            "path": str(build_manifest_path),
        }

    config_path = Path(str(build_manifest.get("config_path", "")))
    declared_bytes = build_manifest.get("config_bytes")
    declared_sha256 = str(build_manifest.get("config_sha256", ""))
    exists = config_path.is_file()
    actual_bytes = config_path.stat().st_size if exists else None
    actual_sha256 = _sha256(config_path) if exists else None
    parsed_config: dict[str, Any] | None = None
    semantic_error: str | None = None
    if exists:
        try:
            parsed_config = _parse_config(config_path.read_bytes())
        except Exception as exc:
            semantic_error = f"{type(exc).__name__}:{exc}"
    selection = (
        parsed_config.get("selection_policy", {})
        if isinstance(parsed_config, dict)
        else {}
    )
    scope_binding = selection.get("dataset_scope_manifest")
    inventory_binding = (
        parsed_config.get("input_inventory")
        if isinstance(parsed_config, dict)
        else None
    )
    dataset_ids = (
        [str(row.get("dataset_id", "")) for row in parsed_config["datasets"]]
        if isinstance(parsed_config, dict)
        else []
    )
    project_root: Path | None = None
    git_snapshot = builder_audit_validation.get("git_prelaunch_snapshot")
    audit_root_text = builder_audit_validation.get("audit_root")
    if isinstance(git_snapshot, Mapping):
        candidate_root = Path(str(git_snapshot.get("repository", "")))
        if candidate_root.is_absolute():
            project_root = candidate_root
    stage_id = (
        str(parsed_config.get("stage_id", ""))
        if isinstance(parsed_config, dict)
        else ""
    )

    config_repository_path = (
        str(parsed_config.get("config_repository_path"))
        if isinstance(parsed_config, dict)
        and parsed_config.get("config_repository_path")
        else EXPECTED_CONFIG_REPOSITORY_PATH
    )
    scope_repository_path = (
        str(scope_binding.get("repository_path"))
        if isinstance(scope_binding, Mapping) and scope_binding.get("repository_path")
        else EXPECTED_SCOPE_REPOSITORY_PATH
    )
    inventory_repository_path = (
        str(inventory_binding.get("repository_path"))
        if isinstance(inventory_binding, Mapping)
        and inventory_binding.get("repository_path")
        else f"artifacts/stages/{stage_id}/D1/input_inventory.json"
    )
    inventory_path = (
        Path(str(inventory_binding.get("path")))
        if isinstance(inventory_binding, Mapping) and inventory_binding.get("path")
        else (
            project_root / inventory_repository_path
            if project_root is not None
            else Path("")
        )
    )
    inventory: dict[str, Any] | None = None
    inventory_error: str | None = None
    inventory_bytes = inventory_path.stat().st_size if inventory_path.is_file() else -1
    inventory_sha256 = _sha256(inventory_path) if inventory_path.is_file() else ""
    if inventory_path.is_file():
        try:
            inventory = _load_json(inventory_path)
        except Exception as exc:
            inventory_error = f"{type(exc).__name__}:{exc}"

    prelaunch_bindings: dict[str, Any] = {}
    if (
        audit_root_text
        and isinstance(git_snapshot, Mapping)
        and project_root is not None
    ):
        audit_root = Path(str(audit_root_text))
        for name, relative, size, digest in (
            (
                "config",
                config_repository_path,
                int(declared_bytes) if isinstance(declared_bytes, int) else -1,
                declared_sha256,
            ),
            (
                "scope",
                scope_repository_path,
                (
                    int(scope_binding.get("bytes", -1))
                    if isinstance(scope_binding, Mapping)
                    else -1
                ),
                (
                    str(scope_binding.get("sha256", ""))
                    if isinstance(scope_binding, Mapping)
                    else ""
                ),
            ),
            (
                "input_inventory",
                inventory_repository_path,
                inventory_bytes,
                inventory_sha256,
            ),
        ):
            prelaunch_bindings[name] = _prelaunch_file_binding(
                audit_root,
                git_snapshot,
                relative,
                expected_bytes=size,
                expected_sha256=digest,
            )

    inventory_files = (
        inventory.get("files", []) if isinstance(inventory, Mapping) else []
    )
    inventory_by_path = {
        str(item.get("path")): item
        for item in inventory_files
        if isinstance(item, Mapping)
    }
    role_map = {
        "sequence_and_provided_label_input": "SEQUENCE_AND_PROVIDED_LABEL_INPUT",
        "construct_table": "STRUCTURAL_SEQUENCE_AND_PAIR_INPUT",
        "processed_label_table": "PROCESSED_LABEL_INPUT",
        "three_utr_sequence_and_processed_label_input": "THREE_UTR_SEQUENCE_AND_PROCESSED_LABEL_INPUT",
        "five_utr_sequence_and_processed_label_input": "FIVE_UTR_SEQUENCE_AND_PROCESSED_LABEL_INPUT",
        "historically_exposed_sequence_pair_input": "HISTORICALLY_EXPOSED_SEQUENCE_PAIR_INPUT",
    }
    configured_inputs: list[tuple[str, str, str]] = []
    inventory_semantic_failures: list[str] = []
    if isinstance(parsed_config, Mapping):
        for dataset in parsed_config["datasets"]:
            for item in dataset.get("input_files", []):
                path = str(item.get("path", ""))
                expected_role = role_map.get(str(item.get("role", "")))
                configured_inputs.append(
                    (str(dataset["dataset_id"]), path, str(expected_role))
                )
                inventoried = inventory_by_path.get(path)
                if (
                    not isinstance(inventoried, Mapping)
                    or inventoried.get("dataset_id") != dataset["dataset_id"]
                    or inventoried.get("role") != expected_role
                ):
                    inventory_semantic_failures.append(path)
    live_inventory_failures = [
        str(item.get("path", ""))
        for item in inventory_files
        if not Path(str(item.get("path", ""))).is_file()
        or Path(str(item.get("path", ""))).stat().st_size != item.get("bytes")
        or _sha256(Path(str(item.get("path", "")))) != item.get("sha256")
    ]
    expected_counts = Counter(
        str(item.get("dataset_id", "")) for item in inventory_files
    )
    frozen_inventory_counts = {
        "GSE114002": 2,
        "GSE200304": 3,
        "GSE217518": 2,
        "GSE246381": 1,
        "MPRAu_processed_ENCSR854RUF": 1,
    }
    frozen_blocked_without_copy = {
        "GSE149487",
        "GSE145046",
        "ENCSR854RUF_raw62",
        "GSE330741",
        "GSE291719",
        "GSE207584",
        "GSE173083",
    }
    gse246_inventory = [
        item for item in inventory_files if item.get("dataset_id") == "GSE246381"
    ]
    mprau_inventory = [
        item
        for item in inventory_files
        if item.get("dataset_id") == "MPRAu_processed_ENCSR854RUF"
    ]
    production_control_binding_required = (
        builder_audit_validation.get("waived_for_unit_fixture") is not True
    )
    checks = {
        "config_path_absolute": config_path.is_absolute(),
        "config_path_exists": exists,
        "config_bytes_bound": (isinstance(declared_bytes, int) and declared_bytes >= 0),
        "config_sha256_bound": bool(re.fullmatch(r"[0-9a-f]{64}", declared_sha256)),
        "config_size_matches": exists and actual_bytes == declared_bytes,
        "config_sha256_matches": exists and actual_sha256 == declared_sha256,
        "config_semantics_valid": parsed_config is not None,
        "config_exact_12_scope": (
            len(dataset_ids) == len(D1_SCOPE_DATASETS)
            and len(dataset_ids) == len(set(dataset_ids))
            and set(dataset_ids) == D1_SCOPE_DATASETS
        ),
        "contract_hash_bound": (
            selection.get("goal_contract_sha256") == EXPECTED_GOAL_CONTRACT_SHA256
            and build_manifest.get("goal_contract_sha256")
            == EXPECTED_GOAL_CONTRACT_SHA256
        ),
        "scope_manifest_path_bytes_sha_bound": (
            isinstance(scope_binding, dict)
            and build_manifest.get("dataset_scope_manifest") == scope_binding
        ),
        "repository_paths_exact": (
            not production_control_binding_required
            or (
                config_repository_path == EXPECTED_CONFIG_REPOSITORY_PATH
                and scope_repository_path == EXPECTED_SCOPE_REPOSITORY_PATH
                and inventory_repository_path
                == f"artifacts/stages/{stage_id}/D1/input_inventory.json"
            )
        ),
        "all_control_files_bound_to_prelaunch_snapshot": (
            not production_control_binding_required
            or (
                set(prelaunch_bindings) == {"config", "scope", "input_inventory"}
                and all(
                    binding.get("passed") is True
                    for binding in prelaunch_bindings.values()
                )
            )
        ),
        "input_inventory_schema_and_stage_exact": (
            not production_control_binding_required
            or (
                isinstance(inventory, Mapping)
                and inventory.get("schema_version") == "d1_input_inventory.v1"
                and inventory.get("artifact_type") == "d1_input_inventory"
                and inventory.get("stage_id") == stage_id
                and inventory.get("selection_is_label_independent") is True
            )
        ),
        "configured_inputs_exactly_in_inventory": (
            not production_control_binding_required
            or (
                not inventory_semantic_failures
                and len(configured_inputs) == 6
                and len({path for _, path, _ in configured_inputs}) == 6
            )
        ),
        "inventory_counts_recomputed": (
            not production_control_binding_required
            or (
                isinstance(inventory, Mapping)
                and inventory.get("dataset_input_counts")
                == dict(sorted(expected_counts.items()))
            )
        ),
        "input_inventory_frozen_scope_exact": (
            not production_control_binding_required
            or (
                len(inventory_files) == 9
                and dict(sorted(expected_counts.items())) == frozen_inventory_counts
                and set(inventory.get("blocked_or_excluded_without_input_copy", []))
                == frozen_blocked_without_copy
                and len(gse246_inventory) == 1
                and gse246_inventory[0].get("role")
                == "HISTORICALLY_EXPOSED_SEQUENCE_PAIR_INPUT"
                and gse246_inventory[0].get("label_access")
                == "E4_RETROSPECTIVE_ONLY_NO_SELECTION"
                and len(mprau_inventory) == 1
                and mprau_inventory[0].get("role")
                == "CONDITIONAL_PROCESSED_INPUT_BLOCKED_WITHOUT_GRCH37"
                and mprau_inventory[0].get("label_access")
                == ("NO_D1_RECORD_ADMISSION_WHILE_SEQUENCE_" "RECONSTRUCTION_BLOCKED")
            )
        ),
        "inventory_live_path_bytes_sha_exact": (
            not production_control_binding_required or not live_inventory_failures
        ),
        "dataset_role_selection_label_independent": (
            selection.get("candidate_final_labels_used_for_dataset_role_selection")
            is False
            and build_manifest.get(
                "candidate_final_labels_used_for_dataset_role_selection"
            )
            is False
        ),
    }
    return {
        "passed": all(checks.values()),
        "reason": None if all(checks.values()) else "stale_or_unbound_build_config",
        "checks": checks,
        "config_path": str(config_path),
        "declared_bytes": declared_bytes,
        "actual_bytes": actual_bytes,
        "declared_sha256": declared_sha256,
        "actual_sha256": actual_sha256,
        "semantic_error": semantic_error,
        "dataset_ids": dataset_ids,
        "scope_manifest_binding": scope_binding,
        "input_inventory_binding": {
            "path": str(inventory_path),
            "bytes": inventory_bytes,
            "sha256": inventory_sha256,
            "legacy_inference_used": not isinstance(inventory_binding, Mapping),
        },
        "prelaunch_bindings": prelaunch_bindings,
        "inventory_error": inventory_error,
        "inventory_semantic_failures": inventory_semantic_failures,
        "inventory_live_failures": live_inventory_failures,
    }


def _validate_dataset_manifest_bindings(
    stage_d1_root: Path,
    *,
    required: bool,
) -> dict[str, Any]:
    if not required:
        return {"passed": True, "waived_for_unit_fixture": True}
    build_manifest_path = stage_d1_root / "build_manifest.json"
    try:
        build_manifest = json.loads(build_manifest_path.read_text(encoding="utf-8"))
        summaries = build_manifest["datasets"]
    except Exception as exc:
        return {
            "passed": False,
            "reason": f"build_manifest_parse_failure:{type(exc).__name__}:{exc}",
            "build_manifest": {
                "path": str(build_manifest_path),
                "exists": build_manifest_path.is_file(),
            },
        }
    dataset_ids = [str(summary.get("dataset_id", "")) for summary in summaries]
    failures: dict[str, Any] = {}
    for summary in summaries:
        dataset_id = str(summary.get("dataset_id", ""))
        declared = summary.get("manifest")
        expected_relative = f"datasets/{dataset_id}/manifest.json"
        if not isinstance(declared, dict):
            failures[dataset_id] = "manifest_binding_not_object"
            continue
        path = stage_d1_root / str(declared.get("path", ""))
        try:
            dataset_manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures[dataset_id] = f"manifest_parse_failure:{type(exc).__name__}:{exc}"
            continue
        checks = {
            "path_exact": declared.get("path") == expected_relative,
            "path_resolves_within_stage": path.resolve()
            == (stage_d1_root / expected_relative).resolve(),
            "bytes_match": path.stat().st_size == declared.get("bytes"),
            "sha256_match": _sha256(path) == declared.get("sha256"),
            "dataset_id_match": dataset_manifest.get("dataset_id") == dataset_id,
            "summary_status_match": summary.get("status")
            == dataset_manifest.get("status"),
            "summary_reason_match": summary.get("reason_code")
            == dataset_manifest.get("reason_code"),
            "summary_paper_eligible_match": summary.get("paper_eligible")
            == dataset_manifest.get("paper_eligible"),
            "summary_fixture_mode_match": summary.get("fixture_mode")
            == dataset_manifest.get("fixture_mode"),
            "summary_accounting_match": summary.get("accounting")
            == dataset_manifest.get("accounting"),
            "summary_roundtrip_match": summary.get("roundtrip_audit")
            == dataset_manifest.get("roundtrip_audit"),
        }
        if not all(checks.values()):
            failures[dataset_id] = checks
    top_level_checks = {
        "build_manifest_exact_path": build_manifest_path
        == stage_d1_root / "build_manifest.json",
        "dataset_summaries_list": isinstance(summaries, list),
        "dataset_ids_unique": len(dataset_ids) == len(set(dataset_ids)),
        "dataset_scope_exact": (
            len(dataset_ids) == len(D1_SCOPE_DATASETS)
            and set(dataset_ids) == D1_SCOPE_DATASETS
        ),
        "expected_scope_declaration_exact": set(
            build_manifest.get("expected_d1_scope_datasets", [])
        )
        == D1_SCOPE_DATASETS,
        "missing_scope_empty": build_manifest.get("missing_d1_scope_datasets") == [],
        "all_dataset_bindings_valid": not failures,
    }
    return {
        "passed": all(top_level_checks.values()),
        "checks": top_level_checks,
        "failures": failures,
        "build_manifest": {
            "path": str(build_manifest_path.resolve()),
            "bytes": build_manifest_path.stat().st_size,
            "sha256": _sha256(build_manifest_path),
        },
    }


def _validate_global_stores(
    stage_d1_root: Path,
    *,
    required: bool,
) -> dict[str, Any]:
    if not required:
        return {"passed": True, "waived_for_unit_fixture": True}
    build_manifest_path = stage_d1_root / "build_manifest.json"
    if not build_manifest_path.is_file():
        return {
            "passed": False,
            "reason": "build_manifest_missing",
            "path": str(build_manifest_path),
        }
    try:
        build_manifest = json.loads(build_manifest_path.read_text(encoding="utf-8"))
        stores = build_manifest["global_stores"]
        label_meta = stores["canonical_label_store"]
        candidate_meta = stores["sealed_label_free_candidate_store"]
        label_path = stage_d1_root / label_meta["path"]
        candidate_path = stage_d1_root / candidate_meta["path"]
        labels = _load_jsonl(label_path)
        candidates = _load_jsonl(candidate_path)
    except Exception as exc:
        return {
            "passed": False,
            "reason": f"global_store_parse_failure:{type(exc).__name__}:{exc}",
        }

    label_ids = [str(record.get("record_id", "")) for record in labels]
    candidate_ids = [str(record.get("record_id", "")) for record in candidates]
    candidate_leaks = sorted(
        {path for record in candidates for path in candidate_store_label_paths(record)}
    )
    strict_failures: dict[str, str] = {}
    for record in labels:
        try:
            validate_canonical_record(record)
        except Exception as exc:
            strict_failures[str(record.get("record_id", "UNKNOWN"))] = str(exc)

    per_dataset_labels: list[dict[str, Any]] = []
    datasets_root = stage_d1_root / "datasets"
    for dataset_root in sorted(datasets_root.iterdir()):
        manifest_path = dataset_root / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        label_output = manifest["outputs"]["label_store"]
        per_dataset_labels.extend(_load_jsonl(dataset_root / label_output["path"]))
    per_dataset_labels.sort(key=lambda record: str(record["record_id"]))
    expected_candidates = [_candidate_record(record) for record in per_dataset_labels]
    expected_id_hash = hashlib.sha256(
        ("\n".join(label_ids) + ("\n" if label_ids else "")).encode("utf-8")
    ).hexdigest()
    checks = {
        "paths_distinct": label_path.resolve() != candidate_path.resolve(),
        "label_store_integrity": (
            label_path.is_file()
            and label_path.stat().st_size == label_meta["bytes"]
            and _sha256(label_path) == label_meta["sha256"]
            and len(labels) == label_meta["records"]
        ),
        "candidate_store_integrity": (
            candidate_path.is_file()
            and candidate_path.stat().st_size == candidate_meta["bytes"]
            and _sha256(candidate_path) == candidate_meta["sha256"]
            and len(candidates) == candidate_meta["records"]
        ),
        "record_ids_unique": (
            len(label_ids) == len(set(label_ids))
            and len(candidate_ids) == len(set(candidate_ids))
        ),
        "record_id_bijection": label_ids == candidate_ids,
        "record_id_hash_bound": (
            label_meta["record_ids_sha256"] == expected_id_hash
            and candidate_meta["record_ids_sha256"] == expected_id_hash
        ),
        "candidate_store_recursively_label_free": not candidate_leaks,
        "canonical_records_strictly_valid": not strict_failures,
        "global_label_store_equals_per_dataset_union": (labels == per_dataset_labels),
        "global_candidate_store_exact_canonical_projection": (
            candidates == expected_candidates
        ),
        "blocked_dataset_records_zero": (stores.get("blocked_dataset_records") == 0),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "candidate_label_leaks": candidate_leaks,
        "strict_validation_failures": strict_failures,
        "label_store": str(label_path),
        "candidate_store": str(candidate_path),
        "records": len(labels),
    }


def _load_frozen_dataset_results(
    stage_d1_root: Path,
    build_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for summary in build_manifest["datasets"]:
        dataset_id = str(summary["dataset_id"])
        binding = summary["manifest"]
        manifest_path = stage_d1_root / str(binding["path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset_root = manifest_path.parent
        outputs = manifest["outputs"]
        result = dict(manifest)
        result["paper_clean_records"] = _load_jsonl(
            dataset_root / outputs["paper_clean"]["path"]
        )
        result["label_records"] = _load_jsonl(
            dataset_root / outputs["label_store"]["path"]
        )
        result["candidate_records"] = _load_jsonl(
            dataset_root / outputs["candidate_store"]["path"]
        )
        result["rejected_records"] = _load_jsonl(
            dataset_root / outputs["rejected"]["path"]
        )
        result["auxiliary_records"] = _load_jsonl(
            dataset_root / outputs["auxiliary"]["path"]
        )
        if result.get("dataset_id") != dataset_id:
            raise ValueError(f"dataset manifest ID mismatch: {dataset_id}")
        results.append(result)
    return results


def _library_report_schema_valid(library: dict[str, Any]) -> tuple[bool, Any]:
    failures: dict[str, Any] = {}
    datasets = library.get("datasets", {})
    allowed_statuses = {"COMPUTED", "DOCUMENTED", "BLOCKED", "NOT_APPLICABLE"}
    for dataset_id in sorted(D1_SCOPE_DATASETS):
        entry = datasets.get(dataset_id)
        if not isinstance(entry, dict):
            failures[dataset_id] = ["missing_dataset_entry"]
            continue
        dataset_failures: list[str] = []
        missing_fields = set(LIBRARY_REQUIRED_FIELDS) - set(entry)
        if missing_fields:
            dataset_failures.append(
                "missing_fields:" + ",".join(sorted(missing_fields))
            )
        audits = entry.get("executed_audits")
        if not isinstance(audits, dict):
            dataset_failures.append("executed_audits_not_object")
            audits = {}
        missing_audits = set(LIBRARY_REQUIRED_AUDITS) - set(audits)
        if missing_audits:
            dataset_failures.append(
                "missing_audits:" + ",".join(sorted(missing_audits))
            )
        for name in list(LIBRARY_REQUIRED_FIELDS) + list(LIBRARY_REQUIRED_AUDITS):
            value = entry.get(name) if name in entry else audits.get(name)
            if not isinstance(value, dict):
                dataset_failures.append(f"{name}:not_object")
                continue
            status = value.get("status")
            if status not in allowed_statuses:
                dataset_failures.append(f"{name}:invalid_status:{status}")
            if status in {"BLOCKED", "NOT_APPLICABLE"} and not re.fullmatch(
                r"[A-Z][A-Z0-9_]+",
                str(value.get("reason_code", "")),
            ):
                dataset_failures.append(f"{name}:missing_stable_reason")
        if entry.get("claim_scope") != "descriptive_ascertainment_only":
            dataset_failures.append("claim_scope_not_descriptive")
        if entry.get("biological_desirability_claimed") is not False:
            dataset_failures.append("biological_desirability_claimed")
        if (
            entry.get(
                "observed_variant_frequency_interpreted_as_biological_desirability"
            )
            is not False
        ):
            dataset_failures.append("observed_frequency_misinterpreted")
        if dataset_failures:
            failures[dataset_id] = dataset_failures
    top_level = (
        library.get("schema_version") == "d1_library_ascertainment_v2"
        and library.get("required_dataset_fields") == list(LIBRARY_REQUIRED_FIELDS)
        and library.get("required_executed_audits") == list(LIBRARY_REQUIRED_AUDITS)
        and library.get("claim_scope") == "descriptive_ascertainment_only"
        and library.get("biological_desirability_claimed") is False
        and set(datasets) == D1_SCOPE_DATASETS
    )
    return top_level and not failures, failures


def _validate_required_artifacts(
    stage_d1_root: Path,
    artifact_root: Path | None,
) -> dict[str, Any]:
    relative_paths = (
        "data/data_exposure_ledger.jsonl",
        "data/library_ascertainment_report.json",
        "data/edit_script_ambiguity_report.json",
        "data/measured_action_coverage_report.json",
        "reports/data_reproduction/summary.csv",
    )
    if artifact_root is None:
        return {
            "passed": False,
            "reason": "artifact_root_not_supplied",
            "artifacts": {},
        }
    build_manifest_path = stage_d1_root / "build_manifest.json"
    try:
        build_manifest = json.loads(build_manifest_path.read_text(encoding="utf-8"))
        declared_artifacts = build_manifest["required_artifacts"]
    except Exception as exc:
        return {
            "passed": False,
            "reason": f"build_manifest_parse_failure:{type(exc).__name__}:{exc}",
            "artifacts": {},
        }
    artifacts: dict[str, Any] = {}
    binding_checks: dict[str, bool] = {}
    for relative in relative_paths:
        path = artifact_root / relative
        declared = declared_artifacts.get(relative)
        actual = {
            "path": str(path.resolve()),
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else 0,
            "sha256": _sha256(path) if path.is_file() else None,
        }
        artifacts[relative] = {
            **actual,
            "declared": declared,
        }
        binding_checks[relative] = (
            isinstance(declared, dict)
            and declared.get("path") == actual["path"]
            and declared.get("bytes") == actual["bytes"]
            and declared.get("sha256") == actual["sha256"]
            and actual["exists"]
        )
    declared_scope_exact = set(declared_artifacts) == set(relative_paths)
    if not all(item["exists"] for item in artifacts.values()):
        return {
            "passed": False,
            "reason": "required_artifact_missing",
            "binding_checks": binding_checks,
            "artifacts": artifacts,
        }
    try:
        actual_payloads: dict[str, Any] = {
            "data/data_exposure_ledger.jsonl": _load_jsonl(
                artifact_root / "data/data_exposure_ledger.jsonl"
            ),
            "data/library_ascertainment_report.json": json.loads(
                (artifact_root / "data/library_ascertainment_report.json").read_text(
                    encoding="utf-8"
                )
            ),
            "data/edit_script_ambiguity_report.json": json.loads(
                (artifact_root / "data/edit_script_ambiguity_report.json").read_text(
                    encoding="utf-8"
                )
            ),
            "data/measured_action_coverage_report.json": json.loads(
                (artifact_root / "data/measured_action_coverage_report.json").read_text(
                    encoding="utf-8"
                )
            ),
        }
        with (artifact_root / "reports/data_reproduction/summary.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            reproduction_columns = tuple(reader.fieldnames or ())
            actual_payloads["reports/data_reproduction/summary.csv"] = list(reader)
        frozen_results = _load_frozen_dataset_results(
            stage_d1_root,
            build_manifest,
        )
        expected_payloads = build_required_artifact_payloads(frozen_results)
    except Exception as exc:
        return {
            "passed": False,
            "reason": f"artifact_recompute_failure:{type(exc).__name__}:{exc}",
            "binding_checks": binding_checks,
            "artifacts": artifacts,
        }
    library = actual_payloads["data/library_ascertainment_report.json"]
    library_schema_valid, library_schema_failures = _library_report_schema_valid(
        library
    )
    content_checks = {
        relative: actual_payloads[relative] == expected_payloads[relative]
        for relative in relative_paths
    }
    ambiguity = actual_payloads["data/edit_script_ambiguity_report.json"]
    coverage = actual_payloads["data/measured_action_coverage_report.json"]
    semantic_checks = {
        "build_manifest_required_artifact_scope_exact": declared_scope_exact,
        "all_path_bytes_sha_bindings_match": all(binding_checks.values()),
        "all_artifacts_exactly_recomputed_from_frozen_stores": all(
            content_checks.values()
        ),
        "library_ascertainment_schema_strict": library_schema_valid,
        "reproduction_columns_exact": reproduction_columns == REPRODUCTION_COLUMNS,
        "ambiguity_count_scope_preserved": (
            "minimum_cost_character_alignments" in ambiguity.get("count_scope", [])
        ),
        "constructed_paths_marked_observed_zero": (
            ambiguity.get("constructed_paths_marked_observed") == 0
        ),
        "action_coverage_observed_trajectory_zero": (
            coverage.get("observed_trajectory_records") == 0
        ),
    }
    return {
        "passed": all(semantic_checks.values()),
        "reason": None if all(semantic_checks.values()) else "artifact_gate_failed",
        "semantic_checks": semantic_checks,
        "content_checks": content_checks,
        "binding_checks": binding_checks,
        "library_schema_failures": library_schema_failures,
        "build_manifest": {
            "path": str(build_manifest_path.resolve()),
            "bytes": build_manifest_path.stat().st_size,
            "sha256": _sha256(build_manifest_path),
        },
        "artifacts": artifacts,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite D1 acceptance artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        handle.write(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        )
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="root containing exact contract paths data/... and reports/...",
    )
    parser.add_argument(
        "--builder-audit-root",
        type=Path,
        help="exclusive audited-command run root for the production D1 build",
    )
    parser.add_argument("--fixture-mode", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = validate_d1_root(
        args.d1_root,
        fixture_mode=args.fixture_mode,
        artifact_root=args.artifact_root,
        builder_audit_root=args.builder_audit_root,
        require_global_stores=True,
    )
    if args.output:
        _write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if args.fixture_mode:
        return 0 if result["structural_validation_passed"] else 2
    return 0 if result["phase_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
