"""Outcome-isolated Development projections for Route 2 V3 models.

The canonical JSONL files contain TRAIN, VALIDATION, and protected TEST rows.
This module resolves a record's frozen split from the outcome-free manifest
before decoding the complete canonical row.  New V3 training code consumes the
resulting TRAIN/VALIDATION projections and never opens canonical outcome files.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECTION_SCHEMA_VERSION = "route_a_v3_route2_development_projection.v3"
SUMMARY_SCHEMA_VERSION = "route_a_v3_route2_development_projection_summary.v3"
ENDPOINT_SCHEMA_VERSION = "route_a_v3_route2_endpoint_descriptors.v1"
MODELING_SPLITS = ("TRAIN", "VALIDATION")
PROTECTED_SPLIT = "TEST"
RNA_ALPHABET = frozenset("ACGU")


class DevelopmentProjectionError(RuntimeError):
    """A projection would violate the frozen Development boundary."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DevelopmentProjectionError(message)


_RECORD_ID_PATTERN = re.compile(
    r'"canonical_record_id"\s*:\s*(?P<value>"(?:\\.|[^"\\])*")'
)


def extract_canonical_record_id(raw_line: str) -> str:
    """Extract only the JSON string record id without decoding the row.

    Canonical files are emitted by project-owned JSON serializers, so a narrow
    JSON-string-field extractor is sufficient and lets protected rows remain
    otherwise unparsed.
    """

    match = _RECORD_ID_PATTERN.search(raw_line)
    _require(match is not None, "canonical JSONL line lacks canonical_record_id")
    value = json.loads(match.group("value"))
    _require(isinstance(value, str) and value, "canonical_record_id is empty")
    return value


def load_development_manifest(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            _require(
                row.get("pool_assignment") == "DEVELOPMENT",
                "non-Development record entered the frozen manifest",
            )
            record_id = str(row["canonical_record_id"])
            split = str(row["split"])
            _require(split in {*MODELING_SPLITS, PROTECTED_SPLIT}, "unknown Development split")
            _require(record_id not in result, f"manifest record is duplicated: {record_id}")
            result[record_id] = {
                "split": split,
                "study_unit_id": str(row["study_unit_id"]),
                "connected_source_component_id": str(
                    row["connected_source_component_id"]
                ),
            }
    _require(result, "Development manifest is empty")
    return result


def load_endpoint_descriptors(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(
        payload.get("schema_version") == ENDPOINT_SCHEMA_VERSION,
        "unexpected endpoint descriptor schema",
    )
    result: dict[str, dict[str, Any]] = {}
    required = {
        "endpoint_id",
        "quantity_family",
        "measurement_form",
        "numerator_family",
        "denominator_family",
    }
    for descriptor in payload.get("endpoints", []):
        _require(isinstance(descriptor, dict), "endpoint descriptor must be an object")
        _require(required <= set(descriptor), "endpoint descriptor fields are incomplete")
        endpoint_id = str(descriptor["endpoint_id"])
        _require(endpoint_id and endpoint_id not in result, "endpoint descriptor id is empty or duplicated")
        _require(
            not any("outcome" in key.lower() or "target" in key.lower() for key in descriptor),
            "endpoint descriptor contains an outcome-like field",
        )
        result[endpoint_id] = {key: descriptor[key] for key in sorted(descriptor)}
    _require(result, "endpoint descriptor registry is empty")
    return result


def _normalize_sequence(value: Any) -> str:
    sequence = str(value).upper().replace("T", "U")
    _require(sequence and set(sequence) <= RNA_ALPHABET, "sequence is outside the RNA alphabet")
    return sequence


def _normalize_region(value: Any) -> tuple[str, int]:
    region = str(value).replace("′", "").replace("'", "")
    _require(region in {"5UTR", "3UTR"}, f"unsupported region: {value}")
    return region, 0 if region == "5UTR" else 1


def _finite_target(value: Any, record_id: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"invalid target: {record_id}",
    )
    return float(value)


def project_canonical_row(
    row: Mapping[str, Any],
    manifest_row: Mapping[str, str],
    endpoint_descriptors: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Create one label-bearing TRAIN/VALIDATION row from a decoded row."""

    record_id = str(row["canonical_record_id"])
    split = str(manifest_row["split"])
    _require(split in MODELING_SPLITS, "protected split reached project_canonical_row")
    _require(row.get("pool_assignment") == "DEVELOPMENT", "non-Development canonical row entered projection")
    study = str(row["study_unit_id"])
    _require(study == manifest_row["study_unit_id"], "canonical study differs from manifest")
    source = _normalize_sequence(row["source_sequence"])
    candidate = _normalize_sequence(row["candidate_sequence"])
    _require(len(source) == len(candidate), f"length-changing row entered SUB projection: {record_id}")
    region, region_id = _normalize_region(row["region"])
    endpoint_id = str(row["endpoint_id"])
    _require(endpoint_id in endpoint_descriptors, f"endpoint descriptor is missing: {endpoint_id}")
    context = str(row["biological_context_id"])
    source_group = "::".join((study, str(row["source_id"]), context, endpoint_id))
    edits = [
        {
            "position": index,
            "source_base": left,
            "candidate_base": right,
        }
        for index, (left, right) in enumerate(zip(source, candidate))
        if left != right
    ]
    return {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "canonical_record_id": record_id,
        "split": split,
        "study_unit_id": study,
        "connected_source_component_id": manifest_row[
            "connected_source_component_id"
        ],
        "source_group_id": source_group,
        "source_id": str(row["source_id"]),
        "task_id": f"{endpoint_id}::region={region_id}",
        "endpoint_id": endpoint_id,
        "endpoint_descriptor": dict(endpoint_descriptors[endpoint_id]),
        "region": region,
        "region_id": region_id,
        "assay_id": str(row["assay_id"]),
        "biological_context_id": context,
        "source_sequence": source,
        "candidate_sequence": candidate,
        "source_relative_edits": edits,
        "direction_normalized_delta": _finite_target(
            row["direction_normalized_delta"], record_id
        ),
    }


def build_development_projection(
    *,
    manifest_path: Path,
    canonical_paths: Iterable[Path],
    endpoint_descriptor_path: Path,
    output_directory: Path,
    included_splits: Sequence[str] = MODELING_SPLITS,
) -> dict[str, Any]:
    """Build outcome-bearing TRAIN/VALIDATION projections only.

    TEST is deliberately unsupported here.  A future one-shot TEST adjudicator
    must implement its own authorization and atomic scoring path.
    """

    requested_splits = tuple(str(value) for value in included_splits)
    _require(
        requested_splits == MODELING_SPLITS,
        "Development projection is fixed to TRAIN and VALIDATION; TEST requires the one-shot adjudicator",
    )
    _require(not output_directory.exists(), f"projection output already exists: {output_directory}")
    manifest = load_development_manifest(manifest_path)
    descriptors = load_endpoint_descriptors(endpoint_descriptor_path)
    manifest_counts = Counter(value["split"] for value in manifest.values())
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.", dir=output_directory.parent
        )
    )
    handles: dict[str, Any] = {}
    seen: set[str] = set()
    written_counts: Counter[str] = Counter()
    decoded_counts: Counter[str] = Counter()
    endpoint_counts: Counter[str] = Counter()
    try:
        for split in MODELING_SPLITS:
            handles[split] = (temporary / f"{split.lower()}.jsonl").open(
                "w", encoding="utf-8"
            )
        for canonical_path in canonical_paths:
            with canonical_path.open(encoding="utf-8") as input_handle:
                for raw_line in input_handle:
                    record_id = extract_canonical_record_id(raw_line)
                    if record_id not in manifest:
                        continue
                    _require(record_id not in seen, f"canonical record is duplicated: {record_id}")
                    seen.add(record_id)
                    split = manifest[record_id]["split"]
                    if split == PROTECTED_SPLIT:
                        continue
                    row = json.loads(raw_line)
                    decoded_counts[split] += 1
                    projected = project_canonical_row(
                        row, manifest[record_id], descriptors
                    )
                    handles[split].write(
                        json.dumps(projected, sort_keys=True) + "\n"
                    )
                    written_counts[split] += 1
                    endpoint_counts[projected["endpoint_id"]] += 1
        _require(seen == set(manifest), "canonical inputs do not exactly cover the Development manifest")
        for split in MODELING_SPLITS:
            _require(
                written_counts[split] == manifest_counts[split],
                f"projection count differs from manifest for {split}",
            )
        _require(decoded_counts[PROTECTED_SPLIT] == 0, "TEST row was fully decoded")
        summary = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": "TRAIN_VALIDATION_PROJECTION_COMPLETE_TEST_UNPARSED",
            "manifest_path": str(manifest_path),
            "endpoint_descriptor_path": str(endpoint_descriptor_path),
            "included_splits": list(MODELING_SPLITS),
            "manifest_record_counts": dict(sorted(manifest_counts.items())),
            "projection_record_counts": dict(sorted(written_counts.items())),
            "canonical_full_decode_counts": {
                "TRAIN": decoded_counts["TRAIN"],
                "VALIDATION": decoded_counts["VALIDATION"],
                "TEST": 0,
            },
            "endpoint_record_counts": dict(sorted(endpoint_counts.items())),
            "development_test_record_count_withheld": manifest_counts[PROTECTED_SPLIT],
            "development_test_outcomes_accessed": False,
            "evaluation_outcomes_accessed": False,
            "projection_files": {
                split: f"{split.lower()}.jsonl" for split in MODELING_SPLITS
            },
        }
        (temporary / "projection_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for handle in handles.values():
            handle.close()
        handles.clear()
        temporary.rename(output_directory)
        return summary
    except Exception:
        for handle in handles.values():
            handle.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_projection_rows(
    paths: Iterable[Path], *, allowed_splits: Sequence[str] = MODELING_SPLITS
) -> list[dict[str, Any]]:
    """Load a V3 projection without any canonical-file fallback."""

    allowed = {str(value) for value in allowed_splits}
    _require(allowed and allowed <= set(MODELING_SPLITS), "projection loader may only read TRAIN/VALIDATION")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                _require(row.get("schema_version") == PROJECTION_SCHEMA_VERSION, "unexpected projection schema")
                _require(row.get("split") in allowed, "projection row is outside the authorized modeling splits")
                record_id = str(row["canonical_record_id"])
                _require(record_id not in seen, f"projection record is duplicated: {record_id}")
                seen.add(record_id)
                rows.append(row)
    _require(rows, "projection is empty")
    return rows
