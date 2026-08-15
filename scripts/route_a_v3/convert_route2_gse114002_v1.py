#!/usr/bin/env python3
"""Convert the public GSE114002 designed library to Development-relaxed Route 2 records."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/route_a_v3_route2_gse114002_converter_v1.json"
CANONICAL_SCHEMA_VERSION = "mrna_editflow_route2_canonical.v1"
BASES = set("ACGT")


class ConversionError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConversionError(message)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    _require(config["schema_version"] == "route_a_v3_route2_gse114002_converter.v1", "unexpected schema version")
    study = config["study"]
    _require(study["study_unit_id"] == "GSE114002", "unexpected study")
    _require(study["pool_assignment"] == "DEVELOPMENT", "GSE114002 is not Development")
    _require(study["qualification_class"] == "DEVELOPMENT_RELAXED", "qualification class changed")
    _require(study["study_role"] == "TRUE_A2_LISTWISE_DEVELOPMENT_RELAXED", "study role changed")
    _require(study["included_libraries"] == ["human_utrs", "snv"], "included libraries changed")
    _require(study["endpoint_id"] == "MEAN_RIBOSOME_LOAD", "endpoint changed")
    expected = config["input"]
    _require(expected["expected_source_row_count"] == 100017, "source row count changed")
    _require(expected["expected_eligible_pool_count"] == 959, "eligible pool count changed")
    _require(expected["expected_distinct_candidate_count"] == 3899, "candidate count changed")
    _require(expected["expected_candidate_edit_distance_counts"] == {"1": 2925, "2": 870, "3": 104}, "edit geometry changed")
    _require(expected["expected_sequence_length"] == 50, "sequence length changed")
    output = config["output"]
    _require(output["overwrite_allowed"] is False, "successful output overwrite enabled")
    _require(output["public_redistribution_allowed"] is False, "private rows enabled for public redistribution")
    _require(output["directory"].startswith("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"), "output leaves Route 2 root")
    development = config["development_policy"]
    _require(development["training_eligible"] is True and development["model_selection_eligible"] is True, "Development disabled")
    _require(development["near_duplicate_split_status"] == "NOT_RUN", "near-duplicate split overstated")
    _require(development["missing_standard_error_representation"] is None, "missing SE is not null")
    credit = config["credit_policy"]
    _require(credit["measured_candidate"] is True and credit["generated_candidate"] is False, "candidate measurement role changed")
    _require(not any(credit["qualified_credit_delta"].values()), "Development conversion increases qualified credit")
    _require(credit["qualified_counts_after_conversion"] == {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547}, "qualified facts changed")
    _require(config["scientific_claim_status"] == "NOT_ESTABLISHED", "scientific claim overstated")


def _parse_bool(value: str) -> bool | None:
    if value == "True":
        return True
    if value == "False":
        return False
    return None


def _hamming(left: str, right: str) -> int | None:
    if len(left) != len(right):
        return None
    return sum(a != b for a, b in zip(left, right))


def _finite(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_pools(config: Mapping[str, Any], input_path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], Counter[str]]:
    included_libraries = set(config["study"]["included_libraries"])
    pools: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"identity": [], "candidates": defaultdict(list)})
    counts: Counter[str] = Counter()
    with gzip.open(input_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames is not None, "source CSV has no header")
        missing = sorted(set(config["input"]["required_columns"]) - set(reader.fieldnames))
        _require(not missing, f"source CSV missing columns: {missing}")
        for row in reader:
            counts["source_rows"] += 1
            library = row["library"]
            if library not in included_libraries:
                counts["out_of_scope_library_rows"] += 1
                continue
            counts["included_library_rows"] += 1
            utr = row["utr"].upper()
            mother = row["mother"].upper()
            designed = _parse_bool(row["designed"])
            outcome = _finite(row["rl"])
            if (
                len(utr) != config["input"]["expected_sequence_length"]
                or len(mother) != config["input"]["expected_sequence_length"]
                or set(utr) - BASES
                or set(mother) - BASES
                or designed is None
                or outcome is None
            ):
                counts["invalid_schema_or_outcome_rows"] += 1
                continue
            distance = _hamming(utr, mother)
            pool = pools[(mother, library)]
            row_value = {"outcome": outcome, "legacy_id": row["id"]}
            if designed is True and utr == mother:
                pool["identity"].append(row_value)
                counts["identity_rows"] += 1
            elif designed is False and distance is not None and 1 <= distance <= 3:
                pool["candidates"][utr].append(row_value)
                counts["edited_rows"] += 1
            else:
                counts["out_of_rule_included_rows"] += 1
    return pools, counts


def _canonical_records(config: Mapping[str, Any], pools: Mapping[tuple[str, str], Mapping[str, Any]]) -> tuple[list[dict[str, Any]], Counter[int]]:
    eligible = {
        key: value
        for key, value in pools.items()
        if len(value["identity"]) == 1 and len(value["candidates"]) >= 3
    }
    records: list[dict[str, Any]] = []
    edit_distances: Counter[int] = Counter()
    study = config["study"]
    development = config["development_policy"]
    credit = config["credit_policy"]
    for pool_index, ((mother, library), pool) in enumerate(sorted(eligible.items()), start=1):
        source_id = f"GSE114002:{library}:source:{pool_index:04d}"
        source_outcome = float(pool["identity"][0]["outcome"])
        for candidate_index, (candidate, measurements) in enumerate(sorted(pool["candidates"].items()), start=1):
            candidate_outcome = statistics.fmean(float(item["outcome"]) for item in measurements)
            edit_operations = [
                {"type": "SUB", "position_zero_based": position, "ref": left, "alt": right}
                for position, (left, right) in enumerate(zip(mother, candidate))
                if left != right
            ]
            edit_distances[len(edit_operations)] += 1
            records.append({
                "schema_version": CANONICAL_SCHEMA_VERSION,
                "canonical_record_id": f"{source_id}:candidate:{candidate_index:03d}",
                "study_unit_id": study["study_unit_id"],
                "pool_assignment": study["pool_assignment"],
                "qualification_class": study["qualification_class"],
                "study_role": study["study_role"],
                "region": study["region"],
                "biological_context_id": study["biological_context_id"],
                "assay_id": study["assay_id"],
                "endpoint_id": study["endpoint_id"],
                "endpoint_direction": study["endpoint_direction"],
                "source_id": source_id,
                "source_sequence": mother,
                "source_endpoint_value": source_outcome,
                "candidate_id": f"{source_id}:candidate:{candidate_index:03d}",
                "candidate_sequence": candidate,
                "candidate_endpoint_value": candidate_outcome,
                "edit_operations": edit_operations,
                "direction_normalized_delta": candidate_outcome - source_outcome,
                "biological_standard_error": development["missing_standard_error_representation"],
                "measurement_row_count": len(measurements),
                "legacy_ids": sorted(str(item["legacy_id"]) for item in measurements),
                "library": library,
                "measured_candidate": credit["measured_candidate"],
                "generated_candidate": credit["generated_candidate"],
                "training_eligible": development["training_eligible"],
                "model_selection_eligible": development["model_selection_eligible"],
                "near_duplicate_split_status": development["near_duplicate_split_status"],
            })
    return records, edit_distances


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def execute(config: Mapping[str, Any], input_path: Path, output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"output directory already exists: {output_dir}")
    _require(input_path.is_file(), f"public source absent: {input_path}")
    pools, counts = _read_pools(config, input_path)
    records, edit_distances = _canonical_records(config, pools)
    expected = config["input"]
    eligible_pool_count = len({record["source_id"] for record in records})
    observed = {
        "source_rows": counts["source_rows"],
        "included_library_rows": counts["included_library_rows"],
        "out_of_scope_library_rows": counts["out_of_scope_library_rows"],
        "out_of_rule_included_rows": counts["out_of_rule_included_rows"] + counts["invalid_schema_or_outcome_rows"],
        "provisional_pool_count": len(pools),
        "eligible_pool_count": eligible_pool_count,
        "converted_record_count": len(records),
        "edit_distance_counts": {str(key): value for key, value in sorted(edit_distances.items())},
    }
    exact = (
        observed["source_rows"] == expected["expected_source_row_count"]
        and observed["included_library_rows"] == expected["expected_included_library_row_count"]
        and observed["out_of_scope_library_rows"] == expected["expected_out_of_scope_library_row_count"]
        and observed["out_of_rule_included_rows"] == expected["expected_out_of_rule_included_row_count"]
        and observed["provisional_pool_count"] == expected["expected_provisional_pool_count"]
        and observed["eligible_pool_count"] == expected["expected_eligible_pool_count"]
        and observed["converted_record_count"] == expected["expected_distinct_candidate_count"]
        and observed["edit_distance_counts"] == expected["expected_candidate_edit_distance_counts"]
    )
    status = "CONVERTED_DEVELOPMENT_RELAXED" if exact else "UNCONVERTIBLE_FOR_ROUTE2_V1_GEOMETRY_MISMATCH"
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=parent))
    try:
        canonical_path = temporary / config["output"]["canonical_filename"]
        with canonical_path.open("w", encoding="utf-8") as handle:
            if exact:
                for record in records:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
        summary = {
            "converter_id": config["converter_id"],
            "study_unit_id": config["study"]["study_unit_id"],
            "pool_assignment": config["study"]["pool_assignment"],
            "qualification_class": config["study"]["qualification_class"],
            "status": status,
            **observed,
            "measured_candidate_count": len(records) if exact else 0,
            "generated_candidate_count": 0,
            "missing_standard_error_count": len(records) if exact else 0,
            "qualified_credit_delta": config["credit_policy"]["qualified_credit_delta"],
            "qualified_counts_after_conversion": config["credit_policy"]["qualified_counts_after_conversion"],
            "public_redistribution_allowed": config["output"]["public_redistribution_allowed"],
            "near_duplicate_split_status": config["development_policy"]["near_duplicate_split_status"],
            "limitations": config["limitations"],
            "scientific_claim_status": config["scientific_claim_status"],
        }
        reject_summary = {
            "converter_id": config["converter_id"],
            "study_unit_id": config["study"]["study_unit_id"],
            "out_of_scope_library_row_count": observed["out_of_scope_library_rows"],
            "out_of_rule_or_invalid_included_row_count": observed["out_of_rule_included_rows"],
            "excluded_provisional_pool_count": observed["provisional_pool_count"] - observed["eligible_pool_count"],
            "reject_payload_in_summary": False,
        }
        _write_json(temporary / config["output"]["conversion_summary_filename"], summary)
        _write_json(temporary / config["output"]["reject_summary_filename"], reject_summary)
        os.rename(temporary, output_dir)
        return summary
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    input_path = args.input or Path(config["input"]["public_source_path"])
    output_dir = args.output_dir or Path(config["output"]["directory"])
    summary = execute(config, input_path, output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "CONVERTED_DEVELOPMENT_RELAXED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
