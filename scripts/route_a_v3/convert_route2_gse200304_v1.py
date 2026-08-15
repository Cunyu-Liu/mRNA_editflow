#!/usr/bin/env python3
"""Convert the accepted GSE200304 Development rows to Route 2 canonical v1."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/route_a_v3_route2_gse200304_converter_v1.json"
CANONICAL_SCHEMA_VERSION = "mrna_editflow_route2_canonical.v1"
BASES = set("ACGT")


class ConversionError(RuntimeError):
    pass


class RejectRow(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConversionError(message)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    _require(config["schema_version"] == "route_a_v3_route2_gse200304_converter.v1", "unexpected schema version")
    study = config["study"]
    _require(study["study_unit_id"] == "GSE200304", "unexpected study")
    _require(study["pool_assignment"] == "DEVELOPMENT", "GSE200304 is not Development")
    _require(study["qualification_class"] == "QUALIFIED_CURRENT", "qualification class changed")
    _require(study["region"] == "3UTR", "region changed")
    _require(study["endpoint_id"] == "TOTAL_POLYSOME_TRANSLATION_EFFICIENCY", "endpoint changed")
    input_spec = config["input"]
    _require(input_spec["expected_row_count"] > 0, "expected row count must be positive")
    _require(input_spec["expected_sequence_length"] == 201, "sequence length changed")
    _require(input_spec["expected_edit_position_zero_based"] == 100, "edit position changed")
    _require(input_spec["upstream_reject_counts"] == {"s2_not_in_s3": 338, "s3_na_or_nonfinite": 225, "nonpositive_se": 0}, "upstream rejects changed")
    output = config["output"]
    _require(output["overwrite_allowed"] is False, "successful output overwrite enabled")
    _require(output["directory"].startswith("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"), "output leaves Route 2 root")
    credit = config["credit_policy"]
    _require(credit["measured_candidate"] is True, "measured rows lost measured status")
    _require(credit["generated_candidate"] is False, "measured rows marked generated")
    _require(not any(credit["qualified_credit_delta"].values()), "replay increases qualified credit")
    _require(credit["qualified_counts_after_conversion"] == {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547}, "qualified facts changed")
    _require(config["scientific_claim_status"] == "NOT_ESTABLISHED", "scientific claim overstated")


def _finite_number(value: Any, reason: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RejectRow(reason)
    number = float(value)
    if not math.isfinite(number):
        raise RejectRow(reason)
    return number


def _finite_vector(value: Any, length: int, reason: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise RejectRow(reason)
    return [_finite_number(item, reason) for item in value]


def convert_row(row: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    input_spec = config["input"]
    missing = sorted(set(input_spec["required_fields"]) - set(row))
    if missing:
        raise RejectRow("MISSING_REQUIRED_FIELD")

    record_key = row["record_key"]
    source_group = row["source_group"]
    source = row["source_sequence"]
    candidate = row["candidate_sequence"]
    if not isinstance(record_key, str) or not record_key:
        raise RejectRow("INVALID_RECORD_KEY")
    if not isinstance(source_group, str) or not source_group.startswith("GSE200304|GSE200302|"):
        raise RejectRow("INVALID_SOURCE_GROUP")
    expected_length = input_spec["expected_sequence_length"]
    if not isinstance(source, str) or not isinstance(candidate, str):
        raise RejectRow("INVALID_SEQUENCE_TYPE")
    if len(source) != expected_length or len(candidate) != expected_length:
        raise RejectRow("INVALID_SEQUENCE_LENGTH")
    if set(source) - BASES or set(candidate) - BASES:
        raise RejectRow("INVALID_SEQUENCE_ALPHABET")
    differences = [index for index, pair in enumerate(zip(source, candidate)) if pair[0] != pair[1]]
    if len(differences) != 1:
        raise RejectRow("NOT_EXACTLY_ONE_SUBSTITUTION")
    position = differences[0]
    if position != input_spec["expected_edit_position_zero_based"]:
        raise RejectRow("SUBSTITUTION_POSITION_MISMATCH")

    effect = _finite_number(row["direction_normalized_effect"], "NONFINITE_EFFECT")
    standard_error = _finite_number(row["biological_standard_error"], "INVALID_STANDARD_ERROR")
    if standard_error <= 0.0:
        raise RejectRow("INVALID_STANDARD_ERROR")
    context_vector = _finite_vector(row["context_vector"], 64, "INVALID_CONTEXT_VECTOR")
    edit_features = _finite_vector(row["edit_features"], 12, "INVALID_EDIT_FEATURES")

    study = config["study"]
    credit = config["credit_policy"]
    return {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "canonical_record_id": f"GSE200304:{record_key}",
        "study_unit_id": study["study_unit_id"],
        "member_accessions": study["member_accessions"],
        "pool_assignment": study["pool_assignment"],
        "qualification_class": study["qualification_class"],
        "study_role": study["study_role"],
        "region": study["region"],
        "biological_context_id": study["biological_context_id"],
        "assay_id": study["assay_id"],
        "endpoint_id": study["endpoint_id"],
        "endpoint_direction": study["endpoint_direction"],
        "source_id": source_group,
        "source_sequence": source,
        "candidate_id": record_key,
        "candidate_sequence": candidate,
        "edit_operations": [{"type": "SUB", "position_zero_based": position, "ref": source[position], "alt": candidate[position]}],
        "direction_normalized_delta": effect,
        "biological_standard_error": standard_error,
        "context_vector": context_vector,
        "edit_features": edit_features,
        "measured_candidate": credit["measured_candidate"],
        "generated_candidate": credit["generated_candidate"],
        "legacy_record_key": record_key,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def execute(config: Mapping[str, Any], input_path: Path, output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"output directory already exists: {output_dir}")
    _require(input_path.is_file(), f"input rows absent: {input_path}")

    expected = config["input"]["expected_row_count"]
    seen_record_ids: set[str] = set()
    input_count = 0
    converted_count = 0
    reject_counts: Counter[str] = Counter()
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=parent))
    try:
        canonical_path = temporary / config["output"]["canonical_filename"]
        with input_path.open(encoding="utf-8") as source_handle, canonical_path.open("w", encoding="utf-8") as output_handle:
            for input_count, line in enumerate(source_handle, start=1):
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, dict):
                        raise RejectRow("ROW_IS_NOT_OBJECT")
                    canonical = convert_row(raw, config)
                    record_id = canonical["canonical_record_id"]
                    if record_id in seen_record_ids:
                        raise RejectRow("DUPLICATE_CANONICAL_RECORD_ID")
                    seen_record_ids.add(record_id)
                except json.JSONDecodeError:
                    reject_counts["INVALID_JSON"] += 1
                    continue
                except RejectRow as exc:
                    reject_counts[exc.reason] += 1
                    continue
                output_handle.write(json.dumps(canonical, sort_keys=True) + "\n")
                converted_count += 1

        rejected_count = sum(reject_counts.values())
        if input_count == expected and converted_count == expected and rejected_count == 0:
            status = "CONVERTED"
        elif converted_count == 0:
            status = "UNCONVERTIBLE_FOR_ROUTE2_V1"
        else:
            status = "PARTIAL"
        summary = {
            "converter_id": config["converter_id"],
            "study_unit_id": config["study"]["study_unit_id"],
            "pool_assignment": config["study"]["pool_assignment"],
            "status": status,
            "input_row_count": input_count,
            "expected_input_row_count": expected,
            "converted_record_count": converted_count,
            "rejected_record_count": rejected_count,
            "measured_candidate_count": converted_count,
            "generated_candidate_count": 0,
            "qualified_credit_delta": config["credit_policy"]["qualified_credit_delta"],
            "qualified_counts_after_conversion": config["credit_policy"]["qualified_counts_after_conversion"],
            "scientific_claim_status": config["scientific_claim_status"],
        }
        reject_summary = {
            "converter_id": config["converter_id"],
            "study_unit_id": config["study"]["study_unit_id"],
            "route2_reject_counts": dict(sorted(reject_counts.items())),
            "route2_rejected_record_count": rejected_count,
            "upstream_reject_counts": config["input"]["upstream_reject_counts"],
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
    input_path = args.input or Path(config["input"]["legacy_rows_path"])
    output_dir = args.output_dir or Path(config["output"]["directory"])
    summary = execute(config, input_path, output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "CONVERTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
