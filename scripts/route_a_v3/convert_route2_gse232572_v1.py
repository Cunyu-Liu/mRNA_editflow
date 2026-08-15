#!/usr/bin/env python3
"""Convert GSE232572 private rows to Evaluation-reserved Route 2 canonical v1."""

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
CONFIG_PATH = REPO_ROOT / "configs/route_a_v3_route2_gse232572_converter_v1.json"
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
    _require(config["schema_version"] == "route_a_v3_route2_gse232572_converter.v1", "unexpected schema version")
    study = config["study"]
    _require(study["study_unit_id"] == "GSE232572", "unexpected study")
    _require(study["pool_assignment"] == "EVALUATION", "GSE232572 is not Evaluation")
    _require(study["qualification_class"] == "EVALUATION_RESERVED", "qualification class changed")
    _require(study["study_role"] == "A1_ZERO_SHOT", "study role changed")
    _require(study["region"] == "3UTR", "region changed")
    _require(study["biological_context_id"] == "GSE232572|HeLa", "context changed")
    _require(study["endpoint_direction"] == "HIGHER_IS_BETTER", "endpoint direction changed")
    input_spec = config["input"]
    _require(input_spec["expected_row_count"] > 0, "expected row count must be positive")
    _require(input_spec["expected_sequence_length"] == 165, "sequence length changed")
    _require(input_spec["upstream_reject_counts"] == {"NO_UNIQUE_SEQUENCE_PAIR": 3404, "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS": 457}, "upstream rejects changed")
    output = config["output"]
    _require(output["overwrite_allowed"] is False, "successful output overwrite enabled")
    _require(output["public_redistribution_allowed"] is False, "private rows enabled for public redistribution")
    _require(output["directory"].startswith("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"), "output leaves Route 2 root")
    policy = config["evaluation_policy"]
    for key in ("training_eligible", "model_selection_eligible", "hpo_eligible", "threshold_selection_eligible", "zero_shot_result_recorded"):
        _require(policy[key] is False, f"Evaluation policy incorrectly enabled: {key}")
    credit = config["credit_policy"]
    _require(credit["measured_candidate"] is True, "measured rows lost measured status")
    _require(credit["generated_candidate"] is False, "measured rows marked generated")
    _require(not any(credit["qualified_credit_delta"].values()), "Evaluation conversion increases qualified credit")
    _require(credit["qualified_counts_after_conversion"] == {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547}, "qualified facts changed")
    _require(config["scientific_claim_status"] == "NOT_ESTABLISHED", "scientific claim overstated")


def _finite_number(value: Any, reason: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RejectRow(reason)
    number = float(value)
    if not math.isfinite(number):
        raise RejectRow(reason)
    return number


def _require_mapping(value: Any, reason: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RejectRow(reason)
    return value


def convert_row(row: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    input_spec = config["input"]
    if sorted(set(input_spec["required_fields"]) - set(row)):
        raise RejectRow("MISSING_REQUIRED_FIELD")
    record_id = row["record_id"]
    if not isinstance(record_id, str) or not record_id:
        raise RejectRow("INVALID_RECORD_ID")
    if row["data_role"] != input_spec["required_legacy_data_role"]:
        raise RejectRow("LEGACY_DATA_ROLE_MISMATCH")

    study_row = _require_mapping(row["study"], "INVALID_STUDY_METADATA")
    context = _require_mapping(row["context"], "INVALID_CONTEXT_METADATA")
    assay = _require_mapping(row["assay"], "INVALID_ASSAY_METADATA")
    endpoint = _require_mapping(row["endpoint"], "INVALID_ENDPOINT_METADATA")
    eligibility = _require_mapping(row["eligibility"], "INVALID_ELIGIBILITY_METADATA")
    source_metadata = _require_mapping(row["source"], "INVALID_SOURCE_METADATA")
    candidate_metadata = _require_mapping(row["candidate"], "INVALID_CANDIDATE_METADATA")
    expected_study = config["study"]
    if study_row.get("study_id") != expected_study["study_unit_id"]:
        raise RejectRow("STUDY_ID_MISMATCH")
    if row["region"] != expected_study["region"]:
        raise RejectRow("REGION_MISMATCH")
    if context.get("context_id") != expected_study["biological_context_id"]:
        raise RejectRow("CONTEXT_MISMATCH")
    if assay.get("assay_id") != expected_study["assay_id"] or assay.get("assay_type") != expected_study["assay_type"]:
        raise RejectRow("ASSAY_MISMATCH")
    if endpoint.get("endpoint_id") != expected_study["endpoint_id"] or endpoint.get("beneficial_direction") != expected_study["endpoint_direction"]:
        raise RejectRow("ENDPOINT_MISMATCH")
    if eligibility.get("status") != input_spec["required_legacy_eligibility_status"]:
        raise RejectRow("LEGACY_ELIGIBILITY_MISMATCH")

    source = row["source_sequence"]
    candidate = row["candidate_sequence"]
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

    edit_set = row["edit_set"]
    if not isinstance(edit_set, list) or len(edit_set) != 1 or not isinstance(edit_set[0], dict):
        raise RejectRow("INVALID_EDIT_SET")
    edit = edit_set[0]
    if (
        edit.get("coordinate_system") != "ZERO_BASED_SOURCE"
        or edit.get("position") != position
        or edit.get("ref_base") != source[position]
        or edit.get("alt_base") != candidate[position]
    ):
        raise RejectRow("EDIT_SET_SEQUENCE_MISMATCH")

    delta = _finite_number(row["delta"], "NONFINITE_DELTA")
    standard_error = row["standard_error"]
    if standard_error is not None:
        standard_error = _finite_number(standard_error, "INVALID_STANDARD_ERROR")
        if standard_error <= 0.0:
            raise RejectRow("INVALID_STANDARD_ERROR")

    source_id = row["biological_source_group_id"]
    if not isinstance(source_id, str) or not source_id:
        raise RejectRow("INVALID_BIOLOGICAL_SOURCE_GROUP")
    candidate_id = candidate_metadata.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise RejectRow("INVALID_CANDIDATE_ID")

    policy = config["evaluation_policy"]
    credit = config["credit_policy"]
    return {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "canonical_record_id": f"GSE232572:{record_id}",
        "study_unit_id": expected_study["study_unit_id"],
        "independent_study_group_id": expected_study["independent_study_group_id"],
        "publication_doi": expected_study["publication_doi"],
        "pool_assignment": expected_study["pool_assignment"],
        "qualification_class": expected_study["qualification_class"],
        "study_role": expected_study["study_role"],
        "region": expected_study["region"],
        "biological_context_id": expected_study["biological_context_id"],
        "context": context,
        "assay_id": expected_study["assay_id"],
        "assay_type": expected_study["assay_type"],
        "endpoint_id": expected_study["endpoint_id"],
        "endpoint_direction": expected_study["endpoint_direction"],
        "source_id": source_id,
        "gene_group_id": row["gene_group_id"],
        "source_metadata": source_metadata,
        "source_sequence": source,
        "candidate_id": candidate_id,
        "candidate_metadata": candidate_metadata,
        "candidate_sequence": candidate,
        "edit_operations": [{"type": "SUB", "position_zero_based": position, "ref": source[position], "alt": candidate[position]}],
        "direction_normalized_delta": delta,
        "biological_standard_error": standard_error,
        "measured_candidate": credit["measured_candidate"],
        "generated_candidate": credit["generated_candidate"],
        "evaluation_outcome_access_stage": policy["outcome_access_stage"],
        "training_eligible": policy["training_eligible"],
        "model_selection_eligible": policy["model_selection_eligible"],
        "hpo_eligible": policy["hpo_eligible"],
        "threshold_selection_eligible": policy["threshold_selection_eligible"],
        "zero_shot_result_recorded": policy["zero_shot_result_recorded"],
        "legacy_data_role": row["data_role"],
        "legacy_eligibility_status": eligibility["status"],
        "replicate_metadata": row["replicate"],
        "provenance": row["provenance"],
        "license": row["license"],
        "exposure": row["exposure"],
        "paper_faithful_transform": row["paper_faithful_transform"],
        "legacy_record_id": record_id,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def execute(config: Mapping[str, Any], input_path: Path, output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"output directory already exists: {output_dir}")
    _require(input_path.is_file(), f"input rows absent: {input_path}")
    expected = config["input"]["expected_row_count"]
    seen_ids: set[str] = set()
    input_count = 0
    converted_count = 0
    missing_se_count = 0
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
                    canonical_id = canonical["canonical_record_id"]
                    if canonical_id in seen_ids:
                        raise RejectRow("DUPLICATE_CANONICAL_RECORD_ID")
                    seen_ids.add(canonical_id)
                except json.JSONDecodeError:
                    reject_counts["INVALID_JSON"] += 1
                    continue
                except RejectRow as exc:
                    reject_counts[exc.reason] += 1
                    continue
                output_handle.write(json.dumps(canonical, sort_keys=True) + "\n")
                converted_count += 1
                missing_se_count += canonical["biological_standard_error"] is None

        rejected_count = sum(reject_counts.values())
        if input_count == expected and converted_count == expected and rejected_count == 0:
            status = "CONVERTED_EVALUATION_RESERVED"
        elif converted_count == 0:
            status = "UNCONVERTIBLE_FOR_ROUTE2_V1"
        else:
            status = "PARTIAL_EVALUATION_RESERVED"
        summary = {
            "converter_id": config["converter_id"],
            "study_unit_id": config["study"]["study_unit_id"],
            "pool_assignment": config["study"]["pool_assignment"],
            "status": status,
            "input_row_count": input_count,
            "expected_input_row_count": expected,
            "converted_record_count": converted_count,
            "rejected_record_count": rejected_count,
            "missing_standard_error_count": missing_se_count,
            "measured_candidate_count": converted_count,
            "generated_candidate_count": 0,
            "training_eligible_record_count": 0,
            "model_selection_eligible_record_count": 0,
            "zero_shot_result_recorded": False,
            "evaluation_outcome_access_stage": config["evaluation_policy"]["outcome_access_stage"],
            "qualified_credit_delta": config["credit_policy"]["qualified_credit_delta"],
            "qualified_counts_after_conversion": config["credit_policy"]["qualified_counts_after_conversion"],
            "public_redistribution_allowed": config["output"]["public_redistribution_allowed"],
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
    return 0 if summary["status"] == "CONVERTED_EVALUATION_RESERVED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
