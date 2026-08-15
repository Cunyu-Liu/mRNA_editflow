#!/usr/bin/env python3
"""Convert the validated GSE149487 293T companion effects to partial Development records."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/route_a_v3_route2_gse149487_converter_v1.json"
CANONICAL_SCHEMA_VERSION = "mrna_editflow_route2_canonical.v1"


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
    _require(config["schema_version"] == "route_a_v3_route2_gse149487_converter.v1", "unexpected schema version")
    study = config["study"]
    _require(study["study_unit_id"] == "GSE149487", "unexpected study")
    _require(study["pool_assignment"] == "DEVELOPMENT", "GSE149487 is not Development")
    _require(study["qualification_class"] == "DEVELOPMENT_RELAXED", "qualification class changed")
    _require(study["study_role"] == "A1_MULTI_ENDPOINT_PARTIAL_293T", "study role changed")
    _require(study["conversion_scope"] == "PARTIAL_293T_ONLY", "partial scope overstated")
    _require(set(study["endpoint_ids"]) == {
        "te_log2_polysome_over_totalrna", "transcript_log2_totalrna_over_dna"
    }, "endpoint set changed")
    expected = config["input"]
    _require(expected["expected_input_record_count"] == 204, "input record count changed")
    _require(expected["expected_accepted_record_count"] == 192, "accepted record count changed")
    _require(expected["expected_missing_effect_or_se_count"] == 12, "missing record count changed")
    _require(expected["expected_distinct_accepted_pair_count"] == 96, "accepted pair count changed")
    _require(expected["required_biological_replicate_count"] == 3, "replicate rule changed")
    _require(expected["required_hamming_distance"] == 1, "edit rule changed")
    output = config["output"]
    _require(output["overwrite_allowed"] is False, "successful output overwrite enabled")
    _require(output["public_redistribution_allowed"] is False, "private rows enabled for public redistribution")
    _require(output["directory"].startswith("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"), "output leaves Route 2 root")
    development = config["development_policy"]
    _require(development["training_eligible"] is True and development["model_selection_eligible"] is True, "Development disabled")
    _require(development["near_duplicate_split_status"] == "NOT_RUN", "near-duplicate split overstated")
    _require(development["endpoint_or_context_increases_independent_study_count"] is False, "study count inflated")
    _require(development["barcode_is_independent_n"] is False, "barcode count inflated effective n")
    credit = config["credit_policy"]
    _require(credit["measured_candidate"] is True and credit["generated_candidate"] is False, "candidate measurement role changed")
    _require(not any(credit["qualified_credit_delta"].values()), "partial conversion increases qualified credit")
    _require(credit["qualified_counts_after_conversion"] == {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547}, "qualified facts changed")
    _require(config["scientific_claim_status"] == "NOT_ESTABLISHED", "scientific claim overstated")


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _hamming(left: str, right: str) -> int | None:
    if len(left) != len(right):
        return None
    return sum(a != b for a, b in zip(left, right))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConversionError(f"invalid JSON on line {line_number}") from exc
            _require(isinstance(value, dict), f"line {line_number} is not an object")
            rows.append(value)
    return rows


def _classify(config: Mapping[str, Any], rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str], dict[str, int]]:
    expected = config["input"]
    endpoint_counts: Counter[str] = Counter()
    accepted_endpoint_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    accepted: list[dict[str, Any]] = []
    seen_units: set[tuple[str, str]] = set()
    all_pairs: set[str] = set()
    accepted_pairs: set[str] = set()
    max_effect_recompute_error = 0.0
    max_se_recompute_error = 0.0
    for row in rows:
        endpoint = row.get("endpoint_id")
        pair_id = row.get("pair_id")
        endpoint_counts[str(endpoint)] += 1
        if isinstance(pair_id, str):
            all_pairs.add(pair_id)
        unit = (str(pair_id), str(endpoint))
        if unit in seen_units:
            rejection_counts["DUPLICATE_PAIR_ENDPOINT"] += 1
            continue
        seen_units.add(unit)
        effect = row.get("effect_delta_mutant_minus_wt")
        standard_error = row.get("standard_error")
        if effect is None or standard_error is None:
            if effect is None and standard_error is None:
                rejection_counts["MISSING_EFFECT_AND_STANDARD_ERROR"] += 1
            else:
                rejection_counts["PARTIAL_EFFECT_OR_STANDARD_ERROR"] += 1
            continue
        source = row.get("source_sequence")
        candidate = row.get("candidate_sequence")
        replicates = row.get("biological_replicate_deltas")
        valid = (
            row.get("dataset_id") == "GSE149487"
            and row.get("context") == config["study"]["biological_context_id"]
            and endpoint in config["study"]["endpoint_ids"]
            and isinstance(pair_id, str)
            and isinstance(source, str)
            and isinstance(candidate, str)
            and _hamming(source, candidate) == expected["required_hamming_distance"]
            and _finite_number(effect)
            and _finite_number(standard_error)
            and standard_error >= 0
            and isinstance(replicates, list)
            and len(replicates) == expected["required_biological_replicate_count"]
            and all(_finite_number(value) for value in replicates)
            and row.get("effective_n_unit") == "BIOLOGICAL_REPLICATE"
            and row.get("barcode_is_independent_n") is False
        )
        if not valid:
            rejection_counts["INVALID_CANONICAL_GEOMETRY_OR_MEASUREMENT"] += 1
            continue
        effect_error = abs(float(effect) - statistics.fmean(float(value) for value in replicates))
        se_error = abs(float(standard_error) - statistics.stdev(float(value) for value in replicates) / math.sqrt(3))
        max_effect_recompute_error = max(max_effect_recompute_error, effect_error)
        max_se_recompute_error = max(max_se_recompute_error, se_error)
        if effect_error > 1e-12 or se_error > 1e-12:
            rejection_counts["EFFECT_OR_STANDARD_ERROR_RECOMPUTE_MISMATCH"] += 1
            continue
        accepted.append(row)
        accepted_endpoint_counts[str(endpoint)] += 1
        accepted_pairs.add(pair_id)
    observed = {
        "input_record_count": len(rows),
        "accepted_record_count": len(accepted),
        "rejected_record_count": len(rows) - len(accepted),
        "distinct_input_pair_count": len(all_pairs),
        "distinct_accepted_pair_count": len(accepted_pairs),
        "max_effect_recompute_error": max_effect_recompute_error,
        "max_standard_error_recompute_error": max_se_recompute_error,
    }
    observed["endpoint_counts"] = dict(sorted(endpoint_counts.items()))
    observed["accepted_endpoint_counts"] = dict(sorted(accepted_endpoint_counts.items()))
    return accepted, rejection_counts, observed


def _canonical_record(config: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    edit = row["edit"]
    development = config["development_policy"]
    credit = config["credit_policy"]
    return {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "canonical_record_id": row["record_id"],
        "study_unit_id": "GSE149487",
        "pool_assignment": config["study"]["pool_assignment"],
        "qualification_class": config["study"]["qualification_class"],
        "study_role": config["study"]["study_role"],
        "conversion_scope": config["study"]["conversion_scope"],
        "region": row["region"],
        "biological_context_id": row["context"],
        "assay_id": config["study"]["assay_id"],
        "endpoint_id": row["endpoint_id"],
        "endpoint_definition": row["endpoint_definition"],
        "endpoint_direction": row["direction_convention"],
        "source_id": row["wt_construct_id"],
        "source_sequence": row["source_sequence"],
        "source_endpoint_value": None,
        "candidate_id": row["mutant_construct_id"],
        "candidate_sequence": row["candidate_sequence"],
        "candidate_endpoint_value": None,
        "edit_operations": [{
            "type": "SUB",
            "position_zero_based": edit["sequence_index_0_based"],
            "ref": edit["ref"],
            "alt": edit["alt"],
        }],
        "direction_normalized_delta": row["effect_delta_mutant_minus_wt"],
        "biological_standard_error": row["standard_error"],
        "biological_replicate_deltas": row["biological_replicate_deltas"],
        "effective_n": row["effective_n"],
        "effective_n_unit": development["effective_n_unit"],
        "barcode_is_independent_n": development["barcode_is_independent_n"],
        "pair_id": row["pair_id"],
        "group_id": row["group_id"],
        "study_group_id": row["study_group_id"],
        "gene": row["gene"],
        "measured_candidate": credit["measured_candidate"],
        "generated_candidate": credit["generated_candidate"],
        "training_eligible": development["training_eligible"],
        "model_selection_eligible": development["model_selection_eligible"],
        "near_duplicate_split_status": development["near_duplicate_split_status"],
        "analysis_method": row["analysis_method"],
        "paper_inferential_test_reproduced": row["paper_inferential_test_reproduced"],
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def execute(config: Mapping[str, Any], input_path: Path, report_path: Path, output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"output directory already exists: {output_dir}")
    _require(input_path.is_file(), f"companion input absent: {input_path}")
    _require(report_path.is_file(), f"reconstruction report absent: {report_path}")
    rows = _load_jsonl(input_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    accepted, rejection_counts, observed = _classify(config, rows)
    expected = config["input"]
    exact = (
        observed["input_record_count"] == expected["expected_input_record_count"]
        and observed["accepted_record_count"] == expected["expected_accepted_record_count"]
        and observed["rejected_record_count"] == expected["expected_missing_effect_or_se_count"]
        and observed["distinct_input_pair_count"] == expected["expected_distinct_input_pair_count"]
        and observed["distinct_accepted_pair_count"] == expected["expected_distinct_accepted_pair_count"]
        and observed["endpoint_counts"] == expected["expected_endpoint_counts"]
        and observed["accepted_endpoint_counts"] == expected["expected_accepted_endpoint_counts"]
        and rejection_counts == Counter({"MISSING_EFFECT_AND_STANDARD_ERROR": expected["expected_missing_effect_or_se_count"]})
        and report.get("summary", {}).get("rejected_endpoint_pair_count") == expected["expected_rejected_endpoint_pair_count_upstream"]
        and report.get("summary", {}).get("sequence_rejected_pair_count") == expected["expected_sequence_rejected_pair_count_upstream"]
        and report.get("reconstruction_status") == "DEVELOPMENT_RECONSTRUCTED_NOT_QUALIFIED"
        and report.get("whole_study_context_closed") is False
    )
    status = "CONVERTED_DEVELOPMENT_RELAXED_PARTIAL_293T_ONLY" if exact else "UNCONVERTIBLE_FOR_ROUTE2_V1_GEOMETRY_MISMATCH"
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=parent))
    try:
        canonical_path = temporary / config["output"]["canonical_filename"]
        with canonical_path.open("w", encoding="utf-8") as handle:
            if exact:
                for row in sorted(accepted, key=lambda item: (item["pair_id"], item["endpoint_id"])):
                    handle.write(json.dumps(_canonical_record(config, row), sort_keys=True) + "\n")
        summary = {
            "converter_id": config["converter_id"],
            "study_unit_id": "GSE149487",
            "pool_assignment": "DEVELOPMENT",
            "qualification_class": "DEVELOPMENT_RELAXED",
            "conversion_scope": "PARTIAL_293T_ONLY",
            "status": status,
            **observed,
            "measured_candidate_count": len(accepted) if exact else 0,
            "generated_candidate_count": 0,
            "qualified_credit_delta": config["credit_policy"]["qualified_credit_delta"],
            "qualified_counts_after_conversion": config["credit_policy"]["qualified_counts_after_conversion"],
            "public_redistribution_allowed": config["output"]["public_redistribution_allowed"],
            "near_duplicate_split_status": config["development_policy"]["near_duplicate_split_status"],
            "limitations": config["limitations"],
            "scientific_claim_status": config["scientific_claim_status"],
        }
        reject_summary = {
            "converter_id": config["converter_id"],
            "study_unit_id": "GSE149487",
            "conversion_rejection_counts": dict(sorted(rejection_counts.items())),
            "upstream_rejected_endpoint_pair_count": report.get("summary", {}).get("rejected_endpoint_pair_count"),
            "upstream_sequence_rejected_pair_count": report.get("summary", {}).get("sequence_rejected_pair_count"),
            "reject_payload_in_summary": False,
        }
        _write_json(temporary / config["output"]["conversion_summary_filename"], summary)
        _write_json(temporary / config["output"]["reject_summary_filename"], reject_summary)
        os.rename(temporary, output_dir)
        return summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    summary = execute(
        config,
        args.input or Path(config["input"]["development_companion_path"]),
        args.report or Path(config["input"]["reconstruction_report_path"]),
        args.output_dir or Path(config["output"]["directory"]),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"].startswith("CONVERTED_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
