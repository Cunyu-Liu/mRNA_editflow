from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/route_a_v3/convert_route2_gse149487_v1.py"
CONFIG_PATH = REPO_ROOT / "configs/route_a_v3_route2_gse149487_converter_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("convert_route2_gse149487_v1", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _row(pair: str, endpoint: str, effect: float | None, se: float | None) -> dict:
    replicates = [0.0, 1.0, 2.0]
    return {
        "analysis_method": "ROUTE_A_COMPANION_NOT_PAPER_TEST",
        "barcode_is_independent_n": False,
        "biological_replicate_deltas": replicates,
        "candidate_sequence": "ACGT",
        "context": "293T",
        "dataset_id": "GSE149487",
        "direction_convention": "HIGHER_IS_BETTER",
        "edit": {"alt": "C", "ref": "A", "sequence_index_0_based": 1, "type": "SNV"},
        "effect_delta_mutant_minus_wt": effect,
        "effective_n": 3,
        "effective_n_unit": "BIOLOGICAL_REPLICATE",
        "endpoint_definition": endpoint,
        "endpoint_id": endpoint,
        "gene": "GENE",
        "group_id": "GROUP",
        "mutant_construct_id": f"MUT:{pair}",
        "pair_id": pair,
        "paper_inferential_test_reproduced": False,
        "record_id": f"{pair}::{endpoint}",
        "region": "5UTR",
        "source_sequence": "AAGT",
        "standard_error": se,
        "study_group_id": "PLUMAGE_LIM_2021",
        "wt_construct_id": f"WT:{pair}",
    }


def _write_input(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_report(path: Path, rejected_endpoint_pairs: int = 0, rejected_sequence_pairs: int = 0) -> None:
    path.write_text(json.dumps({
        "reconstruction_status": "DEVELOPMENT_RECONSTRUCTED_NOT_QUALIFIED",
        "whole_study_context_closed": False,
        "summary": {
            "rejected_endpoint_pair_count": rejected_endpoint_pairs,
            "sequence_rejected_pair_count": rejected_sequence_pairs,
        },
    }), encoding="utf-8")


def _small_config(rows: int, accepted: int, missing: int, input_pairs: int, accepted_pairs: int, endpoint_counts: dict, accepted_endpoint_counts: dict) -> dict:
    config = _config()
    expected = config["input"]
    expected["expected_input_record_count"] = rows
    expected["expected_accepted_record_count"] = accepted
    expected["expected_missing_effect_or_se_count"] = missing
    expected["expected_distinct_input_pair_count"] = input_pairs
    expected["expected_distinct_accepted_pair_count"] = accepted_pairs
    expected["expected_endpoint_counts"] = endpoint_counts
    expected["expected_accepted_endpoint_counts"] = accepted_endpoint_counts
    expected["expected_rejected_endpoint_pair_count_upstream"] = 0
    expected["expected_sequence_rejected_pair_count_upstream"] = 0
    return config


def test_converter_materializes_two_endpoint_records_with_recomputed_effect_and_se(tmp_path: Path) -> None:
    module = _module()
    endpoints = _config()["study"]["endpoint_ids"]
    effect = 1.0
    se = 1.0 / math.sqrt(3)
    rows = [_row("PAIR1", endpoint, effect, se) for endpoint in endpoints]
    input_path = tmp_path / "input.jsonl"
    report_path = tmp_path / "report.json"
    output_dir = tmp_path / "output"
    _write_input(input_path, rows)
    _write_report(report_path)
    counts = {endpoint: 1 for endpoint in endpoints}
    config = _small_config(2, 2, 0, 1, 1, counts, counts)
    summary = module.execute(config, input_path, report_path, output_dir)
    assert summary["status"] == "CONVERTED_DEVELOPMENT_RELAXED_PARTIAL_293T_ONLY"
    records = [json.loads(line) for line in (output_dir / "canonical_records.private.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert all(record["direction_normalized_delta"] == pytest.approx(1.0) for record in records)
    assert all(record["biological_standard_error"] == pytest.approx(se) for record in records)
    assert all(record["effective_n"] == 3 and not record["barcode_is_independent_n"] for record in records)
    assert all(record["training_eligible"] and record["model_selection_eligible"] for record in records)


def test_null_effect_and_se_are_rejected_not_zero_filled(tmp_path: Path) -> None:
    module = _module()
    endpoint = _config()["study"]["endpoint_ids"][0]
    row = _row("PAIR1", endpoint, None, None)
    input_path = tmp_path / "input.jsonl"
    report_path = tmp_path / "report.json"
    output_dir = tmp_path / "output"
    _write_input(input_path, [row])
    _write_report(report_path)
    config = _small_config(1, 0, 1, 1, 0, {endpoint: 1}, {})
    summary = module.execute(config, input_path, report_path, output_dir)
    assert summary["status"] == "CONVERTED_DEVELOPMENT_RELAXED_PARTIAL_293T_ONLY"
    assert (output_dir / "canonical_records.private.jsonl").read_text() == ""
    rejects = json.loads((output_dir / "reject_summary.json").read_text())
    assert rejects["conversion_rejection_counts"] == {"MISSING_EFFECT_AND_STANDARD_ERROR": 1}


def test_recompute_mismatch_fails_geometry_and_emits_no_canonical_record(tmp_path: Path) -> None:
    module = _module()
    endpoint = _config()["study"]["endpoint_ids"][0]
    row = _row("PAIR1", endpoint, 99.0, 1.0 / math.sqrt(3))
    input_path = tmp_path / "input.jsonl"
    report_path = tmp_path / "report.json"
    output_dir = tmp_path / "output"
    _write_input(input_path, [row])
    _write_report(report_path)
    config = _small_config(1, 1, 0, 1, 1, {endpoint: 1}, {endpoint: 1})
    summary = module.execute(config, input_path, report_path, output_dir)
    assert summary["status"] == "UNCONVERTIBLE_FOR_ROUTE2_V1_GEOMETRY_MISMATCH"
    assert (output_dir / "canonical_records.private.jsonl").read_text() == ""


def test_config_preserves_partial_scope_limitations_and_qualified_counts() -> None:
    module = _module()
    config = _config()
    module.validate_config(config)
    assert config["study"]["conversion_scope"] == "PARTIAL_293T_ONLY"
    assert "WHOLE_STUDY_CONTEXT_NOT_CLOSED" in config["limitations"]
    assert config["development_policy"]["near_duplicate_split_status"] == "NOT_RUN"
    assert not any(config["credit_policy"]["qualified_credit_delta"].values())


def test_existing_output_is_not_overwritten(tmp_path: Path) -> None:
    module = _module()
    input_path = tmp_path / "input.jsonl"
    report_path = tmp_path / "report.json"
    output_dir = tmp_path / "output"
    _write_input(input_path, [])
    _write_report(report_path)
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(module.ConversionError, match="already exists"):
        module.execute(_config(), input_path, report_path, output_dir)
    assert marker.read_text() == "keep"
