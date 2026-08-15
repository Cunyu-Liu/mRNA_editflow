from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/route_a_v3/convert_route2_gse114002_v1.py"
CONFIG_PATH = REPO_ROOT / "configs/route_a_v3_route2_gse114002_converter_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("convert_route2_gse114002_v1", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _candidate(source: str, position: int, alt: str) -> str:
    return source[:position] + alt + source[position + 1 :]


def _row(utr: str, mother: str, rl: str, row_id: str, designed: str, library: str = "human_utrs") -> dict[str, str]:
    return {"utr": utr, "mother": mother, "rl": rl, "id": row_id, "designed": designed, "library": library}


def _write_source(path: Path, rows: list[dict[str, str]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["utr", "rl", "id", "library", "mother", "designed"])
        writer.writeheader()
        writer.writerows(rows)


def _small_config(source_rows: int, included: int, out_scope: int, out_rule: int, pools: int, eligible: int, candidates: int, distances: dict[str, int]) -> dict:
    config = _config()
    expected = config["input"]
    expected["expected_source_row_count"] = source_rows
    expected["expected_included_library_row_count"] = included
    expected["expected_out_of_scope_library_row_count"] = out_scope
    expected["expected_out_of_rule_included_row_count"] = out_rule
    expected["expected_provisional_pool_count"] = pools
    expected["expected_eligible_pool_count"] = eligible
    expected["expected_distinct_candidate_count"] = candidates
    expected["expected_candidate_edit_distance_counts"] = distances
    return config


def test_converter_builds_measured_candidate_deltas_without_row_expansion(tmp_path: Path) -> None:
    module = _module()
    source = "A" * 50
    rows = [
        _row(source, source, "5.0", "anchor", "True"),
        _row(_candidate(source, 1, "C"), source, "5.5", "c1", "False"),
        _row(_candidate(source, 2, "G"), source, "4.5", "c2", "False"),
        _row(_candidate(source, 3, "T"), source, "6.0", "c3", "False"),
    ]
    input_path = tmp_path / "source.csv.gz"
    output_dir = tmp_path / "output"
    _write_source(input_path, rows)
    config = _small_config(4, 4, 0, 0, 1, 1, 3, {"1": 3})
    summary = module.execute(config, input_path, output_dir)
    assert summary["status"] == "CONVERTED_DEVELOPMENT_RELAXED"
    assert summary["eligible_pool_count"] == 1
    assert summary["converted_record_count"] == 3
    records = [json.loads(line) for line in (output_dir / "canonical_records.private.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sorted(record["direction_normalized_delta"] for record in records) == [-0.5, 0.5, 1.0]
    assert all(record["biological_standard_error"] is None for record in records)
    assert all(record["measured_candidate"] and not record["generated_candidate"] for record in records)
    assert all(record["measurement_row_count"] == 1 for record in records)


def test_duplicate_candidate_rows_are_collapsed_not_expanded(tmp_path: Path) -> None:
    module = _module()
    source = "A" * 50
    candidate = _candidate(source, 1, "C")
    rows = [
        _row(source, source, "5.0", "anchor", "True"),
        _row(candidate, source, "5.2", "c1a", "False"),
        _row(candidate, source, "5.4", "c1b", "False"),
        _row(_candidate(source, 2, "G"), source, "4.5", "c2", "False"),
        _row(_candidate(source, 3, "T"), source, "6.0", "c3", "False"),
    ]
    input_path = tmp_path / "source.csv.gz"
    output_dir = tmp_path / "output"
    _write_source(input_path, rows)
    config = _small_config(5, 5, 0, 0, 1, 1, 3, {"1": 3})
    summary = module.execute(config, input_path, output_dir)
    assert summary["converted_record_count"] == 3
    records = [json.loads(line) for line in (output_dir / "canonical_records.private.jsonl").read_text(encoding="utf-8").splitlines()]
    aggregated = next(record for record in records if record["candidate_sequence"] == candidate)
    assert aggregated["measurement_row_count"] == 2
    assert aggregated["candidate_endpoint_value"] == pytest.approx(5.3)


def test_pool_without_unique_anchor_or_three_candidates_is_excluded(tmp_path: Path) -> None:
    module = _module()
    source = "A" * 50
    rows = [
        _row(source, source, "5.0", "anchor1", "True"),
        _row(source, source, "5.1", "anchor2", "True"),
        _row(_candidate(source, 1, "C"), source, "5.5", "c1", "False"),
        _row(_candidate(source, 2, "G"), source, "4.5", "c2", "False"),
    ]
    input_path = tmp_path / "source.csv.gz"
    output_dir = tmp_path / "output"
    _write_source(input_path, rows)
    config = _small_config(4, 4, 0, 0, 1, 0, 0, {})
    summary = module.execute(config, input_path, output_dir)
    assert summary["status"] == "CONVERTED_DEVELOPMENT_RELAXED"
    assert summary["converted_record_count"] == 0
    assert (output_dir / "canonical_records.private.jsonl").read_text(encoding="utf-8") == ""


def test_nonfinite_outcome_is_rejected_not_zero_filled(tmp_path: Path) -> None:
    module = _module()
    source = "A" * 50
    rows = [
        _row(source, source, "5.0", "anchor", "True"),
        _row(_candidate(source, 1, "C"), source, "nan", "bad", "False"),
    ]
    input_path = tmp_path / "source.csv.gz"
    output_dir = tmp_path / "output"
    _write_source(input_path, rows)
    config = _small_config(2, 2, 0, 1, 1, 0, 0, {})
    summary = module.execute(config, input_path, output_dir)
    assert summary["converted_record_count"] == 0
    rejects = json.loads((output_dir / "reject_summary.json").read_text(encoding="utf-8"))
    assert rejects["out_of_rule_or_invalid_included_row_count"] == 1


def test_config_preserves_relaxed_status_limitations_and_no_overwrite(tmp_path: Path) -> None:
    module = _module()
    config = _config()
    module.validate_config(config)
    assert config["study"]["qualification_class"] == "DEVELOPMENT_RELAXED"
    assert config["development_policy"]["near_duplicate_split_status"] == "NOT_RUN"
    assert "BIOLOGICAL_STANDARD_ERROR_ABSENT_BY_DESIGN" in config["limitations"]
    assert not any(config["credit_policy"]["qualified_credit_delta"].values())
    input_path = tmp_path / "source.csv.gz"
    _write_source(input_path, [])
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(module.ConversionError, match="already exists"):
        module.execute(config, input_path, output_dir)
    assert marker.read_text(encoding="utf-8") == "keep"
