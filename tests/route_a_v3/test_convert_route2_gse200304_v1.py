from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/route_a_v3/convert_route2_gse200304_v1.py"
CONFIG_PATH = REPO_ROOT / "configs/route_a_v3_route2_gse200304_converter_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("convert_route2_gse200304_v1", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(expected: int = 2) -> dict:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    value["input"]["expected_row_count"] = expected
    return value


def _row(key: str, alt: str = "C") -> dict:
    source = "A" * 201
    candidate = source[:100] + alt + source[101:]
    return {
        "record_key": key,
        "source_group": f"GSE200304|GSE200302|LOCUS|100|A|{source}",
        "source_sequence": source,
        "candidate_sequence": candidate,
        "context_vector": [0.25] * 64,
        "edit_features": [0.0] * 12,
        "direction_normalized_effect": 0.2,
        "biological_standard_error": 0.05,
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_converter_emits_explicit_route2_canonical_fields(tmp_path: Path) -> None:
    module = _module()
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"
    _write_rows(input_path, [_row("r1", "C"), _row("r2", "G")])
    summary = module.execute(_config(), input_path, output_dir)
    assert summary["status"] == "CONVERTED"
    assert summary["converted_record_count"] == 2
    assert summary["generated_candidate_count"] == 0
    assert summary["qualified_credit_delta"] == {"ordinary": 0, "a1": 0, "true_a2": 0, "canonical_records": 0}

    records = [json.loads(line) for line in (output_dir / "canonical_records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [record["canonical_record_id"] for record in records] == ["GSE200304:r1", "GSE200304:r2"]
    assert {record["pool_assignment"] for record in records} == {"DEVELOPMENT"}
    assert {record["region"] for record in records} == {"3UTR"}
    assert {record["endpoint_id"] for record in records} == {"TOTAL_POLYSOME_TRANSLATION_EFFICIENCY"}
    assert all(record["measured_candidate"] and not record["generated_candidate"] for record in records)
    assert records[0]["edit_operations"] == [{"type": "SUB", "position_zero_based": 100, "ref": "A", "alt": "C"}]


def test_converter_records_rejects_without_missing_as_zero(tmp_path: Path) -> None:
    module = _module()
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"
    invalid = _row("bad")
    invalid["biological_standard_error"] = None
    _write_rows(input_path, [_row("good"), invalid])
    summary = module.execute(_config(), input_path, output_dir)
    assert summary["status"] == "PARTIAL"
    assert summary["converted_record_count"] == 1
    assert summary["rejected_record_count"] == 1
    rejects = json.loads((output_dir / "reject_summary.json").read_text(encoding="utf-8"))
    assert rejects["route2_reject_counts"] == {"INVALID_STANDARD_ERROR": 1}
    assert rejects["route2_rejected_record_count"] == 1
    assert rejects["reject_payload_in_summary"] is False


def test_duplicate_record_and_noncentral_edit_are_rejected(tmp_path: Path) -> None:
    module = _module()
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"
    duplicate = _row("same")
    noncentral = _row("other")
    source = noncentral["source_sequence"]
    noncentral["candidate_sequence"] = "C" + source[1:]
    _write_rows(input_path, [_row("same"), duplicate, noncentral])
    config = _config(expected=3)
    summary = module.execute(config, input_path, output_dir)
    assert summary["status"] == "PARTIAL"
    rejects = json.loads((output_dir / "reject_summary.json").read_text(encoding="utf-8"))
    assert rejects["route2_reject_counts"] == {
        "DUPLICATE_CANONICAL_RECORD_ID": 1,
        "SUBSTITUTION_POSITION_MISMATCH": 1,
    }


def test_existing_output_is_not_overwritten(tmp_path: Path) -> None:
    module = _module()
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    _write_rows(input_path, [_row("r1")])
    with pytest.raises(module.ConversionError, match="already exists"):
        module.execute(_config(expected=1), input_path, output_dir)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_config_preserves_qualification_and_evaluation_boundaries() -> None:
    module = _module()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    module.validate_config(config)
    assert config["study"]["pool_assignment"] == "DEVELOPMENT"
    assert config["credit_policy"]["qualified_counts_after_conversion"] == {
        "ordinary": 1,
        "a1": 1,
        "true_a2": 0,
        "canonical_records": 6547,
    }
    assert not any(config["credit_policy"]["qualified_credit_delta"].values())
    assert config["scientific_claim_status"] == "NOT_ESTABLISHED"
