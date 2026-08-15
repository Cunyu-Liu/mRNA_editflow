from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/route_a_v3/adjudicate_route2_gse256185_v1.py"
CONFIG_PATH = REPO_ROOT / "configs/route_a_v3_route2_gse256185_adjudication_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("adjudicate_route2_gse256185_v1", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _report(config: dict) -> dict:
    expected = config["input"]
    return {
        "schema_version": expected["expected_report_schema_version"],
        "protocol_id": expected["expected_report_protocol_id"],
        "dataset_id": "GSE256185",
        "status": expected["expected_report_status"],
        "preflight_complete": True,
        "all_required_gates_pass": False,
        "qualified": False,
        "aggregate_observation": {
            "candidate_universe": {"review_candidate_row_count": expected["expected_review_candidate_row_count"]},
            "edit_replay": {
                "replay_closed_total": expected["expected_replay_closed_total"],
                "unexplained_count": expected["expected_unexplained_count"],
                "expected_edit_length_delta_counts": expected["expected_review_edit_length_delta_counts"],
            },
            "endpoint_transform": {
                "finite_endpoint_and_replicate_row_count": expected["expected_finite_endpoint_and_replicate_row_count"],
                "nonfinite_or_undefined_row_count": expected["expected_nonfinite_or_undefined_row_count"],
            },
            "eligible_after_row_preflight_exclusions": {
                "pool_count": expected["expected_retained_pool_count"],
                "parent_row_count": expected["expected_retained_parent_row_count"],
                "candidate_row_count": expected["expected_retained_candidate_row_count"],
                "row_count": expected["expected_retained_row_count"],
                "candidate_family_counts": expected["expected_retained_candidate_family_counts"],
            },
        },
    }


def test_config_freezes_zero_canonical_and_no_ins_del_expansion() -> None:
    module = _module()
    config = _config()
    module.validate_config(config)
    assert config["study"]["pool_assignment"] == "DEVELOPMENT"
    assert config["adjudication_policy"]["canonical_record_count"] == 0
    assert config["action_policy"]["ins_supported"] is False
    assert config["action_policy"]["del_supported"] is False
    assert config["development_policy"]["parser_reject_qa_eligible"] is True
    assert config["development_policy"]["training_eligible"] is False
    assert not any(config["credit_policy"]["qualified_credit_delta"].values())


def test_report_closes_every_review_action_as_nonzero_length_delta() -> None:
    module = _module()
    config = _config()
    observations = module._validate_report(config, _report(config))
    assert observations["legal_zero_length_delta_candidate_count"] == 0
    assert sum(observations["review_edit_length_delta_counts"].values()) == 7292
    assert observations["retained_candidate_row_count"] == 7288


def test_zero_delta_in_report_fails_closed() -> None:
    module = _module()
    config = _config()
    report = _report(config)
    deltas = report["aggregate_observation"]["edit_replay"]["expected_edit_length_delta_counts"]
    deltas["0"] = deltas.pop("+3")
    config["input"]["expected_review_edit_length_delta_counts"] = deltas.copy()
    with pytest.raises(module.AdjudicationError, match="zero-length candidate action appeared"):
        module._validate_report(config, report)


def test_execute_writes_empty_canonical_and_aggregate_summaries(tmp_path: Path) -> None:
    module = _module()
    config = _config()
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_report(config)), encoding="utf-8")
    output_dir = tmp_path / "output"
    summary = module.execute(config, report_path, output_dir)
    assert summary["canonical_record_count"] == 0
    assert summary["primary_reject_reason"] == "ACTION_OUTSIDE_SUB_STOP_V1"
    assert (output_dir / config["output"]["canonical_filename"]).read_text(encoding="utf-8") == ""
    rejects = json.loads((output_dir / config["output"]["reject_summary_filename"]).read_text(encoding="utf-8"))
    assert rejects["aggregate_preflight_observations"]["retained_candidate_row_count"] == 7288
    assert rejects["raw_or_row_derived_values_redistributed"] == 0


def test_execute_does_not_overwrite_existing_output(tmp_path: Path) -> None:
    module = _module()
    config = _config()
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(module.AdjudicationError, match="already exists"):
        module.execute(config, tmp_path / "missing.json", output_dir)
    assert marker.read_text(encoding="utf-8") == "keep"
