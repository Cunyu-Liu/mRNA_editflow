#!/usr/bin/env python3
"""Close GSE256185 as an action-incompatible Route 2 V1 negative control."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/route_a_v3_route2_gse256185_adjudication_v1.json"


class AdjudicationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdjudicationError(message)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    _require(config["schema_version"] == "route_a_v3_route2_gse256185_adjudication.v1", "unexpected schema version")
    study = config["study"]
    _require(study["study_unit_id"] == "GSE256185", "unexpected study")
    _require(study["pool_assignment"] == "DEVELOPMENT", "study left Development inventory")
    _require(study["qualification_class"] == "AGGREGATE_ONLY_UNCONVERTIBLE_V1", "qualification class changed")
    _require(study["study_role"] == "NEGATIVE_CONTROL_POOL_PARSER_REJECT_QA_WITHIN_DEVELOPMENT_INVENTORY", "study role changed")
    _require(study["conversion_scope"] == "SUB_STOP_V1_ACTION_COMPATIBILITY_ADJUDICATION_ONLY", "scope changed")

    action = config["action_policy"]
    _require(action["allowed_candidate_actions"] == ["SUB", "STOP"], "V1 action set changed")
    _require(action["ins_supported"] is False and action["del_supported"] is False, "INS or DEL was enabled")
    _require(action["zero_length_delta_required_for_sub"] is True, "SUB geometry changed")

    decision = config["adjudication_policy"]
    _require(decision["canonical_record_count"] == 0, "canonical records were enabled")
    _require(decision["legal_sub_candidate_count"] == 0, "legal SUB candidates were asserted")
    _require(decision["retained_preflight_candidates_rejected_for_action_space"] == 7288, "reject count changed")
    _require(decision["primary_reject_reason"] == "ACTION_OUTSIDE_SUB_STOP_V1", "primary reject reason changed")
    _require(decision["qualification_blockers_are_not_reinterpreted_as_action_compatibility"] is True, "qualification and action compatibility were conflated")
    _require(decision["future_ins_del_expansion_authorized"] is False, "future action expansion was authorized")

    development = config["development_policy"]
    _require(development == {
        "training_eligible": False,
        "model_selection_eligible": False,
        "confirmatory_evaluation_eligible": False,
        "parser_reject_qa_eligible": True,
    }, "Development use policy changed")
    credit = config["credit_policy"]
    _require(not any(credit["qualified_credit_delta"].values()), "adjudication increases qualified credit")
    _require(credit["qualified_counts_after_adjudication"] == {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547}, "qualified facts changed")
    output = config["output"]
    _require(output["overwrite_allowed"] is False and output["public_redistribution_allowed"] is False, "output policy changed")
    _require(output["directory"].startswith("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"), "output leaves Route 2 root")
    _require(config["scientific_claim_status"] == "NOT_ESTABLISHED", "scientific claim overstated")


def _validate_report(config: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    expected = config["input"]
    _require(report["schema_version"] == expected["expected_report_schema_version"], "aggregate report schema changed")
    _require(report["protocol_id"] == expected["expected_report_protocol_id"], "aggregate report protocol changed")
    _require(report["dataset_id"] == "GSE256185", "aggregate report study changed")
    _require(report["status"] == expected["expected_report_status"], "aggregate report status changed")
    _require(report["preflight_complete"] is True, "aggregate preflight is incomplete")
    _require(report["all_required_gates_pass"] is False and report["qualified"] is False, "aggregate preflight qualification truth changed")

    aggregate = report["aggregate_observation"]
    candidate = aggregate["candidate_universe"]
    replay = aggregate["edit_replay"]
    endpoint = aggregate["endpoint_transform"]
    retained = aggregate["eligible_after_row_preflight_exclusions"]
    _require(candidate["review_candidate_row_count"] == expected["expected_review_candidate_row_count"], "review candidate count changed")
    _require(replay["replay_closed_total"] == expected["expected_replay_closed_total"], "replay-closed count changed")
    _require(replay["unexplained_count"] == expected["expected_unexplained_count"], "unexplained count changed")
    _require(endpoint["finite_endpoint_and_replicate_row_count"] == expected["expected_finite_endpoint_and_replicate_row_count"], "finite endpoint count changed")
    _require(endpoint["nonfinite_or_undefined_row_count"] == expected["expected_nonfinite_or_undefined_row_count"], "nonfinite endpoint count changed")
    _require(retained["pool_count"] == expected["expected_retained_pool_count"], "retained pool count changed")
    _require(retained["parent_row_count"] == expected["expected_retained_parent_row_count"], "retained parent count changed")
    _require(retained["candidate_row_count"] == expected["expected_retained_candidate_row_count"], "retained candidate count changed")
    _require(retained["row_count"] == expected["expected_retained_row_count"], "retained row count changed")
    _require(retained["candidate_family_counts"] == expected["expected_retained_candidate_family_counts"], "retained candidate families changed")

    length_deltas = replay["expected_edit_length_delta_counts"]
    _require(length_deltas == expected["expected_review_edit_length_delta_counts"], "review length-delta distribution changed")
    _require(sum(length_deltas.values()) == candidate["review_candidate_row_count"], "length-delta distribution does not close")
    _require(all(int(delta) != 0 for delta in length_deltas), "zero-length candidate action appeared")
    _require(sum(retained["candidate_family_counts"].values()) == retained["candidate_row_count"], "retained candidate families do not close")
    return {
        "review_candidate_row_count": candidate["review_candidate_row_count"],
        "replay_closed_total": replay["replay_closed_total"],
        "unexplained_count": replay["unexplained_count"],
        "retained_pool_count": retained["pool_count"],
        "retained_parent_row_count": retained["parent_row_count"],
        "retained_candidate_row_count": retained["candidate_row_count"],
        "retained_row_count": retained["row_count"],
        "retained_candidate_family_counts": retained["candidate_family_counts"],
        "review_edit_length_delta_counts": length_deltas,
        "legal_zero_length_delta_candidate_count": 0,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def execute(config: Mapping[str, Any], report_path: Path, output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"output directory already exists: {output_dir}")
    _require(report_path.is_file(), f"aggregate preflight report absent: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    observations = _validate_report(config, report)
    decision = config["adjudication_policy"]
    _require(observations["retained_candidate_row_count"] == decision["retained_preflight_candidates_rejected_for_action_space"], "action-space reject count does not close")

    output = config["output"]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        (temporary / output["canonical_filename"]).write_text("", encoding="utf-8")
        summary = {
            "status": "CLOSED_GSE256185_ZERO_CANONICAL_ACTION_OUTSIDE_SUB_STOP_V1",
            "study_unit_id": "GSE256185",
            "pool_assignment": config["study"]["pool_assignment"],
            "study_role": config["study"]["study_role"],
            "canonical_record_count": 0,
            "legal_sub_candidate_count": 0,
            "retained_preflight_candidates_rejected_for_action_space": observations["retained_candidate_row_count"],
            "primary_reject_reason": decision["primary_reject_reason"],
            "training_eligible": False,
            "model_selection_eligible": False,
            "confirmatory_evaluation_eligible": False,
            "future_ins_del_expansion_authorized": False,
            "qualified_credit_delta": config["credit_policy"]["qualified_credit_delta"],
            "qualified_counts_after_adjudication": config["credit_policy"]["qualified_counts_after_adjudication"],
            "scientific_claim_status": config["scientific_claim_status"],
            "limitations": config["limitations"],
        }
        rejects = {
            "aggregate_preflight_observations": observations,
            "action_policy": config["action_policy"],
            "primary_reject_reason": decision["primary_reject_reason"],
            "canonical_records_materialized": 0,
            "raw_or_row_derived_values_redistributed": 0,
        }
        _write_json(temporary / output["conversion_summary_filename"], summary)
        _write_json(temporary / output["reject_summary_filename"], rejects)
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Adjudicate GSE256185 against the Route 2 V1 action space")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--aggregate-report", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    summary = execute(
        config,
        args.aggregate_report or Path(config["input"]["aggregate_preflight_report_path"]),
        args.output_dir or Path(config["output"]["directory"]),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
