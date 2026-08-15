#!/usr/bin/env python3
"""DEC028 single-study metadata-only successor P0.

The evaluator reads its static config and Git metadata, validates exactly eleven
groups, and publishes one aggregate record.  It has no data, split assignment,
PyTorch, CUDA, model, optimizer, checkpoint, parameter-update or G1-launch path.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/route_a_v3_dec028_single_study_successor_p0_v1.json"
PROTOCOL_ID = "ROUTE_A_V3_DEC028_SINGLE_STUDY_SUCCESSOR_METADATA_ONLY_P0_V1"
REPORT_FILENAME = "ROUTE_A_V3_DEC028_SINGLE_STUDY_SUCCESSOR_P0_RECORD_V1.json"
UNKNOWN = "UNKNOWN_NOT_ASSERTED"
GATES = (
    ("P0.1", "INPUT_MEMBERSHIP_AND_BINDING"),
    ("P0.2", "PRIOR_USE_ATTESTATION"),
    ("P0.3", "EXPOSURE_ROLE"),
    ("P0.4", "RIGHTS"),
    ("P0.5", "SCIENTIFIC_ROW_CONTRACT"),
    ("P0.6", "SCRATCH_ONLY_ROUTE"),
    ("P0.7", "PROSPECTIVE_SPLIT"),
    ("P0.8", "SINGLE_RUN_POLICY"),
    ("P0.9", "EXECUTABLE_SCIENTIFIC_GATES"),
    ("P0.10", "SUCCESSOR_LEARNED_RUN_IMPLEMENTATION"),
    ("P0.11", "STATE_LOCKS"),
)


class ProtocolError(RuntimeError):
    pass


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot load P0 config: {path}") from exc
    if type(value) is not dict:
        raise ProtocolError("P0 config root must be an object")
    validate_config(value)
    return value


def _require(value: bool, message: str) -> None:
    if not value:
        raise ProtocolError(message)


def _all_false(value: Mapping[str, Any]) -> bool:
    return bool(value) and all(item is False for item in value.values())


def validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("protocol_id") == PROTOCOL_ID, "protocol ID differs")
    _require(config.get("decision_id") == "V3-DEC-028", "decision differs")
    _require(config.get("runtime_authority_event_id") == "A1-EVT-061", "runtime authority differs")
    _require(config.get("document_status") == "ACTIVE_METADATA_ONLY_P0_EXACTLY_ONCE", "P0 is not active")
    _require(config["predecessor_record"]["result_preserved"] == {
        "pass": 3,
        "fail_closed": 7,
        "unknown_not_asserted": 1,
        "g1_launched": False,
    }, "DEC026 predecessor result differs")
    _require(config["predecessor_record"]["overwritten"] is False, "DEC026 predecessor was overwritten")
    _require([(item["gate_id"], item["gate_name"]) for item in config["gates"]] == list(GATES), "P0 gate list differs")
    _require(set(config["current_submission"]) == {gate_id for gate_id, _ in GATES}, "P0 submission closure differs")
    _require(config["evaluation_scope"]["exact_gate_count"] == 11, "P0 gate count differs")
    _require(config["evaluation_scope"]["success_action"] == "ELIGIBLE_TO_REQUEST_MATERIALIZATION_NOT_G1_NOT_LAUNCHED", "P0 success action differs")
    truth = config["current_truth"]
    _require(truth["qualified_counts"] == {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547}, "qualified counts differ")
    _require(truth["scientific_claim_status"] == "NOT_ESTABLISHED", "claim was promoted")
    for key in (
        "data_rows_read",
        "split_assignments_generated",
        "cuda_touches",
        "model_constructions",
        "optimizer_constructions",
        "checkpoint_reads",
        "checkpoint_writes",
        "parameter_updates",
    ):
        _require(truth[key] == 0, f"current truth is nonzero: {key}")
    _require(truth["g1_launched"] is False and truth["materialization_authority_granted"] is False, "downstream action occurred")


def _p01(e: Mapping[str, Any]) -> None:
    _require(e["membership_count"] == 6547, "P0.1 membership count differs")
    _require(e["study_unit"] == "GSE200304_SUPERSERIES_ONE_STUDY", "P0.1 study unit differs")
    _require(e["materialization_exercised"] is False and e["materialization_authority_granted"] is False, "P0.1 materialization occurred")


def _p02(e: Mapping[str, Any]) -> None:
    _require(e["scope"] == "DISCLOSED_EXPOSED_DEVELOPMENT_ONLY", "P0.2 scope differs")
    _require(e["predecessor_historical_status"] == "UNKNOWN_NOT_ASSERTED", "P0.2 rewrites historical status")
    _require(e["residual_historical_uncertainty_disclosed"] is True, "P0.2 residual uncertainty hidden")
    _require(e["untouched_or_confirmatory_claim_allowed"] is False, "P0.2 untouched claim allowed")


def _p03(e: Mapping[str, Any]) -> None:
    _require(e["role"] == "EXPOSED_DEVELOPMENT_ONLY", "P0.3 role differs")
    _require(e["participating_study_count"] == 1 and e["untouched_study_count"] == 0 and e["confirmatory_study_count"] == 0, "P0.3 study-role geometry differs")


def _p04(e: Mapping[str, Any]) -> None:
    _require(e["permission_scope"] == "PRIVATE_INTERNAL_PROCESS_TRAIN_EVALUATE_NO_PUBLIC_MEMBER_REDISTRIBUTION", "P0.4 rights scope differs")
    _require(len(e["basis"]) == 3, "P0.4 rights basis is incomplete")
    _require(e["public_member_redistribution_allowed"] is False and e["sealed_or_restricted_input_allowed"] is False, "P0.4 rights exceed scope")


def _p05(e: Mapping[str, Any]) -> None:
    _require(e["member_count"] == 6547, "P0.5 member count differs")
    expected = [
        "record_key", "source_group", "source_sequence", "candidate_sequence",
        "context_vector", "edit_features", "direction_normalized_effect", "biological_standard_error",
    ]
    _require(e["required_fields_exactly"] == expected, "P0.5 row fields differ")
    _require(all(e[key] is True for key in (
        "source_candidate_identity_required", "endpoint_direction_transform_required",
        "biological_group_and_positive_se_required", "rights_and_exposure_required",
        "materialization_conformance_not_yet_run",
    )), "P0.5 row contract is incomplete")


def _p06(e: Mapping[str, Any]) -> None:
    _require(e["route"] == "SCRATCH_ONLY_NO_FOUNDATION_NO_EXTERNAL_LEARNED_INPUTS", "P0.6 route differs")
    _require(e["initialization"] == "RANDOM_INITIALIZATION_ONLY", "P0.6 initialization differs")
    _require(e["external_learned_input_count"] == 0 and e["checkpoint_read_count_before_first_update"] == 0, "P0.6 external learned input present")


def _p07(e: Mapping[str, Any]) -> None:
    _require(e["contract_id"] == "ONE_FROZEN_COMPONENT_DISJOINT_TRAIN_CALIBRATION_TEST_SPLIT_V1", "P0.7 split contract differs")
    _require(e["outcome_blind"] is True and e["model_or_endpoint_result_used"] is False, "P0.7 is not outcome blind")
    _require(e["split_assignment_count"] == 0, "P0.7 generated assignments")
    _require(set(e["grouping_rules"]) == {
        "BIOLOGICAL_SOURCE_GROUP", "EXACT_GENE_TOKEN",
        "ORIENTATION_NORMALIZED_WT201_HAMMING_LE_10",
        "CANONICAL_15MER_SET_JACCARD_GE_0_80", "REVERSE_PAIR_SAME_COMPONENT",
    }, "P0.7 grouping rules differ")
    _require(abs(sum(e["target_proportions"].values()) - 1.0) < 1e-12, "P0.7 proportions do not sum to one")


def _p08(e: Mapping[str, Any]) -> None:
    _require([e[key] for key in (
        "authorized_execution_count", "optimizer_fit_count", "fold_model_count",
        "checkpoint_count", "final_refit_count", "seed_count",
    )] == [1, 1, 1, 1, 0, 1], "P0.8 execution counts differ")
    _require(e["terminal_checkpoint_only"] is True, "P0.8 terminal checkpoint differs")
    _require(all(e[key] is False for key in (
        "early_stopping_allowed", "best_checkpoint_selection_allowed",
        "hyperparameter_search_allowed", "automatic_retry_allowed",
    )), "P0.8 selection or retry is enabled")


def _p09(e: Mapping[str, Any]) -> None:
    _require(e["gate_bundle_id"] == "GSE200304_SOURCE_RELATIVE_CRITIC_G1_FAIL_CLOSED_GATE_BUNDLE_V1", "P0.9 bundle differs")
    _require(e["pre_model_gate_count"] == 5 and e["runtime_gate_count"] == 6, "P0.9 gate counts differ")
    _require(e["binding_state"] == "BOUND_EXECUTABLE_NOT_RUN" and e["runtime_gate_execution_count"] == 0, "P0.9 execution state differs")
    _require(e["any_nonpass_action"] == "STOP_WITH_EVIDENCE_NO_RETRY", "P0.9 failure action differs")
    _require((REPO_ROOT / e["implementation_path"]).is_file(), "P0.9 implementation path is absent")


def _p10(e: Mapping[str, Any]) -> None:
    _require(e["implementation_id"] == "ROUTE_A_V3_GSE200304_SOURCE_RELATIVE_CRITIC_G1_IMPLEMENTATION_CANDIDATE_V1", "P0.10 implementation differs")
    _require(e["review_status"] == "PASS_STATIC_IMPLEMENTATION_BINDING_ONLY_NOT_ACTIVE_NOT_RUN", "P0.10 review differs")
    _require(e["current_activation_state"] == "INACTIVE_FAIL_BEFORE_DATA_MODEL_CUDA_OUTPUT", "P0.10 current state differs")
    _require(e["future_activation_requires_separate_authority"] is True and e["a6_learned_base_value_implementation"] is False, "P0.10 scope differs")
    commit = e["implementation_commit"]
    observed = subprocess.run(["git", "rev-parse", commit], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    _require(observed == commit, "P0.10 implementation commit is absent")
    for key in ("implementation_config_path", "implementation_script_path", "implementation_test_path"):
        path = e[key]
        _require((REPO_ROOT / path).is_file(), f"P0.10 missing path: {key}")
        committed = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        _require((REPO_ROOT / path).read_bytes() == committed, f"P0.10 working bytes differ: {path}")


def _p11(e: Mapping[str, Any]) -> None:
    _require(_all_false(e["pre_p0_locks"]), "P0.11 pre-P0 locks differ")
    _require(_all_false(e["persistent_locks"]), "P0.11 persistent locks differ")


VALIDATORS: dict[str, Callable[[Mapping[str, Any]], None]] = {
    "P0.1": _p01, "P0.2": _p02, "P0.3": _p03, "P0.4": _p04,
    "P0.5": _p05, "P0.6": _p06, "P0.7": _p07, "P0.8": _p08,
    "P0.9": _p09, "P0.10": _p10, "P0.11": _p11,
}


def evaluate(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    statuses = []
    for gate_id, gate_name in GATES:
        evidence = config["current_submission"][gate_id]
        raw = evidence.get("status", "MISSING")
        if raw == "PASS":
            try:
                VALIDATORS[gate_id](evidence)
            except Exception as exc:
                raw = f"FAIL_CLOSED_METADATA_MISMATCH:{type(exc).__name__}:{exc}"
        statuses.append({"gate_id": gate_id, "gate_name": gate_name, "status": raw})
    all_pass = all(item["status"] == "PASS" for item in statuses)
    return {
        "protocol_id": PROTOCOL_ID,
        "decision_id": "V3-DEC-028",
        "predecessor_result_preserved": config["predecessor_record"]["result_preserved"],
        "gate_statuses": statuses,
        "status_counts": {
            "PASS": sum(item["status"] == "PASS" for item in statuses),
            "NONPASS": sum(item["status"] != "PASS" for item in statuses),
        },
        "overall_status": (
            "ELIGIBLE_TO_REQUEST_MATERIALIZATION_NOT_G1_NOT_LAUNCHED"
            if all_pass
            else "STOP_BEFORE_DATA_ROWS_CUDA_MODEL_OPTIMIZER_CHECKPOINT_PARAMETER_UPDATE_OR_TRAINING"
        ),
        "materialization_authority_granted": False,
        "g1_launched": False,
        "forbidden_touchpoint_count": 0,
        "qualified_counts": config["current_truth"]["qualified_counts"],
        "scientific_claim_status": "NOT_ESTABLISHED",
    }


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def audit_repository(config: Mapping[str, Any]) -> None:
    group = config["implementation_binding"]
    if group["status"] != "BOUND":
        raise ProtocolError("implementation binding is not BOUND")
    if _git("status", "--porcelain"):
        raise ProtocolError("production repository is not clean")
    head = _git("rev-parse", "HEAD")
    implementation = group["implementation_commit"]
    _require(_git("rev-parse", f"{head}^") == implementation, "binding commit parent differs")
    binding_paths = [line for line in _git("diff-tree", "--no-commit-id", "--name-only", "-r", head).splitlines() if line]
    _require(sorted(binding_paths) == sorted(group["binding_exact_changed_paths"]), "binding changed paths differ")
    _require(_git("rev-parse", f"{implementation}^") == group["implementation_expected_parent"], "implementation parent differs")
    implementation_paths = [line for line in _git("diff-tree", "--no-commit-id", "--name-only", "-r", implementation).splitlines() if line]
    _require(sorted(implementation_paths) == sorted(group["implementation_exact_changed_paths"]), "implementation changed paths differ")
    for path in group["implementation_exact_changed_paths"][1:]:
        committed = subprocess.run(["git", "show", f"{implementation}:{path}"], cwd=REPO_ROOT, check=True, capture_output=True).stdout
        _require((REPO_ROOT / path).read_bytes() == committed, f"working bytes differ: {path}")


def publish(result: Mapping[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / REPORT_FILENAME
    if target.exists():
        raise ProtocolError("P0 report already exists; exactly-once execution is exhausted")
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{REPORT_FILENAME}.", dir=output_dir)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    audit_repository(config)
    result = evaluate(config)
    target = publish(result, args.output_dir)
    print(json.dumps({"overall_status": result["overall_status"], "report": str(target)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
