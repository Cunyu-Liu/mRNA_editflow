#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


EXPECTED_CURRENT_TRUTH = {
    "qualified_ordinary_studies": 1,
    "qualified_a1_studies": 1,
    "qualified_true_a2_studies": 0,
    "qualified_canonical_records": 6547,
    "diagnostic_model_parameter_count": 81794,
    "diagnostic_model_status": "COMPLETED_NEGATIVE_NO_PREDICTIVE_SIGNAL_ESTABLISHED",
    "edit_flow_status": "ACTIVE_IMPLEMENTATION_TARGET_NOT_YET_SCIENTIFICALLY_ESTABLISHED",
    "scientific_claim_status": "NOT_ESTABLISHED",
}
EXPECTED_DEVELOPMENT = {
    "GSE200304",
    "GSE114002",
    "GSE149487",
    "GSE217518",
    "ENCSR854RUF",
    "GSE186455",
    "GSE256185",
    "GSE269595",
}
EXPECTED_EVALUATION = {"GSE232572", "E-MTAB-10902"}
EXPECTED_ALL_STUDIES = EXPECTED_DEVELOPMENT | EXPECTED_EVALUATION | {
    "GSE145046",
    "GSE207584",
    "GSE261709",
    "GSE246381",
}
EXPECTED_RUN_ROLES = {
    "PREDICTION_BASELINE",
    "DELTA_PREDICTOR",
    "SEARCH_BASELINE",
    "FLOW_G0_BASE",
    "GUIDED_XEDITFLOW",
}


class Route2ConfigError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Route2ConfigError(message)


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _study_sets(config: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    inventory = config["study_inventory"]
    ids = [item["study_unit_id"] for item in inventory]
    _require(len(ids) == len(set(ids)), "study inventory contains duplicate identifiers")
    development = {item["study_unit_id"] for item in inventory if item["pool"] == "DEVELOPMENT"}
    evaluation = {item["study_unit_id"] for item in inventory if item["pool"] == "EVALUATION"}
    return set(ids), development, evaluation


def guided_generation_allowed(config: dict[str, Any]) -> bool:
    readiness = config["readiness"]
    return (
        readiness["critic"] == "CRITIC_READY_FOR_GUIDANCE"
        and readiness["flow"] == "FLOW_G0_READY"
    )


def validate_run_role(config: dict[str, Any], run_role: str) -> None:
    _require(run_role in config["run_roles"], f"unknown run role: {run_role}")
    if run_role == "GUIDED_XEDITFLOW":
        _require(guided_generation_allowed(config), "guided generation dependencies are not met")


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    _require(config["schema_version"] == "route_a_v3_route2.v1", "unexpected schema version")
    _require(config["status"] == "ACTIVE_DUAL_TRACK_DEVELOPMENT", "Route 2 is not active")
    _require(config["contract"]["version"] == "V3.3.0", "unexpected contract version")
    _require(config["contract"]["parallel_contract_allowed"] is False, "parallel contract is enabled")
    _require(config["current_truth"] == EXPECTED_CURRENT_TRUTH, "current scientific facts changed")

    policy = config["development_policy"]
    _require(policy["full_route_a_qualification_is_development_gate"] is False, "3/2/1 still blocks development")
    for key in (
        "canonicalization_enabled",
        "baseline_execution_enabled",
        "prediction_training_enabled",
        "flow_g0_implementation_enabled",
        "training_requires_nvidia_gpu",
        "evaluation_protected",
    ):
        _require(policy[key] is True, f"required Route 2 policy disabled: {key}")
    _require(policy["training_cuda_fallback_to_cpu_allowed"] is False, "CPU training fallback is enabled")
    _require(policy["eligible_physical_gpu_indices"] == [0, 1, 2, 3, 4, 5], "unexpected GPU set")
    for key in (
        "successor_authority_required",
        "runtime_ledger_required",
        "one_read_required",
        "resource_once_required",
    ):
        _require(policy[key] is False, f"obsolete dependency restored: {key}")

    ids, development, evaluation = _study_sets(config)
    _require(ids == EXPECTED_ALL_STUDIES, "14-study inventory changed")
    _require(len(ids) == 14, "study inventory does not contain exactly 14 units")
    _require(development == EXPECTED_DEVELOPMENT, "Development pool changed")
    _require(evaluation == EXPECTED_EVALUATION, "Evaluation pool changed")
    _require(development.isdisjoint(evaluation), "Development and Evaluation overlap")

    studies = {item["study_unit_id"]: item for item in config["study_inventory"]}
    _require(studies["GSE200304"]["canonical_records"] == 6547, "GSE200304 record count changed")
    _require(studies["GSE200304"]["qualification_class"] == "QUALIFIED_CURRENT", "GSE200304 qualification changed")
    _require(studies["GSE232572"]["canonical_records"] == 8068, "GSE232572 replay count changed")
    _require(studies["GSE207584"]["canonical_records"] == 0, "GSE207584 gained unsupported records")
    _require(studies["GSE261709"]["canonical_records"] == 0, "GSE261709 gained unsupported records")
    _require(studies["GSE246381"]["pool"] == "SEALED_EXCLUDED", "sealed study became readable")

    _require(set(config["run_roles"]) == EXPECTED_RUN_ROLES, "run roles changed")
    _require(config["run_roles"]["DELTA_PREDICTOR"]["track"] == "PREDICTION", "predictor role is not distinct")
    _require(config["run_roles"]["FLOW_G0_BASE"]["track"] == "GENERATION", "flow role is not distinct")
    _require(
        config["run_roles"]["GUIDED_XEDITFLOW"]["requires"]
        == ["CRITIC_READY_FOR_GUIDANCE", "FLOW_G0_READY"],
        "guided readiness dependencies changed",
    )

    action_space = config["action_space"]
    _require(action_space["allowed_actions"] == ["SUB", "STOP"], "V1 action space changed")
    _require(action_space["ins_supported"] is False and action_space["del_supported"] is False, "INS/DEL enabled in V1")
    _require(action_space["repeat_position_edit_allowed"] is False, "repeat-position edits enabled")
    _require(action_space["hard_legality_mask_before_rate_normalization"] is True, "legality mask ordering changed")

    isolation = config["evaluation_isolation"]
    _require(set(isolation["evaluation_study_ids"]) == EXPECTED_EVALUATION, "Evaluation isolation list changed")
    _require(isolation["evaluation_in_training_allowed"] is False, "Evaluation entered training")
    _require(
        isolation["evaluation_for_architecture_hpo_or_threshold_selection_allowed"] is False,
        "Evaluation entered model selection",
    )
    _require(isolation["evaluation_model_gradient_to_generator_allowed"] is False, "Evaluation gradients reach generator")

    credit = config["credit_policy"]
    _require(not any(credit.values()), "development or generated data gained scientific credit")
    _require(
        config["storage"]["route2_data_root"] == "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2",
        "Route 2 artifact root changed",
    )
    _require(config["storage"]["large_artifacts_in_git_allowed"] is False, "large artifacts enabled in Git")

    _require(config["readiness"]["critic"] == "CRITIC_NOT_READY_FOR_GUIDANCE", "critic readiness is overstated")
    _require(config["readiness"]["flow"] == "FLOW_G0_NOT_READY", "flow readiness is overstated")
    _require(not guided_generation_allowed(config), "guided generation is prematurely enabled")

    return {
        "status": "PASS_ROUTE2_V1_EXECUTION_CONFIG",
        "study_unit_count": len(ids),
        "development_study_count": len(development),
        "evaluation_study_count": len(evaluation),
        "qualified_counts": "1/1/0/6547",
        "prediction_training_enabled": policy["prediction_training_enabled"],
        "flow_g0_implementation_enabled": policy["flow_g0_implementation_enabled"],
        "guided_generation_allowed": guided_generation_allowed(config),
        "scientific_claim_status": config["current_truth"]["scientific_claim_status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the active Route A V3.3 Route 2 execution config")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs/route_a_v3_route2_v1.json",
    )
    parser.add_argument("--run-role", choices=sorted(EXPECTED_RUN_ROLES))
    args = parser.parse_args()

    config = load_config(args.config)
    summary = validate_config(config)
    if args.run_role:
        validate_run_role(copy.deepcopy(config), args.run_role)
        summary["requested_run_role"] = args.run_role
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
