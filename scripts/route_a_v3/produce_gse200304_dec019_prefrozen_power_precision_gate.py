#!/usr/bin/env python3
"""Produce the aggregate GSE200304 prefrozen power/precision gate.

This is a planning calculation, not observed model power and not an interval
estimated from model results.  It uses the 6,544 outcome-blind outer-OOF
biological source groups frozen by the accepted split authority and the
prefrozen Spearman alternative from the A1 qualification contract.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import re
from pathlib import Path
from statistics import NormalDist
from types import ModuleType
from typing import Any, Mapping, Sequence


UNKNOWN = "UNKNOWN_NOT_ASSERTED"
PASS = "PASS"
CONTRACT_ID = "mrna_xeditflow_route_a_v3"
DATASET_ID = "GSE200304"
DECISION_ID = "V3-DEC-019"
GATE_ID = "PREFROZEN_POWER_PRECISION"
PROTOCOL_ID = "ROUTE_A_V3_GSE200304_DEC019_PREFROZEN_POWER_PRECISION_GATE_V1"
EVIDENCE_SCHEMA = "route_a_v3_dec019_aggregate_gate_evidence.v3"
EVIDENCE_RECORD_TYPE = "ROUTE_A_V3_DEC019_ACCEPTED_AGGREGATE_GATE_EVIDENCE_V3"
ANALYSIS_UNIT = "BIOLOGICAL_SOURCE_GROUP"
EVALUATION_POPULATION = (
    "A1_ELIGIBLE_OUTCOME_BLIND_OUTER_OOF_GROUPS_NOT_A2_FINAL_MEMBERSHIP"
)
TARGET_METRIC = "WITHIN_STUDY_SPEARMAN"
POWER_METHOD = "BONETT_WRIGHT_FISHER_Z_ASYMPTOTIC_TWO_SIDED_SPEARMAN"
CI_METHOD = "BONETT_WRIGHT_FISHER_Z_SPEARMAN_AT_PREFROZEN_ALTERNATIVE"
WORKING_DISTRIBUTION_ASSUMPTION = (
    "MONOTONIC_TRANSFORMATION_OF_BIVARIATE_NORMAL_AT_PREFROZEN_SPEARMAN_RHO"
)
EXPECTED_GROUP_COUNT = 6544
EXPECTED_OUTER_GROUP_COUNTS = [1309, 1309, 1309, 1309, 1308]
EXPECTED_ASSIGNMENT_COMMITMENT = (
    "0df20ce2e419af7573d45d2fabce6162e02606847b0c26c3ffd0b9a01b63395e"
)
SPLIT_COMMITMENT_KEY = "split_assignment_commitment_sha256"
CONFIG_REPO_PATH = (
    "configs/route_a_v3_gse200304_dec019_prefrozen_power_precision_gate_v1.json"
)
SCRIPT_REPO_PATH = (
    "scripts/route_a_v3/produce_gse200304_dec019_prefrozen_power_precision_gate.py"
)
TEST_REPO_PATH = (
    "tests/route_a_v3/test_produce_gse200304_dec019_prefrozen_power_precision_gate.py"
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ProducerError(RuntimeError):
    """A required aggregate authority or prefrozen value differs."""


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")


def read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProducerError(f"cannot read valid JSON for {label}") from exc
    if type(value) is not dict:
        raise ProducerError(f"{label} root is not an object")
    return value


def validate_static_config(config: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": (
            "route_a_v3_gse200304_dec019_prefrozen_power_precision_gate.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "contract_id": CONTRACT_ID,
        "phase_id": "A1",
        "dataset_id": DATASET_ID,
        "decision_id": DECISION_ID,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ProducerError(f"config {key} differs")
    repository = config["repository_authority"]
    if repository["implementation_commit_exact_changed_paths"] != [
        CONFIG_REPO_PATH,
        SCRIPT_REPO_PATH,
        TEST_REPO_PATH,
    ]:
        raise ProducerError("implementation exact-three path set differs")
    if repository["binding_commit_exact_changed_paths"] != [CONFIG_REPO_PATH]:
        raise ProducerError("binding path set differs")
    base = repository["implementation_base_commit"]
    if base != UNKNOWN and HEX40.fullmatch(str(base)) is None:
        raise ProducerError("implementation base is neither UNKNOWN nor HEX40")
    calculation = config["calculation_contract"]
    expected_calculation = {
        "analysis_unit": ANALYSIS_UNIT,
        "bootstrap_unit": ANALYSIS_UNIT,
        "evaluation_population": EVALUATION_POPULATION,
        "target_metric": TARGET_METRIC,
        "alternative_spearman_rho": 0.25,
        "two_sided_alpha": 0.05,
        "target_power": 0.8,
        "confidence_level": 0.95,
        "maximum_full_confidence_interval_width": 0.3,
        "power_method": POWER_METHOD,
        "confidence_interval_method": CI_METHOD,
        "working_distribution_assumption": WORKING_DISTRIBUTION_ASSUMPTION,
        "prefrozen_before_model_results": True,
    }
    if calculation != expected_calculation:
        raise ProducerError("calculation contract differs")


def validate_implementation_binding(config: Mapping[str, Any]) -> None:
    binding = config["implementation_binding"]
    if binding["status"] != "BOUND":
        raise ProducerError("producer implementation binding is UNKNOWN")
    if HEX40.fullmatch(str(binding["implementation_commit"])) is None:
        raise ProducerError("producer implementation commit is not bound")
    if HEX40.fullmatch(str(binding["consumer_upgrade_binding_commit"])) is None:
        raise ProducerError("consumer upgrade binding commit is not bound")
    if (
        binding["consumer_upgrade_binding_commit"]
        != config["repository_authority"]["implementation_base_commit"]
    ):
        raise ProducerError("producer base differs from consumer upgrade binding")
    for key in (
        "implementation_script_sha256",
        "implementation_test_sha256",
        "consumer_upgrade_config_sha256",
        "consumer_upgrade_script_sha256",
    ):
        if HEX64.fullmatch(str(binding[key])) is None:
            raise ProducerError(f"{key} is not bound")


def validate_split_authority(
    audit: Mapping[str, Any], gate: Mapping[str, Any]
) -> tuple[int, str]:
    if (
        audit.get("schema_version")
        != "route_a_v3_gse200304_split_leakage_audit.v1"
        or audit.get("dataset_id") != DATASET_ID
        or audit.get("decision_id") != DECISION_ID
        or audit.get("status") != "GO_PASS_CONDITIONS_MET"
        or audit.get("record_count") != 6547
        or audit.get("biological_group_node_count") != EXPECTED_GROUP_COUNT
        or audit.get("outcome_columns_read") != []
        or audit.get("final_benchmark_membership_deferred_to_a2") is not True
        or audit.get("all_outer_folds_nonempty") is not True
        or audit.get("all_outer_train_inner_folds_nonempty") is not True
        or audit.get("all_required_cross_fold_leakage_counts_zero") is not True
    ):
        raise ProducerError("split aggregate audit does not authorize the calculation")
    outer = audit.get("outer_fold_counts")
    if type(outer) is not list or [item.get("group_count") for item in outer] != (
        EXPECTED_OUTER_GROUP_COUNTS
    ):
        raise ProducerError("split outer group counts differ")
    commitment = audit.get("assignment_commitment_sha256")
    if commitment != EXPECTED_ASSIGNMENT_COMMITMENT:
        raise ProducerError("split assignment commitment differs")
    if (
        gate.get("schema_version") != EVIDENCE_SCHEMA
        or gate.get("record_type") != EVIDENCE_RECORD_TYPE
        or gate.get("dataset_id") != DATASET_ID
        or gate.get("decision_id") != DECISION_ID
        or gate.get("gate_id") != "OUTCOME_BLIND_SPLIT_LEAKAGE"
        or gate.get("status") != PASS
        or gate.get("accepted") is not True
        or gate.get("aggregate_only") is not True
        or gate.get("provenance", {}).get(SPLIT_COMMITMENT_KEY) != commitment
    ):
        raise ProducerError("split PASS gate does not match the aggregate audit")
    return EXPECTED_GROUP_COUNT, commitment


def validate_prefreeze(qualification: Mapping[str, Any]) -> None:
    prefreeze = qualification.get("power_prefreeze")
    expected = {
        "analysis_unit": ANALYSIS_UNIT,
        "bootstrap_unit": ANALYSIS_UNIT,
        "target_metric": TARGET_METRIC,
        "minimum_effect_at_alternative": 0.25,
        "alpha_two_sided": 0.05,
        "target_power": 0.8,
        "confidence_level": 0.95,
        "maximum_ci_full_width": 0.3,
        "simulation_seed": 20260810,
        "bootstrap_resamples": 2000,
        "simulation_trials": 1000,
        "model_results_may_change_this_rule": False,
    }
    if prefreeze != expected:
        raise ProducerError("A1 power prefreeze differs")


def fisher_z_plan(
    group_count: int,
    alternative_spearman_rho: float,
    two_sided_alpha: float,
    confidence_level: float,
) -> dict[str, float]:
    if group_count <= 3 or not -1.0 < alternative_spearman_rho < 1.0:
        raise ProducerError("Fisher-z inputs are outside their valid planning domain")
    normal = NormalDist()
    null_standard_error = 1.0 / math.sqrt(group_count - 3)
    alternative_z = math.atanh(alternative_spearman_rho)
    alternative_standard_error = (
        math.sqrt(1.0 + alternative_spearman_rho**2 / 2.0)
        * null_standard_error
    )
    critical = normal.inv_cdf(1.0 - two_sided_alpha / 2.0) * null_standard_error
    estimated_power = (
        1.0
        - normal.cdf(
            (critical - alternative_z) / alternative_standard_error
        )
        + normal.cdf(
            (-critical - alternative_z) / alternative_standard_error
        )
    )
    ci_critical = normal.inv_cdf(0.5 + confidence_level / 2.0)
    lower = math.tanh(
        alternative_z - ci_critical * alternative_standard_error
    )
    upper = math.tanh(
        alternative_z + ci_critical * alternative_standard_error
    )
    return {
        "null_fisher_z_standard_error": null_standard_error,
        "alternative_fisher_z_standard_error": alternative_standard_error,
        "estimated_design_power": estimated_power,
        "planned_confidence_interval_lower": lower,
        "planned_confidence_interval_upper": upper,
        "planned_full_confidence_interval_width": upper - lower,
    }


def build_audit(
    config: Mapping[str, Any], group_count: int, split_commitment: str
) -> dict[str, Any]:
    contract = config["calculation_contract"]
    result = fisher_z_plan(
        group_count,
        contract["alternative_spearman_rho"],
        contract["two_sided_alpha"],
        contract["confidence_level"],
    )
    power_pass = result["estimated_design_power"] >= contract["target_power"]
    precision_pass = (
        result["planned_full_confidence_interval_width"]
        <= contract["maximum_full_confidence_interval_width"]
    )
    return {
        "schema_version": "route_a_v3_gse200304_prefrozen_power_precision_audit.v1",
        "record_type": "GSE200304_PREFROZEN_POWER_PRECISION_PLANNING_AUDIT_V1",
        "dataset_id": DATASET_ID,
        "decision_id": DECISION_ID,
        "status": "GO_PASS_CONDITIONS_MET" if power_pass and precision_pass else "STOP_PASS_CONDITIONS_NOT_MET",
        "analysis_unit": ANALYSIS_UNIT,
        "evaluation_population": EVALUATION_POPULATION,
        "evaluation_group_count": group_count,
        "split_assignment_commitment_sha256": split_commitment,
        "target_metric": TARGET_METRIC,
        "alternative_spearman_rho": contract["alternative_spearman_rho"],
        "two_sided_alpha": contract["two_sided_alpha"],
        "target_power": contract["target_power"],
        "confidence_level": contract["confidence_level"],
        "maximum_full_confidence_interval_width": contract[
            "maximum_full_confidence_interval_width"
        ],
        "power_method": POWER_METHOD,
        "confidence_interval_method": CI_METHOD,
        "working_distribution_assumption": WORKING_DISTRIBUTION_ASSUMPTION,
        "planning_assumption": (
            "FISHER_TRANSFORM_OF_SAMPLE_SPEARMAN_IS_APPROXIMATELY_NORMAL_"
            "WITH_MEAN_ATANH_RHO_S_AND_BONETT_WRIGHT_VARIANCE_"
            "ONE_PLUS_RHO_S_SQUARED_OVER_TWO_DIVIDED_BY_N_MINUS_3"
        ),
        **result,
        "power_pass": power_pass,
        "precision_pass": precision_pass,
        "prefrozen_before_model_results": True,
        "observed_model_power_claimed": False,
        "actual_model_confidence_interval_claimed": False,
        "final_a2_benchmark_membership_claimed": False,
    }


def load_consumer_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("gse200304_power_consumer", path)
    if spec is None or spec.loader is None:
        raise ProducerError("consumer module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_gate_record(
    config: Mapping[str, Any], audit: Mapping[str, Any], consumer: Mapping[str, Any]
) -> dict[str, Any]:
    if audit["status"] != "GO_PASS_CONDITIONS_MET":
        raise ProducerError("power/precision calculation did not meet PASS conditions")
    binding = config["implementation_binding"]
    predecessor = consumer["evidence_contract"]["required_predecessor_authority"]
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "record_type": EVIDENCE_RECORD_TYPE,
        "contract_id": CONTRACT_ID,
        "decision_id": DECISION_ID,
        "dataset_id": DATASET_ID,
        "gate_id": GATE_ID,
        "status": PASS,
        "accepted": True,
        "aggregate_only": True,
        "privacy": {
            "contains_row_level_payload": False,
            "contains_sequence": False,
            "contains_row_identifier": False,
            "contains_raw_label_or_effect": False,
            "contains_member_identifiers_or_hashes": False,
        },
        "provenance": {
            "producer_protocol_id": PROTOCOL_ID,
            "producer_commit": binding["implementation_commit"],
            "producer_script_sha256": binding["implementation_script_sha256"],
            "source_bundle_id": predecessor["bundle_id"],
            "source_bundle_root_or_target_sha256": predecessor[
                "terminal_marker_final_output_target_sha256"
            ],
            "predecessor_authority": copy.deepcopy(predecessor),
            "acceptance_authority": copy.deepcopy(
                consumer["evidence_contract"]["gate_record_provenance_contract"]
                ["acceptance_authority"]
            ),
        },
        "facts": {
            "analysis_unit": audit["analysis_unit"],
            "bootstrap_unit": ANALYSIS_UNIT,
            "evaluation_population": audit["evaluation_population"],
            "evaluation_group_count": audit["evaluation_group_count"],
            "target_metric": audit["target_metric"],
            "alternative_spearman_rho": audit["alternative_spearman_rho"],
            "two_sided_alpha": audit["two_sided_alpha"],
            "power_method": audit["power_method"],
            "working_distribution_assumption": audit[
                "working_distribution_assumption"
            ],
            "estimated_design_power": audit["estimated_design_power"],
            "confidence_level": audit["confidence_level"],
            "confidence_interval_method": audit["confidence_interval_method"],
            "planned_full_confidence_interval_width": audit[
                "planned_full_confidence_interval_width"
            ],
            "prefrozen_before_model_results": True,
        },
        "unknown_fields": [],
        "reason_codes": [],
    }


def validate_with_consumer(
    gate: Mapping[str, Any], consumer_config: Mapping[str, Any], module: ModuleType
) -> None:
    module.validate_static_config(consumer_config)
    slot = next(
        item
        for item in consumer_config["evidence_contract"]["slots"]
        if item["slot_id"] == GATE_ID
    )
    module._validate_gate_record(json_bytes(gate), slot, consumer_config)


def produce(config_path: Path, consumer_config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = read_json(config_path, label="producer config")
    validate_static_config(config)
    validate_implementation_binding(config)
    source = config["source_authority"]
    audit = read_json(Path(source["split_aggregate_audit_path"]), label="split audit")
    split_gate = read_json(Path(source["split_pass_gate_path"]), label="split gate")
    qualification = read_json(Path(source["a1_qualification_config_path"]), label="A1 qualification")
    group_count, split_commitment = validate_split_authority(audit, split_gate)
    validate_prefreeze(qualification)
    result = build_audit(config, group_count, split_commitment)
    consumer_config = read_json(consumer_config_path, label="consumer config")
    module = load_consumer_module(Path(config["consumer_contract"]["script_path"]))
    gate = build_gate_record(config, result, consumer_config)
    validate_with_consumer(gate, consumer_config, module)
    if output_dir.exists():
        raise ProducerError("output directory already exists")
    output_dir.mkdir(parents=True)
    (output_dir / config["output_contract"]["audit_basename"]).write_bytes(
        json_bytes(result)
    )
    (output_dir / config["output_contract"]["gate_basename"]).write_bytes(
        json_bytes(gate)
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--consumer-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = produce(args.config, args.consumer_config, args.output_dir)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
