from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/produce_gse200304_dec019_prefrozen_power_precision_gate.py"
CONFIG = ROOT / "configs/route_a_v3_gse200304_dec019_prefrozen_power_precision_gate_v1.json"
CONSUMER_CONFIG = ROOT / "configs/route_a_v3_gse200304_dec019_reported_endpoint_a1_activation_v3.json"
CONSUMER_SCRIPT = ROOT / "scripts/route_a_v3/adjudicate_gse200304_dec019_reported_endpoint_a1_v3.py"
QUALIFICATION = ROOT / "configs/route_a_v3_a1_qualification.json"

spec = importlib.util.spec_from_file_location("gse200304_power_producer", SCRIPT)
assert spec is not None and spec.loader is not None
PRODUCER = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PRODUCER)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bound_config() -> dict[str, Any]:
    config = read_json(CONFIG)
    config["implementation_binding"] = {
        "status": "BOUND",
        "implementation_commit": "0" * 39 + "1",
        "implementation_script_sha256": "0" * 63 + "2",
        "implementation_test_sha256": "0" * 63 + "3",
        "consumer_upgrade_binding_commit": "0" * 39 + "4",
        "consumer_upgrade_config_sha256": "0" * 63 + "5",
        "consumer_upgrade_script_sha256": "0" * 63 + "6",
    }
    config["repository_authority"]["implementation_base_commit"] = "0" * 39 + "4"
    return config


def split_audit() -> dict[str, Any]:
    return {
        "schema_version": "route_a_v3_gse200304_split_leakage_audit.v1",
        "dataset_id": PRODUCER.DATASET_ID,
        "decision_id": PRODUCER.DECISION_ID,
        "status": "GO_PASS_CONDITIONS_MET",
        "record_count": 6547,
        "biological_group_node_count": 6544,
        "outcome_columns_read": [],
        "final_benchmark_membership_deferred_to_a2": True,
        "outer_fold_counts": [
            {"fold": index, "group_count": count}
            for index, count in enumerate(PRODUCER.EXPECTED_OUTER_GROUP_COUNTS)
        ],
        "all_outer_folds_nonempty": True,
        "all_outer_train_inner_folds_nonempty": True,
        "all_required_cross_fold_leakage_counts_zero": True,
        "assignment_commitment_sha256": PRODUCER.EXPECTED_ASSIGNMENT_COMMITMENT,
    }


def split_gate() -> dict[str, Any]:
    return {
        "schema_version": PRODUCER.EVIDENCE_SCHEMA,
        "record_type": PRODUCER.EVIDENCE_RECORD_TYPE,
        "dataset_id": PRODUCER.DATASET_ID,
        "decision_id": PRODUCER.DECISION_ID,
        "gate_id": "OUTCOME_BLIND_SPLIT_LEAKAGE",
        "status": "PASS",
        "accepted": True,
        "aggregate_only": True,
        "provenance": {
            PRODUCER.SPLIT_COMMITMENT_KEY: PRODUCER.EXPECTED_ASSIGNMENT_COMMITMENT
        },
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_direct_fisher_z_plan_passes_prefrozen_thresholds() -> None:
    result = PRODUCER.fisher_z_plan(6544, 0.25, 0.05, 0.95)
    assert result["estimated_design_power"] == 1.0
    assert result["planned_confidence_interval_lower"] == pytest.approx(
        0.2267902054719841
    )
    assert result["planned_confidence_interval_upper"] == pytest.approx(
        0.2729260036827754
    )
    assert result["planned_full_confidence_interval_width"] == pytest.approx(
        0.04613579821079131
    )
    audit = PRODUCER.build_audit(
        bound_config(), 6544, PRODUCER.EXPECTED_ASSIGNMENT_COMMITMENT
    )
    assert audit["status"] == "GO_PASS_CONDITIONS_MET"
    assert audit["power_pass"] is audit["precision_pass"] is True
    assert audit["observed_model_power_claimed"] is False
    assert audit["actual_model_confidence_interval_claimed"] is False


def test_split_authority_supplies_groups_not_records_as_n() -> None:
    group_count, commitment = PRODUCER.validate_split_authority(
        split_audit(), split_gate()
    )
    assert group_count == 6544
    assert commitment == PRODUCER.EXPECTED_ASSIGNMENT_COMMITMENT
    bad = split_audit()
    bad["biological_group_node_count"] = 6547
    with pytest.raises(PRODUCER.ProducerError, match="does not authorize"):
        PRODUCER.validate_split_authority(bad, split_gate())


def test_power_gate_is_accepted_by_upgraded_consumer() -> None:
    config = bound_config()
    consumer = read_json(CONSUMER_CONFIG)
    audit = PRODUCER.build_audit(
        config, 6544, PRODUCER.EXPECTED_ASSIGNMENT_COMMITMENT
    )
    gate = PRODUCER.build_gate_record(config, audit, consumer)
    module = PRODUCER.load_consumer_module(CONSUMER_SCRIPT)
    PRODUCER.validate_with_consumer(gate, consumer, module)
    assert gate["facts"]["analysis_unit"] == "BIOLOGICAL_SOURCE_GROUP"
    assert gate["facts"]["working_distribution_assumption"] == (
        PRODUCER.WORKING_DISTRIBUTION_ASSUMPTION
    )
    assert "observed_power" not in gate["facts"]
    assert gate["facts"]["estimated_design_power"] == 1.0

    invalid = copy.deepcopy(gate)
    invalid["facts"]["working_distribution_assumption"] = "UNSPECIFIED"
    with pytest.raises(module.AdjudicationError, match="working_distribution_assumption"):
        PRODUCER.validate_with_consumer(invalid, consumer, module)


def test_producer_reads_only_aggregate_authorities_and_writes_two_outputs(
    tmp_path: Path,
) -> None:
    config = bound_config()
    split_audit_path = tmp_path / "split-audit.json"
    split_gate_path = tmp_path / "split-gate.json"
    config_path = tmp_path / "producer.json"
    write_json(split_audit_path, split_audit())
    write_json(split_gate_path, split_gate())
    config["source_authority"]["split_aggregate_audit_path"] = str(split_audit_path)
    config["source_authority"]["split_pass_gate_path"] = str(split_gate_path)
    config["source_authority"]["a1_qualification_config_path"] = str(QUALIFICATION)
    config["consumer_contract"]["script_path"] = str(CONSUMER_SCRIPT)
    write_json(config_path, config)
    output = tmp_path / "power"
    result = PRODUCER.produce(config_path, CONSUMER_CONFIG, output)
    assert result["status"] == "GO_PASS_CONDITIONS_MET"
    assert sorted(path.name for path in output.iterdir()) == [
        "GSE200304_DEC019_PREFROZEN_POWER_PRECISION_AUDIT.json",
        "GSE200304_DEC019_PREFROZEN_POWER_PRECISION_GATE.json",
    ]
