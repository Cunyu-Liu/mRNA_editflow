from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/route_a_v3/dec028_single_study_successor_p0.py"
CONFIG_PATH = REPO_ROOT / "configs/route_a_v3_dec028_single_study_successor_p0_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("dec028_successor_p0", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_successor_p0_has_exactly_eleven_passes_and_only_materialization_eligibility() -> None:
    module = _module()
    result = module.evaluate(_config())
    assert [item["gate_id"] for item in result["gate_statuses"]] == [f"P0.{index}" for index in range(1, 12)]
    assert {item["status"] for item in result["gate_statuses"]} == {"PASS"}
    assert result["status_counts"] == {"PASS": 11, "NONPASS": 0}
    assert result["overall_status"] == "ELIGIBLE_TO_REQUEST_MATERIALIZATION_NOT_G1_NOT_LAUNCHED"
    assert result["materialization_authority_granted"] is False
    assert result["g1_launched"] is False
    assert result["forbidden_touchpoint_count"] == 0
    assert result["qualified_counts"] == {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547}
    assert result["scientific_claim_status"] == "NOT_ESTABLISHED"


def test_p02_is_scoped_pass_without_rewriting_historical_unknown() -> None:
    evidence = _config()["current_submission"]["P0.2"]
    assert evidence["status"] == "PASS"
    assert evidence["scope"] == "DISCLOSED_EXPOSED_DEVELOPMENT_ONLY"
    assert evidence["predecessor_historical_status"] == "UNKNOWN_NOT_ASSERTED"
    assert evidence["residual_historical_uncertainty_disclosed"] is True
    assert evidence["untouched_or_confirmatory_claim_allowed"] is False


@pytest.mark.parametrize(
    ("gate_id", "mutate"),
    [
        ("P0.1", lambda item: item.update(membership_count=6546)),
        ("P0.2", lambda item: item.update(predecessor_historical_status="PASS")),
        ("P0.4", lambda item: item.update(public_member_redistribution_allowed=True)),
        ("P0.5", lambda item: item["required_fields_exactly"].remove("biological_standard_error")),
        ("P0.7", lambda item: item.update(split_assignment_count=1)),
        ("P0.8", lambda item: item.update(optimizer_fit_count=2)),
        ("P0.9", lambda item: item.update(any_nonpass_action="CONTINUE")),
        ("P0.10", lambda item: item.update(current_activation_state="ACTIVE_FOR_THIS_G1_ONE_RUN_ONLY")),
        ("P0.11", lambda item: item["persistent_locks"].update(a7_allowed=True)),
    ],
)
def test_any_semantic_nonpass_stops_before_data_cuda_model(gate_id, mutate) -> None:
    module = _module()
    config = deepcopy(_config())
    mutate(config["current_submission"][gate_id])
    result = module.evaluate(config)
    status = next(item["status"] for item in result["gate_statuses"] if item["gate_id"] == gate_id)
    assert status.startswith("FAIL_CLOSED_METADATA_MISMATCH")
    assert result["overall_status"] == "STOP_BEFORE_DATA_ROWS_CUDA_MODEL_OPTIMIZER_CHECKPOINT_PARAMETER_UPDATE_OR_TRAINING"
    assert result["materialization_authority_granted"] is False
    assert result["g1_launched"] is False
    assert result["forbidden_touchpoint_count"] == 0


def test_p07_is_contract_only_and_p08_is_one_fit() -> None:
    submission = _config()["current_submission"]
    assert submission["P0.7"]["split_assignment_count"] == 0
    assert submission["P0.7"]["model_or_endpoint_result_used"] is False
    assert [submission["P0.8"][key] for key in (
        "authorized_execution_count", "optimizer_fit_count", "fold_model_count",
        "checkpoint_count", "final_refit_count", "seed_count",
    )] == [1, 1, 1, 1, 0, 1]
    assert submission["P0.8"]["automatic_retry_allowed"] is False


def test_p010_binds_the_a0_clean_deferred_import_implementation() -> None:
    config = _config()
    evidence = config["current_submission"]["P0.10"]
    assert evidence["implementation_commit"] == "20a4198eb022e7a8abc63a4dd763f0dc154c3488"
    assert config["previous_implementation_binding"]["production_execution_count"] == 0
    assert config["previous_implementation_binding"]["binding_commit"] == "d8d51ef66c79373dcd0064b572676422bc1f2ca3"


def test_publish_is_exactly_once(tmp_path: Path) -> None:
    module = _module()
    result = module.evaluate(_config())
    target = module.publish(result, tmp_path / "p0")
    assert json.loads(target.read_text(encoding="utf-8"))["status_counts"] == {"PASS": 11, "NONPASS": 0}
    with pytest.raises(module.ProtocolError, match="exactly-once"):
        module.publish(result, tmp_path / "p0")
