from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/route_a_v3/validate_route2_v1.py"
CONFIG_PATH = REPO_ROOT / "configs/route_a_v3_route2_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("validate_route2_v1", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_active_route2_config_validates_without_overstating_science() -> None:
    module = _module()
    result = module.validate_config(_config())
    assert result == {
        "status": "PASS_ROUTE2_V1_EXECUTION_CONFIG",
        "study_unit_count": 14,
        "development_study_count": 8,
        "evaluation_study_count": 2,
        "qualified_counts": "1/1/0/6547",
        "prediction_training_enabled": True,
        "flow_g0_implementation_enabled": True,
        "guided_generation_allowed": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }


def test_development_is_not_blocked_by_old_qualification_or_authority_workflows() -> None:
    module = _module()
    config = _config()
    policy = config["development_policy"]
    assert policy["full_route_a_qualification_is_development_gate"] is False
    assert policy["canonicalization_enabled"] is True
    assert policy["baseline_execution_enabled"] is True
    assert policy["prediction_training_enabled"] is True
    assert policy["flow_g0_implementation_enabled"] is True
    assert policy["training_requires_nvidia_gpu"] is True
    assert policy["training_cuda_fallback_to_cpu_allowed"] is False
    assert policy["successor_authority_required"] is False
    assert policy["runtime_ledger_required"] is False
    assert policy["one_read_required"] is False
    assert policy["resource_once_required"] is False
    module.validate_run_role(config, "DELTA_PREDICTOR")
    module.validate_run_role(config, "FLOW_G0_BASE")


def test_evaluation_and_credit_boundaries_are_frozen() -> None:
    module = _module()
    config = _config()
    module.validate_config(config)
    inventory = {item["study_unit_id"]: item for item in config["study_inventory"]}
    assert inventory["GSE232572"]["pool"] == "EVALUATION"
    assert inventory["E-MTAB-10902"]["pool"] == "EVALUATION"
    assert inventory["GSE246381"]["pool"] == "SEALED_EXCLUDED"
    assert inventory["GSE207584"]["canonical_records"] == 0
    assert inventory["GSE261709"]["canonical_records"] == 0
    assert not any(config["credit_policy"].values())


def test_guided_generation_requires_both_readiness_states() -> None:
    module = _module()
    config = _config()
    with pytest.raises(module.Route2ConfigError, match="dependencies are not met"):
        module.validate_run_role(config, "GUIDED_XEDITFLOW")

    critic_only = copy.deepcopy(config)
    critic_only["readiness"]["critic"] = "CRITIC_READY_FOR_GUIDANCE"
    assert module.guided_generation_allowed(critic_only) is False

    both_ready = copy.deepcopy(critic_only)
    both_ready["readiness"]["flow"] = "FLOW_G0_READY"
    assert module.guided_generation_allowed(both_ready) is True
    module.validate_run_role(both_ready, "GUIDED_XEDITFLOW")


def test_validator_rejects_evaluation_leakage_and_unsupported_credit() -> None:
    module = _module()

    leaked = _config()
    leaked["study_inventory"][8]["pool"] = "DEVELOPMENT"
    with pytest.raises(module.Route2ConfigError, match="Evaluation pool changed|Development pool changed"):
        module.validate_config(leaked)

    credited = _config()
    credited["credit_policy"]["generated_candidates_count_as_canonical"] = True
    with pytest.raises(module.Route2ConfigError, match="scientific credit"):
        module.validate_config(credited)
