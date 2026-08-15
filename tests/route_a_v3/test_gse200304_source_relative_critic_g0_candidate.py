from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/route_a_v3/gse200304_source_relative_critic_g0_candidate.py"
CONFIG_PATH = REPO_ROOT / "configs/route_a_v3_gse200304_source_relative_critic_g0_candidate_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("critic_g0_candidate", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_config_is_inactive_and_zero_update() -> None:
    module = _module()
    config = module.load_config(CONFIG_PATH)
    result = module.validate_only(config)
    assert result["status"] == "PASS_STATIC_SYNTHETIC_ZERO_UPDATE_PREPARATION_ONLY_NOT_ACTIVE"
    assert result["future_run_authorized"] is False
    assert result["future_run_id"] == "GSE200304_SOURCE_RELATIVE_CRITIC_G1"
    assert set(result["validate_only_truth"].values()) == {0}
    assert result["scientific_claim_status"] == "NOT_ESTABLISHED"


def test_direction_normalized_candidate_minus_source_and_identity() -> None:
    module = _module()
    assert module.direction_normalized_effect(
        source_endpoint=2.0, candidate_endpoint=5.0, direction_multiplier=1
    ) == 3.0
    assert module.direction_normalized_effect(
        source_endpoint=2.0, candidate_endpoint=5.0, direction_multiplier=-1
    ) == -3.0
    assert module.direction_normalized_effect(
        source_endpoint=2.0, candidate_endpoint=2.0, direction_multiplier=1
    ) == 0.0


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_missing_or_nonfinite_is_never_zero_imputed(value: float) -> None:
    module = _module()
    with pytest.raises(module.ContractError, match="never zero-imputed"):
        module.direction_normalized_effect(
            source_endpoint=1.0, candidate_endpoint=value, direction_multiplier=1
        )


def test_antisymmetry_and_lcb_formulas_are_pure() -> None:
    module = _module()
    forward = module.antisymmetric_pair_mean(forward_score=3.0, reverse_score=1.0)
    reverse = module.antisymmetric_pair_mean(forward_score=1.0, reverse_score=3.0)
    identity = module.antisymmetric_pair_mean(forward_score=2.0, reverse_score=2.0)
    assert forward == 1.0
    assert reverse == -forward
    assert identity == 0.0
    assert module.calibrated_lower_confidence_bound(
        mean=0.8, predictive_scale=0.2, calibration_quantile=1.5
    ) == pytest.approx(0.5)


def test_future_execution_shape_is_exactly_one_fit_without_selection() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    run = config["future_single_fit_contract"]
    assert {
        "authorized_execution_count": run["authorized_execution_count"],
        "optimizer_fit_count": run["optimizer_fit_count"],
        "fold_model_count": run["fold_model_count"],
        "checkpoint_count": run["checkpoint_count"],
        "final_refit_count": run["final_refit_count"],
        "seed_count": run["seed_count"],
    } == {
        "authorized_execution_count": 1,
        "optimizer_fit_count": 1,
        "fold_model_count": 1,
        "checkpoint_count": 1,
        "final_refit_count": 0,
        "seed_count": 1,
    }
    assert run["nested_cross_validation_authorized"] is False
    assert run["checkpoint_policy"] == "TERMINAL_CHECKPOINT_ONLY"
    assert run["early_stopping_allowed"] is False
    assert run["best_checkpoint_selection_allowed"] is False
    assert run["hyperparameter_search_allowed"] is False
    assert run["automatic_retry_allowed"] is False


def test_evaluator_is_isolated_and_real_split_is_not_generated() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    boundary = config["split_and_evaluator_boundary"]
    assert boundary["real_membership_frozen"] is False
    assert boundary["real_split_assignments_generated"] is False
    assert boundary["split_assignment_count"] == 0
    assert boundary["evaluator_receives_guide_output"] is False
    assert boundary["evaluator_selects_model_or_checkpoint"] is False
    assert boundary["test_selects_threshold"] is False


def test_cli_validate_only_writes_stdout_and_no_artifact(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--config", str(CONFIG_PATH), "--validate-only"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["future_run_authorized"] is False
    assert list(tmp_path.iterdir()) == []


def test_candidate_has_no_learned_or_device_dependency() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import tensorflow" not in source
    assert "torch.cuda" not in source
    assert "DataLoader" not in source
