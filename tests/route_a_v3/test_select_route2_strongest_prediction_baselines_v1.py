from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/select_route2_strongest_prediction_baselines_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("route2_baseline_selection_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _entry(baseline_id: str, spearman: float, mae: float, parameters: int):
    return {
        "baseline_id": baseline_id,
        "baseline_family": "TEST",
        "parameter_count": parameters,
        "evaluation": {
            "split": "VALIDATION",
            "evaluation_release_state": "CLOSED",
            "metrics": {"task_numeric": {"5UTR|MRL": {"spearman": spearman, "mae": mae, "record_count": 20}}},
        },
    }


def _payload():
    return {
        "schema_version": "route_a_v3_route2_baseline_selection_input.v1",
        "selection_pool": "DEVELOPMENT_VALIDATION",
        "evaluation_outcomes_accessed": False,
        "comparison_policy": "POINT_LEADER_VS_SMALLER_FINITE",
        "baseline_evaluations": [_entry("ridge", 0.2, 0.3, 10), _entry("cnn", 0.4, 0.2, 1000)],
        "paired_validation_bootstrap": [{
            "task": "5UTR|MRL", "split": "VALIDATION",
            "left_baseline_id": "cnn", "right_baseline_id": "ridge",
            "spearman_difference_ci_95": [0.05, 0.35],
            "defined_bootstrap_iterations": 2000,
        }],
    }


def test_selects_actual_best_spearman_and_keeps_candidates() -> None:
    module = _load()
    result = module.select(_payload())
    selected = result["tasks"]["5UTR|MRL"]
    assert selected["strongest_baseline_id"] == "cnn"
    assert [row["baseline_id"] for row in selected["all_candidates_ranked"]] == ["cnn", "ridge"]
    assert result["unseen_endpoint_fallbacks"]["regions"]["5UTR"]["strongest_baseline_id"] == "cnn"


def test_exact_spearman_tie_prefers_smaller_model() -> None:
    module = _load()
    payload = _payload()
    payload["baseline_evaluations"] = [_entry("large", 0.4, 0.1, 1000), _entry("small", 0.4, 0.2, 10)]
    payload["paired_validation_bootstrap"] = [{
        "task": "5UTR|MRL", "split": "VALIDATION",
        "left_baseline_id": "large", "right_baseline_id": "small",
        "spearman_difference_ci_95": [-0.1, 0.1], "defined_bootstrap_iterations": 2000,
    }]
    result = module.select(payload)
    assert result["tasks"]["5UTR|MRL"]["strongest_baseline_id"] == "small"


def test_evaluation_access_fails_closed() -> None:
    module = _load()
    payload = deepcopy(_payload())
    payload["evaluation_outcomes_accessed"] = True
    with pytest.raises(module.BaselineSelectionError, match="Evaluation"):
        module.select(payload)


def test_constant_control_with_undefined_spearman_is_retained_but_not_selected() -> None:
    module = _load()
    payload = _payload()
    payload["baseline_evaluations"].append(_entry("global_mean", None, 0.5, 0))
    result = module.select(payload)
    task = result["tasks"]["5UTR|MRL"]
    assert task["strongest_baseline_id"] == "cnn"
    assert task["all_candidates_ranked"][-1]["baseline_id"] == "global_mean"
    assert task["finite_spearman_candidate_count"] == 2


def test_bootstrap_indistinguishable_point_leader_yields_to_smaller_model() -> None:
    module = _load()
    payload = _payload()
    payload["paired_validation_bootstrap"][0]["spearman_difference_ci_95"] = [-0.02, 0.05]
    result = module.select(payload)["tasks"]["5UTR|MRL"]
    assert result["strongest_baseline_id"] == "ridge"
    assert result["bootstrap_uncertainty_equivalent_baseline_ids"] == ["cnn", "ridge"]


def test_missing_pairwise_bootstrap_fails_closed() -> None:
    module = _load()
    payload = _payload()
    payload["paired_validation_bootstrap"] = []
    with pytest.raises(module.BaselineSelectionError, match="bootstrap is absent"):
        module.select(payload)


def test_larger_candidate_does_not_require_bootstrap_to_keep_point_leader() -> None:
    payload = _payload()
    payload["baseline_evaluations"][1]["parameter_count"] = 10
    payload["baseline_evaluations"][0]["parameter_count"] = 1000
    payload["paired_validation_bootstrap"] = []
    selected = _load().select(payload)["tasks"]["5UTR|MRL"]
    assert selected["strongest_baseline_id"] == "cnn"
    assert selected["bootstrap_decision_relevant_candidate_count"] == 0


def test_unseen_endpoint_fallback_requires_complete_task_coverage() -> None:
    module = _load()
    ridge = _entry("ridge", 0.2, 0.3, 10)
    ridge["evaluation"]["metrics"]["task_numeric"]["5UTR|TE"] = {
        "spearman": 0.3, "mae": 0.2, "record_count": 20,
    }
    specialist = _entry("specialist", 0.9, 0.1, 100)
    fallback = module.complete_coverage_fallback([ridge, specialist], {"5UTR|MRL", "5UTR|TE"})
    assert fallback["strongest_baseline_id"] == "ridge"
    assert fallback["complete_coverage_candidate_count"] == 1
