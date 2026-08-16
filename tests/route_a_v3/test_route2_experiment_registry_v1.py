from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs/route_a_v3_route2_experiment_registry_v1.json"
CAPACITY = ROOT / "configs/route_a_v3_route2_delta_capacity_profiles_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_registry_is_frozen_before_runs_and_keeps_test_and_evaluation_closed() -> None:
    registry = _load(REGISTRY)
    assert registry["status"] == "FROZEN_BEFORE_FORMAL_DEVELOPMENT_RUNS"
    assert registry["data_access"] == {
        "hpo_fit_split": "TRAIN",
        "hpo_selection_split": "VALIDATION",
        "development_test_access": "ONE_FROZEN_REFIT_AFTER_ALL_SELECTION",
        "evaluation_access": "ZERO_SHOT_ONLY_AFTER_PREDICTOR_GENERATOR_AND_BASELINES_FROZEN",
        "evaluation_outcomes_used_for_training_hpo_or_selection": 0,
    }
    assert registry["selection"]["development_test_reselection_allowed"] is False
    assert registry["selection"]["evaluation_reselection_allowed"] is False
    assert registry["scientific_claim_status"] == "NOT_ESTABLISHED"


def test_three_final_seeds_are_fixed_without_best_seed_selection() -> None:
    registry = _load(REGISTRY)
    seeds = registry["seeds"]["final_seeds"]
    assert len(seeds) == len(set(seeds)) == 3
    assert registry["seeds"]["best_seed_selection_allowed"] is False
    assert registry["loso"]["final_seed_count"] == len(seeds)
    assert registry["loso"]["expected_nonempty_development_study_count"] == 7
    assert registry["loso"]["zero_record_development_studies"] == ["GSE256185"]


def test_capacity_and_neural_baseline_profiles_are_bound_to_existing_geometry() -> None:
    registry = _load(REGISTRY)
    capacity = _load(CAPACITY)
    available = {row["profile_id"] for row in capacity["profiles"]}
    assert set(registry["delta_predictor"]["capacity_profile_ids"]) == available
    historical = registry["delta_predictor"]["historical_diagnostic_reference"]
    assert historical["parameter_count"] == 81794
    assert historical["replace_with_new_capacity_profile_allowed"] is False
    assert registry["delta_predictor"]["position_aware_diagnostic_profile_relation"].endswith(
        "NOT_EXACT_HISTORICAL_REPLICATION"
    )
    baseline_profiles = capacity["architecture_controlled_neural_baseline_profiles"]
    configured_kinds = set(registry["architecture_controlled_neural_baselines"]["model_kinds"])
    for profile_id in registry["architecture_controlled_neural_baselines"]["capacity_profile_ids"]:
        assert {row["model_kind"] for row in baseline_profiles[profile_id]} == configured_kinds


def test_all_parameter_updates_require_cuda_and_guided_remains_dual_gated() -> None:
    registry = _load(REGISTRY)
    assert registry["neural_common_budget"]["cuda_required"] is True
    assert registry["neural_common_budget"]["cpu_fallback_allowed"] is False
    assert registry["classical_baselines"]["fit_and_prediction_cuda_required"] is True
    assert registry["base_flow_g0"]["cuda_required"] is True
    assert registry["base_flow_g0"]["cpu_fallback_allowed"] is False
    assert registry["guided_generation"] == {
        "required_readiness_states": ["CRITIC_READY_FOR_GUIDANCE", "FLOW_G0_READY"],
        "start_before_both_ready_allowed": False,
    }
