from __future__ import annotations

import copy

import pytest

from scripts.route_a_v3.score_route2_xeditflow_closed_controls_v4 import (
    first_order_candidate_scores_v4,
    validate_closed_control_score_config_v4,
)


def _config() -> dict:
    prefix = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2"
    return {
        "schema_version": "route_a_v3_route2_xeditflow_closed_control_score_config.v4",
        "method_id": "first_order_guidance",
        "base_flow_training_seed": 20260913,
        "kappa": 0.5,
        "temperature": 1.0,
        "beta_max": 2.0,
        "critic_seeds": [20260908, 20260909, 20260910],
        "critic_refit_runtime_config_paths": {
            "20260908": f"{prefix}/a.json",
            "20260909": f"{prefix}/b.json",
            "20260910": f"{prefix}/c.json",
        },
        "critic_readiness_path": f"{prefix}/critic_ready.json",
        "setflow_confirmation_path": f"{prefix}/flow_ready.json",
        "critic_refit_manifest_path": f"{prefix}/refit.json",
        "mrnabert_model_path": f"{prefix}/mrnabert",
        "source_eligibility_manifest": f"{prefix}/sources.jsonl",
        "validation_projection_path": f"{prefix}/validation.jsonl",
        "measured_neighborhood_path": f"{prefix}/measured.jsonl",
        "guidance_screen_gate_path": f"{prefix}/guidance_gate.json",
        "output_dir": f"{prefix}/closed_scores/first_order",
        "pool_assignment": "DEVELOPMENT",
        "split": "VALIDATION",
        "expected_source_count": 891,
        "study_policy": "UNKNOWN_STUDY_SCALE_FIXED_1",
        "bottom_six_maximum_sequences_per_batch": 8,
        "bottom_six_batch_token_budget": 4096,
        "attention_backend": "PYTORCH_SDPA_AUTO",
        "physical_gpu_index": 4,
        "device": "cuda:4",
        "independent_evaluator_used": False,
        "measured_outcome_used_to_construct_score": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def test_v4_closed_control_config_freezes_three_methods_and_final_seeds() -> None:
    for method in (
        "first_order_guidance",
        "simple_rate_guidance",
        "generate_then_rerank",
    ):
        config = copy.deepcopy(_config())
        config["method_id"] = method
        config["base_flow_training_seed"] = 20260914
        validate_closed_control_score_config_v4(config)


def test_v4_closed_control_config_rejects_outcome_or_method_drift() -> None:
    config = _config()
    config["measured_outcome_used_to_construct_score"] = True
    with pytest.raises(Exception, match="protected-input"):
        validate_closed_control_score_config_v4(config)
    config = _config()
    config["method_id"] = "full_soft_value_smc"
    with pytest.raises(Exception, match="unknown"):
        validate_closed_control_score_config_v4(config)


def test_v4_first_order_closed_score_matches_source_anchored_additive_potential() -> None:
    rewards = {
        "AAA": 0.2,
        "CAA": 0.7,
        "ACA": -0.1,
        "AAC": 0.5,
    }
    scores = first_order_candidate_scores_v4(
        "AAA",
        ["AAA", "CAA", "CAC"],
        single_state_rewards=rewards,
    )
    assert scores == pytest.approx([0.0, 0.5, 0.8])


def test_v4_first_order_closed_score_requires_every_single_edit_coefficient() -> None:
    with pytest.raises(Exception, match="lacks a single-edit"):
        first_order_candidate_scores_v4(
            "AAA",
            ["CAA"],
            single_state_rewards={"AAA": 0.0},
        )
