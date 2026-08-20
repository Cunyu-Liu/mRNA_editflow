from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from core.route2_legal_xeditflow import (
    STOP,
    initial_state,
    legal_actions,
    sample_trajectory,
)
from scripts.route_a_v3 import run_route2_guided_xeditflow_v1 as runner


class FakeCritic:
    def potentials(self, states, *, endpoint_id, region):
        assert endpoint_id == "E"
        assert region == "3UTR"
        values = {"A": 0.0, "C": 2.0, "G": 1.0, "U": -1.0}
        return [values[state.current_sequence] for state in states]


def _readiness():
    return {
        "schema_version": runner.READINESS_INPUT_SCHEMA,
        "critic": {
            "online_encoder_validation": {
                "status": "ONLINE_FROZEN_MRNABERT_MATCHES_CANONICAL_CACHE",
                "novel_candidate_encoding_supported": True,
                "evaluation_records_read": 0,
            },
            "reward_policy": {
                "evaluation_records_used_for_training_hpo_threshold_or_reward": 0,
            },
            "refit_checkpoint": str(runner.EXPECTED_CRITIC_CHECKPOINT),
        },
        "flow": {"checkpoint": "/flow.pt"},
        "guided_generation_executed": False,
        "evaluation_opened_by_readiness_builder": False,
    }


def _adjudication():
    return {
        "schema_version": runner.READINESS_ADJUDICATION_SCHEMA,
        "critic_status": "CRITIC_READY_FOR_GUIDANCE",
        "flow_status": "FLOW_G0_READY",
        "guided_generation_status": "GUIDED_XEDITFLOW_DEVELOPMENT_ALLOWED",
        "guided_unlocked": True,
        "guided_generation_executed": False,
        "evaluation_opened": False,
        "biological_optimization_established": False,
    }


def _config():
    return {
        "schema_version": runner.GUIDED_CONFIG_SCHEMA,
        "status": "WAITING_FOR_CRITIC_V2_AND_FLOW_READINESS",
        "seed": 20260825,
        "readiness_input_path": str(runner.EXPECTED_READINESS_INPUT),
        "readiness_adjudication_path": str(
            runner.EXPECTED_READINESS_ADJUDICATION
        ),
        "critic_checkpoint_path": str(runner.EXPECTED_CRITIC_CHECKPOINT),
        "base_flow_checkpoint_path": "/flow.pt",
        "matched_search_budget_rule": (
            "GUIDED_TOTAL_FORWARD_EQUIVALENTS_AS_SEARCH_CRITIC_CAP_PER_SOURCE"
        ),
        "evaluation_outcomes_accessed": False,
        "generated_candidates_grant_canonical_credit": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }


def test_readiness_must_unlock_exact_bound_checkpoints() -> None:
    runner.validate_guided_config(_config())
    runner.validate_readiness(_readiness(), _adjudication(), _config())
    adjudication = _adjudication()
    adjudication["guided_unlocked"] = False
    with pytest.raises(runner.GuidedRunError, match="not ready"):
        runner.validate_readiness(_readiness(), adjudication, _config())
    config = _config()
    config["critic_checkpoint_path"] = "/other.pt"
    with pytest.raises(runner.GuidedRunError, match="artifact binding"):
        runner.validate_guided_config(config)
    with pytest.raises(runner.GuidedRunError, match="critic path differs"):
        runner.validate_readiness(_readiness(), _adjudication(), config)


def test_historical_v1_config_is_rejected_before_artifact_access() -> None:
    config = _config()
    config["schema_version"] = "route_a_v3_route2_guided_xeditflow_development.v1"
    with pytest.raises(runner.GuidedRunError, match="historical"):
        runner.validate_guided_config(config)


def test_repository_config_uses_only_critic_v2_readiness() -> None:
    root = Path(__file__).resolve().parents[2]
    current = json.loads(
        (
            root
            / "configs/route_a_v3_route2_mrnabert_critic_v2_guided_xeditflow_development_gpu0_v1.json"
        ).read_text(encoding="utf-8")
    )
    historical = json.loads(
        (
            root / "configs/route_a_v3_route2_guided_xeditflow_development_gpu0_v1.json"
        ).read_text(encoding="utf-8")
    )
    runner.validate_guided_config(current)
    assert historical["status"] == "RETIRED_HISTORICAL_V1_READINESS_PATH_NOT_AUTHORIZED"
    with pytest.raises(runner.GuidedRunError, match="historical"):
        runner.validate_guided_config(historical)


def test_batched_guidance_matches_frozen_potential_difference() -> None:
    root = initial_state("A", budget=1, assay_id="a", context_id="c")
    base_calls = []

    def base(state, actions):
        base_calls.append(state)
        return {action: 1.0 for action in actions}

    counters = {}
    guided = runner.batched_guided_rate_function(
        base,
        FakeCritic(),
        endpoint_id="E",
        region="3UTR",
        guidance_strength=1.0,
        counters=counters,
    )
    rates = guided(root, legal_actions(root))
    by_id = {action.action_id: rates[action] for action in rates}
    assert by_id["SUB:0:C"] == pytest.approx(math.exp(2.0))
    assert by_id["SUB:0:G"] == pytest.approx(math.exp(1.0))
    assert by_id[STOP] == pytest.approx(1.0)
    assert by_id["SUB:0:U"] == pytest.approx(math.exp(-1.0))
    assert counters["base_flow_forwards"] == 1
    assert counters["guided_rate_requests"] == 1
    assert counters["unique_state_rate_evaluations"] == 1

    repeated = guided(root, legal_actions(root))
    assert repeated == rates
    assert base_calls == [root]
    assert counters["base_flow_forwards"] == 1
    assert counters["guided_rate_requests"] == 2
    assert counters["guided_rate_cache_hits"] == 1
    assert counters["unique_state_rate_evaluations"] == 1

    first = sample_trajectory(root, guided, seed=20260817)
    second = sample_trajectory(root, guided, seed=20260817)
    assert first == second
    assert counters["guided_rate_requests"] == 4
    assert counters["guided_rate_cache_hits"] == 3
    assert counters["unique_state_rate_evaluations"] == 1


def test_online_encoder_and_evaluation_are_required() -> None:
    readiness = _readiness()
    readiness["critic"]["online_encoder_validation"]["novel_candidate_encoding_supported"] = False
    with pytest.raises(runner.GuidedRunError, match="online"):
        runner.validate_readiness(readiness, _adjudication(), _config())


def test_backend_selection_requires_completed_non_evaluation_adjudication() -> None:
    value = {
        "schema_version": "route_a_v3_route2_mrnabert_sdpa_backend_adjudication.v1",
        "status": "ONLINE_ENCODER_BACKEND_ADJUDICATED",
        "selected_attention_backend": "PYTORCH_SDPA_AUTO",
        "evaluation_opened": False,
    }
    assert runner.selected_attention_backend(value) == "PYTORCH_SDPA_AUTO"
    value["evaluation_opened"] = True
    with pytest.raises(runner.GuidedRunError, match="backend adjudication"):
        runner.selected_attention_backend(value)


def test_guided_output_contract_contains_ranking_scores_for_later_evaluation() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert '"critic_score": terminal_critic_score' in source
    assert '"source_critic_score": source_critic_score' in source


def test_search_budget_matches_guided_total_not_only_critic() -> None:
    summary = runner.summarize_compute_rows([
        {
            "critic_candidate_forward_equivalent_count": 100,
            "generator_nfe": 7,
            "matched_search_critic_forward_budget": 107,
        },
        {
            "critic_candidate_forward_equivalent_count": 200,
            "generator_nfe": 11,
            "matched_search_critic_forward_budget": 211,
        },
    ])
    assert summary["critic_candidate_forward_equivalent_count"] == 300
    assert summary["total_forward_equivalent_count"] == 318
    assert summary["matched_search_budget_minimum"] == 107
    with pytest.raises(runner.GuidedRunError, match="does not close"):
        runner.summarize_compute_rows([
            {
                "critic_candidate_forward_equivalent_count": 100,
                "generator_nfe": 7,
                "matched_search_critic_forward_budget": 100,
            }
        ])
    readiness = _readiness()
    readiness["critic"]["reward_policy"][
        "evaluation_records_used_for_training_hpo_threshold_or_reward"
    ] = 1
    with pytest.raises(runner.GuidedRunError, match="Evaluation"):
        runner.validate_readiness(readiness, _adjudication(), _config())
