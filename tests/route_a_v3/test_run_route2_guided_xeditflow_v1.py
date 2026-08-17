from __future__ import annotations

import math
from pathlib import Path

import pytest

from core.route2_legal_xeditflow import STOP, initial_state, legal_actions
from scripts.route_a_v3 import run_route2_guided_xeditflow_v1 as runner


class FakeCritic:
    def potentials(self, states, *, endpoint_id, region):
        assert endpoint_id == "E"
        assert region == "3UTR"
        values = {"A": 0.0, "C": 2.0, "G": 1.0, "U": -1.0}
        return [values[state.current_sequence] for state in states]


def _readiness():
    return {
        "schema_version": "route_a_v3_route2_readiness_input.v1",
        "critic": {
            "generated_candidate_online_encoder_ready": True,
            "evaluation_records_used_for_training_hpo_threshold_or_reward": 0,
            "final_refit_checkpoint": "/critic.pt",
        },
        "flow": {"validation_checkpoint": "/flow.pt"},
    }


def _adjudication():
    return {
        "schema_version": "route_a_v3_route2_readiness_adjudication.v1",
        "critic_status": "CRITIC_READY_FOR_GUIDANCE",
        "flow_status": "FLOW_G0_READY",
        "guided_unlocked": True,
    }


def _config():
    return {
        "critic_checkpoint_path": "/critic.pt",
        "base_flow_checkpoint_path": "/flow.pt",
    }


def test_readiness_must_unlock_exact_bound_checkpoints() -> None:
    runner.validate_readiness(_readiness(), _adjudication(), _config())
    adjudication = _adjudication()
    adjudication["guided_unlocked"] = False
    with pytest.raises(runner.GuidedRunError, match="not ready"):
        runner.validate_readiness(_readiness(), adjudication, _config())
    config = _config()
    config["critic_checkpoint_path"] = "/other.pt"
    with pytest.raises(runner.GuidedRunError, match="critic path differs"):
        runner.validate_readiness(_readiness(), _adjudication(), config)


def test_batched_guidance_matches_frozen_potential_difference() -> None:
    root = initial_state("A", budget=1, assay_id="a", context_id="c")

    def base(_state, actions):
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


def test_online_encoder_and_evaluation_are_required() -> None:
    readiness = _readiness()
    readiness["critic"]["generated_candidate_online_encoder_ready"] = False
    with pytest.raises(runner.GuidedRunError, match="online"):
        runner.validate_readiness(readiness, _adjudication(), _config())


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
    readiness["critic"]["evaluation_records_used_for_training_hpo_threshold_or_reward"] = 1
    with pytest.raises(runner.GuidedRunError, match="Evaluation"):
        runner.validate_readiness(readiness, _adjudication(), _config())
