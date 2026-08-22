from __future__ import annotations

import pytest

from scripts.route_a_v3.prepare_route2_xeditflow_guidance_screen_input_v3 import (
    GuidanceScreenInputV3Error,
    build_guidance_screen_prepare_input_v3,
)


def _critic() -> dict:
    return {
        "status": "CRITIC_READY_FOR_GUIDANCE",
        "frozen_test_passed": True,
        "all_development_refit_complete": True,
        "loso_readiness_passed": True,
    }


def _flow() -> dict:
    return {
        "status": "XEDITSETFLOW_V3_CONFIRMATION_PASS",
        "flow_status": "FLOW_G0_READY",
        "selected_arm": "f2",
    }


def _runtime() -> dict:
    return {
        "seed": 20260904,
        "selected_arm": "f2",
        "source_token_cache_path": "/mnt/cache.pt",
        "experiment_ledger_path": "/mnt/ledger.csv",
        "output_root": "/mnt/run",
        "source_eligibility_manifest": "/mnt/sources.jsonl",
        "validation_projection_path": "/mnt/validation.jsonl",
        "measured_neighborhood_path": "/mnt/measured.jsonl",
    }


def _refit() -> dict:
    return {
        "status": "XEDITCRITIC_V3_ALL_DEVELOPMENT_REFIT_COMPLETE",
        "checkpoints": [
            {"seed": seed, "checkpoint_path": f"/mnt/critic_{seed}.pt"}
            for seed in (20260831, 20260901, 20260902)
        ],
    }


def _summary() -> dict:
    return {
        "status": "XEDITFLOW_V3_VALUE_ROLLOUTS_COMPLETE",
        "base_flow_training_seed": 20260904,
        "setflow_arm": "f2",
        "rollouts_per_state": 8,
        "study_neutral": True,
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
        "state_path": "/mnt/states.jsonl",
        "frozen_rollout_score_path": "/mnt/scores.jsonl",
    }


def test_guidance_screen_input_binds_terminal_rollouts_and_seed_20260904() -> None:
    protocol = {
        "schema_version": "route_a_v3_route2_xeditflow_v3_guidance_protocol.v1",
        "guidance_screen_output_root": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/guidance",
        "critic_readiness_path": "/mnt/critic.json",
        "setflow_confirmation_path": "/mnt/flow.json",
        "decoder_seed_base": 20261001,
        "independent_evaluator_checkpoint_path": "/mnt/evaluator.pt",
        "strongest_generation_baseline_path": "/mnt/strongest.json",
        "baseline_selection_input_path": "/mnt/baseline_input.json",
        "independent_evaluator_bootstrap_iterations": 10_000,
        "critic_refit_manifest_path": "/mnt/refit.json",
        "mrnabert_model_path": "/mnt/mrnabert",
    }
    result = build_guidance_screen_prepare_input_v3(
        protocol, _critic(), _flow(), _refit(), _runtime(), _summary(), physical_gpu_index=3
    )
    assert result["train_state_path"] == "/mnt/states.jsonl"
    assert result["frozen_rollout_score_path"] == "/mnt/scores.jsonl"
    assert result["setflow_checkpoint_path"] == "/mnt/run/f2/best.pt"
    assert result["base_flow_training_seed"] == 20260904


def test_guidance_screen_input_rejects_nonterminal_value_rollouts() -> None:
    summary = {**_summary(), "status": "RUNNING"}
    with pytest.raises(GuidanceScreenInputV3Error, match="incomplete"):
        build_guidance_screen_prepare_input_v3(
            {
                "schema_version": "route_a_v3_route2_xeditflow_v3_guidance_protocol.v1",
                "guidance_screen_output_root": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/guidance",
            },
            _critic(),
            _flow(),
            _refit(),
            _runtime(),
            summary,
            physical_gpu_index=0,
        )
