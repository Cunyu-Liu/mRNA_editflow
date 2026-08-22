from __future__ import annotations

import json

from core.route2_xeditflow_gate_v3 import GUIDANCE_GRID_V3
from scripts.route_a_v3.prepare_route2_xeditflow_guidance_screen_configs_v3 import build_guidance_screen_configs_v3


def test_prepare_guidance_screen_has_six_values_and_exact_eighteen_grid(tmp_path) -> None:
    critic = {"status": "CRITIC_READY_FOR_GUIDANCE", "frozen_test_passed": True, "all_development_refit_complete": True, "loso_readiness_passed": True}
    flow = {"status": "XEDITSETFLOW_V3_CONFIRMATION_PASS", "flow_status": "FLOW_G0_READY"}
    for name, payload in (("critic.json", critic), ("flow.json", flow)):
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    config = {
        "schema_version": "route_a_v3_route2_xeditflow_guidance_screen_prepare.v1",
        "base_flow_training_seed": 20260904,
        "setflow_arm": "f2",
        "physical_gpu_index": 2,
        "output_root": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/xeditflow_v3",
        "critic_readiness_path": str(tmp_path / "critic.json"),
        "setflow_confirmation_path": str(tmp_path / "flow.json"),
        "train_state_path": "/mnt/train_states.jsonl",
        "frozen_rollout_score_path": "/mnt/rollouts.jsonl",
        "source_token_cache_path": "/mnt/cache.pt",
        "experiment_ledger_path": "/mnt/ledger.csv",
        "setflow_checkpoint_path": "/mnt/f2.pt",
        "source_eligibility_manifest": "/mnt/sources.jsonl",
        "validation_projection_path": "/mnt/validation.jsonl",
        "measured_neighborhood_path": "/mnt/measured.jsonl",
        "decoder_seed_base": 20261001,
        "guiding_checkpoint_path": "/mnt/critic_refit.pt",
        "independent_evaluator_checkpoint_path": "/mnt/evaluator.pt",
        "strongest_generation_baseline_path": "/mnt/strongest.json",
        "baseline_selection_input_path": "/mnt/baseline_input.json",
        "independent_evaluator_bootstrap_iterations": 10_000,
        "critic_refit_manifest_path": "/mnt/refit.json",
        "mrnabert_model_path": "/mnt/mrnabert",
    }
    payload = build_guidance_screen_configs_v3(config)
    assert payload["value_job_count"] == 6
    assert payload["guidance_combination_count"] == 18
    assert {tuple(row["combination"]) for row in payload["guidance_jobs"]} == set(GUIDANCE_GRID_V3)
    assert {row["training_config"]["base_flow_training_seed"] for row in payload["value_jobs"]} == {20260904}
    assert all(row["smc_config"]["reserved_terminal_critic_forwards"] == 3 for row in payload["guidance_jobs"])
    assert all(row["open_metric_config"]["undefined_outcome_policy"] == "UNKNOWN_NOT_ZERO" for row in payload["guidance_jobs"])
    assert all(row["independent_evaluator_config"]["evaluation_outcomes_used_to_select_evaluator"] == 0 for row in payload["guidance_jobs"])
    assert payload["additional_grid_combination_authorized"] is False


def test_prepare_guidance_screen_blocks_without_loso_readiness(tmp_path) -> None:
    critic = {"status": "CRITIC_READY_FOR_GUIDANCE", "frozen_test_passed": True, "all_development_refit_complete": True, "loso_readiness_passed": False}
    flow = {"status": "XEDITSETFLOW_V3_CONFIRMATION_PASS", "flow_status": "FLOW_G0_READY"}
    for name, payload in (("critic.json", critic), ("flow.json", flow)):
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    try:
        build_guidance_screen_configs_v3({
            "schema_version": "route_a_v3_route2_xeditflow_guidance_screen_prepare.v1",
            "base_flow_training_seed": 20260904,
            "setflow_arm": "f2",
            "physical_gpu_index": 0,
            "output_root": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/x",
            "critic_readiness_path": str(tmp_path / "critic.json"),
            "setflow_confirmation_path": str(tmp_path / "flow.json"),
        })
    except Exception as exc:
        assert "remain blocked" in str(exc)
    else:
        raise AssertionError("guidance configs were prepared before readiness")
