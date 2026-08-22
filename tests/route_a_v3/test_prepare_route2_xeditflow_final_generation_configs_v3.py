from __future__ import annotations

import pytest

from scripts.route_a_v3.prepare_route2_xeditflow_final_generation_configs_v3 import (
    build_final_generation_configs_v3,
)


def test_final_generation_prepares_three_seeds_without_second_hpo() -> None:
    config = {
        "schema_version": "route_a_v3_route2_xeditflow_final_generation_prepare.v1",
        "critic_readiness_path": "/mnt/critic.json",
        "setflow_confirmation_path": "/mnt/flow.json",
        "critic_refit_manifest_path": "/mnt/refit.json",
        "mrnabert_model_path": "/mnt/model",
        "guidance_screen_output_root": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/screen",
        "guidance_screen_gate_path": "/mnt/guidance_gate.json",
        "output_root": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/final",
        "decoder_seed_base": 20261001,
        "physical_gpu_index": 2,
        "measured_neighborhood_path": "/mnt/measured.jsonl",
        "strongest_generation_baseline_path": "/mnt/strongest.json",
        "baseline_selection_input_path": "/mnt/selection.json",
        "independent_evaluator_checkpoint_path": "/mnt/evaluator.pt",
    }
    critic = {
        "status": "CRITIC_READY_FOR_GUIDANCE", "frozen_test_passed": True,
        "all_development_refit_complete": True, "loso_readiness_passed": True,
    }
    flow = {
        "status": "XEDITSETFLOW_V3_CONFIRMATION_PASS", "flow_status": "FLOW_G0_READY",
        "selected_arm": "f2",
    }
    refit = {
        "status": "XEDITCRITIC_V3_ALL_DEVELOPMENT_REFIT_COMPLETE",
        "checkpoints": [
            {"seed": seed, "checkpoint_path": f"/mnt/critic_{seed}.pt"}
            for seed in (20260831, 20260901, 20260902)
        ],
    }
    screen = {
        "status": "XEDITFLOW_V3_GUIDANCE_SCREEN_FROZEN",
        "base_flow_training_seed": 20260904,
        "selected_kappa": 0.5, "selected_temperature": 1.0, "selected_beta_max": 2.0,
    }
    strongest = {
        "status": "DEVELOPMENT_STRONGEST_GENERATION_BASELINE_FROZEN_INDEPENDENT_EVALUATOR_ONLY",
        "strongest_generation_baseline_id": "genetic",
        "evaluation_outcomes_accessed": False,
        "forward_equivalent_budget_per_source": 320,
        "independent_evaluator_checkpoint_path": "/mnt/evaluator.pt",
    }
    runtimes = {
        seed: {
            "seed": seed, "selected_arm": "f2", "output_root": f"/mnt/run/{seed}",
            "train_projection_path": "/mnt/train", "source_token_cache_path": "/mnt/cache",
            "source_eligibility_manifest": "/mnt/sources", "validation_projection_path": "/mnt/validation",
            "experiment_ledger_path": "/mnt/ledger",
        }
        for seed in (20260904, 20260905, 20260906)
    }
    manifest = build_final_generation_configs_v3(
        config,
        critic_readiness=critic,
        setflow_confirmation=flow,
        critic_refit_manifest=refit,
        guidance_screen_gate=screen,
        strongest_generation_baseline=strongest,
        setflow_runtimes=runtimes,
    )
    assert manifest["required_base_flow_training_seeds"] == [20260904, 20260905, 20260906]
    assert len(manifest["seed_jobs"]) == 3
    assert manifest["seed_jobs"][0]["value_rollout_config"] is None
    assert manifest["seed_jobs"][1]["value_rollout_config"]["base_flow_training_seed"] == 20260905
    assert manifest["seed_jobs"][2]["value_training_config"]["checkpoint_selection"] == "FINAL_PASS_NO_EPOCH_RESELECTION"
    assert all(len(job["matched_control_configs"]) == 4 for job in manifest["seed_jobs"])
    assert all(job["full_smc_config"]["decoder_seed_base"] == 20261001 for job in manifest["seed_jobs"])
    assert all(set(job["closed_trajectory_configs"]) == {
        "full_soft_value_smc", "unguided_setflow"
    } for job in manifest["seed_jobs"])
    assert all(job["closed_trajectory_configs"]["unguided_setflow"]["potential_kind"] == "ZERO" for job in manifest["seed_jobs"])
    assert all(set(job["closed_frozen_score_configs"]) == {
        "first_order_guidance", "simple_rate_guidance", "generate_then_rerank", "strongest_matched_baseline"
    } for job in manifest["seed_jobs"])
    assert manifest["seed_jobs"][0]["closed_score_metric_configs"]["strongest_matched_baseline"]["score_transform"] == "SOURCEWISE_EXP_SHIFTED_MAX"
    assert manifest["seed_jobs"][0]["strongest_adapter_job"]["base_flow_training_seed"] == 20260904
    assert all(len(job["open_metric_configs"]) == 5 for job in manifest["seed_jobs"])
    assert manifest["seed_jobs"][0]["independent_evaluator_config"]["guiding_checkpoint_paths"] == [
        "/mnt/critic_20260831.pt", "/mnt/critic_20260901.pt", "/mnt/critic_20260902.pt"
    ]
    assert all(len(job["final_seed_evidence_config"]["methods"]) == 6 for job in manifest["seed_jobs"])
    assert manifest["seed_jobs"][0]["final_seed_evidence_config"]["methods"]["strongest_matched_baseline"]["closed_summary_path"].endswith("closed_strongest_matched_baseline.json")
    assert len(manifest["three_seed_finalization"]["seed_manifest_row_paths"]) == 3
    assert manifest["three_seed_finalization"]["replacement_evaluation_authorized_only_by_final_adjudication"] is True
    assert manifest["additional_seed_authorized"] is False
    drifted = dict(strongest)
    drifted["independent_evaluator_checkpoint_path"] = "/mnt/different_evaluator.pt"
    with pytest.raises(Exception, match="independent evaluator differs"):
        build_final_generation_configs_v3(
            config,
            critic_readiness=critic,
            setflow_confirmation=flow,
            critic_refit_manifest=refit,
            guidance_screen_gate=screen,
            strongest_generation_baseline=drifted,
            setflow_runtimes=runtimes,
        )
