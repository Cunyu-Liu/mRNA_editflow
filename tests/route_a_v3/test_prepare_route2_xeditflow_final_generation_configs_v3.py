from __future__ import annotations

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
        "output_root": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/final",
        "decoder_seed_base": 20261001,
        "physical_gpu_index": 2,
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
        "checkpoints": [{"seed": seed} for seed in (20260831, 20260901, 20260902)],
    }
    screen = {
        "status": "XEDITFLOW_V3_GUIDANCE_SCREEN_FROZEN",
        "base_flow_training_seed": 20260904,
        "selected_kappa": 0.5, "selected_temperature": 1.0, "selected_beta_max": 2.0,
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
        setflow_runtimes=runtimes,
    )
    assert manifest["required_base_flow_training_seeds"] == [20260904, 20260905, 20260906]
    assert len(manifest["seed_jobs"]) == 3
    assert manifest["seed_jobs"][0]["value_rollout_config"] is None
    assert manifest["seed_jobs"][1]["value_rollout_config"]["base_flow_training_seed"] == 20260905
    assert manifest["seed_jobs"][2]["value_training_config"]["checkpoint_selection"] == "FINAL_PASS_NO_EPOCH_RESELECTION"
    assert all(len(job["matched_control_configs"]) == 4 for job in manifest["seed_jobs"])
    assert all(job["full_smc_config"]["decoder_seed_base"] == 20261001 for job in manifest["seed_jobs"])
    assert manifest["additional_seed_authorized"] is False
