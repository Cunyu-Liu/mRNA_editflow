from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from core.route2_xeditflow_gate_v4 import authorize_xeditflow_guidance_v4
from scripts.route_a_v3.prepare_route2_xeditflow_v4_value_configs import (
    build_value_configs_v4,
    write_value_configs_v4,
)


ROOT = Path(__file__).resolve().parents[2]
GUIDANCE_PROTOCOL = (
    ROOT / "configs/route_a_v3_route2_xeditflow_v4_guidance_protocol_v1.json"
)
POSTTEST_PROTOCOL = (
    ROOT / "configs/route_a_v3_route2_xeditcritic_v4_posttest_protocol_v1.json"
)


def _critic_ready() -> dict:
    return {
        "status": "CRITIC_V4_READY_FOR_GUIDANCE",
        "three_seed_passed": True,
        "frozen_test_passed": True,
        "all_development_refit_complete": True,
        "loso_readiness_passed": True,
        "development_test_access_event_count": 1,
        "general_test_projection_persisted": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
        "guidance_authorized": True,
    }


def _setflow_ready() -> dict:
    return {
        "status": "XEDITSETFLOW_V4_G0_READY",
        "required_seeds": [20260912, 20260913, 20260914],
        "seed_results": {
            "20260912": {
                "passed": True,
                "selected_checkpoint_pass": 6,
            },
            "20260913": {
                "passed": True,
                "selected_checkpoint_pass": 8,
            },
            "20260914": {
                "passed": True,
                "selected_checkpoint_pass": 10,
            },
        },
        "critic_used": False,
        "independent_evaluator_used": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def _refit() -> dict:
    return {
        "status": "XEDITCRITIC_V4_ALL_DEVELOPMENT_REFIT_COMPLETE",
        "required_seeds": [20260908, 20260909, 20260910],
        "completed_refit_count": 3,
        "refit_pass_count": 8,
        "loso_authorized": True,
        "checkpoints": [
            {
                "seed": seed,
                "checkpoint_path": (
                    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"
                    f"experiments/xeditcritic_v4/refit-{seed}.pt"
                ),
                "physical_batch_size": 8,
            }
            for seed in (20260908, 20260909, 20260910)
        ],
        "development_test_outcomes_accessed_during_refit": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def _source_audit() -> dict:
    return {
        "status": "XEDITSETFLOW_V4_SOURCE_LEVEL_DATA_AUDIT_PASS",
        "train_source_count": 101,
        "validation_source_count": 891,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def _payload() -> dict:
    protocol = json.loads(GUIDANCE_PROTOCOL.read_text(encoding="utf-8"))
    critic = _critic_ready()
    flow = _setflow_ready()
    return build_value_configs_v4(
        protocol,
        authorize_xeditflow_guidance_v4(critic, flow),
        critic,
        flow,
        _refit(),
        _source_audit(),
        rollout_gpu=5,
        critic_gpu=4,
        value_gpus=(0, 1, 2, 3, 4, 5),
    )


def test_v4_guidance_protocol_uses_actual_readiness_and_refit_outputs() -> None:
    guidance = json.loads(GUIDANCE_PROTOCOL.read_text(encoding="utf-8"))
    posttest = json.loads(POSTTEST_PROTOCOL.read_text(encoding="utf-8"))
    assert guidance["critic_readiness_path"] == posttest["readiness_output"]
    assert (
        guidance["critic_refit_manifest_path"]
        == posttest["all_development_refit"]["terminal_manifest_output"]
    )


def test_v4_value_config_producer_emits_one_one_six_six_eighteen_exact_chain(
    tmp_path: Path,
) -> None:
    payload = _payload()
    assert payload["state_mode_count"] == 404
    assert payload["terminal_rollout_count"] == 3232
    assert payload["rollout_job_count"] == 1
    assert payload["critic_score_job_count"] == 1
    assert payload["value_target_package_count"] == 6
    assert payload["value_training_job_count"] == 6
    assert payload["later_guidance_combination_count"] == 18
    assert len(payload["guidance_jobs"]) == 18
    assert {
        tuple(row["combination"]) for row in payload["guidance_jobs"]
    } == {
        (kappa, temperature, beta)
        for kappa in (0.0, 0.5, 1.0)
        for temperature in (0.5, 1.0)
        for beta in (0.5, 1.0, 2.0)
    }
    assert all(
        row["smc_config"]["decoder_seed_base"] == 20261001
        and row["smc_config"]["physical_gpu_index"] == 5
        and row["smc_config"]["terminal_critic_forwards_by_member"]
        == [4, 4, 4]
        for row in payload["guidance_jobs"]
    )
    assert all(
        row["open_metric_config"]["method_id"]
        == row["smc_config"]["method_id"]
        and row["open_metric_config"]["critic_self_score_used_for_ranking"]
        is False
        for row in payload["guidance_jobs"]
    )
    assert all(
        row["closed_config"]["method_id"] == row["smc_config"]["method_id"]
        and row["closed_config"]["latent_mode_policy"]
        == "ROOT_PRIOR_WEIGHTED_SUM_OF_EIGHT_FIXED_MODE_TERMINAL_PROBABILITIES"
        for row in payload["guidance_jobs"]
    )
    assert all(
        row["critic_ensemble_config"]["kappa"] == row["combination"][0]
        and row["critic_ensemble_config"]
        ["critic_self_score_used_for_generation_or_selection"]
        is False
        for row in payload["guidance_jobs"]
    )
    expected_guiding = [
        "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"
        f"experiments/xeditcritic_v4/refit-{seed}.pt"
        for seed in (20260908, 20260909, 20260910)
    ]
    assert all(
        row["independent_evaluator_config"]["guiding_checkpoint_paths"]
        == expected_guiding
        and row["independent_evaluator_config"][
            "independent_evaluator_in_gradient"
        ]
        is False
        and row["independent_evaluator_comparison_config"][
            "strongest_baseline_path"
        ]
        == json.loads(GUIDANCE_PROTOCOL.read_text(encoding="utf-8"))[
            "strongest_generation_baseline_path"
        ]
        for row in payload["guidance_jobs"]
    )
    assert payload["beta_max_used_in_value_target_or_training"] is False
    assert payload["rollout_config"]["expected_train_source_count"] == 101
    assert (
        payload["critic_score_config"]["expected_terminal_rollout_count"]
        == 3232
    )
    assert len(payload["target_grid_config"]["grid"]) == 6
    assert {row["physical_gpu_index"] for row in payload["value_jobs"]} == set(
        range(6)
    )
    assert all("beta_max" not in row["config"] for row in payload["value_jobs"])
    write_value_configs_v4(payload, tmp_path / "configs")
    manifest = json.loads(
        (tmp_path / "configs" / "manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["value_training_config_paths"]) == 6
    assert len(manifest["guidance_smc_config_paths"]) == 18
    assert len(manifest["guidance_critic_config_paths"]) == 18
    assert len(manifest["guidance_closed_config_paths"]) == 18
    assert len(manifest["guidance_open_metric_config_paths"]) == 18
    assert len(manifest["guidance_independent_evaluator_config_paths"]) == 18
    assert (
        len(manifest["guidance_independent_evaluator_comparison_config_paths"])
        == 18
    )
    assert len(manifest["guidance_result_paths"]) == 18
    assert all(
        "matched_compute.scored.jsonl" in row["matched_compute_path"]
        for row in manifest["guidance_result_paths"]
    )
    assert set(manifest["config_paths"]) == {
        "value_rollout.json",
        "value_critic_score.json",
        "value_target_grid.json",
    }


def test_v4_value_config_producer_rejects_authorization_or_gpu_drift() -> None:
    protocol = json.loads(GUIDANCE_PROTOCOL.read_text(encoding="utf-8"))
    critic = _critic_ready()
    flow = _setflow_ready()
    authorization = authorize_xeditflow_guidance_v4(critic, flow)
    changed = copy.deepcopy(authorization)
    changed["guidance_grid_combination_count"] = 17
    with pytest.raises(Exception, match="exact joint authorization"):
        build_value_configs_v4(
            protocol,
            changed,
            critic,
            flow,
            _refit(),
            _source_audit(),
            rollout_gpu=0,
            critic_gpu=1,
            value_gpus=(0, 1, 2, 3, 4, 5),
        )
    with pytest.raises(Exception, match="once each"):
        build_value_configs_v4(
            protocol,
            authorization,
            critic,
            flow,
            _refit(),
            _source_audit(),
            rollout_gpu=0,
            critic_gpu=1,
            value_gpus=(0, 0, 1, 2, 3, 4),
        )
