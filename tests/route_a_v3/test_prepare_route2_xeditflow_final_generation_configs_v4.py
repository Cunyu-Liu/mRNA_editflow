from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.route_a_v3.prepare_route2_xeditflow_final_generation_configs_v4 import (
    FINAL_METHODS_V4,
    build_final_generation_configs_v4,
    write_final_generation_configs_v4,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/route_a_v3_route2_xeditflow_v4_guidance_protocol_v1.json"
ROUTE2 = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2"


def _critic_ready() -> dict:
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_guidance_readiness.v1",
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
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_confirmation_gate.v1",
        "status": "XEDITSETFLOW_V4_G0_READY",
        "required_seeds": [20260912, 20260913, 20260914],
        "seed_results": {
            "20260912": {"passed": True, "selected_checkpoint_pass": 4},
            "20260913": {"passed": True, "selected_checkpoint_pass": 6},
            "20260914": {"passed": True, "selected_checkpoint_pass": 10},
        },
        "additional_seed_authorized": False,
        "development_test_authorized": False,
        "guidance_authorized": False,
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
        "checkpoints": [
            {
                "seed": seed,
                "checkpoint_path": f"{ROUTE2}/critic/refit-{seed}.pt",
                "physical_batch_size": batch,
            }
            for seed, batch in zip(
                (20260908, 20260909, 20260910), (4, 8, 16), strict=True
            )
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


def _gate() -> dict:
    return {
        "schema_version": "route_a_v3_route2_xeditflow_v4_guidance_screen_gate.v1",
        "status": "XEDITFLOW_V4_GUIDANCE_SCREEN_FROZEN",
        "base_flow_training_seed": 20260912,
        "combination_count": 18,
        "selected_kappa": 0.5,
        "selected_temperature": 1.0,
        "selected_beta_max": 2.0,
    }


def _strongest() -> dict:
    return {
        "status": "DEVELOPMENT_STRONGEST_GENERATION_BASELINE_FROZEN_INDEPENDENT_EVALUATOR_ONLY",
        "strongest_generation_baseline_id": "genetic",
        "evaluation_outcomes_accessed": False,
        "forward_equivalent_budget_per_source": 320,
        "critic_forward_budget_per_source": 320,
        "guiding_checkpoint_path": f"{ROUTE2}/baselines/guiding.pt",
        "independent_evaluator_checkpoint_path": f"{ROUTE2}/baselines/evaluator.pt",
    }


def _baseline_selection() -> dict:
    return {
        "selection_pool": "DEVELOPMENT_MEASURED_NEIGHBORHOOD",
        "evaluation_release_state": "CLOSED",
    }


def _payload() -> dict:
    return build_final_generation_configs_v4(
        json.loads(PROTOCOL.read_text(encoding="utf-8")),
        _critic_ready(),
        _setflow_ready(),
        _refit(),
        _source_audit(),
        _gate(),
        _strongest(),
        _baseline_selection(),
        generation_gpus={20260912: 0, 20260913: 1, 20260914: 2},
        critic_gpus={20260912: 3, 20260913: 4, 20260914: 5},
        value_gpus={20260913: 0, 20260914: 1},
        strongest_timing_gpu=2,
        guidance_screen_gate_path=f"{ROUTE2}/guidance/gate.json",
        strongest_closed_score_table_path=(
            f"{ROUTE2}/baselines/strongest_closed_scores.private.jsonl"
        ),
    )


def test_v4_final_configs_freeze_one_screen_value_and_two_seed_local_values() -> None:
    payload = _payload()
    assert payload["required_base_flow_training_seeds"] == [
        20260912,
        20260913,
        20260914,
    ]
    assert payload["selected_combination"] == [0.5, 1.0, 2.0]
    assert payload["terminal_critic_forwards_by_member"] == [8, 4, 2]
    assert payload["strongest_timing_config"]["seed"] == 20260816
    jobs = {row["base_flow_training_seed"]: row for row in payload["seed_jobs"]}
    assert jobs[20260912]["screen_value_checkpoint_reused"] is True
    assert jobs[20260912]["value_training_config"] is None
    assert "/screen_seed_20260912/value_models/" in jobs[20260912][
        "value_checkpoint_path"
    ]
    for seed in (20260913, 20260914):
        job = jobs[seed]
        assert job["screen_value_checkpoint_reused"] is False
        assert job["value_rollout_config"]["base_flow_training_seed"] == seed
        assert job["value_critic_score_config"]["base_flow_training_seed"] == seed
        assert job["value_target_config"]["base_flow_training_seed"] == seed
        assert "beta_max" not in job["value_target_config"]
        assert job["value_training_config"]["base_flow_training_seed"] == seed
        assert job["value_training_config"]["checkpoint_selection"] == (
            "FINAL_PASS_8_NO_EPOCH_RESELECTION"
        )


def test_v4_final_configs_match_methods_compute_and_rerank_boundaries() -> None:
    payload = _payload()
    for job in payload["seed_jobs"]:
        assert set(job["matched_control_configs"]) == {
            "unguided_setflow",
            "first_order_guidance",
            "simple_rate_guidance",
            "generate_then_rerank",
        }
        assert set(job["terminal_critic_score_configs"]) == set(FINAL_METHODS_V4)
        assert set(job["open_metric_configs"]) == set(FINAL_METHODS_V4)
        configs = [job["full_smc_config"], *job["matched_control_configs"].values()]
        assert all(
            row["decoder_seed_base"] == 20261001
            and row["candidate_cap"] == 32
            and row["forward_equivalent_ceiling_per_source"] == 320
            for row in configs
        )
        for method in FINAL_METHODS_V4:
            scorer = job["terminal_critic_score_configs"][method]
            metric = job["open_metric_configs"][method]
            assert scorer["critic_self_score_used_for_generation_or_selection"] is (
                method == "generate_then_rerank"
            )
            assert metric["critic_self_score_used_for_ranking"] is (
                method == "generate_then_rerank"
            )
            assert scorer["candidate_path"].endswith(
                f"/{method}/generated_candidates.private.jsonl"
            )


def test_v4_final_configs_close_exact_and_frozen_score_benchmarks() -> None:
    payload = _payload()
    strongest_paths = set()
    for job in payload["seed_jobs"]:
        exact = job["closed_exact_configs"]
        assert exact["full_soft_value_smc"]["potential_kind"] == "SOFT_VALUE"
        assert "value_checkpoint_path" in exact["full_soft_value_smc"]
        assert exact["unguided_setflow"]["potential_kind"] == "ZERO"
        assert "value_checkpoint_path" not in exact["unguided_setflow"]
        assert set(job["closed_control_score_configs"]) == {
            "first_order_guidance",
            "simple_rate_guidance",
            "generate_then_rerank",
        }
        assert set(job["closed_metric_configs"]) == {
            "first_order_guidance",
            "simple_rate_guidance",
            "generate_then_rerank",
            "strongest_matched_baseline",
        }
        strongest = job["closed_metric_configs"]["strongest_matched_baseline"]
        strongest_paths.add(strongest["score_table_path"])
        assert strongest["strongest_baseline_frozen_before_v4_candidate_generation"]
        assert strongest["baseline_reselected_for_v4"] is False
        assert job["equal_wall_time_config"]["methods"][
            "full_soft_value_smc"
        ]["timing_path"].endswith("terminal_critic/matched_compute.scored.jsonl")
        assert job["final_seed_evidence_config"]["methods"][
            "strongest_matched_baseline"
        ].get("terminal_critic_summary_path") is None
    assert strongest_paths == {
        f"{ROUTE2}/baselines/strongest_closed_scores.private.jsonl"
    }


def test_v4_final_config_writer_emits_exact_non_overwriting_inventory(
    tmp_path: Path,
) -> None:
    payload = copy.deepcopy(_payload())
    payload["runtime_config_root"] = str(tmp_path / "configs")
    write_final_generation_configs_v4(payload, tmp_path / "configs")
    manifest = json.loads(
        (tmp_path / "configs" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["written_runtime_config_count"] == 97
    assert len(manifest["written_runtime_config_paths"]) == 97
    assert not (tmp_path / "configs" / "seed_20260912_value_training.json").exists()
    assert (tmp_path / "configs" / "seed_20260913_value_training.json").exists()
    with pytest.raises(Exception, match="config root exists"):
        write_final_generation_configs_v4(payload, tmp_path / "configs")


def test_v4_final_configs_reject_gate_or_seed_inventory_drift() -> None:
    arguments = {
        "protocol": json.loads(PROTOCOL.read_text(encoding="utf-8")),
        "critic_readiness": _critic_ready(),
        "setflow_confirmation": _setflow_ready(),
        "critic_refit_manifest": _refit(),
        "source_data_audit": _source_audit(),
        "guidance_gate": _gate(),
        "strongest_generation_baseline": _strongest(),
        "baseline_selection_input": _baseline_selection(),
        "generation_gpus": {20260912: 0, 20260913: 1, 20260914: 2},
        "critic_gpus": {20260912: 3, 20260913: 4, 20260914: 5},
        "value_gpus": {20260913: 0, 20260914: 1},
        "strongest_timing_gpu": 2,
        "guidance_screen_gate_path": f"{ROUTE2}/guidance/gate.json",
        "strongest_closed_score_table_path": (
            f"{ROUTE2}/baselines/strongest_closed_scores.private.jsonl"
        ),
    }
    arguments["guidance_gate"]["selected_beta_max"] = 3.0
    with pytest.raises(Exception, match="frozen grid"):
        build_final_generation_configs_v4(**arguments)
    arguments["guidance_gate"] = _gate()
    arguments["value_gpus"] = {20260913: 0}
    with pytest.raises(Exception, match="seed inventory"):
        build_final_generation_configs_v4(**arguments)
