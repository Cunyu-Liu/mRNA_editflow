from __future__ import annotations

from pathlib import Path

import pytest

from scripts.route_a_v3.run_route2_xeditflow_smc_v4 import (
    maximum_round_forward_equivalents_v4,
    terminal_critic_forward_reservation_v4,
    validate_smc_run_config_v4,
)


def _config() -> dict:
    return {
        "schema_version": "route_a_v3_route2_xeditflow_smc_run_config.v4",
        "particle_count": 32,
        "candidate_cap": 32,
        "ess_threshold": 16.0,
        "resampling": "STRATIFIED",
        "forward_equivalent_ceiling_per_source": 320,
        "terminal_critic_forwards_by_member": [4, 4, 4],
        "maximum_sampling_rounds": 32,
        "base_flow_training_seed": 20260912,
        "kappa": 0.5,
        "temperature": 1.0,
        "beta_max": 2.0,
        "action_space": "SUB+STOP",
        "replay_check": True,
        "decoder_seed_base": 20261001,
        "expected_source_count": 891,
        "physical_gpu_index": 4,
        "device": "cuda:4",
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def test_v4_smc_runner_freezes_grid_seed_replay_and_compute() -> None:
    validate_smc_run_config_v4(_config())
    final_seed = _config()
    final_seed["base_flow_training_seed"] = 20260914
    validate_smc_run_config_v4(final_seed)
    assert maximum_round_forward_equivalents_v4(1) == 20
    assert maximum_round_forward_equivalents_v4(3) == 60
    assert maximum_round_forward_equivalents_v4(5) == 100
    refit = {
        "status": "XEDITCRITIC_V4_ALL_DEVELOPMENT_REFIT_COMPLETE",
        "checkpoints": [
            {"seed": 20260908, "physical_batch_size": 4},
            {"seed": 20260909, "physical_batch_size": 8},
            {"seed": 20260910, "physical_batch_size": 32},
        ],
    }
    assert terminal_critic_forward_reservation_v4(refit) == (8, 4, 1)


def test_v4_smc_critic_reservation_requires_all_frozen_physical_batches() -> None:
    refit = {
        "status": "XEDITCRITIC_V4_ALL_DEVELOPMENT_REFIT_COMPLETE",
        "checkpoints": [
            {"seed": 20260908, "physical_batch_size": 4},
            {"seed": 20260909, "physical_batch_size": 2},
            {"seed": 20260910, "physical_batch_size": 32},
        ],
    }
    with pytest.raises(Exception, match="physical batch"):
        terminal_critic_forward_reservation_v4(refit)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("particle_count", 64, "particle count"),
        ("candidate_cap", 64, "candidate cap"),
        ("ess_threshold", 15.0, "ESS threshold"),
        ("forward_equivalent_ceiling_per_source", 321, "compute ceiling"),
        ("terminal_critic_forwards_by_member", [1, 1, 0], "critic reservation"),
        ("maximum_sampling_rounds", 31, "additional-round ceiling"),
        ("base_flow_training_seed", 20260911, "seed changed"),
        ("beta_max", 4.0, "beta"),
        ("replay_check", False, "replay check"),
        ("decoder_seed_base", 20261002, "decoder seed"),
        ("expected_source_count", 890, "source count"),
        ("physical_gpu_index", 6, "GPU"),
    ],
)
def test_v4_smc_runner_rejects_protocol_drift(
    field: str, value: object, message: str
) -> None:
    config = _config()
    config[field] = value
    with pytest.raises(Exception, match=message):
        validate_smc_run_config_v4(config)


def test_v4_smc_runner_records_synchronized_equal_wall_time_and_reservation() -> None:
    source = Path("scripts/route_a_v3/run_route2_xeditflow_smc_v4.py").read_text(
        encoding="utf-8"
    )
    assert "torch.cuda.synchronize(device)" in source
    assert "torch.cuda.reset_peak_memory_stats(device)" in source
    assert "source_equal_wall_time_seconds" in source
    assert "source_equal_wall_peak_vram_mb" in source
    assert "terminal_critic_forwards_are_reserved_pending_scoring" in source
    assert "terminal_critic_scoring_performed\": False" in source
    assert "CUDA_VISIBLE_DEVICES remapping is forbidden" in source
