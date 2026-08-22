from __future__ import annotations

import pytest

from scripts.route_a_v3.run_route2_xeditflow_smc_v3 import (
    total_maximum_forward_equivalents_v3,
    validate_smc_run_config_v3,
)


def _config():
    return {
        "schema_version": "route_a_v3_route2_xeditflow_smc_run_config.v1",
        "particle_count": 32,
        "candidate_cap": 32,
        "ess_threshold": 16.0,
        "resampling": "STRATIFIED",
        "forward_equivalent_ceiling_per_source": 320,
        "reserved_terminal_critic_forwards": 3,
        "maximum_sampling_rounds": 32,
        "base_flow_training_seed": 20260904,
        "kappa": 0.5,
        "temperature": 1.0,
        "beta_max": 2.0,
        "action_space": "SUB+STOP",
        "replay_check": True,
    }


def test_smc_run_config_matches_frozen_particles_grid_compute_and_replay() -> None:
    validate_smc_run_config_v3(_config())
    assert total_maximum_forward_equivalents_v3(317, 3) == 320


def test_smc_final_compute_rejects_unreported_terminal_overage() -> None:
    with pytest.raises(Exception, match="exceeds"):
        total_maximum_forward_equivalents_v3(318, 3)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("particle_count", 64, "particle count"),
        ("candidate_cap", 64, "candidate cap"),
        ("ess_threshold", 15, "ESS threshold"),
        ("forward_equivalent_ceiling_per_source", 321, "compute ceiling"),
        ("reserved_terminal_critic_forwards", 0, "critic ensemble reservation"),
        ("maximum_sampling_rounds", 31, "additional-round ceiling"),
        ("replay_check", False, "replay check"),
        ("beta_max", 4.0, "beta"),
    ],
)
def test_smc_run_config_rejects_frozen_protocol_drift(field, value, message) -> None:
    config = _config()
    config[field] = value
    with pytest.raises(Exception, match=message):
        validate_smc_run_config_v3(config)
