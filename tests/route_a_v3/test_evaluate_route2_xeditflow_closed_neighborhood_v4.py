from __future__ import annotations

import pytest

from core.route2_legal_xeditflow import legal_actions
from scripts.route_a_v3.evaluate_route2_xeditflow_closed_neighborhood_v4 import (
    mode_marginal_terminal_probability_v4,
    validate_closed_run_config_v4,
)


def _config() -> dict:
    return {
        "schema_version": (
            "route_a_v3_route2_xeditflow_closed_neighborhood_config.v4"
        ),
        "pool_assignment": "DEVELOPMENT",
        "split": "VALIDATION",
        "maximum_enumerated_edits": 5,
        "maximum_permutation_paths": 120,
        "enumeration": "ALL_EDIT_PERMUTATIONS_EXACT_SUM",
        "analysis_unit": "SOURCE",
        "undefined_source_policy": "EXCLUDE_NOT_ZERO_FILL",
        "potential_kind": "SOFT_VALUE",
        "latent_mode_policy": (
            "ROOT_PRIOR_WEIGHTED_SUM_OF_EIGHT_FIXED_MODE_TERMINAL_PROBABILITIES"
        ),
        "base_flow_training_seed": 20260912,
        "kappa": 0.5,
        "temperature": 1.0,
        "beta_max": 2.0,
        "method_id": "method",
        "expected_source_count": 891,
        "root_prior_forward_batch_size": 32,
        "value_child_forward_batch_size": 32,
        "source_token_cache_path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/cache.pt",
        "source_eligibility_manifest": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/sources.jsonl",
        "validation_projection_path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/validation.jsonl",
        "measured_neighborhood_path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/measured.jsonl",
        "value_checkpoint_path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/value.pt",
        "guidance_screen_gate_path": (
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/guidance_gate.json"
        ),
        "output_dir": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/closed",
        "physical_gpu_index": 2,
        "device": "cuda:2",
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def test_v4_closed_config_freezes_mode_marginalization_and_enumeration() -> None:
    validate_closed_run_config_v4(_config())
    final_seed = _config()
    final_seed["base_flow_training_seed"] = 20260914
    validate_closed_run_config_v4(final_seed)
    changed = _config()
    changed["latent_mode_policy"] = "SELECT_BEST_MODE"
    with pytest.raises(Exception, match="mode marginalization"):
        validate_closed_run_config_v4(changed)


def test_v4_unguided_closed_config_uses_zero_potential_without_value_checkpoint() -> None:
    config = _config()
    config["method_id"] = "unguided_setflow"
    config["potential_kind"] = "ZERO"
    del config["value_checkpoint_path"]
    validate_closed_run_config_v4(config)
    config["value_checkpoint_path"] = (
        "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/value.pt"
    )
    with pytest.raises(Exception, match="received a value checkpoint"):
        validate_closed_run_config_v4(config)


@pytest.mark.parametrize("candidate", ["A", "C"])
def test_v4_closed_probability_marginalizes_all_modes_and_supports_identity(
    candidate: str,
) -> None:
    cache = {}
    calls = []

    def builder(state):
        calls.append(state)
        actions = tuple(legal_actions(state))
        return tuple({action: 1.0 for action in actions} for _ in range(8))

    result = mode_marginal_terminal_probability_v4(
        "A",
        candidate,
        edit_budget=1,
        assay_id="assay",
        context_id="context",
        mode_prior=[0.125] * 8,
        rate_maps=cache,
        rate_map_builder=builder,
    )
    assert result["terminal_probability"] == pytest.approx(0.25)
    assert result["conditional_terminal_probability_by_mode"] == pytest.approx(
        [0.25] * 8
    )
    assert result["latent_mode_marginalized"] is True
    assert len(calls) == 1
