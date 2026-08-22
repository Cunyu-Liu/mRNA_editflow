from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.route_a_v3.run_route2_xeditflow_strongest_timing_v3 import (
    strongest_timing_command_v3,
    validate_strongest_timing_config_v3,
)


def _inputs():
    strongest = {
        "status": "DEVELOPMENT_STRONGEST_GENERATION_BASELINE_FROZEN_INDEPENDENT_EVALUATOR_ONLY",
        "strongest_generation_baseline_id": "genetic",
        "evaluation_outcomes_accessed": False,
        "forward_equivalent_budget_per_source": 320,
        "critic_forward_budget_per_source": 256,
        "guiding_checkpoint_path": "/mnt/critic.pt",
    }
    selection = {
        "selection_pool": "DEVELOPMENT_MEASURED_NEIGHBORHOOD",
        "evaluation_release_state": "CLOSED",
    }
    config = {
        "schema_version": "route_a_v3_route2_xeditflow_strongest_timing_config.v1",
        "method_id": "genetic",
        "source_manifest_path": "/mnt/sources.jsonl",
        "guiding_checkpoint_path": "/mnt/critic.pt",
        "critic_forward_budget_per_source": 256,
        "beam_width": 16,
        "genetic_population_size": 32,
        "oversample_factor": 8,
        "exhaustive_space_limit": 4096,
        "seed": 20260816,
        "physical_gpu_index": 2,
        "device": "cuda:2",
        "output_dir": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/final/timing",
        "timing_only_no_baseline_reselection": True,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    return config, strongest, selection


def test_strongest_timing_is_exact_frozen_genetic_rerun_without_reselection(tmp_path) -> None:
    config, strongest, selection = _inputs()
    validate_strongest_timing_config_v3(config, strongest, selection)
    command = strongest_timing_command_v3(config, tmp_path / "candidates.jsonl")
    assert command[command.index("--method") + 1] == "genetic"
    assert command[command.index("--max-critic-forwards") + 1] == "256"
    assert command[command.index("--seed") + 1] == "20260816"
    assert command[command.index("--physical-gpu-index") + 1] == "2"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("method_id", "beam", "method or decoder seed"),
        ("seed", 20260817, "method or decoder seed"),
        ("critic_forward_budget_per_source", 255, "budget differs"),
        ("genetic_population_size", 64, "hyperparameters differ"),
        ("physical_gpu_index", 6, "outside physical GPU 0-5"),
        ("timing_only_no_baseline_reselection", False, "evidence boundary"),
    ],
)
def test_strongest_timing_rejects_protocol_drift(field, value, message) -> None:
    config, strongest, selection = _inputs()
    drifted = deepcopy(config)
    drifted[field] = value
    with pytest.raises(Exception, match=message):
        validate_strongest_timing_config_v3(drifted, strongest, selection)
