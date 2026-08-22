from __future__ import annotations

import pytest

from scripts.route_a_v3.score_route2_xeditflow_closed_frozen_methods_v3 import (
    validate_closed_frozen_score_config_v3,
)


def _config(method_id: str) -> dict:
    return {
        "schema_version": "route_a_v3_route2_xeditflow_closed_frozen_score_config.v1",
        "method_id": method_id,
        "pool_assignment": "DEVELOPMENT",
        "split": "VALIDATION",
        "base_flow_training_seed": 20260904,
        "expected_source_count": 891,
        "physical_gpu_index": 2,
        "device": "cuda:2",
        "kappa": 0.5,
        "critic_online_microbatch_size": 4,
        "critic_refit_manifest_path": "/mnt/refit.json",
        "strongest_generation_baseline_path": "/mnt/strongest.json",
        "baseline_selection_input_path": "/mnt/selection.json",
    }


@pytest.mark.parametrize("method_id", ["generate_then_rerank", "strongest_matched_baseline"])
def test_closed_frozen_score_config_accepts_only_frozen_rerank_search_methods(method_id: str) -> None:
    validate_closed_frozen_score_config_v3(_config(method_id))


def test_closed_frozen_score_config_rejects_guided_trajectory_method() -> None:
    with pytest.raises(Exception, match="method differs"):
        validate_closed_frozen_score_config_v3(_config("simple_rate_guidance"))


def test_closed_frozen_score_config_rejects_test_cohort() -> None:
    config = _config("generate_then_rerank")
    config["split"] = "TEST"
    with pytest.raises(Exception, match="cohort"):
        validate_closed_frozen_score_config_v3(config)


def test_closed_frozen_score_config_requires_strongest_selection_boundary() -> None:
    config = _config("strongest_matched_baseline")
    config["baseline_selection_input_path"] = ""
    with pytest.raises(Exception, match="selection input"):
        validate_closed_frozen_score_config_v3(config)
