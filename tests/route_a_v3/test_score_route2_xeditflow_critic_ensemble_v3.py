from __future__ import annotations

import pytest

from scripts.route_a_v3.score_route2_xeditflow_critic_ensemble_v3 import (
    XEditFlowCriticEnsembleScorerV3Error,
    validate_critic_ensemble_score_config_v3,
)


def test_critic_ensemble_score_config_freezes_three_member_batching() -> None:
    config = {
        "schema_version": "route_a_v3_route2_xeditflow_critic_ensemble_score_config.v1",
        "kappa": 0.5,
        "base_flow_training_seed": 20260904,
        "critic_batch_size": 256,
        "critic_online_microbatch_size": 4,
        "physical_gpu_index": 4,
        "device": "cuda:4",
    }
    validate_critic_ensemble_score_config_v3(config)
    with pytest.raises(XEditFlowCriticEnsembleScorerV3Error):
        validate_critic_ensemble_score_config_v3({**config, "critic_online_microbatch_size": 8})
