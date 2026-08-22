from __future__ import annotations

import pytest

from scripts.route_a_v3.generate_route2_xeditflow_value_rollouts_v3 import (
    XEditFlowValueRolloutRunnerV3Error,
    _critic_examples_v3,
    validate_value_rollout_config_v3,
)


def _config() -> dict:
    return {
        "schema_version": "route_a_v3_route2_xeditflow_value_rollout_config.v1",
        "base_flow_training_seed": 20260904,
        "rollouts_per_state": 8,
        "states_per_record": 2,
        "state_pass_index": 0,
        "setflow_arm": "f2",
        "sampling_state_batch_size": 64,
        "trajectory_forward_batch_size": 64,
        "critic_batch_size": 32,
        "critic_online_microbatch_size": 4,
        "physical_gpu_index": 2,
        "device": "cuda:2",
        "output_dir": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/value_rollouts",
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def test_value_rollout_config_freezes_k_seed_and_gpu_scope() -> None:
    validate_value_rollout_config_v3(_config())
    changed = {**_config(), "rollouts_per_state": 9}
    with pytest.raises(XEditFlowValueRolloutRunnerV3Error):
        validate_value_rollout_config_v3(changed)
    changed = {**_config(), "physical_gpu_index": 6, "device": "cuda:6"}
    with pytest.raises(XEditFlowValueRolloutRunnerV3Error):
        validate_value_rollout_config_v3(changed)


def test_generated_critic_examples_are_study_neutral_and_bundle_complete() -> None:
    vocabs = {
        "study": {"__UNK__": 0, "study": 1},
        "assay": {"__UNK__": 0, "assay": 1},
        "context": {"__UNK__": 0, "context": 1},
        "quantity": {"__UNK__": 0, "RNA abundance": 1},
        "measurement": {"__UNK__": 0, "log2 fold": 1},
        "numerator": {"__UNK__": 0, "__NONE__": 1},
        "denominator": {"__UNK__": 0, "__NONE__": 1},
    }
    rows = [{
        "state_id": "s",
        "rollout_index": 0,
        "source_group_id": "g",
        "task_id": "t",
        "source_sequence": "ACGU",
        "candidate_sequence": "AAGU",
        "source_relative_edits": [{"position": 1, "source_base": "C", "candidate_base": "A"}],
        "endpoint_descriptor": {
            "quantity_family": "RNA abundance",
            "measurement_form": "log2 fold",
            "numerator_family": "__NONE__",
            "denominator_family": "__NONE__",
        },
        "assay_category": "assay",
        "context_category": "context",
        "region_id": 0,
    }]
    example = _critic_examples_v3(rows, vocabs)[0]
    assert example["study"] == 0
    assert example["edits"] == ((1, "C", "A"),)
    assert example["target_scale"] == 1.0
