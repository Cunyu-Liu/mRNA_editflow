from __future__ import annotations

from pathlib import Path

import pytest

from core.route2_legal_xeditflow import initial_state
from scripts.route_a_v3.run_route2_xeditflow_matched_controls_v3 import (
    _critic_adapter_rows_for_states_v3,
    _final_compute_v3,
    _terminal_state_from_candidate_v3,
    validate_matched_control_config_v3,
)


def _config():
    return {
        "schema_version": "route_a_v3_route2_xeditflow_matched_control_run_config.v1",
        "method_id": "first_order_guidance",
        "base_flow_training_seed": 20260904,
        "setflow_arm": "f2",
        "kappa": 0.5,
        "beta_max": 1.0,
        "particle_count": 32,
        "candidate_cap": 32,
        "ess_threshold": 16.0,
        "resampling": "STRATIFIED",
        "forward_equivalent_ceiling_per_source": 320,
        "maximum_sampling_rounds": 32,
        "reserved_terminal_critic_forwards": 3,
        "critic_online_microbatch_size": 4,
        "action_space": "SUB+STOP",
        "replay_check": True,
        "physical_gpu_index": 2,
        "device": "cuda:2",
        "output_dir": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/out",
    }


def test_matched_control_config_freezes_inventory_and_compute() -> None:
    for method in (
        "unguided_setflow", "first_order_guidance", "simple_rate_guidance", "generate_then_rerank"
    ):
        config = _config()
        config["method_id"] = method
        validate_matched_control_config_v3(config)
    config = _config()
    config["forward_equivalent_ceiling_per_source"] = 321
    with pytest.raises(Exception, match="ceiling"):
        validate_matched_control_config_v3(config)


def test_terminal_reconstruction_and_critic_adapter_are_source_relative() -> None:
    root = initial_state("AA", budget=1, assay_id="a", context_id="c")
    candidate = {"candidate_sequence": "AC", "terminal_cause": "BUDGET_EXHAUSTED"}
    terminal = _terminal_state_from_candidate_v3(root, candidate)
    assert terminal.source_relative_edits == ((1, "C"),)
    rows = _critic_adapter_rows_for_states_v3(
        [terminal],
        source_row={
            "source_key": "s", "source_sequence": "AA", "region": "5UTR",
            "assay_id": "assay", "biological_context_id": "context",
        },
        representative={
            "source_group_id": "g", "task_id": "t",
            "endpoint_descriptor": {
                "quantity_family": "q", "measurement_form": "m",
                "numerator_family": None, "denominator_family": None,
            },
        },
    )
    assert rows[0]["candidate_sequence"] == "AC"
    assert rows[0]["source_relative_edits"] == [
        {"position": 1, "source_base": "A", "candidate_base": "C"}
    ]


def test_terminal_critic_calls_are_added_member_by_member() -> None:
    compute = {
        "source_key": "s",
        "base_flow_forwards": 3,
        "value_forwards": 0,
        "critic_forwards_by_member": [2, 2, 2],
        "candidate_count": 20,
        "wall_time_seconds": 1.0,
        "peak_vram_mb": 0.0,
    }
    final = _final_compute_v3(compute, (1, 1, 1))
    assert final["critic_forwards_by_member"] == [3, 3, 3]
    assert final["total_forward_equivalents"] == 12


def test_matched_control_runner_times_full_per_source_generation_scope() -> None:
    source = Path(
        "scripts/route_a_v3/run_route2_xeditflow_matched_controls_v3.py"
    ).read_text(encoding="utf-8")
    assert "torch.cuda.synchronize(device)" in source
    assert "generation_without_posthoc_scoring_finished" in source
    assert "posthoc_terminal_critic_scoring_in_equal_wall_time" in source
    assert "source_equal_wall_peak_vram_mb" in source
    assert "EQUAL_WALL_TIME_SCOPE_V3" in source
