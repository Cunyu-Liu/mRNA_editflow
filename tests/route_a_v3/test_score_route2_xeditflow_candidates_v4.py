from __future__ import annotations

import copy

import pytest

from core.route2_xeditflow_guidance_v4 import MatchedComputeRecordV4
from core.route2_xeditcritic_training_data_v3 import UNKNOWN_CATEGORY
from scripts.route_a_v3.score_route2_xeditflow_candidates_v4 import (
    candidate_projection_rows_v4,
    reconcile_candidate_compute_v4,
    validate_candidate_score_config_v4,
)


def _config() -> dict:
    return {
        "schema_version": (
            "route_a_v3_route2_xeditflow_candidate_critic_score_config.v4"
        ),
        "critic_seeds": [20260908, 20260909, 20260910],
        "critic_refit_runtime_config_paths": {
            "20260908": "/mnt/a.json",
            "20260909": "/mnt/b.json",
            "20260910": "/mnt/c.json",
        },
        "base_flow_training_seed": 20260912,
        "kappa": 0.5,
        "temperature": 1.0,
        "beta_max": 2.0,
        "method_id": "method",
        "expected_source_count": 891,
        "candidate_cap_per_source": 32,
        "study_policy": "UNKNOWN_STUDY_SCALE_FIXED_1",
        "prediction_scale": "TASK_ROBUST_STANDARDIZED_EFFECT",
        "physical_gpu_index": 5,
        "device": "cuda:5",
        "critic_self_score_used_for_generation_or_selection": False,
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
        "generation_summary_path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/summary.json",
        "candidate_path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/candidates.jsonl",
        "generation_compute_path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/compute.jsonl",
        "output_dir": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/scored",
    }


def test_v4_candidate_scorer_config_freezes_identity_and_diagnostic_scope() -> None:
    validate_candidate_score_config_v4(_config())
    changed = copy.deepcopy(_config())
    changed["critic_self_score_used_for_generation_or_selection"] = True
    with pytest.raises(Exception, match="protected-input policy"):
        validate_candidate_score_config_v4(changed)


def test_v4_candidate_projection_accepts_identity_stop_and_complete_edit_bundle() -> None:
    source = {
        "source_key": "source",
        "source_sequence": "AA",
        "region": "5UTR",
        "assay_id": "assay",
        "biological_context_id": "context",
    }
    representative = {
        "task_id": "task",
        "source_group_id": "group",
        "endpoint_descriptor": {"quantity_family": "RNA_ABUNDANCE"},
    }
    candidates = [
        {
            "schema_version": "route_a_v3_route2_xeditflow_generated_candidate.v4",
            "source_key": "source",
            "generation_rank": 1,
            "candidate_sequence": "AA",
            "trajectory_replay_ok": True,
            "generated_candidate_grants_canonical_credit": False,
        },
        {
            "schema_version": "route_a_v3_route2_xeditflow_generated_candidate.v4",
            "source_key": "source",
            "generation_rank": 2,
            "candidate_sequence": "CA",
            "trajectory_replay_ok": True,
            "generated_candidate_grants_canonical_credit": False,
        },
    ]
    rows = candidate_projection_rows_v4(
        candidates, source=source, representative=representative
    )
    assert rows[0]["source_relative_edits"] == []
    assert rows[1]["source_relative_edits"] == [
        {"position": 0, "source_base": "A", "candidate_base": "C"}
    ]
    assert all(row["study_unit_id"] == UNKNOWN_CATEGORY for row in rows)
    assert all(row["dummy_target_for_inference_only"] is True for row in rows)


def test_v4_candidate_scorer_reconciles_reserved_to_actual_member_forwards() -> None:
    compute = MatchedComputeRecordV4(
        source_key="source",
        trunk_forwards=10,
        mode_forwards=80,
        value_forwards=10,
        critic_forwards_by_member=[8, 4, 1],
        candidate_count=2,
        trajectory_count=32,
        wall_time_seconds=1.0,
        peak_vram_mb=100.0,
    ).to_dict()
    compute["source_equal_wall_time_seconds"] = 4.0
    compute["source_equal_wall_peak_vram_mb"] = 300.0
    compute["source_equal_wall_time_scope"] = "GENERATION_AND_POSTHOC_SCORING"
    compute["source_cuda_device_name"] = "A100"
    reconciled = reconcile_candidate_compute_v4(
        compute,
        actual_critic_forwards_by_member=[1, 1, 1],
        scorer_wall_time_seconds=2.0,
        scorer_peak_vram_mb=200.0,
    )
    assert reconciled["terminal_critic_forwards_reserved_by_member"] == [8, 4, 1]
    assert reconciled["terminal_critic_forwards_actual_by_member"] == [1, 1, 1]
    assert reconciled["critic_forwards_by_member"] == [1, 1, 1]
    assert reconciled["total_forward_equivalents"] == 103
    assert reconciled["wall_time_seconds"] == 6.0
    assert reconciled["peak_vram_mb"] == 300.0
    assert reconciled["source_equal_wall_time_seconds"] == 6.0
    assert reconciled["source_equal_wall_peak_vram_mb"] == 300.0
    with pytest.raises(Exception, match="exceeds"):
        reconcile_candidate_compute_v4(
            compute,
            actual_critic_forwards_by_member=[9, 1, 1],
            scorer_wall_time_seconds=0.0,
            scorer_peak_vram_mb=0.0,
        )
