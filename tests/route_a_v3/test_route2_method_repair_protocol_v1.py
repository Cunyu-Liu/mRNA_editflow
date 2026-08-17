from __future__ import annotations

import json
from pathlib import Path

from core.route2_delta_predictor import (
    Route2DeltaPredictor,
    Route2EditCenteredDeltaPredictor,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/route_a_v3_route2_method_repair_protocol_v2.json"


def _load(path: str | Path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return json.loads(path.read_text(encoding="utf-8"))


def test_method_repair_screen_is_development_only_and_factorially_aligned() -> None:
    protocol = _load(PROTOCOL)
    config_paths = protocol["screen_arms"] + protocol["matched_controls"]
    configs = [_load(path) for path in config_paths]
    assert len(configs) == 6
    shared = {
        key: configs[0][key]
        for key in (
            "result_stage",
            "run_mode",
            "development_manifest",
            "canonical_paths",
            "metadata_mode",
            "training_weighting_mode",
            "loss_kind",
            "ranking_loss_weight",
            "checkpoint_selection",
            "checkpoint_metric",
            "batch_size",
            "seed",
            "learning_rate",
            "weight_decay",
            "epochs",
        )
    }
    assert shared["result_stage"] == "HPO_VALIDATION_ONLY"
    assert shared["metadata_mode"] == "TRANSFERABLE_CONTEXT"
    assert shared["training_weighting_mode"] == "TASK_THEN_SOURCE_CONTEXT_ENDPOINT_GROUP"
    assert shared["checkpoint_selection"] == "BEST_VALIDATION"
    assert shared["checkpoint_metric"] == "TASK_MACRO_SPEARMAN_THEN_STANDARDIZED_MAE"
    for config in configs:
        assert {key: config[key] for key in shared} == shared
        assert config["evaluation_outcomes_accessed"] is False
        assert config["development_test_outcomes_accessed"] is False
        assert config["device"] == f"cuda:{config['physical_gpu_index']}"
        assert all(
            name not in path
            for path in config["canonical_paths"]
            for name in ("GSE232572", "E-MTAB-10902", "GSE246381")
        )
    roles = {config["scientific_role"]: config for config in configs}
    assert {
        (
            roles[role]["model_kind"],
            roles[role]["target_scaling_mode"],
        )
        for role in (
            "FACTORIAL_GLOBAL_RAW",
            "FACTORIAL_GLOBAL_SCALED",
            "FACTORIAL_EDIT_CENTERED_RAW",
            "FACTORIAL_EDIT_CENTERED_SCALED",
        )
    } == {
        ("delta_anchored_position_aware_antisymmetric", "NONE"),
        ("delta_anchored_position_aware_antisymmetric", "TRAIN_TASK_ROBUST"),
        ("delta_edit_centered_antisymmetric", "NONE"),
        ("delta_edit_centered_antisymmetric", "TRAIN_TASK_ROBUST"),
    }
    assert roles["MATCHED_SOURCE_ONLY_CONTROL"]["model_kind"] == "delta_edit_centered_source_only_control"
    assert roles["MATCHED_TRAIN_CANDIDATE_PERMUTATION_CONTROL"]["candidate_control"] == "WITHIN_EXACT_SOURCE_TASK_TRAIN_CANDIDATE_PERMUTATION"
    assert len({config["output_directory"] for config in configs}) == len(configs)


def test_edit_centered_and_global_factorial_arms_have_matched_capacity() -> None:
    counts = {
        "study_count": 9,
        "assay_count": 16,
        "context_count": 32,
        "endpoint_count": 16,
    }
    global_model = Route2DeltaPredictor(hidden_dim=113, depth=7, **counts)
    edit_model = Route2EditCenteredDeltaPredictor(hidden_dim=108, depth=7, **counts)
    global_parameters = sum(parameter.numel() for parameter in global_model.parameters())
    edit_parameters = sum(parameter.numel() for parameter in edit_model.parameters())
    assert abs(edit_parameters - global_parameters) / global_parameters < 0.03


def test_protocol_keeps_new_method_claim_and_guidance_fail_closed() -> None:
    protocol = _load(PROTOCOL)
    boundary = protocol["post_exposure_boundary"]
    assert boundary["GSE232572"] == "OUTCOME_EXPOSED_DO_NOT_USE_FOR_MODEL_SELECTION"
    assert boundary["new_external_confirmation_required_for_new_method_claim"] is True
    assert protocol["screen_selection"]["single_seed_role"] == "EXPLORATORY_SELECTION_ONLY"
    assert protocol["screen_selection"]["development_test_role"] == "NOT_ACCESSED"
    assert protocol["guided_generation_status"].startswith("BLOCKED_")
    assert protocol["gpu_policy"]["parameter_updates"] == "CUDA_ONLY"
    assert protocol["gpu_policy"]["cpu_fallback"] == "FORBIDDEN"
    recovery = protocol["runtime_recovery"]
    assert recovery["recovery_change"] == "PHYSICAL_GPU_ONLY"
    assert recovery["scientific_configuration_changed"] is False
    assert "gpu7" in recovery["failed_config"]
    invalidated = protocol["invalidated_control"]
    assert invalidated["status"] == "INVALID_CROSS_SOURCE_CANDIDATE_PERMUTATION"
    assert "exact_source" in invalidated["replacement"]
    assert protocol["loso_study_unit_ids"] == [
        "ENCSR854RUF", "GSE114002", "GSE149487", "GSE186455",
        "GSE200304", "GSE217518", "GSE269595",
    ]
    units = protocol["development_independence_units"]
    assert units["study_unit_count"] == len(protocol["loso_study_unit_ids"]) == 7
    assert units["record_count"] == sum(units["fixed_split_record_counts"].values())
    legacy = protocol["legacy_best_observed_validation_reference"]
    assert legacy["defined_task_count"] == units["task_count"] == 9
    assert legacy["prior_run_count_replayed"] == 42
    assert legacy["development_test_used"] is False
    assert legacy["evaluation_used"] is False
    breadth = protocol["screen_breadth_requirements"]
    assert breadth["task_count"] == 9
    assert breadth["minimum_tasks_improved_over_global_raw"] == 5
    assert breadth["minimum_tasks_improved_over_source_only_control"] == 5
    support = protocol["candidate_permutation_support"]
    assert support["audit_timing"] == "BEFORE_EDIT_CENTERED_OR_CONTROL_SCREEN_OUTCOMES_WERE_READ"
    assert support["all_task_primary_control"] == "MATCHED_SOURCE_ONLY_CONTROL"
    assert support["changed_candidate_sequence_count"] == 26464
    assert support["minimum_eligible_task_wins"] == len(support["eligible_tasks"]) == 2
    audit = _load(support["audit"])
    assert audit["status"] == "EXACT_SOURCE_PERMUTATION_VALID_BUT_TASK_SUPPORT_PARTIAL"
    assert audit["permutation_gate_eligible_tasks"] == support["eligible_tasks"]
    assert audit["task_balanced_weighting_audit"]["source_groups_spanning_multiple_tasks"] == 0
