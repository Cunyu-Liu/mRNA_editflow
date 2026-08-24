"""Central training-attempt identities and metadata for XEditCritic V4."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


class XEditCriticLedgerV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticLedgerV4Error(message)


def _screen_run(config: Mapping[str, Any], run_id: str) -> Mapping[str, Any]:
    matches = [
        run
        for run in config["required_screen_runs"]
        if str(run["run_id"]) == str(run_id)
    ]
    _require(len(matches) == 1, "Critic V4 run id is not an exact frozen screen run")
    return matches[0]


def critic_v4_attempt_config(
    config: Mapping[str, Any],
    *,
    run_id: str,
    physical_gpu_index: int,
    physical_batch_size: int,
) -> dict[str, Any]:
    run = _screen_run(config, run_id)
    _require(int(config["training"]["screen_seed"]) == 20260907, "Critic V4 screen seed changed")
    _require(int(physical_gpu_index) in config["gpu_policy"]["physical_gpu_scope"], "Critic V4 physical GPU is outside 0–5")
    _require(int(physical_batch_size) in config["memory_preflight"]["physical_batch_candidates"], "Critic V4 physical batch is undeclared")
    is_full_geometry = str(run["model"]) == "V4-FULL"
    permutation = bool(run.get("candidate_bundle_permutation", False))
    control = "CANDIDATE_BUNDLE_PERMUTATION" if permutation else str(run["control"])
    mechanism = str(run["mechanism"])
    learning_rates = config["training"]["learning_rates"]
    return {
        **dict(config),
        "attempt_id": f"xeditcritic_v4_screen_seed20260907::{run_id}",
        "baseline_id": f"xeditcritic_v4_{run_id}_seed20260907",
        "attempt_purpose": "XEDITCRITIC_V4_SCREEN",
        "scientific_role": f"XEDITCRITIC_V4_SCREEN_{run['model']}_{control}_{mechanism}",
        "result_stage": "DEVELOPMENT_VALIDATION",
        "run_mode": "FROZEN_SCREEN",
        "model_kind": str(run["model"]),
        "pretrained_model_id": config["model_id"] if is_full_geometry else "",
        "pretrained_feature_cache_path": config["bottom_six_cache"] if is_full_geometry else "",
        "hidden_dim": 768 if is_full_geometry else 65,
        "depth": 12 if is_full_geometry else 2,
        "epochs": int(config["data_geometry"]["pass_count"]),
        "learning_rate": float(learning_rates["new_head_and_v4_trunk"]),
        "loss_kind": "STANDARDIZED_HUBER_PLUS_CROSS_GROUP_PAIRWISE_THEN_SOFT_SPEARMAN",
        "metadata_mode": "OUTCOME_FREE_ENDPOINT_DESCRIPTORS",
        "training_weighting_mode": "STUDY_THEN_SOURCE_GROUP",
        "training_sampling_mode": "SQRT_TASK_SIZE_FIXED_EFFECTIVE32_REPEAT_CAP4",
        "loss_aggregation_mode": "TASK_ROBUST_STANDARDIZED_EFFECTIVE_TASK_BATCH",
        "target_scaling_mode": "TRAIN_TASK_ROBUST_WITH_REGION_GLOBAL_FALLBACK",
        "candidate_control": control,
        "seed": 20260907,
        "physical_gpu_index": int(physical_gpu_index),
        "device": f"cuda:{physical_gpu_index}",
        "optimizer_name": "AdamW",
        "optimizer_fused": False,
        "training_precision": "BF16",
        "encoder_attention_backend": config["memory_preflight"]["attention_backend"] if is_full_geometry else "NOT_APPLICABLE_RAW_C0",
        "critic_position_features": "SOURCE_RELATIVE_EDIT_SET_AND_RADIUS32_LOCAL_CROSS_ATTENTION" if is_full_geometry else "RAW_FULL_CONTEXT",
        "batch_size": 32,
        "weight_decay": float(config["training"]["weight_decay"]),
        "huber_delta": float(config["training"]["huber_delta"]),
        "notes": f"physical_batch={physical_batch_size}; mechanism={mechanism}; final_pass_8_fixed; no TEST/Evaluation access",
    }


def critic_v4_attempt_details(
    config: Mapping[str, Any],
    *,
    trainable_parameter_count: int | None = None,
    physical_batch_size: int | None = None,
    peak_vram_mb: float | None = None,
) -> dict[str, Any]:
    geometry = config["data_geometry"]
    result: dict[str, Any] = {
        "record_counts": {
            "TRAIN": int(geometry["expected_train_count"]),
            "VALIDATION": int(geometry["expected_validation_count"]),
        },
        "development_test_record_count_withheld": int(
            geometry["withheld_development_test_record_count"]
        ),
        "evaluation_record_count": 0,
        "included_regions": ["5UTR", "3UTR"],
        "protected_outcome_reads": {
            "development_test": 0,
            "new_final_evaluation": 0,
        },
    }
    if trainable_parameter_count is not None:
        result["trainable_parameter_count"] = int(trainable_parameter_count)
    if physical_batch_size is not None:
        result["physical_batch_size"] = int(physical_batch_size)
    if peak_vram_mb is not None:
        result["peak_vram_mb"] = float(peak_vram_mb)
    return result


def critic_v4_ledger_paths(
    config: Mapping[str, Any], output_directory: Path
) -> tuple[Path, Path]:
    return (
        Path(config["experiment_ledger_path"]),
        output_directory / "training_attempt.json",
    )
