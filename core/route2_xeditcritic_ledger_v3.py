"""Central-attempt metadata shared by Critic V3 cache and LoRA runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def critic_v3_attempt_config(
    config: Mapping[str, Any],
    *,
    run_id: str,
    arm: str,
    control_mode: str,
    candidate_bundle_permutation: bool,
    physical_gpu_index: int,
) -> dict[str, Any]:
    control = (
        "CANDIDATE_BUNDLE_PERMUTATION"
        if candidate_bundle_permutation
        else control_mode
    )
    pretrained = arm in {"C1", "C2", "C3"}
    return {
        **dict(config),
        "attempt_id": f"xeditcritic_v3_screen_seed{config['screen_seed']}::{run_id}",
        "baseline_id": f"xeditcritic_v3_{run_id}_seed{config['screen_seed']}",
        "attempt_purpose": "XEDITCRITIC_V3_SCREEN",
        "scientific_role": f"XEDITCRITIC_V3_{arm}_{control}",
        "result_stage": "DEVELOPMENT_VALIDATION",
        "run_mode": "FROZEN_SCREEN",
        "model_kind": f"XEDITCRITIC_V3_{arm}",
        "pretrained_model_id": (
            "YYLY66/mRNABERT@a1eb7df25804d23f08646e1cb996b234d7208a40"
            if pretrained else ""
        ),
        "pretrained_feature_cache_path": (
            str(config["edit_site_cache"]) if pretrained else ""
        ),
        "hidden_dim": 512 if arm in {"C2", "C3"} else 65,
        "depth": 8 if arm in {"C2", "C3"} else 2,
        "epochs": int(config["passes"]),
        "learning_rate": float(config["head_learning_rate"]),
        "loss_kind": "STANDARDIZED_HUBER_THEN_HUBER_PLUS_PAIRWISE_LOGISTIC",
        "metadata_mode": "OUTCOME_FREE_ENDPOINT_DESCRIPTORS",
        "training_weighting_mode": "STUDY_THEN_SOURCE_GROUP",
        "training_sampling_mode": "SQRT_TASK_SIZE_TASK_HOMOGENEOUS_REPEAT_CAP_4",
        "loss_aggregation_mode": "TASK_ROBUST_STANDARDIZED",
        "target_scaling_mode": "TRAIN_TASK_ROBUST_WITH_REGION_GLOBAL_FALLBACK",
        "candidate_control": control,
        "seed": int(config["screen_seed"]),
        "physical_gpu_index": int(physical_gpu_index),
        "device": f"cuda:{physical_gpu_index}",
        "optimizer_name": "AdamW",
        "optimizer_fused": False,
        "training_precision": "BF16",
        "encoder_attention_backend": "OFFICIAL_PYTORCH_FALLBACK" if arm == "C3" else "FROZEN_CACHE",
        "critic_position_features": "SOURCE_RELATIVE_EDIT_SITE_AND_RADIUS16",
        "batch_size": int(config["batch_size"]),
        "weight_decay": float(config["weight_decay"]),
        "huber_delta": float(config["huber_delta"]),
    }


def critic_v3_attempt_details(
    config: Mapping[str, Any],
    *,
    trainable_parameter_count: int | None = None,
    train_record_count: int | None = None,
    validation_record_count: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "development_test_record_count_withheld": int(
            config["withheld_development_test_record_count"]
        ),
        "evaluation_record_count": 0,
        "included_regions": ["5UTR", "3UTR"],
    }
    if trainable_parameter_count is not None:
        result["trainable_parameter_count"] = int(trainable_parameter_count)
    if train_record_count is not None and validation_record_count is not None:
        result["record_counts"] = {
            "TRAIN": int(train_record_count),
            "VALIDATION": int(validation_record_count),
        }
    return result


def critic_v3_ledger_paths(config: Mapping[str, Any], output_directory: Path) -> tuple[Path, Path]:
    return Path(config["experiment_ledger_path"]), output_directory / "training_attempt.json"
