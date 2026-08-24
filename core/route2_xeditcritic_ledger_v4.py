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
    run_stage = str(config.get("run_stage", "SCREEN"))
    _require(run_stage in {"SCREEN", "CONFIRMATION"}, "Critic V4 run stage changed")
    seed = (
        int(config["training"]["screen_seed"])
        if run_stage == "SCREEN"
        else int(config["training_seed"])
    )
    _require(
        seed == 20260907
        if run_stage == "SCREEN"
        else seed in {20260908, 20260909, 20260910},
        "Critic V4 run seed changed",
    )
    if run_stage == "CONFIRMATION":
        _require(
            run_id in {"v4_full", "c0_v4"}
            and config.get("required_confirmation_run_ids")
            == ["v4_full", "c0_v4"],
            "Critic V4 confirmation attempted an undeclared run",
        )
    _require(int(physical_gpu_index) in config["gpu_policy"]["physical_gpu_scope"], "Critic V4 physical GPU is outside 0–5")
    _require(int(physical_batch_size) in config["memory_preflight"]["physical_batch_candidates"], "Critic V4 physical batch is undeclared")
    is_full_geometry = str(run["model"]) == "V4-FULL"
    permutation = bool(run.get("candidate_bundle_permutation", False))
    control = "CANDIDATE_BUNDLE_PERMUTATION" if permutation else str(run["control"])
    mechanism = str(run["mechanism"])
    learning_rates = config["training"]["learning_rates"]
    return {
        **dict(config),
        "attempt_id": f"xeditcritic_v4_{run_stage.lower()}_seed{seed}::{run_id}",
        "baseline_id": f"xeditcritic_v4_{run_id}_seed{seed}",
        "attempt_purpose": f"XEDITCRITIC_V4_{run_stage}",
        "scientific_role": f"XEDITCRITIC_V4_{run_stage}_{run['model']}_{control}_{mechanism}",
        "result_stage": "DEVELOPMENT_VALIDATION",
        "run_mode": f"FROZEN_{run_stage}",
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
        "seed": seed,
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
        "notes": f"stage={run_stage}; physical_batch={physical_batch_size}; mechanism={mechanism}; final_pass_8_fixed; no TEST/Evaluation access",
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
