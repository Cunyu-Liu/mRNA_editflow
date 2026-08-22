"""Central-attempt metadata shared by Critic V3 cache and LoRA runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


POSTTEST_STUDIES_V3 = (
    "GSE200304",
    "GSE114002",
    "GSE149487",
    "GSE217518",
    "GSE186455",
    "GSE256185",
    "GSE269595",
)


def critic_v3_seed_and_stage(config: Mapping[str, Any]) -> tuple[int, str]:
    stage = str(config.get("run_stage", "SCREEN"))
    if stage not in {"SCREEN", "CONFIRMATION", "REFIT", "LOSO"}:
        raise ValueError(f"unsupported Critic V3 stage: {stage}")
    seed = int(config.get("seed", config["screen_seed"]))
    if stage == "SCREEN" and seed != 20260830:
        raise ValueError("Critic V3 screen seed differs from the freeze")
    if stage == "CONFIRMATION" and seed not in {20260831, 20260901, 20260902}:
        raise ValueError("Critic V3 confirmation seed is undeclared")
    if stage in {"REFIT", "LOSO"} and seed not in {20260831, 20260901, 20260902}:
        raise ValueError(f"Critic V3 {stage.lower()} seed is undeclared")
    return seed, stage


def require_critic_v3_confirmation_authorization(
    config: Mapping[str, Any], *, arm: str
) -> None:
    _, stage = critic_v3_seed_and_stage(config)
    if stage == "SCREEN":
        return
    if stage != "CONFIRMATION":
        raise ValueError("confirmation authorization called for a post-TEST stage")
    gate_path = Path(str(config["screen_gate_path"]))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    selected = str(gate.get("selected_arm"))
    if (
        gate.get("status") != "XEDITCRITIC_V3_SCREEN_PASS"
        or gate.get("confirmation_authorized") is not True
        or selected not in {"C2", "C3"}
    ):
        raise ValueError("Critic V3 screen does not authorize confirmation")
    if str(config.get("selected_arm")) != selected:
        raise ValueError("Critic V3 confirmation config selected arm differs from screen")
    if arm not in {"C0", selected}:
        raise ValueError("Critic V3 confirmation arm is not selected full model or matched C0")


def _atomic_frozen_test_gate(config: Mapping[str, Any]) -> Mapping[str, Any]:
    atomic = json.loads(
        Path(str(config["atomic_frozen_test_path"])).read_text(encoding="utf-8")
    )
    if atomic.get("status") != "ATOMIC_FROZEN_DEVELOPMENT_TEST_TERMINAL":
        raise ValueError("atomic frozen TEST artifact is not terminal")
    gate = atomic.get("frozen_test_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("atomic frozen TEST gate is absent")
    return gate


def require_critic_v3_posttest_authorization(
    config: Mapping[str, Any], *, arm: str
) -> None:
    _, stage = critic_v3_seed_and_stage(config)
    if stage not in {"REFIT", "LOSO"}:
        return
    confirmation = json.loads(
        Path(str(config["three_seed_gate_path"])).read_text(encoding="utf-8")
    )
    selected = str(confirmation.get("selected_arm"))
    if (
        confirmation.get("status") != "XEDITCRITIC_V3_THREE_SEED_PASS"
        or selected not in {"C2", "C3"}
        or str(config.get("selected_arm")) != selected
    ):
        raise ValueError("Critic V3 post-TEST stage lacks a passing three-seed selection")
    frozen_gate = _atomic_frozen_test_gate(config)
    if (
        frozen_gate.get("status") != "XEDITCRITIC_V3_FROZEN_TEST_PASS"
        or frozen_gate.get("all_development_refit_authorized") is not True
    ):
        raise ValueError("Critic V3 frozen TEST does not authorize all-Development refit")
    if stage == "REFIT":
        if arm != selected:
            raise ValueError("all-Development refit only authorizes the selected Critic arm")
        return
    if arm not in {"C0", selected}:
        raise ValueError("LOSO arm is not selected full model or matched C0")
    if str(config.get("held_out_study", "")) not in POSTTEST_STUDIES_V3:
        raise ValueError("LOSO held-out study is undeclared")
    refit = json.loads(
        Path(str(config["refit_manifest_path"])).read_text(encoding="utf-8")
    )
    if (
        refit.get("status") != "XEDITCRITIC_V3_ALL_DEVELOPMENT_REFIT_COMPLETE"
        or refit.get("required_seeds") != [20260831, 20260901, 20260902]
        or int(refit.get("completed_refit_count", -1)) != 3
    ):
        raise ValueError("LOSO remains blocked until all three refits complete")


def critic_v3_attempt_config(
    config: Mapping[str, Any],
    *,
    run_id: str,
    arm: str,
    control_mode: str,
    candidate_bundle_permutation: bool,
    physical_gpu_index: int,
) -> dict[str, Any]:
    seed, stage = critic_v3_seed_and_stage(config)
    control = (
        "CANDIDATE_BUNDLE_PERMUTATION"
        if candidate_bundle_permutation
        else control_mode
    )
    pretrained = arm in {"C1", "C2", "C3"}
    held_out_suffix = f"::{config['held_out_study']}" if stage == "LOSO" else ""
    result_stage = {
        "SCREEN": "DEVELOPMENT_VALIDATION",
        "CONFIRMATION": "DEVELOPMENT_VALIDATION",
        "REFIT": "ALL_DEVELOPMENT_REFIT",
        "LOSO": "DEVELOPMENT_LOSO",
    }[stage]
    return {
        **dict(config),
        "attempt_id": f"xeditcritic_v3_{stage.lower()}_seed{seed}::{run_id}{held_out_suffix}",
        "baseline_id": f"xeditcritic_v3_{run_id}_seed{seed}",
        "attempt_purpose": f"XEDITCRITIC_V3_{stage}",
        "scientific_role": f"XEDITCRITIC_V3_{stage}_{arm}_{control}",
        "result_stage": result_stage,
        "run_mode": f"FROZEN_{stage}",
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
        "seed": seed,
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
