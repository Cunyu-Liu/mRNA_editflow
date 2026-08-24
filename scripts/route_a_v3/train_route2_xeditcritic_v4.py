#!/usr/bin/env python3
"""Train one frozen XEditCritic V4 screen run after its launch barriers pass."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_bottom_encoder_chunk_cache_v4 import (
    load_frozen_bottom_encoder_chunk_cache_v4,
    require_frozen_bottom_encoder_chunk_cache_identity_v4,
)
from core.route2_development_projection_v3 import load_projection_rows
from core.route2_experiment_ledger import (
    build_training_attempt_row,
    record_training_attempt,
)
from core.route2_xeditcritic_batch_v4 import (
    FrozenBottomEncoderChunkCacheViewV4,
    XEditCriticCollatorV4,
    XEditCriticDatasetV4,
)
from core.route2_xeditcritic_ledger_v4 import (
    critic_v4_attempt_config,
    critic_v4_attempt_details,
    critic_v4_ledger_paths,
)
from core.route2_xeditcritic_training_data_v3 import (
    XEditCriticRecordV3,
    build_exact_source_task_candidate_bundle_permutation,
    build_vocabs,
    records_from_projection_rows,
)
from core.route2_xeditcritic_training_v4 import (
    FixedEffectiveTaskBatchSamplerV4,
    backward_replayed_prediction_gradient_v4,
    collect_replayable_predictions_v4,
    critic_v4_learning_rate_factor,
    critic_v4_loss_weights,
    critic_v4_optimizer_parameter_groups,
    effective_prediction_objective_v4,
    physical_microbatch_partitions_v4,
)
from core.route2_xeditcritic_v3 import XEditCriticV3
from core.route2_xeditcritic_v4 import (
    XEditCriticV4,
    require_v4_trainable_parameter_range,
)
from core.route2_xeditcritic_gate_v4 import (
    CONFIRMATION_SEEDS_V4,
    LOSO_STUDIES_V4,
)
from scripts.route_a_v3.route2_mrnabert_upper_six_encoder_v4 import (
    TrainableMRNABERTUpperSixEncoderV4,
)
from scripts.route_a_v3.train_route2_xeditcritic_v3 import (
    XEditCriticCollatorV3,
    XEditCriticDatasetV3,
    fit_task_robust_scaler,
    require_cuda,
    validation_metrics,
)


class XEditCriticTrainingV4RunnerError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticTrainingV4RunnerError(message)


@dataclass(frozen=True)
class ScreenRunSpecV4:
    run_id: str
    model_kind: str
    control_mode: str
    mechanism_mode: str
    candidate_bundle_permutation: bool
    selectable: bool


def screen_run_spec_v4(
    config: Mapping[str, Any], run_id: str
) -> ScreenRunSpecV4:
    matches = [
        row
        for row in config["required_screen_runs"]
        if str(row["run_id"]) == str(run_id)
    ]
    _require(len(matches) == 1, "run id is not one exact frozen Critic V4 screen run")
    row = matches[0]
    permutation = bool(row.get("candidate_bundle_permutation", False))
    control = "CANDIDATE_BUNDLE_PERMUTATION" if permutation else str(row["control"])
    return ScreenRunSpecV4(
        run_id=str(row["run_id"]),
        model_kind=str(row["model"]),
        control_mode=control,
        mechanism_mode=str(row["mechanism"]),
        candidate_bundle_permutation=permutation,
        selectable=bool(row["selectable"]),
    )


def evaluation_index_batches_v4(
    record_count: int, physical_batch_size: int
) -> list[tuple[list[int], int]]:
    """Pad only the final inference batch while preserving its valid row count."""

    _require(record_count > 0, "Validation record count is empty")
    _require(physical_batch_size in {4, 8, 16, 32}, "Validation physical batch is undeclared")
    result: list[tuple[list[int], int]] = []
    for start in range(0, record_count, physical_batch_size):
        indices = list(range(start, min(record_count, start + physical_batch_size)))
        valid_count = len(indices)
        if valid_count < physical_batch_size:
            pad_cursor = 0
            while len(indices) < physical_batch_size:
                indices.append(pad_cursor % record_count)
                pad_cursor += 1
        result.append((indices, valid_count))
    _require(sum(valid for _, valid in result) == record_count, "Validation batching changed the measured cohort")
    _require(all(len(indices) == physical_batch_size for indices, _ in result), "Validation physical geometry changed")
    return result


def require_screen_launch_authorization_v4(
    config: Mapping[str, Any],
    authorization: Mapping[str, Any],
    preflight: Mapping[str, Any],
    *,
    run_id: str,
    physical_batch_size: int,
    current_git_head: str,
) -> None:
    """Hard barrier checked before creating a run directory or ledger row."""

    frozen_run_ids = {str(row["run_id"]) for row in config["required_screen_runs"]}
    _require(
        authorization.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_screen_launch_authorization.v1",
        "Critic V4 launch authorization schema is absent",
    )
    _require(
        authorization.get("status") == "XEDITCRITIC_V4_SCREEN_LAUNCH_AUTHORIZED",
        "Critic V4 launch is not authorized",
    )
    _require(str(authorization.get("authorized_git_head")) == str(current_git_head), "Critic V4 authorization is for another Git HEAD")
    _require(set(authorization.get("authorized_run_ids", [])) == frozen_run_ids, "Critic V4 authorization does not cover the exact frozen package")
    _require(run_id in frozen_run_ids, "Critic V4 requested run is not authorized")
    barriers = authorization.get("barriers", {})
    required_true = (
        "all_five_c3_jobs_terminal",
        "c3_terminal_summaries_read_exactly_once",
        "a100_current_head_focused_tests_passed",
        "a100_current_head_v332_tests_passed",
        "bottom_six_cache_terminal_complete",
        "formal_parameter_preflight_passed",
        "formal_memory_preflight_passed",
    )
    _require(all(barriers.get(key) is True for key in required_true), "a Critic V4 launch barrier is not satisfied")
    _require(int(authorization.get("development_test_outcome_reads", -1)) == 0, "launch authorization reports a Development TEST read")
    _require(int(authorization.get("new_final_evaluation_outcome_reads", -1)) == 0, "launch authorization reports a new Evaluation read")
    _require(preflight.get("status") == "XEDITCRITIC_V4_PREFLIGHT_PASS", "formal Critic V4 preflight did not pass")
    _require(preflight.get("passed") is True, "formal Critic V4 preflight pass flag is absent")
    _require(int(preflight.get("selected_physical_batch", -1)) == int(physical_batch_size), "runner physical batch differs from frozen preflight selection")
    count = int(preflight.get("trainable_parameter_count", -1))
    _require(165_000_000 <= count <= 175_000_000, "formal preflight parameter count missed the frozen design target")
    _require(20.0 <= float(preflight.get("selected_peak_allocated_gib", -1)) <= 35.0, "formal preflight peak memory is outside 20–35 GiB")
    _require(int(preflight.get("development_test_outcome_reads", -1)) == 0, "preflight reports a Development TEST read")
    _require(int(preflight.get("new_final_evaluation_outcome_reads", -1)) == 0, "preflight reports a new Evaluation read")


def critic_v4_run_stage_seed(
    config: Mapping[str, Any], run_id: str
) -> tuple[str, int]:
    run_stage = str(config.get("run_stage", "SCREEN"))
    _require(
        run_stage in {"SCREEN", "CONFIRMATION", "REFIT", "LOSO"},
        "unknown Critic V4 run stage",
    )
    seed = (
        int(config["training"]["screen_seed"])
        if run_stage == "SCREEN"
        else int(config["training_seed"])
    )
    _require(
        seed == 20260907
        if run_stage == "SCREEN"
        else seed in set(CONFIRMATION_SEEDS_V4),
        "Critic V4 seed is undeclared",
    )
    if run_stage == "CONFIRMATION":
        _require(
            config.get("schema_version")
            == "route_a_v3_route2_xeditcritic_v4_confirmation_runtime.v1"
            and config.get("required_confirmation_run_ids")
            == ["v4_full", "c0_v4"]
            and run_id in {"v4_full", "c0_v4"},
            "Critic V4 confirmation run scope changed",
        )
    if run_stage in {"REFIT", "LOSO"}:
        expected = ["v4_full"] if run_stage == "REFIT" else ["v4_full", "c0_v4"]
        _require(
            config.get("schema_version")
            == "route_a_v3_route2_xeditcritic_v4_posttest_runtime.v1"
            and config.get("required_posttest_run_ids") == expected
            and run_id in expected,
            "Critic V4 posttest run scope changed",
        )
        if run_stage == "LOSO":
            _require(
                config.get("held_out_study") in LOSO_STUDIES_V4
                and config.get("held_out_study_scale_policy")
                == "UNKNOWN_STUDY_SCALE_FIXED_1",
                "Critic V4 LOSO holdout identity changed",
            )
    return run_stage, seed


def split_posttest_records_v4(
    records: Sequence[XEditCriticRecordV3],
    *,
    run_stage: str,
    held_out_study: str | None,
) -> tuple[list[XEditCriticRecordV3], list[XEditCriticRecordV3]]:
    _require(
        all(record.split in {"TRAIN", "VALIDATION"} for record in records),
        "Critic V4 posttest received a protected split",
    )
    if run_stage == "REFIT":
        _require(held_out_study is None, "Critic V4 refit unexpectedly declares a holdout")
        return [replace(record, split="TRAIN") for record in records], []
    _require(
        run_stage == "LOSO" and held_out_study in LOSO_STUDIES_V4,
        "Critic V4 posttest split stage or holdout changed",
    )
    train = [
        replace(record, split="TRAIN")
        for record in records
        if record.study != held_out_study
    ]
    validation = [
        replace(record, split="VALIDATION")
        for record in records
        if record.study == held_out_study
    ]
    _require(train and validation, "Critic V4 LOSO train or holdout records are empty")
    return train, validation


def posttest_selection_policy_v4(run_stage: str) -> str:
    return (
        "FINAL_PASS_8_FIXED_NO_TEST_OR_VALIDATION_SELECTION"
        if run_stage in {"REFIT", "LOSO"}
        else "FINAL_PASS_8_FIXED_NO_VALIDATION_PEAK_RESELECTION"
    )


def require_confirmation_launch_authorization_v4(
    config: Mapping[str, Any],
    authorization: Mapping[str, Any],
    preflight: Mapping[str, Any],
    screen_gate: Mapping[str, Any],
    *,
    run_id: str,
    physical_batch_size: int,
    current_git_head: str,
) -> None:
    run_stage, seed = critic_v4_run_stage_seed(config, run_id)
    _require(run_stage == "CONFIRMATION", "Critic V4 confirmation authorization used outside confirmation")
    _require(
        authorization.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_confirmation_launch_authorization.v1"
        and authorization.get("status")
        == "XEDITCRITIC_V4_CONFIRMATION_LAUNCH_AUTHORIZED"
        and str(authorization.get("authorized_git_head")) == str(current_git_head)
        and authorization.get("authorized_seeds")
        == [20260908, 20260909, 20260910]
        and authorization.get("authorized_run_ids") == ["v4_full", "c0_v4"]
        and seed in authorization.get("authorized_seeds", [])
        and run_id in authorization.get("authorized_run_ids", []),
        "Critic V4 confirmation launch authorization scope changed",
    )
    _require(
        screen_gate.get("status") == "XEDITCRITIC_V4_SCREEN_PASS"
        and screen_gate.get("passed") is True
        and screen_gate.get("confirmation_authorized") is True
        and screen_gate.get("development_test_authorized") is False,
        "Critic V4 screen gate does not authorize confirmation",
    )
    barriers = authorization.get("barriers", {})
    required = (
        "screen_gate_passed",
        "a100_current_head_focused_tests_passed",
        "a100_current_head_v332_tests_passed",
        "bottom_six_cache_terminal_complete",
        "formal_parameter_preflight_passed",
        "formal_memory_preflight_passed",
    )
    _require(all(barriers.get(key) is True for key in required), "a Critic V4 confirmation barrier is not satisfied")
    _require(
        preflight.get("status") == "XEDITCRITIC_V4_PREFLIGHT_PASS"
        and preflight.get("passed") is True
        and int(preflight.get("selected_physical_batch", -1)) == physical_batch_size
        and 165_000_000 <= int(preflight.get("trainable_parameter_count", -1)) <= 175_000_000
        and 20.0 <= float(preflight.get("selected_peak_allocated_gib", -1)) <= 35.0,
        "Critic V4 confirmation preflight identity changed",
    )
    for payload, label in (
        (authorization, "authorization"),
        (preflight, "preflight"),
        (screen_gate, "screen gate"),
    ):
        _require(
            int(payload.get("development_test_outcome_reads", -1)) == 0
            and int(payload.get("new_final_evaluation_outcome_reads", -1)) == 0,
            f"Critic V4 confirmation {label} reports a protected read",
        )


def require_posttest_launch_authorization_v4(
    config: Mapping[str, Any],
    authorization: Mapping[str, Any],
    preflight: Mapping[str, Any],
    three_seed_gate: Mapping[str, Any],
    posttest_receipt: Mapping[str, Any],
    *,
    run_id: str,
    physical_batch_size: int,
    current_git_head: str,
) -> None:
    run_stage, seed = critic_v4_run_stage_seed(config, run_id)
    _require(run_stage in {"REFIT", "LOSO"}, "Critic V4 posttest authorization used outside posttest")
    expected_runs = ["v4_full"] if run_stage == "REFIT" else ["v4_full", "c0_v4"]
    _require(
        authorization.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_posttest_launch_authorization.v1"
        and authorization.get("status")
        == f"XEDITCRITIC_V4_{run_stage}_LAUNCH_AUTHORIZED"
        and authorization.get("authorized_stage") == run_stage
        and str(authorization.get("authorized_git_head")) == str(current_git_head)
        and authorization.get("authorized_seeds") == list(CONFIRMATION_SEEDS_V4)
        and authorization.get("authorized_run_ids") == expected_runs
        and seed in authorization.get("authorized_seeds", [])
        and run_id in expected_runs
        and authorization.get("atomic_frozen_test_passed") is True
        and int(authorization.get("development_test_access_event_count_before_posttest", -1)) == 1
        and int(authorization.get("development_test_outcome_reads_during_posttest", -1)) == 0
        and int(authorization.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "Critic V4 posttest launch authorization scope changed",
    )
    _require(
        three_seed_gate.get("status") == "XEDITCRITIC_V4_THREE_SEED_PASS"
        and three_seed_gate.get("required_seeds") == list(CONFIRMATION_SEEDS_V4)
        and three_seed_gate.get("development_test_authorized") is True
        and three_seed_gate.get("atomic_development_test_only") is True,
        "Critic V4 posttest three-seed authority changed",
    )
    _require(
        posttest_receipt.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_posttest_authorization_receipt.v1"
        and posttest_receipt.get("status") == "XEDITCRITIC_V4_POSTTEST_AUTHORIZED"
        and posttest_receipt.get("required_seeds") == list(CONFIRMATION_SEEDS_V4)
        and posttest_receipt.get("frozen_test_gate_status")
        == "XEDITCRITIC_V4_FROZEN_TEST_PASS"
        and posttest_receipt.get("all_development_refit_authorized") is True
        and int(posttest_receipt.get("development_test_access_event_count", -1)) == 1
        and posttest_receipt.get("development_test_metrics_in_receipt") is False
        and posttest_receipt.get("general_test_projection_persisted") is False
        and posttest_receipt.get("test_bottom_six_cache_persisted") is False
        and posttest_receipt.get("new_final_evaluation_outcomes_accessed") is False,
        "Critic V4 posttest frozen TEST outcome-free receipt changed",
    )
    if run_stage == "LOSO":
        _require(
            authorization.get("authorized_held_out_studies") == list(LOSO_STUDIES_V4)
            and config.get("held_out_study")
            in authorization.get("authorized_held_out_studies", [])
            and authorization.get("all_three_refits_complete") is True,
            "Critic V4 LOSO authorization lacks the exact holdout/refit scope",
        )
        refit = _load_json(Path(config["refit_manifest_path"]))
        _require(
            refit.get("status") == "XEDITCRITIC_V4_ALL_DEVELOPMENT_REFIT_COMPLETE"
            and refit.get("required_seeds") == list(CONFIRMATION_SEEDS_V4)
            and int(refit.get("completed_refit_count", -1)) == 3
            and int(refit.get("refit_pass_count", -1)) == 8
            and refit.get("loso_authorized") is True,
            "Critic V4 LOSO refit predecessor changed",
        )
    _require(
        preflight.get("status") == "XEDITCRITIC_V4_PREFLIGHT_PASS"
        and preflight.get("passed") is True
        and int(preflight.get("selected_physical_batch", -1)) == physical_batch_size
        and 165_000_000
        <= int(preflight.get("trainable_parameter_count", -1))
        <= 175_000_000
        and 20.0 <= float(preflight.get("selected_peak_allocated_gib", -1)) <= 35.0
        and int(preflight.get("development_test_outcome_reads", -1)) == 0
        and int(preflight.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "Critic V4 posttest preflight identity changed",
    )


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _forward_bf16(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(batch)
    mean = output["mean"]
    router_balance = output.get("router_balance_loss")
    if router_balance is None:
        # C0-V4 has no semantic mixture.  A graph-connected exact zero keeps
        # the common pass schedule without inventing a baseline-only router.
        router_balance = mean.sum() * 0.0
    return {"mean": mean, "router_balance_loss": router_balance}


def _physical_batches(
    dataset: XEditCriticDatasetV3,
    collator: Any,
    effective_indices: Sequence[int],
    *,
    physical_batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    partitions = physical_microbatch_partitions_v4(
        effective_batch_size=len(effective_indices),
        physical_batch_size=physical_batch_size,
    )
    return [
        _move(
            collator([dataset[effective_indices[position]] for position in partition]),
            device,
        )
        for partition in partitions
    ]


def _evaluate(
    model: torch.nn.Module,
    dataset: XEditCriticDatasetV3,
    collator: Any,
    *,
    physical_batch_size: int,
    device: torch.device,
    prediction_path: Path,
) -> dict[str, Any]:
    _require(not prediction_path.exists(), "Validation prediction artifact already exists")
    targets: list[float] = []
    predictions: list[float] = []
    scaled_targets: list[float] = []
    scaled_predictions: list[float] = []
    tasks: list[str] = []
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for indices, valid_count in evaluation_index_batches_v4(
            len(dataset), physical_batch_size
        ):
            batch = _move(collator([dataset[index] for index in indices]), device)
            output = _forward_bf16(model, batch)
            scaled_prediction = output["mean"].float()[:valid_count]
            prediction = scaled_prediction * batch["target_scale"][:valid_count]
            batch_targets = batch["target"].float()[:valid_count]
            batch_scaled_targets = batch["scaled_target"].float()[:valid_count]
            targets.extend(batch_targets.cpu().tolist())
            predictions.extend(prediction.cpu().tolist())
            scaled_targets.extend(batch_scaled_targets.cpu().tolist())
            scaled_predictions.extend(scaled_prediction.cpu().tolist())
            tasks.extend(batch["task_ids"][:valid_count])
            for index in range(valid_count):
                rows.append(
                    {
                        "record_id": batch["record_ids"][index],
                        "source_group_id": batch["source_groups"][index],
                        "task_id": batch["task_ids"][index],
                        "target": float(batch_targets[index].cpu()),
                        "prediction": float(prediction[index].cpu()),
                        "scaled_target": float(batch_scaled_targets[index].cpu()),
                        "scaled_prediction": float(scaled_prediction[index].cpu()),
                    }
                )
    _require(len(rows) == len(dataset), "Validation padded rows entered the measured cohort")
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return validation_metrics(
        targets,
        predictions,
        scaled_targets,
        scaled_predictions,
        tasks,
    )


def _permutation_overrides(
    train_records: Sequence[XEditCriticRecordV3],
    validation_records: Sequence[XEditCriticRecordV3],
    *,
    seed: int,
    enabled: bool,
) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    if not enabled:
        return {}, {}, {
            "complete_candidate_bundle_permuted": False,
            "train_recipient_count": 0,
            "validation_recipient_count": 0,
            "eligible_tasks": [],
        }
    train, train_summary = build_exact_source_task_candidate_bundle_permutation(
        train_records, seed=seed
    )
    validation, validation_summary = build_exact_source_task_candidate_bundle_permutation(
        validation_records, seed=seed
    )
    return train, validation, {
        "complete_candidate_bundle_permuted": True,
        "exact_source_task_strata": True,
        "train_recipient_count": len(train),
        "validation_recipient_count": len(validation),
        "eligible_tasks": sorted(
            set(train_summary["eligible_tasks"])
            | set(validation_summary["eligible_tasks"])
        ),
    }


def _build_model(
    config: Mapping[str, Any],
    spec: ScreenRunSpecV4,
    vocabs: Mapping[str, Mapping[str, int]],
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    common_counts = {
        "study_count": len(vocabs["study"]),
        "assay_count": len(vocabs["assay"]),
        "context_count": len(vocabs["context"]),
        "quantity_count": len(vocabs["quantity"]),
        "measurement_count": len(vocabs["measurement"]),
        "numerator_count": len(vocabs["numerator"]),
        "denominator_count": len(vocabs["denominator"]),
        "region_count": 2,
    }
    architecture = config["architecture"]
    if spec.model_kind == "C0-V4":
        model = XEditCriticV3(
            arm="C0",
            control_mode="NONE",
            **common_counts,
            raw_hidden_dim=int(architecture["raw_hidden_dim"]),
            raw_depth=int(architecture["raw_depth"]),
            dropout=float(architecture["dropout"]),
        ).to(device)
        return model, {
            "trainable_parameter_count": model.trainable_parameter_count,
            "capacity_gate_not_applicable": "MATCHED_RAW_C0_V4_BASELINE",
        }
    _require(spec.model_kind == "V4-FULL", "unknown Critic V4 model kind")
    upper = TrainableMRNABERTUpperSixEncoderV4(
        Path(config["mrnabert_model_path"]),
        device,
        attention_backend=str(config["memory_preflight"]["attention_backend"]),
        activation_checkpointing=bool(config["memory_preflight"]["activation_checkpointing"]),
    )
    control_mode = (
        "NONE"
        if spec.control_mode == "CANDIDATE_BUNDLE_PERMUTATION"
        else spec.control_mode
    )
    model = XEditCriticV4(
        upper_encoder=upper,
        **common_counts,
        control_mode=control_mode,
        mechanism_mode=spec.mechanism_mode,
        pretrained_width=int(architecture["pretrained_width"]),
        model_width=int(architecture["model_width"]),
        block_count=int(architecture["edit_block_count"]),
        heads=int(architecture["attention_heads"]),
        ffn_width=int(architecture["ffn_width"]),
        expert_count=int(architecture["semantic_expert_count"]),
        expert_bottleneck_width=int(architecture["semantic_expert_bottleneck_width"]),
        expert_top_k=int(architecture["semantic_router_top_k"]),
        raw_hidden_dim=int(architecture["raw_hidden_dim"]),
        raw_depth=int(architecture["raw_depth"]),
        readout_hidden_width=int(architecture["readout_hidden_width"]),
        dropout=float(architecture["dropout"]),
        minimum_physical_batch=int(config["memory_preflight"]["minimum_physical_batch"]),
        activation_checkpointing=bool(config["memory_preflight"]["activation_checkpointing"]),
    ).to(device)
    capacity = require_v4_trainable_parameter_range(
        model,
        minimum=int(architecture["minimum_trainable_parameter_count"]),
        maximum=int(architecture["maximum_trainable_parameter_count"]),
        design_target_minimum=int(architecture["design_target_minimum_trainable_parameter_count"]),
        design_target_maximum=int(architecture["design_target_maximum_trainable_parameter_count"]),
    )
    capacity["upper_six_scope"] = upper.scope_summary()
    return model, capacity


def _build_optimizer(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    *,
    is_c0: bool,
) -> tuple[torch.optim.Optimizer, list[float]]:
    rates = config["training"]["learning_rates"]
    if is_c0:
        groups: list[dict[str, object]] = [
            {
                "name": "C0_V4_ENDPOINT_AWARE_RAW",
                "params": list(model.parameters()),
                "lr": float(rates["new_head_and_v4_trunk"]),
            }
        ]
    else:
        groups = critic_v4_optimizer_parameter_groups(
            model,
            head_learning_rate=float(rates["new_head_and_v4_trunk"]),
            semantic_learning_rate=float(rates["semantic_experts_and_router"]),
            upper_six_learning_rate=float(rates["mrnabert_top_six"]),
        )
    optimizer = torch.optim.AdamW(
        groups,
        weight_decay=float(config["training"]["weight_decay"]),
    )
    return optimizer, [float(group["lr"]) for group in optimizer.param_groups]


def run(
    config: Mapping[str, Any],
    *,
    run_id: str,
    physical_gpu_index: int,
    launch_authorization_path: Path,
) -> dict[str, Any]:
    spec = screen_run_spec_v4(config, run_id)
    run_stage, training_seed = critic_v4_run_stage_seed(config, run_id)
    current_head = _git_head()
    authorization = _load_json(launch_authorization_path)
    preflight = _load_json(Path(config["preflight_output"]))
    physical_batch_size = int(preflight.get("selected_physical_batch", -1))
    if run_stage == "SCREEN":
        require_screen_launch_authorization_v4(
            config,
            authorization,
            preflight,
            run_id=run_id,
            physical_batch_size=physical_batch_size,
            current_git_head=current_head,
        )
    elif run_stage == "CONFIRMATION":
        require_confirmation_launch_authorization_v4(
            config,
            authorization,
            preflight,
            _load_json(Path(config["screen_gate_path"])),
            run_id=run_id,
            physical_batch_size=physical_batch_size,
            current_git_head=current_head,
        )
    else:
        require_posttest_launch_authorization_v4(
            config,
            authorization,
            preflight,
            _load_json(Path(config["three_seed_gate_path"])),
            _load_json(Path(config["posttest_authorization_receipt_path"])),
            run_id=run_id,
            physical_batch_size=physical_batch_size,
            current_git_head=current_head,
        )
    _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden")
    _set_seed(training_seed)
    device = require_cuda(physical_gpu_index)
    output_directory = Path(config["output_root"]) / run_id
    _require(not output_directory.exists(), f"Critic V4 run directory already exists: {output_directory}")
    output_directory.mkdir(parents=True)
    started = time.time()
    attempt_config = critic_v4_attempt_config(
        config,
        run_id=run_id,
        physical_gpu_index=physical_gpu_index,
        physical_batch_size=physical_batch_size,
    )
    ledger_path, attempt_path = critic_v4_ledger_paths(config, output_directory)
    attempt_details = critic_v4_attempt_details(
        config, physical_batch_size=physical_batch_size
    )
    record_training_attempt(
        ledger_path,
        attempt_path,
        build_training_attempt_row(
            attempt_config,
            output_directory,
            "RUNNING",
            repository_root=REPO_ROOT,
            details=attempt_details,
        ),
    )
    try:
        projection_rows = load_projection_rows(
            [Path(path) for path in config["projection_paths"]]
        )
        records = records_from_projection_rows(projection_rows)
        geometry = config["data_geometry"]
        _require(len(records) == int(geometry["expected_record_count"]), "projection record count changed")
        if run_stage in {"REFIT", "LOSO"}:
            train_records, validation_records = split_posttest_records_v4(
                records,
                run_stage=run_stage,
                held_out_study=config.get("held_out_study"),
            )
        else:
            train_records = [record for record in records if record.split == "TRAIN"]
            validation_records = [record for record in records if record.split == "VALIDATION"]
        _require(len(train_records) == int(geometry["expected_train_count"]), "TRAIN count changed")
        _require(len(validation_records) == int(geometry["expected_validation_count"]), "VALIDATION count changed")
        record_by_id = {record.record_id: record for record in records}
        vocabs = build_vocabs(records)
        scaler = fit_task_robust_scaler(
            train_records,
            floor=float(config["training"]["target_scale_floor"]),
        )
        train_overrides, validation_overrides, permutation_summary = _permutation_overrides(
            train_records,
            validation_records,
            seed=training_seed,
            enabled=spec.candidate_bundle_permutation,
        )
        neutral_studies = (
            {str(config["held_out_study"])} if run_stage == "LOSO" else set()
        )
        bottom_six_cache_identity: dict[str, Any] | None = None
        if spec.model_kind == "C0-V4":
            train_dataset: XEditCriticDatasetV3 = XEditCriticDatasetV3(
                train_records,
                all_records=record_by_id,
                vocabs=vocabs,
                target_scaler=scaler,
                cache=None,
                neutral_studies=neutral_studies,
            )
            validation_dataset: XEditCriticDatasetV3 | None = (
                XEditCriticDatasetV3(
                    validation_records,
                    all_records=record_by_id,
                    vocabs=vocabs,
                    target_scaler=scaler,
                    cache=None,
                    neutral_studies=neutral_studies,
                )
                if validation_records
                else None
            )
            collator: Any = XEditCriticCollatorV3(
                pretrained_width=int(config["architecture"]["pretrained_width"])
            )
        else:
            cache_payload = load_frozen_bottom_encoder_chunk_cache_v4(
                Path(config["bottom_six_cache"])
            )
            bottom_six_cache_identity = (
                require_frozen_bottom_encoder_chunk_cache_identity_v4(
                    cache_payload,
                    expected_model_id=str(config["model_id"]),
                    expected_record_count=int(geometry["expected_record_count"]),
                    expected_unique_sequence_count=43730,
                    expected_embedding_width=int(
                        config["architecture"]["pretrained_width"]
                    ),
                )
            )
            cache = FrozenBottomEncoderChunkCacheViewV4(
                cache_payload, set(record_by_id)
            )
            train_dataset = XEditCriticDatasetV4(
                train_records,
                all_records=record_by_id,
                vocabs=vocabs,
                target_scaler=scaler,
                cache=None,
                candidate_bundle_overrides=train_overrides,
                neutral_studies=neutral_studies,
            )
            validation_dataset = (
                XEditCriticDatasetV4(
                    validation_records,
                    all_records=record_by_id,
                    vocabs=vocabs,
                    target_scaler=scaler,
                    cache=None,
                    candidate_bundle_overrides=validation_overrides,
                    neutral_studies=neutral_studies,
                )
                if validation_records
                else None
            )
            collator = XEditCriticCollatorV4(
                cache,
                minimum_physical_batch=int(config["memory_preflight"]["minimum_physical_batch"]),
            )
        model, capacity = _build_model(config, spec, vocabs, device=device)
        _require(
            spec.model_kind == "C0-V4"
            or int(capacity["trainable_parameter_count"])
            == int(preflight["trainable_parameter_count"]),
            "formal model parameter count differs from preflight",
        )
        optimizer, initial_learning_rates = _build_optimizer(
            model, config, is_c0=spec.model_kind == "C0-V4"
        )
        attempt_details = critic_v4_attempt_details(
            config,
            trainable_parameter_count=int(capacity["trainable_parameter_count"]),
            physical_batch_size=physical_batch_size,
            peak_vram_mb=torch.cuda.max_memory_allocated(device) / 1024**2,
        )
        record_training_attempt(
            ledger_path,
            attempt_path,
            build_training_attempt_row(
                attempt_config,
                output_directory,
                "RUNNING",
                repository_root=REPO_ROOT,
                details=attempt_details,
            ),
        )
        sampler = FixedEffectiveTaskBatchSamplerV4(
            train_records,
            seed=training_seed,
            repeat_cap=int(geometry["maximum_record_repeats_per_pass"]),
            effective_batch=int(geometry["effective_batch_size"]),
        )
        initial_parameter = next(model.parameters()).detach().clone()
        update_count = 0
        pass_rows: list[dict[str, Any]] = []
        for pass_index in range(int(geometry["pass_count"])):
            pass_number = pass_index + 1
            sampler.set_pass(pass_index)
            effective_batches = sampler.batches_for_pass()
            _require(len(effective_batches) == int(geometry["updates_per_pass"]), "Critic V4 updates/pass changed")
            task_losses: list[float] = []
            huber_losses: list[float] = []
            pairwise_losses: list[float] = []
            soft_losses: list[float] = []
            pair_counts: list[int] = []
            model.train()
            for effective_indices in effective_batches:
                physical_batches = _physical_batches(
                    train_dataset,
                    collator,
                    effective_indices,
                    physical_batch_size=physical_batch_size,
                    device=device,
                )
                predictions, states, first_pass_predictions = collect_replayable_predictions_v4(
                    physical_batches,
                    device=device,
                    forward=lambda batch: _forward_bf16(model, batch),
                )
                targets = torch.cat(
                    [batch["scaled_target"].float() for batch in physical_batches]
                )
                sample_weights = torch.cat(
                    [batch["sample_weight"].float() for batch in physical_batches]
                )
                source_groups = [
                    value
                    for batch in physical_batches
                    for value in batch["source_groups"]
                ]
                task_ids = [
                    value
                    for batch in physical_batches
                    for value in batch["task_ids"]
                ]
                objective = effective_prediction_objective_v4(
                    predictions,
                    targets,
                    sample_weights,
                    source_groups,
                    task_ids,
                    pass_number=pass_number,
                    huber_delta=float(config["training"]["huber_delta"]),
                    soft_rank_temperature=float(config["training"]["soft_rank_temperature"]),
                )
                optimizer.zero_grad(set_to_none=True)
                backward_replayed_prediction_gradient_v4(
                    physical_batches,
                    states,
                    first_pass_predictions,
                    objective.prediction_gradient,
                    device=device,
                    forward=lambda batch: _forward_bf16(model, batch),
                    router_balance_weight=float(
                        critic_v4_loss_weights(pass_number)["router_balance"]
                    ),
                )
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(config["training"]["gradient_clip_norm"]),
                )
                _require(torch.isfinite(gradient_norm).item(), "Critic V4 gradient norm is nonfinite")
                factor = critic_v4_learning_rate_factor(
                    update_count,
                    total_updates=int(geometry["total_optimizer_updates"]),
                    warmup_fraction=float(config["training"]["warmup_fraction"]),
                )
                for group, initial_rate in zip(
                    optimizer.param_groups, initial_learning_rates, strict=True
                ):
                    group["lr"] = initial_rate * factor
                optimizer.step()
                update_count += 1
                task_losses.append(objective.total_loss)
                huber_losses.append(objective.huber_loss)
                pairwise_losses.append(objective.pairwise_loss)
                soft_losses.append(objective.soft_spearman_loss)
                pair_counts.append(objective.pair_count)
            pass_rows.append(
                {
                    "pass": pass_number,
                    "update_count_cumulative": update_count,
                    "mean_task_objective_excluding_router_balance": float(np.mean(task_losses)),
                    "mean_huber_loss": float(np.mean(huber_losses)),
                    "mean_pairwise_loss": float(np.mean(pairwise_losses)),
                    "mean_soft_spearman_loss": float(np.mean(soft_losses)),
                    "mean_pair_count": float(np.mean(pair_counts)),
                    "validation_metric_read": False,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "XEDITCRITIC_V4_PASS_COMPLETE_ALIVE_ONLY",
                        "run_id": run_id,
                        "pass": pass_number,
                        "update_count": update_count,
                        "cuda": True,
                        "active_performance_metric_emitted": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        _require(update_count == int(geometry["total_optimizer_updates"]), "Critic V4 total update budget changed")
        parameter_changed = not torch.equal(
            initial_parameter, next(model.parameters()).detach()
        )
        _require(parameter_changed, "Critic V4 performed no learned parameter update")
        prediction_path = (
            output_directory / "final_validation_predictions.jsonl"
            if validation_dataset is not None
            else None
        )
        final_metrics = (
            _evaluate(
                model,
                validation_dataset,
                collator,
                physical_batch_size=physical_batch_size,
                device=device,
                prediction_path=prediction_path,
            )
            if validation_dataset is not None and prediction_path is not None
            else None
        )
        checkpoint_path = output_directory / "final_pass_8_checkpoint.pt"
        selection_policy = posttest_selection_policy_v4(run_stage)
        torch.save(
            {
                "schema_version": f"route_a_v3_route2_xeditcritic_v4_{run_stage.lower()}_checkpoint.v1",
                "run_stage": run_stage,
                "run_id": run_id,
                "model_kind": spec.model_kind,
                "control_mode": spec.control_mode,
                "mechanism_mode": spec.mechanism_mode,
                "candidate_bundle_permutation": spec.candidate_bundle_permutation,
                "seed": training_seed,
                "selected_pass": 8,
                "selection_policy": selection_policy,
                "model_state_dict": model.state_dict(),
                "vocabs": vocabs,
                "target_scaler": scaler.to_dict(),
                "capacity": capacity,
                "physical_batch_size": physical_batch_size,
                "effective_batch_size": 32,
                "validation_metrics": final_metrics,
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
            checkpoint_path,
        )
        summary = {
            "schema_version": f"route_a_v3_route2_xeditcritic_v4_{run_stage.lower()}_run.v1",
            "status": f"TERMINAL_XEDITCRITIC_V4_{run_stage}_RUN_COMPLETE",
            "run_stage": run_stage,
            "run_id": run_id,
            "model_kind": spec.model_kind,
            "control_mode": spec.control_mode,
            "mechanism_mode": spec.mechanism_mode,
            "candidate_bundle_permutation": spec.candidate_bundle_permutation,
            "candidate_permutation_summary": permutation_summary,
            "selectable": spec.selectable,
            "seed": training_seed,
            "held_out_study": config.get("held_out_study"),
            "held_out_study_scale_policy": config.get(
                "held_out_study_scale_policy"
            ),
            "physical_gpu_index": physical_gpu_index,
            "cuda_device_name": torch.cuda.get_device_name(device),
            "precision": "BF16_FORWARD_FP32_EFFECTIVE_OBJECTIVE",
            "capacity": capacity,
            "bottom_six_cache_identity": bottom_six_cache_identity,
            "train_record_count": len(train_records),
            "validation_record_count": len(validation_records),
            "pass_count": 8,
            "selected_pass": 8,
            "update_count": update_count,
            "physical_batch_size": physical_batch_size,
            "effective_batch_size": 32,
            "parameter_changed": parameter_changed,
            "singleton_forward_count": 0,
            "cpu_fallback_used": False,
            "selection_policy": selection_policy,
            "sampler": {
                "policy": "SQRT_TASK_SIZE_TASK_HOMOGENEOUS_SOURCE_GROUP_BALANCED",
                "repeat_cap": 4,
                "updates_per_pass": int(geometry["updates_per_pass"]),
            },
            "target_scaler": scaler.to_dict(),
            "passes": pass_rows,
            "final_validation": final_metrics,
            "checkpoint_path": str(checkpoint_path),
            "validation_prediction_path": (
                str(prediction_path) if prediction_path is not None else None
            ),
            "launch_authorization_path": str(launch_authorization_path),
            "preflight_path": str(config["preflight_output"]),
            "elapsed_seconds": time.time() - started,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        }
        (output_directory / "run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        record_training_attempt(
            ledger_path,
            attempt_path,
            build_training_attempt_row(
                attempt_config,
                output_directory,
                "COMPLETED",
                repository_root=REPO_ROOT,
                details={
                    **attempt_details,
                    "optimizer_steps": update_count,
                    "selected_epoch": 8,
                    "validation_metrics": final_metrics,
                    "wall_time_seconds": summary["elapsed_seconds"],
                    "peak_vram_mb": summary["peak_vram_bytes"] / 1024**2,
                    "notes": f"terminal prospective Critic V4 {run_stage.lower()} run; final-pass-8 fixed; no TEST or Evaluation access",
                },
            ),
        )
        return summary
    except Exception as exc:
        failure = {
            "schema_version": f"route_a_v3_route2_xeditcritic_v4_{run_stage.lower()}_run_failure.v1",
            "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
            "run_stage": run_stage,
            "run_id": run_id,
            "model_kind": spec.model_kind,
            "control_mode": spec.control_mode,
            "mechanism_mode": spec.mechanism_mode,
            "candidate_bundle_permutation": spec.candidate_bundle_permutation,
            "seed": training_seed,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": time.time() - started,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        }
        (output_directory / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        record_training_attempt(
            ledger_path,
            attempt_path,
            build_training_attempt_row(
                attempt_config,
                output_directory,
                "FAILED",
                repository_root=REPO_ROOT,
                details={
                    **attempt_details,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "wall_time_seconds": failure["elapsed_seconds"],
                },
            ),
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--launch-authorization", required=True, type=Path)
    arguments = parser.parse_args()
    config = _load_json(arguments.config)
    print(
        json.dumps(
            run(
                config,
                run_id=arguments.run_id,
                physical_gpu_index=arguments.physical_gpu_index,
                launch_authorization_path=arguments.launch_authorization,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
