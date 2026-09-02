#!/usr/bin/env python3
"""Train one frozen XEditCritic V4 screen run after its launch barriers pass."""

from __future__ import annotations

import argparse
import json
import math
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
    require_frozen_bottom_encoder_chunk_cache_identity_receipt_v4,
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
from core.route2_xeditcritic_pair_mean_v6 import (
    MPRAU_TASK_ID,
    apply_pair_mean_targets_v6,
    apply_rank_gaussian_targets_v6,
    extended_validation_metrics_v6,
    pair_key_v6,
)
from core.route2_xeditcritic_training_v4 import (
    FixedEffectiveTaskBatchSamplerV4,
    backward_replayed_prediction_gradient_v4,
    backward_retained_effective_batch_v4,
    collect_replayable_predictions_v4,
    critic_v4_learning_rate_factor,
    critic_v4_loss_weights,
    critic_v4_optimizer_parameter_groups,
    effective_prediction_objective_v4,
    forward_retained_effective_batch_v4,
    physical_microbatch_partitions_v4,
    require_physical_gpu_scope_v4,
)
from core.route2_xeditcritic_gradient_norm_scale_v7 import (
    GradientNormScalerV7,
    XEditCriticV7GradientNormScaleError,
)
from core.route2_mrnabert_lora_v3 import LoRALinearV3
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


def _write_atomic_terminal_v4(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists(), f"Critic V4 terminal artifact already exists: {path}")
    partial = path.with_suffix(path.suffix + ".partial")
    _require(
        not partial.exists(),
        f"Critic V4 partial terminal artifact already exists: {partial}",
    )
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


@dataclass(frozen=True)
class ScreenRunSpecV4:
    run_id: str
    model_kind: str
    control_mode: str
    mechanism_mode: str
    candidate_bundle_permutation: bool
    selectable: bool
    lambda_pairwise_weight: float | None = None


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
    per_run_lambda = row.get("lambda_pairwise_weight")
    _require(
        per_run_lambda is None or (isinstance(per_run_lambda, (int, float)) and float(per_run_lambda) >= 0.0),
        "per-run lambda_pairwise_weight must be a non-negative float or absent",
    )
    return ScreenRunSpecV4(
        run_id=str(row["run_id"]),
        model_kind=str(row["model"]),
        control_mode=control,
        mechanism_mode=str(row["mechanism"]),
        candidate_bundle_permutation=permutation,
        selectable=bool(row["selectable"]),
        lambda_pairwise_weight=(
            float(per_run_lambda) if per_run_lambda is not None else None
        ),
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


def _require_bottom_six_preflight_identity_v4(
    config: Mapping[str, Any], preflight: Mapping[str, Any]
) -> None:
    geometry = config["data_geometry"]
    require_frozen_bottom_encoder_chunk_cache_identity_receipt_v4(
        preflight.get("bottom_six_cache_identity"),
        expected_model_id=str(config["model_id"]),
        expected_record_count=int(
            geometry.get(
                "bottom_six_cache_record_count", geometry["expected_record_count"]
            )
        ),
        expected_unique_sequence_count=int(
            geometry.get("bottom_six_cache_unique_sequence_count", 43730)
        ),
        expected_embedding_width=int(config["architecture"]["pretrained_width"]),
    )


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
    _require(
        str(authorization.get("preflight_runner_git_head"))
        == str(preflight.get("git_head")),
        "Critic V4 authorization is bound to another preflight runner HEAD",
    )
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
        "cache_online_equivalence_passed",
    )
    _require(all(barriers.get(key) is True for key in required_true), "a Critic V4 launch barrier is not satisfied")
    _require(int(authorization.get("development_test_outcome_reads", -1)) == 0, "launch authorization reports a Development TEST read")
    _require(int(authorization.get("new_final_evaluation_outcome_reads", -1)) == 0, "launch authorization reports a new Evaluation read")
    _require(preflight.get("status") == "XEDITCRITIC_V4_PREFLIGHT_PASS", "formal Critic V4 preflight did not pass")
    _require(preflight.get("passed") is True, "formal Critic V4 preflight pass flag is absent")
    _require(int(preflight.get("selected_physical_batch", -1)) == int(physical_batch_size), "runner physical batch differs from frozen preflight selection")
    count = int(preflight.get("trainable_parameter_count", -1))
    _require(165_000_000 <= count <= 175_000_000, "formal preflight parameter count missed the frozen design target")
    selected_peak = float(preflight.get("selected_peak_allocated_gib", -1))
    _require(
        math.isfinite(selected_peak) and 0.0 < selected_peak <= 35.0,
        "formal preflight peak memory is nonpositive, nonfinite, or above 35 GiB",
    )
    _require(int(preflight.get("development_test_outcome_reads", -1)) == 0, "preflight reports a Development TEST read")
    _require(int(preflight.get("new_final_evaluation_outcome_reads", -1)) == 0, "preflight reports a new Evaluation read")
    _require_bottom_six_preflight_identity_v4(config, preflight)


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
        and str(config.get("confirmation_runner_git_head"))
        == str(current_git_head)
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
        and math.isfinite(float(preflight.get("selected_peak_allocated_gib", -1)))
        and 0.0 < float(preflight.get("selected_peak_allocated_gib", -1)) <= 35.0,
        "Critic V4 confirmation preflight identity changed",
    )
    _require_bottom_six_preflight_identity_v4(config, preflight)
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
        and str(config.get("posttest_runner_git_head")) == str(current_git_head)
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
        and math.isfinite(float(preflight.get("selected_peak_allocated_gib", -1)))
        and 0.0 < float(preflight.get("selected_peak_allocated_gib", -1)) <= 35.0
        and int(preflight.get("development_test_outcome_reads", -1)) == 0
        and int(preflight.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "Critic V4 posttest preflight identity changed",
    )
    _require_bottom_six_preflight_identity_v4(config, preflight)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_worktree_v4() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(not status, "Critic V4 training requires a clean Git worktree")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _terminal_training_environment_v4(
    *,
    training_seed: int,
    device: torch.device,
    training_git_head: str,
    spec: ScreenRunSpecV4,
) -> dict[str, Any]:
    """Capture fail-closed device and initialization provenance before build."""

    _require(device.type == "cuda", "Critic V4 parameter updates require CUDA")
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    device_name = torch.cuda.get_device_name(device)
    _require("A100" in device_name, "formal Critic V4 updates require an A100")
    _require(
        torch.cuda.is_bf16_supported(),
        "Critic V4 parameter updates require CUDA BF16 support",
    )
    if spec.model_kind == "C0-V4":
        tensor_identity_scope = "NOT_CLAIMED_DIFFERENT_C0_ARCHITECTURE"
    elif spec.mechanism_mode == "NO_CROSS":
        tensor_identity_scope = "NOT_CLAIMED_PARAMETER_MATCHED_DIFFERENT_MODULE"
    else:
        tensor_identity_scope = "SHARED_V4_CONSTRUCTOR_WITHIN_IDENTICAL_ARCHITECTURE"
    return {
        "parameter_initialization_seed": int(training_seed),
        "parameter_initialization_seed_applied_before_model_construction": True,
        "parameter_initialization_tensor_identity_scope": tensor_identity_scope,
        "cuda_available": True,
        "cuda_device": str(device),
        "cuda_device_name": device_name,
        "a100_device_verified": True,
        "bf16_supported": True,
        "cpu_fallback_used": False,
        "training_git_head": str(training_git_head),
    }


CPU_RAGGED_STRUCTURE_KEYS_V4 = frozenset(
    {
        "cache_record_indices",
        "cache_chunk_indices",
        "record_edit_offsets",
        "edit_positions",
        "edit_source_chunk_indices",
        "edit_candidate_chunk_indices",
        "edit_source_token_centers",
        "edit_candidate_token_centers",
        "edit_source_window_starts",
        "edit_source_window_ends",
        "edit_candidate_window_starts",
        "edit_candidate_window_ends",
    }
)


def _move(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        and key not in CPU_RAGGED_STRUCTURE_KEYS_V4
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
    result = {"mean": mean, "router_balance_loss": router_balance}
    if "cell_offset" in output:
        result["cell_offset"] = output["cell_offset"]
    return result


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
            batch_target_values = batch_targets.cpu().tolist()
            batch_prediction_values = prediction.cpu().tolist()
            batch_scaled_target_values = batch_scaled_targets.cpu().tolist()
            batch_scaled_prediction_values = scaled_prediction.cpu().tolist()
            targets.extend(batch_target_values)
            predictions.extend(batch_prediction_values)
            scaled_targets.extend(batch_scaled_target_values)
            scaled_predictions.extend(batch_scaled_prediction_values)
            tasks.extend(batch["task_ids"][:valid_count])
            for index in range(valid_count):
                rows.append(
                    {
                        "record_id": batch["record_ids"][index],
                        "source_group_id": batch["source_groups"][index],
                        "task_id": batch["task_ids"][index],
                        "target": float(batch_target_values[index]),
                        "prediction": float(batch_prediction_values[index]),
                        "scaled_target": float(batch_scaled_target_values[index]),
                        "scaled_prediction": float(
                            batch_scaled_prediction_values[index]
                        ),
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


def _apply_w1_finetune_policy(
    model: torch.nn.Module, w1_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Freeze + init + (optional) LoRA policy for the W1' two-stage arm."""

    mode = str(w1_config["mode"])
    _require(mode in {"head_only", "lora_top_six_head"}, "unknown W1 finetune mode")
    init_path = Path(str(w1_config["init_checkpoint"]))
    _require(init_path.is_file(), "W1 init checkpoint is absent")
    payload = torch.load(init_path, map_location="cpu", weights_only=False)
    _require(
        isinstance(payload, dict) and "model_state_dict" in payload,
        "W1 init checkpoint lacks a model state dict",
    )
    state = payload["model_state_dict"]
    total_built = sum(value.numel() for value in model.state_dict().values())
    total_init = sum(value.numel() for value in state.values())
    _require(
        total_built == total_init,
        "W1 init checkpoint total parameter count differs from the built model",
    )
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    wrapped = 0
    if mode == "lora_top_six_head":
        rank = int(w1_config.get("lora_rank", 16))
        alpha = float(w1_config.get("lora_alpha", 32.0))
        dropout = float(w1_config.get("lora_dropout", 0.05))
        for layer in model.upper_encoder.layers:
            for attr_path in (
                ("attention", "self", "Wqkv"),
                ("attention", "output", "dense"),
                ("intermediate", "dense"),
                ("output", "dense"),
            ):
                parent = layer
                for step in attr_path[:-1]:
                    parent = getattr(parent, step)
                base = getattr(parent, attr_path[-1])
                if isinstance(base, torch.nn.Linear):
                    setattr(
                        parent,
                        attr_path[-1],
                        LoRALinearV3(base, rank=rank, alpha=alpha, dropout=dropout),
                    )
                    wrapped += 1
        _require(wrapped > 0, "W1 LoRA wrapping found no target Linears")
    trainable_modules: list[str] = []
    for name in ("readout", "effect_head"):
        module = getattr(model, name, None)
        if module is not None:
            module.requires_grad_(True)
            trainable_modules.append(name)
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    _require(trainable_count > 0, "W1 finetune left nothing trainable")
    _require(
        trainable_count
        <= int(w1_config.get("maximum_trainable_parameter_count", 30_000_000)),
        "W1 trainable budget exceeded",
    )
    return {
        "mode": mode,
        "init_checkpoint": str(init_path),
        "init_checkpoint_total_parameter_count": total_init,
        "w1_trainable_parameter_count": trainable_count,
        "w1_trainable_modules": trainable_modules,
        "w1_lora_wrapped_linears": wrapped,
        "w1_total_parameter_count_after_policy": sum(
            value.numel() for value in model.state_dict().values()
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
        cell_offset_head=bool(architecture.get("cell_offset_head", False)),
        cell_offset_hidden_width=int(architecture.get("cell_offset_hidden_width", 256)),
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
    w1_config: Mapping[str, Any] | None = None,
) -> tuple[torch.optim.Optimizer, list[float]]:
    rates = config["training"]["learning_rates"]
    if w1_config is not None:
        trainable = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        _require(bool(trainable), "W1 optimizer received no trainable parameters")
        groups: list[dict[str, object]] = [
            {
                "name": "W1_HEAD_AND_LORA",
                "params": trainable,
                "lr": float(rates["new_head_and_v4_trunk"]),
            }
        ]
        optimizer = torch.optim.AdamW(
            groups,
            weight_decay=float(config["training"]["weight_decay"]),
        )
        return optimizer, [float(rates["new_head_and_v4_trunk"])]
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
    training_attempt_id: str | None = None,
) -> dict[str, Any]:
    spec = screen_run_spec_v4(config, run_id)
    run_stage, training_seed = critic_v4_run_stage_seed(config, run_id)
    current_head = _git_head()
    _require_clean_worktree_v4()
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
    require_physical_gpu_scope_v4(config, physical_gpu_index)
    _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden")
    _set_seed(training_seed)
    device = require_cuda(physical_gpu_index)
    terminal_training_environment = _terminal_training_environment_v4(
        training_seed=training_seed,
        device=device,
        training_git_head=current_head,
        spec=spec,
    )
    output_directory = Path(config["output_root"]) / run_id
    _require(not output_directory.exists(), f"Critic V4 run directory already exists: {output_directory}")
    output_directory.mkdir(parents=True)
    started = time.time()
    ledger_path, attempt_path = critic_v4_ledger_paths(config, output_directory)
    checkpoint_path = output_directory / "final_pass_8_checkpoint.pt"
    training_summary_path = output_directory / "run_summary.json"
    terminal_paths = {
        "output_directory": str(output_directory),
        "training_summary_path": str(training_summary_path),
        "checkpoint_path": str(checkpoint_path),
        "training_attempt_path": str(attempt_path),
    }
    attempt_config = {
        **critic_v4_attempt_config(
            config,
            run_id=run_id,
            physical_gpu_index=physical_gpu_index,
            physical_batch_size=physical_batch_size,
            training_attempt_id=training_attempt_id,
        ),
        **terminal_training_environment,
        **terminal_paths,
    }
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
        all_records = records_from_projection_rows(projection_rows)
        geometry = config["data_geometry"]
        study_filter = config.get("study_filter")
        if study_filter is not None:
            allowed_studies = {str(study) for study in study_filter}
            _require(bool(allowed_studies), "study_filter must name at least one study")
            records = [
                record for record in all_records if record.study in allowed_studies
            ]
            _require(bool(records), "study_filter removed every projection record")
        else:
            records = all_records
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
        # Apply W1-a pair-mean label aggregation before scaling.  D1 counts
        # the 12,048 VALIDATION MPRAU rows collapsing to 2,008 pair means, so
        # both splits adopt the shared-effect label (row count unchanged).
        pair_mean_label_map: dict[str, float] | None = None
        if bool(config["training"].get("pair_mean_targets", False)):
            train_records, train_pair_map = apply_pair_mean_targets_v6(
                train_records, pair_tasks=None
            )
            validation_records, validation_pair_map = apply_pair_mean_targets_v6(
                validation_records, pair_tasks=None
            )
            pair_mean_label_map = {**train_pair_map, **validation_pair_map}
        # Apply W1-c rank-Gaussian transform before scaling (train labels only,
        # per the handover "训练时任务内标签 rank-Gaussian/分位数变换").
        rank_gaussian_metadata: dict[str, object] | None = None
        if bool(config["training"].get("per_task_rank_gaussian", False)):
            train_records, rank_gaussian_metadata = apply_rank_gaussian_targets_v6(
                train_records, rank_tasks=None
            )
        _require(len(train_records) == int(geometry["expected_train_count"]), "TRAIN count changed")
        _require(len(validation_records) == int(geometry["expected_validation_count"]), "VALIDATION count changed")
        record_by_id = {record.record_id: record for record in records}
        # W0 single-task diagnosis: vocab is built over the full projection so
        # the model capacity (study/assay calibration heads) stays identical to
        # the multi-task screen preflight; only the training splits are filtered.
        vocabs = build_vocabs(records if study_filter is None else all_records)
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
                    expected_record_count=int(
                        geometry.get(
                            "bottom_six_cache_record_count",
                            geometry["expected_record_count"],
                        )
                    ),
                    expected_unique_sequence_count=int(
                        geometry.get("bottom_six_cache_unique_sequence_count", 43730)
                    ),
                    expected_embedding_width=int(
                        config["architecture"]["pretrained_width"]
                    ),
                    validate_payload=False,
                )
            )
            cache = FrozenBottomEncoderChunkCacheViewV4(
                cache_payload,
                set(record_by_id),
                validate_payload=False,
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
        w1_config = config.get("w1_finetune")
        w1_policy_summary: dict[str, Any] | None = None
        if w1_config is not None:
            w1_policy_summary = _apply_w1_finetune_policy(model, w1_config)
            capacity = dict(capacity)
            capacity["w1_full_backbone_trainable_parameter_count"] = capacity[
                "trainable_parameter_count"
            ]
            capacity["trainable_parameter_count"] = int(
                w1_policy_summary["w1_trainable_parameter_count"]
            )
            capacity["w1_finetune"] = w1_policy_summary
        if w1_config is None:
            _require(
                spec.model_kind == "C0-V4"
                or int(capacity["trainable_parameter_count"])
                == int(preflight["trainable_parameter_count"]),
                "formal model parameter count differs from preflight",
            )
        else:
            assert w1_policy_summary is not None
            _require(
                int(w1_policy_summary["w1_total_parameter_count_after_policy"])
                == int(w1_policy_summary["init_checkpoint_total_parameter_count"]),
                "W1 policy changed the total parameter count",
            )
        optimizer, initial_learning_rates = _build_optimizer(
            model, config, is_c0=spec.model_kind == "C0-V4", w1_config=w1_config
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
        initial_parameter = next(
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ).detach().clone()
        update_count = 0
        pass_rows: list[dict[str, Any]] = []
        gradient_norm_scale_config = config["training"].get("gradient_norm_scale", None)
        gradient_norm_scaler: GradientNormScalerV7 | None = None
        if gradient_norm_scale_config is not None:
            _require(
                isinstance(gradient_norm_scale_config, dict)
                and bool(gradient_norm_scale_config.get("enabled", False)),
                "gradient_norm_scale must be an enabled dict or absent",
            )
            gradient_norm_scaler = GradientNormScalerV7(
                ema_alpha=float(gradient_norm_scale_config.get("ema_alpha", 0.05)),
                floor=float(gradient_norm_scale_config.get("floor", 1e-6)),
            )
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
                retained_graph = None
                if len(physical_batches) == 1:
                    retained_graph = forward_retained_effective_batch_v4(
                        physical_batches,
                        forward=lambda batch: _forward_bf16(model, batch),
                    )
                    predictions = retained_graph.objective_predictions
                else:
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
                cell_offset_targets = None
                if float(config["training"].get("cell_offset_weight", 0.0)) > 0.0:
                    _require(
                        pair_mean_label_map is not None,
                        "Critic V6 cell-offset target requires pair-mean labels",
                    )
                    # W1-b: the auxiliary head predicts each record's deviation
                    # from its six-cell pair mean, expressed in scaled units.
                    clips: list[torch.Tensor] = []
                    for batch in physical_batches:
                        scales = batch["target_scale"].float()
                        values = torch.tensor(
                            [
                                (
                                    record_by_id[record_id].target
                                    - pair_mean_label_map[record_id]
                                )
                                / float(scale)
                                for record_id, scale in zip(
                                    batch["record_ids"], scales.tolist()
                                )
                            ],
                            dtype=torch.float32,
                            device=scales.device,
                        )
                        clips.append(values)
                    cell_offset_targets = torch.cat(clips)
                objective = effective_prediction_objective_v4(
                    predictions,
                    targets,
                    sample_weights,
                    source_groups,
                    task_ids,
                    pass_number=pass_number,
                    huber_delta=float(config["training"]["huber_delta"]),
                    soft_rank_temperature=float(config["training"]["soft_rank_temperature"]),
                    within_source_ranking_weight=float(
                        config["training"].get("within_source_ranking_weight", 0.0)
                    ),
                    lambda_pairwise_weight=float(
                        spec.lambda_pairwise_weight
                        if spec.lambda_pairwise_weight is not None
                        else config["training"].get("lambda_pairwise_weight", 0.0)
                    ),
                    cell_offset_predictions=(
                        retained_graph.cell_offset.detach().float()
                        if retained_graph is not None
                        and retained_graph.cell_offset is not None
                        else None
                    ),
                    cell_offset_targets=cell_offset_targets,
                    cell_offset_weight=float(
                        config["training"].get("cell_offset_weight", 0.0)
                    ),
                )
                gradient_for_backward = objective.prediction_gradient
                gradient_norm_scale_event = None
                if gradient_norm_scaler is not None:
                    _require(
                        len(set(task_ids)) == 1,
                        "gradient-norm scaling requires task-homogeneous batches",
                    )
                    gradient_for_backward, multiplier, norm = gradient_norm_scaler.scale(
                        str(task_ids[0]), gradient_for_backward
                    )
                    gradient_norm_scale_event = {
                        "task_id": str(task_ids[0]),
                        "gradient_norm": norm,
                        "multiplier": multiplier,
                    }
                optimizer.zero_grad(set_to_none=True)
                router_balance_weight = float(
                    critic_v4_loss_weights(pass_number)["router_balance"]
                )
                if retained_graph is not None:
                    backward_retained_effective_batch_v4(
                        retained_graph,
                        gradient_for_backward,
                        router_balance_weight=router_balance_weight,
                        cell_offset_gradient=objective.cell_offset_gradient,
                    )
                else:
                    # The W1-b cell-offset head sails through the retained
                    # batch-32 graph; the sub-four replay path keeps V5's
                    # mean-only contract.
                    _require(
                        objective.cell_offset_gradient is None,
                        "Critic V6 cell-offset requires the retained batch-32 path",
                    )
                    backward_replayed_prediction_gradient_v4(
                        physical_batches,
                        states,
                        first_pass_predictions,
                        gradient_for_backward,
                        device=device,
                        forward=lambda batch: _forward_bf16(model, batch),
                        router_balance_weight=router_balance_weight,
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
            pass_row = {
                    "pass": pass_number,
                    "update_count_cumulative": update_count,
                    "mean_task_objective_excluding_router_balance": float(np.mean(task_losses)),
                    "mean_huber_loss": float(np.mean(huber_losses)),
                    "mean_pairwise_loss": float(np.mean(pairwise_losses)),
                    "mean_soft_spearman_loss": float(np.mean(soft_losses)),
                    "mean_pair_count": float(np.mean(pair_counts)),
                    "validation_metric_read": False,
                }
            if bool(config["training"].get("per_pass_validation", False)):
                # W1-d: persist a checkpoint and a validation metric for the
                # just-finished pass (H1 prerequisite).  The final pass reuses
                # the canonical final prediction artifact below; earlier passes
                # write pass-scoped artifacts that never touch TEST/Eval.
                pass_checkpoint_path = output_directory / f"pass_{pass_number}_checkpoint.pt"
                torch.save(
                    {
                        "schema_version": f"route_a_v3_route2_xeditcritic_v4_{run_stage.lower()}_checkpoint.v2",
                        "run_stage": run_stage,
                        "run_id": run_id,
                        "model_kind": spec.model_kind,
                        "control_mode": spec.control_mode,
                        "mechanism_mode": spec.mechanism_mode,
                        "candidate_bundle_permutation": spec.candidate_bundle_permutation,
                        "seed": training_seed,
                        "physical_gpu_index": physical_gpu_index,
                        "precision": "BF16_FORWARD_FP32_EFFECTIVE_OBJECTIVE",
                        "selected_pass": pass_number,
                        "selection_policy": "PER_PASS_W1_D",
                        "model_state_dict": model.state_dict(),
                        "vocabs": vocabs,
                        "target_scaler": scaler.to_dict(),
                        "capacity": capacity,
                        "physical_batch_size": physical_batch_size,
                        "effective_batch_size": 32,
                        "retained_graph_fast_path": physical_batch_size == 32,
                        "full_cache_validation_per_batch": False,
                        "development_test_outcome_reads": 0,
                        "new_final_evaluation_outcome_reads": 0,
                    },
                    pass_checkpoint_path,
                )
                if validation_dataset is not None:
                    pass_prediction_path = (
                        output_directory / f"pass_{pass_number}_validation_predictions.jsonl"
                    )
                    pass_metrics = _evaluate(
                        model,
                        validation_dataset,
                        collator,
                        physical_batch_size=physical_batch_size,
                        device=device,
                        prediction_path=pass_prediction_path,
                    )
                    pass_row["validation_metric_read"] = True
                    pass_row["validation_metrics"] = pass_metrics
            pass_rows.append(pass_row)
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
        extended_metrics: dict[str, object] | None = None
        if (
            bool(config["training"].get("extended_validation_metrics", False))
            and final_metrics is not None
            and prediction_path is not None
        ):
            # W1-e: within-source rho, pair-mean rho + ceiling ratio, hit@K / NDCG@K
            # over the measured pair neighborhood, plus Tier-B markers.  Read the
            # already-written prediction rows (never TEST/Eval) and pair keys from
            # the validation records.
            pair_key_by_record_id = {
                record.record_id: pair_key_v6(record)
                for record in validation_records
            }
            prediction_rows: list[dict[str, Any]] = []
            with prediction_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    prediction_rows.append(json.loads(line))
            ceiling_by_task = None
            if config["training"].get("ceiling_by_task"):
                ceiling_by_task = {
                    str(task): float(ceiling)
                    for task, ceiling in config["training"]["ceiling_by_task"].items()
                }
            extended_metrics = extended_validation_metrics_v6(
                [row["target"] for row in prediction_rows],
                [row["prediction"] for row in prediction_rows],
                [str(row["task_id"]) for row in prediction_rows],
                [str(row["source_group_id"]) for row in prediction_rows],
                [
                    pair_key_by_record_id[str(row["record_id"])]
                    for row in prediction_rows
                ],
                ceiling_by_task=ceiling_by_task,
            )
        selection_policy = posttest_selection_policy_v4(run_stage)
        torch.save(
            {
                "schema_version": f"route_a_v3_route2_xeditcritic_v4_{run_stage.lower()}_checkpoint.v2",
                "run_stage": run_stage,
                "run_id": run_id,
                "model_kind": spec.model_kind,
                "control_mode": spec.control_mode,
                "mechanism_mode": spec.mechanism_mode,
                "candidate_bundle_permutation": spec.candidate_bundle_permutation,
                "seed": training_seed,
                "physical_gpu_index": physical_gpu_index,
                "precision": "BF16_FORWARD_FP32_EFFECTIVE_OBJECTIVE",
                "selected_pass": 8,
                "selection_policy": selection_policy,
                "model_state_dict": model.state_dict(),
                "vocabs": vocabs,
                "target_scaler": scaler.to_dict(),
                "capacity": capacity,
                "physical_batch_size": physical_batch_size,
                "effective_batch_size": 32,
                "retained_graph_fast_path": physical_batch_size == 32,
                "full_cache_validation_per_batch": False,
                "validation_metrics": final_metrics,
                **terminal_training_environment,
                **terminal_paths,
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
            checkpoint_path,
        )
        summary = {
            "schema_version": f"route_a_v3_route2_xeditcritic_v4_{run_stage.lower()}_run.v2",
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
            **terminal_training_environment,
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
            "training_forward_count_per_update": (
                1 if physical_batch_size == 32 else 2 * math.ceil(32 / physical_batch_size)
            ),
            "retained_graph_fast_path": physical_batch_size == 32,
            "full_cache_validation_count_before_batching": (
                0 if spec.model_kind == "C0-V4" else 1
            ),
            "full_cache_validation_per_batch": False,
            "parameter_changed": parameter_changed,
            "singleton_forward_count": 0,
            "selection_policy": selection_policy,
            "sampler": {
                "policy": "SQRT_TASK_SIZE_TASK_HOMOGENEOUS_SOURCE_GROUP_BALANCED",
                "repeat_cap": 4,
                "updates_per_pass": int(geometry["updates_per_pass"]),
            },
            "target_scaler": scaler.to_dict(),
            "lambda_pairwise_weight": (
                float(spec.lambda_pairwise_weight)
                if spec.lambda_pairwise_weight is not None
                else float(config["training"].get("lambda_pairwise_weight", 0.0))
            ),
            "gradient_norm_scale": (
                {
                    **gradient_norm_scale_config,
                    "ema_norms": gradient_norm_scaler.ema_norms(),
                    "ema_reference": gradient_norm_scaler.reference(),
                }
                if gradient_norm_scaler is not None
                else None
            ),
            "passes": pass_rows,
            "final_validation": final_metrics,
            "extended_validation_metrics": extended_metrics,
            "pair_mean_targets_enabled": bool(
                config["training"].get("pair_mean_targets", False)
            ),
            "per_task_rank_gaussian_enabled": bool(
                config["training"].get("per_task_rank_gaussian", False)
            ),
            **terminal_paths,
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
        _write_atomic_terminal_v4(training_summary_path, summary)
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
                    **terminal_training_environment,
                    **terminal_paths,
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
        if not (output_directory / "run_summary.json").exists():
            _write_atomic_terminal_v4(output_directory / "failure.json", failure)
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
    parser.add_argument("--training-attempt-id")
    arguments = parser.parse_args()
    config = _load_json(arguments.config)
    print(
        json.dumps(
            run(
                config,
                run_id=arguments.run_id,
                physical_gpu_index=arguments.physical_gpu_index,
                launch_authorization_path=arguments.launch_authorization,
                training_attempt_id=arguments.training_attempt_id,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
