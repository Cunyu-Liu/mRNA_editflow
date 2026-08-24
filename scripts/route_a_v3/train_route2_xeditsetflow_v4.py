#!/usr/bin/env python3
"""Train one frozen XEditSetFlow V4 screen run without active Validation reads."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_development_projection_v3 import load_projection_rows
from core.route2_experiment_ledger import (
    build_training_attempt_row,
    record_training_attempt,
)
from core.route2_source_token_cache_v3 import (
    SourceTokenCacheIndexV3,
    load_source_token_cache_v3,
    require_source_token_cache_identity_v3,
)
from core.route2_xeditsetflow_runtime_v4 import (
    build_setflow_screen_model_v4,
    pad_source_batches_v4,
    require_setflow_v4_confirmation_launch_authorization,
    require_setflow_v4_screen_launch_authorization,
    screen_run_spec_v4,
    setflow_v4_learning_rate_factor,
)
from core.route2_xeditsetflow_training_v3 import (
    BalancedTaskSourceGroupPassSamplerV3,
)
from core.route2_xeditsetflow_training_v4 import (
    SetFlowSourceStateDatasetV4,
    collate_setflow_source_states_v4,
    expanded_source_state_batches_v4,
    setflow_source_records_from_projection_rows_v4,
    setflow_source_vocabs_v4,
)
from core.route2_xeditsetflow_v4 import mixture_setflow_loss_v4


class SetFlowTrainingV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SetFlowTrainingV4Error(message)


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


def record_failed_attempt_if_started_v4(
    config: Mapping[str, Any],
    output_directory: Path,
    error: Exception,
) -> bool:
    """Move an existing central RUNNING row to FAILED without adding an attempt."""

    attempt_path = output_directory / "training_attempt.json"
    if not attempt_path.exists():
        return False
    row = _load_json(attempt_path)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    failed = {
        **row,
        "status": "FAILED",
        "updated_at": now,
        "completed_at": now,
        "error_type": type(error).__name__,
        "error": str(error),
        "notes": "terminal SetFlow V4 implementation or runtime failure; result retained",
    }
    record_training_attempt(
        Path(config["experiment_ledger_path"]), attempt_path, failed
    )
    return True


def derive_training_update_geometry_v4(
    train_source_count: int, *, passes: int = 10
) -> dict[str, int]:
    _require(train_source_count >= 8, "SetFlow V4 has fewer than eight TRAIN sources")
    _require(passes == 10, "SetFlow V4 pass count changed")
    updates_per_pass = math.ceil(train_source_count / 8)
    return {
        "train_source_count": int(train_source_count),
        "sources_per_update": 8,
        "states_per_source": 4,
        "effective_state_batch": 32,
        "updates_per_pass": updates_per_pass,
        "pass_count": passes,
        "total_optimizer_updates": updates_per_pass * passes,
    }


def pass_complete_alive_event_v4(
    *, run_id: str, pass_number: int, update_count: int
) -> dict[str, Any]:
    return {
        "event": "XEDITSETFLOW_V4_PASS_COMPLETE_ALIVE_ONLY",
        "run_id": run_id,
        "pass": int(pass_number),
        "update_count": int(update_count),
        "cuda": True,
        "active_performance_metric_emitted": False,
    }


def _move(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _collate_indices(
    dataset: SetFlowSourceStateDatasetV4,
    indices: Sequence[tuple[int, int]],
    cache: SourceTokenCacheIndexV3,
) -> dict[str, Any]:
    return collate_setflow_source_states_v4(
        [dataset.state(source_index, state_slot) for source_index, state_slot in indices],
        source_cache=cache,
    )


def train(
    config: Mapping[str, Any],
    *,
    run_id: str,
    authorization_path: Path,
    output_directory: Path,
    physical_gpu_index: int,
) -> dict[str, Any]:
    spec = screen_run_spec_v4(config, run_id)
    run_stage = str(config.get("run_stage", "SCREEN"))
    _require(run_stage in {"SCREEN", "CONFIRMATION"}, "unknown SetFlow V4 run stage")
    training_seed = (
        int(config["training"]["screen_seed"])
        if run_stage == "SCREEN"
        else int(config["training_seed"])
    )
    current_head = _git_head()
    authorization = _load_json(authorization_path)
    preflight = _load_json(Path(config["preflight_output_path"]))
    source_data_audit = _load_json(Path(config["source_level_data_audit_path"]))
    if run_stage == "SCREEN":
        require_setflow_v4_screen_launch_authorization(
            config,
            authorization,
            preflight,
            source_data_audit,
            run_id=run_id,
            current_git_head=current_head,
        )
        _require(training_seed == 20260911, "SetFlow V4 screen seed changed")
    else:
        screen_gate = _load_json(Path(config["screen_gate_path"]))
        require_setflow_v4_confirmation_launch_authorization(
            config,
            authorization,
            preflight,
            source_data_audit,
            screen_gate,
            run_id=run_id,
            current_git_head=current_head,
        )
    _require(
        not output_directory.exists(),
        f"terminal SetFlow V4 output already exists: {output_directory}",
    )
    _require(
        physical_gpu_index in config["gpu_policy"]["physical_gpu_scope"],
        "SetFlow V4 GPU is outside 0–5",
    )
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    _require(
        not os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES remapping is forbidden",
    )
    device = torch.device(f"cuda:{physical_gpu_index}")
    torch.cuda.set_device(device)
    _require(torch.cuda.is_bf16_supported(), "BF16 is unavailable on selected GPU")

    train_rows = load_projection_rows(
        [Path(config["train_projection_path"])], allowed_splits=("TRAIN",)
    )
    _require(
        len(train_rows)
        == int(config["data_geometry"]["expected_train_projection_candidate_row_count"]),
        "SetFlow V4 TRAIN projection count changed",
    )
    train_records, train_inventory = setflow_source_records_from_projection_rows_v4(
        train_rows
    )
    _require(
        len(train_records) == int(source_data_audit["train_source_count"]),
        "SetFlow V4 TRAIN source inventory differs from preflight",
    )
    vocabs = setflow_source_vocabs_v4(train_records)
    cache_payload = load_source_token_cache_v3(Path(config["source_token_cache_path"]))
    source_token_cache_identity = require_source_token_cache_identity_v3(
        cache_payload,
        expected_model_id="YYLY66/mRNABERT@a1eb7df25804d23f08646e1cb996b234d7208a40",
        expected_record_count=84218,
        expected_unique_source_count=19303,
        expected_token_count=2817781,
        expected_maximum_source_length=837,
        expected_embedding_width=int(
            config["architecture"]["frozen_source_mrnabert_width"]
        ),
    )
    cache = SourceTokenCacheIndexV3(cache_payload)
    model, capacity = build_setflow_screen_model_v4(
        config, vocabs, run_id=run_id
    )
    expected_preflight_count = int(
        preflight[
            "full_trainable_parameter_count"
            if spec.mode_count == 8
            else "single_mode_trainable_parameter_count"
        ]
    )
    _require(
        int(capacity["trainable_parameter_count"]) == expected_preflight_count,
        "SetFlow V4 runner parameter count differs from preflight",
    )
    training = config["training"]
    _require(int(training["pass_count"]) == 10, "SetFlow V4 pass count changed")
    _require(training["saved_checkpoint_passes"] == [4, 6, 8, 10], "SetFlow V4 checkpoint passes changed")
    _require(training["validation_generation_during_training"] is False, "active Validation generation was enabled")
    update_geometry = derive_training_update_geometry_v4(
        len(train_records), passes=int(training["pass_count"])
    )
    dataset = SetFlowSourceStateDatasetV4(
        train_records, vocabs, seed=training_seed
    )
    sampler = BalancedTaskSourceGroupPassSamplerV3(
        train_records,
        record_batch_size=8,
        seed=training_seed,
        repeat_cap=int(config["data_geometry"]["maximum_source_repeats_per_pass"]),
    )
    torch.manual_seed(training_seed)
    torch.cuda.manual_seed_all(training_seed)
    model = model.to(device)
    _require(next(model.parameters()).is_cuda, "SetFlow V4 model parameters left CUDA")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        fused=True,
    )
    initial_parameter = next(model.parameters()).detach().clone()
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir()
    (output_directory / "training_config.json").write_text(
        json.dumps(
            {
                **dict(config),
                "run_id": run_id,
                "run_stage": run_stage,
                "training_seed": training_seed,
                "physical_gpu_index": physical_gpu_index,
                "device": str(device),
                "authorized_git_head": current_head,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    ledger = Path(config["experiment_ledger_path"])
    attempt_path = output_directory / "training_attempt.json"
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    frozen_pretrained_count = int(cache_payload["pretrained_parameter_count"])
    attempt_details = {
        "started_at": started_at,
        "record_counts": {
            "TRAIN": len(train_records),
            "VALIDATION": int(source_data_audit["validation_source_count"]),
        },
        "development_test_record_count_withheld": int(
            config["data_geometry"]["withheld_development_test_record_count"]
        ),
        "evaluation_record_count": 0,
        "included_study_unit_ids": sorted(
            {study for record in train_records for study in record.studies}
        ),
        "included_regions": ["5UTR", "3UTR"],
        "trainable_parameter_count": int(capacity["trainable_parameter_count"]),
        "frozen_pretrained_parameter_count": frozen_pretrained_count,
        "total_effective_parameter_count": int(capacity["trainable_parameter_count"])
        + frozen_pretrained_count,
    }
    attempt_config = {
        "attempt_id": f"xeditsetflow_v4_{run_id}_seed{training_seed}",
        "run_id": f"xeditsetflow_v4_{run_id}_seed{training_seed}",
        "baseline_id": f"xeditsetflow_v4_{run_id}_seed{training_seed}",
        "attempt_purpose": f"XEDITSETFLOW_V4_{run_stage}",
        "scientific_role": "SELECTABLE_FULL" if spec.selectable else "SINGLE_MODE_MECHANISM_CONTROL",
        "result_stage": "DEVELOPMENT_VALIDATION_TRAINING_PENDING_GENERATION",
        "model_kind": "XEDITSETFLOW_V4_FULL" if spec.mode_count == 8 else "XEDITSETFLOW_V4_SINGLE_MODE",
        "pretrained_feature_cache_path": str(config["source_token_cache_path"]),
        "hidden_dim": int(config["architecture"]["model_width"]),
        "depth": int(config["architecture"]["depth"]),
        "batch_size": 32,
        "epochs": 10,
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "loss_kind": "COMMON_SET_MARGINAL_PLUS_PER_CANDIDATE_COVERAGE_PLUS_COUNT_PLUS_MODE_INFORMATION",
        "training_sampling_mode": "TASK_SOURCE_BALANCED_SOURCE_LEVEL",
        "loss_aggregation_mode": "SOURCE_AND_UNIQUE_TERMINAL_CANDIDATE_EQUAL_WEIGHT",
        "seed": training_seed,
        "physical_gpu_index": physical_gpu_index,
        "device": str(device),
        "optimizer_name": "AdamW",
        "optimizer_fused": True,
        "training_precision": "BF16",
        "generation_action_space": "SUB+STOP",
    }
    record_training_attempt(
        ledger,
        attempt_path,
        build_training_attempt_row(
            attempt_config,
            output_directory,
            "RUNNING",
            repository_root=REPO_ROOT,
            details=attempt_details,
        ),
    )
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    update_count = 0
    pass_rows: list[dict[str, Any]] = []
    saved_checkpoint_paths: dict[str, str] = {}
    for pass_index in range(int(training["pass_count"])):
        pass_number = pass_index + 1
        dataset.set_pass(pass_index)
        sampler.set_pass(pass_index)
        source_batches = pad_source_batches_v4(
            sampler.batches_for_pass(),
            source_count=len(train_records),
            sources_per_batch=8,
            repeat_cap=int(config["data_geometry"]["maximum_source_repeats_per_pass"]),
        )
        state_batches = expanded_source_state_batches_v4(
            source_batches, batch_size=32
        )
        _require(
            len(state_batches) == int(update_geometry["updates_per_pass"]),
            "SetFlow V4 updates/pass changed",
        )
        model.train()
        component_sums = {
            "total": 0.0,
            "common": 0.0,
            "coverage": 0.0,
            "count": 0.0,
            "mode_information": 0.0,
        }
        for state_indices in state_batches:
            _require(len(state_indices) == 32, "SetFlow V4 state batch is not 32")
            batch = _move(_collate_indices(dataset, state_indices, cache), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(batch)
                objective = mixture_setflow_loss_v4(
                    output,
                    batch,
                    coverage_weight=float(
                        config["objective"]["source_candidate_coverage_weight"]
                    ),
                    remaining_count_weight=float(
                        config["objective"]["remaining_count_weight"]
                    ),
                    mode_information_weight=spec.mode_information_weight,
                )
            _require(
                objective.total.is_cuda and torch.isfinite(objective.total).item(),
                "SetFlow V4 training loss is invalid",
            )
            objective.total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            _require(torch.isfinite(gradient_norm).item(), "SetFlow V4 gradient norm is nonfinite")
            factor = setflow_v4_learning_rate_factor(
                update_count,
                total_updates=int(update_geometry["total_optimizer_updates"]),
                warmup_fraction=float(training["warmup_fraction"]),
            )
            for group in optimizer.param_groups:
                group["lr"] = float(training["learning_rate"]) * factor
            optimizer.step()
            update_count += 1
            component_sums["total"] += float(objective.total.detach().float().cpu())
            component_sums["common"] += float(
                objective.common_set_marginal.detach().float().cpu()
            )
            component_sums["coverage"] += float(
                objective.source_candidate_coverage.detach().float().cpu()
            )
            component_sums["count"] += float(
                objective.remaining_count.detach().float().cpu()
            )
            component_sums["mode_information"] += float(
                objective.mode_information.detach().float().cpu()
            )
        divisor = len(state_batches)
        pass_rows.append(
            {
                "pass": pass_number,
                "update_count_cumulative": update_count,
                "mean_train_total_loss": component_sums["total"] / divisor,
                "mean_train_common_set_marginal": component_sums["common"] / divisor,
                "mean_train_source_candidate_coverage": component_sums["coverage"] / divisor,
                "mean_train_remaining_count": component_sums["count"] / divisor,
                "mean_train_mode_information": component_sums["mode_information"] / divisor,
                "validation_generation_read": False,
            }
        )
        if pass_number in training["saved_checkpoint_passes"]:
            checkpoint_path = output_directory / f"pass_{pass_number}.pt"
            torch.save(
                {
                    "schema_version": "route_a_v3_route2_xeditsetflow_v4_checkpoint.v1",
                    "run_stage": run_stage,
                    "run_id": run_id,
                    "selectable": spec.selectable,
                    "mode_count": spec.mode_count,
                    "mode_information_weight": spec.mode_information_weight,
                    "seed": training_seed,
                    "completed_pass": pass_number,
                    "model_state_dict": model.state_dict(),
                    "vocabs": vocabs,
                    "capacity": capacity,
                    "source_token_cache_path": str(config["source_token_cache_path"]),
                    "update_geometry": update_geometry,
                    "selection_status": "PENDING_TERMINAL_OUTCOME_FREE_VALIDATION_GENERATION",
                    "critic_used": False,
                    "independent_evaluator_used": False,
                    "development_test_outcome_reads": 0,
                    "new_final_evaluation_outcome_reads": 0,
                },
                checkpoint_path,
            )
            saved_checkpoint_paths[str(pass_number)] = str(checkpoint_path)
        print(
            json.dumps(
                pass_complete_alive_event_v4(
                    run_id=run_id,
                    pass_number=pass_number,
                    update_count=update_count,
                ),
                sort_keys=True,
            ),
            flush=True,
        )
    _require(
        update_count == int(update_geometry["total_optimizer_updates"]),
        "SetFlow V4 total update budget changed",
    )
    _require(
        set(saved_checkpoint_paths) == {"4", "6", "8", "10"},
        "SetFlow V4 checkpoint package is incomplete",
    )
    parameter_changed = not torch.equal(
        initial_parameter, next(model.parameters()).detach()
    )
    _require(parameter_changed, "SetFlow V4 performed no parameter update")
    torch.cuda.synchronize(device)
    elapsed = time.time() - started
    summary = {
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_training_summary.v1",
        "status": "TERMINAL_XEDITSETFLOW_V4_TRAINING_COMPLETE_PENDING_VALIDATION",
        "run_stage": run_stage,
        "run_id": run_id,
        "selectable": spec.selectable,
        "mode_count": spec.mode_count,
        "mode_information_weight": spec.mode_information_weight,
        "seed": training_seed,
        "train_projection_candidate_row_count": len(train_rows),
        "train_source_count": len(train_records),
        "train_inventory": train_inventory,
        "states_per_source_per_pass": 4,
        "physical_and_effective_state_batch": 32,
        "completed_passes": 10,
        "early_stopping_used": False,
        "saved_checkpoint_paths": saved_checkpoint_paths,
        "checkpoint_selection_status": "PENDING_TERMINAL_OUTCOME_FREE_VALIDATION_GENERATION",
        "validation_generation_during_training": False,
        "update_geometry": update_geometry,
        "optimizer_update_count": update_count,
        "trainable_parameter_count": int(capacity["trainable_parameter_count"]),
        "frozen_pretrained_parameter_count": frozen_pretrained_count,
        "source_token_cache_identity": source_token_cache_identity,
        "parameter_changed": parameter_changed,
        "passes": pass_rows,
        "wall_time_seconds": elapsed,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        "physical_gpu_index": physical_gpu_index,
        "torch_device": str(device),
        "training_precision": "BF16",
        "cpu_fallback_used": False,
        "critic_used": False,
        "independent_evaluator_used": False,
        "development_test_record_count_withheld": int(
            config["data_geometry"]["withheld_development_test_record_count"]
        ),
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    (output_directory / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    record_training_attempt(
        ledger,
        attempt_path,
        build_training_attempt_row(
            attempt_config,
            output_directory,
            "COMPLETED",
            repository_root=REPO_ROOT,
            details={
                **attempt_details,
                "optimizer_steps": update_count,
                "wall_time_seconds": elapsed,
                "peak_vram_mb": summary["peak_vram_bytes"] / 1024**2,
                "notes": "terminal SetFlow V4 training only; four checkpoints saved; outcome-free Validation generation and selection pending",
            },
        ),
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", required=True, choices=("v4_full", "v4_single_mode"))
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    config = _load_json(arguments.config)
    run_stage = str(config.get("run_stage", "SCREEN"))
    training_seed = (
        int(config["training"]["screen_seed"])
        if run_stage == "SCREEN"
        else int(config["training_seed"])
    )
    output_directory = arguments.output_dir or Path(config["output_root"]) / arguments.run_id
    try:
        result = train(
            config,
            run_id=arguments.run_id,
            authorization_path=arguments.authorization,
            output_directory=output_directory,
            physical_gpu_index=arguments.physical_gpu_index,
        )
    except Exception as error:
        if output_directory.exists():
            failure = {
                "schema_version": "route_a_v3_route2_xeditsetflow_v4_training_failure.v1",
                "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
                "run_id": arguments.run_id,
                "run_stage": run_stage,
                "seed": training_seed,
                "error_type": type(error).__name__,
                "error": str(error),
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            }
            (output_directory / "failure.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            record_failed_attempt_if_started_v4(
                config, output_directory, error
            )
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
