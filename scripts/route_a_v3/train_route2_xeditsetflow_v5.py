#!/usr/bin/env python3
"""Train one frozen XEditSetFlow V5 screen run without active Validation reads.

V5 successor of the V4/S1 training entry point.  The V5 config carries the
arm definition (run id, mode count, coverage weight, architecture profile) and
the training schedule (pass count, saved checkpoint passes, screen seed).  All
V5 switches are explicit in the config; with V4-S1-equivalent values the
training dynamics are bit-identical to the frozen V4/S1 paths.
"""

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
from core.route2_xeditsetflow_runtime_v5 import (
    build_setflow_screen_model_v5,
    gate_b0_convergence_judgment,
    require_setflow_v5_screen_launch_authorization,
    screen_run_spec_v5,
    setflow_v5_learning_rate_factor,
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
from core.route2_xeditsetflow_temperature_control_v5 import (
    mode_prior_entropy_v5,
)


class SetFlowTrainingV5Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SetFlowTrainingV5Error(message)


def _write_atomic_terminal_v5(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists(), f"SetFlow V5 terminal artifact already exists: {path}")
    partial = path.with_suffix(path.suffix + ".partial")
    _require(
        not partial.exists(),
        f"SetFlow V5 partial terminal artifact already exists: {partial}",
    )
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


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


def record_failed_attempt_if_started_v5(
    config: Mapping[str, Any],
    output_directory: Path,
    error: Exception,
) -> bool:
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
        "notes": "terminal SetFlow V5 implementation or runtime failure; result retained",
    }
    record_training_attempt(
        Path(config["experiment_ledger_path"]), attempt_path, failed
    )
    return True


def derive_training_update_geometry_v5(
    train_source_count: int,
    *,
    sources_per_update: int = 8,
    states_per_source: int = 4,
    physical_and_effective_state_batch: int = 32,
    pass_count: int,
    repeat_cap: int = 4,
) -> dict[str, int]:
    _require(train_source_count >= 8, "SetFlow V5 has fewer than eight TRAIN sources")
    _require(
        sources_per_update == 8
        and states_per_source == 4
        and physical_and_effective_state_batch == 32,
        "SetFlow V5 training geometry changed",
    )
    _require(pass_count >= 1, "SetFlow V5 pass count is invalid")
    updates_per_pass = math.ceil(train_source_count / sources_per_update)
    return {
        "train_source_count": int(train_source_count),
        "sources_per_update": sources_per_update,
        "states_per_source": states_per_source,
        "effective_state_batch": physical_and_effective_state_batch,
        "updates_per_pass": updates_per_pass,
        "pass_count": int(pass_count),
        "total_optimizer_updates": updates_per_pass * int(pass_count),
        "maximum_source_repeats_per_pass": int(repeat_cap),
    }


def pass_complete_alive_event_v5(
    *, run_id: str, pass_number: int, update_count: int
) -> dict[str, Any]:
    return {
        "event": "XEDITSETFLOW_V5_PASS_COMPLETE_ALIVE_ONLY",
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
    spec = screen_run_spec_v5(config, run_id)
    run_stage = str(config.get("run_stage", "SCREEN"))
    _require(run_stage in {"SCREEN", "CONFIRMATION"}, "unknown SetFlow V5 run stage")
    training_seed = (
        int(config["training"]["screen_seed"])
        if run_stage == "SCREEN"
        else int(config["training_seed"])
    )
    current_head = _git_head()
    authorization = _load_json(authorization_path)
    preflight = _load_json(Path(config["preflight_output_path"]))
    source_data_audit = _load_json(Path(config["source_level_data_audit_path"]))
    require_setflow_v5_screen_launch_authorization(
        config,
        authorization,
        preflight,
        source_data_audit,
        run_id=run_id,
        current_git_head=current_head,
    )
    _require(
        not output_directory.exists(),
        f"terminal SetFlow V5 output already exists: {output_directory}",
    )
    _require(
        physical_gpu_index in config["gpu_policy"]["physical_gpu_scope"],
        "SetFlow V5 GPU is outside the config-declared scope",
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
        "SetFlow V5 TRAIN projection count changed",
    )
    train_records, train_inventory = setflow_source_records_from_projection_rows_v4(
        train_rows
    )
    _require(
        len(train_records) == int(source_data_audit["train_source_count"]),
        "SetFlow V5 TRAIN source inventory differs from preflight",
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

    torch.manual_seed(training_seed)
    torch.cuda.manual_seed_all(training_seed)
    model, capacity = build_setflow_screen_model_v5(
        config, vocabs, run_id=run_id
    )
    preflight_capacity = preflight.get("per_run_capacity", {})
    _require(
        spec.run_id in preflight_capacity
        and int(capacity["trainable_parameter_count"])
        == int(preflight_capacity[spec.run_id]["trainable_parameter_count"]),
        "SetFlow V5 runner parameter count differs from preflight",
    )

    training = config["training"]
    pass_count = int(training["pass_count"])
    saved_checkpoint_passes = [int(p) for p in training["saved_checkpoint_passes"]]
    _require(
        saved_checkpoint_passes == sorted(saved_checkpoint_passes)
        and saved_checkpoint_passes
        and max(saved_checkpoint_passes) <= pass_count,
        "SetFlow V5 checkpoint pass schedule is invalid",
    )
    _require(sorted(set(saved_checkpoint_passes)) == saved_checkpoint_passes,
             "SetFlow V5 saved checkpoint passes must be unique")
    _require(training["validation_generation_during_training"] is False,
             "active Validation generation was enabled")
    update_geometry = derive_training_update_geometry_v5(
        len(train_records),
        pass_count=pass_count,
        repeat_cap=int(config["data_geometry"]["maximum_source_repeats_per_pass"]),
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
    model = model.to(device)
    _require(next(model.parameters()).is_cuda, "SetFlow V5 model parameters left CUDA")
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
        "attempt_id": f"xeditsetflow_v5_{run_id}_seed{training_seed}",
        "run_id": f"xeditsetflow_v5_{run_id}_seed{training_seed}",
        "baseline_id": f"xeditsetflow_v5_{run_id}_seed{training_seed}",
        "attempt_purpose": f"XEDITSETFLOW_V5_{run_stage}",
        "scientific_role": "SELECTABLE" if spec.selectable else "MECHANISM_CONTROL",
        "result_stage": "DEVELOPMENT_VALIDATION_TRAINING_PENDING_GENERATION",
        "model_kind": f"XEDITSETFLOW_V5_MODE_{spec.mode_count}",
        "pretrained_feature_cache_path": str(config["source_token_cache_path"]),
        "hidden_dim": int(config["architecture"]["model_width"]),
        "depth": int(config["architecture"]["depth"]),
        "batch_size": 32,
        "epochs": pass_count,
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
    cross_entropy_bonus_weight = float(
        config["objective"].get("unconditional_action_entropy_bonus_weight", 0.0)
    )
    for pass_index in range(pass_count):
        pass_number = pass_index + 1
        dataset.set_pass(pass_index)
        sampler.set_pass(pass_index)
        source_batches = pad_source_batches_v5(
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
            "SetFlow V5 updates/pass changed",
        )
        model.train()
        component_sums = {
            "total": 0.0,
            "common": 0.0,
            "coverage": 0.0,
            "count": 0.0,
            "mode_information": 0.0,
            "action_entropy": 0.0,
        }
        for state_indices in state_batches:
            _require(len(state_indices) == 32, "SetFlow V5 state batch is not 32")
            batch = _move(_collate_indices(dataset, state_indices, cache), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(batch)
                objective = mixture_setflow_loss_v4(
                    output,
                    batch,
                    coverage_weight=spec.coverage_weight,
                    remaining_count_weight=float(
                        config["objective"]["remaining_count_weight"]
                    ),
                    mode_information_weight=spec.mode_information_weight,
                )
                total = objective.total
                action_entropy = output["mode_rates"].new_zeros(())
                if cross_entropy_bonus_weight > 0.0:
                    rates = output["mode_rates"].float()
                    legal = output["legal_action_mask"]
                    prior = output["mode_prior"].float()
                    probabilities = rates / rates.sum(dim=2, keepdim=True).clamp_min(1e-20)
                    unconditional = (probabilities * prior[:, :, None]).sum(dim=1)
                    masked = torch.where(
                        legal, unconditional, unconditional.new_ones(())
                    )
                    entropy = -(
                        torch.where(masked > 0, masked * masked.log(), masked.new_zeros(()))
                        * legal
                    ).sum(dim=1)
                    action_entropy = entropy[legal.any(dim=1)].mean() if legal.any(dim=1).any() else entropy.mean()
                    total = total - cross_entropy_bonus_weight * action_entropy
            _require(
                total.is_cuda and torch.isfinite(total).item(),
                "SetFlow V5 training loss is invalid",
            )
            total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            _require(torch.isfinite(gradient_norm).item(), "SetFlow V5 gradient norm is nonfinite")
            factor = setflow_v5_learning_rate_factor(
                update_count,
                total_updates=int(update_geometry["total_optimizer_updates"]),
                warmup_fraction=float(training["warmup_fraction"]),
            )
            for group in optimizer.param_groups:
                group["lr"] = float(training["learning_rate"]) * factor
            optimizer.step()
            update_count += 1
            component_sums["total"] += float(total.detach().float().cpu())
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
            component_sums["action_entropy"] += float(
                action_entropy.detach().float().cpu()
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
                "mean_train_unconditional_action_entropy": component_sums["action_entropy"] / divisor,
                "validation_generation_read": False,
            }
        )
        if pass_number in saved_checkpoint_passes:
            checkpoint_path = output_directory / f"pass_{pass_number}.pt"
            torch.save(
                {
                    "schema_version": "route_a_v3_route2_xeditsetflow_v5_checkpoint.v1",
                    "run_stage": run_stage,
                    "run_id": run_id,
                    "selectable": spec.selectable,
                    "mode_count": spec.mode_count,
                    "mode_information_weight": spec.mode_information_weight,
                    "coverage_weight": spec.coverage_weight,
                    "architecture_profile": spec.architecture_profile,
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
                pass_complete_alive_event_v5(
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
        "SetFlow V5 total update budget changed",
    )
    _require(
        set(saved_checkpoint_paths) == {str(p) for p in saved_checkpoint_passes},
        "SetFlow V5 checkpoint package is incomplete",
    )
    parameter_changed = not torch.equal(
        initial_parameter, next(model.parameters()).detach()
    )
    _require(parameter_changed, "SetFlow V5 performed no parameter update")
    gate_b0 = gate_b0_convergence_judgment(pass_rows)
    torch.cuda.synchronize(device)
    elapsed = time.time() - started
    summary = {
        "schema_version": "route_a_v3_route2_xeditsetflow_v5_training_summary.v1",
        "status": "TERMINAL_XEDITSETFLOW_V5_TRAINING_COMPLETE_PENDING_VALIDATION",
        "run_stage": run_stage,
        "run_id": run_id,
        "selectable": spec.selectable,
        "mode_count": spec.mode_count,
        "mode_information_weight": spec.mode_information_weight,
        "coverage_weight": spec.coverage_weight,
        "architecture_profile": spec.architecture_profile,
        "seed": training_seed,
        "train_projection_candidate_row_count": len(train_rows),
        "train_source_count": len(train_records),
        "train_inventory": train_inventory,
        "states_per_source_per_pass": 4,
        "physical_and_effective_state_batch": 32,
        "completed_passes": pass_count,
        "early_stopping_used": False,
        "gate_b0_convergence": gate_b0,
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
    _write_atomic_terminal_v5(output_directory / "training_summary.json", summary)
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
                "notes": "terminal SetFlow V5 training only; checkpoints saved; outcome-free Validation generation and selection pending",
            },
        ),
    )
    return summary


def pad_source_batches_v5(
    batches: Sequence[Sequence[int]],
    *,
    source_count: int,
    sources_per_batch: int = 8,
    repeat_cap: int = 4,
) -> list[list[int]]:
    """Deterministically fill only short source batches to 8 sources / 32 states."""
    from core.route2_xeditsetflow_runtime_v4 import pad_source_batches_v4

    return pad_source_batches_v4(
        batches,
        source_count=source_count,
        sources_per_batch=sources_per_batch,
        repeat_cap=repeat_cap,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--run-id",
        required=True,
        help="one run id declared in the V5 config required_screen_runs",
    )
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
                "schema_version": "route_a_v3_route2_xeditsetflow_v5_training_failure.v1",
                "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
                "run_id": arguments.run_id,
                "run_stage": run_stage,
                "seed": training_seed,
                "error_type": type(error).__name__,
                "error": str(error),
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            }
            if not (output_directory / "training_summary.json").exists():
                _write_atomic_terminal_v5(output_directory / "failure.json", failure)
            record_failed_attempt_if_started_v5(
                config, output_directory, error
            )
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
