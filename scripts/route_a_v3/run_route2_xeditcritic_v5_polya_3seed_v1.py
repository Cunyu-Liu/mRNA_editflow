#!/usr/bin/env python3
"""P0-1: V5 polyA 3-seed supplement (seeds 20260921/20260922), v1 2026-09-20.

Bespoke runner cloning the V5 screen run (v5_full, seed 20260907) with a
SINGLE changed surface: the parameter-initialization seed. Model / data /
loss / optimizer / schedule / sampling / FINAL-PASS-8-FIXED protocol are the
official train_route2_xeditcritic_v4.py V4-FULL channel, imported as a library
(shared functions reused verbatim - no model/loss/optimizer code is modified).

Why bespoke (declared, M1-intervention precedent, journal batch 112): the
official runner's SCREEN channel hard-gates seed==20260907 + a launch
authorization bound to historical git HEAD 1113cd2c + a clean worktree. A new
seed cannot pass those historical gates; this runner bypasses EXACTLY the
SCREEN-channel four gates (seed whitelist, old-HEAD authorization, preflight
binding, clean worktree) and keeps every other discipline: CUDA A100 hard
gate, BF16 hard gate, experiment-ledger reporting, TEST/Eval reads = 0,
append-only outputs, failure terminal artifact.

Preregistration: docs/paper/polya_3seed_mini_prereg_v1.md (frozen before
launch). Verdict handling deferred to harvest: 3-seed mean + range reported;
leaderboard main row 0.8219 NOT replaced (new row only).

Heartbeat: heartbeat.json refreshed per pass; run_summary.json at terminal.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / "scripts/route_a_v3"))

from train_route2_xeditcritic_v4 import (  # noqa: E402
    _forward_bf16,
    _move,
    _physical_batches,
    _set_seed,
    _write_atomic_terminal_v4,
    evaluation_index_batches_v4,
)
from train_route2_xeditcritic_v4 import _evaluate as v4_evaluate  # noqa: E402
from core.route2_bottom_encoder_chunk_cache_v4 import (  # noqa: E402
    load_frozen_bottom_encoder_chunk_cache_v4,
)
from core.route2_experiment_ledger import (  # noqa: E402
    build_training_attempt_row,
    record_training_attempt,
)
from core.route2_xeditcritic_batch_v4 import (  # noqa: E402
    FrozenBottomEncoderChunkCacheViewV4,
    XEditCriticCollatorV4,
    XEditCriticDatasetV4,
)
from core.route2_xeditcritic_ledger_v4 import (  # noqa: E402
    critic_v4_attempt_details,
    critic_v4_ledger_paths,
)
from core.route2_xeditcritic_training_data_v3 import (  # noqa: E402
    build_vocabs,
    records_from_projection_rows,
)
from core.route2_xeditcritic_training_v4 import (  # noqa: E402
    FixedEffectiveTaskBatchSamplerV4,
    backward_replayed_prediction_gradient_v4,
    backward_retained_effective_batch_v4,
    collect_replayable_predictions_v4,
    critic_v4_learning_rate_factor,
    critic_v4_loss_weights,
    critic_v4_optimizer_parameter_groups,
    effective_prediction_objective_v4,
    forward_retained_effective_batch_v4,
)
from core.route2_development_projection_v3 import load_projection_rows  # noqa: E402
from scripts.route_a_v3.route2_mrnabert_upper_six_encoder_v4 import (  # noqa: E402
    TrainableMRNABERTUpperSixEncoderV4,
)
from core.route2_xeditcritic_v4 import (  # noqa: E402
    XEditCriticV4,
    require_v4_trainable_parameter_range,
)
from train_route2_xeditcritic_v3 import fit_task_robust_scaler  # noqa: E402

# ---- frozen V5 screen config (verbatim values from the archived
# authorizations/xeditcritic_v5/.../screen_config.json; only seed differs) ----
MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
FROZEN_CONFIG_PATH = (
    MNT
    / "authorizations/xeditcritic_v5/v5_screen_seed_20260907_runner_1113cd2c0dd9acb508f58782eecb40f458d2cab3/screen_config.json"
)
OUT_ROOT = MNT / "experiments/xeditcritic_v5_polya_3seed"
LEDGER_PATH = MNT / "experiment_tracking/route2_training_attempts.csv"
PREREG = "docs/paper/polya_3seed_mini_prereg_v1.md"

NEW_SEEDS = (20260921, 20260922)
GEOMETRY_KEYS = (
    "expected_record_count",
    "expected_train_count",
    "expected_validation_count",
    "pass_count",
    "updates_per_pass",
    "total_optimizer_updates",
    "effective_batch_size",
    "maximum_record_repeats_per_pass",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _git_head() -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _terminal_training_environment(training_seed, device, training_git_head):
    _require(device.type == "cuda", "P0-1 parameter updates require CUDA")
    _require(torch.cuda.is_available(), "CUDA unavailable; CPU fallback forbidden")
    device_name = torch.cuda.get_device_name(device)
    _require("A100" in device_name, "P0-1 requires an A100")
    _require(torch.cuda.is_bf16_supported(), "P0-1 requires CUDA BF16 support")
    return {
        "parameter_initialization_seed": int(training_seed),
        "parameter_initialization_seed_applied_before_model_construction": True,
        "parameter_initialization_tensor_identity_scope": "SHARED_V4_CONSTRUCTOR_WITHIN_IDENTICAL_ARCHITECTURE",
        "cuda_available": True,
        "cuda_device": str(device),
        "cuda_device_name": device_name,
        "a100_device_verified": True,
        "bf16_supported": True,
        "cpu_fallback_used": False,
        "training_git_head": str(training_git_head),
    }


def _build_model_v4_full(config, vocabs, device):
    architecture = config["architecture"]
    upper = TrainableMRNABERTUpperSixEncoderV4(
        Path(config["mrnabert_model_path"]),
        device,
        attention_backend=str(config["memory_preflight"]["attention_backend"]),
        activation_checkpointing=bool(config["memory_preflight"]["activation_checkpointing"]),
    )
    model = XEditCriticV4(
        upper_encoder=upper,
        study_count=len(vocabs["study"]),
        assay_count=len(vocabs["assay"]),
        context_count=len(vocabs["context"]),
        quantity_count=len(vocabs["quantity"]),
        measurement_count=len(vocabs["measurement"]),
        numerator_count=len(vocabs["numerator"]),
        denominator_count=len(vocabs["denominator"]),
        region_count=2,
        control_mode="NONE",
        mechanism_mode="FULL",
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


def run_one_seed(config, seed, physical_gpu_index, training_git_head) -> dict:
    geometry = config["data_geometry"]
    for key in GEOMETRY_KEYS:
        _require(key in geometry, f"frozen V5 geometry key absent: {key}")

    _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden")
    _set_seed(seed)
    device = torch.device(f"cuda:{physical_gpu_index}")
    environment = _terminal_training_environment(seed, device, training_git_head)

    output_directory = OUT_ROOT / f"seed_{seed}"
    _require(
        not output_directory.exists(),
        f"P0-1 run directory already exists: {output_directory}",
    )
    output_directory.mkdir(parents=True)
    heartbeat_path = output_directory / "heartbeat.json"

    def write_heartbeat(status, **fields):
        payload = {
            "status": status,
            "seed": int(seed),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "schema_version": "route_a_v3_v5_polya_3seed_heartbeat.v1",
        }
        payload.update(fields)
        heartbeat_path.write_text(json.dumps(payload, indent=1, sort_keys=True))

    write_heartbeat("INIT", pass_number=0, note="data loading")

    started = time.time()
    checkpoint_path = output_directory / "final_pass_8_checkpoint.pt"
    training_summary_path = output_directory / "run_summary.json"
    attempt_path = output_directory / "training_attempt.json"
    physical_batch_size = 32

    attempt_config = {
        "attempt_id": f"xeditcritic_v5_polya_3seed_seed{seed}::v5_full_replica",
        "attempt_purpose": "P0_1_POLYA_3SEED_SUPPLEMENT",
        "baseline_id": f"xeditcritic_v5_polya_3seed_seed{seed}",
        "scientific_role": "XEDITCRITIC_V5_POLYA_3SEED_ROBUSTNESS_SUPPLEMENT",
        "result_stage": "DEVELOPMENT_VALIDATION",
        "run_mode": "FROZEN_V5_REPLICA_SEED_ONLY",
        "model_kind": "V4-FULL",
        "pretrained_model_id": str(config["model_id"]),
        "pretrained_feature_cache_path": str(config["bottom_six_cache"]),
        "canonical_paths": "",
        "included_regions": "5UTR;3UTR",
        "hidden_dim": 768,
        "depth": 12,
        "batch_size": 32,
        "epochs": 8,
        "learning_rate": 0.0002,
        "weight_decay": 0.0001,
        "loss_kind": "STANDARDIZED_HUBER_PLUS_CROSS_GROUP_PAIRWISE_THEN_SOFT_SPEARMAN",
        "huber_delta": float(config["training"]["huber_delta"]),
        "metadata_mode": "OUTCOME_FREE_ENDPOINT_DESCRIPTORS",
        "training_weighting_mode": "STUDY_THEN_SOURCE_GROUP",
        "training_sampling_mode": "SQRT_TASK_SIZE_FIXED_EFFECTIVE32_REPEAT_CAP4",
        "loss_aggregation_mode": "TASK_ROBUST_STANDARDIZED_EFFECTIVE_TASK_BATCH",
        "target_scaling_mode": "TRAIN_TASK_ROBUST_WITH_REGION_GLOBAL_FALLBACK",
        "candidate_control": "NONE",
        "seed": int(seed),
        **environment,
        "physical_gpu_index": int(physical_gpu_index),
        "optimizer_name": "AdamW",
        "optimizer_fused": False,
        "training_precision": "BF16",
        "encoder_attention_backend": "PYTORCH_SDPA_AUTO",
        "num_workers": 0,
        "pin_memory": False,
        "torch_compile": False,
        "output_directory": str(output_directory),
        "training_summary_path": str(training_summary_path),
        "checkpoint_path": str(checkpoint_path),
        "training_attempt_path": str(attempt_path),
        "notes": (
            "P0-1 polyA 3-seed supplement; V5 screen replica, single changed "
            "surface = seed; FINAL-PASS-8-FIXED; no TEST or Evaluation access"
        ),
    }
    attempt_details = critic_v4_attempt_details(
        config,
        trainable_parameter_count=None,
        physical_batch_size=physical_batch_size,
        peak_vram_mb=None,
    )

    record_training_attempt(
        LEDGER_PATH,
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
        _require(
            len(records) == int(geometry["expected_record_count"]),
            "projection record count changed",
        )
        train_records = [record for record in records if record.split == "TRAIN"]
        validation_records = [
            record for record in records if record.split == "VALIDATION"
        ]
        _require(
            len(train_records) == int(geometry["expected_train_count"]),
            "TRAIN count changed",
        )
        _require(
            len(validation_records) == int(geometry["expected_validation_count"]),
            "VALIDATION count changed",
        )
        record_by_id = {record.record_id: record for record in records}
        vocabs = build_vocabs(records)
        scaler = fit_task_robust_scaler(
            train_records,
            floor=float(config["training"]["target_scale_floor"]),
        )
        cache_payload = load_frozen_bottom_encoder_chunk_cache_v4(
            Path(config["bottom_six_cache"])
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
            candidate_bundle_overrides={},
            neutral_studies=set(),
        )
        validation_dataset = XEditCriticDatasetV4(
            validation_records,
            all_records=record_by_id,
            vocabs=vocabs,
            target_scaler=scaler,
            cache=None,
            candidate_bundle_overrides={},
            neutral_studies=set(),
        )
        collator = XEditCriticCollatorV4(
            cache,
            minimum_physical_batch=int(config["memory_preflight"]["minimum_physical_batch"]),
        )
        model, capacity = _build_model_v4_full(config, vocabs, device)
        _require(
            int(capacity["trainable_parameter_count"]) == 170481957,
            "P0-1 replica trainable parameter count differs from V5 main row",
        )
        attempt_details = critic_v4_attempt_details(
            config,
            trainable_parameter_count=int(capacity["trainable_parameter_count"]),
            physical_batch_size=physical_batch_size,
            peak_vram_mb=torch.cuda.max_memory_allocated(device) / 1024**2,
        )
        record_training_attempt(
            LEDGER_PATH,
            attempt_path,
            build_training_attempt_row(
                attempt_config,
                output_directory,
                "RUNNING",
                repository_root=REPO_ROOT,
                details=attempt_details,
            ),
        )
        rates = config["training"]["learning_rates"]
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
        initial_learning_rates = [float(group["lr"]) for group in optimizer.param_groups]

        sampler = FixedEffectiveTaskBatchSamplerV4(
            train_records,
            seed=seed,
            repeat_cap=int(geometry["maximum_record_repeats_per_pass"]),
            effective_batch=int(geometry["effective_batch_size"]),
        )
        initial_parameter = next(
            parameter for parameter in model.parameters() if parameter.requires_grad
        ).detach().clone()
        update_count = 0
        pass_rows = []
        total_updates = int(geometry["total_optimizer_updates"])
        warmup_fraction = float(config["training"]["warmup_fraction"])
        gradient_clip_norm = float(config["training"]["gradient_clip_norm"])
        soft_rank_temperature = float(config["training"]["soft_rank_temperature"])
        within_source_ranking_weight = float(
            config["training"]["within_source_ranking_weight"]
        )
        huber_delta = float(config["training"]["huber_delta"])
        lambda_pairwise_weight = float(
            config["training"].get("lambda_pairwise_weight", 0.0)
        )
        torch.cuda.reset_peak_memory_stats(device)
        t_start = time.time()

        for pass_index in range(int(geometry["pass_count"])):
            pass_number = pass_index + 1
            sampler.set_pass(pass_index)
            effective_batches = sampler.batches_for_pass()
            _require(
                len(effective_batches) == int(geometry["updates_per_pass"]),
                "P0-1 updates/pass changed",
            )
            task_losses, huber_losses, pairwise_losses, soft_losses, pair_counts = (
                [],
                [],
                [],
                [],
                [],
            )
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
                    predictions, states, _first = collect_replayable_predictions_v4(
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
                    value for batch in physical_batches for value in batch["task_ids"]
                ]
                objective = effective_prediction_objective_v4(
                    predictions,
                    targets,
                    sample_weights,
                    source_groups,
                    task_ids,
                    pass_number=pass_number,
                    huber_delta=huber_delta,
                    soft_rank_temperature=soft_rank_temperature,
                    within_source_ranking_weight=within_source_ranking_weight,
                    lambda_pairwise_weight=lambda_pairwise_weight,
                    cell_offset_predictions=None,
                    cell_offset_targets=None,
                    cell_offset_weight=0.0,
                )
                gradient_for_backward = objective.prediction_gradient
                optimizer.zero_grad(set_to_none=True)
                router_balance_weight = float(
                    critic_v4_loss_weights(pass_number)["router_balance"]
                )
                if retained_graph is not None:
                    backward_retained_effective_batch_v4(
                        retained_graph,
                        gradient_for_backward,
                        router_balance_weight=router_balance_weight,
                        cell_offset_gradient=None,
                    )
                else:
                    backward_replayed_prediction_gradient_v4(
                        physical_batches,
                        states,
                        _first,
                        gradient_for_backward,
                        device=device,
                        forward=lambda batch: _forward_bf16(model, batch),
                        router_balance_weight=router_balance_weight,
                    )
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    gradient_clip_norm,
                )
                _require(
                    torch.isfinite(gradient_norm).item(),
                    "P0-1 gradient norm is nonfinite",
                )
                factor = critic_v4_learning_rate_factor(
                    update_count,
                    total_updates=total_updates,
                    warmup_fraction=warmup_fraction,
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
                    "mean_task_objective_excluding_router_balance": float(
                        np.mean(task_losses)
                    ),
                    "mean_huber_loss": float(np.mean(huber_losses)),
                    "mean_pairwise_loss": float(np.mean(pairwise_losses)),
                    "mean_soft_spearman_loss": float(np.mean(soft_losses)),
                    "mean_pair_count": float(np.mean(pair_counts)),
                    "validation_metric_read": False,
                }
            )
            write_heartbeat(
                "PASS_DONE",
                pass_number=pass_number,
                update_count=update_count,
                total_updates=total_updates,
                elapsed_s=round(time.time() - t_start, 1),
            )
            print(
                json.dumps(
                    {
                        "event": "P0_1_PASS_COMPLETE",
                        "seed": seed,
                        "pass": pass_number,
                        "update_count": update_count,
                        "mean_huber_loss": pass_rows[-1]["mean_huber_loss"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        _require(
            update_count == total_updates,
            "P0-1 total update budget changed",
        )
        parameter_changed = not torch.equal(
            initial_parameter, next(model.parameters()).detach()
        )
        _require(parameter_changed, "P0-1 performed no learned parameter update")

        prediction_path = output_directory / "final_validation_predictions.jsonl"
        final_metrics = v4_evaluate(
            model,
            validation_dataset,
            collator,
            physical_batch_size=physical_batch_size,
            device=device,
            prediction_path=prediction_path,
        )

        torch.save(
            {
                "schema_version": "route_a_v3_route2_xeditcritic_v4_screen_checkpoint.v2",
                "run_stage": "P0_1_POLYA_3SEED_SUPPLEMENT",
                "run_id": f"v5_full_replica_seed{seed}",
                "model_kind": "V4-FULL",
                "control_mode": "NONE",
                "mechanism_mode": "FULL",
                "candidate_bundle_permutation": False,
                "seed": seed,
                "physical_gpu_index": physical_gpu_index,
                "precision": "BF16_FORWARD_FP32_EFFECTIVE_OBJECTIVE",
                "selected_pass": 8,
                "selection_policy": "FINAL_PASS_8_FIXED_NO_VALIDATION_PEAK_RESELECTION",
                "model_state_dict": model.state_dict(),
                "vocabs": vocabs,
                "target_scaler": scaler.to_dict(),
                "capacity": capacity,
                "physical_batch_size": physical_batch_size,
                "effective_batch_size": 32,
                "retained_graph_fast_path": True,
                "full_cache_validation_per_batch": False,
                "validation_metrics": final_metrics,
                **environment,
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
            checkpoint_path,
        )
        summary = {
            "schema_version": "route_a_v3_v5_polya_3seed_run.v1",
            "status": "TERMINAL_P0_1_POLYA_3SEED_RUN_COMPLETE",
            "prereg": PREREG,
            "run_stage": "P0_1_POLYA_3SEED_SUPPLEMENT",
            "seed": seed,
            "run_id": f"v5_full_replica_seed{seed}",
            "model_kind": "V4-FULL",
            "control_mode": "NONE",
            "mechanism_mode": "FULL",
            "selectable": False,
            "output_directory": str(output_directory),
            "checkpoint_path": str(checkpoint_path),
            "training_attempt_path": str(attempt_path),
            **environment,
            "precision": "BF16_FORWARD_FP32_EFFECTIVE_OBJECTIVE",
            "capacity": capacity,
            "train_record_count": len(train_records),
            "validation_record_count": len(validation_records),
            "pass_count": 8,
            "selected_pass": 8,
            "update_count": update_count,
            "physical_batch_size": physical_batch_size,
            "effective_batch_size": 32,
            "sampler": {
                "policy": "SQRT_TASK_SIZE_TASK_HOMOGENEOUS_SOURCE_GROUP_BALANCED",
                "repeat_cap": 4,
                "updates_per_pass": int(geometry["updates_per_pass"]),
            },
            "target_scaler": scaler.to_dict(),
            "passes": pass_rows,
            "final_validation": final_metrics,
            "validation_prediction_path": str(prediction_path),
            "elapsed_seconds": time.time() - started,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
            "reference": {
                "v5_main_row_seed_20260907_polya_spearman": 0.8218686245779881,
                "v5_main_row_elapsed_seconds": 55636.713121175766,
                "prereg": PREREG,
            },
        }
        _write_atomic_terminal_v4(training_summary_path, summary)
        record_training_attempt(
            LEDGER_PATH,
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
                    **environment,
                    "notes": (
                        "P0-1 polyA 3-seed supplement terminal; FINAL-PASS-8-FIXED; "
                        "no TEST or Evaluation access; prereg "
                        "docs/paper/polya_3seed_mini_prereg_v1.md"
                    ),
                },
            ),
        )
        write_heartbeat(
            "DONE",
            pass_number=8,
            update_count=update_count,
            final_task_macro_spearman=final_metrics.get("task_macro_spearman"),
            wallclock_h=round(summary["elapsed_seconds"] / 3600, 2),
        )
        return summary
    except Exception as exc:
        failure = {
            "schema_version": "route_a_v3_v5_polya_3seed_run_failure.v1",
            "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
            "seed": seed,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": time.time() - started,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        }
        if not training_summary_path.exists():
            training_summary_path.parent.mkdir(parents=True, exist_ok=True)
            (output_directory / "failure.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n"
            )
        record_training_attempt(
            LEDGER_PATH,
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
        write_heartbeat("FAILED", error=f"{type(exc).__name__}: {exc}"[:300])
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=list(NEW_SEEDS))
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    args = parser.parse_args()

    config = json.loads(FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))
    _require(
        int(args.physical_gpu_index) in config["gpu_policy"]["physical_gpu_scope"],
        "P0-1 physical GPU is outside the frozen 0-5 scope",
    )
    summary = run_one_seed(
        config,
        seed=args.seed,
        physical_gpu_index=args.physical_gpu_index,
        training_git_head=_git_head(),
    )
    print(json.dumps({k: summary[k] for k in ("status", "seed", "update_count")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
