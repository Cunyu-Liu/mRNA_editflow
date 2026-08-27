#!/usr/bin/env python3
"""Run one target-free full-model Critic V4 RNG-replay update on CUDA."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_bottom_encoder_chunk_cache_v4 import (
    load_frozen_bottom_encoder_chunk_cache_v4,
)
from core.route2_development_projection_v3 import load_projection_rows
from core.route2_xeditcritic_batch_v4 import (
    FrozenBottomEncoderChunkCacheViewV4,
    XEditCriticCollatorV4,
)
from core.route2_xeditcritic_training_v4 import (
    EFFECTIVE_BATCH_V4,
    FixedEffectiveTaskBatchSamplerV4,
    backward_replayed_prediction_gradient_v4,
    collect_replayable_predictions_v4,
    critic_v4_loss_weights,
    critic_v4_optimizer_parameter_groups,
    physical_microbatch_partitions_v4,
    require_physical_gpu_scope_v4,
)
from scripts.route_a_v3.preflight_route2_xeditcritic_v4 import (
    _build_model,
    _set_seed,
    build_preflight_vocabs_v4,
    preflight_example_v4,
)
from scripts.route_a_v3.smoke_route2_xeditcritic_v402_recovery import (
    sampler_records_without_targets_v402,
)
from scripts.route_a_v3.train_route2_xeditcritic_v3 import require_cuda


class XEditCriticV403RNGReplaySmokeError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticV403RNGReplaySmokeError(message)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _move(
    batch: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
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
    return {
        "mean": output["mean"],
        "router_balance_loss": output["router_balance_loss"],
    }


def run(
    config: Mapping[str, Any],
    *,
    expected_head: str,
    physical_gpu_index: int,
    output: Path,
) -> dict[str, Any]:
    _require(_git_head() == expected_head, "V4.0.3 smoke Git HEAD differs")
    _require(not output.exists(), f"V4.0.3 smoke already exists: {output}")
    partial = output.with_suffix(output.suffix + ".partial")
    _require(not partial.exists(), f"V4.0.3 smoke partial exists: {partial}")
    _require(
        not os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES remapping is forbidden",
    )
    require_physical_gpu_scope_v4(config, physical_gpu_index)

    rows = load_projection_rows([Path(path) for path in config["projection_paths"]])
    records = sampler_records_without_targets_v402(rows)
    geometry = config["data_geometry"]
    _require(
        len(records) == int(geometry["expected_train_count"]),
        "TRAIN count changed",
    )
    sampler = FixedEffectiveTaskBatchSamplerV4(
        records,
        seed=int(config["training"]["screen_seed"]),
        repeat_cap=int(geometry["maximum_record_repeats_per_pass"]),
        effective_batch=int(geometry["effective_batch_size"]),
    )
    sampler.set_pass(0)
    effective_indices = sampler.batches_for_pass()[0]
    _require(
        len(effective_indices) == EFFECTIVE_BATCH_V4,
        "first effective TRAIN batch is not 32",
    )

    device = require_cuda(physical_gpu_index)
    preflight = _load(Path(config["preflight_output"]))
    _require(
        preflight.get("status") == "XEDITCRITIC_V4_PREFLIGHT_PASS",
        "formal preflight is not PASS",
    )
    physical_batch_size = int(preflight["selected_physical_batch"])
    _require(
        physical_batch_size in {4, 8, 16, 32},
        "formal physical batch is unsupported",
    )
    required_free_bytes = math.ceil(
        (float(preflight["selected_peak_allocated_gib"]) + 2.0) * 1024**3
    )
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    _require(
        free_bytes >= required_free_bytes,
        "selected GPU free memory is below measured peak plus 2 GiB",
    )

    row_by_id = {str(row["canonical_record_id"]): row for row in rows}
    selected_rows = [row_by_id[records[index].record_id] for index in effective_indices]
    vocabs = build_preflight_vocabs_v4(rows)
    cache_payload = load_frozen_bottom_encoder_chunk_cache_v4(
        Path(config["bottom_six_cache"])
    )
    cache = FrozenBottomEncoderChunkCacheViewV4(
        cache_payload,
        set(str(value) for value in cache_payload["record_ids"]),
    )
    collator = XEditCriticCollatorV4(cache, minimum_physical_batch=4)
    examples = [preflight_example_v4(row, vocabs) for row in selected_rows]
    partitions = physical_microbatch_partitions_v4(
        effective_batch_size=EFFECTIVE_BATCH_V4,
        physical_batch_size=physical_batch_size,
    )
    physical_batches = [
        _move(
            collator([examples[position] for position in partition]),
            device,
        )
        for partition in partitions
    ]

    _set_seed(int(config["training"]["screen_seed"]))
    torch.cuda.empty_cache()
    model = _build_model(config, vocabs, device=device)
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    _require(
        trainable_parameter_count == int(preflight["trainable_parameter_count"]),
        "full-model trainable parameter count differs from formal preflight",
    )
    rates = config["training"]["learning_rates"]
    optimizer = torch.optim.AdamW(
        critic_v4_optimizer_parameter_groups(
            model,
            head_learning_rate=float(rates["new_head_and_v4_trunk"]),
            semantic_learning_rate=float(rates["semantic_experts_and_router"]),
            upper_six_learning_rate=float(rates["mrnabert_top_six"]),
        ),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    optimizer.zero_grad(set_to_none=True)
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    predictions, states, first_pass_predictions = collect_replayable_predictions_v4(
        physical_batches,
        device=device,
        forward=lambda batch: _forward_bf16(model, batch),
    )
    _require(
        all(parameter.grad is None for parameter in model.parameters()),
        "collection unexpectedly populated parameter gradients",
    )
    prediction_gradient = torch.linspace(
        -0.5,
        0.5,
        EFFECTIVE_BATCH_V4,
        device=device,
        dtype=predictions.dtype,
    )
    replayed = backward_replayed_prediction_gradient_v4(
        physical_batches,
        states,
        first_pass_predictions,
        prediction_gradient,
        device=device,
        forward=lambda batch: _forward_bf16(model, batch),
        router_balance_weight=float(critic_v4_loss_weights(1)["router_balance"]),
    )
    _require(
        torch.equal(torch.cat(replayed).float(), predictions),
        "full-model replay predictions changed",
    )
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        float(config["training"]["gradient_clip_norm"]),
    )
    _require(
        torch.isfinite(gradient_norm).item() and float(gradient_norm) > 0,
        "full-model replay gradient norm is invalid",
    )
    optimizer.step()
    torch.cuda.synchronize(device)
    peak_bytes = int(torch.cuda.max_memory_allocated(device))
    _require(bool(optimizer.state), "AdamW state was not materialized")

    result = {
        "schema_version": "route_a_v3_route2_xeditcritic_v403_rng_replay_smoke.v1",
        "status": "XEDITCRITIC_V403_FULL_MODEL_RNG_REPLAY_SMOKE_PASS",
        "git_head": expected_head,
        "physical_gpu_index": physical_gpu_index,
        "train_record_count": len(records),
        "effective_batch_size": EFFECTIVE_BATCH_V4,
        "physical_batch_size": physical_batch_size,
        "physical_batch_count": len(physical_batches),
        "trainable_parameter_count": trainable_parameter_count,
        "forward_precision": "BF16",
        "activation_checkpointing": True,
        "strict_replay_prediction_equal": True,
        "gradient_norm_before_clipping": float(gradient_norm),
        "optimizer_state_materialized": True,
        "peak_allocated_bytes": peak_bytes,
        "peak_allocated_gib": peak_bytes / 1024**3,
        "launch_free_memory_bytes": int(free_bytes),
        "required_free_memory_bytes": int(required_free_bytes),
        "target_value_accessed": False,
        "validation_metric_read": False,
        "cpu_fallback_used": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(
                _load(arguments.config),
                expected_head=arguments.expected_head,
                physical_gpu_index=arguments.physical_gpu_index,
                output=arguments.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
