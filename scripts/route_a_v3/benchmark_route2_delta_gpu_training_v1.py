#!/usr/bin/env python3
"""Benchmark Route 2 critic GPU steps without producing scientific metrics."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_delta_predictor import Route2PretrainedEditCenteredDeltaPredictor
from core.route2_experiment_ledger import (
    build_training_attempt_row,
    record_training_attempt,
)
from scripts.route_a_v3 import train_route2_delta_predictor_v1 as trainer


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def prepare(config: dict[str, Any]):
    manifest = trainer.load_manifest(Path(config["development_manifest"]))
    records = trainer.load_records(
        [Path(path) for path in config["canonical_paths"]], manifest
    )
    pretrained = trainer.FrozenPretrainedPairFeatures(
        Path(config["pretrained_feature_cache_path"]),
        {row.record_id for row in records},
    )
    records, included_studies, _excluded = trainer.select_study_subset(
        records, config.get("included_study_unit_ids")
    )
    records, included_regions, _region_excluded = trainer.select_region_subset(
        records, config.get("included_regions")
    )
    by_split, withheld = trainer.fixed_split_records(records, "HPO_VALIDATION_ONLY")
    train_records = by_split["TRAIN"]
    scaler = trainer.fit_route2_target_scaler(
        train_records,
        mode=str(config.get("target_scaling_mode", trainer.TARGET_SCALING_NONE)),
        minimum_task_records=int(config.get("target_scale_minimum_task_records", 20)),
        floor=float(config.get("target_scale_floor", 1e-3)),
    )
    metadata_mode = str(config.get("metadata_mode", "FULL_CONTEXT"))
    if metadata_mode == "FULL_CONTEXT":
        vocabs = {
            field: trainer.build_vocab(train_records, field)
            for field in ("study", "assay", "context", "endpoint")
        }
    elif metadata_mode == "TRANSFERABLE_CONTEXT":
        vocabs = {
            "study": {"__UNK__": 0},
            **{
                field: trainer.build_vocab(train_records, field)
                for field in ("assay", "context", "endpoint")
            },
        }
    else:
        vocabs = {
            field: {"__UNK__": 0}
            for field in ("study", "assay", "context", "endpoint")
        }
    dataset = trainer.DeltaDataset(
        train_records,
        vocabs,
        metadata_mode=metadata_mode,
        weighting_mode=str(
            config.get(
                "training_weighting_mode", "SOURCE_CONTEXT_ENDPOINT_GROUP"
            )
        ),
        target_scaler=scaler,
        pretrained_features=pretrained,
    )
    model_config = {
        "hidden_dim": int(config["hidden_dim"]),
        "depth": int(config["depth"]),
        "study_count": len(vocabs["study"]),
        "assay_count": len(vocabs["assay"]),
        "context_count": len(vocabs["context"]),
        "endpoint_count": len(vocabs["endpoint"]),
        "pretrained_width": pretrained.width,
        "learned_uncertainty": False,
    }
    return (
        dataset,
        train_records,
        model_config,
        pretrained,
        included_studies,
        included_regions,
        len(by_split["VALIDATION"]),
        withheld,
    )


def make_loader(
    dataset: trainer.DeltaDataset,
    records: list[trainer.DeltaRecord],
    profile: dict[str, Any],
    seed: int,
) -> DataLoader:
    sampler = trainer.LengthBucketBatchSampler(
        records, int(profile["batch_size"]), seed, True
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=trainer.collate,
        **trainer.data_loader_options(profile),
    )


def new_model(
    model_config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> Route2PretrainedEditCenteredDeltaPredictor:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return Route2PretrainedEditCenteredDeltaPredictor(**model_config).to(device)


def benchmark_profile(
    dataset: trainer.DeltaDataset,
    records: list[trainer.DeltaRecord],
    model_config: dict[str, Any],
    profile: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    warmup_steps: int,
    measured_steps: int,
) -> dict[str, Any]:
    loader = make_loader(dataset, records, profile, int(config["seed"]))
    iterator = iter(loader)
    model = new_model(model_config, device, int(config["seed"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        fused=bool(profile.get("fused_adamw", False)),
    )
    precision = str(profile["training_precision"])
    _require(precision in {"FP32", "BF16"}, "unsupported benchmark precision")
    non_blocking = bool(profile.get("non_blocking_transfer", False))
    torch.cuda.reset_peak_memory_stats(device)
    losses = []
    measured_records = 0
    total_steps = warmup_steps + measured_steps
    started = None
    for step in range(total_steps):
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw = next(iterator)
        if step == warmup_steps:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
        batch = trainer._move(raw, device, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=precision == "BF16",
        ):
            output = trainer._forward(model, batch)
            loss = trainer.training_loss(
                output,
                batch,
                str(config["loss_kind"]),
                float(config.get("ranking_loss_weight", 1.0)),
                float(config.get("huber_delta", 1.0)),
            )
        _require(bool(torch.isfinite(loss).item()), "benchmark loss is nonfinite")
        loss.backward()
        optimizer.step()
        if step >= warmup_steps:
            losses.append(float(loss.detach().cpu()))
            measured_records += int(batch["target"].shape[0])
    torch.cuda.synchronize(device)
    _require(started is not None, "benchmark timer was not started")
    elapsed = time.perf_counter() - started
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    result = {
        "status": "PASS",
        "profile_id": str(profile["profile_id"]),
        "batch_size": int(profile["batch_size"]),
        "training_precision": precision,
        "fused_adamw": bool(profile.get("fused_adamw", False)),
        "num_workers": int(profile.get("num_workers", 0)),
        "pin_memory": bool(profile.get("pin_memory", False)),
        "non_blocking_transfer": non_blocking,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "measured_records": measured_records,
        "elapsed_seconds": elapsed,
        "steps_per_second": measured_steps / elapsed,
        "records_per_second": measured_records / elapsed,
        "mean_loss": sum(losses) / len(losses),
        "all_losses_finite": all(math.isfinite(value) for value in losses),
        "peak_vram_mb": peak_vram_mb,
        "temporary_parameter_updates": total_steps,
        "checkpoint_saved": False,
        "scientific_metrics_computed": False,
        "evaluation_pool_records_read": 0,
    }
    del optimizer, model, iterator, loader
    torch.cuda.empty_cache()
    return result


def run_profile_or_record_oom(*args, **kwargs) -> dict[str, Any]:
    profile = dict(args[3])
    device = args[5]
    try:
        return benchmark_profile(*args, **kwargs)
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        properties = torch.cuda.get_device_properties(device)
        return {
            "status": "OUT_OF_MEMORY",
            "profile_id": str(profile["profile_id"]),
            "batch_size": int(profile["batch_size"]),
            "training_precision": str(profile["training_precision"]),
            "fused_adamw": bool(profile.get("fused_adamw", False)),
            "num_workers": int(profile.get("num_workers", 0)),
            "pin_memory": bool(profile.get("pin_memory", False)),
            "non_blocking_transfer": bool(
                profile.get("non_blocking_transfer", False)
            ),
            "device_name": properties.name,
            "device_total_memory_gib": properties.total_memory / (1024 ** 3),
            "error_type": type(exc).__name__,
            "error": "profile exceeded the visible CUDA device memory",
            "temporary_parameter_updates": 0,
            "checkpoint_saved": False,
            "scientific_metrics_computed": False,
            "evaluation_pool_records_read": 0,
        }


def precision_probe(
    dataset: trainer.DeltaDataset,
    records: list[trainer.DeltaRecord],
    model_config: dict[str, Any],
    config: dict[str, Any],
    benchmark: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    profile = {
        "batch_size": 16,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
    }
    raw = next(iter(make_loader(dataset, records, profile, int(config["seed"]))))
    batch = trainer._move(raw, device)
    model = new_model(model_config, device, int(config["seed"]))
    model.eval()
    with torch.no_grad():
        fp32 = trainer._forward(model, batch)["mean"].float()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            bf16 = trainer._forward(model, batch)["mean"].float()
    difference = (fp32 - bf16).abs()
    if float(fp32.std()) > 0 and float(bf16.std()) > 0:
        pearson = float(torch.corrcoef(torch.stack([fp32, bf16]))[0, 1])
    else:
        pearson = None
    max_difference = float(difference.max())
    mean_difference = float(difference.mean())
    passed = (
        max_difference <= float(benchmark["bf16_fp32_max_absolute_difference"])
        and mean_difference <= float(benchmark["bf16_fp32_mean_absolute_difference"])
        and pearson is not None
        and pearson >= float(benchmark["bf16_fp32_minimum_pearson"])
    )
    result = {
        "record_count": int(fp32.numel()),
        "fp32_all_finite": bool(torch.isfinite(fp32).all().item()),
        "bf16_all_finite": bool(torch.isfinite(bf16).all().item()),
        "maximum_absolute_difference": max_difference,
        "mean_absolute_difference": mean_difference,
        "pearson": pearson,
        "maximum_absolute_difference_threshold": float(
            benchmark["bf16_fp32_max_absolute_difference"]
        ),
        "mean_absolute_difference_threshold": float(
            benchmark["bf16_fp32_mean_absolute_difference"]
        ),
        "minimum_pearson_threshold": float(
            benchmark["bf16_fp32_minimum_pearson"]
        ),
        "precision_tolerance_pass": passed,
    }
    del model
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    benchmark = json.loads(args.config.read_text(encoding="utf-8"))
    training_config = json.loads(
        Path(benchmark["training_config"]).read_text(encoding="utf-8")
    )
    device = trainer.require_cuda(
        str(benchmark["device"]), int(benchmark["physical_gpu_index"])
    )
    output_dir = Path(benchmark["output_directory"])
    _require(not output_dir.exists(), f"output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    ledger_config = {
        **training_config,
        **benchmark,
        "baseline_id": str(benchmark["baseline_id"]),
        "attempt_purpose": "GPU_TRAINING_THROUGHPUT_AND_PRECISION_BENCHMARK_NO_SCIENTIFIC_RESULT",
        "result_stage": "PERFORMANCE_BENCHMARK_NO_SCIENTIFIC_RESULT",
        "scientific_role": "ENGINEERING_THROUGHPUT_ONLY",
        "output_directory": str(output_dir),
    }
    record_training_attempt(
        Path(benchmark["experiment_ledger_path"]),
        output_dir / "training_attempt.json",
        build_training_attempt_row(
            ledger_config,
            output_dir,
            "RUNNING",
            repository_root=REPO_ROOT,
            details={"evaluation_record_count": 0},
        ),
    )
    (
        dataset,
        train_records,
        model_config,
        pretrained,
        included_studies,
        included_regions,
        validation_count,
        withheld_count,
    ) = prepare(training_config)
    profiles = [
        run_profile_or_record_oom(
            dataset,
            train_records,
            model_config,
            dict(profile),
            training_config,
            device,
            int(benchmark["warmup_steps"]),
            int(benchmark["measured_steps"]),
        )
        for profile in benchmark["profiles"]
    ]
    precision_comparison = precision_probe(
        dataset,
        train_records,
        model_config,
        training_config,
        benchmark,
        device,
    )
    payload = {
        "schema_version": "route_a_v3_route2_gpu_training_benchmark.v1",
        "status": (
            "GPU_TRAINING_BENCHMARK_COMPLETE_WITH_PROFILE_OOM_NO_SCIENTIFIC_RESULT"
            if any(row["status"] != "PASS" for row in profiles)
            else "GPU_TRAINING_BENCHMARK_COMPLETE_NO_SCIENTIFIC_RESULT"
        ),
        "device_name": torch.cuda.get_device_properties(device).name,
        "device_total_memory_gib": torch.cuda.get_device_properties(device).total_memory
        / (1024 ** 3),
        "included_study_unit_ids": included_studies,
        "included_regions": included_regions,
        "train_record_count": len(train_records),
        "validation_record_count": validation_count,
        "withheld_test_record_count": withheld_count,
        "pretrained_model_id": pretrained.model_id,
        "trainable_parameter_count": sum(
            value.numel()
            for value in new_model(
                model_config, device, int(training_config["seed"])
            ).parameters()
        ),
        "frozen_pretrained_parameter_count": pretrained.pretrained_parameter_count,
        "profiles": profiles,
        "fp32_bf16_precision_comparison": precision_comparison,
        "scientific_metrics_computed": False,
        "evaluation_pool_records_read": 0,
        "checkpoint_saved": False,
    }
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    successful_profiles = [row for row in profiles if row["status"] == "PASS"]
    _require(bool(successful_profiles), "all GPU benchmark profiles failed")
    best = max(successful_profiles, key=lambda row: row["records_per_second"])
    record_training_attempt(
        Path(benchmark["experiment_ledger_path"]),
        output_dir / "training_attempt.json",
        build_training_attempt_row(
            ledger_config,
            output_dir,
            "COMPLETED",
            repository_root=REPO_ROOT,
            details={
                "included_study_unit_ids": included_studies,
                "included_regions": included_regions,
                "record_counts": {
                    "TRAIN": len(train_records),
                    "VALIDATION": validation_count,
                },
                "development_test_record_count_withheld": withheld_count,
                "evaluation_record_count": 0,
                "trainable_parameter_count": payload["trainable_parameter_count"],
                "frozen_pretrained_parameter_count": pretrained.pretrained_parameter_count,
                "total_effective_parameter_count": payload["trainable_parameter_count"]
                + pretrained.pretrained_parameter_count,
                "optimizer_steps": sum(
                    row["temporary_parameter_updates"] for row in profiles
                ),
                "peak_vram_mb": max(
                    row["peak_vram_mb"] for row in successful_profiles
                ),
                "wall_time_seconds": sum(
                    row["elapsed_seconds"] for row in successful_profiles
                ),
                "notes": f"best_profile={best['profile_id']}; no checkpoint retained",
            },
        ),
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
