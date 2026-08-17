#!/usr/bin/env python3
"""Diagnose full-loop mRNABERT critic throughput after the Huber run completes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


class ThroughputAnalysisError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ThroughputAnalysisError(message)


def load(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def finite_positive(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    require(math.isfinite(result) and result > 0, f"{label} is not positive finite")
    return result


def indexed_profiles(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("profiles")
    require(isinstance(rows, list) and rows, "benchmark profiles are missing")
    result = {str(row["profile_id"]): row for row in rows}
    require(len(result) == len(rows), "benchmark profile IDs are duplicated")
    for row in rows:
        require(row.get("status") == "PASS", f"profile did not pass: {row.get('profile_id')}")
        finite_positive(row.get("records_per_second"), "records per second")
    return result


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator


def analyze(
    huber_summary: dict[str, Any],
    gpu_benchmark: dict[str, Any],
    dataloader_benchmark: dict[str, Any],
) -> dict[str, Any]:
    require(
        huber_summary.get("status") == "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
        "Huber training is incomplete",
    )
    require(huber_summary.get("loss_kind") == "huber", "input is not the Huber run")
    require(
        huber_summary.get("result_stage") == "HPO_VALIDATION_ONLY",
        "Huber run is not Development VALIDATION-only",
    )
    require(huber_summary.get("evaluation_outcomes_read") == 0, "Evaluation entered Huber")
    require(huber_summary.get("test_metrics") is None, "Development TEST entered Huber")
    require(
        gpu_benchmark.get("fp32_bf16_precision_comparison", {}).get(
            "precision_tolerance_pass"
        )
        is True,
        "BF16 precision comparison did not pass",
    )
    require(
        gpu_benchmark.get("evaluation_pool_records_read") == 0
        and dataloader_benchmark.get("evaluation_pool_records_read") == 0,
        "Evaluation entered an engineering benchmark",
    )

    epochs = int(huber_summary["final_training_epoch"])
    require(epochs > 0, "Huber epoch count is invalid")
    wall_time = finite_positive(huber_summary["wall_time_seconds"], "Huber wall time")
    optimizer_steps = int(huber_summary["optimizer_steps"])
    require(optimizer_steps > 0, "Huber optimizer steps are invalid")
    train_records = int(huber_summary["record_counts"]["TRAIN"])
    validation_records = int(huber_summary["record_counts"]["VALIDATION"])
    require(train_records > 0 and validation_records > 0, "Huber record counts are invalid")

    gpu_profiles = indexed_profiles(gpu_benchmark)
    loader_profiles = indexed_profiles(dataloader_benchmark)
    required_gpu_profiles = {
        "B16_FP32_ADAMW",
        "B16_BF16_FUSED_ADAMW",
        "B32_BF16_FUSED_ADAMW",
        "B64_BF16_FUSED_ADAMW",
    }
    require(required_gpu_profiles <= set(gpu_profiles), "GPU benchmark profile set is incomplete")
    required_loader_profiles = {
        "B32_BF16_FUSED_WORKERS0",
        "B32_BF16_FUSED_WORKERS4",
        "B32_BF16_FUSED_WORKERS8",
    }
    require(
        set(loader_profiles) == required_loader_profiles,
        "dataloader benchmark is not the matched workers 0/4/8 comparison",
    )

    throughput = {
        profile_id: finite_positive(row["records_per_second"], profile_id)
        for profile_id, row in {**gpu_profiles, **loader_profiles}.items()
    }
    bf16_speedup = ratio(
        throughput["B16_BF16_FUSED_ADAMW"], throughput["B16_FP32_ADAMW"]
    )
    batch32_speedup = ratio(
        throughput["B32_BF16_FUSED_ADAMW"],
        throughput["B16_BF16_FUSED_ADAMW"],
    )
    batch64_relative = ratio(
        throughput["B64_BF16_FUSED_ADAMW"],
        throughput["B32_BF16_FUSED_ADAMW"],
    )
    worker_rows = {
        int(row["num_workers"]): finite_positive(
            row["records_per_second"], f"workers={row['num_workers']}"
        )
        for row in loader_profiles.values()
    }
    best_worker_count = max(worker_rows, key=worker_rows.get)
    best_worker_speedup = ratio(worker_rows[best_worker_count], worker_rows[0])

    bottlenecks = []
    if bf16_speedup >= 1.10:
        bottlenecks.append("FP32_COMPUTE_WAS_MATERIALLY_SLOWER_BF16_RETAINED")
    if batch32_speedup >= 1.10:
        bottlenecks.append("BATCH16_UNDERUTILIZED_GPU_FOR_NEW_MATCHED_RUNS")
    if batch64_relative < 0.95:
        bottlenecks.append("BATCH64_THROUGHPUT_REGRESSION_DO_NOT_SCALE_FURTHER")
    if best_worker_count == 0 or best_worker_speedup < 1.05:
        bottlenecks.append("HOST_DATALOADER_WORKERS_NOT_A_MATERIAL_BOTTLENECK")
        selected_workers = 0
    else:
        bottlenecks.append("HOST_DATALOADER_WORKERS_MATERIALLY_IMPROVE_THROUGHPUT")
        selected_workers = best_worker_count

    full_loop_records_per_second = train_records * epochs / wall_time
    b16_microbenchmark = throughput["B16_BF16_FUSED_ADAMW"]
    full_loop_to_microbenchmark = ratio(full_loop_records_per_second, b16_microbenchmark)
    if full_loop_to_microbenchmark < 0.75:
        bottlenecks.append(
            "FULL_LOOP_VALIDATION_CHECKPOINT_OR_PYTHON_OVERHEAD_IS_MATERIAL"
        )

    return {
        "schema_version": "route_a_v3_route2_mrnabert_huber_throughput_analysis.v1",
        "status": "HUBER_FULL_LOOP_AND_MATCHED_ENGINEERING_BENCHMARKS_COMPLETE",
        "huber_full_loop": {
            "epochs": epochs,
            "train_record_count": train_records,
            "validation_record_count": validation_records,
            "optimizer_steps": optimizer_steps,
            "wall_time_seconds": wall_time,
            "average_wall_time_seconds_per_epoch_including_validation_and_checkpointing": wall_time
            / epochs,
            "average_optimizer_steps_per_epoch": optimizer_steps / epochs,
            "apparent_train_record_presentations_per_second_including_full_loop_overhead": full_loop_records_per_second,
            "ratio_to_b16_bf16_microbenchmark": full_loop_to_microbenchmark,
        },
        "matched_engineering_results": {
            "b16_bf16_over_b16_fp32_speedup": bf16_speedup,
            "b32_over_b16_bf16_speedup": batch32_speedup,
            "b64_relative_to_b32_bf16": batch64_relative,
            "worker_records_per_second": {
                str(key): value for key, value in sorted(worker_rows.items())
            },
            "best_worker_count": best_worker_count,
            "best_worker_speedup_over_workers0": best_worker_speedup,
        },
        "engineering_bottleneck_findings": bottlenecks,
        "recommended_profile_for_a_new_fully_matched_cohort": {
            "batch_size": 32 if batch32_speedup >= 1.10 and batch64_relative <= 1.0 else 16,
            "training_precision": "BF16",
            "optimizer_fused": True,
            "num_workers": selected_workers,
            "pin_memory": True,
            "non_blocking_transfer": True,
        },
        "current_three_loss_comparison_configuration_changed": False,
        "recommendation_scope": (
            "FUTURE_FULLY_MATCHED_COHORT_ONLY;_DO_NOT_CHANGE_FIXED_OR_LEARNED_"
            "LOSS_RUNS_RELATIVE_TO_THE_ALREADY_RUNNING_B16_HUBER_RUN"
        ),
        "development_test_opened": False,
        "evaluation_opened": False,
        "scientific_result": "NOT_EVALUATED_ENGINEERING_THROUGHPUT_ONLY",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--huber-summary", type=Path, required=True)
    parser.add_argument("--gpu-benchmark", type=Path, required=True)
    parser.add_argument("--dataloader-benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"output already exists: {args.output}")
    result = analyze(
        load(args.huber_summary),
        load(args.gpu_benchmark),
        load(args.dataloader_benchmark),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
