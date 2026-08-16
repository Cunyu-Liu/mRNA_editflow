#!/usr/bin/env python3
"""Aggregate aligned Route 2 model/baseline LOSO evaluations by study."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


class LosoAggregationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LosoAggregationError(message)


def _spearman(result: Mapping[str, Any], study: str) -> float:
    _require(result["split"] == f"LOSO::{study}", f"LOSO result study differs: {study}")
    metrics = result["metrics"]
    _require(
        metrics["task_spearman_defined_count"] == metrics["task_count"],
        f"LOSO task Spearman is undefined: {study}",
    )
    value = metrics["task_macro_spearman"]
    _require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)), f"LOSO task-macro Spearman is undefined: {study}")
    return float(value)


def _verified_model_training(summary: Mapping[str, Any], study: str, seed: int) -> bool:
    physical_index = summary.get("physical_gpu_index")
    total_memory = summary.get("cuda_total_memory_mb")
    return (
        summary.get("status") == "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE"
        and summary.get("run_mode") == "LOSO_FROZEN_HYPERPARAMETERS"
        and summary.get("result_stage") == "LOSO_FROZEN_HYPERPARAMETERS"
        and summary.get("loso_holdout_study_unit_id") == study
        and summary.get("seed") == seed
        and summary.get("optimizer_steps", 0) > 0
        and summary.get("parameter_changed") is True
        and summary.get("cuda_training_tensors_verified") is True
        and isinstance(physical_index, int)
        and not isinstance(physical_index, bool)
        and physical_index >= 0
        and summary.get("device") == f"cuda:{physical_index}"
        and summary.get("cpu_fallback_used") is False
        and summary.get("cuda_device_index") == physical_index
        and isinstance(summary.get("cuda_device_uuid"), str)
        and bool(summary.get("cuda_device_uuid"))
        and isinstance(total_memory, (int, float))
        and not isinstance(total_memory, bool)
        and math.isfinite(float(total_memory))
        and float(total_memory) > 0.0
        and summary.get("evaluation_outcomes_read") == 0
    )


def aggregate(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload["schema_version"] == "route_a_v3_route2_loso_aggregation_input.v1", "unexpected LOSO input schema")
    inventory = set(payload["development_inventory_studies"])
    expected = set(payload["expected_loso_studies"])
    zero_record = set(payload["zero_record_development_studies"])
    _require(len(inventory) == 8, "Development inventory requires exactly eight studies")
    _require(len(expected) >= 2, "LOSO requires at least two nonempty Development studies")
    _require(expected.isdisjoint(zero_record), "zero-record study entered LOSO")
    _require(expected | zero_record == inventory, "LOSO and zero-record study sets do not close to inventory")
    model_rows = {row["study_unit_id"]: row for row in payload["model_results"]}
    baseline_rows = {row["study_unit_id"]: row for row in payload["baseline_results"]}
    _require(set(model_rows) == set(baseline_rows) == expected, "model/baseline LOSO study sets differ")
    per_study = []
    for study in sorted(expected):
        _require(
            _verified_model_training(model_rows[study]["training_summary"], study, int(payload["seed"])),
            f"LOSO model training provenance is invalid: {study}",
        )
        model = _spearman(model_rows[study]["evaluation"], study)
        baseline = _spearman(baseline_rows[study]["evaluation"], study)
        per_study.append({
            "study_unit_id": study,
            "model_task_macro_spearman": model,
            "baseline_task_macro_spearman": baseline,
            "improvement": model - baseline,
        })
    model_macro = sum(row["model_task_macro_spearman"] for row in per_study) / len(per_study)
    baseline_macro = sum(row["baseline_task_macro_spearman"] for row in per_study) / len(per_study)
    return {
        "schema_version": "route_a_v3_route2_loso_aggregation.v1",
        "status": "LOSO_MODEL_BASELINE_ALIGNED_COMPLETE",
        "seed": payload["seed"],
        "study_count": len(per_study),
        "development_inventory_study_count": len(inventory),
        "zero_record_development_studies": sorted(zero_record),
        "model_macro_spearman": model_macro,
        "baseline_macro_spearman": baseline_macro,
        "within_study_metric": "TASK_MACRO_SPEARMAN_REGION_ENDPOINT",
        "macro_improvement": model_macro - baseline_macro,
        "per_study": per_study,
        "all_model_training_gpu_provenance_verified": True,
        "evaluation_studies_included": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"output already exists: {args.output}")
    result = aggregate(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
