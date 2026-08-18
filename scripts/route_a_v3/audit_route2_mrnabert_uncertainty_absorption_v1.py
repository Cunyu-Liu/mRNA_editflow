#!/usr/bin/env python3
"""Audit mean collapse and uncertainty absorption across matched mRNABERT losses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from scipy.stats import spearmanr


EXPECTED_LOSSES = {
    "huber",
    "fixed_variance_gaussian_nll",
    "learned_variance_gaussian_nll",
}

# Training metrics are accumulated from float32 tensors, while this audit
# reconstructs scaled targets from JSON-serialized canonical values and scales.
# The reconstruction is scientifically identical but can differ at float32
# round-off.  This tolerance remains much smaller than any reported metric
# precision and still rejects a material metric mismatch.
FLOAT32_METRIC_RECONSTRUCTION_ATOL = 5e-7


class UncertaintyAuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise UncertaintyAuditError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL input: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def finite_number(value: Any, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    result = float(value)
    require(math.isfinite(result), f"{label} is nonfinite")
    return result


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return None
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else None


def load_validation_truth(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    assignments = {}
    for row in load_jsonl(Path(config["development_manifest"])):
        record_id = str(row["canonical_record_id"])
        require(record_id not in assignments, "Development manifest contains duplicate records")
        assignments[record_id] = str(row["split"])
    validation_ids = {record_id for record_id, split in assignments.items() if split == "VALIDATION"}
    require(validation_ids, "Development VALIDATION is empty")
    truth = {}
    for path_text in config["canonical_paths"]:
        for row in load_jsonl(Path(path_text)):
            record_id = str(row["canonical_record_id"])
            if record_id not in validation_ids:
                continue
            require(row["pool_assignment"] == "DEVELOPMENT", "Evaluation record entered uncertainty audit")
            require(record_id not in truth, "canonical VALIDATION record is duplicated")
            region = str(row["region"]).replace("′", "").replace("'", "")
            require(region in {"5UTR", "3UTR"}, "unsupported region entered uncertainty audit")
            truth[record_id] = {
                "target": finite_number(row["direction_normalized_delta"], "target"),
                "task_key": f"{row['endpoint_id']}::region={0 if region == '5UTR' else 1}",
            }
    require(set(truth) == validation_ids, "canonical inputs do not exactly cover VALIDATION")
    return truth


def summarize_run(
    *,
    summary_path: Path,
    shared_truth: dict[str, dict[str, Any]] | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    summary = load_json(summary_path)
    config = load_json(summary_path.parent / "training_config.json")
    require(summary.get("status") == "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE", "loss run is incomplete")
    require(summary.get("result_stage") == "HPO_VALIDATION_ONLY", "audit is not VALIDATION-only")
    require(summary.get("evaluation_outcomes_read") == 0, "Evaluation entered uncertainty audit")
    require(summary.get("test_metrics") is None, "Development TEST entered uncertainty audit")
    truth = load_validation_truth(config)
    if shared_truth is not None:
        require(truth == shared_truth, "matched loss runs do not share identical VALIDATION truth")
    prediction_rows = load_jsonl(summary_path.parent / "validation_predictions.jsonl")
    predictions = {str(row["canonical_record_id"]): row for row in prediction_rows}
    require(len(predictions) == len(prediction_rows), "prediction records are duplicated")
    require(set(predictions) == set(truth), "predictions do not exactly cover VALIDATION")
    by_task: dict[str, list[str]] = {}
    for record_id, value in truth.items():
        by_task.setdefault(value["task_key"], []).append(record_id)
    task_rows = []
    for task_key, record_ids in sorted(by_task.items()):
        targets = []
        means = []
        standard_deviations = []
        for record_id in record_ids:
            prediction = predictions[record_id]
            target_scale = finite_number(prediction["target_scale"], "target scale")
            require(target_scale > 0, "target scale must be positive")
            targets.append(truth[record_id]["target"] / target_scale)
            means.append(finite_number(prediction["predicted_standardized_delta"], "standardized mean"))
            variance = prediction.get("predicted_variance")
            if variance is not None:
                variance_value = finite_number(variance, "predicted variance")
                require(variance_value >= 0, "predicted variance is negative")
                standard_deviations.append(math.sqrt(variance_value) / target_scale)
        target_values = np.asarray(targets)
        mean_values = np.asarray(means)
        target_std = float(np.std(target_values))
        prediction_std = float(np.std(mean_values))
        row: dict[str, Any] = {
            "task_key": task_key,
            "record_count": len(record_ids),
            "mean_spearman": rank_correlation(target_values, mean_values),
            "standardized_mae": float(np.mean(np.abs(mean_values - target_values))),
            "standardized_target_std": target_std,
            "standardized_prediction_std": prediction_std,
            "prediction_spread_ratio": prediction_std / max(target_std, 1e-12),
        }
        if standard_deviations:
            require(len(standard_deviations) == len(record_ids), "uncertainty is partial within a task")
            scale_values = np.asarray(standard_deviations)
            row.update(
                {
                    "predicted_standard_deviation_mean": float(np.mean(scale_values)),
                    "uncertainty_absolute_residual_spearman": rank_correlation(
                        scale_values, np.abs(mean_values - target_values)
                    ),
                }
            )
        task_rows.append(row)
    correlations = [row["mean_spearman"] for row in task_rows if row["mean_spearman"] is not None]
    spread_ratios = [row["prediction_spread_ratio"] for row in task_rows]
    uncertainty_correlations = [
        row["uncertainty_absolute_residual_spearman"]
        for row in task_rows
        if row.get("uncertainty_absolute_residual_spearman") is not None
    ]
    result = {
        "loss_kind": str(summary["loss_kind"]),
        "baseline_id": str(summary["baseline_id"]),
        "selected_epoch": int(summary["selected_epoch"]),
        "task_count": len(task_rows),
        "task_macro_spearman": float(np.mean(correlations)),
        "task_median_spearman": float(median(correlations)),
        "positive_task_count": sum(value > 0 for value in correlations),
        "task_macro_standardized_mae": float(
            np.mean([row["standardized_mae"] for row in task_rows])
        ),
        "task_macro_prediction_spread_ratio": float(np.mean(spread_ratios)),
        "task_median_prediction_spread_ratio": float(median(spread_ratios)),
        "task_macro_uncertainty_absolute_residual_spearman": (
            float(np.mean(uncertainty_correlations)) if uncertainty_correlations else None
        ),
        "task_rows": task_rows,
    }
    recorded = summary["validation_metrics"]
    require(
        abs(result["task_macro_spearman"] - float(recorded["task_macro_spearman"])) < 1e-9,
        "recomputed task-macro Spearman differs from training summary",
    )
    require(
        abs(result["task_macro_standardized_mae"] - float(recorded["task_macro_standardized_mae"]))
        <= FLOAT32_METRIC_RECONSTRUCTION_ATOL,
        "recomputed standardized MAE differs from training summary",
    )
    return result, truth


def audit(summary_paths: list[Path], loss_comparison_path: Path) -> dict[str, Any]:
    require(len(summary_paths) == 3, "exactly three loss summaries are required")
    rows = []
    shared_truth = None
    for summary_path in summary_paths:
        row, shared_truth = summarize_run(
            summary_path=summary_path,
            shared_truth=shared_truth,
        )
        rows.append(row)
    by_loss = {row["loss_kind"]: row for row in rows}
    require(set(by_loss) == EXPECTED_LOSSES, "loss set differs from the matched comparison")
    comparison = load_json(loss_comparison_path)
    selected_loss = str(comparison["selected_loss_for_controls"])
    require(selected_loss in EXPECTED_LOSSES, "loss comparison selected an unknown loss")
    learned = by_loss["learned_variance_gaussian_nll"]
    huber = by_loss["huber"]
    fixed = by_loss["fixed_variance_gaussian_nll"]
    return {
        "schema_version": "route_a_v3_route2_mrnabert_uncertainty_absorption_audit.v1",
        "status": "THREE_LOSS_DEVELOPMENT_VALIDATION_UNCERTAINTY_AUDIT_COMPLETE",
        "selection_rule_unchanged": True,
        "selected_loss_for_controls": selected_loss,
        "rows": sorted(rows, key=lambda row: row["loss_kind"]),
        "learned_uncertainty_comparison": {
            "task_macro_spearman_difference_from_huber": learned["task_macro_spearman"] - huber["task_macro_spearman"],
            "task_macro_spearman_difference_from_fixed_variance": learned["task_macro_spearman"] - fixed["task_macro_spearman"],
            "task_median_prediction_spread_ratio_difference_from_huber": learned["task_median_prediction_spread_ratio"] - huber["task_median_prediction_spread_ratio"],
            "task_median_prediction_spread_ratio_difference_from_fixed_variance": learned["task_median_prediction_spread_ratio"] - fixed["task_median_prediction_spread_ratio"],
            "task_macro_uncertainty_absolute_residual_spearman": learned["task_macro_uncertainty_absolute_residual_spearman"],
            "interpretation": "MEAN_METRICS_SELECT_THE_LOSS;_UNCERTAINTY_AND_SPREAD_ARE_DIAGNOSTIC_ONLY",
        },
        "development_validation_record_count": len(shared_truth or {}),
        "development_test_opened": False,
        "evaluation_opened": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--loss-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    require(not arguments.output.exists(), f"output already exists: {arguments.output}")
    result = audit(arguments.summary, arguments.loss_comparison)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "selected_loss_for_controls")}, sort_keys=True))


if __name__ == "__main__":
    main()
