#!/usr/bin/env python3
"""Evaluate Route 2 A1 effects and true-A2 rankings with source-group units."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.stats import spearmanr


class EvaluationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


def _finite_number(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    number = float(value)
    _require(math.isfinite(number), f"{label} is not finite")
    return number


def load_predictions(path: Path) -> dict[str, float]:
    predictions: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(f"invalid prediction JSON at line {line_number}") from exc
            record_id = str(row["canonical_record_id"])
            _require(record_id not in predictions, f"prediction duplicated: {record_id}")
            predictions[record_id] = _finite_number(
                row["predicted_direction_normalized_delta"], "predicted direction-normalized delta"
            )
    return predictions


def load_manifest(
    path: Path,
    requested_split: str | None = None,
    requested_study: str | None = None,
) -> tuple[set[str], set[str]]:
    _require((requested_split is None) != (requested_study is None), "select exactly one split or study")
    selected: set[str] = set()
    pool_assignments: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(f"invalid manifest JSON at line {line_number}") from exc
            if (
                (requested_split is not None and row["split"] == requested_split)
                or (requested_study is not None and row["study_unit_id"] == requested_study)
            ):
                record_id = str(row["canonical_record_id"])
                _require(record_id not in selected, f"manifest id duplicated: {record_id}")
                selected.add(record_id)
                pool_assignments.add(str(row["pool_assignment"]))
    _require(selected, f"manifest selection is empty: {requested_split or requested_study}")
    _require(pool_assignments <= {"DEVELOPMENT", "EVALUATION"}, "manifest selection has an unsupported pool")
    _require(len(pool_assignments) == 1, "manifest selection mixes Development and Evaluation")
    return selected, pool_assignments


def require_evaluation_release(pool_assignments: set[str], release_state: str) -> None:
    if pool_assignments == {"EVALUATION"}:
        _require(
            release_state == "PREDICTOR_GENERATOR_AND_BASELINES_FROZEN",
            "Evaluation outcomes remain closed until predictor, generator, and baselines are frozen",
        )


def load_observations(canonical_paths: list[Path], selected_ids: set[str]) -> list[dict[str, Any]]:
    observations: dict[str, dict[str, Any]] = {}
    for path in canonical_paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EvaluationError(f"invalid canonical JSON in {path.name}:{line_number}") from exc
                record_id = str(row["canonical_record_id"])
                if record_id not in selected_ids:
                    continue
                _require(record_id not in observations, f"canonical id duplicated: {record_id}")
                observations[record_id] = {
                    "canonical_record_id": record_id,
                    "study_unit_id": str(row["study_unit_id"]),
                    "source_id": str(row["source_id"]),
                    "biological_context_id": str(row["biological_context_id"]),
                    "endpoint_id": str(row["endpoint_id"]),
                    "stratum": (str(row["study_unit_id"]), str(row["region"]), str(row["endpoint_id"])),
                    "task": (str(row["region"]), str(row["endpoint_id"])),
                    "observed": _finite_number(row["direction_normalized_delta"], "observed direction-normalized delta"),
                }
    _require(set(observations) == selected_ids, "canonical observations do not exactly cover the requested manifest split")
    return [observations[record_id] for record_id in sorted(observations)]


def _spearman(observed: np.ndarray, predicted: np.ndarray) -> float | None:
    if len(observed) < 3 or np.std(observed) == 0.0 or np.std(predicted) == 0.0:
        return None
    value = spearmanr(observed, predicted).statistic
    return None if not math.isfinite(float(value)) else float(value)


def numeric_metrics(observed: Iterable[float], predicted: Iterable[float]) -> dict[str, Any]:
    observed_array = np.asarray(list(observed), dtype=float)
    predicted_array = np.asarray(list(predicted), dtype=float)
    _require(len(observed_array) == len(predicted_array) and len(observed_array) > 0, "numeric metric inputs differ")
    error = predicted_array - observed_array
    return {
        "record_count": int(len(observed_array)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "spearman": _spearman(observed_array, predicted_array),
        "sign_accuracy": float(np.mean(np.sign(observed_array) == np.sign(predicted_array))),
    }


def _descending_tie_blocks(values: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(-values, kind="stable")
    blocks = []
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        blocks.append(order[start:end])
        start = end
    return blocks


def _tie_aware_dcg(gains: np.ndarray, scores: np.ndarray, k: int) -> float:
    result = 0.0
    rank = 0
    for block in _descending_tie_blocks(scores):
        used = min(len(block), max(0, k - rank))
        if used:
            mean_gain = float(np.mean(gains[block]))
            result += mean_gain * sum(1.0 / math.log2(position + 2.0) for position in range(rank, rank + used))
        rank += len(block)
        if rank >= k:
            break
    return result


def _top_k_inclusion_probabilities(scores: np.ndarray, k: int) -> np.ndarray:
    probabilities = np.zeros(len(scores), dtype=float)
    rank = 0
    for block in _descending_tie_blocks(scores):
        remaining = max(0, k - rank)
        if remaining <= 0:
            break
        probabilities[block] = min(1.0, remaining / len(block))
        rank += len(block)
    return probabilities


def ranking_metrics(observed: np.ndarray, predicted: np.ndarray, k: int) -> dict[str, float | int | None]:
    _require(len(observed) == len(predicted) and len(observed) >= 2, "ranking pool must contain at least two candidates")
    k = min(k, len(observed))
    true_order = np.argsort(-observed, kind="stable")
    shifted = observed - np.min(observed)
    dcg = _tie_aware_dcg(shifted, predicted, k)
    idcg = _tie_aware_dcg(shifted, observed, k)
    true_cutoff = observed[true_order[k - 1]]
    true_top_eligible = observed >= true_cutoff
    predicted_inclusion = _top_k_inclusion_probabilities(predicted, k)
    span = float(np.max(observed) - np.min(observed))
    predicted_best = predicted == np.max(predicted)
    selected_expected_outcome = float(np.mean(observed[predicted_best]))
    regret = 0.0 if span == 0.0 else float((np.max(observed) - selected_expected_outcome) / span)
    true_best = observed == np.max(observed)
    pairwise_scores = []
    for left in range(len(observed)):
        for right in range(left + 1, len(observed)):
            observed_sign = float(np.sign(observed[left] - observed[right]))
            if observed_sign == 0.0:
                continue
            predicted_sign = float(np.sign(predicted[left] - predicted[right]))
            pairwise_scores.append(0.5 if predicted_sign == 0.0 else float(predicted_sign == observed_sign))
    return {
        "candidate_count": int(len(observed)),
        "ndcg_at_k": None if idcg == 0.0 else dcg / idcg,
        "top_k_recall": float(np.sum(predicted_inclusion[true_top_eligible]) / k),
        "normalized_regret": regret,
        "top_1_accuracy": float(np.mean(true_best[predicted_best])),
        "pairwise_accuracy": None if not pairwise_scores else float(np.mean(pairwise_scores)),
        "within_source_spearman": _spearman(observed, predicted),
    }


def _macro(metrics: list[Mapping[str, Any]], key: str) -> float | None:
    values = [float(metric[key]) for metric in metrics if metric.get(key) is not None]
    return None if not values else float(np.mean(values))


def evaluate(observations: list[dict[str, Any]], predictions: Mapping[str, float], k: int) -> dict[str, Any]:
    observation_ids = {row["canonical_record_id"] for row in observations}
    _require(set(predictions) == observation_ids, "predictions do not exactly cover the requested manifest split")
    observed = np.asarray([row["observed"] for row in observations], dtype=float)
    predicted = np.asarray([predictions[row["canonical_record_id"]] for row in observations], dtype=float)
    overall = numeric_metrics(observed, predicted)

    by_source: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    by_stratum: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    by_task: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(observations):
        by_source[(row["study_unit_id"], row["source_id"], row["biological_context_id"], row["endpoint_id"])].append(index)
        by_stratum[row["stratum"]].append(index)
        by_task[row.get("task", (row["stratum"][1], row["endpoint_id"]))].append(index)
    source_numeric = [
        numeric_metrics(observed[indices], predicted[indices])
        for indices in by_source.values()
    ]
    rankable = [
        ranking_metrics(observed[indices], predicted[indices], k)
        for indices in by_source.values() if len(indices) >= 2
    ]
    stratum_metrics = {
        "|".join(stratum): numeric_metrics(observed[indices], predicted[indices])
        for stratum, indices in sorted(by_stratum.items())
    }
    task_metrics = {
        "|".join(task): numeric_metrics(observed[indices], predicted[indices])
        for task, indices in sorted(by_task.items())
    }
    task_spearman_values = [metric["spearman"] for metric in task_metrics.values()]
    task_spearman_defined = [float(value) for value in task_spearman_values if value is not None]
    return {
        "overall_numeric": overall,
        "source_group_count": len(by_source),
        "source_macro_mae": _macro(source_numeric, "mae"),
        "rankable_source_group_count": len(rankable),
        "rankable_record_count": sum(metric["candidate_count"] for metric in rankable),
        "source_macro_ndcg_at_k": _macro(rankable, "ndcg_at_k"),
        "source_macro_top_k_recall": _macro(rankable, "top_k_recall"),
        "source_macro_normalized_regret": _macro(rankable, "normalized_regret"),
        "source_macro_top_1_accuracy": _macro(rankable, "top_1_accuracy"),
        "source_macro_pairwise_accuracy": _macro(rankable, "pairwise_accuracy"),
        "source_macro_within_source_spearman": _macro(rankable, "within_source_spearman"),
        "stratum_numeric": stratum_metrics,
        "task_numeric": task_metrics,
        "task_count": len(task_metrics),
        "task_spearman_defined_count": len(task_spearman_defined),
        "task_macro_spearman": (
            float(np.mean(task_spearman_defined))
            if len(task_spearman_defined) == len(task_metrics) else None
        ),
    }


def paired_group_bootstrap(
    observations: list[dict[str, Any]],
    predictions: Mapping[str, float],
    baseline_predictions: Mapping[str, float],
    iterations: int,
    seed: int,
    k: int,
) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        groups[(row["study_unit_id"], row["source_id"], row["biological_context_id"], row["endpoint_id"])].append(row)
    keys = sorted(groups)
    _require(len(keys) >= 2, "source-group bootstrap requires at least two groups")
    per_group_mae_improvement = []
    for key in keys:
        rows = groups[key]
        model_error = np.mean([
            abs(predictions[row["canonical_record_id"]] - row["observed"]) for row in rows
        ])
        baseline_error = np.mean([
            abs(baseline_predictions[row["canonical_record_id"]] - row["observed"]) for row in rows
        ])
        per_group_mae_improvement.append(float(baseline_error - model_error))

    def task_macro_spearman(rows: list[dict[str, Any]], values: Mapping[str, float]) -> float | None:
        by_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            task = row.get("task", (row["stratum"][1], row["endpoint_id"]))
            by_task[tuple(task)].append(row)
        correlations = []
        for task_rows in by_task.values():
            observed = np.asarray([row["observed"] for row in task_rows], dtype=float)
            predicted = np.asarray([values[row["canonical_record_id"]] for row in task_rows], dtype=float)
            correlation = _spearman(observed, predicted)
            if correlation is None:
                return None
            correlations.append(correlation)
        return float(np.mean(correlations))

    point_model_spearman = task_macro_spearman(observations, predictions)
    point_baseline_spearman = task_macro_spearman(observations, baseline_predictions)
    ranking_improvements = []
    for key in keys:
        rows = groups[key]
        if len(rows) < 2:
            continue
        observed = np.asarray([row["observed"] for row in rows], dtype=float)
        model_values = np.asarray([predictions[row["canonical_record_id"]] for row in rows], dtype=float)
        baseline_values = np.asarray([baseline_predictions[row["canonical_record_id"]] for row in rows], dtype=float)
        model_ranking = ranking_metrics(observed, model_values, k)
        baseline_ranking = ranking_metrics(observed, baseline_values, k)
        ranking_improvements.append({
            "ndcg": (
                None
                if model_ranking["ndcg_at_k"] is None or baseline_ranking["ndcg_at_k"] is None
                else float(model_ranking["ndcg_at_k"] - baseline_ranking["ndcg_at_k"])
            ),
            "regret_reduction": float(
                baseline_ranking["normalized_regret"] - model_ranking["normalized_regret"]
            ),
            "top_1": float(model_ranking["top_1_accuracy"] - baseline_ranking["top_1_accuracy"]),
        })
    randomizer = np.random.default_rng(seed)
    mae_samples = []
    spearman_samples = []
    for _ in range(iterations):
        sampled_indices = randomizer.integers(0, len(keys), size=len(keys))
        sampled_rows = [row for index in sampled_indices for row in groups[keys[int(index)]]]
        mae_samples.append(float(np.mean([per_group_mae_improvement[int(index)] for index in sampled_indices])))
        model_spearman = task_macro_spearman(sampled_rows, predictions)
        baseline_spearman = task_macro_spearman(sampled_rows, baseline_predictions)
        if model_spearman is not None and baseline_spearman is not None:
            spearman_samples.append(model_spearman - baseline_spearman)
    mae_sample_array = np.asarray(mae_samples, dtype=float)
    spearman_sample_array = np.asarray(spearman_samples, dtype=float)
    ranking_summary = None
    if ranking_improvements:
        ranking_summary = {"rankable_source_group_count": len(ranking_improvements)}
        randomizer_ranking = np.random.default_rng(seed + 1)
        for field, label in (
            ("ndcg", "MODEL_NDCG_MINUS_BASELINE_NDCG"),
            ("regret_reduction", "BASELINE_REGRET_MINUS_MODEL_REGRET"),
            ("top_1", "MODEL_TOP1_MINUS_BASELINE_TOP1"),
        ):
            values = [float(row[field]) for row in ranking_improvements if row[field] is not None]
            if not values:
                ranking_summary[field] = {
                    "metric": label,
                    "defined_source_group_count": 0,
                    "mean_improvement": None,
                    "bootstrap_ci_95": None,
                }
                continue
            bootstrap = np.asarray([
                np.mean(randomizer_ranking.choice(values, size=len(values), replace=True))
                for _ in range(iterations)
            ])
            ranking_summary[field] = {
                "metric": label,
                "defined_source_group_count": len(values),
                "mean_improvement": float(np.mean(values)),
                "bootstrap_ci_95": [
                    float(np.quantile(bootstrap, 0.025)),
                    float(np.quantile(bootstrap, 0.975)),
                ],
            }
    return {
        "analysis_unit": "SOURCE_GROUP",
        "source_group_count": len(keys),
        "bootstrap_iterations": iterations,
        "task_macro_spearman": {
            "model": point_model_spearman,
            "baseline": point_baseline_spearman,
            "improvement": (
                None
                if point_model_spearman is None or point_baseline_spearman is None
                else point_model_spearman - point_baseline_spearman
            ),
            "defined_bootstrap_iterations": len(spearman_samples),
            "bootstrap_ci_95": (
                None
                if not spearman_samples
                else [
                    float(np.quantile(spearman_sample_array, 0.025)),
                    float(np.quantile(spearman_sample_array, 0.975)),
                ]
            ),
        },
        "mae": {
            "metric": "BASELINE_MAE_MINUS_MODEL_MAE",
            "mean_improvement": float(np.mean(per_group_mae_improvement)),
            "bootstrap_ci_95": [
                float(np.quantile(mae_sample_array, 0.025)),
                float(np.quantile(mae_sample_array, 0.975)),
            ],
        },
        "ranking": ranking_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--split")
    selection.add_argument("--study-unit-id")
    parser.add_argument("--canonical", type=Path, action="append", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path)
    parser.add_argument("--evaluation-release-state", default="CLOSED")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected, pool_assignments = load_manifest(args.manifest, args.split, args.study_unit_id)
    require_evaluation_release(pool_assignments, args.evaluation_release_state)
    observations = load_observations(args.canonical, selected)
    predictions = load_predictions(args.predictions)
    result = {
        "schema_version": "route_a_v3_route2_prediction_evaluation.v1",
        "split": args.split or f"LOSO::{args.study_unit_id}",
        "metrics": evaluate(observations, predictions, args.k),
        "evaluation_release_state": args.evaluation_release_state,
    }
    if args.baseline_predictions:
        baseline_predictions = load_predictions(args.baseline_predictions)
        _require(set(baseline_predictions) == selected, "baseline predictions do not exactly cover the requested split")
        result["paired_baseline_comparison"] = paired_group_bootstrap(
            observations,
            predictions,
            baseline_predictions,
            args.bootstrap_iterations,
            args.seed,
            args.k,
        )
    _require(not args.output.exists(), f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
