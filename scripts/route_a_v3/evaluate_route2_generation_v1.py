#!/usr/bin/env python3
"""Evaluate Route 2 generated candidates without granting canonical credit."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


ALPHABET = set("ACGU")
TERMINAL_CAUSES = {
    "EXPLICIT_STOP",
    "BUDGET_EXHAUSTED",
    "NO_LEGAL_ACTION",
    "NUMERICAL_FAILURE",
}
CANDIDATE_SUPPORT_MODES = {
    "OPEN_GENERATED_SUPPORT",
    "CLOSED_MEASURED_SUPPORT",
}


class GenerationEvaluationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GenerationEvaluationError(message)


def _finite(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    result = float(value)
    _require(math.isfinite(result), f"{label} is not finite")
    return result


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GenerationEvaluationError(f"invalid JSON in {path.name}:{line_number}") from exc
            _require(isinstance(row, dict), f"row is not an object in {path.name}:{line_number}")
            rows.append(row)
    _require(rows, f"input is empty: {path}")
    return rows


def load_source_manifest(path: Path) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        source_key = str(row["source_key"])
        source = str(row["source_sequence"]).upper().replace("T", "U")
        _require(source_key not in sources, f"source duplicated: {source_key}")
        _require(source and set(source) <= ALPHABET, f"invalid source sequence: {source_key}")
        edit_budget = row["edit_budget"]
        candidate_budget = row["candidate_budget"]
        _require(isinstance(edit_budget, int) and not isinstance(edit_budget, bool) and edit_budget >= 0, "invalid edit budget")
        _require(isinstance(candidate_budget, int) and not isinstance(candidate_budget, bool) and candidate_budget > 0, "invalid candidate budget")
        sources[source_key] = {
            "source_sequence": source,
            "edit_budget": edit_budget,
            "candidate_budget": candidate_budget,
            "source_critic_score": row.get("source_critic_score"),
            "source_independent_evaluator_score": row.get("source_independent_evaluator_score"),
        }
    return sources


def _edit_distance(source: str, candidate: str) -> int | None:
    if len(source) != len(candidate):
        return None
    return sum(left != right for left, right in zip(source, candidate))


def _pairwise_diversity(sequences: list[str]) -> float | None:
    if len(sequences) < 2:
        return None
    distances = []
    for left_index, left in enumerate(sequences):
        for right in sequences[left_index + 1 :]:
            if len(left) == len(right):
                distances.append(sum(a != b for a, b in zip(left, right)) / len(left))
    return None if not distances else float(np.mean(distances))


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return None if not finite else float(np.mean(finite))


def _score_summary(rows: list[dict[str, Any]], field: str, source_score: Any) -> dict[str, Any] | None:
    values = [_finite(row[field], field) for row in rows if row.get(field) is not None]
    if not values:
        return None
    result: dict[str, Any] = {"mean": float(np.mean(values)), "max": float(np.max(values)), "count": len(values)}
    if source_score is not None:
        source_value = _finite(source_score, f"source {field}")
        result["mean_uplift_over_source"] = float(np.mean(values) - source_value)
        result["max_uplift_over_source"] = float(np.max(values) - source_value)
    return result


def evaluate_generation(
    sources: Mapping[str, Mapping[str, Any]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    methods = {str(row["method_id"]) for row in candidates}
    _require(len(methods) == 1, "one evaluation file must contain exactly one method")
    for row in candidates:
        source_key = str(row["source_key"])
        _require(source_key in sources, f"candidate has unknown source: {source_key}")
        by_source[source_key].append(row)
    _require(set(by_source) == set(sources), "generated candidates do not cover the exact source manifest")

    per_source: dict[str, dict[str, Any]] = {}
    all_terminal = Counter()
    total_rows = 0
    total_legal = 0
    total_budget_violations = 0
    total_candidate_budget_violations = 0
    for source_key in sorted(sources):
        spec = sources[source_key]
        rows = by_source[source_key]
        total_rows += len(rows)
        source = str(spec["source_sequence"])
        edit_budget = int(spec["edit_budget"])
        candidate_budget = int(spec["candidate_budget"])
        candidate_budget_violation = max(0, len(rows) - candidate_budget)
        total_candidate_budget_violations += candidate_budget_violation
        valid_sequences: list[str] = []
        edit_distances: list[int] = []
        budget_violations = 0
        for row in rows:
            candidate = str(row["candidate_sequence"]).upper().replace("T", "U")
            cause = str(row["terminal_cause"])
            _require(cause in TERMINAL_CAUSES, f"unknown terminal cause: {cause}")
            all_terminal[cause] += 1
            distance = _edit_distance(source, candidate)
            alphabet_legal = bool(candidate) and set(candidate) <= ALPHABET
            legal = alphabet_legal and distance is not None and distance <= edit_budget
            if distance is not None and distance > edit_budget:
                budget_violations += 1
            if legal:
                valid_sequences.append(candidate)
                edit_distances.append(int(distance))
                total_legal += 1
        total_budget_violations += budget_violations
        unique_sequences = sorted(set(valid_sequences))
        row_source_scores = {
            _finite(row["source_critic_score"], "source critic score")
            for row in rows if row.get("source_critic_score") is not None
        }
        _require(len(row_source_scores) <= 1, "source critic score differs within source")
        source_critic_score = spec.get("source_critic_score")
        if row_source_scores:
            row_source_score = next(iter(row_source_scores))
            if source_critic_score is not None:
                _require(
                    math.isclose(_finite(source_critic_score, "manifest source critic score"), row_source_score),
                    "manifest and candidate source critic scores differ",
                )
            source_critic_score = row_source_score
        row_source_evaluator_scores = {
            _finite(row["source_independent_evaluator_score"], "source independent evaluator score")
            for row in rows if row.get("source_independent_evaluator_score") is not None
        }
        _require(len(row_source_evaluator_scores) <= 1, "source independent evaluator score differs within source")
        source_evaluator_score = spec.get("source_independent_evaluator_score")
        if row_source_evaluator_scores:
            row_source_evaluator_score = next(iter(row_source_evaluator_scores))
            if source_evaluator_score is not None:
                _require(
                    math.isclose(
                        _finite(source_evaluator_score, "manifest source independent evaluator score"),
                        row_source_evaluator_score,
                    ),
                    "manifest and candidate source independent evaluator scores differ",
                )
            source_evaluator_score = row_source_evaluator_score
        forward_budgets = {
            int(row["critic_forward_budget"])
            for row in rows if row.get("critic_forward_budget") is not None
        }
        _require(len(forward_budgets) <= 1, "critic forward budget differs within source")
        compute = {
            field: float(sum(_finite(row.get(field, 0), field) for row in rows))
            for field in ("generator_nfe", "critic_forwards", "independent_evaluator_forwards")
        }
        compute["total_forward_equivalents"] = (
            compute["generator_nfe"] + compute["critic_forwards"] + compute["independent_evaluator_forwards"]
        )
        compute["critic_forward_budget"] = None if not forward_budgets else next(iter(forward_budgets))
        compute["critic_budget_utilization"] = (
            None if not forward_budgets else compute["critic_forwards"] / next(iter(forward_budgets))
        )
        wall_times = [_finite(row["wall_time_seconds"], "wall time") for row in rows if row.get("wall_time_seconds") is not None]
        peak_vram = [_finite(row["peak_vram_mb"], "peak VRAM") for row in rows if row.get("peak_vram_mb") is not None]
        per_source[source_key] = {
            "candidate_count": len(rows),
            "candidate_budget": candidate_budget,
            "candidate_budget_violation": candidate_budget_violation,
            "legal_candidate_count": len(valid_sequences),
            "legality_rate": len(valid_sequences) / len(rows),
            "budget_violation_count": budget_violations,
            "unique_candidate_count": len(unique_sequences),
            "unique_candidate_rate": len(unique_sequences) / len(rows),
            "edit_distance_distribution": dict(sorted(Counter(edit_distances).items())),
            "mean_pairwise_hamming_diversity": _pairwise_diversity(unique_sequences),
            "terminal_causes": dict(sorted(Counter(str(row["terminal_cause"]) for row in rows).items())),
            "critic_score": _score_summary(rows, "critic_score", source_critic_score),
            "independent_evaluator_score": _score_summary(
                rows,
                "independent_evaluator_score",
                source_evaluator_score,
            ),
            "compute": compute,
            "wall_time_seconds": None if not wall_times else float(max(wall_times)),
            "peak_vram_mb": None if not peak_vram else float(max(peak_vram)),
        }
    return {
        "method_id": next(iter(methods)),
        "source_count": len(sources),
        "candidate_count": total_rows,
        "legal_candidate_count": total_legal,
        "hard_legality_rate": total_legal / total_rows,
        "edit_budget_violation_count": total_budget_violations,
        "candidate_budget_violation_count": total_candidate_budget_violations,
        "source_macro_unique_candidate_rate": _mean(row["unique_candidate_rate"] for row in per_source.values()),
        "source_macro_pairwise_hamming_diversity": _mean(
            row["mean_pairwise_hamming_diversity"] for row in per_source.values()
        ),
        "terminal_causes": dict(sorted(all_terminal.items())),
        "generated_candidates_grant_canonical_credit": False,
        "per_source": per_source,
    }


def measured_neighborhood_metrics(
    sources: Mapping[str, Mapping[str, Any]],
    candidates: list[dict[str, Any]],
    measured_rows: list[dict[str, Any]],
    *,
    k: int,
    candidate_support_mode: str,
) -> dict[str, Any]:
    _require(candidate_support_mode in CANDIDATE_SUPPORT_MODES, "candidate support mode must be explicit")
    measured: dict[str, dict[str, float]] = defaultdict(dict)
    for row in measured_rows:
        source_key = str(row["source_key"])
        _require(source_key in sources, f"measured row has unknown source: {source_key}")
        sequence = str(row["candidate_sequence"]).upper().replace("T", "U")
        _require(sequence not in measured[source_key], f"measured candidate duplicated: {source_key}/{sequence}")
        measured[source_key][sequence] = _finite(row["measured_direction_normalized_delta"], "measured outcome")
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_source[str(row["source_key"])].append(row)
    per_source: dict[str, dict[str, Any]] = {}
    for source_key in sorted(sources):
        pool = measured.get(source_key, {})
        _require(pool, f"measured neighborhood missing: {source_key}")
        generated: dict[str, tuple[dict[str, Any], float, str]] = {}
        for row in by_source[source_key]:
            sequence = str(row["candidate_sequence"]).upper().replace("T", "U")
            if row.get("critic_score") is not None:
                score_field = "critic_score"
            else:
                _require(row.get("generation_score") is not None, f"candidate has no ranking score: {source_key}")
                score_field = "generation_score"
            score = _finite(row[score_field], score_field)
            previous = generated.get(sequence)
            if previous is None or score > previous[1]:
                generated[sequence] = (row, score, score_field)
        score_fields = {value[2] for value in generated.values()}
        _require(len(score_fields) == 1, f"ranking score field differs within source: {source_key}")
        hits = [sequence for sequence in generated if sequence in pool]
        unmeasured = [sequence for sequence in generated if sequence not in pool]
        if candidate_support_mode == "CLOSED_MEASURED_SUPPORT":
            _require(
                not unmeasured,
                f"closed measured support contains an unmeasured generated candidate: {source_key}",
            )
        true_order = sorted(pool, key=lambda sequence: (-pool[sequence], sequence))
        generated_sequences = sorted(generated)
        generated_scores = np.asarray([generated[sequence][1] for sequence in generated_sequences], dtype=float)
        top_k = min(k, len(pool))
        shifted = {sequence: pool[sequence] - min(pool.values()) for sequence in pool}
        true_scores = np.asarray([pool[sequence] for sequence in true_order], dtype=float)
        true_gains = np.asarray([shifted[sequence] for sequence in true_order], dtype=float)
        if candidate_support_mode == "CLOSED_MEASURED_SUPPORT":
            closed_gains = np.asarray([shifted[sequence] for sequence in generated_sequences], dtype=float)
            closed_dcg = _tie_aware_dcg(closed_gains, generated_scores, top_k)
            closed_idcg = _tie_aware_dcg(true_gains, true_scores, top_k)
            closed_ndcg = None if closed_idcg == 0.0 else closed_dcg / closed_idcg
        else:
            closed_ndcg = None
        recovered_sequences = sorted(hits)
        recovered_k = min(k, len(recovered_sequences))
        if recovered_k:
            recovered_scores = np.asarray(
                [generated[sequence][1] for sequence in recovered_sequences], dtype=float
            )
            recovered_gains = np.asarray(
                [shifted[sequence] for sequence in recovered_sequences], dtype=float
            )
            recovered_dcg = _tie_aware_dcg(recovered_gains, recovered_scores, recovered_k)
            recovered_idcg = _tie_aware_dcg(true_gains, true_scores, recovered_k)
            recovered_ndcg = None if recovered_idcg == 0.0 else recovered_dcg / recovered_idcg
        else:
            recovered_ndcg = None
        true_cutoff = true_scores[top_k - 1]
        true_top_eligible = {
            sequence for sequence, value in pool.items() if value >= true_cutoff
        }
        generated_inclusion = _top_k_inclusion_probabilities(generated_scores, top_k)
        top_tied_sequences = [
            sequence for sequence, score in zip(generated_sequences, generated_scores)
            if score == np.max(generated_scores)
        ]
        selected_value = (
            float(np.mean([pool[sequence] for sequence in top_tied_sequences]))
            if all(sequence in pool for sequence in top_tied_sequences)
            else None
        )
        if selected_value is not None:
            span = max(pool.values()) - min(pool.values())
            regret = 0.0 if span == 0.0 else (max(pool.values()) - selected_value) / span
        else:
            regret = None
        per_source[source_key] = {
            "measured_candidate_count": len(pool),
            "recovered_candidate_count": len(hits),
            "unmeasured_generated_candidate_count": len(unmeasured),
            "all_generated_candidates_measured": not unmeasured,
            "candidate_recovery_rate": len(hits) / len(pool),
            "measured_top_k_recovery_at_k": float(sum(
                probability
                for sequence, probability in zip(generated_sequences, generated_inclusion)
                if sequence in true_top_eligible
            ) / top_k),
            "closed_measured_ndcg_at_k": closed_ndcg,
            "recovered_measured_ndcg_at_k": recovered_ndcg,
            "closed_measured_ndcg_status": (
                "DEFINED_CLOSED_MEASURED_SUPPORT"
                if closed_ndcg is not None
                else (
                    "UNDEFINED_ZERO_MEASURED_GAIN"
                    if candidate_support_mode == "CLOSED_MEASURED_SUPPORT"
                    else "UNDEFINED_OPEN_SUPPORT_HAS_UNKNOWN_OUTCOMES"
                )
            ),
            "normalized_regret": regret,
            "selected_measured_outcome": selected_value,
            "ranking_score_field": next(iter(score_fields)),
        }
    closed_ndcg_values = [row["closed_measured_ndcg_at_k"] for row in per_source.values()]
    recovered_ndcg_values = [row["recovered_measured_ndcg_at_k"] for row in per_source.values()]
    regret_values = [row["normalized_regret"] for row in per_source.values()]
    defined_closed_ndcg = [float(value) for value in closed_ndcg_values if value is not None]
    defined_recovered_ndcg = [float(value) for value in recovered_ndcg_values if value is not None]
    defined_regret = [float(value) for value in regret_values if value is not None]
    return {
        "candidate_support_mode": candidate_support_mode,
        "unknown_generated_candidates_are_zero_gain": False,
        "source_count": len(per_source),
        "source_macro_candidate_recovery_rate": _mean(row["candidate_recovery_rate"] for row in per_source.values()),
        "source_macro_measured_top_k_recovery_at_k": _mean(
            row["measured_top_k_recovery_at_k"] for row in per_source.values()
        ),
        "source_closed_measured_ndcg_defined_count": len(defined_closed_ndcg),
        "source_macro_closed_measured_ndcg_at_k": (
            None if not defined_closed_ndcg else float(np.mean(defined_closed_ndcg))
        ),
        "source_recovered_measured_ndcg_defined_count": len(defined_recovered_ndcg),
        "source_macro_recovered_measured_ndcg_at_k": (
            None if not defined_recovered_ndcg else float(np.mean(defined_recovered_ndcg))
        ),
        "source_normalized_regret_defined_count": len(defined_regret),
        "source_macro_normalized_regret": (
            float(np.mean(defined_regret))
            if len(defined_regret) == len(per_source)
            else None
        ),
        "per_source": per_source,
    }


def validate_measured_pool(rows: list[dict[str, Any]], pool: str, release_state: str) -> None:
    _require(pool in {"DEVELOPMENT", "EVALUATION"}, "measured-neighborhood pool must be explicit")
    _require(all(row.get("pool_assignment") == pool for row in rows), "measured row pool differs from declared pool")
    if pool == "DEVELOPMENT":
        _require(release_state == "CLOSED", "Development measured selection must not open Evaluation")
    else:
        _require(
            release_state == "PREDICTOR_GENERATOR_AND_BASELINES_FROZEN",
            "measured Evaluation outcomes remain closed until predictor, generator, and baselines are frozen",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--measured-neighborhood", type=Path)
    parser.add_argument("--measured-neighborhood-pool", choices=("DEVELOPMENT", "EVALUATION"))
    parser.add_argument("--candidate-support-mode", choices=tuple(sorted(CANDIDATE_SUPPORT_MODES)))
    parser.add_argument("--evaluation-release-state", default="CLOSED")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = load_source_manifest(args.source_manifest)
    candidate_rows = _read_jsonl(args.candidates)
    result = {
        "schema_version": "route_a_v3_route2_generation_evaluation.v2",
        "generation": evaluate_generation(sources, candidate_rows),
        "evaluation_release_state": args.evaluation_release_state,
    }
    if args.measured_neighborhood:
        _require(args.candidate_support_mode is not None, "candidate support mode is required with measured neighborhood")
        measured_rows = _read_jsonl(args.measured_neighborhood)
        validate_measured_pool(measured_rows, args.measured_neighborhood_pool, args.evaluation_release_state)
        result["measured_neighborhood"] = measured_neighborhood_metrics(
            sources,
            candidate_rows,
            measured_rows,
            k=args.k,
            candidate_support_mode=args.candidate_support_mode,
        )
        result["measured_neighborhood_pool"] = args.measured_neighborhood_pool
    _require(not args.output.exists(), f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
