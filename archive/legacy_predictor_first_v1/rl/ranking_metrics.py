"""Deterministic offline metrics for proposal-ranker validation.

These metrics evaluate only teacher-labelled, legal one-step candidates.  They
are ranking diagnostics, not measurements of translation efficiency or other
experimental properties.  A global regret is reported only when the caller
declares ``candidate_cap=0``; capped pools are explicitly named restricted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Mapping, Sequence


@dataclass(frozen=True)
class RankingCandidate:
    """One validation candidate with teacher delta and model log score."""

    op: str
    teacher_delta: float
    model_score: float


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _dcg(items: Sequence[RankingCandidate], k: int) -> float:
    return sum(
        max(0.0, float(item.teacher_delta)) / math.log2(rank + 2.0)
        for rank, item in enumerate(items[:k])
    )


def _ndcg(items: Sequence[RankingCandidate], k: int) -> float:
    ideal = sorted(items, key=lambda item: -float(item.teacher_delta))
    denom = _dcg(ideal, k)
    return _dcg(items, k) / denom if denom > 0.0 else 0.0


def _prefix(candidate_cap: int) -> tuple[str, str]:
    if int(candidate_cap) > 0:
        return "restricted_", "restricted"
    return "", "global"


def compute_ranking_metrics(
    candidates_by_transcript: Mapping[str, Sequence[RankingCandidate]],
    *,
    candidate_cap: int = 0,
    ks: Sequence[int] = (1, 8, 32),
) -> dict[str, object]:
    """Compute validation metrics from model-scored teacher candidates.

    ``candidate_cap`` truncates each group to model top-k before evaluation.
    This is intentionally a *restricted* protocol: it cannot be called global
    regret because a teacher-better candidate may have been removed first.
    """
    cap = max(0, int(candidate_cap))
    clean_ks = tuple(sorted({max(1, int(k)) for k in ks}))
    prefix, scope = _prefix(cap)
    regrets: list[float] = []
    selected_deltas: list[float] = []
    recall: dict[int, list[float]] = {k: [] for k in clean_ks}
    ndcg: dict[int, list[float]] = {k: [] for k in clean_ks if k >= 8}
    precision: dict[int, list[float]] = {k: [] for k in clean_ks}
    stop_correct: list[float] = []
    evaluated = 0

    for transcript_id in sorted(candidates_by_transcript):
        raw = [
            candidate for candidate in candidates_by_transcript[transcript_id]
            if math.isfinite(float(candidate.teacher_delta))
            and math.isfinite(float(candidate.model_score))
        ]
        if not raw:
            continue
        ranked = sorted(enumerate(raw), key=lambda row: (-float(row[1].model_score), row[0]))
        ranked_items = [row[1] for row in ranked]
        if cap:
            ranked_items = ranked_items[:cap]
        if not ranked_items:
            continue
        evaluated += 1
        teacher_best = max(float(item.teacher_delta) for item in ranked_items)
        selected_deltas.append(float(ranked_items[0].teacher_delta))
        regrets.append(teacher_best - float(ranked_items[0].teacher_delta))
        for k in clean_ks:
            top = ranked_items[:k]
            recall[k].append(
                1.0 if any(float(item.teacher_delta) == teacher_best for item in top) else 0.0
            )
            precision[k].append(_mean([1.0 if float(item.teacher_delta) > 0.0 else 0.0 for item in top]))
        for k in ndcg:
            ndcg[k].append(_ndcg(ranked_items, k))

        non_stop = [item for item in ranked_items if str(item.op).lower() != "stop"]
        if not non_stop or max(float(item.teacher_delta) for item in non_stop) <= 0.0:
            stop_correct.append(1.0 if str(ranked_items[0].op).lower() == "stop" else 0.0)

    result: dict[str, object] = {
        "candidate_pool_scope": scope,
        "candidate_cap": cap,
        "n_records": evaluated,
        f"{prefix}mean_model_regret": _mean(regrets),
        f"{prefix}median_model_regret": float(median(regrets)) if regrets else 0.0,
        f"{prefix}mean_selected_teacher_delta": _mean(selected_deltas),
        f"{prefix}stop_accuracy": _mean(stop_correct),
        "stop_accuracy_denominator": len(stop_correct),
    }
    for k in clean_ks:
        result[f"{prefix}oracle_best_recall_at_{k}"] = _mean(recall[k])
        result[f"{prefix}positive_edit_precision_at_{k}"] = _mean(precision[k])
    for k, values in ndcg.items():
        result[f"{prefix}ndcg_at_{k}"] = _mean(values)
    return result


__all__ = ["RankingCandidate", "compute_ranking_metrics"]
