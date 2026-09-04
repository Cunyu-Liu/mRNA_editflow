"""Pair-mean labels, per-task rank transform, and extended eval metrics for Critic V6.

D1 (2026-08-31 frozen): shared effect as primary supervision.  The MPRAU
task (ENCSR854RUF) measures every allelic variant in exactly six cell lines,
and the label signal-to-noise ceiling analysis (split-half rho 0.683) shows
the six-cell pair mean is a much cleaner supervision target than any single
cell measurement.  This module provides, as pure CPU-testable functions:

1. ``pair_mean_target_map_v6`` — replace each record's target with its
   six-cell pair mean (row count unchanged; MPRAU VALIDATION labels collapse
   from 12,048 to 2,008 distinct values).
2. ``per_task_rank_gaussian_v6`` — per-task monotone rank/Gaussian transform
   that flattens the long effect-size tail (Spearman invariant, MAE reported
   in two currencies).
3. ``extended_validation_metrics_v6`` — W1-e helpers: within-source rho,
   pair-mean rho + ceiling ratio (all-task pooled values under
   ``*_pooled_legacy`` keys plus per-task columns ``*_by_task``; 2026-09-04
   Task 3.1), hit@K / NDCG@K over the measured pair neighborhood, and Tier-B
   task marking.

Every switch defaults OFF so the V5 executor stays bit-identical; a run only
adopts a transform when its config field is explicitly true.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Mapping, Sequence

from core.route2_xeditcritic_training_data_v3 import XEditCriticRecordV3

from scipy.stats import norm, spearmanr


class XEditCriticPairMeanV6Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticPairMeanV6Error(message)


def pair_key_v6(record: XEditCriticRecordV3) -> tuple[str, str, str, str]:
    """Deterministic pair identity for the shared-effect supervision target.

    A measured pair is the (task, study, source, candidate) bundle: MPRAU
    re-measures the identical allele under six biological contexts, so all six
    rows share one pair key and must collapse to one mean label.  The pair key
    deliberately excludes context (cell line), which is the *offset* axis.
    """

    return (record.task, record.study, record.source, record.candidate)


MPRAU_TASK_ID = "MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE::region=1"


def pair_mean_target_map_v6(
    records: Sequence[XEditCriticRecordV3],
    *,
    pair_tasks: set[str] | None = None,
) -> dict[str, float]:
    """Map record_id -> six-cell pair-mean target for the enabled pair tasks.

    ``pair_tasks`` defaults to the frozen MPRAU task id; rows outside the
    enabled set keep their own target (identity mapping).  Every record in an
    enabled task MUST have its full measured pair present in ``records``,
    otherwise the map is rejected as ambiguous.  Using only one split at a time
    guarantees the six-cell pair is complete within the split (source-group
    splits are measured to completeness by design).
    """

    enabled = (
        set(pair_tasks)
        if pair_tasks is not None
        else {MPRAU_TASK_ID}
    )
    buckets: dict[tuple[str, str, str, str], list[tuple[str, float]]] = {}
    for record in records:
        if record.task not in enabled:
            continue
        buckets.setdefault(pair_key_v6(record), []).append(
            (record.record_id, float(record.target))
        )
    out: dict[str, float] = {}
    seen_pair_sizes: set[int] = set()
    for key, members in buckets.items():
        _require(
            len(members) >= 2,
            f"pair task record is not a measured pair: {key}",
        )
        seen_pair_sizes.add(len(members))
        mean = sum(value for _, value in members) / len(members)
        out.update({record_id: mean for record_id, _ in members})
    _require(
        len(seen_pair_sizes) <= 1,
        "pair sizes are not uniform within one split",
    )
    for record in records:
        if record.task not in enabled:
            out[record.record_id] = float(record.target)
    _require(
        set(out) == {record.record_id for record in records},
        "pair-mean map does not cover every record",
    )
    return out


def apply_pair_mean_targets_v6(
    records: Sequence[XEditCriticRecordV3],
    *,
    pair_tasks: set[str] | None = None,
) -> tuple[list[XEditCriticRecordV3], dict[str, float]]:
    """Return pair-mean relabelled records plus the map (pure; no mutation)."""

    target_map = pair_mean_target_map_v6(records, pair_tasks=pair_tasks)
    relabelled = [
        (
            replace(record, target=target_map[record.record_id])
            if record.record_id in target_map
            else record
        )
        for record in records
    ]
    return relabelled, target_map


def per_task_rank_gaussian_v6(values: Sequence[float]) -> list[float]:
    """Mid-rank to standard-normal (Gaussian) transform of one task's values.

    Deterministic; tied values share their mid-rank and therefore map to one
    Gaussian value.  Monotone, so Spearman is invariant to it.  Used in W1-c.
    """

    _require(bool(values), "rank transform received no values")
    ordered = sorted((float(value), index) for index, value in enumerate(values))
    zs = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        anchor, _ = ordered[cursor]
        end = cursor
        while end + 1 < len(ordered) and ordered[end + 1][0] == anchor:
            end += 1
        mid_rank = (cursor + end) / 2.0 + 1.0
        z = float(norm.ppf((mid_rank - 0.5) / len(values)))
        for group_index in range(cursor, end + 1):
            zs[ordered[group_index][1]] = z
        cursor = end + 1
    return zs


def apply_rank_gaussian_targets_v6(
    records: Sequence[XEditCriticRecordV3],
    *,
    rank_tasks: set[str] | None = None,
) -> tuple[list[XEditCriticRecordV3], dict[str, object]]:
    """Per-task rank-Gaussian transform of record targets (W1-c, default off).

    Applied task-wise over the given split only; returns relabelled records
    and a small metadata map (transform is stored implicitly by task order).
    """

    enabled = (
        set(rank_tasks)
        if rank_tasks is not None
        else {record.task for record in records}
    )
    by_task: dict[str, list[float]] = {}
    for record in records:
        if record.task in enabled:
            by_task.setdefault(record.task, []).append(float(record.target))
    transformed: dict[str, float] = {}
    for task in sorted(by_task):
        zs = per_task_rank_gaussian_v6(by_task[task])
        slider = 0
        for record in records:
            if record.task == task:
                transformed[record.record_id] = zs[slider]
                slider += 1
        _require(slider == len(zs), "rank transform lost a record")
    relabelled = [
        (
            replace(record, target=transformed[record.record_id])
            if record.record_id in transformed
            else record
        )
        for record in records
    ]
    return relabelled, {"task_count": len(enabled)}


def extended_validation_metrics_v6(
    targets: Sequence[float],
    predictions: Sequence[float],
    tasks: Sequence[str],
    source_groups: Sequence[str],
    pair_keys: Sequence[tuple[str, str, str, str]],
    *,
    ceiling_by_task: Mapping[str, float] | None = None,
    hit_at_k: Sequence[int] = (1, 3, 5),
    ndcg_at_k: Sequence[int] = (1, 3, 5),
) -> dict[str, object]:
    """W1-e extension: within-source rho, pair-mean rho + ceiling ratio, hit@K, NDCG@K.

    Pair-mean metrics are emitted in two currencies: all-task pooled values
    under ``*_pooled_legacy`` keys (kept for continuity; NOT any task's
    criterion — see the 2026-09-03 caliber-trap incident) and per-task
    columns ``pair_mean_{spearman,pair_count,ceiling,ceiling_ratio}_by_task``.
    The per-task caliber matches the adjudication scripts: pair/variant
    groups (pair key excludes context) with >= 2 measured contexts,
    group means of target and prediction, Spearman across group means
    within each task.

    All metric families are computed from pure record-level bundles; no TEST or
    Evaluation outcome is read.
    """

    _require(
        len(targets)
        == len(predictions)
        == len(tasks)
        == len(source_groups)
        == len(pair_keys),
        "extended metric bundles are misaligned",
    )

    def _safe_rho(left: Sequence[float], right: Sequence[float]) -> float | None:
        if len(left) < 3:
            return None
        uni = len({round(float(value), 12) for value in left})
        unp = len({round(float(value), 12) for value in right})
        if uni <= 1 or unp <= 1:
            return None
        return float(spearmanr(left, right).statistic)

    # Within-source-group rho: pairs that share the identical source group.
    group_buckets: dict[str, list[tuple[float, float]]] = {}
    for target, prediction, group in zip(targets, predictions, source_groups):
        group_buckets.setdefault(str(group), []).append(
            (float(target), float(prediction))
        )
    eligible = [members for members in group_buckets.values() if len(members) >= 3]
    within_rho = None
    if eligible:
        flat_targets = [t for members in eligible for t, _ in members]
        flat_predictions = [p for members in eligible for _, p in members]
        within_rho = _safe_rho(flat_targets, flat_predictions)

    # Pair-mean rho + ceiling ratio (all-task pooled) and per-task columns.
    # 2026-09-04 Task 3.1 (SPECS_CRITIC_V6): the pooled pair-mean spearman
    # (e.g. 2,660 pooled pairs = MPRAU 2,008 variants + polyA 321 + ~331
    # others) is NOT any task's criterion and was once misquoted as the MPRAU
    # caliber (2026-09-03 caliber-trap incident).  Pooled fields keep a
    # _pooled_legacy suffix so stale readers fail loudly (KeyError) instead
    # of silently quoting a pooled number; pair_mean_spearman_by_task is the
    # supported read.  Per-task caliber matches the adjudication scripts:
    # pair/variant groups (pair key excludes context) with >= 2 measured
    # contexts, group means of target and prediction, Spearman across group
    # means within each task.
    task_groups: dict[str, dict[tuple[str, str, str, str], list[tuple[float, float]]]] = {}
    for target, prediction, task, key in zip(targets, predictions, tasks, pair_keys):
        task_groups.setdefault(str(task), {}).setdefault(key, []).append(
            (float(target), float(prediction))
        )

    def _pair_mean_rows(
        groups: Mapping[tuple[str, str, str, str], list[tuple[float, float]]],
    ) -> list[tuple[float, float]]:
        return [
            (
                sum(t for t, _ in members) / len(members),
                sum(p for _, p in members) / len(members),
            )
            for members in groups.values()
            if len(members) >= 2
        ]

    pair_rows = [
        row
        for groups in task_groups.values()
        for row in _pair_mean_rows(groups)
    ]
    pair_rho = _safe_rho([t for t, _ in pair_rows], [p for _, p in pair_rows])
    pair_mean_spearman_by_task: dict[str, float | None] = {}
    pair_mean_pair_count_by_task: dict[str, int] = {}
    for task in sorted(task_groups):
        rows = _pair_mean_rows(task_groups[task])
        pair_mean_pair_count_by_task[task] = len(rows)
        pair_mean_spearman_by_task[task] = _safe_rho(
            [t for t, _ in rows], [p for _, p in rows]
        )
    ceiling_ratio = None
    tier_b_tasks: set[str] = set()
    if pair_rho is not None and ceiling_by_task:
        ratios = []
        for task in ceiling_by_task:
            ceiling = float(ceiling_by_task[task])
            if 0.0 < ceiling < 1.0:
                ratios.append(pair_rho / ceiling)
                tier_b_tasks.add(task)
        if ratios:
            ceiling_ratio = float(sum(ratios) / len(ratios))
    # Task 3.2 (same site): per-task ceiling echo + completion ratio
    # (rho_task / ceiling_task).  Ceilings arrive via config
    # training.ceiling_by_task; tasks without a registered ceiling read null.
    pair_mean_ceiling_by_task: dict[str, float | None] = {}
    pair_mean_ceiling_ratio_by_task: dict[str, float | None] = {}
    for task, rho in pair_mean_spearman_by_task.items():
        ceiling = None
        if ceiling_by_task is not None and task in ceiling_by_task:
            ceiling = float(ceiling_by_task[task])
        pair_mean_ceiling_by_task[task] = ceiling
        pair_mean_ceiling_ratio_by_task[task] = (
            rho / ceiling
            if ceiling is not None and ceiling > 0.0 and rho is not None
            else None
        )

    # hit@K / NDCG@K over the measured pair neighborhood.
    per_task_hits: dict[str, dict[str, float]] = {}
    per_task_ndcg: dict[str, dict[str, float]] = {}
    for task, groups in task_groups.items():
        valid_groups = [members for members in groups.values() if len(members) >= 2]
        for k in set(hit_at_k) | set(ndcg_at_k):
            if k <= 1 or any(len(members) < k for members in valid_groups):
                continue
            hit_values = []
            ndcg_values = []
            for members in valid_groups:
                by_target = sorted(members, key=lambda item: item[0], reverse=True)
                by_prediction = sorted(members, key=lambda item: item[1], reverse=True)
                cut_pred = [t for t, _ in by_prediction[:k]]
                hit_values.append(
                    1.0 if any(t == by_target[0][0] for t in cut_pred) else 0.0
                )
                dcg = sum(
                    by_prediction[i][1] / math.log2(i + 2) for i in range(k)
                )
                idcg = sum(
                    by_target[i][0] / math.log2(i + 2) for i in range(k)
                )
                ndcg_values.append(dcg / idcg if idcg > 0 else 0.0)
            if hit_values and k in set(hit_at_k):
                per_task_hits.setdefault(task, {})[f"hit@{k}"] = float(
                    sum(hit_values) / len(hit_values)
                )
            if ndcg_values and k in set(ndcg_at_k):
                per_task_ndcg.setdefault(task, {})[f"ndcg@{k}"] = float(
                    sum(ndcg_values) / len(ndcg_values)
                )

    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v6_extended_metrics.v2",
        "within_source_spearman": within_rho,
        "pair_mean_spearman_pooled_legacy": pair_rho,
        "pair_mean_ceiling_ratio_pooled_legacy": ceiling_ratio,
        "pair_mean_pair_count_pooled_legacy": len(pair_rows),
        "pair_mean_spearman_by_task": pair_mean_spearman_by_task,
        "pair_mean_pair_count_by_task": pair_mean_pair_count_by_task,
        "pair_mean_ceiling_by_task": pair_mean_ceiling_by_task,
        "pair_mean_ceiling_ratio_by_task": pair_mean_ceiling_ratio_by_task,
        "hit_at_k_by_task": per_task_hits,
        "ndcg_at_k_by_task": per_task_ndcg,
        "tier_b_tasks": sorted(tier_b_tasks),
    }