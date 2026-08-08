"""Cluster-aware inferential utilities for R3X ordinary evaluation.

The publication-facing paired comparison is deliberately narrower than the
row-micro diagnostics retained at the bottom of this module.  Its primary
utility is the same sign estimand used by the effect gate: non-zero rows are
averaged within global source-relation components, components are averaged
within frozen study/estimand/context/edit-count cells, and cells are averaged
without row-count weighting.  Resampling and prediction swaps always operate
on a whole global component across every cell in which it occurs.
"""
from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from .metrics import MetricProtocolError, _aligned, _finite_vector


PRIMARY_PAIRED_METRIC_ID = "MACRO_COMPONENT_CELL_BALANCED_SIGN_ACCURACY"


def _primary_cell_id(row: Mapping) -> str:
    values = (
        row.get("study"),
        row.get("estimand_id") or row.get("endpoint_id") or row.get("endpoint"),
        row.get("context_id_v2") or row.get("publication_context_id"),
        row.get("actual_edit_count") or row.get("edit_count"),
    )
    if any(value is None or str(value) == "" for value in values):
        raise MetricProtocolError(
            "paired primary rows require study, estimand, publication context, and edit count"
        )
    return "|".join(str(value) for value in values)


def _stable_row_id(row: Mapping) -> str:
    value = str(row.get("r3_input_row_sha256") or row.get("record_id") or "")
    if not value:
        raise MetricProtocolError("paired primary rows require a stable row identity")
    return value


def _paired_sign_scores(
    rows: Sequence[Mapping],
    candidate_prediction: Sequence[float],
    baseline_prediction: Sequence[float],
) -> dict:
    """Return frozen cell/component sign scores for both paired methods."""

    rows = list(rows)
    candidate = _finite_vector(candidate_prediction, "candidate_prediction")
    baseline = _finite_vector(baseline_prediction, "baseline_prediction")
    if not rows or candidate.shape != (len(rows),) or baseline.shape != (len(rows),):
        raise MetricProtocolError("paired predictions must align one-to-one with non-empty rows")
    row_ids = [_stable_row_id(row) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise MetricProtocolError("paired primary rows contain duplicate identities")

    observations: dict[str, dict[str, list[tuple[bool, bool]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    registered_cells = set()
    for index, row in enumerate(rows):
        cell = _primary_cell_id(row)
        registered_cells.add(cell)
        component = str(row.get("source_component_id") or "")
        if not component:
            raise MetricProtocolError("paired primary rows require source_component_id")
        truth = row.get("delta")
        if isinstance(truth, bool) or not isinstance(truth, (int, float)) or not math.isfinite(float(truth)):
            raise MetricProtocolError("paired primary rows require finite delta")
        truth = float(truth)
        if truth == 0.0:
            continue
        expected_positive = truth > 0.0
        observations[cell][component].append(
            (bool(candidate[index] >= 0.0) == expected_positive,
             bool(baseline[index] >= 0.0) == expected_positive)
        )

    cells: dict[str, dict] = {}
    for cell in sorted(registered_cells):
        component_rows = observations.get(cell, {})
        if not component_rows:
            raise MetricProtocolError(
                f"primary sign estimand is undefined in registered cell {cell!r}"
            )
        component_scores = {}
        for component, values in sorted(component_rows.items()):
            candidate_score = float(np.mean([value[0] for value in values]))
            baseline_score = float(np.mean([value[1] for value in values]))
            component_scores[component] = {
                "candidate": candidate_score,
                "baseline": baseline_score,
                "improvement": candidate_score - baseline_score,
                "n_nonzero_rows": len(values),
            }
        cells[cell] = {
            "components": component_scores,
            "n_components": len(component_scores),
            "candidate": float(np.mean([
                value["candidate"] for value in component_scores.values()
            ])),
            "baseline": float(np.mean([
                value["baseline"] for value in component_scores.values()
            ])),
        }
        cells[cell]["improvement"] = cells[cell]["candidate"] - cells[cell]["baseline"]

    return {
        "metric_id": PRIMARY_PAIRED_METRIC_ID,
        "cell_definition": "STUDY_ESTIMAND_PUBLICATION_CONTEXT_EDIT_COUNT",
        "zero_policy": "EXCLUDE_TRUE_ZERO",
        "row_within_component_weighting": "EQUAL_NONZERO_ROW",
        "component_within_cell_weighting": "EQUAL_COMPONENT",
        "cell_weighting": "EQUAL_REGISTERED_CELL",
        "sign_decision_rule": "RAW_PREDICTED_DELTA_MEAN_GE_ZERO",
        "cells": cells,
        "candidate": float(np.mean([value["candidate"] for value in cells.values()])),
        "baseline": float(np.mean([value["baseline"] for value in cells.values()])),
        "improvement": float(np.mean([value["improvement"] for value in cells.values()])),
    }


def paired_primary_sign_bootstrap(
    rows: Sequence[Mapping],
    candidate_prediction: Sequence[float],
    baseline_prediction: Sequence[float],
    *,
    n_resamples: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
    min_components_per_cell: int = 10,
    max_attempt_multiplier: int = 100,
) -> dict:
    """Paired cluster bootstrap for the frozen primary sign utility.

    A sampled global component is repeated in every cell where it occurs.  A
    draw that omits all supported components from any frozen cell is invalid;
    it is never allowed to silently change the macro-cell denominator.
    """

    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise MetricProtocolError("n_resamples must be a positive integer")
    if not 0.0 < float(confidence) < 1.0:
        raise MetricProtocolError("confidence must be in (0,1)")
    if (isinstance(min_components_per_cell, bool)
            or not isinstance(min_components_per_cell, int)
            or min_components_per_cell < 2):
        raise MetricProtocolError("min_components_per_cell must be an integer >=2")
    if (isinstance(max_attempt_multiplier, bool)
            or not isinstance(max_attempt_multiplier, int)
            or max_attempt_multiplier < 1):
        raise MetricProtocolError("max_attempt_multiplier must be a positive integer")

    score = _paired_sign_scores(rows, candidate_prediction, baseline_prediction)
    support = {
        cell: int(value["n_components"]) for cell, value in score["cells"].items()
    }
    underpowered = sorted(
        cell for cell, count in support.items() if count < min_components_per_cell
    )
    if underpowered:
        raise MetricProtocolError(
            "paired primary bootstrap has fewer than the frozen minimum components "
            f"in cells: {underpowered}"
        )
    levels = sorted({
        component
        for cell in score["cells"].values()
        for component in cell["components"]
    })
    rng = np.random.default_rng(int(seed))
    values: list[float] = []
    attempts = 0
    max_attempts = int(n_resamples) * int(max_attempt_multiplier)
    while len(values) < int(n_resamples) and attempts < max_attempts:
        attempts += 1
        sampled = rng.choice(levels, size=len(levels), replace=True).tolist()
        cell_values = []
        for cell in score["cells"].values():
            component_scores = cell["components"]
            selected = [
                component_scores[component]["improvement"]
                for component in sampled if component in component_scores
            ]
            if not selected:
                break
            cell_values.append(float(np.mean(selected)))
        if len(cell_values) == len(score["cells"]):
            values.append(float(np.mean(cell_values)))
    if len(values) != int(n_resamples):
        raise MetricProtocolError(
            "paired primary bootstrap could not obtain the frozen number of complete-cell resamples"
        )
    alpha = (1.0 - float(confidence)) / 2.0
    array = np.asarray(values, dtype=np.float64)
    return {
        "method": "PAIRED_GLOBAL_SOURCE_COMPONENT_BOOTSTRAP",
        "metric_id": PRIMARY_PAIRED_METRIC_ID,
        "observed_oriented_improvement": score["improvement"],
        "confidence": float(confidence),
        "confidence_interval_oriented_improvement": [
            float(np.quantile(array, alpha)),
            float(np.quantile(array, 1.0 - alpha)),
        ],
        "n_global_components": len(levels),
        "components_per_cell": support,
        "min_components_per_cell": int(min_components_per_cell),
        "n_resamples": int(n_resamples),
        "n_attempted_resamples": int(attempts),
        "n_invalid_incomplete_cell_resamples": int(attempts - len(values)),
        "seed": int(seed),
        "resampling_unit": "GLOBAL_SOURCE_RELATION_COMPONENT",
        "registered_cell_denominator_fixed": True,
    }


def paired_primary_sign_permutation(
    rows: Sequence[Mapping],
    candidate_prediction: Sequence[float],
    baseline_prediction: Sequence[float],
    *,
    n_permutations: int = 10000,
    seed: int = 42,
    exact_component_limit: int = 20,
    min_components_per_cell: int = 10,
) -> dict:
    """Whole-component paired prediction swap test for the primary utility."""

    if isinstance(n_permutations, bool) or not isinstance(n_permutations, int) or n_permutations < 1:
        raise MetricProtocolError("n_permutations must be a positive integer")
    if exact_component_limit != 20:
        raise MetricProtocolError("exact_component_limit is frozen at 20")
    score = _paired_sign_scores(rows, candidate_prediction, baseline_prediction)
    support = {
        cell: int(value["n_components"]) for cell, value in score["cells"].items()
    }
    underpowered = sorted(
        cell for cell, count in support.items() if count < int(min_components_per_cell)
    )
    if underpowered:
        raise MetricProtocolError(
            "paired primary permutation has fewer than the frozen minimum components "
            f"in cells: {underpowered}"
        )
    levels = sorted({
        component
        for cell in score["cells"].values()
        for component in cell["components"]
    })
    observed = float(score["improvement"])
    if len(levels) <= int(exact_component_limit):
        patterns = itertools.product((-1.0, 1.0), repeat=len(levels))
        mode = "EXACT"
    else:
        if int(n_permutations) < 10000:
            raise MetricProtocolError(
                "Monte Carlo primary permutation requires at least 10000 frozen draws"
            )
        rng = np.random.default_rng(int(seed))
        patterns = (
            rng.choice((-1.0, 1.0), size=len(levels), replace=True)
            for _ in range(int(n_permutations))
        )
        mode = "MONTE_CARLO"
    component_index = {component: index for index, component in enumerate(levels)}
    extreme = 0
    total = 0
    for pattern in patterns:
        signs = np.asarray(pattern, dtype=np.float64)
        cell_values = []
        for cell in score["cells"].values():
            differences = [
                value["improvement"] * signs[component_index[component]]
                for component, value in cell["components"].items()
            ]
            cell_values.append(float(np.mean(differences)))
        permuted = float(np.mean(cell_values))
        extreme += int(abs(permuted) >= abs(observed) - 1e-15)
        total += 1
    p_value = (
        extreme / total if mode == "EXACT"
        else (extreme + 1.0) / (total + 1.0)
    )
    return {
        "method": "PAIRED_GLOBAL_SOURCE_COMPONENT_PREDICTION_SWAP",
        "mode": mode,
        "metric_id": PRIMARY_PAIRED_METRIC_ID,
        "observed_oriented_improvement": observed,
        "p_value_two_sided": float(min(1.0, p_value)),
        "n_global_components": len(levels),
        "components_per_cell": support,
        "min_components_per_cell": int(min_components_per_cell),
        "n_permutations": int(total),
        "seed": int(seed),
        "permutation_unit": "GLOBAL_SOURCE_RELATION_COMPONENT",
        "component_swap_is_synchronized_across_cells": True,
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return Holm family-wise adjusted p-values keyed independent of input order."""

    if not p_values:
        raise MetricProtocolError("Holm correction refuses an empty family")
    checked: list[tuple[str, float]] = []
    for key, value in p_values.items():
        p = float(value)
        if not np.isfinite(p) or not 0.0 <= p <= 1.0:
            raise MetricProtocolError(f"invalid p-value for {key!r}: {value!r}")
        checked.append((str(key), p))
    checked.sort(key=lambda item: (item[1], item[0]))
    m = len(checked)
    running = 0.0
    adjusted: dict[str, float] = {}
    for rank, (key, p) in enumerate(checked):
        running = max(running, (m - rank) * p)
        adjusted[key] = float(min(1.0, running))
    return {key: adjusted[key] for key in sorted(adjusted)}


def _metric_value(y: np.ndarray, pred: np.ndarray, metric: str) -> float:
    if metric == "sign_accuracy":
        nz = y != 0.0
        if not np.any(nz):
            raise MetricProtocolError("sign comparison has no non-zero true effects")
        true_sign = np.where(y[nz] > 0.0, 1.0, -1.0)
        predicted_sign = np.where(pred[nz] >= 0.0, 1.0, -1.0)
        return float(np.mean(true_sign == predicted_sign))
    if metric == "mae":
        # All comparison utilities are oriented so that larger is better.
        return -float(np.mean(np.abs(pred - y)))
    if metric == "rmse":
        return -float(np.sqrt(np.mean((pred - y) ** 2)))
    if metric == "spearman":
        if y.size < 3 or np.ptp(y) == 0.0 or np.ptp(pred) == 0.0:
            raise MetricProtocolError("Spearman comparison is undefined")
        value, _ = spearmanr(y, pred)
        if not np.isfinite(value):
            raise MetricProtocolError("Spearman comparison is non-finite")
        return float(value)
    raise MetricProtocolError(f"unsupported paired metric: {metric!r}")


def _validated_inputs(
    true_delta: Sequence[float],
    pred_a: Sequence[float],
    pred_b: Sequence[float],
    components: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    y, a = _aligned(true_delta, pred_a)
    _, b = _aligned(true_delta, pred_b)
    comp = np.asarray(components, dtype=object)
    if comp.ndim != 1 or comp.shape != y.shape:
        raise MetricProtocolError("components must align with predictions")
    if any(not isinstance(v, str) or not v for v in comp.tolist()):
        raise MetricProtocolError("every row requires a non-empty source component")
    levels = sorted(set(comp.tolist()))
    if len(levels) < 2:
        raise MetricProtocolError("clustered inference requires at least two components")
    return y, a, b, comp, levels


def paired_component_bootstrap(
    true_delta: Sequence[float],
    pred_a: Sequence[float],
    pred_b: Sequence[float],
    components: Sequence[str],
    *,
    metric: str,
    n_resamples: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict:
    """Paired bootstrap resampling source components, never individual rows."""

    y, a, b, comp, levels = _validated_inputs(true_delta, pred_a, pred_b, components)
    if n_resamples < 1:
        raise MetricProtocolError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise MetricProtocolError("confidence must be in (0, 1)")
    estimate = _metric_value(y, a, metric) - _metric_value(y, b, metric)
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(n_resamples), dtype=np.float64)
    masks = {level: np.flatnonzero(comp == level) for level in levels}
    for i in range(int(n_resamples)):
        sampled = rng.choice(levels, size=len(levels), replace=True)
        idx = np.concatenate([masks[level] for level in sampled])
        values[i] = _metric_value(y[idx], a[idx], metric) - _metric_value(y[idx], b[idx], metric)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "method": "PAIRED_SOURCE_COMPONENT_BOOTSTRAP",
        "metric": metric,
        "orientation": "LARGER_UTILITY_IS_BETTER",
        "estimate_a_minus_b": float(estimate),
        "confidence": float(confidence),
        "ci": [float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))],
        "n_components": int(len(levels)),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "resampling_unit": "source_component",
        "gate_eligibility": "DEVELOPMENT_ROW_MICRO_DIAGNOSTIC_ONLY",
    }


def paired_component_permutation(
    true_delta: Sequence[float],
    pred_a: Sequence[float],
    pred_b: Sequence[float],
    components: Sequence[str],
    *,
    metric: str,
    n_permutations: int = 10000,
    seed: int = 42,
) -> dict:
    """Swap A/B predictions at whole-component level under the paired null."""

    y, a, b, comp, levels = _validated_inputs(true_delta, pred_a, pred_b, components)
    observed = _metric_value(y, a, metric) - _metric_value(y, b, metric)
    masks = {level: np.flatnonzero(comp == level) for level in levels}
    total_exact = 2 ** len(levels)
    if total_exact <= int(n_permutations):
        patterns = itertools.product((False, True), repeat=len(levels))
        mode = "EXACT"
    else:
        if n_permutations < 1:
            raise MetricProtocolError("n_permutations must be positive")
        rng = np.random.default_rng(int(seed))
        patterns = (rng.integers(0, 2, size=len(levels), dtype=np.int8) for _ in range(int(n_permutations)))
        mode = "MONTE_CARLO"
    extreme = 0
    total = 0
    for pattern in patterns:
        aa = a.copy()
        bb = b.copy()
        for swap, level in zip(pattern, levels):
            if bool(swap):
                idx = masks[level]
                aa[idx], bb[idx] = b[idx], a[idx]
        value = _metric_value(y, aa, metric) - _metric_value(y, bb, metric)
        extreme += int(abs(value) >= abs(observed) - 1e-15)
        total += 1
    p_value = (
        extreme / total
        if mode == "EXACT"
        else (extreme + 1.0) / (total + 1.0)
    )
    return {
        "method": "PAIRED_SOURCE_COMPONENT_PERMUTATION",
        "mode": mode,
        "metric": metric,
        "orientation": "LARGER_UTILITY_IS_BETTER",
        "estimate_a_minus_b": float(observed),
        "p_value_two_sided": float(min(1.0, p_value)),
        "n_components": int(len(levels)),
        "n_permutations": int(total),
        "seed": int(seed),
        "permutation_unit": "source_component",
        "gate_eligibility": "DEVELOPMENT_ROW_MICRO_DIAGNOSTIC_ONLY",
    }


def summarize_seeds(values: Mapping[int, float]) -> dict:
    """Descriptive optimizer-repeat summary; seeds are not independent studies."""

    if not values:
        raise MetricProtocolError("seed summary refuses empty values")
    ordered = sorted((int(seed), float(value)) for seed, value in values.items())
    array = _finite_vector([value for _, value in ordered], "seed_values")
    return {
        "seeds": [seed for seed, _ in ordered],
        "values": [float(value) for _, value in ordered],
        "mean": float(np.mean(array)),
        "range": [float(np.min(array)), float(np.max(array))],
        "inferential_unit": "NOT_BIOLOGICAL_REPLICATES",
        "status": "DESCRIPTIVE_OPTIMIZATION_REPEAT_SUMMARY",
    }
