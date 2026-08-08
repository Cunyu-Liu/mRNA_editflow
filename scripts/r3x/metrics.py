"""Publication-oriented ordinary-only effect metrics.

The routines in this module are deliberately ignorant of file paths and model
families.  They operate on already qualified arrays and make unsafe historical
selection ratios impossible to call accidentally.
"""
from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr


ZERO_POLICY = "EXCLUDE_TRUE_ZERO"
METRIC_CONTRACT = "R3X_ORDINARY_EFFECT_METRICS_V1"


class MetricProtocolError(ValueError):
    """Raised when a metric request violates the frozen metric protocol."""


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64)
    if out.ndim != 1:
        raise MetricProtocolError(f"{name} must be a one-dimensional vector")
    if not np.all(np.isfinite(out)):
        raise MetricProtocolError(f"{name} contains non-finite values")
    return out


def _aligned(true: Sequence[float], pred: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    y = _finite_vector(true, "true_delta")
    p = _finite_vector(pred, "pred_delta")
    if y.shape != p.shape:
        raise MetricProtocolError("true_delta and pred_delta must have identical shape")
    if y.size == 0:
        raise MetricProtocolError("metrics refuse empty arrays")
    return y, p


def _spearman(y: np.ndarray, pred: np.ndarray) -> Optional[float]:
    if y.size < 3 or np.ptp(y) == 0.0 or np.ptp(pred) == 0.0:
        return None
    value, _ = spearmanr(y, pred)
    return None if not np.isfinite(value) else float(value)


def cell_metrics(
    true_delta: Sequence[float],
    pred_delta: Sequence[float],
    *,
    logvar: Optional[Sequence[float]] = None,
    zero_policy: str = ZERO_POLICY,
) -> dict:
    """Compute metrics for one frozen study-endpoint-context cell.

    Zeros are excluded only from the sign denominator.  They remain in all
    continuous metrics.  Sign decisions are always the sign of predicted mean.
    """

    if zero_policy != ZERO_POLICY:
        raise MetricProtocolError(
            f"zero policy is frozen as {ZERO_POLICY}; got {zero_policy!r}"
        )
    y, pred = _aligned(true_delta, pred_delta)
    err = pred - y
    nonzero = y != 0.0
    if np.any(nonzero):
        predicted_sign = np.where(pred[nonzero] >= 0.0, 1.0, -1.0)
        true_sign = np.where(y[nonzero] > 0.0, 1.0, -1.0)
        sign_accuracy: Optional[float] = float(np.mean(predicted_sign == true_sign))
        sign_status = "ESTIMATED"
    else:
        sign_accuracy = None
        sign_status = "HOLD_NO_NONZERO_TRUE_EFFECTS"

    if logvar is None:
        gaussian_nll = None
    else:
        lv = _finite_vector(logvar, "logvar")
        if lv.shape != y.shape:
            raise MetricProtocolError("logvar must align with true_delta")
        # Bound only for numerical evaluation.  Training heads remain unmodified.
        lv_eval = np.clip(lv, -20.0, 20.0)
        gaussian_nll = float(
            np.mean(0.5 * (math.log(2.0 * math.pi) + lv_eval + err**2 * np.exp(-lv_eval)))
        )

    return {
        "metric_contract": METRIC_CONTRACT,
        "n": int(y.size),
        "n_sign": int(np.sum(nonzero)),
        "zero_policy": ZERO_POLICY,
        "sign_decision_rule": "RAW_PREDICTED_DELTA_MEAN_GE_ZERO",
        "sign_accuracy": sign_accuracy,
        "sign_status": sign_status,
        "delta_spearman": _spearman(y, pred),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "gaussian_nll": gaussian_nll,
    }


def _component_weights(components: Sequence[str], n: int) -> tuple[np.ndarray, int]:
    values = np.asarray(components, dtype=object)
    if values.ndim != 1 or len(values) != n or any(
        not isinstance(value, str) or not value for value in values.tolist()
    ):
        raise MetricProtocolError("components must be non-empty strings aligned to rows")
    levels = sorted(set(values.tolist()))
    weights = np.zeros(n, dtype=np.float64)
    for level in levels:
        mask = values == level
        weights[mask] = 1.0 / (len(levels) * int(np.sum(mask)))
    return weights, len(levels)


def _weighted_correlation(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> Optional[float]:
    if len(left) < 3 or np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return None
    left_rank = rankdata(left, method="average")
    right_rank = rankdata(right, method="average")
    left_centered = left_rank - np.sum(weights * left_rank)
    right_centered = right_rank - np.sum(weights * right_rank)
    denominator = math.sqrt(
        float(np.sum(weights * left_centered**2))
        * float(np.sum(weights * right_centered**2))
    )
    if denominator <= 0.0:
        return None
    return float(np.sum(weights * left_centered * right_centered) / denominator)


def component_balanced_cell_metrics(
    true_delta: Sequence[float],
    pred_delta: Sequence[float],
    components: Sequence[str],
    *,
    logvar: Optional[Sequence[float]] = None,
    zero_policy: str = ZERO_POLICY,
) -> dict:
    """Point estimates that give each global relation component equal mass."""

    if zero_policy != ZERO_POLICY:
        raise MetricProtocolError(f"zero policy is frozen as {ZERO_POLICY}")
    y, pred = _aligned(true_delta, pred_delta)
    weights, n_components = _component_weights(components, len(y))
    nonzero = y != 0.0
    if np.any(nonzero):
        # The sign estimand is defined on non-zero truths.  Recompute the
        # component weights after applying that zero policy so that a component
        # with many zero-labelled rows does not lose weight relative to a
        # component with the same number of informative sign observations.
        component_values = np.asarray(components, dtype=object)
        nz_weights, _ = _component_weights(
            component_values[nonzero].tolist(), int(np.sum(nonzero))
        )
        correct = (np.where(pred[nonzero] >= 0.0, 1.0, -1.0)
                   == np.where(y[nonzero] > 0.0, 1.0, -1.0))
        sign_accuracy: Optional[float] = float(np.sum(nz_weights * correct))
        sign_status = "ESTIMATED_COMPONENT_BALANCED"
    else:
        sign_accuracy = None
        sign_status = "HOLD_NO_NONZERO_TRUE_EFFECTS"
    err = pred - y
    if logvar is None:
        nll = None
    else:
        lv = _finite_vector(logvar, "logvar")
        if lv.shape != y.shape:
            raise MetricProtocolError("logvar must align with true_delta")
        bounded = np.clip(lv, -20.0, 20.0)
        nll_rows = 0.5 * (
            math.log(2.0 * math.pi) + bounded + err**2 * np.exp(-bounded)
        )
        nll = float(np.sum(weights * nll_rows))
    return {
        "metric_contract": "R3X_COMPONENT_BALANCED_EFFECT_METRICS_V2",
        "n": int(len(y)),
        "n_components": int(n_components),
        "n_sign": int(np.sum(nonzero)),
        "zero_policy": ZERO_POLICY,
        "component_weighting": "EQUAL_COMPONENT_THEN_EQUAL_ROW_WITHIN_COMPONENT",
        "sign_decision_rule": "RAW_PREDICTED_DELTA_MEAN_GE_ZERO",
        "sign_accuracy": sign_accuracy,
        "sign_status": sign_status,
        "effect_weighted_spearman_diagnostic": _weighted_correlation(y, pred, weights),
        # The primary delta_spearman is filled only by qualified within-pool
        # ranking below; cross-pool offsets may not satisfy a rank gate.
        "delta_spearman": None,
        "rmse": float(math.sqrt(np.sum(weights * err**2))),
        "mae": float(np.sum(weights * np.abs(err))),
        "gaussian_nll": nll,
    }


def within_pool_rank_metrics(
    true_delta: Sequence[float],
    pred_delta: Sequence[float],
    ranking_pools: Sequence[str],
    components: Sequence[str],
    *,
    min_candidates: int = 3,
    selection_fraction: float = 0.10,
) -> dict:
    """Rank/selection estimands computed within measured source pools only."""

    y, pred = _aligned(true_delta, pred_delta)
    pools = np.asarray(ranking_pools, dtype=object)
    comps = np.asarray(components, dtype=object)
    if pools.shape != y.shape or comps.shape != y.shape:
        raise MetricProtocolError("ranking pools/components must align with rows")
    if isinstance(min_candidates, bool) or not isinstance(min_candidates, int) or min_candidates < 2:
        raise MetricProtocolError("min_candidates must be an integer >=2")
    pool_results = {}
    structurally_eligible: dict[str, str] = {}
    for pool in sorted(set(pools.tolist())):
        if not isinstance(pool, str) or not pool:
            raise MetricProtocolError("ranking pool IDs must be non-empty strings")
        idx = np.flatnonzero(pools == pool)
        component_levels = sorted(set(comps[idx].tolist()))
        if len(component_levels) != 1:
            raise MetricProtocolError("one ranking pool cannot cross source components")
        if len(idx) < min_candidates:
            pool_results[pool] = {
                "status": "NOT_STRUCTURALLY_ELIGIBLE_MIN_CANDIDATES",
                "n": int(len(idx)),
                "source_component_id": component_levels[0],
            }
            continue
        structurally_eligible[pool] = component_levels[0]
        rho = _spearman(y[idx], pred[idx])
        selection = normalized_selection_gain(
            y[idx], pred[idx], fraction=selection_fraction
        )
        if np.ptp(y[idx]) == 0.0:
            status = "HOLD_REGISTERED_POOL_NO_TRUE_VARIATION"
        elif rho is None:
            status = "HOLD_REGISTERED_POOL_UNDEFINED_SPEARMAN"
        elif selection["value"] is None:
            status = "HOLD_REGISTERED_POOL_UNDEFINED_SELECTION"
        else:
            status = "ESTIMATED"
        pool_results[pool] = {
            "status": status,
            "n": int(len(idx)),
            "source_component_id": component_levels[0],
            "spearman": rho,
            "selection_gain": selection,
        }

    by_component: dict[str, list[str]] = {}
    for pool, component in structurally_eligible.items():
        by_component.setdefault(component, []).append(pool)

    def component_macro(field: str) -> tuple[Optional[float], dict[str, Optional[float]], bool]:
        component_values: dict[str, Optional[float]] = {}
        complete = bool(structurally_eligible)
        for component, component_pools in sorted(by_component.items()):
            values = []
            for pool in sorted(component_pools):
                value = (
                    pool_results[pool].get("spearman")
                    if field == "spearman"
                    else pool_results[pool]["selection_gain"].get("value")
                )
                if value is None:
                    complete = False
                else:
                    values.append(float(value))
            component_values[component] = (
                float(np.mean(values)) if len(values) == len(component_pools) else None
            )
        overall = (
            float(np.mean([float(value) for value in component_values.values()]))
            if complete and component_values else None
        )
        return overall, component_values, complete

    spearman_value, component_spearman, spearman_complete = component_macro("spearman")
    selection_value, component_selection, selection_complete = component_macro("selection")
    n_eligible = len(structurally_eligible)
    return {
        "aggregation": "WITHIN_POOL_THEN_EQUAL_POOL_WITHIN_COMPONENT_THEN_EQUAL_COMPONENT",
        "min_candidates": int(min_candidates),
        "n_registered_pools": len(pool_results),
        "n_structurally_eligible_pools": int(n_eligible),
        "n_components_with_structurally_eligible_pools": len(by_component),
        "spearman": spearman_value,
        "selection_gain": selection_value,
        "spearman_status": (
            "COMPLETE_ALL_STRUCTURALLY_ELIGIBLE_POOLS"
            if spearman_complete else "HOLD_REGISTERED_POOL_UNDEFINED"
        ),
        "selection_status": (
            "COMPLETE_ALL_STRUCTURALLY_ELIGIBLE_POOLS"
            if selection_complete else "HOLD_REGISTERED_POOL_UNDEFINED"
        ),
        "status": (
            "COMPLETE_ALL_STRUCTURALLY_ELIGIBLE_POOLS"
            if spearman_complete and selection_complete
            else "HOLD_REGISTERED_POOL_UNDEFINED"
        ),
        "component_spearman": component_spearman,
        "component_selection_gain": component_selection,
        "pools": pool_results,
    }


def normalized_selection_gain(
    true_delta: Sequence[float],
    pred_delta: Sequence[float],
    *,
    fraction: float = 0.10,
    min_headroom: float = 1e-12,
) -> dict:
    """Monotone oracle-normalized selection diagnostic.

    This value has no activation threshold in R3X.  It cannot satisfy the old
    top-10 enrichment gate until a future stable metric policy is frozen.
    """

    y, pred = _aligned(true_delta, pred_delta)
    if not (0.0 < float(fraction) <= 1.0):
        raise MetricProtocolError("selection fraction must be in (0, 1]")
    k = max(1, int(math.ceil(y.size * float(fraction))))
    def expected_top_mean(score: np.ndarray, outcome: np.ndarray) -> tuple[float, dict]:
        threshold = float(np.sort(score)[-k])
        strict = score > threshold
        tied = score == threshold
        strict_count = int(np.sum(strict))
        slots = k - strict_count
        tie_count = int(np.sum(tied))
        if slots < 0 or slots > tie_count or tie_count < 1:
            raise MetricProtocolError("internal cutoff-tie accounting is invalid")
        expected_sum = float(np.sum(outcome[strict]))
        expected_sum += float(slots) * float(np.mean(outcome[tied]))
        return expected_sum / float(k), {
            "cutoff_score": threshold,
            "strictly_selected": strict_count,
            "cutoff_tie_count": tie_count,
            "fractional_slots_from_tie": slots,
        }

    # Fractional expectation over the cutoff tie block is invariant to JSONL
    # row order and does not use outcomes as a secondary tie-breaker.
    selected, selected_tie = expected_top_mean(pred, y)
    oracle, oracle_tie = expected_top_mean(y, y)
    overall = float(np.mean(y))
    headroom = oracle - overall
    if not np.isfinite(headroom) or headroom <= float(min_headroom):
        value = None
        status = "HOLD_NO_STABLE_ORACLE_HEADROOM"
    else:
        value = float((selected - overall) / headroom)
        status = "DIAGNOSTIC_ONLY_NO_FROZEN_GATE"
    return {
        "metric": "ORACLE_NORMALIZED_SELECTION_GAIN",
        "value": value,
        "status": status,
        "fraction": float(fraction),
        "k": int(k),
        "selected_mean": selected,
        "oracle_mean": oracle,
        "overall_mean": overall,
        "oracle_headroom": float(headroom),
        "tie_policy": "EXPECTED_OUTCOME_OVER_CUTOFF_TIE_BLOCK",
        "selected_tie_accounting": selected_tie,
        "oracle_tie_accounting": oracle_tie,
    }


def legacy_signed_ratio_enrichment(*_args, **_kwargs) -> float:
    """Permanently reject the non-monotone historical signed ratio."""

    raise MetricProtocolError(
        "PROHIBITED_LEGACY_SIGNED_RATIO: the signed mean ratio reverses order "
        "on negative-valued endpoints and cannot be a publication gate"
    )


def macro_metrics(cells: Mapping[str, Mapping[str, Optional[float]]]) -> dict:
    """Unweighted macro average over already qualified metric cells."""

    if not cells:
        raise MetricProtocolError("macro metrics refuse an empty cell mapping")

    metric_keys = (
        "sign_accuracy", "delta_spearman", "rmse", "mae", "gaussian_nll"
    )

    def coverage(key: str) -> dict:
        defined = sorted(cell for cell, value in cells.items() if value.get(key) is not None)
        missing = sorted(set(cells) - set(defined))
        return {
            "n_registered_cells": len(cells),
            "n_defined_cells": len(defined),
            "defined_cell_ids": defined,
            "undefined_cell_ids": missing,
            "complete": not missing,
        }

    coverages = {key: coverage(key) for key in metric_keys}

    def average(key: str) -> Optional[float]:
        # A primary macro never silently changes its denominator.  One
        # undefined registered cell makes the macro a structured HOLD.
        if not coverages[key]["complete"]:
            return None
        return float(np.mean([float(v[key]) for v in cells.values()]))

    return {
        "metric_contract": METRIC_CONTRACT,
        "aggregation": "UNWEIGHTED_STUDY_ENDPOINT_CONTEXT_CELL_MACRO",
        "n_cells": int(len(cells)),
        "n_rows": int(sum(int(v.get("n", 0)) for v in cells.values())),
        "macro_sign_accuracy": average("sign_accuracy"),
        "macro_delta_spearman": average("delta_spearman"),
        "macro_rmse": average("rmse"),
        "macro_mae": average("mae"),
        "macro_gaussian_nll": average("gaussian_nll"),
        "metric_cell_coverage": coverages,
        "primary_macro_status": (
            "COMPLETE_ALL_REGISTERED_CELLS"
            if all(value["complete"] for value in coverages.values())
            else "HOLD_PRIMARY_CELL_UNDEFINED"
        ),
    }
