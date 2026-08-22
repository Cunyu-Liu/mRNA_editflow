"""Common-prefix equal-wall-time sensitivity for the final XEditFlow V3 benchmark."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


METHODS_V3 = {
    "full_soft_value_smc",
    "unguided_setflow",
    "first_order_guidance",
    "simple_rate_guidance",
    "generate_then_rerank",
    "strongest_matched_baseline",
}
EQUAL_WALL_TIME_SCOPE_V3 = (
    "A100_END_TO_END_GENERATION_INCLUDING_REPLAY_AND_REQUIRED_SELECTION_SCORING"
)


class XEditFlowEqualWallTimeV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowEqualWallTimeV3Error(message)


def _finite_positive(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    _require(math.isfinite(result) and result > 0.0, f"{label} is not finite-positive")
    return result


def _a100_name(value: Any, label: str) -> str:
    _require(isinstance(value, str) and "A100" in value.upper(), f"{label} is not A100")
    return value


def _prefix_count_within_budget(values: Sequence[float], budget: float) -> int:
    cumulative = np.cumsum(np.asarray(values, dtype=np.float64))
    return int(np.searchsorted(cumulative, budget + 1e-12, side="right"))


def _macro_on_sources(
    closed: Mapping[str, Any], source_keys: Sequence[str], *, method: str
) -> dict[str, Any]:
    _require(
        closed.get("status") == "XEDITFLOW_V3_CLOSED_NEIGHBORHOOD_COMPLETE"
        and closed.get("undefined_sources_are_not_filled_with_zero") is True,
        f"equal-wall closed evidence is incomplete: {method}",
    )
    _require(
        closed.get("development_test_outcomes_accessed") is False
        and closed.get("new_final_evaluation_outcomes_accessed") is False,
        f"equal-wall closed evidence accessed protected outcome: {method}",
    )
    rows = closed.get("per_source")
    _require(isinstance(rows, Mapping), f"equal-wall closed per-source rows are absent: {method}")
    selected = [rows[key] for key in source_keys]
    defined = []
    for row in selected:
        if row.get("status") != "DEFINED":
            _require(
                row.get("ndcg") is None
                and row.get("normalized_regret") is None
                and row.get("top_1_recall") is None,
                f"equal-wall undefined source carries a metric: {method}",
            )
        else:
            _require(
                all(
                    isinstance(row.get(key), (int, float))
                    and not isinstance(row.get(key), bool)
                    and math.isfinite(float(row[key]))
                    for key in ("ndcg", "normalized_regret", "top_1_recall")
                ),
                f"equal-wall defined source metric is invalid: {method}",
            )
            defined.append(row)
    _require(len(defined) >= 2, f"equal-wall closed defined support is too small: {method}")
    return {
        "source_count": len(selected),
        "defined_source_count": len(defined),
        "source_macro_ndcg": float(np.mean([float(row["ndcg"]) for row in defined])),
        "source_macro_normalized_regret": float(
            np.mean([float(row["normalized_regret"]) for row in defined])
        ),
        "source_macro_top_1_recall": float(
            np.mean([float(row["top_1_recall"]) for row in defined])
        ),
    }


def equal_wall_time_sensitivity_v3(
    time_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    closed_results: Mapping[str, Mapping[str, Any]],
    *,
    source_order: Sequence[str],
    base_flow_training_seed: int,
) -> dict[str, Any]:
    """Evaluate all methods on the largest common source prefix within one wall budget.

    The budget is the smallest observed full-cohort generation time among methods.
    Each method retains only completely processed sources before that wall time.  The
    final sensitivity uses the intersection expressed as a common frozen-order prefix,
    so methods never receive different source cohorts and a partially processed source
    is never treated as complete.
    """

    _require(base_flow_training_seed in {20260904, 20260905, 20260906}, "equal-wall seed differs")
    _require(set(time_rows) == set(closed_results) == METHODS_V3, "equal-wall method inventory differs")
    ordered_sources = [str(value) for value in source_order]
    _require(
        len(ordered_sources) == 891 and len(set(ordered_sources)) == 891,
        "equal-wall source order differs from the frozen 891-source cohort",
    )
    wall_by_method: dict[str, list[float]] = {}
    scope_by_method = {}
    accelerator_by_method = {}
    peak_vram_by_method = {}
    total_by_method = {}
    for method in sorted(METHODS_V3):
        rows = list(time_rows[method])
        _require(len(rows) == 891, f"equal-wall time row count differs: {method}")
        _require(
            [str(row.get("source_key")) for row in rows] == ordered_sources,
            f"equal-wall source order differs: {method}",
        )
        values = [
            _finite_positive(row.get("source_wall_time_seconds"), f"source wall time {method}")
            for row in rows
        ]
        scopes = {str(row.get("wall_time_scope")) for row in rows}
        _require(
            scopes == {EQUAL_WALL_TIME_SCOPE_V3},
            f"equal-wall time scope differs: {method}",
        )
        wall_by_method[method] = values
        scope_by_method[method] = next(iter(scopes))
        accelerators = {
            _a100_name(row.get("accelerator_name"), f"equal-wall accelerator {method}")
            for row in rows
        }
        _require(len(accelerators) == 1, f"equal-wall accelerator differs: {method}")
        accelerator_by_method[method] = next(iter(accelerators))
        peak_vram_by_method[method] = max(
            _finite_positive(row.get("peak_vram_mb"), f"equal-wall peak VRAM {method}")
            for row in rows
        )
        total_by_method[method] = float(math.fsum(values))
        per_source = closed_results[method].get("per_source")
        _require(
            isinstance(per_source, Mapping)
            and set(map(str, per_source)) == set(ordered_sources),
            f"equal-wall closed source inventory differs: {method}",
        )
    _require(
        len(set(accelerator_by_method.values())) == 1,
        "equal-wall accelerator model differs across methods",
    )
    common_budget = min(total_by_method.values())
    completed_counts = {
        method: _prefix_count_within_budget(values, common_budget)
        for method, values in wall_by_method.items()
    }
    common_count = min(completed_counts.values())
    _require(common_count >= 2, "equal-wall common completed-source prefix is too small")
    common_sources = ordered_sources[:common_count]
    methods = {}
    for method in sorted(METHODS_V3):
        used_wall = float(math.fsum(wall_by_method[method][:common_count]))
        _require(used_wall <= common_budget + 1e-9, f"equal-wall method exceeds common budget: {method}")
        methods[method] = {
            "full_cohort_generation_wall_time_seconds": total_by_method[method],
            "completed_source_count_within_common_budget": completed_counts[method],
            "common_prefix_generation_wall_time_seconds": used_wall,
            "wall_time_scope": scope_by_method[method],
            "accelerator_name": accelerator_by_method[method],
            "peak_vram_mb": peak_vram_by_method[method],
            **_macro_on_sources(
                closed_results[method], common_sources, method=method
            ),
        }
    full = methods["full_soft_value_smc"]
    unguided = methods["unguided_setflow"]
    baseline = methods["strongest_matched_baseline"]
    return {
        "schema_version": "route_a_v3_route2_xeditflow_equal_wall_time_sensitivity.v1",
        "status": "XEDITFLOW_V3_EQUAL_WALL_TIME_SENSITIVITY_COMPLETE",
        "base_flow_training_seed": base_flow_training_seed,
        "analysis_unit": "SOURCE",
        "source_order_policy": "FROZEN_MANIFEST_ORDER_COMMON_COMPLETED_PREFIX",
        "common_wall_time_budget_seconds": common_budget,
        "common_source_prefix_count": common_count,
        "common_source_keys": common_sources,
        "partial_source_counted_as_complete": False,
        "methods": methods,
        "direction_diagnostics_not_a_separate_gate": {
            "full_ndcg_above_unguided": full["source_macro_ndcg"]
            > unguided["source_macro_ndcg"],
            "full_ndcg_above_strongest": full["source_macro_ndcg"]
            > baseline["source_macro_ndcg"],
            "full_regret_below_unguided": full["source_macro_normalized_regret"]
            < unguided["source_macro_normalized_regret"],
            "full_regret_below_strongest": full["source_macro_normalized_regret"]
            < baseline["source_macro_normalized_regret"],
        },
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
