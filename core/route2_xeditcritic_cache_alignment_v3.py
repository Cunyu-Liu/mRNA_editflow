"""Predeclared numerical adjudication for Critic V3 cache/online features."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch


class CacheAlignmentV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CacheAlignmentV3Error(message)


def compare_cache_online_features_v3(
    cached: Mapping[str, torch.Tensor],
    online: Mapping[str, torch.Tensor],
    *,
    active_edit_count: int,
    maximum_absolute_tolerance: float,
    mean_absolute_tolerance: float,
) -> dict[str, Any]:
    _require(active_edit_count > 0, "alignment record has no edits")
    _require(maximum_absolute_tolerance > 0 and mean_absolute_tolerance > 0, "alignment tolerances are invalid")
    local = (
        "source_site", "candidate_site",
        "source_window_mean", "candidate_window_mean",
        "source_window_max", "candidate_window_max",
    )
    global_names = ("source_global", "candidate_global")
    rows = {}
    all_differences = []
    for name in (*local, *global_names):
        left = cached[name]
        right = online[name]
        if name in local:
            left = left[:, :active_edit_count]
            right = right[:, :active_edit_count]
        _require(left.shape == right.shape, f"alignment geometry differs: {name}")
        difference = (left.float() - right.float()).abs()
        _require(torch.isfinite(difference).all().item(), f"alignment difference is nonfinite: {name}")
        rows[name] = {
            "maximum_absolute_difference": float(difference.max().item()),
            "mean_absolute_difference": float(difference.mean().item()),
            "value_count": int(difference.numel()),
        }
        all_differences.append(difference.reshape(-1))
    combined = torch.cat(all_differences)
    maximum = float(combined.max().item())
    mean = float(combined.mean().item())
    passed = maximum <= maximum_absolute_tolerance and mean <= mean_absolute_tolerance
    return {
        "status": "CACHE_ONLINE_NUMERIC_ALIGNMENT_PASS" if passed else "CACHE_ONLINE_NUMERIC_ALIGNMENT_FAIL",
        "maximum_absolute_difference": maximum,
        "mean_absolute_difference": mean,
        "maximum_absolute_tolerance": maximum_absolute_tolerance,
        "mean_absolute_tolerance": mean_absolute_tolerance,
        "feature_rows": rows,
        "passed": passed,
    }
