from __future__ import annotations

import torch

from core.route2_xeditcritic_cache_alignment_v3 import compare_cache_online_features_v3


def _features(offset=0.0):
    result = {}
    for name in (
        "source_site", "candidate_site", "source_window_mean", "candidate_window_mean",
        "source_window_max", "candidate_window_max",
    ):
        result[name] = torch.zeros(1, 2, 3) + offset
    result["source_global"] = torch.zeros(1, 3) + offset
    result["candidate_global"] = torch.zeros(1, 3) + offset
    return result


def test_alignment_uses_predeclared_max_and_mean_tolerances() -> None:
    passed = compare_cache_online_features_v3(
        _features(), _features(0.004), active_edit_count=2,
        maximum_absolute_tolerance=0.02, mean_absolute_tolerance=0.005,
    )
    assert passed["passed"] is True
    failed = compare_cache_online_features_v3(
        _features(), _features(0.006), active_edit_count=2,
        maximum_absolute_tolerance=0.02, mean_absolute_tolerance=0.005,
    )
    assert failed["passed"] is False


def test_padded_edit_rows_do_not_enter_alignment() -> None:
    cached = _features()
    online = _features()
    for name in ("source_site", "candidate_site", "source_window_mean", "candidate_window_mean", "source_window_max", "candidate_window_max"):
        online[name][:, 1] = 100.0
    result = compare_cache_online_features_v3(
        cached, online, active_edit_count=1,
        maximum_absolute_tolerance=0.02, mean_absolute_tolerance=0.005,
    )
    assert result["passed"] is True
