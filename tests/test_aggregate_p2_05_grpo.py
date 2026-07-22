"""Unit tests for P2-05 GRPO aggregation script.

Tests verify curve loading, metric extraction, family-cluster bootstrap CI,
paired bootstrap p-value, and verdict determination.
"""
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


def _write_curves(path: Path, curves):
    """Write a list of metric dicts as JSONL."""
    with path.open("w") as f:
        for c in curves:
            f.write(json.dumps(c) + "\n")


def _write_trajectories(path: Path, trajs):
    """Write trajectory entries as JSONL."""
    with path.open("w") as f:
        for t in trajs:
            f.write(json.dumps(t) + "\n")


def _make_seed_dir(base: Path, seed: int, curves, trajs=None, metadata=None):
    """Create a seed directory with curves.jsonl and optional trajectories."""
    sd = base / ("seed%d" % seed)
    sd.mkdir(parents=True, exist_ok=True)
    _write_curves(sd / "curves.jsonl", curves)
    if trajs is not None:
        _write_trajectories(sd / "trajectories.jsonl", trajs)
    if metadata is not None:
        with (sd / "run_metadata.json").open("w") as f:
            json.dump(metadata, f)
    return sd


def _load_aggregator():
    """Import the aggregation module from the scripts directory."""
    import importlib.util
    # Try multiple candidate locations
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "scripts", "aggregate_p2_05_grpo.py"),
        "/home/cunyuliu/mrna_editflow_goal/mrna_editflow/scripts/aggregate_p2_05_grpo.py",
        "/tmp/aggregate_p2_05_grpo.py",
    ]
    for path in candidates:
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("agg_p2_05", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("aggregate_p2_05_grpo.py not found in candidates: %s" % candidates)


def test_load_curves(tmp_path):
    """Curves JSONL is loaded correctly."""
    mod = _load_aggregator()
    curves = [
        {"iter": 0, "mean_return": 0.1, "loss": 1.0},
        {"iter": 1, "mean_return": 0.2, "loss": 0.9},
        {"iter": 2, "mean_return": 0.3, "loss": 0.8},
    ]
    p = tmp_path / "curves.jsonl"
    _write_curves(p, curves)
    loaded = mod.load_curves(p)
    assert len(loaded) == 3
    assert loaded[0]["iter"] == 0
    assert loaded[2]["mean_return"] == 0.3


def test_extract_metric_series(tmp_path):
    """Metric series extraction returns correct values."""
    mod = _load_aggregator()
    curves = [
        {"iter": 0, "mean_return": 0.1},
        {"iter": 1, "mean_return": 0.5},
        {"iter": 2, "mean_return": 0.3},
    ]
    series = mod.extract_metric_series(curves, "mean_return")
    assert series == [0.1, 0.5, 0.3]


def test_compute_final_metric(tmp_path):
    """Final metric is the last iteration's value."""
    mod = _load_aggregator()
    curves = [
        {"iter": 0, "mean_return": 0.1},
        {"iter": 1, "mean_return": 0.5},
        {"iter": 2, "mean_return": 0.3},
    ]
    assert mod.compute_final_metric(curves, "mean_return") == pytest.approx(0.3)


def test_compute_best_metric_maximize(tmp_path):
    """Best metric (maximize=True) returns the maximum value."""
    mod = _load_aggregator()
    curves = [
        {"iter": 0, "mean_return": 0.1},
        {"iter": 1, "mean_return": 0.5},
        {"iter": 2, "mean_return": 0.3},
    ]
    assert mod.compute_best_metric(curves, "mean_return", maximize=True) == pytest.approx(0.5)


def test_compute_best_metric_minimize(tmp_path):
    """Best metric (maximize=False) returns the minimum value (e.g. loss)."""
    mod = _load_aggregator()
    curves = [
        {"iter": 0, "loss": 1.0},
        {"iter": 1, "loss": 0.5},
        {"iter": 2, "loss": 0.8},
    ]
    assert mod.compute_best_metric(curves, "loss", maximize=False) == pytest.approx(0.5)


def test_family_cluster_bootstrap_ci_basic():
    """Bootstrap CI contains the sample mean."""
    mod = _load_aggregator()
    values = [0.1, 0.2, 0.3]
    lo, hi = mod.family_cluster_bootstrap_ci(values, n_bootstrap=1000, seed=42)
    mean = np.mean(values)
    assert lo <= mean <= hi
    assert lo < hi


def test_family_cluster_bootstrap_ci_empty():
    """Empty input returns NaN CI."""
    mod = _load_aggregator()
    lo, hi = mod.family_cluster_bootstrap_ci([])
    assert math.isnan(lo) and math.isnan(hi)


def test_family_cluster_bootstrap_ci_constant():
    """Constant values produce a zero-width CI."""
    mod = _load_aggregator()
    values = [0.5, 0.5, 0.5]
    lo, hi = mod.family_cluster_bootstrap_ci(values, n_bootstrap=1000, seed=42)
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)


def test_paired_bootstrap_pvalue_significant():
    """Paired bootstrap detects significant improvement."""
    mod = _load_aggregator()
    # Treatment consistently higher than baseline (10 pairs for power)
    treatment = [0.5, 0.6, 0.7, 0.55, 0.65, 0.75, 0.52, 0.62, 0.72, 0.58]
    baseline = [0.1, 0.2, 0.3, 0.15, 0.25, 0.35, 0.12, 0.22, 0.32, 0.18]
    p = mod.paired_bootstrap_pvalue(treatment, baseline, n_bootstrap=5000, seed=42)
    assert p < 0.05


def test_paired_bootstrap_pvalue_no_effect():
    """Paired bootstrap returns high p-value when no difference."""
    mod = _load_aggregator()
    treatment = [0.5, 0.5, 0.5]
    baseline = [0.5, 0.5, 0.5]
    p = mod.paired_bootstrap_pvalue(treatment, baseline, n_bootstrap=5000, seed=42)
    assert p > 0.3  # Should not be significant


def test_aggregate_seed_results_basic(tmp_path):
    """Aggregation across 3 seeds produces correct structure."""
    mod = _load_aggregator()
    seeds_data = [
        [({"iter": 0, "mean_return": 0.1, "loss": 1.0, "mean_kl": 0.01,
           "mean_entropy": 0.5, "mean_advantage": 0.0, "return_std_mean": 0.1}),
         ({"iter": 1, "mean_return": 0.3, "loss": 0.8, "mean_kl": 0.02,
           "mean_entropy": 0.4, "mean_advantage": 0.1, "return_std_mean": 0.15})],
        [({"iter": 0, "mean_return": 0.15, "loss": 1.0, "mean_kl": 0.01,
           "mean_entropy": 0.5, "mean_advantage": 0.0, "return_std_mean": 0.1}),
         ({"iter": 1, "mean_return": 0.35, "loss": 0.85, "mean_kl": 0.02,
           "mean_entropy": 0.45, "mean_advantage": 0.05, "return_std_mean": 0.12})],
        [({"iter": 0, "mean_return": 0.12, "loss": 1.0, "mean_kl": 0.01,
           "mean_entropy": 0.5, "mean_advantage": 0.0, "return_std_mean": 0.1}),
         ({"iter": 1, "mean_return": 0.32, "loss": 0.82, "mean_kl": 0.02,
           "mean_entropy": 0.42, "mean_advantage": 0.08, "return_std_mean": 0.13})],
    ]
    seed_dirs = []
    for i, curves in enumerate(seeds_data):
        sd = _make_seed_dir(tmp_path, i, curves)
        seed_dirs.append(sd)

    result = mod.aggregate_seed_results(seed_dirs)
    assert result["n_seeds"] == 3
    assert len(result["per_seed"]) == 3
    # Final returns: 0.3, 0.35, 0.32
    finals = [s["final_mean_return"] for s in result["per_seed"]]
    assert finals[0] == pytest.approx(0.3)
    assert finals[1] == pytest.approx(0.35)
    assert finals[2] == pytest.approx(0.32)
    # Cross-seed mean
    cs = result["cross_seed"]["final_mean_return"]
    assert cs["mean"] == pytest.approx(np.mean([0.3, 0.35, 0.32]))
    assert cs["n"] == 3
    assert cs["ci_95"]["low"] <= cs["mean"] <= cs["ci_95"]["high"]


def test_aggregate_improvement(tmp_path):
    """Improvement = final - initial mean_return."""
    mod = _load_aggregator()
    curves = [
        {"iter": 0, "mean_return": 0.1, "loss": 1.0, "mean_kl": 0.0,
         "mean_entropy": 0.5, "mean_advantage": 0.0, "return_std_mean": 0.1},
        {"iter": 1, "mean_return": 0.4, "loss": 0.8, "mean_kl": 0.0,
         "mean_entropy": 0.4, "mean_advantage": 0.1, "return_std_mean": 0.1},
    ]
    sd = _make_seed_dir(tmp_path, 0, curves)
    result = mod.aggregate_seed_results([sd])
    assert result["per_seed"][0]["improvement"] == pytest.approx(0.3)
    assert result["per_seed"][0]["initial_mean_return"] == pytest.approx(0.1)


def test_determine_verdict_improves():
    """Verdict is 'improves' when CI lower bound > baseline."""
    mod = _load_aggregator()
    aggregated = {
        "cross_seed": {
            "final_mean_return": {
                "mean": 0.5,
                "ci_95": {"low": 0.4, "high": 0.6},
            }
        }
    }
    verdict = mod.determine_verdict(aggregated, baseline_final_return=0.3)
    assert verdict["verdict"] == "improves"


def test_determine_verdict_degraded():
    """Verdict is 'degraded' when CI upper bound < baseline."""
    mod = _load_aggregator()
    aggregated = {
        "cross_seed": {
            "final_mean_return": {
                "mean": 0.2,
                "ci_95": {"low": 0.1, "high": 0.3},
            }
        }
    }
    verdict = mod.determine_verdict(aggregated, baseline_final_return=0.5)
    assert verdict["verdict"] == "degraded"


def test_determine_verdict_inconclusive():
    """Verdict is 'inconclusive' when CI overlaps baseline."""
    mod = _load_aggregator()
    aggregated = {
        "cross_seed": {
            "final_mean_return": {
                "mean": 0.4,
                "ci_95": {"low": 0.3, "high": 0.5},
            }
        }
    }
    verdict = mod.determine_verdict(aggregated, baseline_final_return=0.4)
    assert verdict["verdict"] == "inconclusive"


def test_determine_verdict_no_baseline():
    """Verdict is 'no_baseline' when baseline is None."""
    mod = _load_aggregator()
    aggregated = {
        "cross_seed": {
            "final_mean_return": {
                "mean": 0.5,
                "ci_95": {"low": 0.4, "high": 0.6},
            }
        }
    }
    verdict = mod.determine_verdict(aggregated, baseline_final_return=None)
    assert verdict["verdict"] == "no_baseline"


def test_main_writes_output(tmp_path):
    """Main function writes output JSON."""
    mod = _load_aggregator()
    curves = [
        {"iter": 0, "mean_return": 0.1, "loss": 1.0, "mean_kl": 0.0,
         "mean_entropy": 0.5, "mean_advantage": 0.0, "return_std_mean": 0.1},
        {"iter": 1, "mean_return": 0.3, "loss": 0.8, "mean_kl": 0.0,
         "mean_entropy": 0.4, "mean_advantage": 0.1, "return_std_mean": 0.1},
    ]
    seed_dirs = []
    for i in range(3):
        sd = _make_seed_dir(tmp_path, i, curves)
        seed_dirs.append(str(sd))
    out = tmp_path / "results.json"
    rc = mod.main(["--seed-dirs"] + seed_dirs + ["--output", str(out)])
    assert rc == 0
    assert out.exists()
    result = json.loads(out.read_text())
    assert result["n_policy_seeds"] == 3
    assert "verdict" in result
    assert "aggregated" in result
