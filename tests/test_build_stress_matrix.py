"""Unit tests for build_stress_matrix.py (P2-09 stress matrix packaging).

Tests verify that the stress_matrix.json artifact is correctly constructed
from the P2-09 v3 robustness summary, including worst-group identification,
failure-rate computation, and subset aggregation.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest


def _make_summary():
    """Construct a minimal robustness_summary.json-like dict for testing."""
    return {
        "reward_qualifier": "predicted_te_internal_proxy",
        "primary_endpoint": "delta_oracle_te_vs_source",
        "alpha_corrected": 0.00625,
        "n_subsets": 3,
        "n_baselines": 2,
        "n_sources": 100,
        "subset_sizes": {"a": 10, "b": 10, "c": 80},
        "robustness_scores": {"baseline_a": -0.001, "baseline_b": 0.002},
        "baselines": {
            "baseline_a": [
                {"baseline": "baseline_a", "subset": "a",
                 "diff_ood_minus_id": -0.005, "significant_after_bonferroni": True,
                 "n_ood": 10, "n_id": 90, "ood_mean": 0.001, "id_mean": 0.006,
                 "cohens_d": -0.3, "p_value": 0.001},
                {"baseline": "baseline_a", "subset": "b",
                 "diff_ood_minus_id": 0.002, "significant_after_bonferroni": False,
                 "n_ood": 10, "n_id": 90, "ood_mean": 0.008, "id_mean": 0.006,
                 "cohens_d": 0.1, "p_value": 0.5},
                {"baseline": "baseline_a", "subset": "c",
                 "diff_ood_minus_id": -0.001, "significant_after_bonferroni": False,
                 "n_ood": 80, "n_id": 20, "ood_mean": 0.005, "id_mean": 0.006,
                 "cohens_d": -0.05, "p_value": 0.3},
            ],
            "baseline_b": [
                {"baseline": "baseline_b", "subset": "a",
                 "diff_ood_minus_id": 0.003, "significant_after_bonferroni": False,
                 "n_ood": 10, "n_id": 90, "ood_mean": 0.009, "id_mean": 0.006,
                 "cohens_d": 0.2, "p_value": 0.4},
                {"baseline": "baseline_b", "subset": "b",
                 "diff_ood_minus_id": -0.004, "significant_after_bonferroni": True,
                 "n_ood": 10, "n_id": 90, "ood_mean": 0.002, "id_mean": 0.006,
                 "cohens_d": -0.25, "p_value": 0.002},
                {"baseline": "baseline_b", "subset": "c",
                 "diff_ood_minus_id": -0.002, "significant_after_bonferroni": True,
                 "n_ood": 80, "n_id": 20, "ood_mean": 0.004, "id_mean": 0.006,
                 "cohens_d": -0.15, "p_value": 0.001},
            ],
        },
    }


def _run_build(src_path, dst_path):
    """Run the build_stress_matrix logic with given paths."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("bsm", "/tmp/build_stress_matrix.py")
    mod = importlib.util.module_from_spec(spec)
    # Patch SRC/DST before exec
    mod.__dict__["SRC"] = str(src_path)
    mod.__dict__["DST"] = str(dst_path)
    spec.loader.exec_module(mod)
    mod.SRC = str(src_path)
    mod.DST = str(dst_path)
    mod.main()


def test_stress_matrix_basic_structure(tmp_path):
    """Stress matrix has required top-level keys and both baselines."""
    src = tmp_path / "summary.json"
    src.write_text(json.dumps(_make_summary()))
    dst = tmp_path / "stress_matrix.json"
    _run_build(src, dst)

    out = json.loads(dst.read_text())
    assert out["schema_version"] == "1.0"
    assert out["n_baselines"] == 2
    assert out["n_subsets"] == 3
    assert set(out["stress_matrix"].keys()) == {"baseline_a", "baseline_b"}


def test_worst_group_identification(tmp_path):
    """Worst group is the subset with the most negative diff_ood_minus_id."""
    src = tmp_path / "summary.json"
    src.write_text(json.dumps(_make_summary()))
    dst = tmp_path / "stress_matrix.json"
    _run_build(src, dst)

    out = json.loads(dst.read_text())
    # baseline_a: deltas are -0.005, 0.002, -0.001 -> worst is "a" at -0.005
    assert out["stress_matrix"]["baseline_a"]["worst_group"] == "a"
    assert out["stress_matrix"]["baseline_a"]["worst_group_delta"] == pytest.approx(-0.005)
    # baseline_b: deltas are 0.003, -0.004, -0.002 -> worst is "b" at -0.004
    assert out["stress_matrix"]["baseline_b"]["worst_group"] == "b"
    assert out["stress_matrix"]["baseline_b"]["worst_group_delta"] == pytest.approx(-0.004)


def test_failure_rate_computation(tmp_path):
    """Failure rate = fraction of subsets with sig AND negative delta."""
    src = tmp_path / "summary.json"
    src.write_text(json.dumps(_make_summary()))
    dst = tmp_path / "stress_matrix.json"
    _run_build(src, dst)

    out = json.loads(dst.read_text())
    # baseline_a: only "a" is sig+neg -> 1/3
    assert out["stress_matrix"]["baseline_a"]["failure_rate"] == pytest.approx(1.0 / 3.0)
    assert out["stress_matrix"]["baseline_a"]["n_significant_degradation"] == 1
    # baseline_b: "b" and "c" are sig+neg -> 2/3
    assert out["stress_matrix"]["baseline_b"]["failure_rate"] == pytest.approx(2.0 / 3.0)
    assert out["stress_matrix"]["baseline_b"]["n_significant_degradation"] == 2


def test_robustness_score_propagated(tmp_path):
    """Robustness score from summary is propagated to each baseline."""
    src = tmp_path / "summary.json"
    src.write_text(json.dumps(_make_summary()))
    dst = tmp_path / "stress_matrix.json"
    _run_build(src, dst)

    out = json.loads(dst.read_text())
    assert out["stress_matrix"]["baseline_a"]["robustness_score"] == pytest.approx(-0.001)
    assert out["stress_matrix"]["baseline_b"]["robustness_score"] == pytest.approx(0.002)


def test_subset_data_preserved(tmp_path):
    """Per-subset details are preserved in the output."""
    src = tmp_path / "summary.json"
    src.write_text(json.dumps(_make_summary()))
    dst = tmp_path / "stress_matrix.json"
    _run_build(src, dst)

    out = json.loads(dst.read_text())
    subs = out["stress_matrix"]["baseline_a"]["subsets"]
    assert set(subs.keys()) == {"a", "b", "c"}
    assert subs["a"]["diff_ood_minus_id"] == pytest.approx(-0.005)
    assert subs["a"]["significant_after_bonferroni"] is True
    assert subs["b"]["significant_after_bonferroni"] is False


def test_reward_qualifier_propagated(tmp_path):
    """Reward qualifier is carried through to the output artifact."""
    src = tmp_path / "summary.json"
    src.write_text(json.dumps(_make_summary()))
    dst = tmp_path / "stress_matrix.json"
    _run_build(src, dst)

    out = json.loads(dst.read_text())
    assert out["reward_qualifier"] == "predicted_te_internal_proxy"
    assert out["primary_endpoint"] == "delta_oracle_te_vs_source"
    assert out["alpha_corrected"] == pytest.approx(0.00625)


def test_creates_output_directory(tmp_path):
    """Output directory is created if it does not exist."""
    src = tmp_path / "summary.json"
    src.write_text(json.dumps(_make_summary()))
    dst = tmp_path / "nested" / "dir" / "stress_matrix.json"
    _run_build(src, dst)
    assert dst.exists()


def test_empty_baselines_handled(tmp_path):
    """A baseline with no subsets yields zero failure rate and None worst group."""
    summary = _make_summary()
    summary["baselines"]["empty"] = []
    src = tmp_path / "summary.json"
    src.write_text(json.dumps(summary))
    dst = tmp_path / "stress_matrix.json"
    _run_build(src, dst)

    out = json.loads(dst.read_text())
    assert out["stress_matrix"]["empty"]["worst_group"] is None
    assert out["stress_matrix"]["empty"]["worst_group_delta"] is None
    assert out["stress_matrix"]["empty"]["failure_rate"] == 0.0
    assert out["stress_matrix"]["empty"]["n_subsets"] == 0
