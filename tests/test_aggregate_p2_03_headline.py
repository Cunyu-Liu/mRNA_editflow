"""Tests for P2-03 headline aggregation script (scripts/aggregate_p2_03_headline.py)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

# Make scripts/ importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_PARENT = _REPO_ROOT.parent
for _p in (str(_PACKAGE_PARENT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "aggregate_p2_03_headline",
    _REPO_ROOT / "scripts" / "aggregate_p2_03_headline.py",
)
assert _spec is not None and _spec.loader is not None
agg = importlib.util.module_from_spec(_spec)
sys.modules["aggregate_p2_03_headline"] = agg
_spec.loader.exec_module(agg)


def _make_eval_summary(
    delta: float,
    oracle_te: float,
    edit_dist: float,
    mfe: float,
) -> Dict[str, Any]:
    """Build a minimal eval_summary.json for testing."""
    return {
        "bootstrap_ci": {
            "oracle_ensemble_te": {"mean": oracle_te, "low": oracle_te - 0.01, "high": oracle_te + 0.01, "n": 1024},
            "edit_distance": {"mean": edit_dist, "low": edit_dist - 0.1, "high": edit_dist + 0.1, "n": 1024},
            "mfe_proxy": {"mean": mfe, "low": mfe - 1.0, "high": mfe + 1.0, "n": 1024},
            "delta_oracle_te_vs_source": {"mean": delta, "low": delta - 0.001, "high": delta + 0.001, "n": 1024},
        },
        "metrics": {
            "oracle_ensemble_te": oracle_te,
            "edit_distance": edit_dist,
            "mfe_proxy": mfe,
        },
        "n_candidates": 1024,
        "n_sources": 1024,
    }


def _make_baseline_dir(
    root: Path,
    name: str,
    n_seeds: int = 10,
    delta_base: float = 0.005,
    delta_noise: float = 0.001,
    oracle_te: float = 0.8,
    edit_dist: float = 3.0,
    mfe: float = -45.0,
) -> Path:
    """Create a baseline directory with n_seeds seed dirs + eval_summary.json."""
    bd = root / name
    bd.mkdir(parents=True, exist_ok=True)
    for i in range(n_seeds):
        sd = bd / f"seed_{i:03d}"
        sd.mkdir(parents=True, exist_ok=True)
        delta = delta_base + delta_noise * (i - n_seeds / 2) / n_seeds
        ev = _make_eval_summary(delta, oracle_te, edit_dist, mfe)
        with (sd / "eval_summary.json").open("w") as fh:
            json.dump(ev, fh)
    return bd


class TestExtractMetric(unittest.TestCase):
    def test_extract_from_bootstrap_ci(self):
        ev = _make_eval_summary(0.005, 0.8, 3.0, -45.0)
        self.assertAlmostEqual(agg.extract_metric(ev, "oracle_ensemble_te"), 0.8)

    def test_extract_from_metrics(self):
        ev = {"metrics": {"oracle_ensemble_te": 0.75}, "bootstrap_ci": {}}
        self.assertAlmostEqual(agg.extract_metric(ev, "oracle_ensemble_te"), 0.75)

    def test_returns_none_if_missing(self):
        ev = {"metrics": {}, "bootstrap_ci": {}}
        self.assertIsNone(agg.extract_metric(ev, "nonexistent"))

    def test_extract_delta_from_bootstrap_ci(self):
        ev = _make_eval_summary(0.005, 0.8, 3.0, -45.0)
        self.assertAlmostEqual(agg.extract_delta(ev), 0.005)

    def test_extract_delta_returns_none_if_missing(self):
        ev = {"metrics": {}, "bootstrap_ci": {}}
        self.assertIsNone(agg.extract_delta(ev))


class TestLoadProgressDeltas(unittest.TestCase):
    """Tests for load_progress_deltas (reads multiseed_progress.jsonl)."""

    def test_loads_deltas_from_progress_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            bd = Path(td) / "baseline_seed0"
            bd.mkdir()
            prog = bd / "multiseed_progress.jsonl"
            lines = [
                json.dumps({"decoder_seed": 0, "event": "seed_start"}),
                json.dumps({"decoder_seed": 0, "delta_oracle_te_vs_source": 0.003, "event": "seed_evaluated"}),
                json.dumps({"decoder_seed": 1, "event": "seed_start"}),
                json.dumps({"decoder_seed": 1, "delta_oracle_te_vs_source": 0.005, "event": "seed_evaluated"}),
            ]
            prog.write_text("\n".join(lines))
            deltas = agg.load_progress_deltas(bd)
            self.assertEqual(deltas[0], 0.003)
            self.assertEqual(deltas[1], 0.005)

    def test_returns_empty_if_no_progress_file(self):
        with tempfile.TemporaryDirectory() as td:
            bd = Path(td) / "baseline_seed0"
            bd.mkdir()
            deltas = agg.load_progress_deltas(bd)
            self.assertEqual(deltas, {})

    def test_skips_malformed_lines(self):
        with tempfile.TemporaryDirectory() as td:
            bd = Path(td) / "baseline_seed0"
            bd.mkdir()
            prog = bd / "multiseed_progress.jsonl"
            prog.write_text(
                json.dumps({"decoder_seed": 0, "delta_oracle_te_vs_source": 0.003}) + "\n"
                + "not valid json\n"
                + json.dumps({"decoder_seed": 1, "delta_oracle_te_vs_source": 0.005}) + "\n"
            )
            deltas = agg.load_progress_deltas(bd)
            self.assertEqual(len(deltas), 2)
            self.assertAlmostEqual(deltas[0], 0.003)
            self.assertAlmostEqual(deltas[1], 0.005)


class TestProgressDeltaFallback(unittest.TestCase):
    """Test that aggregate_baseline uses progress deltas when eval_summary lacks them."""

    def test_falls_back_to_progress_deltas(self):
        with tempfile.TemporaryDirectory() as td:
            bd = Path(td) / "te_only_seed0"
            bd.mkdir()
            # Write eval_summary WITHOUT delta_oracle_te_vs_source.
            for i in range(5):
                sd = bd / f"seed_{i:03d}"
                sd.mkdir()
                ev = {
                    "bootstrap_ci": {
                        "oracle_ensemble_te": {"mean": 0.8, "low": 0.79, "high": 0.81, "n": 1024},
                    },
                    "metrics": {"oracle_ensemble_te": 0.8},
                }
                (sd / "eval_summary.json").write_text(json.dumps(ev))
            # Write progress JSONL WITH deltas.
            prog_lines = [
                json.dumps({"decoder_seed": i, "delta_oracle_te_vs_source": 0.003 + i * 0.001})
                for i in range(5)
            ]
            (bd / "multiseed_progress.jsonl").write_text("\n".join(prog_lines))
            result = agg.aggregate_baseline(bd)
            self.assertIsNotNone(result)
            self.assertEqual(result["n_seeds"], 5)
            # Primary should be from progress deltas.
            self.assertAlmostEqual(result["primary"]["mean"], 0.003 + 2 * 0.001, places=4)


class TestAggregateBaseline(unittest.TestCase):
    def setUp(self):
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmpdir_obj.name)

    def tearDown(self):
        self.tmpdir_obj.cleanup()

    def test_aggregate_10_seeds(self):
        bd = _make_baseline_dir(self.tmpdir, "te_only_seed0", n_seeds=10, delta_base=0.003)
        result = agg.aggregate_baseline(bd)
        self.assertIsNotNone(result)
        self.assertEqual(result["baseline"], "te_only_seed0")
        self.assertEqual(result["n_seeds"], 10)
        self.assertAlmostEqual(result["primary"]["mean"], 0.003, places=4)
        self.assertIn("ci_low", result["primary"])
        self.assertIn("ci_high", result["primary"])
        self.assertLess(result["primary"]["ci_low"], result["primary"]["mean"])
        self.assertGreater(result["primary"]["ci_high"], result["primary"]["mean"])

    def test_aggregate_empty_dir(self):
        bd = self.tmpdir / "empty_seed0"
        bd.mkdir()
        result = agg.aggregate_baseline(bd)
        self.assertIsNone(result)

    def test_aggregate_missing_eval_files(self):
        bd = self.tmpdir / "bad_seed0"
        bd.mkdir()
        (bd / "seed_000").mkdir()
        # No eval_summary.json in seed_000.
        result = agg.aggregate_baseline(bd)
        self.assertIsNone(result)

    def test_secondary_endpoints_collected(self):
        bd = _make_baseline_dir(self.tmpdir, "scalar_seed0", n_seeds=5)
        result = agg.aggregate_baseline(bd)
        self.assertIn("oracle_ensemble_te", result["secondary"])
        self.assertIn("edit_distance", result["secondary"])
        self.assertIn("mfe_proxy", result["secondary"])
        self.assertEqual(result["secondary"]["oracle_ensemble_te"]["n"], 5)


class TestPairwiseCompare(unittest.TestCase):
    def setUp(self):
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmpdir_obj.name)

    def tearDown(self):
        self.tmpdir_obj.cleanup()

    def test_paired_comparison_same_length(self):
        b1 = agg.aggregate_baseline(_make_baseline_dir(self.tmpdir, "a_seed0", n_seeds=10, delta_base=0.003))
        b2 = agg.aggregate_baseline(_make_baseline_dir(self.tmpdir, "b_seed0", n_seeds=10, delta_base=0.006))
        cmp = agg.pairwise_compare(b1, b2)
        self.assertIsNotNone(cmp)
        self.assertEqual(cmp["baseline_a"], "a_seed0")
        self.assertEqual(cmp["baseline_b"], "b_seed0")
        self.assertGreater(cmp["diff_mean"], 0)  # b > a
        self.assertEqual(cmp["test_type"], "paired_bootstrap")
        self.assertIn("p_value", cmp)
        self.assertIn("significant", cmp)

    def test_unpaired_comparison_different_length(self):
        b1 = agg.aggregate_baseline(_make_baseline_dir(self.tmpdir, "a_seed0", n_seeds=10, delta_base=0.003))
        b2 = agg.aggregate_baseline(_make_baseline_dir(self.tmpdir, "b_seed0", n_seeds=8, delta_base=0.006))
        cmp = agg.pairwise_compare(b1, b2)
        self.assertIsNotNone(cmp)
        self.assertEqual(cmp["test_type"], "unpaired_bootstrap")

    def test_significant_difference(self):
        # Large difference should be significant.
        b1 = agg.aggregate_baseline(_make_baseline_dir(self.tmpdir, "a_seed0", n_seeds=10, delta_base=0.001, delta_noise=0.0001))
        b2 = agg.aggregate_baseline(_make_baseline_dir(self.tmpdir, "b_seed0", n_seeds=10, delta_base=0.020, delta_noise=0.0001))
        cmp = agg.pairwise_compare(b1, b2)
        self.assertTrue(cmp["significant"])
        self.assertLess(cmp["p_value"], 0.05)


class TestAggregateAll(unittest.TestCase):
    def setUp(self):
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmpdir_obj.name)

    def tearDown(self):
        self.tmpdir_obj.cleanup()

    def test_aggregate_multiple_baselines(self):
        _make_baseline_dir(self.tmpdir, "te_only_seed0", n_seeds=10, delta_base=0.003)
        _make_baseline_dir(self.tmpdir, "scalar_seed0", n_seeds=10, delta_base=0.006)
        _make_baseline_dir(self.tmpdir, "pareto_seed0", n_seeds=10, delta_base=0.005)
        result = agg.aggregate_all(self.tmpdir)
        self.assertEqual(result["n_baselines"], 3)
        self.assertEqual(len(result["baselines"]), 3)
        self.assertEqual(len(result["pairwise_comparisons"]), 3)  # C(3,2) = 3 pairs
        # Ranking should put scalar first (highest delta).
        self.assertEqual(result["ranked_by_primary"][0], "scalar_seed0")

    def test_returns_error_on_empty_root(self):
        empty = self.tmpdir / "empty"
        empty.mkdir()
        result = agg.aggregate_all(empty)
        self.assertIn("error", result)

    def test_includes_qualifier(self):
        _make_baseline_dir(self.tmpdir, "te_only_seed0", n_seeds=5)
        result = agg.aggregate_all(self.tmpdir)
        self.assertEqual(result["primary_qualifier"], "predicted_te_internal_proxy")
        self.assertEqual(result["baselines"][0]["primary"]["qualifier"], "predicted_te_internal_proxy")


class TestWriteMarkdown(unittest.TestCase):
    def setUp(self):
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmpdir_obj.name)

    def tearDown(self):
        self.tmpdir_obj.cleanup()

    def test_markdown_contains_ranking_and_comparisons(self):
        _make_baseline_dir(self.tmpdir, "a_seed0", n_seeds=10, delta_base=0.003)
        _make_baseline_dir(self.tmpdir, "b_seed0", n_seeds=10, delta_base=0.006)
        result = agg.aggregate_all(self.tmpdir)
        out_md = self.tmpdir / "summary.md"
        agg.write_markdown(result, out_md)
        content = out_md.read_text()
        self.assertIn("# P2-03: Leakage-Free Headline", content)
        self.assertIn("a_seed0", content)
        self.assertIn("b_seed0", content)
        self.assertIn("Ranking", content)
        self.assertIn("Pairwise", content)
        self.assertIn("predicted_te_internal_proxy", content)


class TestEndToEnd(unittest.TestCase):
    """End-to-end test via main()."""
    def setUp(self):
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmpdir_obj.name)

    def tearDown(self):
        self.tmpdir_obj.cleanup()

    def test_main_creates_outputs(self):
        _make_baseline_dir(self.tmpdir, "te_only_seed0", n_seeds=10, delta_base=0.003)
        _make_baseline_dir(self.tmpdir, "scalar_seed0", n_seeds=10, delta_base=0.006)
        out_json = self.tmpdir / "headline.json"
        out_md = self.tmpdir / "summary.md"

        # Patch sys.argv.
        old_argv = sys.argv
        sys.argv = [
            "aggregate_p2_03_headline.py",
            "--root", str(self.tmpdir),
            "--out-json", str(out_json),
            "--out-md", str(out_md),
        ]
        try:
            rc = agg.main()
        finally:
            sys.argv = old_argv

        self.assertEqual(rc, 0)
        self.assertTrue(out_json.exists())
        self.assertTrue(out_md.exists())
        data = json.loads(out_json.read_text())
        self.assertEqual(data["n_baselines"], 2)


if __name__ == "__main__":
    unittest.main()
