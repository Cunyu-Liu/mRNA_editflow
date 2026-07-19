"""Offline tests for the leakage-gated frozen-backbone comparison.

Run:
    /Users/bytedance/Documents/research/editflow/.venv/bin/python \
        -m unittest mrna_editflow.tests.test_frozen_backbone_comparison -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mrna_editflow.core.config import MEFConfig
from mrna_editflow.data.download_mrna import synthesize_corpus
from mrna_editflow.eval.frozen_backbone_comparison import (
    DEFAULT_BACKBONES,
    run_frozen_backbone_comparison,
    write_comparison_report,
)


def _tiny_config() -> MEFConfig:
    cfg = MEFConfig()
    cfg.model.model_dim = 32
    cfg.model.num_layers = 2
    cfg.model.num_heads = 2
    cfg.model.max_seq_len = 256
    cfg.backbone.hidden_dim = 32
    cfg.train.batch_size = 2
    cfg.train.amp = False
    return cfg


class TestFrozenBackboneComparison(unittest.TestCase):
    def test_matched_budget_and_real_stub_tagging(self):
        cfg = _tiny_config()
        res = run_frozen_backbone_comparison(
            query_records=[],
            reference_records=None,
            backbones=("none", "helix_mrna", "mrnabert"),
            base_config=cfg,
            hidden_dim=32,
            steps=2,
            synthetic_n=4,
            seed=0,
            device="cpu",
        )
        runs = res["runs"]
        self.assertEqual(len(runs), 3)
        # Matched budget: identical trainable params across every arm.
        counts = {r["trainable_params"] for r in runs}
        self.assertEqual(len(counts), 1)
        self.assertTrue(res["matched_budget"]["trainable_params_consistent"])
        self.assertEqual(res["matched_budget"]["trainable_params"], next(iter(counts)))
        by_name = {r["backbone"]: r for r in runs}
        # ``none`` is the genuine encoder; externals are stubs offline.
        self.assertTrue(by_name["none"]["is_real"])
        self.assertTrue(by_name["none"]["valid_quality_signal"])
        self.assertEqual(by_name["none"]["kind"], "real")
        for stub in ("helix_mrna", "mrnabert"):
            self.assertFalse(by_name[stub]["is_real"])
            self.assertFalse(by_name[stub]["valid_quality_signal"])
            self.assertEqual(by_name[stub]["kind"], "adapter-stub")
            self.assertIn("SOTA", by_name[stub]["note"])
        self.assertEqual(res["n_real_arms"], 1)
        self.assertEqual(res["n_stub_arms"], 2)

    def test_all_runs_have_finite_loss(self):
        cfg = _tiny_config()
        res = run_frozen_backbone_comparison(
            query_records=[],
            reference_records=None,
            backbones=("none", "helix_mrna"),
            base_config=cfg,
            hidden_dim=32,
            steps=2,
            synthetic_n=4,
            seed=1,
            device="cpu",
        )
        for r in res["runs"]:
            self.assertTrue(r["finite_loss"], f"non-finite loss for {r['backbone']}")

    def test_leakage_gate_skipped_when_no_reference(self):
        cfg = _tiny_config()
        res = run_frozen_backbone_comparison(
            query_records=[],
            reference_records=None,
            backbones=("none",),
            base_config=cfg,
            hidden_dim=32,
            steps=1,
            synthetic_n=4,
            seed=0,
            device="cpu",
        )
        gate = res["leakage_gate"]
        self.assertFalse(gate["enabled"])
        self.assertFalse(gate["audited"])
        # Skipped gate does not block training arms.
        self.assertEqual(len(res["runs"]), 1)
        self.assertIn("NOT audited", res["interpretation"])

    def test_leakage_gate_flags_self_overlap_and_refuses(self):
        cfg = _tiny_config()
        corpus = synthesize_corpus(6, seed=7)
        query = corpus[:3]  # exact subset -> guaranteed leakage
        res = run_frozen_backbone_comparison(
            query_records=query,
            reference_records=corpus,
            backbones=("none",),
            base_config=cfg,
            hidden_dim=32,
            steps=1,
            synthetic_n=4,
            seed=0,
            device="cpu",
            require_gate=True,
        )
        gate = res["leakage_gate"]
        self.assertTrue(gate["enabled"])
        self.assertTrue(gate["audited"])
        self.assertFalse(gate["passed"])
        self.assertGreaterEqual(gate["exact_match_count"], 3)
        # Fair-comparison refusal: no training arms are executed.
        self.assertEqual(len(res["runs"]), 0)
        self.assertIsNotNone(res["skipped_reason"])
        self.assertIn("FAILED", res["interpretation"])

    def test_leakage_gate_passes_on_disjoint_corpora(self):
        cfg = _tiny_config()
        query = synthesize_corpus(3, seed=101)
        reference = synthesize_corpus(3, seed=202)
        res = run_frozen_backbone_comparison(
            query_records=query,
            reference_records=reference,
            backbones=("none", "mrnabert"),
            base_config=cfg,
            hidden_dim=32,
            steps=1,
            synthetic_n=4,
            seed=0,
            device="cpu",
        )
        gate = res["leakage_gate"]
        self.assertTrue(gate["enabled"])
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["exact_match_count"], 0)
        self.assertEqual(len(res["runs"]), 2)
        self.assertIn("PASSED", res["interpretation"])

    def test_require_gate_false_runs_despite_leakage(self):
        cfg = _tiny_config()
        corpus = synthesize_corpus(6, seed=7)
        query = corpus[:3]
        res = run_frozen_backbone_comparison(
            query_records=query,
            reference_records=corpus,
            backbones=("none",),
            base_config=cfg,
            hidden_dim=32,
            steps=1,
            synthetic_n=4,
            seed=0,
            device="cpu",
            require_gate=False,
        )
        self.assertFalse(res["leakage_gate"]["passed"])
        # Diagnostic mode: training proceeds even though the gate failed.
        self.assertEqual(len(res["runs"]), 1)
        self.assertIsNone(res["skipped_reason"])

    def test_unknown_backbone_raises(self):
        cfg = _tiny_config()
        with self.assertRaises(ValueError):
            run_frozen_backbone_comparison(
                query_records=[],
                backbones=("none", "definitely_not_a_backbone"),
                base_config=cfg,
                steps=1,
                device="cpu",
            )

    def test_report_writer_emits_json_and_markdown(self):
        cfg = _tiny_config()
        res = run_frozen_backbone_comparison(
            query_records=[],
            reference_records=None,
            backbones=("none", "helix_mrna"),
            base_config=cfg,
            hidden_dim=32,
            steps=1,
            synthetic_n=4,
            seed=0,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as d:
            out_json = os.path.join(d, "cmp.json")
            out_md = os.path.join(d, "cmp.md")
            oj, om = write_comparison_report(res, out_json, out_md)
            self.assertTrue(os.path.exists(oj))
            self.assertTrue(os.path.exists(om))
            with open(om, encoding="utf-8") as fh:
                md = fh.read()
            self.assertIn("Frozen-Backbone Adapter Comparison", md)
            self.assertIn("Matched-budget runs", md)
            self.assertIn("adapter-stub", md)

    def test_default_backbones_include_real_reference(self):
        self.assertIn("none", DEFAULT_BACKBONES)


if __name__ == "__main__":
    unittest.main()
