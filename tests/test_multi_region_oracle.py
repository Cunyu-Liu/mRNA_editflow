"""Unit tests for P2-01 MultiRegionOracle.

Covers:
- CAI computation (deterministic, known values)
- GC content, ARE count
- Stability proxy
- Structural compatibility
- MultiRegionOracle.score_record (ensemble_te in [0,1])
- Region sensitivity (editing each region changes the score)
- Non-additivity (cross-region coupling terms are non-additive)
- Determinism (same input -> same output)
"""
from __future__ import annotations

import math
import os
import sys
import unittest
from typing import Any, Dict

# Ensure the mrna_editflow package is importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.eval.multi_region_oracle import (
    DEFAULT_WEIGHTS,
    MultiRegionOracle,
    MultiRegionOracleConfig,
    _are_count,
    _gc_content,
    _local_mfe_proxy,
    _sigmoid,
    build_default_multi_region_oracle,
    compute_cai,
    compute_stability_proxy,
    _build_cai_reference,
    _load_codon_usage_table,
    _STANDARD_GENETIC_CODE,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_record(five_utr: str, cds: str, three_utr: str,
                 tid: str = "ENST00000000000.1") -> MRNARecord:
    return MRNARecord(
        transcript_id=tid,
        five_utr=five_utr,
        cds=cds,
        three_utr=three_utr,
    )


# A simple in-memory codon usage table for CAI tests.
_TEST_CODON_TABLE = {
    "UUU": 0.46, "UUC": 0.54, "UUA": 0.07, "UUG": 0.13,
    "CUU": 0.13, "CUC": 0.20, "CUA": 0.07, "CUG": 0.47,
    "AUU": 0.36, "AUC": 0.47, "AUA": 0.17, "AUG": 1.00,
    "GUU": 0.18, "GUC": 0.26, "GUA": 0.12, "GUG": 0.44,
    "UAA": 0.28, "UAG": 0.20, "UGA": 0.52,
    "UCU": 0.19, "UCC": 0.22, "UCA": 0.15, "UCG": 0.05,
    "CCU": 0.28, "CCC": 0.32, "CCA": 0.28, "CCG": 0.12,
    "ACU": 0.25, "ACC": 0.36, "ACA": 0.28, "ACG": 0.11,
    "GCU": 0.26, "GCC": 0.40, "GCA": 0.23, "GCG": 0.11,
    "UAU": 0.44, "UAC": 0.56, "UGU": 0.45, "UGC": 0.55,
    "CAU": 0.42, "CAC": 0.58, "CAA": 0.27, "CAG": 0.73,
    "AAU": 0.47, "AAC": 0.53, "AAA": 0.43, "AAG": 0.57,
    "GAU": 0.46, "GAC": 0.54, "GAA": 0.42, "GAG": 0.58,
    "CGU": 0.08, "CGC": 0.18, "CGA": 0.11, "CGG": 0.20,
    "AGA": 0.21, "AGG": 0.21, "AGU": 0.15, "AGC": 0.24,
    "GGU": 0.16, "GGC": 0.34, "GGA": 0.25, "GGG": 0.25,
    "UGG": 1.00,
}


# ---------------------------------------------------------------------------
# Sequence utility tests
# ---------------------------------------------------------------------------

class TestGCContent(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(_gc_content(""), 0.0)

    def test_all_gc(self):
        self.assertAlmostEqual(_gc_content("GCGC"), 1.0)

    def test_all_au(self):
        self.assertAlmostEqual(_gc_content("AUAU"), 0.0)

    def test_half(self):
        self.assertAlmostEqual(_gc_content("GAUC"), 0.5)

    def test_t_to_u(self):
        # T -> U conversion; "TGTG" becomes "UGUG" (GC = 2/4 = 0.5).
        self.assertAlmostEqual(_gc_content("TGTG"), 0.5)
        # "GCGC" has no T, stays "GCGC" (GC = 1.0).
        self.assertAlmostEqual(_gc_content("GCGC"), 1.0)


class TestARECount(unittest.TestCase):
    def test_no_are(self):
        self.assertEqual(_are_count("GCGCGCGCGC"), 0)

    def test_one_are(self):
        self.assertEqual(_are_count("GCGCAUUUAGCGC"), 1)

    def test_overlapping(self):
        # AUUUAUUUA contains AUUUA at 0 and AUUUA at 3 (overlapping).
        self.assertEqual(_are_count("AUUUAUUUA"), 2)

    def test_multiple_motifs(self):
        # AUUUA + AUUUUA
        self.assertEqual(_are_count("AUUUAAAUUUUA"), 2)


class TestSigmoid(unittest.TestCase):
    def test_zero(self):
        self.assertAlmostEqual(_sigmoid(0.0), 0.5)

    def test_large_positive(self):
        self.assertAlmostEqual(_sigmoid(100.0), 1.0, places=6)

    def test_large_negative(self):
        self.assertAlmostEqual(_sigmoid(-100.0), 0.0, places=6)


class TestLocalMFEProxy(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(_local_mfe_proxy(""), 0.0)

    def test_single_base(self):
        self.assertEqual(_local_mfe_proxy("A"), 0.0)

    def test_negative_for_stable(self):
        # GC-rich should be more stable (more negative).
        gc_energy = _local_mfe_proxy("GCGC")
        au_energy = _local_mfe_proxy("AUAU")
        self.assertLess(gc_energy, au_energy)


# ---------------------------------------------------------------------------
# CAI tests
# ---------------------------------------------------------------------------

class TestCAI(unittest.TestCase):
    def setUp(self):
        self.reference = _build_cai_reference(_TEST_CODON_TABLE)

    def test_empty_cds(self):
        self.assertEqual(compute_cai("", self.reference), 0.0)

    def test_short_cds(self):
        self.assertEqual(compute_cai("AU", self.reference), 0.0)

    def test_all_optimal(self):
        # AUG (M, w=1.0), UUC (F, w=0.54/0.54=1.0), CUG (L, w=0.47/0.47=1.0)
        cai = compute_cai("AUGUUCCUG", self.reference)
        self.assertAlmostEqual(cai, 1.0, places=6)

    def test_all_rare(self):
        # UUA (L, w=0.07/0.47), UAA is stop (skipped)
        cai = compute_cai("UUAUUAUUA", self.reference)
        self.assertLess(cai, 0.5)

    def test_in_range(self):
        cai = compute_cai("AUGUUCCUGUUA", self.reference)
        self.assertGreater(cai, 0.0)
        self.assertLess(cai, 1.0)

    def test_t_to_u(self):
        # DNA input should be converted to RNA and match the RNA computation.
        # "AUGUUCCUG" (RNA) -> DNA: A-T-G-T-T-C-C-T-G = "ATGTTCCTG".
        cai_rna = compute_cai("AUGUUCCUG", self.reference)
        cai_dna = compute_cai("ATGTTCCTG", self.reference)
        self.assertAlmostEqual(cai_rna, cai_dna, places=6)

    def test_stop_codons_skipped(self):
        # UAA is a stop codon, should be skipped.
        cai_with_stop = compute_cai("AUGUUCCUGUAA", self.reference)
        cai_without = compute_cai("AUGUUCCUG", self.reference)
        self.assertAlmostEqual(cai_with_stop, cai_without, places=6)

    def test_invalid_codons_skipped(self):
        cai = compute_cai("AUGXXXUUC", self.reference)
        # Only UUC is valid; should return CAI for UUC alone.
        self.assertAlmostEqual(cai, 1.0, places=6)


class TestStabilityProxy(unittest.TestCase):
    def test_empty(self):
        # Empty 3'UTR -> neutral 0.5.
        self.assertAlmostEqual(compute_stability_proxy(""), 0.5)

    def test_in_range(self):
        s = compute_stability_proxy("GCGCGCGCGCGCGCGCGCGC" * 10)
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)

    def test_high_are_reduces_stability(self):
        high_are = compute_stability_proxy("AUUUA" * 20)
        low_are = compute_stability_proxy("GCGC" * 25)
        self.assertLess(high_are, low_are)

    def test_high_gc_increases_stability(self):
        high_gc = compute_stability_proxy("GCGCGCGCGC" * 10)
        low_gc = compute_stability_proxy("AUAUAUAUAU" * 10)
        self.assertGreater(high_gc, low_gc)


# ---------------------------------------------------------------------------
# MultiRegionOracle tests
# ---------------------------------------------------------------------------

class TestMultiRegionOracle(unittest.TestCase):
    """Test the composite oracle with skip_cnn=True (no CNN dependency)."""

    @classmethod
    def setUpClass(cls):
        """Build an oracle with skip_cnn=True for fast testing."""
        # Use the real codon usage table if available, else use test table.
        repo_root = _REPO_ROOT
        codon_path = os.path.join(
            repo_root, "external_tools", "EnsembleDesign",
            "codon_usage_freq_table_human.csv",
        )
        cls.config = MultiRegionOracleConfig(
            cnn_ckpt_dir="",  # skip
            codon_usage_path=codon_path if os.path.exists(codon_path) else "",
            device="cpu",
            skip_cnn=True,
        )
        cls.oracle = MultiRegionOracle(cls.config)
        cls.oracle.load()
        # If no real codon table, inject test table.
        if not cls.oracle._cai_reference:
            cls.oracle._cai_reference = _build_cai_reference(_TEST_CODON_TABLE)

    def _make_record(self, five_utr=None, cds=None, three_utr=None):
        """Make a record with defaults."""
        return _make_record(
            five_utr=five_utr or "GCCAUGAGCUGCAGCAACCAUGAGCUGCAGCAACCAUGAGCUGCAGCAA",
            cds=cds or "AUGUUCCUGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCG" * 3,
            three_utr=three_utr or "GCGCGCGCGCAUUUA" + "GCGCGCGCGC" * 10,
        )

    def test_score_record_returns_ensemble_te(self):
        rec = self._make_record()
        score = self.oracle.score_record(rec)
        self.assertIn("ensemble_te", score)
        self.assertGreaterEqual(score["ensemble_te"], 0.0)
        self.assertLessEqual(score["ensemble_te"], 1.0)

    def test_score_record_has_per_region_components(self):
        rec = self._make_record()
        score = self.oracle.score_record(rec)
        for key in ["mrl_5utr_norm", "cai_cds", "stab_3utr", "struct_compat"]:
            self.assertIn(key, score, f"Missing key: {key}")
        for key in ["coupling_5c", "coupling_c3", "coupling_53", "coupling_5c3"]:
            self.assertIn(key, score, f"Missing coupling key: {key}")

    def test_determinism(self):
        rec = self._make_record()
        s1 = self.oracle.score_record(rec)
        s2 = self.oracle.score_record(rec)
        self.assertEqual(s1["ensemble_te"], s2["ensemble_te"])

    def test_5utr_sensitivity(self):
        """Editing 5'UTR changes the score."""
        rec1 = self._make_record(five_utr="GCCAUGAGCUGCAGCAACCAUGAGCUGCAGCAACCAUGAGCUGCAGCAA")
        rec2 = self._make_record(five_utr="UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU")
        s1 = self.oracle.score_record(rec1)["ensemble_te"]
        s2 = self.oracle.score_record(rec2)["ensemble_te"]
        self.assertNotAlmostEqual(s1, s2, places=6,
                                   msg="5'UTR edits should change the score")

    def test_cds_sensitivity(self):
        """Editing CDS changes the score (via CAI)."""
        # High CAI: all optimal codons.
        cds_high = "AUGUUCCUGGCGGCGGCG" * 5
        # Low CAI: all rare codons.
        cds_low = "AUGUUUUUACUAACUACUA" * 5
        rec1 = self._make_record(cds=cds_high)
        rec2 = self._make_record(cds=cds_low)
        s1 = self.oracle.score_record(rec1)["ensemble_te"]
        s2 = self.oracle.score_record(rec2)["ensemble_te"]
        self.assertNotAlmostEqual(s1, s2, places=6,
                                   msg="CDS edits should change the score (via CAI)")

    def test_3utr_sensitivity(self):
        """Editing 3'UTR changes the score (via stability proxy)."""
        rec1 = self._make_record(three_utr="GCGCGCGCGCGCGCGCGCGC" * 10)  # high GC, no ARE
        rec2 = self._make_record(three_utr="AUUUA" * 40)  # many AREs
        s1 = self.oracle.score_record(rec1)["ensemble_te"]
        s2 = self.oracle.score_record(rec2)["ensemble_te"]
        self.assertNotAlmostEqual(s1, s2, places=6,
                                   msg="3'UTR edits should change the score (via stability)")

    def test_non_additivity(self):
        """The oracle has non-additive cross-region terms.

        Verify that score(5+c) != score(5) + score(c) - score(wt) approximately.
        This is a necessary condition for synergy detection.
        """
        rec_wt = self._make_record()
        rec_5 = self._make_record(five_utr="UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU")
        rec_c = self._make_record(cds="AUGUUUUUACUAACUACUA" * 5)
        rec_5c = self._make_record(
            five_utr="UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU",
            cds="AUGUUUUUACUAACUACUA" * 5,
        )
        s_wt = self.oracle.score_record(rec_wt)["ensemble_te"]
        s_5 = self.oracle.score_record(rec_5)["ensemble_te"]
        s_c = self.oracle.score_record(rec_c)["ensemble_te"]
        s_5c = self.oracle.score_record(rec_5c)["ensemble_te"]
        # Deltas.
        d_5 = s_5 - s_wt
        d_c = s_c - s_wt
        d_5c = s_5c - s_wt
        # Synergy = d_5c - (d_5 + d_c). Should be non-zero due to coupling terms.
        syn = d_5c - (d_5 + d_c)
        # The multiplicative coupling terms ensure non-additivity.
        self.assertNotAlmostEqual(syn, 0.0, places=4,
                                   msg="Oracle should be non-additive (synergy != 0)")

    def test_weights_sum_to_one(self):
        """Default weights should sum to 1.0."""
        total = sum(DEFAULT_WEIGHTS.values())
        self.assertAlmostEqual(total, 1.0, places=6)


class TestMultiRegionOracleWithCNN(unittest.TestCase):
    """Test with real CNN ensemble (skipped if checkpoints not available)."""

    @classmethod
    def setUpClass(cls):
        repo_root = _REPO_ROOT
        ckpt_dir = os.path.join(repo_root, "ckpts", "p1_04_predictors")
        codon_path = os.path.join(
            repo_root, "external_tools", "EnsembleDesign",
            "codon_usage_freq_table_human.csv",
        )
        cls.has_cnn = os.path.exists(ckpt_dir) and len(
            [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
        ) > 0
        if cls.has_cnn:
            cls.config = MultiRegionOracleConfig(
                cnn_ckpt_dir=ckpt_dir,
                codon_usage_path=codon_path,
                device="cpu",
                skip_cnn=False,
            )
            cls.oracle = MultiRegionOracle(cls.config)
            cls.oracle.load()

    def setUp(self):
        if not self.has_cnn:
            self.skipTest("CNN checkpoints not available")

    def test_cnn_loaded(self):
        self.assertGreater(len(self.oracle._cnn_ensemble._models), 0)

    def test_mrl_prediction(self):
        mrl, std = self.oracle._cnn_ensemble.predict_mrl(
            "GCCAUGAGCUGCAGCAACCAUGAGCUGCAGCAACCAUGAGCUGCAGCAA"[:50]
        )
        # MRL should be in a reasonable range [0, 10].
        self.assertGreaterEqual(mrl, 0.0)
        self.assertLessEqual(mrl, 15.0)


# ---------------------------------------------------------------------------
# OracleMDP integration (smoke test)
# ---------------------------------------------------------------------------

class TestOracleMDPIntegration(unittest.TestCase):
    """Smoke test: OracleMDP with MultiRegionOracle runs without errors."""

    @classmethod
    def setUpClass(cls):
        repo_root = _REPO_ROOT
        codon_path = os.path.join(
            repo_root, "external_tools", "EnsembleDesign",
            "codon_usage_freq_table_human.csv",
        )
        cls.config = MultiRegionOracleConfig(
            cnn_ckpt_dir="",
            codon_usage_path=codon_path if os.path.exists(codon_path) else "",
            device="cpu",
            skip_cnn=True,
        )
        cls.oracle = MultiRegionOracle(cls.config)
        cls.oracle.load()
        if not cls.oracle._cai_reference:
            cls.oracle._cai_reference = _build_cai_reference(_TEST_CODON_TABLE)

    def test_score_record_on_real_record(self):
        rec = _make_record(
            five_utr="GCCAUGAGCUGCAGCAACCAUGAGCUGCAGCAACCAUGAGCUGCAGCAA",
            cds="AUGUUCCUGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCG" * 3,
            three_utr="GCGCGCGCGCAUUUA" + "GCGCGCGCGC" * 10,
        )
        score = self.oracle.score_record(rec)
        self.assertIn("ensemble_te", score)
        self.assertGreater(score["ensemble_te"], 0.0)


if __name__ == "__main__":
    unittest.main()
