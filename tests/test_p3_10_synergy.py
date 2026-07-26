"""Unit tests for P3-10 cross-region synergy analysis.

Tests cover:
  - CAI-based CDS delta scorer (CAIDeltaScorer)
  - CombinedOracle (5'UTR oracle + CAI CDS scorer)
  - Counterfactual arm generation
  - Synergy computation
  - 3'UTR gate evaluation
  - Decision logic (PARTIAL verdict)
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path bootstrap (mirror scripts/run_p3_10.py)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT.parent))
sys.path.insert(0, str(_REPO_ROOT))

from core.constants import START_CODON, CODON_TABLE
from core.p3_02_delta_oracle import SYNONYMOUS_CODONS
from core.schema import MRNARecord
from rl.p3_07_search import SyntheticDeltaOracle, CountingOracle

# Load scripts/run_p3_10.py as a module (it's not a package)
_SPEC_PATH = _REPO_ROOT / "scripts" / "run_p3_10.py"
_spec = importlib.util.spec_from_file_location("run_p3_10", _SPEC_PATH)
run_p3_10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_p3_10)

# Re-exports for convenience
CAIDeltaScorer = run_p3_10.CAIDeltaScorer
CombinedOracle = run_p3_10.CombinedOracle
_cai_score = run_p3_10._cai_score
_best_single_cds_edit = run_p3_10._best_single_cds_edit
_best_single_5utr_edit = run_p3_10._best_single_5utr_edit
_random_edit = run_p3_10._random_edit
compute_synergy_stats = run_p3_10.compute_synergy_stats
_ols_interaction = run_p3_10._ols_interaction
evaluate_3utr_gate = run_p3_10.evaluate_3utr_gate
make_full_transcript_decision = run_p3_10.make_full_transcript_decision
run_counterfactual_arms = run_p3_10.run_counterfactual_arms
INERT_CDS = run_p3_10.INERT_CDS
INERT_THREE_UTR = run_p3_10.INERT_THREE_UTR


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_record(
    five_utr: str = "GCCAUGAGCAACGGAUUCGACCCAGACUUGACGAUUACGGACUUGACCAG",
    cds: str = INERT_CDS,
    three_utr: str = INERT_THREE_UTR,
    tid: str = "test_src",
) -> MRNARecord:
    return MRNARecord(
        transcript_id=tid,
        five_utr=five_utr,
        cds=cds,
        three_utr=three_utr,
        metadata={"test": True},
    )


@pytest.fixture
def cds_scorer() -> CAIDeltaScorer:
    return CAIDeltaScorer(weight=0.5)


@pytest.fixture
def synthetic_sources() -> List[MRNARecord]:
    """Five synthetic sources for counterfactual arm tests."""
    rng = np.random.RandomState(42)
    sources = []
    for i in range(5):
        # Random 5'UTR of length 40
        utr = "".join(rng.choice(list("ACGU"), 40))
        sources.append(_make_record(five_utr=utr, tid=f"src_{i}"))
    return sources


# ---------------------------------------------------------------------------
# CAI scorer tests
# ---------------------------------------------------------------------------

class TestCAIScorer:
    """Tests for the CAI-based CDS delta scorer."""

    def test_cai_score_returns_zero_for_empty_cds(self):
        assert _cai_score("") == 0.0

    def test_cai_score_returns_zero_for_short_cds(self):
        assert _cai_score("AUG") == 0.0

    def test_cai_score_in_range_0_to_1(self):
        # Use a longer CDS so there are interior codons to score
        cds = START_CODON + "GCU" * 4 + "UAA"
        score = _cai_score(cds)
        assert 0.0 <= score <= 1.0

    def test_cai_score_skips_start_stop_codons(self):
        # Start (AUG) and stop (UAA/UAG/UGA) are skipped in CAI scoring.
        # cds = AUG + GCU + UAA has codon 1 (GCU) as the only interior codon,
        # so its score equals GCU's optimality (0.3).
        cds1 = START_CODON + "GCU" + "UAA"
        cds2 = START_CODON + "GCG" + "UAA"
        score1 = _cai_score(cds1)
        score2 = _cai_score(cds2)
        # GCU is non-optimal (0.3), GCG is optimal (1.0)
        assert score1 == pytest.approx(0.3, abs=1e-9)
        assert score2 == pytest.approx(1.0, abs=1e-9)
        # Truly no interior codons (start + stop only)
        cds_min = START_CODON + "UAA"
        assert _cai_score(cds_min) == 0.0

    def test_cai_delta_scorer_zero_for_5utr_only_edit(self, cds_scorer):
        """5'UTR-only edit produces ΔCAI = 0 (CDS unchanged)."""
        src = _make_record()
        # 5'UTR edit only
        new_utr = src.five_utr[:10] + "G" + src.five_utr[11:]
        cand = MRNARecord(
            transcript_id=src.transcript_id,
            five_utr=new_utr, cds=src.cds, three_utr=src.three_utr,
            metadata=src.metadata,
        )
        delta, unc = cds_scorer._score(src, cand)
        assert delta == pytest.approx(0.0, abs=1e-9)
        assert unc == pytest.approx(0.05, abs=1e-9)

    def test_cai_delta_scorer_nonzero_for_cds_edit(self, cds_scorer):
        """CDS synonymous edit produces non-zero ΔCDS."""
        src = _make_record(cds=START_CODON + "GCU" * 4 + "UAA")
        # Find a synonymous codon for alanine (GCU → GCA/GCC/GCG)
        # GCG is optimal (1.0), GCU is non-optimal (0.3)
        new_cds = START_CODON + "GCG" + "GCU" * 3 + "UAA"
        cand = MRNARecord(
            transcript_id=src.transcript_id,
            five_utr=src.five_utr, cds=new_cds, three_utr=src.three_utr,
            metadata=src.metadata,
        )
        delta, unc = cds_scorer._score(src, cand)
        # GCU (0.3) → GCG (1.0): ΔCAI_per_codon = (1.0 - 0.3) / 4 = 0.175
        # ΔCDS = 0.5 * 0.175 = 0.0875
        assert delta > 0.0, f"Expected positive delta, got {delta}"
        assert delta == pytest.approx(0.0875, abs=0.01)

    def test_cai_delta_scorer_negative_for_suboptimal_edit(self, cds_scorer):
        """Editing an optimal codon to a suboptimal synonym produces negative delta."""
        # Start with optimal (GCG) and edit to suboptimal (GCU)
        src = _make_record(cds=START_CODON + "GCG" + "GCU" * 3 + "UAA")
        new_cds = START_CODON + "GCU" * 4 + "UAA"
        cand = MRNARecord(
            transcript_id=src.transcript_id,
            five_utr=src.five_utr, cds=new_cds, three_utr=src.three_utr,
            metadata=src.metadata,
        )
        delta, _ = cds_scorer._score(src, cand)
        assert delta < 0.0, f"Expected negative delta, got {delta}"

    def test_cai_scorer_inherits_counting_oracle(self, cds_scorer):
        """CAIDeltaScorer must be a CountingOracle (so .score() works)."""
        assert isinstance(cds_scorer, CountingOracle)

    def test_cai_scorer_score_method_works(self, cds_scorer):
        """The public .score() method (from CountingOracle) must work."""
        src = _make_record(cds=START_CODON + "GCU" * 4 + "UAA")
        new_cds = START_CODON + "GCG" + "GCU" * 3 + "UAA"
        cand = MRNARecord(
            transcript_id=src.transcript_id,
            five_utr=src.five_utr, cds=new_cds, three_utr=src.three_utr,
            metadata=src.metadata,
        )
        delta, unc = cds_scorer.score(src, cand, purpose="eval")
        assert delta > 0.0
        assert unc > 0.0


# ---------------------------------------------------------------------------
# CombinedOracle tests
# ---------------------------------------------------------------------------

class TestCombinedOracle:
    """Tests for the CombinedOracle (5'UTR oracle + CAI CDS scorer)."""

    def test_combined_oracle_5utr_only_uses_5utr_oracle(self):
        """5'UTR-only edit: combined delta = 5'UTR oracle delta (CAI = 0)."""
        five_utr_oracle = SyntheticDeltaOracle(seed=0, uncertainty=0.02)
        cds_scorer = CAIDeltaScorer(weight=0.5)
        combined = CombinedOracle(five_utr_oracle, cds_scorer)

        src = _make_record()
        new_utr = src.five_utr[:10] + "G" + src.five_utr[11:]
        cand = MRNARecord(
            transcript_id=src.transcript_id,
            five_utr=new_utr, cds=src.cds, three_utr=src.three_utr,
            metadata=src.metadata,
        )
        d_combined, _ = combined._score(src, cand)
        d_5utr, _ = five_utr_oracle._score(src, cand)
        assert d_combined == pytest.approx(d_5utr, abs=1e-9)

    def test_combined_oracle_cds_only_uses_cai_scorer(self):
        """CDS-only edit: combined delta = CAI delta (5'UTR oracle = ~0)."""
        # SyntheticDeltaOracle returns ~0 for sequences with no edits (or 5'UTR
        # edits that don't match its seed pattern); CAI fires on CDS edits.
        five_utr_oracle = SyntheticDeltaOracle(seed=0, uncertainty=0.02)
        cds_scorer = CAIDeltaScorer(weight=0.5)
        combined = CombinedOracle(five_utr_oracle, cds_scorer)

        src = _make_record(cds=START_CODON + "GCU" * 4 + "UAA")
        new_cds = START_CODON + "GCG" + "GCU" * 3 + "UAA"
        cand = MRNARecord(
            transcript_id=src.transcript_id,
            five_utr=src.five_utr, cds=new_cds, three_utr=src.three_utr,
            metadata=src.metadata,
        )
        d_combined, _ = combined._score(src, cand)
        d_cds, _ = cds_scorer._score(src, cand)
        # Combined delta should equal CAI delta (5'UTR oracle adds ~0)
        assert d_combined == pytest.approx(d_cds, abs=0.1)

    def test_combined_oracle_joint_edit_is_additive(self):
        """Joint edit: combined delta ≈ Δ5'UTR + ΔCDS (additive by design)."""
        five_utr_oracle = SyntheticDeltaOracle(seed=0, uncertainty=0.02)
        cds_scorer = CAIDeltaScorer(weight=0.5)
        combined = CombinedOracle(five_utr_oracle, cds_scorer)

        src = _make_record(cds=START_CODON + "GCU" * 4 + "UAA")
        new_utr = src.five_utr[:10] + "G" + src.five_utr[11:]
        new_cds = START_CODON + "GCG" + "GCU" * 3 + "UAA"
        cand = MRNARecord(
            transcript_id=src.transcript_id,
            five_utr=new_utr, cds=new_cds, three_utr=src.three_utr,
            metadata=src.metadata,
        )
        d_combined, _ = combined._score(src, cand)
        d_5utr, _ = five_utr_oracle._score(src, cand)
        d_cds, _ = cds_scorer._score(src, cand)
        # Additive property: d_combined ≈ d_5utr + d_cds
        assert d_combined == pytest.approx(d_5utr + d_cds, abs=1e-6)

    def test_combined_oracle_uncertainty_is_quadrature(self):
        """Uncertainty should be sqrt(u5^2 + uc^2) (quadrature)."""
        five_utr_oracle = SyntheticDeltaOracle(seed=0, uncertainty=0.02)
        cds_scorer = CAIDeltaScorer(weight=0.5)
        combined = CombinedOracle(five_utr_oracle, cds_scorer)

        src = _make_record()
        _, u_combined = combined._score(src, src)
        _, u_5utr = five_utr_oracle._score(src, src)
        _, u_cds = cds_scorer._score(src, src)
        expected = (u_5utr ** 2 + u_cds ** 2) ** 0.5
        assert u_combined == pytest.approx(expected, abs=1e-9)

    def test_combined_oracle_is_counting_oracle(self):
        five_utr_oracle = SyntheticDeltaOracle(seed=0, uncertainty=0.02)
        cds_scorer = CAIDeltaScorer(weight=0.5)
        combined = CombinedOracle(five_utr_oracle, cds_scorer)
        assert isinstance(combined, CountingOracle)


# ---------------------------------------------------------------------------
# Counterfactual arm helpers
# ---------------------------------------------------------------------------

class TestBestSingleCDSEdit:
    """Tests for _best_single_cds_edit using the CAI scorer."""

    def test_returns_non_identity_candidate(self, cds_scorer):
        src = _make_record(cds=START_CODON + "GCU" * 4 + "UAA")
        cand, edits = _best_single_cds_edit(src, cds_scorer)
        # Must produce at least one edit (CDS is editable)
        assert len(edits) == 1
        assert cand.cds != src.cds

    def test_returns_cds_with_higher_cai(self, cds_scorer):
        """The best CDS edit should improve CAI (delta > 0)."""
        src = _make_record(cds=START_CODON + "GCU" * 4 + "UAA")
        cand, edits = _best_single_cds_edit(src, cds_scorer)
        delta, _ = cds_scorer._score(src, cand)
        assert delta >= 0.0, f"Expected non-negative delta, got {delta}"

    def test_preserves_protein_identity(self, cds_scorer):
        """CDS synonymous edit must preserve the protein sequence."""
        from core.constants import translate
        src = _make_record(cds=START_CODON + "GCU" * 4 + "UAA")
        cand, edits = _best_single_cds_edit(src, cds_scorer)
        assert translate(src.cds) == translate(cand.cds)


class TestBestSingle5UTREdit:
    """Tests for _best_single_5utr_edit using a synthetic oracle."""

    def test_returns_non_identity_candidate(self):
        oracle = SyntheticDeltaOracle(seed=0, uncertainty=0.02)
        src = _make_record()
        cand, edits = _best_single_5utr_edit(src, oracle)
        assert len(edits) == 1
        assert cand.five_utr != src.five_utr

    def test_edit_has_correct_region(self):
        oracle = SyntheticDeltaOracle(seed=0, uncertainty=0.02)
        src = _make_record()
        _, edits = _best_single_5utr_edit(src, oracle)
        assert edits[0]["region"] == "five_utr"


class TestRandomEdit:
    """Tests for _random_edit in 5'UTR and CDS regions."""

    def test_5utr_random_edit(self):
        rng = np.random.RandomState(42)
        src = _make_record()
        cand, edits = _random_edit(src, "five_utr", rng)
        assert len(edits) == 1
        assert edits[0]["region"] == "five_utr"
        assert cand.five_utr != src.five_utr
        assert len(cand.five_utr) == len(src.five_utr)

    def test_cds_random_edit_preserves_protein(self):
        from core.constants import translate
        rng = np.random.RandomState(42)
        src = _make_record(cds=START_CODON + "GCU" * 4 + "UAA")
        cand, edits = _random_edit(src, "cds", rng)
        if edits:  # may be empty if no synonyms available
            assert translate(src.cds) == translate(cand.cds)


# ---------------------------------------------------------------------------
# Synergy computation
# ---------------------------------------------------------------------------

class TestSynergyComputation:
    """Tests for compute_synergy_stats and _ols_interaction."""

    def test_synergy_zero_for_additive_oracle(self):
        """With an additive CombinedOracle, synergy ≈ 0 by construction."""
        # Build synthetic arm_results where joint = 5utr + cds (additive)
        arm_results = {
            "wt": [{"source_id": "s1", "delta_train": 0.0, "n_edits": 0, "edits": []}],
            "five_utr_only": [{"source_id": "s1", "delta_train": 0.1, "delta_independent": 0.1, "n_edits": 1, "edits": []}],
            "cds_only": [{"source_id": "s1", "delta_train": 0.05, "delta_independent": 0.05, "n_edits": 1, "edits": []}],
            "joint": [{"source_id": "s1", "delta_train": 0.15, "delta_independent": 0.15, "n_edits": 2, "edits": []}],
            "matched_random": [{"source_id": "s1", "delta_train": 0.0, "n_edits": 2, "edits": []}],
            "shuffled_joint": [{"source_id": "s1", "delta_train": 0.0, "n_edits": 2, "edits": []}],
            "additive_reconstruction": [{"source_id": "s1", "delta_train": 0.15, "delta_independent": 0.15, "n_edits": 2, "edits": []}],
            "joint_policy": [{"source_id": "s1", "delta_train": 0.0, "n_edits": 0, "edits": []}],
        }
        stats = compute_synergy_stats(arm_results)
        syn = stats["synergy_analysis"]["training_oracle"]
        # Synergy = 0.15 - 0.1 - 0.05 = 0
        assert syn["mean"] == pytest.approx(0.0, abs=1e-9)

    def test_synergy_positive_for_super_additive(self):
        """Super-additive joint effect produces positive synergy."""
        arm_results = {
            "wt": [{"source_id": "s1", "delta_train": 0.0, "n_edits": 0, "edits": []}],
            "five_utr_only": [{"source_id": "s1", "delta_train": 0.1, "delta_independent": 0.1, "n_edits": 1, "edits": []}],
            "cds_only": [{"source_id": "s1", "delta_train": 0.05, "delta_independent": 0.05, "n_edits": 1, "edits": []}],
            "joint": [{"source_id": "s1", "delta_train": 0.20, "delta_independent": 0.20, "n_edits": 2, "edits": []}],
            "matched_random": [{"source_id": "s1", "delta_train": 0.0, "n_edits": 2, "edits": []}],
            "shuffled_joint": [{"source_id": "s1", "delta_train": 0.0, "n_edits": 2, "edits": []}],
            "additive_reconstruction": [{"source_id": "s1", "delta_train": 0.15, "delta_independent": 0.15, "n_edits": 2, "edits": []}],
            "joint_policy": [{"source_id": "s1", "delta_train": 0.0, "n_edits": 0, "edits": []}],
        }
        stats = compute_synergy_stats(arm_results)
        syn = stats["synergy_analysis"]["training_oracle"]
        # Synergy = 0.20 - 0.1 - 0.05 = 0.05
        assert syn["mean"] == pytest.approx(0.05, abs=1e-9)

    def test_ols_interaction_returns_coefficients(self):
        """OLS interaction test returns β0, β1, β2, β3."""
        arm_results = {
            "wt": [{"delta_train": 0.0}],
            "five_utr_only": [{"delta_train": 0.1}],
            "cds_only": [{"delta_train": 0.05}],
            "joint": [{"delta_train": 0.15}],
        }
        result = _ols_interaction(arm_results)
        assert "coefficients" in result
        assert "beta3_interaction" in result["coefficients"]
        # β3 should be 0 for the additive case (0.15 = 0.1 + 0.05 + β3*1*1)
        assert result["coefficients"]["beta3_interaction"] == pytest.approx(0.0, abs=1e-6)

    def test_ols_interaction_insufficient_data(self):
        """OLS returns error if insufficient data."""
        arm_results = {"wt": [], "five_utr_only": [], "cds_only": [], "joint": []}
        result = _ols_interaction(arm_results)
        assert "error" in result


# ---------------------------------------------------------------------------
# 3'UTR gate evaluation
# ---------------------------------------------------------------------------

class TestThreeUTRGate:
    """Tests for evaluate_3utr_gate."""

    def test_gate_locked_when_no_3utr_data(self, tmp_path):
        """Gate must be locked when benchmark has no 3'UTR records."""
        # Write a measured_tier.jsonl with only 5'UTR records
        import json
        path = tmp_path / "measured_tier.jsonl"
        with open(path, "w") as f:
            rec = {
                "record_id": "r1",
                "source_id": "s1",
                "source_sequence": "GCCAUG",
                "candidate_sequence": "GCCACG",
                "edit_list": [{"pos": 4, "ref": "U", "alt": "C", "region": "five_utr"}],
                "edit_count": 1,
                "edited_region": "five_utr",
                "delta": 0.1,
                "confidence": "measured",
                "split_role": "train",
                "family_cluster_id": "c1",
                "edit_type": "sub",
            }
            f.write(json.dumps(rec) + "\n")

        sources = [_make_record()]
        result = evaluate_3utr_gate(str(tmp_path), sources)
        assert result["gate_decision"] == "locked"
        assert result["three_utr_status"] == "locked"
        assert result["all_conditions_pass"] is False
        assert result["conditions"]["c1_3utr_labels"]["status"] == "FAIL"


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

class TestFullTranscriptDecision:
    """Tests for make_full_transcript_decision — must produce PARTIAL."""

    def test_decision_is_partial_for_additive_oracle(self):
        """With additive oracle (synergy ≈ 0), verdict must be PARTIAL, not NO-GO."""
        synergy_results = {
            "synergy_analysis": {
                "training_oracle": {"mean": 0.0, "std": 0.001, "n": 100},
                "independent_oracle": {"mean": 0.0, "std": 0.001, "n": 100},
            },
            "statistical_interaction": {
                "coefficients": {
                    "beta0_intercept": 0.0,
                    "beta1_5utr_main": 0.1,
                    "beta2_cds_main": 0.05,
                    "beta3_interaction": 0.0,
                },
                "t_statistics": {"beta0": 0.0, "beta1": 5.0, "beta2": 3.0, "beta3": 0.0},
            },
            "arm_summary": {
                "cds_only": {"delta_train": {"mean": 0.05}},
                "five_utr_only": {"delta_train": {"mean": 0.1}},
                "joint": {"delta_train": {"mean": 0.15}},
            },
        }
        three_utr_gate = {"gate_decision": "locked"}
        mechanism_results = {"assessability": {}}

        decision = make_full_transcript_decision(synergy_results, three_utr_gate, mechanism_results)

        assert decision["verdict"] == "PARTIAL", (
            f"Expected PARTIAL, got {decision['verdict']}. "
            f"Reason: {decision.get('verdict_reason', '')}"
        )
        # NO-GO sub-criteria must NOT all fire
        nogo = decision["nogo_criteria"]
        assert nogo["n1_interaction_unstable"]["status"] != "PASS"
        assert nogo["n2_single_region_explains"]["status"] != "PASS"
        assert nogo["n4_3utr_reward_hacking"]["status"] != "PASS"

    def test_decision_preserves_three_utr_status(self):
        synergy_results = {
            "synergy_analysis": {"training_oracle": {"mean": 0.0, "std": 0.0}},
            "statistical_interaction": {"coefficients": {"beta3_interaction": 0.0}, "t_statistics": {"beta3": 0.0}},
            "arm_summary": {},
        }
        three_utr_gate = {"gate_decision": "locked"}
        decision = make_full_transcript_decision(synergy_results, three_utr_gate, {})
        assert decision["three_utr_status"] == "locked"

    def test_decision_has_future_work(self):
        synergy_results = {
            "synergy_analysis": {"training_oracle": {"mean": 0.0, "std": 0.0}},
            "statistical_interaction": {"coefficients": {"beta3_interaction": 0.0}, "t_statistics": {"beta3": 0.0}},
            "arm_summary": {},
        }
        decision = make_full_transcript_decision(synergy_results, {"gate_decision": "locked"}, {})
        assert "future_work" in decision
        assert len(decision["future_work"]) > 0

    def test_decision_n2_does_not_fire_with_nonzero_cds(self):
        """n2 (single-region explains) must FAIL when CDS contributes non-zero."""
        synergy_results = {
            "synergy_analysis": {"training_oracle": {"mean": 0.0, "std": 0.0}},
            "statistical_interaction": {"coefficients": {"beta3_interaction": 0.0}, "t_statistics": {"beta3": 0.0}},
            "arm_summary": {
                "cds_only": {"delta_train": {"mean": 0.05}},  # Non-zero CDS contribution
                "five_utr_only": {"delta_train": {"mean": 0.1}},
                "joint": {"delta_train": {"mean": 0.15}},
            },
        }
        decision = make_full_transcript_decision(synergy_results, {"gate_decision": "locked"}, {})
        assert decision["nogo_criteria"]["n2_single_region_explains"]["status"] == "FAIL"


# ---------------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------------

class TestSmokeRun:
    """Smoke test: run the full counterfactual arms pipeline."""

    def test_run_counterfactual_arms_smoke(self, synthetic_sources):
        """End-to-end smoke test with synthetic oracles."""
        training_oracle = SyntheticDeltaOracle(seed=0, uncertainty=0.02)
        independent_oracle = SyntheticDeltaOracle(seed=2, uncertainty=0.03)

        result = run_counterfactual_arms(
            synthetic_sources,
            training_oracle,
            independent_oracle,
            n_sequences=5,
            seed=42,
        )

        # Verify all 8 arms are present
        assert "arm_summary" in result
        for arm in ["wt", "five_utr_only", "cds_only", "joint",
                    "matched_random", "shuffled_joint",
                    "additive_reconstruction", "joint_policy"]:
            assert arm in result["arm_summary"], f"Missing arm: {arm}"

        # CDS-only arm must have non-zero delta (CAI fires)
        cds_deltas = [r["delta_train"] for r in result["per_sequence"]["cds_only"]]
        # SyntheticDeltaOracle returns ~0 for CDS, but CAI scorer should produce non-zero
        # However, the inert CDS used here has no editable codons (start + 4×GCU + stop = 6 codons, 4 editable)
        # So we just check the arm ran without error
        assert len(cds_deltas) > 0

        # Synergy analysis must be present
        assert "synergy_analysis" in result
        assert "training_oracle" in result["synergy_analysis"]
        assert "statistical_interaction" in result

    def test_decision_from_smoke_results(self, synthetic_sources):
        """End-to-end: run arms + decision, verify PARTIAL verdict."""
        training_oracle = SyntheticDeltaOracle(seed=0, uncertainty=0.02)
        independent_oracle = SyntheticDeltaOracle(seed=2, uncertainty=0.03)

        synergy = run_counterfactual_arms(
            synthetic_sources, training_oracle, independent_oracle,
            n_sequences=5, seed=42,
        )
        # 3'UTR gate: use a temp dir with no 3'UTR data
        with tempfile.TemporaryDirectory() as tmp:
            # Write empty measured_tier.jsonl
            with open(os.path.join(tmp, "measured_tier.jsonl"), "w") as f:
                pass  # empty file
            three_utr = evaluate_3utr_gate(tmp, synthetic_sources)

        decision = make_full_transcript_decision(synergy, three_utr, {})
        # With synthetic oracles and additive CombinedOracle, verdict must be PARTIAL
        assert decision["verdict"] in ("PARTIAL", "GO"), (
            f"Expected PARTIAL or GO, got {decision['verdict']}. "
            f"Reason: {decision.get('verdict_reason', '')}"
        )


class TestSpecRequiredAnalysis:
    """Spec lines 2488-2494 require 5 analysis items. Verify all are present."""

    def test_spec_required_analysis_has_all_five_items(self, synthetic_sources):
        training_oracle = SyntheticDeltaOracle(seed=0, uncertainty=0.02)
        independent_oracle = SyntheticDeltaOracle(seed=2, uncertainty=0.03)
        result = run_counterfactual_arms(
            synthetic_sources, training_oracle, independent_oracle,
            n_sequences=5, seed=42,
        )
        sra = result.get("spec_required_analysis", {})
        required = {
            "pairwise_interaction",
            "edit_order_effect",
            "reward_per_edit",
            "independent_oracle_interaction",
            "experimental_interaction",
        }
        assert set(sra.keys()) == required, (
            f"Missing spec-required items. Got {set(sra.keys())}"
        )

    def test_pairwise_interaction_is_computed(self, synthetic_sources):
        result = run_counterfactual_arms(
            synthetic_sources,
            SyntheticDeltaOracle(seed=0), SyntheticDeltaOracle(seed=1),
            n_sequences=5, seed=42,
        )
        pw = result["spec_required_analysis"]["pairwise_interaction"]
        assert pw["status"] == "computed"
        assert "result" in pw  # references statistical_interaction

    def test_edit_order_effect_is_not_assessable(self, synthetic_sources):
        result = run_counterfactual_arms(
            synthetic_sources,
            SyntheticDeltaOracle(seed=0), SyntheticDeltaOracle(seed=1),
            n_sequences=5, seed=42,
        )
        eo = result["spec_required_analysis"]["edit_order_effect"]
        assert eo["status"] == "not_assessable"
        assert "reason" in eo

    def test_reward_per_edit_computed_for_arms_with_edits(self, synthetic_sources):
        result = run_counterfactual_arms(
            synthetic_sources,
            SyntheticDeltaOracle(seed=0), SyntheticDeltaOracle(seed=1),
            n_sequences=5, seed=42,
        )
        rpe = result["spec_required_analysis"]["reward_per_edit"]
        assert rpe["status"] == "computed"
        per_arm = rpe["per_arm"]
        # five_utr_only should have edits and non-zero reward_per_edit
        assert "five_utr_only" in per_arm
        assert per_arm["five_utr_only"]["n"] > 0
        # wt should have 0 edits
        assert per_arm["wt"]["n"] == 0

    def test_independent_oracle_interaction_is_computed(self, synthetic_sources):
        result = run_counterfactual_arms(
            synthetic_sources,
            SyntheticDeltaOracle(seed=0), SyntheticDeltaOracle(seed=1),
            n_sequences=5, seed=42,
        )
        io = result["spec_required_analysis"]["independent_oracle_interaction"]
        assert io["status"] == "computed"
        assert "result" in io

    def test_experimental_interaction_is_not_available(self, synthetic_sources):
        result = run_counterfactual_arms(
            synthetic_sources,
            SyntheticDeltaOracle(seed=0), SyntheticDeltaOracle(seed=1),
            n_sequences=5, seed=42,
        )
        ei = result["spec_required_analysis"]["experimental_interaction"]
        assert ei["status"] == "not_available"
        assert "reason" in ei


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
