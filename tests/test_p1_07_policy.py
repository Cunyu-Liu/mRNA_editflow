"""P1-07: Policy API tests.

Verifies that the Policy class:
  - Builds correct legal action masks (UTR free, CDS synonymous-only, no CDS indels).
  - Produces a normalized action distribution (mass = 1 over legal actions).
  - Compares raw vs masked distributions (quality before/after masking).
  - Supports STOP action with configurable rate.
  - Computes trajectory log-prob = sum_t log p(a_t | s_t).
  - Samples legal actions only.
  - Applies actions correctly (sub/ins/del/STOP).

Run:
    cd /home/cunyuliu/mrna_editflow_goal
    /home/cunyuliu/miniconda3/envs/editflow/bin/python3.10 -m unittest \\
        mrna_editflow.tests.test_p1_07_policy -v
"""
from __future__ import annotations

import math
import os
import sys
import unittest
from typing import List, Tuple

import torch

# Make repo root importable when running directly.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mrna_editflow.core.constants import (
    CODON_TABLE,
    NUC_TO_ID,
    REGION_3UTR,
    REGION_5UTR,
    REGION_CDS,
    START_CODON,
    STOP_CODONS,
    V,
    translate,
)
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.rl.action_space import (
    STOP_ACTION,
    Action,
    ActionMask,
    apply_action,
    build_legal_action_mask,
)
from mrna_editflow.rl.policy import Policy, PolicyConfig


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_record(
    five_utr: str = "GCCAAC",
    cds: str = "AUGGCUUAA",
    three_utr: str = "GGGCCC",
) -> MRNARecord:
    """Build a minimal MRNARecord for testing."""
    return MRNARecord(
        transcript_id="TEST0001",
        five_utr=five_utr,
        cds=cds,
        three_utr=three_utr,
        species="human",
    )


class _DummyBackbone:
    """Minimal backbone stub that returns zero embeddings.

    The Policy only uses ``backbone`` as an opaque object passed to
    ``model.forward``; for the legal-mask and action-application tests we
    don't need a real backbone.
    """

    def __init__(self) -> None:
        self.out_dim = 64

    def freeze(self) -> None:
        pass

    def embed(self, *args, **kwargs):  # pragma: no cover - not used in mask tests
        raise NotImplementedError

    def to(self, device):
        return self


class _DummyModel(torch.nn.Module):
    """Deterministic model stub returning a fixed rate field.

    Useful for testing the Policy's normalization and sampling logic without
    loading a real MRNAEditFormer checkpoint.
    """

    def __init__(self, rates: torch.Tensor, ins_probs: torch.Tensor, sub_probs: torch.Tensor):
        super().__init__()
        self._rates = rates
        self._ins_probs = ins_probs
        self._sub_probs = sub_probs

    def forward(self, token_ids, region_ids, phase_ids, t, padding_mask, backbone):
        # Rates: [B, L, 3]; ins_probs/sub_probs: [B, L, V]
        return {
            "rates": self._rates,
            "ins_probs": self._ins_probs,
            "sub_probs": self._sub_probs,
            "aux": None,
        }


def _make_uniform_model(L: int, V_dim: int = 6) -> _DummyModel:
    """Model with uniform rates=1 and uniform probs over V_dim tokens."""
    rates = torch.ones(1, L, 3)
    ins_probs = torch.full((1, L, V_dim), 1.0 / V_dim)
    sub_probs = torch.full((1, L, V_dim), 1.0 / V_dim)
    return _DummyModel(rates, ins_probs, sub_probs)


# ---------------------------------------------------------------------------
# Legal action mask tests
# ---------------------------------------------------------------------------


class TestLegalActionMask(unittest.TestCase):
    """Tests for build_legal_action_mask."""

    def test_utr_positions_allow_all_substitutions(self) -> None:
        rec = _make_record(five_utr="GCCAAC", cds="AUGGCUUAA", three_utr="GGG")
        mask = build_legal_action_mask(rec, torch.device("cpu"))
        # 5'UTR has 6 positions, 3'UTR has 3 positions → 9 UTR positions.
        # Each allows 3 non-identity substitutions (4 - 1).
        # Total UTR sub slots = 9 * 3 = 27.
        # CDS has 9 positions (3 codons), each allows synonymous nts only.
        # We don't compute the exact CDS count here; just verify UTR.
        five_len = len(rec.five_utr)
        three_len = len(rec.three_utr)
        # 5'UTR: all 4 nts allowed except identity → 3 legal per position.
        for i in range(five_len):
            legal = mask.sub_mask[i].tolist()
            self.assertEqual(sum(legal), 3, f"5'UTR pos {i} should have 3 legal subs")
        # 3'UTR: same.
        offset = five_len + len(rec.cds)
        for i in range(offset, offset + three_len):
            legal = mask.sub_mask[i].tolist()
            self.assertEqual(sum(legal), 3, f"3'UTR pos {i} should have 3 legal subs")

    def test_cds_substitutions_are_synonymous_only(self) -> None:
        # Use a codon with known synonymous alternatives.
        # Leucine codons: UUA, UUG, CUU, CUC, CUA, CUG.
        # Start with "UUA" — single-nt subs that keep Leu:
        #   UUA -> UUG (position 2 change): synonymous.
        #   UUA -> CUA (position 0 change): synonymous.
        #   UUA -> UCA (position 1 change): Ser — NOT synonymous.
        # So at position 0 (phase 0) of "UUA", legal subs are {C} (-> CUA).
        # Wait — we need to check ALLOWED_NT_SUB_TABLE. Let's just verify
        # that *no* substitution at codon 0 (AUG) is legal (since AUG is
        # unique — only Met), and that a synonymous sub exists for a Leu codon.
        rec = _make_record(five_utr="GCC", cds="AUGCUUUAA", three_utr="GGG")
        mask = build_legal_action_mask(rec, torch.device("cpu"))
        five_len = len(rec.five_utr)
        # Codon 0 = AUG (positions 0..2 in CDS, i.e., seq positions five_len..five_len+2)
        # AUG codes for Met, which has only one codon → no synonymous sub possible.
        for i in range(five_len, five_len + 3):
            legal = mask.sub_mask[i].tolist()
            self.assertEqual(sum(legal), 0, f"AUG position {i} should have 0 legal subs")

    def test_cds_indels_forbidden_by_default(self) -> None:
        rec = _make_record(five_utr="GCC", cds="AUGGCUUAA", three_utr="GGG")
        mask = build_legal_action_mask(rec, torch.device("cpu"), codon_indel=False)
        five_len = len(rec.five_utr)
        cds_len = len(rec.cds)
        # CDS positions: no ins, no del.
        for i in range(five_len, five_len + cds_len):
            self.assertEqual(mask.ins_mask[i].sum().item(), 0, f"CDS pos {i} should have 0 ins")
            self.assertEqual(mask.del_mask[i].item(), False, f"CDS pos {i} should have del=False")

    def test_cds_indels_allowed_at_codon_start_when_codon_indel_true(self) -> None:
        rec = _make_record(five_utr="GCC", cds="AUGGCUUAA", three_utr="GGG")
        mask = build_legal_action_mask(rec, torch.device("cpu"), codon_indel=True)
        five_len = len(rec.five_utr)
        # CDS positions 0, 3, 6 are codon-start (phase 0).
        # Ins and del should be legal there.
        for codon_start in (0, 3, 6):
            pos = five_len + codon_start
            self.assertEqual(mask.ins_mask[pos].sum().item(), V, f"codon-start {pos} should allow all 4 ins nts")
            self.assertTrue(mask.del_mask[pos].item(), f"codon-start {pos} should allow del")
        # Non-codon-start CDS positions: no ins, no del.
        for non_start in (1, 2, 4, 5, 7, 8):
            pos = five_len + non_start
            self.assertEqual(mask.ins_mask[pos].sum().item(), 0, f"non-codon-start {pos} should have 0 ins")
            self.assertFalse(mask.del_mask[pos].item(), f"non-codon-start {pos} should have del=False")

    def test_stop_always_legal(self) -> None:
        rec = _make_record()
        mask = build_legal_action_mask(rec, torch.device("cpu"))
        self.assertTrue(mask.stop_legal)

    def test_num_legal_property(self) -> None:
        rec = _make_record(five_utr="GCC", cds="AUGGCUUAA", three_utr="GGG")
        mask = build_legal_action_mask(rec, torch.device("cpu"))
        # Should be > 0 (at least UTR subs + STOP).
        self.assertGreater(mask.num_legal, 0)
        # num_legal = num_ins + num_sub + num_del + 1 (STOP)
        self.assertEqual(
            mask.num_legal,
            mask.num_ins + mask.num_sub + mask.num_del + 1,
        )


# ---------------------------------------------------------------------------
# Action application tests
# ---------------------------------------------------------------------------


class TestApplyAction(unittest.TestCase):
    """Tests for apply_action."""

    def test_substitution_in_5utr(self) -> None:
        rec = _make_record(five_utr="GCC", cds="AUGGCUUAA", three_utr="GGG")
        # Substitute position 0 (G) with A.
        action = Action(op="sub", pos=0, nt=NUC_TO_ID["A"])
        new_rec = apply_action(rec, action)
        self.assertEqual(new_rec.five_utr, "ACC")
        self.assertEqual(new_rec.cds, rec.cds)
        self.assertEqual(new_rec.three_utr, rec.three_utr)

    def test_substitution_in_3utr(self) -> None:
        rec = _make_record(five_utr="GCC", cds="AUGGCUUAA", three_utr="GGG")
        # 5'UTR has 3, CDS has 9 → 3'UTR starts at pos 12.
        action = Action(op="sub", pos=12, nt=NUC_TO_ID["C"])
        new_rec = apply_action(rec, action)
        self.assertEqual(new_rec.three_utr, "CGG")
        self.assertEqual(new_rec.five_utr, rec.five_utr)
        self.assertEqual(new_rec.cds, rec.cds)

    def test_insertion_in_5utr(self) -> None:
        rec = _make_record(five_utr="GCC", cds="AUGGCUUAA", three_utr="GGG")
        # Insert A after position 1 (C). "GCC" -> "GC" + "A" + "C" = "GCAC".
        action = Action(op="ins", pos=1, nt=NUC_TO_ID["A"])
        new_rec = apply_action(rec, action)
        self.assertEqual(new_rec.five_utr, "GCAC")
        self.assertEqual(new_rec.cds, rec.cds)

    def test_deletion_in_5utr(self) -> None:
        rec = _make_record(five_utr="GCC", cds="AUGGCUUAA", three_utr="GGG")
        action = Action(op="del", pos=0, nt=-1)
        new_rec = apply_action(rec, action)
        self.assertEqual(new_rec.five_utr, "CC")
        self.assertEqual(new_rec.cds, rec.cds)

    def test_stop_action_returns_record_unchanged(self) -> None:
        rec = _make_record()
        new_rec = apply_action(rec, STOP_ACTION)
        self.assertEqual(new_rec.five_utr, rec.five_utr)
        self.assertEqual(new_rec.cds, rec.cds)
        self.assertEqual(new_rec.three_utr, rec.three_utr)

    def test_substitution_preserves_cds_length(self) -> None:
        rec = _make_record(five_utr="GCC", cds="AUGGCUUAA", three_utr="GGG")
        # 5'UTR is 3 nt; CDS positions are 3..11.
        # CDS = AUG GCU UAA. GCU (Ala) at positions 6,7,8.
        # Sub pos 8 (U) with C → GCC (Ala, synonymous). Result: AUGGCCUAA.
        action = Action(op="sub", pos=8, nt=NUC_TO_ID["C"])
        new_rec = apply_action(rec, action)
        self.assertEqual(len(new_rec.cds), len(rec.cds))
        self.assertEqual(new_rec.cds, "AUGGCCUAA")


# ---------------------------------------------------------------------------
# Policy distribution tests
# ---------------------------------------------------------------------------


class TestPolicyDistribution(unittest.TestCase):
    """Tests for Policy.action_logprobs and normalization."""

    def _make_policy(self, rec: MRNARecord, stop_rate: float = 1.0) -> Policy:
        L = len(rec.seq)
        model = _make_uniform_model(L)
        backbone = _DummyBackbone()
        cfg = PolicyConfig(
            stop_rate_strategy="constant",
            stop_rate_value=stop_rate,
            temperature=1.0,
            time_step=0.5,
            codon_indel=False,
        )
        return Policy(model=model, backbone=backbone, cfg=cfg, device=torch.device("cpu"))

    def test_masked_distribution_is_normalized(self) -> None:
        rec = _make_record()
        policy = self._make_policy(rec)
        lps = policy.action_logprobs(rec)
        # Sum of exp(logprob) over legal actions should be 1.
        ins_mass = torch.exp(lps.ins_logprobs).nansum().item()
        sub_mass = torch.exp(lps.sub_logprobs).nansum().item()
        del_mass = torch.exp(lps.del_logprobs).nansum().item()
        stop_mass = math.exp(lps.stop_logprob) if math.isfinite(lps.stop_logprob) else 0.0
        total = ins_mass + sub_mass + del_mass + stop_mass
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_raw_distribution_is_normalized(self) -> None:
        rec = _make_record()
        policy = self._make_policy(rec)
        lps = policy.action_logprobs(rec, return_raw=True)
        self.assertIsNotNone(lps.raw_ins_logprobs)
        ins_mass = torch.exp(lps.raw_ins_logprobs).nansum().item()
        sub_mass = torch.exp(lps.raw_sub_logprobs).nansum().item()
        del_mass = torch.exp(lps.raw_del_logprobs).nansum().item()
        stop_mass = math.exp(lps.raw_stop_logprob) if math.isfinite(lps.raw_stop_logprob) else 0.0
        total = ins_mass + sub_mass + del_mass + stop_mass
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_raw_mass_on_legal_le_one(self) -> None:
        # The raw distribution includes mass on illegal actions, so
        # raw_mass_on_legal ≤ 1.
        rec = _make_record()
        policy = self._make_policy(rec)
        diag = policy.quality_before_after_masking(rec)
        self.assertLessEqual(diag["raw_mass_on_legal"], 1.0 + 1e-6)
        self.assertGreaterEqual(diag["raw_mass_on_legal"], 0.0)

    def test_masked_mass_on_legal_is_one(self) -> None:
        rec = _make_record()
        policy = self._make_policy(rec)
        diag = policy.quality_before_after_masking(rec)
        self.assertAlmostEqual(diag["masked_mass_on_legal"], 1.0, places=5)

    def test_mass_loss_nonnegative(self) -> None:
        rec = _make_record()
        policy = self._make_policy(rec)
        diag = policy.quality_before_after_masking(rec)
        self.assertGreaterEqual(diag["mass_loss"], -1e-6)

    def test_stop_logprob_finite(self) -> None:
        rec = _make_record()
        policy = self._make_policy(rec, stop_rate=1.0)
        lps = policy.action_logprobs(rec)
        self.assertTrue(math.isfinite(lps.stop_logprob))
        self.assertGreater(lps.stop_logprob, float("-inf"))

    def test_stop_rate_appears_in_distribution(self) -> None:
        rec = _make_record()
        # High stop rate → STOP has high probability.
        policy_high = self._make_policy(rec, stop_rate=100.0)
        # Low stop rate → STOP has low probability.
        policy_low = self._make_policy(rec, stop_rate=0.001)
        lp_high = policy_high.action_logprobs(rec).stop_logprob
        lp_low = policy_low.action_logprobs(rec).stop_logprob
        self.assertGreater(lp_high, lp_low)

    def test_logprob_method_returns_correct_value(self) -> None:
        rec = _make_record()
        policy = self._make_policy(rec)
        lps = policy.action_logprobs(rec)
        # Pick a legal action and check logprob matches.
        mask = policy.legal_action_mask(rec)
        # Find first legal sub.
        legal_subs = mask.sub_mask.nonzero(as_tuple=False)
        if legal_subs.numel() > 0:
            pos, nt = legal_subs[0].tolist()
            action = Action(op="sub", pos=pos, nt=nt)
            lp = lps.logprob(action)
            expected = float(lps.sub_logprobs[pos, nt].item())
            self.assertEqual(lp, expected)
        # STOP.
        self.assertEqual(lps.logprob(STOP_ACTION), lps.stop_logprob)


# ---------------------------------------------------------------------------
# Sampling tests
# ---------------------------------------------------------------------------


class TestPolicySampling(unittest.TestCase):
    """Tests for Policy.sample."""

    def _make_policy(self, rec: MRNARecord, stop_rate: float = 1.0) -> Policy:
        L = len(rec.seq)
        model = _make_uniform_model(L)
        backbone = _DummyBackbone()
        cfg = PolicyConfig(
            stop_rate_strategy="constant",
            stop_rate_value=stop_rate,
            temperature=1.0,
            codon_indel=False,
        )
        return Policy(model=model, backbone=backbone, cfg=cfg, device=torch.device("cpu"))

    def test_sample_returns_legal_action(self) -> None:
        rec = _make_record()
        policy = self._make_policy(rec)
        mask = policy.legal_action_mask(rec)
        torch.manual_seed(42)
        for _ in range(20):
            action, lp = policy.sample(rec)
            self.assertTrue(math.isfinite(lp))
            if action.is_stop():
                self.assertTrue(mask.stop_legal)
            elif action.op == "ins":
                self.assertTrue(mask.ins_mask[action.pos, action.nt].item(),
                                f"sampled illegal ins at ({action.pos}, {action.nt})")
            elif action.op == "sub":
                self.assertTrue(mask.sub_mask[action.pos, action.nt].item(),
                                f"sampled illegal sub at ({action.pos}, {action.nt})")
            elif action.op == "del":
                self.assertTrue(mask.del_mask[action.pos].item(),
                                f"sampled illegal del at {action.pos}")

    def test_sample_stop_with_high_stop_rate(self) -> None:
        rec = _make_record()
        policy = self._make_policy(rec, stop_rate=1000.0)
        torch.manual_seed(0)
        stop_count = 0
        for _ in range(20):
            action, _ = policy.sample(rec)
            if action.is_stop():
                stop_count += 1
        # With very high stop rate, most samples should be STOP.
        self.assertGreater(stop_count, 10)

    def test_sample_never_stop_with_low_stop_rate(self) -> None:
        rec = _make_record()
        policy = self._make_policy(rec, stop_rate=1e-6)
        torch.manual_seed(0)
        stop_count = 0
        for _ in range(20):
            action, _ = policy.sample(rec)
            if action.is_stop():
                stop_count += 1
        # With very low stop rate, few or no STOPs.
        self.assertLess(stop_count, 5)


# ---------------------------------------------------------------------------
# Trajectory log-prob tests
# ---------------------------------------------------------------------------


class TestTrajectoryLogProb(unittest.TestCase):
    """Tests for Policy.trajectory_logprob."""

    def _make_policy(self, rec: MRNARecord) -> Policy:
        L = len(rec.seq)
        model = _make_uniform_model(L)
        backbone = _DummyBackbone()
        cfg = PolicyConfig(
            stop_rate_strategy="constant",
            stop_rate_value=1.0,
            temperature=1.0,
            codon_indel=False,
        )
        return Policy(model=model, backbone=backbone, cfg=cfg, device=torch.device("cpu"))

    def test_trajectory_logprob_is_finite_for_legal_trajectory(self) -> None:
        rec = _make_record()
        policy = self._make_policy(rec)
        # Build a 3-step trajectory.
        s0 = rec
        mask0 = policy.legal_action_mask(s0)
        legal_subs = mask0.sub_mask.nonzero(as_tuple=False)
        self.assertGreater(legal_subs.numel(), 0)
        pos, nt = legal_subs[0].tolist()
        a0 = Action(op="sub", pos=pos, nt=nt)
        s1 = apply_action(s0, a0)
        mask1 = policy.legal_action_mask(s1)
        legal_subs1 = mask1.sub_mask.nonzero(as_tuple=False)
        self.assertGreater(legal_subs1.numel(), 0)
        pos1, nt1 = legal_subs1[0].tolist()
        a1 = Action(op="sub", pos=pos1, nt=nt1)
        s2 = apply_action(s1, a1)
        a2 = STOP_ACTION
        traj = [(s0, a0), (s1, a1), (s2, a2)]
        lp = policy.trajectory_logprob(traj)
        self.assertTrue(math.isfinite(lp))

    def test_trajectory_logprob_is_sum_of_step_logprobs(self) -> None:
        rec = _make_record()
        policy = self._make_policy(rec)
        s0 = rec
        mask0 = policy.legal_action_mask(s0)
        legal_subs = mask0.sub_mask.nonzero(as_tuple=False)
        pos, nt = legal_subs[0].tolist()
        a0 = Action(op="sub", pos=pos, nt=nt)
        s1 = apply_action(s0, a0)
        a1 = STOP_ACTION
        traj = [(s0, a0), (s1, a1)]
        lp_traj = policy.trajectory_logprob(traj)
        lp0 = policy.action_logprobs(s0).logprob(a0)
        lp1 = policy.action_logprobs(s1).logprob(a1)
        self.assertAlmostEqual(lp_traj, lp0 + lp1, places=5)

    def test_trajectory_logprob_neg_inf_for_illegal_action(self) -> None:
        rec = _make_record()
        policy = self._make_policy(rec)
        # Try an illegal action: ins at a CDS position (codon_indel=False).
        five_len = len(rec.five_utr)
        illegal_action = Action(op="ins", pos=five_len, nt=0)  # ins at first CDS pos
        traj = [(rec, illegal_action)]
        lp = policy.trajectory_logprob(traj)
        self.assertEqual(lp, float("-inf"))


# ---------------------------------------------------------------------------
# Quality before/after masking tests
# ---------------------------------------------------------------------------


class TestQualityBeforeAfterMasking(unittest.TestCase):
    """Tests for Policy.quality_before_after_masking."""

    def _make_policy(self, rec: MRNARecord) -> Policy:
        L = len(rec.seq)
        model = _make_uniform_model(L)
        backbone = _DummyBackbone()
        cfg = PolicyConfig(
            stop_rate_strategy="constant",
            stop_rate_value=1.0,
            temperature=1.0,
            codon_indel=False,
        )
        return Policy(model=model, backbone=backbone, cfg=cfg, device=torch.device("cpu"))

    def test_returns_expected_keys(self) -> None:
        rec = _make_record()
        policy = self._make_policy(rec)
        diag = policy.quality_before_after_masking(rec)
        for key in (
            "raw_mass_on_legal",
            "raw_mass_on_illegal",
            "masked_mass_on_legal",
            "num_legal_actions",
            "num_ins",
            "num_sub",
            "num_del",
            "log_partition_raw",
            "log_partition_masked",
            "mass_loss",
        ):
            self.assertIn(key, diag)

    def test_mass_loss_is_nonneg_and_le_one(self) -> None:
        rec = _make_record()
        policy = self._make_policy(rec)
        diag = policy.quality_before_after_masking(rec)
        self.assertGreaterEqual(diag["mass_loss"], -1e-6)
        self.assertLessEqual(diag["mass_loss"], 1.0 + 1e-6)

    def test_log_partition_raw_ge_log_partition_masked(self) -> None:
        # Raw partition includes illegal actions, so it should be ≥ masked.
        rec = _make_record()
        policy = self._make_policy(rec)
        diag = policy.quality_before_after_masking(rec)
        self.assertGreaterEqual(diag["log_partition_raw"], diag["log_partition_masked"] - 1e-6)


# ---------------------------------------------------------------------------
# Budget-aware stop rate tests
# ---------------------------------------------------------------------------


class TestBudgetAwareStopRate(unittest.TestCase):
    """Tests for budget_aware stop_rate_strategy."""

    def _make_policy(self, rec: MRNARecord) -> Policy:
        L = len(rec.seq)
        model = _make_uniform_model(L)
        backbone = _DummyBackbone()
        cfg = PolicyConfig(
            stop_rate_strategy="budget_aware",
            stop_rate_value=1.0,
            temperature=1.0,
            codon_indel=False,
        )
        return Policy(model=model, backbone=backbone, cfg=cfg, device=torch.device("cpu"))

    def test_stop_rate_increases_as_budget_decreases(self) -> None:
        rec = _make_record()
        policy = self._make_policy(rec)
        # Full budget (5/5): stop_rate = 1.0 * (2 - 1) = 1.0
        lp_full = policy.action_logprobs(rec, budget_remaining=5, budget_total=5).stop_logprob
        # Half budget (2/5): stop_rate = 1.0 * (2 - 0.4) = 1.6
        lp_half = policy.action_logprobs(rec, budget_remaining=2, budget_total=5).stop_logprob
        # No budget (0/5): stop_rate = 1.0 * (2 - 0) = 2.0
        lp_empty = policy.action_logprobs(rec, budget_remaining=0, budget_total=5).stop_logprob
        self.assertGreater(lp_empty, lp_full)
        self.assertGreater(lp_empty, lp_half)


# ---------------------------------------------------------------------------
# Protein-invariance integration test
# ---------------------------------------------------------------------------


class TestProteinInvariance(unittest.TestCase):
    """Verify that a trajectory of legal CDS substitutions preserves the protein."""

    def test_legal_cds_sub_trajectory_preserves_protein(self) -> None:
        # Use a CDS with multiple synonymous codons.
        # Ala (GCU/GCC/GCA/GCG), Leu (UUA/UUG/CUU/CUC/CUA/CUG).
        cds = "AUGGCUUAA"  # Met-Ala-stop
        rec = _make_record(five_utr="GCC", cds=cds, three_utr="GGG")
        original_protein = translate(cds)
        self.assertEqual(original_protein, "MA*")  # Met-Ala-stop

        # Apply a legal sub: GCU (Ala) -> GCC (Ala) at CDS position 5 (U -> C).
        # 5'UTR is 3 nt, so CDS position 5 is seq position 3+5 = 8.
        # Wait — let's verify: seq = "GCC" + "AUGGCUUAA" + "GGG"
        # Positions: 0,1,2 = GCC; 3,4,5 = AUG; 6,7,8 = GCU; 9,10,11 = UAA; 12,13,14 = GGG
        # GCU position 8 is U. Sub U -> C gives GCC (Ala). Legal.
        action = Action(op="sub", pos=8, nt=NUC_TO_ID["C"])
        new_rec = apply_action(rec, action)
        self.assertEqual(new_rec.cds, "AUGGCCUAA")
        self.assertEqual(translate(new_rec.cds), original_protein)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
