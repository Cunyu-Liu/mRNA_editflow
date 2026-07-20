"""Tests for P1-12 Innovation 2: Counterfactual Cross-Region Synergy RL.

Covers:
- LambdaSchedule
- build_region_restricted_mask
- make_tiny_synergy_mdp
- SynergyREINFORCE.collect_synergy_sample
- SynergyREINFORCE.compute_synergy_loss
- SynergyREINFORCE.update_synergy
- SynergyREINFORCE.train_synergy
- synergy_convergence_check (tiny MDP)
"""
from __future__ import annotations

import unittest

import torch

from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.rl.action_space import build_legal_action_mask
from mrna_editflow.rl.synergy import (
    ALL_REGIONS,
    LambdaSchedule,
    SynergyConfig,
    SynergyREINFORCE,
    build_region_restricted_mask,
    make_tiny_synergy_mdp,
    synergy_convergence_check,
)
from mrna_editflow.rl.policy import Policy, PolicyConfig
from mrna_editflow.rl.tiny_mdp import TinyTrainableModel


def _make_policy_and_mdp(
    target: str = "AAACCCCCCCGGG",
    initial: str = "UUUGGGGGGGAAA",
    max_steps: int = 12,
) -> tuple:
    """Build a synergy policy + MDP for testing."""
    device = torch.device("cpu")
    mdp = make_tiny_synergy_mdp(
        target_seq=target, initial_seq=initial, max_steps=max_steps
    )
    model = TinyTrainableModel(vocab_dim=4, hidden=8)
    backbone = type(
        "B", (), {"out_dim": 8, "forward": lambda self, *a, **k: None}
    )()
    policy = Policy(model=model, backbone=backbone, cfg=PolicyConfig(), device=device)
    return policy, mdp


class TestLambdaSchedule(unittest.TestCase):
    """Test LambdaSchedule."""

    def test_warmup_returns_zero(self) -> None:
        sched = LambdaSchedule(warmup_steps=10, anneal_steps=20, final_lambda=1.0)
        self.assertEqual(sched(0), 0.0)
        self.assertEqual(sched(9), 0.0)

    def test_anneal_increases_linearly(self) -> None:
        sched = LambdaSchedule(warmup_steps=10, anneal_steps=20, final_lambda=1.0)
        self.assertAlmostEqual(sched(10), 0.0)
        self.assertAlmostEqual(sched(20), 0.5)
        self.assertAlmostEqual(sched(30), 1.0)

    def test_final_returns_final_lambda(self) -> None:
        sched = LambdaSchedule(warmup_steps=10, anneal_steps=20, final_lambda=1.0)
        self.assertEqual(sched(100), 1.0)
        self.assertEqual(sched(1000), 1.0)

    def test_custom_final_lambda(self) -> None:
        sched = LambdaSchedule(warmup_steps=0, anneal_steps=10, final_lambda=0.5)
        self.assertAlmostEqual(sched(10), 0.5)
        self.assertEqual(sched(20), 0.5)


class TestRegionRestrictedMask(unittest.TestCase):
    """Test build_region_restricted_mask."""

    def test_joint_mask_allows_all_regions(self) -> None:
        record = MRNARecord(
            transcript_id="T", five_utr="AAA", cds="GCCGCC", three_utr="GGG"
        )
        device = torch.device("cpu")
        full_mask = build_region_restricted_mask(record, device, allowed_region=None)
        # Compare with the unrestricted mask.
        expected = build_legal_action_mask(record, device)
        self.assertTrue(torch.equal(full_mask.ins_mask, expected.ins_mask))
        self.assertTrue(torch.equal(full_mask.sub_mask, expected.sub_mask))

    def test_restricted_mask_blocks_other_regions(self) -> None:
        """Single-region mask should only allow edits in the chosen region."""
        record = MRNARecord(
            transcript_id="T", five_utr="AAA", cds="GCCGCC", three_utr="GGG"
        )
        device = torch.device("cpu")
        # Restrict to 5'UTR (region 0, positions 0..2).
        mask_5utr = build_region_restricted_mask(record, device, allowed_region=0)
        # CDS positions (3..8) should have no legal ins/sub.
        self.assertFalse(mask_5utr.ins_mask[3:9].any())
        self.assertFalse(mask_5utr.sub_mask[3:9].any())
        # 3'UTR positions (9..11) should also have no legal ins/sub.
        self.assertFalse(mask_5utr.ins_mask[9:].any())
        # 5'UTR positions (0..2) should have some legal ins.
        self.assertTrue(mask_5utr.ins_mask[:3].any())

    def test_restricted_to_cds(self) -> None:
        record = MRNARecord(
            transcript_id="T", five_utr="AAA", cds="GCCGCC", three_utr="GGG"
        )
        device = torch.device("cpu")
        mask_cds = build_region_restricted_mask(record, device, allowed_region=1)
        # 5'UTR and 3'UTR should be blocked.
        self.assertFalse(mask_cds.ins_mask[:3].any())
        self.assertFalse(mask_cds.ins_mask[9:].any())
        # CDS should have some legal sub (synonymous).
        # Note: CDS ins is forbidden by default (codon_indel=False).
        self.assertFalse(mask_cds.ins_mask[3:9].any())  # no ins in CDS
        # But CDS sub should have some legal positions (synonymous subs).
        # GCC -> GCU, GGC, etc. — there should be at least one.
        # Actually, for "GCCGCC", synonymous subs exist.
        self.assertTrue(mask_cds.sub_mask[3:9].any())


class TestTinySynergyMDP(unittest.TestCase):
    """Test make_tiny_synergy_mdp."""

    def test_default_mdp_has_3_regions(self) -> None:
        mdp = make_tiny_synergy_mdp()
        region_ids = mdp.initial_record.region_ids()
        self.assertEqual(len(set(region_ids)), 3)
        # 5'UTR = 0, CDS = 1, 3'UTR = 2.
        self.assertIn(0, region_ids)
        self.assertIn(1, region_ids)
        self.assertIn(2, region_ids)

    def test_target_and_initial_differ(self) -> None:
        mdp = make_tiny_synergy_mdp()
        self.assertNotEqual(mdp.target_seq, mdp.initial_record.seq)

    def test_default_layout(self) -> None:
        mdp = make_tiny_synergy_mdp()
        # Default: 3 + 6 + 3 = 12.
        self.assertEqual(len(mdp.target_seq), 12)
        region_ids = mdp.initial_record.region_ids()
        self.assertEqual(sum(1 for r in region_ids if r == 0), 3)  # 5'UTR
        self.assertEqual(sum(1 for r in region_ids if r == 1), 6)  # CDS
        self.assertEqual(sum(1 for r in region_ids if r == 2), 3)  # 3'UTR

    def test_cds_length_is_multiple_of_3(self) -> None:
        mdp = make_tiny_synergy_mdp()
        cds_len = len(mdp.initial_record.cds)
        self.assertEqual(cds_len % 3, 0)


class TestSynergyCollectSample(unittest.TestCase):
    """Test SynergyREINFORCE.collect_synergy_sample."""

    def test_returns_joint_and_singles(self) -> None:
        policy, mdp = _make_policy_and_mdp()
        trainer = SynergyREINFORCE(policy=policy, mdp=mdp, cfg=SynergyConfig())
        sample = trainer.collect_synergy_sample(step=100)
        self.assertIsNotNone(sample.joint)
        self.assertEqual(len(sample.singles), 3)  # 3 counterfactual regions
        for single in sample.singles:
            self.assertIn(single.region, ALL_REGIONS)

    def test_synergy_reward_computed(self) -> None:
        policy, mdp = _make_policy_and_mdp()
        trainer = SynergyREINFORCE(policy=policy, mdp=mdp, cfg=SynergyConfig())
        sample = trainer.collect_synergy_sample(step=100)
        self.assertIsInstance(sample.synergy_reward, float)
        self.assertGreaterEqual(abs(sample.synergy_reward), 0.0)

    def test_lambda_zero_during_warmup(self) -> None:
        """During warmup, lambda=0, so synergy = R_joint."""
        policy, mdp = _make_policy_and_mdp()
        cfg = SynergyConfig(
            lambda_schedule=LambdaSchedule(warmup_steps=100, anneal_steps=100)
        )
        trainer = SynergyREINFORCE(policy=policy, mdp=mdp, cfg=cfg)
        sample = trainer.collect_synergy_sample(step=0)
        self.assertAlmostEqual(sample.lambda_used, 0.0)
        self.assertAlmostEqual(sample.synergy_reward, sample.joint.total_reward)

    def test_lambda_one_after_anneal(self) -> None:
        """After anneal, lambda=1, so synergy = R_joint - sum(R_single)."""
        policy, mdp = _make_policy_and_mdp()
        cfg = SynergyConfig(
            lambda_schedule=LambdaSchedule(warmup_steps=0, anneal_steps=10)
        )
        trainer = SynergyREINFORCE(policy=policy, mdp=mdp, cfg=cfg)
        sample = trainer.collect_synergy_sample(step=100)
        self.assertAlmostEqual(sample.lambda_used, 1.0)
        expected = sample.joint.total_reward - sum(
            s.total_reward for s in sample.singles
        )
        self.assertAlmostEqual(sample.synergy_reward, expected, places=5)


class TestSynergyLoss(unittest.TestCase):
    """Test SynergyREINFORCE.compute_synergy_loss."""

    def test_loss_is_differentiable(self) -> None:
        policy, mdp = _make_policy_and_mdp()
        trainer = SynergyREINFORCE(policy=policy, mdp=mdp, cfg=SynergyConfig())
        samples = [trainer.collect_synergy_sample(step=10) for _ in range(2)]
        loss, info = trainer.compute_synergy_loss(samples)
        self.assertTrue(loss.requires_grad)
        self.assertGreater(info["n_samples"], 0)

    def test_loss_info_has_metrics(self) -> None:
        policy, mdp = _make_policy_and_mdp()
        trainer = SynergyREINFORCE(policy=policy, mdp=mdp, cfg=SynergyConfig())
        samples = [trainer.collect_synergy_sample(step=10) for _ in range(2)]
        _, info = trainer.compute_synergy_loss(samples)
        self.assertIn("mean_synergy_reward", info)
        self.assertIn("mean_joint_reward", info)
        self.assertIn("mean_single_reward", info)
        self.assertIn("mean_lambda", info)

    def test_empty_samples_raises(self) -> None:
        policy, mdp = _make_policy_and_mdp()
        trainer = SynergyREINFORCE(policy=policy, mdp=mdp, cfg=SynergyConfig())
        with self.assertRaises(ValueError):
            trainer.compute_synergy_loss([])


class TestSynergyUpdate(unittest.TestCase):
    """Test SynergyREINFORCE.update_synergy."""

    def test_update_changes_params(self) -> None:
        policy, mdp = _make_policy_and_mdp()
        trainer = SynergyREINFORCE(policy=policy, mdp=mdp, cfg=SynergyConfig())
        samples = [trainer.collect_synergy_sample(step=10) for _ in range(4)]
        params_before = [p.clone() for p in policy.model.parameters()]
        info = trainer.update_synergy(samples)
        changed = any(
            not torch.allclose(p_before, p_after)
            for p_before, p_after in zip(params_before, policy.model.parameters())
        )
        self.assertTrue(changed, "Policy params should change after update")

    def test_update_returns_info(self) -> None:
        policy, mdp = _make_policy_and_mdp()
        trainer = SynergyREINFORCE(policy=policy, mdp=mdp, cfg=SynergyConfig())
        samples = [trainer.collect_synergy_sample(step=10) for _ in range(2)]
        info = trainer.update_synergy(samples)
        self.assertIn("loss", info)
        self.assertIn("n_samples", info)


class TestSynergyTrain(unittest.TestCase):
    """Test SynergyREINFORCE.train_synergy."""

    def test_train_returns_history(self) -> None:
        policy, mdp = _make_policy_and_mdp()
        trainer = SynergyREINFORCE(policy=policy, mdp=mdp, cfg=SynergyConfig())
        history = trainer.train_synergy(n_episodes=20, batch_size=4)
        self.assertGreater(len(history), 0)
        for h in history:
            self.assertIn("batch_idx", h)
            self.assertIn("mean_synergy_reward", h)

    def test_lambda_progresses_through_schedule(self) -> None:
        """Over training, lambda should increase from 0 to 1."""
        policy, mdp = _make_policy_and_mdp()
        cfg = SynergyConfig(
            lambda_schedule=LambdaSchedule(warmup_steps=5, anneal_steps=10)
        )
        trainer = SynergyREINFORCE(policy=policy, mdp=mdp, cfg=cfg)
        history = trainer.train_synergy(n_episodes=80, batch_size=4)
        # First batch lambda should be ~0, last should be ~1.
        first_lambdas = [h.get("mean_lambda", 0) for h in history[:3]]
        last_lambdas = [h.get("mean_lambda", 0) for h in history[-3:]]
        self.assertLess(sum(first_lambdas) / len(first_lambdas), 0.3)
        self.assertGreater(sum(last_lambdas) / len(last_lambdas), 0.7)


class TestSynergyConvergenceCheck(unittest.TestCase):
    """Test synergy_convergence_check on a tiny MDP."""

    def test_synergy_becomes_positive(self) -> None:
        """Synergy reward should become positive (joint > sum of singles)."""
        policy, mdp = _make_policy_and_mdp()
        trainer = SynergyREINFORCE(policy=policy, mdp=mdp, cfg=SynergyConfig())
        result = synergy_convergence_check(trainer, n_episodes=120, batch_size=4)
        self.assertTrue(
            result["synergy_significant"],
            f"Synergy not significant: {result}",
        )
        self.assertGreater(result["mean_synergy_last"], 0.0)
        self.assertTrue(result["converged"], f"Did not converge: {result}")

    def test_lambda_reaches_final(self) -> None:
        policy, mdp = _make_policy_and_mdp()
        trainer = SynergyREINFORCE(policy=policy, mdp=mdp, cfg=SynergyConfig())
        result = synergy_convergence_check(trainer, n_episodes=120, batch_size=4)
        self.assertAlmostEqual(result["mean_lambda_last"], 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
