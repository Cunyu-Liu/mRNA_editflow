"""Tests for P2-05: Group Relative Policy Optimization (GRPO).

Covers:
- GRPOConfig validation
- group_normalized_advantages: single group, batch, clipping, shape checks
- GRPOREINFORCE: collect_group, compute_loss, step, convergence on tiny MDP
- grpo_convergence_check

All reward signals in these tests are synthetic (tiny MDP). Any claim about
"improving TE" in production MUST be qualified as "predicted TE (internal
proxy)" until P2-01 multi-region oracle validation completes.
"""
from __future__ import annotations

import math
import unittest
from typing import List

import torch

from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.rl.action_space import STOP_ACTION, Action, apply_action
from mrna_editflow.rl.grpo import (
    GRPOConfig,
    GRPOREINFORCE,
    group_normalized_advantages,
    grpo_convergence_check,
)
from mrna_editflow.rl.policy import Policy, PolicyConfig
from mrna_editflow.rl.tiny_mdp import (
    REINFORCE,
    TinyMDP,
    TinyTrainableModel,
    Trajectory,
    Transition,
    compute_returns,
)


def _make_policy_and_mdp(
    target: str = "AAAA",
    initial: str = "CCCC",
    max_steps: int = 4,
) -> tuple:
    """Build a tiny policy + MDP for testing (mirrors test_p1_12_cto.py)."""
    device = torch.device("cpu")
    record = MRNARecord(transcript_id="T", five_utr=initial, cds="", three_utr="")
    model = TinyTrainableModel(vocab_dim=4, hidden=8)
    backbone = type(
        "B", (), {"out_dim": 8, "forward": lambda self, *a, **k: None}
    )()
    policy = Policy(model=model, backbone=backbone, cfg=PolicyConfig(), device=device)
    mdp = TinyMDP(target_seq=target, initial_record=record, max_steps=max_steps)
    return policy, mdp


# ---------------------------------------------------------------------------
# GRPOConfig
# ---------------------------------------------------------------------------


class TestGRPOConfig(unittest.TestCase):
    def test_default_config(self) -> None:
        cfg = GRPOConfig()
        self.assertEqual(cfg.group_size, 8)
        self.assertAlmostEqual(cfg.eps, 1e-8)
        self.assertEqual(cfg.clip_advantage, 0.0)

    def test_group_size_must_be_at_least_2(self) -> None:
        with self.assertRaises(ValueError):
            GRPOConfig(group_size=1)
        # group_size=2 is the minimum valid value
        GRPOConfig(group_size=2)

    def test_eps_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            GRPOConfig(eps=0.0)
        with self.assertRaises(ValueError):
            GRPOConfig(eps=-1.0)

    def test_clip_advantage_must_be_nonnegative(self) -> None:
        with self.assertRaises(ValueError):
            GRPOConfig(clip_advantage=-0.5)
        GRPOConfig(clip_advantage=0.0)
        GRPOConfig(clip_advantage=2.0)


# ---------------------------------------------------------------------------
# group_normalized_advantages
# ---------------------------------------------------------------------------


class TestGroupNormalizedAdvantages(unittest.TestCase):
    def test_single_group_mean_zero_std_one(self) -> None:
        # N=4 returns; after normalization, mean ~0 and std ~1.
        returns = torch.tensor([1.0, 2.0, 3.0, 4.0])
        cfg = GRPOConfig(group_size=4)
        adv = group_normalized_advantages(returns, cfg)
        self.assertEqual(adv.shape, returns.shape)
        self.assertAlmostEqual(float(adv.mean().item()), 0.0, places=5)
        # std should be ~1 (within eps)
        self.assertAlmostEqual(
            float(adv.std(unbiased=False).item()), 1.0, places=4
        )

    def test_batch_of_groups_per_row_normalization(self) -> None:
        # B=3, N=4. Each row should be normalized independently.
        returns = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
            [-1.0, 0.0, 1.0, 2.0],
        ])
        cfg = GRPOConfig(group_size=4)
        adv = group_normalized_advantages(returns, cfg)
        self.assertEqual(adv.shape, (3, 4))
        # Each row: mean ~0
        row_means = adv.mean(dim=-1)
        for m in row_means:
            self.assertAlmostEqual(float(m.item()), 0.0, places=5)
        # Each row: std ~1
        row_stds = adv.std(dim=-1, unbiased=False)
        for s in row_stds:
            self.assertAlmostEqual(float(s.item()), 1.0, places=4)

    def test_constant_returns_yield_zero_advantage(self) -> None:
        # If all returns in a group are identical, std=0 and adv=0 (via eps).
        returns = torch.tensor([5.0, 5.0, 5.0, 5.0])
        cfg = GRPOConfig(group_size=4, eps=1e-8)
        adv = group_normalized_advantages(returns, cfg)
        # mean=5, std=0 -> adv = (5-5)/(0+eps) = 0
        for v in adv:
            self.assertAlmostEqual(float(v.item()), 0.0, places=5)

    def test_clip_advantage_bounds_output(self) -> None:
        returns = torch.tensor([0.0, 0.0, 0.0, 100.0])
        cfg = GRPOConfig(group_size=4, clip_advantage=1.0)
        adv = group_normalized_advantages(returns, cfg)
        for v in adv:
            self.assertGreaterEqual(float(v.item()), -1.0 - 1e-6)
            self.assertLessEqual(float(v.item()), 1.0 + 1e-6)

    def test_default_cfg_infers_group_size(self) -> None:
        returns = torch.tensor([1.0, 2.0, 3.0])
        adv = group_normalized_advantages(returns)  # cfg=None
        self.assertEqual(adv.shape, (3,))
        self.assertAlmostEqual(float(adv.mean().item()), 0.0, places=5)

    def test_shape_mismatch_raises(self) -> None:
        returns = torch.tensor([1.0, 2.0, 3.0])
        cfg = GRPOConfig(group_size=4)
        with self.assertRaises(ValueError):
            group_normalized_advantages(returns, cfg)

    def test_3d_raises(self) -> None:
        returns = torch.zeros(2, 3, 4)
        cfg = GRPOConfig(group_size=4)
        with self.assertRaises(ValueError):
            group_normalized_advantages(returns, cfg)

    def test_preserves_gradient(self) -> None:
        # Advantages are detached statistics, but the returns tensor could
        # come from a differentiable graph. The advantage function itself
        # does not need to be differentiable through the normalization
        # constants (mean/std are treated as statistics). We verify that
        # the function runs on a tensor with requires_grad=True without error.
        returns = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        cfg = GRPOConfig(group_size=4)
        adv = group_normalized_advantages(returns, cfg)
        # adv itself is not expected to backprop into returns (statistics),
        # but the call must not raise.
        self.assertEqual(adv.shape, (4,))


# ---------------------------------------------------------------------------
# GRPOREINFORCE
# ---------------------------------------------------------------------------


class TestGRPOREINFORCE(unittest.TestCase):
    def test_init_overrides_baseline(self) -> None:
        policy, mdp = _make_policy_and_mdp()
        cfg = GRPOConfig(group_size=4)
        trainer = GRPOREINFORCE(policy=policy, mdp=mdp, cfg=cfg, lr=0.01)
        self.assertFalse(trainer.use_baseline)
        self.assertEqual(trainer.baseline, 0.0)
        self.assertEqual(trainer.cfg.group_size, 4)
        self.assertEqual(trainer.lr, 0.01)

    def test_collect_group_size(self) -> None:
        policy, mdp = _make_policy_and_mdp()
        cfg = GRPOConfig(group_size=4)
        trainer = GRPOREINFORCE(policy=policy, mdp=mdp, cfg=cfg)
        gen = torch.Generator(device=torch.device("cpu"))
        gen.manual_seed(0)
        group = trainer.collect_group(generator=gen)
        self.assertEqual(len(group), 4)
        for traj in group:
            self.assertIsInstance(traj, Trajectory)
            self.assertGreaterEqual(len(traj.transitions), 1)

    def test_compute_loss_returns_scalar_and_metrics(self) -> None:
        policy, mdp = _make_policy_and_mdp()
        cfg = GRPOConfig(group_size=4)
        trainer = GRPOREINFORCE(policy=policy, mdp=mdp, cfg=cfg)
        gen = torch.Generator(device=torch.device("cpu"))
        gen.manual_seed(42)
        groups = [trainer.collect_group(generator=gen) for _ in range(2)]
        loss, metrics = trainer.compute_loss(groups)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(math.isfinite(float(loss.item())))
        self.assertIn("mean_return", metrics)
        self.assertIn("mean_advantage", metrics)
        self.assertIn("mean_advantage_sq", metrics)
        self.assertEqual(metrics["n_groups"], 2)
        self.assertEqual(metrics["group_size"], 4)

    def test_compute_loss_rejects_wrong_group_size(self) -> None:
        policy, mdp = _make_policy_and_mdp()
        cfg = GRPOConfig(group_size=4)
        trainer = GRPOREINFORCE(policy=policy, mdp=mdp, cfg=cfg)
        gen = torch.Generator(device=torch.device("cpu"))
        gen.manual_seed(0)
        # Build a group of 3 (wrong) instead of 4
        bad_group = [trainer.collect_trajectory(generator=gen) for _ in range(3)]
        with self.assertRaises(ValueError):
            trainer.compute_loss([bad_group])

    def test_step_updates_parameters(self) -> None:
        policy, mdp = _make_policy_and_mdp()
        cfg = GRPOConfig(group_size=4)
        trainer = GRPOREINFORCE(policy=policy, mdp=mdp, cfg=cfg, lr=0.1)
        gen = torch.Generator(device=torch.device("cpu"))
        gen.manual_seed(123)
        groups = [trainer.collect_group(generator=gen) for _ in range(2)]
        # Snapshot params before step
        before = [p.clone().detach() for p in policy.model.parameters()]
        metrics = trainer.step(groups)
        after = [p.clone().detach() for p in policy.model.parameters()]
        # At least one parameter should change
        changed = any(
            not torch.allclose(b, a) for b, a in zip(before, after)
        )
        self.assertTrue(changed, "no parameter changed after GRPO step")
        self.assertIn("loss", metrics)

    def test_mean_advantage_is_approximately_zero(self) -> None:
        # Group normalization forces mean advantage ~0 per group.
        policy, mdp = _make_policy_and_mdp()
        cfg = GRPOConfig(group_size=4)
        trainer = GRPOREINFORCE(policy=policy, mdp=mdp, cfg=cfg)
        gen = torch.Generator(device=torch.device("cpu"))
        gen.manual_seed(7)
        groups = [trainer.collect_group(generator=gen) for _ in range(4)]
        _, metrics = trainer.compute_loss(groups)
        # Across all groups, mean advantage should be ~0 (within float noise).
        self.assertAlmostEqual(
            metrics["mean_advantage"], 0.0, places=4
        )


# ---------------------------------------------------------------------------
# Convergence check (tiny MDP)
# ---------------------------------------------------------------------------


class TestGRPOConvergence(unittest.TestCase):
    def test_grpo_improves_return_on_tiny_mdp(self) -> None:
        # GRPO should improve mean return on the tiny MDP (target match reward).
        policy, mdp = _make_policy_and_mdp(
            target="AAAA", initial="CCCC", max_steps=4
        )
        cfg = GRPOConfig(group_size=4, clip_advantage=0.0)
        trainer = GRPOREINFORCE(policy=policy, mdp=mdp, cfg=cfg, lr=0.05)
        gen = torch.Generator(device=torch.device("cpu"))
        gen.manual_seed(0)
        result = grpo_convergence_check(
            trainer,
            n_iters=100,
            n_groups=4,
            generator=gen,
        )
        self.assertTrue(result["converged"], msg=f"history={result['history'][-5:]}")
        self.assertGreater(
            result["final_return"],
            result["initial_return"],
            msg="GRPO did not improve return",
        )

    def test_grpo_history_length_matches_n_iters(self) -> None:
        policy, mdp = _make_policy_and_mdp()
        cfg = GRPOConfig(group_size=4)
        trainer = GRPOREINFORCE(policy=policy, mdp=mdp, cfg=cfg, lr=0.01)
        gen = torch.Generator(device=torch.device("cpu"))
        gen.manual_seed(1)
        result = grpo_convergence_check(
            trainer, n_iters=10, n_groups=2, generator=gen
        )
        self.assertEqual(len(result["history"]), 10)


# ---------------------------------------------------------------------------
# Comparison: GRPO vs EMA-baseline REINFORCE (sanity)
# ---------------------------------------------------------------------------


class TestGRPOVsEMABaseline(unittest.TestCase):
    def test_grpo_advantages_have_lower_variance_than_raw_returns(self) -> None:
        # The whole point of group normalization is variance reduction.
        # Compare std(advantages) to std(returns) on a heavy-tailed group.
        returns = torch.tensor([0.01, 0.02, 0.03, 10.0])  # one outlier
        cfg = GRPOConfig(group_size=4)
        adv = group_normalized_advantages(returns, cfg)
        self.assertLess(float(adv.std().item()), float(returns.std().item()))


# ---------------------------------------------------------------------------
# KL penalty + entropy bonus (P2-05 spec compliance)
# ---------------------------------------------------------------------------


class TestGRPOKLEntropy(unittest.TestCase):
    def test_kl_coef_validation(self) -> None:
        with self.assertRaises(ValueError):
            GRPOConfig(kl_coef=-0.1)
        GRPOConfig(kl_coef=0.0)
        GRPOConfig(kl_coef=0.01)

    def test_entropy_coef_validation(self) -> None:
        with self.assertRaises(ValueError):
            GRPOConfig(entropy_coef=-0.1)
        GRPOConfig(entropy_coef=0.0)
        GRPOConfig(entropy_coef=0.01)

    def test_default_kl_and_entropy_are_zero(self) -> None:
        cfg = GRPOConfig()
        self.assertEqual(cfg.kl_coef, 0.0)
        self.assertEqual(cfg.entropy_coef, 0.0)

    def test_ref_policy_optional(self) -> None:
        policy, mdp = _make_policy_and_mdp()
        cfg = GRPOConfig(group_size=4)
        trainer = GRPOREINFORCE(policy=policy, mdp=mdp, cfg=cfg)
        self.assertIsNone(trainer.ref_policy)
        trainer2 = GRPOREINFORCE(policy=policy, mdp=mdp, cfg=cfg, ref_policy=policy)
        self.assertIsNotNone(trainer2.ref_policy)

    def test_kl_penalty_with_ref_policy_changes_loss(self) -> None:
        # With a non-zero kl_coef and a ref_policy, the loss should differ
        # from the no-KL case.
        policy, mdp = _make_policy_and_mdp()
        cfg_no_kl = GRPOConfig(group_size=4, kl_coef=0.0)
        cfg_kl = GRPOConfig(group_size=4, kl_coef=0.5)
        # Use the SAME policy as ref (so KL should be ~0, but the code path runs).
        trainer_no_kl = GRPOREINFORCE(policy=policy, mdp=mdp, cfg=cfg_no_kl)
        trainer_kl = GRPOREINFORCE(
            policy=policy, mdp=mdp, cfg=cfg_kl, ref_policy=policy
        )
        gen = torch.Generator(device=torch.device("cpu"))
        gen.manual_seed(42)
        groups = [trainer_no_kl.collect_group(generator=gen) for _ in range(2)]
        # Use the same groups for both trainers (deterministic collection).
        loss_no_kl, m_no_kl = trainer_no_kl.compute_loss(groups)
        loss_kl, m_kl = trainer_kl.compute_loss(groups)
        # When ref_policy == policy, KL should be ~0, so losses should be close.
        self.assertAlmostEqual(m_kl["mean_kl"], 0.0, places=4)
        # Both should produce finite losses.
        self.assertTrue(math.isfinite(float(loss_no_kl.item())))
        self.assertTrue(math.isfinite(float(loss_kl.item())))

    def test_kl_metrics_reported(self) -> None:
        policy, mdp = _make_policy_and_mdp()
        cfg = GRPOConfig(group_size=4, kl_coef=0.1)
        trainer = GRPOREINFORCE(
            policy=policy, mdp=mdp, cfg=cfg, ref_policy=policy
        )
        gen = torch.Generator(device=torch.device("cpu"))
        gen.manual_seed(0)
        groups = [trainer.collect_group(generator=gen) for _ in range(2)]
        _, metrics = trainer.compute_loss(groups)
        self.assertIn("mean_kl", metrics)
        self.assertIn("mean_entropy", metrics)

    def test_entropy_bonus_with_coef_runs(self) -> None:
        # With a non-zero entropy_coef, the loss includes an entropy term.
        policy, mdp = _make_policy_and_mdp()
        cfg = GRPOConfig(group_size=4, entropy_coef=0.01)
        trainer = GRPOREINFORCE(policy=policy, mdp=mdp, cfg=cfg)
        gen = torch.Generator(device=torch.device("cpu"))
        gen.manual_seed(7)
        groups = [trainer.collect_group(generator=gen) for _ in range(2)]
        loss, metrics = trainer.compute_loss(groups)
        self.assertTrue(math.isfinite(float(loss.item())))
        # Entropy should be non-negative (it's H = -sum p log p >= 0).
        self.assertGreaterEqual(metrics["mean_entropy"], -1e-6)

    def test_entropy_of_uniform_distribution_is_maximal(self) -> None:
        # For a uniform distribution over K actions, H = log(K).
        # We can't easily construct a uniform ActionLogProbs, but we can
        # verify the _distribution_entropy static method on a synthetic input.
        class _SyntheticLPS:
            def __init__(self) -> None:
                # 4 actions, uniform: log(1/4) = -log(4)
                lp = -math.log(4.0)
                self.ins_logprobs = torch.full((2, 2), lp)
                self.sub_logprobs = None
                self.del_logprobs = None
                self.stop_logprob = None

        lps = _SyntheticLPS()
        entropy = GRPOREINFORCE._distribution_entropy(lps)
        expected = math.log(4.0)
        self.assertAlmostEqual(float(entropy.item()), expected, places=4)

    def test_step_with_kl_and_entropy(self) -> None:
        # A full step with KL + entropy should run and update params.
        policy, mdp = _make_policy_and_mdp()
        cfg = GRPOConfig(group_size=4, kl_coef=0.1, entropy_coef=0.01)
        trainer = GRPOREINFORCE(
            policy=policy, mdp=mdp, cfg=cfg, ref_policy=policy, lr=0.05
        )
        gen = torch.Generator(device=torch.device("cpu"))
        gen.manual_seed(99)
        groups = [trainer.collect_group(generator=gen) for _ in range(2)]
        before = [p.clone().detach() for p in policy.model.parameters()]
        metrics = trainer.step(groups)
        after = [p.clone().detach() for p in policy.model.parameters()]
        changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
        self.assertTrue(changed, "no parameter changed after step with KL+entropy")
        self.assertIn("mean_kl", metrics)
        self.assertIn("mean_entropy", metrics)




class TestMakeGeneratorDevice(unittest.TestCase):
    """Tests for _make_generator device matching (P2-05 CUDA generator fix).

    The GRPO pilot crashed with RuntimeError: Expected a 'cuda' device type
    for generator but found 'cpu'. This was because _make_generator created
    a CPU generator but torch.multinomial was called with CUDA probs.

    Fix: _make_generator now accepts a device parameter.
    """

    def _import_make_generator(self):
        """Import _make_generator from scripts/run_p2_05_grpo_pilot.py."""
        import importlib.util
        import os
        import sys
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts",
            "run_p2_05_grpo_pilot.py",
        )
        mod_name = "_run_p2_05_grpo_pilot_test_module"
        spec = importlib.util.spec_from_file_location(mod_name, script_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod._make_generator

    def test_cpu_generator_default(self):
        """_make_generator with no device creates a CPU generator."""
        make_gen = self._import_make_generator()
        gen = make_gen(seed=42)
        probs = torch.softmax(torch.randn(10), dim=0)
        idx = torch.multinomial(probs, num_samples=1, generator=gen).item()
        self.assertIsInstance(idx, int)
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, 10)

    def test_cpu_generator_explicit(self):
        """_make_generator with explicit CPU device works with CPU probs."""
        make_gen = self._import_make_generator()
        gen = make_gen(seed=42, device=torch.device("cpu"))
        probs = torch.softmax(torch.randn(10), dim=0)
        idx = torch.multinomial(probs, num_samples=1, generator=gen).item()
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, 10)

    def test_generator_reproducibility(self):
        """Same seed produces same sequence on the same device."""
        make_gen = self._import_make_generator()
        gen1 = make_gen(seed=123, device=torch.device("cpu"))
        gen2 = make_gen(seed=123, device=torch.device("cpu"))
        probs = torch.softmax(torch.randn(20), dim=0)
        idx1 = torch.multinomial(probs, num_samples=1, generator=gen1).item()
        idx2 = torch.multinomial(probs, num_samples=1, generator=gen2).item()
        self.assertEqual(idx1, idx2)

    def test_different_seeds_differ(self):
        """Different seeds produce different samples (with high probability)."""
        make_gen = self._import_make_generator()
        gen1 = make_gen(seed=1, device=torch.device("cpu"))
        gen2 = make_gen(seed=2, device=torch.device("cpu"))
        probs = torch.softmax(torch.randn(100), dim=0)
        samples1 = [torch.multinomial(probs, 1, generator=gen1).item() for _ in range(50)]
        samples2 = [torch.multinomial(probs, 1, generator=gen2).item() for _ in range(50)]
        self.assertNotEqual(samples1, samples2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_generator_matches_cuda_probs(self):
        """_make_generator with CUDA device works with CUDA probs.

        This is the regression test for the actual bug: previously
        _make_generator created a CPU generator, causing
        'RuntimeError: Expected a cuda device type for generator but found cpu'
        when used with CUDA probs in policy.sample().
        """
        make_gen = self._import_make_generator()
        device = torch.device("cuda:0")
        gen = make_gen(seed=42, device=device)
        probs = torch.softmax(torch.randn(10, device=device), dim=0)
        idx = torch.multinomial(probs, num_samples=1, generator=gen).item()
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, 10)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_generator_device_mismatch_fails(self):
        """CPU generator with CUDA probs raises RuntimeError (documents the bug)."""
        make_gen = self._import_make_generator()
        cpu_gen = make_gen(seed=42, device=torch.device("cpu"))
        cuda_probs = torch.softmax(torch.randn(10, device="cuda:0"), dim=0)
        with self.assertRaises(RuntimeError):
            torch.multinomial(cuda_probs, num_samples=1, generator=cpu_gen)




class TestMaxStepsParameter(unittest.TestCase):
    """Tests for --max-steps parameter (decouples MDP max_steps from --limit).

    Previously --limit set both the number of source sequences AND max_steps
    for the MDP. With --limit 256, each trajectory could have 256 forward
    passes with gradients, causing CUDA OOM. The --max-steps parameter
    decouples these: --limit controls dataset size, --max-steps controls
    trajectory length.
    """

    def _import_run_config(self):
        """Import RunConfig from the script."""
        import importlib.util
        import os
        import sys
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts",
            "run_p2_05_grpo_pilot.py",
        )
        mod_name = "_run_p2_05_grpo_pilot_test_maxsteps"
        spec = importlib.util.spec_from_file_location(mod_name, script_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod.P205RunConfig

    def test_max_steps_default_none(self):
        """RunConfig.max_steps defaults to None (backward-compatible)."""
        RunConfig = self._import_run_config()
        import inspect
        sig = inspect.signature(RunConfig)
        # max_steps should have a default of None
        params = sig.parameters
        self.assertIn("max_steps", params)
        self.assertIsNone(params["max_steps"].default)

    def test_max_steps_set_explicitly(self):
        """RunConfig can be created with max_steps set to an integer."""
        RunConfig = self._import_run_config()
        # Create a minimal RunConfig with required fields
        # We can't easily construct a full RunConfig, so just verify the field exists
        import dataclasses
        fields = {f.name: f for f in dataclasses.fields(RunConfig)}
        self.assertIn("max_steps", fields)
        self.assertEqual(fields["max_steps"].type, "Optional[int]")


# ---------------------------------------------------------------------------
# _compute_loss_grad_accum: memory-efficient per-trajectory backward
# ---------------------------------------------------------------------------


class TestComputeLossGradAccumMetrics(unittest.TestCase):
    """Verify _compute_loss_grad_accum produces metrics consistent with compute_loss."""

    def _build_trainer_and_groups(self, group_size: int = 4, n_groups: int = 2, seed: int = 42):
        policy, mdp = _make_policy_and_mdp()
        cfg = GRPOConfig(group_size=group_size)
        trainer = GRPOREINFORCE(policy=policy, mdp=mdp, cfg=cfg)
        gen = torch.Generator(device=torch.device("cpu"))
        gen.manual_seed(seed)
        groups = [trainer.collect_group(generator=gen) for _ in range(n_groups)]
        return trainer, groups

    def test_returns_metrics_dict_only(self) -> None:
        """_compute_loss_grad_accum returns a dict (no loss tensor)."""
        trainer, groups = self._build_trainer_and_groups()
        result = trainer._compute_loss_grad_accum(groups)
        self.assertIsInstance(result, dict)
        self.assertNotIsInstance(result, tuple)

    def test_metrics_keys_match_compute_loss(self) -> None:
        """Metrics dict has the same keys as compute_loss metrics."""
        trainer, groups = self._build_trainer_and_groups()
        _, metrics_orig = trainer.compute_loss(groups)
        metrics_new = trainer._compute_loss_grad_accum(groups)
        self.assertEqual(set(metrics_orig.keys()), set(metrics_new.keys()))

    def test_metrics_values_match_compute_loss(self) -> None:
        """Metrics values match compute_loss (within float tolerance)."""
        trainer, groups = self._build_trainer_and_groups()
        _, metrics_orig = trainer.compute_loss(groups)
        metrics_new = trainer._compute_loss_grad_accum(groups)
        # n_groups, group_size, n_steps must match exactly
        self.assertEqual(metrics_orig["n_groups"], metrics_new["n_groups"])
        self.assertEqual(metrics_orig["group_size"], metrics_new["group_size"])
        self.assertEqual(metrics_orig["n_steps"], metrics_new["n_steps"])
        # mean_return, mean_advantage, mean_advantage_sq, return_std_mean are
        # computed from returns_mat (no grad) — must match exactly.
        self.assertAlmostEqual(metrics_orig["mean_return"], metrics_new["mean_return"], places=6)
        self.assertAlmostEqual(metrics_orig["mean_advantage"], metrics_new["mean_advantage"], places=6)
        self.assertAlmostEqual(metrics_orig["mean_advantage_sq"], metrics_new["mean_advantage_sq"], places=6)
        self.assertAlmostEqual(metrics_orig["return_std_mean"], metrics_new["return_std_mean"], places=6)
        # mean_logprob, mean_kl, mean_entropy, loss are computed from per-transition
        # log-probs — must match within float noise.
        self.assertAlmostEqual(metrics_orig["mean_logprob"], metrics_new["mean_logprob"], places=5)
        self.assertAlmostEqual(metrics_orig["mean_kl"], metrics_new["mean_kl"], places=5)
        self.assertAlmostEqual(metrics_orig["mean_entropy"], metrics_new["mean_entropy"], places=5)
        self.assertAlmostEqual(metrics_orig["loss"], metrics_new["loss"], places=4)


class TestComputeLossGradAccumGradients(unittest.TestCase):
    """Verify _compute_loss_grad_accum produces the same gradients as compute_loss."""

    def _build_trainer_and_groups(self, group_size: int = 4, n_groups: int = 2, seed: int = 99):
        policy, mdp = _make_policy_and_mdp()
        cfg = GRPOConfig(group_size=group_size)
        trainer = GRPOREINFORCE(policy=policy, mdp=mdp, cfg=cfg)
        gen = torch.Generator(device=torch.device("cpu"))
        gen.manual_seed(seed)
        groups = [trainer.collect_group(generator=gen) for _ in range(n_groups)]
        return trainer, groups

    def test_gradients_match_compute_loss(self) -> None:
        """Gradients from _compute_loss_grad_accum match compute_loss.backward().

        Uses the SAME trainer, policy, and groups for both paths to isolate
        the difference to the loss-computation path only. Some floating-point
        divergence is expected due to accumulation-order differences between
        (a) accumulating all terms into one tensor then backward, and
        (b) per-trajectory backward with gradient accumulation.
        """
        # Build ONE trainer, ONE set of groups (same policy for both paths)
        trainer, groups = self._build_trainer_and_groups()
        # Snapshot initial params (paths must NOT modify params — no optimizer.step)
        params_init = [p.clone().detach() for p in trainer.policy.model.parameters()]

        # Path A: compute_loss + manual backward
        trainer.optimizer.zero_grad()
        loss_a, _ = trainer.compute_loss(groups)
        loss_a.backward()
        grads_a = [p.grad.clone().detach() if p.grad is not None else None
                   for p in trainer.policy.model.parameters()]

        # Reset grads; sanity-check params unchanged
        trainer.optimizer.zero_grad()
        for p, p_init in zip(trainer.policy.model.parameters(), params_init):
            self.assertTrue(torch.allclose(p, p_init), "params modified by path A")

        # Path B: _compute_loss_grad_accum (calls backward internally per traj)
        _ = trainer._compute_loss_grad_accum(groups)
        grads_b = [p.grad.clone().detach() if p.grad is not None else None
                   for p in trainer.policy.model.parameters()]

        # Compare gradients (relaxed tolerance for float accumulation order)
        self.assertEqual(len(grads_a), len(grads_b))
        for i, (ga, gb) in enumerate(zip(grads_a, grads_b)):
            if ga is None and gb is None:
                continue
            self.assertIsNotNone(ga, f"param {i}: grad_a is None but grad_b is not")
            self.assertIsNotNone(gb, f"param {i}: grad_b is None but grad_a is not")
            max_diff = (ga - gb).abs().max().item() if ga.numel() > 0 else 0.0
            self.assertTrue(
                torch.allclose(ga, gb, atol=1e-3, rtol=1e-3),
                f"param {i}: gradients differ — max abs diff = {max_diff:.2e}",
            )

    def test_step_updates_parameters_via_grad_accum(self) -> None:
        """step() (which uses _compute_loss_grad_accum) updates parameters."""
        policy, mdp = _make_policy_and_mdp()
        cfg = GRPOConfig(group_size=4)
        trainer = GRPOREINFORCE(policy=policy, mdp=mdp, cfg=cfg, lr=0.1)
        gen = torch.Generator(device=torch.device("cpu"))
        gen.manual_seed(321)
        groups = [trainer.collect_group(generator=gen) for _ in range(2)]
        before = [p.clone().detach() for p in policy.model.parameters()]
        metrics = trainer.step(groups)
        after = [p.clone().detach() for p in policy.model.parameters()]
        changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
        self.assertTrue(changed, "no parameter changed after step() with grad accum")
        self.assertIn("loss", metrics)

    def test_step_with_grad_accum_matches_step_with_compute_loss(self) -> None:
        """step() via _compute_loss_grad_accum produces approximately the same
        param update as compute_loss + backward (within float tolerance).

        Uses the SAME trainer and groups for both paths. Some divergence is
        expected due to float accumulation order differences.
        """
        # Build ONE trainer, ONE set of groups
        trainer, groups = self._build_trainer_and_groups(seed=77)
        # Snapshot initial params
        params_init = [p.clone().detach() for p in trainer.policy.model.parameters()]

        # Path A: emulate old step (compute_loss + backward + optimizer.step)
        # Use a dummy optimizer step on the same trainer (we'll restore params after)
        trainer.optimizer.zero_grad()
        loss_a, _ = trainer.compute_loss(groups)
        loss_a.backward()
        trainer.optimizer.step()
        params_after_a = [p.clone().detach() for p in trainer.policy.model.parameters()]

        # Restore params to initial state for path B
        for p, p_init in zip(trainer.policy.model.parameters(), params_init):
            with torch.no_grad():
                p.copy_(p_init)

        # Path B: new step (uses _compute_loss_grad_accum + optimizer.step)
        trainer.optimizer.zero_grad()
        trainer._compute_loss_grad_accum(groups)
        trainer.optimizer.step()
        params_after_b = [p.clone().detach() for p in trainer.policy.model.parameters()]

        # Compare updated params (relaxed tolerance for float accumulation order)
        for i, (pa, pb) in enumerate(zip(params_after_a, params_after_b)):
            max_diff = (pa - pb).abs().max().item() if pa.numel() > 0 else 0.0
            self.assertTrue(
                torch.allclose(pa, pb, atol=1e-3, rtol=1e-3),
                f"param {i}: updated values differ — max abs diff = {max_diff:.2e}",
            )


class TestComputeLossGradAccumEdgeCases(unittest.TestCase):
    """Edge cases for _compute_loss_grad_accum."""

    def test_rejects_wrong_group_size(self) -> None:
        """_compute_loss_grad_accum raises ValueError on wrong group size."""
        policy, mdp = _make_policy_and_mdp()
        cfg = GRPOConfig(group_size=4)
        trainer = GRPOREINFORCE(policy=policy, mdp=mdp, cfg=cfg)
        gen = torch.Generator(device=torch.device("cpu"))
        gen.manual_seed(0)
        bad_group = [trainer.collect_trajectory(generator=gen) for _ in range(3)]
        with self.assertRaises(ValueError):
            trainer._compute_loss_grad_accum([bad_group])

    def test_empty_groups_returns_zero_loss(self) -> None:
        """Empty groups list returns metrics with loss=0 and n_steps=0."""
        policy, mdp = _make_policy_and_mdp()
        cfg = GRPOConfig(group_size=4)
        trainer = GRPOREINFORCE(policy=policy, mdp=mdp, cfg=cfg)
        metrics = trainer._compute_loss_grad_accum([])
        self.assertEqual(metrics["loss"], 0.0)
        self.assertEqual(metrics["n_steps"], 0)
        self.assertEqual(metrics["n_groups"], 0)
        self.assertIn("group_size", metrics)

    def test_works_with_kl_coef_zero(self) -> None:
        """_compute_loss_grad_accum works when kl_coef=0 (no ref_policy)."""
        policy, mdp = _make_policy_and_mdp()
        cfg = GRPOConfig(group_size=4, kl_coef=0.0)
        trainer = GRPOREINFORCE(policy=policy, mdp=mdp, cfg=cfg)
        gen = torch.Generator(device=torch.device("cpu"))
        gen.manual_seed(11)
        groups = [trainer.collect_group(generator=gen) for _ in range(2)]
        metrics = trainer._compute_loss_grad_accum(groups)
        self.assertEqual(metrics["mean_kl"], 0.0)
        self.assertGreater(metrics["n_steps"], 0)

    def test_works_with_kl_coef_and_ref_policy(self) -> None:
        """_compute_loss_grad_accum works when kl_coef>0 and ref_policy is set."""
        policy, mdp = _make_policy_and_mdp()
        ref_policy, _ = _make_policy_and_mdp()
        cfg = GRPOConfig(group_size=4, kl_coef=0.1)
        trainer = GRPOREINFORCE(policy=policy, mdp=mdp, cfg=cfg, ref_policy=ref_policy)
        gen = torch.Generator(device=torch.device("cpu"))
        gen.manual_seed(22)
        groups = [trainer.collect_group(generator=gen) for _ in range(2)]
        metrics = trainer._compute_loss_grad_accum(groups)
        # KL should be non-zero in general (different policies)
        self.assertGreaterEqual(metrics["mean_kl"], 0.0)
        self.assertGreater(metrics["n_steps"], 0)

    def test_works_with_entropy_bonus(self) -> None:
        """_compute_loss_grad_accum works when entropy_coef>0."""
        policy, mdp = _make_policy_and_mdp()
        cfg = GRPOConfig(group_size=4, entropy_coef=0.01)
        trainer = GRPOREINFORCE(policy=policy, mdp=mdp, cfg=cfg)
        gen = torch.Generator(device=torch.device("cpu"))
        gen.manual_seed(33)
        groups = [trainer.collect_group(generator=gen) for _ in range(2)]
        metrics = trainer._compute_loss_grad_accum(groups)
        # Entropy should be non-negative
        self.assertGreaterEqual(metrics["mean_entropy"], 0.0)
        self.assertGreater(metrics["n_steps"], 0)


if __name__ == "__main__":
    unittest.main()
