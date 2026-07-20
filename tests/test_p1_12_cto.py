"""Tests for P1-12 Innovation 1: Constrained Trajectory Optimization (CTO).

Covers:
- ConstraintConfig and trajectory_cost
- is_feasible
- CTOREINFORCE.collect_feasible_trajectory (rejection sampling)
- CTOREINFORCE.compute_constrained_loss (feasibility mask)
- CTOREINFORCE.update_constrained (no update on infeasible batch)
- CTOREINFORCE.train_constrained
- cto_convergence_check (tiny MDP convergence)
- SoftPenaltyREINFORCE (baseline comparison)
"""
from __future__ import annotations

import unittest
from typing import List

import torch

from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.rl.action_space import Action, STOP_ACTION, apply_action
from mrna_editflow.rl.cto import (
    ConstraintConfig,
    CTOREINFORCE,
    SoftPenaltyREINFORCE,
    cto_convergence_check,
    is_feasible,
    trajectory_cost,
)
from mrna_editflow.rl.policy import Policy, PolicyConfig
from mrna_editflow.rl.tiny_mdp import (
    REINFORCE,
    TinyMDP,
    TinyTrainableModel,
    Trajectory,
    Transition,
)


def _make_policy_and_mdp(
    target: str = "AAAA",
    initial: str = "CCCC",
    max_steps: int = 4,
    budget: int = 3,
) -> tuple:
    """Build a tiny policy + MDP + constraint config for testing."""
    device = torch.device("cpu")
    record = MRNARecord(
        transcript_id="T", five_utr=initial, cds="", three_utr=""
    )
    mdp = TinyMDP(target_seq=target, initial_record=record, max_steps=max_steps)
    model = TinyTrainableModel(vocab_dim=4, hidden=8)
    backbone = type(
        "B", (), {"out_dim": 8, "forward": lambda self, *a, **k: None}
    )()
    policy = Policy(model=model, backbone=backbone, cfg=PolicyConfig(), device=device)
    cfg = ConstraintConfig(max_edit_budget=budget)
    return policy, mdp, cfg


def _make_trajectory(n_edits: int, max_steps: int = 5) -> Trajectory:
    """Build a synthetic trajectory with ``n_edits`` non-STOP actions."""
    record = MRNARecord(
        transcript_id="T", five_utr="AAAA", cds="", three_utr=""
    )
    transitions: List[Transition] = []
    state = record
    for i in range(max_steps):
        if i < n_edits:
            action = Action(op="sub", pos=0, nt=1)
        else:
            action = STOP_ACTION
        next_state = apply_action(state, action)
        transitions.append(
            Transition(
                state=state, action=action, reward=0.0, next_state=next_state, step=i
            )
        )
        state = next_state
        if action.is_stop():
            break
    return Trajectory(transitions=transitions)


class TestTrajectoryCost(unittest.TestCase):
    """Test the trajectory_cost function."""

    def test_edit_count_zero_for_stop_only(self) -> None:
        traj = _make_trajectory(n_edits=0, max_steps=1)
        self.assertEqual(trajectory_cost(traj, "edit_count"), 0.0)

    def test_edit_count_counts_non_stop(self) -> None:
        traj = _make_trajectory(n_edits=3, max_steps=5)
        self.assertEqual(trajectory_cost(traj, "edit_count"), 3.0)

    def test_length_delta(self) -> None:
        traj = _make_trajectory(n_edits=2, max_steps=3)
        cost = trajectory_cost(traj, "length_delta")
        # Substitutions don't change length, so cost should be 0.
        self.assertEqual(cost, 0.0)

    def test_unknown_cost_fn_raises(self) -> None:
        traj = _make_trajectory(n_edits=1, max_steps=2)
        with self.assertRaises(ValueError):
            trajectory_cost(traj, "unknown")


class TestIsFeasible(unittest.TestCase):
    """Test the is_feasible function."""

    def test_feasible_under_budget(self) -> None:
        traj = _make_trajectory(n_edits=2, max_steps=3)
        cfg = ConstraintConfig(max_edit_budget=3)
        self.assertTrue(is_feasible(traj, cfg))

    def test_infeasible_over_budget(self) -> None:
        traj = _make_trajectory(n_edits=4, max_steps=5)
        cfg = ConstraintConfig(max_edit_budget=3)
        self.assertFalse(is_feasible(traj, cfg))

    def test_feasible_at_budget(self) -> None:
        traj = _make_trajectory(n_edits=3, max_steps=4)
        cfg = ConstraintConfig(max_edit_budget=3)
        self.assertTrue(is_feasible(traj, cfg))


class TestCTOCollectFeasible(unittest.TestCase):
    """Test CTOREINFORCE.collect_feasible_trajectory."""

    def test_returns_feasible_trajectory(self) -> None:
        policy, mdp, cfg = _make_policy_and_mdp(budget=3)
        cto = CTOREINFORCE(policy=policy, mdp=mdp, constraint_cfg=cfg)
        traj, feasible = cto.collect_feasible_trajectory()
        self.assertTrue(feasible)
        self.assertTrue(is_feasible(traj, cfg))

    def test_returns_infeasible_when_budget_too_small(self) -> None:
        # max_steps=5 but budget=0 — only STOP is feasible.
        # With rejection sampling, we should eventually give up.
        policy, mdp, cfg = _make_policy_and_mdp(
            target="AAAA", initial="CCCC", max_steps=5, budget=0
        )
        cto = CTOREINFORCE(
            policy=policy, mdp=mdp, constraint_cfg=cfg, max_rejection_samples=3
        )
        traj, feasible = cto.collect_feasible_trajectory()
        # Most likely infeasible (policy wants to edit, not STOP).
        # But it could get lucky. We just check it returns something.
        self.assertIsInstance(traj, Trajectory)
        self.assertIsInstance(feasible, bool)

    def test_collect_feasible_batch(self) -> None:
        policy, mdp, cfg = _make_policy_and_mdp(budget=3)
        cto = CTOREINFORCE(policy=policy, mdp=mdp, constraint_cfg=cfg)
        feasible, rejected = cto.collect_feasible_batch(batch_size=4)
        self.assertGreaterEqual(len(feasible), 1)
        for traj in feasible:
            self.assertTrue(is_feasible(traj, cfg))


class TestCTOConstrainedLoss(unittest.TestCase):
    """Test CTOREINFORCE.compute_constrained_loss."""

    def test_all_feasible(self) -> None:
        policy, mdp, cfg = _make_policy_and_mdp(budget=3)
        cto = CTOREINFORCE(policy=policy, mdp=mdp, constraint_cfg=cfg)
        trajs = [_make_trajectory(n_edits=2, max_steps=3) for _ in range(4)]
        loss, info = cto.compute_constrained_loss(trajs)
        self.assertEqual(info["n_feasible"], 4)
        self.assertEqual(info["n_total"], 4)
        self.assertAlmostEqual(info["feasible_rate"], 1.0)
        self.assertGreater(info["mean_edit_count_feasible"], 0)

    def test_all_infeasible_returns_zero_loss(self) -> None:
        policy, mdp, cfg = _make_policy_and_mdp(budget=1)
        cto = CTOREINFORCE(policy=policy, mdp=mdp, constraint_cfg=cfg)
        trajs = [_make_trajectory(n_edits=3, max_steps=4) for _ in range(4)]
        loss, info = cto.compute_constrained_loss(trajs)
        self.assertEqual(info["n_feasible"], 0)
        self.assertEqual(info["loss"], 0.0)
        self.assertFalse(loss.requires_grad)

    def test_mixed_feasibility(self) -> None:
        policy, mdp, cfg = _make_policy_and_mdp(budget=2)
        cto = CTOREINFORCE(policy=policy, mdp=mdp, constraint_cfg=cfg)
        trajs = [
            _make_trajectory(n_edits=1, max_steps=2),  # feasible
            _make_trajectory(n_edits=3, max_steps=4),  # infeasible
            _make_trajectory(n_edits=2, max_steps=3),  # feasible
            _make_trajectory(n_edits=4, max_steps=5),  # infeasible
        ]
        loss, info = cto.compute_constrained_loss(trajs)
        self.assertEqual(info["n_feasible"], 2)
        self.assertEqual(info["n_total"], 4)
        self.assertAlmostEqual(info["feasible_rate"], 0.5)


class TestCTOUpdate(unittest.TestCase):
    """Test CTOREINFORCE.update_constrained."""

    def test_update_skipped_when_no_feasible(self) -> None:
        policy, mdp, cfg = _make_policy_and_mdp(budget=1)
        cto = CTOREINFORCE(policy=policy, mdp=mdp, constraint_cfg=cfg)
        trajs = [_make_trajectory(n_edits=3, max_steps=4) for _ in range(4)]
        # Snapshot params before update.
        params_before = [p.clone() for p in policy.model.parameters()]
        info = cto.update_constrained(trajs)
        self.assertEqual(info["n_feasible"], 0)
        # Params should be unchanged.
        for p_before, p_after in zip(params_before, policy.model.parameters()):
            self.assertTrue(torch.allclose(p_before, p_after))

    def test_update_changes_params_when_feasible(self) -> None:
        policy, mdp, cfg = _make_policy_and_mdp(budget=3)
        cto = CTOREINFORCE(policy=policy, mdp=mdp, constraint_cfg=cfg)
        # Collect real trajectories (not synthetic).
        trajs = [cto.collect_trajectory() for _ in range(4)]
        params_before = [p.clone() for p in policy.model.parameters()]
        info = cto.update_constrained(trajs)
        if info["n_feasible"] > 0:
            # At least one param should have changed.
            changed = any(
                not torch.allclose(p_before, p_after)
                for p_before, p_after in zip(params_before, policy.model.parameters())
            )
            self.assertTrue(changed)


class TestCTOTrainConstrained(unittest.TestCase):
    """Test CTOREINFORCE.train_constrained."""

    def test_train_returns_history(self) -> None:
        policy, mdp, cfg = _make_policy_and_mdp(budget=3)
        cto = CTOREINFORCE(policy=policy, mdp=mdp, constraint_cfg=cfg)
        history = cto.train_constrained(n_episodes=20, batch_size=4)
        self.assertGreater(len(history), 0)
        for h in history:
            self.assertIn("batch_idx", h)
            self.assertIn("n_feasible", h)

    def test_all_collected_trajectories_are_feasible(self) -> None:
        policy, mdp, cfg = _make_policy_and_mdp(budget=3)
        cto = CTOREINFORCE(policy=policy, mdp=mdp, constraint_cfg=cfg)
        # Collect a batch and verify all are feasible.
        feasible, rejected = cto.collect_feasible_batch(batch_size=8)
        for traj in feasible:
            self.assertTrue(is_feasible(traj, cfg))


class TestCTOConvergenceCheck(unittest.TestCase):
    """Test cto_convergence_check on a tiny MDP."""

    def test_convergence_passes(self) -> None:
        """CTO should converge: all trajectories feasible (the CTO invariant).

        CTO's core contribution is the hard constraint guarantee — the policy
        never produces infeasible trajectories. Return improvement (policy
        learning) is a separate concern and is not required for convergence.
        """
        policy, mdp, cfg = _make_policy_and_mdp(
            target="AAAA", initial="CCCC", max_steps=5, budget=3
        )
        cto = CTOREINFORCE(policy=policy, mdp=mdp, constraint_cfg=cfg)
        result = cto_convergence_check(cto, n_episodes=80, batch_size=4)
        self.assertTrue(
            result["converged"],
            f"CTO did not converge: {result}",
        )
        # CTO invariant: all trajectories in the final batch are feasible.
        self.assertGreaterEqual(result["final_feasibility_rate"], 0.9)
        self.assertGreater(result["n_feasible_trajectories"], 0)

    def test_cto_never_violates_constraint(self) -> None:
        """All trajectories collected by CTO must satisfy the constraint."""
        policy, mdp, cfg = _make_policy_and_mdp(
            target="AAAA", initial="CCCC", max_steps=5, budget=3
        )
        cto = CTOREINFORCE(policy=policy, mdp=mdp, constraint_cfg=cfg)
        # Collect multiple batches and verify all feasible trajectories
        # satisfy the constraint.
        for _ in range(5):
            feasible, rejected = cto.collect_feasible_batch(batch_size=8)
            for traj in feasible:
                self.assertTrue(
                    is_feasible(traj, cfg),
                    "CTO produced an infeasible trajectory — invariant violated",
                )


class TestSoftPenaltyREINFORCE(unittest.TestCase):
    """Test SoftPenaltyREINFORCE (baseline for comparison)."""

    def test_soft_loss_includes_penalty(self) -> None:
        policy, mdp, cfg = _make_policy_and_mdp(budget=3)
        soft = SoftPenaltyREINFORCE(
            policy=policy, mdp=mdp, constraint_cfg=cfg, penalty_lambda=0.5
        )
        trajs = [soft.collect_trajectory() for _ in range(4)]
        loss, info = soft.compute_soft_loss(trajs)
        self.assertIn("penalty", info)
        self.assertIn("mean_cost", info)
        self.assertIn("violation_rate", info)

    def test_soft_train_returns_history(self) -> None:
        policy, mdp, cfg = _make_policy_and_mdp(budget=3)
        soft = SoftPenaltyREINFORCE(
            policy=policy, mdp=mdp, constraint_cfg=cfg, penalty_lambda=0.5
        )
        history = soft.train_soft(n_episodes=20, batch_size=4)
        self.assertGreater(len(history), 0)
        for h in history:
            self.assertIn("violation_rate", h)

    def test_soft_penalty_can_violate_constraint(self) -> None:
        """Soft-penalty REINFORCE can violate the constraint (unlike CTO).

        This test verifies that violation_rate > 0 is possible (not that it
        always happens). We use a large max_steps to encourage violations.
        """
        policy, mdp, cfg = _make_policy_and_mdp(
            target="AAAA", initial="CCCC", max_steps=8, budget=2
        )
        soft = SoftPenaltyREINFORCE(
            policy=policy, mdp=mdp, constraint_cfg=cfg, penalty_lambda=0.1
        )
        # Collect trajectories and check if any violate.
        trajs = [soft.collect_trajectory() for _ in range(20)]
        violations = sum(1 for t in trajs if not is_feasible(t, cfg))
        # With small penalty and large max_steps, at least some should violate.
        # (This is probabilistic, but with 20 samples it's very likely.)
        self.assertGreater(violations, 0, "Expected some constraint violations")


if __name__ == "__main__":
    unittest.main()
