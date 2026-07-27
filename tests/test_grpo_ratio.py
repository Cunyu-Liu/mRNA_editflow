"""P0-04: GRPO ratio, clipping, and clip-fraction semantics.

Acceptance:
* ratio = exp(new_log_prob - old_log_prob) with old fixed from rollout.
* First epoch ratio ≈ 1 when policy == rollout policy -> clip_fraction ≈ 0.
* Clipped surrogate = min(ratio * A, clip(ratio) * A) (PPO form).
* clip_fraction has actual meaning: grows when the policy moves away from
  the rollout snapshot.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.schema import MRNARecord
from core.constants import START_CODON
from rl.p3_06_mdp import RewardV3Config
from rl.p3_07_search import SyntheticDeltaOracle
from rl.p3_08_grpo import (
    P3O8Policy, ReferencePolicy, AdaptiveKLController,
    collect_batch, grpo_update,
)

INERT_CDS = START_CODON + "GCU" * 4 + "UAA"
REWARD_CFG = RewardV3Config(context="protein_output_focused")


def make_source(sid: str, utr: str = "ACGU" * 10) -> MRNARecord:
    return MRNARecord(transcript_id=sid, five_utr=utr,
                      cds=INERT_CDS, three_utr="UGCU")


class TestRatioSemantics:
    def test_surrogate_clipping_formula(self):
        """PPO min-clipping: min(ratio*A, clip(ratio,1-eps,1+eps)*A)."""
        ratio = torch.tensor([0.9, 1.0, 1.3, 0.7])
        adv = torch.tensor([1.0, -1.0, 1.0, -1.0])
        eps = 0.2
        clipped = ratio.clamp(1 - eps, 1 + eps)
        surrogate = torch.minimum(ratio * adv, clipped * adv)
        # ratio=1.3, A=+1 -> clipped to 1.2; ratio=0.7, A=-1 -> clipped at 0.8*-1=-0.8 vs 0.7*-1=-0.7 -> min is -0.8
        expected = torch.tensor([0.9, -1.0, 1.2, -0.8])
        assert torch.allclose(surrogate, expected), surrogate

    def test_fresh_policy_ratio_near_one(self):
        """When policy == rollout policy, first-epoch ratio ≈ 1 -> clip_fraction ≈ 0."""
        torch.manual_seed(0)
        policy = P3O8Policy(max_utr_len=50, hidden_dim=32)
        ref = ReferencePolicy(P3O8Policy(max_utr_len=50, hidden_dim=32))
        ref.policy.load_state_dict(policy.state_dict())
        sources = [make_source("s1", "ACGU" * 10), make_source("s2", "GCUA" * 10)]
        oracle = SyntheticDeltaOracle(seed=0, query_budget=1000)
        batch = collect_batch(sources, policy, oracle, edit_budget=1, group_size=4,
                              reward_config=REWARD_CFG, seed=0)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=0.0)  # no movement
        kl_ctrl = AdaptiveKLController(0.0, max_kl=100.0)
        kl_ctrl.coefficient = 0.0
        metrics = grpo_update(policy, ref, batch, optimizer, kl_ctrl,
                              clip_epsilon=0.2, entropy_coef=0.0,
                              gradient_clip=1.0, n_policy_epochs=1)
        assert metrics["clip_fraction"] < 0.05, (
            f"fresh-policy clip_fraction should be ~0, got {metrics['clip_fraction']}")

    def test_clip_fraction_grows_after_policy_moves(self):
        """After large updates, ratio deviates from 1 -> clip_fraction > 0."""
        torch.manual_seed(0)
        policy = P3O8Policy(max_utr_len=50, hidden_dim=32)
        ref = ReferencePolicy(P3O8Policy(max_utr_len=50, hidden_dim=32))
        ref.policy.load_state_dict(policy.state_dict())
        sources = [make_source("s1", "ACGU" * 10), make_source("s2", "GCUA" * 10)]
        oracle = SyntheticDeltaOracle(seed=0, query_budget=1000)
        batch = collect_batch(sources, policy, oracle, edit_budget=1, group_size=4,
                              reward_config=REWARD_CFG, seed=0)
        # Large LR: later epochs see a policy far from the rollout snapshot
        optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-2)
        kl_ctrl = AdaptiveKLController(0.0, max_kl=100.0)
        kl_ctrl.coefficient = 0.0
        metrics = grpo_update(policy, ref, batch, optimizer, kl_ctrl,
                              clip_epsilon=0.2, entropy_coef=0.0,
                              gradient_clip=10.0, n_policy_epochs=3)
        assert metrics["n_policy_epochs"] == 3
        assert metrics["clip_fraction"] > 0.0, (
            "clip_fraction must become positive once the policy moves "
            "away from the fixed rollout snapshot")

    def test_old_log_probs_are_rollout_snapshot(self):
        """old log-probs used in the ratio must equal rollout step.log_prob."""
        torch.manual_seed(0)
        policy = P3O8Policy(max_utr_len=50, hidden_dim=32)
        sources = [make_source("s1", "ACGU" * 10)]
        oracle = SyntheticDeltaOracle(seed=0, query_budget=1000)
        batch = collect_batch(sources, policy, oracle, edit_budget=1, group_size=4,
                              reward_config=REWARD_CFG, seed=0)
        rollout_lps = [step.log_prob for group in batch for traj in group
                       for step in traj.steps]
        assert all(math.isfinite(lp) and lp <= 0 for lp in rollout_lps)
        # Snapshot must not be recomputed: mutate policy, rollout log-probs stay
        with torch.no_grad():
            for p in policy.parameters():
                p.add_(0.1)
        rollout_lps_after = [step.log_prob for group in batch for traj in group
                             for step in traj.steps]
        assert rollout_lps == rollout_lps_after
