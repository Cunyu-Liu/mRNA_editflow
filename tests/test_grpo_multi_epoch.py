"""P0-04: GRPO must run 2–4 policy epochs per batch with fixed old log-probs.

Acceptance:
* n_policy_epochs is honored (2–4) and reported.
* Old log-prob snapshot does not change between policy epochs.
* Each epoch performs a real backward+step (params keep moving).
"""
from __future__ import annotations

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


def _setup(lr: float = 1e-3):
    torch.manual_seed(0)
    policy = P3O8Policy(max_utr_len=50, hidden_dim=32)
    ref = ReferencePolicy(P3O8Policy(max_utr_len=50, hidden_dim=32))
    ref.policy.load_state_dict(policy.state_dict())
    sources = [make_source("s1", "ACGU" * 10), make_source("s2", "GCUA" * 10)]
    oracle = SyntheticDeltaOracle(seed=0, query_budget=1000)
    batch = collect_batch(sources, policy, oracle, edit_budget=1, group_size=4,
                          reward_config=REWARD_CFG, seed=0)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
    kl_ctrl = AdaptiveKLController(0.0, max_kl=100.0)
    kl_ctrl.coefficient = 0.0
    return policy, ref, batch, optimizer, kl_ctrl


class TestMultiEpoch:
    def test_epochs_honored_and_reported(self):
        for n_epochs in (2, 3, 4):
            policy, ref, batch, optimizer, kl_ctrl = _setup()
            metrics = grpo_update(policy, ref, batch, optimizer, kl_ctrl,
                                  clip_epsilon=0.2, entropy_coef=0.0,
                                  gradient_clip=1.0, n_policy_epochs=n_epochs)
            assert metrics["n_policy_epochs"] == n_epochs, (
                f"expected {n_epochs} epochs, got {metrics['n_policy_epochs']}")

    def test_epochs_clamped_to_valid_range(self):
        policy, ref, batch, optimizer, kl_ctrl = _setup()
        metrics = grpo_update(policy, ref, batch, optimizer, kl_ctrl,
                              clip_epsilon=0.2, entropy_coef=0.0,
                              gradient_clip=1.0, n_policy_epochs=10)
        assert 1 <= metrics["n_policy_epochs"] <= 4

    def test_old_log_probs_fixed_across_epochs(self):
        """The rollout snapshot (step.log_prob) must not change between epochs."""
        policy, ref, batch, optimizer, kl_ctrl = _setup(lr=1e-2)
        before = [step.log_prob for group in batch for traj in group
                  for step in traj.steps]
        metrics = grpo_update(policy, ref, batch, optimizer, kl_ctrl,
                              clip_epsilon=0.2, entropy_coef=0.0,
                              gradient_clip=1.0, n_policy_epochs=3)
        after = [step.log_prob for group in batch for traj in group
                 for step in traj.steps]
        assert metrics["n_policy_epochs"] == 3
        assert before == after, "old log-probs must be fixed across policy epochs"

    def test_multi_epoch_moves_params_more_than_single(self):
        """3 epochs with a meaningful LR should move params at least as much
        as 1 epoch (each epoch is a real backward+step)."""
        deltas = {}
        for n_epochs in (1, 3):
            policy, ref, batch, optimizer, kl_ctrl = _setup(lr=1e-2)
            before = [p.clone() for p in policy.parameters()]
            grpo_update(policy, ref, batch, optimizer, kl_ctrl,
                        clip_epsilon=0.2, entropy_coef=0.0,
                        gradient_clip=10.0, n_policy_epochs=n_epochs)
            delta = sum(float((a - b).norm())
                        for a, b in zip(policy.parameters(), before))
            deltas[n_epochs] = delta
        assert deltas[3] > deltas[1], (
            f"3 epochs ({deltas[3]:.6f}) should move params more than "
            f"1 epoch ({deltas[1]:.6f})")

    def test_single_epoch_still_supported(self):
        policy, ref, batch, optimizer, kl_ctrl = _setup()
        metrics = grpo_update(policy, ref, batch, optimizer, kl_ctrl,
                              clip_epsilon=0.2, entropy_coef=0.0,
                              gradient_clip=1.0, n_policy_epochs=1)
        assert metrics["n_policy_epochs"] == 1
        assert metrics["n_steps"] > 0
