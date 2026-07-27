"""P0-04: reference policy must be fully frozen; resets must be reported.

Acceptance:
* reference.parameters() all have requires_grad == False.
* After grpo_update backward, reference gradients are all None.
* Reference weights never change during grpo_update.
* train_single_seed reports n_reference_resets.
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
    P3O8Policy, ReferencePolicy, AdaptiveKLController, GRPOTrainConfig,
    collect_batch, grpo_update, train_single_seed,
)

INERT_CDS = START_CODON + "GCU" * 4 + "UAA"
REWARD_CFG = RewardV3Config(context="protein_output_focused")


def make_source(sid: str, utr: str = "ACGU" * 10) -> MRNARecord:
    return MRNARecord(transcript_id=sid, five_utr=utr,
                      cds=INERT_CDS, three_utr="UGCU")


class TestReferenceFreeze:
    def test_reference_params_requires_grad_false(self):
        policy = P3O8Policy(max_utr_len=50, hidden_dim=32)
        ref = ReferencePolicy(P3O8Policy(max_utr_len=50, hidden_dim=32))
        ref.policy.load_state_dict(policy.state_dict())
        for p in ref.policy.parameters():
            assert p.requires_grad is False, "reference params must be frozen"

    def test_reference_grad_none_after_update(self):
        torch.manual_seed(0)
        policy = P3O8Policy(max_utr_len=50, hidden_dim=32)
        ref = ReferencePolicy(P3O8Policy(max_utr_len=50, hidden_dim=32))
        ref.policy.load_state_dict(policy.state_dict())
        sources = [make_source("s1", "ACGU" * 10), make_source("s2", "GCUA" * 10)]
        oracle = SyntheticDeltaOracle(seed=0, query_budget=1000)
        batch = collect_batch(sources, policy, oracle, edit_budget=1, group_size=4,
                              reward_config=REWARD_CFG, seed=0)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
        kl_ctrl = AdaptiveKLController(0.3, max_kl=100.0)
        grpo_update(policy, ref, batch, optimizer, kl_ctrl,
                    clip_epsilon=0.2, entropy_coef=0.05,
                    gradient_clip=1.0, n_policy_epochs=2)
        for p in ref.policy.parameters():
            assert p.grad is None, "reference gradients must all be None"

    def test_reference_weights_unchanged_after_update(self):
        torch.manual_seed(0)
        policy = P3O8Policy(max_utr_len=50, hidden_dim=32)
        ref = ReferencePolicy(P3O8Policy(max_utr_len=50, hidden_dim=32))
        ref.policy.load_state_dict(policy.state_dict())
        ref_before = [p.clone() for p in ref.policy.parameters()]
        sources = [make_source("s1", "ACGU" * 10), make_source("s2", "GCUA" * 10)]
        oracle = SyntheticDeltaOracle(seed=0, query_budget=1000)
        batch = collect_batch(sources, policy, oracle, edit_budget=1, group_size=4,
                              reward_config=REWARD_CFG, seed=0)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
        kl_ctrl = AdaptiveKLController(0.3, max_kl=100.0)
        grpo_update(policy, ref, batch, optimizer, kl_ctrl,
                    clip_epsilon=0.2, entropy_coef=0.05,
                    gradient_clip=1.0, n_policy_epochs=2)
        for before, after in zip(ref_before, ref.policy.parameters()):
            assert torch.equal(before, after), "reference weights must not change"

    def test_reference_not_in_optimizer(self):
        """Optimizer must only own policy parameters."""
        policy = P3O8Policy(max_utr_len=50, hidden_dim=32)
        ref = ReferencePolicy(P3O8Policy(max_utr_len=50, hidden_dim=32))
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
        opt_param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
        for p in ref.policy.parameters():
            assert id(p) not in opt_param_ids


class TestReferenceResetReporting:
    def test_train_single_seed_reports_reset_count(self):
        sources = [make_source(f"t{i}", "ACGU" * 10) for i in range(6)]

        def oracle_factory():
            return SyntheticDeltaOracle(seed=0, query_budget=1000)

        config = GRPOTrainConfig(
            n_updates=3, edit_budget=1, sources_per_batch=2, group_size=2,
            validation_interval=3, checkpoint_interval=3, seed=42,
            n_validation_trajectories=2, warmup_steps=1, n_policy_epochs=2,
        )
        result = train_single_seed(
            seed=42, train_sources=sources[:3], validation_sources=sources[3:],
            oracle_factory=oracle_factory, config=config,
            reward_config=REWARD_CFG, device="cpu",
        )
        assert "n_reference_resets" in result, "reset count must be reported"
        assert isinstance(result["n_reference_resets"], int)
        assert result["n_reference_resets"] >= 0
