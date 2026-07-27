"""P0-04: GRPO entropy term must be a differentiable tensor with real gradient.

Acceptance:
* beta_entropy = 0 vs beta_entropy > 0 produce significantly different gradients.
* Entropy backward populates policy gradients (reference grads stay None).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.schema import MRNARecord
from core.constants import START_CODON
from rl.p3_06_mdp import RewardV3Config, initial_state
from rl.p3_07_search import SyntheticDeltaOracle
from rl.p3_08_grpo import (
    P3O8Policy, ReferencePolicy, AdaptiveKLController,
    collect_batch, grpo_update, build_legal_edit_actions_task_a,
)

INERT_CDS = START_CODON + "GCU" * 4 + "UAA"
REWARD_CFG = RewardV3Config(context="protein_output_focused")


def make_source(sid: str, utr: str = "ACGU" * 10) -> MRNARecord:
    return MRNARecord(transcript_id=sid, five_utr=utr,
                      cds=INERT_CDS, three_utr="UGCU")


def _make_batch(policy):
    sources = [make_source("s1", "ACGU" * 10), make_source("s2", "GCUA" * 10)]
    oracle = SyntheticDeltaOracle(seed=0, query_budget=1000)
    return collect_batch(sources, policy, oracle, edit_budget=1, group_size=4,
                         reward_config=REWARD_CFG, seed=0)


class TestEntropyGradient:
    def test_entropy_backward_populates_policy_grad_only(self):
        torch.manual_seed(0)
        policy = P3O8Policy(max_utr_len=50, hidden_dim=32)
        ref = ReferencePolicy(P3O8Policy(max_utr_len=50, hidden_dim=32))
        ref.policy.load_state_dict(policy.state_dict())
        source = make_source("s1")
        state = initial_state(source, budget=1)
        legal = build_legal_edit_actions_task_a(source, state.visited_states)
        _, ent_t = policy.kl_entropy_tensor(ref.policy, state, legal)
        assert float(ent_t.detach()) > 0, "entropy of a stochastic policy must be positive"
        ent_t.backward()
        policy_grads = [p.grad for p in policy.parameters()]
        assert any(g is not None and g.abs().sum() > 0 for g in policy_grads), \
            "entropy backward must populate policy gradients"
        for p in ref.policy.parameters():
            assert p.grad is None, "reference gradients must be None"

    def test_beta_entropy_changes_gradients(self):
        """beta_entropy = 0 vs > 0 -> significantly different parameter updates."""
        torch.manual_seed(0)
        base = P3O8Policy(max_utr_len=50, hidden_dim=32)

        deltas = {}
        for beta_ent in (0.0, 0.2):
            torch.manual_seed(0)
            policy = P3O8Policy(max_utr_len=50, hidden_dim=32)
            policy.load_state_dict(base.state_dict())
            ref = ReferencePolicy(P3O8Policy(max_utr_len=50, hidden_dim=32))
            ref.policy.load_state_dict(policy.state_dict())
            batch = _make_batch(policy)
            optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
            kl_ctrl = AdaptiveKLController(0.0, max_kl=100.0)
            kl_ctrl.coefficient = 0.0  # disable KL term to isolate entropy
            params_before = [p.clone() for p in policy.parameters()]
            metrics = grpo_update(policy, ref, batch, optimizer, kl_ctrl,
                                  clip_epsilon=0.2, entropy_coef=beta_ent,
                                  gradient_clip=1.0, n_policy_epochs=1)
            assert metrics["updated"], "update must not be skipped"
            delta = torch.cat([(a - b).flatten()
                               for a, b in zip(policy.parameters(), params_before)])
            deltas[beta_ent] = delta
        diff = (deltas[0.0] - deltas[0.2]).norm()
        scale = deltas[0.0].norm() + 1e-12
        assert diff / scale > 0.01, (
            f"beta_entropy must significantly change gradients "
            f"(relative diff {float(diff/scale):.6f})")
