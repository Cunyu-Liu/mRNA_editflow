"""P0-04: GRPO KL term must be a differentiable tensor with real gradient.

Acceptance:
* beta_kl = 0 vs beta_kl > 0 produce significantly different gradients.
* KL tensor backward populates policy gradients (reference grads stay None).
* Numerical (finite-difference) gradient direction agrees with autograd.
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


def _perturbed_reference(policy: P3O8Policy) -> ReferencePolicy:
    """Reference that differs from policy so KL > 0 and KL-grad != 0."""
    ref = ReferencePolicy(P3O8Policy(max_utr_len=50, hidden_dim=32))
    ref.policy.load_state_dict(policy.state_dict())
    with torch.no_grad():
        for p in ref.policy.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return ref


class TestKLGradient:
    def test_kl_tensor_requires_grad(self):
        torch.manual_seed(0)
        policy = P3O8Policy(max_utr_len=50, hidden_dim=32)
        ref = _perturbed_reference(policy)
        source = make_source("s1")
        state = initial_state(source, budget=1)
        legal = build_legal_edit_actions_task_a(source, state.visited_states)
        kl_t, ent_t = policy.kl_entropy_tensor(ref.policy, state, legal)
        assert kl_t.requires_grad, "KL must be a differentiable tensor"
        assert ent_t.requires_grad, "entropy must be a differentiable tensor"
        assert float(kl_t.detach()) > 0, "KL vs perturbed ref should be positive"

    def test_kl_backward_populates_policy_grad_only(self):
        torch.manual_seed(0)
        policy = P3O8Policy(max_utr_len=50, hidden_dim=32)
        ref = _perturbed_reference(policy)
        source = make_source("s1")
        state = initial_state(source, budget=1)
        legal = build_legal_edit_actions_task_a(source, state.visited_states)
        kl_t, _ = policy.kl_entropy_tensor(ref.policy, state, legal)
        kl_t.backward()
        policy_grads = [p.grad for p in policy.parameters()]
        assert any(g is not None and g.abs().sum() > 0 for g in policy_grads), \
            "KL backward must populate policy gradients"
        for p in ref.policy.parameters():
            assert p.grad is None, "reference gradients must be None"

    def test_beta_kl_changes_gradients(self):
        """beta_kl = 0 vs beta_kl > 0 -> significantly different parameter updates."""
        torch.manual_seed(0)
        base = P3O8Policy(max_utr_len=50, hidden_dim=32)

        deltas = {}
        for beta in (0.0, 0.5):
            torch.manual_seed(0)
            policy = P3O8Policy(max_utr_len=50, hidden_dim=32)
            policy.load_state_dict(base.state_dict())
            ref = _perturbed_reference(policy)
            batch = _make_batch(policy)
            optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
            kl_ctrl = AdaptiveKLController(beta, max_kl=100.0)  # high max_kl: never skip
            kl_ctrl.coefficient = beta  # exact beta_kl, no MIN floor for the test
            params_before = [p.clone() for p in policy.parameters()]
            metrics = grpo_update(policy, ref, batch, optimizer, kl_ctrl,
                                  clip_epsilon=0.2, entropy_coef=0.0,
                                  gradient_clip=1.0, n_policy_epochs=1)
            assert metrics["updated"], "update must not be skipped"
            delta = torch.cat([(a - b).flatten()
                               for a, b in zip(policy.parameters(), params_before)])
            deltas[beta] = delta
        diff = (deltas[0.0] - deltas[0.5]).norm()
        scale = deltas[0.0].norm() + 1e-12
        assert diff / scale > 0.01, (
            f"beta_kl must significantly change gradients "
            f"(relative diff {float(diff/scale):.6f})")

    def test_finite_difference_agrees_with_autograd(self):
        """Numerical gradient of the KL loss matches autograd direction."""
        torch.manual_seed(0)
        policy = P3O8Policy(max_utr_len=50, hidden_dim=32)
        # Strongly perturbed reference -> KL gradient large enough that the
        # float32 finite-difference estimate is not dominated by rounding.
        ref = ReferencePolicy(P3O8Policy(max_utr_len=50, hidden_dim=32))
        ref.policy.load_state_dict(policy.state_dict())
        with torch.no_grad():
            for p in ref.policy.parameters():
                p.add_(0.5 * torch.randn_like(p))
        source = make_source("s1")
        state = initial_state(source, budget=1)
        legal = build_legal_edit_actions_task_a(source, state.visited_states)

        def kl_value() -> float:
            kl_t, _ = policy.kl_entropy_tensor(ref.policy, state, legal)
            return float(kl_t.detach())

        # Autograd gradient, flattened
        policy.zero_grad()
        kl_t, _ = policy.kl_entropy_tensor(ref.policy, state, legal)
        kl_t.backward()
        params = [p for p in policy.parameters() if p.grad is not None]
        auto = torch.cat([p.grad.flatten() for p in params])
        auto_norm = auto.norm()
        assert float(auto_norm) > 1e-6, "KL gradient must be non-negligible"

        # Probe along the autograd direction itself (maximal projection ->
        # best signal-to-noise for the finite-difference comparison).
        direction = auto / auto_norm
        auto_directional = float(auto_norm)
        eps = 1e-3
        flat = torch.cat([p.detach().flatten() for p in params])
        shapes = [p.shape for p in params]
        sizes = [p.numel() for p in params]

        def set_flat(vec):
            offset = 0
            with torch.no_grad():
                for p, shp, sz in zip(params, shapes, sizes):
                    p.copy_(vec[offset:offset + sz].view(shp))
                    offset += sz

        orig = flat.clone()
        set_flat(orig + eps * direction)
        plus = kl_value()
        set_flat(orig - eps * direction)
        minus = kl_value()
        set_flat(orig)
        fd_directional = (plus - minus) / (2 * eps)
        # Direction must agree (same sign, similar magnitude)
        assert fd_directional * auto_directional > 0, (
            f"FD {fd_directional:.6f} vs autograd {auto_directional:.6f}: sign mismatch")
        assert abs(fd_directional - auto_directional) / (abs(auto_directional) + 1e-12) < 0.10, (
            f"FD {fd_directional:.6f} vs autograd {auto_directional:.6f}: >10% mismatch")
