"""Tests for F0-X source-anchored legal Edit Flow.

Covers the acceptance invariants: legality=100%, length preservation=100%,
budget violation=0, reproducible fixed-seed trajectory, hard-mask-before-
normalization, non-negative rates, the Bregman/Edit Flow loss, and the
toy/base distribution test.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from scripts.f0x.flow import (
    NUC_ORDER,
    EditFlowState,
    FirstOrderConstrainedSampler,
    LegalAction,
    apply_action,
    apply_hard_mask,
    bregman_flow_loss,
    build_state,
    enumerate_legal_actions,
    legal_matrix,
    nonnegative_rates,
    policy_from_masked_logits,
    uniform_policy,
)
from scripts.f0x.base import FlowRateNet


def _state(seq="ACGUAC", budget=3, editable=None):
    if editable is None:
        editable = [True] * len(seq)
    return build_state(seq, editable, budget)


def _one_hot(seq: str) -> torch.Tensor:
    arr = np.zeros((1, 100, 4), dtype=np.float32)
    for i, ch in enumerate(seq):
        if ch in NUC_ORDER:
            arr[0, i, NUC_ORDER.index(ch)] = 1.0
    return torch.tensor(arr)


class _FlatRate(nn.Module):
    """Tiny trainable rate model for the Bregman-loss toy test (one scalar per
    position,nt)."""
    def __init__(self, L=6):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(1, L, 4))
        self.legal = torch.ones(1, L, 4, dtype=torch.bool)


# ---------------------------------------------------------------------------
# legality / length / budget / reproducibility
# ---------------------------------------------------------------------------

def test_enumerator_no_identity_and_all_legal():
    st = _state("ACGUAC", budget=3)
    acts = enumerate_legal_actions(st)
    # each action: editable pos, target != current
    for a in acts:
        assert st.editable[a.pos]
        assert a.target_nt != st.seq[a.pos]
    # identity never emitted
    for pos in range(len(st.seq)):
        assert not any(a.pos == pos and a.target_nt == st.seq[pos] for a in acts)
    # count: for each editable pos, 3 targets (4 nt - identity)
    assert len(acts) == 6 * 3


def test_apply_action_preserves_length_and_spends_budget():
    st = _state("ACGUAC", budget=3)
    a = LegalAction(0, "G")
    nxt = apply_action(st, a)
    assert len(nxt.seq) == len(st.seq)          # length preserved
    assert nxt.budget_remaining == st.budget_remaining - 1
    assert nxt.seq[0] == "G"
    # source anchor unchanged
    assert nxt.source_seq == "ACGUAC"


def test_budget_never_negative():
    st = _state("ACGUAC", budget=1)
    a = LegalAction(1, "U")
    nxt = apply_action(st, a)
    assert nxt.budget_remaining == 0
    with pytest.raises(AssertionError):
        apply_action(nxt, LegalAction(2, "U"))


def test_identity_action_rejected():
    st = _state("ACGUAC", budget=2)
    with pytest.raises(AssertionError):
        apply_action(st, LegalAction(0, "A"))  # A already at pos 0


def test_sampler_legality_length_budget_at_k():
    for k in (1, 3, 5):
        st = _state("ACGUACGU", budget=k)
        m = FirstOrderConstrainedSampler(uniform_policy, seed=7)
        out = m.sample(st)
        assert out["n_steps"] == k                     # used full budget
        assert out["budget_remaining"] == 0             # no violation
        assert out["length"] == out["source_length"]    # length preserved
        # legality of every step: each action is a legal UTR5_SUB
        cur = st
        for t in out["trajectory"]:
            acts = enumerate_legal_actions(cur)
            legal_ids = {(a.pos, a.target_nt) for a in acts}
            assert (t["pos"], t["target"]) in legal_ids
            cur = apply_action(cur, LegalAction(t["pos"], t["target"]))


def test_sampler_reproducible_same_seed():
    st = _state("ACGUACGUAC", budget=5)
    s1 = FirstOrderConstrainedSampler(uniform_policy, seed=123)
    s2 = FirstOrderConstrainedSampler(uniform_policy, seed=123)
    o1 = s1.sample(st)
    o2 = s2.sample(st)
    assert o1["trajectory"] == o2["trajectory"]
    assert o1["final_seq"] == o2["final_seq"]


def test_sampler_different_seed_differs():
    st = _state("ACGUACGUAC", budget=5)
    s1 = FirstOrderConstrainedSampler(uniform_policy, seed=1)
    s2 = FirstOrderConstrainedSampler(uniform_policy, seed=2)
    o1 = s1.sample(st)
    o2 = s2.sample(st)
    # trajectories are not guaranteed to differ, but sequences very likely do;
    # assert at least one differs to confirm the seed is actually used
    assert (o1["trajectory"] != o2["trajectory"]) or (o1["final_seq"] != o2["final_seq"])


def test_sampler_stops_when_no_legal_actions():
    # all positions fixed -> no legal actions -> 0 steps, budget untouched semantics
    st = _state("ACGU", budget=3, editable=[False, False, False, False])
    m = FirstOrderConstrainedSampler(uniform_policy, seed=0)
    out = m.sample(st)
    assert out["n_steps"] == 0
    assert out["budget_remaining"] == 3


# ---------------------------------------------------------------------------
# hard mask before normalization / non-negative rates
# ---------------------------------------------------------------------------

def test_hard_mask_before_normalization():
    logits = torch.zeros(1, 4, 4)
    legal = torch.zeros(1, 4, 4, dtype=torch.bool)
    legal[0, 0, :] = True            # only row 0 legal
    legal[0, 0, 1] = False           # col 1 illegal
    masked = apply_hard_mask(logits, legal)
    assert torch.isinf(masked[0, 0, 1]) and masked[0, 0, 1] < 0
    assert torch.isfinite(masked[0, 0, 0])
    assert torch.isinf(masked[0, 1, 0])   # non-editable row -> -inf
    # policy puts zero mass on illegal cells
    pol = policy_from_masked_logits(masked, legal)
    flat = pol[0].reshape(-1)
    assert torch.allclose(flat.sum(), torch.tensor(1.0), atol=1e-5)
    assert pol[0, 0, 1].item() == 0.0
    assert pol[0, 1, 0].item() == 0.0


def test_nonnegative_rates_and_zero_on_illegal():
    logits = torch.randn(1, 5, 4)
    legal = torch.zeros(1, 5, 4, dtype=torch.bool)
    legal[0, 0, 0] = True
    rates = nonnegative_rates(logits, legal)
    assert (rates >= 0).all()
    # illegal cells exactly 0
    assert torch.all(rates[~legal] == 0)


# ---------------------------------------------------------------------------
# Bregman / Edit Flow loss
# ---------------------------------------------------------------------------

def test_bregman_loss_known_optimum():
    """Toy target-action test: minimise Bregman loss -> rate concentrates on the
    target action with magnitude w; non-target rates -> 0."""
    L = 6
    w = 3.0
    model = _FlatRate(L)
    target = torch.zeros(1, L, 4)
    target[0, 2, 1] = 1.0            # target action: (pos=2, nt=1)
    opt = torch.optim.Adam(model.parameters(), lr=0.3)
    losses = []
    for _ in range(1500):
        opt.zero_grad()
        loss = bregman_flow_loss(model.logits, model.legal, target, w=w)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    # loss should have decreased substantially
    assert losses[-1] < losses[0] * 0.5
    lam = torch.exp(torch.where(model.legal, model.logits,
                                torch.full_like(model.logits, -float('inf'))))
    target_rate = float(lam[0, 2, 1])
    others = float(lam[0, 2, :].sum() - target_rate)
    # target rate ~ w; non-target mass ~ 0
    assert abs(target_rate - w) < 0.5
    assert others < 0.5


# ---------------------------------------------------------------------------
# toy / base distribution test
# ---------------------------------------------------------------------------

def test_toy_base_distribution_sampler_reproduces_target():
    """Toy/base distribution test: drive a fixed policy toward a target on a
    source, sample many trajectories with the first-order sampler, and verify the
    empirical distribution over the first edit matches the target within tolerance."""
    seq = "ACGUACGU"
    L = len(seq)
    target = np.zeros((L, 4))
    # target: strong preference for editing pos 1 -> U and pos 3 -> A
    target[1, NUC_ORDER.index("U")] = 0.7
    target[3, NUC_ORDER.index("A")] = 0.3
    # only legal cells (target != current) may hold mass
    st = _state(seq, budget=1)
    leg = legal_matrix(st)
    target = target * leg
    target = target / target.sum()

    # policy_fn returns LOG-scores: the sampler softmaxes them, so returning
    # log(target) lets the softmax reproduce the target distribution exactly.
    def policy(state, actions):
        return np.array([np.log(np.maximum(target[a.pos, NUC_ORDER.index(a.target_nt)],
                                           1e-12)) for a in actions])

    m = FirstOrderConstrainedSampler(policy, seed=99)
    n = 8000
    em = np.zeros_like(target)
    for _ in range(n):
        out = m.sample(_state(seq, budget=1))
        t = out["trajectory"][0]
        em[t["pos"], NUC_ORDER.index(t["target"])] += 1.0
    em = em / em.sum()
    # assert the two preferred edits dominate and empirical approx matches target
    assert em[1, NUC_ORDER.index("U")] == pytest.approx(0.7, abs=0.03)
    assert em[3, NUC_ORDER.index("A")] == pytest.approx(0.3, abs=0.03)


# ---------------------------------------------------------------------------
# FlowRateNet forward + coupling
# ---------------------------------------------------------------------------

def _cfg():
    from types import SimpleNamespace
    return SimpleNamespace(HIDDEN_DIM=16, NHEAD=2, N_LAYERS=1, DIM_FF=32,
                           DROP=0.0, CONV_KS=3, MAX_SEQ_LEN=100)


def test_flowratenet_forward_masks_and_policy():
    cfg = _cfg()
    net = FlowRateNet(cfg)
    src = _one_hot("ACGUACGUAC")
    cur = _one_hot("ACGUACGUAC")
    budget_idx = torch.tensor([3], dtype=torch.long)
    editable = torch.zeros(1, 100, 4, dtype=torch.bool)
    for i in range(10):
        for nt in "ACGU":
            if nt != "ACGU"[i % 4]:
                editable[0, i, NUC_ORDER.index(nt)] = True
    out = net(src, cur, budget_idx, editable)
    assert out["logits"].shape == (1, 100, 4)
    assert (out["rates"] >= 0).all()
    # illegal cells (target == current) have 0 rate
    assert float(out["rates"][0, 0, NUC_ORDER.index("A")]) == 0.0  # cur A
    assert torch.allclose(out["policy"][0].sum(), torch.tensor(1.0))
    # policy zero on illegal cells
    assert float(out["policy"][0, 0, NUC_ORDER.index("A")]) == 0.0
