"""G0-X unit tests: exact guidance theory + enumerable toy graph.

Verifies the math behind Phase G0-X: base/target/density-ratio, the h-transform
identity, support preservation, row-stochasticity, and that only the exact
guidance head attains terminal TV ~ 0 (first-order/L2/Bregman do not).
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.g0x.toy_graph import ToyGraph, build_standard_toy, default_reward_matrix
from scripts.g0x import guidance


def _toy():
    return build_standard_toy(L=3, source="AAA", k=2, beta=1.0)


def _tv(a, b):
    return float(0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum())


def test_state_index_bijection():
    g = ToyGraph(L=3)
    assert g.n == 4 ** 3 == 64
    for idx, seq in enumerate(g.states):
        assert g.state_idx[seq] == idx


def test_base_terminal_is_distribution():
    g, p1, q, w, Ph, ge, gi = _toy()
    assert p1.sum() == pytest.approx(1.0, abs=1e-12)
    assert (p1 >= 0).all()


def test_target_is_absolutely_continuous():
    g, p1, q, w, Ph, ge, gi = _toy()
    assert q.sum() == pytest.approx(1.0, abs=1e-12)
    # support of q subset of p1
    assert np.all(q[p1 == 0] == 0)


def test_density_ratio_finite_on_support():
    g, p1, q, w, Ph, ge, gi = _toy()
    nz = p1 > 0
    assert np.all(np.isfinite(w[nz]))
    assert np.all(w[nz] > 0)


def test_htransform_identity_exact():
    """p_k^h = p1*w/E[w] must equal q exactly (the exact-guidance theorem)."""
    g, p1, q, w, Ph, ge, gi = _toy()
    assert _tv(ge, q) < 1e-12


def test_guided_iteration_matches_identity():
    """Iterating the guided kernel P^h reproduces the closed-form terminal."""
    g, p1, q, w, Ph, ge, gi = _toy()
    rel = np.abs(gi - ge).max() / max(np.abs(ge).max(), 1e-12)
    assert rel < 1e-9


def test_guided_kernel_row_stochastic():
    g, p1, q, w, kernels, ge, gi = _toy()
    for K in kernels:
        for x in range(g.n):
            if g.deg[x] == 0:
                continue
            assert K[x, :].sum() == pytest.approx(1.0, abs=1e-12)


def test_support_violation_zero():
    g, p1, q, w, kernels, ge, gi = _toy()
    for K in kernels:
        assert g.support_violations(g.P, K) == 0


def test_target_rate_relative_error_budget():
    """guided terminal via P^h vs closed form within 1e-5 relative error."""
    g, p1, q, w, Ph, ge, gi = _toy()
    denom = np.abs(ge).max()
    rel = np.abs(gi - ge).max() / denom
    assert rel <= 1e-5


def test_variable_action_degree():
    # one non-editable position -> degree shrinks -> variable action degree
    g_full = ToyGraph(L=3)
    g_part = ToyGraph(L=3, editable=[True, False, True])
    assert g_part.deg.sum() < g_full.deg.sum()
    assert g_part.deg[0] == 2 * (4 - 1)  # positions 0 and 2, 3-nt each
    assert g_part.deg[0] != g_full.deg[0]


def test_exact_beats_approximations():
    """Exact guidance reaches TV~0; first-order/L2/Bregman do not."""
    cmp = guidance.compare_guidance(L=3, source="AAA", k=2, beta=1.0, seed=7)
    m = cmp["methods"]
    assert m["exact"]["terminal_tv_vs_q"] < 1e-9
    for name in ("first_order", "l2", "bregman"):
        assert m[name]["terminal_tv_vs_q"] > 1e-3


def test_reward_is_reproducible():
    r1 = default_reward_matrix(L=3, seed=1)
    r2 = default_reward_matrix(L=3, seed=1)
    assert np.array_equal(r1, r2)