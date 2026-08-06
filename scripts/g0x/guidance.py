"""G0-X exact vs approximate guidance comparison (Bregman / L2 / first-order).

On a toy graph whose target is NOT exactly in the additive main-effects basis,
compares the exact density-ratio guidance against three approximate ratio heads:

  - exact        : w = q/p1  -> guided kernel P^h -> terminal = q (TV~0)
  - first-order  : linear (first-order) expansion of log w on the additive basis
  - L2           : least-squares fit of log w on the additive basis
  - Bregman(KL)  : fit q_param = p1*exp(phi theta)/Z by forward KL to q

Because the interaction reward is not additive, the approximate heads underfit
and have terminal TV > 0, while exact guidance attains TV ~ 0.  Only the exact
head may be called "exact guidance"; the others are learned/approximate rate
guidance (per the theory wording boundary).
"""
from __future__ import annotations

import numpy as np

from scripts.g0x.toy_graph import ToyGraph, _state_index


def _feature(seq: str, L: int) -> np.ndarray:
    """Additive main-effects features: [1, L*A] (constant + per-pos one-hot)."""
    A = len("ACGU")
    phi = np.zeros(1 + L * A)
    for pos, ch in enumerate(seq):
        phi[1 + pos * A + "ACGU".index(ch)] = 1.0
    return phi


def build_features(g: ToyGraph) -> np.ndarray:
    """Phi: [n, 1+L*A]."""
    return np.stack([_feature(seq, g.L) for seq in g.states])


def interaction_reward(g: ToyGraph, seed: int = 7) -> np.ndarray:
    """Additive + one pairwise interaction so log-target is NOT additive."""
    rng = np.random.default_rng(seed)
    wmat = rng.normal(0.0, 1.0, size=(g.L, g.A))
    R = np.zeros(g.n)
    for idx, seq in enumerate(g.states):
        R[idx] = sum(wmat[pos, "ACGU".index(ch)] for pos, ch in enumerate(seq))
        # pairwise interaction: bonus if pos0==pos1 == 'A' (non-additive)
        if g.L >= 2 and seq[0] == seq[1] == "A":
            R[idx] += 1.5
    return R


def exact_ratio(g: ToyGraph, p1: np.ndarray, q: np.ndarray) -> np.ndarray:
    return g.density_ratio(p1, q)


def first_order_ratio(g: ToyGraph, p1: np.ndarray, R: np.ndarray,
                      beta: float) -> np.ndarray:
    """Linear (first-order) expansion of log w on the additive basis.

    Fit log(p1*exp(beta R)) surface by its additive projection; the residual
    (interaction) is dropped, so the ratio is only first-order accurate.
    """
    phi = build_features(g)
    f = beta * R
    # least squares: f_approx = phi theta  (additive projection)
    theta, *_ = np.linalg.lstsq(phi, f, rcond=None)
    f_approx = phi @ theta
    un = p1 * np.exp(f_approx - np.max(f_approx))
    q_fo = un / un.sum()
    return g.density_ratio(p1, q_fo)


def l2_ratio(g: ToyGraph, p1: np.ndarray, R: np.ndarray,
             beta: float) -> np.ndarray:
    """L2 fit of log w = beta R on the additive basis (least squares)."""
    phi = build_features(g)
    theta, *_ = np.linalg.lstsq(phi, beta * R, rcond=None)
    logw = phi @ theta
    un = p1 * np.exp(logw - np.max(logw))
    q_l2 = un / un.sum()
    return g.density_ratio(p1, q_l2)


def bregman_ratio(g: ToyGraph, p1: np.ndarray, R: np.ndarray,
                  beta: float, iters: int = 2000, lr: float = 0.1) -> np.ndarray:
    """Forward-KL (Bregman) fit of q_param = p1 exp(phi theta)/Z to q.

    Minimizes KL(q || q_param) via gradient descent on theta.
    """
    phi = build_features(g)
    n = g.n
    q = g.target(p1, R, beta)
    theta = np.zeros(phi.shape[1])
    for _ in range(iters):
        logits = phi @ theta
        logits = logits - logits.max()
        qp = p1 * np.exp(logits)
        Z = qp.sum()
        qp = qp / Z
        grad = phi.T @ (qp - q)  # d/dtheta KL(q||qp) = phi^T (qp - q)
        theta = theta - lr * grad
    logits = phi @ theta
    logits = logits - logits.max()
    qp = p1 * np.exp(logits)
    qp = qp / qp.sum()
    return g.density_ratio(p1, qp)


def terminal_tv(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(a - b).sum())


def compare_guidance(L: int = 3, source: str = "AAA", k: int = 2,
                     beta: float = 1.0, seed: int = 7) -> dict:
    """Run exact / first-order / L2 / Bregman guidance and report terminal TV.

    Returns per-method TV vs the target q.
    """
    g = ToyGraph(L=L)
    p1 = g.base_terminal(source, k)
    R = interaction_reward(g, seed)
    q = g.target(p1, R, beta)

    methods = {
        "exact": exact_ratio(g, p1, q),
        "first_order": first_order_ratio(g, p1, R, beta),
        "l2": l2_ratio(g, p1, R, beta),
        "bregman": bregman_ratio(g, p1, R, beta),
    }
    out = {"L": L, "source": source, "k": k, "beta": beta, "methods": {}}
    for name, w in methods.items():
        kernels = g.exact_guided_kernels(g.P, w, k)
        guided = g.exhaust_guided_terminal(kernels, source, k)
        out["methods"][name] = {"terminal_tv_vs_q": terminal_tv(guided, q)}
    return out