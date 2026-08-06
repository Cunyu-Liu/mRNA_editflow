"""G0-X enumerable toy graph: exact base/target/density-ratio/guided quantities.

Implements a small, fully enumerable legal edit graph and the exact-guidance
quantities on it:

  - state space  S = { length-L sequences over ALPHABET }
  - legal edges  x -> y  iff y is a single substitution of x (hard mask)
  - base kernel  P[x,y] = 1/deg(x) for y legal (uniform base policy)
  - base terminal p1 = P^k[s, :]  (fixed budget k, budget-absorbing)
  - target       q = p1 * exp(beta R) / Z
  - density ratio w = q / p1
  - exact guided kernel P^h[x,y] = P[x,y] w(y) / (P w)[x]
  - guided terminal via the h-transform identity and via iterating P^h

All quantities are computed by exact enumeration (numpy dense), so they serve as
numerical ground truth / golden vectors for the G0-X acceptance.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np

from scripts.g0x.theory import ALPHABET


def _state_index(seq: str) -> int:
    return sum(ALPHABET.index(c) * (len(ALPHABET) ** i) for i, c in enumerate(seq))


def _seq_from_index(idx: int, L: int) -> str:
    out = []
    for i in range(L):
        out.append(ALPHABET[(idx // (len(ALPHABET) ** i)) % len(ALPHABET)])
    return "".join(out)


class ToyGraph:
    """Enumerable legal edit graph with exact-guidance quantities."""

    def __init__(self, L: int = 3, editable: Optional[List[bool]] = None,
                 seed: int = 0):
        self.L = L
        self.A = len(ALPHABET)                      # alphabet size
        self.n = self.A ** L                         # number of states
        self.states: List[str] = []
        self.state_idx = {}                          # seq -> index
        for idx in range(self.n):
            seq = _seq_from_index(idx, L) if L else ""
            self.states.append(seq)
            self.state_idx[seq] = idx

        # editable mask: which positions can be substituted (variable degree)
        if editable is None:
            editable = [True] * L
        assert len(editable) == L
        self.editable = list(editable)

        # legal adjacency + base kernel P (uniform over legal neighbors)
        self.neighbors: List[List[int]] = [[] for _ in range(self.n)]
        self.P = np.zeros((self.n, self.n), dtype=float)
        self.deg = np.zeros(self.n, dtype=int)
        for idx, seq in enumerate(self.states):
            nbrs = []
            for pos in range(L):
                if not self.editable[pos]:
                    continue
                cur = seq[pos]
                for nt in ALPHABET:
                    if nt == cur:
                        continue
                    ns = seq[:pos] + nt + seq[pos + 1:]
                    j = self.state_idx[ns]
                    nbrs.append(j)
            nbrs = sorted(set(nbrs))
            self.neighbors[idx] = nbrs
            self.deg[idx] = len(nbrs)
            if nbrs:
                self.P[idx, nbrs] = 1.0 / len(nbrs)

        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # reward / target
    # ------------------------------------------------------------------
    def reward(self, wmat: np.ndarray) -> np.ndarray:
        """R[idx] = sum_pos wmat[pos, nt].  wmat: [L, A]."""
        R = np.zeros(self.n, dtype=float)
        for idx, seq in enumerate(self.states):
            for pos, ch in enumerate(seq):
                R[idx] += wmat[pos, ALPHABET.index(ch)]
        return R

    def base_terminal(self, source: str, k: int) -> np.ndarray:
        """p1 = P^k[s, :] (budget-absorbing base terminal)."""
        s = self.state_idx[source]
        Pk = np.linalg.matrix_power(self.P, k)
        return Pk[s].copy()

    def target(self, p1: np.ndarray, R: np.ndarray, beta: float) -> np.ndarray:
        """q = p1 * exp(beta R) / Z (support subset of p1)."""
        logw = beta * R
        un = p1 * np.exp(logw - np.max(logw))
        Z = un.sum()
        assert Z > 0, "target denominator zero"
        return un / Z

    def density_ratio(self, p1: np.ndarray, q: np.ndarray) -> np.ndarray:
        """w = q/p1 (finite on supp(p1); 0 where p1==0 and q==0)."""
        w = np.zeros_like(p1)
        nz = p1 > 0
        assert np.all(q[nz] >= 0)
        w[nz] = q[nz] / p1[nz]
        return w

    # ------------------------------------------------------------------
    # exact guidance (time-inhomogeneous twisted / Doob h-transform)
    # ------------------------------------------------------------------
    def backward_ratio(self, P: np.ndarray, w: np.ndarray, k: int) -> list:
        """h_t = P^{k-t} w  (backward reward-to-go), t = 0..k.

        h_k = w; h_{t-1} = P h_t.  These are the twisted functions used to
        build the exact time-inhomogeneous guided kernels.
        """
        hs = [None] * (k + 1)
        hs[k] = w.copy()
        for t in range(k - 1, -1, -1):
            hs[t] = P @ hs[t + 1]
        return hs

    def exact_guided_kernels(self, P: np.ndarray, w: np.ndarray,
                             k: int) -> list:
        """Ptilde_t[x,y] = P[x,y] h_{t+1}(y) / h_t(x),  t = 0..k-1.

        This time-inhomogeneous twisted kernel is the EXACT guidance: its k-step
        terminal equals q = p1*w/Z even when w is not an eigenfunction of P.
        """
        hs = self.backward_ratio(P, w, k)
        kernels = []
        for t in range(k):
            ker = np.zeros_like(P)
            ht, ht1 = hs[t], hs[t + 1]
            for x in range(self.n):
                if ht[x] == 0:
                    continue
                ker[x] = P[x] * ht1 / ht[x]
            kernels.append(ker)
        return kernels

    def exhaust_guided_terminal(self, kernels: list, source: str,
                                k: int) -> np.ndarray:
        """Iterate the (time-inhomogeneous) guided kernels k steps."""
        s = self.state_idx[source]
        vec = np.zeros(self.n)
        vec[s] = 1.0
        for t in range(k):
            vec = vec @ kernels[t]
        return vec

    def htransform_identity_terminal(self, p1: np.ndarray,
                                     w: np.ndarray) -> np.ndarray:
        """Closed-form exact-guidance terminal: q = p1 * w / (p1 . w)."""
        Z = float(p1 @ w)
        assert Z > 0
        return p1 * w / Z

    # ------------------------------------------------------------------
    # support / complexity
    # ------------------------------------------------------------------
    def support_violations(self, P: np.ndarray, Ph: np.ndarray) -> int:
        """Count (x,y) where Ph[x,y]>0 but P[x,y]==0 (new support)."""
        return int(np.sum((Ph > 0) & (P == 0)))

    def count_edges(self) -> int:
        return int(self.P.sum() > 0 and self.P.astype(bool).sum())


def default_reward_matrix(L: int = 3, seed: int = 1) -> np.ndarray:
    """A fixed, reproducible position-specific reward matrix [L,4]."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(L, len(ALPHABET)))


def build_standard_toy(L: int = 3, source: str = "AAA", k: int = 2,
                       beta: float = 1.0,
                       editable: Optional[List[bool]] = None) -> Tuple[
                           ToyGraph, np.ndarray, np.ndarray, np.ndarray,
                           np.ndarray, np.ndarray, np.ndarray]:
    """Convenience: build graph + all exact-guidance quantities.

    Returns (graph, p1, q, w, kernels, guided_exact, guided_iter) where
    kernels = the k time-inhomogeneous twisted kernels (h_t = P^{k-t} w),
    guided_exact = p1*w/E_s[w] (closed-form identity) and
    guided_iter = iterate(kernels, k) starting from source.
    """
    g = ToyGraph(L=L, editable=editable)
    R = g.reward(default_reward_matrix(L))          # per-state reward vector
    p1 = g.base_terminal(source, k)
    q = g.target(p1, R, beta)
    w = g.density_ratio(p1, q)
    kernels = g.exact_guided_kernels(g.P, w, k)     # time-inhomogeneous guided kernels
    guided_exact = g.htransform_identity_terminal(p1, w)
    guided_iter = g.exhaust_guided_terminal(kernels, source, k)
    return g, p1, q, w, kernels, guided_exact, guided_iter