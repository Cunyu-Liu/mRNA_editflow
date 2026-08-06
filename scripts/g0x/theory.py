"""G0-X exact density-ratio guidance — formal theory (state/action, base/target,
density ratio, absolute continuity, exact guidance theorem, wording boundary).

This module is the *math* of Phase G0-X.  It does not train anything; it states
the object definitions, the exact-guidance theorem, the assumptions under which
the theorem holds, and the exactness wording boundary.  ``toy_graph.py`` and
``run_g0x.py`` verify the theorem numerically on an enumerable graph.

NOTATION
--------
We work on a finite state space ``S`` (length-``L`` sequences over alphabet
``Sigma``).  A **legal edit graph** ``G = (S, E)`` has an edge ``x -> y`` iff
``y`` is obtainable from ``x`` by a single legal substitution (hard mask: no
identity, no length change).  ``deg(x)`` is the number of legal outgoing edges.
Variable action degree is allowed (some positions may be non-editable).

A **base kernel** ``P`` is a row-stochastic ``|S| x |S|`` matrix with support
``supp(P) = E`` (``P[x,y] > 0`` iff ``x -> y``).

For a fixed budget ``k`` (number of substitutions) and a source ``s``, define
the base terminal distribution
    p1(y) = P^k(s, y) = (P^k)[s, y].
``p1`` is the (budget-absorbing) base terminal distribution.

A **reward** ``R: S -> R`` scores each state.  The **target** terminal
distribution is, for ``beta >= 0``,
    q(y) = p1(y) * exp(beta * R(y)) / Z,   Z = sum_y p1(y) exp(beta R(y)).
``q`` is absolutely continuous w.r.t. ``p1`` (``supp(q) subset supp(p1)``).

The **terminal density ratio** is
    w(y) = q(y) / p1(y)  =  exp(beta R(y)) / Z   (defined where p1(y) > 0).

EXACT GUIDANCE THEOREM (budgeted Doob h-transform, terminal identity)
---------------------------------------------------------------------
For a fixed budget ``k`` and source ``s``, define the **backward reward-to-go**
functions ``h_t``, ``t = 0..k``, by
    h_k(y) = w(y),   h_{t-1}(x) = (P h_t)(x) = sum_z P[x, z] h_t(z),
i.e. ``h_t = P^{k-t} w``.  Define the **time-inhomogeneous twisted kernel** at
step ``t`` (``0 <= t < k``) by
    Ptilde_t[x, y] = P[x, y] * h_{t+1}(y) / h_t(x),    h_t(x) > 0.
Each ``Ptilde_t`` is row-stochastic on the reachable shell, since
    sum_y Ptilde_t[x, y] = (P h_{t+1})(x) / h_t(x) = h_t(x) / h_t(x) = 1.

Terminal identity:  starting from ``s``, the ``k``-step law of the
time-inhomogeneous guided chain telescopes to
    (Ptilde_0 ... Ptilde_{k-1})[s, y]
        = P^k[s, y] * (h_1/h_0) ... (h_k/h_{k-1})  (ratio products along the path)
        = P^k[s, y] * h_k(y) / h_0(s)
        = p1(y) * w(y) / (P^k w)[s]
        = p1(y) * w(y) / E_s[w(X_k)].                            (1)

With ``h = w`` (so ``h_k = w`` and ``h_0 = P^k w``):
    terminal(y) = p1(y) * w(y) / E_s[w(X_k)]
                = p1(y) * (q(y)/p1(y)) / sum_z p1(z) w(z)
                = q(y) / sum_z q(z)   =  q(y).                   (2)
So the time-inhomogeneous guided chain has **exactly** the target ``q`` as its
k-step terminal distribution -- exact density-ratio guidance on the legal graph,
for arbitrary non-negative ``w`` (no eigenfunction / harmonicity assumption).

REMARK (why the transform must be time-inhomogeneous).  A *single*
time-homogeneous h-transform kernel ``P^h[x,y] = P[x,y] w(y)/(P w)[x]`` applied
``k`` times does NOT generically have the target terminal; its ``k``-step
composition equals ``P^k w / (P^k w)`` only when ``w`` is an eigenfunction of
``P`` (``P w = lambda w``).  In that special case every ``Ptilde_t`` collapses to
the same ``P^h``.  The budgeted Doob transform must therefore be
time-inhomogeneous in general; ``toy_graph.py`` and ``run_g0x.py`` verify (1)-(2)
numerically on an enumerable graph.

ASSUMPTIONS / CONDITIONS FOR THE THEOREM TO HOLD
------------------------------------------------
A1 (finite enumerable graph): ``|S| < infinity``; ``G`` connected within the
   reachable budget shell; ``P`` row-stochastic with support exactly ``E``.
A2 (absolute continuity): ``supp(q) subset supp(p1)``; ``w`` finite on ``supp(p1)``.
A3 (budget absorbing state): termination is ``FIXED_BUDGET``; the process makes
   exactly ``k`` jumps then stops (no learned general STOP).  The k-step kernel
   ``P^k`` is the budget-absorbed terminal.
A4 (hard mask = base support): every legal transition is in ``E``; the hard mask
   is part of the base support (not a post-hoc truncation).  ``P[x,y]=0`` for
   ``y`` not legal.  Consequently ``P^w[x,y]=0`` wherever ``P[x,y]=0`` (no
   support violation).
A5 (one-forward-pass ratio head): the ratio ``w`` is produced by a single
   forward pass of a guidance ratio head (one evaluation of ``w`` per state);
   the guided kernel is then formed point-wise.  Complexity is ``O(1)`` head
   forward passes + ``O(|E|)`` to form ``P^h``.

EXACTNESS WORDING BOUNDARY
--------------------------
- "exact guidance" is claimed ONLY when ``w`` is the true terminal density ratio
  ``q/p1`` and the guided kernel is ``P^w``; then terminal ``= q`` exactly (2).
- If ``w`` is only approximated (first-order linearization, L2/Bregman fit),
  the result is NOT exact; it must be called **learned/approximate rate
  guidance** and the paper positioning must be lowered accordingly.
- Old multi-alignment / native / fresh / multi-QK work is NEVER relabeled as
  exact guidance; it remains a base-flow alignment robustness / estimator
  sensitivity baseline.

The numerical acceptance (see ``run_g0x.py``) verifies: toy target-rate relative
error <= 1e-5, terminal TV within pre-registered tolerance, support violation = 0,
and the one-pass complexity accounting.
"""
from __future__ import annotations

__all__ = [
    "ALPHABET",
    "beta",
    "BUDGET_ABSORBING",
    "WordingBoundary",
]


ALPHABET = "ACGU"
beta = 1.0
BUDGET_ABSORBING = True  # fixed-budget termination (no learned general STOP)


class WordingBoundary:
    """Exactness wording: which quantities may be called 'exact'."""

    exact_allowed = False  # set True only when the G0-X acceptance passes
    note = (
        "exact guidance is claimed only when w = q/p1 (true terminal density "
        "ratio) and the guided kernel is P^h; approximations are "
        "learned/approximate rate guidance."
    )