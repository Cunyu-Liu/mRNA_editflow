"""F0-X source-anchored legal Edit Flow (5'UTR substitution-only).

Primary action set: UTR5_SUB(position, target_nt).  Termination is a FIXED
budget k in {1,3,5} (no learned general STOP).  The flow starts from a real
*source* 5'UTR and applies a sequence of legal single-nucleotide substitutions
within an editable region, preserving length by construction.

Design invariants (all enforced by construction, not by reward penalty):

* **Legality = 100%**: every emitted action is a UTR5_SUB whose target
  differs from the current nucleotide at an editable position (the legal action
  enumerator never emits identity or out-of-region actions).
* **Length preservation = 100%**: substitution-only, so the sequence length is
  invariant under apply_action.
* **Budget violation = 0**: the first-order sampler stops once the fixed budget
  is exhausted; it never exceeds k steps.
* **Non-negative rate**: the rate head outputs softplus rates (non-negative).
* **Hard mask before normalization**: the legal action mask is applied to the
  logits BEFORE softmax/exp, so illegal actions get rate 0 and never receive
  probability mass.
* **Bregman/Edit Flow loss**: loss = sum_a lambda(a) - w(t) * log lambda(a*)
  (Campbell et al. CTMC-flow form), substitution-only.
* **First-order constrained sampler**: honest naming -- this is a FIRST-ORDER
  (Euler) trajectory sampler over the legal edit graph with a fixed budget, NOT
  an exact CTMC.

The module is pure NumPy/torch with no dependency on the legacy INS/DEL/STOP
paths.  FIRST-order sampler naming matches its numerical behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

NUC_ORDER = "ACGU"
NUC_TO_IDX = {ch: i for i, ch in enumerate(NUC_ORDER)}
NUC_FROM_IDX = {i: ch for i, ch in enumerate(NUC_ORDER)}


# ---------------------------------------------------------------------------
# Legal action / state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LegalAction:
    """A single legal UTR5_SUB(position, target_nt) action."""
    pos: int
    target_nt: str


@dataclass
class EditFlowState:
    """Biological-coordinate state of the flow.

    seq is the current (possibly mutated) 5'UTR; source_seq is the
    immutable source anchor; editable marks which positions may be edited;
    budget_remaining tracks the fixed edit budget.
    """
    seq: str
    editable: np.ndarray      # bool[len] True where editable
    budget_remaining: int
    source_seq: str

    @property
    def length(self) -> int:
        return len(self.seq)


def build_state(source_seq: str, editable_mask: Sequence[bool],
                budget: int) -> EditFlowState:
    """Build a legal flow state from a source 5'UTR and an editable mask."""
    editable = np.asarray(list(editable_mask), dtype=bool)
    assert len(editable) == len(source_seq), "editable mask must match seq length"
    assert budget >= 0, "budget must be non-negative"
    return EditFlowState(source_seq, editable, int(budget), source_seq)


def enumerate_legal_actions(state: EditFlowState) -> List[LegalAction]:
    """All legal UTR5_SUB actions: editable positions, target != current nt.

    Identity actions (target == current) are NOT emitted, so every enumerated
    action is a genuine edit.  Legality is 100% by construction.
    """
    actions: List[LegalAction] = []
    for pos in range(len(state.seq)):
        if not bool(state.editable[pos]):
            continue
        cur = state.seq[pos]
        for nt in NUC_ORDER:
            if nt != cur:
                actions.append(LegalAction(pos, nt))
    return actions


def apply_action(state: EditFlowState, action: LegalAction) -> EditFlowState:
    """Apply action (substitution-only): length preserved, budget spent."""
    assert state.budget_remaining > 0, "budget exhausted"
    assert 0 <= action.pos < len(state.seq), "position out of range"
    assert bool(state.editable[action.pos]), "action on non-editable position"
    assert action.target_nt != state.seq[action.pos], "identity action is not legal"
    new_seq = (state.seq[:action.pos] + action.target_nt
               + state.seq[action.pos + 1:])
    return EditFlowState(new_seq, state.editable,
                         state.budget_remaining - 1, state.source_seq)


def legal_matrix(state: EditFlowState) -> np.ndarray:
    """[L,4] boolean legal grid: editable and nt != current nucleotide."""
    L = len(state.seq)
    m = np.zeros((L, len(NUC_ORDER)), dtype=bool)
    for pos in range(L):
        if not bool(state.editable[pos]):
            continue
        cur = state.seq[pos]
        for nt in NUC_ORDER:
            if nt != cur:
                m[pos, NUC_TO_IDX[nt]] = True
    return m


def action_id(state: EditFlowState, action: LegalAction) -> int:
    """Flat action id = pos * 4 + nt_idx over the padded [L,4] grid."""
    return action.pos * len(NUC_ORDER) + NUC_TO_IDX[action.target_nt]


# ---------------------------------------------------------------------------
# Hard mask + non-negative rate + Bregman/Edit Flow loss
# ---------------------------------------------------------------------------

def apply_hard_mask(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """Hard legality mask BEFORE normalization: illegal cells -> -inf."""
    neg = torch.full_like(logits, -float('inf'))
    return torch.where(legal.bool(), logits, neg)


def nonnegative_rates(masked_logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """Non-negative rates: softplus(masked_logit) for legal cells, 0 otherwise."""
    rates = F.softplus(masked_logits)
    return torch.where(legal.bool(), rates, torch.zeros_like(rates))


def bregman_flow_loss(masked_logits: torch.Tensor, legal: torch.Tensor,
                      target: torch.Tensor, w: float = 1.0) -> torch.Tensor:
    """Bregman/Edit Flow loss (substitution-only), Campbell et al. form.

    loss = mean_batch [ sum_a lambda(a) - w * sum_a target_a * log lambda(a) ]
    with lambda(a) = exp(masked_logit_a) (non-negative, hard-masked so illegal
    actions have rate 0).  target is a one-hot over the legal action grid
    [B, L, 4]; w is the scheduler hazard weight (bounded).
    """
    lam = torch.exp(masked_logits)                      # 0 for illegal (-inf)
    lam = torch.where(legal.bool(), lam, torch.zeros_like(lam))
    mass = lam.sum(dim=(-2, -1))
    log_lam = torch.log(lam.clamp_min(1e-9))
    ce = (target * log_lam).sum(dim=(-2, -1))
    return (mass - w * ce).mean()


def policy_from_masked_logits(masked_logits: torch.Tensor,
                              legal: torch.Tensor) -> torch.Tensor:
    """Softmax policy over legal actions only (hard-masked before normalization).

    Returns a distribution over the [L,4] grid with 0 mass on illegal cells.
    """
    logits = torch.where(legal.bool(), masked_logits,
                         torch.full_like(masked_logits, -float('inf')))
    L, _ = logits.shape[-2], logits.shape[-1]
    flat = logits.reshape(*logits.shape[:-2], -1)
    flat_p = F.softmax(flat, dim=-1)
    return flat_p.reshape_as(logits)


# ---------------------------------------------------------------------------
# First-order constrained sampler (honest naming: first-order, not exact CTMC)
# ---------------------------------------------------------------------------

class FirstOrderConstrainedSampler:
    """First-order (Euler) constrained sampler over the legal edit graph.

    At each budget step we sample ONE legal UTR5_SUB from the hard-masked
    action distribution, apply it, and decrement the budget.  Fixed budget
    k; no learned STOP.  Length is preserved by construction (substitution
    only).  Naming is honest: this is a FIRST-ORDER trajectory sampler, NOT an
    exact CTMC -- it does not integrate a continuous-time rate field.
    """

    def __init__(self, policy_fn: Callable[[EditFlowState, List[LegalAction]],
                                           np.ndarray],
                 seed: int = 42):
        """policy_fn(state, actions) -> unnormalized scores over ."""
        self.policy_fn = policy_fn
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed if seed is not None else self._seed)

    def _categorical(self, actions: List[LegalAction],
                     state: EditFlowState) -> LegalAction:
        scores = np.asarray(self.policy_fn(state, actions), dtype=float)
        scores = scores - scores.max()
        probs = np.exp(scores)
        probs = probs / probs.sum()
        idx = int(self.rng.choice(len(actions), p=probs))
        return actions[idx]

    def sample(self, state: EditFlowState) -> Dict:
        """Run one fixed-budget trajectory from state."""
        cur = state
        traj: List[Dict] = []
        step = 0
        while cur.budget_remaining > 0:
            actions = enumerate_legal_actions(cur)
            if not actions:
                break
            a = self._categorical(actions, cur)
            traj.append({"step": step, "pos": a.pos, "target": a.target_nt})
            cur = apply_action(cur, a)
            step += 1
        return {
            "trajectory": traj,
            "n_steps": len(traj),
            "final_seq": cur.seq,
            "length": len(cur.seq),
            "source_length": len(state.seq),
            "budget_remaining": cur.budget_remaining,
            "existing_edits": sum(1 for i in range(len(state.seq))
                                   if state.seq[i] != state.source_seq[i]),
        }


def uniform_policy(state: EditFlowState, actions: List[LegalAction]) -> np.ndarray:
    """Uniform baseline policy (no model)."""
    return np.ones(len(actions), dtype=float)


__all__ = [
    "NUC_ORDER", "NUC_TO_IDX", "NUC_FROM_IDX",
    "LegalAction", "EditFlowState", "build_state",
    "enumerate_legal_actions", "apply_action", "legal_matrix", "action_id",
    "apply_hard_mask", "nonnegative_rates", "bregman_flow_loss",
    "policy_from_masked_logits", "FirstOrderConstrainedSampler", "uniform_policy",
]
