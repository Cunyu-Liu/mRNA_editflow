"""G1-X guided flow sampler: first-order constrained sampler that records the
full guidance quantities the contract requires (base rate, guidance ratio,
guided rate, critic mean/variance, policy entropy, legality mask,
state/action/time/budget, cycle/revisit, OOD score).

It reuses the F0-X legal-action enumerator and apply_action (no rewrite of the
edit rules).  The step policy returns a dict with at least ``guided_logits``
over the legal actions; optional ``base_logits``, ``critic_mean``,
``critic_logvar``, ``ratio`` are recorded per step.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from scripts.f0x.flow import (
    EditFlowState,
    apply_action,
    enumerate_legal_actions,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    l = np.asarray(logits, float)
    l = l - l.max()
    e = np.exp(l)
    return e / e.sum()


class GuidedFlowSampler:
    """First-order guided sampler with full per-step guidance recording.

    step_policy(state, actions) -> dict with keys:
        guided_logits : unnormalized guided scores over actions (required)
        base_logits   : base-flow scores over actions (optional)
        critic_mean   : scalar critic mean for the step (optional)
        critic_logvar : scalar critic log-var for the step (optional)
        ratio         : guidance ratio per action (optional)
        ood_score     : scalar OOD/uncertainty score (optional)
    """

    def __init__(self, step_policy: Callable, seed: int = 42):
        self.step_policy = step_policy
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed if seed is not None else self._seed)

    def sample(self, state: EditFlowState) -> Dict:
        """Run one fixed-budget guided trajectory and record all quantities."""
        cur = state
        traj: List[Dict] = []
        visited = {state.seq}
        cycles = 0
        step = 0
        while cur.budget_remaining > 0:
            actions = enumerate_legal_actions(cur)
            if not actions:
                break
            info = self.step_policy(cur, actions)
            guided = _softmax(info["guided_logits"])
            base = None
            if info.get("base_logits") is not None:
                base = _softmax(info["base_logits"])
            ent = float(-np.sum(guided * np.log(guided + 1e-12)))
            idx = int(self.rng.choice(len(actions), p=guided))
            a = actions[idx]
            ratio = info.get("ratio")
            if ratio is not None and not np.isscalar(ratio):
                ratio = float(np.asarray(ratio)[idx]) if idx < len(np.asarray(ratio)) else None
            elif ratio is not None:
                ratio = float(ratio)
            nxt = apply_action(cur, a)
            if nxt.seq in visited:
                cycles += 1
            visited.add(nxt.seq)
            traj.append({
                "step": step,
                "pos": a.pos,
                "target": a.target_nt,
                "guided_prob": float(guided[idx]),
                "base_prob": float(base[idx]) if base is not None else None,
                "guided_logit": float(info["guided_logits"][idx]),
                "base_logit": float(np.asarray(info["base_logits"])[idx])
                    if info.get("base_logits") is not None else None,
                "ratio": ratio,
                "critic_mean": info.get("critic_mean"),
                "critic_logvar": info.get("critic_logvar"),
                "ood_score": info.get("ood_score"),
                "policy_entropy": ent,
                "legal": True,
                "budget_remaining": nxt.budget_remaining,
            })
            cur = nxt
            step += 1
        return {
            "trajectory": traj,
            "n_steps": len(traj),
            "final_seq": cur.seq,
            "length": len(cur.seq),
            "source_length": len(state.seq),
            "budget_remaining": cur.budget_remaining,
            "cycles": cycles,
            "distinct_states": len(visited),
            "existing_edits": sum(1 for i in range(len(state.seq))
                                  if state.seq[i] != state.source_seq[i]),
        }


def first_step_preference(step_policy: Callable, state: EditFlowState,
                          pos: int, target: str) -> Dict:
    """The guided/base first-step preference for a single (pos, target) edit.

    Used to rank MEASURED candidates (single-edit) by the flow's preference:
    the guided probability mass the flow assigns to that candidate's edit at
    step 0.  Returns guided_prob, base_prob, and the info dict.
    """
    actions = enumerate_legal_actions(state)
    info = step_policy(state, actions)
    guided = _softmax(info["guided_logits"])
    base = None
    if info.get("base_logits") is not None:
        base = _softmax(info["base_logits"])
    for i, a in enumerate(actions):
        if a.pos == pos and a.target_nt == target:
            return {
                "guided_prob": float(guided[i]),
                "base_prob": float(base[i]) if base is not None else None,
                "info": info,
            }
    return {"guided_prob": None, "base_prob": None, "info": info}


def preference_scores(step_policy: Callable, state: EditFlowState) -> Dict:
    """Full first-step guided & base probability maps over the legal actions.

    Returns {"guided": {pos: {nt: prob}}, "base": {pos: {nt: prob}} or None,
             "info": step-info dict}.  Used to rank measured candidates by the
    flow's first-step preference.
    """
    actions = enumerate_legal_actions(state)
    info = step_policy(state, actions)
    guided = _softmax(info["guided_logits"])
    base = None
    if info.get("base_logits") is not None:
        base = _softmax(info["base_logits"])
    gmap: Dict[int, Dict[str, float]] = {}
    bmap: Dict[int, Dict[str, float]] = {}
    for i, a in enumerate(actions):
        gmap.setdefault(a.pos, {})[a.target_nt] = float(guided[i])
        if base is not None:
            bmap.setdefault(a.pos, {})[a.target_nt] = float(base[i])
    return {"guided": gmap, "base": bmap if base is not None else None, "info": info}


__all__ = ["GuidedFlowSampler", "first_step_preference", "preference_scores"]