from __future__ import annotations

from typing import Callable

from mrna_editflow.core.mixed_resolution_state import MixedAction, MixedResolutionState, apply_action, legal_actions
from mrna_editflow.search.exact_search import SearchResult


def beam_search(state: MixedResolutionState, score_fn: Callable[[MixedResolutionState], float], budget: int, beam_width: int = 4) -> SearchResult:
    beam = [(state, tuple(), float(score_fn(state)))]; calls = 1
    for _ in range(budget):
        expanded = []
        for current, path, _ in beam:
            for action in legal_actions(current):
                if action.is_stop():
                    continue
                nxt = apply_action(current, action); expanded.append((nxt, path + (action,), float(score_fn(nxt))))
        calls += len(expanded)
        beam = sorted(expanded, key=lambda x: x[2], reverse=True)[:beam_width]
        if not beam:
            break
    current, path, score = max(beam or [(state, tuple(), float(score_fn(state)))], key=lambda x: x[2])
    return SearchResult(current, tuple(path), score, calls)
