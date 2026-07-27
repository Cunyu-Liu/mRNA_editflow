from __future__ import annotations

from typing import Callable

from mrna_editflow.core.mixed_resolution_state import MixedAction, MixedResolutionState, apply_action, legal_actions
from mrna_editflow.search.exact_search import SearchResult


def greedy_search(state: MixedResolutionState, score_fn: Callable[[MixedResolutionState], float], budget: int) -> SearchResult:
    current, path, calls = state, [], 1
    best_score = float(score_fn(state))
    for _ in range(budget):
        candidates = [(apply_action(current, a), a) for a in legal_actions(current) if not a.is_stop()]
        if not candidates:
            break
        scored = [(float(score_fn(s)), s, a) for s, a in candidates]; calls += len(scored)
        score, nxt, action = max(scored, key=lambda x: x[0])
        if score <= best_score:
            break
        current, best_score = nxt, score; path.append(action)
    return SearchResult(current, tuple(path), best_score, calls)
