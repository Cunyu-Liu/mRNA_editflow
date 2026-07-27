from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from mrna_editflow.core.mixed_resolution_state import MixedAction, MixedResolutionState, apply_action, legal_actions


@dataclass
class SearchResult:
    state: MixedResolutionState
    actions: tuple[MixedAction, ...]
    score: float
    query_calls: int


def exact_search(state: MixedResolutionState, score_fn: Callable[[MixedResolutionState], float], budget: int, max_nodes: int = 4096) -> SearchResult:
    best = SearchResult(state, tuple(), float(score_fn(state)), 1)
    nodes = 1

    def visit(current: MixedResolutionState, path: tuple[MixedAction, ...]) -> None:
        nonlocal best, nodes
        if len(path) >= budget or nodes >= max_nodes:
            return
        for action in legal_actions(current):
            if action.is_stop():
                continue
            nxt = apply_action(current, action); nodes += 1
            score = float(score_fn(nxt))
            if score > best.score:
                best = SearchResult(nxt, path + (action,), score, nodes)
            visit(nxt, path + (action,))
            if nodes >= max_nodes:
                return
    visit(state, tuple())
    best.query_calls = nodes
    return best
