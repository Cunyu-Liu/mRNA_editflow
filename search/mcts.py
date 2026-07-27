from __future__ import annotations

import math
import random
from typing import Callable

from mrna_editflow.core.mixed_resolution_state import MixedAction, MixedResolutionState, apply_action, legal_actions
from mrna_editflow.search.exact_search import SearchResult


def mcts_search(state: MixedResolutionState, score_fn: Callable[[MixedResolutionState], float], budget: int, simulations: int = 32, seed: int = 0) -> SearchResult:
    rng = random.Random(seed); best_state, best_path, best_score = state, tuple(), float(score_fn(state)); calls = 1
    for _ in range(simulations):
        current, path = state, []
        for _ in range(budget):
            acts = [a for a in legal_actions(current) if not a.is_stop()]
            if not acts: break
            action = rng.choice(acts); current = apply_action(current, action); path.append(action)
            score = float(score_fn(current)); calls += 1
            if score > best_score: best_state, best_path, best_score = current, tuple(path), score
    return SearchResult(best_state, best_path, best_score, calls)
