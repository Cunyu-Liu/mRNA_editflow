"""State tracking and STOP-aware selection for constrained mRNA decoding."""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

from mrna_editflow.rl.action_scoring import softmax_from_log_scores


def sequence_hash(sequence: str) -> str:
    """Stable hash for complete-sequence cycle detection."""
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DecoderAction:
    op: str
    pos: Optional[int]
    nt: Optional[str]
    log_score: float
    next_sequence_hash: Optional[str] = None
    old_nt: Optional[str] = None


@dataclass
class DecoderState:
    initial_sequence: str
    max_edit_budget: int
    visited_sequence_hashes: set[str] = field(default_factory=set)
    edited_action_history: list[DecoderAction] = field(default_factory=list)
    cycle_rejections: int = 0
    terminated_by_stop: bool = False
    out_of_training_action_space: bool = False

    def __post_init__(self) -> None:
        self.max_edit_budget = max(0, int(self.max_edit_budget))
        self.visited_sequence_hashes.add(sequence_hash(self.initial_sequence))

    @property
    def applied_edit_count(self) -> int:
        return len(self.edited_action_history)

    def is_immediate_reverse(self, action: DecoderAction) -> bool:
        """Reject a direct substitution reversal such as ``A -> G -> A``.

        Insert/delete inverses have coordinate shifts and are additionally
        protected by complete-sequence hashes.  The explicit substitution rule
        covers the common high-score oscillation without weakening constraints.
        """
        if not self.edited_action_history:
            return False
        previous = self.edited_action_history[-1]
        return bool(
            previous.op == "sub"
            and action.op == "sub"
            and previous.pos == action.pos
            and previous.old_nt is not None
            and action.nt == previous.old_nt
        )

    def filter_legal(self, actions: Sequence[DecoderAction]) -> list[DecoderAction]:
        legal: list[DecoderAction] = []
        for action in actions:
            if action.next_sequence_hash and action.next_sequence_hash in self.visited_sequence_hashes:
                self.cycle_rejections += 1
                continue
            if self.is_immediate_reverse(action):
                self.cycle_rejections += 1
                continue
            legal.append(action)
        return legal

    def accept(self, action: DecoderAction) -> None:
        if action.op == "stop":
            self.terminated_by_stop = True
            return
        if action.next_sequence_hash:
            self.visited_sequence_hashes.add(action.next_sequence_hash)
        self.edited_action_history.append(action)

    def to_metadata(self) -> dict[str, object]:
        return {
            "terminated_by_stop": bool(self.terminated_by_stop),
            "applied_edit_count": int(self.applied_edit_count),
            "max_edit_budget": int(self.max_edit_budget),
            "cycle_rejections": int(self.cycle_rejections),
            "visited_sequence_hashes": sorted(self.visited_sequence_hashes),
            "edited_action_history": [asdict(action) for action in self.edited_action_history],
            "out_of_training_action_space": bool(self.out_of_training_action_space),
        }


def choose_stop_aware_action(
    actions: Sequence[DecoderAction],
    state: DecoderState,
    rng: random.Random,
    *,
    top_k: int,
    temperature: float,
    allow_stop: bool = True,
    stop_logit_bias: float = 0.0,
    min_action_margin: float = 0.0,
) -> Optional[DecoderAction]:
    """Select from legal actions using log-score softmax and an explicit STOP.

    A STOP is selected deterministically when no legal edit reaches
    ``stop_logit_bias + min_action_margin``.  Otherwise STOP remains in the
    stochastic candidate set and does not consume an edit budget.
    """
    legal = state.filter_legal(actions)
    if not legal:
        if allow_stop:
            state.accept(DecoderAction("stop", None, None, float(stop_logit_bias)))
        return None
    ranked = sorted(legal, key=lambda item: (-float(item.log_score), item.op, item.pos or -1, item.nt or ""))
    stop_score = float(stop_logit_bias)
    if allow_stop and float(ranked[0].log_score) < stop_score + float(min_action_margin):
        state.accept(DecoderAction("stop", None, None, stop_score))
        return None
    k = len(ranked) if int(top_k) <= 0 else max(1, min(int(top_k), len(ranked)))
    retained = ranked[:k]
    if float(temperature) <= 0.0:
        chosen = retained[0]
        state.accept(chosen)
        return chosen
    choices = list(retained)
    if allow_stop:
        choices.append(DecoderAction("stop", None, None, stop_score))
    probs = softmax_from_log_scores([item.log_score for item in choices], float(temperature))
    threshold = rng.random()
    cumulative = 0.0
    chosen = choices[-1]
    for action, probability in zip(choices, probs.tolist()):
        cumulative += float(probability)
        if threshold <= cumulative:
            chosen = action
            break
    state.accept(chosen)
    return None if chosen.op == "stop" else chosen


__all__ = ["DecoderAction", "DecoderState", "choose_stop_aware_action", "sequence_hash"]
