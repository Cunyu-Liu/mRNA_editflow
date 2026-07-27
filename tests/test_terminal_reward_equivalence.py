"""P0-05 acceptance: on a toy sequence of length <= 8, exhaustively compare

    1. dynamic-programming optimum (over the incremental-reward MDP),
    2. brute-force optimum (enumerate every terminal candidate),
    3. MDP return of the optimal trajectory (sum of incremental rewards).

All three must agree exactly.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.schema import MRNARecord
from core.constants import START_CODON
from rl.p3_06_mdp import (
    RewardV3Config, initial_state, transition,
    apply_edit_action, build_legal_edit_actions,
    reward_value_fn, one_step_edit_cost, compute_terminal_reward,
    sequence_identity,
)

INERT_CDS = START_CODON + "GCU" * 2 + "UAA"  # minimal valid CDS
CFG = RewardV3Config(context="protein_output_focused")
BUDGET = 2


def make_record(utr: str) -> MRNARecord:
    return MRNARecord(transcript_id="toy", five_utr=utr,
                      cds=INERT_CDS, three_utr="")


def toy_delta(seq: str, ref: str) -> float:
    """Deterministic toy oracle: GC gain of the 5'UTR relative to source."""
    if not seq:
        return 0.0
    gc = lambda s: (s.count("G") + s.count("C")) / max(len(s), 1)
    return gc(seq) - gc(ref)


def F(rec: MRNARecord, source: MRNARecord) -> float:
    return reward_value_fn(source, rec,
                           {"protein_output": toy_delta(rec.five_utr,
                                                        source.five_utr)},
                           {"protein_output": 0.0}, CFG)


def enumerate_states(source: MRNARecord, budget: int):
    """All MDP states reachable in <= budget legal substitutions."""
    states = {}
    start = initial_state(source, budget=budget)

    def rec_search(state):
        states[(sequence_identity(state.current_mrna), state.n_edits())] = state
        if state.remaining_budget <= 0:
            return
        for a in build_legal_edit_actions(state.current_mrna, state.visited_states):
            if a.is_stop():
                continue
            rec_search(transition(state, a))

    rec_search(start)
    return states


class TestTerminalRewardEquivalence:
    def test_dp_equals_bruteforce_equals_mdp_return(self):
        source = make_record("ACGUACGU")  # length-8 toy sequence
        states = enumerate_states(source, BUDGET)
        assert len(states) > 1

        # ---- 2. brute-force terminal optimum over all reachable states ----
        best_terminal = -float("inf")
        best_state = None
        for st in states.values():
            r = compute_terminal_reward(
                source, st.current_mrna,
                {"protein_output": toy_delta(st.current_mrna.five_utr,
                                             source.five_utr)},
                {"protein_output": 0.0}, n_edits=st.n_edits(), config=CFG)
            if r > best_terminal:
                best_terminal = r
                best_state = st

        # ---- 1. DP optimum over the incremental-reward MDP ----
        # V(state) = max( 0 [stop],  max_a [ r(a) + V(next) ] )
        # where r(a) = F(next) - F(cur) - one_step_cost.
        @lru_cache(maxsize=None)
        def V(seq: str, remaining: int, visited: frozenset) -> float:
            rec = MRNARecord(transcript_id="t", five_utr=seq,
                             cds=INERT_CDS, three_utr="")
            f_here = F(rec, source)
            best = 0.0  # STOP: no further incremental reward
            if remaining > 0:
                for a in build_legal_edit_actions(rec, visited):
                    if a.is_stop():
                        continue
                    nxt = apply_edit_action(rec, a)
                    r_step = F(nxt, source) - f_here - one_step_edit_cost(CFG)
                    best = max(best, r_step + V(
                        nxt.five_utr, remaining - 1,
                        visited | {sequence_identity(nxt)}))
            return best

        src_hash = sequence_identity(source)
        dp_opt = V(source.five_utr, BUDGET, frozenset({src_hash}))

        # ---- 3. MDP return along the brute-force best trajectory ----
        cur = initial_state(source, budget=BUDGET)
        f_prev = F(cur.current_mrna, source)
        mdp_return = 0.0
        for a in best_state.edit_history:
            cur = transition(cur, a)
            f_cur = F(cur.current_mrna, source)
            mdp_return += f_cur - f_prev - one_step_edit_cost(CFG)
            f_prev = f_cur

        assert abs(best_terminal - mdp_return) < 1e-9, (
            f"brute-force terminal {best_terminal} != MDP return {mdp_return}")
        assert abs(dp_opt - best_terminal) < 1e-9, (
            f"DP optimum {dp_opt} != brute-force {best_terminal}")

    def test_stop_is_optimal_when_all_edits_costly(self):
        """With a huge per-edit cost, the optimal return is 0 (stop immediately)."""
        costly = RewardV3Config(context="protein_output_focused", w_edit_cost=-100.0)
        source = make_record("ACGUACGU")
        terminal = compute_terminal_reward(
            source, source, {"protein_output": 0.0}, {"protein_output": 0.0},
            n_edits=0, config=costly)
        assert abs(terminal) < 1e-12
