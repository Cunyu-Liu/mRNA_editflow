"""P0-05: cycle prevention — visited states (sha256) block returning to a
previously seen sequence.

Acceptance:
* an edit that would reproduce a visited sequence is excluded from legal actions;
* visited set grows monotonically along a trajectory;
* reversing an edit immediately is impossible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.schema import MRNARecord
from core.constants import START_CODON
from rl.p3_06_mdp import (
    EditAction, sequence_identity, initial_state, transition,
    apply_edit_action, build_legal_edit_actions,
)

INERT_CDS = START_CODON + "GCU" * 4 + "UAA"


def make_record(utr: str = "ACGUACGU") -> MRNARecord:
    return MRNARecord(transcript_id="t", five_utr=utr,
                      cds=INERT_CDS, three_utr="UGCU")


class TestCyclePrevention:
    def test_reverse_edit_is_blocked(self):
        rec = make_record()
        state = initial_state(rec, budget=5)
        fwd = EditAction(op="five_utr_sub", pos=0, nt="G")
        state2 = transition(state, fwd)
        # The reverse action returns to the source sequence (visited).
        rev = EditAction(op="five_utr_sub", pos=0, nt=rec.five_utr[0])
        legal = build_legal_edit_actions(state2.current_mrna, state2.visited_states)
        assert rev not in legal, "reverse edit must be blocked by visited set"

    def test_visited_grows_monotonically(self):
        rec = make_record()
        state = initial_state(rec, budget=5)
        seen = [len(state.visited_states)]
        cur = state
        for pos, nt in [(0, "G"), (1, "U"), (2, "A")]:
            cur = transition(cur, EditAction(op="five_utr_sub", pos=pos, nt=nt))
            seen.append(len(cur.visited_states))
        assert seen == sorted(seen)
        assert seen[-1] == seen[0] + 3

    def test_no_legal_action_revisits_any_visited_state(self):
        rec = make_record()
        state = initial_state(rec, budget=4)
        cur = transition(state, EditAction(op="five_utr_sub", pos=3, nt="A"))
        legal = build_legal_edit_actions(cur.current_mrna, cur.visited_states)
        for a in legal:
            if a.is_stop():
                continue
            nxt = apply_edit_action(cur.current_mrna, a)
            assert sequence_identity(nxt) not in cur.visited_states, (
                f"action {a} leads to a visited state")

    def test_builder_without_visited_allows_reverse(self):
        """Sanity: without visited set, the reverse action IS available."""
        rec = make_record()
        state = initial_state(rec, budget=5)
        cur = transition(state, EditAction(op="five_utr_sub", pos=0, nt="G"))
        rev = EditAction(op="five_utr_sub", pos=0, nt=rec.five_utr[0])
        legal = build_legal_edit_actions(cur.current_mrna, set())
        assert rev in legal
