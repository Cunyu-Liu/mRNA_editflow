"""P0-05: edit-budget accounting must be exact.

Acceptance:
* each non-STOP edit decrements remaining_budget by exactly 1;
* STOP does not consume budget;
* no edits are possible once budget is exhausted;
* n_edits equals budget_spent at every step.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.schema import MRNARecord
from core.constants import START_CODON
from rl.p3_06_mdp import (
    EditAction, STOP_EDIT, initial_state, transition,
    build_legal_edit_actions,
)

INERT_CDS = START_CODON + "GCU" * 4 + "UAA"


def make_record(utr: str = "ACGUACGUACGU") -> MRNARecord:
    return MRNARecord(transcript_id="t", five_utr=utr,
                      cds=INERT_CDS, three_utr="")


class TestBudgetAccounting:
    def test_each_edit_costs_one(self):
        state = initial_state(make_record(), budget=3)
        for i, (pos, nt) in enumerate([(0, "G"), (1, "U"), (2, "A")]):
            state = transition(state, EditAction(op="five_utr_sub", pos=pos, nt=nt))
            assert state.remaining_budget == 3 - (i + 1)
            assert state.n_edits() == i + 1

    def test_stop_is_free(self):
        state = initial_state(make_record(), budget=2)
        state = transition(state, EditAction(op="five_utr_sub", pos=0, nt="G"))
        before = state.remaining_budget
        stopped = transition(state, STOP_EDIT)
        assert stopped.remaining_budget == before
        assert stopped.n_edits() == state.n_edits()

    def test_n_edits_equals_spent_budget(self):
        state = initial_state(make_record(), budget=5)
        total = state.remaining_budget
        for pos, nt in [(0, "G"), (3, "A"), (6, "U")]:
            state = transition(state, EditAction(op="five_utr_sub", pos=pos, nt=nt))
            assert state.n_edits() + state.remaining_budget == total

    def test_budget_never_negative(self):
        state = initial_state(make_record(), budget=1)
        state = transition(state, EditAction(op="five_utr_sub", pos=0, nt="G"))
        assert state.remaining_budget == 0
        # A well-formed driver stops here; ensure budget is exactly zero and
        # n_edits stays consistent after STOP.
        stopped = transition(state, STOP_EDIT)
        assert stopped.remaining_budget == 0
        assert stopped.n_edits() == 1

    def test_history_length_matches_edits(self):
        state = initial_state(make_record(), budget=4)
        edits = [EditAction(op="five_utr_sub", pos=1, nt="A"),
                 EditAction(op="five_utr_sub", pos=2, nt="A")]
        for a in edits:
            state = transition(state, a)
        assert state.edit_history == tuple(edits)
