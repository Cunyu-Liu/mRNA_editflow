"""P0-05: strict incremental reward must telescope.

Frozen semantics:
    r_t = F(x_t) - F(x_{t-1}) - one_step_cost
    sum_t r_t = F(x_final) - F(x_0) - n_edits * one_step_cost
              = terminal_delta - total_cost

Acceptance: for arbitrary trajectories, the incremental sum equals the
terminal reward exactly (within float tolerance).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.schema import MRNARecord
from core.constants import START_CODON
from rl.p3_06_mdp import (
    EditAction, RewardV3Config, initial_state, transition,
    reward_value_fn, one_step_edit_cost, compute_terminal_reward,
)

INERT_CDS = START_CODON + "GCU" * 4 + "UAA"
CFG = RewardV3Config(context="protein_output_focused")


def make_record(utr: str = "ACGUACGU") -> MRNARecord:
    return MRNARecord(transcript_id="t", five_utr=utr,
                      cds=INERT_CDS, three_utr="UGCU")


def _gc(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / max(len(seq), 1)


class TestRewardTelescoping:
    def _run_trajectory(self, edits):
        rec = make_record()
        src_gc = _gc(rec.five_utr)

        def delta(candidate: MRNARecord) -> float:
            """Predicted improvement of candidate vs source (0 at source)."""
            return _gc(candidate.five_utr) - src_gc

        state = initial_state(rec, budget=len(edits))
        f_prev = reward_value_fn(rec, state.current_mrna,
                                 {"protein_output": delta(state.current_mrna)},
                                 {"protein_output": 0.0}, CFG)
        steps = []
        cur = state
        for a in edits:
            cur = transition(cur, a)
            f_cur = reward_value_fn(rec, cur.current_mrna,
                                    {"protein_output": delta(cur.current_mrna)},
                                    {"protein_output": 0.0}, CFG)
            r_t = f_cur - f_prev - one_step_edit_cost(CFG)
            steps.append(r_t)
            f_prev = f_cur
        return rec, cur, steps, delta

    def test_single_step_telescopes(self):
        rec, cur, steps, delta = self._run_trajectory(
            [EditAction(op="five_utr_sub", pos=0, nt="G")])
        terminal = compute_terminal_reward(
            rec, cur.current_mrna,
            {"protein_output": delta(cur.current_mrna)},
            {"protein_output": 0.0}, n_edits=1, config=CFG)
        assert abs(sum(steps) - terminal) < 1e-9

    def test_multi_step_telescopes(self):
        edits = [EditAction(op="five_utr_sub", pos=0, nt="G"),
                 EditAction(op="five_utr_sub", pos=1, nt="G"),
                 EditAction(op="five_utr_sub", pos=2, nt="U")]
        rec, cur, steps, delta = self._run_trajectory(edits)
        terminal = compute_terminal_reward(
            rec, cur.current_mrna,
            {"protein_output": delta(cur.current_mrna)},
            {"protein_output": 0.0}, n_edits=len(edits), config=CFG)
        assert abs(sum(steps) - terminal) < 1e-9

    def test_terminal_equals_delta_minus_total_cost(self):
        edits = [EditAction(op="five_utr_sub", pos=4, nt="C"),
                 EditAction(op="five_utr_sub", pos=5, nt="G")]
        rec, cur, steps, delta = self._run_trajectory(edits)
        f_final = reward_value_fn(rec, cur.current_mrna,
                                  {"protein_output": delta(cur.current_mrna)},
                                  {"protein_output": 0.0}, CFG)
        f_source = reward_value_fn(rec, rec,
                                   {"protein_output": 0.0},
                                   {"protein_output": 0.0}, CFG)
        expected = f_final - f_source - len(edits) * one_step_edit_cost(CFG)
        assert abs(sum(steps) - expected) < 1e-9

    def test_zero_edits_zero_return(self):
        rec = make_record()
        terminal = compute_terminal_reward(
            rec, rec, {"protein_output": 0.0}, {"protein_output": 0.0},
            n_edits=0, config=CFG)
        assert abs(terminal) < 1e-12
