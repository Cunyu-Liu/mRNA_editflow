"""P0-05: MDP state identity must be sha256(full_current_sequence) only.

Acceptance:
* identity is sha256 hex digest of the full current sequence;
* identical sequences from different edit paths share identity;
* different sequences never share identity;
* no tuple / mixed hashing anywhere in the MDP module.
"""
from __future__ import annotations

import hashlib
import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.schema import MRNARecord
from core.constants import START_CODON
from rl.p3_06_mdp import (
    EditAction, sequence_identity, initial_state, transition,
    apply_edit_action,
)
import rl.p3_06_mdp as mdp_mod

INERT_CDS = START_CODON + "GCU" * 4 + "UAA"


def make_record(utr: str = "ACGUACGU") -> MRNARecord:
    return MRNARecord(transcript_id="t", five_utr=utr,
                      cds=INERT_CDS, three_utr="UGCU")


class TestSequenceIdentity:
    def test_identity_is_sha256_of_full_sequence(self):
        rec = make_record()
        expected = hashlib.sha256(rec.seq.encode()).hexdigest()
        assert sequence_identity(rec) == expected

    def test_identity_changes_with_any_single_nt(self):
        rec = make_record()
        base = sequence_identity(rec)
        edited = apply_edit_action(rec, EditAction(op="five_utr_sub", pos=0, nt="G"))
        assert sequence_identity(edited) != base

    def test_same_sequence_same_identity_regardless_of_metadata(self):
        a = make_record()
        b = MRNARecord(transcript_id="other", five_utr=a.five_utr,
                       cds=a.cds, three_utr=a.three_utr,
                       metadata={"x": 1})
        assert sequence_identity(a) == sequence_identity(b)

    def test_state_sequence_hash_matches_identity(self):
        rec = make_record()
        state = initial_state(rec, budget=3)
        assert state.sequence_hash() == sequence_identity(rec)

    def test_state_hash_tracks_current_not_source(self):
        rec = make_record()
        state = initial_state(rec, budget=3)
        action = EditAction(op="five_utr_sub", pos=1, nt="G")
        nxt = transition(state, action)
        assert nxt.sequence_hash() == sequence_identity(nxt.current_mrna)
        assert nxt.sequence_hash() != sequence_identity(rec)


class TestNoMixedHashing:
    def test_module_does_not_use_tuple_identity(self):
        """Source must not build identity from python tuples of edits."""
        src = inspect.getsource(mdp_mod)
        # No tuple-of-edits based identity like `hash((...))` or `, )` keys.
        assert "hash((" not in src
        assert "md5" not in src.lower()

    def test_visited_states_store_strings(self):
        rec = make_record()
        state = initial_state(rec, budget=5)
        action = EditAction(op="five_utr_sub", pos=0, nt="G")
        nxt = transition(state, action)
        for v in nxt.visited_states:
            assert isinstance(v, str) and len(v) == 64  # sha256 hex
