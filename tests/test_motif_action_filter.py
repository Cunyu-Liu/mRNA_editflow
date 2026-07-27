"""P0-05: the legal-action builder must enforce the hard motif policy.

Acceptance: actions introducing any of the following are filtered:
* upstream AUG
* cryptic splice donor/acceptor (proxy motifs)
* homopolymer run >= 6
* premature stop
* start/stop violation
* reading-frame violation
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.schema import MRNARecord
from core.constants import START_CODON
from core.motif_policy import (
    creates_upstream_aug, creates_cryptic_splice_site, creates_homopolymer,
    creates_premature_stop, violates_start_stop, violates_reading_frame,
    hard_motif_violations, is_hard_legal,
)
from rl.p3_06_mdp import (
    EditAction, apply_edit_action, build_legal_edit_actions,
)

INERT_CDS = START_CODON + "GCU" * 4 + "UAA"


def make_record(utr: str, cds: str = INERT_CDS) -> MRNARecord:
    return MRNARecord(transcript_id="t", five_utr=utr,
                      cds=cds, three_utr="")


def _child(parent, action):
    return apply_edit_action(parent, action)


class TestIndividualChecks:
    def test_upstream_aug_detected(self):
        parent = make_record("AACAAA")
        child = make_record("AUGAAA")
        assert creates_upstream_aug(parent, child)

    def test_upstream_aug_not_triggered_when_preexisting(self):
        parent = make_record("AUGAAA")
        child = make_record("AUGACA")
        assert not creates_upstream_aug(parent, child)

    def test_cryptic_splice_donor_detected(self):
        parent = make_record("AAAAAA")
        child = make_record("AGGUAU")
        assert creates_cryptic_splice_site(parent, child)

    def test_cryptic_splice_acceptor_detected(self):
        parent = make_record("AAAAAAA")
        child = make_record("AUUCAGA")
        assert creates_cryptic_splice_site(parent, child)

    def test_homopolymer_detected(self):
        parent = make_record("GGGGGA")
        child = make_record("GGGGGG")
        assert creates_homopolymer(parent, child)

    def test_homopolymer_below_threshold_ok(self):
        parent = make_record("GGGGA")
        child = make_record("GGGGG")
        assert not creates_homopolymer(parent, child)

    def test_premature_stop_detected(self):
        parent = make_record("AAAAAA", cds=START_CODON + "GCU" * 4 + "UAA")
        # GCU -> UAA would create premature stop, but that is nonsynonymous and
        # impossible via legal actions; test the check directly.
        child = make_record("AAAAAA", cds=START_CODON + "UAA" + "GCU" * 3 + "UAA")
        assert creates_premature_stop(parent, child)

    def test_start_stop_violation(self):
        bad = make_record("AAAA", cds="CCC" + "GCU" * 3 + "UAA")
        assert violates_start_stop(bad)

    def test_reading_frame_violation(self):
        parent = make_record("AAAA", cds=START_CODON + "GCU" * 3 + "UAA")
        child = make_record("AAAA", cds=START_CODON + "GCU" * 3 + "UA")
        assert violates_reading_frame(parent, child)


class TestLegalActionFilter:
    def test_aug_creating_actions_filtered(self):
        """Edits that create a new AUG in 5'UTR must not be legal."""
        rec = make_record("AUCAAA")  # C->G at pos 2 would create AUG
        legal = build_legal_edit_actions(rec)
        for a in legal:
            if a.is_stop():
                continue
            child = apply_edit_action(rec, a)
            assert not creates_upstream_aug(rec, child), (
                f"illegal uAUG action present: {a}")
        # Sanity: the specific uAUG action must be excluded
        aug_action = EditAction(op="five_utr_sub", pos=2, nt="G")
        assert creates_upstream_aug(rec, apply_edit_action(rec, aug_action))
        assert aug_action not in legal

    def test_homopolymer_actions_filtered(self):
        rec = make_record("GGGGGA")
        legal = build_legal_edit_actions(rec)
        for a in legal:
            if a.is_stop():
                continue
            child = apply_edit_action(rec, a)
            assert not creates_homopolymer(rec, child)

    def test_all_legal_actions_pass_hard_policy(self):
        rec = make_record("ACGUACGUACGU")
        legal = build_legal_edit_actions(rec)
        assert len(legal) > 1
        for a in legal:
            if a.is_stop():
                continue
            child = apply_edit_action(rec, a)
            assert is_hard_legal(rec, child), (
                f"action {a} violates {hard_motif_violations(rec, child)}")

    def test_filter_can_be_disabled_for_audit(self):
        """With enforce_hard_motif=False, uAUG actions reappear (audit path)."""
        rec = make_record("AUCAAA")  # C->G at pos 2 creates AUG
        legal_off = build_legal_edit_actions(rec, enforce_hard_motif=False)
        aug_actions = [a for a in legal_off if not a.is_stop()
                       and creates_upstream_aug(rec, apply_edit_action(rec, a))]
        assert len(aug_actions) > 0

    def test_cds_actions_never_create_premature_stop(self):
        rec = make_record("ACGUACGU", cds=START_CODON + "AAA" * 4 + "UAA")
        legal = build_legal_edit_actions(rec)
        for a in legal:
            if a.is_cds():
                child = apply_edit_action(rec, a)
                assert not creates_premature_stop(rec, child)
                assert not violates_reading_frame(rec, child)
