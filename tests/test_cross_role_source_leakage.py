"""P0-03: cross-role source (mother sequence) leakage tests.

Hard rule: the same source_id must never appear in both the training pool
and the final test pool.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.p3_02_delta_oracle import DeltaRecord
from scripts.audit_training_membership import audit_source_overlap


def make_rec(record_id: str, source_id: str, role: str) -> DeltaRecord:
    return DeltaRecord(
        record_id=record_id,
        source_id=source_id,
        source_sequence="ACGUACGUAC",
        candidate_sequence="ACGUACGUAC",
        edit_list=[],
        edit_count=1,
        edited_region="five_utr",
        delta=0.1,
        source_value=1.0,
        candidate_value=1.1,
        value_std=0.1,
        confidence="measured",
        split_role=role,
        family_cluster_id=f"fam_{source_id}",
        edit_type="measured_single",
    )


class TestCrossRoleSourceLeakage:
    def test_no_overlap_passes(self):
        train = [make_rec("t1", "srcA", "train"), make_rec("t2", "srcB", "val")]
        test = [make_rec("s1", "srcC", "test"), make_rec("s2", "srcD", "test")]
        report = audit_source_overlap(train, test)
        assert report["pass"] is True
        assert report["n_overlap"] == 0

    def test_overlap_detected_and_fails(self):
        train = [make_rec("t1", "srcA", "train")]
        test = [make_rec("s1", "srcA", "test")]
        report = audit_source_overlap(train, test)
        assert report["pass"] is False
        assert report["n_overlap"] == 1
        assert report["overlap_ids"] == ["srcA"]

    def test_val_counts_as_training_side(self):
        """val-role records are part of the training pool for leakage checks."""
        val = [make_rec("v1", "srcA", "val")]
        test = [make_rec("s1", "srcA", "test")]
        report = audit_source_overlap(val, test)
        assert report["pass"] is False

    def test_empty_pools(self):
        assert audit_source_overlap([], [])["pass"] is True
