"""P0-03: cross-role family-cluster leakage tests.

Hard rule: a family_cluster_id must never span the training pool and the
final test pool (homologous sequences share a family cluster).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.p3_02_delta_oracle import DeltaRecord
from scripts.audit_training_membership import audit_family_overlap


def make_rec(record_id: str, family: str, role: str) -> DeltaRecord:
    return DeltaRecord(
        record_id=record_id,
        source_id=f"src_{record_id}",
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
        family_cluster_id=family,
        edit_type="measured_single",
    )


class TestCrossRoleFamilyLeakage:
    def test_disjoint_families_pass(self):
        train = [make_rec("t1", "fam1", "train"), make_rec("t2", "fam2", "val")]
        test = [make_rec("s1", "fam3", "test")]
        report = audit_family_overlap(train, test)
        assert report["pass"] is True
        assert report["n_overlap"] == 0

    def test_shared_family_fails(self):
        train = [make_rec("t1", "fam1", "train")]
        test = [make_rec("s1", "fam1", "test")]
        report = audit_family_overlap(train, test)
        assert report["pass"] is False
        assert report["overlap_ids"] == ["fam1"]

    def test_empty_family_ids_ignored(self):
        train = [make_rec("t1", "", "train")]
        test = [make_rec("s1", "", "test")]
        report = audit_family_overlap(train, test)
        assert report["pass"] is True

    def test_multiple_overlaps_reported(self):
        train = [make_rec("t1", "fam1", "train"), make_rec("t2", "fam2", "train")]
        test = [make_rec("s1", "fam1", "test"), make_rec("s2", "fam2", "test"),
                make_rec("s3", "fam3", "test")]
        report = audit_family_overlap(train, test)
        assert report["n_overlap"] == 2
        assert report["pass"] is False
