"""P0-03: proxy role filtering tests.

Verifies the hard rule that only split_role == "train" proxy records may
enter the P3-02 training pool.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.p3_02_delta_oracle import DeltaRecord
from scripts.audit_training_membership import (
    filter_train_role_proxy,
    audit_proxy_role_compliance,
)


def make_proxy(record_id: str, role: str, seq: str = "ACGUACGUAC") -> DeltaRecord:
    return DeltaRecord(
        record_id=record_id,
        source_id=f"src_{record_id}",
        source_sequence=seq,
        candidate_sequence=seq,
        edit_list=[{"pos": 0, "ref": "A", "alt": "G", "region": "five_utr"}],
        edit_count=1,
        edited_region="five_utr",
        delta=0.1,
        source_value=1.0,
        candidate_value=1.1,
        value_std=0.1,
        confidence="proxy",
        split_role=role,
        family_cluster_id=f"fam_{record_id}",
        edit_type="proxy_single",
    )


class TestProxyRoleFiltering:
    def test_filter_keeps_only_train_role(self):
        records = [
            make_proxy("r1", "train"),
            make_proxy("r2", "val"),
            make_proxy("r3", "test"),
            make_proxy("r4", "train"),
            make_proxy("r5", "ood"),
        ]
        kept = filter_train_role_proxy(records)
        assert {r.record_id for r in kept} == {"r1", "r4"}
        assert all(r.split_role == "train" for r in kept)

    def test_filter_empty_input(self):
        assert filter_train_role_proxy([]) == []

    def test_filter_all_train(self):
        records = [make_proxy(f"r{i}", "train") for i in range(5)]
        assert len(filter_train_role_proxy(records)) == 5

    def test_filter_none_train(self):
        records = [make_proxy(f"r{i}", "test") for i in range(5)]
        assert filter_train_role_proxy(records) == []

    def test_audit_compliance_report(self):
        records = [
            make_proxy("r1", "train"),
            make_proxy("r2", "val"),
            make_proxy("r3", "test"),
        ]
        report = audit_proxy_role_compliance(records)
        assert report["pass"] is True
        assert report["role_counts"] == {"train": 1, "val": 1, "test": 1}
        assert report["n_kept_train_role"] == 1
        assert report["n_excluded"] == 2

    def test_subsample_deterministic_given_seed(self):
        """Proxy subsampling after role filtering must be reproducible."""
        records = [make_proxy(f"r{i:03d}", "train") for i in range(100)]
        kept = filter_train_role_proxy(records)
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        idx1 = rng1.choice(len(kept), 10, replace=False)
        idx2 = rng2.choice(len(kept), 10, replace=False)
        assert (idx1 == idx2).all()


class TestRunP302ProxyFiltering:
    """Guard test: run_p3_02 must not add non-train proxy records to train_all."""

    def test_proxy_role_filter_in_source(self):
        src = (Path(__file__).resolve().parent.parent
               / "scripts" / "run_p3_02.py").read_text()
        assert 'record.split_role == "train"' in src, (
            "run_p3_02.py must filter proxy records by split_role == 'train'"
        )
