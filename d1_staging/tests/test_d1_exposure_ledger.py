#!/usr/bin/env python
"""Unit tests for D1-02 exposure ledger.

Tests:
- build_ledger_entry produces correct fields for each dataset
- GSE246381 entries have historically_exposed=True and labels forbidden
- D_A datasets (GSE207584, GSE173083) have labels forbidden
- D_C datasets (GSE114002, GSE200304, etc.) have labels allowed
- D_D dataset (GSE145046) has labels allowed
- Incomplete records get exposure_status=incomplete
- Unknown accession triggers fail-safe policy
- Coverage check: every record gets an entry
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "d1"))

from build_exposure_ledger import (
    DATASET_EXPOSURE_POLICY,
    build_ledger_entry,
    main,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _make_record(accession, record_type="paired"):
    """Create a minimal canonical record for testing."""
    rec = {
        "record_id": f"test_{accession}_0",
        "dataset": accession.lower(),
        "accession": accession,
        "region": "5'UTR",
        "source_sequence": "ACGT",
        "candidate_sequence": "ACGT",
        "edit_script": [],
        "edit_script_verified": True,
        "edit_distance": 0,
        "n_ins": 0,
        "n_del": 0,
        "n_sub": 0,
        "path_ambiguity": 1,
        "labels": {"rl": 1.0},
        "metadata": {},
    }
    if record_type:
        rec["metadata"]["record_type"] = record_type
    return rec


# ---------------------------------------------------------------------------
# Tests: per-dataset policy
# ---------------------------------------------------------------------------

class TestPerDatasetPolicy:
    """Verify each dataset gets the correct exposure policy."""

    def test_gse114002_is_dc_e2_unexposed(self):
        rec = _make_record("GSE114002")
        entry = build_ledger_entry(rec)
        assert entry["data_role"] == "D_C"
        assert entry["evidence_grade"] == "E2"
        assert entry["exposure_status"] == "unexposed"
        assert entry["historically_exposed"] is False
        assert entry["labels_allowed_for_new_training"] is True
        assert entry["labels_allowed_for_new_hyperparameter_selection"] is True

    def test_gse149487_is_dc_e2_unexposed(self):
        rec = _make_record("GSE149487")
        entry = build_ledger_entry(rec)
        assert entry["data_role"] == "D_C"
        assert entry["evidence_grade"] == "E2"
        assert entry["exposure_status"] == "unexposed"

    def test_gse217518_is_dc_e2_unexposed(self):
        rec = _make_record("GSE217518")
        entry = build_ledger_entry(rec)
        assert entry["data_role"] == "D_C"
        assert entry["evidence_grade"] == "E2"

    def test_gse200304_is_dc_e2_unexposed(self):
        rec = _make_record("GSE200304")
        entry = build_ledger_entry(rec)
        assert entry["data_role"] == "D_C"
        assert entry["evidence_grade"] == "E2"

    def test_gse145046_is_dd_e2(self):
        rec = _make_record("GSE145046", record_type="observational")
        entry = build_ledger_entry(rec)
        assert entry["data_role"] == "D_D"
        assert entry["evidence_grade"] == "E2"
        assert entry["labels_allowed_for_new_training"] is True

    def test_gse246381_is_de_e4_historically_exposed(self):
        rec = _make_record("GSE246381")
        entry = build_ledger_entry(rec)
        assert entry["data_role"] == "D_E"
        assert entry["evidence_grade"] == "E4"
        assert entry["historically_exposed"] is True
        assert entry["exposure_status"] == "historically_exposed"
        assert entry["labels_allowed_for_new_training"] is False
        assert entry["labels_allowed_for_new_hyperparameter_selection"] is False
        assert entry["historical_exposure_path"] is not None
        assert len(entry["historical_exposure_path"]) > 0

    def test_gse246381_forbidden_wording(self):
        """GSE246381 must not allow 'sealed'/'untouched'/'never-seen' wording."""
        rec = _make_record("GSE246381")
        entry = build_ledger_entry(rec)
        forbidden = set(entry["forbidden_claims"])
        assert "sealed" in forbidden
        assert "untouched" in forbidden
        assert "never-seen_external_test" in forbidden

    def test_gse207584_is_da_observational(self):
        rec = _make_record("GSE207584", record_type="observational")
        entry = build_ledger_entry(rec)
        assert entry["data_role"] == "D_A"
        assert entry["evidence_grade"] == "E2"
        assert entry["exposure_status"] == "observational_no_labels"
        assert entry["labels_allowed_for_new_training"] is False
        assert entry["labels_allowed_for_new_hyperparameter_selection"] is False

    def test_gse173083_is_da_observational(self):
        rec = _make_record("GSE173083", record_type="observational")
        entry = build_ledger_entry(rec)
        assert entry["data_role"] == "D_A"
        assert entry["exposure_status"] == "observational_no_labels"
        assert entry["labels_allowed_for_new_training"] is False

    def test_encsr854ruf_incomplete(self):
        rec = _make_record("ENCSR854RUF", record_type="incomplete")
        entry = build_ledger_entry(rec)
        assert entry["exposure_status"] == "incomplete"
        assert entry["data_role"] == "D_C"


# ---------------------------------------------------------------------------
# Tests: required fields
# ---------------------------------------------------------------------------

class TestRequiredFields:
    """Verify all required fields are present in every entry."""

    REQUIRED = {
        "record_id", "accession", "dataset", "region",
        "data_role", "evidence_grade", "exposure_status",
        "historically_exposed",
        "labels_allowed_for_new_training",
        "labels_allowed_for_new_hyperparameter_selection",
        "allowed_claims", "forbidden_claims",
        "historical_exposure_path",
        "record_type", "notes",
    }

    @pytest.mark.parametrize("accession", list(DATASET_EXPOSURE_POLICY.keys()))
    def test_all_fields_present(self, accession):
        rec = _make_record(accession)
        entry = build_ledger_entry(rec)
        missing = self.REQUIRED - set(entry.keys())
        assert missing == set(), f"Missing fields for {accession}: {missing}"


# ---------------------------------------------------------------------------
# Tests: unknown accession fail-safe
# ---------------------------------------------------------------------------

class TestFailSafe:
    def test_unknown_accession_is_most_restrictive(self):
        rec = _make_record("GSE999999")
        entry = build_ledger_entry(rec)
        assert entry["data_role"] == "D_A"
        assert entry["evidence_grade"] == "E1"
        assert entry["exposure_status"] == "unknown"
        assert entry["labels_allowed_for_new_training"] is False
        assert entry["labels_allowed_for_new_hyperparameter_selection"] is False
        assert "all_claims_until_classified" in entry["forbidden_claims"]


# ---------------------------------------------------------------------------
# Tests: incomplete records
# ---------------------------------------------------------------------------

class TestIncompleteRecords:
    def test_incomplete_gets_incomplete_status(self):
        for acc in ("ENCSR854RUF", "GSE149487", "GSE173083", "GSE217518", "GSE246381"):
            rec = _make_record(acc, record_type="incomplete")
            entry = build_ledger_entry(rec)
            assert entry["exposure_status"] == "incomplete", \
                f"{acc} incomplete record should have exposure_status=incomplete"

    def test_incomplete_gse246381_keeps_labels_forbidden(self):
        """Even as incomplete, GSE246381 keeps historically_exposed constraints."""
        rec = _make_record("GSE246381", record_type="incomplete")
        entry = build_ledger_entry(rec)
        assert entry["historically_exposed"] is True
        assert entry["labels_allowed_for_new_training"] is False
        assert entry["historical_exposure_path"] is not None


# ---------------------------------------------------------------------------
# Tests: coverage (end-to-end)
# ---------------------------------------------------------------------------

class TestCoverage:
    def test_coverage_100_percent(self, tmp_path):
        """Every canonical record gets exactly one ledger entry."""
        records = []
        for acc in DATASET_EXPOSURE_POLICY:
            records.append(_make_record(acc))
            records.append(_make_record(acc, record_type="observational"))
            records.append(_make_record(acc, record_type="incomplete"))

        records_file = tmp_path / "records.jsonl"
        ledger_file = tmp_path / "ledger.jsonl"
        with open(records_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        old_argv = sys.argv
        sys.argv = [
            "build_exposure_ledger.py",
            "--input", str(records_file),
            "--output", str(ledger_file),
        ]
        try:
            exit_code = main()
        except SystemExit as e:
            exit_code = e.code if e.code is not None else 0
        finally:
            sys.argv = old_argv
        assert exit_code == 0

        with open(ledger_file) as f:
            entries = [json.loads(line) for line in f if line.strip()]
        assert len(entries) == len(records)
        record_ids = set(r["record_id"] for r in records)
        ledger_ids = set(e["record_id"] for e in entries)
        assert record_ids == ledger_ids


# ---------------------------------------------------------------------------
# Tests: policy consistency
# ---------------------------------------------------------------------------

class TestPolicyConsistency:
    def test_all_entries_for_dataset_share_policy(self):
        """All records from same dataset must have consistent policy fields."""
        for acc in DATASET_EXPOSURE_POLICY:
            rec1 = _make_record(acc)
            rec2 = _make_record(acc, record_type="observational")
            e1 = build_ledger_entry(rec1)
            e2 = build_ledger_entry(rec2)
            for key in ("data_role", "evidence_grade", "historically_exposed",
                        "labels_allowed_for_new_training",
                        "labels_allowed_for_new_hyperparameter_selection"):
                assert e1[key] == e2[key], \
                    f"{acc}: {key} differs between paired and observational"


# ---------------------------------------------------------------------------
# Tests: all 9 datasets covered
# ---------------------------------------------------------------------------

class TestDatasetCoverage:
    def test_all_9_datasets_have_policy(self):
        expected = {
            "GSE114002", "GSE149487", "GSE217518", "GSE200304",
            "ENCSR854RUF", "GSE145046", "GSE246381",
            "GSE207584", "GSE173083",
        }
        assert set(DATASET_EXPOSURE_POLICY.keys()) == expected
