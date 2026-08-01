"""Unit tests for B0-04 evaluation track assignment + ambiguity audit.

B0-04 acceptance:
  - track-role ambiguity = 0

Covers:
  - record_is_paired / record_has_source
  - is_eligible per track
  - role_for_track
  - exposure_class_for_role
  - assign_track_roles
  - check_cross_track_consistency (clean + ambiguous)
  - check_single_role_per_track
  - check_eligibility_correctness (clean + violations)
  - check_train_eval_boundary
  - summarize_cross_split_roles
  - summarize_track_counts
  - write_manifest (deterministic sha256)
  - run_eval_track_audit (integration with temp fixtures)

Run: pytest d1_staging/tests/test_b0_eval_tracks.py -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
B0_SCRIPTS = os.path.join(HERE, "..", "scripts", "b0")
sys.path.insert(0, B0_SCRIPTS)

from canonical_schemas import EVAL_TRACKS  # noqa: E402
from eval_tracks import (  # noqa: E402
    EXPOSURE_EVAL_TEST,
    EXPOSURE_EVAL_VAL,
    EXPOSURE_NONE,
    EXPOSURE_TRAIN,
    SPLIT_FILES,
    SPLIT_ROLE_TO_EXPOSURE,
    assign_track_roles,
    check_cross_track_consistency,
    check_eligibility_correctness,
    check_single_role_per_track,
    check_train_eval_boundary,
    exposure_class_for_role,
    is_eligible,
    load_exposure_ledger,
    load_jsonl,
    record_has_source,
    record_is_paired,
    role_for_track,
    run_eval_track_audit,
    summarize_cross_split_roles,
    summarize_track_counts,
    write_manifest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _paired_rec(rid="r1", region="5'UTR", accession="GSE114002"):
    return {
        "record_id": rid,
        "accession": accession,
        "region": region,
        "source_sequence": "ACGTACGT",
        "candidate_sequence": "ACGTACGA",
    }


def _source_only_rec(rid="r2", region="5'UTR", accession="GSE114002"):
    return {
        "record_id": rid,
        "accession": accession,
        "region": region,
        "source_sequence": "ACGTACGT",
        "candidate_sequence": None,
    }


def _split_entry(rid, split_role, region="5'UTR", accession="GSE114002"):
    return {
        "record_id": rid,
        "accession": accession,
        "region": region,
        "split": split_role,
        "split_type": "test_split",
    }


def _ledger_entry(rid, data_role="D_C", record_type="paired"):
    return {
        "record_id": rid,
        "data_role": data_role,
        "record_type": record_type,
    }


# ---------------------------------------------------------------------------
# record_is_paired / record_has_source
# ---------------------------------------------------------------------------

class TestRecordPredicates:
    def test_paired_has_both_sequences(self):
        assert record_is_paired(_paired_rec()) is True

    def test_paired_missing_candidate(self):
        rec = _paired_rec()
        rec["candidate_sequence"] = None
        assert record_is_paired(rec) is False

    def test_paired_missing_source(self):
        rec = _paired_rec()
        rec["source_sequence"] = ""
        assert record_is_paired(rec) is False

    def test_paired_empty_candidate(self):
        rec = _paired_rec()
        rec["candidate_sequence"] = ""
        assert record_is_paired(rec) is False

    def test_has_source_true(self):
        assert record_has_source(_paired_rec()) is True

    def test_has_source_false(self):
        assert record_has_source(_source_only_rec()) is True
        rec = _source_only_rec()
        rec["source_sequence"] = None
        assert record_has_source(rec) is False


# ---------------------------------------------------------------------------
# is_eligible
# ---------------------------------------------------------------------------

class TestEligibility:
    def test_paired_eligible_all_tracks(self):
        rec = _paired_rec()
        for track in EVAL_TRACKS:
            assert is_eligible(track, rec) is True

    def test_source_only_not_eligible_for_paired_tracks(self):
        rec = _source_only_rec()
        assert is_eligible("closed_measured_pool", rec) is False
        assert is_eligible("heldout_generative", rec) is False

    def test_source_only_eligible_for_open_generation(self):
        rec = _source_only_rec()
        assert is_eligible("open_legal_generation", rec) is True

    def test_empty_record_not_eligible(self):
        rec = {"record_id": "x"}
        for track in EVAL_TRACKS:
            assert is_eligible(track, rec) is False


# ---------------------------------------------------------------------------
# role_for_track
# ---------------------------------------------------------------------------

class TestRoleForTrack:
    def test_closed_measured_pool_train(self):
        assert role_for_track("closed_measured_pool", "train", True) == "train_pair"

    def test_closed_measured_pool_test(self):
        assert role_for_track("closed_measured_pool", "test", True) == "test_pair"

    def test_heldout_generative_val(self):
        assert role_for_track("heldout_generative", "val", True) == "val_heldout_source"

    def test_open_legal_generation_train(self):
        assert role_for_track("open_legal_generation", "train", True) == "train_gen_source"

    def test_not_eligible_returns_none(self):
        for track in EVAL_TRACKS:
            assert role_for_track(track, "train", False) == "none"

    def test_unknown_split_role_returns_none(self):
        assert role_for_track("closed_measured_pool", "weird", True) == "none"

    def test_all_track_role_combinations(self):
        for track in EVAL_TRACKS:
            for sr in ("train", "val", "test"):
                role = role_for_track(track, sr, True)
                assert role.startswith(sr + "_")
                assert role != "none"


# ---------------------------------------------------------------------------
# exposure_class_for_role
# ---------------------------------------------------------------------------

class TestExposureClass:
    def test_train_roles(self):
        assert exposure_class_for_role("train_pair") == EXPOSURE_TRAIN
        assert exposure_class_for_role("train_heldout_source") == EXPOSURE_TRAIN
        assert exposure_class_for_role("train_gen_source") == EXPOSURE_TRAIN

    def test_val_roles(self):
        assert exposure_class_for_role("val_pair") == EXPOSURE_EVAL_VAL
        assert exposure_class_for_role("val_heldout_source") == EXPOSURE_EVAL_VAL
        assert exposure_class_for_role("val_gen_source") == EXPOSURE_EVAL_VAL

    def test_test_roles(self):
        assert exposure_class_for_role("test_pair") == EXPOSURE_EVAL_TEST
        assert exposure_class_for_role("test_heldout_source") == EXPOSURE_EVAL_TEST
        assert exposure_class_for_role("test_gen_source") == EXPOSURE_EVAL_TEST

    def test_none(self):
        assert exposure_class_for_role("none") == EXPOSURE_NONE

    def test_unknown_role(self):
        assert exposure_class_for_role("garbage") == EXPOSURE_NONE


# ---------------------------------------------------------------------------
# assign_track_roles
# ---------------------------------------------------------------------------

class TestAssignTrackRoles:
    def test_paired_train_record(self):
        recs = {"r1": _paired_rec("r1")}
        entries = [_split_entry("r1", "train")]
        assignments = assign_track_roles("5utr_source_disjoint", entries, recs, {})
        assert len(assignments) == 1
        a = assignments[0]
        assert a["split_role"] == "train"
        assert a["exposure_class"] == EXPOSURE_TRAIN
        assert set(a["eligible_tracks"]) == set(EVAL_TRACKS)
        assert a["track_roles"]["closed_measured_pool"] == "train_pair"
        assert a["track_roles"]["heldout_generative"] == "train_heldout_source"
        assert a["track_roles"]["open_legal_generation"] == "train_gen_source"
        # All exposure classes are TRAIN
        for v in a["track_exposure_classes"].values():
            assert v == EXPOSURE_TRAIN

    def test_paired_test_record(self):
        recs = {"r1": _paired_rec("r1")}
        entries = [_split_entry("r1", "test")]
        assignments = assign_track_roles("study_disjoint", entries, recs, {})
        a = assignments[0]
        assert a["exposure_class"] == EXPOSURE_EVAL_TEST
        assert a["track_roles"]["closed_measured_pool"] == "test_pair"

    def test_source_only_record_open_generation_only(self):
        recs = {"r2": _source_only_rec("r2")}
        entries = [_split_entry("r2", "test")]
        assignments = assign_track_roles("study_disjoint", entries, recs, {})
        a = assignments[0]
        assert a["eligible_tracks"] == ["open_legal_generation"]
        assert a["track_roles"]["closed_measured_pool"] == "none"
        assert a["track_roles"]["heldout_generative"] == "none"
        assert a["track_roles"]["open_legal_generation"] == "test_gen_source"
        assert a["track_exposure_classes"]["closed_measured_pool"] == EXPOSURE_NONE
        assert a["track_exposure_classes"]["open_legal_generation"] == EXPOSURE_EVAL_TEST

    def test_ledger_fields_propagated(self):
        recs = {"r1": _paired_rec("r1")}
        ledger = {"r1": _ledger_entry("r1", "D_C", "paired")}
        entries = [_split_entry("r1", "train")]
        a = assign_track_roles("s", entries, recs, ledger)[0]
        assert a["data_role"] == "D_C"
        assert a["record_type"] == "paired"

    def test_missing_canonical_record(self):
        entries = [_split_entry("missing", "train")]
        assignments = assign_track_roles("s", entries, {}, {})
        a = assignments[0]
        # Missing record -> not eligible for any track
        assert a["eligible_tracks"] == []
        for role in a["track_roles"].values():
            assert role == "none"

    def test_val_record(self):
        recs = {"r1": _paired_rec("r1")}
        entries = [_split_entry("r1", "val")]
        a = assign_track_roles("s", entries, recs, {})[0]
        assert a["exposure_class"] == EXPOSURE_EVAL_VAL
        assert a["track_roles"]["closed_measured_pool"] == "val_pair"


# ---------------------------------------------------------------------------
# check_cross_track_consistency
# ---------------------------------------------------------------------------

class TestCrossTrackConsistency:
    def test_clean_passes(self):
        recs = {"r1": _paired_rec("r1")}
        entries = [_split_entry("r1", "train"), _split_entry("r1b", "test")]
        # Use distinct records to avoid within-list conflict
        recs["r1b"] = _paired_rec("r1b")
        assignments = assign_track_roles("s", entries, recs, {})
        res = check_cross_track_consistency(assignments)
        assert res["pass"] is True
        assert res["n_violations"] == 0
        assert res["hard_gate"] is True

    def test_train_vs_eval_violation(self):
        # Fabricate an assignment where one record is TRAIN in one track and
        # EVAL_TEST in another (this cannot happen via assign_track_roles but
        # tests the auditor directly).
        a = {
            "split": "s",
            "record_id": "r1",
            "track_exposure_classes": {
                "closed_measured_pool": EXPOSURE_TRAIN,
                "heldout_generative": EXPOSURE_EVAL_TEST,
                "open_legal_generation": EXPOSURE_TRAIN,
            },
        }
        res = check_cross_track_consistency([a])
        assert res["pass"] is False
        assert res["n_violations"] == 1
        assert "TRAIN vs EVAL" in res["violations"][0]["reason"]

    def test_val_vs_test_violation(self):
        a = {
            "split": "s",
            "record_id": "r1",
            "track_exposure_classes": {
                "closed_measured_pool": EXPOSURE_EVAL_VAL,
                "heldout_generative": EXPOSURE_EVAL_TEST,
                "open_legal_generation": EXPOSURE_NONE,
            },
        }
        res = check_cross_track_consistency([a])
        assert res["pass"] is False
        assert "EVAL_VAL vs EVAL_TEST" in res["violations"][0]["reason"]

    def test_none_only_passes(self):
        a = {
            "split": "s",
            "record_id": "r1",
            "track_exposure_classes": {t: EXPOSURE_NONE for t in EVAL_TRACKS},
        }
        res = check_cross_track_consistency([a])
        assert res["pass"] is True

    def test_source_only_record_no_conflict(self):
        # A source-only record is NONE in 2 tracks and EVAL_TEST in 1 — no conflict
        a = {
            "split": "s",
            "record_id": "r1",
            "track_exposure_classes": {
                "closed_measured_pool": EXPOSURE_NONE,
                "heldout_generative": EXPOSURE_NONE,
                "open_legal_generation": EXPOSURE_EVAL_TEST,
            },
        }
        res = check_cross_track_consistency([a])
        assert res["pass"] is True


# ---------------------------------------------------------------------------
# check_single_role_per_track
# ---------------------------------------------------------------------------

class TestSingleRolePerTrack:
    def test_clean_passes(self):
        recs = {"r1": _paired_rec("r1")}
        entries = [_split_entry("r1", "train")]
        assignments = assign_track_roles("s", entries, recs, {})
        res = check_single_role_per_track(assignments)
        assert res["pass"] is True
        assert res["n_violations"] == 0

    def test_duplicate_detected(self):
        # Same (split, record, track) appearing twice
        a1 = {
            "split": "s", "record_id": "r1",
            "track_roles": {"closed_measured_pool": "train_pair"},
        }
        a2 = {
            "split": "s", "record_id": "r1",
            "track_roles": {"closed_measured_pool": "val_pair"},
        }
        res = check_single_role_per_track([a1, a2])
        assert res["pass"] is False
        assert res["n_violations"] >= 1


# ---------------------------------------------------------------------------
# check_eligibility_correctness
# ---------------------------------------------------------------------------

class TestEligibilityCorrectness:
    def test_clean_passes(self):
        recs = {"r1": _paired_rec("r1")}
        entries = [_split_entry("r1", "train")]
        assignments = assign_track_roles("s", entries, recs, {})
        res = check_eligibility_correctness(assignments, recs)
        assert res["pass"] is True
        assert res["n_violations"] == 0
        assert res["hard_gate"] is True

    def test_missing_canonical_record(self):
        entries = [_split_entry("ghost", "train")]
        assignments = assign_track_roles("s", entries, {}, {})
        res = check_eligibility_correctness(assignments, {})
        assert res["pass"] is False
        assert res["n_violations"] == 1
        assert "missing" in res["violations"][0]["reason"]

    def test_assigned_role_but_not_eligible(self):
        # Fabricate: paired role assigned to a source-only record
        recs = {"r2": _source_only_rec("r2")}
        entries = [_split_entry("r2", "test")]
        assignments = assign_track_roles("s", entries, recs, {})
        # Manually corrupt: assign closed_measured_pool a non-none role
        assignments[0]["track_roles"]["closed_measured_pool"] = "test_pair"
        res = check_eligibility_correctness(assignments, recs)
        assert res["pass"] is False
        found = [v for v in res["violations"] if v.get("track") == "closed_measured_pool"]
        assert len(found) == 1

    def test_eligible_but_assigned_none(self):
        recs = {"r1": _paired_rec("r1")}
        entries = [_split_entry("r1", "train")]
        assignments = assign_track_roles("s", entries, recs, {})
        # Manually corrupt: set a paired record's role to none
        assignments[0]["track_roles"]["closed_measured_pool"] = "none"
        res = check_eligibility_correctness(assignments, recs)
        assert res["pass"] is False


# ---------------------------------------------------------------------------
# check_train_eval_boundary
# ---------------------------------------------------------------------------

class TestTrainEvalBoundary:
    def test_clean_passes(self):
        recs = {"r1": _paired_rec("r1"), "r2": _paired_rec("r2")}
        entries = [_split_entry("r1", "train"), _split_entry("r2", "test")]
        assignments = assign_track_roles("s", entries, recs, {})
        res = check_train_eval_boundary(assignments)
        assert res["pass"] is True
        assert res["n_violations"] == 0

    def test_violation(self):
        a = {
            "split": "s",
            "record_id": "r1",
            "track_exposure_classes": {
                "closed_measured_pool": EXPOSURE_TRAIN,
                "heldout_generative": EXPOSURE_EVAL_VAL,
                "open_legal_generation": EXPOSURE_NONE,
            },
        }
        res = check_train_eval_boundary([a])
        assert res["pass"] is False
        assert res["n_violations"] == 1


# ---------------------------------------------------------------------------
# summarize_cross_split_roles
# ---------------------------------------------------------------------------

class TestCrossSplitSummary:
    def test_single_split_no_multi(self):
        recs = {"r1": _paired_rec("r1")}
        entries = [_split_entry("r1", "train")]
        a = assign_track_roles("s1", entries, recs, {})
        res = summarize_cross_split_roles(a)
        assert res["n_records_with_multiple_split_roles"] == 0
        assert res["hard_gate"] is False

    def test_multi_split_multi_role(self):
        recs = {"r1": _paired_rec("r1")}
        a1 = assign_track_roles("s1", [_split_entry("r1", "test")], recs, {})
        a2 = assign_track_roles("s2", [_split_entry("r1", "train")], recs, {})
        res = summarize_cross_split_roles(a1 + a2)
        assert res["n_records_with_multiple_split_roles"] == 1
        assert "r1" in res["examples"]


# ---------------------------------------------------------------------------
# summarize_track_counts
# ---------------------------------------------------------------------------

class TestTrackCounts:
    def test_counts_match(self):
        recs = {"r1": _paired_rec("r1"), "r2": _paired_rec("r2")}
        entries = [_split_entry("r1", "train"), _split_entry("r2", "test")]
        a = assign_track_roles("s1", entries, recs, {})
        res = summarize_track_counts(a)
        assert res["hard_gate"] is False
        # Each track should have 1 train + 1 test
        for track in EVAL_TRACKS:
            key = f"s1/{track}"
            assert key in res["counts"]


# ---------------------------------------------------------------------------
# write_manifest
# ---------------------------------------------------------------------------

class TestWriteManifest:
    def test_deterministic_sha256(self, tmp_path):
        recs = {"r1": _paired_rec("r1")}
        entries = [_split_entry("r1", "train")]
        a = assign_track_roles("s", entries, recs, {})
        out1 = tmp_path / "m1.jsonl"
        out2 = tmp_path / "m2.jsonl"
        sha1 = write_manifest(a, str(out1))
        sha2 = write_manifest(a, str(out2))
        assert sha1 == sha2
        assert out1.exists()
        # Verify content is valid JSONL
        lines = out1.read_text().strip().split("\n")
        assert len(lines) == 1
        d = json.loads(lines[0])
        assert d["record_id"] == "r1"
        assert "track_roles" in d

    def test_sha_changes_with_data(self, tmp_path):
        recs = {"r1": _paired_rec("r1")}
        a1 = assign_track_roles("s", [_split_entry("r1", "train")], recs, {})
        a2 = assign_track_roles("s", [_split_entry("r1", "test")], recs, {})
        sha1 = write_manifest(a1, str(tmp_path / "m1.jsonl"))
        sha2 = write_manifest(a2, str(tmp_path / "m2.jsonl"))
        assert sha1 != sha2


# ---------------------------------------------------------------------------
# run_eval_track_audit (integration)
# ---------------------------------------------------------------------------

class TestRunEvalTrackAudit:
    def _setup_fixtures(self, tmp_path):
        splits_dir = tmp_path / "splits"
        splits_dir.mkdir()
        # Build a tiny split with 2 paired records (train + test) for 5utr
        recs = {
            "r1": _paired_rec("r1", "5'UTR"),
            "r2": _paired_rec("r2", "5'UTR"),
        }
        canonical_path = tmp_path / "canonical.jsonl"
        with open(canonical_path, "w") as f:
            for r in recs.values():
                f.write(json.dumps(r) + "\n")
        ledger_path = tmp_path / "ledger.jsonl"
        with open(ledger_path, "w") as f:
            for rid in recs:
                f.write(json.dumps(_ledger_entry(rid)) + "\n")
        # Only write the 5utr split; others will be skipped
        split_path = splits_dir / SPLIT_FILES["5utr_source_disjoint"]
        with open(split_path, "w") as f:
            f.write(json.dumps(_split_entry("r1", "train", "5'UTR")) + "\n")
            f.write(json.dumps(_split_entry("r2", "test", "5'UTR")) + "\n")
        return str(splits_dir), str(canonical_path), str(ledger_path)

    def test_audit_passes_on_clean_fixtures(self, tmp_path):
        splits_dir, canonical_path, ledger_path = self._setup_fixtures(tmp_path)
        report = run_eval_track_audit(splits_dir, canonical_path, ledger_path)
        assert report["overall_pass"] is True
        assert report["acceptance"]["track_role_ambiguity_must_be_zero"] is True
        # All hard-gate checks pass
        for c in report["hard_gate_checks"]:
            assert c["pass"] is True, c["name"]
        assert report["per_split"]["5utr_source_disjoint"]["n_records"] == 2

    def test_audit_reports_track_list(self, tmp_path):
        splits_dir, canonical_path, ledger_path = self._setup_fixtures(tmp_path)
        report = run_eval_track_audit(splits_dir, canonical_path, ledger_path)
        assert set(report["tracks"]) == set(EVAL_TRACKS)

    def test_audit_missing_split_skipped(self, tmp_path):
        splits_dir, canonical_path, ledger_path = self._setup_fixtures(tmp_path)
        report = run_eval_track_audit(splits_dir, canonical_path, ledger_path)
        # 3 of 4 splits missing -> error entries
        for name in ("3utr_source_disjoint", "study_disjoint", "cross_region_transfer"):
            assert "error" in report["per_split"][name]

    def test_source_only_record_in_split(self, tmp_path):
        """A source-only record should be eligible only for open_legal_generation
        and should not create ambiguity."""
        splits_dir = tmp_path / "splits"
        splits_dir.mkdir()
        recs = {"r1": _source_only_rec("r1", "5'UTR")}
        canonical_path = tmp_path / "canonical.jsonl"
        with open(canonical_path, "w") as f:
            for r in recs.values():
                f.write(json.dumps(r) + "\n")
        ledger_path = tmp_path / "ledger.jsonl"
        with open(ledger_path, "w") as f:
            f.write(json.dumps(_ledger_entry("r1", "D_A", "observational")) + "\n")
        split_path = splits_dir / SPLIT_FILES["5utr_source_disjoint"]
        with open(split_path, "w") as f:
            f.write(json.dumps(_split_entry("r1", "test", "5'UTR")) + "\n")
        report = run_eval_track_audit(str(splits_dir), str(canonical_path), str(ledger_path))
        assert report["overall_pass"] is True
        # The source-only record should be eligible for 1 track only
        # (checked via eligibility_correctness passing)


# ---------------------------------------------------------------------------
# load helpers
# ---------------------------------------------------------------------------

class TestLoadHelpers:
    def test_load_jsonl(self, tmp_path):
        p = tmp_path / "f.jsonl"
        p.write_text('{"a":1}\n{"b":2}\n\n')
        out = load_jsonl(str(p))
        assert out == [{"a": 1}, {"b": 2}]

    def test_load_exposure_ledger_missing(self, tmp_path):
        out = load_exposure_ledger(str(tmp_path / "nope.jsonl"))
        assert out == {}

    def test_load_exposure_ledger_ok(self, tmp_path):
        p = tmp_path / "ledger.jsonl"
        p.write_text(json.dumps(_ledger_entry("r1")) + "\n")
        out = load_exposure_ledger(str(p))
        assert "r1" in out
        assert out["r1"]["data_role"] == "D_C"


# ---------------------------------------------------------------------------
# EVAL_TRACKS sanity
# ---------------------------------------------------------------------------

class TestEvalTracksConstant:
    def test_three_tracks(self):
        assert len(EVAL_TRACKS) == 3

    def test_tracks_match_contract(self):
        assert set(EVAL_TRACKS) == {
            "closed_measured_pool",
            "heldout_generative",
            "open_legal_generation",
        }

    def test_split_role_mapping_complete(self):
        assert set(SPLIT_ROLE_TO_EXPOSURE) == {"train", "val", "test"}
