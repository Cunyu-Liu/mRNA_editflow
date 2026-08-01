"""Unit tests for B0-01 canonical_schemas.

B0-01 acceptance: schemas defined and tested.

Covers:
  - EditOp construction + validation
  - EditScript construction + invariants (no STOP, counts match)
  - UTREditRecord construction + round-trip with D1 flat form
  - UTREditRecord paired vs observational consistency
  - GenerationTask track/region/budget constraints
  - validate_canonical_records_file on a temp file
  - Cross-check that D1 canonical_records.jsonl format is accepted
    (via a hand-built record matching the D1 flat shape)

Run: pytest d1_staging/tests/test_b0_canonical_schemas.py -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Make the b0 scripts dir importable
HERE = os.path.dirname(os.path.abspath(__file__))
B0_SCRIPTS = os.path.join(HERE, "..", "scripts", "b0")
sys.path.insert(0, B0_SCRIPTS)

from canonical_schemas import (  # noqa: E402
    ALPHABET,
    ALLOWED_CLAIMS_POOL,
    ALL_REGION_VALUES,
    EVAL_TRACKS,
    EditOp,
    EditScript,
    FORBIDDEN_SEALED_WORDING,
    GenerationTask,
    OP_VALUES,
    REGION_VALUES,
    RECORD_TYPES,
    SchemaError,
    UTREditRecord,
    validate_canonical_records_file,
    validate_generation_task,
    validate_utr_edit_record,
)


# ---------------------------------------------------------------------------
# EditOp
# ---------------------------------------------------------------------------


class TestEditOp:
    def test_construct_ins(self):
        op = EditOp("INS", 3, "A")
        assert op.op == "INS"
        assert op.pos == 3
        assert op.token == "A"

    def test_construct_del(self):
        op = EditOp("DEL", 2, "")
        assert op.op == "DEL" and op.token == ""

    def test_construct_sub(self):
        op = EditOp("SUB", 0, "G")
        assert op.op == "SUB" and op.token == "G"

    def test_construct_stop(self):
        op = EditOp("STOP", 0, "")
        assert op.op == "STOP" and op.token == ""

    def test_frozen(self):
        op = EditOp("SUB", 0, "G")
        with pytest.raises(Exception):
            op.op = "INS"  # type: ignore

    def test_to_from_dict_roundtrip(self):
        op = EditOp("INS", 5, "T")
        d = op.to_dict()
        assert d == {"op": "INS", "pos": 5, "token": "T"}
        op2 = EditOp.from_dict(d)
        assert op == op2

    def test_from_dict_missing_token_defaults_empty(self):
        op = EditOp.from_dict({"op": "DEL", "pos": 1})
        assert op.token == ""

    def test_invalid_op_raises(self):
        with pytest.raises(SchemaError):
            EditOp("XXX", 0, "")

    def test_negative_pos_raises(self):
        with pytest.raises(SchemaError):
            EditOp("INS", -1, "A")

    def test_ins_invalid_token_raises(self):
        with pytest.raises(SchemaError):
            EditOp("INS", 0, "X")

    def test_sub_invalid_token_raises(self):
        with pytest.raises(SchemaError):
            EditOp("SUB", 0, "U")

    def test_del_nonempty_token_raises(self):
        with pytest.raises(SchemaError):
            EditOp("DEL", 0, "A")

    def test_stop_nonempty_token_raises(self):
        with pytest.raises(SchemaError):
            EditOp("STOP", 0, "A")


# ---------------------------------------------------------------------------
# EditScript
# ---------------------------------------------------------------------------


class TestEditScript:
    def _make_ops(self):
        return (EditOp("SUB", 0, "G"), EditOp("INS", 3, "T"))

    def test_basic_construct(self):
        es = EditScript(
            ops=self._make_ops(),
            verified=True,
            edit_distance=2,
            n_ins=1,
            n_del=0,
            n_sub=1,
            path_ambiguity=3,
        )
        assert es.edit_distance == 2
        assert es.path_ambiguity == 3

    def test_empty_script_observational(self):
        es = EditScript(ops=(), verified=True, edit_distance=0, n_ins=0, n_del=0, n_sub=0, path_ambiguity=1)
        assert es.ops == ()

    def test_frozen(self):
        es = EditScript(ops=self._make_ops(), verified=True, edit_distance=2, n_ins=1, n_del=0, n_sub=1, path_ambiguity=1)
        with pytest.raises(Exception):
            es.verified = False  # type: ignore

    def test_stop_in_ops_rejected(self):
        with pytest.raises(SchemaError):
            EditScript(
                ops=(EditOp("STOP", 0, ""),),
                verified=True,
                edit_distance=0,
                n_ins=0, n_del=0, n_sub=0,
                path_ambiguity=1,
            )

    def test_count_mismatch_ins_rejected(self):
        with pytest.raises(SchemaError, match="n_ins mismatch"):
            EditScript(
                ops=(EditOp("INS", 0, "A"),),
                verified=True,
                edit_distance=1,
                n_ins=0, n_del=0, n_sub=0,
                path_ambiguity=1,
            )

    def test_count_mismatch_del_rejected(self):
        # n_del declared 0 but op is DEL
        with pytest.raises(SchemaError, match="n_del mismatch"):
            EditScript(
                ops=(EditOp("DEL", 0, ""),),
                verified=True,
                edit_distance=1,
                n_ins=0, n_del=0, n_sub=0,
                path_ambiguity=1,
            )

    def test_count_mismatch_sub_rejected(self):
        with pytest.raises(SchemaError, match="n_sub mismatch"):
            EditScript(
                ops=(EditOp("SUB", 0, "G"),),
                verified=True,
                edit_distance=1,
                n_ins=0, n_del=0, n_sub=0,
                path_ambiguity=1,
            )

    def test_edit_distance_mismatch_rejected(self):
        with pytest.raises(SchemaError, match="edit_distance mismatch"):
            EditScript(
                ops=(EditOp("SUB", 0, "G"),),
                verified=True,
                edit_distance=2,  # wrong
                n_ins=0, n_del=0, n_sub=1,
                path_ambiguity=1,
            )

    def test_path_ambiguity_zero_rejected(self):
        with pytest.raises(SchemaError):
            EditScript(
                ops=(),
                verified=True,
                edit_distance=0,
                n_ins=0, n_del=0, n_sub=0,
                path_ambiguity=0,
            )

    def test_to_dict_roundtrip(self):
        es = EditScript(
            ops=self._make_ops(),
            verified=True,
            edit_distance=2, n_ins=1, n_del=0, n_sub=1,
            path_ambiguity=2,
        )
        d = es.to_dict()
        assert d["ops"] == [
            {"op": "SUB", "pos": 0, "token": "G"},
            {"op": "INS", "pos": 3, "token": "T"},
        ]
        assert d["edit_distance"] == 2
        assert d["path_ambiguity"] == 2


# ---------------------------------------------------------------------------
# UTREditRecord
# ---------------------------------------------------------------------------


def _paired_record_dict():
    """A dict matching D1 canonical_records.jsonl flat form (paired)."""
    return {
        "record_id": "GSE114002_test_0",
        "dataset": "sample2019",
        "accession": "GSE114002",
        "region": "5'UTR",
        "source_sequence": "ACGTACGT",
        "candidate_sequence": "ACGTACGA",
        "edit_script": [{"op": "SUB", "pos": 7, "token": "A"}],
        "edit_script_verified": True,
        "edit_distance": 1,
        "n_ins": 0,
        "n_del": 0,
        "n_sub": 1,
        "path_ambiguity": 1,
        "labels": {"rl": 4.5},
        "metadata": {"source_file": "test.csv", "library": "test", "record_type": "paired"},
    }


def _observational_record_dict():
    """A dict matching D1 flat form (observational, D_A)."""
    return {
        "record_id": "GSE207584_0",
        "dataset": "gse207584",
        "accession": "GSE207584",
        "region": "CDS",
        "source_sequence": None,
        "candidate_sequence": "ATGCATGC",
        "edit_script": [],
        "edit_script_verified": True,
        "edit_distance": 0,
        "n_ins": 0,
        "n_del": 0,
        "n_sub": 0,
        "path_ambiguity": 1,
        "labels": {},
        "metadata": {"source_file": "test.csv", "data_role": "D_A", "record_type": "observational"},
    }


class TestUTREditRecord:
    def test_from_dict_paired_roundtrip(self):
        d = _paired_record_dict()
        rec = UTREditRecord.from_dict(d)
        assert rec.record_id == "GSE114002_test_0"
        assert rec.is_paired
        assert rec.edit_script.n_sub == 1
        # back to dict should match the original flat form
        d2 = rec.to_dict()
        for k in d:
            assert d2[k] == d[k], f"mismatch on {k}: {d2[k]!r} vs {d[k]!r}"

    def test_from_dict_observational_roundtrip(self):
        d = _observational_record_dict()
        rec = UTREditRecord.from_dict(d)
        assert rec.is_observational
        assert rec.source_sequence is None
        assert rec.edit_script.ops == ()
        d2 = rec.to_dict()
        for k in d:
            assert d2[k] == d[k], f"mismatch on {k}: {d2[k]!r} vs {d[k]!r}"

    def test_paired_requires_source_sequence(self):
        d = _paired_record_dict()
        d["source_sequence"] = None
        with pytest.raises(SchemaError, match="paired records must have non-None source_sequence"):
            UTREditRecord.from_dict(d)

    def test_paired_requires_verified_script(self):
        d = _paired_record_dict()
        d["edit_script_verified"] = False
        with pytest.raises(SchemaError, match="paired records must have edit_script.verified"):
            UTREditRecord.from_dict(d)

    def test_observational_rejects_source_sequence(self):
        d = _observational_record_dict()
        d["source_sequence"] = "ACGT"
        with pytest.raises(SchemaError, match="observational records must have source_sequence == None"):
            UTREditRecord.from_dict(d)

    def test_observational_rejects_nonempty_edit_script(self):
        d = _observational_record_dict()
        d["edit_script"] = [{"op": "SUB", "pos": 0, "token": "G"}]
        d["edit_distance"] = 1
        d["n_sub"] = 1
        with pytest.raises(SchemaError, match="observational records must have empty edit_script.ops"):
            UTREditRecord.from_dict(d)

    def test_missing_record_type_inferred_paired(self):
        """Legacy D1 paired records lacked record_type — must be inferred."""
        d = _paired_record_dict()
        del d["metadata"]["record_type"]
        rec = UTREditRecord.from_dict(d)
        assert rec.is_paired
        assert rec.metadata["record_type"] == "paired"

    def test_missing_record_type_inferred_observational(self):
        d = _observational_record_dict()
        del d["metadata"]["record_type"]
        rec = UTREditRecord.from_dict(d)
        assert rec.is_observational
        assert rec.metadata["record_type"] == "observational"

    def test_invalid_record_type_rejected(self):
        d = _paired_record_dict()
        d["metadata"]["record_type"] = "bogus"
        with pytest.raises(SchemaError, match="metadata.record_type"):
            UTREditRecord.from_dict(d)

    def test_invalid_region_rejected(self):
        d = _paired_record_dict()
        d["region"] = "CDS"
        # paired record on CDS region — region value itself is valid, but paired on
        # CDS is a contract issue (CDS out-of-scope). The schema accepts the region
        # string but downstream splits (B0-02) will exclude it. We only check that
        # the region enum is enforced.
        rec = UTREditRecord.from_dict(d)
        assert rec.region == "CDS"

    def test_unknown_region_rejected(self):
        d = _paired_record_dict()
        d["region"] = "UTR5"
        with pytest.raises(SchemaError, match="region must be one of"):
            UTREditRecord.from_dict(d)

    def test_non_acgt_source_rejected(self):
        d = _paired_record_dict()
        d["source_sequence"] = "ACGTNACGT"
        with pytest.raises(SchemaError, match="source_sequence contains non-ACGT"):
            UTREditRecord.from_dict(d)

    def test_non_acgt_candidate_rejected(self):
        d = _paired_record_dict()
        d["candidate_sequence"] = "ACGTACGX"
        with pytest.raises(SchemaError, match="candidate_sequence contains non-ACGT"):
            UTREditRecord.from_dict(d)

    def test_empty_record_id_rejected(self):
        d = _paired_record_dict()
        d["record_id"] = ""
        with pytest.raises(SchemaError, match="record_id"):
            UTREditRecord.from_dict(d)

    def test_non_numeric_label_rejected(self):
        d = _paired_record_dict()
        d["labels"] = {"rl": "high"}
        with pytest.raises(SchemaError, match="label value must be numeric"):
            UTREditRecord.from_dict(d)

    def test_nested_edit_script_form(self):
        """UTREditRecord.from_dict also accepts the nested EditScript form."""
        d = _paired_record_dict()
        d["edit_script"] = {
            "ops": d["edit_script"],
            "verified": True,
            "edit_distance": 1,
            "n_ins": 0,
            "n_del": 0,
            "n_sub": 1,
            "path_ambiguity": 1,
        }
        rec = UTREditRecord.from_dict(d)
        assert rec.edit_script.n_sub == 1
        assert rec.is_paired

    def test_frozen(self):
        rec = UTREditRecord.from_dict(_paired_record_dict())
        with pytest.raises(Exception):
            rec.record_id = "other"  # type: ignore

    def test_validate_utr_edit_record_helper(self):
        rec = validate_utr_edit_record(_paired_record_dict())
        assert rec.is_paired

    def test_incomplete_record_accepted(self):
        """GSE173083-style incomplete records must be accepted."""
        d = {
            "record_id": "GSE173083_INCOMPLETE",
            "dataset": "gse173083",
            "accession": "GSE173083",
            "region": "full-length",
            "source_sequence": None,
            "candidate_sequence": None,
            "edit_script": [],
            "edit_script_verified": True,
            "edit_distance": 0,
            "n_ins": 0, "n_del": 0, "n_sub": 0,
            "path_ambiguity": 1,
            "labels": {},
            "metadata": {
                "record_type": "incomplete",
                "data_role": "D_A",
                "note": "Only .rdat files in GEO download",
            },
        }
        rec = UTREditRecord.from_dict(d)
        assert rec.is_incomplete
        assert rec.candidate_sequence is None
        # round-trip
        d2 = rec.to_dict()
        rec2 = UTREditRecord.from_dict(d2)
        assert rec2.is_incomplete


# ---------------------------------------------------------------------------
# GenerationTask
# ---------------------------------------------------------------------------


class TestGenerationTask:
    def _make_task(self, **overrides):
        defaults = dict(
            task_id="t_001",
            record_id="GSE114002_test_0",
            track="closed_measured_pool",
            region="5'UTR",
            edit_budget=1,
            target_endpoint="rl",
            target_direction="increase",
            target_quantile=0.9,
            max_length=100,
            min_length=10,
            must_preserve_motifs=("ACGT",),
            allowed_claims=("edit_effect", "generation_grounding"),
            forbidden_claims=("improves_therapeutic_efficacy",),
            notes="test",
        )
        defaults.update(overrides)
        return GenerationTask(**defaults)

    def test_basic_construct(self):
        t = self._make_task()
        assert t.task_id == "t_001"
        assert t.track == "closed_measured_pool"

    def test_frozen(self):
        t = self._make_task()
        with pytest.raises(Exception):
            t.task_id = "other"  # type: ignore

    def test_invalid_track_rejected(self):
        with pytest.raises(SchemaError, match="track must be one of"):
            self._make_task(track="bogus")

    def test_cds_region_rejected(self):
        with pytest.raises(SchemaError, match="region must be one of"):
            self._make_task(region="CDS")

    def test_full_length_region_rejected(self):
        with pytest.raises(SchemaError, match="region must be one of"):
            self._make_task(region="full-length")

    def test_invalid_edit_budget_rejected(self):
        with pytest.raises(SchemaError, match="edit_budget must be in"):
            self._make_task(edit_budget=2)

    def test_zero_edit_budget_allowed(self):
        t = self._make_task(edit_budget=0)
        assert t.edit_budget == 0

    def test_negative_edit_budget_rejected(self):
        with pytest.raises(SchemaError):
            self._make_task(edit_budget=-1)

    def test_direction_without_endpoint_rejected(self):
        with pytest.raises(SchemaError, match="target_direction requires target_endpoint"):
            self._make_task(target_endpoint=None, target_direction="increase")

    def test_invalid_direction_rejected(self):
        with pytest.raises(SchemaError, match="target_direction must be one of"):
            self._make_task(target_direction="bogus")

    def test_quantile_out_of_range_rejected(self):
        with pytest.raises(SchemaError, match="target_quantile must be in"):
            self._make_task(target_quantile=1.5)

    def test_quantile_zero_rejected(self):
        with pytest.raises(SchemaError, match="target_quantile must be in"):
            self._make_task(target_quantile=0.0)

    def test_max_lt_min_rejected(self):
        with pytest.raises(SchemaError, match="max_length .* < min_length"):
            self._make_task(max_length=5, min_length=10)

    def test_invalid_motif_rejected(self):
        with pytest.raises(SchemaError, match="motif"):
            self._make_task(must_preserve_motifs=("ACNX",))

    def test_empty_motif_rejected(self):
        with pytest.raises(SchemaError, match="must_preserve_motifs entries"):
            self._make_task(must_preserve_motifs=("",))

    def test_invalid_allowed_claim_rejected(self):
        with pytest.raises(SchemaError, match="allowed_claims entry"):
            self._make_task(allowed_claims=("bogus_claim",))

    def test_to_from_dict_roundtrip(self):
        t = self._make_task()
        d = t.to_dict()
        t2 = GenerationTask.from_dict(d)
        assert t == t2

    def test_validate_generation_task_helper(self):
        t = validate_generation_task(self._make_task().to_dict())
        assert t.track in EVAL_TRACKS

    def test_all_three_tracks_constructible(self):
        for track in EVAL_TRACKS:
            t = self._make_task(track=track)
            assert t.track == track


# ---------------------------------------------------------------------------
# validate_canonical_records_file (uses a temp jsonl)
# ---------------------------------------------------------------------------


class TestValidateCanonicalRecordsFile:
    def test_valid_file_passes(self, tmp_path):
        path = tmp_path / "recs.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps(_paired_record_dict()) + "\n")
            f.write(json.dumps(_observational_record_dict()) + "\n")
            # also an incomplete record
            f.write(json.dumps({
                "record_id": "GSE173083_INCOMPLETE",
                "dataset": "gse173083",
                "accession": "GSE173083",
                "region": "full_length",  # underscore alias
                "source_sequence": None,
                "candidate_sequence": None,
                "edit_script": [],
                "edit_script_verified": True,
                "edit_distance": 0,
                "n_ins": 0, "n_del": 0, "n_sub": 0,
                "path_ambiguity": 1,
                "labels": {},
                "metadata": {"record_type": "incomplete", "data_role": "D_A"},
            }) + "\n")
        result = validate_canonical_records_file(str(path))
        assert result["passed"] is True
        assert result["n_total"] == 3
        assert result["n_paired"] == 1
        assert result["n_observational"] == 1
        assert result["n_incomplete"] == 1
        assert result["n_invalid"] == 0
        assert result["by_accession"].get("GSE114002") == 1
        assert result["by_accession"].get("GSE207584") == 1
        assert result["by_accession"].get("GSE173083") == 1
        # full_length alias should be normalized to full-length
        assert result["by_region"].get("full-length") == 1
        assert "full_length" not in result["by_region"]

    def test_invalid_record_reported(self, tmp_path):
        path = tmp_path / "recs.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps(_paired_record_dict()) + "\n")
            f.write(json.dumps({"bad": "row"}) + "\n")
        result = validate_canonical_records_file(str(path))
        assert result["passed"] is False
        assert result["n_invalid"] == 1
        assert result["n_total"] == 1
        assert len(result["issues"]) == 1
        assert "line 2" in result["issues"][0]

    def test_empty_lines_skipped(self, tmp_path):
        path = tmp_path / "recs.jsonl"
        with open(path, "w") as f:
            f.write("\n")
            f.write(json.dumps(_paired_record_dict()) + "\n")
            f.write("\n")
        result = validate_canonical_records_file(str(path))
        assert result["passed"] is True
        assert result["n_total"] == 1

    def test_max_records_cap(self, tmp_path):
        path = tmp_path / "recs.jsonl"
        with open(path, "w") as f:
            for _ in range(10):
                f.write(json.dumps(_paired_record_dict()) + "\n")
        result = validate_canonical_records_file(str(path), max_records=3)
        assert result["n_total"] == 3


# ---------------------------------------------------------------------------
# Contract constants sanity
# ---------------------------------------------------------------------------


class TestContractConstants:
    def test_alphabet_is_acgt(self):
        assert ALPHABET == ("A", "C", "G", "T")

    def test_op_values_complete(self):
        assert set(OP_VALUES) == {"INS", "DEL", "SUB", "STOP"}

    def test_eval_tracks_match_contract(self):
        assert set(EVAL_TRACKS) == {
            "closed_measured_pool",
            "heldout_generative",
            "open_legal_generation",
        }

    def test_record_types(self):
        assert set(RECORD_TYPES) == {"paired", "observational", "incomplete"}

    def test_region_values(self):
        assert "5'UTR" in REGION_VALUES
        assert "3'UTR" in REGION_VALUES
        assert "CDS" in ALL_REGION_VALUES
        assert "full-length" in ALL_REGION_VALUES

    def test_forbidden_sealed_wording_present(self):
        """GSE246381 forbidden wording from v2 contract §4.2 must be enforced."""
        for w in ("sealed", "untouched", "never-seen_external_test"):
            assert w in FORBIDDEN_SEALED_WORDING

    def test_allowed_claims_pool(self):
        # at minimum, the two primary D_C claims must be present
        assert "edit_effect" in ALLOWED_CLAIMS_POOL
        assert "generation_grounding" in ALLOWED_CLAIMS_POOL
