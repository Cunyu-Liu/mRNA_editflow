"""Unit tests for B0-02 split manifests (build + audit).

B0-02 acceptance: unexplained overlap = 0, reverse/path leakage = 0.

Covers:
  Build (build_split_manifests):
    - split_clusters keeps all records in a cluster in the same split
    - split_clusters produces no cluster overlap across train/val/test
    - split_clusters deterministic given seed
    - assign_by_accession assigns by accession + raises on unknown
    - write_manifest writes correct JSONL + summary (counts, sha256, excluded)
    - write_manifest excludes non-train/val/test splits (e.g. train_val_unused)
    - load_paired_records filters to paired only
  Audit (audit_split_manifests):
    - audit_manifest_format: valid / missing field / bad split / bad split_type
    - audit_source_overlap: clean vs leaky (same source in train+test)
    - audit_accession_overlap: clean vs leaky
    - audit_reverse_leakage: candidate(train) == source(test) detected
    - audit_path_leakage: train intermediate == test candidate detected
    - compute_intermediate_states: correct prefix states
    - audit_one_split overall pass/fail + acceptance aggregation
    - integration: a clean synthetic split passes end-to-end

Run: pytest d1_staging/tests/test_b0_split_manifests.py -v
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
    EditOp,
    EditScript,
    SchemaError,
    UTREditRecord,
)
from build_split_manifests import (  # noqa: E402
    assign_by_accession,
    load_paired_records,
    split_clusters,
    write_manifest,
)
from audit_split_manifests import (  # noqa: E402
    audit_accession_overlap,
    audit_manifest_format,
    audit_one_split,
    audit_path_leakage,
    audit_reverse_leakage,
    audit_source_overlap,
    compute_intermediate_states,
    load_manifest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_edit_script(ops_list):
    """Build an EditScript from a list of (op, pos, token) tuples."""
    ops = tuple(EditOp(op=o, pos=p, token=t) for o, p, t in ops_list)
    counts = {"INS": 0, "DEL": 0, "SUB": 0}
    for op in ops:
        counts[op.op] += 1
    return EditScript(
        ops=ops,
        verified=True,
        edit_distance=len(ops),
        n_ins=counts["INS"],
        n_del=counts["DEL"],
        n_sub=counts["SUB"],
        path_ambiguity=1,
    )


def make_paired_record(
    record_id, accession, region, source, candidate, ops_list=None, dataset="test"
):
    """Build a paired UTREditRecord. If ops_list is None, infer a single SUB."""
    if ops_list is None:
        # find first differing position for a single SUB
        diff = [i for i in range(min(len(source), len(candidate)))
                if source[i] != candidate[i]]
        if diff:
            ops_list = [("SUB", diff[0], candidate[diff[0]])]
        else:
            ops_list = []
    es = make_edit_script(ops_list)
    return UTREditRecord(
        record_id=record_id,
        dataset=dataset,
        accession=accession,
        region=region,
        source_sequence=source,
        candidate_sequence=candidate,
        edit_script=es,
        labels={"rl": 1.0},
        metadata={"record_type": "paired"},
    )


def make_observational_record(record_id, accession, region, sequence, dataset="test"):
    """Build an observational UTREditRecord (source=None, no edits)."""
    es = EditScript(ops=(), verified=True, edit_distance=0, n_ins=0, n_del=0,
                    n_sub=0, path_ambiguity=1)
    return UTREditRecord(
        record_id=record_id,
        dataset=dataset,
        accession=accession,
        region=region,
        source_sequence=None,
        candidate_sequence=sequence,
        edit_script=es,
        labels={},
        metadata={"record_type": "observational"},
    )


def write_records_jsonl(records, path):
    """Write records as canonical_records.jsonl format."""
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict()) + "\n")


# ---------------------------------------------------------------------------
# Build: split_clusters
# ---------------------------------------------------------------------------

class TestSplitClusters:
    def test_no_cluster_overlap_across_splits(self):
        # 10 clusters, each with 2 records
        r2c = {}
        for c in range(10):
            for r in range(2):
                r2c[f"rec_{c}_{r}"] = f"cluster_{c}"
        assignments = split_clusters(r2c, train_frac=0.6, val_frac=0.2, seed=42)
        # group clusters by split
        from collections import defaultdict
        split_clusters_map = defaultdict(set)
        for rid, split in assignments.items():
            split_clusters_map[split].add(r2c[rid])
        # no cluster in more than one split
        all_clusters = set()
        for s, clusters in split_clusters_map.items():
            assert not (all_clusters & clusters), f"cluster overlap in {s}"
            all_clusters |= clusters
        assert len(all_clusters) == 10

    def test_all_records_assigned(self):
        r2c = {f"rec_{i}": f"cluster_{i % 3}" for i in range(9)}
        assignments = split_clusters(r2c, train_frac=0.6, val_frac=0.2, seed=1)
        assert set(assignments.keys()) == set(r2c.keys())
        assert all(v in ("train", "val", "test") for v in assignments.values())

    def test_deterministic_given_seed(self):
        r2c = {f"rec_{i}": f"cluster_{i}" for i in range(20)}
        a1 = split_clusters(r2c, seed=123)
        a2 = split_clusters(r2c, seed=123)
        assert a1 == a2

    def test_different_seeds_may_differ(self):
        r2c = {f"rec_{i}": f"cluster_{i}" for i in range(20)}
        a1 = split_clusters(r2c, seed=1)
        a2 = split_clusters(r2c, seed=999)
        # not guaranteed to differ, but extremely likely with 20 clusters
        assert a1 != a2

    def test_single_cluster_all_same_split(self):
        # all records in one cluster -> all in same split
        r2c = {f"rec_{i}": "only_cluster" for i in range(5)}
        assignments = split_clusters(r2c, seed=7)
        splits = set(assignments.values())
        assert len(splits) == 1


# ---------------------------------------------------------------------------
# Build: assign_by_accession
# ---------------------------------------------------------------------------

class TestAssignByAccession:
    def test_assigns_correctly(self):
        recs = [
            make_paired_record("r1", "GSE_A", "5'UTR", "ACGT", "ACGT"),
            make_paired_record("r2", "GSE_B", "3'UTR", "ACGT", "ACGT"),
            make_paired_record("r3", "GSE_C", "5'UTR", "ACGT", "ACGT"),
        ]
        a = assign_by_accession(
            recs,
            train_accessions={"GSE_A"},
            val_accessions={"GSE_C"},
            test_accessions={"GSE_B"},
        )
        assert a == {"r1": "train", "r2": "test", "r3": "val"}

    def test_raises_on_unknown_accession(self):
        recs = [make_paired_record("r1", "GSE_X", "5'UTR", "ACGT", "ACGT")]
        with pytest.raises(ValueError, match="not in any split set"):
            assign_by_accession(recs, {"GSE_A"}, set(), {"GSE_B"})


# ---------------------------------------------------------------------------
# Build: write_manifest
# ---------------------------------------------------------------------------

class TestWriteManifest:
    def test_writes_correct_counts_and_format(self, tmp_path):
        recs = [
            make_paired_record("r1", "GSE_A", "5'UTR", "ACGTACGT", "ACGTACGT"),
            make_paired_record("r2", "GSE_A", "5'UTR", "TTTTCCCC", "TTTTCCCC"),
            make_paired_record("r3", "GSE_A", "5'UTR", "GGGGAAAA", "GGGGAAAA"),
        ]
        assignments = {"r1": "train", "r2": "val", "r3": "test"}
        out = str(tmp_path / "manifest.jsonl")
        summary = write_manifest("test_split", recs, assignments, out)
        lines = load_manifest(out)
        assert len(lines) == 3
        assert summary["n_train"] == 1
        assert summary["n_val"] == 1
        assert summary["n_test"] == 1
        assert summary["n_total"] == 3
        assert summary["n_excluded"] == 0
        assert summary["split_type"] == "test_split"
        # each line has required fields
        for e in lines:
            assert e["split"] in ("train", "val", "test")
            assert e["split_type"] == "test_split"

    def test_excludes_non_standard_splits(self, tmp_path):
        recs = [
            make_paired_record("r1", "GSE_A", "5'UTR", "ACGTACGT", "ACGTACGT"),
            make_paired_record("r2", "GSE_A", "5'UTR", "TTTTCCCC", "TTTTCCCC"),
        ]
        assignments = {"r1": "train", "r2": "train_val_unused"}
        out = str(tmp_path / "manifest.jsonl")
        summary = write_manifest("crt", recs, assignments, out)
        assert summary["n_train"] == 1
        assert summary["n_excluded"] == 1
        lines = load_manifest(out)
        assert len(lines) == 1

    def test_sha256_computed(self, tmp_path):
        recs = [make_paired_record("r1", "GSE_A", "5'UTR", "ACGTACGT", "ACGTACGT")]
        out = str(tmp_path / "manifest.jsonl")
        summary = write_manifest("s", recs, {"r1": "train"}, out)
        assert len(summary["sha256"]) == 64
        import hashlib
        with open(out, "rb") as f:
            expected = hashlib.sha256(f.read()).hexdigest()
        assert summary["sha256"] == expected


# ---------------------------------------------------------------------------
# Build: load_paired_records
# ---------------------------------------------------------------------------

class TestLoadPairedRecords:
    def test_filters_to_paired_only(self, tmp_path):
        paired = make_paired_record("p1", "GSE_A", "5'UTR", "ACGT", "ACGT")
        obs = make_observational_record("o1", "GSE_A", "5'UTR", "ACGT")
        path = str(tmp_path / "records.jsonl")
        write_records_jsonl([paired, obs], path)
        loaded = load_paired_records(path)
        assert len(loaded) == 1
        assert loaded[0].record_id == "p1"


# ---------------------------------------------------------------------------
# Audit: manifest_format
# ---------------------------------------------------------------------------

class TestAuditManifestFormat:
    def test_valid_entries_pass(self):
        entries = [
            {"record_id": "r1", "accession": "GSE_A", "region": "5'UTR",
             "split": "train", "split_type": "5utr_source_disjoint"},
            {"record_id": "r2", "accession": "GSE_A", "region": "5'UTR",
             "split": "test", "split_type": "5utr_source_disjoint"},
        ]
        res = audit_manifest_format(entries, "5utr_source_disjoint")
        assert res["pass"]
        assert res["n_errors"] == 0

    def test_missing_field_fails(self):
        entries = [{"record_id": "r1", "accession": "GSE_A", "region": "5'UTR",
                    "split": "train"}]  # missing split_type
        res = audit_manifest_format(entries, "5utr_source_disjoint")
        assert not res["pass"]
        assert res["n_errors"] >= 1

    def test_bad_split_value_fails(self):
        entries = [{"record_id": "r1", "accession": "GSE_A", "region": "5'UTR",
                    "split": "train_val_unused", "split_type": "5utr_source_disjoint"}]
        res = audit_manifest_format(entries, "5utr_source_disjoint")
        assert not res["pass"]

    def test_wrong_split_type_fails(self):
        entries = [{"record_id": "r1", "accession": "GSE_A", "region": "5'UTR",
                    "split": "train", "split_type": "study_disjoint"}]
        res = audit_manifest_format(entries, "5utr_source_disjoint")
        assert not res["pass"]


# ---------------------------------------------------------------------------
# Audit: source_overlap
# ---------------------------------------------------------------------------

class TestAuditSourceOverlap:
    def test_clean_pass(self):
        recs = {
            "r1": make_paired_record("r1", "GSE_A", "5'UTR", "AAAACCCC", "AAAACCCC"),
            "r2": make_paired_record("r2", "GSE_A", "5'UTR", "TTTTGGGG", "TTTTGGGG"),
        }
        entries = [
            {"record_id": "r1", "split": "train"},
            {"record_id": "r2", "split": "test"},
        ]
        res = audit_source_overlap(entries, recs)
        assert res["pass"]
        assert res["n_overlap"] == 0

    def test_leaky_fails(self):
        recs = {
            "r1": make_paired_record("r1", "GSE_A", "5'UTR", "AAAACCCC", "AAAACCCC"),
            "r2": make_paired_record("r2", "GSE_A", "5'UTR", "AAAACCCC", "AAAACCCC"),
        }
        entries = [
            {"record_id": "r1", "split": "train"},
            {"record_id": "r2", "split": "test"},
        ]
        res = audit_source_overlap(entries, recs)
        assert not res["pass"]
        assert res["n_overlap"] == 1


# ---------------------------------------------------------------------------
# Audit: accession_overlap
# ---------------------------------------------------------------------------

class TestAuditAccessionOverlap:
    def test_clean_pass(self):
        entries = [
            {"accession": "GSE_A", "split": "train"},
            {"accession": "GSE_B", "split": "test"},
        ]
        res = audit_accession_overlap(entries)
        assert res["pass"]
        assert res["n_overlap"] == 0

    def test_leaky_fails(self):
        entries = [
            {"accession": "GSE_A", "split": "train"},
            {"accession": "GSE_A", "split": "test"},
        ]
        res = audit_accession_overlap(entries)
        assert not res["pass"]
        assert res["n_overlap"] == 1
        assert "GSE_A" in res["overlap"]


# ---------------------------------------------------------------------------
# Audit: reverse_leakage
# ---------------------------------------------------------------------------

class TestAuditReverseLeakage:
    def test_clean_pass(self):
        recs = {
            "r1": make_paired_record("r1", "GSE_A", "5'UTR", "AAAACCCC", "TTTTGGGG"),
            "r2": make_paired_record("r2", "GSE_A", "5'UTR", "CCCCGGGG", "AAAATTTT"),
        }
        entries = [{"record_id": "r1", "split": "train"},
                   {"record_id": "r2", "split": "test"}]
        res = audit_reverse_leakage(entries, recs)
        assert res["pass"]

    def test_candidate_train_equals_source_test_fails(self):
        # train candidate == test source -> reverse leakage
        recs = {
            "r1": make_paired_record("r1", "GSE_A", "5'UTR", "AAAACCCC", "TTTTGGGG"),
            "r2": make_paired_record("r2", "GSE_A", "5'UTR", "TTTTGGGG", "CCCCAAAA"),
        }
        entries = [{"record_id": "r1", "split": "train"},
                   {"record_id": "r2", "split": "test"}]
        res = audit_reverse_leakage(entries, recs)
        assert not res["pass"]
        assert res["n_reverse_leakage"] == 1

    def test_source_train_equals_candidate_test_fails(self):
        # symmetric channel: source(train) == candidate(test)
        recs = {
            "r1": make_paired_record("r1", "GSE_A", "5'UTR", "TTTTGGGG", "CCCCAAAA"),
            "r2": make_paired_record("r2", "GSE_A", "5'UTR", "AAAACCCC", "TTTTGGGG"),
        }
        entries = [{"record_id": "r1", "split": "train"},
                   {"record_id": "r2", "split": "test"}]
        res = audit_reverse_leakage(entries, recs)
        assert not res["pass"]
        assert res["n_symmetric_leakage"] == 1


# ---------------------------------------------------------------------------
# Audit: compute_intermediate_states + path_leakage
# ---------------------------------------------------------------------------

class TestComputeIntermediateStates:
    def test_no_intermediates_for_empty_script(self):
        rec = make_paired_record("r1", "GSE_A", "5'UTR", "ACGTACGT", "ACGTACGT",
                                 ops_list=[])
        assert compute_intermediate_states(rec) == []

    def test_single_op_has_no_intermediate(self):
        # one SUB -> only the final candidate (no prefix of length < len(ops))
        rec = make_paired_record("r1", "GSE_A", "5'UTR", "ACGTACGT", "TCGTACGT")
        states = compute_intermediate_states(rec)
        assert states == []  # only 1 op, no intermediate

    def test_two_ops_has_one_intermediate(self):
        source = "ACGTACGT"
        # op1: SUB pos0 A->T, op2: SUB pos1 C->G => candidate TGGTACGT
        rec = make_paired_record("r1", "GSE_A", "5'UTR", source, "TGGTACGT",
                                 ops_list=[("SUB", 0, "T"), ("SUB", 1, "G")])
        states = compute_intermediate_states(rec)
        assert len(states) == 1
        # after op1: TCGTACGT
        assert states[0] == "TCGTACGT"

    def test_three_ops_has_two_intermediates(self):
        source = "AAAA"
        rec = make_paired_record("r1", "GSE_A", "5'UTR", source, "TTTA",
                                 ops_list=[("SUB", 0, "T"), ("SUB", 1, "T"),
                                           ("SUB", 2, "T")])
        states = compute_intermediate_states(rec)
        assert len(states) == 2
        assert states[0] == "TAAA"
        assert states[1] == "TTAA"


class TestAuditPathLeakage:
    def test_clean_pass(self):
        recs = {
            "r1": make_paired_record("r1", "GSE_A", "5'UTR", "AAAA", "TTTT",
                                     ops_list=[("SUB", 0, "T"), ("SUB", 1, "T"),
                                               ("SUB", 2, "T"), ("SUB", 3, "T")]),
            "r2": make_paired_record("r2", "GSE_A", "5'UTR", "CCCC", "GGGG",
                                     ops_list=[("SUB", 0, "G"), ("SUB", 1, "G"),
                                               ("SUB", 2, "G"), ("SUB", 3, "G")]),
        }
        entries = [{"record_id": "r1", "split": "train"},
                   {"record_id": "r2", "split": "test"}]
        res = audit_path_leakage(entries, recs)
        assert res["pass"]

    def test_train_intermediate_equals_test_candidate_fails(self):
        # train: AAAA -> (SUB0 T) -> TAAA -> (SUB1 T) -> TTAA  (intermediate=TAAA)
        # test candidate = TAAA  => forward path leakage
        recs = {
            "r1": make_paired_record("r1", "GSE_A", "5'UTR", "AAAA", "TTAA",
                                     ops_list=[("SUB", 0, "T"), ("SUB", 1, "T")]),
            "r2": make_paired_record("r2", "GSE_A", "5'UTR", "CCCC", "TAAA",
                                     ops_list=[("SUB", 0, "T"), ("SUB", 1, "A"),
                                               ("SUB", 2, "A")]),
        }
        entries = [{"record_id": "r1", "split": "train"},
                   {"record_id": "r2", "split": "test"}]
        res = audit_path_leakage(entries, recs)
        assert not res["pass"]
        assert res["n_forward_path_leakage"] == 1

    def test_test_intermediate_equals_train_candidate_fails(self):
        # reverse path: test intermediate == train candidate
        recs = {
            "r1": make_paired_record("r1", "GSE_A", "5'UTR", "CCCC", "TAAA",
                                     ops_list=[("SUB", 0, "T"), ("SUB", 1, "A"),
                                               ("SUB", 2, "A")]),
            "r2": make_paired_record("r2", "GSE_A", "5'UTR", "AAAA", "TTAA",
                                     ops_list=[("SUB", 0, "T"), ("SUB", 1, "T")]),
        }
        entries = [{"record_id": "r1", "split": "train"},
                   {"record_id": "r2", "split": "test"}]
        res = audit_path_leakage(entries, recs)
        # train candidate = TAAA; test intermediate (after SUB0) = TAAA
        assert not res["pass"]
        assert res["n_reverse_path_leakage"] == 1


# ---------------------------------------------------------------------------
# Audit: audit_one_split integration
# ---------------------------------------------------------------------------

class TestAuditOneSplit:
    def _write_clean_manifest(self, tmp_path, split_type):
        """Write a clean 5utr_source_disjoint manifest + records."""
        recs = [
            make_paired_record("r1", "GSE_A", "5'UTR", "AAAACCCC", "AAAATCCC",
                               ops_list=[("SUB", 4, "T")]),
            make_paired_record("r2", "GSE_A", "5'UTR", "TTTTGGGG", "TTTCGGGG",
                               ops_list=[("SUB", 3, "C")]),
            make_paired_record("r3", "GSE_A", "5'UTR", "CCCCAAAA", "CCCGAAAA",
                               ops_list=[("SUB", 3, "G")]),
        ]
        records_by_id = {r.record_id: r for r in recs}
        manifest_path = str(tmp_path / "manifest.jsonl")
        entries = [
            {"record_id": "r1", "accession": "GSE_A", "region": "5'UTR",
             "split": "train", "split_type": split_type},
            {"record_id": "r2", "accession": "GSE_A", "region": "5'UTR",
             "split": "val", "split_type": split_type},
            {"record_id": "r3", "accession": "GSE_A", "region": "5'UTR",
             "split": "test", "split_type": split_type},
        ]
        with open(manifest_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        return manifest_path, records_by_id

    def test_clean_source_disjoint_passes(self, tmp_path):
        manifest_path, recs = self._write_clean_manifest(tmp_path, "5utr_source_disjoint")
        res = audit_one_split(manifest_path, "5utr_source_disjoint", recs)
        assert res["pass"]
        assert res["acceptance"]["unexplained_overlap"] == 0
        assert res["acceptance"]["reverse_leakage"] == 0
        assert res["acceptance"]["path_leakage"] == 0
        # source_overlap is applicable
        assert res["checks"]["source_overlap"] is not None
        # accession_overlap is NOT applicable for source_disjoint
        assert res["checks"]["accession_overlap"] is None

    def test_leaky_source_disjoint_fails(self, tmp_path):
        manifest_path, recs = self._write_clean_manifest(tmp_path, "5utr_source_disjoint")
        # make r1 and r3 share a source (leak)
        recs["r1"] = make_paired_record("r1", "GSE_A", "5'UTR", "GGGGCCCC", "GGGGCCCC")
        recs["r3"] = make_paired_record("r3", "GSE_A", "5'UTR", "GGGGCCCC", "GGGGCCCC")
        res = audit_one_split(manifest_path, "5utr_source_disjoint", recs)
        assert not res["pass"]
        assert res["acceptance"]["unexplained_overlap"] >= 1

    def test_study_disjoint_checks_accession(self, tmp_path):
        # study_disjoint: accession_overlap applies, source_overlap does not
        recs = {
            "r1": make_paired_record("r1", "GSE_A", "5'UTR", "AAAACCCC", "AAAACCCC"),
            "r2": make_paired_record("r2", "GSE_B", "3'UTR", "TTTTGGGG", "TTTTGGGG"),
        }
        manifest_path = str(tmp_path / "manifest.jsonl")
        entries = [
            {"record_id": "r1", "accession": "GSE_A", "region": "5'UTR",
             "split": "train", "split_type": "study_disjoint"},
            {"record_id": "r2", "accession": "GSE_B", "region": "3'UTR",
             "split": "test", "split_type": "study_disjoint"},
        ]
        with open(manifest_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        res = audit_one_split(manifest_path, "study_disjoint", recs)
        assert res["pass"]
        assert res["checks"]["accession_overlap"] is not None
        assert res["checks"]["source_overlap"] is None

    def test_sha256_in_result(self, tmp_path):
        manifest_path, recs = self._write_clean_manifest(tmp_path, "5utr_source_disjoint")
        res = audit_one_split(manifest_path, "5utr_source_disjoint", recs)
        assert len(res["sha256"]) == 64
