"""Unit tests for B0-03 leakage audit.

B0-03 acceptance:
  - final endpoint as train intermediate = 0
  - exposure ledger coverage = 100%

Covers:
  - extract_scaffold: NC_ / chr / none patterns
  - compute_intermediate_states: prefix states
  - check_exact_overlap: clean vs leaky
  - check_reverse_overlap: clean vs leaky
  - check_intermediate_overlap: endpoint-as-train-intermediate detection
  - check_cluster_overlap: proxy detection
  - check_scaffold_overlap: informational, always pass
  - check_study_overlap: clean vs leaky
  - check_gene/context/barcode: N/A stubs; foundation is pending without an
    FM0 manifest and audited at source level when one is supplied
  - check_exposure_ledger_coverage: 100% vs <100%
  - audit_split_leakage: integration pass/fail

Run: pytest d1_staging/tests/test_b0_leakage_audit.py -v
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

from canonical_schemas import EditOp, EditScript, UTREditRecord  # noqa: E402
from leakage_audit import (  # noqa: E402
    audit_split_leakage,
    check_barcode_overlap,
    check_cluster_overlap,
    check_context_overlap,
    check_exact_overlap,
    check_exposure_ledger_coverage,
    check_foundation_overlap,
    check_gene_overlap,
    check_intermediate_overlap,
    check_reverse_overlap,
    check_scaffold_overlap,
    check_study_overlap,
    compute_intermediate_states,
    extract_scaffold,
    load_exposure_ledger_ids,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_edit_script(ops_list):
    ops = tuple(EditOp(op=o, pos=p, token=t) for o, p, t in ops_list)
    counts = {"INS": 0, "DEL": 0, "SUB": 0}
    for op in ops:
        counts[op.op] += 1
    return EditScript(
        ops=ops, verified=True, edit_distance=len(ops),
        n_ins=counts["INS"], n_del=counts["DEL"], n_sub=counts["SUB"],
        path_ambiguity=1,
    )


def make_record(record_id, accession, region, source, candidate, ops_list=None):
    if ops_list is None:
        diff = [i for i in range(min(len(source), len(candidate)))
                if source[i] != candidate[i]]
        ops_list = [("SUB", diff[0], candidate[diff[0]])] if diff else []
    es = make_edit_script(ops_list)
    return UTREditRecord(
        record_id=record_id, dataset="test", accession=accession, region=region,
        source_sequence=source, candidate_sequence=candidate, edit_script=es,
        labels={"rl": 1.0}, metadata={"record_type": "paired"},
    )


def entry(rid, acc, split):
    return {"record_id": rid, "accession": acc, "region": "5'UTR",
            "split": split, "split_type": "test"}


# ---------------------------------------------------------------------------
# extract_scaffold
# ---------------------------------------------------------------------------

class TestExtractScaffold:
    def test_nc_pattern(self):
        assert extract_scaffold("GSE114002_NC_000012.12:g.4911352C>T_0") == "NC_000012.12"

    def test_chr_pattern(self):
        assert extract_scaffold("GSE200304_chr2:69461620_G-C") == "chr2"

    def test_no_scaffold(self):
        assert extract_scaffold("GSE114002_8527_1") is None

    def test_chrX(self):
        assert extract_scaffold("GSE_chrX:123_A-T") == "chrX"


# ---------------------------------------------------------------------------
# compute_intermediate_states
# ---------------------------------------------------------------------------

class TestComputeIntermediateStates:
    def test_empty_script(self):
        rec = make_record("r1", "A", "5'UTR", "AAAA", "AAAA", ops_list=[])
        assert compute_intermediate_states(rec) == []

    def test_two_ops_one_intermediate(self):
        rec = make_record("r1", "A", "5'UTR", "AAAA", "TTAA",
                          ops_list=[("SUB", 0, "T"), ("SUB", 1, "T")])
        states = compute_intermediate_states(rec)
        assert states == ["TAAA"]


# ---------------------------------------------------------------------------
# Exact overlap
# ---------------------------------------------------------------------------

class TestExactOverlap:
    def test_clean(self):
        recs = {
            "r1": make_record("r1", "A", "5'UTR", "AAAA", "TTTT"),
            "r2": make_record("r2", "A", "5'UTR", "CCCC", "GGGG"),
        }
        res = check_exact_overlap([entry("r1", "A", "train")],
                                  [entry("r2", "A", "test")], recs)
        assert res["pass"]
        assert res["n_overlap"] == 0

    def test_leaky_source(self):
        recs = {
            "r1": make_record("r1", "A", "5'UTR", "SHARED", "SHARED")
                  if False else make_record("r1", "A", "5'UTR", "AAAACCCC", "AAAACCCC"),
            "r2": make_record("r2", "A", "5'UTR", "AAAACCCC", "TTTTGGGG"),
        }
        res = check_exact_overlap([entry("r1", "A", "train")],
                                  [entry("r2", "A", "test")], recs)
        assert not res["pass"]
        assert res["n_overlap"] >= 1


# ---------------------------------------------------------------------------
# Reverse overlap
# ---------------------------------------------------------------------------

class TestReverseOverlap:
    def test_clean(self):
        recs = {
            "r1": make_record("r1", "A", "5'UTR", "AAAACCCC", "TTTTGGGG"),
            "r2": make_record("r2", "A", "5'UTR", "CCCCAAAA", "GGGGTTTT"),
        }
        res = check_reverse_overlap([entry("r1", "A", "train")],
                                    [entry("r2", "A", "test")], recs)
        assert res["pass"]

    def test_leaky(self):
        recs = {
            "r1": make_record("r1", "A", "5'UTR", "AAAACCCC", "TTTTGGGG"),
            "r2": make_record("r2", "A", "5'UTR", "TTTTGGGG", "CCCCAAAA"),
        }
        res = check_reverse_overlap([entry("r1", "A", "train")],
                                    [entry("r2", "A", "test")], recs)
        assert not res["pass"]


# ---------------------------------------------------------------------------
# Intermediate overlap (endpoint-as-train-intermediate)
# ---------------------------------------------------------------------------

class TestIntermediateOverlap:
    def test_clean(self):
        recs = {
            "r1": make_record("r1", "A", "5'UTR", "AAAA", "TTTT",
                              ops_list=[("SUB", 0, "T"), ("SUB", 1, "T"),
                                        ("SUB", 2, "T"), ("SUB", 3, "T")]),
            "r2": make_record("r2", "A", "5'UTR", "CCCC", "GGGG",
                              ops_list=[("SUB", 0, "G"), ("SUB", 1, "G"),
                                        ("SUB", 2, "G"), ("SUB", 3, "G")]),
        }
        res = check_intermediate_overlap([entry("r1", "A", "train")],
                                         [entry("r2", "A", "test")], recs)
        assert res["pass"]
        assert res["n_endpoint_as_train_intermediate"] == 0

    def test_endpoint_as_train_intermediate(self):
        # train: AAAA -> TAAA (intermediate) -> TTAA (candidate)
        # test candidate = TAAA => endpoint as train intermediate
        recs = {
            "r1": make_record("r1", "A", "5'UTR", "AAAA", "TTAA",
                              ops_list=[("SUB", 0, "T"), ("SUB", 1, "T")]),
            "r2": make_record("r2", "A", "5'UTR", "CCCC", "TAAA",
                              ops_list=[("SUB", 0, "T"), ("SUB", 1, "A"),
                                        ("SUB", 2, "A")]),
        }
        res = check_intermediate_overlap([entry("r1", "A", "train")],
                                         [entry("r2", "A", "test")], recs)
        assert not res["pass"]
        assert res["n_endpoint_as_train_intermediate"] == 1


# ---------------------------------------------------------------------------
# Cluster overlap (proxy)
# ---------------------------------------------------------------------------

class TestClusterOverlap:
    def test_clean(self):
        recs = {
            "r1": make_record("r1", "A", "5'UTR", "AAAACCCC", "AAAACCCC"),
            "r2": make_record("r2", "A", "5'UTR", "TTTTGGGG", "TTTTGGGG"),
        }
        res = check_cluster_overlap([entry("r1", "A", "train")],
                                    [entry("r2", "A", "test")], recs)
        assert res["pass"]

    def test_leaky(self):
        recs = {
            "r1": make_record("r1", "A", "5'UTR", "AAAACCCC", "AAAACCCC"),
            "r2": make_record("r2", "A", "5'UTR", "AAAACCCC", "TTTTGGGG"),
        }
        res = check_cluster_overlap([entry("r1", "A", "train")],
                                    [entry("r2", "A", "test")], recs)
        assert not res["pass"]


# ---------------------------------------------------------------------------
# Scaffold overlap
# ---------------------------------------------------------------------------

class TestScaffoldOverlap:
    def test_informational_always_pass(self):
        train = [entry("GSE114002_NC_000012.12:g.1C>T_0", "GSE114002", "train")]
        test = [entry("GSE114002_NC_000012.12:g.2C>T_0", "GSE114002", "test")]
        res = check_scaffold_overlap(train, test)
        assert res["pass"]  # informational
        assert res["informational"]
        assert res["n_overlap"] == 1  # same scaffold, but not a hard leak

    def test_different_scaffolds(self):
        train = [entry("GSE114002_NC_000012.12:g.1C>T_0", "GSE114002", "train")]
        test = [entry("GSE200304_chr2:123_A-T", "GSE200304", "test")]
        res = check_scaffold_overlap(train, test)
        assert res["n_overlap"] == 0


# ---------------------------------------------------------------------------
# Study overlap
# ---------------------------------------------------------------------------

class TestStudyOverlap:
    def test_clean(self):
        res = check_study_overlap([entry("r1", "GSE_A", "train")],
                                  [entry("r2", "GSE_B", "test")])
        assert res["pass"]

    def test_leaky(self):
        res = check_study_overlap([entry("r1", "GSE_A", "train")],
                                  [entry("r2", "GSE_A", "test")])
        assert not res["pass"]
        assert "GSE_A" in res["overlap"]


# ---------------------------------------------------------------------------
# N/A channels
# ---------------------------------------------------------------------------

class TestNAChannels:
    def test_gene_not_applicable(self):
        res = check_gene_overlap([], [], {})
        assert not res["applicable"]
        assert res["pass"]

    def test_context_not_applicable(self):
        res = check_context_overlap([], [], {})
        assert not res["applicable"]
        assert res["pass"]

    def test_barcode_not_applicable(self):
        res = check_barcode_overlap([], [], {})
        assert not res["applicable"]
        assert res["pass"]

    def test_foundation_pending_fm0(self):
        res = check_foundation_overlap([], [], {})
        assert not res["applicable"]
        assert res["status"] == "PENDING_FM0"
        assert res["pass"]

    def test_foundation_manifest_classifies_source_exposure(self):
        records = {
            "r1": make_record(
                "r1", "GSE_A", "5'UTR", "AAAACCCC", "AAAATCCC",
                ops_list=[("SUB", 4, "T")],
            )
        }
        manifest = {
            "schema_version": "fm0-foundation-training-data/v1",
            "model_id": "multimolecule/utrlm-mrl",
            "revision": "r" * 40,
            "checkpoint_sha256": "a" * 64,
            "license": {"type": "agpl-3.0"},
            "corpus_sources": [{"source_id": "sample2019"}],
            "dataset_exposure": [{
                "accession": "GSE_A",
                "region": "5'UTR",
                "historically_exposed_to_model": True,
                "exposure_type": "sequence_prior_only",
                "evidence_grade": "E4",
                "labels_exposed_to_model": False,
            }],
            "exposure_assertions_complete": True,
            "exact_sequence_manifest_available": False,
        }
        res = check_foundation_overlap(
            [], [{"record_id": "r1"}], records, manifest
        )
        assert res["applicable"]
        assert res["status"] == "AUDITED_SOURCE_LEVEL_EXPOSURE"
        assert res["pass"]
        assert res["n_known_source_overlap"] == 1
        assert res["n_overlap"] is None
        assert res["exact_sequence_overlap_status"] == "NOT_AVAILABLE_NOT_ASSERTED"

    def test_foundation_manifest_missing_test_coverage_fails_closed(self):
        records = {
            "r1": make_record(
                "r1", "GSE_A", "5'UTR", "AAAACCCC", "AAAATCCC",
                ops_list=[("SUB", 4, "T")],
            )
        }
        manifest = {
            "schema_version": "fm0-foundation-training-data/v1",
            "model_id": "multimolecule/utrlm-mrl",
            "revision": "r" * 40,
            "checkpoint_sha256": "a" * 64,
            "license": {"type": "agpl-3.0"},
            "corpus_sources": [{"source_id": "sample2019"}],
            "dataset_exposure": [],
            "exposure_assertions_complete": True,
            "exact_sequence_manifest_available": False,
        }
        res = check_foundation_overlap(
            [], [{"record_id": "r1"}], records, manifest
        )
        assert res["status"] == "FOUNDATION_MANIFEST_INVALID"
        assert not res["pass"]


# ---------------------------------------------------------------------------
# Exposure ledger coverage
# ---------------------------------------------------------------------------

class TestExposureLedgerCoverage:
    def test_full_coverage(self):
        manifest_ids = {"r1", "r2", "r3"}
        ledger_ids = {"r1", "r2", "r3", "r4"}
        res = check_exposure_ledger_coverage(manifest_ids, ledger_ids)
        assert res["pass"]
        assert res["coverage"] == 1.0
        assert res["n_missing"] == 0

    def test_partial_coverage(self):
        manifest_ids = {"r1", "r2", "r3"}
        ledger_ids = {"r1", "r2"}
        res = check_exposure_ledger_coverage(manifest_ids, ledger_ids)
        assert not res["pass"]
        assert res["coverage"] < 1.0
        assert res["n_missing"] == 1
        assert "r3" in res["missing_examples"]

    def test_empty_manifest(self):
        res = check_exposure_ledger_coverage(set(), {"r1"})
        assert res["pass"]
        assert res["coverage"] == 1.0


class TestLoadExposureLedgerIds:
    def test_loads_ids(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        path.write_text(
            json.dumps({"record_id": "r1", "accession": "A"}) + "\n"
            + json.dumps({"record_id": "r2", "accession": "B"}) + "\n"
        )
        ids = load_exposure_ledger_ids(str(path))
        assert ids == {"r1", "r2"}


# ---------------------------------------------------------------------------
# Integration: audit_split_leakage
# ---------------------------------------------------------------------------

class TestAuditSplitLeakage:
    def _write_clean_manifest(self, tmp_path, split_type="5utr_source_disjoint"):
        recs = {
            "r1": make_record("r1", "GSE_A", "5'UTR", "AAAACCCC", "AAAATCCC",
                              ops_list=[("SUB", 4, "T")]),
            "r2": make_record("r2", "GSE_A", "5'UTR", "TTTTGGGG", "TTTCGGGG",
                              ops_list=[("SUB", 3, "C")]),
            "r3": make_record("r3", "GSE_A", "5'UTR", "CCCCAAAA", "CCCGAAAA",
                              ops_list=[("SUB", 3, "G")]),
        }
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
        return manifest_path, recs

    def test_clean_passes(self, tmp_path):
        manifest_path, recs = self._write_clean_manifest(tmp_path)
        res = audit_split_leakage("5utr_source_disjoint", manifest_path, recs)
        assert res["pass"]
        assert res["acceptance"]["endpoint_as_train_intermediate"] == 0

    def test_endpoint_leak_fails(self, tmp_path):
        manifest_path, recs = self._write_clean_manifest(tmp_path)
        # train intermediate == test candidate
        recs["r1"] = make_record("r1", "GSE_A", "5'UTR", "AAAA", "TTAA",
                                 ops_list=[("SUB", 0, "T"), ("SUB", 1, "T")])
        recs["r3"] = make_record("r3", "GSE_A", "5'UTR", "CCCC", "TAAA",
                                 ops_list=[("SUB", 0, "T"), ("SUB", 1, "A"),
                                           ("SUB", 2, "A")])
        res = audit_split_leakage("5utr_source_disjoint", manifest_path, recs)
        assert not res["pass"]
        assert res["acceptance"]["endpoint_as_train_intermediate"] == 1
