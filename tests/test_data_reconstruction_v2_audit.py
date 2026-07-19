"""Offline contract tests for P0 Data Reconstruction v2 audit."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest

from mrna_editflow.core.constants import translate
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.download_mrna import write_records_jsonl
from mrna_editflow.data.reconstruction import build_family_assignments
from mrna_editflow.data.reconstruction_v2_audit import (
    V2AuditError,
    exhaustive_cross_role_near_neighbor_audit,
    gene_symbol_alias_audit,
    kmer_hashes,
    promote_split_to_paper_eligible,
    record_split_audit_outcome,
)
from mrna_editflow.data.split_contract import (
    SPLIT_ROLES,
    build_split_manifest,
    load_and_verify_split_manifest,
)


def _valid_cds(body: str = "GCU") -> str:
    return "AUG" + body + "UAA"


def _long_rna(seed: str, length: int = 40) -> str:
    """Deterministic pseudo-RNA of the requested length from a seed string."""
    bases = "ACGU"
    out = []
    h = hashlib.sha256(seed.encode("ascii")).digest()
    i = 0
    while len(out) < length:
        if i >= len(h):
            h = hashlib.sha256(h).digest()
            i = 0
        out.append(bases[h[i] % 4])
        i += 1
    return "".join(out)


class TestKmerHashes(unittest.TestCase):
    def test_empty_sequence_returns_empty_set(self) -> None:
        self.assertEqual(kmer_hashes("", 15), set())

    def test_sequence_shorter_than_k_returns_empty_set(self) -> None:
        self.assertEqual(kmer_hashes("ACGUACGUACGUAC", 15), set())

    def test_known_sequence_produces_expected_hash_count(self) -> None:
        # Use a non-repeating sequence so each k-mer is unique.
        seq = _long_rna("hash_count_seed", 32)
        kmers = kmer_hashes(seq, 15)
        self.assertEqual(len(kmers), 32 - 15 + 1)

    def test_non_acgu_resets_window(self) -> None:
        clean = kmer_hashes("ACGUACGUACGUACGUACGUACGUACGUACGU", 15)
        with_n = kmer_hashes("ACGUACGUACGUACGNACGUACGUACGUACGU", 15)
        # The N splits the sequence into two windows of 15 and 15, but only
        # the first window (length 15) yields a k-mer; the second window
        # (length 15 starting after N) also yields one. So we still get
        # k-mers, but strictly fewer than the clean version.
        self.assertLess(len(with_n), len(clean))

    def test_case_insensitive(self) -> None:
        upper = kmer_hashes("ACGUACGUACGUACGU", 15)
        lower = kmer_hashes("acguacguacguacgu", 15)
        self.assertEqual(upper, lower)

    def test_t_and_u_are_equivalent(self) -> None:
        rna = kmer_hashes("ACGUACGUACGUACGU", 15)
        dna = kmer_hashes("ACGTACGTACGTACGT", 15)
        self.assertEqual(rna, dna)


class TestExhaustiveCrossRoleNearNeighborAudit(unittest.TestCase):
    def _records(self):
        # Two very different records + one near-duplicate of record 0.
        seq_a = _long_rna("record_a", 40)
        seq_b = _long_rna("record_b", 40)
        seq_c = seq_a  # identical to A -> 100% containment, 100% jaccard
        return [
            MRNARecord("tx_a", "", _valid_cds(seq_a[:30]), ""),
            MRNARecord("tx_b", "", _valid_cds(seq_b[:30]), ""),
            MRNARecord("tx_c", "", _valid_cds(seq_c[:30]), ""),
        ]

    def test_clean_split_passes(self) -> None:
        records = [
            MRNARecord("tx_a", "", _valid_cds(_long_rna("a", 30)), ""),
            MRNARecord("tx_b", "", _valid_cds(_long_rna("b", 30)), ""),
            MRNARecord("tx_c", "", _valid_cds(_long_rna("c", 30)), ""),
        ]
        roles = {"train": [0], "val": [1], "test": [2]}
        result = exhaustive_cross_role_near_neighbor_audit(records, roles)
        self.assertTrue(result["passed"])
        self.assertEqual(result["violations"], [])
        self.assertEqual(result["stats"]["n_violations"], 0)

    def test_identical_sequences_in_different_roles_are_flagged(self) -> None:
        records = self._records()
        # Put record 0 in train and record 2 (identical seq) in val.
        roles = {"train": [0, 1], "val": [2], "test": []}
        result = exhaustive_cross_role_near_neighbor_audit(records, roles)
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["violations"]), 1)
        v = result["violations"][0]
        self.assertEqual(v["role_a"], "train")
        self.assertEqual(v["index_a"], 0)
        self.assertEqual(v["role_b"], "val")
        self.assertEqual(v["index_b"], 2)
        self.assertGreaterEqual(v["jaccard"], 0.8)
        self.assertGreaterEqual(v["containment_a_to_b"], 0.95)

    def test_identical_sequences_in_same_role_are_not_flagged(self) -> None:
        records = self._records()
        # Records 0 and 2 (identical) both in train -> no cross-role pair.
        roles = {"train": [0, 2], "val": [1], "test": []}
        result = exhaustive_cross_role_near_neighbor_audit(records, roles)
        self.assertTrue(result["passed"])

    def test_missing_role_raises(self) -> None:
        records = self._records()
        with self.assertRaises(V2AuditError):
            exhaustive_cross_role_near_neighbor_audit(records, {"train": [0], "val": [1]})


class TestGeneSymbolAliasAudit(unittest.TestCase):
    def _fixture_clean(self):
        records = [
            MRNARecord("gencode:ENST1", "A", _valid_cds("GCU"), "C"),
            MRNARecord("refseq:NM_1", "G", _valid_cds("GCU"), "U"),
            MRNARecord("gencode:ENST2", "C", _valid_cds("AAA"), "A"),
            MRNARecord("refseq:NM_2", "U", _valid_cds("GGG"), "G"),
        ]
        metadata = [
            {"canonical_id": "gencode:ENST1", "source": "gencode_v45", "gene_symbol": "GENE_A"},
            {"canonical_id": "refseq:NM_1", "source": "refseq", "gene_symbol": "GENE_A"},
            {"canonical_id": "gencode:ENST2", "source": "gencode_v45", "gene_symbol": "GENE_B"},
            {"canonical_id": "refseq:NM_2", "source": "refseq", "gene_symbol": "GENE_C"},
        ]
        assignments, evidence = build_family_assignments(records, metadata)
        return records, metadata, assignments, evidence

    def test_clean_fixture_passes(self) -> None:
        records, metadata, assignments, evidence = self._fixture_clean()
        result = gene_symbol_alias_audit(records, metadata, assignments, evidence)
        self.assertTrue(result["passed"])
        self.assertEqual(result["stats"]["n_protein_sha256_gaps"], 0)
        self.assertEqual(result["stats"]["n_rna_sha256_gaps"], 0)

    def test_protein_gap_is_flagged(self) -> None:
        records, metadata, assignments, evidence = self._fixture_clean()
        # Force record 3 into a different family despite sharing protein with record 2.
        # Records 2 and 3 both have CDS = AUGAAAUAA / AUGGGGUAA — different proteins.
        # Instead, craft two records with the SAME CDS but different families.
        records2 = [
            MRNARecord("tx1", "", _valid_cds("GCU"), ""),
            MRNARecord("tx2", "", _valid_cds("GCU"), ""),
        ]
        metadata2 = [
            {"canonical_id": "tx1", "source": "gencode_v45", "gene_symbol": "X"},
            {"canonical_id": "tx2", "source": "refseq", "gene_symbol": "Y"},
        ]
        # Same CDS -> same protein -> build_family_assignments unions them.
        assignments2, evidence2 = build_family_assignments(records2, metadata2)
        self.assertEqual(assignments2[0], assignments2[1])
        # Now artificially break the assignment to simulate a gap.
        broken = [assignments2[0], assignments2[0] + 1]
        broken_evidence = [
            {"cluster_id": 0, "count": 1, "members": [0], "sources": ["gencode_v45"], "n_sources": 1, "gene_symbols": ["X"]},
            {"cluster_id": 1, "count": 1, "members": [1], "sources": ["refseq"], "n_sources": 1, "gene_symbols": ["Y"]},
        ]
        result = gene_symbol_alias_audit(records2, metadata2, broken, broken_evidence)
        self.assertFalse(result["passed"])
        self.assertGreater(result["stats"]["n_protein_sha256_gaps"], 0)

    def test_rna_gap_is_flagged(self) -> None:
        records = [
            MRNARecord("tx1", "AAAA", _valid_cds("GCU"), "CCCC"),
            MRNARecord("tx2", "AAAA", _valid_cds("GCU"), "CCCC"),
        ]
        metadata = [
            {"canonical_id": "tx1", "source": "gencode_v45", "gene_symbol": "X"},
            {"canonical_id": "tx2", "source": "refseq", "gene_symbol": "Y"},
        ]
        # Same seq -> same rna_sha256 -> same family.
        assignments, evidence = build_family_assignments(records, metadata)
        self.assertEqual(assignments[0], assignments[1])
        broken = [0, 1]
        broken_evidence = [
            {"cluster_id": 0, "count": 1, "members": [0], "sources": ["gencode_v45"], "n_sources": 1, "gene_symbols": ["X"]},
            {"cluster_id": 1, "count": 1, "members": [1], "sources": ["refseq"], "n_sources": 1, "gene_symbols": ["Y"]},
        ]
        result = gene_symbol_alias_audit(records, metadata, broken, broken_evidence)
        self.assertFalse(result["passed"])
        self.assertGreater(result["stats"]["n_rna_sha256_gaps"], 0)

    def test_alias_observations_documented_but_not_failures(self) -> None:
        records, metadata, assignments, evidence = self._fixture_clean()
        # The clean fixture may or may not have alias observations;
        # the key invariant is that passed=True regardless.
        result = gene_symbol_alias_audit(records, metadata, assignments, evidence)
        self.assertTrue(result["passed"])
        # alias_observations is a list (may be empty).
        self.assertIsInstance(result["alias_observations"], list)


class TestPromoteSplitToPaperEligible(unittest.TestCase):
    def _build_split_package(self, tmp, records, assignments, roles, *, excluded=None, excluded_reason=None, seed=17):
        records_path = os.path.join(tmp, "records.jsonl")
        write_records_jsonl(records, records_path)
        role_paths = {}
        for role, indices in roles.items():
            path = os.path.join(tmp, f"{role}.idx")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("".join(f"{idx}\n" for idx in indices))
            role_paths[role] = path
        excluded_path = None
        if excluded is not None:
            excluded_path = os.path.join(tmp, "excluded.idx")
            with open(excluded_path, "w", encoding="utf-8") as fh:
                fh.write("".join(f"{idx}\n" for idx in excluded))
        cluster_path = os.path.join(tmp, "clusters.json")
        with open(cluster_path, "w", encoding="utf-8") as fh:
            json.dump(assignments, fh, separators=(",", ":"))
        leakage_path = os.path.join(tmp, "leakage.json")
        with open(leakage_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "artifact_kind": "p0_data_reconstruction_split_audit",
                    "dataset_id": "fixture",
                    "method": "exact_sequence_and_family_assignment_audit",
                    "split": {"cluster_disjoint": True},
                    "summary": {
                        "n_records": len(records),
                        "n_train": len(roles["train"]),
                        "n_val": len(roles["val"]),
                        "n_test": len(roles["test"]),
                        "n_excluded": len(excluded or []),
                        "exact_match_count": 0,
                        "near_neighbor_threshold_passed": False,
                        "near_neighbor_exhaustive": False,
                    },
                    "scientific_warning": "v1 placeholder",
                },
                fh,
            )
        manifest = build_split_manifest(
            dataset_id="fixture_split",
            records_path=records_path,
            role_idx_paths=role_paths,
            excluded_idx_path=excluded_path,
            excluded_reason=excluded_reason,
            leakage_report_path=leakage_path,
            algorithm="fixture_family_union_v1",
            seed=seed,
            family_threshold=1.0,
            family_disjoint=True,
            exact_cross_role_matches=0,
            near_neighbor_threshold_passed=False,
            cluster_assignment_path=cluster_path,
            paper_eligible=False,
            block_reasons=[
                "exhaustive_cross_role_near_neighbor_audit_pending",
                "gene_symbol_alias_mapping_not_independently_audited",
            ],
        )
        manifest_path = os.path.join(tmp, "split_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
        return manifest_path

    def test_promotion_flips_paper_eligible_and_clears_blockers(self) -> None:
        # Three distinct records -> no near-neighbor violations.
        records = [
            MRNARecord("tx_a", "A", _valid_cds(_long_rna("a", 30)), "C"),
            MRNARecord("tx_b", "G", _valid_cds(_long_rna("b", 30)), "U"),
            MRNARecord("tx_c", "C", _valid_cds(_long_rna("c", 30)), "A"),
        ]
        metadata = [
            {"canonical_id": "tx_a", "source": "gencode_v45", "gene_symbol": "A"},
            {"canonical_id": "tx_b", "source": "gencode_v45", "gene_symbol": "B"},
            {"canonical_id": "tx_c", "source": "gencode_v45", "gene_symbol": "C"},
        ]
        assignments, evidence = build_family_assignments(records, metadata)
        roles = {"train": [0], "val": [1], "test": [2]}
        with tempfile.TemporaryDirectory(prefix="p0_v2_") as tmp:
            manifest_path = self._build_split_package(tmp, records, assignments, roles)
            # Run the two v2 audits.
            nn_audit = exhaustive_cross_role_near_neighbor_audit(records, roles)
            self.assertTrue(nn_audit["passed"])
            alias_audit = gene_symbol_alias_audit(records, metadata, assignments, evidence)
            self.assertTrue(alias_audit["passed"])
            # Promote.
            summary = promote_split_to_paper_eligible(
                tmp,
                near_neighbor_audit=nn_audit,
                alias_audit=alias_audit,
            )
            self.assertTrue(summary["paper_eligible"])
            self.assertEqual(summary["block_reasons"], [])
            # Re-verify with the contract verifier.
            contract = load_and_verify_split_manifest(manifest_path)
            self.assertTrue(contract.paper_eligible)
            self.assertEqual(contract.block_reasons, ())
            self.assertTrue(contract.near_neighbor_threshold_passed)
            # Audit artifacts present.
            self.assertTrue(os.path.isfile(os.path.join(tmp, "near_neighbor_audit.json")))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "alias_audit.json")))
            # Leakage report updated.
            with open(contract.leakage_report_path, "r", encoding="utf-8") as fh:
                leakage = json.load(fh)
            self.assertTrue(leakage["summary"]["near_neighbor_threshold_passed"])
            self.assertTrue(leakage["summary"]["near_neighbor_exhaustive"])

    def test_promotion_refused_when_near_neighbor_audit_fails(self) -> None:
        # Two records with identical full sequences but different gene symbols,
        # manually assigned to different families so they can be in different
        # roles.  The near-neighbor audit will flag the identical sequences.
        identical_seq = _long_rna("identical", 40)
        records = [
            MRNARecord("tx_a", "A", _valid_cds(identical_seq[:30]), "C"),
            MRNARecord("tx_b", "A", _valid_cds(identical_seq[:30]), "C"),  # identical
            MRNARecord("tx_c", "C", _valid_cds(_long_rna("c", 30)), "A"),
        ]
        metadata = [
            {"canonical_id": "tx_a", "source": "gencode_v45", "gene_symbol": "A"},
            {"canonical_id": "tx_b", "source": "gencode_v45", "gene_symbol": "B"},
            {"canonical_id": "tx_c", "source": "gencode_v45", "gene_symbol": "C"},
        ]
        # Manually assign to different families (0, 1, 0) so records 0 and 1
        # can be placed in different roles despite identical sequences.
        assignments = [0, 1, 0]
        evidence = [
            {"cluster_id": 0, "count": 2, "members": [0, 2], "sources": ["gencode_v45"], "n_sources": 1, "gene_symbols": ["A", "C"]},
            {"cluster_id": 1, "count": 1, "members": [1], "sources": ["gencode_v45"], "n_sources": 1, "gene_symbols": ["B"]},
        ]
        roles = {"train": [0], "val": [1], "test": [2]}
        with tempfile.TemporaryDirectory(prefix="p0_v2_") as tmp:
            self._build_split_package(tmp, records, assignments, roles)
            nn_audit = exhaustive_cross_role_near_neighbor_audit(records, roles)
            self.assertFalse(nn_audit["passed"])
            alias_audit = gene_symbol_alias_audit(records, metadata, assignments, evidence)
            with self.assertRaises(V2AuditError):
                promote_split_to_paper_eligible(
                    tmp,
                    near_neighbor_audit=nn_audit,
                    alias_audit=alias_audit,
                )


class TestRecordSplitAuditOutcome(unittest.TestCase):
    def _build_split_package(self, tmp, records, assignments, roles, *, excluded=None, excluded_reason=None, seed=17):
        records_path = os.path.join(tmp, "records.jsonl")
        write_records_jsonl(records, records_path)
        role_paths = {}
        for role, indices in roles.items():
            path = os.path.join(tmp, f"{role}.idx")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("".join(f"{idx}\n" for idx in indices))
            role_paths[role] = path
        excluded_path = None
        if excluded is not None:
            excluded_path = os.path.join(tmp, "excluded.idx")
            with open(excluded_path, "w", encoding="utf-8") as fh:
                fh.write("".join(f"{idx}\n" for idx in excluded))
        cluster_path = os.path.join(tmp, "clusters.json")
        with open(cluster_path, "w", encoding="utf-8") as fh:
            json.dump(assignments, fh, separators=(",", ":"))
        leakage_path = os.path.join(tmp, "leakage.json")
        with open(leakage_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "artifact_kind": "p0_data_reconstruction_split_audit",
                    "dataset_id": "fixture",
                    "method": "exact_sequence_and_family_assignment_audit",
                    "split": {"cluster_disjoint": True},
                    "summary": {
                        "n_records": len(records),
                        "n_train": len(roles["train"]),
                        "n_val": len(roles["val"]),
                        "n_test": len(roles["test"]),
                        "n_excluded": len(excluded or []),
                        "exact_match_count": 0,
                        "near_neighbor_threshold_passed": False,
                        "near_neighbor_exhaustive": False,
                    },
                    "scientific_warning": "v1 placeholder",
                },
                fh,
            )
        manifest = build_split_manifest(
            dataset_id="fixture_split",
            records_path=records_path,
            role_idx_paths=role_paths,
            excluded_idx_path=excluded_path,
            excluded_reason=excluded_reason,
            leakage_report_path=leakage_path,
            algorithm="fixture_family_union_v1",
            seed=seed,
            family_threshold=1.0,
            family_disjoint=True,
            exact_cross_role_matches=0,
            near_neighbor_threshold_passed=False,
            cluster_assignment_path=cluster_path,
            paper_eligible=False,
            block_reasons=[
                "exhaustive_cross_role_near_neighbor_audit_pending",
                "gene_symbol_alias_mapping_not_independently_audited",
            ],
        )
        manifest_path = os.path.join(tmp, "split_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
        return manifest_path

    def test_records_outcome_and_clears_pending_blockers(self) -> None:
        # Near-neighbor audit fails (one record's k-mers are fully contained
        # in another's), but alias audit passes because the records have
        # different rna_sha256 and (failed-or-different) protein_sha256.
        shared_body = _long_rna("shared", 100)
        long_body = shared_body + _long_rna("extension", 300)
        records = [
            MRNARecord("tx_a", "", _valid_cds(shared_body), ""),
            MRNARecord("tx_b", "", _valid_cds(long_body), ""),
            MRNARecord("tx_c", "", _valid_cds(_long_rna("c", 90)), ""),
        ]
        metadata = [
            {"canonical_id": "tx_a", "source": "gencode_v45", "gene_symbol": "A"},
            {"canonical_id": "tx_b", "source": "gencode_v45", "gene_symbol": "B"},
            {"canonical_id": "tx_c", "source": "gencode_v45", "gene_symbol": "C"},
        ]
        # Records a and b have different rna_sha256, so they may legitimately
        # be in different families.  Assign a→0, b→1, c→2 (all disjoint).
        assignments = [0, 1, 2]
        evidence = [
            {"cluster_id": 0, "count": 1, "members": [0], "sources": ["gencode_v45"], "n_sources": 1, "gene_symbols": ["A"]},
            {"cluster_id": 1, "count": 1, "members": [1], "sources": ["gencode_v45"], "n_sources": 1, "gene_symbols": ["B"]},
            {"cluster_id": 2, "count": 1, "members": [2], "sources": ["gencode_v45"], "n_sources": 1, "gene_symbols": ["C"]},
        ]
        roles = {"train": [0], "val": [1], "test": [2]}
        with tempfile.TemporaryDirectory(prefix="p0_v2_") as tmp:
            manifest_path = self._build_split_package(tmp, records, assignments, roles)
            nn_audit = exhaustive_cross_role_near_neighbor_audit(records, roles)
            # tx_a's k-mers are fully contained in tx_b → violation.
            self.assertFalse(nn_audit["passed"])
            self.assertGreater(nn_audit["stats"]["n_violations"], 0)
            alias_audit = gene_symbol_alias_audit(records, metadata, assignments, evidence)
            self.assertTrue(alias_audit["passed"])
            summary = record_split_audit_outcome(
                tmp,
                near_neighbor_audit=nn_audit,
                alias_audit=alias_audit,
            )
            # paper_eligible must remain False.
            self.assertFalse(summary["paper_eligible"])
            # "pending" blockers replaced with "violations_found"; alias blocker dropped.
            self.assertNotIn(
                "exhaustive_cross_role_near_neighbor_audit_pending",
                summary["block_reasons"],
            )
            self.assertNotIn(
                "gene_symbol_alias_mapping_not_independently_audited",
                summary["block_reasons"],
            )
            self.assertIn(
                "exhaustive_cross_role_near_neighbor_audit_violations_found",
                summary["block_reasons"],
            )
            # Audit artifacts present.
            self.assertTrue(os.path.isfile(os.path.join(tmp, "near_neighbor_audit.json")))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "alias_audit.json")))
            # Contract verifier confirms paper_eligible=False.
            contract = load_and_verify_split_manifest(manifest_path)
            self.assertFalse(contract.paper_eligible)
            # Leakage report updated with violation count.
            with open(contract.leakage_report_path, "r", encoding="utf-8") as fh:
                leakage = json.load(fh)
            self.assertTrue(leakage["summary"]["near_neighbor_exhaustive"])
            self.assertGreater(leakage["summary"]["near_neighbor_violation_count"], 0)
            self.assertFalse(leakage["summary"]["near_neighbor_threshold_passed"])


if __name__ == "__main__":
    unittest.main()
