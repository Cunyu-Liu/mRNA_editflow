"""Offline contract tests for P0 Data Reconstruction v1."""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest

from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.download_mrna import load_records_jsonl, write_records_jsonl
from mrna_editflow.data.reconstruction import (
    CanonicalInput,
    CanonicalRecordError,
    RawIntegrityError,
    build_cross_source_roles,
    build_combined_reconstruction,
    build_family_assignments,
    build_source_bundle,
    canonicalize_records,
    derive_model_view,
    verify_raw_artifact,
    verify_source_bundle,
)
from mrna_editflow.data.split_contract import (
    build_split_manifest,
    load_and_verify_split_manifest,
)


def _valid_cds(body: str = "GCU") -> str:
    return "AUG" + body + "UAA"


class RawIntegrityTest(unittest.TestCase):
    def test_complete_gzip_is_verified_and_truncated_gzip_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p0_raw_") as tmp:
            complete = os.path.join(tmp, "source.fa.gz")
            with gzip.open(complete, "wt", encoding="utf-8") as fh:
                fh.write(">tx\nAUGGCUUAA\n")
            evidence = verify_raw_artifact(complete)
            self.assertTrue(evidence["gzip_complete"])
            self.assertGreater(evidence["uncompressed_size_bytes"], 0)
            self.assertEqual(len(evidence["sha256"]), 64)

            truncated = os.path.join(tmp, "truncated.fa.gz")
            with open(complete, "rb") as src, open(truncated, "wb") as dst:
                dst.write(src.read()[:-5])
            with self.assertRaises(RawIntegrityError):
                verify_raw_artifact(truncated)

    def test_expected_size_and_sha_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p0_raw_hash_") as tmp:
            path = os.path.join(tmp, "source.txt")
            with open(path, "wb") as fh:
                fh.write(b"abc")
            with self.assertRaises(RawIntegrityError):
                verify_raw_artifact(path, expected_size_bytes=4)
            with self.assertRaises(RawIntegrityError):
                verify_raw_artifact(path, expected_sha256="0" * 64)


class CanonicalAndDerivedTest(unittest.TestCase):
    def test_canonical_keeps_full_regions_and_derived_view_is_traceable(self) -> None:
        raw = MRNARecord("ENST1.1", "A" * 20, _valid_cds("GCU" * 5), "C" * 30)
        inputs = [
            CanonicalInput(
                record=raw,
                source="gencode_v45",
                source_accession="ENST1.1",
                source_record_index=0,
                gene_id="ENSG1",
                gene_symbol="GENE1",
            )
        ]
        canonical, metadata, stats = canonicalize_records(inputs)
        self.assertEqual(canonical[0].five_utr, "A" * 20)
        self.assertEqual(canonical[0].cds, _valid_cds("GCU" * 5))
        self.assertEqual(canonical[0].three_utr, "C" * 30)
        self.assertEqual(stats["truncated_5utr"], 0)
        self.assertEqual(stats["truncated_3utr"], 0)

        view, lineage, view_stats = derive_model_view(
            canonical,
            metadata,
            max_5utr=8,
            max_cds=24,
            max_3utr=9,
        )
        self.assertEqual(len(view), 1)
        self.assertEqual(view[0].five_utr, "A" * 8)
        self.assertEqual(view[0].three_utr, "C" * 9)
        self.assertEqual(lineage[0]["canonical_index"], 0)
        self.assertEqual(lineage[0]["canonical_id"], "gencode_v45:ENST1.1")
        self.assertTrue(lineage[0]["truncated_5utr"])
        self.assertTrue(lineage[0]["truncated_3utr"])
        self.assertEqual(view_stats["kept"], 1)

    def test_canonical_rejects_duplicate_source_identity(self) -> None:
        record = MRNARecord("NM_1.1", "", _valid_cds(), "")
        row = CanonicalInput(record, "refseq", "NM_1.1", 0)
        with self.assertRaises(CanonicalRecordError):
            canonicalize_records([row, row])

    def test_invalid_orf_is_attributed_without_silent_repair(self) -> None:
        bad = CanonicalInput(
            MRNARecord("NM_BAD.1", "", "CCCGCUUAA", ""),
            "refseq",
            "NM_BAD.1",
            0,
        )
        records, metadata, stats = canonicalize_records([bad])
        self.assertEqual(records, [])
        self.assertEqual(metadata, [])
        self.assertEqual(stats["bad_start_codon"], 1)


class FamilyAndSplitTest(unittest.TestCase):
    def _fixture(self):
        records = [
            MRNARecord("gencode_v45:ENST1.1", "A", _valid_cds("GCU"), "C"),
            MRNARecord("refseq:NM_1.1", "G", _valid_cds("GCC"), "U"),
            MRNARecord("gencode_v45:ENST2.1", "C", _valid_cds("AAA"), "A"),
            MRNARecord("refseq:NM_2.1", "U", _valid_cds("AAA"), "G"),
            MRNARecord("refseq:NM_3.1", "AC", _valid_cds("GGG"), "GU"),
        ]
        metadata = [
            {"canonical_id": "gencode_v45:ENST1.1", "source": "gencode_v45", "gene_symbol": "SAME"},
            {"canonical_id": "refseq:NM_1.1", "source": "refseq", "gene_symbol": "SAME"},
            {"canonical_id": "gencode_v45:ENST2.1", "source": "gencode_v45", "gene_symbol": "LEFT"},
            {"canonical_id": "refseq:NM_2.1", "source": "refseq", "gene_symbol": "RIGHT"},
            {"canonical_id": "refseq:NM_3.1", "source": "refseq", "gene_symbol": "OTHER"},
        ]
        return records, metadata

    def test_family_assignment_unions_cross_source_gene_and_exact_protein(self) -> None:
        records, metadata = self._fixture()
        assignments, evidence = build_family_assignments(records, metadata)
        self.assertEqual(assignments[0], assignments[1])  # shared gene symbol
        self.assertEqual(assignments[2], assignments[3])  # synonymous exact protein
        self.assertNotEqual(assignments[3], assignments[4])
        self.assertEqual(len(assignments), len(records))
        self.assertTrue(any(row["n_sources"] == 2 for row in evidence))

    def test_cross_source_roles_exclude_shared_families(self) -> None:
        records, metadata = self._fixture()
        assignments, _ = build_family_assignments(records, metadata)
        roles, excluded = build_cross_source_roles(assignments, metadata, seed=17)
        self.assertEqual(roles["train"], [0, 2])
        self.assertEqual(excluded, [1, 3])
        self.assertEqual(roles["val"] + roles["test"], [4])

    def test_split_manifest_supports_reasoned_excluded_universe(self) -> None:
        records, metadata = self._fixture()
        assignments, _ = build_family_assignments(records, metadata)
        roles, excluded = build_cross_source_roles(assignments, metadata, seed=17)
        with tempfile.TemporaryDirectory(prefix="p0_split_") as tmp:
            records_path = os.path.join(tmp, "records.jsonl")
            write_records_jsonl(records, records_path)
            role_paths = {}
            for role, indices in roles.items():
                path = os.path.join(tmp, f"{role}.idx")
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write("".join(f"{idx}\n" for idx in indices))
                role_paths[role] = path
            excluded_path = os.path.join(tmp, "excluded.idx")
            with open(excluded_path, "w", encoding="utf-8") as fh:
                fh.write("".join(f"{idx}\n" for idx in excluded))
            cluster_path = os.path.join(tmp, "clusters.json")
            with open(cluster_path, "w", encoding="utf-8") as fh:
                json.dump(assignments, fh, separators=(",", ":"))
            leakage_path = os.path.join(tmp, "leakage.json")
            with open(leakage_path, "w", encoding="utf-8") as fh:
                json.dump({"split": {"cluster_disjoint": True}, "summary": {"exact_match_count": 0}}, fh)
            manifest = build_split_manifest(
                dataset_id="p0_cross_source_fixture",
                records_path=records_path,
                role_idx_paths=role_paths,
                excluded_idx_path=excluded_path,
                excluded_reason="refseq_family_overlaps_gencode_training_family",
                leakage_report_path=leakage_path,
                algorithm="source_holdout_exact_family_union_v1",
                seed=17,
                family_threshold=1.0,
                family_disjoint=True,
                exact_cross_role_matches=0,
                near_neighbor_threshold_passed=False,
                cluster_assignment_path=cluster_path,
                paper_eligible=False,
                block_reasons=["exhaustive_cross_role_near_neighbor_audit_pending"],
            )
            manifest_path = os.path.join(tmp, "split_manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2, sort_keys=True)
            contract = load_and_verify_split_manifest(manifest_path)
            self.assertEqual(contract.excluded.indices, tuple(excluded))
            self.assertFalse(contract.paper_eligible)
            self.assertEqual(load_records_jsonl(contract.records_path)[0].transcript_id, records[0].transcript_id)


class ReconstructionBundleTest(unittest.TestCase):
    def test_module_cli_defines_writers_before_main_executes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p0_cli_") as tmp:
            gencode_raw = os.path.join(tmp, "gencode.v45.pc_transcripts.fa.gz")
            with gzip.open(gencode_raw, "wt", encoding="utf-8") as fh:
                fh.write(">ENSTCLI.1|ENSGCLI.1|x|x|TX|GCLI|21|CDS:7-15|\n")
                fh.write("GCCACCATGGCTTAAATAAAA\n")
            refseq_raw = os.path.join(tmp, "human.1.rna.gbff.gz")
            with gzip.open(refseq_raw, "wt", encoding="utf-8") as fh:
                fh.write(
                    "LOCUS       NM_CLI.1 21 bp mRNA\n"
                    "ACCESSION   NM_CLI\n"
                    "VERSION     NM_CLI.1\n"
                    "FEATURES             Location/Qualifiers\n"
                    "     source          1..21\n"
                    "     CDS             7..15\n"
                    "                     /gene=\"RCLI\"\n"
                    "ORIGIN\n"
                    "        1 aaaaaaatgaaataacccccc\n"
                    "//\n"
                )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mrna_editflow.data.reconstruction",
                    "--gencode-source",
                    gencode_raw,
                    "--refseq-source",
                    refseq_raw,
                    "--allow-incomplete-refseq",
                    "--frozen-root",
                    os.path.join(tmp, "frozen"),
                    "--split-root",
                    os.path.join(tmp, "splits"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(os.path.exists(os.path.join(tmp, "frozen", "combined", "combined_reconstruction_manifest.json")))

    def test_refseq_bundle_aggregates_multiple_release_partitions(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p0_refseq_parts_") as tmp:
            raw_dir = os.path.join(tmp, "raw")
            os.makedirs(raw_dir)
            for number in (1, 2):
                path = os.path.join(raw_dir, f"human.{number}.rna.gbff.gz")
                accession = f"NM_PART{number}.1"
                with gzip.open(path, "wt", encoding="utf-8") as fh:
                    fh.write(
                        f"LOCUS       {accession} 21 bp mRNA\n"
                        f"ACCESSION   NM_PART{number}\n"
                        f"VERSION     {accession}\n"
                        "FEATURES             Location/Qualifiers\n"
                        "     source          1..21\n"
                        "     CDS             7..15\n"
                        f"                     /gene=\"GENE{number}\"\n"
                        "ORIGIN\n"
                        "        1 gccaccatggcttaaataaaa\n"
                        "//\n"
                    )
            bundle = build_source_bundle(
                source="refseq_human_rna",
                raw_path=raw_dir,
                output_dir=os.path.join(tmp, "bundle"),
            )
            manifest = verify_source_bundle(bundle["manifest_path"])
            self.assertEqual(manifest["raw"]["artifact_count"], 2)
            self.assertEqual(manifest["canonical"]["count"], 2)
            metadata = []
            with open(manifest["canonical"]["metadata_path"], "r", encoding="utf-8") as fh:
                metadata = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual({row["source_file"] for row in metadata}, {
                "human.1.rna.gbff.gz", "human.2.rna.gbff.gz"
            })

    def test_two_source_bundle_builds_four_verified_split_contracts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p0_bundle_") as tmp:
            gencode_raw = os.path.join(tmp, "gencode.v45.pc_transcripts.fa.gz")
            with gzip.open(gencode_raw, "wt", encoding="utf-8") as fh:
                fh.write(
                    ">ENST1.1|ENSG1.1|x|x|TX1|SHARED|22|CDS:7-15|\n"
                    "GCCACCATGGCTTAAAATAAAA\n"
                    ">ENST2.1|ENSG2.1|x|x|TX2|GONLY|22|CDS:7-15|\n"
                    "AAAAAAATGAAATAACCCCCC\n"
                )
            refseq_raw = os.path.join(tmp, "human.1.rna.gbff.gz")
            with gzip.open(refseq_raw, "wt", encoding="utf-8") as fh:
                for accession, gene, sequence in (
                    ("NM_1.1", "SHARED", "CCCCCCATGGCTTAAGGGGGG"),
                    ("NM_2.1", "RONLY", "UUUUUUATGGGGTAAAAAAAA"),
                ):
                    fh.write(
                        f"LOCUS       {accession} 22 bp mRNA\n"
                        f"ACCESSION   {accession.split('.')[0]}\n"
                        f"VERSION     {accession}\n"
                        "FEATURES             Location/Qualifiers\n"
                        "     source          1..22\n"
                        "     CDS             7..15\n"
                        f"                     /gene=\"{gene}\"\n"
                        f"                     /protein_id=\"NP_{accession}\"\n"
                        "ORIGIN\n"
                        f"        1 {sequence.lower()}\n"
                        "//\n"
                    )
            gencode = build_source_bundle(
                source="gencode_v45",
                raw_path=gencode_raw,
                output_dir=os.path.join(tmp, "gencode"),
            )
            refseq = build_source_bundle(
                source="refseq_human_rna",
                raw_path=refseq_raw,
                output_dir=os.path.join(tmp, "refseq"),
            )
            combined = build_combined_reconstruction(
                gencode_manifest_path=gencode["manifest_path"],
                refseq_manifest_path=refseq["manifest_path"],
                output_dir=os.path.join(tmp, "combined"),
                split_root=os.path.join(tmp, "splits"),
                seed=31,
            )
            self.assertEqual(combined["combined_records"]["count"], 4)
            self.assertEqual(set(combined["split_manifests"]), {
                "gencode_family", "refseq_family", "combined_family", "gencode_to_refseq"
            })
            for row in combined["split_manifests"].values():
                contract = load_and_verify_split_manifest(row["manifest_path"])
                self.assertFalse(contract.paper_eligible)
                self.assertIn("exhaustive_cross_role_near_neighbor_audit_pending", contract.block_reasons)

            manifest = verify_source_bundle(gencode["manifest_path"])
            metadata_path = manifest["canonical"]["metadata_path"]
            with open(metadata_path, "a", encoding="utf-8") as fh:
                fh.write("{}\n")
            with self.assertRaises(Exception):
                verify_source_bundle(gencode["manifest_path"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
