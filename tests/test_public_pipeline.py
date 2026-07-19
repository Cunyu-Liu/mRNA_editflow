"""Stdlib tests for the public-corpus build path."""
from __future__ import annotations

import gzip
import json
import os
import tempfile
import unittest

from mrna_editflow.data.download_mrna import (
    load_records_jsonl,
    parse_gencode_cds_range,
    parse_genbank_cds_location,
    transcript_id_from_gencode_header,
)
from mrna_editflow.data.public_pipeline import build_public_corpus


class PublicPipelineTest(unittest.TestCase):
    def test_gencode_header_parsing(self) -> None:
        header = "ENST000001.5|GENE|protein_coding|CDS:7-15|extra"
        self.assertEqual(parse_gencode_cds_range(header), (6, 15))
        self.assertEqual(transcript_id_from_gencode_header(header), "ENST000001.5")
        self.assertIsNone(parse_gencode_cds_range("ENST000002|no_cds"))

    def test_genbank_cds_location_parsing_is_conservative(self) -> None:
        self.assertEqual(parse_genbank_cds_location("7..15"), (6, 15))
        self.assertEqual(parse_genbank_cds_location("<7..>15"), (6, 15))
        self.assertEqual(parse_genbank_cds_location("join(7..9,10..15)"), (6, 15))
        self.assertIsNone(parse_genbank_cds_location("join(7..9,12..15)"))
        self.assertIsNone(parse_genbank_cds_location("complement(7..15)"))
        self.assertIsNone(parse_genbank_cds_location("NM_000000.1:7..15"))

    def test_build_public_corpus_manifest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mef_public_") as tmp:
            raw_dir = os.path.join(tmp, "raw")
            out_dir = os.path.join(tmp, "processed")
            os.makedirs(raw_dir, exist_ok=True)
            fasta_path = os.path.join(raw_dir, "gencode.v45.pc_transcripts.fa.gz")
            with gzip.open(fasta_path, "wt", encoding="utf-8") as fh:
                fh.write(">ENST_VALID.1|gene|protein_coding|CDS:7-15\n")
                fh.write("GCCACCATGGCTTAAAATAAA\n")
                fh.write(">ENST_BAD.1|gene|protein_coding|CDS:7-15\n")
                fh.write("GCCACCCCCGCTTAAAATAAA\n")
                fh.write(">ENST_NOCDS.1|gene|lncRNA\n")
                fh.write("GCCACCATGGCTTAAAATAAA\n")

            result = build_public_corpus(data_dir=raw_dir, out_dir=out_dir)
            self.assertEqual(result.n_raw, 2)
            self.assertEqual(result.n_clean, 1)
            self.assertTrue(os.path.exists(result.records_path))
            self.assertTrue(os.path.exists(result.manifest_path))

            records = load_records_jsonl(result.records_path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].transcript_id, "ENST_VALID.1")
            self.assertNotIn("T", records[0].seq)
            self.assertEqual(records[0].cds, "AUGGCUUAA")

            with open(result.manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            self.assertEqual(manifest["dataset"]["name"], "gencode_human_transcripts")
            self.assertEqual(manifest["clean_summary"]["n_records"], 1)
            self.assertEqual(manifest["cleaning_drop_counts"]["bad_start_codon"], 1)
            self.assertIn("preprocessing_contract", manifest)
            self.assertEqual(len(manifest["records_sha256"]), 64)

    def test_build_refseq_genbank_corpus_manifest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mef_refseq_") as tmp:
            raw_dir = os.path.join(tmp, "raw")
            out_dir = os.path.join(tmp, "processed")
            os.makedirs(raw_dir, exist_ok=True)
            gbff_path = os.path.join(raw_dir, "human.1.rna.gbff.gz")
            with gzip.open(gbff_path, "wt", encoding="utf-8") as fh:
                fh.write(
                    "LOCUS       NM_VALID                 22 bp    mRNA    linear   PRI 01-JAN-2000\n"
                    "ACCESSION   NM_VALID\n"
                    "VERSION     NM_VALID.1\n"
                    "FEATURES             Location/Qualifiers\n"
                    "     source          1..22\n"
                    "     CDS             7..15\n"
                    "                     /protein_id=\"NP_VALID.1\"\n"
                    "ORIGIN\n"
                    "        1 gccaccatgg cttaaaataa a\n"
                    "//\n"
                    "LOCUS       NM_BAD                   22 bp    mRNA    linear   PRI 01-JAN-2000\n"
                    "ACCESSION   NM_BAD\n"
                    "VERSION     NM_BAD.1\n"
                    "FEATURES             Location/Qualifiers\n"
                    "     source          1..22\n"
                    "     CDS             7..15\n"
                    "ORIGIN\n"
                    "        1 gccacccccg cttaaaataa a\n"
                    "//\n"
                    "LOCUS       NM_SKIP                  22 bp    mRNA    linear   PRI 01-JAN-2000\n"
                    "ACCESSION   NM_SKIP\n"
                    "VERSION     NM_SKIP.1\n"
                    "FEATURES             Location/Qualifiers\n"
                    "     source          1..22\n"
                    "     CDS             complement(7..15)\n"
                    "ORIGIN\n"
                    "        1 gccaccatgg cttaaaataa a\n"
                    "//\n"
                )

            result = build_public_corpus(
                dataset_name="refseq_human_rna",
                data_dir=raw_dir,
                out_dir=out_dir,
            )
            self.assertEqual(result.n_raw, 2)
            self.assertEqual(result.n_clean, 1)

            records = load_records_jsonl(result.records_path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].transcript_id, "NM_VALID.1")
            self.assertEqual(records[0].cds, "AUGGCUUAA")

            with open(result.manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            self.assertEqual(manifest["dataset"]["name"], "refseq_human_rna")
            self.assertEqual(manifest["raw_summary"]["n_records"], 2)
            self.assertEqual(manifest["clean_summary"]["n_records"], 1)
            self.assertEqual(manifest["cleaning_drop_counts"]["bad_start_codon"], 1)
            self.assertEqual(len(manifest["records_sha256"]), 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
