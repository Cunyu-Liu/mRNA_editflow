"""Offline tests for downstream mRNA-EditFlow data preparation.

Uses stdlib :mod:`unittest` only. These tests exercise the Task 2 downstream
data modules without scipy/biopython/ViennaRNA/mmseqs2 or internet access.
"""
from __future__ import annotations

import math
import os
import tempfile
import unittest

from mrna_editflow.core.constants import translate
from mrna_editflow.data.augment import (
    motif_preserving_utr_perturb,
    protein_identity,
    synonymously_perturb_cds,
)
from mrna_editflow.data.download_mrna import synthesize_corpus
from mrna_editflow.data.element_library import find_motifs, insert_element
from mrna_editflow.data.prepare_codon import prepare_codon_dataset
from mrna_editflow.data.prepare_mpra import prepare_mpra_dataset
from mrna_editflow.data.prepare_ortholog import (
    REGION_CDS,
    prepare_ortholog_pairs,
    validate_ortholog_pair,
)
from mrna_editflow.data.prepare_varlen_tasks import (
    TASK_T5,
    TASK_T6,
    TASK_T7,
    levenshtein_distance,
    prepare_varlen_task_pairs,
)


class DownstreamDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = synthesize_corpus(12, seed=20260711)

    def test_mpra_reads_table_normalises_mrl_and_preserves_official_split(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mef_mpra_test_") as tmp:
            path = os.path.join(tmp, "mpra.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("sequence,mrl,split\n")
                fh.write("ATGCGT,10,train\n")
                fh.write("TTTTAA,20,val\n")
                fh.write("CCCGGG,30,test\n")

            samples = prepare_mpra_dataset(path=path)

        self.assertEqual([s["split"] for s in samples], ["train", "val", "test"])
        self.assertTrue(all("T" not in s["sequence"] for s in samples))
        z = [float(s["mrl_z"]) for s in samples]
        self.assertTrue(all(math.isfinite(v) for v in z))
        self.assertAlmostEqual(sum(z), 0.0, places=7)
        self.assertAlmostEqual(sum(v * v for v in z) / len(z), 1.0, places=7)

        fallback = prepare_mpra_dataset(n_synthetic=5, seed=7)
        self.assertEqual(len(fallback), 5)
        self.assertTrue(all(math.isfinite(float(s["mrl_z"])) for s in fallback))

    def test_codon_samples_translate_and_split_by_protein(self) -> None:
        samples = prepare_codon_dataset(records=self.records, seed=19)
        self.assertEqual(len(samples), len(self.records))
        protein_to_split = {}
        for sample in samples:
            self.assertEqual(translate(sample["cds"]), sample["protein"] + "*")
            self.assertGreaterEqual(float(sample["cai"]), 0.0)
            self.assertLessEqual(float(sample["cai"]), 1.0)
            self.assertGreaterEqual(float(sample["gc"]), 0.0)
            self.assertLessEqual(float(sample["gc"]), 1.0)
            self.assertGreaterEqual(float(sample["gc3"]), 0.0)
            self.assertLessEqual(float(sample["gc3"]), 1.0)
            self.assertTrue(math.isfinite(float(sample["mfe_proxy"])))
            protein = sample["protein"]
            split = sample["split"]
            protein_to_split.setdefault(protein, split)
            self.assertEqual(protein_to_split[protein], split)

    def test_synonymous_perturbation_keeps_protein_identity_100_percent(self) -> None:
        cds = "AUG" + "GCU" * 5 + "CUU" * 5 + "UAA"
        mutated = synonymously_perturb_cds(cds, edit_fraction=0.8, seed=5)
        self.assertNotEqual(mutated, cds)
        self.assertEqual(len(mutated), len(cds))
        self.assertEqual(mutated[:3], "AUG")
        self.assertEqual(mutated[-3:], "UAA")
        self.assertEqual(translate(mutated), translate(cds))
        self.assertEqual(protein_identity(translate(cds), translate(mutated)), 1.0)

        utr = "AAAA" + "GCCACC" + "CCCC" + "AUUUA" + "GGGG"
        perturbed = motif_preserving_utr_perturb(
            utr, protected_motifs=("GCCACC", "AUUUA"), edit_fraction=0.4, seed=3
        )
        self.assertIn("GCCACC", perturbed)
        self.assertIn("AUUUA", perturbed)

    def test_ortholog_pairs_are_region_consistent_and_cds_safe(self) -> None:
        pairs = prepare_ortholog_pairs(records=self.records[:6], n_pairs=6, seed=13)
        self.assertEqual(len(pairs), 6)
        self.assertTrue(all(validate_ortholog_pair(pair) for pair in pairs))
        cds_pairs = [p for p in pairs if p["region"] == REGION_CDS]
        self.assertGreaterEqual(len(cds_pairs), 1)
        for pair in cds_pairs:
            self.assertEqual(pair["source_region"], REGION_CDS)
            self.assertEqual(pair["target_region"], REGION_CDS)
            self.assertEqual(len(pair["source_seq"]) % 3, 0)
            self.assertEqual(len(pair["target_seq"]) % 3, 0)
            self.assertEqual(translate(pair["source_seq"]), translate(pair["target_seq"]))
            self.assertEqual((pair["source_start"] - pair["source_cds_start"]) % 3, 0)
            self.assertEqual((pair["source_end"] - pair["source_cds_start"]) % 3, 0)

    def test_varlen_t5_t6_t7_samples_have_constraints(self) -> None:
        record = self.records[0]
        tasks = prepare_varlen_task_pairs(records=[record], seed=23)
        groups = {t["task_group"] for t in tasks}
        self.assertTrue({"T5", "T6", "T7"} <= groups)
        self.assertTrue(any(t["constraints"].get("action") == "insert" for t in tasks))
        self.assertTrue(any(t["constraints"].get("action") == "remove" for t in tasks))

        for task in tasks:
            source = task["source"]
            target = task["target"]
            constraints = task["constraints"]
            self.assertIsNotNone(target)
            budget = levenshtein_distance(source, target)
            self.assertEqual(constraints["minimal_edit_budget"], budget)
            self.assertGreaterEqual(constraints["max_edit_budget"], budget)

        t5 = next(t for t in tasks if t["task_id"] == TASK_T5)
        target_cds = t5["target"][len(record.five_utr):len(record.five_utr) + len(record.cds)]
        self.assertEqual(translate(target_cds), translate(record.cds))
        self.assertTrue(t5["constraints"]["protein_identity_required"])

        t6 = next(t for t in tasks if t["task_id"] == TASK_T6)
        self.assertEqual(len(t6["target"]), t6["constraints"]["length_target"])
        self.assertEqual(t6["constraints"]["region"], "5UTR")

        t7_insert = next(
            t for t in tasks
            if t["task_id"] == TASK_T7 and t["constraints"].get("action") == "insert"
        )
        target_three = t7_insert["target"][record.cds_end:]
        hits = find_motifs(target_three)
        self.assertTrue(any(h["family"] == "polyA" for h in hits))

    def test_motif_insertion_is_detectable(self) -> None:
        seq = "ACGUACGU"
        inserted, info = insert_element(seq, "Kozak", position=4)
        hits = find_motifs(inserted)
        self.assertEqual(inserted[info["start"]:info["end"]], info["sequence"])
        self.assertTrue(
            any(
                hit["family"] == "Kozak"
                and hit["start"] == info["start"]
                and hit["end"] == info["end"]
                for hit in hits
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
