"""CPU tests for protein-conditioned CDS design (roadmap upgrade #3).

Run:
    /Users/bytedance/Documents/research/editflow/.venv/bin/python \
        -m unittest mrna_editflow.tests.test_protein_conditioned_cds -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mrna_editflow.baselines.codon_lattice_dp import CodonLatticeDPConfig
from mrna_editflow.baselines.protein_conditioned_cds import (
    design_cds_for_protein,
    run_protein_conditioned_gc_sweep,
    run_protein_conditioned_design,
    seed_cds_from_protein,
    summarize_designs,
)
from mrna_editflow.core.constants import is_valid_cds, translate
from mrna_editflow.eval import audit_protein_conditioned_gc_sweep


class TestProteinConditionedCDS(unittest.TestCase):
    def test_seed_is_valid_and_translates_to_target(self):
        protein = "MAKELVSTG"
        seed, prepended = seed_cds_from_protein(protein)
        self.assertFalse(prepended)  # already starts with Met
        self.assertTrue(is_valid_cds(seed))
        self.assertEqual(translate(seed).rstrip("*"), protein)

    def test_prepends_met_when_missing(self):
        protein = "AKELV"
        seed, prepended = seed_cds_from_protein(protein)
        self.assertTrue(prepended)
        self.assertTrue(is_valid_cds(seed))
        # Leading Met is added, so designed protein is M + target.
        self.assertEqual(translate(seed).rstrip("*"), "M" + protein)

    def test_design_preserves_protein_identity_exactly(self):
        protein = "MAKELVSTGGGCCDEFHIK"
        result = design_cds_for_protein(
            protein,
            config=CodonLatticeDPConfig(
                cai_weight=1.0, gc_weight=0.1, boundary_weight=0.05, target_gc=0.55
            ),
        )
        # Hard constraint: protein identity must be exactly 1.0.
        self.assertEqual(result.protein_identity, 1.0)
        self.assertTrue(is_valid_cds(result.designed_cds))
        self.assertEqual(translate(result.designed_cds).rstrip("*"), protein)
        # Design should not reduce CAI relative to the deterministic seed.
        self.assertGreaterEqual(result.designed_cai + 1e-9, result.seed_cai)

    def test_invalid_amino_acid_raises(self):
        with self.assertRaises(ValueError):
            seed_cds_from_protein("MAKZLV")  # Z is not a standard amino acid
        with self.assertRaises(ValueError):
            seed_cds_from_protein("")

    def test_native_baseline_scores_design_against_real_cds(self):
        # A native CDS made of low-usage synonymous codons: the DP design should
        # match or beat its CAI while still encoding the same protein.
        protein = "MAKELVSTG"
        native_seed, _ = seed_cds_from_protein(protein)  # deterministic valid CDS
        result = design_cds_for_protein(
            protein,
            config=CodonLatticeDPConfig(cai_weight=1.0, gc_weight=0.1, target_gc=0.55),
            native_cds=native_seed,
        )
        # Native baseline fields must be populated and self-consistent.
        self.assertIsNotNone(result.native_cai)
        self.assertIsNotNone(result.native_gc)
        self.assertEqual(result.native_cds, native_seed)
        # Designed CDS still encodes the native protein exactly.
        self.assertEqual(result.native_protein_identity, 1.0)
        # CAI-optimizing design should not fall below the native CDS's CAI.
        self.assertGreaterEqual(result.designed_cai + 1e-9, result.native_cai)
        # Summary surfaces the native comparison stats.
        summary = summarize_designs([result])
        self.assertEqual(summary["n_with_native"], 1)
        self.assertIn("mean_designed_vs_native_cai_delta", summary)
        self.assertEqual(summary["designed_cai_ge_native_fraction"], 1.0)
        self.assertEqual(summary["native_protein_identity_eq_1_fraction"], 1.0)

    def test_run_from_records_with_native_baseline(self):
        # Two synthetic records; design-from-protein must recover each protein and
        # the native baseline stats must appear in the summary.
        with tempfile.TemporaryDirectory(prefix="mef_prot_native_") as tmp:
            recs = os.path.join(tmp, "recs.jsonl")
            with open(recs, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"transcript_id": "r1", "five_utr": "GCCACC",
                                     "cds": "AUGGCUAAAUAA", "three_utr": "GGAA"}) + "\n")
                fh.write(json.dumps({"transcript_id": "r2", "five_utr": "ACCUCC",
                                     "cds": "AUGCCCGGGUAA", "three_utr": "UGCA"}) + "\n")
            out_jsonl = os.path.join(tmp, "designs.jsonl")
            out_json = os.path.join(tmp, "summary.json")
            progress_jsonl = os.path.join(tmp, "progress.jsonl")
            payload = run_protein_conditioned_design(
                records_jsonl=recs,
                out_jsonl=out_jsonl,
                out_json=out_json,
                use_native_baseline=True,
                config=CodonLatticeDPConfig(cai_weight=1.0, gc_weight=0.1),
                progress_jsonl=progress_jsonl,
                progress_every=1,
            )
            self.assertTrue(payload["uses_native_baseline"])
            self.assertEqual(payload["progress_jsonl"], progress_jsonl)
            summary = payload["summary"]
            self.assertEqual(summary["n_with_native"], 2)
            self.assertEqual(summary["mean_protein_identity"], 1.0)
            self.assertEqual(summary["native_protein_identity_eq_1_fraction"], 1.0)
            with open(progress_jsonl, "r", encoding="utf-8") as fh:
                events = [json.loads(line)["event"] for line in fh if line.strip()]
            self.assertIn("protein_design_start", events)
            self.assertIn("protein_design_progress", events)
            self.assertEqual(events[-1], "protein_design_complete")
            with open(out_jsonl, "r", encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            for row in rows:
                self.assertIsNotNone(row["native_cds"])
                self.assertIsNotNone(row["native_cai"])

    def test_gc_weight_sweep_writes_pareto_frontier_artifacts(self):
        with tempfile.TemporaryDirectory(prefix="mef_prot_gc_sweep_") as tmp:
            recs = os.path.join(tmp, "recs.jsonl")
            with open(recs, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"transcript_id": "r1", "five_utr": "GCCACC",
                                     "cds": "AUGGCUAAAUAA", "three_utr": "GGAA"}) + "\n")
                fh.write(json.dumps({"transcript_id": "r2", "five_utr": "ACCUCC",
                                     "cds": "AUGCCCGGGUAA", "three_utr": "UGCA"}) + "\n")
            out_jsonl = os.path.join(tmp, "sweep.jsonl")
            out_json = os.path.join(tmp, "sweep.json")
            out_md = os.path.join(tmp, "sweep.md")
            payload = run_protein_conditioned_gc_sweep(
                gc_weights=[0.0, 0.5, 4.0],
                records_jsonl=recs,
                out_jsonl=out_jsonl,
                out_json=out_json,
                out_md=out_md,
                use_native_baseline=True,
                config=CodonLatticeDPConfig(cai_weight=1.0, target_gc=0.55),
            )
            self.assertEqual(payload["sweep_kind"], "protein_conditioned_cai_gc_pareto")
            self.assertEqual(len(payload["points"]), 3)
            self.assertTrue(payload["pareto_front_gc_weights"])
            for point in payload["points"]:
                summary = point["summary"]
                self.assertIn("mean_abs_gc_error", summary)
                self.assertEqual(summary["protein_identity_eq_1_fraction"], 1.0)
                self.assertIn("pareto_rank", point)
                self.assertIn("is_pareto_front", point)
            self.assertTrue(all(point["pareto_rank"] == 0 for point in payload["pareto_front"]))
            with open(out_jsonl, "r", encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual(len(rows), 6)
            self.assertEqual({row["gc_weight"] for row in rows}, {0.0, 0.5, 4.0})
            with open(out_json, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded["artifact_contract"]["hard_constraint"],
                             "protein_identity_eq_1_fraction must remain 1.0")
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("CAI-GC Pareto Sweep", text)
            self.assertIn("| gc_weight |", text)

    def test_gc_weight_sweep_audit_accepts_complete_artifacts(self):
        with tempfile.TemporaryDirectory(prefix="mef_prot_gc_audit_") as tmp:
            recs = os.path.join(tmp, "recs.jsonl")
            with open(recs, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"transcript_id": "r1", "five_utr": "GCCACC",
                                     "cds": "AUGGCUAAAUAA", "three_utr": "GGAA"}) + "\n")
                fh.write(json.dumps({"transcript_id": "r2", "five_utr": "ACCUCC",
                                     "cds": "AUGCCCGGGUAA", "three_utr": "UGCA"}) + "\n")
            out_jsonl = os.path.join(tmp, "sweep.jsonl")
            out_json = os.path.join(tmp, "sweep.json")
            out_md = os.path.join(tmp, "sweep.md")
            run_protein_conditioned_gc_sweep(
                gc_weights=[0.0, 1.0, 4.0],
                records_jsonl=recs,
                out_jsonl=out_jsonl,
                out_json=out_json,
                out_md=out_md,
                use_native_baseline=True,
                config=CodonLatticeDPConfig(cai_weight=1.0, target_gc=0.55),
            )
            audit_json = os.path.join(tmp, "audit.json")
            audit_md = os.path.join(tmp, "audit.md")
            payload = audit_protein_conditioned_gc_sweep.audit_protein_conditioned_gc_sweep(
                summary_json=out_json,
                jsonl_path=out_jsonl,
                md_path=out_md,
                project_root=tmp,
                out_json=audit_json,
                out_md=audit_md,
            )
            self.assertEqual(payload["artifact_kind"], "protein_conditioned_gc_sweep_audit")
            self.assertTrue(payload["summary"]["ready_for_pareto_claim_audit"])
            self.assertTrue(payload["summary"]["all_points_identity_exact_1"])
            self.assertTrue(payload["summary"]["all_point_metrics_finite"])
            self.assertTrue(payload["summary"]["pareto_metadata_ok"])
            self.assertTrue(payload["summary"]["jsonl_row_count_ok"])
            with open(audit_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("Ready for Pareto claim audit: True", text)

    def test_gc_weight_sweep_audit_reports_pending_when_summary_missing(self):
        with tempfile.TemporaryDirectory(prefix="mef_prot_gc_audit_pending_") as tmp:
            payload = audit_protein_conditioned_gc_sweep.audit_protein_conditioned_gc_sweep(
                summary_json=os.path.join(tmp, "missing.summary.json"),
                jsonl_path=os.path.join(tmp, "missing.jsonl"),
                md_path=os.path.join(tmp, "missing.md"),
                project_root=tmp,
            )
            self.assertFalse(payload["summary"]["ready_for_pareto_claim_audit"])
            self.assertFalse(payload["summary"]["summary_exists"])
            self.assertEqual(len(payload["summary"]["missing_artifacts"]), 3)

    def test_batch_run_writes_artifacts_and_all_identity_one(self):
        proteins = ["MAKELVSTG", "MGGCCDEFH", "MHIKLMNPQ"]
        with tempfile.TemporaryDirectory(prefix="mef_prot_cds_") as tmp:
            out_jsonl = os.path.join(tmp, "designs.jsonl")
            out_json = os.path.join(tmp, "designs_summary.json")
            payload = run_protein_conditioned_design(
                proteins=proteins,
                out_jsonl=out_jsonl,
                out_json=out_json,
                config=CodonLatticeDPConfig(cai_weight=1.0, gc_weight=0.1),
            )
            self.assertTrue(os.path.exists(out_jsonl))
            self.assertTrue(os.path.exists(out_json))
            summary = payload["summary"]
            self.assertEqual(summary["n"], 3)
            # Every designed CDS must exactly preserve its target protein.
            self.assertEqual(summary["mean_protein_identity"], 1.0)
            self.assertEqual(summary["protein_identity_eq_1_fraction"], 1.0)
            with open(out_jsonl, "r", encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual(len(rows), 3)
            for row in rows:
                self.assertTrue(is_valid_cds(row["designed_cds"]))
                self.assertEqual(row["protein_identity"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
