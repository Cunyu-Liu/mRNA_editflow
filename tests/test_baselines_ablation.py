"""Offline smoke tests for Task 6 baselines and ablation scaffolding.

Run:
    /Users/bytedance/Documents/research/editflow/.venv/bin/python \
        -m unittest mrna_editflow.tests.test_baselines_ablation -v
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mrna_editflow.baselines.ar_lm import ARLMConfig, sample_ar_lm, train_ar_lm
from mrna_editflow.baselines.codon_lattice_dp import (
    CodonLatticeDPConfig,
    optimize_cds_synonymous,
    run_codon_lattice_dp,
)
from mrna_editflow.baselines.utr_local_search import (
    UTRLocalSearchConfig,
    optimize_record_five_utr,
    run_utr_local_search,
)
from mrna_editflow.baselines.hybrid_teacher_export import export_hybrid_teacher_jsonl
from mrna_editflow.baselines.cascade_hard_negative_teacher import mine_cascade_hard_negative_teacher
from mrna_editflow.baselines.utr_teacher_export import (
    export_utr_teacher_jsonl,
    score_record_utr_teacher_rows,
)
from mrna_editflow.baselines.multiobjective_teacher_export import (
    OBJECTIVE_LABELS,
    FUSION_MODES,
    MultiObjectiveConfig,
    export_multiobjective_teacher_jsonl,
    fast_non_dominated_sort,
    _zscore_standardize,
    score_record_multiobjective_rows,
)
from mrna_editflow.baselines.external_models import (
    available_external_models,
    get_external_result,
    list_external_results,
)
from mrna_editflow.baselines.external_sota_dry_run import dry_run_external_sota
from mrna_editflow.baselines.external_sota_input_pack import (
    EXTERNAL_BENCHMARK_MODELS,
    build_external_sota_input_pack,
    external_metric_schema,
)
from mrna_editflow.baselines.external_lineardesign_adapter import run_lineardesign_adapter
from mrna_editflow.baselines.external_ensembledesign_adapter import (
    run_ensembledesign_adapter,
)
from mrna_editflow.baselines.external_codongpt_adapter import (
    run_codongpt_adapter,
)
from mrna_editflow.baselines.external_utrgan_adapter import run_utrgan_adapter
from mrna_editflow.baselines.external_utailor_adapter import run_utailor_adapter
from mrna_editflow.baselines.masked_diffusion import (
    MaskedDiffusionConfig,
    make_fixed_canvas_batch,
    sample_masked_diffusion,
    train_masked_diffusion,
)
from mrna_editflow.baselines.mrnabench_probe import (
    MRNABenchProbeConfig,
    run_mrnabench_probe,
)
from mrna_editflow.core.constants import CODON_TABLE, NUC_TO_ID, is_valid_cds, translate
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.sample import model_guided_edit_record


def _tiny_records():
    return [
        MRNARecord("r1", "GCCACC", "AUGGCUUAA", "AAUAAA"),
        MRNARecord("r2", "ACCUCC", "AUGGCCUAA", "UGCAAA"),
        MRNARecord("r3", "UUCGGA", "AUGAAAUAA", "CCGGAA"),
        MRNARecord("r4", "CGCGAA", "AUGCCCUAA", "AUUAAA"),
    ]


class TestMaskedDiffusionBaseline(unittest.TestCase):
    def test_smoke_train_sample_and_logits_are_finite(self):
        records = _tiny_records()
        cfg = MaskedDiffusionConfig(
            max_len=32,
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            batch_size=2,
            steps=2,
            mask_prob=0.5,
            seed=101,
        )
        result = train_masked_diffusion(records, cfg, device="cpu")
        self.assertEqual(len(result.losses), 2)
        self.assertTrue(math.isfinite(result.final_loss))

        seq = sample_masked_diffusion(result.model, length=18, denoise_steps=2, seed=102)
        self.assertEqual(len(seq), 18)
        self.assertLessEqual(set(seq), set("ACGU"))

        batch = make_fixed_canvas_batch(records[:2], max_len=32)
        with torch.no_grad():
            logits = result.model(
                batch["token_ids"],
                region_ids=batch["region_ids"],
                padding_mask=batch["padding_mask"],
            )
        self.assertEqual(tuple(logits.shape), (2, 32, 4))
        self.assertTrue(bool(torch.isfinite(logits).all()))


class TestAutoregressiveLMBaseline(unittest.TestCase):
    def test_smoke_train_sample_and_loss_are_finite(self):
        cfg = ARLMConfig(
            max_len=32,
            hidden_dim=16,
            num_layers=1,
            batch_size=2,
            steps=2,
            seed=201,
        )
        result = train_ar_lm(_tiny_records(), cfg, device="cpu")
        self.assertEqual(len(result.losses), 2)
        self.assertTrue(math.isfinite(result.final_loss))

        seq = sample_ar_lm(result.model, length=18, seed=202)
        self.assertEqual(len(seq), 18)
        self.assertLessEqual(set(seq), set("ACGU"))


class TestCodonLatticeDPBaseline(unittest.TestCase):
    def test_optimizes_synonymous_codon_without_changing_protein(self):
        weights = {codon: 0.1 for codon, aa in CODON_TABLE.items() if aa != "*"}
        weights["GCU"] = 0.01
        weights["GCC"] = 1.0
        result = optimize_cds_synonymous(
            "AUGGCUUAA",
            config=CodonLatticeDPConfig(gc_weight=0.0, boundary_weight=0.0),
            codon_weights=weights,
        )
        self.assertEqual(result.optimized_cds, "AUGGCCUAA")
        self.assertEqual(result.source_protein, result.optimized_protein)
        self.assertGreater(result.optimized_cai, result.source_cai)
        self.assertEqual(result.codon_changes, 1)

    def test_respects_codon_change_budget(self):
        weights = {codon: 0.1 for codon, aa in CODON_TABLE.items() if aa != "*"}
        weights["GCC"] = 1.0
        result = optimize_cds_synonymous(
            "AUGGCUUAA",
            config=CodonLatticeDPConfig(
                gc_weight=0.0,
                boundary_weight=0.0,
                max_codon_changes=0,
            ),
            codon_weights=weights,
        )
        self.assertEqual(result.optimized_cds, "AUGGCUUAA")
        self.assertEqual(result.codon_changes, 0)

    def test_unbudgeted_fast_path_matches_explicit_full_budget(self):
        weights = {codon: 0.1 for codon, aa in CODON_TABLE.items() if aa != "*"}
        weights.update({"GCC": 1.0, "AAG": 0.9, "CCG": 0.95})
        cds = "AUGGCUAAACCCUAA"
        unbudgeted = optimize_cds_synonymous(
            cds,
            config=CodonLatticeDPConfig(gc_weight=0.1, boundary_weight=0.05),
            codon_weights=weights,
        )
        full_budget = optimize_cds_synonymous(
            cds,
            config=CodonLatticeDPConfig(
                gc_weight=0.1,
                boundary_weight=0.05,
                max_codon_changes=len(cds) // 3,
            ),
            codon_weights=weights,
        )
        self.assertEqual(unbudgeted.optimized_cds, full_budget.optimized_cds)
        self.assertEqual(unbudgeted.codon_changes, full_budget.codon_changes)
        self.assertAlmostEqual(unbudgeted.objective, full_budget.objective)

    def test_runner_writes_valid_records_and_summary(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory(prefix="mef_codon_dp_") as tmp:
            out_jsonl = os.path.join(tmp, "optimized.jsonl")
            out_json = os.path.join(tmp, "summary.json")
            payload = run_codon_lattice_dp(
                _tiny_records(),
                out_jsonl=out_jsonl,
                out_json=out_json,
                limit=3,
                config=CodonLatticeDPConfig(max_codon_changes=1),
            )
            self.assertTrue(os.path.exists(out_jsonl))
            self.assertTrue(os.path.exists(out_json))
            self.assertEqual(payload["summary"]["n"], 3)
            with open(out_jsonl, "r", encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual(len(rows), 3)
            for src, row in zip(_tiny_records(), rows):
                self.assertTrue(is_valid_cds(row["cds"]))
                self.assertEqual(translate(src.cds), translate(row["cds"]))


class TestUTRLocalSearchBaseline(unittest.TestCase):
    def test_optimizes_five_utr_without_touching_cds_or_three_utr(self):
        record = MRNARecord("bad_utr", "AUGGGGCCCC", "AUGGCCUAA", "AAUAAA")
        optimized, result = optimize_record_five_utr(
            record,
            config=UTRLocalSearchConfig(
                edit_budget=2,
                beam_width=8,
                start_window_nt=20,
                max_edit_positions=20,
            ),
        )
        self.assertEqual(optimized.cds, record.cds)
        self.assertEqual(optimized.three_utr, record.three_utr)
        self.assertLessEqual(result.utr_edit_distance, 2)
        self.assertGreaterEqual(result.optimized_te, result.source_te)
        self.assertEqual(result.optimized_five_utr, optimized.five_utr)

    def test_runner_writes_valid_utr_baseline_artifacts(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory(prefix="mef_utr_search_") as tmp:
            out_jsonl = os.path.join(tmp, "optimized.jsonl")
            out_json = os.path.join(tmp, "summary.json")
            payload = run_utr_local_search(
                _tiny_records(),
                out_jsonl=out_jsonl,
                out_json=out_json,
                limit=3,
                config=UTRLocalSearchConfig(
                    edit_budget=1,
                    beam_width=6,
                    start_window_nt=12,
                    max_edit_positions=12,
                ),
            )
            self.assertTrue(os.path.exists(out_jsonl))
            self.assertTrue(os.path.exists(out_json))
            self.assertEqual(payload["summary"]["n"], 3)
            self.assertEqual(payload["summary"]["cds_unchanged_fraction"], 1.0)
            self.assertEqual(payload["summary"]["three_utr_unchanged_fraction"], 1.0)
            with open(out_jsonl, "r", encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual(len(rows), 3)
            for src, row in zip(_tiny_records(), rows):
                self.assertEqual(row["cds"], src.cds)
                self.assertEqual(row["three_utr"], src.three_utr)


class TestEditableRegionConstraint(unittest.TestCase):
    class _RegionBiasedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(
            self,
            token_ids,
            region_ids,
            phase_ids,
            t,
            padding_mask,
            backbone,
        ):
            length = token_ids.shape[1]
            rates = torch.zeros((1, length, 3), device=token_ids.device)
            rates[0, :, 1] = torch.arange(
                1,
                length + 1,
                dtype=torch.float32,
                device=token_ids.device,
            )
            sub_probs = torch.zeros((1, length, 4), device=token_ids.device)
            sub_probs[0, :, NUC_TO_ID["G"]] = 1.0
            return {"rates": rates, "sub_probs": sub_probs}

    def test_utr5_constraint_excludes_higher_scoring_three_utr_positions(self):
        record = MRNARecord("r", "AAA", "AUGGCCUAA", "CCC")
        model = self._RegionBiasedModel()
        restricted = model_guided_edit_record(
            record,
            model,
            object(),
            task_id="T5",
            edit_budget=1,
            proposal_top_k=1,
            proposal_temperature=0.0,
            editable_regions=("utr5",),
            device="cpu",
        )
        unrestricted = model_guided_edit_record(
            record,
            model,
            object(),
            task_id="T5",
            edit_budget=1,
            proposal_top_k=1,
            proposal_temperature=0.0,
            device="cpu",
        )
        self.assertNotEqual(restricted.five_utr, record.five_utr)
        self.assertEqual(restricted.three_utr, record.three_utr)
        self.assertEqual(restricted.cds, record.cds)
        self.assertEqual(unrestricted.five_utr, record.five_utr)
        self.assertNotEqual(unrestricted.three_utr, record.three_utr)


class TestUTRTeacherExport(unittest.TestCase):
    def test_exports_ranker_compatible_one_step_teacher_rows(self):
        record = MRNARecord("teacher_utr", "AUGGGGCCCC", "AUGGCCUAA", "AAUAAA")
        rows = score_record_utr_teacher_rows(
            record,
            config=UTRLocalSearchConfig(
                edit_budget=1,
                max_length_delta=1,
                start_window_nt=20,
                max_edit_positions=20,
            ),
            candidate_cap=8,
        )
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row.transcript_id == "teacher_utr" for row in rows))
        self.assertTrue({row.op for row in rows} <= {"sub", "ins", "del"})
        self.assertTrue(all(row.utr_edit_distance == 1 for row in rows))
        self.assertTrue(all(row.candidate["cds"] == record.cds for row in rows))
        self.assertTrue(all(row.candidate["three_utr"] == record.three_utr for row in rows))
        self.assertGreater(max(row.teacher_score for row in rows), min(row.teacher_score for row in rows))

    def test_teacher_export_writes_jsonl_and_summary(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory(prefix="mef_utr_teacher_") as tmp:
            out_jsonl = os.path.join(tmp, "teacher.jsonl")
            out_json = os.path.join(tmp, "teacher_summary.json")
            payload = export_utr_teacher_jsonl(
                _tiny_records(),
                out_jsonl=out_jsonl,
                out_json=out_json,
                limit=2,
                config=UTRLocalSearchConfig(
                    edit_budget=1,
                    max_length_delta=1,
                    start_window_nt=12,
                    max_edit_positions=12,
                ),
                candidate_cap=10,
            )
            self.assertTrue(os.path.exists(out_jsonl))
            self.assertTrue(os.path.exists(out_json))
            self.assertEqual(payload["summary"]["n_records"], 2)
            self.assertEqual(payload["summary"]["n_rows"], 20)
            self.assertGreaterEqual(payload["summary"]["mean_best_teacher_score"], 0.0)
            with open(out_jsonl, "r", encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual(len(rows), 20)
            for row in rows:
                self.assertIn(row["op"], {"sub", "ins", "del"})
                self.assertIn("teacher_score", row)
                self.assertIn("candidate_te", row)
                self.assertEqual(row["task_id"], "T5")


class TestMultiObjectiveTeacherExport(unittest.TestCase):
    def test_rows_carry_normalized_source_scores_and_fused_teacher(self):
        record = MRNARecord("mo_utr", "AUGGGGCCCCUUU", "AUGGCCUAA", "AAUAAA")
        rows = score_record_multiobjective_rows(
            record,
            config=MultiObjectiveConfig(
                start_window_nt=24,
                max_edit_positions=24,
                candidate_cap=12,
            ),
        )
        self.assertTrue(rows)
        self.assertLessEqual(len(rows), 12)
        for row in rows:
            self.assertEqual(row.transcript_id, "mo_utr")
            self.assertIn(row.op, {"sub", "ins", "del"})
            # source_scores must expose every objective label, normalized to [0,1].
            self.assertEqual(set(row.source_scores), set(OBJECTIVE_LABELS))
            for label in OBJECTIVE_LABELS:
                value = row.source_scores[label]
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)
            self.assertEqual(set(row.objective_deltas), set(OBJECTIVE_LABELS))
            # Fused teacher score is a convex combination -> also within [0,1].
            self.assertGreaterEqual(row.teacher_score, -1e-9)
            self.assertLessEqual(row.teacher_score, 1.0 + 1e-9)
        # The fused ranking must not be degenerate across the candidate pool.
        self.assertGreater(
            max(row.teacher_score for row in rows),
            min(row.teacher_score for row in rows),
        )

    def test_weights_are_renormalized_and_default_covers_all_objectives(self):
        cfg = MultiObjectiveConfig(weights={"te": 2.0, "gc": 2.0})
        weights = cfg.normalized_weights()
        self.assertEqual(set(weights), set(OBJECTIVE_LABELS))
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        self.assertAlmostEqual(weights["te"], 0.5, places=6)
        self.assertAlmostEqual(weights["gc"], 0.5, places=6)
        self.assertAlmostEqual(weights["mrl"], 0.0, places=6)

    def test_export_is_consumable_by_source_balanced_ranker(self):
        import json
        import tempfile

        from mrna_editflow.train_proposal_ranker import load_teacher_rows

        with tempfile.TemporaryDirectory(prefix="mef_mo_teacher_") as tmp:
            out_jsonl = os.path.join(tmp, "mo_teacher.jsonl")
            out_json = os.path.join(tmp, "mo_teacher_summary.json")
            payload = export_multiobjective_teacher_jsonl(
                _tiny_records(),
                out_jsonl=out_jsonl,
                out_json=out_json,
                limit=3,
                config=MultiObjectiveConfig(
                    start_window_nt=12,
                    max_edit_positions=12,
                    candidate_cap=8,
                ),
            )
            self.assertTrue(os.path.exists(out_jsonl))
            self.assertTrue(os.path.exists(out_json))
            self.assertEqual(payload["summary"]["objectives"], list(OBJECTIVE_LABELS))
            self.assertGreater(payload["summary"]["n_rows"], 0)
            # Rows must be loadable by the ranker and expose per-objective labels.
            grouped = load_teacher_rows(out_jsonl)
            self.assertTrue(grouped)
            for rows in grouped.values():
                for row in rows:
                    self.assertEqual(set(row.source_scores), set(OBJECTIVE_LABELS))
            with open(out_jsonl, "r", encoding="utf-8") as fh:
                raw = [json.loads(line) for line in fh if line.strip()]
            for row in raw:
                self.assertIn("source_scores", row)
                self.assertIn("objective_deltas", row)
                self.assertEqual(row["task_id"], "T5")


class TestParetoFusion(unittest.TestCase):
    def test_fast_non_dominated_sort_matches_known_fronts(self):
        # 2D maximization: (1,0) and (0,1) are mutually non-dominated; (0.5,0.5)
        # is a tradeoff point also on front 0; (0.2,0.2) is dominated by the
        # tradeoff point (front 1); (0.1,0.1) is dominated again (front 2).
        pts = [(1.0, 0.0), (0.0, 1.0), (0.5, 0.5), (0.2, 0.2), (0.1, 0.1)]
        self.assertEqual(fast_non_dominated_sort(pts), [0, 0, 0, 1, 2])

    def test_fast_non_dominated_sort_edge_cases(self):
        self.assertEqual(fast_non_dominated_sort([]), [])
        self.assertEqual(fast_non_dominated_sort([(0.3, 0.7)]), [0])
        # Identical points never strictly dominate each other -> all front 0.
        self.assertEqual(fast_non_dominated_sort([(0.5, 0.5)] * 4), [0, 0, 0, 0])
        # A strict chain of domination yields consecutive fronts.
        self.assertEqual(
            fast_non_dominated_sort([(0.9, 0.9), (0.5, 0.5), (0.1, 0.1)]),
            [0, 1, 2],
        )

    def test_rows_expose_valid_pareto_metadata(self):
        record = MRNARecord("mo_pareto", "AUGGGGCCCCUUUAAA", "AUGGCCUAA", "AAUAAA")
        rows = score_record_multiobjective_rows(
            record,
            config=MultiObjectiveConfig(
                start_window_nt=24, max_edit_positions=24, candidate_cap=0
            ),
        )
        self.assertTrue(rows)
        ranks = sorted({row.pareto_rank for row in rows})
        # Ranks form a contiguous 0..R range with a non-empty non-dominated front.
        self.assertEqual(ranks[0], 0)
        self.assertEqual(ranks, list(range(len(ranks))))
        front0 = [row for row in rows if row.pareto_rank == 0]
        self.assertTrue(front0)
        for row in rows:
            self.assertGreaterEqual(row.pareto_rank, 0)
            self.assertGreaterEqual(row.pareto_front_size, 1)
            self.assertGreaterEqual(row.pareto_fused_score, 0.0)
            self.assertLessEqual(row.pareto_fused_score, 1.0)
            self.assertGreaterEqual(row.scalar_fused_score, 0.0)
            self.assertLessEqual(row.scalar_fused_score, 1.0)
        # front_size is consistent with the number of rows sharing that rank.
        counts = {}
        for row in rows:
            counts[row.pareto_rank] = counts.get(row.pareto_rank, 0) + 1
        for row in rows:
            self.assertEqual(row.pareto_front_size, counts[row.pareto_rank])

    def test_pareto_fused_score_is_front_major(self):
        # The banded fusion must never let a worse front outrank a better one:
        # every front-0 row scores at least as high as any front-1+ row.
        record = MRNARecord("mo_band", "AUGGGGCCCCUUUAAAGGG", "AUGGCCUAA", "AAUAAA")
        rows = score_record_multiobjective_rows(
            record,
            config=MultiObjectiveConfig(
                start_window_nt=30, max_edit_positions=30, candidate_cap=0
            ),
        )
        by_front = {}
        for row in rows:
            by_front.setdefault(row.pareto_rank, []).append(row.pareto_fused_score)
        fronts = sorted(by_front)
        if len(fronts) >= 2:
            for lo, hi in zip(fronts, fronts[1:]):
                self.assertGreaterEqual(min(by_front[lo]), max(by_front[hi]) - 1e-9)
        best = max(rows, key=lambda r: r.pareto_fused_score)
        self.assertEqual(best.pareto_rank, 0)

    def test_fusion_mode_selects_teacher_score_source(self):
        cfg_kwargs = dict(start_window_nt=24, max_edit_positions=24, candidate_cap=0)
        scalar_rows = score_record_multiobjective_rows(
            MRNARecord("mo_s", "AUGGGGCCCCUUUAAA", "AUGGCCUAA", "AAUAAA"),
            config=MultiObjectiveConfig(fusion_mode="scalar", **cfg_kwargs),
        )
        pareto_rows = score_record_multiobjective_rows(
            MRNARecord("mo_s", "AUGGGGCCCCUUUAAA", "AUGGCCUAA", "AAUAAA"),
            config=MultiObjectiveConfig(fusion_mode="pareto", **cfg_kwargs),
        )
        grpo_rows = score_record_multiobjective_rows(
            MRNARecord("mo_s", "AUGGGGCCCCUUUAAA", "AUGGCCUAA", "AAUAAA"),
            config=MultiObjectiveConfig(fusion_mode="grpo_standardized", **cfg_kwargs),
        )
        self.assertEqual(len(scalar_rows), len(pareto_rows))
        self.assertEqual(len(scalar_rows), len(grpo_rows))
        for row in scalar_rows:
            self.assertAlmostEqual(row.teacher_score, row.scalar_fused_score, places=9)
        for row in pareto_rows:
            self.assertAlmostEqual(row.teacher_score, row.pareto_fused_score, places=9)
        for row in grpo_rows:
            self.assertAlmostEqual(row.teacher_score, row.grpo_fused_score, places=9)

    def test_zscore_standardize_properties(self):
        # Zero-mean by construction; constant and empty inputs are safe.
        z = _zscore_standardize([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(sum(z), 0.0, places=9)
        # Unit population std.
        n = len(z)
        var = sum(v * v for v in z) / n
        self.assertAlmostEqual(var, 1.0, places=6)
        self.assertEqual(_zscore_standardize([7.0, 7.0, 7.0]), [0.0, 0.0, 0.0])
        self.assertEqual(_zscore_standardize([]), [])

    def test_grpo_fusion_is_scale_invariant_per_objective(self):
        # Every row exposes a finite grpo_fused_score alongside scalar/pareto,
        # and FUSION_MODES advertises the new route.
        self.assertIn("grpo_standardized", FUSION_MODES)
        rows = score_record_multiobjective_rows(
            MRNARecord("mo_grpo", "AUGGGGCCCCUUUAAAGGG", "AUGGCCUAA", "AAUAAA"),
            config=MultiObjectiveConfig(
                start_window_nt=30, max_edit_positions=30, candidate_cap=0
            ),
        )
        self.assertTrue(rows)
        for row in rows:
            self.assertTrue(math.isfinite(row.grpo_fused_score))
            self.assertTrue(math.isfinite(row.scalar_fused_score))
            self.assertTrue(math.isfinite(row.pareto_fused_score))
        # The GRPO fusion is centered: mean over the candidate pool is ~0 when
        # objective weights are uniform-ish and each metric is z-scored.
        mean_grpo = sum(r.grpo_fused_score for r in rows) / len(rows)
        self.assertLess(abs(mean_grpo), 0.5)

    def test_fusion_modes_produce_distinct_rankings(self):
        # The three fusion scores must not induce identical candidate orderings;
        # divergence in the mid/tail ranking is what makes the fusion choice
        # matter for Bradley-Terry distillation pairs. We score once (all three
        # scores are always emitted) and rank by each.
        rows = score_record_multiobjective_rows(
            MRNARecord("mo_rank", "AUGGGGCCCCUUUAAAGGGCCCUUU", "AUGGCCUAA", "AAUAAA"),
            config=MultiObjectiveConfig(
                start_window_nt=48, max_edit_positions=48, candidate_cap=0
            ),
        )
        self.assertGreaterEqual(len(rows), 3)

        def order(key):
            return [
                (r.op, r.pos, r.nt)
                for r in sorted(rows, key=lambda x: getattr(x, key), reverse=True)
            ]

        o_scalar = order("scalar_fused_score")
        o_pareto = order("pareto_fused_score")
        o_grpo = order("grpo_fused_score")
        # At least one pair of fusions must differ somewhere in the full ranking.
        self.assertTrue(
            o_scalar != o_grpo or o_scalar != o_pareto or o_pareto != o_grpo,
            "all three fusion modes produced identical rankings",
        )

    def test_invalid_fusion_mode_raises(self):
        with self.assertRaises(ValueError):
            MultiObjectiveConfig(fusion_mode="not_a_mode")

    def test_summary_reports_pareto_statistics(self):
        rows = [
            score_record_multiobjective_rows(
                rec,
                config=MultiObjectiveConfig(
                    start_window_nt=18, max_edit_positions=18, candidate_cap=0
                ),
            )
            for rec in _tiny_records()
        ]
        from mrna_editflow.baselines.multiobjective_teacher_export import (
            summarize_multiobjective_rows,
        )

        summary = summarize_multiobjective_rows(rows)
        for key in (
            "pareto_front0_rows",
            "pareto_front0_fraction",
            "mean_pareto_rank",
            "max_pareto_rank",
            "mean_pareto_fused_score",
        ):
            self.assertIn(key, summary)
        self.assertGreaterEqual(summary["pareto_front0_rows"], 1)
        self.assertGreaterEqual(summary["pareto_front0_fraction"], 0.0)
        self.assertLessEqual(summary["pareto_front0_fraction"], 1.0)

    def test_pareto_export_consumable_by_source_balanced_ranker(self):
        import json
        import tempfile

        from mrna_editflow.train_proposal_ranker import load_teacher_rows

        with tempfile.TemporaryDirectory(prefix="mef_mo_pareto_") as tmp:
            out_jsonl = os.path.join(tmp, "mo_pareto.jsonl")
            out_json = os.path.join(tmp, "mo_pareto_summary.json")
            payload = export_multiobjective_teacher_jsonl(
                _tiny_records(),
                out_jsonl=out_jsonl,
                out_json=out_json,
                limit=4,
                config=MultiObjectiveConfig(
                    start_window_nt=18,
                    max_edit_positions=18,
                    candidate_cap=8,
                    fusion_mode="pareto",
                ),
            )
            self.assertEqual(payload["config"]["fusion_mode"], "pareto")
            grouped = load_teacher_rows(out_jsonl)
            self.assertTrue(grouped)
            for group in grouped.values():
                for row in group:
                    self.assertEqual(set(row.source_scores), set(OBJECTIVE_LABELS))
            with open(out_jsonl, "r", encoding="utf-8") as fh:
                raw = [json.loads(line) for line in fh if line.strip()]
            for row in raw:
                self.assertIn("pareto_rank", row)
                self.assertIn("pareto_fused_score", row)
                self.assertIn("scalar_fused_score", row)


class TestHybridTeacherExport(unittest.TestCase):
    def test_merges_duplicate_teacher_rows_with_weighted_score(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory(prefix="mef_hybrid_teacher_") as tmp:
            full = os.path.join(tmp, "full.jsonl")
            utr = os.path.join(tmp, "utr.jsonl")
            out_jsonl = os.path.join(tmp, "hybrid.jsonl")
            out_json = os.path.join(tmp, "hybrid.json")
            full_rows = [
                {"transcript_id": "r1", "task_id": "T5", "op": "sub", "pos": 1, "nt": "A", "teacher_score": 0.10},
                {"transcript_id": "r1", "task_id": "T5", "op": "sub", "pos": 2, "nt": "C", "teacher_score": -0.20},
            ]
            utr_rows = [
                {"transcript_id": "r1", "task_id": "T5", "op": "sub", "pos": 1, "nt": "A", "teacher_score": 0.30},
                {"transcript_id": "r1", "task_id": "T5", "op": "ins", "pos": 3, "nt": "G", "teacher_score": 0.40},
            ]
            with open(full, "w", encoding="utf-8") as fh:
                for row in full_rows:
                    fh.write(json.dumps(row) + "\n")
            with open(utr, "w", encoding="utf-8") as fh:
                for row in utr_rows:
                    fh.write(json.dumps(row) + "\n")

            payload = export_hybrid_teacher_jsonl(
                full_jsonl=full,
                utr_jsonl=utr,
                out_jsonl=out_jsonl,
                out_json=out_json,
                full_weight=1.0,
                utr_weight=3.0,
                max_rows_per_record=0,
            )
            self.assertEqual(payload["summary"]["n_rows"], 3)
            self.assertEqual(payload["summary"]["overlap_rows"], 1)
            with open(out_jsonl, "r", encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            merged = {
                (row["op"], row["pos"], row["nt"]): row
                for row in rows
            }
            self.assertAlmostEqual(merged[("sub", 1, "A")]["teacher_score"], 0.25)
            self.assertEqual(merged[("sub", 1, "A")]["source_labels"], ["full", "utr"])
            self.assertIn("teacher_score", merged[("ins", 3, "G")])

    def test_hybrid_export_caps_extremes_per_record(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory(prefix="mef_hybrid_cap_") as tmp:
            full = os.path.join(tmp, "full.jsonl")
            utr = os.path.join(tmp, "utr.jsonl")
            rows = [
                {"transcript_id": "r1", "task_id": "T5", "op": "sub", "pos": i, "nt": "A", "teacher_score": float(i)}
                for i in range(6)
            ]
            with open(full, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
            with open(utr, "w", encoding="utf-8") as fh:
                fh.write("")
            out_jsonl = os.path.join(tmp, "hybrid.jsonl")
            payload = export_hybrid_teacher_jsonl(
                full_jsonl=full,
                utr_jsonl=utr,
                out_jsonl=out_jsonl,
                out_json=os.path.join(tmp, "hybrid.json"),
                max_rows_per_record=4,
            )
            self.assertEqual(payload["summary"]["n_rows"], 4)
            with open(out_jsonl, "r", encoding="utf-8") as fh:
                kept = [json.loads(line) for line in fh if line.strip()]
            kept_scores = sorted(row["teacher_score"] for row in kept)
            self.assertEqual(kept_scores, [0.0, 1.0, 4.0, 5.0])


class TestCascadeHardNegativeTeacher(unittest.TestCase):
    def test_mines_rescue_and_precision_source_labels(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory(prefix="mef_cascade_hn_teacher_") as tmp:
            teacher = os.path.join(tmp, "teacher.jsonl")
            cascade = os.path.join(tmp, "cascade.records.jsonl")
            out_jsonl = os.path.join(tmp, "teacher_hn.jsonl")
            out_json = os.path.join(tmp, "teacher_hn.json")
            out_md = os.path.join(tmp, "teacher_hn.md")
            teacher_rows = [
                {
                    "transcript_id": "r_rescue",
                    "task_id": "T5",
                    "op": "sub",
                    "pos": i,
                    "nt": "A",
                    "teacher_score": float(i) / 10.0,
                    "source_scores": {"utr": float(i) / 10.0, "full": -float(i) / 10.0},
                    "source_weights": {"utr": 1.0, "full": 1.0},
                }
                for i in range(4)
            ] + [
                {
                    "transcript_id": "r_precision",
                    "task_id": "T5",
                    "op": "sub",
                    "pos": i,
                    "nt": "C",
                    "teacher_score": -float(i) / 10.0,
                    "source_scores": {"utr": -float(i) / 10.0, "full": float(i) / 10.0},
                    "source_weights": {"utr": 1.0, "full": 1.0},
                }
                for i in range(4)
            ] + [
                {
                    "transcript_id": "r_neutral",
                    "task_id": "T5",
                    "op": "ins",
                    "pos": i,
                    "nt": "G",
                    "teacher_score": float(i),
                }
                for i in range(4)
            ]
            cascade_rows = [
                {
                    "transcript_id": "r_rescue",
                    "cascade_gain_mean": 0.02,
                    "cascade_win_fraction": 0.8,
                    "cascade_loss_fraction": 0.1,
                    "source_te": 0.6,
                    "source_uaug_presence": 1,
                    "source_kozak_score": 0.0,
                    "source_gc": 0.5,
                    "five_utr_len": 128,
                },
                {
                    "transcript_id": "r_precision",
                    "cascade_gain_mean": -0.02,
                    "cascade_win_fraction": 0.1,
                    "cascade_loss_fraction": 0.8,
                    "source_te": 0.9,
                    "source_uaug_presence": 0,
                    "source_kozak_score": 1.0,
                    "source_gc": 0.6,
                    "five_utr_len": 64,
                },
                {
                    "transcript_id": "r_neutral",
                    "cascade_gain_mean": 0.0,
                    "cascade_win_fraction": 0.5,
                    "cascade_loss_fraction": 0.5,
                    "source_te": 0.7,
                    "source_uaug_presence": 0,
                    "source_kozak_score": 0.5,
                    "source_gc": 0.4,
                    "five_utr_len": 32,
                },
            ]
            with open(teacher, "w", encoding="utf-8") as fh:
                for row in teacher_rows:
                    fh.write(json.dumps(row) + "\n")
            with open(cascade, "w", encoding="utf-8") as fh:
                for row in cascade_rows:
                    fh.write(json.dumps(row) + "\n")

            summary = mine_cascade_hard_negative_teacher(
                teacher_jsonl=teacher,
                cascade_records_jsonl=cascade,
                out_jsonl=out_jsonl,
                out_json=out_json,
                out_md=out_md,
                rescue_cap_per_record=2,
                precision_cap_per_record=2,
                neutral_cap_per_record=1,
            )
            self.assertTrue(os.path.exists(out_jsonl))
            self.assertTrue(os.path.exists(out_json))
            self.assertTrue(os.path.exists(out_md))
            self.assertEqual(summary["row_counts_by_group"]["cascade_rescue"], 2)
            self.assertEqual(summary["row_counts_by_group"]["cascade_precision"], 2)
            self.assertEqual(summary["row_counts_by_group"]["neutral"], 1)
            with open(out_jsonl, "r", encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            by_group = {}
            for row in rows:
                by_group.setdefault(row["cascade_teacher_group"], []).append(row)
            self.assertIn("cascade_rescue", by_group["cascade_rescue"][0]["source_scores"])
            self.assertIn("cascade_precision", by_group["cascade_precision"][0]["source_scores"])
            self.assertEqual(by_group["neutral"][0]["cascade_teacher_group"], "neutral")


class TestExternalModelRecords(unittest.TestCase):
    def test_external_results_record_protocol_differences(self):
        names = available_external_models()
        self.assertIn("UTRGAN", names)
        self.assertIn("Prot2RNA", names)
        self.assertIn("CodonFM", names)
        results = list_external_results(task_id="T4")
        self.assertGreaterEqual(len(results), 8)
        for item in results:
            self.assertEqual(item.status, "offline_placeholder")
            self.assertTrue(item.offline)
            self.assertIn("protocol", item.protocol_difference.lower())
            self.assertEqual(item.metrics, {})

        one = get_external_result("Optimus+GA", task_id="T7")
        self.assertIn("GA", one.model_name)
        self.assertIn("T7", one.notes)

    def test_external_benchmark_model_set_is_complete_and_default(self):
        expected = {
            "LinearDesign",
            "EnsembleDesign",
            "codonGPT",
            "Prot2RNA",
            "UTailoR",
            "UTRGAN",
        }
        self.assertEqual(set(EXTERNAL_BENCHMARK_MODELS), expected)
        with tempfile.TemporaryDirectory(prefix="mef_external_default_set_") as tmp:
            payload = dry_run_external_sota(
                task_id="T5",
                out_dir=tmp,
                write_artifacts=False,
            )
        self.assertEqual(
            {row["model_name"] for row in payload["summary"]["rows"]},
            expected,
        )

    def test_external_sota_dry_run_writes_standard_artifacts(self):
        import hashlib
        import json
        import tempfile

        with tempfile.TemporaryDirectory(prefix="mef_external_sota_") as tmp:
            records_jsonl = os.path.join(tmp, "records.jsonl")
            with open(records_jsonl, "w", encoding="utf-8") as fh:
                for idx in range(3):
                    fh.write(json.dumps({"transcript_id": f"tx{idx}"}, sort_keys=True) + "\n")
            with open(records_jsonl, "rb") as fh:
                expected_sha256 = hashlib.sha256(fh.read()).hexdigest()
            fake_bin = os.path.join(tmp, "fake-lineardesign")
            with open(fake_bin, "w", encoding="utf-8") as fh:
                fh.write("#!/bin/sh\nexit 0\n")
            os.chmod(fake_bin, 0o755)
            old = os.environ.get("LINEARDESIGN_BIN")
            os.environ["LINEARDESIGN_BIN"] = fake_bin
            try:
                payload = dry_run_external_sota(
                    models=["LinearDesign", "UTailoR"],
                    task_id="T5",
                    records_jsonl=records_jsonl,
                    out_dir=tmp,
                    limit=2,
                    split_name="unit_head3",
                    seed=17,
                    hardware_label="unit-cpu",
                )
            finally:
                if old is None:
                    os.environ.pop("LINEARDESIGN_BIN", None)
                else:
                    os.environ["LINEARDESIGN_BIN"] = old
            for filename in ("summary.json", "runtime.json", "table.md"):
                self.assertTrue(os.path.exists(os.path.join(tmp, filename)))
            summary = payload["summary"]
            self.assertEqual(summary["n_models"], 2)
            self.assertEqual(summary["n_executable_ready"], 1)
            self.assertEqual(summary["n_not_configured"], 1)
            self.assertEqual(summary["dataset"]["sha256"], expected_sha256)
            self.assertEqual(summary["dataset"]["record_count_total"], 3)
            self.assertEqual(summary["dataset"]["record_count_effective"], 2)
            self.assertEqual(summary["dataset"]["split_name"], "unit_head3")
            self.assertEqual(summary["dataset"]["seed"], 17)
            self.assertEqual(summary["hardware"]["label"], "unit-cpu")
            self.assertEqual(payload["runtime"]["dataset_sha256"], expected_sha256)
            self.assertEqual(payload["runtime"]["record_count_effective"], 2)
            rows = {row["model_name"]: row for row in summary["rows"]}
            self.assertEqual(rows["LinearDesign"]["status"], "executable_ready")
            self.assertEqual(rows["UTailoR"]["status"], "not_configured")
            self.assertEqual(rows["LinearDesign"]["metrics"], {})
            self.assertEqual(rows["LinearDesign"]["candidate_audit"][0]["status"], "executable")
            self.assertEqual(rows["LinearDesign"]["executable"], fake_bin)
            self.assertEqual(rows["UTailoR"]["candidate_audit"][0]["status"], "env_unset")
            with open(os.path.join(tmp, "summary.json"), "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded["artifact_contract"]["summary_json"], "external SOTA status/protocol metadata")
            self.assertIn("dataset.sha256", loaded["artifact_contract"]["required_real_run_metadata"])
            with open(os.path.join(tmp, "table.md"), "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("External SOTA Dry-Run", text)
            self.assertIn("Dataset audit", text)
            self.assertIn(expected_sha256, text)
            self.assertIn("Candidate audit", text)

    def test_external_sota_dry_run_rejects_stale_env_path(self):
        import tempfile

        with tempfile.TemporaryDirectory(prefix="mef_external_sota_stale_") as tmp:
            stale_bin = os.path.join(tmp, "missing-lineardesign")
            old = os.environ.get("LINEARDESIGN_BIN")
            os.environ["LINEARDESIGN_BIN"] = stale_bin
            try:
                payload = dry_run_external_sota(
                    models=["LinearDesign"],
                    task_id="T4",
                    out_dir=tmp,
                    write_artifacts=False,
                )
            finally:
                if old is None:
                    os.environ.pop("LINEARDESIGN_BIN", None)
                else:
                    os.environ["LINEARDESIGN_BIN"] = old

            summary = payload["summary"]
            self.assertEqual(summary["n_executable_ready"], 0)
            self.assertEqual(summary["n_not_configured"], 1)
            row = summary["rows"][0]
            self.assertEqual(row["status"], "not_configured")
            self.assertEqual(row["candidate_audit"][0]["status"], "not_executable")
            self.assertFalse(row["candidate_audit"][0]["exists"])

    def test_external_sota_input_pack_writes_cds_and_utr_contracts(self):
        import tempfile

        from mrna_editflow.data.download_mrna import write_records_jsonl

        with tempfile.TemporaryDirectory(prefix="mef_external_input_pack_") as tmp:
            records_jsonl = os.path.join(tmp, "records.jsonl")
            records = _tiny_records() + [MRNARecord("bad", "AAA", "AUGAAU", "CCC")]
            write_records_jsonl(records, records_jsonl)
            out_dir = os.path.join(tmp, "pack")
            summary = build_external_sota_input_pack(
                records_jsonl=records_jsonl,
                out_dir=out_dir,
                limit=5,
                split_name="unit_external_pack",
                seed=23,
            )
            self.assertEqual(summary["artifact_kind"], "external_sota_input_pack")
            self.assertEqual(summary["dataset"]["record_count_total"], 5)
            self.assertEqual(summary["dataset"]["record_count_effective"], 5)
            self.assertEqual(summary["n_cds_protein_rows"], 4)
            self.assertEqual(summary["n_utr5_rows"], 4)
            self.assertEqual(summary["n_skipped_invalid_cds"], 1)
            self.assertTrue(summary["ready_for_external_real_run"])
            self.assertFalse(summary["ready_for_external_sota_claim"])

            outputs = summary["outputs"]
            for key in (
                "summary_json",
                "table_md",
                "cds_protein_jsonl",
                "utr5_jsonl",
                "metric_schema_json",
            ):
                self.assertTrue(os.path.exists(outputs[key]))
            for key in (
                "summary_json_sha256",
                "cds_protein_jsonl_sha256",
                "utr5_jsonl_sha256",
                "metric_schema_json_sha256",
                "table_md_sha256",
            ):
                self.assertRegex(outputs[key], r"^[0-9a-f]{64}$")

            with open(outputs["cds_protein_jsonl"], "r", encoding="utf-8") as fh:
                cds_rows = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual(len(cds_rows), 4)
            self.assertEqual(cds_rows[0]["row_kind"], "cds_protein_conditioned_external_input")
            self.assertIn("protein_target", cds_rows[0])
            self.assertIn("native_cds", cds_rows[0])
            self.assertIn("LinearDesign", cds_rows[0]["expected_models"])
            self.assertTrue(cds_rows[0]["required_hard_constraints"]["protein_identity_exact_1"])

            with open(outputs["utr5_jsonl"], "r", encoding="utf-8") as fh:
                utr_rows = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual(len(utr_rows), 4)
            self.assertEqual(utr_rows[0]["row_kind"], "utr5_external_input")
            self.assertIn("native_five_utr", utr_rows[0])
            self.assertIn("UTailoR", utr_rows[0]["expected_models"])
            self.assertIn("UTRGAN", utr_rows[0]["expected_models"])
            self.assertTrue(utr_rows[0]["required_hard_constraints"]["cds_unchanged"])

            schema = external_metric_schema()
            self.assertIn("designed_cds", schema["cds_protein_conditioned"]["required_output_jsonl_fields"])
            self.assertIn("designed_five_utr", schema["utr5_only"]["required_output_jsonl_fields"])
            self.assertIn("UTRGAN", schema["utr5_only"]["models"])
            with open(outputs["table_md"], "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("External SOTA Input Pack", text)
            self.assertIn("CDS/protein-conditioned", text)
            self.assertIn("5'UTR-only", text)

    def test_lineardesign_adapter_writes_auditable_real_run_contract(self):
        import tempfile

        from mrna_editflow.data.download_mrna import write_records_jsonl
        from mrna_editflow.eval.audit_external_sota_real_runs import (
            audit_external_sota_real_runs,
        )

        with tempfile.TemporaryDirectory(prefix="mef_lineardesign_adapter_") as tmp:
            records_jsonl = os.path.join(tmp, "records.jsonl")
            write_records_jsonl([MRNARecord("r1", "GCC", "AUGGCUUAA", "AAA")], records_jsonl)
            pack = build_external_sota_input_pack(
                records_jsonl=records_jsonl,
                out_dir=os.path.join(tmp, "benchmark", "external_sota", "input_pack_t5_head1024"),
                limit=1,
                split_name="unit_lineardesign",
                seed=7,
            )
            executable = os.path.join(tmp, "mock-lineardesign")
            with open(executable, "w", encoding="utf-8") as fh:
                fh.write(
                    "#!/bin/sh\n"
                    "if [ \"${1:-}\" = \"--version\" ]; then echo mock-lineardesign-v1; exit 0; fi\n"
                    "cat >/dev/null\n"
                    "echo 'mRNA sequence:  AUGGCU'\n"
                    "echo 'mRNA structure: ......'\n"
                    "echo 'mRNA folding free energy: -0.20 kcal/mol; mRNA CAI: 0.900'\n"
                )
            os.chmod(executable, 0o755)
            out_dir = os.path.join(
                tmp,
                "benchmark",
                "external_sota",
                "real_runs_t5_head1024",
                "LinearDesign",
            )
            summary = run_lineardesign_adapter(
                input_pack_summary=pack["outputs"]["summary_json"],
                executable=executable,
                out_dir=out_dir,
                workers=1,
            )
            self.assertEqual(summary["n_inputs"], 1)
            self.assertEqual(summary["n_outputs"], 1)
            self.assertEqual(summary["n_failures"], 0)
            self.assertEqual(summary["valid_cds_fraction"], 1.0)
            self.assertEqual(summary["protein_identity_exact_1_fraction"], 1.0)
            with open(os.path.join(out_dir, "cds_outputs.jsonl"), "r", encoding="utf-8") as fh:
                row = json.loads(fh.readline())
            self.assertEqual(row["designed_cds"], "AUGGCUUAA")
            self.assertTrue(row["postprocessing"]["terminal_stop_appended"])
            self.assertEqual(row["mfe"], -0.2)

            audit = audit_external_sota_real_runs(
                project_root=tmp,
                models=["LinearDesign"],
            )
            self.assertEqual(audit["rows"][0]["status"], "measured")
            self.assertTrue(audit["summary"]["ready_for_external_real_metric_table"])
            self.assertFalse(audit["summary"]["ready_for_external_sota_metric_claim"])

    def test_ensembledesign_adapter_writes_resumable_budgeted_contract(self):
        import tempfile

        from mrna_editflow.data.download_mrna import write_records_jsonl
        from mrna_editflow.eval.audit_external_sota_real_runs import (
            audit_external_sota_real_runs,
        )

        with tempfile.TemporaryDirectory(prefix="mef_ensemble_adapter_") as tmp:
            records_jsonl = os.path.join(tmp, "records.jsonl")
            write_records_jsonl(
                [MRNARecord("r1", "GCC", "AUGGCUUAA", "AAA")],
                records_jsonl,
            )
            pack = build_external_sota_input_pack(
                records_jsonl=records_jsonl,
                out_dir=os.path.join(
                    tmp,
                    "benchmark",
                    "external_sota",
                    "input_pack_t5_head1024",
                ),
                limit=1,
                split_name="unit_ensembledesign",
                seed=8,
            )
            executable = os.path.join(tmp, "mock-ensembledesign")
            with open(executable, "w", encoding="utf-8") as fh:
                fh.write(
                    "#!/bin/sh\n"
                    "if [ \"${1:-}\" = \"--version\" ]; then "
                    "echo mock-ensembledesign-v1; exit 0; fi\n"
                    "fasta=''\n"
                    "while [ \"$#\" -gt 0 ]; do\n"
                    "  if [ \"$1\" = \"--fasta\" ]; then fasta=\"$2\"; "
                    "shift 2; else shift; fi\n"
                    "done\n"
                    "id=$(sed -n '1s/^>//p' \"$fasta\")\n"
                    "printf '>%s|Ensemble Free Energy: -1.25 kcal/mol\\n"
                    "AUGGCU\\n' \"$id\"\n"
                )
            os.chmod(executable, 0o755)
            out_dir = os.path.join(
                tmp,
                "benchmark",
                "external_sota",
                "real_runs_t5_head1024",
                "EnsembleDesign",
            )
            summary = run_ensembledesign_adapter(
                input_pack_summary=pack["outputs"]["summary_json"],
                executable=executable,
                out_dir=out_dir,
                workers=1,
                num_iters=1,
                num_runs=1,
            )
            self.assertEqual(summary["n_outputs"], 1)
            self.assertEqual(summary["n_failures"], 0)
            self.assertEqual(summary["valid_cds_fraction"], 1.0)
            self.assertEqual(
                summary["protein_identity_exact_1_fraction"],
                1.0,
            )
            self.assertFalse(
                summary[
                    "protocol_fidelity_sufficient_for_sota_reproduction"
                ]
            )
            resumed = run_ensembledesign_adapter(
                input_pack_summary=pack["outputs"]["summary_json"],
                executable=executable,
                out_dir=out_dir,
                workers=1,
                num_iters=1,
                num_runs=1,
                resume=True,
            )
            self.assertEqual(resumed["n_outputs"], 1)
            with open(
                os.path.join(out_dir, "cds_outputs.jsonl"),
                "r",
                encoding="utf-8",
            ) as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["designed_cds"], "AUGGCUUAA")
            self.assertEqual(rows[0]["ensemble_free_energy"], -1.25)

            audit = audit_external_sota_real_runs(
                project_root=tmp,
                models=["EnsembleDesign"],
            )
            self.assertEqual(audit["rows"][0]["status"], "measured")
            self.assertFalse(
                audit["summary"]["ready_for_external_sota_metric_claim"]
            )

    def test_ensembledesign_adapter_rescues_search_error_with_larger_beam(self):
        import tempfile

        from mrna_editflow.data.download_mrna import write_records_jsonl
        from mrna_editflow.eval.audit_external_sota_real_runs import (
            audit_external_sota_real_runs,
        )

        with tempfile.TemporaryDirectory(
            prefix="mef_ensemble_rescue_"
        ) as tmp:
            records_jsonl = os.path.join(tmp, "records.jsonl")
            write_records_jsonl(
                [MRNARecord("r1", "GCC", "AUGGCUUAA", "AAA")],
                records_jsonl,
            )
            pack = build_external_sota_input_pack(
                records_jsonl=records_jsonl,
                out_dir=os.path.join(
                    tmp,
                    "benchmark",
                    "external_sota",
                    "input_pack_t5_head1024",
                ),
                limit=1,
                split_name="unit_ensembledesign_rescue",
                seed=9,
            )
            executable = os.path.join(tmp, "mock-ensembledesign")
            with open(executable, "w", encoding="utf-8") as fh:
                fh.write(
                    "#!/bin/sh\n"
                    "if [ \"${1:-}\" = \"--version\" ]; then "
                    "echo mock-ensembledesign-rescue-v1; exit 0; fi\n"
                    "fasta=''; beam=''\n"
                    "while [ \"$#\" -gt 0 ]; do\n"
                    "  case \"$1\" in\n"
                    "    --fasta) fasta=\"$2\"; shift 2 ;;\n"
                    "    --beam_size) beam=\"$2\"; shift 2 ;;\n"
                    "    *) shift ;;\n"
                    "  esac\n"
                    "done\n"
                    "if [ \"$beam\" = \"100\" ]; then "
                    "echo 'beam search error'; exit 9; fi\n"
                    "id=$(sed -n '1s/^>//p' \"$fasta\")\n"
                    "printf '>%s|Ensemble Free Energy: -1.50 kcal/mol\\n"
                    "AUGGCU\\n' \"$id\"\n"
                )
            os.chmod(executable, 0o755)
            out_dir = os.path.join(
                tmp,
                "benchmark",
                "external_sota",
                "real_runs_t5_head1024",
                "EnsembleDesign",
            )
            failed = run_ensembledesign_adapter(
                input_pack_summary=pack["outputs"]["summary_json"],
                executable=executable,
                out_dir=out_dir,
                workers=1,
                beam_size=100,
                num_iters=1,
                num_runs=1,
            )
            self.assertEqual(failed["n_outputs"], 0)
            self.assertEqual(failed["n_failures"], 1)

            state_path = os.path.join(out_dir, "run_state.json")
            with open(state_path, "r", encoding="utf-8") as fh:
                legacy_state = json.load(fh)
            legacy_state["config"].pop("rescue_beam_size")
            with open(state_path, "w", encoding="utf-8") as fh:
                json.dump(legacy_state, fh)

            rescued = run_ensembledesign_adapter(
                input_pack_summary=pack["outputs"]["summary_json"],
                executable=executable,
                out_dir=out_dir,
                workers=1,
                beam_size=100,
                rescue_beam_size=200,
                num_iters=1,
                num_runs=1,
                resume=True,
            )
            self.assertEqual(rescued["n_outputs"], 1)
            self.assertEqual(rescued["n_failures"], 0)
            self.assertEqual(rescued["n_beam_rescued"], 1)
            self.assertEqual(rescued["beam_rescue_fraction"], 1.0)
            self.assertIn(
                "beam200_search_error_rescue_1_rows",
                rescued["protocol_fidelity"],
            )
            self.assertFalse(
                rescued[
                    "protocol_fidelity_sufficient_for_sota_reproduction"
                ]
            )
            with open(
                os.path.join(out_dir, "cds_outputs.jsonl"),
                "r",
                encoding="utf-8",
            ) as fh:
                row = json.loads(fh.readline())
            postprocessing = row["postprocessing"]
            self.assertTrue(postprocessing["beam_rescue_used"])
            self.assertEqual(postprocessing["primary_beam_size"], 100)
            self.assertEqual(postprocessing["effective_beam_size"], 200)
            self.assertIn("exit=9", postprocessing["primary_error"])

            audit = audit_external_sota_real_runs(
                project_root=tmp,
                models=["EnsembleDesign"],
            )
            self.assertEqual(audit["rows"][0]["status"], "measured")
            self.assertTrue(
                audit["summary"]["ready_for_external_real_metric_table"]
            )

    def test_codongpt_adapter_writes_pretrained_checkpoint_contract(self):
        from mrna_editflow.data.download_mrna import write_records_jsonl
        from mrna_editflow.eval.audit_external_sota_real_runs import (
            audit_external_sota_real_runs,
        )

        with tempfile.TemporaryDirectory(prefix="mef_codongpt_adapter_") as tmp:
            records_jsonl = os.path.join(tmp, "records.jsonl")
            write_records_jsonl(
                [MRNARecord("r1", "GCC", "AUGGCUUAA", "AAA")],
                records_jsonl,
            )
            pack = build_external_sota_input_pack(
                records_jsonl=records_jsonl,
                out_dir=os.path.join(
                    tmp,
                    "benchmark",
                    "external_sota",
                    "input_pack_t5_head1024",
                ),
                limit=1,
                split_name="unit_codongpt",
                seed=11,
            )
            executable = os.path.join(tmp, "mock-codongpt")
            with open(executable, "w", encoding="utf-8") as fh:
                fh.write(
                    "#!/bin/sh\n"
                    "if [ \"${1:-}\" = \"--version\" ]; then "
                    "echo mock-codongpt-v1; exit 0; fi\n"
                    "exit 2\n"
                )
            os.chmod(executable, 0o755)
            model_dir = os.path.join(tmp, "model")
            os.makedirs(model_dir)
            with open(
                os.path.join(model_dir, "model_manifest.json"),
                "w",
                encoding="utf-8",
            ) as fh:
                json.dump(
                    {
                        "hf_repo": "naniltx/codonGPT",
                        "hf_revision": "unit-revision",
                        "license": "free_for_research_use_model_card",
                    },
                    fh,
                )
            out_dir = os.path.join(
                tmp,
                "benchmark",
                "external_sota",
                "real_runs_t5_head1024",
                "codonGPT",
            )

            def mock_generator(rows, batch_seed):
                self.assertEqual(batch_seed, 17)
                return ["AUGGCUUAA" for _ in rows]

            summary = run_codongpt_adapter(
                input_pack_summary=pack["outputs"]["summary_json"],
                model_dir=model_dir,
                executable=executable,
                out_dir=out_dir,
                batch_size=8,
                seed=17,
                device="cpu",
                batch_generator=mock_generator,
            )
            self.assertEqual(summary["n_inputs"], 1)
            self.assertEqual(summary["n_outputs"], 1)
            self.assertEqual(summary["n_failures"], 0)
            self.assertEqual(summary["valid_cds_fraction"], 1.0)
            self.assertEqual(
                summary["protein_identity_exact_1_fraction"],
                1.0,
            )
            self.assertEqual(summary["mean_codon_accuracy_vs_native"], 1.0)
            self.assertFalse(
                summary[
                    "protocol_fidelity_sufficient_for_sota_reproduction"
                ]
            )
            with open(
                os.path.join(out_dir, "cds_outputs.jsonl"),
                "r",
                encoding="utf-8",
            ) as fh:
                row = json.loads(fh.readline())
            self.assertEqual(row["designed_cds"], "AUGGCUUAA")
            self.assertEqual(row["codon_accuracy_vs_native"], 1.0)

            audit = audit_external_sota_real_runs(
                project_root=tmp,
                models=["codonGPT"],
            )
            self.assertEqual(audit["rows"][0]["status"], "measured")
            self.assertTrue(
                audit["summary"]["ready_for_external_real_metric_table"]
            )
            self.assertFalse(
                audit["summary"]["ready_for_external_sota_metric_claim"]
            )

    def test_utrgan_adapter_writes_auditable_fixed_region_contract(self):
        import tempfile

        from mrna_editflow.data.download_mrna import write_records_jsonl
        from mrna_editflow.eval.audit_external_sota_real_runs import (
            audit_external_sota_real_runs,
        )

        with tempfile.TemporaryDirectory(prefix="mef_utrgan_adapter_") as tmp:
            records_jsonl = os.path.join(tmp, "records.jsonl")
            write_records_jsonl([MRNARecord("r1", "AAAAAA", "AUGGCUUAA", "CCCUAA")], records_jsonl)
            pack = build_external_sota_input_pack(
                records_jsonl=records_jsonl,
                out_dir=os.path.join(tmp, "benchmark", "external_sota", "input_pack_t5_head1024"),
                limit=1,
                split_name="unit_utrgan",
                seed=9,
            )
            tool_root = os.path.join(tmp, "tool")
            work_dir = os.path.join(tool_root, "src", "mrl_te_optimization")
            os.makedirs(work_dir, exist_ok=True)
            executable = os.path.join(tmp, "mock-utrgan")
            with open(executable, "w", encoding="utf-8") as fh:
                fh.write(
                    "#!/bin/sh\n"
                    "if [ \"${1:-}\" = \"--version\" ]; then echo mock-utrgan-v1; exit 0; fi\n"
                    "mkdir -p outputs\n"
                    "echo 'GCCACCGCCACC' > outputs/opt_seqs_FMRL.txt\n"
                    "echo '7.5' > outputs/opt_mrl_FMRL.txt\n"
                    "echo '6.0' > outputs/init_mrl_FMRL.txt\n"
                    "echo 'AAAAAA' > outputs/init_seqs_FMRL.txt\n"
                )
            os.chmod(executable, 0o755)
            out_dir = os.path.join(
                tmp,
                "benchmark",
                "external_sota",
                "real_runs_t5_head1024",
                "UTRGAN",
            )
            summary = run_utrgan_adapter(
                input_pack_summary=pack["outputs"]["summary_json"],
                executable=executable,
                tool_root=tool_root,
                out_dir=out_dir,
                limit=1,
                steps=1,
            )
            self.assertEqual(summary["n_inputs"], 1)
            self.assertEqual(summary["n_outputs"], 1)
            self.assertEqual(summary["n_failures"], 0)
            self.assertEqual(summary["cds_unchanged_fraction"], 1.0)
            self.assertEqual(summary["protein_identity_exact_1_fraction"], 1.0)
            with open(os.path.join(out_dir, "utr5_outputs.jsonl"), "r", encoding="utf-8") as fh:
                row = json.loads(fh.readline())
            self.assertEqual(row["designed_five_utr"], "GCCACCGCCACC")
            self.assertTrue(row["cds_unchanged"])
            self.assertTrue(row["three_utr_unchanged"])
            self.assertIn("te_proxy_delta_vs_native", row)
            self.assertGreater(row["utr_edit_distance_vs_native"], 0)
            self.assertEqual(summary["exact_native_utr_match_fraction"], 0.0)

            audit = audit_external_sota_real_runs(
                project_root=tmp,
                models=["UTRGAN"],
            )
            self.assertEqual(audit["rows"][0]["status"], "measured")
            self.assertTrue(audit["summary"]["ready_for_external_real_metric_table"])
            self.assertFalse(audit["summary"]["ready_for_external_sota_metric_claim"])

    def test_utailor_adapter_audits_strict_eligible_subset(self):
        import tempfile

        from mrna_editflow.data.download_mrna import write_records_jsonl
        from mrna_editflow.eval.audit_external_sota_real_runs import (
            audit_external_sota_real_runs,
        )

        with tempfile.TemporaryDirectory(prefix="mef_utailor_adapter_") as tmp:
            records_jsonl = os.path.join(tmp, "records.jsonl")
            write_records_jsonl(
                [
                    MRNARecord("eligible", "A" * 30, "AUGGCUUAA", "CCCUAA"),
                    MRNARecord("short", "A" * 10, "AUGGCUUAA", "CCCUAA"),
                ],
                records_jsonl,
            )
            pack = build_external_sota_input_pack(
                records_jsonl=records_jsonl,
                out_dir=os.path.join(
                    tmp,
                    "benchmark",
                    "external_sota",
                    "input_pack_t5_head1024",
                ),
                limit=2,
                split_name="unit_utailor",
                seed=10,
            )
            executable = os.path.join(tmp, "mock-utailor")
            with open(executable, "w", encoding="utf-8") as fh:
                fh.write(
                    "#!/usr/bin/env python3\n"
                    "import argparse, json\n"
                    "p=argparse.ArgumentParser(); "
                    "p.add_argument('--version', action='store_true'); "
                    "p.add_argument('--input-fasta'); "
                    "p.add_argument('--output-json'); "
                    "p.add_argument('--task-id'); a=p.parse_args()\n"
                    "if a.version: print('mock-utailor-v1'); raise SystemExit\n"
                    "lines=[x.strip() for x in open(a.input_fasta) if x.strip()]\n"
                    "records=[]\n"
                    "for i in range(0,len(lines),2):\n"
                    "  tid=lines[i][1:]; seq=lines[i+1]; opt='G'+seq[1:]\n"
                    "  records.append({'Sequence name':tid,'Original UTR':seq,"
                    "'Original RL':5.0,'Optimized UTR':opt,'Optimized RL':6.0,"
                    "'Optimized RL_50nt':6.1,'Increased RL':1.0})\n"
                    "json.dump({'records':records,'official_xlsx':None},"
                    "open(a.output_json,'w'))\n"
                )
            os.chmod(executable, 0o755)
            out_dir = os.path.join(
                tmp,
                "benchmark",
                "external_sota",
                "real_runs_t5_head1024",
                "UTailoR",
            )
            summary = run_utailor_adapter(
                input_pack_summary=pack["outputs"]["summary_json"],
                executable=executable,
                out_dir=out_dir,
            )
            self.assertEqual(summary["n_inputs"], 2)
            self.assertEqual(summary["n_eligible_inputs"], 1)
            self.assertEqual(summary["n_ineligible_inputs"], 1)
            self.assertEqual(summary["n_outputs"], 1)
            self.assertEqual(summary["n_failures"], 0)
            self.assertEqual(
                summary["eligibility"]["policy"],
                "official_input_length_25_100_strict",
            )

            audit = audit_external_sota_real_runs(
                project_root=tmp,
                models=["UTailoR"],
            )
            row = audit["rows"][0]
            self.assertEqual(row["status"], "measured")
            self.assertTrue(row["protocol_subset"])
            self.assertEqual(row["expected_input_rows"], 2)
            self.assertEqual(row["expected_eligible_input_rows"], 1)
            self.assertTrue(
                audit["summary"]["ready_for_external_real_metric_table"]
            )
            self.assertFalse(
                audit["summary"]["ready_for_external_sota_metric_claim"]
            )


class TestMRNABenchProbe(unittest.TestCase):
    def test_probe_records_trainable_parameter_count(self):
        cfg = MRNABenchProbeConfig(steps=2, seed=301)
        result = run_mrnabench_probe(records=_tiny_records(), config=cfg, device="cpu")
        self.assertEqual(result.mode, "synthetic_labels")
        self.assertEqual(result.n_records, 4)
        self.assertGreater(result.trainable_params, 0)
        self.assertEqual(result.trainable_params, result.feature_dim * cfg.num_labels + cfg.num_labels)
        self.assertTrue(math.isfinite(result.final_loss))

        fallback = run_mrnabench_probe(config=MRNABenchProbeConfig(steps=1, synthetic_size=3, seed=302))
        self.assertEqual(fallback.mode, "synthetic_fallback")
        self.assertGreater(fallback.trainable_params, 0)


class TestAblationScript(unittest.TestCase):
    def test_run_ablation_dry_run_succeeds(self):
        script = os.path.join(_REPO_ROOT, "mrna_editflow", "scripts", "run_ablation.sh")
        self.assertTrue(os.path.exists(script))
        self.assertTrue(os.access(script, os.X_OK))
        proc = subprocess.run(
            [script, "--dry-run"],
            cwd=_REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("editflow_vs_masked_diffusion", proc.stdout)
        self.assertIn("coupling_mixture_empty_corruption_ortholog", proc.stdout)
        self.assertIn("backbone_family", proc.stdout)
        self.assertIn("utr_local_search_baseline", proc.stdout)
        self.assertIn("utr_teacher_export", proc.stdout)
        self.assertIn("hybrid_teacher_export", proc.stdout)

    def test_long_running_server_scripts_have_dry_runs(self):
        scripts = [
            ("run_after_stage_a_full1k.sh", "proposal_ranking_t5_full1k_final"),
            ("run_after_stage_a_a100_max.sh", "a100_max_posteval"),
            ("run_head256_ranker_fair_eval.sh", "t5_ranker_full1k_head256_comparison"),
            ("watch_head256_refresh_sota.sh", "t5_ranker_full1k_head256_comparison"),
            ("run_stage_a_scaleup_10k.sh", "stage_a_public_full_10k_bs8ga4"),
            ("run_stage_a_a100_max_train.sh", "stage_a_a100_max_train"),
            ("run_stage_a_scalelaw_sweep.sh", "stage_a_scalelaw_sweep"),
            ("run_stage_a_downstream_eval_queue.sh", "stage_a_downstream_eval_queue"),
            ("run_refseq_public_build.sh", "refseq_public_build"),
            ("run_gencode_family_leakage_audit.sh", "gencode_family_leakage_protocol"),
            ("run_refseq_family_leakage_audit.sh", "refseq_family_leakage_protocol"),
            ("run_downstream_predictor_protocol.sh", "downstream_predictor_protocol"),
            ("watch_gencode_family_readiness.sh", "gencode_family_readiness_watcher"),
            ("watch_p3_readiness.sh", "p3_readiness_watcher"),
            ("run_after_stage_a_10k.sh", "proposal_ranking_t5_stage_a10k_head1024"),
            ("run_region_adapter_ablation.sh", "region_adapter_t5_all_head256"),
            ("eval_region_adapter_ablation.sh", "region_adapter_all_top64"),
            ("watch_region_adapter_eval.sh", "stage_b_region_t5_best"),
            ("run_region_adapter_ablation_chain.sh", "region_adapter_vs_hardneg_v2_top64_head256"),
            ("run_protein_conditioned_cds_gc_sweep.sh", "protein_conditioned_cds_gc_sweep_head256"),
            ("run_protein_conditioned_t4_slice.sh", "protein_conditioned_t4_head1024"),
            ("run_frozen_backbone_protocol_check.sh", "frozen_backbone_protocol_head256"),
            ("audit_sota_readiness.sh", "sota_readiness_audit_head256"),
            ("sync_external_sota_real_run_audit.sh", "EXTERNAL SOTA REAL-RUN AUDIT SYNC"),
            ("run_external_lineardesign_head1024.sh", "EXTERNAL LINEARDESIGN HEAD1024"),
            ("run_external_ensembledesign_head1024.sh", "EXTERNAL ENSEMBLEDESIGN HEAD1024"),
            ("run_external_codongpt_head1024.sh", "EXTERNAL CODONGPT HEAD1024"),
            ("run_external_codongpt_multiseed_head1024.sh", "EXTERNAL CODONGPT MULTISEED HEAD1024"),
            ("setup_external_codongpt.sh", "SETUP EXTERNAL CODONGPT"),
            ("watch_external_ensembledesign_retry.sh", "WATCH EXTERNAL ENSEMBLEDESIGN RETRY"),
            ("run_external_utrgan_head1024.sh", "EXTERNAL UTRGAN HEAD1024"),
            ("run_external_utrgan_paper10000.sh", "EXTERNAL UTRGAN PAPER10000"),
            ("run_external_utailor_head1024.sh", "EXTERNAL UTAILOR HEAD1024"),
            ("run_mef_utr5only_head1024.sh", "MEF UTR5-ONLY HEAD1024"),
            ("run_mef_utailor_subset_budget5.sh", "MEF UTAILOR SUBSET BUDGET5"),
            ("watch_external_lineardesign_t4_comparison.sh", "WATCH EXTERNAL LINEARDESIGN T4 COMPARISON"),
            ("harvest_sota_artifacts.sh", "sota_harvest_manifest_head256"),
            ("check_remote_sota_status.sh", "remote_execution_status"),
            ("summarize_t6_length_curve_report.sh", "t6_length_curve_report_head256_head1024"),
            ("merge_t6_head1024_shards.sh", "merge_multiseed_shards"),
        ]
        for name, expected in scripts:
            script = os.path.join(_REPO_ROOT, "mrna_editflow", "scripts", name)
            self.assertTrue(os.path.exists(script), name)
            self.assertTrue(os.access(script, os.X_OK), name)
            proc = subprocess.run(
                [script, "--dry-run"],
                cwd=_REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn(expected, proc.stdout)
            if name == "eval_region_adapter_ablation.sh":
                self.assertIn("region_adapter_result_audit_head256", proc.stdout)
            if name == "run_protein_conditioned_cds_gc_sweep.sh":
                self.assertIn("protein_conditioned_cds_gc_sweep_head256.audit", proc.stdout)
            if name == "run_protein_conditioned_t4_slice.sh":
                self.assertIn("CODON_DP_MAX_CODON_CHANGES=3", proc.stdout)
                self.assertIn("PROTEIN_MAX_CODON_CHANGES=<none>", proc.stdout)
            if name == "run_mef_utailor_subset_budget5.sh":
                self.assertIn("EDIT_BUDGET=5", proc.stdout)
                self.assertIn("LIMIT=315", proc.stdout)

    def test_merge_t6_head1024_shards_skips_existing_complete_summary(self):
        script = os.path.join(_REPO_ROOT, "mrna_editflow", "scripts", "merge_t6_head1024_shards.sh")
        expected = list(range(10))
        aggregate = {
            metric: {"n": len(expected), "mean": 1.0}
            for metric in (
                "mean_abs_length_error",
                "legal_fraction",
                "mean_protein_identity",
                "within_budget_fraction",
                "reading_frame_intact_fraction",
                "delta_oracle_te_vs_source",
                "mean_oracle_te",
                "mean_edit_distance",
            )
        }
        summary = {
            "config": {"seeds": expected},
            "per_seed": [{"seed": seed} for seed in expected],
            "aggregate": aggregate,
        }
        with tempfile.TemporaryDirectory(prefix="mef_t6_merge_idempotent_") as tmp:
            out_dir = os.path.join(
                tmp,
                "benchmark",
                "multiseed_t6_public_head1024_stagea10k_len_neg15_existing_top64",
            )
            os.makedirs(out_dir)
            with open(os.path.join(out_dir, "multiseed_summary.json"), "w", encoding="utf-8") as fh:
                json.dump(summary, fh)

            env = dict(os.environ)
            env["ROOT"] = tmp
            env["PYTHON_BIN"] = sys.executable
            env["MERGE_TAG"] = "idempotent_probe"
            proc = subprocess.run(
                [script],
                cwd=_REPO_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("SKIP delta=-15: already complete summary", proc.stdout)
            duplicate = os.path.join(
                tmp,
                "benchmark",
                "multiseed_t6_public_head1024_stagea10k_len_neg15_idempotent_probe_top64",
            )
            self.assertFalse(os.path.exists(duplicate), proc.stdout)

    def test_multiobjective_eval_limit_tracks_slice_unless_overridden(self):
        script = os.path.join(
            _REPO_ROOT,
            "mrna_editflow",
            "scripts",
            "eval_multiobjective_ranker_ablation_head256.sh",
        )
        self.assertTrue(os.path.exists(script))

        env = dict(os.environ)
        env["SLICE"] = "head1024"
        proc = subprocess.run(
            ["bash", script, "--dry-run"],
            cwd=_REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("SLICE=head1024", proc.stdout)
        self.assertIn("LIMIT=1024", proc.stdout)
        self.assertIn("multiseed_t5_public_head1024_hardneg_v2_top64", proc.stdout)
        self.assertIn("proposal_ranker_t5_mo_grpo_head1024", proc.stdout)
        self.assertIn("compare_mo_fusion_vs_hardneg_v2_head1024", proc.stdout)

        env["LIMIT"] = "128"
        override = subprocess.run(
            ["bash", script, "--dry-run"],
            cwd=_REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(override.returncode, 0, override.stderr)
        self.assertIn("SLICE=head1024", override.stdout)
        self.assertIn("LIMIT=128", override.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
