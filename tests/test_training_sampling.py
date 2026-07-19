"""CPU smoke tests for Task 4 training and sampling entry points.

Run:
    /Users/bytedance/Documents/research/editflow/.venv/bin/python \
        -m unittest mrna_editflow.tests.test_training_sampling -v
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from unittest import mock

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mrna_editflow.core.config import (
    BackboneConfig,
    CouplingConfig,
    DataConfig,
    MEFConfig,
    ModelConfig,
    TrainConfig,
)
from mrna_editflow.core.constants import REGION_3UTR, REGION_5UTR, REGION_CDS, is_valid_cds, translate
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.core import mrna_flow_utils as U
from mrna_editflow.eval import cascade_proposal_ranking, proposal_ranking
from mrna_editflow.eval.oracle import LocalTranslationOracle
from mrna_editflow.sample import (
    _replace_nt,
    _synonymous_substitution_candidates,
    generate_candidate_records,
    levenshtein_distance,
    load_stage_a_checkpoint,
    sample_mrna,
    sample_sequence,
)
from mrna_editflow.train_adapter import train_region_stage_b, train_stage_b
from mrna_editflow.train_backbone import (
    _aux_struct_loss,
    _flow_batch_loss,
    build_stage_a_model,
    train_stage_a,
)
from mrna_editflow.train_proposal_ranker import (
    ProposalTeacherRow,
    _pair_specs,
    train_proposal_ranker,
)


def _tiny_records():
    return [
        MRNARecord("r1", "ACGUAC", "AUGGCUUAA", "GGCCAA"),
        MRNARecord("r2", "CCAACC", "AUGGCCUAA", "UUCGGA"),
        MRNARecord("r3", "GGAAUU", "AUGAAAUAA", "CCGGAA"),
        MRNARecord("r4", "UAUAUA", "AUGCCCUAA", "AACCGG"),
    ]


def _tiny_config(tmpdir: str) -> MEFConfig:
    return MEFConfig(
        data=DataConfig(seed=17, max_5utr=32, max_cds=96, max_3utr=32),
        backbone=BackboneConfig(name="none", hidden_dim=8, freeze=True),
        model=ModelConfig(
            model_dim=16,
            num_layers=1,
            num_heads=4,
            ffn_mult=2,
            dropout=0.0,
            max_seq_len=128,
            use_aux_struct=False,
            aux_loss_weight=0.0,
        ),
        coupling=CouplingConfig(
            empty_prob=0.0,
            corruption_prob=1.0,
            ortholog_prob=0.0,
            sub_prob=0.25,
            ins_prob=0.05,
            del_prob=0.05,
        ),
        train=TrainConfig(
            epochs=1,
            batch_size=2,
            grad_accum=1,
            lr=1e-3,
            amp=True,
            grad_clip=0.5,
            oom_batch_ladder=(2, 1),
            nan_retry=1,
            save_dir=os.path.join(tmpdir, "ckpts"),
            profile_path=os.path.join(tmpdir, "profile.jsonl"),
            log_every=1,
        ),
    )


class TestFlowLossValidityContract(unittest.TestCase):
    def test_bridge_sampler_called_exactly_once(self):
        with tempfile.TemporaryDirectory(prefix="mef_flow_contract_") as tmp:
            cfg = _tiny_config(tmp)
            backbone, model = build_stage_a_model(cfg, "cpu")
            scheduler = U.CubicScheduler()
            with mock.patch.object(U, "sample_cond_pt", wraps=U.sample_cond_pt) as wrapped:
                losses = _flow_batch_loss(
                    model, backbone, _tiny_records()[:2], cfg, torch.device("cpu"),
                    scheduler, seed=123,
                )
            self.assertEqual(wrapped.call_count, 1)
            self.assertTrue(torch.isfinite(losses["loss"]))

    def test_loss_and_gradients_are_reproducible(self):
        with tempfile.TemporaryDirectory(prefix="mef_flow_repro_") as tmp:
            cfg = _tiny_config(tmp)
            backbone, model = build_stage_a_model(cfg, "cpu")
            backbone.eval()
            model.eval()
            scheduler = U.CubicScheduler()

            def run_once():
                model.zero_grad(set_to_none=True)
                torch.manual_seed(991)
                result = _flow_batch_loss(
                    model, backbone, _tiny_records()[:2], cfg, torch.device("cpu"),
                    scheduler, seed=456,
                )
                result["loss"].backward()
                gradients = {
                    name: param.grad.detach().clone()
                    for name, param in model.named_parameters()
                    if param.grad is not None
                }
                values = {name: value.detach().clone() for name, value in result.items()}
                return values, gradients

            first_values, first_gradients = run_once()
            second_values, second_gradients = run_once()
            self.assertEqual(first_gradients.keys(), second_gradients.keys())
            for name in first_values:
                torch.testing.assert_close(first_values[name], second_values[name], rtol=1e-6, atol=1e-7)
            for name in first_gradients:
                torch.testing.assert_close(
                    first_gradients[name], second_gradients[name], rtol=1e-6, atol=1e-7
                )

    def test_auxiliary_disabled_is_exact_zero(self):
        rates = torch.ones((2, 3, 3), requires_grad=True)
        out = {"rates": rates, "aux": torch.ones((2, 3, 2), requires_grad=True)}
        loss = _aux_struct_loss(
            out, torch.zeros((2, 3), dtype=torch.bool), enabled=False
        )
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_auxiliary_enabled_requires_target_and_provenance(self):
        out = {"rates": torch.ones((1, 2, 3)), "aux": torch.ones((1, 2, 2))}
        mask = torch.zeros((1, 2), dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "explicit target"):
            _aux_struct_loss(out, mask, enabled=True)
        with self.assertRaisesRegex(ValueError, "provenance"):
            _aux_struct_loss(out, mask, enabled=True, target=torch.zeros((1, 2, 2)))

    def test_auxiliary_target_shape_and_finite_supervision(self):
        aux = torch.tensor([[[0.5, -0.5], [0.25, 0.75]]], requires_grad=True)
        out = {"rates": torch.ones((1, 2, 3)), "aux": aux}
        mask = torch.tensor([[False, True]])
        provenance = {
            "artifact_sha256": "a" * 64,
            "source": "unit_test_fixture",
            "target_kind": "synthetic_structural_labels",
        }
        with self.assertRaisesRegex(ValueError, "shape"):
            _aux_struct_loss(
                out, mask, enabled=True, target=torch.zeros((1, 2, 1)), provenance=provenance
            )
        loss = _aux_struct_loss(
            out,
            mask,
            enabled=True,
            target=torch.tensor([[[0.2, -0.1], [9.0, 9.0]]]),
            provenance=provenance,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(aux.grad).all())
        self.assertEqual(float(aux.grad[0, 1].abs().sum()), 0.0)


class TestStageATrainingSmoke(unittest.TestCase):
    def test_train_backbone_smoke_writes_profile_and_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="mef_stage_a_") as tmp:
            cfg = _tiny_config(tmp)
            result = train_stage_a(cfg, records=_tiny_records(), steps=2, device="cpu", seed=101)

            self.assertTrue(os.path.exists(result["checkpoint_path"]))
            self.assertTrue(os.path.exists(result["profile_path"]))
            self.assertTrue(math.isfinite(float(result["best_loss"])))

            with open(result["profile_path"], "r", encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row["stage"], "A")
                self.assertTrue(row["finite_loss"])
                self.assertTrue(row["finite_grad"])
                self.assertTrue(math.isfinite(float(row["loss"])))
                self.assertGreater(float(row["samples_per_s"]), 0.0)
                self.assertTrue(math.isfinite(float(row["grad_norm"])))
                self.assertFalse(row["amp_enabled"])  # CPU must disable AMP automatically.

    def test_checkpoint_guided_candidate_generation_preserves_constraints(self):
        with tempfile.TemporaryDirectory(prefix="mef_stage_a_gen_") as tmp:
            cfg = _tiny_config(tmp)
            result = train_stage_a(cfg, records=_tiny_records(), steps=1, device="cpu", seed=111)
            records = _tiny_records()[:2]

            t4 = generate_candidate_records(
                records,
                task_id="T4",
                checkpoint_path=result["checkpoint_path"],
                limit=2,
                edit_budget=2,
                device="cpu",
            )
            self.assertEqual(len(t4), 2)
            for src, cand in zip(records, t4):
                self.assertTrue(is_valid_cds(cand.cds))
                self.assertEqual(translate(src.cds), translate(cand.cds))

            t5 = generate_candidate_records(
                records,
                task_id="T5",
                checkpoint_path=result["checkpoint_path"],
                limit=2,
                edit_budget=2,
                guidance_scale=1.0,
                target_start_accessibility=0.6,
                proposal_top_k=0,
                device="cpu",
            )
            self.assertEqual(len(t5), 2)
            for src, cand in zip(records, t5):
                self.assertTrue(is_valid_cds(cand.cds))
                self.assertEqual(src.cds, cand.cds)
                self.assertLessEqual(levenshtein_distance(src.seq, cand.seq), 2)

            cascade_t5 = generate_candidate_records(
                records,
                task_id="T5",
                checkpoint_path=result["checkpoint_path"],
                cascade_recall_checkpoint_path=result["checkpoint_path"],
                limit=2,
                edit_budget=2,
                proposal_top_k=4,
                cascade_recall_top_k=4,
                device="cpu",
            )
            self.assertEqual(len(cascade_t5), 2)
            for src, cand in zip(records, cascade_t5):
                self.assertTrue(is_valid_cds(cand.cds))
                self.assertEqual(src.cds, cand.cds)
                self.assertLessEqual(levenshtein_distance(src.seq, cand.seq), 2)

    def test_proposal_ranking_audit_writes_teacher_jsonl(self):
        with tempfile.TemporaryDirectory(prefix="mef_rank_audit_") as tmp:
            cfg = _tiny_config(tmp)
            result = train_stage_a(cfg, records=_tiny_records(), steps=1, device="cpu", seed=113)
            out_json = os.path.join(tmp, "proposal_ranking.json")
            out_jsonl = os.path.join(tmp, "proposal_candidates.jsonl")

            audit = proposal_ranking.run_proposal_ranking_audit(
                _tiny_records()[:1],
                checkpoint_path=result["checkpoint_path"],
                out_json=out_json,
                out_jsonl=out_jsonl,
                task_id="T5",
                limit=1,
                device="cpu",
                candidate_cap=6,
                top_k=3,
            )

            self.assertTrue(os.path.exists(out_json))
            self.assertTrue(os.path.exists(out_jsonl))
            self.assertEqual(audit["aggregate"]["n_records"], 1)
            self.assertEqual(audit["aggregate"]["n_candidates"], 6)
            self.assertIn("mean_model_regret", audit["aggregate"])
            with open(out_jsonl, "r", encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual(len(rows), 6)
            self.assertIn("teacher_score", rows[0])
            self.assertIn("student_score", rows[0])

    def test_cascade_proposal_ranking_audit_writes_two_stage_metrics(self):
        with tempfile.TemporaryDirectory(prefix="mef_cascade_rank_") as tmp:
            cfg = _tiny_config(tmp)
            result = train_stage_a(cfg, records=_tiny_records(), steps=1, device="cpu", seed=115)
            out_json = os.path.join(tmp, "cascade.json")
            out_jsonl = os.path.join(tmp, "cascade_candidates.jsonl")

            audit = cascade_proposal_ranking.run_cascade_proposal_ranking_audit(
                _tiny_records()[:1],
                recall_checkpoint=result["checkpoint_path"],
                precision_checkpoint=result["checkpoint_path"],
                out_json=out_json,
                out_jsonl=out_jsonl,
                task_id="T5",
                limit=1,
                device="cpu",
                candidate_cap=6,
                recall_top_k=3,
            )

            self.assertTrue(os.path.exists(out_json))
            self.assertTrue(os.path.exists(out_jsonl))
            self.assertEqual(audit["aggregate"]["n_records"], 1)
            self.assertEqual(audit["aggregate"]["n_candidates"], 6)
            self.assertIn("mean_cascade_regret", audit["aggregate"])
            self.assertIn("oracle_best_in_recall_top_k_fraction", audit["aggregate"])
            with open(out_jsonl, "r", encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual(len(rows), 6)
            self.assertIn("recall_score", rows[0])
            self.assertIn("precision_score", rows[0])
            self.assertIn("precision_rank_in_recall", rows[0])

    def test_proposal_ranker_finetune_writes_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="mef_rank_train_") as tmp:
            cfg = _tiny_config(tmp)
            stage_a = train_stage_a(cfg, records=_tiny_records(), steps=1, device="cpu", seed=117)
            teacher_path = os.path.join(tmp, "teacher.jsonl")
            rows = [
                {"transcript_id": "r1", "op": "sub", "pos": 0, "nt": "G", "teacher_score": 0.20},
                {"transcript_id": "r1", "op": "sub", "pos": 1, "nt": "U", "teacher_score": -0.05},
                {"transcript_id": "r1", "op": "sub", "pos": 2, "nt": "A", "teacher_score": 0.08},
            ]
            with open(teacher_path, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, sort_keys=True) + "\n")

            result = train_proposal_ranker(
                records=_tiny_records(),
                teacher_jsonl=teacher_path,
                base_checkpoint=stage_a["checkpoint_path"],
                save_dir=os.path.join(tmp, "ranker_ckpts"),
                profile_path=os.path.join(tmp, "ranker_profile.jsonl"),
                steps=2,
                batch_records=1,
                max_pairs_per_record=3,
                device="cpu",
                seed=119,
            )

            self.assertTrue(os.path.exists(result["checkpoint_path"]))
            self.assertTrue(os.path.exists(result["profile_path"]))
            self.assertTrue(math.isfinite(float(result["best_loss"])))
            self.assertEqual(result["usable_transcripts"], 1)
            with open(result["profile_path"], "r", encoding="utf-8") as fh:
                profile_rows = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual(len(profile_rows), 2)
            self.assertTrue(all(row["stage"] == "proposal_ranker" for row in profile_rows))
            self.assertTrue(all(row["pair_count"] > 0 for row in profile_rows))

    def test_proposal_ranker_source_balanced_pairs_use_source_scores(self):
        with tempfile.TemporaryDirectory(prefix="mef_rank_source_balanced_") as tmp:
            cfg = _tiny_config(tmp)
            stage_a = train_stage_a(cfg, records=_tiny_records(), steps=1, device="cpu", seed=121)
            teacher_path = os.path.join(tmp, "hybrid_teacher.jsonl")
            rows = [
                {
                    "transcript_id": "r1",
                    "op": "sub",
                    "pos": 0,
                    "nt": "G",
                    "teacher_score": 0.05,
                    "source_scores": {"full": 0.20, "utr": -0.10},
                },
                {
                    "transcript_id": "r1",
                    "op": "sub",
                    "pos": 1,
                    "nt": "U",
                    "teacher_score": 0.12,
                    "source_scores": {"full": -0.05, "utr": 0.30},
                },
                {
                    "transcript_id": "r1",
                    "op": "sub",
                    "pos": 2,
                    "nt": "A",
                    "teacher_score": 0.04,
                    "source_scores": {"full": 0.08, "utr": 0.00},
                },
            ]
            with open(teacher_path, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, sort_keys=True) + "\n")

            result = train_proposal_ranker(
                records=_tiny_records(),
                teacher_jsonl=teacher_path,
                base_checkpoint=stage_a["checkpoint_path"],
                save_dir=os.path.join(tmp, "ranker_ckpts"),
                profile_path=os.path.join(tmp, "ranker_profile.jsonl"),
                steps=2,
                batch_records=1,
                max_pairs_per_record=4,
                device="cpu",
                seed=123,
                pair_source_mode="source_balanced",
            )

            self.assertEqual(result["pair_source_mode"], "source_balanced")
            with open(result["profile_path"], "r", encoding="utf-8") as fh:
                profile_rows = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual(len(profile_rows), 2)
            for row in profile_rows:
                self.assertEqual(row["pair_source_mode"], "source_balanced")
                self.assertGreater(row["pair_source_counts"].get("full", 0), 0)
                self.assertGreater(row["pair_source_counts"].get("utr", 0), 0)

    def test_pairing_mode_invariant_global_uses_teacher_score_balanced_ignores_it(self):
        """Guard the fusion-ablation invariant.

        A multi-objective fusion ablation varies only ``teacher_score`` across
        modes while ``source_scores`` stay identical (they are the per-objective
        normalized levels). ``global`` pairing must derive ``teacher_delta`` from
        ``teacher_score`` (so fusion modes train different rankers), whereas
        ``source_balanced`` derives it from ``source_scores`` and therefore
        ignores ``teacher_score`` entirely (which would collapse a fusion
        ablation to one ranker). This test locks that contract so the trap that
        nulled the first ablation attempts cannot silently return.
        """
        # Two rows whose teacher_score ORDER is the OPPOSITE of every
        # source_scores label order, so the two modes cannot coincidentally agree.
        rows = [
            ProposalTeacherRow(
                transcript_id="r1", op="sub", pos=0, nt="G",
                teacher_score=0.90,
                source_scores={"a": 0.10, "b": 0.10},
            ),
            ProposalTeacherRow(
                transcript_id="r1", op="sub", pos=1, nt="U",
                teacher_score=0.10,
                source_scores={"a": 0.90, "b": 0.90},
            ),
        ]

        global_specs = _pair_specs(
            rows, max_pairs=8, min_teacher_delta=1e-9, pair_source_mode="global"
        )
        balanced_specs = _pair_specs(
            rows, max_pairs=8, min_teacher_delta=1e-9, pair_source_mode="source_balanced"
        )

        self.assertTrue(global_specs)
        self.assertTrue(balanced_specs)

        # global: every pair's teacher_delta equals the teacher_score difference.
        for spec in global_specs:
            self.assertEqual(spec.source_label, "global")
            expected = rows[spec.i].teacher_score - rows[spec.j].teacher_score
            self.assertAlmostEqual(spec.teacher_delta, expected, places=9)

        # source_balanced: every pair's teacher_delta equals the source_scores
        # difference for its label, and NEVER the teacher_score difference here
        # (constructed to differ in sign), proving teacher_score is ignored.
        for spec in balanced_specs:
            self.assertIn(spec.source_label, {"a", "b"})
            expected = (
                rows[spec.i].source_scores[spec.source_label]
                - rows[spec.j].source_scores[spec.source_label]
            )
            self.assertAlmostEqual(spec.teacher_delta, expected, places=9)
            teacher_diff = rows[spec.i].teacher_score - rows[spec.j].teacher_score
            # Opposite sign => source_balanced did not use teacher_score.
            self.assertNotAlmostEqual(spec.teacher_delta, teacher_diff, places=6)


class TestStageBAdapterSmoke(unittest.TestCase):
    def test_train_adapter_updates_only_adapter_and_task_head(self):
        with tempfile.TemporaryDirectory(prefix="mef_stage_b_") as tmp:
            cfg = _tiny_config(tmp)
            result = train_stage_b(
                cfg,
                records=_tiny_records(),
                task_id="T5",
                steps=2,
                device="cpu",
                seed=202,
                adapter_rank=2,
            )

            self.assertTrue(os.path.exists(result["checkpoint_path"]))
            self.assertEqual(result["changed_frozen_names"], [])
            self.assertGreater(len(result["changed_trainable_names"]), 0)

            allowed = (
                "adapters.",
                "property_emb.",
                "property_film.",
                "task_norm.",
                "rates_out.",
                "ins_logits.",
                "sub_logits.",
                "aux_struct.",
            )
            for name in result["trainable_names"]:
                self.assertTrue(name.startswith(allowed), name)
            for name in result["changed_trainable_names"]:
                self.assertTrue(name.startswith(allowed), name)

            with open(result["profile_path"], "r", encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["stage"] == "B" and row["finite_loss"] for row in rows))

    def test_train_region_adapter_updates_only_region_adapters(self):
        with tempfile.TemporaryDirectory(prefix="mef_stage_b_region_") as tmp:
            cfg = _tiny_config(tmp)
            result = train_region_stage_b(
                cfg,
                records=_tiny_records(),
                task_id="T5",
                steps=2,
                device="cpu",
                seed=303,
                adapter_bottleneck=4,
            )

            self.assertTrue(os.path.exists(result["checkpoint_path"]))
            self.assertEqual(result["region_ids"], [REGION_5UTR, REGION_CDS, REGION_3UTR])
            self.assertEqual(result["changed_frozen_names"], [])
            self.assertGreater(len(result["changed_trainable_names"]), 0)
            for name in result["trainable_names"]:
                self.assertTrue(name.startswith("adapters."), name)
            for name in result["changed_trainable_names"]:
                self.assertTrue(name.startswith("adapters."), name)

            with open(result["profile_path"], "r", encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["stage"] == "B_region" and row["finite_loss"] for row in rows))

    def test_region_adapter_checkpoint_can_drive_candidate_generation(self):
        with tempfile.TemporaryDirectory(prefix="mef_stage_b_region_gen_") as tmp:
            cfg = _tiny_config(tmp)
            result = train_region_stage_b(
                cfg,
                records=_tiny_records(),
                task_id="T5",
                steps=1,
                device="cpu",
                seed=404,
                adapter_bottleneck=4,
                regions=[REGION_5UTR, REGION_CDS, REGION_3UTR],
            )
            _cfg, _backbone, model = load_stage_a_checkpoint(result["checkpoint_path"], device="cpu")
            self.assertEqual(model.__class__.__name__, "RegionSpecializedEditFormer")

            records = _tiny_records()[:2]
            candidates = generate_candidate_records(
                records,
                task_id="T5",
                checkpoint_path=result["checkpoint_path"],
                limit=2,
                edit_budget=2,
                proposal_top_k=4,
                device="cpu",
            )
            self.assertEqual(len(candidates), 2)
            for src, cand in zip(records, candidates):
                self.assertTrue(is_valid_cds(cand.cds))
                self.assertEqual(src.cds, cand.cds)
                self.assertLessEqual(levenshtein_distance(src.seq, cand.seq), 2)


class TestRegionAwareSampling(unittest.TestCase):
    def setUp(self):
        self.record = MRNARecord("base", "AACCGG", "AUGGCUUAA", "CCGGAA")

    def assertFiniteLegalGuidedRecord(self, out, oracle):
        self.assertIsInstance(out, MRNARecord)
        self.assertTrue(set(out.seq) <= set("ACGU"))
        self.assertTrue(is_valid_cds(out.cds))
        score = oracle.score_record(out)
        self.assertTrue(math.isfinite(float(score["ensemble_te"])))
        self.assertGreaterEqual(float(score["ensemble_te"]), 0.0)
        self.assertLessEqual(float(score["ensemble_te"]), 1.0)
        self.assertTrue(math.isfinite(float(score["features"]["start_accessibility"])))

    def test_t1_unconditional_returns_valid_record_or_sequence(self):
        rec = sample_mrna(task_id="T1", seed=7, return_record=True)
        self.assertIsInstance(rec, MRNARecord)
        self.assertEqual(rec.seq, rec.five_utr + rec.cds + rec.three_utr)
        self.assertTrue(is_valid_cds(rec.cds))
        seq = sample_sequence(task_id="T1", seed=7)
        self.assertIsInstance(seq, str)
        self.assertTrue(set(seq) <= set("ACGU"))

    def test_t5_obeys_edit_budget(self):
        out = sample_mrna(
            task_id="T5",
            record=self.record,
            edit_budget=3,
            seed=11,
            return_record=True,
        )
        self.assertLessEqual(levenshtein_distance(self.record.seq, out.seq), 3)
        self.assertEqual(out.cds, self.record.cds)
        self.assertTrue(is_valid_cds(out.cds))

    def test_t6_moves_toward_target_length_without_touching_cds(self):
        target = len(self.record.seq) + 7
        out = sample_mrna(
            task_id="T6",
            record=self.record,
            target_length=target,
            seed=13,
            return_record=True,
        )
        self.assertLess(abs(len(out.seq) - target), abs(len(self.record.seq) - target))
        self.assertEqual(out.cds, self.record.cds)
        self.assertTrue(is_valid_cds(out.cds))

    def test_t7_motif_insertion_and_excision(self):
        motif = "GCCACC"
        inserted = sample_mrna(
            task_id="T7",
            record=self.record,
            motif=motif,
            motif_action="insert",
            motif_region="5utr",
            seed=19,
            return_record=True,
        )
        self.assertIn(motif, inserted.five_utr)
        self.assertEqual(inserted.cds, self.record.cds)

        excised = sample_mrna(
            task_id="T7",
            record=inserted,
            motif=motif,
            motif_action="excise",
            motif_region="5utr",
            seed=19,
            return_record=True,
        )
        self.assertNotIn(motif, excised.five_utr)
        self.assertEqual(excised.cds, self.record.cds)
        self.assertTrue(is_valid_cds(excised.cds))

        pred = LocalTranslationOracle()
        for task_id in ("T2", "T3"):
            guided = sample_mrna(
                task_id=task_id,
                record=self.record,
                edit_budget=2,
                guidance_scale=1.5,
                target_te=0.72,
                target_start_accessibility=0.65,
                oracle=pred,
                guidance_candidates=3,
                seed=31,
                return_record=True,
            )
            self.assertEqual(guided.cds, self.record.cds)
            self.assertFiniteLegalGuidedRecord(guided, pred)

        guided_inserted = sample_mrna(
            task_id="T7",
            record=self.record,
            motif=motif,
            motif_action="insert",
            motif_region="5utr",
            guidance_scale=1.0,
            target_te=0.70,
            target_start_accessibility=0.60,
            oracle=pred,
            guidance_candidates=3,
            seed=37,
            return_record=True,
        )
        self.assertIn(motif, guided_inserted.five_utr)
        self.assertEqual(guided_inserted.cds, self.record.cds)
        self.assertFiniteLegalGuidedRecord(guided_inserted, pred)

    def test_t4_synonymous_optimization_preserves_protein(self):
        before = translate(self.record.cds)
        out = sample_mrna(
            task_id="T4",
            record=self.record,
            edit_budget=4,
            seed=23,
            return_record=True,
        )
        self.assertEqual(translate(out.cds), before)
        self.assertEqual(len(out.cds), len(self.record.cds))
        self.assertTrue(is_valid_cds(out.cds))

    def test_model_guided_t4_candidates_are_single_step_synonymous(self):
        rec = MRNARecord("arg", "AAA", "AUGCGUUAA", "CCC")
        seq_len = len(rec.seq)
        fake_out = {
            "rates": torch.ones(1, seq_len, 3),
            "sub_probs": torch.full((1, seq_len, 4), 0.25),
        }
        choices = _synonymous_substitution_candidates(rec, fake_out)
        self.assertGreater(len(choices), 0)

        unsafe_arg_to_ser = (len(rec.five_utr) + 3, "A")
        seen = {(pos, nt) for _score, pos, nt in choices}
        self.assertNotIn(unsafe_arg_to_ser, seen)
        for _score, pos, nt in choices:
            cand = _replace_nt(rec, pos, nt, "arg_candidate")
            self.assertEqual(translate(rec.cds), translate(cand.cds))


if __name__ == "__main__":
    unittest.main(verbosity=2)
