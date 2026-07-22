"""Stage 2 validation, checkpoint-selection, and leakage tests."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import torch

from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.split_contract import TrainValidationOverlapError, assert_train_validation_disjoint
from mrna_editflow.rl.ranking_metrics import RankingCandidate, compute_ranking_metrics
from mrna_editflow.train_backbone import train_stage_a
from mrna_editflow.train_proposal_ranker import _parse_args, train_proposal_ranker
from mrna_editflow.tests.test_training_sampling import _tiny_config, _tiny_records


def _write_teacher(path: str, transcript_id: str) -> None:
    rows = [
        {"transcript_id": transcript_id, "op": "sub", "pos": 0, "nt": "G", "teacher_score": 0.2},
        {"transcript_id": transcript_id, "op": "sub", "pos": 1, "nt": "U", "teacher_score": 0.0},
        {"transcript_id": transcript_id, "op": "stop", "pos": None, "nt": "", "teacher_score": 0.0},
    ]
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _summary(regret: float) -> dict[str, object]:
    return {
        "candidate_pool_scope": "global",
        "candidate_cap": 0,
        "n_records": 1,
        "mean_model_regret": regret,
        "median_model_regret": regret,
        "oracle_best_recall_at_1": 1.0,
        "oracle_best_recall_at_8": 1.0,
        "oracle_best_recall_at_32": 1.0,
        "ndcg_at_8": 1.0,
        "ndcg_at_32": 1.0,
        "positive_edit_precision_at_1": 1.0,
        "positive_edit_precision_at_8": 1.0,
        "positive_edit_precision_at_32": 1.0,
        "mean_selected_teacher_delta": 0.2,
        "stop_accuracy": 0.0,
        "stop_accuracy_denominator": 0,
    }


class TestRankingMetrics(unittest.TestCase):
    def test_hand_constructed_metrics_and_stop_accuracy(self):
        result = compute_ranking_metrics(
            {
                "a": [
                    RankingCandidate("sub", 1.0, 0.0),
                    RankingCandidate("sub", 0.0, 2.0),
                    RankingCandidate("sub", -1.0, 1.0),
                ],
                "b": [
                    RankingCandidate("stop", 0.0, 3.0),
                    RankingCandidate("sub", -0.2, 1.0),
                ],
            }
        )
        self.assertEqual(result["candidate_pool_scope"], "global")
        self.assertAlmostEqual(float(result["mean_model_regret"]), 0.5)
        self.assertAlmostEqual(float(result["oracle_best_recall_at_1"]), 0.5)
        self.assertAlmostEqual(float(result["oracle_best_recall_at_8"]), 1.0)
        self.assertAlmostEqual(float(result["positive_edit_precision_at_1"]), 0.0)
        self.assertAlmostEqual(float(result["positive_edit_precision_at_8"]), 1.0 / 6.0)
        self.assertAlmostEqual(float(result["ndcg_at_8"]), 0.25)
        self.assertAlmostEqual(float(result["mean_selected_teacher_delta"]), 0.0)
        self.assertEqual(result["stop_accuracy_denominator"], 1)
        self.assertEqual(result["stop_accuracy"], 1.0)

    def test_capped_pool_is_explicitly_restricted(self):
        result = compute_ranking_metrics(
            {"a": [RankingCandidate("sub", 1.0, 0.0), RankingCandidate("sub", -1.0, 2.0)]},
            candidate_cap=1,
        )
        self.assertEqual(result["candidate_pool_scope"], "restricted")
        self.assertIn("restricted_mean_model_regret", result)
        self.assertNotIn("mean_model_regret", result)


class TestRankerValidationProtocol(unittest.TestCase):
    def test_cli_exposes_independent_validation_and_selection_controls(self):
        args = _parse_args([
            "--base-checkpoint", "base.pt", "--save-dir", "out", "--profile-path", "profile.jsonl",
            "--train-records-jsonl", "train.jsonl", "--val-records-jsonl", "val.jsonl",
            "--train-teacher-jsonl", "train_teacher.jsonl", "--val-teacher-jsonl", "val_teacher.jsonl",
            "--checkpoint-metric", "val_ndcg_at_32", "--checkpoint-mode", "max",
            "--validation-interval", "7", "--early-stopping-patience", "3",
            "--minimum-improvement", "0.02",
        ])
        self.assertEqual(args.train_records_jsonl, "train.jsonl")
        self.assertEqual(args.val_records_jsonl, "val.jsonl")
        self.assertEqual(args.checkpoint_metric, "val_ndcg_at_32")
        self.assertEqual(args.checkpoint_mode, "max")
        self.assertEqual(args.validation_interval, 7)

    def test_overlap_rejected_for_id_sequence_and_family(self):
        base = MRNARecord("same", "AAA", "AUGGCCUAA", "CCC")
        with self.assertRaises(TrainValidationOverlapError):
            assert_train_validation_disjoint([base], [base])
        with self.assertRaises(TrainValidationOverlapError):
            train_proposal_ranker(
                base_checkpoint="unused-before-overlap-check.pt",
                save_dir="/tmp/stage2_overlap_ranker",
                profile_path="/tmp/stage2_overlap_profile.jsonl",
                train_records=[base], val_records=[base],
                train_teacher_jsonl="unused-train.jsonl",
                val_teacher_jsonl="unused-val.jsonl",
                device="cpu",
            )
        with self.assertRaises(TrainValidationOverlapError):
            assert_train_validation_disjoint(
                [MRNARecord("train", "AAA", "AUGGCCUAA", "CCC")],
                [MRNARecord("val", "AAA", "AUGGCCUAA", "CCC")],
            )
        with self.assertRaises(TrainValidationOverlapError):
            assert_train_validation_disjoint(
                [MRNARecord("train", "AAA", "AUGGCCUAA", "CCC", metadata={"family_cluster": "f"})],
                [MRNARecord("val", "GGG", "AUGGCUUAA", "UUU", metadata={"family_cluster": "f"})],
            )
        contract = SimpleNamespace(
            family_disjoint=True,
            cluster_assignments=(7, 7),
            roles={
                "train": SimpleNamespace(indices=(0,)),
                "val": SimpleNamespace(indices=(1,)),
            },
        )
        with self.assertRaises(TrainValidationOverlapError):
            assert_train_validation_disjoint(
                [MRNARecord("train", "AAA", "AUGGCCUAA", "CCC")],
                [MRNARecord("val", "GGG", "AUGGCUUAA", "UUU")],
                split_contract=contract,
            )

    def test_checkpoint_selection_and_early_stopping_use_validation_only(self):
        with tempfile.TemporaryDirectory(prefix="mef_stage2_validation_") as tmp:
            stage_a = train_stage_a(_tiny_config(tmp), records=_tiny_records(), steps=1, device="cpu", seed=301)
            train_teacher = os.path.join(tmp, "train.jsonl")
            val_teacher = os.path.join(tmp, "val.jsonl")
            _write_teacher(train_teacher, "r1")
            _write_teacher(val_teacher, "r2")
            summaries = [_summary(0.1), _summary(0.4)]
            with mock.patch(
                "mrna_editflow.train_proposal_ranker.evaluate_ranker_validation",
                side_effect=summaries,
            ):
                result = train_proposal_ranker(
                    train_records=_tiny_records()[:1], val_records=_tiny_records()[1:2],
                    train_teacher_jsonl=train_teacher, val_teacher_jsonl=val_teacher,
                    base_checkpoint=stage_a["checkpoint_path"],
                    save_dir=os.path.join(tmp, "ranker"), profile_path=os.path.join(tmp, "profile.jsonl"),
                    steps=4, batch_records=1, max_pairs_per_record=2, device="cpu", seed=302,
                    validation_interval=1, early_stopping_patience=1, minimum_improvement=0.01,
                )
            self.assertEqual(result["best_validation_step"], 1)
            self.assertEqual(result["early_stopping_reason"], "no_validation_improvement_for_1_intervals")
            payload = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=False)
            self.assertEqual(payload["best_validation_step"], 1)
            self.assertAlmostEqual(payload["best_validation_metric"], 0.1)
            self.assertEqual(payload["early_stopping_reason"], result["early_stopping_reason"])
            with open(result["profile_path"], "r", encoding="utf-8") as fh:
                profile = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual(len(profile), 2)
            self.assertIn("val_mean_model_regret", profile[0])
            self.assertIn("early_stopping_reason", profile[-1])

    def test_fixed_seed_reproduces_validation_selection(self):
        with tempfile.TemporaryDirectory(prefix="mef_stage2_seed_") as tmp:
            stage_a = train_stage_a(_tiny_config(tmp), records=_tiny_records(), steps=1, device="cpu", seed=303)
            train_teacher = os.path.join(tmp, "train.jsonl")
            val_teacher = os.path.join(tmp, "val.jsonl")
            _write_teacher(train_teacher, "r1")
            _write_teacher(val_teacher, "r2")
            kwargs = dict(
                train_records=_tiny_records()[:1], val_records=_tiny_records()[1:2],
                train_teacher_jsonl=train_teacher, val_teacher_jsonl=val_teacher,
                base_checkpoint=stage_a["checkpoint_path"], steps=2, batch_records=1,
                max_pairs_per_record=2, device="cpu", seed=304, validation_interval=1,
            )
            first = train_proposal_ranker(save_dir=os.path.join(tmp, "first"), profile_path=os.path.join(tmp, "first.jsonl"), **kwargs)
            second = train_proposal_ranker(save_dir=os.path.join(tmp, "second"), profile_path=os.path.join(tmp, "second.jsonl"), **kwargs)
            self.assertEqual(first["best_validation_step"], second["best_validation_step"])
            self.assertEqual(first["validation_summary"], second["validation_summary"])


if __name__ == "__main__":
    unittest.main()
