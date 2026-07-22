"""Stage 3 vector-reward schema, redundancy, preference, and compatibility tests."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import torch

from mrna_editflow.baselines.multiobjective_teacher_export import MultiObjectiveConfig, score_record_multiobjective_rows
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.rl.preference_conditioning import (
    PreferenceConditionedObjectiveHead, combine_reward_vector, preference_from_name,
    combine_objective_scores, select_with_absolute_stop,
)
from mrna_editflow.rl.reward_diagnostics import reward_correlation_report, write_reward_correlation_report
from mrna_editflow.rl.reward_vector import RewardComponent, RewardVector
from mrna_editflow.train_backbone import train_stage_a
from mrna_editflow.train_proposal_ranker import ProposalTeacherRow, load_objective_head_from_checkpoint, load_teacher_rows, train_proposal_ranker
from mrna_editflow.tests.test_training_sampling import _tiny_config, _tiny_records


def _vector(te: float, access: float, *, uncertainty: float = 0.0, cai_valid: bool = False) -> RewardVector:
    components = [
        RewardComponent("te", te, "oracle", "functional", True, uncertainty, True),
        RewardComponent("access", access, "oracle_shared", "structure", True, uncertainty, True),
        RewardComponent("cai", 0.0, "cai", "manufacturability", True, None, cai_valid),
        RewardComponent("uncertainty", -uncertainty, "oracle", "functional", True, uncertainty, True),
        RewardComponent("protein_identity", 0.0, "gate", "hard_constraint", False, None, True),
    ]
    return RewardVector(
        raw_absolute_level={"te": te, "access": access},
        raw_delta_from_source={"te": te, "access": access, "cai": 0.0, "uncertainty": -uncertainty},
        normalized_within_group={"te": te, "access": access}, components=components,
        validity={component.name: component.valid for component in components},
        redundancy_groups=("te_access_shared_oracle_features",),
    )


class TestStage3VectorReward(unittest.TestCase):
    def test_linear_mrl_te_redundancy_and_report_serialization(self):
        vectors = [
            RewardVector.from_legacy({"te": value, "mrl": 8.0 * value + 2.0}, {})
            for value in (-1.0, 0.0, 1.0)
        ]
        report = reward_correlation_report(vectors)
        self.assertIn(["mrl", "te"], report["redundancy_groups"])
        with tempfile.TemporaryDirectory(prefix="reward_report_") as tmp:
            jpath, mpath = os.path.join(tmp, "report.json"), os.path.join(tmp, "report.md")
            write_reward_correlation_report(report, jpath, mpath)
            self.assertTrue(os.path.exists(jpath))
            self.assertIn("Redundancy groups", open(mpath, encoding="utf-8").read())

    def test_utr_cai_invalid_and_stop_wins_all_negative(self):
        rows = score_record_multiobjective_rows(
            MRNARecord("r", "AAACCC", "AUGGCCUAA", "GGG"),
            config=MultiObjectiveConfig(max_edit_positions=2),
        )
        self.assertTrue(any(row.op == "stop" for row in rows))
        non_stop = next(row for row in rows if row.op != "stop")
        vector = RewardVector.from_dict(non_stop.reward_vector)
        self.assertFalse(vector.validity["cai"])
        self.assertNotIn("cai", [component.name for component in vector.components if component.valid and component.independent])
        self.assertIsNone(select_with_absolute_stop([_vector(-0.1, -0.2), _vector(-0.2, -0.1)], {"te": 1.0, "access": 1.0}))

    def test_preference_and_uncertainty_change_ranking(self):
        left, right = _vector(1.0, 0.0), _vector(0.0, 1.0)
        self.assertEqual(select_with_absolute_stop([left, right], {"te": 3.0, "access": 1.0}), 0)
        self.assertEqual(select_with_absolute_stop([left, right], {"te": 1.0, "access": 3.0}), 1)
        certain, uncertain = _vector(0.5, 0.0, uncertainty=0.0), _vector(0.8, 0.0, uncertainty=1.0)
        self.assertGreater(combine_reward_vector(uncertain, {"te": 1.0}), combine_reward_vector(certain, {"te": 1.0}))
        self.assertLess(combine_reward_vector(uncertain, {"te": 1.0}, uncertainty_penalty=1.0), combine_reward_vector(certain, {"te": 1.0}, uncertainty_penalty=1.0))

    def test_independent_objective_heads_are_not_one_label(self):
        head = PreferenceConditionedObjectiveHead(("te", "access"))
        self.assertEqual(set(head.state_dict()), {"heads.te.weight", "heads.te.bias", "heads.access.weight", "heads.access.bias"})
        output = head(torch.tensor([1.0, 2.0]), torch.tensor([1, 1]))
        self.assertEqual(set(output), {"te", "access"})
        self.assertEqual(tuple(output["te"].shape), (2,))
        combined = combine_objective_scores(output, preference_from_name("translation_focused"))
        self.assertEqual(tuple(combined.shape), (2,))

    def test_legacy_teacher_load_and_vector_checkpoint_schema(self):
        legacy = ProposalTeacherRow.from_json({"transcript_id": "r", "op": "sub", "pos": 0, "nt": "G", "teacher_score": 0.1})
        self.assertEqual(legacy.source_scores, {"teacher": 0.1})
        with tempfile.TemporaryDirectory(prefix="stage3_ranker_") as tmp:
            stage_a = train_stage_a(_tiny_config(tmp), records=_tiny_records(), steps=1, device="cpu", seed=701)
            train_path, val_path = os.path.join(tmp, "train.jsonl"), os.path.join(tmp, "val.jsonl")
            vector = _vector(0.2, 0.1).to_dict()
            rows = [
                {"op": "sub", "pos": 0, "nt": "G", "teacher_score": 0.2, "source_scores": {"te": 0.9, "access": 0.6}, "reward_vector": vector, "raw_delta_from_source": vector["raw_delta_from_source"], "validity": vector["validity"]},
                {"op": "sub", "pos": 1, "nt": "U", "teacher_score": -0.1, "source_scores": {"te": 0.1, "access": 0.2}, "reward_vector": _vector(-0.1, -0.2).to_dict(), "raw_delta_from_source": _vector(-0.1, -0.2).raw_delta_from_source, "validity": _vector(-0.1, -0.2).validity},
            ]
            for path, transcript_id in ((train_path, "r1"), (val_path, "r2")):
                with open(path, "w", encoding="utf-8") as fh:
                    for row in rows:
                        emitted = dict(row, transcript_id=transcript_id)
                        fh.write(json.dumps(emitted) + "\n")
            result = train_proposal_ranker(
                train_records=_tiny_records()[:1], val_records=_tiny_records()[1:2],
                train_teacher_jsonl=train_path, val_teacher_jsonl=val_path,
                base_checkpoint=stage_a["checkpoint_path"], save_dir=os.path.join(tmp, "ranker"),
                profile_path=os.path.join(tmp, "profile.jsonl"), steps=1, batch_records=1,
                max_pairs_per_record=2, validation_interval=1, device="cpu", seed=702,
            )
            payload = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=False)
            self.assertIn("te", payload["objective_schema"]["objective_names"])
            self.assertIsNotNone(payload["objective_head_state"])
            loaded_head, schema = load_objective_head_from_checkpoint(result["checkpoint_path"])
            self.assertEqual(schema["objective_names"], result["objective_schema"]["objective_names"])
            self.assertTrue(loaded_head.objective_names)


if __name__ == "__main__":
    unittest.main()
