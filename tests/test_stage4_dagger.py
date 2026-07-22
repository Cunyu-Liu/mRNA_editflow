"""Stage 4 offline DAgger rollout/relabel/replay contract tests."""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from mrna_editflow.rl.dagger_teacher_export import (
    DaggerRolloutConfig,
    export_dagger_teacher_jsonl,
    relabel_trajectory,
    rollout_model_guided_trajectory,
)
from mrna_editflow.rl.decoder_state import DecoderAction, DecoderState, choose_stop_aware_action, sequence_hash
from mrna_editflow.rl.rollout_buffer import ReplayBuffer, ReplayMixConfig, iteration_directory
from mrna_editflow.rl.trajectory_schema import OfflineTrajectory, TrajectoryState, validate_telescoping
from mrna_editflow.sample import load_stage_a_checkpoint
from mrna_editflow.baselines.multiobjective_teacher_export import MultiObjectiveConfig, export_multiobjective_teacher_jsonl
from mrna_editflow.train_dagger_ranker import DaggerIterationConfig, train_dagger_iteration
from mrna_editflow.train_backbone import train_stage_a
from mrna_editflow.tests.test_training_sampling import _tiny_config, _tiny_records


class TestStage4Dagger(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="stage4_dagger_")
        stage_a = train_stage_a(_tiny_config(cls.tmp.name), records=_tiny_records(), steps=1, device="cpu", seed=801)
        cls.checkpoint = stage_a["checkpoint_path"]
        _cfg, cls.backbone, cls.model = load_stage_a_checkpoint(cls.checkpoint, device="cpu")
        cls.record = _tiny_records()[0]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_model_only_rollout_visits_intermediate_state_and_records_hash(self):
        config = DaggerRolloutConfig(edit_budget=1, proposal_top_k=1, proposal_temperature=0.0, allow_stop=False)
        with mock.patch("mrna_editflow.rl.dagger_teacher_export.LocalTranslationOracle", side_effect=AssertionError("oracle must not be constructed during rollout")):
            trajectory = rollout_model_guided_trajectory(
                self.record, self.model, self.backbone, config=config, seed=802, device="cpu"
            )
        self.assertGreaterEqual(len(trajectory.states), 2)
        self.assertNotEqual(trajectory.states[0].sequence, trajectory.states[1].sequence)
        self.assertEqual(trajectory.policy_checkpoint_sha256, getattr(self.model, "_checkpoint_sha256"))
        self.assertFalse(trajectory.to_dict()["oracle_used_for_action_selection"])

    def test_relabel_records_stop_provenance_and_potential_telescoping(self):
        trajectory = rollout_model_guided_trajectory(
            self.record, self.model, self.backbone,
            config=DaggerRolloutConfig(edit_budget=2, proposal_top_k=1, proposal_temperature=0.0, allow_stop=False),
            seed=803, device="cpu",
        )
        rows, state_records, summary = relabel_trajectory(trajectory)
        self.assertTrue(rows)
        self.assertTrue(state_records)
        self.assertTrue(any(row.candidate_action["op"] == "stop" for row in rows))
        self.assertTrue(all(row.policy_checkpoint_sha256 == trajectory.policy_checkpoint_sha256 for row in rows))
        self.assertTrue(all(row.source_properties and row.state_properties and row.candidate_properties for row in rows))
        self.assertTrue(summary["trajectory_telescoping_valid"])
        self.assertTrue(validate_telescoping([{"te": 0.0}, {"te": 0.3}, {"te": -0.2}]))

    def test_stop_is_retained_when_decoder_stops_before_an_edit(self):
        trajectory = rollout_model_guided_trajectory(
            self.record, self.model, self.backbone,
            config=DaggerRolloutConfig(edit_budget=2, proposal_top_k=1, proposal_temperature=0.0, allow_stop=True, min_action_margin=1e9),
            seed=804, device="cpu",
        )
        self.assertTrue(trajectory.terminated)
        self.assertEqual(trajectory.action_history[-1].op, "stop")
        self.assertEqual(trajectory.cycle_rejections, 0)

    def test_visited_state_cycle_guard_still_forces_stop(self):
        state = DecoderState("AAAA", 2)
        action = DecoderAction("sub", 0, "G", 1.0, sequence_hash("AAAA"), "A")
        chosen = choose_stop_aware_action([action], state, __import__("random").Random(806), top_k=1, temperature=0.0)
        self.assertIsNone(chosen)
        self.assertTrue(state.terminated_by_stop)
        self.assertGreaterEqual(state.cycle_rejections, 1)

    def test_replay_mixture_has_exact_configured_proportions(self):
        buffer = ReplayBuffer(policy_checkpoint_sha256="a" * 64, iteration=1)
        for bucket in ("original", "rollout", "hard_negative", "stop"):
            buffer.add([{"transcript_id": f"{bucket}_{i}", "state_sequence": f"S{i}"} for i in range(10)], bucket=bucket)
        rows = buffer.sample_mixed(10, config=ReplayMixConfig(), seed=805)
        counts = {bucket: sum(row["replay_bucket"] == bucket for row in rows) for bucket in ("original", "rollout", "hard_negative", "stop")}
        self.assertEqual(counts, {"original": 4, "rollout": 4, "hard_negative": 1, "stop": 1})

    def test_iteration_artifacts_do_not_overwrite_and_validation_ids_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="stage4_paths_") as tmp:
            path = iteration_directory(tmp, 1)
            self.assertTrue(os.path.isdir(path))
            with self.assertRaises(FileExistsError):
                iteration_directory(tmp, 1)
            trajectory = OfflineTrajectory(
                source_transcript_id="validation_only",
                task_id="T5",
                policy_checkpoint_sha256="b" * 64,
                states=[TrajectoryState("s0", "AA", "A", "AUGUAA", "", 0)],
            )
            with self.assertRaises(ValueError):
                export_dagger_teacher_jsonl(
                    [trajectory], out_jsonl=os.path.join(tmp, "teacher.jsonl"),
                    out_states_jsonl=os.path.join(tmp, "states.jsonl"),
                    out_trajectories_jsonl=os.path.join(tmp, "trajectories.jsonl"),
                    out_summary_json=os.path.join(tmp, "summary.json"),
                    train_transcript_ids=["train_only"], validation_transcript_ids=["validation_only"],
                )

    def test_one_iteration_smoke_writes_isolated_manifest(self):
        with tempfile.TemporaryDirectory(prefix="stage4_iteration_") as tmp:
            train_teacher = os.path.join(tmp, "train_teacher.jsonl")
            train_summary = os.path.join(tmp, "train_summary.json")
            val_teacher = os.path.join(tmp, "val_teacher.jsonl")
            val_summary = os.path.join(tmp, "val_summary.json")
            teacher_cfg = MultiObjectiveConfig(max_edit_positions=2, candidate_cap=8)
            export_multiobjective_teacher_jsonl([self.record], out_jsonl=train_teacher, out_json=train_summary, config=teacher_cfg)
            export_multiobjective_teacher_jsonl([_tiny_records()[1]], out_jsonl=val_teacher, out_json=val_summary, config=teacher_cfg)
            result = train_dagger_iteration(
                train_records=[self.record], validation_records=[_tiny_records()[1]],
                original_train_teacher_jsonl=train_teacher, validation_teacher_jsonl=val_teacher,
                policy_checkpoint=self.checkpoint, output_root=os.path.join(tmp, "iterations"),
                config=DaggerIterationConfig(
                    iteration=1, rollout_edit_budget=1, rollout_top_k=1, rollout_temperature=0.0,
                    rollout_allow_stop=False, mixed_rows=10, ranker_steps=1,
                    ranker_batch_records=1, ranker_validation_interval=1, seed=807,
                ), device="cpu",
            )
            self.assertTrue(os.path.isfile(result["iteration_manifest"]))
            self.assertEqual(result["metrics"]["validation_transcripts_used_for_dagger"], 0)
            self.assertIsNotNone(result["metrics"]["validation_regret"])
            self.assertIsNotNone(result["metrics"]["positive_edit_precision"])
            self.assertFalse(result["online_grpo"])


if __name__ == "__main__":
    unittest.main()
