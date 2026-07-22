"""Stage 5 constrained GRPO correctness and short-run smoke coverage."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import torch

from mrna_editflow.eval.artifact_contract import OracleContractError
from mrna_editflow.rl.grpo import GRPOConfig, clipped_policy_loss, group_advantages
from mrna_editflow.rl.kl_regularization import AdaptiveKLController
from mrna_editflow.rl.policy import Policy, PolicyConfig
from mrna_editflow.rl.trajectory_sampler import SamplerConfig, constrained_distribution, sample_group
from mrna_editflow.sample import load_stage_a_checkpoint
from mrna_editflow.train_backbone import train_stage_a
from mrna_editflow.train_grpo import TrainGRPOConfig, train_grpo
from mrna_editflow.tests.test_training_sampling import _tiny_config, _tiny_records


class TestStage5ConstrainedGRPO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="stage5_grpo_")
        cls.stage_a = train_stage_a(_tiny_config(cls.tmp.name), records=_tiny_records(), steps=1, device="cpu", seed=901)
        _cfg, cls.backbone, cls.model = load_stage_a_checkpoint(cls.stage_a["checkpoint_path"], device="cpu")
        cls.record = _tiny_records()[0]

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_task_masks_make_illegal_actions_exactly_zero(self):
        policy = Policy(self.model, self.backbone, PolicyConfig(), torch.device("cpu"))
        actions, log_probs, _hash = constrained_distribution(policy, self.record, cfg=SamplerConfig(task_id="T5"), budget_remaining=2)
        self.assertTrue(all(action.is_stop() or (action.op == "sub" and action.pos < len(self.record.five_utr)) for action in actions))
        # No insertion/deletion/3'UTR action appears in the support, hence its probability is exactly zero.
        self.assertAlmostEqual(float(torch.exp(log_probs).sum()), 1.0, places=6)

    def test_clipped_ratio_and_constant_groups_are_finite(self):
        new, old, advantage = torch.tensor([0.2]), torch.tensor([0.0]), torch.tensor([2.0])
        loss, ratio = clipped_policy_loss(new, old, advantage, clip_epsilon=0.1)
        self.assertAlmostEqual(float(ratio), float(torch.exp(torch.tensor(0.2))), places=6)
        self.assertAlmostEqual(float(loss), -2.2, places=5)
        grouped = group_advantages([{"te": 1.0}, {"te": 1.0}], {"te": 1.0}, config=GRPOConfig())
        self.assertEqual(grouped.constant_objective_count, 1)
        self.assertTrue(torch.isfinite(grouped.aggregated).all())
        self.assertEqual(grouped.audit_dict()["per_objective_advantages"]["te"], [0.0, 0.0])

    def test_stop_and_t4_protein_identity_are_preserved(self):
        policy = Policy(self.model, self.backbone, PolicyConfig(), torch.device("cpu"))
        reference = Policy(self.model, self.backbone, PolicyConfig(), torch.device("cpu"))
        actions, _lps, _hash = constrained_distribution(policy, self.record, cfg=SamplerConfig(task_id="T5"), budget_remaining=0)
        self.assertEqual(len(actions), 1); self.assertTrue(actions[0].is_stop())
        trajectories = sample_group(self.record, policy, reference, group_size=3, cfg=SamplerConfig(task_id="T4", max_edits=1), seed=902)
        self.assertTrue(all(item.final_record.cds == self.record.cds or item.final_record.cds != "" for item in trajectories))
        from mrna_editflow.core.constants import translate
        self.assertTrue(all(translate(item.final_record.cds) == translate(self.record.cds) for item in trajectories))

    def test_kl_guard_and_paper_rejection(self):
        controller = AdaptiveKLController(max_kl=0.01)
        _coefficient, skip = controller.update(0.02)
        self.assertTrue(skip)
        with self.assertRaises(OracleContractError):
            train_grpo(records=[self.record], base_checkpoint=self.stage_a["checkpoint_path"], save_dir=os.path.join(self.tmp.name, "paper"), profile_path=os.path.join(self.tmp.name, "paper.jsonl"), config=TrainGRPOConfig(steps=1), device="cpu", run_mode="paper")

    def test_update_reference_freeze_resume_and_seed_reproducibility(self):
        def run(name: str, steps: int, resume: str | None = None):
            return train_grpo(records=[self.record], base_checkpoint=self.stage_a["checkpoint_path"], save_dir=os.path.join(self.tmp.name, name), profile_path=os.path.join(self.tmp.name, f"{name}.jsonl"), config=TrainGRPOConfig(steps=steps, group_size=2, max_edits=1, checkpoint_interval=1, seed=903), resume_checkpoint=resume, device="cpu")
        first = run("first", 1)
        payload = torch.load(first["checkpoint_path"], map_location="cpu", weights_only=False)
        base = torch.load(self.stage_a["checkpoint_path"], map_location="cpu", weights_only=False)
        self.assertTrue(any(not torch.equal(payload["model_state"][key], base["model_state"][key]) for key in payload["model_state"]))
        second = run("second", 2, first["checkpoint_path"])
        resumed = torch.load(second["checkpoint_path"], map_location="cpu", weights_only=False)
        self.assertEqual(resumed["step"], 2)
        replica = run("replica", 1)
        replica_payload = torch.load(replica["checkpoint_path"], map_location="cpu", weights_only=False)
        self.assertTrue(all(torch.equal(payload["model_state"][key], replica_payload["model_state"][key]) for key in payload["model_state"]))
        with open(first["profile_path"], encoding="utf-8") as fh:
            profile = json.loads(next(line for line in fh if line.strip()))
        self.assertTrue(profile["updated"]); self.assertTrue(first["reference_frozen"]); self.assertTrue(first["reference_unchanged"])
        audit_step = profile["sampled_trajectories"][0]["steps"][0]
        self.assertTrue({"state", "action", "old_log_prob", "reference_log_prob", "action_mask_hash", "reward_components", "termination_reason"}.issubset(audit_step))


if __name__ == "__main__": unittest.main()
