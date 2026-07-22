"""Focused Stage 1 tests for shared scoring and STOP-aware decoding."""
from __future__ import annotations

import math
import unittest

import torch

from mrna_editflow.core.constants import NUC_TO_ID, is_valid_cds, translate
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.rl.action_scoring import operation_log_score, softmax_from_log_scores
from mrna_editflow.rl.decoder_state import DecoderAction, DecoderState, choose_stop_aware_action, sequence_hash
from mrna_editflow.sample import _model_score_for_choice, model_guided_edit_record, sample_mrna
from mrna_editflow.train_proposal_ranker import ProposalTeacherRow, _operation_log_score


class _ScoreModel(torch.nn.Module):
    def __init__(self, action_rate: float = 1.0) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.action_rate = float(action_rate)

    def forward(self, token_ids, region_ids, phase_ids, t, padding_mask, backbone):
        length = token_ids.shape[1]
        rates = torch.full((1, length, 3), self.action_rate, device=token_ids.device)
        sub_probs = torch.zeros((1, length, 4), device=token_ids.device)
        sub_probs[..., NUC_TO_ID["G"]] = 1.0
        ins_probs = torch.full((1, length, 4), 0.25, device=token_ids.device)
        return {"rates": rates, "sub_probs": sub_probs, "ins_probs": ins_probs}


class TestStage1DecoderSemantics(unittest.TestCase):
    def setUp(self) -> None:
        self.record = MRNARecord("r", "AAA", "AUGGCCUAA", "CCC")

    def _output(self):
        return {
            "rates": torch.tensor([[[2.0, 3.0, 5.0]]]),
            "sub_probs": torch.tensor([[[0.1, 0.2, 0.3, 0.4]]]),
            "ins_probs": torch.tensor([[[0.4, 0.3, 0.2, 0.1]]]),
        }

    def test_ranker_and_decoder_use_identical_log_score(self):
        out = self._output()
        row = ProposalTeacherRow("r", "sub", 0, "G", 1.0, {})
        expected = operation_log_score(out, "sub", 0, "G")
        self.assertTrue(torch.equal(_operation_log_score(out, row), expected))
        self.assertAlmostEqual(_model_score_for_choice(out, "sub", 0, "G"), float(expected))
        self.assertAlmostEqual(float(expected), math.log(3.0 * 0.3), places=6)

    def test_all_negative_actions_select_stop_without_spending_budget(self):
        out = model_guided_edit_record(
            self.record, _ScoreModel(action_rate=1e-5), object(), task_id="T5",
            edit_budget=3, proposal_temperature=0.0, device="cpu",
        )
        self.assertEqual(out.seq, self.record.seq)
        self.assertTrue(out.metadata["terminated_by_stop"])
        self.assertEqual(out.metadata["applied_edit_count"], 0)
        self.assertEqual(out.metadata["max_edit_budget"], 3)

    def test_reverse_and_visited_actions_are_rejected(self):
        state = DecoderState("AAA", 2)
        first = DecoderAction("sub", 0, "G", 2.0, sequence_hash("GAA"), "A")
        state.accept(first)
        reverse = DecoderAction("sub", 0, "A", 3.0, sequence_hash("AAA"), "G")
        chosen = choose_stop_aware_action([reverse], state, __import__("random").Random(3), top_k=1, temperature=0.0)
        self.assertIsNone(chosen)
        self.assertTrue(state.terminated_by_stop)
        self.assertGreaterEqual(state.cycle_rejections, 1)

    def test_checkpoint_action_space_fails_closed_then_marks_expansion(self):
        model = _ScoreModel(action_rate=3.0)
        model._trained_action_space = {
            "trained_task": "T5",
            "trained_editable_regions": ["utr5"],
            "trained_operations": ["sub", "ins", "del"],
        }
        with self.assertRaises(ValueError):
            model_guided_edit_record(
                self.record, model, object(), task_id="T5", edit_budget=1,
                editable_regions=("utr3",), proposal_temperature=0.0, device="cpu",
            )
        expanded = model_guided_edit_record(
            self.record, model, object(), task_id="T5", edit_budget=1,
            editable_regions=("utr3",), allow_action_space_expansion=True,
            proposal_temperature=0.0, device="cpu",
        )
        self.assertTrue(expanded.metadata["out_of_training_action_space"])

    def test_sample_mrna_routes_model_and_is_reproducible(self):
        model = _ScoreModel(action_rate=3.0)
        first = sample_mrna(
            task_id="T5", record=self.record, model=model, backbone=object(),
            edit_budget=1, proposal_temperature=0.0, seed=17, device="cpu", return_record=True,
        )
        second = sample_mrna(
            task_id="T5", record=self.record, model=model, backbone=object(),
            edit_budget=1, proposal_temperature=0.0, seed=17, device="cpu", return_record=True,
        )
        self.assertEqual(first.seq, second.seq)
        self.assertEqual(first.metadata["decoder_type"], "model_guided")
        self.assertNotEqual(first.seq, self.record.seq)
        self.assertEqual(first.cds, self.record.cds)
        self.assertTrue(is_valid_cds(first.cds))
        self.assertEqual(translate(first.cds), translate(self.record.cds))

    def test_temperature_uses_log_score_softmax(self):
        scores = torch.tensor([math.log(0.2), math.log(0.8)])
        got = softmax_from_log_scores(scores, temperature=1.0)
        expected = torch.tensor([0.2, 0.8], dtype=torch.float64)
        self.assertTrue(torch.allclose(got, expected, atol=1e-12))
        colder = softmax_from_log_scores(scores, temperature=0.5)
        self.assertGreater(float(colder[1]), float(got[1]))


if __name__ == "__main__":
    unittest.main()
