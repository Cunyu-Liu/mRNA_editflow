import json
import tempfile
import unittest
from pathlib import Path

import torch

from mrna_editflow.core.mixed_resolution_state import MixedAction, MixedResolutionState, apply_action, legal_actions
from mrna_editflow.models.mixed_resolution_editformer import MixedResolutionEditFormer
from mrna_editflow.rl.grpo_v2 import grpo_loss


class Phase15ContractsTest(unittest.TestCase):
    def test_atomic_synonymous_swap_preserves_protein(self):
        state = MixedResolutionState("ACGU", "AUGGCUUAA")
        actions = legal_actions(state, include_three_utr=False)
        swaps = [a for a in actions if a.kind == "CDS_SYN_SWAP"]
        self.assertTrue(swaps)
        out = apply_action(state, swaps[0])
        self.assertEqual(state.protein, out.protein)
        self.assertEqual(len(state.cds), len(out.cds))

    def test_stop_is_in_same_normalized_distribution(self):
        state = MixedResolutionState("ACGU", "AUGGCUUAA")
        model = MixedResolutionEditFormer(hidden_dim=32)
        logp, actions = model.log_probs(state)
        self.assertIn(MixedAction.stop(), actions)
        self.assertAlmostEqual(float(logp.exp().sum()), 1.0, places=5)

    def test_grpo_regularizers_have_gradient(self):
        logp = torch.tensor([-.4, -.8, -.2], requires_grad=True)
        old = logp.detach(); ref = torch.tensor([-.5, -.7, -.3])
        out = grpo_loss(logp, old, ref, torch.tensor([1.0, 0.0, 0.5]), torch.tensor([0, 0, 0]))
        out["loss"].backward()
        self.assertIsNotNone(logp.grad)
        self.assertGreater(float(logp.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
