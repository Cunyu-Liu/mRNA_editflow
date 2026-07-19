"""End-to-end shape / finiteness tests for the coupling + forward + loss path.

Run: ``.../python -m unittest mrna_editflow.tests.test_flow_shapes``.
"""
from __future__ import annotations

import os
import sys
import unittest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from mrna_editflow.core.config import BackboneConfig, CouplingConfig, ModelConfig
from mrna_editflow.core.constants import VOCAB_MODEL_SIZE
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.core import mrna_flow_utils as U
from mrna_editflow.models.backbones import FrozenBackbone
from mrna_editflow.models.mrna_editformer import MRNAEditFormer


def _tiny_records():
    return [
        MRNARecord("t1", "ACGU", "AUGGCUGCCUAA", "GGCC"),
        MRNARecord("t2", "AC", "AUGCCCGGGUAA", "GGGG"),
        MRNARecord("t3", "ACGUAC", "AUGAAAUAA", "GG"),
    ]


class TestFlowShapes(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cpu")
        self.bb = FrozenBackbone(BackboneConfig(name="none", hidden_dim=16))
        self.cfg = ModelConfig(
            model_dim=32,
            num_layers=2,
            num_heads=4,
            max_seq_len=128,
            use_aux_struct=True,
        )
        self.head = MRNAEditFormer(self.cfg, backbone_dim=self.bb.out_dim)

    def test_hybrid_batch_shapes_consistent(self):
        hb = U.make_hybrid_batch(_tiny_records(), CouplingConfig(), self.device, seed=1)
        B = len(_tiny_records())
        self.assertEqual(hb.z0.shape, hb.z1.shape)
        self.assertEqual(hb.z1.shape, hb.region_ids.shape)
        self.assertEqual(hb.z1.shape, hb.phase_ids.shape)
        self.assertEqual(hb.t.shape, (B, 1))
        self.assertEqual(hb.x1.shape[0], B)
        self.assertEqual(len(hb.route), B)

    def test_forward_shapes_and_finiteness(self):
        hb = U.make_hybrid_batch(_tiny_records(), CouplingConfig(), self.device, seed=2)
        zt = U.sample_cond_pt(U.x2prob(hb.z0), U.x2prob(hb.z1), hb.t, U.CubicScheduler())
        xt, x_pad, z_gap, z_pad, region_x, phase_x = U.rm_gap_tokens_with_aux(
            zt, hb.region_ids, hb.phase_ids
        )
        out = self.head.forward(xt, region_x, phase_x, hb.t, x_pad, self.bb)
        B, L = xt.shape
        self.assertEqual(tuple(out["rates"].shape), (B, L, 3))
        self.assertEqual(tuple(out["ins_probs"].shape), (B, L, VOCAB_MODEL_SIZE))
        self.assertEqual(tuple(out["sub_probs"].shape), (B, L, VOCAB_MODEL_SIZE))
        self.assertEqual(tuple(out["aux"].shape), (B, L, 2))
        for k in ("rates", "ins_probs", "sub_probs", "aux"):
            self.assertTrue(torch.isfinite(out[k]).all(), f"{k} not finite")
        self.assertTrue((out["rates"] >= 0).all())

    def test_full_training_loss_is_finite_and_backprops(self):
        """Reproduce the reference Edit-Flow loss on the mRNA pipeline."""
        scheduler = U.CubicScheduler(a=1.0, b=1.0)
        hb = U.make_hybrid_batch(_tiny_records(), CouplingConfig(), self.device, seed=3)
        zt = U.sample_cond_pt(U.x2prob(hb.z0), U.x2prob(hb.z1), hb.t, scheduler)
        xt, x_pad, z_gap, z_pad, region_x, phase_x = U.rm_gap_tokens_with_aux(
            zt, hb.region_ids, hb.phase_ids
        )
        out = self.head.forward(xt, region_x, phase_x, hb.t, x_pad, self.bb)

        # Shared, gradient-safe CTMC loss helper (numerical guards inside).
        losses = U.edit_flow_loss(
            out, zt, hb.z1, x_pad, z_gap, z_pad, hb.t, scheduler,
            vocab_size=VOCAB_MODEL_SIZE,
        )
        loss = losses["loss"]
        for k in ("loss", "loss_ins", "loss_sub", "loss_del"):
            self.assertTrue(torch.isfinite(losses[k]), f"{k} not finite")

        loss.backward()
        grad_norm = sum(
            float(p.grad.abs().sum())
            for p in self.head.parameters()
            if p.grad is not None
        )
        self.assertEqual(grad_norm, grad_norm)  # not NaN
        self.assertGreater(grad_norm, 0.0)

    def test_loss_clips_near_terminal_time_in_half_precision(self):
        scheduler = U.CubicScheduler(a=1.0, b=1.0)
        hb = U.make_hybrid_batch(_tiny_records(), CouplingConfig(), self.device, seed=33)
        hb.t.fill_(0.999999)
        zt = U.sample_cond_pt(U.x2prob(hb.z0), U.x2prob(hb.z1), hb.t, scheduler)
        xt, x_pad, z_gap, z_pad, _region_x, _phase_x = U.rm_gap_tokens_with_aux(
            zt, hb.region_ids, hb.phase_ids
        )
        B, L = xt.shape
        rates = torch.ones((B, L, 3), dtype=torch.float16, requires_grad=True)
        probs = torch.full(
            (B, L, VOCAB_MODEL_SIZE),
            1.0 / VOCAB_MODEL_SIZE,
            dtype=torch.float16,
        )
        out = {"rates": rates, "ins_probs": probs, "sub_probs": probs, "aux": None}
        losses = U.edit_flow_loss(
            out, zt, hb.z1, x_pad, z_gap, z_pad, hb.t, scheduler,
            vocab_size=VOCAB_MODEL_SIZE,
        )
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        self.assertIsNotNone(rates.grad)
        self.assertTrue(torch.isfinite(rates.grad).all())

    def test_padded_positions_are_zeroed(self):
        hb = U.make_hybrid_batch(_tiny_records(), CouplingConfig(), self.device, seed=4)
        zt = U.sample_cond_pt(U.x2prob(hb.z0), U.x2prob(hb.z1), hb.t, U.CubicScheduler())
        xt, x_pad, z_gap, z_pad, region_x, phase_x = U.rm_gap_tokens_with_aux(
            zt, hb.region_ids, hb.phase_ids
        )
        out = self.head.forward(xt, region_x, phase_x, hb.t, x_pad, self.bb)
        if x_pad.any():
            self.assertAlmostEqual(float(out["rates"][x_pad].abs().sum().detach()), 0.0, places=6)
            self.assertAlmostEqual(float(out["aux"][x_pad].abs().sum().detach()), 0.0, places=6)

    def test_no_film_no_rope_variant_runs(self):
        cfg = ModelConfig(
            model_dim=32, num_layers=2, num_heads=4, max_seq_len=128,
            use_rope=False, use_region_film=False, use_aux_struct=False,
        )
        head = MRNAEditFormer(cfg, backbone_dim=self.bb.out_dim)
        hb = U.make_hybrid_batch(_tiny_records(), CouplingConfig(), self.device, seed=5)
        zt = U.sample_cond_pt(U.x2prob(hb.z0), U.x2prob(hb.z1), hb.t, U.CubicScheduler())
        xt, x_pad, z_gap, z_pad, region_x, phase_x = U.rm_gap_tokens_with_aux(
            zt, hb.region_ids, hb.phase_ids
        )
        out = head.forward(xt, region_x, phase_x, hb.t, x_pad, self.bb)
        self.assertIsNone(out["aux"])
        self.assertTrue(torch.isfinite(out["rates"]).all())


if __name__ == "__main__":
    unittest.main()
