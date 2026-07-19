"""Freeze / trainable-parameter accounting tests.

Run: ``.../python -m unittest mrna_editflow.tests.test_backbone_freeze``.
"""
from __future__ import annotations

import os
import sys
import unittest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from mrna_editflow.core.config import BackboneConfig, ModelConfig
from mrna_editflow.core.constants import NUC_TO_ID, REGION_CDS, REGION_5UTR, VOCAB_MODEL_SIZE
from mrna_editflow.models.backbones import FrozenBackbone
from mrna_editflow.models.mrna_editformer import MRNAEditFormer
from mrna_editflow.models.adapters import AdapterWrappedMEF


def _tiny_inputs(B=2, L=9):
    tok = torch.randint(0, 4, (B, L))
    region = torch.full((B, L), REGION_5UTR)  # UTR: nothing masked, clean grads
    phase = torch.tensor([[i % 3 for i in range(L)]] * B)
    t = torch.rand(B, 1)
    pad = torch.zeros(B, L, dtype=torch.bool)
    return tok, region, phase, t, pad


class TestBackboneFreeze(unittest.TestCase):
    def test_frozen_backbone_params_have_no_requires_grad(self):
        bb = FrozenBackbone(BackboneConfig(name="none", hidden_dim=16, freeze=True))
        self.assertTrue(bb.frozen)
        self.assertTrue(all(not p.requires_grad for p in bb.parameters()))
        self.assertGreater(sum(p.numel() for p in bb.parameters()), 0)

    def test_frozen_backbone_receives_no_grad_after_backward(self):
        bb = FrozenBackbone(BackboneConfig(name="none", hidden_dim=16, freeze=True))
        head = MRNAEditFormer(
            ModelConfig(model_dim=32, num_layers=2, num_heads=4, max_seq_len=64),
            backbone_dim=bb.out_dim,
        )
        tok, region, phase, t, pad = _tiny_inputs()
        out = head.forward(tok, region, phase, t, pad, bb)
        loss = out["rates"].sum() + out["ins_probs"].sum() + out["sub_probs"].sum()
        loss.backward()
        # Backbone params: no grad populated.
        for p in bb.parameters():
            self.assertIsNone(p.grad)
        # Head params: at least some grads populated (learning signal exists).
        grad_norm = sum(
            float(p.grad.abs().sum()) for p in head.parameters() if p.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)

    def test_unfrozen_backbone_can_receive_grad(self):
        bb = FrozenBackbone(BackboneConfig(name="none", hidden_dim=16, freeze=False))
        self.assertFalse(bb.frozen)
        head = MRNAEditFormer(
            ModelConfig(model_dim=32, num_layers=1, num_heads=4, max_seq_len=64),
            backbone_dim=bb.out_dim,
        )
        tok, region, phase, t, pad = _tiny_inputs()
        out = head.forward(tok, region, phase, t, pad, bb)
        out["rates"].sum().backward()
        got = any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in bb.parameters())
        self.assertTrue(got)

    def test_external_backbone_falls_back_to_stub_offline(self):
        for name in ("mrnabert", "orthrus", "lamar", "rna_fm", "rinalmo"):
            bb = FrozenBackbone(BackboneConfig(name=name, hidden_dim=16, freeze=True))
            self.assertFalse(bb.is_real)  # adapter-stub offline
            tok, region, phase, t, pad = _tiny_inputs()
            emb = bb.embed(tok, region, pad)
            self.assertEqual(tuple(emb.shape), (2, 9, 16))
            self.assertTrue(torch.isfinite(emb).all())

    def test_stub_backbone_is_deterministic_across_instances(self):
        cfg = BackboneConfig(name="orthrus", hidden_dim=16, freeze=True)
        bb1, bb2 = FrozenBackbone(cfg), FrozenBackbone(cfg)
        tok, region, phase, t, pad = _tiny_inputs()
        e1, e2 = bb1.embed(tok, region, pad), bb2.embed(tok, region, pad)
        self.assertTrue(torch.allclose(e1, e2))


class TestAdapterTrainableAccounting(unittest.TestCase):
    def _build_adapter(self, rank):
        bb = FrozenBackbone(BackboneConfig(name="none", hidden_dim=16, freeze=True))
        base = MRNAEditFormer(
            ModelConfig(model_dim=32, num_layers=3, num_heads=4, max_seq_len=64),
            backbone_dim=bb.out_dim,
        )
        wrap = AdapterWrappedMEF(bb, base, adapter_rank=rank, num_property_buckets=8)
        return bb, base, wrap

    def test_only_adapters_and_task_head_are_trainable(self):
        bb, base, wrap = self._build_adapter(rank=4)
        # Frozen: backbone + base head.
        self.assertTrue(all(not p.requires_grad for p in bb.parameters()))
        self.assertTrue(all(not p.requires_grad for p in base.parameters()))

        # Trainable set == adapters + task-head components exactly.
        trainable_ids = {id(p) for p in wrap.parameters() if p.requires_grad}
        expected = []
        for m in (
            wrap.adapters, wrap.property_emb, wrap.property_film,
            wrap.task_norm, wrap.rates_out, wrap.ins_logits, wrap.sub_logits,
        ):
            expected += list(m.parameters())
        if wrap.use_aux:
            expected += list(wrap.aux_struct.parameters())
        expected_ids = {id(p) for p in expected}
        self.assertEqual(trainable_ids, expected_ids)
        self.assertEqual(wrap.num_trainable_params(), sum(p.numel() for p in expected))

    def test_backward_grads_only_adapters_and_head(self):
        bb, base, wrap = self._build_adapter(rank=16)
        tok, region, phase, t, pad = _tiny_inputs()
        out = wrap.forward(tok, region, phase, t, pad, torch.tensor([0, 1]))
        loss = out["rates"].sum() + out["sub_probs"].pow(2).sum() + out["ins_probs"].sum()
        loss.backward()
        # Frozen never receives grad.
        for p in list(bb.parameters()) + list(base.parameters()):
            self.assertIsNone(p.grad)
        # Every adapter up-projection receives gradient (grad flows through the
        # whole frozen stack, not just the final block).
        for adapter in wrap.adapters:
            self.assertIsNotNone(adapter.up.weight.grad)
            self.assertGreater(float(adapter.up.weight.grad.abs().sum()), 0.0)

    def test_adapter_rank_controls_param_count(self):
        _, _, wrap4 = self._build_adapter(rank=4)
        _, _, wrap16 = self._build_adapter(rank=16)
        self.assertLess(wrap4.num_trainable_params(), wrap16.num_trainable_params())
        # Adapters are a small fraction of the frozen base.
        self.assertLess(wrap4.num_trainable_params(), wrap4.num_frozen_params())

    def test_adapter_half_precision_constraint_mask_is_finite(self):
        _, _, wrap = self._build_adapter(rank=4)
        cds = "AUGGCUUAA"
        tok = torch.tensor([[NUC_TO_ID[c] for c in cds]], dtype=torch.long)
        region = torch.full((1, len(cds)), REGION_CDS, dtype=torch.long)
        phase = torch.tensor([[i % 3 for i in range(len(cds))]], dtype=torch.long)
        rates = torch.ones((1, len(cds), 3), dtype=torch.float16)
        sub_logits = torch.zeros((1, len(cds), VOCAB_MODEL_SIZE), dtype=torch.float16)
        masked_rates, masked_logits = wrap._apply_codon_constraints(
            rates, sub_logits, tok, region, phase
        )
        self.assertEqual(masked_rates.dtype, torch.float16)
        self.assertEqual(masked_logits.dtype, torch.float16)
        self.assertTrue(bool(torch.isfinite(masked_rates).all()))
        self.assertTrue(bool(torch.isfinite(masked_logits).all()))
        self.assertLess(float(masked_logits.min()), -1000.0)


if __name__ == "__main__":
    unittest.main()
