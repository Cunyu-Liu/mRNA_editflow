"""CPU tests for region-specialized Edit-Flow adapters (roadmap upgrade #2).

Run:
    /Users/bytedance/Documents/research/editflow/.venv/bin/python \
        -m unittest mrna_editflow.tests.test_region_adapters -v
"""
from __future__ import annotations

import os
import sys
import unittest

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mrna_editflow.core.config import BackboneConfig, DataConfig, MEFConfig, ModelConfig
from mrna_editflow.core.constants import REGION_CDS, REGION_5UTR
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.models.region_adapters import RegionSpecializedEditFormer
from mrna_editflow.sample import _record_tensors
from mrna_editflow.train_backbone import build_stage_a_model


def _tiny_cfg() -> MEFConfig:
    return MEFConfig(
        data=DataConfig(seed=11, max_5utr=32, max_cds=96, max_3utr=32),
        backbone=BackboneConfig(name="none", hidden_dim=8, freeze=True),
        model=ModelConfig(
            model_dim=16,
            num_layers=1,
            num_heads=4,
            ffn_mult=2,
            dropout=0.0,
            max_seq_len=128,
            use_aux_struct=True,
        ),
    )


def _record() -> MRNARecord:
    # 5'UTR, then in-frame CDS (AUG ... UAA), then 3'UTR.
    return MRNARecord("radapt", "ACGUACGU", "AUGGCCGCauaa".upper(), "GGCCAAUU")


def _forward(model, backbone, record, device):
    token_ids, region_ids, phase_ids, padding_mask = _record_tensors(record, device)
    t = torch.full((1, 1), 0.5, dtype=torch.float32, device=device)
    return model.forward(token_ids, region_ids, phase_ids, t, padding_mask, backbone), region_ids


class TestRegionSpecializedAdapters(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cpu")
        self.backbone, self.base = build_stage_a_model(_tiny_cfg(), device=self.device)
        self.base.eval()
        self.record = _record()

    def test_identity_at_initialization(self):
        wrapper = RegionSpecializedEditFormer(self.base, bottleneck=8)
        wrapper.eval()
        base_out, _ = _forward(self.base, self.backbone, self.record, self.device)
        wrap_out, _ = _forward(wrapper, self.backbone, self.record, self.device)
        for key in ("rates", "ins_probs", "sub_probs"):
            self.assertTrue(
                torch.allclose(base_out[key], wrap_out[key], atol=1e-6),
                f"{key} must match base at init (zero-init adapters)",
            )

    def test_perturbed_adapter_changes_output(self):
        wrapper = RegionSpecializedEditFormer(self.base, bottleneck=8)
        wrapper.eval()
        # Perturb the CDS adapter's zero-initialized up-projection.
        with torch.no_grad():
            up = wrapper.adapters["cds"].up
            up.weight.add_(0.5)
            up.bias.add_(0.1)
        base_out, _ = _forward(self.base, self.backbone, self.record, self.device)
        wrap_out, _ = _forward(wrapper, self.backbone, self.record, self.device)
        self.assertFalse(
            torch.allclose(base_out["rates"], wrap_out["rates"], atol=1e-6),
            "a non-zero CDS adapter must change the output",
        )
        for key in ("rates", "ins_probs", "sub_probs"):
            self.assertTrue(bool(torch.isfinite(wrap_out[key]).all()))

    def test_region_gating_localizes_effect_to_region(self):
        wrapper = RegionSpecializedEditFormer(self.base, bottleneck=8)
        wrapper.eval()
        with torch.no_grad():
            up = wrapper.adapters["cds"].up
            up.weight.add_(0.7)
            up.bias.add_(0.2)
        base_out, region_ids = _forward(self.base, self.backbone, self.record, self.device)
        wrap_out, _ = _forward(wrapper, self.backbone, self.record, self.device)
        region_ids = region_ids[0]
        diff = (wrap_out["rates"][0] - base_out["rates"][0]).detach().abs().sum(dim=-1)  # [L]
        cds_mask = region_ids == REGION_CDS
        utr5_mask = region_ids == REGION_5UTR
        # Only CDS positions should move; non-CDS residual stays identity.
        self.assertGreater(float(diff[cds_mask].sum()), 1e-4)
        self.assertLess(float(diff[utr5_mask].abs().sum()), 1e-6)

    def test_freeze_base_only_trains_adapters(self):
        wrapper = RegionSpecializedEditFormer(self.base, bottleneck=8, freeze_base=True)
        base_trainable = sum(p.numel() for p in wrapper.base.parameters() if p.requires_grad)
        self.assertEqual(base_trainable, 0)
        adapter_trainable = wrapper.num_trainable_params()
        self.assertEqual(adapter_trainable, wrapper.num_adapter_params())
        self.assertGreater(adapter_trainable, 0)
        # Three regions each add 2*D*b + D + b parameters.
        d, b = wrapper.dim, wrapper.bottleneck
        expected_per_region = 2 * d * b + d + b
        self.assertEqual(adapter_trainable, 3 * expected_per_region)

    def test_base_state_dict_is_unchanged_and_loadable(self):
        # Wrapping must not rename/remove base params: the base state dict from a
        # Stage A checkpoint must still load into a fresh base model.
        wrapper = RegionSpecializedEditFormer(self.base, bottleneck=8)
        _fresh_backbone, fresh_base = build_stage_a_model(_tiny_cfg(), device=self.device)
        missing, unexpected = fresh_base.load_state_dict(wrapper.base.state_dict(), strict=True)
        self.assertEqual(list(missing), [])
        self.assertEqual(list(unexpected), [])

    def test_adapter_gradients_flow_only_to_adapters(self):
        wrapper = RegionSpecializedEditFormer(self.base, bottleneck=8, freeze_base=True)
        wrapper.train()
        out, _ = _forward(wrapper, self.backbone, self.record, self.device)
        loss = out["rates"].sum() + out["sub_probs"].sum()
        loss.backward()
        base_grads = [p.grad for p in wrapper.base.parameters() if p.grad is not None]
        adapter_grads = [p.grad for p in wrapper.adapters.parameters() if p.grad is not None]
        self.assertEqual(base_grads, [])
        self.assertTrue(len(adapter_grads) > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
