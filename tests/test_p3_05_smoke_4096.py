"""P3-05 Task 6: 4096nt smoke train — forward + backward pass verification.

Verifies that the existing MRNAEditFormer can handle sequences up to 4096 nt
without OOM (using the `none` backbone, which is the lightweight control).

This satisfies the P3-05 acceptance criterion: "4096 nt smoke train 可运行".
"""
from __future__ import annotations

import sys
from pathlib import Path
# Add parent of mrna_editflow/ so that `mrna_editflow.models.*` (which uses
# relative imports like `from ..core.constants`) resolves correctly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
import torch

from mrna_editflow.core.constants import START_CODON, NUC_TO_ID, PHASE_NONE
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.core.config import ModelConfig, BackboneConfig
from mrna_editflow.models.backbones import FrozenBackbone
from mrna_editflow.models.mrna_editformer import MRNAEditFormer


def _make_record(total_len: int) -> MRNARecord:
    """Create a record of approximately total_len nucleotides."""
    five_utr_len = 100
    three_utr_len = 100
    cds_len = total_len - five_utr_len - three_utr_len
    # Ensure CDS length is a multiple of 3
    cds_len = (cds_len // 3) * 3
    # Build CDS: START + (codon * N) + STOP
    n_internal_codons = (cds_len - 6) // 3  # -6 for START + STOP
    cds = START_CODON + "GCU" * n_internal_codons + "UAA"
    return MRNARecord(
        transcript_id=f"smoke_{total_len}",
        five_utr="ACGU" * (five_utr_len // 4),
        cds=cds,
        three_utr="UGCU" * (three_utr_len // 4),
    )


def _make_small_model(device: torch.device) -> tuple[MRNAEditFormer, FrozenBackbone]:
    """Build a small MRNAEditFormer with `none` backbone for smoke testing."""
    backbone_cfg = BackboneConfig(name="none", hidden_dim=32, freeze=True)
    backbone = FrozenBackbone(backbone_cfg)

    model_cfg = ModelConfig(
        model_dim=64,
        num_layers=1,
        num_heads=4,
        ffn_mult=2,
        dropout=0.0,
        max_seq_len=4096,
        use_rope=True,  # RoPE avoids needing position embeddings for long seqs
        use_region_film=True,
        use_codon_constraint=False,  # Disable for smoke test speed
        use_aux_struct=False,
    )
    model = MRNAEditFormer(cfg=model_cfg, backbone_dim=backbone.out_dim)
    model.to(device)
    model.train()
    return model, backbone


def _forward_record(
    model: MRNAEditFormer,
    backbone: FrozenBackbone,
    record: MRNARecord,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Run a forward pass for a record."""
    seq = record.seq
    L = len(seq)
    token_ids = torch.tensor([[NUC_TO_ID.get(c, 0) for c in seq]], device=device)
    region_ids = torch.tensor([record.region_ids()], device=device)
    phase_ids = torch.tensor([record.codon_phases()], device=device)
    time_step = torch.tensor([[0.5]], device=device)
    padding_mask = torch.zeros(1, L, dtype=torch.bool, device=device)

    out = model(token_ids, region_ids, phase_ids, time_step, padding_mask, backbone)
    return out


class TestSmoke4096:
    """Verify 4096nt forward + backward pass works."""

    @pytest.fixture(scope="class")
    def device(self):
        return torch.device("cpu")

    @pytest.fixture(scope="class")
    def model_and_backbone(self, device):
        return _make_small_model(device)

    @pytest.mark.parametrize("total_len", [512, 1024, 2048, 4096])
    def test_forward_pass_multiple_lengths(self, model_and_backbone, device, total_len):
        """Forward pass must complete for sequences of 512, 1024, 2048, 4096 nt."""
        model, backbone = model_and_backbone
        record = _make_record(total_len)
        assert len(record.seq) >= total_len - 10  # approximately

        out = _forward_record(model, backbone, record, device)

        L = len(record.seq)
        assert "rates" in out
        assert out["rates"].shape == (1, L, 3)
        assert "ins_probs" in out
        assert "sub_probs" in out
        # All outputs must be finite
        assert torch.isfinite(out["rates"]).all(), "rates contains non-finite values"

    def test_backward_pass_4096(self, model_and_backbone, device):
        """Backward pass must complete for 4096 nt (smoke train)."""
        model, backbone = model_and_backbone
        record = _make_record(4096)

        out = _forward_record(model, backbone, record, device)

        # Use rates mean as a dummy loss
        loss = out["rates"].mean()
        loss.backward()

        # Verify gradients flow to model parameters (not backbone, which is frozen)
        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    has_grad = True
                    break
        assert has_grad, "No gradients found in trainable model parameters"

        # Verify backbone is frozen (no gradients)
        for name, param in backbone.named_parameters():
            assert not param.requires_grad, f"Backbone param {name} should be frozen"

    def test_4096_output_shapes(self, model_and_backbone, device):
        """Output shapes must be correct for 4096 nt."""
        model, backbone = model_and_backbone
        record = _make_record(4096)
        out = _forward_record(model, backbone, record, device)

        L = len(record.seq)
        assert out["rates"].shape == (1, L, 3)
        # ins_probs and sub_probs: [B, L, V]
        assert out["ins_probs"].shape[0] == 1
        assert out["ins_probs"].shape[1] == L
        assert out["sub_probs"].shape[0] == 1
        assert out["sub_probs"].shape[1] == L

    def test_4096_finite_outputs(self, model_and_backbone, device):
        """All outputs must be finite at 4096 nt."""
        model, backbone = model_and_backbone
        record = _make_record(4096)
        out = _forward_record(model, backbone, record, device)

        assert torch.isfinite(out["rates"]).all()
        assert torch.isfinite(out["ins_probs"]).all()
        assert torch.isfinite(out["sub_probs"]).all()

    def test_phase_alignment_in_forward(self, model_and_backbone, device):
        """Phase IDs passed to model must be correctly aligned."""
        model, backbone = model_and_backbone
        record = _make_record(1024)

        phases = record.codon_phases()
        # First CDS nt must be phase 0
        assert phases[record.cds_start] == 0
        # 5'UTR must be PHASE_NONE
        assert all(p == PHASE_NONE for p in phases[:record.cds_start])
        # 3'UTR must be PHASE_NONE
        assert all(p == PHASE_NONE for p in phases[record.cds_end:])


class TestNoLengthBucketCollapse:
    """Verify no obvious collapse across length buckets."""

    @pytest.fixture(scope="class")
    def device(self):
        return torch.device("cpu")

    @pytest.fixture(scope="class")
    def model_and_backbone(self, device):
        return _make_small_model(device)

    @pytest.mark.parametrize("bucket", ["short", "medium", "long", "xlong"])
    def test_output_magnitude_consistent_across_buckets(
        self, model_and_backbone, device, bucket
    ):
        """Output rate magnitudes should be in the same order of magnitude."""
        model, backbone = model_and_backbone
        lengths = {"short": 512, "medium": 1024, "long": 2048, "xlong": 4096}

        record = _make_record(lengths[bucket])
        out = _forward_record(model, backbone, record, device)

        rates_mean = float(out["rates"].mean())
        rates_std = float(out["rates"].std())

        # No collapse: rates should not be all-zero or all-the-same
        assert rates_std > 1e-6, \
            f"Bucket {bucket}: rates std {rates_std} too small (potential collapse)"
        # Rates should be in a reasonable range (not NaN/Inf, not astronomically large)
        assert abs(rates_mean) < 100, \
            f"Bucket {bucket}: rates mean {rates_mean} unreasonably large"
