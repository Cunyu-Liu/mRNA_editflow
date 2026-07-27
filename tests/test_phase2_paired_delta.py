import os

import pytest
import torch

from mrna_editflow.models.paired_delta_former import PairedDeltaFormer
from mrna_editflow.train.train_paired_delta import paired_delta_loss


def _inputs(batch=3, length=12, edits=2):
    source = torch.randint(0, 4, (batch, length))
    candidate = source.clone()
    candidate[:, 2] = (candidate[:, 2] + 1) % 4
    return {
        "source_tokens": source,
        "candidate_tokens": candidate,
        "source_mask": torch.ones(batch, length, dtype=torch.bool),
        "candidate_mask": torch.ones(batch, length, dtype=torch.bool),
        "edit_tokens": torch.tensor([
            [[0, 2, 0, 1, -3], [-1, -1, -1, -1, 0]],
            [[0, 2, 1, 2, -3], [-1, -1, -1, -1, 0]],
            [[0, 2, 2, 3, -3], [-1, -1, -1, -1, 0]],
        ]),
        "context_ids": torch.zeros(batch, 3, dtype=torch.long),
        "protein_embedding": torch.zeros(batch, 16),
        "cell_embedding": torch.zeros(batch, 16),
        "assay_embedding": torch.zeros(batch, 16),
        "source_value": torch.zeros(batch),
    }


def test_all_backbone_contracts_and_outputs():
    for backbone in ("small", "frozen_foundation", "partial_foundation"):
        model = PairedDeltaFormer(
            hidden_dim=32, layers=1, max_len=16, backbone=backbone,
            allow_foundation_stub=True,
        )
        output = model(**_inputs())
        assert output["mean"].shape == (3,)
        assert output["variance"].shape == (3,)
        assert output["beneficial_probability"].shape == (3,)
        assert output["rank"].shape == (3,)


def test_frozen_backbone_has_no_encoder_gradients():
    model = PairedDeltaFormer(
        hidden_dim=32, layers=1, max_len=16, backbone="frozen_foundation",
        allow_foundation_stub=True,
    )
    out = model(**_inputs())
    loss, _ = paired_delta_loss(out, torch.tensor([0.1, -0.2, 0.3]))
    loss.backward()
    encoder_parameters = list(model.sequence_encoder.parameters())
    assert encoder_parameters
    assert all(not p.requires_grad for p in encoder_parameters)
    assert all(p.grad is None for p in encoder_parameters)


def test_multitask_loss_propagates_to_all_heads():
    model = PairedDeltaFormer(hidden_dim=32, layers=1, max_len=16, backbone="small")
    out = model(**_inputs())
    loss, parts = paired_delta_loss(out, torch.tensor([0.1, -0.2, 0.3]))
    assert set(("huber", "ranking", "beneficial", "nll", "calibration")) <= set(parts)
    loss.backward()
    assert model.head.mean.weight.grad is not None
    assert model.head.logvar.weight.grad is not None
    assert model.head.beneficial.weight.grad is not None
    assert model.head.rank.weight.grad is not None


def test_stage_a_checkpoint_adapter_when_explicitly_provided():
    path = os.environ.get("PHASE2_STAGE_A_CHECKPOINT")
    if not path:
        pytest.skip("set PHASE2_STAGE_A_CHECKPOINT to run the real internal mRNA-pretrained adapter test")
    model = PairedDeltaFormer(
        hidden_dim=32, layers=1, max_len=32, backbone="frozen_foundation",
        foundation_path=path, allow_foundation_stub=False,
    )
    assert model.is_real_foundation
    assert model.foundation_kind == "internal_stage_a_mrna_pretrained"
    assert model.backbone_status == "real_internal_stage_a_mrna_pretrained"
    assert sum(parameter.requires_grad for parameter in model.sequence_encoder.parameters()) == 0
    output = model(**_inputs(length=16))
    assert torch.isfinite(output["mean"]).all()
