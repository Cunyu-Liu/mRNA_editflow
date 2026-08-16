from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).with_name("run_route2_utrlm_baseline_v1.py")
if not MODULE_PATH.is_file():
    MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/route_a_v3/run_route2_utrlm_baseline_v1.py"
SPEC = importlib.util.spec_from_file_location("run_route2_utrlm_baseline_v1", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

REMOTE_ASSET_ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_models/utrlm")
REMOTE_CHECKPOINT = REMOTE_ASSET_ROOT / "Model/Pretrained" / MODULE.OFFICIAL_CHECKPOINT_NAME


def _cuda_device() -> torch.device:
    index = int(os.environ.get("ROUTE2_TEST_CUDA_INDEX", "0"))
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        pytest.skip(f"physical CUDA device {index} is unavailable")
    return torch.device(f"cuda:{index}")


@pytest.mark.skipif(not REMOTE_CHECKPOINT.is_file(), reason="official UTR-LM asset is remote-only")
def test_official_encoder_strict_load_and_cuda_bos_embedding() -> None:
    device = _cuda_device()
    model, alphabet = MODULE.load_official_encoder(REMOTE_ASSET_ROOT, REMOTE_CHECKPOINT, device)
    embeddings = MODULE.encode_bos_embeddings(
        model,
        alphabet,
        ["ACGT" * 12 + "AC", "TGCA" * 12 + "TG"],
        device,
        batch_size=2,
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_208_559
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert set(embeddings) == {"ACGT" * 12 + "AC", "TGCA" * 12 + "TG"}
    assert all(value.shape == (128,) for value in embeddings.values())
    assert all(value.is_cuda and value.device == device for value in embeddings.values())
    assert all(torch.isfinite(value).all().item() for value in embeddings.values())


def test_probe_trains_only_cuda_linear_head_and_predicts_every_split() -> None:
    device = _cuda_device()
    generator = torch.Generator(device=device).manual_seed(20260816)
    sequences = [f"sequence-{index}" for index in range(12)]
    embeddings = {
        sequence: torch.randn(128, generator=generator, device=device)
        for sequence in sequences
    }
    records = [
        MODULE.TaskRecord("train-1", "source-a", sequences[0], sequences[1], 0.8, "TRAIN"),
        MODULE.TaskRecord("train-2", "source-a", sequences[0], sequences[2], -0.3, "TRAIN"),
        MODULE.TaskRecord("train-3", "source-b", sequences[3], sequences[4], 1.1, "TRAIN"),
        MODULE.TaskRecord("train-4", "source-c", sequences[5], sequences[6], -0.7, "TRAIN"),
        MODULE.TaskRecord("validation-1", "source-d", sequences[7], sequences[8], 0.4, "VALIDATION"),
        MODULE.TaskRecord("validation-2", "source-e", sequences[9], sequences[10], -0.6, "VALIDATION"),
        MODULE.TaskRecord("test-1", "source-f", sequences[10], sequences[11], 0.2, "TEST"),
    ]
    progress_rows = []
    predictions, provenance, artifact = MODULE.train_probe(
        records,
        embeddings,
        device,
        seed=20260816,
        epochs=12,
        learning_rate=1e-2,
        weight_decay=1e-4,
        progress=progress_rows.append,
    )
    assert set(predictions) == {record.record_id for record in records}
    assert provenance["probe_optimizer_steps"] == 12
    assert provenance["probe_parameter_changed"] is True
    assert provenance["probe_parameter_count"] == 129
    assert artifact["feature_mean"].device.type == "cpu"
    assert artifact["feature_std"].device.type == "cpu"
    assert set(artifact["probe_state"]) == {"weight", "bias"}
    assert [row["epoch"] for row in progress_rows] == [1, 10]
    assert all(row["event"] == "PROBE_EPOCH_COMPLETED" for row in progress_rows)
    frozen_predictions, frozen_provenance, _ = MODULE.train_probe(
        records,
        embeddings,
        device,
        seed=20260816,
        epochs=12,
        learning_rate=1e-2,
        weight_decay=1e-4,
        result_stage="FROZEN_DEVELOPMENT_TEST",
    )
    assert set(frozen_predictions) == {record.record_id for record in records}
    assert frozen_provenance["development_validation_folded_into_probe_training"] is True
    assert frozen_provenance["best_validation_source_group_weighted_mse"] is None


def test_source_group_weights_give_each_source_equal_total_weight() -> None:
    device = _cuda_device()
    records = [
        MODULE.TaskRecord("a-1", "a", "s0", "s1", 0.0, "TRAIN"),
        MODULE.TaskRecord("a-2", "a", "s0", "s2", 0.0, "TRAIN"),
        MODULE.TaskRecord("b-1", "b", "s3", "s4", 0.0, "TRAIN"),
    ]
    weights = MODULE._source_group_weights(records, device)
    assert weights.is_cuda and weights.device == device
    assert torch.isclose(weights[:2].sum(), weights[2]).item()
    assert torch.isclose(weights.mean(), torch.tensor(1.0, device=device)).item()


def test_result_stage_withholds_test_before_probe_training() -> None:
    assert MODULE.splits_for_result_stage("HPO_VALIDATION_ONLY") == ("TRAIN", "VALIDATION")
    assert MODULE.splits_for_result_stage("FROZEN_DEVELOPMENT_TEST") == MODULE.SPLITS
    with pytest.raises(MODULE.UtrLmBaselineError, match="invalid result_stage"):
        MODULE.splits_for_result_stage("")
