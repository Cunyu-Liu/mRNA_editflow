from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/run_route2_aparent_baseline_v1.py"
OFFICIAL_WEIGHT = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_models/aparent/"
    "saved_models/aparent_large_lessdropout_all_libs_no_sampleweights.h5"
)
PSMC6_PROXIMAL = (
    "AGATAGTGGTATAAGAAAGCATTTCTTATGACTTATTTTGTATCATTTGTTTTCCTCATCTAAAAAGTTG"
    "AATAAAATCTGTTTGATTCAGTTCTCCTACATATATATTCTTGTCTTTTCTGAGTATATTTACTGTGGTCC"
    "TTTAGGTTCTTTAGCAAGTAAACTATTTGATAACCCAGATGGATTGTGGATTTTTGAATATTAT"
)


def _module():
    spec = importlib.util.spec_from_file_location("run_route2_aparent_baseline_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("APARENT parity validation requires CUDA")
    return torch.device(f"cuda:{int(os.environ.get('ROUTE2_TEST_CUDA_INDEX', '0'))}")


def test_official_encoder_right_pads_164nt_with_zero_vectors() -> None:
    module = _module()
    device = _device()
    encoded = module.one_hot(["A" * 164], device)
    assert tuple(encoded.shape) == (1, 1, 205, 4)
    assert torch.all(encoded[0, 0, :164, 0] == 1)
    assert torch.all(encoded[0, 0, 164:, :] == 0)


def test_pytorch_port_matches_official_psmc6_notebook_cut_logit() -> None:
    module = _module()
    if not OFFICIAL_WEIGHT.is_file():
        pytest.skip("official APARENT weight is not materialized")
    device = _device()
    model = module.AparentBase(OFFICIAL_WEIGHT).to(device).eval()
    with torch.no_grad():
        _isoform, cut = model(module.one_hot([PSMC6_PROXIMAL], device))
        probability = cut[:, 80:105].sum(dim=1)
        natural_logit = torch.log(probability / (1.0 - probability))
    # The repository notebook was produced by legacy TensorFlow/Keras; the CUDA
    # PyTorch port stays within 3e-4 natural-logit units on the published anchor.
    assert float(natural_logit.cpu()) == pytest.approx(2.0683621599430855, abs=3e-4)
    assert math.isfinite(float(module.proximal_log2_odds(cut, 80, 105).cpu()))


def test_one_hot_refuses_cpu_execution() -> None:
    module = _module()
    with pytest.raises(module.AparentBaselineError, match="declared GPU"):
        module.one_hot(["A" * 164], torch.device("cpu"))


def test_result_stage_withholds_test_before_aparent_inference() -> None:
    module = _module()
    assert module.splits_for_result_stage("HPO_VALIDATION_ONLY") == ("TRAIN", "VALIDATION")
    assert module.splits_for_result_stage("FROZEN_DEVELOPMENT_TEST") == module.SPLITS
    with pytest.raises(module.AparentBaselineError, match="invalid result_stage"):
        module.splits_for_result_stage("")


def test_execute_persists_live_contract_artifacts(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    device = _device()
    records = [
        module.TaskRecord("train", "s1", "A" * 164, "C" * 164, 0.1, "TRAIN"),
        module.TaskRecord("validation", "s2", "G" * 164, "T" * 164, 0.2, "VALIDATION"),
    ]
    manifest_rows = [
        {"canonical_record_id": row.record_id, "study_unit_id": module.TASK_STUDY}
        for row in records
    ]

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.ones(1))

    monkeypatch.setattr(module, "load_task_records", lambda *_args: (records, manifest_rows))
    monkeypatch.setattr(module, "AparentBase", lambda _path: DummyModel())
    monkeypatch.setattr(
        module,
        "predict",
        lambda _model, rows, _device, _batch_size, _start, _end: {
            row.record_id: 0.25 for row in rows
        },
    )
    output = tmp_path / "aparent_run"
    summary = module.execute({
        "evaluation_outcomes_accessed": False,
        "device": str(device),
        "physical_gpu_index": device.index,
        "official_git_revision": "69ad29791709b48689ff5d9e3a3daefc568de9ce",
        "cut_start": 80,
        "cut_end": 105,
        "result_stage": "HPO_VALIDATION_ONLY",
        "canonical_path": str(tmp_path / "canonical.jsonl"),
        "development_manifest_path": str(tmp_path / "manifest.jsonl"),
        "weight_path": str(tmp_path / "weights.h5"),
        "batch_size": 2,
    }, output)
    assert summary["status"] == "APARENT_GSE269595_COMMON_TASK_COMPLETED"
    for name in (
        "config.yaml", "run_config.json", "task_manifest.jsonl", "train.log",
        "metrics.jsonl", "validation_predictions.jsonl", "summary.json", "final_summary.json",
    ):
        assert (output / name).is_file(), name
    assert "RUN_STARTED" in (output / "train.log").read_text()
    assert "INFERENCE_COMPLETED" in (output / "metrics.jsonl").read_text()


def test_summary_records_explicit_cuda_validation_without_cpu_fallback() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"cpu_fallback_used": False' in source
    assert '"cuda_validation_tensors_verified": True' in source
