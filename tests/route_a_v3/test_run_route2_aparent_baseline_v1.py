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
