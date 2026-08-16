from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/predict_route2_framepool_loso_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("route2_framepool_loso_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_loso_loader_covers_all_holdout_rows_without_outcomes(tmp_path: Path) -> None:
    module = _load()
    canonical = tmp_path / "canonical.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    rows = []
    manifests = []
    for index, split in enumerate(("TRAIN", "VALIDATION", "TEST"), start=1):
        record_id = f"R{index}"
        rows.append({
            "canonical_record_id": record_id,
            "study_unit_id": "GSE114002",
            "pool_assignment": "DEVELOPMENT",
            "region": "5UTR",
            "endpoint_id": "MEAN_RIBOSOME_LOAD",
            "source_id": f"S{index}",
            "source_sequence": "A" * 50,
            "candidate_sequence": "C" + "A" * 49,
            "direction_normalized_delta": 999999.0,
        })
        manifests.append({
            "canonical_record_id": record_id,
            "study_unit_id": "GSE114002",
            "pool_assignment": "DEVELOPMENT",
            "split": split,
        })
    canonical.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest.write_text("".join(json.dumps(row) + "\n" for row in manifests), encoding="utf-8")
    records = module.load_loso_records(canonical, manifest, "GSE114002")
    assert len(records) == 3
    assert {record.record_id for record in records} == {"R1", "R2", "R3"}
    assert {record.target for record in records} == {0.0}
    assert {record.split for record in records} == {"LOSO_HOLDOUT"}


def test_loso_loader_rejects_non_task_holdout(tmp_path: Path) -> None:
    module = _load()
    with pytest.raises(module.FramePoolLosoError, match="only task-matched"):
        module.load_loso_records(tmp_path / "missing", tmp_path / "missing", "GSE200304")


def test_native_external_forward_has_explicit_cuda_guard() -> None:
    source = (ROOT / "scripts/route_a_v3/run_route2_external_prediction_baselines_v1.py").read_text(encoding="utf-8")
    assert "native external prediction left CUDA or became nonfinite" in source


def test_native_external_forward_runs_on_declared_cuda_device() -> None:
    if not torch.cuda.is_available():
        pytest.skip("native external forward test requires CUDA")
    module = _load()
    external = module.external
    device = torch.device(f"cuda:{int(os.environ.get('ROUTE2_TEST_CUDA_INDEX', '0'))}")

    class SumModel(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value.sum(dim=(1, 2))

    records = [
        external.TaskRecord("R1", "S1", "A" * 50, "C" + "A" * 49, 0.0, "LOSO_HOLDOUT")
    ]
    predictions = external._predict_native(SumModel().to(device), records, device, 1)
    assert predictions == {"R1": 0.0}
