from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/run_route2_external_prediction_baselines_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("route2_external_baselines_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_even_kernel_same_padding_matches_tensorflow_length() -> None:
    module = _load()
    x = torch.arange(2 * 4 * 50, dtype=torch.float32).reshape(2, 4, 50)
    weight = torch.ones((3, 4, 8))
    result = module._same_conv(x, weight, torch.zeros(3))
    assert result.shape == (2, 3, 50)
    expected_first = x[0, :, :5].sum()
    assert result[0, 0, 0] == expected_first


def test_task_loader_is_development_only_and_exact(tmp_path: Path) -> None:
    module = _load()
    canonical = tmp_path / "canonical.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    rows = []
    manifests = []
    for index, split in enumerate(("TRAIN", "VALIDATION", "TEST"), start=1):
        record_id = f"R{index}"
        rows.append({
            "canonical_record_id": record_id, "study_unit_id": "GSE114002",
            "pool_assignment": "DEVELOPMENT", "region": "5UTR", "endpoint_id": "MEAN_RIBOSOME_LOAD",
            "source_id": f"S{index}", "source_sequence": "A" * 50,
            "candidate_sequence": "C" + "A" * 49, "direction_normalized_delta": 0.1 * index,
        })
        manifests.append({"canonical_record_id": record_id, "study_unit_id": "GSE114002", "split": split})
    canonical.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest.write_text("".join(json.dumps(row) + "\n" for row in manifests), encoding="utf-8")
    records, task_manifest = module.load_task_records(canonical, manifest)
    assert len(records) == len(task_manifest) == 3
    assert {record.split for record in records} == {"TRAIN", "VALIDATION", "TEST"}
    records, task_manifest = module.load_task_records(
        canonical, manifest, module.splits_for_result_stage("HPO_VALIDATION_ONLY")
    )
    assert len(records) == len(task_manifest) == 2
    assert {record.split for record in records} == {"TRAIN", "VALIDATION"}


def test_result_stage_rejects_implicit_test_exposure() -> None:
    module = _load()
    assert module.splits_for_result_stage("FROZEN_DEVELOPMENT_TEST") == module.SPLITS
    with pytest.raises(module.ExternalBaselineError, match="invalid result_stage"):
        module.splits_for_result_stage("")


def test_task_loader_rejects_non_development(tmp_path: Path) -> None:
    module = _load()
    canonical = tmp_path / "canonical.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    canonical.write_text(json.dumps({
        "canonical_record_id": "R", "study_unit_id": "GSE114002", "pool_assignment": "EVALUATION",
        "region": "5UTR", "endpoint_id": "MEAN_RIBOSOME_LOAD", "source_id": "S",
        "source_sequence": "A" * 50, "candidate_sequence": "C" + "A" * 49,
        "direction_normalized_delta": 0.1,
    }) + "\n", encoding="utf-8")
    manifest.write_text(json.dumps({"canonical_record_id": "R", "study_unit_id": "GSE114002", "split": "TRAIN"}) + "\n", encoding="utf-8")
    with pytest.raises(module.ExternalBaselineError, match="non-Development"):
        module.load_task_records(canonical, manifest)


def test_constant_external_predictions_keep_spearman_undefined() -> None:
    module = _load()
    records = [
        module.TaskRecord(f"R{index}", f"S{index}", "A" * 50, "C" + "A" * 49, float(index), "TEST")
        for index in range(3)
    ]
    predictions = {record.record_id: 0.0 for record in records}
    assert module._metrics(records, predictions, "TEST")["spearman"] is None


def test_rnafm_artifact_is_not_described_as_official_checkpoint() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "multimolecule_rnafm_frozen_linear_probe" in source
    assert '"official_original_checkpoint_used": False' in source
    assert '"artifact_identity": "multimolecule/rnafm unofficial conversion"' in source


def test_native_weight_ports_disclose_missing_tensorflow_numeric_parity() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"optimus5prime_official_d53df410"' in source
    assert '"framepool_official_c575f9cd"' in source
    assert '"pytorch_port_numeric_parity_status": "NOT_RUN_TENSORFLOW_UNAVAILABLE"' in source


def test_summary_records_explicit_cuda_execution_without_cpu_fallback() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"cpu_fallback_used": False' in source
    assert '"cuda_training_tensors_verified": True' in source


def test_rnafm_probe_refits_train_plus_validation_only_after_freeze() -> None:
    module = _load()
    if not torch.cuda.is_available():
        pytest.skip("RNA-FM probe test requires CUDA")
    device = torch.device(f"cuda:{int(os.environ.get('ROUTE2_TEST_CUDA_INDEX', '0'))}")
    generator = torch.Generator(device=device).manual_seed(7)
    embeddings = {f"s{index}": torch.randn(4, generator=generator, device=device) for index in range(10)}
    records = [
        module.TaskRecord("t1", "g1", "s0", "s1", 0.2, "TRAIN"),
        module.TaskRecord("t2", "g2", "s2", "s3", -0.1, "TRAIN"),
        module.TaskRecord("v1", "g3", "s4", "s5", 0.3, "VALIDATION"),
        module.TaskRecord("v2", "g4", "s6", "s7", -0.2, "VALIDATION"),
        module.TaskRecord("x1", "g5", "s8", "s9", 0.1, "TEST"),
    ]
    progress_rows = []
    _, hpo, hpo_artifact = module._train_multimolecule_rnafm_probe(
        records[:-1], embeddings, device, 7, 3, 1e-2, 0.0, "HPO_VALIDATION_ONLY",
        progress_rows.append,
    )
    assert hpo["development_validation_folded_into_probe_training"] is False
    assert hpo["best_validation_source_group_weighted_mse"] is not None
    assert set(hpo_artifact["probe_state"]) == {"weight", "bias"}
    assert [row["epoch"] for row in progress_rows] == [1]
    predictions, frozen, _ = module._train_multimolecule_rnafm_probe(
        records, embeddings, device, 7, 3, 1e-2, 0.0, "FROZEN_DEVELOPMENT_TEST"
    )
    assert set(predictions) == {row.record_id for row in records}
    assert frozen["development_validation_folded_into_probe_training"] is True
    assert frozen["best_validation_source_group_weighted_mse"] is None
