from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "core/route2_base_flow_model.py"
TRAIN_PATH = ROOT / "scripts/route_a_v3/train_route2_base_flow_g0_v1.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_model_masks_reedit_revert_padding_and_keeps_stop() -> None:
    model_module = _load(MODEL_PATH, "route2_base_flow_model_for_test")
    model = model_module.Route2BaseFlowModel(hidden_dim=16, assay_count=1, context_count=1)
    source = torch.tensor([[0, 1, 4]])
    current = torch.tensor([[2, 1, 4]])
    padding = torch.tensor([[False, False, True]])
    rates, legal = model.rates(
        source, current, padding, torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), torch.tensor([1])
    )
    assert rates.shape == legal.shape == (1, 13)
    assert legal[0, :4].sum() == 0
    assert legal[0, 4:8].sum() == 3
    assert legal[0, 8:12].sum() == 0
    assert legal[0, -1]
    assert torch.all(rates[~legal] == 0)
    assert torch.all(rates[legal] > 0)


def test_dataset_target_is_always_legal_and_t_is_normalized() -> None:
    sys.path.insert(0, str(ROOT))
    trainer = _load(TRAIN_PATH, "train_route2_base_flow_for_dataset_test")
    record = trainer.FlowRecord("r", "AAAU", "CAAU", ((0, 1),), 1, "g", 0, "a", "c")
    dataset = trainer.FlowTrajectoryDataset([record], {"a": 0}, {"c": 0}, seed=3)
    model_module = _load(MODEL_PATH, "route2_base_flow_model_for_dataset_test")
    model = model_module.Route2BaseFlowModel(hidden_dim=16, assay_count=1, context_count=1)
    for epoch in range(4):
        dataset.set_epoch(epoch)
        batch = trainer.collate_examples([dataset[0]])
        _, legal = model(
            batch["source_tokens"], batch["current_tokens"], batch["padding_mask"],
            batch["region_ids"], batch["assay_ids"], batch["context_ids"], batch["remaining_budget"],
        )
        assert legal.gather(1, batch["target"][:, None]).all()
    stop_record = trainer.FlowRecord("stop", "AAAU", "CAAU", ((0, 1),), 3, "g", 0, "a", "c")
    stop_dataset = trainer.FlowTrajectoryDataset([stop_record], {"a": 0}, {"c": 0}, seed=0)
    observed_stop = False
    for epoch in range(20):
        stop_dataset.set_epoch(epoch)
        example = stop_dataset[0]
        observed_stop |= example["target_stop"]
    assert observed_stop
    assert trainer.normalize_sequence("ACTT") == "ACUU"


def test_loader_uses_only_manifest_selected_development_rows(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT))
    trainer = _load(TRAIN_PATH, "train_route2_base_flow_for_loader_test")
    path = tmp_path / "canonical.jsonl"
    rows = [
        {
            "canonical_record_id": "train",
            "pool_assignment": "DEVELOPMENT",
            "source_sequence": "AAAA",
            "candidate_sequence": "CAAA",
            "region": "5UTR",
            "study_unit_id": "S",
            "source_id": "SRC",
            "endpoint_id": "E",
            "assay_id": "a",
            "biological_context_id": "c",
        },
        {
            "canonical_record_id": "evaluation",
            "pool_assignment": "EVALUATION",
            "source_sequence": "AAAA",
            "candidate_sequence": "GAAA",
            "region": "5UTR",
            "study_unit_id": "S",
            "source_id": "SRC",
            "endpoint_id": "E",
            "assay_id": "a",
            "biological_context_id": "c",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    records = trainer.load_records([path], {"train"}, allowed_budgets=(1, 3, 5))
    assert [record.record_id for record in records] == ["train"]


def test_manifest_loader_rejects_evaluation_pool_even_if_split_text_matches(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT))
    trainer = _load(TRAIN_PATH, "train_route2_base_flow_for_manifest_pool_test")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({
        "canonical_record_id": "e",
        "pool_assignment": "EVALUATION",
        "split": "TRAIN",
    }) + "\n")
    with pytest.raises(trainer.BaseFlowTrainingError, match="non-Development"):
        trainer.load_manifest_ids(manifest, "TRAIN")


def test_training_refuses_cpu_fallback(monkeypatch) -> None:
    sys.path.insert(0, str(ROOT))
    trainer = _load(TRAIN_PATH, "train_route2_base_flow_for_cuda_test")
    with pytest.raises(trainer.BaseFlowTrainingError, match="CPU fallback is forbidden"):
        trainer.require_cuda_device("cpu", 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(trainer.BaseFlowTrainingError, match="CUDA is unavailable"):
        trainer.require_cuda_device("cuda:0", 0)


def test_training_refuses_cuda_device_remapping(monkeypatch) -> None:
    sys.path.insert(0, str(ROOT))
    trainer = _load(TRAIN_PATH, "train_route2_base_flow_for_cuda_remap_test")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    with pytest.raises(trainer.BaseFlowTrainingError, match="remapping is forbidden"):
        trainer.require_cuda_device("cuda:0", 0)


def test_gpu_base_flow_training_persists_live_contract_artifacts(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for base-flow training artifact integration test")
    sys.path.insert(0, str(ROOT))
    trainer = _load(TRAIN_PATH, "train_route2_base_flow_live_artifact_test")
    physical_index = int(os.environ.get("ROUTE2_TEST_CUDA_INDEX", "0"))
    manifest = tmp_path / "manifest.jsonl"
    canonical = tmp_path / "canonical.jsonl"
    manifest_rows = []
    canonical_rows = []
    for index, split in enumerate(("TRAIN", "VALIDATION")):
        record_id = f"r{index}"
        manifest_rows.append({
            "canonical_record_id": record_id,
            "pool_assignment": "DEVELOPMENT",
            "split": split,
        })
        canonical_rows.append({
            "canonical_record_id": record_id,
            "pool_assignment": "DEVELOPMENT",
            "source_sequence": "AAAA",
            "candidate_sequence": "CAAA",
            "region": "5UTR",
            "study_unit_id": "S",
            "source_id": f"SRC{index}",
            "endpoint_id": "E",
            "assay_id": "A",
            "biological_context_id": "C",
        })
    manifest.write_text("".join(json.dumps(row) + "\n" for row in manifest_rows))
    canonical.write_text("".join(json.dumps(row) + "\n" for row in canonical_rows))
    output = tmp_path / "flow_run"
    summary = trainer.train({
        "device": f"cuda:{physical_index}",
        "physical_gpu_index": physical_index,
        "development_manifest": str(manifest),
        "canonical_paths": [str(canonical)],
        "allowed_edit_budgets": [1, 3, 5],
        "seed": 7,
        "batch_size": 1,
        "num_workers": 0,
        "hidden_dim": 16,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "epochs": 1,
    }, output)
    assert summary["optimizer_steps"] == 1
    for name in (
        "train.log", "metrics.jsonl", "config.yaml", "latest.pt", "best.pt",
        "final_summary.json", "training_summary.json", "base_flow_checkpoint.pt",
    ):
        assert (output / name).is_file(), name
    assert len((output / "metrics.jsonl").read_text().splitlines()) == 1
    assert "TRAINING_COMPLETED" in (output / "train.log").read_text()


def test_base_flow_source_group_weights_have_equal_group_mass() -> None:
    sys.path.insert(0, str(ROOT))
    trainer = _load(TRAIN_PATH, "train_route2_base_flow_for_weight_test")
    records = [
        trainer.FlowRecord("a", "AA", "CA", ((0, 1),), 1, "g1", 0, "a", "c"),
        trainer.FlowRecord("b", "AA", "GA", ((0, 2),), 1, "g1", 0, "a", "c"),
        trainer.FlowRecord("c", "AA", "UA", ((0, 3),), 1, "g2", 0, "a", "c"),
    ]
    dataset = trainer.FlowTrajectoryDataset(records, {"a": 1, "__UNK__": 0}, {"c": 1, "__UNK__": 0}, seed=1)
    group_one = dataset[0]["sample_weight"] + dataset[1]["sample_weight"]
    group_two = dataset[2]["sample_weight"]
    assert group_one == pytest.approx(group_two)


def test_training_nll_matches_the_normalized_rates_used_for_sampling() -> None:
    sys.path.insert(0, str(ROOT))
    trainer = _load(TRAIN_PATH, "train_route2_base_flow_for_rate_nll_test")
    model_module = _load(MODEL_PATH, "route2_base_flow_model_for_rate_nll_test")
    model = model_module.Route2BaseFlowModel(hidden_dim=16, assay_count=1, context_count=1)
    record = trainer.FlowRecord("r", "AA", "CA", ((0, 1),), 1, "g", 0, "a", "c")
    batch = trainer.collate_examples([
        trainer.FlowTrajectoryDataset([record], {"a": 0}, {"c": 0}, seed=1)[0]
    ])
    observed = trainer._loss(model, batch)
    rates, legal = model.rates(
        batch["source_tokens"], batch["current_tokens"], batch["padding_mask"],
        batch["region_ids"], batch["assay_ids"], batch["context_ids"], batch["remaining_budget"],
    )
    target = int(batch["target"][0])
    expected = -torch.log(rates[0, target] / rates[0, legal[0]].sum())
    assert float(observed.detach()) == pytest.approx(float(expected.detach()), abs=1e-6)
