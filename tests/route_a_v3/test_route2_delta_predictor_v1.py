from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MODEL_PATH = ROOT / "core/route2_delta_predictor.py"
TRAIN_PATH = ROOT / "scripts/route_a_v3/train_route2_delta_predictor_v1.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _model(module):
    return module.Route2DeltaPredictor(
        hidden_dim=16, depth=2, study_count=1, assay_count=1, context_count=1, endpoint_count=1
    ).eval()


def _forward(model, source, candidate):
    batch = source.shape[0]
    return model(
        source, candidate, torch.zeros_like(source, dtype=torch.bool),
        *[torch.zeros(batch, dtype=torch.long) for _ in range(5)],
    )["mean"]


def test_swap_antisymmetry_and_identity_zero_are_exact() -> None:
    module = _load(MODEL_PATH, "route2_delta_model_test")
    model = _model(module)
    source = torch.tensor([[0, 0, 0, 0], [0, 1, 2, 3]])
    candidate = torch.tensor([[1, 0, 0, 0], [3, 2, 1, 0]])
    forward = _forward(model, source, candidate)
    reverse = _forward(model, candidate, source)
    identity = _forward(model, source, source)
    assert torch.equal(forward, -reverse)
    assert torch.equal(identity, torch.zeros_like(identity))


def test_structural_constraints_remain_exact_in_training_mode() -> None:
    module = _load(MODEL_PATH, "route2_delta_model_training_constraint_test")
    model = _model(module).train()
    source = torch.tensor([[0, 1, 2, 3]])
    candidate = torch.tensor([[1, 1, 2, 3]])
    assert torch.equal(_forward(model, source, candidate), -_forward(model, candidate, source))
    assert torch.equal(_forward(model, source, source), torch.zeros(1))


def test_study_specific_scale_calibration_preserves_constraints_and_scales_delta() -> None:
    module = _load(MODEL_PATH, "route2_delta_study_scale_test")
    model = module.Route2DeltaPredictor(
        hidden_dim=16, depth=2, study_count=2, assay_count=1,
        context_count=1, endpoint_count=1, study_specific_scale_calibration=True,
    ).eval()
    with torch.no_grad():
        model.study.weight[1].copy_(model.study.weight[0])
        model.study_log_scale.weight[0] = 0.0
        model.study_log_scale.weight[1] = torch.log(torch.tensor(2.0))
    source = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    candidate = torch.tensor([[1, 1, 2, 3], [1, 1, 2, 3]])
    padding = torch.zeros_like(source, dtype=torch.bool)
    studies = torch.tensor([0, 1])
    zeros = torch.zeros(2, dtype=torch.long)
    forward = model(source, candidate, padding, studies, zeros, zeros, zeros, zeros)["mean"]
    reverse = model(candidate, source, padding, studies, zeros, zeros, zeros, zeros)["mean"]
    identity = model(source, source, padding, studies, zeros, zeros, zeros, zeros)["mean"]
    assert forward[1].item() == pytest.approx(2.0 * forward[0].item())
    assert torch.equal(forward, -reverse)
    assert torch.equal(identity, torch.zeros_like(identity))


def test_normalized_position_channels_are_length_relative_and_edit_gated() -> None:
    module = _load(MODEL_PATH, "route2_delta_model_position_channel_test")
    padding_mask = torch.tensor([
        [False, False, False, True],
        [False, False, False, False],
    ])
    edited = torch.tensor([
        [[0.0], [1.0], [0.0], [0.0]],
        [[1.0], [0.0], [0.0], [1.0]],
    ])
    position, edited_position = module.normalized_position_channels(padding_mask, edited)
    torch.testing.assert_close(position.squeeze(-1), torch.tensor([
        [0.0, 0.5, 1.0, 0.0],
        [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
    ]))
    torch.testing.assert_close(edited_position.squeeze(-1), torch.tensor([
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]))


def test_main_and_full_pair_baseline_receive_the_same_position_channels() -> None:
    module = _load(MODEL_PATH, "route2_delta_model_position_width_test")
    main = _model(module)
    baseline = module.Route2NeuralBaseline(
        mode="full_pair_cnn", hidden_dim=16, depth=1, study_count=1,
        assay_count=1, context_count=1, endpoint_count=1,
    )
    assert main.input_projection.in_features == 16 * 3 + 3
    assert baseline.pair_input.in_features == 16 * 3 + 3


def test_length_bucket_batches_avoid_global_max_padding() -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_trainer_bucket_test")
    records = [
        trainer.DeltaRecord(str(index), "TRAIN", "A" * length, "C" + "A" * (length - 1), 1.0, f"g{index}", "s", "a", "c", "e", 0)
        for index, length in enumerate((50, 51, 52, 1800))
    ]
    sampler = trainer.LengthBucketBatchSampler(records, batch_size=2, seed=1, shuffle=False)
    batches = list(sampler)
    assert batches == [[0, 1], [2, 3]]
    # The long row shares only its local batch, not every short record.
    assert 0 not in batches[-1] and 1 not in batches[-1]


def test_collate_uses_dynamic_batch_length_and_normalizes_t() -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_trainer_collate_test")
    assert trainer._normalize("ACTT") == "ACUU"
    rows = [
        {"record_id": "a", "source": [0, 0], "candidate": [1, 0], "target": 1.0, "sample_weight": 1.0, "study": 0, "assay": 0, "context": 0, "endpoint": 0, "region": 0},
        {"record_id": "b", "source": [0, 0, 0], "candidate": [0, 1, 0], "target": 2.0, "sample_weight": 1.0, "study": 0, "assay": 0, "context": 0, "endpoint": 0, "region": 0},
    ]
    rows[0]["source_group"] = "g1"
    rows[1]["source_group"] = "g2"
    batch = trainer.collate(rows)
    assert batch["source_tokens"].shape == (2, 3)
    assert batch["padding_mask"].tolist() == [[False, False, True], [False, False, False]]


def test_source_group_weights_sum_to_one_per_group() -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_trainer_weight_test")
    records = [
        trainer.DeltaRecord("a", "TRAIN", "AA", "CA", 1.0, "g1", "s", "a", "c", "e", 0),
        trainer.DeltaRecord("b", "TRAIN", "AA", "GA", 2.0, "g1", "s", "a", "c", "e", 0),
        trainer.DeltaRecord("c", "TRAIN", "AA", "UA", 3.0, "g2", "s", "a", "c", "e", 0),
    ]
    vocabs = {field: {"__UNK__": 0, value: 1} for field, value in {"study": "s", "assay": "a", "context": "c", "endpoint": "e"}.items()}
    dataset = trainer.DeltaDataset(records, vocabs)
    group_one = dataset[0]["sample_weight"] + dataset[1]["sample_weight"]
    group_two = dataset[2]["sample_weight"]
    assert group_one == pytest.approx(group_two)
    assert np.mean([dataset[index]["sample_weight"] for index in range(3)]) == pytest.approx(1.0)


def test_group_batch_sampler_never_splits_source_pool() -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_group_sampler_test")
    records = [
        trainer.DeltaRecord("a", "TRAIN", "AA", "CA", 1.0, "g1", "s", "a", "c", "e", 0),
        trainer.DeltaRecord("b", "TRAIN", "AA", "GA", 2.0, "g1", "s", "a", "c", "e", 0),
        trainer.DeltaRecord("c", "TRAIN", "AA", "UA", 3.0, "g2", "s", "a", "c", "e", 0),
    ]
    batches = list(trainer.SourceGroupBatchSampler(records, batch_size=1, seed=1, shuffle=False))
    assert any(batch == [0, 1] for batch in batches)
    assert not any((0 in batch) ^ (1 in batch) for batch in batches)


def test_pairwise_and_listwise_losses_reward_correct_within_group_order() -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_ranking_loss_test")
    batch = {
        "target": torch.tensor([0.0, 1.0, 2.0]),
        "sample_weight": torch.ones(3),
        "source_groups": ["g", "g", "g"],
    }
    correct = {"mean": torch.tensor([0.0, 1.0, 2.0]), "log_variance": torch.zeros(3)}
    reversed_output = {"mean": torch.tensor([2.0, 1.0, 0.0]), "log_variance": torch.zeros(3)}
    for loss_kind in ("pairwise", "listwise"):
        assert trainer.ranking_loss(correct, batch, loss_kind) < trainer.ranking_loss(reversed_output, batch, loss_kind)


def test_huber_regression_does_not_train_unjustified_variance_head() -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_huber_loss_test")
    target = torch.tensor([0.0, 1.0])
    weights = torch.tensor([1.0, 2.0])
    first = {"mean": torch.tensor([0.0, 2.0]), "log_variance": torch.tensor([-8.0, 8.0])}
    second = {"mean": torch.tensor([0.0, 2.0]), "log_variance": torch.tensor([8.0, -8.0])}
    assert trainer.huber_loss(first, target, weights) == trainer.huber_loss(second, target, weights)


def test_multitask_loss_keeps_singleton_a1_regression_with_rankable_a2_group() -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_multitask_loss_test")
    batch = {
        "target": torch.tensor([5.0, 0.0, 1.0]),
        "sample_weight": torch.ones(3),
        "source_groups": ["a1_singleton", "a2", "a2"],
    }
    good_a1 = {"mean": torch.tensor([5.0, 0.0, 1.0]), "log_variance": torch.zeros(3)}
    bad_a1 = {"mean": torch.tensor([-5.0, 0.0, 1.0]), "log_variance": torch.zeros(3)}
    assert trainer.ranking_loss(good_a1, batch, "pairwise") == trainer.ranking_loss(bad_a1, batch, "pairwise")
    assert trainer.multitask_loss(good_a1, batch, "pairwise", 1.0, 1.0) < trainer.multitask_loss(bad_a1, batch, "pairwise", 1.0, 1.0)


def _delta_record(trainer, record_id, study, group, target=0.0):
    return trainer.DeltaRecord(
        record_id=record_id,
        split="TRAIN",
        source="AAAA",
        candidate="CAAA",
        target=target,
        source_group=f"{study}::{group}",
        study=study,
        assay=f"assay-{study}",
        context=f"context-{study}",
        endpoint=f"endpoint-{study}",
        region=0,
    )


def test_study_balanced_weights_equalize_studies_then_source_groups() -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_study_weight_test")
    records = [
        _delta_record(trainer, "a1", "A", "g1"),
        _delta_record(trainer, "a2", "A", "g1"),
        _delta_record(trainer, "a3", "A", "g2"),
        _delta_record(trainer, "b1", "B", "g1"),
    ]
    vocabs = {field: trainer.build_vocab(records, field) for field in ("study", "assay", "context", "endpoint")}
    dataset = trainer.DeltaDataset(
        records,
        vocabs,
        weighting_mode="STUDY_THEN_SOURCE_CONTEXT_ENDPOINT_GROUP",
    )
    weights = [dataset[index]["sample_weight"] for index in range(len(dataset))]
    assert sum(weights[:3]) == pytest.approx(weights[3])
    assert weights[0] + weights[1] == pytest.approx(weights[2])
    assert sum(weights) / len(weights) == pytest.approx(1.0)


def test_no_context_mode_masks_all_categorical_metadata_but_keeps_region() -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_no_context_test")
    records = [_delta_record(trainer, "a", "A", "g")]
    vocabs = {field: {"__UNK__": 0} for field in ("study", "assay", "context", "endpoint")}
    row = trainer.DeltaDataset(
        records,
        vocabs,
        metadata_mode="SEQUENCE_AND_REGION_ONLY",
    )[0]
    assert (row["study"], row["assay"], row["context"], row["endpoint"]) == (0, 0, 0, 0)
    assert row["region"] == 0


def test_explicit_study_subset_is_auditable_and_rejects_duplicates() -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_study_subset_test")
    records = [
        _delta_record(trainer, "a", "A", "g"),
        _delta_record(trainer, "b", "B", "g"),
    ]
    selected, included, excluded = trainer.select_study_subset(records, ["B"])
    assert [row.record_id for row in selected] == ["b"]
    assert included == ["B"]
    assert excluded == 1
    with pytest.raises(trainer.DeltaTrainingError, match="duplicated"):
        trainer.select_study_subset(records, ["A", "A"])


def test_explicit_region_subset_is_auditable_and_rejects_mixed_region_labels() -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_region_subset_test")
    records = [
        _delta_record(trainer, "five", "A", "g"),
        trainer.DeltaRecord(
            record_id="three", split="TRAIN", source="AAAA", candidate="CAAA",
            target=1.0, source_group="h", study="A", assay="A", context="C",
            endpoint="E", region=1,
        ),
    ]
    selected, included, excluded = trainer.select_region_subset(records, ["3′UTR"])
    assert [row.record_id for row in selected] == ["three"]
    assert included == ["3UTR"]
    assert excluded == 1
    with pytest.raises(trainer.DeltaTrainingError, match="unsupported"):
        trainer.select_region_subset(records, ["CDS"])


def _fixed_split_records(trainer):
    return [
        trainer.DeltaRecord(
            record_id=split.lower(), split=split, source="AAAA", candidate="CAAA",
            target=float(index), source_group=f"g{index}", study="S", assay="A",
            context="C", endpoint="E", region=0,
        )
        for index, split in enumerate(("TRAIN", "VALIDATION", "TEST"))
    ]


def test_hpo_stage_withholds_development_test_outcomes() -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_hpo_stage_test")
    selected, withheld = trainer.fixed_split_records(
        _fixed_split_records(trainer), "HPO_VALIDATION_ONLY"
    )
    assert set(selected) == {"TRAIN", "VALIDATION"}
    assert withheld == 1


def test_frozen_validation_stage_keeps_validation_out_of_training() -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_frozen_validation_stage_test")
    selected, withheld = trainer.fixed_split_records(
        _fixed_split_records(trainer), "FROZEN_DEVELOPMENT_VALIDATION"
    )
    assert set(selected) == {"TRAIN", "VALIDATION"}
    assert [row.record_id for row in selected["TRAIN"]] == ["train"]
    assert [row.record_id for row in selected["VALIDATION"]] == ["validation"]
    assert withheld == 1


def test_frozen_stage_exposes_development_test_once() -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_frozen_test_stage_test")
    selected, withheld = trainer.fixed_split_records(
        _fixed_split_records(trainer), "FROZEN_DEVELOPMENT_TEST"
    )
    assert set(selected) == {"TRAIN", "TEST"}
    assert [row.record_id for row in selected["TRAIN"]] == ["train", "validation"]
    assert withheld == 0


def test_fixed_split_rejects_unregistered_result_stage() -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_invalid_result_stage_test")
    with pytest.raises(trainer.DeltaTrainingError, match="invalid result_stage"):
        trainer.fixed_split_records(_fixed_split_records(trainer), "")


def test_manifest_loader_retains_component_metadata_for_loso(tmp_path: Path) -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_manifest_loso_test")
    import json
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps({
        "canonical_record_id": "r", "pool_assignment": "DEVELOPMENT", "split": "TRAIN",
        "study_unit_id": "S", "connected_source_component_id": "COMP",
    }) + "\n")
    result = trainer.load_manifest(path)
    assert result["r"] == {
        "split": "TRAIN", "study_unit_id": "S", "connected_source_component_id": "COMP"
    }


def test_holdout_category_is_unknown_when_vocab_is_built_from_training_only() -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_training_vocab_test")
    train = trainer.DeltaRecord("a", "TRAIN", "AA", "CA", 1.0, "g1", "TRAIN_STUDY", "a", "c", "e", 0)
    holdout = trainer.DeltaRecord("b", "TEST", "AA", "GA", 2.0, "g2", "HELD_OUT_STUDY", "a", "c", "e", 0)
    vocab = trainer.build_vocab([train], "study")
    assert vocab["__UNK__"] == 0
    assert "HELD_OUT_STUDY" not in vocab
    assert trainer.DeltaDataset([holdout], {"study": vocab, "assay": {"__UNK__": 0}, "context": {"__UNK__": 0}, "endpoint": {"__UNK__": 0}})[0]["study"] == 0


def test_delta_training_refuses_cpu_fallback(monkeypatch) -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_trainer_cuda_test")
    with pytest.raises(trainer.DeltaTrainingError, match="CPU fallback is forbidden"):
        trainer.require_cuda("cpu", 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(trainer.DeltaTrainingError, match="CUDA is unavailable"):
        trainer.require_cuda("cuda:0", 0)


def test_delta_training_refuses_cuda_device_remapping(monkeypatch) -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_trainer_cuda_remap_test")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    with pytest.raises(trainer.DeltaTrainingError, match="remapping is forbidden"):
        trainer.require_cuda("cuda:0", 0)


def test_gpu_training_persists_live_contract_artifacts(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for training artifact integration test")
    trainer = _load(TRAIN_PATH, "route2_delta_live_artifact_test")
    physical_index = int(os.environ.get("ROUTE2_TEST_CUDA_INDEX", "0"))
    manifest = tmp_path / "manifest.jsonl"
    canonical = tmp_path / "canonical.jsonl"
    manifest_rows = []
    canonical_rows = []
    for index, split in enumerate(("TRAIN", "VALIDATION", "TEST")):
        record_id = f"r{index}"
        manifest_rows.append({
            "canonical_record_id": record_id,
            "pool_assignment": "DEVELOPMENT",
            "split": split,
            "study_unit_id": "S",
            "connected_source_component_id": f"C{index}",
        })
        canonical_rows.append({
            "canonical_record_id": record_id,
            "pool_assignment": "DEVELOPMENT",
            "source_sequence": "AAAA",
            "candidate_sequence": "CAAA",
            "direction_normalized_delta": float(index),
            "study_unit_id": "S",
            "source_id": f"SOURCE{index}",
            "assay_id": "A",
            "biological_context_id": "C",
            "endpoint_id": "E",
            "region": "5UTR",
        })
    manifest.write_text("".join(json.dumps(row) + "\n" for row in manifest_rows))
    canonical.write_text("".join(json.dumps(row) + "\n" for row in canonical_rows))
    output = tmp_path / "run"
    summary = trainer.train({
        "device": f"cuda:{physical_index}",
        "physical_gpu_index": physical_index,
        "development_manifest": str(manifest),
        "canonical_paths": [str(canonical)],
        "run_mode": "FIXED_GROUPED_SPLIT",
        "result_stage": "HPO_VALIDATION_ONLY",
        "baseline_id": "delta_main_test",
        "model_kind": trainer.ROUTE2_DELTA_MODEL_KIND,
        "metadata_mode": "FULL_CONTEXT",
        "training_weighting_mode": "STUDY_THEN_SOURCE_CONTEXT_ENDPOINT_GROUP",
        "loss_kind": "huber",
        "batch_size": 1,
        "seed": 7,
        "hidden_dim": 16,
        "depth": 1,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "epochs": 1,
        "num_workers": 0,
    }, output)
    assert summary["optimizer_steps"] == 1
    for name in (
        "train.log", "metrics.jsonl", "config.yaml", "latest.pt", "best.pt",
        "final_summary.json", "training_summary.json", "delta_predictor_checkpoint.pt",
    ):
        assert (output / name).is_file(), name
    assert len((output / "metrics.jsonl").read_text().splitlines()) == 1
    assert "TRAINING_COMPLETED" in (output / "train.log").read_text()


@pytest.mark.parametrize("mode", ["candidate_cnn", "siamese_cnn", "full_pair_cnn", "small_transformer"])
def test_neural_baselines_share_prediction_interface(mode: str) -> None:
    module = _load(MODEL_PATH, f"route2_neural_baseline_{mode}_test")
    model = module.Route2NeuralBaseline(
        mode=mode, hidden_dim=16, depth=1, study_count=1, assay_count=1,
        context_count=1, endpoint_count=1, max_length=16,
    ).eval()
    source = torch.tensor([[0, 0, 0, 0]])
    candidate = torch.tensor([[1, 0, 0, 0]])
    output = model(
        source, candidate, torch.zeros_like(source, dtype=torch.bool),
        *[torch.zeros(1, dtype=torch.long) for _ in range(5)],
    )
    assert output["mean"].shape == output["log_variance"].shape == (1,)
    assert torch.isfinite(output["mean"]).all()
