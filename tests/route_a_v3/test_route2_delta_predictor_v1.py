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


def test_edit_centered_model_is_exactly_antisymmetric_and_study_invariant() -> None:
    module = _load(MODEL_PATH, "route2_edit_centered_constraint_test")
    model = module.Route2EditCenteredDeltaPredictor(
        hidden_dim=16,
        depth=2,
        study_count=2,
        assay_count=1,
        context_count=1,
        endpoint_count=1,
    ).train()
    source = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    candidate = torch.tensor([[1, 1, 2, 3], [1, 1, 2, 3]])
    padding = torch.zeros_like(source, dtype=torch.bool)
    studies = torch.tensor([0, 1])
    zeros = torch.zeros(2, dtype=torch.long)
    forward = model(source, candidate, padding, studies, zeros, zeros, zeros, zeros)["mean"]
    reverse = model(candidate, source, padding, studies, zeros, zeros, zeros, zeros)["mean"]
    identity = model(source, source, padding, studies, zeros, zeros, zeros, zeros)["mean"]
    assert torch.equal(forward, -reverse)
    assert torch.equal(identity, torch.zeros_like(identity))
    assert forward[0].item() == pytest.approx(forward[1].item())


def test_edit_centered_source_only_control_is_candidate_invariant_and_parameter_matched() -> None:
    module = _load(MODEL_PATH, "route2_edit_centered_source_control_test")
    common = {
        "hidden_dim": 16,
        "depth": 1,
        "study_count": 1,
        "assay_count": 1,
        "context_count": 1,
        "endpoint_count": 1,
    }
    main = module.Route2EditCenteredDeltaPredictor(**common).eval()
    control = module.Route2EditCenteredDeltaPredictor(**common, source_only_control=True).eval()
    assert sum(parameter.numel() for parameter in main.parameters()) == sum(
        parameter.numel() for parameter in control.parameters()
    )
    source = torch.tensor([[0, 1, 2, 3]])
    first_candidate = torch.tensor([[1, 1, 2, 3]])
    second_candidate = torch.tensor([[0, 1, 2, 0]])
    padding = torch.zeros_like(source, dtype=torch.bool)
    zeros = torch.zeros(1, dtype=torch.long)
    first = control(
        source, first_candidate, padding, zeros, zeros, zeros, zeros, zeros
    )["mean"]
    second = control(
        source, second_candidate, padding, zeros, zeros, zeros, zeros, zeros
    )["mean"]
    assert torch.equal(first, second)


def test_pretrained_edit_centered_mean_is_antisymmetric_and_scale_is_symmetric() -> None:
    module = _load(MODEL_PATH, "route2_pretrained_edit_centered_constraint_test")
    model = module.Route2PretrainedEditCenteredDeltaPredictor(
        hidden_dim=32,
        depth=2,
        study_count=2,
        assay_count=2,
        context_count=2,
        endpoint_count=2,
        pretrained_width=24,
        learned_uncertainty=True,
    ).train()
    source = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    candidate = torch.tensor([[1, 1, 2, 3], [0, 0, 1, 2]])
    padding = torch.zeros_like(source, dtype=torch.bool)
    categories = torch.zeros(2, dtype=torch.long)
    source_pretrained = torch.randn(2, 24)
    candidate_pretrained = torch.randn(2, 24)
    forward = model(
        source,
        candidate,
        padding,
        categories,
        categories,
        categories,
        categories,
        categories,
        source_pretrained,
        candidate_pretrained,
    )
    reverse = model(
        candidate,
        source,
        padding,
        categories,
        categories,
        categories,
        categories,
        categories,
        candidate_pretrained,
        source_pretrained,
    )
    identity = model(
        source,
        source,
        padding,
        categories,
        categories,
        categories,
        categories,
        categories,
        source_pretrained,
        source_pretrained,
    )
    assert torch.equal(forward["mean"], -reverse["mean"])
    assert torch.equal(identity["mean"], torch.zeros_like(identity["mean"]))
    assert torch.equal(forward["log_variance"], reverse["log_variance"])


def test_final_refit_uses_all_development_records_without_internal_evaluation() -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_final_all_refit_test")
    records = [
        trainer.DeltaRecord(
            split.lower(), split, "AA", "CA", 1.0, split, "s", "a", "c", "e", 0
        )
        for split in ("TRAIN", "VALIDATION", "TEST")
    ]
    by_split, withheld = trainer.fixed_split_records(
        records, "FINAL_ALL_DEVELOPMENT_REFIT"
    )
    assert list(by_split) == ["TRAIN"]
    assert len(by_split["TRAIN"]) == 3
    assert withheld == 0


def test_uncertainty_loss_can_absorb_residual_and_is_therefore_diagnostic_only() -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_uncertainty_absorption_test")
    target = torch.tensor([1.0, -1.0])
    sample_weight = torch.ones(2)
    narrow = {
        "mean": torch.zeros(2),
        "log_variance": torch.full((2,), -4.0),
    }
    broad = {
        "mean": torch.zeros(2),
        "log_variance": torch.zeros(2),
    }
    assert trainer.gaussian_nll(broad, target, sample_weight) < trainer.gaussian_nll(
        narrow, target, sample_weight
    )


def test_frozen_pretrained_feature_table_requires_exact_record_universe(tmp_path) -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_pretrained_feature_table_test")
    path = tmp_path / "features.pt"
    torch.save(
        {
            "schema_version": "route_a_v3_route2_rnafm_pair_features.v1",
            "model_id": "multimolecule/rnafm",
            "pretrained_parameter_count": 99_000_000,
            "record_ids": ["a", "b"],
            "source_embeddings": torch.randn(2, 16).half(),
            "candidate_embeddings": torch.randn(2, 16).half(),
        },
        path,
    )
    table = trainer.FrozenPretrainedPairFeatures(path, {"a", "b"})
    assert table.width == 16
    assert table.pretrained_parameter_count == 99_000_000
    with pytest.raises(trainer.DeltaTrainingError, match="exactly cover"):
        trainer.FrozenPretrainedPairFeatures(path, {"a"})


def test_generic_mrnabert_feature_table_is_accepted(tmp_path) -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_mrnabert_feature_table_test")
    path = tmp_path / "mrnabert_features.pt"
    torch.save(
        {
            "schema_version": "route_a_v3_route2_frozen_pair_features.v1",
            "encoder_family": "mRNABERT",
            "model_id": "YYLY66/mRNABERT",
            "pretrained_parameter_count": 86_000_000,
            "record_ids": ["a", "b"],
            "source_embeddings": torch.randn(2, 768).half(),
            "candidate_embeddings": torch.randn(2, 768).half(),
        },
        path,
    )
    table = trainer.FrozenPretrainedPairFeatures(path, {"a", "b"})
    assert table.width == 768
    assert table.model_id == "YYLY66/mRNABERT"
    assert table.pretrained_parameter_count == 86_000_000


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


def test_task_gradient_norm_multipliers_center_on_geometric_mean() -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_gradient_multiplier_test")
    multipliers = trainer.task_gradient_norm_loss_multipliers(
        {"task-a": 1.0, "task-b": 4.0}
    )
    assert multipliers == pytest.approx({"task-a": 2.0, "task-b": 0.5})
    with pytest.raises(trainer.DeltaTrainingError, match="invalid"):
        trainer.task_gradient_norm_loss_multipliers({"task-a": 1.0, "task-b": 0.0})


def test_task_gradient_scaled_loss_splits_tasks_before_scaling() -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_gradient_scaled_loss_test")
    output = {
        "mean": torch.tensor([1.0, 1.0, 2.0, 2.0]),
        "log_variance": torch.zeros(4),
    }
    batch = {
        "target": torch.zeros(4),
        "scaled_target": torch.zeros(4),
        "sample_weight": torch.ones(4),
        "source_groups": ["a1", "a2", "b1", "b2"],
        "task_keys": ["task-a", "task-a", "task-b", "task-b"],
    }
    loss = trainer.task_gradient_scaled_training_loss(
        output,
        batch,
        {"task-a": 2.0, "task-b": 0.5},
        "huber",
        1.0,
        1.0,
    )
    # Huber(1) = 0.5 and Huber(2) = 1.5; scaled task mean is (1 + 0.75) / 2.
    assert loss == pytest.approx(0.875)


def test_shared_effect_parameters_exclude_categorical_task_adapters() -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_shared_parameter_test")
    model = trainer.Route2EditCenteredDeltaPredictor(
        hidden_dim=16,
        depth=1,
        study_count=1,
        assay_count=2,
        context_count=2,
        endpoint_count=2,
    )
    names = {name for name, _parameter in trainer.shared_effect_parameters(model)}
    assert "nucleotide.weight" in names
    assert "pair_fusion.0.weight" in names
    assert "assay.weight" not in names
    assert "endpoint.weight" not in names
    assert "region_scale.weight" not in names


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


def test_task_balanced_weights_equalize_tasks_then_source_groups() -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_task_weight_test")
    records = [
        _delta_record(trainer, "a1", "A", "g1"),
        _delta_record(trainer, "a2", "A", "g1"),
        _delta_record(trainer, "a3", "A", "g2"),
        trainer.DeltaRecord(
            record_id="b1", split="TRAIN", source="AAAA", candidate="CAAA",
            target=0.0, source_group="B::g1", study="B", assay="assay-B",
            context="context-B", endpoint="different-task", region=1,
        ),
    ]
    vocabs = {field: trainer.build_vocab(records, field) for field in ("study", "assay", "context", "endpoint")}
    dataset = trainer.DeltaDataset(
        records,
        vocabs,
        weighting_mode="TASK_THEN_SOURCE_CONTEXT_ENDPOINT_GROUP",
    )
    weights = [dataset[index]["sample_weight"] for index in range(len(dataset))]
    assert sum(weights[:3]) == pytest.approx(weights[3])
    assert weights[0] + weights[1] == pytest.approx(weights[2])


def test_transferable_context_masks_only_study_identity() -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_transferable_context_test")
    records = [_delta_record(trainer, "a", "A", "g")]
    vocabs = {field: trainer.build_vocab(records, field) for field in ("study", "assay", "context", "endpoint")}
    row = trainer.DeltaDataset(
        records,
        vocabs,
        metadata_mode="TRANSFERABLE_CONTEXT",
    )[0]
    assert row["study"] == 0
    assert (row["assay"], row["context"], row["endpoint"], row["region"]) == (1, 1, 1, 0)


def test_candidate_permutation_is_deterministic_and_train_only_ready() -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_candidate_permutation_test")
    records = [
        trainer.DeltaRecord(
            record_id=f"r{index}", split="TRAIN", source="AAAA", candidate=candidate,
            target=float(index), source_group=f"g{index}", study="S", assay="A",
            context="C", endpoint="E", region=0,
        )
        for index, candidate in enumerate(("CAAA", "GAAA", "UAAA"))
    ]
    first, summary = trainer.build_training_candidate_permutation(records, 17)
    second, _summary = trainer.build_training_candidate_permutation(records, 17)
    assert first == second
    assert set(first) == {row.record_id for row in records}
    assert summary["changed_candidate_sequence_count"] == len(records)
    assert {first[row.record_id] for row in records} == {row.candidate for row in records}


def test_candidate_permutation_never_crosses_exact_source_task_support() -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_candidate_permutation_support_test")
    records = [
        trainer.DeltaRecord(
            record_id="source_a_1", split="TRAIN", source="AAAA", candidate="CAAA",
            target=1.0, source_group="a1", study="S", assay="A", context="C",
            endpoint="E", region=0,
        ),
        trainer.DeltaRecord(
            record_id="source_a_2", split="TRAIN", source="AAAA", candidate="GAAA",
            target=2.0, source_group="a2", study="S", assay="A", context="D",
            endpoint="E", region=0,
        ),
        trainer.DeltaRecord(
            record_id="source_c", split="TRAIN", source="CCCC", candidate="ACCC",
            target=3.0, source_group="c", study="S", assay="A", context="C",
            endpoint="E", region=0,
        ),
    ]
    overrides, summary = trainer.build_training_candidate_permutation(records, 17)
    assert {overrides["source_a_1"], overrides["source_a_2"]} == {"CAAA", "GAAA"}
    assert "source_c" not in overrides
    assert summary["permutation_stratum"] == "EXACT_SOURCE_SEQUENCE_ENDPOINT_REGION"
    assert summary["candidate_pool_membership_preserved"] is True
    assert summary["edit_distance_multiset_preserved"] is True


def test_dataset_uses_train_scaler_without_changing_raw_target() -> None:
    trainer = _load(TRAIN_PATH, "route2_delta_scaled_dataset_test")
    records = [
        _delta_record(trainer, f"r{index}", "A", f"g{index}", target=value)
        for index, value in enumerate((-2.0, -1.0, 0.0, 1.0, 2.0))
    ]
    vocabs = {field: trainer.build_vocab(records, field) for field in ("study", "assay", "context", "endpoint")}
    scaler = trainer.fit_route2_target_scaler(
        records,
        mode=trainer.TARGET_SCALING_TRAIN_TASK_ROBUST,
        minimum_task_records=3,
    )
    row = trainer.DeltaDataset(records, vocabs, target_scaler=scaler)[0]
    assert row["target"] == -2.0
    assert row["scaled_target"] == pytest.approx(-2.0 / 1.4826)
    assert row["target_scale_source"] == "TASK"


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


def test_task_gradient_calibration_is_cuda_and_zero_update() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for task-gradient calibration")
    trainer = _load(TRAIN_PATH, "route2_delta_gradient_calibration_gpu_test")
    physical_index = int(os.environ.get("ROUTE2_TEST_CUDA_INDEX", "0"))
    device = trainer.require_cuda(f"cuda:{physical_index}", physical_index)
    records = []
    for task_number, (endpoint, region) in enumerate((("E1", 0), ("E2", 1))):
        for index, candidate in enumerate(("CAAA", "GAAA", "UAAA")):
            records.append(trainer.DeltaRecord(
                record_id=f"{endpoint}-{index}",
                split="TRAIN",
                source="AAAA",
                candidate=candidate,
                target=float(index - 1 + task_number),
                source_group=f"{endpoint}-g{index}",
                study=f"S{task_number}",
                assay=f"A{task_number}",
                context=f"C{task_number}",
                endpoint=endpoint,
                region=region,
            ))
    vocabs = {
        "study": {"__UNK__": 0},
        **{
            field: trainer.build_vocab(records, field)
            for field in ("assay", "context", "endpoint")
        },
    }
    scaler = trainer.fit_route2_target_scaler(
        records,
        mode=trainer.TARGET_SCALING_TRAIN_TASK_ROBUST,
        minimum_task_records=2,
    )
    model = trainer.Route2EditCenteredDeltaPredictor(
        hidden_dim=16,
        depth=1,
        study_count=1,
        assay_count=len(vocabs["assay"]),
        context_count=len(vocabs["context"]),
        endpoint_count=len(vocabs["endpoint"]),
    ).to(device)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    result = trainer.calibrate_task_gradient_norms(
        model=model,
        records=records,
        vocabs=vocabs,
        metadata_mode="TRANSFERABLE_CONTEXT",
        weighting_mode="TASK_THEN_SOURCE_CONTEXT_ENDPOINT_GROUP",
        target_scaler=scaler,
        candidate_overrides={},
        loss_kind="huber",
        ranking_loss_weight=1.0,
        huber_delta=1.0,
        batch_size=2,
        seed=7,
        maximum_batches_per_task=2,
        device=device,
    )
    assert result["task_count"] == 2
    assert result["cuda_losses_verified"] is True
    assert result["optimizer_steps"] == result["parameter_updates"] == 0
    assert all(value > 0 for value in result["loss_multipliers"].values())
    for name, value in model.named_parameters():
        assert torch.equal(before[name], value.detach())


def test_gradnorm_training_mode_persists_calibration_and_updates_on_cuda(
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for gradnorm training integration")
    trainer = _load(TRAIN_PATH, "route2_delta_gradnorm_training_gpu_test")
    physical_index = int(os.environ.get("ROUTE2_TEST_CUDA_INDEX", "0"))
    manifest = tmp_path / "manifest.jsonl"
    canonical = tmp_path / "canonical.jsonl"
    manifest_rows = []
    canonical_rows = []
    candidates = ("CAAA", "GAAA", "UAAA", "ACAA", "AGAA", "AUAA", "AACA")
    for task_number, (endpoint, region) in enumerate((("E1", "5UTR"), ("E2", "3UTR"))):
        for index, (split, candidate) in enumerate(zip(
            ("TRAIN", "TRAIN", "TRAIN", "VALIDATION", "VALIDATION", "VALIDATION", "TEST"),
            candidates,
        )):
            record_id = f"{endpoint}-{index}"
            manifest_rows.append({
                "canonical_record_id": record_id,
                "pool_assignment": "DEVELOPMENT",
                "split": split,
                "study_unit_id": f"S{task_number}",
                "connected_source_component_id": f"COMP-{record_id}",
            })
            canonical_rows.append({
                "canonical_record_id": record_id,
                "pool_assignment": "DEVELOPMENT",
                "source_sequence": "AAAA",
                "candidate_sequence": candidate,
                "direction_normalized_delta": float(index - 2 + task_number),
                "study_unit_id": f"S{task_number}",
                "source_id": f"SOURCE-{record_id}",
                "assay_id": f"A{task_number}",
                "biological_context_id": f"C{task_number}",
                "endpoint_id": endpoint,
                "region": region,
            })
    manifest.write_text("".join(json.dumps(row) + "\n" for row in manifest_rows))
    canonical.write_text("".join(json.dumps(row) + "\n" for row in canonical_rows))
    output = tmp_path / "gradnorm-run"
    summary = trainer.train({
        "device": f"cuda:{physical_index}",
        "physical_gpu_index": physical_index,
        "development_manifest": str(manifest),
        "canonical_paths": [str(canonical)],
        "run_mode": "FIXED_GROUPED_SPLIT",
        "result_stage": "HPO_VALIDATION_ONLY",
        "baseline_id": "gradnorm_gpu_smoke",
        "model_kind": trainer.ROUTE2_EDIT_CENTERED_MODEL_KIND,
        "metadata_mode": "TRANSFERABLE_CONTEXT",
        "training_weighting_mode": "TASK_THEN_SOURCE_CONTEXT_ENDPOINT_GROUP",
        "training_update_mode": trainer.TRAINING_UPDATE_TASK_GRADIENT_NORM_CALIBRATED,
        "task_gradient_calibration_max_batches_per_task": 2,
        "target_scaling_mode": trainer.TARGET_SCALING_TRAIN_TASK_ROBUST,
        "target_scale_minimum_task_records": 2,
        "loss_kind": "huber",
        "checkpoint_selection": "BEST_VALIDATION",
        "checkpoint_metric": "TASK_MACRO_SPEARMAN_THEN_STANDARDIZED_MAE",
        "batch_size": 2,
        "seed": 23,
        "hidden_dim": 16,
        "depth": 1,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "epochs": 1,
        "num_workers": 0,
    }, output)
    calibration = summary["task_gradient_calibration"]
    assert summary["training_update_mode"] == trainer.TRAINING_UPDATE_TASK_GRADIENT_NORM_CALIBRATED
    assert calibration["task_count"] == 2
    assert calibration["cuda_losses_verified"] is True
    assert calibration["optimizer_steps"] == calibration["parameter_updates"] == 0
    assert summary["cuda_training_tensors_verified"] is True
    assert summary["cpu_fallback_used"] is False
    assert summary["optimizer_steps"] > 0 and summary["parameter_changed"] is True
    checkpoint = torch.load(
        output / "delta_predictor_checkpoint.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["training_provenance"]["task_gradient_calibration"] == calibration


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


def test_gpu_method_repair_smoke_is_scaled_edit_centered_and_best_selected(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for method-repair integration test")
    trainer = _load(TRAIN_PATH, "route2_delta_method_repair_gpu_test")
    physical_index = int(os.environ.get("ROUTE2_TEST_CUDA_INDEX", "0"))
    manifest = tmp_path / "manifest.jsonl"
    canonical = tmp_path / "canonical.jsonl"
    manifest_rows = []
    canonical_rows = []
    splits = ("TRAIN", "TRAIN", "TRAIN", "VALIDATION", "VALIDATION", "VALIDATION", "TEST")
    candidates = ("CAAA", "GAAA", "UAAA", "ACAA", "AGAA", "AUAA", "AACA")
    for index, (split, candidate) in enumerate(zip(splits, candidates)):
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
            "candidate_sequence": candidate,
            "direction_normalized_delta": float(index - 3),
            "study_unit_id": "S",
            "source_id": f"SOURCE{index}",
            "assay_id": "A",
            "biological_context_id": "C",
            "endpoint_id": "E",
            "region": "5UTR",
        })
    manifest.write_text("".join(json.dumps(row) + "\n" for row in manifest_rows))
    canonical.write_text("".join(json.dumps(row) + "\n" for row in canonical_rows))
    output = tmp_path / "method-repair-run"
    summary = trainer.train({
        "device": f"cuda:{physical_index}",
        "physical_gpu_index": physical_index,
        "development_manifest": str(manifest),
        "canonical_paths": [str(canonical)],
        "run_mode": "FIXED_GROUPED_SPLIT",
        "result_stage": "HPO_VALIDATION_ONLY",
        "baseline_id": "method_repair_gpu_smoke",
        "model_kind": trainer.ROUTE2_EDIT_CENTERED_MODEL_KIND,
        "metadata_mode": "TRANSFERABLE_CONTEXT",
        "training_weighting_mode": "TASK_THEN_SOURCE_CONTEXT_ENDPOINT_GROUP",
        "target_scaling_mode": trainer.TARGET_SCALING_TRAIN_TASK_ROBUST,
        "target_scale_minimum_task_records": 2,
        "target_scale_floor": 1e-3,
        "candidate_control": "WITHIN_EXACT_SOURCE_TASK_TRAIN_CANDIDATE_PERMUTATION",
        "loss_kind": "huber_plus_pairwise",
        "ranking_loss_weight": 0.25,
        "checkpoint_selection": "BEST_VALIDATION",
        "checkpoint_metric": "TASK_MACRO_SPEARMAN_THEN_STANDARDIZED_MAE",
        "batch_size": 3,
        "seed": 19,
        "hidden_dim": 16,
        "depth": 1,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "epochs": 2,
        "num_workers": 0,
    }, output)
    assert summary["model_kind"] == trainer.ROUTE2_EDIT_CENTERED_MODEL_KIND
    assert summary["target_scaler"]["fit_scope"] == "TRAIN_ONLY"
    assert summary["target_scaler"]["training_record_count"] == 3
    assert summary["candidate_control_summary"]["changed_candidate_sequence_count"] == 3
    assert summary["checkpoint_selection"] == "BEST_VALIDATION"
    selected = torch.load(output / "delta_predictor_checkpoint.pt", map_location="cpu", weights_only=False)
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    assert selected["selection_provenance"]["selected_checkpoint"] == "best.pt"
    assert selected["completed_epoch"] == best["completed_epoch"]
    for name, value in selected["model_state"].items():
        torch.testing.assert_close(value, best["model_state"][name])
    prediction = json.loads((output / "validation_predictions.jsonl").read_text().splitlines()[0])
    assert prediction["target_scale_source"] == "TASK"
    assert "predicted_standardized_delta" in prediction


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
