from __future__ import annotations

import math

import torch

from scripts.route_a_v3.train_route2_xeditcritic_v3 import (
    EditSiteCacheViewV3,
    TaskRobustScalerV3,
    XEditCriticCollatorV3,
    XEditCriticDatasetV3,
    fit_task_robust_scaler,
    study_source_group_weights,
    validation_metrics,
)
from core.route2_xeditcritic_training_data_v3 import XEditCriticRecordV3, build_vocabs


def _record(index: int, *, split: str = "TRAIN", group: str = "g") -> XEditCriticRecordV3:
    return XEditCriticRecordV3(
        record_id=f"r{index}", split=split, source="AAAA", candidate="ACAA",
        edits=((1, "A", "C"),), target=float(index + 1), task="t", study="s",
        source_group=group, assay="a", context="c", region=0, quantity="q",
        measurement="m", numerator="n", denominator="d",
    )


def _cache_payload() -> dict:
    return {
        "record_ids": ["r0", "r1"],
        "record_edit_offsets": torch.tensor([0, 1, 2]),
        "edit_positions": torch.tensor([1, 1]),
        "edit_source_feature_indices": torch.tensor([0, 0]),
        "edit_candidate_feature_indices": torch.tensor([1, 1]),
        "record_source_sequence_indices": torch.tensor([0, 0]),
        "record_candidate_sequence_indices": torch.tensor([1, 1]),
        "position_site_hidden": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "position_window_mean": torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
        "position_window_max": torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
        "global_residuals": torch.tensor([[7.0, 8.0], [9.0, 10.0]]),
        "embedding_width": 2,
    }


def test_collator_aligns_ragged_cache_bundle_and_explicit_edit_metadata() -> None:
    records = [_record(0), _record(1)]
    by_id = {record.record_id: record for record in records}
    vocabs = build_vocabs(records)
    scaler = fit_task_robust_scaler(records)
    cache = EditSiteCacheViewV3(_cache_payload(), set(by_id))
    dataset = XEditCriticDatasetV3(
        records, all_records=by_id, vocabs=vocabs, target_scaler=scaler, cache=cache
    )
    batch = XEditCriticCollatorV3(pretrained_width=2)([dataset[0], dataset[1]])
    assert batch["source_site"].shape == (2, 1, 2)
    assert batch["source_edit_base_ids"].tolist() == [[0], [0]]
    assert batch["candidate_edit_base_ids"].tolist() == [[1], [1]]
    assert batch["edit_positions"].tolist() == [[1], [1]]
    assert torch.allclose(
        batch["normalized_edit_positions"], torch.full((2, 1), 1 / 3)
    )
    assert batch["edit_padding_mask"].tolist() == [[False], [False]]


def test_candidate_override_moves_sequence_edit_and_cache_as_one_bundle() -> None:
    recipient = _record(0)
    donor = XEditCriticRecordV3(
        **{**recipient.__dict__, "record_id": "r1", "candidate": "AGAA", "edits": ((1, "A", "G"),)}
    )
    records = [recipient, donor]
    by_id = {record.record_id: record for record in records}
    dataset = XEditCriticDatasetV3(
        records,
        all_records=by_id,
        vocabs=build_vocabs(records),
        target_scaler=fit_task_robust_scaler(records),
        cache=EditSiteCacheViewV3(_cache_payload(), set(by_id)),
        candidate_bundle_overrides={"r0": "r1"},
    )
    example = dataset[0]
    assert example["candidate"].tolist() == [0, 2, 0, 0]
    assert example["edits"] == donor.edits
    assert example["feature_bundle"]["candidate_site"].tolist() == [[3.0, 4.0]]


def test_study_source_group_weights_do_not_scale_with_dense_rows() -> None:
    records = [_record(index, group="dense") for index in range(10)] + [_record(10, group="sparse")]
    weights = study_source_group_weights(records)
    assert math.isclose(
        sum(weights[f"r{index}"] for index in range(10)), weights["r10"]
    )


def test_validation_metrics_are_task_macro_not_dense_row_weighted() -> None:
    metrics = validation_metrics(
        targets=[0, 1, 2, 0, 1, 2],
        predictions=[0, 1, 2, 2, 1, 0],
        scaled_targets=[0, 1, 2, 0, 1, 2],
        scaled_predictions=[0, 1, 2, 2, 1, 0],
        tasks=["a", "a", "a", "b", "b", "b"],
    )
    assert metrics["task_macro_spearman"] == 0.0
    assert metrics["positive_task_count"] == 1


def test_target_scaler_is_zero_anchored_and_train_only_serializable() -> None:
    scaler = fit_task_robust_scaler([_record(0), _record(1)])
    payload = scaler.to_dict()
    assert payload["fit_scope"] == "TRAIN_ONLY"
    assert payload["center_subtracted"] is False
    assert scaler.scale("t", 0) > 0
    assert scaler.scale("unseen_validation_task", 0) == scaler.region_scales[0]
