from __future__ import annotations

import pytest
import torch

from core.route2_bottom_encoder_chunk_cache_v4 import (
    BottomEncodedChunkV4,
    BottomEncodedSequenceV4,
    assemble_frozen_bottom_encoder_chunk_cache_v4,
)
from core.route2_mrnabert_edit_site_features_v3 import ChunkSpan
from core.route2_xeditcritic_batch_v4 import (
    FrozenBottomEncoderChunkCacheViewV4,
    XEditCriticBatchV4Error,
    XEditCriticCollatorV4,
)


def _encoded(value: float) -> BottomEncodedSequenceV4:
    hidden = torch.arange(6 * 8, dtype=torch.float32).reshape(6, 8) + value
    return BottomEncodedSequenceV4(
        chunks=(
            BottomEncodedChunkV4(
                span=ChunkSpan(0, 4),
                hidden=hidden,
                attention_mask=torch.ones(6, dtype=torch.bool),
            ),
        ),
        global_residual=torch.full((8,), value),
    )


def _payload() -> dict:
    source = "AAAA"
    candidate = "ACAA"
    rows = [
        {
            "canonical_record_id": f"record-{index}",
            "split": "TRAIN",
            "source_sequence": source,
            "candidate_sequence": candidate,
            "source_relative_edits": [{"position": 1}],
        }
        for index in range(4)
    ]
    return assemble_frozen_bottom_encoder_chunk_cache_v4(
        rows,
        sequence_to_index={source: 0, candidate: 1},
        encoded={0: _encoded(1.0), 1: _encoded(2.0)},
        model_id="fixed",
        pretrained_parameter_count=113_389_056,
        attention_backend="PYTORCH_SDPA_AUTO",
    )


def _example(index: int, *, cache_record_id: str | None = None) -> dict:
    return {
        "record_id": f"recipient-{index}",
        "cache_record_id": cache_record_id or f"record-{index}",
        "source_group": f"group-{index}",
        "task": "task",
        "source": torch.tensor([0, 0, 0, 0]),
        "candidate": torch.tensor([0, 1, 0, 0]),
        "edits": ((1, "A", "C"),),
        "target": float(index),
        "scaled_target": float(index) / 2,
        "target_scale": 2.0,
        "sample_weight": 1.0,
        "study": 0,
        "assay": 1,
        "context": 2,
        "quantity": 1,
        "measurement": 1,
        "numerator": 0,
        "denominator": 0,
        "region": index % 2,
    }


def test_v4_collator_materializes_each_shared_chunk_once_and_aligns_ragged_edits() -> None:
    payload = _payload()
    view = FrozenBottomEncoderChunkCacheViewV4(
        payload, {f"record-{index}" for index in range(4)}
    )
    batch = XEditCriticCollatorV4(view)([_example(index) for index in range(4)])
    assert batch["source_tokens"].shape == (4, 4)
    assert batch["edit_padding_mask"].shape == (4, 1)
    assert batch["record_edit_offsets"].tolist() == [0, 1, 2, 3, 4]
    assert batch["cache_chunk_indices"].numel() == 2
    assert batch["chunk_hidden"].shape == (2, 6, 8)
    assert batch["edit_source_chunk_indices"].tolist() == [0, 0, 0, 0]
    assert batch["edit_candidate_chunk_indices"].tolist() == [1, 1, 1, 1]
    assert batch["record_source_global"].shape == (4, 8)
    assert batch["record_candidate_global"].shape == (4, 8)
    assert not any("window_mean" in key or "window_max" in key for key in batch)


def test_v4_collator_rejects_sub_four_physical_batches() -> None:
    payload = _payload()
    view = FrozenBottomEncoderChunkCacheViewV4(
        payload, {f"record-{index}" for index in range(4)}
    )
    collator = XEditCriticCollatorV4(view)
    with pytest.raises(XEditCriticBatchV4Error, match="sub-four"):
        collator([_example(index) for index in range(3)])


def test_v4_collator_uses_complete_permuted_donor_bundle_not_recipient_cache() -> None:
    payload = _payload()
    view = FrozenBottomEncoderChunkCacheViewV4(
        payload, {f"record-{index}" for index in range(4)}
    )
    examples = [
        _example(index, cache_record_id=f"record-{(index + 1) % 4}")
        for index in range(4)
    ]
    batch = XEditCriticCollatorV4(view)(examples)
    assert batch["cache_record_ids"] == ["record-1", "record-2", "record-3", "record-0"]
    assert batch["cache_record_indices"].tolist() == [1, 2, 3, 0]


def test_v4_cache_view_requires_exact_projection_coverage() -> None:
    with pytest.raises(XEditCriticBatchV4Error, match="exactly cover"):
        FrozenBottomEncoderChunkCacheViewV4(_payload(), {"record-0"})
