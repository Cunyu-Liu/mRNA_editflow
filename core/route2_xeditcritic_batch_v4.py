"""Projection-record and bottom-six-cache batching for XEditCritic V4."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from core.route2_bottom_encoder_chunk_cache_v4 import (
    materialize_bottom_chunk_batch_v4,
    validate_frozen_bottom_encoder_chunk_cache_v4,
)
from core.route2_xeditcritic_training_data_v3 import PAD_TOKEN, RNA_TOKEN
from core.route2_xedit_v4_interfaces import CriticStateBatchV4
from scripts.route_a_v3.train_route2_xeditcritic_v3 import XEditCriticDatasetV3


class XEditCriticBatchV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticBatchV4Error(message)


class FrozenBottomEncoderChunkCacheViewV4:
    """Record-id view over a validated outcome-free V4 cache."""

    def __init__(
        self,
        payload: Mapping[str, Any],
        expected_record_ids: set[str],
        *,
        validate_payload: bool = True,
    ) -> None:
        if validate_payload:
            validate_frozen_bottom_encoder_chunk_cache_v4(payload)
        record_ids = [str(value) for value in payload["record_ids"]]
        _require(set(record_ids) == expected_record_ids, "V4 bottom-six cache does not exactly cover the projection")
        self.payload = payload
        self.record_index = {
            record_id: index for index, record_id in enumerate(record_ids)
        }
        self.width = int(payload["embedding_width"])

    def materialize(self, record_ids: Sequence[str]) -> dict[str, torch.Tensor]:
        _require(bool(record_ids), "V4 cache batch record ids are empty")
        try:
            indices = [self.record_index[str(record_id)] for record_id in record_ids]
        except KeyError as exc:
            raise XEditCriticBatchV4Error("V4 cache donor record is absent") from exc
        # The immutable payload was validated when this view was constructed.
        # Repeating the full 9.65 GB tensor/ragged-reference scan for every
        # training batch dominated the formal Critic runtime.  Materialization
        # still performs all batch-local index and mapping checks below.
        return materialize_bottom_chunk_batch_v4(
            self.payload,
            indices,
            validate_payload=False,
        )


class XEditCriticDatasetV4(XEditCriticDatasetV3):
    """V3 projection semantics plus an explicit V4 cache donor identity."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _require(kwargs.get("cache") is None, "V4 dataset cannot receive the V3 pooled-feature cache")
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        result = super().__getitem__(index)
        record = self.records[index]
        donor_id = self.overrides.get(record.record_id, record.record_id)
        result["cache_record_id"] = donor_id
        return result


class XEditCriticCollatorV4:
    """Collate raw records and deduplicated bottom-six chunks for one physical batch."""

    def __init__(
        self,
        cache: FrozenBottomEncoderChunkCacheViewV4,
        *,
        minimum_physical_batch: int = 4,
    ) -> None:
        _require(minimum_physical_batch >= 4, "V4 collator minimum physical batch must be at least four")
        self.cache = cache
        self.minimum_physical_batch = int(minimum_physical_batch)

    def __call__(self, examples: list[dict[str, Any]]) -> CriticStateBatchV4:
        batch_size = len(examples)
        _require(batch_size >= self.minimum_physical_batch, "V4 collator received a sub-four physical batch")
        maximum_length = max(len(example["source"]) for example in examples)
        maximum_edits = max(1, max(len(example["edits"]) for example in examples))
        source = torch.full((batch_size, maximum_length), PAD_TOKEN, dtype=torch.long)
        candidate = torch.full_like(source, PAD_TOKEN)
        padding_mask = torch.ones_like(source, dtype=torch.bool)
        edit_padding_mask = torch.ones((batch_size, maximum_edits), dtype=torch.bool)
        source_edit_base_ids = torch.full((batch_size, maximum_edits), PAD_TOKEN, dtype=torch.long)
        candidate_edit_base_ids = torch.full_like(source_edit_base_ids, PAD_TOKEN)
        normalized_positions = torch.zeros((batch_size, maximum_edits), dtype=torch.float32)
        edit_positions = torch.zeros((batch_size, maximum_edits), dtype=torch.long)
        flattened_positions: list[int] = []
        for batch_index, example in enumerate(examples):
            length = len(example["source"])
            _require(len(example["candidate"]) == length, "length-changing record entered the V4 collator")
            source[batch_index, :length] = example["source"]
            candidate[batch_index, :length] = example["candidate"]
            padding_mask[batch_index, :length] = False
            edits = example["edits"]
            denominator = max(1, length - 1)
            if edits:
                edit_padding_mask[batch_index, : len(edits)] = False
            for edit_index, (position, source_base, candidate_base) in enumerate(edits):
                _require(0 <= int(position) < length, "V4 edit position is outside the sequence")
                source_edit_base_ids[batch_index, edit_index] = RNA_TOKEN[source_base]
                candidate_edit_base_ids[batch_index, edit_index] = RNA_TOKEN[candidate_base]
                edit_positions[batch_index, edit_index] = int(position)
                normalized_positions[batch_index, edit_index] = int(position) / denominator
                flattened_positions.append(int(position))
        cache_batch = self.cache.materialize(
            [str(example["cache_record_id"]) for example in examples]
        )
        _require(cache_batch["edit_positions"].tolist() == flattened_positions, "V4 cache edit mapping differs from the complete candidate bundle")
        result: CriticStateBatchV4 = {
            "record_ids": [str(example["record_id"]) for example in examples],
            "cache_record_ids": [str(example["cache_record_id"]) for example in examples],
            "source_groups": [str(example["source_group"]) for example in examples],
            "task_ids": [str(example["task"]) for example in examples],
            "source_tokens": source,
            "candidate_tokens": candidate,
            "padding_mask": padding_mask,
            "edit_padding_mask": edit_padding_mask,
            "source_edit_base_ids": source_edit_base_ids,
            "candidate_edit_base_ids": candidate_edit_base_ids,
            "normalized_edit_positions": normalized_positions,
            "edit_positions": edit_positions,
            "study_ids": torch.tensor([example["study"] for example in examples], dtype=torch.long),
            "assay_ids": torch.tensor([example["assay"] for example in examples], dtype=torch.long),
            "context_ids": torch.tensor([example["context"] for example in examples], dtype=torch.long),
            "quantity_ids": torch.tensor([example["quantity"] for example in examples], dtype=torch.long),
            "measurement_ids": torch.tensor([example["measurement"] for example in examples], dtype=torch.long),
            "numerator_ids": torch.tensor([example["numerator"] for example in examples], dtype=torch.long),
            "denominator_ids": torch.tensor([example["denominator"] for example in examples], dtype=torch.long),
            "region_ids": torch.tensor([example["region"] for example in examples], dtype=torch.long),
            "target": torch.tensor([example["target"] for example in examples], dtype=torch.float32),
            "scaled_target": torch.tensor([example["scaled_target"] for example in examples], dtype=torch.float32),
            "target_scale": torch.tensor([example["target_scale"] for example in examples], dtype=torch.float32),
            "sample_weight": torch.tensor([example["sample_weight"] for example in examples], dtype=torch.float32),
        }
        result.update(cache_batch)
        # Cache tensors retain float16 storage until CUDA autocast.  Labels and
        # endpoint ids never enter the cache artifact.
        return result
