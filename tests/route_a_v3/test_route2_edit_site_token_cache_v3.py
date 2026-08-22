from __future__ import annotations

import torch

from core.route2_edit_site_token_cache_v3 import (
    assemble_edit_site_token_cache_v3,
    validate_edit_site_token_cache_v3,
)
from core.route2_mrnabert_edit_site_features_v3 import (
    ChunkSpan,
    EncodedSequenceFeaturesV3,
    PositionFeature,
)


def _encoded(value: float, positions: list[int]) -> EncodedSequenceFeaturesV3:
    def tensor(offset: float) -> torch.Tensor:
        return torch.tensor([value + offset, value - offset], dtype=torch.float32)

    return EncodedSequenceFeaturesV3(
        global_residual=tensor(0.0),
        positions={
            position: PositionFeature(
                site=tensor(float(position)),
                window_mean=tensor(float(position) + 0.1),
                window_max=tensor(float(position) + 0.2),
                chunk=ChunkSpan(0, 64),
            )
            for position in positions
        },
        local_chunk_spans=(ChunkSpan(0, 64),),
    )


def test_cache_uses_ragged_offsets_without_truncating_multi_edits() -> None:
    source = "A" * 64
    candidate = "C" * 38 + "A" * 26
    positions = list(range(38))
    rows = [
        {
            "canonical_record_id": "record-b",
            "source_sequence": source,
            "candidate_sequence": candidate,
            "source_relative_edits": [{"position": position} for position in positions],
        }
    ]
    payload = assemble_edit_site_token_cache_v3(
        rows,
        sequence_to_index={source: 0, candidate: 1},
        encoded={0: _encoded(1.0, positions), 1: _encoded(2.0, positions)},
        model_id="frozen-model",
        pretrained_parameter_count=100_000_001,
        attention_backend="OFFICIAL_PYTORCH_FALLBACK",
    )
    validate_edit_site_token_cache_v3(payload)
    assert payload["record_edit_offsets"].tolist() == [0, 38]
    assert payload["edit_positions"].tolist() == positions
    assert payload["position_site_hidden"].dtype == torch.float16
    assert payload["raw_sequence_payload_written"] == 0
    assert source not in repr(payload)
    assert candidate not in repr(payload)


def test_record_order_is_deterministic_and_features_are_shared() -> None:
    left = "AAAA"
    right = "ACAA"
    rows = [
        {"canonical_record_id": "z", "source_sequence": left, "candidate_sequence": right, "source_relative_edits": [{"position": 1}]},
        {"canonical_record_id": "a", "source_sequence": left, "candidate_sequence": right, "source_relative_edits": [{"position": 1}]},
    ]
    payload = assemble_edit_site_token_cache_v3(
        rows,
        sequence_to_index={left: 0, right: 1},
        encoded={0: _encoded(1.0, [1]), 1: _encoded(2.0, [1])},
        model_id="frozen-model",
        pretrained_parameter_count=100_000_001,
        attention_backend="OFFICIAL_PYTORCH_FALLBACK",
    )
    assert payload["record_ids"] == ["a", "z"]
    assert payload["position_site_hidden"].shape == (2, 2)
    assert payload["edit_source_feature_indices"].tolist() == [0, 0]
    assert payload["edit_candidate_feature_indices"].tolist() == [1, 1]
