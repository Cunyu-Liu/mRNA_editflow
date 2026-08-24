from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from core.route2_bottom_encoder_chunk_cache_v4 import (
    BottomEncodedChunkV4,
    BottomEncodedSequenceV4,
    FrozenBottomEncoderChunkCacheV4Error,
    assemble_frozen_bottom_encoder_chunk_cache_v4,
    materialize_bottom_chunk_batch_v4,
    validate_frozen_bottom_encoder_chunk_cache_v4,
)
from core.route2_mrnabert_edit_site_features_v3 import (
    ChunkSpan,
    overlapping_chunk_spans,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/route_a_v3_route2_xeditcritic_v4_bottom_six_cache_v1.json"


def _encoded(sequence: str, value: float, width: int = 4) -> BottomEncodedSequenceV4:
    chunks = []
    for span_index, span in enumerate(overlapping_chunk_spans(len(sequence))):
        token_count = span.length + 2
        hidden = torch.arange(token_count * width, dtype=torch.float32).reshape(token_count, width)
        hidden = hidden / 1000 + value + span_index
        chunks.append(
            BottomEncodedChunkV4(
                span=span,
                hidden=hidden,
                attention_mask=torch.ones(token_count, dtype=torch.long),
            )
        )
    return BottomEncodedSequenceV4(
        chunks=tuple(chunks),
        global_residual=torch.full((width,), value),
    )


def _row(record_id: str, source: str, candidate: str, positions: list[int], split: str = "TRAIN") -> dict:
    return {
        "canonical_record_id": record_id,
        "split": split,
        "source_sequence": source,
        "candidate_sequence": candidate,
        "source_relative_edits": [{"position": position} for position in positions],
    }


def test_v4_cache_config_is_projection_only_and_writes_only_under_large_artifact_root() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"].startswith("FROZEN_BEFORE_V4_")
    assert config["allowed_splits"] == ["TRAIN", "VALIDATION"]
    assert all("development_train_validation" in path for path in config["projection_paths"])
    assert config["frozen_encoder_blocks"] == [0, 1, 2, 3, 4, 5]
    assert config["trainable_encoder_blocks"] == [6, 7, 8, 9, 10, 11]
    assert (config["chunk_nucleotides"], config["chunk_overlap"], config["local_context_radius"]) == (1000, 64, 32)
    assert config["special_token_offset"] == 1
    assert config["attention_backend"] == "PYTORCH_SDPA_AUTO"
    assert config["raw_sequence_payload_written"] == 0
    assert config["label_or_outcome_payload_written"] == 0
    assert config["development_test_record_count"] == 0
    assert config["evaluation_record_count"] == 0
    artifact_root = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"
    assert config["output_path"].startswith(artifact_root)
    assert config["summary_path"].startswith(artifact_root)


def test_v4_cache_keeps_ragged_edits_and_raw_outcome_payloads_absent() -> None:
    source = "A" * 80
    candidate = "C" * 38 + "A" * 42
    positions = list(range(38))
    payload = assemble_frozen_bottom_encoder_chunk_cache_v4(
        [_row("record", source, candidate, positions)],
        sequence_to_index={source: 0, candidate: 1},
        encoded={0: _encoded(source, 1.0), 1: _encoded(candidate, 2.0)},
        model_id="fixed-mrnabert",
        pretrained_parameter_count=113_389_056,
        attention_backend="PYTORCH_SDPA_AUTO",
    )
    validate_frozen_bottom_encoder_chunk_cache_v4(payload)
    assert payload["record_edit_offsets"].tolist() == [0, 38]
    assert payload["edit_positions"].tolist() == positions
    assert payload["token_hidden"].dtype == torch.float16
    assert payload["raw_sequence_payload_written"] == 0
    assert payload["label_or_outcome_payload_written"] == 0
    assert source not in repr(payload)
    assert candidate not in repr(payload)
    assert "direction_normalized_delta" not in repr(payload)


def test_v4_cache_rejects_development_test_before_any_tensor_is_assembled() -> None:
    source = "AAAA"
    candidate = "ACAA"
    with pytest.raises(FrozenBottomEncoderChunkCacheV4Error, match="TRAIN/VALIDATION"):
        assemble_frozen_bottom_encoder_chunk_cache_v4(
            [_row("test-record", source, candidate, [1], split="TEST")],
            sequence_to_index={source: 0, candidate: 1},
            encoded={0: _encoded(source, 1.0), 1: _encoded(candidate, 2.0)},
            model_id="fixed-mrnabert",
            pretrained_parameter_count=113_389_056,
            attention_backend="PYTORCH_SDPA_AUTO",
        )


def test_radius_32_uses_the_most_centered_chunk_and_never_includes_specials() -> None:
    source = "A" * 1500
    candidate = list(source)
    candidate[549] = "C"
    candidate[950] = "G"
    candidate = "".join(candidate)
    payload = assemble_frozen_bottom_encoder_chunk_cache_v4(
        [_row("long", source, candidate, [549, 950], split="VALIDATION")],
        sequence_to_index={source: 0, candidate: 1},
        encoded={0: _encoded(source, 1.0), 1: _encoded(candidate, 2.0)},
        model_id="fixed-mrnabert",
        pretrained_parameter_count=113_389_056,
        attention_backend="PYTORCH_SDPA_AUTO",
    )
    spans = overlapping_chunk_spans(1500)
    assert spans == [ChunkSpan(0, 1000), ChunkSpan(500, 1500)]
    # Position 549 is closer to the first chunk center; 950 is closer to the second.
    assert payload["edit_source_chunk_indices"].tolist() == [0, 1]
    assert payload["edit_source_token_centers"].tolist() == [550, 451]
    assert payload["edit_source_window_starts"].tolist() == [518, 419]
    assert payload["edit_source_window_ends"].tolist() == [583, 484]


def test_physical_batch_materialization_deduplicates_chunks_and_preserves_records() -> None:
    source = "A" * 64
    candidate = "A" * 10 + "C" + "A" * 53
    rows = [
        _row("b", source, candidate, [10]),
        _row("a", source, candidate, [10]),
    ]
    payload = assemble_frozen_bottom_encoder_chunk_cache_v4(
        rows,
        sequence_to_index={source: 0, candidate: 1},
        encoded={0: _encoded(source, 1.0), 1: _encoded(candidate, 2.0)},
        model_id="fixed-mrnabert",
        pretrained_parameter_count=113_389_056,
        attention_backend="PYTORCH_SDPA_AUTO",
    )
    batch = materialize_bottom_chunk_batch_v4(payload, [0, 1])
    assert batch["cache_chunk_indices"].numel() == 2
    assert batch["chunk_hidden"].shape[:2] == (2, 66)
    assert batch["record_edit_offsets"].tolist() == [0, 1, 2]
    assert batch["edit_source_chunk_indices"].tolist() == [0, 0]
    assert batch["edit_candidate_chunk_indices"].tolist() == [1, 1]
    assert batch["record_source_global"].shape == (2, 4)
    assert batch["record_candidate_global"].shape == (2, 4)


def test_physical_batch_allows_repeated_rows_but_loads_each_chunk_once() -> None:
    source = "A" * 64
    candidate = "A" * 10 + "C" + "A" * 53
    payload = assemble_frozen_bottom_encoder_chunk_cache_v4(
        [_row("repeated", source, candidate, [10])],
        sequence_to_index={source: 0, candidate: 1},
        encoded={0: _encoded(source, 1.0), 1: _encoded(candidate, 2.0)},
        model_id="fixed-mrnabert",
        pretrained_parameter_count=113_389_056,
        attention_backend="PYTORCH_SDPA_AUTO",
    )
    batch = materialize_bottom_chunk_batch_v4(payload, [0, 0, 0, 0])
    assert batch["cache_record_indices"].tolist() == [0, 0, 0, 0]
    assert batch["cache_chunk_indices"].numel() == 2
    assert batch["record_edit_offsets"].tolist() == [0, 1, 2, 3, 4]
    assert batch["edit_source_chunk_indices"].tolist() == [0, 0, 0, 0]
    assert batch["edit_candidate_chunk_indices"].tolist() == [1, 1, 1, 1]


def test_materialized_records_do_not_cross_ragged_edit_boundaries() -> None:
    source = "A" * 96
    candidate_a = "C" * 3 + "A" * 93
    candidate_b = "A" * 70 + "G" * 5 + "A" * 21
    payload = assemble_frozen_bottom_encoder_chunk_cache_v4(
        [
            _row("a", source, candidate_a, [0, 1, 2]),
            _row("b", source, candidate_b, [70, 71, 72, 73, 74]),
        ],
        sequence_to_index={source: 0, candidate_a: 1, candidate_b: 2},
        encoded={
            0: _encoded(source, 1.0),
            1: _encoded(candidate_a, 2.0),
            2: _encoded(candidate_b, 3.0),
        },
        model_id="fixed-mrnabert",
        pretrained_parameter_count=113_389_056,
        attention_backend="PYTORCH_SDPA_AUTO",
    )
    batch = materialize_bottom_chunk_batch_v4(payload, [1])
    assert batch["record_edit_offsets"].tolist() == [0, 5]
    assert batch["edit_positions"].tolist() == [70, 71, 72, 73, 74]
    assert batch["record_source_global"].tolist() == [[1.0, 1.0, 1.0, 1.0]]
    assert batch["record_candidate_global"].tolist() == [[3.0, 3.0, 3.0, 3.0]]
