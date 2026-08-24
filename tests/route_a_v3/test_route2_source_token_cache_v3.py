from __future__ import annotations

import copy

import pytest
import torch

from core.route2_mrnabert_edit_site_features_v3 import extract_nucleotide_token_hidden
from core.route2_source_token_cache_v3 import (
    SourceTokenCacheV3Error,
    assemble_source_token_cache_v3,
    require_source_token_cache_identity_v3,
)


def test_nucleotide_token_extraction_excludes_both_special_tokens() -> None:
    hidden = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    mask = torch.ones(6, dtype=torch.long)
    observed = extract_nucleotide_token_hidden(hidden, mask, chunk_length=4)
    assert torch.equal(observed, hidden[1:5])


def test_source_cache_is_ragged_shared_float16_and_has_no_raw_sequence() -> None:
    first = "AAAA"
    second = "ACGUAC"
    rows = [
        {"canonical_record_id": "z", "source_sequence": first},
        {"canonical_record_id": "a", "source_sequence": second},
        {"canonical_record_id": "b", "source_sequence": first},
    ]
    payload = assemble_source_token_cache_v3(
        rows,
        sequence_to_index={first: 0, second: 1},
        encoded_tokens={0: torch.ones(4, 3), 1: torch.ones(6, 3) * 2},
        model_id="frozen-model",
        pretrained_parameter_count=100_000_001,
        attention_backend="OFFICIAL_PYTORCH_FALLBACK",
    )
    assert payload["record_ids"] == ["a", "b", "z"]
    assert payload["record_source_sequence_indices"].tolist() == [1, 0, 0]
    assert payload["sequence_token_offsets"].tolist() == [0, 4, 10]
    assert payload["source_token_hidden"].shape == (10, 3)
    assert payload["source_token_hidden"].dtype == torch.float16
    assert payload["raw_sequence_payload_written"] == 0
    assert first not in repr(payload) and second not in repr(payload)

    identity = require_source_token_cache_identity_v3(
        payload,
        expected_model_id="frozen-model",
        expected_record_count=3,
        expected_unique_source_count=2,
        expected_token_count=10,
        expected_maximum_source_length=6,
        expected_embedding_width=3,
    )
    assert identity["unique_source_count"] == 2
    assert identity["token_count"] == 10

    wrong_model = copy.deepcopy(payload)
    wrong_model["model_id"] = "another-model"
    with pytest.raises(SourceTokenCacheV3Error, match="revision"):
        require_source_token_cache_identity_v3(
            wrong_model,
            expected_model_id="frozen-model",
            expected_record_count=3,
            expected_unique_source_count=2,
            expected_token_count=10,
            expected_maximum_source_length=6,
            expected_embedding_width=3,
        )
