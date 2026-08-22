from __future__ import annotations

import torch

from core.route2_legal_xeditflow import initial_state
from core.route2_source_token_cache_v3 import SourceTokenCacheIndexV3, assemble_source_token_cache_v3
from core.route2_xeditsetflow_sampling_v3 import (
    SetFlowGenerationMetadataV3,
    build_generation_metadata_v3,
    collate_generation_states_v3,
)


def test_generation_collator_aligns_cache_current_state_and_endpoint() -> None:
    source = "ACGU"
    cache = SourceTokenCacheIndexV3(assemble_source_token_cache_v3(
        [{"canonical_record_id": "r", "source_sequence": source}],
        sequence_to_index={source: 0},
        encoded_tokens={0: torch.arange(12, dtype=torch.float32).reshape(4, 3)},
        model_id="frozen", pretrained_parameter_count=100_000_001,
        attention_backend="OFFICIAL_PYTORCH_FALLBACK",
    ))
    meta = SetFlowGenerationMetadataV3("r", 1, 2, 3, 4, 5, 6, 1)
    batch = collate_generation_states_v3(
        [initial_state(source, budget=3, assay_id="a", context_id="c")],
        [meta],
        source_cache=cache,
    )
    assert batch["source_tokens"].tolist() == [[0, 1, 2, 3]]
    assert torch.equal(batch["source_tokens"], batch["current_tokens"])
    assert batch["source_pretrained_tokens"].shape == (1, 4, 3)
    assert batch["remaining_budget"].tolist() == [3]
    assert batch["measurement_ids"].tolist() == [2]


def test_generation_metadata_uses_descriptors_without_outcome_access() -> None:
    class Protected(dict):
        def __getitem__(self, key):
            if key == "direction_normalized_delta":
                raise AssertionError("outcome accessed")
            return super().__getitem__(key)

    source = {
        "source_sequence": "AAAA", "endpoint_id": "e", "biological_context_id": "c",
        "assay_id": "a", "region": "5UTR",
    }
    projection = Protected(
        split="VALIDATION", source_sequence="AAAA", endpoint_id="e",
        biological_context_id="c", assay_id="a", canonical_record_id="r",
        source_relative_edits=[{"position": 1}], direction_normalized_delta=object(),
        endpoint_descriptor={
            "quantity_family": "q", "measurement_form": "m",
            "numerator_family": None, "denominator_family": None,
        },
    )
    vocabs = {
        "quantity": {"__UNK__": 0, "q": 1}, "measurement": {"__UNK__": 0, "m": 1},
        "numerator": {"__UNK__": 0, "__NONE__": 1}, "denominator": {"__UNK__": 0, "__NONE__": 1},
        "assay": {"__UNK__": 0, "a": 1}, "context": {"__UNK__": 0, "c": 1},
    }
    result = build_generation_metadata_v3([source], [projection], vocabs)
    assert result == [SetFlowGenerationMetadataV3("r", 1, 1, 1, 1, 1, 1, 0)]
