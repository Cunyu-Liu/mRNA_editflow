from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts/route_a_v3/adjudicate_route2_mrnabert_sdpa_backend_v1.py"
SPEC = importlib.util.spec_from_file_location("sdpa_adjudication", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def validation(backend: str, throughput: float) -> dict:
    return {
        "schema_version": "route_a_v3_route2_mrnabert_online_encoder_validation.v1",
        "status": "ONLINE_FROZEN_MRNABERT_MATCHES_CANONICAL_CACHE",
        "attention_backend": backend,
        "model_id": "mRNABERT",
        "sample_record_count": 32,
        "compared_embedding_count": 64,
        "embedding_width": 768,
        "maximum_sequences_per_batch": 8,
        "batch_token_budget": 4096,
        "throughput_repetitions": 3,
        "median_encoded_sequences_per_second": throughput,
        "maximum_absolute_difference": 0.005,
        "absolute_tolerance": 0.01,
        "novel_candidate_encoding_supported": True,
        "evaluation_records_read": 0,
    }


def screen(decision: str = "ELIGIBLE_FOR_FULL_ENCODER_CACHE_ALIGNMENT_BENCHMARK") -> dict:
    return {"summary": {"decision": decision}}


def test_selects_sdpa_only_after_alignment_and_material_speedup() -> None:
    result = MODULE.adjudicate(
        screen(),
        validation("OFFICIAL_PYTORCH_FALLBACK", 100.0),
        validation("PYTORCH_SDPA_AUTO", 125.0),
        minimum_speedup=1.1,
    )
    assert result["selected_attention_backend"] == "PYTORCH_SDPA_AUTO"
    assert result["formal_encoder_backend_changed"] is True


@pytest.mark.parametrize("failure", ["primitive", "alignment", "speed"])
def test_keeps_official_backend_when_any_required_gate_fails(failure: str) -> None:
    attention = screen()
    official = validation("OFFICIAL_PYTORCH_FALLBACK", 100.0)
    candidate = validation("PYTORCH_SDPA_AUTO", 120.0)
    if failure == "primitive":
        attention = screen("KEEP_OFFICIAL_PYTORCH_FALLBACK")
    elif failure == "alignment":
        candidate["status"] = "ONLINE_FROZEN_MRNABERT_VALIDATION_FAILED"
    else:
        candidate["median_encoded_sequences_per_second"] = 105.0
    result = MODULE.adjudicate(
        attention, official, candidate, minimum_speedup=1.1
    )
    assert result["selected_attention_backend"] == "OFFICIAL_PYTORCH_FALLBACK"
    assert result["formal_encoder_backend_changed"] is False


def test_rejects_nonmatched_full_encoder_runs() -> None:
    official = validation("OFFICIAL_PYTORCH_FALLBACK", 100.0)
    candidate = validation("PYTORCH_SDPA_AUTO", 120.0)
    candidate["sample_record_count"] = 31
    with pytest.raises(MODULE.BackendAdjudicationError, match="sample_record_count"):
        MODULE.adjudicate(screen(), official, candidate, minimum_speedup=1.1)
