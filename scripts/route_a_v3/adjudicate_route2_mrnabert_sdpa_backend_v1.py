#!/usr/bin/env python3
"""Choose the online mRNABERT attention backend after full-encoder validation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


class BackendAdjudicationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BackendAdjudicationError(message)


def finite(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    require(math.isfinite(result), f"{label} is nonfinite")
    return result


def adjudicate(
    attention_screen: Mapping[str, Any],
    official: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    *,
    minimum_speedup: float,
) -> dict[str, Any]:
    threshold = finite(minimum_speedup, "minimum speedup")
    require(threshold >= 1.0, "minimum speedup must be at least one")
    require(
        attention_screen.get("summary", {}).get("decision")
        in {
            "ELIGIBLE_FOR_FULL_ENCODER_CACHE_ALIGNMENT_BENCHMARK",
            "KEEP_OFFICIAL_PYTORCH_FALLBACK",
        },
        "attention screening decision is absent",
    )
    for value, backend in ((official, "OFFICIAL_PYTORCH_FALLBACK"),):
        require(
            value.get("schema_version")
            == "route_a_v3_route2_mrnabert_online_encoder_validation.v1",
            "full-encoder validation schema differs",
        )
        require(value.get("attention_backend") == backend, "attention backend differs")
        require(value.get("evaluation_records_read") == 0, "Evaluation entered backend selection")
    primitive_passed = (
        attention_screen["summary"]["decision"]
        == "ELIGIBLE_FOR_FULL_ENCODER_CACHE_ALIGNMENT_BENCHMARK"
    )
    if not primitive_passed:
        return {
            "schema_version": "route_a_v3_route2_mrnabert_sdpa_backend_adjudication.v1",
            "status": "ONLINE_ENCODER_BACKEND_ADJUDICATED",
            "selected_attention_backend": "OFFICIAL_PYTORCH_FALLBACK",
            "primitive_attention_equivalence_passed": False,
            "full_encoder_cache_alignment_passed": False,
            "minimum_speedup_required": threshold,
            "observed_full_encoder_speedup": None,
            "speedup_gate_passed": False,
            "formal_encoder_backend_changed": False,
            "evaluation_opened": False,
            "scientific_claim_status": "NOT_ESTABLISHED",
        }
    require(candidate is not None, "SDPA candidate validation is required after primitive PASS")
    require(
        candidate.get("schema_version")
        == "route_a_v3_route2_mrnabert_online_encoder_validation.v1",
        "full-encoder validation schema differs",
    )
    require(candidate.get("attention_backend") == "PYTORCH_SDPA_AUTO", "attention backend differs")
    require(candidate.get("evaluation_records_read") == 0, "Evaluation entered backend selection")
    for field in (
        "model_id",
        "sample_record_count",
        "compared_embedding_count",
        "embedding_width",
        "maximum_sequences_per_batch",
        "batch_token_budget",
        "throughput_repetitions",
    ):
        require(official.get(field) == candidate.get(field), f"backend runs differ in {field}")
    official_throughput = finite(
        official.get("median_encoded_sequences_per_second"), "official throughput"
    )
    candidate_throughput = finite(
        candidate.get("median_encoded_sequences_per_second"), "candidate throughput"
    )
    require(official_throughput > 0 and candidate_throughput > 0, "throughput must be positive")
    speedup = candidate_throughput / official_throughput
    full_encoder_aligned = (
        candidate.get("status")
        == "ONLINE_FROZEN_MRNABERT_MATCHES_CANONICAL_CACHE"
        and finite(candidate.get("maximum_absolute_difference"), "candidate maximum difference")
        <= finite(candidate.get("absolute_tolerance"), "candidate tolerance")
        and candidate.get("novel_candidate_encoding_supported") is True
    )
    speedup_passed = speedup >= threshold
    use_sdpa = primitive_passed and full_encoder_aligned and speedup_passed
    return {
        "schema_version": "route_a_v3_route2_mrnabert_sdpa_backend_adjudication.v1",
        "status": "ONLINE_ENCODER_BACKEND_ADJUDICATED",
        "selected_attention_backend": (
            "PYTORCH_SDPA_AUTO" if use_sdpa else "OFFICIAL_PYTORCH_FALLBACK"
        ),
        "primitive_attention_equivalence_passed": primitive_passed,
        "full_encoder_cache_alignment_passed": full_encoder_aligned,
        "minimum_speedup_required": threshold,
        "observed_full_encoder_speedup": speedup,
        "speedup_gate_passed": speedup_passed,
        "formal_encoder_backend_changed": use_sdpa,
        "evaluation_opened": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-screen", type=Path, required=True)
    parser.add_argument("--official-validation", type=Path, required=True)
    parser.add_argument("--candidate-validation", type=Path)
    parser.add_argument("--minimum-speedup", type=float, default=1.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"backend adjudication output exists: {args.output}")
    result = adjudicate(
        json.loads(args.attention_screen.read_text(encoding="utf-8")),
        json.loads(args.official_validation.read_text(encoding="utf-8")),
        (
            json.loads(args.candidate_validation.read_text(encoding="utf-8"))
            if args.candidate_validation is not None
            else None
        ),
        minimum_speedup=args.minimum_speedup,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
