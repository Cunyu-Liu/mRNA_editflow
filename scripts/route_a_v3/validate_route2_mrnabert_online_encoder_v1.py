#!/usr/bin/env python3
"""Validate online frozen mRNABERT against the canonical feature cache."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_gpu_failure_evidence import cuda_device_observation
from scripts.route_a_v3.route2_mrnabert_online_encoder_v1 import (
    FrozenMRNABERTOnlineEncoder,
)
from scripts.route_a_v3.train_route2_delta_predictor_v1 import (
    load_manifest,
    load_records,
    require_cuda,
)


class OnlineEncoderValidationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OnlineEncoderValidationError(message)


def _sample_indices(count: int, sample_count: int) -> list[int]:
    _require(count > 0 and sample_count > 0, "sample geometry must be positive")
    if count <= sample_count:
        return list(range(count))
    return sorted({
        round(index * (count - 1) / (sample_count - 1))
        for index in range(sample_count)
    })


def _novel_single_substitution(source: str, known: set[str]) -> str:
    bases = "ACGU"
    for position, base in enumerate(source):
        for replacement in bases:
            if replacement == base:
                continue
            candidate = source[:position] + replacement + source[position + 1 :]
            if candidate not in known:
                return candidate
    raise OnlineEncoderValidationError("could not construct a novel legal substitution")


def validate(
    config: Mapping[str, Any],
    *,
    encoder_class=FrozenMRNABERTOnlineEncoder,
) -> dict[str, Any]:
    device = require_cuda(str(config["device"]), int(config["physical_gpu_index"]))
    cache = torch.load(
        Path(config["pretrained_feature_cache_path"]),
        map_location="cpu",
        weights_only=False,
    )
    _require(
        cache.get("schema_version") == "route_a_v3_route2_frozen_pair_features.v1",
        "unexpected cache schema",
    )
    _require(
        cache.get("model_id") == config.get("model_id"),
        "online encoder and canonical cache model ids differ",
    )
    manifest = load_manifest(Path(config["development_manifest"]))
    records = load_records(
        [Path(path) for path in config["canonical_paths"]], manifest
    )
    ordered = sorted(records, key=lambda row: row.record_id)
    record_ids = [str(value) for value in cache["record_ids"]]
    _require(
        record_ids == [row.record_id for row in ordered],
        "cache and canonical record order differ",
    )
    _require(len(ordered) == int(config["expected_record_count"]), "record count differs")
    sample_indices = _sample_indices(len(ordered), int(config["sample_record_count"]))
    encoder = encoder_class(
        Path(config["mrnabert_model_path"]),
        device,
        maximum_chunk_nucleotides=int(config["maximum_chunk_nucleotides"]),
        maximum_sequences_per_batch=int(config["maximum_sequences_per_batch"]),
        batch_token_budget=int(config["batch_token_budget"]),
        attention_backend=str(
            config.get("attention_backend", "OFFICIAL_PYTORCH_FALLBACK")
        ),
    )
    sequences = []
    expected = []
    for index in sample_indices:
        sequences.extend((ordered[index].source, ordered[index].candidate))
        expected.extend(
            (cache["source_embeddings"][index], cache["candidate_embeddings"][index])
        )
    observed = encoder.encode_sequences(sequences).float()
    expected_tensor = torch.stack(expected).float()
    difference = (observed - expected_tensor).abs()
    maximum_difference = float(difference.max())
    mean_difference = float(difference.mean())
    tolerance = float(config["absolute_tolerance"])
    known_sequences = {
        sequence for row in ordered for sequence in (row.source, row.candidate)
    }
    novel = _novel_single_substitution(ordered[0].source, known_sequences)
    novel_embedding = encoder.encode_sequences([novel])
    novel_supported = (
        novel not in known_sequences
        and novel_embedding.shape == (1, int(cache["source_embeddings"].shape[1]))
        and torch.isfinite(novel_embedding).all().item()
    )
    passed = maximum_difference <= tolerance and novel_supported
    _require(math.isfinite(maximum_difference), "difference is nonfinite")
    return {
        "schema_version": "route_a_v3_route2_mrnabert_online_encoder_validation.v1",
        "status": (
            "ONLINE_FROZEN_MRNABERT_MATCHES_CANONICAL_CACHE"
            if passed
            else "ONLINE_FROZEN_MRNABERT_VALIDATION_FAILED"
        ),
        "sample_record_count": len(sample_indices),
        "compared_embedding_count": len(sequences),
        "embedding_width": int(observed.shape[1]),
        "maximum_absolute_difference": maximum_difference,
        "mean_absolute_difference": mean_difference,
        "absolute_tolerance": tolerance,
        "novel_candidate_encoding_supported": novel_supported,
        "frozen_parameter_count": int(encoder.parameter_count),
        "online_sequence_cache_count": int(encoder.cached_sequence_count),
        "attention_backend": encoder.attention_backend,
        "evaluation_records_read": 0,
        "sequence_payload_written": 0,
        "scientific_claim_status": "NOT_ESTABLISHED",
        **cuda_device_observation(
            int(config["physical_gpu_index"]), require_physical_index_match=True
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = Path(config["output_path"])
    _require(not output.exists(), f"online encoder validation output exists: {output}")
    result = validate(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ONLINE_FROZEN_MRNABERT_MATCHES_CANONICAL_CACHE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
