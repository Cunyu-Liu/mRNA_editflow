#!/usr/bin/env python3
"""Build record-aligned frozen mRNABERT features for Route 2 Development."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.route_a_v3.train_route2_delta_predictor_v1 import (
    DeltaTrainingError,
    load_manifest,
    load_records,
    require_cuda,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DeltaTrainingError(message)


def _batches(
    chunks: list[tuple[str, str]], maximum_sequences: int, token_budget: int
) -> Iterable[list[tuple[str, str]]]:
    """Group similarly sized chunks under an approximate padded-token budget."""

    ordered = sorted(chunks, key=lambda value: (len(value[1]), value[0]))
    current: list[tuple[str, str]] = []
    longest = 0
    for item in ordered:
        proposed_longest = max(longest, len(item[1]) + 2)
        if current and (
            len(current) >= maximum_sequences
            or proposed_longest * (len(current) + 1) > token_budget
        ):
            yield current
            current = []
            longest = 0
        current.append(item)
        longest = max(longest, len(item[1]) + 2)
    if current:
        yield current


def _sequence_chunks(sequence: str, maximum_nucleotides: int) -> list[str]:
    return [
        sequence[start : start + maximum_nucleotides]
        for start in range(0, len(sequence), maximum_nucleotides)
    ]


def _format_utr_chunk(chunk: str) -> str:
    """Apply the official mRNABERT UTR single-letter tokenization."""

    dna = chunk.upper().replace("U", "T")
    _require(bool(dna) and set(dna) <= set("ACGTN"), "mRNABERT UTR chunk has an unsupported base")
    return " ".join(dna)


def _pool_last_hidden(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Match the official attention-masked mean-pooling example."""

    weights = attention_mask.to(last_hidden_state.dtype).unsqueeze(-1)
    return (last_hidden_state * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


def build(config: Mapping[str, Any]) -> dict[str, Any]:
    output_path = Path(config["output_path"])
    summary_path = Path(config["summary_path"])
    _require(not output_path.exists(), f"feature cache already exists: {output_path}")
    _require(not summary_path.exists(), f"feature summary already exists: {summary_path}")
    device = require_cuda(str(config["device"]), int(config["physical_gpu_index"]))
    manifest = load_manifest(Path(config["development_manifest"]))
    records = load_records(
        [Path(path) for path in config["canonical_paths"]], manifest
    )
    _require(
        len(records) == int(config["expected_record_count"]),
        "Development row count changed",
    )

    try:
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except ImportError as exc:
        raise DeltaTrainingError("Transformers is unavailable for mRNABERT") from exc
    model_path = Path(config["mrnabert_model_path"])
    _require(model_path.is_dir(), "mRNABERT model directory is absent")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model_config = AutoConfig.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModel.from_config(
        model_config, trust_remote_code=True, add_pooling_layer=False
    )
    # The official modeling file already implements this equivalent PyTorch
    # attention fallback.  Select it explicitly because its bundled Triton
    # kernel predates the server's current Triton API.
    modeling_module = sys.modules[model.__class__.__module__]
    modeling_module.flash_attn_qkvpacked_func = None
    checkpoint = torch.load(
        model_path / "pytorch_model.bin", map_location="cpu", weights_only=False
    )
    base_state = {
        key.removeprefix("bert."): value
        for key, value in checkpoint.items()
        if key.startswith("bert.")
    }
    _require(bool(base_state), "mRNABERT base encoder weights are absent")
    model.load_state_dict(base_state, strict=True)
    del checkpoint, base_state
    model = model.to(device).eval()
    model.requires_grad_(False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    _require(parameter_count > 100_000_000, "mRNABERT parameter geometry changed")
    hidden_size = int(model.config.hidden_size)
    _require(hidden_size == 768, "mRNABERT hidden width changed")

    maximum_nucleotides = int(config["maximum_chunk_nucleotides"])
    _require(
        maximum_nucleotides + 2 <= int(tokenizer.model_max_length),
        "mRNABERT chunk exceeds tokenizer maximum length",
    )
    sequences = sorted(
        {sequence for record in records for sequence in (record.source, record.candidate)}
    )
    chunks: list[tuple[str, str]] = []
    for sequence in sequences:
        chunks.extend(
            (sequence, chunk)
            for chunk in _sequence_chunks(sequence, maximum_nucleotides)
        )
    sums: dict[str, torch.Tensor] = {}
    lengths: dict[str, int] = {}
    batch_count = 0
    with torch.inference_mode():
        for batch in _batches(
            chunks,
            int(config["maximum_sequences_per_batch"]),
            int(config["batch_token_budget"]),
        ):
            tokenized = tokenizer(
                [_format_utr_chunk(chunk) for _sequence, chunk in batch],
                add_special_tokens=True,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            tokenized = {key: value.to(device) for key, value in tokenized.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                last_hidden_state = model(**tokenized)[0]
                pooled = _pool_last_hidden(
                    last_hidden_state, tokenized["attention_mask"]
                )
            pooled = pooled.float().cpu()
            _require(torch.isfinite(pooled).all().item(), "mRNABERT feature is nonfinite")
            for (sequence, chunk), embedding in zip(batch, pooled):
                weight = len(chunk)
                sums[sequence] = sums.get(sequence, torch.zeros_like(embedding)) + weight * embedding
                lengths[sequence] = lengths.get(sequence, 0) + weight
            batch_count += 1
            if batch_count % int(config.get("progress_every_batches", 100)) == 0:
                print(
                    json.dumps(
                        {
                            "event": "MRNABERT_CACHE_PROGRESS",
                            "completed_sequence_count": len(lengths),
                            "batch_count": batch_count,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    embeddings = {
        sequence: sums[sequence] / lengths[sequence] for sequence in sequences
    }
    ordered_records = sorted(records, key=lambda row: row.record_id)
    source_embeddings = torch.stack(
        [embeddings[row.source] for row in ordered_records]
    ).to(torch.float16)
    candidate_embeddings = torch.stack(
        [embeddings[row.candidate] for row in ordered_records]
    ).to(torch.float16)
    payload = {
        "schema_version": "route_a_v3_route2_frozen_pair_features.v1",
        "encoder_family": "mRNABERT",
        "model_id": str(config["model_id"]),
        "pretrained_parameter_count": parameter_count,
        "tokenization_policy": "UTR_SINGLE_NUCLEOTIDE_SPACE_SEPARATED_DNA_ALPHABET",
        "pooling_policy": "OFFICIAL_ATTENTION_MASKED_MEAN_THEN_LENGTH_WEIGHTED_CHUNK_MEAN",
        "attention_backend": "OFFICIAL_PYTORCH_FALLBACK",
        "record_ids": [row.record_id for row in ordered_records],
        "source_embeddings": source_embeddings,
        "candidate_embeddings": candidate_embeddings,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    _require(not temporary.exists(), f"partial cache already exists: {temporary}")
    torch.save(payload, temporary)
    temporary.rename(output_path)
    summary = {
        "schema_version": "route_a_v3_route2_frozen_pair_feature_summary.v1",
        "status": "MRNABERT_FROZEN_FEATURE_CACHE_COMPLETE",
        "record_count": len(ordered_records),
        "unique_sequence_count": len(sequences),
        "chunk_count": len(chunks),
        "maximum_sequence_length": max(len(sequence) for sequence in sequences),
        "maximum_chunk_nucleotides": maximum_nucleotides,
        "embedding_width": int(source_embeddings.shape[1]),
        "pretrained_parameter_count": parameter_count,
        "model_id": str(config["model_id"]),
        "tokenization_policy": payload["tokenization_policy"],
        "pooling_policy": payload["pooling_policy"],
        "attention_backend": payload["attention_backend"],
        "evaluation_record_count": 0,
        "sequence_payload_written": 0,
        "output_path": str(output_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    print(json.dumps(build(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
