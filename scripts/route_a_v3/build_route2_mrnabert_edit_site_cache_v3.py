#!/usr/bin/env python3
"""Build the outcome-isolated, raw-sequence-free XEditCritic V3 token cache."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_development_projection_v3 import load_projection_rows
from core.route2_edit_site_token_cache_v3 import assemble_edit_site_token_cache_v3
from scripts.route_a_v3.route2_mrnabert_edit_site_encoder_v3 import (
    FrozenMRNABERTEditSiteEncoderV3,
)


class CacheBuildError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CacheBuildError(message)


def build(config: Mapping[str, Any]) -> dict[str, Any]:
    output_path = Path(config["output_path"])
    summary_path = Path(config["summary_path"])
    _require(not output_path.exists(), f"feature cache already exists: {output_path}")
    _require(not summary_path.exists(), f"feature summary already exists: {summary_path}")
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    device = torch.device(str(config.get("device", "cuda:0")))
    _require(device.type == "cuda", "cache construction requires CUDA")
    rows = load_projection_rows([Path(path) for path in config["projection_paths"]])
    _require(len(rows) == int(config["expected_record_count"]), "projection row count changed")
    _require(all(row["split"] in {"TRAIN", "VALIDATION"} for row in rows), "protected row entered the cache")

    sequences = sorted(
        {str(sequence) for row in rows for sequence in (row["source_sequence"], row["candidate_sequence"])}
    )
    sequence_to_index = {sequence: index for index, sequence in enumerate(sequences)}
    positions_by_sequence: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        positions = {int(edit["position"]) for edit in row["source_relative_edits"]}
        positions_by_sequence[sequence_to_index[str(row["source_sequence"])]].update(positions)
        positions_by_sequence[sequence_to_index[str(row["candidate_sequence"])]].update(positions)
    for index in range(len(sequences)):
        positions_by_sequence[index]
    _require(len(sequences) == int(config["expected_unique_sequence_count"]), "unique sequence count changed")
    unique_position_count = sum(len(values) for values in positions_by_sequence.values())
    _require(unique_position_count == int(config["expected_unique_sequence_position_count"]), "unique sequence-position count changed")

    encoder = FrozenMRNABERTEditSiteEncoderV3(
        Path(config["mrnabert_model_path"]),
        device,
        chunk_nucleotides=int(config["chunk_nucleotides"]),
        chunk_overlap=int(config["chunk_overlap"]),
        local_radius=int(config["local_radius"]),
        maximum_sequences_per_batch=int(config["maximum_sequences_per_batch"]),
        batch_token_budget=int(config["batch_token_budget"]),
        attention_backend=str(config["attention_backend"]),
    )
    progress_every = int(config.get("progress_every_batches", 1000))

    def report(batch_index: int, batch_count: int) -> None:
        if batch_index % progress_every == 0 or batch_index == batch_count:
            print(
                json.dumps(
                    {
                        "event": "XEDITCRITIC_V3_CACHE_PROGRESS",
                        "batch_index": batch_index,
                        "batch_count": batch_count,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    encoded = encoder.encode_requested_features(
        {index: sequence for index, sequence in enumerate(sequences)},
        positions_by_sequence,
        progress_callback=report,
    )
    payload = assemble_edit_site_token_cache_v3(
        rows,
        sequence_to_index=sequence_to_index,
        encoded=encoded,
        model_id=str(config["model_id"]),
        pretrained_parameter_count=encoder.parameter_count,
        attention_backend=encoder.attention_backend,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    _require(not temporary.exists(), f"partial cache already exists: {temporary}")
    torch.save(payload, temporary)
    os.replace(temporary, output_path)
    summary = {
        "schema_version": "route_a_v3_route2_edit_site_token_feature_summary.v3",
        "status": "XEDITCRITIC_V3_EDIT_SITE_CACHE_COMPLETE",
        "record_count": len(payload["record_ids"]),
        "record_edit_count": int(payload["edit_positions"].numel()),
        "unique_sequence_count": len(sequences),
        "unique_sequence_position_count": unique_position_count,
        "maximum_sequence_length": max(map(len, sequences)),
        "maximum_record_edit_count": max(len(row["source_relative_edits"]) for row in rows),
        "chunk_nucleotides": int(config["chunk_nucleotides"]),
        "chunk_overlap": int(config["chunk_overlap"]),
        "local_radius": int(config["local_radius"]),
        "embedding_width": int(payload["embedding_width"]),
        "pretrained_parameter_count": encoder.parameter_count,
        "model_id": str(config["model_id"]),
        "attention_backend": encoder.attention_backend,
        "projection_paths": list(config["projection_paths"]),
        "development_test_record_count": 0,
        "development_test_outcomes_accessed": False,
        "evaluation_record_count": 0,
        "evaluation_outcomes_accessed": False,
        "raw_sequence_payload_written": 0,
        "output_path": str(output_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    print(json.dumps(build(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
