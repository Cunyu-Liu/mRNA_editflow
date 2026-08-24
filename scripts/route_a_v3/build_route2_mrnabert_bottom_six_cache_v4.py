#!/usr/bin/env python3
"""Build the outcome-free XEditCritic V4 bottom-six token cache."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_bottom_encoder_chunk_cache_v4 import (
    assemble_frozen_bottom_encoder_chunk_cache_v4,
)
from core.route2_development_projection_v3 import load_projection_rows
from scripts.route_a_v3.route2_mrnabert_bottom_six_encoder_v4 import (
    FrozenMRNABERTBottomSixEncoderV4,
)
from scripts.route_a_v3.authorize_route2_xedit_v4_screen_stages import (
    require_cache_launch_authorization_v4,
    require_cache_runtime_policy_v4,
)


class BottomSixCacheBuildV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BottomSixCacheBuildV4Error(message)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build(
    config: Mapping[str, Any], authorization: Mapping[str, Any]
) -> dict[str, Any]:
    current_head = _git_head()
    require_cache_launch_authorization_v4(
        "critic", authorization, current_git_head=current_head
    )
    physical_gpu_index = require_cache_runtime_policy_v4(config)
    output_path = Path(config["output_path"])
    summary_path = Path(config["summary_path"])
    _require(not output_path.exists(), f"V4 bottom-six cache already exists: {output_path}")
    _require(not summary_path.exists(), f"V4 bottom-six summary already exists: {summary_path}")
    _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden")
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    _require(physical_gpu_index < torch.cuda.device_count(), "selected physical GPU is unavailable")
    device = torch.device(f"cuda:{physical_gpu_index}")
    torch.cuda.set_device(device)
    _require(torch.cuda.is_bf16_supported(), "BF16 is unavailable on selected GPU")
    rows = load_projection_rows([Path(path) for path in config["projection_paths"]])
    _require(len(rows) == int(config["expected_record_count"]), "V4 projection record count changed")
    _require(all(str(row["split"]) in {"TRAIN", "VALIDATION"} for row in rows), "protected split entered V4 cache construction")
    sequences = sorted(
        {
            str(sequence)
            for row in rows
            for sequence in (row["source_sequence"], row["candidate_sequence"])
        }
    )
    _require(len(sequences) == int(config["expected_unique_sequence_count"]), "V4 unique sequence count changed")
    sequence_to_index = {sequence: index for index, sequence in enumerate(sequences)}
    encoder = FrozenMRNABERTBottomSixEncoderV4(
        Path(config["mrnabert_model_path"]),
        device,
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
                        "event": "XEDITCRITIC_V4_BOTTOM_SIX_CACHE_PROGRESS",
                        "batch_index": batch_index,
                        "batch_count": batch_count,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    encoded = encoder.encode_sequences(
        {index: sequence for index, sequence in enumerate(sequences)},
        progress_callback=report,
    )
    payload = assemble_frozen_bottom_encoder_chunk_cache_v4(
        rows,
        sequence_to_index=sequence_to_index,
        encoded=encoded,
        model_id=str(config["model_id"]),
        pretrained_parameter_count=encoder.parameter_count,
        attention_backend=encoder.attention_backend,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    _require(not temporary.exists(), f"partial V4 cache already exists: {temporary}")
    torch.save(payload, temporary)
    os.replace(temporary, output_path)
    summary = {
        "schema_version": "route_a_v3_route2_frozen_bottom_encoder_chunk_cache_summary.v4",
        "status": "XEDITCRITIC_V4_BOTTOM_SIX_CACHE_COMPLETE",
        "record_count": len(payload["record_ids"]),
        "record_edit_count": int(payload["edit_positions"].numel()),
        "unique_sequence_count": len(sequences),
        "chunk_count": int(payload["chunk_starts"].numel()),
        "cached_token_count": int(payload["token_hidden"].shape[0]),
        "maximum_sequence_length": max(map(len, sequences)),
        "maximum_record_edit_count": max(len(row["source_relative_edits"]) for row in rows),
        "chunk_nucleotides": int(payload["chunk_length"]),
        "chunk_overlap": int(payload["chunk_overlap"]),
        "local_context_radius": int(payload["local_context_radius"]),
        "embedding_width": int(payload["embedding_width"]),
        "pretrained_parameter_count": encoder.parameter_count,
        "frozen_encoder_blocks": payload["frozen_encoder_blocks"],
        "trainable_encoder_blocks": payload["trainable_encoder_blocks"],
        "model_id": str(config["model_id"]),
        "attention_backend": encoder.attention_backend,
        "git_head": current_head,
        "cache_launch_authorization_status": authorization["status"],
        "physical_gpu_index": physical_gpu_index,
        "cuda_device_name": torch.cuda.get_device_name(device),
        "forward_precision": "BF16",
        "cpu_fallback": False,
        "projection_paths": list(config["projection_paths"]),
        "raw_sequence_payload_written": 0,
        "label_or_outcome_payload_written": 0,
        "development_test_record_count": 0,
        "development_test_outcomes_accessed": False,
        "evaluation_record_count": 0,
        "evaluation_outcomes_accessed": False,
        "output_path": str(output_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    authorization = json.loads(arguments.authorization.read_text(encoding="utf-8"))
    print(json.dumps(build(config, authorization), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
