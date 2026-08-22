#!/usr/bin/env python3
"""Build the outcome-free, raw-sequence-free SetFlow V3 source-token cache."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_development_projection_v3 import load_projection_rows
from core.route2_source_token_cache_v3 import assemble_source_token_cache_v3
from core.route2_xeditsetflow_training_v3 import setflow_records_from_projection_rows
from scripts.route_a_v3.route2_mrnabert_edit_site_encoder_v3 import FrozenMRNABERTEditSiteEncoderV3


class SetFlowSourceCacheBuildError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SetFlowSourceCacheBuildError(message)


def build(config: Mapping[str, Any]) -> dict[str, Any]:
    output_path = Path(config["output_path"])
    summary_path = Path(config["summary_path"])
    _require(not output_path.exists(), f"source-token cache already exists: {output_path}")
    _require(not summary_path.exists(), f"source-token cache summary already exists: {summary_path}")
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    device = torch.device(str(config["device"]))
    _require(device.type == "cuda", "source-token cache construction requires CUDA")
    rows = load_projection_rows([Path(path) for path in config["projection_paths"]])
    records, eligibility = setflow_records_from_projection_rows(rows)
    _require(len(records) == int(config["expected_eligible_record_count"]), "eligible SetFlow record count changed")
    eligible_ids = {record.record_id for record in records}
    eligible_rows = [row for row in rows if str(row["canonical_record_id"]) in eligible_ids]
    _require(len(eligible_rows) == len(records), "eligible projection rows are incomplete")
    sequences = sorted({record.source for record in records})
    _require(len(sequences) == int(config["expected_unique_source_count"]), "unique source count changed")
    _require(sum(map(len, sequences)) == int(config["expected_unique_source_token_count"]), "unique source token count changed")
    _require(max(map(len, sequences)) == int(config["expected_maximum_source_length"]), "maximum source length changed")
    sequence_to_index = {sequence: index for index, sequence in enumerate(sequences)}
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
    progress_every = int(config["progress_every_batches"])

    def report(batch_index: int, batch_count: int) -> None:
        if batch_index % progress_every == 0 or batch_index == batch_count:
            print(json.dumps({"event": "XEDITSETFLOW_V3_SOURCE_CACHE_PROGRESS", "batch_index": batch_index, "batch_count": batch_count}, sort_keys=True), flush=True)

    encoded = encoder.encode_full_nucleotide_tokens(
        {index: sequence for index, sequence in enumerate(sequences)},
        progress_callback=report,
    )
    payload = assemble_source_token_cache_v3(
        eligible_rows,
        sequence_to_index=sequence_to_index,
        encoded_tokens=encoded,
        model_id=str(config["model_id"]),
        pretrained_parameter_count=encoder.parameter_count,
        attention_backend=encoder.attention_backend,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    _require(not partial.exists(), f"partial source-token cache already exists: {partial}")
    torch.save(payload, partial)
    os.replace(partial, output_path)
    summary = {
        "schema_version": "route_a_v3_route2_setflow_source_token_cache_summary.v3",
        "status": "XEDITSETFLOW_V3_SOURCE_TOKEN_CACHE_COMPLETE",
        "projection_record_count": len(rows),
        "eligible_record_count": len(records),
        "skipped_over_budget_count": eligibility["skipped_over_budget_count"],
        "unique_source_count": len(sequences),
        "unique_source_token_count": int(payload["source_token_hidden"].shape[0]),
        "maximum_source_length": max(map(len, sequences)),
        "embedding_width": int(payload["embedding_width"]),
        "pretrained_parameter_count": encoder.parameter_count,
        "model_id": str(config["model_id"]),
        "attention_backend": encoder.attention_backend,
        "development_test_record_count": 0,
        "development_test_outcomes_accessed": False,
        "evaluation_record_count": 0,
        "evaluation_outcomes_accessed": False,
        "outcome_value_access_count": eligibility["outcome_value_access_count"],
        "raw_sequence_payload_written": 0,
        "output_path": str(output_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(build(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
