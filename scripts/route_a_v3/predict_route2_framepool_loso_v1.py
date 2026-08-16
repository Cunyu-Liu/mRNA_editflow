#!/usr/bin/env python3
"""Generate GSE114002 LOSO predictions with the frozen native FramePool model."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_gpu_failure_evidence import cuda_device_observation, write_gpu_failure_evidence
from scripts.route_a_v3 import run_route2_external_prediction_baselines_v1 as external


class FramePoolLosoError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FramePoolLosoError(message)


def load_loso_records(
    canonical_path: Path,
    manifest_path: Path,
    holdout_study_unit_id: str,
) -> list[external.TaskRecord]:
    _require(holdout_study_unit_id == external.TASK_STUDY, "FramePool LOSO is only task-matched to GSE114002")
    manifest_rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = {
        str(row["canonical_record_id"]): row
        for row in manifest_rows
        if str(row["study_unit_id"]) == holdout_study_unit_id
    }
    _require(selected, "GSE114002 is absent from the Development manifest")
    _require(
        all(row["pool_assignment"] == "DEVELOPMENT" for row in selected.values()),
        "non-Development row entered FramePool LOSO manifest",
    )
    records: dict[str, external.TaskRecord] = {}
    with canonical_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = str(row["canonical_record_id"])
            if record_id not in selected:
                continue
            _require(record_id not in records, f"canonical record is duplicated: {record_id}")
            _require(row["pool_assignment"] == "DEVELOPMENT", "non-Development record entered FramePool LOSO")
            _require(
                row["study_unit_id"] == external.TASK_STUDY
                and row["region"] == external.TASK_REGION
                and row["endpoint_id"] == external.TASK_ENDPOINT,
                "FramePool LOSO task identity changed",
            )
            source = str(row["source_sequence"]).upper()
            candidate = str(row["candidate_sequence"]).upper()
            _require(len(source) == len(candidate) == 50, "FramePool LOSO requires exact 50-nt pairs")
            _require(not ((set(source) | set(candidate)) - set(external.BASES)), "FramePool LOSO alphabet changed")
            records[record_id] = external.TaskRecord(
                record_id=record_id,
                source_id=str(row["source_id"]),
                source=source,
                candidate=candidate,
                target=0.0,
                split="LOSO_HOLDOUT",
            )
    _require(set(records) == set(selected), "FramePool LOSO canonical inputs do not exactly cover holdout manifest")
    return [records[record_id] for record_id in sorted(records)]


def execute(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"output already exists: {output_dir}")
    _require(
        config["schema_version"] == "route_a_v3_route2_framepool_loso_prediction_config.v1",
        "unexpected FramePool LOSO config schema",
    )
    _require(config["evaluation_outcomes_accessed"] is False, "FramePool LOSO accessed Evaluation")
    _require(config["holdout_outcomes_accessed"] is False, "FramePool LOSO accessed holdout outcomes")
    _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden")
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    physical_index = int(config["physical_gpu_index"])
    _require(0 <= physical_index < torch.cuda.device_count(), "physical GPU index is unavailable")
    device = torch.device(str(config["device"]))
    _require(device.type == "cuda" and device.index == physical_index, "CUDA device differs from declared physical GPU")
    torch.cuda.set_device(device)
    cuda_provenance = cuda_device_observation(physical_index, require_physical_index_match=True)
    records = load_loso_records(
        Path(config["canonical_path"]),
        Path(config["development_manifest_path"]),
        str(config["loso_holdout_study_unit_id"]),
    )
    model = external.FramePool(Path(config["framepool_weight_path"])).to(device).eval()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    predictions = external._predict_native(model, records, device, int(config["batch_size"]))
    torch.cuda.synchronize(device)
    _require(set(predictions) == {record.record_id for record in records}, "FramePool LOSO predictions do not close")
    rows = [
        {
            "canonical_record_id": record.record_id,
            "baseline_id": "external_framepool",
            "predicted_direction_normalized_delta": predictions[record.record_id],
            "prediction_role": "DEVELOPMENT_LOSO_FROZEN_NATIVE_MODEL",
        }
        for record in records
    ]
    summary = {
        "schema_version": "route_a_v3_route2_framepool_loso_predictions.v1",
        "status": "DEVELOPMENT_LOSO_PREDICTIONS_GENERATED",
        "baseline_id": "external_framepool",
        "loso_holdout_study_unit_id": external.TASK_STUDY,
        "holdout_record_count": len(rows),
        "pretrained_parameters_frozen": True,
        "prediction_definition": "f(candidate)-f(source)",
        "artifact_provenance_id": "framepool_official_c575f9cd",
        "pytorch_port_numeric_parity_status": "NOT_RUN_TENSORFLOW_UNAVAILABLE",
        "evaluation_outcomes_accessed": False,
        "holdout_outcomes_accessed": False,
        "device": str(device),
        "physical_gpu_index": physical_index,
        "cpu_fallback_used": False,
        "cuda_inference_tensors_verified": True,
        "wall_time_seconds": time.monotonic() - started,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        **cuda_provenance,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        (temporary / "loso_predictions.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
        (temporary / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "run_config.json").write_text(
            json.dumps(dict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.rename(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or Path(config["output_directory"])
    try:
        result = execute(config, output_dir)
    except Exception as exc:
        write_gpu_failure_evidence(
            output_dir.with_name(output_dir.name + ".failed.json"),
            config,
            exc,
            entrypoint="predict_route2_framepool_loso_v1",
            evaluation_outcomes_accessed=False,
        )
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
