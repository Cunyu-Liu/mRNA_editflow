#!/usr/bin/env python3
"""Generate zero-shot Evaluation predictions from one frozen Route 2 neural checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from core.route2_delta_predictor import (
    ROUTE2_DELTA_MODEL_KIND,
    Route2DeltaPredictor,
    Route2NeuralBaseline,
)
from core.route2_gpu_failure_evidence import cuda_device_observation, write_gpu_failure_evidence


TOKEN = {"A": 0, "C": 1, "G": 2, "U": 3}
REGION = {"5UTR": 0, "3UTR": 1}


class FrozenNeuralPredictionError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FrozenNeuralPredictionError(message)


def _normalize(value: Any) -> str:
    sequence = str(value).upper().replace("T", "U")
    _require(sequence and set(sequence) <= set(TOKEN), "sequence is outside RNA alphabet")
    return sequence


def validate_frozen_checkpoint_provenance(provenance: Mapping[str, Any]) -> None:
    physical_index = provenance.get("physical_gpu_index")
    total_memory = provenance.get("cuda_total_memory_mb")
    _require(
        provenance.get("result_stage") == "FROZEN_DEVELOPMENT_TEST"
        and isinstance(provenance.get("optimizer_steps"), int)
        and provenance["optimizer_steps"] > 0
        and provenance.get("parameter_changed") is True
        and provenance.get("cuda_training_tensors_verified") is True
        and provenance.get("cpu_fallback_used") is False
        and isinstance(physical_index, int)
        and not isinstance(physical_index, bool)
        and physical_index >= 0
        and provenance.get("device", provenance.get("torch_device")) == f"cuda:{physical_index}"
        and provenance.get("cuda_device_index") == physical_index
        and isinstance(provenance.get("cuda_device_uuid"), str)
        and bool(provenance.get("cuda_device_uuid"))
        and isinstance(total_memory, (int, float))
        and not isinstance(total_memory, bool)
        and math.isfinite(float(total_memory))
        and float(total_memory) > 0.0,
        "checkpoint does not prove a frozen learned GPU update",
    )


def validate_checkpoint_identity(checkpoint: Mapping[str, Any], baseline_id: str) -> None:
    _require(
        bool(baseline_id) and checkpoint.get("baseline_id") == baseline_id,
        "frozen neural checkpoint identity differs from selected baseline",
    )


@dataclass(frozen=True)
class PredictionRecord:
    record_id: str
    source: str
    candidate: str
    study: str
    assay: str
    context: str
    endpoint: str
    region: int


def load_evaluation_manifest(path: Path) -> set[str]:
    selected = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        _require(row["pool_assignment"] == "EVALUATION", "Development row entered zero-shot Evaluation manifest")
        _require(row["split"] == "EVALUATION_ZERO_SHOT", "Evaluation split is not zero-shot")
        record_id = str(row["canonical_record_id"])
        _require(record_id not in selected, "Evaluation manifest record is duplicated")
        selected.add(record_id)
    _require(selected, "Evaluation manifest is empty")
    return selected


def load_prediction_records(canonical_paths: list[Path], selected_ids: set[str]) -> list[PredictionRecord]:
    records = {}
    for path in canonical_paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                record_id = str(row["canonical_record_id"])
                if record_id not in selected_ids:
                    continue
                _require(record_id not in records, f"Evaluation canonical record is duplicated: {record_id}")
                _require(row["pool_assignment"] == "EVALUATION", "Development record entered zero-shot inference")
                _require(row["training_eligible"] is False, "Evaluation record is training eligible")
                source, candidate = _normalize(row["source_sequence"]), _normalize(row["candidate_sequence"])
                _require(len(source) == len(candidate), "length-changing record entered SUB-only predictor")
                region_text = str(row["region"]).replace("′", "").replace("'", "")
                _require(region_text in REGION, f"unsupported region: {row['region']}")
                records[record_id] = PredictionRecord(
                    record_id, source, candidate, str(row["study_unit_id"]), str(row["assay_id"]),
                    str(row["biological_context_id"]), str(row["endpoint_id"]), REGION[region_text],
                )
    _require(set(records) == selected_ids, "Evaluation canonical inputs do not exactly cover manifest")
    return [records[record_id] for record_id in sorted(records)]


def _load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    validate_frozen_checkpoint_provenance(checkpoint.get("training_provenance", {}))
    model_kind = str(checkpoint.get("model_kind", ""))
    if model_kind == ROUTE2_DELTA_MODEL_KIND:
        model = Route2DeltaPredictor(**checkpoint["model_config"])
    else:
        _require(model_kind in Route2NeuralBaseline.MODES, f"unsupported model kind: {model_kind}")
        model = Route2NeuralBaseline(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval().requires_grad_(False)
    return model, checkpoint, model_kind


@torch.no_grad()
def predict_records(
    model,
    records: list[PredictionRecord],
    vocabs: Mapping[str, Mapping[str, int]],
    device: torch.device,
    batch_size: int,
    baseline_id: str,
) -> list[dict[str, Any]]:
    _require(batch_size > 0, "batch size must be positive")
    _require(bool(baseline_id), "frozen neural baseline identity is empty")
    output = []
    ordered = sorted(records, key=lambda row: (len(row.source), row.record_id))
    for start in range(0, len(ordered), batch_size):
        rows = ordered[start:start + batch_size]
        maximum = max(len(row.source) for row in rows)
        source = torch.full((len(rows), maximum), 4, dtype=torch.long, device=device)
        candidate = torch.full_like(source, 4)
        padding = torch.ones_like(source, dtype=torch.bool)
        for index, row in enumerate(rows):
            length = len(row.source)
            source[index, :length] = torch.tensor([TOKEN[base] for base in row.source], device=device)
            candidate[index, :length] = torch.tensor([TOKEN[base] for base in row.candidate], device=device)
            padding[index, :length] = False
        categories = [
            torch.tensor([vocabs[field].get(str(getattr(row, field)), 0) for row in rows], device=device)
            for field in ("study", "assay", "context", "endpoint")
        ]
        region = torch.tensor([row.region for row in rows], device=device)
        prediction = model(source, candidate, padding, *categories, region)["mean"]
        _require(prediction.is_cuda and prediction.device == device, "zero-shot prediction silently left CUDA")
        for row, value in zip(rows, prediction.cpu().tolist()):
            output.append({
                "canonical_record_id": row.record_id,
                "baseline_id": baseline_id,
                "predicted_direction_normalized_delta": float(value),
                "prediction_role": "EVALUATION_ZERO_SHOT_FROZEN_MODEL",
            })
    return sorted(output, key=lambda row: row["canonical_record_id"])


def execute(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"output already exists: {output_dir}")
    _require(config["schema_version"] == "route_a_v3_route2_frozen_neural_prediction_config.v1", "unexpected config schema")
    _require(config["evaluation_release_state"] == "PREDICTOR_GENERATOR_AND_BASELINES_FROZEN", "Evaluation remains closed")
    _require(config["evaluation_outcomes_used_for_training_hpo_or_selection"] == 0, "Evaluation selected the model")
    _require(str(config["device"]).startswith("cuda:"), "zero-shot neural inference requires CUDA")
    _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden")
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    physical_index = int(config["physical_gpu_index"])
    _require(0 <= physical_index < torch.cuda.device_count(), "physical GPU index is unavailable")
    device = torch.device(str(config["device"]))
    _require(device.index == physical_index, "CUDA device differs from declared physical GPU")
    torch.cuda.set_device(device)
    cuda_provenance = cuda_device_observation(physical_index)
    selected_ids = load_evaluation_manifest(Path(config["evaluation_manifest_path"]))
    records = load_prediction_records([Path(path) for path in config["evaluation_canonical_paths"]], selected_ids)
    model, checkpoint, model_kind = _load_checkpoint(Path(config["checkpoint_path"]), device)
    started = time.time()
    baseline_id = str(config["baseline_id"])
    validate_checkpoint_identity(checkpoint, baseline_id)
    rows = predict_records(
        model, records, checkpoint["vocabs"], device, int(config["batch_size"]), baseline_id
    )
    summary = {
        "schema_version": "route_a_v3_route2_frozen_neural_predictions.v1",
        "status": "EVALUATION_ZERO_SHOT_PREDICTIONS_GENERATED",
        "model_kind": model_kind,
        "baseline_id": baseline_id,
        "evaluation_record_count": len(rows),
        "evaluation_release_state": config["evaluation_release_state"],
        "evaluation_outcome_metrics_computed": False,
        "evaluation_outcomes_used_for_training_hpo_or_selection": 0,
        "physical_gpu_index": physical_index,
        "device": str(device),
        "cpu_fallback_used": False,
        "wall_time_seconds": time.time() - started,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        **cuda_provenance,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        (temporary / "predictions.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
        (temporary / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "run_config.json").write_text(json.dumps(dict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
            output_dir.with_name(output_dir.name + ".failed.json"), config, exc,
            entrypoint="predict_route2_frozen_neural_v1",
            evaluation_outcomes_accessed=True,
        )
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
