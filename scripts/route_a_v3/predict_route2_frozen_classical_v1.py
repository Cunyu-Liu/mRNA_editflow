#!/usr/bin/env python3
"""Generate zero-shot Evaluation predictions from one frozen Route 2 classical artifact."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import joblib
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_gpu_failure_evidence import cuda_device_observation, write_gpu_failure_evidence
from scripts.route_a_v3 import run_route2_classical_prediction_baselines_v1 as classical


class FrozenClassicalPredictionError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FrozenClassicalPredictionError(message)


def validate_frozen_result(result: Mapping[str, Any]) -> None:
    provenance = result.get("execution_provenance", {})
    total_memory = provenance.get("cuda_total_memory_mb")
    physical_index = provenance.get("physical_gpu_index")
    _require(
        result.get("status") == "COMPLETED_FROZEN_DEVELOPMENT_TEST"
        and result.get("evaluation_outcomes_accessed") is False
        and provenance.get("parameter_fit_execution") == "CUDA"
        and provenance.get("prediction_execution") == "CUDA"
        and provenance.get("cpu_fallback_used") is False
        and isinstance(physical_index, int)
        and not isinstance(physical_index, bool)
        and physical_index >= 0
        and provenance.get("device") == f"cuda:{physical_index}"
        and provenance.get("cuda_device_index") == physical_index
        and isinstance(provenance.get("cuda_device_uuid"), str)
        and bool(provenance.get("cuda_device_uuid"))
        and isinstance(total_memory, (int, float))
        and not isinstance(total_memory, bool)
        and math.isfinite(float(total_memory))
        and float(total_memory) > 0,
        "classical artifact does not prove a frozen CUDA fit",
    )


def validate_artifact_identity(
    artifact: Mapping[str, Any], baseline_id: str, baseline_kind: str
) -> None:
    _require(
        artifact.get("artifact_baseline_id") == baseline_id
        and artifact.get("artifact_baseline_kind") == baseline_kind,
        "frozen artifact identity differs from selected baseline",
    )


def load_evaluation_manifest(path: Path) -> set[str]:
    selected: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FrozenClassicalPredictionError(f"invalid manifest JSON at line {line_number}") from exc
            _require(row["pool_assignment"] == "EVALUATION", "Development row entered zero-shot manifest")
            _require(row["split"] == "EVALUATION_ZERO_SHOT", "Evaluation split is not zero-shot")
            record_id = str(row["canonical_record_id"])
            _require(record_id not in selected, "Evaluation manifest record is duplicated")
            selected.add(record_id)
    _require(selected, "Evaluation manifest is empty")
    return selected


def load_prediction_records(canonical_paths: list[Path], selected_ids: set[str]) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in canonical_paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise FrozenClassicalPredictionError(
                        f"invalid canonical JSON in {path.name}:{line_number}"
                    ) from exc
                record_id = str(row["canonical_record_id"])
                if record_id not in selected_ids:
                    continue
                _require(record_id not in records, f"Evaluation canonical record is duplicated: {record_id}")
                _require(row["pool_assignment"] == "EVALUATION", "Development record entered zero-shot inference")
                _require(row["training_eligible"] is False, "Evaluation record is training eligible")
                source = str(row["source_sequence"]).upper().replace("U", "T")
                candidate = str(row["candidate_sequence"]).upper().replace("U", "T")
                _require(
                    len(source) == len(candidate)
                    and source
                    and not ((set(source) | set(candidate)) - set(classical.BASES)),
                    "invalid Evaluation sequence pair",
                )
                records[record_id] = {
                    "canonical_record_id": record_id,
                    "study_unit_id": str(row["study_unit_id"]),
                    "region": str(row["region"]),
                    "endpoint_id": str(row["endpoint_id"]),
                    "assay_id": str(row["assay_id"]),
                    "biological_context_id": str(row["biological_context_id"]),
                    "source_id": str(row["source_id"]),
                    "source_sequence": source,
                    "candidate_sequence": candidate,
                    "edit_operations": row["edit_operations"],
                    "source_endpoint_value": None,
                    "candidate_endpoint_value": None,
                }
    _require(set(records) == selected_ids, "Evaluation canonical inputs do not exactly cover manifest")
    return [records[record_id] for record_id in sorted(records)]


def execute(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"output already exists: {output_dir}")
    _require(
        config["schema_version"] == "route_a_v3_route2_frozen_classical_prediction_config.v1",
        "unexpected config schema",
    )
    _require(
        config["evaluation_release_state"] == "PREDICTOR_GENERATOR_AND_BASELINES_FROZEN",
        "Evaluation remains closed",
    )
    _require(config["evaluation_outcomes_used_for_training_hpo_or_selection"] == 0, "Evaluation selected the model")
    _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden")
    physical_index = int(config["physical_gpu_index"])
    device, observed_free_bytes = classical.require_cuda_device(
        str(config["device"]), physical_index, int(config.get("minimum_free_gpu_memory_bytes", 0))
    )
    cuda_provenance = cuda_device_observation(physical_index)
    result = json.loads(Path(config["frozen_result_path"]).read_text(encoding="utf-8"))
    validate_frozen_result(result)
    baseline = dict(config["baseline"])
    _require(str(config["baseline_id"]) == str(baseline["baseline_id"]), "baseline identity differs")
    artifact = joblib.load(Path(config["frozen_model_path"]))
    validate_artifact_identity(artifact, str(config["baseline_id"]), str(baseline["kind"]))
    selected_ids = load_evaluation_manifest(Path(config["evaluation_manifest_path"]))
    records = load_prediction_records(
        [Path(path) for path in config["evaluation_canonical_paths"]], selected_ids
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    values = classical.predict_from_frozen_artifact(baseline, artifact, records, device)
    torch.cuda.synchronize(device)
    _require(len(values) == len(records), "frozen classical prediction count differs")
    rows = [
        {
            "canonical_record_id": record["canonical_record_id"],
            "baseline_id": str(config["baseline_id"]),
            "predicted_direction_normalized_delta": float(value),
            "prediction_role": "EVALUATION_ZERO_SHOT_FROZEN_MODEL",
        }
        for record, value in zip(records, values)
    ]
    summary = {
        "schema_version": "route_a_v3_route2_frozen_classical_predictions.v1",
        "status": "EVALUATION_ZERO_SHOT_PREDICTIONS_GENERATED",
        "baseline_id": str(config["baseline_id"]),
        "baseline_kind": str(baseline["kind"]),
        "evaluation_record_count": len(rows),
        "evaluation_release_state": config["evaluation_release_state"],
        "evaluation_outcome_metrics_computed": False,
        "evaluation_outcomes_used_for_training_hpo_or_selection": 0,
        "physical_gpu_index": physical_index,
        "device": str(device),
        "cpu_fallback_used": False,
        "observed_free_gpu_memory_bytes_before_run": observed_free_bytes,
        "wall_time_seconds": time.monotonic() - started,
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
            entrypoint="predict_route2_frozen_classical_v1",
            evaluation_outcomes_accessed=True,
        )
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
