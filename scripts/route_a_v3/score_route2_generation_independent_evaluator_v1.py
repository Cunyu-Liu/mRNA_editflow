#!/usr/bin/env python3
"""Score generated candidates with a pre-frozen, non-guiding Route 2 evaluator."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_delta_predictor import (
    ROUTE2_DELTA_MODEL_KIND,
    ROUTE2_EDIT_CENTERED_MODEL_KIND,
    ROUTE2_EDIT_CENTERED_SOURCE_ONLY_KIND,
    Route2DeltaPredictor,
    Route2EditCenteredDeltaPredictor,
    Route2NeuralBaseline,
)
from core.route2_gpu_failure_evidence import cuda_device_observation, write_gpu_failure_evidence
from core.route2_target_scaling import target_scaler_from_checkpoint


TOKEN = {"A": 0, "C": 1, "G": 2, "U": 3}
REGION = {"5UTR": 0, "3UTR": 1}


class IndependentEvaluatorError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IndependentEvaluatorError(message)


def _normalize(value: Any) -> str:
    sequence = str(value).upper().replace("T", "U")
    _require(sequence and set(sequence) <= set(TOKEN), "sequence is outside the RNA alphabet")
    return sequence


def _finite(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    result = float(value)
    _require(math.isfinite(result), f"{label} is not finite")
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _require(rows, f"input is empty: {path}")
    return rows


def validate_frozen_evaluator_provenance(provenance: Mapping[str, Any]) -> None:
    physical_index = provenance.get("physical_gpu_index")
    total_memory = provenance.get("cuda_total_memory_mb")
    training_device = provenance.get("device", provenance.get("torch_device"))
    _require(
        provenance.get("result_stage") == "FROZEN_DEVELOPMENT_TEST"
        and isinstance(provenance.get("optimizer_steps"), int)
        and int(provenance["optimizer_steps"]) > 0
        and provenance.get("parameter_changed") is True
        and provenance.get("cuda_training_tensors_verified") is True
        and provenance.get("cpu_fallback_used") is False
        and isinstance(physical_index, int)
        and not isinstance(physical_index, bool)
        and physical_index >= 0
        and training_device == f"cuda:{physical_index}"
        and provenance.get("cuda_device_index") == physical_index
        and isinstance(provenance.get("cuda_device_uuid"), str)
        and bool(provenance.get("cuda_device_uuid"))
        and isinstance(total_memory, (int, float))
        and not isinstance(total_memory, bool)
        and math.isfinite(float(total_memory))
        and float(total_memory) > 0.0,
        "evaluator checkpoint does not prove a frozen learned GPU update",
    )


def load_sources(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in _read_jsonl(path):
        key = str(row["source_key"])
        _require(key not in result, f"source duplicated: {key}")
        _require(row["evaluation_outcomes_included"] is False, "Evaluation outcome entered evaluator source manifest")
        result[key] = {**row, "source_sequence": _normalize(row["source_sequence"])}
    return result


class CheckpointScorer:
    def __init__(self, checkpoint_path: Path, device: torch.device):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        provenance = checkpoint.get("training_provenance", {})
        validate_frozen_evaluator_provenance(provenance)
        model_kind = str(checkpoint.get("model_kind", ""))
        if model_kind == ROUTE2_DELTA_MODEL_KIND:
            model = Route2DeltaPredictor(**checkpoint["model_config"])
        elif model_kind in {ROUTE2_EDIT_CENTERED_MODEL_KIND, ROUTE2_EDIT_CENTERED_SOURCE_ONLY_KIND}:
            model = Route2EditCenteredDeltaPredictor(**checkpoint["model_config"])
        else:
            _require(model_kind in Route2NeuralBaseline.MODES, f"unsupported evaluator model kind: {model_kind}")
            model = Route2NeuralBaseline(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state"])
        self.model = model.to(device).eval()
        self.model.requires_grad_(False)
        self.vocabs = checkpoint["vocabs"]
        self.target_scaler = target_scaler_from_checkpoint(checkpoint)
        self.device = device
        self.model_kind = model_kind
        self.training_provenance = provenance

    @torch.no_grad()
    def __call__(self, source_row: Mapping[str, Any], candidate: str) -> float:
        source = _normalize(source_row["source_sequence"])
        candidate = _normalize(candidate)
        _require(len(source) == len(candidate), "candidate length differs from source")
        region_text = str(source_row["region"]).replace("′", "").replace("'", "")
        _require(region_text in REGION, f"unsupported region: {source_row['region']}")
        source_tokens = torch.tensor([[TOKEN[base] for base in source]], device=self.device)
        candidate_tokens = torch.tensor([[TOKEN[base] for base in candidate]], device=self.device)
        padding = torch.zeros_like(source_tokens, dtype=torch.bool)
        category = {
            field: int(self.vocabs[field].get(str(source_row[source_field]), 0))
            for field, source_field in (
                ("study", "study_unit_id"),
                ("assay", "assay_id"),
                ("context", "biological_context_id"),
                ("endpoint", "endpoint_id"),
            )
        }
        output = self.model(
            source_tokens,
            candidate_tokens,
            padding,
            torch.tensor([category["study"]], device=self.device),
            torch.tensor([category["assay"]], device=self.device),
            torch.tensor([category["context"]], device=self.device),
            torch.tensor([category["endpoint"]], device=self.device),
            torch.tensor([REGION[region_text]], device=self.device),
        )
        scale, _scale_source = self.target_scaler.scale(
            str(source_row["endpoint_id"]), REGION[region_text]
        )
        return _finite(float(output["mean"].item()) * scale, "independent evaluator score")


def augment_candidates(
    sources: Mapping[str, Mapping[str, Any]],
    candidates: list[dict[str, Any]],
    score_function: Callable[[Mapping[str, Any], str], float],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    source_keys = {str(row["source_key"]) for row in candidates}
    _require(source_keys == set(sources), "candidate source coverage differs from source manifest")
    output = []
    forward_count_by_source = {}
    for source_key in sorted(sources):
        source_row = sources[source_key]
        rows = [row for row in candidates if str(row["source_key"]) == source_key]
        _require(rows, f"candidate source is empty: {source_key}")
        source_sequence = str(source_row["source_sequence"])
        cache = {source_sequence: _finite(score_function(source_row, source_sequence), "source evaluator score")}
        source_forward_pending = 1
        forwards = 1
        for row in rows:
            _require(_finite(row.get("independent_evaluator_forwards", 0), "existing evaluator forwards") == 0.0, "candidate was already independently scored")
            candidate = _normalize(row["candidate_sequence"])
            new_forward = 0
            if candidate not in cache:
                cache[candidate] = _finite(score_function(source_row, candidate), "candidate evaluator score")
                forwards += 1
                new_forward = 1
            output.append({
                **row,
                "source_independent_evaluator_score": cache[source_sequence],
                "independent_evaluator_score": cache[candidate],
                "independent_evaluator_forwards": new_forward + source_forward_pending,
            })
            source_forward_pending = 0
        forward_count_by_source[source_key] = forwards
    return output, forward_count_by_source


def execute(config: Mapping[str, Any], output_path: Path) -> dict[str, Any]:
    _require(not output_path.exists(), f"output already exists: {output_path}")
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    _require(not summary_path.exists(), f"summary output already exists: {summary_path}")
    _require(config["evaluator_frozen_before_candidate_generation"] is True, "independent evaluator was not pre-frozen")
    _require(config["evaluation_outcomes_used_to_select_evaluator"] == 0, "Evaluation selected the independent evaluator")
    evaluator_path = Path(config["evaluator_checkpoint_path"]).resolve()
    guiding_path = config.get("guiding_checkpoint_path")
    if guiding_path is not None:
        _require(evaluator_path != Path(guiding_path).resolve(), "independent evaluator and guiding critic are the same checkpoint")
    _require(str(config["device"]).startswith("cuda"), "independent evaluator scoring requires CUDA")
    _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden for physical-device provenance")
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    device = torch.device(str(config["device"]))
    _require(0 <= int(config["physical_gpu_index"]) < torch.cuda.device_count(), "physical GPU index is unavailable")
    _require(device.index == int(config["physical_gpu_index"]), "CUDA device index differs from declared physical GPU")
    torch.cuda.set_device(device)
    cuda_provenance = cuda_device_observation(int(config["physical_gpu_index"]), require_physical_index_match=True)
    sources = load_sources(Path(config["source_manifest_path"]))
    candidates = _read_jsonl(Path(config["candidate_path"]))
    scorer = CheckpointScorer(evaluator_path, device)
    started = time.time()
    rows, forwards = augment_candidates(sources, candidates, scorer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "schema_version": "route_a_v3_route2_independent_generation_evaluator.v1",
        "status": "FROZEN_INDEPENDENT_EVALUATOR_SCORING_COMPLETE",
        "model_kind": scorer.model_kind,
        "candidate_row_count": len(rows),
        "source_count": len(sources),
        "independent_evaluator_forward_count": sum(forwards.values()),
        "independent_evaluator_forwards_by_source": forwards,
        "evaluator_frozen_before_candidate_generation": True,
        "guiding_checkpoint_distinct": guiding_path is None or evaluator_path != Path(guiding_path).resolve(),
        "evaluation_outcomes_used_to_select_evaluator": 0,
        "physical_gpu_index": int(config["physical_gpu_index"]),
        "device": str(device),
        "wall_time_seconds": time.time() - started,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        **cuda_provenance,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        result = execute(config, args.output)
    except Exception as exc:
        write_gpu_failure_evidence(
            args.output.with_suffix(args.output.suffix + ".failed.json"), config, exc,
            entrypoint="score_route2_generation_independent_evaluator_v1",
            evaluation_outcomes_accessed=False,
        )
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
