#!/usr/bin/env python3
"""Diagnose TRAIN-task gradient conflict after a Development screen NO-GO.

This is a conditional, read-only model diagnostic.  It requires a terminal
support-aware screen adjudication that did not authorize confirmation, loads a
Development-only edit-centered checkpoint, and computes task-gradient cosine
similarities on CUDA.  It performs no optimizer step and reads neither
Development TEST nor Evaluation outcomes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_delta_predictor import (  # noqa: E402
    ROUTE2_EDIT_CENTERED_MODEL_KIND,
    Route2EditCenteredDeltaPredictor,
)
from core.route2_gpu_failure_evidence import (  # noqa: E402
    cuda_device_observation,
    write_gpu_failure_evidence,
)
from core.route2_target_scaling import target_scaler_from_checkpoint  # noqa: E402
from scripts.route_a_v3.train_route2_delta_predictor_v1 import (  # noqa: E402
    DeltaDataset,
    SourceGroupBatchSampler,
    _forward,
    _move,
    collate,
    load_manifest,
    load_records,
    multitask_loss,
    require_cuda,
    task_key,
)


class TaskGradientDiagnosticError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TaskGradientDiagnosticError(message)


def validate_terminal_no_go(adjudication: Mapping[str, Any]) -> None:
    """Require a terminal v2 screen decision that did not authorize seeds."""

    _require(
        adjudication.get("schema_version")
        == "route_a_v3_route2_method_repair_screen_adjudication.v2",
        "support-aware screen adjudication is missing",
    )
    status = str(adjudication.get("status", ""))
    _require(status.startswith("EXPLORATORY_"), "screen adjudication is not terminal")
    _require(
        status != "EXPLORATORY_SCREEN_SUPPORTS_FRESH_SEED_CONFIRMATION",
        "confirmation was authorized; gradient-conflict repair diagnosis is not allowed",
    )
    _require(not adjudication.get("fresh_confirmation_seeds"), "NO-GO contains seeds")
    _require(
        adjudication.get("evaluation_used_for_selection") is False,
        "Evaluation entered screen selection",
    )
    _require(
        adjudication.get("development_test_used_for_selection") is False,
        "Development TEST entered screen selection",
    )


def shared_gradient_parameters(
    model: Route2EditCenteredDeltaPredictor,
) -> list[tuple[str, torch.nn.Parameter]]:
    """Return parameters shared across tasks, excluding categorical adapters."""

    excluded_prefixes = (
        "assay.",
        "context.",
        "endpoint.",
        "region.",
        "region_scale.",
        "region_shift.",
    )
    selected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not any(name.startswith(prefix) for prefix in excluded_prefixes)
    ]
    _require(bool(selected), "shared gradient parameter set is empty")
    return selected


def gradient_vector(
    parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> torch.Tensor:
    values = []
    for _name, parameter in parameters:
        values.append(
            parameter.grad.reshape(-1)
            if parameter.grad is not None
            else torch.zeros_like(parameter).reshape(-1)
        )
    return torch.cat(values)


def cosine_matrix(
    gradients: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    tasks = sorted(gradients)
    _require(len(tasks) >= 2, "at least two TRAIN tasks are required")
    for task in tasks:
        gradient = gradients[task]
        _require(gradient.ndim == 1, f"gradient for {task} is not a vector")
        _require(torch.isfinite(gradient).all().item(), f"gradient for {task} is not finite")
        _require(float(torch.linalg.vector_norm(gradient)) > 0.0, f"gradient for {task} is zero")
    return {
        left: {
            right: float(
                torch.nn.functional.cosine_similarity(
                    gradients[left], gradients[right], dim=0
                )
            )
            for right in tasks
        }
        for left in tasks
    }


def evenly_spaced_batches(
    sampler: SourceGroupBatchSampler, maximum: int
) -> list[list[int]]:
    _require(maximum > 0, "maximum batch count must be positive")
    batches = list(sampler.batches)
    _require(bool(batches), "task sampler has no batches")
    if len(batches) <= maximum:
        return batches
    positions = [
        round(index * (len(batches) - 1) / (maximum - 1))
        for index in range(maximum)
    ] if maximum > 1 else [len(batches) // 2]
    return [batches[index] for index in positions]


def load_edit_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    _require(
        checkpoint.get("model_kind") == ROUTE2_EDIT_CENTERED_MODEL_KIND,
        "checkpoint is not the edit-centered candidate model",
    )
    provenance = checkpoint.get("training_provenance") or {}
    _require(
        provenance.get("result_stage") == "HPO_VALIDATION_ONLY",
        "checkpoint is not Development Validation exploratory",
    )
    _require(
        provenance.get("cuda_training_tensors_verified") is True
        and provenance.get("cpu_fallback_used") is False,
        "checkpoint lacks CUDA-only training proof",
    )
    _require(
        provenance.get("parameter_changed") is True,
        "checkpoint lacks a learned parameter update",
    )
    model = Route2EditCenteredDeltaPredictor(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval().requires_grad_(True)
    return model, checkpoint


def execute(
    *,
    training_config_path: Path,
    checkpoint_path: Path,
    adjudication_path: Path,
    output_path: Path,
    physical_gpu_index: int,
    maximum_batches_per_task: int,
) -> dict[str, Any]:
    _require(not output_path.exists(), f"output already exists: {output_path}")
    adjudication = json.loads(adjudication_path.read_text(encoding="utf-8"))
    validate_terminal_no_go(adjudication)
    config = json.loads(training_config_path.read_text(encoding="utf-8"))
    _require(config.get("result_stage") == "HPO_VALIDATION_ONLY", "config stage differs")
    _require(config.get("evaluation_outcomes_accessed") is False, "Evaluation entered config")
    _require(config.get("development_test_outcomes_accessed") is False, "TEST entered config")
    device = require_cuda(f"cuda:{physical_gpu_index}", physical_gpu_index)
    cuda_provenance = cuda_device_observation(
        physical_gpu_index, require_physical_index_match=True
    )
    model, checkpoint = load_edit_checkpoint(checkpoint_path, device)
    manifest = load_manifest(Path(config["development_manifest"]))
    all_records = load_records(
        [Path(value) for value in config["canonical_paths"]], manifest
    )
    train_records = [record for record in all_records if record.split == "TRAIN"]
    _require(len(train_records) == 89580, "TRAIN record count differs from frozen screen")
    records_by_task: dict[str, list[Any]] = {}
    for record in train_records:
        records_by_task.setdefault(task_key(record.endpoint, record.region), []).append(record)

    parameters = shared_gradient_parameters(model)
    gradients = {}
    task_rows = {}
    cuda_losses_verified = True
    for task in sorted(records_by_task):
        records = records_by_task[task]
        dataset = DeltaDataset(
            records,
            checkpoint["vocabs"],
            metadata_mode=str(config["metadata_mode"]),
            weighting_mode=str(config["training_weighting_mode"]),
            target_scaler=target_scaler_from_checkpoint(checkpoint),
        )
        sampler = SourceGroupBatchSampler(
            records,
            int(config["batch_size"]),
            int(config["seed"]),
            False,
        )
        batches = evenly_spaced_batches(sampler, maximum_batches_per_task)
        loader = DataLoader(dataset, batch_sampler=batches, collate_fn=collate, num_workers=0)
        model.zero_grad(set_to_none=True)
        loss_values = []
        for raw_batch in loader:
            batch = _move(raw_batch, device)
            output = _forward(model, batch)
            loss = multitask_loss(
                output,
                batch,
                str(config["loss_kind"]).removeprefix("huber_plus_"),
                float(config["ranking_loss_weight"]),
                float(config["huber_delta"]),
            )
            _require(loss.is_cuda and loss.device == device, "task loss left CUDA")
            (loss / len(batches)).backward()
            loss_values.append(float(loss.detach().cpu()))
        vector = gradient_vector(parameters).detach()
        _require(vector.is_cuda and vector.device == device, "task gradient left CUDA")
        gradients[task] = vector
        task_rows[task] = {
            "training_record_count": len(records),
            "sampled_batch_count": len(batches),
            "sampled_record_count": sum(len(batch) for batch in batches),
            "mean_sampled_loss": sum(loss_values) / len(loss_values),
            "shared_gradient_norm": float(torch.linalg.vector_norm(vector)),
        }

    matrix = cosine_matrix(gradients)
    tasks = sorted(matrix)
    off_diagonal = [
        matrix[left][right]
        for left_index, left in enumerate(tasks)
        for right in tasks[left_index + 1 :]
    ]
    _require(all(math.isfinite(value) for value in off_diagonal), "cosine is not finite")
    result = {
        "schema_version": "route_a_v3_route2_task_gradient_conflict_diagnostic.v1",
        "status": "TRAIN_TASK_GRADIENT_MATRIX_COMPUTED_NOT_A_MODEL_GATE",
        "screen_adjudication_status": adjudication["status"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_baseline_id": checkpoint["baseline_id"],
        "training_config_path": str(training_config_path),
        "training_record_count": len(train_records),
        "development_test_outcomes_read": 0,
        "evaluation_outcomes_read": 0,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "physical_gpu_index": physical_gpu_index,
        "device": str(device),
        "cpu_fallback_used": False,
        "cuda_losses_verified": cuda_losses_verified,
        "shared_parameter_count": sum(parameter.numel() for _, parameter in parameters),
        "shared_parameter_names": [name for name, _ in parameters],
        "task_diagnostics": task_rows,
        "task_gradient_cosine": matrix,
        "negative_off_diagonal_pair_count": sum(value < 0.0 for value in off_diagonal),
        "off_diagonal_pair_count": len(off_diagonal),
        "minimum_off_diagonal_cosine": min(off_diagonal),
        "maximum_off_diagonal_cosine": max(off_diagonal),
        "scientific_interpretation": "REQUIRES_COMPARISON_TO_TASKS_THAT_REGRESSED_VS_GLOBAL_RAW;NO_AUTOMATIC_ARCHITECTURE_DECISION",
        **cuda_provenance,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--maximum-batches-per-task", type=int, default=4)
    args = parser.parse_args()
    failure_config = {
        "device": f"cuda:{args.physical_gpu_index}",
        "physical_gpu_index": args.physical_gpu_index,
        "evaluation_outcomes_accessed": False,
        "development_test_outcomes_accessed": False,
    }
    try:
        result = execute(
            training_config_path=args.training_config,
            checkpoint_path=args.checkpoint,
            adjudication_path=args.adjudication,
            output_path=args.output,
            physical_gpu_index=args.physical_gpu_index,
            maximum_batches_per_task=args.maximum_batches_per_task,
        )
    except Exception as exc:
        write_gpu_failure_evidence(
            args.output.with_name(args.output.name + ".failed.json"),
            failure_config,
            exc,
            entrypoint="diagnose_route2_task_gradient_conflict_v1",
            evaluation_outcomes_accessed=False,
        )
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
