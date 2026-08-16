"""Minimal fail-closed evidence writer shared by Route 2 CUDA entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch


def cuda_device_observation(device_index: int | None) -> dict[str, Any]:
    """Record the CUDA device that PyTorch actually sees at an index."""

    result: dict[str, Any] = {
        "cuda_device_index": device_index,
        "cuda_device_name": None,
        "cuda_device_uuid": None,
        "cuda_total_memory_mb": None,
        "cuda_free_memory_mb_at_observation": None,
    }
    if not torch.cuda.is_available() or device_index is None:
        return result
    if not 0 <= device_index < torch.cuda.device_count():
        return result
    try:
        properties = torch.cuda.get_device_properties(device_index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        result.update({
            "cuda_device_name": properties.name,
            "cuda_device_uuid": str(getattr(properties, "uuid", "UNKNOWN")),
            "cuda_total_memory_mb": total_bytes / 1024**2,
            "cuda_free_memory_mb_at_observation": free_bytes / 1024**2,
        })
    except Exception as observation_error:
        result["cuda_device_observation_error"] = (
            f"{type(observation_error).__name__}: {observation_error}"
        )
    return result


def write_gpu_failure_evidence(
    evidence_path: Path,
    config: Mapping[str, Any],
    error: BaseException,
    *,
    entrypoint: str,
    evaluation_outcomes_accessed: Any,
) -> None:
    requested_index = config.get("physical_gpu_index")
    requested_index = requested_index if isinstance(requested_index, int) else None
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with evidence_path.open("x", encoding="utf-8") as handle:
        json.dump({
            "status": "STOPPED_WITH_EVIDENCE",
            "entrypoint": entrypoint,
            "error_type": type(error).__name__,
            "error": str(error),
            "requested_device": config.get("device"),
            "physical_gpu_index": config.get("physical_gpu_index"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "requested_cuda_observation": cuda_device_observation(requested_index),
            "cpu_fallback_used": False,
            "evaluation_outcomes_accessed": evaluation_outcomes_accessed,
        }, handle, indent=2, sort_keys=True)
        handle.write("\n")
