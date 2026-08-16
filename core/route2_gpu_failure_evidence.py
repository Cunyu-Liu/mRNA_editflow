"""Minimal fail-closed evidence writer shared by Route 2 CUDA entrypoints."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import torch


def _physical_gpu_uuid(device_index: int) -> str:
    completed = subprocess.run(
        [
            "nvidia-smi", "--id", str(device_index), "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"physical GPU UUID lookup returned {len(lines)} rows")
    return lines[0].removeprefix("GPU-")


def cuda_device_observation(
    device_index: int | None,
    *,
    require_physical_index_match: bool = False,
) -> dict[str, Any]:
    """Record the CUDA device that PyTorch actually sees at an index."""

    result: dict[str, Any] = {
        "cuda_device_index": device_index,
        "cuda_device_name": None,
        "cuda_device_uuid": None,
        "cuda_total_memory_mb": None,
        "cuda_free_memory_mb_at_observation": None,
        "declared_physical_gpu_uuid": None,
        "cuda_parent_uuid_matches_declared_physical_index": None,
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
        physical_uuid = _physical_gpu_uuid(device_index)
        result["declared_physical_gpu_uuid"] = physical_uuid
        result["cuda_parent_uuid_matches_declared_physical_index"] = (
            str(getattr(properties, "uuid", "UNKNOWN")).removeprefix("GPU-") == physical_uuid
        )
    except Exception as observation_error:
        result["cuda_device_observation_error"] = (
            f"{type(observation_error).__name__}: {observation_error}"
        )
    if require_physical_index_match and result["cuda_parent_uuid_matches_declared_physical_index"] is not True:
        detail = result.get("cuda_device_observation_error", "UUID_MISMATCH")
        raise RuntimeError(
            f"CUDA logical device {device_index} does not belong to physical GPU {device_index}: {detail}"
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
