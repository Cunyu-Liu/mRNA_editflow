"""Train-only robust target scaling for Route 2 Delta prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


TARGET_SCALING_NONE = "NONE"
TARGET_SCALING_TRAIN_TASK_ROBUST = "TRAIN_TASK_ROBUST"
TARGET_SCALING_MODES = {
    TARGET_SCALING_NONE,
    TARGET_SCALING_TRAIN_TASK_ROBUST,
}


def task_key(endpoint: str, region: int) -> str:
    return f"{endpoint}::region={int(region)}"


def region_key(region: int) -> str:
    return f"region={int(region)}"


def _robust_scale(values: Iterable[float], floor: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("target scale input must be non-empty and finite")
    median = float(np.median(array))
    mad_scale = 1.4826 * float(np.median(np.abs(array - median)))
    zero_anchored_scale = float(np.median(np.abs(array)))
    return max(mad_scale, zero_anchored_scale, floor)


@dataclass(frozen=True)
class Route2TargetScaler:
    """Multiplicative scaler fitted exclusively from training targets.

    Scaling never subtracts a center, so an observed or predicted zero Delta
    remains exactly zero.  Task-specific scales fall back to a train-derived
    region scale and then to a train-derived global scale for unseen tasks.
    """

    mode: str
    task_scales: Mapping[str, float]
    region_scales: Mapping[str, float]
    global_scale: float
    minimum_task_records: int
    floor: float
    training_record_count: int

    def __post_init__(self) -> None:
        if self.mode not in TARGET_SCALING_MODES:
            raise ValueError(f"unknown target scaling mode: {self.mode}")
        if self.minimum_task_records < 1:
            raise ValueError("minimum_task_records must be positive")
        if not np.isfinite(self.floor) or self.floor <= 0.0:
            raise ValueError("target scale floor must be positive and finite")
        if not np.isfinite(self.global_scale) or self.global_scale <= 0.0:
            raise ValueError("global target scale must be positive and finite")
        if self.training_record_count < 1:
            raise ValueError("target scaler training record count must be positive")
        for mapping in (self.task_scales, self.region_scales):
            if any(not np.isfinite(value) or value <= 0.0 for value in mapping.values()):
                raise ValueError("all target scales must be positive and finite")

    def scale(self, endpoint: str, region: int) -> tuple[float, str]:
        if self.mode == TARGET_SCALING_NONE:
            return 1.0, "NONE"
        task = task_key(endpoint, region)
        if task in self.task_scales:
            return float(self.task_scales[task]), "TASK"
        region_name = region_key(region)
        if region_name in self.region_scales:
            return float(self.region_scales[region_name]), "REGION_FALLBACK"
        return float(self.global_scale), "GLOBAL_FALLBACK"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "route_a_v3_route2_target_scaler.v1",
            "mode": self.mode,
            "task_scales": dict(sorted(self.task_scales.items())),
            "region_scales": dict(sorted(self.region_scales.items())),
            "global_scale": float(self.global_scale),
            "minimum_task_records": int(self.minimum_task_records),
            "floor": float(self.floor),
            "training_record_count": int(self.training_record_count),
            "center_subtracted": False,
            "fit_scope": "TRAIN_ONLY",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "Route2TargetScaler":
        if not payload:
            return cls(
                mode=TARGET_SCALING_NONE,
                task_scales={},
                region_scales={},
                global_scale=1.0,
                minimum_task_records=1,
                floor=1.0,
                training_record_count=1,
            )
        if payload.get("schema_version") != "route_a_v3_route2_target_scaler.v1":
            raise ValueError("unexpected target scaler schema")
        if payload.get("center_subtracted") is not False or payload.get("fit_scope") != "TRAIN_ONLY":
            raise ValueError("target scaler does not prove zero-preserving train-only fitting")
        return cls(
            mode=str(payload["mode"]),
            task_scales={str(key): float(value) for key, value in payload["task_scales"].items()},
            region_scales={str(key): float(value) for key, value in payload["region_scales"].items()},
            global_scale=float(payload["global_scale"]),
            minimum_task_records=int(payload["minimum_task_records"]),
            floor=float(payload["floor"]),
            training_record_count=int(payload["training_record_count"]),
        )


def fit_route2_target_scaler(
    records: Iterable[Any],
    *,
    mode: str,
    minimum_task_records: int = 20,
    floor: float = 1e-3,
) -> Route2TargetScaler:
    records = list(records)
    if not records:
        raise ValueError("cannot fit target scaler without training records")
    if mode not in TARGET_SCALING_MODES:
        raise ValueError(f"unknown target scaling mode: {mode}")
    if minimum_task_records < 1:
        raise ValueError("minimum_task_records must be positive")
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("target scale floor must be positive and finite")
    if mode == TARGET_SCALING_NONE:
        return Route2TargetScaler(
            mode=mode,
            task_scales={},
            region_scales={},
            global_scale=1.0,
            minimum_task_records=minimum_task_records,
            floor=floor,
            training_record_count=len(records),
        )

    by_task: dict[str, list[float]] = {}
    by_region: dict[str, list[float]] = {}
    targets = []
    for record in records:
        target = float(record.target)
        if not np.isfinite(target):
            raise ValueError("target scaler received a non-finite target")
        task = task_key(str(record.endpoint), int(record.region))
        region = region_key(int(record.region))
        by_task.setdefault(task, []).append(target)
        by_region.setdefault(region, []).append(target)
        targets.append(target)
    task_scales = {
        key: _robust_scale(values, floor)
        for key, values in by_task.items()
        if len(values) >= minimum_task_records
    }
    region_scales = {
        key: _robust_scale(values, floor)
        for key, values in by_region.items()
    }
    return Route2TargetScaler(
        mode=mode,
        task_scales=task_scales,
        region_scales=region_scales,
        global_scale=_robust_scale(targets, floor),
        minimum_task_records=minimum_task_records,
        floor=floor,
        training_record_count=len(records),
    )


def target_scaler_from_checkpoint(checkpoint: Mapping[str, Any]) -> Route2TargetScaler:
    return Route2TargetScaler.from_dict(checkpoint.get("target_scaler"))
