"""Canonical evidence records and fail-closed MK0 gate aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import math
import re
from typing import Any, Iterable, Mapping, Optional


HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class GateResult:
    gate_id: str
    name: str
    passed: bool
    test_domain: str
    exhaustive_or_sampled: str
    sample_count: int
    dtype: str
    atol: Optional[float]
    rtol: Optional[float]
    seed: int
    failure_count: int
    denominator: int
    artifact_path: str
    artifact_sha256: str
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not self.gate_id
            or not self.name
            or isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count <= 0
        ):
            raise ValueError("gate identity and positive sample count are required")
        if (
            isinstance(self.denominator, bool)
            or not isinstance(self.denominator, int)
            or self.denominator <= 0
            or isinstance(self.failure_count, bool)
            or not isinstance(self.failure_count, int)
            or not 0 <= self.failure_count <= self.denominator
        ):
            raise ValueError("gate failure denominator is invalid")
        if self.passed != (self.failure_count == 0):
            raise ValueError("gate pass must equal zero failures")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("gate seed must be an integer")
        if HEX64.fullmatch(self.artifact_sha256) is None:
            raise ValueError("gate artifact requires a SHA-256 digest")
        artifact = Path(self.artifact_path)
        if (
            artifact.is_absolute()
            or ".." in artifact.parts
            or artifact.parts[:2] != ("artifacts", "mk0")
        ):
            raise ValueError("gate artifact path must be relative under artifacts/mk0")
        for name, tolerance in (("atol", self.atol), ("rtol", self.rtol)):
            if tolerance is not None and (
                isinstance(tolerance, bool)
                or not isinstance(tolerance, (int, float))
                or not math.isfinite(tolerance)
                or tolerance < 0.0
            ):
                raise ValueError(f"gate {name} must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "name": self.name,
            "passed": self.passed,
            "test_domain": self.test_domain,
            "exhaustive_or_sampled": self.exhaustive_or_sampled,
            "sample_count": self.sample_count,
            "dtype": self.dtype,
            "atol": self.atol,
            "rtol": self.rtol,
            "seed": self.seed,
            "failure_count": self.failure_count,
            "failure_denominator": self.denominator,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "metrics": dict(self.metrics),
        }


def _require_finite_json_metrics(value: Any, *, path: str = "metrics") -> None:
    """Reject metrics which cannot be represented as stable finite JSON."""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} requires non-empty string keys")
            _require_finite_json_metrics(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_json_metrics(item, path=f"{path}[{index}]")
        return
    raise ValueError(f"{path} contains unsupported type {type(value).__name__}")


def gate_result_from_runtime_binding(
    binding: Mapping[str, Any],
    gate_config: Mapping[str, Any],
    *,
    actual_artifact_sha256: str,
) -> GateResult:
    """Validate an observed gate binding against the frozen configuration.

    The finalizer uses this function instead of manufacturing successful gate
    records.  A binding is accepted only when its runtime metadata, failure
    accounting, support path and runner-recorded digest all match the frozen
    gate and the bytes observed by the finalizer.
    """

    if not isinstance(binding, Mapping):
        raise ValueError("runtime gate binding must be an object")
    required = {
        "gate_id",
        "name",
        "passed",
        "test_domain",
        "exhaustive_or_sampled",
        "sample_count",
        "dtype",
        "atol",
        "rtol",
        "seed",
        "failure_count",
        "failure_denominator",
        "artifact_path",
        "artifact_sha256",
        "metrics",
    }
    missing = sorted(required - set(binding))
    if missing:
        raise ValueError(f"runtime gate binding missing fields: {missing}")
    gate_id = str(gate_config.get("id", ""))
    exact_fields = {
        "gate_id": gate_id,
        "name": gate_config.get("name"),
        "test_domain": gate_config.get("domain"),
        "exhaustive_or_sampled": gate_config.get("coverage"),
        "dtype": str(gate_config.get("dtype")),
        "atol": gate_config.get("atol"),
        "rtol": gate_config.get("rtol"),
        "seed": gate_config.get("seed"),
        "artifact_path": gate_config.get("artifact_path"),
    }
    for key, expected in exact_fields.items():
        if binding[key] != expected:
            raise ValueError(
                f"gate {gate_id} runtime {key} drift: {binding[key]!r} != {expected!r}"
            )
    configured_count = gate_config.get("sample_count")
    if configured_count == "RUNTIME_REQUIRED":
        if (
            isinstance(binding["sample_count"], bool)
            or not isinstance(binding["sample_count"], int)
            or binding["sample_count"] <= 0
        ):
            raise ValueError(f"gate {gate_id} requires a positive runtime sample count")
    elif binding["sample_count"] != configured_count:
        raise ValueError(
            f"gate {gate_id} runtime sample-count drift: "
            f"{binding['sample_count']!r} != {configured_count!r}"
        )
    if not isinstance(binding["passed"], bool):
        raise ValueError(f"gate {gate_id} passed must be boolean")
    metrics = binding["metrics"]
    if not isinstance(metrics, Mapping) or not metrics:
        raise ValueError(f"gate {gate_id} requires non-empty observed metrics")
    _require_finite_json_metrics(metrics)
    if HEX64.fullmatch(str(binding["artifact_sha256"])) is None:
        raise ValueError(f"gate {gate_id} runner artifact digest is invalid")
    if binding["artifact_sha256"] != actual_artifact_sha256:
        raise ValueError(f"gate {gate_id} support artifact was substituted or tampered")
    return GateResult(
        gate_id=gate_id,
        name=str(binding["name"]),
        passed=binding["passed"],
        test_domain=str(binding["test_domain"]),
        exhaustive_or_sampled=str(binding["exhaustive_or_sampled"]),
        sample_count=binding["sample_count"],
        dtype=str(binding["dtype"]),
        atol=binding["atol"],
        rtol=binding["rtol"],
        seed=binding["seed"],
        failure_count=binding["failure_count"],
        denominator=binding["failure_denominator"],
        artifact_path=str(binding["artifact_path"]),
        artifact_sha256=str(binding["artifact_sha256"]),
        metrics=dict(metrics),
    )


def verify_bound_file(
    path: str | Path,
    *,
    expected_path: str | Path | None = None,
    expected_sha256: str,
    expected_size_bytes: int | None = None,
) -> dict[str, Any]:
    """Verify an exact path, size and SHA-256 binding and return its evidence."""

    target = Path(path).resolve(strict=True)
    if not target.is_file():
        raise ValueError(f"bound path is not a file: {target}")
    if expected_path is not None and target != Path(expected_path).resolve(strict=True):
        raise ValueError(f"bound file path substitution: {target} != {expected_path}")
    if HEX64.fullmatch(expected_sha256) is None:
        raise ValueError("bound file expected SHA-256 is invalid")
    size = target.stat().st_size
    if expected_size_bytes is not None and size != expected_size_bytes:
        raise ValueError(f"bound file size drift: {target}")
    observed = sha256_file(target)
    if observed != expected_sha256:
        raise ValueError(f"bound file SHA-256 drift: {target}")
    return {"path": str(target), "size_bytes": size, "sha256": observed}


def aggregate_acceptance(
    gates: Iterable[GateResult], *, run_id: str, goal_sha256: str
) -> dict[str, Any]:
    if not run_id.strip():
        raise ValueError("run_id is required")
    if not re.fullmatch(r"[0-9a-f]{64}", goal_sha256):
        raise ValueError("goal_sha256 must be a lowercase SHA-256 digest")
    records = list(gates)
    identifiers = [record.gate_id for record in records]
    if len(records) != 35 or identifiers != [f"M{i:02d}" for i in range(1, 36)]:
        raise ValueError("MK0 acceptance requires ordered gates M01..M35")
    passed = all(record.passed for record in records)
    return {
        "schema_version": "mk0_acceptance_v1",
        "run_id": run_id,
        "goal_sha256": goal_sha256,
        "evidence_level": "E0_MATH_ENGINEERING_ONLY",
        "status": (
            "PASS_E0_MATH_ENGINEERING_ONLY_READY_FOR_EF0"
            if passed
            else "FAILED_WITH_EVIDENCE"
        ),
        "pass": passed,
        "gate_count": len(records),
        "failed_gate_ids": [record.gate_id for record in records if not record.passed],
        "gates": [record.to_dict() for record in records],
        "scientific_claims": {
            "functional_improvement": False,
            "matched_budget_superiority": False,
            "paper_success": False,
        },
    }


def verify_artifact_binding(path: str | Path, expected_sha256: str) -> bool:
    return sha256_file(path) == expected_sha256
