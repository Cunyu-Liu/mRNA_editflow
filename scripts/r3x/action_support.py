"""Frozen action-support qualification for the optional B interaction head."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from .protocol import ProtocolHoldError, assert_ordinary_path


SCHEMA = "r3_action_support_policy_v1"
REQUIRED_HASHES = (
    "ready_dataset_sha256",
    "inclusion_manifest_sha256",
    "context_source_crosswalk_sha256",
    "endpoint_estimand_registry_sha256",
    "source_component_split_manifest_sha256",
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ActionSupportHoldError(RuntimeError):
    def __init__(self, code: str, message: str, *, details: Mapping[str, Any] | None = None):
        self.code = code
        self.details = dict(details or {})
        super().__init__(f"{code}: {message}")


def _number(mapping: Mapping, key: str, *, integer: bool = False) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ActionSupportHoldError("NOT_RUN_ACTION_SUPPORT", f"{key} must be finite")
    if integer and (int(value) != float(value) or int(value) < 1):
        raise ActionSupportHoldError("NOT_RUN_ACTION_SUPPORT", f"{key} must be a positive integer")
    return float(value)


def validate_action_support_policy(payload: Mapping[str, Any]) -> dict:
    if not isinstance(payload, Mapping):
        raise ActionSupportHoldError("NOT_RUN_ACTION_SUPPORT", "action-support policy must be an object")
    if payload.get("schema_version") != SCHEMA:
        raise ActionSupportHoldError("NOT_RUN_ACTION_SUPPORT", "unsupported action-support schema")
    if payload.get("status") != "FROZEN":
        raise ActionSupportHoldError("NOT_RUN_ACTION_SUPPORT", "action-support policy must be FROZEN")
    if payload.get("qualification_status") != "PUBLICATION_QUALIFIED":
        raise ActionSupportHoldError(
            "NOT_RUN_ACTION_SUPPORT",
            "interaction requires publication qualification, not DEV_AVAILABLE",
        )
    hashes = payload.get("authority_hashes")
    if not isinstance(hashes, Mapping):
        raise ActionSupportHoldError("NOT_RUN_ACTION_SUPPORT", "policy lacks authority_hashes")
    for name in REQUIRED_HASHES:
        if not HEX64.fullmatch(str(hashes.get(name) or "")):
            raise ActionSupportHoldError("NOT_RUN_ACTION_SUPPORT", f"invalid or missing {name}")
    thresholds = payload.get("thresholds")
    observed = payload.get("observed")
    if not isinstance(thresholds, Mapping) or not isinstance(observed, Mapping):
        raise ActionSupportHoldError("NOT_RUN_ACTION_SUPPORT", "policy lacks frozen thresholds/observations")
    min_components = _number(thresholds, "min_independent_source_components_per_action", integer=True)
    min_cells = _number(thresholds, "min_study_context_cells_per_action", integer=True)
    max_dominance = _number(thresholds, "max_single_component_dominance")
    actual_components = _number(observed, "min_independent_source_components_per_action", integer=True)
    actual_cells = _number(observed, "min_study_context_cells_per_action", integer=True)
    actual_dominance = _number(observed, "max_single_component_dominance")
    strata = _number(observed, "qualified_action_strata", integer=True)
    if not 0.0 <= max_dominance < 1.0 or not 0.0 <= actual_dominance <= 1.0:
        raise ActionSupportHoldError("NOT_RUN_ACTION_SUPPORT", "dominance values must be in [0,1)")
    failures = {}
    if actual_components < min_components:
        failures["independent_source_components"] = [actual_components, min_components]
    if actual_cells < min_cells:
        failures["study_context_cells"] = [actual_cells, min_cells]
    if actual_dominance > max_dominance:
        failures["single_component_dominance"] = [actual_dominance, max_dominance]
    if strata < 1:
        failures["qualified_action_strata"] = [strata, 1]
    if failures:
        raise ActionSupportHoldError(
            "NOT_RUN_ACTION_SUPPORT",
            "observed action support does not meet the frozen policy",
            details=failures,
        )
    result = copy.deepcopy(dict(payload))
    result["status"] = "PUBLICATION_QUALIFIED"
    result["interaction_training_allowed"] = True
    result["sealed_access"] = False
    return result


def load_action_support_policy(path: Path, expected_sha256: str) -> dict:
    path = Path(path)
    try:
        assert_ordinary_path(path, purpose="action-support policy")
    except ProtocolHoldError as exc:
        raise ActionSupportHoldError("NOT_RUN_ACTION_SUPPORT", str(exc)) from exc
    expected = str(expected_sha256 or "").lower()
    if not HEX64.fullmatch(expected):
        raise ActionSupportHoldError("NOT_RUN_ACTION_SUPPORT", "expected SHA-256 is invalid")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ActionSupportHoldError("NOT_RUN_ACTION_SUPPORT", f"policy bytes unavailable: {exc}") from exc
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ActionSupportHoldError(
            "NOT_RUN_ACTION_SUPPORT",
            f"action-support SHA-256 mismatch: expected {expected}, got {actual}",
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ActionSupportHoldError("NOT_RUN_ACTION_SUPPORT", f"policy is not JSON: {exc}") from exc
    result = validate_action_support_policy(payload)
    result["policy_sha256"] = actual
    result["policy_path"] = str(path)
    return result
