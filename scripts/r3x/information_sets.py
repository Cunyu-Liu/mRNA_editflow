"""Immutable feature views enforcing six non-comparable information sets."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence


HEX64 = re.compile(r"^[0-9a-f]{64}$")
INFORMATION_SET_FINGERPRINT_SCHEMA = "r3_information_set_v1"


class InformationSet(str, Enum):
    NO_ANCHOR_ZERO_PRIOR = "NO_ANCHOR_ZERO_PRIOR"
    EDIT_ONLY = "EDIT_ONLY"
    CANDIDATE_ONLY = "CANDIDATE_ONLY"
    PAIRED_SEQUENCE_NO_MEASURED_ANCHOR = "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR"
    MEASURED_SOURCE_ANCHOR_ASSISTED = "MEASURED_SOURCE_ANCHOR_ASSISTED"
    FOUNDATION_EXPOSURE_STRATIFIED = "FOUNDATION_EXPOSURE_STRATIFIED"


ALLOWLISTS = {
    InformationSet.NO_ANCHOR_ZERO_PRIOR: (
        "endpoint", "publication_context_id", "region",
    ),
    InformationSet.EDIT_ONLY: (
        "endpoint", "publication_context_id", "region", "edit_list", "edit_count",
    ),
    InformationSet.CANDIDATE_ONLY: (
        "endpoint", "publication_context_id", "region", "normalized_candidate_sequence",
    ),
    InformationSet.PAIRED_SEQUENCE_NO_MEASURED_ANCHOR: (
        "endpoint", "publication_context_id", "region",
        "normalized_source_sequence", "normalized_candidate_sequence", "edit_list", "edit_count",
    ),
    InformationSet.MEASURED_SOURCE_ANCHOR_ASSISTED: (
        "endpoint", "publication_context_id", "region",
        "normalized_source_sequence", "normalized_candidate_sequence", "edit_list", "edit_count",
        "source_value", "source_anchor_status",
    ),
    InformationSet.FOUNDATION_EXPOSURE_STRATIFIED: (
        "endpoint", "publication_context_id", "region", "foundation_embedding",
        "foundation_model_id", "foundation_exposure_stratum",
    ),
}


class InformationSetError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class FoundationQualification:
    status: str
    license_status: str
    exposure_status: str
    exposure_audit_sha256: str
    license_audit_sha256: str
    feature_manifest_sha256: str


@dataclass(frozen=True)
class FeatureView:
    information_set: InformationSet
    values: Mapping[str, Any]
    provenance: Mapping[str, str]
    fingerprint: str


def information_set_allowlist(information_set: InformationSet | str) -> tuple[str, ...]:
    try:
        return ALLOWLISTS[InformationSet(information_set)]
    except (ValueError, KeyError) as exc:
        raise InformationSetError("HOLD_UNKNOWN_INFORMATION_SET", str(information_set)) from exc


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        # Tuple-of-pairs is deeply immutable and remains JSON serializable;
        # nested mappingproxy objects would make audit serialization fail.
        return tuple(
            (str(key), _freeze(child))
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    if isinstance(value, tuple):
        return tuple(_freeze(child) for child in value)
    return value


def _foundation_payload(qualification: FoundationQualification | None) -> dict:
    if qualification is None:
        raise InformationSetError("NOT_RUN_FOUNDATION_UNQUALIFIED", "foundation qualification is missing")
    if (
        qualification.status != "QUALIFIED"
        or qualification.license_status != "QUALIFIED"
        or qualification.exposure_status != "QUALIFIED"
    ):
        raise InformationSetError("NOT_RUN_FOUNDATION_UNQUALIFIED", "foundation exposure/license status is not QUALIFIED")
    payload = {
        "status": qualification.status,
        "license_status": qualification.license_status,
        "exposure_status": qualification.exposure_status,
        "exposure_audit_sha256": qualification.exposure_audit_sha256,
        "license_audit_sha256": qualification.license_audit_sha256,
        "feature_manifest_sha256": qualification.feature_manifest_sha256,
    }
    if any(not HEX64.fullmatch(str(payload[key])) for key in (
        "exposure_audit_sha256", "license_audit_sha256", "feature_manifest_sha256"
    )):
        raise InformationSetError("NOT_RUN_FOUNDATION_UNQUALIFIED", "foundation audit hashes are invalid")
    return payload


def information_set_fingerprint(
    information_set: InformationSet | str,
    *,
    foundation_qualification: FoundationQualification | None = None,
) -> str:
    """Return the canonical fingerprint of one frozen information-set definition.

    The digest is derived only from the registered information-set identity,
    ordered allowlist, and frozen leakage-boundary declarations.  Foundation
    fingerprints additionally bind the exact qualified exposure/license
    authorities.  It is never a caller-selected run identifier.
    """

    try:
        info = InformationSet(information_set)
    except ValueError as exc:
        raise InformationSetError(
            "HOLD_UNKNOWN_INFORMATION_SET", str(information_set)
        ) from exc
    payload: dict[str, Any] = {
        "schema_version": INFORMATION_SET_FINGERPRINT_SCHEMA,
        "information_set": info.value,
        "allowlist": ALLOWLISTS[info],
        "study_inference_input": False,
        "target_in_feature_view": False,
    }
    if info is InformationSet.FOUNDATION_EXPOSURE_STRATIFIED:
        payload["foundation_qualification"] = _foundation_payload(
            foundation_qualification
        )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_alias(
    row: Mapping[str, Any], canonical: str, aliases: Sequence[str], *, normalizer=None
) -> Any:
    present = [(name, row[name]) for name in aliases if name in row and row[name] is not None]
    if not present:
        return None
    normalized = [normalizer(value) if normalizer else value for _, value in present]
    if any(value != normalized[0] for value in normalized[1:]):
        raise InformationSetError(
            "HOLD_CONFLICTING_INFORMATION_ALIAS",
            f"{canonical} aliases disagree: {[name for name, _ in present]}",
        )
    return normalized[0]


def _normalize_sequence_alias(value: Any) -> str:
    if not isinstance(value, str):
        raise InformationSetError(
            "HOLD_INVALID_INFORMATION_FIELD", "sequence aliases must be strings"
        )
    normalized = "".join(value.upper().replace("T", "U").split())
    if not normalized or any(base not in "ACGU" for base in normalized):
        raise InformationSetError(
            "HOLD_INVALID_INFORMATION_FIELD", "sequence must normalize to A/C/G/U"
        )
    return normalized


def canonical_feature_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve aliases once and fail if any module could see a different value."""

    if not isinstance(row, Mapping):
        raise InformationSetError("HOLD_INVALID_FEATURE_ROW", "row must be an object")
    canonical = dict(row)
    canonical["endpoint"] = _canonical_alias(
        row, "endpoint", ("endpoint", "endpoint_id"), normalizer=lambda value: str(value)
    )
    canonical["publication_context_id"] = _canonical_alias(
        row,
        "publication_context_id",
        ("publication_context_id", "context_id_v2", "context_id"),
        normalizer=lambda value: str(value),
    )
    canonical["normalized_source_sequence"] = _canonical_alias(
        row,
        "normalized_source_sequence",
        ("normalized_source_sequence", "source_sequence"),
        normalizer=_normalize_sequence_alias,
    )
    canonical["normalized_candidate_sequence"] = _canonical_alias(
        row,
        "normalized_candidate_sequence",
        ("normalized_candidate_sequence", "candidate_sequence"),
        normalizer=_normalize_sequence_alias,
    )
    canonical["edit_count"] = _canonical_alias(
        row,
        "edit_count",
        ("edit_count", "actual_edit_count"),
        normalizer=lambda value: int(value),
    )
    return canonical


def build_feature_view(
    row: Mapping[str, Any],
    information_set: InformationSet | str,
    *,
    foundation_qualification: FoundationQualification | None = None,
) -> FeatureView:
    row = canonical_feature_row(row)
    info = InformationSet(information_set)
    if info is InformationSet.FOUNDATION_EXPOSURE_STRATIFIED:
        _foundation_payload(foundation_qualification)
    values = {}
    provenance = {}
    for field in ALLOWLISTS[info]:
        if field not in row or row[field] is None:
            raise InformationSetError("HOLD_MISSING_INFORMATION_FIELD", f"{info.value} requires {field}")
        value = row[field]
        if field in {"source_value"} and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
        ):
            raise InformationSetError("HOLD_INVALID_INFORMATION_FIELD", f"{field} must be finite")
        if field == "foundation_embedding":
            if not isinstance(value, (list, tuple)) or not value or any(
                isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item))
                for item in value
            ):
                raise InformationSetError("NOT_RUN_FOUNDATION_UNQUALIFIED", "foundation embedding is invalid")
        values[field] = _freeze(value)
        provenance[field] = f"ordinary_row:{field}"
    return FeatureView(
        information_set=info,
        values=MappingProxyType(values),
        provenance=MappingProxyType(provenance),
        fingerprint=information_set_fingerprint(
            info, foundation_qualification=foundation_qualification
        ),
    )


def assert_same_information_set(views: Sequence[FeatureView]) -> str:
    views = list(views)
    if not views:
        raise InformationSetError("HOLD_EMPTY_COMPARISON", "comparison has no feature views")
    fingerprints = {view.fingerprint for view in views}
    if len(fingerprints) != 1:
        raise InformationSetError("HOLD_UNEQUAL_INFORMATION_SET", "leaderboard mixes unequal information sets")
    return views[0].fingerprint
