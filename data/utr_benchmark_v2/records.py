"""Fail-closed canonical UTR benchmark records for the D1/B0 contract."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from math import isclose, isfinite
from typing import Any, Mapping, Sequence

from .edit_script import (
    EditAction,
    EditScriptError,
    apply_edit_script,
    canonicalize_edit_script,
)


REGIONS = frozenset({"five_utr", "three_utr"})
PAIR_TYPES = frozenset(
    {
        "true_wt_mutant",
        "dense_measured_neighbor",
        "measured_multi_edit_family",
        "measured_indel_pair",
        "absolute_property_only",
        "unlabeled_pretraining",
        "retrospective_constructed_neighbor",
    }
)
ABSOLUTE_PAIR_TYPES = frozenset(
    {"absolute_property_only", "unlabeled_pretraining"}
)
MEASURED_PAIR_TYPES = frozenset(
    {
        "true_wt_mutant",
        "dense_measured_neighbor",
        "measured_multi_edit_family",
        "measured_indel_pair",
    }
)
TRAJECTORY_SOURCES = frozenset({"latent", "constructed", "observed"})
COUPLING_TYPES = frozenset(
    {
        "observed_endpoint_coupling",
        "constructed_alignment_coupling",
        "corruption_denoising_coupling",
        "dense_landscape_coupling",
        "property_conditioned_target_coupling",
    }
)

REQUIRED_FIELDS = (
    "record_id",
    "dataset_id",
    "study_id",
    "assay_id",
    "context_id",
    "evidence_grade",
    "exposure_grade",
    "region",
    "organism",
    "cell_context",
    "reporter",
    "cargo",
    "endpoint",
    "endpoint_provenance",
    "timepoint",
    "source_id",
    "source_sequence",
    "candidate_sequence",
    "source_length",
    "candidate_length",
    "edit_script",
    "edit_types",
    "edit_positions",
    "reference_alleles",
    "alternate_alleles",
    "edit_count",
    "edit_distance",
    "source_value_raw",
    "candidate_value_raw",
    "delta_raw",
    "delta_normalized",
    "effect_standard_error",
    "replicate_count",
    "pair_type",
    "trajectory_observed",
    "trajectory_source",
    "trajectory_provenance",
    "coupling_type",
    "paper_split",
    "canonical_split",
    "source_group",
    "gene_group",
    "study_group",
    "context_group",
    "sequence_cluster",
    "scaffold_group",
    "barcode_batch",
    "library_batch",
    "sequence_provenance",
    "label_provenance",
    "download_manifest",
    "license",
    "quality_flags",
    "historical_exposure",
)


class CanonicalRecordError(ValueError):
    """Raised when a record violates the frozen D1 canonical schema."""


_BARE_PLACEHOLDERS = frozenset(
    {
        "",
        "UNKNOWN",
        "NA",
        "N/A",
        "N.A.",
        "NONE",
        "NULL",
        "TBD",
        "MISSING",
        "UNAVAILABLE",
        "UNRESOLVED",
        "NOT_AVAILABLE",
        "NOT_APPLICABLE",
        "NOT_OPENED_OR_NOT_APPLICABLE",
        "NOT_PROVIDED",
        "NOT_REPORTED",
    }
)


def _is_bare_placeholder(value: str) -> bool:
    return value.strip().upper().replace(" ", "_") in _BARE_PLACEHOLDERS


def _nonempty_string(payload: Mapping[str, Any], field: str) -> None:
    if not isinstance(payload[field], str) or not payload[field].strip():
        raise CanonicalRecordError(f"{field} must be a non-empty string")


def _meaningful_string(payload: Mapping[str, Any], field: str) -> None:
    _nonempty_string(payload, field)
    if _is_bare_placeholder(payload[field]):
        raise CanonicalRecordError(
            f"{field} cannot be a bare UNKNOWN/NA placeholder"
        )


def _contains_placeholder_marker(value: str) -> bool:
    if _is_bare_placeholder(value):
        return True
    return any(
        segment and _is_bare_placeholder(segment)
        for segment in value.split(":")
    )


def _provenance_string(payload: Mapping[str, Any], field: str) -> None:
    _nonempty_string(payload, field)
    if _contains_placeholder_marker(payload[field]):
        raise CanonicalRecordError(
            f"{field} cannot contain an UNKNOWN/NA placeholder"
        )


def _scoped_nonplaceholder_string(
    payload: Mapping[str, Any], field: str
) -> None:
    _meaningful_string(payload, field)
    value = payload[field]
    parts = value.split(":")
    if (
        value != value.strip()
        or any(character.isspace() for character in value)
        or len(parts) < 2
        or any(not part for part in parts)
    ):
        raise CanonicalRecordError(
            f"{field} must be a scoped non-empty identifier such as namespace:value"
        )


def _nonempty_mapping(payload: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = payload[field]
    if not isinstance(value, Mapping) or not value:
        raise CanonicalRecordError(f"{field} must be a non-empty mapping")
    return value


def _has_meaningful_provenance_value(value: Any) -> bool:
    if isinstance(value, str):
        return not _contains_placeholder_marker(value)
    if isinstance(value, Mapping):
        return bool(value) and any(
            _has_meaningful_provenance_value(item) for item in value.values()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return bool(value) and any(
            _has_meaningful_provenance_value(item) for item in value
        )
    if isinstance(value, bool):
        return False
    return value is not None


def _is_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return isfinite(float(value))
    except OverflowError:
        return False


def _check_endpoint(payload: Mapping[str, Any]) -> None:
    _nonempty_string(payload, "endpoint")
    endpoint = payload["endpoint"].strip().lower().replace("-", "_").replace(" ", "_")
    if any(separator in payload["endpoint"] for separator in ("|", ";", ",", "+")):
        raise CanonicalRecordError(
            "endpoint must name one endpoint; combined endpoint labels are forbidden"
        )
    if endpoint in {
        "expression",
        "expression_score",
        "unified_expression",
        "combined_expression",
    }:
        raise CanonicalRecordError(
            "endpoint cannot use a biologically undefined unified expression label"
        )
    provenance = _nonempty_mapping(payload, "endpoint_provenance")
    for field in ("raw_endpoint", "transformation"):
        if not isinstance(provenance.get(field), str) or not provenance[field].strip():
            raise CanonicalRecordError(
                f"endpoint_provenance must contain non-empty {field}"
            )


def _check_provenance(payload: Mapping[str, Any]) -> None:
    sequence = _nonempty_mapping(payload, "sequence_provenance")
    raw_values = [
        value
        for key, value in sequence.items()
        if str(key).lower().startswith("raw")
    ]
    processed_values = [
        value
        for key, value in sequence.items()
        if str(key).lower().startswith("processed")
    ]
    if not any(_has_meaningful_provenance_value(value) for value in raw_values):
        raise CanonicalRecordError(
            "sequence_provenance raw provenance cannot be empty or UNKNOWN/NA"
        )
    if not any(
        _has_meaningful_provenance_value(value)
        for value in processed_values
    ):
        raise CanonicalRecordError(
            "sequence_provenance processed provenance cannot be empty or UNKNOWN/NA"
        )

    manifest = payload["download_manifest"]
    if not (
        (
            isinstance(manifest, str)
            and _has_meaningful_provenance_value(manifest)
        )
        or (
            isinstance(manifest, Mapping)
            and _has_meaningful_provenance_value(manifest)
        )
    ):
        raise CanonicalRecordError(
            "download_manifest cannot be empty or an UNKNOWN/NA placeholder"
        )

    values = (
        payload["source_value_raw"],
        payload["candidate_value_raw"],
        payload["delta_raw"],
        payload["delta_normalized"],
    )
    if any(value is not None for value in values):
        _nonempty_mapping(payload, "label_provenance")
    elif not isinstance(payload["label_provenance"], Mapping):
        raise CanonicalRecordError("label_provenance must be a mapping")


def _check_script(payload: Mapping[str, Any]) -> dict[str, Any]:
    source = payload["source_sequence"]
    candidate = payload["candidate_sequence"]
    if (
        not isinstance(source, str)
        or not source
        or not isinstance(candidate, str)
        or not candidate
    ):
        raise CanonicalRecordError(
            "intervention source_sequence and candidate_sequence must be non-empty strings"
        )
    try:
        canonical = canonicalize_edit_script(source, candidate)
        if not isinstance(payload["edit_script"], list):
            raise CanonicalRecordError("edit_script must be a list")
        actions = [EditAction.from_dict(action) for action in payload["edit_script"]]
        if any(action.op == "STOP" for action in actions):
            raise CanonicalRecordError(
                "canonical endpoint edit_script must omit trajectory STOP"
            )
        if apply_edit_script(source, actions) != candidate:
            raise CanonicalRecordError(
                "applying edit_script does not reproduce candidate_sequence"
            )
    except EditScriptError as exc:
        raise CanonicalRecordError(f"invalid edit_script: {exc}") from exc

    supplied = [action.to_dict() for action in actions]
    if supplied != canonical["actions"]:
        raise CanonicalRecordError(
            "edit_script is not the deterministic canonical minimum-edit script"
        )

    expected = {
        "source_length": len(source),
        "candidate_length": len(candidate),
        "edit_count": len(actions),
        "edit_distance": canonical["minimal_edit_count"],
        "edit_types": [action.op for action in actions],
        "edit_positions": [action.pos for action in actions],
        "reference_alleles": [action.ref for action in actions],
        "alternate_alleles": [action.alt for action in actions],
    }
    for field, value in expected.items():
        if payload[field] != value:
            raise CanonicalRecordError(
                f"{field} does not match the canonical source/candidate mapping"
            )
    return canonical


def _check_absolute_representation(payload: Mapping[str, Any]) -> dict[str, Any]:
    candidate = payload["candidate_sequence"]
    if not isinstance(candidate, str) or not candidate:
        raise CanonicalRecordError(
            "absolute candidate_sequence must be a non-empty string"
        )
    try:
        canonical = canonicalize_edit_script(candidate, candidate)
    except EditScriptError as exc:
        raise CanonicalRecordError(
            f"invalid absolute candidate_sequence: {exc}"
        ) from exc

    null_fields = (
        "source_id",
        "source_sequence",
        "source_length",
        "edit_script",
        "edit_distance",
    )
    if any(payload[field] is not None for field in null_fields):
        raise CanonicalRecordError(
            "absolute record must keep source, edit_script, and edit_distance null "
            "(never synthesize an intervention anchor)"
        )
    if payload["candidate_length"] != len(candidate):
        raise CanonicalRecordError(
            "candidate_length does not match the absolute candidate_sequence"
        )
    empty_fields = (
        "edit_types",
        "edit_positions",
        "reference_alleles",
        "alternate_alleles",
    )
    if any(payload[field] != [] for field in empty_fields):
        raise CanonicalRecordError(
            "absolute record edit-derived lists must be empty"
        )
    if payload["edit_count"] != 0:
        raise CanonicalRecordError("absolute record edit_count must be zero")
    return canonical


def _check_pair_semantics(payload: Mapping[str, Any], canonical: Mapping[str, Any]) -> None:
    pair_type = payload["pair_type"]
    if pair_type not in PAIR_TYPES:
        raise CanonicalRecordError(f"pair_type must be one of {sorted(PAIR_TYPES)}")

    source = payload["source_sequence"]
    candidate = payload["candidate_sequence"]
    if pair_type in ABSOLUTE_PAIR_TYPES:
        if (
            source is not None
            or canonical["actions"]
            or payload["edit_count"] != 0
            or payload["edit_distance"] is not None
        ):
            raise CanonicalRecordError(
                "absolute pair cannot carry source-to-candidate intervention edits"
            )
        if payload["source_value_raw"] is not None:
            raise CanonicalRecordError(
                "absolute pair cannot invent a measured source value"
            )
        if payload["delta_raw"] is not None or payload["delta_normalized"] is not None:
            raise CanonicalRecordError(
                "absolute pair cannot carry an intervention delta"
            )
        if (
            payload["trajectory_source"] != "latent"
            or payload["trajectory_observed"] is not False
        ):
            raise CanonicalRecordError(
                "absolute pair must use a latent, unobserved trajectory"
            )
    elif source == candidate or not canonical["actions"]:
        raise CanonicalRecordError(
            "intervention pair must have distinct source/candidate and non-empty edits"
        )


def _check_labels(payload: Mapping[str, Any]) -> None:
    source_value = payload["source_value_raw"]
    candidate_value = payload["candidate_value_raw"]
    delta = payload["delta_raw"]

    for field in (
        "source_value_raw",
        "candidate_value_raw",
        "delta_raw",
        "delta_normalized",
        "effect_standard_error",
    ):
        if payload[field] is not None and not _is_number(payload[field]):
            raise CanonicalRecordError(f"{field} must be finite numeric or null")

    if source_value is not None and candidate_value is not None:
        expected = float(candidate_value) - float(source_value)
        if delta is None or not isclose(
            float(delta), expected, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise CanonicalRecordError(
                f"delta_raw must equal candidate_value_raw - source_value_raw ({expected})"
            )
    elif delta is not None:
        raise CanonicalRecordError(
            "delta_raw requires both source_value_raw and candidate_value_raw"
        )

    if payload["pair_type"] in MEASURED_PAIR_TYPES and (
        source_value is None or candidate_value is None or delta is None
    ):
        raise CanonicalRecordError(
            f"{payload['pair_type']} requires paired measured values and delta_raw"
        )
    if (
        payload["pair_type"] == "absolute_property_only"
        and candidate_value is None
    ):
        raise CanonicalRecordError(
            "absolute_property_only requires a measured candidate_value_raw"
        )
    if (
        payload["pair_type"] == "unlabeled_pretraining"
        and candidate_value is not None
    ):
        raise CanonicalRecordError(
            "unlabeled_pretraining cannot carry candidate_value_raw"
        )
    if payload["delta_normalized"] is not None and delta is None:
        raise CanonicalRecordError("delta_normalized requires delta_raw")
    if (
        payload["effect_standard_error"] is not None
        and payload["effect_standard_error"] < 0
    ):
        raise CanonicalRecordError("effect_standard_error cannot be negative")
    replicate_count = payload["replicate_count"]
    if replicate_count is not None and (
        isinstance(replicate_count, bool)
        or not isinstance(replicate_count, int)
        or replicate_count < 0
    ):
        raise CanonicalRecordError("replicate_count must be a non-negative integer or null")


def _check_trajectory(payload: Mapping[str, Any]) -> None:
    source = payload["trajectory_source"]
    observed = payload["trajectory_observed"]
    coupling = payload["coupling_type"]
    if source not in TRAJECTORY_SOURCES:
        raise CanonicalRecordError(
            f"trajectory_source must be one of {sorted(TRAJECTORY_SOURCES)}"
        )
    if not isinstance(observed, bool):
        raise CanonicalRecordError("trajectory_observed must be bool")
    if coupling not in COUPLING_TYPES:
        raise CanonicalRecordError(
            f"coupling_type must be one of {sorted(COUPLING_TYPES)}"
        )
    if source == "constructed" and observed:
        raise CanonicalRecordError(
            "constructed trajectory cannot be marked observed"
        )
    if source == "observed":
        if not observed:
            raise CanonicalRecordError(
                "observed trajectory_source requires trajectory_observed=true"
            )
        _nonempty_mapping(payload, "trajectory_provenance")
        if coupling != "observed_endpoint_coupling":
            raise CanonicalRecordError(
                "observed trajectory requires observed_endpoint_coupling"
            )
    else:
        if observed:
            raise CanonicalRecordError(
                f"{source} trajectory cannot be marked observed"
            )
        if not isinstance(payload["trajectory_provenance"], Mapping):
            raise CanonicalRecordError("trajectory_provenance must be a mapping")
    if source == "constructed" and coupling == "observed_endpoint_coupling":
        raise CanonicalRecordError(
            "constructed trajectory cannot use observed_endpoint_coupling"
        )
    compatible_couplings = {
        "observed": {"observed_endpoint_coupling"},
        "constructed": {
            "constructed_alignment_coupling",
            "dense_landscape_coupling",
        },
        "latent": {
            "corruption_denoising_coupling",
            "property_conditioned_target_coupling",
        },
    }
    if coupling not in compatible_couplings[source]:
        raise CanonicalRecordError(
            f"{source} trajectory_source is incompatible with coupling_type {coupling}"
        )


def validate_canonical_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a defensive copy of one canonical D1 record."""

    if not isinstance(payload, Mapping):
        raise CanonicalRecordError("canonical record must be a mapping")
    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        raise CanonicalRecordError(
            "canonical record missing required fields: " + ", ".join(missing)
        )

    for field in (
        "record_id",
        "dataset_id",
        "study_id",
        "assay_id",
        "context_id",
        "evidence_grade",
        "exposure_grade",
        "organism",
    ):
        _nonempty_string(payload, field)
    _provenance_string(payload, "license")
    _provenance_string(payload, "historical_exposure")
    for field in ("scaffold_group", "barcode_batch", "library_batch"):
        _scoped_nonplaceholder_string(payload, field)
    if payload["region"] not in REGIONS:
        raise CanonicalRecordError(f"region must be one of {sorted(REGIONS)}")
    if not isinstance(payload["quality_flags"], list) or not all(
        isinstance(flag, str) and flag for flag in payload["quality_flags"]
    ):
        raise CanonicalRecordError("quality_flags must be a list of non-empty strings")

    _check_endpoint(payload)
    if payload["pair_type"] not in PAIR_TYPES:
        raise CanonicalRecordError(
            f"pair_type must be one of {sorted(PAIR_TYPES)}"
        )
    if payload["pair_type"] in ABSOLUTE_PAIR_TYPES:
        canonical = _check_absolute_representation(payload)
    else:
        _nonempty_string(payload, "source_id")
        canonical = _check_script(payload)
    _check_pair_semantics(payload, canonical)
    _check_labels(payload)
    _check_trajectory(payload)
    _check_provenance(payload)
    return deepcopy(dict(payload))


def canonical_record_id(payload: Mapping[str, Any]) -> str:
    """Return a deterministic content id, excluding any existing record id."""

    material = deepcopy(dict(payload))
    material["record_id"] = ""
    try:
        encoded = json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CanonicalRecordError(
            f"record is not canonical-JSON serializable: {exc}"
        ) from exc
    return "utr2:" + hashlib.sha256(encoded).hexdigest()[:24]


@dataclass(frozen=True, init=False)
class CanonicalUTRRecord:
    """Validated wrapper around a defensively copied canonical-record mapping."""

    _payload: Mapping[str, Any]

    def __init__(self, payload: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_payload", validate_canonical_record(payload))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CanonicalUTRRecord":
        return cls(payload)

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(dict(self._payload))

    def __getattr__(self, name: str) -> Any:
        payload = object.__getattribute__(self, "_payload")
        try:
            return deepcopy(payload[name])
        except KeyError as exc:
            raise AttributeError(name) from exc


CanonicalRecord = CanonicalUTRRecord


__all__ = [
    "ABSOLUTE_PAIR_TYPES",
    "COUPLING_TYPES",
    "MEASURED_PAIR_TYPES",
    "PAIR_TYPES",
    "REGIONS",
    "REQUIRED_FIELDS",
    "TRAJECTORY_SOURCES",
    "CanonicalRecord",
    "CanonicalRecordError",
    "CanonicalUTRRecord",
    "canonical_record_id",
    "validate_canonical_record",
]
