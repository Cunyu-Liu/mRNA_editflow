"""Exact row schemas layered over the physical staged-disclosure boundary.

The physical disclosure module proves which file may be opened at each stage.
It intentionally does not decide which biological/model fields are allowed.
This module supplies that second boundary for the primary paired-sequence
ordinary benchmark.  Unknown fields never become model inputs merely because
an activation manifest called them a feature.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .features import normalize_sequence, validated_substitutions
from .predictor_provenance import PREDICTOR_INPUT_ROW_SCHEMA
from .protocol import ProtocolHoldError, canonical_sha256
from .staged_disclosure import DevelopmentDisclosure, FeatureDisclosure


TYPED_ROW_SCHEMA = "r3x_primary_paired_sequence_row_v1"
TYPED_FEATURE_STAGE_SCHEMA = "r3x_primary_paired_sequence_test_feature_row_v1"
FEATURE_ROW_ID_SCHEMA = "r3x_target_free_feature_row_identity_v1"
MAX_SEQUENCE_LENGTH = 512
PRIMARY_EDIT_COUNTS = frozenset({1, 3, 5})

# Exact authority feature schema.  The first seven fields are the only model
# information set.  The remaining four are audit/aggregation identities and
# must be stripped before predictor provenance is built.
MODEL_FEATURE_FIELDS = (
    "edit_count",
    "edit_list",
    "endpoint",
    "normalized_candidate_sequence",
    "normalized_source_sequence",
    "publication_context_id",
    "region",
)
AUDIT_FEATURE_FIELDS = (
    "estimand_id",
    "source_component_id",
    "source_group_id_v2",
    "study",
)
EXACT_FEATURE_FIELDS = tuple(sorted((*MODEL_FEATURE_FIELDS, *AUDIT_FEATURE_FIELDS)))
EXACT_OUTCOME_FIELDS = ("delta",)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class TypedStagedRowError(ProtocolHoldError):
    """A staged row is physically available but scientifically untyped."""


def _hold(code: str, message: str, **details: Any) -> None:
    raise TypedStagedRowError(code, message, details=details or None)


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _hold("HOLD_TYPED_ROW_SCHEMA", f"{label} must be non-empty canonical text")
    return value


def _row_id(value: object) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        _hold("HOLD_TYPED_ROW_IDENTITY", "row_id must be lowercase SHA-256")
    return value


def _strict_mapping(value: object, expected: set[str], *, label: str) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        _hold(
            "HOLD_TYPED_ROW_SCHEMA",
            f"{label} fields differ from the exact schema",
            expected=sorted(expected),
            actual=actual,
        )
    return value


def _normalized_feature_row(
    row_id: str,
    partition: str,
    features: object,
    *,
    delta: float | None,
    enforce_row_id: bool = True,
) -> dict:
    raw = _strict_mapping(features, set(EXACT_FEATURE_FIELDS), label="staged features")
    try:
        source = normalize_sequence(raw["normalized_source_sequence"])
        candidate = normalize_sequence(raw["normalized_candidate_sequence"])
    except (TypeError, ValueError) as exc:
        _hold("HOLD_TYPED_ROW_SEQUENCE", str(exc), row_id=row_id)
    if (
        raw["normalized_source_sequence"] != source
        or raw["normalized_candidate_sequence"] != candidate
    ):
        _hold(
            "HOLD_TYPED_ROW_SEQUENCE",
            "typed sequences must already be canonical uppercase RNA A/C/G/U",
            row_id=row_id,
        )
    if len(source) != len(candidate):
        _hold("HOLD_TYPED_ROW_EDIT_REPLAY", "source and candidate must be equal length")
    if len(source) > MAX_SEQUENCE_LENGTH:
        _hold(
            "HOLD_TYPED_ROW_SEQUENCE_LENGTH",
            "sequence exceeds the frozen maximum; truncation is prohibited",
            length=len(source),
        )
    edit_count = raw["edit_count"]
    if (
        isinstance(edit_count, bool)
        or not isinstance(edit_count, int)
        or edit_count not in PRIMARY_EDIT_COUNTS
    ):
        _hold(
            "HOLD_TYPED_ROW_EDIT_BUDGET",
            "primary typed rows require exactly 1, 3, or 5 substitutions",
        )
    raw_edits = raw["edit_list"]
    if not isinstance(raw_edits, list) or len(raw_edits) != edit_count:
        _hold(
            "HOLD_TYPED_ROW_EDIT_REPLAY",
            "edit_list must contain exactly edit_count records",
        )
    positions: list[int] = []
    for index, edit in enumerate(raw_edits):
        if not isinstance(edit, Mapping) or set(edit) != {"op", "pos", "ref", "alt"}:
            _hold(
                "HOLD_TYPED_ROW_EDIT_REPLAY",
                "each edit must use the exact op/pos/ref/alt schema",
                edit_index=index,
            )
        if edit["op"] != "SUB":
            _hold("HOLD_TYPED_ROW_EDIT_REPLAY", "edit op must be uppercase SUB")
        pos = edit["pos"]
        if isinstance(pos, bool) or not isinstance(pos, int):
            _hold("HOLD_TYPED_ROW_EDIT_REPLAY", "edit pos must be an integer")
        if (
            not isinstance(edit["ref"], str)
            or not isinstance(edit["alt"], str)
            or edit["ref"] not in "ACGU"
            or edit["alt"] not in "ACGU"
        ):
            _hold(
                "HOLD_TYPED_ROW_EDIT_REPLAY",
                "edit ref/alt must be canonical uppercase A/C/G/U",
            )
        positions.append(pos)
    if positions != sorted(set(positions)):
        _hold(
            "HOLD_TYPED_ROW_EDIT_REPLAY",
            "edit positions must be strictly increasing and unique",
        )
    normalized = {
        "schema_version": TYPED_ROW_SCHEMA,
        "r3_input_row_sha256": row_id,
        "partition": partition,
        "endpoint": _text(raw["endpoint"], label="endpoint"),
        "publication_context_id": _text(
            raw["publication_context_id"], label="publication_context_id"
        ),
        "region": _text(raw["region"], label="region"),
        "normalized_source_sequence": source,
        "normalized_candidate_sequence": candidate,
        "edit_list": raw["edit_list"],
        "edit_count": edit_count,
        "study": _text(raw["study"], label="study"),
        "estimand_id": _text(raw["estimand_id"], label="estimand_id"),
        "source_component_id": _text(
            raw["source_component_id"], label="source_component_id"
        ),
        "source_group_id_v2": _text(
            raw["source_group_id_v2"], label="source_group_id_v2"
        ),
    }
    try:
        edits = validated_substitutions(normalized)
    except (TypeError, ValueError) as exc:
        _hold("HOLD_TYPED_ROW_EDIT_REPLAY", str(exc), row_id=row_id)
    if len(edits) != edit_count:
        _hold("HOLD_TYPED_ROW_EDIT_REPLAY", "edit_count disagrees with exact replay")
    # Canonicalize edit aliases so later modules cannot interpret the same row
    # differently.  Only the exact four-key SUB schema is emitted.
    normalized["edit_list"] = [
        {"op": "SUB", "pos": pos, "ref": ref, "alt": alt}
        for pos, ref, alt in edits
    ]
    expected_row_id = canonical_sha256(
        {
            "schema_version": FEATURE_ROW_ID_SCHEMA,
            "partition": partition,
            "features": {
                field: normalized[field] for field in EXACT_FEATURE_FIELDS
            },
        }
    )
    if enforce_row_id and row_id != expected_row_id:
        _hold(
            "HOLD_TYPED_ROW_TARGET_FREE_IDENTITY",
            "row_id must be the canonical feature-only identity and must not depend on outcomes",
            declared=row_id,
            expected=expected_row_id,
        )
    normalized["r3_input_row_sha256"] = expected_row_id
    if delta is not None:
        normalized["delta"] = float(delta)
    return normalized


def target_free_feature_row_id(partition: str, features: Mapping) -> str:
    """Derive the sole staged row identity without reading any outcome.

    This helper is intended for the external authority-preparation process.
    It deliberately includes the partition and the complete exact feature/audit
    payload, but no delta, target, label, outcome, or caller-selected ID.
    """

    if partition not in {"train", "development", "internal_test"}:
        _hold("HOLD_TYPED_ROW_PARTITION", "unsupported feature-row partition")
    normalized = _normalized_feature_row(
        "0" * 64,
        partition,
        features,
        delta=None,
        enforce_row_id=False,
    )
    return normalized["r3_input_row_sha256"]


def _assert_activation_field_contract(
    feature_fields: Sequence[str], outcome_fields: Sequence[str]
) -> None:
    if tuple(feature_fields) != EXACT_FEATURE_FIELDS:
        _hold(
            "HOLD_TYPED_ROW_SCHEMA",
            "activation feature_fields differ from the primary typed schema",
            expected=list(EXACT_FEATURE_FIELDS),
            actual=list(feature_fields),
        )
    if tuple(outcome_fields) != EXACT_OUTCOME_FIELDS:
        _hold(
            "HOLD_TYPED_ROW_SCHEMA",
            "activation outcome_fields differ from the primary typed schema",
            expected=list(EXACT_OUTCOME_FIELDS),
            actual=list(outcome_fields),
        )


def _canonicalize_partition(
    rows: Sequence[Mapping], *, partition: str, with_outcome: bool
) -> tuple[dict, ...]:
    normalized: list[dict] = []
    for staged in rows:
        raw = _strict_mapping(
            staged,
            {"row_id", "partition", "features", "outcomes"}
            if with_outcome
            else {"row_id", "partition", "features"},
            label=f"{partition} staged row",
        )
        row_id = _row_id(raw["row_id"])
        if raw["partition"] != partition:
            _hold(
                "HOLD_TYPED_ROW_PARTITION",
                "typed row partition disagrees with its disclosure stage",
                row_id=row_id,
                expected=partition,
                actual=raw["partition"],
            )
        delta = None
        if with_outcome:
            outcomes = _strict_mapping(
                raw["outcomes"], set(EXACT_OUTCOME_FIELDS), label="staged outcomes"
            )
            value = outcomes["delta"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                _hold("HOLD_TYPED_ROW_OUTCOME", "delta must be finite")
            delta = float(value)
        normalized.append(
            _normalized_feature_row(
                row_id, partition, raw["features"], delta=delta
            )
        )
    if not normalized:
        _hold("HOLD_TYPED_ROW_EMPTY", f"{partition} typed rows are empty")
    normalized.sort(key=lambda row: row["r3_input_row_sha256"])
    ids = [row["r3_input_row_sha256"] for row in normalized]
    if len(ids) != len(set(ids)):
        _hold("HOLD_TYPED_ROW_IDENTITY", "typed row IDs are duplicated")
    return tuple(normalized)


@dataclass(frozen=True)
class TypedDevelopmentRows:
    train: tuple[dict, ...]
    development: tuple[dict, ...]
    train_row_set_sha256: str
    development_row_set_sha256: str
    feature_schema_sha256: str
    typed_rows_ready: bool = True
    internal_test_disclosed: bool = False
    sealed_access: bool = False


def normalize_development_disclosure(
    disclosure: DevelopmentDisclosure,
) -> TypedDevelopmentRows:
    """Validate and flatten only already-disclosed train/development rows.

    The physical disclosure remains ``training_ready=False`` until typed
    scientific-authority projections are separately verified.  This function
    proves only the row schema and never upgrades that capability by itself.
    """

    if not isinstance(disclosure, DevelopmentDisclosure):
        _hold("HOLD_TYPED_ROW_DISCLOSURE", "a DevelopmentDisclosure is required")
    if disclosure.training_ready is not False:
        _hold(
            "HOLD_TYPED_ROW_DISCLOSURE",
            "physical disclosure must not self-assert training readiness",
        )
    _assert_activation_field_contract(
        disclosure.activation.feature_fields,
        disclosure.activation.outcome_fields,
    )
    train = _canonicalize_partition(
        disclosure.train_rows, partition="train", with_outcome=True
    )
    development = _canonicalize_partition(
        disclosure.development_rows, partition="development", with_outcome=True
    )
    train_ids = [row["r3_input_row_sha256"] for row in train]
    development_ids = [row["r3_input_row_sha256"] for row in development]
    if set(train_ids).intersection(development_ids):
        _hold("HOLD_TYPED_ROW_IDENTITY", "train/development row IDs overlap")
    return TypedDevelopmentRows(
        train=train,
        development=development,
        train_row_set_sha256=canonical_sha256(sorted(train_ids)),
        development_row_set_sha256=canonical_sha256(sorted(development_ids)),
        feature_schema_sha256=canonical_sha256(
            {
                "schema_version": TYPED_ROW_SCHEMA,
                "model_feature_fields": list(MODEL_FEATURE_FIELDS),
                "audit_feature_fields": list(AUDIT_FEATURE_FIELDS),
                "outcome_fields": list(EXACT_OUTCOME_FIELDS),
                "max_sequence_length": MAX_SEQUENCE_LENGTH,
                "edit_counts": sorted(PRIMARY_EDIT_COUNTS),
                "edit_operation": "SUB",
            }
        ),
    )


def normalize_test_feature_disclosure(
    disclosure: FeatureDisclosure,
) -> tuple[dict, ...]:
    """Validate feature-only test rows after a canonical freeze commit."""

    if not isinstance(disclosure, FeatureDisclosure):
        _hold("HOLD_TYPED_ROW_DISCLOSURE", "a FeatureDisclosure is required")
    _assert_activation_field_contract(
        disclosure.activation.feature_fields,
        disclosure.activation.outcome_fields,
    )
    rows = _canonicalize_partition(
        disclosure.internal_test_rows,
        partition="internal_test",
        with_outcome=False,
    )
    return tuple(
        {**row, "schema_version": TYPED_FEATURE_STAGE_SCHEMA}
        for row in rows
    )


def predictor_input_from_typed_row(row: Mapping, *, include_training_target: bool) -> dict:
    """Project one typed row to the exact same-information-set predictor input."""

    expected = {
        "schema_version",
        "r3_input_row_sha256",
        "partition",
        *MODEL_FEATURE_FIELDS,
        *AUDIT_FEATURE_FIELDS,
    }
    expected.add("delta")
    raw = _strict_mapping(row, expected, label="typed predictor source row")
    expected_partition = "train" if include_training_target else "development"
    if raw["partition"] != expected_partition:
        _hold(
            "HOLD_TYPED_ROW_PARTITION",
            "predictor target exposure disagrees with the typed partition",
            partition=raw["partition"],
            include_training_target=include_training_target,
        )
    predictor_input = {field: raw[field] for field in MODEL_FEATURE_FIELDS}
    cell_material = "|".join(
        str(raw[field])
        for field in (
            "study",
            "estimand_id",
            "publication_context_id",
            "region",
            "edit_count",
        )
    )
    return {
        "schema_version": PREDICTOR_INPUT_ROW_SCHEMA,
        "row_id": raw["r3_input_row_sha256"],
        "partition": (
            "ORDINARY_TRAIN" if include_training_target else "ORDINARY_DEVELOPMENT"
        ),
        "estimand_id": raw["estimand_id"],
        "cell_id": "CELL_"
        + hashlib.sha256(cell_material.encode("utf-8")).hexdigest()[:24].upper(),
        "predictor_input": predictor_input,
        "training_target": float(raw["delta"]) if include_training_target else None,
    }


__all__ = [
    "AUDIT_FEATURE_FIELDS",
    "EXACT_FEATURE_FIELDS",
    "EXACT_OUTCOME_FIELDS",
    "FEATURE_ROW_ID_SCHEMA",
    "MAX_SEQUENCE_LENGTH",
    "MODEL_FEATURE_FIELDS",
    "PRIMARY_EDIT_COUNTS",
    "TYPED_FEATURE_STAGE_SCHEMA",
    "TYPED_ROW_SCHEMA",
    "TypedDevelopmentRows",
    "TypedStagedRowError",
    "normalize_development_disclosure",
    "normalize_test_feature_disclosure",
    "predictor_input_from_typed_row",
    "target_free_feature_row_id",
]
