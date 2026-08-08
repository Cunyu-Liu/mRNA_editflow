"""Exact row-level projections for outcome-sensitive R3X authorities.

The staged-disclosure layer controls which physical data authority may be
opened.  :mod:`typed_staged_rows` controls the exact scientific row schema.
This module closes the remaining boundary for the three non-data authorities
whose historical full forms could encode row or outcome support:

* the inclusion manifest;
* the context/source crosswalk; and
* the source-component split manifest.

Only stage-specific, feature-only commitments are emitted.  Development
projections cover ordinary train plus development rows.  Internal-test-feature
projections cover only already disclosed feature rows.  No projection can
carry a target, label, outcome, sequence, filesystem path, generic payload, or
sealed/downstream authorization.

A pre-training firewall certificate is useful only after exact replay by a
verifier in an explicit trusted registry.  Merely supplying a certificate (or
setting a readiness flag in JSON) never upgrades capability.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import marshal
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType
from typing import Any, Callable, Mapping, Sequence

from .artifact_io import (
    ArtifactIOError,
    atomic_write_json_exclusive,
    canonical_json_bytes,
    read_exact_regular_file,
)
from .features import normalize_sequence, validated_substitutions
from .protocol import ProtocolHoldError, canonical_sha256
from .splits import (
    SPLIT_ASSIGNMENT_ALGORITHM_ID,
    SPLIT_ASSIGNMENT_PARAMETERS,
)
from .typed_safe_authorities import DEVELOPMENT_PROJECTION_HASH_KEYS
from .typed_staged_rows import (
    AUDIT_FEATURE_FIELDS,
    MAX_SEQUENCE_LENGTH,
    MODEL_FEATURE_FIELDS,
    PRIMARY_EDIT_COUNTS,
    TYPED_FEATURE_STAGE_SCHEMA,
    TYPED_ROW_SCHEMA,
    TypedDevelopmentRows,
    target_free_feature_row_id,
)


SIGN_ACCURACY_GATE = 0.60
DEVELOPMENT_STAGE = "development"
INTERNAL_TEST_FEATURE_STAGE = "internal_test_features"
STAGES = (DEVELOPMENT_STAGE, INTERNAL_TEST_FEATURE_STAGE)

INCLUSION_MANIFEST = "inclusion_manifest"
CONTEXT_SOURCE_CROSSWALK = "context_source_crosswalk"
SOURCE_COMPONENT_SPLIT_MANIFEST = "source_component_split_manifest"
ROW_AUTHORITIES = (
    INCLUSION_MANIFEST,
    CONTEXT_SOURCE_CROSSWALK,
    SOURCE_COMPONENT_SPLIT_MANIFEST,
)

TEST_FEATURE_PROJECTION_HASH_KEYS = {
    "inclusion_manifest_internal_test_feature_projection",
    "context_source_crosswalk_internal_test_feature_projection",
    "source_component_split_manifest_internal_test_feature_projection",
}

PROJECTION_HASH_KEYS = {
    DEVELOPMENT_STAGE: {
        INCLUSION_MANIFEST: "inclusion_manifest_development_projection",
        CONTEXT_SOURCE_CROSSWALK: "context_source_crosswalk_development_projection",
        SOURCE_COMPONENT_SPLIT_MANIFEST:
            "source_component_split_manifest_development_projection",
    },
    INTERNAL_TEST_FEATURE_STAGE: {
        INCLUSION_MANIFEST:
            "inclusion_manifest_internal_test_feature_projection",
        CONTEXT_SOURCE_CROSSWALK:
            "context_source_crosswalk_internal_test_feature_projection",
        SOURCE_COMPONENT_SPLIT_MANIFEST:
            "source_component_split_manifest_internal_test_feature_projection",
    },
}

# Import-time failure is intentional if the seven-authority layer and this
# row-authority layer ever disagree about the development binding names.
if set(PROJECTION_HASH_KEYS[DEVELOPMENT_STAGE].values()) != set(
    DEVELOPMENT_PROJECTION_HASH_KEYS
):
    raise RuntimeError("development projection hash keys drifted")
if set(PROJECTION_HASH_KEYS[INTERNAL_TEST_FEATURE_STAGE].values()) != set(
    TEST_FEATURE_PROJECTION_HASH_KEYS
):
    raise RuntimeError("internal-test-feature projection hash keys drifted")

DATA_AUTHORITY_NAMES = {
    DEVELOPMENT_STAGE: "train_development_outcomes",
    INTERNAL_TEST_FEATURE_STAGE: "internal_test_features",
}
STAGE_PARTITIONS = {
    DEVELOPMENT_STAGE: frozenset({"train", "development"}),
    INTERNAL_TEST_FEATURE_STAGE: frozenset({"internal_test"}),
}
STAGE_ROW_SCHEMAS = {
    DEVELOPMENT_STAGE: TYPED_ROW_SCHEMA,
    INTERNAL_TEST_FEATURE_STAGE: TYPED_FEATURE_STAGE_SCHEMA,
}

PROJECTION_SCHEMAS = {
    (authority, stage): f"r3x_{authority}_{stage}_typed_projection_v1"
    for authority in ROW_AUTHORITIES
    for stage in STAGES
}
PROJECTION_ROLES = {
    (authority, stage): f"TYPED_{stage.upper()}_{authority.upper()}"
    for authority in ROW_AUTHORITIES
    for stage in STAGES
}
PROJECTION_STATUS = "TYPED_STAGE_ROW_PROJECTION_FROZEN"
FEATURE_COMMITMENT_SCHEMA = "r3x_feature_only_row_commitment_v1"
SPLIT_METHOD = SPLIT_ASSIGNMENT_ALGORITHM_ID

FIREWALL_SCHEMA = "r3x_pretraining_row_authority_firewall_certificate_v1"
FIREWALL_ROLE = "PRETRAINING_ROW_AUTHORITY_FIREWALL_CERTIFICATE"
FIREWALL_STATUS = "OFFLINE_TRUSTED_VERIFIER_FULL_REPLAY_PASS"
FIREWALL_EVIDENCE_SCHEMA = "r3x_row_authority_firewall_replay_evidence_v1"

HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
PROHIBITED_VALUE = re.compile(
    r"(?:sealed|restricted|gse246381|internal[_ -]?test[_ -]?outcome|ground[_ -]?truth|"
    r"observed[_ -]?(?:outcome|result)|(?:^|[_ -])label(?:$|[_ -])|"
    r"(?:^|[_ -])target(?:$|[_ -]))",
    re.IGNORECASE,
)

BASE_ENTRY_KEYS = {
    "r3_input_row_sha256",
    "feature_only_commitment_sha256",
    "estimand_id",
    "edit_count",
    "partition",
    "source_component_id",
}
INCLUSION_ENTRY_KEYS = BASE_ENTRY_KEYS | {"admission_status"}
CROSSWALK_ENTRY_KEYS = BASE_ENTRY_KEYS | {
    "study",
    "endpoint",
    "publication_context_id",
    "source_group_id_v2",
    "source_mapping_status",
    "conflict_status",
    "component_scope_status",
}
SPLIT_ENTRY_KEYS = BASE_ENTRY_KEYS | {
    "assignment_method",
    "split_seed",
    "component_order_key_sha256",
}
ENTRY_KEYS = {
    INCLUSION_MANIFEST: INCLUSION_ENTRY_KEYS,
    CONTEXT_SOURCE_CROSSWALK: CROSSWALK_ENTRY_KEYS,
    SOURCE_COMPONENT_SPLIT_MANIFEST: SPLIT_ENTRY_KEYS,
}

PROJECTION_COMMON_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "authority_name",
    "disclosure_stage",
    "source_authority_sha256",
    "data_authority_name",
    "data_authority_sha256",
    "feature_schema_sha256",
    "row_count",
    "row_set_sha256",
    "row_commitment_sha256",
    "source_component_count",
    "source_component_set_sha256",
    "entries",
    "sign_accuracy_gate",
    "sign_gate_change_authorized",
    "ordinary_only",
    "sealed_access",
    "downstream_unlock",
}
PROJECTION_KEYS = {
    INCLUSION_MANIFEST: PROJECTION_COMMON_KEYS,
    CONTEXT_SOURCE_CROSSWALK: PROJECTION_COMMON_KEYS,
    SOURCE_COMPONENT_SPLIT_MANIFEST: PROJECTION_COMMON_KEYS | {"split_policy"},
}
SPLIT_POLICY_KEYS = {
    "method",
    "parameters",
    "seed",
    "projected_component_order_sha256",
}

FIREWALL_MEASUREMENT_KEYS = {
    "schema_version",
    "development_projection_hashes",
    "internal_test_feature_projection_hashes",
    "development_row_count",
    "internal_test_feature_row_count",
    "development_component_count",
    "internal_test_feature_component_count",
    "development_row_commitment_sha256",
    "internal_test_feature_row_commitment_sha256",
    "development_component_commitment_sha256",
    "internal_test_feature_component_commitment_sha256",
    "row_overlap_count",
    "component_overlap_count",
}
FIREWALL_CERTIFICATE_KEYS = FIREWALL_MEASUREMENT_KEYS | {
    "artifact_role",
    "status",
    "verifier_id",
    "verifier_code_sha256",
    "source_authority_sha256",
    "data_authority_sha256",
    "offline_full_replay_verified",
    "runtime_trust_status",
    "sign_accuracy_gate",
    "sign_gate_change_authorized",
    "ordinary_only",
    "sealed_access",
    "downstream_unlock",
}
FIREWALL_DATA_HASH_KEYS = {
    "train_development_outcomes",
    "internal_test_features",
}


class StageRowProjectionError(ProtocolHoldError):
    """Fail-closed outcome-sensitive authority projection error."""


def _hold(code: str, message: str, **details: Any) -> None:
    raise StageRowProjectionError(code, message, details=details or None)


def _strict_keys(value: object, expected: set[str], *, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        _hold("HOLD_ROW_PROJECTION_SCHEMA", f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        _hold(
            "HOLD_ROW_PROJECTION_SCHEMA",
            f"{label} keys differ from the exact schema",
            missing=sorted(expected - actual),
            unexpected=sorted(actual - expected),
        )
    return value


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        _hold("HOLD_ROW_PROJECTION_SCHEMA", f"{label} must be lowercase SHA-256")
    return value


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        _hold("HOLD_ROW_PROJECTION_SCHEMA", f"{label} must be a safe identifier")
    if PROHIBITED_VALUE.search(value):
        _hold(
            "HOLD_ROW_PROJECTION_PROTECTED_CONTENT",
            f"{label} names prohibited support",
        )
    return value


def _canonical_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _hold("HOLD_ROW_PROJECTION_SCHEMA", f"{label} must be canonical text")
    if PROHIBITED_VALUE.search(value):
        _hold(
            "HOLD_ROW_PROJECTION_PROTECTED_CONTENT",
            f"{label} names prohibited support",
        )
    return value


def _bounded_int(value: object, *, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _hold("HOLD_ROW_PROJECTION_SCHEMA", f"{label} must be an integer")
    if not minimum <= value <= maximum:
        _hold("HOLD_ROW_PROJECTION_SCHEMA", f"{label} is out of range")
    return value


def _exact_gate(value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) != SIGN_ACCURACY_GATE
    ):
        _hold("HOLD_SIGN_GATE_MUTATION", "sign gate must remain exactly 0.60")


def _source_hashes(value: object) -> dict[str, str]:
    raw = _strict_keys(value, set(ROW_AUTHORITIES), label="source authority hashes")
    result = {
        authority: _digest(raw[authority], label=f"{authority} source SHA-256")
        for authority in ROW_AUTHORITIES
    }
    if len(set(result.values())) != len(result):
        _hold("HOLD_ROW_PROJECTION_BINDING", "source authority hashes must be distinct")
    return result


def _feature_schema_sha256() -> str:
    return canonical_sha256(
        {
            "schema_version": TYPED_ROW_SCHEMA,
            "model_feature_fields": list(MODEL_FEATURE_FIELDS),
            "audit_feature_fields": list(AUDIT_FEATURE_FIELDS),
            "outcome_fields": ["delta"],
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "edit_counts": sorted(PRIMARY_EDIT_COUNTS),
            "edit_operation": "SUB",
        }
    )


def _typed_row_keys(stage: str) -> set[str]:
    keys = {
        "schema_version",
        "r3_input_row_sha256",
        "partition",
        *MODEL_FEATURE_FIELDS,
        *AUDIT_FEATURE_FIELDS,
    }
    if stage == DEVELOPMENT_STAGE:
        keys.add("delta")
    return keys


def _validate_typed_row(row: object, *, stage: str) -> dict:
    raw = _strict_keys(row, _typed_row_keys(stage), label=f"{stage} typed row")
    if raw["schema_version"] != STAGE_ROW_SCHEMAS[stage]:
        _hold("HOLD_ROW_PROJECTION_STAGE", "typed row schema disagrees with stage")
    row_id = _digest(raw["r3_input_row_sha256"], label="r3 input row identity")
    partition = raw["partition"]
    if partition not in STAGE_PARTITIONS[stage]:
        _hold(
            "HOLD_ROW_PROJECTION_STAGE",
            "typed row partition disagrees with projection stage",
            partition=partition,
            stage=stage,
        )
    try:
        source = normalize_sequence(raw["normalized_source_sequence"])
        candidate = normalize_sequence(raw["normalized_candidate_sequence"])
    except (TypeError, ValueError) as exc:
        _hold("HOLD_ROW_PROJECTION_FEATURE_REPLAY", str(exc))
    if (
        source != raw["normalized_source_sequence"]
        or candidate != raw["normalized_candidate_sequence"]
        or len(source) != len(candidate)
        or len(source) > MAX_SEQUENCE_LENGTH
    ):
        _hold(
            "HOLD_ROW_PROJECTION_FEATURE_REPLAY",
            "sequences must be equal-length canonical RNA within the frozen maximum",
        )
    edit_count = raw["edit_count"]
    if (
        isinstance(edit_count, bool)
        or not isinstance(edit_count, int)
        or edit_count not in PRIMARY_EDIT_COUNTS
    ):
        _hold(
            "HOLD_ROW_PROJECTION_FEATURE_REPLAY",
            "edit_count must be exactly 1, 3, or 5",
        )
    try:
        edits = validated_substitutions(raw)
    except (TypeError, ValueError) as exc:
        _hold("HOLD_ROW_PROJECTION_FEATURE_REPLAY", str(exc))
    if len(edits) != edit_count:
        _hold("HOLD_ROW_PROJECTION_FEATURE_REPLAY", "edit replay count mismatch")
    if stage == DEVELOPMENT_STAGE:
        delta = raw["delta"]
        if (
            isinstance(delta, bool)
            or not isinstance(delta, (int, float))
            or not math.isfinite(float(delta))
        ):
            _hold("HOLD_ROW_PROJECTION_SCHEMA", "development delta must be finite")
    for field in (
        "endpoint",
        "publication_context_id",
        "region",
        "study",
        "estimand_id",
        "source_component_id",
        "source_group_id_v2",
    ):
        _canonical_text(raw[field], label=field)
    normalized = dict(raw)
    normalized["r3_input_row_sha256"] = row_id
    normalized["normalized_source_sequence"] = source
    normalized["normalized_candidate_sequence"] = candidate
    normalized["edit_list"] = [
        {"op": "SUB", "pos": pos, "ref": ref, "alt": alt}
        for pos, ref, alt in edits
    ]
    expected_row_id = target_free_feature_row_id(
        partition,
        {
            field: normalized[field]
            for field in (*MODEL_FEATURE_FIELDS, *AUDIT_FEATURE_FIELDS)
        },
    )
    if row_id != expected_row_id:
        _hold(
            "HOLD_ROW_PROJECTION_TARGET_FREE_IDENTITY",
            "r3 input row identity must be recomputed from feature-only support",
        )
    return normalized


def feature_only_commitment(row: Mapping) -> str:
    """Commit to the exact typed feature row without reading its outcome."""

    stage = (
        DEVELOPMENT_STAGE if "delta" in row else INTERNAL_TEST_FEATURE_STAGE
    )
    normalized = _validate_typed_row(row, stage=stage)
    material = {
        "schema_version": FEATURE_COMMITMENT_SCHEMA,
        "r3_input_row_sha256": normalized["r3_input_row_sha256"],
        "partition": normalized["partition"],
        **{field: normalized[field] for field in MODEL_FEATURE_FIELDS},
        **{field: normalized[field] for field in AUDIT_FEATURE_FIELDS},
    }
    return canonical_sha256(material)


def _cross_stage_model_row_identity(row: Mapping) -> str:
    """Identify the same model-supported edit even when partition IDs differ."""

    stage = DEVELOPMENT_STAGE if "delta" in row else INTERNAL_TEST_FEATURE_STAGE
    normalized = _validate_typed_row(row, stage=stage)
    return canonical_sha256(
        {
            "schema_version": "r3x_cross_stage_model_row_identity_v1",
            **{field: normalized[field] for field in MODEL_FEATURE_FIELDS},
            "estimand_id": normalized["estimand_id"],
        }
    )


def _normalize_rows(rows: Sequence[Mapping], *, stage: str) -> tuple[dict, ...]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        _hold("HOLD_ROW_PROJECTION_EMPTY", f"{stage} typed rows are empty")
    normalized = [_validate_typed_row(row, stage=stage) for row in rows]
    normalized.sort(key=lambda row: row["r3_input_row_sha256"])
    row_ids = [row["r3_input_row_sha256"] for row in normalized]
    if len(row_ids) != len(set(row_ids)):
        _hold("HOLD_ROW_PROJECTION_ROW_SET", "typed row identities are duplicated")
    return tuple(normalized)


def _development_rows(typed_rows: TypedDevelopmentRows) -> tuple[dict, ...]:
    if not isinstance(typed_rows, TypedDevelopmentRows):
        _hold("HOLD_ROW_PROJECTION_STAGE", "TypedDevelopmentRows is required")
    if typed_rows.typed_rows_ready is not True:
        _hold("HOLD_ROW_PROJECTION_STAGE", "typed development rows are not ready")
    if typed_rows.internal_test_disclosed is not False or typed_rows.sealed_access is not False:
        _hold(
            "HOLD_ROW_PROJECTION_PROTECTED_CONTENT",
            "development rows cannot disclose internal-test or sealed support",
        )
    train = _normalize_rows(typed_rows.train, stage=DEVELOPMENT_STAGE)
    development = _normalize_rows(typed_rows.development, stage=DEVELOPMENT_STAGE)
    if {row["partition"] for row in train} != {"train"}:
        _hold("HOLD_ROW_PROJECTION_STAGE", "train tuple contains another partition")
    if {row["partition"] for row in development} != {"development"}:
        _hold(
            "HOLD_ROW_PROJECTION_STAGE",
            "development tuple contains another partition",
        )
    train_ids = sorted(row["r3_input_row_sha256"] for row in train)
    development_ids = sorted(row["r3_input_row_sha256"] for row in development)
    if set(train_ids).intersection(development_ids):
        _hold("HOLD_ROW_PROJECTION_ROW_SET", "train/development row overlap")
    if typed_rows.train_row_set_sha256 != canonical_sha256(train_ids):
        _hold("HOLD_ROW_PROJECTION_ROW_SET", "train row-set binding is stale")
    if typed_rows.development_row_set_sha256 != canonical_sha256(development_ids):
        _hold("HOLD_ROW_PROJECTION_ROW_SET", "development row-set binding is stale")
    if typed_rows.feature_schema_sha256 != _feature_schema_sha256():
        _hold("HOLD_ROW_PROJECTION_BINDING", "typed feature schema binding is stale")
    return tuple(sorted((*train, *development), key=lambda row: row["r3_input_row_sha256"]))


def _base_entry(row: Mapping) -> dict:
    return {
        "r3_input_row_sha256": row["r3_input_row_sha256"],
        "feature_only_commitment_sha256": feature_only_commitment(row),
        "estimand_id": row["estimand_id"],
        "edit_count": row["edit_count"],
        "partition": row["partition"],
        "source_component_id": row["source_component_id"],
    }


def _component_order_key(component: str, seed: int) -> str:
    return canonical_sha256(
        {
            "assignment_algorithm_id": SPLIT_METHOD,
            "seed": seed,
            "source_component_id": component,
        }
    )


def _ordered_components(component_ids: Sequence[str], seed: int) -> list[str]:
    """Exact parity implementation of ``splits._ordered_components``."""

    _bounded_int(seed, label="frozen split seed", minimum=0, maximum=2**31 - 1)
    if (
        isinstance(component_ids, (str, bytes))
        or not isinstance(component_ids, Sequence)
        or not component_ids
    ):
        _hold("HOLD_ROW_PROJECTION_SPLIT", "component universe must be non-empty")
    components = [
        _canonical_text(component, label="source_component_id")
        for component in component_ids
    ]
    if len(components) != len(set(components)):
        _hold("HOLD_ROW_PROJECTION_SPLIT", "component universe must be unique")
    return sorted(
        components,
        key=lambda component: (_component_order_key(component, seed), component),
    )


def _partition_counts(n_components: int) -> tuple[int, int, int]:
    """Exact parity implementation of ``splits._partition_counts``."""

    _bounded_int(
        n_components,
        label="source-component count",
        minimum=1,
        maximum=10**9,
    )
    if n_components < 3:
        return max(1, n_components), 0, max(0, n_components - 1)
    n_train = max(1, int(math.floor(0.70 * n_components)))
    n_development = max(1, int(math.floor(0.15 * n_components)))
    if n_train + n_development >= n_components:
        n_train = n_components - 2
        n_development = 1
    return n_train, n_development, n_components - n_train - n_development


def replay_component_partition_assignment(
    component_ids: Sequence[str],
    seed: int,
) -> dict[str, str]:
    """Replay the authoritative global seeded component partition exactly."""

    ordered = _ordered_components(component_ids, seed)
    n_train, n_development, _ = _partition_counts(len(ordered))
    return {
        component: (
            "train"
            if index < n_train
            else "development"
            if index < n_train + n_development
            else "internal_test"
        )
        for index, component in enumerate(ordered)
    }


def _entries(authority: str, rows: Sequence[Mapping], *, split_seed: int) -> list[dict]:
    output: list[dict] = []
    for row in rows:
        base = _base_entry(row)
        if authority == INCLUSION_MANIFEST:
            entry = {**base, "admission_status": "ADMITTED_ORDINARY_PRIMARY"}
        elif authority == CONTEXT_SOURCE_CROSSWALK:
            entry = {
                **base,
                "study": row["study"],
                "endpoint": row["endpoint"],
                "publication_context_id": row["publication_context_id"],
                "source_group_id_v2": row["source_group_id_v2"],
                "source_mapping_status": "FROZEN_UNIQUE",
                "conflict_status": "CONFLICT_FREE",
                "component_scope_status": "GLOBAL_COMPONENT",
            }
        else:
            entry = {
                **base,
                "assignment_method": SPLIT_METHOD,
                "split_seed": split_seed,
                "component_order_key_sha256": _component_order_key(
                    row["source_component_id"], split_seed
                ),
            }
        output.append(entry)
    return output


def _build_projection(
    authority: str,
    stage: str,
    rows: Sequence[Mapping],
    *,
    source_authority_sha256: str,
    data_authority_sha256: str,
    split_seed: int,
) -> dict:
    entries = _entries(authority, rows, split_seed=split_seed)
    row_ids = [entry["r3_input_row_sha256"] for entry in entries]
    row_commitments = [
        {
            "r3_input_row_sha256": entry["r3_input_row_sha256"],
            "feature_only_commitment_sha256": entry[
                "feature_only_commitment_sha256"
            ],
        }
        for entry in entries
    ]
    components = sorted({entry["source_component_id"] for entry in entries})
    payload = {
        "schema_version": PROJECTION_SCHEMAS[(authority, stage)],
        "artifact_role": PROJECTION_ROLES[(authority, stage)],
        "status": PROJECTION_STATUS,
        "authority_name": authority,
        "disclosure_stage": stage,
        "source_authority_sha256": source_authority_sha256,
        "data_authority_name": DATA_AUTHORITY_NAMES[stage],
        "data_authority_sha256": data_authority_sha256,
        "feature_schema_sha256": _feature_schema_sha256(),
        "row_count": len(entries),
        "row_set_sha256": canonical_sha256(row_ids),
        "row_commitment_sha256": canonical_sha256(row_commitments),
        "source_component_count": len(components),
        "source_component_set_sha256": canonical_sha256(components),
        "entries": entries,
        "sign_accuracy_gate": SIGN_ACCURACY_GATE,
        "sign_gate_change_authorized": False,
        "ordinary_only": True,
        "sealed_access": False,
        "downstream_unlock": False,
    }
    if authority == SOURCE_COMPONENT_SPLIT_MANIFEST:
        ordered_components = _ordered_components(components, split_seed)
        payload["split_policy"] = {
            "method": SPLIT_METHOD,
            "parameters": dict(SPLIT_ASSIGNMENT_PARAMETERS),
            "seed": split_seed,
            "projected_component_order_sha256": canonical_sha256(
                ordered_components
            ),
        }
    return validate_stage_row_projection(payload)


def _build_projection_set(
    rows: Sequence[Mapping],
    *,
    stage: str,
    source_authority_sha256: Mapping,
    data_authority_sha256: str,
    split_seed: int,
) -> dict[str, dict]:
    sources = _source_hashes(source_authority_sha256)
    data_sha = _digest(data_authority_sha256, label=f"{stage} data authority")
    _bounded_int(split_seed, label="frozen split seed", minimum=0, maximum=2**31 - 1)
    if data_sha in set(sources.values()):
        _hold("HOLD_ROW_PROJECTION_BINDING", "data and source authorities must be distinct")
    return {
        authority: _build_projection(
            authority,
            stage,
            rows,
            source_authority_sha256=sources[authority],
            data_authority_sha256=data_sha,
            split_seed=split_seed,
        )
        for authority in ROW_AUTHORITIES
    }


def build_development_row_projections(
    typed_rows: TypedDevelopmentRows,
    *,
    source_authority_sha256: Mapping,
    train_development_outcomes_sha256: str,
    split_seed: int,
) -> dict[str, dict]:
    """Build exact train+development feature-only authority projections."""

    rows = _development_rows(typed_rows)
    return _build_projection_set(
        rows,
        stage=DEVELOPMENT_STAGE,
        source_authority_sha256=source_authority_sha256,
        data_authority_sha256=train_development_outcomes_sha256,
        split_seed=split_seed,
    )


def build_internal_test_feature_row_projections(
    typed_rows: Sequence[Mapping],
    *,
    source_authority_sha256: Mapping,
    internal_test_features_sha256: str,
    split_seed: int,
) -> dict[str, dict]:
    """Build exact internal-test feature projections after the freeze commit."""

    rows = _normalize_rows(typed_rows, stage=INTERNAL_TEST_FEATURE_STAGE)
    return _build_projection_set(
        rows,
        stage=INTERNAL_TEST_FEATURE_STAGE,
        source_authority_sha256=source_authority_sha256,
        data_authority_sha256=internal_test_features_sha256,
        split_seed=split_seed,
    )


def _validate_base_entry(entry: Mapping, *, stage: str) -> None:
    _digest(entry["r3_input_row_sha256"], label="projection row identity")
    _digest(entry["feature_only_commitment_sha256"], label="feature-only commitment")
    _canonical_text(entry["estimand_id"], label="estimand_id")
    edit_count = _bounded_int(entry["edit_count"], label="edit_count", minimum=1, maximum=5)
    if edit_count not in PRIMARY_EDIT_COUNTS:
        _hold("HOLD_ROW_PROJECTION_SCHEMA", "edit_count must be exactly 1, 3, or 5")
    if entry["partition"] not in STAGE_PARTITIONS[stage]:
        _hold("HOLD_ROW_PROJECTION_STAGE", "entry partition disagrees with stage")
    _canonical_text(entry["source_component_id"], label="source_component_id")


def validate_stage_row_projection(payload: Mapping) -> dict:
    """Validate one exact stage/authority projection without trusting flags."""

    if not isinstance(payload, Mapping):
        _hold("HOLD_ROW_PROJECTION_SCHEMA", "row projection must be an object")
    authority = payload.get("authority_name")
    stage = payload.get("disclosure_stage")
    if authority not in ROW_AUTHORITIES or stage not in STAGES:
        _hold("HOLD_ROW_PROJECTION_STAGE", "unsupported authority or disclosure stage")
    raw = _strict_keys(payload, PROJECTION_KEYS[authority], label="row projection")
    exact = {
        "schema_version": PROJECTION_SCHEMAS[(authority, stage)],
        "artifact_role": PROJECTION_ROLES[(authority, stage)],
        "status": PROJECTION_STATUS,
        "data_authority_name": DATA_AUTHORITY_NAMES[stage],
        "feature_schema_sha256": _feature_schema_sha256(),
        "sign_gate_change_authorized": False,
        "ordinary_only": True,
        "sealed_access": False,
        "downstream_unlock": False,
    }
    for field, expected in exact.items():
        value = raw[field]
        if isinstance(expected, bool):
            matches = value is expected
        else:
            matches = type(value) is type(expected) and value == expected
        if not matches:
            _hold(
                "HOLD_ROW_PROJECTION_SCHEMA",
                f"{field} differs from the exact projection policy",
            )
    _exact_gate(raw["sign_accuracy_gate"])
    _digest(raw["source_authority_sha256"], label="projection source authority")
    _digest(raw["data_authority_sha256"], label="projection data authority")
    entries = raw["entries"]
    if not isinstance(entries, list) or not entries:
        _hold("HOLD_ROW_PROJECTION_EMPTY", "projection entries are empty")
    row_ids: list[str] = []
    row_commitments: list[dict] = []
    components: set[str] = set()
    previous: str | None = None
    split_seed: int | None = None
    if authority == SOURCE_COMPONENT_SPLIT_MANIFEST:
        policy = _strict_keys(raw["split_policy"], SPLIT_POLICY_KEYS, label="split policy")
        if policy["method"] != SPLIT_METHOD:
            _hold("HOLD_ROW_PROJECTION_SPLIT", "unsupported component split method")
        if policy["parameters"] != SPLIT_ASSIGNMENT_PARAMETERS:
            _hold(
                "HOLD_ROW_PROJECTION_SPLIT",
                "component split parameters differ from the authoritative policy",
            )
        split_seed = _bounded_int(
            policy["seed"], label="frozen split seed", minimum=0, maximum=2**31 - 1
        )
        _digest(
            policy["projected_component_order_sha256"],
            label="projected component order",
        )
    for index, value in enumerate(entries):
        entry = _strict_keys(value, ENTRY_KEYS[authority], label=f"entry {index}")
        _validate_base_entry(entry, stage=stage)
        row_id = entry["r3_input_row_sha256"]
        if previous is not None and row_id <= previous:
            _hold(
                "HOLD_ROW_PROJECTION_ROW_SET",
                "entries must be strictly sorted by unique row identity",
            )
        previous = row_id
        row_ids.append(row_id)
        row_commitments.append(
            {
                "r3_input_row_sha256": row_id,
                "feature_only_commitment_sha256": entry[
                    "feature_only_commitment_sha256"
                ],
            }
        )
        components.add(entry["source_component_id"])
        if authority == INCLUSION_MANIFEST:
            if entry["admission_status"] != "ADMITTED_ORDINARY_PRIMARY":
                _hold("HOLD_ROW_PROJECTION_INCLUSION", "row is not ordinarily admitted")
        elif authority == CONTEXT_SOURCE_CROSSWALK:
            for field in (
                "study",
                "endpoint",
                "publication_context_id",
                "source_group_id_v2",
            ):
                _canonical_text(entry[field], label=field)
            statuses = {
                "source_mapping_status": "FROZEN_UNIQUE",
                "conflict_status": "CONFLICT_FREE",
                "component_scope_status": "GLOBAL_COMPONENT",
            }
            if any(entry[field] != expected for field, expected in statuses.items()):
                _hold("HOLD_ROW_PROJECTION_CROSSWALK", "crosswalk status is not frozen")
        else:
            if entry["assignment_method"] != SPLIT_METHOD:
                _hold("HOLD_ROW_PROJECTION_SPLIT", "entry split method drifted")
            if entry["split_seed"] != split_seed:
                _hold("HOLD_ROW_PROJECTION_SPLIT", "entry split seed drifted")
            expected_order = _component_order_key(entry["source_component_id"], split_seed)
            if entry["component_order_key_sha256"] != expected_order:
                _hold("HOLD_ROW_PROJECTION_SPLIT", "component order key is stale")
    if raw["row_count"] != len(entries):
        _hold("HOLD_ROW_PROJECTION_ROW_SET", "row count is stale")
    if raw["row_set_sha256"] != canonical_sha256(row_ids):
        _hold("HOLD_ROW_PROJECTION_ROW_SET", "row-set commitment is stale")
    if raw["row_commitment_sha256"] != canonical_sha256(row_commitments):
        _hold("HOLD_ROW_PROJECTION_FEATURE_REPLAY", "feature commitment set is stale")
    sorted_components = sorted(components)
    if raw["source_component_count"] != len(sorted_components):
        _hold("HOLD_ROW_PROJECTION_COMPONENT", "source-component count is stale")
    if raw["source_component_set_sha256"] != canonical_sha256(sorted_components):
        _hold("HOLD_ROW_PROJECTION_COMPONENT", "source-component commitment is stale")
    if authority == SOURCE_COMPONENT_SPLIT_MANIFEST:
        expected_order = _ordered_components(sorted_components, split_seed)
        if raw["split_policy"]["projected_component_order_sha256"] != canonical_sha256(
            expected_order
        ):
            _hold("HOLD_ROW_PROJECTION_SPLIT", "projected component order is stale")
    return dict(raw)


def canonical_stage_row_projection_bytes(payload: Mapping) -> bytes:
    try:
        return canonical_json_bytes(validate_stage_row_projection(payload))
    except ArtifactIOError as exc:
        _hold("HOLD_ROW_PROJECTION_JSON", str(exc))


def stage_projection_hashes(
    projections: Mapping,
    *,
    stage: str,
) -> dict[str, str]:
    raw = _strict_keys(projections, set(ROW_AUTHORITIES), label=f"{stage} projections")
    output: dict[str, str] = {}
    for authority in ROW_AUTHORITIES:
        payload = validate_stage_row_projection(raw[authority])
        if payload["authority_name"] != authority or payload["disclosure_stage"] != stage:
            _hold("HOLD_ROW_PROJECTION_BINDING", "projection map binding mismatch")
        output[PROJECTION_HASH_KEYS[stage][authority]] = hashlib.sha256(
            canonical_stage_row_projection_bytes(payload)
        ).hexdigest()
    return output


def _parse_json_object(raw: bytes, *, label: str) -> dict:
    def unique_object(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        _hold("HOLD_ROW_PROJECTION_JSON", f"{label} is not strict JSON: {exc}")
    if not isinstance(payload, dict):
        _hold("HOLD_ROW_PROJECTION_JSON", f"{label} must be a JSON object")
    return payload


def _checked_run_root(run_root: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(run_root)))
    if root.is_symlink() or not root.is_dir():
        _hold("HOLD_ROW_PROJECTION_PATH", "run root must be an existing real directory")
    resolved = root.resolve(strict=True)
    if resolved != root:
        _hold("HOLD_ROW_PROJECTION_PATH", "run root cannot traverse symlinks")
    return root


def _checked_artifact_path(
    run_root: Path,
    path: Path,
    *,
    output: bool,
    required_namespace: tuple[str, ...],
) -> Path:
    root = _checked_run_root(run_root)
    candidate = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        _hold("HOLD_ROW_PROJECTION_PATH", "artifact must remain beneath run root")
    if not relative.parts:
        _hold("HOLD_ROW_PROJECTION_PATH", "artifact cannot replace run root")
    if relative.parts[: len(required_namespace)] != required_namespace:
        _hold(
            "HOLD_ROW_PROJECTION_STAGE_NAMESPACE",
            "artifact path is outside its literal disclosure-stage namespace",
            required_namespace=list(required_namespace),
        )
    current = root
    parts = relative.parts[:-1] if output else relative.parts
    for part in parts:
        current = current / part
        if current.is_symlink():
            _hold("HOLD_ROW_PROJECTION_PATH", "artifact path cannot traverse symlinks")
    if output:
        if candidate.parent.is_symlink() or not candidate.parent.is_dir():
            _hold("HOLD_ROW_PROJECTION_PATH", "artifact parent must be an existing real directory")
    elif candidate.is_symlink():
        _hold("HOLD_ROW_PROJECTION_PATH", "artifact cannot be a symlink")
    return candidate


def write_stage_row_projection_exclusive(
    run_root: Path,
    path: Path,
    payload: Mapping,
) -> str:
    validated = validate_stage_row_projection(payload)
    namespace = (
        ("development",)
        if validated["disclosure_stage"] == DEVELOPMENT_STAGE
        else ("authority_preparation", "internal_test_features")
    )
    candidate = _checked_artifact_path(
        run_root,
        path,
        output=True,
        required_namespace=namespace,
    )
    try:
        return atomic_write_json_exclusive(
            candidate,
            validated,
            validator=validate_stage_row_projection,
        )
    except ArtifactIOError as exc:
        _hold("HOLD_ROW_PROJECTION_WRITE", str(exc))


def load_stage_row_projection_exact(
    run_root: Path,
    path: Path,
    expected_sha256: str,
    *,
    expected_authority_name: str,
    expected_stage: str,
    expected_source_authority_sha256: str | None = None,
    expected_data_authority_sha256: str | None = None,
) -> dict:
    if expected_stage not in STAGES:
        _hold("HOLD_ROW_PROJECTION_STAGE", "unsupported expected disclosure stage")
    namespace = (
        ("development",)
        if expected_stage == DEVELOPMENT_STAGE
        else ("authority_preparation", "internal_test_features")
    )
    candidate = _checked_artifact_path(
        run_root,
        path,
        output=False,
        required_namespace=namespace,
    )
    _digest(expected_sha256, label="row projection artifact")
    try:
        raw = read_exact_regular_file(candidate, expected_sha256)
    except ArtifactIOError as exc:
        _hold("HOLD_ROW_PROJECTION_HASH_MISMATCH", str(exc))
    payload = validate_stage_row_projection(_parse_json_object(raw, label="row projection"))
    if raw != canonical_stage_row_projection_bytes(payload):
        _hold("HOLD_ROW_PROJECTION_JSON", "row projection bytes are not canonical")
    if (
        payload["authority_name"] != expected_authority_name
        or payload["disclosure_stage"] != expected_stage
    ):
        _hold("HOLD_ROW_PROJECTION_BINDING", "authority/stage binding mismatch")
    if (
        expected_source_authority_sha256 is not None
        and payload["source_authority_sha256"]
        != _digest(expected_source_authority_sha256, label="expected source authority")
    ):
        _hold("HOLD_ROW_PROJECTION_BINDING", "source authority binding mismatch")
    if (
        expected_data_authority_sha256 is not None
        and payload["data_authority_sha256"]
        != _digest(expected_data_authority_sha256, label="expected data authority")
    ):
        _hold("HOLD_ROW_PROJECTION_BINDING", "data authority binding mismatch")
    return payload


@dataclass(frozen=True)
class FirewallReplayContext:
    development_projections: Mapping[str, Mapping]
    internal_test_feature_projections: Mapping[str, Mapping]
    typed_development_rows: TypedDevelopmentRows
    typed_internal_test_feature_rows: tuple[Mapping, ...]
    source_authority_sha256: Mapping[str, str]
    train_development_outcomes_sha256: str
    internal_test_features_sha256: str
    split_seed: int


@dataclass(frozen=True)
class TrustedFirewallVerifier:
    verifier_id: str
    verifier_code_sha256: str
    verifier: Callable[[FirewallReplayContext], Mapping]


@dataclass(frozen=True)
class RowAuthorityCapability:
    status: str
    blockers: tuple[str, ...]
    development_projection_verified: bool
    offline_full_replay_verified: bool
    certificate_sha256: str | None
    row_authorities_ready: bool = False
    aggregate_typed_safe_authorities_ready: bool = False
    training_ready: bool = False
    ordinary_only: bool = True
    sealed_access: bool = False
    downstream_unlock: bool = False
    sign_accuracy_gate: float = SIGN_ACCURACY_GATE


def verifier_code_sha256(verifier: Callable) -> str:
    """Return a deterministic digest of a plain verifier function's code."""

    if not isinstance(verifier, FunctionType):
        _hold("HOLD_FIREWALL_VERIFIER", "trusted verifier must be a plain function")
    if verifier.__closure__:
        _hold(
            "HOLD_FIREWALL_VERIFIER",
            "trusted verifier cannot depend on mutable closure state",
        )
    try:
        default_bytes = canonical_json_bytes(
            {
                "defaults": list(verifier.__defaults__ or ()),
                "kwdefaults": dict(verifier.__kwdefaults__ or {}),
            }
        )
    except ArtifactIOError as exc:
        _hold("HOLD_FIREWALL_VERIFIER", f"verifier defaults are not canonical: {exc}")
    source_name = inspect.getsourcefile(verifier)
    if not source_name:
        _hold("HOLD_FIREWALL_VERIFIER", "trusted verifier source file is unavailable")
    source_path = Path(os.path.abspath(source_name))
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(source_path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            _hold(
                "HOLD_FIREWALL_VERIFIER",
                "trusted verifier source must be a regular single-link file",
            )
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            source_bytes = handle.read()
            after = os.fstat(handle.fileno())
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or after.st_nlink != 1
        ):
            _hold("HOLD_FIREWALL_VERIFIER", "trusted verifier source changed during read")
    except OSError as exc:
        _hold("HOLD_FIREWALL_VERIFIER", f"trusted verifier source cannot be read: {exc}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    material = b"\0".join(
        (
            str(verifier.__module__).encode("utf-8"),
            str(verifier.__qualname__).encode("utf-8"),
            marshal.dumps(verifier.__code__),
            default_bytes,
            hashlib.sha256(source_bytes).digest(),
        )
    )
    return hashlib.sha256(material).hexdigest()


def register_trusted_firewall_verifier(
    verifier_id: str,
    verifier: Callable[[FirewallReplayContext], Mapping],
    *,
    expected_code_sha256: str,
) -> TrustedFirewallVerifier:
    identifier = _identifier(verifier_id, label="verifier_id")
    expected = _digest(expected_code_sha256, label="expected verifier code")
    actual = verifier_code_sha256(verifier)
    if actual != expected:
        _hold("HOLD_FIREWALL_VERIFIER_CODE", "verifier code SHA-256 mismatch")
    return TrustedFirewallVerifier(identifier, actual, verifier)


def _projection_rows(projections: Mapping, *, stage: str) -> tuple[set[str], set[str]]:
    raw = _strict_keys(projections, set(ROW_AUTHORITIES), label=f"{stage} projection set")
    validated: dict[str, dict] = {}
    for authority in ROW_AUTHORITIES:
        item = validate_stage_row_projection(raw[authority])
        if item["authority_name"] != authority or item["disclosure_stage"] != stage:
            _hold("HOLD_ROW_PROJECTION_BINDING", "projection set authority/stage mismatch")
        validated[authority] = item
    inclusion = validated[INCLUSION_MANIFEST]
    inclusion_by_id = {
        entry["r3_input_row_sha256"]: entry for entry in inclusion["entries"]
    }
    for authority in (CONTEXT_SOURCE_CROSSWALK, SOURCE_COMPONENT_SPLIT_MANIFEST):
        comparison = {
            entry["r3_input_row_sha256"]: entry
            for entry in validated[authority]["entries"]
        }
        if set(comparison) != set(inclusion_by_id):
            _hold("HOLD_ROW_PROJECTION_ROW_SET", "three projections have different row sets")
        for row_id, base in inclusion_by_id.items():
            other = comparison[row_id]
            for field in BASE_ENTRY_KEYS:
                if other[field] != base[field]:
                    _hold(
                        "HOLD_ROW_PROJECTION_CROSS_BINDING",
                        "three projections disagree on a row binding",
                    )
    source_hashes = {
        authority: validated[authority]["source_authority_sha256"]
        for authority in ROW_AUTHORITIES
    }
    _source_hashes(source_hashes)
    data_hashes = {item["data_authority_sha256"] for item in validated.values()}
    if len(data_hashes) != 1:
        _hold("HOLD_ROW_PROJECTION_BINDING", "stage projections bind different data authorities")
    return set(inclusion_by_id), {
        entry["source_component_id"] for entry in inclusion["entries"]
    }


def _exact_projection_sets(actual: Mapping, expected: Mapping, *, stage: str) -> None:
    _projection_rows(actual, stage=stage)
    _projection_rows(expected, stage=stage)
    for authority in ROW_AUTHORITIES:
        if canonical_stage_row_projection_bytes(actual[authority]) != canonical_stage_row_projection_bytes(
            expected[authority]
        ):
            _hold(
                "HOLD_ROW_PROJECTION_REPLAY",
                "projection differs from typed-row full replay",
                authority=authority,
                stage=stage,
            )


def _assert_global_component_partition_replay(
    development_projections: Mapping,
    internal_test_feature_projections: Mapping,
    *,
    split_seed: int,
) -> None:
    """Replay the full component universe before any training capability."""

    development_split = validate_stage_row_projection(
        development_projections[SOURCE_COMPONENT_SPLIT_MANIFEST]
    )
    test_split = validate_stage_row_projection(
        internal_test_feature_projections[SOURCE_COMPONENT_SPLIT_MANIFEST]
    )
    declared: dict[str, str] = {}
    for projection in (development_split, test_split):
        if projection["split_policy"]["seed"] != split_seed:
            _hold("HOLD_FIREWALL_GLOBAL_SPLIT_REPLAY", "split seed differs by stage")
        for entry in projection["entries"]:
            component = entry["source_component_id"]
            partition = entry["partition"]
            previous = declared.setdefault(component, partition)
            if previous != partition:
                _hold(
                    "HOLD_FIREWALL_GLOBAL_SPLIT_REPLAY",
                    "one global component has multiple declared partitions",
                )
    expected = replay_component_partition_assignment(sorted(declared), split_seed)
    if declared != expected:
        _hold(
            "HOLD_FIREWALL_GLOBAL_SPLIT_REPLAY",
            "declared partitions differ from the authoritative global seeded replay",
        )


def _firewall_measurements(context: FirewallReplayContext) -> dict:
    sources = _source_hashes(context.source_authority_sha256)
    train_sha = _digest(
        context.train_development_outcomes_sha256,
        label="train/development outcomes authority",
    )
    test_sha = _digest(
        context.internal_test_features_sha256,
        label="internal-test feature authority",
    )
    if train_sha == test_sha or {train_sha, test_sha}.intersection(sources.values()):
        _hold("HOLD_ROW_PROJECTION_BINDING", "all source/data authorities must be distinct")
    expected_development = build_development_row_projections(
        context.typed_development_rows,
        source_authority_sha256=sources,
        train_development_outcomes_sha256=train_sha,
        split_seed=context.split_seed,
    )
    expected_test = build_internal_test_feature_row_projections(
        context.typed_internal_test_feature_rows,
        source_authority_sha256=sources,
        internal_test_features_sha256=test_sha,
        split_seed=context.split_seed,
    )
    _exact_projection_sets(
        context.development_projections,
        expected_development,
        stage=DEVELOPMENT_STAGE,
    )
    _exact_projection_sets(
        context.internal_test_feature_projections,
        expected_test,
        stage=INTERNAL_TEST_FEATURE_STAGE,
    )
    development_ids, development_components = _projection_rows(
        context.development_projections, stage=DEVELOPMENT_STAGE
    )
    test_ids, test_components = _projection_rows(
        context.internal_test_feature_projections,
        stage=INTERNAL_TEST_FEATURE_STAGE,
    )
    development_model_rows = {
        _cross_stage_model_row_identity(row)
        for row in _development_rows(context.typed_development_rows)
    }
    test_model_rows = {
        _cross_stage_model_row_identity(row)
        for row in _normalize_rows(
            context.typed_internal_test_feature_rows,
            stage=INTERNAL_TEST_FEATURE_STAGE,
        )
    }
    row_overlap = len(development_ids.intersection(test_ids)) + len(
        development_model_rows.intersection(test_model_rows)
    )
    component_overlap = len(development_components.intersection(test_components))
    if row_overlap:
        _hold("HOLD_FIREWALL_ROW_OVERLAP", "development/test row overlap is nonzero")
    if component_overlap:
        _hold(
            "HOLD_FIREWALL_COMPONENT_OVERLAP",
            "development/test source-component overlap is nonzero",
        )
    _assert_global_component_partition_replay(
        context.development_projections,
        context.internal_test_feature_projections,
        split_seed=context.split_seed,
    )
    development_inclusion = validate_stage_row_projection(
        context.development_projections[INCLUSION_MANIFEST]
    )
    test_inclusion = validate_stage_row_projection(
        context.internal_test_feature_projections[INCLUSION_MANIFEST]
    )
    return {
        "schema_version": FIREWALL_EVIDENCE_SCHEMA,
        "development_projection_hashes": stage_projection_hashes(
            context.development_projections, stage=DEVELOPMENT_STAGE
        ),
        "internal_test_feature_projection_hashes": stage_projection_hashes(
            context.internal_test_feature_projections,
            stage=INTERNAL_TEST_FEATURE_STAGE,
        ),
        "development_row_count": len(development_ids),
        "internal_test_feature_row_count": len(test_ids),
        "development_component_count": len(development_components),
        "internal_test_feature_component_count": len(test_components),
        "development_row_commitment_sha256": development_inclusion[
            "row_commitment_sha256"
        ],
        "internal_test_feature_row_commitment_sha256": test_inclusion[
            "row_commitment_sha256"
        ],
        "development_component_commitment_sha256": development_inclusion[
            "source_component_set_sha256"
        ],
        "internal_test_feature_component_commitment_sha256": test_inclusion[
            "source_component_set_sha256"
        ],
        "row_overlap_count": 0,
        "component_overlap_count": 0,
    }


def _independent_stage_replay(
    supplied: Mapping,
    rebuilt: Mapping,
    *,
    stage: str,
) -> tuple[set[str], set[str], dict]:
    """Second implementation of the three-authority stage cross-check."""

    supplied_map = _strict_keys(
        supplied, set(ROW_AUTHORITIES), label=f"independent {stage} projections"
    )
    rebuilt_map = _strict_keys(
        rebuilt, set(ROW_AUTHORITIES), label=f"independent rebuilt {stage} projections"
    )
    validated: dict[str, dict] = {}
    for authority in ROW_AUTHORITIES:
        candidate = validate_stage_row_projection(supplied_map[authority])
        expected = validate_stage_row_projection(rebuilt_map[authority])
        if (
            candidate["authority_name"] != authority
            or candidate["disclosure_stage"] != stage
            or canonical_stage_row_projection_bytes(candidate)
            != canonical_stage_row_projection_bytes(expected)
        ):
            _hold(
                "HOLD_FIREWALL_INDEPENDENT_REPLAY",
                "independent verifier found a typed-row projection mismatch",
            )
        validated[authority] = candidate

    entry_maps = {
        authority: {
            entry["r3_input_row_sha256"]: entry
            for entry in validated[authority]["entries"]
        }
        for authority in ROW_AUTHORITIES
    }
    row_sets = [set(entries) for entries in entry_maps.values()]
    if any(row_set != row_sets[0] for row_set in row_sets[1:]):
        _hold(
            "HOLD_FIREWALL_INDEPENDENT_REPLAY",
            "independent verifier found authority row-set disagreement",
        )
    for row_id in row_sets[0]:
        reference = entry_maps[INCLUSION_MANIFEST][row_id]
        for authority in (
            CONTEXT_SOURCE_CROSSWALK,
            SOURCE_COMPONENT_SPLIT_MANIFEST,
        ):
            comparison = entry_maps[authority][row_id]
            if any(comparison[field] != reference[field] for field in BASE_ENTRY_KEYS):
                _hold(
                    "HOLD_FIREWALL_INDEPENDENT_REPLAY",
                    "independent verifier found row binding disagreement",
                )
    source_bindings = {
        authority: validated[authority]["source_authority_sha256"]
        for authority in ROW_AUTHORITIES
    }
    _source_hashes(source_bindings)
    if len({item["data_authority_sha256"] for item in validated.values()}) != 1:
        _hold(
            "HOLD_FIREWALL_INDEPENDENT_REPLAY",
            "independent verifier found data-authority disagreement",
        )
    inclusion = validated[INCLUSION_MANIFEST]
    components = {
        entry["source_component_id"] for entry in inclusion["entries"]
    }
    return row_sets[0], components, inclusion


def independent_firewall_verifier(context: FirewallReplayContext) -> dict:
    """Independent reference verifier; callers register its exact source hash.

    This implementation deliberately does not call ``_firewall_measurements``
    or ``_projection_rows``.  It rebuilds both stages from typed rows, performs
    a second three-authority comparison, and derives its own overlap evidence.
    """

    if not isinstance(context, FirewallReplayContext):
        _hold("HOLD_FIREWALL_VERIFIER", "firewall replay context is required")
    sources = _source_hashes(context.source_authority_sha256)
    train_sha = _digest(
        context.train_development_outcomes_sha256,
        label="independent train/development outcomes authority",
    )
    test_sha = _digest(
        context.internal_test_features_sha256,
        label="independent internal-test feature authority",
    )
    if train_sha == test_sha or {train_sha, test_sha}.intersection(sources.values()):
        _hold(
            "HOLD_FIREWALL_INDEPENDENT_REPLAY",
            "independent verifier found aliased source/data authorities",
        )
    rebuilt_development = build_development_row_projections(
        context.typed_development_rows,
        source_authority_sha256=sources,
        train_development_outcomes_sha256=train_sha,
        split_seed=context.split_seed,
    )
    rebuilt_test = build_internal_test_feature_row_projections(
        context.typed_internal_test_feature_rows,
        source_authority_sha256=sources,
        internal_test_features_sha256=test_sha,
        split_seed=context.split_seed,
    )
    development_ids, development_components, development_inclusion = (
        _independent_stage_replay(
            context.development_projections,
            rebuilt_development,
            stage=DEVELOPMENT_STAGE,
        )
    )
    test_ids, test_components, test_inclusion = _independent_stage_replay(
        context.internal_test_feature_projections,
        rebuilt_test,
        stage=INTERNAL_TEST_FEATURE_STAGE,
    )
    row_overlap = len(development_ids & test_ids)
    development_model_rows = {
        _cross_stage_model_row_identity(row)
        for row in _development_rows(context.typed_development_rows)
    }
    test_model_rows = {
        _cross_stage_model_row_identity(row)
        for row in _normalize_rows(
            context.typed_internal_test_feature_rows,
            stage=INTERNAL_TEST_FEATURE_STAGE,
        )
    }
    row_overlap += len(development_model_rows & test_model_rows)
    component_overlap = len(development_components & test_components)
    if row_overlap or component_overlap:
        _hold(
            "HOLD_FIREWALL_INDEPENDENT_REPLAY",
            "independent verifier found nonzero row/component overlap",
        )
    _assert_global_component_partition_replay(
        context.development_projections,
        context.internal_test_feature_projections,
        split_seed=context.split_seed,
    )
    return {
        "schema_version": FIREWALL_EVIDENCE_SCHEMA,
        "development_projection_hashes": stage_projection_hashes(
            context.development_projections,
            stage=DEVELOPMENT_STAGE,
        ),
        "internal_test_feature_projection_hashes": stage_projection_hashes(
            context.internal_test_feature_projections,
            stage=INTERNAL_TEST_FEATURE_STAGE,
        ),
        "development_row_count": len(development_ids),
        "internal_test_feature_row_count": len(test_ids),
        "development_component_count": len(development_components),
        "internal_test_feature_component_count": len(test_components),
        "development_row_commitment_sha256": development_inclusion[
            "row_commitment_sha256"
        ],
        "internal_test_feature_row_commitment_sha256": test_inclusion[
            "row_commitment_sha256"
        ],
        "development_component_commitment_sha256": development_inclusion[
            "source_component_set_sha256"
        ],
        "internal_test_feature_component_commitment_sha256": test_inclusion[
            "source_component_set_sha256"
        ],
        "row_overlap_count": 0,
        "component_overlap_count": 0,
    }


def _resolve_verifier(
    verifier_id: str,
    verifier_registry: Mapping[str, TrustedFirewallVerifier] | None,
) -> TrustedFirewallVerifier:
    identifier = _identifier(verifier_id, label="verifier_id")
    if not isinstance(verifier_registry, Mapping) or not verifier_registry:
        _hold("HOLD_FIREWALL_VERIFIER_MISSING", "explicit trusted verifier registry is required")
    registration = verifier_registry.get(identifier)
    if not isinstance(registration, TrustedFirewallVerifier):
        _hold("HOLD_FIREWALL_VERIFIER_MISSING", "trusted verifier ID is not registered")
    if registration.verifier_id != identifier:
        _hold("HOLD_FIREWALL_VERIFIER_ID", "trusted verifier registry key/ID mismatch")
    actual = verifier_code_sha256(registration.verifier)
    if registration.verifier_code_sha256 != actual:
        _hold("HOLD_FIREWALL_VERIFIER_CODE", "registered verifier code changed")
    return registration


def _run_trusted_verifier(
    context: FirewallReplayContext,
    *,
    verifier_id: str,
    verifier_registry: Mapping[str, TrustedFirewallVerifier] | None,
) -> tuple[dict, TrustedFirewallVerifier]:
    # This replay is performed by the authority layer even if a registered
    # verifier is malicious or buggy.  The trusted verifier then has to return
    # the same exact independently derived evidence.
    expected = _firewall_measurements(context)
    registration = _resolve_verifier(verifier_id, verifier_registry)
    try:
        reported = registration.verifier(context)
    except StageRowProjectionError:
        raise
    except Exception as exc:
        _hold("HOLD_FIREWALL_VERIFIER", f"trusted verifier failed: {exc}")
    raw = _strict_keys(reported, FIREWALL_MEASUREMENT_KEYS, label="verifier evidence")
    if dict(raw) != expected:
        _hold("HOLD_FIREWALL_VERIFIER_DISAGREEMENT", "trusted verifier evidence disagrees")
    return expected, registration


def _certificate_from_replay(
    context: FirewallReplayContext,
    *,
    verifier_id: str,
    verifier_registry: Mapping[str, TrustedFirewallVerifier] | None,
) -> dict:
    measurements, registration = _run_trusted_verifier(
        context,
        verifier_id=verifier_id,
        verifier_registry=verifier_registry,
    )
    certificate = {
        **measurements,
        "schema_version": FIREWALL_SCHEMA,
        "artifact_role": FIREWALL_ROLE,
        "status": FIREWALL_STATUS,
        "verifier_id": registration.verifier_id,
        "verifier_code_sha256": registration.verifier_code_sha256,
        "source_authority_sha256": _source_hashes(
            context.source_authority_sha256
        ),
        "data_authority_sha256": {
            "train_development_outcomes": _digest(
                context.train_development_outcomes_sha256,
                label="train/development outcomes authority",
            ),
            "internal_test_features": _digest(
                context.internal_test_features_sha256,
                label="internal-test feature authority",
            ),
        },
        "offline_full_replay_verified": True,
        "runtime_trust_status": "PREPARED_OFFLINE_NOT_RUNTIME_TRUSTED",
        "sign_accuracy_gate": SIGN_ACCURACY_GATE,
        "sign_gate_change_authorized": False,
        "ordinary_only": True,
        "sealed_access": False,
        "downstream_unlock": False,
    }
    return validate_pretraining_firewall_certificate(certificate)


def attest_pretraining_firewall(
    context: FirewallReplayContext,
    *,
    verifier_id: str,
    verifier_registry: Mapping[str, TrustedFirewallVerifier] | None,
) -> dict:
    """Create a certificate only after trusted-verifier full replay."""

    if not isinstance(context, FirewallReplayContext):
        _hold("HOLD_FIREWALL_VERIFIER", "firewall replay context is required")
    return _certificate_from_replay(
        context,
        verifier_id=verifier_id,
        verifier_registry=verifier_registry,
    )


def validate_pretraining_firewall_certificate(payload: Mapping) -> dict:
    raw = _strict_keys(payload, FIREWALL_CERTIFICATE_KEYS, label="firewall certificate")
    exact = {
        "schema_version": FIREWALL_SCHEMA,
        "artifact_role": FIREWALL_ROLE,
        "status": FIREWALL_STATUS,
        "offline_full_replay_verified": True,
        "runtime_trust_status": "PREPARED_OFFLINE_NOT_RUNTIME_TRUSTED",
        "sign_gate_change_authorized": False,
        "ordinary_only": True,
        "sealed_access": False,
        "downstream_unlock": False,
    }
    for field, expected in exact.items():
        value = raw[field]
        matches = value is expected if isinstance(expected, bool) else (
            type(value) is type(expected) and value == expected
        )
        if not matches:
            _hold("HOLD_FIREWALL_CERTIFICATE", f"invalid certificate field: {field}")
    _exact_gate(raw["sign_accuracy_gate"])
    _identifier(raw["verifier_id"], label="verifier_id")
    _digest(raw["verifier_code_sha256"], label="verifier code")
    _source_hashes(raw["source_authority_sha256"])
    data_hashes = _strict_keys(
        raw["data_authority_sha256"],
        FIREWALL_DATA_HASH_KEYS,
        label="firewall data authority hashes",
    )
    normalized_data_values = [
        _digest(data_hashes[name], label=f"{name} data authority")
        for name in sorted(FIREWALL_DATA_HASH_KEYS)
    ]
    if len(set(normalized_data_values)) != 2:
        _hold("HOLD_FIREWALL_CERTIFICATE", "data authority hashes must be distinct")
    development_hashes = _strict_keys(
        raw["development_projection_hashes"],
        set(DEVELOPMENT_PROJECTION_HASH_KEYS),
        label="development projection hashes",
    )
    test_hashes = _strict_keys(
        raw["internal_test_feature_projection_hashes"],
        set(TEST_FEATURE_PROJECTION_HASH_KEYS),
        label="internal-test-feature projection hashes",
    )
    for name, value in (*development_hashes.items(), *test_hashes.items()):
        _digest(value, label=name)
    for field in (
        "development_row_count",
        "internal_test_feature_row_count",
        "development_component_count",
        "internal_test_feature_component_count",
    ):
        _bounded_int(raw[field], label=field, minimum=1, maximum=10**9)
    for field in (
        "development_row_commitment_sha256",
        "internal_test_feature_row_commitment_sha256",
        "development_component_commitment_sha256",
        "internal_test_feature_component_commitment_sha256",
    ):
        _digest(raw[field], label=field)
    if raw["row_overlap_count"] != 0 or raw["component_overlap_count"] != 0:
        _hold("HOLD_FIREWALL_OVERLAP", "firewall overlap counts must be integer zero")
    return dict(raw)


def canonical_pretraining_firewall_certificate_bytes(payload: Mapping) -> bytes:
    try:
        return canonical_json_bytes(validate_pretraining_firewall_certificate(payload))
    except ArtifactIOError as exc:
        _hold("HOLD_FIREWALL_CERTIFICATE", str(exc))


def verify_offline_row_authority_preparation(
    context: FirewallReplayContext,
    certificate: Mapping,
    *,
    verifier_registry: Mapping[str, TrustedFirewallVerifier] | None,
) -> RowAuthorityCapability:
    """Replay both stages offline without granting development-runtime trust.

    This authority-preparation function is deliberately not callable from the
    development/training data path: its context contains internal-test feature
    rows and projections.  A successful result proves that the prepared
    artifacts are mutually consistent, but the current staged activation
    schema does not pre-bind this combined certificate digest.  The result is
    therefore a HOLD capability, never ``row_authorities_ready``.
    """

    validated = validate_pretraining_firewall_certificate(certificate)
    expected = _certificate_from_replay(
        context,
        verifier_id=validated["verifier_id"],
        verifier_registry=verifier_registry,
    )
    if canonical_pretraining_firewall_certificate_bytes(validated) != canonical_pretraining_firewall_certificate_bytes(
        expected
    ):
        _hold("HOLD_FIREWALL_CERTIFICATE", "certificate differs from trusted full replay")
    raw = canonical_pretraining_firewall_certificate_bytes(validated)
    return RowAuthorityCapability(
        status="PREPARED_OFFLINE_NOT_RUNTIME_TRUSTED",
        blockers=(
            "ACTIVATION_SCHEMA_DOES_NOT_PREBIND_ROW_FIREWALL_CERTIFICATE_SHA256",
        ),
        development_projection_verified=True,
        offline_full_replay_verified=True,
        certificate_sha256=hashlib.sha256(raw).hexdigest(),
    )


def verify_row_authority_capability(
    context: FirewallReplayContext,
    certificate: Mapping,
    *,
    verifier_registry: Mapping[str, TrustedFirewallVerifier] | None,
) -> RowAuthorityCapability:
    """Backward-named offline check retained fail-closed for current callers.

    The returned object explicitly has ``row_authorities_ready=False``.  New
    authority-preparation code should call
    :func:`verify_offline_row_authority_preparation` to make the phase clear.
    """

    return verify_offline_row_authority_preparation(
        context,
        certificate,
        verifier_registry=verifier_registry,
    )


def assess_development_runtime_row_authority_capability(
    typed_development_rows: TypedDevelopmentRows,
    development_projections: Mapping[str, Mapping],
    *,
    source_authority_sha256: Mapping,
    train_development_outcomes_sha256: str,
    split_seed: int,
) -> RowAuthorityCapability:
    """Replay development-only material and expose the remaining trust block.

    No argument in this API can carry internal-test rows, test-feature
    projections, a test-feature path, or a verifier callback.  The current
    ``StagedActivation`` type has no field that pre-binds the combined offline
    firewall-certificate digest, so development runtime cannot safely exact-
    load and authenticate that certificate yet.  This function consequently
    verifies only the development projection and returns a permanent HOLD
    until a versioned activation schema supplies that binding.
    """

    expected = build_development_row_projections(
        typed_development_rows,
        source_authority_sha256=source_authority_sha256,
        train_development_outcomes_sha256=train_development_outcomes_sha256,
        split_seed=split_seed,
    )
    _exact_projection_sets(
        development_projections,
        expected,
        stage=DEVELOPMENT_STAGE,
    )
    return RowAuthorityCapability(
        status="DEVELOPMENT_PROJECTION_VERIFIED_FIREWALL_ACTIVATION_BINDING_MISSING",
        blockers=(
            "ACTIVATION_SCHEMA_DOES_NOT_PREBIND_ROW_FIREWALL_CERTIFICATE_SHA256",
            "DEVELOPMENT_RUNTIME_MUST_NOT_OPEN_INTERNAL_TEST_FEATURE_PROJECTIONS",
        ),
        development_projection_verified=True,
        offline_full_replay_verified=False,
        certificate_sha256=None,
    )


def write_pretraining_firewall_certificate_exclusive(
    run_root: Path,
    path: Path,
    payload: Mapping,
) -> str:
    candidate = _checked_artifact_path(
        run_root,
        path,
        output=True,
        required_namespace=("authority_preparation", "firewall"),
    )
    validated = validate_pretraining_firewall_certificate(payload)
    try:
        return atomic_write_json_exclusive(
            candidate,
            validated,
            validator=validate_pretraining_firewall_certificate,
        )
    except ArtifactIOError as exc:
        _hold("HOLD_FIREWALL_WRITE", str(exc))


def load_pretraining_firewall_certificate_exact(
    run_root: Path,
    path: Path,
    expected_sha256: str,
) -> dict:
    candidate = _checked_artifact_path(
        run_root,
        path,
        output=False,
        required_namespace=("authority_preparation", "firewall"),
    )
    _digest(expected_sha256, label="firewall certificate artifact")
    try:
        raw = read_exact_regular_file(candidate, expected_sha256)
    except ArtifactIOError as exc:
        _hold("HOLD_FIREWALL_HASH_MISMATCH", str(exc))
    payload = validate_pretraining_firewall_certificate(
        _parse_json_object(raw, label="firewall certificate")
    )
    if raw != canonical_pretraining_firewall_certificate_bytes(payload):
        _hold("HOLD_FIREWALL_CERTIFICATE", "firewall certificate bytes are not canonical")
    return payload


__all__ = [
    "CONTEXT_SOURCE_CROSSWALK",
    "DEVELOPMENT_STAGE",
    "FIREWALL_SCHEMA",
    "FirewallReplayContext",
    "INCLUSION_MANIFEST",
    "INTERNAL_TEST_FEATURE_STAGE",
    "PROJECTION_HASH_KEYS",
    "PROJECTION_SCHEMAS",
    "ROW_AUTHORITIES",
    "RowAuthorityCapability",
    "SIGN_ACCURACY_GATE",
    "SOURCE_COMPONENT_SPLIT_MANIFEST",
    "SPLIT_METHOD",
    "StageRowProjectionError",
    "TEST_FEATURE_PROJECTION_HASH_KEYS",
    "TrustedFirewallVerifier",
    "attest_pretraining_firewall",
    "assess_development_runtime_row_authority_capability",
    "build_development_row_projections",
    "build_internal_test_feature_row_projections",
    "canonical_pretraining_firewall_certificate_bytes",
    "canonical_stage_row_projection_bytes",
    "feature_only_commitment",
    "independent_firewall_verifier",
    "load_pretraining_firewall_certificate_exact",
    "load_stage_row_projection_exact",
    "register_trusted_firewall_verifier",
    "replay_component_partition_assignment",
    "stage_projection_hashes",
    "validate_pretraining_firewall_certificate",
    "validate_stage_row_projection",
    "verifier_code_sha256",
    "verify_offline_row_authority_preparation",
    "verify_row_authority_capability",
    "write_pretraining_firewall_certificate_exclusive",
    "write_stage_row_projection_exclusive",
]
