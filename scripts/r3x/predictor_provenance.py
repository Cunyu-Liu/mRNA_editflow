"""Exact, development-only predictor provenance for the R3X selector.

Metric replay is not predictor replay.  A prediction file can have internally
consistent metrics while being unrelated to the checkpoint or fitted state it
claims to represent.  This module closes that gap with one immutable commit per
method.  Every ``estimand x registered metric cell`` binds canonical train and
development inference rows, the exact predictor artifacts, and the exact
published prediction bytes.

The artifact is deliberately verifier-agnostic but not verifier-optional.  A
trusted verifier registry is supplied by the caller and is matched against a
frozen identity, schema, code hash, predictor-code hash, and predictor-config
hash.  Exact replay invokes that verifier and compares canonical prediction
bytes; a missing or mismatched verifier is a hard hold.  No self-reported
metric or self-reported ``verified`` flag is accepted.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from numbers import Real
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, Sequence

from .artifact_graph import (
    ArtifactGraphError,
    PRIMARY_INFORMATION_SET,
    parse_prediction_jsonl,
    prediction_jsonl_bytes,
    read_artifact_ref,
    validate_artifact_ref,
)
from .artifact_io import (
    ArtifactIOError,
    atomic_publish_bytes,
    atomic_write_json_exclusive,
    canonical_json_bytes,
)
PREDICTOR_PROVENANCE_SCHEMA = "r3x_development_predictor_provenance_v2"
PREDICTOR_INPUT_ROW_SCHEMA = "r3x_predictor_input_row_v2"
RECOMPUTATION_VERIFIER_SCHEMA = "r3x_predictor_recomputation_verifier_v1"
DEVELOPMENT_CHECKPOINT_SELECTION_SCHEMA = (
    "r3x_predictor_development_checkpoint_selection_envelope_v2"
)
DEVELOPMENT_CHECKPOINT_SCHEMA = "r3x_predictor_development_checkpoint_envelope_v2"

# Development-stage data authority is intentionally singular.  These non-data
# hashes freeze code, protocol, split policy, and configuration but contain no
# feature-only or outcome-only protected partition digest.
NON_DATA_AUTHORITY_HASH_KEYS = (
    "route_authorization_sha256",
    "inclusion_manifest_sha256",
    "context_source_crosswalk_sha256",
    "endpoint_estimand_registry_sha256",
    "source_component_split_manifest_sha256",
    "metric_zero_policy_registry_sha256",
    "model_optimizer_seed_config_sha256",
    "analysis_code_tree_manifest_sha256",
    "method_protocol_sha256",
    "paired_strongest_baseline_comparison_policy_sha256",
)

PROVENANCE_ROLE = "ORDINARY_DEVELOPMENT_EXACT_PREDICTOR_PROVENANCE"
PROVENANCE_STATUS = "COMPLETE_EXACT_PREDICTOR_RECOMPUTATION"
TRAIN_PARTITION = "ORDINARY_TRAIN"
DEVELOPMENT_PARTITION = "ORDINARY_DEVELOPMENT"

SCALAR_KIND = "SCALAR_DETERMINISTIC_FIT"
NEURAL_KIND = "NEURAL_FIXED_FIVE_SEED_ENSEMBLE"
FROZEN_NEURAL_SEEDS = (11, 23, 42, 67, 101)
ENSEMBLE_RULE = "FIXED_FIVE_MODEL_TOTAL_VARIANCE_ENSEMBLE"
ENSEMBLE_VARIANCE_FORMULA = (
    "MEAN_EXP_LOGVAR_PLUS_MEAN_SQUARED_MINUS_ENSEMBLE_MEAN_SQUARED"
)
VARIANCE_FLOOR = 1e-12

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_METHOD_ID = re.compile(r"^[A-Z][A-Z0-9_]*$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_VERIFIER_ID = re.compile(r"^[A-Z][A-Z0-9_.:-]*$")
_PROHIBITED_PATH_PARTS = {
    "test",
    "internal_test",
    "sealed",
    "restricted",
    "gse246381",
}

# This is a positive schema, not a blacklist.  A trusted verifier receives no
# caller-defined feature name or opaque digest: it can consume only the frozen
# primary information set below.  Historical aliases are intentionally not
# accepted because accepting two spellings would let different verifier
# implementations disagree about which value has authority.
PRIMARY_PREDICTOR_INPUT_KEYS = {
    "endpoint",
    "publication_context_id",
    "region",
    "normalized_source_sequence",
    "normalized_candidate_sequence",
    "edit_list",
    "edit_count",
}
PREDICTOR_EDIT_KEYS = {"op", "pos", "ref", "alt"}
VALID_PRIMARY_EDIT_COUNTS = frozenset({1, 3, 5})
_RNA_BASES = frozenset("ACGU")

_INPUT_ROW_KEYS = {
    "schema_version",
    "row_id",
    "partition",
    "estimand_id",
    "cell_id",
    "predictor_input",
    "training_target",
}
_VERIFIER_IDENTITY_KEYS = {
    "verifier_id",
    "schema_version",
    "verifier_code_sha256",
    "expected_predictor_code_sha256",
    "expected_predictor_config_sha256",
}
_INPUT_CELL_KEYS = {
    "cell_id",
    "train_rows_ref",
    "development_rows_ref",
    "prediction_jsonl_ref",
    "predictor_artifacts",
}
_OUTPUT_CELL_KEYS = _INPUT_CELL_KEYS | {
    "train_n_rows",
    "train_ordered_row_ids_sha256",
    "train_row_set_sha256",
    "development_n_rows",
    "development_ordered_row_ids_sha256",
    "development_row_set_sha256",
}
_ESTIMAND_KEYS = {
    "estimand_id",
    "estimand_binding_sha256",
    "cells",
}
_SCALAR_ARTIFACT_KEYS = {
    "kind",
    "fitted_state_ref",
    "fit_config_ref",
}
_NEURAL_ARTIFACT_KEYS = {
    "kind",
    "seeds",
    "seed_artifacts",
    "ensemble_prediction_ref",
    "ensemble_rule",
    "ensemble_variance_formula",
    "variance_floor",
}
_SEED_ARTIFACT_KEYS = {
    "seed",
    "checkpoint_selection_ref",
    "checkpoint_ref",
    "development_prediction_ref",
}
_DEVELOPMENT_SELECTION_KEYS = {
    "schema_version",
    "artifact_role",
    "run_start_sha256",
    "activation_manifest_sha256",
    "train_development_outcomes_sha256",
    "non_data_authority_hashes",
    "method_id",
    "estimand_id",
    "estimand_binding_sha256",
    "cell_id",
    "seed",
    "train_row_set_sha256",
    "development_row_set_sha256",
    "predictor_code_sha256",
    "predictor_config_sha256",
    "selected_epoch",
    "ordinary_only",
    "downstream_unlock",
}
_DEVELOPMENT_CHECKPOINT_KEYS = {
    "schema_version",
    "artifact_role",
    "run_start_sha256",
    "activation_manifest_sha256",
    "train_development_outcomes_sha256",
    "non_data_authority_hashes",
    "method_id",
    "estimand_id",
    "estimand_binding_sha256",
    "cell_id",
    "seed",
    "train_row_set_sha256",
    "development_row_set_sha256",
    "information_set",
    "information_set_fingerprint",
    "predictor_code_sha256",
    "predictor_config_sha256",
    "checkpoint_selection_ref",
    "state_ref",
    "architecture_id",
    "ordinary_only",
    "downstream_unlock",
}
_PROVENANCE_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "run_start_sha256",
    "activation_manifest_sha256",
    "train_development_outcomes_sha256",
    "non_data_authority_hashes",
    "method_id",
    "partition",
    "information_set",
    "information_set_fingerprint",
    "predictor_code_sha256",
    "predictor_config_sha256",
    "verifier_identity",
    "estimands",
    "ordinary_only",
    "downstream_unlock",
}


class PredictorProvenanceError(RuntimeError):
    """Fail-closed predictor provenance or recomputation error."""


@dataclass(frozen=True)
class PredictorVerifier:
    """Trusted executable verifier registered outside the evidence artifact.

    ``recompute`` receives the checked run root and one normalized cell commit.
    Scalar verifiers return ``{"prediction": rows}``.  Neural verifiers return
    one entry for each frozen seed, keyed by its decimal string.  The module
    canonicalizes and byte-compares every returned row sequence.
    """

    verifier_id: str
    code_sha256: str
    expected_predictor_code_sha256: str
    expected_predictor_config_sha256: str
    recompute: Callable[[Path, Mapping], Mapping[str, Sequence[Mapping]]]
    schema_version: str = RECOMPUTATION_VERIFIER_SCHEMA


def _strict_keys(value: Mapping, expected: set[str], *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise PredictorProvenanceError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise PredictorProvenanceError(
            f"{label} keys differ from strict schema: {sorted(actual ^ expected)}"
        )


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise PredictorProvenanceError(f"{label} must be 64 lowercase hex")
    return value


def _method_id(value: object, *, label: str = "method_id") -> str:
    if not isinstance(value, str) or not _METHOD_ID.fullmatch(value):
        raise PredictorProvenanceError(f"{label} is not a safe frozen method ID")
    return value


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise PredictorProvenanceError(f"{label} is not a canonical identifier")
    return value


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PredictorProvenanceError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise PredictorProvenanceError(f"{label} must be a finite number")
    return result


def _seed(value: object, *, label: str = "seed") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PredictorProvenanceError(f"{label} must be an integer")
    if value not in FROZEN_NEURAL_SEEDS:
        raise PredictorProvenanceError(f"{label} is not one of the five frozen seeds")
    return value


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PredictorProvenanceError(f"{label} must be a positive integer")
    return value


def _non_data_authority_hashes(value: Mapping) -> dict[str, str]:
    _strict_keys(
        value,
        set(NON_DATA_AUTHORITY_HASH_KEYS),
        label="non_data_authority_hashes",
    )
    return {
        name: _digest(value[name], label=f"non_data_authority_hashes.{name}")
        for name in NON_DATA_AUTHORITY_HASH_KEYS
    }


def _canonical_sha256(value: object) -> str:
    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PredictorProvenanceError(
            f"value is not canonical finite JSON: {exc}"
        ) from exc
    return hashlib.sha256(raw).hexdigest()


def _canonical_feature_text(value: object, *, label: str) -> str:
    """Return one non-empty, already-canonical metadata string.

    Metadata values are not rewritten.  In particular, whitespace trimming or
    Unicode/control-character normalization here would create a second
    interpretation of the feature bytes seen by a verifier.
    """

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise PredictorProvenanceError(
            f"predictor_input.{label} must be a non-empty canonical string"
        )
    return value


def _canonical_rna_sequence(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(base not in _RNA_BASES for base in value)
    ):
        raise PredictorProvenanceError(
            f"predictor_input.{label} must be non-empty uppercase A/C/G/U"
        )
    return value


def _normalize_primary_predictor_input(value: Mapping) -> dict:
    """Validate and copy the exact primary-information-set feature object."""

    _strict_keys(value, PRIMARY_PREDICTOR_INPUT_KEYS, label="predictor_input")
    endpoint = _canonical_feature_text(value.get("endpoint"), label="endpoint")
    publication_context_id = _canonical_feature_text(
        value.get("publication_context_id"), label="publication_context_id"
    )
    region = _canonical_feature_text(value.get("region"), label="region")
    source = _canonical_rna_sequence(
        value.get("normalized_source_sequence"),
        label="normalized_source_sequence",
    )
    candidate = _canonical_rna_sequence(
        value.get("normalized_candidate_sequence"),
        label="normalized_candidate_sequence",
    )
    if len(source) != len(candidate):
        raise PredictorProvenanceError(
            "predictor_input source and candidate sequences must have equal length"
        )

    edit_count = value.get("edit_count")
    if (
        isinstance(edit_count, bool)
        or not isinstance(edit_count, int)
        or edit_count not in VALID_PRIMARY_EDIT_COUNTS
    ):
        raise PredictorProvenanceError(
            "predictor_input.edit_count must be one of the frozen budgets {1,3,5}"
        )
    raw_edits = value.get("edit_list")
    if not isinstance(raw_edits, list) or len(raw_edits) != edit_count:
        raise PredictorProvenanceError(
            "predictor_input.edit_list length must exactly equal edit_count"
        )

    edits = []
    replay = list(source)
    prior_position = -1
    for index, raw_edit in enumerate(raw_edits):
        label = f"predictor_input.edit_list[{index}]"
        _strict_keys(raw_edit, PREDICTOR_EDIT_KEYS, label=label)
        if raw_edit.get("op") != "SUB":
            raise PredictorProvenanceError(f"{label}.op must be exactly SUB")
        position = raw_edit.get("pos")
        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or not 0 <= position < len(source)
        ):
            raise PredictorProvenanceError(
                f"{label}.pos must be a zero-based in-range integer"
            )
        if position <= prior_position:
            raise PredictorProvenanceError(
                "predictor_input.edit_list positions must be unique and strictly increasing"
            )
        prior_position = position
        ref = raw_edit.get("ref")
        alt = raw_edit.get("alt")
        if (
            not isinstance(ref, str)
            or len(ref) != 1
            or ref not in _RNA_BASES
            or not isinstance(alt, str)
            or len(alt) != 1
            or alt not in _RNA_BASES
            or ref == alt
        ):
            raise PredictorProvenanceError(
                f"{label}.ref/alt must be distinct uppercase A/C/G/U bases"
            )
        if source[position] != ref:
            raise PredictorProvenanceError(
                f"{label}.ref does not match normalized_source_sequence"
            )
        if candidate[position] != alt:
            raise PredictorProvenanceError(
                f"{label}.alt does not match normalized_candidate_sequence"
            )
        replay[position] = alt
        edits.append({"op": "SUB", "pos": position, "ref": ref, "alt": alt})

    if "".join(replay) != candidate:
        raise PredictorProvenanceError(
            "predictor_input.edit_list exact replay does not reproduce "
            "normalized_candidate_sequence"
        )
    return {
        "endpoint": endpoint,
        "publication_context_id": publication_context_id,
        "region": region,
        "normalized_source_sequence": source,
        "normalized_candidate_sequence": candidate,
        "edit_list": edits,
        "edit_count": edit_count,
    }


def _development_ref(value: Mapping, *, label: str) -> dict:
    try:
        reference = validate_artifact_ref(value)
    except ArtifactGraphError as exc:
        raise PredictorProvenanceError(str(exc)) from exc
    parts = [part.lower() for part in PurePosixPath(reference["relative_path"]).parts]
    if "development" not in parts:
        raise PredictorProvenanceError(
            f"{label} must be beneath a literal development namespace"
        )
    if any(
        part in _PROHIBITED_PATH_PARTS
        or "internal_test" in part
        or "gse246381" in part
        for part in parts
    ):
        raise PredictorProvenanceError(
            f"{label} enters a prohibited protected-partition namespace"
        )
    return reference


def _read_ref(run_root: Path, reference: Mapping, *, label: str) -> bytes:
    normalized = _development_ref(reference, label=label)
    try:
        return read_artifact_ref(Path(run_root), normalized)
    except ArtifactGraphError as exc:
        raise PredictorProvenanceError(str(exc)) from exc


def _normalize_input_row(value: Mapping) -> dict:
    _strict_keys(value, _INPUT_ROW_KEYS, label="predictor input row")
    if value.get("schema_version") != PREDICTOR_INPUT_ROW_SCHEMA:
        raise PredictorProvenanceError("unsupported predictor input-row schema")
    partition = value.get("partition")
    if partition not in {TRAIN_PARTITION, DEVELOPMENT_PARTITION}:
        raise PredictorProvenanceError(
            "predictor input row must be ordinary train or development"
        )
    target = value.get("training_target")
    if partition == TRAIN_PARTITION:
        target = _finite_float(target, label="training_target")
    elif target is not None:
        raise PredictorProvenanceError(
            "development inference row cannot contain a training target"
        )
    return {
        "schema_version": PREDICTOR_INPUT_ROW_SCHEMA,
        "row_id": _digest(value.get("row_id"), label="predictor input row_id"),
        "partition": partition,
        "estimand_id": _identifier(
            value.get("estimand_id"), label="predictor input estimand_id"
        ),
        "cell_id": _identifier(value.get("cell_id"), label="predictor input cell_id"),
        "predictor_input": _normalize_primary_predictor_input(
            value.get("predictor_input")
        ),
        "training_target": target,
    }


def normalize_predictor_input_rows(rows: Sequence[Mapping]) -> list[dict]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise PredictorProvenanceError(
            "predictor input artifact requires at least one row"
        )
    normalized = [_normalize_input_row(row) for row in rows]
    row_ids = [row["row_id"] for row in normalized]
    if len(row_ids) != len(set(row_ids)):
        raise PredictorProvenanceError("predictor input row IDs must be unique")
    normalized.sort(key=lambda row: row["row_id"])
    return normalized


def predictor_input_jsonl_bytes(rows: Sequence[Mapping]) -> bytes:
    normalized = normalize_predictor_input_rows(rows)
    return b"".join(
        (
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for row in normalized
    )


def _unique_json_object(pairs) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_predictor_input_jsonl(raw: bytes) -> list[dict]:
    if not isinstance(raw, bytes) or not raw or not raw.endswith(b"\n"):
        raise PredictorProvenanceError(
            "predictor input JSONL must be non-empty and end with LF"
        )
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise PredictorProvenanceError("predictor input JSONL is not UTF-8") from exc
    if not lines or any(not line for line in lines):
        raise PredictorProvenanceError("predictor input JSONL cannot contain blank rows")
    parsed = []
    for index, line in enumerate(lines, start=1):
        try:
            value = json.loads(line, object_pairs_hook=_unique_json_object)
        except (json.JSONDecodeError, ValueError) as exc:
            raise PredictorProvenanceError(
                f"predictor input row {index} is invalid: {exc}"
            ) from exc
        if not isinstance(value, Mapping):
            raise PredictorProvenanceError(
                f"predictor input row {index} must be an object"
            )
        parsed.append(_normalize_input_row(value))
    normalized = normalize_predictor_input_rows(parsed)
    if parsed != normalized:
        raise PredictorProvenanceError(
            "predictor input rows are not in canonical row_id order"
        )
    if predictor_input_jsonl_bytes(normalized) != raw:
        raise PredictorProvenanceError("predictor input JSONL bytes are not canonical")
    return normalized


def write_predictor_input_jsonl_exclusive(
    path: Path, rows: Sequence[Mapping]
) -> str:
    if "development" not in {part.lower() for part in Path(path).parts}:
        raise PredictorProvenanceError(
            "predictor input rows must be published beneath development"
        )
    raw = predictor_input_jsonl_bytes(rows)
    try:
        return atomic_publish_bytes(
            Path(path), raw, replay_validator=parse_predictor_input_jsonl
        )
    except ArtifactIOError as exc:
        raise PredictorProvenanceError(str(exc)) from exc


def _row_hashes(rows: Sequence[Mapping]) -> tuple[int, str, str]:
    row_ids = [str(row["row_id"]) for row in rows]
    ordered_raw = json.dumps(
        row_ids, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    set_raw = json.dumps(
        sorted(row_ids), separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return (
        len(row_ids),
        hashlib.sha256(ordered_raw).hexdigest(),
        hashlib.sha256(set_raw).hexdigest(),
    )


def _verifier_identity(value: Mapping) -> dict:
    _strict_keys(value, _VERIFIER_IDENTITY_KEYS, label="verifier_identity")
    verifier_id = value.get("verifier_id")
    if not isinstance(verifier_id, str) or not _VERIFIER_ID.fullmatch(verifier_id):
        raise PredictorProvenanceError("verifier_id is not a frozen safe identifier")
    if value.get("schema_version") != RECOMPUTATION_VERIFIER_SCHEMA:
        raise PredictorProvenanceError("unsupported recomputation verifier schema")
    return {
        "verifier_id": verifier_id,
        "schema_version": RECOMPUTATION_VERIFIER_SCHEMA,
        "verifier_code_sha256": _digest(
            value.get("verifier_code_sha256"), label="verifier_code_sha256"
        ),
        "expected_predictor_code_sha256": _digest(
            value.get("expected_predictor_code_sha256"),
            label="expected_predictor_code_sha256",
        ),
        "expected_predictor_config_sha256": _digest(
            value.get("expected_predictor_config_sha256"),
            label="expected_predictor_config_sha256",
        ),
    }


def verifier_identity(verifier: PredictorVerifier) -> dict:
    if not isinstance(verifier, PredictorVerifier) or not callable(verifier.recompute):
        raise PredictorProvenanceError("registered verifier is invalid")
    return _verifier_identity({
        "verifier_id": verifier.verifier_id,
        "schema_version": verifier.schema_version,
        "verifier_code_sha256": verifier.code_sha256,
        "expected_predictor_code_sha256": verifier.expected_predictor_code_sha256,
        "expected_predictor_config_sha256": verifier.expected_predictor_config_sha256,
    })


def _registered_verifier(
    identity: Mapping,
    verifiers: Mapping[str, PredictorVerifier] | None,
) -> PredictorVerifier:
    normalized = _verifier_identity(identity)
    if not isinstance(verifiers, Mapping):
        raise PredictorProvenanceError(
            "trusted predictor verifier registry is required for exact replay"
        )
    if normalized["verifier_id"] not in verifiers:
        raise PredictorProvenanceError(
            "verifier registry does not contain the frozen verifier identity"
        )
    verifier = verifiers[normalized["verifier_id"]]
    if verifier_identity(verifier) != normalized:
        raise PredictorProvenanceError(
            "registered verifier identity, schema, or expected hashes mismatch"
        )
    return verifier


def _development_checkpoint_common(
    payload: Mapping,
    *,
    expected_schema: str,
    expected_role: str,
    expected_keys: set[str],
) -> dict:
    """Validate bindings shared by staged-safe neural checkpoint envelopes."""

    _strict_keys(payload, expected_keys, label=expected_role.lower().replace("_", " "))
    if payload.get("schema_version") != expected_schema:
        raise PredictorProvenanceError(
            "legacy or unsupported neural checkpoint schema; staged development "
            "checkpoint role is required"
        )
    if payload.get("artifact_role") != expected_role:
        raise PredictorProvenanceError("development checkpoint artifact role is invalid")
    _digest(payload.get("run_start_sha256"), label="checkpoint run_start_sha256")
    _digest(
        payload.get("activation_manifest_sha256"),
        label="checkpoint activation_manifest_sha256",
    )
    _digest(
        payload.get("train_development_outcomes_sha256"),
        label="checkpoint train_development_outcomes_sha256",
    )
    authorities = _non_data_authority_hashes(
        payload.get("non_data_authority_hashes")
    )
    _method_id(payload.get("method_id"), label="checkpoint method_id")
    _identifier(payload.get("estimand_id"), label="checkpoint estimand_id")
    _digest(
        payload.get("estimand_binding_sha256"),
        label="checkpoint estimand_binding_sha256",
    )
    _identifier(payload.get("cell_id"), label="checkpoint cell_id")
    _seed(payload.get("seed"), label="checkpoint seed")
    _digest(
        payload.get("train_row_set_sha256"),
        label="checkpoint train_row_set_sha256",
    )
    _digest(
        payload.get("development_row_set_sha256"),
        label="checkpoint development_row_set_sha256",
    )
    code_sha = _digest(
        payload.get("predictor_code_sha256"),
        label="checkpoint predictor_code_sha256",
    )
    config_sha = _digest(
        payload.get("predictor_config_sha256"),
        label="checkpoint predictor_config_sha256",
    )
    if code_sha != authorities["analysis_code_tree_manifest_sha256"]:
        raise PredictorProvenanceError("checkpoint code authority binding mismatch")
    if config_sha != authorities["model_optimizer_seed_config_sha256"]:
        raise PredictorProvenanceError("checkpoint config authority binding mismatch")
    if payload.get("ordinary_only") is not True:
        raise PredictorProvenanceError("development checkpoint must remain ordinary-only")
    if payload.get("downstream_unlock") is not False:
        raise PredictorProvenanceError("development checkpoint cannot unlock downstream")
    return dict(payload)


def validate_development_checkpoint_selection(payload: Mapping) -> dict:
    """Validate a checkpoint-selection envelope containing no protected split."""

    normalized = _development_checkpoint_common(
        payload,
        expected_schema=DEVELOPMENT_CHECKPOINT_SELECTION_SCHEMA,
        expected_role="ORDINARY_DEVELOPMENT_CHECKPOINT_SELECTION",
        expected_keys=_DEVELOPMENT_SELECTION_KEYS,
    )
    _positive_integer(normalized.get("selected_epoch"), label="selected_epoch")
    return normalized


def build_development_checkpoint_selection(
    *,
    run_start_sha256: str,
    activation_manifest_sha256: str,
    train_development_outcomes_sha256: str,
    non_data_authority_hashes: Mapping,
    method_id: str,
    estimand_id: str,
    estimand_binding_sha256: str,
    cell_id: str,
    seed: int,
    train_row_set_sha256: str,
    development_row_set_sha256: str,
    predictor_code_sha256: str,
    predictor_config_sha256: str,
    selected_epoch: int,
) -> dict:
    return validate_development_checkpoint_selection({
        "schema_version": DEVELOPMENT_CHECKPOINT_SELECTION_SCHEMA,
        "artifact_role": "ORDINARY_DEVELOPMENT_CHECKPOINT_SELECTION",
        "run_start_sha256": run_start_sha256,
        "activation_manifest_sha256": activation_manifest_sha256,
        "train_development_outcomes_sha256": train_development_outcomes_sha256,
        "non_data_authority_hashes": dict(non_data_authority_hashes),
        "method_id": method_id,
        "estimand_id": estimand_id,
        "estimand_binding_sha256": estimand_binding_sha256,
        "cell_id": cell_id,
        "seed": seed,
        "train_row_set_sha256": train_row_set_sha256,
        "development_row_set_sha256": development_row_set_sha256,
        "predictor_code_sha256": predictor_code_sha256,
        "predictor_config_sha256": predictor_config_sha256,
        "selected_epoch": selected_epoch,
        "ordinary_only": True,
        "downstream_unlock": False,
    })


def write_development_checkpoint_selection_exclusive(
    path: Path, payload: Mapping
) -> str:
    try:
        return atomic_write_json_exclusive(
            Path(path), payload, validator=validate_development_checkpoint_selection
        )
    except ArtifactIOError as exc:
        raise PredictorProvenanceError(str(exc)) from exc


def validate_development_checkpoint(payload: Mapping) -> dict:
    """Validate a staged-safe checkpoint envelope around an opaque state blob."""

    normalized = _development_checkpoint_common(
        payload,
        expected_schema=DEVELOPMENT_CHECKPOINT_SCHEMA,
        expected_role="ORDINARY_DEVELOPMENT_CHECKPOINT",
        expected_keys=_DEVELOPMENT_CHECKPOINT_KEYS,
    )
    if normalized.get("information_set") != PRIMARY_INFORMATION_SET:
        raise PredictorProvenanceError("development checkpoint information set is invalid")
    _digest(
        normalized.get("information_set_fingerprint"),
        label="checkpoint information_set_fingerprint",
    )
    normalized["checkpoint_selection_ref"] = _development_ref(
        normalized.get("checkpoint_selection_ref"),
        label="checkpoint checkpoint_selection_ref",
    )
    normalized["state_ref"] = _development_ref(
        normalized.get("state_ref"), label="checkpoint state_ref"
    )
    architecture_id = normalized.get("architecture_id")
    if not isinstance(architecture_id, str) or not _VERIFIER_ID.fullmatch(
        architecture_id
    ):
        raise PredictorProvenanceError("checkpoint architecture_id is invalid")
    return normalized


def build_development_checkpoint(
    *,
    run_start_sha256: str,
    activation_manifest_sha256: str,
    train_development_outcomes_sha256: str,
    non_data_authority_hashes: Mapping,
    method_id: str,
    estimand_id: str,
    estimand_binding_sha256: str,
    cell_id: str,
    seed: int,
    train_row_set_sha256: str,
    development_row_set_sha256: str,
    information_set_fingerprint: str,
    predictor_code_sha256: str,
    predictor_config_sha256: str,
    checkpoint_selection_ref: Mapping,
    state_ref: Mapping,
    architecture_id: str,
) -> dict:
    return validate_development_checkpoint({
        "schema_version": DEVELOPMENT_CHECKPOINT_SCHEMA,
        "artifact_role": "ORDINARY_DEVELOPMENT_CHECKPOINT",
        "run_start_sha256": run_start_sha256,
        "activation_manifest_sha256": activation_manifest_sha256,
        "train_development_outcomes_sha256": train_development_outcomes_sha256,
        "non_data_authority_hashes": dict(non_data_authority_hashes),
        "method_id": method_id,
        "estimand_id": estimand_id,
        "estimand_binding_sha256": estimand_binding_sha256,
        "cell_id": cell_id,
        "seed": seed,
        "train_row_set_sha256": train_row_set_sha256,
        "development_row_set_sha256": development_row_set_sha256,
        "information_set": PRIMARY_INFORMATION_SET,
        "information_set_fingerprint": information_set_fingerprint,
        "predictor_code_sha256": predictor_code_sha256,
        "predictor_config_sha256": predictor_config_sha256,
        "checkpoint_selection_ref": dict(checkpoint_selection_ref),
        "state_ref": dict(state_ref),
        "architecture_id": architecture_id,
        "ordinary_only": True,
        "downstream_unlock": False,
    })


def write_development_checkpoint_exclusive(path: Path, payload: Mapping) -> str:
    try:
        return atomic_write_json_exclusive(
            Path(path), payload, validator=validate_development_checkpoint
        )
    except ArtifactIOError as exc:
        raise PredictorProvenanceError(str(exc)) from exc


def _parse_strict_json_artifact(raw: bytes, *, label: str) -> dict:
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_json_object)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise PredictorProvenanceError(
            f"{label} is not strict development-only JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise PredictorProvenanceError(f"{label} must be a JSON object")
    return payload


def _normalize_scalar_artifacts(
    run_root: Path | None, value: Mapping, *, label: str
) -> dict:
    _strict_keys(value, _SCALAR_ARTIFACT_KEYS, label=label)
    if value.get("kind") != SCALAR_KIND:
        raise PredictorProvenanceError("scalar predictor kind is invalid")
    result = {
        "kind": SCALAR_KIND,
        "fitted_state_ref": _development_ref(
            value.get("fitted_state_ref"), label=f"{label}.fitted_state_ref"
        ),
        "fit_config_ref": _development_ref(
            value.get("fit_config_ref"), label=f"{label}.fit_config_ref"
        ),
    }
    if run_root is not None:
        _read_ref(run_root, result["fitted_state_ref"], label="scalar fitted state")
        _read_ref(run_root, result["fit_config_ref"], label="scalar fit config")
    return result


def _normalize_neural_artifacts(
    run_root: Path | None,
    value: Mapping,
    *,
    label: str,
    prediction_ref: Mapping,
) -> dict:
    _strict_keys(value, _NEURAL_ARTIFACT_KEYS, label=label)
    if value.get("kind") != NEURAL_KIND:
        raise PredictorProvenanceError("neural predictor kind is invalid")
    if value.get("seeds") != list(FROZEN_NEURAL_SEEDS):
        raise PredictorProvenanceError("neural predictor must bind the five frozen seeds")
    if value.get("ensemble_rule") != ENSEMBLE_RULE:
        raise PredictorProvenanceError("neural ensemble rule is invalid")
    if value.get("ensemble_variance_formula") != ENSEMBLE_VARIANCE_FORMULA:
        raise PredictorProvenanceError("neural ensemble variance formula is invalid")
    if value.get("variance_floor") != VARIANCE_FLOOR:
        raise PredictorProvenanceError("neural ensemble variance floor is invalid")
    ensemble_ref = _development_ref(
        value.get("ensemble_prediction_ref"),
        label=f"{label}.ensemble_prediction_ref",
    )
    if ensemble_ref != prediction_ref:
        raise PredictorProvenanceError(
            "neural ensemble prediction ref must equal the committed cell prediction"
        )
    seed_artifacts = value.get("seed_artifacts")
    if isinstance(seed_artifacts, (str, bytes)) or not isinstance(
        seed_artifacts, Sequence
    ):
        raise PredictorProvenanceError("seed_artifacts must be a sequence")
    if len(seed_artifacts) != len(FROZEN_NEURAL_SEEDS):
        raise PredictorProvenanceError("neural predictor seed artifacts are incomplete")
    normalized_seeds = []
    for expected_seed, item in zip(FROZEN_NEURAL_SEEDS, seed_artifacts):
        _strict_keys(item, _SEED_ARTIFACT_KEYS, label=f"{label} seed artifact")
        if item.get("seed") != expected_seed:
            raise PredictorProvenanceError(
                "neural seed artifacts do not follow the frozen seed order"
            )
        normalized_item = {
            "seed": expected_seed,
            "checkpoint_selection_ref": _development_ref(
                item.get("checkpoint_selection_ref"),
                label="checkpoint selection reference",
            ),
            "checkpoint_ref": _development_ref(
                item.get("checkpoint_ref"), label="checkpoint reference"
            ),
            "development_prediction_ref": _development_ref(
                item.get("development_prediction_ref"),
                label="seed development prediction reference",
            ),
        }
        if run_root is not None:
            _read_ref(
                run_root,
                normalized_item["checkpoint_selection_ref"],
                label="checkpoint selection",
            )
            _read_ref(
                run_root,
                normalized_item["checkpoint_ref"],
                label="checkpoint",
            )
            _read_ref(
                run_root,
                normalized_item["development_prediction_ref"],
                label="seed development prediction",
            )
        normalized_seeds.append(normalized_item)
    return {
        "kind": NEURAL_KIND,
        "seeds": list(FROZEN_NEURAL_SEEDS),
        "seed_artifacts": normalized_seeds,
        "ensemble_prediction_ref": ensemble_ref,
        "ensemble_rule": ENSEMBLE_RULE,
        "ensemble_variance_formula": ENSEMBLE_VARIANCE_FORMULA,
        "variance_floor": VARIANCE_FLOOR,
    }


def _normalize_artifacts(
    run_root: Path | None,
    value: Mapping,
    *,
    label: str,
    prediction_ref: Mapping,
) -> dict:
    if not isinstance(value, Mapping):
        raise PredictorProvenanceError(f"{label} must be an object")
    kind = value.get("kind")
    if kind == SCALAR_KIND:
        return _normalize_scalar_artifacts(run_root, value, label=label)
    if kind == NEURAL_KIND:
        return _normalize_neural_artifacts(
            run_root, value, label=label, prediction_ref=prediction_ref
        )
    raise PredictorProvenanceError("unsupported predictor artifact kind")


def _prediction_rows(run_root: Path, reference: Mapping, *, label: str) -> list[dict]:
    raw = _read_ref(run_root, reference, label=label)
    try:
        return parse_prediction_jsonl(raw)
    except ArtifactGraphError as exc:
        raise PredictorProvenanceError(str(exc)) from exc


def _compare_prediction_rows(
    actual: Sequence[Mapping], expected: Sequence[Mapping], *, label: str
) -> None:
    try:
        actual_raw = prediction_jsonl_bytes(actual)
        expected_raw = prediction_jsonl_bytes(expected)
    except ArtifactGraphError as exc:
        raise PredictorProvenanceError(str(exc)) from exc
    if actual_raw != expected_raw:
        raise PredictorProvenanceError(
            f"{label} does not match exact predictor recomputation"
        )


def _recompute_neural_ensemble(seed_rows: Sequence[Sequence[Mapping]]) -> list[dict]:
    if len(seed_rows) != len(FROZEN_NEURAL_SEEDS):
        raise PredictorProvenanceError("neural ensemble seed predictions are incomplete")
    row_ids = [row["row_id"] for row in seed_rows[0]]
    if not row_ids:
        raise PredictorProvenanceError("neural ensemble predictions are empty")
    by_seed = []
    for rows in seed_rows:
        mapping = {row["row_id"]: row for row in rows}
        if len(mapping) != len(rows) or set(mapping) != set(row_ids):
            raise PredictorProvenanceError(
                "neural seed prediction row coverage differs across seeds"
            )
        if any(mapping[row_id]["predicted_delta_logvar"] is None for row_id in row_ids):
            raise PredictorProvenanceError(
                "neural seed predictions require finite log-variance"
            )
        by_seed.append(mapping)
    output = []
    denominator = float(len(by_seed))
    for row_id in row_ids:
        means = [float(mapping[row_id]["predicted_delta_mean"]) for mapping in by_seed]
        logvars = [
            float(mapping[row_id]["predicted_delta_logvar"]) for mapping in by_seed
        ]
        mean = math.fsum(means) / denominator
        variance = (
            math.fsum(
                math.exp(logvar) + value * value
                for value, logvar in zip(means, logvars)
            )
            / denominator
            - mean * mean
        )
        output.append({
            "schema_version": "r3x_prediction_row_v1",
            "row_id": row_id,
            "predicted_delta_mean": mean,
            "predicted_delta_logvar": math.log(max(variance, VARIANCE_FLOOR)),
        })
    return output


def _normalize_cell(
    run_root: Path | None,
    value: Mapping,
    *,
    estimand_id: str,
    label: str,
) -> dict:
    _strict_keys(
        value, _INPUT_CELL_KEYS if run_root is not None else _OUTPUT_CELL_KEYS, label=label
    )
    cell_id = _identifier(value.get("cell_id"), label=f"{label}.cell_id")
    train_ref = _development_ref(value.get("train_rows_ref"), label="train rows ref")
    development_ref = _development_ref(
        value.get("development_rows_ref"), label="development rows ref"
    )
    prediction_ref = _development_ref(
        value.get("prediction_jsonl_ref"), label="development prediction ref"
    )
    artifacts = _normalize_artifacts(
        run_root,
        value.get("predictor_artifacts"),
        label=f"{label}.predictor_artifacts",
        prediction_ref=prediction_ref,
    )
    if run_root is not None:
        train_rows = parse_predictor_input_jsonl(
            _read_ref(run_root, train_ref, label="train predictor rows")
        )
        development_rows = parse_predictor_input_jsonl(
            _read_ref(run_root, development_ref, label="development predictor rows")
        )
        predictions = _prediction_rows(
            run_root, prediction_ref, label="development cell predictions"
        )
        for rows, partition, row_label in (
            (train_rows, TRAIN_PARTITION, "train"),
            (development_rows, DEVELOPMENT_PARTITION, "development"),
        ):
            if any(row["partition"] != partition for row in rows):
                raise PredictorProvenanceError(
                    f"{row_label} predictor rows reference the wrong partition"
                )
            if any(
                row["estimand_id"] != estimand_id or row["cell_id"] != cell_id
                for row in rows
            ):
                raise PredictorProvenanceError(
                    f"{row_label} predictor rows are bound to wrong estimand/cell"
                )
        train_ids = {row["row_id"] for row in train_rows}
        development_ids = [row["row_id"] for row in development_rows]
        if train_ids.intersection(development_ids):
            raise PredictorProvenanceError(
                "train and development predictor rows must be disjoint"
            )
        if [row["row_id"] for row in predictions] != development_ids:
            raise PredictorProvenanceError(
                "development prediction rows must exactly match inference-row order"
            )
        train_count, train_ordered, train_set = _row_hashes(train_rows)
        dev_count, dev_ordered, dev_set = _row_hashes(development_rows)
        return {
            "cell_id": cell_id,
            "train_rows_ref": train_ref,
            "development_rows_ref": development_ref,
            "prediction_jsonl_ref": prediction_ref,
            "predictor_artifacts": artifacts,
            "train_n_rows": train_count,
            "train_ordered_row_ids_sha256": train_ordered,
            "train_row_set_sha256": train_set,
            "development_n_rows": dev_count,
            "development_ordered_row_ids_sha256": dev_ordered,
            "development_row_set_sha256": dev_set,
        }
    normalized = dict(value)
    normalized["predictor_artifacts"] = artifacts
    for field in ("train_n_rows", "development_n_rows"):
        count = normalized.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise PredictorProvenanceError(f"{label}.{field} must be positive")
    for field in (
        "train_ordered_row_ids_sha256",
        "train_row_set_sha256",
        "development_ordered_row_ids_sha256",
        "development_row_set_sha256",
    ):
        _digest(normalized.get(field), label=f"{label}.{field}")
    return normalized


def _normalize_estimands(
    run_root: Path | None, values: Sequence[Mapping]
) -> list[dict]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise PredictorProvenanceError("predictor provenance estimands are empty")
    normalized = []
    seen_development_rows: set[str] = set()
    for index, item in enumerate(values):
        _strict_keys(item, _ESTIMAND_KEYS, label=f"estimand {index}")
        estimand_id = _identifier(
            item.get("estimand_id"), label=f"estimand {index}.estimand_id"
        )
        estimand_binding = _digest(
            item.get("estimand_binding_sha256"),
            label=f"estimand {estimand_id}.estimand_binding_sha256",
        )
        cells = item.get("cells")
        if isinstance(cells, (str, bytes)) or not isinstance(cells, Sequence) or not cells:
            raise PredictorProvenanceError("every predictor estimand needs cells")
        normalized_cells = [
            _normalize_cell(
                run_root,
                cell,
                estimand_id=estimand_id,
                label=f"estimand {estimand_id} cell {cell_index}",
            )
            for cell_index, cell in enumerate(cells)
        ]
        normalized_cells.sort(key=lambda cell: cell["cell_id"])
        if len({cell["cell_id"] for cell in normalized_cells}) != len(
            normalized_cells
        ):
            raise PredictorProvenanceError("predictor cell IDs must be unique")
        if run_root is not None:
            for cell in normalized_cells:
                rows = parse_predictor_input_jsonl(
                    _read_ref(
                        run_root,
                        cell["development_rows_ref"],
                        label="development predictor rows",
                    )
                )
                row_ids = {row["row_id"] for row in rows}
                if seen_development_rows.intersection(row_ids):
                    raise PredictorProvenanceError(
                        "one development row appears in multiple predictor cells"
                    )
                seen_development_rows.update(row_ids)
        normalized.append({
            "estimand_id": estimand_id,
            "estimand_binding_sha256": estimand_binding,
            "cells": normalized_cells,
        })
    normalized.sort(key=lambda item: item["estimand_id"])
    if len({item["estimand_id"] for item in normalized}) != len(normalized):
        raise PredictorProvenanceError("predictor estimand IDs must be unique")
    return normalized


def _recompute_estimands(run_root: Path, values: Sequence[Mapping]) -> list[dict]:
    """Re-read every output-cell ref and reproduce all derived row bindings."""

    schema_validated = _normalize_estimands(None, values)
    replayed = []
    seen_development_rows: set[str] = set()
    for estimand in schema_validated:
        cells = []
        for cell in estimand["cells"]:
            recomputed = _normalize_cell(
                Path(run_root),
                {key: cell[key] for key in _INPUT_CELL_KEYS},
                estimand_id=estimand["estimand_id"],
                label=f"replay {estimand['estimand_id']} {cell['cell_id']}",
            )
            if recomputed != cell:
                raise PredictorProvenanceError(
                    "predictor row hashes do not match exact row-artifact replay"
                )
            rows = parse_predictor_input_jsonl(
                _read_ref(
                    run_root,
                    cell["development_rows_ref"],
                    label="development predictor rows",
                )
            )
            row_ids = {row["row_id"] for row in rows}
            if seen_development_rows.intersection(row_ids):
                raise PredictorProvenanceError(
                    "one development row appears in multiple predictor cells"
                )
            seen_development_rows.update(row_ids)
            cells.append(recomputed)
        replayed.append({
            "estimand_id": estimand["estimand_id"],
            "estimand_binding_sha256": estimand["estimand_binding_sha256"],
            "cells": cells,
        })
    return replayed


def _validate_neural_checkpoint_chain(
    run_root: Path,
    provenance: Mapping,
    estimand: Mapping,
    cell: Mapping,
) -> None:
    """Reject legacy checkpoints and bind each staged-safe envelope exactly."""

    artifacts = cell["predictor_artifacts"]
    if artifacts["kind"] != NEURAL_KIND:
        return
    common_expected = {
        "run_start_sha256": provenance["run_start_sha256"],
        "activation_manifest_sha256": provenance["activation_manifest_sha256"],
        "train_development_outcomes_sha256": provenance[
            "train_development_outcomes_sha256"
        ],
        "non_data_authority_hashes": provenance["non_data_authority_hashes"],
        "method_id": provenance["method_id"],
        "estimand_id": estimand["estimand_id"],
        "estimand_binding_sha256": estimand["estimand_binding_sha256"],
        "cell_id": cell["cell_id"],
        "train_row_set_sha256": cell["train_row_set_sha256"],
        "development_row_set_sha256": cell["development_row_set_sha256"],
        "predictor_code_sha256": provenance["predictor_code_sha256"],
        "predictor_config_sha256": provenance["predictor_config_sha256"],
    }
    for seed_artifact in artifacts["seed_artifacts"]:
        selection_raw = _read_ref(
            run_root,
            seed_artifact["checkpoint_selection_ref"],
            label="development checkpoint selection",
        )
        selection = validate_development_checkpoint_selection(
            _parse_strict_json_artifact(
                selection_raw, label="development checkpoint selection"
            )
        )
        if canonical_json_bytes(selection) != selection_raw:
            raise PredictorProvenanceError(
                "development checkpoint selection bytes are not canonical"
            )
        checkpoint_raw = _read_ref(
            run_root,
            seed_artifact["checkpoint_ref"],
            label="development checkpoint",
        )
        checkpoint = validate_development_checkpoint(
            _parse_strict_json_artifact(
                checkpoint_raw, label="development checkpoint"
            )
        )
        if canonical_json_bytes(checkpoint) != checkpoint_raw:
            raise PredictorProvenanceError(
                "development checkpoint bytes are not canonical"
            )
        expected = {**common_expected, "seed": seed_artifact["seed"]}
        for field, expected_value in expected.items():
            if checkpoint[field] != expected_value or selection[field] != expected_value:
                raise PredictorProvenanceError(
                    f"development checkpoint {field} binding mismatch"
                )
        if checkpoint["information_set_fingerprint"] != provenance[
            "information_set_fingerprint"
        ]:
            raise PredictorProvenanceError(
                "development checkpoint information fingerprint mismatch"
            )
        if checkpoint["checkpoint_selection_ref"] != seed_artifact[
            "checkpoint_selection_ref"
        ]:
            raise PredictorProvenanceError(
                "development checkpoint selection reference mismatch"
            )
        _read_ref(run_root, checkpoint["state_ref"], label="development model state")


def _verify_cell(
    run_root: Path,
    provenance: Mapping,
    estimand: Mapping,
    cell: Mapping,
    verifier: PredictorVerifier,
) -> None:
    _validate_neural_checkpoint_chain(run_root, provenance, estimand, cell)
    try:
        recomputed = verifier.recompute(Path(run_root), dict(cell))
    except PredictorProvenanceError:
        raise
    except Exception as exc:
        raise PredictorProvenanceError(
            f"predictor verifier failed closed: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(recomputed, Mapping):
        raise PredictorProvenanceError("predictor verifier must return an object")
    artifacts = cell["predictor_artifacts"]
    expected = _prediction_rows(
        run_root, cell["prediction_jsonl_ref"], label="committed prediction"
    )
    if artifacts["kind"] == SCALAR_KIND:
        if set(recomputed) != {"prediction"}:
            raise PredictorProvenanceError(
                "scalar verifier must return exactly the prediction output"
            )
        _compare_prediction_rows(
            recomputed["prediction"], expected, label="scalar prediction"
        )
        return
    expected_keys = {str(seed) for seed in FROZEN_NEURAL_SEEDS}
    if set(recomputed) != expected_keys:
        raise PredictorProvenanceError(
            "neural verifier must return exactly all five frozen seed outputs"
        )
    exact_seed_rows = []
    for item in artifacts["seed_artifacts"]:
        seed = str(item["seed"])
        committed = _prediction_rows(
            run_root,
            item["development_prediction_ref"],
            label=f"seed {seed} prediction",
        )
        _compare_prediction_rows(
            recomputed[seed], committed, label=f"neural seed {seed} prediction"
        )
        exact_seed_rows.append(committed)
    ensemble = _recompute_neural_ensemble(exact_seed_rows)
    _compare_prediction_rows(ensemble, expected, label="five-seed ensemble prediction")


def _replay_payload(
    run_root: Path,
    payload: Mapping,
    *,
    verifiers: Mapping[str, PredictorVerifier] | None,
) -> dict:
    validated = validate_predictor_provenance(payload)
    verifier = _registered_verifier(validated["verifier_identity"], verifiers)
    if validated["predictor_code_sha256"] != verifier.expected_predictor_code_sha256:
        raise PredictorProvenanceError("predictor code hash differs from verifier freeze")
    if validated["predictor_config_sha256"] != verifier.expected_predictor_config_sha256:
        raise PredictorProvenanceError("predictor config hash differs from verifier freeze")
    replayed_estimands = _recompute_estimands(
        Path(run_root), validated["estimands"]
    )
    if replayed_estimands != validated["estimands"]:
        raise PredictorProvenanceError(
            "predictor input rows or artifact hashes changed during exact replay"
        )
    for estimand in replayed_estimands:
        for cell in estimand["cells"]:
            _verify_cell(Path(run_root), validated, estimand, cell, verifier)
    return validated


def build_predictor_provenance(
    run_root: Path,
    *,
    run_start_sha256: str,
    activation_manifest_sha256: str,
    train_development_outcomes_sha256: str,
    non_data_authority_hashes: Mapping,
    method_id: str,
    information_set_fingerprint: str,
    predictor_code_sha256: str,
    predictor_config_sha256: str,
    verifier: PredictorVerifier,
    estimands: Sequence[Mapping],
) -> dict:
    """Build and exactly recompute one method's development predictor commit."""

    development_data_sha = _digest(
        train_development_outcomes_sha256,
        label="train_development_outcomes_sha256",
    )
    authorities = _non_data_authority_hashes(non_data_authority_hashes)
    code_sha = _digest(predictor_code_sha256, label="predictor_code_sha256")
    config_sha = _digest(predictor_config_sha256, label="predictor_config_sha256")
    if code_sha != authorities["analysis_code_tree_manifest_sha256"]:
        raise PredictorProvenanceError(
            "predictor code hash must equal the frozen analysis code-tree authority"
        )
    if config_sha != authorities["model_optimizer_seed_config_sha256"]:
        raise PredictorProvenanceError(
            "predictor config hash must equal the frozen model/config authority"
        )
    identity = verifier_identity(verifier)
    if identity["expected_predictor_code_sha256"] != code_sha:
        raise PredictorProvenanceError("verifier expected predictor code hash mismatch")
    if identity["expected_predictor_config_sha256"] != config_sha:
        raise PredictorProvenanceError("verifier expected predictor config hash mismatch")
    normalized_estimands = _normalize_estimands(Path(run_root), estimands)
    payload = {
        "schema_version": PREDICTOR_PROVENANCE_SCHEMA,
        "artifact_role": PROVENANCE_ROLE,
        "status": PROVENANCE_STATUS,
        "run_start_sha256": _digest(run_start_sha256, label="run_start_sha256"),
        "activation_manifest_sha256": _digest(
            activation_manifest_sha256, label="activation_manifest_sha256"
        ),
        "train_development_outcomes_sha256": development_data_sha,
        "non_data_authority_hashes": authorities,
        "method_id": _method_id(method_id),
        "partition": DEVELOPMENT_PARTITION,
        "information_set": PRIMARY_INFORMATION_SET,
        "information_set_fingerprint": _digest(
            information_set_fingerprint, label="information_set_fingerprint"
        ),
        "predictor_code_sha256": code_sha,
        "predictor_config_sha256": config_sha,
        "verifier_identity": identity,
        "estimands": normalized_estimands,
        "ordinary_only": True,
        "downstream_unlock": False,
    }
    validated = validate_predictor_provenance(payload)
    return _replay_payload(
        Path(run_root), validated, verifiers={verifier.verifier_id: verifier}
    )


def validate_predictor_provenance(payload: Mapping) -> dict:
    """Validate v2 schema; exact prediction recomputation is a separate replay."""

    _strict_keys(payload, _PROVENANCE_KEYS, label="predictor provenance")
    if payload.get("schema_version") != PREDICTOR_PROVENANCE_SCHEMA:
        raise PredictorProvenanceError("unsupported predictor provenance schema")
    if payload.get("artifact_role") != PROVENANCE_ROLE:
        raise PredictorProvenanceError("predictor provenance role is invalid")
    if payload.get("status") != PROVENANCE_STATUS:
        raise PredictorProvenanceError("predictor provenance is not replay-complete")
    _digest(payload.get("run_start_sha256"), label="run_start_sha256")
    _digest(
        payload.get("activation_manifest_sha256"),
        label="activation_manifest_sha256",
    )
    _digest(
        payload.get("train_development_outcomes_sha256"),
        label="train_development_outcomes_sha256",
    )
    authorities = _non_data_authority_hashes(
        payload.get("non_data_authority_hashes")
    )
    _method_id(payload.get("method_id"))
    if payload.get("partition") != DEVELOPMENT_PARTITION:
        raise PredictorProvenanceError("predictor provenance is not development-only")
    if payload.get("information_set") != PRIMARY_INFORMATION_SET:
        raise PredictorProvenanceError("predictor provenance information set is invalid")
    _digest(
        payload.get("information_set_fingerprint"),
        label="information_set_fingerprint",
    )
    code_sha = _digest(
        payload.get("predictor_code_sha256"), label="predictor_code_sha256"
    )
    config_sha = _digest(
        payload.get("predictor_config_sha256"), label="predictor_config_sha256"
    )
    if code_sha != authorities["analysis_code_tree_manifest_sha256"]:
        raise PredictorProvenanceError("predictor code authority binding mismatch")
    if config_sha != authorities["model_optimizer_seed_config_sha256"]:
        raise PredictorProvenanceError("predictor config authority binding mismatch")
    identity = _verifier_identity(payload.get("verifier_identity"))
    if identity["expected_predictor_code_sha256"] != code_sha:
        raise PredictorProvenanceError("verifier predictor-code binding mismatch")
    if identity["expected_predictor_config_sha256"] != config_sha:
        raise PredictorProvenanceError("verifier predictor-config binding mismatch")
    estimands = _normalize_estimands(None, payload.get("estimands"))
    if estimands != payload.get("estimands"):
        raise PredictorProvenanceError("predictor estimands are not canonical")
    if payload.get("ordinary_only") is not True:
        raise PredictorProvenanceError("predictor provenance must remain ordinary-only")
    if payload.get("downstream_unlock") is not False:
        raise PredictorProvenanceError("predictor provenance cannot unlock downstream")
    return dict(payload)


def canonical_predictor_provenance_bytes(payload: Mapping) -> bytes:
    try:
        return canonical_json_bytes(validate_predictor_provenance(payload))
    except ArtifactIOError as exc:
        raise PredictorProvenanceError(str(exc)) from exc


def _assert_output_within_run_root(run_root: Path, path: Path) -> None:
    try:
        root = Path(run_root).resolve(strict=True)
        candidate = Path(path).resolve(strict=False)
    except OSError as exc:
        raise PredictorProvenanceError(
            f"predictor provenance path cannot be resolved: {exc}"
        ) from exc
    if not root.is_dir() or not candidate.is_relative_to(root):
        raise PredictorProvenanceError(
            "predictor provenance output must remain beneath the run root"
        )
    if "development" not in {part.lower() for part in candidate.parts}:
        raise PredictorProvenanceError(
            "predictor provenance output must be beneath development"
        )


def write_predictor_provenance_exclusive(
    run_root: Path,
    path: Path,
    payload: Mapping,
    *,
    verifiers: Mapping[str, PredictorVerifier] | None,
) -> str:
    """Exactly recompute predictors, then publish the immutable commit."""

    _assert_output_within_run_root(run_root, path)
    replayed = _replay_payload(Path(run_root), payload, verifiers=verifiers)
    try:
        return atomic_write_json_exclusive(
            Path(path), replayed, validator=validate_predictor_provenance
        )
    except ArtifactIOError as exc:
        raise PredictorProvenanceError(str(exc)) from exc


def _parse_json_object(raw: bytes) -> dict:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_json_object)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise PredictorProvenanceError(
            f"predictor provenance is not strict JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise PredictorProvenanceError("predictor provenance must be a JSON object")
    return value


def replay_predictor_provenance(
    run_root: Path,
    provenance_ref: Mapping,
    *,
    verifiers: Mapping[str, PredictorVerifier] | None,
    expected_run_start_sha256: str | None = None,
    expected_activation_manifest_sha256: str | None = None,
    expected_train_development_outcomes_sha256: str | None = None,
    expected_non_data_authority_hashes: Mapping | None = None,
    expected_method_id: str | None = None,
    expected_information_set_fingerprint: str | None = None,
) -> dict:
    """Read exact provenance bytes and independently recompute every output."""

    reference = _development_ref(
        provenance_ref, label="predictor provenance reference"
    )
    raw = _read_ref(run_root, reference, label="predictor provenance")
    payload = validate_predictor_provenance(_parse_json_object(raw))
    if canonical_predictor_provenance_bytes(payload) != raw:
        raise PredictorProvenanceError("predictor provenance bytes are not canonical")
    expected_pairs = (
        ("run_start_sha256", expected_run_start_sha256),
        ("activation_manifest_sha256", expected_activation_manifest_sha256),
        ("method_id", expected_method_id),
        ("information_set_fingerprint", expected_information_set_fingerprint),
    )
    for field, expected in expected_pairs:
        if expected is not None and payload[field] != expected:
            raise PredictorProvenanceError(
                f"predictor provenance {field} binding mismatch"
            )
    if expected_train_development_outcomes_sha256 is not None:
        if payload["train_development_outcomes_sha256"] != _digest(
            expected_train_development_outcomes_sha256,
            label="expected_train_development_outcomes_sha256",
        ):
            raise PredictorProvenanceError(
                "predictor provenance train/development outcome authority mismatch"
            )
    if expected_non_data_authority_hashes is not None:
        if payload["non_data_authority_hashes"] != _non_data_authority_hashes(
            expected_non_data_authority_hashes
        ):
            raise PredictorProvenanceError(
                "predictor provenance non-data authority binding mismatch"
            )
    return _replay_payload(Path(run_root), payload, verifiers=verifiers)


def predictor_layout(payload: Mapping) -> list[dict]:
    """Expose only immutable selector bindings after strict schema validation."""

    validated = validate_predictor_provenance(payload)
    return [
        {
            "estimand_id": estimand["estimand_id"],
            "cells": [
                {
                    "cell_id": cell["cell_id"],
                    "prediction_jsonl_ref": cell["prediction_jsonl_ref"],
                    "development_n_rows": cell["development_n_rows"],
                    "development_ordered_row_ids_sha256": cell[
                        "development_ordered_row_ids_sha256"
                    ],
                    "development_row_set_sha256": cell[
                        "development_row_set_sha256"
                    ],
                }
                for cell in estimand["cells"]
            ],
        }
        for estimand in validated["estimands"]
    ]
