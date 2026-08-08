"""Immutable development-only checkpoint-selection evidence for R3X.

The artifact in this module is deliberately narrower than a training report.
It binds one estimand/method/variant/seed run to its train/development row-set
hashes and records only the three frozen development metrics needed to replay
checkpoint selection.  Internal-test observations, predictions, row IDs,
sealed access, and downstream permissions are outside this schema.
"""
from __future__ import annotations

import json
import math
import re
from numbers import Real
from pathlib import Path
from typing import Mapping, Sequence

from .artifact_io import (
    ArtifactIOError,
    atomic_write_json_exclusive,
    canonical_json_bytes,
    read_exact_regular_file,
)
from .checkpoint import (
    ALLOWED_TRAINING_VARIANTS,
    FROZEN_SEEDS,
    REQUIRED_AUTHORITY_HASHES,
)


SCHEMA = "r3x_checkpoint_selection_v1"
ARTIFACT_ROLE = "DEVELOPMENT_ONLY_CHECKPOINT_SELECTION_EVIDENCE"
SELECTION_DATA_ROLE = "ORDINARY_DEVELOPMENT_ONLY"
SIGN_GATE = 0.60
SELECTION_RULE = "SIGN_GATE_THEN_SPEARMAN_THEN_NLL_THEN_EARLIEST_EPOCH"
GATE_MET_PHASE = "AT_LEAST_ONE_DEVELOPMENT_EPOCH_MET_SIGN_GATE"
FALLBACK_PHASE = "NO_DEVELOPMENT_EPOCH_MET_SIGN_GATE_FALLBACK"

HEX64 = re.compile(r"^[0-9a-f]{64}$")
METHOD_ID = re.compile(r"^[A-Z][A-Z0-9_]*$")

ARTIFACT_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "run_start_sha256",
    "activation_manifest_sha256",
    "authority_hashes",
    "estimand_binding_sha256",
    "method_id",
    "training_variant_id",
    "seed",
    "partition_row_set_sha256",
    "training_config_sha256",
    "sign_gate",
    "selection_rule",
    "selection_data_role",
    "selection_phase",
    "lexicographic_selection_tuple_fields",
    "epoch_selection_records",
    "selected_epoch",
    "selected_development_metrics",
    "ordinary_only",
    "test_based_selection",
    "best_seed_selection",
    "internal_test_access",
    "predictions_included",
    "row_ids_included",
    "sealed_access",
    "downstream_unlock",
}
INPUT_EPOCH_KEYS = {
    "epoch",
    "macro_sign_accuracy",
    "macro_delta_spearman",
    "macro_gaussian_nll",
}
METRIC_KEYS = {
    "macro_sign_accuracy",
    "macro_delta_spearman",
    "macro_gaussian_nll",
}
EPOCH_RECORD_KEYS = {
    "epoch",
    "selection_data_role",
    "development_metrics",
    "sign_gate_met",
    "lexicographic_selection_tuple",
}
PARTITION_HASH_KEYS = {"train", "development"}

# These names are present only as explicit false boundary assertions.  Any
# other field that attempts to disclose or authorize them is rejected before
# ordinary strict-schema validation.
_ALLOWED_BOUNDARY_ASSERTION_KEYS = {
    "test_based_selection",
    "internal_test_access",
    "predictions_included",
    "row_ids_included",
    "sealed_access",
    "downstream_unlock",
}


class SelectionArtifactProtocolError(RuntimeError):
    """Fail-closed checkpoint-selection artifact or replay error."""


def _strict_keys(value: Mapping, expected: set[str], *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise SelectionArtifactProtocolError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise SelectionArtifactProtocolError(
            f"{label} keys differ from strict schema: {sorted(actual ^ expected)}"
        )


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise SelectionArtifactProtocolError(f"{label} must be 64 lowercase hex")
    return value


def _authority_hashes(value: Mapping) -> dict[str, str]:
    _strict_keys(
        value, set(REQUIRED_AUTHORITY_HASHES), label="authority_hashes"
    )
    return {
        name: _digest(value[name], label=name)
        for name in REQUIRED_AUTHORITY_HASHES
    }


def _partition_hashes(value: Mapping) -> dict[str, str]:
    _strict_keys(value, PARTITION_HASH_KEYS, label="partition_row_set_sha256")
    return {
        name: _digest(value[name], label=f"{name} row-set SHA-256")
        for name in ("train", "development")
    }


def _method_id(value: object) -> str:
    if not isinstance(value, str) or not METHOD_ID.fullmatch(value):
        raise SelectionArtifactProtocolError(
            "method_id must be a safe uppercase frozen identifier"
        )
    return value


def _seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in FROZEN_SEEDS:
        raise SelectionArtifactProtocolError(
            "seed must be one of the five frozen seeds"
        )
    return value


def _variant(value: object) -> str:
    if value not in ALLOWED_TRAINING_VARIANTS:
        raise SelectionArtifactProtocolError("training_variant_id is not frozen")
    return str(value)


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SelectionArtifactProtocolError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SelectionArtifactProtocolError(f"{label} must be a finite number")
    return result


def _assert_no_forbidden_disclosure_keys(value: object, *, path: str = "artifact") -> None:
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key)
            lowered = key.lower()
            if key not in _ALLOWED_BOUNDARY_ASSERTION_KEYS and (
                "internal_test" in lowered
                or lowered.startswith("test_")
                or lowered.endswith("_test")
                or "prediction" in lowered
                or "row_id" in lowered
                or "sealed" in lowered
                or "downstream" in lowered
            ):
                raise SelectionArtifactProtocolError(
                    f"forbidden non-development disclosure or permission key at {path}.{key}"
                )
            _assert_no_forbidden_disclosure_keys(nested, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            _assert_no_forbidden_disclosure_keys(nested, path=f"{path}[{index}]")


def _normalize_epoch_candidates(candidates: Sequence[Mapping]) -> list[dict]:
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise SelectionArtifactProtocolError("epoch_candidates must be a sequence")
    if not candidates:
        raise SelectionArtifactProtocolError("checkpoint selection refuses empty epochs")
    normalized = []
    seen_epochs: set[int] = set()
    for index, candidate in enumerate(candidates):
        _assert_no_forbidden_disclosure_keys(
            candidate, path=f"epoch_candidates[{index}]"
        )
        _strict_keys(candidate, INPUT_EPOCH_KEYS, label=f"epoch candidate {index}")
        epoch = candidate.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
            raise SelectionArtifactProtocolError(
                "epoch candidate requires a positive integer epoch"
            )
        if epoch in seen_epochs:
            raise SelectionArtifactProtocolError("epoch values must be unique")
        seen_epochs.add(epoch)
        sign = _finite_float(
            candidate.get("macro_sign_accuracy"), label="macro_sign_accuracy"
        )
        spearman = _finite_float(
            candidate.get("macro_delta_spearman"), label="macro_delta_spearman"
        )
        nll = _finite_float(
            candidate.get("macro_gaussian_nll"), label="macro_gaussian_nll"
        )
        if not 0.0 <= sign <= 1.0:
            raise SelectionArtifactProtocolError(
                "macro_sign_accuracy must be in [0, 1]"
            )
        if not -1.0 <= spearman <= 1.0:
            raise SelectionArtifactProtocolError(
                "macro_delta_spearman must be in [-1, 1]"
            )
        normalized.append({
            "epoch": epoch,
            "macro_sign_accuracy": sign,
            "macro_delta_spearman": spearman,
            "macro_gaussian_nll": nll,
        })
    return sorted(normalized, key=lambda item: item["epoch"])


def _build_payload(
    *,
    run_start_sha256: str,
    activation_manifest_sha256: str,
    authority_hashes: Mapping,
    estimand_binding_sha256: str,
    method_id: str,
    training_variant_id: str,
    seed: int,
    partition_row_set_sha256: Mapping,
    training_config_sha256: str,
    epoch_candidates: Sequence[Mapping],
    sign_gate: float,
) -> dict:
    if isinstance(sign_gate, bool) or not isinstance(sign_gate, Real):
        raise SelectionArtifactProtocolError("sign_gate must remain exactly 0.60")
    if not math.isfinite(float(sign_gate)) or float(sign_gate) != SIGN_GATE:
        raise SelectionArtifactProtocolError("sign_gate must remain exactly 0.60")
    candidates = _normalize_epoch_candidates(epoch_candidates)
    any_gate_met = any(
        item["macro_sign_accuracy"] >= SIGN_GATE for item in candidates
    )
    if any_gate_met:
        phase = GATE_MET_PHASE
        tuple_fields = [
            "sign_gate_met",
            "macro_delta_spearman",
            "negative_macro_gaussian_nll",
            "negative_epoch",
        ]

        def selection_tuple(item: Mapping) -> list[float]:
            return [
                1.0 if item["macro_sign_accuracy"] >= SIGN_GATE else 0.0,
                float(item["macro_delta_spearman"]),
                -float(item["macro_gaussian_nll"]),
                -float(item["epoch"]),
            ]

        status = "DEV_SIGN_GATE_MET"
    else:
        phase = FALLBACK_PHASE
        tuple_fields = [
            "macro_sign_accuracy",
            "macro_delta_spearman",
            "negative_macro_gaussian_nll",
            "negative_epoch",
        ]

        def selection_tuple(item: Mapping) -> list[float]:
            return [
                float(item["macro_sign_accuracy"]),
                float(item["macro_delta_spearman"]),
                -float(item["macro_gaussian_nll"]),
                -float(item["epoch"]),
            ]

        status = "HOLD_DEV_SIGN_GATE_NOT_MET"

    records = []
    for item in candidates:
        records.append({
            "epoch": item["epoch"],
            "selection_data_role": SELECTION_DATA_ROLE,
            "development_metrics": {
                key: item[key]
                for key in (
                    "macro_sign_accuracy",
                    "macro_delta_spearman",
                    "macro_gaussian_nll",
                )
            },
            "sign_gate_met": item["macro_sign_accuracy"] >= SIGN_GATE,
            "lexicographic_selection_tuple": selection_tuple(item),
        })
    selected_record = max(
        records, key=lambda record: tuple(record["lexicographic_selection_tuple"])
    )
    return {
        "schema_version": SCHEMA,
        "artifact_role": ARTIFACT_ROLE,
        "status": status,
        "run_start_sha256": _digest(
            run_start_sha256, label="run_start_sha256"
        ),
        "activation_manifest_sha256": _digest(
            activation_manifest_sha256, label="activation_manifest_sha256"
        ),
        "authority_hashes": _authority_hashes(authority_hashes),
        "estimand_binding_sha256": _digest(
            estimand_binding_sha256, label="estimand_binding_sha256"
        ),
        "method_id": _method_id(method_id),
        "training_variant_id": _variant(training_variant_id),
        "seed": _seed(seed),
        "partition_row_set_sha256": _partition_hashes(
            partition_row_set_sha256
        ),
        "training_config_sha256": _digest(
            training_config_sha256, label="training_config_sha256"
        ),
        "sign_gate": SIGN_GATE,
        "selection_rule": SELECTION_RULE,
        "selection_data_role": SELECTION_DATA_ROLE,
        "selection_phase": phase,
        "lexicographic_selection_tuple_fields": tuple_fields,
        "epoch_selection_records": records,
        "selected_epoch": selected_record["epoch"],
        "selected_development_metrics": dict(
            selected_record["development_metrics"]
        ),
        "ordinary_only": True,
        "test_based_selection": False,
        "best_seed_selection": False,
        "internal_test_access": False,
        "predictions_included": False,
        "row_ids_included": False,
        "sealed_access": False,
        "downstream_unlock": False,
    }


def build_checkpoint_selection_artifact(
    *,
    run_start_sha256: str,
    activation_manifest_sha256: str,
    authority_hashes: Mapping,
    estimand_binding_sha256: str,
    method_id: str,
    training_variant_id: str,
    seed: int,
    partition_row_set_sha256: Mapping,
    training_config_sha256: str,
    epoch_candidates: Sequence[Mapping],
    sign_gate: float = SIGN_GATE,
) -> dict:
    """Build one strict, replayable development-only selection artifact."""

    payload = _build_payload(
        run_start_sha256=run_start_sha256,
        activation_manifest_sha256=activation_manifest_sha256,
        authority_hashes=authority_hashes,
        estimand_binding_sha256=estimand_binding_sha256,
        method_id=method_id,
        training_variant_id=training_variant_id,
        seed=seed,
        partition_row_set_sha256=partition_row_set_sha256,
        training_config_sha256=training_config_sha256,
        epoch_candidates=epoch_candidates,
        sign_gate=sign_gate,
    )
    return validate_checkpoint_selection_artifact(payload)


def validate_checkpoint_selection_artifact(
    payload: Mapping,
    *,
    expected_run_start_sha256: str | None = None,
    expected_activation_manifest_sha256: str | None = None,
    expected_authority_hashes: Mapping | None = None,
    expected_estimand_binding_sha256: str | None = None,
    expected_method_id: str | None = None,
    expected_training_variant_id: str | None = None,
    expected_seed: int | None = None,
    expected_partition_row_set_sha256: Mapping | None = None,
    expected_training_config_sha256: str | None = None,
) -> dict:
    """Validate all fields and independently recompute the selected epoch."""

    _assert_no_forbidden_disclosure_keys(payload)
    _strict_keys(payload, ARTIFACT_KEYS, label="checkpoint selection artifact")
    if payload.get("schema_version") != SCHEMA:
        raise SelectionArtifactProtocolError("unsupported selection artifact schema")
    if payload.get("artifact_role") != ARTIFACT_ROLE:
        raise SelectionArtifactProtocolError("selection artifact role is invalid")
    records = payload.get("epoch_selection_records")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence) or not records:
        raise SelectionArtifactProtocolError(
            "epoch_selection_records must be a non-empty sequence"
        )
    candidates = []
    for index, record in enumerate(records):
        _strict_keys(record, EPOCH_RECORD_KEYS, label=f"epoch selection record {index}")
        metrics = record.get("development_metrics")
        _strict_keys(metrics, METRIC_KEYS, label=f"development metrics {index}")
        candidates.append({"epoch": record.get("epoch"), **dict(metrics)})

    rebuilt = _build_payload(
        run_start_sha256=payload.get("run_start_sha256"),
        activation_manifest_sha256=payload.get("activation_manifest_sha256"),
        authority_hashes=payload.get("authority_hashes"),
        estimand_binding_sha256=payload.get("estimand_binding_sha256"),
        method_id=payload.get("method_id"),
        training_variant_id=payload.get("training_variant_id"),
        seed=payload.get("seed"),
        partition_row_set_sha256=payload.get("partition_row_set_sha256"),
        training_config_sha256=payload.get("training_config_sha256"),
        epoch_candidates=candidates,
        sign_gate=payload.get("sign_gate"),
    )
    if dict(payload) != rebuilt:
        raise SelectionArtifactProtocolError(
            "selection artifact does not exactly match recomputed development-only evidence"
        )

    expected_pairs = (
        ("run_start_sha256", expected_run_start_sha256),
        ("activation_manifest_sha256", expected_activation_manifest_sha256),
        ("estimand_binding_sha256", expected_estimand_binding_sha256),
        ("method_id", expected_method_id),
        ("training_variant_id", expected_training_variant_id),
        ("seed", expected_seed),
        ("training_config_sha256", expected_training_config_sha256),
    )
    for field, expected in expected_pairs:
        if expected is not None and rebuilt[field] != expected:
            raise SelectionArtifactProtocolError(
                f"selection artifact {field} binding mismatch"
            )
    if expected_authority_hashes is not None:
        expected = _authority_hashes(expected_authority_hashes)
        if rebuilt["authority_hashes"] != expected:
            raise SelectionArtifactProtocolError(
                "selection artifact authority_hashes binding mismatch"
            )
    if expected_partition_row_set_sha256 is not None:
        expected = _partition_hashes(expected_partition_row_set_sha256)
        if rebuilt["partition_row_set_sha256"] != expected:
            raise SelectionArtifactProtocolError(
                "selection artifact partition row-set binding mismatch"
            )
    return rebuilt


def canonical_checkpoint_selection_bytes(payload: Mapping) -> bytes:
    """Return canonical finite JSON bytes only after full recomputation."""

    try:
        return canonical_json_bytes(validate_checkpoint_selection_artifact(payload))
    except ArtifactIOError as exc:
        raise SelectionArtifactProtocolError(str(exc)) from exc


def write_checkpoint_selection_artifact_exclusive(
    path: Path, payload: Mapping
) -> str:
    """Atomically publish a validated artifact without overwriting a name."""

    try:
        return atomic_write_json_exclusive(
            Path(path), payload, validator=validate_checkpoint_selection_artifact
        )
    except ArtifactIOError as exc:
        raise SelectionArtifactProtocolError(str(exc)) from exc


def _unique_json_object(pairs) -> dict:
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def load_checkpoint_selection_artifact_exact(
    path: Path,
    expected_sha256: str,
    **expected_bindings,
) -> dict:
    """Read exact regular-file bytes, recompute selection, and bind callers."""

    _digest(expected_sha256, label="expected_sha256")
    try:
        raw = read_exact_regular_file(Path(path), expected_sha256)
    except ArtifactIOError as exc:
        raise SelectionArtifactProtocolError(str(exc)) from exc
    try:
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_json_object
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SelectionArtifactProtocolError(
            f"selection artifact is not strict JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SelectionArtifactProtocolError("selection artifact must be a JSON object")
    validated = validate_checkpoint_selection_artifact(
        payload, **expected_bindings
    )
    if canonical_checkpoint_selection_bytes(validated) != raw:
        raise SelectionArtifactProtocolError(
            "selection artifact bytes are not canonical finite JSON"
        )
    return validated
