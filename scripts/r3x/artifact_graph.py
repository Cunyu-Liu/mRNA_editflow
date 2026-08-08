"""Strict, hash-linked R3X prediction and ensemble artifacts.

This module deliberately contains no trainer or label loader.  It publishes
only row identity plus predictions, commits one estimand/method/variant/seed
unit after every referenced byte stream replays, and constructs the fixed
five-model ensemble exclusively from those committed bytes.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from numbers import Real
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .artifact_io import (
    ArtifactIOError,
    atomic_publish_bytes,
    atomic_write_json_exclusive,
    read_exact_regular_file,
)
from .checkpoint import ALLOWED_TRAINING_VARIANTS, FROZEN_SEEDS
from .protocol import ProtocolHoldError, assert_ordinary_path


PREDICTION_ROW_SCHEMA = "r3x_prediction_row_v1"
PREDICTION_MANIFEST_SCHEMA = "r3x_prediction_manifest_v1"
SEED_COMMIT_SCHEMA = "r3x_seed_commit_v1"
ENSEMBLE_MANIFEST_SCHEMA = "r3x_ensemble_manifest_v1"

FROZEN_SEED_SEQUENCE = (11, 23, 42, 67, 101)
INTERNAL_TEST_PARTITION = "ORDINARY_INTERNAL_TEST"
PRIMARY_INFORMATION_SET = "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR"
TOTAL_VARIANCE_AGGREGATION = "FIXED_FIVE_MODEL_TOTAL_VARIANCE_ENSEMBLE"
TOTAL_VARIANCE_FORMULA = (
    "MEAN_EXP_LOGVAR_PLUS_MEAN_SQUARED_MINUS_ENSEMBLE_MEAN_SQUARED"
)
VARIANCE_FLOOR = 1e-12

HEX64 = re.compile(r"^[0-9a-f]{64}$")
METHOD_ID = re.compile(r"^[A-Z][A-Z0-9_]*$")

PREDICTION_ROW_KEYS = {
    "schema_version",
    "row_id",
    "predicted_delta_mean",
    "predicted_delta_logvar",
}
ARTIFACT_REF_KEYS = {"relative_path", "sha256"}
PREDICTION_MANIFEST_KEYS = {
    "schema_version",
    "artifact_role",
    "run_start_sha256",
    "estimand_binding_sha256",
    "method_id",
    "training_variant_id",
    "seed",
    "partition",
    "checkpoint_sha256",
    "prediction_jsonl_ref",
    "n_rows",
    "ordered_row_ids_sha256",
    "row_set_sha256",
    "information_set",
    "information_set_fingerprint",
    "variance_semantics",
    "ordinary_only",
    "sealed_access",
    "downstream_unlock",
}
SEED_COMMIT_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "run_start_sha256",
    "estimand_binding_sha256",
    "method_id",
    "training_variant_id",
    "seed",
    "partition",
    "checkpoint_selection_ref",
    "checkpoint_ref",
    "prediction_manifest_ref",
    "prediction_jsonl_ref",
    "n_rows",
    "ordered_row_ids_sha256",
    "row_set_sha256",
    "information_set",
    "information_set_fingerprint",
    "ordinary_only",
    "sealed_access",
    "downstream_unlock",
}
ENSEMBLE_MANIFEST_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "run_start_sha256",
    "estimand_binding_sha256",
    "method_id",
    "training_variant_id",
    "partition",
    "seeds",
    "seed_commit_refs",
    "seed_prediction_jsonl_sha256",
    "aggregation",
    "variance_formula",
    "variance_floor",
    "n_models",
    "prediction_jsonl_ref",
    "n_rows",
    "ordered_row_ids_sha256",
    "row_set_sha256",
    "information_set",
    "information_set_fingerprint",
    "ordinary_only",
    "sealed_access",
    "downstream_unlock",
}


class ArtifactGraphError(RuntimeError):
    """Fail-closed artifact schema, reference, or replay error."""


def _strict_keys(value: Mapping, expected: set[str], *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ArtifactGraphError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ArtifactGraphError(
            f"{label} keys differ from strict schema: {sorted(actual ^ expected)}"
        )


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ArtifactGraphError(f"{label} must be 64 lowercase hex")
    return value


def _method_id(value: object) -> str:
    if not isinstance(value, str) or not METHOD_ID.fullmatch(value):
        raise ArtifactGraphError("method_id is not a safe frozen identifier")
    return value


def _variant(value: object) -> str:
    if value not in ALLOWED_TRAINING_VARIANTS:
        raise ArtifactGraphError("training_variant_id is not frozen")
    return str(value)


def _seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in FROZEN_SEEDS:
        raise ArtifactGraphError("seed must be one of the five frozen seeds")
    return int(value)


def _positive_count(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ArtifactGraphError(f"{label} must be a positive integer")
    return int(value)


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ArtifactGraphError(f"{label} must be a finite JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ArtifactGraphError(f"{label} must be a finite number")
    return result


def _ordinary_boundaries(payload: Mapping, *, label: str) -> None:
    if payload.get("ordinary_only") is not True:
        raise ArtifactGraphError(f"{label} must remain ordinary-only")
    if payload.get("sealed_access") is not False:
        raise ArtifactGraphError(f"{label} cannot carry sealed access")
    if payload.get("downstream_unlock") is not False:
        raise ArtifactGraphError(f"{label} cannot unlock downstream phases")


def _unique_json_object(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _parse_json_object(raw: bytes, *, label: str) -> dict:
    try:
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_json_object
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ArtifactGraphError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactGraphError(f"{label} must be a JSON object")
    return payload


def _row_id_hashes(rows: Sequence[Mapping]) -> tuple[str, str]:
    ids = [str(row["row_id"]) for row in rows]
    ordered_raw = json.dumps(ids, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    set_raw = json.dumps(
        sorted(ids), separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return (
        hashlib.sha256(ordered_raw).hexdigest(),
        hashlib.sha256(set_raw).hexdigest(),
    )


def _normalize_prediction_row(row: Mapping) -> dict:
    _strict_keys(row, PREDICTION_ROW_KEYS, label="prediction row")
    if row.get("schema_version") != PREDICTION_ROW_SCHEMA:
        raise ArtifactGraphError("prediction row schema is unsupported")
    row_id = _digest(row.get("row_id"), label="prediction row_id")
    raw_logvar = row.get("predicted_delta_logvar")
    return {
        "schema_version": PREDICTION_ROW_SCHEMA,
        "row_id": row_id,
        "predicted_delta_mean": _finite_float(
            row.get("predicted_delta_mean"), label="predicted_delta_mean"
        ),
        # Point baselines use explicit null; neural seed/ensemble manifests
        # separately require finite log-variance before uncertainty replay.
        "predicted_delta_logvar": (
            None if raw_logvar is None
            else _finite_float(raw_logvar, label="predicted_delta_logvar")
        ),
    }


def normalize_prediction_rows(rows: Sequence[Mapping]) -> list[dict]:
    """Validate the four-field disclosure boundary and return canonical order."""

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ArtifactGraphError("prediction artifact requires at least one row")
    normalized = [_normalize_prediction_row(row) for row in rows]
    row_ids = [row["row_id"] for row in normalized]
    if len(row_ids) != len(set(row_ids)):
        raise ArtifactGraphError("prediction row_id values must be unique")
    normalized.sort(key=lambda row: row["row_id"])
    return normalized


def prediction_jsonl_bytes(rows: Sequence[Mapping]) -> bytes:
    normalized = normalize_prediction_rows(rows)
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


def parse_prediction_jsonl(raw: bytes) -> list[dict]:
    if not isinstance(raw, bytes) or not raw or not raw.endswith(b"\n"):
        raise ArtifactGraphError("prediction JSONL must be non-empty and end with one LF")
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise ArtifactGraphError("prediction JSONL is not UTF-8") from exc
    lines = text.splitlines()
    if not lines or any(not line for line in lines):
        raise ArtifactGraphError("prediction JSONL cannot contain blank rows")
    parsed = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line, object_pairs_hook=_unique_json_object)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ArtifactGraphError(
                f"prediction JSONL line {line_number} is invalid: {exc}"
            ) from exc
        if not isinstance(row, Mapping):
            raise ArtifactGraphError(
                f"prediction JSONL line {line_number} must be an object"
            )
        parsed.append(_normalize_prediction_row(row))
    normalized = normalize_prediction_rows(parsed)
    if parsed != normalized:
        raise ArtifactGraphError("prediction JSONL rows are not in canonical row_id order")
    canonical = prediction_jsonl_bytes(normalized)
    if canonical != raw:
        raise ArtifactGraphError("prediction JSONL bytes are not canonical")
    return normalized


def write_prediction_jsonl_exclusive(path: Path, rows: Sequence[Mapping]) -> str:
    raw = prediction_jsonl_bytes(rows)
    try:
        return atomic_publish_bytes(path, raw, replay_validator=parse_prediction_jsonl)
    except ArtifactIOError as exc:
        raise ArtifactGraphError(str(exc)) from exc


def validate_artifact_ref(reference: Mapping) -> dict:
    _strict_keys(reference, ARTIFACT_REF_KEYS, label="artifact reference")
    relative = reference.get("relative_path")
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ArtifactGraphError("artifact relative_path is invalid")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != relative
    ):
        raise ArtifactGraphError("artifact reference must be a canonical safe relative path")
    try:
        assert_ordinary_path(Path(relative), purpose="R3X relative artifact reference")
    except ProtocolHoldError as exc:
        raise ArtifactGraphError(str(exc)) from exc
    return {
        "relative_path": relative,
        "sha256": _digest(reference.get("sha256"), label="artifact reference SHA-256"),
    }


def _checked_run_root(run_root: Path) -> Path:
    root = Path(run_root)
    try:
        assert_ordinary_path(root, purpose="R3X artifact graph root")
    except ProtocolHoldError as exc:
        raise ArtifactGraphError(str(exc)) from exc
    if not root.is_dir() or root.is_symlink():
        raise ArtifactGraphError("artifact graph root must be an existing real directory")
    return root.resolve(strict=True)


def resolve_artifact_ref(run_root: Path, reference: Mapping) -> Path:
    root = _checked_run_root(run_root)
    normalized = validate_artifact_ref(reference)
    candidate = root.joinpath(*PurePosixPath(normalized["relative_path"]).parts)
    try:
        assert_ordinary_path(candidate, purpose="R3X artifact reference")
    except ProtocolHoldError as exc:
        raise ArtifactGraphError(str(exc)) from exc
    current = root
    for part in PurePosixPath(normalized["relative_path"]).parts:
        current = current / part
        if current.is_symlink():
            raise ArtifactGraphError("artifact reference cannot traverse a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ArtifactGraphError(
            f"artifact reference does not exist: {normalized['relative_path']}"
        ) from exc
    if not resolved.is_relative_to(root):
        raise ArtifactGraphError("artifact reference escapes the run root")
    return candidate


def read_artifact_ref(run_root: Path, reference: Mapping) -> bytes:
    normalized = validate_artifact_ref(reference)
    path = resolve_artifact_ref(run_root, normalized)
    try:
        return read_exact_regular_file(path, normalized["sha256"])
    except ArtifactIOError as exc:
        raise ArtifactGraphError(str(exc)) from exc


def make_artifact_ref(run_root: Path, path: Path, sha256: str) -> dict:
    root = _checked_run_root(run_root)
    candidate = Path(path)
    try:
        relative = candidate.resolve(strict=True).relative_to(root).as_posix()
    except (FileNotFoundError, ValueError) as exc:
        raise ArtifactGraphError("artifact path is not an existing child of run root") from exc
    reference = validate_artifact_ref(
        {"relative_path": relative, "sha256": _digest(sha256, label="artifact SHA-256")}
    )
    read_artifact_ref(root, reference)
    return reference


def validate_prediction_manifest(payload: Mapping) -> dict:
    _strict_keys(payload, PREDICTION_MANIFEST_KEYS, label="prediction manifest")
    if payload.get("schema_version") != PREDICTION_MANIFEST_SCHEMA:
        raise ArtifactGraphError("unsupported prediction manifest schema")
    if payload.get("artifact_role") != "ORDINARY_ESTIMAND_METHOD_VARIANT_SEED_PREDICTIONS":
        raise ArtifactGraphError("prediction manifest artifact role is invalid")
    _digest(payload.get("run_start_sha256"), label="run_start_sha256")
    _digest(payload.get("estimand_binding_sha256"), label="estimand_binding_sha256")
    _method_id(payload.get("method_id"))
    _variant(payload.get("training_variant_id"))
    _seed(payload.get("seed"))
    if payload.get("partition") != INTERNAL_TEST_PARTITION:
        raise ArtifactGraphError("prediction partition must be ordinary internal test")
    _digest(payload.get("checkpoint_sha256"), label="checkpoint_sha256")
    validate_artifact_ref(payload.get("prediction_jsonl_ref"))
    _positive_count(payload.get("n_rows"), label="prediction n_rows")
    _digest(payload.get("ordered_row_ids_sha256"), label="ordered_row_ids_sha256")
    _digest(payload.get("row_set_sha256"), label="row_set_sha256")
    if payload.get("information_set") != PRIMARY_INFORMATION_SET:
        raise ArtifactGraphError("prediction information set is invalid")
    _digest(
        payload.get("information_set_fingerprint"),
        label="information_set_fingerprint",
    )
    if payload.get("variance_semantics") != "PER_MODEL_ALEATORIC_LOG_VARIANCE":
        raise ArtifactGraphError("seed prediction variance semantics are invalid")
    _ordinary_boundaries(payload, label="prediction manifest")
    return dict(payload)


def build_prediction_manifest(
    *,
    run_start_sha256: str,
    estimand_binding_sha256: str,
    method_id: str,
    training_variant_id: str,
    seed: int,
    checkpoint_sha256: str,
    prediction_jsonl_ref: Mapping,
    prediction_rows: Sequence[Mapping],
    information_set_fingerprint: str,
) -> dict:
    rows = normalize_prediction_rows(prediction_rows)
    if any(row["predicted_delta_logvar"] is None for row in rows):
        raise ArtifactGraphError(
            "seed prediction manifests require finite aleatoric log-variance"
        )
    ordered_hash, set_hash = _row_id_hashes(rows)
    payload = {
        "schema_version": PREDICTION_MANIFEST_SCHEMA,
        "artifact_role": "ORDINARY_ESTIMAND_METHOD_VARIANT_SEED_PREDICTIONS",
        "run_start_sha256": run_start_sha256,
        "estimand_binding_sha256": estimand_binding_sha256,
        "method_id": method_id,
        "training_variant_id": training_variant_id,
        "seed": seed,
        "partition": INTERNAL_TEST_PARTITION,
        "checkpoint_sha256": checkpoint_sha256,
        "prediction_jsonl_ref": dict(prediction_jsonl_ref),
        "n_rows": len(rows),
        "ordered_row_ids_sha256": ordered_hash,
        "row_set_sha256": set_hash,
        "information_set": PRIMARY_INFORMATION_SET,
        "information_set_fingerprint": information_set_fingerprint,
        "variance_semantics": "PER_MODEL_ALEATORIC_LOG_VARIANCE",
        "ordinary_only": True,
        "sealed_access": False,
        "downstream_unlock": False,
    }
    return validate_prediction_manifest(payload)


def write_prediction_manifest_exclusive(path: Path, payload: Mapping) -> str:
    try:
        return atomic_write_json_exclusive(
            path, payload, validator=validate_prediction_manifest
        )
    except ArtifactIOError as exc:
        raise ArtifactGraphError(str(exc)) from exc


def validate_seed_commit(payload: Mapping) -> dict:
    _strict_keys(payload, SEED_COMMIT_KEYS, label="seed commit")
    if payload.get("schema_version") != SEED_COMMIT_SCHEMA:
        raise ArtifactGraphError("unsupported seed commit schema")
    if payload.get("artifact_role") != "ORDINARY_ESTIMAND_METHOD_VARIANT_SEED_COMMIT":
        raise ArtifactGraphError("seed commit artifact role is invalid")
    if payload.get("status") != "COMMITTED":
        raise ArtifactGraphError("seed commit status must be COMMITTED")
    _digest(payload.get("run_start_sha256"), label="run_start_sha256")
    _digest(payload.get("estimand_binding_sha256"), label="estimand_binding_sha256")
    _method_id(payload.get("method_id"))
    _variant(payload.get("training_variant_id"))
    _seed(payload.get("seed"))
    if payload.get("partition") != INTERNAL_TEST_PARTITION:
        raise ArtifactGraphError("seed commit partition is invalid")
    for field in (
        "checkpoint_selection_ref",
        "checkpoint_ref",
        "prediction_manifest_ref",
        "prediction_jsonl_ref",
    ):
        validate_artifact_ref(payload.get(field))
    _positive_count(payload.get("n_rows"), label="seed commit n_rows")
    _digest(payload.get("ordered_row_ids_sha256"), label="ordered_row_ids_sha256")
    _digest(payload.get("row_set_sha256"), label="row_set_sha256")
    if payload.get("information_set") != PRIMARY_INFORMATION_SET:
        raise ArtifactGraphError("seed commit information set is invalid")
    _digest(
        payload.get("information_set_fingerprint"),
        label="information_set_fingerprint",
    )
    _ordinary_boundaries(payload, label="seed commit")
    return dict(payload)


def build_seed_commit(
    *,
    checkpoint_selection_ref: Mapping,
    checkpoint_ref: Mapping,
    prediction_manifest_ref: Mapping,
    prediction_jsonl_ref: Mapping,
    prediction_manifest: Mapping,
) -> dict:
    manifest = validate_prediction_manifest(prediction_manifest)
    selection = validate_artifact_ref(checkpoint_selection_ref)
    checkpoint = validate_artifact_ref(checkpoint_ref)
    manifest_ref = validate_artifact_ref(prediction_manifest_ref)
    prediction_ref = validate_artifact_ref(prediction_jsonl_ref)
    if checkpoint["sha256"] != manifest["checkpoint_sha256"]:
        raise ArtifactGraphError("prediction manifest/checkpoint SHA-256 mismatch")
    if prediction_ref != validate_artifact_ref(manifest["prediction_jsonl_ref"]):
        raise ArtifactGraphError("prediction manifest/JSONL reference mismatch")
    payload = {
        "schema_version": SEED_COMMIT_SCHEMA,
        "artifact_role": "ORDINARY_ESTIMAND_METHOD_VARIANT_SEED_COMMIT",
        "status": "COMMITTED",
        "run_start_sha256": manifest["run_start_sha256"],
        "estimand_binding_sha256": manifest["estimand_binding_sha256"],
        "method_id": manifest["method_id"],
        "training_variant_id": manifest["training_variant_id"],
        "seed": manifest["seed"],
        "partition": manifest["partition"],
        "checkpoint_selection_ref": selection,
        "checkpoint_ref": checkpoint,
        "prediction_manifest_ref": manifest_ref,
        "prediction_jsonl_ref": prediction_ref,
        "n_rows": manifest["n_rows"],
        "ordered_row_ids_sha256": manifest["ordered_row_ids_sha256"],
        "row_set_sha256": manifest["row_set_sha256"],
        "information_set": manifest["information_set"],
        "information_set_fingerprint": manifest["information_set_fingerprint"],
        "ordinary_only": True,
        "sealed_access": False,
        "downstream_unlock": False,
    }
    return validate_seed_commit(payload)


def write_seed_commit_exclusive(path: Path, payload: Mapping) -> str:
    try:
        return atomic_write_json_exclusive(path, payload, validator=validate_seed_commit)
    except ArtifactIOError as exc:
        raise ArtifactGraphError(str(exc)) from exc


def _load_json_ref(run_root: Path, reference: Mapping, *, label: str) -> dict:
    raw = read_artifact_ref(run_root, reference)
    return _parse_json_object(raw, label=label)


def replay_seed_commit(
    run_root: Path,
    seed_commit_ref: Mapping,
    *,
    expected_seed: int | None = None,
) -> dict:
    """Replay a seed commit and every byte stream to which it points."""

    commit = validate_seed_commit(
        _load_json_ref(run_root, seed_commit_ref, label="seed commit")
    )
    if expected_seed is not None and commit["seed"] != _seed(expected_seed):
        raise ArtifactGraphError("seed commit is referenced under the wrong seed")
    # Checkpoint schema validation belongs to checkpoint.py; this graph layer
    # still proves the exact committed bytes exist under the bound digest.
    read_artifact_ref(run_root, commit["checkpoint_selection_ref"])
    read_artifact_ref(run_root, commit["checkpoint_ref"])
    manifest = validate_prediction_manifest(
        _load_json_ref(
            run_root,
            commit["prediction_manifest_ref"],
            label="prediction manifest",
        )
    )
    comparison_fields = (
        "run_start_sha256",
        "estimand_binding_sha256",
        "method_id",
        "training_variant_id",
        "seed",
        "partition",
        "n_rows",
        "ordered_row_ids_sha256",
        "row_set_sha256",
        "information_set",
        "information_set_fingerprint",
    )
    if any(commit[field] != manifest[field] for field in comparison_fields):
        raise ArtifactGraphError("seed commit/prediction manifest binding mismatch")
    if commit["checkpoint_ref"]["sha256"] != manifest["checkpoint_sha256"]:
        raise ArtifactGraphError("seed commit/checkpoint binding mismatch")
    if commit["prediction_jsonl_ref"] != manifest["prediction_jsonl_ref"]:
        raise ArtifactGraphError("seed commit/prediction JSONL binding mismatch")
    rows = parse_prediction_jsonl(
        read_artifact_ref(run_root, commit["prediction_jsonl_ref"])
    )
    ordered_hash, set_hash = _row_id_hashes(rows)
    if (
        len(rows) != commit["n_rows"]
        or ordered_hash != commit["ordered_row_ids_sha256"]
        or set_hash != commit["row_set_sha256"]
    ):
        raise ArtifactGraphError("committed prediction rows do not replay the manifest")
    return {"commit": commit, "manifest": manifest, "rows": rows}


def _normalize_seed_refs(seed_commit_refs: Mapping) -> dict[str, dict]:
    if not isinstance(seed_commit_refs, Mapping):
        raise ArtifactGraphError("seed_commit_refs must be an object")
    normalized: dict[str, dict] = {}
    for raw_seed, reference in seed_commit_refs.items():
        if isinstance(raw_seed, bool):
            raise ArtifactGraphError("seed commit reference key is not canonical")
        if isinstance(raw_seed, int):
            seed = raw_seed
        elif isinstance(raw_seed, str) and raw_seed in {
            str(value) for value in FROZEN_SEED_SEQUENCE
        }:
            seed = int(raw_seed)
        else:
            raise ArtifactGraphError("seed commit reference key is not canonical")
        key = str(_seed(seed))
        if key in normalized:
            raise ArtifactGraphError("duplicate seed commit reference")
        normalized[key] = validate_artifact_ref(reference)
    expected = {str(seed) for seed in FROZEN_SEED_SEQUENCE}
    if set(normalized) != expected:
        raise ArtifactGraphError("ensemble requires exactly the five frozen seed commits")
    return {str(seed): normalized[str(seed)] for seed in FROZEN_SEED_SEQUENCE}


def replay_fixed_ensemble(
    run_root: Path, seed_commit_refs: Mapping
) -> tuple[list[dict], dict]:
    """Compute fixed-ensemble predictions only from exact committed seed bytes."""

    refs = _normalize_seed_refs(seed_commit_refs)
    replays = [
        replay_seed_commit(run_root, refs[str(seed)], expected_seed=seed)
        for seed in FROZEN_SEED_SEQUENCE
    ]
    first = replays[0]["commit"]
    common_fields = (
        "run_start_sha256",
        "estimand_binding_sha256",
        "method_id",
        "training_variant_id",
        "partition",
        "n_rows",
        "ordered_row_ids_sha256",
        "row_set_sha256",
        "information_set",
        "information_set_fingerprint",
    )
    for replay in replays[1:]:
        if any(replay["commit"][field] != first[field] for field in common_fields):
            raise ArtifactGraphError("seed commits do not describe one ensemble unit")
    expected_ids = [row["row_id"] for row in replays[0]["rows"]]
    if any(
        [row["row_id"] for row in replay["rows"]] != expected_ids
        for replay in replays[1:]
    ):
        raise ArtifactGraphError("seed prediction row order/identity differs")

    ensemble_rows = []
    for index, row_id in enumerate(expected_ids):
        means = [
            replay["rows"][index]["predicted_delta_mean"] for replay in replays
        ]
        logvars = [
            replay["rows"][index]["predicted_delta_logvar"] for replay in replays
        ]
        if any(logvar is None for logvar in logvars):
            raise ArtifactGraphError("fixed neural ensemble cannot consume point-only variance")
        mean = math.fsum(means) / len(means)
        try:
            second_moment = math.fsum(
                math.exp(logvar) + value * value
                for value, logvar in zip(means, logvars)
            ) / len(means)
        except OverflowError as exc:
            raise ArtifactGraphError("seed log-variance overflows total variance") from exc
        variance = max(second_moment - mean * mean, VARIANCE_FLOOR)
        logvar = math.log(variance)
        if not all(math.isfinite(value) for value in (mean, variance, logvar)):
            raise ArtifactGraphError("ensemble total variance is non-finite")
        ensemble_rows.append(
            {
                "schema_version": PREDICTION_ROW_SCHEMA,
                "row_id": row_id,
                "predicted_delta_mean": mean,
                "predicted_delta_logvar": logvar,
            }
        )
    context = {
        "run_start_sha256": first["run_start_sha256"],
        "estimand_binding_sha256": first["estimand_binding_sha256"],
        "method_id": first["method_id"],
        "training_variant_id": first["training_variant_id"],
        "partition": first["partition"],
        "n_rows": first["n_rows"],
        "ordered_row_ids_sha256": first["ordered_row_ids_sha256"],
        "row_set_sha256": first["row_set_sha256"],
        "information_set": first["information_set"],
        "information_set_fingerprint": first["information_set_fingerprint"],
        "seed_commit_refs": refs,
        "seed_prediction_jsonl_sha256": {
            str(seed): replays[index]["commit"]["prediction_jsonl_ref"]["sha256"]
            for index, seed in enumerate(FROZEN_SEED_SEQUENCE)
        },
    }
    return normalize_prediction_rows(ensemble_rows), context


def validate_ensemble_manifest(payload: Mapping) -> dict:
    _strict_keys(payload, ENSEMBLE_MANIFEST_KEYS, label="ensemble manifest")
    if payload.get("schema_version") != ENSEMBLE_MANIFEST_SCHEMA:
        raise ArtifactGraphError("unsupported ensemble manifest schema")
    if payload.get("artifact_role") != "FIXED_FIVE_MODEL_PREDICTION_ENSEMBLE":
        raise ArtifactGraphError("ensemble artifact role is invalid")
    if payload.get("status") != "COMMITTED":
        raise ArtifactGraphError("ensemble status must be COMMITTED")
    _digest(payload.get("run_start_sha256"), label="run_start_sha256")
    _digest(payload.get("estimand_binding_sha256"), label="estimand_binding_sha256")
    _method_id(payload.get("method_id"))
    _variant(payload.get("training_variant_id"))
    if payload.get("partition") != INTERNAL_TEST_PARTITION:
        raise ArtifactGraphError("ensemble partition is invalid")
    if payload.get("seeds") != list(FROZEN_SEED_SEQUENCE):
        raise ArtifactGraphError("ensemble seed sequence is not frozen")
    _normalize_seed_refs(payload.get("seed_commit_refs"))
    prediction_hashes = payload.get("seed_prediction_jsonl_sha256")
    if not isinstance(prediction_hashes, Mapping) or set(prediction_hashes) != {
        str(seed) for seed in FROZEN_SEED_SEQUENCE
    }:
        raise ArtifactGraphError("ensemble seed prediction hashes are incomplete")
    for seed in FROZEN_SEED_SEQUENCE:
        _digest(
            prediction_hashes[str(seed)],
            label=f"seed {seed} prediction JSONL SHA-256",
        )
    if payload.get("aggregation") != TOTAL_VARIANCE_AGGREGATION:
        raise ArtifactGraphError("ensemble aggregation is invalid")
    if payload.get("variance_formula") != TOTAL_VARIANCE_FORMULA:
        raise ArtifactGraphError("ensemble variance formula is invalid")
    if payload.get("variance_floor") != VARIANCE_FLOOR:
        raise ArtifactGraphError("ensemble variance floor is invalid")
    if payload.get("n_models") != len(FROZEN_SEED_SEQUENCE):
        raise ArtifactGraphError("ensemble must contain exactly five models")
    validate_artifact_ref(payload.get("prediction_jsonl_ref"))
    _positive_count(payload.get("n_rows"), label="ensemble n_rows")
    _digest(payload.get("ordered_row_ids_sha256"), label="ordered_row_ids_sha256")
    _digest(payload.get("row_set_sha256"), label="row_set_sha256")
    if payload.get("information_set") != PRIMARY_INFORMATION_SET:
        raise ArtifactGraphError("ensemble information set is invalid")
    _digest(
        payload.get("information_set_fingerprint"),
        label="information_set_fingerprint",
    )
    _ordinary_boundaries(payload, label="ensemble manifest")
    return dict(payload)


def build_ensemble_manifest(
    *, context: Mapping,
    prediction_jsonl_ref: Mapping,
    prediction_rows: Sequence[Mapping],
) -> dict:
    rows = normalize_prediction_rows(prediction_rows)
    ordered_hash, set_hash = _row_id_hashes(rows)
    for field in (
        "run_start_sha256",
        "estimand_binding_sha256",
        "method_id",
        "training_variant_id",
        "partition",
        "n_rows",
        "ordered_row_ids_sha256",
        "row_set_sha256",
        "information_set",
        "information_set_fingerprint",
        "seed_commit_refs",
        "seed_prediction_jsonl_sha256",
    ):
        if field not in context:
            raise ArtifactGraphError(f"ensemble replay context lacks {field}")
    if (
        len(rows) != context["n_rows"]
        or ordered_hash != context["ordered_row_ids_sha256"]
        or set_hash != context["row_set_sha256"]
    ):
        raise ArtifactGraphError("ensemble rows do not match committed seed row bindings")
    payload = {
        "schema_version": ENSEMBLE_MANIFEST_SCHEMA,
        "artifact_role": "FIXED_FIVE_MODEL_PREDICTION_ENSEMBLE",
        "status": "COMMITTED",
        "run_start_sha256": context["run_start_sha256"],
        "estimand_binding_sha256": context["estimand_binding_sha256"],
        "method_id": context["method_id"],
        "training_variant_id": context["training_variant_id"],
        "partition": context["partition"],
        "seeds": list(FROZEN_SEED_SEQUENCE),
        "seed_commit_refs": dict(context["seed_commit_refs"]),
        "seed_prediction_jsonl_sha256": dict(
            context["seed_prediction_jsonl_sha256"]
        ),
        "aggregation": TOTAL_VARIANCE_AGGREGATION,
        "variance_formula": TOTAL_VARIANCE_FORMULA,
        "variance_floor": VARIANCE_FLOOR,
        "n_models": len(FROZEN_SEED_SEQUENCE),
        "prediction_jsonl_ref": dict(prediction_jsonl_ref),
        "n_rows": len(rows),
        "ordered_row_ids_sha256": ordered_hash,
        "row_set_sha256": set_hash,
        "information_set": context["information_set"],
        "information_set_fingerprint": context["information_set_fingerprint"],
        "ordinary_only": True,
        "sealed_access": False,
        "downstream_unlock": False,
    }
    return validate_ensemble_manifest(payload)


def write_ensemble_from_seed_commits_exclusive(
    *,
    run_root: Path,
    seed_commit_refs: Mapping,
    prediction_path: Path,
    manifest_path: Path,
) -> dict:
    """Replay five committed seeds, then publish prediction bytes and commit last."""

    rows, context = replay_fixed_ensemble(run_root, seed_commit_refs)
    prediction_sha = write_prediction_jsonl_exclusive(prediction_path, rows)
    prediction_ref = make_artifact_ref(run_root, prediction_path, prediction_sha)
    manifest = build_ensemble_manifest(
        context=context,
        prediction_jsonl_ref=prediction_ref,
        prediction_rows=rows,
    )
    try:
        manifest_sha = atomic_write_json_exclusive(
            manifest_path, manifest, validator=validate_ensemble_manifest
        )
    except ArtifactIOError as exc:
        raise ArtifactGraphError(str(exc)) from exc
    return {
        "manifest": manifest,
        "manifest_sha256": manifest_sha,
        "manifest_ref": make_artifact_ref(run_root, manifest_path, manifest_sha),
        "prediction_sha256": prediction_sha,
        "prediction_ref": prediction_ref,
    }


def replay_ensemble_manifest(run_root: Path, manifest_ref: Mapping) -> dict:
    """Verify a committed ensemble against fresh replay of its five seed commits."""

    manifest = validate_ensemble_manifest(
        _load_json_ref(run_root, manifest_ref, label="ensemble manifest")
    )
    rows = parse_prediction_jsonl(
        read_artifact_ref(run_root, manifest["prediction_jsonl_ref"])
    )
    ordered_hash, set_hash = _row_id_hashes(rows)
    if (
        len(rows) != manifest["n_rows"]
        or ordered_hash != manifest["ordered_row_ids_sha256"]
        or set_hash != manifest["row_set_sha256"]
    ):
        raise ArtifactGraphError("ensemble prediction bytes do not replay manifest")
    recomputed, context = replay_fixed_ensemble(
        run_root, manifest["seed_commit_refs"]
    )
    expected_raw = prediction_jsonl_bytes(recomputed)
    actual_raw = prediction_jsonl_bytes(rows)
    if expected_raw != actual_raw:
        raise ArtifactGraphError("ensemble prediction bytes differ from exact seed replay")
    comparison_fields = (
        "run_start_sha256",
        "estimand_binding_sha256",
        "method_id",
        "training_variant_id",
        "partition",
        "n_rows",
        "ordered_row_ids_sha256",
        "row_set_sha256",
        "information_set",
        "information_set_fingerprint",
        "seed_commit_refs",
        "seed_prediction_jsonl_sha256",
    )
    if any(manifest[field] != context[field] for field in comparison_fields):
        raise ArtifactGraphError("ensemble manifest differs from exact seed replay context")
    return {"manifest": manifest, "rows": rows}
