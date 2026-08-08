"""Strict global prediction commits assembled from per-estimand ensembles.

This boundary never consumes model objects, labels, sequences, or protected
partitions.  It replays each fixed five-seed ensemble from its committed bytes,
checks it against one frozen :class:`PrimaryEstimandSlice`, and only then
publishes a four-field global prediction JSONL followed by its commit manifest.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Mapping, Sequence

from .artifact_graph import (
    ArtifactGraphError,
    INTERNAL_TEST_PARTITION,
    PRIMARY_INFORMATION_SET,
    make_artifact_ref,
    parse_prediction_jsonl,
    prediction_jsonl_bytes,
    read_artifact_ref,
    replay_ensemble_manifest,
    validate_artifact_ref,
    write_prediction_jsonl_exclusive,
)
from .artifact_io import ArtifactIOError, atomic_write_json_exclusive
from .checkpoint import ALLOWED_TRAINING_VARIANTS
from .estimands import PrimaryEstimandSlice, stable_row_id
from .protocol import canonical_sha256
from .report import ReportProtocolError


GLOBAL_PREDICTION_COMMIT_SCHEMA = "r3x_global_method_prediction_commit_v1"
GLOBAL_TEST_BINDING_SCHEMA = "r3x_global_internal_test_binding_v1"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_METHOD_ID = re.compile(r"^[A-Z][A-Z0-9_]*$")
_ROW_PARTITION = "internal_test"

_SOURCE_KEYS = {
    "estimand_id",
    "estimand_binding_sha256",
    "registry_sha256",
    "registry_entry_set_sha256",
    "partition_binding_sha256",
    "internal_test_ordered_row_ids_sha256",
    "internal_test_row_set_sha256",
    "ensemble_manifest_ref",
    "ensemble_prediction_jsonl_ref",
    "n_rows",
}
_COMMIT_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "run_start_sha256",
    "method_id",
    "training_variant_id",
    "partition",
    "information_set",
    "information_set_fingerprint",
    "estimand_registry_sha256",
    "primary_estimand_ids",
    "source_ensembles",
    "global_test_binding_sha256",
    "prediction_jsonl_ref",
    "n_estimands",
    "n_rows",
    "ordered_row_ids_sha256",
    "row_set_sha256",
    "ordinary_only",
    "sealed_access",
    "downstream_unlock",
}


class GlobalPredictionError(RuntimeError):
    """Fail-closed global merge, schema, or exact-replay error."""


def _strict_keys(value: Mapping, expected: set[str], *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise GlobalPredictionError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise GlobalPredictionError(
            f"{label} keys differ from strict schema: {sorted(actual ^ expected)}"
        )


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise GlobalPredictionError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _positive_count(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise GlobalPredictionError(f"{label} must be a positive integer")
    return value


def _stable_row_id(row: Mapping) -> str:
    try:
        return stable_row_id(row)
    except ReportProtocolError as exc:
        raise GlobalPredictionError(str(exc)) from exc


def _row_id_hashes(row_ids: Sequence[str]) -> tuple[str, str]:
    if not row_ids or len(row_ids) != len(set(row_ids)):
        raise GlobalPredictionError("row identities must be non-empty and unique")
    for row_id in row_ids:
        _digest(row_id, label="row identity")
    ordered = json.dumps(
        list(row_ids), separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    unordered = json.dumps(
        sorted(row_ids), separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(ordered).hexdigest(), hashlib.sha256(unordered).hexdigest()


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json_object(raw: bytes, *, label: str) -> dict:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_json_object)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise GlobalPredictionError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise GlobalPredictionError(f"{label} must be a JSON object")
    return value


def _validate_source(source: Mapping) -> dict:
    _strict_keys(source, _SOURCE_KEYS, label="global prediction source")
    estimand_id = source.get("estimand_id")
    if (
        not isinstance(estimand_id, str)
        or not estimand_id
        or estimand_id != estimand_id.strip()
    ):
        raise GlobalPredictionError("source estimand_id must be a non-empty string")
    for field in (
        "estimand_binding_sha256",
        "registry_sha256",
        "registry_entry_set_sha256",
        "partition_binding_sha256",
        "internal_test_ordered_row_ids_sha256",
        "internal_test_row_set_sha256",
    ):
        _digest(source.get(field), label=field)
    if source["estimand_binding_sha256"] != source["partition_binding_sha256"]:
        raise GlobalPredictionError(
            "source ensemble estimand binding must equal the frozen partition binding"
        )
    try:
        validate_artifact_ref(source.get("ensemble_manifest_ref"))
        validate_artifact_ref(source.get("ensemble_prediction_jsonl_ref"))
    except ArtifactGraphError as exc:
        raise GlobalPredictionError(str(exc)) from exc
    _positive_count(source.get("n_rows"), label="source n_rows")
    return dict(source)


def validate_global_prediction_commit(payload: Mapping) -> dict:
    """Validate only the strict manifest schema; exact scientific replay is separate."""

    _strict_keys(payload, _COMMIT_KEYS, label="global prediction commit")
    if payload.get("schema_version") != GLOBAL_PREDICTION_COMMIT_SCHEMA:
        raise GlobalPredictionError("unsupported global prediction commit schema")
    if payload.get("artifact_role") != "ORDINARY_GLOBAL_METHOD_PREDICTIONS":
        raise GlobalPredictionError("global prediction artifact role is invalid")
    if payload.get("status") != "COMMITTED":
        raise GlobalPredictionError("global prediction status must be COMMITTED")
    _digest(payload.get("run_start_sha256"), label="run_start_sha256")
    method_id = payload.get("method_id")
    if not isinstance(method_id, str) or not _METHOD_ID.fullmatch(method_id):
        raise GlobalPredictionError("method_id is not a safe frozen identifier")
    if payload.get("training_variant_id") not in ALLOWED_TRAINING_VARIANTS:
        raise GlobalPredictionError("training_variant_id is not frozen")
    if payload.get("partition") != INTERNAL_TEST_PARTITION:
        raise GlobalPredictionError("global prediction partition is not ordinary internal test")
    if payload.get("information_set") != PRIMARY_INFORMATION_SET:
        raise GlobalPredictionError("global prediction information set is invalid")
    _digest(
        payload.get("information_set_fingerprint"),
        label="information_set_fingerprint",
    )
    _digest(payload.get("estimand_registry_sha256"), label="estimand_registry_sha256")
    _digest(payload.get("global_test_binding_sha256"), label="global_test_binding_sha256")
    ids = payload.get("primary_estimand_ids")
    if (
        not isinstance(ids, list)
        or not ids
        or any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in ids
        )
        or ids != sorted(ids)
        or len(ids) != len(set(ids))
    ):
        raise GlobalPredictionError("primary_estimand_ids must be sorted and unique")
    sources = payload.get("source_ensembles")
    if not isinstance(sources, list) or not sources:
        raise GlobalPredictionError("source_ensembles must be a non-empty list")
    normalized_sources = [_validate_source(source) for source in sources]
    if [source["estimand_id"] for source in normalized_sources] != ids:
        raise GlobalPredictionError(
            "source ensembles must contain every primary estimand exactly once in canonical order"
        )
    if _positive_count(payload.get("n_estimands"), label="n_estimands") != len(ids):
        raise GlobalPredictionError("n_estimands disagrees with source ensembles")
    try:
        validate_artifact_ref(payload.get("prediction_jsonl_ref"))
    except ArtifactGraphError as exc:
        raise GlobalPredictionError(str(exc)) from exc
    _positive_count(payload.get("n_rows"), label="global prediction n_rows")
    _digest(payload.get("ordered_row_ids_sha256"), label="ordered_row_ids_sha256")
    _digest(payload.get("row_set_sha256"), label="row_set_sha256")
    if payload.get("ordinary_only") is not True:
        raise GlobalPredictionError("global predictions must remain ordinary-only")
    if payload.get("sealed_access") is not False:
        raise GlobalPredictionError("global predictions cannot carry sealed access")
    if payload.get("downstream_unlock") is not False:
        raise GlobalPredictionError("global predictions cannot unlock downstream phases")
    return dict(payload)


def _validate_frozen_inputs(
    estimand_slices: Sequence[PrimaryEstimandSlice],
    global_test_rows: Sequence[Mapping],
) -> tuple[dict[str, PrimaryEstimandSlice], tuple[str, ...], str]:
    if (
        isinstance(estimand_slices, (str, bytes))
        or not isinstance(estimand_slices, Sequence)
        or not estimand_slices
    ):
        raise GlobalPredictionError("primary estimand slices must be non-empty")
    slice_index: dict[str, PrimaryEstimandSlice] = {}
    for item in estimand_slices:
        if not isinstance(item, PrimaryEstimandSlice):
            raise GlobalPredictionError("every primary estimand slice must be frozen")
        if item.estimand_id in slice_index:
            raise GlobalPredictionError("primary estimand IDs must be unique")
        slice_index[item.estimand_id] = item
    registry_hashes = {item.registry_sha256 for item in estimand_slices}
    if len(registry_hashes) != 1:
        raise GlobalPredictionError("primary estimands use different registry authorities")

    if (
        isinstance(global_test_rows, (str, bytes))
        or not isinstance(global_test_rows, Sequence)
        or not global_test_rows
    ):
        raise GlobalPredictionError("global internal-test rows must be non-empty")
    expected_owner = {
        _stable_row_id(row): item.estimand_id
        for item in estimand_slices
        for row in item.internal_test
    }
    if sum(len(item.internal_test) for item in estimand_slices) != len(expected_owner):
        raise GlobalPredictionError("primary estimand internal-test rows overlap")
    global_ids: list[str] = []
    for row in global_test_rows:
        if not isinstance(row, Mapping):
            raise GlobalPredictionError("global internal-test rows must be objects")
        if row.get("partition") != _ROW_PARTITION:
            raise GlobalPredictionError("global rows must belong to internal_test")
        row_id = _stable_row_id(row)
        owner = row.get("estimand_id")
        if owner != expected_owner.get(row_id):
            raise GlobalPredictionError(
                "global row estimand ownership differs from frozen primary slices"
            )
        global_ids.append(row_id)
    if set(global_ids) != set(expected_owner) or len(global_ids) != len(expected_owner):
        raise GlobalPredictionError(
            "global internal-test rows do not exactly cover the primary estimand slices"
        )
    if global_ids != sorted(global_ids):
        raise GlobalPredictionError(
            "global internal-test row order must be the frozen canonical row_id order"
        )
    _row_id_hashes(global_ids)
    return slice_index, tuple(global_ids), next(iter(registry_hashes))


def _global_test_binding(
    *,
    registry_sha256: str,
    slices: Mapping[str, PrimaryEstimandSlice],
    global_row_ids: Sequence[str],
) -> str:
    ordered_hash, set_hash = _row_id_hashes(global_row_ids)
    return canonical_sha256({
        "schema_version": GLOBAL_TEST_BINDING_SCHEMA,
        "estimand_registry_sha256": registry_sha256,
        "primary_estimands": [
            {
                "estimand_id": estimand_id,
                "registry_entry_set_sha256": slices[estimand_id].registry_entry_set_sha256,
                "partition_binding_sha256": slices[estimand_id].partition_binding_sha256,
                "internal_test_ordered_row_ids_sha256": _row_id_hashes(
                    tuple(
                        _stable_row_id(row)
                        for row in slices[estimand_id].internal_test
                    )
                )[0],
            }
            for estimand_id in sorted(slices)
        ],
        "global_internal_test_ordered_row_ids_sha256": ordered_hash,
        "global_internal_test_row_set_sha256": set_hash,
    })


def _replay_sources(
    *,
    run_root: Path,
    ensemble_manifest_refs_by_estimand: Mapping,
    estimand_slices: Sequence[PrimaryEstimandSlice],
    global_test_rows: Sequence[Mapping],
) -> dict:
    slice_index, global_ids, registry_sha = _validate_frozen_inputs(
        estimand_slices, global_test_rows
    )
    if not isinstance(ensemble_manifest_refs_by_estimand, Mapping):
        raise GlobalPredictionError("ensemble manifest references must be an object")
    if set(ensemble_manifest_refs_by_estimand) != set(slice_index):
        raise GlobalPredictionError(
            "global merge requires exactly one ensemble for every primary estimand"
        )

    source_entries = []
    predictions_by_id: dict[str, dict] = {}
    common: dict | None = None
    common_fields = (
        "run_start_sha256",
        "method_id",
        "training_variant_id",
        "partition",
        "information_set",
        "information_set_fingerprint",
    )
    for estimand_id in sorted(slice_index):
        item = slice_index[estimand_id]
        try:
            manifest_ref = validate_artifact_ref(
                ensemble_manifest_refs_by_estimand[estimand_id]
            )
            replay = replay_ensemble_manifest(run_root, manifest_ref)
        except ArtifactGraphError as exc:
            raise GlobalPredictionError(str(exc)) from exc
        manifest = replay["manifest"]
        rows = replay["rows"]
        expected_ids = tuple(_stable_row_id(row) for row in item.internal_test)
        actual_ids = tuple(row["row_id"] for row in rows)
        if manifest["estimand_binding_sha256"] != item.partition_binding_sha256:
            raise GlobalPredictionError(
                f"ensemble estimand binding differs for {estimand_id}"
            )
        if actual_ids != expected_ids:
            raise GlobalPredictionError(
                f"ensemble rows do not exactly match frozen estimand {estimand_id}"
            )
        ordered_hash, set_hash = _row_id_hashes(expected_ids)
        if (
            manifest["n_rows"] != len(expected_ids)
            or manifest["ordered_row_ids_sha256"] != ordered_hash
            or manifest["row_set_sha256"] != set_hash
        ):
            raise GlobalPredictionError(
                f"ensemble row binding differs for {estimand_id}"
            )
        context = {field: manifest[field] for field in common_fields}
        if common is None:
            common = context
        elif context != common:
            raise GlobalPredictionError(
                "per-estimand ensembles disagree on method, variant, run start, or information set"
            )
        for row in rows:
            if row["row_id"] in predictions_by_id:
                raise GlobalPredictionError("per-estimand ensembles contain overlapping rows")
            predictions_by_id[row["row_id"]] = dict(row)
        source_entries.append({
            "estimand_id": estimand_id,
            "estimand_binding_sha256": manifest["estimand_binding_sha256"],
            "registry_sha256": item.registry_sha256,
            "registry_entry_set_sha256": item.registry_entry_set_sha256,
            "partition_binding_sha256": item.partition_binding_sha256,
            "internal_test_ordered_row_ids_sha256": ordered_hash,
            "internal_test_row_set_sha256": set_hash,
            "ensemble_manifest_ref": manifest_ref,
            "ensemble_prediction_jsonl_ref": dict(manifest["prediction_jsonl_ref"]),
            "n_rows": len(expected_ids),
        })
    if common is None:  # guarded by non-empty slices; keeps the type invariant explicit.
        raise GlobalPredictionError("no per-estimand ensemble was replayed")
    if common["partition"] != INTERNAL_TEST_PARTITION:
        raise GlobalPredictionError("source ensembles are not ordinary internal-test predictions")
    if common["information_set"] != PRIMARY_INFORMATION_SET:
        raise GlobalPredictionError("source ensembles use a non-primary information set")
    if set(predictions_by_id) != set(global_ids):
        raise GlobalPredictionError("source ensembles do not exactly cover global test rows")
    rows = [predictions_by_id[row_id] for row_id in global_ids]
    # The artifact graph canonicalizer must preserve, rather than reorder, the
    # already-frozen global row sequence.
    try:
        if parse_prediction_jsonl(prediction_jsonl_bytes(rows)) != rows:
            raise GlobalPredictionError("global prediction order is not canonical")
    except ArtifactGraphError as exc:
        raise GlobalPredictionError(str(exc)) from exc
    return {
        "rows": rows,
        "sources": source_entries,
        "common": common,
        "registry_sha256": registry_sha,
        "global_row_ids": global_ids,
        "global_test_binding_sha256": _global_test_binding(
            registry_sha256=registry_sha,
            slices=slice_index,
            global_row_ids=global_ids,
        ),
    }


def _build_commit(*, replay: Mapping, prediction_jsonl_ref: Mapping) -> dict:
    rows = replay["rows"]
    row_ids = tuple(row["row_id"] for row in rows)
    ordered_hash, set_hash = _row_id_hashes(row_ids)
    common = replay["common"]
    payload = {
        "schema_version": GLOBAL_PREDICTION_COMMIT_SCHEMA,
        "artifact_role": "ORDINARY_GLOBAL_METHOD_PREDICTIONS",
        "status": "COMMITTED",
        "run_start_sha256": common["run_start_sha256"],
        "method_id": common["method_id"],
        "training_variant_id": common["training_variant_id"],
        "partition": common["partition"],
        "information_set": common["information_set"],
        "information_set_fingerprint": common["information_set_fingerprint"],
        "estimand_registry_sha256": replay["registry_sha256"],
        "primary_estimand_ids": [source["estimand_id"] for source in replay["sources"]],
        "source_ensembles": [dict(source) for source in replay["sources"]],
        "global_test_binding_sha256": replay["global_test_binding_sha256"],
        "prediction_jsonl_ref": dict(prediction_jsonl_ref),
        "n_estimands": len(replay["sources"]),
        "n_rows": len(rows),
        "ordered_row_ids_sha256": ordered_hash,
        "row_set_sha256": set_hash,
        "ordinary_only": True,
        "sealed_access": False,
        "downstream_unlock": False,
    }
    return validate_global_prediction_commit(payload)


def write_global_method_predictions_exclusive(
    *,
    run_root: Path,
    ensemble_manifest_refs_by_estimand: Mapping,
    estimand_slices: Sequence[PrimaryEstimandSlice],
    global_test_rows: Sequence[Mapping],
    prediction_path: Path,
    commit_path: Path,
) -> dict:
    """Publish exact global prediction bytes and then their immutable commit."""

    replay = _replay_sources(
        run_root=run_root,
        ensemble_manifest_refs_by_estimand=ensemble_manifest_refs_by_estimand,
        estimand_slices=estimand_slices,
        global_test_rows=global_test_rows,
    )
    try:
        prediction_sha = write_prediction_jsonl_exclusive(
            prediction_path, replay["rows"]
        )
        prediction_ref = make_artifact_ref(run_root, prediction_path, prediction_sha)
        commit = _build_commit(
            replay=replay, prediction_jsonl_ref=prediction_ref
        )
        commit_sha = atomic_write_json_exclusive(
            commit_path, commit, validator=validate_global_prediction_commit
        )
        commit_ref = make_artifact_ref(run_root, commit_path, commit_sha)
    except (ArtifactGraphError, ArtifactIOError) as exc:
        raise GlobalPredictionError(str(exc)) from exc
    verified = replay_global_prediction_commit(
        run_root,
        commit_ref,
        estimand_slices=estimand_slices,
        global_test_rows=global_test_rows,
    )
    if verified["commit"] != commit:
        raise GlobalPredictionError("published global commit changed during replay")
    return {
        "commit": commit,
        "commit_sha256": commit_sha,
        "commit_ref": commit_ref,
        "prediction_sha256": prediction_sha,
        "prediction_ref": prediction_ref,
    }


def replay_global_prediction_commit(
    run_root: Path,
    commit_ref: Mapping,
    *,
    estimand_slices: Sequence[PrimaryEstimandSlice],
    global_test_rows: Sequence[Mapping],
) -> dict:
    """Replay every source ensemble and verify exact global prediction bytes."""

    try:
        commit = validate_global_prediction_commit(
            _parse_json_object(
                read_artifact_ref(run_root, commit_ref),
                label="global prediction commit",
            )
        )
    except ArtifactGraphError as exc:
        raise GlobalPredictionError(str(exc)) from exc
    refs = {
        source["estimand_id"]: source["ensemble_manifest_ref"]
        for source in commit["source_ensembles"]
    }
    replay = _replay_sources(
        run_root=run_root,
        ensemble_manifest_refs_by_estimand=refs,
        estimand_slices=estimand_slices,
        global_test_rows=global_test_rows,
    )
    try:
        raw = read_artifact_ref(run_root, commit["prediction_jsonl_ref"])
        rows = parse_prediction_jsonl(raw)
    except ArtifactGraphError as exc:
        raise GlobalPredictionError(str(exc)) from exc
    expected_raw = prediction_jsonl_bytes(replay["rows"])
    if raw != expected_raw or rows != replay["rows"]:
        raise GlobalPredictionError(
            "global prediction bytes differ from exact per-estimand ensemble replay"
        )
    rebuilt = _build_commit(
        replay=replay,
        prediction_jsonl_ref=commit["prediction_jsonl_ref"],
    )
    if commit != rebuilt:
        raise GlobalPredictionError(
            "global prediction commit differs from frozen-input exact replay"
        )
    return {"commit": commit, "rows": rows, "sources": replay["sources"]}
