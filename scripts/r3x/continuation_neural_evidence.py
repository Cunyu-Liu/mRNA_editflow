"""Exact neural-state receipt for ordinary-development A/B continuation.

The historical continuation layer can verify a generic checkpoint envelope and
five prediction files without proving that those files came from the strict
``neural_development_state`` fitted states.  This module closes that gap.  It
exact-loads every state blob referenced by predictor provenance, recomputes all
development predictions, rebuilds the frozen five-seed total-variance
ensemble, and emits a compact receipt containing the decision-relevant
bindings.

The receipt is intentionally not an authority issuer.  It is ordinary-
development evidence only, has no best-seed path, and cannot unlock training,
internal-test, sealed, execution, or publication stages.  A continuation
decision must exact-load this receipt and compare it with its independently
frozen selector, policy, and model-binding authorities.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

from .artifact_graph import (
    ArtifactGraphError,
    parse_prediction_jsonl,
    prediction_jsonl_bytes,
    read_artifact_ref,
    resolve_artifact_ref,
    validate_artifact_ref,
)
from .artifact_io import ArtifactIOError, atomic_write_json_exclusive, canonical_json_bytes
from .neural_development_state import (
    NEURAL_ENSEMBLE_SCHEMA,
    NeuralDevelopmentStateError,
    ensemble_neural_development_states,
    load_neural_development_state_exact,
    recompute_neural_development_predictions,
)
from .predictor_provenance import (
    DEVELOPMENT_PARTITION,
    ENSEMBLE_RULE,
    ENSEMBLE_VARIANCE_FORMULA,
    FROZEN_NEURAL_SEEDS,
    NEURAL_KIND,
    NON_DATA_AUTHORITY_HASH_KEYS,
    PRIMARY_INFORMATION_SET,
    PredictorProvenanceError,
    canonical_predictor_provenance_bytes,
    parse_predictor_input_jsonl,
    predictor_input_jsonl_bytes,
    validate_development_checkpoint,
    validate_development_checkpoint_selection,
    validate_predictor_provenance,
)


CONTINUATION_NEURAL_EVIDENCE_SCHEMA = "r3x_continuation_neural_evidence_receipt_v1"
CONTINUATION_NEURAL_EVIDENCE_ROLE = (
    "ORDINARY_DEVELOPMENT_EXACT_NEURAL_STATE_RECEIPT"
)
CONTINUATION_NEURAL_EVIDENCE_STATUS = (
    "COMPLETE_EXACT_STATE_LOAD_AND_DEVELOPMENT_REPLAY_"
    "NO_INDEPENDENT_TRAIN_REFIT_COMPONENT_OR_TARGET_REPLAY"
)

BOUNDARY = {
    "ordinary_development_only": True,
    "internal_test_access": False,
    "sealed_access": False,
    "best_seed_selection": False,
    "independent_train_refit_verified": False,
    "independent_component_assignment_replay_verified": False,
    "independent_development_target_replay_verified": False,
    "execution_authorized": False,
    "training_authorized": False,
    "independent_authority_verified": False,
    "publication_evidence": False,
    "downstream_unlock": False,
}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_PROHIBITED_PATH_PARTS = {
    "test",
    "internal_test",
    "sealed",
    "restricted",
    "gse246381",
}

_ROW_BINDING_KEYS = {
    "n_rows",
    "ordered_row_ids_sha256",
    "row_set_sha256",
    "content_sha256",
}
_CELL_KEYS = {
    "cell_id",
    "train_rows_ref",
    "development_rows_ref",
    "ensemble_prediction_ref",
    "checkpoint_selection_refs",
    "checkpoint_refs",
    "state_refs",
    "seed_development_prediction_refs",
}
_ESTIMAND_KEYS = {
    "estimand_id",
    "estimand_binding_sha256",
    "partition_row_bindings",
    "state_refs",
    "seeds",
    "state_tensor_sha256",
    "checkpoint_selection_sha256",
    "development_prediction_sha256",
    "component_assignment_binding",
    "training_design_sha256",
    "development_target_sha256",
    "action_replay_sha256",
    "resource_preflight_sha256",
    "resource_preflight_status",
    "ensemble_schema",
    "ensemble_prediction_sha256",
    "cells",
}
_RECEIPT_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "predictor_provenance_ref",
    "run_start_sha256",
    "activation_manifest_sha256",
    "train_development_outcomes_sha256",
    "non_data_authority_hashes",
    "method_id",
    "training_variant_id",
    "control_intervention",
    "architecture_id",
    "registry_schema",
    "registry_sha256",
    "compiled_binding_sha256",
    "model_config_sha256",
    "fit_config_sha256",
    "information_set",
    "information_set_fingerprint",
    "predictor_code_authority_sha256",
    "predictor_config_authority_sha256",
    "action_policy",
    "allowed_edit_counts",
    "resource_preflight_policy",
    "exact_recompute_policy",
    "ensemble_rule",
    "ensemble_variance_formula",
    "registered_cell_ids",
    "continuation_development_ordered_row_ids_sha256",
    "continuation_development_row_set_sha256",
    "estimands",
} | set(BOUNDARY)


class ContinuationNeuralEvidenceError(RuntimeError):
    """Fail-closed exact neural receipt or replay error."""


def _strict_keys(value: Mapping, expected: set[str], *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ContinuationNeuralEvidenceError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ContinuationNeuralEvidenceError(
            f"{label} keys differ from strict schema: {sorted(actual ^ expected)}"
        )


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise ContinuationNeuralEvidenceError(
            f"{label} must be 64 lowercase hexadecimal characters"
        )
    return value


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ContinuationNeuralEvidenceError(f"{label} must be a stable identifier")
    return value


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
        raise ContinuationNeuralEvidenceError(
            f"value is not canonical finite JSON: {exc}"
        ) from exc
    return hashlib.sha256(raw).hexdigest()


def _unique_json_object(pairs) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json_object(raw: bytes, *, label: str) -> dict:
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_json_object
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContinuationNeuralEvidenceError(
            f"{label} is not strict JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ContinuationNeuralEvidenceError(f"{label} must be a JSON object")
    return value


def _development_ref(value: Mapping, *, label: str) -> dict:
    try:
        reference = validate_artifact_ref(value)
    except ArtifactGraphError as exc:
        raise ContinuationNeuralEvidenceError(str(exc)) from exc
    parts = [part.lower() for part in PurePosixPath(reference["relative_path"]).parts]
    if "development" not in parts:
        raise ContinuationNeuralEvidenceError(
            f"{label} must be beneath a literal development namespace"
        )
    if any(
        part in _PROHIBITED_PATH_PARTS
        or "internal_test" in part
        or "gse246381" in part
        or part.startswith("sealed")
        or part.startswith("restricted")
        for part in parts
    ):
        raise ContinuationNeuralEvidenceError(
            f"{label} enters a protected partition namespace"
        )
    return reference


def _read_ref(run_root: Path, reference: Mapping, *, label: str) -> bytes:
    normalized = _development_ref(reference, label=label)
    try:
        return read_artifact_ref(Path(run_root), normalized)
    except ArtifactGraphError as exc:
        raise ContinuationNeuralEvidenceError(str(exc)) from exc


def _load_canonical_json_ref(
    run_root: Path,
    reference: Mapping,
    *,
    label: str,
    validator,
) -> dict:
    raw = _read_ref(run_root, reference, label=label)
    try:
        value = validator(_parse_json_object(raw, label=label))
        canonical = canonical_json_bytes(value)
    except (ArtifactIOError, PredictorProvenanceError) as exc:
        raise ContinuationNeuralEvidenceError(f"{label} is invalid: {exc}") from exc
    if canonical != raw:
        raise ContinuationNeuralEvidenceError(f"{label} bytes are not canonical")
    return value


def _row_binding(rows: Sequence[Mapping]) -> dict:
    row_ids = [row["row_id"] for row in rows]
    try:
        content = predictor_input_jsonl_bytes(rows)
    except PredictorProvenanceError as exc:
        raise ContinuationNeuralEvidenceError(str(exc)) from exc
    return {
        "n_rows": len(rows),
        "ordered_row_ids_sha256": _canonical_sha256(row_ids),
        "row_set_sha256": _canonical_sha256(sorted(row_ids)),
        "content_sha256": hashlib.sha256(content).hexdigest(),
    }


def _validate_row_binding(value: Mapping, *, label: str) -> dict:
    _strict_keys(value, _ROW_BINDING_KEYS, label=label)
    count = value.get("n_rows")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ContinuationNeuralEvidenceError(f"{label}.n_rows must be positive")
    return {
        "n_rows": count,
        "ordered_row_ids_sha256": _digest(
            value.get("ordered_row_ids_sha256"),
            label=f"{label}.ordered_row_ids_sha256",
        ),
        "row_set_sha256": _digest(
            value.get("row_set_sha256"), label=f"{label}.row_set_sha256"
        ),
        "content_sha256": _digest(
            value.get("content_sha256"), label=f"{label}.content_sha256"
        ),
    }


def _partition_row_bindings(train_rows: Sequence[Mapping], development_rows: Sequence[Mapping]) -> dict:
    return {
        "train": _row_binding(train_rows),
        "development": _row_binding(development_rows),
    }


def _parse_predictor_rows(
    run_root: Path, reference: Mapping, *, partition: str, label: str
) -> list[dict]:
    try:
        rows = parse_predictor_input_jsonl(_read_ref(run_root, reference, label=label))
    except PredictorProvenanceError as exc:
        raise ContinuationNeuralEvidenceError(f"{label} is invalid: {exc}") from exc
    if any(row["partition"] != partition for row in rows):
        raise ContinuationNeuralEvidenceError(f"{label} uses the wrong partition")
    return rows


def _parse_predictions(run_root: Path, reference: Mapping, *, label: str) -> list[dict]:
    try:
        return parse_prediction_jsonl(_read_ref(run_root, reference, label=label))
    except ArtifactGraphError as exc:
        raise ContinuationNeuralEvidenceError(f"{label} is invalid: {exc}") from exc


def _same_across_states(states: Sequence[Mapping], field: str):
    first = states[0][field]
    if any(state[field] != first for state in states[1:]):
        raise ContinuationNeuralEvidenceError(
            f"five neural fitted states differ in {field}"
        )
    return copy.deepcopy(first)


def _state_refs_for_estimand(
    run_root: Path,
    provenance: Mapping,
    estimand: Mapping,
) -> tuple[
    tuple[list[dict], list[dict], list[dict], list[list[dict]], list[list[dict]]],
    list[dict],
]:
    """Load exact cell inputs/envelopes and return one state ref per seed."""

    train_rows: list[dict] = []
    development_rows: list[dict] = []
    cell_material: list[dict] = []
    state_refs_by_seed: list[dict | None] = [None] * len(FROZEN_NEURAL_SEEDS)
    selections_by_cell: list[list[dict]] = []
    checkpoints_by_cell: list[list[dict]] = []

    for cell in estimand["cells"]:
        artifacts = cell["predictor_artifacts"]
        if artifacts["kind"] != NEURAL_KIND:
            raise ContinuationNeuralEvidenceError(
                "continuation neural receipt requires neural evidence in every cell"
            )
        if artifacts["seeds"] != list(FROZEN_NEURAL_SEEDS):
            raise ContinuationNeuralEvidenceError("cell seed order is not frozen")
        cell_train = _parse_predictor_rows(
            run_root,
            cell["train_rows_ref"],
            partition="ORDINARY_TRAIN",
            label=f"{estimand['estimand_id']} {cell['cell_id']} train rows",
        )
        cell_development = _parse_predictor_rows(
            run_root,
            cell["development_rows_ref"],
            partition=DEVELOPMENT_PARTITION,
            label=f"{estimand['estimand_id']} {cell['cell_id']} development rows",
        )
        if any(
            row["estimand_id"] != estimand["estimand_id"]
            or row["cell_id"] != cell["cell_id"]
            for row in (*cell_train, *cell_development)
        ):
            raise ContinuationNeuralEvidenceError(
                "predictor rows differ from their estimand/cell binding"
            )
        cell_train_binding = _row_binding(cell_train)
        cell_development_binding = _row_binding(cell_development)
        train_rows.extend(cell_train)
        development_rows.extend(cell_development)

        cell_selections = []
        cell_checkpoints = []
        for index, seed_artifact in enumerate(artifacts["seed_artifacts"]):
            seed = FROZEN_NEURAL_SEEDS[index]
            selection = _load_canonical_json_ref(
                run_root,
                seed_artifact["checkpoint_selection_ref"],
                label=f"seed {seed} checkpoint selection",
                validator=validate_development_checkpoint_selection,
            )
            checkpoint = _load_canonical_json_ref(
                run_root,
                seed_artifact["checkpoint_ref"],
                label=f"seed {seed} checkpoint",
                validator=validate_development_checkpoint,
            )
            if selection["seed"] != seed or checkpoint["seed"] != seed:
                raise ContinuationNeuralEvidenceError("checkpoint seed binding changed")
            expected_checkpoint_bindings = {
                "run_start_sha256": provenance["run_start_sha256"],
                "activation_manifest_sha256": provenance[
                    "activation_manifest_sha256"
                ],
                "train_development_outcomes_sha256": provenance[
                    "train_development_outcomes_sha256"
                ],
                "non_data_authority_hashes": provenance[
                    "non_data_authority_hashes"
                ],
                "method_id": provenance["method_id"],
                "estimand_id": estimand["estimand_id"],
                "estimand_binding_sha256": estimand["estimand_binding_sha256"],
                "cell_id": cell["cell_id"],
                "seed": seed,
                "train_row_set_sha256": cell_train_binding["row_set_sha256"],
                "development_row_set_sha256": cell_development_binding[
                    "row_set_sha256"
                ],
                "predictor_code_sha256": provenance["predictor_code_sha256"],
                "predictor_config_sha256": provenance[
                    "predictor_config_sha256"
                ],
            }
            for field, expected in expected_checkpoint_bindings.items():
                if selection[field] != expected or checkpoint[field] != expected:
                    raise ContinuationNeuralEvidenceError(
                        f"checkpoint/selection {field} differs from exact provenance rows"
                    )
            if checkpoint["information_set"] != provenance["information_set"] or checkpoint[
                "information_set_fingerprint"
            ] != provenance["information_set_fingerprint"]:
                raise ContinuationNeuralEvidenceError(
                    "checkpoint information set differs from predictor provenance"
                )
            if checkpoint["checkpoint_selection_ref"] != seed_artifact[
                "checkpoint_selection_ref"
            ]:
                raise ContinuationNeuralEvidenceError(
                    "checkpoint selection reference binding changed"
                )
            state_ref = _development_ref(
                checkpoint["state_ref"], label=f"seed {seed} neural state reference"
            )
            if state_refs_by_seed[index] is None:
                state_refs_by_seed[index] = state_ref
            elif state_refs_by_seed[index] != state_ref:
                raise ContinuationNeuralEvidenceError(
                    "one estimand must use one exact fitted state per seed across cells"
                )
            cell_selections.append(selection)
            cell_checkpoints.append(checkpoint)
        selections_by_cell.append(cell_selections)
        checkpoints_by_cell.append(cell_checkpoints)
        cell_material.append(
            {
                "cell": cell,
                "train_rows": cell_train,
                "development_rows": cell_development,
            }
        )

    train_ids = [row["row_id"] for row in train_rows]
    development_ids = [row["row_id"] for row in development_rows]
    if len(train_ids) != len(set(train_ids)) or len(development_ids) != len(
        set(development_ids)
    ):
        raise ContinuationNeuralEvidenceError(
            "train/development rows repeat across provenance cells"
        )
    if set(train_ids).intersection(development_ids):
        raise ContinuationNeuralEvidenceError(
            "train and development row bindings overlap"
        )
    train_rows.sort(key=lambda row: row["row_id"])
    development_rows.sort(key=lambda row: row["row_id"])
    if any(reference is None for reference in state_refs_by_seed):
        raise ContinuationNeuralEvidenceError("five neural state references are incomplete")
    return (
        train_rows,
        development_rows,
        cell_material,
        selections_by_cell,
        checkpoints_by_cell,
    ), [dict(reference) for reference in state_refs_by_seed]


def _build_estimand_receipt(
    run_root: Path,
    provenance: Mapping,
    estimand: Mapping,
) -> tuple[dict, dict, list[str]]:
    materials, state_refs = _state_refs_for_estimand(run_root, provenance, estimand)
    (
        train_rows,
        development_rows,
        cell_material,
        selections_by_cell,
        checkpoints_by_cell,
    ) = materials
    states = []
    for seed, reference in zip(FROZEN_NEURAL_SEEDS, state_refs, strict=True):
        try:
            state_path = resolve_artifact_ref(run_root, reference)
            state = load_neural_development_state_exact(
                state_path, reference["sha256"]
            )
        except (ArtifactGraphError, NeuralDevelopmentStateError) as exc:
            raise ContinuationNeuralEvidenceError(
                f"seed {seed} neural state exact-load failed: {exc}"
            ) from exc
        states.append(state)

    row_bindings = _partition_row_bindings(train_rows, development_rows)
    expected_pairs = {
        "method_id": provenance["method_id"],
        "estimand_id": estimand["estimand_id"],
        "run_start_sha256": provenance["run_start_sha256"],
        "development_activation_sha256": provenance["activation_manifest_sha256"],
        "train_development_outcomes_sha256": provenance[
            "train_development_outcomes_sha256"
        ],
        "non_data_authority_hashes": provenance["non_data_authority_hashes"],
        "information_set": provenance["information_set"],
        "information_set_fingerprint": provenance["information_set_fingerprint"],
        "partition_row_bindings": row_bindings,
    }
    for state, seed in zip(states, FROZEN_NEURAL_SEEDS, strict=True):
        if state["seed"] != seed:
            raise ContinuationNeuralEvidenceError("neural state seed order changed")
        for field, expected in expected_pairs.items():
            if state[field] != expected:
                raise ContinuationNeuralEvidenceError(
                    f"neural state {field} differs from exact provenance/row binding"
                )

    common_fields = (
        "method_id",
        "training_variant_id",
        "architecture_id",
        "registry_schema",
        "registry_sha256",
        "compiled_binding_sha256",
        "model_config_sha256",
        "fit_config_sha256",
        "information_set",
        "information_set_fingerprint",
        "component_assignment_binding",
        "training_design_sha256",
        "development_target_sha256",
        "action_policy",
        "allowed_edit_counts",
        "action_replay_sha256",
        "resource_preflight_policy",
        "resource_preflight",
        "exact_recompute_policy",
    )
    common = {field: _same_across_states(states, field) for field in common_fields}
    control_intervention = _same_across_states(states, "compiled_binding").get(
        "control_intervention"
    )

    recomputed_by_seed = []
    for state in states:
        try:
            recomputed_by_seed.append(
                recompute_neural_development_predictions(state, development_rows)
            )
        except NeuralDevelopmentStateError as exc:
            raise ContinuationNeuralEvidenceError(
                f"neural development recomputation failed: {exc}"
            ) from exc
    try:
        ensemble = ensemble_neural_development_states(states)
    except NeuralDevelopmentStateError as exc:
        raise ContinuationNeuralEvidenceError(
            f"five-seed ensemble rebuild failed: {exc}"
        ) from exc
    recomputed_maps = [
        {row["row_id"]: row for row in rows} for rows in recomputed_by_seed
    ]
    ensemble_map = {
        row["row_id"]: row for row in ensemble["ensemble_predictions"]
    }

    cell_receipts = []
    for material, selections, checkpoints in zip(
        cell_material, selections_by_cell, checkpoints_by_cell, strict=True
    ):
        cell = material["cell"]
        cell_ids = [row["row_id"] for row in material["development_rows"]]
        for index, seed_artifact in enumerate(cell["predictor_artifacts"]["seed_artifacts"]):
            expected_rows = [recomputed_maps[index][row_id] for row_id in cell_ids]
            committed_rows = _parse_predictions(
                run_root,
                seed_artifact["development_prediction_ref"],
                label=f"seed {seed_artifact['seed']} development prediction",
            )
            try:
                exact_match = prediction_jsonl_bytes(expected_rows) == prediction_jsonl_bytes(
                    committed_rows
                )
            except ArtifactGraphError as exc:
                raise ContinuationNeuralEvidenceError(str(exc)) from exc
            if not exact_match:
                raise ContinuationNeuralEvidenceError(
                    "seed prediction bytes differ from exact neural state recomputation"
                )
            state = states[index]
            selection = selections[index]
            checkpoint = checkpoints[index]
            if selection["selected_epoch"] != state["selected_epoch"]:
                raise ContinuationNeuralEvidenceError(
                    "generic checkpoint selection differs from neural fitted state"
                )
            if checkpoint["architecture_id"] != state["architecture_id"]:
                raise ContinuationNeuralEvidenceError(
                    "generic checkpoint architecture differs from neural fitted state"
                )
            for field in (
                "run_start_sha256",
                "train_development_outcomes_sha256",
                "non_data_authority_hashes",
                "method_id",
                "information_set_fingerprint",
            ):
                state_field = field
                if checkpoint[field] != state[state_field]:
                    raise ContinuationNeuralEvidenceError(
                        f"generic checkpoint {field} differs from neural fitted state"
                    )

        expected_ensemble_rows = [ensemble_map[row_id] for row_id in cell_ids]
        committed_ensemble_rows = _parse_predictions(
            run_root,
            cell["prediction_jsonl_ref"],
            label=f"{cell['cell_id']} ensemble prediction",
        )
        try:
            ensemble_matches = prediction_jsonl_bytes(
                expected_ensemble_rows
            ) == prediction_jsonl_bytes(committed_ensemble_rows)
        except ArtifactGraphError as exc:
            raise ContinuationNeuralEvidenceError(str(exc)) from exc
        if not ensemble_matches:
            raise ContinuationNeuralEvidenceError(
                "ensemble prediction bytes differ from exact five-state rebuild"
            )
        cell_receipts.append(
            {
                "cell_id": cell["cell_id"],
                "train_rows_ref": cell["train_rows_ref"],
                "development_rows_ref": cell["development_rows_ref"],
                "ensemble_prediction_ref": cell["prediction_jsonl_ref"],
                "checkpoint_selection_refs": [
                    item["checkpoint_selection_ref"]
                    for item in cell["predictor_artifacts"]["seed_artifacts"]
                ],
                "checkpoint_refs": [
                    item["checkpoint_ref"]
                    for item in cell["predictor_artifacts"]["seed_artifacts"]
                ],
                "state_refs": copy.deepcopy(state_refs),
                "seed_development_prediction_refs": [
                    item["development_prediction_ref"]
                    for item in cell["predictor_artifacts"]["seed_artifacts"]
                ],
            }
        )

    receipt = {
        "estimand_id": estimand["estimand_id"],
        "estimand_binding_sha256": estimand["estimand_binding_sha256"],
        "partition_row_bindings": row_bindings,
        "state_refs": state_refs,
        "seeds": list(FROZEN_NEURAL_SEEDS),
        "state_tensor_sha256": [state["state_tensor_sha256"] for state in states],
        "checkpoint_selection_sha256": [
            state["checkpoint_selection_sha256"] for state in states
        ],
        "development_prediction_sha256": [
            state["development_prediction_sha256"] for state in states
        ],
        "component_assignment_binding": common["component_assignment_binding"],
        "training_design_sha256": common["training_design_sha256"],
        "development_target_sha256": common["development_target_sha256"],
        "action_replay_sha256": common["action_replay_sha256"],
        "resource_preflight_sha256": _canonical_sha256(common["resource_preflight"]),
        "resource_preflight_status": common["resource_preflight"]["status"],
        "ensemble_schema": NEURAL_ENSEMBLE_SCHEMA,
        "ensemble_prediction_sha256": ensemble["ensemble_prediction_sha256"],
        "cells": cell_receipts,
    }
    top_common = {
        **common,
        "control_intervention": control_intervention,
    }
    continuation_row_ids = [
        row["row_id"]
        for material in cell_material
        for row in material["development_rows"]
    ]
    return receipt, top_common, continuation_row_ids


def build_continuation_neural_evidence(
    run_root: Path,
    predictor_provenance_ref: Mapping,
) -> dict:
    """Rebuild exact neural evidence and return a strict continuation receipt."""

    provenance_ref = _development_ref(
        predictor_provenance_ref, label="predictor provenance reference"
    )
    raw = _read_ref(run_root, provenance_ref, label="predictor provenance")
    try:
        provenance = validate_predictor_provenance(
            _parse_json_object(raw, label="predictor provenance")
        )
        canonical = canonical_predictor_provenance_bytes(provenance)
    except (PredictorProvenanceError, ArtifactIOError) as exc:
        raise ContinuationNeuralEvidenceError(
            f"predictor provenance is invalid: {exc}"
        ) from exc
    if canonical != raw:
        raise ContinuationNeuralEvidenceError(
            "predictor provenance bytes are not canonical"
        )

    estimand_receipts = []
    common_bindings = []
    continuation_row_ids = []
    for estimand in provenance["estimands"]:
        receipt, common, estimand_row_ids = _build_estimand_receipt(
            run_root, provenance, estimand
        )
        estimand_receipts.append(receipt)
        common_bindings.append(common)
        continuation_row_ids.extend(estimand_row_ids)
    if not estimand_receipts:
        raise ContinuationNeuralEvidenceError("neural receipt has no estimands")

    shared_fields = (
        "method_id",
        "training_variant_id",
        "architecture_id",
        "registry_schema",
        "registry_sha256",
        "compiled_binding_sha256",
        "model_config_sha256",
        "fit_config_sha256",
        "information_set",
        "information_set_fingerprint",
        "action_policy",
        "allowed_edit_counts",
        "resource_preflight_policy",
        "exact_recompute_policy",
        "control_intervention",
    )
    shared = {}
    for field in shared_fields:
        first = common_bindings[0][field]
        if any(item[field] != first for item in common_bindings[1:]):
            raise ContinuationNeuralEvidenceError(
                f"neural estimands differ in continuation binding: {field}"
            )
        shared[field] = copy.deepcopy(first)

    if shared["method_id"] != provenance["method_id"]:
        raise ContinuationNeuralEvidenceError(
            "neural state method differs from predictor provenance"
        )
    if shared["information_set"] != provenance["information_set"] or shared[
        "information_set_fingerprint"
    ] != provenance["information_set_fingerprint"]:
        raise ContinuationNeuralEvidenceError(
            "neural state information set differs from predictor provenance"
        )

    payload = {
        "schema_version": CONTINUATION_NEURAL_EVIDENCE_SCHEMA,
        "artifact_role": CONTINUATION_NEURAL_EVIDENCE_ROLE,
        "status": CONTINUATION_NEURAL_EVIDENCE_STATUS,
        "predictor_provenance_ref": provenance_ref,
        "run_start_sha256": provenance["run_start_sha256"],
        "activation_manifest_sha256": provenance["activation_manifest_sha256"],
        "train_development_outcomes_sha256": provenance[
            "train_development_outcomes_sha256"
        ],
        "non_data_authority_hashes": provenance["non_data_authority_hashes"],
        "method_id": shared["method_id"],
        "training_variant_id": shared["training_variant_id"],
        "control_intervention": shared["control_intervention"],
        "architecture_id": shared["architecture_id"],
        "registry_schema": shared["registry_schema"],
        "registry_sha256": shared["registry_sha256"],
        "compiled_binding_sha256": shared["compiled_binding_sha256"],
        "model_config_sha256": shared["model_config_sha256"],
        "fit_config_sha256": shared["fit_config_sha256"],
        "information_set": shared["information_set"],
        "information_set_fingerprint": shared["information_set_fingerprint"],
        "predictor_code_authority_sha256": provenance["predictor_code_sha256"],
        "predictor_config_authority_sha256": provenance["predictor_config_sha256"],
        "action_policy": shared["action_policy"],
        "allowed_edit_counts": shared["allowed_edit_counts"],
        "resource_preflight_policy": shared["resource_preflight_policy"],
        "exact_recompute_policy": shared["exact_recompute_policy"],
        "ensemble_rule": ENSEMBLE_RULE,
        "ensemble_variance_formula": ENSEMBLE_VARIANCE_FORMULA,
        "registered_cell_ids": [
            f"{estimand['estimand_id']}::{cell['cell_id']}"
            for estimand in estimand_receipts
            for cell in estimand["cells"]
        ],
        "continuation_development_ordered_row_ids_sha256": _canonical_sha256(
            continuation_row_ids
        ),
        "continuation_development_row_set_sha256": _canonical_sha256(
            sorted(continuation_row_ids)
        ),
        "estimands": estimand_receipts,
        **BOUNDARY,
    }
    return validate_continuation_neural_evidence(payload)


def _validate_refs(values: object, *, label: str, length: int = 5) -> list[dict]:
    if isinstance(values, (str, bytes)) or not isinstance(values, list):
        raise ContinuationNeuralEvidenceError(f"{label} must be a list")
    if len(values) != length:
        raise ContinuationNeuralEvidenceError(f"{label} must contain {length} refs")
    return [
        _development_ref(value, label=f"{label}[{index}]")
        for index, value in enumerate(values)
    ]


def validate_continuation_neural_evidence(payload: Mapping) -> dict:
    """Validate receipt structure; exact state/provenance replay is separate."""

    _strict_keys(payload, _RECEIPT_KEYS, label="continuation neural receipt")
    if payload.get("schema_version") != CONTINUATION_NEURAL_EVIDENCE_SCHEMA:
        raise ContinuationNeuralEvidenceError("unsupported neural receipt schema")
    if payload.get("artifact_role") != CONTINUATION_NEURAL_EVIDENCE_ROLE:
        raise ContinuationNeuralEvidenceError("neural receipt role changed")
    if payload.get("status") != CONTINUATION_NEURAL_EVIDENCE_STATUS:
        raise ContinuationNeuralEvidenceError("neural receipt is not replay-complete")
    for field, expected in BOUNDARY.items():
        if payload.get(field) is not expected:
            raise ContinuationNeuralEvidenceError(
                f"neural receipt boundary changed: {field}"
            )
    normalized = copy.deepcopy(dict(payload))
    normalized["predictor_provenance_ref"] = _development_ref(
        payload.get("predictor_provenance_ref"), label="predictor provenance ref"
    )
    for field in (
        "run_start_sha256",
        "activation_manifest_sha256",
        "train_development_outcomes_sha256",
        "registry_sha256",
        "compiled_binding_sha256",
        "model_config_sha256",
        "fit_config_sha256",
        "information_set_fingerprint",
        "predictor_code_authority_sha256",
        "predictor_config_authority_sha256",
        "continuation_development_ordered_row_ids_sha256",
        "continuation_development_row_set_sha256",
    ):
        _digest(payload.get(field), label=field)
    authorities = payload.get("non_data_authority_hashes")
    _strict_keys(
        authorities,
        set(NON_DATA_AUTHORITY_HASH_KEYS),
        label="non_data_authority_hashes",
    )
    for field in NON_DATA_AUTHORITY_HASH_KEYS:
        _digest(authorities[field], label=f"non_data_authority_hashes.{field}")
    for field in (
        "method_id",
        "training_variant_id",
        "control_intervention",
        "architecture_id",
        "registry_schema",
        "information_set",
        "action_policy",
        "resource_preflight_policy",
        "exact_recompute_policy",
    ):
        _identifier(payload.get(field), label=field)
    if payload.get("information_set") != PRIMARY_INFORMATION_SET:
        raise ContinuationNeuralEvidenceError("neural receipt information set changed")
    if payload.get("allowed_edit_counts") != [1, 3, 5]:
        raise ContinuationNeuralEvidenceError("neural receipt action budgets changed")
    if payload.get("ensemble_rule") != ENSEMBLE_RULE or payload.get(
        "ensemble_variance_formula"
    ) != ENSEMBLE_VARIANCE_FORMULA:
        raise ContinuationNeuralEvidenceError("neural receipt ensemble rule changed")

    registered_cell_ids = payload.get("registered_cell_ids")
    if (
        not isinstance(registered_cell_ids, list)
        or not registered_cell_ids
        or registered_cell_ids != sorted(registered_cell_ids)
        or len(registered_cell_ids) != len(set(registered_cell_ids))
    ):
        raise ContinuationNeuralEvidenceError(
            "neural receipt registered cells must be non-empty, sorted, and unique"
        )
    for cell_id in registered_cell_ids:
        _identifier(cell_id, label="registered_cell_id")

    estimands = payload.get("estimands")
    if isinstance(estimands, (str, bytes)) or not isinstance(estimands, list) or not estimands:
        raise ContinuationNeuralEvidenceError("neural receipt estimands are empty")
    normalized_estimands = []
    prior_estimand = None
    for index, item in enumerate(estimands):
        _strict_keys(item, _ESTIMAND_KEYS, label=f"estimand receipt {index}")
        estimand_id = _identifier(item.get("estimand_id"), label="estimand_id")
        if prior_estimand is not None and estimand_id <= prior_estimand:
            raise ContinuationNeuralEvidenceError(
                "estimand receipts must be unique and sorted"
            )
        prior_estimand = estimand_id
        _digest(item.get("estimand_binding_sha256"), label="estimand_binding_sha256")
        bindings = item.get("partition_row_bindings")
        _strict_keys(bindings, {"train", "development"}, label="partition row bindings")
        normalized_bindings = {
            partition: _validate_row_binding(
                bindings[partition], label=f"partition_row_bindings.{partition}"
            )
            for partition in ("train", "development")
        }
        if normalized_bindings["train"]["row_set_sha256"] == normalized_bindings[
            "development"
        ]["row_set_sha256"]:
            raise ContinuationNeuralEvidenceError(
                "train and development row sets cannot alias"
            )
        state_refs = _validate_refs(item.get("state_refs"), label="state_refs")
        if item.get("seeds") != list(FROZEN_NEURAL_SEEDS):
            raise ContinuationNeuralEvidenceError("estimand seed order changed")
        for field in (
            "state_tensor_sha256",
            "checkpoint_selection_sha256",
            "development_prediction_sha256",
        ):
            values = item.get(field)
            if not isinstance(values, list) or len(values) != len(FROZEN_NEURAL_SEEDS):
                raise ContinuationNeuralEvidenceError(f"{field} is incomplete")
            for value in values:
                _digest(value, label=field)
        component = item.get("component_assignment_binding")
        if not isinstance(component, Mapping) or not component:
            raise ContinuationNeuralEvidenceError(
                "component assignment binding is absent"
            )
        for key, value in component.items():
            _identifier(key, label="component assignment binding key")
            _digest(value, label=f"component_assignment_binding.{key}")
        for field in (
            "training_design_sha256",
            "development_target_sha256",
            "action_replay_sha256",
            "resource_preflight_sha256",
            "ensemble_prediction_sha256",
        ):
            _digest(item.get(field), label=field)
        if item.get("resource_preflight_status") != "PASS_BEFORE_MODEL_CONSTRUCTION":
            raise ContinuationNeuralEvidenceError("resource preflight is not PASS")
        if item.get("ensemble_schema") != NEURAL_ENSEMBLE_SCHEMA:
            raise ContinuationNeuralEvidenceError("ensemble schema changed")
        cells = item.get("cells")
        if isinstance(cells, (str, bytes)) or not isinstance(cells, list) or not cells:
            raise ContinuationNeuralEvidenceError("estimand receipt cells are empty")
        normalized_cells = []
        prior_cell = None
        for cell_index, cell in enumerate(cells):
            _strict_keys(cell, _CELL_KEYS, label=f"cell receipt {cell_index}")
            cell_id = _identifier(cell.get("cell_id"), label="cell_id")
            if prior_cell is not None and cell_id <= prior_cell:
                raise ContinuationNeuralEvidenceError(
                    "cell receipts must be unique and sorted"
                )
            prior_cell = cell_id
            normalized_cell = {"cell_id": cell_id}
            for field in (
                "train_rows_ref",
                "development_rows_ref",
                "ensemble_prediction_ref",
            ):
                normalized_cell[field] = _development_ref(
                    cell.get(field), label=f"cell receipt {field}"
                )
            for field in (
                "checkpoint_selection_refs",
                "checkpoint_refs",
                "state_refs",
                "seed_development_prediction_refs",
            ):
                normalized_cell[field] = _validate_refs(
                    cell.get(field), label=f"cell receipt {field}"
                )
            if normalized_cell["state_refs"] != state_refs:
                raise ContinuationNeuralEvidenceError(
                    "cell receipt state refs differ from estimand refs"
                )
            normalized_cells.append(normalized_cell)
        normalized_item = copy.deepcopy(dict(item))
        normalized_item["partition_row_bindings"] = normalized_bindings
        normalized_item["state_refs"] = state_refs
        normalized_item["cells"] = normalized_cells
        normalized_estimands.append(normalized_item)
    normalized["estimands"] = normalized_estimands
    derived_cell_ids = [
        f"{estimand['estimand_id']}::{cell['cell_id']}"
        for estimand in normalized_estimands
        for cell in estimand["cells"]
    ]
    if registered_cell_ids != derived_cell_ids:
        raise ContinuationNeuralEvidenceError(
            "registered cells differ from exact estimand/cell receipts"
        )
    return normalized


def canonical_continuation_neural_evidence_bytes(payload: Mapping) -> bytes:
    try:
        return canonical_json_bytes(validate_continuation_neural_evidence(payload))
    except ArtifactIOError as exc:
        raise ContinuationNeuralEvidenceError(str(exc)) from exc


def _assert_output_within_run_root(run_root: Path, path: Path) -> None:
    try:
        root = Path(run_root).resolve(strict=True)
        candidate = Path(path).resolve(strict=False)
    except OSError as exc:
        raise ContinuationNeuralEvidenceError(
            f"neural receipt output containment cannot be established: {exc}"
        ) from exc
    if not root.is_dir() or not candidate.is_relative_to(root):
        raise ContinuationNeuralEvidenceError(
            "neural receipt output must remain beneath the exact run root"
        )
    _development_ref(
        {"relative_path": candidate.relative_to(root).as_posix(), "sha256": "0" * 64},
        label="neural receipt output",
    )


def write_continuation_neural_evidence_exclusive(
    run_root: Path,
    path: Path,
    payload: Mapping,
) -> str:
    """Publish only if the receipt exactly rebuilds from its frozen refs."""

    _assert_output_within_run_root(run_root, path)
    validated = validate_continuation_neural_evidence(payload)
    rebuilt = build_continuation_neural_evidence(
        run_root, validated["predictor_provenance_ref"]
    )
    if rebuilt != validated:
        raise ContinuationNeuralEvidenceError(
            "neural receipt differs from exact state/provenance replay"
        )
    try:
        return atomic_write_json_exclusive(
            Path(path), rebuilt, validator=validate_continuation_neural_evidence
        )
    except ArtifactIOError as exc:
        raise ContinuationNeuralEvidenceError(str(exc)) from exc


def load_continuation_neural_evidence_exact(
    run_root: Path,
    reference: Mapping,
) -> dict:
    """Exact-load canonical receipt bytes and replay every bound neural state."""

    raw = _read_ref(run_root, reference, label="continuation neural receipt")
    payload = validate_continuation_neural_evidence(
        _parse_json_object(raw, label="continuation neural receipt")
    )
    if canonical_continuation_neural_evidence_bytes(payload) != raw:
        raise ContinuationNeuralEvidenceError("neural receipt bytes are not canonical")
    rebuilt = build_continuation_neural_evidence(
        run_root, payload["predictor_provenance_ref"]
    )
    if rebuilt != payload:
        raise ContinuationNeuralEvidenceError(
            "neural receipt does not match exact state/provenance replay"
        )
    return payload


__all__ = [
    "CONTINUATION_NEURAL_EVIDENCE_ROLE",
    "CONTINUATION_NEURAL_EVIDENCE_SCHEMA",
    "CONTINUATION_NEURAL_EVIDENCE_STATUS",
    "ContinuationNeuralEvidenceError",
    "build_continuation_neural_evidence",
    "canonical_continuation_neural_evidence_bytes",
    "load_continuation_neural_evidence_exact",
    "validate_continuation_neural_evidence",
    "write_continuation_neural_evidence_exclusive",
]
