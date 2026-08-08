"""Development-only strongest same-information-set baseline selection.

This module is intentionally independent from the ordinary runner.  A caller
must publish one prediction JSONL per ``estimand x registered metric cell``
under a development namespace, then publish one strict evidence artifact per
eligible method.  The selector replays every referenced byte stream and
requires identical information-set fingerprints, estimand/cell coverage, row
sets, and row order before applying the frozen selection rule.

The selector never accepts internal-test fields or paths.  Every cell binds a
second immutable development-evaluation JSONL containing true delta, source
component, and ``source_group_id_v2``.  Build and exact replay independently
join those rows to predictions by row ID and recompute every selection metric.

Version 2 additionally requires an exact predictor-provenance reference for
every method.  Predictor replay must reconstruct the published predictions
from the committed fitted state/checkpoints through a separately frozen trusted
verifier.  Version-1 evidence has no upgrade path to a qualified selection.

Selection version 3 separates deterministic comparator selection from
qualification.  A comparator is always selected from the development evidence
that can be exactly replayed, using raw component/cell-balanced sign accuracy
before rank, MAE, and frozen order.  The comparator is recovery-only unless the
entire frozen same-information method set (including every required strong
baseline) is evidenced and the selected comparator meets the unchanged 0.60
sign gate.  Version-2 selector artifacts have no implicit upgrade path.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from numbers import Real
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

from .artifact_graph import (
    ArtifactGraphError,
    PRIMARY_INFORMATION_SET,
    parse_prediction_jsonl,
    read_artifact_ref,
    validate_artifact_ref,
)
from .artifact_io import (
    ArtifactIOError,
    atomic_publish_bytes,
    atomic_write_json_exclusive,
    canonical_json_bytes,
    read_exact_regular_file,
)
from .metrics import (
    MetricProtocolError,
    component_balanced_cell_metrics,
    within_pool_rank_metrics,
)
from .protocol import SIGN_ACCURACY_GATE
from .predictor_provenance import (
    NON_DATA_AUTHORITY_HASH_KEYS,
    PredictorProvenanceError,
    PredictorVerifier,
    predictor_layout,
    replay_predictor_provenance,
)


DEVELOPMENT_EVIDENCE_SCHEMA = "r3x_development_baseline_evidence_v2"
DEVELOPMENT_EVALUATION_ROW_SCHEMA = "r3x_development_evaluation_row_v1"
SELECTION_SCHEMA = "r3x_development_strongest_baseline_selection_v3"
DEVELOPMENT_PARTITION = "ORDINARY_DEVELOPMENT"

EVIDENCE_ROLE = "ORDINARY_DEVELOPMENT_SAME_INFORMATION_BASELINE_EVIDENCE"
SELECTION_ROLE = "ORDINARY_DEVELOPMENT_STRONGEST_BASELINE_SELECTION"

SIGN_METRIC = "MACRO_COMPONENT_CELL_BALANCED_SIGN_ACCURACY"
RANK_METRIC = "MACRO_WITHIN_POOL_SPEARMAN"
ERROR_METRIC = "MACRO_COMPONENT_CELL_BALANCED_MAE"
SELECTION_RULE = (
    "COMPONENT_CELL_BALANCED_SIGN_THEN_WITHIN_POOL_SPEARMAN_THEN_NEGATIVE_MAE_"
    "THEN_FROZEN_METHOD_ORDER"
)
QUALIFIED_STATUS = "SELECTED_DEVELOPMENT_BASELINE_MET_SIGN_GATE"
HOLD_STATUS = "HOLD_RECOVERY_ONLY_NON_QUALIFIED_COMPARATOR"

MINIMUM_REQUIRED_METHOD_IDS = frozenset({
    "FULL_PAIR_UNRESTRICTED_SIAMESE",
    "HISTOGRAM_GRADIENT_BOOSTING",
    "PAIRED_LOCAL_GLOBAL_KMER_RIDGE",
    "PAIRED_FEATURE_NEAREST_NEIGHBOR",
})

METRIC_POLICY = {
    "policy_id": "R3X_DEVELOPMENT_STRONGEST_BASELINE_SELECTION_V2",
    "partition": DEVELOPMENT_PARTITION,
    "required_information_set": PRIMARY_INFORMATION_SET,
    "cell_sign_metric": "COMPONENT_BALANCED_SIGN_ACCURACY",
    "sign_decision_rule": "RAW_PREDICTED_DELTA_MEAN_GE_ZERO",
    "zero_policy": "EXCLUDE_TRUE_ZERO",
    "sign_gate": SIGN_ACCURACY_GATE,
    "cell_rank_metric": "WITHIN_POOL_SPEARMAN",
    "within_pool_min_candidates": 3,
    "cell_error_metric": "COMPONENT_BALANCED_MAE",
    "macro_aggregation": "UNWEIGHTED_COMPLETE_ESTIMAND_CELL_MACRO",
    "selection_rule": SELECTION_RULE,
    "test_access": False,
}

# This constant is deliberately executable documentation for the future
# runner patch; this module itself does not alter runner control flow.
RUNNER_INTEGRATION_REQUIREMENTS = (
    "Freeze eligible_method_order before loading development outcomes; publish "
    "one canonical prediction JSONL per method/estimand/cell and one shared "
    "hash-bound evaluation-row JSONL with true_delta, source_component_id, and "
    "source_group_id_v2 under a literal development path; derive the required "
    "estimand/cell/pool layout from frozen registry and split authorities; "
    "publish and exact-replay one v2 predictor-provenance commit per method, "
    "then publish method evidence and this selector before any internal-test loader; "
    "bind only train_development_outcomes_sha256 plus the "
    "exact non-data authority set; "
    "use selected_method_id for pre-test freeze only when status is qualified; "
    "a recovery-only comparator can guide ordinary development but cannot satisfy "
    "pre-test freeze."
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_METHOD_ID = re.compile(r"^[A-Z][A-Z0-9_]*$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")

_METRIC_POLICY_KEYS = {
    "policy_id",
    "partition",
    "required_information_set",
    "cell_sign_metric",
    "sign_decision_rule",
    "zero_policy",
    "sign_gate",
    "cell_rank_metric",
    "within_pool_min_candidates",
    "cell_error_metric",
    "macro_aggregation",
    "selection_rule",
    "test_access",
}
_INPUT_CELL_KEYS = {
    "cell_id",
    "prediction_jsonl_ref",
    "development_evaluation_jsonl_ref",
}
_CELL_KEYS = _INPUT_CELL_KEYS | {
    "n_rows",
    "ordered_row_ids_sha256",
    "row_set_sha256",
    "evaluation_rows_sha256",
    "registered_source_group_ids_v2",
    "structurally_eligible_source_group_ids_v2",
    "n_components",
    "n_registered_pools",
    "n_structurally_eligible_pools",
    "component_cell_sign_accuracy",
    "component_cell_sign_status",
    "within_pool_spearman",
    "within_pool_spearman_status",
    "component_cell_mae",
    "component_cell_mae_status",
}
_ESTIMAND_KEYS = {"estimand_id", "cells"}
_AGGREGATE_KEYS = {
    "n_estimands",
    "n_cells",
    "n_rows",
    "macro_component_cell_sign_accuracy",
    "macro_within_pool_spearman",
    "macro_component_cell_mae",
    "complete",
}
_EVIDENCE_KEYS = {
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
    "predictor_provenance_ref",
    "metric_policy",
    "metric_policy_sha256",
    "estimands",
    "aggregate_development_metrics",
    "ordinary_only",
    "test_access",
    "sealed_access",
    "downstream_unlock",
}
_LAYOUT_CELL_KEYS = {
    "cell_id",
    "n_rows",
    "ordered_row_ids_sha256",
    "row_set_sha256",
    "evaluation_rows_sha256",
    "registered_source_group_ids_v2",
    "structurally_eligible_source_group_ids_v2",
    "n_components",
    "n_registered_pools",
    "n_structurally_eligible_pools",
}
_LAYOUT_ESTIMAND_KEYS = {"estimand_id", "cells"}
_REQUIRED_ESTIMAND_KEYS = {"estimand_id", "cells"}
_REQUIRED_CELL_KEYS = {"cell_id", "registered_source_group_ids_v2"}

_EVALUATION_ROW_KEYS = {
    "schema_version",
    "row_id",
    "partition",
    "estimand_id",
    "cell_id",
    "true_delta",
    "source_component_id",
    "source_group_id_v2",
}
_SELECTION_METRIC_KEYS = {
    "macro_component_cell_sign_accuracy",
    "macro_within_pool_spearman",
    "macro_component_cell_mae",
}
_SELECTION_RECORD_KEYS = {
    "method_id",
    "development_evidence_ref",
    "method_order_index",
    "development_metrics",
    "sign_gate_met",
    "lexicographic_selection_tuple",
}
_SELECTION_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "run_start_sha256",
    "activation_manifest_sha256",
    "train_development_outcomes_sha256",
    "non_data_authority_hashes",
    "partition",
    "information_set",
    "information_set_fingerprint",
    "required_method_ids",
    "required_method_set_sha256",
    "missing_required_method_ids",
    "missing_required_evidence_method_ids",
    "missing_eligible_evidence_method_ids",
    "eligible_method_ids",
    "eligible_method_order",
    "eligible_method_set_sha256",
    "eligible_method_order_sha256",
    "development_evidence_refs",
    "required_estimand_cells",
    "required_estimand_cells_sha256",
    "development_layout",
    "development_layout_sha256",
    "metric_policy",
    "metric_policy_sha256",
    "selection_tuple_fields",
    "method_selection_records",
    "selected_method_id",
    "selected_development_metrics",
    "comparator_selected",
    "baseline_sign_gate_met",
    "strongest_same_information_claim",
    "can_satisfy_pretest_freeze",
    "ordinary_only",
    "test_access",
    "sealed_access",
    "downstream_unlock",
}


class BaselineSelectionError(RuntimeError):
    """Fail-closed development evidence or selector replay error."""


def _strict_keys(value: Mapping, expected: set[str], *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise BaselineSelectionError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise BaselineSelectionError(
            f"{label} keys differ from strict schema: {sorted(actual ^ expected)}"
        )


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise BaselineSelectionError(f"{label} must be 64 lowercase hex")
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
        raise BaselineSelectionError(f"value is not canonical finite JSON: {exc}") from exc
    return hashlib.sha256(raw).hexdigest()


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


def _method_id(value: object, *, label: str = "method_id") -> str:
    if not isinstance(value, str) or not _METHOD_ID.fullmatch(value):
        raise BaselineSelectionError(f"{label} is not a safe frozen method identifier")
    return value


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise BaselineSelectionError(f"{label} is not a canonical identifier")
    return value


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BaselineSelectionError(f"{label} must be a positive integer")
    return value


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise BaselineSelectionError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise BaselineSelectionError(f"{label} must be a finite number")
    return result


def _boundaries(payload: Mapping, *, label: str) -> None:
    if payload.get("ordinary_only") is not True:
        raise BaselineSelectionError(f"{label} must remain ordinary-only")
    if payload.get("test_access") is not False:
        raise BaselineSelectionError(f"{label} cannot use test data")
    if payload.get("sealed_access") is not False:
        raise BaselineSelectionError(f"{label} cannot use sealed data")
    if payload.get("downstream_unlock") is not False:
        raise BaselineSelectionError(f"{label} cannot unlock downstream phases")


def _metric_policy(value: Mapping) -> dict:
    _strict_keys(value, _METRIC_POLICY_KEYS, label="metric_policy")
    if dict(value) != METRIC_POLICY:
        raise BaselineSelectionError("metric_policy differs from the frozen policy")
    return dict(METRIC_POLICY)


def _development_ref(value: Mapping, *, label: str) -> dict:
    try:
        reference = validate_artifact_ref(value)
    except ArtifactGraphError as exc:
        raise BaselineSelectionError(str(exc)) from exc
    parts = [part.lower() for part in PurePosixPath(reference["relative_path"]).parts]
    if "development" not in parts:
        raise BaselineSelectionError(
            f"{label} must be published beneath a literal development namespace"
        )
    if any(
        part in {"test", "internal_test", "sealed", "restricted"}
        or "internal_test" in part
        or "gse246381" in part
        for part in parts
    ):
        raise BaselineSelectionError(f"{label} enters a prohibited partition namespace")
    return reference


def _row_id_hashes(rows: Sequence[Mapping]) -> tuple[str, str]:
    row_ids = [str(row["row_id"]) for row in rows]
    if not row_ids or len(row_ids) != len(set(row_ids)):
        raise BaselineSelectionError("development prediction row IDs must be unique")
    ordered = json.dumps(
        row_ids, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    unordered = json.dumps(
        sorted(row_ids), separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(ordered).hexdigest(), hashlib.sha256(unordered).hexdigest()


def _normalize_evaluation_row(value: Mapping) -> dict:
    _strict_keys(value, _EVALUATION_ROW_KEYS, label="development evaluation row")
    if value.get("schema_version") != DEVELOPMENT_EVALUATION_ROW_SCHEMA:
        raise BaselineSelectionError("unsupported development evaluation-row schema")
    if value.get("partition") != DEVELOPMENT_PARTITION:
        raise BaselineSelectionError(
            "development evaluation row cannot reference another partition"
        )
    return {
        "schema_version": DEVELOPMENT_EVALUATION_ROW_SCHEMA,
        "row_id": _digest(value.get("row_id"), label="evaluation row_id"),
        "partition": DEVELOPMENT_PARTITION,
        "estimand_id": _identifier(
            value.get("estimand_id"), label="evaluation estimand_id"
        ),
        "cell_id": _identifier(value.get("cell_id"), label="evaluation cell_id"),
        "true_delta": _finite_float(value.get("true_delta"), label="true_delta"),
        "source_component_id": _identifier(
            value.get("source_component_id"), label="source_component_id"
        ),
        "source_group_id_v2": _identifier(
            value.get("source_group_id_v2"), label="source_group_id_v2"
        ),
    }


def normalize_development_evaluation_rows(rows: Sequence[Mapping]) -> list[dict]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise BaselineSelectionError(
            "development evaluation artifact requires at least one row"
        )
    normalized = [_normalize_evaluation_row(row) for row in rows]
    row_ids = [row["row_id"] for row in normalized]
    if len(row_ids) != len(set(row_ids)):
        raise BaselineSelectionError("development evaluation row IDs must be unique")
    normalized.sort(key=lambda row: row["row_id"])
    return normalized


def development_evaluation_jsonl_bytes(rows: Sequence[Mapping]) -> bytes:
    normalized = normalize_development_evaluation_rows(rows)
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


def parse_development_evaluation_jsonl(raw: bytes) -> list[dict]:
    if not isinstance(raw, bytes) or not raw or not raw.endswith(b"\n"):
        raise BaselineSelectionError(
            "development evaluation JSONL must be non-empty and end with LF"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise BaselineSelectionError(
            "development evaluation JSONL is not UTF-8"
        ) from exc
    lines = text.splitlines()
    if not lines or any(not line for line in lines):
        raise BaselineSelectionError(
            "development evaluation JSONL cannot contain blank rows"
        )
    parsed = []
    for index, line in enumerate(lines, start=1):
        try:
            value = json.loads(line, object_pairs_hook=_unique_json_object)
        except (json.JSONDecodeError, ValueError) as exc:
            raise BaselineSelectionError(
                f"development evaluation row {index} is invalid: {exc}"
            ) from exc
        if not isinstance(value, Mapping):
            raise BaselineSelectionError(
                f"development evaluation row {index} must be an object"
            )
        parsed.append(_normalize_evaluation_row(value))
    normalized = normalize_development_evaluation_rows(parsed)
    if parsed != normalized:
        raise BaselineSelectionError(
            "development evaluation rows are not in canonical row_id order"
        )
    if development_evaluation_jsonl_bytes(normalized) != raw:
        raise BaselineSelectionError(
            "development evaluation JSONL bytes are not canonical"
        )
    return normalized


def write_development_evaluation_jsonl_exclusive(
    path: Path, rows: Sequence[Mapping]
) -> str:
    if "development" not in {part.lower() for part in Path(path).parts}:
        raise BaselineSelectionError(
            "development evaluation rows must be published beneath development"
        )
    raw = development_evaluation_jsonl_bytes(rows)
    try:
        return atomic_publish_bytes(
            Path(path), raw, replay_validator=parse_development_evaluation_jsonl
        )
    except ArtifactIOError as exc:
        raise BaselineSelectionError(str(exc)) from exc


def _read_prediction_rows(run_root: Path, reference: Mapping) -> list[dict]:
    normalized = _development_ref(reference, label="development prediction reference")
    try:
        raw = read_artifact_ref(Path(run_root), normalized)
        return parse_prediction_jsonl(raw)
    except ArtifactGraphError as exc:
        raise BaselineSelectionError(str(exc)) from exc


def _read_evaluation_rows(run_root: Path, reference: Mapping) -> tuple[list[dict], str]:
    normalized = _development_ref(
        reference, label="development evaluation-row reference"
    )
    try:
        raw = read_artifact_ref(Path(run_root), normalized)
    except ArtifactGraphError as exc:
        raise BaselineSelectionError(str(exc)) from exc
    return parse_development_evaluation_jsonl(raw), normalized["sha256"]


def _normalize_input_cell(
    run_root: Path,
    value: Mapping,
    *,
    label: str,
    expected_estimand_id: str,
) -> dict:
    _strict_keys(value, _INPUT_CELL_KEYS, label=label)
    cell_id = _identifier(value.get("cell_id"), label=f"{label}.cell_id")
    prediction_reference = _development_ref(
        value.get("prediction_jsonl_ref"),
        label=f"{label}.prediction_jsonl_ref",
    )
    evaluation_reference = _development_ref(
        value.get("development_evaluation_jsonl_ref"),
        label=f"{label}.development_evaluation_jsonl_ref",
    )
    predictions = _read_prediction_rows(run_root, prediction_reference)
    evaluation_rows, evaluation_sha = _read_evaluation_rows(
        run_root, evaluation_reference
    )
    prediction_ids = [row["row_id"] for row in predictions]
    evaluation_ids = [row["row_id"] for row in evaluation_rows]
    if prediction_ids != evaluation_ids:
        raise BaselineSelectionError(
            "development predictions and evaluation rows must have identical row order"
        )
    if any(row["estimand_id"] != expected_estimand_id for row in evaluation_rows):
        raise BaselineSelectionError(
            "development evaluation rows are bound to the wrong estimand"
        )
    if any(row["cell_id"] != cell_id for row in evaluation_rows):
        raise BaselineSelectionError(
            "development evaluation rows are bound to the wrong metric cell"
        )
    ordered_hash, set_hash = _row_id_hashes(predictions)
    true_delta = [row["true_delta"] for row in evaluation_rows]
    predicted_delta = [row["predicted_delta_mean"] for row in predictions]
    components = [row["source_component_id"] for row in evaluation_rows]
    pools = [row["source_group_id_v2"] for row in evaluation_rows]
    try:
        cell_metrics = component_balanced_cell_metrics(
            true_delta, predicted_delta, components
        )
        rank_metrics = within_pool_rank_metrics(
            true_delta,
            predicted_delta,
            pools,
            components,
            min_candidates=int(METRIC_POLICY["within_pool_min_candidates"]),
        )
    except MetricProtocolError as exc:
        raise BaselineSelectionError(f"development metric replay failed: {exc}") from exc
    sign = cell_metrics["sign_accuracy"]
    spearman = rank_metrics["spearman"]
    mae = cell_metrics["mae"]
    if sign is None or cell_metrics["sign_status"] != "ESTIMATED_COMPONENT_BALANCED":
        raise BaselineSelectionError("component-cell sign metric is incomplete")
    if (
        spearman is None
        or rank_metrics["spearman_status"]
        != "COMPLETE_ALL_STRUCTURALLY_ELIGIBLE_POOLS"
    ):
        raise BaselineSelectionError("within-pool Spearman metric is incomplete")
    registered_pools = sorted(set(pools))
    pool_counts = {pool: pools.count(pool) for pool in registered_pools}
    eligible_pools = sorted(
        pool
        for pool, count in pool_counts.items()
        if count >= int(METRIC_POLICY["within_pool_min_candidates"])
    )
    if len(eligible_pools) != rank_metrics["n_structurally_eligible_pools"]:
        raise BaselineSelectionError("within-pool eligibility replay is inconsistent")
    return {
        "cell_id": cell_id,
        "prediction_jsonl_ref": prediction_reference,
        "development_evaluation_jsonl_ref": evaluation_reference,
        "n_rows": len(predictions),
        "ordered_row_ids_sha256": ordered_hash,
        "row_set_sha256": set_hash,
        "evaluation_rows_sha256": evaluation_sha,
        "registered_source_group_ids_v2": registered_pools,
        "structurally_eligible_source_group_ids_v2": eligible_pools,
        "n_components": len(set(components)),
        "n_registered_pools": len(registered_pools),
        "n_structurally_eligible_pools": len(eligible_pools),
        "component_cell_sign_accuracy": float(sign),
        "component_cell_sign_status": cell_metrics["sign_status"],
        "within_pool_spearman": float(spearman),
        "within_pool_spearman_status": rank_metrics["spearman_status"],
        "component_cell_mae": float(mae),
        "component_cell_mae_status": "ESTIMATED_COMPONENT_BALANCED",
    }


def _normalize_output_cell(value: Mapping, *, label: str) -> dict:
    _strict_keys(value, _CELL_KEYS, label=label)
    normalized = dict(value)
    _identifier(normalized.get("cell_id"), label=f"{label}.cell_id")
    _development_ref(
        normalized.get("prediction_jsonl_ref"),
        label=f"{label}.prediction_jsonl_ref",
    )
    evaluation_reference = _development_ref(
        normalized.get("development_evaluation_jsonl_ref"),
        label=f"{label}.development_evaluation_jsonl_ref",
    )
    _positive_integer(normalized.get("n_rows"), label=f"{label}.n_rows")
    _digest(
        normalized.get("ordered_row_ids_sha256"),
        label=f"{label}.ordered_row_ids_sha256",
    )
    _digest(normalized.get("row_set_sha256"), label=f"{label}.row_set_sha256")
    evaluation_sha = _digest(
        normalized.get("evaluation_rows_sha256"),
        label=f"{label}.evaluation_rows_sha256",
    )
    if evaluation_reference["sha256"] != evaluation_sha:
        raise BaselineSelectionError(
            "evaluation-row reference SHA does not match cell binding"
        )
    registered_pool_ids = normalized.get("registered_source_group_ids_v2")
    eligible_pool_ids = normalized.get("structurally_eligible_source_group_ids_v2")
    for pool_ids, pool_label in (
        (registered_pool_ids, "registered_source_group_ids_v2"),
        (eligible_pool_ids, "structurally_eligible_source_group_ids_v2"),
    ):
        if (
            not isinstance(pool_ids, list)
            or not pool_ids
            or pool_ids != sorted(pool_ids)
            or len(pool_ids) != len(set(pool_ids))
        ):
            raise BaselineSelectionError(
                f"{pool_label} must be a non-empty sorted unique list"
            )
        for pool_id in pool_ids:
            _identifier(pool_id, label=pool_label)
    if not set(eligible_pool_ids).issubset(set(registered_pool_ids)):
        raise BaselineSelectionError(
            "eligible source groups must be a subset of registered source groups"
        )
    _positive_integer(
        normalized.get("n_components"), label=f"{label}.n_components"
    )
    registered = _positive_integer(
        normalized.get("n_registered_pools"),
        label=f"{label}.n_registered_pools",
    )
    eligible = _positive_integer(
        normalized.get("n_structurally_eligible_pools"),
        label=f"{label}.n_structurally_eligible_pools",
    )
    if eligible > registered:
        raise BaselineSelectionError(
            "n_structurally_eligible_pools cannot exceed n_registered_pools"
        )
    if registered != len(registered_pool_ids) or eligible != len(eligible_pool_ids):
        raise BaselineSelectionError("source-group lists disagree with pool counts")
    sign = _finite_float(
        normalized.get("component_cell_sign_accuracy"),
        label=f"{label}.component_cell_sign_accuracy",
    )
    spearman = _finite_float(
        normalized.get("within_pool_spearman"),
        label=f"{label}.within_pool_spearman",
    )
    mae = _finite_float(
        normalized.get("component_cell_mae"),
        label=f"{label}.component_cell_mae",
    )
    if not 0.0 <= sign <= 1.0:
        raise BaselineSelectionError("component_cell_sign_accuracy must be in [0, 1]")
    if not -1.0 <= spearman <= 1.0:
        raise BaselineSelectionError("within_pool_spearman must be in [-1, 1]")
    if mae < 0.0:
        raise BaselineSelectionError("component_cell_mae cannot be negative")
    if normalized.get("component_cell_sign_status") != "ESTIMATED_COMPONENT_BALANCED":
        raise BaselineSelectionError("component-cell sign metric is incomplete")
    if (
        normalized.get("within_pool_spearman_status")
        != "COMPLETE_ALL_STRUCTURALLY_ELIGIBLE_POOLS"
    ):
        raise BaselineSelectionError("within-pool Spearman metric is incomplete")
    if normalized.get("component_cell_mae_status") != "ESTIMATED_COMPONENT_BALANCED":
        raise BaselineSelectionError("component-cell MAE metric is incomplete")
    return normalized


def _normalize_estimands(
    values: Sequence[Mapping],
    *,
    run_root: Path | None,
) -> list[dict]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise BaselineSelectionError("estimands must be a non-empty sequence")
    normalized = []
    all_prediction_row_ids: set[str] = set()
    for estimand_index, estimand in enumerate(values):
        _strict_keys(estimand, _ESTIMAND_KEYS, label=f"estimand {estimand_index}")
        estimand_id = _identifier(
            estimand.get("estimand_id"), label=f"estimand {estimand_index}.estimand_id"
        )
        cells = estimand.get("cells")
        if isinstance(cells, (str, bytes)) or not isinstance(cells, Sequence) or not cells:
            raise BaselineSelectionError("every estimand requires at least one metric cell")
        normalized_cells = []
        for cell_index, cell in enumerate(cells):
            label = f"estimand {estimand_id} cell {cell_index}"
            if run_root is None:
                normalized_cell = _normalize_output_cell(cell, label=label)
            else:
                normalized_cell = _normalize_input_cell(
                    run_root,
                    cell,
                    label=label,
                    expected_estimand_id=estimand_id,
                )
                rows = _read_prediction_rows(
                    run_root, normalized_cell["prediction_jsonl_ref"]
                )
                row_ids = {row["row_id"] for row in rows}
                overlap = all_prediction_row_ids.intersection(row_ids)
                if overlap:
                    raise BaselineSelectionError(
                        "one development prediction row cannot appear in multiple cells"
                    )
                all_prediction_row_ids.update(row_ids)
            normalized_cells.append(normalized_cell)
        normalized_cells.sort(key=lambda item: item["cell_id"])
        cell_ids = [item["cell_id"] for item in normalized_cells]
        if len(cell_ids) != len(set(cell_ids)):
            raise BaselineSelectionError("cell IDs must be unique within an estimand")
        normalized.append({"estimand_id": estimand_id, "cells": normalized_cells})
    normalized.sort(key=lambda item: item["estimand_id"])
    estimand_ids = [item["estimand_id"] for item in normalized]
    if len(estimand_ids) != len(set(estimand_ids)):
        raise BaselineSelectionError("estimand IDs must be unique")
    return normalized


def _aggregate_metrics(estimands: Sequence[Mapping]) -> dict:
    cells = [cell for estimand in estimands for cell in estimand["cells"]]
    if not cells:
        raise BaselineSelectionError("aggregate metrics refuse empty cell evidence")
    denominator = float(len(cells))
    return {
        "n_estimands": len(estimands),
        "n_cells": len(cells),
        "n_rows": sum(int(cell["n_rows"]) for cell in cells),
        "macro_component_cell_sign_accuracy": math.fsum(
            float(cell["component_cell_sign_accuracy"]) for cell in cells
        ) / denominator,
        "macro_within_pool_spearman": math.fsum(
            float(cell["within_pool_spearman"]) for cell in cells
        ) / denominator,
        "macro_component_cell_mae": math.fsum(
            float(cell["component_cell_mae"]) for cell in cells
        ) / denominator,
        "complete": True,
    }


def _assert_predictor_evidence_binding(
    provenance: Mapping,
    estimands: Sequence[Mapping],
) -> None:
    """Bind every evidence prediction to an exactly replayed predictor unit."""

    provenance_estimands = predictor_layout(provenance)
    expected = []
    for estimand in estimands:
        cells = []
        for cell in estimand["cells"]:
            cells.append({
                "cell_id": cell["cell_id"],
                "prediction_jsonl_ref": cell["prediction_jsonl_ref"],
                "development_n_rows": cell["n_rows"],
                "development_ordered_row_ids_sha256": cell[
                    "ordered_row_ids_sha256"
                ],
                "development_row_set_sha256": cell["row_set_sha256"],
            })
        expected.append({"estimand_id": estimand["estimand_id"], "cells": cells})
    if provenance_estimands != expected:
        raise BaselineSelectionError(
            "predictor provenance does not bind every exact evidence prediction row"
        )


def build_development_baseline_evidence(
    run_root: Path,
    *,
    run_start_sha256: str,
    activation_manifest_sha256: str,
    train_development_outcomes_sha256: str,
    non_data_authority_hashes: Mapping,
    method_id: str,
    information_set_fingerprint: str,
    predictor_provenance_ref: Mapping,
    predictor_verifiers: Mapping[str, PredictorVerifier] | None,
    estimands: Sequence[Mapping],
) -> dict:
    """Build evidence only after exact predictor and metric recomputation."""

    normalized_estimands = _normalize_estimands(estimands, run_root=Path(run_root))
    provenance_reference = _development_ref(
        predictor_provenance_ref, label="predictor provenance reference"
    )
    try:
        provenance = replay_predictor_provenance(
            Path(run_root),
            provenance_reference,
            verifiers=predictor_verifiers,
            expected_run_start_sha256=run_start_sha256,
            expected_activation_manifest_sha256=activation_manifest_sha256,
            expected_train_development_outcomes_sha256=(
                train_development_outcomes_sha256
            ),
            expected_non_data_authority_hashes=non_data_authority_hashes,
            expected_method_id=method_id,
            expected_information_set_fingerprint=information_set_fingerprint,
        )
    except PredictorProvenanceError as exc:
        raise BaselineSelectionError(f"predictor provenance replay failed: {exc}") from exc
    _assert_predictor_evidence_binding(provenance, normalized_estimands)
    payload = {
        "schema_version": DEVELOPMENT_EVIDENCE_SCHEMA,
        "artifact_role": EVIDENCE_ROLE,
        "status": "COMPLETE_DEVELOPMENT_EVIDENCE",
        "run_start_sha256": _digest(run_start_sha256, label="run_start_sha256"),
        "activation_manifest_sha256": _digest(
            activation_manifest_sha256, label="activation_manifest_sha256"
        ),
        "train_development_outcomes_sha256": _digest(
            train_development_outcomes_sha256,
            label="train_development_outcomes_sha256",
        ),
        "non_data_authority_hashes": _non_data_authority_hashes(
            non_data_authority_hashes
        ),
        "method_id": _method_id(method_id),
        "partition": DEVELOPMENT_PARTITION,
        "information_set": PRIMARY_INFORMATION_SET,
        "information_set_fingerprint": _digest(
            information_set_fingerprint, label="information_set_fingerprint"
        ),
        "predictor_provenance_ref": provenance_reference,
        "metric_policy": dict(METRIC_POLICY),
        "metric_policy_sha256": _canonical_sha256(METRIC_POLICY),
        "estimands": normalized_estimands,
        "aggregate_development_metrics": _aggregate_metrics(normalized_estimands),
        "ordinary_only": True,
        "test_access": False,
        "sealed_access": False,
        "downstream_unlock": False,
    }
    return validate_development_baseline_evidence(payload)


def validate_development_baseline_evidence(payload: Mapping) -> dict:
    """Validate schema and independently recompute all macro development metrics."""

    _strict_keys(payload, _EVIDENCE_KEYS, label="development baseline evidence")
    if payload.get("schema_version") != DEVELOPMENT_EVIDENCE_SCHEMA:
        raise BaselineSelectionError("unsupported development evidence schema")
    if payload.get("artifact_role") != EVIDENCE_ROLE:
        raise BaselineSelectionError("development evidence artifact role is invalid")
    if payload.get("status") != "COMPLETE_DEVELOPMENT_EVIDENCE":
        raise BaselineSelectionError("development evidence must be complete")
    _digest(payload.get("run_start_sha256"), label="run_start_sha256")
    _digest(
        payload.get("activation_manifest_sha256"),
        label="activation_manifest_sha256",
    )
    _digest(
        payload.get("train_development_outcomes_sha256"),
        label="train_development_outcomes_sha256",
    )
    _non_data_authority_hashes(payload.get("non_data_authority_hashes"))
    _method_id(payload.get("method_id"))
    if payload.get("partition") != DEVELOPMENT_PARTITION:
        raise BaselineSelectionError("evidence partition must be ordinary development")
    if payload.get("information_set") != PRIMARY_INFORMATION_SET:
        raise BaselineSelectionError("evidence information set is not eligible")
    _digest(
        payload.get("information_set_fingerprint"),
        label="information_set_fingerprint",
    )
    _development_ref(
        payload.get("predictor_provenance_ref"),
        label="predictor provenance reference",
    )
    _metric_policy(payload.get("metric_policy"))
    if payload.get("metric_policy_sha256") != _canonical_sha256(METRIC_POLICY):
        raise BaselineSelectionError("metric policy SHA-256 binding is invalid")
    estimands = _normalize_estimands(payload.get("estimands"), run_root=None)
    if estimands != payload.get("estimands"):
        raise BaselineSelectionError("estimand/cell evidence is not in canonical order")
    aggregate = payload.get("aggregate_development_metrics")
    _strict_keys(aggregate, _AGGREGATE_KEYS, label="aggregate_development_metrics")
    expected_aggregate = _aggregate_metrics(estimands)
    if dict(aggregate) != expected_aggregate:
        raise BaselineSelectionError(
            "aggregate development metrics do not match complete cell evidence"
        )
    _boundaries(payload, label="development baseline evidence")
    return dict(payload)


def canonical_development_baseline_evidence_bytes(payload: Mapping) -> bytes:
    try:
        return canonical_json_bytes(validate_development_baseline_evidence(payload))
    except ArtifactIOError as exc:
        raise BaselineSelectionError(str(exc)) from exc


def _assert_output_within_run_root(run_root: Path, path: Path) -> None:
    root = Path(run_root)
    try:
        resolved_root = root.resolve(strict=True)
        candidate = Path(path).resolve(strict=False)
    except OSError as exc:
        raise BaselineSelectionError(f"artifact output path cannot be resolved: {exc}") from exc
    if not resolved_root.is_dir() or not candidate.is_relative_to(resolved_root):
        raise BaselineSelectionError("artifact output must remain beneath the run root")


def _recompute_evidence_estimands(run_root: Path, payload: Mapping) -> list[dict]:
    replayed_estimands = []
    all_rows: set[str] = set()
    for estimand in payload["estimands"]:
        cells = []
        for cell in estimand["cells"]:
            recomputed = _normalize_input_cell(
                Path(run_root),
                {key: cell[key] for key in _INPUT_CELL_KEYS},
                label=f"replay {estimand['estimand_id']} {cell['cell_id']}",
                expected_estimand_id=estimand["estimand_id"],
            )
            if recomputed != cell:
                raise BaselineSelectionError(
                    "development cell metrics do not replay exact truth and predictions"
                )
            rows = _read_prediction_rows(run_root, cell["prediction_jsonl_ref"])
            row_ids = {row["row_id"] for row in rows}
            if all_rows.intersection(row_ids):
                raise BaselineSelectionError(
                    "one development prediction row appears in multiple metric cells"
                )
            all_rows.update(row_ids)
            cells.append(dict(cell))
        replayed_estimands.append(
            {"estimand_id": estimand["estimand_id"], "cells": cells}
        )
    return replayed_estimands


def _replay_bound_predictor(
    run_root: Path,
    payload: Mapping,
    predictor_verifiers: Mapping[str, PredictorVerifier] | None,
) -> dict:
    try:
        provenance = replay_predictor_provenance(
            Path(run_root),
            payload["predictor_provenance_ref"],
            verifiers=predictor_verifiers,
            expected_run_start_sha256=payload["run_start_sha256"],
            expected_activation_manifest_sha256=payload[
                "activation_manifest_sha256"
            ],
            expected_train_development_outcomes_sha256=payload[
                "train_development_outcomes_sha256"
            ],
            expected_non_data_authority_hashes=payload[
                "non_data_authority_hashes"
            ],
            expected_method_id=payload["method_id"],
            expected_information_set_fingerprint=payload[
                "information_set_fingerprint"
            ],
        )
    except PredictorProvenanceError as exc:
        raise BaselineSelectionError(f"predictor provenance replay failed: {exc}") from exc
    _assert_predictor_evidence_binding(provenance, payload["estimands"])
    return provenance


def write_development_baseline_evidence_exclusive(
    run_root: Path,
    path: Path,
    payload: Mapping,
    *,
    predictor_verifiers: Mapping[str, PredictorVerifier] | None,
) -> str:
    """Recompute exact dev metrics, then atomically publish without overwrite."""

    _assert_output_within_run_root(run_root, path)
    if "development" not in {part.lower() for part in Path(path).parts}:
        raise BaselineSelectionError(
            "development evidence must be published beneath a development namespace"
        )
    validated = validate_development_baseline_evidence(payload)
    _replay_bound_predictor(Path(run_root), validated, predictor_verifiers)
    if _recompute_evidence_estimands(Path(run_root), validated) != validated["estimands"]:
        raise BaselineSelectionError("development evidence changed during exact replay")
    try:
        return atomic_write_json_exclusive(
            Path(path), validated, validator=validate_development_baseline_evidence
        )
    except ArtifactIOError as exc:
        raise BaselineSelectionError(str(exc)) from exc


def _unique_json_object(pairs) -> dict:
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
        raise BaselineSelectionError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise BaselineSelectionError(f"{label} must be a JSON object")
    return payload


def replay_development_baseline_evidence(
    run_root: Path,
    evidence_ref: Mapping,
    *,
    predictor_verifiers: Mapping[str, PredictorVerifier] | None,
    expected_run_start_sha256: str | None = None,
    expected_activation_manifest_sha256: str | None = None,
    expected_train_development_outcomes_sha256: str | None = None,
    expected_non_data_authority_hashes: Mapping | None = None,
    expected_method_id: str | None = None,
    expected_information_set_fingerprint: str | None = None,
) -> dict:
    """Replay exact evidence, truth rows, predictions, and all cell metrics."""

    reference = _development_ref(
        evidence_ref, label="development method evidence reference"
    )
    try:
        raw = read_artifact_ref(Path(run_root), reference)
    except ArtifactGraphError as exc:
        raise BaselineSelectionError(str(exc)) from exc
    payload = validate_development_baseline_evidence(
        _parse_json_object(raw, label="development baseline evidence")
    )
    if canonical_development_baseline_evidence_bytes(payload) != raw:
        raise BaselineSelectionError("development evidence bytes are not canonical")

    expected_pairs = (
        ("run_start_sha256", expected_run_start_sha256),
        ("activation_manifest_sha256", expected_activation_manifest_sha256),
        ("method_id", expected_method_id),
        ("information_set_fingerprint", expected_information_set_fingerprint),
    )
    for field, expected in expected_pairs:
        if expected is not None and payload[field] != expected:
            raise BaselineSelectionError(f"development evidence {field} binding mismatch")
    if expected_train_development_outcomes_sha256 is not None:
        if payload["train_development_outcomes_sha256"] != _digest(
            expected_train_development_outcomes_sha256,
            label="expected_train_development_outcomes_sha256",
        ):
            raise BaselineSelectionError(
                "development evidence train/development outcome authority mismatch"
            )
    if expected_non_data_authority_hashes is not None:
        if payload["non_data_authority_hashes"] != _non_data_authority_hashes(
            expected_non_data_authority_hashes
        ):
            raise BaselineSelectionError(
                "development evidence non-data authority binding mismatch"
            )

    replayed_estimands = _recompute_evidence_estimands(Path(run_root), payload)
    if replayed_estimands != payload["estimands"]:
        raise BaselineSelectionError("development estimand replay changed evidence")
    _replay_bound_predictor(Path(run_root), payload, predictor_verifiers)
    return payload


def _eligible_method_order(values: Sequence[str]) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise BaselineSelectionError("eligible_method_order must be a sequence")
    normalized = [
        _method_id(value, label=f"eligible_method_order[{index}]")
        for index, value in enumerate(values)
    ]
    if not normalized:
        raise BaselineSelectionError("eligible_method_order cannot be empty")
    if len(normalized) != len(set(normalized)):
        raise BaselineSelectionError("eligible_method_order contains duplicates")
    return normalized


def _required_method_ids() -> list[str]:
    """Return the canonical minimum method set required for a strongest claim."""

    return sorted(MINIMUM_REQUIRED_METHOD_IDS)


def _development_layout(evidence: Mapping) -> list[dict]:
    return [
        {
            "estimand_id": estimand["estimand_id"],
            "cells": [
                {
                    "cell_id": cell["cell_id"],
                    "n_rows": cell["n_rows"],
                    "ordered_row_ids_sha256": cell["ordered_row_ids_sha256"],
                    "row_set_sha256": cell["row_set_sha256"],
                    "evaluation_rows_sha256": cell["evaluation_rows_sha256"],
                    "registered_source_group_ids_v2": cell[
                        "registered_source_group_ids_v2"
                    ],
                    "structurally_eligible_source_group_ids_v2": cell[
                        "structurally_eligible_source_group_ids_v2"
                    ],
                    "n_components": cell["n_components"],
                    "n_registered_pools": cell["n_registered_pools"],
                    "n_structurally_eligible_pools": cell[
                        "n_structurally_eligible_pools"
                    ],
                }
                for cell in estimand["cells"]
            ],
        }
        for estimand in evidence["estimands"]
    ]


def _validate_layout(values: Sequence[Mapping]) -> list[dict]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise BaselineSelectionError("development_layout must be a non-empty sequence")
    normalized = []
    for estimand_index, estimand in enumerate(values):
        _strict_keys(
            estimand,
            _LAYOUT_ESTIMAND_KEYS,
            label=f"development layout estimand {estimand_index}",
        )
        estimand_id = _identifier(
            estimand.get("estimand_id"),
            label=f"development layout estimand {estimand_index}.estimand_id",
        )
        cells = estimand.get("cells")
        if isinstance(cells, (str, bytes)) or not isinstance(cells, Sequence) or not cells:
            raise BaselineSelectionError("every layout estimand requires cells")
        normalized_cells = []
        for cell_index, cell in enumerate(cells):
            _strict_keys(
                cell,
                _LAYOUT_CELL_KEYS,
                label=f"development layout cell {cell_index}",
            )
            normalized_cells.append({
                "cell_id": _identifier(cell.get("cell_id"), label="layout cell_id"),
                "n_rows": _positive_integer(cell.get("n_rows"), label="layout n_rows"),
                "ordered_row_ids_sha256": _digest(
                    cell.get("ordered_row_ids_sha256"), label="layout ordered row hash"
                ),
                "row_set_sha256": _digest(
                    cell.get("row_set_sha256"), label="layout row-set hash"
                ),
                "evaluation_rows_sha256": _digest(
                    cell.get("evaluation_rows_sha256"),
                    label="layout evaluation-row hash",
                ),
                "registered_source_group_ids_v2": list(
                    cell.get("registered_source_group_ids_v2") or []
                ),
                "structurally_eligible_source_group_ids_v2": list(
                    cell.get("structurally_eligible_source_group_ids_v2") or []
                ),
                "n_components": _positive_integer(
                    cell.get("n_components"), label="layout n_components"
                ),
                "n_registered_pools": _positive_integer(
                    cell.get("n_registered_pools"),
                    label="layout n_registered_pools",
                ),
                "n_structurally_eligible_pools": _positive_integer(
                    cell.get("n_structurally_eligible_pools"),
                    label="layout n_structurally_eligible_pools",
                ),
            })
            if (
                normalized_cells[-1]["n_structurally_eligible_pools"]
                > normalized_cells[-1]["n_registered_pools"]
            ):
                raise BaselineSelectionError(
                    "layout eligible pool count exceeds registered pool count"
                )
            registered_pools = normalized_cells[-1][
                "registered_source_group_ids_v2"
            ]
            eligible_pools = normalized_cells[-1][
                "structurally_eligible_source_group_ids_v2"
            ]
            for pool_ids, pool_label in (
                (registered_pools, "layout registered source groups"),
                (eligible_pools, "layout eligible source groups"),
            ):
                if (
                    not pool_ids
                    or pool_ids != sorted(pool_ids)
                    or len(pool_ids) != len(set(pool_ids))
                ):
                    raise BaselineSelectionError(
                        f"{pool_label} must be non-empty, sorted, and unique"
                    )
                for pool_id in pool_ids:
                    _identifier(pool_id, label=pool_label)
            if not set(eligible_pools).issubset(set(registered_pools)):
                raise BaselineSelectionError(
                    "layout eligible source groups are not registered"
                )
            if (
                len(registered_pools) != normalized_cells[-1]["n_registered_pools"]
                or len(eligible_pools)
                != normalized_cells[-1]["n_structurally_eligible_pools"]
            ):
                raise BaselineSelectionError(
                    "layout source-group lists disagree with pool counts"
                )
        if normalized_cells != sorted(normalized_cells, key=lambda item: item["cell_id"]):
            raise BaselineSelectionError("development layout cells are not canonical")
        if len({cell["cell_id"] for cell in normalized_cells}) != len(normalized_cells):
            raise BaselineSelectionError("development layout contains duplicate cells")
        normalized.append({"estimand_id": estimand_id, "cells": normalized_cells})
    if normalized != sorted(normalized, key=lambda item: item["estimand_id"]):
        raise BaselineSelectionError("development layout estimands are not canonical")
    if len({item["estimand_id"] for item in normalized}) != len(normalized):
        raise BaselineSelectionError("development layout contains duplicate estimands")
    return normalized


def _required_estimand_cells(values: Sequence[Mapping]) -> list[dict]:
    """Normalize the label-blind layout derived from frozen registry and split."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise BaselineSelectionError(
            "required_estimand_cells must be a non-empty frozen sequence"
        )
    normalized = []
    for index, value in enumerate(values):
        _strict_keys(
            value,
            _REQUIRED_ESTIMAND_KEYS,
            label=f"required estimand {index}",
        )
        estimand_id = _identifier(
            value.get("estimand_id"), label=f"required estimand {index}.estimand_id"
        )
        raw_cells = value.get("cells")
        if (
            isinstance(raw_cells, (str, bytes))
            or not isinstance(raw_cells, Sequence)
            or not raw_cells
        ):
            raise BaselineSelectionError("every required estimand needs cells")
        cells = []
        for cell_index, cell in enumerate(raw_cells):
            _strict_keys(
                cell,
                _REQUIRED_CELL_KEYS,
                label=f"required {estimand_id} cell {cell_index}",
            )
            cell_id = _identifier(
                cell.get("cell_id"), label=f"required {estimand_id} cell_id"
            )
            pool_ids = cell.get("registered_source_group_ids_v2")
            if (
                not isinstance(pool_ids, list)
                or not pool_ids
                or pool_ids != sorted(pool_ids)
                or len(pool_ids) != len(set(pool_ids))
            ):
                raise BaselineSelectionError(
                    "required registered source groups must be sorted and unique"
                )
            for pool_id in pool_ids:
                _identifier(pool_id, label="required source_group_id_v2")
            cells.append({
                "cell_id": cell_id,
                "registered_source_group_ids_v2": list(pool_ids),
            })
        if cells != sorted(cells, key=lambda item: item["cell_id"]) or len(
            {cell["cell_id"] for cell in cells}
        ) != len(cells):
            raise BaselineSelectionError(
                "required cells must be sorted and unique"
            )
        normalized.append({"estimand_id": estimand_id, "cells": cells})
    if normalized != sorted(normalized, key=lambda item: item["estimand_id"]):
        raise BaselineSelectionError("required estimands must be in canonical order")
    if len({item["estimand_id"] for item in normalized}) != len(normalized):
        raise BaselineSelectionError("required estimand IDs must be unique")
    return normalized


def _layout_identity(layout: Sequence[Mapping]) -> list[dict]:
    return [
        {
            "estimand_id": estimand["estimand_id"],
            "cells": [
                {
                    "cell_id": cell["cell_id"],
                    "registered_source_group_ids_v2": cell[
                        "registered_source_group_ids_v2"
                    ],
                }
                for cell in estimand["cells"]
            ],
        }
        for estimand in layout
    ]


def _selection_metrics(value: Mapping, *, label: str) -> dict[str, float]:
    _strict_keys(value, _SELECTION_METRIC_KEYS, label=label)
    sign = _finite_float(
        value.get("macro_component_cell_sign_accuracy"),
        label=f"{label}.macro_component_cell_sign_accuracy",
    )
    spearman = _finite_float(
        value.get("macro_within_pool_spearman"),
        label=f"{label}.macro_within_pool_spearman",
    )
    mae = _finite_float(
        value.get("macro_component_cell_mae"),
        label=f"{label}.macro_component_cell_mae",
    )
    if not 0.0 <= sign <= 1.0:
        raise BaselineSelectionError("macro sign accuracy must be in [0, 1]")
    if not -1.0 <= spearman <= 1.0:
        raise BaselineSelectionError("macro within-pool Spearman must be in [-1, 1]")
    if mae < 0.0:
        raise BaselineSelectionError("macro MAE cannot be negative")
    return {
        "macro_component_cell_sign_accuracy": sign,
        "macro_within_pool_spearman": spearman,
        "macro_component_cell_mae": mae,
    }


def _record(
    method_id: str,
    evidence_ref: Mapping,
    aggregate: Mapping,
    method_order_index: int,
) -> dict:
    metrics = _selection_metrics(
        {
            "macro_component_cell_sign_accuracy": aggregate[
                "macro_component_cell_sign_accuracy"
            ],
            "macro_within_pool_spearman": aggregate[
                "macro_within_pool_spearman"
            ],
            "macro_component_cell_mae": aggregate["macro_component_cell_mae"],
        },
        label=f"{method_id} development metrics",
    )
    gate_met = metrics["macro_component_cell_sign_accuracy"] >= SIGN_ACCURACY_GATE
    return {
        "method_id": method_id,
        "development_evidence_ref": _development_ref(
            evidence_ref, label=f"{method_id} development evidence reference"
        ),
        "method_order_index": method_order_index,
        "development_metrics": metrics,
        "sign_gate_met": gate_met,
        "lexicographic_selection_tuple": [
            metrics["macro_component_cell_sign_accuracy"],
            metrics["macro_within_pool_spearman"],
            -metrics["macro_component_cell_mae"],
            -float(method_order_index),
        ],
    }


def _build_selection_payload(
    *,
    run_start_sha256: str,
    activation_manifest_sha256: str,
    train_development_outcomes_sha256: str,
    non_data_authority_hashes: Mapping,
    eligible_method_order: Sequence[str],
    development_evidence_refs: Mapping[str, Mapping],
    required_estimand_cells: Sequence[Mapping],
    evidences: Mapping[str, Mapping],
) -> dict:
    order = _eligible_method_order(eligible_method_order)
    if not isinstance(development_evidence_refs, Mapping):
        raise BaselineSelectionError("development_evidence_refs must be an object")
    evidence_methods = set(development_evidence_refs)
    unknown_evidence_methods = sorted(evidence_methods - set(order))
    if unknown_evidence_methods:
        raise BaselineSelectionError(
            "development evidence refs contain methods outside the frozen eligible "
            f"order: {unknown_evidence_methods}"
        )
    if not evidence_methods:
        raise BaselineSelectionError(
            "at least one exactly replayed development evidence artifact is required "
            "to select a comparator"
        )
    if not isinstance(evidences, Mapping) or set(evidences) != evidence_methods:
        raise BaselineSelectionError(
            "replayed development evidence must cover exactly every referenced method"
        )
    evidenced_order = [method for method in order if method in evidence_methods]
    refs = {
        method: _development_ref(
            development_evidence_refs[method],
            label=f"{method} development evidence reference",
        )
        for method in evidenced_order
    }
    development_data_sha = _digest(
        train_development_outcomes_sha256,
        label="train_development_outcomes_sha256",
    )
    normalized_authorities = _non_data_authority_hashes(
        non_data_authority_hashes
    )
    run_sha = _digest(run_start_sha256, label="run_start_sha256")
    activation_sha = _digest(
        activation_manifest_sha256, label="activation_manifest_sha256"
    )
    fingerprints = set()
    layouts = []
    records = []
    for index, method in enumerate(order):
        if method not in evidence_methods:
            continue
        evidence = validate_development_baseline_evidence(evidences[method])
        if evidence["method_id"] != method:
            raise BaselineSelectionError("development evidence is assigned to wrong method")
        if evidence["run_start_sha256"] != run_sha:
            raise BaselineSelectionError("development evidence run-start binding mismatch")
        if evidence["activation_manifest_sha256"] != activation_sha:
            raise BaselineSelectionError("development evidence activation binding mismatch")
        if evidence["train_development_outcomes_sha256"] != development_data_sha:
            raise BaselineSelectionError(
                "development evidence outcome-authority binding mismatch"
            )
        if evidence["non_data_authority_hashes"] != normalized_authorities:
            raise BaselineSelectionError(
                "development evidence non-data authority binding mismatch"
            )
        fingerprints.add(evidence["information_set_fingerprint"])
        layouts.append(_development_layout(evidence))
        records.append(
            _record(method, refs[method], evidence["aggregate_development_metrics"], index)
        )
    if len(fingerprints) != 1:
        raise BaselineSelectionError(
            "eligible methods do not share one information-set fingerprint"
        )
    first_layout = layouts[0]
    if any(layout != first_layout for layout in layouts[1:]):
        raise BaselineSelectionError(
            "eligible methods do not share complete estimand/cell rows and order"
        )
    required_layout = _required_estimand_cells(required_estimand_cells)
    if _layout_identity(first_layout) != required_layout:
        raise BaselineSelectionError(
            "development evidence does not cover every frozen estimand/cell"
        )
    selected = max(
        records, key=lambda item: tuple(item["lexicographic_selection_tuple"])
    )
    eligible_ids = sorted(order)
    required_ids = _required_method_ids()
    missing_required_methods = sorted(set(required_ids) - set(order))
    missing_required_evidence = sorted(set(required_ids) - evidence_methods)
    missing_eligible_evidence = [
        method for method in order if method not in evidence_methods
    ]
    strongest_claim = not missing_required_methods and not missing_eligible_evidence
    baseline_sign_gate_met = bool(selected["sign_gate_met"])
    can_satisfy_pretest_freeze = bool(
        strongest_claim and baseline_sign_gate_met
    )
    status = (
        QUALIFIED_STATUS if can_satisfy_pretest_freeze else HOLD_STATUS
    )
    payload = {
        "schema_version": SELECTION_SCHEMA,
        "artifact_role": SELECTION_ROLE,
        "status": status,
        "run_start_sha256": run_sha,
        "activation_manifest_sha256": activation_sha,
        "train_development_outcomes_sha256": development_data_sha,
        "non_data_authority_hashes": normalized_authorities,
        "partition": DEVELOPMENT_PARTITION,
        "information_set": PRIMARY_INFORMATION_SET,
        "information_set_fingerprint": next(iter(fingerprints)),
        "required_method_ids": required_ids,
        "required_method_set_sha256": _canonical_sha256(required_ids),
        "missing_required_method_ids": missing_required_methods,
        "missing_required_evidence_method_ids": missing_required_evidence,
        "missing_eligible_evidence_method_ids": missing_eligible_evidence,
        "eligible_method_ids": eligible_ids,
        "eligible_method_order": order,
        "eligible_method_set_sha256": _canonical_sha256(eligible_ids),
        "eligible_method_order_sha256": _canonical_sha256(order),
        "development_evidence_refs": refs,
        "required_estimand_cells": required_layout,
        "required_estimand_cells_sha256": _canonical_sha256(required_layout),
        "development_layout": first_layout,
        "development_layout_sha256": _canonical_sha256(first_layout),
        "metric_policy": dict(METRIC_POLICY),
        "metric_policy_sha256": _canonical_sha256(METRIC_POLICY),
        "selection_tuple_fields": [
            "macro_component_cell_sign_accuracy",
            "macro_within_pool_spearman",
            "negative_macro_component_cell_mae",
            "negative_frozen_method_order_index",
        ],
        "method_selection_records": records,
        "selected_method_id": selected["method_id"],
        "selected_development_metrics": dict(selected["development_metrics"]),
        "comparator_selected": True,
        "baseline_sign_gate_met": baseline_sign_gate_met,
        "strongest_same_information_claim": strongest_claim,
        "can_satisfy_pretest_freeze": can_satisfy_pretest_freeze,
        "ordinary_only": True,
        "test_access": False,
        "sealed_access": False,
        "downstream_unlock": False,
    }
    return validate_development_baseline_selection(payload)


def build_development_baseline_selection(
    run_root: Path,
    *,
    run_start_sha256: str,
    activation_manifest_sha256: str,
    train_development_outcomes_sha256: str,
    non_data_authority_hashes: Mapping,
    eligible_method_order: Sequence[str],
    development_evidence_refs: Mapping[str, Mapping],
    required_estimand_cells: Sequence[Mapping],
    predictor_verifiers: Mapping[str, PredictorVerifier] | None,
) -> dict:
    """Replay all eligible development evidence and select one comparator."""

    order = _eligible_method_order(eligible_method_order)
    if not isinstance(development_evidence_refs, Mapping):
        raise BaselineSelectionError("development_evidence_refs must be an object")
    evidence_methods = set(development_evidence_refs)
    unknown_evidence_methods = sorted(evidence_methods - set(order))
    if unknown_evidence_methods:
        raise BaselineSelectionError(
            "development evidence refs contain methods outside the frozen eligible "
            f"order: {unknown_evidence_methods}"
        )
    if not evidence_methods:
        raise BaselineSelectionError(
            "at least one development evidence artifact is required to select a "
            "comparator"
        )
    evidences = {
        method: replay_development_baseline_evidence(
            run_root,
            development_evidence_refs[method],
            predictor_verifiers=predictor_verifiers,
            expected_run_start_sha256=run_start_sha256,
            expected_activation_manifest_sha256=activation_manifest_sha256,
            expected_train_development_outcomes_sha256=(
                train_development_outcomes_sha256
            ),
            expected_non_data_authority_hashes=non_data_authority_hashes,
            expected_method_id=method,
        )
        for method in order
        if method in evidence_methods
    }
    return _build_selection_payload(
        run_start_sha256=run_start_sha256,
        activation_manifest_sha256=activation_manifest_sha256,
        train_development_outcomes_sha256=train_development_outcomes_sha256,
        non_data_authority_hashes=non_data_authority_hashes,
        eligible_method_order=order,
        development_evidence_refs=development_evidence_refs,
        required_estimand_cells=required_estimand_cells,
        evidences=evidences,
    )


def validate_development_baseline_selection(payload: Mapping) -> dict:
    """Validate strict schema and recompute the frozen selection tuple."""

    if not isinstance(payload, Mapping):
        raise BaselineSelectionError("development baseline selection must be an object")
    if payload.get("schema_version") != SELECTION_SCHEMA:
        raise BaselineSelectionError("unsupported baseline selection schema")
    _strict_keys(payload, _SELECTION_KEYS, label="development baseline selection")
    if payload.get("artifact_role") != SELECTION_ROLE:
        raise BaselineSelectionError("baseline selection artifact role is invalid")
    _digest(payload.get("run_start_sha256"), label="run_start_sha256")
    _digest(
        payload.get("activation_manifest_sha256"), label="activation_manifest_sha256"
    )
    _digest(
        payload.get("train_development_outcomes_sha256"),
        label="train_development_outcomes_sha256",
    )
    _non_data_authority_hashes(payload.get("non_data_authority_hashes"))
    if payload.get("partition") != DEVELOPMENT_PARTITION:
        raise BaselineSelectionError("baseline selection is not development-only")
    if payload.get("information_set") != PRIMARY_INFORMATION_SET:
        raise BaselineSelectionError("baseline selection information set is invalid")
    _digest(
        payload.get("information_set_fingerprint"),
        label="information_set_fingerprint",
    )
    order = _eligible_method_order(payload.get("eligible_method_order"))
    required_ids = _required_method_ids()
    if payload.get("required_method_ids") != required_ids:
        raise BaselineSelectionError("required strong-baseline method set changed")
    if payload.get("required_method_set_sha256") != _canonical_sha256(required_ids):
        raise BaselineSelectionError("required method-set SHA-256 is invalid")
    eligible_ids = payload.get("eligible_method_ids")
    if eligible_ids != sorted(order):
        raise BaselineSelectionError("eligible_method_ids do not bind the frozen set")
    if payload.get("eligible_method_set_sha256") != _canonical_sha256(eligible_ids):
        raise BaselineSelectionError("eligible method-set SHA-256 is invalid")
    if payload.get("eligible_method_order_sha256") != _canonical_sha256(order):
        raise BaselineSelectionError("eligible method-order SHA-256 is invalid")
    refs = payload.get("development_evidence_refs")
    if not isinstance(refs, Mapping) or not refs:
        raise BaselineSelectionError(
            "development_evidence_refs must contain at least one replayable method"
        )
    evidence_methods = set(refs)
    unknown_evidence_methods = sorted(evidence_methods - set(order))
    if unknown_evidence_methods:
        raise BaselineSelectionError(
            "development evidence refs contain methods outside the frozen eligible "
            f"order: {unknown_evidence_methods}"
        )
    evidenced_order = [method for method in order if method in evidence_methods]
    normalized_refs = {
        method: _development_ref(
            refs[method], label=f"{method} development evidence reference"
        )
        for method in evidenced_order
    }
    if dict(refs) != normalized_refs:
        raise BaselineSelectionError("development evidence refs are not canonical")
    missing_required_methods = sorted(set(required_ids) - set(order))
    missing_required_evidence = sorted(set(required_ids) - evidence_methods)
    missing_eligible_evidence = [
        method for method in order if method not in evidence_methods
    ]
    if payload.get("missing_required_method_ids") != missing_required_methods:
        raise BaselineSelectionError("missing required-method declaration is invalid")
    if (
        payload.get("missing_required_evidence_method_ids")
        != missing_required_evidence
    ):
        raise BaselineSelectionError(
            "missing required-evidence declaration is invalid"
        )
    if (
        payload.get("missing_eligible_evidence_method_ids")
        != missing_eligible_evidence
    ):
        raise BaselineSelectionError(
            "missing eligible-evidence declaration is invalid"
        )
    required_layout = _required_estimand_cells(payload.get("required_estimand_cells"))
    if payload.get("required_estimand_cells_sha256") != _canonical_sha256(
        required_layout
    ):
        raise BaselineSelectionError("required estimand/cell SHA-256 is invalid")
    layout = _validate_layout(payload.get("development_layout"))
    if payload.get("development_layout_sha256") != _canonical_sha256(layout):
        raise BaselineSelectionError("development layout SHA-256 is invalid")
    if _layout_identity(layout) != required_layout:
        raise BaselineSelectionError(
            "development layout does not cover every required estimand/cell"
        )
    _metric_policy(payload.get("metric_policy"))
    if payload.get("metric_policy_sha256") != _canonical_sha256(METRIC_POLICY):
        raise BaselineSelectionError("selection metric-policy SHA-256 is invalid")
    if payload.get("selection_tuple_fields") != [
        "macro_component_cell_sign_accuracy",
        "macro_within_pool_spearman",
        "negative_macro_component_cell_mae",
        "negative_frozen_method_order_index",
    ]:
        raise BaselineSelectionError("selection tuple fields differ from frozen rule")
    records = payload.get("method_selection_records")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise BaselineSelectionError("method_selection_records must be a sequence")
    if len(records) != len(evidenced_order):
        raise BaselineSelectionError(
            "method selection records must cover exactly the evidenced methods"
        )
    normalized_records = []
    for record_index, record in enumerate(records):
        _strict_keys(
            record,
            _SELECTION_RECORD_KEYS,
            label=f"selection record {record_index}",
        )
        method = _method_id(record.get("method_id"))
        if method != evidenced_order[record_index]:
            raise BaselineSelectionError("selection records do not follow frozen order")
        method_order_index = order.index(method)
        if record.get("method_order_index") != method_order_index:
            raise BaselineSelectionError("selection record method-order index is invalid")
        if record.get("development_evidence_ref") != normalized_refs[method]:
            raise BaselineSelectionError("selection record evidence reference mismatch")
        metrics = _selection_metrics(
            record.get("development_metrics"),
            label=f"selection record {record_index} metrics",
        )
        gate_met = metrics["macro_component_cell_sign_accuracy"] >= SIGN_ACCURACY_GATE
        expected_tuple = [
            metrics["macro_component_cell_sign_accuracy"],
            metrics["macro_within_pool_spearman"],
            -metrics["macro_component_cell_mae"],
            -float(method_order_index),
        ]
        if record.get("sign_gate_met") is not gate_met:
            raise BaselineSelectionError("selection record sign-gate result is invalid")
        if record.get("lexicographic_selection_tuple") != expected_tuple:
            raise BaselineSelectionError("selection record tuple is not recomputable")
        normalized_records.append(dict(record))
    selected = max(
        normalized_records,
        key=lambda item: tuple(item["lexicographic_selection_tuple"]),
    )
    strongest_claim = not missing_required_methods and not missing_eligible_evidence
    baseline_sign_gate_met = bool(selected["sign_gate_met"])
    can_satisfy_pretest_freeze = bool(
        strongest_claim and baseline_sign_gate_met
    )
    expected_status = (
        QUALIFIED_STATUS if can_satisfy_pretest_freeze else HOLD_STATUS
    )
    if payload.get("status") != expected_status:
        raise BaselineSelectionError(
            "selection status does not match evidence completeness and sign gate"
        )
    if payload.get("selected_method_id") != selected["method_id"]:
        raise BaselineSelectionError("selected_method_id does not match frozen rule")
    selected_metrics = _selection_metrics(
        payload.get("selected_development_metrics"), label="selected metrics"
    )
    if selected_metrics != selected["development_metrics"]:
        raise BaselineSelectionError("selected metrics do not match selected method")
    if payload.get("comparator_selected") is not True:
        raise BaselineSelectionError("one deterministic comparator must be selected")
    if payload.get("baseline_sign_gate_met") is not baseline_sign_gate_met:
        raise BaselineSelectionError("top-level baseline sign-gate result is invalid")
    if payload.get("strongest_same_information_claim") is not strongest_claim:
        raise BaselineSelectionError(
            "strongest same-information claim does not match evidence completeness"
        )
    if (
        payload.get("can_satisfy_pretest_freeze")
        is not can_satisfy_pretest_freeze
    ):
        raise BaselineSelectionError(
            "pre-test-freeze qualification does not match strict selector state"
        )
    _boundaries(payload, label="development baseline selection")
    return dict(payload)


def canonical_development_baseline_selection_bytes(payload: Mapping) -> bytes:
    try:
        return canonical_json_bytes(validate_development_baseline_selection(payload))
    except ArtifactIOError as exc:
        raise BaselineSelectionError(str(exc)) from exc


def write_development_baseline_selection_exclusive(
    run_root: Path,
    path: Path,
    payload: Mapping,
    *,
    predictor_verifiers: Mapping[str, PredictorVerifier] | None,
) -> str:
    """Replay all evidence, rebuild selection, then atomically publish it."""

    _assert_output_within_run_root(run_root, path)
    validated = validate_development_baseline_selection(payload)
    evidences = {
        method: replay_development_baseline_evidence(
            run_root,
            validated["development_evidence_refs"][method],
            predictor_verifiers=predictor_verifiers,
            expected_run_start_sha256=validated["run_start_sha256"],
            expected_activation_manifest_sha256=validated[
                "activation_manifest_sha256"
            ],
            expected_train_development_outcomes_sha256=validated[
                "train_development_outcomes_sha256"
            ],
            expected_non_data_authority_hashes=validated[
                "non_data_authority_hashes"
            ],
            expected_method_id=method,
            expected_information_set_fingerprint=validated[
                "information_set_fingerprint"
            ],
        )
        for method in validated["eligible_method_order"]
        if method in validated["development_evidence_refs"]
    }
    rebuilt = _build_selection_payload(
        run_start_sha256=validated["run_start_sha256"],
        activation_manifest_sha256=validated["activation_manifest_sha256"],
        train_development_outcomes_sha256=validated[
            "train_development_outcomes_sha256"
        ],
        non_data_authority_hashes=validated["non_data_authority_hashes"],
        eligible_method_order=validated["eligible_method_order"],
        development_evidence_refs=validated["development_evidence_refs"],
        required_estimand_cells=validated["required_estimand_cells"],
        evidences=evidences,
    )
    if rebuilt != validated:
        raise BaselineSelectionError(
            "baseline selection changed during exact evidence replay"
        )
    try:
        return atomic_write_json_exclusive(
            Path(path), rebuilt, validator=validate_development_baseline_selection
        )
    except ArtifactIOError as exc:
        raise BaselineSelectionError(str(exc)) from exc


def load_development_baseline_selection_exact(
    run_root: Path,
    path: Path,
    expected_sha256: str,
    *,
    expected_run_start_sha256: str | None = None,
    expected_activation_manifest_sha256: str | None = None,
    expected_train_development_outcomes_sha256: str | None = None,
    expected_non_data_authority_hashes: Mapping | None = None,
    expected_eligible_method_order: Sequence[str] | None = None,
    expected_required_estimand_cells: Sequence[Mapping] | None = None,
    predictor_verifiers: Mapping[str, PredictorVerifier] | None,
) -> dict:
    """Read exact bytes, replay every dev ref, and rebuild the full selector."""

    _digest(expected_sha256, label="expected_sha256")
    try:
        raw = read_exact_regular_file(Path(path), expected_sha256)
    except ArtifactIOError as exc:
        raise BaselineSelectionError(str(exc)) from exc
    payload = validate_development_baseline_selection(
        _parse_json_object(raw, label="development baseline selection")
    )
    if canonical_development_baseline_selection_bytes(payload) != raw:
        raise BaselineSelectionError("baseline selection bytes are not canonical")
    expected_pairs = (
        ("run_start_sha256", expected_run_start_sha256),
        ("activation_manifest_sha256", expected_activation_manifest_sha256),
    )
    for field, expected in expected_pairs:
        if expected is not None and payload[field] != expected:
            raise BaselineSelectionError(f"baseline selection {field} binding mismatch")
    if expected_train_development_outcomes_sha256 is not None:
        if payload["train_development_outcomes_sha256"] != _digest(
            expected_train_development_outcomes_sha256,
            label="expected_train_development_outcomes_sha256",
        ):
            raise BaselineSelectionError(
                "baseline selection outcome-authority binding mismatch"
            )
    if expected_non_data_authority_hashes is not None:
        if payload["non_data_authority_hashes"] != _non_data_authority_hashes(
            expected_non_data_authority_hashes
        ):
            raise BaselineSelectionError(
                "baseline selection non-data authority binding mismatch"
            )
    if expected_eligible_method_order is not None:
        if payload["eligible_method_order"] != _eligible_method_order(
            expected_eligible_method_order
        ):
            raise BaselineSelectionError("baseline selection eligible order mismatch")
    if expected_required_estimand_cells is not None:
        if payload["required_estimand_cells"] != _required_estimand_cells(
            expected_required_estimand_cells
        ):
            raise BaselineSelectionError(
                "baseline selection required estimand/cell binding mismatch"
            )
    evidences = {
        method: replay_development_baseline_evidence(
            run_root,
            payload["development_evidence_refs"][method],
            predictor_verifiers=predictor_verifiers,
            expected_run_start_sha256=payload["run_start_sha256"],
            expected_activation_manifest_sha256=payload[
                "activation_manifest_sha256"
            ],
            expected_train_development_outcomes_sha256=payload[
                "train_development_outcomes_sha256"
            ],
            expected_non_data_authority_hashes=payload[
                "non_data_authority_hashes"
            ],
            expected_method_id=method,
            expected_information_set_fingerprint=payload[
                "information_set_fingerprint"
            ],
        )
        for method in payload["eligible_method_order"]
        if method in payload["development_evidence_refs"]
    }
    rebuilt = _build_selection_payload(
        run_start_sha256=payload["run_start_sha256"],
        activation_manifest_sha256=payload["activation_manifest_sha256"],
        train_development_outcomes_sha256=payload[
            "train_development_outcomes_sha256"
        ],
        non_data_authority_hashes=payload["non_data_authority_hashes"],
        eligible_method_order=payload["eligible_method_order"],
        development_evidence_refs=payload["development_evidence_refs"],
        required_estimand_cells=payload["required_estimand_cells"],
        evidences=evidences,
    )
    if rebuilt != payload:
        raise BaselineSelectionError(
            "baseline selection does not match exact development evidence replay"
        )
    return payload
