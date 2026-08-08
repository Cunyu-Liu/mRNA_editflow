"""Component-balanced train/development diagnostics for the R3X sign blocker.

This module deliberately does not estimate a learnability ceiling.  It measures
only a frozen majority-sign baseline and target-distribution summaries on
ordinary train/development rows.  In particular, a majority-class proportion is
never interpreted as an upper bound on a learned predictor, and a diagnostic
value above the contractual 0.60 sign threshold never unlocks a model, test
partition, downstream stage, or sealed evaluation.

The scientific unit is a formal cell followed by a global source component and
then a row.  The cell definition is exactly the frozen study x estimand x
publication-context x edit-count tuple; endpoint and region do not silently
split its denominator.  True zero effects are reported separately and excluded
from sign accuracy.  A predicted value of zero follows the frozen hard-sign
rule and is positive, although the majority predictor emitted here is always
either -1 or +1.  Identity-bearing per-component outcomes and outcome-bearing
component hashes are never published.  Only controlled, cell-level sign
aggregates are retained, and non-empty cells without minimum component and sign
diversity are suppressed before artifact creation.  These aggregates are not a
confidentiality mechanism and must remain inside the controlled development
outcome scope.

The supplied authority digests are checked against canonical semantic content,
but this module cannot prove that their issuer is independent.  Every artifact
therefore records that limitation and remains ineligible for model training,
pre-test freeze, publication claims, or downstream unlock.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Mapping, Sequence

from .artifact_io import (
    ArtifactIOError,
    atomic_write_json_exclusive,
    canonical_json_bytes,
    read_exact_regular_file,
)
from .development_estimands import (
    DEVELOPMENT_PARTITIONS,
    build_development_estimand_slices,
)
from .estimands import row_order_sha256, stable_row_id
from .protocol import ProtocolHoldError, canonical_sha256


SCHEMA_VERSION = "r3x_development_learnability_diagnostic_v2"
INPUT_BINDING_SCHEMA = "r3x_development_learnability_input_binding_v2"
FORMAL_CELL_SCHEMA = "r3x_development_learnability_formal_cell_v2"
REGISTERED_CELL_UNIVERSE_SCHEMA = "r3x_registered_primary_cell_universe_v2"
SIGN_GATE = 0.60
MINIMUM_COMPONENTS_PER_RELEASED_CELL_PARTITION = 10
MINIMUM_POSITIVE_AND_NEGATIVE_ROWS_PER_RELEASED_CELL_PARTITION = 3
ALLOWED_EDIT_COUNTS = (1, 3, 5)
FORMAL_CELL_FIELDS = (
    "study",
    "estimand_id",
    "publication_context_id",
    "edit_count",
)
DIAGNOSTIC_ROLE = "DIAGNOSTIC_NOT_LEARNABILITY_CEILING_NOT_MODEL_GATE"
MAJORITY_INTERPRETATION = "NAIVE_BASELINE_NOT_A_LEARNABILITY_UPPER_BOUND"
NOISE_CEILING_STATUS = (
    "NOISE_CEILING_UNIDENTIFIED_REPLICATE_IDENTITY_AUTHORITY_REQUIRED"
)
COMPLETE_STATUS = "COMPLETE_DEVELOPMENT_DIAGNOSTIC_NOT_A_GATE"
HOLD_STATUS = "HOLD_UNDEFINED_REGISTERED_CELL"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_UNDEFINED_REASON_ORDER = (
    "MISSING_TRAIN_PARTITION",
    "MISSING_DEVELOPMENT_PARTITION",
    "NO_NONZERO_TRAIN_TARGETS",
    "NO_NONZERO_DEVELOPMENT_TARGETS",
)

_POLICIES = {
    "formal_cell_fields": list(FORMAL_CELL_FIELDS),
    "weight_policy": "CELL_THEN_GLOBAL_SOURCE_COMPONENT_THEN_ROW_EQUAL_MASS_V1",
    "truth_sign_policy": "POSITIVE_IF_DELTA_GT_ZERO_NEGATIVE_IF_DELTA_LT_ZERO_TRUE_ZERO_EXCLUDED",
    "hard_prediction_sign_policy": "PREDICTED_ZERO_IS_POSITIVE",
    "majority_tie_policy": "COMPONENT_BALANCED_POSITIVE_FRACTION_AT_LEAST_HALF_SELECTS_POSITIVE",
    "sign_gate": SIGN_GATE,
    "allowed_edit_counts": list(ALLOWED_EDIT_COUNTS),
    "minimum_components_per_released_cell_partition": (
        MINIMUM_COMPONENTS_PER_RELEASED_CELL_PARTITION
    ),
    "minimum_positive_and_negative_rows_per_released_cell_partition": (
        MINIMUM_POSITIVE_AND_NEGATIVE_ROWS_PER_RELEASED_CELL_PARTITION
    ),
    "small_cell_policy": "HOLD_BEFORE_ARTIFACT_NO_OUTCOME_SUMMARY_RELEASE",
    "component_outcome_commitment_policy": "PROHIBITED_UNSALTED_OUTCOME_HASH_NOT_EMITTED",
    "majority_prior_interpretation": MAJORITY_INTERPRETATION,
    "noise_ceiling_status": NOISE_CEILING_STATUS,
}

_BOUNDARIES = {
    "ordinary_only": True,
    "disclosed_partitions": list(DEVELOPMENT_PARTITIONS),
    "internal_test_disclosed": False,
    "sealed_access": False,
    "downstream_unlock": False,
    "active_contract_mutation": False,
    "publication_claim": False,
    "go_effect": False,
    "selection_gate_used": False,
    "model_training_authorized": False,
    "pretest_freeze_eligible": False,
    "semantic_content_hashes_replayed_by_structure_validator": False,
    "trust_requires_exact_input_replay": True,
    "independent_authority_issuer_verified": False,
    "aggregate_confidentiality_claim": False,
    "access_scope": "INTERNAL_CONTROLLED_DEVELOPMENT_AGGREGATES_ONLY",
    "sign_gate_unchanged": True,
}


def _hold(code: str, message: str, **details: object) -> None:
    raise ProtocolHoldError(code, message, details=details)


def _exact_keys(value: Mapping, expected: set[str], *, label: str) -> None:
    if not isinstance(value, Mapping):
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_SCHEMA", f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        _hold(
            "HOLD_DEVELOPMENT_LEARNABILITY_SCHEMA",
            f"{label} has unexpected or missing fields",
            missing=sorted(expected - actual),
            extra=sorted(actual - expected),
        )


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_SCHEMA", f"{label} must be non-empty text")
    return value.strip()


def _sha256(value: object, *, label: str) -> str:
    text = _text(value, label=label).lower()
    if not _HEX64.fullmatch(text):
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_SCHEMA", f"{label} must be lowercase SHA-256")
    return text


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_SCHEMA", f"{label} must be a non-negative integer")
    return value


def _finite_nonnegative(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_SCHEMA", f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_SCHEMA", f"{label} must be finite and non-negative")
    return result


def _optional_fraction(value: object, *, label: str) -> float | None:
    if value is None:
        return None
    result = _finite_nonnegative(value, label=label)
    if result > 1.0:
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_SCHEMA", f"{label} must be in [0, 1]")
    return result


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(math.fsum(values) / len(values))


def canonical_estimand_registry_semantic_sha256(registry: Mapping) -> str:
    """Hash the exact canonical registry object consumed by this module."""

    if not isinstance(registry, Mapping):
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_AUTHORITY", "estimand registry must be an object")
    return canonical_sha256(dict(registry))


def canonical_train_development_outcomes_semantic_sha256(
    rows: Sequence[Mapping],
) -> str:
    """Hash all supplied row fields in canonical stable-row order.

    This is a semantic content commitment, not a claim that the caller is an
    independent authority issuer.  A production caller must compare it with a
    pre-bound digest obtained before this process receives outcome rows.
    """

    indexed: dict[str, dict] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            _hold("HOLD_DEVELOPMENT_LEARNABILITY_AUTHORITY", "outcome rows must be objects")
        row_id = stable_row_id(raw)
        if row_id in indexed:
            _hold("HOLD_DEVELOPMENT_LEARNABILITY_AUTHORITY", "outcome row IDs must be unique")
        indexed[row_id] = dict(raw)
    return canonical_sha256([indexed[row_id] for row_id in sorted(indexed)])


def _registry_index(registry: Mapping) -> dict[tuple[str, str], dict]:
    entries = registry.get("entries") if isinstance(registry, Mapping) else None
    if not isinstance(entries, list):
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_AUTHORITY", "estimand registry lacks entries")
    result: dict[tuple[str, str], dict] = {}
    for raw in entries:
        if not isinstance(raw, Mapping) or raw.get("role") != "PRIMARY":
            continue
        entry = dict(raw)
        if entry.get("valid_domain") != "FINITE":
            _hold(
                "HOLD_DEVELOPMENT_LEARNABILITY_AUTHORITY",
                "only finite-domain primary estimands are supported",
            )
        if entry.get("replicate_policy") != "SINGLE_OBSERVATION_ONLY":
            _hold(
                "HOLD_DEVELOPMENT_LEARNABILITY_AUTHORITY",
                "replicate identity is not qualified for this diagnostic schema",
            )
        key = (
            _text(entry.get("study"), label="registry study"),
            _text(entry.get("endpoint"), label="registry endpoint"),
        )
        if key in result:
            _hold("HOLD_DEVELOPMENT_LEARNABILITY_AUTHORITY", "duplicate registry key")
        result[key] = entry
    return result


def _replay_row_delta(row: Mapping, entry: Mapping) -> None:
    endpoint = row.get("endpoint")
    endpoint_id = row.get("endpoint_id")
    if endpoint is not None and endpoint_id is not None and str(endpoint) != str(endpoint_id):
        _hold(
            "HOLD_DEVELOPMENT_LEARNABILITY_ESTIMAND_REPLAY",
            "endpoint aliases disagree",
            row_id=stable_row_id(row),
        )
    source_field = _text(entry.get("raw_source_field"), label="raw_source_field")
    candidate_field = _text(entry.get("raw_candidate_field"), label="raw_candidate_field")
    values = (row.get(source_field), row.get(candidate_field))
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        _hold(
            "HOLD_DEVELOPMENT_LEARNABILITY_ESTIMAND_REPLAY",
            "estimand raw source/candidate values must be finite",
            row_id=stable_row_id(row),
        )
    source_value, candidate_value = map(float, values)
    transform = entry.get("transform")
    if transform == "IDENTITY_DIFFERENCE":
        expected = candidate_value - source_value
    elif transform == "NATURAL_LOG_RATIO":
        if source_value <= 0.0 or candidate_value <= 0.0:
            _hold(
                "HOLD_DEVELOPMENT_LEARNABILITY_ESTIMAND_REPLAY",
                "log-ratio estimand requires positive values",
            )
        expected = math.log(candidate_value / source_value)
    elif transform == "DECAY_RATE_DIFFERENCE":
        if source_value <= 0.0 or candidate_value <= 0.0:
            _hold(
                "HOLD_DEVELOPMENT_LEARNABILITY_ESTIMAND_REPLAY",
                "decay-rate estimand requires positive values",
            )
        expected = math.log(2.0) / source_value - math.log(2.0) / candidate_value
    else:
        _hold(
            "HOLD_DEVELOPMENT_LEARNABILITY_ESTIMAND_REPLAY",
            "unsupported estimand transform",
            transform=transform,
        )
    direction = entry.get("direction")
    if direction == "LOWER_IS_MORE_STABLE":
        expected = -expected
    elif direction != "HIGHER_IS_MORE_STABLE":
        _hold(
            "HOLD_DEVELOPMENT_LEARNABILITY_ESTIMAND_REPLAY",
            "unsupported estimand direction",
            direction=direction,
        )
    observed = row.get("delta")
    if (
        isinstance(observed, bool)
        or not isinstance(observed, (int, float))
        or not math.isfinite(float(observed))
        or not math.isclose(float(observed), expected, rel_tol=1e-10, abs_tol=1e-12)
    ):
        _hold(
            "HOLD_DEVELOPMENT_LEARNABILITY_ESTIMAND_REPLAY",
            "row delta does not replay the frozen estimand",
            row_id=stable_row_id(row),
        )


def _formal_cell_fields(row: Mapping) -> dict:
    fields = {
        "study": _text(row.get("study"), label="row study"),
        "estimand_id": _text(row.get("estimand_id"), label="row estimand_id"),
        "publication_context_id": _text(
            row.get("publication_context_id"), label="row publication_context_id"
        ),
        "edit_count": row.get("edit_count"),
    }
    edit_count = fields["edit_count"]
    if isinstance(edit_count, bool) or not isinstance(edit_count, int):
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_ACTION_BUDGET", "edit_count must be an integer")
    if edit_count not in ALLOWED_EDIT_COUNTS:
        _hold(
            "HOLD_DEVELOPMENT_LEARNABILITY_ACTION_BUDGET",
            "development diagnostic accepts only the frozen 1/3/5-edit primary tracks",
            edit_count=edit_count,
        )
    return fields


def _formal_cell_id(fields: Mapping) -> str:
    return canonical_sha256({"schema_version": FORMAL_CELL_SCHEMA, **dict(fields)})


def _normalize_registered_cells(cells: Sequence[Mapping]) -> list[dict]:
    normalized: list[dict] = []
    for index, raw in enumerate(cells):
        _exact_keys(raw, set(FORMAL_CELL_FIELDS), label=f"registered cell[{index}]")
        fields = {
            field: (
                raw[field]
                if field == "edit_count"
                else _text(raw[field], label=f"registered cell[{index}].{field}")
            )
            for field in FORMAL_CELL_FIELDS
        }
        edit_count = fields["edit_count"]
        if (
            isinstance(edit_count, bool)
            or not isinstance(edit_count, int)
            or edit_count not in ALLOWED_EDIT_COUNTS
        ):
            _hold(
                "HOLD_DEVELOPMENT_LEARNABILITY_ACTION_BUDGET",
                "registered cells require frozen 1/3/5 edit counts",
            )
        normalized.append(fields)
    normalized.sort(key=lambda value: tuple(value[field] for field in FORMAL_CELL_FIELDS))
    ids = [_formal_cell_id(value) for value in normalized]
    if not normalized or len(ids) != len(set(ids)):
        _hold(
            "HOLD_DEVELOPMENT_LEARNABILITY_CELL_UNIVERSE",
            "registered cell universe must be non-empty and unique",
        )
    return normalized


def canonical_registered_cell_universe_semantic_sha256(
    cells: Sequence[Mapping],
) -> str:
    normalized = _normalize_registered_cells(cells)
    return canonical_sha256({
        "schema_version": REGISTERED_CELL_UNIVERSE_SCHEMA,
        "cell_definition": "STUDY_ESTIMAND_PUBLICATION_CONTEXT_EDIT_COUNT",
        "cells": normalized,
    })


def _component_summaries(rows: Sequence[Mapping]) -> list[dict]:
    by_component: dict[str, list[Mapping]] = {}
    for row in rows:
        component = _text(row.get("source_component_id"), label="source_component_id")
        by_component.setdefault(component, []).append(row)
    summaries: list[dict] = []
    for component, members in sorted(by_component.items()):
        deltas = [float(row["delta"]) for row in sorted(members, key=stable_row_id)]
        summaries.append({
            "source_component_id": component,
            "n_rows": len(deltas),
            "n_positive": sum(delta > 0.0 for delta in deltas),
            "n_negative": sum(delta < 0.0 for delta in deltas),
            "n_zero": sum(delta == 0.0 for delta in deltas),
        })
    return summaries


def _partition_summary_from_components(components: Sequence[Mapping]) -> dict:
    n_rows = sum(int(component["n_rows"]) for component in components)
    nonzero_components = [
        component
        for component in components
        if int(component["n_positive"]) + int(component["n_negative"]) > 0
    ]
    positive_by_component = [
        int(component["n_positive"])
        / (int(component["n_positive"]) + int(component["n_negative"]))
        for component in nonzero_components
    ]
    zero_by_component = [
        int(component["n_zero"]) / int(component["n_rows"])
        for component in components
    ]
    n_positive = sum(int(component["n_positive"]) for component in components)
    n_negative = sum(int(component["n_negative"]) for component in components)
    n_zero = sum(int(component["n_zero"]) for component in components)
    nonzero_rows = n_positive + n_negative
    sum_component_positive_fraction = float(math.fsum(positive_by_component))
    sum_component_zero_fraction = float(math.fsum(zero_by_component))
    return {
        "n_rows": n_rows,
        "n_components": len(components),
        "n_components_with_nonzero": len(nonzero_components),
        "n_nonzero_rows": nonzero_rows,
        "n_positive_rows": n_positive,
        "n_negative_rows": n_negative,
        "n_zero_rows": n_zero,
        "sum_component_positive_fraction_nonzero": (
            sum_component_positive_fraction
        ),
        "sum_component_zero_fraction": sum_component_zero_fraction,
        "component_balanced_positive_fraction_nonzero": _mean(positive_by_component),
        "component_balanced_zero_fraction": _mean(zero_by_component),
        "row_micro_positive_fraction_nonzero": (
            float(n_positive / nonzero_rows) if nonzero_rows else None
        ),
        "row_micro_zero_fraction": float(n_zero / n_rows) if n_rows else None,
    }


def _partition_summary(rows: Sequence[Mapping]) -> dict:
    summary = _partition_summary_from_components(_component_summaries(rows))
    if summary["n_components"] and (
        summary["n_components"] < MINIMUM_COMPONENTS_PER_RELEASED_CELL_PARTITION
        or (
            summary["n_components_with_nonzero"]
            and summary["n_components_with_nonzero"]
            < MINIMUM_COMPONENTS_PER_RELEASED_CELL_PARTITION
        )
    ):
        _hold(
            "HOLD_DEVELOPMENT_LEARNABILITY_SMALL_CELL_SUPPRESSION",
            "non-empty released cell partitions require at least ten components; "
            "if any outcomes are non-zero, at least ten non-zero components are required",
            n_components=summary["n_components"],
            n_components_with_nonzero=summary["n_components_with_nonzero"],
        )
    if summary["n_rows"] and (
        summary["n_positive_rows"]
        < MINIMUM_POSITIVE_AND_NEGATIVE_ROWS_PER_RELEASED_CELL_PARTITION
        or summary["n_negative_rows"]
        < MINIMUM_POSITIVE_AND_NEGATIVE_ROWS_PER_RELEASED_CELL_PARTITION
    ):
        _hold(
            "HOLD_DEVELOPMENT_LEARNABILITY_ATTRIBUTE_DIVERSITY_SUPPRESSION",
            "released non-empty cell partitions require at least three positive "
            "and three negative rows; exact homogeneous or low-diversity outcome "
            "aggregates are not emitted",
        )
    return summary


def _cell_result(fields: Mapping, partitions: Mapping[str, Sequence[Mapping]]) -> dict:
    summaries = {
        partition: _partition_summary(partitions.get(partition, ()))
        for partition in DEVELOPMENT_PARTITIONS
    }
    train = summaries["train"]
    development = summaries["development"]
    train_fraction = train["component_balanced_positive_fraction_nonzero"]
    prior_sign = None if train_fraction is None else (1 if train_fraction >= 0.5 else -1)

    if prior_sign is None or not development["n_components_with_nonzero"]:
        component_accuracy = None
    else:
        positive_accuracy = (
            float(development["sum_component_positive_fraction_nonzero"])
            / int(development["n_components_with_nonzero"])
        )
        component_accuracy = (
            positive_accuracy if prior_sign == 1 else 1.0 - positive_accuracy
        )
    if prior_sign is None or not development["n_nonzero_rows"]:
        row_micro_accuracy = None
    else:
        correct = int(
            development["n_positive_rows"]
            if prior_sign == 1
            else development["n_negative_rows"]
        )
        row_micro_accuracy = float(correct / int(development["n_nonzero_rows"]))

    reasons: list[str] = []
    if not train["n_rows"]:
        reasons.append("MISSING_TRAIN_PARTITION")
    elif train_fraction is None:
        reasons.append("NO_NONZERO_TRAIN_TARGETS")
    if not development["n_rows"]:
        reasons.append("MISSING_DEVELOPMENT_PARTITION")
    elif not development["n_nonzero_rows"]:
        reasons.append("NO_NONZERO_DEVELOPMENT_TARGETS")
    reasons = [reason for reason in _UNDEFINED_REASON_ORDER if reason in reasons]
    return {
        "cell_id": _formal_cell_id(fields),
        "formal_cell": dict(fields),
        "status": "DEFINED" if not reasons else "UNDEFINED",
        "undefined_reasons": reasons,
        "partition_summaries": summaries,
        "train_majority_sign": prior_sign,
        "development_component_balanced_majority_sign_accuracy": component_accuracy,
        "development_row_micro_majority_sign_accuracy": row_micro_accuracy,
    }


def _aggregate(cells: Sequence[Mapping]) -> dict:
    defined = [cell for cell in cells if cell["status"] == "DEFINED"]
    train_fractions = [
        float(cell["partition_summaries"]["train"]["component_balanced_positive_fraction_nonzero"])
        for cell in defined
    ]
    component_accuracies = [
        float(cell["development_component_balanced_majority_sign_accuracy"])
        for cell in defined
    ]
    cell_row_micro_accuracies = [
        float(cell["development_row_micro_majority_sign_accuracy"])
        for cell in defined
    ]
    pooled_correct = 0
    pooled_nonzero = 0
    for cell in defined:
        prior_sign = int(cell["train_majority_sign"])
        development = cell["partition_summaries"]["development"]
        pooled_correct += int(
            development["n_positive_rows"]
            if prior_sign == 1
            else development["n_negative_rows"]
        )
        pooled_nonzero += int(development["n_nonzero_rows"])
    complete = len(defined) == len(cells)
    macro_accuracy = _mean(component_accuracies) if complete else None
    return {
        "n_cells": len(cells),
        "n_defined_cells": len(defined),
        "n_undefined_cells": len(cells) - len(defined),
        "undefined_cell_ids": sorted(
            str(cell["cell_id"]) for cell in cells if cell["status"] != "DEFINED"
        ),
        "cell_macro_train_component_balanced_positive_fraction_nonzero": (
            _mean(train_fractions) if complete else None
        ),
        "cell_macro_development_component_balanced_majority_sign_accuracy": macro_accuracy,
        "cell_macro_development_row_micro_majority_sign_accuracy": (
            _mean(cell_row_micro_accuracies) if complete else None
        ),
        "pooled_row_micro_development_majority_sign_accuracy": (
            float(pooled_correct / pooled_nonzero)
            if complete and pooled_nonzero
            else None
        ),
        "diagnostic_numeric_threshold_reached": (
            bool(macro_accuracy >= SIGN_GATE) if macro_accuracy is not None else None
        ),
        "go_effect": False,
    }


def _partition_binding(rows: Sequence[Mapping]) -> dict:
    ordered = sorted((dict(row) for row in rows), key=stable_row_id)
    row_ids = [stable_row_id(row) for row in ordered]
    outcomes = [
        {"row_id": stable_row_id(row), "delta": float(row["delta"])} for row in ordered
    ]
    assignments = [
        {
            "row_id": stable_row_id(row),
            **_formal_cell_fields(row),
            "source_component_id": _text(
                row.get("source_component_id"), label="source_component_id"
            ),
            "source_group_id_v2": _text(
                row.get("source_group_id_v2"), label="source_group_id_v2"
            ),
        }
        for row in ordered
    ]
    return {
        "n_rows": len(ordered),
        "row_order_sha256": row_order_sha256(row_ids),
        "row_set_sha256": canonical_sha256(sorted(row_ids)),
        "outcome_content_sha256": canonical_sha256(outcomes),
        "formal_assignment_sha256": canonical_sha256(assignments),
    }


def _study_endpoint_scope(registry: Mapping) -> dict:
    entries = [
        dict(entry)
        for entry in registry["entries"]
        if isinstance(entry, Mapping) and entry.get("role") == "PRIMARY"
    ]
    study_to_endpoints: dict[str, set[str]] = {}
    endpoint_to_studies: dict[str, set[str]] = {}
    for entry in entries:
        study = _text(entry.get("study"), label="registry study")
        endpoint = _text(entry.get("endpoint"), label="registry endpoint")
        study_to_endpoints.setdefault(study, set()).add(endpoint)
        endpoint_to_studies.setdefault(endpoint, set()).add(study)
    studies = sorted(study_to_endpoints)
    endpoints = sorted(endpoint_to_studies)
    if len(studies) < 2:
        status = "DEVELOPMENT_ONLY_SINGLE_STUDY"
    elif all(len(values) == 1 for values in study_to_endpoints.values()) and all(
        len(values) == 1 for values in endpoint_to_studies.values()
    ):
        status = "DEVELOPMENT_ONLY_STUDY_ENDPOINT_ONE_TO_ONE_NESTED"
    else:
        status = "DEVELOPMENT_ONLY_MULTI_STUDY_ENDPOINTS_NOT_ONE_TO_ONE"
    return {
        "status": status,
        "n_studies": len(studies),
        "n_endpoints": len(endpoints),
        "study_to_endpoints": {
            study: sorted(study_to_endpoints[study]) for study in studies
        },
        "endpoint_to_studies": {
            endpoint: sorted(endpoint_to_studies[endpoint]) for endpoint in endpoints
        },
        "publication_cross_study_claim": False,
    }


def _primary_entry_set_sha256(registry: Mapping) -> str:
    entries = [
        dict(entry)
        for entry in registry["entries"]
        if isinstance(entry, Mapping) and entry.get("role") == "PRIMARY"
    ]
    return canonical_sha256(
        sorted(entries, key=lambda entry: (str(entry.get("study")), str(entry.get("endpoint"))))
    )


def build_development_learnability_diagnostic(
    rows: Sequence[Mapping],
    estimand_registry: Mapping,
    *,
    train_development_outcomes_sha256: str,
    estimand_registry_sha256: str,
    registered_cells: Sequence[Mapping],
    registered_cell_universe_sha256: str,
) -> dict:
    """Build a deterministic ordinary train/development diagnostic artifact."""

    data_sha = _sha256(
        train_development_outcomes_sha256,
        label="train/development outcomes authority SHA-256",
    )
    registry_sha = _sha256(
        estimand_registry_sha256, label="endpoint estimand registry SHA-256"
    )
    expected_registry_sha = canonical_estimand_registry_semantic_sha256(
        estimand_registry
    )
    if registry_sha != expected_registry_sha:
        _hold(
            "HOLD_DEVELOPMENT_LEARNABILITY_AUTHORITY",
            "estimand registry semantic SHA-256 does not match supplied object",
        )
    normalized_registered_cells = _normalize_registered_cells(registered_cells)
    cell_universe_sha = _sha256(
        registered_cell_universe_sha256,
        label="registered cell universe SHA-256",
    )
    expected_cell_universe_sha = canonical_registered_cell_universe_semantic_sha256(
        normalized_registered_cells
    )
    if cell_universe_sha != expected_cell_universe_sha:
        _hold(
            "HOLD_DEVELOPMENT_LEARNABILITY_CELL_UNIVERSE",
            "registered cell-universe semantic SHA-256 does not match supplied cells",
        )
    slices = build_development_estimand_slices(
        rows,
        estimand_registry,
        estimand_registry_sha256=registry_sha,
    )
    partition_rows: dict[str, list[dict]] = {
        partition: [] for partition in DEVELOPMENT_PARTITIONS
    }
    cell_rows: dict[tuple, dict[str, list[dict]]] = {}
    registry_by_key = _registry_index(estimand_registry)
    component_partition: dict[str, str] = {}
    group_partition: dict[str, str] = {}
    for item in slices:
        for partition in DEVELOPMENT_PARTITIONS:
            for raw in item.rows_for(partition):
                row = dict(raw)
                endpoint = _text(
                    row.get("endpoint") or row.get("endpoint_id"), label="row endpoint"
                )
                entry = registry_by_key.get((str(row["study"]), endpoint))
                if entry is None:
                    _hold(
                        "HOLD_DEVELOPMENT_LEARNABILITY_ESTIMAND_REPLAY",
                        "row has no primary estimand entry",
                    )
                _replay_row_delta(row, entry)
                component = _text(
                    row.get("source_component_id"), label="source_component_id"
                )
                previous_component_partition = component_partition.setdefault(
                    component, partition
                )
                if previous_component_partition != partition:
                    _hold(
                        "HOLD_DEVELOPMENT_LEARNABILITY_PARTITION_LEAKAGE",
                        "one global source component crosses train/development",
                        source_component_id=component,
                    )
                group = _text(row.get("source_group_id_v2"), label="source_group_id_v2")
                previous_group_partition = group_partition.setdefault(group, partition)
                if previous_group_partition != partition:
                    _hold(
                        "HOLD_DEVELOPMENT_LEARNABILITY_PARTITION_LEAKAGE",
                        "one source ranking group crosses train/development",
                        source_group_id_v2=group,
                    )
                fields = _formal_cell_fields(row)
                key = tuple(fields[field] for field in FORMAL_CELL_FIELDS)
                cell_rows.setdefault(
                    key, {name: [] for name in DEVELOPMENT_PARTITIONS}
                )[partition].append(row)
                partition_rows[partition].append(row)

    semantic_data_sha = canonical_train_development_outcomes_semantic_sha256(
        [*partition_rows["train"], *partition_rows["development"]]
    )
    if data_sha != semantic_data_sha:
        _hold(
            "HOLD_DEVELOPMENT_LEARNABILITY_AUTHORITY",
            "train/development semantic SHA-256 does not match supplied rows",
        )

    registered_keys = {
        tuple(fields[field] for field in FORMAL_CELL_FIELDS): fields
        for fields in normalized_registered_cells
    }
    unexpected_keys = sorted(set(cell_rows) - set(registered_keys))
    if unexpected_keys:
        _hold(
            "HOLD_DEVELOPMENT_LEARNABILITY_CELL_UNIVERSE",
            "observed rows contain cells outside the registered universe",
            unexpected=unexpected_keys,
        )
    for key in registered_keys:
        cell_rows.setdefault(key, {name: [] for name in DEVELOPMENT_PARTITIONS})

    cells = []
    for key, partitions in sorted(cell_rows.items()):
        fields = dict(zip(FORMAL_CELL_FIELDS, key))
        cells.append(_cell_result(fields, partitions))
    input_partitions = {
        partition: _partition_binding(partition_rows[partition])
        for partition in DEVELOPMENT_PARTITIONS
    }
    input_binding_without_sha = {
        "schema_version": INPUT_BINDING_SCHEMA,
        "partitions": input_partitions,
    }
    input_binding = {
        **input_binding_without_sha,
        "overall_binding_sha256": canonical_sha256(input_binding_without_sha),
    }
    aggregate = _aggregate(cells)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": COMPLETE_STATUS if not aggregate["n_undefined_cells"] else HOLD_STATUS,
        "diagnostic_role": DIAGNOSTIC_ROLE,
        "train_development_outcomes_sha256": data_sha,
        "estimand_registry_sha256": registry_sha,
        "registered_cell_universe_sha256": cell_universe_sha,
        "registry_primary_entry_set_sha256": _primary_entry_set_sha256(estimand_registry),
        "input_binding": input_binding,
        "study_endpoint_scope": _study_endpoint_scope(estimand_registry),
        "policies": dict(_POLICIES),
        "cells": cells,
        "aggregate": aggregate,
        "boundaries": dict(_BOUNDARIES),
    }
    return validate_development_learnability_diagnostic(payload)


def _validate_partition_summary(summary: Mapping, *, label: str) -> dict:
    expected_keys = {
        "n_rows",
        "n_components",
        "n_components_with_nonzero",
        "n_nonzero_rows",
        "n_positive_rows",
        "n_negative_rows",
        "n_zero_rows",
        "sum_component_positive_fraction_nonzero",
        "sum_component_zero_fraction",
        "component_balanced_positive_fraction_nonzero",
        "component_balanced_zero_fraction",
        "row_micro_positive_fraction_nonzero",
        "row_micro_zero_fraction",
    }
    _exact_keys(summary, expected_keys, label=label)
    normalized = {
        "n_rows": _nonnegative_int(summary["n_rows"], label=f"{label} n_rows"),
        "n_components": _nonnegative_int(
            summary["n_components"], label=f"{label} n_components"
        ),
        "n_components_with_nonzero": _nonnegative_int(
            summary["n_components_with_nonzero"],
            label=f"{label} n_components_with_nonzero",
        ),
        "n_nonzero_rows": _nonnegative_int(
            summary["n_nonzero_rows"], label=f"{label} n_nonzero_rows"
        ),
        "n_positive_rows": _nonnegative_int(
            summary["n_positive_rows"], label=f"{label} n_positive_rows"
        ),
        "n_negative_rows": _nonnegative_int(
            summary["n_negative_rows"], label=f"{label} n_negative_rows"
        ),
        "n_zero_rows": _nonnegative_int(
            summary["n_zero_rows"], label=f"{label} n_zero_rows"
        ),
        "sum_component_positive_fraction_nonzero": _finite_nonnegative(
            summary["sum_component_positive_fraction_nonzero"],
            label=f"{label} summed component positive fraction",
        ),
        "sum_component_zero_fraction": _finite_nonnegative(
            summary["sum_component_zero_fraction"],
            label=f"{label} summed component zero fraction",
        ),
        "component_balanced_positive_fraction_nonzero": _optional_fraction(
            summary["component_balanced_positive_fraction_nonzero"],
            label=f"{label} component positive fraction",
        ),
        "component_balanced_zero_fraction": _optional_fraction(
            summary["component_balanced_zero_fraction"],
            label=f"{label} component zero fraction",
        ),
        "row_micro_positive_fraction_nonzero": _optional_fraction(
            summary["row_micro_positive_fraction_nonzero"],
            label=f"{label} row-micro positive fraction",
        ),
        "row_micro_zero_fraction": _optional_fraction(
            summary["row_micro_zero_fraction"],
            label=f"{label} row-micro zero fraction",
        ),
    }
    if (
        normalized["n_positive_rows"]
        + normalized["n_negative_rows"]
        + normalized["n_zero_rows"]
        != normalized["n_rows"]
        or normalized["n_positive_rows"] + normalized["n_negative_rows"]
        != normalized["n_nonzero_rows"]
        or normalized["n_components_with_nonzero"] > normalized["n_components"]
        or normalized["sum_component_positive_fraction_nonzero"]
        > normalized["n_components_with_nonzero"]
        or normalized["sum_component_zero_fraction"] > normalized["n_components"]
    ):
        _hold(
            "HOLD_DEVELOPMENT_LEARNABILITY_SCHEMA",
            f"{label} aggregate sufficient statistics are inconsistent",
        )
    if normalized["n_components"] and (
        normalized["n_components"]
        < MINIMUM_COMPONENTS_PER_RELEASED_CELL_PARTITION
        or (
            normalized["n_components_with_nonzero"]
            and normalized["n_components_with_nonzero"]
            < MINIMUM_COMPONENTS_PER_RELEASED_CELL_PARTITION
        )
    ):
        _hold(
            "HOLD_DEVELOPMENT_LEARNABILITY_SMALL_CELL_SUPPRESSION",
            f"{label} has insufficient component support for aggregate release",
        )
    if normalized["n_rows"] and (
        normalized["n_positive_rows"]
        < MINIMUM_POSITIVE_AND_NEGATIVE_ROWS_PER_RELEASED_CELL_PARTITION
        or normalized["n_negative_rows"]
        < MINIMUM_POSITIVE_AND_NEGATIVE_ROWS_PER_RELEASED_CELL_PARTITION
    ):
        _hold(
            "HOLD_DEVELOPMENT_LEARNABILITY_ATTRIBUTE_DIVERSITY_SUPPRESSION",
            f"{label} has insufficient positive/negative diversity for aggregate release",
        )
    if normalized["n_rows"] == 0:
        expected = _partition_summary_from_components([])
    else:
        n_nonzero_components = normalized["n_components_with_nonzero"]
        n_components = normalized["n_components"]
        expected = {
            **normalized,
            "component_balanced_positive_fraction_nonzero": (
                normalized["sum_component_positive_fraction_nonzero"]
                / n_nonzero_components
                if n_nonzero_components
                else None
            ),
            "component_balanced_zero_fraction": (
                normalized["sum_component_zero_fraction"] / n_components
                if n_components
                else None
            ),
            "row_micro_positive_fraction_nonzero": (
                normalized["n_positive_rows"] / normalized["n_nonzero_rows"]
                if normalized["n_nonzero_rows"]
                else None
            ),
            "row_micro_zero_fraction": (
                normalized["n_zero_rows"] / normalized["n_rows"]
            ),
        }
    for field in (
        "component_balanced_positive_fraction_nonzero",
        "component_balanced_zero_fraction",
        "row_micro_positive_fraction_nonzero",
        "row_micro_zero_fraction",
    ):
        if normalized[field] != expected[field]:
            _hold(
                "HOLD_DEVELOPMENT_LEARNABILITY_REPLAY",
                f"{label} {field} does not replay from aggregate sufficient statistics",
            )
    return normalized


def _validate_cell(cell: Mapping, *, index: int) -> dict:
    label = f"cell[{index}]"
    _exact_keys(
        cell,
        {
            "cell_id",
            "formal_cell",
            "status",
            "undefined_reasons",
            "partition_summaries",
            "train_majority_sign",
            "development_component_balanced_majority_sign_accuracy",
            "development_row_micro_majority_sign_accuracy",
        },
        label=label,
    )
    formal = cell["formal_cell"]
    _exact_keys(formal, set(FORMAL_CELL_FIELDS), label=f"{label} formal_cell")
    normalized_formal = {
        field: (
            formal[field]
            if field == "edit_count"
            else _text(formal[field], label=f"{label} {field}")
        )
        for field in FORMAL_CELL_FIELDS
    }
    edit_count = normalized_formal["edit_count"]
    if isinstance(edit_count, bool) or not isinstance(edit_count, int) or edit_count not in ALLOWED_EDIT_COUNTS:
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_SCHEMA", f"{label} edit_count is invalid")
    cell_id = _sha256(cell["cell_id"], label=f"{label} cell_id")
    if cell_id != _formal_cell_id(normalized_formal):
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_REPLAY", f"{label} cell_id does not replay")
    partition_summaries = cell["partition_summaries"]
    _exact_keys(
        partition_summaries, set(DEVELOPMENT_PARTITIONS), label=f"{label} partition_summaries"
    )
    normalized_summaries = {
        partition: _validate_partition_summary(
            partition_summaries[partition], label=f"{label} {partition}"
        )
        for partition in DEVELOPMENT_PARTITIONS
    }
    recomputed = _cell_result(
        normalized_formal,
        {
            "train": (),
            "development": (),
        },
    )
    # Replace the empty-row summaries with the structurally replayed cell-level
    # sufficient aggregates, then recompute the cell-level sign quantities
    # without trusting caller-provided accuracy values.  Exact association with
    # source rows is intentionally left to the input-bound loader.
    train = normalized_summaries["train"]
    development = normalized_summaries["development"]
    train_fraction = train["component_balanced_positive_fraction_nonzero"]
    prior = None if train_fraction is None else (1 if train_fraction >= 0.5 else -1)
    if prior is None or not development["n_components_with_nonzero"]:
        component_accuracy = None
    else:
        positive_accuracy = (
            development["sum_component_positive_fraction_nonzero"]
            / development["n_components_with_nonzero"]
        )
        component_accuracy = positive_accuracy if prior == 1 else 1.0 - positive_accuracy
    if prior is None or not development["n_nonzero_rows"]:
        row_accuracy = None
    else:
        correct = (
            development["n_positive_rows"]
            if prior == 1
            else development["n_negative_rows"]
        )
        row_accuracy = float(correct / development["n_nonzero_rows"])
    reasons = []
    if not train["n_rows"]:
        reasons.append("MISSING_TRAIN_PARTITION")
    elif train_fraction is None:
        reasons.append("NO_NONZERO_TRAIN_TARGETS")
    if not development["n_rows"]:
        reasons.append("MISSING_DEVELOPMENT_PARTITION")
    elif not development["n_nonzero_rows"]:
        reasons.append("NO_NONZERO_DEVELOPMENT_TARGETS")
    reasons = [reason for reason in _UNDEFINED_REASON_ORDER if reason in reasons]
    expected_status = "DEFINED" if not reasons else "UNDEFINED"
    if cell["status"] != expected_status or cell["undefined_reasons"] != reasons:
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_REPLAY", f"{label} status does not replay")
    if cell["train_majority_sign"] != prior:
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_REPLAY", f"{label} majority sign does not replay")
    if _optional_fraction(
        cell["development_component_balanced_majority_sign_accuracy"],
        label=f"{label} component accuracy",
    ) != component_accuracy:
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_REPLAY", f"{label} component accuracy does not replay")
    if _optional_fraction(
        cell["development_row_micro_majority_sign_accuracy"],
        label=f"{label} row accuracy",
    ) != row_accuracy:
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_REPLAY", f"{label} row accuracy does not replay")
    recomputed.update({
        "partition_summaries": normalized_summaries,
        "status": expected_status,
        "undefined_reasons": reasons,
        "train_majority_sign": prior,
        "development_component_balanced_majority_sign_accuracy": component_accuracy,
        "development_row_micro_majority_sign_accuracy": row_accuracy,
    })
    return recomputed


def _validate_input_binding(binding: Mapping) -> dict:
    _exact_keys(
        binding,
        {"schema_version", "partitions", "overall_binding_sha256"},
        label="input_binding",
    )
    if binding["schema_version"] != INPUT_BINDING_SCHEMA:
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_SCHEMA", "unsupported input binding schema")
    partitions = binding["partitions"]
    _exact_keys(partitions, set(DEVELOPMENT_PARTITIONS), label="input partitions")
    normalized_partitions = {}
    for partition in DEVELOPMENT_PARTITIONS:
        value = partitions[partition]
        _exact_keys(
            value,
            {
                "n_rows",
                "row_order_sha256",
                "row_set_sha256",
                "outcome_content_sha256",
                "formal_assignment_sha256",
            },
            label=f"input partition {partition}",
        )
        normalized_partitions[partition] = {
            "n_rows": _nonnegative_int(value["n_rows"], label=f"{partition} n_rows"),
            "row_order_sha256": _sha256(
                value["row_order_sha256"], label=f"{partition} row order SHA-256"
            ),
            "row_set_sha256": _sha256(
                value["row_set_sha256"], label=f"{partition} row set SHA-256"
            ),
            "outcome_content_sha256": _sha256(
                value["outcome_content_sha256"], label=f"{partition} outcome SHA-256"
            ),
            "formal_assignment_sha256": _sha256(
                value["formal_assignment_sha256"], label=f"{partition} assignment SHA-256"
            ),
        }
    normalized_without_sha = {
        "schema_version": INPUT_BINDING_SCHEMA,
        "partitions": normalized_partitions,
    }
    overall = _sha256(binding["overall_binding_sha256"], label="overall input binding SHA-256")
    if overall != canonical_sha256(normalized_without_sha):
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_REPLAY", "overall input binding does not replay")
    return {**normalized_without_sha, "overall_binding_sha256": overall}


def _validate_scope(scope: Mapping) -> dict:
    _exact_keys(
        scope,
        {
            "status",
            "n_studies",
            "n_endpoints",
            "study_to_endpoints",
            "endpoint_to_studies",
            "publication_cross_study_claim",
        },
        label="study_endpoint_scope",
    )
    study_map = scope["study_to_endpoints"]
    endpoint_map = scope["endpoint_to_studies"]
    if not isinstance(study_map, Mapping) or not isinstance(endpoint_map, Mapping):
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_SCHEMA", "scope maps must be objects")
    normalized_study_map = {}
    for study, endpoints in sorted(study_map.items()):
        study_name = _text(study, label="scope study")
        if not isinstance(endpoints, list) or not endpoints:
            _hold("HOLD_DEVELOPMENT_LEARNABILITY_SCHEMA", "scope study endpoints must be non-empty lists")
        values = [_text(value, label="scope endpoint") for value in endpoints]
        if values != sorted(set(values)):
            _hold("HOLD_DEVELOPMENT_LEARNABILITY_SCHEMA", "scope endpoint lists must be canonical")
        normalized_study_map[study_name] = values
    normalized_endpoint_map = {}
    for endpoint, studies in sorted(endpoint_map.items()):
        endpoint_name = _text(endpoint, label="scope endpoint")
        if not isinstance(studies, list) or not studies:
            _hold("HOLD_DEVELOPMENT_LEARNABILITY_SCHEMA", "scope endpoint studies must be non-empty lists")
        values = [_text(value, label="scope study") for value in studies]
        if values != sorted(set(values)):
            _hold("HOLD_DEVELOPMENT_LEARNABILITY_SCHEMA", "scope study lists must be canonical")
        normalized_endpoint_map[endpoint_name] = values
    reverse_study = {
        study: sorted(
            endpoint for endpoint, studies in normalized_endpoint_map.items() if study in studies
        )
        for study in normalized_study_map
    }
    reverse_endpoint = {
        endpoint: sorted(
            study for study, endpoints in normalized_study_map.items() if endpoint in endpoints
        )
        for endpoint in normalized_endpoint_map
    }
    if reverse_study != normalized_study_map or reverse_endpoint != normalized_endpoint_map:
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_REPLAY", "study/endpoint scope maps disagree")
    n_studies = _nonnegative_int(scope["n_studies"], label="scope n_studies")
    n_endpoints = _nonnegative_int(scope["n_endpoints"], label="scope n_endpoints")
    if n_studies != len(normalized_study_map) or n_endpoints != len(normalized_endpoint_map):
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_REPLAY", "scope counts do not replay")
    expected_status = (
        "DEVELOPMENT_ONLY_SINGLE_STUDY"
        if n_studies < 2
        else (
            "DEVELOPMENT_ONLY_STUDY_ENDPOINT_ONE_TO_ONE_NESTED"
            if all(len(value) == 1 for value in normalized_study_map.values())
            and all(len(value) == 1 for value in normalized_endpoint_map.values())
            else "DEVELOPMENT_ONLY_MULTI_STUDY_ENDPOINTS_NOT_ONE_TO_ONE"
        )
    )
    if scope["status"] != expected_status or scope["publication_cross_study_claim"] is not False:
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_REPLAY", "scope status or claim does not replay")
    return {
        "status": expected_status,
        "n_studies": n_studies,
        "n_endpoints": n_endpoints,
        "study_to_endpoints": normalized_study_map,
        "endpoint_to_studies": normalized_endpoint_map,
        "publication_cross_study_claim": False,
    }


def validate_development_learnability_diagnostic(payload: Mapping) -> dict:
    """Validate diagnostic structure and self-contained arithmetic only.

    This public compatibility name is deliberately *not* a provenance or
    scientific trust boundary.  It cannot associate embedded aggregates with
    independently issued outcome bytes.  Scientific consumers must call
    :func:`load_development_learnability_diagnostic_exact`, which rebuilds the
    complete payload from exact expected rows, registry, and cell universe.
    """

    _exact_keys(
        payload,
        {
            "schema_version",
            "status",
            "diagnostic_role",
            "train_development_outcomes_sha256",
            "estimand_registry_sha256",
            "registered_cell_universe_sha256",
            "registry_primary_entry_set_sha256",
            "input_binding",
            "study_endpoint_scope",
            "policies",
            "cells",
            "aggregate",
            "boundaries",
        },
        label="diagnostic",
    )
    if payload["schema_version"] != SCHEMA_VERSION:
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_SCHEMA", "unsupported diagnostic schema")
    if payload["diagnostic_role"] != DIAGNOSTIC_ROLE:
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_SCHEMA", "diagnostic role cannot be upgraded")
    if payload["policies"] != _POLICIES or payload["boundaries"] != _BOUNDARIES:
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_SCHEMA", "diagnostic policies or boundaries changed")
    data_sha = _sha256(
        payload["train_development_outcomes_sha256"], label="train/development SHA-256"
    )
    registry_sha = _sha256(payload["estimand_registry_sha256"], label="registry SHA-256")
    cell_universe_sha = _sha256(
        payload["registered_cell_universe_sha256"],
        label="registered cell-universe SHA-256",
    )
    entry_set_sha = _sha256(
        payload["registry_primary_entry_set_sha256"], label="registry entry-set SHA-256"
    )
    binding = _validate_input_binding(payload["input_binding"])
    scope = _validate_scope(payload["study_endpoint_scope"])
    raw_cells = payload["cells"]
    if not isinstance(raw_cells, list) or not raw_cells:
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_SCHEMA", "diagnostic requires at least one cell")
    cells = [_validate_cell(cell, index=index) for index, cell in enumerate(raw_cells)]
    order = [tuple(cell["formal_cell"][field] for field in FORMAL_CELL_FIELDS) for cell in cells]
    if order != sorted(order) or len({cell["cell_id"] for cell in cells}) != len(cells):
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_SCHEMA", "cells must be unique and canonical")
    if cell_universe_sha != canonical_registered_cell_universe_semantic_sha256(
        [cell["formal_cell"] for cell in cells]
    ):
        _hold(
            "HOLD_DEVELOPMENT_LEARNABILITY_REPLAY",
            "registered cell-universe SHA-256 does not replay from embedded cells",
        )
    for partition in DEVELOPMENT_PARTITIONS:
        embedded_rows = sum(
            int(cell["partition_summaries"][partition]["n_rows"])
            for cell in cells
        )
        if embedded_rows != binding["partitions"][partition]["n_rows"]:
            _hold(
                "HOLD_DEVELOPMENT_LEARNABILITY_REPLAY",
                "input partition row count does not match embedded cell summaries",
                partition=partition,
            )
    expected_aggregate = _aggregate(cells)
    if payload["aggregate"] != expected_aggregate:
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_REPLAY", "aggregate metrics do not replay from cells")
    expected_status = COMPLETE_STATUS if not expected_aggregate["n_undefined_cells"] else HOLD_STATUS
    if payload["status"] != expected_status:
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_REPLAY", "diagnostic status does not replay")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": expected_status,
        "diagnostic_role": DIAGNOSTIC_ROLE,
        "train_development_outcomes_sha256": data_sha,
        "estimand_registry_sha256": registry_sha,
        "registered_cell_universe_sha256": cell_universe_sha,
        "registry_primary_entry_set_sha256": entry_set_sha,
        "input_binding": binding,
        "study_endpoint_scope": scope,
        "policies": dict(_POLICIES),
        "cells": cells,
        "aggregate": expected_aggregate,
        "boundaries": dict(_BOUNDARIES),
    }


def canonical_development_learnability_diagnostic_bytes(payload: Mapping) -> bytes:
    """Canonicalize a structurally valid payload; this is not exact input replay."""

    return canonical_json_bytes(validate_development_learnability_diagnostic(payload))


def validate_development_learnability_structure_only(payload: Mapping) -> dict:
    """Explicit structural-only alias for the compatibility validator."""

    return validate_development_learnability_diagnostic(payload)


def _assert_matches_expected_inputs(
    payload: Mapping,
    *,
    expected_rows: Sequence[Mapping],
    expected_estimand_registry: Mapping,
    expected_train_development_outcomes_sha256: str,
    expected_estimand_registry_sha256: str,
    expected_registered_cells: Sequence[Mapping],
    expected_registered_cell_universe_sha256: str,
) -> dict:
    expected = build_development_learnability_diagnostic(
        expected_rows,
        expected_estimand_registry,
        train_development_outcomes_sha256=expected_train_development_outcomes_sha256,
        estimand_registry_sha256=expected_estimand_registry_sha256,
        registered_cells=expected_registered_cells,
        registered_cell_universe_sha256=(
            expected_registered_cell_universe_sha256
        ),
    )
    actual = validate_development_learnability_diagnostic(payload)
    if canonical_json_bytes(actual) != canonical_json_bytes(expected):
        _hold(
            "HOLD_DEVELOPMENT_LEARNABILITY_INPUT_BINDING",
            "diagnostic does not exactly replay from expected train/development inputs",
        )
    return actual


def write_development_learnability_diagnostic_exclusive(
    path: Path,
    payload: Mapping,
    *,
    expected_rows: Sequence[Mapping],
    expected_estimand_registry: Mapping,
    expected_train_development_outcomes_sha256: str,
    expected_estimand_registry_sha256: str,
    expected_registered_cells: Sequence[Mapping],
    expected_registered_cell_universe_sha256: str,
) -> str:
    validated = _assert_matches_expected_inputs(
        payload,
        expected_rows=expected_rows,
        expected_estimand_registry=expected_estimand_registry,
        expected_train_development_outcomes_sha256=(
            expected_train_development_outcomes_sha256
        ),
        expected_estimand_registry_sha256=expected_estimand_registry_sha256,
        expected_registered_cells=expected_registered_cells,
        expected_registered_cell_universe_sha256=(
            expected_registered_cell_universe_sha256
        ),
    )
    try:
        return atomic_write_json_exclusive(
            Path(path), validated, validator=validate_development_learnability_diagnostic
        )
    except ArtifactIOError as exc:
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_ARTIFACT_IO", str(exc))


def _strict_json_object(raw: bytes) -> dict:
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicates)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_ARTIFACT_IO", "diagnostic JSON is invalid", error=str(exc))
    if not isinstance(value, dict):
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_SCHEMA", "diagnostic JSON must be an object")
    return value


def load_development_learnability_diagnostic_exact(
    path: Path,
    expected_sha256: str,
    *,
    expected_rows: Sequence[Mapping],
    expected_estimand_registry: Mapping,
    expected_train_development_outcomes_sha256: str,
    expected_estimand_registry_sha256: str,
    expected_registered_cells: Sequence[Mapping],
    expected_registered_cell_universe_sha256: str,
) -> dict:
    try:
        raw = read_exact_regular_file(Path(path), _sha256(expected_sha256, label="artifact SHA-256"))
    except ArtifactIOError as exc:
        _hold("HOLD_DEVELOPMENT_LEARNABILITY_ARTIFACT_IO", str(exc))
    payload = _strict_json_object(raw)
    validated = _assert_matches_expected_inputs(
        payload,
        expected_rows=expected_rows,
        expected_estimand_registry=expected_estimand_registry,
        expected_train_development_outcomes_sha256=(
            expected_train_development_outcomes_sha256
        ),
        expected_estimand_registry_sha256=expected_estimand_registry_sha256,
        expected_registered_cells=expected_registered_cells,
        expected_registered_cell_universe_sha256=(
            expected_registered_cell_universe_sha256
        ),
    )
    if canonical_json_bytes(validated) != raw:
        _hold(
            "HOLD_DEVELOPMENT_LEARNABILITY_ARTIFACT_IO",
            "diagnostic artifact is not canonical exact JSON",
        )
    return validated


__all__ = [
    "ALLOWED_EDIT_COUNTS",
    "COMPLETE_STATUS",
    "DIAGNOSTIC_ROLE",
    "FORMAL_CELL_FIELDS",
    "HOLD_STATUS",
    "MINIMUM_COMPONENTS_PER_RELEASED_CELL_PARTITION",
    "NOISE_CEILING_STATUS",
    "REGISTERED_CELL_UNIVERSE_SCHEMA",
    "SCHEMA_VERSION",
    "SIGN_GATE",
    "build_development_learnability_diagnostic",
    "canonical_development_learnability_diagnostic_bytes",
    "canonical_estimand_registry_semantic_sha256",
    "canonical_registered_cell_universe_semantic_sha256",
    "canonical_train_development_outcomes_semantic_sha256",
    "load_development_learnability_diagnostic_exact",
    "validate_development_learnability_diagnostic",
    "validate_development_learnability_structure_only",
    "write_development_learnability_diagnostic_exclusive",
]
