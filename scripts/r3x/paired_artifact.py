"""Immutable, recomputable R3X strongest-baseline comparison artifacts.

This module deliberately does not infer a multiple-comparison family from the
observed results.  The exact member list, primary member, inferential settings,
candidate, baseline, and information set must already be present in a frozen
policy.  The primary sign comparison is recomputed from exact ordinary rows and
prediction vectors; every additional family member must carry both a raw
p-value and an independently hashed evidence artifact.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from .metrics import MetricProtocolError
from .artifact_io import ArtifactIOError, atomic_publish_bytes, read_exact_regular_file
from .protocol import (
    ProtocolHoldError,
    assert_ordinary_path,
    assert_ordinary_record,
    canonical_sha256,
)
from .statistics import (
    PRIMARY_PAIRED_METRIC_ID,
    _paired_sign_scores,
    holm_adjust,
    paired_primary_sign_bootstrap,
    paired_primary_sign_permutation,
)


SCHEMA = "r3x_paired_comparison_result_v2"
VERIFIED_REFERENCE_SCHEMA = "r3x_verified_paired_comparison_reference_v1"
POLICY_SCHEMA = "r3x_paired_strongest_baseline_comparison_policy_v1"
INFORMATION_SET = "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$")


class PairedArtifactProtocolError(RuntimeError):
    """Fail-closed error for malformed or non-recomputable paired evidence."""


def _fail(message: str) -> None:
    raise PairedArtifactProtocolError(message)


def _plain_mapping(value: Any, label: str) -> dict:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object")
    return dict(value)


def _exact_keys(value: Mapping, required: set[str], label: str) -> None:
    non_string = [repr(key) for key in value if not isinstance(key, str)]
    if non_string:
        _fail(f"{label} contains non-string keys: {sorted(non_string)}")
    actual = set(value)
    missing = sorted(required - actual)
    extra = sorted(actual - required)
    if missing or extra:
        _fail(f"{label} keys mismatch; missing={missing}, extra={extra}")


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        _fail(f"{label} must be a non-empty stable identifier")
    return value


def _nonempty_text(value: Any, label: str, *, maximum_length: int = 1000) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum_length
        or any(ord(character) < 32 for character in value)
    ):
        _fail(f"{label} must be non-empty bounded text without control characters")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _finite_probability(value: Any, label: str, *, open_interval: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        _fail(f"{label} must be finite")
    valid = 0.0 < number < 1.0 if open_interval else 0.0 <= number <= 1.0
    if not valid:
        interval = "(0,1)" if open_interval else "[0,1]"
        _fail(f"{label} must be in {interval}")
    return number


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        _fail(f"{label} must be finite")
    return number


def _positive_int(value: Any, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{label} must be an integer >= {minimum}")
    return value


def _stable_row_id(row: Mapping) -> str:
    value = str(row.get("r3_input_row_sha256") or row.get("record_id") or "")
    if not value:
        _fail("evaluation rows require stable identities")
    return value


def _row_bindings(rows: Sequence[Mapping]) -> dict:
    normalized_rows = []
    row_ids = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            _fail(f"evaluation row {index} must be an object")
        try:
            assert_ordinary_record(row, purpose="paired comparison row")
        except ProtocolHoldError as exc:
            raise PairedArtifactProtocolError(str(exc)) from exc
        row_id = _stable_row_id(row)
        row_ids.append(row_id)
        normalized_rows.append({
            "row_id": row_id,
            "study": row.get("study"),
            "estimand": row.get("estimand_id") or row.get("endpoint_id") or row.get("endpoint"),
            "publication_context": row.get("context_id_v2") or row.get("publication_context_id"),
            "edit_count": row.get("actual_edit_count") or row.get("edit_count"),
            "source_component_id": row.get("source_component_id"),
            "delta": row.get("delta"),
        })
    if not row_ids:
        _fail("paired comparison refuses an empty evaluation row set")
    if len(row_ids) != len(set(row_ids)):
        _fail("paired comparison rows contain duplicate stable identities")
    try:
        return {
            "n_rows": len(row_ids),
            "row_order_sha256": canonical_sha256(row_ids),
            "row_set_sha256": canonical_sha256(sorted(row_ids)),
            "evaluation_rows_sha256": canonical_sha256(normalized_rows),
        }
    except ProtocolHoldError as exc:
        raise PairedArtifactProtocolError(
            f"evaluation rows are not canonical finite JSON: {exc}"
        ) from exc


def _prediction_values(values: Sequence[float], n_rows: int, label: str) -> list[float]:
    if isinstance(values, (str, bytes, bytearray)):
        _fail(f"{label} must be a numeric vector")
    try:
        result = list(values)
    except TypeError as exc:
        raise PairedArtifactProtocolError(f"{label} must be a numeric vector") from exc
    if len(result) != n_rows:
        _fail(f"{label} does not align one-to-one with evaluation rows")
    normalized = []
    for index, value in enumerate(result):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _fail(f"{label}[{index}] must be numeric")
        number = float(value)
        if not math.isfinite(number):
            _fail(f"{label}[{index}] must be finite")
        normalized.append(number)
    return normalized


def _authority_hashes(values: Mapping) -> dict[str, str]:
    source = _plain_mapping(values, "authority_sha256")
    if not source:
        _fail("authority_sha256 must be non-empty")
    result = {}
    for key, value in source.items():
        name = _identifier(key, "authority_sha256 key")
        if not name.endswith("_sha256"):
            _fail(f"authority hash key must end in _sha256: {name!r}")
        result[name] = _sha256(value, f"authority_sha256.{name}")
    return {key: result[key] for key in sorted(result)}


def _policy_settings(policy: Mapping) -> dict:
    policy = _plain_mapping(policy, "paired comparison policy")
    try:
        assert_ordinary_record(policy, purpose="paired comparison policy")
    except ProtocolHoldError as exc:
        raise PairedArtifactProtocolError(str(exc)) from exc
    if policy.get("schema_version") != POLICY_SCHEMA or policy.get("status") != "FROZEN":
        _fail("paired comparison policy must be the frozen v1 authority")

    members_raw = policy.get("comparison_family_member_ids")
    if not isinstance(members_raw, list) or not members_raw:
        _fail("policy must freeze a non-empty comparison_family_member_ids list")
    members = [_identifier(value, "comparison family member") for value in members_raw]
    if len(members) != len(set(members)):
        _fail("comparison_family_member_ids contains duplicates")
    primary_member = _identifier(
        policy.get("primary_comparison_member_id"), "primary_comparison_member_id"
    )
    if primary_member not in members:
        _fail("primary_comparison_member_id is absent from the frozen family")

    inference = _plain_mapping(policy.get("primary_sign_inference"), "primary_sign_inference")
    _exact_keys(
        inference,
        {"metric_id", "alpha", "confidence", "bootstrap", "permutation"},
        "primary_sign_inference",
    )
    if inference.get("metric_id") != PRIMARY_PAIRED_METRIC_ID:
        _fail("policy primary metric is not the frozen component/cell-balanced sign utility")
    alpha = _finite_probability(inference.get("alpha"), "primary_sign_inference.alpha", open_interval=True)
    confidence = _finite_probability(
        inference.get("confidence"), "primary_sign_inference.confidence", open_interval=True
    )
    if not math.isclose(confidence, 1.0 - alpha, rel_tol=0.0, abs_tol=1e-12):
        _fail("bootstrap confidence must equal 1 - family alpha")

    bootstrap = _plain_mapping(inference.get("bootstrap"), "primary_sign_inference.bootstrap")
    _exact_keys(
        bootstrap,
        {"n_resamples", "seed", "min_components_per_cell", "max_attempt_multiplier"},
        "primary_sign_inference.bootstrap",
    )
    bootstrap_settings = {
        "n_resamples": _positive_int(
            bootstrap.get("n_resamples"), "bootstrap.n_resamples", minimum=2000
        ),
        "seed": _positive_int(bootstrap.get("seed"), "bootstrap.seed", minimum=0),
        "min_components_per_cell": _positive_int(
            bootstrap.get("min_components_per_cell"), "bootstrap.min_components_per_cell", minimum=2
        ),
        "max_attempt_multiplier": _positive_int(
            bootstrap.get("max_attempt_multiplier"), "bootstrap.max_attempt_multiplier"
        ),
    }

    permutation = _plain_mapping(inference.get("permutation"), "primary_sign_inference.permutation")
    _exact_keys(
        permutation,
        {"n_permutations", "seed", "exact_component_limit", "min_components_per_cell"},
        "primary_sign_inference.permutation",
    )
    permutation_settings = {
        "n_permutations": _positive_int(
            permutation.get("n_permutations"), "permutation.n_permutations", minimum=10000
        ),
        "seed": _positive_int(permutation.get("seed"), "permutation.seed", minimum=0),
        "exact_component_limit": _positive_int(
            permutation.get("exact_component_limit"), "permutation.exact_component_limit"
        ),
        "min_components_per_cell": _positive_int(
            permutation.get("min_components_per_cell"),
            "permutation.min_components_per_cell",
            minimum=10,
        ),
    }
    if permutation_settings["exact_component_limit"] != 20:
        _fail("permutation exact_component_limit must remain frozen at 20")
    if (
        bootstrap_settings["min_components_per_cell"]
        != permutation_settings["min_components_per_cell"]
    ):
        _fail("bootstrap and permutation minimum cell support must match")
    if bootstrap_settings["min_components_per_cell"] < 10:
        _fail("publication paired inference requires at least 10 components per cell")

    information_set = policy.get("information_set")
    if information_set != INFORMATION_SET:
        _fail("paired strongest-baseline policy must use the paired no-anchor information set")
    return {
        "policy_id": _identifier(policy.get("policy_id"), "policy_id"),
        "candidate_model_id": _identifier(
            policy.get("primary_candidate_model_id"), "primary_candidate_model_id"
        ),
        "baseline_model_id": _identifier(
            policy.get("strongest_baseline_id"), "strongest_baseline_id"
        ),
        "information_set": information_set,
        "family_id": _identifier(policy.get("comparison_family_id"), "comparison_family_id"),
        "family_member_ids": members,
        "primary_member_id": primary_member,
        "alpha": alpha,
        "confidence": confidence,
        "bootstrap": bootstrap_settings,
        "permutation": permutation_settings,
    }


def _additional_family_evidence(
    settings: Mapping,
    raw_p_values: Mapping,
    evidence_sha256: Mapping,
) -> tuple[dict[str, float], dict[str, str]]:
    expected = set(settings["family_member_ids"]) - {settings["primary_member_id"]}
    raw = _plain_mapping(raw_p_values, "additional_family_raw_p_values")
    evidence = _plain_mapping(evidence_sha256, "additional_family_evidence_sha256")
    for label, values in (("additional raw p-value", raw), ("additional evidence", evidence)):
        non_string = [repr(key) for key in values if not isinstance(key, str)]
        if non_string:
            _fail(f"{label} map contains non-string keys: {sorted(non_string)}")
    raw_keys = set(raw)
    evidence_keys = set(evidence)
    for label, actual in (("additional raw p-value", raw_keys), ("additional evidence", evidence_keys)):
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            _fail(f"{label} family membership mismatch; missing={missing}, extra={extra}")
    normalized_p = {
        member: _finite_probability(raw[member], f"raw p-value for {member}")
        for member in sorted(expected)
    }
    normalized_evidence = {
        member: _sha256(evidence[member], f"evidence SHA-256 for {member}")
        for member in sorted(expected)
    }
    return normalized_p, normalized_evidence


def _cell_support(score: Mapping) -> dict[str, dict]:
    result = {}
    for cell_id, cell in sorted(score["cells"].items()):
        components = cell["components"]
        component_ids = sorted(components)
        result[cell_id] = {
            "n_components": int(cell["n_components"]),
            "n_nonzero_rows": int(sum(value["n_nonzero_rows"] for value in components.values())),
            "component_ids_sha256": canonical_sha256(component_ids),
            "candidate_raw_sign_utility": float(cell["candidate"]),
            "baseline_raw_sign_utility": float(cell["baseline"]),
            "candidate_minus_baseline": float(cell["improvement"]),
        }
    return result


def build_paired_comparison_artifact(
    rows: Sequence[Mapping],
    candidate_prediction: Sequence[float],
    baseline_prediction: Sequence[float],
    *,
    policy: Mapping,
    policy_sha256: str,
    authority_sha256: Mapping,
    candidate_model_id: str,
    baseline_model_id: str,
    candidate_information_set_fingerprint: str,
    baseline_information_set_fingerprint: str,
    candidate_prediction_artifact_sha256: str,
    baseline_prediction_artifact_sha256: str,
    additional_family_raw_p_values: Mapping[str, float],
    additional_family_evidence_sha256: Mapping[str, str],
) -> dict:
    """Build a fully recomputable ordinary-only paired comparison payload."""

    settings = _policy_settings(policy)
    candidate_model_id = _identifier(candidate_model_id, "candidate_model_id")
    baseline_model_id = _identifier(baseline_model_id, "baseline_model_id")
    if candidate_model_id != settings["candidate_model_id"]:
        _fail("candidate_model_id does not match the frozen policy")
    if baseline_model_id != settings["baseline_model_id"]:
        _fail("baseline_model_id does not match the frozen policy")

    candidate_fingerprint = _sha256(
        candidate_information_set_fingerprint, "candidate information-set fingerprint"
    )
    baseline_fingerprint = _sha256(
        baseline_information_set_fingerprint, "baseline information-set fingerprint"
    )
    if candidate_fingerprint != baseline_fingerprint:
        _fail("candidate and baseline information-set fingerprints differ")

    row_binding = _row_bindings(rows)
    candidate_values = _prediction_values(
        candidate_prediction, row_binding["n_rows"], "candidate_prediction"
    )
    baseline_values = _prediction_values(
        baseline_prediction, row_binding["n_rows"], "baseline_prediction"
    )
    additional_p, additional_evidence = _additional_family_evidence(
        settings, additional_family_raw_p_values, additional_family_evidence_sha256
    )
    try:
        score = _paired_sign_scores(rows, candidate_values, baseline_values)
        bootstrap = paired_primary_sign_bootstrap(
            rows,
            candidate_values,
            baseline_values,
            confidence=settings["confidence"],
            **settings["bootstrap"],
        )
        permutation = paired_primary_sign_permutation(
            rows,
            candidate_values,
            baseline_values,
            **settings["permutation"],
        )
        raw_p_values = dict(additional_p)
        raw_p_values[settings["primary_member_id"]] = float(permutation["p_value_two_sided"])
        raw_p_values = {member: raw_p_values[member] for member in settings["family_member_ids"]}
        adjusted = holm_adjust(raw_p_values)
    except (MetricProtocolError, ValueError, TypeError) as exc:
        raise PairedArtifactProtocolError(f"paired statistics are undefined: {exc}") from exc

    primary_member = settings["primary_member_id"]
    ci = [float(value) for value in bootstrap["confidence_interval_oriented_improvement"]]
    primary_adjusted = float(adjusted[primary_member])
    passed = bool(ci[0] > 0.0 and primary_adjusted <= settings["alpha"])
    family_evidence = dict(additional_evidence)
    family_evidence[primary_member] = canonical_sha256({
        "candidate_prediction_artifact_sha256": _sha256(
            candidate_prediction_artifact_sha256, "candidate prediction artifact SHA-256"
        ),
        "baseline_prediction_artifact_sha256": _sha256(
            baseline_prediction_artifact_sha256, "baseline prediction artifact SHA-256"
        ),
        "evaluation_rows_sha256": row_binding["evaluation_rows_sha256"],
    })
    family_evidence = {
        member: family_evidence[member] for member in settings["family_member_ids"]
    }
    authority_hashes = _authority_hashes(authority_sha256)

    payload = {
        "schema_version": SCHEMA,
        "status": (
            "PASS_PAIRED_STRONGEST_BASELINE"
            if passed
            else "FAIL_PAIRED_STRONGEST_BASELINE"
        ),
        "policy_id": settings["policy_id"],
        "candidate_model_id": candidate_model_id,
        "baseline_model_id": baseline_model_id,
        "information_set": settings["information_set"],
        "candidate_information_set_fingerprint": candidate_fingerprint,
        "baseline_information_set_fingerprint": baseline_fingerprint,
        "candidate_row_order_sha256": row_binding["row_order_sha256"],
        "baseline_row_order_sha256": row_binding["row_order_sha256"],
        "candidate_row_set_sha256": row_binding["row_set_sha256"],
        "baseline_row_set_sha256": row_binding["row_set_sha256"],
        "n_paired_rows": row_binding["n_rows"],
        "evaluation_rows_sha256": row_binding["evaluation_rows_sha256"],
        "candidate_prediction_values_sha256": canonical_sha256(candidate_values),
        "baseline_prediction_values_sha256": canonical_sha256(baseline_values),
        "bindings": {
            "policy_sha256": _sha256(policy_sha256, "policy_sha256"),
            "authority_sha256": authority_hashes,
            "candidate_prediction_artifact_sha256": _sha256(
                candidate_prediction_artifact_sha256,
                "candidate_prediction_artifact_sha256",
            ),
            "baseline_prediction_artifact_sha256": _sha256(
                baseline_prediction_artifact_sha256,
                "baseline_prediction_artifact_sha256",
            ),
        },
        "multiple_comparison_family": {
            "family_id": settings["family_id"],
            "frozen_member_ids": list(settings["family_member_ids"]),
            "frozen_membership_sha256": canonical_sha256(settings["family_member_ids"]),
            "primary_member_id": primary_member,
            "raw_p_values_two_sided": raw_p_values,
            "holm_adjusted_p_values": adjusted,
            "member_evidence_sha256": family_evidence,
            "alpha": settings["alpha"],
        },
        "primary_sign_utility": {
            "metric_id": PRIMARY_PAIRED_METRIC_ID,
            "orientation": "LARGER_UTILITY_IS_BETTER",
            "candidate_raw_value": float(score["candidate"]),
            "baseline_raw_value": float(score["baseline"]),
            "candidate_minus_baseline": float(score["improvement"]),
            "confidence_interval_candidate_minus_baseline": ci,
            "raw_p_value_two_sided": float(permutation["p_value_two_sided"]),
            "holm_adjusted_p_value": primary_adjusted,
            "alpha": settings["alpha"],
            "confidence": settings["confidence"],
            "sign_decision_rule": score["sign_decision_rule"],
            "zero_policy": score["zero_policy"],
            "row_within_component_weighting": score["row_within_component_weighting"],
            "component_within_cell_weighting": score["component_within_cell_weighting"],
            "cell_weighting": score["cell_weighting"],
            "cell_definition": score["cell_definition"],
            "cell_support": _cell_support(score),
            "bootstrap": bootstrap,
            "permutation": permutation,
        },
        "sealed_access": False,
        "downstream_unlock": False,
    }
    _validate_shape(payload)
    return payload


_TOP_LEVEL_KEYS = {
    "schema_version", "status", "policy_id", "candidate_model_id",
    "baseline_model_id", "information_set", "candidate_information_set_fingerprint",
    "baseline_information_set_fingerprint", "candidate_row_order_sha256",
    "baseline_row_order_sha256", "candidate_row_set_sha256", "baseline_row_set_sha256",
    "n_paired_rows", "evaluation_rows_sha256", "candidate_prediction_values_sha256",
    "baseline_prediction_values_sha256", "bindings", "multiple_comparison_family",
    "primary_sign_utility", "sealed_access", "downstream_unlock",
}

_BINDING_KEYS = {
    "policy_sha256", "authority_sha256", "candidate_prediction_artifact_sha256",
    "baseline_prediction_artifact_sha256",
}
_FAMILY_KEYS = {
    "family_id", "frozen_member_ids", "frozen_membership_sha256",
    "primary_member_id", "raw_p_values_two_sided", "holm_adjusted_p_values",
    "member_evidence_sha256", "alpha",
}
_PRIMARY_KEYS = {
    "metric_id", "orientation", "candidate_raw_value", "baseline_raw_value",
    "candidate_minus_baseline", "confidence_interval_candidate_minus_baseline",
    "raw_p_value_two_sided", "holm_adjusted_p_value", "alpha", "confidence",
    "sign_decision_rule", "zero_policy", "row_within_component_weighting",
    "component_within_cell_weighting", "cell_weighting", "cell_definition",
    "cell_support", "bootstrap", "permutation",
}
_CELL_SUPPORT_KEYS = {
    "n_components", "n_nonzero_rows", "component_ids_sha256",
    "candidate_raw_sign_utility", "baseline_raw_sign_utility",
    "candidate_minus_baseline",
}


def _validate_shape(payload: Mapping) -> None:
    payload = _plain_mapping(payload, "paired artifact")
    _exact_keys(payload, _TOP_LEVEL_KEYS, "paired artifact")
    if payload.get("schema_version") != SCHEMA:
        _fail("unsupported paired artifact schema")
    if payload.get("status") not in {
        "PASS_PAIRED_STRONGEST_BASELINE", "FAIL_PAIRED_STRONGEST_BASELINE"
    }:
        _fail("paired artifact status is invalid")
    if payload.get("sealed_access") is not False or payload.get("downstream_unlock") is not False:
        _fail("paired artifact cannot grant sealed access or downstream unlock")
    if payload.get("information_set") != INFORMATION_SET:
        _fail("paired artifact information set is invalid")
    candidate_id = _identifier(payload.get("candidate_model_id"), "candidate_model_id")
    baseline_id = _identifier(payload.get("baseline_model_id"), "baseline_model_id")
    if candidate_id == baseline_id:
        _fail("candidate and baseline model IDs must differ")
    _identifier(payload.get("policy_id"), "policy_id")
    for field in (
        "candidate_information_set_fingerprint", "baseline_information_set_fingerprint",
        "candidate_row_order_sha256", "baseline_row_order_sha256",
        "candidate_row_set_sha256", "baseline_row_set_sha256",
        "evaluation_rows_sha256", "candidate_prediction_values_sha256",
        "baseline_prediction_values_sha256",
    ):
        _sha256(payload.get(field), field)
    if payload["candidate_information_set_fingerprint"] != payload["baseline_information_set_fingerprint"]:
        _fail("paired artifact information-set fingerprints differ")
    if payload["candidate_row_order_sha256"] != payload["baseline_row_order_sha256"]:
        _fail("paired artifact row-order hashes differ")
    if payload["candidate_row_set_sha256"] != payload["baseline_row_set_sha256"]:
        _fail("paired artifact row-set hashes differ")
    _positive_int(payload.get("n_paired_rows"), "n_paired_rows")

    bindings = _plain_mapping(payload.get("bindings"), "bindings")
    _exact_keys(bindings, _BINDING_KEYS, "bindings")
    _sha256(bindings.get("policy_sha256"), "bindings.policy_sha256")
    _sha256(
        bindings.get("candidate_prediction_artifact_sha256"),
        "bindings.candidate_prediction_artifact_sha256",
    )
    _sha256(
        bindings.get("baseline_prediction_artifact_sha256"),
        "bindings.baseline_prediction_artifact_sha256",
    )
    _authority_hashes(bindings.get("authority_sha256"))

    family = _plain_mapping(payload.get("multiple_comparison_family"), "multiple_comparison_family")
    _exact_keys(family, _FAMILY_KEYS, "multiple_comparison_family")
    _identifier(family.get("family_id"), "multiple_comparison_family.family_id")
    members = family.get("frozen_member_ids")
    if not isinstance(members, list) or not members:
        _fail("multiple_comparison_family.frozen_member_ids must be a non-empty list")
    members = [_identifier(member, "frozen family member") for member in members]
    if len(members) != len(set(members)):
        _fail("multiple_comparison_family.frozen_member_ids contains duplicates")
    primary_member = _identifier(
        family.get("primary_member_id"), "multiple_comparison_family.primary_member_id"
    )
    if primary_member not in members:
        _fail("multiple-comparison primary member is absent from its frozen family")
    if family.get("frozen_membership_sha256") != canonical_sha256(members):
        _fail("multiple-comparison frozen membership hash is inconsistent")
    raw_p = _plain_mapping(family.get("raw_p_values_two_sided"), "family raw p-values")
    adjusted_p = _plain_mapping(family.get("holm_adjusted_p_values"), "family Holm p-values")
    evidence = _plain_mapping(family.get("member_evidence_sha256"), "family evidence")
    member_keys = set(members)
    _exact_keys(raw_p, member_keys, "family raw p-values")
    _exact_keys(adjusted_p, member_keys, "family Holm p-values")
    _exact_keys(evidence, member_keys, "family evidence")
    checked_raw = {
        member: _finite_probability(raw_p[member], f"family raw p-value {member}")
        for member in members
    }
    checked_adjusted = {
        member: _finite_probability(adjusted_p[member], f"family Holm p-value {member}")
        for member in members
    }
    if checked_adjusted != holm_adjust(checked_raw):
        _fail("family Holm p-values do not exactly match the frozen raw p-values")
    for member in members:
        _sha256(evidence[member], f"family evidence SHA-256 {member}")
    expected_primary_evidence = canonical_sha256({
        "candidate_prediction_artifact_sha256": bindings[
            "candidate_prediction_artifact_sha256"
        ],
        "baseline_prediction_artifact_sha256": bindings[
            "baseline_prediction_artifact_sha256"
        ],
        "evaluation_rows_sha256": payload["evaluation_rows_sha256"],
    })
    if evidence[primary_member] != expected_primary_evidence:
        _fail("primary family evidence is inconsistent with prediction and row bindings")
    family_alpha = _finite_probability(
        family.get("alpha"), "multiple_comparison_family.alpha", open_interval=True
    )

    primary = _plain_mapping(payload.get("primary_sign_utility"), "primary_sign_utility")
    _exact_keys(primary, _PRIMARY_KEYS, "primary_sign_utility")
    if primary.get("metric_id") != PRIMARY_PAIRED_METRIC_ID:
        _fail("primary sign utility metric ID is invalid")
    if primary.get("orientation") != "LARGER_UTILITY_IS_BETTER":
        _fail("primary sign utility orientation is invalid")
    candidate = _finite_probability(primary.get("candidate_raw_value"), "candidate_raw_value")
    baseline = _finite_probability(primary.get("baseline_raw_value"), "baseline_raw_value")
    improvement = _finite_number(primary.get("candidate_minus_baseline"), "candidate_minus_baseline")
    if not math.isclose(improvement, candidate - baseline, rel_tol=0.0, abs_tol=1e-12):
        _fail("primary sign improvement is inconsistent with the raw utilities")
    ci = primary.get("confidence_interval_candidate_minus_baseline")
    if not isinstance(ci, list) or len(ci) != 2:
        _fail("primary sign confidence interval must contain exactly two endpoints")
    ci = [_finite_number(value, "primary sign confidence interval") for value in ci]
    if ci[0] > ci[1]:
        _fail("primary sign confidence interval endpoints are reversed")
    raw_primary = _finite_probability(primary.get("raw_p_value_two_sided"), "primary raw p-value")
    holm_primary = _finite_probability(primary.get("holm_adjusted_p_value"), "primary Holm p-value")
    primary_alpha = _finite_probability(primary.get("alpha"), "primary alpha", open_interval=True)
    confidence = _finite_probability(primary.get("confidence"), "primary confidence", open_interval=True)
    if not math.isclose(confidence, 1.0 - primary_alpha, rel_tol=0.0, abs_tol=1e-12):
        _fail("primary confidence is inconsistent with alpha")
    if primary_alpha != family_alpha:
        _fail("primary alpha differs from the multiple-comparison family alpha")
    if raw_primary != checked_raw[primary_member] or holm_primary != checked_adjusted[primary_member]:
        _fail("primary p-values differ from their frozen family entries")
    expected_status = (
        "PASS_PAIRED_STRONGEST_BASELINE"
        if ci[0] > 0.0 and holm_primary <= primary_alpha
        else "FAIL_PAIRED_STRONGEST_BASELINE"
    )
    if payload.get("status") != expected_status:
        _fail("paired artifact status is inconsistent with CI and Holm evidence")

    cells = _plain_mapping(primary.get("cell_support"), "primary_sign_utility.cell_support")
    if not cells:
        _fail("primary sign utility requires per-cell support")
    for cell_id, cell_value in cells.items():
        _nonempty_text(cell_id, "cell support ID")
        cell = _plain_mapping(cell_value, f"cell support {cell_id}")
        _exact_keys(cell, _CELL_SUPPORT_KEYS, f"cell support {cell_id}")
        n_components = _positive_int(cell.get("n_components"), f"{cell_id}.n_components")
        n_nonzero = _positive_int(cell.get("n_nonzero_rows"), f"{cell_id}.n_nonzero_rows")
        if n_nonzero < n_components:
            _fail(f"cell {cell_id} has fewer informative rows than components")
        _sha256(cell.get("component_ids_sha256"), f"{cell_id}.component_ids_sha256")
        cell_candidate = _finite_probability(
            cell.get("candidate_raw_sign_utility"), f"{cell_id}.candidate_raw_sign_utility"
        )
        cell_baseline = _finite_probability(
            cell.get("baseline_raw_sign_utility"), f"{cell_id}.baseline_raw_sign_utility"
        )
        cell_improvement = _finite_number(
            cell.get("candidate_minus_baseline"), f"{cell_id}.candidate_minus_baseline"
        )
        if not math.isclose(
            cell_improvement, cell_candidate - cell_baseline, rel_tol=0.0, abs_tol=1e-12
        ):
            _fail(f"cell {cell_id} improvement is inconsistent with raw utilities")
    bootstrap = _plain_mapping(primary.get("bootstrap"), "primary bootstrap")
    permutation = _plain_mapping(primary.get("permutation"), "primary permutation")
    if bootstrap.get("metric_id") != PRIMARY_PAIRED_METRIC_ID:
        _fail("primary bootstrap metric ID is invalid")
    if permutation.get("metric_id") != PRIMARY_PAIRED_METRIC_ID:
        _fail("primary permutation metric ID is invalid")
    if bootstrap.get("observed_oriented_improvement") != improvement:
        _fail("primary bootstrap observed improvement is inconsistent")
    if bootstrap.get("confidence_interval_oriented_improvement") != ci:
        _fail("primary bootstrap confidence interval is inconsistent")
    if permutation.get("observed_oriented_improvement") != improvement:
        _fail("primary permutation observed improvement is inconsistent")
    if permutation.get("p_value_two_sided") != raw_primary:
        _fail("primary permutation p-value is inconsistent")


def canonical_paired_artifact_bytes(payload: Mapping) -> bytes:
    _validate_shape(payload)
    try:
        return (
            json.dumps(
                dict(payload), sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PairedArtifactProtocolError(
            f"paired artifact is not canonical finite JSON: {exc}"
        ) from exc


def build_verified_paired_reference(
    *, artifact_relative_path: str, artifact_sha256: str, artifact_payload: Mapping
) -> dict:
    if not isinstance(artifact_relative_path, str) or not artifact_relative_path:
        _fail("paired artifact relative path is required")
    path = PurePosixPath(artifact_relative_path)
    if (
        path.is_absolute() or path.as_posix() != artifact_relative_path
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        _fail("paired artifact reference path must be safe and canonical")
    raw = canonical_paired_artifact_bytes(artifact_payload)
    digest = _sha256(artifact_sha256, "paired artifact file SHA-256")
    if hashlib.sha256(raw).hexdigest() != digest:
        _fail("paired artifact reference SHA-256 does not match canonical payload bytes")
    return {
        "schema_version": VERIFIED_REFERENCE_SCHEMA,
        "verification_status": "EXACT_BYTES_RECOMPUTED_FROM_HASH_BOUND_ROWS_AND_PREDICTIONS",
        "artifact_relative_path": artifact_relative_path,
        "artifact_sha256": digest,
        "artifact_payload": dict(artifact_payload),
    }


def validate_verified_paired_reference(reference: Mapping) -> dict:
    """Validate only the canonical reference envelope and embedded payload.

    This function deliberately does *not* establish that the referenced file
    exists.  GO decisions must use :func:`load_verified_paired_reference_exact`
    so an in-memory payload cannot substitute for an immutable run artifact.
    """

    reference = _plain_mapping(reference, "verified paired comparison reference")
    _exact_keys(reference, {
        "schema_version", "verification_status", "artifact_relative_path",
        "artifact_sha256", "artifact_payload",
    }, "verified paired comparison reference")
    if reference.get("schema_version") != VERIFIED_REFERENCE_SCHEMA:
        _fail("verified paired comparison reference schema is unsupported")
    if reference.get("verification_status") != (
        "EXACT_BYTES_RECOMPUTED_FROM_HASH_BOUND_ROWS_AND_PREDICTIONS"
    ):
        _fail("paired comparison was not exact-byte recomputed")
    return build_verified_paired_reference(
        artifact_relative_path=reference.get("artifact_relative_path"),
        artifact_sha256=reference.get("artifact_sha256"),
        artifact_payload=reference.get("artifact_payload"),
    )


def _resolve_run_artifact_path(run_root: Path, artifact_relative_path: str) -> Path:
    """Resolve a canonical artifact path without following directory aliases."""

    root = Path(run_root)
    try:
        assert_ordinary_path(root, purpose="R3X paired comparison run root")
    except ProtocolHoldError as exc:
        raise PairedArtifactProtocolError(str(exc)) from exc
    if root.is_symlink():
        _fail("paired comparison run root cannot be a symlink")
    try:
        root_metadata = root.stat()
    except FileNotFoundError as exc:
        raise PairedArtifactProtocolError(
            f"paired comparison run root does not exist: {root}"
        ) from exc
    except OSError as exc:
        raise PairedArtifactProtocolError(
            f"paired comparison run root cannot be inspected: {root}: {exc}"
        ) from exc
    if not stat.S_ISDIR(root_metadata.st_mode):
        _fail("paired comparison run root must be a real directory")

    relative = PurePosixPath(artifact_relative_path)
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            _fail("paired artifact path cannot traverse a symlinked directory")
        try:
            metadata = current.stat()
        except FileNotFoundError as exc:
            raise PairedArtifactProtocolError(
                f"paired artifact parent does not exist: {current}"
            ) from exc
        except OSError as exc:
            raise PairedArtifactProtocolError(
                f"paired artifact parent cannot be inspected: {current}: {exc}"
            ) from exc
        if not stat.S_ISDIR(metadata.st_mode):
            _fail("paired artifact path parent must be a real directory")

    # The lexical safe-path check above is the primary boundary.  This resolved
    # containment check is defense in depth against platform-specific aliases.
    try:
        resolved_root = root.resolve(strict=True)
        resolved_parent = candidate.parent.resolve(strict=True)
        common_parent = os.path.commonpath(
            (str(resolved_root), str(resolved_parent))
        )
        if common_parent != str(resolved_root):
            _fail("paired artifact reference escapes the ordinary run root")
    except (OSError, ValueError) as exc:
        raise PairedArtifactProtocolError(
            f"paired artifact containment cannot be established: {exc}"
        ) from exc
    return candidate


def load_verified_paired_reference_exact(
    run_root: Path,
    reference: Mapping,
    verification_context: Mapping,
) -> dict:
    """Verify one reference against its run-local file and full recomputation.

    ``verification_context`` is intentionally external to the report.  It must
    contain the authority-derived evaluation rows, the two aligned prediction
    vectors, and the exact keyword arguments used by
    :func:`build_paired_comparison_artifact`.  Report-controlled in-memory
    payloads therefore cannot create a GO result by themselves.
    """

    envelope = validate_verified_paired_reference(reference)
    context = _plain_mapping(verification_context, "paired verification context")
    _exact_keys(
        context,
        {"rows", "candidate_prediction", "baseline_prediction", "build_kwargs"},
        "paired verification context",
    )
    build_kwargs = _plain_mapping(
        context.get("build_kwargs"), "paired verification context build_kwargs"
    )
    path = _resolve_run_artifact_path(
        Path(run_root), envelope["artifact_relative_path"]
    )
    expected_digest = envelope["artifact_sha256"]

    # First require the exact on-disk representation to be the one canonical
    # representation of the embedded payload.  The exact loader below then
    # independently rebuilds all statistics and binding fields from context.
    try:
        raw = read_exact_regular_file(path, expected_digest)
    except ArtifactIOError as exc:
        raise PairedArtifactProtocolError(f"paired artifact read failed: {exc}") from exc
    parsed = _decode_exact_json(raw, path)
    canonical = canonical_paired_artifact_bytes(parsed)
    if raw != canonical:
        _fail("paired artifact file bytes are not the canonical paired representation")
    if parsed != envelope["artifact_payload"]:
        _fail("paired artifact file payload differs from the verified reference payload")

    recomputed = load_paired_comparison_artifact_exact(
        path,
        expected_digest,
        context.get("rows"),
        context.get("candidate_prediction"),
        context.get("baseline_prediction"),
        **build_kwargs,
    )
    if recomputed != parsed:
        _fail("paired artifact exact-loader result differs from the file snapshot")
    return build_verified_paired_reference(
        artifact_relative_path=envelope["artifact_relative_path"],
        artifact_sha256=expected_digest,
        artifact_payload=recomputed,
    )


def validate_paired_comparison_artifact(
    payload: Mapping,
    rows: Sequence[Mapping],
    candidate_prediction: Sequence[float],
    baseline_prediction: Sequence[float],
    **build_kwargs: Any,
) -> dict:
    """Recompute the complete expected payload and require exact field equality."""

    _validate_shape(payload)
    expected = build_paired_comparison_artifact(
        rows, candidate_prediction, baseline_prediction, **build_kwargs
    )
    actual = dict(payload)
    if actual != expected:
        differing = sorted(
            key for key in _TOP_LEVEL_KEYS if actual.get(key) != expected.get(key)
        )
        _fail(f"paired artifact does not exactly match recomputation; differing={differing}")
    return actual


def write_paired_comparison_artifact_exclusive(path: Path, payload: Mapping) -> str:
    """Write canonical finite JSON exactly once and return its exact-byte SHA-256."""

    path = Path(path)
    try:
        assert_ordinary_path(path, purpose="R3X paired comparison output")
    except ProtocolHoldError as exc:
        raise PairedArtifactProtocolError(str(exc)) from exc
    if not path.parent.is_dir():
        _fail("paired artifact parent must already exist after successful preflight")
    _validate_shape(payload)
    raw = canonical_paired_artifact_bytes(payload)
    def replay(data: bytes) -> None:
        parsed = _decode_exact_json(data, path)
        _validate_shape(parsed)

    try:
        return atomic_publish_bytes(path, raw, replay_validator=replay)
    except ArtifactIOError as exc:
        raise PairedArtifactProtocolError(str(exc)) from exc


def _decode_exact_json(raw: bytes, path: Path) -> dict:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                _fail(f"paired artifact contains duplicate JSON key: {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate,
            parse_constant=lambda value: _fail(f"non-finite JSON constant: {value}"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PairedArtifactProtocolError(
            f"paired artifact cannot be decoded as exact JSON: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        _fail("paired artifact JSON root must be an object")
    return payload


def load_paired_comparison_artifact_exact(
    path: Path,
    expected_sha256: str,
    rows: Sequence[Mapping],
    candidate_prediction: Sequence[float],
    baseline_prediction: Sequence[float],
    **build_kwargs: Any,
) -> dict:
    """Load one exact byte snapshot, verify its hash, and recompute all evidence."""

    path = Path(path)
    expected_digest = _sha256(expected_sha256, "expected paired artifact SHA-256")
    try:
        assert_ordinary_path(path, purpose="R3X paired comparison input")
    except ProtocolHoldError as exc:
        raise PairedArtifactProtocolError(str(exc)) from exc
    try:
        raw = read_exact_regular_file(path, expected_digest)
    except ArtifactIOError as exc:
        raise PairedArtifactProtocolError(f"paired artifact read failed: {exc}") from exc
    payload = _decode_exact_json(raw, path)
    try:
        assert_ordinary_record(payload, purpose="paired comparison artifact")
    except ProtocolHoldError as exc:
        raise PairedArtifactProtocolError(str(exc)) from exc
    return validate_paired_comparison_artifact(
        payload,
        rows,
        candidate_prediction,
        baseline_prediction,
        **build_kwargs,
    )
