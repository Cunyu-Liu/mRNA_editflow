"""Manuscript-safe R3X report and effect-gate logic."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from .artifact_graph import (
    ArtifactGraphError,
    resolve_artifact_ref,
    validate_artifact_ref,
)
from .artifact_io import ArtifactIOError, atomic_write_json_exclusive
from .baseline_selection import (
    BaselineSelectionError,
    QUALIFIED_STATUS as DEVELOPMENT_BASELINE_QUALIFIED_STATUS,
    load_development_baseline_selection_exact,
)
from .control_summary import (
    EvidenceVerifier,
    NegativeControlProtocolError,
    PASS_ALL as NEGATIVE_CONTROL_PASS_ALL,
    load_negative_control_summary,
)
from .paired_artifact import (
    PairedArtifactProtocolError,
    load_verified_paired_reference_exact,
)
from .protocol import ProtocolHoldError, assert_ordinary_path


class ReportProtocolError(RuntimeError):
    pass


PUBLICATION_SCOPE = "PUBLICATION_ELIGIBLE_SAME_ESTIMAND"
ALLOWED_SCIENTIFIC_SCOPES = {
    PUBLICATION_SCOPE,
    "DEVELOPMENT_ONLY_SINGLE_STUDY",
    "DEVELOPMENT_ONLY_TWO_STUDY_NESTED_ENDPOINTS",
    "DEVELOPMENT_ONLY_ENDPOINT_STUDY_CONFOUNDED",
    "DEVELOPMENT_ONLY_SAME_ENDPOINT_ESTIMAND_AUTHORITY_REQUIRED",
    "DEVELOPMENT_ONLY_ACTIVATION_SCOPE",
    "SYNTHETIC_ENGINEERING_ONLY",
}

DEVELOPMENT_BASELINE_VERIFICATION_CONTEXT_KEYS = {
    "expected_run_start_sha256",
    "expected_activation_manifest_sha256",
    "expected_train_development_outcomes_sha256",
    "expected_non_data_authority_hashes",
    "expected_eligible_method_order",
    "expected_required_estimand_cells",
    "predictor_verifiers",
}


def _finite_metric(metrics: Mapping, key: str) -> float:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ReportProtocolError(f"candidate requires finite {key}")
    return float(value)


def _metric_check(metrics: Mapping, key: str, threshold: float) -> dict:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return {"value": None, "threshold": f">={threshold:.2f}", "pass": False,
                "status": "HOLD_PRIMARY_METRIC_UNDEFINED"}
    number = float(value)
    return {"value": number, "threshold": f">={threshold:.2f}", "pass": number >= threshold,
            "status": "ESTIMATED"}


def _paired_comparison_pass(
    comparison: Mapping,
    information_set: str,
    candidate_model_id: str,
    candidate_macro_sign_accuracy: float | None,
    ordinary_run_root: Path | None,
    paired_verification_context: Mapping | None,
) -> tuple[bool, dict]:
    if ordinary_run_root is None or paired_verification_context is None:
        return False, {
            "status": "HOLD_PAIRED_FILE_VERIFICATION_CONTEXT_REQUIRED",
            "reason": (
                "paired-baseline evidence requires an external ordinary run root "
                "and authority-derived recomputation context"
            ),
        }
    try:
        reference = load_verified_paired_reference_exact(
            Path(ordinary_run_root),
            comparison,
            paired_verification_context,
        )
    except (PairedArtifactProtocolError, OSError, TypeError, ValueError) as exc:
        return False, {
            "status": "HOLD_PAIRED_FILE_VERIFICATION_FAILED",
            "reason": str(exc),
        }
    payload = reference["artifact_payload"]
    primary = payload["primary_sign_utility"]
    same_fingerprint = (
        payload["candidate_information_set_fingerprint"]
        == payload["baseline_information_set_fingerprint"]
    )
    same_rows = (
        payload["candidate_row_set_sha256"] == payload["baseline_row_set_sha256"]
        and payload["candidate_row_order_sha256"] == payload["baseline_row_order_sha256"]
    )
    candidate_metric_matches = bool(
        candidate_macro_sign_accuracy is not None
        and math.isclose(
            float(candidate_macro_sign_accuracy),
            float(primary["candidate_raw_value"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    ci = primary["confidence_interval_candidate_minus_baseline"]
    p = primary["holm_adjusted_p_value"]
    alpha = primary["alpha"]
    passed = bool(
        payload["status"] == "PASS_PAIRED_STRONGEST_BASELINE"
        and payload["candidate_model_id"] == candidate_model_id
        and payload["information_set"] == information_set
        and same_fingerprint and same_rows and candidate_metric_matches
        and float(p) <= float(alpha) and float(ci[0]) > 0.0
    )
    return passed, {
        "status": payload["status"],
        "candidate_model_id": payload["candidate_model_id"],
        "baseline_id": payload["baseline_model_id"],
        "information_set": payload["information_set"],
        "candidate_information_set_fingerprint": payload[
            "candidate_information_set_fingerprint"
        ],
        "baseline_information_set_fingerprint": payload[
            "baseline_information_set_fingerprint"
        ],
        "primary_metric_id": primary["metric_id"],
        "same_information_set_fingerprint": same_fingerprint,
        "same_row_set_and_order": same_rows,
        "candidate_row_set_sha256": payload["candidate_row_set_sha256"],
        "candidate_metric_matches_effect_gate": candidate_metric_matches,
        "holm_adjusted_p_value": p,
        "alpha": alpha,
        "confidence_interval_candidate_minus_baseline": ci,
        "paired_result_artifact_path": reference["artifact_relative_path"],
        "paired_result_artifact_sha256": reference["artifact_sha256"],
        "verification_status": reference["verification_status"],
    }


def _development_baseline_selection_pass(
    reference: Mapping | None,
    ordinary_run_root: Path | None,
    verification_context: Mapping | None,
    information_set: str,
    paired_evidence: Mapping,
) -> tuple[bool, dict]:
    """Replay the development-only comparator selector from exact files.

    A selector is intentionally not trusted merely because its own JSON is
    internally consistent.  Its external context must bind the run start,
    activation, the sole train/development outcome authority, every frozen
    non-data authority, the eligible-method order, and the complete required
    estimand/cell layout.  The loader then recursively replays every referenced
    predictor, development truth, and prediction artifact.
    """

    if reference is None:
        return False, {
            "status": "HOLD_DEVELOPMENT_BASELINE_SELECTION_REFERENCE_REQUIRED",
            "reason": (
                "effect GO requires a hash-bound development-only strongest-"
                "baseline selection reference"
            ),
        }
    try:
        artifact_ref = validate_artifact_ref(reference)
    except (ArtifactGraphError, TypeError, ValueError) as exc:
        return False, {
            "status": "HOLD_DEVELOPMENT_BASELINE_SELECTION_REFERENCE_INVALID",
            "reason": str(exc),
        }
    if ordinary_run_root is None:
        return False, {
            "status": "HOLD_DEVELOPMENT_BASELINE_SELECTION_RUN_ROOT_REQUIRED",
            "reason": (
                "development baseline selection requires an external ordinary "
                "run root"
            ),
            "development_baseline_selection_artifact_path": artifact_ref[
                "relative_path"
            ],
            "development_baseline_selection_artifact_sha256": artifact_ref[
                "sha256"
            ],
        }
    if not isinstance(verification_context, Mapping):
        return False, {
            "status": "HOLD_DEVELOPMENT_BASELINE_SELECTION_CONTEXT_REQUIRED",
            "reason": (
                "development baseline replay requires authority-derived external "
                "verification context"
            ),
            "development_baseline_selection_artifact_path": artifact_ref[
                "relative_path"
            ],
            "development_baseline_selection_artifact_sha256": artifact_ref[
                "sha256"
            ],
        }
    if set(verification_context) != DEVELOPMENT_BASELINE_VERIFICATION_CONTEXT_KEYS:
        return False, {
            "status": "HOLD_DEVELOPMENT_BASELINE_SELECTION_CONTEXT_INVALID",
            "reason": (
                "development baseline verification context differs from the "
                "strict authority-binding schema"
            ),
            "development_baseline_selection_artifact_path": artifact_ref[
                "relative_path"
            ],
            "development_baseline_selection_artifact_sha256": artifact_ref[
                "sha256"
            ],
        }
    try:
        selector_path = resolve_artifact_ref(Path(ordinary_run_root), artifact_ref)
        payload = load_development_baseline_selection_exact(
            Path(ordinary_run_root),
            selector_path,
            artifact_ref["sha256"],
            expected_run_start_sha256=verification_context[
                "expected_run_start_sha256"
            ],
            expected_activation_manifest_sha256=verification_context[
                "expected_activation_manifest_sha256"
            ],
            expected_train_development_outcomes_sha256=verification_context[
                "expected_train_development_outcomes_sha256"
            ],
            expected_non_data_authority_hashes=verification_context[
                "expected_non_data_authority_hashes"
            ],
            expected_eligible_method_order=verification_context[
                "expected_eligible_method_order"
            ],
            expected_required_estimand_cells=verification_context[
                "expected_required_estimand_cells"
            ],
            predictor_verifiers=verification_context["predictor_verifiers"],
        )
    except (
        ArtifactGraphError,
        BaselineSelectionError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        return False, {
            "status": "HOLD_DEVELOPMENT_BASELINE_SELECTION_FILE_VERIFICATION_FAILED",
            "reason": str(exc),
            "development_baseline_selection_artifact_path": artifact_ref[
                "relative_path"
            ],
            "development_baseline_selection_artifact_sha256": artifact_ref[
                "sha256"
            ],
        }

    paired_baseline_id = paired_evidence.get("baseline_id")
    paired_candidate_fingerprint = paired_evidence.get(
        "candidate_information_set_fingerprint"
    )
    paired_baseline_fingerprint = paired_evidence.get(
        "baseline_information_set_fingerprint"
    )
    selected_method_matches = payload["selected_method_id"] == paired_baseline_id
    same_information_set = bool(
        payload["information_set"] == information_set
        and paired_evidence.get("information_set") == information_set
    )
    same_information_set_fingerprint = bool(
        payload["information_set_fingerprint"] == paired_candidate_fingerprint
        and payload["information_set_fingerprint"] == paired_baseline_fingerprint
    )
    passed = bool(
        payload["status"] == DEVELOPMENT_BASELINE_QUALIFIED_STATUS
        and selected_method_matches
        and same_information_set
        and same_information_set_fingerprint
    )
    return passed, {
        "status": payload["status"],
        "selected_method_id": payload["selected_method_id"],
        "paired_baseline_id": paired_baseline_id,
        "selected_method_matches_paired_baseline": selected_method_matches,
        "information_set": payload["information_set"],
        "same_information_set": same_information_set,
        "information_set_fingerprint": payload["information_set_fingerprint"],
        "paired_candidate_information_set_fingerprint": paired_candidate_fingerprint,
        "paired_baseline_information_set_fingerprint": paired_baseline_fingerprint,
        "same_information_set_fingerprint": same_information_set_fingerprint,
        "run_start_sha256": payload["run_start_sha256"],
        "activation_manifest_sha256": payload["activation_manifest_sha256"],
        "eligible_method_order_sha256": payload["eligible_method_order_sha256"],
        "required_estimand_cells_sha256": payload[
            "required_estimand_cells_sha256"
        ],
        "development_baseline_selection_artifact_path": artifact_ref[
            "relative_path"
        ],
        "development_baseline_selection_artifact_sha256": artifact_ref["sha256"],
        "verification_status": (
            "EXACT_CANONICAL_SELECTOR_AND_REFERENCED_DEVELOPMENT_EVIDENCE_REPLAYED"
        ),
    }


def _selection_policy_pass(policy: Mapping) -> tuple[bool, dict]:
    value = policy.get("value") if isinstance(policy, Mapping) else None
    threshold = policy.get("threshold") if isinstance(policy, Mapping) else None
    numeric = all(
        not isinstance(item, bool) and isinstance(item, (int, float)) and math.isfinite(float(item))
        for item in (value, threshold)
    )
    passed = bool(
        isinstance(policy, Mapping)
        and policy.get("schema_version") == "r3x_selection_metric_result_v1"
        and policy.get("status") == "FROZEN_EVALUATED"
        and policy.get("metric_id") == "ORACLE_NORMALIZED_SELECTION_GAIN"
        and policy.get("tie_policy") == "EXPECTED_OUTCOME_OVER_CUTOFF_TIE_BLOCK"
        and numeric and float(value) >= float(threshold)
    )
    return passed, {
        "status": policy.get("status") if isinstance(policy, Mapping) else None,
        "metric_id": policy.get("metric_id") if isinstance(policy, Mapping) else None,
        "value": value,
        "threshold": threshold,
        "tie_policy": policy.get("tie_policy") if isinstance(policy, Mapping) else None,
    }


def _negative_control_summary_pass(
    reference: Mapping | None,
    ordinary_run_root: Path | None,
    evidence_verifier: EvidenceVerifier | None,
    verifier_context: Mapping[str, object] | None,
    candidate_internal_test_row_binding_sha256: str | None,
) -> tuple[bool, dict]:
    """Replay a hash-bound seven-control summary from the ordinary run root.

    The summary payload is never trusted from the report itself.  Only a strict
    two-field artifact reference may enter the gate, and even exact canonical
    summary bytes cannot PASS without a caller-supplied independent evidence
    verifier and verifier context.
    """

    if reference is None:
        return False, {
            "status": "HOLD_NEGATIVE_CONTROL_SUMMARY_REFERENCE_REQUIRED",
            "reason": "effect GO requires a hash-bound negative-control summary reference",
        }
    try:
        artifact_ref = validate_artifact_ref(reference)
    except (ArtifactGraphError, TypeError, ValueError) as exc:
        return False, {
            "status": "HOLD_NEGATIVE_CONTROL_SUMMARY_REFERENCE_INVALID",
            "reason": str(exc),
        }
    if ordinary_run_root is None:
        return False, {
            "status": "HOLD_NEGATIVE_CONTROL_RUN_ROOT_REQUIRED",
            "reason": "negative-control evidence requires an external ordinary run root",
            "negative_control_summary_artifact_path": artifact_ref["relative_path"],
            "negative_control_summary_artifact_sha256": artifact_ref["sha256"],
        }
    if evidence_verifier is None or not callable(evidence_verifier):
        return False, {
            "status": "HOLD_NEGATIVE_CONTROL_EVIDENCE_VERIFIER_REQUIRED",
            "reason": "PASS/FAIL controls require an independent structured evidence verifier",
            "negative_control_summary_artifact_path": artifact_ref["relative_path"],
            "negative_control_summary_artifact_sha256": artifact_ref["sha256"],
        }
    if not isinstance(verifier_context, Mapping):
        return False, {
            "status": "HOLD_NEGATIVE_CONTROL_VERIFIER_CONTEXT_REQUIRED",
            "reason": "negative-control replay requires caller-supplied verifier context",
            "negative_control_summary_artifact_path": artifact_ref["relative_path"],
            "negative_control_summary_artifact_sha256": artifact_ref["sha256"],
        }
    expected_test_binding = verifier_context.get(
        "expected_internal_test_row_binding_sha256"
    )
    if not (
        isinstance(expected_test_binding, str)
        and len(expected_test_binding) == 64
        and all(character in "0123456789abcdef" for character in expected_test_binding)
    ):
        return False, {
            "status": "HOLD_NEGATIVE_CONTROL_VERIFIER_CONTEXT_INVALID",
            "reason": (
                "verifier context must bind the frozen candidate/global-prediction "
                "ordinary internal-test row-set SHA-256"
            ),
            "negative_control_summary_artifact_path": artifact_ref["relative_path"],
            "negative_control_summary_artifact_sha256": artifact_ref["sha256"],
        }
    if not (
        isinstance(candidate_internal_test_row_binding_sha256, str)
        and len(candidate_internal_test_row_binding_sha256) == 64
        and all(
            character in "0123456789abcdef"
            for character in candidate_internal_test_row_binding_sha256
        )
    ):
        return False, {
            "status": "HOLD_NEGATIVE_CONTROL_CANDIDATE_TEST_BINDING_REQUIRED",
            "reason": (
                "negative-control GO requires the row-set SHA-256 recomputed from "
                "the exact paired candidate prediction evidence"
            ),
            "negative_control_summary_artifact_path": artifact_ref["relative_path"],
            "negative_control_summary_artifact_sha256": artifact_ref["sha256"],
        }
    if expected_test_binding != candidate_internal_test_row_binding_sha256:
        return False, {
            "status": "HOLD_NEGATIVE_CONTROL_VERIFIER_CONTEXT_TEST_BINDING_MISMATCH",
            "reason": (
                "verifier context does not bind the exact paired candidate/global-"
                "prediction internal-test row set"
            ),
            "candidate_internal_test_row_binding_sha256": (
                candidate_internal_test_row_binding_sha256
            ),
            "verifier_context_internal_test_row_binding_sha256": expected_test_binding,
            "negative_control_summary_artifact_path": artifact_ref["relative_path"],
            "negative_control_summary_artifact_sha256": artifact_ref["sha256"],
        }
    try:
        summary = load_negative_control_summary(
            Path(ordinary_run_root) / artifact_ref["relative_path"],
            artifact_ref["sha256"],
            run_root=Path(ordinary_run_root),
            evidence_verifier=evidence_verifier,
            verifier_context=verifier_context,
        )
    except (NegativeControlProtocolError, OSError, TypeError, ValueError) as exc:
        return False, {
            "status": "HOLD_NEGATIVE_CONTROL_FILE_VERIFICATION_FAILED",
            "reason": str(exc),
            "negative_control_summary_artifact_path": artifact_ref["relative_path"],
            "negative_control_summary_artifact_sha256": artifact_ref["sha256"],
        }

    controls = summary["controls"]
    observed_test_binding = summary["ordinary_internal_test_row_binding_sha256"]
    if observed_test_binding != candidate_internal_test_row_binding_sha256:
        return False, {
            "status": "HOLD_NEGATIVE_CONTROL_TEST_ROW_BINDING_MISMATCH",
            "reason": (
                "negative-control evidence was evaluated on a different ordinary "
                "internal-test row set than the candidate/global prediction"
            ),
            "expected_internal_test_row_binding_sha256": (
                candidate_internal_test_row_binding_sha256
            ),
            "observed_internal_test_row_binding_sha256": observed_test_binding,
            "negative_control_summary_artifact_path": artifact_ref["relative_path"],
            "negative_control_summary_artifact_sha256": artifact_ref["sha256"],
            "verification_status": "EXACT_CANONICAL_SUMMARY_AND_REFERENCED_EVIDENCE_REPLAYED",
        }
    passed = bool(
        summary["overall_status"] == NEGATIVE_CONTROL_PASS_ALL
        and len(controls) == 7
        and all(control["status"] in {"PASS", "NOT_APPLICABLE"} for control in controls)
    )
    return passed, {
        "status": summary["overall_status"],
        "control_count": len(controls),
        "control_statuses": {
            control["control_id"]: control["status"] for control in controls
        },
        "ordinary_internal_test_row_binding_sha256": summary[
            "ordinary_internal_test_row_binding_sha256"
        ],
        "expected_internal_test_row_binding_sha256": (
            candidate_internal_test_row_binding_sha256
        ),
        "negative_control_summary_artifact_path": artifact_ref["relative_path"],
        "negative_control_summary_artifact_sha256": artifact_ref["sha256"],
        "verification_status": "EXACT_CANONICAL_SUMMARY_AND_REFERENCED_EVIDENCE_REPLAYED",
    }


def evaluate_effect_gate(
    *,
    candidate_model_id: str,
    candidate_metrics: Mapping,
    information_set: str,
    strongest_baseline_comparison: Mapping,
    selection_metric_policy: Mapping,
    scientific_scope: str,
    ordinary_run_root: Path | None = None,
    paired_verification_context: Mapping | None = None,
    development_baseline_selection_reference: Mapping | None = None,
    development_baseline_selection_verification_context: Mapping | None = None,
    negative_control_summary_reference: Mapping | None = None,
    negative_control_evidence_verifier: EvidenceVerifier | None = None,
    negative_control_verifier_context: Mapping[str, object] | None = None,
) -> dict:
    """Evaluate one model's complete metric bundle without cross-model splicing."""

    if not isinstance(candidate_model_id, str) or not candidate_model_id:
        raise ReportProtocolError("candidate_model_id is required")
    if information_set not in {
        "EDIT_ONLY", "CANDIDATE_ONLY", "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        "MEASURED_SOURCE_ANCHOR_ASSISTED", "FOUNDATION_EXPOSURE_STRATIFIED",
    }:
        raise ReportProtocolError("unknown candidate information set")
    if scientific_scope not in ALLOWED_SCIENTIFIC_SCOPES:
        raise ReportProtocolError("unknown scientific scope cannot enter the effect gate")
    sign_check = _metric_check(candidate_metrics, "macro_sign_accuracy", 0.60)
    spearman_check = _metric_check(candidate_metrics, "macro_delta_spearman", 0.25)
    selection_pass, selection_evidence = _selection_policy_pass(selection_metric_policy)
    baseline_win, baseline_evidence = _paired_comparison_pass(
        strongest_baseline_comparison,
        information_set,
        candidate_model_id,
        sign_check["value"],
        ordinary_run_root,
        paired_verification_context,
    )
    baseline_selection_pass, baseline_selection_evidence = (
        _development_baseline_selection_pass(
            development_baseline_selection_reference,
            ordinary_run_root,
            development_baseline_selection_verification_context,
            information_set,
            baseline_evidence,
        )
    )
    controls_pass, controls_evidence = _negative_control_summary_pass(
        negative_control_summary_reference,
        ordinary_run_root,
        negative_control_evidence_verifier,
        negative_control_verifier_context,
        baseline_evidence.get("candidate_row_set_sha256"),
    )
    checks = {
        "macro_sign_accuracy": sign_check,
        "macro_delta_spearman": spearman_check,
        "stable_selection_metric": {**selection_evidence, "pass": selection_pass},
        "strongest_same_information_set_baseline_win": {
            **baseline_evidence, "pass": baseline_win,
        },
        "development_only_strongest_baseline_selection": {
            **baseline_selection_evidence, "pass": baseline_selection_pass,
        },
        "preregistered_negative_controls": {
            **controls_evidence, "pass": controls_pass,
        },
    }
    all_pass = all(check["pass"] for check in checks.values())
    if scientific_scope != PUBLICATION_SCOPE:
        status = scientific_scope
        go = False
    elif not selection_pass:
        status = "HOLD_SELECTION_METRIC_POLICY"
        go = False
    elif all_pass:
        status = "GO_EFFECT"
        go = True
    elif (
        sign_check["pass"]
        and spearman_check["pass"]
        and selection_pass
        and baseline_win
        and not controls_pass
    ):
        status = "HOLD_NEGATIVE_CONTROLS"
        go = False
    elif (
        sign_check["pass"]
        and spearman_check["pass"]
        and selection_pass
        and baseline_win
        and controls_pass
        and not baseline_selection_pass
    ):
        status = "HOLD_DEVELOPMENT_BASELINE_SELECTION"
        go = False
    else:
        status = "HOLD_M0_REPAIR"
        go = False
    return {
        "schema_version": "r3x_effect_gate_v1",
        "candidate_model_id": candidate_model_id,
        "information_set": information_set,
        "metric_bundle_source": "SINGLE_MODEL_SINGLE_INFORMATION_SET",
        "scientific_scope": scientific_scope,
        "checks": checks,
        "status": status,
        "GO_EFFECT": go,
        # Even a valid effect GO is not authority to unlock other phases.
        "downstream_unlock": False,
        "sealed_access": False,
        "sign_gate_changed": False,
    }


def _walk_keys(value: Any):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_keys(child)


def validate_run_report(
    report: Mapping,
    *,
    ordinary_run_root: Path | None = None,
    paired_verification_context_by_method: Mapping[str, Mapping] | None = None,
    development_baseline_selection_verification_context_by_method: (
        Mapping[str, Mapping] | None
    ) = None,
    negative_control_evidence_verifier_by_method: Mapping[str, EvidenceVerifier] | None = None,
    negative_control_verifier_context_by_method: Mapping[str, Mapping[str, object]] | None = None,
) -> dict:
    """Validate a report, requiring external file evidence for every GO gate.

    The optional arguments preserve fail-closed compatibility for development
    reports.  Without them, paired-baseline checks remain HOLD and any report
    that claims a GO result is rejected during strict recomputation.
    """

    if not isinstance(report, Mapping) or report.get("schema_version") != "r3x_ordinary_run_report_v1":
        raise ReportProtocolError("unsupported R3X run-report schema")
    if report.get("sealed_access") is not False:
        raise ReportProtocolError("R3X ordinary report cannot assert sealed access")
    if report.get("downstream_unlock") is not False:
        raise ReportProtocolError("R3X report cannot unlock O0/G1/E0")
    if report.get("active_contract_mutation") is not False:
        raise ReportProtocolError("R3X report cannot mutate the active contract")
    if report.get("sign_gate") != 0.60:
        raise ReportProtocolError("R3X report sign gate must remain exactly 0.60")
    scope = report.get("scientific_scope")
    if scope not in ALLOWED_SCIENTIFIC_SCOPES:
        raise ReportProtocolError("R3X report has an unknown scientific scope")
    if report.get("status") == "SYNTHETIC_ENGINEERING_ONLY" and report.get("GO_EFFECT") is not False:
        raise ReportProtocolError("synthetic evidence can never be GO_EFFECT")
    prohibited = {
        "top10_enrichment_legacy_signed_oracle_ratio",
        "top10pct_enrichment_legacy_signed_mean_ratio",
        "legacy_signed_ratio_enrichment",
    }
    present = sorted(prohibited.intersection(_walk_keys(report)))
    if present:
        raise ReportProtocolError(f"report contains prohibited legacy metric keys: {present}")
    methods = report.get("methods")
    if not isinstance(methods, Mapping) or not methods:
        raise ReportProtocolError("run report lacks method results")
    gates = report.get("effect_gates", {})
    if not isinstance(gates, Mapping):
        raise ReportProtocolError("effect_gates must be an object")
    gate_inputs = report.get("effect_gate_inputs", {})
    if not isinstance(gate_inputs, Mapping) or set(gate_inputs) != set(gates):
        raise ReportProtocolError(
            "effect_gate_inputs must exactly cover every effect gate"
        )
    gate_go = []
    for method_id, gate in gates.items():
        if method_id not in methods or not isinstance(gate, Mapping):
            raise ReportProtocolError("effect gate does not bind an existing method")
        if gate.get("candidate_model_id") != method_id:
            raise ReportProtocolError("effect gate candidate ID mismatch")
        if gate.get("scientific_scope") != scope:
            raise ReportProtocolError("effect gate/report scientific scope mismatch")
        if gate.get("sign_gate_changed") is not False:
            raise ReportProtocolError("effect gate cannot change sign threshold")
        if gate.get("downstream_unlock") is not False or gate.get("sealed_access") is not False:
            raise ReportProtocolError("effect gate cannot unlock downstream or sealed access")
        inputs = gate_inputs.get(method_id)
        legacy_input_keys = {
            "candidate_metrics", "information_set",
            "strongest_baseline_comparison", "selection_metric_policy",
            "negative_control_summary_reference",
        }
        selector_input_keys = legacy_input_keys | {
            "development_baseline_selection_reference"
        }
        if not isinstance(inputs, Mapping) or frozenset(inputs) not in {
            frozenset(legacy_input_keys),
            frozenset(selector_input_keys),
        }:
            raise ReportProtocolError("effect gate inputs differ from the strict schema")
        try:
            validate_artifact_ref(inputs["negative_control_summary_reference"])
        except (ArtifactGraphError, TypeError, ValueError) as exc:
            raise ReportProtocolError(
                f"negative-control summary reference differs from strict schema: {exc}"
            ) from exc
        development_baseline_reference = inputs.get(
            "development_baseline_selection_reference"
        )
        if development_baseline_reference is not None:
            try:
                validate_artifact_ref(development_baseline_reference)
            except (ArtifactGraphError, TypeError, ValueError) as exc:
                raise ReportProtocolError(
                    "development baseline selection reference differs from strict "
                    f"schema: {exc}"
                ) from exc
        method = methods[method_id]
        expected_metrics = None
        if isinstance(method, Mapping):
            if isinstance(method.get("ensemble_test_metrics"), Mapping):
                expected_metrics = method["ensemble_test_metrics"].get("macro")
            elif isinstance(method.get("metrics"), Mapping):
                expected_metrics = method["metrics"].get("macro")
        if expected_metrics is None or inputs.get("candidate_metrics") != expected_metrics:
            raise ReportProtocolError(
                "effect-gate candidate metrics do not equal the method prediction metrics"
            )
        if inputs.get("information_set") != method.get("information_set"):
            raise ReportProtocolError("effect-gate information set differs from method")
        verification_context = None
        if isinstance(paired_verification_context_by_method, Mapping):
            verification_context = paired_verification_context_by_method.get(method_id)
        development_baseline_context = None
        if isinstance(
            development_baseline_selection_verification_context_by_method,
            Mapping,
        ):
            development_baseline_context = (
                development_baseline_selection_verification_context_by_method.get(
                    method_id
                )
            )
        negative_control_verifier = None
        if isinstance(negative_control_evidence_verifier_by_method, Mapping):
            negative_control_verifier = negative_control_evidence_verifier_by_method.get(
                method_id
            )
        negative_control_context = None
        if isinstance(negative_control_verifier_context_by_method, Mapping):
            negative_control_context = negative_control_verifier_context_by_method.get(
                method_id
            )
        recomputed = evaluate_effect_gate(
            candidate_model_id=method_id,
            candidate_metrics=inputs["candidate_metrics"],
            information_set=inputs["information_set"],
            strongest_baseline_comparison=inputs["strongest_baseline_comparison"],
            selection_metric_policy=inputs["selection_metric_policy"],
            scientific_scope=scope,
            ordinary_run_root=ordinary_run_root,
            paired_verification_context=verification_context,
            development_baseline_selection_reference=(
                development_baseline_reference
            ),
            development_baseline_selection_verification_context=(
                development_baseline_context
            ),
            negative_control_summary_reference=inputs[
                "negative_control_summary_reference"
            ],
            negative_control_evidence_verifier=negative_control_verifier,
            negative_control_verifier_context=negative_control_context,
        )
        if dict(gate) != recomputed:
            raise ReportProtocolError("effect gate does not equal strict recomputation")
        gate_go.append(gate.get("GO_EFFECT") is True)
    expected_go = any(gate_go)
    if report.get("GO_EFFECT") is not expected_go:
        raise ReportProtocolError("report GO_EFFECT is inconsistent with validated method gates")
    if expected_go:
        if scope != PUBLICATION_SCOPE:
            raise ReportProtocolError("non-publication scope can never report GO_EFFECT")
        if report.get("status") != "ORDINARY_GO_EFFECT_CANDIDATE_EXISTS":
            raise ReportProtocolError("GO report status is inconsistent")
    elif report.get("status") == "ORDINARY_GO_EFFECT_CANDIDATE_EXISTS":
        raise ReportProtocolError("GO status requires a valid GO effect gate")
    return dict(report)


def write_json_exclusive(path: Path, payload: Mapping) -> str:
    path = Path(path)
    try:
        assert_ordinary_path(path, purpose="R3X report output")
    except ProtocolHoldError as exc:
        raise ReportProtocolError(str(exc)) from exc
    if not path.parent.is_dir():
        raise ReportProtocolError("report parent directory does not exist after preflight")
    try:
        return atomic_write_json_exclusive(path, payload)
    except ArtifactIOError as exc:
        raise ReportProtocolError(str(exc)) from exc
