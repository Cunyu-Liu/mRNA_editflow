#!/usr/bin/env python3
"""Static, synthetic and zero-update preparation for the DEC028 critic.

This module validates a future one-fit contract and a few pure scientific
formulas.  It has no model, optimizer, data loader, CUDA path, checkpoint path,
or artifact publisher.  A later reviewed successor and separate run authority
are required before learned execution.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT
    / "configs/route_a_v3_gse200304_source_relative_critic_g0_candidate_v1.json"
)
SCHEMA_VERSION = "route_a_v3_gse200304_source_relative_critic_g0_candidate.v1"
CANDIDATE_ID = "ROUTE_A_V3_GSE200304_SOURCE_RELATIVE_CRITIC_G0_CANDIDATE_V1"
DOCUMENT_STATUS = "DRAFT_FOR_REVIEW_NOT_ACTIVE_PROTOCOL"


class ContractError(RuntimeError):
    """The inactive critic contract or a synthetic formula input is invalid."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ContractError(f"non-finite JSON constant: {value}")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load critic candidate config: {path}") from exc
    if type(payload) is not dict:
        raise ContractError("critic candidate config root must be an object")
    validate_config(payload)
    return payload


def _require_false(mapping: Mapping[str, Any], keys: tuple[str, ...], label: str) -> None:
    for key in keys:
        if mapping.get(key) is not False:
            raise ContractError(f"{label}.{key} must remain false")


def validate_config(config: Mapping[str, Any]) -> None:
    required_root = {
        "schema_version",
        "candidate_id",
        "document_status",
        "authority_status",
        "activation_state",
        "authority_context",
        "permitted_g0_operations",
        "forbidden_operations",
        "estimand_contract",
        "architecture_plan",
        "future_single_fit_contract",
        "split_and_evaluator_boundary",
        "scientific_state",
        "validate_only_truth",
    }
    if set(config) != required_root:
        raise ContractError("critic candidate root closure differs")
    exact = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "document_status": DOCUMENT_STATUS,
        "authority_status": "NON_AUTHORITATIVE",
        "activation_state": "INACTIVE_STATIC_SYNTHETIC_ZERO_UPDATE_CANDIDATE",
    }
    for key, expected in exact.items():
        if config[key] != expected:
            raise ContractError(f"critic candidate {key} differs")

    authority = config["authority_context"]
    if authority["decision_id"] != "V3-DEC-028" or authority["runtime_event_id"] != "A1-EVT-061":
        raise ContractError("critic candidate authority context differs")
    if authority["future_run_id"] != "GSE200304_SOURCE_RELATIVE_CRITIC_G1":
        raise ContractError("future run role differs")
    if authority["future_run_authorized"] is not False or authority["current_authorized_execution_count"] != 0:
        raise ContractError("future critic execution was prematurely authorized")
    if authority["current_qualified_counts"] != {
        "ordinary": 1,
        "a1": 1,
        "true_a2": 0,
        "canonical_records": 6547,
    }:
        raise ContractError("qualified-count projection differs")

    estimand = config["estimand_contract"]
    if estimand["study_unit"] != "GSE200304_SUPERSERIES_ONE_STUDY":
        raise ContractError("critic study unit differs")
    if estimand["analysis_unit"] != "BIOLOGICAL_SOURCE_GROUP":
        raise ContractError("critic analysis unit differs")
    if estimand["endpoint_effect"] != (
        "DIRECTION_NORMALIZED_CANDIDATE_MINUS_SOURCE_ON_FROZEN_TRANSFORMED_ENDPOINT_SCALE"
    ):
        raise ContractError("critic effect definition differs")
    if estimand["source_equals_candidate_mean"] != 0.0:
        raise ContractError("critic identity rule differs")
    if estimand["source_candidate_swap_rule"] != "MEAN_SIGN_FLIP":
        raise ContractError("critic swap rule differs")
    if estimand["technical_replicates_count_as_biological"] is not False:
        raise ContractError("technical replicates cannot count as biological")
    if "NEVER_IMPUTE_ZERO" not in estimand["missing_policy"] or "NEVER_IMPUTE_ZERO" not in estimand["nonfinite_policy"]:
        raise ContractError("missing or nonfinite values could be imputed to zero")

    architecture = config["architecture_plan"]
    if architecture["fixed_prefix_truncation_allowed"] is not False:
        raise ContractError("fixed-prefix truncation is forbidden")
    if architecture["every_edit_must_be_visible"] is not True:
        raise ContractError("the architecture could hide an edit")
    if architecture["padding_attention_mask_required"] is not True:
        raise ContractError("padding attention masking is required")
    if architecture["mean_construction"] != "HALF_FORWARD_MINUS_REVERSE_PAIR_SCORE":
        raise ContractError("antisymmetric mean construction differs")
    if architecture["parameter_tensors_constructed_in_g0"] != 0:
        raise ContractError("G0 constructed parameter tensors")

    run = config["future_single_fit_contract"]
    expected_counts = {
        "authorized_execution_count": 1,
        "optimizer_fit_count": 1,
        "fold_model_count": 1,
        "checkpoint_count": 1,
        "final_refit_count": 0,
        "seed_count": 1,
    }
    for key, expected in expected_counts.items():
        if run[key] != expected:
            raise ContractError(f"future single-fit count differs: {key}")
    if run["nested_cross_validation_authorized"] is not False:
        raise ContractError("nested learned CV is not authorized")
    for key in (
        "early_stopping_allowed",
        "best_checkpoint_selection_allowed",
        "hyperparameter_search_allowed",
        "automatic_retry_allowed",
    ):
        if run[key] is not False:
            raise ContractError(f"future single-fit rule is not frozen: {key}")
    if run["checkpoint_policy"] != "TERMINAL_CHECKPOINT_ONLY":
        raise ContractError("checkpoint policy differs")

    boundary = config["split_and_evaluator_boundary"]
    _require_false(
        boundary,
        (
            "real_membership_frozen",
            "real_split_assignments_generated",
            "evaluator_receives_guide_output",
            "evaluator_selects_model_or_checkpoint",
            "test_selects_threshold",
        ),
        "split_and_evaluator_boundary",
    )
    if boundary["split_assignment_count"] != 0:
        raise ContractError("real split assignments were generated in G0")

    _require_false(
        config["scientific_state"],
        (
            "critic_g1_launched",
            "critic_pass_asserted",
            "a6_learned_base_value_authorized",
            "training_allowed",
            "gpu_work_allowed",
            "model_selection_allowed",
            "a7_allowed",
            "next_phase_authorized",
        ),
        "scientific_state",
    )
    for key, value in config["validate_only_truth"].items():
        if value != 0:
            raise ContractError(f"validate-only truth is nonzero: {key}")


def direction_normalized_effect(
    *, source_endpoint: float, candidate_endpoint: float, direction_multiplier: int
) -> float:
    """Return the pure candidate-minus-source estimand for synthetic checks."""

    if direction_multiplier not in (-1, 1):
        raise ContractError("direction multiplier must be -1 or 1")
    values = (source_endpoint, candidate_endpoint)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ContractError("endpoint values must be numeric")
    if any(not math.isfinite(float(value)) for value in values):
        raise ContractError("missing or nonfinite endpoints are excluded, never zero-imputed")
    return direction_multiplier * (float(candidate_endpoint) - float(source_endpoint))


def antisymmetric_pair_mean(*, forward_score: float, reverse_score: float) -> float:
    """Apply the frozen half-forward-minus-reverse construction."""

    if not math.isfinite(forward_score) or not math.isfinite(reverse_score):
        raise ContractError("pair scores must be finite")
    return 0.5 * (float(forward_score) - float(reverse_score))


def calibrated_lower_confidence_bound(
    *, mean: float, predictive_scale: float, calibration_quantile: float
) -> float:
    """Pure future LCB formula; no calibrator is fit in this candidate."""

    values = (mean, predictive_scale, calibration_quantile)
    if any(not math.isfinite(float(value)) for value in values):
        raise ContractError("LCB inputs must be finite")
    if predictive_scale <= 0.0 or calibration_quantile < 0.0:
        raise ContractError("LCB scale must be positive and quantile nonnegative")
    return float(mean) - float(calibration_quantile) * float(predictive_scale)


def validate_only(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    return {
        "candidate_id": CANDIDATE_ID,
        "status": "PASS_STATIC_SYNTHETIC_ZERO_UPDATE_PREPARATION_ONLY_NOT_ACTIVE",
        "future_run_authorized": False,
        "future_run_id": "GSE200304_SOURCE_RELATIVE_CRITIC_G1",
        "validate_only_truth": dict(config["validate_only_truth"]),
        "scientific_claim_status": "NOT_ESTABLISHED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--validate-only", action="store_true", required=True)
    args = parser.parse_args()
    print(json.dumps(validate_only(load_config(args.config)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
