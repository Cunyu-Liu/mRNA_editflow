#!/usr/bin/env python3
"""Non-authoritative synthetic-only A2 G0 evaluator candidate.

The candidate has no project-row loader, no materialization path, and no real
evaluation mode.  Its sole executable path validates the frozen candidate
schema, groups hard-coded synthetic identifiers into connected components, and
checks a synthetic candidate-minus-source endpoint contract.  It never imports
a learner, probes CUDA, or writes a runtime artifact.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


STAGING_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_CONFIG_PATH = STAGING_ROOT / "configs/route_a_v3_a2_g0_evaluator_candidate_v1.json"

CANDIDATE_ID = "ROUTE_A_V3_A2_G0_SYNTHETIC_EVALUATOR_CANDIDATE_V1"
DOCUMENT_STATUS = "DRAFT_FOR_REVIEW_NOT_ACTIVE_PROTOCOL"
SYNTHETIC_PREFIX = "SYN_"
ENDPOINT_DIRECTION = "CANDIDATE_MINUS_SOURCE"


class G0CandidateError(RuntimeError):
    """The static A2 candidate contract was violated."""


@dataclass(frozen=True)
class SyntheticEdge:
    source_id: str
    candidate_id: str


@dataclass(frozen=True)
class SyntheticSplitPlan:
    components: tuple[tuple[str, ...], ...]
    fold_by_component: tuple[int, ...]
    outcome_values_consumed: bool
    final_assignment_materialized: bool


@dataclass(frozen=True)
class DirectionNormalizedEndpoint:
    source_id: str
    candidate_id: str
    candidate_minus_source: float
    biological_se: float
    biological_replicate_count: int
    endpoint_direction: str


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise G0CandidateError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise G0CandidateError(f"non-finite JSON constant: {value}")


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise G0CandidateError(f"cannot read JSON candidate: {path}") from exc
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G0CandidateError(f"invalid JSON candidate: {path}") from exc
    if not isinstance(payload, dict):
        raise G0CandidateError("candidate JSON root must be an object")
    return payload


def validate_candidate_config(payload: Mapping[str, Any]) -> None:
    expected_root = {
        "schema_version",
        "candidate_id",
        "document_status",
        "authority_status",
        "activation_state",
        "parent_contract",
        "permitted_g0_operations",
        "forbidden_operations",
        "synthetic_evaluator_contract",
        "runtime_truth",
    }
    if set(payload) != expected_root:
        raise G0CandidateError("candidate config root closure differs")
    expected_exact = {
        "schema_version": "route_a_v3_a2_g0_evaluator_candidate.v1",
        "candidate_id": CANDIDATE_ID,
        "document_status": DOCUMENT_STATUS,
        "authority_status": "NON_AUTHORITATIVE",
        "activation_state": "INACTIVE_EVALUATOR_CANDIDATE",
    }
    for key, expected in expected_exact.items():
        if payload[key] != expected:
            raise G0CandidateError(f"candidate {key} differs")
    if payload["parent_contract"] != {
        "path": "docs/contracts/candidates/mrna_xeditflow_route_a_v3_dec028_single_study_mainline_contract_v1.md",
        "required_document_status": DOCUMENT_STATUS,
        "requires_fresh_runtime_authority_sync": True,
        "promotion_requires_separate_later_authority": True,
    }:
        raise G0CandidateError("candidate parent-contract boundary differs")
    required_forbidden = {
        "PROJECT_PRIVATE_OR_SEALED_ROW_IO",
        "FINAL_MEMBERSHIP_OR_SPLIT_ASSIGNMENT",
        "REAL_EVALUATION",
        "GUIDE_EVALUATOR_FEEDBACK",
        "FORMAL_POWER_OR_A2_PASS",
        "MODEL_OR_MODEL_SELECTION_ACCESS",
        "CUDA_PROBE_OR_GPU_TOUCH",
        "CHECKPOINT_READ_OR_WRITE",
        "PARAMETER_UPDATE_OR_TRAINING",
    }
    if not required_forbidden.issubset(set(payload["forbidden_operations"])):
        raise G0CandidateError("candidate forbidden-operation closure is incomplete")
    evaluator = payload["synthetic_evaluator_contract"]
    expected_evaluator = {
        "input_scope": "SYNTHETIC_FIXTURES_OR_AGGREGATE_METADATA_ONLY",
        "synthetic_identifier_prefix": SYNTHETIC_PREFIX,
        "component_algorithm": "CONNECTED_COMPONENTS_FROM_SOURCE_CANDIDATE_RELATIONS",
        "split_planning": "OUTCOME_BLIND_COMPONENT_DISJOINT_SYNTHETIC_ONLY",
        "endpoint_direction": ENDPOINT_DIRECTION,
        "biological_se_rule": "EXPLICIT_FINITE_NONNEGATIVE_BIOLOGICAL_SE_AND_REPLICATE_COUNT_AT_LEAST_2",
        "missing_or_nonfinite_policy": "FAIL_CLOSED_NEVER_COERCE_TO_ZERO",
        "real_evaluation_allowed": False,
        "final_membership_allowed": False,
        "final_split_assignment_allowed": False,
        "guide_or_model_selection_input_allowed": False,
    }
    if evaluator != expected_evaluator:
        raise G0CandidateError("synthetic evaluator boundary differs")
    for value in payload["runtime_truth"].values():
        if value not in (0, False):
            raise G0CandidateError("candidate runtime truth is nonzero")


def load_candidate_config(path: Path = CANDIDATE_CONFIG_PATH) -> dict[str, Any]:
    payload = load_json_object(path)
    validate_candidate_config(payload)
    return payload


def _require_synthetic_identifier(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.startswith(SYNTHETIC_PREFIX) or len(value) == len(SYNTHETIC_PREFIX):
        raise G0CandidateError(f"{label} must be a nonempty synthetic identifier")


def synthetic_connected_components(edges: Sequence[SyntheticEdge]) -> tuple[tuple[str, ...], ...]:
    """Group only supplied synthetic edges; no project source/candidate rows exist."""

    if not edges:
        raise G0CandidateError("at least one synthetic edge is required")
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    seen_edges: set[tuple[str, str]] = set()
    for edge in edges:
        _require_synthetic_identifier(edge.source_id, "source_id")
        _require_synthetic_identifier(edge.candidate_id, "candidate_id")
        if edge.source_id == edge.candidate_id:
            raise G0CandidateError("candidate must differ from source")
        key = (edge.source_id, edge.candidate_id)
        if key in seen_edges:
            raise G0CandidateError("duplicate synthetic source/candidate edge")
        seen_edges.add(key)
        union(*key)

    members: dict[str, list[str]] = {}
    for node in sorted(parent):
        members.setdefault(find(node), []).append(node)
    components = tuple(sorted(tuple(sorted(nodes)) for nodes in members.values()))
    seen_nodes = [node for component in components for node in component]
    if len(seen_nodes) != len(set(seen_nodes)):
        raise G0CandidateError("synthetic components are not component-disjoint")
    return components


def outcome_blind_synthetic_split(
    edges: Sequence[SyntheticEdge], *, fold_count: int = 2
) -> SyntheticSplitPlan:
    """Create an order-stable synthetic plan without accepting endpoint values."""

    if isinstance(fold_count, bool) or not isinstance(fold_count, int) or fold_count < 2:
        raise G0CandidateError("synthetic fold count must be an integer of at least two")
    components = synthetic_connected_components(edges)
    folds = tuple(index % fold_count for index, _ in enumerate(components))
    return SyntheticSplitPlan(
        components=components,
        fold_by_component=folds,
        outcome_values_consumed=False,
        final_assignment_materialized=False,
    )


def _finite_required(value: float | None, label: str) -> float:
    if value is None:
        raise G0CandidateError(f"{label} is missing and cannot be coerced to zero")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise G0CandidateError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise G0CandidateError(f"{label} is non-finite and cannot be coerced to zero")
    return number


def validate_direction_normalized_synthetic_endpoint(
    *,
    source_id: str,
    candidate_id: str,
    source_endpoint: float | None,
    candidate_endpoint: float | None,
    biological_se: float | None,
    biological_replicate_count: int,
    endpoint_direction: str,
) -> DirectionNormalizedEndpoint:
    """Validate a synthetic endpoint metadata fixture, never a real evaluation."""

    _require_synthetic_identifier(source_id, "source_id")
    _require_synthetic_identifier(candidate_id, "candidate_id")
    if source_id == candidate_id:
        raise G0CandidateError("candidate must differ from source")
    if endpoint_direction != ENDPOINT_DIRECTION:
        raise G0CandidateError("endpoint direction must be candidate-minus-source")
    source = _finite_required(source_endpoint, "source endpoint")
    candidate = _finite_required(candidate_endpoint, "candidate endpoint")
    se = _finite_required(biological_se, "biological SE")
    if se < 0.0:
        raise G0CandidateError("biological SE must be nonnegative")
    if isinstance(biological_replicate_count, bool) or not isinstance(biological_replicate_count, int) or biological_replicate_count < 2:
        raise G0CandidateError("biological replicate count must be at least two")
    return DirectionNormalizedEndpoint(
        source_id=source_id,
        candidate_id=candidate_id,
        candidate_minus_source=candidate - source,
        biological_se=se,
        biological_replicate_count=biological_replicate_count,
        endpoint_direction=endpoint_direction,
    )


def validate_only(candidate_path: Path = CANDIDATE_CONFIG_PATH) -> dict[str, Any]:
    candidate = load_candidate_config(candidate_path)
    fixture = (
        SyntheticEdge("SYN_SOURCE_A", "SYN_CANDIDATE_A"),
        SyntheticEdge("SYN_SOURCE_A", "SYN_CANDIDATE_B"),
        SyntheticEdge("SYN_SOURCE_B", "SYN_CANDIDATE_C"),
    )
    split = outcome_blind_synthetic_split(fixture)
    endpoint = validate_direction_normalized_synthetic_endpoint(
        source_id="SYN_SOURCE_A",
        candidate_id="SYN_CANDIDATE_A",
        source_endpoint=1.0,
        candidate_endpoint=1.25,
        biological_se=0.1,
        biological_replicate_count=2,
        endpoint_direction=ENDPOINT_DIRECTION,
    )
    runtime_truth = candidate["runtime_truth"]
    return {
        "status": DOCUMENT_STATUS,
        "authority_status": "NON_AUTHORITATIVE",
        "candidate_id": candidate["candidate_id"],
        "mode": "VALIDATE_ONLY_SYNTHETIC_G0",
        "synthetic_fixture_only": True,
        "component_count": len(split.components),
        "component_disjoint": True,
        "outcome_values_consumed_for_split": split.outcome_values_consumed,
        "final_assignment_materialized": split.final_assignment_materialized,
        "candidate_minus_source_valid": endpoint.candidate_minus_source == 0.25,
        "endpoint_direction": endpoint.endpoint_direction,
        "biological_se_rule_checked": True,
        "missing_or_nonfinite_coerced_to_zero": False,
        "runtime_truth": runtime_truth,
        "real_evaluation": False,
        "a2_pass_asserted": False,
        "scientific_claim_established": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true", help="run the only permitted synthetic G0 operation")
    parser.add_argument("--candidate-config", type=Path, default=CANDIDATE_CONFIG_PATH)
    args = parser.parse_args(argv)
    if not args.validate_only:
        parser.error("--validate-only is required; this candidate has no real-evaluation mode")
    print(json.dumps(validate_only(args.candidate_config), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
