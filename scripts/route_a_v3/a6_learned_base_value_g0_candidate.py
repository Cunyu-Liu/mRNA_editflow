#!/usr/bin/env python3
"""Non-authoritative G0 implementation candidate for the A6 learned protocol.

The active draft explicitly permits only schema/static-validator work.  This
separate candidate therefore implements pure interfaces and a zero-update
``--validate-only`` dry run.  It deliberately contains no Torch import, trainer,
optimizer, CUDA probe implementation, checkpoint I/O, data-row reader, or
runtime output writer.  A later authority must promote a reviewed successor
before any of those operations can exist or run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence


STAGING_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = STAGING_ROOT.parents[1]
PROTOCOL_CONFIG_PATH = STAGING_ROOT / "configs/route_a_v3_a6_learned_base_value_gpu_protocol_draft_v1.json"
CANDIDATE_CONFIG_PATH = STAGING_ROOT / "configs/route_a_v3_a6_learned_base_value_g0_implementation_candidate_v1.json"
PROTOCOL_VALIDATOR_PATH = STAGING_ROOT / "scripts/route_a_v3/validate_a6_learned_base_value_gpu_protocol_draft.py"

PROTOCOL_ID = "ROUTE_A_V3_A6_LEARNED_BASE_VALUE_GPU_PROTOCOL_DRAFT_V1"
CANDIDATE_ID = "ROUTE_A_V3_A6_LEARNED_BASE_VALUE_G0_IMPLEMENTATION_CANDIDATE_V1"
DOCUMENT_STATUS = "DRAFT_FOR_REVIEW_NOT_ACTIVE_PROTOCOL"
STATIC_ONLY_SCOPE = "PROTOCOL_SCHEMA_STATIC_VALIDATOR_AND_FOCUSED_TEST_ONLY"
INPUT_MANIFEST_SCHEMA = "route_a_v3_a6_g0_aggregate_input_contract_manifest.v1"


class G0CandidateError(RuntimeError):
    """Base error for the non-active implementation candidate."""


class ContractError(G0CandidateError):
    """A frozen draft, candidate, or aggregate input contract is invalid."""


class InactiveAuthorityError(G0CandidateError):
    """An execution operation was requested without active authority."""


class NumericalContractError(G0CandidateError):
    """A pure rate or terminal-boundary calculation is invalid."""


@dataclass(frozen=True)
class ZeroUpdateAudit:
    parameter_updates: int = 0
    optimizer_steps: int = 0
    model_constructions: int = 0
    model_forwards: int = 0
    cuda_probe_calls: int = 0
    gpu_runs: int = 0
    checkpoints_read: int = 0
    checkpoints_written: int = 0
    runtime_output_files_written: int = 0
    ordinary_rows_read: int = 0
    private_rows_read: int = 0
    sealed_rows_read: int = 0
    member_sequences_read: int = 0

    def assert_zero(self) -> None:
        nonzero = {key: value for key, value in asdict(self).items() if value != 0}
        if nonzero:
            raise ContractError(f"G0 audit is not zero: {nonzero}")


@dataclass(frozen=True)
class ExecutionAuthority:
    explicit_execution_authorization: bool = False
    active_protocol_registration: bool = False
    parameter_updates_authorized: bool = False
    qualified_data_manifest_frozen: bool = False
    split_rights_exposure_manifest_frozen: bool = False
    terminal_tilt_manifest_frozen: bool = False
    cuda_device_assignment_frozen: bool = False
    frozen_gpu_uuid: str | None = None


@dataclass(frozen=True)
class ArchitecturePlan:
    representation: str
    alphabet_width: int
    per_position_feature_width: int
    encoder_channels: int
    residual_dilations: tuple[int, ...]
    global_sequence_width: int
    context_width: int
    budget_width: int
    time_width: int
    state_vector_width: int
    action_vector_width: int
    base_rate_head: tuple[int, ...]
    value_head: tuple[int, ...]
    base_value_parameter_sharing: bool
    parameter_tensors_constructed: int


@dataclass(frozen=True)
class CanonicalActionPlan:
    action_type: str
    next_state: Any
    raw_alias_ids: tuple[str, ...]
    support_floor: float


@dataclass(frozen=True)
class GuidedGeneratorPlan:
    off_diagonal: tuple[tuple[str, float], ...]
    diagonal: float
    total_exit_hazard: float


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ContractError(f"non-finite JSON constant: {value}")


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ContractError(f"cannot read JSON contract: {path}") from exc
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid JSON contract: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON contract root must be an object: {path}")
    return value


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_protocol(path: Path = PROTOCOL_CONFIG_PATH) -> dict[str, Any]:
    payload = load_json_object(path)
    validator = _load_module(PROTOCOL_VALIDATOR_PATH, "route_a_v3_a6_learned_protocol_validator_g0")
    issues = validator.validate_config(payload)
    if issues:
        raise ContractError(f"parent draft static validation failed: {issues}")
    if payload["protocol_id"] != PROTOCOL_ID:
        raise ContractError("parent protocol ID differs")
    if payload["document_status"] != DOCUMENT_STATUS:
        raise ContractError("parent protocol status is no longer review-only")
    if payload["authority_status"] != "NON_AUTHORITATIVE":
        raise ContractError("parent protocol unexpectedly became authoritative")
    if payload["implementation_scope"] != STATIC_ONLY_SCOPE:
        raise ContractError("parent protocol implementation scope differs")
    return payload


def validate_candidate_config(payload: Mapping[str, Any]) -> None:
    expected_root = {
        "schema_version",
        "candidate_id",
        "document_status",
        "authority_status",
        "activation_state",
        "parent_protocol",
        "permitted_g0_operations",
        "forbidden_operations",
        "aggregate_input_interface",
        "existing_kernel_dependency",
        "runtime_truth",
    }
    if set(payload) != expected_root:
        raise ContractError("candidate config root closure differs")
    exact = {
        "schema_version": "route_a_v3_a6_learned_base_value_g0_implementation_candidate.v1",
        "candidate_id": CANDIDATE_ID,
        "document_status": DOCUMENT_STATUS,
        "authority_status": "NON_AUTHORITATIVE",
        "activation_state": "INACTIVE_IMPLEMENTATION_CANDIDATE",
    }
    for key, expected in exact.items():
        if payload[key] != expected:
            raise ContractError(f"candidate {key} differs")
    parent = payload["parent_protocol"]
    if parent != {
        "path": "configs/route_a_v3_a6_learned_base_value_gpu_protocol_draft_v1.json",
        "protocol_id": PROTOCOL_ID,
        "required_document_status": DOCUMENT_STATUS,
        "required_authority_status": "NON_AUTHORITATIVE",
        "required_implementation_scope": STATIC_ONLY_SCOPE,
        "must_remain_unmodified_by_candidate": True,
        "promotion_requires_new_explicit_owner_authority": True,
        "parameter_updates_require_separate_later_authority": True,
    }:
        raise ContractError("candidate parent-protocol boundary differs")
    required_forbidden = {
        "TRAIN",
        "OPTIMIZER_STEP",
        "PARAMETER_UPDATE",
        "CUDA_PROBE_OR_GPU_TOUCH",
        "CHECKPOINT_READ_OR_WRITE",
        "RUNTIME_OUTPUT_FILE_WRITE",
        "PRIVATE_OR_SEALED_ACCESS",
        "MEMBER_ROW_OR_SEQUENCE_ACCESS",
        "A6_PASS_OR_L3_CLAIM",
        "A7_UNLOCK",
    }
    if not required_forbidden.issubset(set(payload["forbidden_operations"])):
        raise ContractError("candidate forbidden-operation closure is incomplete")
    runtime_truth = payload["runtime_truth"]
    for key, value in runtime_truth.items():
        expected = False if isinstance(value, bool) else 0
        if value != expected:
            raise ContractError(f"candidate runtime truth is nonzero: {key}")
    interface = payload["aggregate_input_interface"]
    if interface["schema_version"] != INPUT_MANIFEST_SCHEMA or interface["member_payload_allowed"] is not False:
        raise ContractError("aggregate input interface is not fail-closed")
    if payload["existing_kernel_dependency"]["validate_only_loads_dependency"] is not False:
        raise ContractError("validate-only must not load the execution kernel")


def load_candidate_config(path: Path = CANDIDATE_CONFIG_PATH) -> dict[str, Any]:
    payload = load_json_object(path)
    validate_candidate_config(payload)
    return payload


def build_architecture_plan(protocol: Mapping[str, Any]) -> ArchitecturePlan:
    """Build a shape-only plan; this constructs no learned parameters."""

    base = protocol["base_architecture"]
    value = protocol["value_architecture"]
    encoder = base["encoder"]
    position_width = 5 + 5 + 1 + 1 + 16
    global_width = 2 * int(encoder["residual_block_channels"])
    context_width = int(base["observable_context_encoder"]["embedding_width"])
    budget_width = int(base["remaining_budget_embedding_width"])
    time_width = len(base["algorithmic_time_features"])
    state_width = global_width + context_width + budget_width + time_width
    action_width = int(encoder["residual_block_channels"]) + state_width + 5 + 1 + 1
    observed = {
        "per_position_feature_width": base["per_position_feature_width"],
        "global_sequence_width": encoder["global_sequence_width"],
        "state_vector_width": base["state_vector_width"],
        "action_vector_width": base["action_vector_width"],
        "value_state_vector_width": value["state_vector_width"],
    }
    expected = {
        "per_position_feature_width": position_width,
        "global_sequence_width": global_width,
        "state_vector_width": state_width,
        "action_vector_width": action_width,
        "value_state_vector_width": state_width,
    }
    if observed != expected:
        raise ContractError(f"architecture width closure failed: expected={expected}, observed={observed}")
    if value["base_encoder_parameters_shared"] is not False:
        raise ContractError("base/value parameters must be separate")
    return ArchitecturePlan(
        representation="PURE_SHAPE_PLAN_NO_TORCH_NO_PARAMETERS",
        alphabet_width=len(base["alphabet"]),
        per_position_feature_width=position_width,
        encoder_channels=int(encoder["residual_block_channels"]),
        residual_dilations=tuple(encoder["residual_dilations"]),
        global_sequence_width=global_width,
        context_width=context_width,
        budget_width=budget_width,
        time_width=time_width,
        state_vector_width=state_width,
        action_vector_width=action_width,
        base_rate_head=tuple([action_width, *base["rate_head_hidden_widths"], base["rate_head_output_width"]]),
        value_head=tuple([state_width, *value["value_head_hidden_widths"], value["value_head_output_width"]]),
        base_value_parameter_sharing=False,
        parameter_tensors_constructed=0,
    )


def _expect_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ContractError(f"{label} keys differ: missing={sorted(expected - set(value))}, extra={sorted(set(value) - expected)}")


def validate_aggregate_input_contract_manifest(
    manifest: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate only an aggregate future-input contract, never member rows."""

    root_keys = {
        "schema_version",
        "aggregate_only",
        "contains_member_payload",
        "data_scope",
        "record_role",
        "qualification_status",
        "declared_fields",
        "forbidden_roles_present",
        "split",
        "rights",
        "exposure",
    }
    _expect_exact_keys(manifest, root_keys, "input manifest")
    if manifest["schema_version"] != INPUT_MANIFEST_SCHEMA:
        raise ContractError("input manifest schema differs")
    if manifest["aggregate_only"] is not True or manifest["contains_member_payload"] is not False:
        raise ContractError("member payload is forbidden in G0 input contracts")
    data = protocol["ordinary_public_data_contract"]
    if manifest["data_scope"] != data["data_scope"]:
        raise ContractError("data scope is not qualified ordinary-public only")
    if manifest["record_role"] != data["allowed_record_role"]:
        raise ContractError("record role is not the frozen development role")
    if manifest["qualification_status"] != "QUALIFIED_UNDER_FUTURE_ACTIVE_AUTHORITY":
        raise ContractError("dataset qualification is absent")
    if set(manifest["declared_fields"]) != set(data["allowed_fields"]):
        raise ContractError("declared field interface differs from the frozen allowed set")
    if manifest["forbidden_roles_present"] != []:
        raise ContractError("a forbidden data role is present")

    split = manifest["split"]
    split_keys = {
        "parent_split_authority",
        "frozen",
        "label_blind",
        "components_indivisible",
        "assignment_unit",
        "development_subroles",
        "leakage_counts",
        "retry_after_labels_or_results",
        "outer_test_label_accessed",
    }
    _expect_exact_keys(split, split_keys, "split")
    frozen_split = protocol["split_contract"]
    split_expectations = {
        "parent_split_authority": frozen_split["parent_split_authority"],
        "frozen": True,
        "label_blind": True,
        "components_indivisible": True,
        "assignment_unit": frozen_split["assignment_unit"],
        "development_subroles": frozen_split["development_subroles"],
        "retry_after_labels_or_results": False,
        "outer_test_label_accessed": False,
    }
    for key, expected in split_expectations.items():
        if split[key] != expected:
            raise ContractError(f"split contract failed: {key}")
    expected_leakage = {
        "source_group": 0,
        "exact_sequence": 0,
        "near_duplicate": 0,
        "reverse_edge": 0,
        "candidate": 0,
        "study_context": 0,
    }
    if split["leakage_counts"] != expected_leakage:
        raise ContractError("split leakage is nonzero or incompletely reported")

    rights = manifest["rights"]
    rights_keys = {
        "qualification_use_authorized",
        "private_processing_and_evaluation_authorized",
        "rights_status",
        "raw_or_member_level_redistribution_allowed",
    }
    _expect_exact_keys(rights, rights_keys, "rights")
    if rights != {
        "qualification_use_authorized": True,
        "private_processing_and_evaluation_authorized": True,
        "rights_status": "PASS_UNDER_FUTURE_ACTIVE_AUTHORITY",
        "raw_or_member_level_redistribution_allowed": False,
    }:
        raise ContractError("rights contract is not closed")

    exposure = manifest["exposure"]
    exposure_keys = {
        "model_input_route",
        "pretrained_foundation_checkpoints",
        "pretrained_weights",
        "warm_start_checkpoints",
        "external_learned_embeddings",
        "external_pretraining_corpora",
        "checkpoint_loads_before_first_update",
    }
    _expect_exact_keys(exposure, exposure_keys, "exposure")
    frozen_exposure = protocol["exposure_contract"]
    if exposure["model_input_route"] != frozen_exposure["model_input_route"]:
        raise ContractError("exposure route is not scratch-only")
    for key in (
        "pretrained_foundation_checkpoints",
        "pretrained_weights",
        "warm_start_checkpoints",
        "external_learned_embeddings",
        "external_pretraining_corpora",
    ):
        if exposure[key] != []:
            raise ContractError(f"external learned exposure is nonempty: {key}")
    if exposure["checkpoint_loads_before_first_update"] != 0:
        raise ContractError("a checkpoint would be loaded before the first update")

    return {
        "status": "PASS_INTERFACE_METADATA_ONLY_NOT_DATA_QUALIFICATION",
        "member_payload_reads": 0,
        "ordinary_row_reads": 0,
        "private_row_reads": 0,
        "sealed_row_reads": 0,
    }


def assign_primary_budget(net_edit_count: int, protocol: Mapping[str, Any]) -> int:
    if isinstance(net_edit_count, bool) or not isinstance(net_edit_count, int) or net_edit_count < 0:
        raise ContractError("net edit count must be a nonnegative integer")
    for budget in protocol["formal_production_interface"]["primary_edit_budgets"]:
        if net_edit_count <= budget:
            return int(budget)
    raise ContractError("net edit count exceeds the frozen primary budget support")


def canonical_action_plans(
    kernel: ModuleType,
    state: Any,
    kernel_config: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[CanonicalActionPlan, ...]:
    """Reuse the existing synthetic kernel's legality and transition interface."""

    required = {"validate_state", "raw_actions", "is_action_legal", "transition_state", "state_sort_key"}
    missing = sorted(name for name in required if not hasattr(kernel, name))
    if missing:
        raise ContractError(f"existing kernel interface is incomplete: {missing}")
    kernel.validate_state(state, kernel_config)
    grouped: dict[Any, list[Any]] = {}
    for action in kernel.raw_actions(state, kernel_config):
        if not kernel.is_action_legal(state, action, kernel_config):
            raise ContractError("kernel emitted an action before hard legality")
        child = kernel.transition_state(state, action, kernel_config)
        grouped.setdefault(child, []).append(action)
    if not grouped and getattr(state, "terminal_cause", None) is None:
        raise ContractError("nonterminal state has no canonical STOP/edit support")
    support_floor = float(protocol["formal_production_interface"]["canonical_support_floor"])
    if not math.isfinite(support_floor) or support_floor <= 0.0:
        raise ContractError("canonical support floor is invalid")
    plans = []
    for child, actions in grouped.items():
        action_types = {action.action_type for action in actions}
        action_type = next(iter(action_types)) if len(action_types) == 1 else "OBSERVABLE_TRANSITION_ALIAS"
        plans.append(
            CanonicalActionPlan(
                action_type=action_type,
                next_state=child,
                raw_alias_ids=tuple(sorted(action.alias_id for action in actions)),
                support_floor=support_floor,
            )
        )
    plans.sort(key=lambda plan: kernel.state_sort_key(plan.next_state))
    return tuple(plans)


def _softplus(value: float) -> float:
    if not math.isfinite(value):
        raise NumericalContractError("base logit is not finite")
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def normalized_base_rates(
    logits: Mapping[str, float], protocol: Mapping[str, Any]
) -> dict[str, float]:
    if not logits:
        raise NumericalContractError("at least one canonical transition is required")
    interface = protocol["formal_production_interface"]
    floor = float(interface["canonical_support_floor"])
    total_hazard = float(interface["base_total_exit_hazard"])
    raw = {key: floor + _softplus(float(value)) for key, value in logits.items()}
    denominator = math.fsum(raw.values())
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise NumericalContractError("base-rate denominator is invalid")
    rates = {key: total_hazard * value / denominator for key, value in raw.items()}
    if any(not math.isfinite(value) or value <= 0.0 for value in rates.values()):
        raise NumericalContractError("base support is not strictly positive")
    if not math.isclose(math.fsum(rates.values()), total_hazard, rel_tol=0.0, abs_tol=1e-12):
        raise NumericalContractError("base total exit hazard is not frozen")
    return rates


def guided_generator_plan(
    base_rates: Mapping[str, float],
    current_potential: float,
    next_potentials: Mapping[str, float],
) -> GuidedGeneratorPlan:
    if set(base_rates) != set(next_potentials):
        raise NumericalContractError("base-rate and next-potential supports differ")
    if not math.isfinite(current_potential):
        raise NumericalContractError("current potential is not finite")
    off_diagonal: list[tuple[str, float]] = []
    for key in sorted(base_rates):
        base = float(base_rates[key])
        nxt = float(next_potentials[key])
        if not math.isfinite(base) or base <= 0.0 or not math.isfinite(nxt):
            raise NumericalContractError("guided-rate input is invalid")
        try:
            rate = base * math.exp(nxt - current_potential)
        except OverflowError as exc:
            raise NumericalContractError("guided-rate exponential overflow") from exc
        if not math.isfinite(rate) or rate <= 0.0:
            raise NumericalContractError("guided rate is not finite and positive")
        off_diagonal.append((key, rate))
    total = math.fsum(rate for _, rate in off_diagonal)
    return GuidedGeneratorPlan(tuple(off_diagonal), -total, total)


def terminal_boundary(standardized_lcb: float, protocol: Mapping[str, Any]) -> tuple[float, float]:
    if not math.isfinite(standardized_lcb):
        raise NumericalContractError("standardized terminal score is not finite")
    tilt = protocol["terminal_tilt_contract"]
    low, high = map(float, tilt["standardized_score_clip"])
    beta = float(tilt["beta"])
    clipped = min(max(beta * standardized_lcb, low), high)
    weight = math.exp(clipped)
    potential = math.log(weight)
    if not math.isfinite(weight) or weight <= 0.0 or not math.isclose(potential, clipped, abs_tol=1e-15):
        raise NumericalContractError("terminal absorbing boundary is invalid")
    return weight, potential


def build_future_manifest_plan(protocol: Mapping[str, Any]) -> dict[str, Any]:
    reference = protocol["independent_exact_reference"]
    gate = protocol["learned_potential_approximation_gate"]
    return {
        "status": "PLAN_ONLY_NOT_CREATED",
        "required_if_later_authorized": list(protocol["future_provenance_and_manifest_outputs"]["required_if_later_authorized"]),
        "objective_order": [
            protocol["training_objectives"]["base_objective"]["name"],
            protocol["training_objectives"]["value_objective"]["name"],
        ],
        "reference": {
            "implementation_role": reference["implementation_role"],
            "graph_count": reference["graph_count"],
            "graphs_per_budget": reference["graphs_per_budget"],
            "budgets": reference["budgets"],
            "dp_vs_enumeration_terminal_tv_max": reference["dp_vs_enumeration_terminal_tv_max"],
        },
        "approximation_gate": {
            "primary_estimand": gate["primary_estimand"],
            "primary_threshold_max": gate["primary_threshold_max"],
            "secondary_estimand": gate["secondary_estimand"],
            "secondary_threshold_max": gate["secondary_threshold_max"],
        },
        "required_gates": protocol["required_gates"],
        "files_created": 0,
    }


def _reject_inactive_operation(
    operation: str,
    protocol: Mapping[str, Any],
    authority: ExecutionAuthority,
    audit: ZeroUpdateAudit,
) -> None:
    audit.assert_zero()
    missing = [
        name
        for name, present in (
            ("explicit_execution_authorization", authority.explicit_execution_authorization),
            ("active_protocol_registration", authority.active_protocol_registration),
            ("parameter_updates_authorized", authority.parameter_updates_authorized),
            ("qualified_data_manifest_frozen", authority.qualified_data_manifest_frozen),
            ("split_rights_exposure_manifest_frozen", authority.split_rights_exposure_manifest_frozen),
            ("terminal_tilt_manifest_frozen", authority.terminal_tilt_manifest_frozen),
            ("cuda_device_assignment_frozen", authority.cuda_device_assignment_frozen),
        )
        if not present
    ]
    parent_inactive = (
        protocol["authority_boundary"]["registered_in_active_authority"] is False
        and protocol["activation_state"] == "INACTIVE_REVIEW_CANDIDATE"
    )
    if parent_inactive or missing:
        raise InactiveAuthorityError(
            f"{operation} rejected before callback/CUDA/I/O; parent_inactive={parent_inactive}; missing={missing}; parameter_updates=0; gpu_runs=0; outputs=0"
        )
    raise InactiveAuthorityError(
        f"{operation} is intentionally absent from the G0 candidate and requires a promoted successor; parameter_updates=0; gpu_runs=0; outputs=0"
    )


def request_training(
    protocol: Mapping[str, Any],
    authority: ExecutionAuthority,
    train_callback: Callable[[], Any],
    audit: ZeroUpdateAudit = ZeroUpdateAudit(),
) -> None:
    _reject_inactive_operation("TRAIN", protocol, authority, audit)


def request_optimizer_step(
    protocol: Mapping[str, Any],
    authority: ExecutionAuthority,
    optimizer_callback: Callable[[], Any],
    audit: ZeroUpdateAudit = ZeroUpdateAudit(),
) -> None:
    _reject_inactive_operation("OPTIMIZER_STEP", protocol, authority, audit)


def request_checkpoint_write(
    protocol: Mapping[str, Any],
    authority: ExecutionAuthority,
    checkpoint_callback: Callable[[], Any],
    audit: ZeroUpdateAudit = ZeroUpdateAudit(),
) -> None:
    _reject_inactive_operation("CHECKPOINT_WRITE", protocol, authority, audit)


def cuda_ownership_preflight(
    protocol: Mapping[str, Any],
    authority: ExecutionAuthority,
    cuda_probe: Callable[[], Mapping[str, Any]],
    audit: ZeroUpdateAudit = ZeroUpdateAudit(),
) -> None:
    """Authority barrier precedes the injected probe, so G0 never touches CUDA."""

    _reject_inactive_operation("CUDA_OWNERSHIP_PREFLIGHT", protocol, authority, audit)


def validate_only(
    *,
    protocol_path: Path = PROTOCOL_CONFIG_PATH,
    candidate_path: Path = CANDIDATE_CONFIG_PATH,
    input_contract_manifest_path: Path | None = None,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    candidate = load_candidate_config(candidate_path)
    architecture = build_architecture_plan(protocol)
    audit = ZeroUpdateAudit()
    audit.assert_zero()
    if input_contract_manifest_path is None:
        input_interface: dict[str, Any] = {
            "status": "INTERFACE_ONLY_NO_DATA_MANIFEST_READ",
            "member_payload_reads": 0,
            "ordinary_row_reads": 0,
            "private_row_reads": 0,
            "sealed_row_reads": 0,
        }
    else:
        input_interface = validate_aggregate_input_contract_manifest(
            load_json_object(input_contract_manifest_path), protocol
        )
    return {
        "status": DOCUMENT_STATUS,
        "authority_status": "NON_AUTHORITATIVE",
        "candidate_id": candidate["candidate_id"],
        "mode": "VALIDATE_ONLY_ZERO_UPDATE_DRY_RUN",
        "parent_protocol_valid": True,
        "candidate_contract_valid": True,
        "parent_implementation_scope_unchanged": protocol["implementation_scope"],
        "architecture_plan": asdict(architecture),
        "input_contract_interface": input_interface,
        "future_manifest_plan": build_future_manifest_plan(protocol),
        "kernel_loaded": False,
        "torch_imported": False,
        "cuda_probe_invoked": False,
        "audit": asdict(audit),
        "training_allowed": False,
        "a6_pass_asserted": False,
        "l3_claim_established": False,
        "a7_unlocked": False,
        "qualification_or_credit_delta": 0,
        "canonical_delta": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true", help="run the only permitted zero-update operation")
    parser.add_argument("--protocol-config", type=Path, default=PROTOCOL_CONFIG_PATH)
    parser.add_argument("--candidate-config", type=Path, default=CANDIDATE_CONFIG_PATH)
    parser.add_argument("--input-contract-manifest", type=Path)
    args = parser.parse_args(argv)
    if not args.validate_only:
        parser.error("--validate-only is required; this candidate has no execution mode")
    result = validate_only(
        protocol_path=args.protocol_config,
        candidate_path=args.candidate_config,
        input_contract_manifest_path=args.input_contract_manifest,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
