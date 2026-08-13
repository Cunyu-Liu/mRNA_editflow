#!/usr/bin/env python3
"""Adjudicate GSE200304 under the DEC-020 scratch-only input route.

This V4 successor reuses exactly seven accepted DEC-019 aggregate gate records.
It does not open the predecessor checkpoint-exposure negative record, row-level
data, private payloads, or sealed payloads.  A PASS is route-scoped scientific
qualification only: it authorizes neither materialization execution nor model
training, model selection, GPU use, or the next phase.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_REPO_PATH = (
    "configs/route_a_v3_gse200304_dec020_reported_endpoint_a1_activation_v4.json"
)
SCRIPT_REPO_PATH = (
    "scripts/route_a_v3/adjudicate_gse200304_dec020_reported_endpoint_a1_v4.py"
)
TEST_REPO_PATH = (
    "tests/route_a_v3/test_adjudicate_gse200304_dec020_reported_endpoint_a1_v4.py"
)
PREDECESSOR_V3_CONFIG_REPO_PATH = (
    "configs/route_a_v3_gse200304_dec019_reported_endpoint_a1_activation_v3.json"
)
PREDECESSOR_V3_SCRIPT_REPO_PATH = (
    "scripts/route_a_v3/adjudicate_gse200304_dec019_reported_endpoint_a1_v3.py"
)
PREDECESSOR_V3_TEST_REPO_PATH = (
    "tests/route_a_v3/test_adjudicate_gse200304_dec019_reported_endpoint_a1_v3.py"
)
PREDECESSOR_V3_FROZEN_BLOBS = {
    PREDECESSOR_V3_CONFIG_REPO_PATH: (
        "0716e4f1a96280d7e33858df037e05d975119a82ff36ee6794f5ffac1c92bb44"
    ),
    PREDECESSOR_V3_SCRIPT_REPO_PATH: (
        "7b79ca1a5fff8bd2640234fe30bdbf39533c52e73aa21a93341ed7ee8e34db53"
    ),
    PREDECESSOR_V3_TEST_REPO_PATH: (
        "a985a94c8258ea58dc7d83103050284aed41dcc4c2dcf997253367d0e6b1a1cf"
    ),
}

AUTHORITY_COMMIT = "d0611622f304d2d621a35b190922ac593a3b8788"
AUTHORITY_A_EXACT_CHANGED_BLOBS = {
    "configs/route_a_v3.yaml": (
        "c908ac57b7c9667398f616a0ccf7101b41451b80bf169e768131844d3b63a678"
    ),
    "configs/route_a_v3_a1_qualification.json": (
        "ac1ed9e78bf88d916f5599e3a2e75e79df1504c16ba108a12f7e28cfd3da2e20"
    ),
    "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec020.yaml": (
        "0cfbe6e35c2c7f3b19756b8aee41dc91b2a8f05b249a5b6e9cacf90185c56026"
    ),
    "docs/contracts/supersession_mrna_xeditflow_v1_1_to_route_a_v3.yaml": (
        "d7c0559742a44b4f0b6f8c941e734da52359c2733ba759ec2acd8ca40b07e62d"
    ),
    "docs/execution/route_a_v3_a1_interim.yaml": (
        "615cbb768d819f8acddfb6a5e86a59f9da21c342598caea267bf6ef101efe683"
    ),
    "docs/execution/route_a_v3_claim_evidence_matrix.yaml": (
        "9f5226ac78dd6c3848ba5ceb42742918de66ec459f951bb845ccaf21958a88f9"
    ),
    "docs/execution/route_a_v3_data_role_registry.yaml": (
        "d06bfcfb8d265153a44d270c7bc40e5dd462a5e3bdde631d91519c7d7e394852"
    ),
    "docs/execution/route_a_v3_decision_log.yaml": (
        "1332e789758a11687d3bcbbe95e0a5c7e852694e25ed90563d280006d94caced"
    ),
    "docs/execution/route_a_v3_registry_manifest.json": (
        "2d6f7166ad60fa7486659069a0e6694a4ea42f6391bd08f6f5e0f5848dd5ea6b"
    ),
    "docs/execution/route_a_v3_split_registry.yaml": (
        "52e1146027956e024dd6194ff18862e542e27fff81e8fc6b6d8aeaa972b8259c"
    ),
    "docs/execution/route_a_v3_task_registry.yaml": (
        "bf3066a7534041374685e9ebe9ac8c840e53ceec1acbb076a72a758d397c63f2"
    ),
    "docs/execution/route_a_v3_task_split_matrix.yaml": (
        "db23e96b6977339237956de57309d04a9e692bf937a8d34427d2e1b6cc150db8"
    ),
    "scripts/route_a_v3/validate_a0_bundle.py": (
        "e75ca6fb98b45122e9ee88e028a3ed34d5f40d70f5cd4cd25c9b4e446c26e2cc"
    ),
    "tests/route_a_v3/test_a0_integrity_guards.py": (
        "b3a1ea125a2264c88db422f5796aca0fcbafa85b5c5be5a3ed08c53bb9d5846c"
    ),
}
ROOT_CONTRACT_FROZEN_BLOB = {
    "docs/goals/MRNA_XEDITFLOW_ROUTE_A_V3.md": (
        "cbac4c3dcba8f1b8df95d8edad52d19e3c126d1c865d0cc423537c754cc90982"
    )
}

AUTHORITY_RUNTIME_CONFIG_REPO_PATH = (
    "configs/route_a_v3_dec020_authority_runtime_sync_v1.json"
)
AUTHORITY_RUNTIME_SCRIPT_REPO_PATH = (
    "scripts/route_a_v3/dec020_authority_runtime_sync.py"
)
AUTHORITY_RUNTIME_TEST_REPO_PATH = (
    "tests/route_a_v3/test_dec020_authority_runtime_sync.py"
)
AUTHORITY_RUNTIME_I1_COMMIT = "4606ec30e74610485ef1ba158c25d8b626d4ae1a"
AUTHORITY_RUNTIME_I1_CONFIG_BYTES = 12042
AUTHORITY_RUNTIME_I1_CONFIG_SHA256 = (
    "f45db39f3e1df251c5f0fd430b7e9e2467e8b5ba838dd04d79458e6ef65c4471"
)
AUTHORITY_RUNTIME_I1_SCRIPT_SHA256 = (
    "03bbef4de6f80f314a3ebc7c13307bf18b2294ee70953870cf08e0531eebc05e"
)
AUTHORITY_RUNTIME_I1_TEST_SHA256 = (
    "da42ac3160ea2518746d8b40d1d964c41f82214903f627d1e4e3f5a65528ca71"
)
AUTHORITY_RUNTIME_I2_COMMIT = "17d0f570bdfb4bf4a3e5ff34cb1d3aa11a2cccdd"
AUTHORITY_RUNTIME_B2_COMMIT = "fb21121525ca13692a4619115f09e99fd99c122a"
AUTHORITY_RUNTIME_I2_CONFIG_SHA256 = (
    "eb9de12f8e8ae98c54a2542831f9b976367f7b7e6eee0d68dcf3ef28ec7f3850"
)
AUTHORITY_RUNTIME_B2_CONFIG_SHA256 = (
    "4f615b1be17fabe7e4f5b4b67845b988f390f2c286a80e8fe0ea52b9b9c4aab9"
)
AUTHORITY_RUNTIME_CONFIG_CORE_SHA256 = (
    "891b99f633c214e4d92e4244a60e50f34b883776af755b7bd870572bee41badf"
)
AUTHORITY_RUNTIME_I2_SCRIPT_SHA256 = (
    "fe4c7f19eecef91b9fd1340f4ea8258e644a6b0139d2880a8a84640bd4721862"
)
AUTHORITY_RUNTIME_I2_TEST_SHA256 = (
    "2909a34d6c1b152f91bfe0a336290066dc73bee8411db70814068e288a9378b1"
)

PRODUCTION_REPO_ROOT = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/routea_v3_a1_20260810"
)
PRODUCTION_CONFIG_PATH = PRODUCTION_REPO_ROOT / CONFIG_REPO_PATH
TRUSTED_A1_OUTPUT_ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1")

UNKNOWN = "UNKNOWN_NOT_ASSERTED"
CONTRACT_ID = "mrna_xeditflow_route_a_v3"
DATASET_ID = "GSE200304"
DECISION_ID = "V3-DEC-020"
PROTOCOL_ID = "ROUTE_A_V3_GSE200304_DEC020_REPORTED_ENDPOINT_A1_ACTIVATION_V4"
SCRATCH_ROUTE = "SCRATCH_ONLY_NO_FOUNDATION_NO_EXTERNAL_LEARNED_INPUTS"
SCRATCH_EXPOSURE_STATUS = (
    "NOT_APPLICABLE_BY_FROZEN_NO_EXTERNAL_LEARNED_INPUT_ROUTE"
)
FOUNDATION_ROUTE_STATUS = "RETAINED_FAIL_CURRENT_PROTOCOL"
FOUNDATION_CHECKPOINT_EVIDENCE_STATUS = UNKNOWN
EVIDENCE_DECISION_ID = "V3-DEC-019"
EVIDENCE_SCHEMA_VERSION = "route_a_v3_dec019_aggregate_gate_evidence.v3"
EVIDENCE_RECORD_TYPE = "ROUTE_A_V3_DEC019_ACCEPTED_AGGREGATE_GATE_EVIDENCE_V3"
EVIDENCE_ACCEPTANCE_AUTHORITY = {
    "contract_id": CONTRACT_ID,
    "decision_id": EVIDENCE_DECISION_ID,
    "protocol_id": "ROUTE_A_V3_GSE200304_DEC019_REPORTED_ENDPOINT_A1_ACTIVATION_V3",
    "rule": "CONFIG_HASH_BOUND_ACCEPTED_AGGREGATE_GATE_RECORD_V3",
}
PRE_AUTHORITY_BASE_COMMIT = "ba6746adeb5dc9b2b41a69d139912006e0f5ad07"
FROZEN_CONFIG_CORE_SHA256 = (
    "ae51c0682e1e45a062f3e1dcb7ee5d2defb684977591877650d9edcc06827545"
)
CANONICAL_RECORD_COUNT = 6547
BIOLOGICAL_GROUP_COUNT = 6544
BLOCKED_STATUS = "BLOCKED_DEC020_SCRATCH_ROUTE_A1_EVIDENCE_INCOMPLETE"
SUCCESS_STATUS = "PASS_DEC020_SCRATCH_ROUTE_SCOPED_REPORTED_ENDPOINT_A1_QUALIFIED"
OUTPUT_ID = "ROUTE_A_V3_GSE200304_DEC020_SCRATCH_ROUTE_A1_ADJUDICATION_BUNDLE_V4"
PUBLICATION_MODE = "ATOMIC_EXCLUSIVE_DIRECTORY_TERMINAL_COMMIT_MARKER_V1"
COMMIT_MARKER = "PUBLICATION_COMMIT.json"
OUTPUT_JSON_NAMES = ("ADJUDICATION_REPORT.json", "INPUT_EVIDENCE_AUDIT.json")
OUTPUT_NAMES_EXCLUDING_MARKER = (*OUTPUT_JSON_NAMES, "SHA256SUMS")

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
REASON_CODE = re.compile(r"^[A-Z][A-Z0-9_]*$")
GROUP_MAPPING_COMMITMENT_KEY = "group_mapping_commitment_sha256"
SPLIT_ASSIGNMENT_COMMITMENT_KEY = "split_assignment_commitment_sha256"
LOCATOR_LINEAGE_COMMITMENT_ALGORITHM = "ROUTE_A_V3_GSE200304_LOCATOR_MERKLE_V1"
POWER_ANALYSIS_UNIT = "BIOLOGICAL_SOURCE_GROUP"
POWER_EVALUATION_POPULATION = (
    "A1_ELIGIBLE_OUTCOME_BLIND_OUTER_OOF_GROUPS_NOT_A2_FINAL_MEMBERSHIP"
)
POWER_TARGET_METRIC = "WITHIN_STUDY_SPEARMAN"
POWER_METHOD = "BONETT_WRIGHT_FISHER_Z_ASYMPTOTIC_TWO_SIDED_SPEARMAN"
POWER_CI_METHOD = "BONETT_WRIGHT_FISHER_Z_SPEARMAN_AT_PREFROZEN_ALTERNATIVE"
POWER_WORKING_DISTRIBUTION_ASSUMPTION = (
    "MONOTONIC_TRANSFORMATION_OF_BIVARIATE_NORMAL_AT_PREFROZEN_SPEARMAN_RHO"
)

SLOT_IDS = (
    "CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE",
    "CANONICAL_REPORTED_ENDPOINT_SEMANTICS",
    "BIOLOGICAL_GROUP_AUTHORITY",
    "ROW_REPLICATE_OR_VALID_SE",
    "LICENSE_RIGHTS",
    "OUTCOME_BLIND_SPLIT_LEAKAGE",
    "PREFROZEN_POWER_PRECISION",
)
EXPECTED_I_TO_B_SCALAR_PATHS = [
    "implementation_binding.status",
    "implementation_binding.implementation_commit",
    "implementation_binding.implementation_script_sha256",
    "implementation_binding.implementation_test_sha256",
]
EXPECTED_I_PATHS = sorted((CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH))
EXPECTED_B_PATHS = [CONFIG_REPO_PATH]
EXPECTED_AUTHORITY_A_PATHS = sorted(AUTHORITY_A_EXACT_CHANGED_BLOBS)
EXPECTED_AUTHORITY_RUNTIME_I1_PATHS = sorted(
    (
        AUTHORITY_RUNTIME_CONFIG_REPO_PATH,
        AUTHORITY_RUNTIME_SCRIPT_REPO_PATH,
        AUTHORITY_RUNTIME_TEST_REPO_PATH,
    )
)
EXPECTED_AUTHORITY_RUNTIME_I_PATHS = list(EXPECTED_AUTHORITY_RUNTIME_I1_PATHS)
EXPECTED_AUTHORITY_RUNTIME_B_PATHS = [AUTHORITY_RUNTIME_CONFIG_REPO_PATH]
EXPECTED_AUTHORITY_RUNTIME_I_TO_B_SCALAR_PATHS = set(EXPECTED_I_TO_B_SCALAR_PATHS)
AUTHORITY_RUNTIME_DYNAMIC_FIELDS = (
    "authority_runtime_implementation_commit",
    "authority_runtime_binding_commit",
    "authority_runtime_implementation_script_sha256",
    "authority_runtime_implementation_test_sha256",
    "base_commit",
    "implementation_commit_expected_parent",
)
EXPECTED_CONFIG_TOP_KEYS = {
    "schema_version",
    "protocol_id",
    "contract_id",
    "phase_id",
    "dataset_id",
    "decision_id",
    "implementation_binding",
    "repository_authority",
    "core_authority",
    "model_input_route_contract",
    "policy_boundary",
    "current_external_state",
    "evidence_contract",
    "output_contract",
    "evidence_descriptor_bindings",
}
EXPECTED_IMPLEMENTATION_BINDING_KEYS = {
    "binding_scheme",
    "status",
    "blocker_if_unbound",
    "implementation_commit",
    "implementation_script_path",
    "implementation_script_sha256",
    "implementation_test_path",
    "implementation_test_sha256",
    "config_core_sha256",
    "unknown_to_bound_scalar_paths",
}
FORBIDDEN_EXTERNAL_LEARNED_INPUTS = [
    "ANY_PARAMETER_TENSOR_FEATURE_OR_SCORE_LEARNED_FROM_EXTERNAL_DATA",
    "CHECKPOINT_DERIVED_CALIBRATION_OR_NORMALIZATION_STATISTICS",
    "EXTERNAL_LEARNED_REPRESENTATIONS_FEATURES_OR_LOGITS",
    "EXTERNAL_PSEUDOLABELS",
    "EXTERNAL_TEACHER_OR_DISTILLATION_TARGETS",
    "EXTERNALLY_LEARNED_TOKEN_EMBEDDINGS",
    "EXTERNALLY_TRAINED_ENCODER_ADAPTER_OR_HEAD",
    "EXTERNALLY_TRAINED_INITIALIZATION_OR_WARM_START",
    "FOUNDATION_OR_OTHER_PRETRAINED_CHECKPOINT_WEIGHTS",
    "LEARNED_RETRIEVAL_INDEX_RERANKER_OR_SCORE",
]
SCRATCH_ROUTE_KEYS = {
    "route_status",
    "checkpoint_specific_exposure_gate_applicable",
    "checkpoint_specific_exposure_status",
    "checkpoint_specific_exposure_pass_claimed",
    "allowed_parameter_initialization",
    "external_checkpoint_count",
    "external_learned_input_count",
    "pretrained_weights_present",
    "warm_start_present",
    "external_embedding_present",
    "external_learned_feature_present",
    "external_logits_present",
    "external_teacher_or_distillation_target_present",
    "external_pseudolabel_present",
    "checkpoint_derived_statistic_present",
    "learned_retrieval_or_reranker_present",
}
SCRATCH_ABSENCE_FIELDS = {
    "pretrained_weights_present",
    "warm_start_present",
    "external_embedding_present",
    "external_learned_feature_present",
    "external_logits_present",
    "external_teacher_or_distillation_target_present",
    "external_pseudolabel_present",
    "checkpoint_derived_statistic_present",
    "learned_retrieval_or_reranker_present",
}

PRIVACY_KEYS = {
    "contains_row_level_payload",
    "contains_sequence",
    "contains_row_identifier",
    "contains_raw_label_or_effect",
    "contains_member_identifiers_or_hashes",
}
COMMON_EVIDENCE_KEYS = {
    "schema_version",
    "record_type",
    "contract_id",
    "decision_id",
    "dataset_id",
    "gate_id",
    "status",
    "accepted",
    "aggregate_only",
    "privacy",
    "provenance",
    "facts",
    "unknown_fields",
    "reason_codes",
}
PROVENANCE_KEYS = {
    "producer_protocol_id",
    "producer_commit",
    "producer_script_sha256",
    "source_bundle_id",
    "source_bundle_root_or_target_sha256",
    "predecessor_authority",
    "acceptance_authority",
}
NEGATIVE_EVIDENCE_STATUSES = {UNKNOWN, "NOT_RUN", "BLOCKED"}
FACT_KEYS = {
    "CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE": {
        "deterministic_row_locator_frozen",
        "table_s2_hash_bound",
        "table_s3_hash_bound",
        "s2_s3_join_rule_frozen",
        "multi_asset_lineage_closed",
        "locator_lineage_commitment_algorithm",
        "locator_lineage_merkle_root_sha256",
        "canonical_record_count",
        "processed_pair_count",
        "raw_replay_role",
        "raw_replay_status",
        "independent_raw_reproduction_claimed",
    },
    "CANONICAL_REPORTED_ENDPOINT_SEMANTICS": {
        "author_published_processed_endpoint_is_primary",
        "endpoint_id_frozen",
        "endpoint_direction_frozen",
        "endpoint_scale_frozen",
        "contrast_and_transform_frozen",
        "paper_faithful_mapping_closed",
    },
    "BIOLOGICAL_GROUP_AUTHORITY": {
        "biological_group_id_frozen",
        "study_unit_is_gse200304",
        "gse200302_is_subseries_not_independent_study",
        "group_mapping_hash_bound",
    },
    "ROW_REPLICATE_OR_VALID_SE": {
        "replicate_or_valid_standard_error_present",
        "replicate_count_or_effective_n_frozen",
        "standard_error_semantics_frozen",
        "technical_uncertainty_not_substituted_for_biological_se",
    },
    "LICENSE_RIGHTS": {
        "rights_source_authority_closed",
        "qualification_use_allowed",
        "private_canonical_materialization_allowed",
        "redistribution_scope",
    },
    "OUTCOME_BLIND_SPLIT_LEAKAGE": {
        "a1_source_graph_frozen",
        "a1_group_graph_frozen",
        "a1_near_duplicate_graph_frozen",
        "split_salt_hash_bound",
        "outcome_blind_assignment",
        "leakage_audit_pass",
        "final_benchmark_membership_deferred_to_a2",
    },
    "PREFROZEN_POWER_PRECISION": {
        "analysis_unit",
        "bootstrap_unit",
        "evaluation_population",
        "evaluation_group_count",
        "target_metric",
        "alternative_spearman_rho",
        "two_sided_alpha",
        "power_method",
        "working_distribution_assumption",
        "estimated_design_power",
        "confidence_level",
        "confidence_interval_method",
        "planned_full_confidence_interval_width",
        "prefrozen_before_model_results",
    },
}
GATE_BLOCKERS = {
    *(f"{slot}_NOT_PASS" for slot in SLOT_IDS),
    "RAW_REPLAY_INDEPENDENT_REPRODUCTION_CLAIM_INVALID",
    "CANONICAL_RECORD_COUNT_NOT_6547",
    "POWER_LT_0_80",
    "FULL_CI_WIDTH_GT_0_30",
}


class AdjudicationError(RuntimeError):
    """Evidence, authority, or execution semantics failed."""


class BindingError(AdjudicationError):
    """A repository or implementation binding is not complete."""


class ScopeViolation(AdjudicationError):
    """A configured path is outside this aggregate-only protocol."""


class PublicationError(AdjudicationError):
    """The exclusive output cannot be created or accepted."""


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AdjudicationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                AdjudicationError(f"non-finite JSON constant in {label}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdjudicationError(f"invalid JSON: {label}") from exc
    if type(value) is not dict:
        raise AdjudicationError(f"JSON root is not an object: {label}")
    return value


def _expect_exact_keys(value: Any, expected: set[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise AdjudicationError(f"{label} keys differ from the closed schema")
    return value


def _expect(value: Any, expected: Any, *, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise AdjudicationError(f"{label} differs")


def _expect_bool(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise AdjudicationError(f"{label} must be a boolean")
    return value


def _expect_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise AdjudicationError(f"{label} must be an integer >= {minimum}")
    return value


def _expect_number(value: Any, *, label: str) -> float:
    if type(value) not in {int, float} or type(value) is bool or not math.isfinite(value):
        raise AdjudicationError(f"{label} must be finite")
    return float(value)


def config_core_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    projected = copy.deepcopy(dict(config))
    projected.pop("implementation_binding", None)
    repository = projected.get("repository_authority")
    if type(repository) is dict:
        for key in AUTHORITY_RUNTIME_DYNAMIC_FIELDS:
            repository[key] = "DYNAMIC_GIT_BINDING"
    descriptors = projected.get("evidence_descriptor_bindings")
    if type(descriptors) is dict:
        slots = descriptors.get("slots")
        projected["evidence_descriptor_bindings"] = {
            "binding_scheme": descriptors.get("binding_scheme"),
            "dynamic_scalar_suffixes": descriptors.get("dynamic_scalar_suffixes"),
            "all_descriptors_required_before_any_input_open": descriptors.get(
                "all_descriptors_required_before_any_input_open"
            ),
            "slots": [
                {"slot_id": slot.get("slot_id")} for slot in slots
            ] if type(slots) is list else slots,
        }
    return projected


def config_core_sha256(config: Mapping[str, Any]) -> str:
    return sha256(json_bytes(config_core_projection(config)))


def descriptor_set_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(config["evidence_descriptor_bindings"]))
    value.pop("descriptor_set_sha256", None)
    return value


def descriptor_set_sha256(config: Mapping[str, Any]) -> str:
    return sha256(json_bytes(descriptor_set_projection(config)))


def _descriptor_bound(slot: Mapping[str, Any]) -> bool:
    return (
        type(slot.get("absolute_path")) is str
        and slot.get("absolute_path") != UNKNOWN
        and HEX64.fullmatch(str(slot.get("sha256"))) is not None
        and type(slot.get("bytes")) is int
        and slot["bytes"] > 0
    )


def _descriptor_unbound(slot: Mapping[str, Any]) -> bool:
    return all(slot.get(key) == UNKNOWN for key in ("absolute_path", "sha256", "bytes"))


def _derived_descriptor_status(config: Mapping[str, Any]) -> str:
    slots = config["evidence_descriptor_bindings"]["slots"]
    count = sum(_descriptor_bound(slot) for slot in slots)
    if count == 0:
        return "UNBOUND"
    if count == len(slots):
        return "BOUND"
    return "PARTIALLY_BOUND"


def foundation_route_passes(route: Mapping[str, Any]) -> bool:
    count = route.get("audited_checkpoint_count")
    return (
        route.get("status") == "PASS"
        and route.get("checkpoint_specific_exposure_gate_applicable") is True
        and route.get("checkpoint_specific_exposure_may_be_waived") is False
        and type(count) is int
        and type(count) is not bool
        and count >= 1
    )


def validate_route_contract(config: Mapping[str, Any]) -> None:
    route = _expect_exact_keys(
        config["model_input_route_contract"],
        {
            "qualification_is_model_input_route_scoped",
            "selected_route",
            "route_selected_before_model_results",
            "outcome_or_model_result_used_for_route_selection",
            "route_switch_after_model_results_allowed",
            "route_fallback_after_failure_allowed",
            "same_dataset_duplicate_credit_across_routes_allowed",
            "scratch_route",
            "foundation_route",
            "forbidden_external_learned_inputs",
        },
        label="model-input route contract",
    )
    for key in (
        "qualification_is_model_input_route_scoped",
        "route_selected_before_model_results",
    ):
        _expect(route[key], True, label=f"route {key}")
    for key in (
        "outcome_or_model_result_used_for_route_selection",
        "route_switch_after_model_results_allowed",
        "route_fallback_after_failure_allowed",
        "same_dataset_duplicate_credit_across_routes_allowed",
    ):
        _expect(route[key], False, label=f"route {key}")
    _expect(route["selected_route"], SCRATCH_ROUTE, label="selected model-input route")
    _expect(
        route["forbidden_external_learned_inputs"],
        FORBIDDEN_EXTERNAL_LEARNED_INPUTS,
        label="forbidden external learned inputs",
    )

    scratch = _expect_exact_keys(route["scratch_route"], SCRATCH_ROUTE_KEYS, label="scratch route")
    expected = {
        "route_status": "FROZEN_AUTHORIZED_FOR_ADJUDICATION",
        "checkpoint_specific_exposure_gate_applicable": False,
        "checkpoint_specific_exposure_status": SCRATCH_EXPOSURE_STATUS,
        "checkpoint_specific_exposure_pass_claimed": False,
        "allowed_parameter_initialization": "RANDOM_INITIALIZATION_ONLY",
        "external_checkpoint_count": 0,
        "external_learned_input_count": 0,
    }
    for key, value in expected.items():
        _expect(scratch[key], value, label=f"scratch route {key}")
    for key in sorted(SCRATCH_ABSENCE_FIELDS):
        _expect(scratch[key], False, label=f"scratch route {key}")

    foundation = _expect_exact_keys(
        route["foundation_route"],
        {
            "status",
            "checkpoint_evidence_status",
            "checkpoint_specific_exposure_gate_applicable",
            "checkpoint_specific_exposure_may_be_waived",
            "minimum_audited_checkpoint_count_for_pass",
            "audited_checkpoint_count",
            "empty_checkpoint_set_can_pass",
        },
        label="foundation route",
    )
    _expect(
        foundation["status"],
        FOUNDATION_ROUTE_STATUS,
        label="foundation route status",
    )
    _expect(
        foundation["checkpoint_evidence_status"],
        FOUNDATION_CHECKPOINT_EVIDENCE_STATUS,
        label="foundation checkpoint evidence status",
    )
    _expect(
        foundation["checkpoint_specific_exposure_gate_applicable"],
        True,
        label="foundation exposure applicability",
    )
    _expect(
        foundation["checkpoint_specific_exposure_may_be_waived"],
        False,
        label="foundation exposure waiver",
    )
    _expect(
        foundation["minimum_audited_checkpoint_count_for_pass"],
        1,
        label="foundation minimum checkpoint count",
    )
    _expect(
        foundation["audited_checkpoint_count"],
        UNKNOWN,
        label="foundation audited checkpoint count",
    )
    _expect(foundation["empty_checkpoint_set_can_pass"], False, label="empty foundation set")
    if foundation_route_passes(foundation):
        raise AdjudicationError("unknown foundation route cannot be reported PASS")


def _dynamic_commit_group(value: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(value[key] for key in AUTHORITY_RUNTIME_DYNAMIC_FIELDS)


def _runtime_group_is_bound(value: Mapping[str, Any]) -> bool:
    runtime_i, runtime_b, script_sha, test_sha, base, v4_parent = (
        _dynamic_commit_group(value)
    )
    return (
        HEX40.fullmatch(str(runtime_i)) is not None
        and HEX40.fullmatch(str(runtime_b)) is not None
        and HEX64.fullmatch(str(script_sha)) is not None
        and HEX64.fullmatch(str(test_sha)) is not None
        and base == runtime_b
        and v4_parent == runtime_b
    )


def validate_static_config(config: Mapping[str, Any]) -> None:
    _expect_exact_keys(config, EXPECTED_CONFIG_TOP_KEYS, label="config")
    for key, expected in {
        "schema_version": "route_a_v3_gse200304_dec020_reported_endpoint_a1_activation.v4",
        "protocol_id": PROTOCOL_ID,
        "contract_id": CONTRACT_ID,
        "phase_id": "A1",
        "dataset_id": DATASET_ID,
        "decision_id": DECISION_ID,
    }.items():
        _expect(config[key], expected, label=f"config {key}")
    validate_route_contract(config)

    binding = _expect_exact_keys(
        config["implementation_binding"],
        EXPECTED_IMPLEMENTATION_BINDING_KEYS,
        label="implementation binding",
    )
    for key, expected in {
        "binding_scheme": "CONFIG_ONLY_POST_IMPLEMENTATION_BINDING_V1",
        "blocker_if_unbound": "IMPLEMENTATION_BINDING_UNKNOWN_NOT_ASSERTED",
        "implementation_script_path": SCRIPT_REPO_PATH,
        "implementation_test_path": TEST_REPO_PATH,
        "unknown_to_bound_scalar_paths": EXPECTED_I_TO_B_SCALAR_PATHS,
    }.items():
        _expect(binding[key], expected, label=f"implementation binding {key}")
    dynamic = (
        "implementation_commit",
        "implementation_script_sha256",
        "implementation_test_sha256",
    )
    if binding["status"] == UNKNOWN:
        if any(binding[key] != UNKNOWN for key in dynamic):
            raise BindingError("UNKNOWN implementation binding is partially bound")
    elif binding["status"] == "BOUND":
        if (
            HEX40.fullmatch(str(binding["implementation_commit"])) is None
            or HEX64.fullmatch(str(binding["implementation_script_sha256"])) is None
            or HEX64.fullmatch(str(binding["implementation_test_sha256"])) is None
        ):
            raise BindingError("BOUND implementation binding has an invalid field")
    else:
        raise BindingError("implementation binding status is outside the closed enum")

    repository = _expect_exact_keys(
        config["repository_authority"],
        {
            "production_repo_root",
            "branch",
            "pre_authority_base_commit",
            "authority_runtime_i1_commit",
            "authority_runtime_i1_config_bytes",
            "authority_runtime_i1_config_sha256",
            "authority_runtime_i1_script_sha256",
            "authority_runtime_i1_test_sha256",
            "authority_runtime_implementation_commit",
            "authority_runtime_binding_commit",
            "authority_runtime_implementation_script_sha256",
            "authority_runtime_implementation_test_sha256",
            "base_commit",
            "implementation_commit_expected_parent",
            "authority_runtime_and_v4_parent_grouped_unknown_fields",
            "authority_runtime_i1_must_be_direct_child_of_authority_commit",
            "authority_runtime_implementation_must_be_direct_child_of_runtime_i1",
            "authority_runtime_i1_commit_exact_changed_paths",
            "authority_runtime_binding_must_be_direct_child_of_runtime_implementation",
            "authority_runtime_implementation_commit_exact_changed_paths",
            "authority_runtime_binding_commit_exact_changed_paths",
            "v4_implementation_must_be_direct_child_of_authority_runtime_binding",
            "binding_commit_expected_parent",
            "implementation_commit_exact_changed_paths",
            "binding_commit_exact_changed_paths",
            "lifecycle",
            "predecessor_v3_preservation",
        },
        label="repository authority",
    )
    for key, expected in {
        "production_repo_root": os.fspath(PRODUCTION_REPO_ROOT),
        "branch": "routea-v3-a1-20260810",
        "pre_authority_base_commit": PRE_AUTHORITY_BASE_COMMIT,
        "authority_runtime_i1_commit": AUTHORITY_RUNTIME_I1_COMMIT,
        "authority_runtime_i1_config_bytes": AUTHORITY_RUNTIME_I1_CONFIG_BYTES,
        "authority_runtime_i1_config_sha256": AUTHORITY_RUNTIME_I1_CONFIG_SHA256,
        "authority_runtime_i1_script_sha256": AUTHORITY_RUNTIME_I1_SCRIPT_SHA256,
        "authority_runtime_i1_test_sha256": AUTHORITY_RUNTIME_I1_TEST_SHA256,
        "authority_runtime_and_v4_parent_grouped_unknown_fields": [
            "repository_authority.authority_runtime_implementation_commit",
            "repository_authority.authority_runtime_binding_commit",
            "repository_authority.authority_runtime_implementation_script_sha256",
            "repository_authority.authority_runtime_implementation_test_sha256",
            "repository_authority.base_commit",
            "repository_authority.implementation_commit_expected_parent",
        ],
        "authority_runtime_i1_must_be_direct_child_of_authority_commit": True,
        "authority_runtime_implementation_must_be_direct_child_of_runtime_i1": True,
        "authority_runtime_i1_commit_exact_changed_paths": EXPECTED_AUTHORITY_RUNTIME_I1_PATHS,
        "authority_runtime_binding_must_be_direct_child_of_runtime_implementation": True,
        "authority_runtime_implementation_commit_exact_changed_paths": EXPECTED_AUTHORITY_RUNTIME_I_PATHS,
        "authority_runtime_binding_commit_exact_changed_paths": EXPECTED_AUTHORITY_RUNTIME_B_PATHS,
        "v4_implementation_must_be_direct_child_of_authority_runtime_binding": True,
        "binding_commit_expected_parent": "IMPLEMENTATION_COMMIT_FROM_BINDING",
        "implementation_commit_exact_changed_paths": EXPECTED_I_PATHS,
        "binding_commit_exact_changed_paths": EXPECTED_B_PATHS,
        "lifecycle": "AUTHORITY_A_THEN_AUTHORITY_RUNTIME_I1_I2_CONFIG_ONLY_B2_THEN_V4_I_CONFIG_ONLY_B",
    }.items():
        _expect(repository[key], expected, label=f"repository {key}")
    grouped = _dynamic_commit_group(repository)
    if grouped != (UNKNOWN,) * len(AUTHORITY_RUNTIME_DYNAMIC_FIELDS) and not (
        _runtime_group_is_bound(repository)
    ):
        raise BindingError("authority-runtime/V4-parent group is partial or invalid")

    predecessor = repository["predecessor_v3_preservation"]
    if (
        predecessor.get("status") != "FROZEN_UNCHANGED"
        or predecessor.get("expected_external_status")
        != "BLOCKED_DEC019_REPORTED_ENDPOINT_A1_EVIDENCE_INCOMPLETE"
        or predecessor.get("checkpoint_negative_slot_retained_in_v3") is not True
        or predecessor.get("checkpoint_negative_slot_relabelled_or_consumed_by_v4")
        is not False
    ):
        raise AdjudicationError("predecessor V3 preservation semantics differ")
    blobs = predecessor.get("frozen_blobs")
    if type(blobs) is not list or {
        item.get("path"): item.get("sha256") for item in blobs if type(item) is dict
    } != PREDECESSOR_V3_FROZEN_BLOBS:
        raise AdjudicationError("predecessor V3 frozen blobs differ")

    core = _expect_exact_keys(
        config["core_authority"],
        {
            "status",
            "amendment_mode",
            "pre_authority_base_commit",
            "authority_commit",
            "authority_commit_parent_must_equal_pre_authority_base",
            "authority_commit_exact_changed_paths",
            "bound_files",
        },
        label="DEC020 core authority",
    )
    for key, expected in {
        "status": "BOUND",
        "amendment_mode": "APPEND_ONLY_DEC020_COMPANION_DEC019_BYTES_UNCHANGED",
        "pre_authority_base_commit": PRE_AUTHORITY_BASE_COMMIT,
        "authority_commit": AUTHORITY_COMMIT,
        "authority_commit_parent_must_equal_pre_authority_base": True,
        "authority_commit_exact_changed_paths": EXPECTED_AUTHORITY_A_PATHS,
    }.items():
        _expect(core[key], expected, label=f"DEC020 core authority {key}")
    bound_files = core.get("bound_files")
    expected_bound_files = {
        **ROOT_CONTRACT_FROZEN_BLOB,
        **AUTHORITY_A_EXACT_CHANGED_BLOBS,
    }
    if (
        type(bound_files) is not list
        or len(bound_files) != len(expected_bound_files)
        or any(
            type(item) is not dict
            or set(item) != {"path", "sha256"}
            or HEX64.fullmatch(str(item["sha256"])) is None
            for item in bound_files
        )
        or {item["path"]: item["sha256"] for item in bound_files}
        != expected_bound_files
    ):
        raise BindingError("DEC020 bound authority files differ")

    policy = config["policy_boundary"]
    for key, expected in {
        "primary_measurement_route": "AUTHOR_PUBLISHED_PROCESSED_ENDPOINT",
        "study_counting_unit": "GSE200304_SUPERSERIES_ONE_STUDY",
        "maximum_study_contribution_per_dataset": 1,
        "ordinary_gate_contribution_on_success": 1,
        "a1_gate_contribution_on_success": 1,
        "true_a2_gate_contribution_on_success": 0,
        "canonical_record_count_on_success": CANONICAL_RECORD_COUNT,
        "canonical_materialization_qualification_eligible_on_success": True,
        "private_payload_access_authorized": False,
        "canonical_materialization_execution_authorized": False,
        "raw_or_row_level_payload_consumption_allowed": False,
        "training_allowed_by_this_adjudicator": False,
        "model_selection_allowed_by_this_adjudicator": False,
        "gpu_allowed_by_this_adjudicator": False,
        "next_phase_allowed_by_this_adjudicator": False,
        "scientific_claim_established_by_this_adjudicator": False,
        "minimum_power": 0.8,
        "maximum_full_confidence_interval_width": 0.3,
    }.items():
        _expect(policy.get(key), expected, label=f"policy {key}")
    state = config["current_external_state"]
    for key in (
        "qualified",
        "canonical_materialization_qualification_eligible",
        "private_payload_access_authorized",
        "canonical_materialization_execution_authorized",
        "training_allowed",
        "model_selection_allowed",
        "gpu_allowed",
        "next_phase_authorized",
    ):
        _expect(state.get(key), False, label=f"current state {key}")
    for key in (
        "ordinary_study_contribution",
        "a1_study_contribution",
        "true_a2_study_contribution",
        "canonical_record_count",
    ):
        _expect(state.get(key), 0, label=f"current state {key}")
    _expect(state.get("scientific_claim_status"), "NOT_ESTABLISHED", label="current claim")

    evidence = config["evidence_contract"]
    _expect(evidence.get("excluded_predecessor_slot"), "CHECKPOINT_SPECIFIC_EXPOSURE", label="excluded slot")
    _expect(
        evidence.get("excluded_predecessor_negative_record_is_not_opened_or_relabelled"),
        True,
        label="excluded predecessor negative record",
    )
    _expect(evidence.get("evidence_schema_version"), EVIDENCE_SCHEMA_VERSION, label="evidence schema")
    _expect(evidence.get("evidence_record_type"), EVIDENCE_RECORD_TYPE, label="evidence record type")
    _expect(evidence.get("evidence_decision_id"), EVIDENCE_DECISION_ID, label="evidence decision")
    _expect(evidence.get("reused_gate_acceptance_authority"), EVIDENCE_ACCEPTANCE_AUTHORITY, label="evidence acceptance authority")
    slots = evidence.get("slots")
    if type(slots) is not list or tuple(slot.get("slot_id") for slot in slots) != SLOT_IDS:
        raise AdjudicationError("V4 evidence slots are not the exact seven-slot order")
    if any(slot.get("slot_id") == "CHECKPOINT_SPECIFIC_EXPOSURE" for slot in slots):
        raise AdjudicationError("V4 must not consume the predecessor checkpoint slot")

    descriptors = config["evidence_descriptor_bindings"]
    descriptor_slots = descriptors.get("slots")
    if type(descriptor_slots) is not list or tuple(
        slot.get("slot_id") for slot in descriptor_slots
    ) != SLOT_IDS:
        raise AdjudicationError("descriptor slots are not the exact seven-slot order")
    for slot in descriptor_slots:
        if not (_descriptor_bound(slot) or _descriptor_unbound(slot)):
            raise BindingError(f"descriptor is partially bound: {slot.get('slot_id')}")
    _expect(descriptors.get("status"), _derived_descriptor_status(config), label="descriptor status")
    _expect(
        descriptors.get("descriptor_set_sha256"),
        descriptor_set_sha256(config),
        label="descriptor set SHA",
    )

    output = config["output_contract"]
    for key, expected in {
        "output_id": OUTPUT_ID,
        "blocked_status": BLOCKED_STATUS,
        "success_status": SUCCESS_STATUS,
        "aggregate_only": True,
        "member_names_excluding_commit_marker": list(OUTPUT_NAMES_EXCLUDING_MARKER),
        "terminal_commit_marker": COMMIT_MARKER,
        "publication_mode": PUBLICATION_MODE,
        "commit_marker_written_last": True,
        "overwrite_allowed": False,
        "predecessor_v3_output_may_be_overwritten": False,
    }.items():
        _expect(output.get(key), expected, label=f"output {key}")

    _expect(binding["config_core_sha256"], FROZEN_CONFIG_CORE_SHA256, label="stored config core")
    _expect(config_core_sha256(config), FROZEN_CONFIG_CORE_SHA256, label="computed config core")


def validate_implementation_binding(config: Mapping[str, Any]) -> None:
    validate_static_config(config)
    binding = config["implementation_binding"]
    if binding["status"] != "BOUND":
        raise BindingError("implementation binding is UNKNOWN; no evidence or output may be touched")


def _semantic_diff_paths(left: Any, right: Any, prefix: str = "") -> set[str]:
    if type(left) is not type(right):
        return {prefix}
    if type(left) is dict:
        paths: set[str] = set()
        for key in set(left) | set(right):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.add(child)
            else:
                paths.update(_semantic_diff_paths(left[key], right[key], child))
        return paths
    if type(left) is list:
        if len(left) != len(right):
            return {prefix}
        paths: set[str] = set()
        for index, (lvalue, rvalue) in enumerate(zip(left, right)):
            paths.update(_semantic_diff_paths(lvalue, rvalue, f"{prefix}[{index}]"))
        return paths
    return set() if left == right else {prefix}


def validate_i_to_b(i_config: Mapping[str, Any], b_config: Mapping[str, Any]) -> None:
    validate_static_config(i_config)
    validate_static_config(b_config)
    i_binding = i_config["implementation_binding"]
    b_binding = b_config["implementation_binding"]
    if i_binding["status"] != UNKNOWN or any(
        i_binding[key] != UNKNOWN
        for key in (
            "implementation_commit",
            "implementation_script_sha256",
            "implementation_test_sha256",
        )
    ):
        raise BindingError("parent I implementation binding is not exact UNKNOWN")
    if b_binding["status"] != "BOUND":
        raise BindingError("current B implementation binding is not BOUND")
    if _semantic_diff_paths(i_config, b_config) != set(EXPECTED_I_TO_B_SCALAR_PATHS):
        raise BindingError("I-to-B transition is not the exact four binding scalars")
    if config_core_sha256(i_config) != config_core_sha256(b_config):
        raise BindingError("I-to-B transition changed the science core")


def _validate_authority_runtime_i_to_b(
    i_config: Mapping[str, Any],
    b_config: Mapping[str, Any],
    *,
    runtime_implementation_commit: str,
    runtime_script_sha256: str,
    runtime_test_sha256: str,
) -> None:
    expected_binding_keys = {
        "binding_scheme",
        "status",
        "implementation_commit",
        "implementation_script_path",
        "implementation_script_sha256",
        "implementation_test_path",
        "implementation_test_sha256",
        "compiled_core_sha256",
        "unknown_to_bound_scalar_paths",
        "activation_rule",
    }
    for label, runtime_config in (("I2", i_config), ("B2", b_config)):
        runtime_binding = runtime_config.get("implementation_binding")
        if type(runtime_binding) is not dict or set(runtime_binding) != expected_binding_keys:
            raise BindingError(f"authority-runtime {label} binding key closure differs")
        if runtime_binding.get("compiled_core_sha256") != (
            AUTHORITY_RUNTIME_CONFIG_CORE_SHA256
        ):
            raise BindingError(f"authority-runtime {label} compiled core binding differs")
        runtime_core = copy.deepcopy(dict(runtime_config))
        runtime_core.pop("implementation_binding", None)
        runtime_core_payload = json.dumps(
            runtime_core,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if sha256(runtime_core_payload) != AUTHORITY_RUNTIME_CONFIG_CORE_SHA256:
            raise BindingError(f"authority-runtime {label} config core differs")
    if _semantic_diff_paths(i_config, b_config) != (
        EXPECTED_AUTHORITY_RUNTIME_I_TO_B_SCALAR_PATHS
    ):
        raise BindingError(
            "authority-runtime I-to-B transition is not the exact four binding scalars"
        )
    i_binding = i_config.get("implementation_binding")
    b_binding = b_config.get("implementation_binding")
    if type(i_binding) is not dict or type(b_binding) is not dict:
        raise BindingError("authority-runtime implementation binding is absent")
    if i_binding.get("status") != UNKNOWN or any(
        i_binding.get(key) != UNKNOWN
        for key in (
            "implementation_commit",
            "implementation_script_sha256",
            "implementation_test_sha256",
        )
    ):
        raise BindingError("authority-runtime I binding is not exact UNKNOWN")
    expected_b = {
        "status": "BOUND",
        "implementation_commit": runtime_implementation_commit,
        "implementation_script_sha256": runtime_script_sha256,
        "implementation_test_sha256": runtime_test_sha256,
    }
    for key, expected in expected_b.items():
        if b_binding.get(key) != expected:
            raise BindingError(f"authority-runtime B {key} differs")
    if b_binding.get("implementation_script_path") != (
        AUTHORITY_RUNTIME_SCRIPT_REPO_PATH
    ) or b_binding.get("implementation_test_path") != AUTHORITY_RUNTIME_TEST_REPO_PATH:
        raise BindingError("authority-runtime B implementation paths differ")


def _descriptor_by_slot(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        slot["slot_id"]: slot
        for slot in config["evidence_descriptor_bindings"]["slots"]
    }


def _all_descriptors_bound(config: Mapping[str, Any]) -> bool:
    return all(
        _descriptor_bound(slot)
        for slot in config["evidence_descriptor_bindings"]["slots"]
    )


def _reject_evidence_path(path: Path, config: Mapping[str, Any], *, label: str) -> None:
    lexical = os.fspath(path)
    if not path.is_absolute():
        raise ScopeViolation(f"{label} path is not absolute")
    folded = lexical.casefold()
    for token in config["evidence_contract"]["forbidden_path_tokens"]:
        if token.casefold() in folded:
            raise ScopeViolation(f"{label} path contains forbidden token: {token}")


def _read_verified_evidence(slot: Mapping[str, Any], config: Mapping[str, Any]) -> bytes:
    path = Path(slot["absolute_path"])
    _reject_evidence_path(path, config, label=f"evidence {slot['slot_id']}")
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ScopeViolation(f"evidence is not a regular non-symlink file: {slot['slot_id']}")
    if before.st_nlink != 1:
        raise ScopeViolation(f"evidence link count differs: {slot['slot_id']}")
    if before.st_size != slot["bytes"]:
        raise AdjudicationError(f"evidence byte count differs: {slot['slot_id']}")
    payload = path.read_bytes()
    after = path.lstat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise AdjudicationError(f"evidence changed during read: {slot['slot_id']}")
    if sha256(payload) != slot["sha256"]:
        raise AdjudicationError(f"evidence SHA differs: {slot['slot_id']}")
    return payload


def _assert_no_forbidden_output_keys(value: Any, forbidden: set[str], *, label: str) -> None:
    if type(value) is dict:
        for key, child in value.items():
            if key.casefold() in forbidden:
                raise AdjudicationError(f"{label} contains forbidden key: {key}")
            _assert_no_forbidden_output_keys(child, forbidden, label=label)
    elif type(value) is list:
        for child in value:
            _assert_no_forbidden_output_keys(child, forbidden, label=label)


def _validate_provenance(
    record: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    slot_id: str,
) -> None:
    required = set(PROVENANCE_KEYS)
    if slot_id == "BIOLOGICAL_GROUP_AUTHORITY" and record["status"] == "PASS":
        required.add(GROUP_MAPPING_COMMITMENT_KEY)
    if slot_id == "OUTCOME_BLIND_SPLIT_LEAKAGE" and record["status"] == "PASS":
        required.add(SPLIT_ASSIGNMENT_COMMITMENT_KEY)
    provenance = _expect_exact_keys(
        record["provenance"], required, label=f"{slot_id} provenance"
    )
    if HEX40.fullmatch(str(provenance["producer_commit"])) is None:
        raise AdjudicationError(f"{slot_id} producer commit is not bound")
    if HEX64.fullmatch(str(provenance["producer_script_sha256"])) is None:
        raise AdjudicationError(f"{slot_id} producer script SHA is not bound")
    if not isinstance(provenance["producer_protocol_id"], str) or not provenance[
        "producer_protocol_id"
    ]:
        raise AdjudicationError(f"{slot_id} producer protocol is absent")
    for key in (GROUP_MAPPING_COMMITMENT_KEY, SPLIT_ASSIGNMENT_COMMITMENT_KEY):
        if key in required and HEX64.fullmatch(str(provenance[key])) is None:
            raise AdjudicationError(f"{slot_id} commitment is not bound")
    predecessor = config["evidence_contract"]["required_predecessor_authority"]
    _expect(
        provenance["source_bundle_id"],
        predecessor["bundle_id"],
        label=f"{slot_id} source bundle",
    )
    _expect(
        provenance["source_bundle_root_or_target_sha256"],
        predecessor["terminal_marker_final_output_target_sha256"],
        label=f"{slot_id} source target",
    )
    _expect(
        provenance["predecessor_authority"],
        predecessor,
        label=f"{slot_id} predecessor authority",
    )
    _expect(
        provenance["acceptance_authority"],
        EVIDENCE_ACCEPTANCE_AUTHORITY,
        label=f"{slot_id} reused DEC019 acceptance authority",
    )


def _validate_fact_types(slot_id: str, facts: Mapping[str, Any]) -> None:
    remaining = set(facts)
    if slot_id == "CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE":
        _expect_int(facts["canonical_record_count"], label="canonical record count")
        _expect_int(facts["processed_pair_count"], label="processed pair count")
        _expect(
            facts["locator_lineage_commitment_algorithm"],
            LOCATOR_LINEAGE_COMMITMENT_ALGORITHM,
            label="locator lineage commitment algorithm",
        )
        if HEX64.fullmatch(str(facts["locator_lineage_merkle_root_sha256"])) is None:
            raise AdjudicationError("locator lineage Merkle root is not bound")
        _expect(
            facts["raw_replay_role"],
            "REPRODUCIBILITY_AUXILIARY_NOT_QUALIFICATION_PREREQUISITE",
            label="raw replay role",
        )
        if facts["raw_replay_status"] not in {
            "NOT_RUN",
            "PASS_INDEPENDENT_REPRODUCTION",
        }:
            raise AdjudicationError("raw replay status is outside the closed enum")
        remaining -= {
            "canonical_record_count",
            "processed_pair_count",
            "locator_lineage_commitment_algorithm",
            "locator_lineage_merkle_root_sha256",
            "raw_replay_role",
            "raw_replay_status",
        }
    elif slot_id == "LICENSE_RIGHTS":
        if facts["redistribution_scope"] not in {
            "PRIVATE_CANONICAL_ONLY",
            "PUBLIC_REDISTRIBUTION_ALLOWED",
        }:
            raise AdjudicationError("redistribution scope is outside the closed enum")
        remaining.remove("redistribution_scope")
    elif slot_id == "PREFROZEN_POWER_PRECISION":
        _expect_int(
            facts["evaluation_group_count"],
            label="evaluation group count",
            minimum=4,
        )
        for key in (
            "alternative_spearman_rho",
            "two_sided_alpha",
            "estimated_design_power",
            "confidence_level",
            "planned_full_confidence_interval_width",
        ):
            _expect_number(facts[key], label=f"power {key}")
        for key, expected in {
            "analysis_unit": POWER_ANALYSIS_UNIT,
            "bootstrap_unit": POWER_ANALYSIS_UNIT,
            "evaluation_population": POWER_EVALUATION_POPULATION,
            "target_metric": POWER_TARGET_METRIC,
            "power_method": POWER_METHOD,
            "working_distribution_assumption": POWER_WORKING_DISTRIBUTION_ASSUMPTION,
            "confidence_interval_method": POWER_CI_METHOD,
        }.items():
            _expect(facts[key], expected, label=f"power {key}")
        remaining -= {
            "analysis_unit",
            "bootstrap_unit",
            "evaluation_population",
            "evaluation_group_count",
            "target_metric",
            "alternative_spearman_rho",
            "two_sided_alpha",
            "power_method",
            "working_distribution_assumption",
            "estimated_design_power",
            "confidence_level",
            "confidence_interval_method",
            "planned_full_confidence_interval_width",
        }
    for key in remaining:
        _expect_bool(facts[key], label=f"{slot_id} {key}")


def _validate_gate_record(
    payload: bytes,
    slot: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    slot_id = slot["slot_id"]
    record = strict_json(payload, label=f"gate evidence {slot_id}")
    _expect_exact_keys(record, COMMON_EVIDENCE_KEYS, label=f"gate evidence {slot_id}")
    for key, expected in {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "record_type": EVIDENCE_RECORD_TYPE,
        "contract_id": CONTRACT_ID,
        "decision_id": EVIDENCE_DECISION_ID,
        "dataset_id": DATASET_ID,
        "gate_id": slot_id,
        "accepted": True,
        "aggregate_only": True,
    }.items():
        _expect(record[key], expected, label=f"{slot_id} {key}")
    if record["status"] not in {"PASS", *NEGATIVE_EVIDENCE_STATUSES}:
        raise AdjudicationError(f"{slot_id} status is outside the closed enum")
    privacy = _expect_exact_keys(record["privacy"], PRIVACY_KEYS, label=f"{slot_id} privacy")
    if any(privacy.values()) or any(type(value) is not bool for value in privacy.values()):
        raise AdjudicationError(f"{slot_id} privacy declaration is not aggregate-only")
    _validate_provenance(record, config, slot_id=slot_id)
    required_facts = FACT_KEYS[slot_id]
    if record["status"] == "PASS":
        facts = _expect_exact_keys(record["facts"], required_facts, label=f"{slot_id} facts")
        _validate_fact_types(slot_id, facts)
        _expect(record["unknown_fields"], [], label=f"{slot_id} PASS unknown fields")
        _expect(record["reason_codes"], [], label=f"{slot_id} PASS reason codes")
    else:
        if record["facts"] is not None:
            raise AdjudicationError(f"{slot_id} negative record requires facts=null")
        _expect(
            record["unknown_fields"],
            sorted(required_facts),
            label=f"{slot_id} negative unknown fields",
        )
        reasons = record["reason_codes"]
        if (
            type(reasons) is not list
            or not reasons
            or reasons != sorted(set(reasons))
            or any(type(reason) is not str or REASON_CODE.fullmatch(reason) is None for reason in reasons)
        ):
            raise AdjudicationError(f"{slot_id} negative reason codes differ")
    forbidden = {key.casefold() for key in config["output_contract"]["forbidden_output_keys"]}
    _assert_no_forbidden_output_keys(record, forbidden, label=f"gate evidence {slot_id}")
    return record


def _slot_passes(slot_id: str, facts: Mapping[str, Any]) -> bool:
    if slot_id == "CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE":
        return (
            all(
                facts[key] is True
                for key in (
                    "deterministic_row_locator_frozen",
                    "table_s2_hash_bound",
                    "table_s3_hash_bound",
                    "s2_s3_join_rule_frozen",
                    "multi_asset_lineage_closed",
                )
            )
            and facts["canonical_record_count"] == CANONICAL_RECORD_COUNT
            and facts["processed_pair_count"] >= CANONICAL_RECORD_COUNT
            and facts["locator_lineage_commitment_algorithm"]
            == LOCATOR_LINEAGE_COMMITMENT_ALGORITHM
        )
    if slot_id in {
        "CANONICAL_REPORTED_ENDPOINT_SEMANTICS",
        "BIOLOGICAL_GROUP_AUTHORITY",
        "ROW_REPLICATE_OR_VALID_SE",
        "OUTCOME_BLIND_SPLIT_LEAKAGE",
    }:
        return all(value is True for value in facts.values())
    if slot_id == "LICENSE_RIGHTS":
        return all(
            facts[key] is True
            for key in (
                "rights_source_authority_closed",
                "qualification_use_allowed",
                "private_canonical_materialization_allowed",
            )
        )
    if slot_id == "PREFROZEN_POWER_PRECISION":
        return (
            facts["evaluation_group_count"] == BIOLOGICAL_GROUP_COUNT
            and facts["prefrozen_before_model_results"] is True
            and facts["analysis_unit"] == POWER_ANALYSIS_UNIT
            and facts["bootstrap_unit"] == POWER_ANALYSIS_UNIT
        )
    raise AdjudicationError(f"unknown V4 slot: {slot_id}")


def _evaluate(records: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]) -> tuple[list[str], bool]:
    blockers: list[str] = []
    for slot in config["evidence_contract"]["slots"]:
        record = records[slot["slot_id"]]
        if record["status"] != "PASS" or not _slot_passes(slot["slot_id"], record["facts"]):
            blockers.append(slot["blocker_if_not_pass"])
    lineage = records["CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE"]
    if lineage["status"] == "PASS":
        facts = lineage["facts"]
        if facts["canonical_record_count"] != CANONICAL_RECORD_COUNT:
            blockers.append("CANONICAL_RECORD_COUNT_NOT_6547")
        raw_status = facts["raw_replay_status"]
        raw_claim = facts["independent_raw_reproduction_claimed"]
        if (raw_status == "NOT_RUN" and raw_claim is not False) or (
            raw_status == "PASS_INDEPENDENT_REPRODUCTION" and raw_claim is not True
        ):
            blockers.append("RAW_REPLAY_INDEPENDENT_REPRODUCTION_CLAIM_INVALID")
    power = records["PREFROZEN_POWER_PRECISION"]
    if power["status"] == "PASS":
        if float(power["facts"]["estimated_design_power"]) < 0.8:
            blockers.append("POWER_LT_0_80")
        if float(power["facts"]["planned_full_confidence_interval_width"]) > 0.3:
            blockers.append("FULL_CI_WIDTH_GT_0_30")
    blockers = sorted(set(blockers))
    if not set(blockers).issubset(GATE_BLOCKERS):
        raise AdjudicationError("an unregistered blocker was produced")
    independent = (
        lineage["status"] == "PASS"
        and lineage["facts"]["raw_replay_status"] == "PASS_INDEPENDENT_REPRODUCTION"
        and lineage["facts"]["independent_raw_reproduction_claimed"] is True
    )
    return blockers, independent


def _synthetic_authority_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    repository = config["repository_authority"]
    return {
        "mode": "SYNTHETIC_NON_PRODUCTION",
        "lifecycle_state": "BOUND_CONFIG_SYNTHETIC_EXECUTION",
        "pre_authority_base_commit": PRE_AUTHORITY_BASE_COMMIT,
        "authority_commit": config["core_authority"]["authority_commit"],
        "authority_runtime_i1_commit": repository["authority_runtime_i1_commit"],
        "authority_runtime_implementation_commit": repository[
            "authority_runtime_implementation_commit"
        ],
        "authority_runtime_binding_commit": repository["authority_runtime_binding_commit"],
        "implementation_commit": config["implementation_binding"]["implementation_commit"],
        "binding_commit": UNKNOWN,
        "current_head": UNKNOWN,
        "config_core_sha256": FROZEN_CONFIG_CORE_SHA256,
        "evidence_descriptor_set_sha256": config["evidence_descriptor_bindings"]["descriptor_set_sha256"],
    }


def _report(
    config: Mapping[str, Any],
    blockers: Sequence[str],
    independent_reproduction: bool,
    authority_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    qualified = not blockers
    return {
        "record_type": "ROUTE_A_V3_GSE200304_DEC020_SCRATCH_ROUTE_A1_ADJUDICATION_REPORT_V4",
        "contract_id": CONTRACT_ID,
        "decision_id": DECISION_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "status": SUCCESS_STATUS if qualified else BLOCKED_STATUS,
        "qualified": qualified,
        "qualification_scope": "GSE200304_SCRATCH_ONLY_MODEL_INPUT_ROUTE",
        "data_role": (
            "A1_ORDINARY_REPORTED_ENDPOINT_ROUTE_SCOPED_QUALIFIED"
            if qualified
            else "A1_ORDINARY_REPORTED_ENDPOINT_CANDIDATE_NOT_QUALIFIED"
        ),
        "primary_measurement_route": "AUTHOR_PUBLISHED_PROCESSED_ENDPOINT",
        "model_input_route": SCRATCH_ROUTE,
        "route_conditional_exposure_status": SCRATCH_EXPOSURE_STATUS,
        "checkpoint_specific_exposure_pass_claimed": False,
        "foundation_route_status": FOUNDATION_ROUTE_STATUS,
        "foundation_checkpoint_evidence_status": FOUNDATION_CHECKPOINT_EVIDENCE_STATUS,
        "raw_replay_role": "REPRODUCIBILITY_AUXILIARY_NOT_QUALIFICATION_PREREQUISITE",
        "ordinary_study_contribution": 1 if qualified else 0,
        "a1_study_contribution": 1 if qualified else 0,
        "true_a2_study_contribution": 0,
        "canonical_record_count": CANONICAL_RECORD_COUNT if qualified else 0,
        "canonical_materialization_qualification_eligible": qualified,
        "private_payload_access_authorized": False,
        "canonical_materialization_execution_authorized": False,
        "independent_raw_reproduction_established": independent_reproduction,
        "row_level_payload_read_count": 0,
        "private_payload_read_count": 0,
        "sealed_payload_read_count": 0,
        "training_allowed": False,
        "model_selection_allowed": False,
        "gpu_allowed": False,
        "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "aggregate_only": True,
        "blockers": list(blockers),
        "config_core_sha256": FROZEN_CONFIG_CORE_SHA256,
        "evidence_descriptor_set_sha256": config["evidence_descriptor_bindings"]["descriptor_set_sha256"],
        "authority_provenance": dict(authority_provenance),
    }


def _input_audit(
    config: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]] | None,
    authority_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    descriptors = _descriptor_by_slot(config)
    slots = []
    for slot_id in SLOT_IDS:
        record = records.get(slot_id) if records is not None else None
        slots.append(
            {
                "slot_id": slot_id,
                "descriptor_bound": _descriptor_bound(descriptors[slot_id]),
                "input_opened": record is not None,
                "hash_verified": record is not None,
                "gate_status": record["status"] if record is not None else UNKNOWN,
            }
        )
    return {
        "record_type": "ROUTE_A_V3_GSE200304_DEC020_SCRATCH_AGGREGATE_INPUT_AUDIT_V1",
        "contract_id": CONTRACT_ID,
        "decision_id": DECISION_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "mode": (
            "EXACT_SEVEN_REUSED_DEC019_PASS_AGGREGATES_VERIFIED"
            if records is not None
            else "NO_INPUT_READ_DESCRIPTOR_SET_INCOMPLETE"
        ),
        "opened_input_count": len(records) if records is not None else 0,
        "excluded_checkpoint_record_open_count": 0,
        "row_level_payload_read_count": 0,
        "private_payload_read_count": 0,
        "sealed_payload_read_count": 0,
        "model_execution_count": 0,
        "training_run_count": 0,
        "all_inputs_aggregate_only": True,
        "slots": slots,
        "evidence_descriptor_set_sha256": config["evidence_descriptor_bindings"]["descriptor_set_sha256"],
        "authority_provenance": dict(authority_provenance),
    }


def recompute_adjudication_outputs(
    config: Mapping[str, Any],
    authority_provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not _all_descriptors_bound(config):
        blockers = sorted(
            slot["blocker_if_unbound"] for slot in config["evidence_contract"]["slots"]
        )
        return (
            _report(config, blockers, False, authority_provenance),
            _input_audit(config, None, authority_provenance),
        )
    descriptors = _descriptor_by_slot(config)
    records: dict[str, dict[str, Any]] = {}
    for slot in config["evidence_contract"]["slots"]:
        slot_id = slot["slot_id"]
        descriptor = descriptors[slot_id]
        path = Path(descriptor["absolute_path"])
        _reject_evidence_path(path, config, label=f"evidence {slot_id}")
        if path.name != slot["allowed_basename"]:
            raise ScopeViolation(f"evidence basename differs: {slot_id}")
        records[slot_id] = _validate_gate_record(
            _read_verified_evidence(descriptor, config), slot, config
        )
    blockers, independent = _evaluate(records, config)
    return (
        _report(config, blockers, independent, authority_provenance),
        _input_audit(config, records, authority_provenance),
    )


def _build_bundle(
    config: Mapping[str, Any],
    report: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, bytes]:
    payloads = {
        "ADJUDICATION_REPORT.json": json_bytes(report),
        "INPUT_EVIDENCE_AUDIT.json": json_bytes(audit),
    }
    sums = "".join(
        f"{sha256(payloads[name])}  {name}\n" for name in OUTPUT_JSON_NAMES
    ).encode("ascii")
    payloads["SHA256SUMS"] = sums
    member_identity = {
        name: {"bytes": len(payload), "sha256": sha256(payload)}
        for name, payload in sorted(payloads.items())
    }
    marker = {
        "schema_version": "route_a_v3_atomic_publication_commit.v1",
        "record_type": "ROUTE_A_V3_GSE200304_DEC020_SCRATCH_A1_PUBLICATION_COMMIT_V1",
        "contract_id": CONTRACT_ID,
        "decision_id": DECISION_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "output_id": OUTPUT_ID,
        "scientific_status": report["status"],
        "publication_mode": PUBLICATION_MODE,
        "sha256sums_sha256": sha256(sums),
        "bundle_member_names_excluding_commit_marker": list(OUTPUT_NAMES_EXCLUDING_MARKER),
        "bundle_file_count_excluding_commit_marker": len(OUTPUT_NAMES_EXCLUDING_MARKER),
        "final_output_target_sha256": sha256(json_bytes(member_identity)),
        "committed": True,
        "commit_marker_written_last": True,
        "aggregate_acceptance_requires_exact_marker": True,
    }
    payloads[COMMIT_MARKER] = json_bytes(marker)
    forbidden = {key.casefold() for key in config["output_contract"]["forbidden_output_keys"]}
    _assert_no_forbidden_output_keys(report, forbidden, label="adjudication report")
    _assert_no_forbidden_output_keys(audit, forbidden, label="input evidence audit")
    return payloads


def _publish_bundle(output: Path, payloads: Mapping[str, bytes]) -> str:
    expected_names = {*OUTPUT_NAMES_EXCLUDING_MARKER, COMMIT_MARKER}
    if output.exists():
        if not output.is_dir() or output.is_symlink():
            raise PublicationError("existing output is not a regular directory")
        actual_names = {entry.name for entry in output.iterdir()}
        if actual_names != expected_names:
            raise PublicationError("existing output is partial or has extra members")
        for name in expected_names:
            path = output / name
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payloads[name]:
                raise PublicationError("existing output differs; overwrite is forbidden")
        return "IDEMPOTENT_EXISTING_EXACT"
    output.mkdir(mode=0o750, parents=False, exist_ok=False)
    try:
        for name in OUTPUT_NAMES_EXCLUDING_MARKER:
            with (output / name).open("xb") as handle:
                handle.write(payloads[name])
                handle.flush()
                os.fsync(handle.fileno())
        with (output / COMMIT_MARKER).open("xb") as handle:
            handle.write(payloads[COMMIT_MARKER])
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(output, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        # A partial exclusive directory is deliberate failure evidence and is
        # never treated as committed or overwritten by a retry.
        raise
    return "PUBLISHED_NEW"


def _inspect_report_semantics(report: Mapping[str, Any], audit: Mapping[str, Any]) -> None:
    if (
        report.get("record_type")
        != "ROUTE_A_V3_GSE200304_DEC020_SCRATCH_ROUTE_A1_ADJUDICATION_REPORT_V4"
        or audit.get("record_type")
        != "ROUTE_A_V3_GSE200304_DEC020_SCRATCH_AGGREGATE_INPUT_AUDIT_V1"
        or any(
            value.get("contract_id") != CONTRACT_ID
            or value.get("decision_id") != DECISION_ID
            or value.get("protocol_id") != PROTOCOL_ID
            or value.get("dataset_id") != DATASET_ID
            for value in (report, audit)
        )
    ):
        raise AdjudicationError("published report/audit identity differs")
    if report.get("model_input_route") != SCRATCH_ROUTE:
        raise AdjudicationError("published report route differs")
    for key in (
        "private_payload_access_authorized",
        "canonical_materialization_execution_authorized",
        "training_allowed",
        "model_selection_allowed",
        "gpu_allowed",
        "next_phase_authorized",
    ):
        if report.get(key) is not False:
            raise AdjudicationError(f"published report {key} is not false")
    for key in (
        "row_level_payload_read_count",
        "private_payload_read_count",
        "sealed_payload_read_count",
    ):
        if report.get(key) != 0:
            raise AdjudicationError(f"published report {key} is not zero")
    if report.get("foundation_route_status") != FOUNDATION_ROUTE_STATUS:
        raise AdjudicationError("published foundation route status differs")
    if report.get("scientific_claim_status") != "NOT_ESTABLISHED":
        raise AdjudicationError("published scientific claim status differs")
    if (
        report.get("foundation_checkpoint_evidence_status")
        != FOUNDATION_CHECKPOINT_EVIDENCE_STATUS
    ):
        raise AdjudicationError("published foundation checkpoint evidence is not UNKNOWN")
    if report.get("qualified") is True:
        expected = (1, 1, 0, CANONICAL_RECORD_COUNT, True)
        observed = (
            report.get("ordinary_study_contribution"),
            report.get("a1_study_contribution"),
            report.get("true_a2_study_contribution"),
            report.get("canonical_record_count"),
            report.get("canonical_materialization_qualification_eligible"),
        )
        if observed != expected or report.get("status") != SUCCESS_STATUS:
            raise AdjudicationError("published success semantics differ")
        slots = audit.get("slots")
        expected_slot_keys = {
            "slot_id",
            "descriptor_bound",
            "input_opened",
            "hash_verified",
            "gate_status",
        }
        if (
            report.get("blockers") != []
            or audit.get("mode")
            != "EXACT_SEVEN_REUSED_DEC019_PASS_AGGREGATES_VERIFIED"
            or audit.get("opened_input_count") != len(SLOT_IDS)
            or audit.get("all_inputs_aggregate_only") is not True
            or audit.get("model_execution_count") != 0
            or audit.get("training_run_count") != 0
            or type(slots) is not list
            or len(slots) != len(SLOT_IDS)
            or tuple(
                slot.get("slot_id") if type(slot) is dict else None for slot in slots
            )
            != SLOT_IDS
            or any(
                set(slot) != expected_slot_keys
                or slot["descriptor_bound"] is not True
                or slot["input_opened"] is not True
                or slot["hash_verified"] is not True
                or slot["gate_status"] != "PASS"
                for slot in slots
            )
        ):
            raise AdjudicationError("published qualified bundle is not exact-seven PASS")
    else:
        observed = (
            report.get("ordinary_study_contribution"),
            report.get("a1_study_contribution"),
            report.get("true_a2_study_contribution"),
            report.get("canonical_record_count"),
            report.get("canonical_materialization_qualification_eligible"),
        )
        if observed != (0, 0, 0, 0, False) or report.get("status") != BLOCKED_STATUS:
            raise AdjudicationError("published blocked semantics differ")
    if (
        audit.get("excluded_checkpoint_record_open_count") != 0
        or audit.get("row_level_payload_read_count") != 0
        or audit.get("private_payload_read_count") != 0
        or audit.get("sealed_payload_read_count") != 0
    ):
        raise AdjudicationError("published audit access boundary differs")
    descriptor_set = report.get("evidence_descriptor_set_sha256")
    provenance = report.get("authority_provenance")
    if (
        HEX64.fullmatch(str(descriptor_set)) is None
        or audit.get("evidence_descriptor_set_sha256") != descriptor_set
        or type(provenance) is not dict
        or provenance.get("evidence_descriptor_set_sha256") != descriptor_set
    ):
        raise AdjudicationError("published descriptor-set identity differs")


def _inspect_authority_provenance(
    report: Mapping[str, Any],
    audit: Mapping[str, Any],
    config: Mapping[str, Any] | None,
) -> tuple[str, bool]:
    provenance = report.get("authority_provenance")
    if type(provenance) is not dict or audit.get("authority_provenance") != provenance:
        raise PublicationError("report/audit authority provenance differs")
    mode = provenance.get("mode")
    lifecycle = provenance.get("lifecycle_state")
    if mode == "PRODUCTION" and lifecycle == "V4_B_BOUND_EXACT_HEAD":
        if config is None:
            raise PublicationError("production inspection requires the bound V4 config")
        validate_implementation_binding(config)
        if config["implementation_binding"]["status"] != "BOUND":
            raise PublicationError("production inspection requires BOUND V4 config")
        validated = validate_production_authority(config)
        if dict(provenance) != validated:
            raise PublicationError("production authority provenance is incomplete")
        return "COMMITTED_EXACT", True
    if (
        mode == "SYNTHETIC_NON_PRODUCTION"
        and lifecycle == "BOUND_CONFIG_SYNTHETIC_EXECUTION"
    ):
        if (
            provenance.get("pre_authority_base_commit") != PRE_AUTHORITY_BASE_COMMIT
            or provenance.get("authority_commit") != AUTHORITY_COMMIT
            or provenance.get("authority_runtime_i1_commit")
            != AUTHORITY_RUNTIME_I1_COMMIT
            or provenance.get("binding_commit") != UNKNOWN
            or provenance.get("current_head") != UNKNOWN
            or provenance.get("config_core_sha256") != FROZEN_CONFIG_CORE_SHA256
        ):
            raise PublicationError("synthetic authority provenance is incomplete")
        return "COMMITTED_SYNTHETIC_NON_PRODUCTION", False
    raise PublicationError("authority mode/lifecycle combination is invalid")


def inspect_committed_bundle(
    output_directory: Path | str,
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(output_directory)
    expected_names = {*OUTPUT_NAMES_EXCLUDING_MARKER, COMMIT_MARKER}
    if not output.is_dir() or output.is_symlink():
        raise PublicationError("committed output directory is absent")
    if {entry.name for entry in output.iterdir()} != expected_names:
        raise PublicationError("committed output member set differs")
    payloads = {}
    for name in expected_names:
        path = output / name
        if path.is_symlink() or not path.is_file():
            raise PublicationError(f"committed member is not regular: {name}")
        payloads[name] = path.read_bytes()
    expected_sums = "".join(
        f"{sha256(payloads[name])}  {name}\n" for name in OUTPUT_JSON_NAMES
    ).encode("ascii")
    if payloads["SHA256SUMS"] != expected_sums:
        raise PublicationError("committed SHA256SUMS differs")
    marker = strict_json(payloads[COMMIT_MARKER], label="publication marker")
    if (
        marker.get("record_type")
        != "ROUTE_A_V3_GSE200304_DEC020_SCRATCH_A1_PUBLICATION_COMMIT_V1"
        or marker.get("output_id") != OUTPUT_ID
        or marker.get("committed") is not True
        or marker.get("commit_marker_written_last") is not True
        or marker.get("sha256sums_sha256") != sha256(expected_sums)
    ):
        raise PublicationError("publication marker semantics differ")
    report = strict_json(payloads["ADJUDICATION_REPORT.json"], label="adjudication report")
    audit = strict_json(payloads["INPUT_EVIDENCE_AUDIT.json"], label="input evidence audit")
    _inspect_report_semantics(report, audit)
    publication_status, production_registerable = _inspect_authority_provenance(
        report, audit, config
    )
    if marker.get("scientific_status") != report.get("status"):
        raise PublicationError("marker/report scientific status differs")
    return {
        "publication_status": publication_status,
        "production_registerable": production_registerable,
        "status": report["status"],
        "qualified": report["qualified"],
        "canonical_record_count": report["canonical_record_count"],
        "private_payload_access_authorized": False,
        "canonical_materialization_execution_authorized": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }


def _git(repo: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", os.fspath(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise BindingError(f"git {' '.join(args)} failed: {message}")
    return completed.stdout


def _git_text(repo: Path, *args: str) -> str:
    return _git(repo, *args).decode("utf-8").strip()


def _commit_parent(repo: Path, commit: str) -> str:
    return _git_text(repo, "rev-parse", f"{commit}^")


def _changed_paths(repo: Path, commit: str) -> list[str]:
    value = _git_text(
        repo,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        commit,
    )
    return sorted(line for line in value.splitlines() if line)


def _show_json(repo: Path, commit: str, path: str) -> dict[str, Any]:
    return strict_json(_git(repo, "show", f"{commit}:{path}"), label=f"{commit}:{path}")


def _show_sha(repo: Path, commit: str, path: str) -> str:
    return sha256(_git(repo, "show", f"{commit}:{path}"))


def validate_production_authority(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_static_config(config)
    repository = config["repository_authority"]
    repo = Path(repository["production_repo_root"])
    if repo != PRODUCTION_REPO_ROOT:
        raise BindingError("production repo root differs")
    if _git_text(repo, "status", "--porcelain"):
        raise BindingError("production worktree is dirty")
    if _git_text(repo, "branch", "--show-current") != repository["branch"]:
        raise BindingError("production branch differs")
    head = _git_text(repo, "rev-parse", "HEAD")
    upstream = _git_text(repo, "rev-parse", "@{u}")
    origin = _git_text(
        repo, "rev-parse", f"refs/remotes/origin/{repository['branch']}"
    )
    if head != upstream or head != origin:
        raise BindingError("production HEAD/upstream/origin differ")

    authority = config["core_authority"]["authority_commit"]
    if authority != AUTHORITY_COMMIT or not _runtime_group_is_bound(repository):
        raise BindingError("authority or authority-runtime/V4-parent binding is UNKNOWN")
    runtime_i = repository["authority_runtime_implementation_commit"]
    runtime_b = repository["authority_runtime_binding_commit"]
    runtime_i1 = repository["authority_runtime_i1_commit"]
    runtime_script_sha = repository[
        "authority_runtime_implementation_script_sha256"
    ]
    runtime_test_sha = repository["authority_runtime_implementation_test_sha256"]
    if (
        runtime_i != AUTHORITY_RUNTIME_I2_COMMIT
        or runtime_b != AUTHORITY_RUNTIME_B2_COMMIT
        or runtime_script_sha != AUTHORITY_RUNTIME_I2_SCRIPT_SHA256
        or runtime_test_sha != AUTHORITY_RUNTIME_I2_TEST_SHA256
    ):
        raise BindingError("authority-runtime I2/B2 frozen identity differs")
    if _commit_parent(repo, authority) != PRE_AUTHORITY_BASE_COMMIT:
        raise BindingError("DEC020 authority commit is not the direct child of ba6746a")
    if _changed_paths(repo, authority) != EXPECTED_AUTHORITY_A_PATHS:
        raise BindingError("DEC020 authority A changed-path set differs")
    for path, expected in {
        **ROOT_CONTRACT_FROZEN_BLOB,
        **AUTHORITY_A_EXACT_CHANGED_BLOBS,
    }.items():
        if _show_sha(repo, authority, path) != expected:
            raise BindingError(f"authority file differs at A: {path}")
        if _show_sha(repo, head, path) != expected:
            raise BindingError(f"authority file drifted after A: {path}")

    if runtime_i1 != AUTHORITY_RUNTIME_I1_COMMIT:
        raise BindingError("authority-runtime frozen I1 commit differs")
    if _commit_parent(repo, runtime_i1) != authority:
        raise BindingError("authority-runtime I1 is not the direct child of authority A")
    if _changed_paths(repo, runtime_i1) != EXPECTED_AUTHORITY_RUNTIME_I1_PATHS:
        raise BindingError("authority-runtime I1 changed-path set differs")
    i1_config_payload = _git(
        repo, "show", f"{runtime_i1}:{AUTHORITY_RUNTIME_CONFIG_REPO_PATH}"
    )
    if (
        len(i1_config_payload) != AUTHORITY_RUNTIME_I1_CONFIG_BYTES
        or _show_sha(repo, runtime_i1, AUTHORITY_RUNTIME_CONFIG_REPO_PATH)
        != AUTHORITY_RUNTIME_I1_CONFIG_SHA256
    ):
        raise BindingError("authority-runtime frozen I1 config identity differs")
    i1_config = strict_json(i1_config_payload, label="authority-runtime frozen I1 config")
    if any(
        i1_config.get("implementation_binding", {}).get(key) != UNKNOWN
        for key in (
            "status",
            "implementation_commit",
            "implementation_script_sha256",
            "implementation_test_sha256",
        )
    ):
        raise BindingError("authority-runtime frozen I1 binding is not exact UNKNOWN")
    if _show_sha(repo, runtime_i1, AUTHORITY_RUNTIME_SCRIPT_REPO_PATH) != (
        AUTHORITY_RUNTIME_I1_SCRIPT_SHA256
    ):
        raise BindingError("authority-runtime frozen I1 script SHA differs")
    if _show_sha(repo, runtime_i1, AUTHORITY_RUNTIME_TEST_REPO_PATH) != (
        AUTHORITY_RUNTIME_I1_TEST_SHA256
    ):
        raise BindingError("authority-runtime frozen I1 test SHA differs")
    if _commit_parent(repo, runtime_i) != runtime_i1:
        raise BindingError("authority-runtime I2 is not the direct child of runtime I1")
    if _changed_paths(repo, runtime_i) != EXPECTED_AUTHORITY_RUNTIME_I_PATHS:
        raise BindingError("authority-runtime I2 changed-path set differs")
    if _commit_parent(repo, runtime_b) != runtime_i:
        raise BindingError("authority-runtime B is not the direct child of runtime I")
    if _changed_paths(repo, runtime_b) != EXPECTED_AUTHORITY_RUNTIME_B_PATHS:
        raise BindingError("authority-runtime B is not config-only")
    runtime_i_config = _show_json(
        repo, runtime_i, AUTHORITY_RUNTIME_CONFIG_REPO_PATH
    )
    runtime_b_config = _show_json(
        repo, runtime_b, AUTHORITY_RUNTIME_CONFIG_REPO_PATH
    )
    if (
        _show_sha(repo, runtime_i, AUTHORITY_RUNTIME_CONFIG_REPO_PATH)
        != AUTHORITY_RUNTIME_I2_CONFIG_SHA256
        or _show_sha(repo, runtime_b, AUTHORITY_RUNTIME_CONFIG_REPO_PATH)
        != AUTHORITY_RUNTIME_B2_CONFIG_SHA256
    ):
        raise BindingError("authority-runtime I2/B2 config identity differs")
    _validate_authority_runtime_i_to_b(
        runtime_i_config,
        runtime_b_config,
        runtime_implementation_commit=runtime_i,
        runtime_script_sha256=runtime_script_sha,
        runtime_test_sha256=runtime_test_sha,
    )
    for commit in (runtime_i, runtime_b, head):
        if _show_sha(repo, commit, AUTHORITY_RUNTIME_SCRIPT_REPO_PATH) != (
            runtime_script_sha
        ):
            raise BindingError("authority-runtime implementation script SHA differs")
        if _show_sha(repo, commit, AUTHORITY_RUNTIME_TEST_REPO_PATH) != runtime_test_sha:
            raise BindingError("authority-runtime implementation test SHA differs")
    if _show_json(repo, head, AUTHORITY_RUNTIME_CONFIG_REPO_PATH) != runtime_b_config:
        raise BindingError("authority-runtime bound config drifted after runtime B")

    for path, expected in PREDECESSOR_V3_FROZEN_BLOBS.items():
        if _show_sha(repo, head, path) != expected:
            raise BindingError(f"predecessor V3 blob drifted: {path}")
    predecessor_v3 = _show_json(repo, head, PREDECESSOR_V3_CONFIG_REPO_PATH)
    if predecessor_v3.get("current_external_state", {}).get("status") != (
        "BLOCKED_DEC019_REPORTED_ENDPOINT_A1_EVIDENCE_INCOMPLETE"
    ):
        raise BindingError("predecessor V3 blocked status changed")

    binding = config["implementation_binding"]
    if binding["status"] == UNKNOWN:
        implementation = head
        if _commit_parent(repo, implementation) != runtime_b:
            raise BindingError("V4 I is not the direct child of authority-runtime B")
        if _changed_paths(repo, implementation) != EXPECTED_I_PATHS:
            raise BindingError("V4 I changed-path set differs")
        if _show_json(repo, implementation, CONFIG_REPO_PATH) != config:
            raise BindingError("V4 I tracked config differs from the validated config")
        return {
            "mode": "PRODUCTION",
            "lifecycle_state": "V4_I_IMPLEMENTATION_UNBOUND",
            "pre_authority_base_commit": PRE_AUTHORITY_BASE_COMMIT,
            "authority_commit": authority,
            "authority_runtime_i1_commit": runtime_i1,
            "authority_runtime_implementation_commit": runtime_i,
            "authority_runtime_binding_commit": runtime_b,
            "implementation_commit": implementation,
            "binding_commit": UNKNOWN,
            "current_head": head,
            "config_core_sha256": FROZEN_CONFIG_CORE_SHA256,
            "evidence_descriptor_set_sha256": config["evidence_descriptor_bindings"]["descriptor_set_sha256"],
        }

    implementation = binding["implementation_commit"]
    binding_commit = head
    if _commit_parent(repo, implementation) != runtime_b:
        raise BindingError("V4 I is not the direct child of authority-runtime B")
    if _changed_paths(repo, implementation) != EXPECTED_I_PATHS:
        raise BindingError("V4 I changed-path set differs")
    if _commit_parent(repo, binding_commit) != implementation:
        raise BindingError("V4 B is not the direct child of V4 I")
    if _changed_paths(repo, binding_commit) != EXPECTED_B_PATHS:
        raise BindingError("V4 B is not config-only")
    i_config = _show_json(repo, implementation, CONFIG_REPO_PATH)
    validate_i_to_b(i_config, config)
    if _show_json(repo, binding_commit, CONFIG_REPO_PATH) != config:
        raise BindingError("V4 B tracked config differs from the validated config")
    if _show_sha(repo, head, SCRIPT_REPO_PATH) != binding["implementation_script_sha256"]:
        raise BindingError("V4 implementation script SHA differs")
    if _show_sha(repo, head, TEST_REPO_PATH) != binding["implementation_test_sha256"]:
        raise BindingError("V4 implementation test SHA differs")
    return {
        "mode": "PRODUCTION",
        "lifecycle_state": "V4_B_BOUND_EXACT_HEAD",
        "pre_authority_base_commit": PRE_AUTHORITY_BASE_COMMIT,
        "authority_commit": authority,
        "authority_runtime_i1_commit": runtime_i1,
        "authority_runtime_implementation_commit": runtime_i,
        "authority_runtime_binding_commit": runtime_b,
        "implementation_commit": implementation,
        "binding_commit": binding_commit,
        "current_head": head,
        "config_core_sha256": FROZEN_CONFIG_CORE_SHA256,
        "evidence_descriptor_set_sha256": config["evidence_descriptor_bindings"]["descriptor_set_sha256"],
    }


def _preflight_output(output: Path, *, production: bool) -> None:
    if production:
        absolute = Path(os.path.abspath(os.fspath(output)))
        if absolute.parent != TRUSTED_A1_OUTPUT_ROOT:
            raise ScopeViolation("production output is not a direct child of the trusted A1 root")
        if not absolute.name.startswith("GSE200304_DEC020_SCRATCH_ROUTE_A1_ADJUDICATION_V4_"):
            raise ScopeViolation("production output basename differs from the V4 namespace")
    if output.exists() and output.is_symlink():
        raise ScopeViolation("output directory is a symlink")


def adjudicate(
    config: Mapping[str, Any],
    output_directory: Path | str,
    *,
    production: bool = False,
) -> dict[str, Any]:
    validate_implementation_binding(config)
    authority = (
        validate_production_authority(config)
        if production
        else _synthetic_authority_provenance(config)
    )
    output = Path(output_directory)
    _preflight_output(output, production=production)
    report, audit = recompute_adjudication_outputs(config, authority)
    publication_status = _publish_bundle(output, _build_bundle(config, report, audit))
    return {
        "publication_status": publication_status,
        "status": report["status"],
        "qualified": report["qualified"],
        "ordinary_study_contribution": report["ordinary_study_contribution"],
        "a1_study_contribution": report["a1_study_contribution"],
        "true_a2_study_contribution": 0,
        "canonical_record_count": report["canonical_record_count"],
        "foundation_route_status": FOUNDATION_ROUTE_STATUS,
        "foundation_checkpoint_evidence_status": FOUNDATION_CHECKPOINT_EVIDENCE_STATUS,
        "private_payload_access_authorized": False,
        "canonical_materialization_execution_authorized": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "gpu_allowed": False,
        "next_phase_authorized": False,
        "blockers": report["blockers"],
    }


def load_production_config() -> dict[str, Any]:
    return strict_json(PRODUCTION_CONFIG_PATH.read_bytes(), label="production V4 config")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--inspect", action="store_true")
    mode.add_argument("--validate-authority", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.validate_authority:
            if args.output_directory is not None:
                parser.error("--validate-authority does not accept --output-directory")
            result = validate_production_authority(load_production_config())
        elif args.inspect:
            if args.output_directory is None:
                parser.error("--inspect requires --output-directory")
            result = inspect_committed_bundle(
                args.output_directory,
                config=load_production_config(),
            )
        else:
            if args.output_directory is None:
                parser.error("adjudication requires --output-directory")
            result = adjudicate(
                load_production_config(), args.output_directory, production=True
            )
    except AdjudicationError as exc:
        print(
            json.dumps(
                {
                    "status": "ERROR",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
