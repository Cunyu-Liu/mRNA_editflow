#!/usr/bin/env python3
"""Fail-closed DEC-019 successor adjudicator for GSE200304.

The author-published processed endpoint may be the A1 primary measurement
route, while raw replay is only a reproducibility auxiliary.  This program
consumes no raw or row-level payload: it accepts only immutable, hash-bound,
aggregate gate records.  A missing implementation binding stops before input;
missing evidence produces a committed aggregate blocked bundle with 0/0/0
study contributions and zero canonical records.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


CONFIG_REPO_PATH = "configs/route_a_v3_gse200304_dec019_reported_endpoint_a1_activation_v2.json"
SCRIPT_REPO_PATH = "scripts/route_a_v3/adjudicate_gse200304_dec019_reported_endpoint_a1.py"
TEST_REPO_PATH = "tests/route_a_v3/test_adjudicate_gse200304_dec019_reported_endpoint_a1.py"
GSE114002_CONFIG_REPO_PATH = (
    "configs/route_a_v3_gse114002_dec019_true_a2_activation_v2.json"
)
GSE200304_CONFIG_REPO_PATH = CONFIG_REPO_PATH
BINDING_CONFIG_REPO_PATHS = (
    GSE114002_CONFIG_REPO_PATH,
    GSE200304_CONFIG_REPO_PATH,
)
EXPECTED_IMPLEMENTATION_FILES = {
    GSE114002_CONFIG_REPO_PATH: (
        "scripts/route_a_v3/adjudicate_gse114002_dec019_true_a2.py",
        "tests/route_a_v3/test_adjudicate_gse114002_dec019_true_a2.py",
    ),
    GSE200304_CONFIG_REPO_PATH: (
        SCRIPT_REPO_PATH,
        TEST_REPO_PATH,
    ),
}
FROZEN_CONFIG_CORE_SHA256_BY_PATH = {
    GSE114002_CONFIG_REPO_PATH: "1c3e4a7aa412e245f6f4680677db60b8241d7873fa126756791bdb0b58f9233a",
    GSE200304_CONFIG_REPO_PATH: "6cbc215d38adf3b3d15de314f674b2ae02b2f1a1a733cb4dec3d75d8f9480943",
}
FROZEN_CONFIG_CORE_SHA256 = FROZEN_CONFIG_CORE_SHA256_BY_PATH[CONFIG_REPO_PATH]
EXPECTED_I_TO_B_SCALAR_PATHS = [
    "implementation_binding.status",
    "implementation_binding.implementation_commit",
    "implementation_binding.implementation_script_sha256",
    "implementation_binding.implementation_test_sha256",
]
ALLOWED_I_TO_B_SCALAR_PATHS = frozenset(EXPECTED_I_TO_B_SCALAR_PATHS)
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
    "policy_boundary",
    "current_external_state",
    "evidence_contract",
    "output_contract",
}
PRODUCTION_REPO_ROOT = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/routea_v3_a1_20260810"
)
PRODUCTION_CONFIG_PATH = PRODUCTION_REPO_ROOT / CONFIG_REPO_PATH
UNKNOWN = "UNKNOWN_NOT_ASSERTED"
CONTRACT_ID = "mrna_xeditflow_route_a_v3"
DATASET_ID = "GSE200304"
DECISION_ID = "V3-DEC-019"
PROTOCOL_ID = "ROUTE_A_V3_GSE200304_DEC019_REPORTED_ENDPOINT_A1_ACTIVATION_V2"
EVIDENCE_SCHEMA_VERSION = "route_a_v3_dec019_accepted_aggregate_gate_evidence.v1"
EVIDENCE_RECORD_TYPE = "ROUTE_A_V3_DEC019_ACCEPTED_AGGREGATE_GATE_EVIDENCE"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DNA_LIKE = re.compile(r"^[ACGTUNacgtun]{20,}$")
OUTPUT_JSON_NAMES = ("ADJUDICATION_REPORT.json", "INPUT_EVIDENCE_AUDIT.json")
OUTPUT_NAMES_EXCLUDING_MARKER = (*OUTPUT_JSON_NAMES, "SHA256SUMS")
COMMIT_MARKER = "PUBLICATION_COMMIT.json"
PUBLICATION_MODE = "ATOMIC_EXCLUSIVE_DIRECTORY_TERMINAL_COMMIT_MARKER_V1"
BLOCKED_STATUS = "BLOCKED_DEC019_REPORTED_ENDPOINT_A1_EVIDENCE_INCOMPLETE"
SUCCESS_STATUS = "PASS_DEC019_REPORTED_ENDPOINT_A1_ACTIVATED_RAW_REPLAY_AUXILIARY"

SLOT_IDS = (
    "CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE",
    "CANONICAL_REPORTED_ENDPOINT_SEMANTICS",
    "BIOLOGICAL_GROUP_AUTHORITY",
    "ROW_REPLICATE_OR_VALID_SE",
    "CHECKPOINT_SPECIFIC_EXPOSURE",
    "LICENSE_RIGHTS",
    "OUTCOME_BLIND_SPLIT_LEAKAGE",
    "PREFROZEN_POWER_PRECISION",
)
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
    "facts",
}
PRIVACY_KEYS = {
    "contains_row_level_payload",
    "contains_sequence",
    "contains_row_identifier",
    "contains_raw_label_or_effect",
    "contains_member_identifiers_or_hashes",
}
FACT_KEYS = {
    "CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE": {
        "deterministic_row_locator_frozen",
        "table_s2_hash_bound",
        "table_s3_hash_bound",
        "s2_s3_join_rule_frozen",
        "multi_asset_lineage_closed",
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
    "CHECKPOINT_SPECIFIC_EXPOSURE": {
        "checkpoint_ids_and_revisions_frozen",
        "checkpoint_artifact_digests_bound",
        "exact_member_exposure_audit_pass",
        "near_duplicate_exposure_audit_pass",
        "audited_checkpoint_count",
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
        "observed_power",
        "full_confidence_interval_width",
        "prefrozen_before_model_results",
    },
}
GATE_BLOCKERS = {
    "CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE_NOT_PASS",
    "CANONICAL_REPORTED_ENDPOINT_SEMANTICS_NOT_PASS",
    "BIOLOGICAL_GROUP_AUTHORITY_NOT_PASS",
    "ROW_REPLICATE_OR_VALID_SE_NOT_PASS",
    "CHECKPOINT_SPECIFIC_EXPOSURE_NOT_PASS",
    "LICENSE_RIGHTS_NOT_PASS",
    "OUTCOME_BLIND_SPLIT_LEAKAGE_NOT_PASS",
    "PREFROZEN_POWER_PRECISION_NOT_PASS",
    "RAW_REPLAY_INDEPENDENT_REPRODUCTION_CLAIM_INVALID",
    "POWER_LT_0_80",
    "FULL_CI_WIDTH_GT_0_30",
}


class AdjudicationError(RuntimeError):
    """Evidence, authority, or execution integrity failed."""


class BindingError(AdjudicationError):
    """The implementation or core authority is not fully bound."""


class ScopeViolation(AdjudicationError):
    """A configured or caller-selected path left the allowed scope."""


class PublicationError(AdjudicationError):
    """A no-overwrite publication invariant failed."""


class PartialPublicationError(PublicationError):
    """An exclusive output exists without an accepted terminal marker."""


FaultInjector = Callable[[str], None]


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
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


def config_core_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    """Stable core: excludes the complete dynamic implementation binding."""

    projected = copy.deepcopy(dict(config))
    projected.pop("implementation_binding", None)
    return projected


def config_core_sha256(config: Mapping[str, Any]) -> str:
    return sha256(json_bytes(config_core_projection(config)))


def _expect_exact_keys(value: Any, expected: set[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise AdjudicationError(f"{label} keys differ from the closed schema")
    return value


def _expect_bool(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise AdjudicationError(f"{label} must be a boolean")
    return value


def _expect_int(value: Any, *, label: str, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise AdjudicationError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise AdjudicationError(f"{label} must be >= {minimum}")
    return value


def _expect_number(value: Any, *, label: str) -> float:
    if type(value) not in {int, float}:
        raise AdjudicationError(f"{label} must be a finite number")
    result = float(value)
    if not (result == result and abs(result) != float("inf")):
        raise AdjudicationError(f"{label} must be finite")
    return result


def _expect_exact(value: Any, expected: Any, *, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise AdjudicationError(f"{label} differs")


def _semantic_diff_paths(left: Any, right: Any, prefix: str = "") -> set[str]:
    """Return exact leaf paths that differ between the I and B JSON objects."""

    if type(left) is not type(right):
        return {prefix or "<root>"}
    if type(left) is dict:
        result: set[str] = set()
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                result.add(path)
            else:
                result.update(_semantic_diff_paths(left[key], right[key], path))
        return result
    if type(left) is list:
        if len(left) != len(right):
            return {prefix}
        result: set[str] = set()
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            result.update(
                _semantic_diff_paths(
                    left_item,
                    right_item,
                    f"{prefix}[{index}]",
                )
            )
        return result
    return {prefix} if left != right else set()


def _validate_i_to_b_config_pair(
    i_config: Mapping[str, Any],
    b_config: Mapping[str, Any],
    *,
    config_path: str,
    implementation_commit: str,
) -> None:
    """Prove one parent-I config changed only through the four allowed B scalars."""

    if config_path not in FROZEN_CONFIG_CORE_SHA256_BY_PATH:
        raise BindingError("I-to-B config path is outside the closed pair")
    if type(i_config) is not dict or set(i_config) != EXPECTED_CONFIG_TOP_KEYS:
        raise BindingError(f"parent-I config top-level schema differs: {config_path}")
    if type(b_config) is not dict or set(b_config) != EXPECTED_CONFIG_TOP_KEYS:
        raise BindingError(f"current-B config top-level schema differs: {config_path}")
    i_binding = i_config.get("implementation_binding")
    b_binding = b_config.get("implementation_binding")
    if type(i_binding) is not dict or set(i_binding) != EXPECTED_IMPLEMENTATION_BINDING_KEYS:
        raise BindingError(f"parent-I implementation binding schema differs: {config_path}")
    if type(b_binding) is not dict or set(b_binding) != EXPECTED_IMPLEMENTATION_BINDING_KEYS:
        raise BindingError(f"current-B implementation binding schema differs: {config_path}")

    expected_script, expected_test = EXPECTED_IMPLEMENTATION_FILES[config_path]
    for binding, label in ((i_binding, "parent-I"), (b_binding, "current-B")):
        if binding.get("binding_scheme") != "CONFIG_ONLY_POST_IMPLEMENTATION_BINDING_V1":
            raise BindingError(f"{label} binding scheme differs: {config_path}")
        if binding.get("blocker_if_unbound") != "IMPLEMENTATION_BINDING_UNKNOWN_NOT_ASSERTED":
            raise BindingError(f"{label} implementation blocker differs: {config_path}")
        if binding.get("implementation_script_path") != expected_script:
            raise BindingError(f"{label} implementation script path differs: {config_path}")
        if binding.get("implementation_test_path") != expected_test:
            raise BindingError(f"{label} implementation test path differs: {config_path}")
        if binding.get("unknown_to_bound_scalar_paths") != EXPECTED_I_TO_B_SCALAR_PATHS:
            raise BindingError(f"{label} I-to-B scalar allowlist differs: {config_path}")

    if i_binding.get("status") != UNKNOWN or any(
        i_binding.get(key) != UNKNOWN
        for key in (
            "implementation_commit",
            "implementation_script_sha256",
            "implementation_test_sha256",
        )
    ):
        raise BindingError(f"parent-I binding is not exact UNKNOWN: {config_path}")
    if (
        b_binding.get("status") != "BOUND"
        or b_binding.get("implementation_commit") != implementation_commit
        or HEX64.fullmatch(str(b_binding.get("implementation_script_sha256"))) is None
        or HEX64.fullmatch(str(b_binding.get("implementation_test_sha256"))) is None
    ):
        raise BindingError(f"current-B binding is not exact BOUND: {config_path}")

    frozen_core = FROZEN_CONFIG_CORE_SHA256_BY_PATH[config_path]
    for value, binding, label in (
        (i_config, i_binding, "parent-I"),
        (b_config, b_binding, "current-B"),
    ):
        if binding.get("config_core_sha256") != frozen_core:
            raise BindingError(f"{label} stored core differs from frozen core: {config_path}")
        if config_core_sha256(value) != frozen_core:
            raise BindingError(f"{label} computed core differs from frozen core: {config_path}")

    for value, label in ((i_config, "parent-I"), (b_config, "current-B")):
        repository = value.get("repository_authority")
        if type(repository) is not dict:
            raise BindingError(f"{label} repository authority is absent: {config_path}")
        if repository.get("base_commit") != "ad1c57b9255c3066510b08e7a4cf0bd571006811":
            raise BindingError(f"{label} base commit differs: {config_path}")
        if repository.get("implementation_commit_expected_parent") != repository.get(
            "base_commit"
        ):
            raise BindingError(f"{label} implementation-parent rule differs: {config_path}")
        if repository.get("binding_commit_expected_parent") != "IMPLEMENTATION_COMMIT_FROM_BINDING":
            raise BindingError(f"{label} binding-parent rule differs: {config_path}")
        if repository.get("binding_commit_exact_changed_paths") != list(
            BINDING_CONFIG_REPO_PATHS
        ):
            raise BindingError(f"{label} binding changed-path set differs: {config_path}")
        if repository.get("implementation_commit_required_paths") != [
            config_path,
            expected_script,
            expected_test,
        ]:
            raise BindingError(
                f"{label} implementation required-path set differs: {config_path}"
            )

    differences = _semantic_diff_paths(i_config, b_config)
    if differences != set(ALLOWED_I_TO_B_SCALAR_PATHS):
        raise BindingError(
            f"I-to-B semantic diff is not the exact four-scalar allowlist: "
            f"{config_path}: {sorted(differences)!r}"
        )



def _assert_no_private_material(value: Any, forbidden_keys: set[str], *, label: str) -> None:
    if type(value) is dict:
        for key, child in value.items():
            if key.casefold() in forbidden_keys:
                raise AdjudicationError(f"{label} contains forbidden key: {key}")
            _assert_no_private_material(child, forbidden_keys, label=label)
    elif type(value) is list:
        for child in value:
            _assert_no_private_material(child, forbidden_keys, label=label)
    elif type(value) is str and DNA_LIKE.fullmatch(value):
        raise AdjudicationError(f"{label} contains a sequence-like string")


def _core_leaf_pairs(config: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    authority = config["core_authority"]
    return (
        (authority["root_contract_path"], authority["root_contract_sha256"]),
        (authority["amendment_path"], authority["amendment_sha256"]),
        (authority["decision_log_path"], authority["decision_log_sha256"]),
        (authority["data_role_registry_path"], authority["data_role_registry_sha256"]),
        (authority["split_registry_path"], authority["split_registry_sha256"]),
        (authority["task_registry_path"], authority["task_registry_sha256"]),
        (authority["task_split_matrix_path"], authority["task_split_matrix_sha256"]),
        (authority["claim_evidence_matrix_path"], authority["claim_evidence_matrix_sha256"]),
        (authority["a1_qualification_path"], authority["a1_qualification_sha256"]),
    )


def validate_static_config(config: Mapping[str, Any]) -> None:
    _expect_exact_keys(
        config,
        EXPECTED_CONFIG_TOP_KEYS,
        label="config",
    )
    for key, expected in {
        "schema_version": "route_a_v3_gse200304_dec019_reported_endpoint_a1_activation.v2",
        "protocol_id": PROTOCOL_ID,
        "contract_id": CONTRACT_ID,
        "phase_id": "A1",
        "dataset_id": DATASET_ID,
        "decision_id": DECISION_ID,
    }.items():
        _expect_exact(config[key], expected, label=f"config {key}")

    binding = _expect_exact_keys(
        config["implementation_binding"],
        EXPECTED_IMPLEMENTATION_BINDING_KEYS,
        label="implementation binding",
    )
    _expect_exact(binding["binding_scheme"], "CONFIG_ONLY_POST_IMPLEMENTATION_BINDING_V1", label="binding scheme")
    _expect_exact(
        binding["blocker_if_unbound"],
        "IMPLEMENTATION_BINDING_UNKNOWN_NOT_ASSERTED",
        label="implementation blocker",
    )
    if binding["status"] not in {UNKNOWN, "BOUND"}:
        raise AdjudicationError("implementation binding status is outside the closed enum")
    dynamic_fields = (
        "implementation_commit",
        "implementation_script_sha256",
        "implementation_test_sha256",
    )
    if binding["status"] == UNKNOWN and any(binding[key] != UNKNOWN for key in dynamic_fields):
        raise AdjudicationError("UNKNOWN implementation binding has a partially bound field")
    if binding["status"] == "BOUND" and (
        HEX40.fullmatch(str(binding["implementation_commit"])) is None
        or HEX64.fullmatch(str(binding["implementation_script_sha256"])) is None
        or HEX64.fullmatch(str(binding["implementation_test_sha256"])) is None
    ):
        raise AdjudicationError("BOUND implementation binding has an invalid dynamic field")
    _expect_exact(
        binding["unknown_to_bound_scalar_paths"],
        [
            "implementation_binding.status",
            "implementation_binding.implementation_commit",
            "implementation_binding.implementation_script_sha256",
            "implementation_binding.implementation_test_sha256",
        ],
        label="I-to-B scalar paths",
    )
    _expect_exact(binding["implementation_script_path"], SCRIPT_REPO_PATH, label="script path")
    _expect_exact(binding["implementation_test_path"], TEST_REPO_PATH, label="test path")
    if HEX64.fullmatch(str(binding["config_core_sha256"])) is None:
        raise AdjudicationError("config core SHA is not bound")
    if binding["config_core_sha256"] != config_core_sha256(config):
        raise AdjudicationError("config core projection SHA differs")

    repository = _expect_exact_keys(
        config["repository_authority"],
        {
            "production_repo_root",
            "branch",
            "base_commit",
            "implementation_commit_expected_parent",
            "binding_commit_expected_parent",
            "implementation_commit_required_paths",
            "binding_commit_exact_changed_paths",
        },
        label="repository authority",
    )
    _expect_exact(repository["production_repo_root"], os.fspath(PRODUCTION_REPO_ROOT), label="production repo root")
    _expect_exact(repository["branch"], "routea-v3-a1-20260810", label="branch")
    _expect_exact(repository["base_commit"], "ad1c57b9255c3066510b08e7a4cf0bd571006811", label="base commit")
    _expect_exact(repository["implementation_commit_expected_parent"], repository["base_commit"], label="implementation parent")
    _expect_exact(repository["binding_commit_expected_parent"], "IMPLEMENTATION_COMMIT_FROM_BINDING", label="binding parent rule")
    _expect_exact(
        repository["implementation_commit_required_paths"],
        [CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH],
        label="implementation required paths",
    )
    _expect_exact(
        repository["binding_commit_exact_changed_paths"],
        list(BINDING_CONFIG_REPO_PATHS),
        label="binding exact changed paths",
    )

    authority = _expect_exact_keys(
        config["core_authority"],
        {
            "status",
            "root_contract_path",
            "root_contract_sha256",
            "amendment_path",
            "amendment_sha256",
            "decision_log_path",
            "decision_log_sha256",
            "data_role_registry_path",
            "data_role_registry_sha256",
            "split_registry_path",
            "split_registry_sha256",
            "task_registry_path",
            "task_registry_sha256",
            "task_split_matrix_path",
            "task_split_matrix_sha256",
            "claim_evidence_matrix_path",
            "claim_evidence_matrix_sha256",
            "a1_qualification_path",
            "a1_qualification_sha256",
            "forbidden_cyclic_dependencies",
        },
        label="core authority",
    )
    _expect_exact(authority["status"], "BOUND", label="core authority status")
    _expect_exact(
        authority["root_contract_sha256"],
        "cbac4c3dcba8f1b8df95d8edad52d19e3c126d1c865d0cc423537c754cc90982",
        label="root contract SHA",
    )
    forbidden_cycles = set(authority["forbidden_cyclic_dependencies"])
    if forbidden_cycles != {
        "docs/execution/route_a_v3_a1_interim.yaml",
        "docs/execution/route_a_v3_registry_manifest.json",
        "scripts/route_a_v3/validate_a0_bundle.py",
    }:
        raise AdjudicationError("forbidden cyclic dependency list differs")
    if any(path in forbidden_cycles for path, _ in _core_leaf_pairs(config)):
        raise AdjudicationError("a core trust leaf creates a forbidden cycle")
    if any(HEX64.fullmatch(str(digest)) is None for _, digest in _core_leaf_pairs(config)):
        raise AdjudicationError("a core authority leaf SHA is not bound")

    policy = config["policy_boundary"]
    fixed_policy = {
        "primary_measurement_route": "AUTHOR_PUBLISHED_PROCESSED_ENDPOINT",
        "raw_replay_role": "REPRODUCIBILITY_AUXILIARY_NOT_QUALIFICATION_PREREQUISITE",
        "raw_replay_not_run_blocks_a1_qualification": False,
        "raw_replay_not_run_allows_independent_reproduction_claim": False,
        "deterministic_row_locator_required": True,
        "table_s2_and_s3_multi_asset_lineage_required": True,
        "endpoint_semantics_and_direction_required": True,
        "biological_group_required": True,
        "replicate_or_valid_standard_error_required": True,
        "outcome_blind_split_and_leakage_required": True,
        "minimum_power": 0.8,
        "maximum_full_confidence_interval_width": 0.3,
        "rights_required": True,
        "checkpoint_specific_exposure_required": True,
        "study_counting_unit": "DATASET",
        "maximum_study_contribution_per_dataset": 1,
        "gsm_pool_subseries_modality_endpoint_replicate_may_multiply_study_count": False,
        "ordinary_gate_contribution_on_success": 1,
        "a1_gate_contribution_on_success": 1,
        "true_a2_gate_contribution_on_success": 0,
        "raw_or_row_level_payload_consumption_allowed": False,
        "training_allowed_by_this_adjudicator": False,
        "model_selection_allowed_by_this_adjudicator": False,
        "next_phase_allowed_by_this_adjudicator": False,
    }
    if policy != fixed_policy:
        raise AdjudicationError("DEC-019 GSE200304 policy boundary differs")

    state = _expect_exact_keys(
        config["current_external_state"],
        {
            "status",
            "qualified",
            "ordinary_study_contribution",
            "a1_study_contribution",
            "true_a2_study_contribution",
            "canonical_record_count",
            "canonical_materialization_allowed",
            "independent_raw_reproduction_established",
            "training_allowed",
            "model_selection_allowed",
            "next_phase_authorized",
            "scientific_claim_status",
            "unresolved_blockers",
        },
        label="current external state",
    )
    for key, expected in {
        "status": BLOCKED_STATUS,
        "qualified": False,
        "ordinary_study_contribution": 0,
        "a1_study_contribution": 0,
        "true_a2_study_contribution": 0,
        "canonical_record_count": 0,
        "canonical_materialization_allowed": False,
        "independent_raw_reproduction_established": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }.items():
        _expect_exact(state[key], expected, label=f"current state {key}")

    evidence_contract = _expect_exact_keys(
        config["evidence_contract"],
        {
            "all_inputs_must_be_hash_bound_accepted_aggregate_json",
            "all_descriptors_must_be_bound_before_any_input_open",
            "root_to_leaf_symlink_rejection_required",
            "single_link_regular_file_required",
            "same_descriptor_verified_snapshot_required",
            "extra_json_keys_rejected",
            "row_level_payload_forbidden",
            "forbidden_path_tokens",
            "slots",
        },
        label="evidence contract",
    )
    for key in {
        "all_inputs_must_be_hash_bound_accepted_aggregate_json",
        "all_descriptors_must_be_bound_before_any_input_open",
        "root_to_leaf_symlink_rejection_required",
        "single_link_regular_file_required",
        "same_descriptor_verified_snapshot_required",
        "extra_json_keys_rejected",
        "row_level_payload_forbidden",
    }:
        _expect_exact(evidence_contract[key], True, label=f"evidence contract {key}")
    slots = evidence_contract["slots"]
    if type(slots) is not list or tuple(slot.get("slot_id") for slot in slots) != SLOT_IDS:
        raise AdjudicationError("evidence slot IDs or order differ")
    for slot in slots:
        _expect_exact_keys(
            slot,
            {
                "slot_id",
                "allowed_basename",
                "absolute_path",
                "sha256",
                "bytes",
                "blocker_if_unbound",
                "blocker_if_not_pass",
            },
            label=f"evidence slot {slot.get('slot_id')}",
        )
    if state["unresolved_blockers"] != [slot["blocker_if_unbound"] for slot in slots]:
        raise AdjudicationError("current eight-blocker order differs")

    output = _expect_exact_keys(
        config["output_contract"],
        {
            "output_id",
            "blocked_status",
            "success_status",
            "aggregate_only",
            "member_names_excluding_commit_marker",
            "terminal_commit_marker",
            "publication_mode",
            "commit_marker_written_last",
            "existing_exact_is_idempotent",
            "overwrite_allowed",
            "partial_publication_is_never_accepted",
            "forbidden_output_keys",
        },
        label="output contract",
    )
    _expect_exact(output["output_id"], "ROUTE_A_V3_GSE200304_DEC019_REPORTED_ENDPOINT_A1_ADJUDICATION_BUNDLE_V2", label="output ID")
    _expect_exact(output["blocked_status"], BLOCKED_STATUS, label="blocked status")
    _expect_exact(output["success_status"], SUCCESS_STATUS, label="success status")
    _expect_exact(output["publication_mode"], PUBLICATION_MODE, label="publication mode")
    _expect_exact(output["member_names_excluding_commit_marker"], list(OUTPUT_NAMES_EXCLUDING_MARKER), label="output members")
    for key, expected in {
        "aggregate_only": True,
        "terminal_commit_marker": COMMIT_MARKER,
        "commit_marker_written_last": True,
        "existing_exact_is_idempotent": True,
        "overwrite_allowed": False,
        "partial_publication_is_never_accepted": True,
    }.items():
        _expect_exact(output[key], expected, label=f"output contract {key}")


def _all_evidence_descriptors_bound(config: Mapping[str, Any]) -> bool:
    for slot in config["evidence_contract"]["slots"]:
        if (
            type(slot["absolute_path"]) is not str
            or slot["absolute_path"] == UNKNOWN
            or HEX64.fullmatch(str(slot["sha256"])) is None
            or type(slot["bytes"]) is not int
            or type(slot["bytes"]) is bool
            or slot["bytes"] <= 0
        ):
            return False
    return True


def validate_implementation_binding(config: Mapping[str, Any]) -> None:
    validate_static_config(config)
    binding = config["implementation_binding"]
    if binding["status"] != "BOUND":
        raise BindingError(
            "IMPLEMENTATION_BINDING_UNKNOWN_NOT_ASSERTED; stopped before evidence input"
        )
    if HEX40.fullmatch(str(binding["implementation_commit"])) is None:
        raise BindingError("implementation commit is not bound")
    if HEX64.fullmatch(str(binding["implementation_script_sha256"])) is None:
        raise BindingError("implementation script SHA is not bound")
    if HEX64.fullmatch(str(binding["implementation_test_sha256"])) is None:
        raise BindingError("implementation test SHA is not bound")


def _git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["/usr/bin/git", *args],
            cwd=repo_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        raise BindingError(f"git authority check failed: {' '.join(args)}") from exc
    return result.stdout.strip()


def _git_bytes(repo_root: Path, *args: str) -> bytes:
    try:
        result = subprocess.run(
            ["/usr/bin/git", *args],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        raise BindingError(f"git blob authority check failed: {' '.join(args)}") from exc
    return result.stdout


def _verify_repo_file(repo_root: Path, relative_path: str, expected_sha: str) -> None:
    try:
        payload = (repo_root / relative_path).read_bytes()
    except OSError as exc:
        raise BindingError(f"authority leaf is unavailable: {relative_path}") from exc
    if sha256(payload) != expected_sha:
        raise BindingError(f"authority leaf SHA differs: {relative_path}")


def validate_production_authority(config: Mapping[str, Any]) -> None:
    """Prove the exact base -> parent-I -> two-config-only B chain."""

    validate_implementation_binding(config)
    repo = Path(config["repository_authority"]["production_repo_root"])
    if repo != PRODUCTION_REPO_ROOT:
        raise BindingError("production repository root differs")
    if config["core_authority"]["status"] != "BOUND":
        raise BindingError("core authority is UNKNOWN_NOT_ASSERTED")

    binding = config["implementation_binding"]
    implementation = binding["implementation_commit"]
    head = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "rev-parse", "HEAD^") != implementation:
        raise BindingError("binding commit is not the direct child of implementation commit")
    if _git(repo, "rev-parse", f"{implementation}^") != config["repository_authority"]["base_commit"]:
        raise BindingError("implementation commit parent differs from the frozen base")

    changed = sorted(
        line
        for line in _git(
            repo, "diff-tree", "--no-commit-id", "--name-only", "-r", head
        ).splitlines()
        if line
    )
    if changed != list(BINDING_CONFIG_REPO_PATHS):
        raise BindingError("binding commit is not the exact two-config-only B commit")

    implementation_changed = {
        line
        for line in _git(
            repo,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            implementation,
        ).splitlines()
        if line
    }
    required_i_paths = set(BINDING_CONFIG_REPO_PATHS)
    for script_path, test_path in EXPECTED_IMPLEMENTATION_FILES.values():
        required_i_paths.update((script_path, test_path))
    if not required_i_paths.issubset(implementation_changed):
        raise BindingError("implementation commit lacks a required successor path")

    if _git(repo, "rev-parse", "--abbrev-ref", "HEAD") != config["repository_authority"]["branch"]:
        raise BindingError("production branch differs")
    if _git(repo, "status", "--porcelain"):
        raise BindingError("production worktree is not clean")
    tracking = _git(
        repo,
        "rev-parse",
        f"refs/remotes/origin/{config['repository_authority']['branch']}",
    )
    if tracking != head:
        raise BindingError("origin tracking ref is not the binding commit")

    b_configs: dict[str, dict[str, Any]] = {}
    for config_path in BINDING_CONFIG_REPO_PATHS:
        i_payload = _git_bytes(repo, "show", f"{implementation}:{config_path}")
        b_payload = _git_bytes(repo, "show", f"{head}:{config_path}")
        try:
            working_payload = (repo / config_path).read_bytes()
        except OSError as exc:
            raise BindingError(f"working config is unavailable: {config_path}") from exc
        if working_payload != b_payload:
            raise BindingError(
                f"working config bytes differ from current-B blob: {config_path}"
            )
        i_config = strict_json(i_payload, label=f"parent-I config {config_path}")
        b_config = strict_json(b_payload, label=f"current-B config {config_path}")
        _validate_i_to_b_config_pair(
            i_config,
            b_config,
            config_path=config_path,
            implementation_commit=implementation,
        )
        b_configs[config_path] = b_config
        b_binding = b_config["implementation_binding"]
        script_path, test_path = EXPECTED_IMPLEMENTATION_FILES[config_path]
        _verify_repo_file(
            repo, script_path, b_binding["implementation_script_sha256"]
        )
        _verify_repo_file(repo, test_path, b_binding["implementation_test_sha256"])

    if b_configs[CONFIG_REPO_PATH] != dict(config):
        raise BindingError(
            "in-memory config differs from the exact current-B repository blob"
        )

    for relative_path, expected_sha in _core_leaf_pairs(config):
        if HEX64.fullmatch(str(expected_sha)) is None:
            raise BindingError(f"core leaf SHA is not bound: {relative_path}")
        _verify_repo_file(repo, relative_path, expected_sha)


def _reject_path(path: Path, config: Mapping[str, Any], *, label: str) -> None:
    lowered = os.fspath(path).casefold()
    hits = [token for token in config["evidence_contract"]["forbidden_path_tokens"] if token.casefold() in lowered]
    if hits:
        raise ScopeViolation(f"{label} contains forbidden path token(s): {','.join(hits)}")
    if not path.is_absolute():
        raise ScopeViolation(f"{label} must be absolute")
    if any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise ScopeViolation(f"{label} contains unsafe components")


def _open_root_to_leaf(path: Path, *, label: str) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise AdjudicationError("O_NOFOLLOW is unavailable")
    parts = path.parts
    if len(parts) < 2 or parts[0] != os.sep:
        raise ScopeViolation(f"{label} is not absolute")
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | nofollow
    descriptor = os.open(os.sep, directory_flags)
    try:
        for component in parts[1:-1]:
            try:
                next_descriptor = os.open(component, directory_flags, dir_fd=descriptor)
            except OSError as exc:
                raise AdjudicationError(f"{label} parent contains a symlink or non-directory") from exc
            os.close(descriptor)
            descriptor = next_descriptor
        try:
            return os.open(parts[-1], os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow, dir_fd=descriptor)
        except OSError as exc:
            raise AdjudicationError(f"{label} cannot be opened safely") from exc
    finally:
        os.close(descriptor)


def _read_verified_evidence(slot: Mapping[str, Any], config: Mapping[str, Any]) -> bytes:
    path = Path(slot["absolute_path"])
    _reject_path(path, config, label=f"evidence {slot['slot_id']}")
    if path.name != slot["allowed_basename"]:
        raise ScopeViolation(f"evidence basename differs: {slot['slot_id']}")
    descriptor = _open_root_to_leaf(path, label=f"evidence {slot['slot_id']}")
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise AdjudicationError(f"evidence is not a single-link regular file: {slot['slot_id']}")
        if before.st_size != slot["bytes"]:
            raise AdjudicationError(f"evidence byte count differs: {slot['slot_id']}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise AdjudicationError(f"evidence changed during verified read: {slot['slot_id']}")
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if sha256(payload) != slot["sha256"]:
        raise AdjudicationError(f"evidence SHA differs: {slot['slot_id']}")
    return payload


def _validate_privacy(record: Mapping[str, Any], config: Mapping[str, Any], *, label: str) -> None:
    privacy = _expect_exact_keys(record["privacy"], PRIVACY_KEYS, label=f"{label} privacy")
    for key in sorted(PRIVACY_KEYS):
        _expect_exact(privacy[key], False, label=f"{label} privacy {key}")
    forbidden = {key.casefold() for key in config["output_contract"]["forbidden_output_keys"]}
    _assert_no_private_material(record, forbidden, label=label)


def _validate_fact_types(slot_id: str, facts: Mapping[str, Any]) -> None:
    boolean_keys = set(facts)
    if slot_id == "CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE":
        for key in {"canonical_record_count", "processed_pair_count"}:
            _expect_int(facts[key], label=f"{slot_id} {key}", minimum=0)
            boolean_keys.remove(key)
        if facts["raw_replay_role"] != "REPRODUCIBILITY_AUXILIARY_NOT_QUALIFICATION_PREREQUISITE":
            raise AdjudicationError("raw replay role differs from DEC-019")
        if facts["raw_replay_status"] not in {"NOT_RUN", "PASS_INDEPENDENT_REPRODUCTION"}:
            raise AdjudicationError("raw replay status is outside the closed enum")
        boolean_keys.remove("raw_replay_role")
        boolean_keys.remove("raw_replay_status")
    elif slot_id == "CHECKPOINT_SPECIFIC_EXPOSURE":
        _expect_int(facts["audited_checkpoint_count"], label="audited checkpoint count", minimum=0)
        boolean_keys.remove("audited_checkpoint_count")
    elif slot_id == "LICENSE_RIGHTS":
        if facts["redistribution_scope"] not in {"PRIVATE_CANONICAL_ONLY", "PUBLIC_REDISTRIBUTION_ALLOWED"}:
            raise AdjudicationError("LICENSE_RIGHTS redistribution_scope is outside the closed enum")
        boolean_keys.remove("redistribution_scope")
    elif slot_id == "PREFROZEN_POWER_PRECISION":
        _expect_number(facts["observed_power"], label="observed power")
        _expect_number(facts["full_confidence_interval_width"], label="full CI width")
        if facts["analysis_unit"] != "BIOLOGICAL_GROUP" or facts["bootstrap_unit"] != "BIOLOGICAL_GROUP":
            raise AdjudicationError("power analysis/bootstrap unit differs")
        boolean_keys -= {"analysis_unit", "bootstrap_unit", "observed_power", "full_confidence_interval_width"}
    for key in boolean_keys:
        _expect_bool(facts[key], label=f"{slot_id} {key}")


def _validate_gate_record(payload: bytes, slot: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    label = f"gate evidence {slot['slot_id']}"
    record = strict_json(payload, label=label)
    _expect_exact_keys(record, COMMON_EVIDENCE_KEYS, label=label)
    for key, expected in {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "record_type": EVIDENCE_RECORD_TYPE,
        "contract_id": CONTRACT_ID,
        "decision_id": DECISION_ID,
        "dataset_id": DATASET_ID,
        "gate_id": slot["slot_id"],
        "accepted": True,
        "aggregate_only": True,
    }.items():
        _expect_exact(record[key], expected, label=f"{label} {key}")
    if record["status"] not in {"PASS", "FAIL", "NOT_RUN", UNKNOWN}:
        raise AdjudicationError(f"{label} status is outside the closed enum")
    _validate_privacy(record, config, label=label)
    facts = _expect_exact_keys(record["facts"], FACT_KEYS[slot["slot_id"]], label=f"{label} facts")
    _validate_fact_types(slot["slot_id"], facts)
    return record


def _slot_gate_pass(slot_id: str, facts: Mapping[str, Any]) -> bool:
    if slot_id == "CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE":
        return (
            all(
                facts[key] is True
                for key in {
                    "deterministic_row_locator_frozen",
                    "table_s2_hash_bound",
                    "table_s3_hash_bound",
                    "s2_s3_join_rule_frozen",
                    "multi_asset_lineage_closed",
                }
            )
            and facts["canonical_record_count"] >= 1
            and facts["processed_pair_count"] >= facts["canonical_record_count"]
            and facts["raw_replay_role"] == "REPRODUCIBILITY_AUXILIARY_NOT_QUALIFICATION_PREREQUISITE"
        )
    if slot_id == "CANONICAL_REPORTED_ENDPOINT_SEMANTICS":
        return all(value is True for value in facts.values())
    if slot_id == "BIOLOGICAL_GROUP_AUTHORITY":
        return all(value is True for value in facts.values())
    if slot_id == "ROW_REPLICATE_OR_VALID_SE":
        return all(value is True for value in facts.values())
    if slot_id == "CHECKPOINT_SPECIFIC_EXPOSURE":
        return all(
            facts[key] is True
            for key in {
                "checkpoint_ids_and_revisions_frozen",
                "checkpoint_artifact_digests_bound",
                "exact_member_exposure_audit_pass",
                "near_duplicate_exposure_audit_pass",
            }
        ) and facts["audited_checkpoint_count"] >= 1
    if slot_id == "LICENSE_RIGHTS":
        return all(
            facts[key] is True
            for key in {
                "rights_source_authority_closed",
                "qualification_use_allowed",
                "private_canonical_materialization_allowed",
            }
        )
    if slot_id == "OUTCOME_BLIND_SPLIT_LEAKAGE":
        return all(value is True for value in facts.values())
    if slot_id == "PREFROZEN_POWER_PRECISION":
        return (
            facts["analysis_unit"] == "BIOLOGICAL_GROUP"
            and facts["bootstrap_unit"] == "BIOLOGICAL_GROUP"
            and facts["prefrozen_before_model_results"] is True
        )
    raise AdjudicationError(f"unknown gate slot: {slot_id}")


def _evaluate(records: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]) -> tuple[list[str], int, bool]:
    blockers: list[str] = []
    for slot in config["evidence_contract"]["slots"]:
        record = records[slot["slot_id"]]
        if record["status"] != "PASS" or not _slot_gate_pass(slot["slot_id"], record["facts"]):
            blockers.append(slot["blocker_if_not_pass"])
    lineage = records["CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE"]["facts"]
    raw_status = lineage["raw_replay_status"]
    raw_claimed = lineage["independent_raw_reproduction_claimed"]
    if raw_status == "NOT_RUN" and raw_claimed is not False:
        blockers.append("RAW_REPLAY_INDEPENDENT_REPRODUCTION_CLAIM_INVALID")
    power = records["PREFROZEN_POWER_PRECISION"]["facts"]
    if float(power["observed_power"]) < 0.8:
        blockers.append("POWER_LT_0_80")
    if float(power["full_confidence_interval_width"]) > 0.3:
        blockers.append("FULL_CI_WIDTH_GT_0_30")
    blockers = sorted(set(blockers))
    if not set(blockers).issubset(GATE_BLOCKERS):
        raise AdjudicationError("an unregistered blocker was produced")
    canonical_count = int(lineage["canonical_record_count"]) if not blockers else 0
    independent_reproduction = raw_status == "PASS_INDEPENDENT_REPRODUCTION" and raw_claimed is True
    return blockers, canonical_count, independent_reproduction


def _blocked_report(config: Mapping[str, Any], blockers: Sequence[str]) -> dict[str, Any]:
    return {
        "record_type": "ROUTE_A_V3_GSE200304_DEC019_REPORTED_ENDPOINT_A1_ADJUDICATION_REPORT_V2",
        "contract_id": CONTRACT_ID,
        "decision_id": DECISION_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "status": BLOCKED_STATUS,
        "qualified": False,
        "data_role": "A1_ORDINARY_REPORTED_ENDPOINT_CANDIDATE_NOT_QUALIFIED",
        "primary_measurement_route": "AUTHOR_PUBLISHED_PROCESSED_ENDPOINT",
        "raw_replay_role": "REPRODUCIBILITY_AUXILIARY_NOT_QUALIFICATION_PREREQUISITE",
        "ordinary_study_contribution": 0,
        "a1_study_contribution": 0,
        "true_a2_study_contribution": 0,
        "canonical_record_count": 0,
        "canonical_materialization_allowed": False,
        "independent_raw_reproduction_established": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "aggregate_only": True,
        "blockers": list(blockers),
        "config_core_sha256": config["implementation_binding"]["config_core_sha256"],
    }


def _success_report(config: Mapping[str, Any], canonical_count: int, independent_reproduction: bool) -> dict[str, Any]:
    return {
        "record_type": "ROUTE_A_V3_GSE200304_DEC019_REPORTED_ENDPOINT_A1_ADJUDICATION_REPORT_V2",
        "contract_id": CONTRACT_ID,
        "decision_id": DECISION_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "status": SUCCESS_STATUS,
        "qualified": True,
        "data_role": "A1_ORDINARY_AUTHOR_PUBLISHED_PROCESSED_ENDPOINT_PRIMARY",
        "primary_measurement_route": "AUTHOR_PUBLISHED_PROCESSED_ENDPOINT",
        "raw_replay_role": "REPRODUCIBILITY_AUXILIARY_NOT_QUALIFICATION_PREREQUISITE",
        "ordinary_study_contribution": 1,
        "a1_study_contribution": 1,
        "true_a2_study_contribution": 0,
        "canonical_record_count": canonical_count,
        "canonical_materialization_allowed": True,
        "independent_raw_reproduction_established": independent_reproduction,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "aggregate_only": True,
        "blockers": [],
        "config_core_sha256": config["implementation_binding"]["config_core_sha256"],
    }


def _input_audit(config: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]] | None) -> dict[str, Any]:
    slots = []
    for slot in config["evidence_contract"]["slots"]:
        bound = (
            type(slot["absolute_path"]) is str
            and slot["absolute_path"] != UNKNOWN
            and HEX64.fullmatch(str(slot["sha256"])) is not None
            and type(slot["bytes"]) is int
            and type(slot["bytes"]) is not bool
            and slot["bytes"] > 0
        )
        record = records.get(slot["slot_id"]) if records is not None else None
        slots.append(
            {
                "slot_id": slot["slot_id"],
                "descriptor_bound": bound,
                "input_opened": record is not None,
                "hash_verified": record is not None,
                "gate_status": record["status"] if record is not None else UNKNOWN,
            }
        )
    return {
        "record_type": "ROUTE_A_V3_DEC019_AGGREGATE_INPUT_EVIDENCE_AUDIT_V1",
        "contract_id": CONTRACT_ID,
        "decision_id": DECISION_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "mode": "ALL_HASH_BOUND_AGGREGATES_VERIFIED" if records is not None else "NO_INPUT_READ_EVIDENCE_BINDING_INCOMPLETE",
        "all_inputs_aggregate_only": True,
        "row_level_payload_read_count": 0,
        "sequence_read_count": 0,
        "opened_input_count": len(records) if records is not None else 0,
        "slots": slots,
    }


def _validate_output_privacy(value: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    forbidden = {key.casefold() for key in config["output_contract"]["forbidden_output_keys"]}
    _assert_no_private_material(value, forbidden, label="output")


def _final_target_sha256(path: Path) -> str:
    return sha256(os.path.abspath(os.fspath(path)).encode("utf-8"))


def _build_bundle(
    config: Mapping[str, Any],
    output_directory: Path,
    report: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, bytes]:
    _validate_output_privacy(report, config)
    _validate_output_privacy(audit, config)
    payloads = {
        "ADJUDICATION_REPORT.json": json_bytes(report),
        "INPUT_EVIDENCE_AUDIT.json": json_bytes(audit),
    }
    payloads["SHA256SUMS"] = "".join(
        f"{sha256(payloads[name])}  {name}\n" for name in sorted(OUTPUT_JSON_NAMES)
    ).encode("ascii")
    marker = {
        "schema_version": "1.0.0",
        "record_type": "ROUTE_A_V3_GSE200304_DEC019_REPORTED_ENDPOINT_A1_ADJUDICATION_COMMIT_V2",
        "contract_id": CONTRACT_ID,
        "decision_id": DECISION_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "output_id": config["output_contract"]["output_id"],
        "scientific_status": report["status"],
        "publication_mode": PUBLICATION_MODE,
        "sha256sums_sha256": sha256(payloads["SHA256SUMS"]),
        "bundle_member_names_excluding_commit_marker": sorted(OUTPUT_NAMES_EXCLUDING_MARKER),
        "bundle_file_count_excluding_commit_marker": len(OUTPUT_NAMES_EXCLUDING_MARKER),
        "final_output_target_sha256": _final_target_sha256(output_directory),
        "committed": True,
        "commit_marker_written_last": True,
        "aggregate_acceptance_requires_exact_marker": True,
    }
    payloads[COMMIT_MARKER] = json_bytes(marker)
    return payloads


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o640,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise PublicationError(f"short write: {path.name}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def inspect_committed_bundle(output_directory: Path | str) -> dict[str, Any]:
    output = Path(output_directory)
    if not output.is_dir() or output.is_symlink():
        raise PartialPublicationError("output is not a non-symlink directory")
    names = set(os.listdir(output))
    expected = set(OUTPUT_NAMES_EXCLUDING_MARKER) | {COMMIT_MARKER}
    if names != expected:
        raise PartialPublicationError("output member closure is incomplete or differs")
    payloads: dict[str, bytes] = {}
    for name in expected:
        path = output / name
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_nlink != 1:
            raise PublicationError(f"published member is unsafe: {name}")
        payloads[name] = path.read_bytes()
    marker = strict_json(payloads[COMMIT_MARKER], label="publication commit marker")
    if marker.get("committed") is not True or marker.get("commit_marker_written_last") is not True:
        raise PartialPublicationError("terminal marker does not establish committed truth")
    if marker.get("publication_mode") != PUBLICATION_MODE:
        raise PublicationError("publication mode differs")
    if marker.get("final_output_target_sha256") != _final_target_sha256(output):
        raise PublicationError("publication target binding differs")
    if marker.get("bundle_member_names_excluding_commit_marker") != sorted(OUTPUT_NAMES_EXCLUDING_MARKER):
        raise PublicationError("publication member declaration differs")
    sums = payloads["SHA256SUMS"]
    if marker.get("sha256sums_sha256") != sha256(sums):
        raise PublicationError("publication checksum-file binding differs")
    declared: dict[str, str] = {}
    try:
        lines = sums.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise PublicationError("SHA256SUMS is not ASCII") from exc
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise PublicationError("malformed SHA256SUMS line")
        digest, name = line[:64], line[66:]
        if HEX64.fullmatch(digest) is None or Path(name).name != name or name in declared:
            raise PublicationError("unsafe SHA256SUMS entry")
        declared[name] = digest
    if set(declared) != set(OUTPUT_JSON_NAMES):
        raise PublicationError("SHA256SUMS declaration set differs")
    for name, digest in declared.items():
        if sha256(payloads[name]) != digest:
            raise PublicationError(f"published member SHA differs: {name}")
    report = strict_json(payloads["ADJUDICATION_REPORT.json"], label="published report")
    if marker.get("scientific_status") != report.get("status"):
        raise PublicationError("marker/report status differs")
    return {
        "publication_status": "COMMITTED_EXACT",
        "scientific_status": report["status"],
        "qualified": report["qualified"],
        "canonical_record_count": report["canonical_record_count"],
        "bundle_file_count": len(expected),
    }


def _publish_bundle(
    output: Path,
    payloads: Mapping[str, bytes],
    *,
    fault_injector: FaultInjector | None = None,
) -> str:
    parent = output.parent
    if not parent.is_dir() or parent.is_symlink():
        raise PublicationError("output parent must be an existing non-symlink directory")
    try:
        os.mkdir(output, 0o750)
    except FileExistsError:
        inspect_committed_bundle(output)
        observed = {name: (output / name).read_bytes() for name in payloads}
        if observed != dict(payloads):
            raise PublicationError("existing committed output differs; overwrite refused")
        return "EXISTING_EXACT"
    try:
        for name in sorted(OUTPUT_NAMES_EXCLUDING_MARKER):
            _write_exclusive(output / name, payloads[name])
            if fault_injector is not None:
                fault_injector(f"after_{name}")
        _fsync_directory(output)
        if fault_injector is not None:
            fault_injector("before_commit_marker")
        _write_exclusive(output / COMMIT_MARKER, payloads[COMMIT_MARKER])
        _fsync_directory(output)
        _fsync_directory(parent)
    except Exception:
        # Preserve partial evidence; a terminal marker is the only commit truth.
        raise
    inspect_committed_bundle(output)
    return "PUBLISHED"


def _preflight_output(output: Path, config: Mapping[str, Any], *, production: bool) -> None:
    if not output.is_absolute() or any(part in {"", ".", ".."} for part in output.parts[1:]):
        raise ScopeViolation("output directory must be an absolute path with safe components")
    lowered = os.fspath(output).casefold()
    severe_tokens = ("gse246381", "/restricted/", "/sealed/", "/sealed_external/", "access_log")
    hits = [token for token in severe_tokens if token in lowered]
    if hits:
        raise ScopeViolation(f"output directory contains forbidden path token(s): {','.join(hits)}")
    if production:
        allowed = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1")
        try:
            common = os.path.commonpath((os.fspath(output), os.fspath(allowed)))
        except ValueError as exc:
            raise ScopeViolation("output root comparison failed") from exc
        if common != os.fspath(allowed) or output == allowed:
            raise ScopeViolation("production output must be a strict descendant of the A1 run root")


def adjudicate(
    config: Mapping[str, Any],
    output_directory: Path | str,
    *,
    production: bool = False,
    fault_injector: FaultInjector | None = None,
) -> dict[str, Any]:
    validate_implementation_binding(config)
    if production:
        validate_production_authority(config)
    output = Path(os.path.abspath(os.fspath(output_directory)))
    _preflight_output(output, config, production=production)

    if not _all_evidence_descriptors_bound(config):
        blockers = [
            slot["blocker_if_unbound"]
            for slot in config["evidence_contract"]["slots"]
            if (
                type(slot["absolute_path"]) is not str
                or slot["absolute_path"] == UNKNOWN
                or HEX64.fullmatch(str(slot["sha256"])) is None
                or type(slot["bytes"]) is not int
                or type(slot["bytes"]) is bool
                or slot["bytes"] <= 0
            )
        ]
        report = _blocked_report(config, blockers)
        audit = _input_audit(config, None)
    else:
        for slot in config["evidence_contract"]["slots"]:
            path = Path(slot["absolute_path"])
            _reject_path(path, config, label=f"evidence {slot['slot_id']}")
            if path.name != slot["allowed_basename"]:
                raise ScopeViolation(f"evidence basename differs: {slot['slot_id']}")
        records: dict[str, dict[str, Any]] = {}
        for slot in config["evidence_contract"]["slots"]:
            records[slot["slot_id"]] = _validate_gate_record(
                _read_verified_evidence(slot, config), slot, config
            )
        blockers, canonical_count, independent_reproduction = _evaluate(records, config)
        report = (
            _blocked_report(config, blockers)
            if blockers
            else _success_report(config, canonical_count, independent_reproduction)
        )
        audit = _input_audit(config, records)

    payloads = _build_bundle(config, output, report, audit)
    publication_status = _publish_bundle(output, payloads, fault_injector=fault_injector)
    return {
        "publication_status": publication_status,
        "status": report["status"],
        "qualified": report["qualified"],
        "ordinary_study_contribution": report["ordinary_study_contribution"],
        "a1_study_contribution": report["a1_study_contribution"],
        "true_a2_study_contribution": report["true_a2_study_contribution"],
        "canonical_record_count": report["canonical_record_count"],
        "independent_raw_reproduction_established": report["independent_raw_reproduction_established"],
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "blockers": report["blockers"],
    }


def load_production_config() -> dict[str, Any]:
    return strict_json(PRODUCTION_CONFIG_PATH.read_bytes(), label="production config")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.inspect:
            result = inspect_committed_bundle(args.output_directory)
        else:
            result = adjudicate(load_production_config(), args.output_directory, production=True)
    except AdjudicationError as exc:
        print(json.dumps({"status": "ERROR", "error_type": type(exc).__name__, "message": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
