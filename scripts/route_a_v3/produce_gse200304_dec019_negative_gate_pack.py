#!/usr/bin/env python3
"""Publish seven honest aggregate negative gate records for DEC-019/GSE200304.

The producer does not inspect a row, sequence, barcode, effect, gene, model
checkpoint, or predecessor payload.  It copies the predecessor and acceptance
authority objects from the exact hash-bound GSE200304 v3 consumer config and
then asks that consumer's own ``_validate_gate_record`` function to accept each
record before publication.

Production is deliberately two-commit bound.  An UNKNOWN implementation
binding stops before the consumer authority is opened and before the output
path is inspected.  Publication writes seven records plus a deterministic
terminal commit marker.  The primary path uses an OS no-replace directory
rename.  On an explicitly approved unsupported-primitive errno, the NFS-safe
fallback atomically creates the final directory and writes the marker last.
Only an exact eight-member directory with an exact marker is published.  An
unmarked partial directory is preserved and can be completed only through the
explicit recovery flag.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import importlib.util
import json
import os
import re
import secrets
import stat
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


CONFIG_REPO_PATH = "configs/route_a_v3_gse200304_dec019_negative_gate_pack_v1.json"
SCRIPT_REPO_PATH = "scripts/route_a_v3/produce_gse200304_dec019_negative_gate_pack.py"
TEST_REPO_PATH = "tests/route_a_v3/test_produce_gse200304_dec019_negative_gate_pack.py"
PRODUCTION_REPO_ROOT = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/routea_v3_a1_20260810"
)
PRODUCTION_CONFIG_PATH = PRODUCTION_REPO_ROOT / CONFIG_REPO_PATH
UNKNOWN = "UNKNOWN_NOT_ASSERTED"
BRANCH = "routea-v3-a1-20260810"
CONTRACT_ID = "mrna_xeditflow_route_a_v3"
DATASET_ID = "GSE200304"
DECISION_ID = "V3-DEC-019"
PROTOCOL_ID = "ROUTE_A_V3_GSE200304_DEC019_NEGATIVE_GATE_PACK_V1"
SCHEMA_VERSION = "route_a_v3_gse200304_dec019_negative_gate_pack.v1"
EVIDENCE_SCHEMA_VERSION = "route_a_v3_dec019_aggregate_gate_evidence.v3"
EVIDENCE_RECORD_TYPE = "ROUTE_A_V3_DEC019_ACCEPTED_AGGREGATE_GATE_EVIDENCE_V3"
ACCEPTANCE_RULE = "CONFIG_HASH_BOUND_ACCEPTED_AGGREGATE_GATE_RECORD_V3"
CONSUMER_PROTOCOL_ID = (
    "ROUTE_A_V3_GSE200304_DEC019_REPORTED_ENDPOINT_A1_ACTIVATION_V3"
)
CONSUMER_SCHEMA_VERSION = (
    "route_a_v3_gse200304_dec019_reported_endpoint_a1_activation.v3"
)
CONSUMER_CONFIG_REPO_PATH = (
    "configs/route_a_v3_gse200304_dec019_reported_endpoint_a1_activation_v3.json"
)
CONSUMER_SCRIPT_REPO_PATH = (
    "scripts/route_a_v3/adjudicate_gse200304_dec019_reported_endpoint_a1_v3.py"
)
CONSUMER_TEST_REPO_PATH = (
    "tests/route_a_v3/test_adjudicate_gse200304_dec019_reported_endpoint_a1_v3.py"
)
CONSUMER_IMPLEMENTATION_PATHS = [
    CONSUMER_CONFIG_REPO_PATH,
    CONSUMER_SCRIPT_REPO_PATH,
    CONSUMER_TEST_REPO_PATH,
]
CONSUMER_I_TO_B_SCALAR_PATHS = {
    "implementation_binding.status",
    "implementation_binding.implementation_commit",
    "implementation_binding.implementation_script_sha256",
    "implementation_binding.implementation_test_sha256",
}
PREDECESSOR_DESCRIPTOR_CONFIG_SHA256 = (
    "955747ffa55cad93c6fbe7950f9ffa89997c5597bdad8add66877e2e1f08b981"
)
REQUIRED_NON_REPOSITORY_SCIENCE_PROJECTION_SHA256 = (
    "aa89c9503a417f1bf7c5edf1a0142252aa6a714d9b9ca74214f9ab72d066691c"
)
REQUIRED_DESCRIPTOR_SET_SHA256 = (
    "079dd5d91df1b6efde42c8277406b16edc99b2ac7181923a529767a8eb97f348"
)
SOURCE_TARGET_SHA256 = (
    "bdb3f54f2aaf0f3f2b090563d712bbebb77ec2ca6de1c8e5f93a126690061a8f"
)
SOURCE_BUNDLE_ID = "ROUTE_A_V3_GSE200304_PUBLISHED_ENDPOINT_A1_BUNDLE_V1"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
REASON_CODE = re.compile(r"^[A-Z][A-Z0-9_]*$")

IMPLEMENTATION_PATHS = [CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH]
BINDING_PATHS = [CONFIG_REPO_PATH]
BINDING_SCALAR_PATHS = [
    "implementation_binding.status",
    "implementation_binding.implementation_commit",
    "implementation_binding.implementation_script_sha256",
    "implementation_binding.implementation_test_sha256",
]
CONSUMER_BINDING_SCALAR_PATHS = [
    "consumer_authority.status",
    "consumer_authority.successor_binding_commit",
    "consumer_authority.config_sha256",
    "consumer_authority.script_sha256",
    "consumer_authority.test_sha256",
    "consumer_authority.science_core_sha256",
]
PRE_I_REBASE_SCALAR_PATHS = {
    "implementation_binding.config_core_sha256",
    "repository_authority.base_commit",
    "repository_authority.implementation_commit_expected_parent",
    *CONSUMER_BINDING_SCALAR_PATHS,
}
PRIVACY = {
    "contains_row_level_payload": False,
    "contains_sequence": False,
    "contains_row_identifier": False,
    "contains_raw_label_or_effect": False,
    "contains_member_identifiers_or_hashes": False,
}
GATE_SPECS = [
    {
        "gate_id": "CANONICAL_REPORTED_ENDPOINT_SEMANTICS",
        "allowed_basename": "GSE200304_DEC019_CANONICAL_REPORTED_ENDPOINT_SEMANTICS_GATE.json",
        "status": "BLOCKED",
        "reason_codes": [
            "AUTHOR_COUNTING_POLICY_AND_PAPER_FAITHFUL_MAPPING_NOT_CLOSED"
        ],
        "unknown_fields": [
            "author_published_processed_endpoint_is_primary",
            "contrast_and_transform_frozen",
            "endpoint_direction_frozen",
            "endpoint_id_frozen",
            "endpoint_scale_frozen",
            "paper_faithful_mapping_closed",
        ],
    },
    {
        "gate_id": "BIOLOGICAL_GROUP_AUTHORITY",
        "allowed_basename": "GSE200304_DEC019_BIOLOGICAL_GROUP_AUTHORITY_GATE.json",
        "status": "BLOCKED",
        "reason_codes": ["BIOLOGICAL_GROUP_AUTHORITY_NOT_CLOSED"],
        "unknown_fields": [
            "biological_group_id_frozen",
            "group_mapping_hash_bound",
            "gse200302_is_subseries_not_independent_study",
            "study_unit_is_gse200304",
        ],
    },
    {
        "gate_id": "ROW_REPLICATE_OR_VALID_SE",
        "allowed_basename": "GSE200304_DEC019_ROW_REPLICATE_OR_VALID_SE_GATE.json",
        "status": "BLOCKED",
        "reason_codes": ["ROW_LEVEL_REPLICATE_OR_VALID_SE_NOT_ESTABLISHED"],
        "unknown_fields": [
            "replicate_count_or_effective_n_frozen",
            "replicate_or_valid_standard_error_present",
            "standard_error_semantics_frozen",
            "technical_uncertainty_not_substituted_for_biological_se",
        ],
    },
    {
        "gate_id": "CHECKPOINT_SPECIFIC_EXPOSURE",
        "allowed_basename": "GSE200304_DEC019_CHECKPOINT_SPECIFIC_EXPOSURE_GATE.json",
        "status": UNKNOWN,
        "reason_codes": ["CHECKPOINT_SPECIFIC_EXPOSURE_UNKNOWN_NOT_ASSERTED"],
        "unknown_fields": [
            "audited_checkpoint_count",
            "checkpoint_artifact_digests_bound",
            "checkpoint_ids_and_revisions_frozen",
            "exact_member_exposure_audit_pass",
            "near_duplicate_exposure_audit_pass",
        ],
    },
    {
        "gate_id": "LICENSE_RIGHTS",
        "allowed_basename": "GSE200304_DEC019_LICENSE_RIGHTS_GATE.json",
        "status": UNKNOWN,
        "reason_codes": ["LICENSE_RIGHTS_UNKNOWN_NOT_ASSERTED"],
        "unknown_fields": [
            "private_canonical_materialization_allowed",
            "qualification_use_allowed",
            "redistribution_scope",
            "rights_source_authority_closed",
        ],
    },
    {
        "gate_id": "OUTCOME_BLIND_SPLIT_LEAKAGE",
        "allowed_basename": "GSE200304_DEC019_OUTCOME_BLIND_SPLIT_LEAKAGE_GATE.json",
        "status": "NOT_RUN",
        "reason_codes": ["OUTCOME_BLIND_SPLIT_LEAKAGE_NOT_RUN"],
        "unknown_fields": [
            "a1_group_graph_frozen",
            "a1_near_duplicate_graph_frozen",
            "a1_source_graph_frozen",
            "final_benchmark_membership_deferred_to_a2",
            "leakage_audit_pass",
            "outcome_blind_assignment",
            "split_salt_hash_bound",
        ],
    },
    {
        "gate_id": "PREFROZEN_POWER_PRECISION",
        "allowed_basename": "GSE200304_DEC019_PREFROZEN_POWER_PRECISION_GATE.json",
        "status": "NOT_RUN",
        "reason_codes": ["PREFROZEN_POWER_PRECISION_NOT_RUN"],
        "unknown_fields": [
            "analysis_unit",
            "bootstrap_unit",
            "full_confidence_interval_width",
            "observed_power",
            "prefrozen_before_model_results",
        ],
    },
]
MEMBER_NAMES = tuple(sorted(spec["allowed_basename"] for spec in GATE_SPECS))
PUBLICATION_COMMIT_FILENAME = "PUBLICATION_COMMIT.json"
PUBLICATION_COMMIT_SCHEMA_VERSION = "1.0.0"
PUBLICATION_COMMIT_RECORD_TYPE = (
    "GSE200304_DEC019_NEGATIVE_GATE_PACK_PUBLICATION_COMMIT_V1"
)
PRIMARY_PUBLICATION_MODE = "ATOMIC_EXCLUSIVE_DIRECTORY_RENAME_NOREPLACE_V1"
FALLBACK_PUBLICATION_MODE = "ATOMIC_MKDIR_TERMINAL_COMMIT_MARKER_V1"
ATOMIC_NOREPLACE_UNSUPPORTED_ERRNO_NAMES = (
    "EINVAL",
    "ENOSYS",
    "ENOTSUP",
    "EOPNOTSUPP",
)
ATOMIC_NOREPLACE_UNSUPPORTED_ERRNOS = frozenset(
    getattr(errno, name)
    for name in ATOMIC_NOREPLACE_UNSUPPORTED_ERRNO_NAMES
    if hasattr(errno, name)
)
PUBLISHED_MEMBER_NAMES = tuple(sorted((*MEMBER_NAMES, PUBLICATION_COMMIT_FILENAME)))
FORBIDDEN_OUTPUT_KEY_TOKENS = {
    "barcode",
    "candidate_id",
    "effect_value",
    "gene",
    "member_hashes",
    "member_identifiers",
    "raw_effect",
    "raw_label",
    "raw_row",
    "row_id",
    "row_ids",
    "sequence",
    "source_id",
    "utr",
}
EXPECTED_TOP_KEYS = {
    "schema_version",
    "protocol_id",
    "contract_id",
    "phase_id",
    "dataset_id",
    "decision_id",
    "implementation_binding",
    "repository_authority",
    "consumer_authority",
    "negative_gate_records",
    "record_policy",
    "output_contract",
}
EXPECTED_BINDING_KEYS = {
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
EXPECTED_REPOSITORY_KEYS = {
    "production_repo_root",
    "branch",
    "base_commit",
    "implementation_commit_expected_parent",
    "binding_commit_expected_parent",
    "implementation_commit_exact_changed_paths",
    "binding_commit_exact_changed_paths",
    "descendant_policy",
}
EXPECTED_DESCENDANT_POLICY_KEYS = {
    "current_head_must_be_clean_and_equal_upstream",
    "current_head_must_descend_from_binding_commit",
    "producer_files_must_not_drift_after_binding",
    "consumer_authority_files_must_match_frozen_hashes",
}
EXPECTED_CONSUMER_KEYS = {
    "binding_scheme",
    "status",
    "blocker_if_unbound",
    "successor_binding_commit",
    "config_path",
    "config_sha256",
    "script_path",
    "script_sha256",
    "test_path",
    "test_sha256",
    "science_core_sha256",
    "predecessor_descriptor_config_sha256",
    "required_non_repository_science_projection_sha256",
    "required_descriptor_set_sha256",
    "schema_version",
    "protocol_id",
    "evidence_schema_version",
    "record_type",
    "source_bundle_id",
    "source_bundle_root_or_target_sha256",
    "successor_binding_commit_must_be_ancestor_of_current_descriptor_base",
    "successor_binding_commit_must_be_exact_consumer_i_to_b",
    "current_descriptor_base_is_repository_base_commit",
    "current_consumer_production_authority_must_validate",
    "authority_objects_must_be_copied_from_exact_consumer_config",
    "consumer_validator_must_accept_every_record",
    "unknown_to_bound_scalar_paths",
}
EXPECTED_GATE_SPEC_KEYS = {
    "gate_id",
    "allowed_basename",
    "status",
    "reason_codes",
    "unknown_fields",
}
EXPECTED_RECORD_POLICY_KEYS = {
    "accepted",
    "aggregate_only",
    "facts",
    "privacy",
    "negative_statuses_only",
    "unknown_fields_must_be_sorted_exact_fact_keys",
    "reason_codes_must_be_nonempty_sorted_unique",
    "unknown_numeric_must_not_be_encoded_as_zero",
    "raw_replay_is_not_a_qualification_blocker",
    "ordinary_study_contribution_delta",
    "a1_study_contribution_delta",
    "true_a2_study_contribution_delta",
    "canonical_record_count_delta",
    "training_allowed",
    "model_selection_allowed",
    "next_phase_authorized",
}
EXPECTED_OUTPUT_KEYS = {
    "trusted_final_directory",
    "primary_publication_mode",
    "fallback_publication_mode",
    "atomic_no_replace_unsupported_errno_names",
    "record_count",
    "exact_member_names",
    "exact_published_member_names",
    "descriptor_binding_scope",
    "descriptor_binder_must_validate_exact8_before_consuming_exact7",
    "terminal_commit_marker_filename",
    "terminal_commit_marker_schema_version",
    "terminal_commit_marker_record_type",
    "terminal_commit_marker_written_last",
    "terminal_commit_marker_is_only_acceptance_point",
    "record_file_fsync_before_marker_required",
    "commit_marker_file_fsync_required",
    "final_directory_fsync_after_marker_required",
    "parent_directory_fsync_after_marker_required",
    "post_fsync_reopen_exact8_required",
    "partial_recovery_mode",
    "partial_recovery_requires_current_euid_owner",
    "partial_without_marker_recoverable_only_if_exact_record_subset",
    "partial_with_marker_is_never_auto_repaired",
    "existing_exact_is_idempotent",
    "overwrite_allowed",
    "partial_temp_directory_is_not_publication",
    "post_rename_failure_must_report_committed_state_explicitly",
    "root_to_leaf_symlink_rejection_required",
    "single_link_regular_file_required",
    "fsync_files_directory_and_parent_required",
    "forbidden_output_key_tokens",
}


class ProducerError(RuntimeError):
    """Base class for authority, schema, and publication failures."""


class BindingError(ProducerError):
    """Implementation binding is absent or does not match the I/B lifecycle."""


class AuthorityError(ProducerError):
    """Git or consumer authority is not the frozen production authority."""


class ScopeViolation(ProducerError):
    """A filesystem path left the exact configured scope."""


class PublicationError(ProducerError):
    """Atomic publication or exact-idempotence validation failed."""


class PartialPublicationError(PublicationError):
    """An existing target is partial/different, or an unpublished temp remains."""


class PublicationStateError(PublicationError):
    """A post-rename fault occurred; ``publication_state`` states known truth."""

    def __init__(self, message: str, *, publication_state: str) -> None:
        super().__init__(message)
        self.publication_state = publication_state


class AtomicNoReplaceUnsupported(PublicationError):
    """The kernel/filesystem rejected the frozen no-replace primitive."""

    def __init__(self, error_number: int) -> None:
        super().__init__(
            "atomic no-replace directory rename is unsupported "
            f"(errno={error_number})"
        )
        self.error_number = error_number


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
            raise ProducerError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ProducerError(f"non-finite JSON constant in {label}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProducerError(f"invalid JSON: {label}") from exc
    if type(value) is not dict:
        raise ProducerError(f"JSON root is not an object: {label}")
    return value


def _expect_exact_keys(value: Any, keys: set[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ProducerError(f"{label} keys differ from the closed schema")
    return value


def _expect(value: Any, expected: Any, *, label: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise ProducerError(f"{label} differs")


def config_core_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    projected = copy.deepcopy(dict(config))
    projected.pop("implementation_binding", None)
    return projected


def config_core_sha256(config: Mapping[str, Any]) -> str:
    return sha256(json_bytes(config_core_projection(config)))


def consumer_non_repository_science_projection(
    consumer_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the consumer science/config surface shared by old and successor.

    The consumer implementation lifecycle and repository ancestry are expected
    to change in its successor.  Everything else, including V3 record identity,
    predecessor/acceptance authority, gate semantics, and descriptor values,
    remains part of this frozen equality projection.
    """

    projected = copy.deepcopy(dict(consumer_config))
    projected.pop("implementation_binding", None)
    projected.pop("repository_authority", None)
    return projected


def consumer_non_repository_science_projection_sha256(
    consumer_config: Mapping[str, Any],
) -> str:
    return sha256(json_bytes(consumer_non_repository_science_projection(consumer_config)))


def _consumer_v3_identity(consumer_config: Mapping[str, Any]) -> dict[str, Any]:
    evidence = consumer_config.get("evidence_contract")
    if type(evidence) is not dict:
        raise AuthorityError("consumer evidence contract is absent")
    provenance = evidence.get("gate_record_provenance_contract")
    if type(provenance) is not dict:
        raise AuthorityError("consumer provenance contract is absent")
    acceptance = provenance.get("acceptance_authority")
    if type(acceptance) is not dict:
        raise AuthorityError("consumer acceptance authority is absent")
    return {
        "schema_version": consumer_config.get("schema_version"),
        "protocol_id": consumer_config.get("protocol_id"),
        "contract_id": consumer_config.get("contract_id"),
        "phase_id": consumer_config.get("phase_id"),
        "dataset_id": consumer_config.get("dataset_id"),
        "decision_id": consumer_config.get("decision_id"),
        "evidence_schema_version": evidence.get("evidence_schema_version"),
        "acceptance_rule": acceptance.get("rule"),
    }


def validate_consumer_science_continuity(
    predecessor_config: Mapping[str, Any],
    successor_config: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> None:
    """Prove the successor changed no non-repository consumer science."""

    expected_identity = {
        "schema_version": CONSUMER_SCHEMA_VERSION,
        "protocol_id": CONSUMER_PROTOCOL_ID,
        "contract_id": CONTRACT_ID,
        "phase_id": "A1",
        "dataset_id": DATASET_ID,
        "decision_id": DECISION_ID,
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "acceptance_rule": ACCEPTANCE_RULE,
    }
    predecessor_projection = consumer_non_repository_science_projection(
        predecessor_config
    )
    successor_projection = consumer_non_repository_science_projection(successor_config)
    _expect(
        successor_projection,
        predecessor_projection,
        label="consumer predecessor/successor non-repository science projection",
    )
    for label, value in (
        ("predecessor", predecessor_config),
        ("successor", successor_config),
    ):
        _expect(
            consumer_non_repository_science_projection_sha256(value),
            authority["required_non_repository_science_projection_sha256"],
            label=f"consumer {label} non-repository science projection SHA",
        )
        descriptors = value.get("evidence_descriptor_bindings")
        if type(descriptors) is not dict:
            raise AuthorityError(f"consumer {label} descriptor bindings are absent")
        _expect(
            descriptors.get("descriptor_set_sha256"),
            authority["required_descriptor_set_sha256"],
            label=f"consumer {label} descriptor-set SHA",
        )
        _expect(
            _consumer_v3_identity(value),
            expected_identity,
            label=f"consumer {label} V3 identity",
        )


def _scalar_differences(before: Any, after: Any, prefix: str = "") -> set[str]:
    if type(before) is not type(after):
        return {prefix}
    if isinstance(before, dict):
        if set(before) != set(after):
            return {prefix}
        result: set[str] = set()
        for key in before:
            child = f"{prefix}.{key}" if prefix else key
            result.update(_scalar_differences(before[key], after[key], child))
        return result
    if isinstance(before, list):
        if len(before) != len(after):
            return {prefix}
        result = set()
        for index, (old, new) in enumerate(zip(before, after)):
            result.update(_scalar_differences(old, new, f"{prefix}[{index}]"))
        return result
    return set() if before == after else {prefix}


def validate_static_config(config: Mapping[str, Any]) -> None:
    value = _expect_exact_keys(config, EXPECTED_TOP_KEYS, label="producer config")
    for key, expected in {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "contract_id": CONTRACT_ID,
        "phase_id": "A1",
        "dataset_id": DATASET_ID,
        "decision_id": DECISION_ID,
    }.items():
        _expect(value[key], expected, label=key)

    binding = _expect_exact_keys(
        value["implementation_binding"],
        EXPECTED_BINDING_KEYS,
        label="implementation binding",
    )
    _expect(
        binding["binding_scheme"],
        "CONFIG_ONLY_POST_IMPLEMENTATION_BINDING_V1",
        label="binding scheme",
    )
    _expect(
        binding["blocker_if_unbound"],
        "NEGATIVE_GATE_PACK_IMPLEMENTATION_BINDING_UNKNOWN_NOT_ASSERTED",
        label="unbound blocker",
    )
    _expect(binding["implementation_script_path"], SCRIPT_REPO_PATH, label="script path")
    _expect(binding["implementation_test_path"], TEST_REPO_PATH, label="test path")
    _expect(
        binding["unknown_to_bound_scalar_paths"],
        BINDING_SCALAR_PATHS,
        label="binding scalar paths",
    )
    stored_core_sha = binding["config_core_sha256"]
    if HEX64.fullmatch(str(stored_core_sha)) is None:
        raise BindingError("stored config core SHA is invalid")
    _expect(config_core_sha256(value), stored_core_sha, label="computed config core SHA")
    status = binding["status"]
    if status == UNKNOWN:
        for key in {
            "implementation_commit",
            "implementation_script_sha256",
            "implementation_test_sha256",
        }:
            _expect(binding[key], UNKNOWN, label=f"unbound {key}")
    elif status == "BOUND":
        if HEX40.fullmatch(str(binding["implementation_commit"])) is None:
            raise BindingError("bound implementation commit is invalid")
        for key in {"implementation_script_sha256", "implementation_test_sha256"}:
            if HEX64.fullmatch(str(binding[key])) is None:
                raise BindingError(f"bound {key} is invalid")
    else:
        raise BindingError("implementation binding status is outside the closed enum")

    repository = _expect_exact_keys(
        value["repository_authority"],
        EXPECTED_REPOSITORY_KEYS,
        label="repository authority",
    )
    _expect(repository["production_repo_root"], os.fspath(PRODUCTION_REPO_ROOT), label="repo root")
    _expect(repository["branch"], BRANCH, label="branch")
    if HEX40.fullmatch(str(repository["base_commit"])) is None:
        raise AuthorityError("base commit is invalid")
    _expect(
        repository["implementation_commit_expected_parent"],
        repository["base_commit"],
        label="I expected parent",
    )
    _expect(
        repository["binding_commit_expected_parent"],
        "IMPLEMENTATION_COMMIT_FROM_BINDING",
        label="B expected parent",
    )
    _expect(
        repository["implementation_commit_exact_changed_paths"],
        IMPLEMENTATION_PATHS,
        label="I changed paths",
    )
    _expect(
        repository["binding_commit_exact_changed_paths"],
        BINDING_PATHS,
        label="B changed paths",
    )
    descendant_policy = _expect_exact_keys(
        repository["descendant_policy"],
        EXPECTED_DESCENDANT_POLICY_KEYS,
        label="descendant policy",
    )
    if any(item is not True for item in descendant_policy.values()):
        raise AuthorityError("every descendant policy invariant must be true")

    consumer = _expect_exact_keys(
        value["consumer_authority"],
        EXPECTED_CONSUMER_KEYS,
        label="consumer authority",
    )
    for key, expected in {
        "binding_scheme": "FINAL_CONSUMER_SUCCESSOR_AND_CURRENT_DESCRIPTOR_BASE_BINDING_V1",
        "blocker_if_unbound": "FINAL_CONSUMER_SUCCESSOR_AUTHORITY_UNKNOWN_NOT_ASSERTED",
        "config_path": CONSUMER_CONFIG_REPO_PATH,
        "script_path": CONSUMER_SCRIPT_REPO_PATH,
        "test_path": CONSUMER_TEST_REPO_PATH,
        "predecessor_descriptor_config_sha256": PREDECESSOR_DESCRIPTOR_CONFIG_SHA256,
        "required_non_repository_science_projection_sha256": (
            REQUIRED_NON_REPOSITORY_SCIENCE_PROJECTION_SHA256
        ),
        "required_descriptor_set_sha256": REQUIRED_DESCRIPTOR_SET_SHA256,
        "schema_version": CONSUMER_SCHEMA_VERSION,
        "protocol_id": CONSUMER_PROTOCOL_ID,
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "record_type": EVIDENCE_RECORD_TYPE,
        "source_bundle_id": SOURCE_BUNDLE_ID,
        "source_bundle_root_or_target_sha256": SOURCE_TARGET_SHA256,
    }.items():
        _expect(consumer[key], expected, label=f"consumer {key}")
    _expect(
        consumer["unknown_to_bound_scalar_paths"],
        CONSUMER_BINDING_SCALAR_PATHS,
        label="consumer binding scalar paths",
    )
    if consumer["status"] == UNKNOWN:
        for key in {
            "successor_binding_commit",
            "config_sha256",
            "script_sha256",
            "test_sha256",
            "science_core_sha256",
        }:
            _expect(consumer[key], UNKNOWN, label=f"unbound consumer {key}")
    elif consumer["status"] == "BOUND":
        if HEX40.fullmatch(str(consumer["successor_binding_commit"])) is None:
            raise AuthorityError("consumer successor B commit is invalid")
        for key in {
            "config_sha256",
            "script_sha256",
            "test_sha256",
            "science_core_sha256",
        }:
            if HEX64.fullmatch(str(consumer[key])) is None:
                raise AuthorityError(f"bound consumer {key} is invalid")
    else:
        raise AuthorityError("consumer authority status is outside the closed enum")
    _expect(
        consumer["successor_binding_commit_must_be_ancestor_of_current_descriptor_base"],
        True,
        label="consumer successor ancestry policy",
    )
    _expect(
        consumer["successor_binding_commit_must_be_exact_consumer_i_to_b"],
        True,
        label="consumer exact I-to-B policy",
    )
    _expect(
        consumer["current_descriptor_base_is_repository_base_commit"],
        True,
        label="current descriptor base policy",
    )
    _expect(
        consumer["current_consumer_production_authority_must_validate"],
        True,
        label="current consumer production-authority policy",
    )
    _expect(consumer["authority_objects_must_be_copied_from_exact_consumer_config"], True, label="authority copy policy")
    _expect(consumer["consumer_validator_must_accept_every_record"], True, label="consumer validation policy")

    negative_gate_records = value["negative_gate_records"]
    if type(negative_gate_records) is not list:
        raise ProducerError("negative gate records are not a list")
    for index, spec in enumerate(negative_gate_records):
        _expect_exact_keys(
            spec,
            EXPECTED_GATE_SPEC_KEYS,
            label=f"negative gate spec {index}",
        )
    _expect(negative_gate_records, GATE_SPECS, label="negative gate specs")
    if any(spec["gate_id"] == "CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE" for spec in GATE_SPECS):
        raise ProducerError("lineage gate is outside this seven-record pack")
    for spec in GATE_SPECS:
        if spec["status"] not in {UNKNOWN, "NOT_RUN", "BLOCKED"}:
            raise ProducerError("negative gate pack contains a non-negative status")
        if spec["unknown_fields"] != sorted(set(spec["unknown_fields"])):
            raise ProducerError("unknown_fields are not sorted/unique")
        reasons = spec["reason_codes"]
        if (
            reasons != sorted(set(reasons))
            or not reasons
            or any(type(reason) is not str or REASON_CODE.fullmatch(reason) is None for reason in reasons)
        ):
            raise ProducerError("reason_codes are not nonempty/sorted/unique")

    policy = _expect_exact_keys(
        value["record_policy"],
        EXPECTED_RECORD_POLICY_KEYS,
        label="record policy",
    )
    for key, expected in {
        "accepted": True,
        "aggregate_only": True,
        "facts": None,
        "privacy": PRIVACY,
        "negative_statuses_only": True,
        "unknown_fields_must_be_sorted_exact_fact_keys": True,
        "reason_codes_must_be_nonempty_sorted_unique": True,
        "unknown_numeric_must_not_be_encoded_as_zero": True,
        "raw_replay_is_not_a_qualification_blocker": True,
        "ordinary_study_contribution_delta": 0,
        "a1_study_contribution_delta": 0,
        "true_a2_study_contribution_delta": 0,
        "canonical_record_count_delta": 0,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
    }.items():
        _expect(policy[key], expected, label=f"record policy {key}")

    output = _expect_exact_keys(
        value["output_contract"],
        EXPECTED_OUTPUT_KEYS,
        label="output contract",
    )
    _expect(
        output["primary_publication_mode"],
        PRIMARY_PUBLICATION_MODE,
        label="primary publication mode",
    )
    _expect(
        output["fallback_publication_mode"],
        FALLBACK_PUBLICATION_MODE,
        label="fallback publication mode",
    )
    _expect(
        output["atomic_no_replace_unsupported_errno_names"],
        list(ATOMIC_NOREPLACE_UNSUPPORTED_ERRNO_NAMES),
        label="atomic no-replace unsupported errno names",
    )
    _expect(output["record_count"], 7, label="record count")
    _expect(output["exact_member_names"], list(MEMBER_NAMES), label="exact output names")
    _expect(
        output["exact_published_member_names"],
        list(PUBLISHED_MEMBER_NAMES),
        label="exact published output names",
    )
    _expect(output["descriptor_binding_scope"], "SEVEN_GATE_JSON_FILES_ONLY", label="descriptor scope")
    for key, expected in {
        "terminal_commit_marker_filename": PUBLICATION_COMMIT_FILENAME,
        "terminal_commit_marker_schema_version": PUBLICATION_COMMIT_SCHEMA_VERSION,
        "terminal_commit_marker_record_type": PUBLICATION_COMMIT_RECORD_TYPE,
        "partial_recovery_mode": "EXPLICIT_CLI_FLAG_ONLY",
    }.items():
        _expect(output[key], expected, label=f"output contract {key}")
    _expect(set(output["forbidden_output_key_tokens"]), FORBIDDEN_OUTPUT_KEY_TOKENS, label="forbidden output keys")
    for key, expected in {
        "terminal_commit_marker_written_last": True,
        "terminal_commit_marker_is_only_acceptance_point": True,
        "descriptor_binder_must_validate_exact8_before_consuming_exact7": True,
        "record_file_fsync_before_marker_required": True,
        "commit_marker_file_fsync_required": True,
        "final_directory_fsync_after_marker_required": True,
        "parent_directory_fsync_after_marker_required": True,
        "post_fsync_reopen_exact8_required": True,
        "partial_recovery_requires_current_euid_owner": True,
        "partial_without_marker_recoverable_only_if_exact_record_subset": True,
        "partial_with_marker_is_never_auto_repaired": True,
        "existing_exact_is_idempotent": True,
        "overwrite_allowed": False,
        "partial_temp_directory_is_not_publication": True,
        "post_rename_failure_must_report_committed_state_explicitly": True,
        "root_to_leaf_symlink_rejection_required": True,
        "single_link_regular_file_required": True,
        "fsync_files_directory_and_parent_required": True,
    }.items():
        _expect(output[key], expected, label=f"output contract {key}")


def validate_implementation_binding(config: Mapping[str, Any]) -> None:
    binding = config["implementation_binding"]
    if binding["status"] != "BOUND":
        raise BindingError(binding["blocker_if_unbound"])
    if HEX40.fullmatch(str(binding["implementation_commit"])) is None:
        raise BindingError("implementation commit is not bound")
    for key in {"implementation_script_sha256", "implementation_test_sha256"}:
        if HEX64.fullmatch(str(binding[key])) is None:
            raise BindingError(f"{key} is not bound")


def validate_consumer_authority_binding(config: Mapping[str, Any]) -> None:
    consumer = config["consumer_authority"]
    if consumer["status"] != "BOUND":
        raise BindingError(consumer["blocker_if_unbound"])
    if HEX40.fullmatch(str(consumer["successor_binding_commit"])) is None:
        raise BindingError("consumer successor B commit is not bound")
    for key in {
        "config_sha256",
        "script_sha256",
        "test_sha256",
        "science_core_sha256",
    }:
        if HEX64.fullmatch(str(consumer[key])) is None:
            raise BindingError(f"consumer {key} is not bound")


def validate_pre_i_consumer_rebase(
    unbound_config: Mapping[str, Any],
    rebound_config: Mapping[str, Any],
    *,
    current_descriptor_base_commit: str,
    successor_binding_commit: str,
    consumer_config_sha256: str,
    consumer_script_sha256: str,
    consumer_test_sha256: str,
    consumer_science_core_sha256: str,
) -> None:
    """Validate the one pre-I rebase once the final consumer B/base exist."""

    validate_static_config(unbound_config)
    validate_static_config(rebound_config)
    if unbound_config["implementation_binding"]["status"] != UNKNOWN:
        raise BindingError("pre-I source config is not producer-UNKNOWN")
    if rebound_config["implementation_binding"]["status"] != UNKNOWN:
        raise BindingError("pre-I rebound config prematurely bound the producer")
    if unbound_config["consumer_authority"]["status"] != UNKNOWN:
        raise BindingError("pre-I source config is not consumer-UNKNOWN")
    validate_consumer_authority_binding(rebound_config)
    differences = _scalar_differences(unbound_config, rebound_config)
    if differences != PRE_I_REBASE_SCALAR_PATHS:
        raise BindingError("pre-I consumer/base rebind changed fields outside the closed set")
    repository = rebound_config["repository_authority"]
    _expect(
        repository["base_commit"],
        current_descriptor_base_commit,
        label="rebound descriptor base",
    )
    _expect(
        repository["implementation_commit_expected_parent"],
        current_descriptor_base_commit,
        label="rebound I parent",
    )
    consumer = rebound_config["consumer_authority"]
    for key, expected in {
        "successor_binding_commit": successor_binding_commit,
        "config_sha256": consumer_config_sha256,
        "script_sha256": consumer_script_sha256,
        "test_sha256": consumer_test_sha256,
        "science_core_sha256": consumer_science_core_sha256,
    }.items():
        _expect(consumer[key], expected, label=f"rebound consumer {key}")
    _expect(
        rebound_config["implementation_binding"]["config_core_sha256"],
        config_core_sha256(rebound_config),
        label="rebound config core",
    )


def validate_i_to_b_transition(
    i_config: Mapping[str, Any],
    b_config: Mapping[str, Any],
    *,
    implementation_commit: str,
    implementation_script_sha256: str,
    implementation_test_sha256: str,
) -> None:
    validate_static_config(i_config)
    validate_static_config(b_config)
    if i_config["implementation_binding"]["status"] != UNKNOWN:
        raise BindingError("I config is not exactly UNKNOWN-bound")
    validate_consumer_authority_binding(i_config)
    if b_config["implementation_binding"]["status"] != "BOUND":
        raise BindingError("B config is not BOUND")
    validate_consumer_authority_binding(b_config)
    differences = _scalar_differences(i_config, b_config)
    if differences != set(BINDING_SCALAR_PATHS):
        raise BindingError("I-to-B config differences are not the exact four scalars")
    binding = b_config["implementation_binding"]
    expected = {
        "implementation_commit": implementation_commit,
        "implementation_script_sha256": implementation_script_sha256,
        "implementation_test_sha256": implementation_test_sha256,
    }
    for key, value in expected.items():
        _expect(binding[key], value, label=f"B {key}")
    _expect(
        config_core_sha256(i_config),
        config_core_sha256(b_config),
        label="I/B science-independent core",
    )


def _git(repo: Path, *arguments: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", os.fspath(repo), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and result.returncode != 0:
        raise AuthorityError(
            f"git {' '.join(arguments)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _commit_paths(repo: Path, commit: str) -> list[str]:
    output = _git(
        repo,
        "diff-tree",
        "--root",
        "--no-commit-id",
        "--name-only",
        "-r",
        commit,
    )
    return sorted(line for line in output.splitlines() if line)


def _git_object(repo: Path, revision_and_path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", os.fspath(repo), "show", revision_and_path],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise AuthorityError(
            f"git show failed for {revision_and_path}: "
            f"{result.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return result.stdout


def _git_regular_blob(repo: Path, commit: str, path: str, *, label: str) -> bytes:
    """Return one exact Git blob only when its tree entry is a regular file."""

    result = subprocess.run(
        [
            "git",
            "-C",
            os.fspath(repo),
            "ls-tree",
            "-z",
            "--full-tree",
            commit,
            "--",
            path,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise AuthorityError(
            f"git ls-tree failed for {label}: "
            f"{result.stderr.decode('utf-8', errors='replace').strip()}"
        )
    entries = result.stdout.split(b"\0")
    if len(entries) != 2 or entries[1] != b"" or b"\t" not in entries[0]:
        raise AuthorityError(f"{label} is not exactly one Git tree entry")
    metadata, observed_path = entries[0].split(b"\t", 1)
    fields = metadata.split()
    try:
        observed = observed_path.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise AuthorityError(f"{label} Git path is not UTF-8") from exc
    if (
        len(fields) != 3
        or fields[0] not in {b"100644", b"100755"}
        or fields[1] != b"blob"
        or observed != path
    ):
        raise AuthorityError(f"{label} is not a regular Git blob at the exact path")
    return _git_object(repo, f"{commit}:{path}")


def _is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "-C", os.fspath(repo), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _find_frozen_predecessor_consumer_config(
    repo: Path,
    *,
    successor_binding_commit: str,
    expected_sha256: str,
) -> dict[str, Any]:
    """Find the nearest exact frozen descriptor config before successor B."""

    history = _git(
        repo,
        "rev-list",
        "--first-parent",
        f"{successor_binding_commit}^",
        "--",
        CONSUMER_CONFIG_REPO_PATH,
    )
    for commit in (line for line in history.splitlines() if line):
        payload = _git_regular_blob(
            repo,
            commit,
            CONSUMER_CONFIG_REPO_PATH,
            label="predecessor consumer descriptor config",
        )
        if sha256(payload) == expected_sha256:
            return strict_json(payload, label="predecessor consumer descriptor config")
    raise AuthorityError("frozen predecessor consumer descriptor config was not found")


def _single_commit_parent(repo: Path, commit: str, *, label: str) -> str:
    parents = _git(repo, "show", "-s", "--format=%P", commit).split()
    if len(parents) != 1:
        raise AuthorityError(f"{label} is not a single-parent commit")
    return parents[0]


def validate_consumer_successor_lifecycle(
    *,
    repo: Path,
    successor_binding_commit: str,
    current_head: str,
    current_consumer_config: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> dict[str, str]:
    """Prove the claimed consumer B is an exact three-file-I/config-only-B."""

    consumer_i = _single_commit_parent(
        repo,
        successor_binding_commit,
        label="consumer successor B",
    )
    b_config = strict_json(
        _git_regular_blob(
            repo,
            successor_binding_commit,
            CONSUMER_CONFIG_REPO_PATH,
            label="consumer successor-B config",
        ),
        label="consumer successor-B config",
    )
    b_binding = b_config.get("implementation_binding")
    if type(b_binding) is not dict:
        raise AuthorityError("consumer successor-B implementation binding is absent")
    _expect(
        b_binding.get("implementation_commit"),
        consumer_i,
        label="consumer successor-B implementation commit",
    )
    _expect(b_binding.get("status"), "BOUND", label="consumer successor-B status")
    for key, expected in {
        "implementation_script_sha256": authority["script_sha256"],
        "implementation_test_sha256": authority["test_sha256"],
        "config_core_sha256": authority["science_core_sha256"],
    }.items():
        _expect(b_binding.get(key), expected, label=f"consumer successor-B {key}")

    repository = b_config.get("repository_authority")
    if type(repository) is not dict:
        raise AuthorityError("consumer successor-B repository authority is absent")
    consumer_base = repository.get("base_commit")
    if HEX40.fullmatch(str(consumer_base)) is None:
        raise AuthorityError("consumer successor-I base commit is invalid")
    _expect(
        repository.get("implementation_commit_expected_parent"),
        consumer_base,
        label="consumer successor-I expected parent",
    )
    _expect(
        repository.get("implementation_commit_exact_changed_paths"),
        CONSUMER_IMPLEMENTATION_PATHS,
        label="consumer successor-I changed-path contract",
    )
    _expect(
        repository.get("binding_commit_exact_changed_paths"),
        [CONSUMER_CONFIG_REPO_PATH],
        label="consumer successor-B changed-path contract",
    )
    _expect(
        _single_commit_parent(repo, consumer_i, label="consumer successor I"),
        consumer_base,
        label="consumer successor-I direct parent",
    )
    if _commit_paths(repo, consumer_i) != CONSUMER_IMPLEMENTATION_PATHS:
        raise AuthorityError("consumer successor I is not the exact three-file commit")
    if _commit_paths(repo, successor_binding_commit) != [CONSUMER_CONFIG_REPO_PATH]:
        raise AuthorityError("consumer successor B is not the exact config-only commit")

    i_config = strict_json(
        _git_regular_blob(
            repo,
            consumer_i,
            CONSUMER_CONFIG_REPO_PATH,
            label="consumer successor-I config",
        ),
        label="consumer successor-I config",
    )
    i_binding = i_config.get("implementation_binding")
    if type(i_binding) is not dict or set(i_binding) != set(b_binding):
        raise AuthorityError("consumer successor I/B binding schema differs")
    _expect(i_binding.get("status"), UNKNOWN, label="consumer successor-I status")
    for key in {
        "implementation_commit",
        "implementation_script_sha256",
        "implementation_test_sha256",
    }:
        _expect(i_binding.get(key), UNKNOWN, label=f"consumer successor-I {key}")
    _expect(
        i_binding.get("config_core_sha256"),
        authority["science_core_sha256"],
        label="consumer successor-I science core",
    )
    if _scalar_differences(i_config, b_config) != CONSUMER_I_TO_B_SCALAR_PATHS:
        raise AuthorityError("consumer I-to-B diff is not the exact four-scalar binding")

    for path, expected_sha in {
        CONSUMER_SCRIPT_REPO_PATH: authority["script_sha256"],
        CONSUMER_TEST_REPO_PATH: authority["test_sha256"],
    }.items():
        payload_at_i = _git_regular_blob(
            repo,
            consumer_i,
            path,
            label=f"consumer implementation at I: {path}",
        )
        if sha256(payload_at_i) != expected_sha:
            raise AuthorityError(f"consumer implementation hash at I differs: {path}")
        payload_at_head = _git_regular_blob(
            repo,
            current_head,
            path,
            label=f"consumer implementation at HEAD: {path}",
        )
        if payload_at_head != payload_at_i:
            raise AuthorityError(f"consumer implementation drifted after I: {path}")

    current_binding = current_consumer_config.get("implementation_binding")
    if current_binding != b_binding or type(current_binding) is not dict:
        raise AuthorityError("current consumer binding differs from successor B")
    return {
        "consumer_base_commit": str(consumer_base),
        "consumer_implementation_commit": consumer_i,
        "consumer_binding_commit": successor_binding_commit,
    }


def validate_production_authority(
    config: Mapping[str, Any],
    *,
    repo: Path = PRODUCTION_REPO_ROOT,
) -> dict[str, str]:
    """Validate clean, pushed, no-drift Git authority after binding."""

    validate_implementation_binding(config)
    validate_consumer_authority_binding(config)
    expected_repo = Path(config["repository_authority"]["production_repo_root"])
    if repo.resolve() != expected_repo:
        raise AuthorityError("production repository root differs")
    head = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "branch", "--show-current") != BRANCH:
        raise AuthorityError("production branch differs")
    if _git(repo, "status", "--porcelain"):
        raise AuthorityError("production worktree is not clean")
    upstream = _git(repo, "rev-parse", "@{upstream}")
    if head != upstream:
        raise AuthorityError("production HEAD is not exactly pushed upstream")

    binding = config["implementation_binding"]
    implementation = binding["implementation_commit"]
    base_commit = config["repository_authority"]["base_commit"]
    consumer = config["consumer_authority"]
    consumer_binding_commit = consumer["successor_binding_commit"]
    if not _is_ancestor(repo, consumer_binding_commit, base_commit):
        raise AuthorityError("consumer successor B is not an ancestor of descriptor base")
    if not _is_ancestor(repo, consumer_binding_commit, head):
        raise AuthorityError("consumer successor B is not an ancestor of current HEAD")
    if _commit_paths(repo, consumer_binding_commit) != [CONSUMER_CONFIG_REPO_PATH]:
        raise AuthorityError("consumer successor B is not config-only")
    parents = _git(repo, "show", "-s", "--format=%P", implementation).split()
    if parents != [base_commit]:
        raise AuthorityError("producer I is not the direct child of frozen base")
    if _commit_paths(repo, implementation) != sorted(IMPLEMENTATION_PATHS):
        raise AuthorityError("producer I changed paths differ")
    if not _is_ancestor(repo, implementation, head):
        raise AuthorityError("producer I is not an ancestor of current HEAD")

    i_config = strict_json(
        _git_object(repo, f"{implementation}:{CONFIG_REPO_PATH}"),
        label="producer I config",
    )
    script_at_i = _git_regular_blob(
        repo,
        implementation,
        SCRIPT_REPO_PATH,
        label="producer script at I",
    )
    test_at_i = _git_regular_blob(
        repo,
        implementation,
        TEST_REPO_PATH,
        label="producer test at I",
    )
    if sha256(script_at_i) != binding["implementation_script_sha256"]:
        raise AuthorityError("producer script hash at I differs")
    if sha256(test_at_i) != binding["implementation_test_sha256"]:
        raise AuthorityError("producer test hash at I differs")

    descendants = _git(repo, "rev-list", "--ancestry-path", f"{implementation}..{head}")
    candidates: list[tuple[str, dict[str, Any]]] = []
    for commit in (line for line in descendants.splitlines() if line):
        commit_parents = _git(repo, "show", "-s", "--format=%P", commit).split()
        if commit_parents != [implementation]:
            continue
        if _commit_paths(repo, commit) != BINDING_PATHS:
            continue
        candidate = strict_json(
            _git_object(repo, f"{commit}:{CONFIG_REPO_PATH}"),
            label="producer B config",
        )
        try:
            validate_i_to_b_transition(
                i_config,
                candidate,
                implementation_commit=implementation,
                implementation_script_sha256=sha256(script_at_i),
                implementation_test_sha256=sha256(test_at_i),
            )
        except ProducerError:
            continue
        candidates.append((commit, candidate))
    if len(candidates) != 1:
        raise AuthorityError("exactly one direct config-only B commit was not found")
    binding_commit, b_config = candidates[0]
    if not _is_ancestor(repo, binding_commit, head):
        raise AuthorityError("producer B is not an ancestor of current HEAD")
    if dict(config) != b_config:
        raise AuthorityError("producer config drifted after B")

    for path, expected_sha in {
        SCRIPT_REPO_PATH: binding["implementation_script_sha256"],
        TEST_REPO_PATH: binding["implementation_test_sha256"],
    }.items():
        git_payload = _git_regular_blob(
            repo,
            head,
            path,
            label=f"producer file at HEAD: {path}",
        )
        if sha256(git_payload) != expected_sha:
            raise AuthorityError(f"producer file drifted after B: {path}")
        worktree_payload = _read_repo_regular(
            repo,
            path,
            label=f"producer worktree file: {path}",
        )
        if sha256(worktree_payload) != expected_sha:
            raise AuthorityError(f"producer worktree file differs: {path}")
    consumer_head_payloads: dict[str, bytes] = {}
    for path_key, sha_key in {
        "config_path": "config_sha256",
        "script_path": "script_sha256",
        "test_path": "test_sha256",
    }.items():
        path = consumer[path_key]
        git_payload = _git_regular_blob(
            repo,
            head,
            path,
            label=f"consumer authority at HEAD: {path}",
        )
        if sha256(git_payload) != consumer[sha_key]:
            raise AuthorityError(f"consumer authority drifted: {path}")
        worktree_payload = _read_repo_regular(
            repo,
            path,
            label=f"consumer worktree authority: {path}",
        )
        if sha256(worktree_payload) != consumer[sha_key]:
            raise AuthorityError(f"consumer worktree authority differs: {path}")
        consumer_head_payloads[path_key] = git_payload
    current_consumer_config = strict_json(
        consumer_head_payloads["config_path"],
        label="current consumer config at HEAD",
    )
    consumer_lifecycle = validate_consumer_successor_lifecycle(
        repo=repo,
        successor_binding_commit=consumer_binding_commit,
        current_head=head,
        current_consumer_config=current_consumer_config,
        authority=consumer,
    )
    predecessor_consumer_config = _find_frozen_predecessor_consumer_config(
        repo,
        successor_binding_commit=consumer_binding_commit,
        expected_sha256=consumer["predecessor_descriptor_config_sha256"],
    )
    validate_consumer_science_continuity(
        predecessor_consumer_config,
        current_consumer_config,
        consumer,
    )
    return {
        "base_commit": base_commit,
        "consumer_successor_binding_commit": consumer_binding_commit,
        "consumer_successor_implementation_commit": consumer_lifecycle[
            "consumer_implementation_commit"
        ],
        "consumer_successor_base_commit": consumer_lifecycle[
            "consumer_base_commit"
        ],
        "implementation_commit": implementation,
        "binding_commit": binding_commit,
        "current_head": head,
        "upstream_head": upstream,
    }


def _safe_relative_path(value: str, *, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ScopeViolation(f"{label} is not a safe repository-relative path")
    return path


def _open_directory_root_to_leaf(path: Path, *, label: str) -> int:
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise ScopeViolation(f"{label} must be an absolute path with safe components")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ScopeViolation("O_NOFOLLOW is unavailable")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | nofollow
    )
    descriptor = os.open(os.sep, flags)
    try:
        for component in path.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                raise ScopeViolation(
                    f"{label} contains a symlink, missing component, or non-directory"
                ) from exc
            os.close(descriptor)
            descriptor = child
        result = descriptor
        descriptor = -1
        return result
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_regular_at(directory_fd: int, name: str, *, label: str) -> bytes:
    if Path(name).name != name:
        raise ScopeViolation(f"{label} basename is unsafe")
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ScopeViolation(f"{label} is not a single-link regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after:
            raise ScopeViolation(f"{label} changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_repo_regular(repo: Path, relative: str, *, label: str) -> bytes:
    relative_path = _safe_relative_path(relative, label=label)
    parent = repo.joinpath(*relative_path.parts[:-1])
    parent_fd = _open_directory_root_to_leaf(parent, label=f"{label} parent")
    try:
        return _read_regular_at(parent_fd, relative_path.name, label=label)
    finally:
        os.close(parent_fd)


def _load_verified_consumer(
    config: Mapping[str, Any],
    *,
    repo: Path,
) -> tuple[dict[str, Any], types.ModuleType]:
    """Open only the three frozen consumer authority files, never its data source."""

    authority = config["consumer_authority"]
    payloads: dict[str, bytes] = {}
    for path_key, sha_key in {
        "config_path": "config_sha256",
        "script_path": "script_sha256",
        "test_path": "test_sha256",
    }.items():
        payload = _read_repo_regular(repo, authority[path_key], label=f"consumer {path_key}")
        if sha256(payload) != authority[sha_key]:
            raise AuthorityError(f"consumer {path_key} SHA differs")
        payloads[path_key] = payload
    consumer = strict_json(payloads["config_path"], label="consumer config")
    for key, expected in {
        "schema_version": authority["schema_version"],
        "protocol_id": CONSUMER_PROTOCOL_ID,
        "contract_id": CONTRACT_ID,
        "dataset_id": DATASET_ID,
        "decision_id": DECISION_ID,
    }.items():
        _expect(consumer[key], expected, label=f"consumer config {key}")
    binding = consumer["implementation_binding"]
    _expect(binding["status"], "BOUND", label="consumer binding status")
    _expect(
        binding["implementation_script_sha256"],
        authority["script_sha256"],
        label="consumer bound script SHA",
    )
    _expect(
        binding["implementation_test_sha256"],
        authority["test_sha256"],
        label="consumer bound test SHA",
    )
    _expect(
        binding["config_core_sha256"],
        authority["science_core_sha256"],
        label="consumer science core SHA",
    )
    _expect(
        consumer_non_repository_science_projection_sha256(consumer),
        authority["required_non_repository_science_projection_sha256"],
        label="consumer non-repository science projection SHA",
    )
    descriptors = consumer.get("evidence_descriptor_bindings")
    if type(descriptors) is not dict:
        raise AuthorityError("consumer descriptor bindings are absent")
    _expect(
        descriptors.get("descriptor_set_sha256"),
        authority["required_descriptor_set_sha256"],
        label="consumer descriptor-set SHA",
    )
    evidence = consumer["evidence_contract"]
    _expect(evidence["evidence_schema_version"], EVIDENCE_SCHEMA_VERSION, label="consumer evidence schema")
    predecessor = evidence["required_predecessor_authority"]
    _expect(predecessor["bundle_id"], authority["source_bundle_id"], label="source bundle ID")
    _expect(
        predecessor["terminal_marker_final_output_target_sha256"],
        SOURCE_TARGET_SHA256,
        label="source target SHA",
    )
    acceptance = evidence["gate_record_provenance_contract"]["acceptance_authority"]
    _expect(acceptance["contract_id"], CONTRACT_ID, label="acceptance contract")
    _expect(acceptance["decision_id"], DECISION_ID, label="acceptance decision")
    _expect(acceptance["protocol_id"], CONSUMER_PROTOCOL_ID, label="acceptance protocol")
    _expect(acceptance["rule"], ACCEPTANCE_RULE, label="acceptance rule")
    slots = {slot["slot_id"]: slot for slot in evidence["slots"]}
    for spec in GATE_SPECS:
        if spec["gate_id"] not in slots:
            raise AuthorityError(f"consumer slot is absent: {spec['gate_id']}")
        _expect(
            slots[spec["gate_id"]]["allowed_basename"],
            spec["allowed_basename"],
            label=f"consumer basename {spec['gate_id']}",
        )

    module = types.ModuleType("_verified_gse200304_dec019_consumer")
    module.__file__ = os.fspath(repo / authority["script_path"])
    module.__spec__ = importlib.util.spec_from_loader(module.__name__, loader=None)
    exec(compile(payloads["script_path"], module.__file__, "exec"), module.__dict__)
    if (
        getattr(module, "EVIDENCE_SCHEMA_VERSION", None) != EVIDENCE_SCHEMA_VERSION
        or getattr(module, "EVIDENCE_RECORD_TYPE", None) != EVIDENCE_RECORD_TYPE
        or not callable(getattr(module, "_validate_gate_record", None))
        or not callable(getattr(module, "descriptor_set_sha256", None))
        or not callable(getattr(module, "validate_production_authority", None))
    ):
        raise AuthorityError("verified consumer module lacks the frozen validator API")
    if not callable(getattr(module, "validate_static_config", None)):
        raise AuthorityError("verified consumer module lacks static config validation")
    module.validate_static_config(consumer)
    _expect(
        module.descriptor_set_sha256(consumer),
        authority["required_descriptor_set_sha256"],
        label="consumer computed descriptor-set SHA",
    )
    for spec in GATE_SPECS:
        consumer_fact_keys = sorted(module.FACT_KEYS[spec["gate_id"]])
        _expect(
            consumer_fact_keys,
            spec["unknown_fields"],
            label=f"consumer FACT_KEYS {spec['gate_id']}",
        )
    return consumer, module


def validate_verified_consumer_production_authority(
    config: Mapping[str, Any],
    consumer: Mapping[str, Any],
    module: types.ModuleType,
    producer_git_authority: Mapping[str, str],
) -> dict[str, Any]:
    """Require the exact current consumer to prove its own Git lifecycle."""

    try:
        result = module.validate_production_authority(consumer)
    except Exception as exc:
        raise AuthorityError("verified consumer production authority failed") from exc
    if type(result) is not dict:
        raise AuthorityError("verified consumer production authority result is invalid")
    authority = config["consumer_authority"]
    for key, expected in {
        "lifecycle_state": "REPAIR_B_BOUND_OR_DESCRIPTOR_DESCENDANT",
        "repair_base_commit": producer_git_authority[
            "consumer_successor_base_commit"
        ],
        "repair_implementation_commit": producer_git_authority[
            "consumer_successor_implementation_commit"
        ],
        "repair_binding_commit": authority["successor_binding_commit"],
        "current_head": producer_git_authority["current_head"],
        "science_core_sha256": authority["science_core_sha256"],
        "evidence_descriptor_set_sha256": authority[
            "required_descriptor_set_sha256"
        ],
    }.items():
        _expect(result.get(key), expected, label=f"consumer Git authority {key}")
    return result


def _assert_no_forbidden_output_keys(value: Any, *, path: str = "record") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.casefold() in FORBIDDEN_OUTPUT_KEY_TOKENS:
                raise ProducerError(f"forbidden output key at {path}: {key}")
            _assert_no_forbidden_output_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_forbidden_output_keys(child, path=f"{path}[{index}]")


def build_records(
    config: Mapping[str, Any],
    consumer: Mapping[str, Any],
    consumer_module: types.ModuleType,
) -> dict[str, bytes]:
    validate_implementation_binding(config)
    validate_consumer_authority_binding(config)
    binding = config["implementation_binding"]
    predecessor = consumer["evidence_contract"]["required_predecessor_authority"]
    acceptance = consumer["evidence_contract"]["gate_record_provenance_contract"][
        "acceptance_authority"
    ]
    consumer_slots = {
        slot["slot_id"]: slot for slot in consumer["evidence_contract"]["slots"]
    }
    provenance = {
        "producer_protocol_id": PROTOCOL_ID,
        "producer_commit": binding["implementation_commit"],
        "producer_script_sha256": binding["implementation_script_sha256"],
        "source_bundle_id": predecessor["bundle_id"],
        "source_bundle_root_or_target_sha256": predecessor[
            "terminal_marker_final_output_target_sha256"
        ],
        "predecessor_authority": copy.deepcopy(predecessor),
        "acceptance_authority": copy.deepcopy(acceptance),
    }
    payloads: dict[str, bytes] = {}
    for spec in GATE_SPECS:
        record = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "record_type": EVIDENCE_RECORD_TYPE,
            "contract_id": CONTRACT_ID,
            "decision_id": DECISION_ID,
            "dataset_id": DATASET_ID,
            "gate_id": spec["gate_id"],
            "status": spec["status"],
            "accepted": True,
            "aggregate_only": True,
            "privacy": copy.deepcopy(PRIVACY),
            "provenance": copy.deepcopy(provenance),
            "facts": None,
            "unknown_fields": list(spec["unknown_fields"]),
            "reason_codes": list(spec["reason_codes"]),
        }
        _assert_no_forbidden_output_keys(record)
        payload = json_bytes(record)
        accepted = consumer_module._validate_gate_record(
            payload,
            consumer_slots[spec["gate_id"]],
            consumer,
        )
        if accepted != record:
            raise ProducerError(f"consumer validator altered record: {spec['gate_id']}")
        payloads[spec["allowed_basename"]] = payload
    if tuple(sorted(payloads)) != MEMBER_NAMES:
        raise ProducerError("generated member set differs from exact seven-record pack")
    return payloads


def _write_exclusive_regular_at(directory_fd: int, name: str, payload: bytes) -> None:
    if Path(name).name != name:
        raise PublicationError("publication member name is unsafe")
    descriptor = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o640,
        dir_fd=directory_fd,
    )
    try:
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise PublicationError(f"short write: {name}")
            view = view[count:]
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise PublicationError(f"published temp member is unsafe: {name}")
    finally:
        os.close(descriptor)


def _open_child_directory(parent_fd: int, name: str, *, label: str) -> int:
    if Path(name).name != name:
        raise ScopeViolation(f"{label} basename is unsafe")
    try:
        return os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
    except OSError as exc:
        raise PublicationError(f"{label} cannot be opened safely") from exc


def _directory_identity(descriptor: int) -> tuple[int, int]:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise PublicationError("pinned descriptor is not a directory")
    return metadata.st_dev, metadata.st_ino


def _open_matching_canonical_directory(
    path: Path,
    expected_identity: tuple[int, int],
    *,
    label: str,
) -> int:
    descriptor = _open_directory_root_to_leaf(path, label=label)
    if _directory_identity(descriptor) != expected_identity:
        os.close(descriptor)
        raise PublicationError(f"{label} identity changed")
    return descriptor


def _assert_named_directory_identity(
    parent_fd: int,
    name: str,
    directory_fd: int,
    *,
    label: str,
) -> None:
    named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    opened = os.fstat(directory_fd)
    if (
        not stat.S_ISDIR(named.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        raise PublicationError(f"{label} identity changed")


def _native_rename_noreplace(parent_fd: int, old_name: str, new_name: str) -> None:
    """Use Linux renameat2 or macOS renameatx_np with exclusive semantics."""

    library = ctypes.CDLL(None, use_errno=True)
    old = os.fsencode(old_name)
    new = os.fsencode(new_name)
    if hasattr(library, "renameat2"):
        function = library.renameat2
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(parent_fd, old, parent_fd, new, 1)  # RENAME_NOREPLACE
    elif hasattr(library, "renameatx_np"):
        function = library.renameatx_np
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(parent_fd, old, parent_fd, new, 0x00000004)  # RENAME_EXCL
    else:
        raise AtomicNoReplaceUnsupported(errno.ENOSYS)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), new_name)
    if error_number in ATOMIC_NOREPLACE_UNSUPPORTED_ERRNOS:
        raise AtomicNoReplaceUnsupported(error_number)
    raise OSError(error_number, os.strerror(error_number), new_name)


def _gate_payload_set_sha256(payloads: Mapping[str, bytes]) -> str:
    if tuple(sorted(payloads)) != MEMBER_NAMES:
        raise PublicationError("commit marker payload set is not the exact seven records")
    digest = hashlib.sha256()
    digest.update(b"GSE200304_DEC019_NEGATIVE_GATE_PAYLOAD_SET_V1\n")
    for name in MEMBER_NAMES:
        encoded_name = os.fsencode(name)
        digest.update(str(len(encoded_name)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded_name)
        digest.update(b"\0")
        digest.update(sha256(payloads[name]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _final_output_directory_name_sha256(name: str) -> str:
    if not name or Path(name).name != name:
        raise PublicationError("final output directory basename is unsafe")
    return sha256(
        b"GSE200304_DEC019_NEGATIVE_GATE_FINAL_DIRECTORY_NAME_V1\n"
        + os.fsencode(name)
        + b"\n"
    )


def _final_output_target_sha256(output: Path) -> str:
    if not output.is_absolute():
        raise PublicationError("final output target is not absolute")
    return sha256(
        b"GSE200304_DEC019_NEGATIVE_GATE_FINAL_TARGET_V1\n"
        + os.fsencode(output)
        + b"\n"
    )


def publication_commit_bytes(
    output: Path,
    payloads: Mapping[str, bytes],
    publication_mode: str,
) -> bytes:
    """Return the exact deterministic eighth member for ``output``."""

    if publication_mode not in {PRIMARY_PUBLICATION_MODE, FALLBACK_PUBLICATION_MODE}:
        raise PublicationError("publication commit marker mode is outside the closed enum")
    return json_bytes(
        {
            "schema_version": PUBLICATION_COMMIT_SCHEMA_VERSION,
            "record_type": PUBLICATION_COMMIT_RECORD_TYPE,
            "contract_id": CONTRACT_ID,
            "protocol_id": PROTOCOL_ID,
            "dataset_id": DATASET_ID,
            "decision_id": DECISION_ID,
            "publication_mode": publication_mode,
            "gate_record_count": len(MEMBER_NAMES),
            "gate_record_names": list(MEMBER_NAMES),
            "gate_payload_set_sha256": _gate_payload_set_sha256(payloads),
            "final_output_directory_name_sha256": (
                _final_output_directory_name_sha256(output.name)
            ),
            "final_output_target_sha256": _final_output_target_sha256(output),
            "descriptor_binding_scope": "SEVEN_GATE_JSON_FILES_ONLY",
            "committed": True,
            "commit_marker_written_last": True,
        }
    )


def _fsync_regular_at(directory_fd: int, name: str, *, label: str) -> None:
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ScopeViolation(f"{label} is not a single-link regular file")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _inspect_committed_directory_at(
    parent_fd: int,
    name: str,
    output: Path,
    expected: Mapping[str, bytes],
    *,
    expected_modes: Sequence[str] = (
        PRIMARY_PUBLICATION_MODE,
        FALLBACK_PUBLICATION_MODE,
    ),
) -> tuple[tuple[int, int], str]:
    directory_fd = _open_child_directory(parent_fd, name, label="final output")
    try:
        _assert_named_directory_identity(parent_fd, name, directory_fd, label="final output")
        names = sorted(os.listdir(directory_fd))
        if names != list(PUBLISHED_MEMBER_NAMES):
            raise PartialPublicationError(
                "final directory is not exact8 committed; overwrite refused"
            )
        for member in MEMBER_NAMES:
            observed = _read_regular_at(directory_fd, member, label=f"final member {member}")
            if observed != expected[member]:
                raise PublicationError(
                    f"existing final member differs; overwrite refused: {member}"
                )
        marker = _read_regular_at(
            directory_fd,
            PUBLICATION_COMMIT_FILENAME,
            label="publication commit marker",
        )
        matched_mode: str | None = None
        for mode in expected_modes:
            if marker == publication_commit_bytes(output, expected, mode):
                matched_mode = mode
                break
        if matched_mode is None:
            raise PartialPublicationError(
                "publication commit marker is early, invalid, or for another target"
            )
        _assert_named_directory_identity(parent_fd, name, directory_fd, label="final output")
        return _directory_identity(directory_fd), matched_mode
    finally:
        os.close(directory_fd)


def validate_committed_publication(
    output: Path,
    payloads: Mapping[str, bytes],
    *,
    expected_parent_identity: tuple[int, int] | None = None,
    expected_modes: Sequence[str] = (
        PRIMARY_PUBLICATION_MODE,
        FALLBACK_PUBLICATION_MODE,
    ),
) -> dict[str, Any]:
    """Validate, durably sync, reopen, and revalidate the exact8 publication.

    This is the acceptance primitive intended for the downstream descriptor
    binder.  Merely observing the marker pathname is never sufficient.
    """

    if tuple(sorted(payloads)) != MEMBER_NAMES:
        raise PublicationError("publication inspector expected the exact seven records")
    parent_fd = _open_directory_root_to_leaf(output.parent, label="publication parent")
    try:
        parent_identity = _directory_identity(parent_fd)
        if (
            expected_parent_identity is not None
            and parent_identity != expected_parent_identity
        ):
            raise PublicationError("publication parent identity changed")
        identity, mode = _inspect_committed_directory_at(
            parent_fd,
            output.name,
            output,
            payloads,
            expected_modes=expected_modes,
        )
        directory_fd = _open_child_directory(
            parent_fd, output.name, label="committed final output"
        )
        try:
            if _directory_identity(directory_fd) != identity:
                raise PublicationError("committed final directory identity changed")
            for member in PUBLISHED_MEMBER_NAMES:
                _fsync_regular_at(
                    directory_fd,
                    member,
                    label=f"committed member {member}",
                )
            os.fsync(directory_fd)
            _assert_named_directory_identity(
                parent_fd,
                output.name,
                directory_fd,
                label="committed final output",
            )
        finally:
            os.close(directory_fd)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)

    reopened_parent = _open_matching_canonical_directory(
        output.parent,
        parent_identity,
        label="publication parent post-fsync",
    )
    try:
        reopened_identity, reopened_mode = _inspect_committed_directory_at(
            reopened_parent,
            output.name,
            output,
            payloads,
            expected_modes=expected_modes,
        )
        if reopened_identity != identity or reopened_mode != mode:
            raise PublicationError("committed publication changed after fsync/reopen")
    finally:
        os.close(reopened_parent)
    return {
        "directory_identity": identity,
        "publication_mode": mode,
        "exact_member_names": list(PUBLISHED_MEMBER_NAMES),
        "descriptor_member_names": list(MEMBER_NAMES),
        "terminal_commit_marker_validated": True,
    }


def _cleanup_unpublished_temp(
    parent_path: Path,
    expected_parent_identity: tuple[int, int],
    temp_name: str,
    expected_temp_identity: tuple[int, int],
    member_names: Sequence[str],
) -> bool:
    """Remove only this call's temp at the still-canonical parent path.

    The retained parent descriptor alone is insufficient: its directory may
    have been renamed outside the configured publication root.  Reopening the
    canonical path and matching both parent and temp identities prevents us
    from deleting an attacker-supplied same-name replacement directory.
    """

    try:
        parent_fd = _open_directory_root_to_leaf(
            parent_path, label="unpublished temp parent"
        )
    except (PublicationError, ScopeViolation):
        return False
    temp_fd = -1
    try:
        if _directory_identity(parent_fd) != expected_parent_identity:
            return False
        named = os.stat(temp_name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(named.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or (named.st_dev, named.st_ino) != expected_temp_identity
        ):
            return False
        temp_fd = _open_child_directory(parent_fd, temp_name, label="unpublished temp")
        if _directory_identity(temp_fd) != expected_temp_identity:
            return False
        _assert_named_directory_identity(
            parent_fd, temp_name, temp_fd, label="unpublished temp"
        )
        names = sorted(os.listdir(temp_fd))
        if any(name not in member_names for name in names):
            return False
        for name in names:
            metadata = os.stat(name, dir_fd=temp_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                return False
        _assert_named_directory_identity(
            parent_fd, temp_name, temp_fd, label="unpublished temp"
        )
        canonical_check = _open_directory_root_to_leaf(
            parent_path, label="unpublished temp parent pre-cleanup"
        )
        try:
            if _directory_identity(canonical_check) != expected_parent_identity:
                return False
        finally:
            os.close(canonical_check)
        for name in names:
            os.unlink(name, dir_fd=temp_fd)
        os.fsync(temp_fd)
        _assert_named_directory_identity(
            parent_fd, temp_name, temp_fd, label="unpublished temp"
        )
        os.close(temp_fd)
        temp_fd = -1
        canonical_check = _open_directory_root_to_leaf(
            parent_path, label="unpublished temp parent pre-rmdir"
        )
        try:
            if _directory_identity(canonical_check) != expected_parent_identity:
                return False
        finally:
            os.close(canonical_check)
        named = os.stat(temp_name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(named.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or (named.st_dev, named.st_ino) != expected_temp_identity
        ):
            return False
        os.rmdir(temp_name, dir_fd=parent_fd)
        os.fsync(parent_fd)
        return True
    except (OSError, PublicationError):
        return False
    finally:
        if temp_fd >= 0:
            os.close(temp_fd)
        os.close(parent_fd)


def _rename_error_state(
    parent_fd: int,
    temp_name: str,
    final_name: str,
    temp_identity: tuple[int, int],
) -> str:
    def identity(name: str) -> tuple[int, int] | None:
        try:
            metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            return (-1, -1)
        return metadata.st_dev, metadata.st_ino

    temp = identity(temp_name)
    final = identity(final_name)
    if final == temp_identity and temp is None:
        return "COMMITTED_UNVERIFIED"
    if temp == temp_identity and final is None:
        return "NOT_COMMITTED_TEMP_PRESENT"
    return "AMBIGUOUS_RENAME_STATE"


def _validate_recoverable_partial_at(
    parent_fd: int,
    final_name: str,
    payloads: Mapping[str, bytes],
) -> tuple[int, set[str]]:
    """Pin and validate an unmarked exact subset; return its open descriptor."""

    directory_fd = _open_child_directory(
        parent_fd, final_name, label="partial final output"
    )
    try:
        _assert_named_directory_identity(
            parent_fd,
            final_name,
            directory_fd,
            label="partial final output",
        )
        metadata = os.fstat(directory_fd)
        if metadata.st_uid != os.geteuid():
            raise ScopeViolation(
                "partial recovery requires a final directory owned by the current euid"
            )
        names = set(os.listdir(directory_fd))
        if PUBLICATION_COMMIT_FILENAME in names:
            raise PartialPublicationError(
                "partial final contains an early or invalid commit marker; manual recovery required"
            )
        unknown = names - set(MEMBER_NAMES)
        if unknown:
            raise PartialPublicationError(
                "partial final contains an unexpected member; manual recovery required"
            )
        for name in sorted(names):
            observed = _read_regular_at(
                directory_fd,
                name,
                label=f"partial final member {name}",
            )
            if observed != payloads[name]:
                raise PartialPublicationError(
                    f"partial final member differs; manual recovery required: {name}"
                )
        _assert_named_directory_identity(
            parent_fd,
            final_name,
            directory_fd,
            label="partial final output",
        )
        return directory_fd, names
    except Exception:
        os.close(directory_fd)
        raise


def _write_missing_record_or_validate_race(
    directory_fd: int,
    name: str,
    payload: bytes,
) -> None:
    try:
        _write_exclusive_regular_at(directory_fd, name, payload)
    except FileExistsError:
        observed = _read_regular_at(
            directory_fd,
            name,
            label=f"concurrent fallback member {name}",
        )
        if observed != payload:
            raise PartialPublicationError(
                f"concurrent fallback member differs; overwrite refused: {name}"
            )


def _publish_with_terminal_marker_fallback(
    output: Path,
    payloads: Mapping[str, bytes],
    *,
    expected_parent_identity: tuple[int, int],
    recover_partial: bool,
    create_if_absent: bool,
    fault_injector: FaultInjector | None,
) -> str:
    """NFS-safe exact8 publication using mkdir exclusivity and marker-last."""

    parent_fd = _open_matching_canonical_directory(
        output.parent,
        expected_parent_identity,
        label="fallback publication parent",
    )
    directory_fd = -1
    created = False
    marker_write_completed = False
    try:
        try:
            directory_fd = _open_child_directory(
                parent_fd, output.name, label="fallback final output"
            )
        except PublicationError as open_error:
            try:
                os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                if not create_if_absent:
                    raise PublicationStateError(
                        "contended final target disappeared; publication state is ambiguous",
                        publication_state="AMBIGUOUS_FINAL_CONTENTION_STATE",
                    ) from open_error
                try:
                    os.mkdir(output.name, 0o750, dir_fd=parent_fd)
                    created = True
                except FileExistsError:
                    pass
                except OSError as exc:
                    raise PublicationError(
                        "atomic fallback final-directory creation failed"
                    ) from exc
                directory_fd = _open_child_directory(
                    parent_fd, output.name, label="fallback final output"
                )
            else:
                raise

        final_identity = _directory_identity(directory_fd)
        _assert_named_directory_identity(
            parent_fd,
            output.name,
            directory_fd,
            label="fallback final output",
        )

        names = set(os.listdir(directory_fd))
        if PUBLICATION_COMMIT_FILENAME in names:
            os.close(directory_fd)
            directory_fd = -1
            committed = validate_committed_publication(
                output,
                payloads,
                expected_parent_identity=expected_parent_identity,
            )
            if committed["directory_identity"] != final_identity:
                raise PublicationError("existing committed final identity changed")
            return "EXISTING_EXACT"

        os.close(directory_fd)
        directory_fd = -1
        directory_fd, present = _validate_recoverable_partial_at(
            parent_fd,
            output.name,
            payloads,
        )
        if _directory_identity(directory_fd) != final_identity:
            raise PublicationError("partial final identity changed before recovery")
        if not created and not recover_partial:
            raise PartialPublicationError(
                "publication state is PARTIAL_NOT_COMMITTED; exact-subset recovery "
                "requires explicit --recover-partial and the directory is preserved"
            )
        if created:
            os.fsync(parent_fd)

        for name in MEMBER_NAMES:
            if name not in present:
                _write_missing_record_or_validate_race(
                    directory_fd,
                    name,
                    payloads[name],
                )
                if fault_injector is not None:
                    fault_injector(f"after_fallback_write:{name}")
        names = set(os.listdir(directory_fd))
        if PUBLICATION_COMMIT_FILENAME in names:
            os.close(directory_fd)
            directory_fd = -1
            committed = validate_committed_publication(
                output,
                payloads,
                expected_parent_identity=expected_parent_identity,
                expected_modes=(FALLBACK_PUBLICATION_MODE,),
            )
            if committed["directory_identity"] != final_identity:
                raise PublicationError(
                    "concurrently committed fallback final identity changed"
                )
            return "EXISTING_EXACT"
        if names != set(MEMBER_NAMES):
            raise PartialPublicationError(
                "fallback final is not the exact seven-record pre-marker set"
            )
        for name in MEMBER_NAMES:
            observed = _read_regular_at(
                directory_fd,
                name,
                label=f"fallback final member {name}",
            )
            if observed != payloads[name]:
                raise PartialPublicationError(
                    f"fallback final member differs before marker: {name}"
                )
        os.fsync(directory_fd)
        _assert_named_directory_identity(
            parent_fd,
            output.name,
            directory_fd,
            label="fallback final output pre-marker",
        )

        marker_payload = publication_commit_bytes(
            output,
            payloads,
            FALLBACK_PUBLICATION_MODE,
        )
        try:
            _write_exclusive_regular_at(
                directory_fd,
                PUBLICATION_COMMIT_FILENAME,
                marker_payload,
            )
            marker_write_completed = True
        except FileExistsError:
            observed_marker = _read_regular_at(
                directory_fd,
                PUBLICATION_COMMIT_FILENAME,
                label="concurrent publication commit marker",
            )
            if observed_marker != marker_payload:
                raise PartialPublicationError(
                    "concurrent publication commit marker differs; manual recovery required"
                )
            marker_write_completed = True
        if fault_injector is not None:
            fault_injector("after_fallback_marker")
        try:
            os.fsync(directory_fd)
            os.fsync(parent_fd)
            _assert_named_directory_identity(
                parent_fd,
                output.name,
                directory_fd,
                label="fallback final output post-marker",
            )
        except Exception as exc:
            raise PublicationStateError(
                "exact fallback marker exists but durability/identity is unverified",
                publication_state="COMMIT_MARKER_EXACT_DURABILITY_UNVERIFIED",
            ) from exc
        os.close(directory_fd)
        directory_fd = -1
        committed = validate_committed_publication(
            output,
            payloads,
            expected_parent_identity=expected_parent_identity,
            expected_modes=(FALLBACK_PUBLICATION_MODE,),
        )
        if committed["directory_identity"] != final_identity:
            raise PublicationStateError(
                "fallback final identity changed after commit-marker validation",
                publication_state="COMMITTED_UNVERIFIED",
            )
        return "PUBLISHED_FALLBACK" if created else "RECOVERED_PARTIAL_FALLBACK"
    except PublicationStateError:
        raise
    except (PartialPublicationError, ScopeViolation):
        raise
    except Exception as exc:
        state = (
            "COMMIT_MARKER_PRESENT_NOT_ACCEPTED"
            if marker_write_completed
            else "PARTIAL_NOT_COMMITTED"
        )
        raise PartialPublicationError(
            f"fallback publication state is {state}; final directory is preserved"
        ) from exc
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)
        os.close(parent_fd)


def publish_records(
    output: Path,
    payloads: Mapping[str, bytes],
    *,
    production: bool,
    config: Mapping[str, Any],
    recover_partial: bool = False,
    fault_injector: FaultInjector | None = None,
) -> str:
    if tuple(sorted(payloads)) != MEMBER_NAMES:
        raise PublicationError("publisher input is not the exact seven-member set")
    if not output.is_absolute() or any(part in {"", ".", ".."} for part in output.parts[1:]):
        raise ScopeViolation("output must be an absolute path with safe components")
    if production and os.fspath(output) != config["output_contract"]["trusted_final_directory"]:
        raise ScopeViolation("production output differs from the exact trusted final directory")
    lowered = os.fspath(output).casefold()
    for token in {"gse246381", "/restricted/", "/sealed/", "/sealed_external/", "access_log"}:
        if token in lowered:
            raise ScopeViolation(f"output path contains forbidden token: {token}")

    parent_fd = _open_directory_root_to_leaf(output.parent, label="output parent")
    parent_identity = _directory_identity(parent_fd)
    temp_name = f".{output.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    temp_fd = -1
    temp_identity: tuple[int, int] | None = None
    temp_active = False
    renamed = False
    fallback_attempted = False
    try:
        try:
            os.mkdir(temp_name, 0o750, dir_fd=parent_fd)
            temp_active = True
        except FileExistsError as exc:
            raise PublicationError("exclusive temp directory collision") from exc
        temp_fd = _open_child_directory(parent_fd, temp_name, label="publication temp")
        temp_identity = _directory_identity(temp_fd)
        _assert_named_directory_identity(parent_fd, temp_name, temp_fd, label="publication temp")
        for name in sorted(payloads):
            _write_exclusive_regular_at(temp_fd, name, payloads[name])
            if fault_injector is not None:
                fault_injector(f"after_write:{name}")
        if sorted(os.listdir(temp_fd)) != sorted(payloads):
            raise PublicationError("temp directory member set differs")
        os.fsync(temp_fd)
        _write_exclusive_regular_at(
            temp_fd,
            PUBLICATION_COMMIT_FILENAME,
            publication_commit_bytes(output, payloads, PRIMARY_PUBLICATION_MODE),
        )
        if sorted(os.listdir(temp_fd)) != list(PUBLISHED_MEMBER_NAMES):
            raise PublicationError("primary temp is not the exact8 marker-last set")
        os.fsync(temp_fd)
        os.fsync(parent_fd)
        _assert_named_directory_identity(parent_fd, temp_name, temp_fd, label="publication temp")
        if fault_injector is not None:
            fault_injector("before_atomic_rename")
        canonical_parent = _open_matching_canonical_directory(
            output.parent,
            parent_identity,
            label="output parent pre-rename",
        )
        os.close(canonical_parent)
        try:
            _native_rename_noreplace(parent_fd, temp_name, output.name)
            renamed = True
            temp_active = False
        except FileExistsError:
            if temp_identity is None or not _cleanup_unpublished_temp(
                output.parent,
                parent_identity,
                temp_name,
                temp_identity,
                PUBLISHED_MEMBER_NAMES,
            ):
                # Cleanup already refused because canonical identity could not
                # be proven.  Preserve the original temp; never make a second
                # pathname-based cleanup attempt from the outer handler.
                temp_active = False
                raise PartialPublicationError(
                    f"final target exists and unpublished temp cleanup failed: {temp_name}"
                )
            temp_active = False
            return _publish_with_terminal_marker_fallback(
                output,
                payloads,
                expected_parent_identity=parent_identity,
                recover_partial=recover_partial,
                create_if_absent=False,
                fault_injector=fault_injector,
            )
        except AtomicNoReplaceUnsupported as exc:
            if temp_identity is None:
                raise PublicationStateError(
                    "rename fallback requested before temp identity was known",
                    publication_state="AMBIGUOUS_RENAME_STATE",
                ) from exc
            state = _rename_error_state(parent_fd, temp_name, output.name, temp_identity)
            if state != "NOT_COMMITTED_TEMP_PRESENT":
                temp_active = False
                raise PublicationStateError(
                    "unsupported atomic rename returned an unsafe state; fallback refused; "
                    f"publication state is {state}",
                    publication_state=state,
                ) from exc
            fallback_attempted = True
            fallback_status = _publish_with_terminal_marker_fallback(
                output,
                payloads,
                expected_parent_identity=parent_identity,
                recover_partial=recover_partial,
                create_if_absent=True,
                fault_injector=fault_injector,
            )
            cleaned = _cleanup_unpublished_temp(
                output.parent,
                parent_identity,
                temp_name,
                temp_identity,
                PUBLISHED_MEMBER_NAMES,
            )
            temp_active = False
            if not cleaned:
                return f"{fallback_status}_COMMITTED_EXACT_SOURCE_TEMP_PRESERVED"
            return fallback_status
        except OSError as exc:
            if temp_identity is None:
                raise PublicationStateError(
                    "rename failed before temp identity was known",
                    publication_state="AMBIGUOUS_RENAME_STATE",
                ) from exc
            state = _rename_error_state(parent_fd, temp_name, output.name, temp_identity)
            if state == "NOT_COMMITTED_TEMP_PRESENT":
                cleaned = _cleanup_unpublished_temp(
                    output.parent,
                    parent_identity,
                    temp_name,
                    temp_identity,
                    PUBLISHED_MEMBER_NAMES,
                )
                if not cleaned:
                    state = "NOT_COMMITTED_TEMP_PRESERVED"
                    temp_active = False
                else:
                    state = "NOT_COMMITTED_TEMP_REMOVED"
                    temp_active = False
            else:
                # Ambiguous or possibly committed states must be preserved for
                # explicit recovery; never attempt pathname-based cleanup.
                temp_active = False
            raise PublicationStateError(
                f"atomic rename failed; publication state is {state}",
                publication_state=state,
            ) from exc

        if fault_injector is not None:
            fault_injector("after_atomic_rename")
        committed = validate_committed_publication(
            output,
            payloads,
            expected_parent_identity=parent_identity,
            expected_modes=(PRIMARY_PUBLICATION_MODE,),
        )
        if committed["directory_identity"] != temp_identity:
            raise PublicationStateError(
                "final directory identity differs after successful atomic rename",
                publication_state="COMMITTED_UNVERIFIED",
            )
        return "PUBLISHED_PRIMARY"
    except Exception as exc:
        if renamed:
            state = "COMMITTED_UNVERIFIED"
            canonical_parent = -1
            try:
                canonical_parent = _open_matching_canonical_directory(
                    output.parent,
                    parent_identity,
                    label="output parent post-commit recovery",
                )
                os.close(canonical_parent)
                canonical_parent = -1
                committed = validate_committed_publication(
                    output,
                    payloads,
                    expected_parent_identity=parent_identity,
                )
                if committed["directory_identity"] == temp_identity:
                    state = "COMMITTED_EXACT"
            except Exception:
                pass
            finally:
                if canonical_parent >= 0:
                    os.close(canonical_parent)
            if isinstance(exc, PublicationStateError):
                raise
            raise PublicationStateError(
                f"post-rename failure; publication state is {state}: {exc}",
                publication_state=state,
            ) from exc
        if temp_identity is not None and temp_active:
            cleaned = _cleanup_unpublished_temp(
                output.parent,
                parent_identity,
                temp_name,
                temp_identity,
                PUBLISHED_MEMBER_NAMES,
            )
            if not cleaned:
                raise PartialPublicationError(
                    f"not committed; unpublished temp preserved: {temp_name}"
                ) from exc
            if fallback_attempted and isinstance(exc, PartialPublicationError):
                raise PartialPublicationError(
                    f"{exc}; source temp was safely removed"
                ) from exc
        raise
    finally:
        if temp_fd >= 0:
            os.close(temp_fd)
        os.close(parent_fd)


def produce(
    config: Mapping[str, Any],
    output: Path,
    *,
    production: bool,
    repo: Path,
    recover_partial: bool = False,
    fault_injector: FaultInjector | None = None,
) -> dict[str, Any]:
    """Validate binding first, then consumer authority, then publish."""

    validate_static_config(config)
    # This ordering is a contractual no-I/O gate: UNKNOWN stops before the
    # consumer source files and the output parent are opened.
    validate_implementation_binding(config)
    validate_consumer_authority_binding(config)
    git_authority: dict[str, Any] | None = None
    if production:
        git_authority = validate_production_authority(config, repo=repo)
    consumer, module = _load_verified_consumer(config, repo=repo)
    if production:
        if git_authority is None:
            raise AuthorityError("producer Git authority result is absent")
        git_authority["consumer_validator_authority"] = (
            validate_verified_consumer_production_authority(
                config,
                consumer,
                module,
                git_authority,
            )
        )
    payloads = build_records(config, consumer, module)
    publication_status = publish_records(
        output,
        payloads,
        production=production,
        config=config,
        recover_partial=recover_partial,
        fault_injector=fault_injector,
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "decision_id": DECISION_ID,
        "publication_status": publication_status,
        "partial_recovery_requested": recover_partial,
        "final_directory": os.fspath(output),
        "record_count": len(payloads),
        "member_names": sorted(payloads),
        "all_records_consumer_accepted": True,
        "aggregate_only": True,
        "ordinary_study_contribution_delta": 0,
        "a1_study_contribution_delta": 0,
        "true_a2_study_contribution_delta": 0,
        "canonical_record_count_delta": 0,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "git_authority": git_authority,
    }


def _load_producer_config(path: Path) -> dict[str, Any]:
    if not path.is_absolute():
        path = path.resolve()
    parent_fd = _open_directory_root_to_leaf(path.parent, label="producer config parent")
    try:
        payload = _read_regular_at(parent_fd, path.name, label="producer config")
    finally:
        os.close(parent_fd)
    return strict_json(payload, label="producer config")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PRODUCTION_CONFIG_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-authority", action="store_true")
    parser.add_argument(
        "--recover-partial",
        action="store_true",
        help=(
            "explicitly complete an owner-matched, unmarked, byte-exact subset; "
            "never repairs an early/invalid marker or an unexpected member"
        ),
    )
    parser.add_argument(
        "--non-production",
        action="store_true",
        help="development-only mode; production is the default",
    )
    arguments = parser.parse_args(argv)
    if arguments.validate_authority and arguments.recover_partial:
        parser.error("--recover-partial cannot be combined with --validate-authority")
    production = not arguments.non_production
    config_path = arguments.config.resolve()
    if production and config_path != PRODUCTION_CONFIG_PATH:
        raise ScopeViolation("production config path differs from exact authority path")
    config = _load_producer_config(config_path)
    repo = PRODUCTION_REPO_ROOT if production else config_path.parents[1]
    if arguments.validate_authority:
        validate_static_config(config)
        validate_implementation_binding(config)
        validate_consumer_authority_binding(config)
        if production:
            git_authority = validate_production_authority(config, repo=repo)
        else:
            git_authority = None
        consumer, module = _load_verified_consumer(config, repo=repo)
        if production:
            if git_authority is None:
                raise AuthorityError("producer Git authority result is absent")
            git_authority["consumer_validator_authority"] = (
                validate_verified_consumer_production_authority(
                    config,
                    consumer,
                    module,
                    git_authority,
                )
            )
        payloads = build_records(config, consumer, module)
        print(
            json.dumps(
                {
                    "status": "PASS_AUTHORITY_VALIDATED_NO_PUBLICATION",
                    "record_count": len(payloads),
                    "all_records_consumer_accepted": True,
                    "git_authority": git_authority,
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.output is None:
        parser.error("--output is required unless --validate-authority is used")
    result = produce(
        config,
        arguments.output,
        production=production,
        repo=repo,
        recover_partial=arguments.recover_partial,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProducerError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
