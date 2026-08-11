#!/usr/bin/env python3
"""Publish three consumer-v3 PASS gate records from one immutable exact6 audit.

The only semantic input is the closed upstream audit JSON.  The verbatim JATS,
SOFT, and processed matrix members are opened only to stream-check their frozen
byte counts and SHA-256 digests; they are never decoded or materialized here.
The formal gate-record provenance remains the exact predecessor and acceptance
authority copied from the frozen D2 consumer config.

This producer never runs the consumer, adjudicator, qualifier, canonicalizer,
row-mapping producer, raw replay, training, model selection, or a next phase.
Its output is an exact six-member marker-last pack: three aggregate PASS gate
JSON files, one aggregate pack audit, SHA256SUMS, and PUBLICATION_COMMIT.json.
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


CONFIG_REPO_PATH = (
    "configs/route_a_v3_gse200304_dec019_upstream_pass_gate_pack_v1.json"
)
SCRIPT_REPO_PATH = (
    "scripts/route_a_v3/produce_gse200304_dec019_upstream_pass_gate_pack.py"
)
TEST_REPO_PATH = (
    "tests/route_a_v3/test_produce_gse200304_dec019_upstream_pass_gate_pack.py"
)
PRODUCTION_REPO_ROOT = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/routea_v3_a1_20260810"
)
PRODUCTION_CONFIG_PATH = PRODUCTION_REPO_ROOT / CONFIG_REPO_PATH
BRANCH = "routea-v3-a1-20260810"
BASE_COMMIT = "9c313d2793880edd2a4355ec3781e045cae27252"
UNKNOWN = "UNKNOWN_NOT_ASSERTED"
BOUND = "BOUND"
CONTRACT_ID = "mrna_xeditflow_route_a_v3"
DATASET_ID = "GSE200304"
DECISION_ID = "V3-DEC-019"
PHASE_ID = "A1"
SCHEMA_VERSION = "route_a_v3_gse200304_dec019_upstream_pass_gate_pack.v1"
PROTOCOL_ID = "ROUTE_A_V3_GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_V1"
EVIDENCE_SCHEMA_VERSION = "route_a_v3_dec019_aggregate_gate_evidence.v3"
EVIDENCE_RECORD_TYPE = "ROUTE_A_V3_DEC019_ACCEPTED_AGGREGATE_GATE_EVIDENCE_V3"
CONSUMER_PROTOCOL_ID = (
    "ROUTE_A_V3_GSE200304_DEC019_REPORTED_ENDPOINT_A1_ACTIVATION_V3"
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
UPSTREAM_PROTOCOL_ID = "GSE200304_UPSTREAM_AUTHORITY_VIABILITY_V1"
UPSTREAM_AUDIT_SCHEMA_VERSION = (
    "route_a_v3_gse200304_upstream_authority_viability.v1"
)
UPSTREAM_AUDIT_RECORD_TYPE = (
    "GSE200304_UPSTREAM_SOURCE_AUTHORITY_VIABILITY_AUDIT_V1"
)
UPSTREAM_AUDIT_STATUS = (
    "CLOSED_SOURCE_AUTHORITY_VIABILITY_READY_COMPONENTS_NO_GATE_CHANGE"
)
UPSTREAM_AUDIT_MODE = "AUDIT_ONLY_NO_GATE_CHANGE"
UPSTREAM_MARKER_RECORD_TYPE = (
    "GSE200304_UPSTREAM_AUTHORITY_VIABILITY_PUBLICATION_COMMIT_V1"
)
UPSTREAM_PUBLICATION_MODE = "NFS_SAFE_EXCLUSIVE_MKDIR_O_EXCL_TERMINAL_MARKER_V1"
UPSTREAM_PARTIAL_DEFAULT = "PRESERVE_AND_REQUIRE_MANUAL_ADJUDICATION"
ACCEPTANCE_RULE = "CONFIG_HASH_BOUND_ACCEPTED_AGGREGATE_GATE_RECORD_V3"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

IMPLEMENTATION_PATHS = [CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH]
BINDING_PATHS = [CONFIG_REPO_PATH]
BINDING_SCALAR_PATHS = [
    "implementation_binding.status",
    "implementation_binding.implementation_commit",
    "implementation_binding.implementation_script_sha256",
    "implementation_binding.implementation_test_sha256",
]
FROZEN_CONFIG_CORE_SHA256 = (
    "b9abdffded9d3eafad8f346fdd571717dc675dc17a7f90e52a4ed41fea2bb907"
)
GATE_SPECS = [
    {
        "gate_id": "CANONICAL_REPORTED_ENDPOINT_SEMANTICS",
        "allowed_basename": (
            "GSE200304_DEC019_CANONICAL_REPORTED_ENDPOINT_SEMANTICS_GATE.json"
        ),
        "facts": {
            "author_published_processed_endpoint_is_primary": True,
            "endpoint_id_frozen": True,
            "endpoint_direction_frozen": True,
            "endpoint_scale_frozen": True,
            "contrast_and_transform_frozen": True,
            "paper_faithful_mapping_closed": True,
        },
    },
    {
        "gate_id": "ROW_REPLICATE_OR_VALID_SE",
        "allowed_basename": (
            "GSE200304_DEC019_ROW_REPLICATE_OR_VALID_SE_GATE.json"
        ),
        "facts": {
            "replicate_or_valid_standard_error_present": True,
            "replicate_count_or_effective_n_frozen": True,
            "standard_error_semantics_frozen": True,
            "technical_uncertainty_not_substituted_for_biological_se": True,
        },
    },
    {
        "gate_id": "LICENSE_RIGHTS",
        "allowed_basename": "GSE200304_DEC019_LICENSE_RIGHTS_GATE.json",
        "facts": {
            "rights_source_authority_closed": True,
            "qualification_use_allowed": True,
            "private_canonical_materialization_allowed": True,
            "redistribution_scope": "PRIVATE_CANONICAL_ONLY",
        },
    },
]
GATE_MEMBER_NAMES = tuple(sorted(spec["allowed_basename"] for spec in GATE_SPECS))
AUDIT_MEMBER_NAME = "GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_AUDIT.json"
CHECKSUMS_MEMBER_NAME = "SHA256SUMS"
MARKER_MEMBER_NAME = "PUBLICATION_COMMIT.json"
CONTENT_MEMBER_NAMES = tuple(sorted((*GATE_MEMBER_NAMES, AUDIT_MEMBER_NAME)))
PRETERMINAL_MEMBER_NAMES = tuple(
    sorted((*CONTENT_MEMBER_NAMES, CHECKSUMS_MEMBER_NAME))
)
PUBLISHED_MEMBER_NAMES = tuple(sorted((*PRETERMINAL_MEMBER_NAMES, MARKER_MEMBER_NAME)))
PRIMARY_PUBLICATION_MODE = "ATOMIC_EXCLUSIVE_DIRECTORY_RENAME_NOREPLACE_V1"
FALLBACK_PUBLICATION_MODE = "ATOMIC_MKDIR_TERMINAL_COMMIT_MARKER_V1"
PUBLICATION_MARKER_RECORD_TYPE = (
    "GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_PUBLICATION_COMMIT_V1"
)
PRIVACY = {
    "contains_row_level_payload": False,
    "contains_sequence": False,
    "contains_row_identifier": False,
    "contains_raw_label_or_effect": False,
    "contains_member_identifiers_or_hashes": False,
}
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
    "upstream_authority",
    "expected_audit_semantics",
    "pass_gate_records",
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


class ProducerError(RuntimeError):
    """Base class for schema, authority, input, or publication failure."""


class BindingError(ProducerError):
    """The two-commit implementation binding is absent or invalid."""


class AuthorityError(ProducerError):
    """Git or consumer authority differs from the frozen authority."""


class InputIntegrityError(ProducerError):
    """The immutable upstream exact6 or its closed semantics differ."""


class ScopeViolation(ProducerError):
    """A path left the exact allowed scope or contains an unsafe component."""


class PublicationError(ProducerError):
    """The exact six-member no-overwrite publication could not be proved."""


class PartialPublicationError(PublicationError):
    """A partial/unexpected target was preserved and not accepted."""


class PublicationStateError(PublicationError):
    """A post-marker/rename failure with an explicit known publication truth."""

    def __init__(self, message: str, *, publication_state: str) -> None:
        super().__init__(message)
        self.publication_state = publication_state


class AtomicNoReplaceUnsupported(PublicationError):
    """The filesystem rejected the frozen no-replace rename primitive."""

    def __init__(self, error_number: int) -> None:
        super().__init__(f"atomic no-replace rename unsupported: errno={error_number}")
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
    if type(value) is not type(expected) or value != expected:
        raise ProducerError(f"{label} differs")


def _scalar_differences(before: Any, after: Any, prefix: str = "") -> set[str]:
    if type(before) is not type(after):
        return {prefix or "$"}
    if isinstance(before, dict):
        if set(before) != set(after):
            return {prefix or "$"}
        result: set[str] = set()
        for key in before:
            child = f"{prefix}.{key}" if prefix else key
            result.update(_scalar_differences(before[key], after[key], child))
        return result
    if isinstance(before, list):
        if len(before) != len(after):
            return {prefix or "$"}
        result: set[str] = set()
        for index, (old, new) in enumerate(zip(before, after)):
            result.update(
                _scalar_differences(old, new, f"{prefix}[{index}]")
            )
        return result
    return set() if before == after else {prefix or "$"}


def config_core_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    projected = copy.deepcopy(dict(config))
    projected.pop("implementation_binding", None)
    return projected


def config_core_sha256(config: Mapping[str, Any]) -> str:
    return sha256(json_bytes(config_core_projection(config)))


def validate_static_config(config: Mapping[str, Any]) -> None:
    value = _expect_exact_keys(config, EXPECTED_TOP_KEYS, label="producer config")
    for key, expected in {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "contract_id": CONTRACT_ID,
        "phase_id": PHASE_ID,
        "dataset_id": DATASET_ID,
        "decision_id": DECISION_ID,
    }.items():
        _expect(value[key], expected, label=key)

    binding = _expect_exact_keys(
        value["implementation_binding"],
        EXPECTED_BINDING_KEYS,
        label="implementation binding",
    )
    for key, expected in {
        "binding_scheme": "CONFIG_ONLY_POST_IMPLEMENTATION_BINDING_V1",
        "blocker_if_unbound": (
            "UPSTREAM_PASS_GATE_PACK_IMPLEMENTATION_BINDING_UNKNOWN_NOT_ASSERTED"
        ),
        "implementation_script_path": SCRIPT_REPO_PATH,
        "implementation_test_path": TEST_REPO_PATH,
        "unknown_to_bound_scalar_paths": BINDING_SCALAR_PATHS,
        "config_core_sha256": FROZEN_CONFIG_CORE_SHA256,
    }.items():
        _expect(binding[key], expected, label=f"implementation {key}")
    _expect(
        config_core_sha256(value),
        FROZEN_CONFIG_CORE_SHA256,
        label="computed config core SHA",
    )
    if binding["status"] == UNKNOWN:
        for key in (
            "implementation_commit",
            "implementation_script_sha256",
            "implementation_test_sha256",
        ):
            _expect(binding[key], UNKNOWN, label=f"UNKNOWN {key}")
    elif binding["status"] == BOUND:
        if HEX40.fullmatch(str(binding["implementation_commit"])) is None:
            raise BindingError("bound implementation commit is invalid")
        for key in (
            "implementation_script_sha256",
            "implementation_test_sha256",
        ):
            if HEX64.fullmatch(str(binding[key])) is None:
                raise BindingError(f"bound {key} is invalid")
    else:
        raise BindingError("implementation binding status is outside the closed enum")

    repository = _expect_exact_keys(
        value["repository_authority"],
        EXPECTED_REPOSITORY_KEYS,
        label="repository authority",
    )
    for key, expected in {
        "production_repo_root": os.fspath(PRODUCTION_REPO_ROOT),
        "branch": BRANCH,
        "base_commit": BASE_COMMIT,
        "implementation_commit_expected_parent": BASE_COMMIT,
        "binding_commit_expected_parent": "IMPLEMENTATION_COMMIT_FROM_BINDING",
        "implementation_commit_exact_changed_paths": IMPLEMENTATION_PATHS,
        "binding_commit_exact_changed_paths": BINDING_PATHS,
    }.items():
        _expect(repository[key], expected, label=f"repository {key}")
    descendant = repository["descendant_policy"]
    if type(descendant) is not dict or not descendant or any(
        item is not True for item in descendant.values()
    ):
        raise AuthorityError("repository descendant policy must be all true")

    consumer = value["consumer_authority"]
    expected_consumer = {
        "binding_scheme": (
            "FROZEN_D2_RECORD_TRUTH_CURRENT_DESCRIPTOR_DESCENDANT_VALIDATION_V1"
        ),
        "frozen_descriptor_commit": "c278f29a18b7858c85686fcec3857a992fd07d5f",
        "lifecycle_binding_commit": "6c42d8e1d75f70906afb7cde5704669b2c8ab6f7",
        "config_path": CONSUMER_CONFIG_REPO_PATH,
        "frozen_config_sha256": (
            "88fa21a08df60935f3d2d1bf44c6573889c22c110021146acf241fd92d6b5a13"
        ),
        "script_path": CONSUMER_SCRIPT_REPO_PATH,
        "script_sha256": (
            "9cd4411fcb02e1feed913b799296351e38ab9071b9506611318645e41b8dbbfe"
        ),
        "test_path": CONSUMER_TEST_REPO_PATH,
        "test_sha256": (
            "8e7b188cfa2e5015fa307acad980f9ff2f45145943384fcadb50d67b1263e1db"
        ),
        "science_core_sha256": (
            "13394ac6a9b9ec6e6241d0d9b1048ecfa5c90874c7447991fc2a8248a574c170"
        ),
        "frozen_descriptor_set_sha256": (
            "14223d0193e4b3a4a3c1d98a5894849dd429e6eed021ff98e6697e73ac286a40"
        ),
        "schema_version": (
            "route_a_v3_gse200304_dec019_reported_endpoint_a1_activation.v3"
        ),
        "protocol_id": CONSUMER_PROTOCOL_ID,
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "record_type": EVIDENCE_RECORD_TYPE,
        "source_bundle_id": "ROUTE_A_V3_GSE200304_PUBLISHED_ENDPOINT_A1_BUNDLE_V1",
        "source_bundle_root_or_target_sha256": (
            "bdb3f54f2aaf0f3f2b090563d712bbebb77ec2ca6de1c8e5f93a126690061a8f"
        ),
        "acceptance_rule": ACCEPTANCE_RULE,
        "frozen_config_is_record_schema_and_provenance_truth": True,
        "current_config_science_core_and_acceptance_identity_must_match": True,
        "current_consumer_production_authority_must_validate": True,
        "consumer_validate_gate_record_must_accept_every_record": True,
        "consumer_slot_gate_pass_must_return_exact_true": True,
    }
    _expect(consumer, expected_consumer, label="consumer authority")

    upstream = value["upstream_authority"]
    if type(upstream) is not dict:
        raise ProducerError("upstream authority is absent")
    for key, expected in {
        "authority_type": (
            "GSE200304_UPSTREAM_SOURCE_AUTHORITY_VIABILITY_EXACT6_V1"
        ),
        "bundle_id": "GSE200304_UPSTREAM_AUTHORITY_VIABILITY_V1",
        "absolute_root": (
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
            "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/"
            "GSE200304_UPSTREAM_AUTHORITY_VIABILITY_V1"
        ),
        "terminal_marker_final_output_target_sha256": (
            "ad9b64166586813d86c99de49589fff565dbe24eb48d7d6aeb07808fb390dfaa"
        ),
        "producer_binding_commit": BASE_COMMIT,
        "producer_implementation_commit": (
            "7e29c13ca778ffa27f3725f4bd1ea270630db044"
        ),
        "producer_config_sha256": (
            "c52688866026122488f1d8eef8d0bffebf864b99d78ddcc40c39a26221da76a1"
        ),
        "producer_script_sha256": (
            "525635da3d84183e325a17f00fc7cece1517acbd9ce317c2cc4e26a4ba38f03d"
        ),
        "producer_test_sha256": (
            "78bca039152874a09dd6a31a0789b712c72b45e60cb9e99e72391809a1bd7035"
        ),
        "producer_config_core_sha256": (
            "49388c339dad75107149911ceb0fa078ec2ff0729951442a4aa034ba82c90cba"
        ),
        "audit_member_name": "GSE200304_UPSTREAM_AUTHORITY_VIABILITY_AUDIT.json",
        "checksums_member_name": CHECKSUMS_MEMBER_NAME,
        "terminal_marker_name": MARKER_MEMBER_NAME,
    }.items():
        _expect(upstream[key], expected, label=f"upstream {key}")
    members = upstream.get("exact_members")
    if type(members) is not list or len(members) != 6:
        raise ProducerError("upstream exact6 member registry differs")
    if [item.get("name") for item in members] != [
        "PMC10540565_EUROPE_PMC_FULLTEXT.xml",
        "GSE200302_family.soft.gz",
        "GSE200302_log2_cpm_counts_all_samples.txt.gz",
        "GSE200304_UPSTREAM_AUTHORITY_VIABILITY_AUDIT.json",
        "SHA256SUMS",
        "PUBLICATION_COMMIT.json",
    ]:
        raise ProducerError("upstream exact6 member names/order differ")
    for item in members:
        if (
            type(item) is not dict
            or set(item) != {"name", "bytes", "sha256", "read_mode"}
            or type(item["bytes"]) is not int
            or item["bytes"] <= 0
            or HEX64.fullmatch(str(item["sha256"])) is None
        ):
            raise ProducerError("upstream exact6 member authority is invalid")
    _expect(
        upstream["raw_source_member_names_hash_only"],
        [item["name"] for item in members[:3]],
        label="raw source hash-only members",
    )
    if any(item["read_mode"] != "SAME_FD_STREAM_HASH_ONLY" for item in members[:3]):
        raise ProducerError("raw source member is not hash-only")

    semantics = value["expected_audit_semantics"]
    if type(semantics) is not dict:
        raise ProducerError("expected audit semantics are absent")
    for key, expected in {
        "status": UPSTREAM_AUDIT_STATUS,
        "mode": UPSTREAM_AUDIT_MODE,
    }.items():
        _expect(semantics.get(key), expected, label=f"audit semantics {key}")
    _expect(
        semantics["endpoint_crosswalk"]["effect_definition"],
        "log2((mutant total-poly/total-RNA)/(WT total-poly/total-RNA))",
        label="endpoint equation",
    )
    _expect(
        semantics["replicate_branch"]["standard_error_status"],
        "ABSENT_NOT_REPORTED_NOT_DERIVED_NOT_USED",
        label="standard error semantics",
    )
    _expect(
        semantics["private_only_rights"]["authorized_scope"],
        "PRIVATE_CANONICAL_ONLY",
        label="private-only rights scope",
    )
    _expect(
        semantics["biological_group_authority"]["status"],
        "BLOCKED_PENDING_AUTHOR_SOURCE_GROUP_MAPPING_ROOT",
        label="unchanged group blocker",
    )
    _expect(
        semantics["unchanged_gates"],
        {
            "changed_by_this_audit": False,
            "checkpoint_specific_exposure": UNKNOWN,
            "outcome_blind_split_leakage": "NOT_RUN",
            "prefrozen_power_precision": "NOT_RUN",
        },
        label="unchanged gates",
    )
    if any(
        semantics["decision_boundary"][key] != expected
        for key, expected in {
            "gate_records_written": 0,
            "consumer_run": False,
            "adjudicator_run": False,
            "qualified": False,
            "canonical_record_count": 0,
            "canonical_materialization_allowed": False,
            "training_allowed": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
        }.items()
    ):
        raise ProducerError("audit decision boundary was loosened")

    _expect(value["pass_gate_records"], GATE_SPECS, label="PASS gate specs")
    policy = value["record_policy"]
    for key, expected in {
        "status": "PASS",
        "accepted": True,
        "aggregate_only": True,
        "unknown_fields": [],
        "reason_codes": [],
        "privacy": PRIVACY,
        "pass_slot_indices_after_descriptor_binding": [0, 1, 3, 5],
        "negative_slot_indices_after_descriptor_binding": [2, 4, 6, 7],
        "ordinary_study_contribution_delta": 0,
        "a1_study_contribution_delta": 0,
        "true_a2_study_contribution_delta": 0,
        "canonical_record_count_delta": 0,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
    }.items():
        _expect(policy[key], expected, label=f"record policy {key}")

    output = value["output_contract"]
    for key, expected in {
        "trusted_final_directory": (
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
            "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/"
            "GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_V1"
        ),
        "primary_publication_mode": PRIMARY_PUBLICATION_MODE,
        "fallback_publication_mode": FALLBACK_PUBLICATION_MODE,
        "exact_gate_member_names": list(GATE_MEMBER_NAMES),
        "audit_member_name": AUDIT_MEMBER_NAME,
        "checksums_member_name": CHECKSUMS_MEMBER_NAME,
        "terminal_commit_marker_filename": MARKER_MEMBER_NAME,
        "exact_published_member_names": list(PUBLISHED_MEMBER_NAMES),
        "descriptor_binding_scope": "THREE_GATE_JSON_FILES_ONLY",
        "partial_recovery_mode": "EXPLICIT_CLI_FLAG_ONLY",
    }.items():
        _expect(output[key], expected, label=f"output contract {key}")
    for key in (
        "descriptor_binder_must_validate_exact6_before_consuming_exact3",
        "terminal_commit_marker_written_last",
        "terminal_commit_marker_is_only_acceptance_point",
        "existing_exact_is_idempotent",
        "root_to_leaf_symlink_rejection_required",
        "single_link_regular_file_required",
        "same_fd_bytes_and_hash_required",
        "fsync_files_directory_and_parent_required",
        "post_fsync_reopen_exact6_required",
        "post_marker_failure_must_report_committed_truth",
    ):
        _expect(output[key], True, label=f"output contract {key}")
    _expect(output["overwrite_allowed"], False, label="overwrite policy")


def validate_implementation_binding(config: Mapping[str, Any]) -> None:
    binding = config["implementation_binding"]
    if binding["status"] != BOUND:
        raise BindingError(binding["blocker_if_unbound"])
    if HEX40.fullmatch(str(binding["implementation_commit"])) is None:
        raise BindingError("implementation commit is not bound")
    for key in (
        "implementation_script_sha256",
        "implementation_test_sha256",
    ):
        if HEX64.fullmatch(str(binding[key])) is None:
            raise BindingError(f"{key} is not bound")


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
    i_binding = i_config["implementation_binding"]
    b_binding = b_config["implementation_binding"]
    _expect(i_binding["status"], UNKNOWN, label="I status")
    for key in (
        "implementation_commit",
        "implementation_script_sha256",
        "implementation_test_sha256",
    ):
        _expect(i_binding[key], UNKNOWN, label=f"I {key}")
    _expect(b_binding["status"], BOUND, label="B status")
    _expect(
        b_binding["implementation_commit"],
        implementation_commit,
        label="B implementation commit",
    )
    _expect(
        b_binding["implementation_script_sha256"],
        implementation_script_sha256,
        label="B script SHA",
    )
    _expect(
        b_binding["implementation_test_sha256"],
        implementation_test_sha256,
        label="B test SHA",
    )
    _expect(
        _scalar_differences(i_config, b_config),
        set(BINDING_SCALAR_PATHS),
        label="I-to-B exact four-scalar diff",
    )


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", os.fspath(repo), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise AuthorityError(
            f"git command failed: {' '.join(arguments)}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _git_bytes(repo: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", os.fspath(repo), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise AuthorityError(f"git byte command failed: {' '.join(arguments)}")
    return result.stdout


def _commit_paths(repo: Path, commit: str) -> list[str]:
    return sorted(
        line
        for line in _git(
            repo,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            commit,
        ).splitlines()
        if line
    )


def _single_parent(repo: Path, commit: str, expected: str, *, label: str) -> None:
    values = _git(repo, "rev-list", "--parents", "-n", "1", commit).split()
    if values != [commit, expected]:
        raise AuthorityError(f"{label} is not the required direct child")


def _require_ancestor(repo: Path, ancestor: str, descendant: str, *, label: str) -> None:
    result = subprocess.run(
        ["git", "-C", os.fspath(repo), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise AuthorityError(f"{label} ancestry differs")


def _git_regular_blob(repo: Path, commit: str, path: str, *, label: str) -> bytes:
    mode = _git(repo, "ls-tree", commit, "--", path).split()
    if len(mode) < 4 or mode[0] not in {"100644", "100755"} or mode[1] != "blob":
        raise AuthorityError(f"{label} is not a regular Git blob")
    return _git_bytes(repo, "show", f"{commit}:{path}")


def validate_production_authority(
    config: Mapping[str, Any],
    config_payload: bytes,
    *,
    config_path: Path = PRODUCTION_CONFIG_PATH,
    script_path: Path | None = None,
) -> dict[str, str]:
    """Prove base -> exact3 I -> config-only B -> clean pushed descendants."""

    validate_static_config(config)
    validate_implementation_binding(config)
    executed = (script_path or Path(os.path.abspath(__file__))).resolve()
    if config_path.resolve() != PRODUCTION_CONFIG_PATH or executed != (
        PRODUCTION_REPO_ROOT / SCRIPT_REPO_PATH
    ):
        raise AuthorityError("production entrypoint path differs")
    repo = Path(config["repository_authority"]["production_repo_root"])
    head = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "rev-parse", "--abbrev-ref", "HEAD") != BRANCH:
        raise AuthorityError("production branch differs")
    if _git(repo, "status", "--porcelain"):
        raise AuthorityError("production worktree is not clean")
    if _git(repo, "rev-parse", "--abbrev-ref", "@{upstream}") != f"origin/{BRANCH}":
        raise AuthorityError("production upstream ref differs")
    if _git(repo, "rev-parse", "@{upstream}") != head:
        raise AuthorityError("production HEAD is not exactly pushed")

    binding = config["implementation_binding"]
    implementation = binding["implementation_commit"]
    _single_parent(repo, implementation, BASE_COMMIT, label="producer I")
    if _commit_paths(repo, implementation) != sorted(IMPLEMENTATION_PATHS):
        raise AuthorityError("producer I is not the exact three-file commit")
    lineage = [
        line
        for line in _git(
            repo,
            "rev-list",
            "--ancestry-path",
            "--reverse",
            f"{implementation}..{head}",
        ).splitlines()
        if line
    ]
    if not lineage:
        raise AuthorityError("producer config-only B commit is absent")
    binding_commit = lineage[0]
    _single_parent(repo, binding_commit, implementation, label="producer B")
    if _commit_paths(repo, binding_commit) != BINDING_PATHS:
        raise AuthorityError("producer B is not exact config-only")
    _require_ancestor(repo, binding_commit, head, label="producer B to current")

    i_config_payload = _git_regular_blob(
        repo, implementation, CONFIG_REPO_PATH, label="producer I config"
    )
    b_config_payload = _git_regular_blob(
        repo, binding_commit, CONFIG_REPO_PATH, label="producer B config"
    )
    if _git_regular_blob(repo, head, CONFIG_REPO_PATH, label="current producer config") != b_config_payload:
        raise AuthorityError("producer config drifted after B")
    if config_payload != b_config_payload:
        raise AuthorityError("running producer config differs from B")
    i_config = strict_json(i_config_payload, label="producer I config")
    b_config = strict_json(b_config_payload, label="producer B config")
    script_i = _git_regular_blob(repo, implementation, SCRIPT_REPO_PATH, label="producer I script")
    test_i = _git_regular_blob(repo, implementation, TEST_REPO_PATH, label="producer I test")
    validate_i_to_b_transition(
        i_config,
        b_config,
        implementation_commit=implementation,
        implementation_script_sha256=sha256(script_i),
        implementation_test_sha256=sha256(test_i),
    )
    for path, expected_payload, expected_sha in (
        (SCRIPT_REPO_PATH, script_i, binding["implementation_script_sha256"]),
        (TEST_REPO_PATH, test_i, binding["implementation_test_sha256"]),
    ):
        if sha256(expected_payload) != expected_sha:
            raise AuthorityError(f"producer binding SHA differs: {path}")
        current = _git_regular_blob(repo, head, path, label=f"current producer {path}")
        worktree = _read_repo_regular(repo, path, label=f"producer worktree {path}")
        if current != expected_payload or worktree != expected_payload:
            raise AuthorityError(f"producer file drifted after B: {path}")
    if _read_repo_regular(repo, CONFIG_REPO_PATH, label="producer worktree config") != b_config_payload:
        raise AuthorityError("producer worktree config differs from B")
    return {
        "status": "PASS_BOUND_IMPLEMENTATION_DESCRIPTOR_DESCENDANTS_ALLOWED",
        "base_commit": BASE_COMMIT,
        "implementation_commit": implementation,
        "binding_commit": binding_commit,
        "current_head": head,
    }


def _safe_relative_path(value: str, *, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ScopeViolation(f"{label} is not a safe repository-relative path")
    return path


def _open_directory_root_to_leaf(path: Path, *, label: str) -> int:
    if not path.is_absolute() or any(
        part in {"", ".", ".."} for part in path.parts[1:]
    ):
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


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _directory_identity(descriptor: int) -> tuple[int, int]:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ScopeViolation("pinned descriptor is not a directory")
    return metadata.st_dev, metadata.st_ino


def _read_exact_member_at(
    directory_fd: int,
    name: str,
    *,
    expected_bytes: int,
    expected_sha256: str,
    collect: bool,
    label: str,
) -> bytes | None:
    if Path(name).name != name:
        raise ScopeViolation(f"{label} basename is unsafe")
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise InputIntegrityError(f"{label} cannot be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != expected_bytes
        ):
            raise InputIntegrityError(f"{label} type/link/size differs")
        identity = _file_identity(before)
        digest = hashlib.sha256()
        count = 0
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            count += len(block)
            if count > expected_bytes:
                raise InputIntegrityError(f"{label} grew while read")
            digest.update(block)
            if collect:
                chunks.append(block)
        after = os.fstat(descriptor)
        if _file_identity(after) != identity:
            raise InputIntegrityError(f"{label} changed during same-FD read")
        path_metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if _file_identity(path_metadata) != identity:
            raise InputIntegrityError(f"{label} path binding changed")
        if count != expected_bytes or digest.hexdigest() != expected_sha256:
            raise InputIntegrityError(f"{label} byte/hash authority differs")
        return b"".join(chunks) if collect else None
    finally:
        os.close(descriptor)


def _read_regular_at(directory_fd: int, name: str, *, label: str) -> bytes:
    if Path(name).name != name:
        raise ScopeViolation(f"{label} basename is unsafe")
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise ScopeViolation(f"{label} cannot be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ScopeViolation(f"{label} is not a single-link regular file")
        identity = _file_identity(before)
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        if _file_identity(after) != identity:
            raise ScopeViolation(f"{label} changed during same-FD read")
        path_metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if _file_identity(path_metadata) != identity:
            raise ScopeViolation(f"{label} path binding changed")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_repo_regular(repo: Path, relative: str, *, label: str) -> bytes:
    path = _safe_relative_path(relative, label=label)
    parent_fd = _open_directory_root_to_leaf(
        repo.joinpath(*path.parts[:-1]), label=f"{label} parent"
    )
    try:
        return _read_regular_at(parent_fd, path.name, label=label)
    finally:
        os.close(parent_fd)


def _consumer_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    evidence = config.get("evidence_contract")
    if type(evidence) is not dict:
        raise AuthorityError("consumer evidence contract is absent")
    provenance = evidence.get("gate_record_provenance_contract")
    if type(provenance) is not dict:
        raise AuthorityError("consumer gate provenance contract is absent")
    acceptance = provenance.get("acceptance_authority")
    if type(acceptance) is not dict:
        raise AuthorityError("consumer acceptance authority is absent")
    return {
        "schema_version": config.get("schema_version"),
        "protocol_id": config.get("protocol_id"),
        "contract_id": config.get("contract_id"),
        "phase_id": config.get("phase_id"),
        "dataset_id": config.get("dataset_id"),
        "decision_id": config.get("decision_id"),
        "evidence_schema_version": evidence.get("evidence_schema_version"),
        "acceptance_rule": acceptance.get("rule"),
    }


def _compile_consumer_module(
    payload: bytes,
    *,
    canonical_path: Path,
) -> types.ModuleType:
    module = types.ModuleType("_verified_gse200304_dec019_consumer_for_pass_pack")
    module.__file__ = os.fspath(canonical_path)
    module.__spec__ = importlib.util.spec_from_loader(module.__name__, loader=None)
    exec(compile(payload, module.__file__, "exec"), module.__dict__)
    required = (
        "validate_static_config",
        "config_core_sha256",
        "descriptor_set_sha256",
        "validate_production_authority",
        "_validate_gate_record",
        "_slot_gate_pass",
    )
    if any(not callable(getattr(module, name, None)) for name in required):
        raise AuthorityError("verified consumer module lacks a required API")
    if (
        getattr(module, "EVIDENCE_SCHEMA_VERSION", None) != EVIDENCE_SCHEMA_VERSION
        or getattr(module, "EVIDENCE_RECORD_TYPE", None) != EVIDENCE_RECORD_TYPE
    ):
        raise AuthorityError("verified consumer record identity differs")
    return module


def _validate_consumer_config_identity(
    consumer: Mapping[str, Any],
    module: types.ModuleType,
    authority: Mapping[str, Any],
    *,
    label: str,
) -> None:
    module.validate_static_config(consumer)
    expected_identity = {
        "schema_version": authority["schema_version"],
        "protocol_id": authority["protocol_id"],
        "contract_id": CONTRACT_ID,
        "phase_id": PHASE_ID,
        "dataset_id": DATASET_ID,
        "decision_id": DECISION_ID,
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "acceptance_rule": ACCEPTANCE_RULE,
    }
    _expect(_consumer_identity(consumer), expected_identity, label=f"{label} identity")
    _expect(
        module.config_core_sha256(consumer),
        authority["science_core_sha256"],
        label=f"{label} science core",
    )
    _expect(
        consumer["implementation_binding"]["config_core_sha256"],
        authority["science_core_sha256"],
        label=f"{label} stored science core",
    )
    predecessor = consumer["evidence_contract"]["required_predecessor_authority"]
    _expect(
        predecessor["bundle_id"],
        authority["source_bundle_id"],
        label=f"{label} predecessor bundle",
    )
    _expect(
        predecessor["terminal_marker_final_output_target_sha256"],
        authority["source_bundle_root_or_target_sha256"],
        label=f"{label} predecessor target",
    )
    acceptance = consumer["evidence_contract"]["gate_record_provenance_contract"][
        "acceptance_authority"
    ]
    _expect(acceptance["rule"], ACCEPTANCE_RULE, label=f"{label} acceptance rule")


def load_verified_consumer(
    config: Mapping[str, Any],
    *,
    repo: Path,
    production: bool,
    current_head: str | None = None,
    frozen_config_payload: bytes | None = None,
) -> tuple[dict[str, Any], dict[str, Any], types.ModuleType, dict[str, Any] | None]:
    """Load frozen D2 record truth and current descriptor-descendant authority."""

    authority = config["consumer_authority"]
    if production:
        if current_head is None:
            raise AuthorityError("current production HEAD is absent")
        _require_ancestor(
            repo,
            authority["lifecycle_binding_commit"],
            authority["frozen_descriptor_commit"],
            label="consumer lifecycle B4 to frozen D2",
        )
        _require_ancestor(
            repo,
            authority["frozen_descriptor_commit"],
            current_head,
            label="consumer frozen D2 to current",
        )
        frozen_payload = _git_regular_blob(
            repo,
            authority["frozen_descriptor_commit"],
            authority["config_path"],
            label="frozen D2 consumer config",
        )
    elif frozen_config_payload is not None:
        frozen_payload = frozen_config_payload
    else:
        frozen_payload = _read_repo_regular(
            repo, authority["config_path"], label="synthetic frozen consumer config"
        )
    if sha256(frozen_payload) != authority["frozen_config_sha256"]:
        raise AuthorityError("frozen D2 consumer config SHA differs")

    script_payload = _read_repo_regular(
        repo, authority["script_path"], label="consumer script"
    )
    test_payload = _read_repo_regular(
        repo, authority["test_path"], label="consumer test"
    )
    if sha256(script_payload) != authority["script_sha256"]:
        raise AuthorityError("consumer script SHA differs")
    if sha256(test_payload) != authority["test_sha256"]:
        raise AuthorityError("consumer test SHA differs")
    if production and current_head is not None:
        for path, expected in (
            (authority["script_path"], script_payload),
            (authority["test_path"], test_payload),
        ):
            if _git_regular_blob(repo, current_head, path, label=f"current {path}") != expected:
                raise AuthorityError(f"consumer authority drifted: {path}")

    current_payload = _read_repo_regular(
        repo, authority["config_path"], label="current consumer config"
    )
    if production and current_head is not None:
        if _git_regular_blob(
            repo, current_head, authority["config_path"], label="current consumer config"
        ) != current_payload:
            raise AuthorityError("consumer worktree config differs from current HEAD")
    frozen = strict_json(frozen_payload, label="frozen D2 consumer config")
    current = strict_json(current_payload, label="current consumer config")
    module = _compile_consumer_module(
        script_payload,
        canonical_path=repo / authority["script_path"],
    )
    _validate_consumer_config_identity(
        frozen, module, authority, label="frozen D2 consumer"
    )
    _validate_consumer_config_identity(
        current, module, authority, label="current consumer"
    )
    _expect(
        module.descriptor_set_sha256(frozen),
        authority["frozen_descriptor_set_sha256"],
        label="frozen D2 descriptor set",
    )
    # D3 may change descriptor triples/status only.  Science, record schema,
    # predecessor, acceptance, fact keys, and basenames remain exact.
    _expect(
        current["evidence_contract"]["required_predecessor_authority"],
        frozen["evidence_contract"]["required_predecessor_authority"],
        label="current/frozen predecessor authority",
    )
    _expect(
        current["evidence_contract"]["gate_record_provenance_contract"][
            "acceptance_authority"
        ],
        frozen["evidence_contract"]["gate_record_provenance_contract"][
            "acceptance_authority"
        ],
        label="current/frozen acceptance authority",
    )
    frozen_slots = {
        slot["slot_id"]: slot for slot in frozen["evidence_contract"]["slots"]
    }
    current_slots = {
        slot["slot_id"]: slot for slot in current["evidence_contract"]["slots"]
    }
    for spec in GATE_SPECS:
        gate_id = spec["gate_id"]
        if gate_id not in frozen_slots or gate_id not in current_slots:
            raise AuthorityError(f"consumer slot is absent: {gate_id}")
        _expect(
            frozen_slots[gate_id]["allowed_basename"],
            spec["allowed_basename"],
            label=f"frozen consumer basename {gate_id}",
        )
        _expect(
            current_slots[gate_id]["allowed_basename"],
            spec["allowed_basename"],
            label=f"current consumer basename {gate_id}",
        )
        _expect(
            set(module.FACT_KEYS[gate_id]),
            set(spec["facts"]),
            label=f"consumer fact schema {gate_id}",
        )

    production_result: dict[str, Any] | None = None
    if production:
        try:
            production_result = module.validate_production_authority(current)
        except Exception as exc:
            raise AuthorityError("current consumer production authority failed") from exc
        if type(production_result) is not dict:
            raise AuthorityError("consumer production authority result is invalid")
        _expect(
            production_result.get("current_head"),
            current_head,
            label="consumer production current HEAD",
        )
        _expect(
            production_result.get("science_core_sha256"),
            authority["science_core_sha256"],
            label="consumer production science core",
        )
    return frozen, current, module, production_result


def validate_upstream_closed_audit(
    audit: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    """Close the three PASS predicates without reopening any scientific source."""

    for key, expected in {
        "schema_version": UPSTREAM_AUDIT_SCHEMA_VERSION,
        "record_type": UPSTREAM_AUDIT_RECORD_TYPE,
        "protocol_id": UPSTREAM_PROTOCOL_ID,
        "contract_id": CONTRACT_ID,
        "phase_id": PHASE_ID,
        "dataset_id": DATASET_ID,
        "decision_id": DECISION_ID,
        "status": UPSTREAM_AUDIT_STATUS,
        "mode": UPSTREAM_AUDIT_MODE,
    }.items():
        _expect(audit.get(key), expected, label=f"upstream audit {key}")
    expected = config["expected_audit_semantics"]
    for key in (
        "endpoint_crosswalk",
        "replicate_branch",
        "private_only_rights",
        "biological_group_authority",
        "unchanged_gates",
        "decision_boundary",
        "execution_boundary",
    ):
        _expect(audit.get(key), expected[key], label=f"upstream audit {key}")

    upstream = config["upstream_authority"]
    binding = audit.get("producer_binding")
    if type(binding) is not dict:
        raise InputIntegrityError("upstream audit producer binding is absent")
    for key, expected_value in {
        "status": "PASS_BOUND_IMPLEMENTATION",
        "implementation_commit": upstream["producer_implementation_commit"],
        "binding_commit": upstream["producer_binding_commit"],
        "implementation_script_sha256": upstream["producer_script_sha256"],
        "implementation_test_sha256": upstream["producer_test_sha256"],
        "config_core_sha256": upstream["producer_config_core_sha256"],
    }.items():
        _expect(binding.get(key), expected_value, label=f"upstream producer {key}")

    official = audit.get("official_source_authority")
    if type(official) is not dict:
        raise InputIntegrityError("official source authority is absent")
    for key, expected_value in {
        "status": "PASS_EXACT_THREE_OFFICIAL_SOURCE_SNAPSHOTS",
        "network_download_count": 3,
        "verbatim_source_member_count": 3,
    }.items():
        _expect(official.get(key), expected_value, label=f"official source {key}")
    sources = official.get("sources")
    member_specs = upstream["exact_members"][:3]
    if type(sources) is not list or len(sources) != 3:
        raise InputIntegrityError("official source audit list differs")
    for source, member in zip(sources, member_specs):
        if type(source) is not dict:
            raise InputIntegrityError("official source audit item differs")
        for key, expected_value in {
            "output_name": member["name"],
            "bytes": member["bytes"],
            "sha256": member["sha256"],
            "same_fd_size_and_hash_verified": True,
        }.items():
            _expect(source.get(key), expected_value, label=f"official source {key}")

    jats = audit.get("jats_authority")
    if type(jats) is not dict:
        raise InputIntegrityError("JATS authority is absent")
    for key, expected_value in {
        "status": "PASS_EXACT_JATS_IDENTITY_LICENSE_LINKAGE_AND_PARAGRAPHS",
        "identity": {
            "doi": "10.1016/j.celrep.2023.112840",
            "pmcid": "PMC10540565",
            "pmid": "37516102",
        },
        "license_ref": "https://creativecommons.org/licenses/by/4.0/",
        "license_text_verified": True,
        "normalized_paragraphs": {
            "endpoint_and_six_biological_replicates": {
                "utf8_bytes": 798,
                "sha256": (
                    "45dd0d8b9c7976748615f2c7b620bcc403fe7bf5c832b2dbb8516d758b27ac3d"
                ),
            },
            "ratio_of_ratios_methods": {
                "utf8_bytes": 534,
                "sha256": (
                    "0fb681090cf10597751369a11ab72fa19552e14ae9ce579d8b350f135b274fd2"
                ),
            },
        },
    }.items():
        _expect(jats.get(key), expected_value, label=f"JATS authority {key}")
    supplement_counts = jats.get("supplement_table_cross_reference_counts")
    if (
        type(supplement_counts) is not dict
        or set(supplement_counts) != {"Table S2", "Table S3"}
        or any(type(value) is not int or value < 1 for value in supplement_counts.values())
    ):
        raise InputIntegrityError("JATS S2/S3 cross-reference authority differs")

    soft = audit.get("geo_soft_authority")
    expected_soft = {
        "status": "PASS_EXACT_GSE200302_SUBSERIES_AND_24_SAMPLE_ROLE_GRID",
        "series_accession": "GSE200302",
        "subseries_of_gse200304": True,
        "sample_count": 24,
        "role_counts": {
            "High_Poly": 6,
            "Low_Poly": 6,
            "Total_RNA": 6,
            "pDNA": 6,
        },
        "replicates_per_role": [1, 2, 3, 4, 5, 6],
        "sample_supplementary_none_count": 24,
        "series_supplementary_file_count": 2,
        "series_processed_matrix_reference_count": 1,
        "processed_matrix_payload_embedded_in_soft": False,
        "geo_dataset_restriction_field_count": 0,
    }
    _expect(soft, expected_soft, label="GEO SOFT closed authority")

    matrix = audit.get("processed_matrix_authority")
    expected_matrix = {
        "status": "PASS_EXACT_6772_BY_61_MATRIX_AND_S3_MEMBERSHIP_CROSSCHECK",
        "header_field_count": 61,
        "value_field_count": 60,
        "row_count": 6772,
        "row_width_error_count": 0,
        "duplicate_key_count": 0,
        "missing_value_count": 0,
        "invalid_numeric_count": 0,
        "closed_role_geometry_count": 60,
        "required_endpoint_families": ["High_Poly", "Low_Poly", "Total_RNA"],
        "required_arms": ["WT", "Mutant"],
        "required_replicates": [1, 2, 3, 4, 5, 6],
        "endpoint_excluded_families": ["80S_RNA", "pDNA"],
        "matrix_key_set_equals_s3_key_set": True,
        "matrix_key_count": 6772,
        "s3_key_count": 6772,
        "finite_totalpoly_key_count": 6547,
        "matrix_covers_every_finite_totalpoly_key": True,
        "standard_error_status": "ABSENT_NOT_REPORTED_NOT_DERIVED_NOT_USED",
        "p_or_fdr_back_calculation_used": False,
    }
    _expect(matrix, expected_matrix, label="processed matrix closed authority")

    predecessor = audit.get("predecessor_authority")
    if type(predecessor) is not dict:
        raise InputIntegrityError("upstream predecessor aggregate authority is absent")
    for key, expected_value in {
        "published_endpoint_config_sha256": (
            "92fc3a3859f7a8949ace67fa4b03a14e8ad102eb257d4f95cace01ea535b41af"
        ),
        "source_exact7_member_count": 7,
        "published_endpoint_bundle_member_count": 5,
        "table_s3_selective_pair_count": 6772,
        "table_s3_finite_totalpoly_pair_count": 6547,
        "table_s3_gene_column_selected_or_persisted": False,
        "table_s3_translation_significance_selected_or_persisted": False,
    }.items():
        _expect(predecessor.get(key), expected_value, label=f"upstream predecessor {key}")
    for key in (
        "published_endpoint_trio_manifest_sha256",
        "source_exact7_manifest_sha256",
        "published_endpoint_bundle_manifest_sha256",
    ):
        if HEX64.fullmatch(str(predecessor.get(key))) is None:
            raise InputIntegrityError(f"upstream predecessor {key} is not bound")

    privacy = audit.get("privacy")
    _expect(
        privacy,
        {
            "derived_row_payload": False,
            "derived_sequence_payload": False,
            "derived_row_identifier_payload": False,
            "derived_effect_value_payload": False,
            "derived_gene_payload": False,
            "verbatim_raw_source_members_are_not_derived_payload": True,
        },
        label="upstream audit privacy",
    )


def _validate_upstream_marker(
    marker: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    upstream = config["upstream_authority"]
    specs = {item["name"]: item for item in upstream["exact_members"]}
    preterminal = sorted(name for name in specs if name != MARKER_MEMBER_NAME)
    expected_keys = {
        "schema_version",
        "record_type",
        "protocol_id",
        "contract_id",
        "dataset_id",
        "bundle_id",
        "preterminal_member_names",
        "preterminal_member_count",
        "exact_final_member_count",
        "sha256sums_sha256",
        "final_output_target_sha256",
        "publication_mode",
        "committed",
        "terminal_marker_written_last",
        "no_overwrite",
        "partial_default",
    }
    _expect_exact_keys(marker, expected_keys, label="upstream terminal marker")
    for key, expected in {
        "schema_version": "1.0.0",
        "record_type": UPSTREAM_MARKER_RECORD_TYPE,
        "protocol_id": UPSTREAM_PROTOCOL_ID,
        "contract_id": CONTRACT_ID,
        "dataset_id": DATASET_ID,
        "bundle_id": upstream["bundle_id"],
        "preterminal_member_names": preterminal,
        "preterminal_member_count": 5,
        "exact_final_member_count": 6,
        "sha256sums_sha256": specs[CHECKSUMS_MEMBER_NAME]["sha256"],
        "final_output_target_sha256": upstream[
            "terminal_marker_final_output_target_sha256"
        ],
        "publication_mode": UPSTREAM_PUBLICATION_MODE,
        "committed": True,
        "terminal_marker_written_last": True,
        "no_overwrite": True,
        "partial_default": UPSTREAM_PARTIAL_DEFAULT,
    }.items():
        _expect(marker[key], expected, label=f"upstream terminal marker {key}")


def inspect_upstream_bundle(
    config: Mapping[str, Any],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Hash all exact6 members; decode only audit and integrity metadata."""

    upstream = config["upstream_authority"]
    bundle_root = root or Path(upstream["absolute_root"])
    directory_fd = _open_directory_root_to_leaf(
        bundle_root, label="upstream exact6 root"
    )
    collected: dict[str, bytes] = {}
    try:
        expected_names = sorted(item["name"] for item in upstream["exact_members"])
        if sorted(os.listdir(directory_fd)) != expected_names:
            raise InputIntegrityError("upstream bundle is not exact6")
        raw_names = set(upstream["raw_source_member_names_hash_only"])
        for item in upstream["exact_members"]:
            payload = _read_exact_member_at(
                directory_fd,
                item["name"],
                expected_bytes=item["bytes"],
                expected_sha256=item["sha256"],
                collect=item["name"] not in raw_names,
                label=f"upstream exact6 member {item['name']}",
            )
            if payload is not None:
                collected[item["name"]] = payload
    finally:
        os.close(directory_fd)

    specs = {item["name"]: item for item in upstream["exact_members"]}
    audit_name = upstream["audit_member_name"]
    expected_sums = "".join(
        f"{specs[name]['sha256']}  {name}\n"
        for name in sorted((*upstream["raw_source_member_names_hash_only"], audit_name))
    ).encode("ascii")
    if collected[CHECKSUMS_MEMBER_NAME] != expected_sums:
        raise InputIntegrityError("upstream SHA256SUMS content differs")
    marker = strict_json(
        collected[MARKER_MEMBER_NAME], label="upstream publication marker"
    )
    _validate_upstream_marker(marker, config)
    audit = strict_json(collected[audit_name], label="upstream closed audit")
    validate_upstream_closed_audit(audit, config)
    return {
        "bundle_id": upstream["bundle_id"],
        "final_output_target_sha256": upstream[
            "terminal_marker_final_output_target_sha256"
        ],
        "exact_member_count": 6,
        "raw_source_same_fd_hash_only_count": 3,
        "decoded_raw_source_count": 0,
        "decoded_semantic_input_count": 1,
        "semantic_audit_sha256": specs[audit_name]["sha256"],
        "audit": audit,
    }


def build_gate_records(
    config: Mapping[str, Any],
    frozen_consumer: Mapping[str, Any],
    consumer_module: types.ModuleType,
) -> dict[str, bytes]:
    """Build three PASS records and require both consumer APIs to pass exactly."""

    validate_implementation_binding(config)
    binding = config["implementation_binding"]
    evidence = frozen_consumer["evidence_contract"]
    predecessor = evidence["required_predecessor_authority"]
    acceptance = evidence["gate_record_provenance_contract"]["acceptance_authority"]
    slots = {slot["slot_id"]: slot for slot in evidence["slots"]}
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
    for spec in config["pass_gate_records"]:
        gate_id = spec["gate_id"]
        record = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "record_type": EVIDENCE_RECORD_TYPE,
            "contract_id": CONTRACT_ID,
            "decision_id": DECISION_ID,
            "dataset_id": DATASET_ID,
            "gate_id": gate_id,
            "status": "PASS",
            "accepted": True,
            "aggregate_only": True,
            "privacy": copy.deepcopy(PRIVACY),
            "provenance": copy.deepcopy(provenance),
            "facts": copy.deepcopy(spec["facts"]),
            "unknown_fields": [],
            "reason_codes": [],
        }
        payload = json_bytes(record)
        try:
            accepted = consumer_module._validate_gate_record(
                payload, slots[gate_id], frozen_consumer
            )
            passed = consumer_module._slot_gate_pass(gate_id, accepted["facts"])
        except Exception as exc:
            raise AuthorityError(f"consumer rejected PASS record: {gate_id}") from exc
        if accepted != record:
            raise AuthorityError(f"consumer altered PASS record: {gate_id}")
        if passed is not True:
            raise AuthorityError(f"consumer slot gate is not exact True: {gate_id}")
        payloads[spec["allowed_basename"]] = payload
    if tuple(sorted(payloads)) != GATE_MEMBER_NAMES:
        raise ProducerError("generated PASS record set is not exact3")
    return payloads


def build_pack_audit(
    config: Mapping[str, Any],
    upstream_summary: Mapping[str, Any],
) -> dict[str, Any]:
    policy = config["record_policy"]
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_AUDIT_V1",
        "protocol_id": PROTOCOL_ID,
        "contract_id": CONTRACT_ID,
        "phase_id": PHASE_ID,
        "dataset_id": DATASET_ID,
        "decision_id": DECISION_ID,
        "status": "PASS_EXACT_THREE_CONSUMER_ACCEPTED_GATES_NO_ADJUDICATION",
        "producer_implementation_commit": config["implementation_binding"][
            "implementation_commit"
        ],
        "producer_config_core_sha256": config["implementation_binding"][
            "config_core_sha256"
        ],
        "consumer_frozen_descriptor_commit": config["consumer_authority"][
            "frozen_descriptor_commit"
        ],
        "consumer_frozen_config_sha256": config["consumer_authority"][
            "frozen_config_sha256"
        ],
        "consumer_science_core_sha256": config["consumer_authority"][
            "science_core_sha256"
        ],
        "upstream_bundle_id": upstream_summary["bundle_id"],
        "upstream_final_output_target_sha256": upstream_summary[
            "final_output_target_sha256"
        ],
        "upstream_exact6_verified": True,
        "upstream_raw_source_same_fd_hash_only_count": upstream_summary[
            "raw_source_same_fd_hash_only_count"
        ],
        "decoded_raw_source_count": 0,
        "decoded_semantic_input_count": 1,
        "pass_gate_ids": sorted(spec["gate_id"] for spec in GATE_SPECS),
        "consumer_validate_gate_record_pass_count": 3,
        "consumer_slot_gate_pass_exact_true_count": 3,
        "unchanged_gate_boundary": copy.deepcopy(
            config["expected_audit_semantics"]["unchanged_gates"]
        ),
        "biological_group_status": config["expected_audit_semantics"]
        ["biological_group_authority"]["status"],
        "ordinary_study_contribution_delta": policy[
            "ordinary_study_contribution_delta"
        ],
        "a1_study_contribution_delta": policy["a1_study_contribution_delta"],
        "true_a2_study_contribution_delta": policy[
            "true_a2_study_contribution_delta"
        ],
        "canonical_record_count_delta": policy["canonical_record_count_delta"],
        "consumer_run": False,
        "adjudicator_run": False,
        "qualifier_run": False,
        "canonicalizer_run": False,
        "raw_replay_run": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "aggregate_only": True,
    }


def build_preterminal_payloads(
    config: Mapping[str, Any],
    gate_payloads: Mapping[str, bytes],
    upstream_summary: Mapping[str, Any],
) -> dict[str, bytes]:
    if tuple(sorted(gate_payloads)) != GATE_MEMBER_NAMES:
        raise ProducerError("gate payload input is not exact3")
    content = {
        **dict(gate_payloads),
        AUDIT_MEMBER_NAME: json_bytes(build_pack_audit(config, upstream_summary)),
    }
    sums = "".join(
        f"{sha256(content[name])}  {name}\n" for name in sorted(content)
    ).encode("ascii")
    result = {**content, CHECKSUMS_MEMBER_NAME: sums}
    if tuple(sorted(result)) != PRETERMINAL_MEMBER_NAMES:
        raise ProducerError("preterminal output set is not exact5")
    return result


def _write_exclusive_regular_at(directory_fd: int, name: str, payload: bytes) -> None:
    if Path(name).name != name:
        raise PublicationError("publication member name is unsafe")
    try:
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
    except OSError as exc:
        if isinstance(exc, FileExistsError):
            raise
        raise PublicationError(f"publication member cannot be created: {name}") from exc
    try:
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise PublicationError(f"short write: {name}")
            view = view[count:]
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != len(payload)
        ):
            raise PublicationError(f"publication member type/link/size differs: {name}")
        os.fsync(descriptor)
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if _file_identity(named) != _file_identity(metadata):
            raise PublicationError(f"publication member path binding changed: {name}")
    finally:
        os.close(descriptor)


def _open_child_directory(parent_fd: int, name: str, *, label: str) -> int:
    if not name or Path(name).name != name:
        raise ScopeViolation(f"{label} basename is unsafe")
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
    except OSError as exc:
        raise PublicationError(f"{label} cannot be opened safely") from exc
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise PublicationError(f"{label} is not a directory")
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


def _open_matching_canonical_directory(
    path: Path,
    expected_identity: tuple[int, int],
    *,
    label: str,
) -> int:
    descriptor = _open_directory_root_to_leaf(path, label=label)
    if _directory_identity(descriptor) != expected_identity:
        os.close(descriptor)
        raise PublicationError(f"{label} canonical identity changed")
    return descriptor


def _native_rename_noreplace(parent_fd: int, old_name: str, new_name: str) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    old = os.fsencode(old_name)
    new = os.fsencode(new_name)
    if hasattr(library, "renameat2"):
        function = library.renameat2
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(parent_fd, old, parent_fd, new, 1)
    elif hasattr(library, "renameatx_np"):
        function = library.renameatx_np
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(parent_fd, old, parent_fd, new, 0x00000004)
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


def _payload_set_sha256(payloads: Mapping[str, bytes]) -> str:
    if tuple(sorted(payloads)) != PRETERMINAL_MEMBER_NAMES:
        raise PublicationError("marker payload set is not exact5")
    digest = hashlib.sha256()
    digest.update(b"GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_PAYLOAD_SET_V1\n")
    for name in PRETERMINAL_MEMBER_NAMES:
        encoded = os.fsencode(name)
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)
        digest.update(b"\0")
        digest.update(sha256(payloads[name]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def publication_marker_bytes(
    output: Path,
    payloads: Mapping[str, bytes],
    publication_mode: str,
) -> bytes:
    if publication_mode not in {PRIMARY_PUBLICATION_MODE, FALLBACK_PUBLICATION_MODE}:
        raise PublicationError("publication mode is outside the closed enum")
    if not output.is_absolute():
        raise PublicationError("publication target must be absolute")
    return json_bytes(
        {
            "schema_version": "1.0.0",
            "record_type": PUBLICATION_MARKER_RECORD_TYPE,
            "protocol_id": PROTOCOL_ID,
            "contract_id": CONTRACT_ID,
            "dataset_id": DATASET_ID,
            "decision_id": DECISION_ID,
            "publication_mode": publication_mode,
            "preterminal_member_names": list(PRETERMINAL_MEMBER_NAMES),
            "preterminal_member_count": 5,
            "exact_final_member_count": 6,
            "gate_record_names": list(GATE_MEMBER_NAMES),
            "descriptor_binding_scope": "THREE_GATE_JSON_FILES_ONLY",
            "sha256sums_sha256": sha256(payloads[CHECKSUMS_MEMBER_NAME]),
            "payload_set_sha256": _payload_set_sha256(payloads),
            "final_output_target_sha256": sha256(os.fsencode(output)),
            "committed": True,
            "commit_marker_written_last": True,
            "overwrite_allowed": False,
        }
    )


def _fsync_regular_at(directory_fd: int, name: str, *, label: str) -> None:
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ScopeViolation(f"{label} is not a single-link regular file")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _inspect_committed_at(
    parent_fd: int,
    output: Path,
    payloads: Mapping[str, bytes],
    *,
    expected_modes: Sequence[str] = (
        PRIMARY_PUBLICATION_MODE,
        FALLBACK_PUBLICATION_MODE,
    ),
) -> tuple[tuple[int, int], str]:
    directory_fd = _open_child_directory(
        parent_fd, output.name, label="published output"
    )
    try:
        _assert_named_directory_identity(
            parent_fd, output.name, directory_fd, label="published output"
        )
        if sorted(os.listdir(directory_fd)) != list(PUBLISHED_MEMBER_NAMES):
            raise PartialPublicationError("published output is not exact6")
        for name in PRETERMINAL_MEMBER_NAMES:
            observed = _read_regular_at(
                directory_fd, name, label=f"published member {name}"
            )
            if observed != payloads[name]:
                raise PublicationError(f"published member differs: {name}")
        marker = _read_regular_at(
            directory_fd, MARKER_MEMBER_NAME, label="publication marker"
        )
        matched: str | None = None
        for mode in expected_modes:
            if marker == publication_marker_bytes(output, payloads, mode):
                matched = mode
                break
        if matched is None:
            raise PartialPublicationError("publication marker is invalid or early")
        _assert_named_directory_identity(
            parent_fd, output.name, directory_fd, label="published output"
        )
        return _directory_identity(directory_fd), matched
    finally:
        os.close(directory_fd)


def inspect_published_pack(
    output: Path,
    payloads: Mapping[str, bytes],
    *,
    expected_parent_identity: tuple[int, int] | None = None,
    expected_modes: Sequence[str] = (
        PRIMARY_PUBLICATION_MODE,
        FALLBACK_PUBLICATION_MODE,
    ),
) -> dict[str, Any]:
    """Exact6 acceptance primitive: validate, fsync, reopen, revalidate."""

    if tuple(sorted(payloads)) != PRETERMINAL_MEMBER_NAMES:
        raise PublicationError("publication inspector input is not exact5")
    parent_fd = _open_directory_root_to_leaf(output.parent, label="output parent")
    try:
        parent_identity = _directory_identity(parent_fd)
        if expected_parent_identity is not None and parent_identity != expected_parent_identity:
            raise PublicationError("output parent identity changed")
        directory_identity, mode = _inspect_committed_at(
            parent_fd, output, payloads, expected_modes=expected_modes
        )
        directory_fd = _open_child_directory(
            parent_fd, output.name, label="committed output"
        )
        try:
            if _directory_identity(directory_fd) != directory_identity:
                raise PublicationError("committed output identity changed")
            for name in PUBLISHED_MEMBER_NAMES:
                _fsync_regular_at(directory_fd, name, label=f"committed {name}")
            os.fsync(directory_fd)
            _assert_named_directory_identity(
                parent_fd, output.name, directory_fd, label="committed output"
            )
        finally:
            os.close(directory_fd)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)

    reopened = _open_matching_canonical_directory(
        output.parent, parent_identity, label="output parent post-fsync"
    )
    try:
        reopened_identity, reopened_mode = _inspect_committed_at(
            reopened, output, payloads, expected_modes=expected_modes
        )
        if reopened_identity != directory_identity or reopened_mode != mode:
            raise PublicationError("committed output changed after fsync/reopen")
    finally:
        os.close(reopened)
    return {
        "publication_mode": mode,
        "exact_member_names": list(PUBLISHED_MEMBER_NAMES),
        "descriptor_member_names": list(GATE_MEMBER_NAMES),
        "terminal_commit_marker_validated": True,
        "directory_identity": directory_identity,
    }


def _cleanup_unpublished_temp(
    parent_path: Path,
    parent_identity: tuple[int, int],
    temp_name: str,
    temp_identity: tuple[int, int],
) -> bool:
    """Delete only this call's unpublished exact temp at the canonical parent."""

    try:
        parent_fd = _open_directory_root_to_leaf(
            parent_path, label="unpublished temp canonical parent"
        )
    except ScopeViolation:
        return False
    temp_fd = -1
    try:
        if _directory_identity(parent_fd) != parent_identity:
            return False
        named = os.stat(temp_name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(named.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or (named.st_dev, named.st_ino) != temp_identity
        ):
            return False
        temp_fd = _open_child_directory(parent_fd, temp_name, label="unpublished temp")
        if _directory_identity(temp_fd) != temp_identity:
            return False
        names = sorted(os.listdir(temp_fd))
        if any(name not in PUBLISHED_MEMBER_NAMES for name in names):
            return False
        for name in names:
            metadata = os.stat(name, dir_fd=temp_fd, follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                return False
        for name in names:
            os.unlink(name, dir_fd=temp_fd)
        os.fsync(temp_fd)
        _assert_named_directory_identity(
            parent_fd, temp_name, temp_fd, label="unpublished temp"
        )
        os.close(temp_fd)
        temp_fd = -1
        os.rmdir(temp_name, dir_fd=parent_fd)
        os.fsync(parent_fd)
        return True
    except (OSError, PublicationError, ScopeViolation):
        return False
    finally:
        if temp_fd >= 0:
            os.close(temp_fd)
        os.close(parent_fd)


def _validate_partial_at(
    parent_fd: int,
    output: Path,
    payloads: Mapping[str, bytes],
) -> tuple[int, set[str]]:
    directory_fd = _open_child_directory(
        parent_fd, output.name, label="partial output"
    )
    try:
        _assert_named_directory_identity(
            parent_fd, output.name, directory_fd, label="partial output"
        )
        if os.fstat(directory_fd).st_uid != os.geteuid():
            raise ScopeViolation("partial recovery requires current-euid ownership")
        names = set(os.listdir(directory_fd))
        if MARKER_MEMBER_NAME in names:
            raise PartialPublicationError(
                "invalid/early marker exists; automatic recovery refused"
            )
        if names - set(PRETERMINAL_MEMBER_NAMES):
            raise PartialPublicationError("partial output has an unexpected member")
        for name in sorted(names):
            if _read_regular_at(directory_fd, name, label=f"partial {name}") != payloads[name]:
                raise PartialPublicationError(f"partial member differs: {name}")
        return directory_fd, names
    except Exception:
        os.close(directory_fd)
        raise


def _write_missing_or_validate_race(
    directory_fd: int,
    name: str,
    payload: bytes,
) -> None:
    try:
        _write_exclusive_regular_at(directory_fd, name, payload)
    except FileExistsError:
        if _read_regular_at(directory_fd, name, label=f"raced member {name}") != payload:
            raise PartialPublicationError(f"raced member differs: {name}")


def _publish_fallback(
    output: Path,
    payloads: Mapping[str, bytes],
    *,
    parent_identity: tuple[int, int],
    create_if_absent: bool,
    recover_partial: bool,
    fault_injector: FaultInjector | None,
) -> str:
    parent_fd = _open_matching_canonical_directory(
        output.parent, parent_identity, label="fallback output parent"
    )
    directory_fd = -1
    created = False
    marker_written = False
    try:
        try:
            directory_fd = _open_child_directory(
                parent_fd, output.name, label="fallback output"
            )
        except PublicationError as open_error:
            try:
                os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                if not create_if_absent:
                    raise PublicationStateError(
                        "contended output disappeared",
                        publication_state="AMBIGUOUS_FINAL_CONTENTION_STATE",
                    ) from open_error
                try:
                    os.mkdir(output.name, 0o750, dir_fd=parent_fd)
                    created = True
                except FileExistsError:
                    pass
                directory_fd = _open_child_directory(
                    parent_fd, output.name, label="fallback output"
                )
            else:
                raise
        identity = _directory_identity(directory_fd)
        names = set(os.listdir(directory_fd))
        if MARKER_MEMBER_NAME in names:
            os.close(directory_fd)
            directory_fd = -1
            inspected = inspect_published_pack(
                output, payloads, expected_parent_identity=parent_identity
            )
            if inspected["directory_identity"] != identity:
                raise PublicationError("existing exact output identity changed")
            return "EXISTING_EXACT"
        os.close(directory_fd)
        directory_fd = -1
        directory_fd, present = _validate_partial_at(parent_fd, output, payloads)
        if not created and not recover_partial:
            raise PartialPublicationError(
                "PARTIAL_NOT_COMMITTED preserved; --recover-partial is required"
            )
        if created:
            os.fsync(parent_fd)
        for name in PRETERMINAL_MEMBER_NAMES:
            if name not in present:
                _write_missing_or_validate_race(directory_fd, name, payloads[name])
                if fault_injector is not None:
                    fault_injector(f"after_fallback_write:{name}")
        if set(os.listdir(directory_fd)) != set(PRETERMINAL_MEMBER_NAMES):
            raise PartialPublicationError("fallback pre-marker set is not exact5")
        for name in PRETERMINAL_MEMBER_NAMES:
            if _read_regular_at(directory_fd, name, label=f"fallback {name}") != payloads[name]:
                raise PartialPublicationError(f"fallback member differs: {name}")
        os.fsync(directory_fd)
        _assert_named_directory_identity(
            parent_fd, output.name, directory_fd, label="fallback pre-marker output"
        )
        marker = publication_marker_bytes(output, payloads, FALLBACK_PUBLICATION_MODE)
        try:
            _write_exclusive_regular_at(directory_fd, MARKER_MEMBER_NAME, marker)
        except FileExistsError:
            if _read_regular_at(
                directory_fd, MARKER_MEMBER_NAME, label="raced fallback marker"
            ) != marker:
                raise PartialPublicationError("raced fallback marker differs")
        marker_written = True
        if fault_injector is not None:
            fault_injector("after_fallback_marker")
        try:
            os.fsync(directory_fd)
            os.fsync(parent_fd)
            _assert_named_directory_identity(
                parent_fd, output.name, directory_fd, label="fallback committed output"
            )
        except Exception as exc:
            raise PublicationStateError(
                "exact fallback marker exists but durability is unverified",
                publication_state="COMMIT_MARKER_EXACT_DURABILITY_UNVERIFIED",
            ) from exc
        os.close(directory_fd)
        directory_fd = -1
        inspected = inspect_published_pack(
            output,
            payloads,
            expected_parent_identity=parent_identity,
            expected_modes=(FALLBACK_PUBLICATION_MODE,),
        )
        if inspected["directory_identity"] != identity:
            raise PublicationStateError(
                "fallback output identity changed after marker",
                publication_state="COMMITTED_UNVERIFIED",
            )
        return "PUBLISHED_FALLBACK" if created else "RECOVERED_PARTIAL_FALLBACK"
    except PublicationStateError:
        raise
    except (PartialPublicationError, ScopeViolation):
        raise
    except Exception as exc:
        if marker_written:
            raise PublicationStateError(
                "fallback marker exists but exact durable truth is unverified",
                publication_state="COMMIT_MARKER_PRESENT_NOT_ACCEPTED",
            ) from exc
        raise PartialPublicationError(
            "fallback output is PARTIAL_NOT_COMMITTED and was preserved"
        ) from exc
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)
        os.close(parent_fd)


def publish_pack(
    output: Path,
    payloads: Mapping[str, bytes],
    *,
    production: bool,
    config: Mapping[str, Any],
    recover_partial: bool = False,
    fault_injector: FaultInjector | None = None,
    rename_noreplace: Callable[[int, str, str], None] = _native_rename_noreplace,
) -> str:
    if tuple(sorted(payloads)) != PRETERMINAL_MEMBER_NAMES:
        raise PublicationError("publisher input is not exact5")
    if not output.is_absolute() or any(
        part in {"", ".", ".."} for part in output.parts[1:]
    ):
        raise ScopeViolation("output must be an absolute safe path")
    if production and os.fspath(output) != config["output_contract"]["trusted_final_directory"]:
        raise ScopeViolation("production output differs from the frozen target")
    lowered = os.fspath(output).casefold()
    for token in ("gse246381", "/restricted/", "/sealed/", "/sealed_external/", "access_log"):
        if token in lowered:
            raise ScopeViolation(f"output path contains forbidden token: {token}")

    parent_fd = _open_directory_root_to_leaf(output.parent, label="output parent")
    parent_identity = _directory_identity(parent_fd)
    try:
        try:
            os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            os.close(parent_fd)
            parent_fd = -1
            return _publish_fallback(
                output,
                payloads,
                parent_identity=parent_identity,
                create_if_absent=False,
                recover_partial=recover_partial,
                fault_injector=fault_injector,
            )

        temp_name = f".{output.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
        os.mkdir(temp_name, 0o750, dir_fd=parent_fd)
        temp_fd = _open_child_directory(parent_fd, temp_name, label="publication temp")
        temp_identity = _directory_identity(temp_fd)
        renamed = False
        temp_active = True
        try:
            for name in PRETERMINAL_MEMBER_NAMES:
                _write_exclusive_regular_at(temp_fd, name, payloads[name])
                if fault_injector is not None:
                    fault_injector(f"after_primary_write:{name}")
            if sorted(os.listdir(temp_fd)) != list(PRETERMINAL_MEMBER_NAMES):
                raise PublicationError("primary pre-marker temp is not exact5")
            os.fsync(temp_fd)
            _write_exclusive_regular_at(
                temp_fd,
                MARKER_MEMBER_NAME,
                publication_marker_bytes(output, payloads, PRIMARY_PUBLICATION_MODE),
            )
            if fault_injector is not None:
                fault_injector("after_primary_marker")
            if sorted(os.listdir(temp_fd)) != list(PUBLISHED_MEMBER_NAMES):
                raise PublicationError("primary marker-last temp is not exact6")
            os.fsync(temp_fd)
            os.fsync(parent_fd)
            _assert_named_directory_identity(
                parent_fd, temp_name, temp_fd, label="publication temp"
            )
            if fault_injector is not None:
                fault_injector("before_atomic_rename")
            canonical = _open_matching_canonical_directory(
                output.parent, parent_identity, label="output parent pre-rename"
            )
            os.close(canonical)
            try:
                rename_noreplace(parent_fd, temp_name, output.name)
                renamed = True
                temp_active = False
            except FileExistsError:
                if not _cleanup_unpublished_temp(
                    output.parent, parent_identity, temp_name, temp_identity
                ):
                    temp_active = False
                    raise PartialPublicationError(
                        f"raced final exists; unpublished temp preserved: {temp_name}"
                    )
                temp_active = False
                return _publish_fallback(
                    output,
                    payloads,
                    parent_identity=parent_identity,
                    create_if_absent=False,
                    recover_partial=recover_partial,
                    fault_injector=fault_injector,
                )
            except AtomicNoReplaceUnsupported:
                status = _publish_fallback(
                    output,
                    payloads,
                    parent_identity=parent_identity,
                    create_if_absent=True,
                    recover_partial=recover_partial,
                    fault_injector=fault_injector,
                )
                cleaned = _cleanup_unpublished_temp(
                    output.parent, parent_identity, temp_name, temp_identity
                )
                temp_active = False
                if not cleaned:
                    return f"{status}_COMMITTED_EXACT_SOURCE_TEMP_PRESERVED"
                return status
            except OSError as exc:
                cleaned = _cleanup_unpublished_temp(
                    output.parent, parent_identity, temp_name, temp_identity
                )
                temp_active = False
                state = (
                    "NOT_COMMITTED_TEMP_REMOVED"
                    if cleaned
                    else "NOT_COMMITTED_TEMP_PRESERVED"
                )
                raise PublicationStateError(
                    f"atomic rename failed; state={state}", publication_state=state
                ) from exc
            if fault_injector is not None:
                fault_injector("after_atomic_rename")
            inspected = inspect_published_pack(
                output,
                payloads,
                expected_parent_identity=parent_identity,
                expected_modes=(PRIMARY_PUBLICATION_MODE,),
            )
            if inspected["directory_identity"] != temp_identity:
                raise PublicationStateError(
                    "renamed output identity differs",
                    publication_state="COMMITTED_UNVERIFIED",
                )
            return "PUBLISHED_PRIMARY"
        except Exception as exc:
            if renamed:
                state = "COMMITTED_UNVERIFIED"
                try:
                    inspected = inspect_published_pack(
                        output, payloads, expected_parent_identity=parent_identity
                    )
                    if inspected["directory_identity"] == temp_identity:
                        state = "COMMITTED_EXACT"
                except Exception:
                    pass
                if isinstance(exc, PublicationStateError):
                    raise
                raise PublicationStateError(
                    f"post-rename failure; state={state}: {exc}",
                    publication_state=state,
                ) from exc
            if isinstance(exc, PublicationStateError):
                # A fallback marker already exists.  Cleanup of the separate
                # unpublished source temp is best-effort and must never
                # downgrade the known final-marker truth to a generic partial.
                if temp_active:
                    _cleanup_unpublished_temp(
                        output.parent, parent_identity, temp_name, temp_identity
                    )
                raise
            if temp_active:
                cleaned = _cleanup_unpublished_temp(
                    output.parent, parent_identity, temp_name, temp_identity
                )
                if not cleaned:
                    raise PartialPublicationError(
                        f"not committed; unpublished temp preserved: {temp_name}"
                    ) from exc
            raise
        finally:
            os.close(temp_fd)
    finally:
        if parent_fd >= 0:
            os.close(parent_fd)


def produce(
    config: Mapping[str, Any],
    output: Path,
    *,
    repo: Path,
    production: bool,
    config_payload: bytes | None = None,
    config_path: Path | None = None,
    upstream_root: Path | None = None,
    inspect_only: bool = False,
    recover_partial: bool = False,
    fault_injector: FaultInjector | None = None,
    rename_noreplace: Callable[[int, str, str], None] = _native_rename_noreplace,
) -> dict[str, Any]:
    """Bind first, then consumer, upstream exact6, records, and publication."""

    validate_static_config(config)
    # Contractual no-I/O boundary: UNKNOWN stops before repository consumer,
    # upstream source, or output handling.
    validate_implementation_binding(config)
    if production:
        trusted_output = Path(config["output_contract"]["trusted_final_directory"])
        trusted_upstream = Path(config["upstream_authority"]["absolute_root"])
        if output != trusted_output:
            raise ScopeViolation(
                "production output differs from the exact trusted final directory"
            )
        if upstream_root is not None and upstream_root != trusted_upstream:
            raise ScopeViolation(
                "production upstream root differs from the exact immutable authority"
            )
    git_authority: dict[str, str] | None = None
    if production:
        if config_payload is None or config_path is None:
            raise AuthorityError("production config bytes/path are absent")
        git_authority = validate_production_authority(
            config,
            config_payload,
            config_path=config_path,
        )
    frozen_consumer, current_consumer, consumer_module, consumer_git = (
        load_verified_consumer(
            config,
            repo=repo,
            production=production,
            current_head=git_authority["current_head"] if git_authority else None,
        )
    )
    upstream = inspect_upstream_bundle(config, root=upstream_root)
    gate_payloads = build_gate_records(config, frozen_consumer, consumer_module)
    payloads = build_preterminal_payloads(config, gate_payloads, upstream)
    if inspect_only:
        inspected = inspect_published_pack(output, payloads)
        publication_status = "EXISTING_EXACT_INSPECTED"
    else:
        publication_status = publish_pack(
            output,
            payloads,
            production=production,
            config=config,
            recover_partial=recover_partial,
            fault_injector=fault_injector,
            rename_noreplace=rename_noreplace,
        )
        inspected = inspect_published_pack(output, payloads)
    return {
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "decision_id": DECISION_ID,
        "publication_status": publication_status,
        "final_directory": os.fspath(output),
        "exact_member_count": 6,
        "gate_record_count": 3,
        "gate_record_names": list(GATE_MEMBER_NAMES),
        "all_records_consumer_validate_gate_record_accepted": True,
        "all_records_consumer_slot_gate_pass_exact_true": True,
        "upstream_exact6_verified": True,
        "upstream_decoded_raw_source_count": 0,
        "upstream_decoded_semantic_input_count": 1,
        "ordinary_study_contribution_delta": 0,
        "a1_study_contribution_delta": 0,
        "true_a2_study_contribution_delta": 0,
        "canonical_record_count_delta": 0,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "publication_inspection": inspected,
        "git_authority": git_authority,
        "consumer_production_authority": consumer_git,
        "current_consumer_descriptor_set_sha256": current_consumer[
            "evidence_descriptor_bindings"
        ]["descriptor_set_sha256"],
    }


def validate_authority_only(
    config: Mapping[str, Any],
    *,
    repo: Path,
    production: bool,
    config_payload: bytes | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Validate code and consumer authority without touching upstream/output."""

    validate_static_config(config)
    validate_implementation_binding(config)
    git_authority: dict[str, str] | None = None
    if production:
        if config_payload is None or config_path is None:
            raise AuthorityError("production config bytes/path are absent")
        git_authority = validate_production_authority(
            config, config_payload, config_path=config_path
        )
    frozen, _current, module, consumer_git = load_verified_consumer(
        config,
        repo=repo,
        production=production,
        current_head=git_authority["current_head"] if git_authority else None,
    )
    payloads = build_gate_records(config, frozen, module)
    return {
        "status": "PASS_CODE_AND_CONSUMER_AUTHORITY_NO_UPSTREAM_OR_OUTPUT",
        "gate_record_count": len(payloads),
        "git_authority": git_authority,
        "consumer_production_authority": consumer_git,
    }


def _load_producer_config(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_absolute():
        path = path.resolve()
    parent_fd = _open_directory_root_to_leaf(path.parent, label="producer config parent")
    try:
        payload = _read_regular_at(parent_fd, path.name, label="producer config")
    finally:
        os.close(parent_fd)
    return strict_json(payload, label="producer config"), payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PRODUCTION_CONFIG_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--validate-authority", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--recover-partial", action="store_true")
    parser.add_argument(
        "--non-production",
        action="store_true",
        help="development-only mode; production is the default",
    )
    arguments = parser.parse_args(argv)
    if arguments.validate_authority and (
        arguments.inspect_only or arguments.recover_partial
    ):
        parser.error(
            "--validate-authority cannot be combined with inspect/recovery"
        )
    if arguments.inspect_only and arguments.recover_partial:
        parser.error("--inspect-only cannot be combined with --recover-partial")
    production = not arguments.non_production
    config_path = arguments.config.resolve()
    if production and config_path != PRODUCTION_CONFIG_PATH:
        raise ScopeViolation("production config path differs")
    config, config_payload = _load_producer_config(config_path)
    repo = (
        PRODUCTION_REPO_ROOT
        if production
        else (arguments.repo.resolve() if arguments.repo else config_path.parents[1])
    )
    if arguments.validate_authority:
        result = validate_authority_only(
            config,
            repo=repo,
            production=production,
            config_payload=config_payload,
            config_path=config_path,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    output = (
        arguments.output.resolve()
        if arguments.output
        else Path(config["output_contract"]["trusted_final_directory"])
    )
    upstream_root = arguments.upstream_root.resolve() if arguments.upstream_root else None
    if production and upstream_root is not None and os.fspath(upstream_root) != config[
        "upstream_authority"
    ]["absolute_root"]:
        raise ScopeViolation("production upstream root differs")
    result = produce(
        config,
        output,
        repo=repo,
        production=production,
        config_payload=config_payload,
        config_path=config_path,
        upstream_root=upstream_root,
        inspect_only=arguments.inspect_only,
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
