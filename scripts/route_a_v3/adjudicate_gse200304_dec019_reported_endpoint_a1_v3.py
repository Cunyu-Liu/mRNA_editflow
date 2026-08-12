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


CONFIG_REPO_PATH = "configs/route_a_v3_gse200304_dec019_reported_endpoint_a1_activation_v3.json"
SCRIPT_REPO_PATH = "scripts/route_a_v3/adjudicate_gse200304_dec019_reported_endpoint_a1_v3.py"
TEST_REPO_PATH = "tests/route_a_v3/test_adjudicate_gse200304_dec019_reported_endpoint_a1_v3.py"
GSE200304_CONFIG_REPO_PATH = CONFIG_REPO_PATH
BINDING_CONFIG_REPO_PATHS = (GSE200304_CONFIG_REPO_PATH,)
EXPECTED_IMPLEMENTATION_FILES = {
    GSE200304_CONFIG_REPO_PATH: (
        SCRIPT_REPO_PATH,
        TEST_REPO_PATH,
    ),
}
FROZEN_CONFIG_CORE_SHA256_BY_PATH = {
    GSE200304_CONFIG_REPO_PATH: (
        "c0cf7852f3a081d5bed329cf4670dc067118e3b1161fbcb0b0713a2de819c71b"
    ),
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
    "evidence_descriptor_bindings",
}
PRODUCTION_REPO_ROOT = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/routea_v3_a1_20260810"
)
PRODUCTION_CONFIG_PATH = PRODUCTION_REPO_ROOT / CONFIG_REPO_PATH
TRUSTED_A1_OUTPUT_ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1")
UNKNOWN = "UNKNOWN_NOT_ASSERTED"
CONTRACT_ID = "mrna_xeditflow_route_a_v3"
DATASET_ID = "GSE200304"
DECISION_ID = "V3-DEC-019"
PROTOCOL_ID = "ROUTE_A_V3_GSE200304_DEC019_REPORTED_ENDPOINT_A1_ACTIVATION_V3"
EVIDENCE_SCHEMA_VERSION = "route_a_v3_dec019_aggregate_gate_evidence.v3"
EVIDENCE_RECORD_TYPE = "ROUTE_A_V3_DEC019_ACCEPTED_AGGREGATE_GATE_EVIDENCE_V3"
GROUP_MAPPING_COMMITMENT_KEY = "group_mapping_commitment_sha256"
LOCATOR_LINEAGE_COMMITMENT_ALGORITHM = (
    "ROUTE_A_V3_GSE200304_LOCATOR_MERKLE_V1"
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DNA_LIKE = re.compile(r"^[ACGTUNacgtun]{20,}$")
OUTPUT_JSON_NAMES = ("ADJUDICATION_REPORT.json", "INPUT_EVIDENCE_AUDIT.json")
OUTPUT_NAMES_EXCLUDING_MARKER = (*OUTPUT_JSON_NAMES, "SHA256SUMS")
COMMIT_MARKER = "PUBLICATION_COMMIT.json"
PUBLICATION_MODE = "ATOMIC_EXCLUSIVE_DIRECTORY_TERMINAL_COMMIT_MARKER_V1"
BLOCKED_STATUS = "BLOCKED_DEC019_REPORTED_ENDPOINT_A1_EVIDENCE_INCOMPLETE"
SUCCESS_STATUS = "PASS_DEC019_REPORTED_ENDPOINT_A1_ACTIVATED_RAW_REPLAY_AUXILIARY"
AUTHORITY_PROVENANCE_KEYS = {
    "mode",
    "lifecycle_state",
    "historical_base_commit",
    "historical_implementation_commit",
    "historical_binding_commit",
    "repair_base_commit",
    "repair_implementation_commit",
    "repair_binding_commit",
    "current_head",
    "science_core_sha256",
    "evidence_descriptor_set_sha256",
    "predecessor_authority_sha256",
}
REPORT_KEYS = {
    "record_type", "contract_id", "decision_id", "protocol_id", "dataset_id",
    "status", "qualified", "data_role", "primary_measurement_route",
    "raw_replay_role", "ordinary_study_contribution", "a1_study_contribution",
    "true_a2_study_contribution", "canonical_record_count",
    "canonical_materialization_allowed", "independent_raw_reproduction_established",
    "training_allowed", "model_selection_allowed", "next_phase_authorized",
    "scientific_claim_status", "aggregate_only", "blockers", "config_core_sha256",
    "evidence_descriptor_set_sha256", "authority_provenance",
}
AUDIT_KEYS = {
    "record_type", "contract_id", "decision_id", "protocol_id", "dataset_id",
    "mode", "all_inputs_aggregate_only", "row_level_payload_read_count",
    "sequence_read_count", "opened_input_count", "slots",
    "evidence_descriptor_set_sha256", "authority_provenance",
}
AUDIT_SLOT_KEYS = {
    "slot_id", "descriptor_bound", "input_opened", "hash_verified", "gate_status"
}
MARKER_KEYS = {
    "schema_version", "record_type", "contract_id", "decision_id", "protocol_id",
    "dataset_id", "output_id", "scientific_status", "publication_mode",
    "sha256sums_sha256", "bundle_member_names_excluding_commit_marker",
    "bundle_file_count_excluding_commit_marker", "final_output_target_sha256",
    "committed", "commit_marker_written_last",
    "aggregate_acceptance_requires_exact_marker",
}

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
DESCRIPTOR_BINDING_KEYS = {
    "binding_scheme",
    "status",
    "descriptor_set_sha256",
    "dynamic_scalar_suffixes",
    "all_descriptors_required_before_any_input_open",
    "slots",
}
DESCRIPTOR_SLOT_KEYS = {"slot_id", "absolute_path", "sha256", "bytes"}
NEGATIVE_EVIDENCE_STATUSES = {UNKNOWN, "NOT_RUN", "BLOCKED"}
REASON_CODE = re.compile(r"^[A-Z][A-Z0-9_]*$")
HISTORICAL_BASE_COMMIT = "139c4e8d9749ae93ed90924bb527127cf2bbf553"
HISTORICAL_IMPLEMENTATION_COMMIT = "4e200ed4048d5b112c6ac324d2376e8de1441419"
HISTORICAL_BINDING_COMMIT = "e495c7ec5b6f00f14a18a4ffe0c5a6f2173bf2d8"
HISTORICAL_CONFIG_CORE_SHA256 = (
    "f4bfde594ce2aa4dbf7d6a9f0cd1607ac1b214a4659089a59258ba0039bb2ff9"
)
HISTORICAL_FROZEN_BLOBS = {
    CONFIG_REPO_PATH: "8ec603142f05d4212610d8745e554626cda37f176d3b2d32c1ca8b934ad89fa8",
    SCRIPT_REPO_PATH: "d7205d67c00e94e3355097d411621f3c380a73f7efda07d7efac65ed2dcbe56d",
    TEST_REPO_PATH: "5f0eee60dfdaa8201d36a5e560c0c59027c93d28692a419e383df2d747d060e0",
}
PREDECESSOR_D1_PARENT_COMMIT = "c764c721b364e19916ba66698552eee86563dbfe"
PREDECESSOR_D1_COMMIT = "c61f3d06ab6cfbc54ff562738d95ba902865b54f"
PREDECESSOR_D1_IMPLEMENTATION_COMMIT = "86d16c181fc9deaf83597da9c1523e4fea9c7493"
PREDECESSOR_D1_CONFIG_CORE_SHA256 = (
    "14aba30da13f2bbad9debca74b9f3c8a8aaae1e5249347a8c1d35eda364a4f50"
)
PREDECESSOR_D1_PARENT_CONFIG_SHA256 = (
    "8c88eb6c708fa309ff0c87a0f64fce1bb205a0212b35a85ad3fb3505e8d7613b"
)
PREDECESSOR_D1_DESCRIPTOR_SET_SHA256 = (
    "079dd5d91df1b6efde42c8277406b16edc99b2ac7181923a529767a8eb97f348"
)
PREDECESSOR_D1_EXACT_CHANGED_PATHS = [CONFIG_REPO_PATH]
PREDECESSOR_D1_SEMANTIC_DIFF_PATHS = frozenset(
    {
        "evidence_descriptor_bindings.descriptor_set_sha256",
        "evidence_descriptor_bindings.slots[0].absolute_path",
        "evidence_descriptor_bindings.slots[0].bytes",
        "evidence_descriptor_bindings.slots[0].sha256",
        "evidence_descriptor_bindings.status",
    }
)
PREDECESSOR_D1_FROZEN_BLOBS = {
    CONFIG_REPO_PATH: "955747ffa55cad93c6fbe7950f9ffa89997c5597bdad8add66877e2e1f08b981",
    SCRIPT_REPO_PATH: "90e840b721e5d07d4437d429d5b42f5a91fc262e560b3b331095db65dbb18fa6",
    TEST_REPO_PATH: "ca0d5221748aaecc10b31edb691f8244a0fe2b94cf67ae9a8f493ac8d3f75ca5",
}
PREDECESSOR_I3_PARENT_COMMIT = PREDECESSOR_D1_COMMIT
PREDECESSOR_I3_COMMIT = "e829464d6ea1953b7a859ba5506946b9cb8e6384"
REPAIR_BASE_COMMIT = "f4922af6dfcd6e8b63064fe8d819edb3971da1fb"
PREDECESSOR_I3_CONFIG_CORE_SHA256 = (
    "bca69bd05c094575bfa860b5492f019810c2845abe9218d7030444821f357a0b"
)
PREDECESSOR_I3_DESCRIPTOR_SET_SHA256 = PREDECESSOR_D1_DESCRIPTOR_SET_SHA256
PREDECESSOR_I3_EXACT_CHANGED_PATHS = sorted(
    (CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH)
)
PREDECESSOR_I3_FROZEN_BLOBS = {
    CONFIG_REPO_PATH: "b9c5948f23a6fdb8d250c65ed370f72fdf4fa8c08f5038169b5dff5584568a26",
    SCRIPT_REPO_PATH: "7ef3e14c7298d04cf03fcf4b86f1a71f87a7917fa1b8d3ef826de9262e3d1295",
    TEST_REPO_PATH: "611ba1788b82b28e5a5390537672a211c128e309ab82095286545c61ed075a96",
}
EXPECTED_BASE_TO_I_DIFF_PATHS = frozenset(
    {
        "implementation_binding.config_core_sha256",
        "implementation_binding.implementation_commit",
        "implementation_binding.implementation_script_sha256",
        "implementation_binding.implementation_test_sha256",
        "implementation_binding.status",
        (
            "evidence_contract.gate_record_provenance_contract."
            "biological_group_pass_requires_mapping_commitment_sha256"
        ),
        "repository_authority.base_commit",
        "repository_authority.implementation_commit_expected_parent",
    }
)
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
    """Stable science core, excluding implementation and descriptor values.

    The descriptor *schema* and ordered slot identities remain frozen in the
    projection.  Only lifecycle values (status/digest/path/hash/bytes) are
    outside the science core, so an evidence-binding commit cannot change a
    gate, fact, threshold, blocker, basename, predecessor, or policy claim.
    """

    projected = copy.deepcopy(dict(config))
    projected.pop("implementation_binding", None)
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
                {"slot_id": slot.get("slot_id")}
                for slot in slots
            ]
            if type(slots) is list
            else slots,
        }
    return projected


def config_core_sha256(config: Mapping[str, Any]) -> str:
    return sha256(json_bytes(config_core_projection(config)))


def descriptor_set_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    descriptors = copy.deepcopy(dict(config["evidence_descriptor_bindings"]))
    descriptors.pop("descriptor_set_sha256", None)
    return descriptors


def descriptor_set_sha256(config: Mapping[str, Any]) -> str:
    return sha256(json_bytes(descriptor_set_projection(config)))


def _descriptor_slot_bound(slot: Mapping[str, Any]) -> bool:
    return (
        type(slot.get("absolute_path")) is str
        and slot.get("absolute_path") != UNKNOWN
        and HEX64.fullmatch(str(slot.get("sha256"))) is not None
        and type(slot.get("bytes")) is int
        and type(slot.get("bytes")) is not bool
        and int(slot["bytes"]) > 0
    )


def _descriptor_slot_unbound(slot: Mapping[str, Any]) -> bool:
    return all(slot.get(key) == UNKNOWN for key in ("absolute_path", "sha256", "bytes"))


def _derived_descriptor_status(config: Mapping[str, Any]) -> str:
    slots = config["evidence_descriptor_bindings"]["slots"]
    bound_count = sum(_descriptor_slot_bound(slot) for slot in slots)
    if bound_count == 0:
        return "UNBOUND"
    if bound_count == len(slots):
        return "BOUND"
    return "PARTIALLY_BOUND"


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
    """Prove parent-I changed only through the four allowed B scalars."""

    if config_path not in EXPECTED_IMPLEMENTATION_FILES:
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
    computed_cores: list[str] = []
    for value, binding, label in (
        (i_config, i_binding, "parent-I"),
        (b_config, b_binding, "current-B"),
    ):
        computed = config_core_sha256(value)
        computed_cores.append(computed)
        if binding.get("config_core_sha256") != computed:
            raise BindingError(f"{label} stored science core differs: {config_path}")
        if binding.get("config_core_sha256") != frozen_core:
            raise BindingError(
                f"{label} science core differs from compiled authority: {config_path}"
            )
    if computed_cores[0] != computed_cores[1]:
        raise BindingError(f"I-to-B science core changed: {config_path}")
    if computed_cores[0] != frozen_core:
        raise BindingError(f"science core differs from compiled authority: {config_path}")

    for value, label in ((i_config, "parent-I"), (b_config, "current-B")):
        repository = value.get("repository_authority")
        if type(repository) is not dict:
            raise BindingError(f"{label} repository authority is absent: {config_path}")
        if repository.get("base_commit") != REPAIR_BASE_COMMIT:
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
        if repository.get("implementation_commit_exact_changed_paths") != sorted(
            {
                *BINDING_CONFIG_REPO_PATHS,
                *(path for pair in EXPECTED_IMPLEMENTATION_FILES.values() for path in pair),
            }
        ):
            raise BindingError(f"{label} implementation changed-path set differs: {config_path}")

    differences = _semantic_diff_paths(i_config, b_config)
    if differences != set(ALLOWED_I_TO_B_SCALAR_PATHS):
        raise BindingError(
            f"I-to-B semantic diff is not the exact four-scalar allowlist: "
            f"{config_path}: {sorted(differences)!r}"
        )


def _allowed_descriptor_diff_paths(config: Mapping[str, Any]) -> set[str]:
    allowed = {
        "evidence_descriptor_bindings.status",
        "evidence_descriptor_bindings.descriptor_set_sha256",
    }
    for index, _slot in enumerate(config["evidence_descriptor_bindings"]["slots"]):
        allowed.update(
            f"evidence_descriptor_bindings.slots[{index}].{field}"
            for field in ("absolute_path", "sha256", "bytes")
        )
    return allowed


def _validate_descriptor_only_transition(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    config_path: str,
) -> None:
    """Reject every post-B config change outside descriptor lifecycle scalars."""

    if type(before) is not dict or type(after) is not dict:
        raise BindingError(f"descriptor transition config is not an object: {config_path}")
    if config_path not in FROZEN_CONFIG_CORE_SHA256_BY_PATH:
        raise BindingError(f"descriptor transition config path is outside the closed pair: {config_path}")
    frozen_core = FROZEN_CONFIG_CORE_SHA256_BY_PATH[config_path]
    before_core = config_core_sha256(before)
    after_core = config_core_sha256(after)
    if before_core != after_core:
        raise BindingError(f"descriptor transition changed science core: {config_path}")
    if before_core != frozen_core or after_core != frozen_core:
        raise BindingError(f"descriptor transition left compiled science core: {config_path}")
    for value, label in ((before, "before"), (after, "after")):
        binding = value.get("implementation_binding")
        if type(binding) is not dict or binding.get("config_core_sha256") != frozen_core:
            raise BindingError(
                f"descriptor transition {label} stored science core differs: {config_path}"
            )
    if before.get("implementation_binding") != after.get("implementation_binding"):
        raise BindingError(f"descriptor transition changed implementation binding: {config_path}")
    differences = _semantic_diff_paths(before, after)
    if not differences:
        raise BindingError(f"empty descriptor transition is not authoritative: {config_path}")
    disallowed = differences - _allowed_descriptor_diff_paths(after)
    if disallowed:
        raise BindingError(
            f"post-B config change is not descriptor-only: {config_path}: {sorted(disallowed)!r}"
        )



def _assert_no_private_material(value: Any, forbidden_keys: set[str], *, label: str) -> None:
    if type(value) is dict:
        for key, child in value.items():
            if key.casefold() in forbidden_keys:
                raise AdjudicationError(f"{label} contains forbidden key: {key}")
            # This one closed fact is an aggregate, domain-separated Merkle
            # root.  A valid digest can lexically resemble a DNA string (for
            # example, 64 lowercase "a" bytes), but it is neither a leaf nor
            # a source-member hash and carries no row value.
            if (
                key == "locator_lineage_merkle_root_sha256"
                and type(child) is str
                and HEX64.fullmatch(child) is not None
            ):
                continue
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
    """Validate the closed v3 science contract and dynamic descriptor state."""

    _expect_exact_keys(config, EXPECTED_CONFIG_TOP_KEYS, label="config")
    for key, expected in {
        "schema_version": "route_a_v3_gse200304_dec019_reported_endpoint_a1_activation.v3",
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
    _expect_exact(
        binding["binding_scheme"],
        "CONFIG_ONLY_POST_IMPLEMENTATION_BINDING_V1",
        label="implementation binding scheme",
    )
    _expect_exact(
        binding["blocker_if_unbound"],
        "IMPLEMENTATION_BINDING_UNKNOWN_NOT_ASSERTED",
        label="implementation blocker",
    )
    _expect_exact(binding["implementation_script_path"], SCRIPT_REPO_PATH, label="script path")
    _expect_exact(binding["implementation_test_path"], TEST_REPO_PATH, label="test path")
    _expect_exact(
        binding["unknown_to_bound_scalar_paths"],
        EXPECTED_I_TO_B_SCALAR_PATHS,
        label="I-to-B scalar paths",
    )
    dynamic_fields = (
        "implementation_commit",
        "implementation_script_sha256",
        "implementation_test_sha256",
    )
    if binding["status"] == UNKNOWN:
        if any(binding[key] != UNKNOWN for key in dynamic_fields):
            raise AdjudicationError("UNKNOWN implementation binding is partially bound")
    elif binding["status"] == "BOUND":
        if (
            HEX40.fullmatch(str(binding["implementation_commit"])) is None
            or HEX64.fullmatch(str(binding["implementation_script_sha256"])) is None
            or HEX64.fullmatch(str(binding["implementation_test_sha256"])) is None
        ):
            raise AdjudicationError("BOUND implementation binding has an invalid field")
    else:
        raise AdjudicationError("implementation binding status is outside the closed enum")
    if binding["config_core_sha256"] != FROZEN_CONFIG_CORE_SHA256:
        raise AdjudicationError("stored science core differs from compiled authority")
    if config_core_sha256(config) != FROZEN_CONFIG_CORE_SHA256:
        raise AdjudicationError("computed science core differs from compiled authority")

    repository = _expect_exact_keys(
        config["repository_authority"],
        {
            "production_repo_root",
            "branch",
            "base_commit",
            "implementation_commit_expected_parent",
            "binding_commit_expected_parent",
            "implementation_commit_exact_changed_paths",
            "binding_commit_exact_changed_paths",
            "historical_dec019_binding",
            "predecessor_descriptor_binding",
            "predecessor_implementation_successor",
            "descendant_policy",
        },
        label="repository authority",
    )
    _expect_exact(repository["production_repo_root"], os.fspath(PRODUCTION_REPO_ROOT), label="repo root")
    _expect_exact(repository["branch"], "routea-v3-a1-20260810", label="branch")
    _expect_exact(repository["base_commit"], REPAIR_BASE_COMMIT, label="repair base")
    _expect_exact(repository["implementation_commit_expected_parent"], REPAIR_BASE_COMMIT, label="repair I parent")
    _expect_exact(repository["binding_commit_expected_parent"], "IMPLEMENTATION_COMMIT_FROM_BINDING", label="repair B parent rule")
    expected_i_paths = sorted(
        {
            *BINDING_CONFIG_REPO_PATHS,
            *(path for pair in EXPECTED_IMPLEMENTATION_FILES.values() for path in pair),
        }
    )
    _expect_exact(repository["implementation_commit_exact_changed_paths"], expected_i_paths, label="repair I paths")
    _expect_exact(repository["binding_commit_exact_changed_paths"], list(BINDING_CONFIG_REPO_PATHS), label="repair B paths")
    historical = repository["historical_dec019_binding"]
    _expect_exact_keys(
        historical,
        {
            "base_commit",
            "implementation_commit",
            "binding_commit",
            "science_core_sha256",
            "frozen_successor_blobs",
        },
        label="historical DEC019 binding",
    )
    _expect_exact(historical["base_commit"], HISTORICAL_BASE_COMMIT, label="historical base")
    _expect_exact(historical["implementation_commit"], HISTORICAL_IMPLEMENTATION_COMMIT, label="historical I")
    _expect_exact(historical["binding_commit"], HISTORICAL_BINDING_COMMIT, label="historical B")
    _expect_exact(
        historical["science_core_sha256"],
        HISTORICAL_CONFIG_CORE_SHA256,
        label="historical science core",
    )
    frozen_blobs = historical["frozen_successor_blobs"]
    if type(frozen_blobs) is not list or {
        item.get("path"): item.get("sha256") for item in frozen_blobs if type(item) is dict
    } != HISTORICAL_FROZEN_BLOBS:
        raise AdjudicationError("historical successor blob registry differs")
    predecessor = _expect_exact_keys(
        repository["predecessor_descriptor_binding"],
        {
            "parent_commit",
            "descriptor_commit",
            "science_core_sha256",
            "parent_config_sha256",
            "descriptor_set_sha256",
            "descriptor_commit_exact_changed_paths",
            "descriptor_semantic_diff_paths",
            "frozen_descriptor_commit_blobs",
        },
        label="predecessor D1 descriptor binding",
    )
    _expect_exact(
        predecessor["parent_commit"],
        PREDECESSOR_D1_PARENT_COMMIT,
        label="predecessor D1 parent",
    )
    _expect_exact(
        predecessor["descriptor_commit"],
        PREDECESSOR_D1_COMMIT,
        label="predecessor D1 commit",
    )
    _expect_exact(
        predecessor["science_core_sha256"],
        PREDECESSOR_D1_CONFIG_CORE_SHA256,
        label="predecessor D1 science core",
    )
    _expect_exact(
        predecessor["parent_config_sha256"],
        PREDECESSOR_D1_PARENT_CONFIG_SHA256,
        label="predecessor D1 parent config SHA",
    )
    _expect_exact(
        predecessor["descriptor_set_sha256"],
        PREDECESSOR_D1_DESCRIPTOR_SET_SHA256,
        label="predecessor D1 descriptor-set SHA",
    )
    _expect_exact(
        predecessor["descriptor_commit_exact_changed_paths"],
        PREDECESSOR_D1_EXACT_CHANGED_PATHS,
        label="predecessor D1 changed paths",
    )
    _expect_exact(
        predecessor["descriptor_semantic_diff_paths"],
        sorted(PREDECESSOR_D1_SEMANTIC_DIFF_PATHS),
        label="predecessor D1 semantic diff paths",
    )
    predecessor_blobs = predecessor["frozen_descriptor_commit_blobs"]
    if type(predecessor_blobs) is not list or {
        item.get("path"): item.get("sha256")
        for item in predecessor_blobs
        if type(item) is dict
    } != PREDECESSOR_D1_FROZEN_BLOBS:
        raise AdjudicationError("predecessor D1 blob registry differs")
    predecessor_i3 = _expect_exact_keys(
        repository["predecessor_implementation_successor"],
        {
            "parent_commit",
            "implementation_commit",
            "science_core_sha256",
            "descriptor_set_sha256",
            "implementation_commit_exact_changed_paths",
            "frozen_implementation_commit_blobs",
        },
        label="predecessor I3 implementation successor",
    )
    _expect_exact(
        predecessor_i3["parent_commit"],
        PREDECESSOR_I3_PARENT_COMMIT,
        label="predecessor I3 parent",
    )
    _expect_exact(
        predecessor_i3["implementation_commit"],
        PREDECESSOR_I3_COMMIT,
        label="predecessor I3 commit",
    )
    _expect_exact(
        predecessor_i3["science_core_sha256"],
        PREDECESSOR_I3_CONFIG_CORE_SHA256,
        label="predecessor I3 science core",
    )
    _expect_exact(
        predecessor_i3["descriptor_set_sha256"],
        PREDECESSOR_I3_DESCRIPTOR_SET_SHA256,
        label="predecessor I3 descriptor-set SHA",
    )
    _expect_exact(
        predecessor_i3["implementation_commit_exact_changed_paths"],
        PREDECESSOR_I3_EXACT_CHANGED_PATHS,
        label="predecessor I3 changed paths",
    )
    predecessor_i3_blobs = predecessor_i3["frozen_implementation_commit_blobs"]
    if type(predecessor_i3_blobs) is not list or {
        item.get("path"): item.get("sha256")
        for item in predecessor_i3_blobs
        if type(item) is dict
    } != PREDECESSOR_I3_FROZEN_BLOBS:
        raise AdjudicationError("predecessor I3 blob registry differs")
    _expect_exact(
        repository["descendant_policy"],
        {
            "current_head_must_be_clean_pushed_descendant_of_historical_binding": True,
            "current_head_must_contain_lifecycle_binding": True,
            "historical_successor_blobs_must_not_drift": True,
            "later_config_changes_must_be_descriptor_only": True,
        },
        label="descendant policy",
    )

    authority = _expect_exact_keys(
        config["core_authority"],
        {
            "status", "root_contract_path", "root_contract_sha256", "amendment_path",
            "amendment_sha256", "decision_log_path", "decision_log_sha256",
            "data_role_registry_path", "data_role_registry_sha256", "split_registry_path",
            "split_registry_sha256", "task_registry_path", "task_registry_sha256",
            "task_split_matrix_path", "task_split_matrix_sha256", "claim_evidence_matrix_path",
            "claim_evidence_matrix_sha256", "a1_qualification_path", "a1_qualification_sha256",
            "forbidden_cyclic_dependencies",
        },
        label="core authority",
    )
    _expect_exact(authority["status"], "BOUND", label="core authority status")
    forbidden_cycles = {
        "docs/execution/route_a_v3_a1_interim.yaml",
        "docs/execution/route_a_v3_registry_manifest.json",
        "scripts/route_a_v3/validate_a0_bundle.py",
    }
    if set(authority["forbidden_cyclic_dependencies"]) != forbidden_cycles:
        raise AdjudicationError("forbidden cyclic dependency list differs")
    if any(path in forbidden_cycles for path, _ in _core_leaf_pairs(config)):
        raise AdjudicationError("a core trust leaf creates a forbidden cycle")
    if any(HEX64.fullmatch(str(digest)) is None for _, digest in _core_leaf_pairs(config)):
        raise AdjudicationError("a core authority leaf SHA is not bound")

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
    _expect_exact(config["policy_boundary"], fixed_policy, label="DEC019 GSE200304 policy")

    state = _expect_exact_keys(
        config["current_external_state"],
        {
            "status", "qualified", "ordinary_study_contribution", "a1_study_contribution",
            "true_a2_study_contribution", "canonical_record_count",
            "canonical_materialization_allowed", "independent_raw_reproduction_established",
            "training_allowed", "model_selection_allowed", "next_phase_authorized",
            "scientific_claim_status", "unresolved_blockers",
        },
        label="current external state",
    )
    expected_state = {
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
    }
    for key, expected in expected_state.items():
        _expect_exact(state[key], expected, label=f"current state {key}")

    evidence = _expect_exact_keys(
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
            "evidence_schema_version",
            "negative_record_policy",
            "required_predecessor_authority",
            "gate_record_provenance_contract",
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
        _expect_exact(evidence[key], True, label=f"evidence contract {key}")
    _expect_exact(evidence["evidence_schema_version"], EVIDENCE_SCHEMA_VERSION, label="evidence schema")
    slots = evidence["slots"]
    if type(slots) is not list or tuple(slot.get("slot_id") for slot in slots) != SLOT_IDS:
        raise AdjudicationError("evidence slot IDs or order differ")
    for slot in slots:
        _expect_exact_keys(
            slot,
            {"slot_id", "allowed_basename", "blocker_if_unbound", "blocker_if_not_pass"},
            label=f"evidence slot {slot.get('slot_id')}",
        )
    if state["unresolved_blockers"] != [slot["blocker_if_unbound"] for slot in slots]:
        raise AdjudicationError("current eight-blocker order differs")
    _expect_exact(
        evidence["negative_record_policy"],
        {
            "allowed_statuses": [UNKNOWN, "NOT_RUN", "BLOCKED"],
            "facts_must_be_null": True,
            "unknown_fields_must_equal_required_fact_keys": True,
            "reason_codes_must_be_nonempty_sorted_unique": True,
            "unknown_numeric_must_not_be_encoded_as_zero": True,
        },
        label="negative evidence policy",
    )
    predecessor = evidence["required_predecessor_authority"]
    _expect_exact_keys(
        predecessor,
        {
            "authority_type", "bundle_id", "run_root_relative_bundle_path",
            "trusted_absolute_bundle_path", "terminal_marker_final_output_target_sha256",
            "runtime_lineage_authority", "members",
        },
        label="predecessor authority",
    )
    _expect_exact(predecessor["authority_type"], "GSE200304_PUBLISHED_ENDPOINT_BUNDLE_V1", label="predecessor type")
    _expect_exact(predecessor["bundle_id"], "ROUTE_A_V3_GSE200304_PUBLISHED_ENDPOINT_A1_BUNDLE_V1", label="predecessor bundle")
    _expect_exact(
        predecessor["run_root_relative_bundle_path"],
        "GSE200304_PUBLISHED_ENDPOINT_A1_20260811T044050+0800_d06bb99",
        label="predecessor relative path",
    )
    expected_predecessor_path = (
        "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
        "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/"
        "GSE200304_PUBLISHED_ENDPOINT_A1_20260811T044050+0800_d06bb99"
    )
    _expect_exact(predecessor["trusted_absolute_bundle_path"], expected_predecessor_path, label="predecessor absolute path")
    _expect_exact(
        predecessor["terminal_marker_final_output_target_sha256"],
        sha256(expected_predecessor_path.encode("utf-8")),
        label="predecessor final-target identity",
    )
    _expect_exact(
        predecessor["runtime_lineage_authority"],
        {
            "event_id": "A1-EVT-037",
            "protocol_id": "ROUTE_A_V3_GSE200304_PUBLISHED_ENDPOINT_RUNTIME_SYNC_V1",
            "implementation_commit": "8c0376470a36c4e3496f401b8e45c829712dcc34",
            "binding_commit": "8e8b4eb41a3367b7d6cbc9513e91518b3e86f930",
            "implementation_script_sha256": "1a8c41502d18cf56885af9e830a320a361c138cfaa93c257bbea2d22b1eff38b",
            "implementation_test_sha256": "7e9f30238ec60709761e55bc892ea43bc0e0fcd9c8a1dadfe5d2c7d0f198f530",
            "compiled_core_sha256": "ae3bd1400f34dfab25bf6b7305b99944c75d5dfbbc8b6973c923e573522bc35d",
            "bound_config_path": "configs/route_a_v3_gse200304_published_endpoint_runtime_sync_v1.json",
            "bound_config_sha256": "1880c718443339b95ded247a276152678cf49cfedb6eae24e76ba8d224e40b7b",
        },
        label="predecessor EVT037 lineage",
    )
    expected_members = [
        ("INPUT_INTEGRITY_AUDIT.json", 3610, "e87723673dfea6dca654b670d1c05f331f240a53d52d81d1207fbfc50d9a4fe8"),
        ("PUBLISHED_ENDPOINT_AUDIT.json", 4981, "d849da8cc29a2a4419c85d69e5084736b6b41b03cac90263aa2620be3fe3acc7"),
        ("QUALIFICATION_REPORT.json", 2095, "006db8da47dc2bbc0c313a156ae16ab79a3f6aebe324d37806820ac9240b100d"),
        ("SHA256SUMS", 281, "e1720881f8bcfaaea1fef613dd4ee059c08da1bbd11bafc32a8fccdea0a43515"),
        ("PUBLICATION_COMMIT.json", 973, "f1e5d0752bcc12db0b0eaabe0e75efdb6f2c48dfba4c3bae6bff99a302194cfc"),
    ]
    observed_members = [
        (member.get("name"), member.get("bytes"), member.get("sha256"))
        for member in predecessor["members"]
        if type(member) is dict
    ] if type(predecessor["members"]) is list else []
    _expect_exact(observed_members, expected_members, label="predecessor members")
    provenance_contract = evidence["gate_record_provenance_contract"]
    _expect_exact_keys(
        provenance_contract,
        {
            "required", "producer_protocol_id_required", "producer_commit_required",
            "producer_script_sha256_required",
            "biological_group_pass_requires_mapping_commitment_sha256",
            "source_bundle_id_must_equal_required_predecessor",
            "source_bundle_root_or_target_sha256_required",
            "predecessor_members_must_equal_required_predecessor", "acceptance_authority",
        },
        label="gate provenance contract",
    )
    for key in set(provenance_contract) - {"acceptance_authority"}:
        _expect_exact(provenance_contract[key], True, label=f"provenance {key}")
    _expect_exact(
        provenance_contract["acceptance_authority"],
        {
            "contract_id": CONTRACT_ID,
            "decision_id": DECISION_ID,
            "protocol_id": PROTOCOL_ID,
            "rule": "CONFIG_HASH_BOUND_ACCEPTED_AGGREGATE_GATE_RECORD_V3",
        },
        label="acceptance authority",
    )

    descriptors = _expect_exact_keys(
        config["evidence_descriptor_bindings"], DESCRIPTOR_BINDING_KEYS, label="descriptor bindings"
    )
    _expect_exact(descriptors["binding_scheme"], "DESCENDANT_CONFIG_ONLY_DESCRIPTOR_BINDING_V1", label="descriptor scheme")
    _expect_exact(descriptors["dynamic_scalar_suffixes"], ["absolute_path", "sha256", "bytes"], label="descriptor suffixes")
    _expect_exact(descriptors["all_descriptors_required_before_any_input_open"], True, label="descriptor all-before-open")
    descriptor_slots = descriptors["slots"]
    if type(descriptor_slots) is not list or tuple(slot.get("slot_id") for slot in descriptor_slots) != SLOT_IDS:
        raise AdjudicationError("descriptor slot IDs or order differ")
    for slot in descriptor_slots:
        _expect_exact_keys(slot, DESCRIPTOR_SLOT_KEYS, label=f"descriptor {slot.get('slot_id')}")
        if not (_descriptor_slot_bound(slot) or _descriptor_slot_unbound(slot)):
            raise AdjudicationError(f"descriptor has a partially bound triple: {slot.get('slot_id')}")
    _expect_exact(descriptors["status"], _derived_descriptor_status(config), label="descriptor status")
    if HEX64.fullmatch(str(descriptors["descriptor_set_sha256"])) is None:
        raise AdjudicationError("descriptor-set SHA is not bound")
    _expect_exact(descriptors["descriptor_set_sha256"], descriptor_set_sha256(config), label="descriptor-set SHA")

    output = _expect_exact_keys(
        config["output_contract"],
        {
            "output_id", "blocked_status", "success_status", "aggregate_only",
            "member_names_excluding_commit_marker", "terminal_commit_marker", "publication_mode",
            "commit_marker_written_last", "existing_exact_is_idempotent", "overwrite_allowed",
            "partial_publication_is_never_accepted", "forbidden_output_keys",
        },
        label="output contract",
    )
    _expect_exact(output["output_id"], "ROUTE_A_V3_GSE200304_DEC019_REPORTED_ENDPOINT_A1_ADJUDICATION_BUNDLE_V3", label="output ID")
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
    descriptors = config["evidence_descriptor_bindings"]
    return descriptors["status"] == "BOUND" and all(
        _descriptor_slot_bound(slot) for slot in descriptors["slots"]
    )


def _descriptor_by_slot(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        slot["slot_id"]: slot
        for slot in config["evidence_descriptor_bindings"]["slots"]
    }


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



def _commit_changed_paths(repo: Path, commit: str) -> list[str]:
    return sorted(
        line
        for line in _git(
            repo, "diff-tree", "--no-commit-id", "--name-only", "-r", commit
        ).splitlines()
        if line
    )


def _require_ancestor(repo: Path, ancestor: str, descendant: str, *, label: str) -> None:
    _git(repo, "merge-base", "--is-ancestor", ancestor, descendant)
    # merge-base emits no identity, so also require both names resolve exactly.
    if _git(repo, "rev-parse", ancestor) != ancestor:
        raise BindingError(f"{label} ancestor does not resolve exactly")
    if _git(repo, "rev-parse", descendant) != descendant:
        raise BindingError(f"{label} descendant does not resolve exactly")


def _verify_commit_blob(repo: Path, commit: str, path: str, expected_sha: str) -> bytes:
    payload = _git_bytes(repo, "show", f"{commit}:{path}")
    if sha256(payload) != expected_sha:
        raise BindingError(f"Git blob SHA differs at {commit}: {path}")
    return payload


def _require_single_parent(
    repo: Path,
    commit: str,
    expected_parent: str,
    *,
    label: str,
) -> None:
    lineage = _git(repo, "rev-list", "--parents", "-n", "1", commit).split()
    if lineage != [commit, expected_parent]:
        raise BindingError(f"{label} is not an exact single-parent commit")


def _validate_historical_chain_and_blobs(repo: Path, head: str) -> None:
    _require_single_parent(
        repo,
        HISTORICAL_IMPLEMENTATION_COMMIT,
        HISTORICAL_BASE_COMMIT,
        label="historical DEC019 implementation",
    )
    _require_single_parent(
        repo,
        HISTORICAL_BINDING_COMMIT,
        HISTORICAL_IMPLEMENTATION_COMMIT,
        label="historical DEC019 binding",
    )
    _require_ancestor(
        repo,
        HISTORICAL_BINDING_COMMIT,
        REPAIR_BASE_COMMIT,
        label="historical-v3-to-successor-base",
    )
    _require_ancestor(repo, HISTORICAL_BINDING_COMMIT, head, label="historical-to-current")
    _require_ancestor(repo, REPAIR_BASE_COMMIT, head, label="successor-base-to-current")
    historical_payloads = {
        path: _verify_commit_blob(
            repo,
            HISTORICAL_BINDING_COMMIT,
            path,
            expected_sha,
        )
        for path, expected_sha in HISTORICAL_FROZEN_BLOBS.items()
    }
    historical_config = strict_json(
        historical_payloads[CONFIG_REPO_PATH],
        label="historical DEC019 v3 bound config",
    )
    historical_binding = historical_config.get("implementation_binding")
    if type(historical_binding) is not dict:
        raise BindingError("historical DEC019 v3 implementation binding is absent")
    if (
        historical_binding.get("status") != "BOUND"
        or historical_binding.get("implementation_commit")
        != HISTORICAL_IMPLEMENTATION_COMMIT
        or historical_binding.get("implementation_script_sha256")
        != HISTORICAL_FROZEN_BLOBS[SCRIPT_REPO_PATH]
        or historical_binding.get("implementation_test_sha256")
        != HISTORICAL_FROZEN_BLOBS[TEST_REPO_PATH]
        or historical_binding.get("config_core_sha256")
        != HISTORICAL_CONFIG_CORE_SHA256
        or config_core_sha256(historical_config) != HISTORICAL_CONFIG_CORE_SHA256
    ):
        raise BindingError("historical DEC019 v3 bound implementation differs")


def _validate_predecessor_d1_descriptor_binding(repo: Path) -> dict[str, Any]:
    """Prove the exact config-only D1 transition that anchors successor I3."""

    _require_single_parent(
        repo,
        PREDECESSOR_D1_COMMIT,
        PREDECESSOR_D1_PARENT_COMMIT,
        label="predecessor D1 descriptor binding",
    )
    if _commit_changed_paths(repo, PREDECESSOR_D1_COMMIT) != (
        PREDECESSOR_D1_EXACT_CHANGED_PATHS
    ):
        raise BindingError("predecessor D1 is not the exact config-only commit")

    parent_payload = _verify_commit_blob(
        repo,
        PREDECESSOR_D1_PARENT_COMMIT,
        CONFIG_REPO_PATH,
        PREDECESSOR_D1_PARENT_CONFIG_SHA256,
    )
    d1_payloads = {
        path: _verify_commit_blob(
            repo,
            PREDECESSOR_D1_COMMIT,
            path,
            expected_sha,
        )
        for path, expected_sha in PREDECESSOR_D1_FROZEN_BLOBS.items()
    }
    parent_config = strict_json(parent_payload, label="predecessor D1 parent config")
    d1_config = strict_json(
        d1_payloads[CONFIG_REPO_PATH],
        label="predecessor D1 descriptor config",
    )
    for value, label in (
        (parent_config, "predecessor D1 parent"),
        (d1_config, "predecessor D1 descriptor"),
    ):
        if set(value) != EXPECTED_CONFIG_TOP_KEYS:
            raise BindingError(f"{label} config top-level schema differs")
        binding = value.get("implementation_binding")
        if type(binding) is not dict or set(binding) != EXPECTED_IMPLEMENTATION_BINDING_KEYS:
            raise BindingError(f"{label} implementation binding schema differs")
        if (
            binding.get("status") != "BOUND"
            or binding.get("implementation_commit")
            != PREDECESSOR_D1_IMPLEMENTATION_COMMIT
            or binding.get("implementation_script_sha256")
            != PREDECESSOR_D1_FROZEN_BLOBS[SCRIPT_REPO_PATH]
            or binding.get("implementation_test_sha256")
            != PREDECESSOR_D1_FROZEN_BLOBS[TEST_REPO_PATH]
            or binding.get("config_core_sha256")
            != PREDECESSOR_D1_CONFIG_CORE_SHA256
            or config_core_sha256(value) != PREDECESSOR_D1_CONFIG_CORE_SHA256
        ):
            raise BindingError(f"{label} implementation or science core differs")

    if parent_config["implementation_binding"] != d1_config["implementation_binding"]:
        raise BindingError("predecessor D1 changed the implementation binding")
    differences = _semantic_diff_paths(parent_config, d1_config)
    if differences != PREDECESSOR_D1_SEMANTIC_DIFF_PATHS:
        raise BindingError(
            "predecessor D1 semantic diff is not the exact five-scalar allowlist"
        )
    parent_descriptors = parent_config.get("evidence_descriptor_bindings")
    d1_descriptors = d1_config.get("evidence_descriptor_bindings")
    if (
        type(parent_descriptors) is not dict
        or parent_descriptors.get("status") != "UNBOUND"
        or descriptor_set_sha256(parent_config)
        != parent_descriptors.get("descriptor_set_sha256")
    ):
        raise BindingError("predecessor D1 parent descriptor state differs")
    if (
        type(d1_descriptors) is not dict
        or d1_descriptors.get("status") != "PARTIALLY_BOUND"
        or d1_descriptors.get("descriptor_set_sha256")
        != PREDECESSOR_D1_DESCRIPTOR_SET_SHA256
        or descriptor_set_sha256(d1_config)
        != PREDECESSOR_D1_DESCRIPTOR_SET_SHA256
        or _derived_descriptor_status(d1_config) != "PARTIALLY_BOUND"
    ):
        raise BindingError("predecessor D1 descriptor state differs")
    return d1_config


def _validate_predecessor_i3_implementation_successor(
    repo: Path,
    d1_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the exact UNKNOWN implementation successor used as I4 base."""

    _require_single_parent(
        repo,
        PREDECESSOR_I3_COMMIT,
        PREDECESSOR_I3_PARENT_COMMIT,
        label="predecessor I3 implementation successor",
    )
    if _commit_changed_paths(repo, PREDECESSOR_I3_COMMIT) != (
        PREDECESSOR_I3_EXACT_CHANGED_PATHS
    ):
        raise BindingError("predecessor I3 is not the exact three-file commit")
    i3_payloads = {
        path: _verify_commit_blob(
            repo,
            PREDECESSOR_I3_COMMIT,
            path,
            expected_sha,
        )
        for path, expected_sha in PREDECESSOR_I3_FROZEN_BLOBS.items()
    }
    i3_config = strict_json(
        i3_payloads[CONFIG_REPO_PATH],
        label="predecessor I3 implementation config",
    )
    if set(i3_config) != EXPECTED_CONFIG_TOP_KEYS:
        raise BindingError("predecessor I3 config top-level schema differs")
    binding = i3_config.get("implementation_binding")
    if type(binding) is not dict or set(binding) != EXPECTED_IMPLEMENTATION_BINDING_KEYS:
        raise BindingError("predecessor I3 implementation binding schema differs")
    if binding.get("status") != UNKNOWN or any(
        binding.get(key) != UNKNOWN
        for key in (
            "implementation_commit",
            "implementation_script_sha256",
            "implementation_test_sha256",
        )
    ):
        raise BindingError("predecessor I3 implementation binding is not exact UNKNOWN")
    if (
        binding.get("config_core_sha256") != PREDECESSOR_I3_CONFIG_CORE_SHA256
        or config_core_sha256(i3_config) != PREDECESSOR_I3_CONFIG_CORE_SHA256
    ):
        raise BindingError("predecessor I3 science core differs")
    descriptors = i3_config.get("evidence_descriptor_bindings")
    if (
        descriptors != d1_config.get("evidence_descriptor_bindings")
        or type(descriptors) is not dict
        or descriptors.get("descriptor_set_sha256")
        != PREDECESSOR_I3_DESCRIPTOR_SET_SHA256
        or descriptor_set_sha256(i3_config)
        != PREDECESSOR_I3_DESCRIPTOR_SET_SHA256
    ):
        raise BindingError("predecessor I3 descriptor binding drifted from D1")
    repository = i3_config.get("repository_authority")
    if (
        type(repository) is not dict
        or repository.get("base_commit") != PREDECESSOR_I3_PARENT_COMMIT
        or repository.get("implementation_commit_expected_parent")
        != PREDECESSOR_I3_PARENT_COMMIT
        or repository.get("implementation_commit_exact_changed_paths")
        != PREDECESSOR_I3_EXACT_CHANGED_PATHS
    ):
        raise BindingError("predecessor I3 repository authority differs")
    return i3_config


def _validate_successor_i_transition_from_base(
    i_config: Mapping[str, Any],
    base_config: Mapping[str, Any],
) -> None:
    """Allow only this upgrade's requirement and lifecycle changes from f492."""

    if i_config.get("evidence_descriptor_bindings") != base_config.get(
        "evidence_descriptor_bindings"
    ):
        raise BindingError("successor I descriptor binding drifted from f492 base")
    differences = _semantic_diff_paths(base_config, i_config)
    if differences != EXPECTED_BASE_TO_I_DIFF_PATHS:
        raise BindingError(
            "f492-to-successor-I semantic diff is not the exact allowlist"
        )


def _validate_post_binding_descriptor_history(
    repo: Path,
    binding_commit: str,
    head: str,
) -> None:
    commits = [
        line
        for line in _git(
            repo,
            "rev-list",
            "--ancestry-path",
            "--reverse",
            f"{binding_commit}..{head}",
        ).splitlines()
        if line
    ]
    config_paths = set(BINDING_CONFIG_REPO_PATHS)
    for commit in commits:
        lineage = _git(repo, "rev-list", "--parents", "-n", "1", commit).split()
        if len(lineage) != 2 or lineage[0] != commit:
            raise BindingError(f"post-B history contains a merge or parentless commit: {commit}")
        changed = set(_commit_changed_paths(repo, commit))
        relevant = changed & config_paths
        if not relevant:
            continue
        if not changed.issubset(config_paths):
            raise BindingError(
                f"post-B descriptor commit also changed non-config paths: {commit}"
            )
        parent = lineage[1]
        for config_path in sorted(relevant):
            before = strict_json(
                _git_bytes(repo, "show", f"{parent}:{config_path}"),
                label=f"descriptor parent {config_path}",
            )
            after = strict_json(
                _git_bytes(repo, "show", f"{commit}:{config_path}"),
                label=f"descriptor commit {config_path}",
            )
            _validate_descriptor_only_transition(
                before,
                after,
                config_path=config_path,
            )
            if config_path == CONFIG_REPO_PATH:
                validate_static_config(after)


def validate_production_authority(config: Mapping[str, Any]) -> dict[str, Any]:
    """Prove the f492 base, exact-three-file I, config-only B, and descendants.

    UNKNOWN I-state is accepted only by this authority-only validator.  Actual
    adjudication separately requires a BOUND implementation before it can
    reach evidence or output handling.
    """

    validate_static_config(config)
    repo = Path(config["repository_authority"]["production_repo_root"])
    if repo != PRODUCTION_REPO_ROOT:
        raise BindingError("production repository root differs")
    branch = config["repository_authority"]["branch"]
    head = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "rev-parse", "--abbrev-ref", "HEAD") != branch:
        raise BindingError("production branch differs")
    if _git(repo, "status", "--porcelain"):
        raise BindingError("production worktree is not clean")
    if _git(repo, "rev-parse", f"refs/remotes/origin/{branch}") != head:
        raise BindingError("origin tracking ref is not current HEAD")
    _require_ancestor(repo, REPAIR_BASE_COMMIT, head, label="f492-base-to-current")
    base_config = strict_json(
        _git_bytes(repo, "show", f"{REPAIR_BASE_COMMIT}:{CONFIG_REPO_PATH}"),
        label="f492 consumer config",
    )
    if set(base_config) != EXPECTED_CONFIG_TOP_KEYS:
        raise BindingError("f492 consumer config top-level schema differs")

    expected_i_paths = config["repository_authority"][
        "implementation_commit_exact_changed_paths"
    ]
    binding = config["implementation_binding"]
    lifecycle_state: str
    binding_commit: str
    if binding["status"] == UNKNOWN:
        implementation = head
        binding_commit = UNKNOWN
        lifecycle_state = "REPAIR_I_IMPLEMENTATION_UNBOUND"
        _require_single_parent(
            repo,
            implementation,
            REPAIR_BASE_COMMIT,
            label="repair I",
        )
        if _commit_changed_paths(repo, implementation) != expected_i_paths:
            raise BindingError("repair I is not the exact three-file implementation commit")
        _validate_successor_i_transition_from_base(
            config,
            base_config,
        )
    else:
        implementation = binding["implementation_commit"]
        lifecycle_state = "REPAIR_B_BOUND_OR_DESCRIPTOR_DESCENDANT"
        _require_single_parent(
            repo,
            implementation,
            REPAIR_BASE_COMMIT,
            label="repair I",
        )
        if _commit_changed_paths(repo, implementation) != expected_i_paths:
            raise BindingError("repair I is not the exact three-file implementation commit")
        _require_ancestor(repo, implementation, head, label="repair-I-to-current")
        successors = [
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
        if not successors:
            raise BindingError("repair B is absent")
        binding_commit = successors[0]
        _require_single_parent(
            repo,
            binding_commit,
            implementation,
            label="repair B",
        )
        if _commit_changed_paths(repo, binding_commit) != list(BINDING_CONFIG_REPO_PATHS):
            raise BindingError("repair B is not the exact config-only binding commit")

        for config_path in BINDING_CONFIG_REPO_PATHS:
            i_config = strict_json(
                _git_bytes(repo, "show", f"{implementation}:{config_path}"),
                label=f"repair-I config {config_path}",
            )
            b_config = strict_json(
                _git_bytes(repo, "show", f"{binding_commit}:{config_path}"),
                label=f"repair-B config {config_path}",
            )
            if config_path == CONFIG_REPO_PATH:
                _validate_successor_i_transition_from_base(
                    i_config,
                    base_config,
                )
            _validate_i_to_b_config_pair(
                i_config,
                b_config,
                config_path=config_path,
                implementation_commit=implementation,
            )
            script_path, test_path = EXPECTED_IMPLEMENTATION_FILES[config_path]
            b_binding = b_config["implementation_binding"]
            for path, digest in (
                (script_path, b_binding["implementation_script_sha256"]),
                (test_path, b_binding["implementation_test_sha256"]),
            ):
                implementation_blob = _verify_commit_blob(repo, implementation, path, digest)
                current_blob = _verify_commit_blob(repo, head, path, digest)
                if implementation_blob != current_blob:
                    raise BindingError(f"bound implementation blob drifted: {path}")
        _validate_post_binding_descriptor_history(repo, binding_commit, head)

    current_configs: dict[str, Mapping[str, Any]] = {}
    for config_path in BINDING_CONFIG_REPO_PATHS:
        current_payload = _git_bytes(repo, "show", f"{head}:{config_path}")
        try:
            working_payload = (repo / config_path).read_bytes()
        except OSError as exc:
            raise BindingError(f"working config is unavailable: {config_path}") from exc
        if current_payload != working_payload:
            raise BindingError(f"working config differs from current HEAD: {config_path}")
        current_config = strict_json(
            current_payload,
            label=f"current config {config_path}",
        )
        if type(current_config) is not dict or set(current_config) != EXPECTED_CONFIG_TOP_KEYS:
            raise BindingError(f"current config top-level schema differs: {config_path}")
        binding_value = current_config.get("implementation_binding")
        if (
            type(binding_value) is not dict
            or set(binding_value) != EXPECTED_IMPLEMENTATION_BINDING_KEYS
        ):
            raise BindingError(f"current implementation binding schema differs: {config_path}")
        frozen_core = FROZEN_CONFIG_CORE_SHA256_BY_PATH[config_path]
        if (
            config_core_sha256(current_config) != frozen_core
            or binding_value.get("config_core_sha256") != frozen_core
        ):
            raise BindingError(f"current science core differs from compiled authority: {config_path}")
        current_configs[config_path] = current_config
    if current_configs[CONFIG_REPO_PATH] != dict(config):
        raise BindingError("in-memory GSE200304 config differs from current HEAD")
    for path, digest in _core_leaf_pairs(config):
        _verify_repo_file(repo, path, digest)
    return {
        "mode": "PRODUCTION_GIT_AUTHORITY",
        "lifecycle_state": lifecycle_state,
        "historical_base_commit": HISTORICAL_BASE_COMMIT,
        "historical_implementation_commit": HISTORICAL_IMPLEMENTATION_COMMIT,
        "historical_binding_commit": HISTORICAL_BINDING_COMMIT,
        "repair_base_commit": REPAIR_BASE_COMMIT,
        "repair_implementation_commit": implementation,
        "repair_binding_commit": binding_commit,
        "current_head": head,
        "science_core_sha256": config["implementation_binding"]["config_core_sha256"],
        "evidence_descriptor_set_sha256": config["evidence_descriptor_bindings"]["descriptor_set_sha256"],
        "predecessor_authority_sha256": sha256(
            json_bytes(config["evidence_contract"]["required_predecessor_authority"])
        ),
    }


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
        if (
            facts["locator_lineage_commitment_algorithm"]
            != LOCATOR_LINEAGE_COMMITMENT_ALGORITHM
        ):
            raise AdjudicationError("locator lineage commitment algorithm differs")
        merkle_root = facts["locator_lineage_merkle_root_sha256"]
        if type(merkle_root) is not str or HEX64.fullmatch(merkle_root) is None:
            raise AdjudicationError(
                "locator lineage Merkle root must be a lowercase SHA-256 digest"
            )
        if facts["raw_replay_role"] != "REPRODUCIBILITY_AUXILIARY_NOT_QUALIFICATION_PREREQUISITE":
            raise AdjudicationError("raw replay role differs from DEC-019")
        if facts["raw_replay_status"] not in {"NOT_RUN", "PASS_INDEPENDENT_REPRODUCTION"}:
            raise AdjudicationError("raw replay status is outside the closed enum")
        boolean_keys.remove("raw_replay_role")
        boolean_keys.remove("raw_replay_status")
        boolean_keys.remove("locator_lineage_commitment_algorithm")
        boolean_keys.remove("locator_lineage_merkle_root_sha256")
    elif slot_id == "CHECKPOINT_SPECIFIC_EXPOSURE":
        _expect_int(facts["audited_checkpoint_count"], label="audited checkpoint count", minimum=0)
        boolean_keys.remove("audited_checkpoint_count")
    elif slot_id == "LICENSE_RIGHTS":
        if facts["redistribution_scope"] not in {"PRIVATE_CANONICAL_ONLY", "PUBLIC_REDISTRIBUTION_ALLOWED"}:
            raise AdjudicationError("LICENSE_RIGHTS redistribution_scope is outside the closed enum")
        boolean_keys.remove("redistribution_scope")
    elif slot_id == "PREFROZEN_POWER_PRECISION":
        observed_power = _expect_number(facts["observed_power"], label="observed power")
        full_width = _expect_number(
            facts["full_confidence_interval_width"], label="full CI width"
        )
        if not 0.0 <= observed_power <= 1.0:
            raise AdjudicationError("observed power must lie in [0, 1]")
        if full_width < 0.0:
            raise AdjudicationError("full CI width must be non-negative")
        if facts["analysis_unit"] != "BIOLOGICAL_GROUP" or facts["bootstrap_unit"] != "BIOLOGICAL_GROUP":
            raise AdjudicationError("power analysis/bootstrap unit differs")
        boolean_keys -= {"analysis_unit", "bootstrap_unit", "observed_power", "full_confidence_interval_width"}
    for key in boolean_keys:
        _expect_bool(facts[key], label=f"{slot_id} {key}")


def _validate_provenance(
    record: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    label: str,
    slot_id: str,
) -> None:
    required_keys = set(PROVENANCE_KEYS)
    if slot_id == "BIOLOGICAL_GROUP_AUTHORITY" and record["status"] == "PASS":
        required_keys.add(GROUP_MAPPING_COMMITMENT_KEY)
    provenance = _expect_exact_keys(
        record["provenance"],
        required_keys,
        label=f"{label} provenance",
    )
    if GROUP_MAPPING_COMMITMENT_KEY in required_keys:
        commitment = provenance[GROUP_MAPPING_COMMITMENT_KEY]
        if type(commitment) is not str or HEX64.fullmatch(commitment) is None:
            raise AdjudicationError(
                f"{label} biological-group mapping commitment is not bound"
            )
    if type(provenance["producer_protocol_id"]) is not str or not provenance["producer_protocol_id"]:
        raise AdjudicationError(f"{label} producer protocol is absent")
    if HEX40.fullmatch(str(provenance["producer_commit"])) is None:
        raise AdjudicationError(f"{label} producer commit is not bound")
    if HEX64.fullmatch(str(provenance["producer_script_sha256"])) is None:
        raise AdjudicationError(f"{label} producer script SHA is not bound")
    predecessor = config["evidence_contract"]["required_predecessor_authority"]
    _expect_exact(
        provenance["source_bundle_id"],
        predecessor["bundle_id"],
        label=f"{label} source bundle ID",
    )
    _expect_exact(
        provenance["source_bundle_root_or_target_sha256"],
        predecessor["terminal_marker_final_output_target_sha256"],
        label=f"{label} source target identity",
    )
    # Exact object equality freezes the locator, EVT037 lineage, and all five
    # predecessor members.  A copied five-file bundle at another target cannot
    # self-prove acceptance.
    _expect_exact(
        provenance["predecessor_authority"],
        predecessor,
        label=f"{label} predecessor authority",
    )
    _expect_exact(
        provenance["acceptance_authority"],
        config["evidence_contract"]["gate_record_provenance_contract"]["acceptance_authority"],
        label=f"{label} acceptance authority",
    )


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
    if record["status"] not in {"PASS", *NEGATIVE_EVIDENCE_STATUSES}:
        raise AdjudicationError(f"{label} status is outside the closed enum")
    _validate_provenance(
        record,
        config,
        label=label,
        slot_id=slot["slot_id"],
    )
    required_fact_keys = sorted(FACT_KEYS[slot["slot_id"]])
    if record["status"] == "PASS":
        facts = _expect_exact_keys(
            record["facts"], FACT_KEYS[slot["slot_id"]], label=f"{label} facts"
        )
        _validate_fact_types(slot["slot_id"], facts)
        _expect_exact(record["unknown_fields"], [], label=f"{label} PASS unknown fields")
        _expect_exact(record["reason_codes"], [], label=f"{label} PASS reason codes")
    else:
        if record["facts"] is not None:
            raise AdjudicationError(
                f"{label} negative status requires facts=null; numeric zero is not unknown"
            )
        _expect_exact(
            record["unknown_fields"],
            required_fact_keys,
            label=f"{label} negative unknown fields",
        )
        reasons = record["reason_codes"]
        if (
            type(reasons) is not list
            or not reasons
            or reasons != sorted(set(reasons))
            or any(type(reason) is not str or REASON_CODE.fullmatch(reason) is None for reason in reasons)
        ):
            raise AdjudicationError(f"{label} negative reason codes are not closed/sorted/unique")
    _validate_privacy(record, config, label=label)
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
            and facts["locator_lineage_commitment_algorithm"]
            == LOCATOR_LINEAGE_COMMITMENT_ALGORITHM
            and type(facts["locator_lineage_merkle_root_sha256"]) is str
            and HEX64.fullmatch(facts["locator_lineage_merkle_root_sha256"])
            is not None
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
        if record["status"] != "PASS":
            blockers.append(slot["blocker_if_not_pass"])
            continue
        if not _slot_gate_pass(slot["slot_id"], record["facts"]):
            blockers.append(slot["blocker_if_not_pass"])
    lineage_record = records["CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE"]
    lineage = lineage_record["facts"] if lineage_record["status"] == "PASS" else None
    raw_status = lineage["raw_replay_status"] if lineage is not None else UNKNOWN
    raw_claimed = lineage["independent_raw_reproduction_claimed"] if lineage is not None else False
    if lineage is not None and (
        (raw_status == "NOT_RUN" and raw_claimed is not False)
        or (
            raw_status == "PASS_INDEPENDENT_REPRODUCTION"
            and raw_claimed is not True
        )
    ):
        blockers.append("RAW_REPLAY_INDEPENDENT_REPRODUCTION_CLAIM_INVALID")
    power_record = records["PREFROZEN_POWER_PRECISION"]
    if power_record["status"] == "PASS":
        power = power_record["facts"]
        if float(power["observed_power"]) < 0.8:
            blockers.append("POWER_LT_0_80")
        if float(power["full_confidence_interval_width"]) > 0.3:
            blockers.append("FULL_CI_WIDTH_GT_0_30")
    blockers = sorted(set(blockers))
    if not set(blockers).issubset(GATE_BLOCKERS):
        raise AdjudicationError("an unregistered blocker was produced")
    canonical_count = int(lineage["canonical_record_count"]) if not blockers and lineage is not None else 0
    independent_reproduction = raw_status == "PASS_INDEPENDENT_REPRODUCTION" and raw_claimed is True
    return blockers, canonical_count, independent_reproduction


def _synthetic_authority_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mode": "SYNTHETIC_NON_PRODUCTION",
        "lifecycle_state": "BOUND_CONFIG_SYNTHETIC_EXECUTION",
        "historical_base_commit": HISTORICAL_BASE_COMMIT,
        "historical_implementation_commit": HISTORICAL_IMPLEMENTATION_COMMIT,
        "historical_binding_commit": HISTORICAL_BINDING_COMMIT,
        "repair_base_commit": REPAIR_BASE_COMMIT,
        "repair_implementation_commit": config["implementation_binding"]["implementation_commit"],
        "repair_binding_commit": UNKNOWN,
        "current_head": UNKNOWN,
        "science_core_sha256": config["implementation_binding"]["config_core_sha256"],
        "evidence_descriptor_set_sha256": config["evidence_descriptor_bindings"]["descriptor_set_sha256"],
        "predecessor_authority_sha256": sha256(
            json_bytes(config["evidence_contract"]["required_predecessor_authority"])
        ),
    }


def _blocked_report(
    config: Mapping[str, Any],
    blockers: Sequence[str],
    authority_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "record_type": "ROUTE_A_V3_GSE200304_DEC019_REPORTED_ENDPOINT_A1_ADJUDICATION_REPORT_V3",
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
        "evidence_descriptor_set_sha256": config["evidence_descriptor_bindings"]["descriptor_set_sha256"],
        "authority_provenance": dict(authority_provenance),
    }


def _success_report(
    config: Mapping[str, Any],
    canonical_count: int,
    independent_reproduction: bool,
    authority_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "record_type": "ROUTE_A_V3_GSE200304_DEC019_REPORTED_ENDPOINT_A1_ADJUDICATION_REPORT_V3",
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
        "evidence_descriptor_set_sha256": config["evidence_descriptor_bindings"]["descriptor_set_sha256"],
        "authority_provenance": dict(authority_provenance),
    }


def _input_audit(
    config: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]] | None,
    authority_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    descriptor_by_slot = _descriptor_by_slot(config)
    slots = []
    for slot in config["evidence_contract"]["slots"]:
        bound = _descriptor_slot_bound(descriptor_by_slot[slot["slot_id"]])
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
        "evidence_descriptor_set_sha256": config["evidence_descriptor_bindings"]["descriptor_set_sha256"],
        "authority_provenance": dict(authority_provenance),
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
        "record_type": "ROUTE_A_V3_GSE200304_DEC019_REPORTED_ENDPOINT_A1_ADJUDICATION_COMMIT_V3",
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


def _write_exclusive_at(directory_fd: int, name: str, payload: bytes) -> None:
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
            written = os.write(descriptor, view)
            if written <= 0:
                raise PublicationError(f"short write: {name}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_regular_at(directory_fd: int, name: str) -> bytes:
    if Path(name).name != name:
        raise PublicationError("published member name is unsafe")
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise PublicationError(f"published member is unsafe: {name}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise PublicationError(f"published member changed during read: {name}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_directory_root_to_leaf(path: Path, *, label: str) -> int:
    if not path.is_absolute():
        raise ScopeViolation(f"{label} must be absolute")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise PublicationError("O_NOFOLLOW is unavailable")
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
                raise PublicationError(
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


def _open_child_directory(parent_fd: int, name: str, *, label: str) -> int:
    if Path(name).name != name:
        raise ScopeViolation(f"{label} name is unsafe")
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


def _assert_named_directory_identity(
    parent_fd: int,
    name: str,
    directory_fd: int,
    *,
    label: str,
) -> None:
    try:
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise PublicationError(f"{label} disappeared during operation") from exc
    opened = os.fstat(directory_fd)
    if (
        not stat.S_ISDIR(named.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        raise PublicationError(f"{label} identity changed during operation")


def _directory_identity(directory_fd: int) -> tuple[int, int]:
    metadata = os.fstat(directory_fd)
    if not stat.S_ISDIR(metadata.st_mode):
        raise PublicationError("pinned directory descriptor is not a directory")
    return metadata.st_dev, metadata.st_ino


def _assert_canonical_directory_identity(
    path: Path,
    expected_identity: tuple[int, int],
    *,
    label: str,
) -> None:
    """Re-open an absolute path root-to-leaf and match its pinned inode."""

    descriptor = _open_directory_root_to_leaf(path, label=label)
    try:
        observed_identity = _directory_identity(descriptor)
    finally:
        os.close(descriptor)
    if observed_identity != expected_identity:
        raise PublicationError(f"{label} identity changed during operation")


def _assert_active_output_identity(
    parent_path: Path,
    parent_fd: int,
    parent_identity: tuple[int, int],
    output_name: str,
    output_fd: int,
) -> None:
    _assert_canonical_directory_identity(
        parent_path,
        parent_identity,
        label="output parent",
    )
    _assert_named_directory_identity(
        parent_fd,
        output_name,
        output_fd,
        label="output directory",
    )


def _assert_reopened_output_identity(
    output: Path,
    parent_identity: tuple[int, int],
    output_identity: tuple[int, int],
) -> None:
    parent_fd = _open_directory_root_to_leaf(output.parent, label="output parent")
    output_fd = -1
    try:
        if _directory_identity(parent_fd) != parent_identity:
            raise PublicationError("output parent identity changed during operation")
        output_fd = _open_child_directory(
            parent_fd,
            output.name,
            label="output directory",
        )
        if _directory_identity(output_fd) != output_identity:
            raise PublicationError("output directory identity changed during operation")
        _assert_named_directory_identity(
            parent_fd,
            output.name,
            output_fd,
            label="output directory",
        )
    finally:
        if output_fd >= 0:
            os.close(output_fd)
        os.close(parent_fd)


def _assert_directory_chain_nofollow(path: Path, *, label: str) -> None:
    descriptor = _open_directory_root_to_leaf(path, label=label)
    os.close(descriptor)


def _validate_output_authority_provenance(
    value: Any,
    *,
    config: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
) -> dict[str, Any]:
    authority = _expect_exact_keys(
        value,
        AUTHORITY_PROVENANCE_KEYS,
        label="output authority provenance",
    )
    if expected is not None:
        _expect_exact(authority, dict(expected), label="expected output authority provenance")
    if authority["mode"] not in {"PRODUCTION_GIT_AUTHORITY", "SYNTHETIC_NON_PRODUCTION"}:
        raise PublicationError("output authority mode is outside the closed enum")
    for key, expected_commit in {
        "historical_base_commit": HISTORICAL_BASE_COMMIT,
        "historical_implementation_commit": HISTORICAL_IMPLEMENTATION_COMMIT,
        "historical_binding_commit": HISTORICAL_BINDING_COMMIT,
        "repair_base_commit": REPAIR_BASE_COMMIT,
    }.items():
        _expect_exact(authority[key], expected_commit, label=f"output authority {key}")
    for key in {"repair_implementation_commit", "repair_binding_commit", "current_head"}:
        if authority[key] != UNKNOWN and HEX40.fullmatch(str(authority[key])) is None:
            raise PublicationError(f"output authority commit is invalid: {key}")
    for key in {
        "science_core_sha256",
        "evidence_descriptor_set_sha256",
        "predecessor_authority_sha256",
    }:
        if HEX64.fullmatch(str(authority[key])) is None:
            raise PublicationError(f"output authority SHA is invalid: {key}")
    _expect_exact(
        authority["science_core_sha256"],
        FROZEN_CONFIG_CORE_SHA256,
        label="output authority science core",
    )
    if config is not None:
        _expect_exact(
            authority["evidence_descriptor_set_sha256"],
            config["evidence_descriptor_bindings"]["descriptor_set_sha256"],
            label="output authority descriptor set",
        )
        _expect_exact(
            authority["predecessor_authority_sha256"],
            sha256(json_bytes(config["evidence_contract"]["required_predecessor_authority"])),
            label="output predecessor authority",
        )
    return authority


def _validate_published_report_and_audit(
    report: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None,
    expected_authority_provenance: Mapping[str, Any] | None,
) -> None:
    report_value = _expect_exact_keys(report, REPORT_KEYS, label="published report")
    audit_value = _expect_exact_keys(audit, AUDIT_KEYS, label="published audit")
    identities = {
        "contract_id": CONTRACT_ID,
        "decision_id": DECISION_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
    }
    for value, label in ((report_value, "report"), (audit_value, "audit")):
        for key, expected_value in identities.items():
            _expect_exact(value[key], expected_value, label=f"published {label} {key}")
    _expect_exact(
        report_value["record_type"],
        "ROUTE_A_V3_GSE200304_DEC019_REPORTED_ENDPOINT_A1_ADJUDICATION_REPORT_V3",
        label="published report type",
    )
    _expect_exact(
        audit_value["record_type"],
        "ROUTE_A_V3_DEC019_AGGREGATE_INPUT_EVIDENCE_AUDIT_V1",
        label="published audit type",
    )
    report_authority = _validate_output_authority_provenance(
        report_value["authority_provenance"],
        config=config,
        expected=expected_authority_provenance,
    )
    audit_authority = _validate_output_authority_provenance(
        audit_value["authority_provenance"],
        config=config,
        expected=expected_authority_provenance,
    )
    _expect_exact(audit_authority, report_authority, label="report/audit authority provenance")
    _expect_exact(
        report_value["config_core_sha256"],
        FROZEN_CONFIG_CORE_SHA256,
        label="published science core",
    )
    if HEX64.fullmatch(str(report_value["evidence_descriptor_set_sha256"])) is None:
        raise PublicationError("published report descriptor SHA is invalid")
    _expect_exact(
        audit_value["evidence_descriptor_set_sha256"],
        report_value["evidence_descriptor_set_sha256"],
        label="report/audit descriptor SHA",
    )
    if config is not None:
        _expect_exact(
            report_value["evidence_descriptor_set_sha256"],
            config["evidence_descriptor_bindings"]["descriptor_set_sha256"],
            label="published/config descriptor SHA",
        )
        _validate_output_privacy(report_value, config)
        _validate_output_privacy(audit_value, config)

    fixed_common = {
        "primary_measurement_route": "AUTHOR_PUBLISHED_PROCESSED_ENDPOINT",
        "raw_replay_role": "REPRODUCIBILITY_AUXILIARY_NOT_QUALIFICATION_PREREQUISITE",
        "true_a2_study_contribution": 0,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "aggregate_only": True,
    }
    for key, expected_value in fixed_common.items():
        _expect_exact(report_value[key], expected_value, label=f"published report {key}")
    _expect_bool(
        report_value["independent_raw_reproduction_established"],
        label="published independent reproduction",
    )
    blockers = report_value["blockers"]
    if type(blockers) is not list or len(blockers) != len(set(blockers)) or any(
        type(blocker) is not str for blocker in blockers
    ):
        raise PublicationError("published blockers are not a unique string list")
    allowed_blockers = set(GATE_BLOCKERS) | {
        f"{slot_id}_EVIDENCE_UNKNOWN_NOT_ASSERTED" for slot_id in SLOT_IDS
    }
    if not set(blockers).issubset(allowed_blockers):
        raise PublicationError("published report contains an unregistered blocker")

    if report_value["status"] == BLOCKED_STATUS:
        expected_blocked = {
            "qualified": False,
            "data_role": "A1_ORDINARY_REPORTED_ENDPOINT_CANDIDATE_NOT_QUALIFIED",
            "ordinary_study_contribution": 0,
            "a1_study_contribution": 0,
            "canonical_record_count": 0,
            "canonical_materialization_allowed": False,
            "independent_raw_reproduction_established": False,
        }
        for key, expected_value in expected_blocked.items():
            _expect_exact(report_value[key], expected_value, label=f"blocked report {key}")
        if not blockers:
            raise PublicationError("blocked report has no blocker")
    elif report_value["status"] == SUCCESS_STATUS:
        expected_success = {
            "qualified": True,
            "data_role": "A1_ORDINARY_AUTHOR_PUBLISHED_PROCESSED_ENDPOINT_PRIMARY",
            "ordinary_study_contribution": 1,
            "a1_study_contribution": 1,
            "canonical_materialization_allowed": True,
            "blockers": [],
        }
        for key, expected_value in expected_success.items():
            _expect_exact(report_value[key], expected_value, label=f"success report {key}")
        _expect_int(
            report_value["canonical_record_count"],
            label="success canonical record count",
            minimum=1,
        )
    else:
        raise PublicationError("published report status is outside the closed enum")

    _expect_exact(audit_value["all_inputs_aggregate_only"], True, label="audit aggregate-only")
    _expect_exact(audit_value["row_level_payload_read_count"], 0, label="audit row reads")
    _expect_exact(audit_value["sequence_read_count"], 0, label="audit sequence reads")
    audit_slots = audit_value["slots"]
    if type(audit_slots) is not list or tuple(slot.get("slot_id") for slot in audit_slots) != SLOT_IDS:
        raise PublicationError("published audit slot IDs/order differ")
    for slot in audit_slots:
        _expect_exact_keys(slot, AUDIT_SLOT_KEYS, label=f"published audit slot {slot.get('slot_id')}")
        for key in {"descriptor_bound", "input_opened", "hash_verified"}:
            _expect_bool(slot[key], label=f"published audit slot {key}")
        if slot["gate_status"] not in {"PASS", *NEGATIVE_EVIDENCE_STATUSES, UNKNOWN}:
            raise PublicationError("published audit gate status is outside the closed enum")
    if audit_value["mode"] == "NO_INPUT_READ_EVIDENCE_BINDING_INCOMPLETE":
        _expect_exact(audit_value["opened_input_count"], 0, label="no-input opened count")
        if any(slot["input_opened"] or slot["hash_verified"] or slot["gate_status"] != UNKNOWN for slot in audit_slots):
            raise PublicationError("no-input audit claims an opened/verified input")
        if config is not None:
            _expect_exact(
                blockers,
                [slot["blocker_if_unbound"] for slot in config["evidence_contract"]["slots"]],
                label="no-input blocker truth",
            )
    elif audit_value["mode"] == "ALL_HASH_BOUND_AGGREGATES_VERIFIED":
        _expect_exact(audit_value["opened_input_count"], len(SLOT_IDS), label="verified opened count")
        if any(not slot["descriptor_bound"] or not slot["input_opened"] or not slot["hash_verified"] for slot in audit_slots):
            raise PublicationError("verified audit has an unbound/unopened/unverified slot")
        if report_value["status"] == SUCCESS_STATUS and any(
            slot["gate_status"] != "PASS" for slot in audit_slots
        ):
            raise PublicationError("success report has a non-PASS gate status")
    else:
        raise PublicationError("published audit mode is outside the closed enum")


def inspect_committed_bundle(
    output_directory: Path | str,
    *,
    production: bool = False,
    config: Mapping[str, Any] | None = None,
    expected_authority_provenance: Mapping[str, Any] | None = None,
    _expected_parent_identity: tuple[int, int] | None = None,
    _expected_output_identity: tuple[int, int] | None = None,
) -> dict[str, Any]:
    if production and config is None:
        config = load_production_config()
        expected_authority_provenance = validate_production_authority(config)
    elif production and expected_authority_provenance is None:
        if config is None:  # Defensive; the first branch already covers this.
            raise PublicationError("production inspection config is absent")
        expected_authority_provenance = validate_production_authority(config)
    elif config is not None and expected_authority_provenance is None:
        expected_authority_provenance = _synthetic_authority_provenance(config)
    if config is not None:
        validate_implementation_binding(config)
    lexical = Path(output_directory)
    _preflight_output(
        lexical,
        production=production,
        require_output_exists=True,
    )
    output = Path(os.path.abspath(os.fspath(lexical)))
    parent_fd = _open_directory_root_to_leaf(output.parent, label="output parent")
    parent_identity = _directory_identity(parent_fd)
    if (
        _expected_parent_identity is not None
        and parent_identity != _expected_parent_identity
    ):
        os.close(parent_fd)
        raise PublicationError("output parent identity changed before inspection")
    output_fd = -1
    output_identity: tuple[int, int]
    try:
        output_fd = _open_child_directory(
            parent_fd,
            output.name,
            label="output directory",
        )
        output_identity = _directory_identity(output_fd)
        if (
            _expected_output_identity is not None
            and output_identity != _expected_output_identity
        ):
            raise PublicationError("output directory identity changed before inspection")
        _assert_active_output_identity(
            output.parent,
            parent_fd,
            parent_identity,
            output.name,
            output_fd,
        )
        names = set(os.listdir(output_fd))
        expected = set(OUTPUT_NAMES_EXCLUDING_MARKER) | {COMMIT_MARKER}
        if names != expected:
            raise PartialPublicationError("output member closure is incomplete or differs")
        payloads: dict[str, bytes] = {}
        for name in sorted(expected):
            _assert_active_output_identity(
                output.parent,
                parent_fd,
                parent_identity,
                output.name,
                output_fd,
            )
            payloads[name] = _read_regular_at(output_fd, name)
            _assert_active_output_identity(
                output.parent,
                parent_fd,
                parent_identity,
                output.name,
                output_fd,
            )
    finally:
        if output_fd >= 0:
            os.close(output_fd)
        os.close(parent_fd)
    marker = strict_json(payloads[COMMIT_MARKER], label="publication commit marker")
    _expect_exact_keys(marker, MARKER_KEYS, label="publication commit marker")
    for key, expected_value in {
        "schema_version": "1.0.0",
        "record_type": "ROUTE_A_V3_GSE200304_DEC019_REPORTED_ENDPOINT_A1_ADJUDICATION_COMMIT_V3",
        "contract_id": CONTRACT_ID,
        "decision_id": DECISION_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "output_id": config["output_contract"]["output_id"]
        if config is not None
        else "ROUTE_A_V3_GSE200304_DEC019_REPORTED_ENDPOINT_A1_ADJUDICATION_BUNDLE_V3",
        "publication_mode": PUBLICATION_MODE,
        "bundle_member_names_excluding_commit_marker": sorted(OUTPUT_NAMES_EXCLUDING_MARKER),
        "bundle_file_count_excluding_commit_marker": len(OUTPUT_NAMES_EXCLUDING_MARKER),
        "final_output_target_sha256": _final_target_sha256(output),
        "committed": True,
        "commit_marker_written_last": True,
        "aggregate_acceptance_requires_exact_marker": True,
    }.items():
        _expect_exact(marker[key], expected_value, label=f"publication marker {key}")
    sums = payloads["SHA256SUMS"]
    _expect_exact(marker["sha256sums_sha256"], sha256(sums), label="publication checksum-file binding")
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
    audit = strict_json(payloads["INPUT_EVIDENCE_AUDIT.json"], label="published input audit")
    _validate_published_report_and_audit(
        report,
        audit,
        config=config,
        expected_authority_provenance=expected_authority_provenance,
    )
    _expect_exact(marker["scientific_status"], report["status"], label="marker/report status")
    if config is not None:
        if expected_authority_provenance is None:
            raise PublicationError("inspection authority provenance is absent")
        expected_report, expected_audit = _recompute_adjudication_outputs(
            config,
            expected_authority_provenance,
        )
        expected_payloads = _build_bundle(
            config,
            output,
            expected_report,
            expected_audit,
        )
        if payloads != expected_payloads:
            raise PublicationError(
                "published bundle differs from current recomputed adjudication"
            )
    _assert_reopened_output_identity(output, parent_identity, output_identity)
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
    production: bool,
    config: Mapping[str, Any],
    authority_provenance: Mapping[str, Any],
    fault_injector: FaultInjector | None = None,
) -> str:
    _preflight_output(output, production=production, require_output_exists=False)
    parent_fd = _open_directory_root_to_leaf(output.parent, label="output parent")
    parent_identity = _directory_identity(parent_fd)
    output_fd = -1
    output_identity: tuple[int, int] | None = None
    created = False
    marker_written = False
    try:
        _assert_canonical_directory_identity(
            output.parent,
            parent_identity,
            label="output parent",
        )
        try:
            os.mkdir(output.name, 0o750, dir_fd=parent_fd)
            created = True
        except FileExistsError:
            created = False
        output_fd = _open_child_directory(
            parent_fd,
            output.name,
            label="output directory",
        )
        output_identity = _directory_identity(output_fd)
        _assert_active_output_identity(
            output.parent,
            parent_fd,
            parent_identity,
            output.name,
            output_fd,
        )
        if not created:
            names = set(os.listdir(output_fd))
            expected_names = set(payloads)
            if COMMIT_MARKER not in names:
                raise PartialPublicationError(
                    "existing output lacks the terminal commit marker"
                )
            if names != expected_names:
                raise PublicationError("existing committed output member set differs")
            observed: dict[str, bytes] = {}
            for name in sorted(names):
                _assert_active_output_identity(
                    output.parent,
                    parent_fd,
                    parent_identity,
                    output.name,
                    output_fd,
                )
                observed[name] = _read_regular_at(output_fd, name)
                _assert_active_output_identity(
                    output.parent,
                    parent_fd,
                    parent_identity,
                    output.name,
                    output_fd,
                )
            if observed != dict(payloads):
                raise PublicationError("existing committed output differs; overwrite refused")
            return "EXISTING_EXACT"

        for name in sorted(OUTPUT_NAMES_EXCLUDING_MARKER):
            _assert_active_output_identity(
                output.parent,
                parent_fd,
                parent_identity,
                output.name,
                output_fd,
            )
            _write_exclusive_at(output_fd, name, payloads[name])
            if fault_injector is not None:
                fault_injector(f"after_{name}")
            _assert_active_output_identity(
                output.parent,
                parent_fd,
                parent_identity,
                output.name,
                output_fd,
            )
        os.fsync(output_fd)
        if fault_injector is not None:
            fault_injector("before_commit_marker")
        _assert_active_output_identity(
            output.parent,
            parent_fd,
            parent_identity,
            output.name,
            output_fd,
        )
        _write_exclusive_at(output_fd, COMMIT_MARKER, payloads[COMMIT_MARKER])
        marker_written = True
        os.fsync(output_fd)
        _assert_active_output_identity(
            output.parent,
            parent_fd,
            parent_identity,
            output.name,
            output_fd,
        )
        os.fsync(parent_fd)
        if output_identity is None:
            raise PublicationError("published output identity is absent")
        inspect_committed_bundle(
            output,
            production=production,
            config=config,
            expected_authority_provenance=authority_provenance,
            _expected_parent_identity=parent_identity,
            _expected_output_identity=output_identity,
        )
        return "PUBLISHED"
    except Exception:
        # Preserve partial members, but retract the terminal truth marker when
        # post-marker path identity or exact recomputation fails.
        if marker_written and output_fd >= 0:
            try:
                os.unlink(COMMIT_MARKER, dir_fd=output_fd)
                os.fsync(output_fd)
                os.fsync(parent_fd)
            except OSError as cleanup_error:
                raise PublicationError(
                    "failed to retract commit marker after publication rejection"
                ) from cleanup_error
        raise
    finally:
        if output_fd >= 0:
            os.close(output_fd)
        os.close(parent_fd)


def _preflight_output(
    output: Path,
    *,
    production: bool,
    require_output_exists: bool,
) -> None:
    if not output.is_absolute() or any(part in {"", ".", ".."} for part in output.parts[1:]):
        raise ScopeViolation("output directory must be an absolute path with safe components")
    lowered = os.fspath(output).casefold()
    severe_tokens = ("gse246381", "/restricted/", "/sealed/", "/sealed_external/", "access_log")
    hits = [token for token in severe_tokens if token in lowered]
    if hits:
        raise ScopeViolation(f"output directory contains forbidden path token(s): {','.join(hits)}")
    normalized = Path(os.path.abspath(os.fspath(output)))
    if production:
        if normalized.parent != TRUSTED_A1_OUTPUT_ROOT:
            raise ScopeViolation(
                "production output must be a direct child of the trusted A1 run root"
            )
        _assert_directory_chain_nofollow(
            TRUSTED_A1_OUTPUT_ROOT,
            label="trusted A1 output root",
        )
    target = normalized if require_output_exists else normalized.parent
    _assert_directory_chain_nofollow(
        target,
        label="output directory" if require_output_exists else "output parent",
    )


def _recompute_adjudication_outputs(
    config: Mapping[str, Any],
    authority_provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Re-run evidence validation and derive report/audit without publishing."""

    if not _all_evidence_descriptors_bound(config):
        blockers = [
            slot["blocker_if_unbound"]
            for slot in config["evidence_contract"]["slots"]
        ]
        return (
            _blocked_report(config, blockers, authority_provenance),
            _input_audit(config, None, authority_provenance),
        )

    descriptor_by_slot = _descriptor_by_slot(config)
    merged_slots: list[dict[str, Any]] = []
    for slot in config["evidence_contract"]["slots"]:
        merged = {**slot, **descriptor_by_slot[slot["slot_id"]]}
        merged_slots.append(merged)
        path = Path(merged["absolute_path"])
        _reject_path(path, config, label=f"evidence {slot['slot_id']}")
        if path.name != slot["allowed_basename"]:
            raise ScopeViolation(f"evidence basename differs: {slot['slot_id']}")

    records: dict[str, dict[str, Any]] = {}
    for slot in merged_slots:
        records[slot["slot_id"]] = _validate_gate_record(
            _read_verified_evidence(slot, config),
            slot,
            config,
        )
    blockers, canonical_count, independent_reproduction = _evaluate(records, config)
    report = (
        _blocked_report(config, blockers, authority_provenance)
        if blockers
        else _success_report(
            config,
            canonical_count,
            independent_reproduction,
            authority_provenance,
        )
    )
    return report, _input_audit(config, records, authority_provenance)


def adjudicate(
    config: Mapping[str, Any],
    output_directory: Path | str,
    *,
    production: bool = False,
    fault_injector: FaultInjector | None = None,
) -> dict[str, Any]:
    validate_implementation_binding(config)
    authority_provenance = (
        validate_production_authority(config)
        if production
        else _synthetic_authority_provenance(config)
    )
    lexical_output = Path(output_directory)
    _preflight_output(
        lexical_output,
        production=production,
        require_output_exists=False,
    )
    output = Path(os.path.abspath(os.fspath(lexical_output)))

    report, audit = _recompute_adjudication_outputs(
        config,
        authority_provenance,
    )

    payloads = _build_bundle(config, output, report, audit)
    publication_status = _publish_bundle(
        output,
        payloads,
        production=production,
        config=config,
        authority_provenance=authority_provenance,
        fault_injector=fault_injector,
    )
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
                production=True,
            )
        else:
            if args.output_directory is None:
                parser.error("adjudication requires --output-directory")
            result = adjudicate(load_production_config(), args.output_directory, production=True)
    except AdjudicationError as exc:
        print(json.dumps({"status": "ERROR", "error_type": type(exc).__name__, "message": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
