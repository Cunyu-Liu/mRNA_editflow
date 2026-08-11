#!/usr/bin/env python3
"""Fail-closed DEC-019 successor lifecycle adjudicator for GSE114002.

Version 3 preserves the historical DEC-019 chain while allowing a clean,
pushed descendant HEAD.  A fresh implementation commit is followed by one
config-only binding commit; later config-only descriptor commits may bind
aggregate evidence without changing the frozen scientific core.  Evidence is
opened only when every descriptor is bound.  Negative records carry explicit
unknown fields and predecessor provenance instead of using numeric zero as an
unknown value.
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


CONFIG_REPO_PATH = "configs/route_a_v3_gse114002_dec019_true_a2_activation_v3.json"
SCRIPT_REPO_PATH = "scripts/route_a_v3/adjudicate_gse114002_dec019_true_a2_v3.py"
TEST_REPO_PATH = "tests/route_a_v3/test_adjudicate_gse114002_dec019_true_a2_v3.py"
GSE114002_CONFIG_REPO_PATH = CONFIG_REPO_PATH
GSE200304_CONFIG_REPO_PATH = (
    "configs/route_a_v3_gse200304_dec019_reported_endpoint_a1_activation_v3.json"
)
BINDING_CONFIG_REPO_PATHS = (
    GSE114002_CONFIG_REPO_PATH,
    GSE200304_CONFIG_REPO_PATH,
)
EXPECTED_IMPLEMENTATION_FILES = {
    GSE114002_CONFIG_REPO_PATH: (
        SCRIPT_REPO_PATH,
        TEST_REPO_PATH,
    ),
    GSE200304_CONFIG_REPO_PATH: (
        "scripts/route_a_v3/adjudicate_gse200304_dec019_reported_endpoint_a1_v3.py",
        "tests/route_a_v3/test_adjudicate_gse200304_dec019_reported_endpoint_a1_v3.py",
    ),
}
FROZEN_CONFIG_CORE_SHA256_BY_PATH = {
    GSE114002_CONFIG_REPO_PATH: "6a2955a9c76edbff45aa79c8c71cf3262cbfc631472b345845b5a612a909d67d",
    GSE200304_CONFIG_REPO_PATH: "f4bfde594ce2aa4dbf7d6a9f0cd1607ac1b214a4659089a59258ba0039bb2ff9",
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
    "evidence_descriptor_bindings",
    "evidence_contract",
    "output_contract",
}
PRODUCTION_REPO_ROOT = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/routea_v3_a1_20260810"
)
PRODUCTION_CONFIG_PATH = PRODUCTION_REPO_ROOT / CONFIG_REPO_PATH
TRUSTED_A1_ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1")
UNKNOWN = "UNKNOWN_NOT_ASSERTED"
CONTRACT_ID = "mrna_xeditflow_route_a_v3"
DATASET_ID = "GSE114002"
DECISION_ID = "V3-DEC-019"
PROTOCOL_ID = "ROUTE_A_V3_GSE114002_DEC019_TRUE_A2_ACTIVATION_V3"
EVIDENCE_SCHEMA_VERSION = "route_a_v3_dec019_aggregate_gate_evidence.v2"
EVIDENCE_RECORD_TYPE = "ROUTE_A_V3_DEC019_ACCEPTED_AGGREGATE_GATE_EVIDENCE"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DNA_LIKE = re.compile(r"^[ACGTUNacgtun]{20,}$")
OUTPUT_JSON_NAMES = ("ADJUDICATION_REPORT.json", "INPUT_EVIDENCE_AUDIT.json")
OUTPUT_NAMES_EXCLUDING_MARKER = (*OUTPUT_JSON_NAMES, "SHA256SUMS")
COMMIT_MARKER = "PUBLICATION_COMMIT.json"
PUBLICATION_MODE = "ATOMIC_EXCLUSIVE_DIRECTORY_TERMINAL_COMMIT_MARKER_V1"
BLOCKED_STATUS = "BLOCKED_DEC019_TRUE_A2_EVIDENCE_INCOMPLETE"
SUCCESS_STATUS = "PASS_DEC019_TRUE_A2_ACTIVATED_WITHIN_ASSAY_DEVELOPMENT_ONLY"

SLOT_IDS = (
    "MECHANICAL_ENDPOINT_GEOMETRY",
    "SOURCE_FIELD_AUTHORITY",
    "CONSTRUCT_RNA_CHEMISTRY",
    "CHECKPOINT_SPECIFIC_EXPOSURE",
    "LICENSE_RIGHTS",
    "OUTCOME_BLIND_SPLIT_LEAKAGE",
    "PREFROZEN_POWER_PRECISION",
)
FUTURE_SLOT_IDS = SLOT_IDS[1:]
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
ACCEPTANCE_AUTHORITY_KEYS = {
    "contract_id",
    "decision_id",
    "protocol_id",
    "rule",
}
NEGATIVE_STATUSES = {UNKNOWN, "NOT_RUN", "BLOCKED"}
PRIVACY_KEYS = {
    "contains_row_level_payload",
    "contains_sequence",
    "contains_row_identifier",
    "contains_raw_label_or_effect",
    "contains_member_identifiers_or_hashes",
}
LEGACY_GEOMETRY_KEYS = {
    "contract_id",
    "protocol_id",
    "dataset_id",
    "status",
    "qualified",
    "data_role",
    "scientific_claim_status",
    "ordinary_study_contribution",
    "a1_intervention_study_contribution",
    "true_a2_dense_study_contribution",
    "canonical_record_count",
    "canonical_materialization_allowed",
    "training_allowed",
    "model_selection_allowed",
    "next_phase_authorized",
    "true_a2_claim_established",
    "aggregate_only",
    "blockers",
    "protocol_provenance",
    "source_provenance",
    "implementation_binding",
}
FACT_KEYS = {
    "SOURCE_FIELD_AUTHORITY": {
        "field_dictionary_closed",
        "mother_join_semantics_closed",
        "source_snapshot_hash_bound",
        "row_crosswalk_hash_bound",
        "complete_design_family_manifest_closed",
        "unsafe_ambiguous_fields_excluded_from_join",
        "source_anchored_pool_count",
        "canonical_record_count",
        "minimum_distinct_edited_candidates_per_source",
        "k5_dense_pool_count",
        "k5_used_as_qualification_gate",
    },
    "CONSTRUCT_RNA_CHEMISTRY": {
        "full_25nt_prefix_authority_closed",
        "reporter_identity_authority_closed",
        "designed_sample_rna_chemistry_closed",
        "assay_context_id_frozen",
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
        "biological_replicate_status",
        "paper_standard_error_status",
        "technical_uncertainty_used_as_biological_standard_error",
        "technical_fraction_uncertainty_role",
        "technical_fraction_uncertainty_used_for_observed_power",
        "technical_fraction_uncertainty_used_for_full_confidence_interval",
        "technical_fraction_uncertainty_used_for_equivalence",
        "technical_fraction_uncertainty_used_for_confirmatory_evidence",
        "technical_fraction_uncertainty_used_for_generalization_evidence",
        "uncertainty_basis",
    },
}

GATE_BLOCKERS = {
    "MECHANICAL_ENDPOINT_GEOMETRY_NOT_ACCEPTED",
    "SOURCE_FIELD_AUTHORITY_NOT_PASS",
    "CONSTRUCT_RNA_CHEMISTRY_NOT_PASS",
    "CHECKPOINT_SPECIFIC_EXPOSURE_NOT_PASS",
    "LICENSE_RIGHTS_NOT_PASS",
    "OUTCOME_BLIND_SPLIT_LEAKAGE_NOT_PASS",
    "PREFROZEN_POWER_PRECISION_NOT_PASS",
    "SOURCE_ANCHORED_K3_NEIGHBORHOOD_NOT_ESTABLISHED",
    "K5_USED_AS_QUALIFICATION_GATE",
    "POWER_LT_0_80",
    "FULL_CI_WIDTH_GT_0_30",
    "TECHNICAL_UNCERTAINTY_MISREPRESENTED_AS_BIOLOGICAL_SE",
    "TECHNICAL_FRACTION_UNCERTAINTY_USED_OUTSIDE_QC",
}


class AdjudicationError(RuntimeError):
    """Evidence, authority, or execution integrity failed."""


class BindingError(AdjudicationError):
    """The implementation or core authority is not fully bound."""


class ScopeViolation(AdjudicationError):
    """A caller-selected or configured path left the ordinary-public scope."""


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
    """Return the stable science projection, excluding lifecycle values only.

    The descriptor *schema* and slot order remain frozen in the projection;
    only locator/digest/size values, their derived status, and their own digest
    are excluded.  Thus a descriptor commit cannot silently change gate facts,
    thresholds, blocker semantics, or claim boundaries.
    """

    projected = copy.deepcopy(dict(config))
    projected.pop("implementation_binding", None)
    descriptors = projected.get("evidence_descriptor_bindings")
    if type(descriptors) is dict:
        projected["evidence_descriptor_bindings"] = {
            "binding_scheme": descriptors.get("binding_scheme"),
            "dynamic_scalar_suffixes": descriptors.get("dynamic_scalar_suffixes"),
            "all_descriptors_required_before_any_input_open": descriptors.get(
                "all_descriptors_required_before_any_input_open"
            ),
            "slots": [
                {"slot_id": slot.get("slot_id")}
                for slot in descriptors.get("slots", [])
                if type(slot) is dict
            ],
        }
    return projected


def config_core_sha256(config: Mapping[str, Any]) -> str:
    return sha256(json_bytes(config_core_projection(config)))


def descriptor_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact mutable descriptor set covered by its own digest."""

    descriptors = config.get("evidence_descriptor_bindings")
    if type(descriptors) is not dict:
        raise AdjudicationError("evidence descriptor bindings are absent")
    projected = copy.deepcopy(descriptors)
    projected.pop("descriptor_set_sha256", None)
    return projected


def descriptor_set_sha256(config: Mapping[str, Any]) -> str:
    return sha256(json_bytes(descriptor_projection(config)))


def _descriptor_state(slot: Mapping[str, Any]) -> str:
    values = (slot.get("absolute_path"), slot.get("sha256"), slot.get("bytes"))
    if values == (UNKNOWN, UNKNOWN, UNKNOWN):
        return "UNBOUND"
    if (
        type(values[0]) is str
        and values[0] != UNKNOWN
        and Path(values[0]).is_absolute()
        and HEX64.fullmatch(str(values[1])) is not None
        and type(values[2]) is int
        and type(values[2]) is not bool
        and values[2] > 0
    ):
        return "BOUND"
    raise AdjudicationError(
        f"descriptor has a mixed or invalid binding state: {slot.get('slot_id')}"
    )


def _descriptor_map(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    descriptors = config["evidence_descriptor_bindings"]["slots"]
    return {slot["slot_id"]: slot for slot in descriptors}


def _descriptor_status(config: Mapping[str, Any]) -> str:
    states = [_descriptor_state(slot) for slot in _descriptor_map(config).values()]
    if all(state == "UNBOUND" for state in states):
        return "UNBOUND"
    if all(state == "BOUND" for state in states):
        return "BOUND"
    return "PARTIALLY_BOUND"


def _allowed_descriptor_diff_paths(config: Mapping[str, Any]) -> set[str]:
    result = {
        "evidence_descriptor_bindings.status",
        "evidence_descriptor_bindings.descriptor_set_sha256",
    }
    for index, _slot in enumerate(config["evidence_descriptor_bindings"]["slots"]):
        for suffix in ("absolute_path", "sha256", "bytes"):
            result.add(f"evidence_descriptor_bindings.slots[{index}].{suffix}")
    return result


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

    for value, label in ((i_config, "parent-I"), (b_config, "repair-B")):
        repository = value.get("repository_authority")
        if type(repository) is not dict:
            raise BindingError(f"{label} repository authority is absent: {config_path}")
        if repository.get("base_commit") != "139c4e8d9749ae93ed90924bb527127cf2bbf553":
            raise BindingError(f"{label} repair base commit differs: {config_path}")
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
        expected_i_paths = sorted(
            list(BINDING_CONFIG_REPO_PATHS)
            + [path for pair in EXPECTED_IMPLEMENTATION_FILES.values() for path in pair]
        )
        if repository.get("implementation_commit_exact_changed_paths") != expected_i_paths:
            raise BindingError(
                f"{label} implementation exact changed-path set differs: {config_path}"
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
    _expect_exact_keys(config, EXPECTED_CONFIG_TOP_KEYS, label="config")
    expected_scalars = {
        "schema_version": "route_a_v3_gse114002_dec019_true_a2_activation.v3",
        "protocol_id": PROTOCOL_ID,
        "contract_id": CONTRACT_ID,
        "phase_id": "A1",
        "dataset_id": DATASET_ID,
        "decision_id": DECISION_ID,
    }
    for key, expected in expected_scalars.items():
        _expect_exact(config[key], expected, label=f"config {key}")

    binding = _expect_exact_keys(
        config["implementation_binding"],
        EXPECTED_IMPLEMENTATION_BINDING_KEYS,
        label="implementation binding",
    )
    _expect_exact(
        binding["binding_scheme"],
        "CONFIG_ONLY_POST_IMPLEMENTATION_BINDING_V1",
        label="binding scheme",
    )
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
            "implementation_commit_exact_changed_paths",
            "binding_commit_exact_changed_paths",
            "historical_dec019_binding",
            "descendant_policy",
        },
        label="repository authority",
    )
    _expect_exact(repository["production_repo_root"], os.fspath(PRODUCTION_REPO_ROOT), label="production repo root")
    _expect_exact(repository["branch"], "routea-v3-a1-20260810", label="branch")
    _expect_exact(repository["base_commit"], "139c4e8d9749ae93ed90924bb527127cf2bbf553", label="repair base commit")
    _expect_exact(repository["implementation_commit_expected_parent"], repository["base_commit"], label="implementation parent")
    _expect_exact(repository["binding_commit_expected_parent"], "IMPLEMENTATION_COMMIT_FROM_BINDING", label="binding parent rule")
    _expect_exact(
        repository["implementation_commit_exact_changed_paths"],
        sorted(
            list(BINDING_CONFIG_REPO_PATHS)
            + [path for pair in EXPECTED_IMPLEMENTATION_FILES.values() for path in pair]
        ),
        label="implementation exact changed paths",
    )
    _expect_exact(
        repository["binding_commit_exact_changed_paths"],
        list(BINDING_CONFIG_REPO_PATHS),
        label="binding exact changed paths",
    )
    historical = _expect_exact_keys(
        repository["historical_dec019_binding"],
        {"base_commit", "implementation_commit", "binding_commit", "frozen_successor_blobs"},
        label="historical DEC-019 binding",
    )
    for key, expected in {
        "base_commit": "ad1c57b9255c3066510b08e7a4cf0bd571006811",
        "implementation_commit": "d54de63605a2df51e91262c99218684a80cb6515",
        "binding_commit": "78827501c7efcef28550b04876c98206d94d4808",
    }.items():
        _expect_exact(historical[key], expected, label=f"historical DEC-019 {key}")
    frozen_blobs = [
        (item.get("path"), item.get("sha256"))
        for item in historical["frozen_successor_blobs"]
        if type(item) is dict and set(item) == {"path", "sha256"}
    ] if type(historical["frozen_successor_blobs"]) is list else []
    expected_frozen_blobs = [
        ("configs/route_a_v3_gse114002_dec019_true_a2_activation_v2.json", "5d659f25c42b9828842948b8c734d083efbf23575b72e5ebf135dda725451017"),
        ("configs/route_a_v3_gse200304_dec019_reported_endpoint_a1_activation_v2.json", "9d017f364274bf05f9172f1e0b36753614f56a27edd608caf8069b6c162d6422"),
        ("scripts/route_a_v3/adjudicate_gse114002_dec019_true_a2.py", "20b1d6e7824921d31ea6a0ab5ecac93707ae3acf2789b11019574813e33c1b6c"),
        ("scripts/route_a_v3/adjudicate_gse200304_dec019_reported_endpoint_a1.py", "7ad39104d6fc908a23c538932a4ad9249e24131c383ce088f7010657d1c8191b"),
        ("tests/route_a_v3/test_adjudicate_gse114002_dec019_true_a2.py", "a1ae6e73ee4f9a44f5de6db9ff09be175ecd1648fc592ff2c9f073a7184302ee"),
        ("tests/route_a_v3/test_adjudicate_gse200304_dec019_reported_endpoint_a1.py", "a153c54c9b7fff228d4702e52e2cc9521afcbe14bfb5658fbc1b8fe96ae74c09"),
    ]
    if frozen_blobs != expected_frozen_blobs:
        raise AdjudicationError("historical DEC-019 frozen successor blobs differ")
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
        "single_gsm_allowed": True,
        "biological_replicate_status": "ABSENT_BY_DESIGN",
        "paper_standard_error_status": "ABSENT",
        "technical_uncertainty_role": "DIAGNOSTIC_ONLY_NEVER_BIOLOGICAL_STANDARD_ERROR",
        "technical_uncertainty_prohibited_uses": [
            "BIOLOGICAL_STANDARD_ERROR",
            "POWER",
            "CONFIDENCE_INTERVAL",
            "EQUIVALENCE",
            "CONFIRMATORY_EVIDENCE",
            "GENERALIZATION_EVIDENCE",
        ],
        "success_role": "TRUE_A2_WITHIN_ASSAY_DEVELOPMENT_AND_OPTIMIZATION_ONLY",
        "study_counting_unit": "DATASET",
        "maximum_study_contribution_per_dataset": 1,
        "gsm_pool_subseries_modality_endpoint_replicate_may_multiply_study_count": False,
        "ordinary_gate_contribution_on_success": 1,
        "a1_gate_contribution_on_success": 0,
        "true_a2_gate_contribution_on_success": 1,
        "confirmatory_contribution_on_success": 0,
        "generalization_contribution_on_success": 0,
        "k5_role": "CLAIM_BOUNDARY_ONLY_NOT_QUALIFICATION_GATE",
        "eligible_edit_distances": [1, 2, 3],
        "primary_reporting_k_values": [1, 3],
        "global_reporting_k_values": [1, 3, 5],
        "minimum_source_anchored_candidate_neighborhood_k": 3,
        "power_analysis_unit": "BIOLOGICAL_SOURCE_GROUP",
        "power_bootstrap_unit": "BIOLOGICAL_SOURCE_GROUP",
        "minimum_power": 0.8,
        "maximum_full_confidence_interval_width": 0.3,
        "raw_or_row_level_payload_consumption_allowed": False,
        "training_allowed_by_this_adjudicator": False,
        "model_selection_allowed_by_this_adjudicator": False,
        "next_phase_allowed_by_this_adjudicator": False,
    }
    if policy != fixed_policy:
        raise AdjudicationError("DEC-019 GSE114002 policy boundary differs")

    state = _expect_exact_keys(
        config["current_external_state"],
        {
            "status",
            "qualified",
            "ordinary_study_contribution",
            "a1_study_contribution",
            "true_a2_study_contribution",
            "confirmatory_contribution",
            "generalization_contribution",
            "canonical_record_count",
            "canonical_materialization_allowed",
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
        "confirmatory_contribution": 0,
        "generalization_contribution": 0,
        "canonical_record_count": 0,
        "canonical_materialization_allowed": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }.items():
        _expect_exact(state[key], expected, label=f"current state {key}")
    if state["unresolved_blockers"] != [
        slot["blocker_if_unbound"]
        for slot in config["evidence_contract"]["slots"]
        if slot["slot_id"] in FUTURE_SLOT_IDS
    ]:
        raise AdjudicationError("current six-blocker order differs")

    descriptors = _expect_exact_keys(
        config["evidence_descriptor_bindings"],
        {
            "binding_scheme",
            "status",
            "descriptor_set_sha256",
            "dynamic_scalar_suffixes",
            "all_descriptors_required_before_any_input_open",
            "slots",
        },
        label="evidence descriptor bindings",
    )
    _expect_exact(
        descriptors["binding_scheme"],
        "DESCENDANT_CONFIG_ONLY_DESCRIPTOR_BINDING_V1",
        label="descriptor binding scheme",
    )
    _expect_exact(
        descriptors["dynamic_scalar_suffixes"],
        ["absolute_path", "sha256", "bytes"],
        label="descriptor dynamic suffixes",
    )
    _expect_exact(
        descriptors["all_descriptors_required_before_any_input_open"],
        True,
        label="descriptor all-before-read rule",
    )
    descriptor_slots = descriptors["slots"]
    if type(descriptor_slots) is not list or tuple(
        slot.get("slot_id") for slot in descriptor_slots if type(slot) is dict
    ) != SLOT_IDS:
        raise AdjudicationError("descriptor slot IDs or order differ")
    for slot in descriptor_slots:
        _expect_exact_keys(
            slot,
            {"slot_id", "absolute_path", "sha256", "bytes"},
            label=f"descriptor slot {slot.get('slot_id')}",
        )
        _descriptor_state(slot)
    _expect_exact(
        descriptors["status"],
        _descriptor_status(config),
        label="derived descriptor status",
    )
    if HEX64.fullmatch(str(descriptors["descriptor_set_sha256"])) is None:
        raise AdjudicationError("descriptor-set SHA is not bound")
    _expect_exact(
        descriptors["descriptor_set_sha256"],
        descriptor_set_sha256(config),
        label="descriptor-set SHA",
    )

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
            "evidence_schema_version",
            "negative_record_policy",
            "gate_record_provenance_contract",
            "required_predecessor_authority",
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
                "blocker_if_unbound",
                "blocker_if_not_pass",
            },
            label=f"evidence slot {slot.get('slot_id')}",
        )
    geometry = slots[0]
    _expect_exact(geometry["allowed_basename"], "QUALIFICATION_REPORT.json", label="legacy geometry basename")
    _expect_exact(
        evidence_contract["evidence_schema_version"],
        EVIDENCE_SCHEMA_VERSION,
        label="evidence schema version",
    )
    _expect_exact(
        evidence_contract["negative_record_policy"],
        {
            "allowed_statuses": [UNKNOWN, "NOT_RUN", "BLOCKED"],
            "facts_must_be_null": True,
            "unknown_fields_must_equal_required_fact_keys": True,
            "reason_codes_must_be_nonempty_sorted_unique": True,
            "unknown_numeric_must_not_be_encoded_as_zero": True,
        },
        label="negative record policy",
    )
    _expect_exact(
        evidence_contract["gate_record_provenance_contract"],
        {
            "required": True,
            "producer_protocol_id_required": True,
            "producer_commit_required": True,
            "producer_script_sha256_required": True,
            "source_bundle_id_must_equal_required_predecessor": True,
            "source_bundle_root_or_target_sha256_required": True,
            "predecessor_authority_must_equal_required_predecessor": True,
            "acceptance_authority": {
                "contract_id": CONTRACT_ID,
                "decision_id": DECISION_ID,
                "protocol_id": PROTOCOL_ID,
                "rule": "CONFIG_HASH_BOUND_ACCEPTED_AGGREGATE_GATE_RECORD_V2",
            },
        },
        label="gate record provenance contract",
    )
    _expect_exact(
        evidence_contract["required_predecessor_authority"],
        {
            "authority_type": "GSE114002_PUBLIC_GAP_EVT040_DEC019_AUTHORITY",
            "bundle_id": "ROUTE_A_V3_GSE114002_PUBLIC_GAP_EVT040_AUTHORITY_V1",
            "public_gap_audit_path": "docs/execution/gse114002_public_authority_gap_audit_v1.json",
            "public_gap_audit_sha256": "3be184767bd297f2b50deff2b056e30e2229b970e9bbf0a9c3e5656e3147821f",
            "public_gap_source_runtime_sync_status": "PENDING_NO_EVT_040",
            "predecessor_runtime_event_id": "A1-EVT-040",
            "evt040_runtime_sync_artifact_basename": "A1_DEC019_AUTHORITY_RUNTIME_SYNC_V1.json",
            "evt040_runtime_sync_artifact_sha256": "d7cea8c92464a7a362d017856ff9afb53bc7311467b3ccbd82a4a3e90524aae3",
            "evt040_live_runtime_sync_status": "SYNCED_EVT_040",
            "dec019_authority_binding_commit": "78827501c7efcef28550b04876c98206d94d4808",
        },
        label="required predecessor authority",
    )

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
    _expect_exact(output["output_id"], "ROUTE_A_V3_GSE114002_DEC019_TRUE_A2_ADJUDICATION_BUNDLE_V3", label="output ID")
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
    return _descriptor_status(config) == "BOUND"


def _slot_with_descriptor(
    slot: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(slot)
    result.update(_descriptor_map(config)[slot["slot_id"]])
    return result


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
    path = repo_root / relative_path
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise BindingError(f"authority leaf is unavailable: {relative_path}") from exc
    if sha256(payload) != expected_sha:
        raise BindingError(f"authority leaf SHA differs: {relative_path}")


def _changed_paths(repo: Path, commit: str) -> list[str]:
    return sorted(
        line
        for line in _git(
            repo, "diff-tree", "--no-commit-id", "--name-only", "-r", commit
        ).splitlines()
        if line
    )


def _require_ancestor(repo: Path, ancestor: str, descendant: str, *, label: str) -> None:
    try:
        subprocess.run(
            ["/usr/bin/git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=repo,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        raise BindingError(f"{label} is not an ancestor of current HEAD") from exc


def _require_single_parent(
    repo: Path, commit: str, expected_parent: str, *, label: str
) -> None:
    lineage = _git(repo, "rev-list", "--parents", "-n", "1", commit).split()
    if lineage != [commit, expected_parent]:
        raise BindingError(f"{label} is not the required single-parent commit")


def _validate_descriptor_only_config_pair(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    config_path: str,
) -> None:
    """Validate one post-repair config transition without dataset policy reuse."""

    if type(before) is not dict or set(before) != EXPECTED_CONFIG_TOP_KEYS:
        raise BindingError(f"descriptor parent config schema differs: {config_path}")
    if type(after) is not dict or set(after) != EXPECTED_CONFIG_TOP_KEYS:
        raise BindingError(f"descriptor child config schema differs: {config_path}")
    frozen_core = FROZEN_CONFIG_CORE_SHA256_BY_PATH[config_path]
    for value, label in ((before, "descriptor parent"), (after, "descriptor child")):
        binding = value.get("implementation_binding")
        if type(binding) is not dict or binding.get("config_core_sha256") != frozen_core:
            raise BindingError(f"{label} stored science core differs: {config_path}")
        if config_core_sha256(value) != frozen_core:
            raise BindingError(f"{label} computed science core differs: {config_path}")
        descriptors = value.get("evidence_descriptor_bindings")
        if type(descriptors) is not dict:
            raise BindingError(f"{label} descriptor block is absent: {config_path}")
        if descriptors.get("status") != _descriptor_status(value):
            raise BindingError(f"{label} descriptor status differs: {config_path}")
        if descriptors.get("descriptor_set_sha256") != descriptor_set_sha256(value):
            raise BindingError(f"{label} descriptor digest differs: {config_path}")
    differences = _semantic_diff_paths(before, after)
    if not differences or not differences.issubset(_allowed_descriptor_diff_paths(before)):
        raise BindingError(
            f"post-repair config change is not descriptor-only: {config_path}: "
            f"{sorted(differences)!r}"
        )


def _validate_post_binding_descriptor_history(
    repo: Path, repair_b: str, head: str
) -> None:
    commits = [
        value
        for value in _git(
            repo,
            "rev-list",
            "--ancestry-path",
            "--reverse",
            f"{repair_b}..{head}",
        ).splitlines()
        if value
    ]
    config_paths = set(BINDING_CONFIG_REPO_PATHS)
    for commit in commits:
        lineage = _git(repo, "rev-list", "--parents", "-n", "1", commit).split()
        if len(lineage) != 2 or lineage[0] != commit:
            raise BindingError(f"post-B history contains a merge: {commit}")
        changed = set(_changed_paths(repo, commit))
        relevant = changed & config_paths
        if not relevant:
            continue
        if not changed.issubset(config_paths):
            raise BindingError(
                f"descriptor commit also changed a non-config path: {commit}"
            )
        parent = lineage[1]
        for config_path in sorted(relevant):
            before = strict_json(
                _git_bytes(repo, "show", f"{parent}:{config_path}"),
                label=f"descriptor parent config {config_path}",
            )
            after = strict_json(
                _git_bytes(repo, "show", f"{commit}:{config_path}"),
                label=f"descriptor child config {config_path}",
            )
            _validate_descriptor_only_config_pair(
                before, after, config_path=config_path
            )


def validate_production_authority(config: Mapping[str, Any]) -> dict[str, Any]:
    """Prove historical DEC-019, repair I/B, and descriptor-only descendants.

    No evidence locator and no output target is opened by this function.
    """

    validate_static_config(config)
    repository = config["repository_authority"]
    repo = Path(repository["production_repo_root"])
    if repo != PRODUCTION_REPO_ROOT:
        raise BindingError("production repository root differs")
    if config["core_authority"]["status"] != "BOUND":
        raise BindingError("core authority is UNKNOWN_NOT_ASSERTED")

    binding = config["implementation_binding"]
    head = _git(repo, "rev-parse", "HEAD")
    branch = repository["branch"]
    if _git(repo, "rev-parse", "--abbrev-ref", "HEAD") != branch:
        raise BindingError("production branch differs")
    if _git(repo, "status", "--porcelain"):
        raise BindingError("production worktree is not clean")
    if _git(repo, "rev-parse", f"refs/remotes/origin/{branch}") != head:
        raise BindingError("origin tracking ref is not current HEAD")

    historical = repository["historical_dec019_binding"]
    old_base = historical["base_commit"]
    old_i = historical["implementation_commit"]
    old_b = historical["binding_commit"]
    _require_single_parent(repo, old_i, old_base, label="historical DEC-019 I2")
    _require_single_parent(repo, old_b, old_i, label="historical DEC-019 B2")
    _require_ancestor(repo, old_b, head, label="historical DEC-019 binding")

    for item in historical["frozen_successor_blobs"]:
        path, expected = item["path"], item["sha256"]
        historical_payload = _git_bytes(repo, "show", f"{old_b}:{path}")
        current_payload = _git_bytes(repo, "show", f"{head}:{path}")
        if sha256(historical_payload) != expected or current_payload != historical_payload:
            raise BindingError(f"historical successor blob drifted: {path}")
        if _git(repo, "log", "--format=%H", f"{old_b}..{head}", "--", path):
            raise BindingError(f"historical successor path was touched after B2: {path}")

    repair_base = repository["base_commit"]
    _require_ancestor(repo, old_b, repair_base, label="repair base")
    lifecycle_state: str
    if binding["status"] == UNKNOWN:
        implementation = head
        repair_b = UNKNOWN
        lifecycle_state = "REPAIR_I_IMPLEMENTATION_UNBOUND"
        _require_single_parent(repo, implementation, repair_base, label="repair I")
        if _changed_paths(repo, implementation) != repository[
            "implementation_commit_exact_changed_paths"
        ]:
            raise BindingError("repair implementation is not the exact six-file I commit")
    else:
        implementation = binding["implementation_commit"]
        lifecycle_state = "REPAIR_B_BOUND_OR_DESCRIPTOR_DESCENDANT"
        _require_single_parent(repo, implementation, repair_base, label="repair I")
        if _changed_paths(repo, implementation) != repository[
            "implementation_commit_exact_changed_paths"
        ]:
            raise BindingError("repair implementation is not the exact six-file I commit")
        _require_ancestor(repo, implementation, head, label="repair implementation")

        successors = [
            value
            for value in _git(
                repo,
                "rev-list",
                "--ancestry-path",
                "--reverse",
                f"{implementation}..{head}",
            ).splitlines()
            if value
        ]
        if not successors:
            raise BindingError("repair config-only binding commit is absent")
        repair_b = successors[0]
        _require_single_parent(repo, repair_b, implementation, label="repair B")
        if _changed_paths(repo, repair_b) != list(BINDING_CONFIG_REPO_PATHS):
            raise BindingError("repair binding is not the exact two-config-only B commit")

        b_configs: dict[str, dict[str, Any]] = {}
        for config_path in BINDING_CONFIG_REPO_PATHS:
            i_config = strict_json(
                _git_bytes(repo, "show", f"{implementation}:{config_path}"),
                label=f"repair-I config {config_path}",
            )
            b_config = strict_json(
                _git_bytes(repo, "show", f"{repair_b}:{config_path}"),
                label=f"repair-B config {config_path}",
            )
            _validate_i_to_b_config_pair(
                i_config,
                b_config,
                config_path=config_path,
                implementation_commit=implementation,
            )
            b_configs[config_path] = b_config

        _validate_post_binding_descriptor_history(repo, repair_b, head)
        for config_path in BINDING_CONFIG_REPO_PATHS:
            current = strict_json(
                _git_bytes(repo, "show", f"{head}:{config_path}"),
                label=f"current config {config_path}",
            )
            if config_core_sha256(current) != FROZEN_CONFIG_CORE_SHA256_BY_PATH[
                config_path
            ]:
                raise BindingError(f"current science core drifted: {config_path}")
            b_binding = b_configs[config_path]["implementation_binding"]
            if current.get("implementation_binding") != b_binding:
                raise BindingError(f"current implementation binding drifted: {config_path}")
            script_path, test_path = EXPECTED_IMPLEMENTATION_FILES[config_path]
            for path, expected_sha in (
                (script_path, b_binding["implementation_script_sha256"]),
                (test_path, b_binding["implementation_test_sha256"]),
            ):
                i_payload = _git_bytes(repo, "show", f"{implementation}:{path}")
                current_file = _git_bytes(repo, "show", f"{head}:{path}")
                if sha256(i_payload) != expected_sha or current_file != i_payload:
                    raise BindingError(f"repair implementation file drifted: {path}")
                if _git(
                    repo,
                    "log",
                    "--format=%H",
                    f"{implementation}..{head}",
                    "--",
                    path,
                ):
                    raise BindingError(
                        f"repair implementation path was touched after I: {path}"
                    )
                _verify_repo_file(repo, path, expected_sha)

    current_payload = _git_bytes(repo, "show", f"{head}:{CONFIG_REPO_PATH}")
    try:
        working_payload = (repo / CONFIG_REPO_PATH).read_bytes()
    except OSError as exc:
        raise BindingError("working GSE114002 config is unavailable") from exc
    if current_payload != working_payload or strict_json(
        current_payload, label="current GSE114002 config"
    ) != dict(config):
        raise BindingError("in-memory/working config differs from current HEAD")

    for relative_path, expected_sha in _core_leaf_pairs(config):
        _verify_repo_file(repo, relative_path, expected_sha)

    return {
        "validation_mode": "PRODUCTION_GIT_AUTHORITY",
        "lifecycle_state": lifecycle_state,
        "historical_dec019_base_commit": old_base,
        "historical_dec019_implementation_commit": old_i,
        "historical_dec019_binding_commit": old_b,
        "repair_base_commit": repair_base,
        "repair_implementation_commit": implementation,
        "repair_binding_commit": repair_b,
        "current_head_commit": head,
        "current_head_is_clean_pushed_descendant": True,
        "science_core_sha256": binding["config_core_sha256"],
        "evidence_descriptor_set_sha256": config["evidence_descriptor_bindings"][
            "descriptor_set_sha256"
        ],
        "predecessor_authority_sha256": sha256(
            json_bytes(config["evidence_contract"]["required_predecessor_authority"])
        ),
    }


def _reject_path(path: Path, config: Mapping[str, Any], *, label: str) -> None:
    text = os.fspath(path)
    lowered = text.casefold()
    hits = [
        token
        for token in config["evidence_contract"]["forbidden_path_tokens"]
        if token.casefold() in lowered
    ]
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


def _open_directory_root_to_leaf(path: Path, *, label: str) -> int:
    """Open an existing absolute directory with O_NOFOLLOW at every level."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise AdjudicationError("O_NOFOLLOW is unavailable")
    parts = path.parts
    if len(parts) < 1 or parts[0] != os.sep:
        raise ScopeViolation(f"{label} is not absolute")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | nofollow
    )
    descriptor = os.open(os.sep, flags)
    try:
        for component in parts[1:]:
            try:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                raise AdjudicationError(
                    f"{label} contains a symlink, missing component, or non-directory"
                ) from exc
            os.close(descriptor)
            descriptor = next_descriptor
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


def _directory_identity(directory_fd: int, *, label: str) -> tuple[int, int]:
    opened = os.fstat(directory_fd)
    if not stat.S_ISDIR(opened.st_mode):
        raise PublicationError(f"{label} is not an opened directory")
    return opened.st_dev, opened.st_ino


def _assert_canonical_directory_identity(
    path: Path,
    expected_identity: tuple[int, int],
    *,
    label: str,
) -> None:
    """Re-open a canonical absolute path and compare it with its pinned inode."""

    try:
        reopened_fd = _open_directory_root_to_leaf(path, label=f"canonical {label}")
    except AdjudicationError as exc:
        raise PublicationError(f"{label} canonical identity changed") from exc
    try:
        if _directory_identity(reopened_fd, label=label) != expected_identity:
            raise PublicationError(f"{label} canonical identity changed")
    finally:
        os.close(reopened_fd)


def _assert_output_anchor(
    output: Path,
    parent_fd: int,
    parent_identity: tuple[int, int],
    directory_fd: int,
    directory_identity: tuple[int, int],
    *,
    label: str,
) -> None:
    """Prove both retained descriptors are still named at the canonical path."""

    _assert_canonical_directory_identity(
        output.parent,
        parent_identity,
        label="output parent",
    )
    _assert_named_directory_identity(
        parent_fd,
        output.name,
        directory_fd,
        label=label,
    )
    _assert_canonical_directory_identity(
        output,
        directory_identity,
        label=label,
    )


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


def _validate_common_gate_record(
    payload: bytes,
    slot: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
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
    if record["status"] not in {"PASS", *NEGATIVE_STATUSES}:
        raise AdjudicationError(f"{label} status is outside the closed enum")
    _validate_privacy(record, config, label=label)
    _validate_record_provenance(record["provenance"], config, label=label)
    if record["status"] == "PASS":
        facts = _expect_exact_keys(
            record["facts"], FACT_KEYS[slot["slot_id"]], label=f"{label} facts"
        )
        _validate_fact_types(slot["slot_id"], facts)
        _expect_exact(record["unknown_fields"], [], label=f"{label} unknown fields")
        _expect_exact(record["reason_codes"], [], label=f"{label} reason codes")
    else:
        _expect_exact(record["facts"], None, label=f"{label} negative facts")
        expected_unknown = sorted(FACT_KEYS[slot["slot_id"]])
        _expect_exact(
            record["unknown_fields"],
            expected_unknown,
            label=f"{label} negative unknown fields",
        )
        reasons = record["reason_codes"]
        if (
            type(reasons) is not list
            or not reasons
            or reasons != sorted(set(reasons))
            or any(
                type(reason) is not str
                or re.fullmatch(r"[A-Z][A-Z0-9_]{2,127}", reason) is None
                for reason in reasons
            )
        ):
            raise AdjudicationError(f"{label} negative reason codes differ")
    return record


def _validate_record_provenance(
    value: Any, config: Mapping[str, Any], *, label: str
) -> None:
    provenance = _expect_exact_keys(value, PROVENANCE_KEYS, label=f"{label} provenance")
    if (
        type(provenance["producer_protocol_id"]) is not str
        or re.fullmatch(r"[A-Z0-9][A-Z0-9_.:-]{7,255}", provenance["producer_protocol_id"])
        is None
    ):
        raise AdjudicationError(f"{label} producer protocol is not closed")
    if HEX40.fullmatch(str(provenance["producer_commit"])) is None:
        raise AdjudicationError(f"{label} producer commit is not bound")
    if HEX64.fullmatch(str(provenance["producer_script_sha256"])) is None:
        raise AdjudicationError(f"{label} producer script SHA is not bound")
    required = config["evidence_contract"]["required_predecessor_authority"]
    _expect_exact(
        provenance["source_bundle_id"],
        required["bundle_id"],
        label=f"{label} source bundle ID",
    )
    _expect_exact(
        provenance["source_bundle_root_or_target_sha256"],
        required["evt040_runtime_sync_artifact_sha256"],
        label=f"{label} source bundle target SHA",
    )
    _expect_exact(
        provenance["predecessor_authority"],
        required,
        label=f"{label} predecessor authority",
    )
    _expect_exact_keys(
        provenance["acceptance_authority"],
        ACCEPTANCE_AUTHORITY_KEYS,
        label=f"{label} acceptance authority",
    )
    _expect_exact(
        provenance["acceptance_authority"],
        config["evidence_contract"]["gate_record_provenance_contract"][
            "acceptance_authority"
        ],
        label=f"{label} acceptance authority",
    )


def _validate_fact_types(slot_id: str, facts: Mapping[str, Any]) -> None:
    boolean_keys = set(facts)
    if slot_id == "SOURCE_FIELD_AUTHORITY":
        for key in {
            "source_anchored_pool_count",
            "canonical_record_count",
            "minimum_distinct_edited_candidates_per_source",
            "k5_dense_pool_count",
        }:
            _expect_int(facts[key], label=f"{slot_id} {key}", minimum=0)
            boolean_keys.remove(key)
    elif slot_id == "CHECKPOINT_SPECIFIC_EXPOSURE":
        _expect_int(facts["audited_checkpoint_count"], label=f"{slot_id} audited_checkpoint_count", minimum=0)
        boolean_keys.remove("audited_checkpoint_count")
    elif slot_id == "LICENSE_RIGHTS":
        if facts["redistribution_scope"] not in {"PRIVATE_CANONICAL_ONLY", "PUBLIC_REDISTRIBUTION_ALLOWED"}:
            raise AdjudicationError("LICENSE_RIGHTS redistribution_scope is outside the closed enum")
        boolean_keys.remove("redistribution_scope")
    elif slot_id == "PREFROZEN_POWER_PRECISION":
        observed_power = _expect_number(facts["observed_power"], label="observed power")
        full_ci_width = _expect_number(
            facts["full_confidence_interval_width"], label="full CI width"
        )
        if not 0.0 <= observed_power <= 1.0:
            raise AdjudicationError("observed power must be in [0, 1]")
        if full_ci_width < 0.0:
            raise AdjudicationError("full confidence interval width must be >= 0")
        for key, expected in {
            "analysis_unit": "BIOLOGICAL_SOURCE_GROUP",
            "bootstrap_unit": "BIOLOGICAL_SOURCE_GROUP",
            "biological_replicate_status": "ABSENT_BY_DESIGN",
            "paper_standard_error_status": "ABSENT",
            "technical_fraction_uncertainty_role": "QC_ONLY_WITHIN_ASSAY_DIAGNOSTIC",
            "uncertainty_basis": "BIOLOGICAL_SOURCE_GROUP_RESAMPLING_WITHOUT_TECHNICAL_FRACTION_UNCERTAINTY",
        }.items():
            if type(facts[key]) is not str:
                raise AdjudicationError(f"{slot_id} {key} must be a string")
            boolean_keys.remove(key)
        boolean_keys.remove("observed_power")
        boolean_keys.remove("full_confidence_interval_width")
    for key in boolean_keys:
        _expect_bool(facts[key], label=f"{slot_id} {key}")


def _validate_legacy_geometry(payload: bytes, config: Mapping[str, Any]) -> dict[str, Any]:
    record = strict_json(payload, label="legacy mechanical endpoint geometry")
    _expect_exact_keys(record, LEGACY_GEOMETRY_KEYS, label="legacy mechanical endpoint geometry")
    for key, expected in {
        "contract_id": CONTRACT_ID,
        "protocol_id": "ROUTE_A_V3_GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2",
        "dataset_id": DATASET_ID,
        "status": "MECHANICAL_ENDPOINT_RECONCILED_NOT_QUALIFIED",
        "qualified": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "ordinary_study_contribution": 0,
        "a1_intervention_study_contribution": 0,
        "true_a2_dense_study_contribution": 0,
        "canonical_record_count": 0,
        "canonical_materialization_allowed": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "true_a2_claim_established": False,
        "aggregate_only": True,
    }.items():
        _expect_exact(record[key], expected, label=f"legacy geometry {key}")
    if type(record["blockers"]) is not list or record["blockers"] != sorted(set(record["blockers"])):
        raise AdjudicationError("legacy geometry blocker list is not closed/deduplicated")
    forbidden = {key.casefold() for key in config["output_contract"]["forbidden_output_keys"]}
    _assert_no_private_material(record, forbidden, label="legacy geometry")
    return record


def _slot_gate_pass(slot_id: str, facts: Mapping[str, Any]) -> bool:
    if slot_id == "SOURCE_FIELD_AUTHORITY":
        return all(
            facts[key] is True
            for key in {
                "field_dictionary_closed",
                "mother_join_semantics_closed",
                "source_snapshot_hash_bound",
                "row_crosswalk_hash_bound",
                "complete_design_family_manifest_closed",
                "unsafe_ambiguous_fields_excluded_from_join",
            }
        ) and facts["source_anchored_pool_count"] >= 1 and facts["canonical_record_count"] >= 1
    if slot_id == "CONSTRUCT_RNA_CHEMISTRY":
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
            facts["analysis_unit"] == "BIOLOGICAL_SOURCE_GROUP"
            and facts["bootstrap_unit"] == "BIOLOGICAL_SOURCE_GROUP"
            and facts["prefrozen_before_model_results"] is True
            and facts["biological_replicate_status"] == "ABSENT_BY_DESIGN"
            and facts["paper_standard_error_status"] == "ABSENT"
            and facts["technical_uncertainty_used_as_biological_standard_error"] is False
            and facts["technical_fraction_uncertainty_role"]
            == "QC_ONLY_WITHIN_ASSAY_DIAGNOSTIC"
            and all(
                facts[key] is False
                for key in {
                    "technical_fraction_uncertainty_used_for_observed_power",
                    "technical_fraction_uncertainty_used_for_full_confidence_interval",
                    "technical_fraction_uncertainty_used_for_equivalence",
                    "technical_fraction_uncertainty_used_for_confirmatory_evidence",
                    "technical_fraction_uncertainty_used_for_generalization_evidence",
                }
            )
            and facts["uncertainty_basis"]
            == "BIOLOGICAL_SOURCE_GROUP_RESAMPLING_WITHOUT_TECHNICAL_FRACTION_UNCERTAINTY"
        )
    raise AdjudicationError(f"unknown gate slot: {slot_id}")


def _evaluate(records: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]) -> tuple[list[str], int]:
    blockers: list[str] = []
    geometry = records["MECHANICAL_ENDPOINT_GEOMETRY"]
    if geometry["status"] != "MECHANICAL_ENDPOINT_RECONCILED_NOT_QUALIFIED":
        blockers.append("MECHANICAL_ENDPOINT_GEOMETRY_NOT_ACCEPTED")
    for slot in config["evidence_contract"]["slots"][1:]:
        record = records[slot["slot_id"]]
        if record["status"] != "PASS":
            blockers.append(slot["blocker_if_not_pass"])
            continue
        if not _slot_gate_pass(slot["slot_id"], record["facts"]):
            blockers.append(slot["blocker_if_not_pass"])

    source_record = records["SOURCE_FIELD_AUTHORITY"]
    source = source_record["facts"]
    if source_record["status"] == "PASS":
        if source["minimum_distinct_edited_candidates_per_source"] < 3:
            blockers.append("SOURCE_ANCHORED_K3_NEIGHBORHOOD_NOT_ESTABLISHED")
        if source["k5_used_as_qualification_gate"] is not False:
            blockers.append("K5_USED_AS_QUALIFICATION_GATE")
    power_record = records["PREFROZEN_POWER_PRECISION"]
    power = power_record["facts"]
    if power_record["status"] == "PASS":
        if float(power["observed_power"]) < 0.8:
            blockers.append("POWER_LT_0_80")
        if float(power["full_confidence_interval_width"]) > 0.3:
            blockers.append("FULL_CI_WIDTH_GT_0_30")
        if power["technical_uncertainty_used_as_biological_standard_error"] is not False:
            blockers.append("TECHNICAL_UNCERTAINTY_MISREPRESENTED_AS_BIOLOGICAL_SE")
        if any(
            power[key] is not False
            for key in {
                "technical_fraction_uncertainty_used_for_observed_power",
                "technical_fraction_uncertainty_used_for_full_confidence_interval",
                "technical_fraction_uncertainty_used_for_equivalence",
                "technical_fraction_uncertainty_used_for_confirmatory_evidence",
                "technical_fraction_uncertainty_used_for_generalization_evidence",
            }
        ):
            blockers.append("TECHNICAL_FRACTION_UNCERTAINTY_USED_OUTSIDE_QC")
    blockers = sorted(set(blockers))
    if not set(blockers).issubset(GATE_BLOCKERS):
        raise AdjudicationError("an unregistered blocker was produced")
    canonical_count = int(source["canonical_record_count"]) if not blockers else 0
    return blockers, canonical_count


def _nonproduction_authority(config: Mapping[str, Any]) -> dict[str, Any]:
    historical = config["repository_authority"]["historical_dec019_binding"]
    return {
        "validation_mode": "SYNTHETIC_NONPRODUCTION",
        "lifecycle_state": "SYNTHETIC_BOUND_CONFIG",
        "historical_dec019_base_commit": historical["base_commit"],
        "historical_dec019_implementation_commit": historical["implementation_commit"],
        "historical_dec019_binding_commit": historical["binding_commit"],
        "repair_base_commit": config["repository_authority"]["base_commit"],
        "repair_implementation_commit": config["implementation_binding"][
            "implementation_commit"
        ],
        "repair_binding_commit": UNKNOWN,
        "current_head_commit": UNKNOWN,
        "current_head_is_clean_pushed_descendant": False,
        "science_core_sha256": config["implementation_binding"]["config_core_sha256"],
        "evidence_descriptor_set_sha256": config["evidence_descriptor_bindings"][
            "descriptor_set_sha256"
        ],
        "predecessor_authority_sha256": sha256(
            json_bytes(config["evidence_contract"]["required_predecessor_authority"])
        ),
    }


def _blocked_report(
    config: Mapping[str, Any],
    blockers: Sequence[str],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "record_type": "ROUTE_A_V3_GSE114002_DEC019_TRUE_A2_ADJUDICATION_REPORT_V3",
        "contract_id": CONTRACT_ID,
        "decision_id": DECISION_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "status": BLOCKED_STATUS,
        "qualified": False,
        "data_role": "TRUE_A2_RECOVERY_CANDIDATE_NOT_QUALIFIED",
        "ordinary_study_contribution": 0,
        "a1_study_contribution": 0,
        "true_a2_study_contribution": 0,
        "confirmatory_contribution": 0,
        "generalization_contribution": 0,
        "canonical_record_count": 0,
        "canonical_materialization_allowed": False,
        "within_assay_development_and_optimization_only": True,
        "technical_uncertainty_is_biological_standard_error": False,
        "k5_is_qualification_gate": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "aggregate_only": True,
        "blockers": list(blockers),
        "config_core_sha256": config["implementation_binding"]["config_core_sha256"],
        "evidence_descriptor_set_sha256": config["evidence_descriptor_bindings"][
            "descriptor_set_sha256"
        ],
        "authority_provenance": dict(authority),
    }


def _success_report(
    config: Mapping[str, Any],
    canonical_count: int,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "record_type": "ROUTE_A_V3_GSE114002_DEC019_TRUE_A2_ADJUDICATION_REPORT_V3",
        "contract_id": CONTRACT_ID,
        "decision_id": DECISION_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "status": SUCCESS_STATUS,
        "qualified": True,
        "data_role": "TRUE_A2_WITHIN_ASSAY_DEVELOPMENT_AND_OPTIMIZATION_ONLY",
        "ordinary_study_contribution": 1,
        "a1_study_contribution": 0,
        "true_a2_study_contribution": 1,
        "confirmatory_contribution": 0,
        "generalization_contribution": 0,
        "canonical_record_count": canonical_count,
        "canonical_materialization_allowed": True,
        "within_assay_development_and_optimization_only": True,
        "technical_uncertainty_is_biological_standard_error": False,
        "k5_is_qualification_gate": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "aggregate_only": True,
        "blockers": [],
        "config_core_sha256": config["implementation_binding"]["config_core_sha256"],
        "evidence_descriptor_set_sha256": config["evidence_descriptor_bindings"][
            "descriptor_set_sha256"
        ],
        "authority_provenance": dict(authority),
    }


def _input_audit(
    config: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]] | None,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    descriptor_map = _descriptor_map(config)
    slots = []
    for slot in config["evidence_contract"]["slots"]:
        bound = _descriptor_state(descriptor_map[slot["slot_id"]]) == "BOUND"
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
        "record_type": "ROUTE_A_V3_DEC019_AGGREGATE_INPUT_EVIDENCE_AUDIT_V3",
        "contract_id": CONTRACT_ID,
        "decision_id": DECISION_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "mode": "ALL_HASH_BOUND_AGGREGATES_VERIFIED" if records is not None else "NO_INPUT_READ_EVIDENCE_BINDING_INCOMPLETE",
        "all_inputs_aggregate_only": True,
        "row_level_payload_read_count": 0,
        "sequence_read_count": 0,
        "opened_input_count": len(records) if records is not None else 0,
        "evidence_descriptor_set_sha256": config["evidence_descriptor_bindings"][
            "descriptor_set_sha256"
        ],
        "authority_provenance": dict(authority),
        "slots": slots,
    }


def _derive_report_and_audit(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute scientific truth from the current descriptor/evidence state."""

    if not _all_evidence_descriptors_bound(config):
        descriptors = _descriptor_map(config)
        blockers = [
            slot["blocker_if_unbound"]
            for slot in config["evidence_contract"]["slots"]
            if _descriptor_state(descriptors[slot["slot_id"]]) != "BOUND"
        ]
        return (
            _blocked_report(config, blockers, authority),
            _input_audit(config, None, authority),
        )

    # No evidence file is opened until every descriptor is bound and every
    # configured path passes lexical scope/basename preflight.
    for slot in config["evidence_contract"]["slots"]:
        bound_slot = _slot_with_descriptor(slot, config)
        path = Path(bound_slot["absolute_path"])
        _reject_path(path, config, label=f"evidence {slot['slot_id']}")
        if path.name != bound_slot["allowed_basename"]:
            raise ScopeViolation(f"evidence basename differs: {slot['slot_id']}")

    records: dict[str, dict[str, Any]] = {}
    for slot in config["evidence_contract"]["slots"]:
        bound_slot = _slot_with_descriptor(slot, config)
        payload = _read_verified_evidence(bound_slot, config)
        if slot["slot_id"] == "MECHANICAL_ENDPOINT_GEOMETRY":
            records[slot["slot_id"]] = _validate_legacy_geometry(payload, config)
        else:
            records[slot["slot_id"]] = _validate_common_gate_record(
                payload,
                slot,
                config,
            )
    blockers, canonical_count = _evaluate(records, config)
    report = (
        _blocked_report(config, blockers, authority)
        if blockers
        else _success_report(config, canonical_count, authority)
    )
    return report, _input_audit(config, records, authority)


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
        "record_type": "ROUTE_A_V3_GSE114002_DEC019_TRUE_A2_ADJUDICATION_COMMIT_V3",
        "contract_id": CONTRACT_ID,
        "decision_id": DECISION_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "output_id": config["output_contract"]["output_id"],
        "scientific_status": report["status"],
        "publication_mode": PUBLICATION_MODE,
        "sha256sums_sha256": sha256(payloads["SHA256SUMS"]),
        "science_core_sha256": config["implementation_binding"]["config_core_sha256"],
        "evidence_descriptor_set_sha256": config["evidence_descriptor_bindings"][
            "descriptor_set_sha256"
        ],
        "authority_provenance_sha256": sha256(
            json_bytes(report["authority_provenance"])
        ),
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
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o640)
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
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, 0o640, dir_fd=directory_fd)
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


def _validate_published_bundle_semantics(
    report: Mapping[str, Any],
    audit: Mapping[str, Any],
    marker: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None,
    expected_authority_provenance: Mapping[str, Any] | None,
) -> None:
    report_keys = {
        "record_type", "contract_id", "decision_id", "protocol_id", "dataset_id",
        "status", "qualified", "data_role", "ordinary_study_contribution",
        "a1_study_contribution", "true_a2_study_contribution",
        "confirmatory_contribution", "generalization_contribution",
        "canonical_record_count", "canonical_materialization_allowed",
        "within_assay_development_and_optimization_only",
        "technical_uncertainty_is_biological_standard_error",
        "k5_is_qualification_gate", "training_allowed", "model_selection_allowed",
        "next_phase_authorized", "scientific_claim_status", "aggregate_only",
        "blockers", "config_core_sha256", "evidence_descriptor_set_sha256",
        "authority_provenance",
    }
    _expect_exact_keys(report, report_keys, label="published report")
    for key, expected in {
        "record_type": "ROUTE_A_V3_GSE114002_DEC019_TRUE_A2_ADJUDICATION_REPORT_V3",
        "contract_id": CONTRACT_ID,
        "decision_id": DECISION_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "within_assay_development_and_optimization_only": True,
        "technical_uncertainty_is_biological_standard_error": False,
        "k5_is_qualification_gate": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "aggregate_only": True,
    }.items():
        _expect_exact(report[key], expected, label=f"published report {key}")
    if HEX64.fullmatch(str(report["config_core_sha256"])) is None:
        raise PublicationError("published report science core is not bound")
    if HEX64.fullmatch(str(report["evidence_descriptor_set_sha256"])) is None:
        raise PublicationError("published report descriptor set is not bound")
    if config is not None:
        _expect_exact(
            report["config_core_sha256"],
            config["implementation_binding"]["config_core_sha256"],
            label="published report science core",
        )

    blockers = report["blockers"]
    if type(blockers) is not list or blockers != list(dict.fromkeys(blockers)):
        raise PublicationError("published blocker list is not closed/unique")
    allowed_blockers = set(GATE_BLOCKERS)
    if config is not None:
        allowed_blockers.update(
            slot["blocker_if_unbound"] for slot in config["evidence_contract"]["slots"]
        )
    else:
        allowed_blockers.update(
            {
                "MECHANICAL_ENDPOINT_GEOMETRY_EVIDENCE_UNKNOWN_NOT_ASSERTED",
                *{
                    f"{slot_id}_EVIDENCE_UNKNOWN_NOT_ASSERTED"
                    for slot_id in FUTURE_SLOT_IDS
                },
            }
        )
    if not set(blockers).issubset(allowed_blockers):
        raise PublicationError("published blocker list contains an unknown blocker")

    if report["status"] == BLOCKED_STATUS:
        expected = {
            "qualified": False,
            "data_role": "TRUE_A2_RECOVERY_CANDIDATE_NOT_QUALIFIED",
            "ordinary_study_contribution": 0,
            "a1_study_contribution": 0,
            "true_a2_study_contribution": 0,
            "confirmatory_contribution": 0,
            "generalization_contribution": 0,
            "canonical_record_count": 0,
            "canonical_materialization_allowed": False,
        }
        if not blockers:
            raise PublicationError("blocked report has no blocker")
    elif report["status"] == SUCCESS_STATUS:
        expected = {
            "qualified": True,
            "data_role": "TRUE_A2_WITHIN_ASSAY_DEVELOPMENT_AND_OPTIMIZATION_ONLY",
            "ordinary_study_contribution": 1,
            "a1_study_contribution": 0,
            "true_a2_study_contribution": 1,
            "confirmatory_contribution": 0,
            "generalization_contribution": 0,
            "canonical_materialization_allowed": True,
        }
        if blockers:
            raise PublicationError("successful report has blockers")
        _expect_int(
            report["canonical_record_count"],
            label="published canonical record count",
            minimum=1,
        )
    else:
        raise PublicationError("published scientific status is outside the closed enum")
    for key, expected_value in expected.items():
        _expect_exact(
            report[key], expected_value, label=f"published report truth {key}"
        )

    authority_keys = {
        "validation_mode", "lifecycle_state", "historical_dec019_base_commit",
        "historical_dec019_implementation_commit", "historical_dec019_binding_commit",
        "repair_base_commit", "repair_implementation_commit", "repair_binding_commit",
        "current_head_commit", "current_head_is_clean_pushed_descendant",
        "science_core_sha256", "evidence_descriptor_set_sha256",
        "predecessor_authority_sha256",
    }
    authority = _expect_exact_keys(
        report["authority_provenance"], authority_keys, label="published authority"
    )
    if expected_authority_provenance is not None:
        _expect_exact(
            authority,
            dict(expected_authority_provenance),
            label="expected published authority provenance",
        )
    for key, expected_value in {
        "historical_dec019_base_commit": "ad1c57b9255c3066510b08e7a4cf0bd571006811",
        "historical_dec019_implementation_commit": "d54de63605a2df51e91262c99218684a80cb6515",
        "historical_dec019_binding_commit": "78827501c7efcef28550b04876c98206d94d4808",
        "repair_base_commit": "139c4e8d9749ae93ed90924bb527127cf2bbf553",
        "science_core_sha256": report["config_core_sha256"],
        "evidence_descriptor_set_sha256": report["evidence_descriptor_set_sha256"],
    }.items():
        _expect_exact(authority[key], expected_value, label=f"published authority {key}")
    if HEX64.fullmatch(str(authority["predecessor_authority_sha256"])) is None:
        raise PublicationError("published predecessor authority SHA is not bound")
    if config is not None:
        _expect_exact(
            authority["predecessor_authority_sha256"],
            sha256(
                json_bytes(
                    config["evidence_contract"]["required_predecessor_authority"]
                )
            ),
            label="published predecessor authority SHA",
        )
    if authority["validation_mode"] == "PRODUCTION_GIT_AUTHORITY":
        if authority["lifecycle_state"] != "REPAIR_B_BOUND_OR_DESCRIPTOR_DESCENDANT":
            raise PublicationError("production bundle lifecycle state differs")
        for key in (
            "repair_implementation_commit",
            "repair_binding_commit",
            "current_head_commit",
        ):
            if HEX40.fullmatch(str(authority[key])) is None:
                raise PublicationError(f"published authority commit is not bound: {key}")
        _expect_exact(
            authority["current_head_is_clean_pushed_descendant"],
            True,
            label="published production descendant truth",
        )
    elif authority["validation_mode"] == "SYNTHETIC_NONPRODUCTION":
        _expect_exact(
            authority["lifecycle_state"],
            "SYNTHETIC_BOUND_CONFIG",
            label="published synthetic lifecycle state",
        )
        if HEX40.fullmatch(str(authority["repair_implementation_commit"])) is None:
            raise PublicationError("synthetic implementation commit is not bound")
        for key in ("repair_binding_commit", "current_head_commit"):
            _expect_exact(authority[key], UNKNOWN, label=f"synthetic authority {key}")
        _expect_exact(
            authority["current_head_is_clean_pushed_descendant"],
            False,
            label="synthetic descendant truth",
        )
    else:
        raise PublicationError("published authority validation mode differs")

    audit_keys = {
        "record_type", "contract_id", "decision_id", "protocol_id", "dataset_id",
        "mode", "all_inputs_aggregate_only", "row_level_payload_read_count",
        "sequence_read_count", "opened_input_count",
        "evidence_descriptor_set_sha256", "authority_provenance", "slots",
    }
    _expect_exact_keys(audit, audit_keys, label="published input audit")
    for key, expected_value in {
        "record_type": "ROUTE_A_V3_DEC019_AGGREGATE_INPUT_EVIDENCE_AUDIT_V3",
        "contract_id": CONTRACT_ID,
        "decision_id": DECISION_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "all_inputs_aggregate_only": True,
        "row_level_payload_read_count": 0,
        "sequence_read_count": 0,
        "evidence_descriptor_set_sha256": report["evidence_descriptor_set_sha256"],
        "authority_provenance": report["authority_provenance"],
    }.items():
        _expect_exact(audit[key], expected_value, label=f"published input audit {key}")
    slots = audit["slots"]
    if type(slots) is not list or [slot.get("slot_id") for slot in slots] != list(SLOT_IDS):
        raise PublicationError("published input audit slot order differs")
    for index, slot in enumerate(slots):
        _expect_exact_keys(
            slot,
            {"slot_id", "descriptor_bound", "input_opened", "hash_verified", "gate_status"},
            label=f"published input audit slot {slot.get('slot_id')}",
        )
        for key in ("descriptor_bound", "input_opened", "hash_verified"):
            _expect_bool(slot[key], label=f"published input audit {slot['slot_id']} {key}")
        allowed_statuses = (
            {UNKNOWN, "MECHANICAL_ENDPOINT_RECONCILED_NOT_QUALIFIED"}
            if index == 0
            else {UNKNOWN, "PASS", *NEGATIVE_STATUSES}
        )
        if slot["gate_status"] not in allowed_statuses:
            raise PublicationError("published input audit gate status differs")
    if audit["mode"] == "NO_INPUT_READ_EVIDENCE_BINDING_INCOMPLETE":
        _expect_exact(audit["opened_input_count"], 0, label="published opened count")
        if any(
            slot["input_opened"]
            or slot["hash_verified"]
            or slot["gate_status"] != UNKNOWN
            for slot in slots
        ):
            raise PublicationError("zero-read audit claims an opened input")
        if config is not None:
            _expect_exact(
                blockers,
                [
                    stable["blocker_if_unbound"]
                    for stable, observed in zip(
                        config["evidence_contract"]["slots"], slots
                    )
                    if not observed["descriptor_bound"]
                ],
                label="zero-read blocker truth",
            )
    elif audit["mode"] == "ALL_HASH_BOUND_AGGREGATES_VERIFIED":
        _expect_exact(
            audit["opened_input_count"], len(SLOT_IDS), label="published opened count"
        )
        if any(
            not slot["descriptor_bound"]
            or not slot["input_opened"]
            or not slot["hash_verified"]
            for slot in slots
        ):
            raise PublicationError("all-verified audit contains an unverified input")
        if report["status"] == SUCCESS_STATUS:
            _expect_exact(
                slots[0]["gate_status"],
                "MECHANICAL_ENDPOINT_RECONCILED_NOT_QUALIFIED",
                label="successful geometry audit status",
            )
            if any(slot["gate_status"] != "PASS" for slot in slots[1:]):
                raise PublicationError("successful report contains a non-PASS science gate")
        elif config is not None:
            for stable, observed in zip(
                config["evidence_contract"]["slots"][1:], slots[1:]
            ):
                if (
                    observed["gate_status"] in NEGATIVE_STATUSES
                    and stable["blocker_if_not_pass"] not in blockers
                ):
                    raise PublicationError(
                        "negative gate status lacks its registered blocker"
                    )
    else:
        raise PublicationError("published input audit mode differs")
    if report["status"] == SUCCESS_STATUS and audit["mode"] != "ALL_HASH_BOUND_AGGREGATES_VERIFIED":
        raise PublicationError("successful report lacks all verified aggregate inputs")

    marker_keys = {
        "schema_version", "record_type", "contract_id", "decision_id", "protocol_id",
        "dataset_id", "output_id", "scientific_status", "publication_mode",
        "sha256sums_sha256", "science_core_sha256",
        "evidence_descriptor_set_sha256", "authority_provenance_sha256",
        "bundle_member_names_excluding_commit_marker",
        "bundle_file_count_excluding_commit_marker", "final_output_target_sha256",
        "committed", "commit_marker_written_last",
        "aggregate_acceptance_requires_exact_marker",
    }
    _expect_exact_keys(marker, marker_keys, label="publication marker")
    for key, expected_value in {
        "schema_version": "1.0.0",
        "record_type": "ROUTE_A_V3_GSE114002_DEC019_TRUE_A2_ADJUDICATION_COMMIT_V3",
        "contract_id": CONTRACT_ID,
        "decision_id": DECISION_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "scientific_status": report["status"],
        "publication_mode": PUBLICATION_MODE,
        "science_core_sha256": report["config_core_sha256"],
        "evidence_descriptor_set_sha256": report["evidence_descriptor_set_sha256"],
        "authority_provenance_sha256": sha256(json_bytes(report["authority_provenance"])),
        "bundle_member_names_excluding_commit_marker": sorted(OUTPUT_NAMES_EXCLUDING_MARKER),
        "bundle_file_count_excluding_commit_marker": len(OUTPUT_NAMES_EXCLUDING_MARKER),
        "committed": True,
        "commit_marker_written_last": True,
        "aggregate_acceptance_requires_exact_marker": True,
    }.items():
        _expect_exact(marker[key], expected_value, label=f"publication marker {key}")
    if config is not None:
        _expect_exact(
            marker["output_id"],
            config["output_contract"]["output_id"],
            label="publication marker output ID",
        )
    else:
        _expect_exact(
            marker["output_id"],
            "ROUTE_A_V3_GSE114002_DEC019_TRUE_A2_ADJUDICATION_BUNDLE_V3",
            label="publication marker output ID",
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def inspect_committed_bundle(
    output_directory: Path | str,
    *,
    production: bool = False,
    config: Mapping[str, Any] | None = None,
    expected_authority_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(os.path.abspath(os.fspath(output_directory)))
    if config is None:
        raise ScopeViolation("inspection requires the current bound config")
    validate_implementation_binding(config)
    _preflight_output(
        output,
        config,
        production=production,
        require_existing_output=True,
    )
    if expected_authority_provenance is None:
        expected_authority_provenance = (
            validate_production_authority(config)
            if production
            else _nonproduction_authority(config)
        )
    parent_fd = _open_directory_root_to_leaf(output.parent, label="output parent")
    parent_identity = _directory_identity(parent_fd, label="output parent")
    directory_fd = -1
    directory_identity: tuple[int, int] | None = None
    expected = set(OUTPUT_NAMES_EXCLUDING_MARKER) | {COMMIT_MARKER}
    try:
        _assert_canonical_directory_identity(
            output.parent,
            parent_identity,
            label="output parent",
        )
        directory_fd = _open_child_directory(
            parent_fd, output.name, label="output directory"
        )
        directory_identity = _directory_identity(
            directory_fd,
            label="output directory",
        )
        _assert_output_anchor(
            output,
            parent_fd,
            parent_identity,
            directory_fd,
            directory_identity,
            label="output directory",
        )
        names = set(os.listdir(directory_fd))
        if names != expected:
            raise PartialPublicationError("output member closure is incomplete or differs")
        payloads: dict[str, bytes] = {}
        for name in sorted(expected):
            _assert_output_anchor(
                output,
                parent_fd,
                parent_identity,
                directory_fd,
                directory_identity,
                label="output directory",
            )
            payloads[name] = _read_regular_at(directory_fd, name)
            _assert_output_anchor(
                output,
                parent_fd,
                parent_identity,
                directory_fd,
                directory_identity,
                label="output directory",
            )
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)
        os.close(parent_fd)
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
    audit = strict_json(
        payloads["INPUT_EVIDENCE_AUDIT.json"], label="published input audit"
    )
    if marker.get("scientific_status") != report.get("status"):
        raise PublicationError("marker/report status differs")
    if marker.get("science_core_sha256") != report.get("config_core_sha256"):
        raise PublicationError("marker/report science-core binding differs")
    if marker.get("evidence_descriptor_set_sha256") != report.get(
        "evidence_descriptor_set_sha256"
    ):
        raise PublicationError("marker/report descriptor binding differs")
    if marker.get("authority_provenance_sha256") != sha256(
        json_bytes(report.get("authority_provenance"))
    ):
        raise PublicationError("marker/report authority provenance binding differs")
    _validate_published_bundle_semantics(
        report,
        audit,
        marker,
        config=config,
        expected_authority_provenance=expected_authority_provenance,
    )
    expected_report, expected_audit = _derive_report_and_audit(
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
            "published bundle differs from evidence-derived expected bundle"
        )
    _assert_canonical_directory_identity(
        output.parent,
        parent_identity,
        label="output parent",
    )
    if directory_identity is None:
        raise PublicationError("output directory identity was not pinned")
    _assert_canonical_directory_identity(
        output,
        directory_identity,
        label="output directory",
    )
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
    config: Mapping[str, Any],
    production: bool,
    fault_injector: FaultInjector | None = None,
) -> str:
    _preflight_output(output, config, production=production)
    expected_authority = strict_json(
        payloads["ADJUDICATION_REPORT.json"], label="publisher report"
    )["authority_provenance"]
    parent = output.parent
    parent_fd = _open_directory_root_to_leaf(parent, label="output parent")
    parent_identity = _directory_identity(parent_fd, label="output parent")
    try:
        _assert_canonical_directory_identity(
            parent,
            parent_identity,
            label="output parent",
        )
        try:
            os.mkdir(output.name, 0o750, dir_fd=parent_fd)
        except FileExistsError:
            _assert_canonical_directory_identity(
                parent,
                parent_identity,
                label="output parent",
            )
            inspect_committed_bundle(
                output,
                production=production,
                config=config,
                expected_authority_provenance=expected_authority,
            )
            _assert_canonical_directory_identity(
                parent,
                parent_identity,
                label="output parent",
            )
            output_fd = _open_child_directory(
                parent_fd,
                output.name,
                label="existing output directory",
            )
            try:
                output_identity = _directory_identity(
                    output_fd,
                    label="existing output directory",
                )
                _assert_output_anchor(
                    output,
                    parent_fd,
                    parent_identity,
                    output_fd,
                    output_identity,
                    label="existing output directory",
                )
                observed: dict[str, bytes] = {}
                for name in sorted(payloads):
                    _assert_output_anchor(
                        output,
                        parent_fd,
                        parent_identity,
                        output_fd,
                        output_identity,
                        label="existing output directory",
                    )
                    observed[name] = _read_regular_at(output_fd, name)
                    _assert_output_anchor(
                        output,
                        parent_fd,
                        parent_identity,
                        output_fd,
                        output_identity,
                        label="existing output directory",
                    )
            finally:
                os.close(output_fd)
            if observed != dict(payloads):
                raise PublicationError(
                    "existing committed output differs; overwrite refused"
                )
            _assert_canonical_directory_identity(
                parent,
                parent_identity,
                label="output parent",
            )
            return "EXISTING_EXACT"

        _assert_canonical_directory_identity(
            parent,
            parent_identity,
            label="output parent",
        )
        output_fd = _open_child_directory(
            parent_fd,
            output.name,
            label="new output directory",
        )
        try:
            output_identity = _directory_identity(
                output_fd,
                label="new output directory",
            )
            _assert_output_anchor(
                output,
                parent_fd,
                parent_identity,
                output_fd,
                output_identity,
                label="new output directory",
            )
            for name in sorted(OUTPUT_NAMES_EXCLUDING_MARKER):
                _assert_output_anchor(
                    output,
                    parent_fd,
                    parent_identity,
                    output_fd,
                    output_identity,
                    label="new output directory",
                )
                _write_exclusive_at(output_fd, name, payloads[name])
                if fault_injector is not None:
                    fault_injector(f"after_{name}")
                _assert_output_anchor(
                    output,
                    parent_fd,
                    parent_identity,
                    output_fd,
                    output_identity,
                    label="new output directory",
                )
            os.fsync(output_fd)
            if fault_injector is not None:
                fault_injector("before_commit_marker")
            _assert_output_anchor(
                output,
                parent_fd,
                parent_identity,
                output_fd,
                output_identity,
                label="new output directory",
            )
            _write_exclusive_at(output_fd, COMMIT_MARKER, payloads[COMMIT_MARKER])
            _assert_output_anchor(
                output,
                parent_fd,
                parent_identity,
                output_fd,
                output_identity,
                label="new output directory",
            )
            os.fsync(output_fd)
            os.fsync(parent_fd)
            _assert_output_anchor(
                output,
                parent_fd,
                parent_identity,
                output_fd,
                output_identity,
                label="new output directory",
            )
        finally:
            os.close(output_fd)
    finally:
        os.close(parent_fd)
    inspect_committed_bundle(
        output,
        production=production,
        config=config,
        expected_authority_provenance=expected_authority,
    )
    return "PUBLISHED"


def _preflight_output(
    output: Path,
    config: Mapping[str, Any],
    *,
    production: bool,
    require_existing_output: bool = False,
) -> None:
    if not output.is_absolute() or any(part in {"", ".", ".."} for part in output.parts[1:]):
        raise ScopeViolation("output directory must be an absolute path with safe components")
    lowered = os.fspath(output).casefold()
    forbidden_tokens = {
        token.casefold()
        for token in config["evidence_contract"]["forbidden_path_tokens"]
    }
    hits = sorted(token for token in forbidden_tokens if token in lowered)
    if hits:
        raise ScopeViolation(f"output directory contains forbidden path token(s): {','.join(hits)}")
    if production:
        trusted_root = Path(os.path.abspath(os.fspath(TRUSTED_A1_ROOT)))
        if output.parent != trusted_root:
            raise ScopeViolation(
                "production output must be a direct child of the trusted A1 root"
            )
        trusted_fd = _open_directory_root_to_leaf(
            trusted_root, label="trusted A1 root"
        )
        os.close(trusted_fd)
    checked_directory = output if require_existing_output else output.parent
    checked_fd = _open_directory_root_to_leaf(
        checked_directory,
        label="output directory" if require_existing_output else "output parent",
    )
    os.close(checked_fd)


def adjudicate(
    config: Mapping[str, Any],
    output_directory: Path | str,
    *,
    production: bool = False,
    fault_injector: FaultInjector | None = None,
) -> dict[str, Any]:
    """Adjudicate bound aggregate evidence and atomically publish one bundle."""

    validate_implementation_binding(config)
    authority = (
        validate_production_authority(config)
        if production
        else _nonproduction_authority(config)
    )
    output = Path(os.path.abspath(os.fspath(output_directory)))
    _preflight_output(output, config, production=production)

    report, audit = _derive_report_and_audit(config, authority)

    payloads = _build_bundle(config, output, report, audit)
    publication_status = _publish_bundle(
        output,
        payloads,
        config=config,
        production=production,
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
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "blockers": report["blockers"],
    }


def load_production_config() -> dict[str, Any]:
    payload = PRODUCTION_CONFIG_PATH.read_bytes()
    return strict_json(payload, label="production config")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--inspect", action="store_true")
    mode.add_argument("--validate-authority", action="store_true")
    args = parser.parse_args(argv)
    if not args.validate_authority and args.output_directory is None:
        parser.error("--output-directory is required unless --validate-authority is used")
    try:
        config = load_production_config()
        if args.validate_authority:
            authority = validate_production_authority(config)
            result = {
                "status": "PRODUCTION_AUTHORITY_VALID",
                "evidence_opened_count": 0,
                "output_access_count": 0,
                "authority_provenance": authority,
            }
        elif args.inspect:
            authority = validate_production_authority(config)
            result = inspect_committed_bundle(
                args.output_directory,
                production=True,
                config=config,
                expected_authority_provenance=authority,
            )
        else:
            result = adjudicate(config, args.output_directory, production=True)
    except AdjudicationError as exc:
        print(json.dumps({"status": "ERROR", "error_type": type(exc).__name__, "message": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
