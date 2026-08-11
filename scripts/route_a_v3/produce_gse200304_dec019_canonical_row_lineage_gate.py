#!/usr/bin/env python3
"""Produce the aggregate DEC-019 GSE200304 row-lineage PASS gate.

The producer is deliberately narrower than the historical endpoint audit.  It
same-FD hashes the exact seven-member ordinary-public source bundle but
selectively materializes only the frozen Table S2/S3 assets, never executes a
formula, and never persists a row key, barcode, sequence, annotation, effect,
p/FDR value, or significance label.  Table S2 is parsed by a byte-level
RFC4180 state machine which captures only ID/Type; all later row fields are
opaque except for quote/newline and non-empty geometry.  Table S3 is read from low-level XLSX
XML and selects only the locator, Comparison, and the numeric-present/exact-NA
state of D--F.  Gene and Translation cells are not decoded by this program.

Production is fail-closed behind a config-only I/B binding.  The final gate is
a single canonical JSON file published with an exclusive hard-link, fsync,
no-overwrite, nlink=1, and exact-idempotence contract.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import posixpath
import re
import secrets
import stat
import struct
import subprocess
import sys
import types
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence
from xml.etree import ElementTree as ET


CONFIG_REPO_PATH = (
    "configs/route_a_v3_gse200304_dec019_canonical_row_lineage_gate_v1.json"
)
SCRIPT_REPO_PATH = (
    "scripts/route_a_v3/produce_gse200304_dec019_canonical_row_lineage_gate.py"
)
TEST_REPO_PATH = (
    "tests/route_a_v3/test_produce_gse200304_dec019_canonical_row_lineage_gate.py"
)
PRODUCTION_REPO_ROOT = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/routea_v3_a1_20260810"
)
PRODUCTION_CONFIG_PATH = PRODUCTION_REPO_ROOT / CONFIG_REPO_PATH
BRANCH = "routea-v3-a1-20260810"
IMPLEMENTATION_BASE_COMMIT = "de35ce44d7744b89c8b52291343d9f1d6ea674a0"
EXPECTED_I_PATHS = [CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH]
EXPECTED_B_PATHS = [CONFIG_REPO_PATH]
AUTHORITY_GIT_MODE = "100644"
MAX_AUTHORITY_FILE_BYTES = 16 * 1024 * 1024

CONTRACT_ID = "mrna_xeditflow_route_a_v3"
DATASET_ID = "GSE200304"
DECISION_ID = "V3-DEC-019"
PHASE_ID = "A1"
PROTOCOL_ID = "ROUTE_A_V3_GSE200304_DEC019_CANONICAL_ROW_LINEAGE_GATE_V1"
SCHEMA_VERSION = "route_a_v3_gse200304_dec019_canonical_row_lineage_gate.v1"
EVIDENCE_SCHEMA_VERSION = "route_a_v3_dec019_aggregate_gate_evidence.v3"
EVIDENCE_RECORD_TYPE = "ROUTE_A_V3_DEC019_ACCEPTED_AGGREGATE_GATE_EVIDENCE_V3"
LOCATOR_LINEAGE_COMMITMENT_ALGORITHM = (
    "ROUTE_A_V3_GSE200304_LOCATOR_MERKLE_V1"
)
GATE_ID = "CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE"
OUTPUT_BASENAME = (
    "GSE200304_DEC019_CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE_GATE.json"
)
UNKNOWN = "UNKNOWN_NOT_ASSERTED"
RAW_REPLAY_ROLE = "REPRODUCIBILITY_AUXILIARY_NOT_QUALIFICATION_PREREQUISITE"
FROZEN_CONFIG_CORE_SHA256 = "5ea2f2dd32f19d4d64a441188c26b80444c0cb89e465d37d348a8b65ae358f2d"

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PAIR_ID_RE = re.compile(r"^[^:]+:[1-9][0-9]*_[ACGT]-[ACGT]$")
CELL_REF_RE = re.compile(r"^([A-Z]+)([1-9][0-9]*)$")
DNA_LIKE = re.compile(r"^[ACGTUNacgtun]{20,}$")

CONFIG_TOP_KEYS = {
    "schema_version",
    "protocol_id",
    "contract_id",
    "phase_id",
    "dataset_id",
    "decision_id",
    "implementation_binding",
    "repository_authority",
    "authority_inputs",
    "source_contract",
    "locator_contract",
    "output_contract",
}
IMPLEMENTATION_BINDING_KEYS = {
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
I_TO_B_SCALAR_PATHS = [
    "implementation_binding.status",
    "implementation_binding.implementation_commit",
    "implementation_binding.implementation_script_sha256",
    "implementation_binding.implementation_test_sha256",
]
GATE_RECORD_KEYS = {
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
PRIVACY_KEYS = {
    "contains_row_level_payload",
    "contains_sequence",
    "contains_row_identifier",
    "contains_raw_label_or_effect",
    "contains_member_identifiers_or_hashes",
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
FACT_KEYS = {
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
    "locator_lineage_commitment_algorithm",
    "locator_lineage_merkle_root_sha256",
}

class GateProducerError(RuntimeError):
    """Base class for all closed producer failures."""


class BindingError(GateProducerError):
    """Implementation or Git authority was not established."""


class ScopeViolation(GateProducerError):
    """An input or output path left the frozen ordinary-public scope."""


class InputIntegrityError(GateProducerError):
    """A source snapshot, hash, header, or closed aggregate differed."""


class TableAuditError(GateProducerError):
    """The selective Table S2/S3 audit failed."""


class PublicationError(GateProducerError):
    """Exclusive, durable, or exact-idempotent publication failed."""


class AmbiguousPublicationError(PublicationError):
    """Existing output was partial, multiply linked, or not exact."""


class PublicationStateUnverifiedError(AmbiguousPublicationError):
    """Canonical output identity was lost after publication may have started."""


FaultInjector = Callable[[str], None]
LocatorDigest = Callable[[bytes, str, Mapping[str, Any]], bytes]


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GateProducerError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                GateProducerError(f"non-finite JSON constant in {label}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateProducerError(f"invalid JSON: {label}") from exc
    if type(value) is not dict:
        raise GateProducerError(f"JSON root is not an object: {label}")
    return value


def _expect_exact_keys(value: Any, keys: set[str], *, label: str) -> Mapping[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise GateProducerError(f"{label} keys differ from the closed schema")
    return value


def _expect_exact(value: Any, expected: Any, *, label: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise GateProducerError(f"{label} differs from the closed value")


def config_core_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    projected = copy.deepcopy(dict(config))
    projected.pop("implementation_binding", None)
    return projected


def config_core_sha256(config: Mapping[str, Any]) -> str:
    return sha256(json_bytes(config_core_projection(config)))


def _semantic_diff_paths(before: Any, after: Any, prefix: str = "") -> set[str]:
    if type(before) is not type(after):
        return {prefix or "<root>"}
    if type(before) is dict:
        paths: set[str] = set()
        for key in set(before) | set(after):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                paths.add(child)
            else:
                paths |= _semantic_diff_paths(before[key], after[key], child)
        return paths
    if type(before) is list:
        if len(before) != len(after):
            return {prefix}
        paths: set[str] = set()
        for index, (left, right) in enumerate(zip(before, after)):
            paths |= _semantic_diff_paths(left, right, f"{prefix}[{index}]")
        return paths
    return set() if before == after else {prefix}


def validate_static_config(config: Mapping[str, Any]) -> None:
    _expect_exact_keys(config, CONFIG_TOP_KEYS, label="producer config")
    for key, expected in {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "contract_id": CONTRACT_ID,
        "phase_id": PHASE_ID,
        "dataset_id": DATASET_ID,
        "decision_id": DECISION_ID,
    }.items():
        _expect_exact(config[key], expected, label=f"config {key}")

    binding = _expect_exact_keys(
        config["implementation_binding"],
        IMPLEMENTATION_BINDING_KEYS,
        label="implementation binding",
    )
    for key, expected in {
        "binding_scheme": "CONFIG_ONLY_POST_IMPLEMENTATION_BINDING_V1",
        "blocker_if_unbound": "IMPLEMENTATION_BINDING_UNKNOWN_NOT_ASSERTED",
        "implementation_script_path": SCRIPT_REPO_PATH,
        "implementation_test_path": TEST_REPO_PATH,
        "unknown_to_bound_scalar_paths": I_TO_B_SCALAR_PATHS,
        "config_core_sha256": FROZEN_CONFIG_CORE_SHA256,
    }.items():
        _expect_exact(binding[key], expected, label=f"implementation binding {key}")
    dynamic = (
        "implementation_commit",
        "implementation_script_sha256",
        "implementation_test_sha256",
    )
    if binding["status"] == UNKNOWN:
        if any(binding[key] != UNKNOWN for key in dynamic):
            raise BindingError("UNKNOWN implementation binding is partially bound")
    elif binding["status"] == "BOUND":
        if HEX40.fullmatch(str(binding["implementation_commit"])) is None:
            raise BindingError("bound implementation commit is invalid")
        if any(
            HEX64.fullmatch(str(binding[key])) is None
            for key in dynamic[1:]
        ):
            raise BindingError("bound implementation blob SHA is invalid")
    else:
        raise BindingError("implementation binding status is outside the closed enum")
    if config_core_sha256(config) != FROZEN_CONFIG_CORE_SHA256:
        raise BindingError("producer config core differs from compiled authority")

    repository = config["repository_authority"]
    if type(repository) is not dict:
        raise BindingError("repository authority is not an object")
    for key, expected in {
        "production_repo_root": os.fspath(PRODUCTION_REPO_ROOT),
        "branch": BRANCH,
        "implementation_base_commit": IMPLEMENTATION_BASE_COMMIT,
        "implementation_commit_expected_parent": IMPLEMENTATION_BASE_COMMIT,
        "implementation_commit_exact_changed_paths": EXPECTED_I_PATHS,
        "binding_commit_expected_parent": "IMPLEMENTATION_COMMIT_FROM_BINDING",
        "binding_commit_exact_changed_paths": EXPECTED_B_PATHS,
        "authority_file_git_mode": "100644",
        "worktree_authority_files_must_be_regular_single_link": True,
        "worktree_authority_files_must_be_root_to_leaf_nofollow": True,
        "current_head_must_be_clean_pushed_descendant_of_binding": True,
        "bound_implementation_blobs_must_not_drift": True,
        "bound_config_must_not_drift_after_binding": True,
    }.items():
        _expect_exact(repository.get(key), expected, label=f"repository authority {key}")

    source = config["source_contract"]
    if type(source) is not dict:
        raise GateProducerError("source contract is not an object")
    if source.get("ordinary_public_only") is not True:
        raise ScopeViolation("source contract is not ordinary-public-only")
    if source.get("readable_asset_ids") != [
        "PMC10540565_TABLE_S2",
        "PMC10540565_TABLE_S3",
    ]:
        raise ScopeViolation("readable source set differs from S2/S3 only")
    if source.get("non_scientific_members_are_streamed_hash_only") is not True:
        raise ScopeViolation("non-scientific source hash-only policy differs")
    members = source.get("members")
    if type(members) is not list or len(members) != 7:
        raise InputIntegrityError("source member contract is not exact-seven")
    relative_names = [member.get("relative_path") for member in members]
    if relative_names != source.get("exact_member_names"):
        raise InputIntegrityError("source member ordering or names differ")
    for member in members:
        if type(member) is not dict or set(member) != {
            "asset_id", "relative_path", "bytes", "sha256", "read_policy"
        }:
            raise InputIntegrityError("source member schema differs")
        if type(member["bytes"]) is not int or member["bytes"] < 0:
            raise InputIntegrityError("source member byte count is invalid")
        if HEX64.fullmatch(str(member["sha256"])) is None:
            raise InputIntegrityError("source member SHA is invalid")
        if member["asset_id"] in source["readable_asset_ids"]:
            if member["read_policy"] != "VERIFIED_SAME_FD_SELECTIVE_PARSE":
                raise ScopeViolation("scientific source read policy differs")
        elif member["read_policy"] != (
            "VERIFIED_SAME_FD_STREAMED_HASH_ONLY_DO_NOT_DECODE"
        ):
            raise ScopeViolation("non-scientific member hash-only policy differs")

    locator = config["locator_contract"]
    if type(locator) is not dict:
        raise GateProducerError("locator contract is not an object")
    for key, expected in {
        "algorithm_id": LOCATOR_LINEAGE_COMMITMENT_ALGORITHM,
        "locator_is_orientation_neutral": True,
        "merkle_rule": {
            "leaf_order": "UNSIGNED_BYTE_LEXICOGRAPHIC",
            "odd_node_rule": "DUPLICATE_LAST",
            "root_binds_original_leaf_count": True,
            "leaf_or_member_digests_may_be_persisted": False,
        },
        "primary_comparison": "TotalPoly:RNA",
        "comparison_allowlist": ["HighPoly:RNA", "TotalPoly:RNA"],
        "membership_rule": "ALL_FINITE_TOTALPOLY_PAIRS_BEFORE_ANY_SIGNIFICANCE_LABEL",
        "significance_or_cached_translation_may_affect_membership": False,
        "processed_pair_count": 6772,
        "canonical_record_count": 6547,
        "primary_na_pair_count": 225,
        "raw_replay_role": RAW_REPLAY_ROLE,
        "raw_replay_status": "NOT_RUN",
        "independent_raw_reproduction_claimed": False,
    }.items():
        _expect_exact(locator.get(key), expected, label=f"locator contract {key}")
    for domain_key in (
        "join_key_domain",
        "locator_domain",
        "s2_physical_row_domain",
        "s2_pair_arm_domain",
        "s3_physical_row_domain",
        "pair_lineage_domain",
        "merkle_leaf_domain",
        "merkle_node_domain",
        "merkle_root_domain",
    ):
        if not isinstance(locator.get(domain_key), str) or not locator[domain_key]:
            raise GateProducerError(f"locator contract {domain_key} is absent")
    domains = [locator[key] for key in locator if key.endswith("_domain")]
    if len(domains) != len(set(domains)):
        raise GateProducerError("locator digest domains are not pairwise distinct")
    if locator["table_s2"].get("deduplicated_pair_count") != 6885:
        raise GateProducerError("frozen S2 pair count differs")
    if locator["table_s3"].get("pair_key_count") != 6772:
        raise GateProducerError("frozen S3 pair count differs")
    if locator["join"] != {
        "table_s2_pair_count": 6885,
        "table_s3_pair_count": 6772,
        "joined_pair_count": 6772,
        "table_s2_absent_from_table_s3_count": 113,
        "table_s3_not_in_table_s2_count": 0,
        "s3_pair_set_must_be_subset_of_s2": True,
        "each_s3_pair_must_join_exactly_one_s2_pair": True,
    }:
        raise GateProducerError("frozen S2/S3 join contract differs")

    output = config["output_contract"]
    if type(output) is not dict:
        raise GateProducerError("output contract is not an object")
    for key, expected in {
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "record_type": EVIDENCE_RECORD_TYPE,
        "gate_id": GATE_ID,
        "allowed_basename": OUTPUT_BASENAME,
        "single_final_file_only": True,
        "publication_mode": "FSYNCED_STAGED_HARDLINK_NO_REPLACE_SINGLE_FILE_V1",
        "exclusive_no_overwrite": True,
        "existing_exact_is_idempotent": True,
        "partial_or_ambiguous_publication_is_never_accepted": True,
        "final_file_nlink_must_equal_one": True,
        "aggregate_only": True,
        "privacy": {key: False for key in sorted(PRIVACY_KEYS)},
        "pass_fact_keys": sorted(FACT_KEYS),
        "raw_or_row_level_payload_may_be_persisted": False,
    }.items():
        _expect_exact(output.get(key), expected, label=f"output contract {key}")


def validate_implementation_binding(config: Mapping[str, Any]) -> None:
    validate_static_config(config)
    binding = config["implementation_binding"]
    if binding["status"] != "BOUND":
        raise BindingError(
            "IMPLEMENTATION_BINDING_UNKNOWN_NOT_ASSERTED; stopped before source/output"
        )


def validate_i_to_b_config_pair(
    i_config: Mapping[str, Any],
    b_config: Mapping[str, Any],
    *,
    implementation_commit: str,
    script_sha256: str,
    test_sha256: str,
) -> None:
    if config_core_sha256(i_config) != FROZEN_CONFIG_CORE_SHA256:
        raise BindingError("I config core differs")
    if config_core_sha256(b_config) != FROZEN_CONFIG_CORE_SHA256:
        raise BindingError("B config core differs")
    i_binding = i_config.get("implementation_binding")
    b_binding = b_config.get("implementation_binding")
    if type(i_binding) is not dict or type(b_binding) is not dict:
        raise BindingError("I/B implementation binding is absent")
    if i_binding.get("status") != UNKNOWN or any(
        i_binding.get(key) != UNKNOWN
        for key in (
            "implementation_commit",
            "implementation_script_sha256",
            "implementation_test_sha256",
        )
    ):
        raise BindingError("I config is not exact UNKNOWN state")
    if b_binding.get("status") != "BOUND":
        raise BindingError("B config is not bound")
    expected_bound = {
        "implementation_commit": implementation_commit,
        "implementation_script_sha256": script_sha256,
        "implementation_test_sha256": test_sha256,
    }
    if any(b_binding.get(key) != value for key, value in expected_bound.items()):
        raise BindingError("B implementation identities differ")
    if _semantic_diff_paths(i_config, b_config) != set(I_TO_B_SCALAR_PATHS):
        raise BindingError("I-to-B diff is not the exact four-scalar allowlist")
    validate_static_config(b_config)


def _git(repo: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["/usr/bin/git", *args],
            cwd=repo,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BindingError(f"git authority check failed: {' '.join(args)}") from exc
    return result.stdout.strip()


def _git_bytes(repo: Path, *args: str) -> bytes:
    try:
        result = subprocess.run(
            ["/usr/bin/git", *args],
            cwd=repo,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BindingError(f"git blob authority check failed: {' '.join(args)}") from exc
    return result.stdout


def _commit_changed_paths(repo: Path, commit: str) -> list[str]:
    return sorted(
        line
        for line in _git(
            repo, "diff-tree", "--no-commit-id", "--name-only", "-r", commit
        ).splitlines()
        if line
    )


def _require_single_parent(repo: Path, commit: str, parent: str, *, label: str) -> None:
    lineage = _git(repo, "rev-list", "--parents", "-n", "1", commit).split()
    if lineage != [commit, parent]:
        raise BindingError(f"{label} is not an exact single-parent commit")


def _require_ancestor(repo: Path, ancestor: str, descendant: str, *, label: str) -> None:
    _git(repo, "merge-base", "--is-ancestor", ancestor, descendant)
    if _git(repo, "rev-parse", ancestor) != ancestor:
        raise BindingError(f"{label} ancestor does not resolve exactly")
    if _git(repo, "rev-parse", descendant) != descendant:
        raise BindingError(f"{label} descendant does not resolve exactly")


def _git_file_mode(repo: Path, commit: str, path: str) -> str:
    raw = _git_bytes(repo, "ls-tree", "-z", commit, "--", path)
    entries = [entry for entry in raw.split(b"\x00") if entry]
    if len(entries) != 1 or b"\t" not in entries[0]:
        raise BindingError(f"Git tree entry is not exact-one: {path}")
    metadata, raw_path = entries[0].split(b"\t", 1)
    fields = metadata.split()
    try:
        decoded_path = raw_path.decode("utf-8")
        mode = fields[0].decode("ascii")
        object_type = fields[1].decode("ascii")
    except (IndexError, UnicodeDecodeError) as exc:
        raise BindingError(f"Git tree entry is malformed: {path}") from exc
    if decoded_path != path or object_type != "blob" or len(fields) != 3:
        raise BindingError(f"Git tree entry identity differs: {path}")
    return mode


def _verify_blob(
    repo: Path,
    commit: str,
    path: str,
    digest: str | None = None,
) -> bytes:
    if _git_file_mode(repo, commit, path) != AUTHORITY_GIT_MODE:
        raise BindingError(f"Git file mode differs from 100644: {path}")
    payload = _git_bytes(repo, "show", f"{commit}:{path}")
    if digest is not None and sha256(payload) != digest:
        raise BindingError(f"Git blob SHA differs: {path}")
    return payload


def _verify_current_file(
    repo: Path,
    head: str,
    path: str,
    digest: str | None = None,
) -> bytes:
    payload = _verify_blob(repo, head, path, digest)
    working = _read_worktree_authority_file(repo, path)
    if working != payload:
        raise BindingError(f"working authority input differs from HEAD: {path}")
    return payload


def _validate_v3_compatibility(
    payload: bytes,
    config: Mapping[str, Any],
) -> None:
    v3 = strict_json(payload, label="DEC019 GSE200304 v3 config")
    binding = v3.get("implementation_binding")
    if type(binding) is not dict or binding.get("config_core_sha256") != config[
        "authority_inputs"
    ]["dec019_gse200304_v3"]["science_core_sha256"]:
        raise BindingError("DEC019 v3 science core differs")
    evidence = v3.get("evidence_contract")
    if type(evidence) is not dict:
        raise BindingError("DEC019 v3 evidence contract is absent")
    if evidence.get("required_predecessor_authority") != config["authority_inputs"][
        "required_predecessor_authority"
    ]:
        raise BindingError("producer predecessor differs from DEC019 v3")
    provenance = evidence.get("gate_record_provenance_contract")
    if type(provenance) is not dict or provenance.get(
        "acceptance_authority"
    ) != config["authority_inputs"]["acceptance_authority"]:
        raise BindingError("producer acceptance authority differs from DEC019 v3")
    if evidence.get("evidence_schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise BindingError("DEC019 v3 evidence schema differs")
    slots = evidence.get("slots")
    lineage_slots = [
        slot for slot in slots if type(slot) is dict and slot.get("slot_id") == GATE_ID
    ] if type(slots) is list else []
    if len(lineage_slots) != 1 or lineage_slots[0].get("allowed_basename") != OUTPUT_BASENAME:
        raise BindingError("DEC019 v3 lineage slot differs")
    output = v3.get("output_contract")
    if type(output) is not dict or set(output.get("forbidden_output_keys", [])) - set(
        config["output_contract"]["forbidden_output_keys"]
    ):
        raise BindingError("producer privacy keys are weaker than DEC019 v3")


def _validate_runtime_lineage_config(
    runtime_config: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    predecessor = config["authority_inputs"]["required_predecessor_authority"]
    lineage = predecessor["runtime_lineage_authority"]
    for key, expected in {
        "event_id": lineage["event_id"],
        "protocol_id": lineage["protocol_id"],
        "contract_id": CONTRACT_ID,
        "phase_id": PHASE_ID,
        "dataset_id": DATASET_ID,
    }.items():
        if runtime_config.get(key) != expected:
            raise BindingError(f"EVT037 bound config {key} differs")
    binding = runtime_config.get("implementation_binding")
    if type(binding) is not dict or binding.get("status") != "BOUND":
        raise BindingError("EVT037 implementation binding is not BOUND")
    for key, expected in {
        "implementation_commit": lineage["implementation_commit"],
        "implementation_script_sha256": lineage["implementation_script_sha256"],
        "implementation_test_sha256": lineage["implementation_test_sha256"],
        "compiled_core_sha256": lineage["compiled_core_sha256"],
    }.items():
        if binding.get(key) != expected:
            raise BindingError(f"EVT037 binding {key} differs")
    runtime = runtime_config.get("runtime")
    if type(runtime) is not dict or runtime.get("artifact_root") != predecessor[
        "trusted_absolute_bundle_path"
    ]:
        raise BindingError("EVT037 artifact root differs from predecessor")
    members = runtime.get("artifact_members")
    if type(members) is not list:
        raise BindingError("EVT037 artifact members are absent")
    projection = [
        {key: member.get(key) for key in ("name", "bytes", "sha256")}
        for member in members
        if type(member) is dict
    ]
    if projection != predecessor["members"]:
        raise BindingError("EVT037 artifact members differ from predecessor")
    truth = runtime_config.get("artifact_truth")
    expected_names = [member["name"] for member in predecessor["members"][:-1]]
    if type(truth) is not dict or any(
        truth.get(key) != expected
        for key, expected in {
            "publication_state": "COMMITTED_ACCEPTED",
            "terminal_record_type": (
                "GSE200304_PUBLISHED_ENDPOINT_A1_PUBLICATION_COMMIT"
            ),
            "terminal_marker_written_last": True,
            "terminal_publication_operation": (
                "FSYNCED_STAGED_HARDLINK_NO_REPLACE"
            ),
            "terminal_declared_member_names": expected_names,
        }.items()
    ):
        raise BindingError("EVT037 artifact truth differs")


def _load_runtime_lineage_config_current(
    repo: Path,
    head: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    lineage = config["authority_inputs"]["required_predecessor_authority"][
        "runtime_lineage_authority"
    ]
    payload = _verify_current_file(
        repo,
        head,
        lineage["bound_config_path"],
        lineage["bound_config_sha256"],
    )
    runtime_config = strict_json(payload, label="EVT037 bound config")
    _validate_runtime_lineage_config(runtime_config, config)
    return runtime_config


def validate_production_authority(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate I or B lifecycle without touching source or output."""

    validate_static_config(config)
    repo = Path(config["repository_authority"]["production_repo_root"])
    if repo != PRODUCTION_REPO_ROOT:
        raise BindingError("production repository root differs")
    head = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "rev-parse", "--abbrev-ref", "HEAD") != BRANCH:
        raise BindingError("production branch differs")
    if _git(repo, "status", "--porcelain"):
        raise BindingError("production worktree is not clean")
    if _git(repo, "rev-parse", f"refs/remotes/origin/{BRANCH}") != head:
        raise BindingError("origin tracking ref is not current HEAD")

    binding = config["implementation_binding"]
    b_payload: bytes | None = None
    if binding["status"] == UNKNOWN:
        implementation = head
        binding_commit = UNKNOWN
        lifecycle = "IMPLEMENTATION_UNBOUND_I"
        _require_single_parent(
            repo, implementation, IMPLEMENTATION_BASE_COMMIT, label="producer I"
        )
        if _commit_changed_paths(repo, implementation) != EXPECTED_I_PATHS:
            raise BindingError("producer I changed-path set differs")
        for path in EXPECTED_I_PATHS:
            _verify_blob(repo, implementation, path)
    else:
        implementation = binding["implementation_commit"]
        lifecycle = "BOUND_B_OR_CLEAN_DESCENDANT"
        _require_single_parent(
            repo, implementation, IMPLEMENTATION_BASE_COMMIT, label="producer I"
        )
        if _commit_changed_paths(repo, implementation) != EXPECTED_I_PATHS:
            raise BindingError("producer I changed-path set differs")
        _require_ancestor(repo, implementation, head, label="producer I to current")
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
            raise BindingError("producer binding commit is absent")
        binding_commit = successors[0]
        _require_single_parent(repo, binding_commit, implementation, label="producer B")
        if _commit_changed_paths(repo, binding_commit) != EXPECTED_B_PATHS:
            raise BindingError("producer B is not config-only")
        i_config = strict_json(
            _verify_blob(repo, implementation, CONFIG_REPO_PATH),
            label="producer I config",
        )
        b_payload = _verify_blob(repo, binding_commit, CONFIG_REPO_PATH)
        b_config = strict_json(b_payload, label="producer B config")
        validate_i_to_b_config_pair(
            i_config,
            b_config,
            implementation_commit=implementation,
            script_sha256=binding["implementation_script_sha256"],
            test_sha256=binding["implementation_test_sha256"],
        )
        if b_config != dict(config):
            raise BindingError("bound producer config drifted after B")
        for path, digest in (
            (SCRIPT_REPO_PATH, binding["implementation_script_sha256"]),
            (TEST_REPO_PATH, binding["implementation_test_sha256"]),
        ):
            implementation_blob = _verify_blob(repo, implementation, path, digest)
            current_blob = _verify_current_file(repo, head, path, digest)
            if implementation_blob != current_blob:
                raise BindingError(f"bound producer implementation drifted: {path}")

    current_payload = _verify_current_file(repo, head, CONFIG_REPO_PATH)
    if b_payload is not None and current_payload != b_payload:
        raise BindingError("bound producer config bytes drifted after B")
    if strict_json(current_payload, label="current producer config") != dict(config):
        raise BindingError("in-memory producer config differs from HEAD")
    if binding["status"] == UNKNOWN:
        _verify_current_file(repo, head, SCRIPT_REPO_PATH)
        _verify_current_file(repo, head, TEST_REPO_PATH)

    authority = config["authority_inputs"]
    v3_spec = authority["dec019_gse200304_v3"]
    v3_payload = _verify_current_file(
        repo, head, v3_spec["config_path"], v3_spec["config_sha256"]
    )
    _verify_current_file(
        repo,
        head,
        v3_spec["implementation_script_path"],
        v3_spec["implementation_script_sha256"],
    )
    _verify_current_file(
        repo,
        head,
        v3_spec["implementation_test_path"],
        v3_spec["implementation_test_sha256"],
    )
    _validate_v3_compatibility(v3_payload, config)
    old = authority["published_endpoint_qualifier"]
    for path_key, sha_key in (
        ("config_path", "config_sha256"),
        ("implementation_script_path", "implementation_script_sha256"),
        ("implementation_test_path", "implementation_test_sha256"),
    ):
        _verify_current_file(repo, head, old[path_key], old[sha_key])
    _load_runtime_lineage_config_current(repo, head, config)
    return {
        "mode": "PRODUCTION_GIT_AUTHORITY",
        "lifecycle_state": lifecycle,
        "implementation_base_commit": IMPLEMENTATION_BASE_COMMIT,
        "implementation_commit": implementation,
        "binding_commit": binding_commit,
        "current_head": head,
        "config_core_sha256": FROZEN_CONFIG_CORE_SHA256,
    }


def _safe_absolute_path(path: Path, *, forbidden_tokens: Sequence[str], label: str) -> None:
    if not path.is_absolute() or len(path.parts) < 2:
        raise ScopeViolation(f"{label} must be absolute")
    if any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise ScopeViolation(f"{label} contains an unsafe component")
    lowered = os.fspath(path).casefold()
    hits = [token for token in forbidden_tokens if token.casefold() in lowered]
    if hits:
        raise ScopeViolation(f"{label} contains forbidden token(s): {','.join(hits)}")


DirectoryIdentityChain = tuple[tuple[int, int], ...]


def _open_directory_root_to_leaf_with_chain(
    path: Path,
    *,
    label: str,
) -> tuple[int, DirectoryIdentityChain]:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise InputIntegrityError("O_NOFOLLOW/O_DIRECTORY is unavailable")
    if not path.is_absolute() or path.parts[0] != os.sep:
        raise ScopeViolation(f"{label} must be absolute")
    flags = os.O_RDONLY | directory | nofollow | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(os.sep, flags)
    root_stat = os.fstat(descriptor)
    identities = [(root_stat.st_dev, root_stat.st_ino)]
    try:
        for component in path.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                raise InputIntegrityError(
                    f"{label} contains a symlink or non-directory component"
                ) from exc
            os.close(descriptor)
            descriptor = child
            child_stat = os.fstat(descriptor)
            identities.append((child_stat.st_dev, child_stat.st_ino))
        return descriptor, tuple(identities)
    except Exception:
        os.close(descriptor)
        raise


def _open_directory_root_to_leaf(path: Path, *, label: str) -> int:
    descriptor, _identity_chain = _open_directory_root_to_leaf_with_chain(
        path,
        label=label,
    )
    return descriptor


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
    )


def _directory_path_identity(path: Path) -> tuple[int, int]:
    try:
        value = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise InputIntegrityError(f"directory path identity is unavailable: {path}") from exc
    if not stat.S_ISDIR(value.st_mode):
        raise InputIntegrityError(f"directory path is no longer a directory: {path}")
    return value.st_dev, value.st_ino


def _assert_directory_identity(
    descriptor: int,
    path: Path,
    expected: tuple[int, int],
    *,
    label: str,
) -> None:
    observed_fd = os.fstat(descriptor)
    if (observed_fd.st_dev, observed_fd.st_ino) != expected:
        raise InputIntegrityError(f"{label} descriptor identity changed")
    if _directory_path_identity(path) != expected:
        raise InputIntegrityError(f"{label} path was renamed or replaced")


def _read_worktree_authority_file(repo: Path, relative_path: str) -> bytes:
    """Read a Git authority file through no-follow directory descriptors."""

    pure = PurePosixPath(relative_path)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or "\\" in relative_path
    ):
        raise BindingError(f"working authority path is unsafe: {relative_path}")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory_flag is None:
        raise BindingError("O_NOFOLLOW/O_DIRECTORY is unavailable for authority read")
    try:
        root_fd = _open_directory_root_to_leaf(repo, label="repository root")
    except GateProducerError as exc:
        raise BindingError("repository root cannot be opened safely") from exc
    parent_fd: int | None = None
    file_fd: int | None = None
    try:
        root_stat = os.fstat(root_fd)
        root_identity = (root_stat.st_dev, root_stat.st_ino)
        _assert_directory_identity(root_fd, repo, root_identity, label="repository root")
        parent_fd = os.dup(root_fd)
        parent_path = repo
        directory_flags = (
            os.O_RDONLY
            | directory_flag
            | nofollow
            | getattr(os, "O_CLOEXEC", 0)
        )
        for component in pure.parts[:-1]:
            try:
                child_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            except OSError as exc:
                raise BindingError(
                    f"working authority parent is unsafe: {relative_path}"
                ) from exc
            os.close(parent_fd)
            parent_fd = child_fd
            parent_path = parent_path / component
        parent_stat = os.fstat(parent_fd)
        parent_identity = (parent_stat.st_dev, parent_stat.st_ino)
        _assert_directory_identity(
            parent_fd,
            parent_path,
            parent_identity,
            label="working authority parent",
        )
        try:
            file_fd = os.open(
                pure.name,
                os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_fd,
            )
        except OSError as exc:
            raise BindingError(
                f"working authority input cannot be opened safely: {relative_path}"
            ) from exc
        before = os.fstat(file_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o644
            or before.st_size < 0
            or before.st_size > MAX_AUTHORITY_FILE_BYTES
        ):
            raise BindingError(
                f"working authority input geometry/mode differs: {relative_path}"
            )
        chunks: list[bytes] = []
        remaining = before.st_size + 1
        while remaining > 0:
            chunk = os.read(file_fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) != before.st_size or os.read(file_fd, 1):
            raise BindingError(
                f"working authority bounded snapshot differs: {relative_path}"
            )
        after = os.fstat(file_fd)
        if _identity(before) != _identity(after):
            raise BindingError(
                f"working authority input changed during read: {relative_path}"
            )
        try:
            visible = os.stat(pure.name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as exc:
            raise BindingError(
                f"working authority input disappeared: {relative_path}"
            ) from exc
        if _identity(visible) != _identity(after):
            raise BindingError(
                f"working authority input path was replaced: {relative_path}"
            )
        _assert_directory_identity(
            parent_fd,
            parent_path,
            parent_identity,
            label="working authority parent",
        )
        _assert_directory_identity(root_fd, repo, root_identity, label="repository root")
        return payload
    except InputIntegrityError as exc:
        raise BindingError(
            f"working authority path identity differs: {relative_path}"
        ) from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if parent_fd is not None:
            os.close(parent_fd)
        os.close(root_fd)


def _read_verified_member_at(
    root_fd: int,
    root_path: Path,
    root_identity: tuple[int, int],
    member: Mapping[str, Any],
    *,
    materialize: bool,
    fault: FaultInjector | None = None,
) -> bytes | None:
    name = str(member["relative_path"])
    if PurePosixPath(name).name != name or name in {"", ".", ".."}:
        raise ScopeViolation(f"source member name is unsafe: {name}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=root_fd)
    except OSError as exc:
        raise InputIntegrityError(f"source member cannot be opened safely: {name}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise InputIntegrityError(f"source is not a single-link regular file: {name}")
        expected_size = int(member["bytes"])
        if before.st_size != expected_size:
            raise InputIntegrityError(f"source byte count differs: {name}")
        if fault is not None:
            fault(f"after_open:{member['asset_id']}")
        chunks: list[bytes] | None = [] if materialize else None
        hasher = hashlib.sha256()
        observed_size = 0
        remaining = expected_size + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            hasher.update(chunk)
            observed_size += len(chunk)
            if chunks is not None:
                chunks.append(chunk)
            remaining -= len(chunk)
        if observed_size != expected_size:
            raise InputIntegrityError(f"source bounded snapshot size differs: {name}")
        if os.read(descriptor, 1):
            raise InputIntegrityError(f"source grew beyond its frozen size: {name}")
        if fault is not None:
            fault(f"after_read:{member['asset_id']}")
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise InputIntegrityError(f"source changed during same-FD read: {name}")
        try:
            visible = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        except OSError as exc:
            raise InputIntegrityError(f"source path disappeared after read: {name}") from exc
        if _identity(visible) != _identity(after):
            raise InputIntegrityError(f"source path was replaced during read: {name}")
        _assert_directory_identity(
            root_fd, root_path, root_identity, label="source root"
        )
        if hasher.hexdigest() != member["sha256"]:
            raise InputIntegrityError(f"source SHA differs: {name}")
        return b"".join(chunks) if chunks is not None else None
    finally:
        os.close(descriptor)


def read_source_inputs(
    config: Mapping[str, Any],
    *,
    fault: FaultInjector | None = None,
) -> dict[str, bytes]:
    """Hash all seven members; materialize only S2/S3 for selective parsing."""

    source = config["source_contract"]
    root = Path(source["data_root"])
    _safe_absolute_path(
        root,
        forbidden_tokens=source["forbidden_path_tokens"],
        label="source root",
    )
    root_fd = _open_directory_root_to_leaf(root, label="source root")
    try:
        root_stat = os.fstat(root_fd)
        root_identity = (root_stat.st_dev, root_stat.st_ino)
        _assert_directory_identity(root_fd, root, root_identity, label="source root")
        observed_names = sorted(os.listdir(root_fd))
        if observed_names != sorted(source["exact_member_names"]):
            raise InputIntegrityError("source directory is not the exact seven-member bundle")
        by_id: dict[str, Mapping[str, Any]] = {}
        for member in source["members"]:
            name = member["relative_path"]
            try:
                visible = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            except OSError as exc:
                raise InputIntegrityError(f"source member stat failed: {name}") from exc
            if not stat.S_ISREG(visible.st_mode) or visible.st_nlink != 1:
                raise InputIntegrityError(
                    f"source member is not a single-link regular file: {name}"
                )
            if visible.st_size != member["bytes"]:
                raise InputIntegrityError(f"source member byte count differs: {name}")
            by_id[member["asset_id"]] = member
        if set(by_id) != {member["asset_id"] for member in source["members"]}:
            raise InputIntegrityError("duplicate source asset ID")
        payloads: dict[str, bytes] = {}
        readable = set(source["readable_asset_ids"])
        for member in source["members"]:
            asset_id = member["asset_id"]
            materialize = asset_id in readable
            payload = _read_verified_member_at(
                root_fd,
                root,
                root_identity,
                by_id[asset_id],
                materialize=materialize,
                fault=fault,
            )
            if materialize:
                if payload is None:
                    raise InputIntegrityError("readable source was not materialized")
                payloads[asset_id] = payload
        _assert_directory_identity(root_fd, root, root_identity, label="source root")
        if sorted(os.listdir(root_fd)) != sorted(source["exact_member_names"]):
            raise InputIntegrityError("source directory membership changed during audit")
        return payloads
    finally:
        os.close(root_fd)


def _read_exact_predecessor_bundle(
    root: Path,
    predecessor: Mapping[str, Any],
    *,
    forbidden_tokens: Sequence[str],
) -> dict[str, bytes]:
    _safe_absolute_path(
        root,
        forbidden_tokens=forbidden_tokens,
        label="required predecessor bundle",
    )
    root_fd = _open_directory_root_to_leaf(
        root, label="required predecessor bundle"
    )
    try:
        root_stat = os.fstat(root_fd)
        root_identity = (root_stat.st_dev, root_stat.st_ino)
        _assert_directory_identity(
            root_fd,
            root,
            root_identity,
            label="required predecessor bundle",
        )
        members = predecessor["members"]
        expected_names = [member["name"] for member in members]
        if sorted(os.listdir(root_fd)) != sorted(expected_names):
            raise InputIntegrityError(
                "required predecessor is not the exact five-member bundle"
            )
        payloads: dict[str, bytes] = {}
        for member in members:
            adapted = {
                "asset_id": f"PREDECESSOR:{member['name']}",
                "relative_path": member["name"],
                "bytes": member["bytes"],
                "sha256": member["sha256"],
            }
            payload = _read_verified_member_at(
                root_fd,
                root,
                root_identity,
                adapted,
                materialize=True,
            )
            if payload is None:
                raise InputIntegrityError("predecessor member was not materialized")
            payloads[member["name"]] = payload
        _assert_directory_identity(
            root_fd,
            root,
            root_identity,
            label="required predecessor bundle",
        )
        if sorted(os.listdir(root_fd)) != sorted(expected_names):
            raise InputIntegrityError(
                "required predecessor membership changed during replay"
            )
        return payloads
    finally:
        os.close(root_fd)


def _validate_predecessor_payloads(
    payloads: Mapping[str, bytes],
    predecessor: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
) -> None:
    checksummed_names = sorted(
        member["name"] for member in predecessor["members"][:3]
    )
    expected_sums = "".join(
        f"{sha256(payloads[name])}  {name}\n" for name in checksummed_names
    ).encode("ascii")
    if payloads.get("SHA256SUMS") != expected_sums:
        raise InputIntegrityError("required predecessor SHA256SUMS closure differs")
    marker = strict_json(
        payloads["PUBLICATION_COMMIT.json"],
        label="required predecessor terminal marker",
    )
    truth = runtime_config["artifact_truth"]
    expected_marker_values = {
        "record_type": truth["terminal_record_type"],
        "bundle_member_names": truth["terminal_declared_member_names"],
        "bundle_member_count": 4,
        "sha256sums_sha256": sha256(payloads["SHA256SUMS"]),
        "final_output_target_sha256": predecessor[
            "terminal_marker_final_output_target_sha256"
        ],
        "terminal_publication_operation": truth[
            "terminal_publication_operation"
        ],
        "committed": True,
        "terminal_marker_written_last": True,
    }
    if any(marker.get(key) != value for key, value in expected_marker_values.items()):
        raise InputIntegrityError("required predecessor terminal marker differs")


def replay_predecessor_authority(config: Mapping[str, Any]) -> dict[str, Any]:
    """Replay EVT037 config and the exact five-member predecessor bundle."""

    repo = Path(config["repository_authority"]["production_repo_root"])
    head = _git(repo, "rev-parse", "HEAD")
    runtime_config = _load_runtime_lineage_config_current(repo, head, config)
    predecessor = config["authority_inputs"]["required_predecessor_authority"]
    payloads = _read_exact_predecessor_bundle(
        Path(predecessor["trusted_absolute_bundle_path"]),
        predecessor,
        forbidden_tokens=config["source_contract"]["forbidden_path_tokens"],
    )
    _validate_predecessor_payloads(payloads, predecessor, runtime_config)
    return {
        "bundle_id": predecessor["bundle_id"],
        "member_count": len(payloads),
        "runtime_event_id": runtime_config["event_id"],
        "terminal_marker_sha256": sha256(payloads["PUBLICATION_COMMIT.json"]),
    }


def _u32(value: int) -> bytes:
    return struct.pack(">I", value)


def _u64(value: int) -> bytes:
    return struct.pack(">Q", value)


def _length_prefixed(parts: Iterable[bytes]) -> bytes:
    output = bytearray()
    for part in parts:
        output.extend(_u64(len(part)))
        output.extend(part)
    return bytes(output)


def _domain_hash(domain: bytes, parts: Iterable[bytes]) -> bytes:
    # Domain is an ordinary first field, not an unframed prefix.  This makes
    # the encoding injective across both domain and payload boundaries.
    return hashlib.sha256(_length_prefixed((domain, *parts))).digest()


@dataclass(frozen=True)
class CsvSelectiveRow:
    captured: tuple[str, ...]
    field_count: int
    all_fields_nonempty: bool


def iter_rfc4180_selective(payload: bytes) -> Iterable[CsvSelectiveRow]:
    """Parse RFC4180 bytes while capturing header-all, then only ID/Type.

    Opaque fields are never accumulated, decoded, or hashed field-by-field.
    The state machine tracks only quoting/newlines and non-empty geometry;
    exact asset SHA plus the frozen predecessor carries sequence/content
    geometry.  This producer independently commits only locator lineage.
    """

    position = 0
    row_index = 0
    field_index = 0
    row_field_lengths: list[int] = []
    captured: list[str] = []
    all_nonempty = True
    in_quotes = False
    after_quote = False
    field_started = False
    field_length = 0
    captured_bytes: bytearray | None = bytearray()

    def reset_field() -> None:
        nonlocal field_length, captured_bytes, field_started
        field_length = 0
        capture_limit = 6 if row_index == 0 else 2
        captured_bytes = bytearray() if field_index < capture_limit else None
        field_started = False

    def add_byte(value: int) -> None:
        nonlocal field_length, field_started
        field_length += 1
        field_started = True
        if captured_bytes is not None:
            captured_bytes.append(value)

    def finish_field() -> None:
        nonlocal field_index, all_nonempty
        row_field_lengths.append(field_length)
        if field_length == 0:
            all_nonempty = False
        if captured_bytes is not None:
            try:
                value = bytes(captured_bytes).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise TableAuditError("captured S2 CSV field is not UTF-8") from exc
            captured.append(value)
        field_index += 1
        reset_field()

    def finish_row() -> CsvSelectiveRow:
        nonlocal row_index, field_index, row_field_lengths, captured, all_nonempty
        finish_field()
        value = CsvSelectiveRow(
            captured=tuple(captured),
            field_count=len(row_field_lengths),
            all_fields_nonempty=all_nonempty,
        )
        row_index += 1
        field_index = 0
        row_field_lengths = []
        captured = []
        all_nonempty = True
        reset_field()
        return value

    reset_field()
    while position < len(payload):
        byte = payload[position]
        if in_quotes:
            if byte == 0x22:
                if position + 1 < len(payload) and payload[position + 1] == 0x22:
                    add_byte(0x22)
                    position += 2
                    continue
                in_quotes = False
                after_quote = True
                position += 1
                continue
            add_byte(byte)
            position += 1
            continue

        if after_quote:
            if byte == 0x2C:
                finish_field()
                after_quote = False
                position += 1
                continue
            if byte == 0x0A:
                after_quote = False
                position += 1
                yield finish_row()
                continue
            if byte == 0x0D and position + 1 < len(payload) and payload[position + 1] == 0x0A:
                after_quote = False
                position += 2
                yield finish_row()
                continue
            raise TableAuditError("S2 CSV has bytes after a closing quote")

        if not field_started and field_length == 0 and byte == 0x22:
            in_quotes = True
            field_started = True
            position += 1
            continue
        if byte == 0x22:
            raise TableAuditError("S2 CSV has a quote inside an unquoted field")
        if byte == 0x2C:
            finish_field()
            position += 1
            continue
        if byte == 0x0A:
            position += 1
            yield finish_row()
            continue
        if byte == 0x0D:
            if position + 1 >= len(payload) or payload[position + 1] != 0x0A:
                raise TableAuditError("S2 CSV has a bare carriage return")
            position += 2
            yield finish_row()
            continue
        add_byte(byte)
        position += 1

    if in_quotes:
        raise TableAuditError("S2 CSV ends inside a quoted field")
    if after_quote or field_started or field_length or row_field_lengths:
        yield finish_row()


def _canonical_header_sha256(header: Sequence[str]) -> str:
    payload = json.dumps(
        list(header), ensure_ascii=True, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    return sha256(payload)


@dataclass(frozen=True)
class S2Audit:
    pair_keys: frozenset[bytes]
    pair_digests: Mapping[bytes, bytes]
    aggregates: Mapping[str, Any]


def _configured_domain(contract: Mapping[str, Any], key: str) -> bytes:
    value = contract.get(key)
    if not isinstance(value, str) or not value:
        raise TableAuditError(f"configured digest domain is absent: {key}")
    return value.encode("utf-8")


def _join_key_digest(locator: str, contract: Mapping[str, Any]) -> bytes:
    if not locator or PAIR_ID_RE.fullmatch(locator) is None:
        raise TableAuditError("locator does not match the frozen exact-key grammar")
    return _domain_hash(
        _configured_domain(contract, "join_key_domain"),
        (locator.encode("utf-8"),),
    )


def audit_table_s2(
    payload: bytes,
    spec: Mapping[str, Any],
    locator_contract: Mapping[str, Any],
) -> S2Audit:
    rows = iter(iter_rfc4180_selective(payload))
    try:
        header_row = next(rows)
    except StopIteration as exc:
        raise TableAuditError("Table S2 is empty") from exc
    header = list(header_row.captured)
    if header and header[0].startswith("\ufeff"):
        header[0] = header[0][1:]
    if header_row.field_count != len(spec["exact_header"]) or header != spec["exact_header"]:
        raise TableAuditError("Table S2 header differs")
    if _canonical_header_sha256(header) != spec["canonical_compact_header_json_sha256"]:
        raise TableAuditError("Table S2 header hash differs")

    raw_count = 0
    source_digest = bytes.fromhex(sha256(payload))
    raw_types: Counter[str] = Counter()
    role_leaves_by_key: dict[bytes, dict[str, list[bytes]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for physical_index, row in enumerate(rows, start=1):
        raw_count += 1
        if row.field_count != len(spec["exact_header"]) or not row.all_fields_nonempty:
            raise TableAuditError("Table S2 row geometry or required field differs")
        if len(row.captured) != 2:
            raise TableAuditError("Table S2 selective capture width differs")
        key, row_type = row.captured
        if row_type not in {"WT", "Mutant", "Control"}:
            raise TableAuditError("Table S2 row type is outside the closed enum")
        if row_type == "Control":
            # Controls are outside mutation-pair lineage.  Hash and discard
            # their locator so physical multiplicity can still be reconciled
            # without retaining the raw identifier.
            join_key = _domain_hash(
                _configured_domain(locator_contract, "join_key_domain"),
                (key.encode("utf-8"),),
            )
        else:
            join_key = _join_key_digest(key, locator_contract)
        physical_leaf = _domain_hash(
            _configured_domain(locator_contract, "s2_physical_row_domain"),
            (
                source_digest,
                _u64(physical_index),
                join_key,
                row_type.encode("ascii"),
            ),
        )
        raw_types[row_type] += 1
        role_leaves_by_key[join_key][row_type].append(physical_leaf)
        del key

    if raw_count != spec["raw_row_count"]:
        raise TableAuditError("Table S2 raw row count differs")
    if dict(raw_types) != spec["raw_type_counts"]:
        raise TableAuditError("Table S2 raw type counts differ")
    raw_key_hist = Counter(
        sum(len(leaves) for leaves in by_role.values())
        for by_role in role_leaves_by_key.values()
    )
    if {str(key): value for key, value in sorted(raw_key_hist.items())} != spec[
        "raw_id_row_multiplicity_counts"
    ]:
        raise TableAuditError("Table S2 raw key multiplicity counts differ")

    pair_digests: dict[bytes, bytes] = {}
    controls = 0
    duplicated_pairs = 0
    logical_types: Counter[str] = Counter()
    for join_key, by_type in role_leaves_by_key.items():
        if set(by_type) == {"Control"} and len(by_type["Control"]) == 1:
            controls += 1
            logical_types["Control"] += 1
            continue
        if set(by_type) != {"WT", "Mutant"}:
            raise TableAuditError("Table S2 pair lacks the exact WT/Mutant roles")
        role_counts = {role: len(by_type[role]) for role in ("WT", "Mutant")}
        if role_counts == {"WT": 2, "Mutant": 2}:
            duplicated_pairs += 1
        elif role_counts != {"WT": 1, "Mutant": 1}:
            raise TableAuditError("Table S2 pair raw multiplicity differs")
        arm_commitments = []
        for role in ("Mutant", "WT"):
            arm_commitments.append(
                _domain_hash(
                    _configured_domain(locator_contract, "s2_pair_arm_domain"),
                    (join_key, role.encode("ascii"), *sorted(by_type[role])),
                )
            )
            logical_types[role] += 1
        pair_digests[join_key] = _domain_hash(
            _configured_domain(locator_contract, "pair_lineage_domain"),
            (join_key, *arm_commitments),
        )
    if len(pair_digests) != spec["deduplicated_pair_count"]:
        raise TableAuditError("Table S2 deduplicated pair count differs")
    if controls != spec["deduplicated_control_count"]:
        raise TableAuditError("Table S2 control count differs")
    if duplicated_pairs != spec["duplicated_pair_count"]:
        raise TableAuditError("Table S2 duplicated pair count differs")
    if dict(logical_types) != spec["deduplicated_type_counts"]:
        raise TableAuditError("Table S2 logical deduplicated type counts differ")
    logical_unique_count = sum(logical_types.values())
    duplicate_extra = raw_count - logical_unique_count
    duplicated_groups = duplicated_pairs * 2
    if logical_unique_count != spec["unique_content_row_count"]:
        raise TableAuditError("Table S2 inherited unique-content count differs")
    if duplicate_extra != spec["duplicate_extra_row_count"]:
        raise TableAuditError("Table S2 inherited duplicate-extra count differs")
    if duplicated_groups != spec["duplicated_content_group_count"]:
        raise TableAuditError("Table S2 inherited duplicate-group count differs")
    if spec["duplicated_content_multiplicity"] != 2:
        raise TableAuditError("Table S2 inherited duplicate multiplicity differs")
    return S2Audit(
        pair_keys=frozenset(pair_digests),
        pair_digests=pair_digests,
        aggregates={
            "raw_row_count": raw_count,
            "unique_content_row_count": logical_unique_count,
            "duplicate_extra_row_count": duplicate_extra,
            "duplicated_content_group_count": duplicated_groups,
            "duplicated_pair_count": duplicated_pairs,
            "deduplicated_pair_count": len(pair_digests),
            "deduplicated_control_count": controls,
        },
    )


@dataclass(frozen=True)
class StringToken:
    kind: str
    value: str | int | bytes


@dataclass(frozen=True)
class PrimaryTokenRow:
    row_number: int
    locator: StringToken
    comparison: StringToken
    statistic_states: tuple[StringToken, StringToken, StringToken]


@dataclass(frozen=True)
class S3Audit:
    pair_keys: frozenset[bytes]
    finite_pair_keys: Mapping[str, frozenset[bytes]]
    primary_finite_row_digests: Mapping[bytes, bytes]
    aggregates: Mapping[str, Any]


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _xml_bytes(payload: bytes, *, label: str) -> bytes:
    if re.search(rb"<!\s*(?:doctype|entity)\b", payload, flags=re.IGNORECASE):
        raise TableAuditError(f"{label} contains a DTD/entity declaration")
    return payload


def _xlsx_read_member(
    archive: zipfile.ZipFile,
    name: str,
    *,
    maximum_bytes: int = 64 * 1024 * 1024,
) -> bytes:
    try:
        info = archive.getinfo(name)
    except KeyError as exc:
        raise TableAuditError(f"XLSX member is absent: {name}") from exc
    if info.file_size < 0 or info.file_size > maximum_bytes:
        raise TableAuditError(f"XLSX member exceeds the bounded size: {name}")
    with archive.open(info, "r") as handle:
        payload = handle.read(maximum_bytes + 1)
    if len(payload) != info.file_size or len(payload) > maximum_bytes:
        raise TableAuditError(f"XLSX member bounded read differs: {name}")
    return payload


def _xlsx_read_through_first_row(
    archive: zipfile.ZipFile,
    name: str,
    *,
    maximum_prefix_bytes: int = 2 * 1024 * 1024,
) -> bytes:
    """Retain only worksheet metadata through the closing header row."""

    try:
        info = archive.getinfo(name)
    except KeyError as exc:
        raise TableAuditError(f"XLSX member is absent: {name}") from exc
    if info.file_size < 0 or info.file_size > 64 * 1024 * 1024:
        raise TableAuditError(f"XLSX member exceeds the bounded size: {name}")
    closing_row = re.compile(rb"</(?:[A-Za-z_][A-Za-z0-9_.-]*:)?row\s*>")
    prefix = bytearray()
    with archive.open(info, "r") as handle:
        while len(prefix) <= maximum_prefix_bytes:
            chunk = handle.read(min(256, maximum_prefix_bytes + 1 - len(prefix)))
            if not chunk:
                break
            prefix.extend(chunk)
            match = closing_row.search(prefix)
            if match is not None:
                return bytes(prefix[: match.end()])
    raise TableAuditError("control worksheet header exceeds the bounded prefix")


def _normalize_xlsx_target(base: str, target: str) -> str:
    raw = PurePosixPath(target)
    if not target or "\\" in target or ".." in raw.parts:
        raise TableAuditError("XLSX relationship target is unsafe")
    if target.startswith("/"):
        normalized = posixpath.normpath(target.lstrip("/"))
    else:
        normalized = posixpath.normpath(posixpath.join(base, target))
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts or not normalized.startswith("xl/"):
        raise TableAuditError("XLSX relationship target leaves xl/")
    return normalized


def _workbook_sheet_paths(
    workbook_xml: bytes,
    relationships_xml: bytes,
) -> list[tuple[str, str]]:
    try:
        workbook = ET.fromstring(_xml_bytes(workbook_xml, label="workbook.xml"))
        relationships = ET.fromstring(
            _xml_bytes(relationships_xml, label="workbook relationships")
        )
    except ET.ParseError as exc:
        raise TableAuditError("XLSX workbook metadata is invalid XML") from exc
    relation_targets: dict[str, str] = {}
    for relation in relationships.iter():
        if _local_name(relation.tag) != "Relationship":
            continue
        relation_id = relation.attrib.get("Id")
        target = relation.attrib.get("Target")
        mode = relation.attrib.get("TargetMode")
        if relation_id and target and mode != "External":
            relation_targets[relation_id] = _normalize_xlsx_target("xl", target)
    result: list[tuple[str, str]] = []
    for sheet in workbook.iter():
        if _local_name(sheet.tag) != "sheet":
            continue
        name = sheet.attrib.get("name")
        relation_id = next(
            (
                value
                for key, value in sheet.attrib.items()
                if _local_name(key) == "id"
            ),
            None,
        )
        if not isinstance(name, str) or relation_id not in relation_targets:
            raise TableAuditError("XLSX sheet relationship is incomplete")
        path = relation_targets[relation_id]
        if not path.startswith("xl/worksheets/") or not path.endswith(".xml"):
            raise TableAuditError("XLSX sheet target is outside worksheets")
        result.append((name, path))
    return result


@dataclass(frozen=True)
class XmlLexicalTag:
    start: int
    end: int
    kind: str
    qualified_name: str
    local_name: str
    raw: bytes


@dataclass(frozen=True)
class XmlElementSpan:
    start_tag: XmlLexicalTag
    start: int
    end: int
    content_start: int
    content_end: int


def _iter_xml_tags(
    payload: bytes,
    *,
    label: str,
    start: int = 0,
    end: int | None = None,
) -> Iterable[XmlLexicalTag]:
    """Lex XML tags without decoding any character-data payload."""

    limit = len(payload) if end is None else end
    if not (0 <= start <= limit <= len(payload)):
        raise TableAuditError(f"{label} XML lexical bounds are invalid")
    if start == 0 and limit == len(payload):
        _xml_bytes(payload, label=label)
    position = start
    whitespace = b" \t\r\n"
    while True:
        tag_start = payload.find(b"<", position, limit)
        if tag_start < 0:
            return
        if payload.startswith(b"<!--", tag_start, limit):
            comment_end = payload.find(b"-->", tag_start + 4, limit)
            if comment_end < 0:
                raise TableAuditError(f"{label} has an unterminated XML comment")
            position = comment_end + 3
            continue
        if payload.startswith(b"<?", tag_start, limit):
            instruction_end = payload.find(b"?>", tag_start + 2, limit)
            if instruction_end < 0:
                raise TableAuditError(f"{label} has an unterminated XML instruction")
            position = instruction_end + 2
            continue
        if payload.startswith(b"<!", tag_start, limit):
            raise TableAuditError(f"{label} contains an unsupported XML declaration")
        quote: int | None = None
        cursor = tag_start + 1
        while cursor < limit:
            value = payload[cursor]
            if quote is None and value in {0x22, 0x27}:
                quote = value
            elif quote is not None and value == quote:
                quote = None
            elif quote is None and value == 0x3E:
                break
            cursor += 1
        if cursor >= limit or quote is not None:
            raise TableAuditError(f"{label} has an unterminated XML tag")
        raw = payload[tag_start : cursor + 1]
        body = raw[1:-1].strip()
        if not body:
            raise TableAuditError(f"{label} has an empty XML tag")
        if body.startswith(b"/"):
            kind = "end"
            name_bytes = body[1:].strip()
            if any(value in whitespace for value in name_bytes):
                raise TableAuditError(f"{label} closing XML tag is malformed")
        else:
            empty = body.endswith(b"/")
            content = body[:-1].rstrip() if empty else body
            name_bytes = content.split(None, 1)[0]
            kind = "empty" if empty else "start"
        try:
            qualified = name_bytes.decode("ascii")
        except UnicodeDecodeError as exc:
            raise TableAuditError(f"{label} XML tag name is not ASCII") from exc
        if not qualified or any(character in qualified for character in "<>/='\""):
            raise TableAuditError(f"{label} XML tag name is invalid")
        yield XmlLexicalTag(
            start=tag_start,
            end=cursor + 1,
            kind=kind,
            qualified_name=qualified,
            local_name=qualified.rsplit(":", 1)[-1],
            raw=raw,
        )
        position = cursor + 1


def _assert_xml_balanced(payload: bytes, *, label: str) -> None:
    stack: list[str] = []
    for tag in _iter_xml_tags(payload, label=label):
        if tag.kind == "start":
            stack.append(tag.qualified_name)
        elif tag.kind == "end":
            if not stack or stack.pop() != tag.qualified_name:
                raise TableAuditError(f"{label} XML nesting differs")
    if stack:
        raise TableAuditError(f"{label} XML document is truncated")


def _xml_element_spans(
    payload: bytes,
    local_name: str,
    *,
    label: str,
    start: int = 0,
    end: int | None = None,
) -> Iterable[XmlElementSpan]:
    active: XmlLexicalTag | None = None
    depth = 0
    for tag in _iter_xml_tags(
        payload,
        label=label,
        start=start,
        end=end,
    ):
        if tag.local_name != local_name:
            continue
        if tag.kind == "empty":
            if depth:
                raise TableAuditError(f"{label} nests {local_name} elements")
            yield XmlElementSpan(tag, tag.start, tag.end, tag.end, tag.end)
        elif tag.kind == "start":
            if depth:
                raise TableAuditError(f"{label} nests {local_name} elements")
            active = tag
            depth = 1
        else:
            if depth != 1 or active is None:
                raise TableAuditError(f"{label} closes an unopened {local_name}")
            yield XmlElementSpan(
                active,
                active.start,
                tag.end,
                active.end,
                tag.start,
            )
            active = None
            depth = 0
    if depth:
        raise TableAuditError(f"{label} truncates a {local_name} element")


def _tag_attributes(tag: XmlLexicalTag, *, label: str) -> dict[str, str]:
    candidate = tag.raw if tag.kind == "empty" else tag.raw[:-1] + b"/>"
    try:
        element = ET.fromstring(candidate)
    except ET.ParseError as exc:
        raise TableAuditError(f"{label} XML attributes are invalid") from exc
    attributes: dict[str, str] = {}
    for key, value in element.attrib.items():
        local = _local_name(key)
        if local in attributes:
            raise TableAuditError(f"{label} duplicates an XML attribute")
        attributes[local] = value
    return attributes


XML_QNAME_BYTES_RE = re.compile(
    rb"[A-Za-z_][A-Za-z0-9_.-]*(?::[A-Za-z_][A-Za-z0-9_.-]*)?"
)
X14AC_NAMESPACE_URI = (
    b"http://schemas.microsoft.com/office/spreadsheetml/2009/9/ac"
)
SPREADSHEETML_NAMESPACE_URI = (
    b"http://schemas.openxmlformats.org/spreadsheetml/2006/main"
)
ROW_DESCENT_RE = re.compile(rb"^(?:0(?:\.[0-9]{1,4})?|1(?:\.0{1,4})?)$")


def _lexical_tag_attributes(
    tag: XmlLexicalTag,
    *,
    label: str,
) -> dict[str, bytes]:
    """Parse start-tag attributes without resolving or materializing subtrees."""

    if tag.kind not in {"start", "empty"}:
        raise TableAuditError(f"{label} is not an opening XML tag")
    body = tag.raw[1:-1]
    if tag.kind == "empty":
        body = body.rstrip()
        if not body.endswith(b"/"):
            raise TableAuditError(f"{label} empty XML tag is malformed")
        body = body[:-1].rstrip()
    name = XML_QNAME_BYTES_RE.match(body)
    if name is None or name.group().decode("ascii") != tag.qualified_name:
        raise TableAuditError(f"{label} XML tag name geometry is invalid")
    position = name.end()
    whitespace = b" \t\r\n"
    attributes: dict[str, bytes] = {}
    while position < len(body):
        before_whitespace = position
        while position < len(body) and body[position] in whitespace:
            position += 1
        if position == len(body):
            break
        if position == before_whitespace:
            raise TableAuditError(f"{label} XML attributes lack a separator")
        match = XML_QNAME_BYTES_RE.match(body, position)
        if match is None:
            raise TableAuditError(f"{label} XML attribute name is invalid")
        qualified_name = match.group().decode("ascii")
        position = match.end()
        while position < len(body) and body[position] in whitespace:
            position += 1
        if position >= len(body) or body[position] != 0x3D:
            raise TableAuditError(f"{label} XML attribute lacks equals")
        position += 1
        while position < len(body) and body[position] in whitespace:
            position += 1
        if position >= len(body) or body[position] not in {0x22, 0x27}:
            raise TableAuditError(f"{label} XML attribute is not quoted")
        quote = body[position]
        position += 1
        value_end = body.find(bytes((quote,)), position)
        if value_end < 0:
            raise TableAuditError(f"{label} XML attribute quote is unterminated")
        value = body[position:value_end]
        if b"<" in value:
            raise TableAuditError(f"{label} XML attribute contains raw markup")
        if qualified_name in attributes:
            raise TableAuditError(f"{label} duplicates an XML attribute")
        attributes[qualified_name] = value
        position = value_end + 1
    return attributes


def _worksheet_has_official_x14ac_namespace(
    payload: bytes,
    *,
    label: str,
) -> bool:
    official_x14ac_namespace: bool | None = None
    for tag in _iter_xml_tags(payload, label=label):
        if tag.kind not in {"start", "empty"}:
            continue
        if official_x14ac_namespace is None:
            if tag.qualified_name != "worksheet" or tag.kind != "start":
                raise TableAuditError(f"{label} root element differs")
            attributes = _lexical_tag_attributes(tag, label=f"{label} root")
            if attributes.get("xmlns") != SPREADSHEETML_NAMESPACE_URI:
                raise TableAuditError(f"{label} default namespace binding differs")
            namespace = attributes.get("xmlns:x14ac")
            if namespace is None:
                official_x14ac_namespace = False
            elif namespace == X14AC_NAMESPACE_URI:
                official_x14ac_namespace = True
            else:
                raise TableAuditError(f"{label} x14ac namespace binding differs")
            continue
        # Reject even same-value redeclarations below the root.  This closed
        # rule keeps every unprefixed row in SpreadsheetML and every
        # x14ac:dyDescent attribute in the official x14ac namespace without
        # interpreting descendant element content.
        if b"xmlns" not in tag.raw:
            continue
        attributes = _lexical_tag_attributes(tag, label=f"{label} descendant")
        if "xmlns" in attributes or "xmlns:x14ac" in attributes:
            raise TableAuditError(
                f"{label} descendant worksheet namespace declaration is forbidden"
            )
    if official_x14ac_namespace is None:
        raise TableAuditError(f"{label} root element is absent")
    return official_x14ac_namespace


def _closed_worksheet_row_number(
    tag: XmlLexicalTag,
    *,
    label: str,
    official_x14ac_namespace: bool,
    expected_span: str,
) -> int:
    if tag.qualified_name != "row":
        raise TableAuditError(f"{label} row namespace/prefix differs")
    attributes = _lexical_tag_attributes(tag, label=label)
    names = set(attributes)
    if names == {"r"}:
        if official_x14ac_namespace:
            raise TableAuditError(f"{label} official row attributes are incomplete")
    elif names == {"r", "spans", "x14ac:dyDescent"}:
        if not official_x14ac_namespace:
            raise TableAuditError(f"{label} uses an unbound x14ac attribute")
        if attributes["spans"] != expected_span.encode("ascii"):
            raise TableAuditError(f"{label} span geometry differs")
        if ROW_DESCENT_RE.fullmatch(attributes["x14ac:dyDescent"]) is None:
            raise TableAuditError(f"{label} descent geometry is invalid")
    else:
        raise TableAuditError(f"{label} attribute set differs from closed profiles")
    raw_row = attributes["r"]
    if re.fullmatch(rb"[1-9][0-9]*", raw_row) is None:
        raise TableAuditError(f"{label} row number is invalid")
    return int(raw_row)


def _cell_ref(tag: XmlLexicalTag) -> tuple[str, int]:
    reference = _tag_attributes(tag, label="XLSX cell").get("r")
    match = CELL_REF_RE.fullmatch(reference or "")
    if match is None:
        raise TableAuditError("XLSX cell reference is invalid")
    return match.group(1), int(match.group(2))


def _selected_element(span: XmlElementSpan, payload: bytes, *, label: str) -> ET.Element:
    try:
        return ET.fromstring(payload[span.start : span.end])
    except ET.ParseError as exc:
        raise TableAuditError(f"{label} selected cell XML is invalid") from exc


def _string_token(
    span: XmlElementSpan,
    payload: bytes,
    *,
    label: str,
) -> StringToken:
    element = _selected_element(span, payload, label=label)
    cell_type = element.attrib.get("t")
    child = lambda name: next(
        (item for item in element if _local_name(item.tag) == name), None
    )
    if cell_type == "s":
        value = child("v")
        text = value.text if value is not None else None
        if text is None or not text.isdigit():
            raise TableAuditError(f"{label} shared-string index is invalid")
        return StringToken("SHARED", int(text))
    if cell_type == "inlineStr":
        inline = child("is")
        if inline is None:
            raise TableAuditError(f"{label} inline string is absent")
        text = "".join(
            child.text or "" for child in inline.iter() if _local_name(child.tag) == "t"
        )
        return StringToken("INLINE", text)
    if cell_type == "str":
        value = child("v")
        if value is None or value.text is None:
            raise TableAuditError(f"{label} string value is absent")
        return StringToken("INLINE", value.text)
    raise TableAuditError(f"{label} is not a string cell")


def _purpose_token(
    token: StringToken,
    purpose: str,
    locator_contract: Mapping[str, Any],
) -> StringToken:
    if token.kind == "SHARED":
        return StringToken(f"SHARED_{purpose}", token.value)
    if token.kind != "INLINE" or not isinstance(token.value, str):
        raise TableAuditError("selective string token has an invalid representation")
    if purpose == "LOCATOR":
        digest = _join_key_digest(token.value, locator_contract)
        return StringToken("JOIN_DIGEST", digest)
    return StringToken(f"INLINE_{purpose}", token.value)


def _statistic_token(
    span: XmlElementSpan,
    payload: bytes,
    *,
    label: str,
    locator_contract: Mapping[str, Any],
) -> StringToken:
    if list(
        _xml_element_spans(
            payload,
            "f",
            label=label,
            start=span.content_start,
            end=span.content_end,
        )
    ):
        raise TableAuditError(f"{label} statistic is a formula")
    cell_type = _tag_attributes(span.start_tag, label=label).get("t")
    if cell_type in {None, "n"}:
        values = list(
            _xml_element_spans(
                payload,
                "v",
                label=label,
                start=span.content_start,
                end=span.content_end,
            )
        )
        if len(values) != 1:
            raise TableAuditError(f"{label} has no numeric-presence payload")
        # Cell type plus exactly one <v> element is the entire observation.
        # Its lexical content is deliberately neither sliced, parsed, nor
        # retained; the exact source SHA binds the frozen public workbook.
        return StringToken("NUMERIC_PRESENT", "NUMERIC_PRESENT")
    if cell_type in {"s", "inlineStr", "str"}:
        return _purpose_token(
            _string_token(span, payload, label=label), "NA", locator_contract
        )
    raise TableAuditError(f"{label} statistic type is outside numeric/exact-NA")


def _record_shared_usage(
    token: StringToken,
    usage: dict[int, set[str]],
) -> None:
    if token.kind.startswith("SHARED_"):
        usage[int(token.value)].add(token.kind.removeprefix("SHARED_"))


def _extract_primary_tokens(
    payload: bytes,
    locator_contract: Mapping[str, Any],
) -> tuple[list[StringToken], list[PrimaryTokenRow], dict[int, set[str]]]:
    header_by_column: dict[str, StringToken] = {}
    rows: list[PrimaryTokenRow] = []
    shared_usage: dict[int, set[str]] = defaultdict(set)
    _assert_xml_balanced(payload, label="primary worksheet")
    official_x14ac_namespace = _worksheet_has_official_x14ac_namespace(
        payload,
        label="primary worksheet",
    )
    for row_span in _xml_element_spans(payload, "row", label="primary worksheet"):
        current_row = _closed_worksheet_row_number(
            row_span.start_tag,
            label="primary worksheet row",
            official_x14ac_namespace=official_x14ac_namespace,
            expected_span="1:7",
        )
        current_cells: dict[str, StringToken] = {}
        seen_columns: set[str] = set()
        for cell_span in _xml_element_spans(
            payload,
            "c",
            label="primary worksheet row",
            start=row_span.content_start,
            end=row_span.content_end,
        ):
            column, row_number = _cell_ref(cell_span.start_tag)
            if row_number != current_row:
                raise TableAuditError("primary worksheet cell/row reference differs")
            if column in seen_columns:
                raise TableAuditError("primary worksheet duplicates a cell")
            seen_columns.add(column)
            token: StringToken | None = None
            if current_row == 1 and column in {"A", "B", "C", "D", "E", "F", "G"}:
                token = _purpose_token(
                    _string_token(
                        cell_span,
                        payload,
                        label=f"primary header {column}",
                    ),
                    "HEADER",
                    locator_contract,
                )
                header_by_column[column] = token
            elif current_row > 1 and column == "A":
                token = _purpose_token(
                    _string_token(
                        cell_span,
                        payload,
                        label="primary selective locator",
                    ),
                    "LOCATOR",
                    locator_contract,
                )
                current_cells[column] = token
            elif current_row > 1 and column == "C":
                token = _purpose_token(
                    _string_token(
                        cell_span,
                        payload,
                        label="primary selective comparison",
                    ),
                    "COMPARISON",
                    locator_contract,
                )
                current_cells[column] = token
            elif current_row > 1 and column in {"D", "E", "F"}:
                token = _statistic_token(
                    cell_span,
                    payload,
                    label=f"primary selective statistic {column}",
                    locator_contract=locator_contract,
                )
                current_cells[column] = token
            # B (Gene) and G (Translation) are never sliced into an XML
            # element, decoded, inspected, counted, or retained.
            if token is not None:
                _record_shared_usage(token, shared_usage)
        if seen_columns != {"A", "B", "C", "D", "E", "F", "G"}:
            raise TableAuditError("primary worksheet row width differs")
        if current_row > 1:
            if set(current_cells) != {"A", "C", "D", "E", "F"}:
                raise TableAuditError("primary selective cell set differs")
            rows.append(
                PrimaryTokenRow(
                    row_number=current_row,
                    locator=current_cells["A"],
                    comparison=current_cells["C"],
                    statistic_states=(
                        current_cells["D"],
                        current_cells["E"],
                        current_cells["F"],
                    ),
                )
            )
    if set(header_by_column) != {"A", "B", "C", "D", "E", "F", "G"}:
        raise TableAuditError("primary worksheet header width differs")
    if [row.row_number for row in rows] != list(range(2, len(rows) + 2)):
        raise TableAuditError("primary worksheet data rows are not contiguous")
    return (
        [header_by_column[column] for column in ("A", "B", "C", "D", "E", "F", "G")],
        rows,
        shared_usage,
    )


def _worksheet_dimension(payload: bytes, *, label: str) -> str:
    for tag in _iter_xml_tags(payload, label=label):
        if tag.local_name == "dimension" and tag.kind in {"start", "empty"}:
            reference = _tag_attributes(tag, label=label).get("ref")
            if not isinstance(reference, str):
                raise TableAuditError(f"{label} dimension is absent")
            return reference
        if tag.local_name == "sheetData" and tag.kind == "start":
            break
    raise TableAuditError(f"{label} dimension is absent")


def _extract_control_header(
    payload: bytes,
    width: int,
    locator_contract: Mapping[str, Any],
) -> tuple[list[StringToken], dict[int, set[str]]]:
    columns = [
        chr(ord("A") + index) if index < 26 else None
        for index in range(width)
    ]
    if any(value is None for value in columns):
        raise TableAuditError("control header width exceeds supported selective range")
    expected = {str(value) for value in columns}
    header: dict[str, StringToken] = {}
    shared_usage: dict[int, set[str]] = defaultdict(set)
    official_x14ac_namespace = _worksheet_has_official_x14ac_namespace(
        payload,
        label="control worksheet prefix",
    )
    row_spans = list(_xml_element_spans(payload, "row", label="control worksheet prefix"))
    if len(row_spans) != 1 or _closed_worksheet_row_number(
        row_spans[0].start_tag,
        label="control header row",
        official_x14ac_namespace=official_x14ac_namespace,
        expected_span=f"1:{width}",
    ) != 1:
        raise TableAuditError("control worksheet header row differs")
    for cell_span in _xml_element_spans(
        payload,
        "c",
        label="control header row",
        start=row_spans[0].content_start,
        end=row_spans[0].content_end,
    ):
        column, row_number = _cell_ref(cell_span.start_tag)
        if row_number != 1 or column in header:
            raise TableAuditError("control header cell identity differs")
        token = _purpose_token(
            _string_token(
                cell_span,
                payload,
                label=f"control header {column}",
            ),
            "HEADER",
            locator_contract,
        )
        header[column] = token
        _record_shared_usage(token, shared_usage)
    if set(header) != expected:
        raise TableAuditError("control worksheet header width differs")
    return [header[str(value)] for value in columns], shared_usage


def _shared_string_values(
    payload: bytes,
    wanted: Mapping[int, set[str]],
    locator_contract: Mapping[str, Any],
    *,
    missing_token: str,
) -> dict[int, dict[str, str | bytes]]:
    values: dict[int, dict[str, str | bytes]] = {}
    _assert_xml_balanced(payload, label="sharedStrings.xml")
    for index, span in enumerate(
        _xml_element_spans(payload, "si", label="sharedStrings.xml")
    ):
        if index not in wanted:
            continue
        try:
            element = ET.fromstring(payload[span.start : span.end])
        except ET.ParseError as exc:
            raise TableAuditError("selected shared string is invalid XML") from exc
        text = "".join(
            child.text or ""
            for child in element.iter()
            if _local_name(child.tag) == "t"
        )
        converted: dict[str, str | bytes] = {}
        for purpose in wanted[index]:
            if purpose == "LOCATOR":
                converted[purpose] = _join_key_digest(text, locator_contract)
            elif purpose == "NA":
                if text != missing_token:
                    raise TableAuditError(
                        "shared statistic string is not the exact NA token"
                    )
                converted[purpose] = "EXACT_NA"
            elif purpose in {"HEADER", "COMPARISON"}:
                if text == "":
                    raise TableAuditError("required selective shared string is empty")
                converted[purpose] = text
            else:
                raise TableAuditError("unknown shared-string purpose")
        values[index] = converted
        del text
    if set(values) != set(wanted):
        raise TableAuditError("a required selective shared string is absent")
    return values


def _resolve_purpose_string(
    token: StringToken,
    shared: Mapping[int, Mapping[str, str | bytes]],
    *,
    purpose: str,
    label: str,
) -> str:
    if token.kind == f"INLINE_{purpose}":
        value = token.value
    elif token.kind == f"SHARED_{purpose}":
        value = shared.get(int(token.value), {}).get(purpose)
    else:
        raise TableAuditError(f"{label} is not a {purpose.casefold()} token")
    if not isinstance(value, str) or value == "":
        raise TableAuditError(f"{label} is empty or unresolved")
    return value


def _resolve_locator_digest(
    token: StringToken,
    shared: Mapping[int, Mapping[str, str | bytes]],
) -> bytes:
    if token.kind == "JOIN_DIGEST":
        value = token.value
    elif token.kind == "SHARED_LOCATOR":
        value = shared.get(int(token.value), {}).get("LOCATOR")
    else:
        raise TableAuditError("primary locator token is not a join digest")
    if not isinstance(value, bytes) or len(value) != 32:
        raise TableAuditError("primary locator digest is unresolved")
    return value


def _resolve_statistic_state(
    token: StringToken,
    shared: Mapping[int, Mapping[str, str | bytes]],
    *,
    missing_token: str,
) -> str:
    if token.kind == "NUMERIC_PRESENT":
        return "NUMERIC_PRESENT"
    if token.kind == "INLINE_NA":
        value = token.value
        if value != missing_token:
            raise TableAuditError("statistic string is not the exact NA token")
        return "EXACT_NA"
    if token.kind == "SHARED_NA":
        value = shared.get(int(token.value), {}).get("NA")
    else:
        raise TableAuditError("statistic state token is invalid")
    if value != "EXACT_NA":
        raise TableAuditError("statistic string is not the exact NA token")
    return "EXACT_NA"


def audit_table_s3(
    payload: bytes,
    spec: Mapping[str, Any],
    locator_contract: Mapping[str, Any],
) -> S3Audit:
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload), "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise TableAuditError("Table S3 is not a valid XLSX archive") from exc
    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise TableAuditError("Table S3 XLSX has duplicate members")
        total_uncompressed = 0
        for info in infos:
            pure = PurePosixPath(info.filename)
            if pure.is_absolute() or ".." in pure.parts or "\\" in info.filename:
                raise TableAuditError("Table S3 XLSX member path is unsafe")
            total_uncompressed += info.file_size
            if info.file_size > 64 * 1024 * 1024:
                raise TableAuditError("Table S3 XLSX member is oversized")
        if total_uncompressed > 128 * 1024 * 1024:
            raise TableAuditError("Table S3 XLSX expansion is oversized")
        lowered = [name.casefold() for name in names]
        if any(
            name.endswith("vbaproject.bin")
            or name.startswith("xl/externallinks/")
            or name.startswith("xl/embeddings/")
            or name.startswith("xl/oleobjects/")
            or name == "xl/connections.xml"
            for name in lowered
        ):
            raise TableAuditError("Table S3 XLSX contains active/external content")
        workbook = _xlsx_read_member(archive, "xl/workbook.xml")
        relationships = _xlsx_read_member(archive, "xl/_rels/workbook.xml.rels")
        sheets = _workbook_sheet_paths(workbook, relationships)
        if [name for name, _path in sheets] != spec["exact_sheet_names"]:
            raise TableAuditError("Table S3 sheet names/order differ")
        paths = dict(sheets)
        primary_xml = _xlsx_read_member(archive, paths[spec["primary_sheet_name"]])
        control_xml = _xlsx_read_through_first_row(
            archive, paths[spec["control_sheet_name"]]
        )
        primary_header_tokens, row_tokens, wanted = _extract_primary_tokens(
            primary_xml, locator_contract
        )
        control_header_tokens, control_wanted = _extract_control_header(
            control_xml, len(spec["control_exact_header"]), locator_contract
        )
        for index, purposes in control_wanted.items():
            wanted[index].update(purposes)
        if wanted:
            shared_payload = _xlsx_read_member(archive, "xl/sharedStrings.xml")
            shared = _shared_string_values(
                shared_payload,
                wanted,
                locator_contract,
                missing_token=spec["statistics_missing_token"],
            )
        else:
            shared = {}

    primary_header = [
        _resolve_purpose_string(
            token, shared, purpose="HEADER", label="primary header"
        )
        for token in primary_header_tokens
    ]
    if primary_header != spec["primary_exact_header"]:
        raise TableAuditError("Table S3 primary header differs")
    if _canonical_header_sha256(primary_header) != spec["primary_header_sha256"]:
        raise TableAuditError("Table S3 primary header hash differs")
    control_header = [
        _resolve_purpose_string(
            token, shared, purpose="HEADER", label="control header"
        )
        for token in control_header_tokens
    ]
    if control_header != spec["control_exact_header"]:
        raise TableAuditError("Table S3 control header differs")
    if _canonical_header_sha256(control_header) != spec["control_header_sha256"]:
        raise TableAuditError("Table S3 control header hash differs")
    expected_control_dimension = (
        f"A1:M{int(spec['control_data_row_count']) + 1}"
    )
    if _worksheet_dimension(control_xml, label="control worksheet") != expected_control_dimension:
        raise TableAuditError("Table S3 opaque control dimensions differ")

    if len(row_tokens) != spec["primary_data_row_count"]:
        raise TableAuditError("Table S3 primary data row count differs")
    source_digest = bytes.fromhex(sha256(payload))
    sheet_digest = hashlib.sha256(
        spec["primary_sheet_name"].encode("utf-8")
    ).digest()
    comparisons = set(spec["comparison_row_counts"])
    per_pair: dict[bytes, set[str]] = defaultdict(set)
    finite_by_comparison: dict[str, set[bytes]] = defaultdict(set)
    comparison_counts: Counter[str] = Counter()
    finite_counts: Counter[str] = Counter()
    na_counts: Counter[str] = Counter()
    primary_finite_row_digests: dict[bytes, bytes] = {}
    for row in row_tokens:
        join_key = _resolve_locator_digest(row.locator, shared)
        comparison = _resolve_purpose_string(
            row.comparison,
            shared,
            purpose="COMPARISON",
            label="primary comparison",
        )
        if comparison not in comparisons:
            raise TableAuditError("Table S3 comparison is outside the closed allowlist")
        if comparison in per_pair[join_key]:
            raise TableAuditError("Table S3 duplicates a pair-comparison locator")
        per_pair[join_key].add(comparison)
        states = tuple(
            _resolve_statistic_state(
                token,
                shared,
                missing_token=spec["statistics_missing_token"],
            )
            for token in row.statistic_states
        )
        if states == ("NUMERIC_PRESENT",) * 3:
            finite_counts[comparison] += 1
            finite_by_comparison[comparison].add(join_key)
            state = "ALL_NUMERIC_PRESENT"
        elif states == ("EXACT_NA",) * 3:
            na_counts[comparison] += 1
            state = "ALL_EXACT_NA"
        else:
            raise TableAuditError("Table S3 statistics are mixed numeric/NA")
        comparison_counts[comparison] += 1
        physical_row_digest = _domain_hash(
            _configured_domain(locator_contract, "s3_physical_row_domain"),
            (
                source_digest,
                sheet_digest,
                _u64(row.row_number),
                join_key,
                comparison.encode("utf-8"),
                state.encode("ascii"),
            ),
        )
        if comparison == spec.get("primary_comparison", "TotalPoly:RNA") and state == "ALL_NUMERIC_PRESENT":
            primary_finite_row_digests[join_key] = physical_row_digest

    if len(per_pair) != spec["pair_key_count"]:
        raise TableAuditError("Table S3 pair-key count differs")
    if any(value != comparisons for value in per_pair.values()):
        raise TableAuditError("Table S3 pair membership is not comparison-complete")
    observed_comparison_counts = {
        comparison: comparison_counts[comparison]
        for comparison in sorted(comparisons)
    }
    observed_finite_counts = {
        comparison: finite_counts[comparison]
        for comparison in sorted(comparisons)
    }
    observed_na_counts = {
        comparison: na_counts[comparison]
        for comparison in sorted(comparisons)
    }
    if observed_comparison_counts != spec["comparison_row_counts"]:
        raise TableAuditError("Table S3 comparison row counts differ")
    if observed_finite_counts != spec["finite_statistic_rows"]:
        raise TableAuditError("Table S3 finite statistic counts differ")
    if observed_na_counts != spec["na_statistic_rows"]:
        raise TableAuditError("Table S3 NA statistic counts differ")
    high = finite_by_comparison["HighPoly:RNA"]
    primary = finite_by_comparison["TotalPoly:RNA"]
    partition = {
        "both_comparisons_finite_pair_count": len(high & primary),
        "primary_only_finite_pair_count": len(primary - high),
        "secondary_only_finite_pair_count": len(high - primary),
        "neither_comparison_finite_pair_count": len(set(per_pair) - (high | primary)),
    }
    if any(partition[key] != spec[key] for key in partition):
        raise TableAuditError("Table S3 finite/NA completeness partition differs")
    return S3Audit(
        pair_keys=frozenset(per_pair),
        finite_pair_keys={
            comparison: frozenset(keys)
            for comparison, keys in finite_by_comparison.items()
        },
        primary_finite_row_digests=primary_finite_row_digests,
        aggregates={
            "primary_data_row_count": len(row_tokens),
            "pair_key_count": len(per_pair),
            "comparison_row_counts": observed_comparison_counts,
            "finite_statistic_rows": observed_finite_counts,
            "na_statistic_rows": observed_na_counts,
            **partition,
            "significance_values_decoded": False,
            "significance_used_for_membership": False,
            "gene_values_decoded": False,
            "numeric_effect_p_fdr_values_retained": False,
            "control_data_cells_read": False,
        },
    )


@dataclass(frozen=True)
class AuditSummary:
    processed_pair_count: int
    canonical_record_count: int
    s2_only_pair_count: int
    s3_only_pair_count: int
    locator_merkle_root_sha256: str
    table_s2_sha256: str
    table_s3_sha256: str


def _locator_digest(join_key: bytes, comparison: str, contract: Mapping[str, Any]) -> bytes:
    domain = _configured_domain(contract, "locator_domain")
    fields = (
        DATASET_ID.encode("ascii"),
        join_key,
        comparison.encode("utf-8"),
    )
    return _domain_hash(domain, fields)


def _merkle_root(leaves: Sequence[bytes], contract: Mapping[str, Any]) -> bytes:
    if not leaves:
        raise TableAuditError("canonical lineage Merkle tree has no leaves")
    leaf_domain = _configured_domain(contract, "merkle_leaf_domain")
    node_domain = _configured_domain(contract, "merkle_node_domain")
    root_domain = _configured_domain(contract, "merkle_root_domain")
    level = sorted(_domain_hash(leaf_domain, (leaf,)) for leaf in leaves)
    while len(level) > 1:
        next_level: list[bytes] = []
        for index in range(0, len(level), 2):
            left = level[index]
            right = level[index + 1] if index + 1 < len(level) else left
            next_level.append(_domain_hash(node_domain, (left, right)))
        level = next_level
    return _domain_hash(root_domain, (_u64(len(leaves)), level[0]))


def audit_tables(
    table_s2_payload: bytes,
    table_s3_payload: bytes,
    locator_contract: Mapping[str, Any],
    *,
    locator_digest: LocatorDigest = _locator_digest,
) -> AuditSummary:
    """Reconcile exact S2/S3 membership and build no-output internal lineage."""

    s2 = audit_table_s2(
        table_s2_payload,
        locator_contract["table_s2"],
        locator_contract,
    )
    s3_spec = copy.deepcopy(dict(locator_contract["table_s3"]))
    s3_spec["primary_comparison"] = locator_contract["primary_comparison"]
    s3 = audit_table_s3(table_s3_payload, s3_spec, locator_contract)
    s2_only = s2.pair_keys - s3.pair_keys
    s3_only = s3.pair_keys - s2.pair_keys
    joined = s2.pair_keys & s3.pair_keys
    join = locator_contract["join"]
    observed_join = {
        "table_s2_pair_count": len(s2.pair_keys),
        "table_s3_pair_count": len(s3.pair_keys),
        "joined_pair_count": len(joined),
        "table_s2_absent_from_table_s3_count": len(s2_only),
        "table_s3_not_in_table_s2_count": len(s3_only),
    }
    for key, value in observed_join.items():
        if value != join[key]:
            raise TableAuditError(f"Table S2/S3 join count differs: {key}")
    if join["s3_pair_set_must_be_subset_of_s2"] is not True or s3_only:
        raise TableAuditError("Table S3 pair set is not a subset of Table S2")
    if join["each_s3_pair_must_join_exactly_one_s2_pair"] is not True:
        raise TableAuditError("exact-one S2 join rule is not frozen")

    primary = locator_contract["primary_comparison"]
    finite_primary = s3.finite_pair_keys.get(primary, frozenset())
    if not finite_primary <= joined:
        raise TableAuditError("finite primary membership is outside the exact join")
    if len(s3.pair_keys) != locator_contract["processed_pair_count"]:
        raise TableAuditError("processed pair count differs")
    if len(finite_primary) != locator_contract["canonical_record_count"]:
        raise TableAuditError("canonical finite-primary record count differs")
    if len(s3.pair_keys) - len(finite_primary) != locator_contract["primary_na_pair_count"]:
        raise TableAuditError("primary NA attrition count differs")

    source_s2_sha = sha256(table_s2_payload)
    source_s3_sha = sha256(table_s3_payload)
    pair_lineage_domain = _configured_domain(locator_contract, "pair_lineage_domain")
    locators: set[bytes] = set()
    leaves: list[bytes] = []
    for join_key in sorted(finite_primary):
        locator = locator_digest(join_key, primary, locator_contract)
        if type(locator) is not bytes or len(locator) != 32:
            raise TableAuditError("canonical locator digest is not SHA-256 bytes")
        if locator in locators:
            raise TableAuditError("duplicate canonical locator digest")
        locators.add(locator)
        s2_pair_digest = s2.pair_digests.get(join_key)
        s3_row_digest = s3.primary_finite_row_digests.get(join_key)
        if s2_pair_digest is None or s3_row_digest is None:
            raise TableAuditError("canonical locator lacks exact S2/S3 lineage")
        leaves.append(
            _domain_hash(
                pair_lineage_domain,
                (
                    locator,
                    s2_pair_digest,
                    s3_row_digest,
                    bytes.fromhex(source_s2_sha),
                    bytes.fromhex(source_s3_sha),
                ),
            )
        )
    if len(locators) != len(finite_primary):
        raise TableAuditError("canonical locator cardinality differs")
    root = _merkle_root(leaves, locator_contract)
    return AuditSummary(
        processed_pair_count=len(s3.pair_keys),
        canonical_record_count=len(finite_primary),
        s2_only_pair_count=len(s2_only),
        s3_only_pair_count=len(s3_only),
        locator_merkle_root_sha256=root.hex(),
        table_s2_sha256=source_s2_sha,
        table_s3_sha256=source_s3_sha,
    )


def _assert_no_forbidden_output(
    value: Any,
    forbidden_keys: set[str],
    *,
    label: str,
    _path: tuple[str, ...] = (),
) -> None:
    if type(value) is dict:
        for key, child in value.items():
            if key.casefold() in forbidden_keys:
                raise GateProducerError(f"{label} contains forbidden output key: {key}")
            child_path = (*_path, key)
            if child_path == (
                "facts",
                "locator_lineage_merkle_root_sha256",
            ) and type(child) is str and HEX64.fullmatch(child):
                # A SHA-256 digest can coincidentally contain only DNA alphabet
                # characters.  The exception is limited to this exact closed,
                # already type-validated aggregate fact path.
                continue
            _assert_no_forbidden_output(
                child,
                forbidden_keys,
                label=label,
                _path=child_path,
            )
    elif type(value) is list:
        for index, child in enumerate(value):
            _assert_no_forbidden_output(
                child,
                forbidden_keys,
                label=label,
                _path=(*_path, str(index)),
            )
    elif type(value) is str and DNA_LIKE.fullmatch(value):
        raise GateProducerError(f"{label} contains a sequence-like scalar")


def build_gate_record(
    config: Mapping[str, Any],
    summary: AuditSummary,
) -> dict[str, Any]:
    locator = config["locator_contract"]
    source_members = {
        member["asset_id"]: member for member in config["source_contract"]["members"]
    }
    if summary.table_s2_sha256 != source_members["PMC10540565_TABLE_S2"]["sha256"]:
        raise InputIntegrityError("Table S2 summary SHA differs from source authority")
    if summary.table_s3_sha256 != source_members["PMC10540565_TABLE_S3"]["sha256"]:
        raise InputIntegrityError("Table S3 summary SHA differs from source authority")
    if summary.processed_pair_count != locator["processed_pair_count"]:
        raise TableAuditError("gate processed pair count differs")
    if summary.canonical_record_count != locator["canonical_record_count"]:
        raise TableAuditError("gate canonical record count differs")
    if summary.s2_only_pair_count != locator["join"][
        "table_s2_absent_from_table_s3_count"
    ] or summary.s3_only_pair_count != 0:
        raise TableAuditError("gate join attrition differs")
    binding = config["implementation_binding"]
    predecessor = config["authority_inputs"]["required_predecessor_authority"]
    record = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "record_type": EVIDENCE_RECORD_TYPE,
        "contract_id": CONTRACT_ID,
        "decision_id": DECISION_ID,
        "dataset_id": DATASET_ID,
        "gate_id": GATE_ID,
        "status": "PASS",
        "accepted": True,
        "aggregate_only": True,
        "privacy": copy.deepcopy(config["output_contract"]["privacy"]),
        "provenance": {
            "producer_protocol_id": PROTOCOL_ID,
            "producer_commit": binding["implementation_commit"],
            "producer_script_sha256": binding["implementation_script_sha256"],
            "source_bundle_id": predecessor["bundle_id"],
            "source_bundle_root_or_target_sha256": predecessor[
                "terminal_marker_final_output_target_sha256"
            ],
            "predecessor_authority": copy.deepcopy(predecessor),
            "acceptance_authority": copy.deepcopy(
                config["authority_inputs"]["acceptance_authority"]
            ),
        },
        "facts": {
            "deterministic_row_locator_frozen": True,
            "table_s2_hash_bound": True,
            "table_s3_hash_bound": True,
            "s2_s3_join_rule_frozen": True,
            "multi_asset_lineage_closed": True,
            "canonical_record_count": locator["canonical_record_count"],
            "processed_pair_count": locator["processed_pair_count"],
            "raw_replay_role": RAW_REPLAY_ROLE,
            "raw_replay_status": "NOT_RUN",
            "independent_raw_reproduction_claimed": False,
            "locator_lineage_commitment_algorithm": (
                LOCATOR_LINEAGE_COMMITMENT_ALGORITHM
            ),
            "locator_lineage_merkle_root_sha256": (
                summary.locator_merkle_root_sha256
            ),
        },
        "unknown_fields": [],
        "reason_codes": [],
    }
    validate_gate_record(record, config)
    return record


def validate_gate_record(record: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    _expect_exact_keys(record, GATE_RECORD_KEYS, label="lineage gate record")
    for key, expected in {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "record_type": EVIDENCE_RECORD_TYPE,
        "contract_id": CONTRACT_ID,
        "decision_id": DECISION_ID,
        "dataset_id": DATASET_ID,
        "gate_id": GATE_ID,
        "status": "PASS",
        "accepted": True,
        "aggregate_only": True,
        "unknown_fields": [],
        "reason_codes": [],
    }.items():
        _expect_exact(record[key], expected, label=f"gate {key}")
    privacy = _expect_exact_keys(record["privacy"], PRIVACY_KEYS, label="gate privacy")
    if any(privacy[key] is not False for key in PRIVACY_KEYS):
        raise GateProducerError("gate privacy flags are not all false")
    provenance = _expect_exact_keys(
        record["provenance"], PROVENANCE_KEYS, label="gate provenance"
    )
    binding = config["implementation_binding"]
    predecessor = config["authority_inputs"]["required_predecessor_authority"]
    expected_provenance = {
        "producer_protocol_id": PROTOCOL_ID,
        "producer_commit": binding["implementation_commit"],
        "producer_script_sha256": binding["implementation_script_sha256"],
        "source_bundle_id": predecessor["bundle_id"],
        "source_bundle_root_or_target_sha256": predecessor[
            "terminal_marker_final_output_target_sha256"
        ],
        "predecessor_authority": predecessor,
        "acceptance_authority": config["authority_inputs"]["acceptance_authority"],
    }
    if dict(provenance) != expected_provenance:
        raise GateProducerError("gate provenance differs from exact authority")
    if HEX40.fullmatch(str(provenance["producer_commit"])) is None:
        raise GateProducerError("gate producer commit is unbound")
    if HEX64.fullmatch(str(provenance["producer_script_sha256"])) is None:
        raise GateProducerError("gate producer script SHA is unbound")
    facts = _expect_exact_keys(record["facts"], FACT_KEYS, label="gate facts")
    expected_facts = {
        "deterministic_row_locator_frozen": True,
        "table_s2_hash_bound": True,
        "table_s3_hash_bound": True,
        "s2_s3_join_rule_frozen": True,
        "multi_asset_lineage_closed": True,
        "canonical_record_count": 6547,
        "processed_pair_count": 6772,
        "raw_replay_role": RAW_REPLAY_ROLE,
        "raw_replay_status": "NOT_RUN",
        "independent_raw_reproduction_claimed": False,
        "locator_lineage_commitment_algorithm": (
            LOCATOR_LINEAGE_COMMITMENT_ALGORITHM
        ),
    }
    root = facts.get("locator_lineage_merkle_root_sha256")
    if type(root) is not str or HEX64.fullmatch(root) is None:
        raise GateProducerError("gate lineage Merkle root is not lowercase SHA-256")
    expected_facts["locator_lineage_merkle_root_sha256"] = root
    if dict(facts) != expected_facts:
        raise GateProducerError("gate PASS facts differ from exact DEC019 facts")
    forbidden = {
        str(key).casefold() for key in config["output_contract"]["forbidden_output_keys"]
    }
    _assert_no_forbidden_output(record, forbidden, label="lineage gate")


def _consume_gate_with_v3_assets(
    payload: bytes,
    config: Mapping[str, Any],
    *,
    v3_config_payload: bytes,
    v3_script_payload: bytes,
) -> dict[str, Any]:
    """Run the exact hash-bound consumer validator over final gate bytes."""

    v3_config = strict_json(v3_config_payload, label="consumer v3 config")
    _validate_v3_compatibility(v3_config_payload, config)
    slots = v3_config["evidence_contract"]["slots"]
    matching = [slot for slot in slots if slot.get("slot_id") == GATE_ID]
    if len(matching) != 1 or matching[0].get("allowed_basename") != OUTPUT_BASENAME:
        raise BindingError("consumer v3 exact lineage slot is absent")
    module_name = f"_g200_dec019_consumer_{sha256(v3_script_payload)[:16]}"
    module = types.ModuleType(module_name)
    module.__file__ = config["authority_inputs"]["dec019_gse200304_v3"][
        "implementation_script_path"
    ]
    sys.modules[module_name] = module
    try:
        compiled = compile(
            v3_script_payload,
            module.__file__,
            "exec",
            dont_inherit=True,
        )
        exec(compiled, module.__dict__)
        validator = getattr(module, "_validate_gate_record", None)
        if not callable(validator):
            raise BindingError("hash-bound consumer validator is absent")
        consumed = validator(payload, matching[0], v3_config)
    except GateProducerError:
        raise
    except Exception as exc:
        raise GateProducerError("hash-bound v3 consumer rejected the gate") from exc
    finally:
        sys.modules.pop(module_name, None)
    expected = strict_json(payload, label="producer final gate for consumer")
    if type(consumed) is not dict or consumed != expected:
        raise GateProducerError("hash-bound consumer returned a non-exact record")
    return consumed


def consume_gate_with_current_v3(
    payload: bytes,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    repo = Path(config["repository_authority"]["production_repo_root"])
    head = _git(repo, "rev-parse", "HEAD")
    spec = config["authority_inputs"]["dec019_gse200304_v3"]
    v3_config_payload = _verify_current_file(
        repo, head, spec["config_path"], spec["config_sha256"]
    )
    v3_script_payload = _verify_current_file(
        repo,
        head,
        spec["implementation_script_path"],
        spec["implementation_script_sha256"],
    )
    _verify_current_file(
        repo,
        head,
        spec["implementation_test_path"],
        spec["implementation_test_sha256"],
    )
    return _consume_gate_with_v3_assets(
        payload,
        config,
        v3_config_payload=v3_config_payload,
        v3_script_payload=v3_script_payload,
    )


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise PublicationError("short write while staging gate")
        offset += written


def _safe_read_output_at(directory_fd: int, name: str) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise AmbiguousPublicationError("existing gate cannot be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise AmbiguousPublicationError(
                "existing gate is not a single-link regular file"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise AmbiguousPublicationError("existing gate changed during same-FD read")
        try:
            visible = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as exc:
            raise AmbiguousPublicationError("existing gate disappeared") from exc
        if _identity(visible) != _identity(after):
            raise AmbiguousPublicationError("existing gate path was replaced")
        return b"".join(chunks), after
    finally:
        os.close(descriptor)


def _stat_at_optional(directory_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise PublicationStateUnverifiedError(
            f"PUBLICATION_STATE_UNVERIFIED: cannot stat {name}"
        ) from exc


def _same_inode(value: os.stat_result, expected: tuple[int, int]) -> bool:
    return (value.st_dev, value.st_ino) == expected


def _assert_gate_directory_membership(
    directory_fd: int,
    *,
    expected: set[str],
) -> None:
    observed = set(os.listdir(directory_fd))
    if observed != expected:
        raise AmbiguousPublicationError(
            "gate output directory membership differs from the single-file contract"
        )


def _fresh_verify_canonical_gate(
    output_directory: Path,
    expected_directory_chain: DirectoryIdentityChain,
    *,
    basename: str,
    expected_file: os.stat_result,
    expected_payload: bytes,
) -> None:
    """Linearize acceptance on a fresh canonical root-to-leaf reopen.

    Retained-descriptor membership and byte checks must already have completed.
    This helper then reopens the canonical path twice: once for an independent
    same-FD byte snapshot, and once as the terminal root/directory/file identity
    comparison immediately before the caller returns success.
    """

    def open_exact_chain() -> int:
        descriptor, observed_chain = _open_directory_root_to_leaf_with_chain(
            output_directory,
            label="fresh canonical gate directory",
        )
        if observed_chain != expected_directory_chain:
            os.close(descriptor)
            raise PublicationStateUnverifiedError(
                "PUBLICATION_STATE_UNVERIFIED: fresh canonical directory chain differs"
            )
        return descriptor

    try:
        fresh_fd = open_exact_chain()
        try:
            if set(os.listdir(fresh_fd)) != {basename}:
                raise PublicationStateUnverifiedError(
                    "PUBLICATION_STATE_UNVERIFIED: fresh canonical membership differs"
                )
            observed, fresh_file = _safe_read_output_at(fresh_fd, basename)
            if observed != expected_payload or _identity(fresh_file) != _identity(
                expected_file
            ):
                raise PublicationStateUnverifiedError(
                    "PUBLICATION_STATE_UNVERIFIED: fresh canonical gate bytes/identity differ"
                )
        finally:
            os.close(fresh_fd)

        terminal_fd = open_exact_chain()
        try:
            terminal_file = os.stat(
                basename,
                dir_fd=terminal_fd,
                follow_symlinks=False,
            )
            if _identity(terminal_file) != _identity(expected_file):
                raise PublicationStateUnverifiedError(
                    "PUBLICATION_STATE_UNVERIFIED: terminal canonical gate identity differs"
                )
        finally:
            os.close(terminal_fd)
    except PublicationStateUnverifiedError:
        raise
    except Exception as verification_exc:
        raise PublicationStateUnverifiedError(
            "PUBLICATION_STATE_UNVERIFIED: fresh canonical gate verification failed"
        ) from verification_exc


def _recover_publication_after_error(
    *,
    original: BaseException,
    directory_fd: int,
    output_directory: Path,
    directory_identity: tuple[int, int],
    directory_chain: DirectoryIdentityChain,
    temp_name: str,
    basename: str,
    staged_inode: tuple[int, int],
    payload: bytes,
) -> str:
    """Resolve post-staging truth without mutating a renamed-out directory."""

    try:
        _assert_directory_identity(
            directory_fd,
            output_directory,
            directory_identity,
            label="output directory",
        )
    except Exception as recovery_exc:
        raise PublicationStateUnverifiedError(
            "PUBLICATION_STATE_UNVERIFIED: canonical output directory identity was lost; "
            "retained-directory cleanup was intentionally not attempted"
        ) from recovery_exc

    final_stat = _stat_at_optional(directory_fd, basename)
    temp_stat = _stat_at_optional(directory_fd, temp_name)
    if final_stat is not None and _same_inode(final_stat, staged_inode):
        try:
            if temp_stat is not None:
                if (
                    not _same_inode(temp_stat, staged_inode)
                    or not stat.S_ISREG(temp_stat.st_mode)
                    or temp_stat.st_nlink != 2
                    or final_stat.st_nlink != 2
                ):
                    raise PublicationStateUnverifiedError(
                        "PUBLICATION_STATE_UNVERIFIED: owned final/temp link geometry differs"
                    )
                _assert_directory_identity(
                    directory_fd,
                    output_directory,
                    directory_identity,
                    label="output directory",
                )
                current_temp = _stat_at_optional(directory_fd, temp_name)
                current_final = _stat_at_optional(directory_fd, basename)
                if (
                    current_temp is None
                    or current_final is None
                    or not _same_inode(current_temp, staged_inode)
                    or not _same_inode(current_final, staged_inode)
                    or current_temp.st_nlink != 2
                    or current_final.st_nlink != 2
                ):
                    raise PublicationStateUnverifiedError(
                        "PUBLICATION_STATE_UNVERIFIED: owned links changed before recovery cleanup"
                    )
                os.unlink(temp_name, dir_fd=directory_fd)
                os.fsync(directory_fd)
            elif final_stat.st_nlink != 1:
                raise PublicationStateUnverifiedError(
                    "PUBLICATION_STATE_UNVERIFIED: owned final has unexplained link count"
                )
            _assert_directory_identity(
                directory_fd,
                output_directory,
                directory_identity,
                label="output directory",
            )
            observed, final_verified = _safe_read_output_at(directory_fd, basename)
            if observed != payload or not _same_inode(final_verified, staged_inode):
                raise PublicationStateUnverifiedError(
                    "PUBLICATION_STATE_UNVERIFIED: owned final bytes/identity differ"
                )
            os.fsync(directory_fd)
            _assert_directory_identity(
                directory_fd,
                output_directory,
                directory_identity,
                label="output directory",
            )
            _assert_gate_directory_membership(directory_fd, expected={basename})
            _fresh_verify_canonical_gate(
                output_directory,
                directory_chain,
                basename=basename,
                expected_file=final_verified,
                expected_payload=payload,
            )
            return "COMMITTED_EXACT_AFTER_RECOVERY"
        except PublicationStateUnverifiedError:
            raise
        except Exception as recovery_exc:
            raise PublicationStateUnverifiedError(
                "PUBLICATION_STATE_UNVERIFIED: committed-gate recovery could not be completed"
            ) from recovery_exc

    if temp_stat is not None:
        if (
            not _same_inode(temp_stat, staged_inode)
            or not stat.S_ISREG(temp_stat.st_mode)
            or temp_stat.st_nlink != 1
        ):
            raise PublicationStateUnverifiedError(
                "PUBLICATION_STATE_UNVERIFIED: temporary path is no longer the staged inode"
            ) from original
        try:
            _assert_directory_identity(
                directory_fd,
                output_directory,
                directory_identity,
                label="output directory",
            )
            current_temp = _stat_at_optional(directory_fd, temp_name)
            if (
                current_temp is None
                or not _same_inode(current_temp, staged_inode)
                or not stat.S_ISREG(current_temp.st_mode)
                or current_temp.st_nlink != 1
            ):
                raise PublicationStateUnverifiedError(
                    "PUBLICATION_STATE_UNVERIFIED: temporary path changed before cleanup"
                )
            os.unlink(temp_name, dir_fd=directory_fd)
            os.fsync(directory_fd)
            _assert_directory_identity(
                directory_fd,
                output_directory,
                directory_identity,
                label="output directory",
            )
        except PublicationStateUnverifiedError:
            raise
        except Exception as recovery_exc:
            raise PublicationStateUnverifiedError(
                "PUBLICATION_STATE_UNVERIFIED: pre-commit cleanup could not be completed"
            ) from recovery_exc

    try:
        _assert_directory_identity(
            directory_fd,
            output_directory,
            directory_identity,
            label="output directory",
        )
    except Exception as recovery_exc:
        raise PublicationStateUnverifiedError(
            "PUBLICATION_STATE_UNVERIFIED: canonical directory could not be "
            "reconfirmed after pre-commit cleanup"
        ) from recovery_exc
    if final_stat is not None:
        try:
            existing, existing_verified = _safe_read_output_at(
                directory_fd,
                basename,
            )
            _assert_directory_identity(
                directory_fd,
                output_directory,
                directory_identity,
                label="output directory",
            )
            _assert_gate_directory_membership(directory_fd, expected={basename})
        except Exception as recovery_exc:
            raise PublicationStateUnverifiedError(
                "PUBLICATION_STATE_UNVERIFIED: concurrent final cannot be verified"
            ) from recovery_exc
        if existing == payload:
            _fresh_verify_canonical_gate(
                output_directory,
                directory_chain,
                basename=basename,
                expected_file=existing_verified,
                expected_payload=payload,
            )
            return "EXISTING_EXACT_AFTER_CONCURRENT_RACE"
        raise AmbiguousPublicationError(
            "PUBLICATION_ABORTED_CONFIRMED_DIFFERENT_CONCURRENT_FINAL"
        ) from original

    try:
        if _stat_at_optional(directory_fd, basename) is not None:
            raise PublicationStateUnverifiedError(
                "PUBLICATION_STATE_UNVERIFIED: final appeared during absence confirmation"
            )
        _assert_directory_identity(
            directory_fd,
            output_directory,
            directory_identity,
            label="output directory",
        )
        _assert_gate_directory_membership(directory_fd, expected=set())
    except PublicationStateUnverifiedError:
        raise
    except Exception as recovery_exc:
        raise PublicationStateUnverifiedError(
            "PUBLICATION_STATE_UNVERIFIED: final absence could not be confirmed"
        ) from recovery_exc
    raise PublicationError("PUBLICATION_ABORTED_CONFIRMED_ABSENT") from original


def publish_single_gate(
    payload: bytes,
    output_directory: Path,
    *,
    basename: str = OUTPUT_BASENAME,
    fault: FaultInjector | None = None,
) -> str:
    """Publish one exact file with no overwrite and crash-detectable truth."""

    if PurePosixPath(basename).name != basename or basename != OUTPUT_BASENAME:
        raise ScopeViolation("gate basename differs from the v3 closed slot")
    directory_fd, directory_chain = _open_directory_root_to_leaf_with_chain(
        output_directory,
        label="output directory",
    )
    temp_name = f".{basename}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    staged_inode: tuple[int, int] | None = None
    directory_identity: tuple[int, int] | None = None
    try:
        directory_stat = os.fstat(directory_fd)
        directory_identity = (directory_stat.st_dev, directory_stat.st_ino)
        _assert_directory_identity(
            directory_fd,
            output_directory,
            directory_identity,
            label="output directory",
        )
        names = set(os.listdir(directory_fd))
        if any(name.startswith(f".{basename}.tmp.") for name in names):
            raise AmbiguousPublicationError("stale gate staging file exists")
        if names not in (set(), {basename}):
            raise AmbiguousPublicationError(
                "gate output directory contains an unrelated entry"
            )
        existing_stat = _stat_at_optional(directory_fd, basename)
        if existing_stat is not None:
            existing, existing_verified = _safe_read_output_at(
                directory_fd,
                basename,
            )
            if existing != payload:
                raise AmbiguousPublicationError("existing gate differs; overwrite forbidden")
            try:
                _assert_directory_identity(
                    directory_fd,
                    output_directory,
                    directory_identity,
                    label="output directory",
                )
                _assert_gate_directory_membership(directory_fd, expected={basename})
            except Exception as verification_exc:
                raise PublicationStateUnverifiedError(
                    "PUBLICATION_STATE_UNVERIFIED: existing exact gate parent identity "
                    "could not be reconfirmed"
                ) from verification_exc
            _fresh_verify_canonical_gate(
                output_directory,
                directory_chain,
                basename=basename,
                expected_file=existing_verified,
                expected_payload=payload,
            )
            return "EXISTING_EXACT_IDEMPOTENT"

        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            temp_fd = os.open(temp_name, flags, 0o600, dir_fd=directory_fd)
        except OSError as exc:
            raise PublicationError("exclusive gate staging create failed") from exc
        try:
            opened = os.fstat(temp_fd)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or opened.st_size != 0
            ):
                raise PublicationError("new staging file geometry differs")
            staged_inode = (opened.st_dev, opened.st_ino)
            _write_all(temp_fd, payload)
            os.fsync(temp_fd)
            staged = os.fstat(temp_fd)
            if (
                not stat.S_ISREG(staged.st_mode)
                or staged.st_nlink != 1
                or staged.st_size != len(payload)
                or not _same_inode(staged, staged_inode)
            ):
                raise PublicationError("staged gate file geometry differs")
        finally:
            os.close(temp_fd)
        _assert_directory_identity(
            directory_fd,
            output_directory,
            directory_identity,
            label="output directory",
        )
        if fault is not None:
            fault("after_temp_fsync")
        _assert_directory_identity(
            directory_fd,
            output_directory,
            directory_identity,
            label="output directory",
        )
        staged_visible = _stat_at_optional(directory_fd, temp_name)
        if (
            staged_visible is None
            or not _same_inode(staged_visible, staged_inode)
            or not stat.S_ISREG(staged_visible.st_mode)
            or staged_visible.st_nlink != 1
            or staged_visible.st_size != len(payload)
        ):
            raise PublicationStateUnverifiedError(
                "PUBLICATION_STATE_UNVERIFIED: staged path identity changed before link"
            )
        try:
            os.link(
                temp_name,
                basename,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise AmbiguousPublicationError(
                "gate appeared concurrently; overwrite forbidden"
            ) from exc
        except OSError as exc:
            raise PublicationError("exclusive gate hard-link publication failed") from exc
        os.fsync(directory_fd)
        visible = os.stat(basename, dir_fd=directory_fd, follow_symlinks=False)
        if not _same_inode(visible, staged_inode) or visible.st_nlink != 2:
            raise PublicationError("published gate link identity differs before unlink")
        _assert_directory_identity(
            directory_fd,
            output_directory,
            directory_identity,
            label="output directory",
        )
        if fault is not None:
            fault("after_final_link")
        _assert_directory_identity(
            directory_fd,
            output_directory,
            directory_identity,
            label="output directory",
        )
        staged_visible = _stat_at_optional(directory_fd, temp_name)
        final_visible = _stat_at_optional(directory_fd, basename)
        if (
            staged_visible is None
            or final_visible is None
            or not _same_inode(staged_visible, staged_inode)
            or not _same_inode(final_visible, staged_inode)
            or staged_visible.st_nlink != 2
            or final_visible.st_nlink != 2
        ):
            raise PublicationStateUnverifiedError(
                "PUBLICATION_STATE_UNVERIFIED: link identities changed before temp cleanup"
            )
        _assert_directory_identity(
            directory_fd,
            output_directory,
            directory_identity,
            label="output directory",
        )
        os.unlink(temp_name, dir_fd=directory_fd)
        os.fsync(directory_fd)
        final_stat = os.stat(basename, dir_fd=directory_fd, follow_symlinks=False)
        if not _same_inode(final_stat, staged_inode) or final_stat.st_nlink != 1:
            raise PublicationError("final gate is not the single-link staged inode")
        _assert_directory_identity(
            directory_fd,
            output_directory,
            directory_identity,
            label="output directory",
        )
        if fault is not None:
            fault("after_temp_unlink")
        observed, final_verified = _safe_read_output_at(directory_fd, basename)
        if observed != payload:
            raise PublicationError("final gate differs from staged bytes")
        if fault is not None:
            fault("before_final_accept")
        _assert_directory_identity(
            directory_fd,
            output_directory,
            directory_identity,
            label="output directory",
        )
        _assert_gate_directory_membership(directory_fd, expected={basename})
        _fresh_verify_canonical_gate(
            output_directory,
            directory_chain,
            basename=basename,
            expected_file=final_verified,
            expected_payload=payload,
        )
        return "CREATED_EXCLUSIVE"
    except Exception as exc:
        if staged_inode is None or directory_identity is None:
            raise
        return _recover_publication_after_error(
            original=exc,
            directory_fd=directory_fd,
            output_directory=output_directory,
            directory_identity=directory_identity,
            directory_chain=directory_chain,
            temp_name=temp_name,
            basename=basename,
            staged_inode=staged_inode,
            payload=payload,
        )
    finally:
        os.close(directory_fd)


def _configured_output_directory(config: Mapping[str, Any]) -> Path:
    output = config["output_contract"]
    parent = Path(output["trusted_output_parent"])
    source_forbidden = config["source_contract"]["forbidden_path_tokens"]
    _safe_absolute_path(parent, forbidden_tokens=source_forbidden, label="output parent")
    subdirectory = output["output_subdirectory"]
    if PurePosixPath(subdirectory).name != subdirectory or subdirectory in {"", ".", ".."}:
        raise ScopeViolation("output subdirectory is unsafe")
    result = parent / subdirectory
    _safe_absolute_path(
        result,
        forbidden_tokens=source_forbidden,
        label="output directory",
    )
    return result


def ensure_production_output_directory(config: Mapping[str, Any]) -> Path:
    output_directory = _configured_output_directory(config)
    parent = output_directory.parent
    subdirectory = output_directory.name
    parent_fd = _open_directory_root_to_leaf(parent, label="output parent")
    try:
        parent_stat = os.fstat(parent_fd)
        parent_identity = (parent_stat.st_dev, parent_stat.st_ino)
        _assert_directory_identity(
            parent_fd, parent, parent_identity, label="output parent"
        )
        try:
            os.mkdir(subdirectory, 0o750, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except FileExistsError:
            pass
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            child_fd = os.open(subdirectory, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise PublicationError("output subdirectory is a symlink or non-directory") from exc
        try:
            child_stat = os.fstat(child_fd)
            if not stat.S_ISDIR(child_stat.st_mode):
                raise PublicationError("output subdirectory is not a directory")
            child_identity = (child_stat.st_dev, child_stat.st_ino)
            _assert_directory_identity(
                child_fd,
                output_directory,
                child_identity,
                label="output directory",
            )
        finally:
            os.close(child_fd)
        _assert_directory_identity(
            parent_fd, parent, parent_identity, label="output parent"
        )
    finally:
        os.close(parent_fd)
    return output_directory


def _read_and_validate_gate_at(
    path: Path,
    config: Mapping[str, Any],
    *,
    enforce_trusted_path: bool,
) -> tuple[dict[str, Any], bytes]:
    if path.name != OUTPUT_BASENAME:
        raise ScopeViolation("inspection gate basename differs")
    if enforce_trusted_path:
        expected = _configured_output_directory(config) / OUTPUT_BASENAME
        if path != expected:
            raise ScopeViolation("inspection gate path differs from trusted output slot")
    _safe_absolute_path(
        path,
        forbidden_tokens=config["source_contract"]["forbidden_path_tokens"],
        label="inspection gate path",
    )
    descriptor = None
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise PublicationError("O_NOFOLLOW is unavailable")
    parts = path.parts
    if not path.is_absolute() or parts[0] != os.sep:
        raise ScopeViolation("inspection gate path must be absolute")
    parent_fd, parent_chain = _open_directory_root_to_leaf_with_chain(
        path.parent,
        label="inspection parent",
    )
    try:
        parent_stat = os.fstat(parent_fd)
        parent_identity = (parent_stat.st_dev, parent_stat.st_ino)
        _assert_directory_identity(
            parent_fd,
            path.parent,
            parent_identity,
            label="inspection parent",
        )
        _assert_gate_directory_membership(parent_fd, expected={OUTPUT_BASENAME})
        try:
            descriptor = os.open(
                path.name,
                os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_fd,
            )
        except OSError as exc:
            raise PublicationError("inspection gate cannot be opened safely") from exc
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise AmbiguousPublicationError("inspection gate is not single-link regular")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise AmbiguousPublicationError("inspection gate changed during read")
        try:
            visible = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as exc:
            raise AmbiguousPublicationError(
                "inspection gate path disappeared"
            ) from exc
        if _identity(visible) != _identity(after):
            raise AmbiguousPublicationError("inspection gate path was replaced")
        _assert_directory_identity(
            parent_fd,
            path.parent,
            parent_identity,
            label="inspection parent",
        )
        _assert_gate_directory_membership(parent_fd, expected={OUTPUT_BASENAME})
        payload = b"".join(chunks)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)
    record = strict_json(payload, label="committed lineage gate")
    validate_gate_record(record, config)
    if payload != json_bytes(record):
        raise AmbiguousPublicationError("committed lineage gate is not canonical JSON")
    _fresh_verify_canonical_gate(
        path.parent,
        parent_chain,
        basename=path.name,
        expected_file=after,
        expected_payload=payload,
    )
    return record, payload


def read_and_validate_committed_gate(
    path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Inspect only the exact configured production slot."""

    record, _payload = _read_and_validate_gate_at(
        path,
        config,
        enforce_trusted_path=True,
    )
    return record


def _audit_expected_gate(config: Mapping[str, Any]) -> tuple[dict[str, Any], AuditSummary]:
    payloads = read_source_inputs(config)
    if set(payloads) != {"PMC10540565_TABLE_S2", "PMC10540565_TABLE_S3"}:
        raise InputIntegrityError("source reader returned anything other than exact S2/S3")
    summary = audit_tables(
        payloads["PMC10540565_TABLE_S2"],
        payloads["PMC10540565_TABLE_S3"],
        config["locator_contract"],
    )
    return build_gate_record(config, summary), summary


def inspect_production_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute source evidence and exact-compare the committed gate."""

    validate_implementation_binding(config)
    authority = validate_production_authority(config)
    replay_predecessor_authority(config)
    expected_record, summary = _audit_expected_gate(config)
    output_directory = _configured_output_directory(config)
    target = output_directory / OUTPUT_BASENAME
    committed, committed_payload = _read_and_validate_gate_at(
        target,
        config,
        enforce_trusted_path=True,
    )
    expected_payload = json_bytes(expected_record)
    if committed != expected_record or committed_payload != expected_payload:
        raise AmbiguousPublicationError(
            "committed gate differs from freshly reconstructed source evidence"
        )
    consume_gate_with_current_v3(committed_payload, config)
    return {
        "status": "PASS",
        "mode": "INSPECT_RECOMPUTED_SOURCE_AND_EXACT_GATE",
        "output_path": os.fspath(target),
        "output_sha256": sha256(expected_payload),
        "output_bytes": len(expected_payload),
        "processed_pair_count": summary.processed_pair_count,
        "canonical_record_count": summary.canonical_record_count,
        "authority_mode": authority["mode"],
    }


def execute(
    config: Mapping[str, Any],
    *,
    production: bool,
    source_reader: Callable[[Mapping[str, Any]], dict[str, bytes]] | None = None,
    output_directory_factory: Callable[[Mapping[str, Any]], Path] | None = None,
    publisher: Callable[[bytes, Path], str] | None = None,
) -> dict[str, Any]:
    """Execute after binding; production never accepts injectable components."""

    validate_implementation_binding(config)
    injected = (source_reader, output_directory_factory, publisher)
    if production and any(component is not None for component in injected):
        raise ScopeViolation("production execution forbids injectable components")
    if production:
        authority = validate_production_authority(config)
        replay_predecessor_authority(config)
        record, summary = _audit_expected_gate(config)
        output_directory = ensure_production_output_directory(config)
        payload = json_bytes(record)
        consume_gate_with_current_v3(payload, config)
        publication = publish_single_gate(payload, output_directory)
        enforce_trusted_path = True
    else:
        authority = {
            "mode": "SYNTHETIC_NON_PRODUCTION_BOUND_CONFIG",
            "implementation_commit": config["implementation_binding"][
                "implementation_commit"
            ],
        }
        selected_reader = source_reader or read_source_inputs
        payloads = selected_reader(config)
        if set(payloads) != {"PMC10540565_TABLE_S2", "PMC10540565_TABLE_S3"}:
            raise InputIntegrityError(
                "source reader returned anything other than exact S2/S3"
            )
        summary = audit_tables(
            payloads["PMC10540565_TABLE_S2"],
            payloads["PMC10540565_TABLE_S3"],
            config["locator_contract"],
        )
        record = build_gate_record(config, summary)
        payload = json_bytes(record)
        selected_output_factory = (
            output_directory_factory or ensure_production_output_directory
        )
        output_directory = selected_output_factory(config)
        selected_publisher = publisher or publish_single_gate
        publication = selected_publisher(payload, output_directory)
        enforce_trusted_path = False
    target = output_directory / OUTPUT_BASENAME
    committed, committed_payload = _read_and_validate_gate_at(
        target,
        config,
        enforce_trusted_path=enforce_trusted_path,
    )
    if committed != record or committed_payload != payload:
        raise PublicationError("committed gate differs from audited record")
    if production:
        consume_gate_with_current_v3(committed_payload, config)
    return {
        "status": "PASS",
        "gate_id": GATE_ID,
        "output_path": os.fspath(target),
        "output_sha256": sha256(payload),
        "output_bytes": len(payload),
        "publication": publication,
        "processed_pair_count": summary.processed_pair_count,
        "canonical_record_count": summary.canonical_record_count,
        "raw_replay_status": "NOT_RUN",
        "independent_raw_reproduction_claimed": False,
        "authority_mode": authority["mode"],
    }


def load_config(path: Path) -> dict[str, Any]:
    if path != PRODUCTION_CONFIG_PATH:
        raise ScopeViolation("production config path differs from frozen path")
    payload = _read_worktree_authority_file(PRODUCTION_REPO_ROOT, CONFIG_REPO_PATH)
    config = strict_json(payload, label="producer config")
    validate_static_config(config)
    return config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PRODUCTION_CONFIG_PATH)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-authority", action="store_true")
    mode.add_argument("--produce", action="store_true")
    mode.add_argument("--inspect", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.config != PRODUCTION_CONFIG_PATH:
            raise ScopeViolation("production CLI config path differs from frozen path")
        config = load_config(arguments.config)
        if arguments.validate_authority:
            result = validate_production_authority(config)
        elif arguments.produce:
            result = execute(config, production=True)
        else:
            result = inspect_production_gate(config)
        sys.stdout.buffer.write(json_bytes(result))
        return 0
    except GateProducerError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
