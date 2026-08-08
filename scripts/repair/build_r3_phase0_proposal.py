"""Build a non-active, outcome-free mechanical Phase-0 inventory proposal.

This entrypoint is deliberately incapable of creating a ready dataset, a
context/estimand authority, frozen relation atoms, or a split.  It reads only
hash-bound ordinary inputs, scans JSONL authorities one record at a time, and
emits aggregate review material without sequences, object identities, member
lists, or numeric outcomes.  The result is a proposal for human review, never
an activation or training capability.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import sqlite3
import stat
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.m4_sparse.ordinary_io import (  # noqa: E402
    assert_ordinary_path,
    protected_row_reason,
)
from scripts.r3x.artifact_io import (  # noqa: E402
    ArtifactIOError,
    atomic_write_json_exclusive,
)
from scripts.r3x.protocol import canonical_json, canonical_sha256  # noqa: E402


SCHEMA_VERSION = (
    "r3_phase0_mechanical_proposal_"
    "v4_opaque_library_presence_aware_exact_nested_replay"
)
BINDING_SCHEMA_VERSION = (
    "r3_phase0_exact_input_binding_v3_presence_aware_manifest_replay"
)
OUTCOME_COMMITMENT_SCHEMA_VERSION = (
    "r3_phase0_outcome_free_inventory_commitment_"
    "v4_opaque_library_presence_aware_exact_nested_replay"
)
CONTEXT_CANDIDATE_KEY_SCHEMA_VERSION = (
    "r3_phase0_context_candidate_key_v2_opaque_library"
)
LIBRARY_VALUE_PROJECTION = (
    "DOMAIN_SEPARATED_SHA256_HEX_OCTETS_V1_NO_RAW_LIBRARY_TEXT"
)
OPAQUE_LIBRARY_PREFIX = "OPAQUE_LIBRARY_STRATUM_V1:"
ARTIFACT_ROLE = "R3_PHASE0_MECHANICAL_INVENTORY_PROPOSAL"
STATUS = "PROPOSAL_NON_ACTIVE"
PROPOSAL_FILE = "r3_phase0_mechanical_proposal.json"
RELATION_STATUS_AUTHORITY_FIELD = "relation_acceptance_status"
MISSING_RELATION_STATUS_SOURCE = "MISSING_NOT_ASSERTED"
RELATION_ACCEPTANCE_STATUS_ALLOWLIST = frozenset(
    {"CANDIDATE", "ACCEPTED", "AMBIGUOUS", "REJECTED"}
)
EFFECT_DATASET_MANIFEST_VERSION = (
    "b0x_effect_dataset_manifest_v4_presence_aware_diff_ref_alt_replay"
)
EFFECT_DATASET_SCHEMA_VERSION = (
    "b0x_effect_row_v4_presence_aware_diff_ref_alt_replay"
)
FUNCTIONAL_OBSERVATION_SCHEMA_VERSION = (
    "lossless_functional_observations_v2_input_presence"
)
INPUT_PRESENCE_PROVENANCE_SCHEMA_VERSION = (
    "functional_observation_input_presence_v1"
)
IMPLEMENTATION_BINDING_SCHEMA_VERSION = (
    "b0x_implementation_binding_v2_presence_aware"
)
EFFECT_QUALIFICATION_STATUS = (
    "MECHANICAL_PRESENCE_REPLAY_REQUIRES_MATCHING_V4_MANIFEST_"
    "SCIENTIFIC_HOLD"
)
MANIFEST_QUALIFICATION_STATUS = (
    "MECHANICAL_PRESENCE_REPLAY_VERIFIED_WITH_MATCHING_V4_MANIFEST_"
    "SCIENTIFIC_HOLD"
)
IMPLEMENTATION_BINDING_FIELDS = {
    "schema_version",
    "build_effect_dataset_path",
    "build_effect_dataset_sha256",
    "config_path",
    "config_sha256",
    "delta_specs_sha256",
    "diff_ref_alt_policy_sha256_by_study",
    "effect_dataset_schema_version",
    "manifest_version",
    "diff_ref_alt_join_policy_schema_version",
    "diff_ref_alt_replay_audit_schema_version",
    "functional_observation_input_presence_schema_version",
    "ordinary_only",
    "scientific_unit_authority_created",
    "estimand_authority_created",
}
EFFECT_MANIFEST_FIELDS = {
    "manifest_version",
    "effect_dataset_schema_version",
    "functional_observation_schema_version",
    "functional_observation_input_presence_schema_version",
    "qualification",
    "implementation_binding",
    "n_records",
    "n_pairs_scanned",
    "per_asset",
    "source_anchor_status_counts",
    "pair_relation_provenance",
    "diff_ref_alt_join_audits",
    "legacy_source_lineage",
    "sha256",
    "inputs",
    "config",
}
DIFF_REF_ALT_POLICY_SCHEMA_VERSION = (
    "diff_ref_alt_join_policy_v3_presence_aware_null_semantics"
)
DIFF_REF_ALT_AUDIT_SCHEMA_VERSION = (
    "diff_ref_alt_join_audit_v4_presence_aware_independent_replay"
)
REQUIRED_DIFF_REF_ALT_POLICY_SHA256 = {
    "ENCSR854RUF": "498a98cf4704ff5bd525f96ec97eb0af10a2141a4b376e222193c5a042667623",
    "GSE186455": "18cfc10a93de15b582e4bf1a85824782cf999081530e79b42b90ea2eb575680f",
    "GSE232571": "e5d935c3cfb26af4d821b50ad87de5c658f25491c7a66b9e58578c9040c9906a",
    "GSE232572": "7443feffd250ab5cd04a4152aabfaa168322460c54f53f934c4fef1478ae0cdb",
}
DIFF_REF_ALT_POLICY_FIELDS = {
    "schema_version",
    "policy_id",
    "key_fields",
    "context_match",
    "context_required_non_null",
    "assay_context_match",
    "assay_context_null_policy",
    "assay_context_absent_policy",
    "replicate_match",
    "replicate_field",
    "replicate_null_policy",
    "replicate_absent_policy",
    "presence_match",
    "input_presence_provenance_schema_version",
    "ref_allowed_raw_unit_tokens",
    "alt_allowed_raw_unit_tokens",
    "comparison_scale_id",
    "raw_unit_semantics",
}
DIFF_REF_ALT_AUDIT_FIELDS = {
    "schema_version",
    "verification_status",
    "join_policy",
    "join_policy_sha256",
    "policy_id",
    "comparison_scale_id",
    "n_materialized_pairs",
    "n_output_pairs",
    "n_output_join_rows",
    "n_expected_missing_both_pairs",
    "n_consumed_missing_both_rows",
    "n_ref_observation_groups",
    "n_alt_observation_groups",
    "n_input_side_groups",
    "n_consumed_ref_observation_groups",
    "n_consumed_alt_observation_groups",
    "n_consumed_side_groups",
    "n_matched_groups",
    "n_unmatched_ref_groups",
    "n_unmatched_alt_groups",
    "n_missing_both_rows",
    "n_derived_rows",
    "n_unavailable_rows",
    "expected_output_rows_from_group_conservation",
    "join_disposition_counts",
    "delta_unavailable_reason_counts",
    "replicate_pairing_status_counts",
    "ref_assay_context_input_presence_group_counts",
    "alt_assay_context_input_presence_group_counts",
    "ref_replicate_input_presence_group_counts",
    "alt_replicate_input_presence_group_counts",
    "ref_raw_unit_token_group_counts",
    "alt_raw_unit_token_group_counts",
    "expected_input_side_group_multiset_sha256",
    "consumed_output_side_group_multiset_sha256",
    "expected_missing_both_multiset_sha256",
    "consumed_missing_both_multiset_sha256",
    "materialized_pair_registry_sha256",
    "output_pair_registry_sha256",
}
DIFF_REF_ALT_DISPOSITIONS = {
    "MATCHED": (True, True),
    "UNMATCHED_REF_ONLY": (True, False),
    "UNMATCHED_ALT_ONLY": (False, True),
    "MISSING_BOTH": (False, False),
}
INPUT_PRESENCE_STATUSES = {
    "ABSENT_CANONICALIZED_TO_NULL",
    "EXPLICIT_NULL",
    "EXPLICIT_VALUE",
}
NOT_APPLICABLE_PRESENCE_STATUS = "NOT_APPLICABLE_NO_OBSERVATION"
OBSERVATION_JOIN_KEY_FIELDS = {
    "context_id",
    "assay_context",
    "assay_context_input_presence",
    "replicate",
    "replicate_input_presence",
    "comparison_scale_id",
}
INPUT_PRESENCE_PROVENANCE_FIELDS = {
    "schema_version",
    "assay_context",
    "replicate",
}
RAW_INPUT_VALUE_FIELD = "raw_input_value_before_numeric_normalization"
EMBEDDED_OBSERVATION_RESERVED_FIELDS = {
    "input_row_sha256",
    "raw_input_row_sha256",
    RAW_INPUT_VALUE_FIELD,
    "input_presence_provenance",
    "upstream_declared_input_row_sha256",
}
EMBEDDED_OBSERVATION_RESERVED_PREFIXES = (
    "input_presence_",
    "input_row_sha256",
    "raw_input_",
    "upstream_declared_input_",
)

D1_FILE_NAMES = {
    "utr_edit_pairs": "utr_edit_pairs.jsonl",
    "functional_observations": "functional_observations.jsonl",
    "group_assignments": "group_assignments.jsonl",
    "group_registry": "group_registry.jsonl",
    "sequence_entities": "sequence_entities.jsonl",
    "object_attributes": "object_attributes.jsonl",
    "grouping_atom_projection": "GROUPING_ATOM_PROJECTION.json",
}
JSONL_D1_AUTHORITIES = tuple(
    logical for logical in D1_FILE_NAMES if logical != "grouping_atom_projection"
)
CONTEXT_KEY_FIELDS = (
    "study",
    "endpoint",
    "assay_arm",
    "library",
    "source_asset",
    "region",
    "source_anchor_rule",
)
MANDATORY_CONTEXT_FIELDS = tuple(
    field for field in CONTEXT_KEY_FIELDS if field != "library"
)
ESTIMAND_REVIEW_FIELDS = [
    "estimand_id",
    "raw_unit",
    "direction",
    "transform",
    "valid_domain",
    "role",
    "replicate_policy",
    "source_evidence",
]
HUMAN_REVIEW_REQUIRED = [
    "PUBLICATION_CONTEXT_CROSSWALK",
    "ENDPOINT_ESTIMAND_REGISTRY",
    "RELATION_ATOM_V2_EDGE_POLICY",
    "OUTCOME_FREE_ROW_IDENTITY_POLICY",
    "FORMAL_CELL_SUPPORT_POLICY",
]
PROPOSAL_LIMITATIONS = [
    "NO_CONTEXT_AUTHORITY",
    "NO_ESTIMAND_AUTHORITY",
    "NO_FROZEN_RELATION_ATOMS",
    "NO_OUTCOME_FREE_TYPED_ROWS",
    "NO_READY_DATASET",
    "NO_SPLIT",
    "NO_TRAINING_CAPABILITY",
    "NO_RAW_LIBRARY_TEXT",
]
BINDING_CHECKS = {
    "effect_manifest_binds_effect_sha256": True,
    "effect_manifest_binds_pairs_sha256": True,
    "effect_manifest_binds_functional_observations_sha256": True,
    "d1_sha256s_binds_all_required_d1_jsonl_authorities": True,
    "grouping_atom_projection_exact_sha256_verified_separately": True,
    "all_inputs_exact_sha256_verified": True,
    "all_jsonl_inputs_streamed": True,
    "effect_manifest_v4_exact_version_verified": True,
    "effect_manifest_effect_schema_verified": True,
    "effect_manifest_functional_observation_schema_verified": True,
    "effect_manifest_functional_observation_input_presence_schema_verified": True,
    "effect_manifest_n_records_matches_effect_scan": True,
    "effect_rows_functional_observation_presence_verified": True,
    "diff_ref_alt_presence_aware_independent_replay_audits_verified": True,
    "diff_ref_alt_materialized_dispositions_crosschecked": True,
    "diff_ref_alt_materialized_presence_crosschecked": True,
    "library_values_projected_to_opaque_domain_separated_commitments": True,
}
INVENTORY_POLICY = {
    "schema_version": (
        "r3_phase0_outcome_free_inventory_policy_v2_opaque_library"
    ),
    "ordinary_only": True,
    "input_row_order_affects_inventory": False,
    "numeric_outcome_values_affect_inventory": False,
    "context_inference": "PROHIBITED",
    "estimand_inference": "PROHIBITED",
    "relation_edge_inference": "PROHIBITED",
    "library_missing_policy": "PRESERVE_NULL_NO_IMPUTATION",
    "library_value_projection": LIBRARY_VALUE_PROJECTION,
    "mrl_semantic_family": "RIBOSOME_LOADING_MRL",
    "split_construction": "PROHIBITED",
}
GROUPING_ATOM_ALLOWLIST = {
    "BARCODE",
    "BIOLOGICAL_PARENT",
    "BIOLOGICAL_REPLICATE",
    "CONTEXT",
    "EXPERIMENTAL_CONTEXT",
    "GENE",
    "LIBRARY_LINEAGE",
    "NO_EDIT_SAMPLING_FRAME",
    "PAIR",
    "SAMPLE",
    "SEQUENCE_CLUSTER",
    "SOURCE",
    "STUDY",
    "TECHNICAL_REPLICATE",
    "TILE_FAMILY",
    "TRANSCRIPT",
}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_INTERNAL_TEST_RE = re.compile(
    r"(?:^|[^a-z0-9])internal(?:[\s_-]*test)(?:$|[^a-z0-9])",
    flags=re.IGNORECASE,
)
_SEALED_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:sealed|restricted)(?:$|[^a-z0-9])|gse246381",
    flags=re.IGNORECASE,
)
_RAW_IDENTITY_TEXT_RE = re.compile(
    r"(?:^|[^a-z0-9])"
    r"(?:assignment|group|member|object|observation|obs|pair|record|row|seq|sequence)"
    r"(?:[_:#=/-]|$)",
    flags=re.IGNORECASE,
)
_RAW_OUTCOME_TEXT_RE = re.compile(
    r"(?:^|[^a-z0-9])"
    r"(?:candidate_value|delta|outcome|sign|source_value)(?:[_:=\s-]|$)",
    flags=re.IGNORECASE,
)
_RAW_SEQUENCE_TEXT_RE = re.compile(
    r"(?:^|[^A-Za-z])[ACGTUNacgtun]{4,}(?:$|[^A-Za-z])"
)
_PROHIBITED_OUTPUT_KEYS = {
    "delta",
    "sign",
    "outcome",
    "source_value",
    "candidate_value",
    "value",
    "row_id",
    "row_ids",
    "record_id",
    "record_ids",
    "pair_id",
    "pair_ids",
    "object_id",
    "object_ids",
    "member_ids",
    "members",
    "source_sequence",
    "candidate_sequence",
    "normalized_sequence",
    "sequence_values",
}


class Phase0ProposalError(RuntimeError):
    """Fail-closed proposal construction error with no readiness semantics."""


def _fail(message: str) -> None:
    raise Phase0ProposalError(message)


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if not _HEX64.fullmatch(text):
        _fail(f"{label} must be a lowercase SHA-256")
    return text


def _unsafe_text(value: object) -> bool:
    if value is None or isinstance(value, (bool, int, float)):
        return False
    text = str(value).strip()
    return bool(_SEALED_RE.search(text) or _INTERNAL_TEST_RE.search(text))


def _assert_phase0_path(path: Path, *, purpose: str) -> None:
    path = Path(path)
    try:
        assert_ordinary_path(path, purpose=purpose)
    except ValueError as exc:
        _fail(f"protected ordinary-only path refused for {purpose}: {exc}")
    candidates = [path]
    try:
        candidates.append(path.resolve(strict=False))
    except OSError:
        pass
    bad = sorted(
        {
            part
            for candidate in candidates
            for part in candidate.parts
            if _unsafe_text(part)
        }
    )
    if bad:
        _fail(
            f"protected path component refused for {purpose}: "
            + ", ".join(bad)
        )


def _walk_input(value: object, path: str = "$"):
    if isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: str(item)):
            child = value[key]
            child_path = f"{path}.{key}"
            yield child_path + "::<field_name>", key
            yield from _walk_input(child, child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk_input(child, f"{path}[{index}]")
    else:
        yield path, value


def _protected_input_reason(row: Mapping) -> str | None:
    ordinary_reason = protected_row_reason(dict(row))
    if ordinary_reason is not None:
        return ordinary_reason
    for location, value in _walk_input(row):
        if _INTERNAL_TEST_RE.search(str(value or "")):
            return f"protected_internal_test_reference:{location}"
    return None


def _open_regular(path: Path, *, purpose: str) -> tuple[int, os.stat_result]:
    _assert_phase0_path(path, purpose=purpose)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError as exc:
        raise Phase0ProposalError(f"input does not exist: {path}") from exc
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.EMLINK}:
            raise Phase0ProposalError("input cannot be a symlink") from exc
        raise Phase0ProposalError(f"input cannot be opened safely: {exc}") from exc
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        _fail("input must be a regular single-link file")
    return descriptor, metadata


def _metadata_stable(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
        and before.st_size == after.st_size
        and after.st_nlink == 1
        and stat.S_ISREG(after.st_mode)
    )


def _read_exact_bytes(path: Path, expected_sha256: str, *, purpose: str) -> bytes:
    expected = _require_sha256(expected_sha256, label=f"{purpose} expected SHA-256")
    descriptor, before = _open_regular(path, purpose=purpose)
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            while True:
                chunk = handle.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                chunks.append(chunk)
            after = os.fstat(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not _metadata_stable(before, after):
        _fail(f"{purpose} metadata changed during exact read")
    actual = digest.hexdigest()
    if actual != expected:
        _fail(f"{purpose} SHA-256 mismatch: expected {expected}, got {actual}")
    return b"".join(chunks)


def _read_exact_json(path: Path, expected_sha256: str, *, purpose: str) -> tuple[dict, dict]:
    raw = _read_exact_bytes(path, expected_sha256, purpose=purpose)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Phase0ProposalError(f"{purpose} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        _fail(f"{purpose} must be a JSON object")
    reason = _protected_input_reason(payload)
    if reason is not None:
        _fail(f"protected content refused in {purpose}: {reason}")
    return payload, {
        "artifact_format": "JSON",
        "byte_count": len(raw),
        "sha256": _require_sha256(expected_sha256, label=f"{purpose} SHA-256"),
    }


def _scan_jsonl(
    path: Path,
    expected_sha256: str,
    *,
    purpose: str,
    visitor: Callable[[dict], None],
) -> dict:
    """Hash and parse a JSONL authority from one descriptor, one row at a time."""

    expected = _require_sha256(expected_sha256, label=f"{purpose} expected SHA-256")
    descriptor, before = _open_regular(path, purpose=purpose)
    digest = hashlib.sha256()
    byte_count = 0
    row_count = 0
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            for line_number, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                byte_count += len(raw_line)
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line.decode("utf-8"))
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise Phase0ProposalError(
                        f"{purpose} row {line_number} is invalid JSON: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    _fail(f"{purpose} row {line_number} is not an object")
                reason = _protected_input_reason(row)
                if reason is not None:
                    _fail(
                        f"protected content refused in {purpose} row "
                        f"{line_number}: {reason}"
                    )
                visitor(row)
                row_count += 1
            after = os.fstat(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not _metadata_stable(before, after):
        _fail(f"{purpose} metadata changed during streaming scan")
    actual = digest.hexdigest()
    if actual != expected:
        _fail(f"{purpose} SHA-256 mismatch: expected {expected}, got {actual}")
    return {
        "artifact_format": "JSONL",
        "byte_count": byte_count,
        "json_row_count": row_count,
        "sha256": actual,
        "streaming_scan": True,
    }


def _parse_d1_sha256s(raw: bytes) -> dict[str, str]:
    entries: dict[str, str] = {}
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise Phase0ProposalError("D1_SHA256SUMS is not UTF-8") from exc
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            _fail(f"D1_SHA256SUMS line {line_number} is malformed")
        digest = _require_sha256(parts[0], label=f"D1_SHA256SUMS line {line_number}")
        name = parts[1].strip().lstrip("*")
        if not name or _unsafe_text(name):
            _fail(f"protected or empty D1_SHA256SUMS entry on line {line_number}")
        basename = Path(name).name
        if basename in entries:
            _fail(f"D1_SHA256SUMS contains duplicate basename: {basename}")
        entries[basename] = digest
    return entries


def _category(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (bool, int, float)):
        return str(value)
    try:
        return canonical_json(value)
    except Exception:
        return str(value).strip() or None


def _opaque_library_value(value: object) -> str | None:
    """Project an ordinary library label without emitting its raw text.

    The domain separator prevents a commitment from being reused as a generic
    SHA-256 identity.  Hex octets are separated so the public token cannot be
    mistaken for an A/C/G/T/U/N sequence by the protected-text validator.
    """
    category = _category(value)
    if category is None:
        return None
    digest = hashlib.sha256(
        ("R3_PHASE0_OPAQUE_LIBRARY_STRATUM_V1\0" + category).encode("utf-8")
    ).hexdigest()
    return OPAQUE_LIBRARY_PREFIX + "-".join(
        digest[index:index + 2] for index in range(0, len(digest), 2)
    )


def _pair_hash(value: object) -> str | None:
    text = _category(value)
    if text is None:
        return None
    return hashlib.sha256(("PAIR_IDENTITY_V1\0" + text).encode("utf-8")).hexdigest()


def _group_hash(atom: object, group_id: object) -> str | None:
    atom_text = _category(atom)
    group_text = _category(group_id)
    if atom_text is None or group_text is None:
        return None
    return hashlib.sha256(
        ("GROUP_IDENTITY_V1\0" + atom_text + "\0" + group_text).encode("utf-8")
    ).hexdigest()


def _endpoint_family(study: str | None, endpoint: str | None, unit: str | None) -> str:
    token = " ".join(value or "" for value in (study, endpoint, unit)).lower()
    if any(marker in token for marker in ("mrl", "ribosome", "ep_rl", "translation")):
        return "RIBOSOME_LOADING_MRL"
    if any(marker in token for marker in ("half_life", "half-life", "decay", "ep_stability")):
        return "RNA_HALF_LIFE"
    return "UNCLASSIFIED_ENDPOINT"


def _edit_script_kind(value: object) -> str:
    if not isinstance(value, list) or not value:
        return "missing_or_empty"
    if all(
        isinstance(item, Mapping)
        and set(item) == {"op", "pos", "ref", "alt"}
        and item.get("op") == "SUB"
        for item in value
    ):
        return "canonical_op_pos_ref_alt"
    if all(
        isinstance(item, Mapping)
        and "pos" in item
        and "token" in item
        and "op" not in item
        for item in value
    ):
        return "legacy_pos_token"
    return "other_or_mixed"


def _initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        CREATE TABLE effect_pairs (
            pair_hash TEXT PRIMARY KEY
        );
        CREATE TABLE effect_diff_pairs (
            study TEXT NOT NULL,
            pair_hash TEXT NOT NULL,
            PRIMARY KEY (study, pair_hash)
        );
        CREATE TABLE d1_pairs (
            pair_hash TEXT PRIMARY KEY,
            resolved_status TEXT NOT NULL,
            status_source_field TEXT NOT NULL,
            accepted INTEGER NOT NULL
        );
        CREATE TABLE context_candidates (
            candidate_key_sha256 TEXT PRIMARY KEY,
            key_json TEXT NOT NULL,
            row_count INTEGER NOT NULL
        );
        CREATE TABLE context_aliases (
            candidate_key_sha256 TEXT NOT NULL,
            alias_sha256 TEXT NOT NULL,
            PRIMARY KEY (candidate_key_sha256, alias_sha256)
        );
        CREATE TABLE estimand_candidates (
            candidate_key_sha256 TEXT PRIMARY KEY,
            study TEXT,
            endpoint TEXT,
            observed_unit TEXT,
            row_count INTEGER NOT NULL
        );
        CREATE TABLE groups (
            atom TEXT NOT NULL,
            group_hash TEXT NOT NULL,
            declared_member_count INTEGER NOT NULL,
            member_ids_valid INTEGER NOT NULL,
            group_sha_valid INTEGER NOT NULL,
            declared_group_sha256 TEXT,
            PRIMARY KEY (atom, group_hash)
        );
        CREATE TABLE registry_pair_members (
            pair_hash TEXT NOT NULL,
            atom TEXT NOT NULL,
            group_hash TEXT NOT NULL,
            PRIMARY KEY (pair_hash, atom, group_hash)
        );
        CREATE TABLE pair_atoms (
            pair_hash TEXT NOT NULL,
            atom TEXT NOT NULL,
            group_hash TEXT NOT NULL,
            PRIMARY KEY (pair_hash, atom, group_hash)
        );
        CREATE TABLE attribute_atoms (
            pair_hash TEXT NOT NULL,
            atom TEXT NOT NULL,
            group_hash TEXT NOT NULL,
            PRIMARY KEY (pair_hash, atom, group_hash)
        );
        """
    )


def _increment_context(
    connection: sqlite3.Connection,
    key: Mapping[str, str | None],
    aliases: Sequence[str],
) -> None:
    key_json = canonical_json(dict(key))
    candidate_sha = canonical_sha256(
        {"schema_version": CONTEXT_CANDIDATE_KEY_SCHEMA_VERSION, **dict(key)}
    )
    connection.execute(
        """
        INSERT INTO context_candidates(candidate_key_sha256, key_json, row_count)
        VALUES (?, ?, 1)
        ON CONFLICT(candidate_key_sha256)
        DO UPDATE SET row_count = row_count + 1
        """,
        (candidate_sha, key_json),
    )
    for alias in sorted(set(aliases)):
        alias_sha = hashlib.sha256(
            ("INPUT_CONTEXT_ALIAS_V1\0" + alias).encode("utf-8")
        ).hexdigest()
        connection.execute(
            "INSERT OR IGNORE INTO context_aliases VALUES (?, ?)",
            (candidate_sha, alias_sha),
        )


def _increment_estimand_candidate(
    connection: sqlite3.Connection,
    *,
    study: str | None,
    endpoint: str | None,
    observed_unit: str | None,
) -> None:
    key = {
        "study": study,
        "endpoint": endpoint,
        "observed_unit": observed_unit,
    }
    candidate_sha = canonical_sha256(
        {"schema_version": "r3_phase0_estimand_candidate_key_v1", **key}
    )
    connection.execute(
        """
        INSERT INTO estimand_candidates(
            candidate_key_sha256, study, endpoint, observed_unit, row_count
        ) VALUES (?, ?, ?, ?, 1)
        ON CONFLICT(candidate_key_sha256)
        DO UPDATE SET row_count = row_count + 1
        """,
        (candidate_sha, study, endpoint, observed_unit),
    )


def _context_inventory(
    connection: sqlite3.Connection,
    missing: Counter,
) -> dict:
    candidates = []
    conflict_count = 0
    rows = connection.execute(
        """
        SELECT c.candidate_key_sha256, c.key_json, c.row_count,
               COUNT(a.alias_sha256) AS alias_count
        FROM context_candidates AS c
        LEFT JOIN context_aliases AS a
          ON a.candidate_key_sha256 = c.candidate_key_sha256
        GROUP BY c.candidate_key_sha256, c.key_json, c.row_count
        ORDER BY c.candidate_key_sha256
        """
    )
    for candidate_sha, key_json, row_count, alias_count in rows:
        conflict = int(alias_count) > 1
        conflict_count += int(conflict)
        candidates.append(
            {
                "candidate_key_sha256": candidate_sha,
                "mechanical_key": json.loads(key_json),
                "row_count": int(row_count),
                "observed_input_context_alias_count": int(alias_count),
                "context_conflict_candidate": conflict,
                "review_status": "HUMAN_REVIEW_REQUIRED",
            }
        )
    return {
        "status": STATUS,
        "context_authority_created": False,
        "context_decision": "NONE",
        "library_missing_policy": "PRESERVE_NULL_NO_IMPUTATION",
        "library_value_projection": LIBRARY_VALUE_PROJECTION,
        "candidate_count": len(candidates),
        "candidate_conflict_count": conflict_count,
        "missing_field_counts": {
            field: int(missing.get(field, 0)) for field in CONTEXT_KEY_FIELDS
        },
        "candidates": candidates,
    }


def _estimand_queue(connection: sqlite3.Connection) -> list[dict]:
    output = []
    rows = connection.execute(
        """
        SELECT candidate_key_sha256, study, endpoint, observed_unit, row_count
        FROM estimand_candidates
        ORDER BY candidate_key_sha256
        """
    )
    for candidate_sha, study, endpoint, unit, row_count in rows:
        output.append(
            {
                "candidate_key_sha256": candidate_sha,
                "study": study,
                "endpoint": endpoint,
                "observed_unit": unit,
                "row_count": int(row_count),
                "mechanical_endpoint_family_hint": _endpoint_family(
                    study, endpoint, unit
                ),
                "review_status": "HUMAN_REVIEW_REQUIRED",
                "estimand_decision": "NONE",
                "required_human_review_fields": list(ESTIMAND_REVIEW_FIELDS),
            }
        )
    return output


def _relation_inventory(
    connection: sqlite3.Connection,
    *,
    counters: Counter,
    d1_status_counts: Mapping[str, int],
    d1_status_source_counts: Mapping[str, int],
    projection: Mapping,
) -> dict:
    effect_pair_count = int(
        connection.execute("SELECT COUNT(*) FROM effect_pairs").fetchone()[0]
    )
    matched_pair_count = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM effect_pairs AS e
            INNER JOIN d1_pairs AS d ON d.pair_hash = e.pair_hash
            """
        ).fetchone()[0]
    )
    accepted_pair_count = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM effect_pairs AS e
            INNER JOIN d1_pairs AS d ON d.pair_hash = e.pair_hash
            WHERE d.accepted = 1
            """
        ).fetchone()[0]
    )
    registry_effect_pair_membership_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM registry_pair_members"
        ).fetchone()[0]
    )
    atoms = [
        row[0]
        for row in connection.execute(
            """
            SELECT atom FROM groups
            UNION SELECT atom FROM pair_atoms
            UNION SELECT atom FROM attribute_atoms
            ORDER BY atom
            """
        )
    ]
    coverage = []
    largest_hub_count = 0
    largest_hub_atom = None
    for atom in atoms:
        pair_count = int(
            connection.execute(
                "SELECT COUNT(DISTINCT pair_hash) FROM pair_atoms WHERE atom = ?",
                (atom,),
            ).fetchone()[0]
        )
        group_count = int(
            connection.execute(
                "SELECT COUNT(DISTINCT group_hash) FROM pair_atoms WHERE atom = ?",
                (atom,),
            ).fetchone()[0]
        )
        hub_rows = [
            int(row[0])
            for row in connection.execute(
                """
                SELECT COUNT(DISTINCT pair_hash)
                FROM pair_atoms
                WHERE atom = ?
                GROUP BY group_hash
                """,
                (atom,),
            )
        ]
        max_hub = max(hub_rows, default=0)
        if max_hub > largest_hub_count:
            largest_hub_count = max_hub
            largest_hub_atom = atom
        attribute_pair_count = int(
            connection.execute(
                "SELECT COUNT(DISTINCT pair_hash) FROM attribute_atoms WHERE atom = ?",
                (atom,),
            ).fetchone()[0]
        )
        assignment_not_in_attributes = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM pair_atoms AS p
                LEFT JOIN attribute_atoms AS a
                  ON a.pair_hash = p.pair_hash
                 AND a.atom = p.atom
                 AND a.group_hash = p.group_hash
                WHERE p.atom = ? AND a.pair_hash IS NULL
                """,
                (atom,),
            ).fetchone()[0]
        )
        attributes_not_in_assignments = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM attribute_atoms AS a
                LEFT JOIN pair_atoms AS p
                  ON p.pair_hash = a.pair_hash
                 AND p.atom = a.atom
                 AND p.group_hash = a.group_hash
                WHERE a.atom = ? AND p.pair_hash IS NULL
                """,
                (atom,),
            ).fetchone()[0]
        )
        coverage.append(
            {
                "atom_type": atom,
                "effect_pair_count_with_assignment": pair_count,
                "effect_pair_coverage_denominator": effect_pair_count,
                "distinct_group_count_for_effect_pairs": group_count,
                "object_attribute_pair_count": attribute_pair_count,
                "max_effect_pairs_in_one_group": max_hub,
                "multi_pair_group_count": sum(value > 1 for value in hub_rows),
                "assignment_not_in_object_attributes_count": (
                    assignment_not_in_attributes
                ),
                "object_attributes_not_in_assignment_count": (
                    attributes_not_in_assignments
                ),
                "mapping_status": "MECHANICAL_EVIDENCE_ONLY",
                "edge_decision": "UNDECIDED_HUMAN_REVIEW_REQUIRED",
            }
        )
    hold_counter_rules = (
        ("D1_RELATION_STATUS_UNREVIEWED", int(d1_status_counts.get("UNREVIEWED", 0))),
        (
            "REGISTRY_MEMBER_IDS_INVALID",
            int(counters.get("registry_member_ids_invalid", 0)),
        ),
        (
            "REGISTRY_GROUP_SHA_MISMATCH",
            int(counters.get("registry_group_sha_mismatch", 0)),
        ),
        (
            "CONFLICTING_GROUP_REGISTRY_ROWS",
            int(counters.get("conflicting_group_registry_rows", 0)),
        ),
        (
            "ASSIGNMENT_MISSING_REGISTRY_GROUP",
            int(counters.get("assignment_missing_registry_group", 0)),
        ),
        (
            "ASSIGNMENT_REGISTRY_GROUP_INVALID",
            int(counters.get("assignment_registry_group_invalid", 0)),
        ),
        (
            "ASSIGNMENT_PAIR_NOT_REGISTRY_MEMBER",
            int(counters.get("assignment_pair_not_registry_member", 0)),
        ),
        (
            "ASSIGNMENT_ALGORITHM_MISMATCH",
            int(counters.get("assignment_algorithm_mismatch", 0)),
        ),
    )
    hold_reasons = [reason for reason, count in hold_counter_rules if count > 0]
    return {
        "status": STATUS,
        "relation_atom_authority_created": False,
        "frozen_relation_atoms_created": False,
        "edge_policy_created": False,
        "review_disposition": (
            "HOLD_FOR_HUMAN_REVIEW"
            if hold_reasons
            else "HUMAN_REVIEW_REQUIRED"
        ),
        "hold_reasons": hold_reasons,
        "pair_join_audit": {
            "unique_effect_pair_count": effect_pair_count,
            "d1_pair_match_count": matched_pair_count,
            "d1_accepted_pair_match_count": accepted_pair_count,
            "d1_pair_missing_count": effect_pair_count - matched_pair_count,
            "relation_status_authority_field": RELATION_STATUS_AUTHORITY_FIELD,
            "relation_status_allowlist": sorted(
                RELATION_ACCEPTANCE_STATUS_ALLOWLIST
            ),
            "canonical_missing_resolution": "UNREVIEWED",
            "canonical_missing_source_field": MISSING_RELATION_STATUS_SOURCE,
            "legacy_status_role": "CONSISTENCY_CHECK_ONLY_NO_PROMOTION",
            "d1_relation_status_counts": dict(sorted(d1_status_counts.items())),
            "d1_relation_status_source_field_counts": dict(
                sorted(d1_status_source_counts.items())
            ),
        },
        "projection_metadata": {
            "artifact_id": _category(projection.get("artifact_id")),
            "schema_version": _category(projection.get("schema_version")),
            "grouping_atom_projection_rule_sha256": _category(
                projection.get("grouping_atom_projection_rule_sha256")
            ),
            "assignment_algorithm_id": _category(
                projection.get("assignment_algorithm_id")
            ),
            "group_id_algorithm_id": _category(
                projection.get("group_id_algorithm_id")
            ),
            "review_status": "HUMAN_REVIEW_REQUIRED",
        },
        "registry_group_sha_mismatch_count": int(
            counters.get("registry_group_sha_mismatch", 0)
        ),
        "conflicting_group_registry_row_count": int(
            counters.get("conflicting_group_registry_rows", 0)
        ),
        "registry_member_ids_invalid_count": int(
            counters.get("registry_member_ids_invalid", 0)
        ),
        "registry_effect_pair_membership_count": (
            registry_effect_pair_membership_count
        ),
        "assignment_missing_registry_group_count": int(
            counters.get("assignment_missing_registry_group", 0)
        ),
        "assignment_registry_group_invalid_count": int(
            counters.get("assignment_registry_group_invalid", 0)
        ),
        "assignment_pair_not_registry_member_count": int(
            counters.get("assignment_pair_not_registry_member", 0)
        ),
        "assignment_algorithm_mismatch_count": int(
            counters.get("assignment_algorithm_mismatch", 0)
        ),
        "coverage_by_atom": coverage,
        "hub_audit": {
            "largest_effect_pair_hub_atom": largest_hub_atom,
            "largest_effect_pair_hub_count": largest_hub_count,
            "hub_interpretation": "MECHANICAL_REVIEW_ONLY_NOT_AN_EDGE_DECISION",
        },
    }


def _authority_entry(logical_name: str, path: Path, stats: Mapping) -> dict:
    return {
        "logical_name": logical_name,
        "basename": Path(path).name,
        **dict(stats),
    }


def _validate_input_maps(
    d1_paths: Mapping[str, Path],
    d1_expected_sha256: Mapping[str, str],
) -> tuple[dict[str, Path], dict[str, str]]:
    if set(d1_paths) != set(D1_FILE_NAMES):
        _fail(
            "d1_paths must contain exactly: " + ", ".join(sorted(D1_FILE_NAMES))
        )
    if set(d1_expected_sha256) != set(D1_FILE_NAMES):
        _fail(
            "d1_expected_sha256 must contain exactly: "
            + ", ".join(sorted(D1_FILE_NAMES))
        )
    normalized_paths = {key: Path(d1_paths[key]) for key in D1_FILE_NAMES}
    normalized_hashes = {
        key: _require_sha256(
            d1_expected_sha256[key], label=f"{key} expected SHA-256"
        )
        for key in D1_FILE_NAMES
    }
    for logical, path in normalized_paths.items():
        if path.name != D1_FILE_NAMES[logical]:
            _fail(
                f"{logical} basename must be exactly {D1_FILE_NAMES[logical]}"
            )
        _assert_phase0_path(path, purpose=f"{logical} input")
    return normalized_paths, normalized_hashes


def _validate_diff_ref_alt_policy(
    value: object,
    *,
    study: str,
    expected_sha256: str,
) -> dict:
    policy = _expect_object(
        value,
        DIFF_REF_ALT_POLICY_FIELDS,
        label=f"effect manifest diff policy {study}",
    )
    if (
        policy["schema_version"] != DIFF_REF_ALT_POLICY_SCHEMA_VERSION
        or policy["key_fields"]
        != [
            "context_id",
            "assay_context",
            "assay_context_input_presence",
            "replicate",
            "replicate_input_presence",
        ]
        or policy["context_match"] != "EXACT"
        or policy["context_required_non_null"] is not True
        or policy["assay_context_match"] != "EXACT"
        or policy["assay_context_null_policy"]
        != "ALLOW_EXACT_NULL_MECHANICAL_NO_ASSAY_CONTEXT_CLAIM"
        or policy["assay_context_absent_policy"]
        != "CANONICALIZE_TO_NULL_MECHANICAL_NO_ASSAY_CONTEXT_CLAIM"
        or policy["replicate_match"]
        != "EXACT_CANONICAL_TOKEN_INCLUDING_NULL"
        or policy["replicate_field"] != "replicate"
        or policy["replicate_null_policy"]
        != "ALLOW_EXACT_NULL_NO_REPLICATE_CLAIM"
        or policy["replicate_absent_policy"]
        != "CANONICALIZE_TO_NULL_MECHANICAL_NO_REPLICATE_CLAIM"
        or policy["presence_match"]
        != "EXACT_ABSENT_EXPLICIT_NULL_EXPLICIT_VALUE"
        or policy["input_presence_provenance_schema_version"]
        != INPUT_PRESENCE_PROVENANCE_SCHEMA_VERSION
        or policy["raw_unit_semantics"]
        != "ENDPOINT_SPECIFIC_RAW_LABEL_TOKEN_NOT_PHYSICAL_UNIT"
    ):
        _fail(f"effect manifest diff policy {study} has stale semantics")
    for field in ("policy_id", "comparison_scale_id"):
        _require_safe_text(policy[field], label=f"diff policy {study} {field}")
    for field in (
        "ref_allowed_raw_unit_tokens",
        "alt_allowed_raw_unit_tokens",
    ):
        tokens = _expect_list(policy[field], label=f"diff policy {study} {field}")
        if not tokens:
            _fail(f"diff policy {study} {field} must be non-empty")
        for token in tokens:
            _require_safe_text(token, label=f"diff policy {study} raw-unit token")
        if tokens != sorted(set(tokens)):
            _fail(f"diff policy {study} {field} must be sorted and unique")
    if canonical_sha256(policy) != expected_sha256:
        _fail(f"effect manifest diff policy {study} SHA-256 mismatch")
    return policy


def _validate_embedded_observation_hashes(
    observation: Mapping,
    *,
    index: int,
    presence: Mapping,
) -> None:
    extra_reserved = sorted(
        key
        for key in observation
        if any(
            key.startswith(prefix)
            for prefix in EMBEDDED_OBSERVATION_RESERVED_PREFIXES
        )
        and key not in EMBEDDED_OBSERVATION_RESERVED_FIELDS
    )
    if extra_reserved:
        _fail(f"effect observation {index} has extra reserved fields")
    declared = observation.get("input_row_sha256")
    raw_declared = observation.get("raw_input_row_sha256")
    if type(declared) is not str or not _HEX64.fullmatch(declared):
        _fail(f"effect observation {index} input_row_sha256 is not canonical")
    if type(raw_declared) is not str or not _HEX64.fullmatch(raw_declared):
        _fail(
            f"effect observation {index} raw_input_row_sha256 is not canonical"
        )
    if RAW_INPUT_VALUE_FIELD not in observation:
        _fail(f"effect observation {index} lacks raw input value provenance")
    normalized = dict(observation)
    normalized.pop("input_row_sha256")
    try:
        normalized_actual = canonical_sha256(normalized)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Phase0ProposalError(
            f"effect observation {index} normalized hash replay failed"
        ) from exc
    if normalized_actual != declared:
        _fail(f"effect observation {index} normalized content hash mismatch")

    reconstructed = dict(normalized)
    reconstructed.pop("raw_input_row_sha256")
    raw_value = reconstructed.pop(RAW_INPUT_VALUE_FIELD)
    reconstructed.pop("input_presence_provenance")
    reconstructed["value"] = raw_value
    upstream_marker = object()
    upstream_declared = reconstructed.pop(
        "upstream_declared_input_row_sha256", upstream_marker
    )
    if upstream_declared is not upstream_marker:
        reconstructed["input_row_sha256"] = upstream_declared
    for field in ("assay_context", "replicate"):
        if presence[field] == "ABSENT_CANONICALIZED_TO_NULL":
            reconstructed.pop(field, None)
    try:
        raw_actual = canonical_sha256(reconstructed)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Phase0ProposalError(
            f"effect observation {index} raw hash replay failed"
        ) from exc
    if raw_actual != raw_declared:
        _fail(f"effect observation {index} raw content hash mismatch")


def _validate_effect_observation_presence(row: Mapping) -> list[tuple[str, str]]:
    observations = row.get("functional_observations")
    if type(observations) is not list:
        _fail("effect row lacks functional_observations v2 evidence")
    statuses: list[tuple[str, str]] = []
    for index, observation in enumerate(observations):
        if type(observation) is not dict:
            _fail(f"effect observation {index} must be an object")
        provenance = _expect_object(
            observation.get("input_presence_provenance"),
            INPUT_PRESENCE_PROVENANCE_FIELDS,
            label=f"effect observation {index} input presence",
        )
        if (
            provenance["schema_version"]
            != INPUT_PRESENCE_PROVENANCE_SCHEMA_VERSION
        ):
            _fail("effect observation input-presence schema is stale")
        row_statuses = []
        for field in ("assay_context", "replicate"):
            status = provenance[field]
            if type(status) is not str or status not in INPUT_PRESENCE_STATUSES:
                _fail(f"effect observation {field} presence is invalid")
            if field not in observation:
                _fail(f"effect observation lacks canonicalized {field}")
            actual = observation[field]
            if (
                status == "EXPLICIT_VALUE" and actual is None
            ) or (
                status != "EXPLICIT_VALUE" and actual is not None
            ):
                _fail(f"effect observation {field} presence/value conflicts")
            row_statuses.append(status)
        _validate_embedded_observation_hashes(
            observation,
            index=index,
            presence=provenance,
        )
        statuses.append((row_statuses[0], row_statuses[1]))
    return statuses


def _validate_effect_join_key_presence(
    row: Mapping,
    *,
    comparison_scale_id: str,
    ref_present: bool,
    alt_present: bool,
    observation_presence: Sequence[tuple[str, str]],
) -> tuple[str, str]:
    join_key = _expect_object(
        row.get("observation_join_key"),
        OBSERVATION_JOIN_KEY_FIELDS,
        label="effect row observation_join_key",
    )
    if join_key["comparison_scale_id"] != comparison_scale_id:
        _fail("effect row observation join key scale mismatch")
    assay_status = join_key["assay_context_input_presence"]
    replicate_status = join_key["replicate_input_presence"]
    any_side = ref_present or alt_present
    if any_side:
        if (
            type(assay_status) is not str
            or assay_status not in INPUT_PRESENCE_STATUSES
            or type(replicate_status) is not str
            or replicate_status not in INPUT_PRESENCE_STATUSES
        ):
            _fail("effect row observation join key presence is invalid")
        if join_key["context_id"] is None:
            _fail("effect row presence-aware join context is null")
        for field, status in (
            ("assay_context", assay_status),
            ("replicate", replicate_status),
        ):
            actual = join_key[field]
            if (
                status == "EXPLICIT_VALUE" and actual is None
            ) or (
                status != "EXPLICIT_VALUE" and actual is not None
            ):
                _fail(f"effect row join key {field} presence/value conflicts")
        if not observation_presence or any(
            pair != (assay_status, replicate_status)
            for pair in observation_presence
        ):
            _fail("effect observations disagree with presence-aware join key")
    else:
        if (
            assay_status != NOT_APPLICABLE_PRESENCE_STATUS
            or replicate_status != NOT_APPLICABLE_PRESENCE_STATUS
            or join_key["context_id"] is not None
            or join_key["assay_context"] is not None
            or join_key["replicate"] is not None
            or observation_presence
        ):
            _fail("missing-both effect row has invented presence context")
    return assay_status, replicate_status


def _validate_diff_ref_alt_audit(
    value: object,
    *,
    study: str,
    expected_policy_sha256: str,
) -> dict:
    audit = _expect_object(
        value,
        DIFF_REF_ALT_AUDIT_FIELDS,
        label=f"effect manifest diff audit {study}",
    )
    if (
        audit["schema_version"] != DIFF_REF_ALT_AUDIT_SCHEMA_VERSION
        or audit["verification_status"]
        != "PRESENCE_AWARE_INDEPENDENT_REPLAY_VERIFIED"
    ):
        _fail(
            f"effect manifest diff audit {study} is not v4 "
            "presence-aware replay-verified"
        )
    policy = _validate_diff_ref_alt_policy(
        audit["join_policy"],
        study=study,
        expected_sha256=expected_policy_sha256,
    )
    if (
        audit["join_policy_sha256"] != expected_policy_sha256
        or audit["policy_id"] != policy["policy_id"]
        or audit["comparison_scale_id"] != policy["comparison_scale_id"]
    ):
        _fail(f"effect manifest diff audit {study} policy binding mismatch")
    count_fields = (
        "n_materialized_pairs",
        "n_output_pairs",
        "n_output_join_rows",
        "n_expected_missing_both_pairs",
        "n_consumed_missing_both_rows",
        "n_ref_observation_groups",
        "n_alt_observation_groups",
        "n_input_side_groups",
        "n_consumed_ref_observation_groups",
        "n_consumed_alt_observation_groups",
        "n_consumed_side_groups",
        "n_matched_groups",
        "n_unmatched_ref_groups",
        "n_unmatched_alt_groups",
        "n_missing_both_rows",
        "n_derived_rows",
        "n_unavailable_rows",
        "expected_output_rows_from_group_conservation",
    )
    for field in count_fields:
        _require_nonnegative_int(
            audit[field], label=f"diff audit {study} {field}"
        )
    dispositions = _validate_nonnegative_count_map(
        audit["join_disposition_counts"],
        label=f"diff audit {study} join dispositions",
        allowed_keys=set(DIFF_REF_ALT_DISPOSITIONS),
    )
    for field in (
        "delta_unavailable_reason_counts",
        "replicate_pairing_status_counts",
        "ref_raw_unit_token_group_counts",
        "alt_raw_unit_token_group_counts",
    ):
        _validate_nonnegative_count_map(
            audit[field], label=f"diff audit {study} {field}"
        )
    presence_counts = {}
    for field in (
        "ref_assay_context_input_presence_group_counts",
        "alt_assay_context_input_presence_group_counts",
        "ref_replicate_input_presence_group_counts",
        "alt_replicate_input_presence_group_counts",
    ):
        counts = _validate_nonnegative_count_map(
            audit[field],
            label=f"diff audit {study} {field}",
            allowed_keys=INPUT_PRESENCE_STATUSES,
        )
        if any(count == 0 for count in counts.values()):
            _fail(f"diff audit {study} {field} contains a zero entry")
        presence_counts[field] = counts
    expected_dispositions = {
        "MATCHED": audit["n_matched_groups"],
        "UNMATCHED_REF_ONLY": audit["n_unmatched_ref_groups"],
        "UNMATCHED_ALT_ONLY": audit["n_unmatched_alt_groups"],
        "MISSING_BOTH": audit["n_missing_both_rows"],
    }
    if any(
        dispositions.get(key, 0) != count
        for key, count in expected_dispositions.items()
    ) or set(dispositions) != {
        key for key, count in expected_dispositions.items() if count > 0
    }:
        _fail(f"diff audit {study} disposition counts are inconsistent")
    expected_output = (
        audit["n_ref_observation_groups"]
        + audit["n_alt_observation_groups"]
        - audit["n_matched_groups"]
        + audit["n_expected_missing_both_pairs"]
    )
    if (
        audit["n_materialized_pairs"] != audit["n_output_pairs"]
        or audit["n_expected_missing_both_pairs"]
        != audit["n_consumed_missing_both_rows"]
        or audit["n_expected_missing_both_pairs"]
        != audit["n_missing_both_rows"]
        or audit["n_input_side_groups"]
        != audit["n_ref_observation_groups"]
        + audit["n_alt_observation_groups"]
        or audit["n_consumed_ref_observation_groups"]
        != audit["n_ref_observation_groups"]
        or audit["n_consumed_alt_observation_groups"]
        != audit["n_alt_observation_groups"]
        or audit["n_consumed_side_groups"] != audit["n_input_side_groups"]
        or sum(
            presence_counts[
                "ref_assay_context_input_presence_group_counts"
            ].values()
        ) != audit["n_consumed_ref_observation_groups"]
        or sum(
            presence_counts[
                "ref_replicate_input_presence_group_counts"
            ].values()
        ) != audit["n_consumed_ref_observation_groups"]
        or sum(
            presence_counts[
                "alt_assay_context_input_presence_group_counts"
            ].values()
        ) != audit["n_consumed_alt_observation_groups"]
        or sum(
            presence_counts[
                "alt_replicate_input_presence_group_counts"
            ].values()
        ) != audit["n_consumed_alt_observation_groups"]
        or audit["expected_output_rows_from_group_conservation"]
        != expected_output
        or audit["n_output_join_rows"] != expected_output
        or sum(dispositions.values()) != audit["n_output_join_rows"]
        or audit["n_derived_rows"] != audit["n_matched_groups"]
        or audit["n_unavailable_rows"]
        != audit["n_output_join_rows"] - audit["n_derived_rows"]
    ):
        _fail(f"effect manifest diff audit {study} violates conservation")
    sha_pairs = (
        (
            "expected_input_side_group_multiset_sha256",
            "consumed_output_side_group_multiset_sha256",
        ),
        (
            "expected_missing_both_multiset_sha256",
            "consumed_missing_both_multiset_sha256",
        ),
        ("materialized_pair_registry_sha256", "output_pair_registry_sha256"),
    )
    for expected_field, consumed_field in sha_pairs:
        for field in (expected_field, consumed_field):
            if type(audit[field]) is not str:
                _fail(f"diff audit {study} {field} must be text")
            _require_sha256(audit[field], label=f"diff audit {study} {field}")
        if audit[expected_field] != audit[consumed_field]:
            _fail(f"diff audit {study} replay multiset hashes disagree")
    return audit


def _validate_effect_manifest_contract(value: Mapping) -> tuple[int, dict]:
    if set(value) != EFFECT_MANIFEST_FIELDS:
        _fail(
            "effect manifest top-level fields are not exact v4 "
            "presence-aware replay schema"
        )
    if (
        value["manifest_version"] != EFFECT_DATASET_MANIFEST_VERSION
        or value["effect_dataset_schema_version"]
        != EFFECT_DATASET_SCHEMA_VERSION
        or value["functional_observation_schema_version"]
        != FUNCTIONAL_OBSERVATION_SCHEMA_VERSION
        or value["functional_observation_input_presence_schema_version"]
        != INPUT_PRESENCE_PROVENANCE_SCHEMA_VERSION
    ):
        _fail("effect manifest has a stale schema version")
    n_records = _require_nonnegative_int(
        value["n_records"], label="effect manifest n_records"
    )
    qualification = _expect_object(
        value["qualification"],
        {
            "status",
            "dataset_without_manifest",
            "replay_audit_completed_before_dataset_write",
            "scientific_qualification",
            "training_ready",
            "split_ready",
        },
        label="effect manifest qualification",
    )
    if not _strict_json_equal(qualification, {
        "status": MANIFEST_QUALIFICATION_STATUS,
        "dataset_without_manifest": "NOT_QUALIFIED",
        "replay_audit_completed_before_dataset_write": True,
        "scientific_qualification": False,
        "training_ready": False,
        "split_ready": False,
    }):
        _fail(
            "effect manifest qualification is not exact v4 "
            "presence-aware replay-only"
        )
    implementation = _expect_object(
        value["implementation_binding"],
        IMPLEMENTATION_BINDING_FIELDS,
        label="effect manifest implementation_binding",
    )
    if (
        implementation["schema_version"]
        != IMPLEMENTATION_BINDING_SCHEMA_VERSION
        or implementation["effect_dataset_schema_version"]
        != EFFECT_DATASET_SCHEMA_VERSION
        or implementation["manifest_version"]
        != EFFECT_DATASET_MANIFEST_VERSION
        or implementation["diff_ref_alt_join_policy_schema_version"]
        != DIFF_REF_ALT_POLICY_SCHEMA_VERSION
        or implementation["diff_ref_alt_replay_audit_schema_version"]
        != DIFF_REF_ALT_AUDIT_SCHEMA_VERSION
        or implementation[
            "functional_observation_input_presence_schema_version"
        ] != INPUT_PRESENCE_PROVENANCE_SCHEMA_VERSION
        or implementation["ordinary_only"] is not True
        or implementation["scientific_unit_authority_created"] is not False
        or implementation["estimand_authority_created"] is not False
        or implementation["diff_ref_alt_policy_sha256_by_study"]
        != REQUIRED_DIFF_REF_ALT_POLICY_SHA256
    ):
        _fail("effect manifest implementation binding is stale or unsafe")
    for field in ("build_effect_dataset_path", "config_path"):
        _require_safe_text(
            implementation[field], label=f"implementation binding {field}"
        )
    for field in (
        "build_effect_dataset_sha256",
        "config_sha256",
        "delta_specs_sha256",
    ):
        if type(implementation[field]) is not str:
            _fail(f"implementation binding {field} must be text")
        _require_sha256(implementation[field], label=f"implementation {field}")
    audits = _expect_object(
        value["diff_ref_alt_join_audits"],
        set(REQUIRED_DIFF_REF_ALT_POLICY_SHA256),
        label="effect manifest diff_ref_alt_join_audits",
    )
    validated = {
        study: _validate_diff_ref_alt_audit(
            audits[study],
            study=study,
            expected_policy_sha256=expected_sha,
        )
        for study, expected_sha in sorted(
            REQUIRED_DIFF_REF_ALT_POLICY_SHA256.items()
        )
    }
    return n_records, validated


def build_phase0_proposal(
    *,
    effect_path: Path,
    effect_sha256: str,
    effect_manifest_path: Path,
    effect_manifest_sha256: str,
    d1_sha256s_path: Path,
    d1_sha256s_sha256: str,
    d1_paths: Mapping[str, Path],
    d1_expected_sha256: Mapping[str, str],
) -> dict:
    """Return one canonical non-active proposal from exact ordinary inputs."""

    effect_path = Path(effect_path)
    effect_manifest_path = Path(effect_manifest_path)
    d1_sha256s_path = Path(d1_sha256s_path)
    _assert_phase0_path(effect_path, purpose="effect dataset input")
    _assert_phase0_path(effect_manifest_path, purpose="effect manifest input")
    _assert_phase0_path(d1_sha256s_path, purpose="D1 SHA-256 manifest input")
    effect_sha = _require_sha256(effect_sha256, label="effect dataset SHA-256")
    effect_manifest_sha = _require_sha256(
        effect_manifest_sha256, label="effect manifest SHA-256"
    )
    d1_sums_sha = _require_sha256(
        d1_sha256s_sha256, label="D1_SHA256SUMS SHA-256"
    )
    normalized_paths, normalized_hashes = _validate_input_maps(
        d1_paths, d1_expected_sha256
    )

    effect_manifest, effect_manifest_stats = _read_exact_json(
        effect_manifest_path,
        effect_manifest_sha,
        purpose="effect dataset manifest",
    )
    if (
        type(effect_manifest.get("sha256")) is not str
        or effect_manifest["sha256"] != effect_sha
    ):
        _fail("effect manifest does not bind the exact effect dataset SHA-256")
    manifest_n_records, manifest_diff_audits = (
        _validate_effect_manifest_contract(effect_manifest)
    )
    manifest_inputs = effect_manifest.get("inputs")
    if not isinstance(manifest_inputs, Mapping):
        _fail("effect manifest lacks inputs")
    expected_manifest_inputs = {
        "pairs": normalized_hashes["utr_edit_pairs"],
        "functional_observations": normalized_hashes["functional_observations"],
    }
    for name, expected in expected_manifest_inputs.items():
        entry = manifest_inputs.get(name)
        if (
            not isinstance(entry, Mapping)
            or type(entry.get("sha256")) is not str
            or entry["sha256"] != expected
        ):
            _fail(f"effect manifest does not bind exact {name} SHA-256")

    sums_raw = _read_exact_bytes(
        d1_sha256s_path, d1_sums_sha, purpose="D1_SHA256SUMS"
    )
    sums = _parse_d1_sha256s(sums_raw)
    for logical in JSONL_D1_AUTHORITIES:
        filename = D1_FILE_NAMES[logical]
        if sums.get(filename) != normalized_hashes[logical]:
            _fail(
                f"D1_SHA256SUMS does not bind exact {logical} SHA-256"
            )

    grouping_projection, grouping_projection_stats = _read_exact_json(
        normalized_paths["grouping_atom_projection"],
        normalized_hashes["grouping_atom_projection"],
        purpose="grouping atom projection",
    )
    projection_algorithm = _category(
        grouping_projection.get("assignment_algorithm_id")
    )
    if not projection_algorithm:
        _fail("grouping atom projection lacks assignment_algorithm_id")

    counters: Counter = Counter()
    context_missing: Counter = Counter()
    edit_counts: Counter = Counter()
    pair_relation_status: Counter = Counter()
    d1_pair_relation_status: Counter = Counter()
    d1_pair_relation_status_source: Counter = Counter()
    functional_missing: Counter = Counter()
    diff_disposition_counts = {
        study: Counter() for study in REQUIRED_DIFF_REF_ALT_POLICY_SHA256
    }
    diff_presence_counts = {
        study: {
            field: Counter()
            for field in (
                "ref_assay_context_input_presence_group_counts",
                "alt_assay_context_input_presence_group_counts",
                "ref_replicate_input_presence_group_counts",
                "alt_replicate_input_presence_group_counts",
            )
        }
        for study in REQUIRED_DIFF_REF_ALT_POLICY_SHA256
    }
    input_stats: dict[str, dict] = {}

    with tempfile.TemporaryDirectory(prefix="r3_phase0_proposal_") as tmp_name:
        database_path = Path(tmp_name) / "mechanical_inventory.sqlite3"
        connection = sqlite3.connect(database_path)
        try:
            _initialize_database(connection)

            def visit_effect(row: dict) -> None:
                counters["effect_rows"] += 1
                if (
                    row.get("effect_dataset_schema_version")
                    != EFFECT_DATASET_SCHEMA_VERSION
                    or row.get("qualification_status")
                    != EFFECT_QUALIFICATION_STATUS
                    or row.get("functional_observation_schema_version")
                    != FUNCTIONAL_OBSERVATION_SCHEMA_VERSION
                ):
                    _fail("effect row has a stale or unsafe v4 schema binding")
                observation_presence = _validate_effect_observation_presence(
                    row
                )
                counters["effect_observation_rows_presence_verified"] += len(
                    observation_presence
                )
                study = _require_safe_text(
                    row.get("study"), label="effect row study"
                )
                pair = _pair_hash(row.get("pair_id"))
                if pair is None:
                    counters["effect_rows_missing_pair"] += 1
                else:
                    cursor = connection.execute(
                        "INSERT OR IGNORE INTO effect_pairs VALUES (?)", (pair,)
                    )
                    if cursor.rowcount == 0:
                        counters["duplicate_effect_pair_rows"] += 1
                if study in REQUIRED_DIFF_REF_ALT_POLICY_SHA256:
                    if pair is None:
                        _fail("diff_ref_alt effect row lacks pair identity")
                    audit = manifest_diff_audits[study]
                    if (
                        row.get("diff_ref_alt_join_policy_schema_version")
                        != DIFF_REF_ALT_POLICY_SCHEMA_VERSION
                        or row.get("diff_ref_alt_join_policy_sha256")
                        != REQUIRED_DIFF_REF_ALT_POLICY_SHA256[study]
                        or row.get("diff_ref_alt_join_policy_id")
                        != audit["policy_id"]
                        or row.get("comparison_scale_id")
                        != audit["comparison_scale_id"]
                    ):
                        _fail("diff_ref_alt effect row policy binding mismatch")
                    disposition = row.get("diff_ref_alt_join_disposition")
                    if disposition not in DIFF_REF_ALT_DISPOSITIONS:
                        _fail("diff_ref_alt effect row has invalid disposition")
                    ref_present = row.get("ref_observation_group_present")
                    alt_present = row.get("alt_observation_group_present")
                    if (
                        type(ref_present) is not bool
                        or type(alt_present) is not bool
                        or (ref_present, alt_present)
                        != DIFF_REF_ALT_DISPOSITIONS[disposition]
                    ):
                        _fail(
                            "diff_ref_alt effect row disposition/presence mismatch"
                        )
                    assay_presence, replicate_presence = (
                        _validate_effect_join_key_presence(
                            row,
                            comparison_scale_id=audit[
                                "comparison_scale_id"
                            ],
                            ref_present=ref_present,
                            alt_present=alt_present,
                            observation_presence=observation_presence,
                        )
                    )
                    connection.execute(
                        "INSERT OR IGNORE INTO effect_diff_pairs VALUES (?, ?)",
                        (study, pair),
                    )
                    diff_disposition_counts[study][disposition] += 1
                    counters[f"diff_effect_rows:{study}"] += 1
                    counters[f"diff_ref_present:{study}"] += int(ref_present)
                    counters[f"diff_alt_present:{study}"] += int(alt_present)
                    if ref_present:
                        diff_presence_counts[study][
                            "ref_assay_context_input_presence_group_counts"
                        ][assay_presence] += 1
                        diff_presence_counts[study][
                            "ref_replicate_input_presence_group_counts"
                        ][replicate_presence] += 1
                    if alt_present:
                        diff_presence_counts[study][
                            "alt_assay_context_input_presence_group_counts"
                        ][assay_presence] += 1
                        diff_presence_counts[study][
                            "alt_replicate_input_presence_group_counts"
                        ][replicate_presence] += 1
                key = {
                    field: _category(row.get(field)) for field in CONTEXT_KEY_FIELDS
                }
                key["library"] = _opaque_library_value(row.get("library"))
                for field, value in key.items():
                    if value is None:
                        context_missing[field] += 1
                aliases = [
                    value
                    for value in (
                        _category(row.get("context_id")),
                        _category(row.get("assay_context")),
                        _category(row.get("observation_context_id")),
                        _category(row.get("observation_assay_context")),
                    )
                    if value is not None
                ]
                _increment_context(connection, key, aliases)
                _increment_estimand_candidate(
                    connection,
                    study=key["study"],
                    endpoint=key["endpoint"],
                    observed_unit=_category(row.get("unit")),
                )
                edit_kind = _edit_script_kind(row.get("edit_list"))
                counters[f"edit_script:{edit_kind}"] += 1
                edit_count = row.get("edit_count")
                if isinstance(edit_count, int) and not isinstance(edit_count, bool):
                    edit_counts[str(edit_count)] += 1
                else:
                    edit_counts["MISSING_OR_INVALID"] += 1
                if _category(row.get("source_sequence")) is not None and _category(
                    row.get("candidate_sequence")
                ) is not None:
                    counters["rows_with_both_sequence_fields"] += 1
                provenance = row.get("pair_provenance")
                status_value = (
                    _category(provenance.get("relation_status"))
                    if isinstance(provenance, Mapping)
                    else None
                )
                pair_relation_status[status_value or "MISSING"] += 1

            input_stats["effect_dataset"] = _scan_jsonl(
                effect_path,
                effect_sha,
                purpose="effect dataset",
                visitor=visit_effect,
            )
            if manifest_n_records != input_stats["effect_dataset"][
                "json_row_count"
            ]:
                _fail("effect manifest n_records disagrees with streamed rows")
            for study, audit in sorted(manifest_diff_audits.items()):
                unique_pairs = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM effect_diff_pairs WHERE study = ?",
                        (study,),
                    ).fetchone()[0]
                )
                if (
                    audit["n_output_join_rows"]
                    != int(counters.get(f"diff_effect_rows:{study}", 0))
                    or audit["n_output_pairs"] != unique_pairs
                    or audit["join_disposition_counts"]
                    != dict(sorted(diff_disposition_counts[study].items()))
                    or audit["n_consumed_ref_observation_groups"]
                    != int(counters.get(f"diff_ref_present:{study}", 0))
                    or audit["n_consumed_alt_observation_groups"]
                    != int(counters.get(f"diff_alt_present:{study}", 0))
                    or any(
                        audit[field]
                        != dict(sorted(counts.items()))
                        for field, counts in diff_presence_counts[
                            study
                        ].items()
                    )
                ):
                    _fail(
                        "effect rows disagree with manifest diff_ref_alt "
                        f"presence-aware mechanical audit for {study}"
                    )

            def visit_pair(row: dict) -> None:
                counters["d1_pair_rows"] += 1
                pair = _pair_hash(row.get("pair_id"))
                if pair is None:
                    counters["d1_pair_rows_missing_identity"] += 1
                    return
                raw_canonical = row.get(RELATION_STATUS_AUTHORITY_FIELD)
                if raw_canonical is None:
                    canonical = None
                elif (
                    type(raw_canonical) is not str
                    or not raw_canonical
                    or raw_canonical.strip() != raw_canonical
                ):
                    _fail(
                        "canonical relation_acceptance_status must be an "
                        "original trim-equal non-empty string"
                    )
                else:
                    canonical = raw_canonical
                # Legacy status is only a fail-closed consistency check; it
                # never resolves or promotes the canonical authority.
                legacy = _category(row.get("relation_status"))
                if canonical is None:
                    resolved = "UNREVIEWED"
                    source_field = MISSING_RELATION_STATUS_SOURCE
                else:
                    if canonical not in RELATION_ACCEPTANCE_STATUS_ALLOWLIST:
                        _fail(
                            "unsupported canonical relation_acceptance_status: "
                            f"{canonical}"
                        )
                    resolved = canonical
                    source_field = RELATION_STATUS_AUTHORITY_FIELD
                    if legacy is not None and legacy != resolved:
                        _fail(
                            "canonical relation_acceptance_status "
                            f"{canonical} conflicts with legacy relation_status "
                            f"{legacy}"
                        )
                accepted = int(resolved == "ACCEPTED")
                previous = connection.execute(
                    "SELECT resolved_status, status_source_field, accepted "
                    "FROM d1_pairs WHERE pair_hash = ?",
                    (pair,),
                ).fetchone()
                if previous is not None:
                    counters["duplicate_d1_pair_rows"] += 1
                    if previous != (resolved, source_field, accepted):
                        counters["conflicting_d1_pair_status"] += 1
                        _fail(
                            "conflicting canonical relation status across "
                            "duplicate D1 pair rows"
                        )
                    return
                connection.execute(
                    "INSERT INTO d1_pairs VALUES (?, ?, ?, ?)",
                    (pair, resolved, source_field, accepted),
                )
                d1_pair_relation_status[resolved] += 1
                d1_pair_relation_status_source[source_field] += 1
                if _category(row.get("source_sequence_id")) is None:
                    counters["d1_pairs_missing_source_sequence_identity"] += 1
                if _category(row.get("candidate_sequence_id")) is None:
                    counters["d1_pairs_missing_candidate_sequence_identity"] += 1

            input_stats["utr_edit_pairs"] = _scan_jsonl(
                normalized_paths["utr_edit_pairs"],
                normalized_hashes["utr_edit_pairs"],
                purpose="D1 utr_edit_pairs",
                visitor=visit_pair,
            )

            def visit_observation(row: dict) -> None:
                counters["functional_observation_rows"] += 1
                for field in ("endpoint_id", "context_id", "replicate", "unit"):
                    if _category(row.get(field)) is None:
                        functional_missing[field] += 1
                # Numeric values and outcome identities are intentionally not
                # inspected, summarized, hashed into the inventory, or emitted.

            input_stats["functional_observations"] = _scan_jsonl(
                normalized_paths["functional_observations"],
                normalized_hashes["functional_observations"],
                purpose="D1 functional_observations",
                visitor=visit_observation,
            )

            def visit_group(row: dict) -> None:
                counters["group_registry_rows"] += 1
                atom = _category(row.get("grouping_atom"))
                group_id = _category(row.get("group_id"))
                members = row.get("member_ids")
                group_key = _group_hash(atom, group_id)
                if atom is None or group_id is None or group_key is None or not isinstance(
                    members, list
                ):
                    counters["invalid_group_registry_rows"] += 1
                    return
                normalized_members = sorted(
                    set(
                        item
                        for item in members
                        if isinstance(item, str) and item.strip() == item and item
                    )
                )
                member_ids_valid = int(
                    len(normalized_members) == len(members)
                    and members == normalized_members
                )
                if not member_ids_valid:
                    counters["registry_member_ids_invalid"] += 1
                material = {
                    "group_id": group_id,
                    "grouping_atom": atom,
                    "member_ids": normalized_members,
                }
                declared_sha = _category(row.get("group_sha256"))
                valid = int(declared_sha == canonical_sha256(material))
                if not valid:
                    counters["registry_group_sha_mismatch"] += 1
                previous = connection.execute(
                    "SELECT declared_member_count, member_ids_valid, "
                    "group_sha_valid, declared_group_sha256 FROM groups "
                    "WHERE atom = ? AND group_hash = ?",
                    (atom, group_key),
                ).fetchone()
                if previous is not None:
                    counters["duplicate_group_registry_rows"] += 1
                    if previous != (
                        len(normalized_members),
                        member_ids_valid,
                        valid,
                        declared_sha,
                    ):
                        counters["conflicting_group_registry_rows"] += 1
                    return
                connection.execute(
                    "INSERT INTO groups VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        atom,
                        group_key,
                        len(normalized_members),
                        member_ids_valid,
                        valid,
                        declared_sha,
                    ),
                )
                if member_ids_valid and valid:
                    connection.executemany(
                        """
                        INSERT OR IGNORE INTO registry_pair_members
                            (pair_hash, atom, group_hash)
                        SELECT ?, ?, ?
                        WHERE EXISTS (
                            SELECT 1 FROM effect_pairs WHERE pair_hash = ?
                        )
                        """,
                        (
                            (pair, atom, group_key, pair)
                            for pair in (
                                _pair_hash(member) for member in normalized_members
                            )
                            if pair is not None
                        ),
                    )

            input_stats["group_registry"] = _scan_jsonl(
                normalized_paths["group_registry"],
                normalized_hashes["group_registry"],
                purpose="D1 group_registry",
                visitor=visit_group,
            )

            def visit_assignment(row: dict) -> None:
                counters["group_assignment_rows"] += 1
                if _category(row.get("object_type")) != "PAIR":
                    return
                pair = _pair_hash(row.get("object_id"))
                atom = _category(row.get("grouping_atom"))
                group_key = _group_hash(atom, row.get("group_id"))
                if pair is None or atom is None or group_key is None:
                    counters["invalid_pair_group_assignments"] += 1
                    return
                if connection.execute(
                    "SELECT 1 FROM effect_pairs WHERE pair_hash = ?", (pair,)
                ).fetchone() is None:
                    return
                counters["effect_pair_group_assignments"] += 1
                if _category(row.get("assignment_algorithm_id")) != projection_algorithm:
                    counters["assignment_algorithm_mismatch"] += 1
                    return
                registry_group = connection.execute(
                    "SELECT member_ids_valid, group_sha_valid FROM groups "
                    "WHERE atom = ? AND group_hash = ?",
                    (atom, group_key),
                ).fetchone()
                if registry_group is None:
                    counters["assignment_missing_registry_group"] += 1
                    return
                if registry_group != (1, 1):
                    counters["assignment_registry_group_invalid"] += 1
                    return
                if connection.execute(
                    "SELECT 1 FROM registry_pair_members "
                    "WHERE pair_hash = ? AND atom = ? AND group_hash = ?",
                    (pair, atom, group_key),
                ).fetchone() is None:
                    counters["assignment_pair_not_registry_member"] += 1
                    return
                connection.execute(
                    "INSERT OR IGNORE INTO pair_atoms VALUES (?, ?, ?)",
                    (pair, atom, group_key),
                )

            input_stats["group_assignments"] = _scan_jsonl(
                normalized_paths["group_assignments"],
                normalized_hashes["group_assignments"],
                purpose="D1 group_assignments",
                visitor=visit_assignment,
            )

            def visit_sequence(row: dict) -> None:
                counters["sequence_entity_rows"] += 1
                if _category(row.get("sequence_id")) is None:
                    counters["sequence_entities_missing_identity"] += 1
                if _category(row.get("normalized_sequence")) is None and _category(
                    row.get("normalized_sequence_sha256")
                ) is None:
                    counters["sequence_entities_missing_normalized_support"] += 1

            input_stats["sequence_entities"] = _scan_jsonl(
                normalized_paths["sequence_entities"],
                normalized_hashes["sequence_entities"],
                purpose="D1 sequence_entities",
                visitor=visit_sequence,
            )

            def visit_attributes(row: dict) -> None:
                counters["object_attribute_rows"] += 1
                if _category(row.get("object_type")) != "PAIR":
                    return
                pair = _pair_hash(row.get("object_id"))
                if pair is None or connection.execute(
                    "SELECT 1 FROM effect_pairs WHERE pair_hash = ?", (pair,)
                ).fetchone() is None:
                    return
                groups_by_atom = row.get("group_ids_by_atom")
                if not isinstance(groups_by_atom, Mapping):
                    counters["invalid_pair_object_attributes"] += 1
                    return
                for atom_value, raw_groups in groups_by_atom.items():
                    atom = _category(atom_value)
                    if atom is None or not isinstance(raw_groups, list):
                        counters["invalid_pair_object_attribute_atoms"] += 1
                        continue
                    for raw_group in raw_groups:
                        group_key = _group_hash(atom, raw_group)
                        if group_key is not None:
                            connection.execute(
                                "INSERT OR IGNORE INTO attribute_atoms VALUES (?, ?, ?)",
                                (pair, atom, group_key),
                            )

            input_stats["object_attributes"] = _scan_jsonl(
                normalized_paths["object_attributes"],
                normalized_hashes["object_attributes"],
                purpose="D1 object_attributes",
                visitor=visit_attributes,
            )
            connection.commit()

            context_inventory = _context_inventory(connection, context_missing)
            estimand_queue = _estimand_queue(connection)
            relation_inventory = _relation_inventory(
                connection,
                counters=counters,
                d1_status_counts=d1_pair_relation_status,
                d1_status_source_counts=d1_pair_relation_status_source,
                projection=grouping_projection,
            )
            unique_effect_pairs = int(
                connection.execute("SELECT COUNT(*) FROM effect_pairs").fetchone()[0]
            )
        finally:
            connection.close()

    input_authorities = {
        "effect_dataset": _authority_entry(
            "effect_dataset", effect_path, input_stats["effect_dataset"]
        ),
        "effect_dataset_manifest": _authority_entry(
            "effect_dataset_manifest", effect_manifest_path, effect_manifest_stats
        ),
        "d1_sha256s": _authority_entry(
            "D1_SHA256SUMS",
            d1_sha256s_path,
            {
                "artifact_format": "SHA256SUMS",
                "byte_count": len(sums_raw),
                "entry_count": len(sums),
                "sha256": d1_sums_sha,
            },
        ),
    }
    for logical in JSONL_D1_AUTHORITIES:
        input_authorities[logical] = _authority_entry(
            logical, normalized_paths[logical], input_stats[logical]
        )
    input_authorities["grouping_atom_projection"] = _authority_entry(
        "grouping_atom_projection",
        normalized_paths["grouping_atom_projection"],
        grouping_projection_stats,
    )
    input_authorities = {
        key: input_authorities[key] for key in sorted(input_authorities)
    }

    binding_checks = dict(BINDING_CHECKS)
    authority_binding_material = {
        "schema_version": BINDING_SCHEMA_VERSION,
        "input_authorities": input_authorities,
        "binding_checks": binding_checks,
    }
    authority_binding_sha = canonical_sha256(authority_binding_material)

    mechanical_inventory = {
        "status": STATUS,
        "effect_row_count": int(counters.get("effect_rows", 0)),
        "unique_effect_pair_count": unique_effect_pairs,
        "effect_rows_missing_pair_count": int(
            counters.get("effect_rows_missing_pair", 0)
        ),
        "duplicate_effect_pair_row_count": int(
            counters.get("duplicate_effect_pair_rows", 0)
        ),
        "rows_with_both_sequence_fields_present_count": int(
            counters.get("rows_with_both_sequence_fields", 0)
        ),
        "canonical_edit_script_row_count": int(
            counters.get("edit_script:canonical_op_pos_ref_alt", 0)
        ),
        "legacy_token_edit_script_row_count": int(
            counters.get("edit_script:legacy_pos_token", 0)
        ),
        "other_or_missing_edit_script_row_count": int(
            counters.get("edit_script:other_or_mixed", 0)
            + counters.get("edit_script:missing_or_empty", 0)
        ),
        "edit_count_histogram": dict(sorted(edit_counts.items())),
        "pair_relation_status_counts": dict(sorted(pair_relation_status.items())),
        "functional_observation_row_count": int(
            counters.get("functional_observation_rows", 0)
        ),
        "functional_observation_missing_field_counts": {
            field: int(functional_missing.get(field, 0))
            for field in ("endpoint_id", "context_id", "replicate", "unit")
        },
        "sequence_entity_row_count": int(
            counters.get("sequence_entity_rows", 0)
        ),
        "sequence_entity_missing_normalized_support_count": int(
            counters.get("sequence_entities_missing_normalized_support", 0)
        ),
        "raw_sequence_values_emitted": False,
        "identity_values_emitted": False,
        "member_lists_emitted": False,
        "outcome_fields_used_for_inventory": False,
        "numeric_outcome_aggregates_emitted": False,
        "mechanical_inventory_interpretation": (
            "PROPOSAL_COUNTS_ONLY_NO_SCIENTIFIC_ADMISSION"
        ),
    }
    inventory_policy = dict(INVENTORY_POLICY)
    outcome_free_material = {
        "schema_version": OUTCOME_COMMITMENT_SCHEMA_VERSION,
        "inventory_policy": inventory_policy,
        "context_candidate_inventory": context_inventory,
        "estimand_review_queue": estimand_queue,
        "relation_atom_inventory": relation_inventory,
        "mechanical_candidate_inventory": mechanical_inventory,
    }
    proposal_without_hash = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": ARTIFACT_ROLE,
        "status": STATUS,
        "training_ready": False,
        "phase0_ready": False,
        "split_created": False,
        "ordinary_only": True,
        "sealed_access": False,
        "internal_test_access": False,
        "downstream_unlock": False,
        "input_authorities": input_authorities,
        "binding_checks": binding_checks,
        "authority_binding_sha256": authority_binding_sha,
        "inventory_policy": inventory_policy,
        "context_candidate_inventory": context_inventory,
        "estimand_review_queue": estimand_queue,
        "relation_atom_inventory": relation_inventory,
        "mechanical_candidate_inventory": mechanical_inventory,
        "outcome_free_inventory_sha256": canonical_sha256(outcome_free_material),
        "human_review_required": list(HUMAN_REVIEW_REQUIRED),
        "proposal_limitations": list(PROPOSAL_LIMITATIONS),
    }
    proposal = {
        **proposal_without_hash,
        "proposal_sha256": canonical_sha256(proposal_without_hash),
    }
    return validate_phase0_proposal(proposal)


def _walk_output(value: object, path: str = "$"):
    if isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: str(item)):
            child = value[key]
            child_path = f"{path}.{key}"
            yield child_path, str(key), child
            yield from _walk_output(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_output(child, f"{path}[{index}]")


def _expect_object(value: object, fields: set[str], *, label: str) -> dict:
    if type(value) is not dict or set(value) != fields:
        _fail(f"{label} fields differ from the exact schema")
    return value


def _strict_json_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return set(actual) == set(expected) and all(
            _strict_json_equal(actual[key], expected[key]) for key in expected
        )
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            _strict_json_equal(left, right)
            for left, right in zip(actual, expected)
        )
    return actual == expected


def _expect_list(value: object, *, label: str) -> list:
    if type(value) is not list:
        _fail(f"{label} must be a JSON array")
    return value


def _require_nonnegative_int(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _fail(f"{label} must be a non-boolean non-negative integer")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    result = _require_nonnegative_int(value, label=label)
    if result == 0:
        _fail(f"{label} must be positive")
    return result


def _require_optional_safe_text(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value or value.strip() != value:
        _fail(f"{label} must be null or canonical non-empty text")
    if (
        _unsafe_text(value)
        or _RAW_IDENTITY_TEXT_RE.search(value)
        or _RAW_OUTCOME_TEXT_RE.search(value)
        or _RAW_SEQUENCE_TEXT_RE.search(value)
    ):
        _fail(f"{label} contains protected raw identity/outcome text")
    return value


def _require_safe_text(value: object, *, label: str) -> str:
    result = _require_optional_safe_text(value, label=label)
    if result is None:
        _fail(f"{label} must be canonical non-empty text")
    return result


def _validate_nonnegative_count_map(
    value: object,
    *,
    label: str,
    allowed_keys: set[str] | None = None,
) -> dict:
    if type(value) is not dict:
        _fail(f"{label} must be an object")
    if allowed_keys is not None and not set(value).issubset(allowed_keys):
        _fail(f"{label} contains a non-allowlisted key")
    for key, count in value.items():
        if type(key) is not str or not key:
            _fail(f"{label} contains a non-canonical key")
        _require_nonnegative_int(count, label=f"{label}.{key}")
    return value


def _validate_input_authorities(value: object) -> dict:
    expected_names = {
        "effect_dataset",
        "effect_dataset_manifest",
        "d1_sha256s",
        *D1_FILE_NAMES.keys(),
    }
    authorities = _expect_object(
        value, expected_names, label="input_authorities"
    )
    jsonl_basenames = {
        "effect_dataset": "effect_dataset.jsonl",
        **{
            logical: D1_FILE_NAMES[logical]
            for logical in JSONL_D1_AUTHORITIES
        },
    }
    json_basenames = {
        "effect_dataset_manifest": "effect_dataset_manifest.json",
        "grouping_atom_projection": D1_FILE_NAMES[
            "grouping_atom_projection"
        ],
    }
    for logical, basename in jsonl_basenames.items():
        authority = _expect_object(
            authorities[logical],
            {
                "logical_name",
                "basename",
                "artifact_format",
                "byte_count",
                "json_row_count",
                "sha256",
                "streaming_scan",
            },
            label=f"input_authorities.{logical}",
        )
        if (
            authority["logical_name"] != logical
            or authority["basename"] != basename
            or authority["artifact_format"] != "JSONL"
            or authority["streaming_scan"] is not True
        ):
            _fail(f"input authority {logical} has non-canonical fixed fields")
        _require_nonnegative_int(
            authority["byte_count"], label=f"{logical}.byte_count"
        )
        _require_nonnegative_int(
            authority["json_row_count"], label=f"{logical}.json_row_count"
        )
        if type(authority["sha256"]) is not str:
            _fail(f"{logical}.sha256 must be text")
        _require_sha256(authority["sha256"], label=f"{logical}.sha256")
    for logical, basename in json_basenames.items():
        authority = _expect_object(
            authorities[logical],
            {
                "logical_name",
                "basename",
                "artifact_format",
                "byte_count",
                "sha256",
            },
            label=f"input_authorities.{logical}",
        )
        if (
            authority["logical_name"] != logical
            or authority["basename"] != basename
            or authority["artifact_format"] != "JSON"
        ):
            _fail(f"input authority {logical} has non-canonical fixed fields")
        _require_nonnegative_int(
            authority["byte_count"], label=f"{logical}.byte_count"
        )
        if type(authority["sha256"]) is not str:
            _fail(f"{logical}.sha256 must be text")
        _require_sha256(authority["sha256"], label=f"{logical}.sha256")
    sums = _expect_object(
        authorities["d1_sha256s"],
        {
            "logical_name",
            "basename",
            "artifact_format",
            "byte_count",
            "entry_count",
            "sha256",
        },
        label="input_authorities.d1_sha256s",
    )
    if (
        sums["logical_name"] != "D1_SHA256SUMS"
        or sums["basename"] != "D1_SHA256SUMS"
        or sums["artifact_format"] != "SHA256SUMS"
    ):
        _fail("D1_SHA256SUMS authority has non-canonical fixed fields")
    _require_nonnegative_int(sums["byte_count"], label="d1_sha256s.byte_count")
    _require_nonnegative_int(sums["entry_count"], label="d1_sha256s.entry_count")
    if type(sums["sha256"]) is not str:
        _fail("d1_sha256s.sha256 must be text")
    _require_sha256(sums["sha256"], label="d1_sha256s.sha256")
    return authorities


def _validate_context_inventory(value: object) -> tuple[dict, int]:
    context = _expect_object(
        value,
        {
            "status",
            "context_authority_created",
            "context_decision",
            "library_missing_policy",
            "library_value_projection",
            "candidate_count",
            "candidate_conflict_count",
            "missing_field_counts",
            "candidates",
        },
        label="context_candidate_inventory",
    )
    if (
        context["status"] != STATUS
        or context["context_authority_created"] is not False
        or context["context_decision"] != "NONE"
        or context["library_missing_policy"]
        != "PRESERVE_NULL_NO_IMPUTATION"
        or context["library_value_projection"] != LIBRARY_VALUE_PROJECTION
    ):
        _fail("context inventory attempted to exceed proposal scope")
    candidate_count = _require_nonnegative_int(
        context["candidate_count"], label="context candidate_count"
    )
    conflict_count = _require_nonnegative_int(
        context["candidate_conflict_count"],
        label="context candidate_conflict_count",
    )
    missing = _expect_object(
        context["missing_field_counts"],
        set(CONTEXT_KEY_FIELDS),
        label="context missing_field_counts",
    )
    for field in CONTEXT_KEY_FIELDS:
        _require_nonnegative_int(
            missing[field], label=f"context missing_field_counts.{field}"
        )
    candidates = _expect_list(context["candidates"], label="context candidates")
    if candidate_count != len(candidates):
        _fail("context candidate_count does not match candidates")
    observed_hashes = []
    observed_conflicts = 0
    row_total = 0
    for index, raw_candidate in enumerate(candidates):
        candidate = _expect_object(
            raw_candidate,
            {
                "candidate_key_sha256",
                "mechanical_key",
                "row_count",
                "observed_input_context_alias_count",
                "context_conflict_candidate",
                "review_status",
            },
            label=f"context candidates[{index}]",
        )
        key = _expect_object(
            candidate["mechanical_key"],
            set(CONTEXT_KEY_FIELDS),
            label=f"context candidates[{index}].mechanical_key",
        )
        for field in CONTEXT_KEY_FIELDS:
            _require_optional_safe_text(
                key[field], label=f"context candidate {index} {field}"
            )
        declared_sha = candidate["candidate_key_sha256"]
        if type(declared_sha) is not str:
            _fail("context candidate SHA-256 must be text")
        _require_sha256(declared_sha, label="context candidate SHA-256")
        expected_sha = canonical_sha256(
            {"schema_version": CONTEXT_CANDIDATE_KEY_SCHEMA_VERSION, **key}
        )
        if declared_sha != expected_sha:
            _fail("context candidate key SHA-256 is inconsistent")
        row_count = _require_positive_int(
            candidate["row_count"], label="context candidate row_count"
        )
        alias_count = _require_nonnegative_int(
            candidate["observed_input_context_alias_count"],
            label="context candidate alias count",
        )
        conflict = candidate["context_conflict_candidate"]
        if type(conflict) is not bool or conflict is not (alias_count > 1):
            _fail("context conflict flag disagrees with alias count")
        if candidate["review_status"] != "HUMAN_REVIEW_REQUIRED":
            _fail("context candidates must remain human-review proposals")
        observed_hashes.append(declared_sha)
        observed_conflicts += int(conflict)
        row_total += row_count
    if observed_hashes != sorted(set(observed_hashes)):
        _fail("context candidates must be unique and hash-sorted")
    if conflict_count != observed_conflicts:
        _fail("context candidate conflict count is inconsistent")
    return context, row_total


def _validate_estimand_queue(value: object) -> tuple[list, int]:
    queue = _expect_list(value, label="estimand_review_queue")
    observed_hashes = []
    row_total = 0
    families = {
        "RIBOSOME_LOADING_MRL",
        "RNA_HALF_LIFE",
        "UNCLASSIFIED_ENDPOINT",
    }
    for index, raw_item in enumerate(queue):
        item = _expect_object(
            raw_item,
            {
                "candidate_key_sha256",
                "study",
                "endpoint",
                "observed_unit",
                "row_count",
                "mechanical_endpoint_family_hint",
                "review_status",
                "estimand_decision",
                "required_human_review_fields",
            },
            label=f"estimand_review_queue[{index}]",
        )
        study = _require_optional_safe_text(
            item["study"], label=f"estimand {index} study"
        )
        endpoint = _require_optional_safe_text(
            item["endpoint"], label=f"estimand {index} endpoint"
        )
        unit = _require_optional_safe_text(
            item["observed_unit"], label=f"estimand {index} observed_unit"
        )
        declared_sha = item["candidate_key_sha256"]
        if type(declared_sha) is not str:
            _fail("estimand candidate SHA-256 must be text")
        _require_sha256(declared_sha, label="estimand candidate SHA-256")
        expected_sha = canonical_sha256(
            {
                "schema_version": "r3_phase0_estimand_candidate_key_v1",
                "study": study,
                "endpoint": endpoint,
                "observed_unit": unit,
            }
        )
        if declared_sha != expected_sha:
            _fail("estimand candidate key SHA-256 is inconsistent")
        row_count = _require_positive_int(
            item["row_count"], label="estimand candidate row_count"
        )
        family = item["mechanical_endpoint_family_hint"]
        if family not in families or family != _endpoint_family(
            study, endpoint, unit
        ):
            _fail("estimand mechanical family hint is inconsistent")
        if (
            item["review_status"] != "HUMAN_REVIEW_REQUIRED"
            or item["estimand_decision"] != "NONE"
            or not _strict_json_equal(
                item["required_human_review_fields"], ESTIMAND_REVIEW_FIELDS
            )
        ):
            _fail("estimand candidate attempted to exceed proposal scope")
        if family == "RIBOSOME_LOADING_MRL" and (
            "stabil" in canonical_json(item).lower()
        ):
            _fail("MRL proposal must retain ribosome-loading semantics")
        observed_hashes.append(declared_sha)
        row_total += row_count
    if observed_hashes != sorted(set(observed_hashes)):
        _fail("estimand candidates must be unique and hash-sorted")
    return queue, row_total


def _validate_relation_shape(value: object) -> tuple[dict, dict]:
    relation = _expect_object(
        value,
        {
            "status",
            "relation_atom_authority_created",
            "frozen_relation_atoms_created",
            "edge_policy_created",
            "review_disposition",
            "hold_reasons",
            "pair_join_audit",
            "projection_metadata",
            "registry_group_sha_mismatch_count",
            "conflicting_group_registry_row_count",
            "registry_member_ids_invalid_count",
            "registry_effect_pair_membership_count",
            "assignment_missing_registry_group_count",
            "assignment_registry_group_invalid_count",
            "assignment_pair_not_registry_member_count",
            "assignment_algorithm_mismatch_count",
            "coverage_by_atom",
            "hub_audit",
        },
        label="relation_atom_inventory",
    )
    if (
        relation["status"] != STATUS
        or relation["relation_atom_authority_created"] is not False
        or relation["frozen_relation_atoms_created"] is not False
        or relation["edge_policy_created"] is not False
    ):
        _fail("relation inventory attempted to create an authority or edge policy")
    _expect_list(relation["hold_reasons"], label="relation hold_reasons")
    pair_join = _expect_object(
        relation["pair_join_audit"],
        {
            "unique_effect_pair_count",
            "d1_pair_match_count",
            "d1_accepted_pair_match_count",
            "d1_pair_missing_count",
            "relation_status_authority_field",
            "relation_status_allowlist",
            "canonical_missing_resolution",
            "canonical_missing_source_field",
            "legacy_status_role",
            "d1_relation_status_counts",
            "d1_relation_status_source_field_counts",
        },
        label="relation pair_join_audit",
    )
    if (
        pair_join["relation_status_authority_field"]
        != RELATION_STATUS_AUTHORITY_FIELD
        or pair_join["relation_status_allowlist"]
        != sorted(RELATION_ACCEPTANCE_STATUS_ALLOWLIST)
        or pair_join["canonical_missing_resolution"] != "UNREVIEWED"
        or pair_join["canonical_missing_source_field"]
        != MISSING_RELATION_STATUS_SOURCE
        or pair_join["legacy_status_role"]
        != "CONSISTENCY_CHECK_ONLY_NO_PROMOTION"
    ):
        _fail("relation status audit does not preserve canonical authority")
    effect_pair_count = _require_nonnegative_int(
        pair_join["unique_effect_pair_count"],
        label="pair_join unique_effect_pair_count",
    )
    matched_count = _require_nonnegative_int(
        pair_join["d1_pair_match_count"], label="pair_join d1_pair_match_count"
    )
    accepted_count = _require_nonnegative_int(
        pair_join["d1_accepted_pair_match_count"],
        label="pair_join d1_accepted_pair_match_count",
    )
    missing_count = _require_nonnegative_int(
        pair_join["d1_pair_missing_count"],
        label="pair_join d1_pair_missing_count",
    )
    if (
        matched_count > effect_pair_count
        or accepted_count > matched_count
        or missing_count != effect_pair_count - matched_count
    ):
        _fail("relation pair join counts violate conservation")
    _validate_nonnegative_count_map(
        pair_join["d1_relation_status_counts"],
        label="d1_relation_status_counts",
        allowed_keys=set(RELATION_ACCEPTANCE_STATUS_ALLOWLIST) | {"UNREVIEWED"},
    )
    _validate_nonnegative_count_map(
        pair_join["d1_relation_status_source_field_counts"],
        label="d1_relation_status_source_field_counts",
        allowed_keys={
            RELATION_STATUS_AUTHORITY_FIELD,
            MISSING_RELATION_STATUS_SOURCE,
        },
    )
    projection = _expect_object(
        relation["projection_metadata"],
        {
            "artifact_id",
            "schema_version",
            "grouping_atom_projection_rule_sha256",
            "assignment_algorithm_id",
            "group_id_algorithm_id",
            "review_status",
        },
        label="relation projection_metadata",
    )
    for field in (
        "artifact_id",
        "schema_version",
        "assignment_algorithm_id",
        "group_id_algorithm_id",
    ):
        _require_safe_text(projection[field], label=f"projection {field}")
    if type(projection["grouping_atom_projection_rule_sha256"]) is not str:
        _fail("projection rule SHA-256 must be text")
    _require_sha256(
        projection["grouping_atom_projection_rule_sha256"],
        label="projection rule SHA-256",
    )
    if projection["review_status"] != "HUMAN_REVIEW_REQUIRED":
        _fail("projection metadata must remain human-review-only")
    for field in (
        "registry_group_sha_mismatch_count",
        "conflicting_group_registry_row_count",
        "registry_member_ids_invalid_count",
        "registry_effect_pair_membership_count",
        "assignment_missing_registry_group_count",
        "assignment_registry_group_invalid_count",
        "assignment_pair_not_registry_member_count",
        "assignment_algorithm_mismatch_count",
    ):
        _require_nonnegative_int(relation[field], label=f"relation {field}")
    coverage = _expect_list(
        relation["coverage_by_atom"], label="relation coverage_by_atom"
    )
    observed_atoms = []
    largest_count = 0
    largest_atom = None
    for index, raw_atom in enumerate(coverage):
        atom = _expect_object(
            raw_atom,
            {
                "atom_type",
                "effect_pair_count_with_assignment",
                "effect_pair_coverage_denominator",
                "distinct_group_count_for_effect_pairs",
                "object_attribute_pair_count",
                "max_effect_pairs_in_one_group",
                "multi_pair_group_count",
                "assignment_not_in_object_attributes_count",
                "object_attributes_not_in_assignment_count",
                "mapping_status",
                "edge_decision",
            },
            label=f"relation coverage_by_atom[{index}]",
        )
        atom_type = atom["atom_type"]
        if atom_type not in GROUPING_ATOM_ALLOWLIST:
            _fail("relation coverage contains a non-allowlisted atom")
        observed_atoms.append(atom_type)
        count_fields = (
            "effect_pair_count_with_assignment",
            "effect_pair_coverage_denominator",
            "distinct_group_count_for_effect_pairs",
            "object_attribute_pair_count",
            "max_effect_pairs_in_one_group",
            "multi_pair_group_count",
            "assignment_not_in_object_attributes_count",
            "object_attributes_not_in_assignment_count",
        )
        for field in count_fields:
            _require_nonnegative_int(
                atom[field], label=f"coverage {atom_type} {field}"
            )
        pair_count = atom["effect_pair_count_with_assignment"]
        if (
            atom["effect_pair_coverage_denominator"] != effect_pair_count
            or pair_count > effect_pair_count
            or atom["object_attribute_pair_count"] > effect_pair_count
            or atom["max_effect_pairs_in_one_group"] > pair_count
            or atom["multi_pair_group_count"]
            > atom["distinct_group_count_for_effect_pairs"]
        ):
            _fail("relation atom coverage counts violate bounds")
        if (
            atom["mapping_status"] != "MECHANICAL_EVIDENCE_ONLY"
            or atom["edge_decision"] != "UNDECIDED_HUMAN_REVIEW_REQUIRED"
        ):
            _fail("relation atoms must remain undecided mechanical evidence")
        hub_count = atom["max_effect_pairs_in_one_group"]
        if hub_count > largest_count:
            largest_count = hub_count
            largest_atom = atom_type
    if observed_atoms != sorted(set(observed_atoms)):
        _fail("relation coverage atoms must be unique and sorted")
    hub = _expect_object(
        relation["hub_audit"],
        {
            "largest_effect_pair_hub_atom",
            "largest_effect_pair_hub_count",
            "hub_interpretation",
        },
        label="relation hub_audit",
    )
    if hub["largest_effect_pair_hub_atom"] != largest_atom:
        _fail("relation hub atom is inconsistent with coverage")
    if (
        _require_nonnegative_int(
            hub["largest_effect_pair_hub_count"],
            label="relation largest hub count",
        )
        != largest_count
        or hub["hub_interpretation"]
        != "MECHANICAL_REVIEW_ONLY_NOT_AN_EDGE_DECISION"
    ):
        _fail("relation hub audit is inconsistent")
    return relation, pair_join


def _validate_mechanical_inventory(value: object, authorities: Mapping) -> dict:
    inventory = _expect_object(
        value,
        {
            "status",
            "effect_row_count",
            "unique_effect_pair_count",
            "effect_rows_missing_pair_count",
            "duplicate_effect_pair_row_count",
            "rows_with_both_sequence_fields_present_count",
            "canonical_edit_script_row_count",
            "legacy_token_edit_script_row_count",
            "other_or_missing_edit_script_row_count",
            "edit_count_histogram",
            "pair_relation_status_counts",
            "functional_observation_row_count",
            "functional_observation_missing_field_counts",
            "sequence_entity_row_count",
            "sequence_entity_missing_normalized_support_count",
            "raw_sequence_values_emitted",
            "identity_values_emitted",
            "member_lists_emitted",
            "outcome_fields_used_for_inventory",
            "numeric_outcome_aggregates_emitted",
            "mechanical_inventory_interpretation",
        },
        label="mechanical_candidate_inventory",
    )
    if inventory["status"] != STATUS:
        _fail("mechanical inventory status is not proposal-only")
    count_fields = (
        "effect_row_count",
        "unique_effect_pair_count",
        "effect_rows_missing_pair_count",
        "duplicate_effect_pair_row_count",
        "rows_with_both_sequence_fields_present_count",
        "canonical_edit_script_row_count",
        "legacy_token_edit_script_row_count",
        "other_or_missing_edit_script_row_count",
        "functional_observation_row_count",
        "sequence_entity_row_count",
        "sequence_entity_missing_normalized_support_count",
    )
    for field in count_fields:
        _require_nonnegative_int(inventory[field], label=f"mechanical {field}")
    effect_rows = inventory["effect_row_count"]
    if (
        inventory["unique_effect_pair_count"] > effect_rows
        or inventory["effect_rows_missing_pair_count"] > effect_rows
        or inventory["rows_with_both_sequence_fields_present_count"] > effect_rows
        or sum(
            inventory[field]
            for field in (
                "canonical_edit_script_row_count",
                "legacy_token_edit_script_row_count",
                "other_or_missing_edit_script_row_count",
            )
        )
        != effect_rows
    ):
        _fail("mechanical effect-row counts violate conservation")
    edit_histogram = _validate_nonnegative_count_map(
        inventory["edit_count_histogram"], label="edit_count_histogram"
    )
    if any(
        key != "MISSING_OR_INVALID" and re.fullmatch(r"-?\d+", key) is None
        for key in edit_histogram
    ) or sum(edit_histogram.values()) != effect_rows:
        _fail("edit-count histogram is non-canonical or non-conserving")
    pair_statuses = _validate_nonnegative_count_map(
        inventory["pair_relation_status_counts"],
        label="pair_relation_status_counts",
        allowed_keys=set(RELATION_ACCEPTANCE_STATUS_ALLOWLIST)
        | {"UNREVIEWED", "MISSING"},
    )
    if sum(pair_statuses.values()) != effect_rows:
        _fail("effect pair relation status counts do not conserve rows")
    missing = _expect_object(
        inventory["functional_observation_missing_field_counts"],
        {"endpoint_id", "context_id", "replicate", "unit"},
        label="functional observation missing counts",
    )
    for field, count in missing.items():
        _require_nonnegative_int(
            count, label=f"functional observation missing {field}"
        )
        if count > inventory["functional_observation_row_count"]:
            _fail("functional observation missing count exceeds row count")
    fixed_flags = (
        "raw_sequence_values_emitted",
        "identity_values_emitted",
        "member_lists_emitted",
        "outcome_fields_used_for_inventory",
        "numeric_outcome_aggregates_emitted",
    )
    if any(inventory[field] is not False for field in fixed_flags):
        _fail("mechanical inventory attempted to emit protected capability")
    if (
        inventory["mechanical_inventory_interpretation"]
        != "PROPOSAL_COUNTS_ONLY_NO_SCIENTIFIC_ADMISSION"
    ):
        _fail("mechanical inventory interpretation was changed")
    if (
        effect_rows != authorities["effect_dataset"]["json_row_count"]
        or inventory["functional_observation_row_count"]
        != authorities["functional_observations"]["json_row_count"]
        or inventory["sequence_entity_row_count"]
        != authorities["sequence_entities"]["json_row_count"]
    ):
        _fail("mechanical inventory row counts disagree with input authorities")
    return inventory


def validate_phase0_proposal(payload: Mapping) -> dict:
    """Replay proposal commitments and reject any attempted capability upgrade."""

    if not isinstance(payload, Mapping):
        _fail("Phase-0 proposal must be an object")
    required_keys = {
        "schema_version",
        "artifact_role",
        "status",
        "training_ready",
        "phase0_ready",
        "split_created",
        "ordinary_only",
        "sealed_access",
        "internal_test_access",
        "downstream_unlock",
        "input_authorities",
        "binding_checks",
        "authority_binding_sha256",
        "inventory_policy",
        "context_candidate_inventory",
        "estimand_review_queue",
        "relation_atom_inventory",
        "mechanical_candidate_inventory",
        "outcome_free_inventory_sha256",
        "human_review_required",
        "proposal_limitations",
        "proposal_sha256",
    }
    if set(payload) != required_keys:
        _fail("Phase-0 proposal fields differ from the exact proposal schema")
    exact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": ARTIFACT_ROLE,
        "status": STATUS,
        "training_ready": False,
        "phase0_ready": False,
        "split_created": False,
        "ordinary_only": True,
        "sealed_access": False,
        "internal_test_access": False,
        "downstream_unlock": False,
    }
    for field, expected in exact.items():
        actual = payload.get(field)
        if isinstance(expected, bool):
            valid = actual is expected
        else:
            valid = type(actual) is type(expected) and actual == expected
        if not valid:
            _fail(f"Phase-0 proposal cannot change {field}")
    for location, key, value in _walk_output(payload):
        if key.lower() in _PROHIBITED_OUTPUT_KEYS:
            _fail(f"proposal contains prohibited identity/outcome field: {location}")
        if isinstance(value, str) and value in {
            "FROZEN",
            "ACTIVE",
            "READY",
            "PASS",
            "GO",
        }:
            _fail(f"proposal contains prohibited activating status at {location}")

    authorities = _validate_input_authorities(payload.get("input_authorities"))
    binding_checks = payload.get("binding_checks")
    if not _strict_json_equal(binding_checks, BINDING_CHECKS):
        _fail("binding_checks differ from the exact all-true schema")
    binding_material = {
        "schema_version": BINDING_SCHEMA_VERSION,
        "input_authorities": dict(authorities),
        "binding_checks": binding_checks,
    }
    authority_binding_sha = payload.get("authority_binding_sha256")
    if type(authority_binding_sha) is not str:
        _fail("authority binding SHA-256 must be text")
    _require_sha256(authority_binding_sha, label="authority binding SHA-256")
    if authority_binding_sha != canonical_sha256(binding_material):
        _fail("proposal authority binding SHA-256 is stale")
    inventory_policy = payload.get("inventory_policy")
    if not _strict_json_equal(inventory_policy, INVENTORY_POLICY):
        _fail("inventory_policy differs from the exact non-capability policy")
    context_inventory, context_row_total = _validate_context_inventory(
        payload.get("context_candidate_inventory")
    )
    queue, estimand_row_total = _validate_estimand_queue(
        payload.get("estimand_review_queue")
    )
    relation, pair_join = _validate_relation_shape(
        payload.get("relation_atom_inventory")
    )
    mechanical_inventory = _validate_mechanical_inventory(
        payload.get("mechanical_candidate_inventory"), authorities
    )
    effect_rows = mechanical_inventory["effect_row_count"]
    if context_row_total != effect_rows or estimand_row_total != effect_rows:
        _fail("context/estimand candidate rows do not conserve effect rows")
    if any(
        count > effect_rows
        for count in context_inventory["missing_field_counts"].values()
    ):
        _fail("context missing-field count exceeds effect rows")
    if (
        pair_join["unique_effect_pair_count"]
        != mechanical_inventory["unique_effect_pair_count"]
    ):
        _fail("relation and mechanical unique-effect-pair counts disagree")
    status_counts = pair_join["d1_relation_status_counts"]
    source_counts = pair_join["d1_relation_status_source_field_counts"]
    if sum(status_counts.values()) != sum(source_counts.values()):
        _fail("D1 relation status/source aggregate counts disagree")
    if pair_join["d1_accepted_pair_match_count"] > status_counts.get(
        "ACCEPTED", 0
    ):
        _fail("accepted effect-pair matches exceed accepted D1 pairs")
    hold_count_fields = (
        (
            "D1_RELATION_STATUS_UNREVIEWED",
            status_counts.get("UNREVIEWED", 0),
        ),
        (
            "REGISTRY_MEMBER_IDS_INVALID",
            relation.get("registry_member_ids_invalid_count"),
        ),
        (
            "REGISTRY_GROUP_SHA_MISMATCH",
            relation.get("registry_group_sha_mismatch_count"),
        ),
        (
            "CONFLICTING_GROUP_REGISTRY_ROWS",
            relation.get("conflicting_group_registry_row_count"),
        ),
        (
            "ASSIGNMENT_MISSING_REGISTRY_GROUP",
            relation.get("assignment_missing_registry_group_count"),
        ),
        (
            "ASSIGNMENT_REGISTRY_GROUP_INVALID",
            relation.get("assignment_registry_group_invalid_count"),
        ),
        (
            "ASSIGNMENT_PAIR_NOT_REGISTRY_MEMBER",
            relation.get("assignment_pair_not_registry_member_count"),
        ),
        (
            "ASSIGNMENT_ALGORITHM_MISMATCH",
            relation.get("assignment_algorithm_mismatch_count"),
        ),
    )
    if any(
        not isinstance(count, int)
        or isinstance(count, bool)
        or count < 0
        for _, count in hold_count_fields
    ):
        _fail("relation HOLD audit contains invalid aggregate counts")
    expected_hold_reasons = [
        reason for reason, count in hold_count_fields if count > 0
    ]
    expected_disposition = (
        "HOLD_FOR_HUMAN_REVIEW"
        if expected_hold_reasons
        else "HUMAN_REVIEW_REQUIRED"
    )
    if (
        relation.get("hold_reasons") != expected_hold_reasons
        or relation.get("review_disposition") != expected_disposition
    ):
        _fail("relation HOLD/review disposition is inconsistent with audit counts")
    if not _strict_json_equal(
        payload.get("human_review_required"), HUMAN_REVIEW_REQUIRED
    ):
        _fail("human_review_required differs from the exact fixed list")
    if not _strict_json_equal(
        payload.get("proposal_limitations"), PROPOSAL_LIMITATIONS
    ):
        _fail("proposal_limitations differs from the exact fixed list")

    outcome_free_material = {
        "schema_version": OUTCOME_COMMITMENT_SCHEMA_VERSION,
        "inventory_policy": inventory_policy,
        "context_candidate_inventory": context_inventory,
        "estimand_review_queue": queue,
        "relation_atom_inventory": relation,
        "mechanical_candidate_inventory": mechanical_inventory,
    }
    outcome_sha = payload.get("outcome_free_inventory_sha256")
    if type(outcome_sha) is not str:
        _fail("outcome-free inventory SHA-256 must be text")
    _require_sha256(outcome_sha, label="outcome-free inventory SHA-256")
    if outcome_sha != canonical_sha256(outcome_free_material):
        _fail("outcome-free inventory SHA-256 is stale")

    if type(payload.get("proposal_sha256")) is not str:
        _fail("proposal SHA-256 must be text")
    declared_proposal_sha = _require_sha256(
        payload["proposal_sha256"], label="proposal SHA-256"
    )
    material = dict(payload)
    material.pop("proposal_sha256")
    if declared_proposal_sha != canonical_sha256(material):
        _fail("proposal SHA-256 is stale")
    return dict(payload)


def write_phase0_proposal_exclusive(path: Path, proposal: Mapping) -> str:
    """Write exactly one validated canonical JSON artifact without overwrite."""

    path = Path(path)
    _assert_phase0_path(path, purpose="Phase-0 proposal output")
    try:
        return atomic_write_json_exclusive(
            path,
            validate_phase0_proposal(proposal),
            validator=validate_phase0_proposal,
        )
    except ArtifactIOError as exc:
        raise Phase0ProposalError(str(exc)) from exc


def _argument_name(logical: str) -> str:
    return "--" + logical.replace("_", "-")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an ordinary-only non-active R3 Phase-0 proposal"
    )
    parser.add_argument("--effect", type=Path, required=True)
    parser.add_argument("--effect-sha256", required=True)
    parser.add_argument("--effect-manifest", type=Path, required=True)
    parser.add_argument("--effect-manifest-sha256", required=True)
    parser.add_argument("--d1-sha256s", type=Path, required=True)
    parser.add_argument("--d1-sha256s-sha256", required=True)
    for logical in D1_FILE_NAMES:
        parser.add_argument(_argument_name(logical), type=Path, required=True)
        parser.add_argument(_argument_name(logical) + "-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    d1_paths = {
        logical: getattr(args, logical) for logical in D1_FILE_NAMES
    }
    d1_expected = {
        logical: getattr(args, logical + "_sha256") for logical in D1_FILE_NAMES
    }
    proposal = build_phase0_proposal(
        effect_path=args.effect,
        effect_sha256=args.effect_sha256,
        effect_manifest_path=args.effect_manifest,
        effect_manifest_sha256=args.effect_manifest_sha256,
        d1_sha256s_path=args.d1_sha256s,
        d1_sha256s_sha256=args.d1_sha256s_sha256,
        d1_paths=d1_paths,
        d1_expected_sha256=d1_expected,
    )
    artifact_sha = write_phase0_proposal_exclusive(args.output, proposal)
    print(
        json.dumps(
            {
                "status": STATUS,
                "output_basename": args.output.name,
                "output_sha256": artifact_sha,
                "training_ready": False,
                "phase0_ready": False,
                "split_created": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
