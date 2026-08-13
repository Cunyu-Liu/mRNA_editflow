#!/usr/bin/env python3
"""GSE256185 ordinary-public identifier/pool-geometry preflight.

The producer verifies one frozen public GEO asset, stream-decompresses it in
binary mode, inspects the header names once, and then partitions every body
line at the first tab.  Only the first ``ID`` field is decoded or parsed.  The
effect, CPM, and sequence body cells are never decoded, split, or inspected.

The sole output contains aggregate role/pool geometry.  It never contains a
member identifier, role, context, sequence, effect, or CPM value, and it never
qualifies or counts GSE256185, materializes a canonical record, trains a model,
selects a model, or authorizes a later phase.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


SCHEMA_VERSION = (
    "route_a_v3_gse256185_public_identifier_pool_geometry_preflight.v1"
)
PROTOCOL_ID = "GSE256185_PUBLIC_IDENTIFIER_POOL_GEOMETRY_PREFLIGHT_V1"
PROTOCOL_BASENAME = (
    "route_a_v3_gse256185_public_identifier_pool_geometry_preflight_v1.json"
)
REPORT_FILENAME = "GSE256185_PUBLIC_IDENTIFIER_POOL_GEOMETRY_PREFLIGHT_V1.json"
DATASET_ID = "GSE256185"
BIOPROJECT_ID = "PRJNA1078388"
DECISION_ID = "V3-DEC-021"
UNKNOWN = "UNKNOWN_NOT_ASSERTED"
BOUND = "BOUND"

SCRIPT_PATH = (
    "scripts/route_a_v3/"
    "preflight_gse256185_public_identifier_pool_geometry.py"
)
TEST_PATH = (
    "tests/route_a_v3/"
    "test_preflight_gse256185_public_identifier_pool_geometry.py"
)
CONFIG_PATH = f"configs/{PROTOCOL_BASENAME}"
EXPECTED_EXACT3 = (CONFIG_PATH, SCRIPT_PATH, TEST_PATH)
UNKNOWN_BINDING_SCALARS = (
    "status",
    "implementation_commit",
    "implementation_script_sha256",
    "implementation_test_sha256",
)

OFFICIAL_ASSET = {
    "filename": "GSE256185_CPMandRRS_VCE_Var.tsv.gz",
    "compressed_bytes": 952533,
    "compressed_sha256": (
        "71a8476a76e9a47a03bc69a2e0cbf79d92019249fba2049f57b7aa60f3f25aeb"
    ),
}
EXPECTED_HEADER = (
    "ID",
    "AVG_log2RRS",
    "X80S.2_CPM",
    "X80S.3_CPM",
    "X80S.4_CPM",
    "X80S.5_CPM",
    "IVT.1_CPM",
    "IVT.2_CPM",
    "IVT.3_CPM",
    "Sequence",
)
ROLE_FAMILIES = ("parent", "win", "+CCC", "-CCC", "rand")
ANOMALY_CLASSES = (
    "MISSING_GROUP_ROLE_DELIMITER",
    "UNSIGNED_CCC_ROLE",
    "OTHER_IDENTIFIER_GRAMMAR",
    "OTHER_ROLE_GRAMMAR",
)
FAMILY_CLOSURE_STATUS = "REASONED_FAMILY_CLOSURE_NOT_PUBLISHER_EXPLICIT"
COMPLETION_STATUS = (
    "PUBLIC_IDENTIFIER_AND_POOL_GEOMETRY_PREFLIGHT_COMPLETE_NOT_QUALIFIED"
)

EXPECTED_GEOMETRY = {
    "total_body_row_count": 11404,
    "strict_grammar_row_count": 11402,
    "strict_role_family_row_counts": {
        "parent": 667,
        "win": 5501,
        "+CCC": 1125,
        "-CCC": 1109,
        "rand": 3000,
    },
    "identifier_grammar_anomaly_counts": {
        "MISSING_GROUP_ROLE_DELIMITER": 1,
        "UNSIGNED_CCC_ROLE": 1,
        "OTHER_IDENTIFIER_GRAMMAR": 0,
        "OTHER_ROLE_GRAMMAR": 0,
    },
    "strict_axis": {
        "group_count": 652,
        "groups_with_parent": 652,
        "single_parent_group_count": 637,
        "dual_parent_group_count": 15,
        "other_parent_multiplicity_group_count": 0,
        "single_parent_groups_with_at_least_3_strict_candidate_rows": 634,
        "single_parent_groups_with_exactly_2_strict_candidate_rows": 3,
        "strict_candidate_rows_in_at_least_3_candidate_groups": 7292,
    },
    "reasoned_family_closure_axis": {
        "status": FAMILY_CLOSURE_STATUS,
        "group_count": 652,
        "groups_with_parent": 652,
        "single_parent_group_count": 637,
        "dual_parent_group_count": 15,
        "other_parent_multiplicity_group_count": 0,
        "single_parent_groups_with_at_least_3_candidate_rows": 634,
        "single_parent_groups_with_exactly_2_candidate_rows": 3,
        "candidate_rows_in_at_least_3_candidate_groups": 7294,
    },
}
EXPECTED_OUTER_TRUTH = {
    "current_qualified_counts": {
        "ordinary": 1,
        "a1": 1,
        "true_a2": 0,
        "canonical_records": 6547,
    },
    "gse256185_contribution": {
        "ordinary": 0,
        "a1": 0,
        "true_a2": 0,
        "canonical_records": 0,
    },
    "gse256185_qualified": False,
    "gse256185_true_a2_established": False,
    "sequence_edit_semantics": "NOT_EVALUATED_OUT_OF_SCOPE",
    "edit_budget_status": "NOT_EVALUATED_OUT_OF_SCOPE",
    "effect_status": "NOT_EVALUATED_OUT_OF_SCOPE",
    "a1_complete": False,
    "qualification_run_count": 0,
    "canonical_materialization_run_count": 0,
    "training_run_count": 0,
    "gpu_work_run_count": 0,
    "model_selection_run_count": 0,
    "training_allowed": False,
    "gpu_work_allowed": False,
    "model_selection_allowed": False,
    "next_phase_authorized": False,
    "scientific_claim_status": "NOT_ESTABLISHED",
}

GROUP_RE = re.compile(r"^ENSG\d+-ENST\d+-\d+$")
WIN_ROLE_RE = re.compile(r"^win\d+$")
CCC_ROLE_RE = re.compile(r"^[+-]\d+CCC$")
RAND_ROLE_RE = re.compile(r"^rand\d+$")
UNSIGNED_CCC_ROLE_RE = re.compile(r"^\d+CCC$")
MISSING_DELIMITER_RE = re.compile(
    r"^(?P<group>ENSG\d+-ENST\d+-\d+)(?P<role>-\d+CCC)\."
    r"(?P<suffix>[^.]+)$"
)
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class PreflightError(RuntimeError):
    """Base class for a fail-closed preflight failure."""


class ProtocolError(PreflightError):
    """The protocol or its authority/binding lifecycle is invalid."""


class BindingNotFrozen(ProtocolError):
    """The exact3-I/config-only-B lifecycle is not complete."""


class AssetIdentityError(PreflightError):
    """The ordinary-public asset is not the frozen asset."""


class GeometryError(PreflightError):
    """The header or aggregate identifier geometry differs from the freeze."""


class OutputError(PreflightError):
    """The aggregate-only output cannot be created without overwriting."""


def _json_bytes(value: Any) -> bytes:
    try:
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
    except (TypeError, ValueError) as exc:
        raise PreflightError("report is not finite JSON") from exc


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite token {token}")
            ),
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(parsed, dict):
        raise ProtocolError(f"{label} must be a JSON object")
    return parsed


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be an object")
    return value


def _expect(mapping: Mapping[str, Any], expected: Mapping[str, Any], *, label: str) -> None:
    if dict(mapping) != dict(expected):
        raise ProtocolError(f"{label} differs from the frozen protocol")


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "contract_id": "mrna_xeditflow_route_a_v3",
        "phase_id": "A1",
        "dataset_id": DATASET_ID,
        "bioproject_id": BIOPROJECT_ID,
        "decision_id": DECISION_ID,
        "protocol_status": (
            "PUBLIC_IDENTIFIER_AND_POOL_GEOMETRY_PREFLIGHT_ONLY_NOT_QUALIFICATION"
        ),
    }
    for key, expected in expected_scalars.items():
        if protocol.get(key) != expected:
            raise ProtocolError(f"protocol {key} differs from the frozen value")

    binding = _mapping(
        protocol.get("implementation_binding"), label="implementation_binding"
    )
    if binding.get("binding_scheme") != (
        "AUTHORITY_A_THEN_AUTHORITY_RUNTIME_I_B_THEN_EXACT3_I_CONFIG_ONLY_B_V1"
    ):
        raise ProtocolError("implementation binding scheme differs")
    authority_fields = ("authority_commit", "authority_runtime_binding_commit")
    authority_values = [binding.get(field) for field in authority_fields]
    authority_unknown = authority_values == [UNKNOWN, UNKNOWN]
    authority_bound = all(
        isinstance(value, str) and HEX40_RE.fullmatch(value)
        for value in authority_values
    )
    if not authority_unknown and not authority_bound:
        raise ProtocolError("authority A/runtime-B binding is partially known")
    if binding.get("pre_implementation_authority_scalar_paths") != [
        "implementation_binding.authority_commit",
        "implementation_binding.authority_runtime_binding_commit",
    ]:
        raise ProtocolError("pre-implementation authority scalar paths differ")
    if binding.get("unknown_to_bound_scalar_paths") != [
        f"implementation_binding.{field}" for field in UNKNOWN_BINDING_SCALARS
    ]:
        raise ProtocolError("the four normal binding scalar paths differ")
    if tuple(binding.get("implementation_commit_exact_changed_paths", ())) != (
        EXPECTED_EXACT3
    ):
        raise ProtocolError("implementation commit is not exact3")
    if binding.get("binding_commit_exact_changed_paths") != [CONFIG_PATH]:
        raise ProtocolError("binding commit must be config-only")
    if binding.get("implementation_script_path") != SCRIPT_PATH:
        raise ProtocolError("implementation script path differs")
    if binding.get("implementation_test_path") != TEST_PATH:
        raise ProtocolError("implementation test path differs")

    status = binding.get("status")
    normal_values = [binding.get(field) for field in UNKNOWN_BINDING_SCALARS]
    if status == UNKNOWN:
        if normal_values != [UNKNOWN] * len(UNKNOWN_BINDING_SCALARS):
            raise ProtocolError("initial-I binding scalars must all remain UNKNOWN")
        if not (authority_unknown or authority_bound):
            raise ProtocolError("UNKNOWN lifecycle authority group is invalid")
    elif status == BOUND:
        if not authority_bound:
            raise ProtocolError(
                "BOUND lifecycle requires authority A and authority-runtime B"
            )
        if not (
            isinstance(binding.get("implementation_commit"), str)
            and HEX40_RE.fullmatch(str(binding["implementation_commit"]))
        ):
            raise ProtocolError("BOUND implementation commit is invalid")
        for field in (
            "implementation_script_sha256",
            "implementation_test_sha256",
        ):
            if not (
                isinstance(binding.get(field), str)
                and HEX64_RE.fullmatch(str(binding[field]))
            ):
                raise ProtocolError(f"BOUND {field} is invalid")
    else:
        raise ProtocolError("implementation binding status is invalid")

    authority = _mapping(protocol.get("decision_authority"), label="decision_authority")
    _expect(
        authority,
        {
            "amendment_path": (
                "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec021.yaml"
            ),
            "data_role_registry_path": (
                "docs/execution/route_a_v3_data_role_registry.yaml"
            ),
            "authorized_role": "PUBLIC_IDENTIFIER_AND_POOL_GEOMETRY_PREFLIGHT_ONLY",
            "allowed_input_field_classes_exactly": [
                "IDENTIFIER",
                "ROLE",
                "CONTEXT",
            ],
            "allowed_output_class": "AGGREGATE_POOL_GEOMETRY_ONLY",
        },
        label="decision_authority",
    )

    asset = _mapping(protocol.get("official_processed_asset"), label="asset")
    for key, expected in OFFICIAL_ASSET.items():
        if asset.get(key) != expected:
            raise ProtocolError(f"official_processed_asset.{key} differs")
    if not str(asset.get("locator", "")).startswith(
        "https://www.ncbi.nlm.nih.gov/geo/download/"
    ):
        raise ProtocolError("processed asset locator is not official NCBI HTTPS")
    if asset.get("identity_mismatch_action") != (
        "STOP_BEFORE_DECOMPRESSION_OR_AGGREGATION"
    ):
        raise ProtocolError("asset identity mismatch action differs")

    context = _mapping(protocol.get("frozen_public_context"), label="context")
    context_expected = {
        "study_accession": DATASET_ID,
        "bioproject_accession": BIOPROJECT_ID,
        "assay_context": "DART_SEQ_TRANSLATION_INITIATION_EFFICIENCY",
        "library_pool_context": "VCE_VARIANT_POOL",
        "biological_material_context": "HELA_CELL_EXTRACT",
        "construct_context": "IN_VITRO_TRANSCRIBED_VCE_CAPPED_MRNA_LIBRARY",
        "measurement_context_categories": [
            "80S_MONOSOME_ASSOCIATED_RNA",
            "TOTAL_IVT_INPUT_RNA",
        ],
        "relevant_public_sample_context_count": 2,
        "context_source_authority_ids": [
            "NCBI_GEO_GSE256185",
            "NCBI_GEO_GSM8087263",
            "NCBI_GEO_GSM8087266",
            "NCBI_PMC_11780321",
        ],
    }
    _expect(context, context_expected, label="frozen_public_context")

    header = _mapping(protocol.get("header_contract"), label="header_contract")
    if tuple(header.get("exact_column_names", ())) != EXPECTED_HEADER:
        raise ProtocolError("header contract differs")
    if header.get("identifier_column_name") != "ID":
        raise ProtocolError("identifier column name differs")
    if header.get("identifier_column_index") != 0:
        raise ProtocolError("identifier column must remain first")
    if header.get("body_parser_rule") != (
        "STREAM_DECOMPRESS_EACH_RAW_LINE_PARTITION_ON_FIRST_TAB_AND_DECODE_ONLY_ID_FIELD"
    ):
        raise ProtocolError("body parser rule differs")

    grammar = _mapping(
        protocol.get("identifier_role_grammar"), label="identifier_role_grammar"
    )
    if grammar.get("strict_role_tokens") != [
        "parent",
        "win<digits>",
        "+<digits>CCC",
        "-<digits>CCC",
        "rand<digits>",
    ]:
        raise ProtocolError("strict role grammar differs")
    if grammar.get("role_families") != list(ROLE_FAMILIES):
        raise ProtocolError("role families differ")
    if grammar.get("frozen_observed_anomaly_classes") != list(
        ANOMALY_CLASSES[:2]
    ):
        raise ProtocolError("frozen anomaly classes differ")
    if grammar.get("family_closure_axis_status") != FAMILY_CLOSURE_STATUS:
        raise ProtocolError("family-closure inference label differs")
    if grammar.get("family_closure_may_replace_strict_axis") is not False:
        raise ProtocolError("family closure may not replace the strict axis")

    expected_geometry = _mapping(
        protocol.get("expected_aggregate_geometry"),
        label="expected_aggregate_geometry",
    )
    _expect(expected_geometry, EXPECTED_GEOMETRY, label="expected geometry")

    inputs = _mapping(protocol.get("input_contract"), label="input_contract")
    if inputs.get("allowed_body_value_columns_exactly") != ["ID"]:
        raise ProtocolError("ID must be the only allowed body-value column")
    for key in (
        "sequence_body_values_allowed",
        "effect_body_values_allowed",
        "cpm_body_values_allowed",
        "private_or_restricted_input_allowed",
        "sealed_contact_allowed",
        "GSE246381_allowed",
    ):
        if inputs.get(key) is not False:
            raise ProtocolError(f"input_contract.{key} must remain false")
    for key in (
        "ordinary_public_only",
        "header_name_inspection_allowed",
        "identifier_body_values_allowed",
        "role_tokens_derived_from_identifier_allowed",
        "context_values_allowed_only_from_frozen_public_context",
    ):
        if inputs.get(key) is not True:
            raise ProtocolError(f"input_contract.{key} must remain true")

    output = _mapping(protocol.get("output_contract"), label="output_contract")
    if output.get("filename") != REPORT_FILENAME:
        raise ProtocolError("sole output filename differs")
    if output.get("single_aggregate_output_only") is not True:
        raise ProtocolError("output must remain single and aggregate-only")
    for key in (
        "row_records_included",
        "member_identifiers_included",
        "member_roles_included",
        "member_contexts_included",
        "sequence_values_included",
        "effect_values_included",
        "cpm_values_included",
        "qualification_or_counting_credit_included",
        "canonical_records_included",
    ):
        if output.get(key) is not False:
            raise ProtocolError(f"output_contract.{key} must remain false")

    outer = _mapping(protocol.get("frozen_outer_truth"), label="frozen_outer_truth")
    _expect(outer, EXPECTED_OUTER_TRUTH, label="frozen outer truth")


def load_protocol(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.name != PROTOCOL_BASENAME:
        raise ProtocolError("protocol basename differs")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ProtocolError("cannot read protocol") from exc
    protocol = _strict_json_object(payload, label="protocol")
    _validate_protocol(protocol)
    return protocol, payload


def _run_git_text(repo_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise ProtocolError("git is unavailable for implementation binding") from exc
    if completed.returncode != 0:
        raise ProtocolError("git implementation-binding check failed")
    return completed.stdout.strip()


def _git_blob(repo_root: Path, commit: str, relative_path: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{commit}:{relative_path}"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise ProtocolError("git is unavailable for implementation binding") from exc
    if completed.returncode != 0:
        raise ProtocolError("cannot read a bound implementation blob")
    return completed.stdout


def _changed_paths(repo_root: Path, commit: str) -> tuple[str, ...]:
    output = _run_git_text(
        repo_root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        commit,
    )
    return tuple(sorted(line for line in output.splitlines() if line))


def _normalise_binding(protocol: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(protocol))
    binding = result["implementation_binding"]
    for field in UNKNOWN_BINDING_SCALARS:
        binding[field] = UNKNOWN
    return result


def _default_binding_auditor(
    protocol: Mapping[str, Any],
    protocol_path: Path,
    protocol_payload: bytes,
    repo_root: Path,
) -> dict[str, str]:
    del protocol_payload
    binding = protocol["implementation_binding"]
    if binding.get("status") != BOUND:
        raise BindingNotFrozen("exact3-I/config-only-B lifecycle is not BOUND")
    if any(binding.get(field) == UNKNOWN for field in UNKNOWN_BINDING_SCALARS):
        raise BindingNotFrozen("the four normal binding scalars remain UNKNOWN")
    if binding.get("authority_commit") == UNKNOWN:
        raise BindingNotFrozen("DEC021 authority commit remains UNKNOWN")
    if binding.get("authority_runtime_binding_commit") == UNKNOWN:
        raise BindingNotFrozen("DEC021 authority-runtime B remains UNKNOWN")

    binding_commit = _run_git_text(repo_root, "rev-parse", "HEAD")
    implementation_commit = str(binding["implementation_commit"])
    authority_commit = str(binding["authority_commit"])
    authority_runtime_binding_commit = str(
        binding["authority_runtime_binding_commit"]
    )
    if _run_git_text(repo_root, "rev-parse", f"{binding_commit}^") != (
        implementation_commit
    ):
        raise ProtocolError("B is not the direct child of I")
    if _run_git_text(repo_root, "rev-parse", f"{implementation_commit}^") != (
        authority_runtime_binding_commit
    ):
        raise ProtocolError("preflight I is not the direct child of authority-runtime B")
    authority_runtime_i = _run_git_text(
        repo_root, "rev-parse", f"{authority_runtime_binding_commit}^"
    )
    if _run_git_text(repo_root, "rev-parse", f"{authority_runtime_i}^") != (
        authority_commit
    ):
        raise ProtocolError("authority-runtime I/B is not based on DEC021 authority A")
    if _changed_paths(repo_root, implementation_commit) != tuple(
        sorted(EXPECTED_EXACT3)
    ):
        raise ProtocolError("I did not change exact3")
    if _changed_paths(repo_root, binding_commit) != (CONFIG_PATH,):
        raise ProtocolError("B did not change config-only")

    i_protocol = _strict_json_object(
        _git_blob(repo_root, implementation_commit, CONFIG_PATH),
        label="implementation protocol",
    )
    if _normalise_binding(protocol) != i_protocol:
        raise ProtocolError("B changed more than the four binding scalars")

    script_blob = _git_blob(repo_root, implementation_commit, SCRIPT_PATH)
    test_blob = _git_blob(repo_root, implementation_commit, TEST_PATH)
    if hashlib.sha256(script_blob).hexdigest() != binding.get(
        "implementation_script_sha256"
    ):
        raise ProtocolError("bound implementation script hash differs")
    if hashlib.sha256(test_blob).hexdigest() != binding.get(
        "implementation_test_sha256"
    ):
        raise ProtocolError("bound focused test hash differs")
    if protocol_path.resolve() != (repo_root / CONFIG_PATH).resolve():
        raise ProtocolError("protocol path is outside the bound repository location")
    if protocol_path.read_bytes() != _git_blob(repo_root, binding_commit, CONFIG_PATH):
        raise ProtocolError("working protocol differs from bound B")
    if (repo_root / SCRIPT_PATH).read_bytes() != script_blob:
        raise ProtocolError("working implementation script differs from I")
    if (repo_root / TEST_PATH).read_bytes() != test_blob:
        raise ProtocolError("working focused test differs from I")

    return {
        "status": "BOUND_EXACT3_I_CONFIG_ONLY_B_VERIFIED",
        "authority_commit": authority_commit,
        "authority_runtime_binding_commit": authority_runtime_binding_commit,
        "implementation_commit": implementation_commit,
        "binding_commit": binding_commit,
    }


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                byte_count += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise AssetIdentityError("cannot read ordinary-public asset") from exc
    return byte_count, digest.hexdigest()


def _default_asset_identity_auditor(
    protocol: Mapping[str, Any], asset_path: Path
) -> dict[str, Any]:
    expected = protocol["official_processed_asset"]
    if asset_path.name != expected["filename"]:
        raise AssetIdentityError("asset filename differs from the frozen GEO asset")
    byte_count, sha256 = _sha256_file(asset_path)
    if byte_count != expected["compressed_bytes"]:
        raise AssetIdentityError("compressed asset byte count differs")
    if sha256 != expected["compressed_sha256"]:
        raise AssetIdentityError("compressed asset SHA-256 differs")
    return {
        "filename": expected["filename"],
        "compressed_bytes": byte_count,
        "compressed_sha256": sha256,
        "identity_status": "PASS_FROZEN_ORDINARY_PUBLIC_ASSET",
    }


def _strict_role_family(role: str) -> str | None:
    if role == "parent":
        return "parent"
    if WIN_ROLE_RE.fullmatch(role):
        return "win"
    if CCC_ROLE_RE.fullmatch(role):
        return "+CCC" if role.startswith("+") else "-CCC"
    if RAND_ROLE_RE.fullmatch(role):
        return "rand"
    return None


def _new_group_counts() -> Counter[str]:
    return Counter({family: 0 for family in ROLE_FAMILIES})


def _parse_identifier(
    identifier: str,
) -> tuple[str, str | None, str | None]:
    """Return ``(classification, group, family)`` without echoing row data."""

    parts = identifier.rsplit(".", 2)
    if len(parts) == 3:
        group, role, suffix = parts
        if not GROUP_RE.fullmatch(group) or not suffix or "." in suffix:
            return "OTHER_IDENTIFIER_GRAMMAR", None, None
        family = _strict_role_family(role)
        if family is not None:
            return "STRICT", group, family
        if UNSIGNED_CCC_ROLE_RE.fullmatch(role):
            return "UNSIGNED_CCC_ROLE", group, "-CCC"
        return "OTHER_ROLE_GRAMMAR", None, None

    missing = MISSING_DELIMITER_RE.fullmatch(identifier)
    if missing is not None:
        return "MISSING_GROUP_ROLE_DELIMITER", missing.group("group"), "-CCC"
    return "OTHER_IDENTIFIER_GRAMMAR", None, None


def _axis_geometry(
    groups: Mapping[str, Counter[str]], *, strict_axis: bool
) -> dict[str, int]:
    parent_multiplicities = Counter(counts["parent"] for counts in groups.values())
    single_parent_candidates = {
        group: sum(
            count for family, count in counts.items() if family != "parent"
        )
        for group, counts in groups.items()
        if counts["parent"] == 1
    }
    at_least_three = {
        group: count
        for group, count in single_parent_candidates.items()
        if count >= 3
    }
    candidate_label = (
        "single_parent_groups_with_at_least_3_strict_candidate_rows"
        if strict_axis
        else "single_parent_groups_with_at_least_3_candidate_rows"
    )
    exactly_two_label = (
        "single_parent_groups_with_exactly_2_strict_candidate_rows"
        if strict_axis
        else "single_parent_groups_with_exactly_2_candidate_rows"
    )
    candidate_rows_label = (
        "strict_candidate_rows_in_at_least_3_candidate_groups"
        if strict_axis
        else "candidate_rows_in_at_least_3_candidate_groups"
    )
    result = {
        "group_count": len(groups),
        "groups_with_parent": sum(
            count for multiplicity, count in parent_multiplicities.items()
            if multiplicity >= 1
        ),
        "single_parent_group_count": parent_multiplicities[1],
        "dual_parent_group_count": parent_multiplicities[2],
        "other_parent_multiplicity_group_count": sum(
            count for multiplicity, count in parent_multiplicities.items()
            if multiplicity not in {1, 2}
        ),
        candidate_label: len(at_least_three),
        exactly_two_label: sum(
            count == 2 for count in single_parent_candidates.values()
        ),
        candidate_rows_label: sum(at_least_three.values()),
    }
    return result


def aggregate_asset_geometry(asset_path: Path) -> dict[str, Any]:
    """Aggregate the first body field only; never decode or split the remainder."""

    strict_family_counts: Counter[str] = Counter(
        {family: 0 for family in ROLE_FAMILIES}
    )
    anomaly_counts: Counter[str] = Counter(
        {classification: 0 for classification in ANOMALY_CLASSES}
    )
    strict_groups: defaultdict[str, Counter[str]] = defaultdict(_new_group_counts)
    closure_groups: defaultdict[str, Counter[str]] = defaultdict(_new_group_counts)
    total_rows = 0

    try:
        with gzip.open(asset_path, "rb") as handle:
            header_line = handle.readline()
            if not header_line:
                raise GeometryError("compressed asset has no header")
            try:
                header = tuple(
                    field.decode("ascii")
                    for field in header_line.rstrip(b"\r\n").split(b"\t")
                )
            except UnicodeDecodeError as exc:
                raise GeometryError("header names are not ASCII") from exc
            if header != EXPECTED_HEADER:
                raise GeometryError("processed asset header names differ")

            for raw_line in handle:
                total_rows += 1
                identifier_bytes, separator, _ = raw_line.partition(b"\t")
                if not separator or not identifier_bytes:
                    raise GeometryError("a body row has no first ID field")
                try:
                    identifier = identifier_bytes.decode("ascii")
                except UnicodeDecodeError as exc:
                    raise GeometryError("an ID body field is not ASCII") from exc
                classification, group, family = _parse_identifier(identifier)
                if classification == "STRICT":
                    assert group is not None and family is not None
                    strict_family_counts[family] += 1
                    strict_groups[group][family] += 1
                    closure_groups[group][family] += 1
                else:
                    anomaly_counts[classification] += 1
                    if classification in ANOMALY_CLASSES[:2]:
                        assert group is not None and family == "-CCC"
                        closure_groups[group][family] += 1
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        raise GeometryError("cannot stream-decompress the frozen asset") from exc

    strict_axis = _axis_geometry(strict_groups, strict_axis=True)
    closure_axis = {
        "status": FAMILY_CLOSURE_STATUS,
        **_axis_geometry(closure_groups, strict_axis=False),
    }
    strict_count = sum(strict_family_counts.values())
    return {
        "header_observation": {
            "status": "PASS_EXACT_HEADER_NAMES",
            "column_name_count": len(EXPECTED_HEADER),
            "identifier_column_index": 0,
            "forbidden_body_value_column_count": len(EXPECTED_HEADER) - 1,
        },
        "total_body_row_count": total_rows,
        "strict_grammar_row_count": strict_count,
        "strict_role_family_row_counts": {
            family: strict_family_counts[family] for family in ROLE_FAMILIES
        },
        "identifier_grammar_anomaly_counts": {
            classification: anomaly_counts[classification]
            for classification in ANOMALY_CLASSES
        },
        "strict_axis": strict_axis,
        "reasoned_family_closure_axis": closure_axis,
        "body_access_attestation": {
            "whole_asset_stream_transport_and_decompression_performed": True,
            "full_row_bytes_discarded_after_first_tab_without_decoding_or_tokenizing_forbidden_fields": True,
            "identifier_body_cell_decoded_count": total_rows,
            "identifier_body_cell_parsed_count": total_rows,
            "role_token_derived_from_identifier_count": total_rows,
            "body_columns_decoded_or_parsed": ["ID"],
            "forbidden_body_cells": {
                field_class: {
                    "decoded_count": 0,
                    "parsed_count": 0,
                    "stored_count": 0,
                    "output_count": 0,
                }
                for field_class in ("SEQUENCE", "EFFECT", "CPM", "OTHER")
            },
        },
    }


def _geometry_projection(observation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(observation[key])
        for key in EXPECTED_GEOMETRY
    }


def _validate_observation(
    protocol: Mapping[str, Any], observation: Mapping[str, Any]
) -> None:
    header = _mapping(
        observation.get("header_observation"), label="header observation"
    )
    if dict(header) != {
        "status": "PASS_EXACT_HEADER_NAMES",
        "column_name_count": 10,
        "identifier_column_index": 0,
        "forbidden_body_value_column_count": 9,
    }:
        raise GeometryError("header observation differs")
    access = _mapping(
        observation.get("body_access_attestation"), label="body access attestation"
    )
    expected_access = {
        "whole_asset_stream_transport_and_decompression_performed": True,
        "full_row_bytes_discarded_after_first_tab_without_decoding_or_tokenizing_forbidden_fields": True,
        "identifier_body_cell_decoded_count": EXPECTED_GEOMETRY[
            "total_body_row_count"
        ],
        "identifier_body_cell_parsed_count": EXPECTED_GEOMETRY[
            "total_body_row_count"
        ],
        "role_token_derived_from_identifier_count": EXPECTED_GEOMETRY[
            "total_body_row_count"
        ],
        "body_columns_decoded_or_parsed": ["ID"],
        "forbidden_body_cells": {
            field_class: {
                "decoded_count": 0,
                "parsed_count": 0,
                "stored_count": 0,
                "output_count": 0,
            }
            for field_class in ("SEQUENCE", "EFFECT", "CPM", "OTHER")
        },
    }
    if dict(access) != expected_access:
        raise GeometryError("body-access boundary differs")
    expected = protocol["expected_aggregate_geometry"]
    if _geometry_projection(observation) != expected:
        raise GeometryError("aggregate identifier/pool geometry differs")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_report(
    protocol: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    binding: Mapping[str, str],
    asset_identity: Mapping[str, Any],
    recorded_at: str,
) -> dict[str, Any]:
    _validate_observation(protocol, observation)
    if dict(asset_identity) != {
        "filename": OFFICIAL_ASSET["filename"],
        "compressed_bytes": OFFICIAL_ASSET["compressed_bytes"],
        "compressed_sha256": OFFICIAL_ASSET["compressed_sha256"],
        "identity_status": "PASS_FROZEN_ORDINARY_PUBLIC_ASSET",
    }:
        raise AssetIdentityError("report asset identity is not the frozen public asset")
    context = protocol["frozen_public_context"]
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "contract_id": "mrna_xeditflow_route_a_v3",
        "phase_id": "A1",
        "dataset_id": DATASET_ID,
        "bioproject_id": BIOPROJECT_ID,
        "decision_id": DECISION_ID,
        "recorded_at": recorded_at,
        "status": COMPLETION_STATUS,
        "preflight_complete": True,
        "implementation_binding": dict(binding),
        "ordinary_public_asset_identity": dict(asset_identity),
        "aggregate_context": {
            "assay_context": context["assay_context"],
            "library_pool_context": context["library_pool_context"],
            "biological_material_context": context["biological_material_context"],
            "construct_context": context["construct_context"],
            "measurement_context_category_count": len(
                context["measurement_context_categories"]
            ),
            "relevant_public_sample_context_count": context[
                "relevant_public_sample_context_count"
            ],
            "public_context_authority_count": len(
                context["context_source_authority_ids"]
            ),
        },
        "header_observation": copy.deepcopy(observation["header_observation"]),
        "aggregate_pool_geometry": _geometry_projection(observation),
        "scope_attestation": {
            "authority_role": "PUBLIC_IDENTIFIER_AND_POOL_GEOMETRY_PREFLIGHT_ONLY",
            "ordinary_public_only": True,
            "aggregate_output_only": True,
            "compressed_asset_bytes_verified_before_decompression": True,
            **copy.deepcopy(observation["body_access_attestation"]),
            "row_record_output_count": 0,
            "member_identifier_output_count": 0,
            "member_role_output_count": 0,
            "member_context_output_count": 0,
            "sequence_value_output_count": 0,
            "effect_value_output_count": 0,
            "cpm_value_output_count": 0,
            "private_or_restricted_input_read_count": 0,
            "sealed_contact_count": 0,
            "GSE246381_contact_count": 0,
        },
        "terminal_truth": copy.deepcopy(dict(protocol["frozen_outer_truth"])),
        "interpretation_boundary": {
            "strict_axis_is_frozen_observed_identifier_grammar": True,
            "publisher_identifier_grammar_documented": False,
            "family_closure_axis_status": FAMILY_CLOSURE_STATUS,
            "family_closure_axis_is_publisher_explicit": False,
            "family_closure_axis_replaces_strict_axis": False,
            "aggregate_geometry_is_dataset_qualification": False,
            "aggregate_geometry_is_study_count_credit": False,
            "aggregate_geometry_establishes_true_a2": False,
            "sequence_edit_semantics": "NOT_EVALUATED_OUT_OF_SCOPE",
            "edit_budget_status": "NOT_EVALUATED_OUT_OF_SCOPE",
            "effect_status": "NOT_EVALUATED_OUT_OF_SCOPE",
        },
        "sole_next_action": "STOP_NO_FURTHER_ACTION_AUTHORIZED_BY_V3_DEC_021",
        "claim_boundary": protocol["claim_boundary"],
    }


def _write_temp_payload(temp_path: Path, payload: bytes) -> None:
    with temp_path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_exclusive(output_dir: Path, report: Mapping[str, Any]) -> Path:
    temporary_path: Path | None = None
    try:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise OutputError("output directory is not empty")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / REPORT_FILENAME
        if output_path.exists():
            raise OutputError("aggregate report already exists")
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{REPORT_FILENAME}.",
            suffix=".tmp",
            dir=output_dir,
        )
        os.close(descriptor)
        temporary_path = Path(temporary)
        _write_temp_payload(temporary_path, _json_bytes(report))
        if output_path.exists():
            raise OutputError("aggregate report appeared during write")
        os.replace(temporary_path, output_path)
        temporary_path = None
    except OutputError:
        raise
    except OSError as exc:
        raise OutputError("cannot write aggregate-only output") from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return output_path


BindingAuditor = Callable[
    [Mapping[str, Any], Path, bytes, Path], Mapping[str, str]
]
AssetIdentityAuditor = Callable[
    [Mapping[str, Any], Path], Mapping[str, Any]
]
GeometryAggregator = Callable[[Path], Mapping[str, Any]]


def execute(
    protocol_path: Path,
    asset_path: Path,
    output_dir: Path,
    *,
    repo_root: Path | None = None,
    binding_auditor: BindingAuditor = _default_binding_auditor,
    asset_identity_auditor: AssetIdentityAuditor = _default_asset_identity_auditor,
    geometry_aggregator: GeometryAggregator = aggregate_asset_geometry,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    protocol, protocol_payload = load_protocol(protocol_path)
    bound_repo_root = repo_root or protocol_path.parent.parent
    binding = binding_auditor(
        protocol,
        protocol_path,
        protocol_payload,
        bound_repo_root,
    )
    asset_identity = asset_identity_auditor(protocol, asset_path)
    observation = geometry_aggregator(asset_path)
    report = build_report(
        protocol,
        observation,
        binding=binding,
        asset_identity=asset_identity,
        recorded_at=recorded_at or _utc_now(),
    )
    _write_exclusive(output_dir, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--asset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = execute(
            args.protocol,
            args.asset,
            args.output_dir,
            repo_root=args.repo_root,
        )
    except PreflightError as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
