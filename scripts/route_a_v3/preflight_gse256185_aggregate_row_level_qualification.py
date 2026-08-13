#!/usr/bin/env python3
"""DEC-022 aggregate-only GSE256185 row-level qualification preflight.

The implementation verifies Git authority before touching either ordinary-public
asset, verifies both compressed assets before decompression, and keeps all row
material in memory.  The sole persisted artifact is a finite aggregate JSON
report that cannot qualify, count, canonicalize, train, or unlock a later phase.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


SCHEMA_VERSION = "route_a_v3_gse256185_aggregate_row_level_qualification_preflight.v1"
PROTOCOL_ID = "GSE256185_AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_V1"
PROTOCOL_BASENAME = (
    "route_a_v3_gse256185_aggregate_row_level_qualification_preflight_v1.json"
)
REPORT_FILENAME = "GSE256185_AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_V1.json"
DECISION_ID = "V3-DEC-022"
DATASET_ID = "GSE256185"
UNKNOWN = "UNKNOWN_NOT_ASSERTED"
BOUND = "BOUND"

CONFIG_PATH = f"configs/{PROTOCOL_BASENAME}"
SCRIPT_PATH = (
    "scripts/route_a_v3/"
    "preflight_gse256185_aggregate_row_level_qualification.py"
)
TEST_PATH = (
    "tests/route_a_v3/"
    "test_preflight_gse256185_aggregate_row_level_qualification.py"
)
EXPECTED_EXACT3 = (CONFIG_PATH, SCRIPT_PATH, TEST_PATH)
AUTHORITY_PARENT = "c57f5aa937d33d7e5ec1c25d3e29b339628c6387"
AUTHORITY_COMMIT = "4fa39abca424bb6ff82e43a847332e92934b278b"
RUNTIME_I_COMMIT = "71a9a327c6878792b165fb6b23ca623307aa6a8b"
RUNTIME_B_COMMIT = "ab511527a110dc17bc3538ee5309600396693534"
AUTHORITY_EXACT10 = (
    "configs/route_a_v3.yaml",
    "configs/route_a_v3_a1_qualification.json",
    "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec022.yaml",
    "docs/contracts/supersession_mrna_xeditflow_v1_1_to_route_a_v3.yaml",
    "docs/execution/route_a_v3_a1_interim.yaml",
    "docs/execution/route_a_v3_data_role_registry.yaml",
    "docs/execution/route_a_v3_decision_log.yaml",
    "docs/execution/route_a_v3_registry_manifest.json",
    "scripts/route_a_v3/validate_a0_bundle.py",
    "tests/route_a_v3/test_a0_integrity_guards.py",
)
RUNTIME_CONFIG_PATH = "configs/route_a_v3_dec022_authority_runtime_sync_v1.json"
RUNTIME_SCRIPT_PATH = "scripts/route_a_v3/dec022_authority_runtime_sync.py"
RUNTIME_TEST_PATH = "tests/route_a_v3/test_dec022_authority_runtime_sync.py"
RUNTIME_EXACT3 = (RUNTIME_CONFIG_PATH, RUNTIME_SCRIPT_PATH, RUNTIME_TEST_PATH)
RUNTIME_I_BLOBS = {
    RUNTIME_CONFIG_PATH: "a57f8c9db71ca5c249478bc21fd47e220ac3d3bab5b21e068e674f98ce0e2a4e",
    RUNTIME_SCRIPT_PATH: "f2e3e2ec4ef0f2589e349e0e6807170e8428433fbafa4723aadc6a593a54be43",
    RUNTIME_TEST_PATH: "e56ade44081ba637600e4343d45aa5da7fb20d5b7889048ea8953b5dfe3a8d96",
}
RUNTIME_B_BLOBS = {
    RUNTIME_CONFIG_PATH: "5fce02d828301ea635b2676e7a8384253d093762c1c0f4ee875b155d60c199ff",
    RUNTIME_SCRIPT_PATH: RUNTIME_I_BLOBS[RUNTIME_SCRIPT_PATH],
    RUNTIME_TEST_PATH: RUNTIME_I_BLOBS[RUNTIME_TEST_PATH],
}
UNKNOWN_BINDING_SCALARS = (
    "status",
    "implementation_commit",
    "implementation_script_sha256",
    "implementation_test_sha256",
)
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

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
GROUP_RE = re.compile(r"^ENSG\d+-ENST\d+-\d+$")
WIN_RE = re.compile(r"^win(\d+)$")
CCC_RE = re.compile(r"^([+-])(\d+)CCC$")
RAND_RE = re.compile(r"^rand\d+$")

GATE_IDS = (
    "STRICT_SINGLE_PARENT_CANDIDATE_UNIVERSE_CLOSED",
    "ROW_LEVEL_MULTI_ASSET_LINEAGE_AND_PRIMARY_MEASUREMENT_ROUTE_CLOSED",
    "SOURCE_AND_CANDIDATE_IDENTITY_CLOSED",
    "PARENT_TO_CANDIDATE_EDIT_REPLAY_CLOSED",
    "FAMILY_AND_CONTEXT_STRATIFICATION_CLOSED",
    "ENDPOINT_DIRECTION_SCALE_AND_SEMANTICS_CLOSED",
    "ORIGINAL_UNIT_AND_PAPER_FAITHFUL_TRANSFORM_CLOSED",
    "BIOLOGICAL_SOURCE_GROUP_AUTHORITY_CLOSED",
    "INDEPENDENT_BIOLOGICAL_REPLICATE_AND_VALID_STANDARD_ERROR_CLOSED",
    "LICENSE_AND_REDISTRIBUTION_RIGHTS_CLOSED",
    "OUTCOME_BLIND_SOURCE_GROUP_NEAR_DUPLICATE_SPLIT_AND_ZERO_LEAKAGE_CLOSED",
    "PREFROZEN_GROUP_POWER_AND_FULL_CI_WIDTH_CLOSED",
    "PUBLIC_PROVENANCE_AND_PRIMARY_MEASUREMENT_ROUTE_CLOSED",
    "MODEL_INPUT_ROUTE_AND_ROUTE_CONDITIONAL_EXPOSURE_CLOSED",
    "BENEFICIAL_SIGNAL_VERSUS_MEASUREMENT_NOISE_CLOSED",
    "POST_DEDUP_INDEPENDENT_EFFECTIVE_N_CLOSED",
    "REJECT_REASON_AND_EXCLUSION_CLOSURE_CLOSED",
)

STATUS_STOP = "STOP_CURRENT_PROTOCOL_NOT_QUALIFIED"
STATUS_READY = "READY_FOR_FORMAL_QUALIFIER_NOT_QUALIFIED"
PRODUCTION_REPO_ROOT = (
    "/home/cunyuliu/mrna_editflow_goal/worktrees/routea_v3_a1_20260810"
)
PRODUCTION_BRANCH = "routea-v3-a1-20260810"


class PreflightError(RuntimeError):
    """Base class for fail-closed preflight failures."""


class ProtocolError(PreflightError):
    """The protocol or its Git binding is invalid."""


class BindingNotFrozen(ProtocolError):
    """Authority or exact3-I/config-only-B binding is incomplete."""


class AssetIdentityError(PreflightError):
    """An ordinary-public asset is not the frozen asset."""


class ObservationError(PreflightError):
    """Aggregate recomputation differs from the frozen evidence."""


class OutputError(PreflightError):
    """The sole aggregate output cannot be published exclusively."""


def _strict_json(payload: bytes, *, label: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite token {token}")
            ),
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be a JSON object")
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be an object")
    return value


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
        raise PreflightError("aggregate report is not finite JSON") from exc


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "contract_id": "mrna_xeditflow_route_a_v3",
        "phase_id": "A1",
        "dataset_id": DATASET_ID,
        "bioproject_id": "PRJNA1078388",
        "decision_id": DECISION_ID,
        "protocol_status": (
            "AGGREGATE_ONLY_ROW_LEVEL_QUALIFICATION_PREFLIGHT_NOT_QUALIFICATION"
        ),
    }
    for key, expected in expected_scalars.items():
        if protocol.get(key) != expected:
            raise ProtocolError(f"protocol {key} differs")

    binding = _mapping(protocol.get("implementation_binding"), label="binding")
    if binding.get("binding_scheme") != (
        "DEC022_AUTHORITY_RUNTIME_BEFORE_EXACT3_I_CONFIG_ONLY_B_V1"
    ):
        raise ProtocolError("binding scheme differs")
    authority = _mapping(binding.get("authority_group"), label="authority group")
    authority_values = (
        authority.get("authority_commit"),
        authority.get("authority_runtime_binding_commit"),
    )
    authority_status = authority.get("status")
    authority_unknown = authority_values == (UNKNOWN, UNKNOWN)
    authority_bound = (
        authority_status == BOUND
        and authority_values == (AUTHORITY_COMMIT, RUNTIME_B_COMMIT)
    )
    if authority_status == UNKNOWN and not authority_unknown:
        raise ProtocolError("partial authority group is forbidden")
    if not authority_unknown and not authority_bound:
        raise ProtocolError("authority group must be all UNKNOWN or all BOUND")
    if authority.get("authority_expected_parent") != AUTHORITY_PARENT:
        raise ProtocolError("authority expected parent differs")
    if tuple(authority.get("authority_exact_changed_paths", ())) != AUTHORITY_EXACT10:
        raise ProtocolError("authority exact10 differs")
    runtime = _mapping(
        authority.get("authority_runtime_lifecycle"), label="authority runtime"
    )
    expected_runtime = {
        "paths": list(RUNTIME_EXACT3),
        "implementation_commit": RUNTIME_I_COMMIT,
        "implementation_expected_parent": AUTHORITY_COMMIT,
        "implementation_blob_sha256_by_path": RUNTIME_I_BLOBS,
        "binding_commit": RUNTIME_B_COMMIT,
        "binding_expected_parent": RUNTIME_I_COMMIT,
        "binding_blob_sha256_by_path": RUNTIME_B_BLOBS,
    }
    if dict(runtime) != expected_runtime:
        raise ProtocolError("authority runtime lifecycle differs")

    if tuple(binding.get("implementation_commit_exact_changed_paths", ())) != (
        EXPECTED_EXACT3
    ):
        raise ProtocolError("implementation commit must be exact3")
    if binding.get("binding_commit_exact_changed_paths") != [CONFIG_PATH]:
        raise ProtocolError("binding commit must be config-only")
    if binding.get("implementation_script_path") != SCRIPT_PATH:
        raise ProtocolError("implementation script path differs")
    if binding.get("implementation_test_path") != TEST_PATH:
        raise ProtocolError("implementation test path differs")
    if binding.get("unknown_to_bound_scalar_paths") != [
        f"implementation_binding.{field}" for field in UNKNOWN_BINDING_SCALARS
    ]:
        raise ProtocolError("implementation scalar group differs")
    normal_values = [binding.get(field) for field in UNKNOWN_BINDING_SCALARS]
    if binding.get("status") == UNKNOWN:
        if normal_values != [UNKNOWN] * 4:
            raise ProtocolError("partial implementation group is forbidden")
    elif binding.get("status") == BOUND:
        if not authority_bound:
            raise ProtocolError("BOUND implementation requires BOUND authority")
        if not (
            isinstance(binding.get("implementation_commit"), str)
            and HEX40_RE.fullmatch(str(binding["implementation_commit"]))
        ):
            raise ProtocolError("implementation commit is invalid")
        for field in ("implementation_script_sha256", "implementation_test_sha256"):
            if not (
                isinstance(binding.get(field), str)
                and HEX64_RE.fullmatch(str(binding[field]))
            ):
                raise ProtocolError(f"{field} is invalid")
    else:
        raise ProtocolError("implementation binding status is invalid")

    decision = _mapping(protocol.get("decision_authority"), label="authority")
    if decision.get("authorized_role") != (
        "AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_ONLY"
    ):
        raise ProtocolError("authorized role differs")
    if decision.get("allowed_input_field_classes_exactly") != [
        "IDENTIFIER",
        "ROLE",
        "SEQUENCE",
        "ENDPOINT",
        "REPLICATE",
        "NECESSARY_CONTEXT",
    ]:
        raise ProtocolError("allowed input classes differ")
    if decision.get("allowed_output_class") != (
        "AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_ONLY"
    ):
        raise ProtocolError("allowed output class differs")
    repository = _mapping(
        protocol.get("repository_authority"), label="repository authority"
    )
    if dict(repository) != {
        "production_repo_root": PRODUCTION_REPO_ROOT,
        "branch": PRODUCTION_BRANCH,
        "upstream_ref": f"origin/{PRODUCTION_BRANCH}",
        "required_state_before_asset_or_output_io": (
            "CLEAN_HEAD_EQUALS_UPSTREAM_EQUALS_ORIGIN"
        ),
    }:
        raise ProtocolError("production repository authority differs")

    assets = _mapping(protocol.get("official_public_assets"), label="assets")
    if assets.get("identity_mismatch_action") != (
        "STOP_BEFORE_DECOMPRESSION_ROW_LEVEL_ACCESS_OR_OUTPUT"
    ):
        raise ProtocolError("asset mismatch action differs")
    for key in ("processed_tsv", "reference_fasta"):
        asset = _mapping(assets.get(key), label=key)
        if not (
            isinstance(asset.get("compressed_bytes"), int)
            and isinstance(asset.get("compressed_sha256"), str)
            and HEX64_RE.fullmatch(str(asset["compressed_sha256"]))
        ):
            raise ProtocolError(f"{key} identity is invalid")
    fasta = assets["reference_fasta"]
    if fasta.get("decompressed_bytes") != 17514341 or fasta.get(
        "decompressed_sha256"
    ) != "5e415248c7ca8c8f74859760a4287710e9fa2147c149a97f15cbbacab45e2d36":
        raise ProtocolError("FASTA content identity differs")

    header = _mapping(protocol.get("header_contract"), label="header")
    if tuple(header.get("exact_column_names", ())) != EXPECTED_HEADER:
        raise ProtocolError("TSV header contract differs")
    universe = _mapping(
        protocol.get("candidate_universe_contract"), label="candidate universe"
    )
    frozen_universe = {
        "total_body_row_count": 11404,
        "strict_grammar_row_count": 11402,
        "strict_group_count": 652,
        "strict_single_parent_group_count": 637,
        "strict_dual_parent_group_count_excluded": 15,
        "strict_single_parent_two_candidate_group_count_excluded": 3,
        "nonstrict_record_count_excluded": 2,
        "review_pool_count": 634,
        "review_parent_row_count": 634,
        "review_candidate_row_count": 7292,
        "review_row_count": 7926,
        "review_candidate_family_counts": {
            "win": 5124,
            "+CCC": 1090,
            "-CCC": 1078,
            "rand": 0,
        },
    }
    for key, expected in frozen_universe.items():
        if universe.get(key) != expected:
            raise ProtocolError(f"candidate universe {key} differs")

    if tuple(protocol.get("required_gate_ids_exactly", ())) != GATE_IDS:
        raise ProtocolError("required gate list differs")
    output = _mapping(protocol.get("output_contract"), label="output")
    if output.get("filename") != REPORT_FILENAME:
        raise ProtocolError("output filename differs")
    if output.get("single_aggregate_output_only") is not True:
        raise ProtocolError("output must remain aggregate-only")
    for key in (
        "persistent_row_level_intermediate_allowed",
        "member_identifier_output_allowed",
        "member_role_output_allowed",
        "member_context_output_allowed",
        "sequence_output_allowed",
        "row_effect_output_allowed",
        "raw_replicate_value_output_allowed",
        "replicate_identifier_output_allowed",
        "split_assignment_output_allowed",
        "canonical_record_output_allowed",
        "qualification_or_credit_output_allowed",
    ):
        if output.get(key) is not False:
            raise ProtocolError(f"output boundary {key} differs")
    outer = _mapping(protocol.get("frozen_outer_truth"), label="outer truth")
    if outer.get("current_qualified_counts") != {
        "ordinary": 1,
        "a1": 1,
        "true_a2": 0,
        "canonical_records": 6547,
    }:
        raise ProtocolError("outer qualified counts differ")
    if outer.get("gse256185_contribution") != {
        "ordinary": 0,
        "a1": 0,
        "true_a2": 0,
        "canonical_records": 0,
    }:
        raise ProtocolError("dataset contribution differs")
    for key in (
        "gse256185_qualified",
        "a1_complete",
        "training_allowed",
        "gpu_work_allowed",
        "model_selection_allowed",
        "next_phase_authorized",
    ):
        if outer.get(key) is not False:
            raise ProtocolError(f"outer lock {key} differs")


def load_protocol(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.name != PROTOCOL_BASENAME:
        raise ProtocolError("protocol basename differs")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ProtocolError("cannot read protocol") from exc
    protocol = _strict_json(payload, label="protocol")
    _validate_protocol(protocol)
    return protocol, payload


def _run_git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise ProtocolError("git unavailable") from exc
    if result.returncode != 0:
        raise ProtocolError("git binding check failed")
    return result.stdout.strip()


def _git_blob(repo_root: Path, commit: str, path: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{commit}:{path}"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise ProtocolError("git unavailable") from exc
    if result.returncode != 0:
        raise ProtocolError("cannot read bound Git blob")
    return result.stdout


def _changed_paths(repo_root: Path, commit: str) -> tuple[str, ...]:
    value = _run_git(
        repo_root, "diff-tree", "--no-commit-id", "--name-only", "-r", commit
    )
    return tuple(sorted(line for line in value.splitlines() if line))


def _normalise_binding(protocol: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(protocol))
    binding = result["implementation_binding"]
    for key in UNKNOWN_BINDING_SCALARS:
        binding[key] = UNKNOWN
    return result


def _verify_frozen_commit(
    repo_root: Path,
    *,
    label: str,
    commit: str,
    expected_parent: str,
    expected_paths: tuple[str, ...],
    expected_blobs: Mapping[str, str] | None = None,
) -> None:
    if _run_git(repo_root, "rev-parse", f"{commit}^") != expected_parent:
        raise ProtocolError(f"{label} parent differs")
    if _changed_paths(repo_root, commit) != tuple(sorted(expected_paths)):
        raise ProtocolError(f"{label} changed-path closure differs")
    for path, expected in (expected_blobs or {}).items():
        if hashlib.sha256(_git_blob(repo_root, commit, path)).hexdigest() != expected:
            raise ProtocolError(f"{label} blob identity differs: {path}")


def _default_binding_auditor(
    protocol: Mapping[str, Any],
    protocol_path: Path,
    protocol_payload: bytes,
    repo_root: Path,
) -> dict[str, str]:
    del protocol_payload
    binding = protocol["implementation_binding"]
    authority = binding["authority_group"]
    if authority.get("status") != BOUND:
        raise BindingNotFrozen("DEC022 authority/runtime group is not BOUND")
    if binding.get("status") != BOUND:
        raise BindingNotFrozen("exact3-I/config-only-B lifecycle is not BOUND")
    if any(binding.get(field) == UNKNOWN for field in UNKNOWN_BINDING_SCALARS):
        raise BindingNotFrozen("implementation binding group remains UNKNOWN")

    repository = protocol["repository_authority"]
    if repo_root.resolve() != Path(repository["production_repo_root"]).resolve():
        raise ProtocolError("execution repository is not the frozen production root")
    binding_commit = _run_git(repo_root, "rev-parse", "HEAD")
    upstream = _run_git(repo_root, "rev-parse", "@{upstream}")
    origin = _run_git(
        repo_root,
        "rev-parse",
        "--verify",
        f"refs/remotes/origin/{repository['branch']}",
    )
    if binding_commit != upstream or binding_commit != origin:
        raise ProtocolError("HEAD, upstream, and origin do not match")
    if _run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD") != repository[
        "branch"
    ]:
        raise ProtocolError("production branch differs")
    if _run_git(repo_root, "rev-parse", "--abbrev-ref", "@{upstream}") != (
        repository["upstream_ref"]
    ):
        raise ProtocolError("production upstream branch differs")
    if _run_git(
        repo_root, "status", "--porcelain=v1", "--untracked-files=all"
    ):
        raise ProtocolError("production worktree or index is dirty")
    implementation_commit = str(binding["implementation_commit"])
    authority_commit = str(authority["authority_commit"])
    runtime_binding_commit = str(authority["authority_runtime_binding_commit"])
    _verify_frozen_commit(
        repo_root,
        label="DEC022 authority A",
        commit=authority_commit,
        expected_parent=AUTHORITY_PARENT,
        expected_paths=AUTHORITY_EXACT10,
    )
    _verify_frozen_commit(
        repo_root,
        label="DEC022 runtime I",
        commit=RUNTIME_I_COMMIT,
        expected_parent=authority_commit,
        expected_paths=RUNTIME_EXACT3,
        expected_blobs=RUNTIME_I_BLOBS,
    )
    _verify_frozen_commit(
        repo_root,
        label="DEC022 runtime B",
        commit=runtime_binding_commit,
        expected_parent=RUNTIME_I_COMMIT,
        expected_paths=(RUNTIME_CONFIG_PATH,),
        expected_blobs=RUNTIME_B_BLOBS,
    )
    if _run_git(repo_root, "rev-parse", f"{implementation_commit}^") != (
        runtime_binding_commit
    ):
        raise ProtocolError("implementation I is not direct child of runtime B")
    if _run_git(repo_root, "rev-parse", f"{binding_commit}^") != implementation_commit:
        raise ProtocolError("binding commit is not direct config-only child of I")
    if _changed_paths(repo_root, implementation_commit) != tuple(sorted(EXPECTED_EXACT3)):
        raise ProtocolError("implementation commit is not exact3")
    if _changed_paths(repo_root, binding_commit) != (CONFIG_PATH,):
        raise ProtocolError("binding commit is not config-only")

    implementation_config = _strict_json(
        _git_blob(repo_root, implementation_commit, CONFIG_PATH),
        label="implementation protocol",
    )
    if _normalise_binding(protocol) != implementation_config:
        raise ProtocolError("binding commit changed more than four scalars")
    script_blob = _git_blob(repo_root, implementation_commit, SCRIPT_PATH)
    test_blob = _git_blob(repo_root, implementation_commit, TEST_PATH)
    if hashlib.sha256(script_blob).hexdigest() != binding.get(
        "implementation_script_sha256"
    ):
        raise ProtocolError("bound script identity differs")
    if hashlib.sha256(test_blob).hexdigest() != binding.get(
        "implementation_test_sha256"
    ):
        raise ProtocolError("bound test identity differs")
    if protocol_path.resolve() != (repo_root / CONFIG_PATH).resolve():
        raise ProtocolError("protocol path is outside bound repository location")
    if protocol_path.read_bytes() != _git_blob(repo_root, binding_commit, CONFIG_PATH):
        raise ProtocolError("working protocol differs from B")
    bound_working_script = (repo_root / SCRIPT_PATH).resolve()
    if bound_working_script.read_bytes() != script_blob:
        raise ProtocolError("working script differs from I")
    executing_script = Path(__file__).resolve()
    if executing_script != bound_working_script:
        raise ProtocolError("executing producer is not the bound repository script")
    if executing_script.read_bytes() != script_blob:
        raise ProtocolError("executing producer differs from bound implementation I")
    if (repo_root / TEST_PATH).read_bytes() != test_blob:
        raise ProtocolError("working test differs from I")
    return {
        "status": "BOUND_DEC022_AUTHORITY_RUNTIME_EXACT3_I_CONFIG_ONLY_B_VERIFIED",
        "authority_commit": authority_commit,
        "authority_runtime_binding_commit": runtime_binding_commit,
        "implementation_commit": implementation_commit,
        "binding_commit": binding_commit,
        "upstream_commit": upstream,
        "origin_commit": origin,
    }


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                total += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise AssetIdentityError("cannot read ordinary-public asset") from exc
    return total, digest.hexdigest()


def _default_asset_identity_auditor(
    protocol: Mapping[str, Any], tsv_path: Path, fasta_path: Path
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    paths = {"processed_tsv": tsv_path, "reference_fasta": fasta_path}
    for key, path in paths.items():
        expected = protocol["official_public_assets"][key]
        if path.name != expected["filename"]:
            raise AssetIdentityError(f"{key} filename differs")
        byte_count, sha256 = _sha256_file(path)
        if byte_count != expected["compressed_bytes"]:
            raise AssetIdentityError(f"{key} compressed byte count differs")
        if sha256 != expected["compressed_sha256"]:
            raise AssetIdentityError(f"{key} compressed SHA-256 differs")
        results[key] = {
            "filename": expected["filename"],
            "compressed_bytes": byte_count,
            "compressed_sha256": sha256,
            "identity_status": "PASS_FROZEN_ORDINARY_PUBLIC_ASSET",
        }
    return results


def _parse_strict_identifier(identifier: str) -> tuple[str, str] | None:
    parts = identifier.rsplit(".", 2)
    if len(parts) != 3:
        return None
    group, role, suffix = parts
    if not GROUP_RE.fullmatch(group) or not suffix:
        return None
    if role == "parent" or WIN_RE.fullmatch(role) or CCC_RE.fullmatch(role) or RAND_RE.fullmatch(role):
        return group, role
    return None


def _family(role: str) -> str:
    if WIN_RE.fullmatch(role):
        return "win"
    if role.startswith("+"):
        return "+CCC"
    if role.startswith("-"):
        return "-CCC"
    if RAND_RE.fullmatch(role):
        return "rand"
    return "parent"


def _expected_delta(role: str) -> int:
    if role == "parent" or RAND_RE.fullmatch(role):
        return 0
    win = WIN_RE.fullmatch(role)
    if win:
        return -6
    ccc = CCC_RE.fullmatch(role)
    if ccc is None:
        raise ObservationError("unexpected role during edit replay")
    magnitude = 3 * int(ccc.group(2))
    return magnitude if ccc.group(1) == "+" else -magnitude


def _publisher_transform(sequence: str) -> str:
    if len(sequence) < 20 or sequence[:19] != "GCTAATACGACTCACTATA":
        raise ObservationError("FASTA promoter prefix differs")
    transformed = sequence[19:]
    if not transformed.startswith("A"):
        raise ObservationError("FASTA first post-promoter base is not A")
    return "G" + transformed[1:]


def _c_run_signature(sequence: str) -> tuple[str, tuple[int, ...]]:
    non_c: list[str] = []
    runs: list[int] = []
    count = 0
    for base in sequence:
        if base == "C":
            count += 1
        else:
            runs.append(count)
            count = 0
            non_c.append(base)
    runs.append(count)
    return "".join(non_c), tuple(runs)


def _ccc_deletion_only(parent: str, candidate: str) -> bool:
    parent_non_c, parent_runs = _c_run_signature(parent)
    candidate_non_c, candidate_runs = _c_run_signature(candidate)
    return (
        parent_non_c == candidate_non_c
        and len(parent_runs) == len(candidate_runs)
        and all(after <= before for before, after in zip(parent_runs, candidate_runs))
    )


def _reverse_publisher_atg_to_agt(sequence: str) -> list[str]:
    positions = [match.start() for match in re.finditer("AGT", sequence)]
    variants: list[str] = []
    for edit_count in range(1, len(positions) + 1):
        for selected in itertools.combinations(positions, edit_count):
            chars = list(sequence)
            for position in selected:
                chars[position + 1] = "T"
                chars[position + 2] = "G"
            variants.append("".join(chars))
    return variants


def replay_edit(parent: str, candidate: str, role: str) -> tuple[str, bool]:
    """Return DIRECT, PUBLISHER_ASSISTED, or UNEXPLAINED plus win legality."""

    win = WIN_RE.fullmatch(role)
    if win:
        offset = int(win.group(1))
        legal = 0 <= offset <= len(parent) - 6
        if not legal:
            return "UNEXPLAINED", False
        expected = parent[:offset] + parent[offset + 6 :]
        if candidate == expected:
            return "DIRECT", True
        if candidate == expected.replace("ATG", "AGT"):
            return "PUBLISHER_ASSISTED", True
        return "UNEXPLAINED", True

    ccc = CCC_RE.fullmatch(role)
    if ccc is None:
        return "UNEXPLAINED", True
    if ccc.group(1) == "+":
        return ("DIRECT" if _ccc_deletion_only(candidate, parent) else "UNEXPLAINED"), True
    if _ccc_deletion_only(parent, candidate):
        return "DIRECT", True
    if any(_ccc_deletion_only(parent, value) for value in _reverse_publisher_atg_to_agt(candidate)):
        return "PUBLISHER_ASSISTED", True
    return "UNEXPLAINED", True


def recompute_endpoint(values: list[float]) -> float:
    if len(values) != 7:
        raise ObservationError("endpoint recomputation needs seven CPM values")
    ivt_mean = sum(values[4:]) / 3.0
    if ivt_mean <= 0 or any(value <= 0 for value in values[:4]):
        raise ObservationError("endpoint is undefined for zero/nonpositive CPM")
    return sum(math.log2(value / ivt_mean) for value in values[:4]) / 4.0


def _read_fasta(path: Path, expected: Mapping[str, Any]) -> dict[str, str]:
    digest = hashlib.sha256()
    decompressed_bytes = 0
    records: dict[str, str] = {}
    header: str | None = None
    sequence_parts: list[str] = []
    try:
        with gzip.open(path, "rb") as handle:
            for raw_line in handle:
                digest.update(raw_line)
                decompressed_bytes += len(raw_line)
                line = raw_line.rstrip(b"\r\n")
                if line.startswith(b">"):
                    if header is not None:
                        if header in records:
                            raise ObservationError("duplicate FASTA header")
                        records[header] = "".join(sequence_parts)
                    try:
                        header = line[1:].decode("ascii")
                    except UnicodeDecodeError as exc:
                        raise ObservationError("FASTA header is not ASCII") from exc
                    sequence_parts = []
                else:
                    if header is None:
                        raise ObservationError("FASTA sequence precedes header")
                    try:
                        sequence_parts.append(line.decode("ascii"))
                    except UnicodeDecodeError as exc:
                        raise ObservationError("FASTA sequence is not ASCII") from exc
        if header is not None:
            if header in records:
                raise ObservationError("duplicate FASTA header")
            records[header] = "".join(sequence_parts)
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        raise ObservationError("cannot decompress FASTA") from exc
    if decompressed_bytes != expected["decompressed_bytes"]:
        raise ObservationError("FASTA decompressed byte count differs")
    if digest.hexdigest() != expected["decompressed_sha256"]:
        raise ObservationError("FASTA decompressed SHA-256 differs")
    if len(records) != expected["fasta_record_count"]:
        raise ObservationError("FASTA record count differs")
    return records


def _scan_tsv_universe(
    path: Path, expected: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """First pass: decode only identifiers and close the candidate universe."""

    strict_roles: defaultdict[str, list[str]] = defaultdict(list)
    nonstrict = 0
    total_rows = 0
    seen_ids: set[str] = set()
    try:
        with gzip.open(path, "rb") as handle:
            header_line = handle.readline().rstrip(b"\r\n")
            try:
                header = tuple(value.decode("ascii") for value in header_line.split(b"\t"))
            except UnicodeDecodeError as exc:
                raise ObservationError("TSV header is not ASCII") from exc
            if header != EXPECTED_HEADER:
                raise ObservationError("TSV header differs")
            for raw_line in handle:
                total_rows += 1
                identifier_bytes, separator, _ = raw_line.partition(b"\t")
                if not separator or not identifier_bytes:
                    raise ObservationError("TSV row has no identifier field")
                try:
                    identifier = identifier_bytes.decode("ascii")
                except UnicodeDecodeError as exc:
                    raise ObservationError("TSV identifier is not ASCII") from exc
                if identifier in seen_ids:
                    raise ObservationError("duplicate TSV identifier")
                seen_ids.add(identifier)
                parsed = _parse_strict_identifier(identifier)
                if parsed is None:
                    nonstrict += 1
                else:
                    strict_roles[parsed[0]].append(parsed[1])
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        raise ObservationError("cannot scan TSV identifier universe") from exc
    geometry = {
        "total_body_row_count": total_rows,
        "strict_grammar_row_count": sum(map(len, strict_roles.values())),
        "strict_group_count": len(strict_roles),
        "strict_single_parent_group_count": sum(
            roles.count("parent") == 1 for roles in strict_roles.values()
        ),
        "strict_dual_parent_group_count_excluded": sum(
            roles.count("parent") == 2 for roles in strict_roles.values()
        ),
        "strict_single_parent_two_candidate_group_count_excluded": sum(
            roles.count("parent") == 1
            and sum(role != "parent" for role in roles) == 2
            for roles in strict_roles.values()
        ),
        "nonstrict_record_count_excluded": nonstrict,
    }
    universe_groups = {
        group
        for group, roles in strict_roles.items()
        if roles.count("parent") == 1
        and sum(role != "parent" for role in roles) >= 3
    }
    universe_roles = [
        role
        for group in universe_groups
        for role in strict_roles[group]
        if role != "parent"
    ]
    family_counts = Counter(_family(role) for role in universe_roles)
    candidate_universe = {
        **geometry,
        "review_pool_count": len(universe_groups),
        "review_parent_row_count": len(universe_groups),
        "review_candidate_row_count": len(universe_roles),
        "review_row_count": len(universe_roles) + len(universe_groups),
        "review_candidate_family_counts": {
            family: family_counts[family]
            for family in ("win", "+CCC", "-CCC", "rand")
        },
    }
    if expected is not None:
        frozen = {
            key: value
            for key, value in expected.items()
            if key not in {"selection_rule", "count_or_rule_drift_action"}
        }
        if candidate_universe != frozen:
            raise ObservationError(
                "candidate universe drift before row-level field access"
            )
    return {
        "candidate_universe": candidate_universe,
        "universe_groups": universe_groups,
        "identifiers": seen_ids,
    }


def _load_tsv_body(
    path: Path, scan: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Second pass: read authorized row-level fields after universe closure."""

    rows: list[dict[str, Any]] = []
    strict_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    try:
        with gzip.open(path, "rt", encoding="ascii", newline="") as handle:
            header_line = handle.readline()
            if tuple(header_line.rstrip("\r\n").split("\t")) != EXPECTED_HEADER:
                raise ObservationError("TSV header differs between passes")
            for raw_line in handle:
                fields = raw_line.rstrip("\r\n").split("\t")
                if len(fields) != len(EXPECTED_HEADER):
                    raise ObservationError("TSV row width differs")
                identifier = fields[0]
                if identifier in seen_ids:
                    raise ObservationError("duplicate TSV identifier")
                seen_ids.add(identifier)
                parsed = _parse_strict_identifier(identifier)
                row = {
                    "identifier": identifier,
                    "endpoint_text": fields[1],
                    "replicate_texts": fields[2:9],
                    "sequence": fields[9],
                }
                rows.append(row)
                if parsed is not None:
                    row["group"], row["role"] = parsed
                    strict_groups[parsed[0]].append(row)
    except (OSError, EOFError, UnicodeDecodeError, gzip.BadGzipFile) as exc:
        raise ObservationError("cannot read authorized TSV row-level fields") from exc
    if seen_ids != scan["identifiers"]:
        raise ObservationError("TSV identifiers changed between passes")
    universe = {
        group: strict_groups[group]
        for group in scan["universe_groups"]
    }
    return rows, {
        "geometry": copy.deepcopy(scan["candidate_universe"]),
        "strict_groups": strict_groups,
        "universe": universe,
    }


def _load_tsv(
    path: Path, expected: Mapping[str, Any] | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scan = _scan_tsv_universe(path, expected)
    return _load_tsv_body(path, scan)


def aggregate_assets(
    protocol: Mapping[str, Any], tsv_path: Path, fasta_path: Path
) -> dict[str, Any]:
    """Read both verified public assets and return aggregate evidence only."""

    expected_universe = protocol["candidate_universe_contract"]
    scan = _scan_tsv_universe(tsv_path, expected_universe)
    fasta = _read_fasta(
        fasta_path, protocol["official_public_assets"]["reference_fasta"]
    )
    rows, parsed = _load_tsv_body(tsv_path, scan)
    geometry = parsed["geometry"]
    universe = parsed["universe"]

    tsv_identities = {row["identifier"]: row for row in rows}
    matched = exact_lineage = 0
    for identifier, row in tsv_identities.items():
        fasta_sequence = fasta.get(identifier)
        if fasta_sequence is None:
            continue
        matched += 1
        if _publisher_transform(fasta_sequence) == row["sequence"]:
            exact_lineage += 1
    lineage = {
        "tsv_rows_reviewed": len(rows),
        "fasta_records_total": len(fasta),
        "fasta_headers_matched_to_tsv_rows": matched,
        "transformed_sequences_exactly_matched": exact_lineage,
        "transformed_sequence_mismatch_count": matched - exact_lineage,
        "missing_fasta_header_count": len(rows) - matched,
    }

    replay = Counter()
    delta_counts = Counter()
    eligible_groups: dict[str, dict[str, Any]] = {}
    nonfinite_count = formula_match = formula_mismatch = 0
    maximum_formula_difference = 0.0
    nonfinite_groups: set[str] = set()
    replay_failed_groups: set[str] = set()
    finite_rows = 0
    for group, group_rows in universe.items():
        parent = next(row for row in group_rows if row["role"] == "parent")
        length = int(group.rsplit("-", 1)[1])
        parent_utr = parent["sequence"][3 : 3 + length]
        eligible_candidates: list[dict[str, Any]] = []
        for row in group_rows:
            try:
                endpoint = float(row["endpoint_text"])
                replicates = [float(value) for value in row["replicate_texts"]]
                calculated = recompute_endpoint(replicates)
                if not math.isfinite(endpoint) or not math.isfinite(calculated):
                    raise ValueError("non-finite")
                finite_rows += 1
                formula_difference = abs(endpoint - calculated)
                maximum_formula_difference = max(
                    maximum_formula_difference, formula_difference
                )
                if formula_difference <= protocol["sequence_and_endpoint_protocol"]["endpoint_transform_absolute_tolerance"]:
                    formula_match += 1
                else:
                    formula_mismatch += 1
                row["endpoint"] = endpoint
            except (ValueError, ObservationError):
                nonfinite_count += 1
                nonfinite_groups.add(group)
                row["endpoint"] = None

            if row["role"] == "parent":
                continue
            delta = _expected_delta(row["role"])
            delta_counts[f"{delta:+d}"] += 1
            candidate_utr = row["sequence"][3 : 3 + length + delta]
            status, win_legal = replay_edit(parent_utr, candidate_utr, row["role"])
            replay[f"{_family(row['role'])}_{status}"] += 1
            replay[status] += 1
            if _family(row["role"]) == "win" and win_legal:
                replay["win_position_legal_count"] += 1
            if status == "UNEXPLAINED":
                replay_failed_groups.add(group)
            elif row["endpoint"] is not None:
                eligible_candidates.append(
                    {"role": row["role"], "utr": candidate_utr, "endpoint": row["endpoint"]}
                )
        if parent["endpoint"] is not None and eligible_candidates:
            eligible_groups[group] = {
                "parent_utr": parent_utr,
                "parent_endpoint": parent["endpoint"],
                "candidates": eligible_candidates,
            }

    endpoint = {
        "rows_reviewed": sum(len(value) for value in universe.values()),
        "finite_endpoint_and_replicate_row_count": finite_rows,
        "nonfinite_or_undefined_row_count": nonfinite_count,
        "formula_match_within_tolerance_count": formula_match,
        "formula_mismatch_count": formula_mismatch,
        "maximum_absolute_formula_difference": maximum_formula_difference,
    }
    replay_observation = {
        "candidate_rows_reviewed": geometry["review_candidate_row_count"],
        "expected_edit_length_delta_counts": {
            key: delta_counts[key]
            for key in ("-15", "-12", "-9", "-6", "-3", "+3", "+6", "+9", "+12", "+15")
        },
        "win_position_legal_count": replay["win_position_legal_count"],
        "win_direct_count": replay["win_DIRECT"],
        "win_publisher_assisted_count": replay["win_PUBLISHER_ASSISTED"],
        "plus_ccc_direct_count": replay["+CCC_DIRECT"],
        "minus_ccc_direct_count": replay["-CCC_DIRECT"],
        "minus_ccc_publisher_assisted_count": replay["-CCC_PUBLISHER_ASSISTED"],
        "direct_total": replay["DIRECT"],
        "publisher_assisted_total": replay["PUBLISHER_ASSISTED"],
        "replay_closed_total": replay["DIRECT"] + replay["PUBLISHER_ASSISTED"],
        "unexplained_count": replay["UNEXPLAINED"],
    }

    eligible_family_candidates = Counter()
    eligible_family_pools = Counter()
    direction = Counter()
    exact_utrs: list[tuple[str, str]] = []
    distinct_within_pool_candidates = 0
    for group, value in eligible_groups.items():
        candidates = value["candidates"]
        family = _family(candidates[0]["role"])
        eligible_family_pools[family] += 1
        eligible_family_candidates.update(_family(row["role"]) for row in candidates)
        candidate_signs: list[int] = []
        distinct_within_pool_candidates += len({row["utr"] for row in candidates})
        exact_utrs.append((group, value["parent_utr"]))
        for row in candidates:
            exact_utrs.append((group, row["utr"]))
            delta = row["endpoint"] - value["parent_endpoint"]
            sign = 1 if delta > 0 else -1 if delta < 0 else 0
            candidate_signs.append(sign)
            direction["candidate_positive_count" if sign > 0 else "candidate_negative_count" if sign < 0 else "candidate_zero_count"] += 1
        if any(sign > 0 for sign in candidate_signs) and any(sign < 0 for sign in candidate_signs):
            direction["mixed_direction_pool_count"] += 1
        elif any(sign > 0 for sign in candidate_signs):
            direction["positive_only_pool_count"] += 1
        elif any(sign < 0 for sign in candidate_signs):
            direction["negative_only_pool_count"] += 1
        else:
            direction["zero_only_pool_count"] += 1

    sequence_counts = Counter(sequence for _, sequence in exact_utrs)
    duplicate_sequences = {sequence for sequence, count in sequence_counts.items() if count > 1}
    genes = Counter(group.split("-", 1)[0] for group in eligible_groups)
    structural = {
        "source_group_count": len(eligible_groups),
        "gene_count": len(genes),
        "genes_with_multiple_source_groups": sum(count > 1 for count in genes.values()),
        "retained_row_count": len(exact_utrs),
        "exact_utr_cluster_count": len(sequence_counts),
        "duplicate_exact_utr_cluster_count": len(duplicate_sequences),
        "rows_in_duplicate_exact_utr_clusters": sum(sequence_counts[value] for value in duplicate_sequences),
        "cross_pool_exact_utr_cluster_count": sum(
            len({group for group, value in exact_utrs if value == sequence}) > 1
            for sequence in duplicate_sequences
        ),
        "candidate_row_count": sum(len(value["candidates"]) for value in eligible_groups.values()),
        "distinct_within_pool_candidate_count": distinct_within_pool_candidates,
        "parent_row_count": len(eligible_groups),
    }
    eligible = {
        "pool_count": len(eligible_groups),
        "parent_row_count": len(eligible_groups),
        "candidate_row_count": sum(eligible_family_candidates.values()),
        "row_count": len(exact_utrs),
        "candidate_family_counts": {
            family: eligible_family_candidates[family]
            for family in ("win", "+CCC", "-CCC")
        },
        "pool_family_counts": {
            family: eligible_family_pools[family]
            for family in ("win", "+CCC", "-CCC")
        },
    }
    direction_observation = {
        key: direction[key]
        for key in (
            "candidate_positive_count",
            "candidate_negative_count",
            "candidate_zero_count",
            "mixed_direction_pool_count",
            "positive_only_pool_count",
            "negative_only_pool_count",
            "zero_only_pool_count",
        )
    }
    strict_groups = parsed["strict_groups"]
    dual_rows = sum(
        len(group_rows)
        for group_rows in strict_groups.values()
        if sum(row["role"] == "parent" for row in group_rows) == 2
    )
    two_rows = sum(
        len(group_rows)
        for group_rows in strict_groups.values()
        if sum(row["role"] == "parent" for row in group_rows) == 1
        and sum(row["role"] != "parent" for row in group_rows) == 2
    )
    retained_rows = structural["retained_row_count"]
    reject = {
        "dual_parent_group_count": geometry["strict_dual_parent_group_count_excluded"],
        "dual_parent_group_row_count": dual_rows,
        "two_candidate_group_count": geometry["strict_single_parent_two_candidate_group_count_excluded"],
        "two_candidate_group_row_count": two_rows,
        "nonstrict_record_count": geometry["nonstrict_record_count_excluded"],
        "unexplained_edit_candidate_count": replay["UNEXPLAINED"],
        "unexplained_edit_affected_group_count": len(replay_failed_groups),
        "nonfinite_endpoint_row_count": nonfinite_count,
        "nonfinite_endpoint_affected_group_count": len(nonfinite_groups),
        "edit_failure_and_nonfinite_affected_group_overlap_count": len(replay_failed_groups & nonfinite_groups),
        "orphaned_parent_after_all_candidates_rejected_count": len(universe) - len(eligible_groups),
        "retained_row_count": retained_rows,
        "mutually_exclusive_row_reason_total": dual_rows + two_rows + geometry["nonstrict_record_count_excluded"] + replay["UNEXPLAINED"] + nonfinite_count + (len(universe) - len(eligible_groups)) + retained_rows,
    }
    return {
        "candidate_universe": copy.deepcopy(geometry),
        "multi_asset_lineage": lineage,
        "edit_replay": replay_observation,
        "endpoint_transform": endpoint,
        "eligible_after_row_preflight_exclusions": eligible,
        "beneficial_direction_counts": direction_observation,
        "structural_dedup_and_split_feasibility": structural,
        "reject_closure": reject,
        "internal_access_attestation": {
            "ordinary_public_assets_read_count": 2,
            "candidate_universe_closed_before_row_level_field_access": True,
            "row_level_values_persisted_count": 0,
            "row_level_values_serialized_count": 0,
            "private_or_restricted_input_read_count": 0,
            "sealed_contact_count": 0,
            "gse246381_contact_count": 0,
        },
    }


def _validate_observation(protocol: Mapping[str, Any], observation: Mapping[str, Any]) -> None:
    expected = protocol["expected_aggregate_observation"]
    for key in (
        "multi_asset_lineage",
        "edit_replay",
        "endpoint_transform",
        "eligible_after_row_preflight_exclusions",
        "beneficial_direction_counts",
        "structural_dedup_and_split_feasibility",
        "reject_closure",
    ):
        if observation.get(key) != expected[key]:
            raise ObservationError(f"aggregate {key} differs from frozen evidence")
    expected_universe = {
        key: value
        for key, value in protocol["candidate_universe_contract"].items()
        if key
        not in {
            "selection_rule",
            "count_or_rule_drift_action",
        }
    }
    if observation.get("candidate_universe") != expected_universe:
        raise ObservationError("aggregate candidate universe differs")
    if observation.get("internal_access_attestation") != {
        "ordinary_public_assets_read_count": 2,
        "candidate_universe_closed_before_row_level_field_access": True,
        "row_level_values_persisted_count": 0,
        "row_level_values_serialized_count": 0,
        "private_or_restricted_input_read_count": 0,
        "sealed_contact_count": 0,
        "gse246381_contact_count": 0,
    }:
        raise ObservationError("internal access attestation differs")


def _gate(gate_id: str, status: str, reason: str, counts: Mapping[str, int]) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "status": status,
        "reason": reason,
        "aggregate_counts": dict(counts),
    }


def build_gates(protocol: Mapping[str, Any], observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    _validate_observation(protocol, observation)
    lineage = observation["multi_asset_lineage"]
    replay = observation["edit_replay"]
    endpoint = observation["endpoint_transform"]
    eligible = observation["eligible_after_row_preflight_exclusions"]
    structural = observation["structural_dedup_and_split_feasibility"]
    reject = observation["reject_closure"]
    direction = observation["beneficial_direction_counts"]
    return [
        _gate(GATE_IDS[0], "PASS", "The authorized 634-pool strict universe replayed exactly before row-level fields were admitted.", {"review_pool_count": 634, "review_candidate_row_count": 7292}),
        _gate(GATE_IDS[1], "PASS", "All TSV rows replay to the frozen FASTA through the paper-authorized promoter and first-base transform.", {"tsv_rows": lineage["tsv_rows_reviewed"], "exact_lineage_rows": lineage["transformed_sequences_exactly_matched"], "mismatch_rows": lineage["transformed_sequence_mismatch_count"]}),
        _gate(GATE_IDS[2], "PASS", "Strict source groups each have exactly one parent and candidate identities are unique across TSV and FASTA.", {"source_groups": 634, "parent_rows": 634, "candidate_rows": 7292}),
        _gate(GATE_IDS[3], "PARTIAL_FAIL_CURRENT_PROTOCOL", "Three minus-CCC candidates remain unexplained after direct and separately reported publisher ATG-to-AGT-assisted replay.", {"direct": replay["direct_total"], "publisher_assisted": replay["publisher_assisted_total"], "unexplained": replay["unexplained_count"]}),
        _gate(GATE_IDS[4], "PASS_LIMITED_VCE_CONTEXT_ONLY", "All retained pools are separately stratified within VCE; CleanCap and in-vivo extrapolation remain prohibited.", {"retained_pools": eligible["pool_count"], "win_pools": eligible["pool_family_counts"]["win"], "plus_ccc_pools": eligible["pool_family_counts"]["+CCC"], "minus_ccc_pools": eligible["pool_family_counts"]["-CCC"]}),
        _gate(GATE_IDS[5], "PARTIAL_FAIL_CURRENT_PROTOCOL", "Direction is frozen as higher RRS but one row has an undefined endpoint.", {"finite_rows": endpoint["finite_endpoint_and_replicate_row_count"], "undefined_rows": endpoint["nonfinite_or_undefined_row_count"]}),
        _gate(GATE_IDS[6], "PASS_FOR_FINITE_ROWS_ONLY", "The published log2 RRS formula was mechanically reproduced for every finite row within the frozen tolerance.", {"formula_matches": endpoint["formula_match_within_tolerance_count"], "formula_mismatches": endpoint["formula_mismatch_count"], "undefined_rows": endpoint["nonfinite_or_undefined_row_count"]}),
        _gate(GATE_IDS[7], "UNKNOWN_NOT_ASSERTED", "The public files do not establish an independent biological source-group authority beyond the pooled VCE construct context.", {"structural_source_groups": structural["source_group_count"], "independent_biological_source_groups_established": 0}),
        _gate(GATE_IDS[8], "FAIL", "Four 80S and three IVT measurements are technical reactions from one sample context per arm; biological independence and row-level valid SE are absent.", {"monosome_technical_reactions": 4, "ivt_technical_reactions": 3, "independent_biological_replicates_established": 0, "valid_row_se_columns": 0}),
        _gate(GATE_IDS[9], "FAIL", "Private analysis is allowed, but raw and row-derived redistribution rights are not established by the Zenodo code license.", {"private_processing_rights_pass": 1, "raw_redistribution_rights_pass": 0, "derived_redistribution_rights_pass": 0}),
        _gate(GATE_IDS[10], "NOT_RUN_FORMAL", "Structural split feasibility was aggregated, but no outcome-blind source/group/near-duplicate split was executed or serialized.", {"source_groups": structural["source_group_count"], "genes": structural["gene_count"], "exact_utr_clusters": structural["exact_utr_cluster_count"], "cross_pool_duplicate_clusters": structural["cross_pool_exact_utr_cluster_count"]}),
        _gate(GATE_IDS[11], "INELIGIBLE_NOT_RUN", "Prefrozen power and CI-width analysis is ineligible until a valid biological analysis unit, SE, and formal split exist; no observed post-hoc power was computed.", {"valid_analysis_units": 0, "power_runs": 0, "bootstrap_runs": 0}),
        _gate(GATE_IDS[12], "PASS_PUBLIC_ORIGIN_ONLY", "The two input assets and method authorities are ordinary-public and frozen; this does not cure replicate, rights, or qualification gates.", {"public_assets": 2, "private_assets": 0, "sealed_assets": 0}),
        _gate(GATE_IDS[13], "CONDITIONAL_PENDING_ZERO_EXTERNAL_LEARNED_INPUT_RUNTIME_ATTESTATION", "The owner selected scratch-only/no-foundation exposure, but the runtime zero-external-learned-input attestation has not yet been produced.", {"scratch_route_selected": 1, "runtime_attestations": 0, "foundation_inputs_allowed": 0}),
        _gate(GATE_IDS[14], "UNKNOWN_NOT_ASSERTED", "Direction counts exist, but measurement-noise separation is impossible without valid biological replication and SE.", {"positive_candidates": direction["candidate_positive_count"], "negative_candidates": direction["candidate_negative_count"], "valid_noise_estimates": 0}),
        _gate(GATE_IDS[15], "FAIL", "Structural post-dedup counts are available, but independent biological effective N is not established.", {"distinct_within_pool_candidates": structural["distinct_within_pool_candidate_count"], "structural_source_groups": structural["source_group_count"], "independent_effective_n": 0}),
        _gate(GATE_IDS[16], "PASS_AGGREGATE_CLOSURE", "Every input row has one aggregate retained or exclusion reason; edit and endpoint failures do not overlap.", {"input_rows": reject["mutually_exclusive_row_reason_total"], "retained_rows": reject["retained_row_count"], "dual_parent_groups": reject["dual_parent_group_count"], "two_candidate_groups": reject["two_candidate_group_count"], "nonstrict_rows": reject["nonstrict_record_count"], "unexplained_edit_rows": reject["unexplained_edit_candidate_count"], "nonfinite_endpoint_rows": reject["nonfinite_endpoint_row_count"]}),
    ]


def build_report(
    protocol: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    binding: Mapping[str, str],
    asset_identity: Mapping[str, Any],
    recorded_at: str,
) -> dict[str, Any]:
    gates = build_gates(protocol, observation)
    if tuple(gate["gate_id"] for gate in gates) != GATE_IDS:
        raise ObservationError("gate order differs")
    all_pass = all(str(gate["status"]).startswith("PASS") for gate in gates)
    terminal = protocol["terminal_rule"]
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "contract_id": "mrna_xeditflow_route_a_v3",
        "phase_id": "A1",
        "dataset_id": DATASET_ID,
        "bioproject_id": "PRJNA1078388",
        "decision_id": DECISION_ID,
        "recorded_at": recorded_at,
        "status": STATUS_READY if all_pass else STATUS_STOP,
        "preflight_complete": True,
        "all_required_gates_pass": all_pass,
        "qualified": False,
        "implementation_binding": dict(binding),
        "ordinary_public_asset_identity": copy.deepcopy(dict(asset_identity)),
        "aggregate_observation": {
            key: copy.deepcopy(observation[key])
            for key in (
                "candidate_universe",
                "multi_asset_lineage",
                "edit_replay",
                "endpoint_transform",
                "eligible_after_row_preflight_exclusions",
                "beneficial_direction_counts",
                "structural_dedup_and_split_feasibility",
                "reject_closure",
            )
        },
        "required_gate_results": gates,
        "scope_attestation": {
            **copy.deepcopy(observation["internal_access_attestation"]),
            "aggregate_output_only": True,
            "member_identifier_output_count": 0,
            "member_role_output_count": 0,
            "member_context_output_count": 0,
            "sequence_output_count": 0,
            "row_effect_output_count": 0,
            "raw_replicate_value_output_count": 0,
            "replicate_identifier_output_count": 0,
            "split_assignment_output_count": 0,
            "canonical_record_output_count": 0,
        },
        "terminal_truth": copy.deepcopy(dict(protocol["frozen_outer_truth"])),
        "remaining_independent_blocker_classes": copy.deepcopy(
            terminal["remaining_independent_blocker_classes"]
        ),
        "sole_next_external_action": terminal["sole_next_external_action"],
        "interpretation_boundary": {
            "source_to_candidate_relation_presumed": False,
            "dataset_qualification_decided": False,
            "ordinary_a1_true_a2_credit_granted": False,
            "canonical_materialized": False,
            "training_or_model_selection_authorized": False,
            "all_gate_pass_would_automatically_qualify": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }


def _write_temp_payload(path: Path, payload: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_exclusive(output_dir: Path, report: Mapping[str, Any]) -> Path:
    temporary: Path | None = None
    output_created = False
    try:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise OutputError("output directory is not empty")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / REPORT_FILENAME
        if output_path.exists():
            raise OutputError("aggregate report already exists")
        descriptor, name = tempfile.mkstemp(
            prefix=f".{REPORT_FILENAME}.", suffix=".tmp", dir=output_dir
        )
        os.close(descriptor)
        temporary = Path(name)
        _write_temp_payload(temporary, _json_bytes(report))
        try:
            os.link(temporary, output_path)
            output_created = True
        except FileExistsError as exc:
            raise OutputError("aggregate report appeared during publish") from exc
        temporary.unlink()
        temporary = None
        directory_fd = os.open(output_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return output_path
    except OutputError:
        raise
    except OSError as exc:
        if output_created:
            try:
                (output_dir / REPORT_FILENAME).unlink()
            except OSError:
                pass
        raise OutputError("cannot publish aggregate-only output") from exc
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


BindingAuditor = Callable[
    [Mapping[str, Any], Path, bytes, Path], Mapping[str, str]
]
AssetAuditor = Callable[
    [Mapping[str, Any], Path, Path], Mapping[str, Any]
]
Aggregator = Callable[[Mapping[str, Any], Path, Path], Mapping[str, Any]]


def execute(
    protocol_path: Path,
    tsv_path: Path,
    fasta_path: Path,
    output_dir: Path,
    *,
    repo_root: Path | None = None,
    binding_auditor: BindingAuditor = _default_binding_auditor,
    asset_identity_auditor: AssetAuditor = _default_asset_identity_auditor,
    aggregator: Aggregator = aggregate_assets,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    protocol, payload = load_protocol(protocol_path)
    root = repo_root or protocol_path.parent.parent
    binding = binding_auditor(protocol, protocol_path, payload, root)
    asset_identity = asset_identity_auditor(protocol, tsv_path, fasta_path)
    observation = aggregator(protocol, tsv_path, fasta_path)
    report = build_report(
        protocol,
        observation,
        binding=binding,
        asset_identity=asset_identity,
        recorded_at=recorded_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )
    _write_exclusive(output_dir, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--tsv", required=True, type=Path)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = execute(
            args.protocol,
            args.tsv,
            args.fasta,
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
