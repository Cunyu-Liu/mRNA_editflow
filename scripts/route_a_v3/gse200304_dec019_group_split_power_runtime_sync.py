#!/usr/bin/env python3
"""Prepare, validate, and publish the A1-EVT-044 runtime sync.

EVT044 is a runtime bookkeeping transition.  The fourteen already-published
artifacts in the config are registered in the successor manifest by metadata;
they are never copied or opened.  The only new immutable outputs are the
three predecessor snapshots and the one sync record.  Mutable runtime files
are committed in the order STATUS, RUN_MANIFEST, EVENT_LOG, with a recoverable
prefix so an interrupted operator retry cannot duplicate the event.

The checked-in I2 config closes the already-pushed I1 authority while leaving
the current implementation binding UNKNOWN_NOT_ASSERTED.  Static validation is
therefore useful without touching a repository, source bundle, or runtime.
Every prepare/publish/validate path requires a fully-bound B2 config and proves
the exact L -> I1 -> I2 -> B2 history before any runtime or registered-artifact
operation.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping


UNKNOWN = "UNKNOWN_NOT_ASSERTED"
CORE_PLACEHOLDER = "CORE_SHA256_TO_REFRESH"
MUTABLE_NAMES = ("STATUS.json", "RUN_MANIFEST.json", "EVENT_LOG.jsonl")
SCRIPT_REPO_PATH = "scripts/route_a_v3/gse200304_dec019_group_split_power_runtime_sync.py"
TEST_REPO_PATH = "tests/route_a_v3/test_gse200304_dec019_group_split_power_runtime_sync.py"
CONFIG_REPO_PATH = "configs/route_a_v3_gse200304_dec019_group_split_power_runtime_sync_v1.json"
PRODUCTION_CONFIG_PATH = Path(__file__).resolve().parents[2] / CONFIG_REPO_PATH
PRODUCTION_REPO_ROOT = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/routea_v3_a1_20260810"
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

FaultInjector = Callable[[str], None]


class RuntimeSyncError(RuntimeError):
    """Base class for an EVT044 validation/publication error."""


class BindingError(RuntimeSyncError):
    """The implementation or predecessor ledger is not fully bound."""


class AuthorityError(RuntimeSyncError):
    """The production Git history does not prove the bound publisher lineage."""


class PredecessorError(RuntimeSyncError):
    """The runtime is not the exact EVT043 predecessor or a supported prefix."""


class PublicationError(RuntimeSyncError):
    """Prepared or immutable/mutable publication failed."""


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def compact_json_line(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeSyncError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeSyncError(f"invalid JSON: {label}") from exc
    if not isinstance(value, dict):
        raise RuntimeSyncError(f"JSON root is not an object: {label}")
    return value


def load_events(payload: bytes, *, label: str) -> list[dict[str, Any]]:
    if payload and not payload.endswith(b"\n"):
        raise RuntimeSyncError(f"JSONL is not newline terminated: {label}")
    try:
        values = [
            json.loads(line, object_pairs_hook=_reject_duplicate_keys)
            for line in payload.splitlines()
        ]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeSyncError(f"invalid JSONL: {label}") from exc
    if any(not isinstance(value, dict) for value in values):
        raise RuntimeSyncError(f"JSONL contains a non-object: {label}")
    return values


def _expect(actual: Any, expected: Any, *, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise RuntimeSyncError(f"{label} drift: expected {expected!r}, observed {actual!r}")


def _expect_keys(value: Any, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise RuntimeSyncError(f"{label} key closure drift")
    return value


def _expect_hex(value: Any, pattern: re.Pattern[str], *, label: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise RuntimeSyncError(f"{label} is not lowercase hexadecimal")
    return value


def _typed_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _typed_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _typed_equal(item, value) for item, value in zip(actual, expected)
        )
    return actual == expected


def compiled_core_projection(config: dict[str, Any]) -> dict[str, Any]:
    """Return the config core that must remain identical between I and B."""

    return {
        key: copy.deepcopy(value)
        for key, value in config.items()
        if key != "implementation_binding"
    }


def compiled_core_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(
        compiled_core_projection(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload)


def _binding_values_are_unknown(binding: Mapping[str, Any]) -> bool:
    return all(
        binding.get(key) == UNKNOWN
        for key in (
            "status",
            "implementation_commit",
            "implementation_script_sha256",
            "implementation_test_sha256",
        )
    )


def _ledger_values(authority: Mapping[str, Any]) -> list[Any]:
    ledger = authority.get("predecessor_ledger")
    if not isinstance(ledger, Mapping):
        return []
    blobs = ledger.get("frozen_blobs")
    if not isinstance(blobs, list):
        return []
    return [ledger.get("status"), ledger.get("commit")] + [
        item.get("sha256") if isinstance(item, Mapping) else None for item in blobs
    ]


def _ledger_values_are_unknown(authority: Mapping[str, Any]) -> bool:
    values = _ledger_values(authority)
    return len(values) == 6 and all(value == UNKNOWN for value in values)


def _validate_runtime_shape(config: dict[str, Any]) -> None:
    runtime = _expect_keys(
        config.get("runtime"),
        {
            "run_root",
            "allowed_prepared_root",
            "predecessor_event_id",
            "predecessor_event_count",
            "successor_event_id",
            "successor_event_count",
            "predecessor_manifest_output_count",
            "successor_manifest_output_count",
            "predecessor_mutables",
            "predecessor_tail",
            "sync_name",
            "output_delta_count",
            "immutable_publish_order",
            "mutable_publish_order",
        },
        label="runtime",
    )
    _expect(runtime["predecessor_event_id"], "A1-EVT-043", label="predecessor event id")
    _expect(runtime["successor_event_id"], "A1-EVT-044", label="successor event id")
    for key, expected in (
        ("predecessor_event_count", 43),
        ("successor_event_count", 44),
        ("predecessor_manifest_output_count", 163),
        ("successor_manifest_output_count", 181),
        ("output_delta_count", 18),
    ):
        if type(runtime.get(key)) is not int or runtime[key] != expected:
            raise RuntimeSyncError(f"runtime {key} drift")
    _expect(runtime["mutable_publish_order"], list(MUTABLE_NAMES), label="mutable publish order")
    snapshots = runtime.get("predecessor_mutables")
    if not isinstance(snapshots, dict) or set(snapshots) != set(MUTABLE_NAMES):
        raise RuntimeSyncError("predecessor mutable metadata closure drift")
    for mutable in MUTABLE_NAMES:
        spec = snapshots[mutable]
        _expect_keys(spec, {"bytes", "sha256", "snapshot_name"}, label=f"{mutable} metadata")
        if (
            isinstance(spec["bytes"], bool)
            or not isinstance(spec["bytes"], int)
            or spec["bytes"] < 0
        ):
            raise RuntimeSyncError(f"{mutable} byte metadata drift")
        _expect_hex(spec["sha256"], HEX64, label=f"{mutable} predecessor SHA")
        if not isinstance(spec["snapshot_name"], str) or not spec["snapshot_name"]:
            raise RuntimeSyncError(f"{mutable} snapshot name drift")
    if not isinstance(runtime["predecessor_tail"], dict):
        raise RuntimeSyncError("predecessor tail metadata is absent")
    _expect(runtime["sync_name"], "A1_GSE200304_DEC019_GROUP_SPLIT_POWER_PASS_AND_D6_RUNTIME_SYNC_V1.json", label="sync name")
    immutable_expected = [
        snapshots[name]["snapshot_name"] for name in MUTABLE_NAMES
    ] + [runtime["sync_name"]]
    _expect(runtime["immutable_publish_order"], immutable_expected, label="immutable publish order")


def validate_static_config(config: dict[str, Any]) -> None:
    """Validate the config core without opening any repository/source/runtime path."""

    _expect_keys(
        config,
        {
            "schema_version",
            "protocol_id",
            "contract_id",
            "phase_id",
            "dataset_id",
            "decision_id",
            "event_id",
            "event_name",
            "implementation_binding",
            "repository_authority",
            "registered_artifacts",
            "runtime",
            "successor_scientific_state",
            "access_boundary",
            "publication_policy",
        },
        label="config root",
    )
    _expect(
        {
            key: config.get(key)
            for key in (
                "schema_version",
                "protocol_id",
                "contract_id",
                "phase_id",
                "dataset_id",
                "decision_id",
                "event_id",
                "event_name",
            )
        },
        {
            "schema_version": "route_a_v3_gse200304_dec019_group_split_power_runtime_sync.v1",
            "protocol_id": "ROUTE_A_V3_GSE200304_DEC019_GROUP_SPLIT_POWER_PASS_AND_D6_RUNTIME_SYNC_V1",
            "contract_id": "mrna_xeditflow_route_a_v3",
            "phase_id": "A1",
            "dataset_id": "GSE200304",
            "decision_id": "V3-DEC-019",
            "event_id": "A1-EVT-044",
            "event_name": "GSE200304_DEC019_GROUP_SPLIT_POWER_PASS_AND_D6_ONE_BLOCKER_ADJUDICATION_SYNCED_QUALIFICATION_GATE_UNCHANGED",
        },
        label="config identity",
    )

    binding = _expect_keys(
        config["implementation_binding"],
        {
            "status",
            "implementation_commit",
            "implementation_script_path",
            "implementation_script_sha256",
            "implementation_test_path",
            "implementation_test_sha256",
            "compiled_core_sha256",
        },
        label="implementation binding",
    )
    _expect(binding["implementation_script_path"], SCRIPT_REPO_PATH, label="script path")
    _expect(binding["implementation_test_path"], TEST_REPO_PATH, label="test path")
    values = [
        binding[key]
        for key in (
            "status",
            "implementation_commit",
            "implementation_script_sha256",
            "implementation_test_sha256",
        )
    ]
    if any(value == UNKNOWN for value in values):
        if not _binding_values_are_unknown(binding):
            raise BindingError("implementation binding is partially known")
    else:
        _expect(binding["status"], "BOUND", label="implementation status")
        _expect_hex(binding["implementation_commit"], HEX40, label="implementation commit")
        _expect_hex(binding["implementation_script_sha256"], HEX64, label="implementation script SHA")
        _expect_hex(binding["implementation_test_sha256"], HEX64, label="implementation test SHA")
    core = binding["compiled_core_sha256"]
    if core != CORE_PLACEHOLDER:
        _expect_hex(core, HEX64, label="compiled core SHA")
        _expect(core, compiled_core_sha256(config), label="compiled core")

    authority = _expect_keys(
        config["repository_authority"],
        {
            "production_repo_root",
            "branch",
            "implementation_exact_changed_paths",
            "binding_exact_changed_paths",
            "predecessor_ledger",
            "historical_implementation_i1",
            "current_implementation_i2",
        },
        label="repository authority",
    )
    _expect(
        authority["production_repo_root"],
        str(PRODUCTION_REPO_ROOT),
        label="production repository root",
    )
    _expect(authority["branch"], "routea-v3-a1-20260810", label="repository branch")
    _expect(
        authority["implementation_exact_changed_paths"],
        [CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH],
        label="implementation exact3 paths",
    )
    _expect(authority["binding_exact_changed_paths"], [CONFIG_REPO_PATH], label="binding path")
    ledger = _expect_keys(
        authority["predecessor_ledger"],
        {"status", "commit", "integration_id", "frozen_blobs"},
        label="predecessor ledger",
    )
    _expect(
        ledger["integration_id"],
        "GSE200304_DEC019_GROUP_SPLIT_POWER_PASS_AND_D6_ONE_BLOCKER_LEDGER_V1",
        label="ledger integration id",
    )
    blobs = ledger["frozen_blobs"]
    if not isinstance(blobs, list) or len(blobs) != 4:
        raise RuntimeSyncError("predecessor ledger must contain exactly four frozen blobs")
    expected_paths = [
        "docs/execution/route_a_v3_a1_interim.yaml",
        "docs/execution/route_a_v3_registry_manifest.json",
        "scripts/route_a_v3/validate_a0_bundle.py",
        "tests/route_a_v3/test_a0_integrity_guards.py",
    ]
    _expect([item.get("path") for item in blobs], expected_paths, label="ledger blob paths")
    for index, item in enumerate(blobs):
        _expect_keys(item, {"path", "sha256"}, label=f"ledger blob {index}")
        digest = item["sha256"]
        if digest != UNKNOWN:
            _expect_hex(digest, HEX64, label=f"ledger blob {index} SHA")
    ledger_values = _ledger_values(authority)
    if any(value == UNKNOWN for value in ledger_values):
        if not _ledger_values_are_unknown(authority):
            raise BindingError("predecessor ledger binding is partially known")
    else:
        _expect(ledger["status"], "BOUND", label="ledger status")
        _expect_hex(ledger["commit"], HEX40, label="ledger commit")

    historical_i1 = _expect_keys(
        authority["historical_implementation_i1"],
        {"status", "commit", "parent_commit", "exact_changed_paths", "frozen_blobs"},
        label="historical I1 authority",
    )
    _expect(historical_i1["status"], "BOUND", label="historical I1 status")
    _expect_hex(historical_i1["commit"], HEX40, label="historical I1 commit")
    _expect_hex(historical_i1["parent_commit"], HEX40, label="historical I1 parent")
    _expect(historical_i1["parent_commit"], ledger["commit"], label="historical I1 parent/L")
    _expect(
        historical_i1["exact_changed_paths"],
        [CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH],
        label="historical I1 exact3 paths",
    )
    historical_blobs = historical_i1["frozen_blobs"]
    if not isinstance(historical_blobs, list) or len(historical_blobs) != 3:
        raise RuntimeSyncError("historical I1 must contain exactly three frozen blobs")
    _expect(
        [item.get("path") for item in historical_blobs],
        [CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH],
        label="historical I1 frozen blob paths",
    )
    for index, item in enumerate(historical_blobs):
        _expect_keys(item, {"path", "sha256"}, label=f"historical I1 blob {index}")
        _expect_hex(item["sha256"], HEX64, label=f"historical I1 blob {index} SHA")

    current_i2 = _expect_keys(
        authority["current_implementation_i2"],
        {"expected_parent_commit"},
        label="current I2 authority",
    )
    _expect_hex(current_i2["expected_parent_commit"], HEX40, label="current I2 parent")
    _expect(
        current_i2["expected_parent_commit"],
        historical_i1["commit"],
        label="current I2 expected parent/I1",
    )

    registered = config["registered_artifacts"]
    if not isinstance(registered, list) or len(registered) != 14:
        raise RuntimeSyncError("registered artifact count is not exact14")
    seen: set[str] = set()
    for index, item in enumerate(registered):
        _expect_keys(item, {"absolute_path", "artifact_type", "bytes", "sha256"}, label=f"registered artifact {index}")
        path = item["absolute_path"]
        if not isinstance(path, str) or not path.startswith("/") or path in seen:
            raise RuntimeSyncError(f"registered artifact path drift: {index}")
        seen.add(path)
        if not isinstance(item["artifact_type"], str) or not item["artifact_type"]:
            raise RuntimeSyncError(f"registered artifact type drift: {index}")
        if isinstance(item["bytes"], bool) or not isinstance(item["bytes"], int) or item["bytes"] < 0:
            raise RuntimeSyncError(f"registered artifact byte metadata drift: {index}")
        _expect_hex(item["sha256"], HEX64, label=f"registered artifact {index} SHA")

    _validate_runtime_shape(config)

    scientific = config["successor_scientific_state"]
    _expect(
        scientific,
        {
            "input_status_counts": {"PASS": 7, "BLOCKED": 0, "UNKNOWN_NOT_ASSERTED": 1, "NOT_RUN": 0},
            "unresolved_blockers": ["CHECKPOINT_SPECIFIC_EXPOSURE_NOT_PASS"],
            "ordinary_study_contribution": 0,
            "a1_study_contribution": 0,
            "true_a2_study_contribution": 0,
            "canonical_intervention_record_count": 0,
            "canonical_materialization_allowed": False,
            "qualified": False,
            "training_started": False,
            "training_allowed": False,
            "training_authorized": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
            "scientific_claim_status": "NOT_ESTABLISHED",
        },
        label="successor scientific truth",
    )
    _expect(
        config["access_boundary"],
        {
            "registered_artifact_count": 14,
            "registered_artifact_metadata_validation_count": 14,
            "registered_artifact_body_parse_count": 0,
            "private_mapping_or_assignment_content_inspected": False,
            "canonical_materialization_count": 0,
            "training_run_count": 0,
            "model_selection_run_count": 0,
            "restricted_or_sealed_path_accessed": False,
            "gse246381_contact": False,
        },
        label="access boundary",
    )
    _expect(
        config["publication_policy"],
        {
            "registered_artifacts_remain_in_place": True,
            "predecessor_snapshots_are_immutable_runtime_outputs": True,
            "sync_record_is_immutable_runtime_output": True,
            "mutables_commit_after_all_immutables": True,
            "event_is_last_commit": True,
            "supported_recovery": "EXACT_PUBLICATION_PREFIX_ONLY",
        },
        label="publication policy",
    )


def validate_bound_config(config: dict[str, Any]) -> None:
    """Require implementation and all five ledger scalars before any runtime I/O."""

    validate_static_config(config)
    authority = config["repository_authority"]
    # Check the ledger first: the checked-in contract deliberately has this
    # grouped UNKNOWN, and no source/output operation may precede this error.
    if _ledger_values_are_unknown(authority) or any(
        value == UNKNOWN for value in _ledger_values(authority)
    ):
        raise BindingError("predecessor ledger authority is not fully BOUND")
    binding = config["implementation_binding"]
    if _binding_values_are_unknown(binding) or any(
        binding.get(key) == UNKNOWN
        for key in (
            "status",
            "implementation_commit",
            "implementation_script_sha256",
            "implementation_test_sha256",
        )
    ):
        raise BindingError("runtime-sync implementation is not BOUND")


def _load_config_payload(
    config_path: Path, *, require_bound: bool
) -> tuple[dict[str, Any], bytes]:
    payload = Path(config_path).read_bytes()
    config = load_json(payload, label=str(config_path))
    if require_bound:
        validate_bound_config(config)
    else:
        validate_static_config(config)
    return config, payload


def load_config(
    config_path: Path = PRODUCTION_CONFIG_PATH, *, require_bound: bool = True
) -> dict[str, Any]:
    return _load_config_payload(config_path, require_bound=require_bound)[0]


def load_bound_config(config_path: Path = PRODUCTION_CONFIG_PATH) -> dict[str, Any]:
    return load_config(config_path, require_bound=True)


def expected_unknown_i2_config(bound_config: dict[str, Any]) -> dict[str, Any]:
    expected = copy.deepcopy(bound_config)
    for key in (
        "status",
        "implementation_commit",
        "implementation_script_sha256",
        "implementation_test_sha256",
    ):
        expected["implementation_binding"][key] = UNKNOWN
    return expected


def _run_git(repo_root: Path, *arguments: str) -> bytes:
    environment = os.environ.copy()
    environment.update({"GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C", "LANG": "C"})
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AuthorityError("read-only Git command failed to start") from exc
    if result.returncode != 0:
        raise AuthorityError(f"read-only Git command failed: {arguments!r}")
    return result.stdout


def _git_blob(repo_root: Path, commit: str, path: str) -> bytes:
    return _run_git(repo_root, "show", f"{commit}:{path}")


def _changed_paths(repo_root: Path, commit: str) -> list[str]:
    return sorted(
        _run_git(
            repo_root,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            commit,
        )
        .decode("utf-8")
        .splitlines()
    )


def _read_repo_file(repo_root: Path, path: str) -> bytes:
    try:
        return (repo_root / path).read_bytes()
    except OSError as exc:
        raise AuthorityError(f"cannot read production repository file: {path}") from exc


def audit_production_repository_authority(
    config: dict[str, Any], config_payload: bytes
) -> dict[str, Any]:
    """Prove the exact L -> I1 -> I2 -> config-only B2 chain."""

    validate_bound_config(config)
    authority = config["repository_authority"]
    binding = config["implementation_binding"]
    ledger = authority["predecessor_ledger"]
    historical_i1 = authority["historical_implementation_i1"]
    current_i2 = authority["current_implementation_i2"]
    repo_root = Path(authority["production_repo_root"])
    branch = authority["branch"]
    implementation_i2_commit = binding["implementation_commit"]
    implementation_i1_commit = historical_i1["commit"]
    ledger_commit = ledger["commit"]

    head = _run_git(repo_root, "rev-parse", "HEAD").decode("utf-8").strip()
    current_branch = _run_git(
        repo_root, "rev-parse", "--abbrev-ref", "HEAD"
    ).decode("utf-8").strip()
    upstream_branch = _run_git(
        repo_root, "rev-parse", "--abbrev-ref", "@{upstream}"
    ).decode("utf-8").strip()
    upstream_head = _run_git(repo_root, "rev-parse", "@{upstream}").decode(
        "utf-8"
    ).strip()
    _expect(current_branch, branch, label="production branch")
    _expect(upstream_branch, f"origin/{branch}", label="production upstream branch")
    _expect(head, upstream_head, label="production HEAD/upstream")
    if _run_git(
        repo_root, "status", "--porcelain=v1", "--untracked-files=all"
    ) != b"":
        raise AuthorityError("production worktree or index is dirty")

    binding_parent = _run_git(repo_root, "rev-parse", f"{head}^").decode(
        "utf-8"
    ).strip()
    implementation_i2_parent = _run_git(
        repo_root, "rev-parse", f"{implementation_i2_commit}^"
    ).decode("utf-8").strip()
    implementation_i1_parent = _run_git(
        repo_root, "rev-parse", f"{implementation_i1_commit}^"
    ).decode("utf-8").strip()
    _expect(binding_parent, implementation_i2_commit, label="B2 parent/I2")
    _expect(
        implementation_i2_parent,
        current_i2["expected_parent_commit"],
        label="I2 parent/I1",
    )
    _expect(
        implementation_i1_parent,
        historical_i1["parent_commit"],
        label="I1 parent/L",
    )
    _expect(
        _changed_paths(repo_root, head),
        sorted(authority["binding_exact_changed_paths"]),
        label="B exact config-only paths",
    )
    _expect(
        _changed_paths(repo_root, implementation_i2_commit),
        sorted(authority["implementation_exact_changed_paths"]),
        label="I2 exact config/script/test paths",
    )
    _expect(
        _changed_paths(repo_root, implementation_i1_commit),
        sorted(historical_i1["exact_changed_paths"]),
        label="I1 exact config/script/test paths",
    )
    ledger_paths = [item["path"] for item in ledger["frozen_blobs"]]
    _expect(
        _changed_paths(repo_root, ledger_commit),
        sorted(ledger_paths),
        label="L exact frozen ledger paths",
    )

    for item in ledger["frozen_blobs"]:
        if sha256(_git_blob(repo_root, ledger_commit, item["path"])) != item["sha256"]:
            raise AuthorityError(f"L frozen ledger blob drift: {item['path']}")

    for item in historical_i1["frozen_blobs"]:
        if sha256(_git_blob(repo_root, implementation_i1_commit, item["path"])) != item["sha256"]:
            raise AuthorityError(f"I1 frozen implementation blob drift: {item['path']}")

    i2_config = load_json(
        _git_blob(repo_root, implementation_i2_commit, CONFIG_REPO_PATH),
        label="implementation I2 config",
    )
    if not _typed_equal(i2_config, expected_unknown_i2_config(config)):
        raise AuthorityError("I2 config is not the exact four-scalar UNKNOWN form")

    if _git_blob(repo_root, head, CONFIG_REPO_PATH) != config_payload:
        raise AuthorityError("B config Git blob differs from the supplied production config")
    if _read_repo_file(repo_root, CONFIG_REPO_PATH) != config_payload:
        raise AuthorityError("worktree config differs from the supplied production config")

    for path, digest in (
        (SCRIPT_REPO_PATH, binding["implementation_script_sha256"]),
        (TEST_REPO_PATH, binding["implementation_test_sha256"]),
    ):
        for commit, label in (
            (implementation_i2_commit, "I2"),
            (head, "B2"),
        ):
            if sha256(_git_blob(repo_root, commit, path)) != digest:
                raise AuthorityError(f"{label} implementation blob drift: {path}")
        if sha256(_read_repo_file(repo_root, path)) != digest:
            raise AuthorityError(f"worktree implementation blob drift: {path}")

    core = binding["compiled_core_sha256"]
    if core != CORE_PLACEHOLDER:
        _expect(core, compiled_core_sha256(config), label="production compiled core")

    return {
        "status": "PASS_EXACT_L_TO_I1_TO_I2_TO_CONFIG_ONLY_B2",
        "ledger_commit": ledger_commit,
        "implementation_i1_commit": implementation_i1_commit,
        "implementation_i2_commit": implementation_i2_commit,
        "binding_b2_commit": head,
    }


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _prepared_path(prepared_directory: Path | str, config: dict[str, Any]) -> Path:
    prepared = _absolute(prepared_directory)
    allowed = _absolute(config["runtime"]["allowed_prepared_root"])
    try:
        common = Path(os.path.commonpath((str(prepared), str(allowed))))
    except ValueError as exc:
        raise PublicationError("prepared directory is outside allowed root") from exc
    if common != allowed or prepared == allowed:
        raise PublicationError("prepared directory must be a strict descendant of allowed root")
    return prepared


@contextmanager
def _locked_run(run_root: Path) -> Iterator[None]:
    try:
        descriptor = os.open(run_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError as exc:
        raise PublicationError(f"cannot open runtime root: {run_root}") from exc
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _read_runtime(run_root: Path) -> dict[str, bytes]:
    try:
        return {name: (run_root / name).read_bytes() for name in MUTABLE_NAMES}
    except OSError as exc:
        raise PublicationError("cannot read runtime mutables") from exc


def _predecessor_snapshot_names(config: dict[str, Any]) -> dict[str, str]:
    return {
        name: config["runtime"]["predecessor_mutables"][name]["snapshot_name"]
        for name in MUTABLE_NAMES
    }


def validate_registered_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the exact14 metadata only; never open an artifact body."""

    validate_static_config(config)
    artifacts = config["registered_artifacts"]
    return {
        "status": "EXACT14_METADATA_VALIDATED",
        "artifact_count": len(artifacts),
        "body_parse_count": 0,
        "registered_artifacts": copy.deepcopy(artifacts),
    }


def _validate_recorded_at(recorded_at: str, predecessor_at: Any) -> None:
    if not isinstance(recorded_at, str) or not isinstance(predecessor_at, str):
        raise PredecessorError("timestamps must be explicit ISO-8601 strings")
    try:
        current = datetime.fromisoformat(recorded_at)
        predecessor = datetime.fromisoformat(predecessor_at)
    except ValueError as exc:
        raise PredecessorError("timestamp is not ISO-8601") from exc
    if current.tzinfo is None or predecessor.tzinfo is None or current <= predecessor:
        raise PredecessorError("EVT044 timestamp must follow EVT043 with an explicit offset")


def _parse_runtime(payloads: Mapping[str, bytes]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    return (
        load_json(payloads["STATUS.json"], label="STATUS.json"),
        load_json(payloads["RUN_MANIFEST.json"], label="RUN_MANIFEST.json"),
        load_events(payloads["EVENT_LOG.jsonl"], label="EVENT_LOG.jsonl"),
    )


def _check_payload_identity(payload: bytes, spec: Mapping[str, Any], *, label: str) -> None:
    if len(payload) != spec["bytes"] or sha256(payload) != spec["sha256"]:
        raise PredecessorError(f"{label} predecessor identity drift")


def validate_predecessor(
    config: dict[str, Any], payloads: Mapping[str, bytes]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Validate EVT043 before creating any prepared member."""

    runtime = config["runtime"]
    for name in MUTABLE_NAMES:
        _check_payload_identity(
            payloads[name], runtime["predecessor_mutables"][name], label=name
        )
    status, manifest, events = _parse_runtime(payloads)
    if len(events) != runtime["predecessor_event_count"]:
        raise PredecessorError("predecessor event count is not 43")
    tail = events[-1] if events else {}
    _expect(tail.get("event_id"), "A1-EVT-043", label="predecessor tail event")
    _expect(tail.get("decision_id"), "V3-DEC-019", label="predecessor decision")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != runtime["predecessor_manifest_output_count"]:
        raise PredecessorError("predecessor manifest output count is not 163")
    return status, manifest, events


def _scientific_fields(target: dict[str, Any]) -> dict[str, Any]:
    # Keep the scientific transition explicit.  Existing non-scientific keys
    # in the predecessor are copied unchanged by build_successors.
    return {
        "input_status_counts": copy.deepcopy(target["input_status_counts"]),
        "unresolved_blockers": copy.deepcopy(target["unresolved_blockers"]),
        "ordinary_study_contribution": target["ordinary_study_contribution"],
        "a1_study_contribution": target["a1_study_contribution"],
        "true_a2_study_contribution": target["true_a2_study_contribution"],
        "canonical_intervention_record_count": target["canonical_intervention_record_count"],
        "canonical_materialization_allowed": target["canonical_materialization_allowed"],
        "qualified": target["qualified"],
        "training_started": target["training_started"],
        "training_allowed": target["training_allowed"],
        "training_authorized": target["training_authorized"],
        "model_selection_allowed": target["model_selection_allowed"],
        "next_phase_authorized": target["next_phase_authorized"],
        "scientific_claim_status": target["scientific_claim_status"],
    }


def _apply_scientific_state(document: dict[str, Any], scientific: dict[str, Any]) -> None:
    document.update(copy.deepcopy(scientific))
    # These aliases are present in older runtime manifests and make the
    # unchanged gate truth explicit without altering unrelated fields.
    document["claim_status"] = scientific["scientific_claim_status"]
    document["canonical_record_count"] = scientific["canonical_intervention_record_count"]


def _event_document(
    config: dict[str, Any],
    *,
    recorded_at: str,
    predecessor: Mapping[str, Any],
    sync_digest: str,
) -> dict[str, Any]:
    runtime = config["runtime"]
    scientific = config["successor_scientific_state"]
    return {
        "event_id": "A1-EVT-044",
        "at": recorded_at,
        "phase_id": "A1",
        "event": config["event_name"],
        "decision_id": "V3-DEC-019",
        "predecessor_event_id": "A1-EVT-043",
        "registered_artifact_count": 14,
        "registered_artifacts_copied": False,
        "predecessor_snapshot_count": 3,
        "predecessor_snapshot_names": list(_predecessor_snapshot_names(config).values()),
        "sync_name": runtime["sync_name"],
        "output_delta_count": 18,
        "manifest_output_count_before": 163,
        "manifest_output_count_after": 181,
        "new_runtime_output_count": 18,
        "scientific_state_changed": True,
        "evidence_gate_statuses_changed_since_evt043": True,
        "overall_qualification_gate_changed": False,
        "successor_scientific_state": copy.deepcopy(scientific),
        "ledger_authority_status": config["repository_authority"]["predecessor_ledger"]["status"],
        "sync_record_sha256": sync_digest,
        "predecessor_event_count": len(predecessor.get("events", [])) if isinstance(predecessor, Mapping) else 43,
        "training_started": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "detail": (
            "Registered the exact14 existing aggregate artifacts in place, added three "
            "immutable EVT043 snapshots and one sync record. The evidence state changed "
            "from four blockers to the single CHECKPOINT_SPECIFIC_EXPOSURE_NOT_PASS "
            "blocker while qualification and training/model-selection authorization "
            "remained unchanged."
        ),
    }


def _build_sync_record(
    config: dict[str, Any], *, recorded_at: str, snapshot_payloads: Mapping[str, bytes]
) -> bytes:
    runtime = config["runtime"]
    return json_bytes(
        {
            "record_type": "ROUTE_A_V3_A1_GSE200304_DEC019_GROUP_SPLIT_POWER_RUNTIME_SYNC",
            "event_id": "A1-EVT-044",
            "decision_id": "V3-DEC-019",
            "recorded_at": recorded_at,
            "predecessor_event_id": "A1-EVT-043",
            "registered_artifact_count": 14,
            "registered_artifacts_copied": False,
            "predecessor_snapshot_count": 3,
            "predecessor_snapshot_names": list(_predecessor_snapshot_names(config).values()),
            "snapshot_sha256": {
                name: sha256(payload) for name, payload in snapshot_payloads.items()
            },
            "output_delta_count": 18,
            "successor_manifest_output_count": 181,
            "successor_scientific_state": copy.deepcopy(config["successor_scientific_state"]),
            "ledger_authority_status": config["repository_authority"]["predecessor_ledger"]["status"],
            "implementation_binding_status": config["implementation_binding"]["status"],
            "scientific_state_changed": True,
            "evidence_gate_statuses_changed_since_evt043": True,
            "overall_qualification_gate_changed": False,
        }
    )


def _immutable_outputs(
    config: dict[str, Any],
    predecessor_outputs: list[dict[str, Any]],
    predecessor_payloads: Mapping[str, bytes],
    sync_payload: bytes,
) -> list[dict[str, Any]]:
    run_root = Path(config["runtime"]["run_root"])
    # EVT043 already has 163 outputs.  EVT044 appends the exact14 registered
    # aggregate records, then the three snapshots and sync record, for 181.
    outputs = copy.deepcopy(predecessor_outputs)
    outputs.extend(copy.deepcopy(config["registered_artifacts"]))
    snapshots = _predecessor_snapshot_names(config)
    for mutable in MUTABLE_NAMES:
        payload = predecessor_payloads[mutable]
        outputs.append(
            {
                "absolute_path": str(run_root / snapshots[mutable]),
                "artifact_type": "PREDECESSOR_RUNTIME_SNAPSHOT",
                "bytes": len(payload),
                "sha256": sha256(payload),
                "status": "COMPLETE",
            }
        )
    outputs.append(
        {
            "absolute_path": str(run_root / config["runtime"]["sync_name"]),
            "artifact_type": "RUNTIME_SYNC_RECORD",
            "bytes": len(sync_payload),
            "sha256": sha256(sync_payload),
            "status": "COMPLETE",
        }
    )
    return outputs


def build_successors(
    config: dict[str, Any], predecessor_payloads: Mapping[str, bytes], recorded_at: str
) -> dict[str, bytes]:
    status, manifest, events = validate_predecessor(config, predecessor_payloads)
    _validate_recorded_at(recorded_at, events[-1].get("at"))
    snapshots = _predecessor_snapshot_names(config)
    snapshot_payloads = {snapshots[name]: predecessor_payloads[name] for name in MUTABLE_NAMES}
    sync_payload = _build_sync_record(
        config, recorded_at=recorded_at, snapshot_payloads=snapshot_payloads
    )
    successor_status = copy.deepcopy(status)
    successor_manifest = copy.deepcopy(manifest)
    scientific = _scientific_fields(config["successor_scientific_state"])
    _apply_scientific_state(successor_status, scientific)
    _apply_scientific_state(successor_manifest, scientific)
    successor_status["updated_at"] = recorded_at
    successor_manifest["outputs"] = _immutable_outputs(
        config, manifest["outputs"], predecessor_payloads, sync_payload
    )
    # Keep an explicit registration count for consumers that do not inspect the
    # output list itself.
    successor_manifest["registered_artifact_count"] = 14
    event_without_digest = _event_document(
        config,
        recorded_at=recorded_at,
        predecessor={"events": events},
        sync_digest=sha256(sync_payload),
    )
    successor_event_payload = predecessor_payloads["EVENT_LOG.jsonl"] + compact_json_line(
        event_without_digest
    )
    artifacts = {
        **{snapshots[name]: predecessor_payloads[name] for name in MUTABLE_NAMES},
        config["runtime"]["sync_name"]: sync_payload,
        "STATUS.json": json_bytes(successor_status),
        "RUN_MANIFEST.json": json_bytes(successor_manifest),
        "EVENT_LOG.jsonl": successor_event_payload,
    }
    validate_successors(config, predecessor_payloads, artifacts)
    return artifacts


def validate_successors(
    config: dict[str, Any],
    predecessor_payloads: Mapping[str, bytes],
    successors: Mapping[str, bytes],
) -> None:
    """Check exact3 mutable prefix, exact14+4 output closure, and EVT044 last."""

    old_status, old_manifest, old_events = validate_predecessor(config, predecessor_payloads)
    status, manifest, events = _parse_runtime(
        {name: successors[name] for name in MUTABLE_NAMES}
    )
    runtime = config["runtime"]
    if len(events) != runtime["successor_event_count"] or events[:-1] != old_events:
        raise RuntimeSyncError("EVT044 is not one append-only event")
    event = events[-1]
    _expect(event.get("event_id"), "A1-EVT-044", label="successor event")
    _expect(event.get("decision_id"), "V3-DEC-019", label="successor event decision")
    _expect(event.get("scientific_state_changed"), True, label="event scientific state change")
    _expect(
        event.get("evidence_gate_statuses_changed_since_evt043"),
        True,
        label="event evidence gate change",
    )
    _expect(
        event.get("overall_qualification_gate_changed"),
        False,
        label="event overall qualification gate change",
    )
    if not isinstance(event.get("at"), str):
        raise RuntimeSyncError("EVT044 timestamp is absent")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != runtime["successor_manifest_output_count"]:
        raise RuntimeSyncError("successor manifest output count is not 181")
    expected_registered = config["registered_artifacts"]
    if outputs[: runtime["predecessor_manifest_output_count"]] != old_manifest["outputs"]:
        raise RuntimeSyncError("predecessor manifest output prefix drift")
    registered_start = runtime["predecessor_manifest_output_count"]
    if outputs[registered_start : registered_start + 14] != expected_registered:
        raise RuntimeSyncError("registered exact14 output metadata drift")
    expected_tail_names = list(_predecessor_snapshot_names(config).values()) + [runtime["sync_name"]]
    actual_tail_names = [Path(item.get("absolute_path", "")).name for item in outputs[-4:]]
    if actual_tail_names != expected_tail_names:
        raise RuntimeSyncError("immutable output order drift")
    if len({item.get("absolute_path") for item in outputs}) != 181:
        raise RuntimeSyncError("successor output paths are not unique")
    expected_scientific = _scientific_fields(config["successor_scientific_state"])
    for document, label in ((status, "STATUS"), (manifest, "RUN_MANIFEST")):
        for key, value in expected_scientific.items():
            _expect(document.get(key), value, label=f"{label}.{key}")
        _expect(document.get("claim_status"), "NOT_ESTABLISHED", label=f"{label}.claim_status")
        _expect(document.get("canonical_record_count"), 0, label=f"{label}.canonical_record_count")
    if successors.get(runtime["sync_name"]) is None:
        raise RuntimeSyncError("sync record is missing from prepared successors")
    sync = load_json(successors[runtime["sync_name"]], label=runtime["sync_name"])
    _expect(sync.get("event_id"), "A1-EVT-044", label="sync event id")
    _expect(sync.get("registered_artifact_count"), 14, label="sync exact14 count")
    _expect(sync.get("output_delta_count"), 18, label="sync output delta")
    _expect(sync.get("scientific_state_changed"), True, label="sync scientific state change")
    _expect(
        sync.get("evidence_gate_statuses_changed_since_evt043"),
        True,
        label="sync evidence gate change",
    )
    _expect(
        sync.get("overall_qualification_gate_changed"),
        False,
        label="sync overall qualification gate change",
    )


def _published_state_is_exact(config: dict[str, Any], payloads: Mapping[str, bytes]) -> bool:
    try:
        status, manifest, events = _parse_runtime(payloads)
    except RuntimeSyncError:
        return False
    runtime = config["runtime"]
    if len(events) != runtime["successor_event_count"]:
        return False
    outputs = manifest.get("outputs")
    return (
        events[-1].get("event_id") == "A1-EVT-044"
        and events[-1].get("decision_id") == "V3-DEC-019"
        and isinstance(outputs, list)
        and len(outputs) == runtime["successor_manifest_output_count"]
        and status.get("qualified") is False
        and status.get("training_started") is False
        and status.get("model_selection_allowed") is False
    )


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_immutable_once(path: Path, payload: bytes) -> str:
    if path.exists():
        try:
            observed = path.read_bytes()
        except OSError as exc:
            raise PublicationError(f"cannot read existing immutable output: {path}") from exc
        if observed != payload:
            raise PublicationError(f"immutable output differs: {path.name}")
        return "EXISTING_EXACT"
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_bytes() != payload:
            raise PublicationError(f"immutable output differs: {path.name}")
        return "EXISTING_EXACT"
    except OSError as exc:
        raise PublicationError(f"cannot create immutable output: {path.name}") from exc
    return "CREATED"


def _prepared_members(
    config: dict[str, Any], predecessor_payloads: Mapping[str, bytes], successors: Mapping[str, bytes]
) -> dict[str, bytes]:
    snapshots = _predecessor_snapshot_names(config)
    members = {snapshots[name]: predecessor_payloads[name] for name in MUTABLE_NAMES}
    members[config["runtime"]["sync_name"]] = successors[config["runtime"]["sync_name"]]
    members.update({name: successors[name] for name in MUTABLE_NAMES})
    return members


def _write_prepared(prepared: Path, members: Mapping[str, bytes]) -> None:
    prepared.mkdir(parents=True, exist_ok=True)
    observed = {item.name for item in prepared.iterdir()}
    if observed - set(members):
        raise PublicationError("prepared directory contains unexpected members")
    for name, payload in members.items():
        target = prepared / name
        if target.exists():
            if target.read_bytes() != payload:
                raise PublicationError(f"prepared member differs: {name}")
            continue
        _write_atomic(target, payload)
    if {item.name for item in prepared.iterdir()} != set(members):
        raise PublicationError("prepared member closure is incomplete")


def _read_prepared(config: dict[str, Any], prepared: Path) -> dict[str, bytes]:
    expected = set(_predecessor_snapshot_names(config).values()) | {
        config["runtime"]["sync_name"],
        *MUTABLE_NAMES,
    }
    try:
        observed = {item.name for item in prepared.iterdir()}
    except OSError as exc:
        raise PublicationError("prepared directory is absent") from exc
    if observed != expected:
        raise PublicationError("prepared member set is incomplete or contains extras")
    try:
        return {name: (prepared / name).read_bytes() for name in expected}
    except OSError as exc:
        raise PublicationError("cannot read prepared members") from exc


def _split_prepared(
    config: dict[str, Any], prepared: Mapping[str, bytes]
) -> tuple[dict[str, bytes], dict[str, bytes]]:
    snapshots = _predecessor_snapshot_names(config)
    predecessor = {name: prepared[snapshots[name]] for name in MUTABLE_NAMES}
    successor = {name: prepared[name] for name in MUTABLE_NAMES}
    return predecessor, successor


def _context(
    config_path: Path,
    *,
    production: bool,
    config_override: dict[str, Any] | None,
) -> dict[str, Any]:
    if production and config_override is not None:
        raise BindingError("config override is forbidden in production")
    if config_override is None:
        config, payload = _load_config_payload(config_path, require_bound=True)
        if production:
            audit_production_repository_authority(config, payload)
        return config
    config = copy.deepcopy(config_override)
    validate_bound_config(config)
    return config


def prepare_runtime_sync(
    *,
    prepared_directory: Path | str,
    recorded_at: str,
    config_path: Path = PRODUCTION_CONFIG_PATH,
    production: bool = True,
    config_override: dict[str, Any] | None = None,
    run_root_override: Path | None = None,
    fault_injector: FaultInjector | None = None,
) -> dict[str, Any]:
    del fault_injector  # Preparation has no supported injected write point.
    config = _context(config_path, production=production, config_override=config_override)
    prepared = _prepared_path(prepared_directory, config)
    run_root = run_root_override or Path(config["runtime"]["run_root"])
    if production and run_root_override is not None:
        raise PublicationError("run-root override is forbidden in production")
    # Metadata validation is deliberately body-free and occurs after binding,
    # so UNKNOWN config cannot touch a source/output path.
    validate_registered_artifacts(config)
    with _locked_run(run_root):
        predecessor = _read_runtime(run_root)
        if _published_state_is_exact(config, predecessor):
            return {"status": "ALREADY_PUBLISHED_VERIFIED", "event_id": "A1-EVT-044"}
        successors = build_successors(config, predecessor, recorded_at)
    _write_prepared(prepared, _prepared_members(config, predecessor, successors))
    return {
        "status": "PREPARED_NOT_PUBLISHED",
        "event_id": "A1-EVT-044",
        "prepared_directory": str(prepared),
        "prepared_member_count": 7,
        "manifest_output_transition": "163_TO_181",
        "new_runtime_output_count": 18,
    }


def publish_prepared(
    *,
    prepared_directory: Path | str,
    config_path: Path = PRODUCTION_CONFIG_PATH,
    production: bool = True,
    config_override: dict[str, Any] | None = None,
    run_root_override: Path | None = None,
    fault_injector: FaultInjector | None = None,
) -> dict[str, Any]:
    config = _context(config_path, production=production, config_override=config_override)
    prepared_path = _prepared_path(prepared_directory, config)
    run_root = run_root_override or Path(config["runtime"]["run_root"])
    if production and run_root_override is not None:
        raise PublicationError("run-root override is forbidden in production")
    validate_registered_artifacts(config)
    prepared = _read_prepared(config, prepared_path)
    predecessor, successor = _split_prepared(config, prepared)
    validate_successors(config, predecessor, {**prepared, **successor})
    snapshots = _predecessor_snapshot_names(config)
    immutable_payloads = {
        **{snapshots[name]: predecessor[name] for name in MUTABLE_NAMES},
        config["runtime"]["sync_name"]: prepared[config["runtime"]["sync_name"]],
    }
    with _locked_run(run_root):
        current = _read_runtime(run_root)
        states: list[str] = []
        for name in MUTABLE_NAMES:
            if current[name] == predecessor[name]:
                states.append("OLD")
            elif current[name] == successor[name]:
                states.append("NEW")
            else:
                raise PredecessorError(
                    f"runtime mutable is neither predecessor nor successor: {name}"
                )
        allowed = (
            ["OLD", "OLD", "OLD"],
            ["NEW", "OLD", "OLD"],
            ["NEW", "NEW", "OLD"],
            ["NEW", "NEW", "NEW"],
        )
        if states not in allowed:
            raise PredecessorError(f"runtime mutable order is not recoverable: {states!r}")
        immutable_results: dict[str, str] = {}
        for name in config["runtime"]["immutable_publish_order"]:
            if fault_injector is not None:
                fault_injector(f"before_immutable:{name}")
            immutable_results[name] = _write_immutable_once(run_root / name, immutable_payloads[name])
        if states == ["NEW", "NEW", "NEW"]:
            return {
                "status": "PUBLISHED_VERIFIED",
                "event_id": "A1-EVT-044",
                "reused": True,
                "immutable_results": immutable_results,
            }
        try:
            for index, name in enumerate(MUTABLE_NAMES):
                if states[index] == "NEW":
                    continue
                if fault_injector is not None:
                    fault_injector(f"before_replace:{name}")
                _write_atomic(run_root / name, successor[name])
                states[index] = "NEW"
        except Exception as exc:
            after = _read_runtime(run_root)
            if all(after[name] == successor[name] for name in MUTABLE_NAMES):
                return {
                    "status": "PUBLISHED_VERIFIED_AFTER_RECHECK",
                    "event_id": "A1-EVT-044",
                    "immutable_results": immutable_results,
                }
            raise PublicationError(
                "EVT044 was not committed; retry with the same prepared directory"
            ) from exc
        final = _read_runtime(run_root)
        if final != successor:
            raise PublicationError("EVT044 publication finished with non-exact mutables")
        return {
            "status": "PUBLISHED_VERIFIED",
            "event_id": "A1-EVT-044",
            "reused": False,
            "immutable_results": immutable_results,
        }


def validate_published(
    *,
    prepared_directory: Path | str,
    config_path: Path = PRODUCTION_CONFIG_PATH,
    production: bool = True,
    config_override: dict[str, Any] | None = None,
    run_root_override: Path | None = None,
) -> dict[str, Any]:
    config = _context(config_path, production=production, config_override=config_override)
    prepared_path = _prepared_path(prepared_directory, config)
    run_root = run_root_override or Path(config["runtime"]["run_root"])
    if production and run_root_override is not None:
        raise PublicationError("run-root override is forbidden in production")
    prepared = _read_prepared(config, prepared_path)
    predecessor, successor = _split_prepared(config, prepared)
    validate_successors(config, predecessor, {**prepared, **successor})
    with _locked_run(run_root):
        current = _read_runtime(run_root)
        if current != successor:
            raise PublicationError("runtime does not exactly match prepared EVT044 successor")
        for name in config["runtime"]["immutable_publish_order"]:
            expected = prepared[name]
            if (run_root / name).read_bytes() != expected:
                raise PublicationError(f"immutable output does not match prepared {name}")
    return {"status": "PUBLISHED_VERIFIED", "event_id": "A1-EVT-044"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--prepared-directory", type=Path, required=True)
    prepare.add_argument("--recorded-at", required=True)
    publish = subparsers.add_parser("publish")
    publish.add_argument("--prepared-directory", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--prepared-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_runtime_sync(
            prepared_directory=args.prepared_directory, recorded_at=args.recorded_at
        )
    elif args.command == "publish":
        result = publish_prepared(prepared_directory=args.prepared_directory)
    else:
        result = validate_published(prepared_directory=args.prepared_directory)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
