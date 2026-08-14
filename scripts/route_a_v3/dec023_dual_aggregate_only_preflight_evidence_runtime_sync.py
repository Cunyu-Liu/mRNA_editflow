#!/usr/bin/env python3
"""Prepare and publish the append-only A1-EVT-057 DEC-023 evidence sync.

The two registered ordinary-public reports are each validated as one exact byte
string.  Their JSON bodies are deliberately never parsed here.  Publication
adds their in-place metadata to RUN_MANIFEST, snapshots the three EVT056 runtime
mutables, writes one sync record, then commits STATUS, RUN_MANIFEST, and
EVENT_LOG in that order.  EVT057 is therefore the commit point.

The lifecycle is directly exact4 ledger L -> exact3 implementation I ->
config-only B.  The staging candidate may retain the entire ledger group and
exactly four implementation-binding scalars as UNKNOWN; production requires
both groups bound.  Every production entry point proves the exact chain, clean
worktree/index, branch, upstream, and origin equality before reading either
report, the prepared directory, or the runtime.
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
MUTABLE_NAMES = ("STATUS.json", "RUN_MANIFEST.json", "EVENT_LOG.jsonl")
SCRIPT_REPO_PATH = (
    "scripts/route_a_v3/"
    "dec023_dual_aggregate_only_preflight_evidence_runtime_sync.py"
)
TEST_REPO_PATH = (
    "tests/route_a_v3/"
    "test_dec023_dual_aggregate_only_preflight_evidence_runtime_sync.py"
)
CONFIG_REPO_PATH = (
    "configs/"
    "route_a_v3_dec023_dual_aggregate_only_preflight_evidence_runtime_sync_v1.json"
)
PRODUCTION_CONFIG_PATH = Path(__file__).resolve().parents[2] / CONFIG_REPO_PATH
PRODUCTION_REPO_ROOT = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/routea_v3_a1_20260810"
)
BRANCH = "routea-v3-a1-20260810"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
LEDGER_PATHS = [
    "docs/execution/route_a_v3_a1_interim.yaml",
    "docs/execution/route_a_v3_registry_manifest.json",
    "scripts/route_a_v3/validate_a0_bundle.py",
    "tests/route_a_v3/test_a0_integrity_guards.py",
]
LINEAGE_IDS = (
    "gse261709_public_identifier_asset_schema_aggregate_geometry_preflight_v1",
    "gse207584_aggregate_dense_family_qualification_preflight_v1",
)
LEDGER_INTEGRATION_ID = (
    "DEC023_GSE261709_AND_GSE207584_DUAL_PREFLIGHT_FINAL_EVIDENCE_V1_"
    "LEDGER_REGISTRATION"
)
LEDGER_MANIFEST_STATUS = (
    "DEC023_GSE261709_PUBLIC_SCHEMA_GEOMETRY_AND_GSE207584_DENSE_FAMILY_"
    "DUAL_PREFLIGHT_EVIDENCE_REGISTERED_EVT056_SETTLED_EVIDENCE_RUNTIME_"
    "PENDING_A1_INCOMPLETE_A6_IN_PROGRESS_L3_NOT_ESTABLISHED_A7_NOT_RUN"
)
TRUTH_SECTION_NAMES = (
    "registered_artifacts",
    "runtime",
    "frozen_scientific_state",
    "registered_evidence_truth",
    "access_boundary",
    "publication_policy",
)

FaultInjector = Callable[[str], None]


class RuntimeSyncError(RuntimeError):
    """Base error for the EVT057 publisher."""


class BindingError(RuntimeSyncError):
    """A grouped authority or implementation binding is incomplete."""


class AuthorityError(RuntimeSyncError):
    """The production Git lineage is not exact4 L -> exact3 I -> config-only B."""


class PredecessorError(RuntimeSyncError):
    """The runtime is neither exact EVT056 nor a supported EVT057 prefix."""


class PublicationError(RuntimeSyncError):
    """Prepared or runtime publication failed."""


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def runtime_science_report_truth_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the semantic surface that config-only B must preserve from I."""

    return {key: copy.deepcopy(config[key]) for key in TRUTH_SECTION_NAMES}


def runtime_science_report_truth_sha256(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        runtime_science_report_truth_projection(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def compact_json_line(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeSyncError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
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
        raise RuntimeSyncError(
            f"{label} drift: expected {expected!r}, observed {actual!r}"
        )


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


def _binding_values(binding: Mapping[str, Any]) -> list[Any]:
    return [
        binding.get(key)
        for key in (
            "status",
            "implementation_commit",
            "implementation_script_sha256",
            "implementation_test_sha256",
        )
    ]


def _binding_is_unknown(binding: Mapping[str, Any]) -> bool:
    return _binding_values(binding) == [UNKNOWN] * 4


def _ledger_values(ledger: Mapping[str, Any]) -> list[Any]:
    lineage_ids = ledger.get("registered_lineage_ids")
    blobs = ledger.get("frozen_blobs")
    lineages = lineage_ids if isinstance(lineage_ids, list) else []
    digests = (
        [item.get("sha256") for item in blobs if isinstance(item, Mapping)]
        if isinstance(blobs, list)
        else []
    )
    return [
        ledger.get("status"),
        ledger.get("commit"),
        ledger.get("integration_id"),
        ledger.get("manifest_status"),
        *lineages,
        *digests,
    ]


def _ledger_is_unknown(ledger: Mapping[str, Any]) -> bool:
    return _ledger_values(ledger) == [UNKNOWN] * 10


def _validate_grouped_binding(binding: Mapping[str, Any]) -> None:
    values = _binding_values(binding)
    if any(value == UNKNOWN for value in values):
        if not _binding_is_unknown(binding):
            raise BindingError("implementation binding is partially known")
        return
    _expect(binding.get("status"), "BOUND", label="implementation status")
    _expect_hex(binding.get("implementation_commit"), HEX40, label="I commit")
    _expect_hex(
        binding.get("implementation_script_sha256"), HEX64, label="script SHA"
    )
    _expect_hex(
        binding.get("implementation_test_sha256"), HEX64, label="test SHA"
    )


def _validate_grouped_ledger(ledger: Mapping[str, Any]) -> None:
    values = _ledger_values(ledger)
    if any(value == UNKNOWN for value in values):
        if not _ledger_is_unknown(ledger):
            raise BindingError("predecessor ledger is partially known")
        return
    _expect(ledger.get("status"), "BOUND", label="ledger status")
    _expect_hex(ledger.get("commit"), HEX40, label="L commit")
    _expect(
        ledger.get("integration_id"),
        LEDGER_INTEGRATION_ID,
        label="ledger integration",
    )
    _expect(
        ledger.get("manifest_status"),
        LEDGER_MANIFEST_STATUS,
        label="ledger manifest status",
    )
    _expect(
        ledger.get("registered_lineage_ids"),
        list(LINEAGE_IDS),
        label="ledger ordered lineage IDs",
    )
    for index, digest in enumerate(values[-4:]):
        _expect_hex(digest, HEX64, label=f"ledger blob {index} SHA")


def _expected_reports() -> list[dict[str, Any]]:
    return [
        {
            "lineage_id": LINEAGE_IDS[0],
            "name": (
                "GSE261709_PUBLIC_IDENTIFIER_ASSET_SCHEMA_AGGREGATE_GEOMETRY_"
                "PREFLIGHT_V1.json"
            ),
            "artifact_type": (
                "PUBLIC_IDENTIFIER_ASSET_SCHEMA_AGGREGATE_GEOMETRY_PREFLIGHT_ONLY"
            ),
            "absolute_path": (
                "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
                "GSE261709_PUBLIC_IDENTIFIER_ASSET_SCHEMA_GEOMETRY_PREFLIGHT_B3_d3177b0/"
                "GSE261709_PUBLIC_IDENTIFIER_ASSET_SCHEMA_AGGREGATE_GEOMETRY_"
                "PREFLIGHT_V1.json"
            ),
            "bytes": 6748,
            "sha256": (
                "ca68fb3a0e18e9c4989c3449f9b88c5112cc737e54ec026ce7c3b0df83386400"
            ),
        },
        {
            "lineage_id": LINEAGE_IDS[1],
            "name": "GSE207584_AGGREGATE_DENSE_FAMILY_QUALIFICATION_PREFLIGHT_V1.json",
            "artifact_type": "AGGREGATE_DENSE_FAMILY_QUALIFICATION_PREFLIGHT_ONLY",
            "absolute_path": (
                "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
                "GSE207584_AGGREGATE_DENSE_FAMILY_PREFLIGHT_B4_021ba2a/"
                "GSE207584_AGGREGATE_DENSE_FAMILY_QUALIFICATION_PREFLIGHT_V1.json"
            ),
            "bytes": 7755,
            "sha256": (
                "a50329b862f41415b3ff33e8fc251d07457449e0d75f31501238eaf30feba6b1"
            ),
        },
    ]


def _expected_scientific_state() -> dict[str, Any]:
    return {
        "current_qualified_counts": {
            "ordinary": 1,
            "a1": 1,
            "true_a2": 0,
            "canonical_records": 6547,
        },
        "gse261709_contribution": {
            "ordinary": 0,
            "a1": 0,
            "true_a2": 0,
            "canonical_records": 0,
        },
        "gse207584_contribution": {
            "ordinary": 0,
            "a1": 0,
            "true_a2": 0,
            "canonical_records": 0,
        },
        "run_status": "IN_PROGRESS",
        "evidence_status": "SCRATCH_ROUTE_QUALIFIED_GLOBAL_PHASE_INCOMPLETE",
        "gate_status": "A1_PHASE_INCOMPLETE_GLOBAL_REQUIREMENTS",
        "a1_complete": False,
        "qualified": False,
        "training_started": False,
        "training_allowed": False,
        "training_authorized": False,
        "gpu_work_started": False,
        "gpu_work_allowed": False,
        "model_selection_allowed": False,
        "a7_allowed": False,
        "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }


def _expected_evidence_truth() -> dict[str, Any]:
    return {
        "combined_status": "STOP_BOTH_PREFLIGHT_PROTOCOLS_NO_QUALIFICATION",
        "qualification_changed": False,
        "gse261709_public_identifier_asset_schema_aggregate_geometry_preflight": {
            "record_type": (
                "PUBLIC_IDENTIFIER_ASSET_SCHEMA_AND_AGGREGATE_GEOMETRY_"
                "PREFLIGHT_ONLY"
            ),
            "status": "STOP_PREFLIGHT_GATES_NOT_CLOSED",
            "required_gate_count": 3,
            "gate_status_counts": {
                "PASS": 1,
                "BLOCKED": 2,
                "UNKNOWN_NOT_ASSERTED": 0,
            },
            "all_required_gates_pass": False,
            "qualified": False,
            "ordinary_study_contribution": 0,
            "a1_study_contribution": 0,
            "true_a2_study_contribution": 0,
            "canonical_record_count": 0,
            "row_level_authority_granted": False,
            "sole_next_action": "STOP_PREFLIGHT_GATES_NOT_CLOSED",
        },
        "gse207584_aggregate_dense_family_qualification_preflight": {
            "record_type": "AGGREGATE_DENSE_FAMILY_QUALIFICATION_PREFLIGHT_ONLY",
            "status": "STOP_CURRENT_PROTOCOL_NOT_QUALIFIED",
            "required_gate_count": 11,
            "gate_status_counts": {
                "PASS_PREFLIGHT_ONLY": 1,
                "FAIL_CLOSED": 2,
                "UNKNOWN_NOT_ASSERTED": 8,
            },
            "all_required_gates_pass": False,
            "qualified": False,
            "ordinary_study_contribution": 0,
            "a1_study_contribution": 0,
            "true_a2_study_contribution": 0,
            "canonical_record_count": 0,
            "sole_next_action": "STOP_CURRENT_PROTOCOL_NOT_QUALIFIED",
        },
    }


def _expected_access_boundary() -> dict[str, Any]:
    return {
        "registered_artifact_count": 2,
        "registered_artifact_exact_byte_validation_count": 2,
        "registered_artifact_body_parse_count": 0,
        "registered_artifact_payload_field_read_count": 0,
        "registered_artifacts_copied": False,
        "public_asset_read_count": 0,
        "private_payload_read_count": 0,
        "sealed_payload_read_count": 0,
        "member_payload_read_count": 0,
        "row_payload_read_count": 0,
        "sequence_payload_read_count": 0,
        "effect_payload_read_count": 0,
        "canonical_materialization_count": 0,
        "qualification_run_count": 0,
        "training_run_count": 0,
        "gpu_work_count": 0,
        "model_selection_run_count": 0,
        "restricted_or_sealed_path_accessed": False,
        "gse246381_contact": False,
    }


def validate_static_config(config: dict[str, Any]) -> None:
    """Validate the staged or bound config without opening external paths."""

    _expect_keys(
        config,
        {
            "schema_version",
            "protocol_id",
            "contract_id",
            "phase_id",
            "dataset_ids",
            "decision_id",
            "event_id",
            "event_name",
            "sync_type",
            "implementation_binding",
            "repository_authority",
            "registered_artifacts",
            "runtime",
            "frozen_scientific_state",
            "registered_evidence_truth",
            "access_boundary",
            "publication_policy",
        },
        label="config root",
    )
    _expect(
        {
            key: config[key]
            for key in (
                "schema_version",
                "protocol_id",
                "contract_id",
                "phase_id",
                "dataset_ids",
                "decision_id",
                "event_id",
                "event_name",
                "sync_type",
            )
        },
        {
            "schema_version": (
                "route_a_v3_dec023_dual_aggregate_only_preflight_evidence_"
                "runtime_sync.v1"
            ),
            "protocol_id": (
                "ROUTE_A_V3_DEC023_DUAL_AGGREGATE_ONLY_PREFLIGHT_EVIDENCE_"
                "RUNTIME_SYNC_V1"
            ),
            "contract_id": "mrna_xeditflow_route_a_v3",
            "phase_id": "A1",
            "dataset_ids": ["GSE261709", "GSE207584"],
            "decision_id": "V3-DEC-023",
            "event_id": "A1-EVT-057",
            "event_name": (
                "GSE261709_PUBLIC_IDENTIFIER_ASSET_SCHEMA_AGGREGATE_GEOMETRY_AND_"
                "GSE207584_AGGREGATE_DENSE_FAMILY_QUALIFICATION_PREFLIGHTS_"
                "REGISTERED_QUALIFICATION_UNCHANGED"
            ),
            "sync_type": (
                "APPEND_ONLY_PUBLIC_AGGREGATE_EVIDENCE_REGISTRATION_"
                "NO_SCIENTIFIC_STATE_CHANGE"
            ),
        },
        label="config identity",
    )

    binding = _expect_keys(
        config["implementation_binding"],
        {
            "binding_scheme",
            "status",
            "implementation_commit",
            "implementation_script_path",
            "implementation_script_sha256",
            "implementation_test_path",
            "implementation_test_sha256",
            "unknown_to_bound_scalar_paths",
            "activation_rule",
        },
        label="implementation binding",
    )
    _expect(
        binding["binding_scheme"],
        "LEDGER_L_EXACT4_THEN_I_EXACT3_THEN_CONFIG_ONLY_B_V1",
        label="binding scheme",
    )
    _expect(binding["implementation_script_path"], SCRIPT_REPO_PATH, label="script path")
    _expect(binding["implementation_test_path"], TEST_REPO_PATH, label="test path")
    _expect(
        binding["unknown_to_bound_scalar_paths"],
        [
            "implementation_binding.status",
            "implementation_binding.implementation_commit",
            "implementation_binding.implementation_script_sha256",
            "implementation_binding.implementation_test_sha256",
        ],
        label="four scalar binding paths",
    )
    if not isinstance(binding["activation_rule"], str) or not binding["activation_rule"]:
        raise BindingError("binding activation rule is absent")
    _validate_grouped_binding(binding)

    authority = _expect_keys(
        config["repository_authority"],
        {
            "production_repo_root",
            "branch",
            "implementation_exact_changed_paths",
            "binding_exact_changed_paths",
            "predecessor_ledger",
        },
        label="repository authority",
    )
    _expect(
        authority["production_repo_root"],
        str(PRODUCTION_REPO_ROOT),
        label="production repo root",
    )
    _expect(authority["branch"], BRANCH, label="production branch")
    _expect(
        authority["implementation_exact_changed_paths"],
        [CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH],
        label="I exact3 paths",
    )
    _expect(
        authority["binding_exact_changed_paths"],
        [CONFIG_REPO_PATH],
        label="B config-only path",
    )
    ledger = _expect_keys(
        authority["predecessor_ledger"],
        {
            "status",
            "commit",
            "integration_id",
            "manifest_status",
            "registered_lineage_ids",
            "exact_changed_paths",
            "frozen_blobs",
        },
        label="predecessor ledger",
    )
    _expect(ledger["exact_changed_paths"], LEDGER_PATHS, label="ledger exact4 paths")
    blobs = ledger["frozen_blobs"]
    if not isinstance(blobs, list) or len(blobs) != 4:
        raise BindingError("ledger frozen blob closure drift")
    for item, path in zip(blobs, LEDGER_PATHS):
        _expect_keys(item, {"path", "sha256"}, label=f"ledger blob {path}")
        _expect(item["path"], path, label="ledger blob path")
    _validate_grouped_ledger(ledger)
    if _ledger_is_unknown(ledger) and not _binding_is_unknown(binding):
        raise BindingError("implementation cannot be bound before the ledger")

    _expect(config["registered_artifacts"], _expected_reports(), label="exact2 reports")
    if not _ledger_is_unknown(ledger) and ledger["registered_lineage_ids"] != [
        item["lineage_id"] for item in config["registered_artifacts"]
    ]:
        raise RuntimeSyncError("report lineage differs from the exact4 ledger")

    runtime = _expect_keys(
        config["runtime"],
        {
            "run_root",
            "allowed_prepared_root",
            "predecessor_event_id",
            "predecessor_event_count",
            "successor_event_id",
            "successor_event_count",
            "predecessor_manifest_output_count",
            "successor_manifest_output_count",
            "predecessor_manifest_registered_artifact_count",
            "successor_manifest_registered_artifact_count",
            "predecessor_mutables",
            "predecessor_tail",
            "sync_name",
            "output_delta_count",
            "immutable_publish_order",
            "mutable_publish_order",
        },
        label="runtime",
    )
    for key in ("run_root", "allowed_prepared_root"):
        value = runtime[key]
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise RuntimeSyncError(f"runtime {key} must be absolute")
    _expect(runtime["predecessor_event_id"], "A1-EVT-056", label="predecessor event")
    _expect(runtime["successor_event_id"], "A1-EVT-057", label="successor event")
    for key, expected in (
        ("predecessor_event_count", 56),
        ("successor_event_count", 57),
        ("predecessor_manifest_output_count", 242),
        ("successor_manifest_output_count", 248),
        ("predecessor_manifest_registered_artifact_count", 6),
        ("successor_manifest_registered_artifact_count", 8),
        ("output_delta_count", 6),
    ):
        _expect(runtime[key], expected, label=f"runtime {key}")
    _expect(runtime["mutable_publish_order"], list(MUTABLE_NAMES), label="mutable order")
    snapshots = runtime["predecessor_mutables"]
    if not isinstance(snapshots, dict) or set(snapshots) != set(MUTABLE_NAMES):
        raise RuntimeSyncError("predecessor mutable closure drift")
    for mutable in MUTABLE_NAMES:
        spec = _expect_keys(
            snapshots[mutable],
            {"bytes", "sha256", "snapshot_name"},
            label=f"{mutable} predecessor identity",
        )
        if isinstance(spec["bytes"], bool) or not isinstance(spec["bytes"], int):
            raise RuntimeSyncError(f"{mutable} byte count is not an integer")
        _expect_hex(spec["sha256"], HEX64, label=f"{mutable} predecessor SHA")
        if not isinstance(spec["snapshot_name"], str) or not spec["snapshot_name"]:
            raise RuntimeSyncError(f"{mutable} snapshot name is absent")
    tail = _expect_keys(
        runtime["predecessor_tail"],
        {"event_id", "predecessor_event_id", "decision_id", "bytes", "sha256"},
        label="predecessor tail",
    )
    _expect(tail["event_id"], "A1-EVT-056", label="tail event")
    _expect(tail["predecessor_event_id"], "A1-EVT-055", label="tail predecessor")
    _expect(tail["decision_id"], "V3-DEC-023", label="tail decision")
    if isinstance(tail["bytes"], bool) or not isinstance(tail["bytes"], int):
        raise RuntimeSyncError("tail byte count is not an integer")
    _expect_hex(tail["sha256"], HEX64, label="tail SHA")
    _expect(
        runtime["sync_name"],
        "A1_DEC023_DUAL_AGGREGATE_ONLY_PREFLIGHT_EVIDENCE_RUNTIME_SYNC_V1.json",
        label="sync name",
    )
    expected_immutable = [snapshots[name]["snapshot_name"] for name in MUTABLE_NAMES] + [
        runtime["sync_name"]
    ]
    _expect(runtime["immutable_publish_order"], expected_immutable, label="immutable order")

    _expect(
        config["frozen_scientific_state"],
        _expected_scientific_state(),
        label="frozen scientific state",
    )
    _expect(
        config["registered_evidence_truth"],
        _expected_evidence_truth(),
        label="aggregate evidence truth",
    )
    _expect(
        config["access_boundary"],
        _expected_access_boundary(),
        label="access boundary",
    )
    _expect(
        config["publication_policy"],
        {
            "registered_artifacts_remain_in_place": True,
            "registered_artifact_order_is_ledger_lineage_order": True,
            "registered_artifact_bodies_are_not_parsed": True,
            "predecessor_snapshots_are_immutable_runtime_outputs": True,
            "sync_record_is_immutable_runtime_output": True,
            "mutables_commit_after_all_immutables": True,
            "event_is_last_commit": True,
            "supported_recovery": "EXACT_PUBLICATION_PREFIX_ONLY",
        },
        label="publication policy",
    )


def validate_bound_config(config: dict[str, Any]) -> None:
    validate_static_config(config)
    if _binding_is_unknown(config["implementation_binding"]):
        raise BindingError("implementation binding remains UNKNOWN_NOT_ASSERTED")
    if _ledger_is_unknown(config["repository_authority"]["predecessor_ledger"]):
        raise BindingError("predecessor ledger remains UNKNOWN_NOT_ASSERTED")


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


def expected_unknown_i_config(bound_config: dict[str, Any]) -> dict[str, Any]:
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
        raise AuthorityError("read-only Git command did not complete") from exc
    if result.returncode != 0:
        raise AuthorityError(f"read-only Git command failed: {arguments!r}")
    return result.stdout


def _git_blob(repo_root: Path, commit: str, path: str) -> bytes:
    return _run_git(repo_root, "show", f"{commit}:{path}")


def _changed_paths(repo_root: Path, commit: str) -> list[str]:
    return sorted(
        _run_git(
            repo_root, "diff-tree", "--no-commit-id", "--name-only", "-r", commit
        )
        .decode("utf-8")
        .splitlines()
    )


def _read_repo_file(repo_root: Path, path: str) -> bytes:
    try:
        return (repo_root / path).read_bytes()
    except OSError as exc:
        raise AuthorityError(f"cannot read repository file: {path}") from exc


def audit_production_repository_authority(
    config: dict[str, Any], config_payload: bytes
) -> dict[str, Any]:
    """Prove the direct exact4 L -> exact3 I -> config-only B chain."""

    validate_bound_config(config)
    authority = config["repository_authority"]
    binding = config["implementation_binding"]
    ledger = authority["predecessor_ledger"]
    repo_root = Path(authority["production_repo_root"])
    branch = authority["branch"]
    l_commit = ledger["commit"]
    i_commit = binding["implementation_commit"]

    expected_executing_path = (repo_root / SCRIPT_REPO_PATH).resolve()
    executing_path = Path(__file__).resolve()
    if executing_path != expected_executing_path:
        raise AuthorityError("executing script path differs from bound repository path")
    try:
        executing_payload = executing_path.read_bytes()
    except OSError as exc:
        raise AuthorityError("cannot read executing script bytes") from exc

    head = _run_git(repo_root, "rev-parse", "HEAD").decode().strip()
    current_branch = _run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD").decode().strip()
    upstream_branch = _run_git(
        repo_root, "rev-parse", "--abbrev-ref", "@{upstream}"
    ).decode().strip()
    upstream_head = _run_git(repo_root, "rev-parse", "@{upstream}").decode().strip()
    origin_head = _run_git(
        repo_root, "rev-parse", "--verify", f"refs/remotes/origin/{branch}"
    ).decode().strip()
    _expect(current_branch, branch, label="current branch")
    _expect(upstream_branch, f"origin/{branch}", label="upstream branch")
    _expect(head, upstream_head, label="HEAD/upstream")
    _expect(head, origin_head, label="HEAD/origin")
    if _run_git(repo_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise AuthorityError("production worktree or index is dirty")

    b_parent = _run_git(repo_root, "rev-parse", f"{head}^").decode().strip()
    i_parent = _run_git(repo_root, "rev-parse", f"{i_commit}^").decode().strip()
    _expect(b_parent, i_commit, label="B parent/I")
    _expect(i_parent, l_commit, label="I parent/L")
    _expect(
        _changed_paths(repo_root, head),
        sorted(authority["binding_exact_changed_paths"]),
        label="B changed paths",
    )
    _expect(
        _changed_paths(repo_root, i_commit),
        sorted(authority["implementation_exact_changed_paths"]),
        label="I changed paths",
    )
    _expect(
        _changed_paths(repo_root, l_commit),
        sorted(ledger["exact_changed_paths"]),
        label="L changed paths",
    )

    for item in ledger["frozen_blobs"]:
        path = item["path"]
        digest = item["sha256"]
        for commit, label in ((l_commit, "L"), (i_commit, "I"), (head, "B")):
            if sha256(_git_blob(repo_root, commit, path)) != digest:
                raise AuthorityError(f"{label} frozen ledger blob drift: {path}")
        if sha256(_read_repo_file(repo_root, path)) != digest:
            raise AuthorityError(f"worktree frozen ledger blob drift: {path}")

    i_config = load_json(
        _git_blob(repo_root, i_commit, CONFIG_REPO_PATH), label="I config"
    )
    if not _typed_equal(i_config, expected_unknown_i_config(config)):
        raise AuthorityError("I config is not the exact four-scalar UNKNOWN form")
    if not _typed_equal(
        runtime_science_report_truth_projection(config),
        runtime_science_report_truth_projection(i_config),
    ):
        raise AuthorityError("B runtime/science/report truth drift from I")
    if _git_blob(repo_root, head, CONFIG_REPO_PATH) != config_payload:
        raise AuthorityError("B config blob differs from supplied config")
    if _read_repo_file(repo_root, CONFIG_REPO_PATH) != config_payload:
        raise AuthorityError("worktree config differs from supplied config")

    for path, digest in (
        (SCRIPT_REPO_PATH, binding["implementation_script_sha256"]),
        (TEST_REPO_PATH, binding["implementation_test_sha256"]),
    ):
        i_blob = _git_blob(repo_root, i_commit, path)
        for commit, label in ((i_commit, "I"), (head, "B")):
            if sha256(i_blob if commit == i_commit else _git_blob(repo_root, commit, path)) != digest:
                raise AuthorityError(f"{label} implementation blob drift: {path}")
        if sha256(_read_repo_file(repo_root, path)) != digest:
            raise AuthorityError(f"worktree implementation blob drift: {path}")
        if path == SCRIPT_REPO_PATH:
            if sha256(executing_payload) != digest:
                raise AuthorityError("executing script SHA-256 differs from bound implementation")
            if executing_payload != i_blob:
                raise AuthorityError("executing script bytes differ from bound I script blob")

    return {
        "status": "PASS_DIRECT_EXACT4_L_TO_EXACT3_I_TO_CONFIG_ONLY_B",
        "ledger_l_commit": l_commit,
        "implementation_i_commit": i_commit,
        "binding_b_commit": head,
        "upstream_head_commit": upstream_head,
        "origin_branch_head_commit": origin_head,
        "worktree_and_index_clean": True,
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
        raise PublicationError("prepared directory must be a strict allowed descendant")
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


def _snapshot_names(config: dict[str, Any]) -> dict[str, str]:
    return {
        name: config["runtime"]["predecessor_mutables"][name]["snapshot_name"]
        for name in MUTABLE_NAMES
    }


def validate_registered_artifacts(
    config: dict[str, Any], *, verify_exact_bytes: bool
) -> dict[str, Any]:
    """Validate exact2 metadata and optionally exact bytes; never parse bodies."""

    if verify_exact_bytes:
        validate_bound_config(config)
    else:
        validate_static_config(config)
    artifacts = config["registered_artifacts"]
    if verify_exact_bytes:
        for item in artifacts:
            path = Path(item["absolute_path"])
            try:
                payload = path.read_bytes()
            except OSError as exc:
                raise PublicationError(f"cannot read report bytes: {path}") from exc
            if len(payload) != item["bytes"] or sha256(payload) != item["sha256"]:
                raise PublicationError(f"registered report identity drift: {path}")
    return {
        "status": (
            "EXACT2_BYTES_AND_SHA256_VALIDATED"
            if verify_exact_bytes
            else "EXACT2_METADATA_VALIDATED"
        ),
        "artifact_count": 2,
        "exact_byte_validation_count": 2 if verify_exact_bytes else 0,
        "body_parse_count": 0,
        "payload_field_read_count": 0,
        "public_asset_read_count": 0,
        "registered_artifacts": copy.deepcopy(artifacts),
    }


def _parse_runtime(
    payloads: Mapping[str, bytes]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    return (
        load_json(payloads["STATUS.json"], label="STATUS.json"),
        load_json(payloads["RUN_MANIFEST.json"], label="RUN_MANIFEST.json"),
        load_events(payloads["EVENT_LOG.jsonl"], label="EVENT_LOG.jsonl"),
    )


def _check_identity(payload: bytes, spec: Mapping[str, Any], *, label: str) -> None:
    if len(payload) != spec["bytes"] or sha256(payload) != spec["sha256"]:
        raise PredecessorError(f"{label} predecessor identity drift")


def _runtime_truth_fields(config: dict[str, Any]) -> dict[str, Any]:
    frozen = config["frozen_scientific_state"]
    counts = frozen["current_qualified_counts"]
    return {
        "qualified_ordinary_studies": counts["ordinary"],
        "qualified_a1_studies": counts["a1"],
        "qualified_a2_dense_studies": counts["true_a2"],
        "canonical_intervention_record_count": counts["canonical_records"],
        "canonical_record_count": counts["canonical_records"],
        "run_status": frozen["run_status"],
        "evidence_status": frozen["evidence_status"],
        "gate_status": frozen["gate_status"],
        "qualified": False,
        "training_started": False,
        "training_allowed": False,
        "training_authorized": False,
        "gpu_work_started": False,
        "gpu_work_allowed": False,
        "model_selection_allowed": False,
        "a7_allowed": False,
        "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }


def _predecessor_authority_fields() -> dict[str, Any]:
    return {
        "dec023_authority_runtime_sync_status": "SYNCED_EVT_056",
        "gse261709_dec023_aggregate_geometry_preflight_status": "AUTHORIZED_NOT_RUN",
        "gse207584_dec023_dense_family_preflight_status": "AUTHORIZED_NOT_RUN",
        "gse261709_contribution": {
            "ordinary": 0,
            "a1": 0,
            "true_a2": 0,
            "canonical_records": 0,
        },
        "gse207584_contribution": {
            "ordinary": 0,
            "a1": 0,
            "true_a2": 0,
            "canonical_records": 0,
        },
    }


def validate_predecessor(
    config: dict[str, Any], payloads: Mapping[str, bytes]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    validate_bound_config(config)
    if set(payloads) != set(MUTABLE_NAMES):
        raise PredecessorError("runtime mutable member closure differs")
    runtime = config["runtime"]
    for name in MUTABLE_NAMES:
        _check_identity(
            payloads[name], runtime["predecessor_mutables"][name], label=name
        )
    status, manifest, events = _parse_runtime(payloads)
    if len(events) != 56:
        raise PredecessorError("predecessor event count is not 56")
    _expect(
        [event.get("event_id") for event in events],
        [f"A1-EVT-{index:03d}" for index in range(1, 57)],
        label="predecessor event sequence",
    )
    _expect(events[-1].get("event_id"), "A1-EVT-056", label="tail event")
    _expect(events[-1].get("predecessor_event_id"), "A1-EVT-055", label="tail parent")
    _expect(events[-1].get("decision_id"), "V3-DEC-023", label="tail decision")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 242:
        raise PredecessorError("predecessor manifest output count is not 242")
    paths = [item.get("absolute_path") for item in outputs if isinstance(item, dict)]
    if len(paths) != 242 or len(set(paths)) != 242:
        raise PredecessorError("predecessor manifest output paths are not unique")
    _expect(
        manifest.get("registered_artifact_count"),
        6,
        label="predecessor registered artifact count",
    )
    for document, label in ((status, "STATUS"), (manifest, "RUN_MANIFEST")):
        for key, expected in _runtime_truth_fields(config).items():
            _expect(document.get(key), expected, label=f"{label}.{key}")
        for key, expected in _predecessor_authority_fields().items():
            _expect(document.get(key), expected, label=f"{label}.{key}")
    tail_payload = payloads["EVENT_LOG.jsonl"].splitlines(keepends=True)[-1]
    _check_identity(tail_payload, runtime["predecessor_tail"], label="event tail")
    return status, manifest, events


def _validate_recorded_at(recorded_at: str, predecessor_at: Any) -> None:
    if not isinstance(recorded_at, str) or not isinstance(predecessor_at, str):
        raise PredecessorError("timestamps must be explicit ISO-8601 strings")
    try:
        current = datetime.fromisoformat(recorded_at)
        predecessor = datetime.fromisoformat(predecessor_at)
    except ValueError as exc:
        raise PredecessorError("timestamp is not ISO-8601") from exc
    if current.tzinfo is None or predecessor.tzinfo is None or current <= predecessor:
        raise PredecessorError("EVT057 timestamp must follow EVT056 with an offset")


def _sync_record(
    config: dict[str, Any], *, recorded_at: str, snapshots: Mapping[str, bytes]
) -> bytes:
    return json_bytes(
        {
            "record_type": (
                "ROUTE_A_V3_A1_DEC023_DUAL_AGGREGATE_ONLY_PREFLIGHT_"
                "EVIDENCE_RUNTIME_SYNC"
            ),
            "event_id": "A1-EVT-057",
            "decision_id": "V3-DEC-023",
            "recorded_at": recorded_at,
            "predecessor_event_id": "A1-EVT-056",
            "registered_artifact_count": 2,
            "registered_artifacts": copy.deepcopy(config["registered_artifacts"]),
            "registered_artifact_exact_byte_validation_count": 2,
            "registered_artifact_body_parse_count": 0,
            "registered_artifact_payload_field_read_count": 0,
            "public_asset_read_count": 0,
            "registered_artifacts_copied": False,
            "predecessor_snapshot_count": 3,
            "predecessor_snapshot_names": list(snapshots),
            "predecessor_snapshot_sha256": {
                name: sha256(payload) for name, payload in snapshots.items()
            },
            "output_delta_count": 6,
            "successor_manifest_output_count": 248,
            "successor_manifest_registered_artifact_count": 8,
            "frozen_scientific_state": copy.deepcopy(
                config["frozen_scientific_state"]
            ),
            "registered_evidence_truth": copy.deepcopy(
                config["registered_evidence_truth"]
            ),
            "access_boundary": copy.deepcopy(config["access_boundary"]),
            "ledger_authority_status": config["repository_authority"][
                "predecessor_ledger"
            ]["status"],
            "implementation_binding_status": config["implementation_binding"][
                "status"
            ],
            "scientific_state_changed": False,
            "evidence_surface_changed": True,
            "evidence_gate_statuses_changed": True,
            "overall_qualification_gate_changed": False,
            "qualification_changed": False,
        }
    )


def _event_document(
    config: dict[str, Any], *, recorded_at: str, sync_digest: str
) -> dict[str, Any]:
    return {
        "event_id": "A1-EVT-057",
        "at": recorded_at,
        "phase_id": "A1",
        "event": config["event_name"],
        "sync_type": config["sync_type"],
        "decision_id": "V3-DEC-023",
        "predecessor_event_id": "A1-EVT-056",
        "registered_artifact_count": 2,
        "registered_lineage_ids": list(LINEAGE_IDS),
        "registered_artifacts_copied": False,
        "registered_artifact_exact_byte_validation_count": 2,
        "registered_artifact_body_parse_count": 0,
        "registered_artifact_payload_field_read_count": 0,
        "public_asset_read_count": 0,
        "predecessor_snapshot_count": 3,
        "predecessor_snapshot_names": list(_snapshot_names(config).values()),
        "sync_name": config["runtime"]["sync_name"],
        "sync_record_sha256": sync_digest,
        "output_delta_count": 6,
        "manifest_output_count_before": 242,
        "manifest_output_count_after": 248,
        "manifest_registered_artifact_count_before": 6,
        "manifest_registered_artifact_count_after": 8,
        "frozen_scientific_state": copy.deepcopy(
            config["frozen_scientific_state"]
        ),
        "registered_evidence_truth": copy.deepcopy(
            config["registered_evidence_truth"]
        ),
        "access_boundary": copy.deepcopy(config["access_boundary"]),
        "scientific_state_changed": False,
        "evidence_surface_changed": True,
        "evidence_gate_statuses_changed": True,
        "overall_qualification_gate_changed": False,
        "qualification_changed": False,
        "training_started": False,
        "training_allowed": False,
        "gpu_work_started": False,
        "gpu_work_allowed": False,
        "model_selection_allowed": False,
        "a7_allowed": False,
        "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "detail": (
            "Registered two in-place ordinary-public aggregate-only reports after "
            "exact-byte validation without parsing either body. GSE261709 retains "
            "one PASS and two BLOCKED gates with STOP_PREFLIGHT_GATES_NOT_CLOSED; "
            "GSE207584 retains exactly eleven gates with one PASS_PREFLIGHT_ONLY, "
            "two FAIL_CLOSED, and eight UNKNOWN_NOT_ASSERTED with "
            "STOP_CURRENT_PROTOCOL_NOT_QUALIFIED. Both datasets remain unqualified "
            "with zero contributions and zero canonical records; global 1/1/0 "
            "counts, 6547 canonical records, incomplete A1 status, and all training, "
            "GPU, model-selection, A7, next-phase, and scientific-claim locks remain "
            "unchanged."
        ),
    }


def _immutable_outputs(
    config: dict[str, Any],
    predecessor_outputs: list[dict[str, Any]],
    predecessor_payloads: Mapping[str, bytes],
    sync_payload: bytes,
) -> list[dict[str, Any]]:
    outputs = copy.deepcopy(predecessor_outputs)
    outputs.extend(copy.deepcopy(config["registered_artifacts"]))
    run_root = Path(config["runtime"]["run_root"])
    snapshots = _snapshot_names(config)
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
    config: dict[str, Any],
    predecessor_payloads: Mapping[str, bytes],
    recorded_at: str,
) -> dict[str, bytes]:
    status, manifest, events = validate_predecessor(config, predecessor_payloads)
    _validate_recorded_at(recorded_at, events[-1].get("at"))
    snapshot_names = _snapshot_names(config)
    snapshots = {
        snapshot_names[name]: predecessor_payloads[name] for name in MUTABLE_NAMES
    }
    sync_payload = _sync_record(config, recorded_at=recorded_at, snapshots=snapshots)
    successor_status = copy.deepcopy(status)
    successor_status["updated_at"] = recorded_at
    successor_manifest = copy.deepcopy(manifest)
    successor_manifest["outputs"] = _immutable_outputs(
        config, manifest["outputs"], predecessor_payloads, sync_payload
    )
    successor_manifest["registered_artifact_count"] = 8
    event = _event_document(
        config, recorded_at=recorded_at, sync_digest=sha256(sync_payload)
    )
    result = {
        **snapshots,
        config["runtime"]["sync_name"]: sync_payload,
        "STATUS.json": json_bytes(successor_status),
        "RUN_MANIFEST.json": json_bytes(successor_manifest),
        "EVENT_LOG.jsonl": predecessor_payloads["EVENT_LOG.jsonl"]
        + compact_json_line(event),
    }
    validate_successors(config, predecessor_payloads, result)
    return result


def validate_successors(
    config: dict[str, Any],
    predecessor_payloads: Mapping[str, bytes],
    successors: Mapping[str, bytes],
) -> None:
    old_status, old_manifest, old_events = validate_predecessor(
        config, predecessor_payloads
    )
    status, manifest, events = _parse_runtime(
        {name: successors[name] for name in MUTABLE_NAMES}
    )
    if len(events) != 57 or events[:-1] != old_events:
        raise RuntimeSyncError("EVT057 is not one append-only event")
    event = events[-1]
    _expect(event.get("event_id"), "A1-EVT-057", label="successor event")
    _expect(event.get("decision_id"), "V3-DEC-023", label="successor decision")
    _expect(event.get("scientific_state_changed"), False, label="scientific change")
    _expect(event.get("evidence_surface_changed"), True, label="evidence change")
    _expect(
        event.get("evidence_gate_statuses_changed"),
        True,
        label="evidence gate status change",
    )
    _expect(
        event.get("overall_qualification_gate_changed"),
        False,
        label="qualification gate change",
    )
    _expect(event.get("qualification_changed"), False, label="qualification change")
    _expect(
        event.get("registered_artifact_body_parse_count"),
        0,
        label="report body parse count",
    )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 248:
        raise RuntimeSyncError("successor manifest output count is not 248")
    if outputs[:242] != old_manifest["outputs"]:
        raise RuntimeSyncError("predecessor manifest output prefix drift")
    if outputs[242:244] != config["registered_artifacts"]:
        raise RuntimeSyncError("registered report output metadata/order drift")
    expected_tail = list(_snapshot_names(config).values()) + [
        config["runtime"]["sync_name"]
    ]
    if [Path(item.get("absolute_path", "")).name for item in outputs[-4:]] != expected_tail:
        raise RuntimeSyncError("immutable output tail order drift")
    if len({item.get("absolute_path") for item in outputs}) != 248:
        raise RuntimeSyncError("successor output paths are not unique")
    expected_status = copy.deepcopy(old_status)
    expected_status["updated_at"] = event["at"]
    _expect(status, expected_status, label="STATUS preservation")
    expected_manifest = copy.deepcopy(old_manifest)
    expected_manifest["outputs"] = outputs
    expected_manifest["registered_artifact_count"] = 8
    _expect(manifest, expected_manifest, label="RUN_MANIFEST preservation")
    sync_name = config["runtime"]["sync_name"]
    sync = load_json(successors[sync_name], label=sync_name)
    _expect(sync.get("event_id"), "A1-EVT-057", label="sync event")
    _expect(sync.get("registered_artifact_count"), 2, label="sync report count")
    _expect(sync.get("registered_artifact_body_parse_count"), 0, label="sync parse count")
    _expect(sync.get("output_delta_count"), 6, label="sync output delta")
    _expect(sync.get("scientific_state_changed"), False, label="sync scientific change")
    _expect(sync.get("evidence_surface_changed"), True, label="sync evidence change")
    _expect(
        sync.get("evidence_gate_statuses_changed"),
        True,
        label="sync evidence gate status change",
    )
    _expect(sync.get("qualification_changed"), False, label="sync qualification")


def _published_state_is_evt057(
    config: dict[str, Any], payloads: Mapping[str, bytes]
) -> bool:
    try:
        status, manifest, events = _parse_runtime(payloads)
    except RuntimeSyncError:
        return False
    outputs = manifest.get("outputs")
    return (
        len(events) == 57
        and events[-1].get("event_id") == "A1-EVT-057"
        and events[-1].get("decision_id") == "V3-DEC-023"
        and events[-1].get("scientific_state_changed") is False
        and isinstance(outputs, list)
        and len(outputs) == 248
        and manifest.get("registered_artifact_count") == 8
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


def _write_immutable_once(
    path: Path,
    payload: bytes,
    *,
    fault_injector: FaultInjector | None = None,
) -> str:
    """Publish a complete immutable via an exclusive same-directory link."""

    if path.exists():
        if path.read_bytes() != payload:
            raise PublicationError(f"immutable output differs: {path.name}")
        return "EXISTING_EXACT"
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.immutable.", dir=path.parent
        )
    except OSError as exc:
        raise PublicationError(f"cannot stage immutable output: {path.name}") from exc
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            split = max(1, len(payload) // 2) if payload else 0
            stream.write(payload[:split])
            stream.flush()
            if fault_injector is not None:
                fault_injector("after_partial_temp_write")
            stream.write(payload[split:])
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise PublicationError(f"immutable output differs: {path.name}")
            return "EXISTING_EXACT"
        except OSError as exc:
            raise PublicationError(f"cannot publish immutable output: {path.name}") from exc
        return "CREATED"
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _prepared_members(
    config: dict[str, Any], successors: Mapping[str, bytes]
) -> dict[str, bytes]:
    names = set(_snapshot_names(config).values()) | {
        config["runtime"]["sync_name"],
        *MUTABLE_NAMES,
    }
    return {name: successors[name] for name in names}


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
        else:
            _write_atomic(target, payload)
    if {item.name for item in prepared.iterdir()} != set(members):
        raise PublicationError("prepared member closure is incomplete")


def _read_prepared(config: dict[str, Any], prepared: Path) -> dict[str, bytes]:
    expected = set(_snapshot_names(config).values()) | {
        config["runtime"]["sync_name"],
        *MUTABLE_NAMES,
    }
    try:
        observed = {item.name for item in prepared.iterdir()}
    except OSError as exc:
        raise PublicationError("prepared directory is absent") from exc
    if observed != expected:
        raise PublicationError("prepared member set is incomplete or has extras")
    try:
        return {name: (prepared / name).read_bytes() for name in expected}
    except OSError as exc:
        raise PublicationError("cannot read prepared members") from exc


def _split_prepared(
    config: dict[str, Any], prepared: Mapping[str, bytes]
) -> tuple[dict[str, bytes], dict[str, bytes]]:
    snapshots = _snapshot_names(config)
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
    if config_override is not None:
        config = copy.deepcopy(config_override)
        validate_bound_config(config)
        return config
    config, payload = _load_config_payload(config_path, require_bound=True)
    if production:
        audit_production_repository_authority(config, payload)
    return config


def prepare_runtime_sync(
    *,
    prepared_directory: Path | str,
    recorded_at: str,
    config_path: Path = PRODUCTION_CONFIG_PATH,
    production: bool = True,
    config_override: dict[str, Any] | None = None,
    run_root_override: Path | None = None,
) -> dict[str, Any]:
    config = _context(
        config_path, production=production, config_override=config_override
    )
    prepared = _prepared_path(prepared_directory, config)
    run_root = run_root_override or Path(config["runtime"]["run_root"])
    if production and run_root_override is not None:
        raise PublicationError("run-root override is forbidden in production")
    validate_registered_artifacts(config, verify_exact_bytes=production)
    with _locked_run(run_root):
        predecessor = _read_runtime(run_root)
        if _published_state_is_evt057(config, predecessor):
            return {"status": "ALREADY_PUBLISHED_VERIFIED", "event_id": "A1-EVT-057"}
        successors = build_successors(config, predecessor, recorded_at)
    _write_prepared(prepared, _prepared_members(config, successors))
    return {
        "status": "PREPARED_NOT_PUBLISHED",
        "event_id": "A1-EVT-057",
        "prepared_directory": str(prepared),
        "prepared_member_count": 7,
        "manifest_output_transition": "242_TO_248",
        "manifest_registered_artifact_transition": "6_TO_8",
        "registered_artifact_body_parse_count": 0,
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
    config = _context(
        config_path, production=production, config_override=config_override
    )
    prepared_path = _prepared_path(prepared_directory, config)
    run_root = run_root_override or Path(config["runtime"]["run_root"])
    if production and run_root_override is not None:
        raise PublicationError("run-root override is forbidden in production")
    validate_registered_artifacts(config, verify_exact_bytes=production)
    prepared = _read_prepared(config, prepared_path)
    predecessor, successor = _split_prepared(config, prepared)
    validate_successors(config, predecessor, prepared)
    snapshots = _snapshot_names(config)
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
            raise PredecessorError(f"runtime prefix is not recoverable: {states!r}")
        immutable_results: dict[str, str] = {}
        for name in config["runtime"]["immutable_publish_order"]:
            if fault_injector is not None:
                fault_injector(f"before_immutable:{name}")
            immutable_results[name] = _write_immutable_once(
                run_root / name,
                immutable_payloads[name],
                fault_injector=fault_injector,
            )
        if states == ["NEW", "NEW", "NEW"]:
            return {
                "status": "PUBLISHED_VERIFIED",
                "event_id": "A1-EVT-057",
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
                    "event_id": "A1-EVT-057",
                    "immutable_results": immutable_results,
                }
            raise PublicationError(
                "EVT057 was not committed; retry the same prepared directory"
            ) from exc
        final = _read_runtime(run_root)
        if final != successor:
            raise PublicationError("EVT057 publication ended with non-exact mutables")
        return {
            "status": "PUBLISHED_VERIFIED",
            "event_id": "A1-EVT-057",
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
    config = _context(
        config_path, production=production, config_override=config_override
    )
    prepared_path = _prepared_path(prepared_directory, config)
    run_root = run_root_override or Path(config["runtime"]["run_root"])
    if production and run_root_override is not None:
        raise PublicationError("run-root override is forbidden in production")
    validate_registered_artifacts(config, verify_exact_bytes=production)
    prepared = _read_prepared(config, prepared_path)
    predecessor, successor = _split_prepared(config, prepared)
    validate_successors(config, predecessor, prepared)
    with _locked_run(run_root):
        current = _read_runtime(run_root)
        if current != successor:
            raise PublicationError("runtime does not match the prepared EVT057 successor")
        for name in config["runtime"]["immutable_publish_order"]:
            if (run_root / name).read_bytes() != prepared[name]:
                raise PublicationError(f"immutable output differs from prepared: {name}")
    return {
        "status": "PUBLISHED_VERIFIED",
        "event_id": "A1-EVT-057",
        "registered_artifact_body_parse_count": 0,
    }


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
