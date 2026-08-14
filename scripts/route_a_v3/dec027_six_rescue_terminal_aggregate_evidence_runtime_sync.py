#!/usr/bin/env python3
"""Prepare/publish the append-only DEC027 six-report A1-EVT-060 sync.

The six already-published aggregate reports are identity checked as byte strings;
their bodies are never parsed or copied.  The publisher registers those six
dynamic artifacts, three immutable EVT059 snapshots, and one immutable sync
record, then commits STATUS, RUN_MANIFEST, and EVENT_LOG in that order.  EVT060
is the commit point and changes no qualification, credit, training, GPU, model
selection, A7, next-phase, or scientific-claim state.

Every production entry fails before Git, report, prepared-directory, or runtime
I/O while either the exact4 ledger group or exact3 implementation group is
unknown/partial.  Once bound, it proves frozen GSE295 I2/B2 -> ledger L ->
runtime-sync I1/B1, including direct parents, exact changed paths and blobs,
clean HEAD/upstream/live-origin equality, and executing script/test bytes.
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
CONFIG_REPO_PATH = (
    "configs/route_a_v3_dec027_six_rescue_terminal_aggregate_evidence_runtime_sync_v1.json"
)
SCRIPT_REPO_PATH = (
    "scripts/route_a_v3/dec027_six_rescue_terminal_aggregate_evidence_runtime_sync.py"
)
TEST_REPO_PATH = (
    "tests/route_a_v3/test_dec027_six_rescue_terminal_aggregate_evidence_runtime_sync.py"
)
PRODUCTION_CONFIG_PATH = Path(__file__).resolve().parents[2] / CONFIG_REPO_PATH
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
LEDGER_PATHS = (
    "docs/execution/route_a_v3_a1_interim.yaml",
    "docs/execution/route_a_v3_registry_manifest.json",
    "scripts/route_a_v3/validate_a0_bundle.py",
    "tests/route_a_v3/test_a0_integrity_guards.py",
)
LINEAGE_IDS = (
    "gse217518_corrected_a1_successor_aggregate_preflight_v1",
    "encsr854ruf_dec027_dataset_specific_a1_preflight_corrected_b4_v1",
    "gse232572_corrected_a1_replay_aggregate_preflight_v1",
    "gse113849_designed_snv_true_a2_aggregate_preflight_v1",
    "gse269595_corrected_role_adjudication_successor_aggregate_recompute_v1",
    "gse295080_independence_overlap_aggregate_preflight_v1",
)
LEDGER_INTEGRATION_ID = (
    "DEC027_SIX_TERMINAL_AGGREGATE_RESCUE_REPORTS_V1_LEDGER_REGISTRATION"
)
LEDGER_MANIFEST_STATUS = (
    "DEC027_SIX_TERMINAL_AGGREGATE_RESCUE_REPORTS_REGISTERED_EVT059_SETTLED_"
    "PENDING_UNALLOCATED_EVT060_NO_PROMOTION_A1_INCOMPLETE_A6_IN_PROGRESS_"
    "L3_NOT_ESTABLISHED_A7_NOT_RUN"
)
TRUTH_SECTIONS = (
    "registered_artifacts",
    "runtime",
    "frozen_scientific_state",
    "evidence_registration_truth",
    "access_boundary",
    "publication_policy",
)
FaultInjector = Callable[[str], None]


class RuntimeSyncError(RuntimeError):
    """Base error for the DEC027 runtime sync."""


class BindingError(RuntimeSyncError):
    """A grouped ledger or implementation binding is not wholly bound."""


class AuthorityError(RuntimeSyncError):
    """The production repository does not prove the frozen lifecycle."""


class PredecessorError(RuntimeSyncError):
    """The runtime is not exact EVT059 or a legal EVT060 publication prefix."""


class PublicationError(RuntimeSyncError):
    """Prepared or runtime publication could not proceed exactly."""


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def json_line(value: Any) -> bytes:
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
        raise RuntimeSyncError(
            f"{label} drift: expected {expected!r}, observed {actual!r}"
        )


def _hex(value: Any, pattern: re.Pattern[str], *, label: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise RuntimeSyncError(f"{label} is not lowercase hexadecimal")
    return value


def _binding_values(binding: Mapping[str, Any]) -> list[Any]:
    return [
        binding.get("status"),
        binding.get("implementation_commit"),
        binding.get("implementation_script_sha256"),
        binding.get("implementation_test_sha256"),
    ]


def _ledger_values(ledger: Mapping[str, Any]) -> list[Any]:
    lineages = ledger.get("registered_lineage_ids", [])
    blobs = ledger.get("frozen_blobs", [])
    return [
        ledger.get("status"),
        ledger.get("commit"),
        ledger.get("integration_id"),
        ledger.get("manifest_status"),
        *(lineages if isinstance(lineages, list) else []),
        *(
            [item.get("sha256") for item in blobs if isinstance(item, Mapping)]
            if isinstance(blobs, list)
            else []
        ),
    ]


def _validate_binding_group(binding: Mapping[str, Any]) -> None:
    values = _binding_values(binding)
    if any(value == UNKNOWN for value in values):
        if values != [UNKNOWN] * 4:
            raise BindingError("implementation binding is partially known")
        return
    _expect(binding.get("status"), "BOUND", label="implementation status")
    _hex(binding.get("implementation_commit"), HEX40, label="implementation commit")
    _hex(
        binding.get("implementation_script_sha256"),
        HEX64,
        label="implementation script SHA",
    )
    _hex(
        binding.get("implementation_test_sha256"),
        HEX64,
        label="implementation test SHA",
    )


def _validate_ledger_group(ledger: Mapping[str, Any]) -> None:
    values = _ledger_values(ledger)
    if any(value == UNKNOWN for value in values):
        if values != [UNKNOWN] * 14:
            raise BindingError("predecessor ledger is partially known")
        return
    _expect(ledger.get("status"), "BOUND", label="ledger status")
    _hex(ledger.get("commit"), HEX40, label="ledger commit")
    _expect(ledger.get("integration_id"), LEDGER_INTEGRATION_ID, label="ledger ID")
    _expect(
        ledger.get("manifest_status"), LEDGER_MANIFEST_STATUS, label="ledger status text"
    )
    _expect(
        tuple(ledger.get("registered_lineage_ids", [])),
        LINEAGE_IDS,
        label="ledger lineage IDs",
    )
    blobs = ledger.get("frozen_blobs")
    if not isinstance(blobs, list) or [item.get("path") for item in blobs] != list(
        LEDGER_PATHS
    ):
        raise RuntimeSyncError("ledger exact4 path closure drift")
    for index, item in enumerate(blobs):
        _hex(item.get("sha256"), HEX64, label=f"ledger blob {index}")


def _science_fields(config: Mapping[str, Any]) -> dict[str, Any]:
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
        "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }


def validate_static_config(config: dict[str, Any]) -> None:
    _expect(
        config.get("schema_version"),
        "route_a_v3_dec027_six_rescue_terminal_aggregate_evidence_runtime_sync.v1",
        label="schema version",
    )
    _expect(config.get("decision_id"), "V3-DEC-027", label="decision")
    _expect(config.get("event_id"), "A1-EVT-060", label="event")
    _expect(
        config.get("event_name"),
        "DEC027_SIX_RESCUE_AGGREGATE_ONLY_TERMINAL_REPORTS_REGISTERED_QUALIFICATION_UNCHANGED",
        label="event name",
    )
    _expect(
        config.get("sync_type"),
        "APPEND_ONLY_PUBLIC_AGGREGATE_EVIDENCE_REGISTRATION_NO_SCIENTIFIC_STATE_CHANGE",
        label="sync type",
    )
    binding = config.get("implementation_binding")
    ledger = config.get("repository_authority", {}).get("predecessor_ledger")
    if not isinstance(binding, dict) or not isinstance(ledger, dict):
        raise RuntimeSyncError("binding groups are absent")
    _validate_binding_group(binding)
    _validate_ledger_group(ledger)
    _expect(
        binding.get("implementation_script_path"), SCRIPT_REPO_PATH, label="script path"
    )
    _expect(binding.get("implementation_test_path"), TEST_REPO_PATH, label="test path")
    repository = config["repository_authority"]
    _expect(
        repository.get("implementation_exact_changed_paths"),
        [CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH],
        label="I exact3 paths",
    )
    _expect(
        repository.get("binding_exact_changed_paths"),
        [CONFIG_REPO_PATH],
        label="B config-only path",
    )
    artifacts = config.get("registered_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 6:
        raise RuntimeSyncError("exactly six reports must be registered")
    _expect(
        tuple(item.get("lineage_id") for item in artifacts),
        LINEAGE_IDS,
        label="report lineage order",
    )
    paths: list[str] = []
    for index, item in enumerate(artifacts):
        if not isinstance(item, dict) or set(item) != {
            "lineage_id",
            "dataset_id",
            "name",
            "artifact_type",
            "absolute_path",
            "bytes",
            "sha256",
        }:
            raise RuntimeSyncError(f"report {index} metadata closure drift")
        path = Path(item["absolute_path"])
        if not path.is_absolute() or path.name != item["name"]:
            raise RuntimeSyncError(f"report {index} path/name drift")
        if type(item["bytes"]) is not int or item["bytes"] <= 0:
            raise RuntimeSyncError(f"report {index} byte count invalid")
        _hex(item["sha256"], HEX64, label=f"report {index} SHA")
        paths.append(item["absolute_path"])
    if len(set(paths)) != 6:
        raise RuntimeSyncError("report paths are not unique")
    runtime = config["runtime"]
    expected_runtime = {
        "predecessor_event_id": "A1-EVT-059",
        "predecessor_event_count": 59,
        "successor_event_id": "A1-EVT-060",
        "successor_event_count": 60,
        "predecessor_manifest_output_count": 256,
        "successor_manifest_output_count": 266,
        "predecessor_manifest_registered_artifact_count": 8,
        "successor_manifest_registered_artifact_count": 14,
        "output_delta_count": 10,
    }
    for key, expected in expected_runtime.items():
        _expect(runtime.get(key), expected, label=f"runtime.{key}")
    if set(runtime.get("predecessor_mutables", {})) != set(MUTABLE_NAMES):
        raise RuntimeSyncError("predecessor mutable closure drift")
    for name, spec in runtime["predecessor_mutables"].items():
        if type(spec.get("bytes")) is not int or spec["bytes"] <= 0:
            raise RuntimeSyncError(f"invalid predecessor bytes for {name}")
        _hex(spec.get("sha256"), HEX64, label=f"predecessor {name} SHA")
    tail = runtime["predecessor_tail"]
    _expect(tail.get("event_id"), "A1-EVT-059", label="tail event")
    _expect(tail.get("predecessor_event_id"), "A1-EVT-058", label="tail parent")
    _expect(tail.get("decision_id"), "V3-DEC-027", label="tail decision")
    _hex(tail.get("sha256"), HEX64, label="tail SHA")
    truth = config["evidence_registration_truth"]
    _expect(truth.get("report_count"), 6, label="truth report count")
    for key, expected in {
        "all_six_terminal_reports_registered": True,
        "stop_rule_evaluation_ready_after_commit": True,
        "stop_rule_evaluated_by_this_event": False,
        "conditional_successor_activated": False,
        "qualification_changed": False,
    }.items():
        _expect(truth.get(key), expected, label=f"truth.{key}")
    _expect(
        config["frozen_scientific_state"]["current_qualified_counts"],
        {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547},
        label="qualified counts",
    )
    _expect(
        config["frozen_scientific_state"]["six_report_contribution_delta"],
        {"ordinary": 0, "a1": 0, "true_a2": 0, "canonical_records": 0},
        label="six-report contribution",
    )
    boundary = config["access_boundary"]
    _expect(boundary.get("registered_dynamic_report_count"), 6, label="dynamic reports")
    _expect(
        boundary.get("registered_static_repository_leaf_count"), 0, label="static leaves"
    )
    _expect(boundary.get("registered_artifact_body_parse_count"), 0, label="body parse")


def validate_bound_config(config: dict[str, Any]) -> None:
    validate_static_config(config)
    if _binding_values(config["implementation_binding"]) == [UNKNOWN] * 4:
        raise BindingError("implementation binding is UNKNOWN_NOT_ASSERTED")
    if _ledger_values(config["repository_authority"]["predecessor_ledger"]) == [
        UNKNOWN
    ] * 14:
        raise BindingError("predecessor ledger is UNKNOWN_NOT_ASSERTED")


def load_config(path: Path = PRODUCTION_CONFIG_PATH, *, require_bound: bool = True) -> dict[str, Any]:
    try:
        config = load_json(path.read_bytes(), label=str(path))
    except OSError as exc:
        raise RuntimeSyncError(f"cannot read config: {path}") from exc
    (validate_bound_config if require_bound else validate_static_config)(config)
    return config


def expected_unknown_i_config(bound_config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(bound_config)
    binding = result["implementation_binding"]
    for key in (
        "status",
        "implementation_commit",
        "implementation_script_sha256",
        "implementation_test_sha256",
    ):
        binding[key] = UNKNOWN
    validate_static_config(result)
    return result


def truth_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(config[key]) for key in TRUTH_SECTIONS}


def _run_git(repo_root: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AuthorityError(f"Git authority command failed: {' '.join(args)}") from exc


def _commit_parent(repo_root: Path, commit: str) -> str:
    return _run_git(repo_root, "rev-parse", f"{commit}^").decode().strip()


def _changed_paths(repo_root: Path, commit: str) -> list[str]:
    payload = _run_git(
        repo_root, "diff-tree", "--no-commit-id", "--name-only", "-r", commit
    )
    return sorted(filter(None, payload.decode().splitlines()))


def _git_blob(repo_root: Path, commit: str, path: str) -> bytes:
    return _run_git(repo_root, "show", f"{commit}:{path}")


def _audit_commit(
    repo_root: Path,
    node: Mapping[str, Any],
    *,
    expected_paths: list[str],
    expected_parent: str,
) -> None:
    commit = node["commit"]
    if _commit_parent(repo_root, commit) != expected_parent:
        raise AuthorityError(f"parent drift at {commit}")
    if _changed_paths(repo_root, commit) != sorted(expected_paths):
        raise AuthorityError(f"changed-path closure drift at {commit}")
    blobs = node.get("blob_sha256_by_path")
    if isinstance(blobs, Mapping):
        for path, expected in blobs.items():
            if sha256(_git_blob(repo_root, commit, path)) != expected:
                raise AuthorityError(f"blob drift at {commit}:{path}")


def audit_production_repository_authority(config: dict[str, Any]) -> dict[str, str]:
    """Prove B2 -> exact4 L -> exact3 I1 -> config-only B1."""

    validate_bound_config(config)
    repository = config["repository_authority"]
    repo_root = Path(repository["production_repo_root"])
    binding = config["implementation_binding"]
    ledger = repository["predecessor_ledger"]
    base = repository["base_lifecycle"]
    base_i = base["implementation_i2"]
    base_b = base["binding_b2"]
    _audit_commit(
        repo_root,
        base_i,
        expected_paths=base_i["exact_changed_paths"],
        expected_parent=base_i["expected_parent"],
    )
    _audit_commit(
        repo_root,
        base_b,
        expected_paths=base_b["exact_changed_paths"],
        expected_parent=base_i["commit"],
    )
    if base_b["expected_parent"] != base_i["commit"]:
        raise AuthorityError("frozen base config parent declaration drift")
    ledger_commit = ledger["commit"]
    if _commit_parent(repo_root, ledger_commit) != base_b["commit"]:
        raise AuthorityError("ledger is not a direct child of GSE295 B2")
    if _changed_paths(repo_root, ledger_commit) != sorted(LEDGER_PATHS):
        raise AuthorityError("ledger exact4 changed-path closure drift")
    for item in ledger["frozen_blobs"]:
        if sha256(_git_blob(repo_root, ledger_commit, item["path"])) != item["sha256"]:
            raise AuthorityError(f"ledger blob drift: {item['path']}")
    implementation = binding["implementation_commit"]
    if _commit_parent(repo_root, implementation) != ledger_commit:
        raise AuthorityError("I1 is not a direct child of ledger L")
    if _changed_paths(repo_root, implementation) != sorted(
        repository["implementation_exact_changed_paths"]
    ):
        raise AuthorityError("I1 exact3 changed-path closure drift")
    i_config_payload = _git_blob(repo_root, implementation, CONFIG_REPO_PATH)
    i_config = load_json(i_config_payload, label="I1 config blob")
    if i_config != expected_unknown_i_config(config):
        raise AuthorityError("I1 config is not the exact clean UNKNOWN precursor of B1")
    script_payload = _git_blob(repo_root, implementation, SCRIPT_REPO_PATH)
    test_payload = _git_blob(repo_root, implementation, TEST_REPO_PATH)
    if sha256(script_payload) != binding["implementation_script_sha256"]:
        raise AuthorityError("I1 script blob does not match binding")
    if sha256(test_payload) != binding["implementation_test_sha256"]:
        raise AuthorityError("I1 test blob does not match binding")
    head = _run_git(repo_root, "rev-parse", "HEAD").decode().strip()
    if head != binding["implementation_commit"] and _commit_parent(repo_root, head) != implementation:
        raise AuthorityError("HEAD is neither legal I1 nor direct B1")
    if head == implementation:
        raise AuthorityError("production requires terminal config-only B1, not I1")
    if _changed_paths(repo_root, head) != repository["binding_exact_changed_paths"]:
        raise AuthorityError("B1 is not config-only")
    head_config = load_json(_git_blob(repo_root, head, CONFIG_REPO_PATH), label="B1 config")
    if head_config != config:
        raise AuthorityError("B1 config blob differs from executing config")
    if _git_blob(repo_root, head, SCRIPT_REPO_PATH) != script_payload:
        raise AuthorityError("B1 script differs from I1")
    if _git_blob(repo_root, head, TEST_REPO_PATH) != test_payload:
        raise AuthorityError("B1 test differs from I1")
    if _run_git(repo_root, "status", "--porcelain").strip():
        raise AuthorityError("production worktree or index is dirty")
    branch = _run_git(repo_root, "symbolic-ref", "--short", "HEAD").decode().strip()
    if branch != repository["branch"]:
        raise AuthorityError("production branch drift")
    upstream = _run_git(repo_root, "rev-parse", "@{u}").decode().strip()
    if upstream != head:
        raise AuthorityError("HEAD differs from upstream")
    remote = _run_git(
        repo_root, "ls-remote", "--heads", "origin", repository["branch"]
    ).decode().split()
    if not remote or remote[0] != head:
        raise AuthorityError("HEAD differs from live origin")
    try:
        disk_script = (repo_root / SCRIPT_REPO_PATH).read_bytes()
        disk_test = (repo_root / TEST_REPO_PATH).read_bytes()
        disk_config = (repo_root / CONFIG_REPO_PATH).read_bytes()
    except OSError as exc:
        raise AuthorityError("cannot read executing exact3 bytes") from exc
    if disk_script != script_payload or disk_test != test_payload:
        raise AuthorityError("executing script/test bytes differ from frozen I1")
    if load_json(disk_config, label="executing config") != config:
        raise AuthorityError("executing config differs from B1")
    return {"base_b2": base_b["commit"], "ledger": ledger_commit, "i1": implementation, "b1": head}


def validate_registered_artifacts(
    config: dict[str, Any], *, verify_exact_bytes: bool
) -> dict[str, Any]:
    (validate_bound_config if verify_exact_bytes else validate_static_config)(config)
    if verify_exact_bytes:
        for item in config["registered_artifacts"]:
            path = Path(item["absolute_path"])
            try:
                payload = path.read_bytes()
            except OSError as exc:
                raise PublicationError(f"cannot read report bytes: {path}") from exc
            if len(payload) != item["bytes"] or sha256(payload) != item["sha256"]:
                raise PublicationError(f"registered report identity drift: {path}")
    return {
        "artifact_count": 6,
        "exact_byte_validation_count": 6 if verify_exact_bytes else 0,
        "body_parse_count": 0,
        "payload_field_read_count": 0,
        "registered_artifacts_copied": False,
    }


def _parse_runtime(
    payloads: Mapping[str, bytes]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    return (
        load_json(payloads["STATUS.json"], label="STATUS.json"),
        load_json(payloads["RUN_MANIFEST.json"], label="RUN_MANIFEST.json"),
        load_events(payloads["EVENT_LOG.jsonl"], label="EVENT_LOG.jsonl"),
    )


def _identity(payload: bytes, spec: Mapping[str, Any], *, label: str) -> None:
    if len(payload) != spec["bytes"] or sha256(payload) != spec["sha256"]:
        raise PredecessorError(f"{label} predecessor identity drift")


def _snapshot_names(config: Mapping[str, Any]) -> dict[str, str]:
    return {
        name: config["runtime"]["predecessor_mutables"][name]["snapshot_name"]
        for name in MUTABLE_NAMES
    }


def validate_predecessor(
    config: dict[str, Any], payloads: Mapping[str, bytes]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    validate_bound_config(config)
    if set(payloads) != set(MUTABLE_NAMES):
        raise PredecessorError("runtime mutable member closure drift")
    runtime = config["runtime"]
    for name in MUTABLE_NAMES:
        _identity(payloads[name], runtime["predecessor_mutables"][name], label=name)
    status, manifest, events = _parse_runtime(payloads)
    if len(events) != 59:
        raise PredecessorError("predecessor event count is not 59")
    _expect(
        [event.get("event_id") for event in events],
        [f"A1-EVT-{index:03d}" for index in range(1, 60)],
        label="predecessor event sequence",
    )
    tail = events[-1]
    _expect(tail.get("event_id"), "A1-EVT-059", label="tail event")
    _expect(tail.get("predecessor_event_id"), "A1-EVT-058", label="tail parent")
    _expect(tail.get("decision_id"), "V3-DEC-027", label="tail decision")
    tail_payload = payloads["EVENT_LOG.jsonl"].splitlines(keepends=True)[-1]
    _identity(tail_payload, runtime["predecessor_tail"], label="event tail")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 256:
        raise PredecessorError("predecessor manifest output count is not 256")
    output_paths = [item.get("absolute_path") for item in outputs if isinstance(item, dict)]
    if len(output_paths) != 256 or len(set(output_paths)) != 256:
        raise PredecessorError("predecessor output paths are not unique")
    _expect(manifest.get("registered_artifact_count"), 8, label="registered count")
    for document, label in ((status, "STATUS"), (manifest, "RUN_MANIFEST")):
        for key, expected in _science_fields(config).items():
            _expect(document.get(key), expected, label=f"{label}.{key}")
        _expect(
            document.get("dec027_authority_runtime_sync_status"),
            "SYNCED_EVT_059",
            label=f"{label}.dec027 authority sync",
        )
    return status, manifest, events


def _validate_timestamp(recorded_at: str, predecessor_at: Any) -> None:
    if not isinstance(predecessor_at, str):
        raise PredecessorError("predecessor timestamp is absent")
    try:
        current = datetime.fromisoformat(recorded_at)
        previous = datetime.fromisoformat(predecessor_at)
    except (TypeError, ValueError) as exc:
        raise PredecessorError("timestamps must be ISO-8601") from exc
    if current.tzinfo is None or previous.tzinfo is None or current <= previous:
        raise PredecessorError("EVT060 timestamp must follow EVT059 with an offset")


def _registration_markers() -> dict[str, Any]:
    return {
        "dec027_six_terminal_aggregate_reports_runtime_sync_status": "SYNCED_EVT_060",
        "all_six_terminal_reports_registered": True,
        "stop_rule_evaluation_ready_after_commit": True,
        "stop_rule_evaluated_by_this_event": False,
        "conditional_successor_activated": False,
    }


def _sync_record(
    config: dict[str, Any], *, recorded_at: str, snapshots: Mapping[str, bytes]
) -> bytes:
    return json_bytes(
        {
            "record_type": "ROUTE_A_V3_A1_DEC027_SIX_RESCUE_TERMINAL_AGGREGATE_EVIDENCE_RUNTIME_SYNC",
            "event_id": "A1-EVT-060",
            "decision_id": "V3-DEC-027",
            "recorded_at": recorded_at,
            "predecessor_event_id": "A1-EVT-059",
            "sync_type": config["sync_type"],
            "registered_artifact_count": 6,
            "registered_artifacts": copy.deepcopy(config["registered_artifacts"]),
            "registered_static_repository_leaf_count": 0,
            "registered_artifact_exact_byte_validation_count": 6,
            "registered_artifact_body_parse_count": 0,
            "registered_artifact_payload_field_read_count": 0,
            "registered_artifacts_copied": False,
            "predecessor_snapshot_count": 3,
            "predecessor_snapshot_names": list(snapshots),
            "predecessor_snapshot_sha256": {
                name: sha256(payload) for name, payload in snapshots.items()
            },
            "output_delta_count": 10,
            "successor_manifest_output_count": 266,
            "successor_manifest_registered_artifact_count": 14,
            **_registration_markers(),
            "frozen_scientific_state": copy.deepcopy(config["frozen_scientific_state"]),
            "evidence_registration_truth": copy.deepcopy(
                config["evidence_registration_truth"]
            ),
            "access_boundary": copy.deepcopy(config["access_boundary"]),
            "scientific_state_changed": False,
            "evidence_surface_changed": True,
            "evidence_gate_statuses_changed": False,
            "overall_qualification_gate_changed": False,
            "qualification_changed": False,
        }
    )


def _event(config: dict[str, Any], *, recorded_at: str, sync_sha: str) -> dict[str, Any]:
    return {
        "event_id": "A1-EVT-060",
        "at": recorded_at,
        "phase_id": "A1",
        "event": config["event_name"],
        "sync_type": config["sync_type"],
        "decision_id": "V3-DEC-027",
        "predecessor_event_id": "A1-EVT-059",
        "registered_artifact_count": 6,
        "registered_lineage_ids": list(LINEAGE_IDS),
        "registered_static_repository_leaf_count": 0,
        "registered_artifacts_copied": False,
        "registered_artifact_exact_byte_validation_count": 6,
        "registered_artifact_body_parse_count": 0,
        "registered_artifact_payload_field_read_count": 0,
        "predecessor_snapshot_count": 3,
        "predecessor_snapshot_names": list(_snapshot_names(config).values()),
        "sync_name": config["runtime"]["sync_name"],
        "sync_record_sha256": sync_sha,
        "output_delta_count": 10,
        "manifest_output_count_before": 256,
        "manifest_output_count_after": 266,
        "manifest_registered_artifact_count_before": 8,
        "manifest_registered_artifact_count_after": 14,
        **_registration_markers(),
        "frozen_scientific_state": copy.deepcopy(config["frozen_scientific_state"]),
        "evidence_registration_truth": copy.deepcopy(
            config["evidence_registration_truth"]
        ),
        "access_boundary": copy.deepcopy(config["access_boundary"]),
        "scientific_state_changed": False,
        "evidence_surface_changed": True,
        "evidence_gate_statuses_changed": False,
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
            "Registered six in-place DEC027 terminal aggregate-only reports after "
            "exact-byte identity validation without parsing or copying their bodies. "
            "This event makes the separate stop-rule adjudication ready after commit "
            "but does not evaluate it or activate a successor. Qualification remains "
            "1 ordinary, 1 A1, 0 true-A2 and 6547 canonical records; all execution and "
            "scientific-claim locks remain unchanged."
        ),
    }


def _output_tail(
    config: dict[str, Any],
    predecessor: Mapping[str, bytes],
    sync_payload: bytes,
) -> list[dict[str, Any]]:
    tail = copy.deepcopy(config["registered_artifacts"])
    run_root = Path(config["runtime"]["run_root"])
    names = _snapshot_names(config)
    for mutable in MUTABLE_NAMES:
        payload = predecessor[mutable]
        tail.append(
            {
                "absolute_path": str(run_root / names[mutable]),
                "artifact_type": "PREDECESSOR_RUNTIME_SNAPSHOT",
                "bytes": len(payload),
                "sha256": sha256(payload),
                "status": "COMPLETE",
            }
        )
    tail.append(
        {
            "absolute_path": str(run_root / config["runtime"]["sync_name"]),
            "artifact_type": "RUNTIME_SYNC_RECORD",
            "bytes": len(sync_payload),
            "sha256": sha256(sync_payload),
            "status": "COMPLETE",
        }
    )
    return tail


def build_successors(
    config: dict[str, Any], predecessor: Mapping[str, bytes], recorded_at: str
) -> dict[str, bytes]:
    status, manifest, events = validate_predecessor(config, predecessor)
    _validate_timestamp(recorded_at, events[-1].get("at"))
    snapshot_names = _snapshot_names(config)
    snapshots = {
        snapshot_names[name]: predecessor[name] for name in MUTABLE_NAMES
    }
    sync_payload = _sync_record(config, recorded_at=recorded_at, snapshots=snapshots)
    successor_status = copy.deepcopy(status)
    successor_status["updated_at"] = recorded_at
    successor_status.update(_registration_markers())
    successor_manifest = copy.deepcopy(manifest)
    successor_manifest["updated_at"] = recorded_at
    successor_manifest.update(_registration_markers())
    successor_manifest["outputs"] = manifest["outputs"] + _output_tail(
        config, predecessor, sync_payload
    )
    successor_manifest["registered_artifact_count"] = 14
    event = _event(config, recorded_at=recorded_at, sync_sha=sha256(sync_payload))
    result = {
        **snapshots,
        config["runtime"]["sync_name"]: sync_payload,
        "STATUS.json": json_bytes(successor_status),
        "RUN_MANIFEST.json": json_bytes(successor_manifest),
        "EVENT_LOG.jsonl": predecessor["EVENT_LOG.jsonl"] + json_line(event),
    }
    validate_successors(config, predecessor, result)
    return result


def validate_successors(
    config: dict[str, Any],
    predecessor: Mapping[str, bytes],
    successors: Mapping[str, bytes],
) -> None:
    old_status, old_manifest, old_events = validate_predecessor(config, predecessor)
    expected_names = set(_snapshot_names(config).values()) | {
        config["runtime"]["sync_name"],
        *MUTABLE_NAMES,
    }
    if set(successors) != expected_names:
        raise RuntimeSyncError("successor member closure drift")
    status, manifest, events = _parse_runtime(successors)
    if len(events) != 60 or events[:-1] != old_events:
        raise RuntimeSyncError("EVT060 is not one append-only event")
    event = events[-1]
    for key, expected in {
        "event_id": "A1-EVT-060",
        "predecessor_event_id": "A1-EVT-059",
        "decision_id": "V3-DEC-027",
        "scientific_state_changed": False,
        "evidence_surface_changed": True,
        "evidence_gate_statuses_changed": False,
        "qualification_changed": False,
        **_registration_markers(),
    }.items():
        _expect(event.get(key), expected, label=f"event.{key}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 266:
        raise RuntimeSyncError("successor manifest output count is not 266")
    if outputs[:256] != old_manifest["outputs"]:
        raise RuntimeSyncError("predecessor output prefix drift")
    if outputs[256:262] != config["registered_artifacts"]:
        raise RuntimeSyncError("six-report metadata/order drift")
    expected_tail = list(_snapshot_names(config).values()) + [
        config["runtime"]["sync_name"]
    ]
    if [Path(item.get("absolute_path", "")).name for item in outputs[-4:]] != expected_tail:
        raise RuntimeSyncError("snapshot/sync output order drift")
    if len({item.get("absolute_path") for item in outputs}) != 266:
        raise RuntimeSyncError("successor output paths are not unique")
    expected_status = copy.deepcopy(old_status)
    expected_status["updated_at"] = event["at"]
    expected_status.update(_registration_markers())
    _expect(status, expected_status, label="STATUS preservation")
    expected_manifest = copy.deepcopy(old_manifest)
    expected_manifest["updated_at"] = event["at"]
    expected_manifest.update(_registration_markers())
    expected_manifest["outputs"] = outputs
    expected_manifest["registered_artifact_count"] = 14
    _expect(manifest, expected_manifest, label="RUN_MANIFEST preservation")
    for mutable, snapshot in _snapshot_names(config).items():
        _expect(successors[snapshot], predecessor[mutable], label=f"snapshot {mutable}")
    sync = load_json(successors[config["runtime"]["sync_name"]], label="sync")
    for key, expected in {
        "event_id": "A1-EVT-060",
        "registered_artifact_count": 6,
        "registered_static_repository_leaf_count": 0,
        "registered_artifact_body_parse_count": 0,
        "output_delta_count": 10,
        "scientific_state_changed": False,
        "evidence_gate_statuses_changed": False,
        "qualification_changed": False,
        **_registration_markers(),
    }.items():
        _expect(sync.get(key), expected, label=f"sync.{key}")
    for document, label in ((status, "STATUS"), (manifest, "RUN_MANIFEST")):
        for key, expected in _science_fields(config).items():
            _expect(document.get(key), expected, label=f"{label}.{key}")


def _absolute(path: Path | str) -> Path:
    result = Path(path)
    if not result.is_absolute():
        raise PublicationError(f"path must be absolute: {result}")
    return result


def _prepared_path(config: Mapping[str, Any], path: Path | str) -> Path:
    prepared = _absolute(path)
    allowed = _absolute(config["runtime"]["allowed_prepared_root"])
    try:
        prepared.relative_to(allowed)
    except ValueError as exc:
        raise PublicationError("prepared directory is outside allowed root") from exc
    if prepared == allowed:
        raise PublicationError("prepared directory must be a child of allowed root")
    return prepared


def _read_runtime(run_root: Path) -> dict[str, bytes]:
    try:
        return {name: (run_root / name).read_bytes() for name in MUTABLE_NAMES}
    except OSError as exc:
        raise PublicationError("cannot read runtime mutables") from exc


@contextmanager
def _run_lock(run_root: Path) -> Iterator[None]:
    lock_path = run_root / ".dec027_six_report_runtime_sync.lock"
    try:
        with lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
    except OSError as exc:
        raise PublicationError("cannot lock runtime") from exc


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError as exc:
        raise PublicationError(f"atomic write failed: {path}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _write_immutable_once(path: Path, payload: bytes) -> None:
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise PublicationError(f"cannot read immutable: {path}") from exc
        if existing != payload:
            raise PublicationError(f"immutable path contains different bytes: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise PublicationError(f"immutable race produced different bytes: {path}")
        temporary.unlink()
        temporary = None
    except (OSError, PublicationError) as exc:
        if isinstance(exc, PublicationError):
            raise
        raise PublicationError(f"immutable publication failed: {path}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _prepared_names(config: Mapping[str, Any]) -> list[str]:
    return [
        *_snapshot_names(config).values(),
        config["runtime"]["sync_name"],
        *MUTABLE_NAMES,
    ]


def _write_prepared(prepared: Path, successors: Mapping[str, bytes]) -> None:
    prepared.mkdir(parents=True, exist_ok=True)
    extras = {path.name for path in prepared.iterdir()} - set(successors)
    if extras:
        raise PublicationError("prepared directory contains unexpected members")
    for name in successors:
        _write_immutable_once(prepared / name, successors[name])


def _read_prepared(config: Mapping[str, Any], prepared: Path) -> dict[str, bytes]:
    expected = set(_prepared_names(config))
    try:
        observed = {path.name for path in prepared.iterdir()}
    except OSError as exc:
        raise PublicationError("cannot read prepared directory") from exc
    if observed != expected:
        raise PublicationError("prepared member closure drift")
    try:
        return {name: (prepared / name).read_bytes() for name in expected}
    except OSError as exc:
        raise PublicationError("cannot read prepared members") from exc


def _publication_prefix(
    predecessor: Mapping[str, bytes], successors: Mapping[str, bytes], current: Mapping[str, bytes]
) -> int:
    states = []
    for count in range(4):
        state = dict(predecessor)
        for name in MUTABLE_NAMES[:count]:
            state[name] = successors[name]
        states.append(state)
    for index, state in enumerate(states):
        if dict(current) == state:
            return index
    raise PredecessorError("runtime is not an exact EVT059/EVT060 publication prefix")


def _production_preflight(config: dict[str, Any]) -> None:
    validate_bound_config(config)
    audit_production_repository_authority(config)
    validate_registered_artifacts(config, verify_exact_bytes=True)


def prepare_runtime_sync(
    config: dict[str, Any], *, recorded_at: str, prepared_directory: Path | str
) -> dict[str, bytes]:
    _production_preflight(config)
    prepared = _prepared_path(config, prepared_directory)
    run_root = _absolute(config["runtime"]["run_root"])
    with _run_lock(run_root):
        predecessor = _read_runtime(run_root)
        successors = build_successors(config, predecessor, recorded_at)
        _write_prepared(prepared, successors)
    return successors


def publish_prepared(
    config: dict[str, Any],
    *,
    prepared_directory: Path | str,
    fault_injector: FaultInjector | None = None,
) -> None:
    _production_preflight(config)
    prepared = _prepared_path(config, prepared_directory)
    successors = _read_prepared(config, prepared)
    run_root = _absolute(config["runtime"]["run_root"])
    snapshots = set(_snapshot_names(config).values()) | {config["runtime"]["sync_name"]}
    with _run_lock(run_root):
        current = _read_runtime(run_root)
        predecessor = {
            mutable: successors[_snapshot_names(config)[mutable]] for mutable in MUTABLE_NAMES
        }
        validate_successors(config, predecessor, successors)
        prefix = _publication_prefix(predecessor, successors, current)
        for name in _prepared_names(config):
            if name in snapshots:
                _write_immutable_once(run_root / name, successors[name])
                if fault_injector:
                    fault_injector(f"immutable:{name}")
        for index, name in enumerate(MUTABLE_NAMES):
            if index < prefix:
                continue
            _write_atomic(run_root / name, successors[name])
            if fault_injector:
                fault_injector(f"mutable:{name}")
    validate_published(config, prepared_directory=prepared)


def validate_published(
    config: dict[str, Any], *, prepared_directory: Path | str
) -> None:
    _production_preflight(config)
    prepared = _prepared_path(config, prepared_directory)
    successors = _read_prepared(config, prepared)
    predecessor = {
        mutable: successors[_snapshot_names(config)[mutable]] for mutable in MUTABLE_NAMES
    }
    validate_successors(config, predecessor, successors)
    run_root = _absolute(config["runtime"]["run_root"])
    with _run_lock(run_root):
        current = _read_runtime(run_root)
        if current != {name: successors[name] for name in MUTABLE_NAMES}:
            raise PublicationError("published runtime differs from prepared EVT060")
        for name in [*_snapshot_names(config).values(), config["runtime"]["sync_name"]]:
            try:
                payload = (run_root / name).read_bytes()
            except OSError as exc:
                raise PublicationError(f"cannot read published immutable: {name}") from exc
            if payload != successors[name]:
                raise PublicationError(f"published immutable differs: {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "publish", "validate"))
    parser.add_argument("--config", type=Path, default=PRODUCTION_CONFIG_PATH)
    parser.add_argument("--prepared-directory", type=Path, required=True)
    parser.add_argument("--recorded-at")
    args = parser.parse_args(argv)
    config = load_config(args.config, require_bound=True)
    if args.action == "prepare":
        if not args.recorded_at:
            parser.error("prepare requires --recorded-at")
        prepare_runtime_sync(
            config,
            recorded_at=args.recorded_at,
            prepared_directory=args.prepared_directory,
        )
    elif args.action == "publish":
        if args.recorded_at:
            parser.error("publish does not accept --recorded-at")
        publish_prepared(config, prepared_directory=args.prepared_directory)
    else:
        if args.recorded_at:
            parser.error("validate does not accept --recorded-at")
        validate_published(config, prepared_directory=args.prepared_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
