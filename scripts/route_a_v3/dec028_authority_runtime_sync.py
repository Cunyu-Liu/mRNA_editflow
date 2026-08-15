#!/usr/bin/env python3
"""Prepare and publish the future DEC028 authority-only runtime sync.

The checked-in configuration is deliberately static-only.  The public CLI
requires a later owner-issued, independently reviewed, repository-bound
configuration before it will read a runtime file or write a prepared directory.
The pure builder and non-production parameters exist only for synthetic tests.
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
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_CONFIG_PATH = REPOSITORY_ROOT / "configs/route_a_v3_dec028_authority_runtime_sync_v1.json"
MUTABLE_NAMES = ("STATUS.json", "RUN_MANIFEST.json", "EVENT_LOG.jsonl")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

BEFORE_DECISION_IDS = (
    "V3-DEC-017",
    "V3-DEC-018",
    "V3-DEC-019",
    "V3-DEC-020",
    "V3-DEC-021",
    "V3-DEC-022",
    "V3-DEC-023",
    "V3-DEC-024",
    "V3-DEC-027",
)
AFTER_DECISION_IDS = BEFORE_DECISION_IDS + ("V3-DEC-028",)
FROZEN_COUNTS = {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547}
OUTER_LOCKS = {
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


class RuntimeSyncError(RuntimeError):
    """Base failure for this fail-closed publisher."""


class AuthorityError(RuntimeSyncError):
    """The owner, review, or repository binding is not sufficient."""


class PredecessorError(RuntimeSyncError):
    """The live predecessor is not a legal append-only source."""


class PublicationError(RuntimeSyncError):
    """Prepared or live publication state is not recoverably exact."""


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def json_line(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
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
    except (UnicodeDecodeError, json.JSONDecodeError, RuntimeSyncError) as exc:
        raise RuntimeSyncError(f"{label} is not a unique-key JSON object") from exc
    if not isinstance(value, dict):
        raise RuntimeSyncError(f"{label} must be a JSON object")
    return value


def load_events(payload: bytes, *, label: str) -> list[dict[str, Any]]:
    if not payload or not payload.endswith(b"\n"):
        raise RuntimeSyncError(f"{label} must be nonempty newline-delimited JSON")
    events = [load_json(line, label=label) for line in payload.splitlines()]
    if not events:
        raise RuntimeSyncError(f"{label} has no events")
    return events


def _expect(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected or type(actual) is not type(expected):
        raise RuntimeSyncError(f"{label} differs from frozen authority")


def _expect_keys(value: Any, keys: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise RuntimeSyncError(f"{label} key closure differs")
    return value


def _hex(value: Any, *, label: str, width: int = 64) -> str:
    pattern = HEX64 if width == 64 else HEX40
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise AuthorityError(f"{label} is not an exact hexadecimal identity")
    return value


def _iso_after(current_at: str, predecessor_at: Any) -> None:
    if not isinstance(current_at, str) or not isinstance(predecessor_at, str):
        raise PredecessorError("timestamps must be explicit ISO-8601 strings")
    try:
        current = datetime.fromisoformat(current_at)
        predecessor = datetime.fromisoformat(predecessor_at)
    except ValueError as exc:
        raise PredecessorError("timestamp is not ISO-8601") from exc
    if current.tzinfo is None or predecessor.tzinfo is None or current <= predecessor:
        raise PredecessorError("successor timestamp must follow the live predecessor")


def _unknown(value: Any) -> bool:
    return value == "UNKNOWN_NOT_ASSERTED"


def _runtime(config: Mapping[str, Any]) -> dict[str, Any]:
    runtime = config.get("runtime")
    if not isinstance(runtime, dict):
        raise RuntimeSyncError("runtime configuration is absent")
    return runtime


def _snapshot_names(config: Mapping[str, Any]) -> dict[str, str]:
    snapshots = _runtime(config)["snapshot_names"]
    if not isinstance(snapshots, dict) or set(snapshots) != set(MUTABLE_NAMES):
        raise RuntimeSyncError("runtime snapshot map is not closed")
    if not all(isinstance(name, str) and name for name in snapshots.values()):
        raise RuntimeSyncError("runtime snapshot name is invalid")
    if len(set(snapshots.values())) != len(snapshots):
        raise RuntimeSyncError("runtime snapshot names are not unique")
    return dict(snapshots)


def expected_member_names(config: Mapping[str, Any]) -> set[str]:
    runtime = _runtime(config)
    return set(MUTABLE_NAMES) | set(_snapshot_names(config).values()) | {runtime["sync_name"]}


def validate_static_config(config: Mapping[str, Any]) -> None:
    """Validate only static authority; this function performs no runtime I/O."""

    _expect(config.get("schema_version"), "route_a_v3_dec028_authority_runtime_sync.v1", label="schema")
    _expect(config.get("protocol_id"), "ROUTE_A_V3_DEC028_AUTHORITY_RUNTIME_SYNC_V1", label="protocol")
    _expect(config.get("contract_id"), "mrna_xeditflow_route_a_v3", label="contract")
    _expect(config.get("phase_id"), "A1", label="phase")
    _expect(config.get("decision_id"), "V3-DEC-028", label="decision")
    _expect(
        config.get("sync_type"),
        "APPEND_ONLY_AUTHORITY_ONLY_RUNTIME_REGISTRATION_NO_SCIENTIFIC_STATE_CHANGE",
        label="sync type",
    )

    candidate = _expect_keys(
        config.get("candidate_contract"), {"path", "sha256", "bytes"}, label="candidate contract"
    )
    _expect(
        candidate["path"],
        "docs/contracts/candidates/mrna_xeditflow_route_a_v3_dec028_single_study_mainline_contract_v1.md",
        label="candidate path",
    )
    _hex(candidate["sha256"], label="candidate hash")
    if not isinstance(candidate["bytes"], int) or candidate["bytes"] <= 0:
        raise RuntimeSyncError("candidate byte count is invalid")

    amendment = _expect_keys(
        config.get("amendment"), {"path", "sha256", "status"}, label="amendment"
    )
    _expect(amendment["path"], "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec028.yaml", label="amendment path")
    _hex(amendment["sha256"], label="amendment hash")
    _expect(
        amendment["status"],
        "FROZEN_OWNER_SELECTED_PENDING_FRESH_RUNTIME_AUTHORITY_SYNC",
        label="amendment status",
    )

    static = _expect_keys(
        config.get("static_authority"),
        {
            "status",
            "effective_active_amendment_decision_ids_before_sync",
            "active_amendment_decision_ids_after_sync",
            "current_qualified_counts",
            "scientific_claim_status",
            "next_operational_scope",
            "locks",
        },
        label="static authority",
    )
    _expect(static["status"], "PENDING_STATIC_AUTHORITY_ONLY", label="static authority status")
    _expect(tuple(static["effective_active_amendment_decision_ids_before_sync"]), BEFORE_DECISION_IDS, label="pre-sync decisions")
    _expect(tuple(static["active_amendment_decision_ids_after_sync"]), AFTER_DECISION_IDS, label="post-sync decisions")
    _expect(static["current_qualified_counts"], FROZEN_COUNTS, label="qualified counts")
    _expect(static["scientific_claim_status"], "NOT_ESTABLISHED", label="scientific claim")
    _expect(static["next_operational_scope"], "SINGLE_STUDY_S0_AUTHORITY_AND_P0_CLOSURE", label="next scope")
    locks = static["locks"]
    expected_locks = {
        "data_row_access_allowed": False,
        "materialization_allowed": False,
        "cuda_probe_allowed": False,
        "model_or_optimizer_construction_allowed": False,
        "checkpoint_read_or_write_allowed": False,
        "parameter_update_or_training_allowed": False,
        "model_selection_allowed": False,
        "g1_launched": False,
        "a7_allowed": False,
        "sealed_access_allowed": False,
        "p0_production_authorized": False,
    }
    _expect(locks, expected_locks, label="static locks")

    owner = _expect_keys(
        config.get("owner_activation"),
        {
            "status",
            "decision_issuance_reference",
            "issued_at",
            "independent_review_status",
            "independent_review_reference",
        },
        label="owner activation",
    )
    if owner["status"] not in {"OWNER_ISSUANCE_REQUIRED", "SYNTHETIC_TEST_ONLY", "BOUND"}:
        raise AuthorityError("owner activation status is invalid")
    if owner["status"] == "OWNER_ISSUANCE_REQUIRED":
        _expect(owner["decision_issuance_reference"], "UNKNOWN_NOT_ASSERTED", label="owner issuance")
        _expect(owner["issued_at"], "UNKNOWN_NOT_ASSERTED", label="owner issue time")
        _expect(owner["independent_review_status"], "PENDING_DISTINCT_REVIEW", label="review status")
        _expect(owner["independent_review_reference"], "UNKNOWN_NOT_ASSERTED", label="review reference")

    implementation = _expect_keys(
        config.get("implementation_binding"),
        {
            "status",
            "implementation_commit",
            "implementation_script_path",
            "implementation_script_sha256",
            "implementation_test_path",
            "implementation_test_sha256",
        },
        label="implementation binding",
    )
    if implementation["status"] not in {"UNKNOWN_NOT_ASSERTED", "SYNTHETIC_TEST_ONLY", "BOUND"}:
        raise AuthorityError("implementation binding status is invalid")
    _expect(
        implementation["implementation_script_path"],
        "scripts/route_a_v3/dec028_authority_runtime_sync.py",
        label="implementation script path",
    )
    _expect(
        implementation["implementation_test_path"],
        "tests/route_a_v3/test_dec028_authority_runtime_sync.py",
        label="implementation test path",
    )
    if implementation["status"] == "UNKNOWN_NOT_ASSERTED":
        for key in ("implementation_commit", "implementation_script_sha256", "implementation_test_sha256"):
            _expect(implementation[key], "UNKNOWN_NOT_ASSERTED", label=f"implementation {key}")

    repository = _expect_keys(
        config.get("production_repository"),
        {"status", "repository_root", "branch", "expected_head_commit"},
        label="production repository",
    )
    if repository["status"] not in {"UNKNOWN_NOT_ASSERTED", "SYNTHETIC_TEST_ONLY", "BOUND"}:
        raise AuthorityError("production repository status is invalid")
    if repository["status"] == "UNKNOWN_NOT_ASSERTED":
        for key in ("repository_root", "branch", "expected_head_commit"):
            _expect(repository[key], "UNKNOWN_NOT_ASSERTED", label=f"repository {key}")

    runtime = _runtime(config)
    required_runtime = {
        "run_root",
        "allowed_prepared_root",
        "predecessor_decision_id",
        "event_id_prefix",
        "event_id_width",
        "sync_name",
        "snapshot_names",
        "immutable_publish_order",
        "mutable_publish_order",
    }
    _expect_keys(runtime, required_runtime, label="runtime")
    _expect(runtime["predecessor_decision_id"], "V3-DEC-027", label="predecessor decision")
    _expect(runtime["event_id_prefix"], "A1-EVT-", label="event prefix")
    _expect(runtime["event_id_width"], 3, label="event id width")
    if not all(isinstance(runtime[key], str) and runtime[key] for key in ("run_root", "allowed_prepared_root", "sync_name")):
        raise RuntimeSyncError("runtime path or sync name is invalid")
    snapshots = _snapshot_names(config)
    immutable = list(snapshots.values()) + [runtime["sync_name"]]
    _expect(runtime["immutable_publish_order"], immutable, label="immutable publish order")
    _expect(runtime["mutable_publish_order"], list(MUTABLE_NAMES), label="mutable publish order")
    if len(expected_member_names(config)) != 7:
        raise RuntimeSyncError("prepared member closure must be seven")


def load_config(path: Path = PRODUCTION_CONFIG_PATH) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise RuntimeSyncError(f"cannot read configuration: {path}") from exc
    config = load_json(payload, label="DEC028 runtime-sync configuration")
    validate_static_config(config)
    if path.resolve() == PRODUCTION_CONFIG_PATH.resolve():
        for item in (config["candidate_contract"], config["amendment"]):
            target = REPOSITORY_ROOT / item["path"]
            try:
                observed = target.read_bytes()
            except OSError as exc:
                raise AuthorityError(f"cannot read bound static authority: {item['path']}") from exc
            if len(observed) != item.get("bytes", len(observed)) and "bytes" in item:
                raise AuthorityError("candidate contract byte identity differs")
            if sha256(observed) != item["sha256"]:
                raise AuthorityError("bound static authority hash differs")
    return config


def _run_git(repo_root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), *arguments], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AuthorityError("cannot audit the production repository") from exc


def audit_production_authority(config: Mapping[str, Any]) -> dict[str, Any]:
    """Reject every production use until the later owner-bound config exists."""

    owner = config["owner_activation"]
    implementation = config["implementation_binding"]
    repository = config["production_repository"]
    if owner["status"] != "BOUND":
        raise AuthorityError("owner issuance is required before any runtime I/O")
    if owner["independent_review_status"] != "PASS" or _unknown(owner["independent_review_reference"]):
        raise AuthorityError("a distinct independent review PASS is required before any runtime I/O")
    if _unknown(owner["decision_issuance_reference"]) or _unknown(owner["issued_at"]):
        raise AuthorityError("owner issuance provenance is incomplete")
    if implementation["status"] != "BOUND" or repository["status"] != "BOUND":
        raise AuthorityError("implementation and repository bindings are required before any runtime I/O")
    _hex(implementation["implementation_commit"], label="implementation commit", width=40)
    _hex(implementation["implementation_script_sha256"], label="script hash")
    _hex(implementation["implementation_test_sha256"], label="test hash")
    _hex(repository["expected_head_commit"], label="repository head", width=40)
    if not isinstance(repository["repository_root"], str) or not repository["repository_root"]:
        raise AuthorityError("repository root binding is invalid")
    if not isinstance(repository["branch"], str) or not repository["branch"]:
        raise AuthorityError("repository branch binding is invalid")
    repo_root = Path(repository["repository_root"])
    if _run_git(repo_root, "rev-parse", "--show-toplevel") != str(repo_root.resolve()):
        raise AuthorityError("repository root does not match its Git worktree")
    if _run_git(repo_root, "status", "--porcelain"):
        raise AuthorityError("production repository worktree or index is not clean")
    head = _run_git(repo_root, "rev-parse", "HEAD")
    upstream = _run_git(repo_root, "rev-parse", "@{upstream}")
    origin = _run_git(repo_root, "rev-parse", f"refs/remotes/origin/{repository['branch']}")
    expected = repository["expected_head_commit"]
    if head != expected or upstream != expected or origin != expected:
        raise AuthorityError("production repository is not exactly pushed and synchronized")
    if head != implementation["implementation_commit"]:
        raise AuthorityError("repository head differs from bound implementation commit")
    script_path = repo_root / implementation["implementation_script_path"]
    test_path = repo_root / implementation["implementation_test_path"]
    try:
        script_hash = sha256(script_path.read_bytes())
        test_hash = sha256(test_path.read_bytes())
    except OSError as exc:
        raise AuthorityError("bound implementation source is unavailable") from exc
    if script_hash != implementation["implementation_script_sha256"] or test_hash != implementation["implementation_test_sha256"]:
        raise AuthorityError("bound implementation source hash differs")
    return {
        "status": "PASS_OWNER_ISSUED_REVIEWED_PUSHED_BOUND",
        "owner_issuance_reference": owner["decision_issuance_reference"],
        "independent_review_reference": owner["independent_review_reference"],
        "implementation_commit": head,
        "repository_branch": repository["branch"],
        "repository_head": head,
    }


def _synthetic_audit() -> dict[str, Any]:
    return {
        "status": "SYNTHETIC_FIXTURE_NOT_PRODUCTION",
        "owner_issuance_reference": "SYNTHETIC_FIXTURE_NOT_PRODUCTION",
        "independent_review_reference": "SYNTHETIC_FIXTURE_NOT_PRODUCTION",
        "implementation_commit": "SYNTHETIC_FIXTURE_NOT_PRODUCTION",
        "repository_branch": "SYNTHETIC_FIXTURE_NOT_PRODUCTION",
        "repository_head": "SYNTHETIC_FIXTURE_NOT_PRODUCTION",
    }


def _context(
    *,
    config_path: Path = PRODUCTION_CONFIG_PATH,
    config_override: dict[str, Any] | None = None,
    production: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = copy.deepcopy(config_override) if config_override is not None else load_config(config_path)
    validate_static_config(config)
    if production:
        return config, audit_production_authority(config)
    owner = config["owner_activation"]
    implementation = config["implementation_binding"]
    repository = config["production_repository"]
    if not (
        owner["status"] == implementation["status"] == repository["status"] == "SYNTHETIC_TEST_ONLY"
    ):
        raise AuthorityError("non-production execution is restricted to explicit synthetic fixtures")
    return config, _synthetic_audit()


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _prepared_path(prepared_directory: Path | str, config: Mapping[str, Any]) -> Path:
    prepared = _absolute(prepared_directory)
    allowed = _absolute(_runtime(config)["allowed_prepared_root"])
    try:
        common = Path(os.path.commonpath((str(prepared), str(allowed))))
    except ValueError as exc:
        raise PublicationError("prepared directory is outside the allowed root") from exc
    if common != allowed or prepared == allowed:
        raise PublicationError("prepared directory must be a strict child of the allowed root")
    return prepared


@contextmanager
def _locked_run(run_root: Path) -> Iterator[None]:
    try:
        descriptor = os.open(run_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError as exc:
        raise PublicationError("cannot open runtime root") from exc
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
        raise PublicationError("cannot read runtime mutable documents") from exc


def _parse_runtime(payloads: Mapping[str, bytes]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if set(payloads) != set(MUTABLE_NAMES):
        raise PredecessorError("runtime mutable member closure differs")
    return (
        load_json(payloads["STATUS.json"], label="STATUS.json"),
        load_json(payloads["RUN_MANIFEST.json"], label="RUN_MANIFEST.json"),
        load_events(payloads["EVENT_LOG.jsonl"], label="EVENT_LOG.jsonl"),
    )


def _validate_outer(document: Mapping[str, Any], *, label: str) -> None:
    expected = {
        "qualified_ordinary_studies": FROZEN_COUNTS["ordinary"],
        "qualified_a1_studies": FROZEN_COUNTS["a1"],
        "qualified_a2_dense_studies": FROZEN_COUNTS["true_a2"],
        "canonical_intervention_record_count": FROZEN_COUNTS["canonical_records"],
        "canonical_record_count": FROZEN_COUNTS["canonical_records"],
        **OUTER_LOCKS,
    }
    for key, value in expected.items():
        _expect(document.get(key), value, label=f"{label}.{key}")


def _event_number(config: Mapping[str, Any], event_id: Any) -> int:
    runtime = _runtime(config)
    prefix = runtime["event_id_prefix"]
    width = runtime["event_id_width"]
    if not isinstance(event_id, str) or not re.fullmatch(re.escape(prefix) + rf"[0-9]{{{width}}}", event_id):
        raise PredecessorError("runtime event identifier has an invalid format")
    return int(event_id[len(prefix) :])


def _event_id(config: Mapping[str, Any], number: int) -> str:
    runtime = _runtime(config)
    if number <= 0 or number >= 10 ** runtime["event_id_width"]:
        raise PredecessorError("successor event number is outside the configured namespace")
    return f"{runtime['event_id_prefix']}{number:0{runtime['event_id_width']}d}"


def validate_predecessor(config: Mapping[str, Any], payloads: Mapping[str, bytes]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Validate a fresh legal DEC027 tail without preallocating its successor."""

    status, manifest, events = _parse_runtime(payloads)
    _validate_outer(status, label="predecessor STATUS")
    _validate_outer(manifest, label="predecessor RUN_MANIFEST")
    _expect(
        status.get("active_amendment_decision_ids"),
        list(BEFORE_DECISION_IDS),
        label="predecessor STATUS decisions",
    )
    _expect(
        manifest.get("active_amendment_decision_ids"),
        list(BEFORE_DECISION_IDS),
        label="predecessor RUN_MANIFEST decisions",
    )
    active_authority = manifest.get("active_authority_commit")
    if not isinstance(active_authority, str) or not HEX40.fullmatch(active_authority):
        raise PredecessorError("historical active authority commit is invalid")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise PredecessorError("predecessor manifest outputs are absent")
    paths = [item.get("absolute_path") for item in outputs if isinstance(item, dict)]
    if len(paths) != len(outputs) or len(paths) != len(set(paths)):
        raise PredecessorError("predecessor manifest output paths are not unique")
    for index, event in enumerate(events, start=1):
        if _event_number(config, event.get("event_id")) != index:
            raise PredecessorError("runtime events are not one contiguous sequence")
    tail = events[-1]
    if tail.get("decision_id") != _runtime(config)["predecessor_decision_id"]:
        raise PredecessorError("live tail does not carry the required DEC027 predecessor decision")
    if tail.get("phase_id") != "A1":
        raise PredecessorError("live tail phase does not match A1")
    return status, manifest, events


def _current_contract_authority(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "decision_id": "V3-DEC-028",
        "scope": config["static_authority"]["next_operational_scope"],
        "activation": "FRESH_RUNTIME_AUTHORITY_SYNC_ONLY",
        "scientific_claim_status": "NOT_ESTABLISHED",
        "counts_preserved": copy.deepcopy(FROZEN_COUNTS),
    }


def _output_record(artifact_type: str, path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "absolute_path": str(path),
        "artifact_type": artifact_type,
        "bytes": len(payload),
        "sha256": sha256(payload),
        "status": "COMPLETE",
    }


def _build_sync_record(
    config: Mapping[str, Any],
    *,
    predecessor_event_id: str,
    successor_event_id: str,
    recorded_at: str,
    snapshots: Mapping[str, bytes],
    historical_active_authority_commit: str,
    authority_audit: Mapping[str, Any],
) -> bytes:
    return json_bytes(
        {
            "schema_version": "1.0.0",
            "record_type": "ROUTE_A_V3_A1_DEC028_AUTHORITY_RUNTIME_SYNC",
            "sync_type": config["sync_type"],
            "contract_id": config["contract_id"],
            "phase_id": "A1",
            "decision_id": "V3-DEC-028",
            "event_id": successor_event_id,
            "predecessor_event_id": predecessor_event_id,
            "recorded_at": recorded_at,
            "candidate_contract": copy.deepcopy(config["candidate_contract"]),
            "amendment": copy.deepcopy(config["amendment"]),
            "predecessor_snapshot_count": len(snapshots),
            "predecessor_snapshot_names": list(snapshots),
            "snapshot_sha256": {name: sha256(payload) for name, payload in snapshots.items()},
            "new_registered_artifact_count": 0,
            "output_delta_count": 4,
            "active_amendment_decision_ids": list(AFTER_DECISION_IDS),
            "current_contract_authority": _current_contract_authority(config),
            "historical_outer_runtime_authority": {
                "active_authority_commit": historical_active_authority_commit,
                "active_authority_commit_rewritten": False,
            },
            "runtime_sync_publisher_authority": copy.deepcopy(dict(authority_audit)),
            "current_qualified_counts": copy.deepcopy(FROZEN_COUNTS),
            "locks": copy.deepcopy(config["static_authority"]["locks"]),
            "preflight_executed": False,
            "p0_production_executed": False,
            "qualification_changed": False,
            "scientific_state_changed": False,
        }
    )


def _event_document(
    config: Mapping[str, Any],
    *,
    predecessor_event_id: str,
    successor_event_id: str,
    recorded_at: str,
    sync_digest: str,
) -> dict[str, Any]:
    return {
        "event_id": successor_event_id,
        "at": recorded_at,
        "phase_id": "A1",
        "event": "DEC028_SINGLE_STUDY_AUTHORITY_RUNTIME_SYNC_REGISTERED_NO_P0_EXECUTION",
        "sync_type": config["sync_type"],
        "decision_id": "V3-DEC-028",
        "predecessor_event_id": predecessor_event_id,
        "sync_name": _runtime(config)["sync_name"],
        "sync_record_sha256": sync_digest,
        "registered_artifacts": [],
        "new_registered_artifact_count": 0,
        "predecessor_snapshot_count": 3,
        "predecessor_snapshot_names": list(_snapshot_names(config).values()),
        "output_delta_count": 4,
        "active_amendment_decision_ids": list(AFTER_DECISION_IDS),
        "current_contract_authority": _current_contract_authority(config),
        "current_qualified_counts": copy.deepcopy(FROZEN_COUNTS),
        "locks": copy.deepcopy(config["static_authority"]["locks"]),
        "preflight_executed": False,
        "p0_production_executed": False,
        "qualification_changed": False,
        "scientific_state_changed": False,
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
        "detail": (
            "Registered V3-DEC-028 authority only. No data-row access, materialization, CUDA probe, "
            "model or optimizer construction, checkpoint I/O, parameter update, training, model selection, "
            "P0 production execution, G1, A7, sealed access, qualification, or scientific-state change occurred."
        ),
    }


def _successor_updates(config: Mapping[str, Any], *, recorded_at: str, sync_digest: str, successor_event_id: str) -> dict[str, Any]:
    return {
        "active_amendment_decision_ids": list(AFTER_DECISION_IDS),
        "current_contract_authority": _current_contract_authority(config),
        "dec028_authority_runtime_sync_status": f"SYNCED_{successor_event_id}",
        "dec028_authority_runtime_sync_recorded_at": recorded_at,
        "dec028_authority_runtime_sync_record_sha256": sync_digest,
        "dec028_authority_runtime_sync_scientific_state_changed": False,
        "dec028_authority_runtime_sync_qualification_changed": False,
        "single_study_s0_authority_registered": True,
        "single_study_p0_production_authorized": False,
    }


def _immutable_output_delta(config: Mapping[str, Any], predecessor: Mapping[str, bytes], sync_payload: bytes) -> list[dict[str, Any]]:
    run_root = Path(_runtime(config)["run_root"])
    snapshots = _snapshot_names(config)
    records = [
        _output_record(
            f"A1_{name.replace('.', '_').upper()}_PRE_DEC028_AUTHORITY_RUNTIME_SYNC_SNAPSHOT",
            run_root / snapshots[name],
            predecessor[name],
        )
        for name in MUTABLE_NAMES
    ]
    records.append(
        _output_record(
            "A1_DEC028_AUTHORITY_RUNTIME_SYNC_V1",
            run_root / _runtime(config)["sync_name"],
            sync_payload,
        )
    )
    return records


def build_successors(
    config: dict[str, Any],
    predecessor_payloads: Mapping[str, bytes],
    recorded_at: str,
    authority_audit: Mapping[str, Any] | None = None,
) -> dict[str, bytes]:
    """Pure dynamic successor construction used by synthetic tests and prepare."""

    validate_static_config(config)
    status, manifest, events = validate_predecessor(config, predecessor_payloads)
    predecessor_event = events[-1]
    _iso_after(recorded_at, predecessor_event.get("at"))
    predecessor_event_id = predecessor_event["event_id"]
    successor_event_id = _event_id(config, _event_number(config, predecessor_event_id) + 1)
    snapshots = _snapshot_names(config)
    snapshot_payloads = {snapshots[name]: predecessor_payloads[name] for name in MUTABLE_NAMES}
    audit = dict(authority_audit) if authority_audit is not None else _synthetic_audit()
    sync_payload = _build_sync_record(
        config,
        predecessor_event_id=predecessor_event_id,
        successor_event_id=successor_event_id,
        recorded_at=recorded_at,
        snapshots=snapshot_payloads,
        historical_active_authority_commit=manifest["active_authority_commit"],
        authority_audit=audit,
    )
    sync_digest = sha256(sync_payload)
    updates = _successor_updates(
        config, recorded_at=recorded_at, sync_digest=sync_digest, successor_event_id=successor_event_id
    )
    successor_status = copy.deepcopy(status)
    successor_status["updated_at"] = recorded_at
    successor_status.update(updates)
    successor_manifest = copy.deepcopy(manifest)
    successor_manifest.update(updates)
    successor_manifest["outputs"] = list(manifest["outputs"]) + _immutable_output_delta(
        config, predecessor_payloads, sync_payload
    )
    event = _event_document(
        config,
        predecessor_event_id=predecessor_event_id,
        successor_event_id=successor_event_id,
        recorded_at=recorded_at,
        sync_digest=sync_digest,
    )
    successors = {
        **snapshot_payloads,
        _runtime(config)["sync_name"]: sync_payload,
        "STATUS.json": json_bytes(successor_status),
        "RUN_MANIFEST.json": json_bytes(successor_manifest),
        "EVENT_LOG.jsonl": predecessor_payloads["EVENT_LOG.jsonl"] + json_line(event),
    }
    validate_successors(config, predecessor_payloads, successors, authority_audit=audit)
    return successors


def validate_successors(
    config: Mapping[str, Any],
    predecessor_payloads: Mapping[str, bytes],
    successors: Mapping[str, bytes],
    authority_audit: Mapping[str, Any] | None = None,
) -> None:
    validate_static_config(config)
    old_status, old_manifest, old_events = validate_predecessor(config, predecessor_payloads)
    if set(successors) != expected_member_names(config):
        raise RuntimeSyncError("prepared member closure is not exact seven")
    snapshots = _snapshot_names(config)
    for mutable, snapshot in snapshots.items():
        if successors[snapshot] != predecessor_payloads[mutable]:
            raise RuntimeSyncError("predecessor snapshot bytes differ")
    audit = dict(authority_audit) if authority_audit is not None else _synthetic_audit()
    predecessor_event = old_events[-1]
    predecessor_event_id = predecessor_event["event_id"]
    successor_event_id = _event_id(config, _event_number(config, predecessor_event_id) + 1)
    status, manifest, events = _parse_runtime({name: successors[name] for name in MUTABLE_NAMES})
    if events[:-1] != old_events or len(events) != len(old_events) + 1:
        raise RuntimeSyncError("EVENT_LOG is not one exact append")
    if not successors["EVENT_LOG.jsonl"].startswith(predecessor_payloads["EVENT_LOG.jsonl"]):
        raise RuntimeSyncError("EVENT_LOG predecessor prefix differs")
    sync_payload = successors[_runtime(config)["sync_name"]]
    expected_sync = _build_sync_record(
        config,
        predecessor_event_id=predecessor_event_id,
        successor_event_id=successor_event_id,
        recorded_at=events[-1].get("at"),
        snapshots={snapshots[name]: predecessor_payloads[name] for name in MUTABLE_NAMES},
        historical_active_authority_commit=old_manifest["active_authority_commit"],
        authority_audit=audit,
    )
    if sync_payload != expected_sync:
        raise RuntimeSyncError("sync record structural closure differs")
    expected_event = _event_document(
        config,
        predecessor_event_id=predecessor_event_id,
        successor_event_id=successor_event_id,
        recorded_at=events[-1].get("at"),
        sync_digest=sha256(sync_payload),
    )
    if events[-1] != expected_event:
        raise RuntimeSyncError("successor event structural closure differs")
    updates = _successor_updates(
        config,
        recorded_at=events[-1]["at"],
        sync_digest=sha256(sync_payload),
        successor_event_id=successor_event_id,
    )
    expected_status = copy.deepcopy(old_status)
    expected_status["updated_at"] = events[-1]["at"]
    expected_status.update(updates)
    if successors["STATUS.json"] != json_bytes(expected_status):
        raise RuntimeSyncError("STATUS whole-document closure differs")
    expected_manifest = copy.deepcopy(old_manifest)
    expected_manifest.update(updates)
    expected_manifest["outputs"] = list(old_manifest["outputs"]) + _immutable_output_delta(
        config, predecessor_payloads, sync_payload
    )
    if successors["RUN_MANIFEST.json"] != json_bytes(expected_manifest):
        raise RuntimeSyncError("RUN_MANIFEST whole-document closure differs")
    _validate_outer(status, label="successor STATUS")
    _validate_outer(manifest, label="successor RUN_MANIFEST")
    _expect(status.get("active_amendment_decision_ids"), list(AFTER_DECISION_IDS), label="successor STATUS decisions")
    _expect(manifest.get("active_amendment_decision_ids"), list(AFTER_DECISION_IDS), label="successor manifest decisions")
    if manifest.get("active_authority_commit") != old_manifest.get("active_authority_commit"):
        raise RuntimeSyncError("historical active authority commit was rewritten")


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise PublicationError(f"cannot atomically write {path.name}") from exc


def _write_immutable_once(path: Path, payload: bytes) -> str:
    if path.exists():
        try:
            if path.read_bytes() == payload:
                return "REUSED_IDENTICAL"
        except OSError as exc:
            raise PublicationError(f"cannot inspect immutable output {path.name}") from exc
        raise PublicationError(f"immutable output already exists with different bytes: {path.name}")
    _write_atomic(path, payload)
    try:
        if path.read_bytes() != payload:
            raise PublicationError(f"immutable output was not written exactly: {path.name}")
    except OSError as exc:
        raise PublicationError(f"cannot verify immutable output {path.name}") from exc
    return "CREATED_EXACT"


def _write_prepared(prepared: Path, members: Mapping[str, bytes]) -> None:
    prepared.mkdir(parents=True, exist_ok=True)
    try:
        observed = {item.name for item in prepared.iterdir()}
    except OSError as exc:
        raise PublicationError("cannot inspect prepared directory") from exc
    if observed - set(members):
        raise PublicationError("prepared directory contains unexpected members")
    for name, payload in members.items():
        target = prepared / name
        if target.exists():
            try:
                if target.read_bytes() != payload:
                    raise PublicationError(f"prepared member differs: {name}")
            except OSError as exc:
                raise PublicationError(f"cannot inspect prepared member: {name}") from exc
        else:
            _write_atomic(target, payload)
    try:
        if {item.name for item in prepared.iterdir()} != set(members):
            raise PublicationError("prepared member closure is incomplete")
    except OSError as exc:
        raise PublicationError("cannot recheck prepared directory") from exc


def _read_prepared(config: Mapping[str, Any], prepared: Path) -> dict[str, bytes]:
    expected = expected_member_names(config)
    try:
        observed = {item.name for item in prepared.iterdir()}
    except OSError as exc:
        raise PublicationError("prepared directory is absent") from exc
    if observed != expected:
        raise PublicationError("prepared member closure is incomplete or has extras")
    try:
        return {name: (prepared / name).read_bytes() for name in expected}
    except OSError as exc:
        raise PublicationError("cannot read prepared members") from exc


def _split_prepared(config: Mapping[str, Any], prepared: Mapping[str, bytes]) -> tuple[dict[str, bytes], dict[str, bytes]]:
    snapshots = _snapshot_names(config)
    predecessor = {mutable: prepared[snapshot] for mutable, snapshot in snapshots.items()}
    successors = {name: prepared[name] for name in expected_member_names(config)}
    return predecessor, successors


def _runtime_prefix(predecessor: Mapping[str, bytes], successor: Mapping[str, bytes], current: Mapping[str, bytes]) -> int:
    prefix = 0
    saw_predecessor = False
    for name in MUTABLE_NAMES:
        payload = current[name]
        if payload == successor[name] and not saw_predecessor:
            prefix += 1
        elif payload == predecessor[name]:
            saw_predecessor = True
        else:
            raise PredecessorError("live mutable state is not a recoverable ordered publication prefix")
    return prefix


def prepare_runtime_sync(
    *,
    prepared_directory: Path | str,
    recorded_at: str,
    config_path: Path = PRODUCTION_CONFIG_PATH,
    config_override: dict[str, Any] | None = None,
    run_root_override: Path | None = None,
    production: bool = True,
) -> dict[str, Any]:
    config, audit = _context(config_path=config_path, config_override=config_override, production=production)
    prepared = _prepared_path(prepared_directory, config)
    if production and run_root_override is not None:
        raise PublicationError("run-root override is forbidden in production")
    run_root = run_root_override or Path(_runtime(config)["run_root"])
    with _locked_run(run_root):
        predecessor = _read_runtime(run_root)
        successors = build_successors(config, predecessor, recorded_at, authority_audit=audit)
        if _read_runtime(run_root) != predecessor:
            raise PredecessorError("live predecessor changed before prepared output")
        _write_prepared(prepared, successors)
    event = load_events(successors["EVENT_LOG.jsonl"], label="prepared EVENT_LOG")[-1]
    return {
        "status": "PREPARED_NOT_PUBLISHED",
        "event_id": event["event_id"],
        "predecessor_event_id": event["predecessor_event_id"],
        "prepared_directory": str(prepared),
        "prepared_member_count": len(successors),
    }


def publish_prepared(
    *,
    prepared_directory: Path | str,
    config_path: Path = PRODUCTION_CONFIG_PATH,
    config_override: dict[str, Any] | None = None,
    run_root_override: Path | None = None,
    production: bool = True,
    fault_injector: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    config, audit = _context(config_path=config_path, config_override=config_override, production=production)
    prepared_path = _prepared_path(prepared_directory, config)
    if production and run_root_override is not None:
        raise PublicationError("run-root override is forbidden in production")
    run_root = run_root_override or Path(_runtime(config)["run_root"])
    prepared = _read_prepared(config, prepared_path)
    predecessor, successor = _split_prepared(config, prepared)
    validate_successors(config, predecessor, successor, authority_audit=audit)
    immutable_results: dict[str, str] = {}
    with _locked_run(run_root):
        current = _read_runtime(run_root)
        prefix = _runtime_prefix(predecessor, successor, current)
        try:
            for name in _runtime(config)["immutable_publish_order"]:
                if fault_injector is not None:
                    fault_injector(f"before_immutable:{name}")
                immutable_results[name] = _write_immutable_once(run_root / name, prepared[name])
            for name in MUTABLE_NAMES[prefix:]:
                if fault_injector is not None:
                    fault_injector(f"before_replace:{name}")
                _write_atomic(run_root / name, successor[name])
        except Exception as exc:
            raise PublicationError("publication interrupted; retry the same prepared directory") from exc
        if _read_runtime(run_root) != {name: successor[name] for name in MUTABLE_NAMES}:
            raise PublicationError("publication finished non-exactly")
    event_id = load_events(successor["EVENT_LOG.jsonl"], label="prepared EVENT_LOG")[-1]["event_id"]
    return {"status": "PUBLISHED_VERIFIED", "event_id": event_id, "immutable_results": immutable_results}


def validate_published(
    *,
    prepared_directory: Path | str,
    config_path: Path = PRODUCTION_CONFIG_PATH,
    config_override: dict[str, Any] | None = None,
    run_root_override: Path | None = None,
    production: bool = True,
) -> dict[str, Any]:
    config, audit = _context(config_path=config_path, config_override=config_override, production=production)
    prepared_path = _prepared_path(prepared_directory, config)
    if production and run_root_override is not None:
        raise PublicationError("run-root override is forbidden in production")
    run_root = run_root_override or Path(_runtime(config)["run_root"])
    prepared = _read_prepared(config, prepared_path)
    predecessor, successor = _split_prepared(config, prepared)
    validate_successors(config, predecessor, successor, authority_audit=audit)
    with _locked_run(run_root):
        if _read_runtime(run_root) != {name: successor[name] for name in MUTABLE_NAMES}:
            raise PublicationError("runtime does not match prepared successor")
        for name in _runtime(config)["immutable_publish_order"]:
            try:
                if (run_root / name).read_bytes() != prepared[name]:
                    raise PublicationError("immutable output does not match prepared bytes")
            except OSError as exc:
                raise PublicationError("cannot read immutable output") from exc
    event_id = load_events(successor["EVENT_LOG.jsonl"], label="prepared EVENT_LOG")[-1]["event_id"]
    return {"status": "PUBLISHED_VERIFIED", "event_id": event_id}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--prepared-directory", type=Path, required=True)
    prepare.add_argument("--recorded-at", required=True)
    publish = subparsers.add_parser("publish")
    publish.add_argument("--prepared-directory", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--prepared-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_runtime_sync(prepared_directory=args.prepared_directory, recorded_at=args.recorded_at)
        elif args.command == "publish":
            result = publish_prepared(prepared_directory=args.prepared_directory)
        else:
            result = validate_published(prepared_directory=args.prepared_directory)
    except RuntimeSyncError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
