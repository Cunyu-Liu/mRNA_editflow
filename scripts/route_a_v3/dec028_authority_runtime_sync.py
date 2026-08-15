#!/usr/bin/env python3
"""Prepare, publish, and validate the authority-only A1-EVT-061 sync.

This transaction activates only the already-signed V3-DEC-028 S0/P0-closure
operating authority.  It does not read a dataset, materialize rows, execute a
split, run P0, touch CUDA, create a model, update parameters, or unlock G1.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


_BASE_PATH = Path(__file__).with_name("dec027_authority_runtime_sync.py")
_BASE_SPEC = importlib.util.spec_from_file_location("_dec027_runtime_base", _BASE_PATH)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise RuntimeError("cannot load the settled runtime-sync implementation")
base = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(base)

UNKNOWN = "UNKNOWN_NOT_ASSERTED"
BOUND = "BOUND"
MUTABLE_NAMES = ("STATUS.json", "RUN_MANIFEST.json", "EVENT_LOG.jsonl")
CONFIG_REPO_PATH = "configs/route_a_v3_dec028_authority_runtime_sync_v1.json"
SCRIPT_REPO_PATH = "scripts/route_a_v3/dec028_authority_runtime_sync.py"
TEST_REPO_PATH = "tests/route_a_v3/test_dec028_authority_runtime_sync.py"
IMPLEMENTATION_PATHS = [CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH]
PRODUCTION_CONFIG_PATH = Path(__file__).resolve().parents[2] / CONFIG_REPO_PATH
AUTHORITY_COMMIT = "89ae669313b0adbc7ca4e05f8ffee37ad4c9d2a7"
AUTHORITY_PARENT = "43d29569aa979fca46cd50ee8e8763fb1f59bb52"
I1_COMMIT = "ac7122858d07d8027e02da4ce2fca841fb75eb87"
I1_FILES = [
    {"path": CONFIG_REPO_PATH, "bytes": 8468, "sha256": "61828137763a1183fcf749f491cf372434789338987c2d51a9d5716a4ddcfdc1"},
    {"path": SCRIPT_REPO_PATH, "bytes": 31106, "sha256": "271a8a541c1452ad58e7c29a6ec403d97925e1d3445421c668ee8cba944ac597"},
    {"path": TEST_REPO_PATH, "bytes": 10096, "sha256": "0573cd5dcfa327b65fa3faeeb85887a1a74fdce21ed647e7370af0cf94ce85c3"},
]
I2_COMMIT = "fba71d896ed72aaedc9466b2919de379b7c240c6"
I2_FILES = [
    {"path": CONFIG_REPO_PATH, "bytes": 9278, "sha256": "efed13c6f09b60cc7bcbc8b64312b53a73ca2070f601dfa6435142ac09dc8798"},
    {"path": SCRIPT_REPO_PATH, "bytes": 32561, "sha256": "c95cad3c9b96aea841f15cdb0073f9c93039612392e7e633fec2520f58a3feaf"},
    {"path": TEST_REPO_PATH, "bytes": 10445, "sha256": "23324fc8e598295da76b7ddd016376eff839df971cf33dbde02b17c0301a93c8"},
]
I3_COMMIT = "0f418e04fcb7ee496caa4469e2f57bcf76f96c4a"
I3_FILES = [
    {"path": CONFIG_REPO_PATH, "bytes": 10081, "sha256": "6c819f3d9fc3b4997abd30a6b9a89ee93cd8efd13a36bb227a48cf5fdbdc89b7"},
    {"path": SCRIPT_REPO_PATH, "bytes": 33855, "sha256": "dad8bcd2f7f8421eefcaf7fe33b91875bd69e0ad81722311cc6698858937e449"},
    {"path": TEST_REPO_PATH, "bytes": 10479, "sha256": "3116cb611c9a570232b7ae350dbe143da9df7fe7dd3f17a692ec520fdf13b155"},
]
FAILED_B3_COMMIT = "b5ce55c853aff6c1ca5604060ac0c2ec2774f530"
FAILED_B3_FILES = [
    {"path": CONFIG_REPO_PATH, "bytes": 10174, "sha256": "2be2ac8bcc64c4183de51d3b7fec2279f092726febec35ec479b7036ffc66bc5"},
    {"path": SCRIPT_REPO_PATH, "bytes": 33855, "sha256": "dad8bcd2f7f8421eefcaf7fe33b91875bd69e0ad81722311cc6698858937e449"},
    {"path": TEST_REPO_PATH, "bytes": 10479, "sha256": "3116cb611c9a570232b7ae350dbe143da9df7fe7dd3f17a692ec520fdf13b155"},
]
BRANCH = "routea-v3-a1-20260810"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ACTIVE_DECISION_IDS = [
    "V3-DEC-017", "V3-DEC-018", "V3-DEC-019", "V3-DEC-020",
    "V3-DEC-021", "V3-DEC-022", "V3-DEC-023", "V3-DEC-024",
    "V3-DEC-027", "V3-DEC-028",
]
UNKNOWN_BINDING_FIELDS = (
    "status", "implementation_commit", "implementation_script_sha256",
    "implementation_test_sha256",
)

RuntimeSyncError = base.RuntimeSyncError
BindingError = base.BindingError
AuthorityError = base.AuthorityError
PredecessorError = base.PredecessorError
PublicationError = base.PublicationError
sha256 = base.sha256
json_bytes = base.json_bytes
compact_json_line = base.compact_json_line
load_json = base.load_json
load_events = base.load_events
_expect = base._expect
_expect_hex = base._expect_hex


def _binding_state(binding: Mapping[str, Any]) -> str:
    values = [binding.get(key) for key in UNKNOWN_BINDING_FIELDS]
    if values == [UNKNOWN] * len(values):
        return "UNKNOWN"
    if UNKNOWN in values:
        raise BindingError("implementation binding is partially known")
    if binding.get("status") != BOUND:
        raise BindingError("implementation binding is not BOUND")
    _expect_hex(binding.get("implementation_commit"), HEX40, label="I commit")
    _expect_hex(binding.get("implementation_script_sha256"), HEX64, label="script SHA")
    _expect_hex(binding.get("implementation_test_sha256"), HEX64, label="test SHA")
    return "BOUND"


def validate_static_config(config: dict[str, Any]) -> None:
    expected_top = {
        "schema_version", "protocol_id", "contract_id", "phase_id",
        "decision_id", "event_id", "event_name", "sync_type",
        "implementation_binding", "repository_authority", "dec028_authority",
        "runtime", "frozen_outer_truth", "access_boundary",
    }
    if set(config) != expected_top:
        raise RuntimeSyncError("config key closure differs")
    _expect(config["schema_version"], "route_a_v3_dec028_authority_runtime_sync.v1", label="schema")
    _expect(config["decision_id"], "V3-DEC-028", label="decision")
    _expect(config["event_id"], "A1-EVT-061", label="event")
    binding = config["implementation_binding"]
    _binding_state(binding)
    _expect(
        binding.get("binding_scheme"),
        "AUTHORITY_A_TO_I1_TO_I2_TO_I3_TO_FAILED_B3_TO_IMPLEMENTATION_I4_TO_CONFIG_ONLY_B4",
        label="binding scheme",
    )
    _expect(
        binding.get("frozen_predecessor_implementation"),
        {
            "status": "FROZEN_BOUND_EXACT3",
            "implementation_commit": I1_COMMIT,
            "implementation_expected_parent": AUTHORITY_COMMIT,
            "implementation_files": I1_FILES,
        },
        label="frozen I1",
    )
    _expect(
        binding.get("frozen_second_implementation"),
        {
            "status": "FROZEN_BOUND_EXACT3",
            "implementation_commit": I2_COMMIT,
            "implementation_expected_parent": I1_COMMIT,
            "implementation_files": I2_FILES,
        },
        label="frozen I2",
    )
    _expect(
        binding.get("frozen_third_implementation"),
        {
            "status": "FROZEN_BOUND_EXACT3",
            "implementation_commit": I3_COMMIT,
            "implementation_expected_parent": I2_COMMIT,
            "implementation_files": I3_FILES,
        },
        label="frozen I3",
    )
    _expect(
        binding.get("frozen_failed_binding"),
        {
            "status": "FROZEN_FAILED_PRODUCTION_AUDIT_NO_RUNTIME_IO",
            "binding_commit": FAILED_B3_COMMIT,
            "binding_expected_parent": I3_COMMIT,
            "binding_files": FAILED_B3_FILES,
        },
        label="failed B3",
    )
    _expect(binding["implementation_exact_changed_paths"], IMPLEMENTATION_PATHS, label="I exact3")
    _expect(binding["binding_exact_changed_paths"], [CONFIG_REPO_PATH], label="B config-only")

    authority = config["repository_authority"]
    _expect(authority["authority_binding_status"], "FROZEN_BOUND_EXACT17", label="authority status")
    _expect(authority["authority_commit"], AUTHORITY_COMMIT, label="authority commit")
    _expect(authority["authority_expected_parent"], AUTHORITY_PARENT, label="authority parent")
    files = authority.get("authority_files")
    if not isinstance(files, list) or len(files) != 17:
        raise BindingError("authority exact17 identity closure differs")
    paths = [item.get("path") for item in files]
    if len(set(paths)) != 17:
        raise BindingError("authority exact17 paths are not unique")
    for item in files:
        _expect_hex(item.get("sha256"), HEX64, label="authority file SHA")
        if not isinstance(item.get("bytes"), int) or item["bytes"] <= 0:
            raise BindingError("authority file byte count is invalid")

    decision = config["dec028_authority"]
    required_decision = {
        "choice": "OWNER_INITIATED_PROSPECTIVE_OPERATIONAL_MAINLINE_CHOICE_NOT_DEC027_AUTOMATIC_TRIGGER",
        "active_operational_mainline": "SINGLE_STUDY_SOURCE_RELATIVE_DEVELOPMENT_ENGINEERING_THEORY",
        "current_phase": "SINGLE_STUDY_S0_AUTHORITY_AND_P0_CLOSURE",
        "primary_study_unit": "GSE200304",
        "full_route_a_retained": True,
        "full_route_a_required_counts": {"ordinary": 3, "a1": 2, "true_a2": 1},
        "current_qualified_counts": {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547},
        "scientific_claim_status": "NOT_ESTABLISHED",
        "successor_p0_exact_gate_count": 11,
        "successor_p0_required_pass_count": 11,
        "p0_nonpass_action": "STOP_BEFORE_DATA_ROWS_CUDA_MODEL_OPTIMIZER_CHECKPOINT_PARAMETER_UPDATE_OR_TRAINING",
        "materialization_authorized": False,
        "data_row_access_authorized": False,
        "split_assignment_execution_authorized": False,
        "g1_development_run_authorized": False,
        "training_authorized": False,
        "gpu_authorized": False,
        "model_selection_authorized": False,
        "a7_allowed": False,
        "next_phase_authorized": False,
        "sealed_access_authorized": False,
    }
    _expect(decision, required_decision, label="DEC028 authority")

    runtime = config["runtime"]
    fixed_runtime = {
        "predecessor_event_id": "A1-EVT-060",
        "predecessor_event_count": 60,
        "successor_event_id": "A1-EVT-061",
        "successor_event_count": 61,
        "predecessor_manifest_output_count": 266,
        "successor_manifest_output_count": 270,
        "predecessor_manifest_registered_artifact_count": 14,
        "successor_manifest_registered_artifact_count": 14,
        "sync_name": "A1_DEC028_AUTHORITY_RUNTIME_SYNC_V1.json",
        "output_delta_count": 4,
        "mutable_publish_order": list(MUTABLE_NAMES),
    }
    for key, expected in fixed_runtime.items():
        _expect(runtime.get(key), expected, label=f"runtime {key}")
    specs = runtime.get("predecessor_mutables")
    if not isinstance(specs, dict) or set(specs) != set(MUTABLE_NAMES):
        raise RuntimeSyncError("predecessor mutable closure differs")
    expected_immutables = [specs[name]["snapshot_name"] for name in MUTABLE_NAMES] + [runtime["sync_name"]]
    _expect(runtime["immutable_publish_order"], expected_immutables, label="immutable order")
    for spec in specs.values():
        _expect_hex(spec.get("sha256"), HEX64, label="predecessor SHA")
    _expect_hex(runtime["predecessor_tail"].get("sha256"), HEX64, label="tail SHA")
    for value in config["access_boundary"].values():
        if value != 0:
            raise RuntimeSyncError("access boundary is not zero")


def normalized_unknown_i_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    for field in UNKNOWN_BINDING_FIELDS:
        result["implementation_binding"][field] = UNKNOWN
    validate_static_config(result)
    return result


def _load_config_payload(path: Path, *, require_bound: bool) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise BindingError("cannot read runtime-sync config") from exc
    config = load_json(payload, label="DEC028 runtime-sync config")
    validate_static_config(config)
    if require_bound and _binding_state(config["implementation_binding"]) != "BOUND":
        raise BindingError("runtime-sync implementation is not BOUND")
    return config, payload


def load_config(path: Path = PRODUCTION_CONFIG_PATH, *, require_bound: bool = False) -> dict[str, Any]:
    return _load_config_payload(path, require_bound=require_bound)[0]


def audit_production_repository_authority(config: dict[str, Any], config_payload: bytes) -> dict[str, Any]:
    validate_static_config(config)
    if _binding_state(config["implementation_binding"]) != "BOUND":
        raise BindingError("runtime-sync implementation is not BOUND")
    authority = config["repository_authority"]
    binding = config["implementation_binding"]
    repo = Path(authority["production_repo_root"])
    if Path(__file__).resolve() != (repo / SCRIPT_REPO_PATH).resolve():
        raise AuthorityError("executing script is not the bound repository script")
    head = base._run_git(repo, "rev-parse", "HEAD").decode().strip()
    upstream = base._run_git(repo, "rev-parse", "@{upstream}").decode().strip()
    origin = base._run_git(repo, "rev-parse", "--verify", f"refs/remotes/origin/{BRANCH}").decode().strip()
    _expect(head, upstream, label="HEAD/upstream")
    _expect(head, origin, label="HEAD/origin")
    _expect(base._run_git(repo, "rev-parse", "--abbrev-ref", "HEAD").decode().strip(), BRANCH, label="branch")
    if base._run_git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise AuthorityError("production worktree or index is dirty")
    implementation = binding["implementation_commit"]
    _expect(base._run_git(repo, "rev-parse", f"{head}^").decode().strip(), implementation, label="B parent/I")
    _expect(base._run_git(repo, "rev-parse", f"{implementation}^").decode().strip(), FAILED_B3_COMMIT, label="I4 parent/B3")
    _expect(base._run_git(repo, "rev-parse", f"{FAILED_B3_COMMIT}^").decode().strip(), I3_COMMIT, label="B3 parent/I3")
    _expect(base._run_git(repo, "rev-parse", f"{I3_COMMIT}^").decode().strip(), I2_COMMIT, label="I3 parent/I2")
    _expect(base._run_git(repo, "rev-parse", f"{I2_COMMIT}^").decode().strip(), I1_COMMIT, label="I2 parent/I1")
    _expect(base._run_git(repo, "rev-parse", f"{I1_COMMIT}^").decode().strip(), AUTHORITY_COMMIT, label="I1 parent/A")
    _expect(base._run_git(repo, "rev-parse", f"{AUTHORITY_COMMIT}^").decode().strip(), AUTHORITY_PARENT, label="A parent")
    authority_paths = sorted(item["path"] for item in authority["authority_files"])
    _expect(base._changed_paths(repo, AUTHORITY_COMMIT), authority_paths, label="A exact17")
    _expect(base._changed_paths(repo, I1_COMMIT), sorted(IMPLEMENTATION_PATHS), label="I1 exact3")
    _expect(base._changed_paths(repo, I2_COMMIT), sorted(IMPLEMENTATION_PATHS), label="I2 exact3")
    _expect(base._changed_paths(repo, I3_COMMIT), sorted(IMPLEMENTATION_PATHS), label="I3 exact3")
    _expect(base._changed_paths(repo, FAILED_B3_COMMIT), [CONFIG_REPO_PATH], label="failed B3 config-only")
    _expect(base._changed_paths(repo, implementation), sorted(IMPLEMENTATION_PATHS), label="I4 exact3")
    _expect(base._changed_paths(repo, head), [CONFIG_REPO_PATH], label="B config-only")
    for item in authority["authority_files"]:
        blob = base._git_blob(repo, AUTHORITY_COMMIT, item["path"])
        if len(blob) != item["bytes"] or sha256(blob) != item["sha256"]:
            raise AuthorityError("authority exact17 blob identity differs")
        if base._git_blob(repo, implementation, item["path"]) != blob or base._git_blob(repo, head, item["path"]) != blob:
            raise AuthorityError("authority blob did not persist through I/B")
        if base._read_repo_file(repo, item["path"]) != blob:
            raise AuthorityError("working authority file differs")
    for item in I1_FILES:
        blob = base._git_blob(repo, I1_COMMIT, item["path"])
        if len(blob) != item["bytes"] or sha256(blob) != item["sha256"]:
            raise AuthorityError("frozen I1 exact3 blob identity differs")
    for item in I2_FILES:
        blob = base._git_blob(repo, I2_COMMIT, item["path"])
        if len(blob) != item["bytes"] or sha256(blob) != item["sha256"]:
            raise AuthorityError("frozen I2 exact3 blob identity differs")
    for item in I3_FILES:
        blob = base._git_blob(repo, I3_COMMIT, item["path"])
        if len(blob) != item["bytes"] or sha256(blob) != item["sha256"]:
            raise AuthorityError("frozen I3 exact3 blob identity differs")
    for item in FAILED_B3_FILES:
        blob = base._git_blob(repo, FAILED_B3_COMMIT, item["path"])
        if len(blob) != item["bytes"] or sha256(blob) != item["sha256"]:
            raise AuthorityError("frozen failed B3 blob identity differs")
    i_config = load_json(base._git_blob(repo, implementation, CONFIG_REPO_PATH), label="I config")
    _expect(i_config, normalized_unknown_i_config(config), label="I unknown config")
    script_blob = base._git_blob(repo, implementation, SCRIPT_REPO_PATH)
    test_blob = base._git_blob(repo, implementation, TEST_REPO_PATH)
    _expect(sha256(script_blob), binding["implementation_script_sha256"], label="I script SHA")
    _expect(sha256(test_blob), binding["implementation_test_sha256"], label="I test SHA")
    _expect(base._git_blob(repo, head, CONFIG_REPO_PATH), config_payload, label="B config")
    _expect(base._git_blob(repo, head, SCRIPT_REPO_PATH), script_blob, label="B script")
    _expect(base._git_blob(repo, head, TEST_REPO_PATH), test_blob, label="B test")
    _expect(base._read_repo_file(repo, CONFIG_REPO_PATH), config_payload, label="working config")
    _expect(base._read_repo_file(repo, SCRIPT_REPO_PATH), script_blob, label="working script")
    _expect(base._read_repo_file(repo, TEST_REPO_PATH), test_blob, label="working test")
    return {
        "status": "PASS_EXACT17_A_I1_I2_I3_FAILED_B3_I4_CONFIG_ONLY_B4",
        "authority_commit": AUTHORITY_COMMIT,
        "predecessor_implementation_commit": I1_COMMIT,
        "second_predecessor_implementation_commit": I2_COMMIT,
        "third_predecessor_implementation_commit": I3_COMMIT,
        "failed_binding_commit": FAILED_B3_COMMIT,
        "implementation_commit": implementation,
        "binding_commit": head,
        "authority_blob_count": 17,
        "worktree_and_index_clean": True,
    }


def _check_identity(payload: bytes, spec: Mapping[str, Any], *, label: str) -> None:
    if len(payload) != spec["bytes"] or sha256(payload) != spec["sha256"]:
        raise PredecessorError(f"{label} identity drift")


def _validate_outer(document: Mapping[str, Any], truth: Mapping[str, Any], *, label: str) -> None:
    for key, expected in truth.items():
        _expect(document.get(key), expected, label=f"{label}.{key}")


def validate_predecessor(config: dict[str, Any], payloads: Mapping[str, bytes]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if set(payloads) != set(MUTABLE_NAMES):
        raise PredecessorError("runtime mutable closure differs")
    for name in MUTABLE_NAMES:
        _check_identity(payloads[name], config["runtime"]["predecessor_mutables"][name], label=name)
    status, manifest, events = base._parse_runtime(payloads)
    if len(events) != 60 or [event.get("event_id") for event in events] != [f"A1-EVT-{index:03d}" for index in range(1, 61)]:
        raise PredecessorError("predecessor event sequence is not exact 1..60")
    _expect(events[-1].get("event_id"), "A1-EVT-060", label="tail event")
    _expect(events[-1].get("decision_id"), "V3-DEC-027", label="tail decision")
    tail = payloads["EVENT_LOG.jsonl"].splitlines(keepends=True)[-1]
    _check_identity(tail, config["runtime"]["predecessor_tail"], label="EVT060 tail")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 266:
        raise PredecessorError("predecessor manifest output count is not 266")
    paths = [item.get("absolute_path") for item in outputs if isinstance(item, dict)]
    if len(paths) != 266 or len(set(paths)) != 266:
        raise PredecessorError("predecessor output paths are not unique")
    _expect(manifest.get("registered_artifact_count"), 14, label="registered count")
    _validate_outer(status, config["frozen_outer_truth"], label="STATUS")
    _validate_outer(manifest, config["frozen_outer_truth"], label="RUN_MANIFEST")
    _expect_hex(manifest.get("active_authority_commit"), HEX40, label="historical active authority")
    return status, manifest, events


def _validate_time(recorded_at: str, predecessor_at: Any) -> None:
    if not isinstance(recorded_at, str) or not isinstance(predecessor_at, str):
        raise PredecessorError("timestamps must be ISO-8601 strings")
    try:
        current = datetime.fromisoformat(recorded_at)
        predecessor = datetime.fromisoformat(predecessor_at)
    except ValueError as exc:
        raise PredecessorError("timestamp is not ISO-8601") from exc
    if current.tzinfo is None or predecessor.tzinfo is None or current <= predecessor:
        raise PredecessorError("EVT061 timestamp must follow EVT060 with an offset")


def _snapshot_names(config: Mapping[str, Any]) -> dict[str, str]:
    return {name: config["runtime"]["predecessor_mutables"][name]["snapshot_name"] for name in MUTABLE_NAMES}


def _current_contract_authority(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "decision_id": "V3-DEC-028",
        "authority_commit": config["repository_authority"]["authority_commit"],
        "authority_expected_parent": AUTHORITY_PARENT,
        "scope": "SINGLE_STUDY_S0_AUTHORITY_AND_METADATA_ONLY_P0_CLOSURE",
        "authority_file_count": 17,
    }


def _output_record(kind: str, path: Path, payload: bytes) -> dict[str, Any]:
    return {"absolute_path": str(path), "artifact_type": kind, "bytes": len(payload), "sha256": sha256(payload), "status": "COMPLETE"}


def _sync_record(config: Mapping[str, Any], *, recorded_at: str, snapshots: Mapping[str, bytes], historical_active_authority_commit: str, authority_audit: Mapping[str, Any]) -> bytes:
    return json_bytes({
        "schema_version": "1.0.0",
        "record_type": "ROUTE_A_V3_A1_DEC028_AUTHORITY_RUNTIME_SYNC",
        "sync_type": config["sync_type"],
        "contract_id": config["contract_id"],
        "phase_id": config["phase_id"],
        "decision_id": "V3-DEC-028",
        "event_id": "A1-EVT-061",
        "recorded_at": recorded_at,
        "predecessor_event_id": "A1-EVT-060",
        "predecessor_snapshot_count": 3,
        "predecessor_snapshot_names": list(snapshots),
        "snapshot_sha256": {name: sha256(payload) for name, payload in snapshots.items()},
        "output_delta_count": 4,
        "manifest_output_count_before": 266,
        "manifest_output_count_after": 270,
        "manifest_registered_artifact_count_before": 14,
        "manifest_registered_artifact_count_after": 14,
        "active_amendment_decision_ids": ACTIVE_DECISION_IDS,
        "current_contract_authority": _current_contract_authority(config),
        "historical_outer_runtime_authority": {"active_authority_commit": historical_active_authority_commit, "active_authority_commit_rewritten": False},
        "runtime_sync_publisher_authority": dict(authority_audit),
        "dec028_authority": copy.deepcopy(config["dec028_authority"]),
        "frozen_outer_truth": copy.deepcopy(config["frozen_outer_truth"]),
        "access_boundary": copy.deepcopy(config["access_boundary"]),
        "p0_executed": False,
        "materialization_executed": False,
        "split_executed": False,
        "g1_executed": False,
        "scientific_state_changed": False,
        "qualification_changed": False,
    })


def _event(config: Mapping[str, Any], *, recorded_at: str, sync_digest: str) -> dict[str, Any]:
    return {
        "event_id": "A1-EVT-061", "at": recorded_at, "phase_id": "A1",
        "event": config["event_name"], "sync_type": config["sync_type"],
        "decision_id": "V3-DEC-028", "predecessor_event_id": "A1-EVT-060",
        "sync_name": config["runtime"]["sync_name"], "sync_record_sha256": sync_digest,
        "output_delta_count": 4, "manifest_output_count_before": 266,
        "manifest_output_count_after": 270, "manifest_registered_artifact_count_before": 14,
        "manifest_registered_artifact_count_after": 14,
        "active_amendment_decision_ids": ACTIVE_DECISION_IDS,
        "current_contract_authority": _current_contract_authority(config),
        "dec028_authority": copy.deepcopy(config["dec028_authority"]),
        "frozen_outer_truth": copy.deepcopy(config["frozen_outer_truth"]),
        "access_boundary": copy.deepcopy(config["access_boundary"]),
        "p0_executed": False, "materialization_executed": False,
        "split_executed": False, "g1_executed": False,
        "scientific_state_changed": False, "qualification_changed": False,
        "qualified": False, "training_started": False, "training_allowed": False,
        "training_authorized": False, "gpu_work_started": False,
        "gpu_work_allowed": False, "model_selection_allowed": False,
        "a7_allowed": False, "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "detail": "Activated only V3-DEC-028 S0/P0-closure authority. No data, materialization, split, P0, model, CUDA, training, promotion, or credit change occurred.",
    }


def _synthetic_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    return {"status": "SYNTHETIC_FIXTURE_NOT_PRODUCTION", "authority_commit": AUTHORITY_COMMIT, "implementation_commit": config["implementation_binding"]["implementation_commit"], "binding_commit": "SYNTHETIC_FIXTURE_NOT_PRODUCTION", "authority_blob_count": 17, "worktree_and_index_clean": False}


def _construct_successors(config: dict[str, Any], predecessor: Mapping[str, bytes], recorded_at: str, audit: Mapping[str, Any]) -> dict[str, bytes]:
    status, manifest, events = validate_predecessor(config, predecessor)
    _validate_time(recorded_at, events[-1].get("at"))
    snapshot_names = _snapshot_names(config)
    snapshots = {snapshot_names[name]: predecessor[name] for name in MUTABLE_NAMES}
    sync_payload = _sync_record(config, recorded_at=recorded_at, snapshots=snapshots, historical_active_authority_commit=manifest["active_authority_commit"], authority_audit=audit)
    sync_digest = sha256(sync_payload)
    updates = {
        "active_amendment_decision_ids": ACTIVE_DECISION_IDS,
        "current_contract_authority": _current_contract_authority(config),
        "dec028_authority_runtime_sync_status": "SYNCED_EVT_061",
        "dec028_authority_runtime_sync_recorded_at": recorded_at,
        "dec028_authority_runtime_sync_record_sha256": sync_digest,
        "dec028_current_phase": "SINGLE_STUDY_S0_AUTHORITY_AND_P0_CLOSURE",
        "dec028_successor_p0_status": "AUTHORIZED_NOT_RUN",
        "dec028_p0_required_pass_count": 11,
        "dec028_p0_executed": False,
        "dec028_materialization_authorized": False,
        "dec028_g1_authorized": False,
    }
    new_status = copy.deepcopy(status)
    new_status["updated_at"] = recorded_at
    new_status.update(updates)
    new_manifest = copy.deepcopy(manifest)
    new_manifest.update(updates)
    run_root = Path(config["runtime"]["run_root"])
    delta = [_output_record(f"A1_{name.replace('.', '_').upper()}_PRE_DEC028_AUTHORITY_RUNTIME_SYNC_SNAPSHOT", run_root / snapshot_names[name], predecessor[name]) for name in MUTABLE_NAMES]
    delta.append(_output_record("A1_DEC028_AUTHORITY_RUNTIME_SYNC_V1", run_root / config["runtime"]["sync_name"], sync_payload))
    new_manifest["outputs"] = list(manifest["outputs"]) + delta
    event_payload = compact_json_line(_event(config, recorded_at=recorded_at, sync_digest=sync_digest))
    return {**snapshots, config["runtime"]["sync_name"]: sync_payload, "STATUS.json": json_bytes(new_status), "RUN_MANIFEST.json": json_bytes(new_manifest), "EVENT_LOG.jsonl": predecessor["EVENT_LOG.jsonl"] + event_payload}


def build_successors(config: dict[str, Any], predecessor: Mapping[str, bytes], recorded_at: str, authority_audit: Mapping[str, Any] | None = None, *, production: bool = False) -> dict[str, bytes]:
    if production and authority_audit is None:
        raise AuthorityError("fresh production publisher authority audit is required")
    audit = dict(authority_audit) if authority_audit is not None else _synthetic_audit(config)
    successors = _construct_successors(config, predecessor, recorded_at, audit)
    validate_successors(config, predecessor, successors, authority_audit=audit, production=production)
    return successors


def validate_successors(config: dict[str, Any], predecessor: Mapping[str, bytes], successors: Mapping[str, bytes], authority_audit: Mapping[str, Any] | None = None, *, production: bool = False) -> None:
    if production and authority_audit is None:
        raise AuthorityError("fresh production publisher authority audit is required")
    audit = dict(authority_audit) if authority_audit is not None else _synthetic_audit(config)
    expected_names = set(MUTABLE_NAMES) | set(_snapshot_names(config).values()) | {config["runtime"]["sync_name"]}
    if set(successors) != expected_names or len(successors) != 7:
        raise RuntimeSyncError("prepared member closure is not exact seven")
    events = load_events(successors["EVENT_LOG.jsonl"], label="successor EVENT_LOG")
    if len(events) != 61 or not successors["EVENT_LOG.jsonl"].startswith(predecessor["EVENT_LOG.jsonl"]):
        raise RuntimeSyncError("EVENT_LOG is not one exact EVT061 append")
    recorded_at = events[-1].get("at")
    expected = _construct_successors(config, predecessor, recorded_at, audit)
    if set(expected) != set(successors) or any(expected[name] != successors[name] for name in expected):
        raise RuntimeSyncError("successor byte closure differs")


def _context(config_path: Path, *, production: bool, config_override: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if production and config_override is not None:
        raise BindingError("config override is forbidden in production")
    if production and config_path.resolve() != PRODUCTION_CONFIG_PATH.resolve():
        raise BindingError("production config is not the repository config")
    if config_override is None:
        config, payload = _load_config_payload(config_path, require_bound=True)
        audit = audit_production_repository_authority(config, payload) if production else None
        return config, audit
    config = copy.deepcopy(config_override)
    validate_static_config(config)
    if _binding_state(config["implementation_binding"]) != "BOUND":
        raise BindingError("runtime-sync implementation is not BOUND")
    return config, None


def prepare_runtime_sync(*, prepared_directory: Path | str, recorded_at: str, config_path: Path = PRODUCTION_CONFIG_PATH, production: bool = True, config_override: dict[str, Any] | None = None, run_root_override: Path | None = None) -> dict[str, Any]:
    config, audit = _context(config_path, production=production, config_override=config_override)
    prepared = base._prepared_path(prepared_directory, config)
    run_root = run_root_override or Path(config["runtime"]["run_root"])
    if production and run_root_override is not None:
        raise PublicationError("run-root override is forbidden in production")
    with base._locked_run(run_root):
        predecessor = base._read_runtime(run_root)
        successors = build_successors(config, predecessor, recorded_at, authority_audit=audit, production=production)
    base._write_prepared(prepared, base._prepared_members(config, predecessor, successors))
    return {"status": "PREPARED_NOT_PUBLISHED", "event_id": "A1-EVT-061", "prepared_directory": str(prepared), "prepared_member_count": 7, "manifest_output_transition": "266_TO_270", "manifest_registered_artifact_transition": "14_TO_14"}


def publish_prepared(*, prepared_directory: Path | str, config_path: Path = PRODUCTION_CONFIG_PATH, production: bool = True, config_override: dict[str, Any] | None = None, run_root_override: Path | None = None, fault_injector: Any = None) -> dict[str, Any]:
    config, audit = _context(config_path, production=production, config_override=config_override)
    prepared_path = base._prepared_path(prepared_directory, config)
    run_root = run_root_override or Path(config["runtime"]["run_root"])
    if production and run_root_override is not None:
        raise PublicationError("run-root override is forbidden in production")
    prepared = base._read_prepared(config, prepared_path)
    predecessor, successor = base._split_prepared(config, prepared)
    validate_successors(config, predecessor, prepared, authority_audit=audit, production=production)
    snapshots = _snapshot_names(config)
    immutables = {**{snapshots[name]: predecessor[name] for name in MUTABLE_NAMES}, config["runtime"]["sync_name"]: prepared[config["runtime"]["sync_name"]]}
    with base._locked_run(run_root):
        current = base._read_runtime(run_root)
        states = ["OLD" if current[name] == predecessor[name] else "NEW" if current[name] == successor[name] else "INVALID" for name in MUTABLE_NAMES]
        if states not in (["OLD", "OLD", "OLD"], ["NEW", "OLD", "OLD"], ["NEW", "NEW", "OLD"], ["NEW", "NEW", "NEW"]):
            raise PredecessorError("runtime mutable prefix is not recoverable")
        for name in config["runtime"]["immutable_publish_order"]:
            if fault_injector is not None:
                fault_injector(f"before_immutable:{name}")
            base._write_immutable_once(run_root / name, immutables[name])
        if states == ["NEW", "NEW", "NEW"]:
            return {"status": "PUBLISHED_VERIFIED", "event_id": "A1-EVT-061", "reused": True}
        for index, name in enumerate(MUTABLE_NAMES):
            if states[index] == "NEW":
                continue
            if fault_injector is not None:
                fault_injector(f"before_replace:{name}")
            base._write_atomic(run_root / name, successor[name])
        if base._read_runtime(run_root) != successor:
            raise PublicationError("EVT061 publication finished non-exactly")
    return {"status": "PUBLISHED_VERIFIED", "event_id": "A1-EVT-061", "reused": False}


def validate_published(*, prepared_directory: Path | str, config_path: Path = PRODUCTION_CONFIG_PATH, production: bool = True, config_override: dict[str, Any] | None = None, run_root_override: Path | None = None) -> dict[str, Any]:
    config, audit = _context(config_path, production=production, config_override=config_override)
    prepared_path = base._prepared_path(prepared_directory, config)
    run_root = run_root_override or Path(config["runtime"]["run_root"])
    if production and run_root_override is not None:
        raise PublicationError("run-root override is forbidden in production")
    prepared = base._read_prepared(config, prepared_path)
    predecessor, successor = base._split_prepared(config, prepared)
    validate_successors(config, predecessor, prepared, authority_audit=audit, production=production)
    with base._locked_run(run_root):
        if base._read_runtime(run_root) != successor:
            raise PublicationError("runtime does not match prepared EVT061")
        for name in config["runtime"]["immutable_publish_order"]:
            if (run_root / name).read_bytes() != prepared[name]:
                raise PublicationError("immutable output does not match prepared")
    return {"status": "PUBLISHED_VERIFIED", "event_id": "A1-EVT-061"}


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
