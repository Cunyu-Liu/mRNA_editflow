#!/usr/bin/env python3
"""Publish EVT043, repairing the active DEC019 runtime metadata only.

The immutable DEC019 authority record is the source of the corrected values.
Preparation preserves the three predecessor mutables outside the run root.
Publication replaces STATUS then RUN_MANIFEST and appends EVT043 last.  No new
runtime output is registered because the append-only event is the repair record.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator


UNKNOWN = "UNKNOWN_NOT_ASSERTED"
MUTABLE_NAMES = ("STATUS.json", "RUN_MANIFEST.json", "EVENT_LOG.jsonl")
PREPARED_PREDECESSOR_NAMES = {
    "STATUS.json": "STATUS_PRE_DEC019_ACTIVE_AUTHORITY_METADATA_REPAIR.json",
    "RUN_MANIFEST.json": "RUN_MANIFEST_PRE_DEC019_ACTIVE_AUTHORITY_METADATA_REPAIR.json",
    "EVENT_LOG.jsonl": "EVENT_LOG_PRE_DEC019_ACTIVE_AUTHORITY_METADATA_REPAIR.jsonl",
}
PRODUCTION_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs/route_a_v3_dec019_active_authority_runtime_metadata_repair_v1.json"
)
FaultInjector = Callable[[str], None]


class RepairError(RuntimeError):
    pass


class BindingError(RepairError):
    pass


class PredecessorError(RepairError):
    pass


class PublicationError(RepairError):
    pass


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def compact_json_line(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def load_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepairError(f"invalid JSON: {label}") from exc
    if not isinstance(value, dict):
        raise RepairError(f"JSON root is not an object: {label}")
    return value


def load_events(payload: bytes, *, label: str) -> list[dict[str, Any]]:
    if payload and not payload.endswith(b"\n"):
        raise RepairError(f"JSONL is not newline terminated: {label}")
    try:
        events = [json.loads(line) for line in payload.splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepairError(f"invalid JSONL: {label}") from exc
    if any(not isinstance(event, dict) for event in events):
        raise RepairError(f"JSONL contains a non-object: {label}")
    return events


def _expect(actual: Any, expected: Any, *, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise RepairError(f"{label} drift: expected {expected!r}, observed {actual!r}")


def validate_config(config: dict[str, Any], *, require_bound: bool) -> None:
    _expect(
        config.get("schema_version"),
        "route_a_v3_dec019_active_authority_runtime_metadata_repair.v1",
        label="schema version",
    )
    _expect(config.get("event_id"), "A1-EVT-043", label="event id")
    _expect(config.get("decision_id"), "V3-DEC-019", label="decision id")
    runtime = config.get("runtime")
    if not isinstance(runtime, dict):
        raise BindingError("runtime config is absent")
    for key, expected in {
        "predecessor_event_id": "A1-EVT-042",
        "predecessor_event_count": 42,
        "successor_event_id": "A1-EVT-043",
        "successor_event_count": 43,
        "predecessor_manifest_output_count": 163,
        "successor_manifest_output_count": 163,
        "mutable_publish_order": list(MUTABLE_NAMES),
        "authority_sync_name": "A1_DEC019_AUTHORITY_RUNTIME_SYNC_V1.json",
    }.items():
        _expect(runtime.get(key), expected, label=f"runtime {key}")

    desired = config.get("desired_active_authority")
    _expect(
        desired,
        {
            "implementation_commit": "d54de63605a2df51e91262c99218684a80cb6515",
            "binding_commit": "78827501c7efcef28550b04876c98206d94d4808",
            "scope": "DEC019_AUTHORITY_AND_SUCCESSOR_ADJUDICATOR_BINDING",
            "active_amendment_decision_ids": [
                "V3-DEC-017",
                "V3-DEC-018",
                "V3-DEC-019",
            ],
        },
        label="desired DEC019 authority",
    )
    predecessor = config.get("predecessor_metadata_truth")
    if not isinstance(predecessor, dict):
        raise BindingError("predecessor metadata truth is absent")
    _expect(
        predecessor.get("outer_active_authority_commit"),
        "d078060c81114687db5068902a5aad5d9bedbee6",
        label="historical outer authority",
    )
    policy = config.get("publication_policy")
    _expect(
        policy,
        {
            "manifest_outputs_unchanged": True,
            "prepared_pre_state_snapshots_are_not_runtime_outputs": True,
            "event_is_append_only_repair_record": True,
            "event_is_last_mutable_commit": True,
        },
        label="publication policy",
    )
    binding = config.get("implementation_binding")
    if not isinstance(binding, dict):
        raise BindingError("implementation binding is absent")
    status = binding.get("status")
    commit = binding.get("implementation_commit")
    if status == UNKNOWN and commit == UNKNOWN and not require_bound:
        return
    if status != "BOUND" or not isinstance(commit, str) or len(commit) != 40:
        raise BindingError("implementation is not BOUND to a 40-character commit")


def load_config(
    config_path: Path = PRODUCTION_CONFIG_PATH, *, require_bound: bool = True
) -> dict[str, Any]:
    config = load_json(config_path.read_bytes(), label=str(config_path))
    validate_config(config, require_bound=require_bound)
    return config


def _read_runtime(run_root: Path) -> dict[str, bytes]:
    try:
        return {name: (run_root / name).read_bytes() for name in MUTABLE_NAMES}
    except OSError as exc:
        raise PublicationError("cannot read the runtime mutables") from exc


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
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _authority_from_sync(run_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    sync_path = run_root / config["runtime"]["authority_sync_name"]
    try:
        sync = load_json(sync_path.read_bytes(), label=str(sync_path))
    except OSError as exc:
        raise PredecessorError("DEC019 runtime authority sync is absent") from exc
    try:
        authority = sync["runtime_authority"]["current_contract_authority"]
    except (KeyError, TypeError) as exc:
        raise PredecessorError("DEC019 runtime authority sync lacks current authority") from exc
    _expect(authority, config["desired_active_authority"], label="immutable DEC019 authority")
    return copy.deepcopy(authority)


def _parse_runtime(
    payloads: dict[str, bytes],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    return (
        load_json(payloads["STATUS.json"], label="STATUS.json"),
        load_json(payloads["RUN_MANIFEST.json"], label="RUN_MANIFEST.json"),
        load_events(payloads["EVENT_LOG.jsonl"], label="EVENT_LOG.jsonl"),
    )


def validate_predecessor(
    config: dict[str, Any], payloads: dict[str, bytes]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    status, manifest, events = _parse_runtime(payloads)
    runtime = config["runtime"]
    old = config["predecessor_metadata_truth"]
    if len(events) != runtime["predecessor_event_count"]:
        raise PredecessorError("predecessor event count is not 42")
    tail = events[-1] if events else {}
    _expect(tail.get("event_id"), "A1-EVT-042", label="predecessor tail event")
    _expect(tail.get("decision_id"), "V3-DEC-019", label="predecessor decision")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 163:
        raise PredecessorError("predecessor manifest output count is not 163")
    _expect(
        status.get("active_amendment_decision_ids"),
        old["status_active_amendment_decision_ids"],
        label="predecessor STATUS active amendments",
    )
    _expect(
        manifest.get("active_amendment_decision_ids"),
        old["manifest_active_amendment_decision_ids"],
        label="predecessor manifest active amendments",
    )
    manifest_authority = {
        "implementation_commit": manifest.get(
            "current_contract_authority_implementation_commit"
        ),
        "binding_commit": manifest.get("current_contract_authority_binding_commit"),
        "scope": manifest.get("current_contract_authority_scope"),
    }
    _expect(
        manifest_authority,
        old["manifest_current_contract_authority"],
        label="predecessor manifest current authority",
    )
    _expect(
        manifest.get("active_authority_commit"),
        old["outer_active_authority_commit"],
        label="predecessor outer active authority",
    )
    return status, manifest, events


def _validate_recorded_at(recorded_at: str, predecessor_at: Any) -> None:
    if not isinstance(predecessor_at, str):
        raise PredecessorError("predecessor event timestamp is absent")
    try:
        current = datetime.fromisoformat(recorded_at)
        predecessor = datetime.fromisoformat(predecessor_at)
    except ValueError as exc:
        raise RepairError("event timestamp is not ISO-8601") from exc
    if current.tzinfo is None or predecessor.tzinfo is None or current <= predecessor:
        raise RepairError("EVT043 timestamp must follow EVT042 with an explicit offset")


def _event_document(config: dict[str, Any], *, recorded_at: str) -> dict[str, Any]:
    old = config["predecessor_metadata_truth"]
    desired = config["desired_active_authority"]
    return {
        "event_id": "A1-EVT-043",
        "at": recorded_at,
        "phase_id": "A1",
        "event": config["event_name"],
        "decision_id": "V3-DEC-019",
        "predecessor_event_id": "A1-EVT-042",
        "correction_type": "APPEND_ONLY_RUNTIME_METADATA_REPAIR",
        "authority_source": config["runtime"]["authority_sync_name"],
        "corrected_fields": [
            "STATUS.active_amendment_decision_ids",
            "RUN_MANIFEST.active_amendment_decision_ids",
            "RUN_MANIFEST.current_contract_authority_implementation_commit",
            "RUN_MANIFEST.current_contract_authority_binding_commit",
            "RUN_MANIFEST.current_contract_authority_scope",
        ],
        "predecessor_metadata_truth": copy.deepcopy(old),
        "successor_active_authority": copy.deepcopy(desired),
        "outer_active_authority_commit_before": old["outer_active_authority_commit"],
        "outer_active_authority_commit_after": old["outer_active_authority_commit"],
        "historical_events_rewritten": False,
        "manifest_output_count_before": 163,
        "manifest_output_count_after": 163,
        "new_runtime_output_count": 0,
        "prepared_pre_state_snapshot_count": 3,
        "prepared_pre_state_snapshots_registered_as_runtime_outputs": False,
        "scientific_state_changed": False,
        "unchanged_scientific_truth": copy.deepcopy(
            config["unchanged_scientific_truth"]
        ),
        "detail": (
            "Corrected the stale DEC018 active-authority metadata to the DEC019 "
            "authority already frozen by A1_DEC019_AUTHORITY_RUNTIME_SYNC_V1.json. "
            "The historical outer active_authority_commit, 163 manifest outputs, "
            "four blockers, zero study contributions, false training/model-selection "
            "authorization, and NOT_ESTABLISHED scientific claim remain unchanged."
        ),
    }


def build_successors(
    config: dict[str, Any], predecessor_payloads: dict[str, bytes], recorded_at: str
) -> dict[str, bytes]:
    status, manifest, events = validate_predecessor(config, predecessor_payloads)
    _validate_recorded_at(recorded_at, events[-1].get("at"))
    desired = config["desired_active_authority"]
    successor_status = copy.deepcopy(status)
    successor_status["updated_at"] = recorded_at
    successor_status["active_amendment_decision_ids"] = copy.deepcopy(
        desired["active_amendment_decision_ids"]
    )
    successor_manifest = copy.deepcopy(manifest)
    successor_manifest["active_amendment_decision_ids"] = copy.deepcopy(
        desired["active_amendment_decision_ids"]
    )
    successor_manifest["current_contract_authority_implementation_commit"] = desired[
        "implementation_commit"
    ]
    successor_manifest["current_contract_authority_binding_commit"] = desired[
        "binding_commit"
    ]
    successor_manifest["current_contract_authority_scope"] = desired["scope"]
    artifacts = {
        "STATUS.json": json_bytes(successor_status),
        "RUN_MANIFEST.json": json_bytes(successor_manifest),
        "EVENT_LOG.jsonl": predecessor_payloads["EVENT_LOG.jsonl"]
        + compact_json_line(_event_document(config, recorded_at=recorded_at)),
    }
    validate_successors(config, predecessor_payloads, artifacts)
    return artifacts


def validate_successors(
    config: dict[str, Any],
    predecessor_payloads: dict[str, bytes],
    successors: dict[str, bytes],
) -> None:
    old_status, old_manifest, old_events = validate_predecessor(
        config, predecessor_payloads
    )
    status, manifest, events = _parse_runtime(successors)
    desired = config["desired_active_authority"]

    status_without_repairs = copy.deepcopy(status)
    status_without_repairs["active_amendment_decision_ids"] = old_status[
        "active_amendment_decision_ids"
    ]
    status_without_repairs["updated_at"] = old_status.get("updated_at")
    _expect(status_without_repairs, old_status, label="non-target STATUS fields")

    manifest_without_repairs = copy.deepcopy(manifest)
    manifest_without_repairs["active_amendment_decision_ids"] = old_manifest[
        "active_amendment_decision_ids"
    ]
    for field in (
        "implementation_commit",
        "binding_commit",
        "scope",
    ):
        manifest_without_repairs[f"current_contract_authority_{field}"] = old_manifest[
            f"current_contract_authority_{field}"
        ]
    _expect(manifest_without_repairs, old_manifest, label="non-target manifest fields")
    _expect(
        manifest.get("active_authority_commit"),
        old_manifest.get("active_authority_commit"),
        label="preserved outer active authority",
    )
    _expect(
        status.get("active_amendment_decision_ids"),
        desired["active_amendment_decision_ids"],
        label="successor STATUS active amendments",
    )
    _expect(
        manifest.get("active_amendment_decision_ids"),
        desired["active_amendment_decision_ids"],
        label="successor manifest active amendments",
    )
    for field in ("implementation_commit", "binding_commit", "scope"):
        _expect(
            manifest.get(f"current_contract_authority_{field}"),
            desired[field],
            label=f"successor manifest authority {field}",
        )
    _expect(manifest.get("outputs"), old_manifest.get("outputs"), label="manifest outputs")
    if len(events) != 43 or events[:-1] != old_events:
        raise RepairError("EVT043 is not one append-only event")
    _expect(events[-1], _event_document(config, recorded_at=events[-1].get("at")), label="EVT043")


def _prepared_path(prepared_directory: Path, config: dict[str, Any]) -> Path:
    prepared = Path(os.path.abspath(prepared_directory))
    allowed = Path(os.path.abspath(config["runtime"]["allowed_prepared_root"]))
    try:
        common = Path(os.path.commonpath((prepared, allowed)))
    except ValueError as exc:
        raise PublicationError("prepared directory is outside the allowed root") from exc
    if common != allowed or prepared == allowed:
        raise PublicationError("prepared directory must be below the allowed root")
    return prepared


def _atomic_replace(path: Path, payload: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _prepared_members(
    predecessor_payloads: dict[str, bytes], successors: dict[str, bytes]
) -> dict[str, bytes]:
    return {
        **{
            PREPARED_PREDECESSOR_NAMES[name]: predecessor_payloads[name]
            for name in MUTABLE_NAMES
        },
        **successors,
    }


def _write_prepared(prepared: Path, members: dict[str, bytes]) -> None:
    prepared.mkdir(exist_ok=True)
    observed = {item.name for item in prepared.iterdir()}
    if observed - set(members):
        raise PublicationError("prepared directory contains unexpected members")
    for name, payload in members.items():
        target = prepared / name
        if target.exists():
            if target.read_bytes() != payload:
                raise PublicationError(f"prepared member differs: {name}")
            continue
        _atomic_replace(target, payload)


def _read_prepared(prepared: Path) -> dict[str, bytes]:
    expected = set(MUTABLE_NAMES) | set(PREPARED_PREDECESSOR_NAMES.values())
    try:
        observed = {item.name for item in prepared.iterdir()}
    except OSError as exc:
        raise PublicationError("prepared directory is absent") from exc
    if observed != expected:
        raise PublicationError("prepared member set is incomplete or contains extras")
    return {name: (prepared / name).read_bytes() for name in expected}


def _split_prepared(
    members: dict[str, bytes],
) -> tuple[dict[str, bytes], dict[str, bytes]]:
    predecessor = {
        name: members[PREPARED_PREDECESSOR_NAMES[name]] for name in MUTABLE_NAMES
    }
    successor = {name: members[name] for name in MUTABLE_NAMES}
    return predecessor, successor


def _published_state_is_exact(config: dict[str, Any], payloads: dict[str, bytes]) -> bool:
    try:
        status, manifest, events = _parse_runtime(payloads)
    except RepairError:
        return False
    desired = config["desired_active_authority"]
    return (
        len(events) == 43
        and events[-1].get("event_id") == "A1-EVT-043"
        and events[-1].get("decision_id") == "V3-DEC-019"
        and status.get("active_amendment_decision_ids")
        == desired["active_amendment_decision_ids"]
        and manifest.get("active_amendment_decision_ids")
        == desired["active_amendment_decision_ids"]
        and manifest.get("current_contract_authority_implementation_commit")
        == desired["implementation_commit"]
        and manifest.get("current_contract_authority_binding_commit")
        == desired["binding_commit"]
        and manifest.get("current_contract_authority_scope") == desired["scope"]
        and manifest.get("active_authority_commit")
        == config["predecessor_metadata_truth"]["outer_active_authority_commit"]
        and isinstance(manifest.get("outputs"), list)
        and len(manifest["outputs"]) == 163
    )


def prepare_repair(
    *,
    prepared_directory: Path,
    recorded_at: str,
    config_override: dict[str, Any] | None = None,
    run_root_override: Path | None = None,
) -> dict[str, Any]:
    config = config_override or load_config()
    validate_config(config, require_bound=True)
    run_root = run_root_override or Path(config["runtime"]["run_root"])
    prepared = _prepared_path(prepared_directory, config)
    with _locked_run(run_root):
        predecessor = _read_runtime(run_root)
        _authority_from_sync(run_root, config)
        if _published_state_is_exact(config, predecessor):
            return {"status": "ALREADY_PUBLISHED_VERIFIED", "event_id": "A1-EVT-043"}
        successors = build_successors(config, predecessor, recorded_at)
    prepared.parent.mkdir(parents=True, exist_ok=True)
    _write_prepared(prepared, _prepared_members(predecessor, successors))
    return {
        "status": "PREPARED_NOT_PUBLISHED",
        "event_id": "A1-EVT-043",
        "prepared_directory": str(prepared),
        "prepared_member_count": 6,
        "manifest_output_transition": "163_TO_163",
        "new_runtime_output_count": 0,
    }


def publish_prepared(
    *,
    prepared_directory: Path,
    config_override: dict[str, Any] | None = None,
    run_root_override: Path | None = None,
    fault_injector: FaultInjector | None = None,
) -> dict[str, Any]:
    config = config_override or load_config()
    validate_config(config, require_bound=True)
    run_root = run_root_override or Path(config["runtime"]["run_root"])
    prepared = _prepared_path(prepared_directory, config)
    predecessor, successor = _split_prepared(_read_prepared(prepared))
    validate_successors(config, predecessor, successor)

    with _locked_run(run_root):
        _authority_from_sync(run_root, config)
        current = _read_runtime(run_root)
        states = []
        for name in MUTABLE_NAMES:
            if current[name] == predecessor[name]:
                states.append("OLD")
            elif current[name] == successor[name]:
                states.append("NEW")
            else:
                raise PredecessorError(f"runtime mutable is neither predecessor nor successor: {name}")
        allowed = (["OLD", "OLD", "OLD"], ["NEW", "OLD", "OLD"], ["NEW", "NEW", "OLD"], ["NEW", "NEW", "NEW"])
        if states not in allowed:
            raise PredecessorError(f"runtime mutable order is not recoverable: {states!r}")
        if states == ["NEW", "NEW", "NEW"]:
            return {"status": "PUBLISHED_VERIFIED", "event_id": "A1-EVT-043", "reused": True}

        try:
            for index, name in enumerate(MUTABLE_NAMES):
                if states[index] == "NEW":
                    continue
                if fault_injector is not None:
                    fault_injector(f"before_replace:{name}")
                _atomic_replace(run_root / name, successor[name])
                states[index] = "NEW"
        except Exception as exc:
            after = _read_runtime(run_root)
            if all(after[name] == successor[name] for name in MUTABLE_NAMES):
                return {
                    "status": "PUBLISHED_VERIFIED_AFTER_RECHECK",
                    "event_id": "A1-EVT-043",
                }
            raise PublicationError(
                "EVT043 was not committed; retry with the same prepared directory"
            ) from exc

        final = _read_runtime(run_root)
        if final != successor:
            raise PublicationError("EVT043 publication finished with non-exact mutables")
        return {"status": "PUBLISHED_VERIFIED", "event_id": "A1-EVT-043", "reused": False}


def validate_published(
    *,
    prepared_directory: Path,
    config_override: dict[str, Any] | None = None,
    run_root_override: Path | None = None,
) -> dict[str, Any]:
    config = config_override or load_config()
    validate_config(config, require_bound=True)
    run_root = run_root_override or Path(config["runtime"]["run_root"])
    prepared = _prepared_path(prepared_directory, config)
    predecessor, successor = _split_prepared(_read_prepared(prepared))
    validate_successors(config, predecessor, successor)
    with _locked_run(run_root):
        _authority_from_sync(run_root, config)
        current = _read_runtime(run_root)
    if current != successor:
        raise PublicationError("runtime does not exactly match the prepared EVT043 successor")
    return {"status": "PUBLISHED_VERIFIED", "event_id": "A1-EVT-043"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--prepared-directory", type=Path, required=True)
    prepare_parser.add_argument("--recorded-at", required=True)
    for command in ("publish", "validate"):
        child = subparsers.add_parser(command)
        child.add_argument("--prepared-directory", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "prepare":
        result = prepare_repair(
            prepared_directory=arguments.prepared_directory,
            recorded_at=arguments.recorded_at,
        )
    elif arguments.command == "publish":
        result = publish_prepared(prepared_directory=arguments.prepared_directory)
    else:
        result = validate_published(prepared_directory=arguments.prepared_directory)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
