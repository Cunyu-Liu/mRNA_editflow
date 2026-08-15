from __future__ import annotations

import copy
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/route_a_v3_dec028_authority_runtime_sync_v1.json"
SCRIPT_PATH = ROOT / "scripts/route_a_v3/dec028_authority_runtime_sync.py"
SPEC = importlib.util.spec_from_file_location("dec028_authority_runtime_sync", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)


def _disk_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def _bound_config() -> dict:
    config = _disk_config()
    binding = config["implementation_binding"]
    binding.update(
        {
            "status": "BOUND",
            "implementation_commit": "1" * 40,
            "implementation_script_sha256": "2" * 64,
            "implementation_test_sha256": "3" * 64,
        }
    )
    return config


def _fixture() -> tuple[dict, dict[str, bytes]]:
    config = _bound_config()
    status = copy.deepcopy(config["frozen_outer_truth"])
    status["updated_at"] = "2026-08-15T05:35:00+08:00"
    manifest = copy.deepcopy(config["frozen_outer_truth"])
    manifest.update(
        {
            "active_authority_commit": "a" * 40,
            "registered_artifact_count": 14,
            "outputs": [
                {"absolute_path": f"/runtime/existing_{index:03d}.json"}
                for index in range(266)
            ],
        }
    )
    start = datetime(2026, 8, 14, 0, 0, tzinfo=timezone(timedelta(hours=8)))
    events = []
    for index in range(1, 61):
        events.append(
            {
                "event_id": f"A1-EVT-{index:03d}",
                "at": (start + timedelta(minutes=index)).isoformat(),
                "decision_id": "V3-DEC-027" if index == 60 else "V3-DEC-024",
            }
        )
    payloads = {
        "STATUS.json": sync.json_bytes(status),
        "RUN_MANIFEST.json": sync.json_bytes(manifest),
        "EVENT_LOG.jsonl": b"".join(sync.compact_json_line(event) for event in events),
    }
    for name, payload in payloads.items():
        spec = config["runtime"]["predecessor_mutables"][name]
        spec["bytes"] = len(payload)
        spec["sha256"] = sync.sha256(payload)
    tail = payloads["EVENT_LOG.jsonl"].splitlines(keepends=True)[-1]
    config["runtime"]["predecessor_tail"].update(
        {"bytes": len(tail), "sha256": sync.sha256(tail)}
    )
    sync.validate_static_config(config)
    return config, payloads


def _materialize_runtime(root: Path, payloads: dict[str, bytes]) -> None:
    root.mkdir(parents=True)
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)


def test_disk_candidate_is_legal_i2_or_b2_and_normalizes_to_unbound_i2() -> None:
    config = _disk_config()
    sync.validate_static_config(config)
    assert sync._binding_state(config["implementation_binding"]) in {"UNKNOWN", "BOUND"}
    normalized = sync.normalized_unknown_i_config(config)
    assert sync._binding_state(normalized["implementation_binding"]) == "UNKNOWN"
    assert normalized["implementation_binding"]["frozen_predecessor_implementation"]["implementation_commit"] == sync.I1_COMMIT
    assert normalized["dec028_authority"]["current_qualified_counts"] == {
        "ordinary": 1,
        "a1": 1,
        "true_a2": 0,
        "canonical_records": 6547,
    }
    assert all(value == 0 for value in normalized["access_boundary"].values())


def test_partial_binding_is_rejected() -> None:
    config = sync.normalized_unknown_i_config(_disk_config())
    config["implementation_binding"]["status"] = "BOUND"
    with pytest.raises(sync.BindingError, match="partially known"):
        sync.validate_static_config(config)


def test_successor_is_exact_evt061_authority_only() -> None:
    config, predecessor = _fixture()
    result = sync.build_successors(
        config,
        predecessor,
        "2026-08-15T06:00:00+08:00",
        authority_audit={"status": "TEST_AUDIT", "authority_commit": sync.AUTHORITY_COMMIT},
    )
    status = sync.load_json(result["STATUS.json"], label="status")
    manifest = sync.load_json(result["RUN_MANIFEST.json"], label="manifest")
    events = sync.load_events(result["EVENT_LOG.jsonl"], label="events")
    assert len(events) == 61
    assert events[-1]["event_id"] == "A1-EVT-061"
    assert events[-1]["decision_id"] == "V3-DEC-028"
    assert events[-1]["p0_executed"] is False
    assert events[-1]["g1_executed"] is False
    assert len(manifest["outputs"]) == 270
    assert manifest["registered_artifact_count"] == 14
    assert status["dec028_successor_p0_status"] == "AUTHORIZED_NOT_RUN"
    for key, expected in config["frozen_outer_truth"].items():
        assert status[key] == expected
        assert manifest[key] == expected


def test_unchecked_event_or_lock_mutation_is_rejected() -> None:
    config, predecessor = _fixture()
    result = sync.build_successors(config, predecessor, "2026-08-15T06:00:00+08:00")
    events = sync.load_events(result["EVENT_LOG.jsonl"], label="events")
    events[-1]["training_authorized"] = True
    mutated = dict(result)
    mutated["EVENT_LOG.jsonl"] = b"".join(sync.compact_json_line(event) for event in events)
    with pytest.raises(sync.RuntimeSyncError, match="byte closure"):
        sync.validate_successors(config, predecessor, mutated)


def test_production_requires_fresh_repository_audit() -> None:
    config, predecessor = _fixture()
    with pytest.raises(sync.AuthorityError, match="fresh production"):
        sync.build_successors(
            config,
            predecessor,
            "2026-08-15T06:00:00+08:00",
            production=True,
        )


def test_predecessor_cas_or_timestamp_drift_stops() -> None:
    config, predecessor = _fixture()
    broken = dict(predecessor)
    broken["STATUS.json"] += b" "
    with pytest.raises(sync.PredecessorError, match="identity drift"):
        sync.validate_predecessor(config, broken)
    with pytest.raises(sync.PredecessorError, match="must follow"):
        sync.build_successors(
            config, predecessor, "2026-08-14T00:00:00+08:00"
        )


def test_unknown_binding_stops_before_runtime_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _disk_config()
    touched = {"runtime": 0}

    def forbidden(_: Path) -> dict[str, bytes]:
        touched["runtime"] += 1
        raise AssertionError("runtime must not be touched")

    monkeypatch.setattr(sync.base, "_read_runtime", forbidden)
    with pytest.raises(sync.BindingError, match="not BOUND"):
        sync.prepare_runtime_sync(
            prepared_directory=tmp_path / "prepared" / "attempt",
            recorded_at="2026-08-15T06:00:00+08:00",
            production=False,
            config_override=config,
            run_root_override=tmp_path / "runtime",
        )
    assert touched == {"runtime": 0}


def test_prepare_publish_validate_and_idempotent_retry(tmp_path: Path) -> None:
    config, predecessor = _fixture()
    run_root = tmp_path / "runtime"
    prepared_parent = tmp_path / "prepared"
    prepared = prepared_parent / "evt061"
    _materialize_runtime(run_root, predecessor)
    config["runtime"]["allowed_prepared_root"] = str(prepared_parent)
    result = sync.prepare_runtime_sync(
        prepared_directory=prepared,
        recorded_at="2026-08-15T06:00:00+08:00",
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    assert result["status"] == "PREPARED_NOT_PUBLISHED"
    first = sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    assert first == {"status": "PUBLISHED_VERIFIED", "event_id": "A1-EVT-061", "reused": False}
    second = sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    assert second["reused"] is True
    assert sync.validate_published(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        run_root_override=run_root,
    )["status"] == "PUBLISHED_VERIFIED"


def test_mutable_prefix_recovery_is_exact(tmp_path: Path) -> None:
    config, predecessor = _fixture()
    run_root = tmp_path / "runtime"
    prepared_parent = tmp_path / "prepared"
    prepared = prepared_parent / "evt061"
    _materialize_runtime(run_root, predecessor)
    config["runtime"]["allowed_prepared_root"] = str(prepared_parent)
    sync.prepare_runtime_sync(
        prepared_directory=prepared,
        recorded_at="2026-08-15T06:00:00+08:00",
        production=False,
        config_override=config,
        run_root_override=run_root,
    )

    def fail_before_manifest(label: str) -> None:
        if label == "before_replace:RUN_MANIFEST.json":
            raise RuntimeError("injected")

    with pytest.raises(RuntimeError, match="injected"):
        sync.publish_prepared(
            prepared_directory=prepared,
            production=False,
            config_override=config,
            run_root_override=run_root,
            fault_injector=fail_before_manifest,
        )
    result = sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    assert result["status"] == "PUBLISHED_VERIFIED"


def test_prepared_member_or_existing_immutable_drift_stops(tmp_path: Path) -> None:
    config, predecessor = _fixture()
    run_root = tmp_path / "runtime"
    prepared_parent = tmp_path / "prepared"
    prepared = prepared_parent / "evt061"
    _materialize_runtime(run_root, predecessor)
    config["runtime"]["allowed_prepared_root"] = str(prepared_parent)
    sync.prepare_runtime_sync(
        prepared_directory=prepared,
        recorded_at="2026-08-15T06:00:00+08:00",
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    (prepared / config["runtime"]["sync_name"]).write_bytes(b"{}\n")
    with pytest.raises(sync.RuntimeSyncError):
        sync.publish_prepared(
            prepared_directory=prepared,
            production=False,
            config_override=config,
            run_root_override=run_root,
        )
