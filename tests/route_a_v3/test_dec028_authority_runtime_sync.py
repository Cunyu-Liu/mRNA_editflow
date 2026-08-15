from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = STAGING_ROOT / "configs/route_a_v3_dec028_authority_runtime_sync_v1.json"
SCRIPT_PATH = STAGING_ROOT / "scripts/route_a_v3/dec028_authority_runtime_sync.py"
SPEC = importlib.util.spec_from_file_location("dec028_authority_runtime_sync", SCRIPT_PATH)
assert SPEC and SPEC.loader
SYNC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SYNC)


def disk_config() -> dict[str, Any]:
    return SYNC.load_config(CONFIG_PATH)


def synthetic_config(tmp_path: Path) -> dict[str, Any]:
    config = copy.deepcopy(disk_config())
    config["owner_activation"].update(
        {
            "status": "SYNTHETIC_TEST_ONLY",
            "decision_issuance_reference": "SYNTHETIC_FIXTURE_NOT_PRODUCTION",
            "issued_at": "2026-08-15T00:00:00+08:00",
            "independent_review_status": "SYNTHETIC_FIXTURE_NOT_PRODUCTION",
            "independent_review_reference": "SYNTHETIC_FIXTURE_NOT_PRODUCTION",
        }
    )
    config["implementation_binding"].update(
        {
            "status": "SYNTHETIC_TEST_ONLY",
            "implementation_commit": "SYNTHETIC_FIXTURE_NOT_PRODUCTION",
            "implementation_script_sha256": "SYNTHETIC_FIXTURE_NOT_PRODUCTION",
            "implementation_test_sha256": "SYNTHETIC_FIXTURE_NOT_PRODUCTION",
        }
    )
    config["production_repository"].update(
        {
            "status": "SYNTHETIC_TEST_ONLY",
            "repository_root": "SYNTHETIC_FIXTURE_NOT_PRODUCTION",
            "branch": "SYNTHETIC_FIXTURE_NOT_PRODUCTION",
            "expected_head_commit": "SYNTHETIC_FIXTURE_NOT_PRODUCTION",
        }
    )
    config["runtime"]["run_root"] = str(tmp_path / "run")
    config["runtime"]["allowed_prepared_root"] = str(tmp_path / "prepared")
    SYNC.validate_static_config(config)
    return config


def outer_fields() -> dict[str, Any]:
    return {
        "qualified_ordinary_studies": 1,
        "qualified_a1_studies": 1,
        "qualified_a2_dense_studies": 0,
        "canonical_intervention_record_count": 6547,
        "canonical_record_count": 6547,
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


def synthetic_predecessor(event_count: int = 60) -> dict[str, bytes]:
    assert event_count >= 1
    status = {
        **outer_fields(),
        "updated_at": "2026-08-14T18:30:00+08:00",
        "active_amendment_decision_ids": list(SYNC.BEFORE_DECISION_IDS),
        "existing_status_field": "PRESERVED",
    }
    manifest = {
        **outer_fields(),
        "active_authority_commit": "f" * 40,
        "active_amendment_decision_ids": list(SYNC.BEFORE_DECISION_IDS),
        "outputs": [
            {
                "absolute_path": f"/synthetic/existing-{index:03d}.json",
                "artifact_type": "SYNTHETIC_EXISTING",
                "bytes": index + 1,
                "sha256": f"{index + 1:064x}",
            }
            for index in range(5)
        ],
        "existing_manifest_field": {"preserve": True},
    }
    events: list[dict[str, Any]] = []
    for index in range(1, event_count + 1):
        event: dict[str, Any] = {
            "event_id": f"A1-EVT-{index:03d}",
            "at": "2026-08-14T18:30:00+08:00",
            "phase_id": "A1",
            "event": "SYNTHETIC_HISTORICAL_EVENT",
        }
        if index == event_count:
            event["decision_id"] = "V3-DEC-027"
        else:
            event["decision_id"] = "V3-DEC-024"
        events.append(event)
    return {
        "STATUS.json": SYNC.json_bytes(status),
        "RUN_MANIFEST.json": SYNC.json_bytes(manifest),
        "EVENT_LOG.jsonl": b"".join(SYNC.json_line(event) for event in events),
    }


def make_context(tmp_path: Path, event_count: int = 60) -> tuple[dict[str, Any], dict[str, bytes], Path, Path]:
    config = synthetic_config(tmp_path)
    run_root = Path(config["runtime"]["run_root"])
    prepared_root = Path(config["runtime"]["allowed_prepared_root"])
    prepared = prepared_root / "dec028-sync"
    run_root.mkdir()
    prepared_root.mkdir()
    predecessor = synthetic_predecessor(event_count)
    for name, payload in predecessor.items():
        (run_root / name).write_bytes(payload)
    return config, predecessor, run_root, prepared


def read_runtime(run_root: Path) -> dict[str, bytes]:
    return {name: (run_root / name).read_bytes() for name in SYNC.MUTABLE_NAMES}


def test_disk_config_is_static_only_and_public_prepare_stops_before_runtime_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = disk_config()
    assert config["static_authority"]["status"] == "PENDING_STATIC_AUTHORITY_ONLY"
    assert config["owner_activation"]["status"] == "OWNER_ISSUANCE_REQUIRED"
    assert config["implementation_binding"]["status"] == "UNKNOWN_NOT_ASSERTED"
    assert config["production_repository"]["status"] == "UNKNOWN_NOT_ASSERTED"

    monkeypatch.setattr(SYNC, "_read_runtime", lambda *_args: pytest.fail("runtime must not be read"))
    with pytest.raises(SYNC.AuthorityError, match="owner issuance"):
        SYNC.prepare_runtime_sync(
            prepared_directory=tmp_path / "must-not-exist",
            recorded_at="2026-08-15T02:00:00+08:00",
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_successor_event_is_derived_from_live_tail_and_preserves_counts_and_locks(tmp_path: Path) -> None:
    config = synthetic_config(tmp_path)
    predecessor = synthetic_predecessor(60)
    successors = SYNC.build_successors(config, predecessor, "2026-08-15T02:00:00+08:00")
    status, manifest, events = SYNC._parse_runtime(
        {name: successors[name] for name in SYNC.MUTABLE_NAMES}
    )
    event = events[-1]
    assert event["event_id"] == "A1-EVT-061"
    assert event["predecessor_event_id"] == "A1-EVT-060"
    assert event["decision_id"] == "V3-DEC-028"
    assert event["scientific_state_changed"] is False
    assert event["qualification_changed"] is False
    assert event["current_qualified_counts"] == SYNC.FROZEN_COUNTS
    assert all(value is False for value in event["locks"].values())
    assert status["active_amendment_decision_ids"] == list(SYNC.AFTER_DECISION_IDS)
    assert manifest["active_amendment_decision_ids"] == list(SYNC.AFTER_DECISION_IDS)
    assert manifest["active_authority_commit"] == "f" * 40
    assert len(manifest["outputs"]) == 9
    assert len(successors) == 7

    later_predecessor = synthetic_predecessor(64)
    later = SYNC.build_successors(config, later_predecessor, "2026-08-15T02:00:00+08:00")
    later_events = SYNC.load_events(later["EVENT_LOG.jsonl"], label="later")
    assert later_events[-1]["event_id"] == "A1-EVT-065"
    assert later_events[-1]["predecessor_event_id"] == "A1-EVT-064"


def test_pure_builder_rejects_any_static_lock_relaxation_before_successor_generation(tmp_path: Path) -> None:
    config = synthetic_config(tmp_path)
    config["static_authority"]["locks"]["cuda_probe_allowed"] = True
    with pytest.raises(SYNC.RuntimeSyncError, match="static locks"):
        SYNC.build_successors(config, synthetic_predecessor(), "2026-08-15T02:00:00+08:00")


def test_illegal_or_changed_predecessor_stops_before_prepared_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, predecessor, run_root, prepared = make_context(tmp_path)
    changed = copy.deepcopy(predecessor)
    changed["STATUS.json"] += b"drift"
    original_read = SYNC._read_runtime
    calls = 0

    def changing_read(root: Path) -> dict[str, bytes]:
        nonlocal calls
        calls += 1
        if calls == 2:
            return changed
        return original_read(root)

    monkeypatch.setattr(SYNC, "_read_runtime", changing_read)
    with pytest.raises(SYNC.PredecessorError, match="changed before prepared"):
        SYNC.prepare_runtime_sync(
            prepared_directory=prepared,
            recorded_at="2026-08-15T02:00:00+08:00",
            config_override=config,
            run_root_override=run_root,
            production=False,
        )
    assert not prepared.exists()

    illegal = synthetic_predecessor(60)
    events = SYNC.load_events(illegal["EVENT_LOG.jsonl"], label="illegal")
    events[-1]["decision_id"] = "V3-DEC-024"
    illegal["EVENT_LOG.jsonl"] = b"".join(SYNC.json_line(event) for event in events)
    with pytest.raises(SYNC.PredecessorError, match="DEC027"):
        SYNC.build_successors(config, illegal, "2026-08-15T02:00:00+08:00")


def test_prepare_publish_uses_immutable_first_ordered_prefix_recovery(tmp_path: Path) -> None:
    config, predecessor, run_root, prepared = make_context(tmp_path)
    prepared_result = SYNC.prepare_runtime_sync(
        prepared_directory=prepared,
        recorded_at="2026-08-15T02:00:00+08:00",
        config_override=config,
        run_root_override=run_root,
        production=False,
    )
    assert prepared_result["status"] == "PREPARED_NOT_PUBLISHED"
    assert prepared_result["event_id"] == "A1-EVT-061"
    assert len(list(prepared.iterdir())) == 7

    stages: list[str] = []

    def fail_before_manifest(stage: str) -> None:
        stages.append(stage)
        if stage == "before_replace:RUN_MANIFEST.json":
            raise RuntimeError("synthetic interruption")

    with pytest.raises(SYNC.PublicationError, match="retry"):
        SYNC.publish_prepared(
            prepared_directory=prepared,
            config_override=config,
            run_root_override=run_root,
            production=False,
            fault_injector=fail_before_manifest,
        )
    assert stages[:4] == [f"before_immutable:{name}" for name in config["runtime"]["immutable_publish_order"]]
    assert stages[4:] == ["before_replace:STATUS.json", "before_replace:RUN_MANIFEST.json"]
    assert (run_root / "STATUS.json").read_bytes() == (prepared / "STATUS.json").read_bytes()
    assert (run_root / "RUN_MANIFEST.json").read_bytes() == predecessor["RUN_MANIFEST.json"]
    assert (run_root / "EVENT_LOG.jsonl").read_bytes() == predecessor["EVENT_LOG.jsonl"]

    published = SYNC.publish_prepared(
        prepared_directory=prepared,
        config_override=config,
        run_root_override=run_root,
        production=False,
    )
    assert published["status"] == "PUBLISHED_VERIFIED"
    assert published["event_id"] == "A1-EVT-061"
    assert SYNC.validate_published(
        prepared_directory=prepared,
        config_override=config,
        run_root_override=run_root,
        production=False,
    ) == {"status": "PUBLISHED_VERIFIED", "event_id": "A1-EVT-061"}
    assert len(SYNC.load_events(read_runtime(run_root)["EVENT_LOG.jsonl"], label="published")) == 61


def test_tampered_prepared_lock_or_claim_is_rejected_before_publish_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, predecessor, run_root, prepared = make_context(tmp_path)
    SYNC.prepare_runtime_sync(
        prepared_directory=prepared,
        recorded_at="2026-08-15T02:00:00+08:00",
        config_override=config,
        run_root_override=run_root,
        production=False,
    )
    events = SYNC.load_events((prepared / "EVENT_LOG.jsonl").read_bytes(), label="prepared")
    events[-1]["training_allowed"] = True
    events[-1]["scientific_claim_status"] = "ESTABLISHED"
    (prepared / "EVENT_LOG.jsonl").write_bytes(b"".join(SYNC.json_line(event) for event in events))
    writes: list[str] = []
    monkeypatch.setattr(SYNC, "_write_immutable_once", lambda *_args, **_kwargs: writes.append("immutable"))
    monkeypatch.setattr(SYNC, "_write_atomic", lambda *_args, **_kwargs: writes.append("runtime"))
    with pytest.raises(SYNC.RuntimeSyncError, match="closure"):
        SYNC.publish_prepared(
            prepared_directory=prepared,
            config_override=config,
            run_root_override=run_root,
            production=False,
        )
    assert writes == []
    assert read_runtime(run_root) == predecessor
