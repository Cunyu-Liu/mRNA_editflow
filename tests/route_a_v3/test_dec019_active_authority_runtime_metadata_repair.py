from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    STAGING_ROOT
    / "configs/route_a_v3_dec019_active_authority_runtime_metadata_repair_v1.json"
)
SCRIPT_PATH = (
    STAGING_ROOT
    / "scripts/route_a_v3/dec019_active_authority_runtime_metadata_repair.py"
)
SPEC = importlib.util.spec_from_file_location("evt043_repair", SCRIPT_PATH)
assert SPEC and SPEC.loader
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


def read_bound_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text())
    config["implementation_binding"] = {
        "status": "BOUND",
        "implementation_commit": "1" * 40,
    }
    repair.validate_config(config, require_bound=True)
    return config


def predecessor_payloads(config: dict[str, Any]) -> dict[str, bytes]:
    old = config["predecessor_metadata_truth"]
    status = {
        "run_status": "IN_PROGRESS",
        "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "gate_status": "A1_BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "claim_status": "NOT_ESTABLISHED",
        "active_amendment_decision_ids": copy.deepcopy(
            old["status_active_amendment_decision_ids"]
        ),
        "updated_at": "2026-08-12T03:29:11+08:00",
        "training_started": False,
        "training_allowed": False,
        "training_authorized": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "historical_field": "PRESERVED",
    }
    authority = old["manifest_current_contract_authority"]
    manifest = {
        "run_status": "IN_PROGRESS",
        "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "claim_status": "NOT_ESTABLISHED",
        "active_amendment_decision_ids": copy.deepcopy(
            old["manifest_active_amendment_decision_ids"]
        ),
        "active_authority_commit": old["outer_active_authority_commit"],
        "current_contract_authority_implementation_commit": authority[
            "implementation_commit"
        ],
        "current_contract_authority_binding_commit": authority["binding_commit"],
        "current_contract_authority_scope": authority["scope"],
        "outputs": [
            {
                "artifact_type": f"EXISTING_{index:03d}",
                "absolute_path": f"/existing/{index:03d}",
                "sha256": f"{index:064x}",
                "status": "COMPLETE",
            }
            for index in range(163)
        ],
        "historical_field": "PRESERVED",
    }
    events = [
        {
            "event_id": f"A1-EVT-{index:03d}",
            "at": "2026-08-10T00:00:00+08:00",
            "event": "HISTORICAL",
        }
        for index in range(1, 42)
    ]
    events.append(
        {
            "event_id": "A1-EVT-042",
            "at": "2026-08-12T03:29:11+08:00",
            "event": "GSE200304_DEC019_UPSTREAM_PASS_RUNTIME_SYNC",
            "decision_id": "V3-DEC-019",
            "training_started": False,
            "training_allowed": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
        }
    )
    return {
        "STATUS.json": repair.json_bytes(status),
        "RUN_MANIFEST.json": repair.json_bytes(manifest),
        "EVENT_LOG.jsonl": b"".join(repair.compact_json_line(event) for event in events),
    }


def make_context(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, bytes], Path, Path]:
    config = read_bound_config()
    run_root = tmp_path / "run"
    allowed = tmp_path / "allowed"
    prepared = allowed / "evt043-job"
    config["runtime"]["run_root"] = str(run_root)
    config["runtime"]["allowed_prepared_root"] = str(allowed)
    run_root.mkdir()
    allowed.mkdir()
    predecessor = predecessor_payloads(config)
    for name, payload in predecessor.items():
        (run_root / name).write_bytes(payload)
    (run_root / config["runtime"]["authority_sync_name"]).write_bytes(
        repair.json_bytes(
            {
                "record_type": "ROUTE_A_V3_A1_DEC019_AUTHORITY_RUNTIME_SYNC",
                "runtime_authority": {
                    "current_contract_authority": copy.deepcopy(
                        config["desired_active_authority"]
                    )
                },
            }
        )
    )
    return config, predecessor, run_root, prepared


def read_runtime(run_root: Path) -> dict[str, bytes]:
    return {name: (run_root / name).read_bytes() for name in repair.MUTABLE_NAMES}


def prepare_context(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, bytes], Path, Path]:
    context = make_context(tmp_path)
    config, _predecessor, run_root, prepared = context
    result = repair.prepare_repair(
        prepared_directory=prepared,
        recorded_at="2026-08-12T04:00:00+08:00",
        config_override=config,
        run_root_override=run_root,
    )
    assert result == {
        "status": "PREPARED_NOT_PUBLISHED",
        "event_id": "A1-EVT-043",
        "prepared_directory": str(prepared),
        "prepared_member_count": 6,
        "manifest_output_transition": "163_TO_163",
        "new_runtime_output_count": 0,
    }
    return context


def test_normal_repair_uses_dec019_authority_and_keeps_science_and_outputs(
    tmp_path: Path,
) -> None:
    # This detects a repair that updates the wrong fields or accidentally creates science/output changes.
    config, predecessor, run_root, prepared = prepare_context(tmp_path)
    result = repair.publish_prepared(
        prepared_directory=prepared,
        config_override=config,
        run_root_override=run_root,
    )
    assert result["status"] == "PUBLISHED_VERIFIED"
    old_status, old_manifest, old_events = repair._parse_runtime(predecessor)
    status, manifest, events = repair._parse_runtime(read_runtime(run_root))
    desired = config["desired_active_authority"]
    assert status["active_amendment_decision_ids"] == desired[
        "active_amendment_decision_ids"
    ]
    assert manifest["active_amendment_decision_ids"] == desired[
        "active_amendment_decision_ids"
    ]
    assert manifest["current_contract_authority_implementation_commit"] == desired[
        "implementation_commit"
    ]
    assert manifest["current_contract_authority_binding_commit"] == desired[
        "binding_commit"
    ]
    assert manifest["current_contract_authority_scope"] == desired["scope"]
    assert manifest["active_authority_commit"] == old_manifest[
        "active_authority_commit"
    ]
    assert manifest["outputs"] == old_manifest["outputs"]
    assert old_status["training_started"] is False
    assert status["training_started"] is False
    assert events[:-1] == old_events
    assert events[-1]["new_runtime_output_count"] == 0
    assert events[-1]["scientific_state_changed"] is False
    truth = events[-1]["unchanged_scientific_truth"]
    assert truth["unresolved_blockers"] == config["unchanged_scientific_truth"][
        "unresolved_blockers"
    ]
    assert truth["ordinary_study_contribution"] == 0
    assert truth["a1_study_contribution"] == 0
    assert truth["true_a2_study_contribution"] == 0
    assert truth["qualified"] is False
    assert truth["training_started"] is False
    assert truth["training_allowed"] is False
    assert truth["model_selection_allowed"] is False


def test_predecessor_mismatch_stops_before_prepared_or_runtime_write(
    tmp_path: Path,
) -> None:
    # This detects applying EVT043 to a runtime other than the real EVT042 predecessor.
    config, _predecessor, run_root, prepared = make_context(tmp_path)
    status = repair.load_json((run_root / "STATUS.json").read_bytes(), label="status")
    status["active_amendment_decision_ids"] = ["V3-DEC-017"]
    (run_root / "STATUS.json").write_bytes(repair.json_bytes(status))
    before = read_runtime(run_root)
    with pytest.raises(repair.RepairError, match="predecessor STATUS active amendments"):
        repair.prepare_repair(
            prepared_directory=prepared,
            recorded_at="2026-08-12T04:00:00+08:00",
            config_override=config,
            run_root_override=run_root,
        )
    assert read_runtime(run_root) == before
    assert not prepared.exists()


def test_event_last_failure_is_not_reported_as_success_and_retry_completes(
    tmp_path: Path,
) -> None:
    # This detects claiming success after STATUS/manifest changed but EVT043 was not appended.
    config, predecessor, run_root, prepared = prepare_context(tmp_path)

    def fail_before_event(point: str) -> None:
        if point == "before_replace:EVENT_LOG.jsonl":
            raise OSError("injected event write failure")

    with pytest.raises(repair.PublicationError, match="was not committed"):
        repair.publish_prepared(
            prepared_directory=prepared,
            config_override=config,
            run_root_override=run_root,
            fault_injector=fail_before_event,
        )
    partial = read_runtime(run_root)
    assert partial["EVENT_LOG.jsonl"] == predecessor["EVENT_LOG.jsonl"]
    assert partial["STATUS.json"] != predecessor["STATUS.json"]
    assert partial["RUN_MANIFEST.json"] != predecessor["RUN_MANIFEST.json"]

    recovered = repair.publish_prepared(
        prepared_directory=prepared,
        config_override=config,
        run_root_override=run_root,
    )
    assert recovered["status"] == "PUBLISHED_VERIFIED"
    assert repair.load_events(
        (run_root / "EVENT_LOG.jsonl").read_bytes(), label="event"
    )[-1]["event_id"] == "A1-EVT-043"


def test_publish_and_prepare_are_idempotent_after_existing_evt043(
    tmp_path: Path,
) -> None:
    # This detects duplicate EVT043 appends or mutation on an ordinary operator retry.
    config, _predecessor, run_root, prepared = prepare_context(tmp_path)
    repair.publish_prepared(
        prepared_directory=prepared,
        config_override=config,
        run_root_override=run_root,
    )
    published = read_runtime(run_root)
    second = repair.publish_prepared(
        prepared_directory=prepared,
        config_override=config,
        run_root_override=run_root,
    )
    assert second == {
        "status": "PUBLISHED_VERIFIED",
        "event_id": "A1-EVT-043",
        "reused": True,
    }
    already = repair.prepare_repair(
        prepared_directory=prepared,
        recorded_at="2026-08-12T04:10:00+08:00",
        config_override=config,
        run_root_override=run_root,
    )
    assert already == {
        "status": "ALREADY_PUBLISHED_VERIFIED",
        "event_id": "A1-EVT-043",
    }
    assert read_runtime(run_root) == published
    assert len(
        repair.load_events(published["EVENT_LOG.jsonl"], label="event")
    ) == 43
