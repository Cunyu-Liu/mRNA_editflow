from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.route_a_v3.train_route2_xeditsetflow_v4 import (
    SetFlowTrainingV4Error,
    _write_atomic_terminal_v4,
    derive_training_update_geometry_v4,
    pass_complete_alive_event_v4,
    record_failed_attempt_if_started_v4,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/route_a_v3_route2_xeditsetflow_v4_screen_v1.json"


def test_setflow_training_terminal_artifact_is_atomic_and_exact(
    tmp_path: Path,
) -> None:
    output = tmp_path / "training_summary.json"
    payload = {"status": "TERMINAL_XEDITSETFLOW_V4_TRAINING_COMPLETE_PENDING_VALIDATION"}
    _write_atomic_terminal_v4(output, payload)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert not output.with_suffix(output.suffix + ".partial").exists()

    stale = tmp_path / "failure.json.partial"
    stale.write_text("interrupted", encoding="utf-8")
    with pytest.raises(SetFlowTrainingV4Error, match="partial terminal"):
        _write_atomic_terminal_v4(tmp_path / "failure.json", payload)
    source = (ROOT / "scripts/route_a_v3/train_route2_xeditsetflow_v4.py").read_text()
    assert 'if not (output_directory / "training_summary.json").exists()' in source


def test_update_budget_is_uniquely_derived_from_sources_and_ten_passes() -> None:
    geometry = derive_training_update_geometry_v4(101)
    assert geometry == {
        "train_source_count": 101,
        "sources_per_update": 8,
        "states_per_source": 4,
        "effective_state_batch": 32,
        "updates_per_pass": 13,
        "pass_count": 10,
        "total_optimizer_updates": 130,
    }


def test_pass_event_is_alive_only_and_contains_no_loss_or_validation_metric() -> None:
    event = pass_complete_alive_event_v4(
        run_id="v4_full", pass_number=4, update_count=100
    )
    assert event["active_performance_metric_emitted"] is False
    assert not any("loss" in key or "recovery" in key or "nll" in key for key in event)


def test_config_requires_four_checkpoints_after_exact_ten_passes() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["training"]["pass_count"] == 10
    assert config["training"]["saved_checkpoint_passes"] == [4, 6, 8, 10]
    assert config["training"]["validation_generation_during_training"] is False
    assert config["training"]["checkpoint_selection_after_terminal_training_only"] is True


def test_training_runner_dispatches_screen_and_confirmation_authorization() -> None:
    source = (
        ROOT / "scripts/route_a_v3/train_route2_xeditsetflow_v4.py"
    ).read_text(encoding="utf-8")
    assert 'run_stage in {"SCREEN", "CONFIRMATION"}' in source
    assert "require_setflow_v4_screen_launch_authorization" in source
    assert "require_setflow_v4_confirmation_launch_authorization" in source
    assert 'attempt_purpose": f"XEDITSETFLOW_V4_{run_stage}"' in source


def test_failure_updates_existing_attempt_in_place_without_adding_a_row(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    attempt = output / "training_attempt.json"
    attempt.write_text(
        json.dumps(
            {
                "attempt_id": "setflow-v4-attempt",
                "status": "RUNNING",
                "started_at": "2026-08-24T00:00:00+08:00",
                "code_commit": "head",
            }
        ),
        encoding="utf-8",
    )
    config = {"experiment_ledger_path": str(tmp_path / "attempts.csv")}
    assert record_failed_attempt_if_started_v4(
        config, output, RuntimeError("technical failure")
    )
    updated = json.loads(attempt.read_text(encoding="utf-8"))
    assert updated["attempt_id"] == "setflow-v4-attempt"
    assert updated["status"] == "FAILED"
    assert updated["error_type"] == "RuntimeError"
    rows = (tmp_path / "attempts.csv").read_text(encoding="utf-8-sig").splitlines()
    assert len(rows) == 2
