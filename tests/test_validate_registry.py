"""Unit tests for scripts/execution/validate_registry.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.execution.validate_registry import _has_cycle, main, validate  # noqa: E402


def _task(tid="D0-01", status="DONE", deps=None):
    return {
        "task_id": tid,
        "phase": "D0",
        "status": status,
        "dependencies": deps or [],
        "description": "test task",
        "acceptance": ["check"],
        # optional fields included for coverage
        "inputs": ["in"],
        "files": ["f.py"],
        "commands": ["make f"],
        "outputs": ["out"],
        "commit_sha": None,
        "report": None,
    }


def _registry(tasks):
    return {"registry_version": "1.0.0",
            "contract_id": "utr_editflow_contract_v2",
            "tasks": tasks}


def test_valid_registry_passes():
    reg = _registry([_task("R0-01"), _task("D0-01", deps=["R0-01"])])
    assert validate(reg) == []


def test_missing_field_fails():
    task = _task()
    del task["acceptance"]  # acceptance is required in v2 schema
    errors = validate(_registry([task]))
    assert any("acceptance" in e for e in errors)


def test_bad_status_fails():
    errors = validate(_registry([_task(status="MAYBE")]))
    assert any("status" in e for e in errors)


def test_bad_task_id_fails():
    errors = validate(_registry([_task(tid="r0-1")]))
    assert any("task_id" in e for e in errors)


def test_duplicate_task_id_fails():
    errors = validate(_registry([_task("D0-01"), _task("D0-01")]))
    assert any("duplicate" in e for e in errors)


def test_unknown_dependency_fails():
    errors = validate(_registry([_task("D0-01", deps=["D0-99"])]))
    assert any("unknown dependency" in e for e in errors)


def test_cycle_detected():
    tasks = [_task("D0-01", deps=["D0-02"]), _task("D0-02", deps=["D0-01"])]
    assert _has_cycle(tasks)
    errors = validate(_registry(tasks))
    assert any("cycle" in e for e in errors)


def test_missing_top_level_keys():
    assert validate({}) != []


def test_main_on_real_registry():
    rc = main([])
    assert rc == 0
