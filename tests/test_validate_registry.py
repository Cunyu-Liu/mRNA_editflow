"""Unit tests for scripts/execution/validate_registry.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.execution.validate_registry import _has_cycle, main, validate  # noqa: E402


def _task(tid="D0-01", status="DONE", deps=None):
    return {
        "task_id": tid,
        "status": status,
        "dependencies": deps or [],
        "inputs": ["in"],
        "in_scope": ["scope"],
        "out_of_scope": ["oos"],
        "files": ["f.py"],
        "commands": ["make f"],
        "outputs": ["out"],
        "acceptance": ["check"],
        "repair_loop": ["retry"],
        "commit_sha": None,
        "report": None,
    }


def _registry(tasks):
    return {"registry_version": "1.0.0",
            "contract_id": "public_intervention_contract_v1",
            "tasks": tasks}


def test_valid_registry_passes():
    reg = _registry([_task("R0-01"), _task("D0-01", deps=["R0-01"])])
    assert validate(reg) == []


def test_missing_field_fails():
    task = _task()
    del task["repair_loop"]
    errors = validate(_registry([task]))
    assert any("repair_loop" in e for e in errors)


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
