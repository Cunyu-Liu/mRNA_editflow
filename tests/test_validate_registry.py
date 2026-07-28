"""Unit tests for scripts/execution/validate_registry.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.execution.validate_registry import _has_cycle, main, validate  # noqa: E402

GOAL_SHA = "c3dc5875868d847b8519fee40b14c43b65e4c5948dc5c3b98101ca61a5671dd5"


def _task(tid="D0-01", status="VERIFIED", deps=None):
    dependencies = deps or []
    return {
        "task_id": tid,
        "phase_id": tid.split("-")[0],
        "status": status,
        "hypotheses": ["H1"],
        "dependencies": dependencies,
        "dependency_gates": [f"{dep}:VERIFIED" for dep in dependencies],
        "scientific_service": "serve the frozen question",
        "forbidden_actions": ["do not weaken gates"],
        "inputs": ["in"],
        "resource_labels": ["CPU_LIGHT"],
        "conflict_keys": [],
        "allowed_parallel_tasks": [],
        "files": ["f.py"],
        "commands": ["make f"],
        "outputs": ["out"],
        "acceptance": ["check"],
        "repair_loop": ["retry"],
        "commit_sha": None,
        "report": None,
    }


def _registry(tasks):
    return {"registry_version": "2.0.0",
            "contract_id": "utr_editflow_goal_v2",
            "goal_contract_sha256": GOAL_SHA,
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


def test_wrong_goal_hash_and_contract_fail():
    reg = _registry([_task()])
    reg["goal_contract_sha256"] = "0" * 64
    reg["contract_id"] = "wrong"
    errors = validate(reg)
    assert any("frozen Goal" in error for error in errors)
    assert any("contract_id" in error for error in errors)


def test_verified_task_requires_verified_dependencies():
    upstream = _task("C0-01", status="RUNNING")
    downstream = _task("D0-01", deps=["C0-01"])
    errors = validate(_registry([upstream, downstream]))
    assert any("unmet dependency" in error for error in errors)


def test_main_on_real_registry():
    rc = main([])
    assert rc == 0
