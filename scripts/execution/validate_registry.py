#!/usr/bin/env python3
"""Validate docs/execution/task_registry_v2.yaml against the registry contract.

The JSON Schema source of truth is schemas/task_registry.schema.json. This
validator implements the same checks without requiring the jsonschema
package (keeps the execution environment dependency-free).

Usage:
    python scripts/execution/validate_registry.py \
        --registry docs/execution/task_registry_v2.yaml
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "task_registry.schema.json"

REQUIRED_TASK_FIELDS = [
    "task_id", "phase_id", "status", "hypotheses", "dependencies",
    "dependency_gates", "scientific_service", "forbidden_actions", "inputs",
    "resource_labels", "conflict_keys", "allowed_parallel_tasks", "files",
    "commands", "outputs", "acceptance", "repair_loop", "commit_sha", "report",
]
LIST_FIELDS = [
    "hypotheses", "dependencies", "dependency_gates", "forbidden_actions",
    "inputs", "resource_labels", "conflict_keys", "allowed_parallel_tasks",
    "files", "commands", "outputs", "acceptance", "repair_loop",
]
TASK_ID_PATTERN = re.compile(r"^[A-Z0-9]+-[0-9]{2}$")
DEPENDENCY_GATE_PATTERN = re.compile(
    r"^(?P<task>[A-Z0-9]+-[0-9]{2}):(?P<state>VERIFIED|FROZEN)$"
)
EXPECTED_GOAL_SHA256 = (
    "c3dc5875868d847b8519fee40b14c43b65e4c5948dc5c3b98101ca61a5671dd5"
)
RESOURCE_LABELS = {
    "GPU_EXCLUSIVE",
    "CPU_LIGHT",
    "CPU_HEAVY",
    "IO_HEAVY",
    "READ_ONLY",
    "MUTATES_CODE",
    "MUTATES_DATA",
    "FINAL_LABEL_ACCESS",
}


def load_schema_statuses() -> list[str]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return schema["properties"]["tasks"]["items"]["properties"]["status"]["enum"]


def validate(registry: dict, schema_statuses: list[str] | None = None) -> list[str]:
    """Return a list of validation error strings (empty == valid)."""
    errors: list[str] = []
    if schema_statuses is None:
        schema_statuses = [
            "PLANNED", "REGISTERED", "PREFLIGHT_PASSED", "RUNNING",
            "WAITING_FOR_GPU", "SAFE_PAUSED", "FAILED_WITH_EVIDENCE",
            "REPAIR_REQUIRED", "SUPERSEDED_WITH_TRACE", "VERIFIED", "FROZEN",
        ]

    for key in ("registry_version", "contract_id", "goal_contract_sha256", "tasks"):
        if key not in registry:
            errors.append(f"registry missing required key: {key}")
    if errors:
        return errors
    if not isinstance(registry["tasks"], list) or not registry["tasks"]:
        errors.append("registry.tasks must be a non-empty list")
        return errors
    if registry["registry_version"] != "2.0.0":
        errors.append("registry_version must equal 2.0.0")
    if registry["contract_id"] != "utr_editflow_goal_v2":
        errors.append("contract_id must equal utr_editflow_goal_v2")
    if registry["goal_contract_sha256"] != EXPECTED_GOAL_SHA256:
        errors.append("goal_contract_sha256 does not match the frozen Goal")

    seen_ids: set[str] = set()
    for idx, task in enumerate(registry["tasks"]):
        where = f"tasks[{idx}]"
        if not isinstance(task, dict):
            errors.append(f"{where} must be a mapping")
            continue
        tid = task.get("task_id", where)
        for field in REQUIRED_TASK_FIELDS:
            if field not in task:
                errors.append(f"{tid}: missing required field '{field}'")
        extra = set(task) - set(REQUIRED_TASK_FIELDS)
        if extra:
            errors.append(f"{tid}: unexpected fields {sorted(extra)}")
        if "task_id" in task:
            if not TASK_ID_PATTERN.match(str(task["task_id"])):
                errors.append(f"{tid}: task_id must match {TASK_ID_PATTERN.pattern}")
            if task["task_id"] in seen_ids:
                errors.append(f"{tid}: duplicate task_id")
            seen_ids.add(task["task_id"])
            if task.get("phase_id") != str(task["task_id"]).split("-")[0]:
                errors.append(f"{tid}: phase_id must match task_id prefix")
        if "status" in task and task["status"] not in schema_statuses:
            errors.append(f"{tid}: status '{task['status']}' not in {schema_statuses}")
        for field in LIST_FIELDS:
            if field in task:
                val = task[field]
                if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
                    errors.append(f"{tid}: field '{field}' must be a list of strings")
        for field in ("commit_sha", "report"):
            if field in task and task[field] is not None and not isinstance(task[field], str):
                errors.append(f"{tid}: field '{field}' must be a string or null")
        if "scientific_service" in task and not isinstance(task["scientific_service"], str):
            errors.append(f"{tid}: scientific_service must be a string")
        hypotheses = task.get("hypotheses", [])
        if any(h not in {f"H{i}" for i in range(1, 9)} for h in hypotheses):
            errors.append(f"{tid}: hypotheses must be drawn from H1-H8")
        labels = task.get("resource_labels", [])
        if any(label not in RESOURCE_LABELS for label in labels):
            errors.append(f"{tid}: unknown resource label")
        gates = task.get("dependency_gates", [])
        gate_tasks: set[str] = set()
        for gate in gates:
            match = DEPENDENCY_GATE_PATTERN.match(gate)
            if match is None:
                errors.append(f"{tid}: malformed dependency gate '{gate}'")
            else:
                gate_tasks.add(match.group("task"))
        if gate_tasks != set(task.get("dependencies", [])):
            errors.append(f"{tid}: dependency_gates must cover dependencies exactly")

    # dependency graph must reference known task ids and be acyclic
    known = {t.get("task_id") for t in registry["tasks"] if isinstance(t, dict)}
    known.discard(None)
    for task in registry["tasks"]:
        if not isinstance(task, dict):
            continue
        for dep in task.get("dependencies", []):
            if dep not in known:
                errors.append(f"{task.get('task_id')}: unknown dependency '{dep}'")
        for parallel in task.get("allowed_parallel_tasks", []):
            if parallel not in known:
                errors.append(
                    f"{task.get('task_id')}: unknown allowed_parallel_task '{parallel}'"
                )
    status_by_task = {
        task["task_id"]: task.get("status")
        for task in registry["tasks"]
        if isinstance(task, dict) and task.get("task_id")
    }
    for task in registry["tasks"]:
        if not isinstance(task, dict) or task.get("status") not in {"VERIFIED", "FROZEN"}:
            continue
        for dependency in task.get("dependencies", []):
            if status_by_task.get(dependency) not in {"VERIFIED", "FROZEN"}:
                errors.append(
                    f"{task.get('task_id')}: verified task has unmet dependency "
                    f"'{dependency}'"
                )
    if _has_cycle(registry["tasks"]):
        errors.append("registry dependency graph contains a cycle")
    return errors


def _has_cycle(tasks: list[dict]) -> bool:
    deps = {t.get("task_id"): [d for d in t.get("dependencies", [])]
            for t in tasks if isinstance(t, dict) and t.get("task_id")}
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {k: WHITE for k in deps}

    def visit(node: str) -> bool:
        color[node] = GRAY
        for dep in deps.get(node, []):
            if dep not in color:
                continue
            if color[dep] == GRAY:
                return True
            if color[dep] == WHITE and visit(dep):
                return True
        color[node] = BLACK
        return False

    return any(color[n] == WHITE and visit(n) for n in deps)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default=str(REPO_ROOT / "docs/execution/task_registry_v2.yaml"),
    )
    args = parser.parse_args(argv)

    path = Path(args.registry)
    if not path.is_file():
        print(f"registry not found: {path}")
        return 2
    registry = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        statuses = load_schema_statuses()
    except Exception:
        statuses = None

    errors = validate(registry, statuses)
    for err in errors:
        print(f"ERROR: {err}")
    n_tasks = len(registry.get("tasks", [])) if isinstance(registry, dict) else 0
    if errors:
        print(f"task registry INVALID: {len(errors)} error(s) across {n_tasks} task(s)")
        return 1
    print(f"task registry VALID: {n_tasks} task(s), 0 errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
