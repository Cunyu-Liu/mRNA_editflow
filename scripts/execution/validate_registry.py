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

# v2 schema: 5 required fields; the rest are optional
REQUIRED_TASK_FIELDS = [
    "task_id",
    "phase",
    "status",
    "dependencies",
    "acceptance",
]
# All valid task fields (required + optional) — used for unexpected-field check
ALL_TASK_FIELDS = [
    "task_id",
    "phase",
    "status",
    "dependencies",
    "acceptance",
    "description",
    "gate",
    "inputs",
    "in_scope",
    "out_of_scope",
    "files",
    "commands",
    "outputs",
    "repair_loop",
    "goal_sha256",
    "run_id",
    "evidence_level",
    "resource_tags",
    "conflict_keys",
    "allowed_parallel_tasks",
    "final_label_access",
    "downstream_stage_started",
    "commit_sha",
    "report",
]
LIST_FIELDS = [
    "dependencies",
    "inputs",
    "in_scope",
    "out_of_scope",
    "files",
    "commands",
    "outputs",
    "acceptance",
    "repair_loop",
    "resource_tags",
    "conflict_keys",
    "allowed_parallel_tasks",
]
TASK_ID_PATTERN = re.compile(r"^[A-Z0-9]+-[A-Za-z0-9]+$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def load_schema_statuses() -> list[str]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return schema["properties"]["tasks"]["items"]["properties"]["status"]["enum"]


def validate(registry: dict, schema_statuses: list[str] | None = None) -> list[str]:
    """Return a list of validation error strings (empty == valid)."""
    errors: list[str] = []
    if schema_statuses is None:
        schema_statuses = ["PENDING", "IN_PROGRESS", "DONE", "BLOCKED", "FAILED"]

    for key in ("registry_version", "contract_id", "tasks"):
        if key not in registry:
            errors.append(f"registry missing required key: {key}")
    if errors:
        return errors
    if not isinstance(registry["tasks"], list) or not registry["tasks"]:
        errors.append("registry.tasks must be a non-empty list")
        return errors

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
        extra = set(task) - set(ALL_TASK_FIELDS)
        if extra:
            errors.append(f"{tid}: unexpected fields {sorted(extra)}")
        if "task_id" in task:
            if not TASK_ID_PATTERN.match(str(task["task_id"])):
                errors.append(f"{tid}: task_id must match {TASK_ID_PATTERN.pattern}")
            if task["task_id"] in seen_ids:
                errors.append(f"{tid}: duplicate task_id")
            seen_ids.add(task["task_id"])
        if "status" in task and task["status"] not in schema_statuses:
            errors.append(f"{tid}: status '{task['status']}' not in {schema_statuses}")
        for field in LIST_FIELDS:
            if field in task:
                val = task[field]
                if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
                    errors.append(f"{tid}: field '{field}' must be a list of strings")
                elif field in {
                    "resource_tags",
                    "conflict_keys",
                    "allowed_parallel_tasks",
                } and len(val) != len(set(val)):
                    errors.append(f"{tid}: field '{field}' must contain unique strings")
        if "goal_sha256" in task and not SHA256_PATTERN.fullmatch(
            str(task["goal_sha256"])
        ):
            errors.append(f"{tid}: field 'goal_sha256' must be a lowercase SHA-256")
        for field in ("run_id", "commit_sha", "report"):
            if field in task and task[field] is not None and not isinstance(task[field], str):
                errors.append(f"{tid}: field '{field}' must be a string or null")
        for field in ("evidence_level",):
            if field in task and not isinstance(task[field], str):
                errors.append(f"{tid}: field '{field}' must be a string")
        for field in ("final_label_access", "downstream_stage_started"):
            if field in task and not isinstance(task[field], bool):
                errors.append(f"{tid}: field '{field}' must be a boolean")

    # dependency graph must reference known task ids and be acyclic
    known = {t.get("task_id") for t in registry["tasks"] if isinstance(t, dict)}
    known.discard(None)
    for task in registry["tasks"]:
        if not isinstance(task, dict):
            continue
        for dep in task.get("dependencies", []):
            if dep not in known:
                errors.append(f"{task.get('task_id')}: unknown dependency '{dep}'")
    if _has_cycle(registry["tasks"]):
        errors.append("registry dependency graph contains a cycle")
    return errors


def _has_cycle(tasks: list[dict]) -> bool:
    deps = {
        t.get("task_id"): [d for d in t.get("dependencies", [])]
        for t in tasks
        if isinstance(t, dict) and t.get("task_id")
    }
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
        "--registry", default=str(REPO_ROOT / "docs/execution/task_registry_v2.yaml")
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
