#!/usr/bin/env python3
"""Validate docs/execution/task_registry.yaml against the registry contract.

The JSON Schema source of truth is schemas/task_registry.schema.json. This
validator implements the same checks without requiring the jsonschema
package (keeps the execution environment dependency-free).

Usage:
    python scripts/execution/validate_registry.py \
        --registry docs/execution/task_registry.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import yaml

try:
    from scripts.execution.acceptance_semantics import validate_phase_acceptance
    from scripts.data.validate_d1_canonical_snapshot import validate_snapshot
except ModuleNotFoundError:  # direct script execution from scripts/execution
    from acceptance_semantics import validate_phase_acceptance

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.data.validate_d1_canonical_snapshot import validate_snapshot

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "task_registry.schema.json"
CANONICAL_GITHUB_URL = "https://github.com/Cunyu-Liu/mRNA_editflow.git"
RELEASE_PROFILES = {
    "D1_B0": {
        *(f"D1-{index:02d}" for index in range(1, 9)),
        *(f"B0-{index:02d}" for index in range(1, 6)),
    }
}
PHASE_GATE_TASKS = {"D1-08", "B0-05"}

REQUIRED_TASK_FIELDS = [
    "task_id",
    "phase_id",
    "status",
    "hypotheses",
    "dependencies",
    "dependency_gates",
    "scientific_service",
    "forbidden_actions",
    "inputs",
    "resource_labels",
    "conflict_keys",
    "allowed_parallel_tasks",
    "files",
    "commands",
    "outputs",
    "acceptance",
    "repair_loop",
    "commit_sha",
    "report",
]
GOVERNED_TASK_FIELDS = [
    "evidence_class",
    "completion_policy",
    "acceptance_artifact",
    "acceptance_sha256",
    "gate_evidence",
    "known_blockers",
    "phase_gate",
]
SNAPSHOT_TASK_FIELDS = {"snapshot_artifact", "snapshot_sha256"}
ALLOWED_TASK_FIELDS = (
    set(REQUIRED_TASK_FIELDS) | set(GOVERNED_TASK_FIELDS) | SNAPSHOT_TASK_FIELDS
)
LIST_FIELDS = [
    "hypotheses",
    "dependencies",
    "dependency_gates",
    "forbidden_actions",
    "inputs",
    "resource_labels",
    "conflict_keys",
    "allowed_parallel_tasks",
    "files",
    "commands",
    "outputs",
    "acceptance",
    "repair_loop",
]
EVIDENCE_CLASSES = {
    "READ_ONLY_PREFLIGHT",
    "FULL_SCOPE_DATA",
    "FULL_SCOPE_BENCHMARK",
    "SMOKE",
    "FIXTURE_ONLY",
}
COMPLETION_POLICIES = {"MUST_PASS_ALL", "FAIL_CLOSED_WITH_BLOCKERS"}
PHASE_GATE_EVIDENCE_CLASSES = {
    "READ_ONLY_PREFLIGHT",
    "FULL_SCOPE_DATA",
    "FULL_SCOPE_BENCHMARK",
}
FINAL_STATUSES = {"VERIFIED", "FROZEN"}
COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
TASK_ID_PATTERN = re.compile(r"^[A-Z0-9]+-[0-9]{2}$")
DEPENDENCY_GATE_PATTERN = re.compile(
    r"^(?P<task>[A-Z0-9]+-[0-9]{2}):(?P<state>VERIFIED|FROZEN)$"
)
EXPECTED_GOAL_SHA256 = (
    "ff5a440910c9c8ef47e460b3f2d6a291c7fce12e687e090f2200dc79796a89c3"
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


def validate(
    registry: dict,
    schema_statuses: list[str] | None = None,
    repo_root: Path | None = None,
    *,
    release_profile: str | None = None,
    remote_name: str = "origin",
    expected_remote_url: str = CANONICAL_GITHUB_URL,
    allow_unfrozen_b0_inventory: bool = True,
) -> list[str]:
    """Return a list of validation error strings (empty == valid)."""
    errors: list[str] = []
    root = REPO_ROOT if repo_root is None else Path(repo_root)
    if not isinstance(registry, dict):
        return ["registry must be a mapping"]
    if schema_statuses is None:
        schema_statuses = [
            "PLANNED",
            "REGISTERED",
            "PREFLIGHT_PASSED",
            "RUNNING",
            "WAITING_FOR_GPU",
            "SAFE_PAUSED",
            "FAILED_WITH_EVIDENCE",
            "REPAIR_REQUIRED",
            "SUPERSEDED_WITH_TRACE",
            "VERIFIED",
            "FROZEN",
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
    if registry["contract_id"] != "mrna_editflow_single_active_contract":
        errors.append("contract_id must equal mrna_editflow_single_active_contract")
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
        extra = set(task) - ALLOWED_TASK_FIELDS
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
                if not isinstance(val, list) or not all(
                    isinstance(x, str) for x in val
                ):
                    errors.append(f"{tid}: field '{field}' must be a list of strings")
        for field in ("commit_sha", "report"):
            if (
                field in task
                and task[field] is not None
                and not isinstance(task[field], str)
            ):
                errors.append(f"{tid}: field '{field}' must be a string or null")
        if "scientific_service" in task and not isinstance(
            task["scientific_service"], str
        ):
            errors.append(f"{tid}: scientific_service must be a string")
        hypotheses = task.get("hypotheses", [])
        if any(h not in {f"H{i}" for i in range(1, 9)} for h in hypotheses):
            errors.append(f"{tid}: hypotheses must be drawn from H1-H8")
        labels = task.get("resource_labels", [])
        if any(label not in RESOURCE_LABELS for label in labels):
            errors.append(f"{tid}: unknown resource label")
        phase = task.get("phase_id")
        if phase in {"D1", "B0"}:
            for field in GOVERNED_TASK_FIELDS:
                if field not in task:
                    errors.append(f"{tid}: D1/B0 task missing required field '{field}'")
            if "FINAL_LABEL_ACCESS" in labels:
                errors.append(f"{tid}: FINAL_LABEL_ACCESS is forbidden for D1/B0 tasks")
            if task.get("evidence_class") not in EVIDENCE_CLASSES:
                errors.append(f"{tid}: invalid evidence_class")
            if task.get("completion_policy") not in COMPLETION_POLICIES:
                errors.append(f"{tid}: invalid completion_policy")
            if not isinstance(task.get("known_blockers"), list) or not all(
                isinstance(item, str) for item in task.get("known_blockers", [])
            ):
                errors.append(f"{tid}: known_blockers must be a list of strings")
            if not isinstance(task.get("phase_gate"), bool):
                errors.append(f"{tid}: phase_gate must be boolean")
            _validate_gate_evidence(task, errors)
            if task.get("status") in FINAL_STATUSES:
                _validate_final_evidence(
                    task,
                    root,
                    errors,
                    remote_name=remote_name,
                    expected_remote_url=expected_remote_url,
                )
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
            else:
                other = next(
                    (
                        candidate
                        for candidate in registry["tasks"]
                        if candidate.get("task_id") == parallel
                    ),
                    None,
                )
                if other is not None and task.get("task_id") not in other.get(
                    "allowed_parallel_tasks", []
                ):
                    errors.append(
                        f"{task.get('task_id')}: allowed_parallel_tasks must be "
                        f"reciprocal with '{parallel}'"
                    )
    status_by_task = {
        task["task_id"]: task.get("status")
        for task in registry["tasks"]
        if isinstance(task, dict) and task.get("task_id")
    }
    for task in registry["tasks"]:
        if not isinstance(task, dict) or task.get("status") not in {
            "VERIFIED",
            "FROZEN",
        }:
            continue
        for dependency in task.get("dependencies", []):
            if status_by_task.get(dependency) not in {"VERIFIED", "FROZEN"}:
                errors.append(
                    f"{task.get('task_id')}: verified task has unmet dependency "
                    f"'{dependency}'"
                )
    _validate_phase_anchors(
        registry["tasks"],
        status_by_task,
        errors,
        allow_unfrozen_b0_inventory=allow_unfrozen_b0_inventory,
    )
    _validate_release_profile(registry["tasks"], release_profile, errors)
    if _has_cycle(registry["tasks"]):
        errors.append("registry dependency graph contains a cycle")
    return errors


def _validate_gate_evidence(task: dict, errors: list[str]) -> None:
    tid = task.get("task_id")
    evidence = task.get("gate_evidence")
    if not isinstance(evidence, list):
        errors.append(f"{tid}: gate_evidence must be a list")
        return
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            errors.append(f"{tid}: gate_evidence[{index}] must be a mapping")
            continue
        if set(item) != {"predicate", "status", "evidence"}:
            errors.append(
                f"{tid}: gate_evidence[{index}] must contain exactly "
                "predicate/status/evidence"
            )
            continue
        if not isinstance(item["predicate"], str) or not item["predicate"]:
            errors.append(f"{tid}: gate_evidence[{index}] predicate is invalid")
        if item["status"] not in {"PASS", "FAIL", "UNKNOWN"}:
            errors.append(f"{tid}: gate_evidence[{index}] status is invalid")
        if not isinstance(item["evidence"], str) or not item["evidence"]:
            errors.append(f"{tid}: gate_evidence[{index}] evidence is invalid")


def _validate_final_evidence(
    task: dict,
    root: Path,
    errors: list[str],
    *,
    remote_name: str,
    expected_remote_url: str,
) -> None:
    tid = task.get("task_id")
    commit_sha = task.get("commit_sha")
    if not isinstance(commit_sha, str) or not COMMIT_SHA_PATTERN.fullmatch(commit_sha):
        errors.append(f"{tid}: final D1/B0 task requires a 40-hex commit_sha")
    if not isinstance(task.get("report"), str) or not task["report"].strip():
        errors.append(f"{tid}: final D1/B0 task requires a non-empty report")
    if task.get("known_blockers"):
        errors.append(f"{tid}: final D1/B0 task has unresolved known_blockers")
    if task.get("phase_gate") and task.get("evidence_class") not in (
        PHASE_GATE_EVIDENCE_CLASSES
    ):
        errors.append(
            f"{tid}: {task.get('evidence_class')} evidence cannot satisfy a phase gate"
        )
    evidence = task.get("gate_evidence")
    if not evidence:
        errors.append(f"{tid}: final D1/B0 task requires gate_evidence")
    elif any(
        item.get("status") != "PASS" for item in evidence if isinstance(item, dict)
    ):
        errors.append(f"{tid}: all final gate_evidence predicates must PASS")

    artifact = task.get("acceptance_artifact")
    artifact_sha = task.get("acceptance_sha256")
    if not isinstance(artifact, str) or not artifact:
        errors.append(f"{tid}: final D1/B0 task requires acceptance_artifact")
        return
    if not isinstance(artifact_sha, str) or not SHA256_PATTERN.fullmatch(artifact_sha):
        errors.append(f"{tid}: final D1/B0 task requires acceptance_sha256")
        return
    path = root / artifact
    if not path.is_file():
        errors.append(f"{tid}: acceptance_artifact not found: {artifact}")
        return
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != artifact_sha:
        errors.append(f"{tid}: acceptance_artifact sha256 mismatch")
        return

    try:
        acceptance = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        errors.append(f"{tid}: acceptance_artifact is not valid JSON")
        return
    phase = str(task.get("phase_id") or "")
    for semantic_error in validate_phase_acceptance(
        phase, acceptance, require_pass=True
    ):
        errors.append(f"{tid}: {semantic_error}")

    if not _git_commit_exists(root, str(commit_sha)):
        errors.append(f"{tid}: commit_sha is not an existing Git commit")
        return
    try:
        artifact_relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        errors.append(f"{tid}: acceptance_artifact must be inside the repository")
        return
    committed = _git_blob(root, str(commit_sha), artifact_relative)
    if committed is None or hashlib.sha256(committed).hexdigest() != artifact_sha:
        errors.append(
            f"{tid}: acceptance_artifact is not the hash-matching blob in commit_sha"
        )

    if tid == "D1-08":
        _validate_d1_snapshot_binding(
            task,
            root,
            str(commit_sha),
            errors,
        )

    publication = [
        item
        for item in task.get("gate_evidence", [])
        if isinstance(item, dict)
        and item.get("predicate") == "published_remote_ref_contains_commit"
    ]
    if len(publication) != 1 or publication[0].get("status") != "PASS":
        errors.append(
            f"{tid}: final status requires one published_remote_ref_contains_commit PASS"
        )
    else:
        remote_ref = str(publication[0].get("evidence") or "")
        if not remote_ref.startswith(("refs/heads/", "refs/tags/")):
            errors.append(
                f"{tid}: published remote ref must be refs/heads/* or refs/tags/*"
            )
        elif not _remote_ref_contains_commit(
            root,
            remote_name,
            remote_ref,
            str(commit_sha),
            expected_remote_url=expected_remote_url,
        ):
            errors.append(f"{tid}: published remote ref does not contain commit_sha")


def _validate_d1_snapshot_binding(
    task: dict,
    root: Path,
    commit_sha: str,
    errors: list[str],
) -> None:
    tid = str(task.get("task_id") or "D1-08")
    artifact = task.get("snapshot_artifact")
    digest = task.get("snapshot_sha256")
    canonical_path = "data/d1/manifests/d1_canonical_snapshot.json"
    if artifact != canonical_path:
        errors.append(f"{tid}: snapshot_artifact must equal {canonical_path}")
        return
    if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
        errors.append(f"{tid}: snapshot_sha256 must be a full SHA-256")
        return
    path = root / canonical_path
    if not path.is_file():
        errors.append(f"{tid}: canonical snapshot artifact is missing")
        return
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        errors.append(f"{tid}: canonical snapshot live sha256 mismatch")
        return
    semantic_errors = validate_snapshot(path, repo_root=root)
    for error in semantic_errors:
        errors.append(f"{tid}: canonical snapshot invalid: {error}")
    committed = _git_blob(root, commit_sha, canonical_path)
    if committed is None or hashlib.sha256(committed).hexdigest() != digest:
        errors.append(
            f"{tid}: canonical snapshot is not the hash-matching blob " "in commit_sha"
        )


def _git_command(root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    command = ["git", "-C", str(root), *args]
    try:
        return subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(
            command,
            124,
            stdout=b"",
            stderr=str(exc).encode("utf-8", errors="replace"),
        )


def _git_commit_exists(root: Path, commit_sha: str) -> bool:
    if not COMMIT_SHA_PATTERN.fullmatch(commit_sha):
        return False
    return (
        _git_command(root, "cat-file", "-e", f"{commit_sha}^{{commit}}").returncode == 0
    )


def _git_blob(root: Path, commit_sha: str, path: str) -> bytes | None:
    result = _git_command(root, "show", f"{commit_sha}:{path}")
    return result.stdout if result.returncode == 0 else None


def _canonical_remote_identity(value: str) -> str:
    text = value.strip()
    if text.startswith("git@github.com:"):
        path = text.split(":", 1)[1]
        return f"github.com/{path.removesuffix('.git')}"
    parsed = urlparse(text)
    if parsed.hostname == "github.com":
        path = parsed.path.lstrip("/").removesuffix(".git")
        return f"github.com/{path}"
    if parsed.scheme == "file":
        return str(Path(parsed.path).resolve())
    return str(Path(text).expanduser().resolve())


def _remote_ref_oid(root: Path, remote_name: str, ref: str) -> str | None:
    result = _git_command(root, "ls-remote", "--exit-code", remote_name, ref)
    if result.returncode != 0:
        return None
    try:
        lines = result.stdout.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError:
        return None
    matches: list[str] = []
    for line in lines:
        fields = line.split("\t")
        if (
            len(fields) == 2
            and fields[1] == ref
            and COMMIT_SHA_PATTERN.fullmatch(fields[0])
        ):
            matches.append(fields[0])
    return matches[0] if len(matches) == 1 else None


def _remote_ref_contains_commit(
    root: Path,
    remote_name: str,
    ref: str,
    commit_sha: str,
    *,
    expected_remote_url: str,
) -> bool:
    remote_url = _git_command(root, "remote", "get-url", remote_name)
    if remote_url.returncode != 0:
        return False
    try:
        observed_url = remote_url.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError:
        return False
    if _canonical_remote_identity(observed_url) != _canonical_remote_identity(
        expected_remote_url
    ):
        return False
    remote_oid = _remote_ref_oid(root, remote_name, ref)
    if remote_oid is None or not _git_commit_exists(root, remote_oid):
        return False
    return (
        _git_command(
            root, "merge-base", "--is-ancestor", commit_sha, remote_oid
        ).returncode
        == 0
    )


def _validate_release_profile(
    tasks: list[dict], release_profile: str | None, errors: list[str]
) -> None:
    if release_profile is None:
        return
    expected = RELEASE_PROFILES.get(release_profile)
    if expected is None:
        errors.append(f"unknown release profile: {release_profile}")
        return
    governed = {
        str(task.get("task_id"))
        for task in tasks
        if isinstance(task, dict) and task.get("phase_id") in {"D1", "B0"}
    }
    missing = expected - governed
    unexpected = governed - expected
    if missing:
        errors.append(
            f"{release_profile} release profile missing tasks: {sorted(missing)}"
        )
    if unexpected:
        errors.append(
            f"{release_profile} release profile has unexpected tasks: "
            f"{sorted(unexpected)}"
        )
    by_id = {
        str(task.get("task_id")): task
        for task in tasks
        if isinstance(task, dict) and task.get("task_id")
    }
    for task_id in expected & set(by_id):
        expected_gate = task_id in PHASE_GATE_TASKS
        if by_id[task_id].get("phase_gate") is not expected_gate:
            errors.append(
                f"{task_id}: phase_gate must be {str(expected_gate).lower()} "
                f"for {release_profile}"
            )


def _validate_phase_anchors(
    tasks: list[dict],
    status_by_task: dict[str, str],
    errors: list[str],
    *,
    allow_unfrozen_b0_inventory: bool,
) -> None:
    deps = {
        task.get("task_id"): set(task.get("dependencies", []))
        for task in tasks
        if isinstance(task, dict) and task.get("task_id")
    }

    def ancestors(task_id: str) -> set[str]:
        found: set[str] = set()
        stack = list(deps.get(task_id, set()))
        while stack:
            current = stack.pop()
            if current in found:
                continue
            found.add(current)
            stack.extend(deps.get(current, set()))
        return found

    for task in tasks:
        if not isinstance(task, dict):
            continue
        tid = task.get("task_id", "")
        phase = task.get("phase_id")
        lineage = ancestors(tid)
        if phase == "D1" and not {"C0-05", "D0-05"} <= lineage:
            errors.append(f"{tid}: D1 lineage must descend from C0-05 and D0-05")
        if phase == "B0":
            if "D1-08" not in lineage:
                errors.append(f"{tid}: B0 lineage must descend from D1-08")
            if status_by_task.get("D1-08") != "FROZEN":
                # Registering the complete future inventory is not execution.
                # Any active or final B0 status still fails before D1 freezes.
                inactive_statuses = {"PLANNED", "REGISTERED", "SAFE_PAUSED"}
                if (
                    not allow_unfrozen_b0_inventory
                    or task.get("status") not in inactive_statuses
                ):
                    errors.append(f"{tid}: B0 requires D1-08:FROZEN")
        if tid == "D1-08" and task.get("status") == "FROZEN":
            for required in (
                "D1-01",
                "D1-02",
                "D1-03",
                "D1-04",
                "D1-05",
                "D1-06",
                "D1-07",
            ):
                if status_by_task.get(required) not in FINAL_STATUSES:
                    errors.append(f"D1-08: cannot freeze before {required}:VERIFIED")
        if tid == "B0-05" and task.get("status") == "FROZEN":
            for required in ("B0-01", "B0-02", "B0-03", "B0-04"):
                if status_by_task.get(required) not in FINAL_STATUSES:
                    errors.append(f"B0-05: cannot freeze before {required}:VERIFIED")


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
        "--registry",
        default=str(REPO_ROOT / "docs/execution/task_registry.yaml"),
    )
    parser.add_argument("--release-profile", choices=sorted(RELEASE_PROFILES))
    parser.add_argument("--remote-name", default="origin")
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

    errors = validate(
        registry,
        statuses,
        release_profile=args.release_profile,
        remote_name=args.remote_name,
    )
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
