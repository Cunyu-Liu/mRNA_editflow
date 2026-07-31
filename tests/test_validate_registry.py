"""Unit tests for scripts/execution/validate_registry.py."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.execution.validate_registry import _has_cycle, main, validate  # noqa: E402
from tests.governance_fixtures import (  # noqa: E402
    valid_b0_acceptance,
    valid_d1_acceptance,
)

GOAL_SHA = "ff5a440910c9c8ef47e460b3f2d6a291c7fce12e687e090f2200dc79796a89c3"


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
    return {
        "registry_version": "2.0.0",
        "contract_id": "mrna_editflow_single_active_contract",
        "goal_contract_sha256": GOAL_SHA,
        "tasks": tasks,
    }


def _governed_task(tid, status="REGISTERED", deps=None):
    task = _task(tid, status=status, deps=deps)
    task.update(
        {
            "evidence_class": (
                "FULL_SCOPE_DATA" if tid.startswith("D1-") else "FULL_SCOPE_BENCHMARK"
            ),
            "completion_policy": "MUST_PASS_ALL",
            "acceptance_artifact": None,
            "acceptance_sha256": None,
            "gate_evidence": [],
            "known_blockers": [],
            "phase_gate": tid in {"D1-08", "B0-05"},
        }
    )
    return task


def _release_inventory_tasks():
    tasks = [_task("C0-05"), _task("D0-05")]
    previous = None
    for index in range(1, 9):
        task_id = f"D1-{index:02d}"
        dependencies = ["C0-05", "D0-05"] if previous is None else [previous]
        tasks.append(_governed_task(task_id, deps=dependencies))
        previous = task_id
    previous = "D1-08"
    for index in range(1, 6):
        task_id = f"B0-{index:02d}"
        tasks.append(_governed_task(task_id, deps=[previous]))
        previous = task_id
    return tasks


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _commit_acceptance(root: Path, *, phase: str = "D1") -> tuple[Path, str, str, Path]:
    _git(root, "init")
    _git(root, "config", "user.email", "tests@example.invalid")
    _git(root, "config", "user.name", "tests")
    acceptance = root / "acceptance.json"
    payload = valid_d1_acceptance(root) if phase == "D1" else valid_b0_acceptance()
    acceptance.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    _git(root, "add", "acceptance.json")
    _git(root, "commit", "-m", "freeze acceptance")
    commit = _git(root, "rev-parse", "HEAD")
    remote = root.parent / f"{root.name}-remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True)
    _git(root, "remote", "add", "origin", str(remote))
    remote_ref = "refs/heads/test"
    _git(root, "push", "origin", f"HEAD:{remote_ref}")
    return acceptance, commit, remote_ref, remote


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


def test_main_release_profile_on_real_prepublication_registry():
    rc = main(["--release-profile", "D1_B0"])
    assert rc == 0


def test_d1_and_b0_tasks_require_structured_gate_fields():
    c0 = _task("C0-05")
    d0 = _task("D0-05")
    d1 = _task("D1-01", status="REGISTERED", deps=["C0-05", "D0-05"])
    errors = validate(_registry([c0, d0, d1]))
    assert any("evidence_class" in error for error in errors)
    assert any("acceptance_artifact" in error for error in errors)


def test_governed_verified_task_requires_full_evidence(tmp_path):
    c0 = _task("C0-05")
    d0 = _task("D0-05")
    d1 = _governed_task("D1-01", status="VERIFIED", deps=["C0-05", "D0-05"])
    errors = validate(_registry([c0, d0, d1]), repo_root=tmp_path)
    assert any("40-hex commit_sha" in error for error in errors)
    assert any("acceptance_artifact" in error for error in errors)
    assert any("gate_evidence" in error for error in errors)

    acceptance, commit, remote_ref, remote = _commit_acceptance(tmp_path)
    d1["commit_sha"] = commit
    d1["report"] = "FULL_SCOPE_PREFLIGHT_VERIFIED"
    d1["acceptance_artifact"] = "acceptance.json"
    d1["acceptance_sha256"] = hashlib.sha256(acceptance.read_bytes()).hexdigest()
    d1["gate_evidence"] = [
        {
            "predicate": "original_worktree_mutations",
            "status": "PASS",
            "evidence": "acceptance.json",
        },
        {
            "predicate": "published_remote_ref_contains_commit",
            "status": "PASS",
            "evidence": remote_ref,
        },
    ]
    assert (
        validate(
            _registry([c0, d0, d1]),
            repo_root=tmp_path,
            expected_remote_url=str(remote),
        )
        == []
    )


def test_final_task_rejects_fake_or_unpublished_commit(tmp_path):
    c0 = _task("C0-05")
    d0 = _task("D0-05")
    acceptance, commit, _, remote = _commit_acceptance(tmp_path)
    d1 = _governed_task("D1-01", status="VERIFIED", deps=["C0-05", "D0-05"])
    d1.update(
        {
            "commit_sha": "a" * 40,
            "report": "FALSE_PASS",
            "acceptance_artifact": "acceptance.json",
            "acceptance_sha256": hashlib.sha256(acceptance.read_bytes()).hexdigest(),
            "gate_evidence": [
                {
                    "predicate": "published_remote_ref_contains_commit",
                    "status": "PASS",
                    "evidence": "refs/remotes/origin/missing",
                }
            ],
        }
    )
    errors = validate(
        _registry([c0, d0, d1]),
        repo_root=tmp_path,
        expected_remote_url=str(remote),
    )
    assert any("not an existing Git commit" in error for error in errors)

    d1["commit_sha"] = commit
    errors = validate(
        _registry([c0, d0, d1]),
        repo_root=tmp_path,
        expected_remote_url=str(remote),
    )
    assert any("published remote ref" in error for error in errors)


def test_final_task_rejects_wrong_phase_acceptance_or_uncommitted_tamper(
    tmp_path,
):
    c0 = _task("C0-05")
    d0 = _task("D0-05")
    acceptance, commit, remote_ref, remote = _commit_acceptance(tmp_path)
    d1 = _governed_task("D1-01", status="VERIFIED", deps=["C0-05", "D0-05"])
    d1.update(
        {
            "commit_sha": commit,
            "report": "FALSE_PASS",
            "acceptance_artifact": "acceptance.json",
            "gate_evidence": [
                {
                    "predicate": "published_remote_ref_contains_commit",
                    "status": "PASS",
                    "evidence": remote_ref,
                }
            ],
        }
    )
    acceptance.write_text(
        json.dumps(
            {
                "schema_version": "utr_b0_acceptance.v2",
                "b0_gate_passed": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    d1["acceptance_sha256"] = hashlib.sha256(acceptance.read_bytes()).hexdigest()
    errors = validate(
        _registry([c0, d0, d1]),
        repo_root=tmp_path,
        expected_remote_url=str(remote),
    )
    assert any("D1 acceptance schema_version is invalid" in error for error in errors)
    assert any("hash-matching blob" in error for error in errors)


def test_smoke_or_fixture_evidence_cannot_close_phase_gate(tmp_path):
    c0 = _task("C0-05")
    d0 = _task("D0-05")
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    d1 = _governed_task("D1-08", status="FROZEN", deps=["C0-05", "D0-05"])
    d1.update(
        {
            "evidence_class": "SMOKE",
            "commit_sha": "b" * 40,
            "report": "SMOKE_ONLY",
            "acceptance_artifact": "acceptance.json",
            "acceptance_sha256": hashlib.sha256(acceptance.read_bytes()).hexdigest(),
            "gate_evidence": [
                {
                    "predicate": "fixture",
                    "status": "PASS",
                    "evidence": "acceptance.json",
                }
            ],
        }
    )
    errors = validate(_registry([c0, d0, d1]), repo_root=tmp_path)
    assert any("cannot satisfy a phase gate" in error for error in errors)


def test_d1_08_frozen_requires_machine_checked_snapshot_binding(
    tmp_path,
):
    c0 = _task("C0-05")
    d0 = _task("D0-05")
    acceptance, commit, remote_ref, remote = _commit_acceptance(tmp_path)
    d1 = _governed_task("D1-08", status="FROZEN", deps=["C0-05", "D0-05"])
    d1.update(
        {
            "commit_sha": commit,
            "report": "D1_STRUCTURAL_DATA_ONLY",
            "acceptance_artifact": "acceptance.json",
            "acceptance_sha256": hashlib.sha256(acceptance.read_bytes()).hexdigest(),
            "gate_evidence": [
                {
                    "predicate": "published_remote_ref_contains_commit",
                    "status": "PASS",
                    "evidence": remote_ref,
                }
            ],
        }
    )
    errors = validate(
        _registry([c0, d0, d1]),
        repo_root=tmp_path,
        expected_remote_url=str(remote),
    )
    assert any(
        "snapshot_artifact must equal "
        "data/d1/manifests/d1_canonical_snapshot.json" in error
        for error in errors
    )


def test_active_b0_requires_frozen_d1_anchor():
    c0 = _task("C0-05")
    d0 = _task("D0-05")
    d1 = _governed_task("D1-08", status="REGISTERED", deps=["C0-05", "D0-05"])
    b0 = _governed_task("B0-01", status="RUNNING", deps=["D1-08"])
    errors = validate(_registry([c0, d0, d1, b0]))
    assert any("D1-08:FROZEN" in error for error in errors)


def test_registered_b0_inventory_is_not_execution():
    errors = validate(
        _registry(_release_inventory_tasks()),
        release_profile="D1_B0",
    )
    assert errors == []


def test_d1_b0_final_label_access_is_forbidden():
    c0 = _task("C0-05")
    d0 = _task("D0-05")
    d1 = _governed_task("D1-01", status="REGISTERED", deps=["C0-05", "D0-05"])
    d1["resource_labels"].append("FINAL_LABEL_ACCESS")
    errors = validate(_registry([c0, d0, d1]))
    assert any("FINAL_LABEL_ACCESS" in error for error in errors)


def test_allowed_parallel_must_be_reciprocal():
    c0 = _task("C0-05")
    d0 = _task("D0-05")
    left = _governed_task("D1-06", status="REGISTERED", deps=["C0-05", "D0-05"])
    right = _governed_task("D1-07", status="REGISTERED", deps=["C0-05", "D0-05"])
    left["allowed_parallel_tasks"] = ["D1-07"]
    errors = validate(_registry([c0, d0, left, right]))
    assert any("reciprocal" in error for error in errors)


def test_minimal_acceptance_stub_cannot_close_d1(tmp_path):
    c0 = _task("C0-05")
    d0 = _task("D0-05")
    acceptance, commit, remote_ref, remote = _commit_acceptance(tmp_path)
    acceptance.write_text(
        json.dumps(
            {
                "schema_version": "d1_acceptance_v2",
                "phase_gate_passed": True,
                "fixture_mode": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _git(tmp_path, "add", "acceptance.json")
    _git(tmp_path, "commit", "-m", "replace with stub")
    stub_commit = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "push", "origin", f"HEAD:{remote_ref}")
    d1 = _governed_task("D1-01", status="VERIFIED", deps=["C0-05", "D0-05"])
    d1.update(
        {
            "commit_sha": stub_commit,
            "report": "FALSE_PASS",
            "acceptance_artifact": "acceptance.json",
            "acceptance_sha256": hashlib.sha256(acceptance.read_bytes()).hexdigest(),
            "gate_evidence": [
                {
                    "predicate": "published_remote_ref_contains_commit",
                    "status": "PASS",
                    "evidence": remote_ref,
                }
            ],
        }
    )
    errors = validate(
        _registry([c0, d0, d1]),
        repo_root=tmp_path,
        expected_remote_url=str(remote),
    )
    assert any("missing required keys" in error for error in errors)


def test_nested_false_acceptance_predicate_cannot_close_d1(tmp_path):
    c0 = _task("C0-05")
    d0 = _task("D0-05")
    acceptance, _, remote_ref, remote = _commit_acceptance(tmp_path)
    payload = json.loads(acceptance.read_text(encoding="utf-8"))
    payload["required_artifact_validation"]["semantic_checks"][
        "all_path_bytes_sha_bindings_match"
    ] = False
    acceptance.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    _git(tmp_path, "add", "acceptance.json")
    _git(tmp_path, "commit", "-m", "nested false")
    commit = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "push", "origin", f"HEAD:{remote_ref}")
    d1 = _governed_task("D1-01", status="VERIFIED", deps=["C0-05", "D0-05"])
    d1.update(
        {
            "commit_sha": commit,
            "report": "FALSE_PASS",
            "acceptance_artifact": "acceptance.json",
            "acceptance_sha256": hashlib.sha256(acceptance.read_bytes()).hexdigest(),
            "gate_evidence": [
                {
                    "predicate": "published_remote_ref_contains_commit",
                    "status": "PASS",
                    "evidence": remote_ref,
                }
            ],
        }
    )
    errors = validate(
        _registry([c0, d0, d1]),
        repo_root=tmp_path,
        expected_remote_url=str(remote),
    )
    assert any("false nested gate predicates" in error for error in errors)


def test_local_tracking_ref_without_remote_is_not_publication(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "tests@example.invalid")
    _git(tmp_path, "config", "user.name", "tests")
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(
        json.dumps(valid_d1_acceptance(tmp_path), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _git(tmp_path, "add", "acceptance.json")
    _git(tmp_path, "commit", "-m", "local only")
    commit = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "update-ref", "refs/remotes/origin/forged", commit)
    d1 = _governed_task("D1-01", status="VERIFIED", deps=["C0-05", "D0-05"])
    d1.update(
        {
            "commit_sha": commit,
            "report": "FALSE_PUBLICATION",
            "acceptance_artifact": "acceptance.json",
            "acceptance_sha256": hashlib.sha256(acceptance.read_bytes()).hexdigest(),
            "gate_evidence": [
                {
                    "predicate": "published_remote_ref_contains_commit",
                    "status": "PASS",
                    "evidence": "refs/remotes/origin/forged",
                }
            ],
        }
    )
    errors = validate(
        _registry([_task("C0-05"), _task("D0-05"), d1]), repo_root=tmp_path
    )
    assert any("refs/heads" in error for error in errors)


def test_release_profile_requires_exact_d1_b0_inventory():
    tasks = [_governed_task(f"D1-{index:02d}") for index in range(1, 9)]
    errors = validate(_registry(tasks), release_profile="D1_B0")
    assert any("missing tasks" in error and "B0-05" in error for error in errors)


def test_release_profile_rejects_each_missing_inventory_item():
    expected = {
        *(f"D1-{index:02d}" for index in range(1, 9)),
        *(f"B0-{index:02d}" for index in range(1, 6)),
    }
    tasks = _release_inventory_tasks()
    for missing in sorted(expected):
        reduced = [task for task in tasks if task["task_id"] != missing]
        errors = validate(_registry(reduced), release_profile="D1_B0")
        assert any(
            "missing tasks" in error and missing in error for error in errors
        ), missing


def test_b0_phase_gate_cannot_skip_required_tasks():
    c0 = _task("C0-05")
    d0 = _task("D0-05")
    d1 = _governed_task("D1-08", status="FROZEN", deps=["C0-05", "D0-05"])
    b0_tasks = [
        _governed_task(
            f"B0-{index:02d}",
            status="FROZEN" if index == 5 else "REGISTERED",
            deps=["D1-08"] if index == 1 else [f"B0-{index - 1:02d}"],
        )
        for index in range(1, 6)
    ]
    errors = validate(_registry([c0, d0, d1, *b0_tasks]))
    for required in ("B0-01", "B0-02", "B0-03", "B0-04"):
        assert any(
            f"B0-05: cannot freeze before {required}:VERIFIED" in error
            for error in errors
        )


def test_d1_phase_gate_cannot_skip_required_tasks():
    tasks = _release_inventory_tasks()
    by_id = {task["task_id"]: task for task in tasks}
    for task_id in (f"D1-{index:02d}" for index in range(1, 8)):
        by_id[task_id]["status"] = "SAFE_PAUSED"
        by_id[task_id]["known_blockers"] = ["not completed"]
    by_id["D1-08"]["status"] = "FROZEN"
    errors = validate(_registry(tasks))
    for required in (f"D1-{index:02d}" for index in range(1, 8)):
        assert any(
            f"D1-08: cannot freeze before {required}:VERIFIED" in error
            for error in errors
        )


def test_release_profile_enforces_anchor_phase_gate_flags():
    tasks = [
        *[_governed_task(f"D1-{index:02d}") for index in range(1, 9)],
        *[_governed_task(f"B0-{index:02d}") for index in range(1, 6)],
    ]
    tasks[-1]["phase_gate"] = False
    errors = validate(_registry(tasks), release_profile="D1_B0")
    assert any("B0-05: phase_gate must be true" in error for error in errors)
