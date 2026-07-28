from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import jsonschema
import yaml

from scripts.execution.validate_stage_completion import (
    GOAL_SHA256,
    _blob_sha256,
    validate,
)
from tests.governance_fixtures import (
    valid_b0_acceptance,
    valid_d1_acceptance,
)


ROOT = Path(__file__).resolve().parents[1]
STAGE_ID = "D1_B0_20260729T000000Z_1234567_A2"
REMOTE_REF = "refs/heads/release"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.decode("utf-8", errors="strict").strip()


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True)
    _git(path, "init")
    _git(path, "config", "user.email", "tests@example.invalid")
    _git(path, "config", "user.name", "tests")
    _git(path, "config", "core.autocrlf", "false")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="",
    )


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="")


def _reference(
    repo: Path,
    path: Path,
    *,
    role: str | None = None,
    commit_role: str | None = None,
) -> dict:
    value = {
        "path": path.resolve().relative_to(repo.resolve()).as_posix(),
        "sha256": _sha(path),
        "bytes": path.stat().st_size,
    }
    if role is not None:
        value = {"role": role, "commit_role": commit_role, **value}
    return value


def _task(
    task_id: str,
    *,
    status: str,
    dependencies: list[str] | None = None,
) -> dict:
    dependencies = dependencies or []
    return {
        "task_id": task_id,
        "phase_id": task_id.split("-", 1)[0],
        "status": status,
        "hypotheses": ["H1"],
        "dependencies": dependencies,
        "dependency_gates": [
            f"{dependency}:FROZEN" for dependency in dependencies
        ],
        "scientific_service": "Release-governed structural evidence.",
        "forbidden_actions": ["claim scientific efficacy"],
        "inputs": ["frozen inputs"],
        "resource_labels": ["CPU_LIGHT"],
        "conflict_keys": [],
        "allowed_parallel_tasks": [],
        "files": ["governed artifact"],
        "commands": ["validate"],
        "outputs": ["acceptance"],
        "acceptance": ["all gates pass or fail closed"],
        "repair_loop": ["preserve evidence and repair"],
        "commit_sha": None,
        "report": None,
    }


def _governed_task(
    task_id: str,
    *,
    status: str,
    dependencies: list[str],
    acceptance_path: str,
    acceptance_sha256: str,
    evidence_commit: str,
) -> dict:
    task = _task(
        task_id,
        status=status,
        dependencies=dependencies,
    )
    is_final = status in {"VERIFIED", "FROZEN"}
    task.update(
        {
            "evidence_class": (
                "FULL_SCOPE_DATA"
                if task_id.startswith("D1-")
                else "FULL_SCOPE_BENCHMARK"
            ),
            "completion_policy": "MUST_PASS_ALL",
            "acceptance_artifact": acceptance_path,
            "acceptance_sha256": acceptance_sha256,
            "gate_evidence": (
                [
                    {
                        "predicate": "published_remote_ref_contains_commit",
                        "status": "PASS",
                        "evidence": REMOTE_REF,
                    }
                ]
                if is_final
                else []
            ),
            "known_blockers": (
                [] if is_final else ["phase gate failed; evidence retained"]
            ),
            "phase_gate": task_id in {"D1-08", "B0-05"},
            "commit_sha": evidence_commit if is_final else None,
            "report": (
                "FULL_SCOPE_STRUCTURAL_EVIDENCE"
                if is_final
                else "SAFE_PAUSED_WITH_EVIDENCE"
            ),
        }
    )
    return task


def _registry(
    *,
    d1_reference: dict,
    b0_reference: dict,
    evidence_commit: str,
    status: str,
    d1_gate_passed: bool,
) -> dict:
    tasks = [
        _task("C0-05", status="FROZEN"),
        _task("D0-05", status="FROZEN"),
    ]
    for index in range(1, 9):
        task_id = f"D1-{index:02d}"
        if d1_gate_passed:
            task_status = "FROZEN" if task_id == "D1-08" else "VERIFIED"
        else:
            task_status = "SAFE_PAUSED"
        tasks.append(
            _governed_task(
                task_id,
                status=task_status,
                dependencies=["C0-05", "D0-05"],
                acceptance_path=d1_reference["path"],
                acceptance_sha256=d1_reference["sha256"],
                evidence_commit=evidence_commit,
            )
        )
    for index in range(1, 6):
        task_id = f"B0-{index:02d}"
        if status == "FROZEN":
            task_status = "FROZEN" if task_id == "B0-05" else "VERIFIED"
        else:
            task_status = "SAFE_PAUSED"
        dependencies = ["D1-08"]
        task = _governed_task(
            task_id,
            status=task_status,
            dependencies=dependencies,
            acceptance_path=b0_reference["path"],
            acceptance_sha256=b0_reference["sha256"],
            evidence_commit=evidence_commit,
        )
        tasks.append(task)
    return {
        "registry_version": "2.0.0",
        "contract_id": "utr_editflow_goal_v2",
        "goal_contract_sha256": GOAL_SHA256,
        "tasks": tasks,
    }


def _fixture(
    tmp_path: Path,
    *,
    status: str = "FROZEN",
    failed_phase: str = "B0",
    protected_initial_dirty: bool = False,
    mutate_d1: Callable[[dict], None] | None = None,
    mutate_b0: Callable[[dict], None] | None = None,
) -> tuple[dict, Path, Path, Path, Path]:
    protected = tmp_path / "protected"
    _init_repo(protected)
    _write_text(protected / "protected.txt", "preserve me\n")
    _git(protected, "add", "protected.txt")
    _git(protected, "commit", "-m", "protected state")
    protected_head = _git(protected, "rev-parse", "HEAD")
    if protected_initial_dirty:
        _write_text(protected / "protected.txt", "initial user edit\n")
    protected_status = _git(
        protected, "status", "--porcelain=v2", "--untracked-files=all"
    ).splitlines()
    if protected_initial_dirty:
        assert protected_status
    else:
        assert protected_status == []
    dirty_diff = subprocess.run(
        ["git", "-C", str(protected), "diff", "--binary", "HEAD", "--"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    dirty_sha256 = hashlib.sha256(dirty_diff).hexdigest()

    repo = tmp_path / "repo"
    _init_repo(repo)
    remote = tmp_path / "remote.git"
    subprocess.run(
        ["git", "init", "--bare", str(remote)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _git(repo, "remote", "add", "origin", str(remote))

    _write_text(repo / "seed.txt", "base\n")
    _git(repo, "add", "seed.txt")
    _git(repo, "commit", "-m", "base")
    base_commit = _git(repo, "rev-parse", "HEAD")

    stage_root = repo / "artifacts" / "stages" / STAGE_ID
    code_path = repo / "scripts" / "release_guard.py"
    _write_text(code_path, "RELEASE_GUARD = True\n")
    code_manifest_path = stage_root / "release" / "code_manifest.json"
    _write_json(
        code_manifest_path,
        {
            "schema_version": "d1_b0_code_manifest.v1",
            "base_commit_sha": base_commit,
            "files": [_reference(repo, code_path)],
            "deleted_paths": [],
        },
    )
    _git(
        repo,
        "add",
        _reference(repo, code_path)["path"],
        _reference(repo, code_manifest_path)["path"],
    )
    _git(repo, "commit", "-m", "code")
    code_commit = _git(repo, "rev-parse", "HEAD")

    d1_path = stage_root / "D1" / "acceptance.json"
    d1_payload = valid_d1_acceptance(stage_root / "D1")
    if status != "FROZEN" and failed_phase == "D1":
        d1_payload["phase_gate_passed"] = False
        d1_payload["structural_validation_passed"] = False
        d1_payload["missing_required_datasets"] = ["GSE114002"]
    if mutate_d1 is not None:
        mutate_d1(d1_payload)
    _write_json(d1_path, d1_payload)

    b0_path = stage_root / "B0" / "acceptance.json"
    b0_payload = valid_b0_acceptance()
    if status != "FROZEN":
        b0_payload["b0_gate_passed"] = False
        b0_payload["failed_gates"] = [
            (
                "blocked_by_d1_gate"
                if failed_phase == "D1"
                else "path_leakage_zero"
            )
        ]
    if mutate_b0 is not None:
        mutate_b0(b0_payload)
    _write_json(b0_path, b0_payload)

    preflight_path = stage_root / "preflight_manifest.json"
    _write_json(
        preflight_path,
        {
            "stage_id": STAGE_ID,
            "git": {
                "original": {
                    "path": str(protected.resolve()),
                    "head": protected_head,
                    "dirty_diff_sha256": dirty_sha256,
                }
            },
        },
    )
    protected_state_path = stage_root / "protected_state.json"
    _write_json(
        protected_state_path,
        {
            "stage_id": STAGE_ID,
            "original_worktree": {
                "path": str(protected.resolve()),
                "head": protected_head,
                "dirty_diff_sha256": dirty_sha256,
                "status_porcelain_v2": protected_status,
            },
            "processes": [{"pid": 123}],
        },
    )
    independent_path = (
        repo
        / "docs"
        / "audits"
        / "2026-07-29-d1-b0-independent-gate-review.md"
    )
    _write_text(
        independent_path,
        "# Independent D1/B0 gate review\n\nNo scientific claim is made.\n",
    )
    evidence_paths = [
        d1_path,
        b0_path,
        preflight_path,
        protected_state_path,
        independent_path,
    ]
    _git(repo, "add", *[_reference(repo, path)["path"] for path in evidence_paths])
    _git(repo, "commit", "-m", "evidence")
    evidence_commit = _git(repo, "rev-parse", "HEAD")

    d1_ref = _reference(repo, d1_path)
    b0_ref = _reference(repo, b0_path)
    registry_path = repo / "docs" / "execution" / "task_registry_v2.yaml"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        yaml.safe_dump(
            _registry(
                d1_reference=d1_ref,
                b0_reference=b0_ref,
                evidence_commit=evidence_commit,
                status=status,
                d1_gate_passed=bool(d1_payload.get("phase_gate_passed")),
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
        newline="",
    )
    decision_path = repo / "docs" / "decision_log.md"
    _write_text(
        decision_path,
        "# Decision log\n\nAppend-only governance record.\n",
    )
    recheck_path = stage_root / "release" / "protection_recheck.json"
    _write_json(
        recheck_path,
        {
            "stage_id": STAGE_ID,
            "original_worktree": {
                "path": str(protected.resolve()),
                "head": protected_head,
                "dirty_diff_sha256": dirty_sha256,
                "status_porcelain_v2": protected_status,
            },
            "actions": {
                "processes_terminated": 0,
                "existing_processes_modified": 0,
                "original_worktree_mutations": 0,
                "existing_results_overwritten": 0,
                "raw_inputs_modified": 0,
            },
            "terminated_protected_pids": [],
        },
    )
    registry_paths = [registry_path, decision_path, recheck_path]
    _git(repo, "add", *[_reference(repo, path)["path"] for path in registry_paths])
    _git(repo, "commit", "-m", "registry")
    registry_commit = _git(repo, "rev-parse", "HEAD")

    artifacts = [
        _reference(
            repo,
            code_manifest_path,
            role="code_manifest",
            commit_role="code",
        ),
        _reference(
            repo, d1_path, role="d1_acceptance", commit_role="evidence"
        ),
        _reference(
            repo, b0_path, role="b0_acceptance", commit_role="evidence"
        ),
        _reference(
            repo,
            preflight_path,
            role="preflight_manifest",
            commit_role="evidence",
        ),
        _reference(
            repo,
            protected_state_path,
            role="protected_state",
            commit_role="evidence",
        ),
        _reference(
            repo,
            independent_path,
            role="independent_gate_review",
            commit_role="evidence",
        ),
        _reference(
            repo,
            registry_path,
            role="task_registry",
            commit_role="registry",
        ),
        _reference(
            repo,
            decision_path,
            role="decision_log",
            commit_role="registry",
        ),
        _reference(
            repo,
            recheck_path,
            role="protection_recheck",
            commit_role="registry",
        ),
    ]
    manifest = {
        "artifact_type": "stage_completion_manifest",
        "schema_version": "utr_stage_completion.v2",
        "stage_id": STAGE_ID,
        "phase_ids": ["D1", "B0"],
        "status": status,
        "started_at_utc": "2026-07-29T00:00:00Z",
        "ended_at_utc": "2026-07-29T01:00:00Z",
        "workload_class": "NON_NEURAL_DATA_BENCHMARK",
        "goal_contract": {
            "id": "utr_editflow_goal_v2",
            "sha256": GOAL_SHA256,
            "repository_snapshot": (
                "docs/contracts/mrna_latest_build_contract_v2.md"
            ),
        },
        "stage_root": stage_root.relative_to(repo).as_posix(),
        "phase_acceptance": {
            "D1": {
                **d1_ref,
                "schema_version": "d1_acceptance_v2",
                "gate_field": "phase_gate_passed",
                "gate_passed": bool(d1_payload.get("phase_gate_passed")),
            },
            "B0": {
                **b0_ref,
                "schema_version": "utr_b0_acceptance.v2",
                "gate_field": "b0_gate_passed",
                "gate_passed": bool(b0_payload.get("b0_gate_passed")),
            },
        },
        "artifacts": artifacts,
        "git": {
            "repository": str(repo.resolve()),
            "remote_name": "origin",
            "canonical_remote_url": str(remote.resolve()),
            "code_commit_sha": code_commit,
            "evidence_commit_sha": evidence_commit,
            "registry_commit_sha": registry_commit,
            "published_remote_ref": REMOTE_REF,
        },
        "protection": {
            "preflight_role": "preflight_manifest",
            "protected_state_role": "protected_state",
            "recheck_role": "protection_recheck",
            "original_worktree_path": str(protected.resolve()),
            "initial_head": protected_head,
            "final_head": protected_head,
            "initial_dirty_diff_sha256": dirty_sha256,
            "final_dirty_diff_sha256": dirty_sha256,
            "original_worktree_unchanged": True,
            "protected_pids": [123],
            "terminated_protected_pids": [],
            "processes_terminated": 0,
            "existing_results_overwritten": 0,
        },
        "execution_boundary": {
            "formal_neural_activity_started": False,
            "gpu_validation_started": False,
            "cuda_fallback_events": 0,
            "gpu_requirement_status": "NOT_APPLICABLE_NO_NEURAL_WORK",
            "smoke_or_proxy_is_final_evidence": False,
        },
        "claim_boundary": {
            "scientific_result_claimed": False,
            "efficacy_claimed": False,
            "sota_claimed": False,
            "foundation_status": "UNKNOWN_PENDING_FM0",
            "allowed_claim": "NONE",
        },
        "stop_reason": (
            None
            if status == "FROZEN"
            else (
                "D1 structural gate failed; B0 remained blocked."
                if failed_phase == "D1"
                else "B0 path leakage gate failed; evidence retained."
            )
        ),
        "known_deviations": (
            [] if status == "FROZEN" else ["B0 gate did not pass"]
        ),
    }
    manifest_path = stage_root / "release" / "completion_manifest.json"
    _write_json(manifest_path, manifest)
    _git(repo, "add", manifest_path.relative_to(repo).as_posix())
    _git(repo, "commit", "-m", "release manifest")
    _git(repo, "push", "origin", f"HEAD:{REMOTE_REF}")
    return manifest, repo, manifest_path, remote, protected


def _validate_fixture(
    manifest: dict,
    repo: Path,
    manifest_path: Path,
    remote: Path,
) -> list[str]:
    return validate(
        manifest,
        repo,
        manifest_path=manifest_path,
        expected_remote_url=str(remote),
    )


def test_completion_schema_is_valid_draft_2020_12() -> None:
    schema = json.loads(
        (ROOT / "schemas/stage_completion_manifest.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["$schema"].endswith("draft/2020-12/schema")
    assert schema["properties"]["workload_class"]["const"] == (
        "NON_NEURAL_DATA_BENCHMARK"
    )


def test_production_cli_does_not_expose_remote_identity_override() -> None:
    for relative in (
        "scripts/execution/validate_registry.py",
        "scripts/execution/validate_stage_completion.py",
    ):
        result = subprocess.run(
            [sys.executable, str(ROOT / relative), "--help"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert "--expected-remote-url" not in result.stdout


def test_decision_corrections_are_append_only_and_do_not_claim_approval() -> None:
    text = (ROOT / "docs" / "decision_log.md").read_text(encoding="utf-8")
    correction = "D-2026-07-29-GOVERNANCE-METADATA-CORRECTION-01"
    assert text.index(correction) > text.index("D-2026-07-29-D1-B0-SCOPE")
    correction_text = text[text.index(correction) :]
    assert "record_type: append_only_metadata_correction" in correction_text
    assert "recorded_value: true" in correction_text
    assert "corrected_value: false" in correction_text
    assert "approved_by_user: false" in correction_text
    assert "historical_records_modified: false" in correction_text


def test_frozen_completion_is_fully_bound_to_real_remote(
    tmp_path: Path,
) -> None:
    manifest, repo, manifest_path, remote, _ = _fixture(tmp_path)
    assert _validate_fixture(manifest, repo, manifest_path, remote) == []


def test_minimal_d1_stub_cannot_close_completion(tmp_path: Path) -> None:
    def minimize(payload: dict) -> None:
        payload.clear()
        payload.update(
            {
                "schema_version": "d1_acceptance_v2",
                "fixture_mode": False,
                "phase_gate_passed": True,
            }
        )

    manifest, repo, manifest_path, remote, _ = _fixture(
        tmp_path, mutate_d1=minimize
    )
    errors = _validate_fixture(manifest, repo, manifest_path, remote)
    assert any("missing required keys" in error for error in errors)


def test_nested_false_d1_predicate_cannot_close_completion(
    tmp_path: Path,
) -> None:
    def add_false(payload: dict) -> None:
        payload["required_artifact_validation"]["semantic_checks"][
            "all_path_bytes_sha_bindings_match"
        ] = False

    manifest, repo, manifest_path, remote, _ = _fixture(
        tmp_path, mutate_d1=add_false
    )
    errors = _validate_fixture(manifest, repo, manifest_path, remote)
    assert any("false nested gate predicates" in error for error in errors)


def test_nested_false_b0_predicate_cannot_close_completion(
    tmp_path: Path,
) -> None:
    def add_false(payload: dict) -> None:
        payload["track_role_audit"]["eligible_identity_binding_complete"] = False

    manifest, repo, manifest_path, remote, _ = _fixture(
        tmp_path, mutate_b0=add_false
    )
    errors = _validate_fixture(manifest, repo, manifest_path, remote)
    assert any("false nested gate predicates" in error for error in errors)


def test_local_tracking_ref_cannot_replace_real_remote_publication(
    tmp_path: Path,
) -> None:
    manifest, repo, manifest_path, remote, _ = _fixture(tmp_path)
    release = _git(repo, "rev-parse", "HEAD")
    _git(repo, "remote", "remove", "origin")
    _git(repo, "update-ref", "refs/remotes/origin/forged", release)
    manifest["git"]["published_remote_ref"] = "refs/remotes/origin/forged"
    errors = _validate_fixture(manifest, repo, manifest_path, remote)
    assert any("real heads/tags ref" in error for error in errors)
    assert any("remote identity" in error for error in errors)


def test_wrong_canonical_remote_identity_is_rejected(tmp_path: Path) -> None:
    manifest, repo, manifest_path, _, _ = _fixture(tmp_path)
    wrong = tmp_path / "wrong.git"
    errors = validate(
        manifest,
        repo,
        manifest_path=manifest_path,
        expected_remote_url=str(wrong),
    )
    assert any("remote identity" in error for error in errors)
    assert any("task_registry: " in error for error in errors)


def test_artifact_inventory_stage_root_and_blob_are_fail_closed(
    tmp_path: Path,
) -> None:
    manifest, repo, manifest_path, remote, _ = _fixture(tmp_path)
    missing = copy.deepcopy(manifest)
    missing["artifacts"] = missing["artifacts"][:-1]
    errors = _validate_fixture(missing, repo, manifest_path, remote)
    assert any("missing required roles" in error for error in errors)

    escaped = copy.deepcopy(manifest)
    preflight = next(
        item for item in escaped["artifacts"] if item["role"] == "preflight_manifest"
    )
    decision = repo / "docs" / "decision_log.md"
    preflight.update(_reference(repo, decision))
    errors = _validate_fixture(escaped, repo, manifest_path, remote)
    assert any("preflight_manifest escapes stage_root" in error for error in errors)

    broad_root = copy.deepcopy(manifest)
    broad_root["stage_root"] = "artifacts"
    errors = _validate_fixture(broad_root, repo, manifest_path, remote)
    assert any(
        "stage_root must equal artifacts/stages/<stage_id>" in error
        for error in errors
    )

    d1_path = repo / manifest["phase_acceptance"]["D1"]["path"]
    d1_path.write_text(d1_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    tampered = copy.deepcopy(manifest)
    replacement = _reference(repo, d1_path)
    next(
        item for item in tampered["artifacts"] if item["role"] == "d1_acceptance"
    ).update(replacement)
    tampered["phase_acceptance"]["D1"].update(replacement)
    errors = _validate_fixture(tampered, repo, manifest_path, remote)
    assert any("not the exact committed blob" in error for error in errors)


def test_commit_roles_require_ordered_ancestry(tmp_path: Path) -> None:
    manifest, repo, manifest_path, remote, _ = _fixture(tmp_path)
    reversed_roles = copy.deepcopy(manifest)
    reversed_roles["git"]["code_commit_sha"] = manifest["git"][
        "evidence_commit_sha"
    ]
    reversed_roles["git"]["evidence_commit_sha"] = manifest["git"][
        "code_commit_sha"
    ]
    errors = _validate_fixture(reversed_roles, repo, manifest_path, remote)
    assert any("code commit is not an ancestor of evidence" in error for error in errors)

    collapsed_roles = copy.deepcopy(manifest)
    collapsed_roles["git"]["evidence_commit_sha"] = manifest["git"][
        "code_commit_sha"
    ]
    errors = _validate_fixture(collapsed_roles, repo, manifest_path, remote)
    assert any("must be distinct stages" in error for error in errors)

    reversed_registry = copy.deepcopy(manifest)
    reversed_registry["git"]["evidence_commit_sha"] = manifest["git"][
        "registry_commit_sha"
    ]
    reversed_registry["git"]["registry_commit_sha"] = manifest["git"][
        "evidence_commit_sha"
    ]
    errors = _validate_fixture(reversed_registry, repo, manifest_path, remote)
    assert any(
        "evidence commit is not an ancestor of registry" in error
        for error in errors
    )

    _git(
        repo,
        "push",
        "--force",
        "origin",
        f"{manifest['git']['evidence_commit_sha']}:{REMOTE_REF}",
    )
    errors = _validate_fixture(manifest, repo, manifest_path, remote)
    assert any(
        "registry commit is not an ancestor of remote release" in error
        for error in errors
    )


def test_runtime_schema_uses_format_checker_and_rejects_nested_extra(
    tmp_path: Path,
) -> None:
    manifest, repo, manifest_path, remote, _ = _fixture(tmp_path)
    invalid = copy.deepcopy(manifest)
    invalid["started_at_utc"] = "2026-07-29T00:00:00"
    invalid["protection"]["unbound_assertion"] = True
    errors = _validate_fixture(invalid, repo, manifest_path, remote)
    assert any(
        "schema:started_at_utc" in error and "date-time" in error
        for error in errors
    )
    assert any(
        "schema:protection" in error and "Additional properties" in error
        for error in errors
    )


def test_truthful_failed_completion_is_valid_but_cannot_be_frozen(
    tmp_path: Path,
) -> None:
    manifest, repo, manifest_path, remote, _ = _fixture(
        tmp_path, status="FAILED_WITH_EVIDENCE"
    )
    assert _validate_fixture(manifest, repo, manifest_path, remote) == []

    false_freeze = copy.deepcopy(manifest)
    false_freeze["status"] = "FROZEN"
    false_freeze["stop_reason"] = None
    false_freeze["known_deviations"] = []
    errors = _validate_fixture(false_freeze, repo, manifest_path, remote)
    assert any("FROZEN requires passing D1 and B0" in error for error in errors)
    assert any("schema:phase_acceptance.B0.gate_passed" in error for error in errors)


def test_truthful_d1_stop_keeps_b0_inventory_without_freezing_anchors(
    tmp_path: Path,
) -> None:
    manifest, repo, manifest_path, remote, _ = _fixture(
        tmp_path,
        status="SAFE_PAUSED",
        failed_phase="D1",
    )
    assert _validate_fixture(manifest, repo, manifest_path, remote) == []


def test_protection_is_bound_to_evidence_and_live_worktree(
    tmp_path: Path,
) -> None:
    manifest, repo, manifest_path, remote, protected = _fixture(tmp_path)
    wrong_path = copy.deepcopy(manifest)
    wrong_path["protection"]["original_worktree_path"] = str(repo)
    errors = _validate_fixture(wrong_path, repo, manifest_path, remote)
    assert any("path is not evidence-bound" in error for error in errors)

    _write_text(protected / "untracked.txt", "unexpected mutation\n")
    errors = _validate_fixture(manifest, repo, manifest_path, remote)
    assert any("live original worktree status differs" in error for error in errors)


def test_protection_detects_content_change_with_same_porcelain_status(
    tmp_path: Path,
) -> None:
    manifest, repo, manifest_path, remote, protected = _fixture(
        tmp_path,
        protected_initial_dirty=True,
    )
    _write_text(protected / "protected.txt", "different user edit\n")
    errors = _validate_fixture(manifest, repo, manifest_path, remote)
    assert not any("live original worktree status differs" in error for error in errors)
    assert any("live original worktree dirty diff differs" in error for error in errors)


def test_binary_blob_hash_preserves_crlf_bytes(tmp_path: Path) -> None:
    repo = tmp_path / "binary"
    _init_repo(repo)
    path = repo / "crlf.bin"
    path.write_bytes(b"alpha\r\nbeta\r\n")
    _git(repo, "add", "crlf.bin")
    _git(repo, "commit", "-m", "binary exactness")
    commit = _git(repo, "rev-parse", "HEAD")
    assert _blob_sha256(repo, commit, "crlf.bin") == hashlib.sha256(
        b"alpha\r\nbeta\r\n"
    ).hexdigest()
