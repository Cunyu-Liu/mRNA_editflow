from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

import scripts.data.build_d1_canonical_snapshot as snapshot_builder
from scripts.data.build_d1_canonical_snapshot import (
    CODE_PATHS,
    EXPECTED_SCOPE,
    REQUIRED_ARTIFACTS,
    _control_file,
    build_snapshot_payload,
    write_json_exclusive,
)
from scripts.data.validate_d1_canonical_snapshot import validate_snapshot
from scripts.execution.validate_registry import _validate_d1_snapshot_binding
from tests.governance_fixtures import valid_d1_acceptance


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ref(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _snapshot_fixture(tmp_path: Path) -> tuple[Path, Path, str, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "snapshot@example.invalid")
    _git(repo, "config", "user.name", "snapshot")
    for relative in CODE_PATHS:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# frozen {relative}\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "frozen code")
    code_commit = _git(repo, "rev-parse", "HEAD")

    stage_root = tmp_path / "external-stage" / "D1"
    stage_root.mkdir(parents=True)
    acceptance = valid_d1_acceptance(stage_root)
    acceptance_results = {
        item["dataset_id"]: item for item in acceptance["dataset_results"]
    }
    dataset_summaries = []
    for dataset_id in sorted(EXPECTED_SCOPE):
        acceptance_result = acceptance_results[dataset_id]
        provenance_check = next(
            check
            for check in acceptance_result["checks"]
            if check["name"] == "production_input_provenance_complete"
        )
        raw_files = copy.deepcopy(provenance_check["detail"]["audit"]["raw_files"])
        dataset_root = stage_root / "datasets" / dataset_id
        dataset_root.mkdir(parents=True)
        output = dataset_root / "paper_clean.jsonl"
        output.write_text("", encoding="utf-8")
        manifest = {
            "dataset_id": dataset_id,
            "status": acceptance_result["status"],
            "paper_eligible": acceptance_result["paper_eligible"],
            "input_provenance": {
                "provenance_audit": {
                    "raw_files": raw_files,
                }
            },
            "outputs": {
                "paper_clean": {
                    "path": output.name,
                    "bytes": 0,
                    "sha256": _sha(output),
                }
            },
        }
        manifest_path = dataset_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        dataset_summaries.append(
            {
                "dataset_id": dataset_id,
                "status": acceptance_result["status"],
                "paper_eligible": acceptance_result["paper_eligible"],
                "fixture_mode": False,
                "accounting": {
                    "total_input_rows": (
                        1 if acceptance_result["status"] == "accepted" else 0
                    )
                },
                "manifest": {
                    "path": (f"datasets/{dataset_id}/manifest.json"),
                    "bytes": manifest_path.stat().st_size,
                    "sha256": _sha(manifest_path),
                },
            }
        )

    record_ids_sha = hashlib.sha256(b"").hexdigest()
    label_store = stage_root / "canonical/records_with_labels.jsonl"
    candidate_store = stage_root / "candidate_store/candidates.jsonl"
    label_store.parent.mkdir()
    candidate_store.parent.mkdir()
    label_store.write_text("", encoding="utf-8")
    candidate_store.write_text("", encoding="utf-8")

    control_dir = repo / "controls"
    control_dir.mkdir()
    config = control_dir / "config.json"
    scope = control_dir / "scope.yaml"
    inventory = control_dir / "inventory.json"
    config.write_text("{}\n", encoding="utf-8")
    scope.write_text("scope: exact\n", encoding="utf-8")
    inventory.write_text("{}\n", encoding="utf-8")

    artifact_root = repo / "contract-artifacts"
    artifact_bindings = {}
    for index, relative in enumerate(sorted(REQUIRED_ARTIFACTS), start=1):
        path = artifact_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"artifact-{index}\n", encoding="utf-8")
        artifact_bindings[relative] = {
            **_ref(path),
            "exists": True,
            "declared": _ref(path),
        }

    audit_root = tmp_path / "audit/attempt_002"
    audit_root.mkdir(parents=True)
    audit_manifest = audit_root / "audit_manifest.json"
    audit_manifest.write_text('{"state":"COMMAND_COMPLETED"}\n', encoding="utf-8")

    build_manifest = {
        "schema_version": "d1_build_manifest_v2",
        "stage_id": "D1_B0_20260728T160012Z_8862125",
        "config_path": str(config.resolve()),
        "config_bytes": config.stat().st_size,
        "config_sha256": _sha(config),
        "dataset_scope_manifest": {
            **_ref(scope),
            "repository_path": "controls/scope.yaml",
        },
        "input_inventory": {
            **_ref(inventory),
            "repository_path": "controls/inventory.json",
        },
        "datasets": dataset_summaries,
        "global_stores": {
            "canonical_label_store": {
                "path": "canonical/records_with_labels.jsonl",
                "bytes": 0,
                "sha256": _sha(label_store),
                "records": 0,
                "record_ids_sha256": record_ids_sha,
            },
            "sealed_label_free_candidate_store": {
                "path": "candidate_store/candidates.jsonl",
                "bytes": 0,
                "sha256": _sha(candidate_store),
                "records": 0,
                "record_ids_sha256": record_ids_sha,
            },
        },
    }
    build_manifest_path = stage_root / "build_manifest.json"
    build_manifest_path.write_text(
        json.dumps(build_manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    build_manifest_path.write_text(
        json.dumps(build_manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    acceptance["required_artifact_validation"]["artifacts"] = artifact_bindings
    acceptance["required_artifact_validation"]["build_manifest"] = _ref(
        build_manifest_path
    )
    acceptance["builder_audit_validation"] = {
        "passed": True,
        "checks": {"causal_chain": True},
        "audit_root": str(audit_root.resolve()),
        "audit_manifest": _ref(audit_manifest),
    }
    acceptance["config_binding_validation"].update(
        {
            "config_path": str(config.resolve()),
            "config_repository_path": "controls/config.json",
            "declared_bytes": config.stat().st_size,
            "declared_sha256": _sha(config),
            "scope_manifest_binding": {
                **_ref(scope),
                "repository_path": "controls/scope.yaml",
            },
            "input_inventory_binding": {
                **_ref(inventory),
                "repository_path": "controls/inventory.json",
            },
            "prelaunch_bindings": {
                name: {"passed": True, "source": "captured_head_blob"}
                for name in ("config", "scope", "input_inventory")
            },
        }
    )
    acceptance_path = (
        repo / "artifacts/stages/D1_B0_20260728T160012Z_8862125/D1/acceptance.json"
    )
    acceptance_path.parent.mkdir(parents=True)
    acceptance_path.write_text(
        json.dumps(acceptance, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    snapshot_path = repo / "data/d1/manifests/d1_canonical_snapshot.json"
    payload = build_snapshot_payload(
        acceptance_path=acceptance_path,
        repo_root=repo,
        code_commit=code_commit,
        generated_at_utc="2026-07-29T01:00:00+00:00",
    )
    write_json_exclusive(snapshot_path, payload)
    return repo, snapshot_path, code_commit, stage_root


def test_snapshot_exact_recomputation_passes(tmp_path: Path) -> None:
    repo, snapshot, _, _ = _snapshot_fixture(tmp_path)
    assert validate_snapshot(snapshot, repo_root=repo) == []


def test_snapshot_code_provenance_includes_acceptance_semantics(
    tmp_path: Path,
) -> None:
    repo, snapshot, _, _ = _snapshot_fixture(tmp_path)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    references = {item["path"]: item for item in payload["code_provenance"]["files"]}
    relative = "scripts/execution/acceptance_semantics.py"

    assert relative in CODE_PATHS
    assert set(references) == set(CODE_PATHS)
    assert references[relative] == {
        "path": relative,
        "bytes": (repo / relative).stat().st_size,
        "sha256": _sha(repo / relative),
    }


def test_snapshot_rejects_live_acceptance_semantics_drift(
    tmp_path: Path,
) -> None:
    repo, snapshot, _, _ = _snapshot_fixture(tmp_path)
    semantics = repo / "scripts/execution/acceptance_semantics.py"
    semantics.write_text(
        semantics.read_text(encoding="utf-8") + "# live drift\n",
        encoding="utf-8",
    )

    errors = validate_snapshot(snapshot, repo_root=repo)

    assert any(
        "snapshot_recompute_failure" in error and "acceptance_semantics.py" in error
        for error in errors
    )


def test_snapshot_rejects_acceptance_semantics_commit_drift(
    tmp_path: Path,
) -> None:
    repo, snapshot, _, _ = _snapshot_fixture(tmp_path)
    semantics = repo / "scripts/execution/acceptance_semantics.py"
    semantics.write_text(
        semantics.read_text(encoding="utf-8") + "# committed drift\n",
        encoding="utf-8",
    )
    _git(repo, "add", "scripts/execution/acceptance_semantics.py")
    _git(repo, "commit", "-m", "drift semantic gate")
    new_commit = _git(repo, "rev-parse", "HEAD")
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["code_provenance"]["code_commit_sha"] = new_commit
    snapshot.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    errors = validate_snapshot(snapshot, repo_root=repo)

    assert "snapshot_differs_from_exact_live_recomputation" in errors


@pytest.mark.parametrize(
    ("mutation", "needle"),
    [
        ("delete_provenance", "missing the required production provenance check"),
        ("promote_blocked", "was promoted from the frozen disposition"),
    ],
)
def test_snapshot_builder_and_validator_reject_d1_semantic_mutations(
    tmp_path: Path,
    mutation: str,
    needle: str,
) -> None:
    repo, snapshot, code_commit, _ = _snapshot_fixture(tmp_path)
    frozen = json.loads(snapshot.read_text(encoding="utf-8"))
    acceptance_path = repo / frozen["acceptance"]["path"]
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if mutation == "delete_provenance":
        result = acceptance["dataset_results"][0]
        result["checks"] = [
            check
            for check in result["checks"]
            if check["name"] != "production_input_provenance_complete"
        ]
    else:
        result = next(
            item
            for item in acceptance["dataset_results"]
            if item["status"] == "blocked"
        )
        result["status"] = "accepted"
        result["paper_eligible"] = True
    acceptance_path.write_text(
        json.dumps(acceptance, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=needle):
        build_snapshot_payload(
            acceptance_path=acceptance_path,
            repo_root=repo,
            code_commit=code_commit,
        )
    errors = validate_snapshot(snapshot, repo_root=repo)
    assert any(
        "snapshot_recompute_failure" in error and needle in error for error in errors
    )


def test_snapshot_rejects_promotion_against_bound_build_manifest_if_semantics_bypassed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, snapshot, code_commit, _ = _snapshot_fixture(tmp_path)
    frozen = json.loads(snapshot.read_text(encoding="utf-8"))
    acceptance_path = repo / frozen["acceptance"]["path"]
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    result = next(
        item for item in acceptance["dataset_results"] if item["status"] == "blocked"
    )
    result["status"] = "accepted"
    result["paper_eligible"] = True
    acceptance_path.write_text(
        json.dumps(acceptance, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        snapshot_builder,
        "validate_phase_acceptance",
        lambda *_args, **_kwargs: [],
    )

    with pytest.raises(
        ValueError,
        match="acceptance/build manifest status mismatch",
    ):
        build_snapshot_payload(
            acceptance_path=acceptance_path,
            repo_root=repo,
            code_commit=code_commit,
        )


def test_snapshot_rejects_paper_eligibility_drift_if_semantics_bypassed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, snapshot, code_commit, _ = _snapshot_fixture(tmp_path)
    frozen = json.loads(snapshot.read_text(encoding="utf-8"))
    acceptance_path = repo / frozen["acceptance"]["path"]
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    result = next(
        item for item in acceptance["dataset_results"] if item["status"] == "accepted"
    )
    result["paper_eligible"] = False
    acceptance_path.write_text(
        json.dumps(acceptance, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        snapshot_builder,
        "validate_phase_acceptance",
        lambda *_args, **_kwargs: [],
    )

    with pytest.raises(
        ValueError,
        match="acceptance/build manifest paper_eligible mismatch",
    ):
        build_snapshot_payload(
            acceptance_path=acceptance_path,
            repo_root=repo,
            code_commit=code_commit,
        )


def test_snapshot_builder_and_validator_reject_raw_file_binding_tamper(
    tmp_path: Path,
) -> None:
    repo, snapshot, code_commit, _ = _snapshot_fixture(tmp_path)
    frozen = json.loads(snapshot.read_text(encoding="utf-8"))
    acceptance_path = repo / frozen["acceptance"]["path"]
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    result = next(
        item for item in acceptance["dataset_results"] if item["status"] == "accepted"
    )
    provenance_check = next(
        check
        for check in result["checks"]
        if check["name"] == "production_input_provenance_complete"
    )
    provenance_check["detail"]["audit"]["raw_files"][0]["sha256"] = "f" * 64
    acceptance_path.write_text(
        json.dumps(acceptance, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="acceptance/dataset manifest raw_files mismatch",
    ):
        build_snapshot_payload(
            acceptance_path=acceptance_path,
            repo_root=repo,
            code_commit=code_commit,
        )
    errors = validate_snapshot(snapshot, repo_root=repo)
    assert any(
        "snapshot_recompute_failure" in error
        and "acceptance/dataset manifest raw_files mismatch" in error
        for error in errors
    )


def test_snapshot_rejects_live_external_artifact_replacement(
    tmp_path: Path,
) -> None:
    repo, snapshot, _, stage_root = _snapshot_fixture(tmp_path)
    target = stage_root / "datasets/GSE114002/paper_clean.jsonl"
    target.write_text('{"forged":true}\n', encoding="utf-8")
    errors = validate_snapshot(snapshot, repo_root=repo)
    assert any("snapshot_recompute_failure" in error for error in errors)


@pytest.mark.parametrize(
    "control_name",
    ["config", "dataset_scope_manifest", "input_inventory"],
)
def test_snapshot_build_rejects_control_file_changed_after_acceptance(
    tmp_path: Path,
    control_name: str,
) -> None:
    repo, snapshot, code_commit, _ = _snapshot_fixture(tmp_path)
    frozen = json.loads(snapshot.read_text(encoding="utf-8"))
    control_path = Path(frozen["control_files"][control_name]["path"])
    control_path.write_text(
        control_path.read_text(encoding="utf-8") + "# changed after acceptance\n",
        encoding="utf-8",
    )
    acceptance_path = repo / frozen["acceptance"]["path"]
    with pytest.raises(ValueError, match="live control file differs"):
        build_snapshot_payload(
            acceptance_path=acceptance_path,
            repo_root=repo,
            code_commit=code_commit,
        )


def test_snapshot_build_rejects_repository_control_copy_divergence(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    repository_copy = repo / "controls/inventory.json"
    repository_copy.parent.mkdir()
    repository_copy.write_text('{"source":"repository"}\n', encoding="utf-8")
    accepted_external = tmp_path / "accepted-inventory.json"
    accepted_external.write_text('{"source":"external"}\n', encoding="utf-8")
    declared = _ref(accepted_external)
    with pytest.raises(
        ValueError,
        match="canonical repository control file differs",
    ):
        _control_file(
            repo_root=repo,
            path=str(accepted_external),
            repository_path="controls/inventory.json",
            declared=declared,
            binding={"passed": True, "source": "captured_head_blob"},
        )


def test_snapshot_rejects_non_commit_git_object_as_code_provenance(
    tmp_path: Path,
) -> None:
    repo, snapshot, _, _ = _snapshot_fixture(tmp_path)
    frozen = json.loads(snapshot.read_text(encoding="utf-8"))
    tree_oid = _git(repo, "rev-parse", "HEAD^{tree}")
    acceptance_path = repo / frozen["acceptance"]["path"]
    with pytest.raises(ValueError, match="not an existing Git commit"):
        build_snapshot_payload(
            acceptance_path=acceptance_path,
            repo_root=repo,
            code_commit=tree_oid,
        )


def test_snapshot_rejects_claim_upgrade_and_exclusive_overwrite(
    tmp_path: Path,
) -> None:
    repo, snapshot, _, _ = _snapshot_fixture(tmp_path)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["claim_boundary"]["model_efficacy_claimed"] = True
    snapshot.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    errors = validate_snapshot(snapshot, repo_root=repo)
    assert any("model_efficacy_claimed" in error for error in errors)
    with pytest.raises(FileExistsError):
        write_json_exclusive(snapshot, payload)


def test_registry_snapshot_binding_requires_exact_evidence_commit_blob(
    tmp_path: Path,
) -> None:
    repo, snapshot, _, _ = _snapshot_fixture(tmp_path)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "D1 evidence")
    evidence_commit = _git(repo, "rev-parse", "HEAD")
    digest = _sha(snapshot)
    task = {
        "task_id": "D1-08",
        "snapshot_artifact": ("data/d1/manifests/d1_canonical_snapshot.json"),
        "snapshot_sha256": digest,
    }
    errors: list[str] = []
    _validate_d1_snapshot_binding(
        task,
        repo,
        evidence_commit,
        errors,
    )
    assert errors == []

    snapshot.write_text(
        snapshot.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    errors = []
    _validate_d1_snapshot_binding(
        task,
        repo,
        evidence_commit,
        errors,
    )
    assert any("live sha256 mismatch" in error for error in errors)
