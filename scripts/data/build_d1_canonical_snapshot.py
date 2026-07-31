#!/usr/bin/env python3
"""Build an exclusive, recomputable D1 canonical structural-data snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import jsonschema

from scripts.execution.acceptance_semantics import validate_phase_acceptance


GOAL_SHA256 = "ff5a440910c9c8ef47e460b3f2d6a291c7fce12e687e090f2200dc79796a89c3"
SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "schemas/d1_canonical_snapshot.schema.json"
)
EXPECTED_SCOPE = frozenset(
    {
        "ENCSR854RUF_raw62",
        "GSE114002",
        "GSE145046",
        "GSE149487",
        "GSE173083",
        "GSE200304",
        "GSE207584",
        "GSE217518",
        "GSE246381",
        "GSE291719",
        "GSE330741",
        "MPRAu_processed_ENCSR854RUF",
    }
)
REQUIRED_ARTIFACTS = frozenset(
    {
        "data/data_exposure_ledger.jsonl",
        "data/library_ascertainment_report.json",
        "data/edit_script_ambiguity_report.json",
        "data/measured_action_coverage_report.json",
        "reports/data_reproduction/summary.csv",
    }
)
CODE_PATHS = (
    "schemas/d1_canonical_snapshot.schema.json",
    "schemas/task_registry.schema.json",
    "scripts/data/build_d1_utr_benchmark.py",
    "scripts/data/validate_d1_acceptance.py",
    "scripts/data/build_d1_canonical_snapshot.py",
    "scripts/data/validate_d1_canonical_snapshot.py",
    "scripts/execution/acceptance_semantics.py",
    "scripts/execution/validate_registry.py",
    "data/utr_benchmark_v2/d1_builder.py",
    "data/utr_benchmark_v2/d1_artifacts.py",
)
STAGE_ID_RE = re.compile(
    r"^D1_B0_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{7}(?:_A[0-9]+)?$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _repo_ref(repo_root: Path, path: Path) -> dict[str, Any]:
    root = repo_root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    relative = resolved.relative_to(root).as_posix()
    return {
        "path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _json_exact(left: Any, right: Any) -> bool:
    """Compare parsed JSON values without Python bool/int equality coercion."""

    options = {
        "ensure_ascii": False,
        "separators": (",", ":"),
        "sort_keys": True,
    }
    return json.dumps(left, **options) == json.dumps(right, **options)


def _safe_stage_child(stage_root: Path, relative: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"unsafe stage-relative path: {relative!r}")
    resolved = (stage_root / raw).resolve(strict=True)
    resolved.relative_to(stage_root.resolve(strict=True))
    return resolved


def _git_blob(repo_root: Path, commit: str, relative: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{commit}:{relative}"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"code path is absent from {commit}: {relative}")
    return result.stdout


def _require_git_commit(repo_root: Path, commit: str) -> None:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "-e", f"{commit}^{{commit}}"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"code_commit is not an existing Git commit: {commit}")


def _code_provenance(repo_root: Path, code_commit: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", code_commit):
        raise ValueError("code_commit must be a full 40-hex Git commit")
    _require_git_commit(repo_root, code_commit)
    files = []
    for relative in CODE_PATHS:
        blob = _git_blob(repo_root, code_commit, relative)
        live = repo_root / relative
        if not live.is_file() or live.read_bytes() != blob:
            raise ValueError(f"live code differs from code_commit for {relative}")
        files.append(
            {
                "path": relative,
                "bytes": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
            }
        )
    return {"code_commit_sha": code_commit, "files": files}


def _artifact_root(
    acceptance: Mapping[str, Any],
) -> Path:
    artifacts = acceptance["required_artifact_validation"]["artifacts"]
    roots = set()
    for relative, binding in artifacts.items():
        path = Path(str(binding["path"])).resolve(strict=True)
        suffix = Path(relative).parts
        root = path
        for _ in suffix:
            root = root.parent
        roots.add(root)
    if len(roots) != 1:
        raise ValueError("required artifacts do not share one artifact root")
    return roots.pop()


def _control_file(
    *,
    repo_root: Path,
    path: str,
    repository_path: str,
    declared: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    if binding.get("passed") is not True:
        raise ValueError(f"prelaunch binding did not pass: {repository_path}")
    result = _file_ref(Path(path))
    expected = {
        "path": str(Path(str(declared.get("path", ""))).resolve()),
        "bytes": declared.get("bytes"),
        "sha256": declared.get("sha256"),
    }
    if not _json_exact(result, expected):
        raise ValueError(
            f"live control file differs from accepted binding: {repository_path}"
        )
    raw_repository_path = Path(repository_path)
    if raw_repository_path.is_absolute() or ".." in raw_repository_path.parts:
        raise ValueError(f"unsafe control repository path: {repository_path!r}")
    resolved_repo_root = repo_root.resolve(strict=True)
    repository_copy = (resolved_repo_root / raw_repository_path).resolve(strict=True)
    repository_copy.relative_to(resolved_repo_root)
    repository_ref = _file_ref(repository_copy)
    if (
        repository_ref["bytes"] != result["bytes"]
        or repository_ref["sha256"] != result["sha256"]
    ):
        raise ValueError(
            f"canonical repository control file differs from accepted binding: "
            f"{repository_path}"
        )
    result.update(
        {
            "repository_path": raw_repository_path.as_posix(),
            "prelaunch_binding_source": str(binding.get("source")),
        }
    )
    return result


def _stage_id_from_acceptance_path(
    repo_root: Path,
    acceptance_path: Path,
) -> str:
    resolved_repo_root = repo_root.resolve(strict=True)
    resolved_acceptance = acceptance_path.resolve(strict=True)
    try:
        relative = resolved_acceptance.relative_to(resolved_repo_root)
    except ValueError as exc:
        raise ValueError("D1 acceptance must be inside the repository") from exc
    parts = relative.parts
    if (
        len(parts) != 5
        or parts[:2] != ("artifacts", "stages")
        or parts[3:] != ("D1", "acceptance.json")
    ):
        raise ValueError(
            "D1 acceptance path must be "
            "artifacts/stages/<stage_id>/D1/acceptance.json"
        )
    stage_id = parts[2]
    if STAGE_ID_RE.fullmatch(stage_id) is None:
        raise ValueError(f"invalid D1 stage_id in acceptance path: {stage_id!r}")
    return stage_id


def _input_inventory_repository_path(
    *,
    build_manifest: Mapping[str, Any],
    accepted_inventory: Mapping[str, Any],
    stage_id: str,
) -> str:
    build_inventory = build_manifest.get("input_inventory")
    required_keys = {"path", "bytes", "sha256", "repository_path"}
    if not isinstance(build_inventory, Mapping) or set(build_inventory) != required_keys:
        raise ValueError(
            "build manifest input_inventory must contain exactly "
            "path, bytes, sha256, and repository_path"
        )
    for key in ("path", "bytes", "sha256"):
        if key not in accepted_inventory or not _json_exact(
            build_inventory[key],
            accepted_inventory[key],
        ):
            raise ValueError(
                "build manifest and accepted input inventory binding differ"
            )
    if accepted_inventory.get("legacy_inference_used") is not False:
        raise ValueError(
            "accepted input inventory must explicitly disable legacy inference"
        )
    repository_path = build_inventory["repository_path"]
    expected_repository_path = (
        f"artifacts/stages/{stage_id}/D1/input_inventory.json"
    )
    if (
        not isinstance(repository_path, str)
        or repository_path != expected_repository_path
    ):
        raise ValueError(
            "build manifest input_inventory repository_path is outside "
            "the canonical D1 stage"
        )
    accepted_repository_path = accepted_inventory.get("repository_path")
    if (
        accepted_repository_path is not None
        and not _json_exact(accepted_repository_path, repository_path)
    ):
        raise ValueError(
            "build manifest and accepted input inventory repository_path differ"
        )
    return repository_path


def _validate_stage_control_content(
    *,
    stage_id: str,
    config_payload: Mapping[str, Any],
    inventory_payload: Mapping[str, Any],
    build_manifest: Mapping[str, Any],
    config_repository_path: str,
) -> None:
    if build_manifest.get("schema_version") != "d1_build_snapshot_v2":
        raise ValueError("unsupported D1 build manifest schema")
    if config_payload.get("schema_version") != "d1_build_config_v2":
        raise ValueError("unsupported D1 build config schema")
    if not _json_exact(config_payload.get("stage_id"), stage_id):
        raise ValueError("D1 build config stage_id differs from acceptance path")
    if not _json_exact(
        config_payload.get("config_repository_path"),
        config_repository_path,
    ):
        raise ValueError("D1 build config repository path binding differs")
    if not _json_exact(
        build_manifest.get("config_repository_path"),
        config_repository_path,
    ):
        raise ValueError("D1 build manifest config repository path binding differs")
    if inventory_payload.get("artifact_type") != "d1_input_inventory":
        raise ValueError("unsupported D1 input inventory artifact type")
    if inventory_payload.get("schema_version") != "d1_input_inventory.v1":
        raise ValueError("unsupported D1 input inventory schema")
    if not _json_exact(inventory_payload.get("stage_id"), stage_id):
        raise ValueError("D1 input inventory stage_id differs from acceptance path")
    if inventory_payload.get("selection_is_label_independent") is not True:
        raise ValueError("D1 input inventory selection is not label-independent")
    if not _json_exact(
        config_payload.get("input_inventory"),
        build_manifest.get("input_inventory"),
    ):
        raise ValueError(
            "D1 build config and build manifest input inventory bindings differ"
        )


def build_snapshot_payload(
    *,
    acceptance_path: Path,
    repo_root: Path,
    code_commit: str,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    acceptance_path = acceptance_path.resolve(strict=True)
    stage_id = _stage_id_from_acceptance_path(repo_root, acceptance_path)
    acceptance = _load_json(acceptance_path)
    semantic_errors = validate_phase_acceptance("D1", acceptance, require_pass=True)
    if semantic_errors:
        raise ValueError(
            "D1 acceptance is not a semantic PASS: " + "; ".join(semantic_errors)
        )

    stage_root = Path(str(acceptance["stage_d1_root"])).resolve(strict=True)
    build_manifest_path = stage_root / "build_manifest.json"
    build_manifest = _load_json(build_manifest_path)
    if build_manifest.get("schema_version") != "d1_build_snapshot_v2":
        raise ValueError("unsupported D1 build manifest schema")
    declared_stage_id = build_manifest.get("stage_id")
    if declared_stage_id is not None and not _json_exact(declared_stage_id, stage_id):
        raise ValueError("build manifest stage_id differs from acceptance path")
    declared_build = acceptance["required_artifact_validation"]["build_manifest"]
    if _file_ref(build_manifest_path) != declared_build:
        raise ValueError("acceptance build_manifest binding is stale")

    dataset_results = {
        str(item["dataset_id"]): item for item in acceptance["dataset_results"]
    }
    if set(dataset_results) != EXPECTED_SCOPE:
        raise ValueError("acceptance dataset scope is not exact")
    summaries = {str(item["dataset_id"]): item for item in build_manifest["datasets"]}
    if set(summaries) != EXPECTED_SCOPE:
        raise ValueError("build manifest dataset scope is not exact")
    datasets = []
    for dataset_id in sorted(EXPECTED_SCOPE):
        result = dataset_results[dataset_id]
        summary = summaries[dataset_id]
        if not isinstance(result.get("status"), str) or summary.get(
            "status"
        ) != result.get("status"):
            raise ValueError(f"acceptance/build manifest status mismatch: {dataset_id}")
        if not isinstance(result.get("paper_eligible"), bool) or summary.get(
            "paper_eligible"
        ) is not result.get("paper_eligible"):
            raise ValueError(
                f"acceptance/build manifest paper_eligible mismatch: {dataset_id}"
            )
        binding = summary["manifest"]
        manifest_path = _safe_stage_child(stage_root, str(binding["path"]))
        manifest = _load_json(manifest_path)
        if _file_ref(manifest_path) != {
            "path": str(manifest_path),
            "bytes": binding["bytes"],
            "sha256": binding["sha256"],
        }:
            raise ValueError(f"stale dataset manifest binding: {dataset_id}")
        if manifest.get("status") != result.get("status"):
            raise ValueError(
                f"acceptance/dataset manifest status mismatch: {dataset_id}"
            )
        if manifest.get("paper_eligible") is not result.get("paper_eligible"):
            raise ValueError(
                f"acceptance/dataset manifest paper_eligible mismatch: {dataset_id}"
            )
        provenance_checks = [
            check
            for check in result.get("checks", [])
            if (
                isinstance(check, Mapping)
                and check.get("name") == "production_input_provenance_complete"
            )
        ]
        if len(provenance_checks) != 1:
            raise ValueError(
                f"acceptance production provenance inventory mismatch: {dataset_id}"
            )
        acceptance_raw_files = (
            provenance_checks[0].get("detail", {}).get("audit", {}).get("raw_files")
        )
        manifest_raw_files = (
            manifest.get("input_provenance", {})
            .get("provenance_audit", {})
            .get("raw_files")
        )
        if not _json_exact(acceptance_raw_files, manifest_raw_files):
            raise ValueError(
                f"acceptance/dataset manifest raw_files mismatch: {dataset_id}"
            )
        if result["status"] == "accepted":
            if not isinstance(acceptance_raw_files, list) or not acceptance_raw_files:
                raise ValueError(
                    f"accepted dataset has no bound raw_files: {dataset_id}"
                )
        elif acceptance_raw_files != [] or manifest_raw_files != []:
            raise ValueError(f"blocked dataset binds non-empty raw_files: {dataset_id}")
        outputs = {
            name: _file_ref(
                _safe_stage_child(manifest_path.parent, str(output["path"]))
            )
            for name, output in sorted(manifest["outputs"].items())
        }
        for name, output in manifest["outputs"].items():
            if (
                outputs[name]["bytes"] != output["bytes"]
                or outputs[name]["sha256"] != output["sha256"]
            ):
                raise ValueError(f"stale dataset output binding: {dataset_id}:{name}")
        datasets.append(
            {
                "dataset_id": dataset_id,
                "status": result["status"],
                "paper_eligible": result["paper_eligible"],
                "fixture_mode": result["fixture_mode"],
                "counts": dict(result["counts"]),
                "manifest": _file_ref(manifest_path),
                "outputs": outputs,
            }
        )

    stores = build_manifest["global_stores"]
    label_meta = stores["canonical_label_store"]
    candidate_meta = stores["sealed_label_free_candidate_store"]

    def global_ref(meta: Mapping[str, Any]) -> dict[str, Any]:
        result = _file_ref(_safe_stage_child(stage_root, str(meta["path"])))
        if result["bytes"] != meta["bytes"] or result["sha256"] != meta["sha256"]:
            raise ValueError("stale global-store binding")
        result.update(
            {
                "records": int(meta["records"]),
                "record_ids_sha256": str(meta["record_ids_sha256"]),
            }
        )
        return result

    label_ref = global_ref(label_meta)
    candidate_ref = global_ref(candidate_meta)
    if (
        label_ref["path"] == candidate_ref["path"]
        or label_ref["record_ids_sha256"] != candidate_ref["record_ids_sha256"]
    ):
        raise ValueError("global label/candidate stores are not safely paired")

    artifacts = acceptance["required_artifact_validation"]["artifacts"]
    if set(artifacts) != REQUIRED_ARTIFACTS:
        raise ValueError("required D1 artifact scope is not exact")
    required_artifacts = {}
    for relative in sorted(REQUIRED_ARTIFACTS):
        actual = _file_ref(Path(str(artifacts[relative]["path"])))
        if (
            actual["bytes"] != artifacts[relative]["bytes"]
            or actual["sha256"] != artifacts[relative]["sha256"]
        ):
            raise ValueError(f"stale required artifact binding: {relative}")
        required_artifacts[relative] = actual

    audit_validation = acceptance["builder_audit_validation"]
    audit_root = Path(str(audit_validation["audit_root"])).resolve(strict=True)
    audit_manifest = _file_ref(audit_root / "audit_manifest.json")
    if audit_manifest != audit_validation["audit_manifest"]:
        raise ValueError("builder audit manifest binding is stale")

    config_validation = acceptance["config_binding_validation"]
    prelaunch = config_validation["prelaunch_bindings"]
    scope = config_validation["scope_manifest_binding"]
    inventory = config_validation["input_inventory_binding"]
    config_repository_path = config_validation.get(
        "config_repository_path",
        "configs/d1_build_20260729.json",
    )
    if not isinstance(config_repository_path, str):
        raise ValueError("D1 config repository path must be a string")
    inventory_repository_path = _input_inventory_repository_path(
        build_manifest=build_manifest,
        accepted_inventory=inventory,
        stage_id=stage_id,
    )
    control_files = {
        "config": _control_file(
            repo_root=repo_root,
            path=str(config_validation["config_path"]),
            repository_path=config_repository_path,
            declared={
                "path": config_validation["config_path"],
                "bytes": config_validation["declared_bytes"],
                "sha256": config_validation["declared_sha256"],
            },
            binding=prelaunch["config"],
        ),
        "dataset_scope_manifest": _control_file(
            repo_root=repo_root,
            path=str(scope["path"]),
            repository_path=str(
                scope.get(
                    "repository_path",
                    "data_registry/d1_dataset_scope_manifest.yaml",
                )
            ),
            declared=scope,
            binding=prelaunch["scope"],
        ),
        "input_inventory": _control_file(
            repo_root=repo_root,
            path=str(inventory["path"]),
            repository_path=inventory_repository_path,
            declared=inventory,
            binding=prelaunch["input_inventory"],
        ),
    }
    config_payload = _load_json(Path(str(config_validation["config_path"])))
    inventory_payload = _load_json(Path(str(inventory["path"])))
    _validate_stage_control_content(
        stage_id=stage_id,
        config_payload=config_payload,
        inventory_payload=inventory_payload,
        build_manifest=build_manifest,
        config_repository_path=config_repository_path,
    )
    if not _json_exact(
        build_manifest.get("dataset_scope_manifest"),
        {
            key: scope[key]
            for key in ("path", "bytes", "sha256", "repository_path")
            if key in scope
        },
    ):
        raise ValueError("build manifest and accepted scope binding differ")
    if (
        not _json_exact(
            build_manifest.get("config_path"),
            config_validation["config_path"],
        )
        or not _json_exact(
            build_manifest.get("config_bytes"),
            config_validation["declared_bytes"],
        )
        or not _json_exact(
            build_manifest.get("config_sha256"),
            config_validation["declared_sha256"],
        )
    ):
        raise ValueError("build manifest and accepted config binding differ")
    summary = {
        "phase_gate_passed": acceptance["phase_gate_passed"],
        "structural_validation_passed": acceptance["structural_validation_passed"],
        "dataset_count": len(dataset_results),
        "dataset_results_passed": all(
            item["passed"] is True for item in dataset_results.values()
        ),
        "required_supported_datasets": sorted(
            acceptance["required_supported_datasets"]
        ),
        "missing_required_datasets": acceptance["missing_required_datasets"],
        "missing_d1_scope_datasets": acceptance["missing_d1_scope_datasets"],
        "required_artifacts_passed": acceptance["required_artifact_validation"][
            "passed"
        ],
        "global_stores_passed": acceptance["global_store_validation"]["passed"],
        "config_binding_passed": config_validation["passed"],
        "dataset_manifest_binding_passed": acceptance[
            "dataset_manifest_binding_validation"
        ]["passed"],
        "builder_audit_passed": audit_validation["passed"],
        "scientific_result_claimed": acceptance["scientific_result_claimed"],
    }
    if not all(
        value is True for key, value in summary.items() if key.endswith("_passed")
    ):
        raise ValueError("acceptance summary contains a failed D1 gate")

    return {
        "artifact_type": "d1_canonical_snapshot",
        "schema_version": "d1_canonical_snapshot.v1",
        "generated_at_utc": generated_at_utc or datetime.now(timezone.utc).isoformat(),
        "stage_id": stage_id,
        "status": "FROZEN_STRUCTURAL_DATA_ONLY",
        "goal_contract": {
            "id": "mrna_editflow_single_active_contract",
            "sha256": GOAL_SHA256,
        },
        "claim_boundary": {
            "scientific_result_claimed": False,
            "model_efficacy_claimed": False,
            "biological_improvement_claimed": False,
            "sota_claimed": False,
            "prospective_validity_claimed": False,
        },
        "acceptance": _repo_ref(repo_root, acceptance_path),
        "stage_d1_root": str(stage_root),
        "artifact_root": str(_artifact_root(acceptance)),
        "build_manifest": _file_ref(build_manifest_path),
        "builder_audit": {
            "audit_root": str(audit_root),
            "audit_manifest": audit_manifest,
            "causal_chain_passed": True,
        },
        "control_files": control_files,
        "dataset_scope": sorted(EXPECTED_SCOPE),
        "datasets": datasets,
        "global_stores": {
            "canonical_label_store": label_ref,
            "sealed_label_free_candidate_store": candidate_ref,
            "paths_distinct": True,
            "record_ids_sha256": label_ref["record_ids_sha256"],
        },
        "required_artifacts": required_artifacts,
        "semantic_summary": summary,
        "code_provenance": _code_provenance(repo_root, code_commit),
    }


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite snapshot: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        handle.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_payload_schema(payload: Mapping[str, Any]) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(
        schema,
        format_checker=jsonschema.FormatChecker(),
    )
    errors = sorted(
        validator.iter_errors(payload),
        key=lambda item: tuple(str(part) for part in item.path),
    )
    if errors:
        detail = "; ".join(
            (".".join(str(part) for part in error.path) or "<root>")
            + ":"
            + error.message
            for error in errors
        )
        raise ValueError(f"generated snapshot schema invalid: {detail}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = build_snapshot_payload(
        acceptance_path=args.acceptance,
        repo_root=args.repo_root,
        code_commit=args.code_commit,
    )
    validate_payload_schema(payload)
    write_json_exclusive(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
