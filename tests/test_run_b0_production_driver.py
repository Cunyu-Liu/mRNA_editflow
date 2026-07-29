from __future__ import annotations

import copy
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts.execution import b0_driver_guard
from tests.governance_fixtures import valid_d1_acceptance


SOURCE_ROOT = Path(__file__).resolve().parents[1]
DRIVER_RELATIVE = Path("scripts/data/run_b0_production.sh")
DRIVER_TEST_TIMEOUT_ENV = "B0_DRIVER_TEST_TIMEOUT_SECONDS"


def _driver_test_timeout_seconds() -> int:
    raw = os.environ.get(DRIVER_TEST_TIMEOUT_ENV, "600")
    try:
        timeout_seconds = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"{DRIVER_TEST_TIMEOUT_ENV} must be a positive integer"
        ) from exc
    if timeout_seconds <= 0:
        raise RuntimeError(f"{DRIVER_TEST_TIMEOUT_ENV} must be a positive integer")
    return timeout_seconds


DRIVER_TEST_TIMEOUT_SECONDS = _driver_test_timeout_seconds()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ref(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _attempt_ref(path: Path, attempt_root: Path) -> dict[str, object]:
    reference = _ref(path)
    reference["path"] = (
        path.resolve(strict=True)
        .relative_to(attempt_root.resolve(strict=True))
        .as_posix()
    )
    return reference


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


FAKE_B0_SCRIPT = r"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path


def arg(name: str) -> str:
    index = sys.argv.index(name)
    return sys.argv[index + 1]


def args_all(name: str) -> list[str]:
    return [
        sys.argv[index + 1]
        for index, value in enumerate(sys.argv[:-1])
        if value == name
    ]


def ref(path_text: str | Path) -> dict[str, object]:
    path = Path(path_text).resolve(strict=True)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def write_json_exclusive(path_text: str, payload: object) -> None:
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


name = Path(__file__).name
if (
    os.environ.get("B0_TEST_MUTATE_TRACKED_CODE") == "1"
    and name == "build_b0_splits.py"
    and "--validate-canonical-only" in sys.argv
):
    Path("data/utr_benchmark_v2/track_loader.py").write_text(
        "# injected code drift\n",
        encoding="utf-8",
    )
if (
    os.environ.get("B0_TEST_MUTATE_RUNTIME_OVERLAY_PATH")
    and name == "build_b0_splits.py"
    and "--validate-canonical-only" in sys.argv
):
    overlay = Path(os.environ["B0_TEST_MUTATE_RUNTIME_OVERLAY_PATH"])
    overlay.write_bytes(overlay.read_bytes() + b"# injected runtime drift\n")
if name == "build_b0_splits.py":
    output = arg("--output")
    if "--validate-canonical-only" in sys.argv:
        sleep_seconds = float(
            os.environ.get("B0_TEST_SLEEP_CANONICAL_SECONDS", "0")
        )
        if sleep_seconds:
            time.sleep(sleep_seconds)
        write_json_exclusive(
            output,
            {
                "schema_version": "utr_b0_canonical_schema_validation.v2",
                "status": "PASS",
                "invalid_record_count": 0,
                "d1_acceptance_bound": True,
                "d1_binding": {"passed": True},
                "legacy_schema_only_validation": False,
                "d1_acceptance_path": arg("--d1-acceptance"),
            },
        )
    else:
        canonical = json.loads(
            Path(arg("--canonical-validation-report")).read_text(encoding="utf-8")
        )
        kind = arg("--split-kind")
        payload = {
            "status": "READY",
            "d1_phase_gate_passed": True,
            "d1_acceptance_path": canonical["d1_acceptance_path"],
            "canonical_validation_report_path": arg(
                "--canonical-validation-report"
            ),
            "partitions": [{"status": "READY"}],
            "partitions_sha256": "a" * 64,
            "split_kind": kind,
        }
        if kind == "cross_region_transfer":
            payload.update(
                {
                    "source_region": arg("--source-region"),
                    "target_region": arg("--target-region"),
                }
            )
        else:
            payload["region"] = arg("--region")
        write_json_exclusive(output, payload)
elif name == "audit_b0_leakage.py":
    write_json_exclusive(
        arg("--output"),
        {
            "gate_passed": True,
            "recomputed_from_bound_structural_records": True,
            "canonical_manifest_exact_recomputation": True,
            "foundation_pretraining_overlap": {
                "status": "UNKNOWN_PENDING_FM0",
                "foundation_selected": False,
                "allowed_claim": "NONE",
                "re_audit_required": True,
            },
            "acceptance_gates": {
                "unexplained_overlap_zero": True,
                "exact_source_overlap_zero": True,
                "exact_candidate_overlap_zero": True,
                "reverse_edge_leakage_zero": True,
                "path_leakage_zero": True,
                "near_neighbor_leakage_zero": True,
                "final_endpoint_as_train_intermediate_zero": True,
                "required_axis_overlap_zero": True,
                "foundation_overlap_gate": True,
            },
            "counts": {
                "unexplained_overlap_count": 0,
                "exact_source_leakage_count": 0,
                "exact_candidate_leakage_count": 0,
                "reverse_edge_leakage_count": 0,
                "path_leakage_count": 0,
                "near_neighbor_leakage_count": 0,
                "final_endpoint_as_train_intermediate_count": 0,
                "required_axis_overlap_count": 0,
            },
        },
    )
elif name == "build_b0_evaluation_artifacts.py":
    root = Path(arg("--output-root"))
    root.mkdir(parents=True, exist_ok=False)
    tracks = root / "evaluation/tracks"
    tracks.mkdir(parents=True)
    output_paths = []
    for track in (
        "closed_measured_pool.yaml",
        "heldout_generative.yaml",
        "open_legal_generation.yaml",
    ):
        track_path = tracks / track
        track_path.write_text("status: READY\n", encoding="utf-8")
        output_paths.append(track_path)
    data_card = root / "docs/data/UTR_EditBench_v2_Data_Card.md"
    data_card.parent.mkdir(parents=True)
    data_card.write_text("# UTR EditBench v2 Data Card\n", encoding="utf-8")
    output_paths.append(data_card)
    role_matrix = root / "evaluation/tracks/track_role_matrix.yaml"
    role_matrix.write_text("status: READY\n", encoding="utf-8")
    output_paths.append(role_matrix)
    claims = root / "evaluation/claims/allowed_unsupported_claims.yaml"
    claims.parent.mkdir(parents=True)
    claims.write_text("allowed_claim: NONE\n", encoding="utf-8")
    output_paths.append(claims)
    artifact_bindings = root / "artifact_bindings.json"
    write_json_exclusive(
        str(artifact_bindings),
        {
            "schema_version": "utr_b0_artifact_bindings.v2",
            "artifacts": {
                "exposure_ledger": ref(arg("--exposure-ledger")),
                "track_role_matrix": ref(role_matrix),
                "data_card": ref(data_card),
                "claims": ref(claims),
            },
        },
    )
    output_paths.append(artifact_bindings)
    write_json_exclusive(
        str(root / "build_manifest.json"),
        {
            "schema_version": "utr_b0_evaluation_artifact_build.v2",
            "status": "PASS",
            "acceptance_preview": {
                "b0_gate_passed": True,
                "failed_gates": [],
            },
            "leakage_evidence_binding": {
                "supplied_reports_exactly_match_recomputation": True,
            },
            "track_a_label_seal_audit": {
                "gate_passed": True,
                "role_policy_exact_binding_passed": True,
                "current_d1_chain_binding_passed": True,
            },
            "required_artifact_audit": {"gate_passed": True},
            "d1_exposure_ledger_binding": {"gate_passed": True},
            "full_d1_binding": {"passed": True},
            "scientific_result_claimed": False,
            "foundation_status": "UNKNOWN_PENDING_FM0",
            "inputs": {
                "split_manifests": [
                    ref(path) for path in args_all("--split-manifest")
                ],
                "supplied_leakage_reports": [
                    ref(path) for path in args_all("--leakage-report")
                ],
            },
            "outputs": [ref(path) for path in output_paths],
        },
    )
elif name == "validate_b0_acceptance.py":
    binding_path = Path(arg("--artifact-bindings")).resolve(strict=True)
    bindings = json.loads(binding_path.read_text(encoding="utf-8"))
    accepted_artifacts = {}
    for artifact_name, reference in bindings["artifacts"].items():
        artifact_path = Path(reference["path"]).resolve(strict=True)
        accepted_artifacts[artifact_name] = {
            **ref(artifact_path),
            "exists": True,
            "schema_valid": True,
        }
    output = Path(arg("--output"))
    write_json_exclusive(
        str(output),
        {
            "schema_version": "utr_b0_acceptance.v2",
            "b0_gate_passed": True,
            "failed_gates": [],
            "allowed_claim": "NONE",
            "requires_fm0_reaudit": True,
            "re_audit_required_before_foundation_use": True,
            "supplied_leakage_reports_match_recomputation": True,
            "exposure_ledger": {"coverage": 1, "gate_passed": True},
            "track_role_audit": {"gate_passed": True},
            "track_a_label_seal_audit": {
                "gate_passed": True,
                "role_policy_exact_binding_passed": True,
                "current_d1_chain_binding_passed": True,
            },
            "required_artifact_audit": {
                "gate_passed": True,
                "binding_manifest_path": str(binding_path),
                "binding_manifest_sha256": hashlib.sha256(
                    binding_path.read_bytes()
                ).hexdigest(),
                "artifacts": accepted_artifacts,
            },
            "d1_exposure_ledger_binding": {"gate_passed": True},
            "supplied_leakage_report_files": [
                ref(path) for path in args_all("--leakage-report")
            ],
        },
    )
    if output.name == "acceptance.json":
        attempt_root = output.resolve().parents[1]
        gate_tamper = os.environ.get("B0_TEST_GATE_TAMPER")
        gate_dir = attempt_root / "provenance/gates"
        gate_evidence = gate_dir / "canonical_validation.json"
        if gate_tamper == "missing":
            gate_evidence.unlink()
        elif gate_tamper == "extra":
            write_json_exclusive(
                str(gate_dir / "unexpected_extra.json"),
                {"label": "unexpected_extra", "passed": True},
            )
        elif gate_tamper == "content":
            payload = json.loads(gate_evidence.read_text(encoding="utf-8"))
            payload["artifact"]["sha256"] = "f" * 64
            gate_evidence.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        nested_tamper = os.environ.get("B0_TEST_NESTED_TAMPER")
        if nested_tamper == "track":
            target = binding_path.parent / "evaluation/tracks/heldout_generative.yaml"
        elif nested_tamper == "data_card":
            target = binding_path.parent / "docs/data/UTR_EditBench_v2_Data_Card.md"
        elif nested_tamper == "artifact_bindings":
            target = binding_path
        elif nested_tamper == "split":
            target = Path(args_all("--split-manifest")[0])
        elif nested_tamper == "leakage":
            target = Path(args_all("--leakage-report")[0])
        else:
            target = None
        if target is not None:
            target.write_bytes(target.read_bytes() + b"\n")
else:
    raise SystemExit(f"unexpected fake B0 entry: {name}")
"""

FAKE_ACCEPTANCE_SEMANTICS = r"""
from __future__ import annotations


def validate_phase_acceptance(phase, payload, require_pass=False):
    if (
        phase == "D1"
        and payload.get("schema_version") == "d1_acceptance_v2"
        and payload.get("phase_gate_passed") is True
        and payload.get("fixture_mode") is False
        and payload.get("evidence_level") == "production_reconstruction"
        and payload.get("scientific_result_claimed") is False
        and isinstance(payload.get("dataset_results"), list)
        and payload.get("dataset_results")
        and payload.get("missing_required_datasets") == []
        and payload.get("missing_d1_scope_datasets") == []
        and payload.get("structural_validation_passed") is True
        and payload.get("required_artifact_validation", {}).get("passed") is True
        and payload.get("builder_audit_validation", {}).get("passed") is True
        and payload.get("global_store_validation", {}).get("passed") is True
        and payload.get("config_binding_validation", {}).get("passed") is True
        and payload.get("dataset_manifest_binding_validation", {}).get("passed")
        is True
    ):
        return []
    if (
        phase == "B0"
        and payload.get("schema_version") == "utr_b0_acceptance.v2"
        and payload.get("b0_gate_passed") is True
    ):
        return []
    return ["fixture semantic rejection"]
"""


def _production_fixture_repo(
    tmp_path: Path,
    *,
    seal_return_delay: bool = False,
    seal_index_delay: bool = False,
) -> Path:
    repo = tmp_path / "isolated_worktree"
    repo.mkdir()
    for relative in b0_driver_guard.CRITICAL_RELATIVE_PATHS:
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = SOURCE_ROOT / relative
        if relative in {
            "scripts/data/run_b0_production.sh",
            "scripts/execution/b0_driver_guard.py",
            "scripts/execution/run_audited_command.py",
        }:
            shutil.copy2(source, destination)
        elif relative == "scripts/execution/acceptance_semantics.py":
            destination.write_text(FAKE_ACCEPTANCE_SEMANTICS, encoding="utf-8")
        elif relative in {
            "scripts/data/build_b0_splits.py",
            "scripts/data/audit_b0_leakage.py",
            "scripts/data/build_b0_evaluation_artifacts.py",
            "scripts/data/validate_b0_acceptance.py",
        }:
            destination.write_text(FAKE_B0_SCRIPT, encoding="utf-8")
        elif relative.endswith(".json"):
            destination.write_text("{}\n", encoding="utf-8")
        else:
            destination.write_text("# frozen fixture entry\n", encoding="utf-8")
    (repo / DRIVER_RELATIVE).chmod(0o750)
    (repo / "scripts/execution/b0_driver_guard.py").chmod(0o750)
    if seal_return_delay:
        guard_path = repo / "scripts/execution/b0_driver_guard.py"
        guard_text = guard_path.read_text(encoding="utf-8")
        marker = (
            "        _ensure_terminal_success_event_logged(attempt_root)\n"
            "        _validate_terminal_success_state(attempt_root)\n"
            "        print(\n"
        )
        assert guard_text.count(marker) == 1
        guard_path.write_text(
            guard_text.replace(
                marker,
                (
                    "        _ensure_terminal_success_event_logged(attempt_root)\n"
                    "        _validate_terminal_success_state(attempt_root)\n"
                    "        time.sleep(2)\n"
                    "        print(\n"
                ),
            ),
            encoding="utf-8",
        )
    if seal_index_delay:
        guard_path = repo / "scripts/execution/b0_driver_guard.py"
        guard_text = guard_path.read_text(encoding="utf-8")
        marker = (
            "        _write_json_exclusive(index_path, _success_index(attempt_root))\n"
        )
        assert guard_text.count(marker) == 1
        guard_path.write_text(
            guard_text.replace(
                marker,
                marker + "        time.sleep(3)\n",
            ),
            encoding="utf-8",
        )
    (repo / "data/__init__.py").write_text(
        '"""Runtime import binding fixture."""\n',
        encoding="utf-8",
    )
    for relative, content in {
        "configs/terminal_noncritical.yaml": "status: frozen\n",
        "tests/terminal_noncritical.py": (
            '"""Frozen noncritical terminal fingerprint fixture."""\n'
        ),
    }.items():
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.name", "B0 Driver Test")
    _git(repo, "config", "user.email", "b0-driver@example.invalid")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "frozen B0 driver fixture")
    return repo


def _d1_fixture(
    tmp_path: Path,
    *,
    relative_build_ledger: bool,
) -> Path:
    d1_root = tmp_path / "d1_stage"
    data = d1_root / "data"
    data.mkdir(parents=True)
    canonical = data / "canonical.jsonl"
    structural = data / "structural.jsonl"
    ledger = data / "data_exposure_ledger.jsonl"
    canonical.write_text('{"record":"canonical"}\n', encoding="utf-8")
    structural.write_text('{"record":"structural"}\n', encoding="utf-8")
    ledger.write_text('{"dataset_id":"fixture"}\n', encoding="utf-8")
    canonical_ref = _ref(canonical)
    structural_ref = _ref(structural)
    ledger_ref = _ref(ledger)
    if relative_build_ledger:
        ledger_ref["path"] = "data/data_exposure_ledger.jsonl"
    build = {
        "global_stores": {
            "canonical_label_store": canonical_ref,
            "sealed_label_free_candidate_store": structural_ref,
        },
        "required_artifacts": {
            "data/data_exposure_ledger.jsonl": ledger_ref,
        },
    }
    build_path = d1_root / "build_manifest.json"
    acceptance = valid_d1_acceptance(d1_root)
    _write_json(build_path, build)
    acceptance["required_artifact_validation"]["build_manifest"] = _ref(build_path)
    acceptance["required_artifact_validation"]["artifacts"][
        "data/data_exposure_ledger.jsonl"
    ] = {
        **_ref(ledger),
        "exists": True,
    }
    acceptance_path = d1_root / "acceptance.json"
    _write_json(acceptance_path, acceptance)
    return acceptance_path


def _shallow_d1_fixture(tmp_path: Path) -> Path:
    acceptance = _d1_fixture(tmp_path, relative_build_ledger=False)
    payload = json.loads(acceptance.read_text(encoding="utf-8"))
    for key in (
        "dataset_results",
        "required_supported_datasets",
        "missing_required_datasets",
        "expected_d1_scope_datasets",
        "missing_d1_scope_datasets",
        "config_binding_validation",
        "dataset_manifest_binding_validation",
        "evidence_level",
        "scientific_result_claimed",
    ):
        payload.pop(key, None)
    _write_json(acceptance, payload)
    return acceptance


def _driver_command(
    repo: Path,
    acceptance: Path,
    approved_parent: Path,
    attempt_root: Path,
    python_launcher: Path,
    runtime_prefix: Path,
    expected_dirty_state_sha256: str | None = None,
) -> list[str]:
    driver = repo / DRIVER_RELATIVE
    runtime_manifest = _runtime_fixture_manifest(
        repo,
        attempt_root,
        python_launcher,
        runtime_prefix,
    )
    if expected_dirty_state_sha256 is None:
        expected_dirty_state_sha256 = b0_driver_guard._capture_git_snapshot(repo)[
            "dirty_state_sha256"
        ]
    return [
        str(driver),
        "--isolated-worktree",
        str(repo.resolve()),
        "--d1-acceptance",
        str(acceptance.resolve()),
        "--b0-attempt-root",
        str(attempt_root),
        "--python-bin",
        str(python_launcher),
        "--runtime-manifest",
        str(runtime_manifest),
        "--expected-commit",
        _git(repo, "rev-parse", "HEAD"),
        "--expected-driver-sha256",
        _sha256(driver),
        "--expected-dirty-state-sha256",
        expected_dirty_state_sha256,
        "--expected-runtime-prefix",
        str(runtime_prefix.resolve()),
        "--expected-runtime-manifest-sha256",
        _sha256(runtime_manifest),
        "--approved-b0-parent",
        str(approved_parent.resolve()),
        "--minimum-free-bytes",
        "1",
    ]


def _propagate_current_runtime_paths_to_nested_venv(
    launcher: Path,
    virtualenv: Path,
    fixture_repo: Path,
) -> Path:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            str(launcher),
            "-c",
            (
                "import json, sysconfig; "
                "print(json.dumps(sysconfig.get_path('purelib')))"
            ),
        ],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    site_packages = Path(json.loads(completed.stdout)).resolve(strict=True)
    virtualenv_root = virtualenv.resolve(strict=True)
    if not site_packages.is_relative_to(virtualenv_root):
        raise AssertionError(f"nested venv purelib escaped its root: {site_packages}")

    fixture_root = fixture_repo.resolve(strict=True)
    parent_source_root = SOURCE_ROOT.resolve(strict=True)
    runtime_paths = [fixture_root]
    seen = {fixture_root}
    for raw_path in sys.path:
        if not isinstance(raw_path, str) or not raw_path:
            continue
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            continue
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved == parent_source_root:
            continue
        if not resolved.is_dir() or resolved in seen:
            continue
        if "\n" in str(resolved) or "\r" in str(resolved):
            raise AssertionError(f"unsafe runtime path for .pth: {resolved!s}")
        seen.add(resolved)
        runtime_paths.append(resolved)
    if not runtime_paths:
        raise AssertionError("current test runtime exposes no absolute import paths")

    bridge = site_packages / "_b0_parent_runtime_paths.pth"
    bridge.write_text(
        "".join(f"{path}\n" for path in runtime_paths),
        encoding="utf-8",
    )
    return bridge


def _runtime_fixture_manifest(
    repo: Path,
    attempt_root: Path,
    python_launcher: Path,
    runtime_prefix: Path,
) -> Path:
    probe = r"""
import importlib
import importlib.metadata
import json
import pathlib
import platform
import sys

packages = {}
for distribution, module_name in (
    ("jsonschema", "jsonschema"),
    ("PyYAML", "yaml"),
    ("numpy", "numpy"),
):
    module = importlib.import_module(module_name)
    packages[distribution] = {
        "distribution": distribution,
        "version": importlib.metadata.version(distribution),
        "module": module_name,
        "module_file": str(pathlib.Path(module.__file__).resolve(strict=True)),
    }
data_module = importlib.import_module("data")
print(json.dumps({
    "python_bin": sys.executable,
    "python_version": platform.python_version(),
    "implementation": platform.python_implementation(),
    "runtime_prefix": str(pathlib.Path(sys.prefix).resolve(strict=True)),
    "base_prefix": str(pathlib.Path(sys.base_prefix).resolve(strict=True)),
    "packages": packages,
    "project_imports": {
        "data": {
            "module": "data",
            "module_file": str(
                pathlib.Path(data_module.__file__).resolve(strict=True)
            ),
        },
    },
}, sort_keys=True))
"""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [str(python_launcher), "-c", probe],
        cwd=repo,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    observed = json.loads(completed.stdout)
    overlay = attempt_root.parent / f".{attempt_root.name}.runtime-overlay.pth"
    overlay.write_text(str(repo.resolve()) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "b0-runtime-manifest-v1",
        "created_at_utc": "2026-07-29T00:00:00Z",
        "runtime_prefix": str(runtime_prefix.resolve()),
        "python_bin": str(python_launcher),
        "python_version": observed["python_version"],
        "implementation": "CPython",
        "base_prefix": observed["base_prefix"],
        "project_root": str(repo.resolve()),
        "task_resource": "CPU_HEAVY",
        "non_neural": True,
        "cuda_required": False,
        "packages": observed["packages"],
        "project_imports": observed["project_imports"],
        "overlay_file": {
            "path": str(overlay.resolve()),
            "sha256": _sha256(overlay),
        },
    }
    manifest_path = attempt_root.parent / f".{attempt_root.name}.runtime-manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def _driver_environment(tmp_path: Path) -> dict[str, str]:
    fake_bin = tmp_path / "test-bin"
    fake_bin.mkdir(exist_ok=True)
    fake_ps = fake_bin / "ps"
    fake_ps.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_ps.chmod(0o750)
    environment = dict(os.environ)
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    return environment


def test_driver_requires_all_approved_values_before_attempt_creation(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "must_not_exist"
    completed = subprocess.run(
        [
            "bash",
            str(SOURCE_ROOT / DRIVER_RELATIVE),
            "--b0-attempt-root",
            str(attempt),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 64
    assert not attempt.exists()
    assert "missing required option" in completed.stderr


def test_driver_watchdog_wires_exact_shell_parent_pid() -> None:
    driver_source = (SOURCE_ROOT / DRIVER_RELATIVE).read_text(encoding="utf-8")
    expected_invocation = """\
"$PYTHON_BIN" "$GUARD" watchdog \\
  --attempt-root "$B0_ATTEMPT_ROOT" \\
  --d1-acceptance "$D1_ACCEPTANCE" \\
  --parent-pid "$$" &
"""

    assert driver_source.count(expected_invocation) == 1
    assert driver_source.count('--parent-pid "$$"') == 1


def test_relative_d1_ledger_fails_in_audited_node00_and_blocks_node01(
    tmp_path: Path,
) -> None:
    repo = _production_fixture_repo(tmp_path)
    acceptance = _d1_fixture(tmp_path, relative_build_ledger=True)
    approved_parent = tmp_path / "approved_b0_parent"
    approved_parent.mkdir()
    attempt = approved_parent / "relative-ledger-attempt"
    command = _driver_command(
        repo,
        acceptance,
        approved_parent,
        attempt,
        Path(sys.executable),
        Path(sys.prefix),
    )

    completed = subprocess.run(
        command,
        env=_driver_environment(tmp_path),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=DRIVER_TEST_TIMEOUT_SECONDS,
    )

    assert completed.returncode != 0
    assert (attempt / "audit/00_preflight/completion.json").is_file()
    assert not (attempt / "audit/01_canonical_validation").exists()
    audit_completion = json.loads(
        (attempt / "audit/00_preflight/completion.json").read_text(encoding="utf-8")
    )
    assert audit_completion["state"] == "FAILED_WITH_EVIDENCE"
    assert audit_completion["observed_process_exit_code"] == 74
    assert not (attempt / "artifacts/preflight.json").exists()
    failure = json.loads((attempt / "failure/failure.json").read_text(encoding="utf-8"))
    assert failure["reason"] == "AUDITED_NODE_FAILED_00_preflight"
    status = json.loads((attempt / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "FAILED_WITH_EVIDENCE"
    assert status["terminal"] is True
    events = [
        json.loads(line)
        for line in (attempt / "logs/events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    terminal = [
        event
        for event in events
        if event["event"] in {"FAILED_WITH_EVIDENCE", "SAFE_PAUSED"}
    ]
    assert len(terminal) == 1


def test_shallow_d1_document_is_rejected_by_full_semantic_preflight(
    tmp_path: Path,
) -> None:
    repo = _production_fixture_repo(tmp_path)
    acceptance = _shallow_d1_fixture(tmp_path)
    approved_parent = tmp_path / "approved_b0_parent"
    approved_parent.mkdir()
    attempt = approved_parent / "shallow-d1-attempt"

    completed = subprocess.run(
        _driver_command(
            repo,
            acceptance,
            approved_parent,
            attempt,
            Path(sys.executable),
            Path(sys.prefix),
        ),
        env=_driver_environment(tmp_path),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=DRIVER_TEST_TIMEOUT_SECONDS,
    )

    assert completed.returncode != 0
    assert (attempt / "audit/00_preflight/completion.json").is_file()
    assert not (attempt / "audit/01_canonical_validation").exists()
    assert not (attempt / "driver_completion.json").exists()
    failure = json.loads((attempt / "failure/failure.json").read_text(encoding="utf-8"))
    assert failure["reason"] == "AUDITED_NODE_FAILED_00_preflight"


def test_production_d1_semantics_reject_the_shallow_document(
    tmp_path: Path,
) -> None:
    acceptance = _shallow_d1_fixture(tmp_path)
    payload = json.loads(acceptance.read_text(encoding="utf-8"))

    with pytest.raises(
        b0_driver_guard.GuardError,
        match="D1 acceptance semantic validation failed",
    ):
        b0_driver_guard._validate_d1_acceptance_payload(payload)


@pytest.mark.parametrize("coverage", (True, False, "1", None))
def test_b0_hard_gate_rejects_non_numeric_or_boolean_coverage(
    coverage: object,
) -> None:
    payload = {
        "schema_version": "utr_b0_acceptance.v2",
        "b0_gate_passed": True,
        "failed_gates": [],
        "allowed_claim": "NONE",
        "requires_fm0_reaudit": True,
        "re_audit_required_before_foundation_use": True,
        "supplied_leakage_reports_match_recomputation": True,
        "exposure_ledger": {"coverage": coverage, "gate_passed": True},
        "track_role_audit": {"gate_passed": True},
        "track_a_label_seal_audit": {
            "gate_passed": True,
            "role_policy_exact_binding_passed": True,
            "current_d1_chain_binding_passed": True,
        },
        "required_artifact_audit": {"gate_passed": True},
        "d1_exposure_ledger_binding": {"gate_passed": True},
    }

    with pytest.raises(b0_driver_guard.GuardError, match="strict numeric"):
        b0_driver_guard._validate_b0_acceptance_hard_gate(payload)


@pytest.mark.parametrize("partitions_sha256", ("A" * 64, "a" * 63, True))
def test_split_gate_rejects_noncanonical_partition_sha256(
    partitions_sha256: object,
) -> None:
    payload = {
        "status": "READY",
        "d1_phase_gate_passed": True,
        "d1_acceptance_path": "/approved/d1.json",
        "canonical_validation_report_path": "/approved/canonical.json",
        "partitions": [{"status": "READY"}],
        "partitions_sha256": partitions_sha256,
    }

    with pytest.raises(b0_driver_guard.GuardError):
        b0_driver_guard._validate_named_gate(
            "split-common",
            payload,
            expected_head=None,
            expected_dirty_state_sha256=None,
            expected_d1_acceptance="/approved/d1.json",
            expected_canonical_validation="/approved/canonical.json",
        )


@pytest.mark.parametrize("tamper", ("missing", "extra", "content"))
def test_seal_rejects_missing_extra_or_tampered_named_gate_evidence(
    tmp_path: Path,
    tamper: str,
) -> None:
    repo = _production_fixture_repo(tmp_path)
    acceptance = _d1_fixture(tmp_path, relative_build_ledger=False)
    approved_parent = tmp_path / "approved_b0_parent"
    approved_parent.mkdir()
    attempt = approved_parent / f"gate-{tamper}-attempt"
    environment = _driver_environment(tmp_path)
    environment["B0_TEST_GATE_TAMPER"] = tamper

    completed = subprocess.run(
        _driver_command(
            repo,
            acceptance,
            approved_parent,
            attempt,
            Path(sys.executable),
            Path(sys.prefix),
        ),
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=DRIVER_TEST_TIMEOUT_SECONDS,
    )

    assert completed.returncode != 0
    assert not (attempt / "driver_completion.json").exists()
    assert not (attempt / "artifact_checksums.json").exists()
    assert (attempt / "failure/failure.json").is_file()
    status = json.loads((attempt / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "FAILED_WITH_EVIDENCE"
    assert status["terminal"] is True


@pytest.mark.parametrize(
    "tamper",
    ("track", "data_card", "artifact_bindings", "split", "leakage"),
)
def test_seal_rejects_post_node13_nested_artifact_tampering(
    tmp_path: Path,
    tamper: str,
) -> None:
    repo = _production_fixture_repo(tmp_path)
    acceptance = _d1_fixture(tmp_path, relative_build_ledger=False)
    approved_parent = tmp_path / "approved_b0_parent"
    approved_parent.mkdir()
    attempt = approved_parent / f"nested-{tamper}-attempt"
    environment = _driver_environment(tmp_path)
    environment["B0_TEST_NESTED_TAMPER"] = tamper

    completed = subprocess.run(
        _driver_command(
            repo,
            acceptance,
            approved_parent,
            attempt,
            Path(sys.executable),
            Path(sys.prefix),
        ),
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=DRIVER_TEST_TIMEOUT_SECONDS,
    )

    assert completed.returncode != 0
    assert (attempt / "audit/13_final_acceptance/completion.json").is_file()
    assert not (attempt / "driver_completion.json").exists()
    assert not (attempt / "artifact_checksums.json").exists()
    assert (attempt / "failure/failure.json").is_file()


def test_seal_rejects_nested_artifact_tampering_after_checksum_index(
    tmp_path: Path,
) -> None:
    repo = _production_fixture_repo(tmp_path, seal_index_delay=True)
    acceptance = _d1_fixture(tmp_path, relative_build_ledger=False)
    approved_parent = tmp_path / "approved_b0_parent"
    approved_parent.mkdir()
    attempt = approved_parent / "checksum-window-tamper-attempt"
    process = subprocess.Popen(
        _driver_command(
            repo,
            acceptance,
            approved_parent,
            attempt,
            Path(sys.executable),
            Path(sys.prefix),
        ),
        env=_driver_environment(tmp_path),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    checksum_index_path = attempt / "artifact_checksums.json"
    deadline = time.monotonic() + DRIVER_TEST_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if checksum_index_path.is_file() or process.poll() is not None:
            break
        time.sleep(0.02)
    if not checksum_index_path.is_file():
        stdout, stderr = process.communicate(
            timeout=DRIVER_TEST_TIMEOUT_SECONDS,
        )
        pytest.fail(f"driver did not reach checksum seal window: {stdout!r} {stderr!r}")

    nested_artifact = (
        attempt / "artifacts/bundle/evaluation/tracks/heldout_generative.yaml"
    )
    indexed_paths = {
        entry["path"]
        for entry in json.loads(checksum_index_path.read_text(encoding="utf-8"))[
            "entries"
        ]
    }
    assert nested_artifact.relative_to(attempt).as_posix() in indexed_paths
    nested_artifact.write_bytes(
        nested_artifact.read_bytes() + b"# seal-window tamper\n"
    )
    stdout, stderr = process.communicate(
        timeout=DRIVER_TEST_TIMEOUT_SECONDS,
    )

    assert process.returncode != 0, stdout
    assert "live artifact inventory or checksums differ" in stderr
    status = json.loads((attempt / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "FAILED_WITH_EVIDENCE"
    assert status["terminal"] is True
    assert (attempt / "failure/failure.json").is_file()
    events = [
        json.loads(line)
        for line in (attempt / "logs/events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    terminal_events = [
        event for event in events if event["event"] in b0_driver_guard.TERMINAL_EVENTS
    ]
    assert [event["event"] for event in terminal_events] == ["FAILED_WITH_EVIDENCE"]


def test_success_preserves_venv_launcher_runs_00_to_13_and_seals_driver(
    tmp_path: Path,
) -> None:
    repo = _production_fixture_repo(tmp_path)
    acceptance = _d1_fixture(tmp_path, relative_build_ledger=False)
    approved_parent = tmp_path / "approved B0 outputs"
    approved_parent.mkdir()
    attempt = approved_parent / "b0 attempt unicode-\u6d4b\u8bd5"
    virtualenv = tmp_path / "nested runtime \u6d4b\u8bd5"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "venv",
            "--without-pip",
            str(virtualenv),
        ],
        check=True,
    )
    launcher = virtualenv / "bin/python"
    assert launcher.is_symlink()
    assert " " in str(launcher)
    assert any(ord(character) > 127 for character in str(launcher))
    runtime_bridge = _propagate_current_runtime_paths_to_nested_venv(
        launcher,
        virtualenv,
        repo,
    )
    assert runtime_bridge.is_file()
    bridge_lines = runtime_bridge.read_text(encoding="utf-8").splitlines()
    assert all(Path(line).is_absolute() for line in bridge_lines)
    assert Path(bridge_lines[0]).resolve(strict=True) == repo.resolve(strict=True)
    assert SOURCE_ROOT.resolve(strict=True) not in {
        Path(line).resolve(strict=True) for line in bridge_lines
    }
    command = _driver_command(
        repo,
        acceptance,
        approved_parent,
        attempt,
        launcher,
        virtualenv,
    )
    runtime_manifest = json.loads(
        Path(command[command.index("--runtime-manifest") + 1]).read_text(
            encoding="utf-8"
        )
    )
    assert set(runtime_manifest["packages"]) == {"jsonschema", "PyYAML", "numpy"}
    assert (
        Path(runtime_manifest["project_imports"]["data"]["module_file"])
        .resolve(strict=True)
        .is_relative_to(repo.resolve(strict=True))
    )

    completed = subprocess.run(
        command,
        env=_driver_environment(tmp_path),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=DRIVER_TEST_TIMEOUT_SECONDS,
    )

    assert completed.returncode == 0, completed.stderr
    stdout_lines = [
        json.loads(line) for line in completed.stdout.splitlines() if line.strip()
    ]
    assert len(stdout_lines) == 1
    assert stdout_lines[0]["event"] == "B0_DRIVER_COMPLETED"
    expected_nodes = list(b0_driver_guard.EXPECTED_AUDIT_NODES)
    assert sorted(path.name for path in (attempt / "audit").iterdir()) == expected_nodes
    status = json.loads((attempt / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "B0_DRIVER_COMPLETED"
    assert status["terminal"] is True
    preflight = json.loads(
        (attempt / "artifacts/preflight.json").read_text(encoding="utf-8")
    )
    assert preflight["runtime"][
        "launcher_path_preserved_without_final_symlink_resolution"
    ] == str(launcher)
    assert Path(preflight["runtime"]["sys_prefix"]).resolve() == virtualenv.resolve()
    completion = json.loads(
        (attempt / "driver_completion.json").read_text(encoding="utf-8")
    )
    assert completion["audit_node_order"] == expected_nodes
    assert completion["sealable_for_post_acceptance_release"] is True
    assert completion["stage_completion_claimed"] is False
    assert completion["scientific_result_claimed"] is False
    assert not (attempt / "artifacts/stages").exists()
    checksum_index = json.loads(
        (attempt / "artifact_checksums.json").read_text(encoding="utf-8")
    )
    for entry in checksum_index["entries"]:
        artifact = attempt / entry["path"]
        assert artifact.stat().st_size == entry["bytes"]
        assert _sha256(artifact) == entry["sha256"]
    events = [
        json.loads(line)
        for line in (attempt / "logs/events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [event["event"] for event in events].count("B0_DRIVER_SEAL_READY") == 1
    assert [event["event"] for event in events].count("B0_DRIVER_COMPLETED") == 1
    terminal_event = json.loads(
        (attempt / "provenance/terminal_success_event.json").read_text(encoding="utf-8")
    )
    assert (
        terminal_event
        == [event for event in events if event["event"] == "B0_DRIVER_COMPLETED"][0]
    )
    assert completion["terminal_event"] == _ref(
        attempt / "provenance/terminal_success_event.json"
    ) | {"path": "provenance/terminal_success_event.json"}
    assert not (attempt / "failure/failure.json").exists()
    metrics = [
        json.loads(line)
        for line in (attempt / "logs/system_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert metrics
    assert all(sample["interval_seconds"] == 300 for sample in metrics)
    assert all(sample["scientific_logs_read"] is False for sample in metrics)

    events_before_delayed_failure = (attempt / "logs/events.jsonl").read_bytes()
    status_before_delayed_failure = (attempt / "status.json").read_bytes()
    delayed_failure = subprocess.run(
        [
            str(launcher),
            str(repo / "scripts/execution/b0_driver_guard.py"),
            "failure",
            "--attempt-root",
            str(attempt),
            "--exit-code",
            "143",
            "--reason",
            "SIGNAL_TERM_AFTER_SEAL",
            "--signal",
            "TERM",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert delayed_failure.returncode == 0, delayed_failure.stderr
    assert (attempt / "logs/events.jsonl").read_bytes() == events_before_delayed_failure
    assert (attempt / "status.json").read_bytes() == status_before_delayed_failure
    assert not (attempt / "failure/failure.json").exists()

    terminal_success_command = [
        str(launcher),
        str(repo / "scripts/execution/b0_driver_guard.py"),
        "terminal-success",
        "--attempt-root",
        str(attempt),
    ]
    checksum_index_path = attempt / "artifact_checksums.json"
    checksum_index_bytes = checksum_index_path.read_bytes()
    checksum_index_payload = json.loads(checksum_index_bytes)
    malformed_indexes = []
    extra_field_index = json.loads(checksum_index_bytes)
    extra_field_index["unexpected"] = True
    malformed_indexes.append(extra_field_index)
    wrong_exclusions_index = json.loads(checksum_index_bytes)
    wrong_exclusions_index["excluded_mutable_or_self_referential_paths"].append(
        "artifacts/bundle/evaluation/tracks/heldout_generative.yaml"
    )
    malformed_indexes.append(wrong_exclusions_index)
    wrong_entries_sha_index = json.loads(checksum_index_bytes)
    wrong_entries_sha_index["entries_sha256"] = "0" * 64
    malformed_indexes.append(wrong_entries_sha_index)
    extra_entry_field_index = json.loads(checksum_index_bytes)
    extra_entry_field_index["entries"][0]["unexpected"] = True
    malformed_indexes.append(extra_entry_field_index)
    for malformed_index in malformed_indexes:
        _write_json(checksum_index_path, malformed_index)
        rejected_index = subprocess.run(
            terminal_success_command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert rejected_index.returncode == 74
        checksum_index_path.write_bytes(checksum_index_bytes)

    healthy_terminal_check = subprocess.run(
        terminal_success_command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert healthy_terminal_check.returncode == 0, healthy_terminal_check.stderr
    assert (
        json.loads(healthy_terminal_check.stdout)["artifact_checksum_index"][
            "entries_sha256"
        ]
        == checksum_index_payload["entries_sha256"]
    )

    completion_path = attempt / "driver_completion.json"
    status_path = attempt / "status.json"
    terminal_event_path = attempt / "provenance/terminal_success_event.json"
    events_path = attempt / "logs/events.jsonl"
    events_snapshot_path = attempt / "provenance/events_at_terminal.jsonl"
    original_terminal_documents = {
        path: path.read_bytes()
        for path in (
            completion_path,
            status_path,
            terminal_event_path,
            events_path,
            events_snapshot_path,
            checksum_index_path,
        )
    }

    def restore_terminal_documents() -> None:
        for path, payload in original_terminal_documents.items():
            path.write_bytes(payload)

    def assert_terminal_rejected(expected_error: str | None = None) -> None:
        rejected = subprocess.run(
            terminal_success_command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert rejected.returncode == 74, rejected.stdout
        if expected_error is not None:
            assert expected_error in rejected.stderr

    for mutation in (
        "claims-true",
        "completion-extra-field",
        "completion-missing-field",
        "status-extra-field",
        "status-missing-field",
        "audit-order",
        "attempt-root",
        "authority",
        "wrong-ref-path",
        "wrong-ref-sha",
        "embedded-acceptance-binding",
        "embedded-bundle-binding",
    ):
        mutated_completion = json.loads(original_terminal_documents[completion_path])
        mutated_status = json.loads(original_terminal_documents[status_path])
        if mutation == "claims-true":
            mutated_completion["stage_completion_claimed"] = True
            mutated_completion["scientific_result_claimed"] = True
        elif mutation == "completion-extra-field":
            mutated_completion["unexpected"] = False
        elif mutation == "completion-missing-field":
            mutated_completion.pop("post_acceptance_git_release_chain_required")
        elif mutation == "status-extra-field":
            mutated_status["unexpected"] = False
        elif mutation == "status-missing-field":
            mutated_status.pop("wrapper_pid")
        elif mutation == "audit-order":
            mutated_completion["audit_node_order"] = list(
                reversed(mutated_completion["audit_node_order"])
            )
        elif mutation == "attempt-root":
            mutated_completion["attempt_root"] = str(attempt.parent)
        elif mutation == "authority":
            mutated_completion[
                "authoritative_only_when_terminal_status_ref_matches"
            ] = False
            mutated_completion["sealable_for_post_acceptance_release"] = False
        elif mutation == "wrong-ref-path":
            mutated_completion["code_manifest"] = dict(
                mutated_completion["attempt_manifest"]
            )
        elif mutation == "wrong-ref-sha":
            mutated_completion["code_manifest"]["sha256"] = "0" * 64
        elif mutation == "embedded-acceptance-binding":
            mutated_completion["accepted_result_binding"]["semantic_validation"][
                "passed"
            ] = False
        elif mutation == "embedded-bundle-binding":
            mutated_completion["bundle_result_binding"]["semantic_validation"][
                "passed"
            ] = False
        else:
            raise AssertionError(f"unknown terminal mutation: {mutation}")

        if mutation not in {"status-extra-field", "status-missing-field"}:
            _write_json(completion_path, mutated_completion)
            mutated_status["driver_completion"] = _attempt_ref(
                completion_path,
                attempt,
            )
        _write_json(status_path, mutated_status)
        assert_terminal_rejected()
        restore_terminal_documents()

    original_events = [
        json.loads(line)
        for line in original_terminal_documents[events_path]
        .decode("utf-8")
        .splitlines()
    ]

    def rewrite_terminal_chain(
        mutated_events: list[dict[str, object]],
        mutated_terminal_event: dict[str, object],
    ) -> None:
        _write_json(terminal_event_path, mutated_terminal_event)
        rewritten_events = b"".join(
            (
                json.dumps(
                    event,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            for event in mutated_events
        )
        events_path.write_bytes(rewritten_events)
        events_snapshot_path.write_bytes(rewritten_events)
        _write_json(
            checksum_index_path,
            b0_driver_guard._success_index(attempt),
        )
        mutated_completion = json.loads(original_terminal_documents[completion_path])
        mutated_completion["terminal_event"] = _attempt_ref(
            terminal_event_path,
            attempt,
        )
        mutated_completion["events_snapshot"] = _attempt_ref(
            events_snapshot_path,
            attempt,
        )
        mutated_completion["artifact_checksum_index"] = _attempt_ref(
            checksum_index_path,
            attempt,
        )
        _write_json(completion_path, mutated_completion)
        mutated_status = json.loads(original_terminal_documents[status_path])
        mutated_status["terminal_event"] = _attempt_ref(
            terminal_event_path,
            attempt,
        )
        mutated_status["driver_completion"] = _attempt_ref(
            completion_path,
            attempt,
        )
        _write_json(status_path, mutated_status)

    for mutation in (
        "terminal-event-extra-field",
        "terminal-event-missing-field",
        "terminal-event-claim",
    ):
        mutated_terminal_event = json.loads(
            original_terminal_documents[terminal_event_path]
        )
        if mutation == "terminal-event-extra-field":
            mutated_terminal_event["unexpected"] = False
        elif mutation == "terminal-event-missing-field":
            mutated_terminal_event.pop("scientific_result_claimed")
        elif mutation == "terminal-event-claim":
            mutated_terminal_event["stage_completion_claimed"] = True
        else:
            raise AssertionError(f"unknown terminal event mutation: {mutation}")
        _write_json(terminal_event_path, mutated_terminal_event)
        mutated_events = [
            (
                mutated_terminal_event
                if event.get("event") == "B0_DRIVER_COMPLETED"
                else event
            )
            for event in original_events
        ]
        rewrite_terminal_chain(mutated_events, mutated_terminal_event)
        assert_terminal_rejected()
        restore_terminal_documents()

    original_terminal_event = json.loads(
        original_terminal_documents[terminal_event_path]
    )
    for mutation in (
        "seal-ready-extra-field",
        "seal-ready-stage-true",
        "seal-ready-stage-integer",
        "nonterminal-stage-true",
        "nonterminal-scientific-string",
        "duplicate-seal-ready",
        "late-nonterminal-event",
    ):
        mutated_events = copy.deepcopy(original_events)
        seal_index = next(
            index
            for index, event in enumerate(mutated_events)
            if event["event"] == "B0_DRIVER_SEAL_READY"
        )
        first_nonterminal_index = next(
            index
            for index, event in enumerate(mutated_events)
            if event["event"] not in b0_driver_guard.TERMINAL_EVENTS
            and event["event"] != "B0_DRIVER_SEAL_READY"
        )
        if mutation == "seal-ready-extra-field":
            mutated_events[seal_index]["unexpected"] = False
        elif mutation == "seal-ready-stage-true":
            mutated_events[seal_index]["stage_completion_claimed"] = True
        elif mutation == "seal-ready-stage-integer":
            mutated_events[seal_index]["stage_completion_claimed"] = 0
        elif mutation == "nonterminal-stage-true":
            mutated_events[first_nonterminal_index]["stage_completion_claimed"] = True
        elif mutation == "nonterminal-scientific-string":
            mutated_events[first_nonterminal_index][
                "scientific_result_claimed"
            ] = "false"
        elif mutation == "duplicate-seal-ready":
            mutated_events.insert(seal_index, copy.deepcopy(mutated_events[seal_index]))
        elif mutation == "late-nonterminal-event":
            mutated_events.insert(
                -1,
                {
                    "schema_version": "b0_driver_event.v1",
                    "at_utc": mutated_events[seal_index]["at_utc"],
                    "event": "POST_SEAL_NONTERMINAL",
                },
            )
        else:
            raise AssertionError(f"unknown event-chain mutation: {mutation}")
        rewrite_terminal_chain(mutated_events, original_terminal_event)
        assert_terminal_rejected()
        restore_terminal_documents()

    restored_terminal_check = subprocess.run(
        terminal_success_command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert restored_terminal_check.returncode == 0, restored_terminal_check.stderr

    for relative in (
        "data/__init__.py",
        "configs/terminal_noncritical.yaml",
        "tests/terminal_noncritical.py",
    ):
        tracked_noncritical = repo / relative
        original_bytes = tracked_noncritical.read_bytes()
        tracked_noncritical.write_bytes(original_bytes + b"# post-success drift\n")
        assert_terminal_rejected(
            "complete frozen manifest",
        )
        tracked_noncritical.write_bytes(original_bytes)

    restored_after_live_code_drift = subprocess.run(
        terminal_success_command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert (
        restored_after_live_code_drift.returncode == 0
    ), restored_after_live_code_drift.stderr

    code_manifest_path = attempt / "provenance/code_manifest.json"
    code_manifest_sidecar_path = attempt / "provenance/code_manifest.sha256"
    fingerprint_paths = sorted((attempt / "provenance/fingerprints").glob("*.json"))
    original_code_documents = {
        path: path.read_bytes()
        for path in (
            code_manifest_path,
            code_manifest_sidecar_path,
            *fingerprint_paths,
        )
    }

    def restore_code_documents() -> None:
        for path, payload in original_code_documents.items():
            path.write_bytes(payload)

    def mutate_fingerprint_type(
        payload: dict[str, object],
        mutation: str,
    ) -> None:
        if mutation == "tracked-diff-bytes-bool":
            payload["tracked_diff"]["bytes"] = False
        elif mutation == "index-entry-count-float":
            payload["git_index_flags"]["entry_count"] = 0.0
        elif mutation == "index-raw-sha-bool":
            payload["git_index_flags"]["raw_sha256"] = True
        else:
            raise AssertionError(f"unknown fingerprint mutation: {mutation}")

    for mutation in (
        "tracked-diff-bytes-bool",
        "index-entry-count-float",
        "index-raw-sha-bool",
    ):
        mutated_manifest = json.loads(original_code_documents[code_manifest_path])
        mutate_fingerprint_type(mutated_manifest, mutation)
        manifest_core = {
            key: value
            for key, value in mutated_manifest.items()
            if key not in {"fingerprint_sha256", "caller_approval"}
        }
        mutated_manifest["fingerprint_sha256"] = b0_driver_guard._canonical_json_sha(
            manifest_core
        )
        _write_json(code_manifest_path, mutated_manifest)
        mutated_manifest_sha256 = _sha256(code_manifest_path)
        code_manifest_sidecar_path.write_text(
            f"{mutated_manifest_sha256}  {code_manifest_path.name}\n",
            encoding="ascii",
        )
        for fingerprint_path in fingerprint_paths:
            mutated_fingerprint = json.loads(original_code_documents[fingerprint_path])
            mutate_fingerprint_type(mutated_fingerprint, mutation)
            fingerprint_core = {
                key: value
                for key, value in mutated_fingerprint.items()
                if key not in {"fingerprint_sha256", "comparison"}
            }
            mutated_fingerprint["fingerprint_sha256"] = (
                b0_driver_guard._canonical_json_sha(fingerprint_core)
            )
            mutated_fingerprint["comparison"]["baseline"] = _ref(code_manifest_path)
            mutated_fingerprint["comparison"][
                "expected_baseline_sha256"
            ] = mutated_manifest_sha256
            _write_json(fingerprint_path, mutated_fingerprint)

        _write_json(
            checksum_index_path,
            b0_driver_guard._success_index(attempt),
        )
        mutated_completion = json.loads(original_terminal_documents[completion_path])
        mutated_completion["code_manifest"] = _attempt_ref(
            code_manifest_path,
            attempt,
        )
        mutated_completion["artifact_checksum_index"] = _attempt_ref(
            checksum_index_path,
            attempt,
        )
        _write_json(completion_path, mutated_completion)
        mutated_status = json.loads(original_terminal_documents[status_path])
        mutated_status["driver_completion"] = _attempt_ref(
            completion_path,
            attempt,
        )
        _write_json(status_path, mutated_status)
        assert_terminal_rejected(
            "complete frozen manifest",
        )
        restore_code_documents()
        restore_terminal_documents()

    restored_after_coherent_code_rewrite = subprocess.run(
        terminal_success_command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert (
        restored_after_coherent_code_rewrite.returncode == 0
    ), restored_after_coherent_code_rewrite.stderr

    nested_artifact = (
        attempt / "artifacts/bundle/evaluation/tracks/heldout_generative.yaml"
    )
    nested_artifact.write_bytes(nested_artifact.read_bytes() + b"# post-seal tamper\n")
    post_seal_tamper = subprocess.run(
        terminal_success_command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert post_seal_tamper.returncode == 74
    assert "live artifact inventory or checksums differ" in post_seal_tamper.stderr

    delayed_failure_after_tamper = subprocess.run(
        [
            str(launcher),
            str(repo / "scripts/execution/b0_driver_guard.py"),
            "failure",
            "--attempt-root",
            str(attempt),
            "--exit-code",
            "143",
            "--reason",
            "SIGNAL_TERM_AFTER_POST_SEAL_TAMPER",
            "--signal",
            "TERM",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert delayed_failure_after_tamper.returncode == 74
    assert (attempt / "logs/events.jsonl").read_bytes() == events_before_delayed_failure
    assert (attempt / "status.json").read_bytes() == status_before_delayed_failure
    assert not (attempt / "failure/failure.json").exists()


def test_tracked_code_drift_after_node_blocks_next_node_and_completion(
    tmp_path: Path,
) -> None:
    repo = _production_fixture_repo(tmp_path)
    acceptance = _d1_fixture(tmp_path, relative_build_ledger=False)
    approved_parent = tmp_path / "approved_b0_parent"
    approved_parent.mkdir()
    attempt = approved_parent / "tracked-drift-attempt"
    command = _driver_command(
        repo,
        acceptance,
        approved_parent,
        attempt,
        Path(sys.executable),
        Path(sys.prefix),
    )
    environment = _driver_environment(tmp_path)
    environment["B0_TEST_MUTATE_TRACKED_CODE"] = "1"

    completed = subprocess.run(
        command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=DRIVER_TEST_TIMEOUT_SECONDS,
    )

    assert completed.returncode == 74
    assert (attempt / "audit/01_canonical_validation/completion.json").is_file()
    assert not (attempt / "audit/02_split_5utr_source").exists()
    assert not (attempt / "driver_completion.json").exists()
    drift = json.loads(
        (
            attempt / "provenance/fingerprints/01_canonical_validation.after.json"
        ).read_text(encoding="utf-8")
    )
    assert drift["comparison"]["matches"] is False
    failure = json.loads((attempt / "failure/failure.json").read_text(encoding="utf-8"))
    assert failure["reason"] == "POST_NODE_FREEZE_DRIFT_01_canonical_validation"


def test_runtime_manifest_hash_tamper_fails_in_audited_node00(
    tmp_path: Path,
) -> None:
    repo = _production_fixture_repo(tmp_path)
    acceptance = _d1_fixture(tmp_path, relative_build_ledger=False)
    approved_parent = tmp_path / "approved_b0_parent"
    approved_parent.mkdir()
    attempt = approved_parent / "runtime-manifest-tamper-attempt"
    command = _driver_command(
        repo,
        acceptance,
        approved_parent,
        attempt,
        Path(sys.executable),
        Path(sys.prefix),
    )
    manifest_path = Path(command[command.index("--runtime-manifest") + 1])
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")

    completed = subprocess.run(
        command,
        env=_driver_environment(tmp_path),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=DRIVER_TEST_TIMEOUT_SECONDS,
    )

    assert completed.returncode != 0
    assert (attempt / "audit/00_preflight/completion.json").is_file()
    assert not (attempt / "audit/01_canonical_validation").exists()
    assert not (attempt / "driver_completion.json").exists()
    failure = json.loads((attempt / "failure/failure.json").read_text(encoding="utf-8"))
    assert failure["reason"] == "AUDITED_NODE_FAILED_00_preflight"


@pytest.mark.parametrize(
    "mutation",
    (
        "extra_top_level",
        "truthy_non_neural",
        "extra_package",
        "uppercase_overlay_sha",
        "boolean_overlay_sha",
        "integer_overlay_sha",
    ),
)
def test_runtime_manifest_strict_v1_rejects_schema_or_type_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    repo = _production_fixture_repo(tmp_path)
    attempt = tmp_path / "attempt-parent" / f"runtime-{mutation}"
    attempt.parent.mkdir()
    manifest_path = _runtime_fixture_manifest(
        repo,
        attempt,
        Path(sys.executable),
        Path(sys.prefix),
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "extra_top_level":
        payload["unexpected"] = False
    elif mutation == "truthy_non_neural":
        payload["non_neural"] = 1
    elif mutation == "extra_package":
        payload["packages"]["unexpected"] = dict(payload["packages"]["numpy"])
    elif mutation == "uppercase_overlay_sha":
        payload["overlay_file"]["sha256"] = payload["overlay_file"]["sha256"].upper()
    elif mutation == "boolean_overlay_sha":
        payload["overlay_file"]["sha256"] = True
    elif mutation == "integer_overlay_sha":
        payload["overlay_file"]["sha256"] = 1
    _write_json(manifest_path, payload)

    with pytest.raises(b0_driver_guard.GuardError):
        b0_driver_guard._validate_runtime_manifest_file(
            manifest_path,
            expected_sha256=_sha256(manifest_path),
            expected_python_bin=str(Path(sys.executable)),
            expected_runtime_prefix=Path(sys.prefix),
            expected_project_root=repo,
        )


def test_runtime_overlay_drift_is_detected_after_node_and_blocks_completion(
    tmp_path: Path,
) -> None:
    repo = _production_fixture_repo(tmp_path)
    acceptance = _d1_fixture(tmp_path, relative_build_ledger=False)
    approved_parent = tmp_path / "approved_b0_parent"
    approved_parent.mkdir()
    attempt = approved_parent / "runtime-overlay-drift-attempt"
    command = _driver_command(
        repo,
        acceptance,
        approved_parent,
        attempt,
        Path(sys.executable),
        Path(sys.prefix),
    )
    manifest_path = Path(command[command.index("--runtime-manifest") + 1])
    runtime_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    environment = _driver_environment(tmp_path)
    environment["B0_TEST_MUTATE_RUNTIME_OVERLAY_PATH"] = runtime_manifest[
        "overlay_file"
    ]["path"]

    completed = subprocess.run(
        command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=DRIVER_TEST_TIMEOUT_SECONDS,
    )

    assert completed.returncode == 74
    assert (attempt / "audit/01_canonical_validation/completion.json").is_file()
    assert not (attempt / "audit/02_split_5utr_source").exists()
    assert not (attempt / "driver_completion.json").exists()
    failure = json.loads((attempt / "failure/failure.json").read_text(encoding="utf-8"))
    assert failure["reason"] == "POST_NODE_FREEZE_DRIFT_01_canonical_validation"


def test_sigterm_is_forwarded_exactly_and_records_safe_pause(
    tmp_path: Path,
) -> None:
    repo = _production_fixture_repo(tmp_path)
    acceptance = _d1_fixture(tmp_path, relative_build_ledger=False)
    approved_parent = tmp_path / "approved_b0_parent"
    approved_parent.mkdir()
    attempt = approved_parent / "signal-forwarding-attempt"
    environment = _driver_environment(tmp_path)
    environment["B0_TEST_SLEEP_CANONICAL_SECONDS"] = "30"
    process = subprocess.Popen(
        _driver_command(
            repo,
            acceptance,
            approved_parent,
            attempt,
            Path(sys.executable),
            Path(sys.prefix),
        ),
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + DRIVER_TEST_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        status_path = attempt / "status.json"
        if status_path.is_file():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                status = {}
            if status.get("current_node") == "01_canonical_validation" and isinstance(
                status.get("wrapper_pid"), int
            ):
                break
        if process.poll() is not None:
            break
        time.sleep(0.05)
    else:
        process.kill()
        pytest.fail("driver did not enter the canonical validation node")

    assert process.poll() is None
    process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(
        timeout=DRIVER_TEST_TIMEOUT_SECONDS,
    )

    assert process.returncode == 143, (stdout, stderr)
    status = json.loads((attempt / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "SAFE_PAUSED"
    assert status["terminal"] is True
    failure = json.loads((attempt / "failure/failure.json").read_text(encoding="utf-8"))
    assert failure["signal"] == "TERM"
    assert failure["reason"] == "SIGNAL_TERM"
    completion = json.loads(
        (attempt / "audit/01_canonical_validation/completion.json").read_text(
            encoding="utf-8"
        )
    )
    assert completion["interrupted_by_signal"] == signal.SIGTERM
    assert completion["stop_reason"] == "INTERRUPTED_BY_SIGNAL_15"
    assert not (attempt / "driver_completion.json").exists()


def test_signal_after_terminal_seal_rolls_forward_to_single_success(
    tmp_path: Path,
) -> None:
    repo = _production_fixture_repo(tmp_path, seal_return_delay=True)
    acceptance = _d1_fixture(tmp_path, relative_build_ledger=False)
    approved_parent = tmp_path / "approved_b0_parent"
    approved_parent.mkdir()
    attempt = approved_parent / "seal-signal-race-attempt"
    process = subprocess.Popen(
        _driver_command(
            repo,
            acceptance,
            approved_parent,
            attempt,
            Path(sys.executable),
            Path(sys.prefix),
        ),
        env=_driver_environment(tmp_path),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + DRIVER_TEST_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        status_path = attempt / "status.json"
        if status_path.is_file():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                status = {}
            if (
                status.get("state") == "B0_DRIVER_COMPLETED"
                and status.get("terminal") is True
            ):
                break
        if process.poll() is not None:
            break
        time.sleep(0.02)
    else:
        process.kill()
        pytest.fail("driver did not reach the terminal seal window")

    assert process.poll() is None
    process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(
        timeout=DRIVER_TEST_TIMEOUT_SECONDS,
    )

    assert process.returncode == 0, stderr
    status = json.loads((attempt / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "B0_DRIVER_COMPLETED"
    assert status["terminal"] is True
    events = [
        json.loads(line)
        for line in (attempt / "logs/events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    terminal = [
        event for event in events if event["event"] in b0_driver_guard.TERMINAL_EVENTS
    ]
    assert [event["event"] for event in terminal] == ["B0_DRIVER_COMPLETED"]
    assert not (attempt / "failure/failure.json").exists()
    assert any(
        json.loads(line).get("event") == "B0_DRIVER_COMPLETED"
        for line in stdout.splitlines()
        if line.strip()
    )


def test_tampered_non_driver_critical_code_is_rejected_before_node00_child(
    tmp_path: Path,
) -> None:
    repo = _production_fixture_repo(tmp_path)
    acceptance = _d1_fixture(tmp_path, relative_build_ledger=False)
    approved_parent = tmp_path / "approved_b0_parent"
    approved_parent.mkdir()
    attempt = approved_parent / "prelaunch-tamper-attempt"
    approved_dirty = b0_driver_guard._capture_git_snapshot(repo)["dirty_state_sha256"]
    (repo / "data/utr_benchmark_v2/track_loader.py").write_text(
        "# tampered after independent approval\n",
        encoding="utf-8",
    )
    command = _driver_command(
        repo,
        acceptance,
        approved_parent,
        attempt,
        Path(sys.executable),
        Path(sys.prefix),
        expected_dirty_state_sha256=approved_dirty,
    )

    completed = subprocess.run(
        command,
        env=_driver_environment(tmp_path),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=DRIVER_TEST_TIMEOUT_SECONDS,
    )

    assert completed.returncode == 74
    assert not (attempt / "audit/00_preflight").exists()
    assert not (attempt / "artifacts/preflight.json").exists()
    assert not (attempt / "audit/01_canonical_validation").exists()
    code_manifest = json.loads(
        (attempt / "provenance/code_manifest.json").read_text(encoding="utf-8")
    )
    assert code_manifest["caller_approval"]["passed"] is False
    assert code_manifest["caller_approval"]["checks"]["dirty_state_sha256"] is False
    assert (
        code_manifest["caller_approval"]["checks"]["critical_files_equal_head"] is False
    )
    assert code_manifest["critical_integrity"]["passed"] is False
    failure = json.loads((attempt / "failure/failure.json").read_text(encoding="utf-8"))
    assert failure["reason"] == "EARLY_DRIVER_BOOTSTRAP_FAILURE"
    assert not (attempt / "driver_completion.json").exists()
