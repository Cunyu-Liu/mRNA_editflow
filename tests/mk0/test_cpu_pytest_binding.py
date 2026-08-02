"""Fail-closed integration tests for CPU consumption of pytest evidence."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_cpu_runner():
    path = ROOT / "scripts" / "mk0" / "run_mk0_cpu_acceptance.py"
    spec = importlib.util.spec_from_file_location("mk0_cpu_pytest_binding_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CPU = _load_cpu_runner()


def _load_finalizer():
    path = ROOT / "scripts" / "mk0" / "finalize_mk0_acceptance.py"
    spec = importlib.util.spec_from_file_location("mk0_finalizer_pytest_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FINALIZER = _load_finalizer()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report(tmp_path: Path) -> tuple[dict, Path]:
    evaluation = tmp_path / "evaluation"
    logs = tmp_path / "logs"
    evaluation.mkdir(parents=True)
    logs.mkdir(parents=True)
    junit = evaluation / "pytest_mk0.junit.xml"
    log = logs / "pytest_mk0.log"
    junit.write_text(
        '<testsuites tests="2" errors="0" failures="0" skipped="0">'
        '<testsuite name="pytest" tests="2" errors="0" failures="0" skipped="0">'
        '<testcase classname="tests.mk0.test_cpu_pytest_binding" '
        'name="test_cpu_accepts_fully_bound_pytest_evidence"/>'
        '<testcase classname="tests.mk0.test_cpu_pytest_binding" '
        'name="test_cpu_rejects_nodeid_inventory_substitution"/>'
        "</testsuite></testsuites>\n",
        encoding="utf-8",
    )
    log.write_text("2 passed\n", encoding="utf-8")
    nodeids = [
        "tests/mk0/test_cpu_pytest_binding.py::test_cpu_accepts_fully_bound_pytest_evidence",
        "tests/mk0/test_cpu_pytest_binding.py::test_cpu_rejects_nodeid_inventory_substitution",
    ]
    nodeids_sha256 = hashlib.sha256(
        "".join(f"{nodeid}\n" for nodeid in nodeids).encode("utf-8")
    ).hexdigest()
    helper = ROOT / "scripts" / "mk0" / "strict_worktree_import.py"
    origin = {
        "module": "mrna_editflow",
        "resolved_init": str(ROOT / "__init__.py"),
        "resolved_search_locations": [str(ROOT)],
        "expected_init": str(ROOT / "__init__.py"),
        "expected_root": str(ROOT),
        "matches_current_worktree": True,
        "stale_module_names_removed": [],
        "strict_importer_path": str(helper),
        "strict_importer_sha256": _sha256(helper),
        "strict_importer_loaded_from_source_bytes": True,
    }
    report = {
        "schema_version": "mk0_bound_pytest_report_v2",
        "status": "PASS",
        "returncode": 0,
        "collection_returncode": 0,
        "pytest_returncode": 0,
        "repo_root": str(ROOT),
        "formal_output_root": str(tmp_path),
        "python_executable": "/usr/bin/python3",
        "pytest_args": ["tests/mk0"],
        "collect_command": ["python", "--collect-only", "tests/mk0"],
        "command": ["python", "tests/mk0"],
        "execution_started": True,
        "environment_contract": {
            "sanitized_pytest_environment_keys": ["PYTEST_ADDOPTS"],
            "sanitized_python_environment_keys": ["PYTHONPATH"],
            "controlled_environment_keys": [
                "MK0_EXPECTED_PACKAGE_INIT",
                "MK0_EXPECTED_PACKAGE_ROOT",
                "MK0_PYTEST_MODE",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
                "PYTHONDONTWRITEBYTECODE",
                "PYTHONNOUSERSITE",
                "PYTHONPATH",
            ],
            "pytest_plugin_autoload_disabled": True,
            "pythonpath_replaced_with_external_binding": True,
        },
        "pytest_version": pytest.__version__,
        "collection_nodeids": nodeids,
        "collection_nodeids_sha256": nodeids_sha256,
        "execution_nodeids": nodeids,
        "execution_nodeids_sha256": nodeids_sha256,
        "collected_count": 2,
        "executed_count": 2,
        "passed_count": 2,
        "failed_count": 0,
        "error_count": 0,
        "skipped_count": 0,
        "deselected_count": 0,
        "xfailed_count": 0,
        "xpassed_count": 0,
        "contract_violations": [],
        "module_origin": copy.deepcopy(origin),
        "collection_module_origin": copy.deepcopy(origin),
        "execution_module_origin": copy.deepcopy(origin),
        "import_isolation": {
            "method": "external_ephemeral_symlink",
            "resolved_target": str(ROOT),
            "inside_formal_output_tree": False,
            "external_import_root_removed": True,
            "ambient_pythonpath_replaced": True,
        },
        "junit": {
            "path": str(junit),
            "exists": True,
            "sha256": _sha256(junit),
            "totals": {
                "tests": 2,
                "errors": 0,
                "failures": 0,
                "skipped": 0,
                "passed": 2,
            },
        },
        "log": {"path": str(log), "sha256": _sha256(log)},
        "formal_output_tree_regular_only": True,
    }
    audits = [
        {
            "schema_version": "mk0_pytest_audit_v1",
            "mode": mode,
            "pytest_version": report["pytest_version"],
            "exitstatus": 0,
            "nodeids": nodeids,
            "collected_count": len(nodeids),
            "deselected_count": 0,
            "xfailed_count": 0,
            "xpassed_count": 0,
        }
        for mode in ("collect", "execute")
    ]
    log.write_text(
        "__MK0_BOUND_MODULE_ORIGIN__="
        + json.dumps(report["collection_module_origin"], sort_keys=True)
        + "\n__MK0_PYTEST_AUDIT__="
        + json.dumps(audits[0], sort_keys=True)
        + "\n__MK0_BOUND_MODULE_ORIGIN__="
        + json.dumps(report["execution_module_origin"], sort_keys=True)
        + "\n__MK0_PYTEST_AUDIT__="
        + json.dumps(audits[1], sort_keys=True)
        + "\n2 passed\n",
        encoding="utf-8",
    )
    report["log"]["sha256"] = _sha256(log)
    return report, ROOT / "scripts" / "mk0" / "run_bound_pytest.py"


def _validate(report: dict, tmp_path: Path, launcher: Path) -> None:
    CPU._validate_bound_pytest_report(
        report,
        output_dir=tmp_path,
        launcher_path=launcher,
    )


def test_cpu_accepts_fully_bound_pytest_evidence(tmp_path: Path) -> None:
    report, launcher = _report(tmp_path)
    _validate(report, tmp_path, launcher)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("skipped_count", "skipped_count"),
        ("deselected_count", "deselected_count"),
        ("xfailed_count", "xfailed_count"),
    ],
)
def test_cpu_rejects_unexecuted_or_relabeled_tests(
    tmp_path: Path, field: str, message: str
) -> None:
    report, launcher = _report(tmp_path)
    report[field] = 1
    with pytest.raises(CPU.AcceptanceFailure, match=message):
        _validate(report, tmp_path, launcher)


def test_cpu_rejects_nodeid_inventory_substitution(tmp_path: Path) -> None:
    report, launcher = _report(tmp_path)
    report["execution_nodeids"] = ["tests/mk0/substituted.py::test_fake"]
    with pytest.raises(CPU.AcceptanceFailure, match="nodeid inventory drift"):
        _validate(report, tmp_path, launcher)


def test_cpu_rejects_stale_junit_or_importer_hash(tmp_path: Path) -> None:
    report, launcher = _report(tmp_path)
    Path(report["junit"]["path"]).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(CPU.AcceptanceFailure, match="JUnit binding drift"):
        _validate(report, tmp_path, launcher)

    second_root = tmp_path / "second"
    report, launcher = _report(second_root)
    report["module_origin"]["strict_importer_sha256"] = "0" * 64
    with pytest.raises(CPU.AcceptanceFailure, match="source-byte import binding"):
        _validate(report, second_root, launcher)


def test_cpu_rejects_unsanitized_pytest_environment(tmp_path: Path) -> None:
    report, launcher = _report(tmp_path)
    report["environment_contract"]["pytest_plugin_autoload_disabled"] = False
    with pytest.raises(CPU.AcceptanceFailure, match="environment was not sanitized"):
        _validate(report, tmp_path, launcher)


def _finalizer_evidence(tmp_path: Path) -> tuple[dict, dict, str]:
    report, launcher = _report(tmp_path)
    python_executable = str(Path(sys.executable).resolve(strict=True))
    junit_path = tmp_path / "evaluation" / "pytest_mk0.junit.xml"
    bootstrap = FINALIZER._ast_string_constant(launcher, "BOOTSTRAP")
    report["python_executable"] = python_executable
    report["collect_command"] = [
        python_executable,
        "-c",
        bootstrap,
        "-q",
        "-p",
        "no:cacheprovider",
        "--collect-only",
        "tests/mk0",
    ]
    report["command"] = [
        python_executable,
        "-c",
        bootstrap,
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/mk0",
        f"--junitxml={junit_path}",
    ]
    persisted_path = tmp_path / "provenance" / "pytest_import_binding.json"
    persisted_path.parent.mkdir()
    persisted_path.write_bytes(FINALIZER.canonical_json_bytes(report))
    extended = {
        **report,
        "launcher_path": str(launcher),
        "launcher_sha256": _sha256(launcher),
        "report_path": str(persisted_path),
        "report_sha256": _sha256(persisted_path),
        "log_path": report["log"]["path"],
        "log_sha256": report["log"]["sha256"],
        "junit_path": report["junit"]["path"],
        "junit_sha256": report["junit"]["sha256"],
    }
    helper = ROOT / "scripts" / "mk0" / "strict_worktree_import.py"
    source_binding = {
        "tracked_source_files": {
            "scripts/mk0/run_bound_pytest.py": _sha256(launcher),
            "scripts/mk0/strict_worktree_import.py": _sha256(helper),
            "tests/mk0/test_cpu_pytest_binding.py": _sha256(Path(__file__)),
        }
    }
    return {"pytest": extended}, source_binding, python_executable


def test_finalizer_independently_accepts_complete_pytest_v2_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu, source_binding, python_executable = _finalizer_evidence(tmp_path)

    def matching_collection(**kwargs):
        nodeids = kwargs["expected_nodeids"]
        digest = hashlib.sha256(
            "".join(f"{nodeid}\n" for nodeid in nodeids).encode("utf-8")
        ).hexdigest()
        return (
            {
                "schema_version": "mk0_finalizer_pytest_collection_v1",
                "status": "PASS",
                "collected_count": len(nodeids),
                "nodeids_sha256": digest,
                "cpu_inventory_equal": True,
            },
            b"independent collection fixture\n",
        )

    monkeypatch.setattr(
        FINALIZER, "_independent_pytest_collection", matching_collection
    )
    verified = FINALIZER.verify_cpu_pytest_evidence(
        cpu,
        tmp_path,
        source_binding=source_binding,
        expected_python_executable=python_executable,
    )
    assert verified["collected_count"] == 2
    assert verified["passed_count"] == 2
    assert verified["all_forbidden_outcome_counts_zero"] is True
    assert verified["independent_collection"]["cpu_inventory_equal"] is True


def test_finalizer_rejects_self_consistent_incomplete_pytest_inventory(
    tmp_path: Path,
) -> None:
    run_root = (tmp_path / "incomplete").resolve()
    cpu, source_binding, python_executable = _finalizer_evidence(run_root)
    with pytest.raises(
        FINALIZER.FinalizeFailure,
        match="independent pytest collection differs from CPU collect/execute inventory",
    ):
        FINALIZER.verify_cpu_pytest_evidence(
            cpu,
            run_root,
            source_binding=source_binding,
            expected_python_executable=python_executable,
        )


def test_finalizer_rejects_missing_or_forged_pytest_evidence(tmp_path: Path) -> None:
    _, source_binding, python_executable = _finalizer_evidence(tmp_path)
    with pytest.raises(FINALIZER.FinalizeFailure, match="pytest evidence is absent"):
        FINALIZER.verify_cpu_pytest_evidence(
            {},
            tmp_path,
            source_binding=source_binding,
            expected_python_executable=python_executable,
        )

    cpu, source_binding, python_executable = _finalizer_evidence(tmp_path / "forged")
    cpu["pytest"]["launcher_sha256"] = "0" * 64
    with pytest.raises(FINALIZER.FinalizeFailure, match="launcher byte binding"):
        FINALIZER.verify_cpu_pytest_evidence(
            cpu,
            tmp_path / "forged",
            source_binding=source_binding,
            expected_python_executable=python_executable,
        )


def test_finalizer_rejects_persisted_pytest_report_drift(tmp_path: Path) -> None:
    cpu, source_binding, python_executable = _finalizer_evidence(tmp_path)
    persisted_path = tmp_path / "provenance" / "pytest_import_binding.json"
    persisted = json.loads(persisted_path.read_text(encoding="utf-8"))
    persisted["status"] = "FAILED"
    persisted_path.write_bytes(FINALIZER.canonical_json_bytes(persisted))

    with pytest.raises(FINALIZER.FinalizeFailure, match="embedded/persisted"):
        FINALIZER.verify_cpu_pytest_evidence(
            cpu,
            tmp_path,
            source_binding=source_binding,
            expected_python_executable=python_executable,
        )
