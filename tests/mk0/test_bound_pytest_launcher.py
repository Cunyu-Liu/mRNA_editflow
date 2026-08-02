"""Fault-focused tests for the external current-worktree pytest launcher."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = ROOT / "scripts" / "mk0" / "run_bound_pytest.py"
SPEC = importlib.util.spec_from_file_location(
    "mk0_bound_pytest_launcher", LAUNCHER_PATH
)
assert SPEC is not None and SPEC.loader is not None
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_regular_tree(root: Path) -> None:
    for path in root.rglob("*"):
        metadata = path.lstat()
        assert not stat.S_ISLNK(metadata.st_mode), path
        assert stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode), path
        if stat.S_ISREG(metadata.st_mode):
            assert metadata.st_nlink == 1, path


def _tiny_import_test(path: Path) -> None:
    expected = str((ROOT / "__init__.py").resolve())
    path.write_text(
        "from pathlib import Path\n"
        "def test_current_worktree_import():\n"
        "    import mrna_editflow\n"
        f"    assert str(Path(mrna_editflow.__file__).resolve()) == {expected!r}\n",
        encoding="utf-8",
    )


def test_audit_marker_may_follow_quiet_progress_on_the_same_line() -> None:
    payload = {
        "schema_version": "mk0_pytest_audit_v1",
        "mode": "execute",
        "pytest_version": "8.3.4",
        "exitstatus": 0,
        "nodeids": ["test_one.py::test_one"],
        "collected_count": 1,
        "deselected_count": 0,
        "xfailed_count": 0,
        "xpassed_count": 0,
    }
    stdout = ". [100%]" + LAUNCHER.AUDIT_MARKER + json.dumps(payload) + "\n"

    assert LAUNCHER._audit_from_stdout(stdout) == payload


def test_current_worktree_import_junit_origin_and_regular_output(
    tmp_path: Path,
) -> None:
    formal_output = tmp_path / "formal-output"
    test_file = tmp_path / "test_current_import.py"
    _tiny_import_test(test_file)

    report = LAUNCHER.run_bound_pytest(
        repo_root=ROOT,
        formal_output_root=formal_output,
        pytest_args=[str(test_file)],
        junit_path=Path("evaluation/pytest_bound.junit.xml"),
        log_path=Path("logs/pytest_bound.log"),
        report_path=Path("evaluation/pytest_bound_origin.json"),
    )

    assert report["status"] == "PASS"
    assert report["returncode"] == 0
    assert report["schema_version"] == "mk0_bound_pytest_report_v2"
    assert report["collection_returncode"] == 0
    assert report["pytest_returncode"] == 0
    assert report["execution_started"] is True
    assert report["contract_violations"] == []
    assert report["collected_count"] == 1
    assert report["executed_count"] == 1
    assert report["passed_count"] == 1
    assert report["failed_count"] == 0
    assert report["error_count"] == 0
    assert report["skipped_count"] == 0
    assert report["deselected_count"] == 0
    assert report["xfailed_count"] == 0
    assert report["xpassed_count"] == 0
    assert report["collection_nodeids"] == report["execution_nodeids"]
    assert report["collection_nodeids"] == [
        "test_current_import.py::test_current_worktree_import"
    ]
    assert report["collection_nodeids_sha256"] == LAUNCHER._nodeids_sha256(
        report["collection_nodeids"]
    )
    assert report["execution_nodeids_sha256"] == LAUNCHER._nodeids_sha256(
        report["execution_nodeids"]
    )
    assert isinstance(report["pytest_version"], str) and report["pytest_version"]
    assert report["collect_command"][-2] == "--collect-only"
    assert report["command"][-1].startswith("--junitxml=")
    assert report["junit"]["totals"] == {
        "tests": 1,
        "errors": 0,
        "failures": 0,
        "skipped": 0,
        "passed": 1,
    }
    assert report["module_origin"]["matches_current_worktree"] is True
    assert report["module_origin"]["resolved_init"] == str(
        (ROOT / "__init__.py").resolve()
    )
    assert report["module_origin"]["resolved_search_locations"] == [str(ROOT)]
    assert report["module_origin"]["strict_importer_loaded_from_source_bytes"] is True
    strict_importer = ROOT / "scripts" / "mk0" / "strict_worktree_import.py"
    assert report["module_origin"]["strict_importer_path"] == str(strict_importer)
    assert report["module_origin"]["strict_importer_sha256"] == _sha256(strict_importer)
    isolation = report["import_isolation"]
    assert isolation == {
        "method": "external_ephemeral_symlink",
        "resolved_target": str(ROOT),
        "inside_formal_output_tree": False,
        "external_import_root_removed": True,
        "ambient_pythonpath_replaced": True,
    }

    junit = Path(report["junit"]["path"])
    log = Path(report["log"]["path"])
    report_path = formal_output / "evaluation/pytest_bound_origin.json"
    assert junit.is_file() and report["junit"]["sha256"] == _sha256(junit)
    assert log.is_file() and report["log"]["sha256"] == _sha256(log)
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert not (formal_output / "import_root").exists()
    _assert_regular_tree(formal_output)


def test_stale_parent_import_and_pythonpath_are_replaced(
    tmp_path: Path, monkeypatch
) -> None:
    stale_root = tmp_path / "stale"
    stale_package = stale_root / "mrna_editflow"
    stale_package.mkdir(parents=True)
    (stale_package / "__init__.py").write_text("ORIGIN = 'stale'\n", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(stale_root))
    stale_module = types.ModuleType("mrna_editflow")
    stale_module.__file__ = str(stale_package / "__init__.py")
    monkeypatch.setitem(sys.modules, "mrna_editflow", stale_module)

    formal_output = tmp_path / "formal-stale-output"
    test_file = tmp_path / "test_stale_replaced.py"
    _tiny_import_test(test_file)
    report = LAUNCHER.run_bound_pytest(
        repo_root=ROOT,
        formal_output_root=formal_output,
        pytest_args=[str(test_file)],
        junit_path=Path("evaluation/junit.xml"),
        log_path=Path("logs/pytest.log"),
        report_path=Path("evaluation/origin.json"),
    )

    assert report["status"] == "PASS"
    assert report["module_origin"]["matches_current_worktree"] is True
    assert report["module_origin"]["resolved_init"] != str(
        stale_package / "__init__.py"
    )
    assert report["import_isolation"]["ambient_pythonpath_replaced"] is True
    assert sys.modules["mrna_editflow"] is stale_module
    _assert_regular_tree(formal_output)


def test_pytest_failure_still_writes_bound_regular_evidence(tmp_path: Path) -> None:
    formal_output = tmp_path / "formal-failed-output"
    test_file = tmp_path / "test_expected_failure.py"
    test_file.write_text(
        "def test_injected_failure():\n    assert False, 'injected'\n",
        encoding="utf-8",
    )

    report = LAUNCHER.run_bound_pytest(
        repo_root=ROOT,
        formal_output_root=formal_output,
        pytest_args=[str(test_file)],
        junit_path=Path("evaluation/junit.xml"),
        log_path=Path("logs/pytest.log"),
        report_path=Path("evaluation/origin.json"),
    )

    assert report["status"] == "FAILED"
    assert report["returncode"] == 1
    assert report["pytest_returncode"] == 1
    assert report["failed_count"] == 1
    assert report["junit"]["totals"]["failures"] == 1
    assert (
        "formal pytest failures count must be zero, observed 1"
        in report["contract_violations"]
    )
    assert report["module_origin"]["matches_current_worktree"] is True
    assert report["junit"]["exists"] is True
    assert Path(report["junit"]["path"]).is_file()
    assert report["import_isolation"]["external_import_root_removed"] is True
    _assert_regular_tree(formal_output)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO unavailable")
def test_preexisting_special_node_is_rejected_before_launcher_writes(
    tmp_path: Path,
) -> None:
    formal_output = tmp_path / "formal-special-output"
    formal_output.mkdir()
    fifo = formal_output / "unexpected_fifo"
    os.mkfifo(fifo)
    before = sorted(path.name for path in formal_output.iterdir())

    with pytest.raises(LAUNCHER.BoundPytestError, match="special node"):
        LAUNCHER.run_bound_pytest(
            repo_root=ROOT,
            formal_output_root=formal_output,
            pytest_args=[str(tmp_path / "unused.py")],
            junit_path=Path("evaluation/junit.xml"),
            log_path=Path("logs/pytest.log"),
            report_path=Path("evaluation/origin.json"),
        )

    assert sorted(path.name for path in formal_output.iterdir()) == before
    assert fifo.exists()


def test_ambient_pytest_and_python_controls_are_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_file = tmp_path / "test_ambient_controls.py"
    test_file.write_text(
        "def test_selected_by_ambient_filter():\n"
        "    assert True\n\n"
        "def test_would_be_hidden_by_ambient_filter():\n"
        "    assert False, 'ambient -k must not prune this test'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k selected_by_ambient_filter")
    monkeypatch.setenv("PYTEST_PLUGINS", "ambient_secret_plugin")
    monkeypatch.setenv("PYTHONHASHSEED", "ambient_secret_seed")
    formal_output = tmp_path / "formal-ambient-output"

    report = LAUNCHER.run_bound_pytest(
        repo_root=ROOT,
        formal_output_root=formal_output,
        pytest_args=[str(test_file)],
        junit_path=Path("evaluation/junit.xml"),
        log_path=Path("logs/pytest.log"),
        report_path=Path("evaluation/origin.json"),
    )

    assert report["status"] == "FAILED"
    assert report["pytest_returncode"] == 1
    assert report["collected_count"] == 2
    assert report["executed_count"] == 2
    assert report["failed_count"] == 1
    assert report["deselected_count"] == 0
    environment = report["environment_contract"]
    assert environment["pytest_plugin_autoload_disabled"] is True
    assert {"PYTEST_ADDOPTS", "PYTEST_PLUGINS"}.issubset(
        environment["sanitized_pytest_environment_keys"]
    )
    assert "PYTHONHASHSEED" in environment["sanitized_python_environment_keys"]
    serialized_report = json.dumps(report, sort_keys=True)
    assert "ambient_secret_plugin" not in serialized_report
    assert "ambient_secret_seed" not in serialized_report
    assert "selected_by_ambient_filter" in serialized_report
    _assert_regular_tree(formal_output)


@pytest.mark.parametrize(
    "pytest_args",
    [
        ["-k", "selected"],
        ["-m=slow"],
        ["--deselect=test_module.py::test_case"],
        ["--ignore=test_module.py"],
        ["test_module.py::test_case"],
        ["--collect-only"],
        ["-p", "unbound_plugin"],
        ["-qk", "selected"],
    ],
)
def test_caller_selection_and_control_arguments_are_rejected(
    tmp_path: Path, pytest_args: list[str]
) -> None:
    formal_output = tmp_path / "formal-rejected-args"

    with pytest.raises(
        LAUNCHER.BoundPytestError,
        match="forbidden selection/control argument",
    ):
        LAUNCHER.run_bound_pytest(
            repo_root=ROOT,
            formal_output_root=formal_output,
            pytest_args=pytest_args,
            junit_path=Path("evaluation/junit.xml"),
            log_path=Path("logs/pytest.log"),
            report_path=Path("evaluation/origin.json"),
        )

    assert list(formal_output.rglob("*")) == []


def test_skip_is_a_formal_contract_failure_with_truthful_junit_totals(
    tmp_path: Path,
) -> None:
    test_file = tmp_path / "test_skip.py"
    test_file.write_text(
        "import pytest\n\n"
        "def test_pass():\n"
        "    assert True\n\n"
        "@pytest.mark.skip(reason='injected unavailable gate')\n"
        "def test_skipped_gate():\n"
        "    assert False\n",
        encoding="utf-8",
    )
    formal_output = tmp_path / "formal-skip-output"

    report = LAUNCHER.run_bound_pytest(
        repo_root=ROOT,
        formal_output_root=formal_output,
        pytest_args=[str(test_file)],
        junit_path=Path("evaluation/junit.xml"),
        log_path=Path("logs/pytest.log"),
        report_path=Path("evaluation/origin.json"),
    )

    assert report["status"] == "FAILED"
    assert report["pytest_returncode"] == 0
    assert report["returncode"] == LAUNCHER.CONTRACT_FAILURE_EXIT_CODE
    assert report["collected_count"] == 2
    assert report["executed_count"] == 2
    assert report["passed_count"] == 1
    assert report["skipped_count"] == 1
    assert report["junit"]["totals"]["skipped"] == 1
    assert (
        "formal pytest skipped count must be zero, observed 1"
        in report["contract_violations"]
    )
    assert Path(report["log"]["path"]).is_file()
    _assert_regular_tree(formal_output)


def test_collect_execute_nodeid_mismatch_fails_closed_with_evidence(
    tmp_path: Path,
) -> None:
    state_file = tmp_path / "collection-state"
    test_file = tmp_path / "test_dynamic_collection.py"
    test_file.write_text(
        "from pathlib import Path\n"
        f"state = Path({str(state_file)!r})\n"
        "if state.exists():\n"
        "    def test_execution_only():\n"
        "        assert True\n"
        "else:\n"
        "    state.write_text('collected', encoding='utf-8')\n"
        "    def test_collection_only():\n"
        "        assert True\n",
        encoding="utf-8",
    )
    formal_output = tmp_path / "formal-mismatch-output"

    report = LAUNCHER.run_bound_pytest(
        repo_root=ROOT,
        formal_output_root=formal_output,
        pytest_args=[str(test_file)],
        junit_path=Path("evaluation/junit.xml"),
        log_path=Path("logs/pytest.log"),
        report_path=Path("evaluation/origin.json"),
    )

    assert report["pytest_returncode"] == 0
    assert report["returncode"] == LAUNCHER.CONTRACT_FAILURE_EXIT_CODE
    assert report["status"] == "FAILED"
    assert report["collected_count"] == report["executed_count"] == 1
    assert report["collection_nodeids"] == [
        "test_dynamic_collection.py::test_collection_only"
    ]
    assert report["execution_nodeids"] == [
        "test_dynamic_collection.py::test_execution_only"
    ]
    assert report["collection_nodeids_sha256"] != report["execution_nodeids_sha256"]
    assert (
        "collect-only and execution nodeid inventories differ"
        in report["contract_violations"]
    )
    assert Path(report["junit"]["path"]).is_file()
    assert Path(report["log"]["path"]).is_file()
    _assert_regular_tree(formal_output)


def test_collection_failure_writes_report_and_log_without_execution(
    tmp_path: Path,
) -> None:
    test_file = tmp_path / "test_collection_error.py"
    test_file.write_text("def test_broken(:\n    pass\n", encoding="utf-8")
    formal_output = tmp_path / "formal-collection-failure"

    report = LAUNCHER.run_bound_pytest(
        repo_root=ROOT,
        formal_output_root=formal_output,
        pytest_args=[str(test_file)],
        junit_path=Path("evaluation/junit.xml"),
        log_path=Path("logs/pytest.log"),
        report_path=Path("evaluation/origin.json"),
    )

    assert report["status"] == "FAILED"
    assert report["collection_returncode"] != 0
    assert report["returncode"] == report["collection_returncode"]
    assert report["execution_started"] is False
    assert report["pytest_returncode"] is None
    assert report["command"] is None
    assert report["junit"] == {
        "path": str(formal_output / "evaluation/junit.xml"),
        "exists": False,
        "sha256": None,
        "totals": None,
    }
    assert Path(report["log"]["path"]).is_file()
    report_path = formal_output / "evaluation/origin.json"
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    _assert_regular_tree(formal_output)
