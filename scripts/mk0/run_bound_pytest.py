#!/usr/bin/env python3
"""Run pytest against the current worktree without polluting a formal run tree.

The repository root is itself the ``mrna_editflow`` package, so an isolated
import needs a parent directory containing a package-named entry.  This
launcher creates that entry only in an external temporary directory.  The
formal output tree receives regular log, JUnit, and origin-report files only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any, Sequence
import xml.etree.ElementTree as ET


ORIGIN_MARKER = "__MK0_BOUND_MODULE_ORIGIN__="
AUDIT_MARKER = "__MK0_PYTEST_AUDIT__="
CONTRACT_FAILURE_EXIT_CODE = 97
BOOTSTRAP = r"""
import hashlib
import json
import os
from pathlib import Path
import sys

stale_names = sorted(
    name
    for name in tuple(sys.modules)
    if name == "mrna_editflow" or name.startswith("mrna_editflow.")
)
expected_init = str(Path(os.environ["MK0_EXPECTED_PACKAGE_INIT"]).resolve(strict=True))
expected_root = str(Path(os.environ["MK0_EXPECTED_PACKAGE_ROOT"]).resolve(strict=True))
strict_importer_path = (
    Path(expected_root) / "scripts" / "mk0" / "strict_worktree_import.py"
).resolve(strict=True)
strict_importer_source = strict_importer_path.read_bytes()
strict_importer_namespace = {
    "__file__": str(strict_importer_path),
    "__name__": "_mk0_bound_strict_worktree_import",
}
exec(
    compile(
        strict_importer_source,
        str(strict_importer_path),
        "exec",
        dont_inherit=True,
        optimize=0,
    ),
    strict_importer_namespace,
)
strict_import = strict_importer_namespace["strict_worktree_package_import"]
strict_context = strict_import(Path(expected_root))
mrna_editflow = strict_context.__enter__()

try:
    resolved_init = str(Path(mrna_editflow.__file__).resolve(strict=True))
    resolved_search_locations = sorted(
        str(Path(location).resolve(strict=True)) for location in mrna_editflow.__path__
    )
    payload = {
        "module": "mrna_editflow",
        "resolved_init": resolved_init,
        "resolved_search_locations": resolved_search_locations,
        "expected_init": expected_init,
        "expected_root": expected_root,
        "matches_current_worktree": (
            resolved_init == expected_init
            and resolved_search_locations == [expected_root]
        ),
        "stale_module_names_removed": stale_names,
        "strict_importer_path": str(strict_importer_path),
        "strict_importer_sha256": hashlib.sha256(strict_importer_source).hexdigest(),
        "strict_importer_loaded_from_source_bytes": True,
    }
    print("__MK0_BOUND_MODULE_ORIGIN__=" + json.dumps(payload, sort_keys=True), flush=True)
    if not payload["matches_current_worktree"]:
        raise SystemExit(86)
    completed_context = strict_context
    strict_context = None
    completed_context.__exit__(None, None, None)

    import pytest

    class Mk0PytestAudit:
        def __init__(self):
            self.nodeids = []
            self.deselected_count = 0
            self.xfailed_nodeids = set()
            self.xpassed_nodeids = set()

        def pytest_collection_finish(self, session):
            self.nodeids = sorted(item.nodeid for item in session.items)

        def pytest_deselected(self, items):
            self.deselected_count += len(items)

        def pytest_runtest_logreport(self, report):
            if not hasattr(report, "wasxfail"):
                return
            if report.skipped:
                self.xfailed_nodeids.add(report.nodeid)
            elif report.passed:
                self.xpassed_nodeids.add(report.nodeid)

        def pytest_sessionfinish(self, session, exitstatus):
            payload = {
                "schema_version": "mk0_pytest_audit_v1",
                "mode": os.environ["MK0_PYTEST_MODE"],
                "pytest_version": pytest.__version__,
                "exitstatus": int(exitstatus),
                "nodeids": self.nodeids,
                "collected_count": len(self.nodeids),
                "deselected_count": self.deselected_count,
                "xfailed_count": len(self.xfailed_nodeids),
                "xpassed_count": len(self.xpassed_nodeids),
            }
            print(
                "__MK0_PYTEST_AUDIT__=" + json.dumps(payload, sort_keys=True),
                flush=True,
            )

    audit = Mk0PytestAudit()
    pytest_exit_code = pytest.main(sys.argv[1:], plugins=[audit])
finally:
    if strict_context is not None:
        strict_context.__exit__(*sys.exc_info())
raise SystemExit(pytest_exit_code)
"""


class BoundPytestError(RuntimeError):
    """The bound pytest launcher contract was violated."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_exclusive_atomic(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return hashlib.sha256(data).hexdigest()


def _assert_regular_tree(root: Path) -> None:
    """Reject aliases and special nodes in the formal output tree."""

    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise BoundPytestError(f"formal output contains a symlink: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise BoundPytestError(f"formal output contains a special node: {relative}")
        if metadata.st_nlink != 1:
            raise BoundPytestError(f"formal output contains a hardlink: {relative}")


def _output_path(root: Path, value: Path) -> Path:
    candidate = value if value.is_absolute() else root / value
    candidate = candidate.absolute()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise BoundPytestError(
            f"output path escapes formal root: {candidate}"
        ) from error
    candidate.parent.mkdir(parents=True, exist_ok=True)
    try:
        candidate.parent.resolve(strict=True).relative_to(root)
    except ValueError as error:
        raise BoundPytestError(
            f"output parent resolves outside formal root: {candidate.parent}"
        ) from error
    if candidate.exists() or candidate.is_symlink():
        raise BoundPytestError(f"refusing to overwrite pytest output: {candidate}")
    return candidate


def _origin_from_stdout(stdout: str) -> dict[str, Any]:
    records = [
        line[len(ORIGIN_MARKER) :]
        for line in stdout.splitlines()
        if line.startswith(ORIGIN_MARKER)
    ]
    if len(records) != 1:
        raise BoundPytestError("pytest process emitted no unique module-origin record")
    try:
        payload = json.loads(records[0])
    except json.JSONDecodeError as error:
        raise BoundPytestError("pytest module-origin record is invalid JSON") from error
    if not isinstance(payload, dict):
        raise BoundPytestError("pytest module-origin record is not an object")
    return payload


def _audit_from_stdout(stdout: str) -> dict[str, Any]:
    records = []
    for line in stdout.splitlines():
        marker_index = line.find(AUDIT_MARKER)
        if marker_index >= 0:
            records.append(line[marker_index + len(AUDIT_MARKER) :])
    if len(records) != 1:
        raise BoundPytestError("pytest process emitted no unique audit record")
    try:
        payload = json.loads(records[0])
    except json.JSONDecodeError as error:
        raise BoundPytestError("pytest audit record is invalid JSON") from error
    if not isinstance(payload, dict):
        raise BoundPytestError("pytest audit record is not an object")
    return payload


def _validate_pytest_args(pytest_args: Sequence[str]) -> list[str]:
    arguments = [str(argument) for argument in pytest_args]
    for argument in arguments:
        option = argument.split("=", 1)[0]
        if option in {"--junitxml", "--junit-xml"}:
            raise BoundPytestError("pytest_args must not override the bound JUnit path")
        if argument.startswith("-") or "::" in argument:
            raise BoundPytestError(
                f"pytest_args contains a forbidden selection/control argument: {argument}"
            )
    return arguments


def _sanitized_environment(
    *, import_root: Path, package_init: Path, repository: Path
) -> tuple[dict[str, str], dict[str, Any]]:
    environment = os.environ.copy()
    removed_pytest_keys = sorted(
        key for key in environment if key.startswith("PYTEST_")
    )
    removed_python_keys = sorted(key for key in environment if key.startswith("PYTHON"))
    for key in set(removed_pytest_keys) | set(removed_python_keys):
        environment.pop(key, None)
    environment.update(
        {
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONPATH": str(import_root),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "MK0_EXPECTED_PACKAGE_INIT": str(package_init),
            "MK0_EXPECTED_PACKAGE_ROOT": str(repository),
        }
    )
    contract = {
        "sanitized_pytest_environment_keys": removed_pytest_keys,
        "sanitized_python_environment_keys": removed_python_keys,
        "controlled_environment_keys": sorted(
            [
                "MK0_EXPECTED_PACKAGE_INIT",
                "MK0_EXPECTED_PACKAGE_ROOT",
                "MK0_PYTEST_MODE",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
                "PYTHONDONTWRITEBYTECODE",
                "PYTHONNOUSERSITE",
                "PYTHONPATH",
            ]
        ),
        "pytest_plugin_autoload_disabled": True,
        "pythonpath_replaced_with_external_binding": True,
    }
    return environment, contract


def _nodeids_sha256(nodeids: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{nodeid}\n" for nodeid in nodeids).encode("utf-8")
    ).hexdigest()


def _validate_audit_payload(
    payload: dict[str, Any], *, expected_mode: str, expected_returncode: int
) -> list[str]:
    violations: list[str] = []
    nodeids = payload.get("nodeids")
    if payload.get("schema_version") != "mk0_pytest_audit_v1":
        violations.append(f"{expected_mode} audit schema is invalid")
    if payload.get("mode") != expected_mode:
        violations.append(f"{expected_mode} audit mode is invalid")
    if payload.get("exitstatus") != expected_returncode:
        violations.append(f"{expected_mode} audit exit status differs from process")
    if not isinstance(payload.get("pytest_version"), str) or not payload.get(
        "pytest_version"
    ):
        violations.append(f"{expected_mode} pytest version is unavailable")
    if (
        not isinstance(nodeids, list)
        or any(not isinstance(nodeid, str) or not nodeid for nodeid in nodeids)
        or nodeids != sorted(nodeids)
        or len(nodeids) != len(set(nodeids))
    ):
        violations.append(f"{expected_mode} nodeid inventory is invalid")
    elif payload.get("collected_count") != len(nodeids):
        violations.append(f"{expected_mode} collected count differs from nodeids")
    for field in ("deselected_count", "xfailed_count", "xpassed_count"):
        value = payload.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            violations.append(f"{expected_mode} {field} is invalid")
    return violations


def _junit_totals(path: Path) -> dict[str, int]:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as error:
        raise BoundPytestError("pytest JUnit evidence is invalid") from error

    def local_name(element: ET.Element) -> str:
        return element.tag.rsplit("}", 1)[-1]

    suites = [
        element
        for element in root.iter()
        if local_name(element) == "testsuite"
        and not any(local_name(child) == "testsuite" for child in element)
    ]
    if not suites:
        raise BoundPytestError("pytest JUnit contains no test suite")
    totals = {name: 0 for name in ("tests", "errors", "failures", "skipped")}
    for suite in suites:
        for name in totals:
            raw_value = suite.get(name)
            try:
                value = int(raw_value) if raw_value is not None else None
            except ValueError as error:
                raise BoundPytestError(
                    f"pytest JUnit {name} total is invalid"
                ) from error
            if value is None or value < 0:
                raise BoundPytestError(f"pytest JUnit {name} total is invalid")
            totals[name] += value
    passed = totals["tests"] - totals["errors"] - totals["failures"] - totals["skipped"]
    if passed < 0:
        raise BoundPytestError("pytest JUnit outcome totals are inconsistent")
    return {**totals, "passed": passed}


def run_bound_pytest(
    *,
    repo_root: Path,
    formal_output_root: Path,
    pytest_args: Sequence[str],
    junit_path: Path,
    log_path: Path,
    report_path: Path,
    python_executable: str | Path = sys.executable,
) -> dict[str, Any]:
    """Run pytest in a fresh interpreter bound to ``repo_root``.

    The returned report is also written atomically to ``report_path``.  A test
    failure is represented by a non-zero ``returncode`` in that report; launcher
    contract failures raise :class:`BoundPytestError`.
    """

    raw_output_root = formal_output_root.expanduser().absolute()
    if raw_output_root.is_symlink():
        raise BoundPytestError("formal output root must not be a symlink")
    raw_output_root.mkdir(parents=True, exist_ok=True)
    output_root = raw_output_root.resolve(strict=True)
    repository = repo_root.expanduser().resolve(strict=True)
    package_init = repository / "__init__.py"
    if not package_init.is_file() or package_init.is_symlink():
        raise BoundPytestError("repo root is not the current mrna_editflow package")
    _assert_regular_tree(output_root)
    bound_pytest_args = _validate_pytest_args(pytest_args)

    junit = _output_path(output_root, junit_path)
    log = _output_path(output_root, log_path)
    report_file = _output_path(output_root, report_path)
    if len({junit, log, report_file}) != 3:
        raise BoundPytestError("JUnit, log, and report paths must be distinct")

    temporary_parent = Path(tempfile.gettempdir()).resolve(strict=True)
    if temporary_parent == output_root or output_root in temporary_parent.parents:
        raise BoundPytestError("system temporary root is inside the formal output tree")

    ambient_pythonpath = os.environ.get("PYTHONPATH")
    collect_command: list[str]
    command: list[str]
    collect_completed: subprocess.CompletedProcess[str]
    completed: subprocess.CompletedProcess[str] | None = None
    collection_origin: dict[str, Any] | None = None
    collection_audit: dict[str, Any] | None = None
    execution_origin: dict[str, Any] | None = None
    execution_audit: dict[str, Any] | None = None
    contract_violations: list[str] = []
    external_import_root_removed = False
    with tempfile.TemporaryDirectory(
        prefix=f"mk0-{output_root.name}-import-", dir=temporary_parent
    ) as temporary_text:
        import_root = Path(temporary_text).resolve(strict=True)
        if import_root == output_root or output_root in import_root.parents:
            raise BoundPytestError(
                "temporary import root entered the formal output tree"
            )
        binding = import_root / "mrna_editflow"
        os.symlink(repository, binding, target_is_directory=True)
        if not binding.is_symlink() or binding.resolve(strict=True) != repository:
            raise BoundPytestError("temporary worktree import binding is invalid")

        environment, environment_contract = _sanitized_environment(
            import_root=import_root,
            package_init=package_init,
            repository=repository,
        )
        collect_command = [
            str(python_executable),
            "-c",
            BOOTSTRAP,
            "-q",
            "-p",
            "no:cacheprovider",
            "--collect-only",
            *bound_pytest_args,
        ]
        command = [
            str(python_executable),
            "-c",
            BOOTSTRAP,
            "-q",
            "-p",
            "no:cacheprovider",
            *bound_pytest_args,
            f"--junitxml={junit}",
        ]
        collect_environment = environment.copy()
        collect_environment["MK0_PYTEST_MODE"] = "collect"
        collect_completed = subprocess.run(
            collect_command,
            cwd=repository,
            env=collect_environment,
            capture_output=True,
            text=True,
        )
        try:
            collection_origin = _origin_from_stdout(collect_completed.stdout)
        except BoundPytestError as error:
            contract_violations.append(str(error))
        try:
            collection_audit = _audit_from_stdout(collect_completed.stdout)
        except BoundPytestError as error:
            contract_violations.append(str(error))
        if collection_audit is not None:
            contract_violations.extend(
                _validate_audit_payload(
                    collection_audit,
                    expected_mode="collect",
                    expected_returncode=collect_completed.returncode,
                )
            )
            for outcome in ("deselected_count", "xfailed_count", "xpassed_count"):
                if collection_audit.get(outcome) != 0:
                    contract_violations.append(
                        f"collect-only {outcome} must be zero, observed "
                        f"{collection_audit.get(outcome)}"
                    )
        if (
            collection_origin is not None
            and collection_origin.get("matches_current_worktree") is not True
        ):
            contract_violations.append(
                "collect-only imported mrna_editflow from another worktree"
            )
        if collect_completed.returncode == 0 and not contract_violations:
            execution_environment = environment.copy()
            execution_environment["MK0_PYTEST_MODE"] = "execute"
            completed = subprocess.run(
                command,
                cwd=repository,
                env=execution_environment,
                capture_output=True,
                text=True,
            )
            try:
                execution_origin = _origin_from_stdout(completed.stdout)
            except BoundPytestError as error:
                contract_violations.append(str(error))
            try:
                execution_audit = _audit_from_stdout(completed.stdout)
            except BoundPytestError as error:
                contract_violations.append(str(error))
            if execution_audit is not None:
                contract_violations.extend(
                    _validate_audit_payload(
                        execution_audit,
                        expected_mode="execute",
                        expected_returncode=completed.returncode,
                    )
                )
                if collection_audit is not None and execution_audit.get(
                    "pytest_version"
                ) != collection_audit.get("pytest_version"):
                    contract_violations.append(
                        "collect-only and execution pytest versions differ"
                    )
            if (
                execution_origin is not None
                and execution_origin.get("matches_current_worktree") is not True
            ):
                contract_violations.append(
                    "pytest execution imported mrna_editflow from another worktree"
                )
    external_import_root_removed = not Path(temporary_text).exists()

    log_sections = [
        "=== MK0 PYTEST COLLECT-ONLY STDOUT ===\n" + collect_completed.stdout,
        "=== MK0 PYTEST COLLECT-ONLY STDERR ===\n" + collect_completed.stderr,
    ]
    if completed is not None:
        log_sections.extend(
            [
                "=== MK0 PYTEST EXECUTION STDOUT ===\n" + completed.stdout,
                "=== MK0 PYTEST EXECUTION STDERR ===\n" + completed.stderr,
            ]
        )
    log_bytes = "\n".join(log_sections).encode("utf-8")
    log_sha256 = _write_exclusive_atomic(log, log_bytes)
    junit_exists = junit.is_file() and not junit.is_symlink()
    junit_totals: dict[str, int] | None = None
    if junit_exists:
        try:
            junit_totals = _junit_totals(junit)
        except BoundPytestError as error:
            contract_violations.append(str(error))
    if not external_import_root_removed:
        contract_violations.append("external import root was not cleaned")
    if completed is not None and completed.returncode == 0 and not junit_exists:
        contract_violations.append("passing pytest run did not produce JUnit evidence")

    collection_nodeids = (
        list(collection_audit.get("nodeids", []))
        if isinstance(collection_audit, dict)
        and isinstance(collection_audit.get("nodeids"), list)
        else []
    )
    execution_nodeids = (
        list(execution_audit.get("nodeids", []))
        if isinstance(execution_audit, dict)
        and isinstance(execution_audit.get("nodeids"), list)
        else []
    )
    if completed is not None and collection_nodeids != execution_nodeids:
        contract_violations.append(
            "collect-only and execution nodeid inventories differ"
        )

    collected_count = len(collection_nodeids) if collection_audit is not None else None
    executed_count = junit_totals["tests"] if junit_totals is not None else None
    passed_count = junit_totals["passed"] if junit_totals is not None else None
    deselected_count = (
        execution_audit.get("deselected_count") if execution_audit is not None else None
    )
    xfailed_count = (
        execution_audit.get("xfailed_count") if execution_audit is not None else None
    )
    xpassed_count = (
        execution_audit.get("xpassed_count") if execution_audit is not None else None
    )
    if completed is not None and junit_totals is not None:
        forbidden_outcomes = {
            "errors": junit_totals["errors"],
            "failures": junit_totals["failures"],
            "skipped": junit_totals["skipped"],
            "deselected": deselected_count,
            "xfailed": xfailed_count,
            "xpassed": xpassed_count,
        }
        for outcome, count in forbidden_outcomes.items():
            if count != 0:
                contract_violations.append(
                    f"formal pytest {outcome} count must be zero, observed {count}"
                )
        if not (
            executed_count == collected_count == passed_count
            and collected_count is not None
            and collected_count > 0
        ):
            contract_violations.append(
                "formal pytest requires executed == collected == passed > 0"
            )

    pytest_returncode = completed.returncode if completed is not None else None
    if collect_completed.returncode != 0:
        returncode = collect_completed.returncode
    elif completed is not None and completed.returncode != 0:
        returncode = completed.returncode
    elif contract_violations or completed is None:
        returncode = CONTRACT_FAILURE_EXIT_CODE
    else:
        returncode = 0
    module_origin = execution_origin or collection_origin or {}
    report: dict[str, Any] = {
        "schema_version": "mk0_bound_pytest_report_v2",
        "status": "PASS" if returncode == 0 else "FAILED",
        "returncode": returncode,
        "collection_returncode": collect_completed.returncode,
        "pytest_returncode": pytest_returncode,
        "repo_root": str(repository),
        "formal_output_root": str(output_root),
        "python_executable": str(Path(python_executable).resolve(strict=True)),
        "pytest_args": bound_pytest_args,
        "collect_command": collect_command,
        "command": command if completed is not None else None,
        "execution_started": completed is not None,
        "environment_contract": environment_contract,
        "pytest_version": (execution_audit or collection_audit or {}).get(
            "pytest_version"
        ),
        "collection_nodeids": collection_nodeids,
        "collection_nodeids_sha256": _nodeids_sha256(collection_nodeids),
        "execution_nodeids": execution_nodeids,
        "execution_nodeids_sha256": _nodeids_sha256(execution_nodeids),
        "collected_count": collected_count,
        "executed_count": executed_count,
        "passed_count": passed_count,
        "failed_count": (
            junit_totals["failures"] if junit_totals is not None else None
        ),
        "error_count": junit_totals["errors"] if junit_totals is not None else None,
        "skipped_count": (
            junit_totals["skipped"] if junit_totals is not None else None
        ),
        "deselected_count": deselected_count,
        "xfailed_count": xfailed_count,
        "xpassed_count": xpassed_count,
        "contract_violations": contract_violations,
        "module_origin": module_origin,
        "collection_module_origin": collection_origin,
        "execution_module_origin": execution_origin,
        "import_isolation": {
            "method": "external_ephemeral_symlink",
            "resolved_target": str(repository),
            "inside_formal_output_tree": False,
            "external_import_root_removed": external_import_root_removed,
            "ambient_pythonpath_replaced": ambient_pythonpath != str(import_root),
        },
        "junit": {
            "path": str(junit),
            "exists": junit_exists,
            "sha256": _sha256_file(junit) if junit_exists else None,
            "totals": junit_totals,
        },
        "log": {"path": str(log), "sha256": log_sha256},
        "formal_output_tree_regular_only": True,
    }

    _write_exclusive_atomic(report_file, _canonical_json_bytes(report))
    _assert_regular_tree(output_root)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--formal-output-root", required=True, type=Path)
    parser.add_argument("--junit-path", required=True, type=Path)
    parser.add_argument("--log-path", required=True, type=Path)
    parser.add_argument("--report-path", required=True, type=Path)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    pytest_args = list(args.pytest_args)
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]
    report = run_bound_pytest(
        repo_root=args.repo_root,
        formal_output_root=args.formal_output_root,
        pytest_args=pytest_args,
        junit_path=args.junit_path,
        log_path=args.log_path,
        report_path=args.report_path,
        python_executable=args.python_executable,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return int(report["returncode"])


if __name__ == "__main__":
    raise SystemExit(main())
