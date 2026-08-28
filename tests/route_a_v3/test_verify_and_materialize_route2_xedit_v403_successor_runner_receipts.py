from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.route_a_v3.verify_and_materialize_route2_xedit_v403_successor_runner_receipts as verifier


HEAD = "a" * 40


def _patch_receipt_roots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path]:
    root = tmp_path / "mnt" / "route2"
    monkeypatch.setattr(verifier.critic_confirmation, "ROOT", root)
    monkeypatch.setattr(verifier.critic_controls, "ROOT", root)
    monkeypatch.setattr(verifier.setflow_s1_screen, "ROOT", root)
    return verifier.canonical_receipt_paths(HEAD)


def _patch_pre_gpu_consumers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        verifier.critic_controls,
        "validate_historical_full_terminal_audit",
        lambda: {"status": "TERMINAL_AUDIT_PASS"},
    )
    monkeypatch.setattr(
        verifier.critic_controls,
        "validate_training_source",
        lambda head: {"runner_git_head": head},
    )
    monkeypatch.setattr(
        verifier.setflow_s1_screen,
        "consume_old_s1_terminal_invalidation_receipt",
        lambda path: {
            "path": str(path),
            "status": "OLD_S1_TERMINAL_INVALIDATED",
        },
    )


def _focused_evidence() -> dict:
    commands = [
        " ".join(arguments) for arguments in verifier.focused_group_arguments()
    ]
    group_counts = [30, 31, 32, 33, 34, 35, 36, 37]
    return {
        "command": commands,
        "isolated_process_groups": True,
        "group_passed_counts": group_counts,
        "passed_count": sum(group_counts),
        "failed_count": 0,
        "passed": True,
    }


def _v332_evidence() -> dict:
    return {
        "command": [
            str(verifier.PYTHON),
            "-m",
            "pytest",
            "-q",
            verifier.V332_LITERAL_GLOB,
        ],
        "passed_count": 96,
        "failed_count": 0,
        "passed": True,
    }


def _valid_receipts() -> tuple[dict, dict]:
    return verifier.build_receipts(
        HEAD,
        focused_tests=_focused_evidence(),
        v332_tests=_v332_evidence(),
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_full_eight_group_contract_is_exactly_shared_and_direct_test_is_group8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oom_terminal_marker = (
        "test_transition_record_route2_xeditcritic_v403_controls_oom_terminal.py"
    )
    groups = verifier.require_exact_focused_group_contract()
    assert groups == (
        verifier.critic_confirmation.FOCUSED_GROUP_REQUIRED_TEST_MARKERS
    )
    assert groups == (
        verifier.setflow_authorizer.FOCUSED_GROUP_REQUIRED_TEST_MARKERS
    )
    assert len(groups) == 8
    commands = verifier.focused_group_arguments()
    assert oom_terminal_marker in groups[6]
    assert oom_terminal_marker in " ".join(commands[6])
    assert verifier.DIRECT_TEST_MODULE not in " ".join(commands[0])
    assert verifier.DIRECT_TEST_MODULE in " ".join(commands[7])

    monkeypatch.setattr(
        verifier.setflow_authorizer,
        "FOCUSED_GROUP_REQUIRED_TEST_MARKERS",
        groups[:-1],
    )
    with pytest.raises(
        verifier.SuccessorRunnerReceiptError, match="contracts differ"
    ):
        verifier.require_exact_focused_group_contract()


def test_focused_groups_are_spawned_concurrently_and_actual_counts_are_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[list[str]] = []

    class FakeProcess:
        def __init__(self, arguments, **_kwargs):
            self.arguments = list(arguments)
            self.index = len(started)
            self.returncode = 0
            started.append(self.arguments)

        def communicate(self):
            assert len(started) == 8
            return f"{self.index + 11} passed in 0.01s\n", ""

    monkeypatch.setattr(verifier.subprocess, "Popen", FakeProcess)
    evidence = verifier.run_focused_groups()
    assert len(started) == 8
    assert evidence["group_passed_counts"] == list(range(11, 19))
    assert evidence["passed_count"] == sum(range(11, 19))
    assert evidence["failed_count"] == 0
    assert evidence["isolated_process_groups"] is True


def test_v332_executes_literal_glob_match_and_preserves_literal_receipt_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tests = tmp_path / "tests" / "route_a_v3"
    tests.mkdir(parents=True)
    (tests / "test_alpha_v332_one.py").write_text("", encoding="utf-8")
    (tests / "test_beta_v332_two.py").write_text("", encoding="utf-8")
    captured: list[str] = []

    def fake_run(arguments, **_kwargs):
        captured.extend(arguments)
        return subprocess.CompletedProcess(arguments, 0, "96 passed in 1s\n", "")

    monkeypatch.setattr(verifier, "WORKTREE", tmp_path)
    monkeypatch.setattr(verifier, "PYTHON", Path("/repo/python"))
    monkeypatch.setattr(verifier, "run_command", fake_run)
    evidence = verifier.run_v332_cohort()
    assert verifier.V332_LITERAL_GLOB not in captured
    assert "tests/route_a_v3/test_alpha_v332_one.py" in captured
    assert "tests/route_a_v3/test_beta_v332_two.py" in captured
    assert evidence["command"][-1] == "tests/route_a_v3/*v332*.py"
    assert evidence["passed_count"] == 96
    assert evidence["failed_count"] == 0


def test_receipts_have_distinct_exact_schemas_and_identical_actual_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared_path, setflow_path = _patch_receipt_roots(monkeypatch, tmp_path)
    shared, setflow = _valid_receipts()
    verifier.validate_in_memory_receipts(
        HEAD, shared_path, setflow_path, shared, setflow
    )
    assert shared["schema_version"] == (
        "route_a_v3_route2_xedit_v403_successor_runner_verification_receipt.v1"
    )
    assert shared["status"] == "XEDIT_V403_SUCCESSOR_RUNNER_VERIFICATION_PASS"
    assert setflow["schema_version"] == (
        "route_a_v3_route2_xeditsetflow_v403_runner_verification_receipt.v1"
    )
    assert setflow["status"] == "XEDITSETFLOW_V403_RUNNER_VERIFICATION_PASS"
    assert shared["focused_tests"] == setflow["focused_tests"]
    assert shared["v332_tests"] == setflow["v332_tests"]
    assert shared["development_test_outcome_reads"] == 0
    assert shared["new_final_evaluation_outcome_reads"] == 0


@pytest.mark.parametrize("conflict", ["final", "partial"])
def test_existing_or_partial_receipt_refuses_before_any_pytest(
    conflict: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_path, _ = _patch_receipt_roots(monkeypatch, tmp_path)
    conflicting = (
        shared_path
        if conflict == "final"
        else shared_path.with_suffix(shared_path.suffix + ".partial")
    )
    conflicting.parent.mkdir(parents=True)
    conflicting.write_text("{}\n", encoding="utf-8")
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    monkeypatch.setattr(verifier, "PYTHON", python)
    monkeypatch.setattr(
        verifier, "require_exact_clean_pushed_identity", lambda _head: None
    )
    monkeypatch.setattr(
        verifier,
        "run_focused_groups",
        lambda: pytest.fail("pytest must not run for a consumed family"),
    )
    with pytest.raises(
        verifier.SuccessorRunnerReceiptError,
        match="already exists" if conflict == "final" else "partial receipt",
    ):
        verifier.verify_or_validate(HEAD)


def test_test_failure_publishes_no_pass_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared_path, setflow_path = _patch_receipt_roots(monkeypatch, tmp_path)
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    monkeypatch.setattr(verifier, "PYTHON", python)
    monkeypatch.setattr(
        verifier, "require_exact_clean_pushed_identity", lambda _head: None
    )
    monkeypatch.setattr(
        verifier,
        "run_focused_groups",
        lambda: (_ for _ in ()).throw(
            verifier.SuccessorRunnerReceiptError("focused group 3 failed")
        ),
    )
    with pytest.raises(verifier.SuccessorRunnerReceiptError, match="group 3"):
        verifier.verify_or_validate(HEAD)
    assert not shared_path.exists()
    assert not setflow_path.exists()
    assert not shared_path.with_suffix(shared_path.suffix + ".partial").exists()
    assert not setflow_path.with_suffix(setflow_path.suffix + ".partial").exists()


def test_validate_only_uses_existing_receipts_without_pytest_or_gpu_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared_path, setflow_path = _patch_receipt_roots(monkeypatch, tmp_path)
    shared, setflow = _valid_receipts()
    _write_json(shared_path, shared)
    _write_json(setflow_path, setflow)
    monkeypatch.setattr(
        verifier, "require_exact_clean_pushed_identity", lambda _head: None
    )
    monkeypatch.setattr(
        verifier,
        "run_focused_groups",
        lambda: pytest.fail("focused pytest cannot run in validate-only mode"),
    )
    monkeypatch.setattr(
        verifier,
        "run_v332_cohort",
        lambda: pytest.fail("V3.3.2 pytest cannot run in validate-only mode"),
    )
    _patch_pre_gpu_consumers(monkeypatch)
    for module, name in (
        (verifier.critic_controls, "run"),
        (verifier.setflow_s1_screen, "run"),
        (verifier.setflow_s1_screen, "gpu_diagnostics"),
        (verifier.setflow_s1_screen, "cuda_bf16_probe"),
    ):
        if hasattr(module, name):
            monkeypatch.setattr(
                module,
                name,
                lambda *_a, _name=name, **_k: pytest.fail(
                    f"GPU launcher path was invoked: {_name}"
                ),
            )

    terminal = verifier.verify_or_validate(
        HEAD, validate_receipts_only=True
    )
    assert terminal["status"] == verifier.TERMINAL_VALIDATED
    assert terminal["mode"] == "VALIDATE_RECEIPTS_ONLY"
    assert terminal["consumers"]["critic_controls_receipt_validation"] is True
    assert terminal["consumers"]["gpu_launcher_invoked"] is False
    assert terminal["consumers"]["runtime_read"] is False


def test_success_materializes_both_receipts_and_rechecks_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared_path, setflow_path = _patch_receipt_roots(monkeypatch, tmp_path)
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    monkeypatch.setattr(verifier, "PYTHON", python)
    identity_checks: list[str] = []
    monkeypatch.setattr(
        verifier,
        "require_exact_clean_pushed_identity",
        lambda head: identity_checks.append(head),
    )
    monkeypatch.setattr(verifier, "run_focused_groups", _focused_evidence)
    monkeypatch.setattr(verifier, "run_v332_cohort", _v332_evidence)
    _patch_pre_gpu_consumers(monkeypatch)

    terminal = verifier.verify_or_validate(HEAD)
    assert identity_checks == [HEAD, HEAD]
    assert terminal["status"] == verifier.TERMINAL_MATERIALIZED
    assert json.loads(shared_path.read_text())["status"] == (
        "XEDIT_V403_SUCCESSOR_RUNNER_VERIFICATION_PASS"
    )
    assert json.loads(setflow_path.read_text())["status"] == (
        "XEDITSETFLOW_V403_RUNNER_VERIFICATION_PASS"
    )
    assert terminal["focused_passed_count"] == sum(
        _focused_evidence()["group_passed_counts"]
    )
    assert terminal["v332_passed_count"] == 96


def test_post_publish_consumer_failure_preserves_validated_receipts_for_validate_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared_path, setflow_path = _patch_receipt_roots(monkeypatch, tmp_path)
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    monkeypatch.setattr(verifier, "PYTHON", python)
    monkeypatch.setattr(
        verifier, "require_exact_clean_pushed_identity", lambda _head: None
    )
    monkeypatch.setattr(verifier, "run_focused_groups", _focused_evidence)
    monkeypatch.setattr(verifier, "run_v332_cohort", _v332_evidence)
    monkeypatch.setattr(
        verifier,
        "validate_production_consumers",
        lambda *_a, **_k: (_ for _ in ()).throw(
            verifier.SuccessorRunnerReceiptError("consumer rejected receipt")
        ),
    )

    with pytest.raises(
        verifier.SuccessorRunnerReceiptError, match="consumer rejected"
    ):
        verifier.verify_or_validate(HEAD)
    assert shared_path.is_file()
    assert setflow_path.is_file()
    assert json.loads(shared_path.read_text())["status"] == (
        "XEDIT_V403_SUCCESSOR_RUNNER_VERIFICATION_PASS"
    )
    assert json.loads(setflow_path.read_text())["status"] == (
        "XEDITSETFLOW_V403_RUNNER_VERIFICATION_PASS"
    )


def test_main_prints_one_terminal_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        verifier,
        "verify_or_validate",
        lambda head, *, validate_receipts_only=False: {
            "schema_version": verifier.TERMINAL_SCHEMA,
            "status": verifier.TERMINAL_VALIDATED,
            "runner_git_head": head,
        },
    )
    assert verifier.main(["--expected-head", HEAD, "--validate-receipts-only"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["status"] == verifier.TERMINAL_VALIDATED
