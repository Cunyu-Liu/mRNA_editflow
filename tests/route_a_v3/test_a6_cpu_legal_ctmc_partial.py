from __future__ import annotations

import copy
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/route_a_v3_a6_cpu_legal_ctmc_partial_v1.json"
SCRIPT_PATH = REPO_ROOT / "scripts/route_a_v3/run_a6_cpu_legal_ctmc_partial.py"
SPEC = importlib.util.spec_from_file_location("a6_cpu_legal_ctmc_partial", SCRIPT_PATH)
assert SPEC and SPEC.loader
a6 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = a6
SPEC.loader.exec_module(a6)


I_COMMIT = "1" * 40
B_COMMIT = "2" * 40
BRANCH = "routea-v3-a6-gillespie"


def config() -> dict[str, Any]:
    return a6.load_config(CONFIG_PATH)


def bound_config() -> dict[str, Any]:
    value = a6.candidate_i_config(config())
    value["implementation_binding"] = {
        "status": "BOUND",
        "implementation_commit": I_COMMIT,
        "implementation_script_sha256": a6.sha256(SCRIPT_PATH.read_bytes()),
        "implementation_test_sha256": a6.sha256(Path(__file__).read_bytes()),
    }
    a6.validate_static_config(value)
    return value


@pytest.fixture(scope="module")
def passed_suite() -> dict[str, Any]:
    return a6.run_sampling_suite(config(), repo_root=REPO_ROOT)


def test_static_contract_step_holding_time_alias_and_replay() -> None:
    frozen = config()
    assert a6._binding_mode(frozen) in {"UNKNOWN", "BOUND"}
    candidate = a6.candidate_i_config(frozen)
    assert a6._binding_mode(candidate) == "UNKNOWN"
    a6.validate_static_config(candidate)
    kernel, kernel_config = a6.load_kernel(REPO_ROOT, frozen)
    case = next(item for item in kernel_config["fixed_cases"] if item["case_id"] == "L2_B2")
    initial = kernel.initial_state(case, kernel_config)
    transitions = kernel.canonical_transitions(initial, kernel_config)
    total = math.fsum(item.rate for item in transitions)

    event = a6.gillespie_step(
        kernel,
        kernel_config,
        initial,
        survival_uniform=0.25,
        jump_uniform=0.375,
    )
    assert event.holding_time == pytest.approx(-math.log(0.25) / total, rel=1e-15)
    assert event.after.algorithmic_time == pytest.approx(initial.algorithmic_time + event.holding_time)
    assert event.total_exit_rate == pytest.approx(total)
    assert sum(rate for _, rate in event.alias_pairs) == pytest.approx(event.selected_rate)
    assert len(event.alias_pairs) == 2

    rng = __import__("random").Random(123)
    trajectory = a6.sample_trajectory(kernel, kernel_config, initial, rng, maximum_jumps=3)
    assert trajectory.final.terminal_cause in kernel.TERMINAL_CAUSES
    assert trajectory.events
    assert a6.replay_trajectory(kernel, kernel_config, trajectory) == trajectory
    assert a6._trajectory_violation_counts(kernel, kernel_config, trajectory) == {}
    if trajectory.final.terminal_cause == "EXPLICIT_STOP":
        assert len(trajectory.events) == trajectory.final.net_edit_count + 1
    else:
        assert len(trajectory.events) == trajectory.final.net_edit_count


def test_structural_terminals_are_distinct_and_invalid_uniform_fails() -> None:
    frozen = config()
    kernel, kernel_config = a6.load_kernel(REPO_ROOT, frozen)
    fixtures = kernel_config["terminal_fixture_states"]
    causes = kernel._fixture_terminal_causes(kernel_config)
    assert causes == {cause: cause for cause in kernel.TERMINAL_CAUSES}

    budget = fixtures["BUDGET_EXHAUSTED"]
    budget_state = kernel.State(
        budget["source_sequence"],
        budget["current_sequence"],
        tuple(tuple(edit) for edit in budget["source_relative_edit_set"]),
        budget["remaining_budget"],
        "TOY_ASSAY_ALPHA",
        "TOY_CONTEXT_LEFT",
        0.0,
    )
    assert kernel.with_structural_terminal(budget_state, kernel_config).terminal_cause == "BUDGET_EXHAUSTED"
    with pytest.raises(a6.NumericalError, match="survival uniform"):
        a6.gillespie_step(kernel, kernel_config, kernel.initial_state(kernel_config["fixed_cases"][2], kernel_config), survival_uniform=0.0, jump_uniform=0.5)


def test_frozen_sampling_suite_passes_exact_dp_and_replay(passed_suite: dict[str, Any]) -> None:
    frozen = config()
    suite = passed_suite
    assert suite["status"] == "PASS"
    assert suite["trajectory_count"] == 20000
    assert suite["replay_trajectory_count"] == 256
    assert set(suite["checks"].values()) == {"PASS"}
    assert sum(suite["terminal_cause_counts"].values()) == 20000
    assert suite["terminal_cause_counts"]["NUMERICAL_FAILURE"] == 0
    assert suite["total_jump_count"] >= suite["total_source_relative_edit_count"]
    assert sum(suite["source_relative_edit_count_histogram"].values()) == 20000
    assert suite["metrics"]["terminal_distribution_tv"] <= frozen["acceptance"]["terminal_distribution_tv_max"]
    assert (
        suite["metrics"]["initial_holding_time_mean_relative_error"]
        <= frozen["acceptance"]["initial_holding_time_mean_relative_error_max"]
    )
    for key, value in suite["metrics"].items():
        if key.endswith("_count"):
            assert value == 0
    assert suite["compute_ledger"]["device"] == "CPU"
    assert suite["compute_ledger"]["parameter_update_count"] == 0


def test_nonproduction_atomic_exact3_is_aggregate_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, passed_suite: dict[str, Any]
) -> None:
    frozen = config()
    output = tmp_path / "bundle"
    monkeypatch.setattr(a6, "run_sampling_suite", lambda *_args, **_kwargs: passed_suite)
    result = a6.execute(
        output_directory=output,
        recorded_at="2026-08-13T20:00:00+08:00",
        production=False,
        config_override=frozen,
        repo_root=REPO_ROOT,
    )
    assert result["status"] == "PASS"
    assert sorted(path.name for path in output.iterdir()) == sorted(a6.OUTPUT_NAMES)
    report = json.loads((output / a6.OUTPUT_NAMES[0]).read_text())
    manifest = json.loads((output / a6.OUTPUT_NAMES[1]).read_text())
    event_lines = (output / a6.OUTPUT_NAMES[2]).read_text().splitlines()
    assert len(event_lines) == 1
    event = json.loads(event_lines[0])
    assert report["run_status"] == manifest["run_status"] == event["run_status"] == "PASS"
    assert report["phase_state"] == {"phase_id": "A6", "evidence_status": "IN_PROGRESS", "phase_complete": False}
    assert report["task_states"]["FLOW_BASE_LEGAL_CTMC"]["evidence_status"] == "IN_PROGRESS"
    assert report["claim_state"]["claim_status"] == "NOT_ESTABLISHED"
    assert report["boundaries"]["a7_unlock"] is False
    assert manifest["output_count"] == 3
    serialized = "\n".join(path.read_text() for path in output.iterdir())
    assert "source_sequence" not in serialized
    assert "current_sequence" not in serialized
    assert "trajectory_events" not in serialized
    with pytest.raises(a6.PublicationError, match="already exists"):
        a6.execute(
            output_directory=output,
            recorded_at="2026-08-13T20:00:01+08:00",
            production=False,
            config_override=frozen,
            repo_root=REPO_ROOT,
        )


def test_unknown_binding_stops_before_git_output_and_numerics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    frozen = a6.candidate_i_config(config())
    monkeypatch.setattr(a6, "_git_bytes", lambda *_args: (_ for _ in ()).throw(AssertionError("Git must not run")))
    monkeypatch.setattr(
        a6,
        "_validate_output_path",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("output must not be inspected")),
    )
    monkeypatch.setattr(
        a6,
        "run_sampling_suite",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("numerics must not run")),
    )
    with pytest.raises(a6.AuthorityError, match="not BOUND"):
        a6.execute(
            output_directory=tmp_path / "never",
            recorded_at="2026-08-13T20:00:00+08:00",
            production=True,
            config_override=frozen,
            repo_root=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


def _fake_git_payloads(repo: Path, frozen: dict[str, Any]) -> dict[tuple[str, ...], bytes]:
    candidate = a6.candidate_i_config(frozen)
    initial_candidate = a6.expected_initial_i_config(candidate)
    for path in (
        a6.CONFIG_REPO_PATH,
        a6.SCRIPT_REPO_PATH,
        a6.TEST_REPO_PATH,
        frozen["authority"]["goal_path"],
        frozen["authority"]["active_config_path"],
        *frozen["authority"]["dependency_leaves"],
    ):
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if path == a6.CONFIG_REPO_PATH:
            target.write_bytes(a6.json_bytes(frozen))
        else:
            target.write_bytes((REPO_ROOT / path).read_bytes())
    script = (repo / a6.SCRIPT_REPO_PATH).read_bytes()
    test = (repo / a6.TEST_REPO_PATH).read_bytes()
    initial_bound = copy.deepcopy(initial_candidate)
    initial_bound["implementation_binding"] = {
        "status": "BOUND",
        "implementation_commit": a6.INITIAL_IMPLEMENTATION_COMMIT,
        "implementation_script_sha256": a6.sha256(script),
        "implementation_test_sha256": a6.sha256(test),
    }
    result = {
        ("rev-parse", "--show-toplevel"): f"{repo}\n".encode(),
        ("status", "--porcelain"): b"",
        ("branch", "--show-current"): f"{BRANCH}\n".encode(),
        ("rev-parse", "HEAD"): f"{B_COMMIT}\n".encode(),
        ("rev-parse", "@{u}"): f"{B_COMMIT}\n".encode(),
        ("rev-parse", f"refs/remotes/origin/{BRANCH}"): f"{B_COMMIT}\n".encode(),
        ("rev-parse", "--symbolic-full-name", "@{u}"): f"refs/remotes/origin/{BRANCH}\n".encode(),
        ("rev-parse", f"{B_COMMIT}^"): f"{I_COMMIT}\n".encode(),
        ("rev-parse", f"{I_COMMIT}^"): f"{a6.INITIAL_BINDING_COMMIT}\n".encode(),
        ("rev-parse", f"{a6.INITIAL_BINDING_COMMIT}^"): f"{a6.INITIAL_IMPLEMENTATION_COMMIT}\n".encode(),
        ("rev-parse", f"{a6.INITIAL_IMPLEMENTATION_COMMIT}^"): f"{a6.FROZEN_BASE_COMMIT}\n".encode(),
        ("diff-tree", "--no-commit-id", "--name-only", "-r", a6.INITIAL_IMPLEMENTATION_COMMIT): (
            "\n".join(a6.EXACT_IMPLEMENTATION_PATHS) + "\n"
        ).encode(),
        ("diff-tree", "--no-commit-id", "--name-only", "-r", a6.INITIAL_BINDING_COMMIT): (
            a6.CONFIG_REPO_PATH + "\n"
        ).encode(),
        ("diff-tree", "--no-commit-id", "--name-only", "-r", I_COMMIT): (
            "\n".join(a6.EXACT_IMPLEMENTATION_PATHS) + "\n"
        ).encode(),
        ("diff-tree", "--no-commit-id", "--name-only", "-r", B_COMMIT): (a6.CONFIG_REPO_PATH + "\n").encode(),
        ("show", f"{I_COMMIT}:{a6.CONFIG_REPO_PATH}"): a6.json_bytes(candidate),
        ("show", f"{B_COMMIT}:{a6.CONFIG_REPO_PATH}"): a6.json_bytes(frozen),
        ("show", f"{a6.INITIAL_IMPLEMENTATION_COMMIT}:{a6.CONFIG_REPO_PATH}"): a6.json_bytes(initial_candidate),
        ("show", f"{a6.INITIAL_BINDING_COMMIT}:{a6.CONFIG_REPO_PATH}"): a6.json_bytes(initial_bound),
        ("show", f"{I_COMMIT}:{a6.SCRIPT_REPO_PATH}"): script,
        ("show", f"{B_COMMIT}:{a6.SCRIPT_REPO_PATH}"): script,
        ("show", f"{a6.INITIAL_IMPLEMENTATION_COMMIT}:{a6.SCRIPT_REPO_PATH}"): script,
        ("show", f"{a6.INITIAL_BINDING_COMMIT}:{a6.SCRIPT_REPO_PATH}"): script,
        ("show", f"{I_COMMIT}:{a6.TEST_REPO_PATH}"): test,
        ("show", f"{B_COMMIT}:{a6.TEST_REPO_PATH}"): test,
        ("show", f"{a6.INITIAL_IMPLEMENTATION_COMMIT}:{a6.TEST_REPO_PATH}"): test,
        ("show", f"{a6.INITIAL_BINDING_COMMIT}:{a6.TEST_REPO_PATH}"): test,
    }
    for path in frozen["authority"]["dependency_leaves"]:
        payload = (repo / path).read_bytes()
        for commit in (a6.FROZEN_BASE_COMMIT, I_COMMIT, B_COMMIT):
            result[("show", f"{commit}:{path}")] = payload
    return result


def test_production_authority_exact_base_i_b_and_dependency_blobs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    frozen = bound_config()
    payloads = _fake_git_payloads(tmp_path, frozen)

    def fake_git(_repo: Path, *args: str) -> bytes:
        return payloads[tuple(args)]

    monkeypatch.setattr(a6, "_git_bytes", fake_git)
    observed = a6.validate_production_authority(frozen, repo_root=tmp_path)
    assert observed["implementation_commit"] == I_COMMIT
    assert observed["binding_commit"] == B_COMMIT

    payloads[("rev-parse", f"{I_COMMIT}^")] = ("3" * 40 + "\n").encode()
    with pytest.raises(a6.AuthorityError, match="binding B1"):
        a6.validate_production_authority(frozen, repo_root=tmp_path)
