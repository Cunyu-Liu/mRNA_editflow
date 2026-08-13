from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = STAGING_ROOT / "configs/route_a_v3_a6_cpu_exact_absorbing_dag_v1.json"
SCRIPT_PATH = STAGING_ROOT / "scripts/route_a_v3/run_a6_cpu_exact_absorbing_dag.py"
SPEC = importlib.util.spec_from_file_location("a6_cpu_exact", SCRIPT_PATH)
assert SPEC and SPEC.loader
a6 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = a6
SPEC.loader.exec_module(a6)


def config() -> dict[str, Any]:
    return a6.load_config(CONFIG_PATH)


I_COMMIT = "1" * 40
B_COMMIT = "2" * 40
BRANCH = "routea-v3-a6-cpu-exact"


def bound_config() -> dict[str, Any]:
    frozen = a6.candidate_i_config(config())
    frozen["implementation_binding"] = {
        "status": "BOUND",
        "implementation_commit": I_COMMIT,
        "implementation_script_sha256": a6.sha256(SCRIPT_PATH.read_bytes()),
        "implementation_test_sha256": a6.sha256(Path(__file__).read_bytes()),
    }
    a6.validate_static_config(frozen)
    return frozen


def fake_git_payloads(repo: Path, frozen: dict[str, Any]) -> dict[tuple[str, ...], bytes]:
    candidate_i = a6.candidate_i_config(frozen)
    script = SCRIPT_PATH.read_bytes()
    test = Path(__file__).read_bytes()
    for relative, payload in (
        (a6.CONFIG_REPO_PATH, a6.json_bytes(frozen)),
        (a6.SCRIPT_REPO_PATH, script),
        (a6.TEST_REPO_PATH, test),
    ):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return {
        ("rev-parse", "--show-toplevel"): f"{repo}\n".encode(),
        ("status", "--porcelain"): b"",
        ("branch", "--show-current"): f"{BRANCH}\n".encode(),
        ("rev-parse", "HEAD"): f"{B_COMMIT}\n".encode(),
        ("rev-parse", "@{u}"): f"{B_COMMIT}\n".encode(),
        ("rev-parse", f"refs/remotes/origin/{BRANCH}"): f"{B_COMMIT}\n".encode(),
        ("rev-parse", "--symbolic-full-name", "@{u}"): f"refs/remotes/origin/{BRANCH}\n".encode(),
        ("rev-parse", f"{B_COMMIT}^"): f"{I_COMMIT}\n".encode(),
        ("rev-parse", f"{I_COMMIT}^"): f"{a6.FROZEN_BASE_COMMIT}\n".encode(),
        ("diff-tree", "--no-commit-id", "--name-only", "-r", I_COMMIT): (
            "\n".join(a6.EXACT_IMPLEMENTATION_PATHS) + "\n"
        ).encode(),
        ("diff-tree", "--no-commit-id", "--name-only", "-r", B_COMMIT): (
            a6.CONFIG_REPO_PATH + "\n"
        ).encode(),
        ("show", f"{I_COMMIT}:{a6.CONFIG_REPO_PATH}"): a6.json_bytes(candidate_i),
        ("show", f"{B_COMMIT}:{a6.CONFIG_REPO_PATH}"): a6.json_bytes(frozen),
        ("show", f"{I_COMMIT}:{a6.SCRIPT_REPO_PATH}"): script,
        ("show", f"{B_COMMIT}:{a6.SCRIPT_REPO_PATH}"): script,
        ("show", f"{I_COMMIT}:{a6.TEST_REPO_PATH}"): test,
        ("show", f"{B_COMMIT}:{a6.TEST_REPO_PATH}"): test,
    }


def active_state(*, budget: int = 2, time: float = 0.25) -> Any:
    return a6.State("AC", "AC", (), budget, "TOY_ASSAY_ALPHA", "TOY_CONTEXT_LEFT", time)


def parse_outputs(directory: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    report = json.loads((directory / a6.OUTPUT_NAMES[0]).read_text())
    manifest = json.loads((directory / a6.OUTPUT_NAMES[1]).read_text())
    lines = (directory / a6.OUTPUT_NAMES[2]).read_text().splitlines()
    assert len(lines) == 1
    return report, manifest, json.loads(lines[0])


def test_static_contract_and_all_fixed_flow_fixtures_pass() -> None:
    frozen = config()
    assert a6._binding_mode(frozen) in {"UNKNOWN", "BOUND"}
    candidate_i = a6.candidate_i_config(frozen)
    assert set(candidate_i["implementation_binding"].values()) == {a6.UNKNOWN}
    a6.validate_static_config(candidate_i)
    bound = bound_config()
    assert a6._binding_mode(bound) == "BOUND"
    assert a6.candidate_i_config(bound) == candidate_i
    assert frozen["production_python"] == "/home/cunyuliu/miniconda3/envs/editflow/bin/python"
    assert frozen["clock_semantics"] == "CONTINUOUS_ALGORITHMIC_TIME"
    assert frozen["rate_time_dependence"] == "NONE"
    assert frozen["terminal_tilt_time_dependence"] == "NONE"
    assert frozen["general_time_inhomogeneous_exactness"] == "NOT_RUN"
    assert {(len(item["source_sequence"]), item["budget"]) for item in frozen["fixed_cases"]} == {
        (length, budget) for length in (2, 3) for budget in (0, 1, 2)
    }

    suite = a6.run_exact_suite(frozen)
    assert suite["status"] == "PASS"
    assert suite["fixed_case_count"] == 6
    assert suite["terminal_fixture_causes"] == {cause: cause for cause in a6.TERMINAL_CAUSES}
    assert set(suite["fixture_results"].values()) == {"PASS"}
    assert {item["case_id"] for item in suite["cases"]} == {
        "L2_B0", "L2_B1", "L2_B2", "L3_B0", "L3_B1", "L3_B2"
    }
    tolerance = frozen["numerical_tolerances"]
    for case in suite["cases"]:
        assert case["status"] == "PASS"
        assert set(case["checks"].values()) == {"PASS"}
        assert case["budget_violation_count"] == 0
        assert case["event_count_violation_count"] == 0
        assert case["metrics"]["true_per_rate_relative_error"] <= tolerance["true_rate_relative_error_max"]
        assert case["metrics"]["guided_terminal_tv_vs_tilted_base"] <= tolerance["terminal_distribution_tv_max"]
        assert case["metrics"]["path_product_relative_error"] <= tolerance["path_product_relative_error_max"]
        assert case["metrics"]["guided_vs_base_total_exit_rate_relative_error"] <= tolerance["total_exit_rate_relative_error_max"]


def test_hard_legality_source_anchor_no_reedit_revert_and_alias_full_state() -> None:
    frozen = config()
    state = active_state()
    legal = a6.RawAction(a6.EDIT, "EDIT_CHANNEL_PRIMARY", 0, "A", "G")
    first = a6.transition_state(state, legal, frozen)
    assert first.current_sequence == "GC"
    assert first.source_relative_edit_set == ((0, "G"),)
    assert first.net_edit_count == 1

    repeated = a6.RawAction(a6.EDIT, "EDIT_CHANNEL_PRIMARY", 0, "A", "U")
    revert = a6.RawAction(a6.EDIT, "EDIT_CHANNEL_PRIMARY", 0, "A", "A")
    current_base_semantics = a6.RawAction(a6.EDIT, "EDIT_CHANNEL_PRIMARY", 0, "G", "U")
    for old_invalid in (repeated, revert, current_base_semantics):
        assert not a6.is_action_legal(first, old_invalid, frozen)
        assert a6.raw_action_rate(first, old_invalid, frozen) == 0.0
        with pytest.raises(a6.StateError, match="illegal action"):
            a6.transition_state(first, old_invalid, frozen)

    canonical = a6.canonical_transitions(state, frozen)
    assert canonical
    assert all(len(item.raw_alias_ids) == 2 for item in canonical)
    assert all(item.rate == pytest.approx(math.fsum(item.raw_rates), abs=1e-15) for item in canonical)
    stop = next(item for item in canonical if item.action_type == a6.STOP)
    edit = next(item for item in canonical if item.action_type == a6.EDIT)
    forged_aliases = (
        a6.RawTransition(a6.RawAction(a6.STOP, "A"), stop.next_state, 1.0),
        a6.RawTransition(a6.RawAction(a6.STOP, "B"), edit.next_state, 2.0),
    )
    assert len(a6.aggregate_raw_transitions(forged_aliases)) == 2


def test_w_h_v_doob_generators_and_independent_enumeration_are_rate_level() -> None:
    frozen = config()
    case = next(item for item in frozen["fixed_cases"] if item["case_id"] == "L3_B2")
    graph = a6.build_graph(a6.initial_state(case, frozen), frozen)
    h_dp = a6.harmonic_extension_dp(graph, frozen)
    h_enum = a6.harmonic_extension_enumeration(graph, frozen)
    base = a6.generator(graph)
    guided = a6.generator(graph, h_dp)
    guided_independent = a6.generator(graph, h_enum)
    assert h_dp == pytest.approx(h_enum, rel=1e-12, abs=1e-12)
    assert all(math.isfinite(math.log(value)) for value in h_dp.values())

    for state, row in guided.items():
        assert row.diagonal == pytest.approx(-sum(rate for _, rate in row.off_diagonal), abs=1e-14)
        assert row.total_exit_rate == pytest.approx(base[state].total_exit_rate, rel=1e-12, abs=1e-12)
        base_rates = dict(base[state].off_diagonal)
        independent_rates = dict(guided_independent[state].off_diagonal)
        for child, rate in row.off_diagonal:
            assert rate == pytest.approx(base_rates[child] * h_dp[child] / h_dp[state], rel=1e-14)
            assert rate == pytest.approx(independent_rates[child], rel=1e-12)

    unit_h = a6.harmonic_extension_dp(graph, frozen, unit_tilt=True)
    assert set(unit_h.values()) == {1.0}
    unit = a6.generator(graph, unit_h)
    assert unit == base

    base_dp = a6.terminal_distribution_dp(graph)
    base_paths, count = a6.terminal_distribution_enumeration(graph)
    assert count > 0
    assert a6._tv(base_dp, base_paths) <= 1e-12


def test_invalid_rate_fails_closed_before_zero_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    frozen = config()
    output = tmp_path / "never-created"
    original = a6._canonical_base_rate

    def invalid_rate(state: Any, action: Any, config_value: Any) -> float:
        if action.action_type == a6.EDIT:
            return float("nan")
        return original(state, action, config_value)

    monkeypatch.setattr(a6, "_canonical_base_rate", invalid_rate)
    with pytest.raises(a6.NumericalFailure, match="strictly positive"):
        a6.execute(
            output_directory=output,
            recorded_at="2026-08-13T18:00:00+08:00",
            production=False,
            config_override=frozen,
        )
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_authority_failure_precedes_numerics_and_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    frozen = config()
    output = tmp_path / "never-created"
    monkeypatch.setattr(
        a6,
        "validate_production_authority",
        lambda _config: (_ for _ in ()).throw(a6.AuthorityError("authority drift")),
    )
    monkeypatch.setattr(
        a6,
        "run_exact_suite",
        lambda _config: (_ for _ in ()).throw(AssertionError("numerics must not run")),
    )
    with pytest.raises(a6.AuthorityError, match="authority drift"):
        a6.execute(
            output_directory=output,
            recorded_at="2026-08-13T18:00:00+08:00",
            production=True,
            config_override=frozen,
        )
    assert not output.exists()


def test_unknown_candidate_i_stops_before_git_numerics_and_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frozen = a6.candidate_i_config(config())
    output = tmp_path / "never-created"
    monkeypatch.setattr(
        a6,
        "_git_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Git must not run")),
    )
    monkeypatch.setattr(
        a6,
        "run_exact_suite",
        lambda _config: (_ for _ in ()).throw(AssertionError("numerics must not run")),
    )
    with pytest.raises(a6.AuthorityError, match="not BOUND"):
        a6.execute(
            output_directory=output,
            recorded_at="2026-08-13T18:00:00+08:00",
            production=True,
            config_override=frozen,
        )
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_fake_git_exact_i_to_b_authority_happy_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frozen = bound_config()
    repo = tmp_path / "repo"
    repo.mkdir()
    payloads = fake_git_payloads(repo, frozen)
    monkeypatch.setattr(a6, "_git_bytes", lambda _repo, *args: payloads[tuple(args)])
    monkeypatch.setattr(a6, "_validate_active_contract", lambda _repo, _config: None)
    result = a6.validate_production_authority(frozen, repo_root=repo)
    assert result["binding_commit"] == B_COMMIT
    assert result["implementation_commit"] == I_COMMIT
    assert result["implementation_script_sha256"] == frozen["implementation_binding"][
        "implementation_script_sha256"
    ]
    assert result["implementation_test_sha256"] == frozen["implementation_binding"][
        "implementation_test_sha256"
    ]


@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("parent", "direct child of the frozen base"),
        ("paths", "changed-path set is not exact3"),
        ("blob", "I/B/worktree blob differs"),
    ],
)
def test_fake_git_rejects_parent_path_and_blob_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    drift: str,
    message: str,
) -> None:
    frozen = bound_config()
    repo = tmp_path / "repo"
    repo.mkdir()
    payloads = fake_git_payloads(repo, frozen)
    if drift == "parent":
        payloads[("rev-parse", f"{I_COMMIT}^")] = ("3" * 40 + "\n").encode()
    elif drift == "paths":
        payloads[("diff-tree", "--no-commit-id", "--name-only", "-r", I_COMMIT)] = (
            a6.CONFIG_REPO_PATH + "\n"
        ).encode()
    else:
        payloads[("show", f"{B_COMMIT}:{a6.SCRIPT_REPO_PATH}")] = b"drifted script\n"
    monkeypatch.setattr(a6, "_git_bytes", lambda _repo, *args: payloads[tuple(args)])
    monkeypatch.setattr(a6, "_validate_active_contract", lambda _repo, _config: None)
    with pytest.raises(a6.AuthorityError, match=message):
        a6.validate_production_authority(frozen, repo_root=repo)


def test_success_atomically_publishes_exactly_three_public_aggregate_files(tmp_path: Path) -> None:
    frozen = config()
    output = tmp_path / "a6-public"
    result = a6.execute(
        output_directory=output,
        recorded_at="2026-08-13T18:00:00+08:00",
        production=False,
        config_override=frozen,
    )
    assert result["status"] == "PASS"
    assert sorted(path.name for path in output.iterdir()) == sorted(a6.OUTPUT_NAMES)
    report, manifest, event = parse_outputs(output)
    assert report["run_scope"] == "DEVELOPMENT_ONLY_CPU_EXACT_ABSORPTION_FIXTURE"
    assert report["run_status"] == "PASS"
    assert report["phase_state"] == {"evidence_status": "IN_PROGRESS", "phase_id": "A6"}
    assert report["task_states"]["EXACT_GUIDANCE_TOY_GRAPH"] == {
        "evidence_status": "PASS",
        "result": "DEVELOPMENT_CPU_EXACT_FIXTURE_PASS",
        "scope": "SYNTHETIC_TIME_HOMOGENEOUS_CPU_EXACT",
    }
    assert report["task_states"]["FLOW_BASE_LEGAL_CTMC"]["evidence_status"] == "NOT_RUN"
    assert report["claim_state"] == {
        "claim_id": "L3_LEGAL_POTENTIAL_CONSISTENT_XEDITFLOW",
        "claim_status": "NOT_ESTABLISHED",
        "evidence_status": "IN_PROGRESS",
    }
    assert report["time_scope"]["general_time_inhomogeneous_exactness"] == "NOT_RUN"
    assert report["boundaries"]["learned_potential_approximation_error"] == "NOT_RUN"
    assert manifest["output_count"] == 3 and len(manifest["outputs"]) == 3
    assert manifest["cpu_only"] is True and manifest["learned_parameter_count"] == 0
    assert manifest["training_run_count"] == manifest["gpu_work_count"] == 0
    assert manifest["private_payload_read_count"] == manifest["sealed_contact_count"] == 0
    assert event["event_id"] == "A6-CPU-EXACT-001"
    assert event["a6_evidence_status"] == "IN_PROGRESS"
    assert event["flow_base_legal_ctmc_evidence_status"] == "NOT_RUN"
    assert event["l3_claim_status"] == "NOT_ESTABLISHED"
    assert event["a7_unlock"] is False
    with pytest.raises(a6.PublicationError, match="already exists"):
        a6.execute(
            output_directory=output,
            recorded_at="2026-08-13T18:01:00+08:00",
            production=False,
            config_override=frozen,
        )


def test_corrupt_static_numerics_and_status_are_rejected() -> None:
    frozen = config()
    bad = copy.deepcopy(frozen)
    bad["rate_parameters"]["stop_rate"] = 0.0
    with pytest.raises(a6.ConfigError, match="strictly positive"):
        a6.validate_static_config(bad)
    bad = copy.deepcopy(frozen)
    bad["status_contract"]["flow_base_legal_ctmc_evidence_status"] = "PASS"
    with pytest.raises(a6.ConfigError, match="flow_base"):
        a6.validate_static_config(bad)
    bad = a6.candidate_i_config(frozen)
    bad["implementation_binding"]["status"] = "BOUND"
    with pytest.raises(a6.ConfigError, match="partially known"):
        a6.validate_static_config(bad)
