from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = (
    ROOT
    / "scripts/route_a_v3/run_route2_mrnabert_critic_v2_post_confirmation_stage_v1.py"
)
WATCHER = (
    ROOT
    / "scripts/route_a_v3/schedule_route2_mrnabert_critic_v2_post_confirmation_v1.sh"
)
HISTORICAL = (
    ROOT / "scripts/route_a_v3/schedule_route2_mrnabert_postselection_controls_v1.sh"
)
TEST_GATE_FIXTURE = (
    ROOT
    / "tests/route_a_v3/test_prepare_route2_mrnabert_critic_v2_frozen_test_config_v1.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_selects_most_free_eligible_gpu_with_stable_tie_break() -> None:
    runner = _load(RUNNER, "critic_v2_post_confirmation_gpu_test")
    memory = {0: 3000, 1: 7000, 2: 7000, 3: 6000, 4: 4096, 5: 1000}
    assert runner.select_gpu(memory, 4096) == 1
    with pytest.raises(
        runner.CriticV2PostConfirmationError, match="enough free memory"
    ):
        runner.select_gpu({gpu: 4095 for gpu in runner.GPU_CANDIDATES}, 4096)


def test_production_gate_rejects_no_go_before_stage_targets() -> None:
    runner = _load(RUNNER, "critic_v2_post_confirmation_gate_test")
    fixture = _load(TEST_GATE_FIXTURE, "critic_v2_post_confirmation_gate_fixture")
    inputs = list(fixture._valid_inputs())
    runner.test_preparer.build_config(*inputs, gpu=0)
    inputs[5]["supports_single_frozen_development_test"] = False
    with pytest.raises(
        runner.test_preparer.CriticV2FrozenTestPreparationError,
        match="TEST gate failed",
    ):
        runner.test_preparer.build_config(*inputs, gpu=0)


def test_target_inventory_covers_every_future_stage() -> None:
    runner = _load(RUNNER, "critic_v2_post_confirmation_targets_test")
    protocols = runner._protocols()
    targets = runner._target_paths(protocols)
    assert len(targets) == len(set(targets)) == 19
    expected = {
        Path(protocols["frozen_test"]["runtime_config"]),
        Path(protocols["frozen_test"]["run_directory"]),
        Path(protocols["refit"]["runtime_config"]),
        Path(protocols["refit"]["run_directory"]),
        Path(protocols["primary_loso"]["runtime_config_root"]),
        Path(protocols["baseline_loso"]["runtime_config_root"]),
        Path(protocols["loso_aggregation"]["aggregation_output_root"]),
        Path(protocols["readiness"]["readiness_input_output"]),
        Path(protocols["readiness"]["readiness_adjudication_output"]),
        runner.generation_runner.RUNTIME_ROOT,
        runner.generation_runner.LOG_ROOT,
    }
    assert expected <= set(targets)


def test_unstarted_guard_refuses_any_existing_target(tmp_path: Path) -> None:
    runner = _load(RUNNER, "critic_v2_post_confirmation_unstarted_test")
    targets = [tmp_path / "test", tmp_path / "refit", tmp_path / "loso"]
    runner.ensure_unstarted(targets)
    targets[1].mkdir()
    with pytest.raises(
        runner.CriticV2PostConfirmationError, match="target already exists"
    ):
        runner.ensure_unstarted(targets)


def test_source_orders_gate_test_refit_loso_readiness_and_generation() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    main = source[source.index("def main") :]
    preview_gate = main.index("test_preparer.build_config")
    unstarted = main.index("ensure_unstarted")
    first_write = main.index("LOG_ROOT.mkdir")
    test_write = main.index("test_preparer.write_config_once")
    test_run = main.index('"single frozen Development TEST"')
    refit_build = main.index("refit_preparer.build_config")
    refit_run = main.index('"all-Development refit"')
    primary_build = main.index("primary_loso_preparer.build_configs")
    baseline_build = main.index("baseline_loso_preparer.build_configs")
    loso_run = main.index('"paired Critic V2/matched-baseline LOSO"')
    readiness_build = main.index("readiness_builder.build_input")
    readiness_adjudicate = main.index("readiness_adjudicator.adjudicate")
    readiness_gate = main.index('if readiness.get("guided_unlocked") is not True')
    generation_run = main.index('"Critic V2 Development generation"')
    assert (
        preview_gate
        < unstarted
        < first_write
        < test_write
        < test_run
        < refit_build
        < refit_run
        < primary_build
        < baseline_build
        < loso_run
        < readiness_build
        < readiness_adjudicate
        < readiness_gate
        < generation_run
    )
    assert '"evaluation_opened": False' in main
    assert '"generated_candidates_grant_canonical_credit": False' in main
    assert '"biological_optimization_established": False' in main


def test_watcher_is_900_second_pass_only_and_shell_valid() -> None:
    source = WATCHER.read_text(encoding="utf-8")
    assert 'POLL_SECONDS="${POLL_SECONDS:-900}"' in source
    assert source.index("while [[ ! -f") < source.index("supports_test=")
    assert source.index("supports_test=") < source.index('if [[ "${supports_test}" != "true" ]]')
    assert source.index('exit 0') < source.index(
        "run_route2_mrnabert_critic_v2_post_confirmation_stage_v1.py"
    )
    result = subprocess.run(
        ["bash", "-n", str(WATCHER)], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_historical_v1_scheduler_refuses_before_any_work() -> None:
    source = HISTORICAL.read_text(encoding="utf-8")
    retired = source.index("RETIRED: historical V1 postselection scheduler")
    refusal = source.index("exit 1", retired)
    assert refusal < source.index("REPO_ROOT=")
