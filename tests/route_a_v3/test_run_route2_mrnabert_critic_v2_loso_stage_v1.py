from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/route_a_v3/run_route2_mrnabert_critic_v2_loso_stage_v1.py"
FIXTURE = (
    ROOT
    / "tests/route_a_v3/test_build_route2_mrnabert_critic_v2_loso_aggregation_inputs_v1.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _planned(tmp_path: Path):
    fixture = _load(FIXTURE, "critic_v2_loso_stage_fixture")
    (
        _,
        primary_configs,
        baseline_configs,
        primary_protocol,
        baseline_protocol,
        aggregation_protocol,
    ) = fixture._configs(tmp_path)
    primary_root = tmp_path / "primary-configs"
    baseline_root = tmp_path / "baseline-configs"
    primary_root.mkdir()
    baseline_root.mkdir()
    for config in primary_configs:
        (primary_root / f"{config['baseline_id']}.json").write_text(
            json.dumps(config) + "\n", encoding="utf-8"
        )
    for config in baseline_configs:
        (baseline_root / f"{config['baseline_id']}.json").write_text(
            json.dumps(config) + "\n", encoding="utf-8"
        )
    runner = _load(RUNNER, "critic_v2_loso_stage_runner")
    jobs = runner.plan_jobs(
        primary_configs,
        baseline_configs,
        primary_protocol,
        baseline_protocol,
        aggregation_protocol,
        primary_root,
        baseline_root,
    )
    return runner, jobs


def test_plans_exact_six_gpu_primary_then_baseline_pairs(tmp_path: Path) -> None:
    runner, jobs = _planned(tmp_path)
    assert set(jobs) == set(runner.PHYSICAL_GPU_INDICES)
    assert sum(len(queue) for queue in jobs.values()) == 21
    keys = set()
    for gpu, queue in jobs.items():
        for job in queue:
            assert job["gpu"] == gpu
            key = (job["study"], job["seed"])
            assert key not in keys
            keys.add(key)
            assert job["primary_config"].name.startswith("mrnabert_critic_v2_loso_")
            assert job["baseline_config"].name.startswith("global_scaled_critic_v2_loso_")
    assert keys == {
        (study, seed)
        for study in runner.HOLDOUT_STUDIES
        for seed in runner.FINAL_SEEDS
    }


def test_unstarted_preflight_rejects_any_existing_stage_root(tmp_path: Path) -> None:
    runner, jobs = _planned(tmp_path)
    clean_jobs = deepcopy(jobs)
    for gpu, queue in clean_jobs.items():
        for index, job in enumerate(queue):
            job["primary_output"] = tmp_path / "future-primary" / str(gpu) / str(index)
            job["baseline_output"] = tmp_path / "future-baseline" / str(gpu) / str(index)
    log_root = tmp_path / "logs"
    input_root = tmp_path / "inputs"
    result_root = tmp_path / "results"
    runner.ensure_unstarted(clean_jobs, log_root, input_root, result_root)

    log_root.mkdir()
    with pytest.raises(runner.CriticV2LosoStageError, match="log root already exists"):
        runner.ensure_unstarted(clean_jobs, log_root, input_root, result_root)


def test_source_orders_all_training_before_input_build_and_aggregation() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    worker = source[source.index("def _run_gpu_queue") : source.index("def _aggregate_three")]
    assert worker.index('Path(job["primary_config"])') < worker.index(
        'Path(job["baseline_config"])'
    )
    main = source[source.index("def main") :]
    assert main.index("future.result()") < main.index("payloads = build_inputs")
    assert main.index("payloads = build_inputs") < main.index("_aggregate_three")
    assert '"development_test_opened": False' in main
    assert '"evaluation_opened": False' in main
