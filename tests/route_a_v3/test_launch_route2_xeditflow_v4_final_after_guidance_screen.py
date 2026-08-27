from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import scripts.route_a_v3.launch_route2_xeditflow_v4_final_after_guidance_screen as launcher
from scripts.route_a_v3.launch_route2_xeditflow_v4_final_after_guidance_screen import build_schedule
from scripts.route_a_v3.run_route2_xeditflow_v4_final_scheduler import run


ROOT = Path(__file__).resolve().parents[2]


def _prepared_manifest(tmp_path: Path) -> tuple[dict, Path]:
    helpers = runpy.run_path(
        str(
            ROOT
            / "tests/route_a_v3/"
            "test_prepare_route2_xeditflow_final_generation_configs_v4.py"
        )
    )
    payload = helpers["_payload"]()
    config_root = tmp_path / "configs"
    payload["runtime_config_root"] = str(config_root)
    helpers["write_final_generation_configs_v4"](payload, config_root)
    return json.loads((config_root / "manifest.json").read_text()), config_root


def test_v4_final_launcher_builds_exact_three_seed_job_graph(tmp_path: Path) -> None:
    manifest, config_root = _prepared_manifest(tmp_path)
    schedule = build_schedule(
        manifest,
        config_root=config_root,
        log_root=tmp_path / "logs",
        failure_root=tmp_path / "failures",
        runtime_manifest=tmp_path / "runtime.json",
        current_head="a" * 40,
        experiment_head="b" * 40,
        guidance_runner_head="c" * 40,
        diagnostic_peak_plus_two_gib_mib=30_000,
        free_memory_mib={gpu: 40_000 for gpu in range(6)},
    )
    assert [row["queue_key"] for row in schedule["prerequisite_queues"]] == [
        "value_seed_20260913",
        "value_seed_20260914",
        "strongest_timing",
    ]
    assert [row["queue_key"] for row in schedule["seed_chains"]] == [
        "seed_20260912",
        "seed_20260913",
        "seed_20260914",
    ]
    assert all(len(row["jobs"]) == 29 for row in schedule["seed_chains"])
    assert len(schedule["finalization_jobs"]) == 2
    all_jobs = [
        job
        for row in schedule["prerequisite_queues"] + schedule["seed_chains"]
        for job in row["jobs"]
    ] + schedule["finalization_jobs"]
    assert len(all_jobs) == 98
    assert len({job["job_key"] for job in all_jobs}) == 98
    assert all(
        not set(job["physical_gpu_indices"]) - set(range(6)) for job in all_jobs
    )
    seed_12 = schedule["seed_chains"][0]["jobs"]
    assert seed_12[0]["job_key"].endswith("strongest_adapter")
    assert seed_12[-1]["job_key"].endswith("final_evidence")
    assert schedule["finalization_jobs"][-1]["job_key"] == (
        "adjudicate_final_comparison"
    )
    assert schedule["development_test_outcomes_accessed_after_atomic_test"] is False
    assert schedule["new_final_evaluation_outcome_reads"] == 0
    assert schedule["free_memory_gate_applied"] is False
    assert schedule["diagnostic_peak_plus_two_gib_mib"] == 30_000


def test_v4_final_launcher_uses_own_repo_and_does_not_memory_gate() -> None:
    assert launcher.WORKTREE == Path(launcher.__file__).resolve().parents[2]
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    assert '"free_memory_gate_applied": False' in source
    assert "all(free_memory[gpu]" not in source


def _job(tmp_path: Path, key: str, *, succeed: bool = True) -> dict:
    success = tmp_path / f"{key}.success"
    failure = tmp_path / f"{key}.failure"
    if succeed:
        code = (
            "from pathlib import Path; "
            f"Path({str(success)!r}).write_text('ok', encoding='utf-8')"
        )
    else:
        code = "raise SystemExit(7)"
    return {
        "job_key": key,
        "command": [sys.executable, "-c", code],
        "physical_gpu_indices": [],
        "success_path": str(success),
        "failure_path": str(failure),
        "log_path": str(tmp_path / f"{key}.log"),
    }


def _scheduler_fixture(tmp_path: Path, *, fail_seed: bool = False) -> dict:
    return {
        "git_head": "a" * 40,
        "experiment_head": "b" * 40,
        "guidance_runner_head": "c" * 40,
        "worktree": str(tmp_path),
        "runtime_manifest": str(tmp_path / "runtime.json"),
        "prerequisite_queues": [
            {"queue_key": "prerequisite", "jobs": [_job(tmp_path, "pre")]}
        ],
        "seed_chains": [
            {
                "queue_key": f"seed_{seed}",
                "jobs": [
                    _job(
                        tmp_path,
                        f"seed_{seed}",
                        succeed=not (fail_seed and seed == 20260913),
                    )
                ],
            }
            for seed in (20260912, 20260913, 20260914)
        ],
        "finalization_jobs": [
            _job(tmp_path, "compose"),
            _job(tmp_path, "adjudicate"),
        ],
    }


def test_v4_final_scheduler_closes_only_after_all_three_seed_chains(
    tmp_path: Path,
) -> None:
    schedule = _scheduler_fixture(tmp_path)
    run(schedule)
    runtime = json.loads((tmp_path / "runtime.json").read_text())
    assert runtime["status"] == "XEDITFLOW_V4_FINAL_COMPARISON_TERMINAL"
    assert all(
        row["status"] == "TERMINAL_COMPLETE" for row in runtime["jobs"].values()
    )
    assert runtime["active_performance_output_read"] is False


def test_v4_final_scheduler_preserves_failure_and_does_not_adjudicate(
    tmp_path: Path,
) -> None:
    schedule = _scheduler_fixture(tmp_path, fail_seed=True)
    run(schedule)
    runtime = json.loads((tmp_path / "runtime.json").read_text())
    assert runtime["status"] == "XEDITFLOW_V4_FINAL_COMPARISON_TECHNICAL_FAILURE"
    assert runtime["jobs"]["seed_20260913"]["status"] == "TERMINAL_FAILURE"
    assert runtime["jobs"]["compose"]["status"] == "NOT_RUN_SEED_CHAIN_FAILURE"
    assert runtime["jobs"]["adjudicate"]["status"] == "NOT_RUN_SEED_CHAIN_FAILURE"
    assert Path(runtime["jobs"]["seed_20260913"]["failure_path"]).is_file()
