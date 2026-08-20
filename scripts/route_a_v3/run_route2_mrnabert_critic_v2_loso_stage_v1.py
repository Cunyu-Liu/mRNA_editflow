#!/usr/bin/env python3
"""Run the frozen six-GPU Critic V2/matched-baseline LOSO stage."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_loso_schedule import (  # noqa: E402
    FINAL_SEEDS,
    HOLDOUT_STUDIES,
    PHYSICAL_GPU_INDICES,
    loso_assignments,
)
from scripts.route_a_v3.build_route2_mrnabert_critic_v2_loso_aggregation_inputs_v1 import (  # noqa: E402
    build_inputs,
    validate_config_pairs,
    write_inputs_once,
)


TRAINER = REPO_ROOT / "scripts/route_a_v3/train_route2_delta_predictor_v1.py"
AGGREGATOR = REPO_ROOT / "scripts/route_a_v3/aggregate_route2_loso_v1.py"
LOG_ROOT = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/logs/"
    "mrnabert_critic_v2/test_preserving_loso_v1"
)


class CriticV2LosoStageError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CriticV2LosoStageError(message)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{label} root is not an object: {path}")
    return value


def _read_configs(root: Path, label: str) -> list[dict[str, Any]]:
    _require(root.is_dir(), f"{label} config root is absent: {root}")
    paths = sorted(root.glob("*.json"))
    _require(len(paths) == 21, f"{label} config root must contain exactly 21 JSON files")
    return [_read_json(path, f"{label} config") for path in paths]


def plan_jobs(
    primary_configs: Sequence[Mapping[str, Any]],
    baseline_configs: Sequence[Mapping[str, Any]],
    primary_protocol: Mapping[str, Any],
    baseline_protocol: Mapping[str, Any],
    aggregation_protocol: Mapping[str, Any],
    primary_config_root: Path,
    baseline_config_root: Path,
) -> dict[int, list[dict[str, Any]]]:
    primary_by_key, baseline_by_key = validate_config_pairs(
        primary_configs,
        baseline_configs,
        primary_protocol,
        baseline_protocol,
        aggregation_protocol,
    )
    jobs = {gpu: [] for gpu in PHYSICAL_GPU_INDICES}
    for study, seed, gpu in loso_assignments():
        primary = primary_by_key[(study, seed)]
        baseline = baseline_by_key[(study, seed)]
        primary_path = primary_config_root / f"{primary['baseline_id']}.json"
        baseline_path = baseline_config_root / f"{baseline['baseline_id']}.json"
        _require(primary_path.is_file(), f"primary config path is absent: {primary_path}")
        _require(baseline_path.is_file(), f"baseline config path is absent: {baseline_path}")
        jobs[gpu].append(
            {
                "study": study,
                "seed": seed,
                "gpu": gpu,
                "primary_config": primary_path,
                "baseline_config": baseline_path,
                "primary_output": Path(str(primary["output_directory"])),
                "baseline_output": Path(str(baseline["output_directory"])),
            }
        )
    _require(sum(len(queue) for queue in jobs.values()) == 21, "LOSO job count differs")
    return jobs


def ensure_unstarted(
    jobs: Mapping[int, Sequence[Mapping[str, Any]]],
    log_root: Path,
    input_root: Path,
    result_root: Path,
) -> None:
    _require(not log_root.exists(), f"LOSO log root already exists: {log_root}")
    _require(not input_root.exists(), f"LOSO input root already exists: {input_root}")
    _require(not result_root.exists(), f"LOSO result root already exists: {result_root}")
    for queue in jobs.values():
        for job in queue:
            _require(
                not Path(job["primary_output"]).exists(),
                f"primary LOSO output already exists: {job['primary_output']}",
            )
            _require(
                not Path(job["baseline_output"]).exists(),
                f"baseline LOSO output already exists: {job['baseline_output']}",
            )


def _free_memory_mb(gpu: int) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
            "-i",
            str(gpu),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    _require(result.returncode == 0, f"nvidia-smi failed for GPU {gpu}")
    return int(result.stdout.strip())


def _wait_for_memory(gpu: int, minimum_free_mb: int, poll_seconds: int) -> None:
    while True:
        free_mb = _free_memory_mb(gpu)
        if free_mb >= minimum_free_mb:
            return
        print(
            f"waiting_for_loso_gpu={gpu} free_mb={free_mb} minimum_free_mb={minimum_free_mb}",
            flush=True,
        )
        time.sleep(poll_seconds)


def _run_training(config: Path, log: Path) -> None:
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            [sys.executable, "-u", str(TRAINER), "--config", str(config)],
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    _require(result.returncode == 0, f"LOSO training failed: {config}")


def _run_gpu_queue(
    gpu: int,
    jobs: Sequence[Mapping[str, Any]],
    *,
    minimum_free_mb: int,
    poll_seconds: int,
    log_root: Path,
) -> None:
    for job in jobs:
        study = str(job["study"])
        seed = int(job["seed"])
        _wait_for_memory(gpu, minimum_free_mb, poll_seconds)
        print(f"starting_loso_primary study={study} seed={seed} gpu={gpu}", flush=True)
        _run_training(
            Path(job["primary_config"]),
            log_root / f"primary_{study.lower()}_seed{seed}_gpu{gpu}.log",
        )
        _wait_for_memory(gpu, minimum_free_mb, poll_seconds)
        print(f"starting_loso_baseline study={study} seed={seed} gpu={gpu}", flush=True)
        _run_training(
            Path(job["baseline_config"]),
            log_root / f"baseline_{study.lower()}_seed{seed}_gpu{gpu}.log",
        )
        print(f"finished_loso_pair study={study} seed={seed} gpu={gpu}", flush=True)


def _aggregate_three(
    payload_paths: Sequence[Path], result_root: Path
) -> list[Path]:
    result_root.mkdir(parents=True)
    outputs = []
    for seed, input_path in zip(FINAL_SEEDS, payload_paths):
        output = result_root / f"critic_v2_test_preserving_loso_seed{seed}.json"
        result = subprocess.run(
            [
                sys.executable,
                str(AGGREGATOR),
                "--input",
                str(input_path),
                "--output",
                str(output),
            ],
            cwd=REPO_ROOT,
            check=False,
            text=True,
        )
        _require(result.returncode == 0, f"LOSO aggregation failed: seed {seed}")
        outputs.append(output)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-protocol", type=Path, required=True)
    parser.add_argument("--baseline-protocol", type=Path, required=True)
    parser.add_argument("--aggregation-protocol", type=Path, required=True)
    parser.add_argument("--minimum-free-mb", type=int, default=4096)
    parser.add_argument("--poll-seconds", type=int, default=900)
    args = parser.parse_args()
    _require(args.minimum_free_mb > 0, "minimum free memory must be positive")
    _require(args.poll_seconds > 0, "poll interval must be positive")

    primary_protocol = _read_json(args.primary_protocol, "primary protocol")
    baseline_protocol = _read_json(args.baseline_protocol, "baseline protocol")
    aggregation_protocol = _read_json(
        args.aggregation_protocol, "aggregation protocol"
    )
    primary_config_root = Path(str(primary_protocol["runtime_config_root"]))
    baseline_config_root = Path(str(baseline_protocol["runtime_config_root"]))
    primary_configs = _read_configs(primary_config_root, "primary")
    baseline_configs = _read_configs(baseline_config_root, "baseline")
    jobs = plan_jobs(
        primary_configs,
        baseline_configs,
        primary_protocol,
        baseline_protocol,
        aggregation_protocol,
        primary_config_root,
        baseline_config_root,
    )
    input_root = Path(str(aggregation_protocol["input_output_root"]))
    result_root = Path(str(aggregation_protocol["aggregation_output_root"]))
    ensure_unstarted(jobs, LOG_ROOT, input_root, result_root)
    LOG_ROOT.mkdir(parents=True)

    with ThreadPoolExecutor(max_workers=len(PHYSICAL_GPU_INDICES)) as executor:
        futures = [
            executor.submit(
                _run_gpu_queue,
                gpu,
                jobs[gpu],
                minimum_free_mb=args.minimum_free_mb,
                poll_seconds=args.poll_seconds,
                log_root=LOG_ROOT,
            )
            for gpu in PHYSICAL_GPU_INDICES
        ]
        for future in futures:
            future.result()

    payloads = build_inputs(
        primary_configs,
        baseline_configs,
        primary_protocol,
        baseline_protocol,
        aggregation_protocol,
    )
    input_paths = write_inputs_once(payloads, input_root, result_root)
    result_paths = _aggregate_three(input_paths, result_root)
    print(
        json.dumps(
            {
                "status": "CRITIC_V2_PAIRED_LOSO_AND_THREE_AGGREGATIONS_COMPLETE",
                "pair_count": 21,
                "training_run_count": 42,
                "seeds": list(FINAL_SEEDS),
                "studies": list(HOLDOUT_STUDIES),
                "physical_gpu_indices": list(PHYSICAL_GPU_INDICES),
                "aggregation_results": [str(path) for path in result_paths],
                "development_test_opened": False,
                "evaluation_opened": False,
                "guided_generation_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
