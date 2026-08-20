#!/usr/bin/env python3
"""Run the gated Critic V2 Development generation and comparison stages."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.route_a_v3 import run_route2_guided_xeditflow_v1 as guided_runner  # noqa: E402
from scripts.route_a_v3 import (  # noqa: E402
    run_route2_mrnabert_generation_comparison_suite_v1 as comparison_runner,
)
from scripts.route_a_v3 import (  # noqa: E402
    run_route2_mrnabert_matched_search_suite_v1 as matched_runner,
)


GUIDED_TEMPLATE = (
    REPO_ROOT
    / "configs/route_a_v3_route2_mrnabert_critic_v2_guided_xeditflow_development_gpu0_v1.json"
)
MATCHED_TEMPLATE = (
    REPO_ROOT
    / "configs/route_a_v3_route2_mrnabert_critic_v2_matched_search_development_gpu0_v1.json"
)
COMPARISON_TEMPLATE = (
    REPO_ROOT
    / "configs/route_a_v3_route2_mrnabert_critic_v2_generation_comparison_development_gpu0_v1.json"
)
ROUTE2_ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
RUNTIME_ROOT = (
    ROUTE2_ROOT
    / "runs/mrnabert_critic_v2/runtime_configs/development_generation_v1"
)
LOG_ROOT = ROUTE2_ROOT / "logs/mrnabert_critic_v2/development_generation_v1"
GPU_CANDIDATES = (0, 1, 2, 3, 4, 5)


class CriticV2GenerationStageError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CriticV2GenerationStageError(message)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{label} root is not an object: {path}")
    return value


def select_gpu(free_memory_mb: Mapping[int, int], minimum_free_mb: int) -> int:
    _require(minimum_free_mb > 0, "minimum free memory must be positive")
    _require(
        set(free_memory_mb) == set(GPU_CANDIDATES),
        "free-memory inventory must contain physical GPU0-5",
    )
    eligible = [
        (int(free_memory_mb[gpu]), gpu)
        for gpu in GPU_CANDIDATES
        if int(free_memory_mb[gpu]) >= minimum_free_mb
    ]
    _require(bool(eligible), "no GPU0-5 has enough free memory")
    return max(eligible, key=lambda item: (item[0], -item[1]))[1]


def _query_free_memory() -> dict[int, int]:
    values = {}
    for gpu in GPU_CANDIDATES:
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
        values[gpu] = int(result.stdout.strip())
    return values


def _wait_for_selected_gpu(gpu: int, minimum_free_mb: int, poll_seconds: int) -> None:
    while True:
        free_mb = _query_free_memory()[gpu]
        if free_mb >= minimum_free_mb:
            return
        print(
            f"waiting_for_generation_gpu={gpu} free_mb={free_mb} minimum_free_mb={minimum_free_mb}",
            flush=True,
        )
        time.sleep(poll_seconds)


def build_runtime_payloads(
    guided: Mapping[str, Any],
    matched: Mapping[str, Any],
    comparison: Mapping[str, Any],
    gpu: int,
) -> dict[str, dict[str, Any]]:
    _require(gpu in GPU_CANDIDATES, "selected physical GPU is outside GPU0-5")
    payloads = {
        "guided": dict(guided),
        "matched": dict(matched),
        "comparison": dict(comparison),
    }
    for payload in payloads.values():
        payload["device"] = f"cuda:{gpu}"
        payload["physical_gpu_index"] = gpu
    guided_runner.validate_guided_config(payloads["guided"])
    matched_runner.validate_config_boundary(payloads["matched"])
    comparison_runner.validate_config_boundary(payloads["comparison"])
    _require(
        payloads["comparison"]["guided_method_id"]
        == guided_runner.GUIDED_METHOD_ID,
        "guided method identity differs across runtime configs",
    )
    return payloads


def write_runtime_payloads_once(
    payloads: Mapping[str, Mapping[str, Any]], runtime_root: Path
) -> dict[str, Path]:
    _require(not runtime_root.exists(), f"generation runtime root already exists: {runtime_root}")
    runtime_root.mkdir(parents=True)
    paths = {}
    for name in ("guided", "matched", "comparison"):
        path = runtime_root / f"{name}.json"
        path.write_text(
            json.dumps(payloads[name], indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        paths[name] = path
    return paths


def _run_child(script: Path, config: Path, log: Path) -> None:
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            [sys.executable, "-u", str(script), "--config", str(config)],
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    _require(result.returncode == 0, f"Development generation child failed: {script.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-free-mb", type=int, default=4096)
    parser.add_argument("--poll-seconds", type=int, default=900)
    args = parser.parse_args()
    _require(args.minimum_free_mb > 0, "minimum free memory must be positive")
    _require(args.poll_seconds > 0, "poll interval must be positive")

    guided = _read_json(GUIDED_TEMPLATE, "guided template")
    matched = _read_json(MATCHED_TEMPLATE, "matched template")
    comparison = _read_json(COMPARISON_TEMPLATE, "comparison template")
    guided_runner.validate_guided_config(guided)
    readiness_input = _read_json(
        Path(str(guided["readiness_input_path"])), "V2 readiness input"
    )
    readiness_adjudication = _read_json(
        Path(str(guided["readiness_adjudication_path"])),
        "V2 readiness adjudication",
    )
    guided_runner.validate_readiness(readiness_input, readiness_adjudication, guided)

    _require(not LOG_ROOT.exists(), f"generation log root already exists: {LOG_ROOT}")
    for template in (guided, matched, comparison):
        _require(
            not Path(str(template["output_directory"])).exists(),
            f"Development generation output already exists: {template['output_directory']}",
        )
    free_memory = _query_free_memory()
    gpu = select_gpu(free_memory, args.minimum_free_mb)
    payloads = build_runtime_payloads(guided, matched, comparison, gpu)
    runtime_paths = write_runtime_payloads_once(payloads, RUNTIME_ROOT)
    LOG_ROOT.mkdir(parents=True)

    stages = (
        (
            "guided",
            REPO_ROOT / "scripts/route_a_v3/run_route2_guided_xeditflow_v1.py",
        ),
        (
            "matched",
            REPO_ROOT
            / "scripts/route_a_v3/run_route2_mrnabert_matched_search_suite_v1.py",
        ),
        (
            "comparison",
            REPO_ROOT
            / "scripts/route_a_v3/run_route2_mrnabert_generation_comparison_suite_v1.py",
        ),
    )
    for name, script in stages:
        _wait_for_selected_gpu(gpu, args.minimum_free_mb, args.poll_seconds)
        print(f"starting_critic_v2_development_stage={name} gpu={gpu}", flush=True)
        _run_child(script, runtime_paths[name], LOG_ROOT / f"{name}.log")
        print(f"finished_critic_v2_development_stage={name} gpu={gpu}", flush=True)

    print(
        json.dumps(
            {
                "status": "CRITIC_V2_DEVELOPMENT_GENERATION_AND_COMPARISON_COMPLETE",
                "physical_gpu_index": gpu,
                "runtime_configs": {
                    name: str(path) for name, path in runtime_paths.items()
                },
                "development_test_opened": False,
                "evaluation_opened": False,
                "generated_candidates_grant_canonical_credit": False,
                "biological_optimization_established": False,
                "scientific_claim_status": "INDEPENDENT_EVALUATOR_DEVELOPMENT_ONLY_NOT_EXTERNAL_OR_MEASURED_SUCCESS",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
