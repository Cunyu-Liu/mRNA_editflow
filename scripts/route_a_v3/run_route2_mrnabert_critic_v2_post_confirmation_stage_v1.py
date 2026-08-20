#!/usr/bin/env python3
"""Run the gated Critic V2 post-confirmation Development pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.route_a_v3 import (  # noqa: E402
    adjudicate_route2_mrnabert_critic_v2_readiness_v1 as readiness_adjudicator,
)
from scripts.route_a_v3 import (  # noqa: E402
    build_route2_mrnabert_critic_v2_guidance_readiness_input_v1 as readiness_builder,
)
from scripts.route_a_v3 import (  # noqa: E402
    prepare_route2_mrnabert_critic_v2_all_development_refit_config_v1 as refit_preparer,
)
from scripts.route_a_v3 import (  # noqa: E402
    prepare_route2_mrnabert_critic_v2_frozen_test_config_v1 as test_preparer,
)
from scripts.route_a_v3 import (  # noqa: E402
    prepare_route2_mrnabert_critic_v2_matched_baseline_loso_configs_v1 as baseline_loso_preparer,
)
from scripts.route_a_v3 import (  # noqa: E402
    prepare_route2_mrnabert_critic_v2_test_preserving_loso_configs_v1 as primary_loso_preparer,
)
from scripts.route_a_v3 import (  # noqa: E402
    run_route2_mrnabert_critic_v2_development_generation_stage_v1 as generation_runner,
)
from scripts.route_a_v3 import (  # noqa: E402
    run_route2_mrnabert_critic_v2_loso_stage_v1 as loso_runner,
)


ROUTE2_ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
CONFIG_ROOT = REPO_ROOT / "configs"
PROTOCOL_PATHS = {
    "readiness": CONFIG_ROOT
    / "route_a_v3_route2_mrnabert_critic_v2_guidance_readiness_protocol_v1.json",
    "control": CONFIG_ROOT
    / "route_a_v3_route2_mrnabert_critic_v2_protocol_v1.json",
    "three_seed": CONFIG_ROOT
    / "route_a_v3_route2_mrnabert_critic_v2_three_seed_protocol_v1.json",
    "frozen_test": CONFIG_ROOT
    / "route_a_v3_route2_mrnabert_critic_v2_frozen_test_protocol_v1.json",
    "refit": CONFIG_ROOT
    / "route_a_v3_route2_mrnabert_critic_v2_all_development_refit_protocol_v1.json",
    "primary_loso": CONFIG_ROOT
    / "route_a_v3_route2_mrnabert_critic_v2_test_preserving_loso_protocol_v1.json",
    "baseline_loso": CONFIG_ROOT
    / "route_a_v3_route2_mrnabert_critic_v2_matched_baseline_loso_protocol_v1.json",
    "loso_aggregation": CONFIG_ROOT
    / "route_a_v3_route2_mrnabert_critic_v2_loso_aggregation_protocol_v1.json",
}
REWARD_POLICY = CONFIG_ROOT / "route_a_v3_route2_mrnabert_guidance_reward_policy_v1.json"
BASELINE_CONFIG = (
    CONFIG_ROOT / "route_a_v3_route2_method_repair_global_scaled_seed20260821_gpu0_v1.json"
)
SELECTED_CONFIRMATION_CONFIG = (
    ROUTE2_ROOT
    / "runs/mrnabert_critic_v2/runtime_configs/"
    "task_study_macro_confirmation_seeds_v1/seed20260823.json"
)
CONTROL_ADJUDICATION = (
    ROUTE2_ROOT
    / "comparisons/mrnabert_critic_v2_task_study_macro_controls_adjudication_v1.json"
)
THREE_SEED_ADJUDICATION = (
    ROUTE2_ROOT
    / "comparisons/mrnabert_critic_v2_task_study_macro_three_seed_adjudication_v1.json"
)
ONLINE_ENCODER_VALIDATION = (
    ROUTE2_ROOT / "runs/mrnabert_online_encoder_validation_v1/validation_summary.json"
)
FLOW_TRAINING_SUMMARY = (
    ROUTE2_ROOT / "runs/base_flow_g0/position_progress_gpu_v2/training_summary.json"
)
FLOW_VALIDATION_SUMMARY = (
    ROUTE2_ROOT
    / "runs/base_flow_g0/position_progress_validation_gpu_v2/validation_summary.json"
)
FLOW_CHECKPOINT = ROUTE2_ROOT / "runs/base_flow_g0/position_progress_gpu_v2/best.pt"
TRAINER = REPO_ROOT / "scripts/route_a_v3/train_route2_delta_predictor_v1.py"
LOSO_STAGE = REPO_ROOT / "scripts/route_a_v3/run_route2_mrnabert_critic_v2_loso_stage_v1.py"
GENERATION_STAGE = (
    REPO_ROOT
    / "scripts/route_a_v3/run_route2_mrnabert_critic_v2_development_generation_stage_v1.py"
)
LOG_ROOT = ROUTE2_ROOT / "logs/mrnabert_critic_v2/post_confirmation_v1"
GPU_CANDIDATES = (0, 1, 2, 3, 4, 5)


class CriticV2PostConfirmationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CriticV2PostConfirmationError(message)


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


def _wait_for_any_gpu(minimum_free_mb: int, poll_seconds: int) -> int:
    while True:
        free_memory = _query_free_memory()
        try:
            return select_gpu(free_memory, minimum_free_mb)
        except CriticV2PostConfirmationError as exc:
            if str(exc) != "no GPU0-5 has enough free memory":
                raise
        print(
            "waiting_for_post_confirmation_gpu="
            f"GPU0-5 minimum_free_mb={minimum_free_mb}",
            flush=True,
        )
        time.sleep(poll_seconds)


def _wait_for_gpu(gpu: int, minimum_free_mb: int, poll_seconds: int) -> None:
    while True:
        free_mb = _query_free_memory()[gpu]
        if free_mb >= minimum_free_mb:
            return
        print(
            f"waiting_for_selected_post_confirmation_gpu={gpu} "
            f"free_mb={free_mb} minimum_free_mb={minimum_free_mb}",
            flush=True,
        )
        time.sleep(poll_seconds)


def ensure_unstarted(paths: Sequence[Path]) -> None:
    for path in paths:
        _require(not path.exists(), f"post-confirmation target already exists: {path}")


def _run_child(command: Sequence[str], log: Path, label: str) -> None:
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    _require(result.returncode == 0, f"post-confirmation child failed: {label}")


def _protocols() -> dict[str, dict[str, Any]]:
    return {
        name: _read_json(path, f"{name} protocol")
        for name, path in PROTOCOL_PATHS.items()
    }


def _target_paths(protocols: Mapping[str, Mapping[str, Any]]) -> list[Path]:
    generation_templates = [
        _read_json(generation_runner.GUIDED_TEMPLATE, "guided template"),
        _read_json(generation_runner.MATCHED_TEMPLATE, "matched template"),
        _read_json(generation_runner.COMPARISON_TEMPLATE, "comparison template"),
    ]
    return [
        LOG_ROOT,
        Path(str(protocols["frozen_test"]["runtime_config"])),
        Path(str(protocols["frozen_test"]["run_directory"])),
        Path(str(protocols["refit"]["runtime_config"])),
        Path(str(protocols["refit"]["run_directory"])),
        Path(str(protocols["primary_loso"]["runtime_config_root"])),
        Path(str(protocols["primary_loso"]["run_root"])),
        Path(str(protocols["baseline_loso"]["runtime_config_root"])),
        Path(str(protocols["baseline_loso"]["run_root"])),
        loso_runner.LOG_ROOT,
        Path(str(protocols["loso_aggregation"]["input_output_root"])),
        Path(str(protocols["loso_aggregation"]["aggregation_output_root"])),
        Path(str(protocols["readiness"]["readiness_input_output"])),
        Path(str(protocols["readiness"]["readiness_adjudication_output"])),
        generation_runner.RUNTIME_ROOT,
        generation_runner.LOG_ROOT,
        *[Path(str(template["output_directory"])) for template in generation_templates],
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-free-mb", type=int, default=4096)
    parser.add_argument("--poll-seconds", type=int, default=900)
    args = parser.parse_args()
    _require(args.minimum_free_mb > 0, "minimum free memory must be positive")
    _require(args.poll_seconds > 0, "poll interval must be positive")

    protocols = _protocols()
    selected_confirmation = _read_json(
        SELECTED_CONFIRMATION_CONFIG, "selected confirmation config"
    )
    control_adjudication = _read_json(CONTROL_ADJUDICATION, "control adjudication")
    three_seed_adjudication = _read_json(
        THREE_SEED_ADJUDICATION, "three-seed adjudication"
    )
    readiness_builder._validate_protocols(protocols)
    test_preparer.build_config(
        selected_confirmation,
        protocols["control"],
        protocols["three_seed"],
        protocols["frozen_test"],
        control_adjudication,
        three_seed_adjudication,
        gpu=0,
    )
    ensure_unstarted(_target_paths(protocols))

    base_config = _read_json(BASELINE_CONFIG, "strongest baseline config")
    reward_policy = _read_json(REWARD_POLICY, "reward policy")
    online_encoder = _read_json(ONLINE_ENCODER_VALIDATION, "online encoder validation")
    flow_training = _read_json(FLOW_TRAINING_SUMMARY, "Flow training summary")
    flow_validation = _read_json(FLOW_VALIDATION_SUMMARY, "Flow validation summary")
    _require(FLOW_CHECKPOINT.is_file(), f"Flow checkpoint is absent: {FLOW_CHECKPOINT}")
    LOG_ROOT.mkdir(parents=True)

    test_gpu = _wait_for_any_gpu(args.minimum_free_mb, args.poll_seconds)
    test_config = test_preparer.build_config(
        selected_confirmation,
        protocols["control"],
        protocols["three_seed"],
        protocols["frozen_test"],
        control_adjudication,
        three_seed_adjudication,
        gpu=test_gpu,
    )
    test_config_path = Path(str(protocols["frozen_test"]["runtime_config"]))
    test_run = Path(str(protocols["frozen_test"]["run_directory"]))
    test_preparer.write_config_once(test_config, test_config_path, test_run)
    _wait_for_gpu(test_gpu, args.minimum_free_mb, args.poll_seconds)
    _run_child(
        [sys.executable, "-u", str(TRAINER), "--config", str(test_config_path)],
        LOG_ROOT / "single_frozen_development_test.log",
        "single frozen Development TEST",
    )

    test_summary = _read_json(test_run / "training_summary.json", "frozen TEST summary")
    refit_gpu = _wait_for_any_gpu(args.minimum_free_mb, args.poll_seconds)
    refit_config = refit_preparer.build_config(
        test_config,
        test_summary,
        protocols["frozen_test"],
        protocols["refit"],
        gpu=refit_gpu,
    )
    refit_config_path = Path(str(protocols["refit"]["runtime_config"]))
    refit_run = Path(str(protocols["refit"]["run_directory"]))
    refit_preparer.write_config_once(refit_config, refit_config_path, refit_run)
    _wait_for_gpu(refit_gpu, args.minimum_free_mb, args.poll_seconds)
    _run_child(
        [sys.executable, "-u", str(TRAINER), "--config", str(refit_config_path)],
        LOG_ROOT / "all_development_refit.log",
        "all-Development refit",
    )

    refit_summary = _read_json(refit_run / "training_summary.json", "refit summary")
    primary_configs = primary_loso_preparer.build_configs(
        refit_config,
        refit_summary,
        protocols["refit"],
        protocols["primary_loso"],
    )
    primary_loso_preparer.write_configs_once(
        primary_configs,
        Path(str(protocols["primary_loso"]["runtime_config_root"])),
    )
    baseline_configs = baseline_loso_preparer.build_configs(
        base_config,
        primary_configs,
        protocols["primary_loso"],
        protocols["baseline_loso"],
    )
    baseline_loso_preparer.write_configs_once(
        baseline_configs,
        Path(str(protocols["baseline_loso"]["runtime_config_root"])),
    )
    _run_child(
        [
            sys.executable,
            "-u",
            str(LOSO_STAGE),
            "--primary-protocol",
            str(PROTOCOL_PATHS["primary_loso"]),
            "--baseline-protocol",
            str(PROTOCOL_PATHS["baseline_loso"]),
            "--aggregation-protocol",
            str(PROTOCOL_PATHS["loso_aggregation"]),
            "--minimum-free-mb",
            str(args.minimum_free_mb),
            "--poll-seconds",
            str(args.poll_seconds),
        ],
        LOG_ROOT / "paired_loso_stage.log",
        "paired Critic V2/matched-baseline LOSO",
    )

    loso_result_root = Path(
        str(protocols["loso_aggregation"]["aggregation_output_root"])
    )
    loso_results = [
        _read_json(
            loso_result_root / f"critic_v2_test_preserving_loso_seed{seed}.json",
            f"LOSO aggregation seed {seed}",
        )
        for seed in (20260822, 20260823, 20260824)
    ]
    refit_checkpoint = refit_run / "delta_predictor_checkpoint.pt"
    readiness_input = readiness_builder.build_input(
        protocols=protocols,
        control_adjudication=control_adjudication,
        three_seed_adjudication=three_seed_adjudication,
        frozen_test_config=test_config,
        frozen_test_summary=test_summary,
        refit_config=refit_config,
        refit_summary=refit_summary,
        refit_checkpoint=refit_checkpoint,
        loso_results=loso_results,
        reward_policy=reward_policy,
        online_encoder_validation=online_encoder,
        flow_training_summary=flow_training,
        flow_validation_summary=flow_validation,
        flow_checkpoint=FLOW_CHECKPOINT,
    )
    readiness_input_path = Path(str(protocols["readiness"]["readiness_input_output"]))
    readiness_builder.write_input_once(readiness_input, readiness_input_path)
    readiness = readiness_adjudicator.adjudicate(readiness_input)
    readiness_output = Path(
        str(protocols["readiness"]["readiness_adjudication_output"])
    )
    readiness_output.parent.mkdir(parents=True, exist_ok=True)
    readiness_output.write_text(
        json.dumps(readiness, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    if readiness.get("guided_unlocked") is not True:
        print(
            json.dumps(
                {
                    "status": "CRITIC_V2_READINESS_TERMINAL_NO_GO_GENERATION_NOT_STARTED",
                    "development_test_executed_once": True,
                    "evaluation_opened": False,
                    "guided_generation_executed": False,
                    "biological_optimization_established": False,
                },
                sort_keys=True,
            )
        )
        return 0

    _run_child(
        [
            sys.executable,
            "-u",
            str(GENERATION_STAGE),
            "--minimum-free-mb",
            str(args.minimum_free_mb),
            "--poll-seconds",
            str(args.poll_seconds),
        ],
        LOG_ROOT / "development_generation_stage.log",
        "Critic V2 Development generation",
    )
    print(
        json.dumps(
            {
                "status": "CRITIC_V2_POST_CONFIRMATION_DEVELOPMENT_PIPELINE_COMPLETE",
                "development_test_executed_once": True,
                "evaluation_opened": False,
                "guided_generation_executed": True,
                "generated_candidates_grant_canonical_credit": False,
                "biological_optimization_established": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
