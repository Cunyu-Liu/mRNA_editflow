#!/usr/bin/env python3
"""Launch SetFlow V4 confirmation only from the terminal V4.0.3 recovery."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

WORKTREE = Path(__file__).resolve().parents[2]
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from scripts.route_a_v3.authorize_route2_xeditsetflow_v403_recovered_confirmation import (
    CONFIRMATION_SEEDS,
    SCREEN_EXPERIMENT_HEAD,
    TRAINING_HEAD,
    VALIDATION_HEAD,
    build_recovered_confirmation_authorization_v403,
    require_recovery_config_derivation_v403,
    require_recovery_terminal_v403,
    require_science_protocol_unchanged_v403,
)


PYTHON = Path("/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10")
ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
PROTOCOL = (
    WORKTREE
    / "configs/route_a_v3_route2_xeditsetflow_v403_recovered_confirmation_protocol_v1.json"
)
BASE_CONFIRMATION_PROTOCOL = (
    WORKTREE
    / "configs/route_a_v3_route2_xeditsetflow_v4_confirmation_protocol_v1.json"
)
SCREEN_CONFIG = (
    WORKTREE / "configs/route_a_v3_route2_xeditsetflow_v4_screen_v1.json"
)
PREPARE = (
    WORKTREE
    / "scripts/route_a_v3/prepare_route2_xeditsetflow_v4_confirmation_configs.py"
)
AUTHORIZE = (
    WORKTREE
    / "scripts/route_a_v3/authorize_route2_xeditsetflow_v403_recovered_confirmation.py"
)
TRAINER = WORKTREE / "scripts/route_a_v3/train_route2_xeditsetflow_v4.py"
SCHEDULER = (
    WORKTREE
    / "scripts/route_a_v3/run_route2_xedit_v4_confirmation_training_scheduler.py"
)


class XEditSetFlowV403ConfirmationLaunchError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditSetFlowV403ConfirmationLaunchError(message)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def write_new_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"artifact already exists: {path}")
    partial = path.with_suffix(path.suffix + ".partial")
    require(not partial.exists(), f"partial artifact already exists: {partial}")
    partial.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def command(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(arguments),
        cwd=WORKTREE,
        text=True,
        capture_output=True,
        check=True,
    )


def gpu_diagnostics() -> dict[int, dict[str, Any]]:
    result = command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    values: dict[int, dict[str, Any]] = {}
    for line in result.stdout.splitlines():
        index, name, free, total = (
            part.strip() for part in line.split(",", maxsplit=3)
        )
        values[int(index)] = {
            "name": name,
            "free_memory_mib": int(free),
            "total_memory_mib": int(total),
        }
    return values


def cuda_bf16_probe(physical_gpu_index: int) -> dict[str, Any]:
    source = """
import json
import sys
import torch

index = int(sys.argv[1])
if not torch.cuda.is_available():
    raise RuntimeError("CUDA_UNAVAILABLE_CPU_FALLBACK_FORBIDDEN")
if index < 0 or index >= torch.cuda.device_count():
    raise RuntimeError("PHYSICAL_GPU_INDEX_UNAVAILABLE")
torch.cuda.set_device(index)
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("BF16_UNAVAILABLE_ON_SELECTED_GPU")
tensor = torch.ones((8,), device=f"cuda:{index}", dtype=torch.bfloat16)
if tensor.device.type != "cuda" or tensor.dtype != torch.bfloat16:
    raise RuntimeError("CUDA_BF16_PROBE_SILENT_CPU_FALLBACK")
print(json.dumps({
    "physical_gpu_index": index,
    "device_name": torch.cuda.get_device_name(index),
    "device_type": tensor.device.type,
    "dtype": str(tensor.dtype).replace("torch.", "").upper(),
    "cuda_available": True,
    "bf16_supported": True,
    "cpu_fallback_used": False,
}))
"""
    result = command([str(PYTHON), "-c", source, str(physical_gpu_index)])
    payload = json.loads(result.stdout)
    require(
        payload.get("physical_gpu_index") == physical_gpu_index
        and payload.get("device_type") == "cuda"
        and payload.get("dtype") == "BFLOAT16"
        and payload.get("cuda_available") is True
        and payload.get("bf16_supported") is True
        and payload.get("cpu_fallback_used") is False,
        f"GPU {physical_gpu_index} failed CUDA/BF16 identity probe",
    )
    return payload


def validate_manifest_v403(
    manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[int, Path]:
    require(
        manifest.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_confirmation_config_manifest.v1"
        and manifest.get("status")
        == "THREE_CONFIRMATION_CONFIGS_PREPARED_NOT_STARTED"
        and manifest.get("selected_model") == "v4_full"
        and manifest.get("required_seeds") == list(CONFIRMATION_SEEDS)
        and len(manifest.get("config_paths", [])) == 3,
        "SetFlow V4.0.3 recovered confirmation manifest changed",
    )
    require(
        int(manifest.get("development_test_outcome_reads", -1)) == 0
        and int(manifest.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "SetFlow V4.0.3 recovered confirmation manifest reports a protected read",
    )
    configs: dict[int, Path] = {}
    provenance = protocol["validation_recovery_provenance"]
    for path_text in manifest["config_paths"]:
        path = Path(path_text)
        payload = read_json(path)
        seed = int(payload.get("training_seed", -1))
        require(
            payload.get("schema_version")
            == "route_a_v3_route2_xeditsetflow_v4_confirmation_runtime.v1"
            and payload.get("run_stage") == "CONFIRMATION"
            and payload.get("selected_model") == "v4_full"
            and payload.get("screen_gate_path")
            == provenance["recovered_screen_gate_path"]
            and payload.get("validation_recovery", {}).get("training_git_head")
            == TRAINING_HEAD
            and payload.get("validation_recovery", {}).get("validation_git_head")
            == VALIDATION_HEAD
            and payload.get("validation_recovery", {}).get("parameter_updates") == 0
            and payload.get("validation_recovery", {}).get(
                "scientific_thresholds_changed"
            )
            is False,
            f"SetFlow V4.0.3 recovered confirmation config changed: {path}",
        )
        require(
            payload.get("development_test_outcomes_accessed") is False
            and payload.get("new_final_evaluation_outcomes_accessed") is False,
            f"SetFlow V4.0.3 config authorizes protected outcomes: {path}",
        )
        configs[seed] = path
    require(
        set(configs) == set(CONFIRMATION_SEEDS),
        "SetFlow V4.0.3 recovered confirmation config seeds changed",
    )
    return configs


def validate_authorization_v403(
    authorization: Mapping[str, Any], *, runner_head: str
) -> None:
    require(
        authorization.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_confirmation_launch_authorization.v1"
        and authorization.get("status")
        == "XEDITSETFLOW_V4_CONFIRMATION_LAUNCH_AUTHORIZED"
        and authorization.get("authorized_git_head") == runner_head
        and authorization.get("training_git_head") == TRAINING_HEAD
        and authorization.get("validation_git_head") == VALIDATION_HEAD
        and authorization.get("authorized_seeds") == list(CONFIRMATION_SEEDS)
        and authorization.get("authorized_run_id") == "v4_full"
        and authorization.get("recovery_parameter_update_count") == 0
        and authorization.get("scientific_thresholds_changed") is False
        and authorization.get("additional_seed_authorized") is False
        and authorization.get("development_test_authorized") is False
        and int(authorization.get("development_test_outcome_reads", -1)) == 0
        and int(authorization.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "SetFlow V4.0.3 recovered confirmation authorization changed",
    )


def build_schedule_v403(
    protocol: Mapping[str, Any],
    recovery_config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    authorization_path: Path,
    configs: Mapping[int, Path],
    selected_gpus: Sequence[int],
    diagnostics: Mapping[int, Mapping[str, Any]],
    cuda_probes: Mapping[int, Mapping[str, Any]],
    *,
    runner_head: str,
    runtime_manifest: Path,
    log_root: Path,
) -> dict[str, Any]:
    gpus = tuple(int(gpu) for gpu in selected_gpus)
    require(
        len(gpus) == len(CONFIRMATION_SEEDS) and len(set(gpus)) == len(gpus),
        "SetFlow V4.0.3 confirmation requires three distinct physical GPUs",
    )
    require(
        all(gpu in diagnostics and gpu in cuda_probes for gpu in gpus),
        "SetFlow V4.0.3 confirmation GPU diagnostics are incomplete",
    )
    queues = []
    for gpu, seed in zip(gpus, CONFIRMATION_SEEDS, strict=True):
        config_path = configs[seed]
        config = read_json(config_path)
        output = Path(config["output_root"]) / "v4_full"
        queues.append(
            {
                "physical_gpu_index": gpu,
                "jobs": [
                    {
                        "job_key": f"setflow:{seed}:v4_full",
                        "component": "setflow",
                        "training_seed": seed,
                        "run_id": "v4_full",
                        "output_directory": str(output),
                        "log_path": str(log_root / f"seed_{seed}_v4_full.log"),
                        "command": [
                            str(PYTHON),
                            str(TRAINER),
                            "--config",
                            str(config_path),
                            "--run-id",
                            "v4_full",
                            "--physical-gpu-index",
                            str(gpu),
                            "--authorization",
                            str(authorization_path),
                            "--output-dir",
                            str(output),
                        ],
                    }
                ],
            }
        )
    provenance = protocol["validation_recovery_provenance"]
    runner_outputs = protocol["runner_outputs"]
    posttraining = protocol["posttraining_binding"]
    return {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v403_recovered_confirmation_training_schedule.v1"
        ),
        "status": "FROZEN_RECOVERY_DERIVED_CONFIRMATION_TRAINING_SCHEDULE",
        "git_head": runner_head,
        "experiment_head": SCREEN_EXPERIMENT_HEAD,
        "training_git_head": TRAINING_HEAD,
        "validation_git_head": VALIDATION_HEAD,
        "worktree": str(WORKTREE),
        "runtime_manifest": str(runtime_manifest),
        "eligible_components": ["setflow"],
        "confirmation_protocol": str(PROTOCOL),
        "recovery_config": provenance["recovery_config_path"],
        "recovery_runtime": provenance["recovery_runtime_path"],
        "recovered_screen_gate": provenance["recovered_screen_gate_path"],
        "confirmation_authorization": str(authorization_path),
        "runner_verification_receipt": runner_outputs[
            "runner_verification_receipt_template"
        ].format(runner_git_head=runner_head),
        "config_manifest": str(
            Path(str(protocol["runtime_config_root"])) / "manifest.json"
        ),
        "required_seeds": list(CONFIRMATION_SEEDS),
        "gpu_diagnostics_before_launch": {
            str(gpu): dict(diagnostics[gpu]) for gpu in gpus
        },
        "cuda_bf16_probes": {
            str(gpu): dict(cuda_probes[gpu]) for gpu in gpus
        },
        "free_memory_gate_applied": False,
        "gpu_queues": queues,
        "posttraining_bindings": {
            "protocol_path": str(PROTOCOL),
            "training_git_head": TRAINING_HEAD,
            "validation_git_head": VALIDATION_HEAD,
            "runner_git_head": runner_head,
            "recovery_config_path": posttraining["recovery_config_path"],
            "recovered_screen_gate_path": posttraining[
                "recovered_screen_gate_path"
            ],
            "confirmation_authorization_path": str(authorization_path),
            "config_manifest_path": posttraining["config_manifest_path"],
            "training_runtime_path": str(runtime_manifest),
            "posttraining_runtime_root": runner_outputs[
                "posttraining_runtime_root_template"
            ].format(runner_git_head=runner_head),
            "posttraining_log_root": runner_outputs[
                "posttraining_log_root_template"
            ].format(runner_git_head=runner_head),
            "confirmation_gate_output": posttraining[
                "confirmation_gate_output"
            ],
        },
        "training_reused_from_screen": False,
        "screen_training_reused_by_recovery": True,
        "recovery_parameter_update_count": 0,
        "active_performance_output_read": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def run(current_head: str) -> dict[str, Any]:
    require(
        re.fullmatch(r"[0-9a-f]{40}", current_head) is not None,
        "expected current Git HEAD is invalid",
    )
    for path, label in (
        (PYTHON, "formal Python"),
        (PROTOCOL, "recovery-derived protocol"),
        (BASE_CONFIRMATION_PROTOCOL, "base confirmation protocol"),
        (SCREEN_CONFIG, "screen config"),
        (PREPARE, "SetFlow prepare entry"),
        (AUTHORIZE, "SetFlow recovery-aware authorizer"),
        (TRAINER, "SetFlow trainer"),
        (SCHEDULER, "confirmation scheduler"),
    ):
        require(path.is_file(), f"{label} is absent: {path}")
    require(
        command(["git", "rev-parse", "HEAD"]).stdout.strip() == current_head,
        "A100 worktree is not at expected runner HEAD",
    )
    require(
        not command(["git", "status", "--porcelain"]).stdout.strip(),
        "A100 runner worktree is dirty",
    )

    base_protocol = read_json(BASE_CONFIRMATION_PROTOCOL)
    protocol = read_json(PROTOCOL)
    screen_config = read_json(SCREEN_CONFIG)
    runner_outputs = protocol["runner_outputs"]
    runner_verification_receipt_path = Path(
        runner_outputs["runner_verification_receipt_template"].format(
            runner_git_head=current_head
        )
    )
    require(
        runner_verification_receipt_path.is_file(),
        "SetFlow V4.0.3 exact-runner-HEAD verification receipt is absent: "
        f"{runner_verification_receipt_path}",
    )
    runner_verification_receipt = read_json(
        runner_verification_receipt_path
    )
    provenance = protocol["validation_recovery_provenance"]
    recovery_config_path = Path(provenance["recovery_config_path"])
    recovery_runtime_path = Path(provenance["recovery_runtime_path"])
    recovered_gate_path = Path(provenance["recovered_screen_gate_path"])
    screen_authorization_path = Path(provenance["original_screen_authorization"])
    preflight_path = Path(screen_config["preflight_output_path"])
    source_data_audit_path = Path(screen_config["source_level_data_audit_path"])
    for path, label in (
        (recovery_config_path, "recovery config"),
        (recovery_runtime_path, "recovery runtime"),
        (recovered_gate_path, "recovered screen gate"),
        (screen_authorization_path, "original screen authorization"),
        (preflight_path, "original preflight"),
        (source_data_audit_path, "source data audit"),
    ):
        require(path.is_file(), f"SetFlow V4.0.3 {label} is absent: {path}")

    recovery_config = read_json(recovery_config_path)
    recovery_runtime = read_json(recovery_runtime_path)
    recovered_gate = read_json(recovered_gate_path)
    screen_authorization = read_json(screen_authorization_path)
    preflight = read_json(preflight_path)
    source_data_audit = read_json(source_data_audit_path)
    require_science_protocol_unchanged_v403(base_protocol, protocol)
    require_recovery_config_derivation_v403(
        screen_config, recovery_config, protocol
    )
    require_recovery_terminal_v403(protocol, recovery_runtime, recovered_gate)
    expected_authorization = build_recovered_confirmation_authorization_v403(
        base_protocol,
        protocol,
        screen_config,
        screen_authorization,
        preflight,
        source_data_audit,
        recovery_config,
        recovery_runtime,
        recovered_gate,
        runner_verification_receipt,
        current_runner_head=current_head,
        runner_verification_receipt_path=str(
            runner_verification_receipt_path
        ),
    )

    authorization_path = Path(
        runner_outputs["authorization_output_template"].format(
            runner_git_head=current_head
        )
    )
    runtime_root = Path(
        runner_outputs["training_runtime_root_template"].format(
            runner_git_head=current_head
        )
    )
    log_root = Path(
        runner_outputs["training_log_root_template"].format(
            runner_git_head=current_head
        )
    )
    config_root = Path(protocol["runtime_config_root"])
    run_root = Path(protocol["run_root"])
    for path, message in (
        (authorization_path, "confirmation authorization exists"),
        (config_root, "confirmation runtime config root exists"),
        (run_root, "confirmation run root exists"),
        (runtime_root, "confirmation training runtime exists"),
    ):
        require(not path.exists(), f"{message}: {path}")

    diagnostics = gpu_diagnostics()
    configured_gpus = tuple(
        int(gpu) for gpu in recovery_config["gpu_policy"]["physical_gpu_scope"]
    )
    require(
        len(configured_gpus) >= len(CONFIRMATION_SEEDS)
        and all(gpu in diagnostics for gpu in configured_gpus),
        "SetFlow V4.0.3 configured physical GPU inventory is incomplete",
    )
    selected_gpus = configured_gpus[: len(CONFIRMATION_SEEDS)]
    try:
        cuda_probes = {gpu: cuda_bf16_probe(gpu) for gpu in selected_gpus}
    except Exception as error:
        failure_path = Path(
            runner_outputs["cuda_failure_evidence_template"].format(
                runner_git_head=current_head
            )
        )
        if not failure_path.exists():
            write_new_atomic(
                failure_path,
                {
                    "schema_version": (
                        "route_a_v3_route2_xeditsetflow_v403_confirmation_cuda_failure.v1"
                    ),
                    "status": "STOPPED_BEFORE_CONFIRMATION_LAUNCH",
                    "runner_git_head": current_head,
                    "training_git_head": TRAINING_HEAD,
                    "validation_git_head": VALIDATION_HEAD,
                    "selected_physical_gpus": list(selected_gpus),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "cpu_fallback_used": False,
                    "parameter_update_count": 0,
                    "development_test_outcome_reads": 0,
                    "new_final_evaluation_outcome_reads": 0,
                },
            )
        raise

    command(
        [
            str(PYTHON),
            str(PREPARE),
            "--base-config",
            str(recovery_config_path),
            "--protocol",
            str(PROTOCOL),
            "--screen-gate",
            str(recovered_gate_path),
        ]
    )
    manifest_path = config_root / "manifest.json"
    manifest = read_json(manifest_path)
    configs = validate_manifest_v403(manifest, protocol)

    command(
        [
            str(PYTHON),
            str(AUTHORIZE),
            "--base-confirmation-protocol",
            str(BASE_CONFIRMATION_PROTOCOL),
            "--protocol",
            str(PROTOCOL),
            "--screen-config",
            str(SCREEN_CONFIG),
            "--screen-authorization",
            str(screen_authorization_path),
            "--preflight",
            str(preflight_path),
            "--source-data-audit",
            str(source_data_audit_path),
            "--recovery-config",
            str(recovery_config_path),
            "--recovery-runtime",
            str(recovery_runtime_path),
            "--recovered-screen-gate",
            str(recovered_gate_path),
            "--runner-verification-receipt",
            str(runner_verification_receipt_path),
            "--output",
            str(authorization_path),
        ]
    )
    authorization = read_json(authorization_path)
    require(
        authorization == expected_authorization,
        "SetFlow V4.0.3 materialized authorization differs from validated payload",
    )
    validate_authorization_v403(authorization, runner_head=current_head)

    runtime_root.mkdir(parents=True)
    log_root.mkdir(parents=True, exist_ok=True)
    runtime_manifest = runtime_root / "runtime.json"
    schedule_path = runtime_root / "schedule.json"
    schedule = build_schedule_v403(
        protocol,
        recovery_config,
        manifest,
        authorization_path,
        configs,
        selected_gpus,
        diagnostics,
        cuda_probes,
        runner_head=current_head,
        runtime_manifest=runtime_manifest,
        log_root=log_root,
    )
    write_new_atomic(schedule_path, schedule)
    scheduler_log = log_root / "scheduler.log"
    stream = scheduler_log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(PYTHON), str(SCHEDULER), "--schedule", str(schedule_path)],
        cwd=WORKTREE,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stream.close()
    launch = {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v403_recovered_confirmation_launch.v1"
        ),
        "status": "XEDITSETFLOW_V403_RECOVERED_CONFIRMATION_SCHEDULER_LAUNCHED",
        "runner_git_head": current_head,
        "training_git_head": TRAINING_HEAD,
        "validation_git_head": VALIDATION_HEAD,
        "scheduler_pid": process.pid,
        "schedule_path": str(schedule_path),
        "runtime_manifest": str(runtime_manifest),
        "scheduler_log": str(scheduler_log),
        "confirmation_authorization": str(authorization_path),
        "runner_verification_receipt": str(
            runner_verification_receipt_path
        ),
        "recovered_screen_gate": str(recovered_gate_path),
        "required_seeds": list(CONFIRMATION_SEEDS),
        "selected_physical_gpus": list(selected_gpus),
        "free_memory_gate_applied": False,
        "cpu_fallback_used": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_new_atomic(runtime_root / "launch.json", launch)
    return launch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.expected_head), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
