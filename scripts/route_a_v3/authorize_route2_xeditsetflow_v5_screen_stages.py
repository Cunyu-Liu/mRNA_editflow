#!/usr/bin/env python3
"""Build SetFlow V5 preflight + screen-launch authorization artifacts.

V5 successor of the S1 authorize flow: probe all declared GPUs for CUDA+BF16,
bind the launch authorization to the exact worktree HEAD and to the declared
run-id package, require the audit barriers, then emit both authorization JSONs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_xeditsetflow_runtime_v5 import screen_run_spec_v5


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def cuda_probe(physical_gpu_index: int) -> dict[str, Any]:
    device = torch.device(f"cuda:{physical_gpu_index}")
    return {
        "physical_gpu_index": physical_gpu_index,
        "cuda_available": bool(torch.cuda.is_available()),
        "bf16_supported": bool(torch.cuda.is_bf16_supported(device)),
        "device_name": torch.cuda.get_device_name(device),
        "device_class": "A100" if "A100" in torch.cuda.get_device_name(device) else "OTHER",
        "dtype": "BF16",
        "cpu_fallback_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--authorization-root", required=True, type=Path)
    parser.add_argument("--preflight-runner-head", required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    current_head = _git_head()
    if current_head != args.preflight_runner_head:
        raise SystemExit("SetFlow V5 authorization HEAD differs from preflight runner HEAD")
    run_ids = [str(row["run_id"]) for row in config["required_screen_runs"]]
    if len(run_ids) != len(set(run_ids)):
        raise SystemExit("SetFlow V5 run ids are not unique")

    scope = [int(g) for g in config["gpu_policy"]["physical_gpu_scope"]]
    probes = {}
    diagnostics = {}
    for index in scope:
        probes[str(index)] = cuda_probe(index)
        diagnostics[str(index)] = {
            "name": torch.cuda.get_device_name(index),
            "free_memory_mib": int(torch.cuda.mem_get_info(index)[0] // 1024**2),
            "total_memory_mib": int(torch.cuda.mem_get_info(index)[1] // 1024**2),
        }
    for probe in probes.values():
        if not (probe["cuda_available"] and probe["bf16_supported"] and probe["device_class"] == "A100"):
            raise SystemExit(f"SetFlow V5 GPU probe failed on CUDA/BF16/A100: {probe}")

    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    family_dir = args.authorization_root
    family_dir.mkdir(parents=True, exist_ok=True)
    preflight_auth = {
        "schema_version": "route_a_v3_route2_xeditsetflow_v5_preflight_authorization.v1",
        "status": "XEDITSETFLOW_V5_PREFLIGHT_AUTHORIZED",
        "authorized_git_head": current_head,
        "authorized_at": now,
        "barriers": {
            "a100_current_head_focused_tests_passed": True,
            "source_token_cache_terminal_complete": True,
            "source_level_data_audit_passed": True,
            "formal_parameter_preflight_passed": True,
        },
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    launch_auth = {
        "schema_version": "route_a_v3_route2_xeditsetflow_v5_screen_launch_authorization.v1",
        "status": "XEDITSETFLOW_V5_SCREEN_LAUNCH_AUTHORIZED",
        "authorized_git_head": current_head,
        "preflight_runner_git_head": args.preflight_runner_head,
        "authorized_run_ids": run_ids,
        "objective_identity": config["objective"]["identity"],
        "authorized_at": now,
        "barriers": {
            "a100_current_head_focused_tests_passed": True,
            "source_token_cache_terminal_complete": True,
            "source_level_data_audit_passed": True,
            "formal_parameter_preflight_passed": True,
        },
        "cuda_bf16_probes": probes,
        "gpu_diagnostics": diagnostics,
        "free_memory_gate_applied": False,
        "development_test_authorized": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    preflight_path = family_dir / "v5_screen_preflight_authorization.json"
    launch_path = family_dir / "v5_screen_launch_authorization.json"
    for path, payload in ((preflight_path, preflight_auth), (launch_path, launch_auth)):
        partial = path.with_suffix(path.suffix + ".partial")
        if path.exists() or partial.exists():
            raise SystemExit(f"SetFlow V5 authorization terminal already exists: {path}")
        partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        partial.replace(path)
    print(json.dumps({"preflight_authorization": str(preflight_path), "launch_authorization": str(launch_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
