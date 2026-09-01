#!/usr/bin/env python3
"""Run the SetFlow V5 screen as one fail-closed package.

Stages:
  1. authorize (exact-HEAD-bound preflight + launch authorizations)
  2. preflight (source audit + per-arm capacity + BF16 batch) on one GPU
  3. TRAIN all four arms concurrently (one GPU each)
  4. VALIDATE every saved checkpoint concurrently (one GPU each)
  5. adjudicate Gate B0 + B1 -> screen_gate.json

Any technical failure at any stage stops the package and writes a terminal
failure artifact.  CUDA BF16 only; CPU fallback forbidden.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from core.route2_xeditsetflow_runtime_v5 import screen_run_spec_v5

PYTHON = "/home/cunyuliu/miniconda3/envs/editflow/bin/python"
ARMS = ["b_fix1", "b_fix2", "b_fix3", "b_arch1"]
ARM_GPU = {"b_fix1": 3, "b_fix2": 6, "b_fix3": 7, "b_arch1": 4}


def eprint(*args): print(*args, file=sys.stderr, flush=True)


def _run(cmd, cwd, log_path=None, check=True):
    eprint("RUN", " ".join(str(x) for x in cmd))
    with (open(log_path, "w") if log_path else open(os.devnull, "w")) as stream:
        result = subprocess.run(cmd, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed rc={result.returncode}: {' '.join(map(str, cmd))}")
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--family-root", required=True, type=Path,
                        help="/mnt/.../route2 (authorizations and experiments families)")
    parser.add_argument("--preflight-gpu", type=int, default=1)
    args = parser.parse_args()

    worktree = REPO_ROOT
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=worktree,
                          check=True, capture_output=True, text=True).stdout.strip()
    auth_root = args.family_root / "authorizations" / "xeditsetflow_v5"
    exp_root = args.family_root / "experiments" / "xeditsetflow_v5"
    log_root = args.family_root / "logs" / "xeditsetflow_v5"
    run_tag = f"screen_{config['training']['screen_seed']}"
    runtime_root = exp_root / run_tag
    validation_root = runtime_root / "outcome_free_validation_generation"
    screen_gate_path = runtime_root / "screen_gate.json"
    fail_gate_path = runtime_root / "screen_failure.json"

    def write_new(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".partial")
        partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        partial.replace(path)

    def stage_fail(stage: str, reason: str, details: Mapping[str, Any] | None = None) -> int:
        payload = {
            "schema_version": "route_a_v3_route2_xeditsetflow_v5_scheduler_failure.v1",
            "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
            "failure_stage": stage,
            "reason": reason,
            "cpu_fallback_used": False,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        }
        if details:
            payload["details"] = dict(details)
        if not fail_gate_path.exists() and not screen_gate_path.exists():
            write_new(fail_gate_path, payload)
        eprint(f"STAGE FAIL {stage}: {reason}")
        return 1

    # ---- stage 1: authorize ----
    try:
        _run([PYTHON, "scripts/route_a_v3/authorize_route2_xeditsetflow_v5_screen_stages.py",
              "--config", str(config_path),
              "--authorization-root", str(auth_root),
              "--preflight-runner-head", head],
             cwd=worktree,
             log_path=log_root / "authorize.log")
    except Exception as error:
        return stage_fail("AUTHORIZE", str(error))

    # ---- stage 2: preflight ----
    try:
        _run([PYTHON, "scripts/route_a_v3/preflight_route2_xeditsetflow_v5.py",
              "--config", str(config_path),
              "--authorization", str(auth_root / "v5_screen_preflight_authorization.json"),
              "--physical-gpu-index", str(args.preflight_gpu)],
             cwd=worktree,
             log_path=log_root / "preflight.log")
    except Exception as error:
        return stage_fail("PREFLIGHT", str(error))

    # ---- stage 3: train four arms concurrently ----
    results: dict[str, int] = {}
    errors: dict[str, str] = {}

    def train_arm(run_id: str) -> None:
        try:
            _run([PYTHON, "scripts/route_a_v3/train_route2_xeditsetflow_v5.py",
                  "--config", str(config_path),
                  "--run-id", run_id,
                  "--authorization", str(auth_root / "v5_screen_launch_authorization.json"),
                  "--physical-gpu-index", str(ARM_GPU[run_id])],
                 cwd=worktree,
                 log_path=log_root / f"train_{run_id}.log")
            results[run_id] = 0
        except Exception as error:
            results[run_id] = 1
            errors[run_id] = str(error)

    threads = [threading.Thread(target=train_arm, args=(arm,)) for arm in ARMS]
    for t in threads: t.start()
    for t in threads: t.join()
    if any(results.values()):
        return stage_fail("TRAINING", "one or more training arms failed", errors)

    # ---- stage 4: validate all saved checkpoints concurrently ----
    val_jobs = []
    for arm in ARMS:
        for checkpoint_pass in config["training"]["saved_checkpoint_passes"]:
            val_jobs.append((arm, int(checkpoint_pass)))
    val_results: dict[str, int] = {}
    val_errors: dict[str, str] = {}

    def validate_job(job: tuple[str, int]) -> None:
        run_id, checkpoint_pass = job
        try:
            _run([PYTHON, "scripts/route_a_v3/validate_route2_xeditsetflow_v5_checkpoint.py",
                  "--config", str(config_path),
                  "--run-id", run_id,
                  "--checkpoint-pass", str(checkpoint_pass),
                  "--authorization", str(auth_root / "v5_screen_launch_authorization.json"),
                  "--physical-gpu-index", str(ARM_GPU[run_id])],
                 cwd=worktree,
                 log_path=log_root / f"validate_{run_id}_pass{checkpoint_pass}.log")
            val_results[f"{run_id}_p{checkpoint_pass}"] = 0
        except Exception as error:
            val_results[f"{run_id}_p{checkpoint_pass}"] = 1
            val_errors[f"{run_id}_p{checkpoint_pass}"] = str(error)

    v_threads = [threading.Thread(target=validate_job, args=(job,)) for job in val_jobs]
    for t in v_threads: t.start()
    for t in v_threads: t.join()
    if any(val_results.values()):
        return stage_fail("VALIDATION", "one or more validation jobs failed", val_errors)

    # ---- stage 5: adjudicate Gate B0/B1 ----
    try:
        _run([PYTHON, "scripts/route_a_v3/adjudicate_route2_xeditsetflow_v5_screen.py",
              "--config", str(config_path),
              "--screen-gate-output", str(screen_gate_path)],
             cwd=worktree,
             log_path=log_root / "adjudicate.log")
    except Exception as error:
        return stage_fail("ADJUDICATION", str(error))

    print(json.dumps({"status": "XEDITSETFLOW_V5_SCREEN_TERMINAL", "screen_gate_path": str(screen_gate_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
