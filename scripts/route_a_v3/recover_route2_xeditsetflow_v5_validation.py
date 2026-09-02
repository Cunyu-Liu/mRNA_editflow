#!/usr/bin/env python3
"""Re-run SetFlow V5 outcome-free checkpoint validation without retraining.

The 2026-09-02 16:32 screen failure at the VALIDATION stage was caused solely
by the stale ledger provenance corrected by
``correct_route2_xeditsetflow_v5_stale_provenance.py`` (see the
provenance_correction_*.json audit at the screen root).  This recovery
re-executes scheduler stages 4-5 (validate all 12 saved checkpoints
concurrently, then adjudicate Gate B0/B1) and, unlike the screen scheduler,
never clears or touches any terminal training artifact.

Fail-closed properties:
  * refuses to start unless the provenance correction audit exists and every
    arm's training_attempt.json already carries its authorized Git HEAD;
  * refuses to start unless every training summary is TERMINAL complete;
  * refuses to start if any target validation output already exists;
  * every validation job runs on a physical A100 with CUDA + BF16 (CPU
    fallback forbidden) and without CUDA_VISIBLE_DEVICES remapping.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PYTHON = "/home/cunyuliu/miniconda3/envs/editflow/bin/python"
ARMS = ("b_fix1", "b_fix2", "b_fix3", "b_arch1")
DEFAULT_GPU_MAP = {"b_fix1": 5, "b_fix2": 1, "b_fix3": 3, "b_arch1": 4}
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class RecoveryError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryError(message)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)


def verify_ready(config: Mapping[str, Any]) -> dict[str, str]:
    """Require the corrected provenance and terminal training packages."""
    screen_root = Path(config["output_root"])
    audits = sorted(screen_root.glob("provenance_correction_*.json"))
    _require(bool(audits), "no provenance correction audit found at the screen root")
    heads: dict[str, str] = {}
    for arm in ARMS:
        directory = screen_root / arm
        summary = _read_json(directory / "training_summary.json")
        _require(
            summary.get("status")
            == "TERMINAL_XEDITSETFLOW_V5_TRAINING_COMPLETE_PENDING_VALIDATION",
            f"training is not terminal: {arm}",
        )
        training_config = _read_json(directory / "training_config.json")
        attempt = _read_json(directory / "training_attempt.json")
        authorized_head = str(training_config.get("authorized_git_head", ""))
        _require(
            COMMIT_RE.fullmatch(authorized_head) is not None,
            f"authorized_git_head invalid: {arm}",
        )
        _require(
            str(attempt.get("code_commit", "")) == authorized_head,
            f"attempt code_commit still disagrees with the authorized head: {arm}",
        )
        heads[arm] = authorized_head
    _require(
        len(set(heads.values())) == 1,
        "arms were trained from different Git HEADs",
    )
    return heads


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--family-root", required=True, type=Path)
    parser.add_argument(
        "--gpu-map", type=str, default=json.dumps(DEFAULT_GPU_MAP),
        help='JSON object mapping run_id -> physical GPU index',
    )
    args = parser.parse_args()

    worktree = REPO_ROOT
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    gpu_map = {str(k): int(v) for k, v in json.loads(args.gpu_map).items()}
    auth_root = args.family_root.resolve() / "authorizations" / "xeditsetflow_v5"
    log_root = args.family_root.resolve() / "logs" / "xeditsetflow_v5"
    screen_root = Path(config["output_root"])
    validation_root = Path(config["validation_output_root"])
    screen_gate_path = screen_root / "screen_gate.json"
    log_root.mkdir(parents=True, exist_ok=True)

    heads = verify_ready(config)
    head = next(iter(heads.values()))
    eprint(f"provenance verified: all arms trained from {head}")

    passes = tuple(sorted(int(p) for p in config["training"]["saved_checkpoint_passes"]))
    for arm in ARMS:
        _require(arm in gpu_map, f"GPU map is missing arm: {arm}")
        for checkpoint_pass in passes:
            target = validation_root / arm / f"pass_{checkpoint_pass}"
            _require(not target.exists(), f"validation output already exists: {target}")
    _require(not screen_gate_path.exists(), "screen gate already adjudicated")

    started = time.time()
    val_results: dict[str, int] = {}
    val_errors: dict[str, str] = {}

    def run_job(arm: str, checkpoint_pass: int) -> None:
        key = f"{arm}_p{checkpoint_pass}"
        log_path = log_root / f"recovery2_validate_{arm}_pass{checkpoint_pass}.log"
        cmd = [
            PYTHON, "scripts/route_a_v3/validate_route2_xeditsetflow_v5_checkpoint.py",
            "--config", str(config_path),
            "--run-id", arm,
            "--checkpoint-pass", str(checkpoint_pass),
            "--authorization", str(auth_root / "v5_screen_launch_authorization.json"),
            "--physical-gpu-index", str(gpu_map[arm]),
        ]
        eprint("RUN", " ".join(cmd))
        try:
            with log_path.open("w") as stream:
                result = subprocess.run(
                    cmd, cwd=worktree, stdout=stream,
                    stderr=subprocess.STDOUT, text=True,
                )
            val_results[key] = result.returncode
            if result.returncode != 0:
                val_errors[key] = f"rc={result.returncode}; see {log_path}"
        except Exception as error:  # pragma: no cover - defensive
            val_results[key] = 1
            val_errors[key] = str(error)

    jobs = [(arm, checkpoint_pass) for arm in ARMS for checkpoint_pass in passes]
    threads = [threading.Thread(target=run_job, args=job) for job in jobs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    eprint(f"validation stage finished in {time.time() - started:.0f}s: {val_results}")

    if any(val_results.values()):
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        failure_path = screen_root / f"recovery2_validation_failure_{stamp}.json"
        failure_path.write_text(
            json.dumps(
                {
                    "schema_version": "route_a_v3_route2_xeditsetflow_v5_validation_recovery_failure.v1",
                    "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
                    "failure_stage": "VALIDATION",
                    "reason": "one or more recovery validation jobs failed",
                    "results": val_results,
                    "errors": val_errors,
                    "gpu_map": gpu_map,
                    "cpu_fallback_used": False,
                    "development_test_outcome_reads": 0,
                    "new_final_evaluation_outcome_reads": 0,
                },
                indent=2, sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        eprint(f"STAGE FAIL VALIDATION: see {failure_path}")
        return 1

    adjudicate_log = log_root / "recovery2_adjudicate.log"
    cmd = [
        PYTHON, "scripts/route_a_v3/adjudicate_route2_xeditsetflow_v5_screen.py",
        "--config", str(config_path),
        "--screen-gate-output", str(screen_gate_path),
    ]
    eprint("RUN", " ".join(cmd))
    with adjudicate_log.open("w") as stream:
        result = subprocess.run(
            cmd, cwd=worktree, stdout=stream, stderr=subprocess.STDOUT, text=True
        )
    if result.returncode != 0:
        eprint(f"STAGE FAIL ADJUDICATION rc={result.returncode}: see {adjudicate_log}")
        return 1

    print(
        json.dumps(
            {
                "status": "XEDITSETFLOW_V5_VALIDATION_RECOVERY_TERMINAL",
                "screen_gate_path": str(screen_gate_path),
                "gpu_map": gpu_map,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
