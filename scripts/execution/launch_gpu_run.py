#!/usr/bin/env python3
"""Fail-closed launcher for formal neural runs under the V2 execution contract."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HEALTH_FIELDS = (
    "torch_cuda_available",
    "model_parameters_on_cuda",
    "input_batch_on_cuda",
    "real_forward_on_cuda",
    "real_backward_on_cuda",
    "optimizer_update_completed",
    "max_memory_allocated_gt_zero",
    "cpu_fallback_count_zero",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def probe_cuda(torch_module=None) -> dict[str, Any]:
    """Run a real one-step CUDA update; return evidence instead of falling back."""
    report: dict[str, Any] = {
        "checked_at_utc": _utc_now(),
        **{field: False for field in HEALTH_FIELDS},
        "device": None,
        "framework_version": None,
        "max_memory_allocated_bytes": 0,
        "error": None,
    }
    try:
        torch = torch_module
        if torch is None:
            import torch as torch_import

            torch = torch_import
        report["framework_version"] = str(torch.__version__)
        report["torch_cuda_available"] = bool(torch.cuda.is_available())
        if not report["torch_cuda_available"]:
            raise RuntimeError("CUDA_UNAVAILABLE")

        device = torch.device("cuda")
        report["device"] = str(device)
        torch.cuda.reset_peak_memory_stats(device)
        model = torch.nn.Linear(4, 2).to(device)
        report["model_parameters_on_cuda"] = all(
            parameter.device.type == "cuda" for parameter in model.parameters()
        )
        batch = torch.ones((2, 4), device=device)
        report["input_batch_on_cuda"] = batch.device.type == "cuda"
        target = torch.zeros((2, 2), device=device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        output = model(batch)
        report["real_forward_on_cuda"] = output.device.type == "cuda"
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        report["real_backward_on_cuda"] = all(
            parameter.grad is not None and parameter.grad.device.type == "cuda"
            for parameter in model.parameters()
        )
        optimizer.step()
        report["optimizer_update_completed"] = any(
            not torch.equal(old, new.detach())
            for old, new in zip(before, model.parameters())
        )
        torch.cuda.synchronize(device)
        allocated = int(torch.cuda.max_memory_allocated(device))
        report["max_memory_allocated_bytes"] = allocated
        report["max_memory_allocated_gt_zero"] = allocated > 0
        report["cpu_fallback_count_zero"] = all(
            bool(report[field])
            for field in (
                "model_parameters_on_cuda",
                "input_batch_on_cuda",
                "real_forward_on_cuda",
                "real_backward_on_cuda",
            )
        )
    except Exception as exc:  # failure evidence must survive any framework error
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["passed"] = all(bool(report[field]) for field in HEALTH_FIELDS)
    return report


def _failure(run_root: Path, reason: str, health: dict[str, Any]) -> int:
    status = {
        "state": "FAILED_WITH_EVIDENCE",
        "updated_at_utc": _utc_now(),
        "stop_reason": reason,
        "automatic_cpu_fallback": False,
    }
    _atomic_json(run_root / "status.json", status)
    _atomic_json(
        run_root / "failure/failure.json",
        {"status": status, "cuda_health": health},
    )
    return 70


def _load_training_health(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            **{field: False for field in HEALTH_FIELDS},
            "passed": False,
            "error": "MISSING_REAL_TRAINING_CUDA_HEALTH",
        }
    except (OSError, json.JSONDecodeError) as exc:
        return {
            **{field: False for field in HEALTH_FIELDS},
            "passed": False,
            "error": f"INVALID_REAL_TRAINING_CUDA_HEALTH: {type(exc).__name__}",
        }
    payload["passed"] = all(bool(payload.get(field)) for field in HEALTH_FIELDS)
    if not payload["passed"] and not payload.get("error"):
        payload["error"] = "REAL_TRAINING_CUDA_HEALTH_FIELD_FALSE_OR_MISSING"
    return payload


def launch(
    run_root: Path,
    command: list[str],
    torch_module=None,
    project_root: Path | None = None,
) -> int:
    run_root = run_root.resolve()
    health_path = run_root / "logs/cuda_health.json"
    preflight_path = run_root / "logs/cuda_preflight.json"
    preflight = probe_cuda(torch_module=torch_module)
    _atomic_json(preflight_path, preflight)
    if not preflight["passed"]:
        _atomic_json(health_path, preflight)
        return _failure(run_root, "CUDA_HEALTH_CHECK_FAILED", preflight)
    if not command:
        return _failure(run_root, "EMPTY_FORMAL_RUN_COMMAND", preflight)
    if health_path.exists():
        return _failure(run_root, "STALE_REAL_TRAINING_CUDA_HEALTH", preflight)

    _atomic_json(
        run_root / "status.json",
        {
            "state": "GPU_VERIFIED",
            "updated_at_utc": _utc_now(),
            "stop_reason": None,
            "automatic_cpu_fallback": False,
        },
    )
    env = dict(os.environ)
    env["EDITFLOW_REQUIRE_CUDA"] = "1"
    env["EDITFLOW_CUDA_HEALTH_FILE"] = str(health_path)
    stdout_path = run_root / "logs/stdout.log"
    stderr_path = run_root / "logs/stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            completed = subprocess.run(
                command,
                cwd=(project_root or Path(__file__).resolve().parents[2]).resolve(),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
    except OSError as exc:
        return _failure(
            run_root,
            f"FORMAL_RUN_LAUNCH_ERROR_{type(exc).__name__}",
            preflight,
        )
    if completed.returncode != 0:
        return _failure(
            run_root,
            f"FORMAL_RUN_EXIT_{completed.returncode}",
            preflight,
        )
    training_health = _load_training_health(health_path)
    if not training_health["passed"]:
        return _failure(
            run_root,
            "REAL_TRAINING_CUDA_HEALTH_FAILED",
            training_health,
        )
    _atomic_json(
        run_root / "status.json",
        {
            "state": "TRAINING_FINISHED",
            "updated_at_utc": _utc_now(),
            "stop_reason": None,
            "automatic_cpu_fallback": False,
            "exit_code": completed.returncode,
        },
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="formal command after '--'; never runs if CUDA health fails",
    )
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return launch(args.run_root, command, project_root=args.project_root)


if __name__ == "__main__":
    raise SystemExit(main())
