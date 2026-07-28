#!/usr/bin/env python3
"""Collect a read-only execution preflight for a later registered run."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run(command: list[str], cwd: Path | None = None) -> dict[str, object]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "returncode": None, "error": type(exc).__name__}


def _cpu_memory() -> dict[str, int | None]:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        values: dict[str, int] = {}
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", maxsplit=1)
            values[key] = int(raw.strip().split()[0]) * 1024
        return {
            "total_bytes": values.get("MemTotal"),
            "available_bytes": values.get("MemAvailable"),
            "swap_total_bytes": values.get("SwapTotal"),
            "swap_free_bytes": values.get("SwapFree"),
        }
    page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else None
    physical_pages = (
        os.sysconf("SC_PHYS_PAGES") if hasattr(os, "sysconf") else None
    )
    return {
        "total_bytes": (
            int(page_size) * int(physical_pages)
            if page_size is not None and physical_pages is not None
            else None
        ),
        "available_bytes": None,
        "swap_total_bytes": None,
        "swap_free_bytes": None,
    }


def collect(project_root: Path) -> dict[str, object]:
    root = project_root.resolve()
    disk = shutil.disk_usage(root)
    commands = {
        "git_root": _run(["git", "rev-parse", "--show-toplevel"], root),
        "git_branch": _run(["git", "branch", "--show-current"], root),
        "git_head": _run(["git", "rev-parse", "HEAD"], root),
        "git_status": _run(["git", "status", "--porcelain=v2", "--branch"], root),
        "nvidia_smi": _run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,memory.total,"
                "memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader",
            ]
        ),
        "gpu_processes": _run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader",
            ]
        ),
        "nvidia_smi_summary": _run(["nvidia-smi"]),
        "processes": _run(
            ["ps", "-u", str(os.getuid()), "-o", "pid=,ppid=,etimes=,stat=,pcpu=,pmem=,args="]
        ),
    }
    return {
        "artifact_type": "read_only_execution_preflight",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "hostname": platform.node(),
        "python": sys.version,
        "cpu_memory": _cpu_memory(),
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "commands": commands,
        "mutations_performed": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = collect(args.project_root)
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
