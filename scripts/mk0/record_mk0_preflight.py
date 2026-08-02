#!/usr/bin/env python3
"""Record a read-only, privacy-minimised MK0 preflight snapshot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
import pwd
import re
import shutil
import subprocess
import sys
from typing import Any

import torch


FORMAL_RUN_ID = re.compile(
    r"^MK0_(?P<model>[A-Za-z0-9]+)_(?P<dataset>[A-Za-z0-9]+)_"
    r"(?P<split>[A-Za-z0-9]+)_(?P<utc>[0-9]{8}T[0-9]{6}Z)_"
    r"(?P<short_sha>[0-9a-f]{7,12})_s(?P<seed>[0-9]+)$"
)


def _formal_run_time(run_id: str, *, label: str) -> datetime:
    match = FORMAL_RUN_ID.fullmatch(run_id)
    if match is None:
        raise ValueError(f"{label} is not a formal MK0 run ID")
    try:
        observed = datetime.strptime(match.group("utc"), "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise ValueError(f"{label} UTC is not a calendar time") from error
    if observed.strftime("%Y%m%dT%H%M%SZ") != match.group("utc"):
        raise ValueError(f"{label} UTC is not canonical")
    return observed


def validate_parent_run_lineage(
    child_run_id: str, parent_run_id: str | None
) -> str | None:
    """Validate optional repair lineage without changing non-repair behavior."""

    if parent_run_id is None:
        return None
    parent_time = _formal_run_time(parent_run_id, label="parent run ID")
    child_time = _formal_run_time(child_run_id, label="child run ID")
    if parent_time >= child_time:
        raise ValueError("parent run ID UTC must precede child run ID UTC")
    return parent_run_id


def preflight_lineage_fields(
    child_run_id: str, parent_run_id: str | None
) -> dict[str, str | None]:
    return {
        "run_id": child_run_id,
        "parent_run_id": validate_parent_run_lineage(child_run_id, parent_run_id),
    }


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def run(command: list[str], *, cwd: Path | None = None, check: bool = True) -> str:
    result = subprocess.run(
        command, cwd=cwd, check=check, capture_output=True, text=True
    )
    return result.stdout.strip()


def observe(command: list[str]) -> dict[str, Any]:
    """Run one read-only probe and retain enough evidence to fail closed."""

    result = subprocess.run(command, check=False, capture_output=True, text=True)
    return {
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr_sha256": hashlib.sha256(result.stderr.encode("utf-8")).hexdigest(),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def disk(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    usage = shutil.disk_usage(resolved)
    return {
        "path": str(resolved),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "reserved_bytes": usage.total - usage.used - usage.free,
    }


def cpu_memory() -> dict[str, int]:
    """Record Linux RAM capacity without reading any user process payload."""

    fields: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        parts = value.split()
        if parts and parts[0].isdigit():
            multiplier = 1024 if len(parts) > 1 and parts[1] == "kB" else 1
            fields[key] = int(parts[0]) * multiplier
    for required in ("MemTotal", "MemAvailable", "MemFree"):
        if required not in fields or fields[required] <= 0:
            raise RuntimeError(f"/proc/meminfo lacks a valid {required}")
    return {
        "total_bytes": fields["MemTotal"],
        "available_bytes": fields["MemAvailable"],
        "free_bytes": fields["MemFree"],
        "swap_total_bytes": fields.get("SwapTotal", 0),
        "swap_free_bytes": fields.get("SwapFree", 0),
    }


def process_owner(pid: int) -> dict[str, Any]:
    """Resolve one GPU PID owner, retaining an explicit race outcome."""

    status_path = Path("/proc") / str(pid) / "status"
    try:
        lines = status_path.read_text(encoding="utf-8").splitlines()
        uid_line = next(line for line in lines if line.startswith("Uid:"))
        uid = int(uid_line.split()[1])
        owner = pwd.getpwuid(uid).pw_name
    except (FileNotFoundError, ProcessLookupError):
        return {
            "owner": None,
            "owner_uid": None,
            "owner_resolution": "PROCESS_EXITED_DURING_PREFLIGHT",
        }
    except (KeyError, OSError, StopIteration, ValueError) as error:
        return {
            "owner": None,
            "owner_uid": None,
            "owner_resolution": f"UNRESOLVED_{type(error).__name__}",
        }
    return {
        "owner": owner,
        "owner_uid": uid,
        "owner_resolution": "RESOLVED_FROM_PROC_STATUS",
    }


def top_level_inventory(
    path: Path, *, name_prefix: str | None = None
) -> dict[str, Any]:
    """Hash a metadata-only, non-recursive directory inventory."""

    resolved = path.resolve(strict=True)
    records: list[dict[str, Any]] = []
    for child in sorted(resolved.iterdir(), key=lambda item: item.name):
        if name_prefix is not None and not child.name.startswith(name_prefix):
            continue
        stat = child.lstat()
        kind = (
            "symlink"
            if child.is_symlink()
            else (
                "directory"
                if child.is_dir()
                else "file" if child.is_file() else "other"
            )
        )
        records.append(
            {
                "name": child.name,
                "path": str(child),
                "kind": kind,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    encoded = canonical_json_bytes(records)
    return {
        "root": str(resolved),
        "entry_count": len(records),
        "entries": records,
        "inventory_sha256": hashlib.sha256(encoded).hexdigest(),
        "recursive": False,
        "metadata_only": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--parent-run-id")
    parser.add_argument("--goal-sha256", required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--main-repo", type=Path, required=True)
    parser.add_argument("--fm0-closure-root", type=Path, required=True)
    parser.add_argument("--d1-data", type=Path, required=True)
    parser.add_argument("--d1-ledger", type=Path, required=True)
    parser.add_argument("--mnt-root", type=Path, required=True)
    args = parser.parse_args()
    lineage = preflight_lineage_fields(args.run_id, args.parent_run_id)
    worktree = args.worktree.resolve(strict=True)
    main_repo = args.main_repo.resolve(strict=True)
    fm0 = args.fm0_closure_root.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    gpu_probe = observe(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    nvidia_smi_probe = observe(["nvidia-smi"])
    nvidia_smi_l_probe = observe(["nvidia-smi", "-L"])
    cuda_version_match = re.search(
        r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)*)", nvidia_smi_probe["stdout"]
    )
    if nvidia_smi_probe["exit_code"] != 0 or cuda_version_match is None:
        raise RuntimeError("cannot parse the nvidia-smi driver-supported CUDA version")
    if nvidia_smi_l_probe["exit_code"] != 0:
        raise RuntimeError("cannot query the nvidia-smi GPU/MIG topology")
    gpu_rows = []
    for line in gpu_probe["stdout"].splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 8:
            gpu_rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "uuid": parts[2],
                    "driver_version": parts[3],
                    "memory_total_mib": int(parts[4]),
                    "memory_used_mib": int(parts[5]),
                    "memory_free_mib": int(parts[6]),
                    "utilization_gpu_percent": (
                        int(parts[7]) if parts[7].lstrip("-").isdigit() else None
                    ),
                }
            )
    if gpu_probe["exit_code"] != 0 or not gpu_rows:
        raise RuntimeError("nvidia-smi GPU inventory failed or was empty")
    mig_instances: list[dict[str, Any]] = []
    parent_uuid: str | None = None
    for line in nvidia_smi_l_probe["stdout"].splitlines():
        physical_match = re.match(r"GPU\s+([0-9]+):.*\(UUID:\s*(GPU-[^)]+)\)", line)
        if physical_match is not None:
            parent_uuid = physical_match.group(2)
            continue
        mig_match = re.match(
            r"\s*MIG\s+([^\s]+)\s+Device\s+([0-9]+):\s*\(UUID:\s*(MIG-[^)]+)\)",
            line,
        )
        if mig_match is not None:
            if parent_uuid is None:
                raise RuntimeError("MIG instance was listed without a physical parent")
            mig_instances.append(
                {
                    "parent_gpu_uuid": parent_uuid,
                    "profile": mig_match.group(1),
                    "device_index_within_parent": int(mig_match.group(2)),
                    "mig_uuid": mig_match.group(3),
                }
            )
    gpu_process_probe = observe(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if gpu_process_probe["exit_code"] != 0:
        raise RuntimeError("nvidia-smi compute-process inventory failed")
    gpu_process_lines = [
        line.strip()
        for line in gpu_process_probe["stdout"].splitlines()
        if line.strip()
    ]
    gpu_process_records: list[dict[str, Any]] = []
    for line in gpu_process_lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4 or not parts[0].isdigit() or not parts[3].isdigit():
            raise RuntimeError("cannot parse nvidia-smi compute-process record")
        pid = int(parts[0])
        gpu_process_records.append(
            {
                "pid": pid,
                "process_name": parts[1],
                "gpu_uuid": parts[2],
                "used_memory_mib": int(parts[3]),
                **process_owner(pid),
            }
        )
        if gpu_process_records[-1]["owner_resolution"].startswith("UNRESOLVED_"):
            raise RuntimeError(f"cannot resolve owner for GPU process {pid}")
    gpu_process_records.sort(key=lambda record: (record["gpu_uuid"], record["pid"]))
    process_probe = observe(
        ["ps", "-u", str(os.getuid()), "-o", "pid=,stat=,etime=,comm="]
    )
    if process_probe["exit_code"] != 0:
        raise RuntimeError("current-user process inventory failed")
    process_lines = process_probe["stdout"].splitlines()
    process_records: list[dict[str, Any]] = []
    for line in process_lines:
        parts = line.split(None, 3)
        if len(parts) != 4 or not parts[0].isdigit():
            raise RuntimeError("cannot parse current-user process record")
        process_records.append(
            {
                "pid": int(parts[0]),
                "stat": parts[1],
                "elapsed": parts[2],
                "command": parts[3],
            }
        )
    if not process_records:
        raise RuntimeError("current-user process inventory was empty")
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count()
    if not cuda_available or cuda_device_count <= 0:
        raise RuntimeError("PyTorch CUDA preflight failed closed")
    if cuda_device_count != len(gpu_rows):
        raise RuntimeError(
            "preflight must see the same complete GPU inventory through PyTorch and nvidia-smi"
        )
    worktrees = run(["git", "worktree", "list", "--porcelain"], cwd=main_repo)
    fm0_ledger = fm0 / "artifact_checksums.sha256"
    mnt_root = args.mnt_root.resolve(strict=True)
    data_root = (main_repo / "data").resolve(strict=True)
    data_registry_root = (main_repo / "data_registry").resolve(strict=True)
    artifacts_root = (main_repo / "artifacts").resolve(strict=True)
    payload: dict[str, Any] = {
        "schema_version": "mk0_preflight_v1",
        **lineage,
        "observed_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "mode": "read_only_metadata_and_hashes",
        "collector": {
            "path": str(Path(__file__).resolve(strict=True)),
            "sha256": sha256_file(Path(__file__).resolve(strict=True)),
            "pid": os.getpid(),
        },
        "goal_sha256": args.goal_sha256,
        "worktree": {
            "path": str(worktree),
            "branch": run(["git", "branch", "--show-current"], cwd=worktree),
            "head": run(["git", "rev-parse", "HEAD"], cwd=worktree),
            "status_porcelain": run(["git", "status", "--porcelain=v1"], cwd=worktree),
        },
        "protected_main_repo": {
            "path": str(main_repo),
            "branch": run(["git", "branch", "--show-current"], cwd=main_repo),
            "head": run(["git", "rev-parse", "HEAD"], cwd=main_repo),
            "status_porcelain": run(["git", "status", "--porcelain=v1"], cwd=main_repo),
            "worktree_list_sha256": hashlib.sha256(
                worktrees.encode("utf-8")
            ).hexdigest(),
            "worktree_list_porcelain": worktrees,
        },
        "resources": {
            "home_filesystem": disk(worktree),
            "mnt_filesystem": disk(mnt_root),
            "cpu_memory": cpu_memory(),
            "framework": {
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "python_executable": os.path.abspath(sys.executable),
                "torch_version": torch.__version__,
                "torch_cuda_build_version": torch.version.cuda,
                "torch_cudnn_version": torch.backends.cudnn.version(),
                "torch_cuda_is_available": cuda_available,
                "torch_cuda_device_count": cuda_device_count,
            },
            "driver_supported_cuda_version": cuda_version_match.group(1),
            "nvidia_smi_exit_code": nvidia_smi_probe["exit_code"],
            "nvidia_smi_stdout_sha256": hashlib.sha256(
                nvidia_smi_probe["stdout"].encode("utf-8")
            ).hexdigest(),
            "nvidia_smi_stderr_sha256": nvidia_smi_probe["stderr_sha256"],
            "nvidia_smi_l_exit_code": nvidia_smi_l_probe["exit_code"],
            "nvidia_smi_l_stdout_sha256": hashlib.sha256(
                nvidia_smi_l_probe["stdout"].encode("utf-8")
            ).hexdigest(),
            "nvidia_smi_l_stderr_sha256": nvidia_smi_l_probe["stderr_sha256"],
            "mig_instance_count": len(mig_instances),
            "mig_instances": mig_instances,
            "mig_instances_sha256": hashlib.sha256(
                canonical_json_bytes(mig_instances)
            ).hexdigest(),
            "gpus": gpu_rows,
            "gpu_query_exit_code": gpu_probe["exit_code"],
            "gpu_query_stderr_sha256": gpu_probe["stderr_sha256"],
            "gpu_compute_process_query_exit_code": gpu_process_probe["exit_code"],
            "gpu_compute_process_query_stderr_sha256": gpu_process_probe[
                "stderr_sha256"
            ],
            "gpu_compute_process_count": len(gpu_process_lines),
            "gpu_compute_processes": gpu_process_records,
            "gpu_compute_process_metadata_sha256": hashlib.sha256(
                canonical_json_bytes(gpu_process_records)
            ).hexdigest(),
            "gpu_process_policy": "no process killed; any card with sufficient free memory may be used per user authorization",
            "current_user_process_count": len(process_lines),
            "current_user_processes": process_records,
            "current_user_process_query_exit_code": process_probe["exit_code"],
            "current_user_process_query_stderr_sha256": process_probe["stderr_sha256"],
            "current_user_process_metadata_sha256": hashlib.sha256(
                canonical_json_bytes(process_records)
            ).hexdigest(),
        },
        "inventory": {
            "project_data": {
                "data": top_level_inventory(data_root),
                "data_registry": top_level_inventory(data_registry_root),
            },
            "existing_artifacts": {
                "main_repo_artifacts": top_level_inventory(artifacts_root),
                "fm0_closure": top_level_inventory(fm0),
                "mnt_mrna_editflow_roots": top_level_inventory(
                    mnt_root, name_prefix="mrna_editflow"
                ),
            },
        },
        "upstream": {
            "fm0_closure_root": str(fm0),
            "fm0_checksum_ledger_sha256": sha256_file(fm0_ledger),
            "d1_canonical_records_path": str(args.d1_data),
            "d1_canonical_records_size_bytes": args.d1_data.stat().st_size,
            "d1_canonical_records_sha256": sha256_file(args.d1_data),
            "d1_exposure_ledger_path": str(args.d1_ledger),
            "d1_exposure_ledger_size_bytes": args.d1_ledger.stat().st_size,
            "d1_exposure_ledger_sha256": sha256_file(args.d1_ledger),
        },
        "safety": {
            "unrelated_processes_killed": 0,
            "existing_results_overwritten": 0,
            "final_labels_read": False,
            "neural_forward_executed": False,
            "downstream_stage_started": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    with output.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    print(hashlib.sha256(data).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
