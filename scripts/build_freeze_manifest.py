"""P0-01: build the Phase-0 freeze manifest.

Freezes the audit baseline as a single JSON manifest recording:

* git commit (+ dirty working-tree state, explicitly listed — never hidden)
* every declared data path and its SHA256
* every split file SHA256
* every checkpoint SHA256
* config hash
* Python / PyTorch / CUDA / GPU environment
* seeds
* generation commands (per artifact group)
* the result dependency graph

Usage:
    python scripts/build_freeze_manifest.py --config configs/nmi_execution.yaml --strict

``--strict`` enforces zero missing files and zero unknown provenance:
the command exits 1 if any declared artifact is absent from disk or any
artifact group lacks a generation command or input declaration.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

CHUNK = 1 << 20  # 1 MiB streaming hash


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def git_info(repo_root: str) -> Dict[str, Any]:
    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", repo_root, *args],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    info: Dict[str, Any] = {}
    try:
        info["commit"] = _git("rev-parse", "HEAD")
        info["branch"] = _git("rev-parse", "--abbrev-ref", "HEAD")
        status = _git("status", "--porcelain")
        dirty_files = [l[3:] for l in status.splitlines() if l.strip()]
        info["working_tree_dirty"] = bool(dirty_files)
        info["dirty_files"] = sorted(dirty_files)
    except subprocess.CalledProcessError as e:
        info["error"] = f"git unavailable: {e}"
        info["working_tree_dirty"] = None
        info["dirty_files"] = []
    return info


def environment_info() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }
    try:
        import torch
        env["pytorch_version"] = torch.__version__
        env["cuda_version"] = torch.version.cuda
        env["cudnn_version"] = (
            ".".join(str(torch.backends.cudnn.version()))
            if torch.backends.cudnn.is_available() else None
        )
        env["gpu_names"] = (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else []
        )
    except ImportError:
        env["pytorch_version"] = None
        env["cuda_version"] = None
        env["cudnn_version"] = None
        env["gpu_names"] = []
    return env


def collect_group_files(group: Dict[str, Any], repo_root: str) -> List[str]:
    """Resolve declared files + glob patterns to a sorted unique path list."""
    paths: List[str] = []
    for f in group.get("files", []) or []:
        paths.append(f)
    for pattern in group.get("glob", []) or []:
        matched = sorted(
            os.path.relpath(p, repo_root)
            for p in glob.glob(os.path.join(repo_root, pattern))
            if os.path.isfile(p)
        )
        paths.extend(matched)
    # unique, deterministic
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return sorted(out)


def build_manifest(config_path: str, strict: bool) -> Dict[str, Any]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    repo_root = str(Path(config_path).resolve().parent.parent)
    config_sha = sha256_file(config_path)

    manifest: Dict[str, Any] = {
        "project": cfg["project"],
        "phase": cfg["phase"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generation_command": " ".join(sys.argv),
        "config_path": os.path.relpath(config_path, repo_root),
        "config_sha256": config_sha,
        "seeds": cfg.get("seeds", []),
        "git": git_info(repo_root) if cfg.get("git", {}).get("record_commit") else {},
        "environment": environment_info(),
        "artifacts": {},
        "dependency_graph": cfg.get("dependency_graph", []),
    }

    missing: List[str] = []
    unknown_prov: List[str] = []
    declared_groups = set()

    for group in cfg.get("artifact_groups", []):
        name = group["name"]
        declared_groups.add(name)
        prov = group.get("provenance", {}) or {}
        command = (prov.get("command") or "").strip()
        inputs = prov.get("inputs")
        if not command or inputs is None:
            unknown_prov.append(name)

        files = collect_group_files(group, repo_root)
        entries = []
        for rel in files:
            abspath = os.path.join(repo_root, rel)
            if not os.path.isfile(abspath):
                missing.append(rel)
                entries.append({"path": rel, "sha256": None,
                                "size_bytes": None, "missing": True})
                continue
            entries.append({
                "path": rel,
                "sha256": sha256_file(abspath),
                "size_bytes": os.path.getsize(abspath),
                "missing": False,
            })
        manifest["artifacts"][name] = {
            "description": group.get("description", ""),
            "provenance": {"command": command, "inputs": inputs or []},
            "files": entries,
        }

    # Dependency-graph consistency: every node must reference declared groups.
    graph_errors: List[str] = []
    for node in manifest["dependency_graph"]:
        for key in ("produces", "consumes"):
            for ref in node.get(key, []):
                if ref not in declared_groups:
                    graph_errors.append(f"{node.get('stage')}:{key}:{ref}")

    strict_passed = not missing and not unknown_prov and not graph_errors
    manifest["validation"] = {
        "missing_files": missing,
        "unknown_provenance": unknown_prov,
        "dependency_graph_errors": graph_errors,
        "strict": strict,
        "strict_passed": strict_passed,
    }

    out_path = os.path.join(repo_root, cfg["manifest_output"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)

    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--strict", action="store_true",
                    help="fail on any missing file or unknown provenance")
    args = ap.parse_args()

    manifest = build_manifest(args.config, args.strict)
    v = manifest["validation"]
    n_files = sum(len(g["files"]) for g in manifest["artifacts"].values())
    print(f"[freeze] wrote {manifest['phase']} manifest: "
          f"{len(manifest['artifacts'])} groups, {n_files} files hashed")
    print(f"[freeze] git commit: {manifest['git'].get('commit')} "
          f"(dirty={manifest['git'].get('working_tree_dirty')})")
    print(f"[freeze] missing={len(v['missing_files'])} "
          f"unknown_provenance={len(v['unknown_provenance'])} "
          f"graph_errors={len(v['dependency_graph_errors'])}")
    if v["missing_files"]:
        for p in v["missing_files"][:20]:
            print(f"  MISSING: {p}")
    if v["unknown_provenance"]:
        for g in v["unknown_provenance"]:
            print(f"  UNKNOWN PROVENANCE: {g}")
    if args.strict and not v["strict_passed"]:
        print("[freeze] STRICT MODE FAILED")
        return 1
    print("[freeze] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
