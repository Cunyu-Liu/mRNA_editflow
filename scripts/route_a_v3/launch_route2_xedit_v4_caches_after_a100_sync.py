#!/usr/bin/env python3
"""Authorize and launch both V4 cache stages after the exact A100 HEAD gate."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


WORKTREE = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/"
    "route_a_v3_route2_method_repair_20260817"
)
PYTHON = Path("/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10")
ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
C3_REFERENCE = (
    ROOT
    / "experiments/xeditcritic_v3/screen_seed_20260830/"
    "c3_v4_reference_read_once.json"
)
CACHE_JOB_RUNNER = (
    WORKTREE / "scripts/route_a_v3/run_route2_xedit_v4_cache_job.py"
)


class XEditV4CacheLaunchError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditV4CacheLaunchError(message)


def command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=WORKTREE,
        text=True,
        capture_output=True,
        check=True,
    )


def write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def expected_authorization_status(component: str) -> str:
    require(component in {"critic", "setflow"}, "unknown V4 cache component")
    prefix = "XEDITCRITIC" if component == "critic" else "XEDITSETFLOW"
    return f"{prefix}_V4_CACHE_LAUNCH_AUTHORIZED"


def component_paths() -> dict[str, dict[str, Path]]:
    return {
        "critic": {
            "config": WORKTREE
            / "configs/route_a_v3_route2_xeditcritic_v4_bottom_six_cache_v1.json",
            "builder": WORKTREE
            / "scripts/route_a_v3/build_route2_mrnabert_bottom_six_cache_v4.py",
            "summary": ROOT
            / "pretrained_features/xeditcritic_v4/"
            "frozen_bottom_six_chunk_cache_v1.summary.json",
            "failure": ROOT
            / "pretrained_features/xeditcritic_v4/"
            "frozen_bottom_six_chunk_cache_v1.failure.json",
        },
        "setflow": {
            "config": WORKTREE
            / "configs/route_a_v3_route2_xeditsetflow_v4_source_cache_adoption_v1.json",
            "builder": WORKTREE
            / "scripts/route_a_v3/adopt_route2_xeditsetflow_v4_source_token_cache.py",
            "summary": ROOT
            / "pretrained_features/xeditsetflow_v4/"
            "source_token_cache_v3_adoption_receipt_v1.json",
            "failure": ROOT
            / "pretrained_features/xeditsetflow_v4/"
            "source_token_cache_v3_adoption_receipt_v1.failure.json",
        },
    }


def run(head: str) -> dict[str, object]:
    require(re.fullmatch(r"[0-9a-f]{40}", head) is not None, "expected Git HEAD is invalid")
    require(PYTHON.is_file(), "formal Python is absent")
    require(CACHE_JOB_RUNNER.is_file(), "current-HEAD cache job runner is absent")
    require(
        command(["git", "rev-parse", "HEAD"]).stdout.strip() == head,
        "A100 worktree is not at expected HEAD",
    )
    require(
        not command(["git", "status", "--porcelain"]).stdout.strip(),
        "A100 worktree is dirty",
    )
    require(C3_REFERENCE.is_file(), "C3 read-once reference is absent")

    a100_audit = ROOT / f"audits/a100_current_head_v4/sync_tests_{head}.json"
    require(a100_audit.is_file(), "exact current-HEAD A100 test audit is absent")
    authorization_root = ROOT / f"authorizations/xedit_v4/cache_launch_{head}"
    authorization_staging_root = authorization_root.with_name(
        authorization_root.name + ".partial"
    )
    runtime_root = ROOT / f"experiments/xedit_v4/cache_launch_{head}"
    log_root = ROOT / f"logs/xedit_v4/cache_launch_{head}"
    require(not authorization_root.exists(), "cache launch authorizations already exist")
    require(
        not authorization_staging_root.exists(),
        "partial cache launch authorization package already exists",
    )
    require(not runtime_root.exists(), "cache launch runtime already exists")

    authorizer = (
        WORKTREE / "scripts/route_a_v3/authorize_route2_xedit_v4_screen_stages.py"
    )
    require(authorizer.is_file(), "current-HEAD cache authorizer is absent")
    components = component_paths()
    for component, paths in components.items():
        require(paths["config"].is_file(), f"{component} cache config is absent")
        require(paths["builder"].is_file(), f"{component} cache builder is absent")
        require(not paths["summary"].exists(), f"{component} cache summary already exists")
        require(not paths["failure"].exists(), f"{component} cache failure already exists")
        authorization = authorization_staging_root / f"{component}.json"
        command(
            [
                str(PYTHON),
                str(authorizer),
                "--component",
                component,
                "--stage",
                "cache",
                "--c3-reference",
                str(C3_REFERENCE),
                "--a100-audit",
                str(a100_audit),
                "--output",
                str(authorization),
            ]
        )
        require(authorization.is_file(), f"{component} cache authorization is absent")
        payload = json.loads(authorization.read_text(encoding="utf-8"))
        require(
            payload.get("status") == expected_authorization_status(component)
            and payload.get("component") == component
            and payload.get("authorized_git_head") == head,
            f"{component} cache authorization content is invalid",
        )
    os.replace(authorization_staging_root, authorization_root)

    launches: dict[str, object] = {}
    runtime_root.mkdir(parents=True)
    log_root.mkdir(parents=True)
    for component, paths in components.items():
        authorization = authorization_root / f"{component}.json"
        runtime = runtime_root / f"{component}.runtime.json"
        log = log_root / f"{component}.log"
        wrapper_log = log_root / f"{component}.wrapper.log"
        stream = wrapper_log.open("w", encoding="utf-8")
        process = subprocess.Popen(
            [
                str(PYTHON),
                str(CACHE_JOB_RUNNER),
                "--component",
                component,
                "--python",
                str(PYTHON),
                "--builder",
                str(paths["builder"]),
                "--config",
                str(paths["config"]),
                "--authorization",
                str(authorization),
                "--summary",
                str(paths["summary"]),
                "--failure",
                str(paths["failure"]),
                "--runtime",
                str(runtime),
                "--log",
                str(log),
                "--git-head",
                head,
            ],
            cwd=WORKTREE,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        stream.close()
        launches[component] = {
            "wrapper_pid": process.pid,
            "authorization": str(authorization),
            "runtime": str(runtime),
            "summary": str(paths["summary"]),
            "failure": str(paths["failure"]),
            "builder_log": str(log),
            "wrapper_log": str(wrapper_log),
        }

    manifest = runtime_root / "launch_manifest.json"
    write_atomic(
        manifest,
        {
            "schema_version": "route_a_v3_route2_xedit_v4_cache_launch_manifest.v1",
            "status": "V4_CACHE_JOBS_LAUNCHED",
            "git_head": head,
            "c3_reference": str(C3_REFERENCE),
            "a100_audit": str(a100_audit),
            "jobs": launches,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )
    return {"manifest": str(manifest), "jobs": launches}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.expected_head), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
