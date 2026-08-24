#!/usr/bin/env python3
"""Authorize the exact Critic V4 refit or LOSO runtime package."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_xeditcritic_gate_v4 import CONFIRMATION_SEEDS_V4, LOSO_STUDIES_V4
from scripts.route_a_v3.prepare_route2_xeditcritic_v4_posttest_configs import (
    require_v4_posttest_authority,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"artifact is not an object: {path}")
    return payload


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_posttest_authorization_v4(
    protocol: Mapping[str, Any],
    *,
    stage: str,
    current_git_head: str,
    refit_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    require_v4_posttest_authority(protocol)
    _require(stage in {"REFIT", "LOSO"}, "Critic V4 posttest stage changed")
    if stage == "LOSO":
        _require(
            isinstance(refit_manifest, Mapping)
            and refit_manifest.get("status")
            == "XEDITCRITIC_V4_ALL_DEVELOPMENT_REFIT_COMPLETE"
            and refit_manifest.get("required_seeds") == list(CONFIRMATION_SEEDS_V4)
            and int(refit_manifest.get("completed_refit_count", -1)) == 3
            and int(refit_manifest.get("refit_pass_count", -1)) == 8
            and refit_manifest.get("loso_authorized") is True
            and refit_manifest.get("development_test_outcomes_accessed_during_refit")
            is False
            and refit_manifest.get("new_final_evaluation_outcomes_accessed") is False,
            "Critic V4 LOSO authorization lacks all three refits",
        )
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_posttest_launch_authorization.v1",
        "status": f"XEDITCRITIC_V4_{stage}_LAUNCH_AUTHORIZED",
        "authorized_git_head": current_git_head,
        "authorized_stage": stage,
        "authorized_seeds": list(CONFIRMATION_SEEDS_V4),
        "authorized_run_ids": ["v4_full"]
        if stage == "REFIT"
        else ["v4_full", "c0_v4"],
        "authorized_held_out_studies": []
        if stage == "REFIT"
        else list(LOSO_STUDIES_V4),
        "atomic_frozen_test_passed": True,
        "all_three_refits_complete": stage == "LOSO",
        "development_test_access_event_count_before_posttest": 1,
        "development_test_outcome_reads_during_posttest": 0,
        "new_final_evaluation_outcome_reads": 0,
        "guidance_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("REFIT", "LOSO"))
    parser.add_argument("--refit-manifest", type=Path)
    arguments = parser.parse_args()
    protocol = _read(arguments.protocol)
    output = Path(
        protocol[
            "all_development_refit"
            if arguments.stage == "REFIT"
            else "test_preserving_loso"
        ]["authorization_output"]
    )
    _require(not output.exists(), f"Critic V4 posttest authorization exists: {output}")
    refit = None
    if arguments.stage == "LOSO":
        _require(arguments.refit_manifest is not None, "Critic V4 refit manifest is absent")
        refit = _read(arguments.refit_manifest)
    result = build_posttest_authorization_v4(
        protocol,
        stage=arguments.stage,
        current_git_head=_git_head(),
        refit_manifest=refit,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
