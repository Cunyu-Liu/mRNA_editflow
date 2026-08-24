#!/usr/bin/env python3
"""Create exact Critic V4 confirmation authorization after terminal screen PASS."""

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

from scripts.route_a_v3.train_route2_xeditcritic_v4 import (
    require_screen_launch_authorization_v4,
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


def build_critic_confirmation_authorization_v4(
    screen_config: Mapping[str, Any],
    screen_authorization: Mapping[str, Any],
    preflight: Mapping[str, Any],
    screen_gate: Mapping[str, Any],
    *,
    current_git_head: str,
) -> dict[str, Any]:
    physical_batch = int(preflight.get("selected_physical_batch", -1))
    for run_id in ("v4_full", "c0_v4"):
        require_screen_launch_authorization_v4(
            screen_config,
            screen_authorization,
            preflight,
            run_id=run_id,
            physical_batch_size=physical_batch,
            current_git_head=current_git_head,
        )
    _require(
        screen_gate.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_screen_gate.v1"
        and screen_gate.get("status") == "XEDITCRITIC_V4_SCREEN_PASS"
        and screen_gate.get("passed") is True
        and screen_gate.get("confirmation_authorized") is True
        and screen_gate.get("development_test_authorized") is False,
        "Critic V4 screen gate does not authorize confirmation",
    )
    for payload, label in (
        (screen_authorization, "screen authorization"),
        (preflight, "preflight"),
        (screen_gate, "screen gate"),
    ):
        _require(
            int(payload.get("development_test_outcome_reads", -1)) == 0
            and int(payload.get("new_final_evaluation_outcome_reads", -1)) == 0,
            f"Critic V4 {label} reports a protected read",
        )
    barriers = screen_authorization["barriers"]
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_confirmation_launch_authorization.v1",
        "status": "XEDITCRITIC_V4_CONFIRMATION_LAUNCH_AUTHORIZED",
        "authorized_git_head": current_git_head,
        "authorized_seeds": [20260908, 20260909, 20260910],
        "authorized_run_ids": ["v4_full", "c0_v4"],
        "barriers": {
            "screen_gate_passed": True,
            "a100_current_head_focused_tests_passed": barriers[
                "a100_current_head_focused_tests_passed"
            ],
            "a100_current_head_v332_tests_passed": barriers[
                "a100_current_head_v332_tests_passed"
            ],
            "bottom_six_cache_terminal_complete": barriers[
                "bottom_six_cache_terminal_complete"
            ],
            "formal_parameter_preflight_passed": barriers[
                "formal_parameter_preflight_passed"
            ],
            "formal_memory_preflight_passed": barriers[
                "formal_memory_preflight_passed"
            ],
        },
        "additional_seed_authorized": False,
        "development_test_authorized": False,
        "guidance_authorized": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-config", required=True, type=Path)
    parser.add_argument("--screen-authorization", required=True, type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--screen-gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    _require(not arguments.output.exists(), f"Critic V4 confirmation authorization exists: {arguments.output}")
    result = build_critic_confirmation_authorization_v4(
        _read(arguments.screen_config),
        _read(arguments.screen_authorization),
        _read(arguments.preflight),
        _read(arguments.screen_gate),
        current_git_head=_git_head(),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    partial = arguments.output.with_suffix(arguments.output.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, arguments.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
