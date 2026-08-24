#!/usr/bin/env python3
"""Create the one exact SetFlow V4 confirmation launch authorization after PASS."""

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

from core.route2_xeditsetflow_runtime_v4 import (
    require_setflow_v4_screen_launch_authorization,
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


def build_confirmation_authorization_v4(
    screen_config: Mapping[str, Any],
    screen_authorization: Mapping[str, Any],
    preflight: Mapping[str, Any],
    source_data_audit: Mapping[str, Any],
    screen_gate: Mapping[str, Any],
    *,
    current_git_head: str,
) -> dict[str, Any]:
    for run_id in ("v4_full", "v4_single_mode"):
        require_setflow_v4_screen_launch_authorization(
            screen_config,
            screen_authorization,
            preflight,
            source_data_audit,
            run_id=run_id,
            current_git_head=current_git_head,
        )
    _require(
        screen_gate.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_screen_gate.v1"
        and screen_gate.get("status") == "XEDITSETFLOW_V4_SCREEN_PASS"
        and screen_gate.get("confirmation_authorized") is True
        and screen_gate.get("confirmation_seeds")
        == [20260912, 20260913, 20260914]
        and screen_gate.get("additional_seed_authorized") is False,
        "SetFlow V4 screen gate does not authorize exact confirmation",
    )
    for payload, label in (
        (screen_authorization, "screen authorization"),
        (preflight, "preflight"),
        (source_data_audit, "source data audit"),
        (screen_gate, "screen gate"),
    ):
        _require(
            int(payload.get("development_test_outcome_reads", -1)) == 0
            and int(payload.get("new_final_evaluation_outcome_reads", -1)) == 0,
            f"SetFlow V4 {label} reports a protected read",
        )
    screen_barriers = screen_authorization["barriers"]
    return {
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_confirmation_launch_authorization.v1",
        "status": "XEDITSETFLOW_V4_CONFIRMATION_LAUNCH_AUTHORIZED",
        "authorized_git_head": current_git_head,
        "authorized_seeds": [20260912, 20260913, 20260914],
        "authorized_run_id": "v4_full",
        "barriers": {
            "screen_gate_passed": True,
            "a100_current_head_focused_tests_passed": screen_barriers[
                "a100_current_head_focused_tests_passed"
            ],
            "a100_current_head_v332_tests_passed": screen_barriers[
                "a100_current_head_v332_tests_passed"
            ],
            "source_token_cache_terminal_complete": screen_barriers[
                "source_token_cache_terminal_complete"
            ],
            "source_level_data_audit_passed": screen_barriers[
                "source_level_data_audit_passed"
            ],
            "formal_parameter_preflight_passed": screen_barriers[
                "formal_parameter_preflight_passed"
            ],
        },
        "screen_selected_checkpoint_pass": screen_gate[
            "selected_checkpoint_pass"
        ],
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
    parser.add_argument("--source-data-audit", required=True, type=Path)
    parser.add_argument("--screen-gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    _require(not arguments.output.exists(), f"confirmation authorization exists: {arguments.output}")
    result = build_confirmation_authorization_v4(
        _read(arguments.screen_config),
        _read(arguments.screen_authorization),
        _read(arguments.preflight),
        _read(arguments.source_data_audit),
        _read(arguments.screen_gate),
        current_git_head=_git_head(),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    partial = arguments.output.with_suffix(arguments.output.suffix + ".partial")
    partial.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, arguments.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
