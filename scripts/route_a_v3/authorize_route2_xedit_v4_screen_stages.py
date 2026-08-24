#!/usr/bin/env python3
"""Create the missing preflight and screen-launch authorizations for V4."""

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


class XEditV4ScreenAuthorizationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditV4ScreenAuthorizationError(message)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"authorization input is not an object: {path}")
    return payload


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_common_barriers(
    c3_reference: Mapping[str, Any],
    a100_audit: Mapping[str, Any],
    *,
    current_git_head: str,
) -> None:
    _require(
        c3_reference.get("status") == "C3_V4_REFERENCE_READ_ONCE_COMPLETE"
        and int(c3_reference.get("terminal_summaries_read_count", -1)) == 5
        and c3_reference.get("c3_terminal_artifacts_retained") is True,
        "C3 read-once terminal package is absent",
    )
    _require(
        int(c3_reference.get("development_test_outcome_reads", -1)) == 0
        and int(c3_reference.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "C3 read-once package reports a protected outcome read",
    )
    sync = a100_audit.get("repository_sync", {})
    verification = a100_audit.get("a100_current_head_verification", {})
    _require(
        str(sync.get("head_after")) == str(current_git_head)
        and sync.get("old_launch_jobs_active_before_sync") is False
        and sync.get("shared_history_rewritten") is False,
        "A100 is not synchronized to the exact current HEAD after old jobs terminal",
    )
    _require(
        int(verification.get("critic_focused_failed", -1)) == 0
        and int(verification.get("setflow_focused_failed", -1)) == 0
        and int(verification.get("exact_v332_failed", -1)) == 0
        and int(verification.get("exact_v332_passed", -1)) == 96,
        "A100 current-HEAD focused or V3.3.2 tests did not pass",
    )
    protected = a100_audit.get("protected_data", {})
    _require(
        protected.get("development_test_outcomes_accessed") is False
        and protected.get("new_final_evaluation_outcomes_accessed") is False,
        "A100 synchronization audit reports a protected outcome read",
    )


def _require_cache(
    component: str,
    cache: Mapping[str, Any],
) -> None:
    if component == "critic":
        _require(
            cache.get("schema_version")
            == "route_a_v3_route2_frozen_bottom_encoder_chunk_cache_summary.v4"
            and cache.get("status") == "XEDITCRITIC_V4_BOTTOM_SIX_CACHE_COMPLETE"
            and int(cache.get("record_count", -1)) == 107873
            and int(cache.get("raw_sequence_payload_written", -1)) == 0
            and int(cache.get("label_or_outcome_payload_written", -1)) == 0,
            "Critic V4 bottom-six cache is not terminal and isolated",
        )
    else:
        _require(
            cache.get("schema_version")
            == "route_a_v3_route2_setflow_source_token_cache_summary.v3"
            and cache.get("status") == "XEDITSETFLOW_V3_SOURCE_TOKEN_CACHE_COMPLETE"
            and int(cache.get("raw_sequence_payload_written", -1)) == 0
            and int(cache.get("outcome_value_access_count", -1)) == 0,
            "SetFlow source-token cache is not terminal and isolated",
        )
    _require(
        cache.get("development_test_outcomes_accessed") is False
        and cache.get("evaluation_outcomes_accessed") is False,
        f"{component} cache reports a protected outcome read",
    )


def build_preflight_authorization_v4(
    component: str,
    c3_reference: Mapping[str, Any],
    a100_audit: Mapping[str, Any],
    cache: Mapping[str, Any],
    *,
    current_git_head: str,
) -> dict[str, Any]:
    _require(component in {"critic", "setflow"}, "unknown V4 component")
    _require_common_barriers(
        c3_reference, a100_audit, current_git_head=current_git_head
    )
    _require_cache(component, cache)
    prefix = "XEDITCRITIC" if component == "critic" else "XEDITSETFLOW"
    cache_barrier = (
        "bottom_six_cache_terminal_complete"
        if component == "critic"
        else "source_token_cache_terminal_complete"
    )
    return {
        "schema_version": f"route_a_v3_route2_xedit{component}_v4_preflight_authorization.v1",
        "status": f"{prefix}_V4_PREFLIGHT_AUTHORIZED",
        "authorized_git_head": current_git_head,
        "component": component,
        "barriers": {
            "all_five_c3_jobs_terminal": True,
            "c3_terminal_summaries_read_exactly_once": True,
            "a100_current_head_focused_tests_passed": True,
            "a100_current_head_v332_tests_passed": True,
            cache_barrier: True,
        },
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def build_screen_launch_authorization_v4(
    component: str,
    screen_config: Mapping[str, Any],
    c3_reference: Mapping[str, Any],
    a100_audit: Mapping[str, Any],
    cache: Mapping[str, Any],
    preflight: Mapping[str, Any],
    source_data_audit: Mapping[str, Any] | None,
    *,
    current_git_head: str,
) -> dict[str, Any]:
    _require_common_barriers(
        c3_reference, a100_audit, current_git_head=current_git_head
    )
    _require_cache(component, cache)
    run_ids = [str(row["run_id"]) for row in screen_config["required_screen_runs"]]
    common = {
        "all_five_c3_jobs_terminal": True,
        "c3_terminal_summaries_read_exactly_once": True,
        "a100_current_head_focused_tests_passed": True,
        "a100_current_head_v332_tests_passed": True,
    }
    if component == "critic":
        _require(
            preflight.get("status") == "XEDITCRITIC_V4_PREFLIGHT_PASS"
            and preflight.get("passed") is True
            and str(preflight.get("git_head")) == current_git_head
            and 165_000_000
            <= int(preflight.get("trainable_parameter_count", -1))
            <= 175_000_000
            and 20.0
            <= float(preflight.get("selected_peak_allocated_gib", -1))
            <= 35.0
            and int(preflight.get("selected_physical_batch", -1)) in {4, 8, 16, 32},
            "Critic V4 formal parameter/memory preflight did not pass",
        )
        barriers = {
            **common,
            "bottom_six_cache_terminal_complete": True,
            "formal_parameter_preflight_passed": True,
            "formal_memory_preflight_passed": True,
        }
        schema = "route_a_v3_route2_xeditcritic_v4_screen_launch_authorization.v1"
        status = "XEDITCRITIC_V4_SCREEN_LAUNCH_AUTHORIZED"
    else:
        _require(
            preflight.get("status") == "XEDITSETFLOW_V4_PREFLIGHT_PASS"
            and preflight.get("passed") is True
            and str(preflight.get("git_head")) == current_git_head
            and int(preflight.get("full_trainable_parameter_count", -1))
            == int(screen_config["architecture"]["formal_full_trainable_parameter_count"])
            and int(preflight.get("single_mode_trainable_parameter_count", -1))
            == int(screen_config["architecture"]["formal_single_mode_trainable_parameter_count"]),
            "SetFlow V4 formal parameter preflight did not pass",
        )
        _require(
            source_data_audit is not None
            and source_data_audit.get("status")
            == "XEDITSETFLOW_V4_SOURCE_LEVEL_DATA_AUDIT_PASS"
            and int(source_data_audit.get("validation_source_count", -1)) == 891,
            "SetFlow V4 source-level data audit did not pass",
        )
        barriers = {
            **common,
            "source_token_cache_terminal_complete": True,
            "source_level_data_audit_passed": True,
            "formal_parameter_preflight_passed": True,
        }
        schema = "route_a_v3_route2_xeditsetflow_v4_screen_launch_authorization.v1"
        status = "XEDITSETFLOW_V4_SCREEN_LAUNCH_AUTHORIZED"
    for payload, label in (
        (preflight, "preflight"),
        (source_data_audit, "source data audit"),
    ):
        if payload is not None:
            _require(
                int(payload.get("development_test_outcome_reads", -1)) == 0
                and int(payload.get("new_final_evaluation_outcome_reads", -1)) == 0,
                f"{component} {label} reports a protected outcome read",
            )
    return {
        "schema_version": schema,
        "status": status,
        "authorized_git_head": current_git_head,
        "authorized_run_ids": run_ids,
        "barriers": barriers,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("critic", "setflow"), required=True)
    parser.add_argument("--stage", choices=("preflight", "screen"), required=True)
    parser.add_argument("--screen-config", type=Path, required=True)
    parser.add_argument("--c3-reference", type=Path, required=True)
    parser.add_argument("--a100-audit", type=Path, required=True)
    parser.add_argument("--cache-summary", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--source-data-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    _require(not arguments.output.exists(), f"authorization output exists: {arguments.output}")
    head = _git_head()
    common = (
        arguments.component,
        _read(arguments.c3_reference),
        _read(arguments.a100_audit),
        _read(arguments.cache_summary),
    )
    if arguments.stage == "preflight":
        result = build_preflight_authorization_v4(*common, current_git_head=head)
    else:
        _require(arguments.preflight is not None, "screen authorization requires preflight")
        result = build_screen_launch_authorization_v4(
            arguments.component,
            _read(arguments.screen_config),
            *common[1:],
            _read(arguments.preflight),
            None
            if arguments.source_data_audit is None
            else _read(arguments.source_data_audit),
            current_git_head=head,
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    partial = arguments.output.with_suffix(arguments.output.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, arguments.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
