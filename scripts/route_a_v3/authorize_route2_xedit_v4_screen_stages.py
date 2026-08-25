#!/usr/bin/env python3
"""Create the missing preflight and screen-launch authorizations for V4."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_bottom_encoder_chunk_cache_v4 import (
    require_frozen_bottom_encoder_chunk_cache_identity_receipt_v4,
)
from core.route2_source_token_cache_v3 import (
    require_source_token_cache_identity_receipt_v3,
)


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
        and sync.get("remote_worktree_clean_after") is True
        and sync.get("shared_history_rewritten") is False,
        "A100 is not synchronized to the exact current HEAD after old jobs terminal",
    )
    _require(
        str(verification.get("verified_git_head")) == str(current_git_head)
        and int(verification.get("critic_focused_total_passed", 0)) > 0
        and int(verification.get("critic_focused_failed", -1)) == 0
        and int(verification.get("setflow_focused_passed", 0)) > 0
        and int(verification.get("setflow_focused_failed", -1)) == 0
        and int(verification.get("exact_v332_failed", -1)) == 0
        and int(verification.get("exact_v332_passed", -1)) == 96,
        "A100 current-HEAD focused or V3.3.2 tests did not run and pass",
    )
    protected = a100_audit.get("protected_data", {})
    _require(
        protected.get("development_test_outcomes_accessed") is False
        and protected.get("new_final_evaluation_outcomes_accessed") is False,
        "A100 synchronization audit reports a protected outcome read",
    )


def build_cache_launch_authorization_v4(
    component: str,
    c3_reference: Mapping[str, Any],
    a100_audit: Mapping[str, Any],
    *,
    current_git_head: str,
) -> dict[str, Any]:
    """Authorize one cache build only after C3 read-once and A100 current-HEAD tests."""

    _require(component in {"critic", "setflow"}, "unknown V4 component")
    _require_common_barriers(
        c3_reference, a100_audit, current_git_head=current_git_head
    )
    prefix = "XEDITCRITIC" if component == "critic" else "XEDITSETFLOW"
    return {
        "schema_version": (
            f"route_a_v3_route2_xedit{component}_v4_cache_launch_authorization.v1"
        ),
        "status": f"{prefix}_V4_CACHE_LAUNCH_AUTHORIZED",
        "authorized_git_head": current_git_head,
        "component": component,
        "barriers": {
            "all_five_c3_jobs_terminal": True,
            "c3_terminal_summaries_read_exactly_once": True,
            "a100_current_head_focused_tests_passed": True,
            "a100_current_head_v332_tests_passed": True,
        },
        "gpu_policy": {
            "physical_gpu_scope": [0, 1, 2, 3, 4, 5],
            "cuda_bf16_only": True,
            "cpu_fallback": False,
            "cuda_visible_devices_remapping_forbidden": True,
        },
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def require_cache_launch_authorization_v4(
    component: str,
    authorization: Mapping[str, Any],
    *,
    current_git_head: str,
) -> None:
    """Fail closed before a Critic or SetFlow cache builder reads a projection."""

    _require(component in {"critic", "setflow"}, "unknown V4 component")
    prefix = "XEDITCRITIC" if component == "critic" else "XEDITSETFLOW"
    _require(
        authorization.get("schema_version")
        == f"route_a_v3_route2_xedit{component}_v4_cache_launch_authorization.v1"
        and authorization.get("status")
        == f"{prefix}_V4_CACHE_LAUNCH_AUTHORIZED"
        and authorization.get("component") == component,
        f"{component} V4 cache launch authorization is absent",
    )
    _require(
        str(authorization.get("authorized_git_head")) == str(current_git_head),
        f"{component} V4 cache authorization is for another Git HEAD",
    )
    barriers = authorization.get("barriers", {})
    required = (
        "all_five_c3_jobs_terminal",
        "c3_terminal_summaries_read_exactly_once",
        "a100_current_head_focused_tests_passed",
        "a100_current_head_v332_tests_passed",
    )
    _require(
        all(barriers.get(key) is True for key in required),
        f"{component} V4 cache launch barrier is incomplete",
    )
    _require(
        authorization.get("gpu_policy")
        == {
            "physical_gpu_scope": [0, 1, 2, 3, 4, 5],
            "cuda_bf16_only": True,
            "cpu_fallback": False,
            "cuda_visible_devices_remapping_forbidden": True,
        },
        f"{component} V4 cache authorization GPU policy changed",
    )
    _require(
        int(authorization.get("development_test_outcome_reads", -1)) == 0
        and int(authorization.get("new_final_evaluation_outcome_reads", -1)) == 0,
        f"{component} V4 cache authorization reports a protected outcome read",
    )


def require_cache_runtime_policy_v4(config: Mapping[str, Any]) -> int:
    """Return the frozen physical GPU index after validating a cache config."""

    _require(
        config.get("gpu_policy")
        == {
            "physical_gpu_scope": [0, 1, 2, 3, 4, 5],
            "cuda_bf16_only": True,
            "cpu_fallback": False,
            "cuda_visible_devices_remapping_forbidden": True,
        },
        "V4 cache GPU policy changed",
    )
    device = str(config.get("device", ""))
    _require(
        re.fullmatch(r"cuda:[0-5]", device) is not None,
        "V4 cache device is outside physical GPU 0–5",
    )
    return int(device.removeprefix("cuda:"))


def _require_cache(
    component: str,
    cache: Mapping[str, Any],
    *,
    cache_git_head: str,
) -> None:
    if component == "critic":
        _require(
            cache.get("schema_version")
            == "route_a_v3_route2_frozen_bottom_encoder_chunk_cache_summary.v4"
            and cache.get("status") == "XEDITCRITIC_V4_BOTTOM_SIX_CACHE_COMPLETE"
            and int(cache.get("record_count", -1)) == 107873
            and int(cache.get("unique_sequence_count", -1)) == 43730
            and int(cache.get("embedding_width", -1)) == 768
            and str(cache.get("model_id"))
            == "YYLY66/mRNABERT@a1eb7df25804d23f08646e1cb996b234d7208a40"
            and int(cache.get("chunk_nucleotides", -1)) == 1000
            and int(cache.get("chunk_overlap", -1)) == 64
            and int(cache.get("local_context_radius", -1)) == 32
            and cache.get("frozen_encoder_blocks") == [0, 1, 2, 3, 4, 5]
            and cache.get("trainable_encoder_blocks") == [6, 7, 8, 9, 10, 11]
            and int(cache.get("raw_sequence_payload_written", -1)) == 0
            and int(cache.get("label_or_outcome_payload_written", -1)) == 0,
            "Critic V4 bottom-six cache is not terminal and isolated",
        )
        _require(
            cache.get("development_test_outcomes_accessed") is False
            and cache.get("evaluation_outcomes_accessed") is False,
            "critic cache reports a protected outcome read",
        )
        _require(
            str(cache.get("git_head")) == str(cache_git_head)
            and cache.get("cache_launch_authorization_status")
            == "XEDITCRITIC_V4_CACHE_LAUNCH_AUTHORIZED"
            and isinstance(cache.get("physical_gpu_index"), int)
            and not isinstance(cache.get("physical_gpu_index"), bool)
            and int(cache["physical_gpu_index"]) in {0, 1, 2, 3, 4, 5}
            and bool(str(cache.get("cuda_device_name", "")))
            and cache.get("forward_precision") == "BF16"
            and cache.get("cpu_fallback") is False,
            "critic cache launch provenance is absent or stale",
        )
    else:
        _require(
            cache.get("schema_version")
            == "route_a_v3_route2_xeditsetflow_v4_source_cache_adoption_receipt.v1"
            and cache.get("status")
            == "XEDITSETFLOW_V4_SOURCE_CACHE_ADOPTED_READ_ONLY"
            and str(cache.get("git_head")) == str(cache_git_head)
            and cache.get("cache_launch_authorization_status")
            == "XEDITSETFLOW_V4_CACHE_LAUNCH_AUTHORIZED"
            and cache.get("legacy_summary_schema_version")
            == "route_a_v3_route2_setflow_source_token_cache_summary.v3"
            and cache.get("legacy_summary_status")
            == "XEDITSETFLOW_V3_SOURCE_TOKEN_CACHE_COMPLETE"
            and cache.get("legacy_artifact_policy")
            == "READ_ONLY_NO_REBUILD_NO_OVERWRITE"
            and int(cache.get("encoder_forward_count", -1)) == 0
            and int(cache.get("parameter_update_count", -1)) == 0
            and cache.get("legacy_payload_modified") is False
            and cache.get("legacy_summary_modified") is False
            and cache.get("cpu_fallback_used") is False
            and cache.get("identity_validation_map_location") == "CPU_READ_ONLY"
            and int(cache.get("development_test_outcome_reads", -1)) == 0
            and int(cache.get("new_final_evaluation_outcome_reads", -1)) == 0,
            "SetFlow source-token cache read-only adoption is absent or stale",
        )
        require_source_token_cache_identity_receipt_v3(
            cache.get("source_token_cache_identity"),
            expected_model_id="YYLY66/mRNABERT@a1eb7df25804d23f08646e1cb996b234d7208a40",
            expected_record_count=84218,
            expected_unique_source_count=19303,
            expected_token_count=2817781,
            expected_maximum_source_length=837,
            expected_embedding_width=768,
        )


def build_preflight_authorization_v4(
    component: str,
    c3_reference: Mapping[str, Any],
    a100_audit: Mapping[str, Any],
    cache: Mapping[str, Any],
    *,
    current_git_head: str,
    cache_git_head: str | None = None,
) -> dict[str, Any]:
    _require(component in {"critic", "setflow"}, "unknown V4 component")
    _require_common_barriers(
        c3_reference, a100_audit, current_git_head=current_git_head
    )
    frozen_cache_head = current_git_head if cache_git_head is None else cache_git_head
    _require_cache(component, cache, cache_git_head=frozen_cache_head)
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
        "cache_experiment_head": frozen_cache_head,
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
    preflight_git_head: str,
    cache_git_head: str | None = None,
) -> dict[str, Any]:
    _require_common_barriers(
        c3_reference, a100_audit, current_git_head=current_git_head
    )
    frozen_cache_head = current_git_head if cache_git_head is None else cache_git_head
    _require(
        re.fullmatch(r"[0-9a-f]{40}", preflight_git_head) is not None,
        "screen authorization requires an exact preflight runner HEAD",
    )
    _require_cache(component, cache, cache_git_head=frozen_cache_head)
    run_ids = [str(row["run_id"]) for row in screen_config["required_screen_runs"]]
    common = {
        "all_five_c3_jobs_terminal": True,
        "c3_terminal_summaries_read_exactly_once": True,
        "a100_current_head_focused_tests_passed": True,
        "a100_current_head_v332_tests_passed": True,
    }
    if component == "critic":
        alignment = preflight.get("cache_online_alignment", {})
        _require(
            preflight.get("status") == "XEDITCRITIC_V4_PREFLIGHT_PASS"
            and preflight.get("passed") is True
            and str(preflight.get("git_head")) == preflight_git_head
            and 165_000_000
            <= int(preflight.get("trainable_parameter_count", -1))
            <= 175_000_000
            and math.isfinite(float(preflight.get("selected_peak_allocated_gib", -1)))
            and 0.0
            < float(preflight.get("selected_peak_allocated_gib", -1))
            <= 35.0
            and int(preflight.get("selected_physical_batch", -1)) in {4, 8, 16, 32},
            "Critic V4 formal parameter/memory preflight did not pass",
        )
        _require(
            alignment.get("passed") is True
            and int(alignment.get("sequence_count", -1)) == 8
            and float(alignment.get("maximum_absolute_tolerance", -1)) == 0.02
            and float(alignment.get("mean_absolute_tolerance", -1)) == 0.005
            and float(alignment.get("maximum_absolute_difference", float("inf"))) <= 0.02
            and float(alignment.get("mean_absolute_difference", float("inf"))) <= 0.005
            and alignment.get("target_value_accessed") is False
            and alignment.get("validation_metric_read") is False,
            "Critic V4 cache/online equivalence did not pass",
        )
        require_frozen_bottom_encoder_chunk_cache_identity_receipt_v4(
            preflight.get("bottom_six_cache_identity"),
            expected_model_id="YYLY66/mRNABERT@a1eb7df25804d23f08646e1cb996b234d7208a40",
            expected_record_count=107873,
            expected_unique_sequence_count=43730,
            expected_embedding_width=768,
        )
        barriers = {
            **common,
            "bottom_six_cache_terminal_complete": True,
            "formal_parameter_preflight_passed": True,
            "formal_memory_preflight_passed": True,
            "cache_online_equivalence_passed": True,
        }
        schema = "route_a_v3_route2_xeditcritic_v4_screen_launch_authorization.v1"
        status = "XEDITCRITIC_V4_SCREEN_LAUNCH_AUTHORIZED"
    else:
        _require(
            preflight.get("status") == "XEDITSETFLOW_V4_PREFLIGHT_PASS"
            and preflight.get("passed") is True
            and str(preflight.get("git_head")) == preflight_git_head
            and int(preflight.get("full_trainable_parameter_count", -1))
            == int(screen_config["architecture"]["formal_full_trainable_parameter_count"])
            and int(preflight.get("single_mode_trainable_parameter_count", -1))
            == int(screen_config["architecture"]["formal_single_mode_trainable_parameter_count"]),
            "SetFlow V4 formal parameter preflight did not pass",
        )
        require_source_token_cache_identity_receipt_v3(
            preflight.get("source_token_cache_identity"),
            expected_model_id="YYLY66/mRNABERT@a1eb7df25804d23f08646e1cb996b234d7208a40",
            expected_record_count=84218,
            expected_unique_source_count=19303,
            expected_token_count=2817781,
            expected_maximum_source_length=837,
            expected_embedding_width=768,
        )
        _require(
            source_data_audit is not None
            and source_data_audit.get("status")
            == "XEDITSETFLOW_V4_SOURCE_LEVEL_DATA_AUDIT_PASS"
            and int(source_data_audit.get("validation_source_count", -1))
            == int(
                screen_config["data_geometry"][
                    "expected_validation_source_record_count"
                ]
            ),
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
        "preflight_runner_git_head": preflight_git_head,
        "cache_experiment_head": frozen_cache_head,
        "authorized_run_ids": run_ids,
        "barriers": barriers,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("critic", "setflow"), required=True)
    parser.add_argument("--stage", choices=("cache", "preflight", "screen"), required=True)
    parser.add_argument("--screen-config", type=Path)
    parser.add_argument("--c3-reference", type=Path, required=True)
    parser.add_argument("--a100-audit", type=Path, required=True)
    parser.add_argument("--cache-summary", type=Path)
    parser.add_argument("--cache-experiment-head")
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--preflight-runner-head")
    parser.add_argument("--source-data-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    _require(not arguments.output.exists(), f"authorization output exists: {arguments.output}")
    head = _git_head()
    c3_reference = _read(arguments.c3_reference)
    a100_audit = _read(arguments.a100_audit)
    if arguments.stage == "cache":
        result = build_cache_launch_authorization_v4(
            arguments.component,
            c3_reference,
            a100_audit,
            current_git_head=head,
        )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        partial = arguments.output.with_suffix(arguments.output.suffix + ".partial")
        partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(partial, arguments.output)
        print(json.dumps(result, sort_keys=True))
        return
    _require(arguments.screen_config is not None, "preflight/screen authorization requires a screen config")
    _require(arguments.cache_summary is not None, "preflight/screen authorization requires a cache summary")
    common = (
        arguments.component,
        c3_reference,
        a100_audit,
        _read(arguments.cache_summary),
    )
    _require(
        arguments.cache_experiment_head is not None
        and re.fullmatch(r"[0-9a-f]{40}", arguments.cache_experiment_head)
        is not None,
        "preflight/screen authorization requires a cache experiment HEAD",
    )
    if arguments.stage == "preflight":
        result = build_preflight_authorization_v4(
            *common,
            current_git_head=head,
            cache_git_head=arguments.cache_experiment_head,
        )
    else:
        _require(arguments.preflight is not None, "screen authorization requires preflight")
        _require(
            arguments.preflight_runner_head is not None
            and re.fullmatch(r"[0-9a-f]{40}", arguments.preflight_runner_head)
            is not None,
            "screen authorization requires a preflight runner HEAD",
        )
        result = build_screen_launch_authorization_v4(
            arguments.component,
            _read(arguments.screen_config),
            *common[1:],
            _read(arguments.preflight),
            None
            if arguments.source_data_audit is None
            else _read(arguments.source_data_audit),
            current_git_head=head,
            preflight_git_head=arguments.preflight_runner_head,
            cache_git_head=arguments.cache_experiment_head,
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    partial = arguments.output.with_suffix(arguments.output.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, arguments.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
