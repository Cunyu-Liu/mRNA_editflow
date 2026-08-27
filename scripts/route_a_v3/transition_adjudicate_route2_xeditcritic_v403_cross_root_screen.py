#!/usr/bin/env python3
"""Adjudicate the repaired Critic V4 eight-arm package across three roots."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


TRAINING_GIT_HEAD = "f34ab7d865bb2477bfe24c1d0a7c9f5301a24cea"
C0_GIT_HEAD = "93703adec7a4c76b4466d3aaae8684620bee985a"
TRAINING_WORKTREE = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/"
    "route_a_v3_route2_v403_critic_rng_replay_20260827"
)
if str(TRAINING_WORKTREE) not in sys.path:
    sys.path.insert(0, str(TRAINING_WORKTREE))

from core.route2_xeditcritic_gate_v4 import evaluate_xeditcritic_v4_screen


ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
FROZEN_CONFIG = (
    TRAINING_WORKTREE
    / "configs/route_a_v3_route2_xeditcritic_v4_screen_v1.json"
)
HISTORICAL_C0_OUTPUT_ROOT = (
    ROOT
    / "experiments/xeditcritic_v4/"
    f"screen_seed_20260907_v402_recovery_runner_{C0_GIT_HEAD}"
)
CURRENT_FULL_OUTPUT_ROOT = (
    ROOT
    / "experiments/xeditcritic_v4/"
    f"screen_seed_20260907_v403_rng_replay_fix_{TRAINING_GIT_HEAD}"
)
CURRENT_FULL_RUNTIME = (
    ROOT
    / "experiments/xeditcritic_v4/"
    f"v403_rng_replay_fix_runner_{TRAINING_GIT_HEAD}/runtime.json"
)
CONTROL_OUTPUT_ROOT = (
    ROOT
    / "experiments/xeditcritic_v4/"
    f"screen_seed_20260907_v403_control_recovery_{TRAINING_GIT_HEAD}"
)
CONTROL_RUNTIME = (
    ROOT
    / "experiments/xeditcritic_v4/"
    f"v403_control_recovery_runner_{TRAINING_GIT_HEAD}/runtime.json"
)
LEGACY_GATE = (
    ROOT
    / "experiments/xeditcritic_v4/screen_seed_20260907/screen_gate.json"
)
NEW_GATE = (
    ROOT
    / "experiments/xeditcritic_v4/"
    f"screen_seed_20260907_v403_cross_root_{TRAINING_GIT_HEAD}/screen_gate.json"
)

ARM_ORDER = (
    "c0_v4",
    "v4_full",
    "v4_source_only",
    "v4_edit_metadata_only",
    "v4_no_candidate_sequence",
    "v4_candidate_bundle_permutation",
    "v4_no_cross",
    "v4_no_moe",
)
CONTROL_RUN_IDS = ARM_ORDER[2:]


@dataclass(frozen=True)
class ArmSource:
    summary_path: Path
    source_role: str


def default_arm_sources() -> dict[str, ArmSource]:
    result = {
        "c0_v4": ArmSource(
            HISTORICAL_C0_OUTPUT_ROOT / "c0_v4/run_summary.json",
            "HISTORICAL_MATCHED_C0_TERMINAL_SUMMARY",
        ),
        "v4_full": ArmSource(
            CURRENT_FULL_OUTPUT_ROOT / "v4_full/run_summary.json",
            "CURRENT_V403_REPAIRED_FULL_TERMINAL_SUMMARY",
        ),
    }
    result.update(
        {
            run_id: ArmSource(
                CONTROL_OUTPUT_ROOT / run_id / "run_summary.json",
                "V403_REPAIRED_CONTROL_TERMINAL_SUMMARY",
            )
            for run_id in CONTROL_RUN_IDS
        }
    )
    return result


class XEditCriticV403CrossRootAdjudicationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticV403CrossRootAdjudicationError(message)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def require_zero_protected_reads(payload: Mapping[str, Any], label: str) -> None:
    require(
        int(payload.get("development_test_outcome_reads", -1)) == 0,
        f"{label} reports a Development TEST read",
    )
    require(
        int(payload.get("new_final_evaluation_outcome_reads", -1)) == 0,
        f"{label} reports a new Evaluation read",
    )


def write_atomic_once(path: Path, payload: Mapping[str, Any]) -> None:
    require(not path.exists(), "new cross-root Critic V4 screen gate already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    require(
        not partial.exists(),
        "partial cross-root Critic V4 screen gate already exists",
    )
    partial.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def validate_full_runtime(path: Path = CURRENT_FULL_RUNTIME) -> dict[str, Any]:
    runtime = read_json(path)
    require(
        runtime.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v403_full_recovery_runtime.v1"
        and runtime.get("status")
        == "XEDITCRITIC_V403_FULL_RECOVERY_TERMINAL"
        and runtime.get("git_head") == TRAINING_GIT_HEAD
        and runtime.get("run_id") == "v4_full"
        and runtime.get("terminal_artifact_kind") == "SUMMARY"
        and int(runtime.get("return_code", -1)) == 0,
        "current V4.0.3 full runtime is not exact successful terminal",
    )
    require_zero_protected_reads(runtime, "current full runtime")
    return runtime


def validate_control_runtime(path: Path = CONTROL_RUNTIME) -> dict[str, Any]:
    runtime = read_json(path)
    require(
        runtime.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v403_control_recovery_runtime.v1"
        and runtime.get("status")
        == "XEDITCRITIC_V403_CONTROL_RECOVERY_ALL_SIX_SUMMARIES_TERMINAL"
        and runtime.get("training_code_git_head") == TRAINING_GIT_HEAD,
        "V4.0.3 repaired controls are not all exact terminal summaries",
    )
    jobs = runtime.get("jobs")
    require(
        isinstance(jobs, Mapping)
        and set(jobs) == set(CONTROL_RUN_IDS)
        and runtime.get("ordered_control_run_ids") == list(CONTROL_RUN_IDS),
        "V4.0.3 control runtime arm set or order changed",
    )
    require(
        all(
            jobs[run_id].get("status") == "TERMINAL_SUMMARY"
            and jobs[run_id].get("terminal_artifact_kind") == "SUMMARY"
            and int(jobs[run_id].get("return_code", -1)) == 0
            for run_id in CONTROL_RUN_IDS
        ),
        "one or more V4.0.3 controls lack exact successful SUMMARY terminal",
    )
    require(
        runtime.get("full_retrained") is False
        and runtime.get("c0_retrained") is False
        and runtime.get("old_v402_stopped_process_resumed") is False
        and runtime.get("free_memory_gate_applied") is False
        and int(runtime.get("terminal_artifact_payloads_read_by_scheduler", -1))
        == 0,
        "V4.0.3 control runtime violates a recovery isolation boundary",
    )
    require_zero_protected_reads(runtime, "V4.0.3 control runtime")
    return runtime


def collect_arm_summaries(
    config: Mapping[str, Any],
    arm_sources: Mapping[str, ArmSource],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    frozen_order = tuple(
        str(row["run_id"]) for row in config["required_screen_runs"]
    )
    require(frozen_order == ARM_ORDER, "frozen Critic eight-arm order changed")
    require(
        tuple(arm_sources) == ARM_ORDER,
        "cross-root source map is not the exact ordered eight-arm package",
    )
    summaries: dict[str, dict[str, Any]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for run_id in ARM_ORDER:
        source = arm_sources[run_id]
        summary_path = source.summary_path
        failure_path = summary_path.with_name("failure.json")
        require(
            summary_path.is_file() and not failure_path.exists(),
            f"{run_id} is not exact terminal SUMMARY",
        )
        summary = read_json(summary_path)
        require(
            summary.get("schema_version")
            == "route_a_v3_route2_xeditcritic_v4_screen_run.v1"
            and summary.get("status")
            == "TERMINAL_XEDITCRITIC_V4_SCREEN_RUN_COMPLETE"
            and summary.get("run_id") == run_id,
            f"{run_id} terminal summary identity is invalid",
        )
        require_zero_protected_reads(summary, f"{run_id} terminal summary")
        authorization_path_value = summary.get("launch_authorization_path")
        require(
            isinstance(authorization_path_value, str)
            and bool(authorization_path_value.strip()),
            f"{run_id} launch authorization path is absent",
        )
        authorization_path = Path(authorization_path_value)
        require(
            authorization_path.is_file(),
            f"{run_id} launch authorization is absent",
        )
        authorization = read_json(authorization_path)
        expected_head = C0_GIT_HEAD if run_id == "c0_v4" else TRAINING_GIT_HEAD
        authorized_run_ids = authorization.get("authorized_run_ids")
        require(
            authorization.get("schema_version")
            == "route_a_v3_route2_xeditcritic_v4_screen_launch_authorization.v1"
            and authorization.get("status")
            == "XEDITCRITIC_V4_SCREEN_LAUNCH_AUTHORIZED"
            and authorization.get("authorized_git_head") == expected_head
            and isinstance(authorized_run_ids, list)
            and run_id in authorized_run_ids,
            f"{run_id} launch authorization identity is invalid",
        )
        require_zero_protected_reads(
            authorization, f"{run_id} launch authorization"
        )
        verified_head = str(authorization["authorized_git_head"])
        summaries[run_id] = summary
        provenance[run_id] = {
            "summary_path": str(summary_path),
            "training_git_head": verified_head,
            "source_role": source.source_role,
            "launch_authorization_path": str(authorization_path),
            "launch_authorization_schema_version": str(
                authorization["schema_version"]
            ),
            "launch_authorization_status": str(authorization["status"]),
            "authorized_git_head": verified_head,
            "run_id_authorization_verified": True,
            "authorization_protected_outcome_reads_verified_zero": True,
        }
    return summaries, provenance


def run(
    *,
    config_path: Path = FROZEN_CONFIG,
    arm_sources: Mapping[str, ArmSource] | None = None,
    full_runtime_path: Path = CURRENT_FULL_RUNTIME,
    control_runtime_path: Path = CONTROL_RUNTIME,
    legacy_gate_path: Path = LEGACY_GATE,
    output_path: Path = NEW_GATE,
) -> dict[str, Any]:
    require(
        output_path != legacy_gate_path,
        "cross-root gate output would overwrite the legacy gate",
    )
    require(not output_path.exists(), "new cross-root Critic V4 screen gate already exists")
    require(
        not output_path.with_suffix(output_path.suffix + ".partial").exists(),
        "partial cross-root Critic V4 screen gate already exists",
    )
    config = read_json(config_path)
    require(
        config.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_screen_config.v1"
        and config.get("status")
        == "FROZEN_BEFORE_V4_PARAMETER_UPDATE_OR_VALIDATION_OUTCOME_READ",
        "frozen Critic V4 screen config identity changed",
    )
    full_runtime = validate_full_runtime(full_runtime_path)
    control_runtime = validate_control_runtime(control_runtime_path)
    summaries, provenance = collect_arm_summaries(
        config,
        default_arm_sources() if arm_sources is None else arm_sources,
    )
    reference = read_json(Path(str(config["c3_read_once_reference_adjudication"])))
    require(
        reference.get("status") == "C3_V4_REFERENCE_READ_ONCE_COMPLETE"
        and int(reference.get("terminal_summaries_read_count", -1)) == 5,
        "C3 read-once reference adjudication is absent or invalid",
    )
    require_zero_protected_reads(reference, "C3 read-once reference")
    preflight = read_json(Path(str(config["preflight_output"])))
    require_zero_protected_reads(preflight, "frozen Critic preflight")

    result = evaluate_xeditcritic_v4_screen(
        config,
        summaries,
        c3_reference_spearman=float(
            reference["c3_reference_task_macro_spearman"]
        ),
        preflight=preflight,
    )
    result["cross_root_transition"] = {
        "schema_version": (
            "route_a_v3_route2_xeditcritic_v403_cross_root_input.v1"
        ),
        "ordered_run_ids": list(ARM_ORDER),
        "arm_sources": provenance,
        "historical_c0_git_head": C0_GIT_HEAD,
        "repaired_full_and_controls_git_head": TRAINING_GIT_HEAD,
        "full_runtime_path": str(full_runtime_path),
        "full_runtime_status": full_runtime["status"],
        "control_runtime_path": str(control_runtime_path),
        "control_runtime_status": control_runtime["status"],
        "frozen_config_path": str(config_path),
        "legacy_gate_path": str(legacy_gate_path),
        "legacy_gate_preserved": True,
        "terminal_summary_payloads_read": 8,
        "full_retrained": False,
        "c0_retrained": False,
        "old_v402_stopped_process_resumed": False,
        "scientific_thresholds_changed": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    result["development_test_outcome_reads"] = 0
    result["new_final_evaluation_outcome_reads"] = 0
    write_atomic_once(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
