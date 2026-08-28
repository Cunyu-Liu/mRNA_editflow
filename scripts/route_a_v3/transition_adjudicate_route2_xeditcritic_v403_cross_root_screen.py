#!/usr/bin/env python3
"""Adjudicate the repaired Critic V4 eight-arm package across three roots."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping


TRAINING_GIT_HEAD = "f34ab7d865bb2477bfe24c1d0a7c9f5301a24cea"
C0_GIT_HEAD = "93703adec7a4c76b4466d3aaae8684620bee985a"
WORKTREE = Path(__file__).resolve().parents[2]
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from core.route2_xeditcritic_gate_v4 import evaluate_xeditcritic_v4_screen


ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
FROZEN_CONFIG = (
    WORKTREE
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
FULL_TERMINAL_AUDIT = (
    WORKTREE
    / "audits/route_a_v3_route2_xeditcritic_v403_full_terminal_v1.json"
)
LEGACY_GATE = (
    ROOT
    / "experiments/xeditcritic_v4/screen_seed_20260907/screen_gate.json"
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


def default_arm_sources(control_output_root: Path) -> dict[str, ArmSource]:
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
                control_output_root / run_id / "run_summary.json",
                "CURRENT_HEAD_CONTROL_TERMINAL_SUMMARY",
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


def validate_full_terminal_audit(
    path: Path = FULL_TERMINAL_AUDIT,
) -> dict[str, Any]:
    audit = read_json(path)
    terminal_facts = audit.get("terminal_facts")
    claim_boundary = audit.get("claim_boundary")
    require(
        audit.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v403_full_terminal.v1"
        and audit.get("status")
        == "XEDITCRITIC_V403_FULL_TERMINAL_SUMMARY_RECORDED"
        and audit.get("evidence_scope")
        == "TERMINAL_FACTS_ALREADY_CONSUMED_BY_THE_LOW_FREQUENCY_HEARTBEAT_ONLY"
        and audit.get("runtime_path") == str(CURRENT_FULL_RUNTIME)
        and audit.get("output_root")
        == str(CURRENT_FULL_OUTPUT_ROOT / "v4_full")
        and audit.get("terminal_summary_path")
        == str(CURRENT_FULL_OUTPUT_ROOT / "v4_full/run_summary.json")
        and isinstance(terminal_facts, Mapping)
        and terminal_facts.get("authorization_git_head") == TRAINING_GIT_HEAD
        and terminal_facts.get("runtime_status")
        == "XEDITCRITIC_V403_FULL_RECOVERY_TERMINAL"
        and terminal_facts.get("run_id") == "v4_full"
        and terminal_facts.get("terminal_artifact_kind") == "SUMMARY"
        and terminal_facts.get("seed") == 20260907
        and terminal_facts.get("completed_passes") == 8
        and terminal_facts.get("selected_pass") == 8
        and terminal_facts.get("optimizer_update_count") == 22416
        and terminal_facts.get("physical_batch_size") == 32
        and terminal_facts.get("effective_batch_size") == 32
        and terminal_facts.get("cuda_used") is True
        and terminal_facts.get("device_class") == "A100"
        and terminal_facts.get("training_precision")
        == "BF16_FORWARD_FP32_EFFECTIVE_OBJECTIVE"
        and terminal_facts.get("cpu_fallback_used") is False
        and terminal_facts.get("protected_outcome_reads") == 0
        and terminal_facts.get("development_test_outcome_reads") == 0
        and terminal_facts.get("new_final_evaluation_outcome_reads") == 0
        and isinstance(claim_boundary, Mapping)
        and claim_boundary.get("single_arm_terminal_summary_is_not_a_screen_pass")
        is True
        and claim_boundary.get(
            "single_arm_terminal_summary_is_not_final_scientific_evidence"
        )
        is True
        and claim_boundary.get("model_advantage_established") is False
        and claim_boundary.get("submission_ready") is False,
        "recorded V4.0.3 full terminal audit is not exact",
    )
    return audit


def validate_control_runtime(
    path: Path, *, expected_control_runner_head: str
) -> dict[str, Any]:
    require(
        re.fullmatch(r"[0-9a-f]{40}", expected_control_runner_head) is not None
        and expected_control_runner_head not in {TRAINING_GIT_HEAD, C0_GIT_HEAD},
        "control runner HEAD must be a new exact licensed HEAD",
    )
    runtime = read_json(path)
    require(
        runtime.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v403_control_recovery_runtime.v1"
        and runtime.get("status")
        == "XEDITCRITIC_V403_CONTROL_RECOVERY_ALL_SIX_SUMMARIES_TERMINAL"
        and runtime.get("training_code_git_head") == expected_control_runner_head,
        "current-HEAD controls are not all exact terminal summaries",
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
    *,
    expected_control_runner_head: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    require(
        re.fullmatch(r"[0-9a-f]{40}", expected_control_runner_head) is not None
        and expected_control_runner_head not in {TRAINING_GIT_HEAD, C0_GIT_HEAD},
        "control runner HEAD must be a new exact licensed HEAD",
    )
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
        historical = run_id in {"c0_v4", "v4_full"}
        expected_schema = (
            "route_a_v3_route2_xeditcritic_v4_screen_run.v1"
            if historical
            else "route_a_v3_route2_xeditcritic_v4_screen_run.v2"
        )
        require(
            summary.get("schema_version") == expected_schema
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
        expected_head = (
            C0_GIT_HEAD
            if run_id == "c0_v4"
            else TRAINING_GIT_HEAD
            if run_id == "v4_full"
            else expected_control_runner_head
        )
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
            "run_id": run_id,
            "summary_path": str(summary_path),
            "training_git_head": verified_head,
            "source_role": source.source_role,
            "launch_authorization_path": str(authorization_path),
            "launch_authorization_schema_version": str(
                authorization["schema_version"]
            ),
            "launch_authorization_status": str(authorization["status"]),
            "authorized_git_head": verified_head,
            "legacy_terminal_summary": historical,
            "run_id_authorization_verified": True,
            "authorization_protected_outcome_reads_verified_zero": True,
        }
    return summaries, provenance


def run(
    *,
    expected_control_runner_head: str,
    control_runtime_path: Path,
    control_output_root: Path | None = None,
    config_path: Path = FROZEN_CONFIG,
    arm_sources: Mapping[str, ArmSource] | None = None,
    full_terminal_audit_path: Path = FULL_TERMINAL_AUDIT,
    legacy_gate_path: Path = LEGACY_GATE,
    output_path: Path,
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
    require(
        config.get("runner_git_head") == expected_control_runner_head,
        "current controls config runner HEAD differs from the licensed HEAD",
    )
    full_terminal_audit = validate_full_terminal_audit(
        full_terminal_audit_path
    )
    control_runtime = validate_control_runtime(
        control_runtime_path,
        expected_control_runner_head=expected_control_runner_head,
    )
    if arm_sources is None:
        require(
            control_output_root is not None,
            "current-HEAD control output root is required",
        )
    summaries, provenance = collect_arm_summaries(
        config,
        default_arm_sources(control_output_root)
        if arm_sources is None
        else arm_sources,
        expected_control_runner_head=expected_control_runner_head,
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
        terminal_provenance=provenance,
        expected_training_git_heads={
            run_id: expected_control_runner_head for run_id in CONTROL_RUN_IDS
        },
    )
    result["cross_root_transition"] = {
        "schema_version": (
            "route_a_v3_route2_xeditcritic_v403_cross_root_input.v1"
        ),
        "ordered_run_ids": list(ARM_ORDER),
        "arm_sources": provenance,
        "historical_c0_git_head": C0_GIT_HEAD,
        "historical_full_git_head": TRAINING_GIT_HEAD,
        "control_runner_git_head": expected_control_runner_head,
        "full_terminal_audit_path": str(full_terminal_audit_path),
        "full_terminal_audit_status": full_terminal_audit["status"],
        "full_terminal_evidence_scope": full_terminal_audit[
            "evidence_scope"
        ],
        "control_runtime_path": str(control_runtime_path),
        "control_runtime_status": control_runtime["status"],
        "frozen_config_path": str(config_path),
        "legacy_gate_path": str(legacy_gate_path),
        "legacy_gate_preserved": True,
        "terminal_summary_payloads_read": 8,
        "historical_terminal_payloads_read_before_cross_root": 0,
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
    parser.add_argument("--expected-control-runner-head", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--control-runtime", required=True, type=Path)
    parser.add_argument("--control-output-root", required=True, type=Path)
    parser.add_argument(
        "--full-terminal-audit",
        type=Path,
        default=FULL_TERMINAL_AUDIT,
    )
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(
                expected_control_runner_head=arguments.expected_control_runner_head,
                config_path=arguments.config,
                control_runtime_path=arguments.control_runtime,
                control_output_root=arguments.control_output_root,
                full_terminal_audit_path=arguments.full_terminal_audit,
                output_path=arguments.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
