#!/usr/bin/env python3
"""Derive and launch recovery-aware V4.0.3 guidance after dual readiness."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


WORKTREE = Path(__file__).resolve().parents[2]
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from scripts.route_a_v3.launch_route2_xeditflow_v4_guidance_authorization_after_dual_readiness import (
    run as authorize_guidance,
)
from scripts.route_a_v3.launch_route2_xeditflow_v4_guidance_screen_after_authorization import (
    run as launch_guidance_screen,
)
from scripts.route_a_v3.launch_route2_xeditcritic_v403_confirmation_after_cross_root_screen import (
    validate_runner_verification_receipt as validate_shared_runner_verification_receipt,
)


ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
PYTHON = Path("/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10")
STRONGEST_SCORE_PRODUCER = (
    WORKTREE
    / "scripts/route_a_v3/score_route2_xeditflow_closed_frozen_methods_v3.py"
)
STRONGEST_SCORE_SEED = 20260904
BASE_PROTOCOL = (
    WORKTREE / "configs/route_a_v3_route2_xeditflow_v4_guidance_protocol_v1.json"
)
CRITIC_PREFLIGHT = (
    ROOT
    / "experiments/xeditcritic_v4/screen_seed_20260907/"
    "preflight_attempt_5/preflight.json"
)
SETFLOW_PREFLIGHT = (
    ROOT
    / "experiments/xeditsetflow_v4/screen_seed_20260911/"
    "preflight_attempt_5/preflight.json"
)
CRITIC_CACHE_SUMMARY = (
    ROOT
    / "pretrained_features/xeditcritic_v4/"
    "frozen_bottom_six_chunk_cache_v1.summary.json"
)
SETFLOW_CACHE_RECEIPT = (
    ROOT
    / "pretrained_features/xeditsetflow_v4/"
    "source_token_cache_v3_adoption_receipt_v1.json"
)
SEEDS = (20260912, 20260913, 20260914)
DERIVED_EXISTING_PATH_FIELDS = {
    "setflow_confirmation_path",
    "setflow_confirmation_runtime_config_paths",
    "authorization_output",
    "guidance_screen_output_root",
    "runtime_config_root",
}
DERIVED_NEW_FIELDS = {
    "guidance_authorization_decision_output",
    "guidance_screen_runtime_root",
    "guidance_screen_log_root",
    "final_runtime_root",
    "final_log_root",
    "strongest_closed_score_config_path",
    "strongest_closed_score_failure_path",
    "strongest_closed_score_summary_path",
    "strongest_closed_score_table_path",
    "v403_recovery_provenance",
}


class XEditFlowV403GuidanceBridgeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowV403GuidanceBridgeError(message)


def valid_head(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    require(
        bool(rows) and all(isinstance(row, dict) for row in rows),
        f"JSONL artifact is empty or invalid: {path}",
    )
    return rows


def write_new_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"artifact already exists: {path}")
    partial = path.with_suffix(path.suffix + ".partial")
    require(not partial.exists(), f"partial artifact already exists: {partial}")
    partial.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def command(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(arguments),
        cwd=WORKTREE,
        text=True,
        capture_output=True,
        check=True,
    )


def route2_path(path: Path, label: str) -> None:
    require(
        str(path).startswith(str(ROOT) + "/"),
        f"{label} is outside Route 2 /mnt: {path}",
    )


def resolve_worktree_path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else WORKTREE / path


def expected_runner_verification_receipt_path_v403(runner_head: str) -> Path:
    require(valid_head(runner_head), "successor runner verification HEAD is invalid")
    return (
        ROOT
        / "audits/xedit_v4"
        / f"v403_successor_runner_verification_{runner_head}.json"
    )


def consume_runner_verification_receipt_v403(
    receipt_path: Path, runner_head: str
) -> dict[str, Any]:
    expected = expected_runner_verification_receipt_path_v403(runner_head)
    require(
        receipt_path == expected,
        "successor runner verification receipt is not the exact-HEAD canonical path",
    )
    require(
        receipt_path.is_file(),
        f"successor runner verification receipt is absent: {receipt_path}",
    )
    receipt = read_json(receipt_path)
    validate_shared_runner_verification_receipt(
        receipt,
        runner_head=runner_head,
        receipt_path=receipt_path,
    )
    require(
        int(receipt["v332_tests"]["passed_count"]) == 96,
        "successor runner receipt does not contain the exact 96-test V3.3.2 PASS",
    )
    return {
        "receipt_path": str(receipt_path),
        "schema_version": receipt["schema_version"],
        "status": receipt["status"],
        "runner_git_head": receipt["runner_git_head"],
        "worktree_clean": receipt["worktree_clean"],
        "focused_tests": dict(receipt["focused_tests"]),
        "v332_tests": dict(receipt["v332_tests"]),
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def validate_setflow_lineage_v403(
    schedule: Mapping[str, Any],
    runtime: Mapping[str, Any],
    gate: Mapping[str, Any],
    recovered_protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    schedule_path: Path,
    runtime_path: Path,
    gate_path: Path,
    recovered_protocol_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    bindings = schedule.get("posttraining_bindings")
    require(
        schedule.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v403_recovered_confirmation_posttraining_schedule.v1"
        and schedule.get("status")
        == "FROZEN_RECOVERED_CONFIRMATION_POSTTRAINING_SCHEDULE"
        and schedule.get("eligible_components") == ["setflow"]
        and schedule.get("runtime_manifest") == str(runtime_path)
        and schedule.get("free_memory_gate_applied") is False
        and schedule.get("cpu_fallback_used") is False
        and int(schedule.get("development_test_outcome_reads", -1)) == 0
        and int(schedule.get("new_final_evaluation_outcome_reads", -1)) == 0
        and isinstance(bindings, Mapping),
        "SetFlow V4.0.3 recovered posttraining schedule changed",
    )
    orchestration_head = schedule.get("orchestration_git_head")
    training_runner_head = schedule.get("training_runner_git_head")
    require(
        valid_head(orchestration_head)
        and valid_head(training_runner_head)
        and schedule.get("git_head") == orchestration_head
        and bindings.get("runner_git_head") == training_runner_head,
        "SetFlow V4.0.3 orchestration and training runner provenance changed",
    )
    require(
        resolve_worktree_path(bindings.get("protocol_path")).resolve()
        == recovered_protocol_path.resolve()
        and bindings.get("config_manifest_path") == str(manifest_path)
        and bindings.get("confirmation_gate_output") == str(gate_path)
        and bindings.get("posttraining_runtime_root") == str(runtime_path.parent),
        "SetFlow V4.0.3 posttraining path bindings changed",
    )
    recovery = recovered_protocol.get("validation_recovery_provenance")
    require(
        recovered_protocol.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_confirmation_protocol.v1"
        and recovered_protocol.get("status")
        == "FROZEN_PROSPECTIVE_BEFORE_SCREEN_OR_CONFIRMATION_RESULT"
        and recovered_protocol.get("required_seeds") == list(SEEDS)
        and recovered_protocol.get("additional_seed_authorized") is False
        and recovered_protocol.get("confirmation_gate_output") == str(gate_path)
        and isinstance(recovery, Mapping)
        and valid_head(recovery.get("training_git_head"))
        and valid_head(recovery.get("validation_git_head"))
        and schedule.get("training_git_head") == recovery.get("training_git_head")
        and schedule.get("recovery_validation_git_head")
        == recovery.get("validation_git_head")
        and int(recovery.get("parameter_update_count", -1)) == 0
        and recovery.get("scientific_thresholds_changed") is False
        and int(recovered_protocol.get("development_test_outcome_reads", -1)) == 0
        and int(recovered_protocol.get("new_final_evaluation_outcome_reads", -1))
        == 0,
        "SetFlow V4.0.3 recovered confirmation protocol provenance changed",
    )
    validations = runtime.get("validation_jobs")
    adjudication = runtime.get("adjudications", {}).get("setflow", {})
    require(
        runtime.get("schema_version")
        == "route_a_v3_route2_xedit_v4_confirmation_posttraining_runtime.v1"
        and runtime.get("status") == "V4_CONFIRMATION_POSTTRAINING_ALL_TERMINAL"
        and runtime.get("git_head") == orchestration_head
        and runtime.get("eligible_components") == ["setflow"]
        and isinstance(validations, Mapping)
        and len(validations) == 12
        and all(
            row.get("status") == "TERMINAL_COMPLETE"
            and row.get("terminal_artifact_kind") == "SUMMARY"
            for row in validations.values()
        )
        and adjudication.get("status") == "TERMINAL_COMPLETE"
        and adjudication.get("terminal_artifact_kind") == "SUMMARY"
        and adjudication.get("gate_path") == str(gate_path)
        and runtime.get("active_performance_output_read") is False
        and int(runtime.get("development_test_outcome_reads", -1)) == 0
        and int(runtime.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "SetFlow V4.0.3 recovered confirmation runtime is not exact G0 terminal",
    )
    seed_results = gate.get("seed_results")
    require(
        gate.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_confirmation_gate.v1"
        and gate.get("status") == "XEDITSETFLOW_V4_G0_READY"
        and gate.get("required_seeds") == list(SEEDS)
        and isinstance(seed_results, Mapping)
        and set(seed_results) == {str(seed) for seed in SEEDS}
        and all(seed_results[str(seed)].get("passed") is True for seed in SEEDS)
        and gate.get("additional_seed_authorized") is False
        and gate.get("development_test_authorized") is False
        and gate.get("guidance_authorized") is False
        and gate.get("critic_used") is False
        and gate.get("independent_evaluator_used") is False
        and int(gate.get("development_test_outcome_reads", -1)) == 0
        and int(gate.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "SetFlow V4.0.3 recovered confirmation gate is not exact G0 READY",
    )
    require(
        manifest.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_confirmation_config_manifest.v1"
        and manifest.get("status")
        == "THREE_CONFIRMATION_CONFIGS_PREPARED_NOT_STARTED"
        and manifest.get("selected_model") == "v4_full"
        and manifest.get("required_seeds") == list(SEEDS)
        and len(manifest.get("config_paths", ())) == 3
        and int(manifest.get("development_test_outcome_reads", -1)) == 0
        and int(manifest.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "SetFlow V4.0.3 recovered confirmation manifest changed",
    )
    configs: dict[str, str] = {}
    for value in manifest["config_paths"]:
        path = Path(str(value))
        config = read_json(path)
        seed = int(config.get("training_seed", -1))
        require(
            seed in SEEDS
            and config.get("schema_version")
            == "route_a_v3_route2_xeditsetflow_v4_confirmation_runtime.v1"
            and config.get("screen_gate_path")
            == recovery.get("recovered_screen_gate_path")
            and config.get("development_test_outcomes_accessed") is False
            and config.get("new_final_evaluation_outcomes_accessed") is False,
            f"SetFlow V4.0.3 recovered runtime config changed: {path}",
        )
        configs[str(seed)] = str(path)
    require(
        set(configs) == {str(seed) for seed in SEEDS},
        "SetFlow V4.0.3 runtime config seed inventory changed",
    )
    return {
        "posttraining_schedule_path": str(schedule_path),
        "posttraining_runtime_path": str(runtime_path),
        "posttraining_runner_git_head": orchestration_head,
        "training_runner_git_head": training_runner_head,
        "training_git_head": recovery["training_git_head"],
        "recovery_validation_git_head": recovery["validation_git_head"],
        "recovered_protocol_path": str(recovered_protocol_path),
        "config_manifest_path": str(manifest_path),
        "confirmation_gate_path": str(gate_path),
        "runtime_config_paths": configs,
    }


def validate_critic_lineage_v403(
    runtime: Mapping[str, Any],
    readiness: Mapping[str, Any],
    refit_manifest: Mapping[str, Any],
    *,
    runtime_path: Path,
    readiness_path: Path,
    refit_manifest_path: Path,
) -> dict[str, Any]:
    runner_head = runtime.get("git_head")
    runtime_readiness = runtime.get("readiness")
    require(
        runtime.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_loso_runtime.v1"
        and runtime.get("status") == "CRITIC_V4_READY_FOR_GUIDANCE"
        and valid_head(runner_head)
        and isinstance(runtime_readiness, Mapping)
        and runtime_readiness.get("terminal_artifact_kind") == "SUMMARY"
        and runtime_readiness.get("readiness_status")
        == "CRITIC_V4_READY_FOR_GUIDANCE"
        and runtime_readiness.get("guidance_authorized") is True
        and runtime_readiness.get("summary_path") == str(readiness_path)
        and runtime.get("active_performance_output_read") is False
        and int(runtime.get("development_test_access_event_count_before_loso", -1))
        == 1
        and int(runtime.get("development_test_outcome_reads_during_loso", -1)) == 0
        and int(runtime.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "Critic V4 LOSO runtime is not exact READY terminal",
    )
    require(
        readiness.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_guidance_readiness.v1"
        and readiness.get("status") == "CRITIC_V4_READY_FOR_GUIDANCE"
        and readiness.get("three_seed_passed") is True
        and readiness.get("frozen_test_passed") is True
        and readiness.get("all_development_refit_complete") is True
        and readiness.get("loso_readiness_passed") is True
        and readiness.get("guidance_authorized") is True
        and int(readiness.get("development_test_access_event_count", -1)) == 1
        and readiness.get("development_test_outcomes_accessed_after_atomic_test")
        is False
        and readiness.get("new_final_evaluation_outcomes_accessed") is False,
        "Critic V4 readiness receipt is not exact READY",
    )
    checkpoints = refit_manifest.get("checkpoints")
    require(
        refit_manifest.get("status")
        == "XEDITCRITIC_V4_ALL_DEVELOPMENT_REFIT_COMPLETE"
        and refit_manifest.get("required_seeds") == [20260908, 20260909, 20260910]
        and int(refit_manifest.get("completed_refit_count", -1)) == 3
        and int(refit_manifest.get("refit_pass_count", -1)) == 8
        and refit_manifest.get("loso_authorized") is True
        and isinstance(checkpoints, list)
        and sorted(int(row["seed"]) for row in checkpoints)
        == [20260908, 20260909, 20260910]
        and refit_manifest.get("development_test_outcomes_accessed_during_refit")
        is False
        and refit_manifest.get("new_final_evaluation_outcomes_accessed") is False,
        "Critic V4 refit manifest is not exact terminal",
    )
    return {
        "loso_runtime_path": str(runtime_path),
        "training_runner_git_head": runner_head,
        "readiness_path": str(readiness_path),
        "refit_manifest_path": str(refit_manifest_path),
    }


def validate_preflight_cache_lineage_v403(
    critic_preflight: Mapping[str, Any],
    setflow_preflight: Mapping[str, Any],
    critic_authorization: Mapping[str, Any],
    setflow_authorization: Mapping[str, Any],
    critic_cache_summary: Mapping[str, Any],
    setflow_cache_receipt: Mapping[str, Any],
    *,
    critic_preflight_path: Path,
    setflow_preflight_path: Path,
    critic_authorization_path: Path,
    setflow_authorization_path: Path,
    critic_cache_summary_path: Path,
    setflow_cache_receipt_path: Path,
    source_token_cache_path: Path,
) -> dict[str, Any]:
    critic_head = critic_preflight.get("git_head")
    setflow_head = setflow_preflight.get("git_head")
    critic_cache_head = critic_cache_summary.get("git_head")
    setflow_cache_head = setflow_cache_receipt.get("git_head")
    require(
        critic_preflight.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_preflight.v1"
        and critic_preflight.get("status") == "XEDITCRITIC_V4_PREFLIGHT_PASS"
        and critic_preflight.get("passed") is True
        and valid_head(critic_head)
        and critic_preflight.get("authorization_path")
        == str(critic_authorization_path)
        and critic_preflight.get("cpu_fallback_used") is False
        and bool(str(critic_preflight.get("cuda_device_name", "")).strip())
        and int(critic_preflight.get("development_test_outcome_reads", -1)) == 0
        and int(critic_preflight.get("new_final_evaluation_outcome_reads", -1))
        == 0,
        "Critic V4 preflight provenance changed",
    )
    require(
        setflow_preflight.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_preflight.v1"
        and setflow_preflight.get("status") == "XEDITSETFLOW_V4_PREFLIGHT_PASS"
        and setflow_preflight.get("passed") is True
        and valid_head(setflow_head)
        and setflow_preflight.get("cpu_fallback_used") is False
        and str(setflow_preflight.get("torch_device", "")).startswith("cuda:")
        and str(setflow_preflight.get("precision", "")).upper() == "BF16"
        and int(setflow_preflight.get("development_test_outcome_reads", -1)) == 0
        and int(setflow_preflight.get("new_final_evaluation_outcome_reads", -1))
        == 0,
        "SetFlow V4 preflight provenance changed",
    )
    require(
        critic_authorization.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_preflight_authorization.v1"
        and critic_authorization.get("status")
        == "XEDITCRITIC_V4_PREFLIGHT_AUTHORIZED"
        and critic_authorization.get("authorized_git_head") == critic_head
        and critic_authorization.get("cache_experiment_head") == critic_cache_head
        and int(critic_authorization.get("development_test_outcome_reads", -1)) == 0
        and int(critic_authorization.get("new_final_evaluation_outcome_reads", -1))
        == 0,
        "Critic V4 preflight authorization/cache binding changed",
    )
    require(
        setflow_authorization.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_preflight_authorization.v1"
        and setflow_authorization.get("status")
        == "XEDITSETFLOW_V4_PREFLIGHT_AUTHORIZED"
        and setflow_authorization.get("authorized_git_head") == setflow_head
        and setflow_authorization.get("cache_experiment_head") == setflow_cache_head
        and int(setflow_authorization.get("development_test_outcome_reads", -1)) == 0
        and int(setflow_authorization.get("new_final_evaluation_outcome_reads", -1))
        == 0,
        "SetFlow V4 preflight authorization/cache binding changed",
    )
    require(
        critic_cache_summary.get("schema_version")
        == "route_a_v3_route2_frozen_bottom_encoder_chunk_cache_summary.v4"
        and critic_cache_summary.get("status")
        == "XEDITCRITIC_V4_BOTTOM_SIX_CACHE_COMPLETE"
        and valid_head(critic_cache_head)
        and critic_cache_summary.get("cpu_fallback") is False
        and critic_cache_summary.get("development_test_outcomes_accessed") is False
        and critic_cache_summary.get("evaluation_outcomes_accessed") is False,
        "Critic V4 bottom-six cache provenance changed",
    )
    require(
        setflow_cache_receipt.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_source_cache_adoption_receipt.v1"
        and setflow_cache_receipt.get("status")
        == "XEDITSETFLOW_V4_SOURCE_CACHE_ADOPTED_READ_ONLY"
        and valid_head(setflow_cache_head)
        and setflow_cache_receipt.get("legacy_cache_path")
        == str(source_token_cache_path)
        and setflow_cache_receipt.get("cpu_fallback_used") is False
        and int(setflow_cache_receipt.get("development_test_outcome_reads", -1))
        == 0
        and int(setflow_cache_receipt.get("new_final_evaluation_outcome_reads", -1))
        == 0
        and setflow_cache_receipt.get("source_token_cache_identity")
        == setflow_preflight.get("source_token_cache_identity"),
        "SetFlow V4 source-cache provenance changed",
    )
    return {
        "critic_preflight_path": str(critic_preflight_path),
        "critic_preflight_authorization_path": str(critic_authorization_path),
        "critic_preflight_runner_git_head": critic_head,
        "critic_cache_summary_path": str(critic_cache_summary_path),
        "critic_cache_experiment_git_head": critic_cache_head,
        "setflow_preflight_path": str(setflow_preflight_path),
        "setflow_preflight_authorization_path": str(setflow_authorization_path),
        "setflow_preflight_runner_git_head": setflow_head,
        "setflow_cache_receipt_path": str(setflow_cache_receipt_path),
        "setflow_cache_experiment_git_head": setflow_cache_head,
    }


def build_derived_guidance_protocol_v403(
    base: Mapping[str, Any],
    *,
    runner_head: str,
    setflow_lineage: Mapping[str, Any],
    critic_lineage: Mapping[str, Any],
    prerequisite_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    require(
        base.get("schema_version")
        == "route_a_v3_route2_xeditflow_v4_guidance_protocol.v1"
        and base.get("status")
        == "FROZEN_PROSPECTIVE_BEFORE_V4_GUIDANCE_AUTHORIZATION_OR_OUTCOME_READ"
        and valid_head(runner_head),
        "base V4 guidance protocol or bridge runner HEAD changed",
    )
    require(
        critic_lineage.get("readiness_path") == base.get("critic_readiness_path")
        and critic_lineage.get("refit_manifest_path")
        == base.get("critic_refit_manifest_path"),
        "Critic V4.0.3 lineage is not the canonical guidance input family",
    )
    family = f"guidance_v403_recovered_{runner_head}"
    experiment_root = ROOT / "experiments/xeditflow_v4" / family
    config_family = ROOT / "runtime_configs/xeditflow_v4" / family
    log_family = ROOT / "logs/xeditflow_v4" / family
    strongest_score_root = (
        experiment_root
        / "pre_guidance_strongest_matched_baseline_scores_seed20260904"
    )
    derived = copy.deepcopy(dict(base))
    derived.update(
        {
            "setflow_confirmation_path": setflow_lineage[
                "confirmation_gate_path"
            ],
            "setflow_confirmation_runtime_config_paths": dict(
                setflow_lineage["runtime_config_paths"]
            ),
            "authorization_output": str(
                experiment_root / "guidance_authorization.json"
            ),
            "guidance_screen_output_root": str(
                experiment_root / "screen_seed_20260912"
            ),
            "runtime_config_root": str(config_family / "guidance_screen_v1"),
            "guidance_authorization_decision_output": str(
                experiment_root / "authorization_decision.json"
            ),
            "guidance_screen_runtime_root": str(
                experiment_root / "screen_execution"
            ),
            "guidance_screen_log_root": str(log_family / "screen_execution"),
            "final_runtime_root": str(experiment_root / "final_execution"),
            "final_log_root": str(log_family / "final_execution"),
            "strongest_closed_score_config_path": str(
                config_family / "strongest_matched_baseline_closed_score.json"
            ),
            "strongest_closed_score_failure_path": str(
                strongest_score_root.with_name(
                    strongest_score_root.name + ".bridge_failure.json"
                )
            ),
            "strongest_closed_score_summary_path": str(
                strongest_score_root / "run_summary.json"
            ),
            "strongest_closed_score_table_path": str(
                strongest_score_root / "frozen_method_scores.private.jsonl"
            ),
            "v403_recovery_provenance": {
                "schema_version": (
                    "route_a_v3_route2_xeditflow_v403_guidance_derivation.v1"
                ),
                "status": "RECOVERY_AWARE_DUAL_READINESS_BOUND",
                "bridge_runner_git_head": runner_head,
                "critic": dict(critic_lineage),
                "setflow": {
                    key: value
                    for key, value in setflow_lineage.items()
                    if key != "runtime_config_paths"
                },
                "prerequisites": dict(prerequisite_lineage),
                "strongest_matched_baseline_closed_score": {
                    "status": "BOUND_FOR_MATERIALIZATION_BEFORE_GUIDANCE_SCREEN",
                    "method_id": "strongest_matched_baseline",
                    "base_flow_training_seed": STRONGEST_SCORE_SEED,
                    "score_provider": "FROZEN_GENETIC_GUIDING_CHECKPOINT",
                    "config_path": str(
                        config_family
                        / "strongest_matched_baseline_closed_score.json"
                    ),
                    "bridge_failure_path": str(
                        strongest_score_root.with_name(
                            strongest_score_root.name + ".bridge_failure.json"
                        )
                    ),
                    "summary_path": str(
                        strongest_score_root / "run_summary.json"
                    ),
                    "score_table_path": str(
                        strongest_score_root
                        / "frozen_method_scores.private.jsonl"
                    ),
                    "guidance_winner_used": False,
                    "baseline_reselected_for_v4": False,
                    "measured_outcome_used_for_score": False,
                },
                "scientific_protocol_changed": False,
                "seed_inventory_changed": False,
                "guidance_grid_changed": False,
                "budget_changed": False,
                "gate_threshold_changed": False,
                "free_memory_gate_applied": False,
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        }
    )
    validate_derived_guidance_protocol_v403(base, derived)
    return derived


def validate_derived_guidance_protocol_v403(
    base: Mapping[str, Any], derived: Mapping[str, Any]
) -> None:
    require(
        set(derived) == set(base) | DERIVED_NEW_FIELDS,
        "derived V4.0.3 guidance protocol field inventory changed",
    )
    changed = {
        key for key in base if derived.get(key) != base.get(key)
    }
    require(
        changed == DERIVED_EXISTING_PATH_FIELDS,
        "derived V4.0.3 guidance protocol changed non-path science",
    )
    require(
        all(
            derived[key] == base[key]
            for key in base
            if key not in DERIVED_EXISTING_PATH_FIELDS
        ),
        "derived V4.0.3 guidance science is not byte-equivalent by field",
    )
    require(
        set(derived["setflow_confirmation_runtime_config_paths"])
        == {str(seed) for seed in SEEDS},
        "derived V4.0.3 SetFlow runtime seed inventory changed",
    )
    for key in (
        "setflow_confirmation_path",
        "authorization_output",
        "guidance_screen_output_root",
        "runtime_config_root",
        "guidance_authorization_decision_output",
        "guidance_screen_runtime_root",
        "guidance_screen_log_root",
        "final_runtime_root",
        "final_log_root",
        "strongest_closed_score_config_path",
        "strongest_closed_score_failure_path",
        "strongest_closed_score_summary_path",
        "strongest_closed_score_table_path",
    ):
        route2_path(Path(str(derived[key])), f"derived protocol {key}")
    for path in derived["setflow_confirmation_runtime_config_paths"].values():
        route2_path(Path(str(path)), "derived SetFlow runtime config")
    require(
        derived["authorization_output"] != base["authorization_output"]
        and derived["guidance_screen_output_root"]
        != base["guidance_screen_output_root"]
        and derived["runtime_config_root"] != base["runtime_config_root"],
        "derived V4.0.3 guidance output family aliases legacy artifacts",
    )
    provenance = derived["v403_recovery_provenance"]
    strongest_score = provenance.get("strongest_matched_baseline_closed_score")
    require(
        provenance.get("schema_version")
        == "route_a_v3_route2_xeditflow_v403_guidance_derivation.v1"
        and provenance.get("status") == "RECOVERY_AWARE_DUAL_READINESS_BOUND"
        and valid_head(provenance.get("bridge_runner_git_head"))
        and provenance.get("scientific_protocol_changed") is False
        and provenance.get("seed_inventory_changed") is False
        and provenance.get("guidance_grid_changed") is False
        and provenance.get("budget_changed") is False
        and provenance.get("gate_threshold_changed") is False
        and provenance.get("free_memory_gate_applied") is False
        and int(provenance.get("development_test_outcome_reads", -1)) == 0
        and int(provenance.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "derived V4.0.3 guidance provenance changed",
    )
    require(
        isinstance(strongest_score, Mapping)
        and strongest_score.get("status")
        == "BOUND_FOR_MATERIALIZATION_BEFORE_GUIDANCE_SCREEN"
        and strongest_score.get("method_id") == "strongest_matched_baseline"
        and int(strongest_score.get("base_flow_training_seed", -1))
        == STRONGEST_SCORE_SEED
        and strongest_score.get("score_provider")
        == "FROZEN_GENETIC_GUIDING_CHECKPOINT"
        and strongest_score.get("config_path")
        == derived["strongest_closed_score_config_path"]
        and strongest_score.get("bridge_failure_path")
        == derived["strongest_closed_score_failure_path"]
        and strongest_score.get("summary_path")
        == derived["strongest_closed_score_summary_path"]
        and strongest_score.get("score_table_path")
        == derived["strongest_closed_score_table_path"]
        and strongest_score.get("guidance_winner_used") is False
        and strongest_score.get("baseline_reselected_for_v4") is False
        and strongest_score.get("measured_outcome_used_for_score") is False,
        "derived strongest matched-baseline score binding changed",
    )


def build_strongest_closed_score_config_v403(
    protocol: Mapping[str, Any], *, physical_gpu_index: int
) -> dict[str, Any]:
    gpu = int(physical_gpu_index)
    require(gpu in range(6), "strongest closed-score physical GPU is outside 0-5")
    return {
        "schema_version": (
            "route_a_v3_route2_xeditflow_closed_frozen_score_config.v1"
        ),
        "method_id": "strongest_matched_baseline",
        "v4_guidance_authorization_path": str(protocol["authorization_output"]),
        "source_eligibility_manifest": str(
            protocol["source_eligibility_manifest"]
        ),
        "measured_neighborhood_path": str(
            protocol["measured_neighborhood_path"]
        ),
        "strongest_generation_baseline_path": str(
            protocol["strongest_generation_baseline_path"]
        ),
        "baseline_selection_input_path": str(
            protocol["baseline_selection_input_path"]
        ),
        "pool_assignment": "DEVELOPMENT",
        "split": "VALIDATION",
        "expected_source_count": 891,
        "base_flow_training_seed": STRONGEST_SCORE_SEED,
        "physical_gpu_index": gpu,
        "device": f"cuda:{gpu}",
        "output_dir": str(
            Path(str(protocol["strongest_closed_score_summary_path"])).parent
        ),
    }


def validate_strongest_closed_score_artifact_v403(
    summary: Mapping[str, Any],
    *,
    summary_path: Path,
    score_table_path: Path,
    measured_neighborhood_path: Path,
    authorization_path: Path,
    physical_gpu_index: int,
) -> dict[str, Any]:
    for path, label in (
        (summary_path, "strongest closed-score summary"),
        (score_table_path, "strongest closed-score table"),
        (measured_neighborhood_path, "common measured neighborhood"),
        (authorization_path, "V4 guidance authorization"),
    ):
        route2_path(path, label)
        require(path.is_file(), f"{label} is absent: {path}")
    require(
        summary.get("schema_version")
        == "route_a_v3_route2_xeditflow_closed_frozen_scores.v3"
        and summary.get("status")
        == "XEDITFLOW_V3_CLOSED_FROZEN_SCORES_COMPLETE"
        and summary.get("method_id") == "strongest_matched_baseline"
        and int(summary.get("base_flow_training_seed", -1))
        == STRONGEST_SCORE_SEED
        and int(summary.get("source_count", -1)) == 891
        and summary.get("score_path") == str(score_table_path)
        and summary.get("score_provider")
        == "FROZEN_GENETIC_GUIDING_CHECKPOINT"
        and summary.get("frozen_baseline_reselected") is False
        and summary.get("measured_outcome_used_for_score") is False
        and summary.get("development_test_outcomes_accessed") is False
        and summary.get("new_final_evaluation_outcomes_accessed") is False
        and summary.get("cpu_fallback_used") is False
        and int(summary.get("cuda_device_index", -1))
        == int(physical_gpu_index)
        and bool(str(summary.get("cuda_device_name", "")).strip())
        and summary.get("cuda_parent_uuid_matches_declared_physical_index")
        is True
        and summary.get("guidance_authorization_path")
        == str(authorization_path)
        and summary.get("guidance_authorization_schema_version")
        == "route_a_v3_route2_xeditflow_v4_guidance_authorization.v1"
        and summary.get("guidance_authorization_status")
        == "XEDITFLOW_V4_GUIDANCE_AUTHORIZED",
        "pre-guidance strongest matched-baseline score summary changed",
    )
    measured_rows = read_jsonl(measured_neighborhood_path)
    score_rows = read_jsonl(score_table_path)
    measured_keys = [
        (str(row["source_key"]), str(row["candidate_sequence"]))
        for row in measured_rows
    ]
    score_keys = [
        (str(row["source_key"]), str(row["candidate_sequence"]))
        for row in score_rows
    ]
    require(
        len(measured_keys) == len(set(measured_keys))
        and len(score_keys) == len(set(score_keys))
        and set(score_keys) == set(measured_keys)
        and int(summary.get("measured_candidate_count", -1))
        == len(score_rows)
        and int(summary.get("strongest_guiding_forward_calls", -1))
        == len(score_rows),
        "pre-guidance strongest score table does not exactly cover the frozen cohort",
    )
    require(
        all(
            row.get("method_id") == "strongest_matched_baseline"
            and int(row.get("base_flow_training_seed", -1))
            == STRONGEST_SCORE_SEED
            and row.get("score_used_measured_outcome") is False
            and isinstance(row.get("frozen_method_score"), (int, float))
            and not isinstance(row.get("frozen_method_score"), bool)
            and math.isfinite(float(row["frozen_method_score"]))
            for row in score_rows
        ),
        "pre-guidance strongest score table row schema or provenance changed",
    )
    return {
        "status": "MATERIALIZED_BEFORE_V4_GUIDANCE_SCREEN",
        "method_id": "strongest_matched_baseline",
        "base_flow_training_seed": STRONGEST_SCORE_SEED,
        "score_provider": "FROZEN_GENETIC_GUIDING_CHECKPOINT",
        "summary_path": str(summary_path),
        "score_table_path": str(score_table_path),
        "measured_candidate_count": len(score_rows),
        "physical_gpu_index": int(physical_gpu_index),
        "cpu_fallback_used": False,
        "guidance_winner_used": False,
        "baseline_reselected_for_v4": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def materialize_strongest_closed_score_v403(
    protocol: Mapping[str, Any], *, physical_gpu_index: int
) -> dict[str, Any]:
    config = build_strongest_closed_score_config_v403(
        protocol, physical_gpu_index=physical_gpu_index
    )
    config_path = Path(str(protocol["strongest_closed_score_config_path"]))
    bridge_failure_path = Path(
        str(protocol["strongest_closed_score_failure_path"])
    )
    summary_path = Path(str(protocol["strongest_closed_score_summary_path"]))
    score_table_path = Path(str(protocol["strongest_closed_score_table_path"]))
    output_dir = summary_path.parent
    for path, label in (
        (config_path, "strongest closed-score config"),
        (output_dir, "strongest closed-score output directory"),
    ):
        route2_path(path, label)
        require(not path.exists(), f"{label} already exists: {path}")
    write_new_atomic(config_path, config)
    arguments = [
        str(PYTHON),
        str(STRONGEST_SCORE_PRODUCER),
        "--config",
        str(config_path),
    ]
    try:
        command(arguments)
    except subprocess.CalledProcessError as error:
        scorer_failure_path = output_dir.with_name(
            output_dir.name + ".failed.json"
        )
        scorer_failure = (
            read_json(scorer_failure_path)
            if scorer_failure_path.is_file()
            else {}
        )
        write_new_atomic(
            bridge_failure_path,
            {
                "schema_version": (
                    "route_a_v3_route2_xeditflow_v403_strongest_closed_"
                    "score_bridge_failure.v1"
                ),
                "status": "STOPPED_STRONGEST_CLOSED_SCORE_PRODUCER_FAILURE",
                "command": [str(value) for value in error.cmd],
                "returncode": int(error.returncode),
                "stdout": error.stdout,
                "stderr": error.stderr,
                "physical_gpu_index": int(physical_gpu_index),
                "requested_device": f"cuda:{int(physical_gpu_index)}",
                "cuda_available": scorer_failure.get("cuda_available"),
                "requested_cuda_observation": scorer_failure.get(
                    "requested_cuda_observation"
                ),
                "scorer_error_type": scorer_failure.get("error_type"),
                "scorer_error": scorer_failure.get("error"),
                "scorer_cpu_fallback_used": scorer_failure.get(
                    "cpu_fallback_used"
                ),
                "scorer_failure_evidence_path": str(scorer_failure_path),
                "scorer_failure_evidence_present": bool(scorer_failure),
                "cpu_fallback_used": False,
                "free_memory_gate_applied": False,
                "automatic_retry_attempted": False,
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )
        raise XEditFlowV403GuidanceBridgeError(
            "strongest closed-score producer failed; bridge stopped with evidence: "
            f"{bridge_failure_path}"
        ) from error
    return validate_strongest_closed_score_artifact_v403(
        read_json(summary_path),
        summary_path=summary_path,
        score_table_path=score_table_path,
        measured_neighborhood_path=Path(
            str(protocol["measured_neighborhood_path"])
        ),
        authorization_path=Path(str(protocol["authorization_output"])),
        physical_gpu_index=physical_gpu_index,
    )


def build_final_successor_v403(
    *,
    current_head: str,
    protocol_path: Path,
    guidance_runtime_path: Path,
    guidance_screen_gate_path: Path,
    derived: Mapping[str, Any],
    prerequisite_lineage: Mapping[str, Any],
    strongest_score_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    require(valid_head(current_head), "final successor Git HEAD is invalid")
    for path, label in (
        (protocol_path, "final successor guidance protocol"),
        (guidance_runtime_path, "final successor guidance runtime"),
        (guidance_screen_gate_path, "final successor guidance screen gate"),
        (
            Path(str(prerequisite_lineage["critic_preflight_path"])),
            "final successor Critic preflight",
        ),
        (
            Path(str(prerequisite_lineage["setflow_preflight_path"])),
            "final successor SetFlow preflight",
        ),
        (
            Path(str(derived["strongest_closed_score_table_path"])),
            "final successor strongest closed-score table",
        ),
        (
            Path(str(derived["strongest_closed_score_summary_path"])),
            "final successor strongest closed-score summary",
        ),
        (Path(str(derived["final_runtime_root"])), "final successor runtime root"),
        (Path(str(derived["final_log_root"])), "final successor log root"),
    ):
        route2_path(path, label)
    for key in (
        "critic_preflight_runner_git_head",
        "setflow_preflight_runner_git_head",
    ):
        require(
            valid_head(prerequisite_lineage.get(key)),
            f"final successor {key} is invalid",
        )
    require(
        strongest_score_lineage.get("status")
        == "MATERIALIZED_BEFORE_V4_GUIDANCE_SCREEN"
        and strongest_score_lineage.get("method_id")
        == "strongest_matched_baseline"
        and strongest_score_lineage.get("score_table_path")
        == derived["strongest_closed_score_table_path"]
        and strongest_score_lineage.get("summary_path")
        == derived["strongest_closed_score_summary_path"]
        and strongest_score_lineage.get("guidance_winner_used") is False
        and strongest_score_lineage.get("baseline_reselected_for_v4") is False,
        "final successor strongest matched-baseline score provenance changed",
    )
    return {
        "status": "NOT_LAUNCHED_BEFORE_GUIDANCE_SCREEN_TERMINAL",
        "launcher_path": str(
            WORKTREE
            / "scripts/route_a_v3/"
            "launch_route2_xeditflow_v4_final_after_guidance_screen.py"
        ),
        "current_head": current_head,
        "experiment_head": current_head,
        "guidance_runner_head": current_head,
        "protocol_path": str(protocol_path),
        "guidance_runtime_path": str(guidance_runtime_path),
        "strongest_closed_score_table_path": derived[
            "strongest_closed_score_table_path"
        ],
        "strongest_closed_score_summary_path": derived[
            "strongest_closed_score_summary_path"
        ],
        "guidance_selection_resolution": {
            "status": "UNRESOLVED_UNTIL_FROZEN_GUIDANCE_SCREEN_GATE",
            "guidance_screen_gate_path": str(guidance_screen_gate_path),
            "selected_combination_gate_fields": [
                "selected_kappa",
                "selected_temperature",
                "selected_beta_max",
            ],
            "winner_resolves_only": [
                "kappa",
                "temperature",
                "beta_max",
            ],
            "strongest_closed_score_table_independent_of_winner": True,
            "single_guidance_winner_bound_before_frozen_gate": False,
        },
        "critic_preflight_path": prerequisite_lineage[
            "critic_preflight_path"
        ],
        "critic_preflight_runner_git_head": prerequisite_lineage[
            "critic_preflight_runner_git_head"
        ],
        "setflow_preflight_path": prerequisite_lineage[
            "setflow_preflight_path"
        ],
        "setflow_preflight_runner_git_head": prerequisite_lineage[
            "setflow_preflight_runner_git_head"
        ],
        "execution_runtime_root": derived["final_runtime_root"],
        "execution_log_root": derived["final_log_root"],
        "required_base_flow_training_seeds": list(SEEDS),
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def run(
    current_head: str,
    *,
    runner_verification_receipt_path: Path,
    base_protocol_path: Path,
    setflow_posttraining_schedule_path: Path,
    setflow_recovered_protocol_path: Path,
    setflow_posttraining_runtime_path: Path,
    setflow_confirmation_gate_path: Path,
    critic_loso_runtime_path: Path,
    critic_readiness_path: Path,
    critic_refit_manifest_path: Path,
    critic_preflight_path: Path,
    setflow_preflight_path: Path,
    critic_cache_summary_path: Path,
    setflow_cache_receipt_path: Path,
    setflow_preflight_authorization_path: Path | None,
    strongest_score_physical_gpu: int,
    protocol_output: Path | None,
) -> dict[str, Any]:
    require(valid_head(current_head), "expected current Git HEAD is invalid")
    require(
        int(strongest_score_physical_gpu) in range(6),
        "strongest closed-score physical GPU is outside 0-5",
    )
    require(
        PYTHON.is_file() and STRONGEST_SCORE_PRODUCER.is_file(),
        "strongest closed-score CUDA producer runtime is absent",
    )
    require(
        command(["git", "rev-parse", "HEAD"]).stdout.strip() == current_head
        and not command(["git", "status", "--porcelain"]).stdout.strip(),
        "A100 worktree is not clean at expected bridge runner HEAD",
    )
    for path, label in (
        (base_protocol_path, "base guidance protocol"),
        (setflow_posttraining_schedule_path, "SetFlow posttraining schedule"),
        (setflow_recovered_protocol_path, "SetFlow recovered protocol"),
        (setflow_posttraining_runtime_path, "SetFlow posttraining runtime"),
        (setflow_confirmation_gate_path, "SetFlow confirmation gate"),
        (critic_loso_runtime_path, "Critic LOSO runtime"),
        (critic_readiness_path, "Critic readiness"),
        (critic_refit_manifest_path, "Critic refit manifest"),
        (critic_preflight_path, "Critic preflight"),
        (setflow_preflight_path, "SetFlow preflight"),
        (critic_cache_summary_path, "Critic cache summary"),
        (setflow_cache_receipt_path, "SetFlow cache receipt"),
        (
            runner_verification_receipt_path,
            "successor runner verification receipt",
        ),
    ):
        if path not in {base_protocol_path, setflow_recovered_protocol_path}:
            route2_path(path, label)
        require(path.is_file(), f"{label} is absent: {path}")

    runner_verification = consume_runner_verification_receipt_v403(
        runner_verification_receipt_path, current_head
    )

    base = read_json(base_protocol_path)
    require(
        critic_readiness_path == Path(str(base["critic_readiness_path"]))
        and critic_refit_manifest_path
        == Path(str(base["critic_refit_manifest_path"])),
        "Critic guidance inputs left the canonical posttraining family",
    )
    schedule = read_json(setflow_posttraining_schedule_path)
    bindings = schedule.get("posttraining_bindings", {})
    manifest_path = Path(str(bindings.get("config_manifest_path", "")))
    require(manifest_path.is_file(), f"SetFlow config manifest is absent: {manifest_path}")
    route2_path(manifest_path, "SetFlow config manifest")
    setflow_lineage = validate_setflow_lineage_v403(
        schedule,
        read_json(setflow_posttraining_runtime_path),
        read_json(setflow_confirmation_gate_path),
        read_json(setflow_recovered_protocol_path),
        read_json(manifest_path),
        schedule_path=setflow_posttraining_schedule_path,
        runtime_path=setflow_posttraining_runtime_path,
        gate_path=setflow_confirmation_gate_path,
        recovered_protocol_path=setflow_recovered_protocol_path,
        manifest_path=manifest_path,
    )
    critic_lineage = validate_critic_lineage_v403(
        read_json(critic_loso_runtime_path),
        read_json(critic_readiness_path),
        read_json(critic_refit_manifest_path),
        runtime_path=critic_loso_runtime_path,
        readiness_path=critic_readiness_path,
        refit_manifest_path=critic_refit_manifest_path,
    )
    critic_preflight = read_json(critic_preflight_path)
    critic_authorization_path = Path(str(critic_preflight["authorization_path"]))
    setflow_preflight_authorization_path = (
        setflow_preflight_authorization_path
        or critic_authorization_path.with_name("setflow.json")
    )
    for path, label in (
        (critic_authorization_path, "Critic preflight authorization"),
        (setflow_preflight_authorization_path, "SetFlow preflight authorization"),
    ):
        route2_path(path, label)
        require(path.is_file(), f"{label} is absent: {path}")
    prerequisite_lineage = validate_preflight_cache_lineage_v403(
        critic_preflight,
        read_json(setflow_preflight_path),
        read_json(critic_authorization_path),
        read_json(setflow_preflight_authorization_path),
        read_json(critic_cache_summary_path),
        read_json(setflow_cache_receipt_path),
        critic_preflight_path=critic_preflight_path,
        setflow_preflight_path=setflow_preflight_path,
        critic_authorization_path=critic_authorization_path,
        setflow_authorization_path=setflow_preflight_authorization_path,
        critic_cache_summary_path=critic_cache_summary_path,
        setflow_cache_receipt_path=setflow_cache_receipt_path,
        source_token_cache_path=Path(str(base["source_token_cache_path"])),
    )
    prerequisite_lineage["successor_runner_verification"] = runner_verification
    derived = build_derived_guidance_protocol_v403(
        base,
        runner_head=current_head,
        setflow_lineage=setflow_lineage,
        critic_lineage=critic_lineage,
        prerequisite_lineage=prerequisite_lineage,
    )
    config_family = Path(str(derived["runtime_config_root"])).parent
    protocol_output = protocol_output or config_family / "guidance_protocol.json"
    route2_path(protocol_output, "derived guidance protocol")
    require(
        protocol_output.parent == config_family,
        "derived guidance protocol is outside its one-shot config family",
    )
    receipt_path = config_family / "bridge_launch_receipt.json"
    targets = (
        protocol_output,
        receipt_path,
        Path(str(derived["authorization_output"])),
        Path(str(derived["guidance_authorization_decision_output"])),
        Path(str(derived["runtime_config_root"])),
        Path(str(derived["guidance_screen_output_root"])),
        Path(str(derived["guidance_screen_runtime_root"])),
        Path(str(derived["guidance_screen_log_root"])),
        Path(str(derived["final_runtime_root"])),
        Path(str(derived["final_log_root"])),
        Path(str(derived["strongest_closed_score_config_path"])),
        Path(str(derived["strongest_closed_score_failure_path"])),
        Path(str(derived["strongest_closed_score_summary_path"])).parent,
    )
    require(
        all(not path.exists() for path in targets),
        "V4.0.3 guidance bridge is one-shot and an output family already exists",
    )
    write_new_atomic(protocol_output, derived)
    authorization = authorize_guidance(
        current_head,
        protocol_path=protocol_output,
        critic_runtime_path=critic_loso_runtime_path,
        critic_runtime_head=str(critic_lineage["training_runner_git_head"]),
        setflow_runtime_path=setflow_posttraining_runtime_path,
        setflow_runtime_head=str(setflow_lineage["posttraining_runner_git_head"]),
        decision_output=Path(
            str(derived["guidance_authorization_decision_output"])
        ),
    )
    require(
        authorization.get("status") == "XEDITFLOW_V4_GUIDANCE_AUTHORIZED"
        and authorization.get("guidance_authorized") is True,
        "V4.0.3 dual readiness did not authorize guidance",
    )
    strongest_score_lineage = materialize_strongest_closed_score_v403(
        derived,
        physical_gpu_index=strongest_score_physical_gpu,
    )
    launch = launch_guidance_screen(
        current_head,
        current_head,
        protocol_path=protocol_output,
        authorization_decision_path=Path(
            str(derived["guidance_authorization_decision_output"])
        ),
        critic_preflight_path=critic_preflight_path,
        critic_preflight_head=str(
            prerequisite_lineage["critic_preflight_runner_git_head"]
        ),
        setflow_preflight_path=setflow_preflight_path,
        setflow_preflight_head=str(
            prerequisite_lineage["setflow_preflight_runner_git_head"]
        ),
        execution_runtime_root=Path(
            str(derived["guidance_screen_runtime_root"])
        ),
        execution_log_root=Path(str(derived["guidance_screen_log_root"])),
    )
    final_successor = build_final_successor_v403(
        current_head=current_head,
        protocol_path=protocol_output,
        guidance_runtime_path=Path(str(launch["runtime_manifest"])),
        guidance_screen_gate_path=Path(
            str(launch["guidance_screen_gate_path"])
        ),
        derived=derived,
        prerequisite_lineage=prerequisite_lineage,
        strongest_score_lineage=strongest_score_lineage,
    )
    receipt = {
        "schema_version": (
            "route_a_v3_route2_xeditflow_v403_guidance_bridge_launch.v1"
        ),
        "status": "XEDITFLOW_V403_GUIDANCE_SCREEN_SCHEDULER_LAUNCHED",
        "bridge_runner_git_head": current_head,
        "derived_protocol_path": str(protocol_output),
        "authorization_decision_path": derived[
            "guidance_authorization_decision_output"
        ],
        "guidance_authorization_output": derived["authorization_output"],
        "guidance_screen_runtime_path": launch["runtime_manifest"],
        "guidance_screen_gate_path": launch["guidance_screen_gate_path"],
        "guidance_screen_scheduler_pid": launch["scheduler_pid"],
        "final_successor": final_successor,
        "critic_training_runner_git_head": critic_lineage[
            "training_runner_git_head"
        ],
        "setflow_training_runner_git_head": setflow_lineage[
            "training_runner_git_head"
        ],
        "setflow_posttraining_runner_git_head": setflow_lineage[
            "posttraining_runner_git_head"
        ],
        "critic_preflight_runner_git_head": prerequisite_lineage[
            "critic_preflight_runner_git_head"
        ],
        "setflow_preflight_runner_git_head": prerequisite_lineage[
            "setflow_preflight_runner_git_head"
        ],
        "critic_cache_experiment_git_head": prerequisite_lineage[
            "critic_cache_experiment_git_head"
        ],
        "setflow_cache_experiment_git_head": prerequisite_lineage[
            "setflow_cache_experiment_git_head"
        ],
        "runner_verification_receipt": runner_verification,
        "strongest_matched_baseline_closed_score": strongest_score_lineage,
        "scientific_protocol_changed": False,
        "guidance_grid_changed": False,
        "seed_inventory_changed": False,
        "budget_changed": False,
        "free_memory_gate_applied": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_new_atomic(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument(
        "--runner-verification-receipt", required=True, type=Path
    )
    parser.add_argument("--base-protocol", type=Path, default=BASE_PROTOCOL)
    parser.add_argument("--setflow-posttraining-schedule", required=True, type=Path)
    parser.add_argument("--setflow-recovered-protocol", required=True, type=Path)
    parser.add_argument("--setflow-posttraining-runtime", required=True, type=Path)
    parser.add_argument("--setflow-confirmation-gate", required=True, type=Path)
    parser.add_argument("--critic-loso-runtime", required=True, type=Path)
    parser.add_argument("--critic-readiness", type=Path)
    parser.add_argument("--critic-refit-manifest", type=Path)
    parser.add_argument("--critic-preflight", type=Path, default=CRITIC_PREFLIGHT)
    parser.add_argument("--setflow-preflight", type=Path, default=SETFLOW_PREFLIGHT)
    parser.add_argument(
        "--critic-cache-summary", type=Path, default=CRITIC_CACHE_SUMMARY
    )
    parser.add_argument(
        "--setflow-cache-receipt", type=Path, default=SETFLOW_CACHE_RECEIPT
    )
    parser.add_argument(
        "--strongest-score-physical-gpu", required=True, type=int
    )
    parser.add_argument("--setflow-preflight-authorization", type=Path)
    parser.add_argument("--protocol-output", type=Path)
    arguments = parser.parse_args()
    base = read_json(arguments.base_protocol)
    critic_readiness = arguments.critic_readiness or Path(
        str(base["critic_readiness_path"])
    )
    critic_refit_manifest = arguments.critic_refit_manifest or Path(
        str(base["critic_refit_manifest_path"])
    )
    print(
        json.dumps(
            run(
                arguments.expected_head,
                runner_verification_receipt_path=(
                    arguments.runner_verification_receipt
                ),
                base_protocol_path=arguments.base_protocol,
                setflow_posttraining_schedule_path=(
                    arguments.setflow_posttraining_schedule
                ),
                setflow_recovered_protocol_path=arguments.setflow_recovered_protocol,
                setflow_posttraining_runtime_path=arguments.setflow_posttraining_runtime,
                setflow_confirmation_gate_path=arguments.setflow_confirmation_gate,
                critic_loso_runtime_path=arguments.critic_loso_runtime,
                critic_readiness_path=critic_readiness,
                critic_refit_manifest_path=critic_refit_manifest,
                critic_preflight_path=arguments.critic_preflight,
                setflow_preflight_path=arguments.setflow_preflight,
                critic_cache_summary_path=arguments.critic_cache_summary,
                setflow_cache_receipt_path=arguments.setflow_cache_receipt,
                setflow_preflight_authorization_path=(
                    arguments.setflow_preflight_authorization
                ),
                strongest_score_physical_gpu=(
                    arguments.strongest_score_physical_gpu
                ),
                protocol_output=arguments.protocol_output,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
