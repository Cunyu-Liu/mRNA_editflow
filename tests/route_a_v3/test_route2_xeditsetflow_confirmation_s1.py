from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from core.route2_xeditsetflow_confirmation_s1 import (
    CONFIRMATION_RUNTIME_SCHEMA,
    CONFIRMATION_RUNTIME_STATUS,
    OBJECTIVE_IDENTITY,
    OBJECTIVE_WEIGHT,
    SCREEN_HEAD,
    XEditSetFlowConfirmationS1Error,
    build_confirmation_configs_s1,
    materialize_confirmation_configs_s1,
    validate_screen_pass_barrier_s1,
)
from core.route2_xeditsetflow_gate_s1 import select_checkpoint_s1


ROOT = Path(__file__).resolve().parents[2]
BASE = json.loads(
    (
        ROOT
        / "configs/route_a_v3_route2_xeditsetflow_v4_s1_mechanics_screen_v1.json"
    ).read_text(encoding="utf-8")
)
PROTOCOL = json.loads(
    (
        ROOT
        / "configs/route_a_v3_route2_xeditsetflow_v4_s1_confirmation_protocol_v1.json"
    ).read_text(encoding="utf-8")
)


def _row(checkpoint_pass: int, *, full: bool) -> dict:
    return {
        "run_id": "v4_s1_full" if full else "v4_s1_single_mode",
        "checkpoint_pass": checkpoint_pass,
        "common_validation_set_marginal_nll": 1.9 + checkpoint_pass / 1000,
        "source_macro_candidate_recovery_rate": (
            0.40 + checkpoint_pass / 1000 if full else 0.36 + checkpoint_pass / 2000
        ),
        "source_macro_measured_top_k_recovery_at_k": 0.24,
        "source_macro_unique_candidate_rate": (
            0.95 + checkpoint_pass / 10000
            if full
            else 0.90 + checkpoint_pass / 20000
        ),
        "checks": {"all": True},
        "eligible": True,
    }


def _bundle(tmp_path: Path) -> dict:
    protocol = copy.deepcopy(PROTOCOL)
    family = tmp_path / f"s1_screen_seed_20260911_runner_{SCREEN_HEAD}"
    schedule_path = family / "schedule.json"
    runtime_config_path = family / "runtime_config.json"
    authorization_path = tmp_path / f"s1_screen_seed_20260911_runner_{SCREEN_HEAD}.json"
    gate_path = family / "screen_gate.json"
    protocol["screen_provenance"].update(
        {
            "schedule_path": str(schedule_path),
            "runtime_config_path": str(runtime_config_path),
            "authorization_path": str(authorization_path),
            "screen_gate_path": str(gate_path),
        }
    )
    protocol["runner_outputs"].update(
        {
            "runtime_config_root_template": str(
                tmp_path / "runtime_configs_{runner_git_head}"
            ),
            "training_runtime_root_template": str(
                tmp_path / "training_{runner_git_head}"
            ),
            "posttraining_runtime_root_template": str(
                tmp_path / "posttraining_{runner_git_head}"
            ),
            "confirmation_gate_output_template": str(
                tmp_path / "posttraining_{runner_git_head}" / "confirmation_gate.json"
            ),
        }
    )
    output_root = family
    validation_root = family / "outcome_free_validation_generation"
    runtime_config = copy.deepcopy(BASE)
    runtime_config.update(
        {
            "runner_git_head": SCREEN_HEAD,
            "run_stage": "SCREEN",
            "output_root": str(output_root),
            "validation_output_root": str(validation_root),
            "screen_gate_output_path": str(gate_path),
        }
    )
    authorization = {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v4_s1_screen_launch_authorization.v1"
        ),
        "status": "XEDITSETFLOW_V4_S1_SCREEN_LAUNCH_AUTHORIZED",
        "authorized_git_head": SCREEN_HEAD,
        "authorized_run_ids": ["v4_s1_full", "v4_s1_single_mode"],
        "screen_seed": 20260911,
        "objective_identity": OBJECTIVE_IDENTITY,
        "cross_state_candidate_mode_responsibility_weight": OBJECTIVE_WEIGHT,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    rows = {
        "v4_s1_full": {
            str(checkpoint_pass): _row(checkpoint_pass, full=True)
            for checkpoint_pass in (4, 6, 8, 10)
        },
        "v4_s1_single_mode": {
            str(checkpoint_pass): _row(checkpoint_pass, full=False)
            for checkpoint_pass in (4, 6, 8, 10)
        },
    }
    decisions = {
        run_id: select_checkpoint_s1(
            {int(key): value for key, value in run_rows.items()}
        )
        for run_id, run_rows in rows.items()
    }
    selected_pass = decisions["v4_s1_full"][
        "generation_constrained_selected_checkpoint"
    ]["checkpoint_pass"]
    gate = {
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_s1_screen_gate.v1",
        "status": "XEDITSETFLOW_V4_S1_SCREEN_PASS",
        "screen_seed": 20260911,
        "checkpoint_rows": rows,
        "checkpoint_decisions": decisions,
        "selected_checkpoint_pass": selected_pass,
        "successor_protocol_required": True,
        "s1_mechanics_screen_passed": True,
        "legacy_v4_confirmation_authorized": False,
        "confirmation_authorized": False,
        "confirmation_seeds": [],
        "additional_seed_authorized": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    training_jobs = [
        {
            "run_id": run_id,
            "terminal_summary": str(output_root / run_id / "training_summary.json"),
        }
        for run_id in ("v4_s1_full", "v4_s1_single_mode")
    ]
    validation_jobs = [
        {
            "run_id": run_id,
            "checkpoint_pass": checkpoint_pass,
            "terminal_summary": str(
                validation_root
                / run_id
                / f"pass_{checkpoint_pass}"
                / "validation_summary.json"
            ),
        }
        for run_id in ("v4_s1_full", "v4_s1_single_mode")
        for checkpoint_pass in (4, 6, 8, 10)
    ]
    schedule = {
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_s1_screen_schedule.v1",
        "status": "FROZEN_XEDITSETFLOW_V4_S1_SCREEN_SCHEDULE",
        "git_head": SCREEN_HEAD,
        "runtime_config": str(runtime_config_path),
        "authorization": str(authorization_path),
        "objective_identity": OBJECTIVE_IDENTITY,
        "cross_state_candidate_mode_responsibility_weight": OBJECTIVE_WEIGHT,
        "training_queues": [{"jobs": training_jobs}],
        "validation_queues": [{"jobs": validation_jobs}],
        "adjudication": {"gate_path": str(gate_path)},
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    return {
        "protocol": protocol,
        "schedule": schedule,
        "runtime_config": runtime_config,
        "authorization": authorization,
        "gate": gate,
        "schedule_path": schedule_path,
        "runtime_config_path": runtime_config_path,
        "authorization_path": authorization_path,
        "gate_path": gate_path,
    }


def _build(bundle: dict, *, confirmation_head: str = "a" * 40) -> list[dict]:
    return build_confirmation_configs_s1(
        BASE,
        bundle["protocol"],
        bundle["schedule"],
        bundle["runtime_config"],
        bundle["authorization"],
        bundle["gate"],
        screen_schedule_path=bundle["schedule_path"],
        screen_runtime_config_path=bundle["runtime_config_path"],
        screen_authorization_path=bundle["authorization_path"],
        screen_gate_path=bundle["gate_path"],
        confirmation_runner_git_head=confirmation_head,
    )


def test_protocol_freezes_three_full_only_twelve_validation_and_gpu_boundaries() -> None:
    assert PROTOCOL["selected_model"] == "v4_s1_full"
    assert PROTOCOL["required_seeds"] == [20260912, 20260913, 20260914]
    assert PROTOCOL["confirmation_design"]["training_job_count"] == 3
    assert PROTOCOL["confirmation_design"]["single_mode_training_job_count"] == 0
    assert PROTOCOL["confirmation_design"]["checkpoint_validation_job_count"] == 12
    assert PROTOCOL["training_policy"]["saved_checkpoint_passes"] == [4, 6, 8, 10]
    assert PROTOCOL["gpu_policy"]["cuda_bf16_only"] is True
    assert PROTOCOL["gpu_policy"]["cpu_fallback"] is False
    assert PROTOCOL["gpu_policy"]["free_or_estimated_memory_gate"] is False
    assert PROTOCOL["package_failure_policy"][
        "first_terminal_failure_stops_pending_launches"
    ] is True
    assert PROTOCOL["package_failure_policy"][
        "technical_failure_is_scientific_no_go"
    ] is False
    assert PROTOCOL["development_test_outcome_reads"] == 0
    assert PROTOCOL["new_final_evaluation_outcome_reads"] == 0


def test_exact_pass_builds_three_full_configs_and_keeps_screen_selection_provenance_only(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    configs = _build(bundle)
    assert [config["training_seed"] for config in configs] == [
        20260912,
        20260913,
        20260914,
    ]
    assert all(config["schema_version"] == CONFIRMATION_RUNTIME_SCHEMA for config in configs)
    assert all(config["status"] == CONFIRMATION_RUNTIME_STATUS for config in configs)
    assert all(config["run_stage"] == "CONFIRMATION" for config in configs)
    assert all(config["selected_model"] == "v4_s1_full" for config in configs)
    assert all(config["screen_runner_git_head"] == SCREEN_HEAD for config in configs)
    assert all(config["objective_identity"] == OBJECTIVE_IDENTITY for config in configs)
    assert all(
        config["cross_state_candidate_mode_responsibility_weight"] == 0.05
        for config in configs
    )
    assert all(
        set(config["screen_provenance"]["checkpoint_decisions"])
        == {"v4_s1_full", "v4_s1_single_mode"}
        for config in configs
    )
    assert all("confirmation_selected_checkpoint_pass" not in config for config in configs)
    assert all(config["confirmation_training_job_count"] == 3 for config in configs)
    assert all(config["confirmation_checkpoint_validation_job_count"] == 12 for config in configs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "XEDITSETFLOW_V4_S1_SCREEN_NO_GO"),
        ("successor_protocol_required", False),
        ("s1_mechanics_screen_passed", False),
        ("confirmation_authorized", True),
        ("development_test_outcome_reads", 1),
    ],
)
def test_only_exact_s1_pass_with_zero_protected_reads_can_build(
    tmp_path: Path, field: str, value: object
) -> None:
    bundle = _bundle(tmp_path)
    bundle["gate"][field] = value
    with pytest.raises(XEditSetFlowConfirmationS1Error):
        _build(bundle)


def test_head_path_objective_and_checkpoint_decision_drift_are_rejected(
    tmp_path: Path,
) -> None:
    mutations = []
    bundle = _bundle(tmp_path / "head")
    bundle["schedule"]["git_head"] = "b" * 40
    mutations.append(bundle)
    bundle = _bundle(tmp_path / "path")
    bundle["schedule"]["runtime_config"] = str(tmp_path / "wrong.json")
    mutations.append(bundle)
    bundle = _bundle(tmp_path / "objective")
    bundle["authorization"]["cross_state_candidate_mode_responsibility_weight"] = 0.04
    mutations.append(bundle)
    bundle = _bundle(tmp_path / "decision")
    bundle["gate"]["checkpoint_decisions"]["v4_s1_full"][
        "eligible_checkpoint_passes"
    ] = [4]
    mutations.append(bundle)
    bundle = _bundle(tmp_path / "selected")
    bundle["gate"]["selected_checkpoint_pass"] = 4
    mutations.append(bundle)
    for changed in mutations:
        with pytest.raises(XEditSetFlowConfirmationS1Error):
            _build(changed)


def test_config_package_is_atomic_and_final_or_partial_family_evidence_forbids_reuse(
    tmp_path: Path,
) -> None:
    confirmation_head = "a" * 40
    bundle = _bundle(tmp_path / "success")
    configs = _build(bundle, confirmation_head=confirmation_head)
    manifest = materialize_confirmation_configs_s1(
        configs,
        bundle["protocol"],
        confirmation_runner_git_head=confirmation_head,
    )
    config_root = Path(
        bundle["protocol"]["runner_outputs"]["runtime_config_root_template"].format(
            runner_git_head=confirmation_head
        )
    )
    assert not config_root.with_name(config_root.name + ".partial").exists()
    assert json.loads((config_root / "manifest.json").read_text()) == manifest
    assert manifest["status"] == "THREE_S1_CONFIRMATION_CONFIGS_PREPARED_NOT_STARTED"
    assert len(manifest["config_paths"]) == 3
    with pytest.raises(XEditSetFlowConfirmationS1Error, match="config root exists"):
        materialize_confirmation_configs_s1(
            configs,
            bundle["protocol"],
            confirmation_runner_git_head=confirmation_head,
        )

    for target in ("partial", "training_final", "posttraining_partial"):
        isolated = _bundle(tmp_path / target)
        isolated_configs = _build(isolated, confirmation_head=confirmation_head)
        outputs = isolated["protocol"]["runner_outputs"]
        if target == "partial":
            path = Path(outputs["runtime_config_root_template"].format(
                runner_git_head=confirmation_head
            ))
            evidence = path.with_name(path.name + ".partial")
        elif target == "training_final":
            evidence = Path(outputs["training_runtime_root_template"].format(
                runner_git_head=confirmation_head
            ))
        else:
            path = Path(outputs["posttraining_runtime_root_template"].format(
                runner_git_head=confirmation_head
            ))
            evidence = path.with_name(path.name + ".partial")
        evidence.mkdir(parents=True)
        with pytest.raises(XEditSetFlowConfirmationS1Error):
            materialize_confirmation_configs_s1(
                isolated_configs,
                isolated["protocol"],
                confirmation_runner_git_head=confirmation_head,
            )
        assert evidence.exists()
