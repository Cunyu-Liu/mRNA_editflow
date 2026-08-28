from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

import scripts.route_a_v3.train_route2_xeditsetflow_s1 as trainer
from core.route2_xeditsetflow_s1 import mixture_setflow_loss_s1
from scripts.route_a_v3.train_route2_xeditsetflow_s1 import (
    AUTHORIZATION_SCHEMA,
    AUTHORIZATION_STATUS,
    CONFIRMATION_AUTHORIZATION_SCHEMA,
    CONFIRMATION_AUTHORIZATION_STATUS,
    CONFIRMATION_CONFIG_SCHEMA,
    CONFIRMATION_SEEDS,
    OBJECTIVE_IDENTITY,
    OBJECTIVE_WEIGHT,
    RUN_IDS,
    SCREEN_RUNNER_GIT_HEAD,
    SetFlowTrainingS1Error,
    _write_atomic_terminal_s1,
    build_seeded_setflow_model_s1,
    complete_attempt_then_publish_training_summary_s1,
    derive_training_update_geometry_s1,
    pass_complete_alive_event_s1,
    record_training_attempt_s1,
    require_s1_confirmation_launch_authorization,
    require_s1_launch_authorization,
    setflow_training_stage_seed_s1,
    training_summary_identity_s1,
    training_attempt_identity_s1,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/train_route2_xeditsetflow_s1.py"


def _confirmation_config(seed: int = 20260912) -> dict:
    return {
        "schema_version": CONFIRMATION_CONFIG_SCHEMA,
        "status": "FROZEN_S1_CONFIRMATION_CONFIG_NOT_STARTED",
        "run_stage": "CONFIRMATION",
        "training_seed": seed,
        "selected_model": "v4_s1_full",
        "required_confirmation_seeds": list(CONFIRMATION_SEEDS),
        "additional_seed_authorized": False,
        "objective_identity": OBJECTIVE_IDENTITY,
        "cross_state_candidate_mode_responsibility_weight": OBJECTIVE_WEIGHT,
        "confirmation_runner_git_head": "a" * 40,
        "screen_runner_git_head": SCREEN_RUNNER_GIT_HEAD,
        "screen_gate_path": "/exact/screen_gate.json",
        "screen_selected_checkpoint_pass": 8,
        "screen_provenance": {
            "checkpoint_decisions": {
                "v4_s1_full": {"selected": 8},
                "v4_s1_single_mode": {"selected": 6},
            }
        },
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def _confirmation_authorization() -> dict:
    config = _confirmation_config()
    return {
        "schema_version": CONFIRMATION_AUTHORIZATION_SCHEMA,
        "status": CONFIRMATION_AUTHORIZATION_STATUS,
        "authorized_git_head": "a" * 40,
        "authorized_run_ids": ["v4_s1_full"],
        "authorized_seeds": list(CONFIRMATION_SEEDS),
        "objective_identity": OBJECTIVE_IDENTITY,
        "cross_state_candidate_mode_responsibility_weight": OBJECTIVE_WEIGHT,
        "screen_runner_git_head": SCREEN_RUNNER_GIT_HEAD,
        "screen_gate_path": config["screen_gate_path"],
        "screen_selected_checkpoint_pass": config[
            "screen_selected_checkpoint_pass"
        ],
        "screen_provenance": config["screen_provenance"],
        "free_memory_gate_applied": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def test_s1_terminal_artifact_is_atomic_and_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "training_summary.json"
    payload = {"status": "TERMINAL_XEDITSETFLOW_V4_S1_TRAINING_COMPLETE_PENDING_VALIDATION"}
    _write_atomic_terminal_s1(output, payload)
    assert json.loads(output.read_text()) == payload
    with pytest.raises(SetFlowTrainingS1Error, match="already exists"):
        _write_atomic_terminal_s1(output, payload)


def test_s1_update_budget_and_alive_event_reuse_exact_v4_geometry() -> None:
    assert derive_training_update_geometry_s1(101) == {
        "train_source_count": 101,
        "sources_per_update": 8,
        "states_per_source": 4,
        "effective_state_batch": 32,
        "updates_per_pass": 13,
        "pass_count": 10,
        "total_optimizer_updates": 130,
    }
    event = pass_complete_alive_event_s1(
        run_id="v4_s1_full", pass_number=4, update_count=52
    )
    assert event["active_performance_metric_emitted"] is False
    assert not any("loss" in key or "nll" in key or "recovery" in key for key in event)


def test_s1_loss_call_uses_exact_imported_contract_and_metadata() -> None:
    parameters = inspect.signature(mixture_setflow_loss_s1).parameters
    assert tuple(parameters)[-4:] == (
        "state_slots",
        "source_occurrence_ids",
        "canonical_candidate_indices",
        "cross_state_candidate_mode_responsibility_weight",
    )
    source = SCRIPT.read_text()
    assert source.count("objective = mixture_setflow_loss_s1(") == 1
    for key in (
        'batch["state_slots"]',
        'batch["source_occurrence_ids"]',
        'batch["canonical_candidate_indices"]',
        '"active_responsibility_constraint_count"',
        '"active_responsibility_candidate_count"',
        '"active_responsibility_occurrence_count"',
    ):
        assert key in source
    assert source.count("objective.total.backward()") == 1


def test_s1_authorization_is_exact_head_objective_and_protected_read_closed() -> None:
    config = {
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_s1_mechanics_screen_config.v1"
    }
    authorization = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "status": AUTHORIZATION_STATUS,
        "authorized_git_head": "a" * 40,
        "authorized_run_ids": list(RUN_IDS),
        "objective_identity": OBJECTIVE_IDENTITY,
        "cross_state_candidate_mode_responsibility_weight": OBJECTIVE_WEIGHT,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    preflight = {
        "status": "XEDITSETFLOW_V4_PREFLIGHT_PASS",
        "passed": True,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    audit = {
        "status": "XEDITSETFLOW_V4_SOURCE_LEVEL_DATA_AUDIT_PASS",
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    require_s1_launch_authorization(
        config,
        authorization,
        preflight,
        audit,
        run_id="v4_s1_full",
        current_git_head="a" * 40,
    )
    authorization["development_test_outcome_reads"] = 1
    with pytest.raises(SetFlowTrainingS1Error, match="protected read"):
        require_s1_launch_authorization(
            config,
            authorization,
            preflight,
            audit,
            run_id="v4_s1_full",
            current_git_head="a" * 40,
        )


def test_s1_trainer_is_train_only_cuda_a100_bf16_and_has_no_critic_gradient() -> None:
    source = SCRIPT.read_text()
    tree = ast.parse(source)
    imported = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert not any("critic" in module or "evaluator" in module for module in imported)
    for token in (
        'allowed_splits=("TRAIN",)',
        '"A100" in device_name',
        "torch.cuda.is_bf16_supported()",
        '"cpu_fallback_used": False',
        '"development_test_outcome_reads": 0',
        '"new_final_evaluation_outcome_reads": 0',
    ):
        assert token in source
    assert 'run_stage == "CONFIRMATION"' in source
    assert "free_memory_gate_applied" in source


def test_s1_stage_seed_contract_keeps_screen_and_allows_only_three_full_confirmation_seeds() -> None:
    screen = {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v4_s1_mechanics_screen_config.v1"
        ),
        "training": {"screen_seed": 20260911},
    }
    assert setflow_training_stage_seed_s1(screen, run_id="v4_s1_full") == (
        "SCREEN",
        20260911,
    )
    for seed in CONFIRMATION_SEEDS:
        assert setflow_training_stage_seed_s1(
            _confirmation_config(seed), run_id="v4_s1_full"
        ) == ("CONFIRMATION", seed)
    with pytest.raises(SetFlowTrainingS1Error, match="only permits"):
        setflow_training_stage_seed_s1(
            _confirmation_config(), run_id="v4_s1_single_mode"
        )
    with pytest.raises(SetFlowTrainingS1Error, match="seed cohort"):
        setflow_training_stage_seed_s1(
            _confirmation_config(20260915), run_id="v4_s1_full"
        )
    changed = _confirmation_config()
    changed["run_stage"] = "POSTTRAINING"
    with pytest.raises(SetFlowTrainingS1Error, match="run stage"):
        setflow_training_stage_seed_s1(changed, run_id="v4_s1_full")


def test_s1_parameter_seed_is_applied_before_model_construction_for_all_confirmation_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, int | str]] = []

    monkeypatch.setattr(
        trainer.torch,
        "manual_seed",
        lambda seed: events.append(("cpu_seed", seed)),
    )
    monkeypatch.setattr(
        trainer.torch.cuda,
        "manual_seed_all",
        lambda seed: events.append(("cuda_seed", seed)),
    )

    def fake_build(_config, _vocabs, *, run_id):
        events.append(("build", run_id))
        return object(), {"trainable_parameter_count": 1}

    monkeypatch.setattr(trainer, "build_setflow_screen_model_s1", fake_build)
    for seed in CONFIRMATION_SEEDS:
        events.clear()
        model, capacity = build_seeded_setflow_model_s1(
            _confirmation_config(seed),
            {},
            run_id="v4_s1_full",
            training_seed=seed,
        )
        assert model is not None
        assert capacity == {"trainable_parameter_count": 1}
        assert events == [
            ("cpu_seed", seed),
            ("cuda_seed", seed),
            ("build", "v4_s1_full"),
        ]
    with pytest.raises(SetFlowTrainingS1Error, match="undeclared"):
        build_seeded_setflow_model_s1(
            _confirmation_config(),
            {},
            run_id="v4_s1_full",
            training_seed=20260915,
        )
    source = SCRIPT.read_text()
    assert source.count("torch.manual_seed(training_seed)") == 1
    assert source.count("torch.cuda.manual_seed_all(training_seed)") == 1


def test_s1_confirmation_authorization_binds_seed_head_objective_and_screen_provenance() -> None:
    config = _confirmation_config()
    authorization = _confirmation_authorization()
    preflight = {
        "status": "XEDITSETFLOW_V4_PREFLIGHT_PASS",
        "passed": True,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    audit = {
        "status": "XEDITSETFLOW_V4_SOURCE_LEVEL_DATA_AUDIT_PASS",
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    require_s1_confirmation_launch_authorization(
        config,
        authorization,
        preflight,
        audit,
        run_id="v4_s1_full",
        training_seed=20260912,
        current_git_head="a" * 40,
    )
    wrong_config_head = dict(config)
    wrong_config_head["confirmation_runner_git_head"] = "b" * 40
    with pytest.raises(SetFlowTrainingS1Error, match="Git HEAD disagree"):
        require_s1_confirmation_launch_authorization(
            wrong_config_head,
            authorization,
            preflight,
            audit,
            run_id="v4_s1_full",
            training_seed=20260912,
            current_git_head="a" * 40,
        )
    for field, value in (
        ("schema_version", AUTHORIZATION_SCHEMA),
        ("authorized_seeds", [20260912]),
        ("screen_selected_checkpoint_pass", 4),
        ("free_memory_gate_applied", True),
        ("development_test_outcome_reads", 1),
    ):
        changed = dict(authorization)
        changed[field] = value
        with pytest.raises(SetFlowTrainingS1Error):
            require_s1_confirmation_launch_authorization(
                config,
                changed,
                preflight,
                audit,
                run_id="v4_s1_full",
                training_seed=20260912,
                current_git_head="a" * 40,
            )


def test_s1_confirmation_summary_identity_is_explicit_and_not_screen_checkpoint_selection() -> None:
    identity = training_summary_identity_s1(
        _confirmation_config(20260914), run_id="v4_s1_full"
    )
    assert identity == {
        "run_stage": "CONFIRMATION",
        "run_id": "v4_s1_full",
        "seed": 20260914,
        "selected_model": "v4_s1_full",
        "objective_identity": OBJECTIVE_IDENTITY,
        "cross_state_candidate_mode_responsibility_weight": OBJECTIVE_WEIGHT,
        "parameter_initialization_seed": 20260914,
        "parameter_initialization_seed_applied_before_model_construction": True,
    }
    assert "checkpoint_pass" not in identity


def test_s1_attempt_json_retains_objective_identity_weight_and_count(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "training_attempt.json"
    record_training_attempt_s1(
        tmp_path / "attempts.csv",
        attempt,
        {
            "attempt_id": "xeditsetflow_v4_s1_full_seed20260911",
            "status": "COMPLETED",
        },
        objective_details={
            "objective_identity": OBJECTIVE_IDENTITY,
            "cross_state_candidate_mode_responsibility_weight": OBJECTIVE_WEIGHT,
            "parameter_initialization_seed": 20260911,
            "parameter_initialization_seed_applied_before_model_construction": True,
            "active_responsibility_constraint_count": 123,
        },
    )
    payload = json.loads(attempt.read_text())
    assert payload["objective_identity"] == OBJECTIVE_IDENTITY
    assert payload["cross_state_candidate_mode_responsibility_weight"] == .05
    assert payload["parameter_initialization_seed"] == 20260911
    assert (
        payload["parameter_initialization_seed_applied_before_model_construction"]
        is True
    )
    assert payload["active_responsibility_constraint_count"] == 123


def test_s1_attempt_identity_is_unique_to_the_independent_runner_family() -> None:
    first = training_attempt_identity_s1(
        run_id="v4_s1_full",
        training_seed=20260911,
        runner_git_head="a" * 40,
    )
    retry = training_attempt_identity_s1(
        run_id="v4_s1_full",
        training_seed=20260911,
        runner_git_head="b" * 40,
    )
    assert first != retry
    assert first.endswith("_runner_" + "a" * 40)


def test_s1_success_summary_is_not_published_when_ledger_completion_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary_path = tmp_path / "training_summary.json"

    def fail_ledger(*_args, **_kwargs):
        raise RuntimeError("ledger completion failed")

    monkeypatch.setattr(trainer, "record_training_attempt_s1", fail_ledger)
    with pytest.raises(RuntimeError, match="ledger completion failed"):
        complete_attempt_then_publish_training_summary_s1(
            ledger_path=tmp_path / "attempts.csv",
            attempt_path=tmp_path / "training_attempt.json",
            attempt_row={"attempt_id": "attempt", "status": "COMPLETED"},
            objective_details={"objective_identity": OBJECTIVE_IDENTITY},
            summary_path=summary_path,
            summary={"status": "TERMINAL"},
        )
    assert not summary_path.exists()
