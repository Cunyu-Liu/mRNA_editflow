from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import torch

import scripts.route_a_v3.validate_route2_xeditsetflow_s1_checkpoint as validator

from scripts.route_a_v3.validate_route2_xeditsetflow_s1_checkpoint import (
    CONFIRMATION_CONFIG_SCHEMA,
    CONFIRMATION_SEEDS,
    OBJECTIVE_IDENTITY,
    OBJECTIVE_WEIGHT,
    SetFlowCheckpointValidationS1Error,
    load_checkpoint_s1,
    require_training_package_provenance_s1,
    require_training_package_terminal_s1,
    setflow_validation_stage_seed_s1,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/validate_route2_xeditsetflow_s1_checkpoint.py"
HEAD = "a" * 40


def _training_summary(
    directory: Path,
    *,
    run_id: str = "v4_s1_full",
    run_stage: str = "SCREEN",
    seed: int = 20260911,
) -> dict:
    result = {
        "status": "TERMINAL_XEDITSETFLOW_V4_S1_TRAINING_COMPLETE_PENDING_VALIDATION",
        "run_stage": run_stage,
        "run_id": run_id,
        "seed": seed,
        "objective_identity": OBJECTIVE_IDENTITY,
        "cross_state_candidate_mode_responsibility_weight": OBJECTIVE_WEIGHT,
        "active_responsibility_constraint_count": 1,
        "parameter_initialization_seed": seed,
        "parameter_initialization_seed_applied_before_model_construction": True,
        "completed_passes": 10,
        "early_stopping_used": False,
        "update_geometry": {"pass_count": 10, "total_optimizer_updates": 100},
        "optimizer_update_count": 100,
        "parameter_changed": True,
        "physical_gpu_index": 0,
        "torch_device": "cuda:0",
        "device_name": "NVIDIA A100-SXM4-80GB",
        "training_precision": "BF16",
        "cuda_available": True,
        "bf16_supported": True,
        "cpu_fallback_used": False,
        "validation_generation_during_training": False,
        "checkpoint_selection_status": (
            "PENDING_TERMINAL_OUTCOME_FREE_VALIDATION_GENERATION"
        ),
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
        "saved_checkpoint_paths": {
            str(checkpoint_pass): str(directory / f"pass_{checkpoint_pass}.pt")
            for checkpoint_pass in (4, 6, 8, 10)
        },
    }
    if run_stage == "CONFIRMATION":
        result["selected_model"] = "v4_s1_full"
    return result


def _confirmation_config(tmp_path: Path, seed: int = 20260912) -> dict:
    return {
        "schema_version": CONFIRMATION_CONFIG_SCHEMA,
        "status": "FROZEN_S1_CONFIRMATION_CONFIG_NOT_STARTED",
        "run_stage": "CONFIRMATION",
        "training_seed": seed,
        "selected_model": "v4_s1_full",
        "required_confirmation_seeds": list(CONFIRMATION_SEEDS),
        "additional_seed_authorized": False,
        "output_root": str(tmp_path),
        "confirmation_runner_git_head": HEAD,
    }


def test_s1_validation_is_no_update_exact_891_by_32_cuda_only() -> None:
    source = SCRIPT.read_text()
    tree = ast.parse(source)
    calls = {
        getattr(node.func, "attr", getattr(node.func, "id", ""))
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert "backward" not in calls and "step" not in calls
    for token in (
        'allowed_splits=("VALIDATION",)',
        "len(roots) == 891 * 32",
        '"duplicate_retry_or_rejection_count": 0',
        '"A100" in device_name',
        "torch.cuda.is_bf16_supported()",
        '"parameter_update_count": 0',
        '"development_test_outcome_reads": 0',
        '"new_final_evaluation_outcome_reads": 0',
    ):
        assert token in source


def test_s1_validation_waits_for_both_complete_training_packages(tmp_path: Path) -> None:
    config = {
        "output_root": str(tmp_path),
        "training": {"screen_seed": 20260911},
    }
    for run_id in ("v4_s1_full", "v4_s1_single_mode"):
        directory = tmp_path / run_id
        directory.mkdir()
        (directory / "training_summary.json").write_text(
            json.dumps(_training_summary(directory, run_id=run_id))
        )
    assert set(require_training_package_terminal_s1(config)) == {
        "v4_s1_full",
        "v4_s1_single_mode",
    }
    (tmp_path / "v4_s1_full" / "failure.json").write_text("{}")
    with pytest.raises(SetFlowCheckpointValidationS1Error):
        require_training_package_terminal_s1(config)


def test_s1_training_provenance_binds_both_runs_to_one_head(tmp_path: Path) -> None:
    config = {
        "output_root": str(tmp_path),
        "training": {"screen_seed": 20260911},
    }
    for run_id in ("v4_s1_full", "v4_s1_single_mode"):
        directory = tmp_path / run_id
        directory.mkdir()
        (directory / "training_summary.json").write_text(
            json.dumps(_training_summary(directory, run_id=run_id))
        )
        (directory / "training_config.json").write_text(
            json.dumps({"authorized_git_head": HEAD})
        )
        (directory / "training_attempt.json").write_text(
            json.dumps({"code_commit": HEAD})
        )
    assert require_training_package_provenance_s1(config) == {
        "v4_s1_full": HEAD,
        "v4_s1_single_mode": HEAD,
    }


def test_s1_validation_stage_seed_accepts_only_screen_or_frozen_confirmation() -> None:
    assert setflow_validation_stage_seed_s1(
        {"training": {"screen_seed": 20260911}}
    ) == ("SCREEN", 20260911)
    for seed in CONFIRMATION_SEEDS:
        assert setflow_validation_stage_seed_s1(
            _confirmation_config(Path("/unused"), seed)
        ) == ("CONFIRMATION", seed)
    with pytest.raises(SetFlowCheckpointValidationS1Error):
        setflow_validation_stage_seed_s1(
            _confirmation_config(Path("/unused"), 20260915)
        )
    changed = _confirmation_config(Path("/unused"))
    changed["run_stage"] = "POSTTRAINING"
    with pytest.raises(SetFlowCheckpointValidationS1Error):
        setflow_validation_stage_seed_s1(changed)


def test_s1_confirmation_validation_waits_for_one_full_package_not_single_mode(
    tmp_path: Path,
) -> None:
    config = _confirmation_config(tmp_path)
    directory = tmp_path / "v4_s1_full"
    directory.mkdir()
    (directory / "training_summary.json").write_text(
        json.dumps(
            _training_summary(
                directory,
                run_stage="CONFIRMATION",
                seed=20260912,
            )
        )
    )
    assert set(require_training_package_terminal_s1(config)) == {"v4_s1_full"}
    changed = _training_summary(
        directory,
        run_stage="CONFIRMATION",
        seed=20260912,
    )
    changed["selected_model"] = "v4_s1_single_mode"
    (directory / "training_summary.json").write_text(json.dumps(changed))
    with pytest.raises(
        SetFlowCheckpointValidationS1Error, match="confirmation summary identity"
    ):
        require_training_package_terminal_s1(config)


def test_s1_validator_dispatches_confirmation_authority_and_retains_cuda_identity() -> None:
    source = SCRIPT.read_text()
    assert "require_s1_confirmation_launch_authorization(" in source
    assert 'run_stage == "SCREEN" or run_id == CONFIRMATION_RUN_ID' in source
    assert '{"selected_model": CONFIRMATION_RUN_ID}' in source
    assert '"checkpoint_path": str(checkpoint_path)' in source
    assert '"training_summary_path": str(training_summary_path)' in source
    assert 'output_directory / "validation_summary.json"' in source
    assert '"precision": "BF16"' in source
    assert '"cpu_fallback_used": False' in source
    assert '"parameter_initialization_seed"' in source


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("parameter_initialization_seed", 20260913, "initialization"),
        (
            "parameter_initialization_seed_applied_before_model_construction",
            False,
            "initialization",
        ),
        ("completed_passes", 9, "parameter-update"),
        ("parameter_changed", False, "parameter-update"),
        ("cuda_available", False, "CUDA/A100/BF16"),
        ("bf16_supported", False, "CUDA/A100/BF16"),
        ("cpu_fallback_used", True, "CUDA/A100/BF16"),
        ("training_precision", "FP32", "CUDA/A100/BF16"),
        ("device_name", "CPU", "CUDA/A100/BF16"),
        ("development_test_outcome_reads", 1, "protected read"),
    ),
)
def test_s1_confirmation_terminal_training_barrier_rejects_evidence_drift(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    config = _confirmation_config(tmp_path)
    directory = tmp_path / "v4_s1_full"
    directory.mkdir()
    summary = _training_summary(
        directory,
        run_stage="CONFIRMATION",
        seed=20260912,
    )
    summary[field] = value
    (directory / "training_summary.json").write_text(json.dumps(summary))
    with pytest.raises(SetFlowCheckpointValidationS1Error, match=message):
        require_training_package_terminal_s1(config)


def test_s1_terminal_training_barrier_rejects_canonical_wrong_directory_path(
    tmp_path: Path,
) -> None:
    config = _confirmation_config(tmp_path)
    directory = tmp_path / "v4_s1_full"
    directory.mkdir()
    summary = _training_summary(
        directory,
        run_stage="CONFIRMATION",
        seed=20260912,
    )
    summary["saved_checkpoint_paths"]["4"] = str(
        tmp_path / "seed_20260913" / "v4_s1_full" / "pass_4.pt"
    )
    (directory / "training_summary.json").write_text(json.dumps(summary))
    with pytest.raises(SetFlowCheckpointValidationS1Error, match="checkpoint paths"):
        require_training_package_terminal_s1(config)


def test_s1_confirmation_training_provenance_rejects_runner_head_drift(
    tmp_path: Path,
) -> None:
    config = _confirmation_config(tmp_path)
    directory = tmp_path / "v4_s1_full"
    directory.mkdir()
    (directory / "training_summary.json").write_text(
        json.dumps(
            _training_summary(
                directory,
                run_stage="CONFIRMATION",
                seed=20260912,
            )
        )
    )
    (directory / "training_config.json").write_text(
        json.dumps(
            {
                "authorized_git_head": HEAD,
                "confirmation_runner_git_head": "b" * 40,
            }
        )
    )
    (directory / "training_attempt.json").write_text(
        json.dumps({"code_commit": HEAD})
    )
    with pytest.raises(SetFlowCheckpointValidationS1Error, match="Git HEAD drifted"):
        require_training_package_provenance_s1(config)


def test_s1_checkpoint_requires_exact_initialization_and_cuda_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _confirmation_config(tmp_path)
    directory = tmp_path / "v4_s1_full"
    directory.mkdir()
    summary = _training_summary(
        directory,
        run_stage="CONFIRMATION",
        seed=20260912,
    )
    (directory / "training_summary.json").write_text(json.dumps(summary))
    checkpoint = {
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_s1_checkpoint.v1",
        "run_stage": "CONFIRMATION",
        "run_id": "v4_s1_full",
        "selected_model": "v4_s1_full",
        "seed": 20260912,
        "completed_pass": 4,
        "objective_identity": OBJECTIVE_IDENTITY,
        "cross_state_candidate_mode_responsibility_weight": OBJECTIVE_WEIGHT,
        "active_responsibility_constraint_count": 1,
        "parameter_initialization_seed": 20260912,
        "parameter_initialization_seed_applied_before_model_construction": True,
        "physical_gpu_index": 0,
        "torch_device": "cuda:0",
        "device_name": "NVIDIA A100-SXM4-80GB",
        "training_precision": "BF16",
        "cuda_available": True,
        "bf16_supported": True,
        "cpu_fallback_used": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
        "vocabs": {},
        "capacity": {"trainable_parameter_count": 1},
        "model_state_dict": {},
    }

    class DummyModel:
        def to(self, _device):
            return self

        def load_state_dict(self, _state, *, strict):
            assert strict is True

        def eval(self):
            return self

    monkeypatch.setattr(validator.torch, "load", lambda *_args, **_kwargs: checkpoint)
    monkeypatch.setattr(
        validator,
        "build_setflow_screen_model_s1",
        lambda *_args, **_kwargs: (
            DummyModel(),
            {"trainable_parameter_count": 1},
        ),
    )
    model, loaded, loaded_summary = load_checkpoint_s1(
        config,
        run_id="v4_s1_full",
        checkpoint_pass=4,
        device=torch.device("cpu"),
    )
    assert isinstance(model, DummyModel)
    assert loaded is checkpoint
    assert loaded_summary == summary
    checkpoint["parameter_initialization_seed"] = 20260913
    with pytest.raises(SetFlowCheckpointValidationS1Error, match="initialization"):
        load_checkpoint_s1(
            config,
            run_id="v4_s1_full",
            checkpoint_pass=4,
            device=torch.device("cpu"),
        )
