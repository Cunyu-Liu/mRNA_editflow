from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from scripts.route_a_v3.validate_route2_xeditsetflow_s1_checkpoint import (
    OBJECTIVE_IDENTITY,
    OBJECTIVE_WEIGHT,
    SetFlowCheckpointValidationS1Error,
    require_training_package_provenance_s1,
    require_training_package_terminal_s1,
    setflow_validation_stage_seed_s1,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/validate_route2_xeditsetflow_s1_checkpoint.py"


def _training_summary() -> dict:
    return {
        "status": "TERMINAL_XEDITSETFLOW_V4_S1_TRAINING_COMPLETE_PENDING_VALIDATION",
        "run_stage": "SCREEN",
        "seed": 20260911,
        "objective_identity": OBJECTIVE_IDENTITY,
        "cross_state_candidate_mode_responsibility_weight": OBJECTIVE_WEIGHT,
        "active_responsibility_constraint_count": 1,
        "saved_checkpoint_paths": {"4": "a", "6": "b", "8": "c", "10": "d"},
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
            json.dumps(_training_summary())
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
        (directory / "training_summary.json").write_text(json.dumps(_training_summary()))
        (directory / "training_config.json").write_text(
            json.dumps({"authorized_git_head": "a" * 40})
        )
        (directory / "training_attempt.json").write_text(
            json.dumps({"code_commit": "a" * 40})
        )
    assert require_training_package_provenance_s1(config) == {
        "v4_s1_full": "a" * 40,
        "v4_s1_single_mode": "a" * 40,
    }


def test_s1_validation_rejects_extra_seed_or_non_screen_stage() -> None:
    assert setflow_validation_stage_seed_s1(
        {"training": {"screen_seed": 20260911}}
    ) == ("SCREEN", 20260911)
    with pytest.raises(SetFlowCheckpointValidationS1Error):
        setflow_validation_stage_seed_s1(
            {"run_stage": "CONFIRMATION", "training": {"screen_seed": 20260911}}
        )
