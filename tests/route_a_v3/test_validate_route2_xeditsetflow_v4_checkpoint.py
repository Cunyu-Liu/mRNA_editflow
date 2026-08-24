from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from scripts.route_a_v3.validate_route2_xeditsetflow_v4_checkpoint import (
    SetFlowCheckpointValidationV4Error,
    require_screen_training_package_terminal_v4,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/validate_route2_xeditsetflow_v4_checkpoint.py"


def test_validation_runner_has_no_optimizer_or_parameter_update_call() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    calls = {
        getattr(node.func, "attr", getattr(node.func, "id", ""))
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert "backward" not in calls
    assert "step" not in calls


def test_validation_runner_loads_only_validation_projection_not_test() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'allowed_splits=("VALIDATION",)' in source
    assert "development_test_outcome_reads\": 0" in source
    assert "new_final_evaluation_outcome_reads\": 0" in source


def test_generation_budget_is_exact_without_retry_or_duplicate_rejection() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "len(roots) == 891 * 32" in source
    assert '"duplicate_retry_or_rejection_count": 0' in source
    assert "sample_many_setflow_v4" in source


def test_validation_requires_terminal_training_and_all_four_checkpoints() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "TERMINAL_XEDITSETFLOW_V4_TRAINING_COMPLETE_PENDING_VALIDATION" in source
    assert '("v4_full", "v4_single_mode")' in source
    assert '{"4", "6", "8", "10"}' in source
    assert "PENDING_TERMINAL_OUTCOME_FREE_VALIDATION_GENERATION" in source


def test_compute_record_counts_prior_trunk_mode_heads_and_replay() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for token in (
        "common_nll_mode_head_forward_state_count",
        '"root_prior"',
        '"primary_generation"',
        '"replay_generation"',
        '"critic_forward_count": 0',
        '"independent_evaluator_forward_count": 0',
    ):
        assert token in source


def test_both_training_runs_must_be_terminal_before_any_checkpoint_validation(
    tmp_path: Path,
) -> None:
    config = {"output_root": str(tmp_path)}
    for run_id in ("v4_full", "v4_single_mode"):
        directory = tmp_path / run_id
        directory.mkdir()
        (directory / "training_summary.json").write_text(
            json.dumps(
                {
                    "status": "TERMINAL_XEDITSETFLOW_V4_TRAINING_COMPLETE_PENDING_VALIDATION",
                    "saved_checkpoint_paths": {
                        "4": "a",
                        "6": "b",
                        "8": "c",
                        "10": "d",
                    },
                }
            ),
            encoding="utf-8",
        )
    assert set(require_screen_training_package_terminal_v4(config)) == {
        "v4_full",
        "v4_single_mode",
    }
    (tmp_path / "v4_single_mode" / "failure.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SetFlowCheckpointValidationV4Error):
        require_screen_training_package_terminal_v4(config)
