from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.route_a_v3 import validate_route2_xeditsetflow_v4_checkpoint as validation
from scripts.route_a_v3.validate_route2_xeditsetflow_v4_checkpoint import (
    SetFlowCheckpointValidationV4Error,
    _write_atomic_terminal_v4,
    require_training_package_provenance_v4,
    require_training_package_terminal_v4,
    setflow_validation_stage_seed_v4,
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


def test_validation_terminal_artifacts_are_atomically_published(tmp_path: Path) -> None:
    output = tmp_path / "pass_4" / "validation_summary.json"
    payload = {"status": "TERMINAL_XEDITSETFLOW_V4_CHECKPOINT_VALIDATION_COMPLETE"}
    _write_atomic_terminal_v4(output, payload)
    partial = output.with_suffix(output.suffix + ".partial")
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert not partial.exists()

    stale = tmp_path / "pass_6.failed.json"
    stale_partial = stale.with_suffix(stale.suffix + ".partial")
    stale_partial.write_text("interrupted", encoding="utf-8")
    with pytest.raises(SetFlowCheckpointValidationV4Error, match="partial SetFlow V4"):
        _write_atomic_terminal_v4(stale, payload)
    assert not stale.exists()
    assert stale_partial.read_text(encoding="utf-8") == "interrupted"


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


def test_small_graph_mixture_distribution_matches_independent_path_enumeration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fixed_priors(*_args, **_kwargs):
        return [[0.25, 0.75]], SimpleNamespace(
            prior_forward_count=1,
            trunk_forward_count=1,
            mode_head_forward_count=2,
        )

    def fixed_rates(_model, _state, _metadata, selected_mode, actions, **_kwargs):
        ordered = list(actions)
        weights = range(1, len(ordered) + 1)
        if selected_mode == 1:
            weights = reversed(tuple(weights))
        return dict(zip(ordered, weights, strict=True))

    monkeypatch.setattr(validation, "root_mode_priors_v4", fixed_priors)
    monkeypatch.setattr(validation, "setflow_rate_map_v4", fixed_rates)
    model = SimpleNamespace(mode_count=2)

    result = validation.small_graph_exact_check_v4(
        model, {}, torch.device("cpu")
    )

    assert result["status"] == "PASS"
    assert result["mode_count"] == 2
    assert result["dynamic_probability_sum"] == pytest.approx(1.0)
    assert result["enumeration_probability_sum"] == pytest.approx(1.0)
    assert result["total_variation"] <= result["tolerance"] == 1e-12


def test_both_training_runs_must_be_terminal_before_any_checkpoint_validation(
    tmp_path: Path,
) -> None:
    config = {
        "output_root": str(tmp_path),
        "training": {"screen_seed": 20260911},
    }
    for run_id in ("v4_full", "v4_single_mode"):
        directory = tmp_path / run_id
        directory.mkdir()
        (directory / "training_summary.json").write_text(
            json.dumps(
                    {
                        "status": "TERMINAL_XEDITSETFLOW_V4_TRAINING_COMPLETE_PENDING_VALIDATION",
                        "run_stage": "SCREEN",
                        "seed": 20260911,
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
    assert set(require_training_package_terminal_v4(config)) == {
        "v4_full",
        "v4_single_mode",
    }
    (tmp_path / "v4_single_mode" / "failure.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SetFlowCheckpointValidationV4Error):
        require_training_package_terminal_v4(config)


def test_confirmation_waits_only_for_full_training_at_declared_seed(
    tmp_path: Path,
) -> None:
    config = {
        "run_stage": "CONFIRMATION",
        "training_seed": 20260913,
        "output_root": str(tmp_path),
    }
    directory = tmp_path / "v4_full"
    directory.mkdir()
    (directory / "training_summary.json").write_text(
        json.dumps(
            {
                "status": "TERMINAL_XEDITSETFLOW_V4_TRAINING_COMPLETE_PENDING_VALIDATION",
                "run_stage": "CONFIRMATION",
                "seed": 20260913,
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
    assert setflow_validation_stage_seed_v4(config) == ("CONFIRMATION", 20260913)
    assert set(require_training_package_terminal_v4(config)) == {"v4_full"}


def test_checkpoint_validation_binds_authorization_to_training_head(
    tmp_path: Path,
) -> None:
    config = {
        "output_root": str(tmp_path),
        "training": {"screen_seed": 20260911},
    }
    training_head = "a" * 40
    for run_id in ("v4_full", "v4_single_mode"):
        directory = tmp_path / run_id
        directory.mkdir()
        (directory / "training_summary.json").write_text(
            json.dumps(
                {
                    "status": "TERMINAL_XEDITSETFLOW_V4_TRAINING_COMPLETE_PENDING_VALIDATION",
                    "run_stage": "SCREEN",
                    "seed": 20260911,
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
        (directory / "training_config.json").write_text(
            json.dumps({"authorized_git_head": training_head}),
            encoding="utf-8",
        )
        (directory / "training_attempt.json").write_text(
            json.dumps({"code_commit": training_head}),
            encoding="utf-8",
        )

    assert require_training_package_provenance_v4(config) == {
        "v4_full": training_head,
        "v4_single_mode": training_head,
    }

    (tmp_path / "v4_single_mode" / "training_attempt.json").write_text(
        json.dumps({"code_commit": "b" * 40}),
        encoding="utf-8",
    )
    with pytest.raises(SetFlowCheckpointValidationV4Error, match="disagree on Git HEAD"):
        require_training_package_provenance_v4(config)
