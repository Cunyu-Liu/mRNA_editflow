from __future__ import annotations

import json
from pathlib import Path

import matplotlib.image as mpimg
import pytest

from scripts.route_a_v3.build_route2_v332_development_learning_curves_figure_v1 import (
    CRITIC_ARMS,
    PREDICTOR_PROFILES,
    LearningCurveInputError,
    build_figure,
)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _predictor_summary(epoch_count: int = 8) -> dict:
    return {
        "development_test_outcomes_evaluated": False,
        "evaluation_outcomes_read": 0,
        "history": [
            {
                "epoch": epoch,
                "train_loss": 1.0 / epoch,
                "validation": {"spearman": 0.02 * epoch},
            }
            for epoch in range(1, epoch_count + 1)
        ],
    }


def _fixture_paths(tmp_path: Path, *, bad_predictor_epochs: int | None = None) -> dict:
    predictor_paths = {}
    selections = {}
    for index, profile in enumerate(PREDICTOR_PROFILES):
        trial_id = f"trial_{index}"
        summary_path = tmp_path / f"predictor_{index}.json"
        epoch_count = bad_predictor_epochs if index == 0 and bad_predictor_epochs else 8
        _write(summary_path, _predictor_summary(epoch_count))
        predictor_paths[profile] = summary_path
        selections[profile] = {
            "selected_trial_id": trial_id,
            "selection_primary_metric": "DEVELOPMENT_VALIDATION_TASK_MACRO_SPEARMAN",
            "all_trials_ranked": [
                {
                    "trial_id": trial_id,
                    "training_summary_path": str(summary_path),
                    "model_kind": "synthetic_model",
                    "parameter_count": 80000 + index,
                    "task_macro_spearman": 0.08 + 0.01 * index,
                }
            ],
        }
    hpo_path = _write(
        tmp_path / "hpo.json",
        {
            "selection_pool": "DEVELOPMENT_VALIDATION",
            "development_test_outcomes_accessed": False,
            "evaluation_outcomes_accessed": False,
            "selections": selections,
        },
    )

    selected_epochs = {
        "full": 98,
        "candidate_permutation": 12,
        "source_only": 1,
        "source_edit_metadata": 2,
    }
    critic_paths = {}
    arms = {}
    for index, arm in enumerate(CRITIC_ARMS):
        values = [0.001 * epoch + 0.005 * index for epoch in range(1, 101)]
        selected = selected_epochs[arm]
        critic_paths[arm] = _write(
            tmp_path / f"critic_{arm}.json",
            {
                "development_test_outcomes_evaluated": False,
                "evaluation_outcomes_read": 0,
                "selected_epoch": selected,
                "history": [
                    {
                        "epoch": epoch,
                        "train_loss": 2.0 / epoch,
                        "validation": {"task_macro_spearman": values[epoch - 1]},
                    }
                    for epoch in range(1, 101)
                ],
            },
        )
        arms[arm] = {
            "selected_epoch": selected,
            "task_macro_spearman": values[selected - 1],
        }
    critic_path = _write(
        tmp_path / "critic_audit.json",
        {
            "control_screen": {
                "status": "CRITIC_V2_CONTROLS_DO_NOT_SUPPORT_THREE_FROZEN_SEEDS",
                "supports_three_frozen_seeds": False,
                "arms": arms,
                "strongest_same_information_baseline": {"task_macro_spearman": 0.13171439492559175},
            },
            "protected_outcomes": {
                "development_test_outcomes_accessed": False,
                "evaluation_outcomes_accessed": False,
            },
        },
    )

    evaluator_values = [0.095 + 0.001 * epoch for epoch in range(1, 9)]
    evaluator_path = _write(
        tmp_path / "evaluator.json",
        {
            "development_test_outcomes_evaluated": False,
            "evaluation_outcomes_read": 0,
            "checkpoint_selection": "FINAL_EPOCH",
            "selected_epoch": 8,
            "history": [
                {
                    "epoch": epoch,
                    "train_loss": 2.0 / epoch,
                    "validation": {"task_macro_spearman": evaluator_values[epoch - 1]},
                }
                for epoch in range(1, 9)
            ],
        },
    )
    freshness_path = _write(
        tmp_path / "freshness.json",
        {
            "independent_evaluator": {
                "adjudication_status": "INDEPENDENT_GENERATION_EVALUATOR_QUALIFIED",
                "development_test_outcomes_accessed": False,
                "evaluation_outcomes_accessed": False,
                "task_macro_spearman": evaluator_values[-1],
                "exclusive_threshold": 0.101,
                "margin": evaluator_values[-1] - 0.101,
            }
        },
    )
    flow_path = _write(
        tmp_path / "flow.json",
        {
            "status": "LEARNED_BASE_FLOW_GPU_UPDATE_COMPLETE",
            "selected_epoch": 1,
            "development_test_outcomes_evaluated": False,
            "evaluation_records_read": 0,
            "guided_critic_used": False,
            "biological_optimization_established": False,
            "history": [
                {"epoch": epoch, "train_nll": 6.0 / epoch, "validation_nll": 5.0 + 0.1 * epoch}
                for epoch in range(1, 31)
            ],
        },
    )
    return {
        "hpo_audit_path": hpo_path,
        "critic_audit_path": critic_path,
        "freshness_audit_path": freshness_path,
        "predictor_summary_paths": predictor_paths,
        "critic_summary_paths": critic_paths,
        "evaluator_summary_path": evaluator_path,
        "flow_summary_path": flow_path,
    }


def test_builder_exports_metric_separated_unsmoothed_learning_curves(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    output = tmp_path / "figures"
    manifest = build_figure(
        **paths,
        output_directory=output,
        formats=("png", "pdf", "svg"),
        dpi=300,
    )

    assert manifest["status"] == "PROVISIONAL_DEVELOPMENT_LEARNING_CURVES_FIGURE_RENDERED"
    assert manifest["publisher_compliance_claimed"] is False
    assert manifest["raw_unsmoothed_histories"] is True
    assert manifest["cross_panel_metric_comparison_allowed"] is False
    assert manifest["predictor_curve_metric"] == "POOLED_DEVELOPMENT_VALIDATION_SPEARMAN"
    assert manifest["predictor_selection_metric"] == "DEVELOPMENT_VALIDATION_TASK_MACRO_SPEARMAN"
    assert manifest["protected_outcomes"] == {
        "development_test_read": False,
        "new_final_evaluation_read": False,
        "guided_xeditflow_run": False,
    }
    assert manifest["panel_evidence"]["critic_status"] == (
        "CRITIC_V2_CONTROLS_DO_NOT_SUPPORT_THREE_FROZEN_SEEDS"
    )
    assert len(manifest["panel_evidence"]["predictors"]) == 6
    assert len(manifest["panel_evidence"]["critic"]) == 4
    assert manifest["panel_evidence"]["flow"]["biological_optimization_established"] is False

    assert set(manifest["outputs"]) == {"png", "pdf", "svg"}
    for output_entry in manifest["outputs"].values():
        path = Path(output_entry["path"])
        assert path.exists()
        assert path.stat().st_size == output_entry["bytes"] > 1000
    png = mpimg.imread(manifest["outputs"]["png"]["path"])
    assert png.shape[:2] == (2520, 2400)
    assert png.shape[2] == 4
    assert png[:, :, 3].min() == 1.0
    svg = Path(manifest["outputs"]["svg"]["path"]).read_text(encoding="utf-8")
    assert "Raw terminal histories" in svg
    assert "image href" not in svg
    alt = Path(manifest["alt_text_path"]).read_text(encoding="utf-8")
    assert "raw, unsmoothed" in alt
    assert "validation NLL worsens" in alt
    assert "Development TEST and new final Evaluation outcomes" in alt

    persisted = json.loads((output / "route2_v332_development_learning_curves_figure_v1_manifest.json").read_text())
    assert persisted == manifest

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_figure(**paths, output_directory=output, formats=("png",), dpi=300)


def test_builder_rejects_incomplete_terminal_predictor_history(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path, bad_predictor_epochs=7)
    with pytest.raises(LearningCurveInputError, match="must contain 8 epochs"):
        build_figure(
            **paths,
            output_directory=tmp_path / "figures",
            formats=("png",),
            dpi=150,
        )
