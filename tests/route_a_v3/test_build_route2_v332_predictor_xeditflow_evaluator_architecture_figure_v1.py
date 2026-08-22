import json
from pathlib import Path

import matplotlib.image as mpimg
import pytest

from scripts.route_a_v3.build_route2_v332_predictor_xeditflow_evaluator_architecture_figure_v1 import (
    STEM,
    build_figure,
)


ROOT = Path(__file__).resolve().parents[2]


def test_builder_exports_truthful_separated_architecture(tmp_path: Path) -> None:
    manifest = build_figure(output_directory=tmp_path)

    assert manifest["status"] == "PROVISIONAL_PREDICTOR_XEDITFLOW_EVALUATOR_ARCHITECTURE_FIGURE_RENDERED"
    assert manifest["publisher_compliance_claimed"] is False
    assert manifest["target_journal"] == "PENDING_SELECTION"
    assert manifest["evidence"]["delta_critic"] == {
        "model_kind": "delta_pretrained_mrnabert_edit_centered_antisymmetric",
        "frozen_encoder": "mRNABERT",
        "frozen_encoder_parameter_count": 113389056,
        "trainable_head_parameter_count": 9342914,
        "total_effective_parameter_count": 122731970,
        "hidden_dim": 384,
        "depth": 10,
        "position_features": "NORMALIZED_ABSOLUTE_PLUS_EDIT_GATED",
        "reward_signal": "STANDARDIZED_PREDICTED_MEAN_DELTA",
        "current_guidance_status": "CRITIC_V2_CONTROLS_DO_NOT_SUPPORT_THREE_FROZEN_SEEDS",
        "critic_ready_for_guidance": False,
    }
    assert manifest["evidence"]["legal_xeditflow"]["engineering_status"] == "FLOW_G0_READY"
    assert manifest["evidence"]["legal_xeditflow"]["guided_xeditflow_run"] is False
    assert manifest["evidence"]["legal_xeditflow"]["action_types_in_scope"] == ["SUB", "STOP"]
    assert manifest["evidence"]["legal_xeditflow"]["action_types_out_of_scope"] == ["INS", "DEL"]
    assert manifest["evidence"]["legal_xeditflow"]["allowed_edit_budgets"] == [1, 3, 5]
    assert manifest["evidence"]["legal_xeditflow"]["generated_candidates_add_canonical_credit"] is False
    assert manifest["evidence"]["independent_evaluator"] == {
        "model_kind": "siamese_cnn",
        "hidden_dim": 103,
        "depth": 7,
        "terminal_actual_trainable_parameter_count": 509845,
        "architecture_distinct_from_guide": True,
        "qualification_pool": "DEVELOPMENT_VALIDATION",
        "terminal_status": "INDEPENDENT_GENERATION_EVALUATOR_QUALIFIED",
        "role": "DEVELOPMENT_GENERATION_METHOD_SELECTION_NOT_BIOLOGICAL_VALIDATION",
    }
    assert manifest["evidence"]["frozen_feedback_boundaries"] == {
        "critic_parameter_update_during_generation": False,
        "generator_gradient_into_critic": False,
        "evaluation_model_gradient_into_generator": False,
        "evaluation_records_used_for_reward": 0,
    }
    assert manifest["protected_outcomes"] == {
        "development_test_read": False,
        "new_final_evaluation_read": False,
        "emtab10902_outcome_read": False,
        "sealed_gse246381_read": False,
        "guided_xeditflow_run": False,
    }
    assert set(manifest["outputs"]) == {"png", "pdf", "svg"}
    for output in manifest["outputs"].values():
        path = Path(output["path"])
        assert path.exists()
        assert path.stat().st_size == output["bytes"] > 1000

    png = mpimg.imread(manifest["outputs"]["png"]["path"])
    assert png.shape == (2040, 2160, 4)
    assert png[:, :, 3].min() == 1.0
    svg = Path(manifest["outputs"]["svg"]["path"]).read_text(encoding="utf-8")
    assert "Route 2 predictor–generator–evaluator separation" in svg
    assert "CURRENT: NOT READY FOR GUIDANCE" in svg
    assert "NO evaluator → generator gradient" in svg
    alt_text = Path(manifest["alt_text_path"]).read_text(encoding="utf-8")
    assert "no evaluator-to-generator gradient" in alt_text
    assert "generated candidates add no canonical" in alt_text
    persisted = json.loads((tmp_path / f"{STEM}_manifest.json").read_text(encoding="utf-8"))
    assert persisted == manifest


def test_builder_refuses_implicit_overwrite(tmp_path: Path) -> None:
    build_figure(output_directory=tmp_path, formats=("png",), dpi=150)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_figure(output_directory=tmp_path, formats=("png",), dpi=150)
