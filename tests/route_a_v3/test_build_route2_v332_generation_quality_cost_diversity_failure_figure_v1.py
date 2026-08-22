import json
from pathlib import Path

import matplotlib.image as mpimg
import pytest

from scripts.route_a_v3.build_route2_v332_generation_quality_cost_diversity_failure_figure_v1 import (
    FIGURE_ID,
    build_figure,
)


ROOT = Path(__file__).resolve().parents[2]
GENERATION_TABLE = (
    ROOT / "docs/paper/route2_v332_generation_action_space_geometry_table_v1.csv"
)
GEOMETRY_AUDIT = (
    ROOT / "audits/route_a_v3_route2_generation_action_space_geometry_v1.json"
)


def test_builder_exports_truthful_quality_cost_diversity_failure_figure(
    tmp_path: Path,
) -> None:
    manifest = build_figure(
        generation_table=GENERATION_TABLE,
        geometry_audit=GEOMETRY_AUDIT,
        output_directory=tmp_path,
        formats=("png", "pdf", "svg"),
        dpi=300,
    )

    assert manifest["figure_id"] == FIGURE_ID
    assert manifest["status"] == "PROVISIONAL_GENERAL_MANUSCRIPT_FIGURE_RENDERED"
    assert manifest["target_journal"] == "PENDING_SELECTION"
    assert manifest["article_type"] == "PENDING_SELECTION"
    assert manifest["submission_phase"] == "PENDING_SELECTION"
    assert manifest["publisher_compliance_claimed"] is False
    assert manifest["scientific_claim_status"] == "NOT_ESTABLISHED"
    assert manifest["palette"]["color_is_redundant_with"] == [
        "marker shape",
        "hatching",
        "direct labels",
        "panel separation",
    ]
    assert manifest["protected_outcomes"] == {
        "development_test_read": False,
        "new_final_evaluation_read": False,
        "generated_candidate_payload_read": False,
        "guided_xeditflow_run": False,
    }
    assert manifest["failure_boundary"] == {
        "hard_legality_rate_all_methods": 1.0,
        "total_edit_budget_violations": 0,
        "total_candidate_budget_violations": 0,
        "total_no_legal_action_terminals": 0,
        "total_numerical_failure_terminals": 0,
    }
    assert any("wall time is absent" in item for item in manifest["missing_data"])
    assert any("Closed measured NDCG is undefined" in item for item in manifest["missing_data"])
    assert "not a bar/area encoding" in manifest["zero_and_axis_policy"][1]
    assert set(manifest["outputs"]) == {"png", "pdf", "svg"}

    for output in manifest["outputs"].values():
        path = Path(output["path"])
        assert path.exists()
        assert path.stat().st_size == output["bytes"] > 1000

    png = mpimg.imread(manifest["outputs"]["png"]["path"])
    assert png.shape[:2] == (2100, 2160)
    assert png.shape[2] == 4
    assert png[:, :, 3].min() == 1.0
    svg = Path(manifest["outputs"]["svg"]["path"]).read_text(encoding="utf-8")
    assert "Development generation quality, cost and failure geometry" in svg
    assert "&lt;image" not in svg and "<image" not in svg
    alt_text = Path(manifest["alt_text_path"]).read_text(encoding="utf-8")
    assert "not measured biological validation" in alt_text
    persisted = json.loads((tmp_path / f"{FIGURE_ID}_manifest.json").read_text())
    assert persisted == manifest


def test_builder_refuses_implicit_overwrite(tmp_path: Path) -> None:
    build_figure(
        generation_table=GENERATION_TABLE,
        geometry_audit=GEOMETRY_AUDIT,
        output_directory=tmp_path,
        formats=("png",),
        dpi=150,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_figure(
            generation_table=GENERATION_TABLE,
            geometry_audit=GEOMETRY_AUDIT,
            output_directory=tmp_path,
            formats=("png",),
            dpi=150,
        )
