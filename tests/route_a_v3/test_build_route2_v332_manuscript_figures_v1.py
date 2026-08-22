import json
from pathlib import Path

import matplotlib.image as mpimg
import pytest

from scripts.route_a_v3.build_route2_v332_manuscript_figures_v1 import (
    build_figures,
)


ROOT = Path(__file__).resolve().parents[2]
GENERATION_TABLE = (
    ROOT / "docs/paper/route2_v332_generation_action_space_geometry_table_v1.csv"
)
CRITIC_TABLE = (
    ROOT / "docs/paper/route2_v332_critic_v2_task_diagnostic_table_v1.csv"
)
HISTORICAL_SUMMARY = (
    ROOT / "audits/route_a_v3_route2_gse232572_zero_shot_summary_v1.json"
)


def test_builder_exports_two_provisional_figures_with_provenance(tmp_path: Path) -> None:
    manifest = build_figures(
        generation_table=GENERATION_TABLE,
        critic_table=CRITIC_TABLE,
        historical_summary=HISTORICAL_SUMMARY,
        output_directory=tmp_path,
        formats=("png", "pdf", "svg"),
        dpi=300,
    )

    assert manifest["status"] == "PROVISIONAL_GENERAL_MANUSCRIPT_FIGURES_RENDERED"
    assert manifest["target_journal"] == "PENDING_SELECTION"
    assert manifest["publisher_compliance_claimed"] is False
    assert manifest["protected_outcomes"] == {
        "development_test_read": False,
        "new_final_evaluation_read": False,
        "historical_outcome_exposed_gse232572_read": True,
        "guided_xeditflow_run": False,
    }
    assert manifest["scientific_claim_status"] == "NOT_ESTABLISHED"
    assert manifest["palette"]["amber"] == "#E6AB5F"
    assert "green" not in manifest["palette"]
    assert len(manifest["palette"]["screening_note"]) == 2
    assert len(manifest["figures"]) == 2

    by_id = {row["figure_id"]: row for row in manifest["figures"]}
    generation = by_id["route2_v332_figure1_generation_benchmark_v1"]
    limits = by_id["route2_v332_figure2_predictor_transfer_limits_v1"]
    assert generation["uncertainty"].startswith("No per-method uncertainty")
    assert any("NDCG is undefined" in item for item in generation["missing_data"])
    assert "paired 95% confidence intervals" in limits["uncertainty"]
    assert "two n=48 tasks remain included" in limits["missing_data"]

    for figure in (generation, limits):
        assert set(figure["outputs"]) == {"png", "pdf", "svg"}
        for output in figure["outputs"].values():
            path = Path(output["path"])
            assert path.exists()
            assert path.stat().st_size == output["bytes"] > 1000

    generation_png = mpimg.imread(generation["outputs"]["png"]["path"])
    limits_png = mpimg.imread(limits["outputs"]["png"]["path"])
    assert generation_png.shape[:2] == (1920, 2160)
    assert limits_png.shape[:2] == (2040, 2160)
    assert generation_png.shape[2] == limits_png.shape[2] == 4
    assert generation_png[:, :, 3].min() == 1.0
    assert limits_png[:, :, 3].min() == 1.0

    generation_svg = Path(generation["outputs"]["svg"]["path"]).read_text(
        encoding="utf-8"
    )
    limits_svg = Path(limits["outputs"]["svg"]["path"]).read_text(
        encoding="utf-8"
    )
    assert "Route 2 Development generation benchmark" in generation_svg
    assert "Predictor and historical-transfer limits" in limits_svg

    alt_text = Path(manifest["alt_text_path"]).read_text(encoding="utf-8")
    assert "not measured biological validation" in alt_text
    assert "not an independent final confirmation" in alt_text

    persisted = json.loads(
        (tmp_path / "route2_v332_figure_manifest_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted == manifest


def test_builder_refuses_implicit_overwrite(tmp_path: Path) -> None:
    build_figures(
        generation_table=GENERATION_TABLE,
        critic_table=CRITIC_TABLE,
        historical_summary=HISTORICAL_SUMMARY,
        output_directory=tmp_path,
        formats=("png",),
        dpi=150,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_figures(
            generation_table=GENERATION_TABLE,
            critic_table=CRITIC_TABLE,
            historical_summary=HISTORICAL_SUMMARY,
            output_directory=tmp_path,
            formats=("png",),
            dpi=150,
        )
