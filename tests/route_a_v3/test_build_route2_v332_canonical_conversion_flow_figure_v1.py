import json
from pathlib import Path

import matplotlib.image as mpimg
import pytest

from scripts.route_a_v3.build_route2_v332_canonical_conversion_flow_figure_v1 import (
    STEM,
    build_figure,
)


ROOT = Path(__file__).resolve().parents[2]
DATASET_TABLE = ROOT / "docs/paper/route2_v332_dataset_qualification_table_v1.csv"
SPLIT_PROTOCOL = ROOT / "configs/route_a_v3_route2_method_repair_protocol_v2.json"


def test_builder_exports_truthful_provisional_conversion_flow(tmp_path: Path) -> None:
    manifest = build_figure(
        dataset_table=DATASET_TABLE,
        split_protocol=SPLIT_PROTOCOL,
        output_directory=tmp_path,
    )

    assert manifest["status"] == "PROVISIONAL_CANONICAL_CONVERSION_FLOW_FIGURE_RENDERED"
    assert manifest["publisher_compliance_claimed"] is False
    assert manifest["target_journal"] == "PENDING_SELECTION"
    assert manifest["evidence"]["registered_study_count"] == 14
    assert manifest["evidence"]["development_record_count"] == 126165
    assert manifest["evidence"]["historical_record_count"] == 8068
    assert manifest["evidence"]["new_final_evaluation_record_count"] == 0
    assert manifest["evidence"]["split"] == {
        "TRAIN": 89580,
        "VALIDATION": 18293,
        "TEST": 18292,
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
    assert png.shape == (1860, 2160, 4)
    assert png[:, :, 3].min() == 1.0
    svg = Path(manifest["outputs"]["svg"]["path"]).read_text(encoding="utf-8")
    assert "Route 2 canonical study-to-evidence flow" in svg
    assert "Arrows encode workflow only" in svg
    alt_text = Path(manifest["alt_text_path"]).read_text(encoding="utf-8")
    assert "Arrow widths are not quantitative" in alt_text
    assert "no new unexposed final Evaluation records" in alt_text
    persisted = json.loads((tmp_path / f"{STEM}_manifest.json").read_text(encoding="utf-8"))
    assert persisted == manifest


def test_builder_refuses_implicit_overwrite(tmp_path: Path) -> None:
    build_figure(
        dataset_table=DATASET_TABLE,
        split_protocol=SPLIT_PROTOCOL,
        output_directory=tmp_path,
        formats=("png",),
        dpi=150,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_figure(
            dataset_table=DATASET_TABLE,
            split_protocol=SPLIT_PROTOCOL,
            output_directory=tmp_path,
            formats=("png",),
            dpi=150,
        )
