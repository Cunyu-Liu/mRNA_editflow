import json
from pathlib import Path

import matplotlib.image as mpimg
import pytest

from scripts.route_a_v3.build_route2_v332_development_evaluation_architecture_figure_v1 import (
    STEM,
    build_figure,
)


ROOT = Path(__file__).resolve().parents[2]
DATASET_TABLE = ROOT / "docs/paper/route2_v332_dataset_qualification_table_v1.csv"
METHOD_PROTOCOL = ROOT / "configs/route_a_v3_route2_method_repair_protocol_v2.json"
READINESS_PROTOCOL = (
    ROOT / "configs/route_a_v3_route2_mrnabert_critic_v2_guidance_readiness_protocol_v1.json"
)
PACKAGE_AUDIT = ROOT / "audits/route_a_v3_route2_v332_minimum_benchmark_package_v1.json"


def test_builder_exports_truthful_provisional_architecture(tmp_path: Path) -> None:
    manifest = build_figure(
        dataset_table=DATASET_TABLE,
        method_protocol=METHOD_PROTOCOL,
        readiness_protocol=READINESS_PROTOCOL,
        package_audit=PACKAGE_AUDIT,
        output_directory=tmp_path,
    )

    assert manifest["status"] == "PROVISIONAL_DEVELOPMENT_EVALUATION_ARCHITECTURE_FIGURE_RENDERED"
    assert manifest["publisher_compliance_claimed"] is False
    assert manifest["target_journal"] == "PENDING_SELECTION"
    assert manifest["evidence"]["development_record_count"] == 126165
    assert manifest["evidence"]["development_split"] == {
        "TRAIN": 89580,
        "VALIDATION": 18293,
        "TEST": 18292,
    }
    assert manifest["evidence"]["critic_ready_for_guidance"] is False
    assert manifest["evidence"]["historical_gse232572"]["record_count"] == 8068
    assert manifest["evidence"]["historical_gse232572"]["final_confirmation_eligible"] is False
    assert manifest["evidence"]["emtab10902"]["outcome_read"] is False
    assert manifest["evidence"]["gse246381"]["outcome_read"] is False
    assert manifest["evidence"]["replacement_evaluation"] == {
        "registered": False,
        "outcome_unexposed_canonical_records": 0,
        "opened": False,
        "execution_order": [
            "FREEZE_PREDICTOR_GENERATOR_BASELINES_METRICS_AND_ADAPTATION_POLICY",
            "RUN_AND_PERMANENTLY_RECORD_ONE_NEW_STUDY_ZERO_SHOT",
            "ONLY_THEN_ALLOW_CALIBRATION_OR_FEW_SHOT_ADAPTATION",
            "ZERO_SHOT_REMAINS_HEADLINE",
        ],
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
    assert "Route 2 Development and external Evaluation firewall" in svg
    assert "CURRENT: NO-GO" in svg
    assert "Future chain is not executed" in svg
    alt_text = Path(manifest["alt_text_path"]).read_text(encoding="utf-8")
    assert "zero outcome-unexposed final Evaluation records" in alt_text
    assert "Arrow widths are not" in alt_text
    persisted = json.loads((tmp_path / f"{STEM}_manifest.json").read_text(encoding="utf-8"))
    assert persisted == manifest


def test_builder_refuses_implicit_overwrite(tmp_path: Path) -> None:
    build_figure(
        dataset_table=DATASET_TABLE,
        method_protocol=METHOD_PROTOCOL,
        readiness_protocol=READINESS_PROTOCOL,
        package_audit=PACKAGE_AUDIT,
        output_directory=tmp_path,
        formats=("png",),
        dpi=150,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_figure(
            dataset_table=DATASET_TABLE,
            method_protocol=METHOD_PROTOCOL,
            readiness_protocol=READINESS_PROTOCOL,
            package_audit=PACKAGE_AUDIT,
            output_directory=tmp_path,
            formats=("png",),
            dpi=150,
        )
