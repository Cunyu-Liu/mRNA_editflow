from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_external_model_provenance_is_pinned_and_truthful() -> None:
    provenance = json.loads(
        (ROOT / "configs/route_a_v3_route2_external_model_provenance_v1.json").read_text(encoding="utf-8")
    )
    artifacts = {row["provenance_id"]: row for row in provenance["artifacts"]}
    assert len(artifacts) == 5
    assert all(len(row["revision"]) == 40 for row in artifacts.values())
    converted = artifacts["multimolecule_rnafm_7d6e73ad"]
    assert converted["artifact_role"] == "UNOFFICIAL_MULTIMOLECULE_CONVERSION_CHECKPOINT"
    assert converted["official_original_checkpoint_used"] is False
    assert converted["artifact_license"] == "AGPL-3.0"
    assert converted["official_original_repository_license"] == "MIT"
    assert provenance["evaluation_outcomes_accessed"] is False


def test_baseline_inventory_references_known_provenance_ids() -> None:
    provenance = json.loads(
        (ROOT / "configs/route_a_v3_route2_external_model_provenance_v1.json").read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (ROOT / "configs/route_a_v3_route2_baseline_inventory_v1.json").read_text(encoding="utf-8")
    )
    known = {row["provenance_id"] for row in provenance["artifacts"]}
    referenced = {row["artifact_provenance_id"] for row in inventory["prediction_common_task_adapters"]}
    assert referenced == known
    converted = next(row for row in inventory["prediction_common_task_adapters"] if row["model_id"].startswith("RNA-FM"))
    assert converted["native_track_status"].startswith("UNOFFICIAL_MULTIMOLECULE_CONVERSION")
