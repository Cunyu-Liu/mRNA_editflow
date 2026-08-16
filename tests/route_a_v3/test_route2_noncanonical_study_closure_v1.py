from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CLOSURE_PATH = ROOT / "configs/route_a_v3_route2_noncanonical_study_closure_v1.json"
ROUTE_PATH = ROOT / "configs/route_a_v3_route2_v1.json"


def test_noncanonical_closures_complete_the_fourteen_study_inventory_without_credit() -> None:
    closure = json.loads(CLOSURE_PATH.read_text(encoding="utf-8"))
    route = json.loads(ROUTE_PATH.read_text(encoding="utf-8"))
    rows = closure["study_closures"]
    assert closure["scope"]["study_unit_count"] == len(rows) == 4
    assert closure["scope"]["route2_manifest_row_count"] == 0
    assert not any(closure["scope"]["qualified_credit_delta"].values())
    assert {row["study_unit_id"] for row in rows} == {
        "GSE145046", "GSE207584", "GSE261709", "GSE246381"
    }
    assert len(route["study_inventory"]) == 14
    route_roles = {
        row["study_unit_id"]: (row["pool"], row["role"])
        for row in route["study_inventory"]
    }
    assert all(route_roles[row["study_unit_id"]] == (row["pool"], row["role"]) for row in rows)


def test_zero_controls_do_not_turn_sealed_unknown_into_a_zero_record_claim() -> None:
    closure = json.loads(CLOSURE_PATH.read_text(encoding="utf-8"))
    by_study = {row["study_unit_id"]: row for row in closure["study_closures"]}
    assert by_study["GSE145046"]["canonical_record_count"] == 0
    assert by_study["GSE207584"]["canonical_record_count"] == 0
    assert by_study["GSE261709"]["canonical_record_count"] == 0
    sealed = by_study["GSE246381"]
    assert sealed["canonical_record_count"] is None
    assert sealed["route2_manifest_rows"] == 0
    assert sealed["historical_exposure_status"] == "NOT_ASSERTED_BY_THIS_ROUTE2_CLOSURE"
