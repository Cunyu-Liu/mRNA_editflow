import csv
import json
from pathlib import Path

import pytest

from scripts.route_a_v3.build_route2_v332_dataset_qualification_table_v1 import (
    HISTORICAL_ROLE,
    build_table,
)


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "configs/route_a_v3_route2_14_study_final_inventory_v1.json"


def test_builder_preserves_qualification_and_role_boundaries(tmp_path: Path) -> None:
    table_path = tmp_path / "dataset_table.csv"
    audit_path = tmp_path / "dataset_table_audit.json"
    audit = build_table(
        inventory_path=INVENTORY,
        table_path=table_path,
        audit_path=audit_path,
    )

    with table_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["study_unit_id"]: row for row in rows}

    assert len(rows) == len(by_id) == audit["study_count"] == 14
    assert sum(int(row["development_canonical_records"]) for row in rows) == 126165
    assert sum(int(row["historical_transfer_canonical_records"]) for row in rows) == 8068
    assert sum(int(row["final_evaluation_unexposed_canonical_records"]) for row in rows) == 0
    assert sum(int(row["qualified_canonical_credit_records"]) for row in rows) == 6547
    assert sum(int(row["ordinary_study_credit"]) for row in rows) == 1
    assert sum(int(row["a1_study_credit"]) for row in rows) == 1
    assert sum(int(row["true_a2_study_credit"]) for row in rows) == 0

    qualified = by_id["GSE200304"]
    assert qualified["qualification_stratum"] == "QUALIFIED"
    assert qualified["qualified_canonical_credit_records"] == "6547"
    assert qualified["ordinary_study_credit"] == qualified["a1_study_credit"] == "1"

    relaxed = by_id["GSE114002"]
    assert relaxed["qualification_stratum"] == "DEVELOPMENT_RELAXED_NOT_QUALIFIED"
    assert relaxed["qualified_canonical_credit_records"] == "0"

    listwise = by_id["GSE269595"]
    assert listwise["qualification_stratum"] == "DEVELOPMENT_LISTWISE_NOT_QUALIFIED"
    assert listwise["qualified_canonical_credit_records"] == "0"

    historical = by_id["GSE232572"]
    assert historical["inventory_use_role"] == "EVALUATION"
    assert historical["current_analysis_role_v332"] == HISTORICAL_ROLE
    assert historical["historical_transfer_canonical_records"] == "8068"
    assert historical["final_evaluation_unexposed_canonical_records"] == "0"

    emtab = by_id["E-MTAB-10902"]
    assert emtab["current_analysis_role_v332"] == "CONVERSION_FAILURE_ONLY_OUTCOME_NOT_READ"
    assert emtab["outcome_exposure"] == "OUTCOME_NOT_READ"

    sealed = by_id["GSE246381"]
    assert sealed["qualification_stratum"] == "SEALED_EXCLUDED"
    assert sealed["outcome_exposure"] == "SEALED_NOT_READ"

    assert audit["status"] == "DATASET_QUALIFICATION_DEVELOPMENT_TABLE_RENDERED"
    assert audit["authority"]["inventory"] == (
        "configs/route_a_v3_route2_14_study_final_inventory_v1.json"
    )
    assert audit["record_totals"]["generated_candidate_canonical_credit"] == 0
    assert audit["protected_outcomes"] == {
        "development_test_read": False,
        "new_final_evaluation_read": False,
        "emtab10902_outcome_read": False,
        "sealed_gse246381_read": False,
        "guided_xeditflow_run": False,
    }
    assert json.loads(audit_path.read_text(encoding="utf-8")) == audit


def test_builder_refuses_implicit_overwrite(tmp_path: Path) -> None:
    table_path = tmp_path / "dataset_table.csv"
    audit_path = tmp_path / "dataset_table_audit.json"
    build_table(
        inventory_path=INVENTORY,
        table_path=table_path,
        audit_path=audit_path,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_table(
            inventory_path=INVENTORY,
            table_path=table_path,
            audit_path=audit_path,
        )
