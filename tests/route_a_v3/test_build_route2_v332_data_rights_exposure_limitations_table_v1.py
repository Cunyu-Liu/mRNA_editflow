import csv
import json
from pathlib import Path

import pytest

from scripts.route_a_v3.build_route2_v332_data_rights_exposure_limitations_table_v1 import (
    build_table,
)


ROOT = Path(__file__).resolve().parents[2]
DATASET_TABLE = ROOT / "docs/paper/route2_v332_dataset_qualification_table_v1.csv"
EVIDENCE_MANIFEST = ROOT / "docs/paper/route2_v332_evidence_manifest_v1.json"


def test_builder_keeps_rights_declarations_narrow_and_fail_closed(tmp_path: Path) -> None:
    table_path = tmp_path / "limitations.csv"
    audit_path = tmp_path / "limitations.json"
    audit = build_table(
        dataset_table_path=DATASET_TABLE,
        evidence_manifest_path=EVIDENCE_MANIFEST,
        table_path=table_path,
        audit_path=audit_path,
    )

    with table_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["study_unit_id"]: row for row in rows}

    assert len(rows) == len(by_id) == audit["row_count"] == 14
    values = [row["declared_public_redistribution_allowed"] for row in rows]
    assert values.count("true") == 1
    assert values.count("false") == 8
    assert values.count("") == 5
    assert by_id["GSE217518"]["declared_public_redistribution_allowed"] == "true"
    assert by_id["GSE217518"]["public_release_authorized"] == "false"
    assert by_id["GSE217518"]["declaration_scope"] == (
        "CONVERTER_OUTPUT_POLICY_NOT_LICENSE_VERIFICATION"
    )
    assert by_id["GSE256185"]["declared_public_redistribution_allowed"] == "false"
    assert "NOT_AUTHORIZED" in by_id["GSE256185"]["rights_evidence_locator"]
    assert by_id["GSE246381"]["rights_evidence_path"] == ""
    assert by_id["GSE246381"]["exposure_limitation"] == (
        "SEALED_PAYLOAD_AND_OUTCOME_NOT_READ"
    )
    assert all(row["license_verification_status"].startswith("HUMAN_REVIEW_PENDING") for row in rows)
    assert all(row["public_release_authorized"] == "false" for row in rows)
    assert all(row["publication_boundary"].startswith("REPORT_AGGREGATES") for row in rows)
    assert all(row["development_test_read"] == "false" for row in rows)
    assert all(row["new_final_evaluation_read"] == "false" for row in rows)
    assert all(row["guided_xeditflow_run"] == "false" for row in rows)

    assert audit["status"] == (
        "DATA_RIGHTS_EXPOSURE_LIMITATIONS_REPORTED_HUMAN_REVIEW_PENDING"
    )
    assert audit["converter_or_preflight_declaration_is_license_verification"] is False
    assert audit["study_bound_human_verified_license_registry_present"] is False
    assert audit["license_human_review_pending_count"] == 14
    assert audit["public_release_authorized_count"] == 0
    assert audit["data_rights_exposure_limitations_complete"] is True
    assert audit["minimum_package_complete"] is False
    assert audit["submission_ready"] is False
    assert all(value is False for value in audit["protected_outcomes"].values())
    assert json.loads(audit_path.read_text(encoding="utf-8")) == audit


def test_builder_refuses_implicit_overwrite(tmp_path: Path) -> None:
    table_path = tmp_path / "limitations.csv"
    audit_path = tmp_path / "limitations.json"
    build_table(
        dataset_table_path=DATASET_TABLE,
        evidence_manifest_path=EVIDENCE_MANIFEST,
        table_path=table_path,
        audit_path=audit_path,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_table(
            dataset_table_path=DATASET_TABLE,
            evidence_manifest_path=EVIDENCE_MANIFEST,
            table_path=table_path,
            audit_path=audit_path,
        )
