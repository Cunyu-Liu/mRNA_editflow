import csv
import json
from pathlib import Path

import pytest

from scripts.route_a_v3.build_route2_v332_study_rights_provider_evidence_table_v1 import (
    build_table,
)


ROOT = Path(__file__).resolve().parents[2]
RIGHTS_TABLE = ROOT / "docs/paper/route2_v332_data_rights_exposure_limitations_table_v1.csv"
SOURCE_SNAPSHOT = (
    ROOT / "audits/route_a_v3_route2_v332_rights_provider_official_source_snapshot_v1.json"
)
COMMITTED_TABLE = (
    ROOT / "docs/paper/route2_v332_study_rights_provider_evidence_table_v1.csv"
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_builder_binds_official_provider_routes_without_authorizing_release(
    tmp_path: Path,
) -> None:
    table_path = tmp_path / "provider_evidence.csv"
    audit_path = tmp_path / "provider_evidence.json"
    audit = build_table(
        rights_table_path=RIGHTS_TABLE,
        source_snapshot_path=SOURCE_SNAPSHOT,
        table_path=table_path,
        audit_path=audit_path,
    )
    rows = _rows(table_path)
    by_id = {row["study_unit_id"]: row for row in rows}

    assert len(rows) == len(by_id) == audit["row_count"] == 14
    assert audit["provider_counts"] == {
        "NCBI_GEO": 12,
        "ENCODE": 1,
        "EMBL_EBI_BIOSTUDIES_ARRAYEXPRESS": 1,
    }
    assert audit["official_provider_policy_bound_count"] == 14
    assert audit["repository_accession_resolution_count"] == 14
    assert audit["analysis_and_publication_use_route_supported_count"] == 14
    assert audit["study_specific_license_record_present_count"] == 0
    assert audit["project_payload_redistribution_authorized_count"] == 0
    assert audit["accountable_human_review_pending_count"] == 14
    assert audit["prior_converter_declaration_counts"] == {
        "true": 1,
        "false": 8,
        "not_declared_as_boolean": 5,
    }
    assert all(row["official_provider_policy_bound"] == "true" for row in rows)
    assert all(
        row["provider_analysis_and_publication_use_supported"] == "true"
        for row in rows
    )
    assert all(row["study_specific_license_record_present"] == "false" for row in rows)
    assert all(
        row["project_payload_redistribution_authorized"] == "false" for row in rows
    )
    assert by_id["GSE217518"]["prior_declared_public_redistribution_allowed"] == "true"
    assert by_id["GSE217518"]["project_payload_redistribution_authorized"] == "false"
    assert json.loads(audit_path.read_text(encoding="utf-8")) == audit


def test_provider_specific_exceptions_and_fair_gaps_remain_explicit(
    tmp_path: Path,
) -> None:
    table_path = tmp_path / "provider_evidence.csv"
    audit_path = tmp_path / "provider_evidence.json"
    audit = build_table(
        rights_table_path=RIGHTS_TABLE,
        source_snapshot_path=SOURCE_SNAPSHOT,
        table_path=table_path,
        audit_path=audit_path,
    )
    by_id = {row["study_unit_id"]: row for row in _rows(table_path)}

    assert "SUBMITTER_IP_EXCEPTION" in by_id["GSE200304"]["provider_distribution_policy"]
    assert by_id["ENCSR854RUF"]["accession_resolution_status"] == (
        "OFFICIAL_LANDING_HTTP_200_REDIRECT_PUBLICATION_DATA"
    )
    assert by_id["E-MTAB-10902"]["provider_distribution_policy"] == (
        "NEW_DATASET_CC0_SCOPE_NOT_BOUND_TO_MIGRATED_E_MTAB_RECORD"
    )
    assert by_id["E-MTAB-10902"]["study_specific_license_value"] == ""
    assert audit["fair_evidence_counts"] == {
        "findable": 14,
        "accessible_metadata": 14,
        "interoperable_metadata_assessed": 0,
        "reusable_license_complete": 0,
    }
    assert audit["provider_policy_is_study_specific_license"] is False
    assert audit["public_release_authorized"] is False
    assert audit["human_content_and_rights_verification_complete"] is False
    assert audit["submission_ready"] is False
    assert all(value is False for value in audit["protected_outcomes"].values())


def test_builder_refuses_implicit_overwrite(tmp_path: Path) -> None:
    table_path = tmp_path / "provider_evidence.csv"
    audit_path = tmp_path / "provider_evidence.json"
    build_table(
        rights_table_path=RIGHTS_TABLE,
        source_snapshot_path=SOURCE_SNAPSHOT,
        table_path=table_path,
        audit_path=audit_path,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_table(
            rights_table_path=RIGHTS_TABLE,
            source_snapshot_path=SOURCE_SNAPSHOT,
            table_path=table_path,
            audit_path=audit_path,
        )


def test_committed_table_is_reproducible_from_frozen_metadata_only(
    tmp_path: Path,
) -> None:
    table_path = tmp_path / "provider_evidence.csv"
    audit_path = tmp_path / "provider_evidence.json"
    build_table(
        rights_table_path=RIGHTS_TABLE,
        source_snapshot_path=SOURCE_SNAPSHOT,
        table_path=table_path,
        audit_path=audit_path,
    )

    assert table_path.read_text(encoding="utf-8") == COMMITTED_TABLE.read_text(
        encoding="utf-8"
    )
