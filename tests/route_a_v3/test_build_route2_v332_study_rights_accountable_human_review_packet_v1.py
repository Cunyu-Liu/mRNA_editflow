import csv
import json
from pathlib import Path

import pytest

from scripts.route_a_v3.build_route2_v332_study_rights_accountable_human_review_packet_v1 import (
    DEFAULT_INSTRUCTIONS,
    DEFAULT_PACKET,
    DEFAULT_PROVIDER_TABLE,
    FIELDNAMES,
    HumanReviewPacketError,
    _audit,
    _instructions_text,
    _read_provider_rows,
    _template_rows,
    audit_review_packet,
    build_review_packet,
)


ROOT = Path(__file__).resolve().parents[2]
COMMITTED_PACKET = DEFAULT_PACKET
COMMITTED_INSTRUCTIONS = DEFAULT_INSTRUCTIONS
COMMITTED_AUDIT = (
    ROOT
    / "audits/route_a_v3_route2_v332_study_rights_accountable_human_review_packet_v1.json"
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_template_freezes_machine_evidence_and_leaves_all_human_reviews_pending(
    tmp_path: Path,
) -> None:
    packet = tmp_path / "review.csv"
    instructions = tmp_path / "instructions.md"
    audit_path = tmp_path / "audit.json"
    audit = build_review_packet(
        packet_path=packet,
        instructions_path=instructions,
        audit_path=audit_path,
    )
    rows = _rows(packet)

    assert len(rows) == audit["row_count"] == 14
    assert len(FIELDNAMES) == 42
    assert len({row["study_unit_id"] for row in rows}) == 14
    assert all(row["human_review_status"] == "PENDING" for row in rows)
    assert all(row["accountable_human_signoff"] == "" for row in rows)
    assert all(
        row["automated_analysis_publication_use_supported"] == "true"
        and row["automated_study_specific_license_record_present"] == "false"
        and row["automated_project_payload_redistribution_authorized"] == "false"
        for row in rows
    )
    assert audit["human_review_counts"] == {
        "pending": 14,
        "completed": 0,
        "hold": 0,
    }
    assert audit["template_ready"] is True
    assert audit["agent_review_substituted_for_human"] is False
    assert audit["human_content_and_rights_verification_complete"] is False
    assert audit["project_public_release_authorized"] is False
    assert audit["submission_ready"] is False
    assert all(value is False for value in audit["protected_outcomes"].values())
    assert json.loads(audit_path.read_text(encoding="utf-8")) == audit
    assert "Even 14 completed rows do not automatically" in instructions.read_text(
        encoding="utf-8"
    )


def test_validator_rejects_completed_row_without_accountable_evidence_and_signoff(
    tmp_path: Path,
) -> None:
    rows = _template_rows(_read_provider_rows(DEFAULT_PROVIDER_TABLE), DEFAULT_PROVIDER_TABLE)
    rows[0]["human_review_status"] = "COMPLETED"
    review_path = tmp_path / "invalid_review.csv"
    _write_rows(review_path, rows)

    with pytest.raises(HumanReviewPacketError, match="identity/signoff incomplete"):
        audit_review_packet(review_input_path=review_path)


def test_validator_accepts_one_complete_analysis_only_human_decision(
    tmp_path: Path,
) -> None:
    rows = _template_rows(_read_provider_rows(DEFAULT_PROVIDER_TABLE), DEFAULT_PROVIDER_TABLE)
    row = rows[0]
    row.update(
        {
            "human_review_status": "COMPLETED",
            "reviewer_name": "Accountable Reviewer",
            "reviewer_role": "Data rights reviewer",
            "reviewer_affiliation": "Author institution",
            "review_date_iso": "2026-08-22",
            "accession_landing_and_non_outcome_metadata_verified": "true",
            "non_outcome_dataset_content_scope_verified": "true",
            "primary_dataset_citation_verified": "true",
            "study_specific_rights_source_url": "https://example.invalid/reviewer-supplied-rights-source",
            "study_specific_rights_source_type": "REVIEWER_VERIFIED_TERMS",
            "study_specific_license_or_terms": "ANALYSIS_AND_CITATION_ONLY_TEST_FIXTURE",
            "analysis_and_publication_use_decision": "CONFIRMED",
            "project_payload_redistribution_decision": "NOT_AUTHORIZED",
            "target_journal_name": "VALIDATOR_TEST_JOURNAL",
            "target_journal_policy_checked": "true",
            "data_availability_wording_approved": "true",
            "accountable_human_signoff": "TEST_FIXTURE_SIGNOFF",
        }
    )
    review_path = tmp_path / "partial_review.csv"
    _write_rows(review_path, rows)
    audit = audit_review_packet(review_input_path=review_path)

    assert audit["human_review_counts"] == {
        "pending": 13,
        "completed": 1,
        "hold": 0,
    }
    assert audit["analysis_citation_only_completed_count"] == 1
    assert audit["exact_file_redistribution_authorized_review_count"] == 0
    assert audit["human_content_and_rights_verification_complete"] is False
    assert audit["project_public_release_authorized"] is False


def test_validator_rejects_machine_evidence_edits(tmp_path: Path) -> None:
    rows = _template_rows(_read_provider_rows(DEFAULT_PROVIDER_TABLE), DEFAULT_PROVIDER_TABLE)
    rows[0]["provider_policy_url"] = "https://example.invalid/tampered"
    review_path = tmp_path / "tampered_review.csv"
    _write_rows(review_path, rows)

    with pytest.raises(HumanReviewPacketError, match="machine evidence changed"):
        audit_review_packet(review_input_path=review_path)


@pytest.mark.parametrize(
    ("status", "expected_message"),
    [
        ("PENDING", "pending row cannot authorize exact files"),
        ("HOLD", "held row cannot authorize exact files"),
    ],
)
def test_pending_or_held_row_cannot_authorize_exact_files(
    tmp_path: Path, status: str, expected_message: str
) -> None:
    rows = _template_rows(_read_provider_rows(DEFAULT_PROVIDER_TABLE), DEFAULT_PROVIDER_TABLE)
    row = rows[0]
    row.update(
        {
            "human_review_status": status,
            "project_payload_redistribution_decision": "AUTHORIZED_EXACT_FILES",
            "authorized_exact_file_scope": "must-not-be-authorized.csv",
        }
    )
    if status == "HOLD":
        row.update(
            {
                "reviewer_name": "Accountable Reviewer",
                "reviewer_role": "Data rights reviewer",
                "reviewer_affiliation": "Author institution",
                "review_date_iso": "2026-08-22",
                "restriction_or_hold_reason": "Rights evidence unresolved",
                "accountable_human_signoff": "TEST_FIXTURE_SIGNOFF",
            }
        )
    review_path = tmp_path / f"{status.lower()}_authorized.csv"
    _write_rows(review_path, rows)

    with pytest.raises(HumanReviewPacketError, match=expected_message):
        audit_review_packet(review_input_path=review_path)


def test_builder_refuses_implicit_overwrite(tmp_path: Path) -> None:
    packet = tmp_path / "review.csv"
    instructions = tmp_path / "instructions.md"
    audit_path = tmp_path / "audit.json"
    build_review_packet(
        packet_path=packet,
        instructions_path=instructions,
        audit_path=audit_path,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_review_packet(
            packet_path=packet,
            instructions_path=instructions,
            audit_path=audit_path,
        )


def test_committed_packet_matches_default_builder_logic() -> None:
    provider_rows = _read_provider_rows(DEFAULT_PROVIDER_TABLE)
    expected_rows = _template_rows(provider_rows, DEFAULT_PROVIDER_TABLE)
    assert _rows(COMMITTED_PACKET) == expected_rows
    assert COMMITTED_INSTRUCTIONS.read_text(encoding="utf-8") == _instructions_text(
        DEFAULT_PACKET
    )
    expected_audit = _audit(
        expected_rows,
        provider_table_path=DEFAULT_PROVIDER_TABLE,
        packet_path=DEFAULT_PACKET,
        instructions_path=DEFAULT_INSTRUCTIONS,
    )
    assert json.loads(COMMITTED_AUDIT.read_text(encoding="utf-8")) == expected_audit
