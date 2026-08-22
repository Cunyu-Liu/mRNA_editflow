import csv
from pathlib import Path

import pytest

from scripts.route_a_v3.build_route2_v332_selected_outcome_claim_evidence_table_v1 import (
    DEFAULT_DRAFT,
    ClaimEvidenceInputError,
    build_table,
)


def _build(tmp_path: Path, *, draft_path: Path = DEFAULT_DRAFT):
    table_path = tmp_path / "claim_evidence.csv"
    audit_path = tmp_path / "claim_evidence.json"
    audit = build_table(
        draft_path=draft_path,
        output_table_path=table_path,
        output_audit_path=audit_path,
    )
    with table_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return audit, rows, table_path, audit_path


def test_builder_closes_all_draft_claims_and_lists_unsupported_claims(tmp_path: Path) -> None:
    audit, rows, table_path, audit_path = _build(tmp_path)

    assert audit["status"] == "SELECTED_OUTCOME_CLAIM_EVIDENCE_CLOSED_UNSUPPORTED_CLAIMS_EXPLICIT"
    assert audit["selected_final_paper_outcome"] == "BENCHMARK_PLUS_TRANSFER_AND_GENERATION_LIMITS_PAPER"
    assert audit["row_count"] == len(rows) == 35
    assert audit["draft_claim_marker_count"] == 22
    assert audit["supported_with_declared_boundary_row_count"] == 22
    assert audit["unsupported_claim_row_count"] == 13
    assert audit["unmapped_draft_claim_marker_count"] == 0
    assert audit["unknown_evidence_id_reference_count"] == 0
    assert audit["unsupported_claims_allowed_in_manuscript_count"] == 0
    assert audit["claim_evidence_table_complete"] is True
    assert audit["outcome_trigger_fully_satisfied"] is False
    assert audit["submission_ready"] is False
    assert audit["minimum_package_complete"] is False
    assert audit["model_or_biological_success_established"] is False

    supported = [row for row in rows if row["claim_status"].startswith("SUPPORTED")]
    unsupported = [row for row in rows if row["claim_status"] == "UNSUPPORTED"]
    assert len(supported) == 22 and len(unsupported) == 13
    assert {row["claim_id"] for row in supported} == {
        f"C-R2-{index:03d}" for index in range(1, 23)
    }
    assert {row["claim_id"] for row in unsupported} == {
        f"U-R2-{index:03d}" for index in range(1, 14)
    }
    assert all(row["claim_marker_present"] == "true" for row in supported)
    assert all(row["claim_marker_present"] == "false" for row in unsupported)
    assert all(row["allowed_in_selected_outcome_manuscript"] == "true" for row in supported)
    assert all(row["allowed_in_selected_outcome_manuscript"] == "false" for row in unsupported)
    assert all(row["evidence_ids"] for row in rows)
    assert all(row["minimum_package_complete"] == "false" for row in rows)
    assert all(row["outcome_trigger_fully_satisfied"] == "false" for row in rows)
    assert all(row["submission_ready"] == "false" for row in rows)
    assert all(row["development_test_read"] == "false" for row in rows)
    assert all(row["new_final_evaluation_read"] == "false" for row in rows)
    assert all(row["guided_xeditflow_run"] == "false" for row in rows)
    assert next(row for row in unsupported if row["claim_id"] == "U-R2-010")["claim_class"] == "HISTORICAL_FINAL_CONFIRMATION"
    assert next(row for row in unsupported if row["claim_id"] == "U-R2-013")["claim_class"] == "CAUSAL_REGION_CONTEXT_MECHANISM"

    with pytest.raises(FileExistsError):
        build_table(output_table_path=table_path, output_audit_path=audit_path)

    overwritten = build_table(
        output_table_path=table_path,
        output_audit_path=audit_path,
        overwrite=True,
    )
    assert overwritten == audit


def test_builder_rejects_an_unmapped_draft_claim_marker(tmp_path: Path) -> None:
    bad_draft = tmp_path / "draft.md"
    bad_draft.write_text(
        DEFAULT_DRAFT.read_text(encoding="utf-8")
        + "\n\nUnsupported accidental marker. [claim:C-R2-999] [evidence:E-R2-CONTRACT]\n",
        encoding="utf-8",
    )
    with pytest.raises(ClaimEvidenceInputError, match="claim marker set changed"):
        _build(tmp_path / "bad", draft_path=bad_draft)
