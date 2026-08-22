import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "docs/paper/route2_v332_methods_results_draft_v1.md"
AUDIT = ROOT / "audits/route_a_v3_route2_v332_data_availability_completion_v1.json"
EVIDENCE = ROOT / "docs/paper/route2_v332_evidence_manifest_v1.json"
CONSISTENCY = ROOT / "docs/paper/route2_v332_consistency_manifest_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _section(draft: str) -> str:
    section = draft.split("## Data Availability\n", 1)[1]
    section = section.split("## Code Availability\n", 1)[0]
    return section.split("## Unresolved items before manuscript integration", 1)[0]


def test_data_availability_is_complete_but_release_and_submission_are_false() -> None:
    draft = DRAFT.read_text(encoding="utf-8")
    section = _section(draft)
    audit = _load(AUDIT)
    consistency = _load(CONSISTENCY)

    assert len(re.findall(r"^## Data Availability$", draft, flags=re.MULTILINE)) == 1
    assert audit["status"] == (
        "DATA_AVAILABILITY_SECTION_COMPLETE_INTERNAL_RIGHTS_REVIEW_PENDING"
    )
    assert audit["data_availability_paragraph_count"] == 5
    assert all(audit["coverage"].values())
    boundary = audit["completion_boundary"]
    assert boundary["data_availability_section_complete"] is True
    assert boundary["accountable_human_rights_review_complete"] is False
    assert boundary["human_evidence_content_verification_complete"] is False
    assert boundary["public_data_release_ready"] is False
    assert boundary["stable_repository_version_assigned"] is False
    assert boundary["minimum_package_complete"] is False
    assert boundary["outcome_trigger_fully_satisfied"] is False
    assert boundary["submission_ready"] is False
    assert all(value is False for value in audit["protected_outcomes"].values())
    current = consistency["manuscript_sections"]["data_availability"]
    assert current["status"] == "COMPLETE_INTERNAL_RIGHTS_REVIEW_PENDING"
    assert current["provider_evidence_audit"] == (
        "audits/route_a_v3_route2_v332_study_rights_provider_evidence_table_v1.json"
    )
    assert current["accountable_human_review_packet_audit"] == (
        "audits/route_a_v3_route2_v332_study_rights_accountable_human_review_packet_v1.json"
    )
    assert current["accountable_human_review_packet_template_ready"] is True
    assert current["accountable_human_review_completed_count"] == 0
    assert current["accountable_human_rights_review_complete"] is False
    assert current["provider_evidence_human_verification_complete"] is False
    assert current["fair_interoperability_assessment_complete"] is False
    assert current["fair_reusable_license_evidence_complete"] is False
    assert current["public_release_ready"] is False
    assert current["submission_ready"] is False
    assert "data availability" not in draft.split(
        "## Unresolved items before manuscript integration", 1
    )[1].lower()
    assert "## Code Availability" not in section


def test_data_availability_retains_rights_storage_and_promise_boundaries() -> None:
    section = " ".join(_section(DRAFT.read_text(encoding="utf-8")).split())
    facts = _load(AUDIT)["availability_facts"]
    consistency = _load(CONSISTENCY)
    method = next(
        row
        for row in consistency["methods"]
        if row["method_id"] == "M-R2-STUDY-RIGHTS-PROVIDER-EVIDENCE"
    )
    review_method = next(
        row
        for row in consistency["methods"]
        if row["method_id"]
        == "M-R2-STUDY-RIGHTS-ACCOUNTABLE-HUMAN-REVIEW-PACKET"
    )

    assert facts["study_count"] == 14
    assert facts["provider_evidence_row_count"] == 14
    assert facts["provider_counts"] == {
        "NCBI_GEO": 12,
        "ENCODE": 1,
        "EMBL_EBI_BIOSTUDIES_ARRAYEXPRESS": 1,
    }
    assert facts["official_accession_resolution_count"] == 14
    assert facts["analysis_and_publication_use_route_supported_count"] == 14
    assert facts["study_specific_license_record_present_count"] == 0
    assert facts["license_human_review_pending_count"] == 14
    assert facts["public_study_payload_release_authorized_count"] == 0
    assert facts["fair_evidence_counts"] == {
        "findable": 14,
        "accessible_metadata": 14,
        "interoperable_metadata_assessed": 0,
        "reusable_license_complete": 0,
    }
    assert facts["official_provider_policy_is_study_specific_license"] is False
    assert facts["official_provider_evidence_human_verified"] is False
    assert facts["accountable_human_review_packet_row_count"] == 14
    assert facts["accountable_human_review_packet_template_ready"] is True
    assert facts["accountable_human_review_completed_count"] == 0
    assert facts["accountable_human_review_hold_count"] == 0
    assert facts["rows_with_accountable_human_signoff"] == 0
    assert facts["rows_with_target_journal_policy_checked"] == 0
    assert facts["agent_review_substituted_for_accountable_human"] is False
    assert facts["human_review_completion_is_project_release_authorization"] is False
    assert method["row_count"] == facts["provider_evidence_row_count"] == 14
    assert method["repository_accession_resolution_count"] == 14
    assert method["analysis_and_publication_use_route_supported_count"] == 14
    assert method["study_specific_license_record_present_count"] == 0
    assert method["project_payload_redistribution_authorized_count"] == 0
    assert method["fair_evidence_counts"] == facts["fair_evidence_counts"]
    assert method["human_content_and_rights_verification_complete"] is False
    assert method["submission_ready"] is False
    assert review_method["row_count"] == 14
    assert review_method["field_count"] == 42
    assert review_method["human_review_counts"] == {
        "pending": 14,
        "completed": 0,
        "hold": 0,
    }
    assert review_method["agent_review_substituted_for_human"] is False
    assert review_method["human_content_and_rights_verification_complete"] is False
    assert review_method["review_completion_is_project_release_authorization"] is False
    assert review_method["project_public_release_authorized"] is False
    assert facts["third_party_current_access_verified_for_all_studies"] is False
    assert facts["third_party_reuse_terms_verified_for_all_studies"] is False
    assert facts["project_specific_study_payload_public_release_declared"] is False
    assert facts["model_weight_public_release_declared"] is False
    assert facts["generated_candidate_collection_public_release_declared"] is False
    assert facts["permanent_public_archive_assigned"] is False
    assert facts["availability_on_request_promise_made"] is False
    assert facts["large_artifact_storage_root"] in section
    assert "does not independently establish current access" in section
    assert "not a permanent archive or an open-data release" in section
    assert "No availability-on-request promise is made" in section
    assert "authorizes zero of 14 study payloads for public release" in section
    assert "12 GEO studies" in section
    assert "NCBI cannot grant unrestricted permission" in section
    assert "current CC0 policy for new BioStudies submissions was not applied retrospectively" in section
    assert "zero study-specific license records" in section
    assert "14 findable, 14 metadata-accessible" in section


def test_data_availability_evidence_is_registered_without_new_claim_markers() -> None:
    draft = DRAFT.read_text(encoding="utf-8")
    section = _section(draft)
    evidence_ids = {row["evidence_id"] for row in _load(EVIDENCE)["sources"]}
    cited = set()
    for group in re.findall(r"\[evidence:([^\]]+)\]", section):
        cited.update(item.strip() for item in group.split(","))

    assert cited <= evidence_ids
    assert "E-R2-DATA-AVAILABILITY-COMPLETION-AUDIT" in evidence_ids
    assert {
        "E-R2-RIGHTS-HUMAN-REVIEW-BUILDER",
        "E-R2-RIGHTS-HUMAN-REVIEW-AUDIT",
    } <= evidence_ids
    assert {
        "E-R2-RIGHTS-PROVIDER-SNAPSHOT",
        "E-R2-RIGHTS-PROVIDER-BUILDER",
        "E-R2-RIGHTS-PROVIDER-AUDIT",
    } <= cited
    assert len(re.findall(r"\[claim:C-R2-\d{3}\]", draft)) == 22
