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
    assert audit["data_availability_paragraph_count"] == 3
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
    assert current["accountable_human_rights_review_complete"] is False
    assert current["public_release_ready"] is False
    assert current["submission_ready"] is False
    assert "data availability" not in draft.split(
        "## Unresolved items before manuscript integration", 1
    )[1].lower()
    assert "## Code Availability" not in section


def test_data_availability_retains_rights_storage_and_promise_boundaries() -> None:
    section = " ".join(_section(DRAFT.read_text(encoding="utf-8")).split())
    facts = _load(AUDIT)["availability_facts"]

    assert facts["study_count"] == 14
    assert facts["license_human_review_pending_count"] == 14
    assert facts["public_study_payload_release_authorized_count"] == 0
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


def test_data_availability_evidence_is_registered_without_new_claim_markers() -> None:
    draft = DRAFT.read_text(encoding="utf-8")
    section = _section(draft)
    evidence_ids = {row["evidence_id"] for row in _load(EVIDENCE)["sources"]}
    cited = set()
    for group in re.findall(r"\[evidence:([^\]]+)\]", section):
        cited.update(item.strip() for item in group.split(","))

    assert cited <= evidence_ids
    assert "E-R2-DATA-AVAILABILITY-COMPLETION-AUDIT" in evidence_ids
    assert len(re.findall(r"\[claim:C-R2-\d{3}\]", draft)) == 22
