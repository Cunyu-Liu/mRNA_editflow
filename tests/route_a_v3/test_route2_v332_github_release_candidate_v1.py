import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
DRAFT = ROOT / "docs/paper/route2_v332_methods_results_draft_v1.md"
AUDIT = ROOT / "audits/route_a_v3_route2_v332_github_release_candidate_v1.json"
EVIDENCE = ROOT / "docs/paper/route2_v332_evidence_manifest_v1.json"
CONSISTENCY = ROOT / "docs/paper/route2_v332_consistency_manifest_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_readme_notice_and_manuscript_title_bind_the_current_route() -> None:
    readme = README.read_text(encoding="utf-8")
    draft = DRAFT.read_text(encoding="utf-8")
    readme_notice = " ".join(readme[:2500].replace(">", " ").split())
    draft_header = " ".join(draft[:800].replace(">", " ").split())

    assert readme.startswith("# mRNA-EditFlow\n")
    assert "Route A V3.3.2 Route 2 branch notice" in readme_notice
    assert "route-a-v3-route2-method-repair-20260817" in readme_notice
    assert "repository-wide v2 material below predates this branch packet" in readme_notice
    assert "BENCHMARK_PLUS_TRANSFER_AND_GENERATION_LIMITS_PAPER" in readme_notice
    assert (
        "Development TEST, new final Evaluation and guided XEditFlow were not opened"
        in readme_notice
    )
    assert draft.startswith("# Route 2 V3.3.2 Benchmark+limits evidence manuscript\n")
    assert "Data and Code Availability are internal review-pending statements" in draft_header


def test_internal_branch_candidate_is_complete_but_formal_release_is_blocked() -> None:
    audit = _load(AUDIT)
    consistency = _load(CONSISTENCY)

    assert audit["status"] == (
        "INTERNAL_GITHUB_BRANCH_CANDIDATE_ASSEMBLED_FORMAL_RELEASE_NOT_AUTHORIZED"
    )
    candidate = audit["candidate"]
    assert candidate["candidate_assembled"] is True
    assert candidate["formal_github_release_created"] is False
    assert candidate["release_tag_created"] is False
    assert candidate["persistent_archive_identifier_assigned"] is False
    assert candidate["unauthenticated_public_access_verified"] is False
    assert candidate["submission_release_authorized"] is False
    packet = audit["current_packet"]
    assert packet["methods_subsection_count"] == 14
    assert packet["results_subsection_count"] == 15
    assert packet["discussion_subsection_count"] == 5
    assert packet["data_availability_internal_section_complete"] is True
    assert packet["code_availability_internal_section_complete"] is True
    assert packet["route2_v332_readme_notice_present"] is True
    assert packet["claim_marker_count"] == 22
    assert packet["supported_claim_row_count"] == 22
    assert packet["unsupported_claim_row_count"] == 13
    assert packet["evidence_source_count"] == 64
    assert packet["evidence_local_or_contract_count"] == 50
    assert packet["evidence_a100_mnt_count"] == 14
    assert packet["provider_rights_evidence_row_count"] == 14
    assert packet["study_specific_license_record_present_count"] == 0
    assert packet["project_payload_redistribution_authorized_count"] == 0
    assert packet["accountable_human_review_packet_template_ready"] is True
    assert packet["accountable_human_review_packet_row_count"] == 14
    assert packet["accountable_human_review_completed_count"] == 0
    assert packet["rows_with_accountable_human_signoff"] == 0
    assert packet["rows_with_target_journal_policy_checked"] == 0
    assert packet["fair_evidence_counts"] == {
        "findable": 14,
        "accessible_metadata": 14,
        "interoperable_metadata_assessed": 0,
        "reusable_license_complete": 0,
    }
    assert packet["minimum_package_blocker_ids"] == [
        "MBP-10", "MBP-13", "MBP-14", "MBP-15"
    ]
    assert all((ROOT / path).is_file() for path in audit["required_git_artifacts"])
    assert len(audit["formal_release_blockers"]) == 6
    assert all(value is False for value in audit["protected_outcomes"].values())
    assert all(
        section["submission_ready"] is False
        for section in consistency["manuscript_sections"].values()
    )
    boundary = audit["release_boundary"]
    assert boundary["github_branch_candidate_assembled"] is True
    assert all(value is False for key, value in boundary.items() if key != "github_branch_candidate_assembled")


def test_candidate_registers_tracked_payload_policy_without_opening_or_removing_it() -> None:
    audit = _load(AUDIT)
    policy = audit["tracked_data_policy"]
    evidence = _load(EVIDENCE)
    draft = DRAFT.read_text(encoding="utf-8")

    assert policy["contract_compliant_for_formal_release"] is False
    assert policy["automatic_removal_performed"] is False
    assert policy["tracked_weight_or_checkpoint_file_count"] == 0
    rows = policy["tracked_parquet_files"] + policy["tracked_legacy_b0_jsonl_files"]
    assert len(rows) == 5
    assert all((ROOT / row["path"]).stat().st_size == row["size_bytes"] for row in rows)
    assert policy["legacy_b0_jsonl_total_size_bytes"] == sum(
        row["size_bytes"] for row in policy["tracked_legacy_b0_jsonl_files"]
    )
    evidence_ids = [row["evidence_id"] for row in evidence["sources"]]
    assert len(evidence_ids) == len(set(evidence_ids)) == 64
    assert {
        "E-R2-RIGHTS-PROVIDER-SNAPSHOT",
        "E-R2-RIGHTS-PROVIDER-BUILDER",
        "E-R2-RIGHTS-PROVIDER-AUDIT",
        "E-R2-RIGHTS-HUMAN-REVIEW-BUILDER",
        "E-R2-RIGHTS-HUMAN-REVIEW-AUDIT",
    } <= set(evidence_ids)
    assert "E-R2-GITHUB-RC-AUDIT" in evidence_ids
    assert len(re.findall(r"\[claim:C-R2-\d{3}\]", draft)) == 22
