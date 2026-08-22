import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "docs/paper/route2_v332_methods_results_draft_v1.md"
AUDIT = ROOT / "audits/route_a_v3_route2_v332_code_availability_completion_v1.json"
EVIDENCE = ROOT / "docs/paper/route2_v332_evidence_manifest_v1.json"
CONSISTENCY = ROOT / "docs/paper/route2_v332_consistency_manifest_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _section(draft: str) -> str:
    return draft.split("## Code Availability\n", 1)[1].split(
        "## Unresolved items before manuscript integration", 1
    )[0]


def test_code_availability_is_complete_but_release_and_submission_are_false() -> None:
    draft = DRAFT.read_text(encoding="utf-8")
    audit = _load(AUDIT)
    consistency = _load(CONSISTENCY)

    assert len(re.findall(r"^## Code Availability$", draft, flags=re.MULTILINE)) == 1
    assert audit["status"] == (
        "CODE_AVAILABILITY_SECTION_COMPLETE_INTERNAL_RELEASE_REVIEW_PENDING"
    )
    assert audit["code_availability_paragraph_count"] == 3
    assert all(audit["coverage"].values())
    boundary = audit["completion_boundary"]
    assert boundary["code_availability_section_complete"] is True
    assert boundary["accountable_human_license_review_complete"] is False
    assert boundary["route2_current_readme_notice_complete"] is True
    assert boundary["route2_clean_environment_reproduction_complete"] is False
    assert boundary["immutable_release_assigned"] is False
    assert boundary["public_code_release_claimed"] is False
    assert boundary["minimum_package_complete"] is False
    assert boundary["outcome_trigger_fully_satisfied"] is False
    assert boundary["submission_ready"] is False
    assert all(value is False for value in audit["protected_outcomes"].values())
    current = consistency["manuscript_sections"]["code_availability"]
    assert current["status"] == "COMPLETE_INTERNAL_RELEASE_REVIEW_PENDING"
    assert current["accountable_human_license_review_complete"] is False
    assert current["route2_current_readme_notice_complete"] is True
    assert current["route2_clean_environment_reproduction_complete"] is False
    assert current["immutable_release_assigned"] is False
    assert current["submission_ready"] is False
    unresolved = draft.split("## Unresolved items before manuscript integration", 1)[1]
    assert "code availability" not in unresolved.lower()
    assert "Accountable code-license review" in unresolved


def test_code_availability_retains_repository_environment_and_license_boundaries() -> None:
    section = " ".join(_section(DRAFT.read_text(encoding="utf-8")).split())
    facts = _load(AUDIT)["repository_facts"]

    assert facts["repository_locator"] in section
    assert facts["working_branch"] in section
    assert facts["unauthenticated_public_access_verified"] is False
    assert facts["route2_v332_release_tag_assigned"] is False
    assert facts["persistent_archive_identifier_assigned"] is False
    assert facts["archived_container_image_assigned"] is False
    assert facts["environment_descriptors"] == [
        "pyproject.toml", "requirements-lock.txt", "environment.yml", "Dockerfile"
    ]
    assert all((ROOT / path).is_file() for path in facts["environment_descriptors"])
    assert facts["route2_v332_clean_environment_reproduction_complete"] is False
    assert facts["route2_v332_readme_notice_present"] is True
    assert facts["legacy_readme_body_marked_non_authoritative_for_route2_v332"] is True
    assert facts["pyproject_license_text"] == "Proprietary"
    assert facts["standalone_license_file_tracked"] is False
    assert not any(ROOT.glob("LICENSE*"))
    assert facts["code_availability_on_request_promise_made"] is False
    assert facts["large_artifact_storage_root"] in section
    assert "does not assert unauthenticated public access" in section
    assert "not yet been independently reproduced" in section
    assert "README now begins with a Route A V3.3.2 branch notice" in section
    assert "no standalone `LICENSE` file is tracked" in section
    assert "No code-availability-on-request promise is made" in section


def test_code_availability_evidence_is_registered_without_new_claim_markers() -> None:
    draft = DRAFT.read_text(encoding="utf-8")
    section = _section(draft)
    evidence_ids = {row["evidence_id"] for row in _load(EVIDENCE)["sources"]}
    cited = set()
    for group in re.findall(r"\[evidence:([^\]]+)\]", section):
        cited.update(item.strip() for item in group.split(","))

    assert cited <= evidence_ids
    assert "E-R2-CODE-AVAILABILITY-COMPLETION-AUDIT" in evidence_ids
    assert len(re.findall(r"\[claim:C-R2-\d{3}\]", draft)) == 22
