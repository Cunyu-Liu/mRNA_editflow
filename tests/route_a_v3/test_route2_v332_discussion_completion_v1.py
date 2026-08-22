import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "docs/paper/route2_v332_methods_results_draft_v1.md"
AUDIT = ROOT / "audits/route_a_v3_route2_v332_discussion_completion_v1.json"
EVIDENCE = ROOT / "docs/paper/route2_v332_evidence_manifest_v1.json"
CONSISTENCY = ROOT / "docs/paper/route2_v332_consistency_manifest_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_discussion_is_complete_but_success_and_submission_remain_false() -> None:
    draft = DRAFT.read_text(encoding="utf-8")
    discussion = draft.split("## Discussion\n", 1)[1].split(
        "## Data, rights and exposure limitations", 1
    )[0]
    headings = re.findall(r"^### (.+)$", discussion, flags=re.MULTILINE)
    audit = _load(AUDIT)
    consistency = _load(CONSISTENCY)

    assert "## Discussion draft" not in draft
    assert headings == audit["required_subsections"]
    assert len(headings) == audit["discussion_subsection_count"] == 5
    assert audit["status"] == (
        "DISCUSSION_SECTION_COMPLETE_INTERNAL_HUMAN_VERIFICATION_PENDING"
    )
    assert all(audit["coverage"].values())
    boundary = audit["completion_boundary"]
    assert boundary["discussion_section_complete"] is True
    assert boundary["human_evidence_content_verification_complete"] is False
    assert boundary["causal_failure_mechanism_established"] is False
    assert boundary["model_success_established"] is False
    assert boundary["biological_success_established"] is False
    assert boundary["external_transfer_established"] is False
    assert boundary["guided_generation_success_established"] is False
    assert boundary["minimum_package_complete"] is False
    assert boundary["outcome_trigger_fully_satisfied"] is False
    assert boundary["submission_ready"] is False
    assert all(value is False for value in audit["protected_outcomes"].values())
    section = consistency["manuscript_sections"]["discussion"]
    assert section["status"] == "COMPLETE_INTERNAL_HUMAN_VERIFICATION_PENDING"
    assert section["subsection_count"] == 5
    assert section["submission_ready"] is False


def test_discussion_retains_endpoint_and_noncausal_boundaries() -> None:
    draft = " ".join(DRAFT.read_text(encoding="utf-8").split())
    facts = _load(AUDIT)["terminal_interpretation_facts"]

    assert facts["critic_v2_spearman_win_count_vs_strongest_baseline"] == 4
    assert facts["critic_v2_spearman_loss_count_vs_strongest_baseline"] == 5
    assert facts["critic_v2_standardized_mae_worse_task_count_vs_strongest_baseline"] == 9
    assert facts["critic_v2_strongest_baseline_spearman_margin"] < 0
    assert facts["closed_measured_ndcg_defined_source_count_all_methods"] == 0
    assert facts["historical_gse232572_preregistered_pass"] is False
    assert facts["historical_gse232572_final_confirmation_eligible"] is False
    assert facts["minimum_package_blocker_ids"] == [
        "MBP-10", "MBP-13", "MBP-14", "MBP-15"
    ]
    assert "Neither is a closed-support biological ranking" in draft
    assert "cannot identify a causal region effect" in draft
    assert "leaves external transfer unresolved" in draft
    assert "not claims of model success, biological efficacy" in draft


def test_discussion_evidence_is_registered_without_new_claim_markers() -> None:
    draft = DRAFT.read_text(encoding="utf-8")
    discussion = draft.split("## Discussion\n", 1)[1].split(
        "## Data, rights and exposure limitations", 1
    )[0]
    evidence_ids = {row["evidence_id"] for row in _load(EVIDENCE)["sources"]}
    cited = set()
    for group in re.findall(r"\[evidence:([^\]]+)\]", discussion):
        cited.update(item.strip() for item in group.split(","))

    assert cited <= evidence_ids
    assert "E-R2-DISCUSSION-COMPLETION-AUDIT" in evidence_ids
    assert len(re.findall(r"\[claim:C-R2-\d{3}\]", draft)) == 22
