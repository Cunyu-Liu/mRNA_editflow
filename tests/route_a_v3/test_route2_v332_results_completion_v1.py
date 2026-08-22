import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "docs/paper/route2_v332_methods_results_draft_v1.md"
AUDIT = ROOT / "audits/route_a_v3_route2_v332_results_completion_v1.json"
EVIDENCE = ROOT / "docs/paper/route2_v332_evidence_manifest_v1.json"
CONSISTENCY = ROOT / "docs/paper/route2_v332_consistency_manifest_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_results_section_is_complete_but_success_and_submission_are_false() -> None:
    draft = DRAFT.read_text(encoding="utf-8")
    results = draft.split("## Results\n", 1)[1].split("## Discussion draft", 1)[0]
    headings = re.findall(r"^### (.+)$", results, flags=re.MULTILINE)
    audit = _load(AUDIT)
    consistency = _load(CONSISTENCY)

    assert "## Results draft" not in draft
    assert headings == audit["required_subsections"]
    assert len(headings) == audit["results_subsection_count"] == 15
    assert audit["status"] == (
        "RESULTS_SECTION_COMPLETE_INTERNAL_HUMAN_VERIFICATION_PENDING"
    )
    assert all(audit["coverage"].values())
    boundary = audit["completion_boundary"]
    assert boundary["results_section_complete"] is True
    assert boundary["human_evidence_content_verification_complete"] is False
    assert boundary["model_success_established"] is False
    assert boundary["biological_success_established"] is False
    assert boundary["external_transfer_established"] is False
    assert boundary["guided_generation_success_established"] is False
    assert boundary["minimum_package_complete"] is False
    assert boundary["outcome_trigger_fully_satisfied"] is False
    assert boundary["submission_ready"] is False
    assert all(value is False for value in audit["protected_outcomes"].values())
    section = consistency["manuscript_sections"]["results"]
    assert section["status"] == "COMPLETE_INTERNAL_HUMAN_VERIFICATION_PENDING"
    assert section["subsection_count"] == 15
    assert section["submission_ready"] is False


def test_results_terminal_facts_retain_negative_and_layered_boundaries() -> None:
    draft = " ".join(DRAFT.read_text(encoding="utf-8").split())
    facts = _load(AUDIT)["terminal_result_facts"]

    assert facts["baseline_inventory_row_count"] == 45
    assert facts["three_track_result_row_count"] == 52
    assert facts["a1_numeric_task_row_count"] == 9
    assert facts["true_a2_terminal_numeric_result_row_count"] == 0
    assert facts["fully_contract_matched_headline_row_count"] == 0
    assert facts["generation_method_count"] == 7
    assert facts["hard_legality_all_methods"] == 1.0
    assert facts["critic_v1_positive_margin_seed_count"] == 1
    assert facts["critic_v1_seed_count"] == 3
    assert facts["critic_v2_status"] == (
        "CRITIC_V2_CONTROLS_DO_NOT_SUPPORT_THREE_FROZEN_SEEDS"
    )
    assert facts["critic_v2_strongest_baseline_margin"] < 0
    assert facts["historical_gse232572_preregistered_pass"] is False
    assert facts["new_outcome_unexposed_evaluation_record_count"] == 0
    assert facts["minimum_package_blocker_ids"] == [
        "MBP-10", "MBP-13", "MBP-14", "MBP-15"
    ]
    assert facts["selected_final_paper_outcome"] == (
        "BENCHMARK_PLUS_TRANSFER_AND_GENERATION_LIMITS_PAPER"
    )
    assert "Every method has zero sources with defined closed measured NDCG" in draft
    assert "no biological or guided-generation improvement is established" in draft
    assert "itemwise closure is complete even though the package itself is not" in draft


def test_results_evidence_ids_are_registered_and_claim_count_is_stable() -> None:
    draft = DRAFT.read_text(encoding="utf-8")
    results = draft.split("## Results\n", 1)[1].split("## Discussion draft", 1)[0]
    evidence = _load(EVIDENCE)
    evidence_ids = {row["evidence_id"] for row in evidence["sources"]}
    cited = set()
    for group in re.findall(r"\[evidence:([^\]]+)\]", results):
        cited.update(item.strip() for item in group.split(","))

    assert cited <= evidence_ids
    assert "E-R2-RESULTS-COMPLETION-AUDIT" in evidence_ids
    assert len(re.findall(r"\[claim:C-R2-\d{3}\]", draft)) == 22
