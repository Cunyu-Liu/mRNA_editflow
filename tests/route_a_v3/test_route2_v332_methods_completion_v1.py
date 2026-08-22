import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "docs/paper/route2_v332_methods_results_draft_v1.md"
AUDIT = ROOT / "audits/route_a_v3_route2_v332_methods_completion_v1.json"
EVIDENCE = ROOT / "docs/paper/route2_v332_evidence_manifest_v1.json"
CONSISTENCY = ROOT / "docs/paper/route2_v332_consistency_manifest_v1.json"
EVALUATOR_CONFIG = (
    ROOT
    / "configs/route_a_v3_route2_independent_evaluator_neural_medium_task_scaled_gpu2_v3.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_methods_section_is_complete_but_not_submission_ready() -> None:
    draft = DRAFT.read_text(encoding="utf-8")
    methods = draft.split("## Methods\n", 1)[1].split("## Results\n", 1)[0]
    headings = re.findall(r"^### (.+)$", methods, flags=re.MULTILINE)
    audit = _load(AUDIT)
    consistency = _load(CONSISTENCY)

    assert "## Methods draft" not in draft
    assert headings == audit["required_subsections"]
    assert len(headings) == audit["methods_subsection_count"] == 14
    assert audit["status"] == (
        "METHODS_SECTION_COMPLETE_INTERNAL_HUMAN_VERIFICATION_PENDING"
    )
    assert all(audit["coverage"].values())
    boundary = audit["completion_boundary"]
    assert boundary["methods_section_complete"] is True
    assert boundary["human_evidence_content_verification_complete"] is False
    assert boundary["data_availability_complete"] is False
    assert boundary["code_availability_complete"] is False
    assert boundary["minimum_package_complete"] is False
    assert boundary["outcome_trigger_fully_satisfied"] is False
    assert boundary["submission_ready"] is False
    assert all(value is False for value in audit["protected_outcomes"].values())
    assert consistency["manuscript_sections"]["methods"]["subsection_count"] == 14
    assert consistency["manuscript_sections"]["methods"]["submission_ready"] is False
    assert consistency["manuscript_sections"]["results"]["status"] == (
        "COMPLETE_INTERNAL_HUMAN_VERIFICATION_PENDING"
    )


def test_methods_report_terminal_actuals_and_current_figure_counts() -> None:
    draft = " ".join(DRAFT.read_text(encoding="utf-8").split())
    audit = _load(AUDIT)
    config = _load(EVALUATOR_CONFIG)
    facts = audit["reproducibility_facts"]

    assert config["frozen_expected_parameter_count"] == facts[
        "independent_evaluator_config_expected_parameter_count"
    ] == 509905
    assert facts["independent_evaluator_terminal_actual_parameter_count"] == 509845
    assert facts["actual_minus_config_expected_parameter_count"] == -60
    assert facts["manuscript_figure_count"] == 7
    assert facts["manuscript_figure_builder_count"] == 6
    assert "Six reproducible builders render seven general-manuscript figures" in draft
    assert "Five reproducible builders render six" not in draft
    assert "The terminal actual count is reported" in draft
    assert "Statistical analysis and model-selection hierarchy" in draft
    assert "unknown generated outcomes were never assigned zero gain" in draft


def test_methods_evidence_ids_are_registered_and_claim_count_is_stable() -> None:
    draft = DRAFT.read_text(encoding="utf-8")
    methods = draft.split("## Methods\n", 1)[1].split("## Results\n", 1)[0]
    evidence = _load(EVIDENCE)
    evidence_ids = {row["evidence_id"] for row in evidence["sources"]}
    cited = set()
    for group in re.findall(r"\[evidence:([^\]]+)\]", methods):
        cited.update(item.strip() for item in group.split(","))

    assert cited <= evidence_ids
    assert "E-R2-METHODS-COMPLETION-AUDIT" in evidence_ids
    assert len(re.findall(r"\[claim:C-R2-\d{3}\]", draft)) == 22
