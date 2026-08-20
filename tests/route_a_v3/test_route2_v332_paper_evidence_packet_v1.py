import json
import re
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "docs/paper/route2_v332_methods_results_draft_v1.md"
EVIDENCE = ROOT / "docs/paper/route2_v332_evidence_manifest_v1.json"
CONSISTENCY = ROOT / "docs/paper/route2_v332_consistency_manifest_v1.json"
BOOTSTRAP_TABLE = ROOT / "docs/paper/route2_v332_generation_bootstrap_table_v1.csv"
EVALUATOR_TASK_TABLE = (
    ROOT / "docs/paper/route2_v332_independent_evaluator_task_table_v1.csv"
)
READINESS = (
    ROOT
    / "configs/route_a_v3_route2_mrnabert_critic_v2_guidance_readiness_protocol_v1.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_claim_and_consistency_evidence_references_are_closed() -> None:
    draft = DRAFT.read_text(encoding="utf-8")
    evidence = _load(EVIDENCE)
    consistency = _load(CONSISTENCY)

    evidence_ids = [row["evidence_id"] for row in evidence["sources"]]
    assert len(evidence_ids) == len(set(evidence_ids)) == 14

    claims = re.findall(r"\[claim:([^\]]+)\]", draft)
    assert len(claims) == len(set(claims)) == 16

    cited = set()
    for group in re.findall(r"\[evidence:([^\]]+)\]", draft):
        cited.update(item.strip() for item in group.split(","))
    for section in (consistency["methods"], consistency["results"]):
        for row in section:
            cited.update(row["evidence_ids"])
    assert cited <= set(evidence_ids)
    assert "E-R2-CRITIC-V2-READINESS" in cited


def test_paper_packet_matches_frozen_critic_v2_readiness_boundary() -> None:
    draft = DRAFT.read_text(encoding="utf-8")
    consistency = _load(CONSISTENCY)
    readiness = _load(READINESS)
    method = next(
        row
        for row in consistency["methods"]
        if row["method_id"] == "M-R2-CRITIC-V2-GATES"
    )

    assert method["status"] == readiness["status"]
    assert method["required_seeds"] == readiness["required_seeds"]
    assert method["single_frozen_test_seed"] == readiness["single_frozen_test_seed"]
    assert method["single_test_metric_policy"] == readiness["single_test_metric_policy"]
    assert method["required_loso_studies"] == readiness["required_loso_studies"]
    assert method["guided_generation_requires"] == readiness["guided_generation_requires"]
    assert "TEST metrics are report-only" in draft
    assert "guided XEditFlow remains unauthorized" in draft
    assert consistency["protected_outcomes"] == {
        "development_test_opened": False,
        "new_final_evaluation_opened": False,
        "guided_xeditflow_authorized": False,
    }


def test_generation_bootstrap_reporting_is_exact_and_source_paired() -> None:
    draft = DRAFT.read_text(encoding="utf-8")
    consistency = _load(CONSISTENCY)
    result = next(
        row
        for row in consistency["results"]
        if row["result_id"] == "R-R2-GENERATION"
    )
    with BOOTSTRAP_TABLE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 6
    assert {row["point_leader_method_id"] for row in rows} == {"genetic"}
    assert {row["analysis_unit"] for row in rows} == {"SOURCE"}
    assert {int(row["source_count"]) for row in rows} == {891}
    assert {int(row["bootstrap_seed"]) for row in rows} == {20260816}
    assert {int(row["bootstrap_iterations"]) for row in rows} == {10000}
    assert {int(row["defined_bootstrap_iterations"]) for row in rows} == {10000}
    assert all(float(row["leader_advantage_ci_95_lower"]) > 0.0 for row in rows)

    nearest = min(rows, key=lambda row: float(row["point_leader_advantage"]))
    assert nearest["candidate_method_id"] == "generate_then_rerank"
    assert float(nearest["point_leader_advantage"]) == result["nearest_competitor"][
        "point_leader_advantage"
    ]
    assert [
        float(nearest["leader_advantage_ci_95_lower"]),
        float(nearest["leader_advantage_ci_95_upper"]),
    ] == result["nearest_competitor"]["leader_advantage_ci_95"]
    assert result["bootstrap_analysis_unit"] == "SOURCE"
    assert result["bootstrap_seed"] == 20260816
    assert result["paired_comparison_count"] == 6
    assert result["all_leader_advantage_ci_95_lower_bounds_positive"] is True
    assert "Development independent-evaluator separation only" in draft


def test_independent_evaluator_task_reporting_preserves_heterogeneity() -> None:
    draft = DRAFT.read_text(encoding="utf-8")
    consistency = _load(CONSISTENCY)
    result = next(
        row
        for row in consistency["results"]
        if row["result_id"] == "R-R2-EVALUATOR"
    )
    with EVALUATOR_TASK_TABLE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == result["task_count"] == 9
    assert sum(int(row["record_count"]) for row in rows) == result[
        "task_record_count_sum"
    ] == 18293
    spearman = [float(row["spearman"]) for row in rows]
    assert sum(value > 0.0 for value in spearman) == result["positive_task_count"] == 5
    assert sum(value <= 0.0 for value in spearman) == result[
        "nonpositive_task_count"
    ] == 4
    assert [min(spearman), max(spearman)] == result["task_spearman_range"]
    worst = max(rows, key=lambda row: float(row["standardized_mae"]))
    assert worst["task_id"] == result["maximum_task_standardized_mae"]["task_id"]
    assert float(worst["standardized_mae"]) == result[
        "maximum_task_standardized_mae"
    ]["value"]
    assert "does not imply uniformly reliable task-level evaluation" in draft
