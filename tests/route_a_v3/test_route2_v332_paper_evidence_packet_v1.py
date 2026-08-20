import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "docs/paper/route2_v332_methods_results_draft_v1.md"
EVIDENCE = ROOT / "docs/paper/route2_v332_evidence_manifest_v1.json"
CONSISTENCY = ROOT / "docs/paper/route2_v332_consistency_manifest_v1.json"
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
    assert len(claims) == len(set(claims)) == 14

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
