"""Contract tests for public_intervention_contract_v1 (R0-02 acceptance).

    pytest tests/test_public_intervention_contract.py -q
"""
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "public_intervention_contract.yaml"
SQ_PATH = ROOT / "docs" / "public_intervention_scientific_question.md"
CLAIM_PATH = ROOT / "docs" / "public_intervention_claim_matrix.md"


@pytest.fixture(scope="module")
def contract():
    with CONTRACT_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_contract_file_exists():
    assert CONTRACT_PATH.is_file()
    assert SQ_PATH.is_file()
    assert CLAIM_PATH.is_file()


def test_contract_identity(contract):
    assert contract["contract_id"] == "public_intervention_contract_v1"
    assert contract["status"] == "FROZEN"
    assert "SUPERSEDED" in contract["supersedes"]


def test_scientific_question_fixed(contract):
    q = contract["scientific_question"]
    assert "publicly measured" in q
    assert "local sequence edits" in q
    for kw in ("sources", "genes", "studies", "contexts", "regions"):
        assert kw in q
    decomp = contract["scientific_question_decomposition"]
    for key in ("Q1_local_effect_predictability", "Q2_transfer_across_public_studies",
                "Q3_endpoint_structure", "Q4_action_representation", "Q5_flow_value"):
        assert key in decomp


def test_primary_benchmark_and_datasets(contract):
    assert contract["primary_benchmark"] == "EditBench-5U-Natural"
    bms = contract["benchmarks"]
    assert set(bms) == {"EditBench-5U-Natural", "EditBench-5U-Dense",
                        "EditBench-3U-Variant", "EditBench-CDS-Synonymous"}
    assert bms["EditBench-5U-Natural"]["allowed_pair_types"] == ["true_wt_mutant"]
    assert bms["EditBench-5U-Dense"]["allowed_pair_types"] == ["dense_synthetic_neighbor"]
    assert bms["EditBench-CDS-Synonymous"]["allowed_pair_types"] == ["same_protein_synonymous_family"]

    ds = contract["primary_datasets"]
    accessions = {d["accession"] for d in ds.values()}
    assert {"GSE114002", "GSE149487", "GSE145046", "GSE246381"} <= accessions


def test_sealed_external_dataset(contract):
    sealed = contract["sealed_external_dataset"]
    assert sealed["accession"] == "GSE246381"
    assert "must not be read" in sealed["rule"]
    ext = contract["targets"]["external_temporal"]
    assert ext["dataset"] == "GSE246381"
    assert ext["delta_spearman"] == pytest.approx(0.25)
    assert ext["sign_accuracy"] == pytest.approx(0.60)
    assert ext["top10pct_enrichment"] == pytest.approx(1.5)
    assert ext["immutable_after_unseal"] is True


def test_endpoint_separation(contract):
    ep = contract["endpoint_separation"]["endpoints"]
    for required in ("mean_ribosome_loading", "translation_efficiency",
                     "rna_abundance", "half_life", "protein_abundance"):
        assert required in ep
    assert len(ep) == len(set(ep)), "endpoints must be distinct"
    rules = " ".join(contract["endpoint_separation"]["rules"])
    assert "not be called protein output" in rules
    assert "never" in rules or "not" in rules


def test_splits_fixed(contract):
    sp = contract["splits"]
    assert sp["random_pair_split_forbidden"] is True
    for key in ("split_A_source_disjoint", "split_B_gene_disjoint",
                "split_C_study_disjoint", "split_D_context_disjoint",
                "split_E_temporal_external"):
        assert key in sp["primary_splits"]
    assert "GSE246381" in sp["primary_splits"]["split_E_temporal_external"]["sealed_external"][0]
    for key in ("block_split", "position_holdout", "motif_holdout"):
        assert key in sp["dense_landscape_splits"]


def test_primary_metrics_fixed(contract):
    metrics = set(contract["primary_metrics"])
    assert {"delta_spearman", "sign_accuracy", "pairwise_ranking_auc",
            "top10pct_enrichment", "ndcg", "rmse", "ece",
            "coverage_risk"} <= metrics
    rules = " ".join(contract["metric_rules"])
    assert "macro" in rules


def test_targets_fixed(contract):
    tg = contract["targets"]
    mv = tg["minimum_scientific_validity"]
    assert mv["delta_spearman"] == pytest.approx(0.30)
    assert mv["sign_accuracy"] == pytest.approx(0.60)
    assert mv["top10pct_enrichment"] == pytest.approx(1.5)
    st = tg["submission_stretch"]
    assert st["macro_delta_spearman"] == pytest.approx(0.35)
    assert st["ece"] == pytest.approx(0.10)
    op = tg["optimization"]
    assert op["top10_recall"] == pytest.approx(0.70)
    assert op["normalized_regret"] == pytest.approx(0.10)


def test_claim_boundaries(contract):
    assert "harmonized benchmark" in contract["claims"]["primary"]
    assert len(contract["claims"]["secondary"]) >= 4
    forbidden = " ".join(contract["claims"]["forbidden"]).lower()
    assert len(contract["claims"]["forbidden"]) >= 7
    for kw in ("protein output", "wet", "observational data",
               "wt-mutant", "synonymous"):
        assert kw in forbidden


def test_no_wetlab_statement(contract):
    stmt = contract["no_wetlab_statement"].lower()
    assert "no new wet-lab" in stmt
    assert "predicted" in stmt


def test_evidence_grades_and_pair_types(contract):
    grades = set(contract["evidence_grades"])
    assert grades == {"A1", "A2", "B1", "B2", "C", "D"}
    pts = set(contract["pair_types"])
    assert {"true_wt_mutant", "dense_synthetic_neighbor",
            "same_protein_synonymous_family"} <= pts


def test_docs_reference_contract_id():
    sq = SQ_PATH.read_text(encoding="utf-8")
    cm = CLAIM_PATH.read_text(encoding="utf-8")
    assert "public_intervention_contract_v1" in sq
    assert "public_intervention_contract_v1" in cm
    assert "sealed" in sq.lower()
    assert "Forbidden claims" in cm
