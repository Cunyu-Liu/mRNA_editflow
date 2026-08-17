import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = (
    REPO_ROOT / "configs" / "route_a_v3_route2_14_study_final_inventory_v1.json"
)


def _load_inventory() -> dict:
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


def test_inventory_has_exact_contract_studies_and_reconciled_totals() -> None:
    inventory = _load_inventory()
    studies = inventory["studies"]

    assert len(studies) == 14
    assert len({study["study_unit_id"] for study in studies}) == 14

    expected_ids = {
        "GSE200304",
        "GSE114002",
        "GSE149487",
        "GSE217518",
        "ENCSR854RUF",
        "GSE186455",
        "GSE256185",
        "GSE269595",
        "GSE232572",
        "E-MTAB-10902",
        "GSE145046",
        "GSE207584",
        "GSE261709",
        "GSE246381",
    }
    assert {study["study_unit_id"] for study in studies} == expected_ids

    development_total = sum(
        study["canonical_records"]
        for study in studies
        if study["use_role"] == "DEVELOPMENT"
    )
    evaluation_total = sum(
        study["canonical_records"]
        for study in studies
        if study["use_role"] == "EVALUATION"
    )
    assert development_total == inventory["totals"]["development_canonical_records"]
    assert evaluation_total == inventory["totals"]["evaluation_canonical_records"]
    assert inventory["totals"]["qualified_canonical_records"] == 6547
    assert inventory["totals"]["generated_candidate_canonical_credit"] == 0


def test_zero_record_studies_have_explicit_terminal_reasons() -> None:
    inventory = _load_inventory()
    zero_record_studies = [
        study for study in inventory["studies"] if study["canonical_records"] == 0
    ]

    assert {study["study_unit_id"] for study in zero_record_studies} == {
        "GSE256185",
        "E-MTAB-10902",
        "GSE145046",
        "GSE207584",
        "GSE261709",
        "GSE246381",
    }
    for study in zero_record_studies:
        assert study["terminal_status"]
        assert study["reason"]

    emtab = next(
        study
        for study in zero_record_studies
        if study["study_unit_id"] == "E-MTAB-10902"
    )
    assert emtab["terminal_status"] == "UNCONVERTIBLE_FOR_ROUTE2_V1"
    assert emtab["outcome_exposure"] == "OUTCOME_NOT_READ"
    assert "5929" in emtab["reason"]
    assert "5730" in emtab["reason"]
    assert "5679" in emtab["reason"]


def test_evaluation_exposure_and_sealed_boundaries_remain_explicit() -> None:
    inventory = _load_inventory()
    by_id = {study["study_unit_id"]: study for study in inventory["studies"]}

    assert by_id["GSE232572"]["outcome_exposure"] == (
        "OUTCOME_EXPOSED_BY_EXISTING_ZERO_SHOT"
    )
    assert by_id["GSE246381"]["terminal_status"] == "SEALED_EXCLUDED"
    assert by_id["GSE246381"]["outcome_exposure"] == "SEALED_NOT_READ"
    assert inventory["goal1_adjudication"]["evaluation_confirmation_status"] == (
        "NOT_ESTABLISHED"
    )
    assert inventory["goal1_adjudication"]["goal1_overall_status"] == (
        "IMPLEMENTED_NOT_DELIVERED"
    )
