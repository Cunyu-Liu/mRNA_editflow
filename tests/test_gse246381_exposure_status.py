from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_gse246381_is_permanently_historically_exposed():
    contract = yaml.safe_load(
        (ROOT / "configs/utr_editflow_execution_policy.yaml").read_text(encoding="utf-8")
    )
    exposure = contract["gse246381"]
    assert exposure["historically_exposed"] is True
    assert exposure["role"] == "historically_exposed_retrospective_external_stress_test"
    assert exposure["evidence_grade"] == "E4"
    assert exposure["confirmatory_primary"] is False
    assert exposure["labels_allowed_for_new_training"] is False
    assert exposure["labels_allowed_for_new_hyperparameter_selection"] is False
    assert exposure["untouched_wording_allowed"] is False
