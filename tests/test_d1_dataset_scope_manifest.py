from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCOPE = ROOT / "data_registry/d1_dataset_scope_manifest.yaml"
CAPABILITY = ROOT / "data_registry/dataset_capability_matrix.csv"
BUILD_CONFIG = ROOT / "configs/d1_build_20260729.json"


def _scope() -> dict:
    return yaml.safe_load(SCOPE.read_text(encoding="utf-8"))


def _rows() -> dict[str, dict]:
    rows = _scope()["datasets"]
    assert len(rows) == len({row["dataset_id"] for row in rows})
    return {row["dataset_id"]: row for row in rows}


def test_every_d0_candidate_has_exactly_one_d1_role() -> None:
    with CAPABILITY.open(newline="", encoding="utf-8") as handle:
        d0_ids = {row["dataset_id"] for row in csv.DictReader(handle)}
    rows = _rows()
    assert set(rows) == d0_ids
    assert len(rows) == _scope()["acceptance"]["d0_candidate_rows_expected"] == 12


def test_every_scope_row_has_claim_exposure_and_reason_contract() -> None:
    required = {
        "d1_role",
        "admission_status",
        "regions",
        "source_candidate_status",
        "label_reproduction_status",
        "exposure_grade",
        "license",
        "allowed_uses",
        "blocked_claims",
        "reason_codes",
    }
    for dataset_id, row in _rows().items():
        assert required <= set(row), dataset_id
        assert row["allowed_uses"], dataset_id
        assert row["blocked_claims"], dataset_id
        assert row["reason_codes"], dataset_id


def test_historically_exposed_and_unopened_roles_fail_closed() -> None:
    rows = _rows()
    exposed = rows["GSE246381"]
    assert exposed["exposure_grade"] == "E4_HISTORICALLY_EXPOSED_RETROSPECTIVE"
    assert "training_label_use" in exposed["blocked_claims"]
    assert "untouched_or_sealed_external" in exposed["blocked_claims"]

    for dataset_id in ("GSE330741", "GSE291719"):
        row = rows[dataset_id]
        assert row["admission_status"] == "EXCLUDED_NO_FINAL_LABEL_ACCESS"
        assert row["label_reproduction_status"] == (
            "FORBIDDEN_BEFORE_FUTURE_PREACCESS_FREEZE"
        )


def test_intervention_roles_require_exact_mapping_not_metadata_promises() -> None:
    rows = _rows()
    assert rows["GSE149487"]["admission_status"] == "BLOCKED"
    assert "MISSING_BARCODE_TO_SEQUENCE_MAP" in rows["GSE149487"]["reason_codes"]

    gse217518 = rows["GSE217518"]
    assert gse217518["d1_role"] == "PRIMARY_FIVE_AND_THREE_UTR_INTERVENTION"
    assert set(gse217518["regions"]) == {"five_utr", "three_utr"}
    assert "UNIQUE_REF_MUT_GROUP_REQUIRED" in gse217518["reason_codes"]

    mprau = rows["MPRAu_processed_ENCSR854RUF"]
    assert mprau["admission_status"] == (
        "BLOCKED_PENDING_FROZEN_GRCH37_RECONSTRUCTION"
    )
    assert "ALL_OR_NOTHING_ROUNDTRIP_GATE" in mprau["reason_codes"]


def test_encode_and_out_of_scope_data_cannot_be_intervention_evidence() -> None:
    rows = _rows()
    assert rows["ENCSR854RUF_raw62"]["d1_role"] == (
        "OBSERVATIONAL_PRETRAINING_CANDIDATE_ONLY"
    )
    assert "intervention_evidence" in rows["ENCSR854RUF_raw62"]["blocked_claims"]
    for dataset_id in ("GSE207584", "GSE173083"):
        assert rows[dataset_id]["regions"] == []
        assert rows[dataset_id]["admission_status"] == (
            "EXCLUDED_CURRENT_UTR_ONLY_PHASE"
        )


def test_dataset_role_selection_is_label_independent() -> None:
    scope = _scope()
    assert scope["candidate_final_labels_used_for_role_selection"] is False
    assert scope["acceptance"]["final_label_selection_allowed"] is False
    assert scope["acceptance"]["core_scientific_question_changed"] is False


def test_production_build_config_binds_exact_scope_contract_and_selection() -> None:
    config = json.loads(BUILD_CONFIG.read_text(encoding="utf-8"))
    selection = config["selection_policy"]
    scope_bytes = SCOPE.read_bytes()
    binding = selection["dataset_scope_manifest"]
    assert [row["dataset_id"] for row in config["datasets"]]
    assert {row["dataset_id"] for row in config["datasets"]} == set(_rows())
    assert len(config["datasets"]) == len(_rows()) == 12
    assert selection[
        "candidate_final_labels_used_for_dataset_role_selection"
    ] is False
    assert selection["goal_contract_sha256"] == _scope()[
        "goal_contract_sha256"
    ]
    assert binding["path"].endswith(
        "/data_registry/d1_dataset_scope_manifest.yaml"
    )
    assert binding["bytes"] == len(scope_bytes)
    assert binding["sha256"] == hashlib.sha256(scope_bytes).hexdigest()
