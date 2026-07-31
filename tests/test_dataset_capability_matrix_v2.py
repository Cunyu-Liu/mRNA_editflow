from __future__ import annotations

import csv
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "data_registry/dataset_capability_matrix.csv"
REQUIRED_COLUMNS = {
    "dataset_id",
    "source_exact",
    "candidate_exact",
    "assay_matched",
    "endpoint_explicit",
    "replicate_noise",
    "edit_script_recoverable",
    "edit_script_ambiguity",
    "substitution_coverage",
    "indel_coverage",
    "multi_edit_coverage",
    "variable_length",
    "candidates_per_source",
    "independent_source_groups",
    "study_context_diversity",
    "library_ascertainment",
    "license",
    "historical_exposure",
    "allowed_tasks",
    "forbidden_claims",
}


def _rows() -> list[dict[str, str]]:
    with MATRIX.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_matrix_has_required_columns_and_current_candidates():
    rows = _rows()
    assert REQUIRED_COLUMNS <= set(rows[0])
    assert {
        "GSE114002",
        "GSE149487",
        "GSE246381",
        "GSE145046",
        "GSE217518",
        "GSE200304",
        "MPRAu_processed_ENCSR854RUF",
        "ENCSR854RUF_raw62",
    } <= {row["dataset_id"] for row in rows}
    assert all(all(row[column].strip() for column in REQUIRED_COLUMNS) for row in rows)


def test_exposure_and_encode_roles_are_fail_closed():
    by_id = {row["dataset_id"]: row for row in _rows()}
    assert by_id["GSE246381"]["historical_exposure"] == (
        "historically_exposed_retrospective_E4"
    )
    assert "no_training_labels" in by_id["GSE246381"]["forbidden_claims"]
    assert "deletion_subset" in by_id["MPRAu_processed_ENCSR854RUF"][
        "allowed_tasks"
    ]
    assert "observational_pretraining_candidate" in by_id["ENCSR854RUF_raw62"][
        "allowed_tasks"
    ]
    assert "intervention_evidence_from_inventory_alone" in by_id[
        "ENCSR854RUF_raw62"
    ]["forbidden_claims"]
    assert "automatic_primary_role" in by_id["ENCSR854RUF_raw62"][
        "forbidden_claims"
    ]
    assert "ENA_INSDC_free_unrestricted_access" in by_id[
        "ENCSR854RUF_raw62"
    ]["license"]
    assert "CC_BY" not in by_id["ENCSR854RUF_raw62"]["license"]
    assert "not_yet_untouched_E5" in by_id["GSE330741"]["forbidden_claims"]
    assert "no_source_paired_edit_claim" in by_id["GSE291719"][
        "forbidden_claims"
    ]


def test_hypothesis_matrix_has_all_hypotheses_and_insufficient_paths():
    text = (
        ROOT / "docs/data/hypothesis_data_requirement_matrix.md"
    ).read_text(encoding="utf-8")
    for index in range(1, 9):
        assert f"| H{index} " in text
    assert "No verified source-paired UTR insertion dataset" in text
    assert "does not identify a unique observed edit trajectory" in text


def test_new_external_candidates_remain_metadata_only_and_label_free():
    by_id = {row["dataset_id"]: row for row in _rows()}
    for accession in ("GSE330741", "GSE291719"):
        row = by_id[accession]
        assert "metadata_only" in row["historical_exposure"]
        assert "no_final_label_access_before_freeze" in row["forbidden_claims"]
        assert "no_training" in row["forbidden_claims"]

    registry = yaml.safe_load(
        (ROOT / "docs/execution/task_registry.yaml").read_text(encoding="utf-8")
    )
    d0_tasks = [task for task in registry["tasks"] if task["phase_id"] == "D0"]
    assert d0_tasks
    assert all("FINAL_LABEL_ACCESS" not in task["resource_labels"] for task in d0_tasks)

    active_script_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (ROOT / "scripts").rglob("*.py")
        if "archive" not in path.parts
    )
    assert "GSE330741" not in active_script_text
    assert "GSE291719" not in active_script_text
