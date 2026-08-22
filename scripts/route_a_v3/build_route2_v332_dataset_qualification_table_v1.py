#!/usr/bin/env python3
"""Build the Route 2 V3.3.2 dataset qualification/development table.

The frozen 14-study inventory remains the source of terminal conversion facts.
V3.3.2 supersedes its stale GSE232572 EVALUATION label because that study's
outcomes were already exposed.  This builder does not open canonical rows,
Development TEST, a new final Evaluation outcome, or sealed data.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = (
    ROOT / "configs/route_a_v3_route2_14_study_final_inventory_v1.json"
)
DEFAULT_TABLE = ROOT / "docs/paper/route2_v332_dataset_qualification_table_v1.csv"
DEFAULT_AUDIT = (
    ROOT / "audits/route_a_v3_route2_v332_dataset_qualification_table_v1.json"
)

EXPECTED_STUDY_IDS = {
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

HISTORICAL_ROLE = (
    "HISTORICAL_OUTCOME_EXPOSED_TRANSFER_DIAGNOSTIC_NOT_FINAL_CONFIRMATION"
)
EMTAB_ROLE = "CONVERSION_FAILURE_ONLY_OUTCOME_NOT_READ"

QUALIFICATION_STRATA = {
    "MATERIALIZED_QUALIFIED": "QUALIFIED",
    "MATERIALIZED_DEVELOPMENT_RELAXED": "DEVELOPMENT_RELAXED_NOT_QUALIFIED",
    "MATERIALIZED_DEVELOPMENT_LISTWISE": "DEVELOPMENT_LISTWISE_NOT_QUALIFIED",
    "MATERIALIZED_EVALUATION_OUTCOME_EXPOSED": (
        "HISTORICAL_OUTCOME_EXPOSED_NOT_FINAL_CONFIRMATION"
    ),
    "UNCONVERTIBLE_FOR_ROUTE2_V1": "UNCONVERTIBLE",
    "AUXILIARY_SUMMARY_ONLY": "AUXILIARY_ONLY",
    "SEALED_EXCLUDED": "SEALED_EXCLUDED",
}

FIELDNAMES = (
    "study_unit_id",
    "inventory_use_role",
    "current_analysis_role_v332",
    "qualification_stratum",
    "terminal_conversion_status",
    "canonical_records",
    "development_canonical_records",
    "historical_transfer_canonical_records",
    "final_evaluation_unexposed_canonical_records",
    "qualified_canonical_credit_records",
    "ordinary_study_credit",
    "a1_study_credit",
    "true_a2_study_credit",
    "outcome_exposure",
    "primary_limitation",
)


class DatasetTableInputError(RuntimeError):
    """The frozen inventory does not match its declared V3.3.2 shape."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetTableInputError(message)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _current_role(study: Mapping[str, Any]) -> str:
    study_id = study["study_unit_id"]
    if study_id == "GSE232572":
        return HISTORICAL_ROLE
    if study_id == "E-MTAB-10902":
        return EMTAB_ROLE
    return str(study["use_role"])


def _validate_inventory(inventory: Mapping[str, Any]) -> None:
    studies = inventory.get("studies", [])
    ids = [study["study_unit_id"] for study in studies]
    _require(len(ids) == len(set(ids)) == 14, "inventory must contain 14 unique studies")
    _require(set(ids) == EXPECTED_STUDY_IDS, "inventory study set changed")

    by_id = {study["study_unit_id"]: study for study in studies}
    _require(
        by_id["GSE232572"]["use_role"] == "EVALUATION"
        and by_id["GSE232572"]["outcome_exposure"]
        == "OUTCOME_EXPOSED_BY_EXISTING_ZERO_SHOT",
        "GSE232572 frozen role/exposure no longer matches the declared override",
    )
    _require(
        by_id["E-MTAB-10902"]["terminal_status"]
        == "UNCONVERTIBLE_FOR_ROUTE2_V1"
        and by_id["E-MTAB-10902"]["outcome_exposure"] == "OUTCOME_NOT_READ",
        "E-MTAB-10902 conversion-failure/outcome boundary changed",
    )
    _require(
        by_id["GSE246381"]["terminal_status"] == "SEALED_EXCLUDED"
        and by_id["GSE246381"]["outcome_exposure"] == "SEALED_NOT_READ",
        "sealed-study boundary changed",
    )
    _require(
        set(study["terminal_status"] for study in studies) <= set(QUALIFICATION_STRATA),
        "inventory contains an unmapped terminal status",
    )


def _table_rows(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for study in inventory["studies"]:
        study_id = str(study["study_unit_id"])
        current_role = _current_role(study)
        canonical_records = int(study["canonical_records"])
        is_qualified = study["terminal_status"] == "MATERIALIZED_QUALIFIED"
        rows.append(
            {
                "study_unit_id": study_id,
                "inventory_use_role": study["use_role"],
                "current_analysis_role_v332": current_role,
                "qualification_stratum": QUALIFICATION_STRATA[
                    study["terminal_status"]
                ],
                "terminal_conversion_status": study["terminal_status"],
                "canonical_records": canonical_records,
                "development_canonical_records": (
                    canonical_records if current_role == "DEVELOPMENT" else 0
                ),
                "historical_transfer_canonical_records": (
                    canonical_records if current_role == HISTORICAL_ROLE else 0
                ),
                "final_evaluation_unexposed_canonical_records": 0,
                "qualified_canonical_credit_records": (
                    canonical_records if is_qualified else 0
                ),
                "ordinary_study_credit": 1 if is_qualified else 0,
                "a1_study_credit": 1 if is_qualified else 0,
                "true_a2_study_credit": 0,
                "outcome_exposure": study["outcome_exposure"],
                "primary_limitation": study["reason"],
            }
        )
    return rows


def _validate_rows(rows: Sequence[Mapping[str, Any]], inventory: Mapping[str, Any]) -> None:
    totals = inventory["totals"]
    _require(
        sum(int(row["development_canonical_records"]) for row in rows)
        == totals["development_canonical_records"]
        == 126165,
        "Development record conservation failed",
    )
    _require(
        sum(int(row["historical_transfer_canonical_records"]) for row in rows)
        == totals["evaluation_canonical_records"]
        == 8068,
        "historical-transfer record conservation failed",
    )
    _require(
        sum(int(row["final_evaluation_unexposed_canonical_records"]) for row in rows)
        == totals["evaluation_unexposed_canonical_records"]
        == 0,
        "unexposed final-Evaluation record conservation failed",
    )
    _require(
        sum(int(row["qualified_canonical_credit_records"]) for row in rows)
        == totals["qualified_canonical_records"]
        == 6547,
        "qualified canonical-credit conservation failed",
    )
    _require(
        sum(int(row["ordinary_study_credit"]) for row in rows) == 1
        and sum(int(row["a1_study_credit"]) for row in rows) == 1
        and sum(int(row["true_a2_study_credit"]) for row in rows) == 0,
        "qualified study-credit totals changed",
    )


def build_table(
    *,
    inventory_path: Path = DEFAULT_INVENTORY,
    table_path: Path = DEFAULT_TABLE,
    audit_path: Path = DEFAULT_AUDIT,
    overwrite: bool = False,
) -> dict[str, Any]:
    inventory_path = inventory_path.resolve()
    table_path = table_path.resolve()
    audit_path = audit_path.resolve()
    for path in (table_path, audit_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing dataset table artifact: {path}")

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    _validate_inventory(inventory)
    rows = _table_rows(inventory)
    _validate_rows(rows, inventory)

    table_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    audit = {
        "schema_version": "route_a_v3_route2_v332_dataset_qualification_table.v1",
        "status": "DATASET_QUALIFICATION_DEVELOPMENT_TABLE_RENDERED",
        "authority": {
            "scientific_contract": (
                "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/提示词/"
                "mrna 数据gate转向后的合同.md"
            ),
            "execution_protocol": (
                "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/提示词/"
                "mrna V3.3.2 执行提示词.md"
            ),
            "inventory": _display_path(inventory_path),
        },
        "table_path": _display_path(table_path),
        "study_count": len(rows),
        "role_counts": {
            role: sum(row["current_analysis_role_v332"] == role for row in rows)
            for role in sorted({str(row["current_analysis_role_v332"]) for row in rows})
        },
        "record_totals": {
            "development_canonical_records": 126165,
            "historical_transfer_canonical_records": 8068,
            "final_evaluation_unexposed_canonical_records": 0,
            "qualified_canonical_credit_records": 6547,
            "generated_candidate_canonical_credit": 0,
        },
        "qualified_credit": {
            "ordinary_studies": 1,
            "a1_studies": 1,
            "true_a2_studies": 0,
            "only_credited_study_unit_id": "GSE200304",
        },
        "authority_overrides": [
            {
                "study_unit_id": "GSE232572",
                "frozen_inventory_role": "EVALUATION",
                "current_v332_role": HISTORICAL_ROLE,
                "reason": "Outcome exposure predates V3.3.2; the study cannot be re-frozen as final confirmation.",
            }
        ],
        "protected_outcomes": {
            "development_test_read": False,
            "new_final_evaluation_read": False,
            "emtab10902_outcome_read": False,
            "sealed_gse246381_read": False,
            "guided_xeditflow_run": False,
        },
        "claim_boundary": (
            "Development-relaxed/listwise rows support Development benchmarking but add no qualified credit; "
            "historical outcome-exposed rows add no final-Evaluation credit; zero-record and sealed studies remain explicit."
        ),
        "new_training_attempt_created": False,
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    audit = build_table(
        inventory_path=args.inventory,
        table_path=args.table,
        audit_path=args.audit,
        overwrite=args.overwrite,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
