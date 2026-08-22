#!/usr/bin/env python3
"""Build the Route 2 V3.3.2 data, rights and exposure limitation table.

The table binds the frozen 14-study reporting inventory to current Route 2
rights declarations without treating public access, converter output policy or
historical license metadata as verified redistribution authority.  It reads no
canonical record, Development TEST outcome, final Evaluation outcome or sealed
payload.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_TABLE = (
    ROOT / "docs/paper/route2_v332_dataset_qualification_table_v1.csv"
)
DEFAULT_EVIDENCE_MANIFEST = (
    ROOT / "docs/paper/route2_v332_evidence_manifest_v1.json"
)
DEFAULT_TABLE = (
    ROOT / "docs/paper/route2_v332_data_rights_exposure_limitations_table_v1.csv"
)
DEFAULT_AUDIT = (
    ROOT / "audits/route_a_v3_route2_v332_data_rights_exposure_limitations_table_v1.json"
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

# A declaration is deliberately narrower than a license determination.  Every
# row remains human-review pending and not authorized for public release.
RIGHTS_DECLARATIONS: Mapping[str, Mapping[str, str]] = {
    "GSE200304": {
        "path": "configs/route_a_v3_gse200304_a1_qualification.json",
        "locator": (
            "paper_and_external_evidence.license_evidence: artifact-specific "
            "code license UNKNOWN_FAIL_CLOSED; this is not a dataset license"
        ),
        "value": "",
        "scope": "SOURCE_METADATA_NOT_DATASET_REDISTRIBUTION_AUTHORITY",
    },
    "GSE114002": {
        "path": "configs/route_a_v3_route2_gse114002_converter_v1.json",
        "locator": "output.public_redistribution_allowed=false; limitations include LICENSE_AND_REDISTRIBUTION_UNKNOWN_NOT_ASSERTED",
        "value": "false",
        "scope": "CONVERTER_OUTPUT_POLICY_NOT_LICENSE_VERIFICATION",
    },
    "GSE149487": {
        "path": "configs/route_a_v3_route2_gse149487_converter_v1.json",
        "locator": "output.public_redistribution_allowed=false; LICENSE_AND_PUBLIC_REDISTRIBUTION_NOT_CLOSED",
        "value": "false",
        "scope": "CONVERTER_OUTPUT_POLICY_NOT_LICENSE_VERIFICATION",
    },
    "GSE217518": {
        "path": "configs/route_a_v3_route2_gse217518_converter_v1.json",
        "locator": "output.public_redistribution_allowed=true; no study-bound human-verified license record in the current 14-study inventory",
        "value": "true",
        "scope": "CONVERTER_OUTPUT_POLICY_NOT_LICENSE_VERIFICATION",
    },
    "ENCSR854RUF": {
        "path": "configs/route_a_v3_route2_encsr854ruf_converter_v1.json",
        "locator": "output.public_redistribution_allowed=false; asset-level reuse basis not closed",
        "value": "false",
        "scope": "CONVERTER_OUTPUT_POLICY_NOT_LICENSE_VERIFICATION",
    },
    "GSE186455": {
        "path": "configs/route_a_v3_route2_gse186455_converter_v1.json",
        "locator": "output.public_redistribution_allowed=false",
        "value": "false",
        "scope": "CONVERTER_OUTPUT_POLICY_NOT_LICENSE_VERIFICATION",
    },
    "GSE256185": {
        "path": "configs/route_a_v3_gse256185_aggregate_row_level_qualification_preflight_v1.json",
        "locator": "raw_asset_redistribution=NOT_AUTHORIZED; derived_row_level_redistribution=NOT_AUTHORIZED",
        "value": "false",
        "scope": "FROZEN_ASSET_PREFLIGHT_POLICY_NOT_GENERAL_LICENSE_VERIFICATION",
    },
    "GSE269595": {
        "path": "configs/route_a_v3_route2_gse269595_converter_v1.json",
        "locator": "output.public_redistribution_allowed=false",
        "value": "false",
        "scope": "CONVERTER_OUTPUT_POLICY_NOT_LICENSE_VERIFICATION",
    },
    "GSE232572": {
        "path": "configs/route_a_v3_route2_gse232572_converter_v1.json",
        "locator": "output.public_redistribution_allowed=false",
        "value": "false",
        "scope": "CONVERTER_OUTPUT_POLICY_NOT_LICENSE_VERIFICATION",
    },
    "GSE207584": {
        "path": "configs/route_a_v3_gse207584_aggregate_dense_family_qualification_preflight_v1.json",
        "locator": "rights_policy.member_payload_redistribution_allowed=false; aggregate reporting permission UNKNOWN_NOT_ASSERTED",
        "value": "false",
        "scope": "MEMBER_PAYLOAD_PREFLIGHT_POLICY_NOT_LICENSE_VERIFICATION",
    },
    "GSE261709": {
        "path": "configs/route_a_v3_gse261709_dec024_aggregate_row_level_a1_qualification_preflight_v1.json",
        "locator": "processed asset license_and_reuse_rights_status=UNKNOWN_NOT_ASSERTED",
        "value": "",
        "scope": "PREFLIGHT_RIGHTS_STATUS_NOT_LICENSE_VERIFICATION",
    },
}

NO_BOUND_RECORD = {
    "path": "",
    "locator": "NO_CURRENT_STUDY_BOUND_RIGHTS_RECORD_IN_ROUTE2_14_STUDY_INVENTORY",
    "value": "",
    "scope": "NO_CURRENT_STUDY_BOUND_RIGHTS_DECLARATION",
}

FIELDNAMES = (
    "limitation_row_id",
    "study_unit_id",
    "current_analysis_role_v332",
    "qualification_stratum",
    "terminal_conversion_status",
    "canonical_records",
    "outcome_exposure",
    "data_and_endpoint_limitation",
    "exposure_limitation",
    "rights_evidence_path",
    "rights_evidence_locator",
    "declared_public_redistribution_allowed",
    "declaration_scope",
    "license_verification_status",
    "public_release_authorized",
    "project_confidentiality",
    "human_content_verification_status",
    "generalization_boundary",
    "publication_boundary",
    "development_test_read",
    "new_final_evaluation_read",
    "guided_xeditflow_run",
)


class LimitationTableInputError(RuntimeError):
    """The frozen reporting inputs no longer match the declared boundary."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LimitationTableInputError(message)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _validate_rights_declarations() -> None:
    for study_id, declaration in RIGHTS_DECLARATIONS.items():
        evidence_path = ROOT / declaration["path"]
        _require(evidence_path.is_file(), f"missing rights evidence for {study_id}")
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        value = declaration["value"]
        if study_id in {
            "GSE114002",
            "GSE149487",
            "GSE217518",
            "ENCSR854RUF",
            "GSE186455",
            "GSE269595",
            "GSE232572",
        }:
            observed = payload["output"]["public_redistribution_allowed"]
            _require(str(observed).lower() == value, f"{study_id} converter declaration changed")
        elif study_id == "GSE256185":
            rights = payload["frozen_external_gate_facts"]["rights"]
            _require(
                rights["raw_asset_redistribution"] == "NOT_AUTHORIZED"
                and rights["derived_row_level_redistribution"] == "NOT_AUTHORIZED",
                "GSE256185 rights boundary changed",
            )
        elif study_id == "GSE207584":
            policy = payload["rights_policy"]
            _require(
                policy["member_payload_redistribution_allowed"] is False
                and policy["aggregate_derived_reporting_allowed"] == "UNKNOWN_NOT_ASSERTED",
                "GSE207584 rights boundary changed",
            )
        elif study_id == "GSE261709":
            status = payload["processed_asset_contract"][
                "official_processed_asset_manifest"
            ]["license_and_reuse_rights_status"]
            _require(status == "UNKNOWN_NOT_ASSERTED", "GSE261709 rights boundary changed")
        elif study_id == "GSE200304":
            status = payload["paper_and_external_evidence"]["license_evidence"][
                "artifact_specific_code_license_status"
            ]
            _require(status == "UNKNOWN_FAIL_CLOSED", "GSE200304 license boundary changed")


def _exposure_limitation(row: Mapping[str, str]) -> str:
    role = row["current_analysis_role_v332"]
    if role == "DEVELOPMENT":
        return "DEVELOPMENT_EVIDENCE_ONLY_NOT_FINAL_CONFIRMATION"
    if role.startswith("HISTORICAL_OUTCOME_EXPOSED"):
        return "OUTCOME_EXPOSED_HISTORICAL_DIAGNOSTIC_NOT_FINAL_CONFIRMATION"
    if role == "CONVERSION_FAILURE_ONLY_OUTCOME_NOT_READ":
        return "CONVERSION_FAILURE_ONLY_OUTCOME_NOT_READ"
    if role == "SEALED_EXCLUDED":
        return "SEALED_PAYLOAD_AND_OUTCOME_NOT_READ"
    return "NO_RECORD_LEVEL_FINAL_EVALUATION_EVIDENCE"


def _generalization_boundary(row: Mapping[str, str]) -> str:
    if row["qualification_stratum"] == "QUALIFIED":
        return "ONE_QUALIFIED_A1_STUDY_ONLY_NO_TRUE_A2_OR_EXTERNAL_GENERALIZATION"
    if int(row["canonical_records"]) > 0:
        return "DESCRIPTIVE_OR_DEVELOPMENT_USE_ONLY_NO_QUALIFIED_STUDY_CREDIT"
    return "NO_CANONICAL_RECORD_LEVEL_PERFORMANCE_OR_GENERALIZATION_CLAIM"


def _table_rows(dataset_rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(dataset_rows, start=1):
        study_id = source["study_unit_id"]
        declaration = RIGHTS_DECLARATIONS.get(study_id, NO_BOUND_RECORD)
        rows.append(
            {
                "limitation_row_id": f"DRX-{index:02d}",
                "study_unit_id": study_id,
                "current_analysis_role_v332": source["current_analysis_role_v332"],
                "qualification_stratum": source["qualification_stratum"],
                "terminal_conversion_status": source["terminal_conversion_status"],
                "canonical_records": source["canonical_records"],
                "outcome_exposure": source["outcome_exposure"],
                "data_and_endpoint_limitation": source["primary_limitation"],
                "exposure_limitation": _exposure_limitation(source),
                "rights_evidence_path": declaration["path"],
                "rights_evidence_locator": declaration["locator"],
                "declared_public_redistribution_allowed": declaration["value"],
                "declaration_scope": declaration["scope"],
                "license_verification_status": (
                    "HUMAN_REVIEW_PENDING_NO_14_STUDY_VERIFIED_LICENSE_REGISTRY"
                ),
                "public_release_authorized": "false",
                "project_confidentiality": "LOCAL_UNPUBLISHED_PROJECT_MATERIAL",
                "human_content_verification_status": "PENDING",
                "generalization_boundary": _generalization_boundary(source),
                "publication_boundary": (
                    "REPORT_AGGREGATES_AND_LOCATORS_ONLY_DO_NOT_RELEASE_STUDY_PAYLOADS"
                ),
                "development_test_read": "false",
                "new_final_evaluation_read": "false",
                "guided_xeditflow_run": "false",
            }
        )
    return rows


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    ids = [row["study_unit_id"] for row in rows]
    _require(len(ids) == len(set(ids)) == 14, "table must contain 14 unique studies")
    _require(set(ids) == EXPECTED_STUDY_IDS, "study set changed")
    values = [row["declared_public_redistribution_allowed"] for row in rows]
    _require(values.count("true") == 1, "expected one converter-level true declaration")
    _require(values.count("false") == 8, "expected eight fail-closed declarations")
    _require(values.count("") == 5, "expected five rows without a boolean declaration")
    _require(
        all(row["public_release_authorized"] == "false" for row in rows),
        "no study payload is currently authorized for public release",
    )
    _require(
        all(row["license_verification_status"].startswith("HUMAN_REVIEW_PENDING") for row in rows),
        "license verification must remain human-review pending",
    )
    _require(
        all(
            row["development_test_read"] == "false"
            and row["new_final_evaluation_read"] == "false"
            and row["guided_xeditflow_run"] == "false"
            for row in rows
        ),
        "protected outcome boundary changed",
    )


def build_table(
    *,
    dataset_table_path: Path = DEFAULT_DATASET_TABLE,
    evidence_manifest_path: Path = DEFAULT_EVIDENCE_MANIFEST,
    table_path: Path = DEFAULT_TABLE,
    audit_path: Path = DEFAULT_AUDIT,
    overwrite: bool = False,
) -> dict[str, Any]:
    dataset_table_path = dataset_table_path.resolve()
    evidence_manifest_path = evidence_manifest_path.resolve()
    table_path = table_path.resolve()
    audit_path = audit_path.resolve()
    for path in (table_path, audit_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing limitation artifact: {path}")

    dataset_rows = _read_rows(dataset_table_path)
    evidence = json.loads(evidence_manifest_path.read_text(encoding="utf-8"))
    _require(
        evidence["confidentiality"] == "LOCAL_UNPUBLISHED_PROJECT_MATERIAL"
        and evidence["human_verification_required"] is True
        and evidence["submission_ready"] is False,
        "project confidentiality or human-verification boundary changed",
    )
    _validate_rights_declarations()
    rows = _table_rows(dataset_rows)
    _validate_rows(rows)

    table_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    audit = {
        "schema_version": "route_a_v3_route2_v332_data_rights_exposure_limitations.v1",
        "status": "DATA_RIGHTS_EXPOSURE_LIMITATIONS_REPORTED_HUMAN_REVIEW_PENDING",
        "authority": {
            "scientific_contract": (
                "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/提示词/"
                "mrna 数据gate转向后的合同.md"
            ),
            "execution_protocol": (
                "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/提示词/"
                "mrna V3.3.2 执行提示词.md"
            ),
            "dataset_qualification_table": _display_path(dataset_table_path),
            "evidence_manifest": _display_path(evidence_manifest_path),
        },
        "table_path": _display_path(table_path),
        "row_count": len(rows),
        "study_count": len(rows),
        "declared_redistribution_value_counts": {
            "true": 1,
            "false": 8,
            "not_declared_as_boolean": 5,
        },
        "converter_or_preflight_declaration_is_license_verification": False,
        "study_bound_human_verified_license_registry_present": False,
        "license_human_review_pending_count": len(rows),
        "public_release_authorized_count": 0,
        "aggregate_and_locator_reporting_only": True,
        "data_rights_exposure_limitations_complete": True,
        "minimum_package_complete": False,
        "submission_ready": False,
        "claim_boundary": (
            "Public accessibility and converter policy do not establish dataset redistribution authority. "
            "Until accountable human review binds study-specific terms, report aggregates and source locators "
            "but do not release study payloads or claim an open-data package."
        ),
        "protected_outcomes": {
            "development_test_read": False,
            "new_final_evaluation_read": False,
            "emtab10902_outcome_read": False,
            "sealed_gse246381_read": False,
            "guided_xeditflow_run": False,
        },
        "new_training_attempt_created": False,
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-table", type=Path, default=DEFAULT_DATASET_TABLE)
    parser.add_argument("--evidence-manifest", type=Path, default=DEFAULT_EVIDENCE_MANIFEST)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    audit = build_table(
        dataset_table_path=args.dataset_table,
        evidence_manifest_path=args.evidence_manifest,
        table_path=args.table,
        audit_path=args.audit,
        overwrite=args.overwrite,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
