#!/usr/bin/env python3
"""Build the V3.3.2 study/provider rights evidence table.

This builder binds the existing 14-study rights table to a frozen snapshot of
official repository/provider policies.  It does not access accession content,
canonical records, Development TEST, new final Evaluation outcomes, sealed
payloads or generated candidates.  Provider access/use policy is deliberately
kept separate from project payload-redistribution authorization.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RIGHTS_TABLE = (
    ROOT / "docs/paper/route2_v332_data_rights_exposure_limitations_table_v1.csv"
)
DEFAULT_SOURCE_SNAPSHOT = (
    ROOT
    / "audits/route_a_v3_route2_v332_rights_provider_official_source_snapshot_v1.json"
)
DEFAULT_TABLE = (
    ROOT / "docs/paper/route2_v332_study_rights_provider_evidence_table_v1.csv"
)
DEFAULT_AUDIT = (
    ROOT / "audits/route_a_v3_route2_v332_study_rights_provider_evidence_table_v1.json"
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

FIELDNAMES = (
    "provider_evidence_row_id",
    "study_unit_id",
    "current_analysis_role_v332",
    "outcome_exposure",
    "provider_family",
    "repository_name",
    "accession_landing_url",
    "accession_resolution_status",
    "provider_policy_url",
    "provider_policy_version_or_date",
    "official_provider_policy_bound",
    "provider_access_supported",
    "provider_analysis_and_publication_use_supported",
    "provider_distribution_policy",
    "provider_policy_exception",
    "study_specific_license_record_present",
    "study_specific_license_value",
    "nature_data_access_route",
    "dataset_citation_action",
    "target_journal_status",
    "fair_findable",
    "fair_accessible_metadata",
    "fair_interoperable_metadata_assessed",
    "fair_reusable_license_complete",
    "prior_declared_public_redistribution_allowed",
    "prior_declaration_scope",
    "project_payload_redistribution_authorized",
    "accountable_human_review_status",
    "accountable_human_review_action",
    "publication_boundary",
    "source_snapshot_path",
    "development_test_read",
    "new_final_evaluation_read",
    "emtab10902_outcome_read",
    "sealed_gse246381_read",
    "guided_xeditflow_run",
)


class ProviderEvidenceInputError(RuntimeError):
    """The frozen rights/provider inputs no longer match the declared boundary."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProviderEvidenceInputError(message)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _provider_fields(
    study_id: str, providers: Mapping[str, Mapping[str, Any]]
) -> dict[str, str]:
    if study_id.startswith("GSE"):
        policy = providers["NCBI_GEO"]
        return {
            "provider_family": "NCBI_GEO",
            "repository_name": policy["repository_name"],
            "accession_landing_url": (
                "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=" + study_id
            ),
            "accession_resolution_status": (
                "OFFICIAL_NCBI_GDS_BATCH_ACCESSION_RETURNED"
            ),
            "provider_policy_url": policy["policy_url"],
            "provider_policy_version_or_date": policy["policy_version_or_date"],
            "provider_distribution_policy": (
                "NCBI_NO_RESTRICTION_WITH_SUBMITTER_IP_EXCEPTION"
            ),
            "provider_policy_exception": policy["distribution_exception"],
            "study_specific_license_record_present": "false",
            "study_specific_license_value": "",
            "nature_data_access_route": (
                "REUSED_PUBLIC_SOURCE_PROVIDER_IP_EXCEPTION"
            ),
            "dataset_citation_action": (
                "CITE_GEO_ACCESSION_AND_PRIMARY_DATASET_PUBLICATION"
            ),
            "accountable_human_review_action": (
                "VERIFY_SUBMITTER_OR_PUBLISHER_RIGHTS_AND_FILE_LEVEL_TERMS"
            ),
        }
    if study_id == "ENCSR854RUF":
        policy = providers["ENCODE"]
        return {
            "provider_family": "ENCODE",
            "repository_name": policy["repository_name"],
            "accession_landing_url": policy["effective_url"],
            "accession_resolution_status": (
                "OFFICIAL_LANDING_HTTP_200_REDIRECT_PUBLICATION_DATA"
            ),
            "provider_policy_url": policy["policy_url"],
            "provider_policy_version_or_date": policy["policy_version_or_date"],
            "provider_distribution_policy": (
                "UNRESTRICTED_USE_POLICY_PROJECT_REPACKAGING_NOT_SEPARATELY_ADJUDICATED"
            ),
            "provider_policy_exception": policy["distribution_exception"],
            "study_specific_license_record_present": "false",
            "study_specific_license_value": "",
            "nature_data_access_route": (
                "REUSED_PUBLIC_SOURCE_PROVIDER_USE_POLICY_LICENSE_REVIEW_PENDING"
            ),
            "dataset_citation_action": (
                "CITE_ENCODE_ACCESSION_AND_ORIGINAL_DATASET_SOURCE"
            ),
            "accountable_human_review_action": (
                "VERIFY_EXACT_ASSET_TERMS_AND_REPACKAGING_SCOPE"
            ),
        }
    _require(study_id == "E-MTAB-10902", f"unknown provider for {study_id}")
    policy = providers["EMBL_EBI_BIOSTUDIES_ARRAYEXPRESS"]
    return {
        "provider_family": "EMBL_EBI_BIOSTUDIES_ARRAYEXPRESS",
        "repository_name": policy["repository_name"],
        "accession_landing_url": (
            "https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-10902"
        ),
        "accession_resolution_status": (
            "OFFICIAL_BIOSTUDIES_API_ACCESSION_RETURNED_LICENSE_FIELD_ABSENT"
        ),
        "provider_policy_url": policy["submission_and_license_policy_url"],
        "provider_policy_version_or_date": policy["policy_version_or_date"],
        "provider_distribution_policy": (
            "NEW_DATASET_CC0_SCOPE_NOT_BOUND_TO_MIGRATED_E_MTAB_RECORD"
        ),
        "provider_policy_exception": policy["distribution_exception"],
        "study_specific_license_record_present": "false",
        "study_specific_license_value": "",
        "nature_data_access_route": (
            "REUSED_PUBLIC_SOURCE_MIGRATED_RECORD_LICENSE_UNKNOWN"
        ),
        "dataset_citation_action": (
            "CITE_BIOSTUDIES_ARRAYEXPRESS_ACCESSION_AND_PRIMARY_DATASET_PUBLICATION"
        ),
        "accountable_human_review_action": (
            "VERIFY_MIGRATED_RECORD_FILE_LEVEL_LICENSE_AND_REUSE_TERMS"
        ),
    }


def _validate_snapshot(snapshot: Mapping[str, Any]) -> None:
    _require(
        snapshot["collection_scope"]
        == "OFFICIAL_PROVIDER_POLICY_AND_ACCESSION_RESOLUTION_ONLY_NO_STUDY_OUTCOME_CONTENT_RETAINED",
        "official source snapshot scope changed",
    )
    providers = snapshot["providers"]
    _require(set(providers) == {"NCBI_GEO", "ENCODE", "EMBL_EBI_BIOSTUDIES_ARRAYEXPRESS"}, "provider set changed")
    _require(
        set(providers["NCBI_GEO"]["target_accessions_returned"])
        == {study_id for study_id in EXPECTED_STUDY_IDS if study_id.startswith("GSE")},
        "NCBI accession resolution set changed",
    )
    _require(providers["ENCODE"]["http_status"] == 200, "ENCODE landing no longer resolved")
    ebi = providers["EMBL_EBI_BIOSTUDIES_ARRAYEXPRESS"]
    _require(ebi["api_accession_returned"] is True, "E-MTAB accession not returned")
    _require(
        ebi["accession_specific_license_field_count"] == 0
        and ebi["accession_specific_release_field_count"] == 0,
        "E-MTAB selected rights metadata changed",
    )
    _require(
        all(value is False for value in snapshot["protected_outcomes"].values()),
        "protected outcome boundary changed",
    )


def _table_rows(
    rights_rows: Sequence[Mapping[str, str]],
    snapshot: Mapping[str, Any],
    snapshot_path: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    providers = snapshot["providers"]
    for index, source in enumerate(rights_rows, start=1):
        provider = _provider_fields(source["study_unit_id"], providers)
        rows.append(
            {
                "provider_evidence_row_id": f"RPE-{index:02d}",
                "study_unit_id": source["study_unit_id"],
                "current_analysis_role_v332": source["current_analysis_role_v332"],
                "outcome_exposure": source["outcome_exposure"],
                **provider,
                "official_provider_policy_bound": "true",
                "provider_access_supported": "true",
                "provider_analysis_and_publication_use_supported": "true",
                "target_journal_status": snapshot["target_journal_status"],
                "fair_findable": "true",
                "fair_accessible_metadata": "true",
                "fair_interoperable_metadata_assessed": "false",
                "fair_reusable_license_complete": "false",
                "prior_declared_public_redistribution_allowed": source[
                    "declared_public_redistribution_allowed"
                ],
                "prior_declaration_scope": source["declaration_scope"],
                "project_payload_redistribution_authorized": "false",
                "accountable_human_review_status": "PENDING",
                "publication_boundary": (
                    "CITE_ACCESSION_AND_REPORT_AGGREGATES_DO_NOT_REDISTRIBUTE_PROJECT_STUDY_PAYLOAD"
                ),
                "source_snapshot_path": _display_path(snapshot_path),
                "development_test_read": "false",
                "new_final_evaluation_read": "false",
                "emtab10902_outcome_read": "false",
                "sealed_gse246381_read": "false",
                "guided_xeditflow_run": "false",
            }
        )
    return rows


def _validate_rows(rows: Sequence[Mapping[str, str]]) -> None:
    ids = [row["study_unit_id"] for row in rows]
    _require(len(ids) == len(set(ids)) == 14, "table must contain 14 unique studies")
    _require(set(ids) == EXPECTED_STUDY_IDS, "study set changed")
    _require(
        sum(row["provider_family"] == "NCBI_GEO" for row in rows) == 12,
        "expected 12 GEO rows",
    )
    _require(
        sum(row["provider_family"] == "ENCODE" for row in rows) == 1,
        "expected one ENCODE row",
    )
    _require(
        sum(row["provider_family"] == "EMBL_EBI_BIOSTUDIES_ARRAYEXPRESS" for row in rows) == 1,
        "expected one BioStudies row",
    )
    _require(
        all(row["official_provider_policy_bound"] == "true" for row in rows),
        "every row must bind an official provider policy",
    )
    _require(
        all(row["provider_analysis_and_publication_use_supported"] == "true" for row in rows),
        "every accession must retain an analysis/citation route",
    )
    _require(
        all(row["study_specific_license_record_present"] == "false" for row in rows),
        "no accession-specific license record is established",
    )
    _require(
        all(row["project_payload_redistribution_authorized"] == "false" for row in rows),
        "provider policy must not authorize project payload redistribution",
    )
    _require(
        all(row["accountable_human_review_status"] == "PENDING" for row in rows),
        "rights review must remain human pending",
    )
    _require(
        all(
            row["development_test_read"] == "false"
            and row["new_final_evaluation_read"] == "false"
            and row["emtab10902_outcome_read"] == "false"
            and row["sealed_gse246381_read"] == "false"
            and row["guided_xeditflow_run"] == "false"
            for row in rows
        ),
        "protected outcomes must remain closed",
    )


def build_table(
    *,
    rights_table_path: Path = DEFAULT_RIGHTS_TABLE,
    source_snapshot_path: Path = DEFAULT_SOURCE_SNAPSHOT,
    table_path: Path = DEFAULT_TABLE,
    audit_path: Path = DEFAULT_AUDIT,
    overwrite: bool = False,
) -> dict[str, Any]:
    for path in (table_path, audit_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")

    rights_rows = _read_rows(rights_table_path)
    snapshot = json.loads(source_snapshot_path.read_text(encoding="utf-8"))
    _validate_snapshot(snapshot)
    rows = _table_rows(rights_rows, snapshot, source_snapshot_path)
    _validate_rows(rows)

    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    access_routes: dict[str, int] = {}
    for row in rows:
        route = row["nature_data_access_route"]
        access_routes[route] = access_routes.get(route, 0) + 1
    prior_values = [row["prior_declared_public_redistribution_allowed"] for row in rows]
    audit: dict[str, Any] = {
        "schema_version": "route_a_v3_route2_v332_study_rights_provider_evidence_table.v1",
        "status": "OFFICIAL_PROVIDER_POLICIES_BOUND_ANALYSIS_AND_CITATION_ROUTE_SUPPORTED_PROJECT_REDISTRIBUTION_NOT_AUTHORIZED_HUMAN_REVIEW_PENDING",
        "authority": {
            "rights_table": _display_path(rights_table_path),
            "official_source_snapshot": _display_path(source_snapshot_path),
            "scientific_contract": "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/提示词/mrna 数据gate转向后的合同.md",
            "execution_protocol": "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/提示词/mrna V3.3.2 执行提示词.md",
        },
        "row_count": len(rows),
        "provider_counts": {
            "NCBI_GEO": 12,
            "ENCODE": 1,
            "EMBL_EBI_BIOSTUDIES_ARRAYEXPRESS": 1,
        },
        "access_route_counts": access_routes,
        "official_provider_policy_bound_count": 14,
        "repository_accession_resolution_count": 14,
        "analysis_and_publication_use_route_supported_count": 14,
        "dataset_citation_action_declared_count": 14,
        "study_specific_license_record_present_count": 0,
        "project_payload_redistribution_authorized_count": 0,
        "accountable_human_review_pending_count": 14,
        "prior_converter_declaration_counts": {
            "true": prior_values.count("true"),
            "false": prior_values.count("false"),
            "not_declared_as_boolean": prior_values.count(""),
        },
        "fair_evidence_counts": {
            "findable": 14,
            "accessible_metadata": 14,
            "interoperable_metadata_assessed": 0,
            "reusable_license_complete": 0,
        },
        "target_journal_status": snapshot["target_journal_status"],
        "provider_policy_is_study_specific_license": False,
        "public_release_authorized": False,
        "human_content_and_rights_verification_complete": False,
        "minimum_package_complete": False,
        "submission_ready": False,
        "table_path": _display_path(table_path),
        "protected_outcomes": {
            "development_test_read": False,
            "new_final_evaluation_read": False,
            "emtab10902_outcome_read": False,
            "sealed_gse246381_read": False,
            "guided_xeditflow_run": False,
        },
        "new_training_attempt_created": False,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rights-table", type=Path, default=DEFAULT_RIGHTS_TABLE)
    parser.add_argument("--source-snapshot", type=Path, default=DEFAULT_SOURCE_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    audit = build_table(
        rights_table_path=args.rights_table,
        source_snapshot_path=args.source_snapshot,
        table_path=args.output,
        audit_path=args.audit,
        overwrite=args.overwrite,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
