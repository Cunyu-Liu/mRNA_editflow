#!/usr/bin/env python3
"""Build and audit the V3.3.2 accountable-human study-rights review packet.

The generated CSV freezes machine-collected provider evidence and leaves every
human judgment/sign-off field blank.  A later accountable reviewer may fill a
copy and use ``--review-input`` to validate completeness.  Review completion
never authorizes a project release by itself; exact files, repository/version,
and the release decision remain separate gates.  No study outcome is read.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROVIDER_TABLE = (
    ROOT / "docs/paper/route2_v332_study_rights_provider_evidence_table_v1.csv"
)
DEFAULT_PACKET = (
    ROOT
    / "docs/paper/route2_v332_study_rights_accountable_human_review_packet_v1.csv"
)
DEFAULT_INSTRUCTIONS = (
    ROOT
    / "docs/paper/route2_v332_study_rights_accountable_human_review_instructions_v1.md"
)
DEFAULT_AUDIT = (
    ROOT
    / "audits/route_a_v3_route2_v332_study_rights_accountable_human_review_packet_v1.json"
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

MACHINE_FIELDS = (
    "review_packet_row_id",
    "study_unit_id",
    "current_analysis_role_v332",
    "outcome_exposure",
    "provider_family",
    "repository_name",
    "accession_landing_url",
    "provider_policy_url",
    "provider_policy_exception",
    "nature_data_access_route",
    "dataset_citation_action",
    "automated_accession_resolution_status",
    "automated_analysis_publication_use_supported",
    "automated_study_specific_license_record_present",
    "automated_project_payload_redistribution_authorized",
    "machine_publication_boundary",
    "provider_evidence_source_path",
)

HUMAN_FIELDS = (
    "human_review_status",
    "reviewer_name",
    "reviewer_role",
    "reviewer_affiliation",
    "review_date_iso",
    "accession_landing_and_non_outcome_metadata_verified",
    "non_outcome_dataset_content_scope_verified",
    "primary_dataset_citation_verified",
    "study_specific_rights_source_url",
    "study_specific_rights_source_type",
    "study_specific_license_or_terms",
    "analysis_and_publication_use_decision",
    "project_payload_redistribution_decision",
    "authorized_exact_file_scope",
    "restriction_or_hold_reason",
    "target_journal_name",
    "target_journal_policy_checked",
    "data_availability_wording_approved",
    "accountable_human_signoff",
    "reviewer_notes",
)

PROTECTED_FIELDS = (
    "development_test_read",
    "new_final_evaluation_read",
    "emtab10902_outcome_read",
    "sealed_gse246381_read",
    "guided_xeditflow_run",
)

FIELDNAMES = MACHINE_FIELDS + HUMAN_FIELDS + PROTECTED_FIELDS

REVIEW_STATUSES = {"PENDING", "COMPLETED", "HOLD"}
ANALYSIS_DECISIONS = {"", "CONFIRMED", "NOT_CONFIRMED", "NOT_APPLICABLE"}
REDISTRIBUTION_DECISIONS = {
    "",
    "NOT_AUTHORIZED",
    "AUTHORIZED_EXACT_FILES",
    "NOT_APPLICABLE",
}
TRUE_FALSE_OR_BLANK = {"", "true", "false"}


class HumanReviewPacketError(RuntimeError):
    """Review packet inputs or human decision fields violate the contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanReviewPacketError(message)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require(tuple(reader.fieldnames or ()) == FIELDNAMES, "review packet schema changed")
        return list(reader)


def _read_provider_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _template_rows(
    provider_rows: Sequence[Mapping[str, str]], provider_table_path: Path
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, source in enumerate(provider_rows, start=1):
        row = {
            "review_packet_row_id": f"RHR-{index:02d}",
            "study_unit_id": source["study_unit_id"],
            "current_analysis_role_v332": source["current_analysis_role_v332"],
            "outcome_exposure": source["outcome_exposure"],
            "provider_family": source["provider_family"],
            "repository_name": source["repository_name"],
            "accession_landing_url": source["accession_landing_url"],
            "provider_policy_url": source["provider_policy_url"],
            "provider_policy_exception": source["provider_policy_exception"],
            "nature_data_access_route": source["nature_data_access_route"],
            "dataset_citation_action": source["dataset_citation_action"],
            "automated_accession_resolution_status": source[
                "accession_resolution_status"
            ],
            "automated_analysis_publication_use_supported": source[
                "provider_analysis_and_publication_use_supported"
            ],
            "automated_study_specific_license_record_present": source[
                "study_specific_license_record_present"
            ],
            "automated_project_payload_redistribution_authorized": source[
                "project_payload_redistribution_authorized"
            ],
            "machine_publication_boundary": source["publication_boundary"],
            "provider_evidence_source_path": _display_path(provider_table_path),
            "human_review_status": "PENDING",
            **{field: "" for field in HUMAN_FIELDS if field != "human_review_status"},
            **{field: "false" for field in PROTECTED_FIELDS},
        }
        rows.append(row)
    return rows


def _validate_machine_fields(
    rows: Sequence[Mapping[str, str]],
    provider_rows: Sequence[Mapping[str, str]],
    provider_table_path: Path,
) -> None:
    ids = [row["study_unit_id"] for row in rows]
    _require(len(ids) == len(set(ids)) == 14, "packet must contain 14 unique studies")
    _require(set(ids) == EXPECTED_STUDY_IDS, "review study set changed")
    provider_by_id = {row["study_unit_id"]: row for row in provider_rows}
    _require(set(provider_by_id) == EXPECTED_STUDY_IDS, "provider study set changed")
    expected_source = _display_path(provider_table_path)
    mapping = {
        "current_analysis_role_v332": "current_analysis_role_v332",
        "outcome_exposure": "outcome_exposure",
        "provider_family": "provider_family",
        "repository_name": "repository_name",
        "accession_landing_url": "accession_landing_url",
        "provider_policy_url": "provider_policy_url",
        "provider_policy_exception": "provider_policy_exception",
        "nature_data_access_route": "nature_data_access_route",
        "dataset_citation_action": "dataset_citation_action",
        "automated_accession_resolution_status": "accession_resolution_status",
        "automated_analysis_publication_use_supported": (
            "provider_analysis_and_publication_use_supported"
        ),
        "automated_study_specific_license_record_present": (
            "study_specific_license_record_present"
        ),
        "automated_project_payload_redistribution_authorized": (
            "project_payload_redistribution_authorized"
        ),
        "machine_publication_boundary": "publication_boundary",
    }
    for row in rows:
        source = provider_by_id[row["study_unit_id"]]
        for packet_field, provider_field in mapping.items():
            _require(
                row[packet_field] == source[provider_field],
                f"machine evidence changed for {row['study_unit_id']}: {packet_field}",
            )
        _require(
            row["provider_evidence_source_path"] == expected_source,
            f"provider evidence path changed for {row['study_unit_id']}",
        )
        _require(
            row["automated_analysis_publication_use_supported"] == "true"
            and row["automated_study_specific_license_record_present"] == "false"
            and row["automated_project_payload_redistribution_authorized"] == "false",
            f"automated rights boundary changed for {row['study_unit_id']}",
        )
        _require(
            all(row[field] == "false" for field in PROTECTED_FIELDS),
            f"protected outcome boundary changed for {row['study_unit_id']}",
        )


def _validate_human_fields(rows: Sequence[Mapping[str, str]]) -> None:
    boolean_fields = (
        "accession_landing_and_non_outcome_metadata_verified",
        "non_outcome_dataset_content_scope_verified",
        "primary_dataset_citation_verified",
        "target_journal_policy_checked",
        "data_availability_wording_approved",
    )
    identity_fields = (
        "reviewer_name",
        "reviewer_role",
        "reviewer_affiliation",
        "review_date_iso",
        "accountable_human_signoff",
    )
    completed_fields = identity_fields + (
        "study_specific_rights_source_url",
        "study_specific_rights_source_type",
        "study_specific_license_or_terms",
        "target_journal_name",
    )
    for row in rows:
        study_id = row["study_unit_id"]
        _require(row["human_review_status"] in REVIEW_STATUSES, f"invalid review status for {study_id}")
        _require(row["analysis_and_publication_use_decision"] in ANALYSIS_DECISIONS, f"invalid analysis decision for {study_id}")
        _require(row["project_payload_redistribution_decision"] in REDISTRIBUTION_DECISIONS, f"invalid redistribution decision for {study_id}")
        _require(
            all(row[field] in TRUE_FALSE_OR_BLANK for field in boolean_fields),
            f"invalid human boolean field for {study_id}",
        )
        if row["human_review_status"] == "PENDING":
            _require(not row["accountable_human_signoff"], f"pending row cannot be signed: {study_id}")
            _require(
                row["project_payload_redistribution_decision"]
                != "AUTHORIZED_EXACT_FILES",
                f"pending row cannot authorize exact files: {study_id}",
            )
            _require(
                not row["authorized_exact_file_scope"],
                f"pending row cannot name an authorized exact-file scope: {study_id}",
            )
            continue
        _require(all(row[field] for field in identity_fields), f"review identity/signoff incomplete for {study_id}")
        if row["human_review_status"] == "HOLD":
            _require(row["restriction_or_hold_reason"], f"hold reason missing for {study_id}")
            _require(
                row["project_payload_redistribution_decision"]
                != "AUTHORIZED_EXACT_FILES",
                f"held row cannot authorize exact files: {study_id}",
            )
            _require(
                not row["authorized_exact_file_scope"],
                f"held row cannot name an authorized exact-file scope: {study_id}",
            )
            continue
        _require(all(row[field] for field in completed_fields), f"completed review evidence incomplete for {study_id}")
        _require(all(row[field] == "true" for field in boolean_fields), f"completed review checks incomplete for {study_id}")
        _require(row["analysis_and_publication_use_decision"] != "", f"analysis decision missing for {study_id}")
        _require(row["project_payload_redistribution_decision"] != "", f"redistribution decision missing for {study_id}")
        if row["project_payload_redistribution_decision"] == "AUTHORIZED_EXACT_FILES":
            _require(row["analysis_and_publication_use_decision"] == "CONFIRMED", f"authorized redistribution requires confirmed use for {study_id}")
            _require(row["authorized_exact_file_scope"], f"authorized exact-file scope missing for {study_id}")
        else:
            _require(not row["authorized_exact_file_scope"], f"exact-file scope set without authorization for {study_id}")


def _audit(
    rows: Sequence[Mapping[str, str]],
    *,
    provider_table_path: Path,
    packet_path: Path,
    instructions_path: Path,
) -> dict[str, Any]:
    counts = {status: sum(row["human_review_status"] == status for row in rows) for status in REVIEW_STATUSES}
    human_complete = counts["COMPLETED"] == 14
    authorized_count = sum(
        row["project_payload_redistribution_decision"] == "AUTHORIZED_EXACT_FILES"
        for row in rows
    )
    analysis_only_count = sum(
        row["human_review_status"] == "COMPLETED"
        and row["project_payload_redistribution_decision"] == "NOT_AUTHORIZED"
        for row in rows
    )
    if human_complete:
        status = "ACCOUNTABLE_HUMAN_RIGHTS_REVIEW_COMPLETE_PROJECT_RELEASE_STILL_SEPARATELY_GATED"
    elif counts["COMPLETED"] or counts["HOLD"]:
        status = "ACCOUNTABLE_HUMAN_RIGHTS_REVIEW_IN_PROGRESS_PROJECT_RELEASE_NOT_AUTHORIZED"
    else:
        status = "ACCOUNTABLE_HUMAN_REVIEW_PACKET_TEMPLATE_READY_ZERO_REVIEWS_COMPLETED_PROJECT_RELEASE_NOT_AUTHORIZED"
    return {
        "schema_version": "route_a_v3_route2_v332_study_rights_accountable_human_review_packet.v1",
        "status": status,
        "authority": {
            "provider_evidence_table": _display_path(provider_table_path),
            "scientific_contract": "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/提示词/mrna 数据gate转向后的合同.md",
            "execution_protocol": "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/提示词/mrna V3.3.2 执行提示词.md",
        },
        "row_count": len(rows),
        "human_review_counts": {
            "pending": counts["PENDING"],
            "completed": counts["COMPLETED"],
            "hold": counts["HOLD"],
        },
        "analysis_citation_only_completed_count": analysis_only_count,
        "exact_file_redistribution_authorized_review_count": authorized_count,
        "rows_with_target_journal_policy_checked": sum(
            row["target_journal_policy_checked"] == "true" for row in rows
        ),
        "rows_with_accountable_human_signoff": sum(
            bool(row["accountable_human_signoff"]) for row in rows
        ),
        "template_ready": True,
        "machine_evidence_frozen": True,
        "review_packet_requires_accountable_human_judgment": True,
        "agent_review_substituted_for_human": False,
        "human_content_and_rights_verification_complete": human_complete,
        "review_completion_is_project_release_authorization": False,
        "project_public_release_authorized": False,
        "stable_repository_version_assigned": False,
        "minimum_package_complete": False,
        "submission_ready": False,
        "packet_path": _display_path(packet_path),
        "instructions_path": _display_path(instructions_path),
        "protected_outcomes": {
            "development_test_read": False,
            "new_final_evaluation_read": False,
            "emtab10902_outcome_read": False,
            "sealed_gse246381_read": False,
            "guided_xeditflow_run": False,
        },
        "new_training_attempt_created": False,
    }


def _instructions_text(packet_path: Path) -> str:
    return f"""# Route 2 V3.3.2 accountable-human study-rights review instructions

## Purpose and boundary

This decision register records accountable human judgment for the 14 reused
third-party studies. Machine-collected provider evidence is frozen in
`{_display_path(packet_path)}`. It supports accession citation and an
analysis/publication route; it is not a study-specific licence and does not
authorize this project to redistribute source payloads.

Do not use Development TEST, new final Evaluation, sealed GSE246381,
E-MTAB-10902 outcomes, generated-candidate outcomes or guided XEditFlow output
for this review. Rights and non-outcome metadata are sufficient.

## Reviewer action

For each row, an accountable reviewer must identify themselves and the review
date; resolve the accession and non-outcome metadata; verify the data scope and
primary dataset citation; record the exact study-specific rights source and
terms; decide analysis/publication use; decide whether redistribution is
`NOT_AUTHORIZED`, `AUTHORIZED_EXACT_FILES` or `NOT_APPLICABLE`; check the
selected target-journal policy; approve the Data Availability wording; and add
an accountable sign-off.

Use `AUTHORIZED_EXACT_FILES` only when `authorized_exact_file_scope` names the
exact files covered by the reviewed authority. A general provider policy,
converter declaration or prior operational setting is not sufficient. Use
`HOLD` with a concrete reason when the evidence is unresolved. Keep public
payload release closed for every pending or held row.

## Validation and release boundary

After a human edits a copy of the CSV, validate it with this builder's
`--review-input` mode and write a new audit. Validation checks completeness and
internal consistency; it does not authenticate the reviewer or independently
adjudicate the legal conclusion. Even 14 completed rows do not automatically
authorize a public release: the exact release files, stable repository/version,
code licence, tracked legacy-payload policy and final project release decision
remain separate gates.

## Current manuscript wording boundary

Reused source accessions and aggregate evidence may be cited. Upstream study
payloads must not be redistributed by this project until accountable review and
the separate release decision authorize exact files. No availability-on-request
promise should be added without a durable responsible route and explicit access
conditions.
"""


def build_review_packet(
    *,
    provider_table_path: Path = DEFAULT_PROVIDER_TABLE,
    packet_path: Path = DEFAULT_PACKET,
    instructions_path: Path = DEFAULT_INSTRUCTIONS,
    audit_path: Path = DEFAULT_AUDIT,
    overwrite: bool = False,
) -> dict[str, Any]:
    for path in (packet_path, instructions_path, audit_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    provider_rows = _read_provider_rows(provider_table_path)
    rows = _template_rows(provider_rows, provider_table_path)
    _validate_machine_fields(rows, provider_rows, provider_table_path)
    _validate_human_fields(rows)
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    with packet_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    instructions_path.write_text(_instructions_text(packet_path), encoding="utf-8")
    audit = _audit(
        rows,
        provider_table_path=provider_table_path,
        packet_path=packet_path,
        instructions_path=instructions_path,
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def audit_review_packet(
    *,
    review_input_path: Path,
    provider_table_path: Path = DEFAULT_PROVIDER_TABLE,
    instructions_path: Path = DEFAULT_INSTRUCTIONS,
    audit_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if audit_path is not None and audit_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifact: {audit_path}")
    provider_rows = _read_provider_rows(provider_table_path)
    rows = _read_rows(review_input_path)
    _validate_machine_fields(rows, provider_rows, provider_table_path)
    _validate_human_fields(rows)
    audit = _audit(
        rows,
        provider_table_path=provider_table_path,
        packet_path=review_input_path,
        instructions_path=instructions_path,
    )
    if audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-table", type=Path, default=DEFAULT_PROVIDER_TABLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--instructions", type=Path, default=DEFAULT_INSTRUCTIONS)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--review-input", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.review_input is not None:
        audit = audit_review_packet(
            review_input_path=args.review_input,
            provider_table_path=args.provider_table,
            instructions_path=args.instructions,
            audit_path=args.audit,
            overwrite=args.overwrite,
        )
    else:
        audit = build_review_packet(
            provider_table_path=args.provider_table,
            packet_path=args.output,
            instructions_path=args.instructions,
            audit_path=args.audit,
            overwrite=args.overwrite,
        )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
