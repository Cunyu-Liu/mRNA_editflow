#!/usr/bin/env python3
"""Build the selected-outcome claim/evidence and unsupported-claim table.

Current draft claim text and evidence IDs are extracted from the manuscript so
the table cannot silently drift away from its markers.  Scientific boundaries
and deliberately unsupported statements are explicit, versioned definitions.
No experiment outcome, Development TEST, final Evaluation or generated-candidate
payload is opened by this reporting builder.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DRAFT = ROOT / "docs/paper/route2_v332_methods_results_draft_v1.md"
DEFAULT_EVIDENCE = ROOT / "docs/paper/route2_v332_evidence_manifest_v1.json"
DEFAULT_OUTCOME_AUDIT = ROOT / "audits/route_a_v3_route2_v332_paper_outcome_adjudication_v1.json"
DEFAULT_OUTPUT_TABLE = ROOT / "docs/paper/route2_v332_selected_outcome_claim_evidence_table_v1.csv"
DEFAULT_OUTPUT_AUDIT = ROOT / "audits/route_a_v3_route2_v332_selected_outcome_claim_evidence_table_v1.json"

SELECTED_OUTCOME = "BENCHMARK_PLUS_TRANSFER_AND_GENERATION_LIMITS_PAPER"

CLAIM_DEFINITIONS = {
    "C-R2-001": ("PROJECT_STATE_AND_ROUTE", "Internal evidence packet and frozen Benchmark+limits route only; package trigger and submission eligibility remain false."),
    "C-R2-002": ("DATA_AND_PROTECTED_OUTCOME", "Frozen Development counts and protected-outcome closure; no TEST or new final Evaluation performance claim."),
    "C-R2-003": ("INDEPENDENT_EVALUATOR_METHOD", "Development method-selection evaluator only; not an external biological assay or final Evaluation."),
    "C-R2-004": ("GENERATION_COMPARISON_METHOD", "Frozen 891-source Development comparison and declared caps; not biological efficacy."),
    "C-R2-005": ("GENERATION_SELECTION_METHOD", "Development independent-evaluator bootstrap and sparse recovery only; unknown generated outcomes remain unknown."),
    "C-R2-006": ("CRITIC_V2_PROTOCOL", "Prospectively frozen Development control-screen design; no readiness or external-transfer inference."),
    "C-R2-007": ("EVALUATOR_QUALIFICATION", "Narrow Development method-selection qualification only; no biological validation."),
    "C-R2-008": ("GENERATION_LEGALITY_AND_COUNTS", "Candidate ceiling was shared but realized candidate count differed; no equal-count claim."),
    "C-R2-009": ("GENERATION_DESCRIPTIVE_RESULTS", "Source-macro Development aggregates only; evidence layers and endpoints remain separate."),
    "C-R2-010": ("GENERATION_ENDPOINT_DEPENDENCE", "Endpoint-dependent Development ranking; neither evaluator uplift nor sparse recovery is verified biological improvement."),
    "C-R2-011": ("CRITIC_V1_NO_GO", "Terminal three-seed readiness failure; no TEST, refit, LOSO or guided authorization."),
    "C-R2-012": ("CRITIC_V2_NO_GO", "Terminal frozen control-gate failure; no downstream confirmation or guided stage."),
    "C-R2-013": ("COMPUTE_REPORTING_GAP", "Missing terminal wall/STOP timing remains missing and is not reconstructed or rerun."),
    "C-R2-014": ("READINESS_SEQUENCE", "Prospective ordering and gate definitions only; readiness is not biological success."),
    "C-R2-015": ("GENERATION_BOOTSTRAP", "Development independent-evaluator separation only; no external or measured biological validation."),
    "C-R2-016": ("EVALUATOR_TASK_HETEROGENEITY", "Nine Development tasks are heterogeneous; macro qualification does not imply uniform task reliability."),
    "C-R2-017": ("EVALUATOR_COMPRESSION_DIAGNOSTIC", "Global heterogeneous-scale spread ratio is diagnostic only and cannot establish per-task mean collapse."),
    "C-R2-018": ("EVALUATOR_ADJUDICATION", "Frozen evaluator checks authorized the completed Development rerun only; scientific success remains unestablished."),
    "C-R2-019": ("CRITIC_TASK_ERROR_GEOMETRY", "Localized Development candidate signal without task-wide rank or calibration superiority; mechanism is not causal."),
    "C-R2-020": ("LEGAL_ACTION_SPACE_GEOMETRY", "Computational SUB+STOP geometry only; not biological performance and not INS/DEL validation."),
    "C-R2-021": ("HISTORICAL_TRANSFER_NEGATIVE", "Outcome-exposed negative historical transfer evidence; not unbiased final confirmation."),
    "C-R2-022": ("MINIMUM_PACKAGE_AND_ROUTE", "All requirements are adjudicated but the package remains incomplete; Benchmark+limits route selection is not submission readiness."),
}

UNSUPPORTED_CLAIMS = (
    ("U-R2-001", "OUTCOME_A_HEADLINE", "A unified source-relative Delta critic improves effect prediction on a new frozen outcome-unexposed external Evaluation and guided XEditFlow improves measured or independent generation performance.", "Outcome A conditions fail: no eligible external Evaluation, critic readiness, guided run, true-A2 ranking or guided improvement.", "E-R2-PAPER-OUTCOME-ADJUDICATION;E-R2-CRITIC-V2-ADJ;E-R2-PACKAGE-AUDIT"),
    ("U-R2-002", "OUTCOME_B_HEADLINE", "Source-relative Delta prediction has stable value on a predeclared outcome-unexposed external prediction task.", "Outcome B conditions fail: no eligible external prediction task established stable Delta value.", "E-R2-PAPER-OUTCOME-ADJUDICATION;E-R2-GSE232-HIST"),
    ("U-R2-003", "BIOLOGICAL_IMPROVEMENT", "Genetic search or Base Flow improves mRNA biology.", "Only Development evaluator uplift and sparse measured-neighborhood recovery are available; closed measured outcome support is absent.", "E-R2-GEN-THREE-LAYER-AUDIT;E-R2-GEN-SELECT"),
    ("U-R2-004", "CRITIC_READINESS", "The critic is ready for guidance.", "Critic V2 failed the frozen strongest-baseline control and does not authorize confirmation seeds or readiness.", "E-R2-CRITIC-V2-ADJ;E-R2-CRITIC-V2-READINESS"),
    ("U-R2-005", "GUIDED_SUCCESS", "Guided XEditFlow succeeds.", "First-order and frozen-critic guidance were not run because Critic V2 is terminal NO-GO.", "E-R2-CRITIC-V2-ADJ;E-R2-MATCHED-BUDGET-AUDIT"),
    ("U-R2-006", "EXTERNAL_GENERATION_VALIDATION", "The generation results are externally validated.", "Generation comparison is Development-only and no outcome-unexposed final Evaluation exists.", "E-R2-GEN-THREE-LAYER-AUDIT;E-R2-DATA-TABLE"),
    ("U-R2-007", "REALIZED_CANDIDATE_COUNT", "Every method produced exactly 32 candidates per source.", "Local search produced 3--32 candidates per source and 21,027 total candidate rows.", "E-R2-GEN-INPUT;E-R2-GEN-QUALITY-COST-FIGURE-MANIFEST"),
    ("U-R2-008", "ACTION_SCOPE", "The first-stage benchmark supports or validates INS/DEL.", "The frozen first-stage action space is SUB+STOP; INS/DEL are explicitly out of scope.", "E-R2-CONTRACT;E-R2-GEN-INPUT"),
    ("U-R2-009", "MISSING_AS_ZERO", "A generated candidate outside measured support has zero measured gain.", "Unknown generated outcomes remain unknown and are not assigned zero gain.", "E-R2-GEN-THREE-LAYER-AUDIT"),
    ("U-R2-010", "HISTORICAL_FINAL_CONFIRMATION", "GSE232572 is an unbiased independent final confirmation.", "GSE232572 is historically outcome-exposed and failed its preregistered cross-seed rank-and-MAE rule.", "E-R2-GSE232-HIST;E-R2-CONTRACT"),
    ("U-R2-011", "PACKAGE_OR_SUBMISSION_COMPLETENESS", "The minimum benchmark package or submission-ready paper is complete.", "Four minimum-package blockers remain and submission eligibility is false.", "E-R2-PACKAGE-AUDIT;E-R2-PAPER-OUTCOME-ADJUDICATION"),
    ("U-R2-012", "UNREAD_OUTCOME", "E-MTAB-10902 outcome evaluation was completed.", "E-MTAB-10902 remains unconvertible and its outcome was not read.", "E-R2-CONTRACT;E-R2-DATA-TABLE"),
    ("U-R2-013", "CAUSAL_REGION_CONTEXT_MECHANISM", "Region or biological context is the causal mechanism of terminal prediction failure.", "Region summaries are post hoc and confounded; within-assay context-specific terminal error metrics are unavailable.", "E-R2-ERROR-DOMAIN-SHIFT-AUDIT"),
)

FIELDS = (
    "claim_row_id",
    "claim_id",
    "claim_marker_present",
    "claim_class",
    "claim_status",
    "allowed_in_selected_outcome_manuscript",
    "selected_final_paper_outcome",
    "claim_text",
    "evidence_ids",
    "evidence_role",
    "claim_boundary_or_refutation",
    "minimum_package_complete",
    "outcome_trigger_fully_satisfied",
    "submission_ready",
    "development_test_read",
    "new_final_evaluation_read",
    "guided_xeditflow_run",
)


class ClaimEvidenceInputError(RuntimeError):
    """Draft, evidence manifest or outcome audit violates the claim contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ClaimEvidenceInputError(message)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _extract_draft_claims(draft: str) -> dict[str, dict[str, Any]]:
    claims: dict[str, dict[str, Any]] = {}
    for block in re.split(r"\n\s*\n", draft):
        claim_ids = re.findall(r"\[claim:([^\]]+)\]", block)
        if not claim_ids:
            continue
        evidence_ids: list[str] = []
        for group in re.findall(r"\[evidence:([^\]]+)\]", block):
            evidence_ids.extend(value.strip() for value in group.split(","))
        text = re.sub(r"\[(?:claim|evidence):[^\]]+\]", "", block)
        text = " ".join(text.split())
        for claim_id in claim_ids:
            _require(claim_id not in claims, f"duplicate claim marker {claim_id}")
            claims[claim_id] = {
                "claim_text": text,
                "evidence_ids": list(dict.fromkeys(evidence_ids)),
            }
    return claims


def derive_rows_and_audit(
    *, draft: str, evidence_manifest: Mapping[str, Any], outcome: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    draft_claims = _extract_draft_claims(draft)
    _require(set(draft_claims) == set(CLAIM_DEFINITIONS), "draft claim marker set changed or is not fully mapped")
    evidence_ids = {row["evidence_id"] for row in evidence_manifest["sources"]}
    _require(outcome["selected_final_paper_outcome"] == SELECTED_OUTCOME, "selected paper outcome changed")
    _require(outcome["final_paper_outcome_frozen"] is True, "paper outcome is not frozen")
    _require(outcome["outcome_trigger_fully_satisfied"] is False, "outcome trigger was unexpectedly completed")
    _require(outcome["submission_ready"] is False, "submission status unexpectedly became ready")

    rows: list[dict[str, Any]] = []
    for index, claim_id in enumerate(sorted(CLAIM_DEFINITIONS), start=1):
        claim_class, boundary = CLAIM_DEFINITIONS[claim_id]
        extracted = draft_claims[claim_id]
        _require(bool(extracted["evidence_ids"]), f"claim {claim_id} has no adjacent evidence marker")
        unknown = set(extracted["evidence_ids"]) - evidence_ids
        _require(not unknown, f"claim {claim_id} references unknown evidence IDs: {sorted(unknown)}")
        rows.append(
            {
                "claim_row_id": f"CE-S-{index:02d}",
                "claim_id": claim_id,
                "claim_marker_present": "true",
                "claim_class": claim_class,
                "claim_status": "SUPPORTED_WITH_DECLARED_BOUNDARY",
                "allowed_in_selected_outcome_manuscript": "true",
                "selected_final_paper_outcome": SELECTED_OUTCOME,
                "claim_text": extracted["claim_text"],
                "evidence_ids": ";".join(extracted["evidence_ids"]),
                "evidence_role": "DIRECT_SUPPORT_WITH_SCOPE_LIMIT",
                "claim_boundary_or_refutation": boundary,
                "minimum_package_complete": "false",
                "outcome_trigger_fully_satisfied": "false",
                "submission_ready": "false",
                "development_test_read": "false",
                "new_final_evaluation_read": "false",
                "guided_xeditflow_run": "false",
            }
        )

    for index, (claim_id, claim_class, text, refutation, evidence) in enumerate(UNSUPPORTED_CLAIMS, start=1):
        refuting_ids = evidence.split(";")
        unknown = set(refuting_ids) - evidence_ids
        _require(not unknown, f"unsupported claim {claim_id} references unknown evidence IDs: {sorted(unknown)}")
        rows.append(
            {
                "claim_row_id": f"CE-U-{index:02d}",
                "claim_id": claim_id,
                "claim_marker_present": "false",
                "claim_class": claim_class,
                "claim_status": "UNSUPPORTED",
                "allowed_in_selected_outcome_manuscript": "false",
                "selected_final_paper_outcome": SELECTED_OUTCOME,
                "claim_text": text,
                "evidence_ids": evidence,
                "evidence_role": "REFUTATION_OR_MISSING_REQUIRED_DEPENDENCY",
                "claim_boundary_or_refutation": refutation,
                "minimum_package_complete": "false",
                "outcome_trigger_fully_satisfied": "false",
                "submission_ready": "false",
                "development_test_read": "false",
                "new_final_evaluation_read": "false",
                "guided_xeditflow_run": "false",
            }
        )

    supported = [row for row in rows if row["claim_status"].startswith("SUPPORTED")]
    unsupported = [row for row in rows if row["claim_status"] == "UNSUPPORTED"]
    audit = {
        "schema_version": "route_a_v3_route2_v332_selected_outcome_claim_evidence_table.v1",
        "status": "SELECTED_OUTCOME_CLAIM_EVIDENCE_CLOSED_UNSUPPORTED_CLAIMS_EXPLICIT",
        "selected_final_paper_outcome": SELECTED_OUTCOME,
        "final_paper_outcome_frozen": True,
        "outcome_trigger_fully_satisfied": False,
        "submission_ready": False,
        "row_count": len(rows),
        "draft_claim_marker_count": len(draft_claims),
        "supported_with_declared_boundary_row_count": len(supported),
        "unsupported_claim_row_count": len(unsupported),
        "unmapped_draft_claim_marker_count": 0,
        "duplicate_claim_id_count": 0,
        "unknown_evidence_id_reference_count": 0,
        "unsupported_claims_allowed_in_manuscript_count": sum(
            row["allowed_in_selected_outcome_manuscript"] == "true" for row in unsupported
        ),
        "claim_evidence_table_complete": True,
        "minimum_package_complete": False,
        "model_or_biological_success_established": False,
        "protected_outcomes": {
            "development_test_read": False,
            "new_final_evaluation_read": False,
            "guided_xeditflow_run": False,
        },
        "scientific_claim_status": "BENCHMARK_AND_LIMITS_CLAIMS_ONLY_MODEL_AND_BIOLOGICAL_SUCCESS_NOT_ESTABLISHED",
    }
    return rows, audit


def build_table(
    *, draft_path: Path = DEFAULT_DRAFT, evidence_path: Path = DEFAULT_EVIDENCE,
    outcome_audit_path: Path = DEFAULT_OUTCOME_AUDIT,
    output_table_path: Path = DEFAULT_OUTPUT_TABLE,
    output_audit_path: Path = DEFAULT_OUTPUT_AUDIT,
) -> dict[str, Any]:
    if output_table_path.exists() or output_audit_path.exists():
        raise FileExistsError("refusing to overwrite an existing claim/evidence artifact")
    rows, audit = derive_rows_and_audit(
        draft=draft_path.read_text(encoding="utf-8"),
        evidence_manifest=_load_json(evidence_path),
        outcome=_load_json(outcome_audit_path),
    )
    output_table_path.parent.mkdir(parents=True, exist_ok=True)
    with output_table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    audit["table_path"] = _display_path(output_table_path)
    audit["source_paths"] = {
        "draft": _display_path(draft_path),
        "evidence_manifest": _display_path(evidence_path),
        "paper_outcome_adjudication": _display_path(outcome_audit_path),
    }
    output_audit_path.parent.mkdir(parents=True, exist_ok=True)
    output_audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--outcome-audit", type=Path, default=DEFAULT_OUTCOME_AUDIT)
    parser.add_argument("--output-table", type=Path, default=DEFAULT_OUTPUT_TABLE)
    parser.add_argument("--output-audit", type=Path, default=DEFAULT_OUTPUT_AUDIT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(json.dumps(build_table(
        draft_path=args.draft,
        evidence_path=args.evidence,
        outcome_audit_path=args.outcome_audit,
        output_table_path=args.output_table,
        output_audit_path=args.output_audit,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
