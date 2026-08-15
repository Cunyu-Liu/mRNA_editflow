#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import yaml

def y(root, rel):
    return yaml.safe_load((root / rel).read_text())

def validate(root):
    issues = []
    amendment = y(root, "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec028.yaml")
    config = y(root, "configs/route_a_v3.yaml")
    super_doc = y(root, "docs/contracts/supersession_mrna_xeditflow_v1_1_to_route_a_v3.yaml")
    roles = y(root, "docs/execution/route_a_v3_data_role_registry.yaml").get("dec028_single_study_track_use_roles", {})
    p0 = json.loads((root / "configs/route_a_v3_dec028_successor_p0_schema_v1.json").read_text())
    protocol = json.loads((root / "configs/route_a_v3_dec028_single_study_protocol_v1.json").read_text())
    a2 = json.loads((root / "configs/route_a_v3_a2_g0_evaluator_candidate_v1.json").read_text())
    if amendment["status"] != "FROZEN_OWNER_SELECTED_PENDING_FRESH_RUNTIME_AUTHORITY_SYNC": issues.append("AMENDMENT_STATUS")
    if amendment["preserved_full_route_a"]["current_counts"] != {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547}: issues.append("COUNTS_DRIFT")
    if any(amendment["locks"][key] for key in amendment["locks"] if key.endswith("_allowed")): issues.append("LOCK_DRIFT")
    if amendment["locks"]["g1_launched"]: issues.append("G1_LAUNCHED")
    if config.get("dec028_single_study_successor", {}).get("scientific_claim_status") != "NOT_ESTABLISHED": issues.append("CLAIM_DRIFT")
    if super_doc.get("dec028_pending_successor", {}).get("current_effective_authority_remains") != "V3-DEC-027": issues.append("SUPERSESSION_DRIFT")
    if len(roles) != 14 or roles.get("GSE200304") != "PRIMARY_SINGLE_STUDY_SOURCE_RELATIVE_DEVELOPMENT" or roles.get("GSE246381") != "SEALED_EXTERNAL_FINAL_ONLY": issues.append("ROLE_MAP_DRIFT")
    if len(p0["p0_groups"]) != 11 or p0["production_p0_authorized"]: issues.append("P0_DRIFT")
    if protocol["authority"]["training_allowed"] or protocol["authority"]["cuda_probe_allowed"] or protocol["authority"]["g1_launched"]: issues.append("PREMATURE_EXECUTION")
    if a2.get("candidate_id") != "ROUTE_A_V3_A2_G0_SYNTHETIC_EVALUATOR_CANDIDATE_V1": issues.append("A2_CANDIDATE_ID_DRIFT")
    if a2.get("document_status") != "DRAFT_FOR_REVIEW_NOT_ACTIVE_PROTOCOL" or a2.get("authority_status") != "NON_AUTHORITATIVE": issues.append("A2_AUTHORITY_DRIFT")
    if a2.get("activation_state") != "INACTIVE_EVALUATOR_CANDIDATE": issues.append("A2_ACTIVATION_DRIFT")
    if not (root / "scripts/route_a_v3/a2_g0_synthetic_evaluator_candidate.py").is_file(): issues.append("A2_SOURCE_MISSING")
    if not (root / "docs/reviews/2026-08-15-route-a-v3-dec028-a2-g0-independent-review.md").is_file(): issues.append("A2_REVIEW_MISSING")
    if not (root / "scripts/route_a_v3/record_dec028_ss1_a2_g0_static_validation.py").is_file(): issues.append("A2_STATIC_RECORDER_MISSING")
    if not (root / "tests/route_a_v3/test_record_dec028_ss1_a2_g0_static_validation.py").is_file(): issues.append("A2_STATIC_RECORDER_TEST_MISSING")
    if not (root / "docs/plans/2026-08-15-route-a-v3-dec028-a2-g0-implementation-plan.md").is_file(): issues.append("A2_PLAN_MISSING")
    if any(value not in (0, False) for value in a2.get("runtime_truth", {}).values()): issues.append("A2_RUNTIME_TRUTH_DRIFT")
    return issues

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    issues = validate(args.repo_root)
    print(json.dumps({"issue_count": len(issues), "issues": issues}, sort_keys=True))
    return 0 if not issues else 1

if __name__ == "__main__":
    raise SystemExit(main())
