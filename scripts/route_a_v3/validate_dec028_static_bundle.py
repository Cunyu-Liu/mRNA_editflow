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
    if amendment["status"] != "FROZEN_OWNER_SELECTED_PENDING_FRESH_RUNTIME_AUTHORITY_SYNC": issues.append("AMENDMENT_STATUS")
    if amendment["preserved_full_route_a"]["current_counts"] != {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547}: issues.append("COUNTS_DRIFT")
    if any(amendment["locks"][key] for key in amendment["locks"] if key.endswith("_allowed")): issues.append("LOCK_DRIFT")
    if amendment["locks"]["g1_launched"]: issues.append("G1_LAUNCHED")
    if config.get("dec028_single_study_successor", {}).get("scientific_claim_status") != "NOT_ESTABLISHED": issues.append("CLAIM_DRIFT")
    if super_doc.get("dec028_pending_successor", {}).get("current_effective_authority_remains") != "V3-DEC-027": issues.append("SUPERSESSION_DRIFT")
    if len(roles) != 14 or roles.get("GSE200304") != "PRIMARY_SINGLE_STUDY_SOURCE_RELATIVE_DEVELOPMENT" or roles.get("GSE246381") != "SEALED_EXTERNAL_FINAL_ONLY": issues.append("ROLE_MAP_DRIFT")
    if len(p0["p0_groups"]) != 11 or p0["production_p0_authorized"]: issues.append("P0_DRIFT")
    if protocol["authority"]["training_allowed"] or protocol["authority"]["cuda_probe_allowed"] or protocol["authority"]["g1_launched"]: issues.append("PREMATURE_EXECUTION")
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
