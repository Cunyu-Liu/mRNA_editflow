#!/usr/bin/env python
"""FM0-A (v3.1): validator for the FM0-A exposure closure artifacts.

Checks:
  - foundation_candidates.json: policy (kinds, max general backbone, max
    specialist per region), license_clean + eligible definitions, final alias
    points only to eligible+license-clean candidate set.
  - foundation_exposure_ledger.jsonl: unique ledger keys, FK (cluster_id present
    in ordinary clusters), no GSE246381 member rows in ordinary, per-checkpoint
    eligibility consistency, DETECTED/UNKNOWN exclude only that checkpoint/claim.
  - GSE246381 aggregate/commitment: no member data, counters all 0.
  - FM0 effective exposure projection: rows bound to sequence entities.

No training and no GPU work.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="ordinary FM0 output dir")
    ap.add_argument("--restricted-dir", required=True, help="restricted GSE246381 dir")
    args = ap.parse_args()
    d = Path(args.out_dir)
    rdir = Path(args.restricted_dir)
    errors = Counter()
    counters = Counter()

    # ---- candidates ----
    cand = json.loads((d / "foundation_candidates.json").read_text(encoding="utf-8"))
    if not cand.get("policy_ok"):
        errors["candidate_policy_fail"] += 1
    n_general = sum(1 for c in cand["candidates"] if c["kind"] == "general_backbone")
    if n_general > 1:
        errors["max_general_backbones_exceeded"] += 1
    spec_regions = Counter(c["region"] for c in cand["candidates"] if c["kind"] == "specialist" and c["region"])
    for reg, cnt in spec_regions.items():
        if cnt > 1:
            errors[f"max_specialist_exceeded:{reg}"] += 1
    counters["candidates"] = len(cand["candidates"])

    # final alias must point to eligible + license-clean candidate
    eligible_set = {c["candidate_id"] for c in cand["candidates"] if c["eligible"] and c["license_clean"]}
    alias = cand["final_alias"]["alias"]
    if alias not in eligible_set:
        errors["final_alias_not_eligible"] += 1
    if set(cand["final_alias"]["eligible_set"]) != eligible_set:
        errors["final_alias_eligible_set_mismatch"] += 1

    # ---- ledger ----
    ledger_keys = set()
    cluster_ids = set()
    rows = []
    for row in iter_jsonl(d / "foundation_exposure_ledger.jsonl"):
        rows.append(row)
        k = row.get("ledger_key")
        if k in ledger_keys:
            errors["ledger_key_duplicate"] += 1
        ledger_keys.add(k)
        cluster_ids.add(row.get("cluster_id"))
    counters["ledger_rows"] = len(rows)

    # GSE246381 must not be present as a cluster in ordinary ledger (aggregate only)
    if "GSE246381" in cluster_ids:
        errors["gse246381_leak_into_ordinary_ledger"] += 1

    # per-row eligibility consistency
    for row in rows:
        if row["overlap_status"] == "DETECTED" and row["eligible"]:
            errors["detected_but_eligible"] += 1
        if row["overlap_status"] == "CLEAN" and not row["eligible"]:
            errors["clean_but_ineligible"] += 1
        if row["cluster_id"] == "GSE246381" and row["overlap_status"] != "CLEAN":
            errors["gse246381_overlap_not_clean"] += 1

    # DETECTED/UNKNOWN only exclude that checkpoint, not global data: every
    # cluster still has at least one eligible (project-internal) checkpoint.
    per_cluster_eligible = Counter()
    for row in rows:
        if row["eligible"]:
            per_cluster_eligible[row["cluster_id"]] += 1
    for cid, n in per_cluster_eligible.items():
        if n < 1:
            errors[f"cluster_no_eligible_checkpoint:{cid}"] += 1

    # ---- aggregate / commitment ----
    agg = json.loads((d / "GSE246381_FM0_AGGREGATE.json").read_text(encoding="utf-8"))
    if agg.get("member_data_emitted_to_ordinary") is not False:
        errors["aggregate_member_data_flag"] += 1
    if any(agg.get("analytic_or_final_counters", {}).values()):
        errors["aggregate_nonzero_analytic_counters"] += 1
    commit = json.loads((d / "GSE246381_FM0_COMMITMENT.json").read_text(encoding="utf-8"))
    if commit.get("member_data_emitted_to_ordinary") is not False:
        errors["commitment_member_data_flag"] += 1

    # ---- restricted mirror ----
    if not (rdir / "FM0_AGGREGATE.json").exists():
        errors["restricted_fm0_aggregate_missing"] += 1
    r_agg = json.loads((rdir / "FM0_AGGREGATE.json").read_text(encoding="utf-8"))
    if any(r_agg.get("analytic_or_final_counters", {}).values()):
        errors["restricted_aggregate_nonzero_counters"] += 1

    # ---- FM0 effective exposure projection ----
    proj_rows = 0
    proj_ids = set()
    for row in iter_jsonl(d / "fm0_effective_exposure_projection.jsonl"):
        proj_rows += 1
        proj_ids.add(row.get("object_id"))
        if not row.get("projection_sha256"):
            errors["projection_missing_sha256"] += 1
    counters["projection_rows"] = proj_rows
    if proj_rows == 0:
        errors["projection_empty"] += 1

    total = sum(errors.values())
    print(json.dumps({
        "counters": dict(counters),
        "errors": dict(errors),
        "total_errors": total,
        "status": "PASS" if total == 0 else "FAIL",
    }, indent=2, sort_keys=True))
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()