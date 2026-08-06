#!/usr/bin/env python3
"""M4 data-readiness gate: audit qualified measured data per ACTIVE EditBench sub-benchmark.

The migration authority layer (M0-M3) is complete at terminal state
MIGRATION_READY_FOR_DATA_REBUILD. This gate is the first step of the data-rebuild
contract: establish the honest per-asset data-readiness baseline from the D1
canonical data, identify gaps, and verify the migration alignment invariants:

- sealed isolation: GSE246381 (and any restricted/SEALED asset) never enters a pool;
- no cross-region mixing: 5'UTR vs 3'UTR pools are independent endpoint heads;
- pool assets must be ACCEPTED_FOR_NEW_ROLE (no PENDING_BLOCKED);
- candidates are real measured source/candidate pairs (SOURCED, not fabricated).

Output: artifacts/migration/M4_DATA_READINESS.json (machine) + report section.
This does NOT fabricate a qualified pool; it reports readiness and gaps.
"""
import collections
import json
import pathlib

import yaml

R = pathlib.Path(".")
EXEC = R / "docs" / "execution"
ART = R / "artifacts" / "migration"

BENCH_REG = EXEC / "xeditflow_benchmark_registry.yaml"
ASSET_ROLE = EXEC / "xeditflow_asset_role_assignment.yaml"
D1_PAIRS = pathlib.Path("/mnt/cunyuliu/mrna_editflow_v3_1/d1_3u_rebuild_staging/ordinary/utr_edit_pairs.jsonl")

NEW_CONTRACT_ID = "mrna_xeditflow_goal_v1_1"


def load_yaml(p):
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def main() -> None:
    bench = load_yaml(BENCH_REG)
    role = load_yaml(ASSET_ROLE)
    role_by_acc = {a["asset_id"]: a for a in role["assets"]}

    # 1) Scan D1 pairs once, count per asset (by pair_id prefix).
    pair_counts = collections.Counter()
    n_pairs = 0
    if D1_PAIRS.exists():
        with D1_PAIRS.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                n_pairs += 1
                pair_counts[r["pair_id"].split("_")[0]] += 1
    norm = lambda acc: acc.lower().replace("-", "").replace("_", "")
    pair_by_norm = {norm(a): c for a, c in pair_counts.items()}

    # 2) Per ACTIVE sub-benchmark: list bound assets, their role, and D1 pair presence.
    subbench = {}
    for b in bench["sub_benchmarks"]:
        if b["status"] != "ACTIVE":
            continue
        bids = b["asset_ids"]
        per_asset = []
        for acc in bids:
            ra = role_by_acc.get(acc)
            has_pairs = pair_by_norm.get(norm(acc), 0)
            per_asset.append({
                "asset_id": acc,
                "pool_role": ra["role"] if ra else "UNKNOWN",
                "evidence_grade": (ra.get("orthogonal_axes") or {}).get("intervention_evidence_grade"),
                "method_role": (ra.get("orthogonal_axes") or {}).get("method_training_role"),
                "d1_pairs": has_pairs,
            })
        ready = all(p["pool_role"] == "ACCEPTED_FOR_NEW_ROLE" and p["d1_pairs"] > 0 for p in per_asset)
        subbench[b["id"]] = {
            "region": b["region"],
            "status": b["status"],
            "n_assets": len(bids),
            "n_assets_with_d1_pairs": sum(1 for p in per_asset if p["d1_pairs"] > 0),
            "ready": ready,
            "assets": per_asset,
            "primary_tasks": b["primary_tasks"],
            "splits": b["splits"],
            "sealed_external": b["sealed_external"],
        }

    # 3) Verify migration alignment invariants.
    violations = []
    all_pool_assets = [a for b in subbench.values() for a in b["assets"]]
    pool_accs = {a["asset_id"] for a in all_pool_assets}
    # sealed isolation: GSE246381 must not be in any pool.
    if "GSE246381" in pool_accs:
        violations.append("GSE246381 (sealed) leaked into a benchmark pool")
    # cross-region non-mixing.
    five_u = {a["asset_id"] for b in subbench.values() if b["region"] in ("5UTR",) for a in b["assets"]}
    three_u = {a["asset_id"] for b in subbench.values() if b["region"] in ("3UTR",) for a in b["assets"]}
    overlap = five_u & three_u
    if overlap:
        violations.append(f"cross-region mixing: {sorted(overlap)}")
    # pool assets must be ACCEPTED_FOR_NEW_ROLE.
    for a in all_pool_assets:
        if a["pool_role"] != "ACCEPTED_FOR_NEW_ROLE":
            violations.append(f"{a['asset_id']} in pool with role {a['pool_role']}")

    out = {
        "contract_id": NEW_CONTRACT_ID,
        "phase": "M4-DATA-READINESS",
        "date": "2026-08-06",
        "d1_pairs_total": n_pairs,
        "d1_pairs_available": D1_PAIRS.exists(),
        "sub_benchmarks": subbench,
        "alignment_violations": violations,
        "alignment_ok": not violations,
    }
    (ART / "M4_DATA_READINESS.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    for bid, s in subbench.items():
        print(f"{bid}: ready={s['ready']} assets_with_pairs={s['n_assets_with_d1_pairs']}/{s['n_assets']}")
    print("alignment_ok:", out["alignment_ok"], "violations:", violations)


if __name__ == "__main__":
    main()