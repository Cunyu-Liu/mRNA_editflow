#!/usr/bin/env python3
"""P3-01 recovery: resolve cross-role exact-sequence collisions in the written
benchmark tiers, then refresh audit / manifest / summary IN PLACE.

Why this exists: the full build assigns split roles atomically per source /
family, but near-homologous sources in different families can still emit
byte-identical candidate sequences that cross roles (hard leakage violation in
``audit_split``). Rebuilding all tiers (~78 min CPU) just to drop those
records is wasteful; this script reuses the written tier JSONLs, applies the
SAME deterministic resolution as the driver's full-build path, and patches the
frozen artifacts. Kept records are rewritten byte-identically
(``json.dumps(sort_keys=True)`` round-trip), so unchanged tiers keep their
SHA-256.

Usage (server, repo root):
    python3 scripts/p3_01_resolve_collisions.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

TIERS = ("measured", "proxy", "unlabeled")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root.parent))

    from mrna_editflow.data.p3_local_edit_schema import (
        read_benchmark_jsonl,
        sha256_file,
        write_benchmark_jsonl,
    )
    from mrna_editflow.data.p3_split import (
        audit_split,
        resolve_cross_role_sequence_collisions,
    )

    bench_dir = repo_root / "data" / "p3" / "benchmark"
    man_dir = repo_root / "data" / "p3" / "manifests"
    audit_path = repo_root / "docs" / "p3_01_split_audit.json"
    summary_path = repo_root / "data" / "p3" / "p3_01_build_summary.json"
    manifest_path = man_dir / "p3_01_benchmark_manifest.json"

    # 1. Load written tiers (split_role already baked in).
    tier_records = {
        t: list(read_benchmark_jsonl(str(bench_dir / f"{t}_tier.jsonl")))
        for t in TIERS
    }

    # 2. Deterministic collision resolution (same policy as driver).
    dropped_ids, stats = resolve_cross_role_sequence_collisions(tier_records)
    if dropped_ids:
        for t in TIERS:
            tier_records[t] = [r for r in tier_records[t]
                               if r.record_id not in dropped_ids]

    # 3. Freeze the resolution artifact (auditable record-level drop list).
    art_path = man_dir / "p3_01_collision_resolution.json"
    with open(art_path, "w") as fh:
        json.dump({
            "resolution_id": "p3_01_collision_resolution",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stats": stats,
            "dropped_record_ids": sorted(dropped_ids),
        }, fh, indent=1, sort_keys=True)

    # 4. Rewrite tiers; recompute hashes/counts.
    tier_hashes, tier_counts = {}, {}
    for t in TIERS:
        p = bench_dir / f"{t}_tier.jsonl"
        tier_counts[t] = write_benchmark_jsonl(tier_records[t], str(p))
        tier_hashes[t] = sha256_file(str(p))
        print(f"      rewrote {p.name} n={tier_counts[t]} "
              f"sha256={tier_hashes[t][:16]}...", flush=True)

    # 5. Re-audit per tier + global.
    audits = {}
    for t in TIERS:
        audits[t] = audit_split(
            (r.to_dict() for r in read_benchmark_jsonl(str(bench_dir / f"{t}_tier.jsonl"))), t)
    audits["global"] = audit_split(
        (r.to_dict() for t in TIERS
         for r in read_benchmark_jsonl(str(bench_dir / f"{t}_tier.jsonl"))), "global")
    passed = all(a["passed"] for a in audits.values())

    role_counts = {"train": 0, "val": 0, "test": 0, "ood": 0}
    for recs in tier_records.values():
        for r in recs:
            role_counts[r.split_role] += 1

    # 6. Patch audit doc in place (preserve split_config / assignment_sha).
    with open(audit_path) as fh:
        audit_doc = json.load(fh)
    audit_doc["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    audit_doc["split_metadata"]["role_record_counts"] = role_counts
    audit_doc["split_metadata"]["collision_resolution"] = stats
    audit_doc["audits"] = audits
    audit_doc["passed"] = passed
    with open(audit_path, "w") as fh:
        json.dump(audit_doc, fh, indent=2, sort_keys=True)

    # 7. Patch benchmark manifest in place.
    with open(manifest_path) as fh:
        man = json.load(fh)
    for t in TIERS:
        man["tiers"][t]["sha256"] = tier_hashes[t]
        man["tiers"][t]["n_records"] = tier_counts[t]
    man["split"]["role_record_counts"] = role_counts
    man["split"]["audit_passed"] = passed
    man["split"]["collision_resolution"] = stats
    man["split"]["collision_resolution_path"] = str(art_path)
    man["build"]["command"] = " ".join(sys.argv)
    with open(manifest_path, "w") as fh:
        json.dump(man, fh, indent=2, sort_keys=True)

    # 8. Patch build summary in place.
    with open(summary_path) as fh:
        summ = json.load(fh)
    for t in TIERS:
        summ["tiers"][t]["n_records"] = tier_counts[t]
        summ["tiers"][t]["sha256"] = tier_hashes[t]
    summ["split"]["role_record_counts"] = role_counts
    summ["split"]["audit_passed"] = passed
    summ["split"]["collision_resolution"] = stats
    with open(summary_path, "w") as fh:
        json.dump(summ, fh, indent=2, sort_keys=True)

    print(json.dumps({
        "collision_resolution": stats,
        "tier_counts": tier_counts,
        "audit_passed": passed,
    }, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
