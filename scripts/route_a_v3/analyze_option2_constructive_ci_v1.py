#!/usr/bin/env python3
"""Option 2 committee arm, step C: paired bootstrap CI for the CONSTRUCTIVE
routing vs single V5 on the SAME holdout split (the dev-routed delta CI is in
step B; the constructive routing needs its own CI for paper-grade claims).

Also computes per-task holdout contributions for the constructive routing.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT_ROOT = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/"
    "analysis_option2_committee_20260909"
)
CHECKPOINTS = ["full", "v8_hbench9", "v8_smprau_in", "v8_smrlpolya"]
SPLIT_SEED = 20260909
BOOT_ITERS, BOOT_SEED = 2000, 20260816

CONSTRUCTIVE_ROUTING = {
    "HL": "full",
    "MRL": "v8_hbench9",
    "MPRAU": "v8_smprau_in",
    "polyA": "v8_smprau_in",
}


def load_persrc() -> dict[str, dict[str, dict]]:
    data: dict[str, dict[str, dict]] = {}
    for name in CHECKPOINTS:
        per = {}
        path = OUT_ROOT / f"persrc_probe_{name}.per_source.jsonl"
        for line in path.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                per[r["source_key"]] = r
        data[name] = per
    return data


def main() -> int:
    data = load_persrc()
    ref_src = data["full"]
    sources = sorted(
        sk for sk, r in ref_src.items() if r.get("n_measured", 0) > 0
    )
    tasks = defaultdict(list)
    for sk in sources:
        tasks[ref_src[sk]["task"]].append(sk)

    import random

    rng = random.Random(SPLIT_SEED)
    dev, holdout = [], []
    for task in sorted(tasks):
        sks = sorted(tasks[task])
        rng.shuffle(sks)
        half = len(sks) // 2
        dev += sks[:half]
        holdout += sks[half:]

    base_vals = np.asarray(
        [data["full"][sk]["critic_conditional_rank_acc@1"] for sk in holdout]
    )
    cons_vals = np.asarray(
        [
            data[CONSTRUCTIVE_ROUTING[ref_src[sk]["task"]]][sk][
                "critic_conditional_rank_acc@1"
            ]
            for sk in holdout
        ]
    )
    rng_b = np.random.default_rng(BOOT_SEED)
    n = len(holdout)
    diffs = []
    for _ in range(BOOT_ITERS):
        idx = rng_b.integers(0, n, n)
        diffs.append(float(np.mean(cons_vals[idx] - base_vals[idx])))
    diffs.sort()
    lo, hi = diffs[int(0.025 * BOOT_ITERS)], diffs[int(0.975 * BOOT_ITERS) - 1]
    delta = float(np.mean(cons_vals - base_vals))

    per_task_holdout = {}
    for task in sorted(tasks):
        sks = [sk for sk in holdout if ref_src[sk]["task"] == task]
        v5 = float(np.mean([data["full"][sk]["critic_conditional_rank_acc@1"] for sk in sks]))
        cons = float(
            np.mean(
                [
                    data[CONSTRUCTIVE_ROUTING[task]][sk][
                        "critic_conditional_rank_acc@1"
                    ]
                    for sk in sks
                ]
            )
        )
        per_task_holdout[task] = {
            "n": len(sks),
            "routed_to": CONSTRUCTIVE_ROUTING[task],
            "v5": v5,
            "constructive": cons,
            "delta": cons - v5,
        }

    payload = {
        "schema_version": "route_a_v3_route2_option2_constructive_ci.v1",
        "bootstrap": f"{BOOT_ITERS} iters seed {BOOT_SEED} paired per-source on holdout n={n}",
        "constructive_routing": CONSTRUCTIVE_ROUTING,
        "delta_constructive_minus_v5": delta,
        "delta_ci95": [lo, hi],
        "delta_ci_excludes_zero": not (lo <= 0.0 <= hi),
        "holdout_n": n,
        "per_task_holdout": per_task_holdout,
    }
    out = OUT_ROOT / "option2_constructive_ci.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
