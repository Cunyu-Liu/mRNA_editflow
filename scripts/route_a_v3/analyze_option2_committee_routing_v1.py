#!/usr/bin/env python3
"""Option 2 committee arm, step B: preregistered dev/holdout task routing.

Protocol (registered BEFORE the probe re-runs are unblinded to this script's
selection logic; commit order guarantees):
  1. 891 sources stratified-split 50/50 into dev/holdout by task, seed 20260909.
  2. On DEV only: per task, select the checkpoint with max mean
     critic_conditional_rank_acc@1 among the 4 candidate checkpoints
     {full(V5), v8_hbench9, v8_smprau_in, v8_smrlpolya}.
  3. Freeze the routing table. On HOLDOUT: committee value (routed per-task
     accuracy), single-best (V5) value, paired per-source bootstrap
     (2000 iters, seed 20260816) CI for committee - V5.
  4. Also report the constructive routing (train-domain membership: task ->
     its own training specialist; V5 for HL, h_bench9 for MRL, s_mprau_in for
     MPRAU+polyA) evaluated on the same holdout -- a zero-peeking fallback
     that does not depend on dev selection at all.

Inputs: persrc_probe_{full,v8_hbench9,v8_smprau_in,v8_smrlpolya}.per_source.jsonl
Outputs: option2_committee_routing.json + .md
"""
from __future__ import annotations

import json
import random
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

    rng = random.Random(SPLIT_SEED)
    dev, holdout = [], []
    for task in sorted(tasks):
        sks = sorted(tasks[task])
        rng.shuffle(sks)
        half = len(sks) // 2
        dev += sks[:half]
        holdout += sks[half:]

    # sanity: identical source sets across the 4 probes
    for name in CHECKPOINTS:
        assert set(data[name]) == set(ref_src), f"source set mismatch: {name}"

    # ---- DEV: routing selection ----
    routing: dict[str, str] = {}
    dev_selection_detail = {}
    for task in sorted(tasks):
        dev_sks = [sk for sk in dev if ref_src[sk]["task"] == task]
        rows = {}
        for name in CHECKPOINTS:
            vals = [
                data[name][sk]["critic_conditional_rank_acc@1"] for sk in dev_sks
            ]
            rows[name] = float(np.mean(vals)) if vals else float("nan")
        best = max(CHECKPOINTS, key=lambda n: (rows[n], -CHECKPOINTS.index(n)))
        routing[task] = best
        dev_selection_detail[task] = {
            "dev_n": len(dev_sks),
            "dev_means": rows,
            "selected": best,
        }

    # ---- HOLDOUT: committee value ----
    def holdout_value(route: dict[str, str]) -> float:
        vals = [
            data[route[ref_src[sk]["task"]]][sk]["critic_conditional_rank_acc@1"]
            for sk in holdout
        ]
        return float(np.mean(vals))

    committee_val = holdout_value(routing)
    v5_val = holdout_value({t: "full" for t in routing})
    constructive_val = holdout_value(CONSTRUCTIVE_ROUTING)

    # paired bootstrap on holdout: committee(dev-routed) - V5
    base_vals = np.asarray(
        [data["full"][sk]["critic_conditional_rank_acc@1"] for sk in holdout]
    )
    comm_vals = np.asarray(
        [
            data[routing[ref_src[sk]["task"]]][sk]["critic_conditional_rank_acc@1"]
            for sk in holdout
        ]
    )
    rng_b = np.random.default_rng(BOOT_SEED)
    n = len(holdout)
    diffs = []
    for _ in range(BOOT_ITERS):
        idx = rng_b.integers(0, n, n)
        diffs.append(float(np.mean(comm_vals[idx] - base_vals[idx])))
    diffs.sort()
    lo, hi = diffs[int(0.025 * BOOT_ITERS)], diffs[int(0.975 * BOOT_ITERS) - 1]
    delta = float(np.mean(comm_vals - base_vals))

    payload = {
        "schema_version": "route_a_v3_route2_option2_committee_routing.v1",
        "protocol": {
            "split": f"stratified by task, 50/50, seed {SPLIT_SEED}",
            "selection": "dev-only per-task argmax over 4 frozen checkpoints",
            "evaluation": "holdout committee value vs single V5, paired bootstrap",
            "bootstrap": f"{BOOT_ITERS} iters seed {BOOT_SEED}",
            "registration": "script committed before probe re-run outputs were read",
        },
        "dev_n": len(dev),
        "holdout_n": len(holdout),
        "per_task_counts": {t: len(v) for t, v in sorted(tasks.items())},
        "dev_selected_routing": routing,
        "dev_selection_detail": dev_selection_detail,
        "constructive_routing": CONSTRUCTIVE_ROUTING,
        "holdout": {
            "committee_dev_routed": committee_val,
            "single_v5": v5_val,
            "constructive_routing": constructive_val,
            "delta_committee_minus_v5": delta,
            "delta_ci95": [lo, hi],
            "delta_ci_excludes_zero": not (lo <= 0.0 <= hi),
        },
        "reference_note": (
            "post-hoc full-set committee (M3 batch66) = 0.0715 vs V5 0.0614; "
            "this run replaces the post-hoc selection with a dev/holdout honest "
            "estimate; constructive routing is the zero-peek variant"
        ),
    }
    out_json = OUT_ROOT / "option2_committee_routing.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    md = [
        "# Option 2 (committee arm): preregistered dev/holdout task routing",
        "",
        f"Split: stratified by task 50/50 (seed {SPLIT_SEED}); dev {len(dev)} / holdout {len(holdout)}.",
        "",
        "## DEV-selected routing",
        "",
        "| task | selected checkpoint | dev mean |",
        "|---|---|---|",
    ]
    for task in sorted(tasks):
        d = dev_selection_detail[task]
        md.append(
            f"| {task} | {d['selected']} | "
            f"{d['dev_means'][d['selected']]:.4f} |"
        )
    md += [
        "",
        "## HOLDOUT results",
        "",
        "| configuration | holdout probe@1 |",
        "|---|---|",
        f"| single V5 (full) | {v5_val:.4f} |",
        f"| committee (dev-routed) | {committee_val:.4f} |",
        f"| committee (constructive, zero-peek) | {constructive_val:.4f} |",
        "",
        f"Committee(dev-routed) − V5 = {delta:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]",
        f"({'CI excludes zero' if lo > 0 or hi < 0 else 'CI crosses zero'}).",
        "",
        "Reference: post-hoc full-set committee 0.0715 vs V5 0.0614 (M3, batch 66)",
        "-- an upper bound; the dev/holdout numbers above are the honest estimates.",
    ]
    (OUT_ROOT / "option2_committee_routing.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
