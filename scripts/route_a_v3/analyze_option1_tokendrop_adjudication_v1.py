#!/usr/bin/env python3
"""Option-1 exploration arm final adjudication: token-dropout (v9c) vs
no-dropout control (v9b) on the 891-source mixed-pool probe.

Preregistered gate (commit 96bf35bc BEFORE training launch):
  - paired per-source bootstrap (2000 iters, seed 20260816) of
    critic_conditional_rank_acc@1 differences;
  - CI excludes zero in the POSITIVE direction => dropout arm wins;
  - crossing zero => honest null; negative => dropout hurts.

Inputs:
  persrc_probe_v9c_drop0.1.per_source.jsonl  (trained with p=0.1)
  persrc_probe_v9b_bench_control.per_source.jsonl (identical recipe, no dropout)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/"
    "analysis_option2_committee_20260909"
)
BOOT_ITERS, BOOT_SEED = 2000, 20260816


def load(name: str) -> dict[str, dict]:
    per = {}
    for line in (BASE / f"persrc_probe_{name}.per_source.jsonl").open():
        if line.strip():
            r = json.loads(line)
            per[r["source_key"]] = r
    return per


def main() -> int:
    v9c = load("v9c_drop0.1")
    v9b = load("v9b_bench_control")
    keys = sorted(
        k for k, r in v9c.items()
        if r.get("n_measured", 0) > 0 and k in v9b and v9b[k].get("n_measured", 0) > 0
    )
    assert len(keys) == 891, f"expected 891 measured sources, got {len(keys)}"

    c_vals = np.asarray([v9c[k]["critic_conditional_rank_acc@1"] for k in keys])
    b_vals = np.asarray([v9b[k]["critic_conditional_rank_acc@1"] for k in keys])
    delta = float(np.mean(c_vals - b_vals))

    rng = np.random.default_rng(BOOT_SEED)
    n = len(keys)
    diffs = []
    for _ in range(BOOT_ITERS):
        idx = rng.integers(0, n, n)
        diffs.append(float(np.mean(c_vals[idx] - b_vals[idx])))
    diffs.sort()
    lo, hi = diffs[int(0.025 * BOOT_ITERS)], diffs[int(0.975 * BOOT_ITERS) - 1]

    per_task = defaultdict(lambda: [[], []])
    for k in keys:
        t = v9c[k]["task"]
        per_task[t][0].append(v9c[k]["critic_conditional_rank_acc@1"])
        per_task[t][1].append(v9b[k]["critic_conditional_rank_acc@1"])
    pt = {}
    for t, (c, b) in sorted(per_task.items()):
        pt[t] = {
            "n": len(c),
            "v9c": float(np.mean(c)),
            "v9b_control": float(np.mean(b)),
            "delta": float(np.mean(np.asarray(c) - np.asarray(b))),
        }

    ci_excludes_zero_pos = lo > 0.0
    verdict = (
        "DROPOUT_WINS"
        if ci_excludes_zero_pos
        else ("DROPOUT_HURTS" if hi < 0.0 else "HONEST_NULL")
    )
    payload = {
        "schema_version": "route_a_v3_route2_option1_tokendrop_adjudication.v1",
        "gate_preregistered_commit": "96bf35bc",
        "comparison": "v9c token-dropout p=0.1 vs v9b no-dropout control (identical recipe, same seed 20260907)",
        "n_sources": n,
        "v9c_overall": float(np.mean(c_vals)),
        "v9b_control_overall": float(np.mean(b_vals)),
        "delta": delta,
        "delta_ci95": [lo, hi],
        "ci_excludes_zero": not (lo <= 0.0 <= hi),
        "verdict": verdict,
        "per_task": pt,
        "bootstrap": f"{BOOT_ITERS} iters seed {BOOT_SEED} paired per-source",
    }
    out = BASE / "option1_tokendrop_adjudication.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
