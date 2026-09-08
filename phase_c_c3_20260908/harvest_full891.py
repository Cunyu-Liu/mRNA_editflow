#!/usr/bin/env python3
"""Full-891 beta confirmation harvest (auto-invoked by watcher).

Both arms (beta 0.25, 0.5) vs unguided B=32 baseline on the full 891 cohort:
support / recovery / sc-hit@1 + per-source paired bootstrap CIs.
Amendment B-tier gate pointwise flags.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

X = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditsetflow_v5")
D = X / "beta_full891_20260909"
UNGU = X / "guided_b2_20260903/b2_full_891/unguided/generated_candidates.private.jsonl"
MEAS = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/generation_eligibility/"
            "development_validation_v1/measured_neighborhood.private.jsonl")
BETAS = ["0.25", "0.5"]


def norm(s):
    return str(s).upper().replace("T", "U")


def load_pool(path):
    gen = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        sk = r["source_key"]
        sq = norm(r["candidate_sequence"])
        s = float(r.get("generation_score") or 0.0)
        if sq not in gen[sk] or gen[sk][sq] < s:
            gen[sk][sq] = s
    return gen


def per_source_metrics(gen, measured):
    out = {}
    for sk, seqs in gen.items():
        ms = measured.get(sk, set())
        covered = [q for q in ms if q in seqs]
        hit1 = 0.0
        if covered:
            ranked = sorted(seqs.items(), key=lambda kv: -kv[1])
            top = ranked[0][1]
            block = [sq for sq, s in ranked if s == top]
            tie = len([sq for sq in block if sq in ms])
            hit1 = tie / len(block)
        out[sk] = {
            "support": 1.0 if covered else 0.0,
            "recovery": (len(covered) / len(ms)) if ms else 0.0,
            "sc_hit1": hit1 if covered else None,
        }
    return out


def bootstrap_ci(deltas, iters=2000, seed=20260816):
    rng = random.Random(seed)
    n = len(deltas)
    if n == 0:
        return None
    means = []
    for _ in range(iters):
        s = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / len(s))
    means.sort()
    return [means[int(0.025 * iters)], means[int(0.975 * iters)]]


def main():
    measured = defaultdict(set)
    for line in open(MEAS):
        r = json.loads(line)
        measured[r["source_key"]].add(norm(r["candidate_sequence"]))
    base = per_source_metrics(load_pool(UNGU), measured)
    base_sc = [v["sc_hit1"] for v in base.values() if v["sc_hit1"] is not None]
    rows = {
        "baseline_unguided_B32_full891": {
            "n": len(base),
            "support": sum(v["support"] for v in base.values()) / len(base),
            "recovery": sum(v["recovery"] for v in base.values()) / len(base),
            "sc_hit1_mean": (sum(base_sc) / len(base_sc)) if base_sc else None,
            "sc_hit1_n": len(base_sc),
        }
    }
    for b in BETAS:
        p = D / f"beta_{b}_full" / "guided" / "generated_candidates.private.jsonl"
        if not p.exists():
            rows[f"beta_{b}"] = {"status": "MISSING"}
            continue
        m = per_source_metrics(load_pool(p), measured)
        vs = [v["sc_hit1"] for v in m.values() if v["sc_hit1"] is not None]
        d_sc = [
            (m[k]["sc_hit1"] or 0.0) - (base.get(k, {}).get("sc_hit1") or 0.0)
            for k in m
            if (m[k]["sc_hit1"] is not None or base.get(k, {}).get("sc_hit1") is not None)
        ]
        d_sup = [m[k]["support"] - base.get(k, {}).get("support", 0.0) for k in m]
        rows[f"beta_{b}"] = {
            "status": "OK",
            "n": len(m),
            "support": sum(v["support"] for v in m.values()) / len(m),
            "recovery": sum(v["recovery"] for v in m.values()) / len(m),
            "sc_hit1_mean": (sum(vs) / len(vs)) if vs else None,
            "sc_hit1_n": len(vs),
            "d_support_point": sum(d_sup) / len(d_sup),
            "d_support_ci95": bootstrap_ci(d_sup),
            "d_sc_hit1_point": (sum(d_sc) / len(d_sc)) if d_sc else None,
            "d_sc_hit1_ci95": bootstrap_ci(d_sc) if d_sc else None,
        }
    out = {
        "schema_version": "route_a_v3_phase_c_full891_confirm_harvest.v1",
        "cohort": "full 891",
        "gate": "amendment B-tier (B2 delta gate authoritative; B3 absolute line needs amendment v2 reference fix)",
        "rows": rows,
    }
    (D / "full891_confirm_harvest.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
