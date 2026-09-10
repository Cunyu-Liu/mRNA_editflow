#!/usr/bin/env python3
"""D2 calib100 harvest: retrieval arms vs unguided B=32 baseline (calib100),
amendment v2 gates (B2 delta sc-hit@1 + B2-I independent MRL Optimus deferred
to 891 tier; calib tier adjudicates B2 delta only per prereg two-tier)."""
import json
import random
import sys
from collections import defaultdict

X = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditsetflow_v5"
D = X + "/d2_retrieval_20260910"
MEAS = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/generation_eligibility/development_validation_v1/measured_neighborhood.private.jsonl"
UNGU = X + "/guided_b2_20260903/b2_full_891/unguided/generated_candidates.private.jsonl"
KEYS = X + "/beta_sweep_20260908/calib100_keys.txt"
BETAS = ["0.25", "0.5", "1", "2"]


def norm(s):
    return str(s).upper().replace("T", "U")


measured = defaultdict(set)
for line in open(MEAS):
    r = json.loads(line)
    measured[r["source_key"]].add(norm(r["candidate_sequence"]))
keys = set(x.strip() for x in open(KEYS) if x.strip())


def per_source(path):
    gen = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        sk = r["source_key"]
        if sk not in keys:
            continue
        sq = norm(r["candidate_sequence"])
        s = float(r.get("generation_score") or 0.0)
        if sq not in gen[sk] or gen[sk][sq] < s:
            gen[sk][sq] = s
    out = {}
    for sk, seqs in gen.items():
        ms = measured.get(sk, set())
        covered = [q for q in ms if q in seqs]
        h1 = 0.0
        if covered:
            ranked = sorted(seqs.items(), key=lambda kv: -kv[1])
            top = ranked[0][1]
            block = [sq for sq, s in ranked if s == top]
            tie = len([sq for sq in block if sq in ms])
            h1 = tie / len(block)
        out[sk] = {"support": 1.0 if covered else 0.0, "sc_hit1": h1 if covered else None}
    return out


def bci(d, iters=2000, seed=20260816):
    rng = random.Random(seed)
    n = len(d)
    means = []
    for _ in range(iters):
        s = [d[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / len(s))
    means.sort()
    return [means[int(0.025 * iters)], means[int(0.975 * iters)]]


base = per_source(UNGU)
rows = {}
bvs = [v["sc_hit1"] for v in base.values() if v["sc_hit1"] is not None]
rows["baseline_unguided_B32_calib100"] = {
    "support": sum(v["support"] for v in base.values()) / len(base),
    "sc_hit1_mean": (sum(bvs) / len(bvs)) if bvs else None,
    "sc_hit1_n": len(bvs),
}
for b in BETAS:
    p = D + f"/calib100_retr_beta_{b}/guided/generated_candidates.private.jsonl"
    try:
        m = per_source(p)
    except FileNotFoundError:
        rows[f"retr_beta_{b}"] = {"status": "MISSING"}
        continue
    vs = [v["sc_hit1"] for v in m.values() if v["sc_hit1"] is not None]
    d_sc = [
        (m[k]["sc_hit1"] or 0.0) - (base.get(k, {}).get("sc_hit1") or 0.0)
        for k in m
        if (m[k]["sc_hit1"] is not None or base.get(k, {}).get("sc_hit1") is not None)
    ]
    d_sup = [m[k]["support"] - base.get(k, {}).get("support", 0.0) for k in m]
    rows[f"retr_beta_{b}"] = {
        "status": "OK",
        "support": sum(v["support"] for v in m.values()) / len(m),
        "sc_hit1_mean": (sum(vs) / len(vs)) if vs else None,
        "sc_hit1_n": len(vs),
        "d_support_point": sum(d_sup) / len(d_sup),
        "d_support_ci95": bci(d_sup),
        "d_sc_hit1_point": (sum(d_sc) / len(d_sc)) if d_sc else None,
        "d_sc_hit1_ci95": bci(d_sc) if d_sc else None,
        "b2_delta_gate_pass": bool(
            d_sc and (sum(d_sc) / len(d_sc)) >= 0.03 and bci(d_sc)[0] > 0
        ),
    }
out = {
    "schema_version": "route_a_v3_d2_calib_harvest.v1",
    "gate": "amendment v2 B2 delta gate (calib tier); B2-I independent gate deferred to 891 tier",
    "rows": rows,
}
(D + "/d2_calib_harvest.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
