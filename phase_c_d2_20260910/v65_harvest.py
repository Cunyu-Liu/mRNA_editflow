#!/usr/bin/env python3
"""V6.5 explore-then-select final harvest: all arms, all amendment-v2 gates."""
import json
import random
from collections import defaultdict

X = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditsetflow_v5"
D = X + "/v65_explore_select_20260910"
M = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/generation_eligibility/development_validation_v1"
UNGU_B32 = X + "/guided_b2_20260903/b2_full_891/unguided/generated_candidates.private.jsonl"
B256_MAIN = X + "/pool256_unguided_20260908/b2_full_891_B256/unguided/generated_candidates.private.jsonl"
ARMS = {
    "B256_main": B256_MAIN,
    "B256_seed16": D + "/B256_seed16/unguided/generated_candidates.private.jsonl",
    "B256_seed17": D + "/B256_seed17/unguided/generated_candidates.private.jsonl",
    "B512": D + "/B512_unguided/unguided/generated_candidates.private.jsonl",
    "B1024": D + "/B1024_unguided/unguided/generated_candidates.private.jsonl",
}


def norm(s):
    return str(s).upper().replace("T", "U")


measured = defaultdict(set)
for line in open(M + "/measured_neighborhood.private.jsonl"):
    r = json.loads(line)
    measured[r["source_key"]].add(norm(r["candidate_sequence"]))


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


def select32(gen):
    out = {}
    for sk, seqs in gen.items():
        ranked = sorted(seqs.items(), key=lambda kv: -kv[1])[:32]
        out[sk] = dict(ranked)
    return out


def per_source(gen):
    out = {}
    for sk, seqs in gen.items():
        ms = measured.get(sk, set())
        covered = [q for q in ms if q in seqs]
        h1 = None
        if covered:
            ranked = sorted(seqs.items(), key=lambda kv: -kv[1])
            top = ranked[0][1]
            block = [sq for sq, s in ranked if s == top]
            tie = len([sq for sq in block if sq in ms])
            h1 = tie / len(block)
        out[sk] = {"cov": 1.0 if covered else 0.0, "rec": len(covered) / len(ms) if ms else 0.0, "h1": h1}
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


base = per_source(load_pool(UNGU_B32))
rows = {}
for tag, path in ARMS.items():
    try:
        m = per_source(select32(load_pool(path)))
    except FileNotFoundError:
        rows[tag] = {"status": "MISSING"}
        continue
    vs = [v["h1"] for v in m.values() if v["h1"] is not None]
    d_sc = [
        (m[k]["h1"] or 0.0) - (base.get(k, {}).get("h1") or 0.0)
        for k in m
    ]
    d_sup = [m[k]["cov"] - base.get(k, {}).get("cov", 0.0) for k in m]
    d_rec = [m[k]["rec"] - base.get(k, {}).get("rec", 0.0) for k in m]
    ci = bci(d_sc)
    pt = sum(d_sc) / len(d_sc)
    rows[tag] = {
        "status": "OK",
        "support": sum(v["cov"] for v in m.values()) / len(m),
        "recovery": sum(v["rec"] for v in m.values()) / len(m),
        "sc_hit1_mean": (sum(vs) / len(vs)) if vs else None,
        "sc_hit1_n": len(vs),
        "hit1_uncond": sum(v["h1"] or 0.0 for v in m.values()) / len(m),
        "d_sc_hit1_point": pt,
        "d_sc_hit1_ci95": ci,
        "b2_gate_pass": bool(pt >= 0.03 and ci[0] > 0),
        "b3_absolute_055": bool((sum(vs) / len(vs)) if vs else 0 >= 0.55),
        "d_support_point": sum(d_sup) / len(d_sup),
        "d_support_ci95": bci(d_sup),
        "d_recovery_point": sum(d_rec) / len(d_rec),
        "d_recovery_ci95": bci(d_rec),
    }

seeds_ok = all(
    rows.get(t, {}).get("b2_gate_pass") for t in ("B256_main", "B256_seed16", "B256_seed17")
)
out = {
    "schema_version": "route_a_v3_v65_explore_select_harvest.v1",
    "gate": "amendment v2: B2 delta (>=+0.03 CI excl zero) x 3-seed robust; B3 absolute 0.55; B2-I independent deferred to session review",
    "three_seed_robust_pass": seeds_ok,
    "rows": rows,
}
from pathlib import Path

Path(D + "/v65_harvest.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
