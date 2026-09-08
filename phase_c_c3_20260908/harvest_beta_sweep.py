#!/usr/bin/env python3
"""C2 beta-sweep harvest (auto-invoked by watcher when all 5 arms terminal).

Computes per-beta-arm (calib100 cohort, guided B=32):
  - support@32 (measured-in-pool rate), recovery@budget
  - sc-hit@1 (tie-aware top-1 admission on S+ sources, amendment criterion)
  - paired per-source deltas vs the unguided B=32 baseline (same cohort)
  - source-level paired bootstrap CI (2000 iters, seed 20260816) for
    d(sc-hit@1) and d(support)
B-tier gate flags recorded pointwise; final adjudication reviewed by session.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

X = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditsetflow_v5")
SW = X / "beta_sweep_20260908"
UNGU = X / "guided_b2_20260903/b2_full_891/unguided/generated_candidates.private.jsonl"
MEAS = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/generation_eligibility/"
            "development_validation_v1/measured_neighborhood.private.jsonl")
BETAS = ["0.25", "0.5", "1", "2", "4"]


def norm(s):
    return str(s).upper().replace("T", "U")


def load_pool(path, keys=None):
    gen = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        sk = r["source_key"]
        if keys is not None and sk not in keys:
            continue
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
    keys = set(x.strip() for x in open(SW / "calib100_keys.txt") if x.strip())
    measured = defaultdict(set)
    for line in open(MEAS):
        r = json.loads(line)
        if r["source_key"] in keys:
            measured[r["source_key"]].add(norm(r["candidate_sequence"]))

    base = per_source_metrics(load_pool(UNGU, keys), measured)
    base_row = {
        "n": len(base),
        "support": sum(v["support"] for v in base.values()) / len(base),
        "recovery": sum(v["recovery"] for v in base.values()) / len(base),
        "sc_hit1_mean": (lambda vs: sum(vs) / len(vs) if vs else None)(
            [v["sc_hit1"] for v in base.values() if v["sc_hit1"] is not None]
        ),
        "sc_hit1_n": len([v for v in base.values() if v["sc_hit1"] is not None]),
    }

    rows = {"baseline_unguided_B32_calib100": base_row}
    for b in BETAS:
        p = SW / f"calib100_beta_{b}" / "guided" / "generated_candidates.private.jsonl"
        if not p.exists():
            rows[f"beta_{b}"] = {"status": "MISSING"}
            continue
        m = per_source_metrics(load_pool(p, keys), measured)
        vs = [v["sc_hit1"] for v in m.values() if v["sc_hit1"] is not None]
        sc = sum(vs) / len(vs) if vs else None
        sup = sum(v["support"] for v in m.values()) / len(m)
        rec = sum(v["recovery"] for v in m.values()) / len(m)
        d_sc = [
            (m[k]["sc_hit1"] or 0.0) - (base.get(k, {}).get("sc_hit1") or 0.0)
            for k in m if (m[k]["sc_hit1"] is not None or base.get(k, {}).get("sc_hit1") is not None)
        ]
        d_sup = [m[k]["support"] - base.get(k, {}).get("support", 0.0) for k in m]
        rows[f"beta_{b}"] = {
            "status": "OK",
            "n": len(m),
            "support": sup,
            "recovery": rec,
            "sc_hit1_mean": sc,
            "sc_hit1_n": len(vs),
            "d_support_point": sum(d_sup) / len(d_sup),
            "d_support_ci95": bootstrap_ci(d_sup),
            "d_sc_hit1_point": (sum(d_sc) / len(d_sc)) if d_sc else None,
            "d_sc_hit1_ci95": bootstrap_ci(d_sc) if d_sc else None,
        }
    out = {
        "schema_version": "route_a_v3_phase_c_beta_sweep_harvest.v1",
        "cohort": "calib100 (74/12/12/2, seed 20260908)",
        "gate": "amendment B-tier (point flags; final adjudication in-session)",
        "rows": rows,
    }
    (SW / "sweep_harvest.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
