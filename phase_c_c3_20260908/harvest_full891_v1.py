#!/usr/bin/env python3
"""C2 full-891 confirmation harvest (beta=0.25 guided vs unguided, 891 sources).

Runs when beta_full891_20260909/beta_0.25_full/guided_run_summary.json exists.
Same logic as phase_c_c3_20260908/harvest_beta_sweep.py (sc-hit@1 tie-aware,
support@32, recovery@budget, paired source-level bootstrap 2000 iters seed
20260816) but on the full 891-source cohort with the b2_full_891 unguided
baseline. Adjudication: amendment B-tier full gates
(docs/paper/route2_setflow_b3_amendment_sc_hit1_v1.md):
  - B2 new primary: d_sc_hit1 >= +0.03 AND CI excludes zero
  - B3 absolute: sc_hit1 >= 0.10 AND >= 2x base line (natural-hit base 0.1514
    -> 2x = 0.3028 on the base-comparison calibre; the per-cohort 2x line is
    computed from the unguided arm itself when available)
  - support co-reported (guard against denominator inflation)
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

X = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditsetflow_v5")
FULL = X / "beta_full891_20260909/beta_0.25_full"
UNGU = X / "guided_b2_20260903/b2_full_891/unguided/generated_candidates.private.jsonl"
MEAS = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/generation_eligibility/"
            "development_validation_v1/measured_neighborhood.private.jsonl")
OUT = X / "beta_full891_20260909/full891_harvest.json"
BETA = "0.25"
BOOT_ITERS, BOOT_SEED = 2000, 20260816


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


def load_measured():
    by_src = defaultdict(set)
    for line in open(MEAS):
        r = json.loads(line)
        by_src[str(r["source_key"])].add(norm(r["candidate_sequence"]))
    return by_src


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


def paired_bootstrap(deltas_support, deltas_hit, valid_pairs):
    rng = random.Random(BOOT_SEED)
    n = len(valid_pairs)
    ds, dh = [], []
    for _ in range(BOOT_ITERS):
        idx = [rng.randrange(n) for _ in range(n)]
        s = sum(deltas_support[i] for i in idx) / n
        h = sum(deltas_hit[i] for i in idx) / n
        ds.append(s)
        dh.append(h)
    def ci(v):
        v = sorted(v)
        return [v[int(0.025 * len(v))], v[int(0.975 * len(v))]]
    return ci(ds), ci(dh)


def main() -> int:
    summary = json.loads((FULL / "guided_run_summary.json").read_text())
    guided_keys = set()
    for line in (FULL / "guided" / "generated_candidates.private.jsonl").open():
        guided_keys.add(json.loads(line)["source_key"])
    meas = load_measured()
    ungu = load_pool(UNGU, keys=guided_keys)
    guid = load_pool(FULL / "guided" / "generated_candidates.private.jsonl")

    mu = per_source_metrics(ungu, meas)
    mg = per_source_metrics(guid, meas)
    shared = sorted(set(mu) & set(mg))
    print(f"shared sources: {len(shared)} (guided {len(mg)}, unguided {len(mu)})")

    n = len(shared)
    sup_u = sum(mu[s]["support"] for s in shared) / n
    sup_g = sum(mg[s]["support"] for s in shared) / n
    rec_u = sum(mu[s]["recovery"] for s in shared) / n
    rec_g = sum(mg[s]["recovery"] for s in shared) / n
    # sc-hit@1: conditional on S+ (supported sources in EITHER arm per harvest convention:
    # the sweep harvest used the guided arm's covered set; here use union for symmetry)
    valid_pairs, d_sup, d_hit = [], [], []
    hit_vals_u, hit_vals_g = [], []
    for s in shared:
        d_sup.append(mg[s]["support"] - mu[s]["support"])
        hu, hg = mu[s]["sc_hit1"], mg[s]["sc_hit1"]
        if hg is not None:
            hit_vals_g.append(hg)
        if hu is not None:
            hit_vals_u.append(hu)
        if hg is not None and hu is not None:
            valid_pairs.append(s)
            d_hit.append(hg - hu)
    sc_u = sum(hit_vals_u) / len(hit_vals_u) if hit_vals_u else None
    sc_g = sum(hit_vals_g) / len(hit_vals_g) if hit_vals_g else None
    paired_u = [mu[s]["sc_hit1"] for s in valid_pairs]
    paired_g = [mg[s]["sc_hit1"] for s in valid_pairs]
    sc_paired_u = sum(paired_u) / len(paired_u)
    sc_paired_g = sum(paired_g) / len(paired_g)

    ci_sup, ci_hit = paired_bootstrap(d_sup, d_hit, list(range(len(shared))))

    # gates
    d_hit_point = sc_paired_g - sc_paired_u
    b2_primary = d_hit_point >= 0.03 and ci_hit[0] > 0
    base_2x = sc_paired_u * 2
    b3_abs = sc_g is not None and sc_g >= 0.10 and sc_g >= base_2x

    payload = {
        "schema_version": "route_a_v3_phase_c_full891_harvest.v1",
        "beta": BETA,
        "n_sources": n,
        "n_sc_hit_pairs": len(valid_pairs),
        "unguided": {"support": sup_u, "recovery": rec_u, "sc_hit1_uncond": sc_u,
                     "sc_hit1_paired": sc_paired_u, "n_hit": len(hit_vals_u)},
        "guided_beta0.25": {"support": sup_g, "recovery": rec_g, "sc_hit1_uncond": sc_g,
                            "sc_hit1_paired": sc_paired_g, "n_hit": len(hit_vals_g)},
        "deltas": {
            "support_point": sup_g - sup_u, "support_ci95": ci_sup,
            "sc_hit1_point_paired": d_hit_point, "sc_hit1_ci95": ci_hit,
        },
        "gates_amendment_B": {
            "B2_primary_d_sc_hit1_ge_0.03_CI_excludes_zero": b2_primary,
            "B3_absolute_sc_hit1_ge_0.10_and_2x_base": b3_abs,
            "2x_base_line": base_2x,
            "support_co_reported": {"point": sup_g - sup_u, "ci95": ci_sup},
            "note": "B3 2x base line uses the unguided arm's paired sc-hit@1 on this cohort (per-cohort, stricter than the 0.1514 natural-hit reference); independent-evaluator dual calibre (frozen Optimus/APARENT delta on top-1) still to be attached per amendment 3.3.",
        },
        "run_summary": {
            "status": summary.get("status"),
            "wall_min": round(summary.get("wall_time_seconds", 0) / 60, 1),
            "cpu_fallback_used": summary.get("cpu_fallback_used"),
        },
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload["deltas"], indent=1))
    print("B2 primary gate:", b2_primary, "| B3 absolute gate:", b3_abs)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
