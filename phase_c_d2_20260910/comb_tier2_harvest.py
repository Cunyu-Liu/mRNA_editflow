#!/usr/bin/env python3
"""COMB tier-2 final harvest: 891 full, three calibres, 3-seed aggregation,
H-add/H-int/H-ind adjudication per prereg v1 sec.3 (frozen gates).
D arms: D891_main (seed base 2026091501), D891_seed16 (2026091601), D891_seed17 (2026091701).
A/B/C reference pools are frozen terminal products (reuse, no regeneration)."""
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

X = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditsetflow_v5"
D = X + "/comb_mechanism_20260911"
M = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/generation_eligibility/development_validation_v1"
A_POOL = X + "/guided_b2_20260903/b2_full_891/unguided/generated_candidates.private.jsonl"
B_POOL = X + "/beta_full891_20260909/beta_0.25_full/guided/generated_candidates.private.jsonl"
C_POOL = X + "/pool256_unguided_20260908/b2_full_891_B256/unguided/generated_candidates.private.jsonl"
D_POOLS = {
    "D_main": D + "/D891_main/guided/generated_candidates.private.jsonl",
    "D_seed16": D + "/D891_seed16/guided/generated_candidates.private.jsonl",
    "D_seed17": D + "/D891_seed17/guided/generated_candidates.private.jsonl",
}


def norm(s):
    return str(s).upper().replace("T", "U")


measured = defaultdict(set)
for line in open(M + "/measured_neighborhood.private.jsonl"):
    r = json.loads(line)
    measured[r["source_key"]].add(norm(r["candidate_sequence"]))
src_seq = {}
TASK = {}
for line in open(M + "/source_eligibility.jsonl"):
    r = json.loads(line)
    src_seq[r["source_key"]] = norm(r["source_sequence"])
    TASK[r["source_key"]] = str(r["endpoint_id"])
MRL = "MEAN_RIBOSOME_LOAD"


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


def summ(m):
    vs = [v["h1"] for v in m.values() if v["h1"] is not None]
    return {
        "support": sum(v["cov"] for v in m.values()) / len(m),
        "recovery": sum(v["rec"] for v in m.values()) / len(m),
        "sc_hit1": (sum(vs) / len(vs)) if vs else None,
        "sc_n": len(vs),
    }


A = load_pool(A_POOL)
B32 = load_pool(B_POOL)
Csel = select32(load_pool(C_POOL))
mains = {}
rows = {}
mains["A_baseline"] = per_source(A)
mains["B_guide"] = per_source(B32)
mains["C_explore"] = per_source(Csel)
rows["A_baseline"] = summ(mains["A_baseline"])
rows["B_guide"] = summ(mains["B_guide"])
rows["C_explore"] = summ(mains["C_explore"])
for tag, path in D_POOLS.items():
    if not Path(path).exists():
        print(f"SKIP missing pool: {tag} {path}", file=sys.stderr)
        continue
    gen = select32(load_pool(path))
    m = per_source(gen)
    mains[tag] = m
    rows[tag] = summ(m)
    d_sc = [(m[k]["h1"] or 0.0) - (mains["A_baseline"].get(k, {}).get("h1") or 0.0) for k in m]
    d_sup = [m[k]["cov"] - mains["A_baseline"].get(k, {}).get("cov", 0.0) for k in m]
    rows[tag]["d_sc_hit1"] = sum(d_sc) / len(d_sc)
    rows[tag]["d_sc_hit1_ci95"] = bci(d_sc)
    rows[tag]["d_support"] = sum(d_sup) / len(d_sup)


def paired_delta(mX, mY, field="h1", default=0.0):
    ks = [k for k in mX if k in mY]
    return [(mX[k][field] if mX[k][field] is not None else default) - (mY[k][field] if mY[k][field] is not None else default) for k in ks]


for tag in ("B_guide", "C_explore"):
    d_sc = paired_delta(mains[tag], mains["A_baseline"])
    d_sup = paired_delta(mains[tag], mains["A_baseline"], "cov", 0.0)
    rows[tag]["d_sc_hit1"] = sum(d_sc) / len(d_sc)
    rows[tag]["d_sc_hit1_ci95"] = bci(d_sc)
    rows[tag]["d_support"] = sum(d_sup) / len(d_sup)


# ---- Optimus independent calibre (MRL top-1 delta) ----
optimus_status = "NOT_RUN"
try:
    import numpy as np
    import torch

    V8 = Path("/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_v8_stage1_prep_20260904")
    HARNESS = V8 / "scripts/route_a_v3/run_route2_external_prediction_baselines_v1.py"
    OPT_W = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/optimus5prime/main_MRL_model.hdf5")
    spec2 = importlib.util.spec_from_file_location("harness", str(HARNESS))
    harness = importlib.util.module_from_spec(spec2)
    sys.modules["harness"] = harness
    spec2.loader.exec_module(harness)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pools = {"A_baseline": A, "B_guide": B32, "C_explore": Csel}
    pools.update({tag: select32(load_pool(p)) for tag, p in D_POOLS.items() if Path(p).exists()})
    opt = harness.Optimus5Prime(OPT_W).to(device).eval()

    def top1_delta_scores(gen):
        res = {}
        mrl = [sk for sk in gen if TASK.get(sk) == MRL]
        batch, bkeys = [], []

        def flush():
            oh = []
            for s in batch:
                t = s.replace("U", "T")
                row = np.zeros((len(t), 4), dtype=np.float32)
                for i, ch in enumerate(t):
                    row[i, "ACGT".index(ch) if ch in "ACGT" else 0] = 1.0
                oh.append(row)
            L = max(x.shape[0] for x in oh)
            arr = np.zeros((len(oh), L, 4), dtype=np.float32)
            for i, x in enumerate(oh):
                arr[i, : x.shape[0]] = x
            t = torch.from_numpy(arr).to(device)
            with torch.inference_mode():
                preds = opt(t).squeeze(-1).float().cpu().numpy().tolist()
            for i, sk in enumerate(bkeys):
                res[sk] = preds[2 * i + 1] - preds[2 * i]

        for sk in mrl:
            ranked = sorted(gen[sk].items(), key=lambda kv: -kv[1])
            top1 = ranked[0][0]
            batch.extend([src_seq[sk], top1])
            bkeys.append(sk)
            if len(batch) >= 64:
                flush()
                batch, bkeys = [], []
        if batch:
            flush()
        return res

    oscores = {tag: top1_delta_scores(g) for tag, g in pools.items()}
    rows["A_baseline"]["optimus_delta"] = sum(oscores["A_baseline"].values()) / len(oscores["A_baseline"])
    for tag in ("B_guide", "C_explore", "D_main", "D_seed16", "D_seed17"):
        if tag not in oscores:
            continue
        d = [oscores[tag][k] - oscores["A_baseline"][k] for k in oscores[tag] if k in oscores["A_baseline"]]
        if d:
            rows[tag]["d_optimus"] = sum(d) / len(d)
            rows[tag]["d_optimus_ci95"] = bci(d)
    optimus_status = "OK"
except Exception as e:  # noqa: BLE001
    optimus_status = f"OPTIMUS_FAILED: {e}"


# ---- 3-seed aggregation + H-add/H-int/H-ind adjudication ----
def arm_gates(tag):
    r = rows.get(tag, {})
    d_sc = r.get("d_sc_hit1")
    ci_sc = r.get("d_sc_hit1_ci95")
    d_op = r.get("d_optimus")
    ci_op = r.get("d_optimus_ci95")
    g_sc = d_sc is not None and ci_sc is not None and d_sc >= 0.03 and ci_sc[1] > 0
    g_op = d_op is not None and ci_op is not None and d_op >= 0.02 and ci_op[1] > 0
    return {"d_sc_hit1": d_sc, "ci_sc": ci_sc, "sc_gate": g_sc, "d_optimus": d_op, "ci_op": ci_op, "optimus_gate": g_op}


aggregation = {tag: arm_gates(tag) for tag in ("D_main", "D_seed16", "D_seed17", "B_guide", "C_explore")}

d_tags = [t for t in ("D_main", "D_seed16", "D_seed17") if t in rows]
verdict = {"n_d_arms": len(d_tags)}
if d_tags:
    sc_pass = sum(1 for t in d_tags if aggregation[t]["sc_gate"])
    op_pass = sum(1 for t in d_tags if aggregation[t]["optimus_gate"])
    verdict["sc_gate_pass_count"] = sc_pass
    verdict["optimus_gate_pass_count"] = op_pass
    m_sc = sum(rows[t]["d_sc_hit1"] for t in d_tags) / len(d_tags)
    m_op = sum(rows[t]["d_optimus"] for t in d_tags if "d_optimus" in rows[t]) / max(1, sum(1 for t in d_tags if "d_optimus" in rows[t]))
    verdict["mean_d_sc_hit1"] = m_sc
    verdict["mean_d_optimus"] = m_op
    # H-int: D significantly BELOW corresponding single arm (paired delta-delta CI excludes zero on negative side)
    if "C_explore" in rows:
        dd_sc = []
        for t in d_tags:
            mD = mains[t]
            mC = mains["C_explore"]
            ks = [k for k in mD if k in mC]
            dd_sc = [(mD[k]["h1"] or 0.0) - (mC[k]["h1"] or 0.0) for k in ks]
        if dd_sc:
            verdict["dd_sc_D_vs_C_mean"] = sum(dd_sc) / len(dd_sc)
            verdict["dd_sc_D_vs_C_ci95"] = bci(dd_sc)
    # H-add fingerprint: B+C~D (point estimates)
    if "B_guide" in rows and "C_explore" in rows:
        verdict["additivity_fingerprint"] = {
            "B_plus_C_d_sc": rows["B_guide"].get("d_sc_hit1"), "D_mean_d_sc": m_sc,
            "B_plus_C_d_optimus": (rows["B_guide"].get("d_optimus", 0.0) or 0.0) + (rows["C_explore"].get("d_optimus", 0.0) or 0.0),
            "D_mean_d_optimus": m_op,
        }
    h_add = sc_pass == len(d_tags) and op_pass == len(d_tags)
    h_int = False
    if "dd_sc_D_vs_C_ci95" in verdict:
        h_int = verdict["dd_sc_D_vs_C_ci95"][1] < 0
    verdict["hypothesis_verdict"] = "H_ADD" if h_add else ("H_INT" if h_int else "H_IND_or_NEGATIVE")

out = {
    "schema_version": "route_a_v3_comb_tier2_harvest.v1",
    "gate": "prereg v1 sec.3: H-add = D sc-gate >= +0.03 CI>0 AND optimus-gate >= +0.02 CI>0 (all arms); H-int = D significantly below single arm",
    "optimus_status": optimus_status,
    "rows": rows,
    "aggregation": aggregation,
    "verdict": verdict,
}
Path(D + "/comb_tier2_harvest.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out["verdict"], indent=1))
