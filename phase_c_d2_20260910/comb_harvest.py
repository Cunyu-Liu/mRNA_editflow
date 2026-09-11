#!/usr/bin/env python3
"""COMB mechanism harvest: 2x2 matrix (A baseline / B guide / C explore / D comb)
x three calibres (recovery family + sc-hit@1 + Optimus independent), tier-1
calib100 quick adjudication."""
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

X = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditsetflow_v5"
D = X + "/comb_mechanism_20260911"
M = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/generation_eligibility/development_validation_v1"
KEYS = X + "/beta_sweep_20260908/calib100_keys.txt"

A_POOL = X + "/guided_b2_20260903/b2_full_891/unguided/generated_candidates.private.jsonl"
B_POOL = X + "/beta_full891_20260909/beta_0.25_full/guided/generated_candidates.private.jsonl"
C_POOL = X + "/pool256_unguided_20260908/b2_full_891_B256/unguided/generated_candidates.private.jsonl"
D_POOL = D + "/calib100_D_comb/guided/generated_candidates.private.jsonl"


def norm(s):
    return str(s).upper().replace("T", "U")


keys = set(x.strip() for x in open(KEYS) if x.strip())
measured = defaultdict(set)
for line in open(M + "/measured_neighborhood.private.jsonl"):
    r = json.loads(line)
    if r["source_key"] in keys:
        measured[r["source_key"]].add(norm(r["candidate_sequence"]))
src_seq = {}
TASK = {}
for line in open(M + "/source_eligibility.jsonl"):
    r = json.loads(line)
    src_seq[r["source_key"]] = norm(r["source_sequence"])
    TASK[r["source_key"]] = str(r["endpoint_id"])
MRL = "MEAN_RIBOSOME_LOAD"


def load(path, restrict=True):
    gen = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        sk = r["source_key"]
        if restrict and sk not in keys:
            continue
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


def rec_metrics(gen):
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


# ---- recovery family ----
A = load(A_POOL)
B32 = load(B_POOL)  # B arm: guided beta 0.25 B=32, submitted as-is
Csel = select32(load(C_POOL, restrict=False))
Csel = {k: v for k, v in Csel.items() if k in keys}
Dsel = select32(load(D_POOL, restrict=False))
Dsel = {k: v for k, v in Dsel.items() if k in keys}

mA = rec_metrics(A)
mB = rec_metrics(B32)
mC = rec_metrics(Csel)
mD = rec_metrics(Dsel)


def summ(m):
    vs = [v["h1"] for v in m.values() if v["h1"] is not None]
    return {
        "support": sum(v["cov"] for v in m.values()) / len(m),
        "recovery": sum(v["rec"] for v in m.values()) / len(m),
        "sc_hit1": (sum(vs) / len(vs)) if vs else None,
        "sc_n": len(vs),
    }


rows = {"A_baseline": summ(mA), "B_guide": summ(mB), "C_explore": summ(mC), "D_comb": summ(mD)}
for tag, m in (("B_guide", mB), ("C_explore", mC), ("D_comb", mD)):
    d_sc = [(m[k]["h1"] or 0.0) - (mA.get(k, {}).get("h1") or 0.0) for k in m]
    d_sup = [m[k]["cov"] - mA.get(k, {}).get("cov", 0.0) for k in m]
    rows[tag]["d_sc_hit1"] = sum(d_sc) / len(d_sc)
    rows[tag]["d_sc_hit1_ci95"] = bci(d_sc)
    rows[tag]["d_support"] = sum(d_sup) / len(d_sup)

# ---- Optimus independent calibre ----
try:
    import numpy as np
    import torch

    V8 = Path("/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_v8_stage1_prep_20260904")
    HARNESS = V8 / "scripts/route_a_v3/run_route2_external_prediction_baselines_v1.py"
    OPT_W = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/optimus5prime/main_MRL_model.hdf5")
    spec2 = importlib.util.spec_from_file_location("harness", str(HARNESS))
    harness = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(harness)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    oA = top1_delta_scores(A)
    oB = top1_delta_scores(B32)
    oC = top1_delta_scores(Csel)
    oD = top1_delta_scores(Dsel)
    rows["A_baseline"]["optimus_delta"] = sum(oA.values()) / len(oA)
    for tag, o in (("B_guide", oB), ("C_explore", oC), ("D_comb", oD)):
        d = [o[k] - oA[k] for k in o if k in oA]
        rows[tag]["d_optimus"] = sum(d) / len(d) if d else None
        rows[tag]["d_optimus_ci95"] = bci(d) if d else None
    out_opt = "OK"
except Exception as e:  # noqa: BLE001
    out_opt = f"OPTIMUS_FAILED: {e}"

out = {
    "schema_version": "route_a_v3_comb_tier1_harvest.v1",
    "gate": "prereg: positive direction in any calibre -> 891 + 3 seeds; both negative -> closure report",
    "optimus_status": out_opt,
    "rows": rows,
}
Path(D + "/comb_tier1_harvest.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
