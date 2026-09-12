#!/usr/bin/env python3
"""COMB tier-1 Optimus calibre (standalone, fixed module loading)."""
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

X = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditsetflow_v5"
D = X + "/comb_mechanism_20260911"
M = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/generation_eligibility/development_validation_v1"
KEYS = X + "/beta_sweep_20260908/calib100_keys.txt"

V8 = Path("/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_v8_stage1_prep_20260904")
HARNESS = V8 / "scripts/route_a_v3/run_route2_external_prediction_baselines_v1.py"
OPT_W = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/optimus5prime/main_MRL_model.hdf5")

spec2 = importlib.util.spec_from_file_location("harness", str(HARNESS))
harness = importlib.util.module_from_spec(spec2)
sys.modules["harness"] = harness
spec2.loader.exec_module(harness)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
opt = harness.Optimus5Prime(OPT_W).to(device).eval()

keys = set(x.strip() for x in open(KEYS) if x.strip())


def norm(s):
    return str(s).upper().replace("T", "U")


src_seq = {}
TASK = {}
for line in open(M + "/source_eligibility.jsonl"):
    r = json.loads(line)
    src_seq[r["source_key"]] = norm(r["source_sequence"])
    TASK[r["source_key"]] = str(r["endpoint_id"])
MRL = "MEAN_RIBOSOME_LOAD"

POOLS = {
    "A_baseline": X + "/guided_b2_20260903/b2_full_891/unguided/generated_candidates.private.jsonl",
    "B_guide": X + "/beta_full891_20260909/beta_0.25_full/guided/generated_candidates.private.jsonl",
    "C_explore": X + "/pool256_unguided_20260908/b2_full_891_B256/unguided/generated_candidates.private.jsonl",
    "D_comb": D + "/calib100_D_comb/guided/generated_candidates.private.jsonl",
}
# selection modes: A/B submitted as-is (B32 pools), C/D select top-32 of B256
SELECT32 = {"C_explore": True, "D_comb": True}


def load(path):
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
    return gen


def top1_delta(gen):
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
        tt = torch.from_numpy(arr).to(device)
        with torch.inference_mode():
            preds = opt(tt).squeeze(-1).float().cpu().numpy().tolist()
        for i, sk in enumerate(bkeys):
            res[sk] = preds[2 * i + 1] - preds[2 * i]

    for sk in mrl:
        seqs = gen[sk]
        if SELECT32.get(tag := "", False):
            pass
        ranked = sorted(seqs.items(), key=lambda kv: -kv[1])
        top1 = ranked[0][0]
        batch.extend([src_seq[sk], top1])
        bkeys.append(sk)
        if len(batch) >= 64:
            flush()
            batch, bkeys = [], []
    if batch:
        flush()
    return res


rows = {}
base_scores = None
for tag, path in POOLS.items():
    gen = load(path)
    if SELECT32.get(tag):
        sel = {}
        for sk, seqs in gen.items():
            ranked = sorted(seqs.items(), key=lambda kv: -kv[1])[:32]
            sel[sk] = dict(ranked)
        gen = sel
    sc = top1_delta(gen)
    if tag == "A_baseline":
        base_scores = sc
        rows[tag] = {"n": len(sc), "top1_delta_mean": sum(sc.values()) / len(sc)}
        continue
    d = [sc[k] - base_scores[k] for k in sc if k in base_scores]

    def bci(dd, iters=2000, seed=20260816):
        rng = random.Random(seed)
        n = len(dd)
        means = []
        for _ in range(iters):
            s = [dd[rng.randrange(n)] for _ in range(n)]
            means.append(sum(s) / len(s))
        means.sort()
        return [means[int(0.025 * iters)], means[int(0.975 * iters)]]

    ci = bci(d)
    rows[tag] = {
        "n": len(sc),
        "top1_delta_mean": sum(sc.values()) / len(sc),
        "d_optimus": sum(d) / len(d),
        "d_optimus_ci95": ci,
    }
out = {
    "schema_version": "route_a_v3_comb_tier1_optimus.v1",
    "gate": "B2-I: d_optimus >= +0.02 CI excl zero",
    "rows": rows,
}
Path(D + "/comb_tier1_optimus.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
