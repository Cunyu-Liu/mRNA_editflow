#!/usr/bin/env python3
"""Route A full-FT V2 3-seed ensemble adjudication (2026-09-03).

Multi-seed preregistration: seeds {20260903, 20260904, 20260905}, FINAL-EPOCH-6-FIXED
per-seed predictions averaged (z-scored per seed then mean) -> frozen-delta
evaluation + paired group bootstrap vs frozen-Optimus (2000 iters, seed 20260816).
Decision rule per spec v5.1 bottom line: CI must not cross zero on Spearman
(rank caliber primary); decision calibers (top-1/NDCG) reported alongside.
"""
import json, importlib.util, sys
from pathlib import Path
import numpy as np

REPO = "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901"
MNT = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2"
OUT_DIR = MNT + "/experiments/analysis_fullft_v2_adjudication_20260903"
K = 10; BOOT_ITERS = 2000; BOOT_SEED = 20260816

spec = importlib.util.spec_from_file_location("ev", REPO + "/scripts/route_a_v3/evaluate_route2_prediction_v1.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

MAN = MNT + "/manifests/route2_development_frozen_v1/development_manifest.jsonl"
CANON = MNT + "/canonical/GSE114002/v1/canonical_records.private.jsonl"
OPTIMUS_PREDS = MNT + "/runs/development_hpo/external_lr1e3_wd1e4_replay_gpu5_v1/optimus5prime/validation_predictions.jsonl"
SEED_DIRS = {
    20260903: MNT + "/experiments/xeditcritic_route_a/280k_fullft_v2_6ep_20260903",
    20260904: MNT + "/experiments/xeditcritic_route_a/280k_fullft_v2_6ep_seed20260904",
    20260905: MNT + "/experiments/xeditcritic_route_a/280k_fullft_v2_6ep_seed20260905",
}

ids = set()
with open(MAN) as f:
    for line in f:
        r = json.loads(line)
        if r["split"] == "VALIDATION" and r["study_unit_id"] == "GSE114002":
            ids.add(str(r["canonical_record_id"]))

def load_preds(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            key = str(r.get("canonical_record_id") or r.get("record_id"))
            out[key] = float(r.get("predicted_direction_normalized_delta", r.get("prediction")))
    return out

# per-seed z-score then mean (rank-safe averaging)
seed_preds = {}
for seed, d in SEED_DIRS.items():
    p = {k: v for k, v in load_preds(d + "/predictions.jsonl").items() if k in ids}
    assert len(p) == len(ids), f"seed {seed} coverage {len(p)}/{len(ids)}"
    vals = np.array(list(p.values()))
    z = (vals - vals.mean()) / vals.std()
    seed_preds[seed] = dict(zip(p.keys(), z))
    print("seed %d loaded (%d recs)" % (seed, len(p)))

ens = {rid: float(np.mean([seed_preds[s][rid] for s in SEED_DIRS])) for rid in ids}
optimus = {k: v for k, v in load_preds(OPTIMUS_PREDS).items() if k in ids}

obs = ev.load_observations([Path(CANON)], ids)
m_ens = ev.evaluate(obs, ens, K)
m_opt = ev.evaluate(obs, optimus, K)
print("%-16s | rho %.4f | top1 %.4f | ndcg@10 %.4f" % ("3seed_ensemble", m_ens["task_macro_spearman"], m_ens["source_macro_top_1_accuracy"], m_ens["source_macro_ndcg_at_k"]))
print("%-16s | rho %.4f | top1 %.4f | ndcg@10 %.4f" % ("frozen_optimus", m_opt["task_macro_spearman"], m_opt["source_macro_top_1_accuracy"], m_opt["source_macro_ndcg_at_k"]))

pb = ev.paired_group_bootstrap(obs, ens, optimus, BOOT_ITERS, BOOT_SEED, K)
ts = pb["task_macro_spearman"]
rk = pb.get("ranking") or {}
def ci(x):
    c = x.get("bootstrap_ci_95") if isinstance(x, dict) else None
    return None if not c else [round(c[0], 4), round(c[1], 4)]
result = {
    "schema_version": "route_a_v3_route2_fullft_v2_3seed_ensemble.v1",
    "seeds": sorted(SEED_DIRS),
    "averaging": "PER_SEED_ZSCORE_MEAN",
    "bootstrap_iterations": BOOT_ITERS, "bootstrap_seed": BOOT_SEED,
    "ensemble": {"task_macro_spearman": m_ens["task_macro_spearman"], "top_1": m_ens["source_macro_top_1_accuracy"], "ndcg_at_10": m_ens["source_macro_ndcg_at_k"]},
    "frozen_optimus": {"task_macro_spearman": m_opt["task_macro_spearman"], "top_1": m_opt["source_macro_top_1_accuracy"], "ndcg_at_10": m_opt["source_macro_ndcg_at_k"]},
    "delta_spearman": {"point": ts["improvement"], "ci_95": ci(ts)},
    "delta_top_1": {"point": rk.get("top_1", {}).get("mean_improvement"), "ci_95": ci(rk.get("top_1", {}))},
    "delta_ndcg": {"point": rk.get("ndcg", {}).get("mean_improvement"), "ci_95": ci(rk.get("ndcg", {}))},
}
print(json.dumps({k: result[k] for k in ["delta_spearman", "delta_top_1", "delta_ndcg"]}, indent=1))
with open(OUT_DIR + "/ensemble_3seed_vs_optimus.json", "w") as f:
    json.dump(result, f, indent=1)
print("saved ->", OUT_DIR + "/ensemble_3seed_vs_optimus.json")
