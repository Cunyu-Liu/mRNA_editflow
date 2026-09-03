#!/usr/bin/env python3
"""Adjudicate Route A full-FT V2 (FINAL_EPOCH_6_FIXED) vs frozen-Optimus on
GSE114002 VALIDATION (K=10) with paired group bootstrap (2000 iters, seed 20260816).

Preregistered decision rule (W-ladder bottom line, per spec v5.1):
PASS requires Spearman CI not crossing zero AND decision-caliber (top-1/NDCG)
reported alongside - mixed verdicts reported honestly.
"""
import json, importlib.util, sys
from pathlib import Path

REPO = "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901"
MNT = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2"
OUT_DIR = MNT + "/experiments/analysis_fullft_v2_adjudication_20260903"
K = 10
BOOT_ITERS = 2000
BOOT_SEED = 20260816

spec = importlib.util.spec_from_file_location("ev", REPO + "/scripts/route_a_v3/evaluate_route2_prediction_v1.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

MAN = MNT + "/manifests/route2_development_frozen_v1/development_manifest.jsonl"
CANON = MNT + "/canonical/GSE114002/v1/canonical_records.private.jsonl"
OPTIMUS_PREDS = MNT + "/runs/development_hpo/external_lr1e3_wd1e4_replay_gpu5_v1/optimus5prime/validation_predictions.jsonl"
FULLFT_PREDS = MNT + "/experiments/xeditcritic_route_a/280k_fullft_v2_6ep_20260903/predictions.jsonl"

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

fullft = load_preds(FULLFT_PREDS)
optimus = load_preds(OPTIMUS_PREDS)
model_sub = {k: v for k, v in fullft.items() if k in ids}
base_sub = {k: v for k, v in optimus.items() if k in ids}
print("coverage: fullft %d/%d optimus %d/%d" % (len(model_sub), len(ids), len(base_sub), len(ids)))
assert len(model_sub) == len(ids) and len(base_sub) == len(ids)

obs = ev.load_observations([Path(CANON)], ids)

# point metrics both models
m_fullft = ev.evaluate(obs, model_sub, K)
m_opt = ev.evaluate(obs, base_sub, K)
for name, m in [("fullft_v2_ep6", m_fullft), ("frozen_optimus", m_opt)]:
    print("%-18s | rho %.4f | top1 %.4f | ndcg@10 %.4f" % (
        name, m["task_macro_spearman"], m["source_macro_top_1_accuracy"], m["source_macro_ndcg_at_k"]))

# paired bootstrap fullft vs optimus
pb = ev.paired_group_bootstrap(obs, model_sub, base_sub, BOOT_ITERS, BOOT_SEED, K)
ts = pb["task_macro_spearman"]
rk = pb.get("ranking") or {}
def ci(x):
    c = x.get("bootstrap_ci_95") if isinstance(x, dict) else None
    return None if not c else [round(c[0], 4), round(c[1], 4)]
result = {
    "schema_version": "route_a_v3_route2_fullft_v2_adjudication.v1",
    "study": "GSE114002",
    "split": "VALIDATION",
    "k": K,
    "bootstrap_iterations": BOOT_ITERS,
    "bootstrap_seed": BOOT_SEED,
    "fullft_v2_ep6": {
        "task_macro_spearman": m_fullft["task_macro_spearman"],
        "top_1": m_fullft["source_macro_top_1_accuracy"],
        "ndcg_at_10": m_fullft["source_macro_ndcg_at_k"],
    },
    "frozen_optimus": {
        "task_macro_spearman": m_opt["task_macro_spearman"],
        "top_1": m_opt["source_macro_top_1_accuracy"],
        "ndcg_at_10": m_opt["source_macro_ndcg_at_k"],
    },
    "delta_spearman": {"point": ts["improvement"], "ci_95": ci(ts)},
    "delta_top_1": {"point": rk.get("top_1", {}).get("mean_improvement"), "ci_95": ci(rk.get("top_1", {}))},
    "delta_ndcg": {"point": rk.get("ndcg", {}).get("mean_improvement"), "ci_95": ci(rk.get("ndcg", {}))},
}
import os
os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT_DIR + "/adjudication_results.json", "w") as f:
    json.dump(result, f, indent=1)
print("delta_spearman", result["delta_spearman"])
print("delta_top_1   ", result["delta_top_1"])
print("delta_ndcg    ", result["delta_ndcg"])
print("saved ->", OUT_DIR + "/adjudication_results.json")
