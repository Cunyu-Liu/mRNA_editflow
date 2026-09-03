#!/usr/bin/env python3
"""Adjudicate polyA Route A (APA 3.5M pre-finetune, FINAL-EPOCH-6-FIXED) vs
APARENT adapter and critic V5 on GSE269595 VALIDATION (K=10), with paired
group bootstrap (2000 iters, seed 20260816). Rank caliber (Spearman) primary,
decision calibers (top-1/NDCG) reported alongside per spec v5.1."""
import json, importlib.util, sys
from pathlib import Path

REPO = "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901"
MNT = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2"
APA_PREDS = MNT + "/experiments/xeditcritic_route_a/apa_3p5m_prefinetune_20260903/predictions.jsonl"
APARENT_PREDS = MNT + "/runs/development_hpo/aparent_v1/validation_predictions.jsonl"
V5_GLOB = MNT + "/experiments/xeditcritic_v5/*/v5_full/final_validation_predictions.jsonl"
OUT = MNT + "/experiments/analysis_apa_route_a_adjudication_20260903"
K = 10; BOOT_ITERS = 2000; BOOT_SEED = 20260816

spec = importlib.util.spec_from_file_location("ev", REPO + "/scripts/route_a_v3/evaluate_route2_prediction_v1.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

import glob
MAN = MNT + "/manifests/route2_development_frozen_v1/development_manifest.jsonl"
CANON = MNT + "/canonical/GSE269595/v1/canonical_records.private.jsonl"

ids = set()
with open(MAN) as f:
    for line in f:
        r = json.loads(line)
        if r["split"] == "VALIDATION" and r["study_unit_id"] == "GSE269595":
            ids.add(str(r["canonical_record_id"]))

def load_preds(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            key = str(r.get("canonical_record_id") or r.get("record_id"))
            out[key] = float(r.get("predicted_direction_normalized_delta", r.get("prediction")))
    return out

apa = {k: v for k, v in load_preds(APA_PREDS).items() if k in ids}
aparent = {k: v for k, v in load_preds(APARENT_PREDS).items() if k in ids}
v5p = {k: v for k, v in load_preds(glob.glob(V5_GLOB)[0]).items() if k in ids}
print("coverage: apa %d aparent %d v5 %d / %d" % (len(apa), len(aparent), len(v5p), len(ids)))
assert len(apa) == len(ids)

obs = ev.load_observations([Path(CANON)], ids)
results = {"schema_version": "route_a_v3_route2_apa_route_a_adjudication.v1", "study": "GSE269595", "split": "VALIDATION", "k": K}
for name, preds in [("apa_route_a_ep6", apa), ("aparent_adapter", aparent), ("critic_v5", v5p)]:
    m = ev.evaluate(obs, preds, K)
    results[name] = {
        "task_macro_spearman": m["task_macro_spearman"],
        "top_1": m["source_macro_top_1_accuracy"],
        "ndcg_at_10": m["source_macro_ndcg_at_k"],
        "regret": m.get("source_macro_normalized_regret"),
    }
    print("%-16s | rho %.4f | top1 %.4f | ndcg %.4f" % (name, m["task_macro_spearman"], m["source_macro_top_1_accuracy"], m["source_macro_ndcg_at_k"]))

def ci(x):
    c = x.get("bootstrap_ci_95") if isinstance(x, dict) else None
    return None if not c else [round(c[0], 4), round(c[1], 4)]

for tag, base in [("vs_aparent", aparent), ("vs_critic_v5", v5p)]:
    if len(base) != len(ids):
        print(tag, "skipped (base coverage)"); continue
    pb = ev.paired_group_bootstrap(obs, apa, base, BOOT_ITERS, BOOT_SEED, K)
    ts = pb["task_macro_spearman"]; rk = pb.get("ranking") or {}
    results["delta_" + tag] = {
        "spearman": {"point": ts["improvement"], "ci_95": ci(ts)},
        "top_1": {"point": rk.get("top_1", {}).get("mean_improvement"), "ci_95": ci(rk.get("top_1", {}))},
        "ndcg": {"point": rk.get("ndcg", {}).get("mean_improvement"), "ci_95": ci(rk.get("ndcg", {}))},
    }
    print(tag, json.dumps(results["delta_" + tag]))

import os
os.makedirs(OUT, exist_ok=True)
json.dump(results, open(OUT + "/adjudication_results.json", "w"), indent=1)
print("saved ->", OUT + "/adjudication_results.json")
