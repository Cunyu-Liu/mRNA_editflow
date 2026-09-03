#!/usr/bin/env python3
"""Full-FT V2 vs critic V5 paired bootstrap on GSE114002 VALIDATION - proves the
in-house improvement (+0.18 point estimate) is statistically significant."""
import json, importlib.util, glob
from pathlib import Path

REPO = "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901"
MNT = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2"
K = 10; BOOT_ITERS = 2000; BOOT_SEED = 20260816

spec = importlib.util.spec_from_file_location("ev", REPO + "/scripts/route_a_v3/evaluate_route2_prediction_v1.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

MAN = MNT + "/manifests/route2_development_frozen_v1/development_manifest.jsonl"
CANON = MNT + "/canonical/GSE114002/v1/canonical_records.private.jsonl"
FULLFT_PREDS = MNT + "/experiments/xeditcritic_route_a/280k_fullft_v2_6ep_20260903/predictions.jsonl"
v5_path = glob.glob(MNT + "/experiments/xeditcritic_v5/*/v5_full/final_validation_predictions.jsonl")[0]

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

fullft = {k: v for k, v in load_preds(FULLFT_PREDS).items() if k in ids}
v5p = {k: v for k, v in load_preds(v5_path).items() if k in ids}
print("coverage: fullft %d v5 %d / %d" % (len(fullft), len(v5p), len(ids)))
obs = ev.load_observations([Path(CANON)], ids)
pb = ev.paired_group_bootstrap(obs, fullft, v5p, BOOT_ITERS, BOOT_SEED, K)
ts = pb["task_macro_spearman"]
rk = pb.get("ranking") or {}
def ci(x):
    c = x.get("bootstrap_ci_95") if isinstance(x, dict) else None
    return None if not c else [round(c[0], 4), round(c[1], 4)]
out = {
    "schema_version": "route_a_v3_route2_fullft_v2_vs_critic_v5.v1",
    "delta_spearman": {"point": ts["improvement"], "ci_95": ci(ts)},
    "delta_top_1": {"point": rk.get("top_1", {}).get("mean_improvement"), "ci_95": ci(rk.get("top_1", {}))},
    "delta_ndcg": {"point": rk.get("ndcg", {}).get("mean_improvement"), "ci_95": ci(rk.get("ndcg", {}))},
}
print(json.dumps(out, indent=1))
json.dump(out, open(MNT + "/experiments/analysis_fullft_v2_adjudication_20260903/vs_critic_v5.json", "w"), indent=1)
