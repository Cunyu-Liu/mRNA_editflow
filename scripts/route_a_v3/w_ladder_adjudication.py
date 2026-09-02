#!/usr/bin/env python3
"""W-ladder formal adjudication (2026-09-03, main session).

Same-caliber evaluation as Task 1 (evaluate_route2_prediction_v1, VALIDATION
split, K=10, frozen manifest) for:

1. V6 H3 final adjudication - main criterion: MPRAU variant pair-mean rho per
   arm, paired bootstrap over variants for delta rho = V6_arm - V5 (gate: CI
   must not cross zero; reference V5 pair-mean 0.1025).
2. W0-polyA decision band - top-1 / NDCG@10 vs preregistered bands
   (pass: top-1 >= 0.55 and NDCG@10 >= 0.885; suspect: top-1 < 0.50), plus
   paired group bootstrap vs APARENT.
3. Aligned MRL eval (GSE114002) for all W-ladder arms so the mechanism map
   uses one caliber.

Read-only on frozen predictions; writes
/mnt/.../experiments/analysis_w_ladder_adjudication_20260903/results.json.
"""
import glob
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901"
MNT = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2"
OUT_DIR = MNT + "/experiments/analysis_w_ladder_adjudication_20260903"
K = 10
BOOT_ITERS = 2000
BOOT_SEED = 20260816

spec = importlib.util.spec_from_file_location("ev", REPO + "/scripts/route_a_v3/evaluate_route2_prediction_v1.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

MAN = MNT + "/manifests/route2_development_frozen_v1/development_manifest.jsonl"


def manifest_ids(study, split="VALIDATION"):
    ids = set()
    with open(MAN) as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == split and r["study_unit_id"] == study:
                ids.add(str(r["canonical_record_id"]))
    return ids


def canonical_path(study):
    for name in ["canonical_records.private.jsonl", "canonical_records.jsonl"]:
        p = f"{MNT}/canonical/{study}/v1/{name}"
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"canonical for {study}")


def load_critic_predictions(path):
    preds, rows = {}, {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            preds[str(r["record_id"])] = float(r["prediction"])
            rows[str(r["record_id"])] = r
    return preds, rows


def run_eval(study, preds):
    ids = manifest_ids(study)
    obs = ev.load_observations([Path(canonical_path(study))], ids)
    sub = {k: v for k, v in preds.items() if k in ids}
    if len(sub) != len(ids):
        return None, len(ids) - len(sub)
    return ev.evaluate(obs, sub, K), 0


def variant_pair_table(rows_by_id, study="ENCSR854RUF"):
    """Variant-level (target_mean, pred_mean) rows, Task-1 mprau_pair caliber."""
    ids = manifest_ids(study)
    by_variant = defaultdict(list)
    for rid, row in rows_by_id.items():
        if rid in ids and rid.startswith("ENCSR854RUF:"):
            by_variant[rid.split(":context:")[0]].append(row)
    variants = {}
    for variant, rs in by_variant.items():
        if len(rs) >= 2:
            variants[variant] = (
                float(np.mean([r["target"] for r in rs])),
                float(np.mean([r["prediction"] for r in rs])),
            )
    return variants


def paired_bootstrap_pair_mean(v5_variants, arm_variants, iters=BOOT_ITERS, seed=BOOT_SEED):
    shared = sorted(set(v5_variants) & set(arm_variants))
    t = np.array([v5_variants[v][0] for v in shared])
    p5 = np.array([v5_variants[v][1] for v in shared])
    pa = np.array([arm_variants[v][1] for v in shared])
    rho5 = spearmanr(t, p5).statistic
    rhoa = spearmanr(t, pa).statistic
    rng = np.random.default_rng(seed)
    n = len(shared)
    deltas = []
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        try:
            r5 = spearmanr(t[idx], p5[idx]).statistic
            ra = spearmanr(t[idx], pa[idx]).statistic
            if np.isfinite(r5) and np.isfinite(ra):
                deltas.append(ra - r5)
        except Exception:
            continue
    deltas = np.asarray(deltas)
    return {
        "shared_variant_count": n,
        "v5_pair_mean_spearman": float(rho5),
        "arm_pair_mean_spearman": float(rhoa),
        "delta_pair_mean_spearman": float(rhoa - rho5),
        "bootstrap_ci_95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "bootstrap_iterations": int(len(deltas)),
        "ci_excludes_zero": bool(np.percentile(deltas, 2.5) > 0 or np.percentile(deltas, 97.5) < 0),
    }


results = {"k": K, "bootstrap_iterations": BOOT_ITERS, "bootstrap_seed": BOOT_SEED,
           "v6_h3_adjudication": {}, "w0_polya": {}, "mrl_aligned": {}}

# ---- load prediction files ----
v5_path = glob.glob(MNT + "/experiments/xeditcritic_v5/*/v5_full/final_validation_predictions.jsonl")[0]
v5p, v5rows = load_critic_predictions(v5_path)
paths = {
    "v6_h3_lambda_0_5": MNT + "/experiments/xeditcritic_v6/v6_screen_seed_20260907_runner_586e08aa/v6_h3_lambda_0_5/pass_8_validation_predictions.jsonl",
    "v6_h3_lambda_0_75": MNT + "/experiments/xeditcritic_v6/v6_screen_seed_20260907_runner_586e08aa/v6_h3_lambda_0_75/pass_8_validation_predictions.jsonl",
    "v6_h3_lambda_1_0": MNT + "/experiments/xeditcritic_v6/v6_screen_seed_20260907_runner_586e08aa/v6_h3_lambda_1_0/pass_8_validation_predictions.jsonl",
    "w0_polya_gse269595": MNT + "/experiments/xeditcritic_w0/w0_screen_seed_20260907_runner_7303417c/w0_polya_gse269595/pass_8_validation_predictions.jsonl",
    "w0_mrl_gse114002": MNT + "/experiments/xeditcritic_w0/w0_screen_seed_20260907_runner_7303417c/w0_mrl_gse114002/pass_8_validation_predictions.jsonl",
    "w0_continue_mrl": MNT + "/experiments/xeditcritic_w0/w0_continue_seed_20260907_runner_10fced68/w0_continue_mrl_gse114002/pass_8_validation_predictions.jsonl",
    "w1_head_mrl": MNT + "/experiments/xeditcritic_w1/w1_screen_seed_20260907_runner_2b014489/w1_head_mrl_gse114002/pass_8_validation_predictions.jsonl",
    "w1_lora_mrl": MNT + "/experiments/xeditcritic_w1/w1_screen_seed_20260907_runner_2b014489/w1_lora_mrl_gse114002/pass_8_validation_predictions.jsonl",
}
arm_rows = {}
for name, path in paths.items():
    if not os.path.exists(path):
        alt = path.replace("pass_8_validation_predictions.jsonl", "final_validation_predictions.jsonl")
        path = alt if os.path.exists(alt) else path
    preds, rows = load_critic_predictions(path)
    arm_rows[name] = rows
    print(f"loaded {name}: {len(preds)} rows")

v5_variants = variant_pair_table(v5rows)

# ---- 1. V6 H3 adjudication ----
print("\n== V6 H3 FINAL ADJUDICATION (MPRAU variant pair-mean, paired bootstrap vs V5) ==")
for arm in ["v6_h3_lambda_0_5", "v6_h3_lambda_0_75", "v6_h3_lambda_1_0"]:
    variants = variant_pair_table(arm_rows[arm])
    pb = paired_bootstrap_pair_mean(v5_variants, variants)
    pb["gate_pass_beats_v5_ci_excludes_zero"] = pb["ci_excludes_zero"]
    pb["ceiling_ratio_0_683"] = pb["arm_pair_mean_spearman"] / 0.683
    results["v6_h3_adjudication"][arm] = pb
    print(f"{arm}: rho {pb['arm_pair_mean_spearman']:.4f} vs V5 {pb['v5_pair_mean_spearman']:.4f}"
          f" | delta {pb['delta_pair_mean_spearman']:+.4f} CI [{pb['bootstrap_ci_95'][0]:.4f},{pb['bootstrap_ci_95'][1]:.4f}]"
          f" | ceiling {pb['ceiling_ratio_0_683']*100:.1f}% | gate {'PASS' if pb['ci_excludes_zero'] else 'FAIL'}")

# ---- 2. W0-polyA decision band (Task-1 aligned caliber) ----
print("\n== W0-polyA DECISION BAND (aligned eval, K=10) ==")
w0p = {rid: r["prediction"] for rid, r in arm_rows["w0_polya_gse269595"].items()}
m, missing = run_eval("GSE269595", w0p)
if m is not None:
    top1 = m["source_macro_top_1_accuracy"]
    ndcg = m["source_macro_ndcg_at_k"]
    band = "ARCH_OK" if (top1 >= 0.55 and ndcg >= 0.885) else ("ARCH_SUSPECT" if top1 < 0.50 else "MIXED")
    results["w0_polya"]["aligned_eval"] = m
    results["w0_polya"]["decision_band"] = {
        "top_1": top1, "ndcg_at_10": ndcg, "band": band,
        "aparent_adapter_reference": {"top_1": 0.6011, "ndcg_at_10": 0.8906},
        "preregistered_pass_band": "top-1 >= 0.55 and NDCG@10 >= 0.885",
        "preregistered_suspect_band": "top-1 < 0.50",
    }
    print(f"W0-polyA: rho {m['task_macro_spearman']:.4f} | top-1 {top1:.4f} | ndcg@10 {ndcg:.4f}"
          f" | within-src {m['source_macro_within_source_spearman']:.4f} | band {band}")
else:
    print(f"W0-polyA aligned eval MISSING {missing} predictions")

# paired bootstrap vs APARENT
aparent_path = MNT + "/runs/development_hpo/aparent_v1/validation_predictions.jsonl"
try:
    base_preds, _ = (lambda: ({}, None))()
    out = {}
    with open(aparent_path) as f:
        for line in f:
            r = json.loads(line)
            key = str(r.get("canonical_record_id") or r.get("record_id"))
            out[key] = float(r.get("predicted_direction_normalized_delta", r.get("prediction")))
    base_preds = out
    ids = manifest_ids("GSE269595")
    obs = ev.load_observations([Path(canonical_path("GSE269595"))], ids)
    model_sub = {k: v for k, v in w0p.items() if k in ids}
    base_sub = {k: v for k, v in base_preds.items() if k in ids}
    if len(model_sub) == len(ids) and len(base_sub) == len(ids):
        pb = ev.paired_group_bootstrap(obs, model_sub, base_sub, BOOT_ITERS, BOOT_SEED, K)
        results["w0_polya"]["paired_bootstrap_vs_aparent"] = pb
        ts = pb["task_macro_spearman"]
        rk = pb.get("ranking") or {}
        print(f"W0-polyA vs APARENT: dSpearman {ts['improvement']:+.4f} CI {ts['bootstrap_ci_95']}"
              f" | dTop1 {rk.get('top_1',{}).get('mean_improvement'):+.4f} CI {rk.get('top_1',{}).get('bootstrap_ci_95')}"
              f" | dNDCG {rk.get('ndcg',{}).get('mean_improvement'):+.4f} CI {rk.get('ndcg',{}).get('bootstrap_ci_95')}")
except Exception as exc:
    print(f"W0-polyA vs APARENT bootstrap failed: {exc}")

# ---- 3. MRL aligned eval for all W-ladder arms ----
print("\n== MRL ALIGNED EVAL (GSE114002, K=10) ==")
for arm in ["w0_mrl_gse114002", "w0_continue_mrl", "w1_head_mrl", "w1_lora_mrl"]:
    preds = {rid: r["prediction"] for rid, r in arm_rows[arm].items()}
    m, missing = run_eval("GSE114002", preds)
    if m is not None:
        results["mrl_aligned"][arm] = {
            "task_macro_spearman": m["task_macro_spearman"],
            "top_1": m["source_macro_top_1_accuracy"],
            "ndcg_at_10": m["source_macro_ndcg_at_k"],
            "within_source": m["source_macro_within_source_spearman"],
        }
        print(f"{arm:22s}: rho {m['task_macro_spearman']:.4f} | top-1 {m['source_macro_top_1_accuracy']:.4f} | ndcg@10 {m['source_macro_ndcg_at_k']:.4f}")
    else:
        print(f"{arm:22s}: MISSING {missing}")

os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT_DIR + "/results.json", "w") as f:
    json.dump(results, f, indent=1, sort_keys=True)
print(f"\nwrote {OUT_DIR}/results.json")
