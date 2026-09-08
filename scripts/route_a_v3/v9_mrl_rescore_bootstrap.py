#!/usr/bin/env python3
"""Rescore MRL VALIDATION per-record predictions for terminal V9-1a seeds.

Loads each seed's final checkpoint, scores the GSE114002 VALIDATION split
(source/candidate -> delta), persists {domain}_predictions.jsonl next to the
checkpoint (the same files the V9 runner will emit for seed 20260911 once the
persistence hook lands), then runs the 2-seed z-mean ensemble + paired
bootstrap vs V5 (source-group cluster, 2000 iters, seed 20260816).
"""
from __future__ import annotations

import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

V8_WT = Path("/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_v8_stage1_prep_20260904")
sys.path.insert(0, str(V8_WT))
sys.path.insert(0, str(V8_WT / "scripts/route_a_v3"))

_spec = importlib.util.spec_from_file_location("r2", V8_WT / "scripts/route_a_v3/run_route2_v8_stage2_adapt_v1.py")
r2 = importlib.util.module_from_spec(_spec)
sys.modules["r2"] = r2
_spec.loader.exec_module(r2)

_spec9 = importlib.util.spec_from_file_location("v9run", V8_WT / "scripts/route_a_v3/run_route2_v9_adapter_zoo_v1.py")
v9run = importlib.util.module_from_spec(_spec9)
sys.modules["v9run"] = v9run
_spec9.loader.exec_module(v9run)

MNT = r2.MNT
V9_DIR = MNT / "experiments/xeditcritic_route_a/v9_adapter_zoo_20260908"
OUT = MNT / "experiments/analysis_v9_stage0_20260908"
SEEDS = ("20260907", "20260915")
BOOT_ITERS, BOOT_SEED = 2000, 20260816


def rescore_seed(seed: str, device) -> dict[str, float]:
    ckpt = V9_DIR / f"seed{seed}" / "v9_adapter_zoo_epoch6.pt"
    pred_file = V9_DIR / f"seed{seed}" / "mrl_predictions.jsonl"
    if pred_file.exists():
        return {json.loads(l)["canonical_record_id"]: float(json.loads(l)["prediction"]) for l in pred_file.open()}
    model = v9run.build_v9(r2.MRNABERT_PATH, num_cells=r2.NUM_CELLS).to(device)
    v9run.load_v9_init(model)
    for p in model.base.parameters():
        p.requires_grad_(False)
    model.wrap_multitask_lora()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(r2.MRNABERT_PATH, local_files_only=True)
    vids = r2._manifest_validation_ids().get("GSE114002", set())
    recs = r2._load_canonical(r2.BENCHMARK_DOMAINS["GSE114002"][1])
    ids = sorted(rid for rid in vids if rid in recs)
    dom = r2.DOMAIN_IDS["mrl"]
    src = r2.score_sequences(model, tok, device, [recs[r]["source_sequence"] for r in ids], dom)
    cnd = r2.score_sequences(model, tok, device, [recs[r]["candidate_sequence"] for r in ids], dom)
    preds = {rid: float(cnd[i] - src[i]) for i, rid in enumerate(ids)}
    with pred_file.open("w") as fh:
        for rid in ids:
            fh.write(json.dumps({"canonical_record_id": rid, "prediction": preds[rid]}) + "\n")
    del model
    torch.cuda.empty_cache()
    return preds


def main() -> int:
    device = torch.device("cuda:5")
    per_seed = {s: rescore_seed(s, device) for s in SEEDS}
    shared = sorted(set(per_seed[SEEDS[0]]) & set(per_seed[SEEDS[1]]))
    recs = r2._load_canonical(r2.BENCHMARK_DOMAINS["GSE114002"][1])
    target = {r: float(recs[r]["direction_normalized_delta"]) for r in shared}
    from scipy.stats import spearmanr

    def rho(pred):
        return float(spearmanr([target[r] for r in shared], [pred[r] for r in shared]).statistic)

    def zscore(p):
        vals = np.array([p[r] for r in shared])
        mu, sd = vals.mean(), vals.std()
        return {r: (p[r] - mu) / (sd if sd > 0 else 1.0) for r in shared}

    zs = [zscore(per_seed[s]) for s in SEEDS]
    ens = {r: float(np.mean([z[r] for z in zs])) for r in shared}
    print("per-seed:", {s: round(rho(per_seed[s]), 4) for s in SEEDS}, "| ensemble: %.4f (n=%d)" % (rho(ens), len(shared)))

    report = {"schema_version": "route_a_v3_v9_mrl_ensemble_bootstrap.v1",
              "seeds": list(SEEDS), "n_records": len(shared),
              "per_seed_rho": {s: rho(per_seed[s]) for s in SEEDS},
              "ensemble_rho_zmean": rho(ens),
              "references": {"frozen_optimus": 0.3132, "route_a_3seed": 0.3158, "v5": 0.1354}}

    v5_paths = sorted(glob.glob(str(MNT / "experiments/xeditcritic_v5/*/v5_full/final_validation_predictions.jsonl")))
    if v5_paths:
        v5 = {}
        for line in open(v5_paths[0]):
            row = json.loads(line)
            rid = str(row.get("canonical_record_id") or row.get("record_id"))
            if rid in target:
                v5[rid] = float(row["prediction"])
        rng = np.random.default_rng(BOOT_SEED)
        n = len(shared)
        deltas = []
        for _ in range(BOOT_ITERS):
            idx = rng.integers(0, n, size=n)
            rs = [shared[i] for i in idx]
            t = np.array([target[r] for r in rs])
            e = np.array([ens[r] for r in rs])
            v = np.array([v5[r] for r in rs])
            deltas.append(spearmanr(t, e).statistic - spearmanr(t, v).statistic)
        ci = np.percentile(deltas, [2.5, 97.5])
        report["vs_v5_paired_bootstrap"] = {
            "delta": rho(ens) - float(spearmanr([target[r] for r in shared], [v5[r] for r in shared]).statistic),
            "ci95": [float(ci[0]), float(ci[1])], "crosses_zero": bool(ci[0] < 0 < ci[1]),
            "iters": BOOT_ITERS, "seed": BOOT_SEED}
        print("vs V5: Δ=%.4f CI [%.4f, %.4f] crosses_zero=%s" % (
            report["vs_v5_paired_bootstrap"]["delta"], ci[0], ci[1],
            report["vs_v5_paired_bootstrap"]["crosses_zero"]))
    (OUT / "mrl_ensemble_bootstrap.json").write_text(json.dumps(report, indent=1))
    print("wrote", OUT / "mrl_ensemble_bootstrap.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
