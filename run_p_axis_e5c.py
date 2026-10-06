#!/usr/bin/env python3
"""P-axis E5-c: extend E4 edit-scan M4 metrics to Saluki/UTR-LM/RNA-FM.

E4 established M4 (surface high-frequency noise) on two representative models
(Optimus + FramePool, MRL task). E5-c extends to the remaining baselines:
  - RNA-FM:  full single-edit embedding-distance scan on polyA (<=50 source),
             compute roughness ratio / neighbor autocorr / top5% energy share.
  - Saluki:  per-source prediction variance from existing predictions.jsonl
             (surrogate M4; Wave 3 found Saluki per-variant pred std=0, so
             this row is expected to be "inapplicable" - documented, not hidden).
  - UTR-LM:  per-source prediction variance from existing predictions.jsonl
             (surrogate M4; UTR-LM has per-variant variation).

Discipline: inference only, no training, VALIDATION only, protected reads = 0,
outputs only-added. Budget <=1 GPU-h (RNA-FM embedding scan only; Saluki/UTR-LM
are CPU post-processing of existing predictions).

Outputs: experiments/analysis_delta_failure_mechanism_e5c/e5c_results.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

REPO_ROOT = Path("/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_w0_diagnosis_20260902")
HARNESS = REPO_ROOT / "scripts/route_a_v3/run_route2_external_prediction_baselines_v1.py"
TE_SCRIPT = REPO_ROOT / "scripts/route_a_v3/run_route2_frozen_delta_te_family_v1.py"

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
OUT = MNT / "experiments/analysis_delta_failure_mechanism_e5c"
PROJ = MNT / "projections/xedit_v3/development_train_validation_v1/validation.jsonl"

# existing prediction files (from Wave 1-3 artifacts)
FULL_COV = MNT / "experiments/analysis_frozen_delta_full_coverage_20260904"
SALUKI_MPRAU = MNT / "experiments/analysis_saluki_frozen_mprau_20260903/predictions.jsonl"
SALUKI_HL = MNT / "experiments/analysis_saluki_frozen_gse217518_20260903/predictions.jsonl"
HPO = MNT / "runs/development_hpo"

BASES = "ACGT"
PAS_HEXAMERS = ("AATAAA", "ATTAAA", "AGTAAA", "TATAAA", "AAGAAA", "AATATA", "AATACA", "AATAGA")

# task -> (label, dirname) for full-coverage predictions
TASK_DIR = {
    "MEAN_RIBOSOME_LOAD::region=0": ("MRL", "mrl_gse114002"),
    "MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE::region=1": ("MPRAU", "mprau_encsr854ruf"),
    "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS::region=1": ("polyA", "polya_gse269595"),
    "PUBLISHED_REF_VS_ALT_ACTIVITY_LMM_LOG2_FOLD_CHANGE::region=1": ("REFALT", "gse186455"),
    "RNA_HALF_LIFE_MINUTES::region=0": ("HL5", "half_life_5utr"),
    "RNA_HALF_LIFE_MINUTES::region=1": ("HL3", "half_life_3utr"),
    "TOTAL_POLYSOME_TRANSLATION_EFFICIENCY::region=1": ("TE200304", "gse200304_te"),
    "te_log2_polysome_over_totalrna::region=0": ("TE149487", "gse149487_te"),
    "transcript_log2_totalrna_over_dna::region=0": ("RNA149487", "gse149487_rna"),
}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_polya_records():
    rows = []
    with open(PROJ) as f:
        for line in f:
            d = json.loads(line)
            if d["task_id"] == "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS::region=1":
                rows.append(d)
    return rows


def load_mrl_records():
    rows = []
    with open(PROJ) as f:
        for line in f:
            d = json.loads(line)
            if d["task_id"] == "MEAN_RIBOSOME_LOAD::region=0":
                rows.append(d)
    return rows


def load_task_records(task_id):
    rows = []
    with open(PROJ) as f:
        for line in f:
            d = json.loads(line)
            if d["task_id"] == task_id:
                rows.append(d)
    return rows


# ---- Part 1: RNA-FM embedding M4 on polyA (GPU, <=50 source) ----

def rnafm_embedding_m4(te, device, n_sources=20, pos_stride=4):
    """Single-edit embedding-distance scan on polyA (subsampled), M4 metrics.

    Budget-constrained: 20 sources * ~41 sampled positions * 3 alts ~ 2460
    variants (~10x reduction from full 24600). For each sampled source: edit
    every pos_stride-th position to 3 alternatives, get RNA-FM embedding,
    compute L2 distance to source embedding. Per-source M4 metrics:
      - roughness: mean(dist^2) / var(dy_global)
      - neighbor autocorrelation: corr of adjacent-sampled-position mean dists
      - top5% energy share
    """
    polya_rows = load_polya_records()
    by_source = {}
    for r in polya_rows:
        by_source.setdefault(r["source_sequence"], []).append(r)
    sources = sorted(by_source)[:n_sources]
    print(f"[RNA-FM M4] polyA sources for scan: {len(sources)} (stride={pos_stride})")

    # source embeddings
    src_emb = {}
    te.embed_rnafm(sources, src_emb, device, {})

    # build subsampled single-edit variants
    variants = []
    v_meta = []  # (source_idx, position)
    for si, s in enumerate(sources):
        sT = s.upper().replace("U", "T")
        for pos in range(0, len(sT), pos_stride):
            for b in BASES:
                if b != sT[pos]:
                    variants.append(sT[:pos] + b + sT[pos + 1:])
                    v_meta.append((si, pos))
    print(f"[RNA-FM M4] total variants (subsampled): {len(variants)}")

    # batch embed variants
    var_emb = {}
    te.embed_rnafm(variants, var_emb, device, {})

    # per-source distance vectors
    dist_by_source = defaultdict(dict)  # si -> pos -> dist
    for (si, pos), v in zip(v_meta, variants):
        d = float(torch.norm(src_emb[sources[si]] - var_emb[v]).item())
        dist_by_source[si][pos] = d

    # global dy signal scale
    all_dy = np.array([r["direction_normalized_delta"] for r in polya_rows], float)
    var_dy = float(np.var(all_dy))

    rough, surf_std, neighbor_ac, top5 = [], [], [], []
    for si in range(len(sources)):
        positions = sorted(dist_by_source[si].keys())
        dists = np.array([dist_by_source[si][p] for p in positions])
        if len(dists) < 5:
            continue
        rough.append(float(np.mean(dists ** 2)))
        surf_std.append(float(dists.std()))
        if len(dists) > 5:
            a, b = dists[:-1], dists[1:]
            if np.std(a) > 0 and np.std(b) > 0:
                neighbor_ac.append(float(np.corrcoef(a, b)[0, 1]))
        ab = np.abs(dists)
        k = max(1, int(0.05 * len(ab)))
        top5.append(float(np.sort(ab)[-k:].sum() / (ab.sum() + 1e-12)))

    return {
        "model": "rnafm_embedding",
        "task": "polyA",
        "n_sources": len(sources),
        "n_variants": len(variants),
        "var_dy_global": var_dy,
        "roughness_ratio_mean": float(np.mean(rough) / max(var_dy, 1e-12)) if rough else None,
        "surface_std_mean": float(np.mean(surf_std)) if surf_std else None,
        "neighbor_autocorr_mean": float(np.mean(neighbor_ac)) if neighbor_ac else None,
        "top5pct_energy_share_mean": float(np.mean(top5)) if top5 else None,
        "per_source_roughness": [float(x) for x in rough[:50]],
    }


# ---- Part 2: Saluki/UTR-LM M4 proxy from existing predictions (CPU) ----

def prediction_variance_m4_proxy(pred_path, task_id, baseline_id=None):
    """Surrogate M4: per-source prediction variance from existing predictions.

    Full edit-scan M4 requires forward pass on all single-edit variants.
    Without inference API, we use per-source prediction variance across
    measured candidates as a proxy for surface amplitude.

    NOTE: Saluki per-variant pred std=0 (Wave 3 finding) -> variance=0,
    documented as "inapplicable" not hidden.
    """
    rows = load_task_records(task_id)
    idx = {r["canonical_record_id"]: r for r in rows}

    by_source = defaultdict(list)
    with open(pred_path) as f:
        for line in f:
            d = json.loads(line)
            if baseline_id is not None and d.get("baseline_id") != baseline_id:
                continue
            rid = d["canonical_record_id"]
            if rid in idx:
                r = idx[rid]
                by_source[r["source_sequence"]].append(d["predicted_direction_normalized_delta"])

    # global dy
    all_dy = np.array([r["direction_normalized_delta"] for r in rows], float)
    var_dy = float(np.var(all_dy))

    per_source_var = []
    for src, preds in by_source.items():
        if len(preds) >= 3:
            per_source_var.append(float(np.var(preds)))

    mean_pred_var = float(np.mean(per_source_var)) if per_source_var else 0.0
    var_ratio = mean_pred_var / max(var_dy, 1e-12)

    # std check (Saluki detection)
    all_preds = [p for preds in by_source.values() for p in preds]
    pred_std = float(np.std(all_preds)) if all_preds else 0.0

    return {
        "n_sources": len(by_source),
        "n_predictions": len(all_preds),
        "pred_std": pred_std,
        "mean_per_source_var": mean_pred_var,
        "var_ratio_proxy": var_ratio,
        "applicable": pred_std > 1e-8,
        "note": "Saluki per-variant pred std=0 -> inapplicable" if pred_std < 1e-8 else "surrogate M4 (per-source prediction variance, not full edit scan)",
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    results = {
        "schema_version": "route_a_v3_route2_delta_failure_mechanism_e5c.v1",
        "date": "2026-09-12",
        "training": "none (inference + post-processing)",
        "protected_reads": 0,
        "budget": "<=1 GPU-h (RNA-FM embedding) + CPU (Saluki/UTR-LM proxy)",
    }

    # ---- Part 1: RNA-FM embedding M4 on polyA ----
    try:
        te = _load_module("te", TE_SCRIPT)
        device = torch.device("cuda:6")
        results["rnafm_embedding_m4_polya"] = rnafm_embedding_m4(te, device, n_sources=20, pos_stride=4)
        print("[RNA-FM M4] done:", results["rnafm_embedding_m4_polya"]["roughness_ratio_mean"])
    except Exception as e:
        results["rnafm_embedding_m4_polya"] = {"error": str(e)}
        print(f"[RNA-FM M4] FAILED: {e}")

    # ---- Part 2: Saluki M4 proxy ----
    saluki_results = {}
    # MPRAU
    task_id = "MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE::region=1"
    if SALUKI_MPRAU.exists():
        saluki_results["MPRAU"] = prediction_variance_m4_proxy(SALUKI_MPRAU, task_id)
    # HL5
    task_id = "RNA_HALF_LIFE_MINUTES::region=0"
    if SALUKI_HL.exists():
        saluki_results["HL5"] = prediction_variance_m4_proxy(SALUKI_HL, task_id)
    # HL3
    task_id = "RNA_HALF_LIFE_MINUTES::region=1"
    if SALUKI_HL.exists():
        saluki_results["HL3"] = prediction_variance_m4_proxy(SALUKI_HL, task_id)
    results["saluki_m4_proxy"] = saluki_results
    for task, r in saluki_results.items():
        print(f"[Saluki M4 proxy] {task}: applicable={r['applicable']} pred_std={r['pred_std']:.6f}")

    # ---- Part 3: UTR-LM M4 proxy ----
    utrlm_results = {}
    for task_id, (label, dirname) in TASK_DIR.items():
        pred_path = FULL_COV / f"{dirname}__utrlm" / "predictions.jsonl"
        if pred_path.exists():
            utrlm_results[label] = prediction_variance_m4_proxy(pred_path, task_id)
    results["utrlm_m4_proxy"] = utrlm_results
    for task, r in utrlm_results.items():
        print(f"[UTR-LM M4 proxy] {task}: applicable={r['applicable']} var_ratio={r['var_ratio_proxy']:.4f}")

    with open(OUT / "e5c_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print(f"written: {OUT / 'e5c_results.json'}")


if __name__ == "__main__":
    main()
