#!/usr/bin/env python3
"""mRNABERT raw-pretrained-weights frozen-Δ evaluation on all 9 tasks.

User question 4: "Critic is just mRNABERT + modifications - what does PURE
mRNABERT achieve per task?"

Protocol (mirrors analysis_frozen_delta_full_coverage_20260904 rnafm/utrlm
calibre so numbers are directly comparable):
  - backbone: official mRNABERT raw pretrained weights (pytorch_model.bin),
    frozen, no LoRA, no fine-tune, no task adaptation
  - readout: per-task ridge regression on frozen mean-pooled embeddings,
    trained on the task's TRAIN-pool (source, candidate) pairs with target =
    direction_normalized_delta (the frozen-Δ protocol: model predicts
    absolute value per sequence; delta = f(cand) - f(src))
  - evaluation: Spearman(predicted_delta, measured_delta) on VALIDATION,
    source-group bootstrap CI (K>=5), pair-mean caliber for MPRAU

Discipline: zero backbone training, protected reads = 0, VALIDATION only for
reporting, outputs only-added to experiments/analysis_mrnabert_raw_frozen9/.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
MRNABERT_PATH = MNT / "external_model_assets/mrnabert_a1eb7df25804d23f08646e1cb996b234d7208a40"
OUT = MNT / "experiments/analysis_mrnabert_raw_frozen9"
PROJ = MNT / "projections/xedit_v3/development_train_validation_v1"

TASKS = {
    "MEAN_RIBOSOME_LOAD::region=0": "MRL",
    "MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE::region=1": "MPRAU",
    "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS::region=1": "polyA",
    "PUBLISHED_REF_VS_ALT_ACTIVITY_LMM_LOG2_FOLD_CHANGE::region=1": "REFALT",
    "RNA_HALF_LIFE_MINUTES::region=0": "HL5",
    "RNA_HALF_LIFE_MINUTES::region=1": "HL3",
    "TOTAL_POLYSOME_TRANSLATION_EFFICIENCY::region=1": "TE200304",
    "te_log2_polysome_over_totalrna::region=0": "TE149487",
    "transcript_log2_totalrna_over_dna::region=0": "RNA149487",
}

BATCH_TOKENS = 60000
EMB_CACHE = {}


def load_splits():
    train_rows, val_rows = defaultdict(list), defaultdict(list)
    for split_file, store in (("train.jsonl", train_rows), ("validation.jsonl", val_rows)):
        with open(PROJ / split_file) as f:
            for line in f:
                d = json.loads(line)
                store[d["task_id"]].append(d)
    return dict(train_rows), dict(val_rows)


def format_sequence(s: str) -> str:
    s = s.upper().replace("T", "U")
    return " ".join(list(s))


@torch.no_grad()
def embed(model, tokenizer, sequences, device):
    out = []
    for i in range(0, len(sequences), 32):
        chunk = sequences[i : i + 32]
        enc = tokenizer([format_sequence(s) for s in chunk], add_special_tokens=True,
                        padding=True, truncation=False, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        h = model(input_ids=ids, attention_mask=mask)
        if isinstance(h, tuple):
            h = h[0]
        h = h.last_hidden_state if hasattr(h, "last_hidden_state") else h
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1)
        out.append(pooled.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.config.hidden_size))


def main():
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:6")

    tokenizer = AutoTokenizer.from_pretrained(MRNABERT_PATH, local_files_only=True)
    model_config = AutoConfig.from_pretrained(MRNABERT_PATH, local_files_only=True, trust_remote_code=True)
    model = AutoModel.from_config(model_config, trust_remote_code=True, add_pooling_layer=False)
    modeling_module = sys.modules[model.__class__.__module__]
    modeling_module.flash_attn_qkvpacked_func = None
    checkpoint = torch.load(MRNABERT_PATH / "pytorch_model.bin", map_location="cpu", weights_only=False)
    base_state = {k.removeprefix("bert."): v for k, v in checkpoint.items() if k.startswith("bert.")}
    model.load_state_dict(base_state, strict=True)
    model = model.to(device).eval()
    print(f"mRNABERT loaded: {sum(p.numel() for p in model.parameters()):,} params (frozen)")

    train_rows, val_rows = load_splits()
    results = {"schema_version": "mrnabert_raw_frozen9.v1", "date": "2026-09-12",
               "protocol": "raw pretrained weights, frozen backbone, per-task ridge readout on TRAIN pool, frozen-delta (f(cand)-f(src))",
               "protected_reads": 0, "tasks": {}}

    for task_id, label in TASKS.items():
        tr = train_rows.get(task_id, [])
        va = val_rows.get(task_id, [])
        if not tr or not va:
            print(f"{label}: no rows, skip")
            continue

        def seqs_rows(rows):
            src = [r["source_sequence"] for r in rows]
            cnd = [r["candidate_sequence"] for r in rows]
            return src, cnd

        # cap train pool for embedding budget (stride sampling if huge)
        if len(tr) > 20000:
            idx = np.linspace(0, len(tr) - 1, 20000).astype(int)
            tr = [tr[i] for i in idx]
        tr_src, tr_cnd = seqs_rows(tr)
        va_src, va_cnd = seqs_rows(va)

        uniq = sorted(set(tr_src) | set(tr_cnd) | set(va_src) | set(va_cnd))
        print(f"{label}: train={len(tr)} val={len(va)} unique_seqs={len(uniq)}", flush=True)
        emb = embed(model, tokenizer, uniq, device)
        emap = {s: emb[i] for i, s in enumerate(uniq)}

        def deltas(rows, s_list, c_list):
            X = np.stack([emap[c] - emap[s] for s, c in zip(s_list, c_list)])
            y = np.array([r["direction_normalized_delta"] for r in rows], float)
            return X, y

        Xtr, ytr = deltas(tr, tr_src, tr_cnd)
        Xva, yva = deltas(va, va_src, va_cnd)

        # ridge readout on TRAIN (alpha grid fixed, no validation peeking)
        best = None
        for alpha in (1.0, 10.0, 100.0):
            r = Ridge(alpha=alpha)
            r.fit(Xtr, ytr)
            rho = float(spearmanr(r.predict(Xtr), ytr).statistic)
            if best is None or rho > best[0]:
                best = (rho, alpha, r)
        _, alpha, ridge = best
        pred = ridge.predict(Xva)
        rho = float(spearmanr(pred, yva).statistic)

        # source-group bootstrap CI
        groups = defaultdict(list)
        for i, row in enumerate(va):
            groups[row.get("source_group_id") or row.get("source_id") or i].append(i)
        gkeys = sorted(g for g in groups if len(groups[g]) >= 5)
        if not gkeys:
            gkeys = sorted(groups.keys())
        rng = np.random.RandomState(20260912)
        boots = []
        for _ in range(500):
            gs = rng.choice(len(gkeys), len(gkeys), replace=True)
            idx = np.concatenate([groups[gkeys[g]] for g in gs])
            if np.std(yva[idx]) > 0:
                boots.append(float(spearmanr(pred[idx], yva[idx]).statistic))
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None

        # MPRAU pair-mean caliber
        pair_mean = None
        if label == "MPRAU":
            by_variant = defaultdict(list)
            for i, row in enumerate(va):
                by_variant[row["connected_source_component_id"]].append(i)
            pm_rhos = []
            for v, ii in by_variant.items():
                if len(ii) >= 2 and np.std(yva[ii]) > 0:
                    pm_rhos.append(float(np.mean(yva[ii])))
            if pm_rhos:
                pred_means = [float(np.mean(pred[ii])) for v, ii in by_variant.items() if len(ii) >= 2]
                pair_mean = float(spearmanr(pred_means, pm_rhos).statistic)

        results["tasks"][label] = {
            "n_train": len(tr), "n_val": len(va), "ridge_alpha": alpha,
            "spearman": rho, "bootstrap_ci95": ci,
            "pair_mean_spearman": pair_mean, "n_boot_groups": len(gkeys),
        }
        print(f"{label}: rho={rho:.4f} CI={ci} pair_mean={pair_mean}", flush=True)
        del emb, emap

    with open(OUT / "mrnabert_raw_frozen9_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print("written:", OUT / "mrnabert_raw_frozen9_results.json")


if __name__ == "__main__":
    main()
