"""M4 evaluation: macro delta Spearman, macro sign accuracy, top-10% enrichment,
plus the abs_candidate reference baseline on the SAME S4 test folds.

Metrics are macro-averaged over metric-contexts (study|endpoint) across S4
folds, matching the B0-X convention.  top10_enrichment (fixed top-10) is kept
for horizontal comparison with B0-X; top10pct_enrichment (top-10% of context)
is the pre-registered gate enrichment metric.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from .dataset import EffectDataset
from .config import NUC_ORDER, NUC_TO_IDX


def metric_context(r: dict) -> str:
    return f"{r['study']}|{r['endpoint']}"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def _spearman(a, b) -> Optional[float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return None
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    rho, _ = spearmanr(a, b)
    return float(rho) if not np.isnan(rho) else None


def context_metrics(true_delta, pred) -> dict:
    true_delta = np.asarray(true_delta, dtype=float)
    pred = np.asarray(pred, dtype=float)
    m = {"n": int(len(true_delta))}
    m["delta_spearman"] = _spearman(pred, true_delta)
    nz = true_delta != 0
    if nz.sum() > 0:
        m["sign_accuracy"] = float(np.mean(np.sign(pred[nz]) == np.sign(true_delta[nz])))
    else:
        m["sign_accuracy"] = None
    # top-10 (fixed, B0-X convention)
    k = min(10, len(true_delta))
    if k >= 1:
        pred_top = np.argsort(pred)[-k:][::-1]
        true_top = np.argsort(true_delta)[-k:][::-1]
        denom = float(np.mean(true_delta[true_top]))
        m["top10_enrichment"] = float(np.mean(true_delta[pred_top]) / denom) if denom != 0 else None
    else:
        m["top10_enrichment"] = None
    # top-10% (percentile) ENRICHMENT-OVER-RANDOM (contract gate metric
    #   §3.2 / §10): enrichment@q = mean(true_delta[pred_top_q]) / mean(true_delta[all]).
    #   Values >1 mean the model concentrates more true effect in its predicted
    #   top-10% than the random expectation.  This is the metric the pre-registered
    #   gate threshold (>=1.50) refers to.  [The old pred_top/true_top ratio is
    #   bounded by 1.0 and could never reach 1.50 -> definition bug, corrected.]
    kp = max(1, math.ceil(len(true_delta) * 0.10))
    pred_top = np.argsort(pred)[-kp:][::-1]
    overall = float(np.mean(true_delta))
    m["top10pct_enrichment"] = (float(np.mean(true_delta[pred_top])) / overall
                                if overall != 0 else None)
    return m


def macro_metrics(cms: List[dict]) -> dict:
    def _avg(key):
        vals = [c[key] for c in cms if c.get(key) is not None]
        return float(np.mean(vals)) if vals else None
    return {
        "macro_delta_spearman": _avg("delta_spearman"),
        "macro_sign_accuracy": _avg("sign_accuracy"),
        "macro_top10_enrichment": _avg("top10_enrichment"),
        "macro_top10pct_enrichment": _avg("top10pct_enrichment"),
        "n_contexts": len(cms),
        "n_records": int(sum(c.get("n", 0) for c in cms)),
    }


# ---------------------------------------------------------------------------
# model prediction / abs_candidate baseline
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_model(model, test_rows, vocab, cfg, device) -> np.ndarray:
    """Predict delta for test rows using the trained model.  If the model was
    trained on candidate_value (TARGET="candidate_value"), delta = mean - MEASURED
    source_value (honest anchor setting, disclosed in report)."""
    model.eval()
    ds = EffectDataset(test_rows, vocab, target=cfg.TARGET)
    loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    preds = []
    for b in loader:
        bb = {k: v.to(device) for k, v in b.items()}
        out = model(bb["src"], bb["cand"], bb["edit"], bb["study"],
                    bb["endpoint"], bb["bench"])
        delta = out["mean"].cpu().numpy()
        if cfg.TARGET == "candidate_value" and cfg.ANCHOR_AT_TEST:
            anchor = bb["source_value"].cpu().numpy()
            delta = delta - anchor
        preds.append(delta)
    return np.concatenate(preds)


def _sequence_features(seq: str) -> np.ndarray:
    """20-dim deterministic sequence features (mirror scripts/b0x/features.py)."""
    import math
    from collections import Counter
    seq = (seq or "").upper().replace("T", "U")
    n = len(seq)
    if n == 0:
        return np.zeros(20, dtype=np.float32)
    c = Counter(seq)
    freqs = np.array([c.get(ch, 0) / n for ch in NUC_ORDER], dtype=np.float32)
    gc = float(freqs[NUC_TO_IDX["G"]] + freqs[NUC_TO_IDX["C"]])
    first = seq[:10]
    gc_first10 = (first.count("G") + first.count("C")) / max(len(first), 1)
    aug = seq.find("AUG")
    aug_norm = aug / n if aug >= 0 else -1.0
    top_dinuc = ["AA", "UU", "GC", "CG", "AU", "UA"]
    din = Counter(seq[i:i + 2] for i in range(n - 1))
    tot = max(sum(din.values()), 1)
    din_freqs = np.array([din.get(d, 0) / tot for d in top_dinuc], dtype=np.float32)
    runs = np.array([max((len(s) for s in seq.split(ch)), default=0) / n
                     for ch in NUC_ORDER], dtype=np.float32)
    ent = 0.0
    for f in freqs:
        if f > 0:
            ent -= f * math.log2(f)
    diversity = np.array([len(set(seq)) / len(NUC_ORDER)], dtype=np.float32)
    return np.concatenate([
        np.array([n / 100.0, gc, gc_first10, aug_norm], dtype=np.float32),
        freqs, din_freqs, runs, np.array([ent], dtype=np.float32), diversity
    ]).astype(np.float32)


def abs_candidate_baseline(train_rows, test_rows) -> np.ndarray:
    """Reference ceiling baseline (B0-X abs_candidate): predict candidate_value
    from candidate features, delta = cand_pred - MEASURED source_value."""
    from sklearn.linear_model import RidgeCV
    X = np.array([_sequence_features(r["candidate_sequence"]) for r in train_rows])
    y = np.array([r["candidate_value"] if r["candidate_value"] is not None else 0.0
                  for r in train_rows])
    Xt = np.array([_sequence_features(r["candidate_sequence"]) for r in test_rows])
    model = RidgeCV(alphas=np.logspace(-3, 3, 7)).fit(X, y)
    cand_pred = model.predict(Xt)
    src = np.array([r["source_value"] if r["source_value"] is not None else 0.0
                    for r in test_rows])
    return cand_pred - src


def run_fold_evaluation(model, fold, vocab, cfg, device) -> Dict:
    """Evaluate the trained model on a held-out-study fold.  Returns per-context
    metrics (model + abs_candidate baseline on the same test rows)."""
    test = fold["test"]
    contexts = defaultdict(list)
    for r in test:
        contexts[metric_context(r)].append(r)
    pred_model = predict_model(model, test, vocab, cfg, device)
    pred_abs = abs_candidate_baseline(fold["train"], test)

    model_cms, abs_cms = [], []
    for ctx, recs in contexts.items():
        idx = [i for i, r in enumerate(test) if metric_context(r) == ctx]
        true_d = np.array([r["delta"] for r in recs])
        cm = context_metrics(true_d, pred_model[idx])
        cm["fold_study"] = fold["held_out_study"]
        cm["context"] = ctx
        model_cms.append(cm)
        cm_abs = context_metrics(true_d, pred_abs[idx])
        cm_abs["fold_study"] = fold["held_out_study"]
        cm_abs["context"] = ctx
        abs_cms.append(cm_abs)

    return {
        "held_out_study": fold["held_out_study"],
        "n_train": len(fold["train"]),
        "n_test": len(test),
        "model": model_cms,
        "abs_candidate": abs_cms,
    }


def macro_over_folds(fold_evals: List[Dict], key: str) -> dict:
    cms = [cm for fe in fold_evals for cm in fe[key]]
    m = macro_metrics(cms)
    m["per_study"] = {fe["held_out_study"]: macro_metrics(fe[key]) for fe in fold_evals}
    return m
