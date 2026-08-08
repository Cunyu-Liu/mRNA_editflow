"""M4 evaluation: macro delta Spearman, macro sign accuracy, top-10% enrichment,
plus the abs_candidate reference baseline on the SAME S4 test folds.

Metrics are macro-averaged over metric-contexts (study|endpoint) across S4
folds, matching the B0-X convention.  Both historical signed-ratio enrichment
headlines are held after the R3-P0 monotonicity audit; oracle-normalized excess
gain is retained only as a diagnostic with no post-hoc threshold.
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
from .config import NUC_ORDER, NUC_TO_IDX, SIGN_DECISION_RULE
from .model import calibrated_sign_probability, standardized_delta_logit


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


def context_metrics(true_delta, pred, sign_prediction=None,
                    sign_rule: Optional[str] = None) -> dict:
    true_delta = np.asarray(true_delta, dtype=float)
    pred = np.asarray(pred, dtype=float)
    m = {"n": int(len(true_delta))}
    m["delta_spearman"] = _spearman(pred, true_delta)
    nz = true_delta != 0
    if nz.sum() > 0:
        signed = np.asarray(sign_prediction if sign_prediction is not None else pred)
        if signed.shape != pred.shape:
            raise ValueError("sign_prediction must align with pred")
        m["sign_accuracy"] = float(
            np.mean(np.sign(signed[nz]) == np.sign(true_delta[nz])))
        m["sign_decision_rule"] = (
            sign_rule or "mean_delta_sign_legacy_or_baseline")
    else:
        m["sign_accuracy"] = None
    # The historical fixed-top10 signed oracle ratio is unsafe: if all effects
    # are negative, a worse selected mean divided by a negative oracle mean can
    # exceed 1.  Preserve it only under an explicit legacy name and HOLD the
    # publication/gate headline.  A monotone excess-gain diagnostic is reported
    # without introducing any result-dependent threshold.
    k = min(10, len(true_delta))
    if 1 <= k < len(true_delta):
        pred_top = np.argsort(pred)[-k:][::-1]
        true_top = np.argsort(true_delta)[-k:][::-1]
        fixed_pred_mean = float(np.mean(true_delta[pred_top]))
        fixed_oracle_mean = float(np.mean(true_delta[true_top]))
        fixed_overall = float(np.mean(true_delta))
        m["top10_enrichment_legacy_signed_oracle_ratio"] = (
            fixed_pred_mean / fixed_oracle_mean
            if abs(fixed_oracle_mean) > 1e-12 else None
        )
        fixed_excess_denom = fixed_oracle_mean - fixed_overall
        m["top10_oracle_normalized_excess_gain"] = (
            (fixed_pred_mean - fixed_overall) / fixed_excess_denom
            if fixed_excess_denom > 1e-12 else None
        )
    else:
        m["top10_enrichment_legacy_signed_oracle_ratio"] = None
        m["top10_oracle_normalized_excess_gain"] = None
    m["top10_enrichment"] = None
    m["top10_enrichment_status"] = (
        "HOLD_SIGNED_ORACLE_RATIO_NON_MONOTONE_ON_NEGATIVE_EFFECTS"
    )
    # Preserve the historical signed-mean ratio only as an explicitly unsafe
    # diagnostic.  With negative or near-zero overall means it reverses order or
    # explodes, so it cannot satisfy the frozen threshold after R1 audit.
    kp = max(1, math.ceil(len(true_delta) * 0.10))
    pred_top = np.argsort(pred)[-kp:][::-1]
    oracle_top = np.argsort(true_delta)[-kp:][::-1]
    overall = float(np.mean(true_delta))
    pred_mean = float(np.mean(true_delta[pred_top]))
    oracle_mean = float(np.mean(true_delta[oracle_top]))
    m["top10pct_enrichment_legacy_signed_mean_ratio"] = (
        pred_mean / overall if abs(overall) > 1e-12 else None)
    excess_denom = oracle_mean - overall
    m["top10pct_oracle_normalized_excess_gain"] = (
        (pred_mean - overall) / excess_denom
        if excess_denom > 1e-12 else None)
    m["top10pct_enrichment"] = None
    m["top10pct_enrichment_status"] = (
        "HOLD_REQUIRES_PREREGISTERED_STABLE_SIGNED_GAIN_DEFINITION")
    m["metric_contract_version"] = "m4_effect_r3_p0_signed_gain_audit_v1"
    return m


def macro_metrics(cms: List[dict]) -> dict:
    def _avg(key):
        vals = [c[key] for c in cms if c.get(key) is not None]
        return float(np.mean(vals)) if vals else None
    return {
        "macro_delta_spearman": _avg("delta_spearman"),
        "macro_sign_accuracy": _avg("sign_accuracy"),
        "macro_top10_enrichment": _avg("top10_enrichment"),
        "macro_top10_oracle_normalized_excess_gain": _avg(
            "top10_oracle_normalized_excess_gain"),
        "legacy_macro_top10_signed_oracle_ratio": _avg(
            "top10_enrichment_legacy_signed_oracle_ratio"),
        "macro_top10pct_enrichment": _avg("top10pct_enrichment"),
        "macro_top10pct_oracle_normalized_excess_gain": _avg(
            "top10pct_oracle_normalized_excess_gain"),
        "legacy_macro_top10pct_signed_mean_ratio": _avg(
            "top10pct_enrichment_legacy_signed_mean_ratio"),
        "top10pct_enrichment_gate_status": (
            "HOLD_REQUIRES_PREREGISTERED_STABLE_SIGNED_GAIN_DEFINITION"),
        "top10_enrichment_gate_status": (
            "HOLD_SIGNED_ORACLE_RATIO_NON_MONOTONE_ON_NEGATIVE_EFFECTS"),
        "n_contexts": len(cms),
        "n_records": int(sum(c.get("n", 0) for c in cms)),
    }


# ---------------------------------------------------------------------------
# model prediction / abs_candidate baseline
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_model_outputs(model, test_rows, vocab, cfg, device,
                          force_unk_study: bool = True) -> dict:
    """Predict delta for test rows using the trained model.  If the model was
    trained on candidate_value (TARGET="candidate_value"), delta = mean - MEASURED
    source_value (honest anchor setting, disclosed in report).

    S4's held-out study is routed to the trainable unknown-context embedding by
    default.  This prevents evaluation from reading an embedding that was
    allocated globally but never updated by that fold's training data.
    """
    sign_rule = getattr(cfg, "SIGN_DECISION_RULE", None)
    if sign_rule != SIGN_DECISION_RULE:
        raise ValueError(
            "R3-P0 evaluation requires the frozen predicted-delta sign rule"
        )
    temperature = getattr(cfg, "SIGN_TEMPERATURE", None)
    if (not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature)) or float(temperature) <= 0):
        raise ValueError("checkpoint sign temperature must be finite and positive")
    temperature = float(temperature)
    model.eval()
    ds = EffectDataset(test_rows, vocab, target=cfg.TARGET,
                       force_unk_study=force_unk_study)
    loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    preds, sign_logits, sign_probabilities = [], [], []
    for b in loader:
        bb = {k: v.to(device) for k, v in b.items()}
        out = model(bb["src"], bb["cand"], bb["edit"], bb["study"],
                    bb["endpoint"], bb["bench"],
                    src_pad_mask=bb.get("src_pad_mask"),
                    cand_pad_mask=bb.get("cand_pad_mask"))
        delta = out["mean"]
        if cfg.TARGET == "candidate_value" and cfg.ANCHOR_AT_TEST:
            anchor = bb["source_value"]
            delta = delta - anchor
        raw_sign_logit = standardized_delta_logit(delta, out["logvar"])
        preds.append(delta.cpu().numpy())
        sign_logits.append(raw_sign_logit.cpu().numpy())
        sign_probabilities.append(
            calibrated_sign_probability(
                delta, out["logvar"], temperature).cpu().numpy()
        )
    raw = np.concatenate(sign_logits)
    return {
        "delta": np.concatenate(preds),
        "sign_logit": raw,
        "sign_probability": np.concatenate(sign_probabilities),
        # Hard sign is computed before probability-only positive-temperature
        # calibration, so temperature can never change this decision.
        "sign_decision": np.where(raw >= 0.0, 1.0, -1.0),
        "sign_decision_rule": sign_rule,
    }


def predict_model(model, test_rows, vocab, cfg, device,
                  force_unk_study: bool = True) -> np.ndarray:
    return predict_model_outputs(
        model, test_rows, vocab, cfg, device,
        force_unk_study=force_unk_study)["delta"]


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
    if any(r.get("candidate_value") is None for r in train_rows):
        raise ValueError("abs_candidate baseline refuses missing candidate_value")
    if any(r.get("source_value") is None for r in test_rows):
        raise ValueError("abs_candidate baseline refuses missing source_value")
    X = np.array([_sequence_features(r["candidate_sequence"]) for r in train_rows])
    y = np.array([float(r["candidate_value"]) for r in train_rows])
    Xt = np.array([_sequence_features(r["candidate_sequence"]) for r in test_rows])
    model = RidgeCV(alphas=np.logspace(-3, 3, 7)).fit(X, y)
    cand_pred = model.predict(Xt)
    src = np.array([float(r["source_value"]) for r in test_rows])
    return cand_pred - src


def run_fold_evaluation(model, fold, vocab, cfg, device) -> Dict:
    """Evaluate the trained model on a held-out-study fold.  Returns per-context
    metrics (model + abs_candidate baseline on the same test rows)."""
    test = fold["test"]
    contexts = defaultdict(list)
    for r in test:
        contexts[metric_context(r)].append(r)
    model_outputs = predict_model_outputs(model, test, vocab, cfg, device)
    pred_model = model_outputs["delta"]
    pred_abs = abs_candidate_baseline(fold["train"], test)
    train_endpoints = {str(row.get("endpoint") or "")
                       for row in fold["train"]}
    test_endpoints = {str(row.get("endpoint") or "") for row in test}
    unseen_test_endpoints = sorted(test_endpoints - train_endpoints)

    model_cms, abs_cms = [], []
    for ctx, recs in contexts.items():
        idx = [i for i, r in enumerate(test) if metric_context(r) == ctx]
        true_d = np.array([r["delta"] for r in recs])
        cm = context_metrics(
            true_d, pred_model[idx],
            sign_prediction=model_outputs["sign_decision"][idx],
            sign_rule=model_outputs["sign_decision_rule"],
        )
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
        "endpoint_routing": {
            "routing_policy": getattr(cfg, "ENDPOINT_HEAD_ROUTING", None),
            "unseen_test_endpoints": unseen_test_endpoints,
            "all_held_out_endpoints_have_training_support": (
                not unseen_test_endpoints),
            "h1_claim_status": (
                "DEVELOPMENT_ONLY_UNSEEN_HELD_OUT_ENDPOINTS"
                if unseen_test_endpoints else "EVALUATION_ELIGIBLE_NOT_GO"),
        },
        "model": model_cms,
        "abs_candidate": abs_cms,
    }


def macro_over_folds(fold_evals: List[Dict], key: str) -> dict:
    per_study = {fe["held_out_study"]: macro_metrics(fe[key]) for fe in fold_evals}
    study_metrics = list(per_study.values())

    def _study_avg(metric):
        values = [entry.get(metric) for entry in study_metrics
                  if entry.get(metric) is not None]
        return float(np.mean(values)) if values else None

    m = {
        "macro_delta_spearman": _study_avg("macro_delta_spearman"),
        "macro_sign_accuracy": _study_avg("macro_sign_accuracy"),
        "macro_top10_enrichment": _study_avg("macro_top10_enrichment"),
        "macro_top10_oracle_normalized_excess_gain": _study_avg(
            "macro_top10_oracle_normalized_excess_gain"),
        "legacy_macro_top10_signed_oracle_ratio": _study_avg(
            "legacy_macro_top10_signed_oracle_ratio"),
        "macro_top10pct_enrichment": _study_avg("macro_top10pct_enrichment"),
        "macro_top10pct_oracle_normalized_excess_gain": _study_avg(
            "macro_top10pct_oracle_normalized_excess_gain"),
        "legacy_macro_top10pct_signed_mean_ratio": _study_avg(
            "legacy_macro_top10pct_signed_mean_ratio"),
        "top10pct_enrichment_gate_status": (
            "HOLD_REQUIRES_PREREGISTERED_STABLE_SIGNED_GAIN_DEFINITION"),
        "top10_enrichment_gate_status": (
            "HOLD_SIGNED_ORACLE_RATIO_NON_MONOTONE_ON_NEGATIVE_EFFECTS"),
        "n_studies": len(per_study),
        "n_contexts": sum(len(fe[key]) for fe in fold_evals),
        "n_records": int(sum(cm.get("n", 0) for fe in fold_evals for cm in fe[key])),
    }
    m["per_study"] = per_study
    return m
