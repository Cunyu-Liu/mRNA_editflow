"""B0-X effect baselines for T5_SOURCE_RELATIVE_EFFECT.

Evaluates a fixed set of effect baselines on the ACTIVE effect pairs using the
S4 (leave-one-study-out) macro structure.  All baselines use the same records,
split, target endpoint, seed, and (fixed) tuning configuration so the ceiling
comparison is fair.  NN baselines run on CUDA when available; otherwise they
are recorded as not_run with a reason (never silently run on CPU and reported
as GPU).

Metrics (macro over test metric-contexts, averaged over S4 folds):
  - macro_delta_spearman   : Spearman(pred_delta, true_delta) per context
  - macro_sign_accuracy    : frac(sign(pred)==sign(true))
  - macro_top10_enrichment : mean(true|top-10 predicted) / mean(true|top-10 true)

Usage:
    python -m scripts.b0x.run_effect_baselines --dataset artifacts/b0x/effect_dataset.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import features as F  # noqa: E402

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_dataset(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("delta") is not None:
                rows.append(r)
    return rows


def _feature_row(r: dict):
    fx = F.extract_features(r["source_sequence"], r["candidate_sequence"], r["edit_list"])
    return fx


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if len(a) < 3:
        return None
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    rho, _ = spearmanr(a, b)
    return float(rho) if not np.isnan(rho) else None


def context_metrics(true_delta: np.ndarray, pred: np.ndarray) -> dict:
    true_delta = np.asarray(true_delta, dtype=float)
    pred = np.asarray(pred, dtype=float)
    m = {"n": int(len(true_delta))}
    m["delta_spearman"] = _spearman(pred, true_delta)
    # sign accuracy
    nz = true_delta != 0
    if nz.sum() > 0:
        m["sign_accuracy"] = float(np.mean(np.sign(pred[nz]) == np.sign(true_delta[nz])))
    else:
        m["sign_accuracy"] = None
    # top-10 enrichment
    k = min(10, len(true_delta))
    if k >= 1:
        pred_top = np.argsort(pred)[-k:][::-1]
        true_top = np.argsort(true_delta)[-k:][::-1]
        denom = float(np.mean(true_delta[true_top]))
        if denom != 0:
            m["top10_enrichment"] = float(np.mean(true_delta[pred_top]) / denom)
        else:
            m["top10_enrichment"] = None
    else:
        m["top10_enrichment"] = None
    return m


def _macro(results: List[dict]) -> dict:
    def _avg(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "macro_delta_spearman": _avg("delta_spearman"),
        "macro_sign_accuracy": _avg("sign_accuracy"),
        "macro_top10_enrichment": _avg("top10_enrichment"),
        "n_contexts": len(results),
        "n_records": int(sum(r.get("n", 0) for r in results)),
    }


# ---------------------------------------------------------------------------
# Baselines.  Each returns a vector of predictions for the test records given
# train records (dicts) and test records (dicts).
# ---------------------------------------------------------------------------


def bl_mean(train, test):
    mu = float(np.mean([r["delta"] for r in train])) if train else 0.0
    return np.full(len(test), mu)


def bl_source_mean(train, test):
    src_map = defaultdict(list)
    for r in train:
        src_map[r["source_id"]].append(r["delta"])
    src_mean = {k: float(np.mean(v)) for k, v in src_map.items()}
    global_mean = float(np.mean([r["delta"] for r in train])) if train else 0.0
    return np.array([src_mean.get(r["source_id"], global_mean) for r in test])


def _concat_features(rows, keys):
    return np.array([np.concatenate([k for k in [F.extract_features(r["source_sequence"],
                                                                    r["candidate_sequence"],
                                                                    r["edit_list"])[fk] for fk in keys]])
                     for r in rows], dtype=np.float32)


def bl_ridge_feature(train, test, keys=("source_feat", "candidate_feat", "diff_feat", "edit_feat")):
    from sklearn.linear_model import RidgeCV
    X = _concat_features(train, keys)
    y = np.array([r["delta"] for r in train])
    Xt = _concat_features(test, keys)
    model = RidgeCV(alphas=np.logspace(-3, 3, 7)).fit(X, y)
    return model.predict(Xt)


def bl_ridge_kmers(train, test):
    from sklearn.linear_model import RidgeCV
    X = np.array([F.kmers_vector(r["source_sequence"], r["candidate_sequence"]) for r in train])
    y = np.array([r["delta"] for r in train])
    Xt = np.array([F.kmers_vector(r["source_sequence"], r["candidate_sequence"]) for r in test])
    model = RidgeCV(alphas=np.logspace(-3, 3, 7)).fit(X, y)
    return model.predict(Xt)


def bl_xgboost(train, test):
    import xgboost as xgb
    X = _concat_features(train, ("source_feat", "candidate_feat", "diff_feat", "edit_feat"))
    y = np.array([r["delta"] for r in train])
    Xt = _concat_features(test, ("source_feat", "candidate_feat", "diff_feat", "edit_feat"))
    model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.08,
                             subsample=0.8, colsample_bytree=0.8, seed=config.SEED,
                             n_jobs=os.cpu_count() or 1)
    model.fit(X, y)
    return model.predict(Xt)


def bl_diff_features(train, test):
    return bl_ridge_feature(train, test, keys=("diff_feat", "edit_feat"))


def bl_abs_candidate(train, test):
    """Predict candidate_value from candidate features; delta = cand_pred - source_value
    (source_value = 0 when the asset is a direct log2fc target)."""
    from sklearn.linear_model import RidgeCV
    X = np.array([F.sequence_features(r["candidate_sequence"]) for r in train])
    y = np.array([r["candidate_value"] if r["candidate_value"] is not None else 0.0 for r in train])
    Xt = np.array([F.sequence_features(r["candidate_sequence"]) for r in test])
    model = RidgeCV(alphas=np.logspace(-3, 3, 7)).fit(X, y)
    cand_pred = model.predict(Xt)
    src = np.array([(r["source_value"] if r["source_value"] is not None else 0.0) for r in test])
    return cand_pred - src


def bl_abs_minus_abs(train, test):
    """Predict candidate_value and source_value separately; delta = cand_pred - src_pred."""
    from sklearn.linear_model import RidgeCV
    Xc = np.array([F.sequence_features(r["candidate_sequence"]) for r in train])
    Xs = np.array([F.sequence_features(r["source_sequence"]) for r in train])
    yc = np.array([r["candidate_value"] if r["candidate_value"] is not None else 0.0 for r in train])
    ys = np.array([r["source_value"] if r["source_value"] is not None else 0.0 for r in train])
    Xct = np.array([F.sequence_features(r["candidate_sequence"]) for r in test])
    Xst = np.array([F.sequence_features(r["source_sequence"]) for r in test])
    mc = RidgeCV(alphas=np.logspace(-3, 3, 7)).fit(Xc, yc)
    ms = RidgeCV(alphas=np.logspace(-3, 3, 7)).fit(Xs, ys)
    return mc.predict(Xct) - ms.predict(Xst)


# --- torch NN baselines -----------------------------------------------------

def _device():
    import torch
    if torch.cuda.is_available():
        dev = os.environ.get("CUDA_DEVICE", "cuda")
        return torch.device(dev if dev.startswith("cuda") else "cuda")
    return None


def _to_onehot_pairs(rows):
    src = np.array([F.one_hot(r["source_sequence"]) for r in rows], dtype=np.float32)
    cand = np.array([F.one_hot(r["candidate_sequence"]) for r in rows], dtype=np.float32)
    return src, cand


def _train_nn(model, src, cand, y, device, epochs=6, batch=256, lr=1e-3):
    import torch
    torch.manual_seed(config.SEED)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.MSELoss()
    n = len(y)
    idx = torch.randperm(n, device=device)
    src_t = torch.from_numpy(src).to(device)
    cand_t = torch.from_numpy(cand).to(device)
    y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=torch.device("cpu"))
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            s = src_t[b]; c = cand_t[b]; t = y_t[b]
            opt.zero_grad()
            pred = model(s, c).squeeze(-1)
            loss = lossf(pred, t)
            loss.backward()
            opt.step()
    return model


def _nn_predict(model, src, cand, device):
    import torch
    model.eval()
    with torch.no_grad():
        s = torch.from_numpy(src).to(device)
        c = torch.from_numpy(cand).to(device)
        out = model(s, c).squeeze(-1)
        return out.cpu().numpy()


def _make_nn_runner(kind):
    def run(train, test):
        import torch
        dev = _device()
        if dev is None:
            raise RuntimeError("CUDA unavailable")
        src, cand = _to_onehot_pairs(train)
        st, ct = _to_onehot_pairs(test)
        y = np.array([r["delta"] for r in train])
        # compute cap for runtime = all train (no subsample) but small model
        model = _build_model(kind)
        _train_nn(model, src, cand, y, dev)
        return _nn_predict(model, st, ct, dev)
    return run


def _build_model(kind):
    import torch
    import torch.nn as nn
    L = config.MAX_SEQ_LEN
    C = 4

    if kind == "cnn":
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(C * 2, 32, kernel_size=5, padding=2)
                self.mp = nn.AdaptiveMaxPool1d(1)
                self.head = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1))

            def forward(self, s, c):
                x = torch.cat([s, c], dim=-1).permute(0, 2, 1)
                x = torch.relu(self.conv(x))
                x = self.mp(x).squeeze(-1)
                return self.head(x)
        return Net()

    if kind == "transformer":
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                d = 64
                self.cat_proj = nn.Linear(C * 2, d)
                self.pos = nn.Parameter(torch.zeros(1, L, d))
                layer = nn.TransformerEncoderLayer(d_model=d, nhead=2, dim_feedforward=128,
                                                   batch_first=True)
                self.enc = nn.TransformerEncoder(layer, num_layers=1)
                self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))

            def forward(self, s, c):
                x = torch.cat([s, c], dim=-1)
                x = self.cat_proj(x) + self.pos
                x = self.enc(x)
                x = x.mean(dim=1)
                return self.head(x)
        return Net()

    if kind == "siamese":
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                tip = L * C
                self.branch = nn.Sequential(nn.Linear(tip, 128), nn.ReLU())  # shared
                self.head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

            def forward(self, s, c):
                fs = self.branch(s.flatten(1))
                fc = self.branch(c.flatten(1))
                return self.head(torch.cat([fs, fc], dim=-1))
        return Net()

    if kind == "full_pair":
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                tip = L * C * 2
                self.net = nn.Sequential(nn.Linear(tip, 256), nn.ReLU(),
                                         nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

            def forward(self, s, c):
                return self.net(torch.cat([s, c], dim=-1).flatten(1))
        return Net()

    raise ValueError(kind)


NN_BASELINES = {
    "small_cnn": _make_nn_runner("cnn"),
    "small_transformer": _make_nn_runner("transformer"),
    "siamese_encoder": _make_nn_runner("siamese"),
    "full_pair_encoder": _make_nn_runner("full_pair"),
}


def _run_baseline(name, train, test):
    if name == "mean":
        return bl_mean(train, test)
    if name == "source_mean":
        return bl_source_mean(train, test)
    if name == "feature_ridge":
        return bl_ridge_feature(train, test)
    if name == "kmer_ridge":
        return bl_ridge_kmers(train, test)
    if name == "xgboost":
        return bl_xgboost(train, test)
    if name == "difference_features":
        return bl_diff_features(train, test)
    if name == "abs_candidate":
        return bl_abs_candidate(train, test)
    if name == "abs_candidate_minus_abs_source":
        return bl_abs_minus_abs(train, test)
    if name in NN_BASELINES:
        return NN_BASELINES[name](train, test)
    raise ValueError(name)


BASELINE_NAMES = [
    "mean", "source_mean", "feature_ridge", "kmer_ridge", "xgboost",
    "difference_features", "abs_candidate", "abs_candidate_minus_abs_source",
    "small_cnn", "small_transformer", "siamese_encoder", "full_pair_encoder",
]


def _metric_context(r: dict) -> str:
    return f"{r['study']}|{r['endpoint']}"


# ---------------------------------------------------------------------------
# S4 evaluation
# ---------------------------------------------------------------------------


def run_benchmark(rows: List[dict], benchmark: str) -> dict:
    studies = sorted({r["study"] for r in rows})
    per_bl: Dict[str, List[dict]] = defaultdict(list)  # bl -> list of context metrics
    fold_info = []
    for held_out in studies:
        train = [r for r in rows if r["study"] != held_out]
        test = [r for r in rows if r["study"] == held_out]
        # group test by metric-context
        contexts = defaultdict(list)
        for r in test:
            contexts[_metric_context(r)].append(r)
        for bl in BASELINE_NAMES:
            t0 = time.time()
            try:
                pred = _run_baseline(bl, train, test)
                status = "ok"
            except Exception as e:  # noqa: BLE001
                pred = np.full(len(test), np.nan)
                status = f"not_run:{type(e).__name__}:{e}"
            for ctx, recs in contexts.items():
                idx = [i for i, r in enumerate(test) if _metric_context(r) == ctx]
                true_d = np.array([r["delta"] for r in recs])
                pred_d = pred[idx]
                cm = context_metrics(true_d, pred_d)
                cm["fold_study"] = held_out
                cm["context"] = ctx
                if status != "ok":
                    cm["baseline_status"] = status
                per_bl[bl].append(cm)
        fold_info.append({"held_out_study": held_out, "n_train": len(train), "n_test": len(test)})

    results = {}
    for bl, cms in per_bl.items():
        m = _macro(cms)
        m["baseline_status"] = "ok" if all(cm.get("baseline_status") is None for cm in cms) else \
            next((cm["baseline_status"] for cm in cms if cm.get("baseline_status")), "ok")
        results[bl] = m
    return {"benchmark": benchmark, "studies": studies, "folds": fold_info, "baselines": results}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("artifacts/b0x/effect_dataset.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/b0x"))
    args = ap.parse_args()

    rows = load_dataset(args.dataset)
    print(f"loaded {len(rows)} delta-defined records")

    out = {"seed": config.SEED, "split": config.PRIMARY_SPLIT, "benchmarks": {}}
    for benchmark in ("5U-A1", "3U-A1"):
        bench_rows = [r for r in rows if r["benchmark"] == benchmark]
        print(f"[{benchmark}] {len(bench_rows)} records")
        res = run_benchmark(bench_rows, benchmark)
        out["benchmarks"][benchmark] = res

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "effect_baseline_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print_table(out)
    return 0


def print_table(out):
    for benchmark, res in out["benchmarks"].items():
        print(f"\n=== {benchmark} ===")
        print(f"{'baseline':<32}{'spearman':>10}{'signacc':>10}{'top10':>10}{'status':>14}")
        for bl, m in res["baselines"].items():
            sp = "n/a" if m["macro_delta_spearman"] is None else f"{m['macro_delta_spearman']:.4f}"
            sa = "n/a" if m["macro_sign_accuracy"] is None else f"{m['macro_sign_accuracy']:.4f}"
            t10 = "n/a" if m["macro_top10_enrichment"] is None else f"{m['macro_top10_enrichment']:.4f}"
            status = m.get("baseline_status", "ok")
            print(f"{bl:<32}{sp:>10}{sa:>10}{t10:>10}{status:>14}")


if __name__ == "__main__":
    raise SystemExit(main())