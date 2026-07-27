#!/usr/bin/env python3
"""Unified Benchmark v2 metrics and preregistered trivial-baseline gate.

The default command evaluates only ``val``. Final roles require
``--allow-final-labels`` and are never opened implicitly. Fitted baselines that
need a model backend use CUDA when available; CPU-only model fitting is marked
``not_run`` under the server GPU execution contract.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - the remote runtime has numpy
    np = None  # type: ignore


ROLES = ["train", "val", "test_id", "test_family", "test_context", "test_assay", "test_ood"]
FINAL_ROLES = set(ROLES[2:])
LOCAL_ROLES = {"train", "val", "test_id", "test_family", "test_ood"}
BASELINE_NAMES = [
    "mean", "source_mean", "gc_delta", "position_only", "ref_alt_transition",
    "local_kmer", "kozak_only", "uaug_only", "RNAfold_delta", "ridge",
    "gradient_boosted_trees", "small_CNN", "absolute_predictor",
]


def finite(value: object) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def seq(rec: Mapping[str, object], key: str) -> str:
    return str(rec.get(key) or "").upper().replace("T", "U")


def edits(rec: Mapping[str, object]) -> List[Mapping[str, object]]:
    value = rec.get("edit_list")
    return [e for e in value if isinstance(e, dict)] if isinstance(value, list) else []


def target(rec: Mapping[str, object]) -> Optional[float]:
    # Raw measured_delta remains the provenance field; scored_delta aligns
    # cross-assay/cargo local effects for pooled Task 1--3 metrics.
    scored = finite(rec.get("scored_delta"))
    return scored if scored is not None else finite(rec.get("measured_delta"))


def load_records(root: Path, role: str, *, allow_final_labels: bool, task_kind: Optional[str] = "local_delta") -> List[Dict]:
    manifest = json.loads((root / "manifests" / f"{role}.json").read_text())
    if role in FINAL_ROLES and not allow_final_labels:
        raise PermissionError(f"{role} is final-test data; pass --allow-final-labels after freeze")
    wanted = {
        line.strip() for line in (root / str(manifest["index_path"])).read_text().splitlines()
        if line.strip()
    }
    if not wanted:
        return []
    records = []
    with (root / str(manifest["records_path"])).open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if str(rec.get("record_id")) not in wanted:
                continue
            if task_kind is not None and str(rec.get("task_kind")) != task_kind:
                continue
            if task_kind == "local_delta" and str(rec.get("confidence")) != "measured":
                continue
            records.append(rec)
    return records


def gc_fraction(value: str) -> float:
    return (value.count("G") + value.count("C")) / max(1, len(value))


def edit_positions(rec: Mapping[str, object]) -> List[int]:
    out = []
    for item in edits(rec):
        try:
            out.append(int(item.get("pos", 0)))
        except (TypeError, ValueError):
            continue
    return out


def kozak_score_raw(value: str) -> float:
    """Small deterministic Kozak motif score around the first AUG."""
    pos = value.find("AUG")
    if pos < 0:
        return 0.0
    context = value[max(0, pos - 6):pos + 5]
    consensus = "GCCRCCAUGG"
    score = 0.0
    start = max(0, 6 - pos)
    for i, expected in enumerate(consensus):
        j = start + i
        if j >= len(context):
            continue
        actual = context[j]
        if expected == "R":
            score += float(actual in "AG")
        else:
            score += float(actual == expected)
    return score / len(consensus)


def kozak_score(value: str) -> float:
    # Kept as a separate public helper name without using a non-ASCII symbol.
    return kozak_score_raw(value)


def uaug_count(value: str) -> float:
    return float(value.count("AUG"))


def transition_key(rec: Mapping[str, object]) -> str:
    vals = []
    for item in edits(rec):
        vals.append(f"{str(item.get('ref', '')).upper().replace('T', 'U')}>{str(item.get('alt', '')).upper().replace('T', 'U')}")
    return vals[0] if len(vals) == 1 else "multi"


def local_kmer_keys(rec: Mapping[str, object], radius: int = 3) -> List[str]:
    source = seq(rec, "source_sequence")
    keys = []
    for item in edits(rec):
        try:
            pos = int(item.get("pos", 0))
        except (TypeError, ValueError):
            continue
        ref = str(item.get("ref", "")).upper().replace("T", "U")
        alt = str(item.get("alt", "")).upper().replace("T", "U")
        flank = source[max(0, pos - radius):min(len(source), pos + radius + 1)]
        keys.append(f"{flank}:{ref}>{alt}")
    return keys or ["none"]


def feature_vector(rec: Mapping[str, object]) -> List[float]:
    source = seq(rec, "source_sequence")
    candidate = seq(rec, "candidate_sequence")
    positions = edit_positions(rec)
    transitions = ["A>U", "A>G", "A>C", "U>A", "U>G", "U>C", "G>A", "G>U", "G>C", "C>A", "C>U", "C>G", "multi"]
    key = transition_key(rec)
    one_hot = [1.0 if key == item else 0.0 for item in transitions]
    pos = sum(positions) / max(1, len(positions)) / max(1, len(source))
    return [
        1.0, gc_fraction(candidate) - gc_fraction(source),
        (len(candidate) - len(source)) / 100.0, float(len(positions)), pos,
        kozak_score(candidate) - kozak_score(source),
        uaug_count(candidate) - uaug_count(source),
        *one_hot,
    ]


def mean_or(values: Sequence[float], fallback: float = 0.0) -> float:
    return float(sum(values) / len(values)) if values else fallback


class Baseline:
    def __init__(self, name: str, fit: Callable[[List[Dict]], None], predict: Callable[[Mapping[str, object]], Tuple[float, float]], *, status: str = "run", reason: Optional[str] = None):
        self.name = name
        self._fit = fit
        self._predict = predict
        self.status = status
        self.reason = reason
        self.predictions: List[float] = []
        self.uncertainties: List[float] = []

    def fit(self, records: List[Dict]) -> None:
        if self.status == "run":
            self._fit(records)

    def predict(self, records: List[Dict]) -> Tuple[List[float], List[float]]:
        if self.status != "run":
            return [], []
        pairs = [self._predict(rec) for rec in records]
        return [float(p[0]) for p in pairs], [max(0.0, float(p[1])) for p in pairs]


def make_statistical_baselines(train: List[Dict]) -> Dict[str, Baseline]:
    y_train = [target(r) for r in train]
    y_values = [float(v) for v in y_train if v is not None]
    global_mean = mean_or(y_values)
    global_std = float(np.std(y_values)) if np is not None and y_values else 1.0
    global_std = max(global_std, 1e-6)

    def constant(value: float, uncertainty: float = global_std) -> Tuple[float, float]:
        return value, uncertainty

    baselines: Dict[str, Baseline] = {}
    baselines["mean"] = Baseline("mean", lambda _: None, lambda _: constant(global_mean))

    source_map: Dict[str, List[float]] = defaultdict(list)
    for r in train:
        if target(r) is not None:
            source_map[str(r.get("source_id"))].append(float(target(r)))
    source_values = {k: mean_or(v, 0.0) for k, v in source_map.items()}
    baselines["source_mean"] = Baseline(
        "source_mean", lambda _: None,
        lambda r: constant(source_values.get(str(r.get("source_id")), 0.0), global_std),
    )

    def fit_scalar(feature: Callable[[Mapping[str, object]], float]) -> Tuple[float, float]:
        xs = [feature(r) for r in train if target(r) is not None]
        ys = [float(target(r)) for r in train if target(r) is not None]
        if not xs or max(xs) - min(xs) < 1e-12:
            return 0.0, global_mean
        mx, my = mean_or(xs), mean_or(ys)
        den = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / max(den, 1e-12)
        return slope, my - slope * mx

    scalar_features = {
        "gc_delta": lambda r: gc_fraction(seq(r, "candidate_sequence")) - gc_fraction(seq(r, "source_sequence")),
        "kozak_only": lambda r: kozak_score(seq(r, "candidate_sequence")) - kozak_score(seq(r, "source_sequence")),
        "uaug_only": lambda r: uaug_count(seq(r, "candidate_sequence")) - uaug_count(seq(r, "source_sequence")),
    }
    for name, feature in scalar_features.items():
        slope, intercept = fit_scalar(feature)
        baselines[name] = Baseline(name, lambda _: None, lambda r, f=feature, a=slope, b=intercept: (a * f(r) + b, global_std))

    # Position-only uses train target averages over 20 normalized bins.
    pos_map: Dict[int, List[float]] = defaultdict(list)
    for r in train:
        if target(r) is None:
            continue
        for p in (edit_positions(r) or [0]):
            pos_map[min(19, int(20 * p / max(1, len(seq(r, "source_sequence")))))].append(float(target(r)))
    position_values = {k: mean_or(v, global_mean) for k, v in pos_map.items()}
    def position_prediction(r: Mapping[str, object]) -> Tuple[float, float]:
        values = []
        for p in (edit_positions(r) or [0]):
            bucket = min(19, int(20 * p / max(1, len(seq(r, "source_sequence")))))
            values.append(position_values.get(bucket, global_mean))
        return constant(mean_or(values, global_mean))
    baselines["position_only"] = Baseline(
        "position_only", lambda _: None,
        position_prediction,
    )

    transition_map: Dict[str, List[float]] = defaultdict(list)
    kmer_map: Dict[str, List[float]] = defaultdict(list)
    for r in train:
        if target(r) is None:
            continue
        transition_map[transition_key(r)].append(float(target(r)))
        for key in local_kmer_keys(r):
            kmer_map[key].append(float(target(r)))
    transition_values = {k: mean_or(v, global_mean) for k, v in transition_map.items()}
    kmer_values = {k: mean_or(v, global_mean) for k, v in kmer_map.items()}
    baselines["ref_alt_transition"] = Baseline(
        "ref_alt_transition", lambda _: None,
        lambda r: constant(transition_values.get(transition_key(r), global_mean)),
    )
    baselines["local_kmer"] = Baseline(
        "local_kmer", lambda _: None,
        lambda r: constant(mean_or([kmer_values.get(k, global_mean) for k in local_kmer_keys(r)], global_mean)),
    )

    rn_available = shutil.which("RNAfold") is not None
    if rn_available:
        def fold_energy(value: str) -> Optional[float]:
            try:
                proc = subprocess.run(["RNAfold", "--noPS"], input=value + "\n", text=True, capture_output=True, timeout=15, check=True)
            except (OSError, subprocess.SubprocessError):
                return None
            lines = proc.stdout.strip().splitlines()
            if len(lines) < 2:
                return None
            try:
                return float(lines[1].rsplit("(", 1)[1].split(")", 1)[0])
            except (IndexError, ValueError):
                return None
        def rn_predict(r: Mapping[str, object]) -> Tuple[float, float]:
            source_e, candidate_e = fold_energy(seq(r, "source_sequence")), fold_energy(seq(r, "candidate_sequence"))
            if source_e is None or candidate_e is None:
                return 0.0, global_std * 2
            return -(candidate_e - source_e), 0.0
        baselines["RNAfold_delta"] = Baseline("RNAfold_delta", lambda _: None, rn_predict)
    else:
        baselines["RNAfold_delta"] = Baseline("RNAfold_delta", lambda _: None, lambda _: (0.0, 0.0), status="not_run", reason="ViennaRNA RNAfold executable is unavailable")

    return baselines


def torch_baselines(train: List[Dict]) -> Dict[str, Baseline]:
    """Fit the small neural baselines only on CUDA."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return {
            "ridge": Baseline("ridge", lambda _: None, lambda _: (0.0, 0.0), status="not_run", reason="PyTorch unavailable"),
            "gradient_boosted_trees": Baseline("gradient_boosted_trees", lambda _: None, lambda _: (0.0, 0.0), status="not_run", reason="GPU model runtime unavailable"),
            "small_CNN": Baseline("small_CNN", lambda _: None, lambda _: (0.0, 0.0), status="not_run", reason="PyTorch unavailable"),
            "absolute_predictor": Baseline("absolute_predictor", lambda _: None, lambda _: (0.0, 0.0), status="not_run", reason="PyTorch unavailable"),
        }
    if not torch.cuda.is_available():
        reason = "CUDA unavailable; model-fitting baselines are fail-closed under the GPU execution contract"
        return {name: Baseline(name, lambda _: None, lambda _: (0.0, 0.0), status="not_run", reason=reason) for name in ("ridge", "gradient_boosted_trees", "small_CNN", "absolute_predictor")}

    device = torch.device("cuda")
    y = torch.tensor([float(target(r) or 0.0) for r in train], dtype=torch.float32, device=device)
    X = torch.tensor([feature_vector(r) for r in train], dtype=torch.float32, device=device)
    alpha = 1.0
    xtx = X.T @ X + alpha * torch.eye(X.shape[1], device=device)
    w = torch.linalg.solve(xtx, X.T @ y).detach().cpu().numpy().tolist()
    ridge = Baseline("ridge", lambda _: None, lambda r: (sum(a * b for a, b in zip(w, feature_vector(r))), 0.0))

    # A small gradient-boosted decision-stump ensemble implemented directly in
    # torch. Threshold search, residual fitting and updates all happen on the
    # selected CUDA device; this avoids silently falling back to CPU sklearn.
    def fit_gpu_gbt() -> Tuple[float, List[Tuple[int, float, float, float]]]:
        initial = float(y.mean().detach().cpu())
        fitted = torch.full_like(y, initial)
        stumps: List[Tuple[int, float, float, float]] = []
        quantiles = torch.linspace(0.05, 0.95, 12, device=device)
        learning_rate = 0.08
        for _ in range(24):
            residual = y - fitted
            best = None
            best_loss = None
            for feature_index in range(X.shape[1]):
                values = X[:, feature_index]
                thresholds = torch.quantile(values, quantiles).unique()
                for threshold_tensor in thresholds:
                    threshold = float(threshold_tensor.detach().cpu())
                    mask = values <= threshold_tensor
                    if int(mask.sum()) == 0 or int(mask.sum()) == len(values):
                        continue
                    left = residual[mask].mean()
                    right = residual[~mask].mean()
                    update = torch.where(mask, left, right)
                    loss = torch.mean((residual - learning_rate * update) ** 2)
                    if best_loss is None or float(loss.detach().cpu()) < best_loss:
                        best_loss = float(loss.detach().cpu())
                        best = (feature_index, threshold, float(left.detach().cpu()), float(right.detach().cpu()))
            if best is None:
                break
            feature_index, threshold, left, right = best
            stumps.append((feature_index, threshold, learning_rate * left, learning_rate * right))
            fitted = fitted + learning_rate * torch.where(X[:, feature_index] <= threshold, torch.tensor(left, device=device), torch.tensor(right, device=device))
        return initial, stumps

    gbt_initial, gbt_stumps = fit_gpu_gbt()
    def gbt_predict(r: Mapping[str, object]) -> Tuple[float, float]:
        values = feature_vector(r)
        score = gbt_initial
        for feature_index, threshold, left, right in gbt_stumps:
            score += left if values[feature_index] <= threshold else right
        return score, 0.0
    gbt = Baseline("gradient_boosted_trees", lambda _: None, gbt_predict)

    torch.manual_seed(17)
    max_len = 256

    def encode(value: str, channels: int = 4) -> torch.Tensor:
        table = {"A": 0, "C": 1, "G": 2, "U": 3}
        out = torch.zeros(channels, max_len, dtype=torch.float32, device=device)
        for i, base in enumerate(value[:max_len]):
            j = table.get(base)
            if j is not None:
                out[j, i] = 1.0
        return out

    class PairCNN(nn.Module):
        def __init__(self, in_channels: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, 24, 5, padding=2), nn.ReLU(),
                nn.Conv1d(24, 16, 5, padding=2), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(16, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    def fit_model(model: nn.Module, values: torch.Tensor, epochs: int = 4) -> nn.Module:
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
        model.train()
        batch = 128
        for _ in range(epochs):
            for start in range(0, len(values), batch):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.mean((model(values[start:start + batch, ...]) - y[start:start + batch]) ** 2)
                loss.backward()
                optimizer.step()
        return model.eval()

    pair_inputs = torch.stack([torch.cat([encode(seq(r, "source_sequence")), encode(seq(r, "candidate_sequence"))], dim=0) for r in train])
    delta_model = fit_model(PairCNN(8), pair_inputs)
    small_cnn = Baseline("small_CNN", lambda _: None, lambda r: (float(delta_model(torch.stack([torch.cat([encode(seq(r, "source_sequence")), encode(seq(r, "candidate_sequence"))], dim=0)]))[0].detach().cpu()), 0.0))

    absolute_train = [r for r in train if finite(r.get("measured_candidate")) is not None]
    if absolute_train:
        abs_y = torch.tensor([float(r["measured_candidate"]) for r in absolute_train], dtype=torch.float32, device=device)
        abs_inputs = torch.stack([encode(seq(r, "candidate_sequence")) for r in absolute_train])
        # Reuse the same training loop with a local target tensor.
        original_y = y
        y = abs_y
        absolute_model = fit_model(PairCNN(4), abs_inputs)
        y = original_y
        def absolute_predict(r: Mapping[str, object]) -> Tuple[float, float]:
            candidate = float(absolute_model(encode(seq(r, "candidate_sequence")).unsqueeze(0))[0].detach().cpu())
            source = float(absolute_model(encode(seq(r, "source_sequence")).unsqueeze(0))[0].detach().cpu())
            return candidate - source, 0.0
        absolute = Baseline("absolute_predictor", lambda _: None, absolute_predict)
    else:
        absolute = Baseline("absolute_predictor", lambda _: None, lambda _: (0.0, 0.0), status="not_run", reason="no measured absolute training records")
    return {"ridge": ridge, "gradient_boosted_trees": gbt, "small_CNN": small_cnn, "absolute_predictor": absolute}


def rankdata(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for k in order[i:j]:
            ranks[k] = rank
        i = j
    return ranks


def corr(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if len(a) < 2:
        return None
    ma, mb = mean_or(a), mean_or(b)
    da = sum((x - ma) ** 2 for x in a)
    db = sum((x - mb) ** 2 for x in b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(max(da * db, 1e-24))


def pairwise_metrics(pred: Sequence[float], truth: Sequence[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    concordant = discordant = tied_pred = comparable = 0.0
    for i in range(len(truth)):
        for j in range(i + 1, len(truth)):
            dt = truth[i] - truth[j]
            if dt == 0:
                continue
            comparable += 1
            dp = pred[i] - pred[j]
            if dp == 0:
                tied_pred += 1
            elif dp * dt > 0:
                concordant += 1
            else:
                discordant += 1
    if comparable == 0:
        return None, None, None
    auc = (concordant + 0.5 * tied_pred) / comparable
    kendall = (concordant - discordant) / comparable
    return auc, kendall, comparable


def ndcg(pred: Sequence[float], truth: Sequence[float], fraction: float = 0.10) -> Optional[float]:
    if not truth:
        return None
    k = max(1, min(len(truth), int(math.ceil(len(truth) * fraction))))
    minimum = min(truth)
    gains = [max(0.0, y - minimum) + 1e-9 for y in truth]
    pred_order = sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)[:k]
    ideal_order = sorted(range(len(truth)), key=lambda i: truth[i], reverse=True)[:k]
    dcg = sum(gains[i] / math.log2(j + 2) for j, i in enumerate(pred_order))
    ideal = sum(gains[i] / math.log2(j + 2) for j, i in enumerate(ideal_order))
    return dcg / ideal if ideal > 0 else 1.0


def ece(pred: Sequence[float], truth: Sequence[float], bins: int = 10) -> Optional[float]:
    if not truth:
        return None
    scale = max(1e-6, (max(pred) - min(pred)) / 2.0)
    center = mean_or(pred)
    probs = [1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, (p - center) / scale)))) for p in pred]
    actual = [float(y > 0) for y in truth]
    error = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        ix = [i for i, p in enumerate(probs) if lo <= p < hi or (b == bins - 1 and p == hi)]
        if ix:
            error += len(ix) / len(probs) * abs(mean_or([probs[i] for i in ix]) - mean_or([actual[i] for i in ix]))
    return error


def binary_auc(score: Sequence[float], label: Sequence[bool]) -> Optional[float]:
    positives = [i for i, value in enumerate(label) if value]
    negatives = [i for i, value in enumerate(label) if not value]
    if not positives or not negatives:
        return None
    wins = 0.0
    for i in positives:
        for j in negatives:
            if score[i] > score[j]:
                wins += 1.0
            elif score[i] == score[j]:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def metrics(
    pred: Sequence[float], uncertainty: Sequence[float], truth: Sequence[float],
    *, error_cut: Optional[float] = None,
) -> Dict[str, object]:
    if not truth:
        return {"n": 0}
    ranks = rankdata(pred)
    true_ranks = rankdata(truth)
    pair_auc, kendall, comparable = pairwise_metrics(pred, truth)
    beneficial = [y > 0 for y in truth]
    selected = [p > 0 for p in pred]
    predicted_beneficial = sum(selected)
    false_positive_beneficial_rate = (
        sum(float(s and not b) for b, s in zip(beneficial, selected)) / predicted_beneficial
        if predicted_beneficial else None
    )
    top_k = max(1, int(math.ceil(len(truth) * 0.10)))
    top_ix = sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)[:top_k]
    base_rate = mean_or([float(v) for v in beneficial])
    top_rate = mean_or([float(beneficial[i]) for i in top_ix])
    order = sorted(range(len(pred)), key=lambda i: uncertainty[i] if i < len(uncertainty) else 0.0)
    selective = {}
    for coverage in (0.25, 0.50, 0.75, 1.0):
        n = max(1, int(math.ceil(len(order) * coverage)))
        ix = order[:n]
        selective[f"coverage_{coverage:.2f}"] = math.sqrt(mean_or([(pred[i] - truth[i]) ** 2 for i in ix]))
    resolved_error_cut = (
        float(error_cut) if error_cut is not None else
        float(np.median(np.abs(np.asarray(pred) - np.asarray(truth))))
    )
    error_labels = [abs(p - y) > resolved_error_cut for p, y in zip(pred, truth)]
    abstention_target = (
        "validation_absolute_error_median" if error_cut is not None
        else "role_absolute_error_median"
    )
    return {
        "n": len(truth),
        "spearman": corr(ranks, true_ranks),
        "kendall": kendall,
        "pairwise_auc": pair_auc,
        "pairwise_comparable": comparable,
        "sign_accuracy": mean_or([float((p > 0) == (y > 0)) for p, y in zip(pred, truth)]),
        "beneficial_precision": (sum(float(b) for b, s in zip(beneficial, selected) if s) / predicted_beneficial) if predicted_beneficial else None,
        "false_positive_beneficial_rate": false_positive_beneficial_rate,
        "top_k_enrichment": (top_rate / base_rate) if base_rate > 0 else None,
        "top_k_beneficial_rate": top_rate,
        "ndcg_at_10pct": ndcg(pred, truth),
        "rmse": math.sqrt(mean_or([(p - y) ** 2 for p, y in zip(pred, truth)])),
        "calibration_ece": ece(pred, truth),
        "selective_risk_rmse": selective,
        "abstention_auroc": binary_auc(uncertainty, error_labels),
        "abstention_target": f"absolute_error_above_{abstention_target}",
        "abstention_error_cut": resolved_error_cut,
    }


def legal_action_space_count(source_sequence: str, budget: int) -> int:
    """Count legal same-length SNV candidates under the benchmark edit contract."""
    normalized = str(source_sequence or "").upper().replace("T", "U")
    length = sum(1 for base in normalized if base in "ACGU")
    return int(sum(math.comb(length, k) * (3 ** k) for k in range(1, min(budget, length) + 1)))


def _observed_edit_signature(rec: Mapping[str, object]):
    value = rec.get("edit_list")
    if not isinstance(value, list):
        return None
    signature = []
    for item in value:
        if not isinstance(item, Mapping):
            return None
        try:
            pos = int(item.get("pos", 0))
        except (TypeError, ValueError):
            return None
        signature.append((
            pos,
            str(item.get("ref") or ""),
            str(item.get("alt") or ""),
            str(item.get("op") or "replace"),
        ))
    return frozenset(signature)


def _observed_beam_indices(
    records: List[Dict], indices: List[int], predictions: Sequence[float], *, budget: int, width: int = 4,
):
    """Beam-search only the measured candidate-state graph for one source."""
    levels: Dict[int, List[int]] = defaultdict(list)
    for i in indices:
        levels[int(records[i].get("edit_count") or 0)].append(i)
    frontier: List[int] = []
    beam_indices: List[int] = []
    evaluations = 0
    for depth in range(1, budget + 1):
        nodes = levels.get(depth, [])
        if not nodes:
            break
        if depth == 1:
            reachable = nodes
        else:
            parent_signatures = [_observed_edit_signature(records[i]) for i in frontier]
            usable_parents = [sig for sig in parent_signatures if sig is not None]
            reachable = []
            for i in nodes:
                child_signature = _observed_edit_signature(records[i])
                if not usable_parents or child_signature is None:
                    reachable.append(i)
                    continue
                if any(parent < child_signature and len(child_signature - parent) == 1 for parent in usable_parents):
                    reachable.append(i)
        evaluations += len(reachable)
        frontier = sorted(reachable, key=lambda i: float(predictions[i]), reverse=True)[:width]
        beam_indices.extend(frontier)
    return beam_indices, frontier, evaluations


def budget_metrics(records: List[Dict], predictions: Sequence[float], *, budget: int) -> Dict[str, object]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        edit_count = int(rec.get("edit_count") or 0)
        if 0 < edit_count <= budget and target(rec) is not None:
            grouped[str(rec.get("source_id"))].append(i)
    if budget >= 3:
        grouped = {
            source_id: ix for source_id, ix in grouped.items()
            if any(int(records[i].get("edit_count") or 0) >= 2 for i in ix)
        }
        if not grouped:
            return {
                "n_sources": 0,
                "regret_exact_mean": None,
                "regret_vs_observed_exact_mean": None,
                "top_action_accuracy": None,
                "beneficial_rate": None,
                "exact_optimum_reach": None,
                "observed_exact_optimum_reach": None,
                "oracle_calls_mean": None,
                "final_delta_mean": None,
                "reward_per_edit": None,
                "failure_rate": 1.0,
                "rows": [],
                "status": "insufficient_multi_edit_measurements",
                "reference": "requires measured edit_count >= 2 records per source",
                "reference_scope": "not_available",
                "exact_optimum_scope": "not_available",
                "dp_beam_reference_status": "not_available_in_current_source_matched_registry",
                "regret_vs_dp_beam_reference": None,
                "regret_vs_observed_reference": None,
                "legal_action_space": {"status": "not_available", "budget": budget},
                "full_legal_action_space_status": "unknown_unmeasured_actions",
                "reference_is_not_full_legal_dp_or_beam": True,
            }
    rows = []
    beam_rows = []
    for source_id, ix in grouped.items():
        truth = [float(target(records[i])) for i in ix if target(records[i]) is not None]
        if not truth:
            continue
        source_sequence = str(records[ix[0]].get("source_sequence") or "")
        legal_count = legal_action_space_count(source_sequence, budget)
        observed_candidates = len({str(records[i].get("candidate_sequence_sha256") or records[i].get("candidate_id")) for i in ix})
        optimum = max(truth)
        beam_indices, final_frontier, beam_evaluations = _observed_beam_indices(
            records, ix, predictions, budget=budget, width=4,
        )
        if not beam_indices:
            continue
        selectable = final_frontier or beam_indices
        selected = max(selectable, key=lambda i: float(predictions[i]))
        selected_truth = float(target(records[selected]))
        beam_best = max(beam_indices, key=lambda i: float(target(records[i])))
        beam_best_truth = float(target(records[beam_best]))
        selected_regret = optimum - selected_truth
        beam_regret = optimum - beam_best_truth
        beam_rows.append({
            "source_id": source_id,
            "beam_width": min(4, len(beam_indices)),
            "beam_selected_delta": beam_best_truth,
            "beam_regret": beam_regret,
            "beam_exact_best": beam_best_truth >= optimum - 1e-12,
            "beam_candidate_evaluations": beam_evaluations,
        })
        rows.append({
            "source_id": source_id,
            "selected_index": selected,
            "selected_delta": selected_truth,
            "exact_best_delta": optimum,
            "regret": selected_regret,
            "top_action": selected_truth >= optimum - 1e-12,
            "beneficial": selected_truth > 0,
            "oracle_calls": beam_evaluations,
            "edit_count": int(records[selected].get("edit_count") or 0),
            "legal_action_count": legal_count,
            "observed_action_count": observed_candidates,
            "observed_action_coverage": observed_candidates / max(1, legal_count),
            "beam_reference_delta": beam_best_truth,
            "beam_reference_regret": beam_regret,
        })
    observed_regret = mean_or([r["regret"] for r in rows]) if rows else None
    beam_regret = mean_or([r["beam_regret"] for r in beam_rows]) if beam_rows else None
    return {
        "n_sources": len(rows),
        "regret_exact_mean": observed_regret,
        "regret_vs_observed_exact_mean": observed_regret,
        "top_action_accuracy": mean_or([float(r["top_action"]) for r in rows]) if rows else None,
        "beneficial_rate": mean_or([float(r["beneficial"]) for r in rows]) if rows else None,
        "exact_optimum_reach": mean_or([float(r["top_action"]) for r in rows]) if rows else None,
        "observed_exact_optimum_reach": mean_or([float(r["top_action"]) for r in rows]) if rows else None,
        "oracle_calls_mean": mean_or([r["oracle_calls"] for r in rows]) if rows else None,
        "final_delta_mean": mean_or([r["selected_delta"] for r in rows]) if rows else None,
        "reward_per_edit": mean_or([r["selected_delta"] / max(1, r["edit_count"]) for r in rows]) if rows else None,
        "beam4_regret_mean": beam_regret,
        "beam4_observed_regret_mean": beam_regret,
        "beam4_exact_optimum_reach": mean_or([float(r["beam_exact_best"]) for r in beam_rows]) if beam_rows else None,
        "beam4_observed_exact_optimum_reach": mean_or([float(r["beam_exact_best"]) for r in beam_rows]) if beam_rows else None,
        "failure_rate": 0.0 if rows else 1.0,
        "rows": rows[:25],
        "beam_rows": beam_rows[:25],
        "status": "observed_pool_dp_beam_reference" if rows else "insufficient_measured_actions",
        "reference": "exact optimum and beam_width=4 over measured candidate-state graph",
        "reference_scope": "observed_measured_candidate_pool_only",
        "exact_optimum_scope": "observed_measured_candidate_pool_only",
        "dp_beam_reference_status": "observed_candidate_state_graph_not_full_legal_action_space",
        "regret_vs_dp_beam_reference": observed_regret,
        "regret_vs_observed_reference": observed_regret,
        "legal_action_space": {
            "status": "enumerated_count_only_unmeasured_actions",
            "edit_contract": "same-length single-nucleotide substitutions; distinct positions; at most budget edits",
            "budget": budget,
            "mean_legal_action_count": mean_or([r["legal_action_count"] for r in rows]) if rows else None,
            "mean_observed_action_count": mean_or([r["observed_action_count"] for r in rows]) if rows else None,
            "mean_observed_action_coverage": mean_or([r["observed_action_coverage"] for r in rows]) if rows else None,
        },
        "full_legal_action_space_status": "unknown_unmeasured_actions",
        "reference_is_not_full_legal_dp_or_beam": True,
    }


def bootstrap_rmse_delta(pred_a: Sequence[float], pred_b: Sequence[float], truth: Sequence[float], seed: int = 17) -> Dict[str, float]:
    if np is None or len(truth) < 2:
        return {"mean_delta_a_minus_b": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = np.random.default_rng(seed)
    a = np.asarray(pred_a, dtype=float)
    b = np.asarray(pred_b, dtype=float)
    y = np.asarray(truth, dtype=float)
    deltas = []
    for _ in range(1000):
        ix = rng.integers(0, len(y), size=len(y))
        deltas.append(float(np.sqrt(np.mean((a[ix] - y[ix]) ** 2)) - np.sqrt(np.mean((b[ix] - y[ix]) ** 2))))
    return {
        "mean_delta_a_minus_b": float(np.mean(deltas)),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
    }


def evaluate(root: Path, roles: Sequence[str], *, allow_final_labels: bool) -> Dict:
    if np is None:
        raise RuntimeError("numpy is required for unified metrics")
    train = load_records(root, "train", allow_final_labels=allow_final_labels)
    train = [r for r in train if target(r) is not None and bool(r.get("local_delta_eligible"))]
    statistical = make_statistical_baselines(train)
    fitted = torch_baselines(train)
    baselines = {**statistical, **fitted}
    for baseline in baselines.values():
        baseline.fit(train)
    registry_meta = {}
    registry_path = root / "registry.json"
    if registry_path.exists():
        registry_meta = json.loads(registry_path.read_text())
    output: Dict[str, object] = {
        "schema_version": "nmi_benchmark_v2_metrics_v1",
        "benchmark_records_sha256": registry_meta.get("records_sha256"),
        "train_local_delta_records": len(train),
        "evaluated_roles": list(roles),
        "allow_final_labels": allow_final_labels,
        "label_policy": "final roles opened only by explicit flag",
        "task_contract": "only measured task_kind=local_delta enters Task 1-3 metrics; target is scored_delta when present",
        "scoring_endpoint": "log2_plus_one_fold_change_candidate_over_source; raw measured_source/measured_candidate/measured_delta remain provenance",
        "roles": {},
        "baselines": {},
        "validation_error_cut_reference": "computed on val predictions and reused for final-role abstention labels",
    }
    all_role_results: Dict[str, Dict[str, object]] = {}
    validation_error_cut: Dict[str, float] = {}
    ordered_roles = (["val"] if "val" in roles else []) + [role for role in roles if role != "val"]
    for role in ordered_roles:
        if role not in ROLES:
            raise ValueError(f"unknown role {role}")
        if role in {"test_context", "test_assay"}:
            axis_records = load_records(root, role, allow_final_labels=allow_final_labels, task_kind=None)
            task_counts = Counter(str(r.get("task_kind")) for r in axis_records)
            output["roles"][role] = {
                "status": "axis_shift_only",
                "records": len(axis_records),
                "task_kinds": dict(task_counts),
                "source_matched_axis_records": int(task_counts.get("context_delta", 0) + task_counts.get("assay_delta", 0)),
                "absolute_property_records": int(sum(v for k, v in task_counts.items() if k.startswith("absolute_property_"))),
                "claim_ready_for_nucleotide_local_delta": False,
            }
            continue
        records = load_records(root, role, allow_final_labels=allow_final_labels)
        truth = [float(target(r)) for r in records if target(r) is not None and bool(r.get("local_delta_eligible"))]
        eval_records = [r for r in records if target(r) is not None and bool(r.get("local_delta_eligible"))]
        role_result: Dict[str, object] = {"records": len(eval_records), "baselines": {}}
        for name, baseline in baselines.items():
            if baseline.status != "run":
                role_result["baselines"][name] = {"status": baseline.status, "reason": baseline.reason}
                continue
            pred, uncertainty = baseline.predict(eval_records)
            task1 = metrics(pred, uncertainty, truth, error_cut=validation_error_cut.get(name))
            if role == "val" and truth:
                validation_error_cut[name] = float(np.median(np.abs(np.asarray(pred) - np.asarray(truth))))
            role_result["baselines"][name] = {
                "status": "run",
                "task1_local_delta": task1,
                "task2_budget1": budget_metrics(eval_records, pred, budget=1),
                "task3_budget3": budget_metrics(eval_records, pred, budget=3),
            }
            if role == "test_ood":
                role_result["baselines"][name]["task4_ood"] = {
                    "coverage": [0.25, 0.50, 0.75, 1.00],
                    "selective_risk": task1.get("selective_risk_rmse"),
                    "false_positive_beneficial_rate": task1.get("false_positive_beneficial_rate"),
                    "calibration_ece": task1.get("calibration_ece"),
                    "abstention_auroc": task1.get("abstention_auroc"),
                    "abstention_target": task1.get("abstention_target"),
                    "abstention_auroc_status": "error_detection_auc; not a biological uncertainty ground truth",
                }
            all_role_results.setdefault(role, {})[name] = {"pred": pred, "truth": truth}
        output["roles"][role] = role_result

    output["baselines"] = {
        name: {"status": baseline.status, "reason": baseline.reason}
        for name, baseline in baselines.items()
    }
    # Gate only on validation, never on a final role. The best trivial is the
    # lowest validation RMSE among the explicitly non-deep baselines.
    gate = {
        "deep_model_advancement_allowed": False,
        "status": "blocked_until_significance",
        "comparison_role": "val" if "val" in all_role_results else None,
        "best_trivial": None,
        "deep_models": {},
    }
    val_results = all_role_results.get("val", {})
    val_report = output["roles"].get("val", {})
    trivial_names = {"mean", "source_mean", "gc_delta", "position_only", "ref_alt_transition", "local_kmer", "kozak_only", "uaug_only", "RNAfold_delta", "ridge", "gradient_boosted_trees"}
    available_trivial = [(name, val_report.get("baselines", {}).get(name, {}).get("task1_local_delta", {}).get("rmse")) for name in trivial_names]
    available_trivial = [(name, value) for name, value in available_trivial if value is not None]
    if available_trivial:
        best_name, best_rmse = min(available_trivial, key=lambda item: item[1])
        gate["best_trivial"] = {"name": best_name, "rmse": best_rmse}
        for deep_name in ("small_CNN", "absolute_predictor"):
            deep_report = val_report.get("baselines", {}).get(deep_name, {})
            if deep_report.get("status") != "run" or deep_name not in val_results:
                gate["deep_models"][deep_name] = {"status": deep_report.get("status", "not_run"), "reason": deep_report.get("reason", "no validation result")}
                continue
            significance = bootstrap_rmse_delta(val_results[deep_name]["pred"], val_results[best_name]["pred"], val_results[deep_name]["truth"])
            deep_rmse = deep_report["task1_local_delta"].get("rmse")
            eligible = bool(deep_rmse < best_rmse and significance["ci95_high"] < 0)
            gate["deep_models"][deep_name] = {
                "status": "run",
                "rmse": deep_rmse,
                "beats_best_trivial": bool(deep_rmse < best_rmse),
                "paired_bootstrap_rmse": significance,
                "significantly_better": eligible,
            }
            gate["deep_model_advancement_allowed"] = gate["deep_model_advancement_allowed"] or eligible
    if not gate["deep_model_advancement_allowed"]:
        gate["reason"] = "No deep baseline has a preregistered significant validation improvement over the best trivial baseline"
    output["baseline_gate"] = gate
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/nmi_benchmark_v2")
    ap.add_argument("--roles", nargs="+", default=["val"])
    ap.add_argument("--out", default="artifacts/phase1/benchmark_v2_metrics.json")
    ap.add_argument("--allow-final-labels", action="store_true")
    args = ap.parse_args()
    report = evaluate(Path(args.root), args.roles, allow_final_labels=args.allow_final_labels)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
