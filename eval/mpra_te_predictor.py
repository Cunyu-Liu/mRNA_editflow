"""Lightweight MPRA TE/MRL predictor protocol.

The goal of this module is to make the MPRA/TE path executable and auditable
without adding a heavy dependency or overclaiming. With a real CSV/TSV input it
fits a small ridge regressor over deterministic 5'UTR features and reports
train/val/test metrics. Without an input file it can run a synthetic smoke test,
which must remain protocol evidence only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Mapping, Optional, Sequence

import numpy as np

from mrna_editflow.data.prepare_mpra import prepare_mpra_dataset, read_mpra_table
from mrna_editflow.eval.oracle import extract_utr_features


CLAIM_POLICY = (
    "MPRA TE predictor reports are downstream-data protocol evidence. Synthetic "
    "fallback runs are smoke tests only. Real TE predictor claims require a real "
    "MPRA/TE input table, split-held-out metrics, preserved split provenance, and "
    "separate stability/half-life validation before being used as SOTA evidence."
)

FEATURE_NAMES: tuple[str, ...] = (
    "length",
    "gc",
    "gc_opt",
    "length_opt",
    "kozak",
    "uaug_count",
    "stop_like_count",
    "start_accessibility",
    "top_motif",
    "poly_gc_run",
    "invalid_fraction",
)
VALID_SPLITS = {"train", "val", "test"}


def _is_finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return None
    return float(np.mean(np.asarray(vals, dtype=float)))


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks with deterministic tie handling."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def _correlation(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if len(a) < 2 or len(b) < 2:
        return None
    ax = a.astype(float)
    bx = b.astype(float)
    if not (np.isfinite(ax).all() and np.isfinite(bx).all()):
        return None
    ax = ax - float(np.mean(ax))
    bx = bx - float(np.mean(bx))
    denom = float(np.linalg.norm(ax) * np.linalg.norm(bx))
    if denom == 0.0:
        return None
    return float(np.dot(ax, bx) / denom)


def _spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if len(a) < 2 or len(b) < 2:
        return None
    return _correlation(_rankdata(a.astype(float)), _rankdata(b.astype(float)))


def _split_counts(samples: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0, "other": 0}
    for sample in samples:
        split = str(sample.get("split", "")).lower()
        counts[split if split in VALID_SPLITS else "other"] += 1
    return counts


def _feature_matrix(samples: Sequence[Mapping[str, object]]) -> np.ndarray:
    rows = []
    for sample in samples:
        features = extract_utr_features(str(sample.get("sequence", "")))
        rows.append([float(features[name]) for name in FEATURE_NAMES])
    return np.asarray(rows, dtype=float)


def _targets(samples: Sequence[Mapping[str, object]], target: str) -> np.ndarray:
    return np.asarray([float(sample[target]) for sample in samples], dtype=float)


def _standardize_train(
    x_train: np.ndarray,
    x_all: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std = np.where(std <= 1e-12, 1.0, std)
    return (x_all - mean) / std, mean, std


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x_design = np.concatenate([np.ones((x.shape[0], 1), dtype=float), x], axis=1)
    penalty = np.eye(x_design.shape[1], dtype=float) * math.sqrt(max(0.0, float(alpha)))
    penalty[0, 0] = 0.0
    augmented_x = np.concatenate([x_design, penalty], axis=0)
    augmented_y = np.concatenate([y, np.zeros(x_design.shape[1], dtype=float)], axis=0)
    return np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]


def _predict(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    x_design = np.concatenate([np.ones((x.shape[0], 1), dtype=float), x], axis=1)
    return x_design @ coef


def _metric_row(
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, object]:
    if len(y_true) == 0:
        return {
            "split": split,
            "n": 0,
            "mae": None,
            "rmse": None,
            "pearson": None,
            "spearman": None,
            "target_mean": None,
            "prediction_mean": None,
        }
    err = y_pred - y_true
    return {
        "split": split,
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(math.sqrt(float(np.mean(err * err)))),
        "pearson": _correlation(y_true, y_pred),
        "spearman": _spearman(y_true, y_pred),
        "target_mean": float(np.mean(y_true)),
        "prediction_mean": float(np.mean(y_pred)),
    }


def _load_samples(
    input_path: Optional[str],
    *,
    n_synthetic: int,
    seed: int,
) -> tuple[list[dict], dict[str, object]]:
    if input_path:
        raw = read_mpra_table(input_path)
        official_split_present = bool(raw) and all(
            str(row.get("split", "")).lower() in VALID_SPLITS for row in raw
        )
        samples = prepare_mpra_dataset(rows=raw)
        source_kind = "external_input"
    else:
        samples = prepare_mpra_dataset(n_synthetic=n_synthetic, seed=seed)
        official_split_present = False
        source_kind = "synthetic_smoke"
    return samples, {
        "input_path": input_path,
        "source_kind": source_kind,
        "official_split_present": official_split_present,
    }


def run_mpra_te_predictor(
    input_path: Optional[str] = None,
    *,
    n_synthetic: int = 64,
    seed: int = 0,
    ridge_alpha: float = 1.0,
    target: str = "mrl_z",
    min_test_n: int = 2,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Fit and audit a lightweight MPRA TE/MRL predictor."""
    samples, input_meta = _load_samples(input_path, n_synthetic=n_synthetic, seed=seed)
    if not samples:
        raise ValueError("MPRA predictor requires at least one sample")
    if target not in samples[0]:
        raise ValueError(f"target {target!r} not present in MPRA samples")

    split_counts = _split_counts(samples)
    train_idx = [
        i for i, sample in enumerate(samples)
        if str(sample.get("split", "")).lower() == "train"
    ]
    if not train_idx:
        raise ValueError("MPRA predictor requires at least one train sample")
    x_raw = _feature_matrix(samples)
    y = _targets(samples, target)
    x_all, feat_mean, feat_std = _standardize_train(x_raw[train_idx], x_raw)
    coef = _fit_ridge(x_all[train_idx], y[train_idx], ridge_alpha)
    pred = _predict(x_all, coef)

    metrics: dict[str, dict[str, object]] = {}
    for split in ("train", "val", "test"):
        idx = [
            i for i, sample in enumerate(samples)
            if str(sample.get("split", "")).lower() == split
        ]
        metrics[split] = _metric_row(split, y[idx], pred[idx])

    predictions = []
    for i, sample in enumerate(samples):
        predictions.append(
            {
                "sample_id": sample.get("sample_id"),
                "split": sample.get("split"),
                "sequence": sample.get("sequence"),
                "mrl": sample.get("mrl"),
                "target": float(y[i]),
                "prediction": float(pred[i]),
                "source": sample.get("source"),
            }
        )

    test_metrics = metrics.get("test", {})
    external_input = input_meta["source_kind"] == "external_input"
    test_ready = (
        int(test_metrics.get("n") or 0) >= int(min_test_n)
        and _is_finite(test_metrics.get("mae"))
        and _is_finite(test_metrics.get("rmse"))
    )
    ready_for_te_predictor_audit = bool(external_input and test_ready)
    report = {
        "artifact_kind": "mpra_te_predictor_protocol",
        "claim_policy": CLAIM_POLICY,
        "input": {
            **input_meta,
            "n_samples": len(samples),
            "split_counts": split_counts,
            "sequence_length": {
                "mean": _safe_mean([len(str(s.get("sequence", ""))) for s in samples]),
                "min": min(len(str(s.get("sequence", ""))) for s in samples),
                "max": max(len(str(s.get("sequence", ""))) for s in samples),
            },
        },
        "model": {
            "model_family": "feature_ridge_regression",
            "target": target,
            "ridge_alpha": float(ridge_alpha),
            "feature_names": list(FEATURE_NAMES),
            "intercept": float(coef[0]),
            "coefficients": {
                name: float(value) for name, value in zip(FEATURE_NAMES, coef[1:])
            },
            "feature_mean": {
                name: float(value) for name, value in zip(FEATURE_NAMES, feat_mean)
            },
            "feature_std": {
                name: float(value) for name, value in zip(FEATURE_NAMES, feat_std)
            },
        },
        "metrics": metrics,
        "summary": {
            "ready_for_mpra_te_predictor_audit": ready_for_te_predictor_audit,
            "ready_for_real_te_or_stability_claim": False,
            "ready_for_wet_lab_design_claim": False,
            "synthetic_smoke_only": input_meta["source_kind"] == "synthetic_smoke",
            "external_input_provided": external_input,
            "official_split_present": input_meta["official_split_present"],
            "test_split_ready": test_ready,
            "stability_or_half_life_labels_present": False,
            "n_train": split_counts["train"],
            "n_val": split_counts["val"],
            "n_test": split_counts["test"],
            "test_mae": test_metrics.get("mae"),
            "test_rmse": test_metrics.get("rmse"),
            "test_pearson": test_metrics.get("pearson"),
            "test_spearman": test_metrics.get("spearman"),
        },
        "limitations": [
            "Feature ridge regression is a protocol baseline, not an external SOTA model.",
            "Synthetic fallback evidence must not be used as real MPRA/TE performance.",
            "Stability or half-life prediction is not covered by MRL-only inputs.",
        ],
    }
    return report, predictions


def write_report_json(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    return path


def write_predictions_jsonl(predictions: Sequence[Mapping[str, object]], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in predictions:
            fh.write(json.dumps(dict(row), sort_keys=True) + "\n")
    return path


def _fmt(value: object, digits: int = 4) -> str:
    if not _is_finite(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def write_report_markdown(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    summary = report.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    metrics = report.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
    input_meta = report.get("input", {})
    if not isinstance(input_meta, Mapping):
        input_meta = {}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# MPRA TE Predictor Protocol\n\n")
        fh.write(f"- Claim policy: {report.get('claim_policy', CLAIM_POLICY)}\n")
        fh.write(
            f"- Source kind: `{input_meta.get('source_kind')}`; samples: "
            f"`{input_meta.get('n_samples')}`; official split present: "
            f"`{summary.get('official_split_present')}`\n"
        )
        fh.write(
            f"- Ready for MPRA TE predictor audit: "
            f"`{summary.get('ready_for_mpra_te_predictor_audit')}`; "
            f"ready for real TE/stability claim: "
            f"`{summary.get('ready_for_real_te_or_stability_claim')}`; "
            f"synthetic smoke only: `{summary.get('synthetic_smoke_only')}`\n\n"
        )
        fh.write("| Split | n | MAE | RMSE | Pearson | Spearman |\n")
        fh.write("|---|---:|---:|---:|---:|---:|\n")
        for split in ("train", "val", "test"):
            row = metrics.get(split, {})
            if not isinstance(row, Mapping):
                row = {}
            fh.write(
                f"| {split} | {row.get('n', 0)} | {_fmt(row.get('mae'))} | "
                f"{_fmt(row.get('rmse'))} | {_fmt(row.get('pearson'))} | "
                f"{_fmt(row.get('spearman'))} |\n"
            )
        fh.write("\n")
        fh.write("## Limitations\n\n")
        limitations = report.get("limitations", [])
        if not isinstance(limitations, Sequence) or isinstance(limitations, (str, bytes)):
            limitations = []
        for item in limitations:
            fh.write(f"- {item}\n")
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="MPRA CSV/TSV with sequence,mrl[,split]")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--predictions-jsonl", default=None)
    parser.add_argument("--n-synthetic", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--target", default="mrl_z", choices=("mrl", "mrl_z"))
    parser.add_argument("--min-test-n", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    report, predictions = run_mpra_te_predictor(
        input_path=args.input,
        n_synthetic=args.n_synthetic,
        seed=args.seed,
        ridge_alpha=args.ridge_alpha,
        target=args.target,
        min_test_n=args.min_test_n,
    )
    write_report_json(report, args.out_json)
    write_report_markdown(report, args.out_md)
    if args.predictions_jsonl:
        write_predictions_jsonl(predictions, args.predictions_jsonl)
    print(json.dumps({"json_path": args.out_json, "markdown_path": args.out_md}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLAIM_POLICY",
    "FEATURE_NAMES",
    "run_mpra_te_predictor",
    "write_predictions_jsonl",
    "write_report_json",
    "write_report_markdown",
    "main",
]
