"""Lightweight mRNA stability / half-life predictor protocol.

This module mirrors the MPRA TE predictor protocol for stability-style labels.
It accepts a CSV/TSV table with ``sequence`` plus one numeric target column such
as ``half_life``, ``stability``, ``stability_score`` or ``degradation_rate``.
Without a real input table it runs a deterministic synthetic smoke test only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from mrna_editflow.data.download_mrna import synthesize_corpus


CLAIM_POLICY = (
    "Stability predictor reports are downstream-data protocol evidence. "
    "Synthetic fallback runs are smoke tests only. Real stability or half-life "
    "claims require an external labelled table, preserved train/val/test split "
    "provenance, held-out metrics, and leakage-aware dataset documentation."
)

TARGET_CANDIDATES: tuple[str, ...] = (
    "half_life",
    "stability",
    "stability_score",
    "degradation_rate",
    "decay_rate",
    "target",
)
VALID_SPLITS = {"train", "val", "test"}
RNA_ALPHABET = "ACGU"
RNA_SET = set(RNA_ALPHABET)
FEATURE_NAMES: tuple[str, ...] = (
    "length",
    "gc",
    "au",
    "u_fraction",
    "a_fraction",
    "poly_a_signal_count",
    "are_count",
    "u_rich_motif_count",
    "gc_run_flag",
    "au_run_flag",
    "start_aug_count",
    "mfe_proxy",
)


def normalise_rna_sequence(sequence: str) -> str:
    seq = "".join(str(sequence or "").split()).upper().replace("T", "U")
    if not seq:
        raise ValueError("sequence must be non-empty")
    bad = set(seq) - RNA_SET
    if bad:
        raise ValueError(f"sequence contains non-ACGU characters: {sorted(bad)}")
    return seq


def _guess_delimiter(path: Path) -> str:
    if path.suffix.lower() in {".tsv", ".tab"}:
        return "\t"
    if path.suffix.lower() == ".csv":
        return ","
    with open(path, "r", encoding="utf-8", newline="") as fh:
        sample = fh.read(2048)
    return "\t" if sample.count("\t") > sample.count(",") else ","


def _count_overlapping(seq: str, motif: str) -> int:
    if not motif:
        return 0
    count = 0
    start = 0
    while True:
        idx = seq.find(motif, start)
        if idx < 0:
            return count
        count += 1
        start = idx + 1


def extract_stability_features(sequence: str) -> dict[str, float]:
    seq = normalise_rna_sequence(sequence)
    length = len(seq)
    gc = (seq.count("G") + seq.count("C")) / length
    au = 1.0 - gc
    u_fraction = seq.count("U") / length
    a_fraction = seq.count("A") / length
    poly_a = _count_overlapping(seq, "AAUAAA") + _count_overlapping(seq, "AUUAAA")
    are = _count_overlapping(seq, "AUUUA")
    u_rich = _count_overlapping(seq, "UUUU")
    gc_run = 1.0 if "GGGG" in seq or "CCCC" in seq else 0.0
    au_run = 1.0 if "AAAA" in seq or "UUUU" in seq else 0.0
    start_aug = _count_overlapping(seq[:120], "AUG")
    mfe_proxy = -1.0 * gc * math.log1p(length)
    return {
        "length": float(length),
        "gc": float(gc),
        "au": float(au),
        "u_fraction": float(u_fraction),
        "a_fraction": float(a_fraction),
        "poly_a_signal_count": float(poly_a),
        "are_count": float(are),
        "u_rich_motif_count": float(u_rich),
        "gc_run_flag": float(gc_run),
        "au_run_flag": float(au_run),
        "start_aug_count": float(start_aug),
        "mfe_proxy": float(mfe_proxy),
    }


def _synthetic_half_life(sequence: str) -> float:
    features = extract_stability_features(sequence)
    length_term = math.exp(-abs(features["length"] - 950.0) / 1800.0)
    return float(
        5.0
        + 2.0 * features["poly_a_signal_count"]
        + 1.2 * length_term
        + 0.9 * features["au"]
        - 0.7 * features["are_count"]
        - 0.4 * features["u_rich_motif_count"]
        - 0.3 * features["gc_run_flag"]
    )


def _fallback_split(i: int, n: int) -> str:
    frac = (i + 0.5) / max(1, n)
    if frac < 0.8:
        return "train"
    if frac < 0.9:
        return "val"
    return "test"


def read_stability_table(
    path: str,
    *,
    sequence_col: str = "sequence",
    target_col: Optional[str] = None,
    split_col: str = "split",
) -> tuple[list[dict[str, object]], str]:
    fp = Path(path)
    delim = _guess_delimiter(fp)
    rows: list[dict[str, object]] = []
    with open(fp, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delim)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        if sequence_col not in reader.fieldnames:
            raise ValueError(f"{path} missing required column {sequence_col!r}")
        selected_target = target_col
        if selected_target is None:
            selected_target = next(
                (name for name in TARGET_CANDIDATES if name in reader.fieldnames),
                None,
            )
        if selected_target is None or selected_target not in reader.fieldnames:
            raise ValueError(
                f"{path} needs one target column from {list(TARGET_CANDIDATES)}"
            )
        for i, row in enumerate(reader):
            seq = normalise_rna_sequence(row[sequence_col])
            try:
                target = float(row[selected_target])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"row {i} has non-numeric {selected_target}: {row[selected_target]!r}"
                ) from exc
            split = str(row.get(split_col, "") or "").strip().lower()
            rows.append(
                {
                    "sample_id": row.get("sample_id") or row.get("id") or f"stability_{i:05d}",
                    "sequence": seq,
                    "target": target,
                    "target_name": selected_target,
                    "split": split,
                    "source": str(fp),
                }
            )
    return rows, selected_target


def synthesize_stability_rows(n: int = 64, seed: int = 0) -> list[dict[str, object]]:
    rows = []
    for i, record in enumerate(synthesize_corpus(n, seed=seed)):
        seq = normalise_rna_sequence(record.seq)
        rows.append(
            {
                "sample_id": f"synthetic_stability_{i:05d}",
                "sequence": seq,
                "target": _synthetic_half_life(seq),
                "target_name": "synthetic_half_life",
                "split": "",
                "source": "synthetic",
                "transcript_id": record.transcript_id,
            }
        )
    return rows


def prepare_stability_dataset(
    *,
    path: Optional[str] = None,
    rows: Optional[Iterable[Mapping[str, object]]] = None,
    target_col: Optional[str] = None,
    n_synthetic: int = 64,
    seed: int = 0,
) -> tuple[list[dict[str, object]], str, bool]:
    if rows is not None:
        raw_rows = [dict(row) for row in rows]
        selected_target = target_col or str(raw_rows[0].get("target_name", "target"))
        official_split_present = bool(raw_rows) and all(
            str(row.get("split", "")).lower() in VALID_SPLITS for row in raw_rows
        )
    elif path is not None:
        raw_rows, selected_target = read_stability_table(path, target_col=target_col)
        official_split_present = bool(raw_rows) and all(
            str(row.get("split", "")).lower() in VALID_SPLITS for row in raw_rows
        )
    else:
        raw_rows = synthesize_stability_rows(n_synthetic, seed=seed)
        selected_target = "synthetic_half_life"
        official_split_present = False

    prepared = []
    n = len(raw_rows)
    for i, row in enumerate(raw_rows):
        split = str(row.get("split", "") or "").strip().lower()
        if not split:
            split = _fallback_split(i, n)
        prepared.append(
            {
                "sample_id": str(row.get("sample_id") or row.get("id") or f"stability_{i:05d}"),
                "sequence": normalise_rna_sequence(str(row["sequence"])),
                "target": float(row["target"]),
                "target_name": str(row.get("target_name") or selected_target),
                "split": split,
                "source": str(row.get("source", "memory")),
            }
        )
    return prepared, selected_target, official_split_present


def _split_counts(samples: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0, "other": 0}
    for sample in samples:
        split = str(sample.get("split", "")).lower()
        counts[split if split in VALID_SPLITS else "other"] += 1
    return counts


def _feature_matrix(samples: Sequence[Mapping[str, object]]) -> np.ndarray:
    rows = []
    for sample in samples:
        features = extract_stability_features(str(sample.get("sequence", "")))
        rows.append([float(features[name]) for name in FEATURE_NAMES])
    return np.asarray(rows, dtype=float)


def _standardize_train(x_train: np.ndarray, x_all: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def _correlation(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if len(a) < 2 or len(b) < 2:
        return None
    ax = a.astype(float) - float(np.mean(a))
    bx = b.astype(float) - float(np.mean(b))
    denom = float(np.linalg.norm(ax) * np.linalg.norm(bx))
    if denom == 0.0:
        return None
    return float(np.dot(ax, bx) / denom)


def _spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if len(a) < 2 or len(b) < 2:
        return None
    return _correlation(_rankdata(a.astype(float)), _rankdata(b.astype(float)))


def _metric_row(split: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
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


def _is_finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def run_stability_predictor(
    input_path: Optional[str] = None,
    *,
    target_col: Optional[str] = None,
    n_synthetic: int = 64,
    seed: int = 0,
    ridge_alpha: float = 1.0,
    min_test_n: int = 2,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    samples, selected_target, official_split = prepare_stability_dataset(
        path=input_path,
        target_col=target_col,
        n_synthetic=n_synthetic,
        seed=seed,
    )
    if not samples:
        raise ValueError("stability predictor requires at least one sample")
    split_counts = _split_counts(samples)
    train_idx = [i for i, s in enumerate(samples) if s["split"] == "train"]
    if not train_idx:
        raise ValueError("stability predictor requires at least one train sample")

    x_raw = _feature_matrix(samples)
    y = np.asarray([float(sample["target"]) for sample in samples], dtype=float)
    x_all, feat_mean, feat_std = _standardize_train(x_raw[train_idx], x_raw)
    coef = _fit_ridge(x_all[train_idx], y[train_idx], ridge_alpha)
    pred = _predict(x_all, coef)

    metrics: dict[str, dict[str, object]] = {}
    for split in ("train", "val", "test"):
        idx = [i for i, sample in enumerate(samples) if sample["split"] == split]
        metrics[split] = _metric_row(split, y[idx], pred[idx])

    predictions = []
    for i, sample in enumerate(samples):
        predictions.append(
            {
                "sample_id": sample["sample_id"],
                "split": sample["split"],
                "sequence": sample["sequence"],
                "target_name": selected_target,
                "target": float(y[i]),
                "prediction": float(pred[i]),
                "source": sample["source"],
            }
        )

    test_metrics = metrics["test"]
    external_input = input_path is not None
    test_ready = (
        int(test_metrics.get("n") or 0) >= int(min_test_n)
        and _is_finite(test_metrics.get("mae"))
        and _is_finite(test_metrics.get("rmse"))
    )
    ready_for_audit = bool(external_input and official_split and test_ready)
    report = {
        "artifact_kind": "stability_predictor_protocol",
        "claim_policy": CLAIM_POLICY,
        "input": {
            "input_path": input_path,
            "source_kind": "external_input" if external_input else "synthetic_smoke",
            "target_name": selected_target,
            "official_split_present": official_split,
            "n_samples": len(samples),
            "split_counts": split_counts,
        },
        "model": {
            "model_family": "feature_ridge_regression",
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
            "ready_for_stability_predictor_audit": ready_for_audit,
            "ready_for_real_te_or_stability_claim": False,
            "ready_for_wet_lab_design_claim": False,
            "synthetic_smoke_only": not external_input,
            "external_input_provided": external_input,
            "official_split_present": official_split,
            "test_split_ready": test_ready,
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
            "Synthetic fallback evidence must not be used as real stability performance.",
            "Half-life/stability labels must be leakage-audited before downstream SOTA claims.",
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
        fh.write("# Stability Predictor Protocol\n\n")
        fh.write(f"- Claim policy: {report.get('claim_policy', CLAIM_POLICY)}\n")
        fh.write(
            f"- Source kind: `{input_meta.get('source_kind')}`; target: "
            f"`{input_meta.get('target_name')}`; samples: `{input_meta.get('n_samples')}`; "
            f"official split present: `{summary.get('official_split_present')}`\n"
        )
        fh.write(
            f"- Ready for stability predictor audit: "
            f"`{summary.get('ready_for_stability_predictor_audit')}`; "
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
        fh.write("\n## Limitations\n\n")
        limitations = report.get("limitations", [])
        if not isinstance(limitations, Sequence) or isinstance(limitations, (str, bytes)):
            limitations = []
        for item in limitations:
            fh.write(f"- {item}\n")
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None)
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--predictions-jsonl", default=None)
    parser.add_argument("--n-synthetic", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--min-test-n", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    report, predictions = run_stability_predictor(
        input_path=args.input,
        target_col=args.target_col,
        n_synthetic=args.n_synthetic,
        seed=args.seed,
        ridge_alpha=args.ridge_alpha,
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
    "extract_stability_features",
    "prepare_stability_dataset",
    "read_stability_table",
    "run_stability_predictor",
    "synthesize_stability_rows",
    "write_predictions_jsonl",
    "write_report_json",
    "write_report_markdown",
    "main",
]
