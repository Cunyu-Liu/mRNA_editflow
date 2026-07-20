"""P1-04: k-fold cross-fitting harness for predictor ensemble.

This module orchestrates training of M seeds × k folds per (architecture, dataset)
pair, and aggregates predictions into an unbiased teacher signal.

Pipeline:
    1. Load all records for a given (dataset, split="train")
    2. Partition into k=5 contiguous folds (with fixed shuffle seed)
    3. For each fold i, for each seed s in [0..M-1]:
        a. Train predictor on union(folds != i) using seed s
        b. Predict on fold i (held-out in-fold)
        c. Predict on val + test splits
        d. Save checkpoint + predictions
    4. Aggregate:
        - held-out in-fold predictions → unbiased teacher signal for entire train split
        - val/test predictions across (fold × seed) → mean + std (uncertainty)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np

# Relative imports (when installed under models/predictors/)
try:
    from .base import PredictorBase, PredictionResult
    from .data_loaders import (
        load_sample2019, load_saluki_halflife, load_codonbert_stability,
        PredictorRecord,
    )
except ImportError:
    # Allow running as standalone module
    from base import PredictorBase, PredictionResult  # type: ignore
    from data_loaders import (  # type: ignore
        load_sample2019, load_saluki_halflife, load_codonbert_stability,
        PredictorRecord,
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CrossFitConfig:
    """Configuration for one (architecture, dataset) cross-fitting run.

    Attributes:
        arch_name: architecture name (e.g., "cnn_50mer", "transformer_utr")
        dataset_name: dataset name (e.g., "sample2019_mpra", "saluki_halflife_human")
        predictor_cls: PredictorBase subclass to instantiate
        n_folds: number of cross-fitting folds (default 5)
        n_seeds: number of random init seeds per fold (default 3)
        base_seed: base random seed (default 42); actual seed = base_seed + s
        ckpt_dir: where to save checkpoints
        predictions_dir: where to save per-fold predictions
        max_train_records: cap on records per fold (None = no cap)
        max_val_records: cap on val records (None = no cap)
        max_test_records: cap on test records (None = no cap)
    """
    arch_name: str
    dataset_name: str
    predictor_cls: Type[PredictorBase]
    n_folds: int = 5
    n_seeds: int = 3
    base_seed: int = 42
    ckpt_dir: Path = Path("ckpts/p1_04_predictors")
    predictions_dir: Path = Path("data/processed/p1_04_predictions")
    max_train_records: Optional[int] = None
    max_val_records: Optional[int] = None
    max_test_records: Optional[int] = None
    hyperparams: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fold partitioning
# ---------------------------------------------------------------------------

def partition_folds(
    n_records: int, n_folds: int = 5, seed: int = 42
) -> np.ndarray:
    """Return fold assignment (n_records,) array with values in [0, n_folds).

    Uses a fixed-seed shuffle so partitions are deterministic.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(n_records)
    rng.shuffle(indices)
    fold_sizes = np.full(n_folds, n_records // n_folds, dtype=int)
    fold_sizes[: n_records % n_folds] += 1
    assignments = np.empty(n_records, dtype=int)
    start = 0
    for fold_idx, size in enumerate(fold_sizes):
        assignments[indices[start : start + size]] = fold_idx
        start += size
    return assignments


# ---------------------------------------------------------------------------
# Cross-fitting driver
# ---------------------------------------------------------------------------

@dataclass
class CrossFitResult:
    """Container for cross-fitting outputs.

    Attributes:
        config: the CrossFitConfig used
        fold_metrics: list of per-fold dicts with metrics
        heldout_predictions: (N_train,) mean predictions across folds (unbiased)
        heldout_uncertainty: (N_train,) std across (fold, seed)
        val_predictions: (N_val,) mean
        val_uncertainty: (N_val,) std
        test_predictions: (N_test,) mean
        test_uncertainty: (N_test,) std
        n_checkpoints: total number of checkpoints saved
    """
    config: CrossFitConfig
    fold_metrics: List[Dict[str, Any]] = field(default_factory=list)
    heldout_predictions: Optional[np.ndarray] = None
    heldout_uncertainty: Optional[np.ndarray] = None
    val_predictions: Optional[np.ndarray] = None
    val_uncertainty: Optional[np.ndarray] = None
    test_predictions: Optional[np.ndarray] = None
    test_uncertainty: Optional[np.ndarray] = None
    n_checkpoints: int = 0


def cross_fit(
    config: CrossFitConfig,
    train_records: List[Dict],
    val_records: List[Dict],
    test_records: List[Dict],
    verbose: bool = True,
) -> CrossFitResult:
    """Run k-fold × M-seed cross-fitting.

    Args:
        config: CrossFitConfig
        train_records: list of dicts (must have "sequence" and "label")
        val_records: held-out validation set
        test_records: FROZEN test set (predictions only, never training)
        verbose: print progress

    Returns:
        CrossFitResult with all predictions and metrics
    """
    n_train = len(train_records)
    n_val = len(val_records)
    n_test = len(test_records)

    if verbose:
        print(f"\n=== Cross-fit: {config.arch_name} × {config.dataset_name} ===")
        print(f"  train={n_train}  val={n_val}  test={n_test}")
        print(f"  folds={config.n_folds}  seeds={config.n_seeds}")

    # Apply caps if specified
    if config.max_train_records and n_train > config.max_train_records:
        train_records = train_records[: config.max_train_records]
        n_train = len(train_records)
    if config.max_val_records and n_val > config.max_val_records:
        val_records = val_records[: config.max_val_records]
        n_val = len(val_records)
    if config.max_test_records and n_test > config.max_test_records:
        test_records = test_records[: config.max_test_records]
        n_test = len(test_records)

    # Partition train into folds
    fold_assignments = partition_folds(n_train, config.n_folds, seed=config.base_seed)

    # Pre-extract sequences/labels
    train_seqs = [r["sequence"] for r in train_records]
    train_labels = np.array([r["label"] for r in train_records], dtype=np.float32)
    val_seqs = [r["sequence"] for r in val_records]
    val_labels = np.array([r["label"] for r in val_records], dtype=np.float32)
    test_seqs = [r["sequence"] for r in test_records]
    test_labels = np.array([r["label"] for r in test_records], dtype=np.float32)

    # Storage for predictions across (fold, seed)
    heldout_preds = np.full((n_train, config.n_folds, config.n_seeds), np.nan,
                             dtype=np.float32)
    val_preds = np.full((n_val, config.n_folds, config.n_seeds), np.nan,
                         dtype=np.float32)
    test_preds = np.full((n_test, config.n_folds, config.n_seeds), np.nan,
                          dtype=np.float32)

    fold_metrics: List[Dict[str, Any]] = []
    n_checkpoints = 0
    config.ckpt_dir.mkdir(parents=True, exist_ok=True)
    config.predictions_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx in range(config.n_folds):
        train_mask = fold_assignments != fold_idx
        heldout_mask = fold_assignments == fold_idx

        X_train_fold = [train_seqs[i] for i in np.where(train_mask)[0]]
        y_train_fold = train_labels[train_mask]
        X_heldout_fold = [train_seqs[i] for i in np.where(heldout_mask)[0]]
        y_heldout_fold = train_labels[heldout_mask]

        for seed_offset in range(config.n_seeds):
            seed = config.base_seed + seed_offset
            t0 = time.time()

            # Instantiate predictor
            predictor = config.predictor_cls(
                name=f"{config.arch_name}__{config.dataset_name}__fold{fold_idx}_seed{seed}",
                dataset_name=config.dataset_name,
                fold_idx=fold_idx,
                seed=seed,
                hyperparams=config.hyperparams,
            )

            # Fit + predict
            result = predictor.fit_predict(
                X_train_fold, y_train_fold, X_heldout_fold,
                X_val=val_seqs, y_val=val_labels,
            )

            # Predict on val + test
            val_result = predictor.predict_with_uncertainty(val_seqs)
            test_result = predictor.predict_with_uncertainty(test_seqs)

            # Store predictions
            heldout_preds[heldout_mask, fold_idx, seed_offset] = result.mean
            val_preds[:, fold_idx, seed_offset] = val_result.mean
            test_preds[:, fold_idx, seed_offset] = test_result.mean

            # Compute metrics
            from scipy.stats import pearsonr, spearmanr
            heldout_r = pearsonr(result.mean, y_heldout_fold)[0] if len(y_heldout_fold) > 1 else float("nan")
            val_r = pearsonr(val_result.mean, val_labels)[0] if n_val > 1 else float("nan")
            test_r = pearsonr(test_result.mean, test_labels)[0] if n_test > 1 else float("nan")

            metrics = {
                "fold_idx": fold_idx,
                "seed": seed,
                "n_train": int(train_mask.sum()),
                "n_heldout": int(heldout_mask.sum()),
                "heldout_pearson": float(heldout_r),
                "val_pearson": float(val_r),
                "test_pearson": float(test_r),
                "elapsed_sec": time.time() - t0,
            }
            fold_metrics.append(metrics)

            # Save checkpoint
            ckpt_name = f"{config.arch_name}__{config.dataset_name}__fold{fold_idx}_seed{seed}"
            predictor.save(config.ckpt_dir / ckpt_name)
            n_checkpoints += 1

            if verbose:
                print(f"  fold={fold_idx} seed={seed} "
                      f"heldout_r={heldout_r:.4f} val_r={val_r:.4f} test_r={test_r:.4f} "
                      f"({metrics['elapsed_sec']:.1f}s)")

    # Aggregate predictions across (fold, seed) — ignore NaN
    heldout_mean = np.nanmean(heldout_preds, axis=(1, 2))
    heldout_std = np.nanstd(heldout_preds, axis=(1, 2))
    val_mean = np.nanmean(val_preds, axis=(1, 2))
    val_std = np.nanstd(val_preds, axis=(1, 2))
    test_mean = np.nanmean(test_preds, axis=(1, 2))
    test_std = np.nanstd(test_preds, axis=(1, 2))

    result = CrossFitResult(
        config=config,
        fold_metrics=fold_metrics,
        heldout_predictions=heldout_mean,
        heldout_uncertainty=heldout_std,
        val_predictions=val_mean,
        val_uncertainty=val_std,
        test_predictions=test_mean,
        test_uncertainty=test_std,
        n_checkpoints=n_checkpoints,
    )

    # Save results
    result_path = config.predictions_dir / f"{config.arch_name}__{config.dataset_name}__crossfit.json"
    _save_result(result, result_path)
    if verbose:
        print(f"  saved: {result_path}")

    return result


def _save_result(result: CrossFitResult, path: Path) -> None:
    """Save cross-fit result metadata to JSON (predictions saved separately as npz)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "arch_name": result.config.arch_name,
        "dataset_name": result.config.dataset_name,
        "n_folds": result.config.n_folds,
        "n_seeds": result.config.n_seeds,
        "base_seed": result.config.base_seed,
        "n_checkpoints": result.n_checkpoints,
        "fold_metrics": result.fold_metrics,
    }
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    # Save predictions as .npz
    npz_path = path.with_suffix(".npz")
    np.savez_compressed(
        str(npz_path),
        heldout_predictions=result.heldout_predictions if result.heldout_predictions is not None else np.array([]),
        heldout_uncertainty=result.heldout_uncertainty if result.heldout_uncertainty is not None else np.array([]),
        val_predictions=result.val_predictions if result.val_predictions is not None else np.array([]),
        val_uncertainty=result.val_uncertainty if result.val_uncertainty is not None else np.array([]),
        test_predictions=result.test_predictions if result.test_predictions is not None else np.array([]),
        test_uncertainty=result.test_uncertainty if result.test_uncertainty is not None else np.array([]),
    )


# ---------------------------------------------------------------------------
# Convenience: build records from a dataset loader
# ---------------------------------------------------------------------------

def build_records(
    loader: Callable, split_filter: Optional[List[str]] = None, **loader_kwargs
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load records from a loader and partition by split.

    Returns:
        (train_records, val_records, test_records)
    """
    train, val, test = [], [], []
    for record in loader(split_filter=split_filter, **loader_kwargs):
        s = record["split"]
        if s == "train":
            train.append(record)
        elif s == "val":
            val.append(record)
        elif s == "test":
            test.append(record)
    return train, val, test


__all__ = [
    "CrossFitConfig",
    "CrossFitResult",
    "partition_folds",
    "cross_fit",
    "build_records",
]
