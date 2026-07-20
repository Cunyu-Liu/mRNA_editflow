#!/usr/bin/env python3
"""P1-04: Cross-fitted predictor ensemble training (CNN-50mer × Sample 2019).

Runs k-fold (k=5) × M-seed (M=3) cross-fitting of CNN-50mer predictor on
Sample 2019 MPRA data. Produces:
    - 15 checkpoints (5 folds × 3 seeds)
    - Held-out in-fold predictions (unbiased teacher signal for train split)
    - Val + test predictions (mean + std across 15 models)
    - Per-fold metrics JSON

Usage:
    PYTHONPATH=/home/cunyuliu/mrna_editflow_goal \\
    /home/cunyuliu/miniconda3/envs/editflow/bin/python \\
    scripts/train_p1_04_crossfit.py \\
        --max-train 500000 --max-val 50000 --max-test 50000 \\
        --n-folds 5 --n-seeds 3 --device cuda

Output:
    ckpts/p1_04_predictors/cnn_50mer__sample2019_mpra__fold{0..4}_seed{42..44}.pt
    data/processed/p1_04_predictions/cnn_50mer__sample2019_mpra__crossfit.json
    data/processed/p1_04_predictions/cnn_50mer__sample2019_mpra__crossfit.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from models.predictors.data_loaders import load_sample2019  # noqa: E402
from models.predictors.cnn_50mer import CNN50merPredictor  # noqa: E402
from models.predictors.crossfit import (  # noqa: E402
    CrossFitConfig,
    cross_fit,
    build_records,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CKPT_DIR = _REPO_ROOT / "ckpts" / "p1_04_predictors"
DEFAULT_PRED_DIR = _REPO_ROOT / "data" / "processed" / "p1_04_predictions"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sample2019_records(
    max_train: Optional[int] = None,
    max_val: Optional[int] = None,
    max_test: Optional[int] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load Sample 2019 records partitioned by deterministic split.

    Returns:
        (train_records, val_records, test_records)
    """
    print("\n=== Loading Sample 2019 MPRA records ===")
    t0 = time.time()
    train_records, val_records, test_records = build_records(
        load_sample2019, split_filter=None
    )
    elapsed = time.time() - t0
    print(f"  Loaded in {elapsed:.1f}s")
    print(f"  train: {len(train_records):,}")
    print(f"  val:   {len(val_records):,}")
    print(f"  test:  {len(test_records):,}")

    # Print per-library breakdown
    for split_name, records in [("train", train_records), ("val", val_records), ("test", test_records)]:
        lib_counts: Dict[str, int] = {}
        for r in records:
            lib = r.get("library_kind", "unknown")
            lib_counts[lib] = lib_counts.get(lib, 0) + 1
        print(f"  {split_name} libraries: {lib_counts}")

    return train_records, val_records, test_records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="P1-04 cross-fitting training")
    parser.add_argument(
        "--arch", type=str, default="cnn_50mer",
        choices=["cnn_50mer"],
        help="Architecture to train",
    )
    parser.add_argument(
        "--dataset", type=str, default="sample2019_mpra",
        help="Dataset name",
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of cross-fitting folds",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=3,
        help="Number of random init seeds per fold",
    )
    parser.add_argument(
        "--base-seed", type=int, default=42,
        help="Base random seed (actual seed = base_seed + offset)",
    )
    parser.add_argument(
        "--max-train", type=int, default=500000,
        help="Cap on training records (None = no cap)",
    )
    parser.add_argument(
        "--max-val", type=int, default=50000,
        help="Cap on val records",
    )
    parser.add_argument(
        "--max-test", type=int, default=50000,
        help="Cap on test records",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=30,
        help="Number of training epochs per model",
    )
    parser.add_argument(
        "--batch-size", type=int, default=512,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Dataloader workers",
    )
    parser.add_argument(
        "--n-mc-samples", type=int, default=30,
        help="MC dropout samples for uncertainty",
    )
    parser.add_argument(
        "--ckpt-dir", type=str, default=str(DEFAULT_CKPT_DIR),
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--pred-dir", type=str, default=str(DEFAULT_PRED_DIR),
        help="Predictions directory",
    )
    parser.add_argument(
        "--start-fold", type=int, default=0,
        help="Start from this fold (for resuming)",
    )
    parser.add_argument(
        "--start-seed-offset", type=int, default=0,
        help="Start from this seed offset within the fold (for resuming)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("P1-04: Cross-Fitted Predictor Ensemble Training")
    print("=" * 70)
    print(f"  arch: {args.arch}")
    print(f"  dataset: {args.dataset}")
    print(f"  folds: {args.n_folds}  seeds: {args.n_seeds}  base_seed: {args.base_seed}")
    print(f"  max_train: {args.max_train}")
    print(f"  max_val: {args.max_val}")
    print(f"  max_test: {args.max_test}")
    print(f"  n_epochs: {args.n_epochs}  batch_size: {args.batch_size}")
    print(f"  device: {args.device}")
    print(f"  ckpt_dir: {args.ckpt_dir}")
    print(f"  pred_dir: {args.pred_dir}")
    if args.start_fold > 0 or args.start_seed_offset > 0:
        print(f"  RESUMING from fold={args.start_fold} seed_offset={args.start_seed_offset}")

    # 1. Load data
    train_records, val_records, test_records = load_sample2019_records(
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
    )

    # 2. Configure predictor hyperparams
    hyperparams = {
        "max_len": 50,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": args.device,
        "num_workers": args.num_workers,
        "n_mc_samples": args.n_mc_samples,
        "patience": 5,
        "log_transform": True,
    }

    # 3. Configure cross-fit
    config = CrossFitConfig(
        arch_name=args.arch,
        dataset_name=args.dataset,
        predictor_cls=CNN50merPredictor,
        n_folds=args.n_folds,
        n_seeds=args.n_seeds,
        base_seed=args.base_seed,
        ckpt_dir=Path(args.ckpt_dir),
        predictions_dir=Path(args.pred_dir),
        max_train_records=args.max_train,
        max_val_records=args.max_val,
        max_test_records=args.max_test,
        hyperparams=hyperparams,
    )

    # 4. Run cross-fitting
    print("\n=== Starting cross-fitting ===")
    t0 = time.time()

    # If resuming, skip already-completed (fold, seed) pairs
    if args.start_fold > 0 or args.start_seed_offset > 0:
        print(f"  Resume mode: skipping folds 0..{args.start_fold-1} and "
              f"seed_offsets 0..{args.start_seed_offset-1} within fold {args.start_fold}")
        # For now, just run the full cross_fit — the harness will overwrite existing
        # checkpoints. A proper resume would require modifying cross_fit() to skip
        # existing checkpoints. TODO: implement resume in cross_fit().
        print("  NOTE: Full resume not yet implemented. Running all folds/seeds.")

    result = cross_fit(
        config=config,
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        verbose=True,
    )
    elapsed = time.time() - t0

    # 5. Print summary
    print("\n" + "=" * 70)
    print("P1-04 Cross-Fitting Complete")
    print("=" * 70)
    print(f"  Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Checkpoints saved: {result.n_checkpoints}")
    print(f"  Fold metrics:")
    for m in result.fold_metrics:
        print(f"    fold={m['fold_idx']} seed={m['seed']} "
              f"heldout_r={m['heldout_pearson']:.4f} "
              f"val_r={m['val_pearson']:.4f} "
              f"test_r={m['test_pearson']:.4f} "
              f"({m['elapsed_sec']:.1f}s)")

    # Aggregate metrics
    heldout_rs = [m["heldout_pearson"] for m in result.fold_metrics]
    val_rs = [m["val_pearson"] for m in result.fold_metrics]
    test_rs = [m["test_pearson"] for m in result.fold_metrics]
    print(f"\n  Heldout Pearson: mean={np.mean(heldout_rs):.4f} std={np.std(heldout_rs):.4f}")
    print(f"  Val Pearson:     mean={np.mean(val_rs):.4f} std={np.std(val_rs):.4f}")
    print(f"  Test Pearson:    mean={np.mean(test_rs):.4f} std={np.std(test_rs):.4f}")

    # Uncertainty summary
    if result.heldout_uncertainty is not None:
        print(f"\n  Heldout uncertainty: mean={np.nanmean(result.heldout_uncertainty):.4f} "
              f"std={np.nanstd(result.heldout_uncertainty):.4f}")
    if result.test_uncertainty is not None:
        print(f"  Test uncertainty:    mean={np.nanmean(result.test_uncertainty):.4f} "
              f"std={np.nanstd(result.test_uncertainty):.4f}")

    print(f"\n  Predictions saved to: {args.pred_dir}")
    print(f"  Checkpoints saved to: {args.ckpt_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
