#!/usr/bin/env python3
"""P1-05: Train the independent final oracle (Oracle #3) and lock & seal it.

Oracle #3 is a Gradient-Boosted Trees (LightGBM) regressor on hand-engineered
features, trained on:
    - Lepkek 2022 PERSIST-Seq (233 mRNAs, label=MRL or half_life)
    - Sample 2019 MPRA val split (~28k 5'UTRs, label=RL, NEVER seen by #1/#2)

Independence from Oracle #1/#2 (P1-04 teacher):
    - Different architecture (GBT vs CNN/Transformer)
    - Different feature space (340+ hand-engineered vs one-hot/embedding)
    - Different training data slice (Sample 2019 VAL vs Sample 2019 TRAIN)
    - Test split is the same frozen test split (never seen by any model)

After training, the oracle is locked & sealed:
    - SHA-256 of model files, feature extractor config, training data record IDs
    - HMAC-SHA256 signature
    - chmod 444 (read-only)
    - Test labels stored as one-way SHA-256 hashes

Usage:
    PYTHONPATH=/home/cunyuliu/mrna_editflow_goal \\
    /home/cunyuliu/miniconda3/envs/editflow/bin/python \\
    scripts/train_oracle_final.py [--label mrl|half_life] [--max-train N]

Output:
    ckpts/p1_05_oracle_final_v1/
        mean_model.txt
        q0.1_model.txt
        q0.5_model.txt
        q0.9_model.txt
        oracle_meta.json
        lock_manifest.json
        .lock_key
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

from models.predictors.data_loaders import (  # noqa: E402
    load_lepplek2022_persistseq,
    load_sample2019,
)
from models.oracle_final.feature_extractor import (  # noqa: E402
    HandEngineeredFeatureExtractor,
    FeatureExtractorConfig,
)
from models.oracle_final.gbt_regressor import (  # noqa: E402
    GBTOracle,
    GBTOracleHyperparams,
)
from models.oracle_final.lock_oracle import lock_oracle, verify_lock  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_ORACLE_DIR = _REPO_ROOT / "ckpts" / "p1_05_oracle_final_v1"
DEFAULT_LABEL = "mrl"  # ribosome load (compatible across Lepkek + Sample 2019)
DEFAULT_MAX_TRAIN = 50000  # cap to keep training time reasonable


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(
    label: str,
    max_train: Optional[int] = None,
) -> Tuple[List[str], np.ndarray, List[str], np.ndarray, List[str], np.ndarray, Dict]:
    """Load training data for Oracle #3.

    Train = Lepkek 2022 (all splits) + Sample 2019 VAL split (unseen by #1/#2)
    Val   = 10% of Train (for GBT early stopping)
    Test  = Sample 2019 TEST split (frozen)

    Returns:
        train_seqs, train_labels, val_seqs, val_labels, test_seqs, test_labels, meta
    """
    print(f"\n=== Loading training data (label={label}) ===")

    # --- Lepkek 2022 ---
    lepplek_records = list(load_lepplek2022_persistseq(
        label_kind=label, split_filter=None  # use all splits for training
    ))
    print(f"  Lepkek 2022: {len(lepplek_records)} records")
    if not lepplek_records:
        raise RuntimeError("No Lepkek 2022 records loaded")

    # --- Sample 2019 val + test splits ---
    # NOTE: We use Sample 2019 VAL split for training (unseen by Oracle #1/#2
    # which train on the TRAIN split). Test split is the frozen test split
    # shared across all oracles.
    s2019_val = list(load_sample2019(split_filter=["val"]))
    s2019_test = list(load_sample2019(split_filter=["test"]))
    print(f"  Sample 2019 val: {len(s2019_val)} records (for #3 train)")
    print(f"  Sample 2019 test: {len(s2019_test)} records (frozen test)")

    # Combine Lepkek + Sample 2019 val for training
    all_train_seqs = [r["sequence"] for r in lepplek_records] + \
                     [r["sequence"] for r in s2019_val]
    all_train_labels = np.array(
        [r["label"] for r in lepplek_records] +
        [r["label"] for r in s2019_val],
        dtype=np.float32,
    )
    all_train_sources = (
        ["lepplek2022"] * len(lepplek_records) +
        ["sample2019_val"] * len(s2019_val)
    )

    # Z-score normalize labels within each source (so Lepkek MRL and
    # Sample 2019 RL are on comparable scales despite different ranges).
    # NOTE: GBTOracle also does log1p + standardization internally, so this
    # pre-normalization is a secondary safeguard for cross-dataset blending.
    print("  Z-score normalizing labels within each source...")
    for source in ["lepplek2022", "sample2019_val"]:
        mask = np.array([s == source for s in all_train_sources])
        if mask.sum() > 1:
            mu = float(all_train_labels[mask].mean())
            sigma = float(all_train_labels[mask].std())
            if sigma > 0:
                all_train_labels[mask] = (all_train_labels[mask] - mu) / sigma
                print(f"    {source}: mean={mu:.4f} std={sigma:.4f} -> normalized")

    # Test labels: z-score normalize using Sample 2019 val stats
    # (so test predictions are on the same scale as training)
    s2019_val_labels = np.array(
        [r["label"] for r in s2019_val], dtype=np.float32
    )
    if len(s2019_val_labels) > 1 and s2019_val_labels.std() > 0:
        test_mu = float(s2019_val_labels.mean())
        test_sigma = float(s2019_val_labels.std())
    else:
        test_mu, test_sigma = 0.0, 1.0

    test_seqs = [r["sequence"] for r in s2019_test]
    test_labels = (np.array([r["label"] for r in s2019_test], dtype=np.float32) - test_mu) / test_sigma
    print(f"  Test normalization: mu={test_mu:.4f} sigma={test_sigma:.4f}")

    # Cap training data if requested
    if max_train is not None and len(all_train_seqs) > max_train:
        print(f"  Capping training data from {len(all_train_seqs)} to {max_train}")
        rng = np.random.default_rng(42)
        idx = rng.choice(len(all_train_seqs), size=max_train, replace=False)
        all_train_seqs = [all_train_seqs[i] for i in idx]
        all_train_labels = all_train_labels[idx]
        all_train_sources = [all_train_sources[i] for i in idx]

    # Split off 10% as validation (for GBT early stopping)
    rng = np.random.default_rng(42)
    n = len(all_train_seqs)
    perm = rng.permutation(n)
    n_val = max(1, n // 10)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_seqs = [all_train_seqs[i] for i in train_idx]
    train_labels = all_train_labels[train_idx]
    val_seqs = [all_train_seqs[i] for i in val_idx]
    val_labels = all_train_labels[val_idx]

    print(f"\n  Final splits:")
    print(f"    train: {len(train_seqs)} sequences")
    print(f"    val:   {len(val_seqs)} sequences (for early stopping)")
    print(f"    test:  {len(test_seqs)} sequences (frozen)")

    meta = {
        "label": label,
        "n_lepplek": len(lepplek_records),
        "n_sample2019_val": len(s2019_val),
        "n_sample2019_test": len(s2019_test),
        "n_train": len(train_seqs),
        "n_val": len(val_seqs),
        "n_test": len(test_seqs),
        "test_mu": test_mu,
        "test_sigma": test_sigma,
        "normalization": "z-score within source; test normalized using sample2019_val stats",
    }
    return train_seqs, train_labels, val_seqs, val_labels, test_seqs, test_labels, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Train Oracle #3 (GBT)")
    parser.add_argument(
        "--label", type=str, default=DEFAULT_LABEL,
        choices=["mrl", "half_life", "protein_output",
                 "in_cell_stability", "in_solution_stability"],
        help="Label to predict (must be available in Lepkek 2022)",
    )
    parser.add_argument(
        "--max-train", type=int, default=DEFAULT_MAX_TRAIN,
        help="Cap on training records (None = no cap)",
    )
    parser.add_argument(
        "--oracle-dir", type=str, default=str(DEFAULT_ORACLE_DIR),
        help="Output directory for oracle artifacts",
    )
    parser.add_argument(
        "--no-lock", action="store_true",
        help="Skip lock & seal procedure (for debugging)",
    )
    parser.add_argument(
        "--no-quantile", action="store_true",
        help="Skip quantile regression (train mean model only, faster)",
    )
    args = parser.parse_args()

    oracle_dir = Path(args.oracle_dir)
    oracle_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("P1-05: Training Oracle #3 (Independent Final Oracle)")
    print("=" * 70)
    print(f"  label: {args.label}")
    print(f"  max_train: {args.max_train}")
    print(f"  oracle_dir: {oracle_dir}")
    print(f"  lock & seal: {not args.no_lock}")
    print(f"  quantile models: {not args.no_quantile}")

    # 1. Load data
    train_seqs, train_labels, val_seqs, val_labels, \
        test_seqs, test_labels, data_meta = load_training_data(
            label=args.label,
            max_train=args.max_train,
        )

    # 2. Configure feature extractor
    print("\n=== Configuring feature extractor ===")
    feature_config = FeatureExtractorConfig()  # defaults: max_kmer_k=6, all features
    feature_extractor = HandEngineeredFeatureExtractor(feature_config)
    n_features = len(feature_extractor.feature_names())
    print(f"  FeatureExtractor: {n_features} features")

    # 3. Configure GBT oracle
    print("\n=== Configuring GBT oracle ===")
    hp = GBTOracleHyperparams(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=8,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=2026,
        n_jobs=-1,
        objective="quantile" if not args.no_quantile else "regression",
        quantile_alphas=(0.1, 0.5, 0.9),
        early_stopping_rounds=50,
        feature_config=feature_config,
        log_transform=True,
    )
    oracle = GBTOracle(hyperparams=hp, feature_extractor=feature_extractor)

    # 4. Train
    print("\n=== Training GBT oracle ===")
    t0 = time.time()
    metrics = oracle.fit(
        sequences=train_seqs,
        labels=train_labels,
        val_sequences=val_seqs,
        val_labels=val_labels,
    )
    train_time = time.time() - t0
    print(f"\n  Training time: {train_time:.1f}s")
    print(f"  Train Pearson r: {metrics.get('train_pearson', 'N/A')}")
    print(f"  Val Pearson r: {metrics.get('val_pearson', 'N/A')}")

    # 5. Compute test metrics (ONE-TIME, will be frozen)
    print("\n=== Computing test metrics (ONE-TIME, will be frozen) ===")
    test_pred = oracle.predict(test_seqs)
    from scipy.stats import pearsonr, spearmanr
    test_pearson = float(pearsonr(test_pred, test_labels)[0]) if len(test_labels) > 1 else float("nan")
    test_spearman = float(spearmanr(test_pred, test_labels)[0]) if len(test_labels) > 1 else float("nan")
    test_rmse = float(np.sqrt(np.mean((test_pred - test_labels) ** 2)))
    test_mae = float(np.mean(np.abs(test_pred - test_labels)))
    print(f"  Test Pearson r: {test_pearson:.4f}")
    print(f"  Test Spearman r: {test_spearman:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test MAE: {test_mae:.4f}")

    # 6. Save oracle (writes mean_model.txt, q*.txt, oracle_meta.json)
    print(f"\n=== Saving oracle to {oracle_dir} ===")
    oracle.save(oracle_dir)

    # Augment oracle_meta.json with extra fields needed for lock & audit
    meta_path = oracle_dir / "oracle_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    meta.update({
        "oracle_id": f"oracle3_v1_{int(time.time())}",
        "oracle_type": "gbt_regressor",
        "label": args.label,
        "training_data": {
            "lepplek2022": data_meta["n_lepplek"],
            "sample2019_val": data_meta["n_sample2019_val"],
        },
        "test_data": {
            "sample2019_test": data_meta["n_sample2019_test"],
        },
        "splits": {
            "train": data_meta["n_train"],
            "val": data_meta["n_val"],
            "test": data_meta["n_test"],
        },
        "normalization": data_meta["normalization"],
        "test_mu": data_meta["test_mu"],
        "test_sigma": data_meta["test_sigma"],
        "test_metrics": {
            "pearson": test_pearson,
            "spearman": test_spearman,
            "rmse": test_rmse,
            "mae": test_mae,
            "computed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": "Computed ONCE at lock time; never recomputed.",
        },
        "training_time_sec": train_time,
        "feature_extractor": {
            "n_features": n_features,
            "config": feature_config.__dict__,
        },
        "lock_status": "unlocked",
    })
    # Convert window_sizes tuple to list for JSON serialization
    if "feature_extractor" in meta and "config" in meta["feature_extractor"]:
        fc = meta["feature_extractor"]["config"]
        if "window_sizes" in fc:
            fc["window_sizes"] = list(fc["window_sizes"])
    # Also fix hyperparams.feature_config if present (already saved by GBTOracle.save)
    if "hyperparams" in meta and isinstance(meta["hyperparams"], dict):
        hp_dict = meta["hyperparams"]
        if hp_dict.get("feature_config") and isinstance(hp_dict["feature_config"], dict):
            fc = hp_dict["feature_config"]
            if "window_sizes" in fc and isinstance(fc["window_sizes"], list):
                pass  # already a list
        if "quantile_alphas" in hp_dict and isinstance(hp_dict["quantile_alphas"], list):
            pass  # already a list

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True, default=str)
    print(f"  oracle_meta.json updated with test metrics + lock fields")

    # 7. Lock & seal
    if not args.no_lock:
        print("\n=== Locking & sealing oracle ===")
        manifest = lock_oracle(
            oracle_dir=oracle_dir,
            training_sequences=train_seqs,
            training_labels=train_labels,
            test_sequences=test_seqs,
            test_labels=test_labels,
            oracle_id=meta["oracle_id"],
            signing_key=f"oracle3_key_{int(time.time())}",
            make_readonly=True,
        )
        print(f"\n  Lock complete. Oracle {manifest.oracle_id} is SEALED.")

        # Verify lock
        print("\n=== Verifying lock ===")
        result = verify_lock(
            oracle_dir=oracle_dir,
            training_sequences=train_seqs,
            test_sequences=test_seqs,
            test_labels=test_labels,
        )
        print(f"  valid: {result.valid}")
        print(f"  manifest_valid: {result.manifest_valid}")
        print(f"  model_hashes_match: {result.model_hashes_match}")
        print(f"  meta_hash_match: {result.meta_hash_match}")
        print(f"  training_data_hash_match: {result.training_data_hash_match}")
        print(f"  test_label_hashes_match: {result.test_label_hashes_match}")
        if result.mismatches:
            print(f"  mismatches: {result.mismatches}")
        if not result.valid:
            print("\n  ERROR: Lock verification failed!", file=sys.stderr)
            return 1
    else:
        print("\n  Skipping lock & seal (--no-lock flag)")

    print("\n" + "=" * 70)
    print("P1-05 Oracle #3 training complete")
    print("=" * 70)
    print(f"  Test Pearson r: {test_pearson:.4f}")
    print(f"  Test Spearman r: {test_spearman:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Oracle dir: {oracle_dir}")
    print(f"  Lock status: {'SEALED' if not args.no_lock else 'UNLOCKED'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
