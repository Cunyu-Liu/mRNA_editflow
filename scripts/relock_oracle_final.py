#!/usr/bin/env python3
"""P1-05: Re-lock Oracle #3 with fixed lock_oracle.py (handles duplicate sequences).

This script does NOT retrain the oracle. It:
  1. Unlocks existing read-only files (chmod 644) for lock_manifest.json + .lock_key
  2. Removes old lock artifacts
  3. Reloads training + test data EXACTLY as in train_oracle_final.py
     (same seed=42, same max_train=30000) so training_data_hash matches
  4. Calls lock_oracle() with fixed code (test_label_hashes uses List[str])
  5. Calls verify_lock() to confirm

Usage:
    PYTHONPATH=/home/cunyuliu/mrna_editflow_goal/mrna_editflow \\
    /home/cunyuliu/miniconda3/envs/editflow/bin/python \\
    scripts/relock_oracle_final.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path("/home/cunyuliu/mrna_editflow_goal/mrna_editflow")
sys.path.insert(0, str(_REPO_ROOT))

from models.predictors.data_loaders import (  # noqa: E402
    load_lepplek2022_persistseq,
    load_sample2019,
)
from models.oracle_final.lock_oracle import lock_oracle, verify_lock  # noqa: E402


ORACLE_DIR = _REPO_ROOT / "ckpts" / "p1_05_oracle_final_v1"
LABEL = "mrl"
MAX_TRAIN = 30000  # must match the original training run


def load_data_for_lock():
    """Reproduce the exact train/test data from train_oracle_final.py.

    Returns (train_seqs, test_seqs, test_labels, oracle_id).
    """
    print("=== Reloading data for re-lock (must match original training) ===")

    # --- Leppek 2022 ---
    lepplek_records = list(load_lepplek2022_persistseq(
        label_kind=LABEL, split_filter=None
    ))
    print(f"  Leppek 2022: {len(lepplek_records)} records")

    # --- Sample 2019 val + test ---
    s2019_val = list(load_sample2019(split_filter=["val"]))
    s2019_test = list(load_sample2019(split_filter=["test"]))
    print(f"  Sample 2019 val: {len(s2019_val)} records")
    print(f"  Sample 2019 test: {len(s2019_test)} records")

    # Combine for training pool
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

    # Z-score normalize within source (same as train_oracle_final.py)
    print("  Z-score normalizing labels within each source...")
    for source in ["lepplek2022", "sample2019_val"]:
        mask = np.array([s == source for s in all_train_sources])
        if mask.sum() > 1:
            mu = float(all_train_labels[mask].mean())
            sigma = float(all_train_labels[mask].std())
            if sigma > 0:
                all_train_labels[mask] = (all_train_labels[mask] - mu) / sigma
                print(f"    {source}: mean={mu:.4f} std={sigma:.4f} -> normalized")

    # Test normalization (same as train_oracle_final.py)
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

    # Cap training data (same as train_oracle_final.py)
    if len(all_train_seqs) > MAX_TRAIN:
        print(f"  Capping training data from {len(all_train_seqs)} to {MAX_TRAIN}")
        rng = np.random.default_rng(42)
        idx = rng.choice(len(all_train_seqs), size=MAX_TRAIN, replace=False)
        all_train_seqs = [all_train_seqs[i] for i in idx]
        all_train_labels = all_train_labels[idx]
        all_train_sources = [all_train_sources[i] for i in idx]

    # Split off 10% as validation (same as train_oracle_final.py)
    rng = np.random.default_rng(42)
    n = len(all_train_seqs)
    perm = rng.permutation(n)
    n_val = max(1, n // 10)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_seqs = [all_train_seqs[i] for i in train_idx]
    train_labels = all_train_labels[train_idx]

    print(f"\n  Final splits (for lock hashing):")
    print(f"    train: {len(train_seqs)} sequences")
    print(f"    test:  {len(test_seqs)} sequences")

    return train_seqs, train_labels, test_seqs, test_labels


def main() -> int:
    print("=" * 70)
    print("P1-05: Re-lock Oracle #3 (fixed duplicate-sequence handling)")
    print("=" * 70)
    print(f"  oracle_dir: {ORACLE_DIR}")

    if not ORACLE_DIR.exists():
        print(f"ERROR: Oracle dir not found: {ORACLE_DIR}", file=sys.stderr)
        return 1

    # 1. Unlock existing lock files (chmod 644) so we can remove them
    print("\n=== Step 1: Unlocking existing lock artifacts ===")
    lock_path = ORACLE_DIR / "lock_manifest.json"
    key_path = ORACLE_DIR / ".lock_key"
    for p in [lock_path, key_path]:
        if p.exists():
            try:
                os.chmod(p, 0o644)
                print(f"  chmod 644 {p.name}")
            except OSError as e:
                print(f"  WARNING: could not chmod {p.name}: {e}")

    # 2. Remove old lock artifacts
    print("\n=== Step 2: Removing old lock artifacts ===")
    for p in [lock_path, key_path]:
        if p.exists():
            p.unlink()
            print(f"  removed {p.name}")

    # 3. Reload training + test data (must match original training)
    train_seqs, train_labels, test_seqs, test_labels = load_data_for_lock()

    # 4. Read oracle_id from oracle_meta.json (preserve original ID)
    meta_path = ORACLE_DIR / "oracle_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    oracle_id = meta.get("oracle_id", f"oracle3_v1_{int(time.time())}")
    print(f"\n  oracle_id (from meta): {oracle_id}")

    # 5. Re-lock with fixed code
    print("\n=== Step 3: Re-locking oracle (fixed code) ===")
    manifest = lock_oracle(
        oracle_dir=ORACLE_DIR,
        training_sequences=train_seqs,
        training_labels=train_labels,
        test_sequences=test_seqs,
        test_labels=test_labels,
        oracle_id=oracle_id,
        signing_key=f"oracle3_key_{int(time.time())}",
        make_readonly=True,
    )
    print(f"\n  Lock complete. Oracle {manifest.oracle_id} is now SEALED.")
    print(f"  lock_version: {manifest.lock_version}")

    # 6. Verify lock
    print("\n=== Step 4: Verifying lock ===")
    result = verify_lock(
        oracle_dir=ORACLE_DIR,
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
        print(f"  mismatches (first 10): {result.mismatches[:10]}")
        print(f"  total mismatches: {len(result.mismatches)}")
        print("\n  ERROR: Lock verification failed!", file=sys.stderr)
        return 1

    print("\n" + "=" * 70)
    print("P1-05 Re-lock SUCCESS")
    print("=" * 70)
    print(f"  Oracle {manifest.oracle_id} is sealed and verified.")
    print(f"  lock_version: {manifest.lock_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
