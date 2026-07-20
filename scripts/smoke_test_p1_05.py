"""P1-05 smoke test: verify GBT oracle works end-to-end."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _gen_synth_data(n: int = 500, seq_len: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    alphabet = "ACGU"
    seqs = []
    labels = []
    for _ in range(n):
        seq = "".join(rng.choice(list(alphabet), size=seq_len))
        # Label: simple rule — count of G + count of AUG + noise
        g_count = seq.count("G")
        aug_count = seq.count("AUG")
        label = float(g_count * 0.5 + aug_count * 2.0 + rng.normal(0, 0.3))
        seqs.append(seq)
        labels.append(label)
    return seqs, np.array(labels, dtype=np.float32)


def test_feature_extractor():
    print("\n=== Feature extractor smoke test ===")
    from models.oracle_final.feature_extractor import (
        HandEngineeredFeatureExtractor,
        FeatureExtractorConfig,
        count_features,
    )

    n_features = count_features()
    print(f"  Total features: {n_features}")
    assert n_features >= 300, f"Expected >=300 features, got {n_features}"

    extractor = HandEngineeredFeatureExtractor()
    seq = "GCCACCAUGGCUUGGCAAAACAGGCAUGGCUUUUAAACGG" * 3  # ~120 nt
    feats = extractor.extract(seq)
    print(f"  Single extract: shape={feats.shape}, dtype={feats.dtype}")
    assert feats.shape == (n_features,), f"Expected ({n_features},), got {feats.shape}"

    seqs = [seq, "AUGGCCAUGG", "UUUUUUUUUUUUUUUU"]
    batch = extractor.extract_batch(seqs)
    print(f"  Batch extract: shape={batch.shape}")
    assert batch.shape == (3, n_features)

    # Determinism check
    feats2 = extractor.extract(seq)
    assert np.array_equal(feats, feats2), "Features not deterministic"

    print("  PASSED")


def test_gbt_oracle():
    print("\n=== GBT oracle smoke test ===")
    try:
        import lightgbm as lgb  # noqa: F401
    except ImportError:
        print("  SKIP: lightgbm not installed")
        return

    from models.oracle_final.gbt_regressor import (
        GBTOracle, GBTOracleHyperparams,
    )

    # Use small hyperparams for fast smoke test
    hp = GBTOracleHyperparams(
        n_estimators=50,
        learning_rate=0.1,
        num_leaves=15,
        objective="quantile",
        quantile_alphas=(0.1, 0.5, 0.9),
        log_transform=True,
    )
    oracle = GBTOracle(hyperparams=hp)

    seqs, labels = _gen_synth_data(n=500, seq_len=50, seed=0)
    val_seqs, val_labels = _gen_synth_data(n=100, seq_len=50, seed=1)
    test_seqs, test_labels = _gen_synth_data(n=100, seq_len=50, seed=2)

    t0 = time.time()
    metrics = oracle.fit(seqs, labels, val_seqs, val_labels)
    print(f"  fit time: {time.time() - t0:.2f}s")
    print(f"  metrics: {metrics}")

    # Predict
    t0 = time.time()
    preds = oracle.predict(test_seqs)
    print(f"  predict time: {time.time() - t0:.2f}s")
    print(f"  preds shape: {preds.shape}, mean={preds.mean():.3f}")
    assert preds.shape == (100,)

    # Predict with uncertainty
    t0 = time.time()
    result = oracle.predict_with_uncertainty(test_seqs)
    print(f"  predict_with_uncertainty time: {time.time() - t0:.2f}s")
    print(f"  mean shape: {result.mean.shape}")
    print(f"  std shape: {result.std.shape}, std mean={result.std.mean():.4f}")
    print(f"  quantiles: {list(result.quantiles.keys()) if result.quantiles else None}")
    print(f"  leaf_embedding shape: {result.leaf_embedding.shape if result.leaf_embedding is not None else None}")
    assert result.std.mean() > 0, "Std should be > 0 with quantile models"

    # Feature importance
    importance = oracle.feature_importance(importance_type="gain", top_k=10)
    print(f"  Top-10 features (by gain):")
    for name, score in importance:
        print(f"    {name}: {score:.1f}")

    # Save / load
    ckpt_dir = Path("/tmp/_smoke_gbt_oracle")
    oracle.save(ckpt_dir)
    print(f"  saved to {ckpt_dir}")

    loaded = GBTOracle.load(ckpt_dir)
    print(f"  loaded: y_mean={loaded._y_mean:.3f}, y_std={loaded._y_std:.3f}")
    preds2 = loaded.predict(test_seqs[:5])
    print(f"  loaded predict shape: {preds2.shape}")
    # Verify predictions match (within tolerance for text format)
    preds_orig = oracle.predict(test_seqs[:5])
    max_diff = np.abs(preds2 - preds_orig).max()
    print(f"  max diff after save/load: {max_diff:.6f}")
    assert max_diff < 1e-3, f"Save/load predictions differ by {max_diff}"

    print("  PASSED")


def main():
    print("=" * 60)
    print("P1-05 Smoke Test")
    print("=" * 60)
    test_feature_extractor()
    test_gbt_oracle()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
