"""P1-04 smoke test: verify CNN + Transformer predictors work end-to-end.

Creates a tiny synthetic dataset, trains both predictors for 2 epochs on CPU,
and verifies predict/predict_with_uncertainty work correctly.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
# scripts/smoke_test_p1_04.py -> parents[0]=scripts, parents[1]=project root
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _gen_synth_data(n: int = 200, seq_len: int = 50, seed: int = 0):
    """Generate synthetic sequences + labels for smoke test."""
    rng = np.random.default_rng(seed)
    alphabet = "ACGU"
    seqs = []
    labels = []
    for _ in range(n):
        seq = "".join(rng.choice(list(alphabet), size=seq_len))
        # Label: simple rule — count of G + noise
        g_count = seq.count("G")
        label = float(g_count + rng.normal(0, 0.5))
        seqs.append(seq)
        labels.append(label)
    return seqs, np.array(labels, dtype=np.float32)


def test_cnn_50mer():
    """Smoke test for CNN-50mer predictor."""
    print("\n=== CNN-50mer smoke test ===")
    from models.predictors.cnn_50mer import CNN50merPredictor

    seqs, labels = _gen_synth_data(n=200, seq_len=50, seed=0)
    val_seqs, val_labels = _gen_synth_data(n=50, seq_len=50, seed=1)
    test_seqs, test_labels = _gen_synth_data(n=50, seq_len=50, seed=2)

    predictor = CNN50merPredictor(
        name="cnn_smoke",
        dataset_name="synth",
        fold_idx=0,
        seed=42,
        device="cpu",
        hyperparams={
            "max_len": 50,
            "n_epochs": 2,
            "batch_size": 64,
            "n_mc_samples": 5,
            "device": "cpu",
            "num_workers": 0,
        },
    )

    t0 = time.time()
    history = predictor.fit(seqs, labels, val_seqs, val_labels)
    print(f"  fit time: {time.time() - t0:.2f}s")
    print(f"  train_loss: {history['train_loss']}")
    print(f"  val_pearson: {history['val_pearson']}")

    # Predict
    t0 = time.time()
    preds = predictor.predict(test_seqs)
    print(f"  predict time: {time.time() - t0:.2f}s")
    print(f"  preds shape: {preds.shape}, mean={preds.mean():.3f}, std={preds.std():.3f}")
    assert preds.shape == (50,), f"Expected (50,), got {preds.shape}"

    # Predict with uncertainty
    t0 = time.time()
    result = predictor.predict_with_uncertainty(test_seqs, n_mc_samples=5)
    print(f"  predict_with_uncertainty time: {time.time() - t0:.2f}s")
    print(f"  mean shape: {result.mean.shape}, std shape: {result.std.shape}")
    print(f"  std mean={result.std.mean():.4f} (should be > 0)")
    assert result.std.mean() > 0, "Std should be > 0 with MC dropout"

    # Save / load roundtrip
    ckpt_path = Path("/tmp/_smoke_cnn_ckpt")
    predictor.save(ckpt_path)
    print(f"  saved to {ckpt_path}.pt")

    loaded = CNN50merPredictor.load(ckpt_path, device="cpu")
    print(f"  loaded: {loaded}")
    # Verify y_mean/y_std persisted
    print(f"  y_mean={loaded._y_mean:.3f}, y_std={loaded._y_std:.3f}")
    assert loaded._y_mean != 0.0 or loaded._y_std != 1.0, "Standardization params not persisted"
    # Verify predict still works
    preds2 = loaded.predict(test_seqs[:5])
    print(f"  loaded predict shape: {preds2.shape}")

    print("  PASSED")


def test_transformer_utr():
    """Smoke test for Transformer-UTR predictor."""
    print("\n=== Transformer-UTR smoke test ===")
    from models.predictors.transformer_utr import TransformerUTRPredictor

    seqs, labels = _gen_synth_data(n=200, seq_len=50, seed=0)
    val_seqs, val_labels = _gen_synth_data(n=50, seq_len=50, seed=1)
    test_seqs, test_labels = _gen_synth_data(n=50, seq_len=50, seed=2)

    predictor = TransformerUTRPredictor(
        name="transformer_smoke",
        dataset_name="synth",
        fold_idx=0,
        seed=42,
        device="cpu",
        hyperparams={
            "max_len": 50,
            "n_epochs": 2,
            "batch_size": 64,
            "warmup_steps": 10,  # small for smoke test
            "device": "cpu",
            "num_workers": 0,
        },
    )

    t0 = time.time()
    history = predictor.fit(seqs, labels, val_seqs, val_labels)
    print(f"  fit time: {time.time() - t0:.2f}s")
    print(f"  train_loss: {history['train_loss']}")
    print(f"  val_pearson: {history['val_pearson']}")

    # Predict
    t0 = time.time()
    preds = predictor.predict(test_seqs)
    print(f"  predict time: {time.time() - t0:.2f}s")
    print(f"  preds shape: {preds.shape}, mean={preds.mean():.3f}")
    assert preds.shape == (50,), f"Expected (50,), got {preds.shape}"

    # Predict with uncertainty
    t0 = time.time()
    result = predictor.predict_with_uncertainty(test_seqs)
    print(f"  predict_with_uncertainty time: {time.time() - t0:.2f}s")
    print(f"  mean shape: {result.mean.shape}, std shape: {result.std.shape}")
    print(f"  std mean={result.std.mean():.4f} (should be >= 0)")
    assert (result.std >= 0).all(), "Std should be non-negative"

    print("  PASSED")


def test_uncertainty():
    """Smoke test for uncertainty metrics."""
    print("\n=== Uncertainty smoke test ===")
    from models.predictors.uncertainty import (
        expected_calibration_error,
        pinball_loss,
        knn_applicability_domain,
        coverage_accuracy_curve,
    )

    rng = np.random.default_rng(0)
    n = 100
    y_true = rng.normal(0, 1, n)
    y_pred = y_true + rng.normal(0, 0.5, n)  # decent predictions
    y_std = np.abs(rng.normal(0.5, 0.1, n))  # predicted std

    ece = expected_calibration_error(y_true, y_pred, y_std, n_bins=5)
    print(f"  ECE: {ece.ece:.4f}")
    print(f"  reliability_bins: {ece.reliability_bins}")
    print(f"  reliability_empirical: {ece.reliability_empirical}")
    assert ece.ece >= 0

    pl = pinball_loss(y_true, y_pred, y_std)
    print(f"  Pinball loss: {pl.pinball_loss}")
    print(f"  Quantiles: {pl.quantiles}")
    assert pl.pinball_loss.shape == (5,)

    # Applicability domain — use y_true/y_pred subset matching test_emb size
    train_emb = rng.normal(0, 1, (200, 16)).astype(np.float32)
    test_emb = rng.normal(0, 1, (50, 16)).astype(np.float32)
    ad = knn_applicability_domain(train_emb, test_emb, k=10, threshold_percentile=95.0)
    print(f"  AD threshold: {ad.threshold:.4f}")
    print(f"  AD fraction abstained: {ad.fraction_abstained:.4f}")
    assert 0 <= ad.fraction_abstained <= 1

    # Use first 50 samples of y_true/y_pred to match ad.distances shape
    curve = coverage_accuracy_curve(y_true[:50], y_pred[:50], ad.distances, n_points=10)
    print(f"  Coverage range: {curve['coverage'].min():.3f} -> {curve['coverage'].max():.3f}")
    assert len(curve["coverage"]) == 10

    print("  PASSED")


def test_ensemble():
    """Smoke test for ensemble aggregation (without loading real checkpoints)."""
    print("\n=== Ensemble smoke test ===")
    from models.predictors.ensemble import EnsembleConfig, PredictorEnsemble

    # Just test the config and instantiation (no actual checkpoints to load)
    config = EnsembleConfig(
        arch_dataset_pairs=[("cnn_50mer", "synth")],
        ckpt_dir=Path("/tmp/nonexistent_ckpts"),
        predictor_cls_map={},
        n_folds=5,
        n_seeds=3,
    )
    ensemble = PredictorEnsemble(config)
    print(f"  Ensemble initialized: {len(ensemble._models)} models (expected 0)")
    assert len(ensemble._models) == 0

    # Test that load_all raises FileNotFoundError for nonexistent dir
    try:
        ensemble.load_all()
        raise AssertionError("Should have raised FileNotFoundError")
    except FileNotFoundError as e:
        print(f"  Correctly raised FileNotFoundError: {str(e)[:60]}...")

    print("  PASSED")


def main():
    print("=" * 60)
    print("P1-04 Smoke Test")
    print("=" * 60)

    test_cnn_50mer()
    test_transformer_utr()
    test_uncertainty()
    test_ensemble()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
