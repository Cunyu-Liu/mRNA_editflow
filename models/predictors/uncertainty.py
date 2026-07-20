"""P1-04: Uncertainty, calibration, and applicability domain metrics.

Implements:
    - Expected Calibration Error (ECE) for regression
    - Reliability diagram data
    - Quantile-based calibration (Pinball loss)
    - k-NN applicability domain via embedding distance
    - Coverage-accuracy curve
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """Result of calibration analysis.

    Attributes:
        ece: Expected Calibration Error (regression)
        reliability_bins: bin centers (predicted std)
        reliability_empirical: per-bin empirical |y - mu|
        reliability_counts: per-bin sample counts
        pinball_loss: quantile loss at specified quantiles
        quantiles: quantile levels evaluated
    """
    ece: float
    reliability_bins: np.ndarray
    reliability_empirical: np.ndarray
    reliability_counts: np.ndarray
    pinball_loss: Optional[np.ndarray] = None
    quantiles: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 10,
) -> CalibrationResult:
    """Compute Expected Calibration Error (ECE) for regression.

    For a well-calibrated regression model with Gaussian assumption:
        |y - mu| should follow a folded normal with scale = sigma

    ECE = sum over bins (n_bin / N) * mean(|empirical_error - predicted_std|)

    Args:
        y_true: (N,) true labels
        y_pred: (N,) predicted means
        y_std: (N,) predicted standard deviations
        n_bins: number of equal-width bins over predicted std

    Returns:
        CalibrationResult with ECE and reliability diagram data
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_std = np.asarray(y_std, dtype=np.float64)
    abs_error = np.abs(y_true - y_pred)

    # Bin by predicted std (equal-width)
    std_min, std_max = float(y_std.min()), float(y_std.max())
    if std_max - std_min < 1e-12:
        # All same std — single bin
        bins = np.array([std_min])
        empirical = np.array([abs_error.mean()])
        counts = np.array([len(y_true)])
        ece_val = float(np.abs(empirical[0] - bins[0]))
    else:
        bin_edges = np.linspace(std_min, std_max, n_bins + 1)
        bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        empirical = np.zeros(n_bins)
        counts = np.zeros(n_bins, dtype=int)
        for i in range(n_bins):
            mask = (y_std >= bin_edges[i]) & (y_std < bin_edges[i + 1])
            if i == n_bins - 1:
                mask = (y_std >= bin_edges[i]) & (y_std <= bin_edges[i + 1])
            if mask.sum() > 0:
                empirical[i] = abs_error[mask].mean()
                counts[i] = mask.sum()
        # ECE = weighted mean of |empirical - predicted_std|
        N = counts.sum()
        if N > 0:
            ece_val = float(np.sum((counts / N) * np.abs(empirical - bins)))
        else:
            ece_val = 0.0
        # Filter empty bins
        nonempty = counts > 0
        bins = bins[nonempty]
        empirical = empirical[nonempty]
        counts = counts[nonempty]

    return CalibrationResult(
        ece=ece_val,
        reliability_bins=bins.astype(np.float32),
        reliability_empirical=empirical.astype(np.float32),
        reliability_counts=counts,
    )


def pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    quantiles: Optional[Sequence[float]] = None,
) -> CalibrationResult:
    """Compute pinball (quantile) loss assuming Gaussian predictions.

    For each quantile q, compute the Gaussian quantile:
        y_q = mu + sigma * Phi^{-1}(q)
    Then pinball loss = mean(max(q*(y - y_q), (q-1)*(y - y_q)))

    Args:
        y_true: (N,) true labels
        y_pred: (N,) predicted means
        y_std: (N,) predicted standard deviations
        quantiles: quantile levels to evaluate (default: [0.1, 0.25, 0.5, 0.75, 0.9])

    Returns:
        CalibrationResult with pinball losses
    """
    from scipy.stats import norm

    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantiles = np.asarray(quantiles, dtype=np.float64)

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_std = np.asarray(y_std, dtype=np.float64)

    losses = np.zeros(len(quantiles))
    for i, q in enumerate(quantiles):
        z_q = norm.ppf(q)
        y_q = y_pred + y_std * z_q
        diff = y_true - y_q
        loss = np.maximum(q * diff, (q - 1) * diff)
        losses[i] = float(loss.mean())

    # Also compute ECE for convenience
    ece_result = expected_calibration_error(y_true, y_pred, y_std)

    return CalibrationResult(
        ece=ece_result.ece,
        reliability_bins=ece_result.reliability_bins,
        reliability_empirical=ece_result.reliability_empirical,
        reliability_counts=ece_result.reliability_counts,
        pinball_loss=losses.astype(np.float32),
        quantiles=quantiles.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Applicability domain via k-NN
# ---------------------------------------------------------------------------

@dataclass
class ApplicabilityDomainResult:
    """Result of applicability domain analysis.

    Attributes:
        abstain_mask: (N,) bool; True = out of domain (abstain)
        distances: (N,) mean distance to k nearest neighbors in training set
        threshold: distance threshold for abstention
        fraction_abstained: fraction of samples flagged as out-of-domain
    """
    abstain_mask: np.ndarray
    distances: np.ndarray
    threshold: float
    fraction_abstained: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def knn_applicability_domain(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    k: int = 10,
    threshold_percentile: float = 95.0,
) -> ApplicabilityDomainResult:
    """Compute k-NN applicability domain.

    For each test sample, compute the mean cosine distance to its k nearest
    neighbors in the training set. Abstain if mean distance > threshold.

    Threshold is set at the `threshold_percentile` of train-to-train distances.

    Args:
        train_embeddings: (N_train, D) training embeddings
        test_embeddings: (N_test, D) test embeddings
        k: number of nearest neighbors
        threshold_percentile: percentile of train distances for threshold

    Returns:
        ApplicabilityDomainResult
    """
    train_embeddings = np.asarray(train_embeddings, dtype=np.float32)
    test_embeddings = np.asarray(test_embeddings, dtype=np.float32)

    # Normalize for cosine distance
    def _normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return x / norms

    train_n = _normalize(train_embeddings)
    test_n = _normalize(test_embeddings)

    def _mean_knn_distance(queries: np.ndarray, references: np.ndarray, k: int) -> np.ndarray:
        """Compute mean cosine distance to k nearest neighbors."""
        # cosine similarity = dot product of normalized vectors
        # distance = 1 - similarity
        sims = queries @ references.T  # (N_q, N_r)
        # For each query, find top-k highest similarities (smallest distances)
        # Use argpartition for efficiency
        n_ref = references.shape[0]
        k_eff = min(k, n_ref)
        # Get top-k similarities
        top_k_idx = np.argpartition(-sims, kth=k_eff - 1, axis=1)[:, :k_eff]
        top_k_sims = np.take_along_axis(sims, top_k_idx, axis=1)
        top_k_dists = 1.0 - top_k_sims
        return top_k_dists.mean(axis=1)

    # Compute train-to-train distances for threshold
    # Sample a subset if train is large (for efficiency)
    n_train = train_n.shape[0]
    if n_train > 5000:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(n_train, size=5000, replace=False)
        train_sample = train_n[sample_idx]
    else:
        train_sample = train_n

    train_dists = _mean_knn_distance(train_sample, train_n, k=k)
    threshold = float(np.percentile(train_dists, threshold_percentile))

    # Compute test-to-train distances
    test_dists = _mean_knn_distance(test_n, train_n, k=k)
    abstain_mask = test_dists > threshold
    fraction_abstained = float(abstain_mask.mean())

    return ApplicabilityDomainResult(
        abstain_mask=abstain_mask,
        distances=test_dists.astype(np.float32),
        threshold=threshold,
        fraction_abstained=fraction_abstained,
        metadata={
            "k": k,
            "threshold_percentile": threshold_percentile,
            "n_train": n_train,
            "n_test": test_embeddings.shape[0],
        },
    )


def coverage_accuracy_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    distances: np.ndarray,
    n_points: int = 20,
) -> Dict[str, np.ndarray]:
    """Compute coverage-accuracy curve by varying the distance threshold.

    Args:
        y_true: (N,) true labels
        y_pred: (N,) predicted means
        distances: (N,) distances to training set (from applicability domain)
        n_points: number of threshold points

    Returns:
        Dict with arrays: thresholds, coverage, pearson_r, mae
    """
    from scipy.stats import pearsonr

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)

    d_min, d_max = float(distances.min()), float(distances.max())
    thresholds = np.linspace(d_min, d_max, n_points)

    coverage = np.zeros(n_points)
    pearson_r_arr = np.zeros(n_points)
    mae_arr = np.zeros(n_points)

    for i, t in enumerate(thresholds):
        mask = distances <= t
        coverage[i] = mask.mean()
        if mask.sum() > 1:
            pearson_r_arr[i] = float(pearsonr(y_pred[mask], y_true[mask])[0])
            mae_arr[i] = float(np.abs(y_pred[mask] - y_true[mask]).mean())
        else:
            pearson_r_arr[i] = float("nan")
            mae_arr[i] = float("nan")

    return {
        "thresholds": thresholds.astype(np.float32),
        "coverage": coverage.astype(np.float32),
        "pearson_r": pearson_r_arr.astype(np.float32),
        "mae": mae_arr.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------------

def full_uncertainty_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    train_embeddings: Optional[np.ndarray] = None,
    test_embeddings: Optional[np.ndarray] = None,
    n_bins: int = 10,
    k: int = 10,
    threshold_percentile: float = 95.0,
) -> Dict[str, Any]:
    """Compute full uncertainty report.

    Args:
        y_true, y_pred, y_std: arrays
        train_embeddings, test_embeddings: optional, for applicability domain
        n_bins: ECE bins
        k: k-NN for applicability domain
        threshold_percentile: AD threshold

    Returns:
        Dict with calibration + applicability domain metrics
    """
    report: Dict[str, Any] = {}

    # Calibration
    cal = pinball_loss(y_true, y_pred, y_std)
    report["calibration"] = {
        "ece": cal.ece,
        "reliability_bins": cal.reliability_bins.tolist(),
        "reliability_empirical": cal.reliability_empirical.tolist(),
        "reliability_counts": cal.reliability_counts.tolist(),
        "pinball_loss": cal.pinball_loss.tolist() if cal.pinball_loss is not None else None,
        "quantiles": cal.quantiles.tolist() if cal.quantiles is not None else None,
    }

    # Applicability domain
    if train_embeddings is not None and test_embeddings is not None:
        ad = knn_applicability_domain(
            train_embeddings, test_embeddings, k=k,
            threshold_percentile=threshold_percentile,
        )
        curve = coverage_accuracy_curve(y_true, y_pred, ad.distances)
        report["applicability_domain"] = {
            "threshold": ad.threshold,
            "fraction_abstained": ad.fraction_abstained,
            "k": k,
            "coverage_curve": {
                "thresholds": curve["thresholds"].tolist(),
                "coverage": curve["coverage"].tolist(),
                "pearson_r": curve["pearson_r"].tolist(),
                "mae": curve["mae"].tolist(),
            },
        }
        # Metrics on non-abstained subset
        keep = ~ad.abstain_mask
        if keep.sum() > 1:
            from scipy.stats import pearsonr, spearmanr
            from sklearn.metrics import r2_score, mean_absolute_error
            report["non_abstained_metrics"] = {
                "n": int(keep.sum()),
                "pearson_r": float(pearsonr(y_pred[keep], y_true[keep])[0]),
                "spearman_r": float(spearmanr(y_pred[keep], y_true[keep])[0]),
                "r2": float(r2_score(y_true[keep], y_pred[keep])),
                "mae": float(mean_absolute_error(y_true[keep], y_pred[keep])),
            }

    return report


__all__ = [
    "CalibrationResult",
    "ApplicabilityDomainResult",
    "expected_calibration_error",
    "pinball_loss",
    "knn_applicability_domain",
    "coverage_accuracy_curve",
    "full_uncertainty_report",
]
