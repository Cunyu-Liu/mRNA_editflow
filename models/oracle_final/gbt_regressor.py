"""P1-05 Oracle #3: LightGBM GBT regressor as independent final oracle.

This oracle is INDEPENDENT from the P1-04 training teacher:
    - Different architecture (GBT vs CNN/Transformer)
    - Different feature space (hand-engineered vs one-hot/embedding)
    - Different training data (Leplek 2022 PERSIST-Seq + held-out Sample 2019)
    - Frozen & sealed before any RL training begins

The oracle supports:
    - Standard regression (point prediction)
    - Quantile regression (for uncertainty via prediction intervals)
    - Feature importance (for interpretability)
    - Applicability domain (via leaf embedding k-NN distance)

Lock & seal procedure (in lock_oracle.py):
    1. Train oracle on Leplek 2022 + held-out Sample 2019 train split
    2. Compute SHA-256 of (model + feature extractor + data record IDs)
    3. Encrypt test labels with one-way hash
    4. chmod 444 on all artifact files
    5. Sign manifest with project key
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from .feature_extractor import (
        HandEngineeredFeatureExtractor,
        FeatureExtractorConfig,
    )
except ImportError:
    from feature_extractor import (  # type: ignore
        HandEngineeredFeatureExtractor,
        FeatureExtractorConfig,
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GBTOracleHyperparams:
    """Hyperparameters for LightGBM GBT oracle.

    Defaults follow standard regression settings; tuned for ~10-100k samples
    with ~600 features.
    """
    # LightGBM params
    objective: str = "regression"          # or "quantile" for intervals
    n_estimators: int = 1000
    learning_rate: float = 0.05
    num_leaves: int = 63
    max_depth: int = -1                    # -1 = no limit
    min_child_samples: int = 20
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    random_state: int = 42
    n_jobs: int = -1
    # Quantile regression (for uncertainty)
    quantile_alphas: Tuple[float, ...] = (0.1, 0.5, 0.9)
    # Early stopping
    early_stopping_rounds: int = 50
    # Feature extraction
    feature_config: Optional[FeatureExtractorConfig] = None
    # Log transform
    log_transform: bool = True             # log1p(label) before training


@dataclass
class GBTOracleResult:
    """Container for GBT oracle prediction.

    Attributes:
        mean: predicted mean (N,)
        std: predicted std (N,) — from quantile range or per-model variance
        quantiles: dict of quantile -> (N,) array (if quantile regression)
        leaf_embedding: (N, n_leaves) leaf activation (for applicability domain)
        metadata: extra info
    """
    mean: np.ndarray
    std: Optional[np.ndarray] = None
    quantiles: Optional[Dict[float, np.ndarray]] = None
    leaf_embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------

class GBTOracle:
    """Independent final oracle using LightGBM GBT.

    This oracle is FROZEN after training and must not be retrained on any
    data that includes test labels. Lock & seal procedure in lock_oracle.py.
    """

    def __init__(
        self,
        hyperparams: Optional[GBTOracleHyperparams] = None,
        feature_extractor: Optional[HandEngineeredFeatureExtractor] = None,
    ) -> None:
        self.hp = hyperparams or GBTOracleHyperparams()
        self.feature_extractor = feature_extractor or HandEngineeredFeatureExtractor(
            self.hp.feature_config
        )
        self._models: Dict[str, Any] = {}  # "mean" / "q0.1" / "q0.5" / "q0.9"
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._fitted = False
        self._feature_names: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        sequences: Sequence[str],
        labels: Sequence[float],
        val_sequences: Optional[Sequence[str]] = None,
        val_labels: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """Train the GBT oracle.

        Trains one model for mean prediction and (optionally) three quantile
        models for uncertainty estimation.

        Args:
            sequences: training sequences
            labels: training labels
            val_sequences: optional validation sequences for early stopping
            val_labels: optional validation labels

        Returns:
            Dict with training metrics
        """
        import lightgbm as lgb

        print(f"[GBTOracle] Extracting features for {len(sequences)} train sequences...")
        t0 = time.time()
        X_train = self.feature_extractor.extract_batch(sequences)
        y_train = np.asarray(labels, dtype=np.float32)
        if self.hp.log_transform:
            y_train_t = np.log1p(np.maximum(y_train, 0))
        else:
            y_train_t = y_train.copy()
        self._y_mean = float(y_train_t.mean())
        self._y_std = float(y_train_t.std() + 1e-8)
        y_train_norm = (y_train_t - self._y_mean) / self._y_std
        self._feature_names = self.feature_extractor.feature_names()
        print(f"  features: {X_train.shape}, extraction time: {time.time() - t0:.1f}s")

        X_val = None
        y_val_norm = None
        if val_sequences is not None and val_labels is not None:
            X_val = self.feature_extractor.extract_batch(val_sequences)
            y_val_arr = np.asarray(val_labels, dtype=np.float32)
            if self.hp.log_transform:
                y_val_t = np.log1p(np.maximum(y_val_arr, 0))
            else:
                y_val_t = y_val_arr.copy()
            y_val_norm = (y_val_t - self._y_mean) / self._y_std

        # Train mean model
        print(f"[GBTOracle] Training mean model (n_estimators={self.hp.n_estimators})...")
        t0 = time.time()
        mean_params = self._base_lgb_params()
        mean_params["objective"] = "regression"
        mean_model = lgb.LGBMRegressor(**mean_params)
        if X_val is not None:
            mean_model.fit(
                X_train, y_train_norm,
                eval_set=[(X_val, y_val_norm)],
                eval_metric="l2",
                callbacks=[lgb.early_stopping(self.hp.early_stopping_rounds, verbose=False)],
            )
        else:
            mean_model.fit(X_train, y_train_norm)
        self._models["mean"] = mean_model
        best_iter_mean = mean_model.best_iteration_ if hasattr(mean_model, "best_iteration_") else self.hp.n_estimators
        print(f"  mean model trained in {time.time() - t0:.1f}s, best_iter={best_iter_mean}")

        # Train quantile models
        if self.hp.objective == "quantile":
            for alpha in self.hp.quantile_alphas:
                print(f"[GBTOracle] Training quantile model (alpha={alpha})...")
                t0 = time.time()
                q_params = self._base_lgb_params()
                q_params["objective"] = "quantile"
                q_params["alpha"] = alpha
                q_model = lgb.LGBMRegressor(**q_params)
                if X_val is not None:
                    q_model.fit(
                        X_train, y_train_norm,
                        eval_set=[(X_val, y_val_norm)],
                        eval_metric="quantile",
                        callbacks=[lgb.early_stopping(self.hp.early_stopping_rounds, verbose=False)],
                    )
                else:
                    q_model.fit(X_train, y_train_norm)
                self._models[f"q{alpha}"] = q_model
                print(f"  quantile model alpha={alpha} trained in {time.time() - t0:.1f}s")

        # Compute train metrics
        train_pred = mean_model.predict(X_train)
        from scipy.stats import pearsonr, spearmanr
        train_metrics = {
            "train_pearson": float(pearsonr(train_pred, y_train_norm)[0]),
            "train_spearman": float(spearmanr(train_pred, y_train_norm)[0]),
            "train_mae": float(np.abs(train_pred - y_train_norm).mean()),
            "train_rmse": float(np.sqrt(((train_pred - y_train_norm) ** 2).mean())),
            "n_train": len(sequences),
            "n_features": X_train.shape[1],
            "best_iter_mean": int(best_iter_mean),
        }
        if X_val is not None:
            val_pred = mean_model.predict(X_val)
            train_metrics.update({
                "val_pearson": float(pearsonr(val_pred, y_val_norm)[0]),
                "val_spearman": float(spearmanr(val_pred, y_val_norm)[0]),
                "val_mae": float(np.abs(val_pred - y_val_norm).mean()),
                "val_rmse": float(np.sqrt(((val_pred - y_val_norm) ** 2).mean())),
                "n_val": len(val_sequences),
            })

        self._fitted = True
        return train_metrics

    def _base_lgb_params(self) -> Dict[str, Any]:
        return {
            "n_estimators": self.hp.n_estimators,
            "learning_rate": self.hp.learning_rate,
            "num_leaves": self.hp.num_leaves,
            "max_depth": self.hp.max_depth,
            "min_child_samples": self.hp.min_child_samples,
            "subsample": self.hp.subsample,
            "subsample_freq": self.hp.subsample_freq,
            "colsample_bytree": self.hp.colsample_bytree,
            "reg_alpha": self.hp.reg_alpha,
            "reg_lambda": self.hp.reg_lambda,
            "random_state": self.hp.random_state,
            "n_jobs": self.hp.n_jobs,
            "verbose": -1,
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, sequences: Sequence[str]) -> np.ndarray:
        """Point prediction (mean)."""
        if not self._fitted:
            raise RuntimeError("GBTOracle not fitted")
        X = self.feature_extractor.extract_batch(sequences)
        pred_norm = self._models["mean"].predict(X)
        return self._inverse_transform(pred_norm)

    def predict_with_uncertainty(
        self, sequences: Sequence[str],
    ) -> GBTOracleResult:
        """Predict mean + uncertainty (via quantile range).

        If quantile models are available:
            mean = q0.5 prediction
            std = (q0.9 - q0.1) / (2 * 1.645)  # 80% PI → std assuming Gaussian
        Else:
            mean = mean model prediction
            std = 0 (no uncertainty)
        """
        if not self._fitted:
            raise RuntimeError("GBTOracle not fitted")
        X = self.feature_extractor.extract_batch(sequences)

        # Mean prediction
        mean_norm = self._models["mean"].predict(X)
        mean_orig = self._inverse_transform(mean_norm)

        # Quantile predictions
        quantile_preds: Dict[float, np.ndarray] = {}
        if self.hp.objective == "quantile":
            for alpha in self.hp.quantile_alphas:
                pred_norm = self._models[f"q{alpha}"].predict(X)
                quantile_preds[alpha] = self._inverse_transform(pred_norm)

            # Compute std from 80% prediction interval
            if 0.1 in quantile_preds and 0.9 in quantile_preds:
                q10 = quantile_preds[0.1]
                q90 = quantile_preds[0.9]
                # For Gaussian, 80% PI = mean ± 1.2816 * std
                # So std ≈ (q90 - q10) / (2 * 1.2816)
                std = (q90 - q10) / (2 * 1.2816)
                std = np.maximum(std, 1e-6)  # avoid zero std
            else:
                std = np.zeros_like(mean_orig)
        else:
            std = np.zeros_like(mean_orig)

        # Leaf embedding for applicability domain
        leaf_embedding = self._models["mean"].predict(X, pred_leaf=True)

        return GBTOracleResult(
            mean=mean_orig.astype(np.float32),
            std=std.astype(np.float32),
            quantiles={a: p.astype(np.float32) for a, p in quantile_preds.items()} if quantile_preds else None,
            leaf_embedding=leaf_embedding,
            metadata={
                "method": "lgb_quantile" if self.hp.objective == "quantile" else "lgb_mean",
                "n_features": X.shape[1],
                "y_mean": self._y_mean,
                "y_std": self._y_std,
            },
        )

    def _inverse_transform(self, y_norm: np.ndarray) -> np.ndarray:
        if self.hp.log_transform:
            return np.expm1(y_norm * self._y_std + self._y_mean)
        return y_norm * self._y_std + self._y_mean

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(
        self, importance_type: str = "gain", top_k: int = 50,
    ) -> List[Tuple[str, float]]:
        """Return top-k feature importances from mean model.

        Args:
            importance_type: "gain" or "split"
            top_k: number of top features to return

        Returns:
            List of (feature_name, importance_score) tuples
        """
        if not self._fitted or "mean" not in self._models:
            raise RuntimeError("Mean model not fitted")
        model = self._models["mean"]
        importances = model.booster_.feature_importance(importance_type=importance_type)
        names = self._feature_names or [f"f{i}" for i in range(len(importances))]
        pairs = list(zip(names, importances))
        pairs.sort(key=lambda p: -p[1])
        return pairs[:top_k]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, ckpt_dir: Path) -> None:
        """Save oracle to ckpt_dir.

        Layout:
            ckpt_dir/
                mean_model.txt           # LightGBM model (text format)
                q0.1_model.txt           # quantile models (if any)
                q0.5_model.txt
                q0.9_model.txt
                feature_extractor.json   # feature extractor config
                oracle_meta.json         # all metadata
        """
        import lightgbm as lgb

        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save models
        for name, model in self._models.items():
            model.booster_.save_model(str(ckpt_dir / f"{name}_model.txt"))

        # Save metadata
        meta = {
            "hyperparams": self.hp.__dict__,
            "y_mean": self._y_mean,
            "y_std": self._y_std,
            "fitted": self._fitted,
            "feature_names": self._feature_names,
            "model_names": list(self._models.keys()),
            "save_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        # Convert non-serializable items
        hp_dict = dict(self.hp.__dict__)
        if hp_dict.get("feature_config") is not None:
            hp_dict["feature_config"] = hp_dict["feature_config"].__dict__
            hp_dict["feature_config"]["window_sizes"] = list(hp_dict["feature_config"]["window_sizes"])
        hp_dict["quantile_alphas"] = list(hp_dict["quantile_alphas"])
        meta["hyperparams"] = hp_dict
        with open(ckpt_dir / "oracle_meta.json", "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, ckpt_dir: Path) -> "GBTOracle":
        """Load oracle from ckpt_dir."""
        import lightgbm as lgb

        ckpt_dir = Path(ckpt_dir)
        with open(ckpt_dir / "oracle_meta.json") as f:
            meta = json.load(f)

        # Reconstruct hyperparams
        hp_dict = meta["hyperparams"]
        feature_config = None
        if hp_dict.get("feature_config"):
            fc_dict = hp_dict["feature_config"]
            fc_dict["window_sizes"] = tuple(fc_dict["window_sizes"])
            feature_config = FeatureExtractorConfig(**fc_dict)
        hp_dict["feature_config"] = feature_config
        hp_dict["quantile_alphas"] = tuple(hp_dict["quantile_alphas"])
        hp = GBTOracleHyperparams(**hp_dict)

        feature_extractor = HandEngineeredFeatureExtractor(feature_config)
        oracle = cls(hyperparams=hp, feature_extractor=feature_extractor)
        oracle._y_mean = meta["y_mean"]
        oracle._y_std = meta["y_std"]
        oracle._fitted = meta["fitted"]
        oracle._feature_names = meta["feature_names"]

        # Load models
        for name in meta["model_names"]:
            model_path = ckpt_dir / f"{name}_model.txt"
            if model_path.exists():
                booster = lgb.Booster(model_file=str(model_path))
                # Wrap in LGBMRegressor-like interface
                # Actually, we need to use the booster directly for predict
                oracle._models[name] = _BoosterWrapper(booster)

        return oracle


class _BoosterWrapper:
    """Wrapper to make LightGBM Booster behave like LGBMRegressor for predict."""

    def __init__(self, booster: Any) -> None:
        self.booster_ = booster
        self.best_iteration_ = booster.current_iteration()

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        return self.booster_.predict(X, **kwargs)

    @property
    def n_estimators_(self) -> int:
        return self.booster_.num_trees()


__all__ = [
    "GBTOracleHyperparams",
    "GBTOracleResult",
    "GBTOracle",
]
