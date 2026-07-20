"""P1-04: Base predictor interface.

All predictors in the cross-fitted ensemble implement this interface so that
the cross-fitting harness can train and evaluate them uniformly.

Design principles:
    - Deterministic given (architecture, dataset, fold, seed)
    - Test split is never seen during fit()
    - predict_with_uncertainty returns both mean and std
    - save/load roundtrip preserves all hyperparameters
"""
from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PredictionResult:
    """Container for predictor output with optional uncertainty.

    Attributes:
        mean: Predicted mean (N,) float array
        std: Predicted std (N,) float array; None if not computed
        abstain_mask: Boolean (N,) array; True = abstained (out of applicability domain)
        metadata: Optional dict with per-sample diagnostics
    """

    mean: np.ndarray
    std: Optional[np.ndarray] = None
    abstain_mask: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float32)
        if self.std is not None:
            self.std = np.asarray(self.std, dtype=np.float32)
        if self.abstain_mask is not None:
            self.abstain_mask = np.asarray(self.abstain_mask, dtype=bool)


class PredictorBase(abc.ABC):
    """Abstract base class for all cross-fitted predictors.

    Subclasses must implement:
        - _build_model(): construct the underlying torch model
        - fit(X_train, y_train, X_val, y_val): train on one fold
        - predict(X): return mean predictions
        - predict_with_uncertainty(X): return mean + std (via MC dropout or deep ensemble)

    The cross-fitting harness (crossfit.py) handles:
        - k-fold split
        - per-fold fit + held-out predict
        - deep ensemble (M seeds per fold)
        - aggregation
    """

    def __init__(
        self,
        name: str,
        dataset_name: str,
        fold_idx: int,
        seed: int = 42,
        device: str = "cuda",
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.dataset_name = dataset_name
        self.fold_idx = fold_idx
        self.seed = seed
        self.device = device
        self.hyperparams: Dict[str, Any] = hyperparams or {}
        self._fitted = False
        self._model: Any = None

    # ------------------------------------------------------------------
    # Abstract methods (must be implemented by subclasses)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _build_model(self) -> Any:
        """Construct the underlying torch model. Store in self._model."""

    @abc.abstractmethod
    def fit(
        self,
        X_train: Sequence,
        y_train: Sequence,
        X_val: Optional[Sequence] = None,
        y_val: Optional[Sequence] = None,
    ) -> Dict[str, List[float]]:
        """Train on one fold. Returns per-epoch metrics dict.

        Args:
            X_train, y_train: Training data (already split by crossfit harness)
            X_val, y_val: Optional validation data for early stopping

        Returns:
            {"train_loss": [...], "val_loss": [...], "val_pearson": [...]}
        """

    @abc.abstractmethod
    def predict(self, X: Sequence) -> np.ndarray:
        """Return mean predictions (N,) float array."""

    @abc.abstractmethod
    def predict_with_uncertainty(
        self, X: Sequence, n_mc_samples: int = 30
    ) -> PredictionResult:
        """Return mean + std via MC dropout or deep ensemble."""

    # ------------------------------------------------------------------
    # Shared methods
    # ------------------------------------------------------------------

    def fit_predict(
        self,
        X_train: Sequence,
        y_train: Sequence,
        X_heldout: Sequence,
        X_val: Optional[Sequence] = None,
        y_val: Optional[Sequence] = None,
    ) -> PredictionResult:
        """Convenience: fit then predict on held-out fold.

        This is the canonical cross-fitting call pattern.
        """
        if self._fitted:
            raise RuntimeError(f"Predictor {self.name} already fitted")
        metrics = self.fit(X_train, y_train, X_val, y_val)
        result = self.predict_with_uncertainty(X_heldout)
        result.metadata["fit_metrics"] = metrics
        result.metadata["predictor_name"] = self.name
        result.metadata["dataset_name"] = self.dataset_name
        result.metadata["fold_idx"] = self.fold_idx
        result.metadata["seed"] = self.seed
        self._fitted = True
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, ckpt_path: Path) -> None:
        """Save model weights + metadata to ckpt_path.

        Layout:
            ckpt_path.pt: torch state dict
            ckpt_path.json: metadata (name, dataset, fold, seed, hyperparams)
        """
        import torch

        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self._model.state_dict() if self._model else None,
                "name": self.name,
                "dataset_name": self.dataset_name,
                "fold_idx": self.fold_idx,
                "seed": self.seed,
                "hyperparams": self.hyperparams,
                "_fitted": self._fitted,
            },
            str(ckpt_path) + ".pt",
        )
        with open(str(ckpt_path) + ".json", "w") as f:
            json.dump(
                {
                    "name": self.name,
                    "dataset_name": self.dataset_name,
                    "fold_idx": self.fold_idx,
                    "seed": self.seed,
                    "hyperparams": self.hyperparams,
                    "ckpt_file": str(ckpt_path) + ".pt",
                    "fitted": self._fitted,
                },
                f,
                indent=2,
                sort_keys=True,
            )

    @classmethod
    def load(cls, ckpt_path: Path, device: str = "cuda") -> "PredictorBase":
        """Load a fitted predictor from disk."""
        import torch

        ckpt_path = Path(ckpt_path)
        with open(str(ckpt_path) + ".json", "r") as f:
            meta = json.load(f)
        instance = cls(
            name=meta["name"],
            dataset_name=meta["dataset_name"],
            fold_idx=meta["fold_idx"],
            seed=meta["seed"],
            device=device,
            hyperparams=meta["hyperparams"],
        )
        state = torch.load(str(ckpt_path) + ".pt", map_location=device)
        instance._build_model()
        instance._model.load_state_dict(state["model_state_dict"])
        instance._model.to(device)
        instance._fitted = True
        return instance

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"dataset={self.dataset_name!r}, "
            f"fold={self.fold_idx}, "
            f"seed={self.seed})"
        )

    def get_config(self) -> Dict[str, Any]:
        """Return predictor config as a serializable dict."""
        return {
            "class": self.__class__.__name__,
            "name": self.name,
            "dataset_name": self.dataset_name,
            "fold_idx": self.fold_idx,
            "seed": self.seed,
            "device": self.device,
            "hyperparams": self.hyperparams,
            "fitted": self._fitted,
        }
