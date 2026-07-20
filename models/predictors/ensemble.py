"""P1-04: Deep ensemble aggregation for cross-fitted predictors.

Loads all checkpoints from a cross-fitting run (k folds × M seeds) and
aggregates their predictions into a single ensemble prediction with
proper epistemic uncertainty.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np

try:
    from .base import PredictorBase, PredictionResult
except ImportError:
    from base import PredictorBase, PredictionResult  # type: ignore


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

@dataclass
class EnsembleConfig:
    """Configuration for ensemble aggregation.

    Attributes:
        arch_dataset_pairs: list of (arch_name, dataset_name) tuples to include
        ckpt_dir: directory containing per-fold checkpoints
        predictor_cls_map: dict mapping arch_name -> PredictorBase subclass
        n_folds: number of folds per pair
        n_seeds: number of seeds per fold
    """
    arch_dataset_pairs: List[Tuple[str, str]]
    ckpt_dir: Path = Path("ckpts/p1_04_predictors")
    predictor_cls_map: Dict[str, Type[PredictorBase]] = field(default_factory=dict)
    n_folds: int = 5
    n_seeds: int = 3
    base_seed: int = 42
    device: str = "cuda"


@dataclass
class EnsemblePrediction:
    """Result of ensemble prediction.

    Attributes:
        mean: ensemble mean (N,)
        std: ensemble std (N,) - epistemic uncertainty
        per_model_means: list of (model_name, mean array) for diagnostics
        abstain_mask: optional applicability-domain mask
    """
    mean: np.ndarray
    std: np.ndarray
    per_model_means: List[Tuple[str, np.ndarray]] = field(default_factory=list)
    abstain_mask: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PredictorEnsemble:
    """Deep ensemble of cross-fitted predictors.

    Loads all (arch × dataset × fold × seed) checkpoints and aggregates
    their predictions via mean + std.
    """

    def __init__(self, config: EnsembleConfig) -> None:
        self.config = config
        self._models: List[PredictorBase] = []
        self._model_names: List[str] = []

    def load_all(self) -> int:
        """Load all checkpoints matching the config.

        Returns:
            Number of models loaded.
        """
        ckpt_dir = Path(self.config.ckpt_dir)
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

        loaded = 0
        for arch_name, dataset_name in self.config.arch_dataset_pairs:
            cls = self.config.predictor_cls_map.get(arch_name)
            if cls is None:
                raise KeyError(f"No predictor class registered for arch '{arch_name}'")

            for fold_idx in range(self.config.n_folds):
                for seed_offset in range(self.config.n_seeds):
                    seed = self.config.base_seed + seed_offset
                    name = f"{arch_name}__{dataset_name}__fold{fold_idx}_seed{seed}"
                    ckpt_path = ckpt_dir / name
                    if not Path(str(ckpt_path) + ".pt").exists():
                        continue
                    try:
                        model = cls.load(ckpt_path, device=self.config.device)
                        self._models.append(model)
                        self._model_names.append(name)
                        loaded += 1
                    except Exception as e:
                        print(f"  WARN: failed to load {name}: {e}")
        return loaded

    def predict(
        self, X: Sequence[str], return_per_model: bool = False,
    ) -> EnsemblePrediction:
        """Aggregate predictions from all ensemble members.

        Args:
            X: input sequences
            return_per_model: if True, store per-model means in result

        Returns:
            EnsemblePrediction with mean + std across all models
        """
        if not self._models:
            raise RuntimeError("No models loaded. Call load_all() first.")

        per_model_means: List[Tuple[str, np.ndarray]] = []
        all_preds: List[np.ndarray] = []

        for name, model in zip(self._model_names, self._models):
            try:
                pred = model.predict(X)
                per_model_means.append((name, pred))
                all_preds.append(pred)
            except Exception as e:
                print(f"  WARN: predict failed for {name}: {e}")

        if not all_preds:
            raise RuntimeError("All ensemble predictions failed")

        # Stack and aggregate
        stacked = np.stack(all_preds, axis=0)  # (M, N)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)

        return EnsemblePrediction(
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            per_model_means=per_model_means if return_per_model else [],
            abstain_mask=None,
            metadata={
                "n_models": len(all_preds),
                "model_names": list(self._model_names),
            },
        )

    def predict_with_uncertainty(
        self, X: Sequence[str], n_mc_samples: int = 0,
    ) -> EnsemblePrediction:
        """Predict with full uncertainty (epistemic + aleatoric).

        For each model, calls predict_with_uncertainty to get mu + sigma.
        Total variance = mean(sigma_i^2) + var(mu_i)  (law of total variance)

        Args:
            X: input sequences
            n_mc_samples: optional MC samples per model (0 = use model default)

        Returns:
            EnsemblePrediction with combined uncertainty
        """
        if not self._models:
            raise RuntimeError("No models loaded. Call load_all() first.")

        all_means: List[np.ndarray] = []
        all_stds: List[np.ndarray] = []

        for name, model in zip(self._model_names, self._models):
            try:
                if n_mc_samples > 0:
                    result = model.predict_with_uncertainty(X, n_mc_samples=n_mc_samples)
                else:
                    result = model.predict_with_uncertainty(X)
                all_means.append(result.mean)
                all_stds.append(result.std if result.std is not None else np.zeros_like(result.mean))
            except Exception as e:
                print(f"  WARN: predict_with_uncertainty failed for {name}: {e}")

        if not all_means:
            raise RuntimeError("All ensemble predictions failed")

        means = np.stack(all_means, axis=0)  # (M, N)
        stds = np.stack(all_stds, axis=0)    # (M, N)

        # Law of total variance:
        # Var(y) = E[Var(y|model)] + Var(E[y|model])
        #        = mean(sigma_i^2) + var(mu_i)
        ensemble_mean = means.mean(axis=0)
        aleatoric_var = (stds ** 2).mean(axis=0)
        epistemic_var = means.var(axis=0)
        total_std = np.sqrt(aleatoric_var + epistemic_var)

        return EnsemblePrediction(
            mean=ensemble_mean.astype(np.float32),
            std=total_std.astype(np.float32),
            per_model_means=[],
            abstain_mask=None,
            metadata={
                "n_models": len(all_means),
                "aleatoric_std": np.sqrt(aleatoric_var).astype(np.float32),
                "epistemic_std": np.sqrt(epistemic_var).astype(np.float32),
            },
        )


# ---------------------------------------------------------------------------
# Convenience: build default ensemble config
# ---------------------------------------------------------------------------

def build_default_ensemble_config(
    arch_dataset_pairs: Optional[List[Tuple[str, str]]] = None,
    ckpt_dir: Optional[Path] = None,
    device: str = "cuda",
) -> EnsembleConfig:
    """Build default ensemble config with both architectures registered.

    Args:
        arch_dataset_pairs: list of (arch, dataset) pairs; default = both archs × sample2019
        ckpt_dir: checkpoint directory
        device: cuda/cpu

    Returns:
        EnsembleConfig ready to use
    """
    # Lazy import to avoid circular deps
    try:
        from .cnn_50mer import CNN50merPredictor
        from .transformer_utr import TransformerUTRPredictor
    except ImportError:
        from cnn_50mer import CNN50merPredictor  # type: ignore
        from transformer_utr import TransformerUTRPredictor  # type: ignore

    if arch_dataset_pairs is None:
        arch_dataset_pairs = [
            ("cnn_50mer", "sample2019_mpra"),
            ("transformer_utr", "sample2019_mpra"),
        ]
    if ckpt_dir is None:
        ckpt_dir = Path("ckpts/p1_04_predictors")

    return EnsembleConfig(
        arch_dataset_pairs=arch_dataset_pairs,
        ckpt_dir=ckpt_dir,
        predictor_cls_map={
            "cnn_50mer": CNN50merPredictor,
            "transformer_utr": TransformerUTRPredictor,
        },
        device=device,
    )


__all__ = [
    "EnsembleConfig",
    "EnsemblePrediction",
    "PredictorEnsemble",
    "build_default_ensemble_config",
]
