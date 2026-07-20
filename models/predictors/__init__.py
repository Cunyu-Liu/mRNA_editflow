"""P1-04 Cross-fitted predictor ensemble.

Submodules:
    base: PredictorBase interface (fit, predict, predict_with_uncertainty, save, load)
    data_loaders: Unified loader for Sample 2019 / Cao 2021 / Saluki / CodonBERT
    crossfit: k-fold cross-fitting harness
    cnn_50mer: Architecture A (Optimus100K-style)
    transformer_utr: Architecture B (UTR-LM-style)
    ensemble: Deep ensemble aggregation
    uncertainty: ECE, reliability, applicability domain

See docs/p1_04_predictor_ensemble_design.md for full design.
"""
from .base import PredictorBase, PredictionResult  # noqa: F401
from .cnn_50mer import CNN50merPredictor, CNN50merHyperparams  # noqa: F401
from .transformer_utr import TransformerUTRPredictor, TransformerUTRHyperparams  # noqa: F401
from .ensemble import (  # noqa: F401
    EnsembleConfig,
    EnsemblePrediction,
    PredictorEnsemble,
    build_default_ensemble_config,
)
from .uncertainty import (  # noqa: F401
    CalibrationResult,
    ApplicabilityDomainResult,
    expected_calibration_error,
    pinball_loss,
    knn_applicability_domain,
    coverage_accuracy_curve,
    full_uncertainty_report,
)
