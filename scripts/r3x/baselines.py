"""Strong, information-set-matched baselines for the R3X ordinary route."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from .features import (
    CANDIDATE_KMER_FEATURE_SCHEMA,
    EDIT_ONLY_FEATURE_SCHEMA,
    PAIRED_KMER_FEATURE_SCHEMA,
    candidate_kmer_features,
    edit_only_features,
    matrix,
    paired_kmer_features,
)


DIRECT_LEADERBOARD = "DIRECT_DELTA_NO_MEASURED_ANCHOR"
ANCHOR_LEADERBOARD = "MEASURED_SOURCE_ANCHOR_ASSISTED"
RIDGE_SOLVER_POLICY = "SMALLER_PRIMAL_OR_DUAL_SQUARE_SYSTEM_V2"
NEAREST_NEIGHBOR_SEARCH_POLICY = (
    "EXACT_TWO_PASS_CHUNKED_DISTANCE_CANONICAL_FSUM_TIES_V2"
)
HGB_TRAINING_WEIGHT_POLICY = "FROZEN_CELL_THEN_SOURCE_COMPONENT_THEN_ROW_BALANCED"

FEATURE_SCHEMA_BY_FUNCTION = {
    edit_only_features: EDIT_ONLY_FEATURE_SCHEMA,
    candidate_kmer_features: CANDIDATE_KMER_FEATURE_SCHEMA,
    paired_kmer_features: PAIRED_KMER_FEATURE_SCHEMA,
}


class BaselineProtocolError(ValueError):
    """Raised when a baseline violates its declared information set."""


@dataclass(frozen=True)
class BaselineResult:
    name: str
    status: str
    information_set: str
    leaderboard: str
    predictions: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    reason: str = ""
    role: str = "PRIMARY_BASELINE"
    metadata: Mapping = field(default_factory=dict)


def _rows(rows: Sequence[Mapping], *, target: Optional[str] = None) -> list[Mapping]:
    out = list(rows)
    if not out:
        raise BaselineProtocolError("baseline refuses empty rows")
    if target is not None:
        for row in out:
            value = row.get(target)
            if value is None or not np.isfinite(float(value)):
                raise BaselineProtocolError(f"training rows require finite {target}")
    return out


def _components(rows: Sequence[Mapping]) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get("source_component_id")
        if not isinstance(value, str) or not value:
            raise BaselineProtocolError("every training row requires source_component_id")
        values.append(value)
    return np.asarray(values, dtype=object)


def _condition(row: Mapping) -> tuple[str, str, str]:
    endpoint_id = row.get("endpoint_id")
    endpoint_alias = row.get("endpoint")
    if endpoint_id not in (None, "") and endpoint_alias not in (None, ""):
        if str(endpoint_id) != str(endpoint_alias):
            raise BaselineProtocolError("endpoint aliases conflict")
    context_id = row.get("publication_context_id")
    context_alias = row.get("context_id_v2")
    if context_id not in (None, "") and context_alias not in (None, ""):
        if str(context_id) != str(context_alias):
            raise BaselineProtocolError("context aliases conflict")
    return (
        str(endpoint_id or endpoint_alias or "__UNK__"),
        str(context_id or context_alias or "__UNK__"),
        str(row.get("region") or "__UNK__"),
    )


def _condition_vocab(rows: Sequence[Mapping]) -> tuple[tuple[str, ...], ...]:
    conditions = [_condition(row) for row in rows]
    return tuple(tuple(sorted({condition[index] for condition in conditions})) for index in range(3))


def _condition_matrix(
    rows: Sequence[Mapping], vocab: tuple[tuple[str, ...], ...]
) -> np.ndarray:
    blocks = []
    for index, levels in enumerate(vocab):
        level_index = {level: position + 1 for position, level in enumerate(levels)}
        block = np.zeros((len(rows), len(levels) + 1), dtype=np.float64)
        for row_index, condition in enumerate(map(_condition, rows)):
            block[row_index, level_index.get(condition[index], 0)] = 1.0
        blocks.append(block)
    return np.concatenate(blocks, axis=1)


def _feature_matrix(
    rows: Sequence[Mapping], feature_fn: Callable, vocab: tuple[tuple[str, ...], ...]
) -> np.ndarray:
    return np.concatenate([matrix(rows, feature_fn), _condition_matrix(rows, vocab)], axis=1)


class _StandardizedRidge:
    def __init__(self, alpha: float):
        if not np.isfinite(alpha) or alpha <= 0:
            raise BaselineProtocolError("ridge alpha must be finite and positive")
        self.alpha = float(alpha)

    def fit(self, features: np.ndarray, target: np.ndarray):
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
            raise BaselineProtocolError("ridge features must be a non-empty matrix")
        if y.ndim != 1 or len(y) != len(x):
            raise BaselineProtocolError("ridge target must be a vector aligned to features")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise BaselineProtocolError("ridge features and target must be finite")
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        z = (x - self.mean_) / self.scale_
        self.y_mean_ = float(y.mean())
        yc = y - self.y_mean_

        # Solve in the smaller of sample space and feature space.  The prior
        # implementation always formed the n_train x n_train dual kernel,
        # which is infeasible for ordinary datasets with tens of thousands of
        # rows and only hundreds of deterministic k-mer features.
        n_rows, n_features = z.shape
        self.n_train_rows_ = int(n_rows)
        self.n_features_in_ = int(n_features)
        if n_features <= n_rows:
            self.solver_ = "PRIMAL_SMALLER_FEATURE_SYSTEM"
            self.square_system_dimension_ = int(n_features)
            system = z.T @ z + self.alpha * np.eye(n_features, dtype=np.float64)
            rhs = z.T @ yc
            try:
                self.coef_ = np.linalg.solve(system, rhs)
            except np.linalg.LinAlgError:
                self.coef_ = np.linalg.pinv(system) @ rhs
            return self

        self.solver_ = "DUAL_SMALLER_SAMPLE_SYSTEM"
        self.square_system_dimension_ = int(n_rows)
        kernel = z @ z.T
        system = kernel + self.alpha * np.eye(n_rows, dtype=np.float64)
        try:
            dual = np.linalg.solve(system, yc)
        except np.linalg.LinAlgError:
            dual = np.linalg.pinv(system) @ yc
        self.coef_ = z.T @ dual
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        z = (np.asarray(features, dtype=np.float64) - self.mean_) / self.scale_
        return z @ self.coef_ + self.y_mean_


def _component_balanced_mse(y: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> float:
    return float(np.mean([
        np.mean((pred[groups == group] - y[groups == group]) ** 2)
        for group in sorted(set(groups.tolist()))
    ]))


def _fit_grouped_ridge(
    train: Sequence[Mapping],
    feature_fn: Callable,
    *,
    target: str,
    alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
) -> tuple[_StandardizedRidge, dict, tuple[tuple[str, ...], ...]]:
    rows = _rows(train, target=target)
    condition_vocab = _condition_vocab(rows)
    x = _feature_matrix(rows, feature_fn, condition_vocab)
    y = np.asarray([float(row[target]) for row in rows], dtype=np.float64)
    groups = _components(rows)
    levels = sorted(set(groups.tolist()))
    if len(levels) < 3:
        selected = float(alphas[len(alphas) // 2])
        cv_status = "FIXED_MIDDLE_ALPHA_TOO_FEW_COMPONENTS"
        scores = {}
    else:
        n_folds = min(5, len(levels))
        fold_of = {group: index % n_folds for index, group in enumerate(levels)}
        scores = {}
        for alpha in alphas:
            fold_scores = []
            for fold in range(n_folds):
                valid = np.asarray([fold_of[group] == fold for group in groups], dtype=bool)
                if not np.any(valid) or not np.any(~valid):
                    continue
                model = _StandardizedRidge(float(alpha)).fit(x[~valid], y[~valid])
                fold_scores.append(_component_balanced_mse(y[valid], model.predict(x[valid]), groups[valid]))
            scores[float(alpha)] = float(np.mean(fold_scores))
        selected = min(scores, key=lambda alpha: (scores[alpha], alpha))
        cv_status = "TRAIN_ONLY_SOURCE_COMPONENT_GROUPED_CV"
    model = _StandardizedRidge(selected).fit(x, y)
    return model, {
        "alpha": float(selected),
        "alpha_grid": [float(value) for value in alphas],
        "cv_status": cv_status,
        "cv_component_balanced_mse": {str(key): value for key, value in scores.items()},
        "n_train_components": len(levels),
        "ridge_solver": model.solver_,
        "ridge_solver_policy": RIDGE_SOLVER_POLICY,
        "ridge_square_system_dimension": model.square_system_dimension_,
        "ridge_n_train_rows": model.n_train_rows_,
        "ridge_n_features": model.n_features_in_,
        "condition_features": "TRAIN_ONLY_ENDPOINT_CONTEXT_REGION_ONE_HOT_WITH_UNK",
        "condition_vocab": [list(values) for values in condition_vocab],
        "feature_schema": FEATURE_SCHEMA_BY_FUNCTION.get(
            feature_fn, "UNREGISTERED_FEATURE_FUNCTION"
        ),
    }, condition_vocab


def zero_predictor(train: Sequence[Mapping], test: Sequence[Mapping]) -> BaselineResult:
    _rows(train)
    tests = _rows(test)
    return BaselineResult(
        "zero_delta", "RUN", "NO_ANCHOR_ZERO_PRIOR", DIRECT_LEADERBOARD,
        np.zeros(len(tests), dtype=np.float64),
    )


def group_balanced_prior(train: Sequence[Mapping], test: Sequence[Mapping]) -> BaselineResult:
    trains = _rows(train, target="delta")
    tests = _rows(test)
    _components(trains)
    by_condition: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    global_groups: dict[str, list[float]] = {}
    for row in trains:
        group = str(row["source_component_id"])
        value = float(row["delta"])
        global_groups.setdefault(group, []).append(value)
        by_condition.setdefault(_condition(row), {}).setdefault(group, []).append(value)

    def balanced(groups: Mapping[str, Sequence[float]]) -> float:
        return float(np.mean([np.mean(values) for _, values in sorted(groups.items())]))

    fallback = balanced(global_groups)
    prediction = [balanced(by_condition[key]) if key in by_condition else fallback for key in map(_condition, tests)]
    return BaselineResult(
        "component_balanced_train_prior", "RUN", "NO_ANCHOR_ZERO_PRIOR",
        DIRECT_LEADERBOARD, np.asarray(prediction, dtype=np.float64),
        metadata={"aggregation": "SOURCE_COMPONENT_BALANCED_WITHIN_ENDPOINT_CONTEXT_REGION"},
    )


def _ridge_result(
    name: str,
    information_set: str,
    train: Sequence[Mapping],
    test: Sequence[Mapping],
    feature_fn: Callable,
    *,
    target: str = "delta",
    leaderboard: str = DIRECT_LEADERBOARD,
) -> BaselineResult:
    tests = _rows(test)
    model, metadata, condition_vocab = _fit_grouped_ridge(train, feature_fn, target=target)
    prediction = model.predict(_feature_matrix(tests, feature_fn, condition_vocab))
    return BaselineResult(
        name, "RUN", information_set, leaderboard,
        np.asarray(prediction, dtype=np.float64), metadata=metadata,
    )


def edit_only_ridge(train: Sequence[Mapping], test: Sequence[Mapping]) -> BaselineResult:
    return _ridge_result("edit_only_ridge", "EDIT_ONLY", train, test, edit_only_features)


def candidate_kmer_ridge(train: Sequence[Mapping], test: Sequence[Mapping]) -> BaselineResult:
    return _ridge_result("candidate_kmer_ridge", "CANDIDATE_ONLY", train, test, candidate_kmer_features)


def paired_kmer_ridge(train: Sequence[Mapping], test: Sequence[Mapping]) -> BaselineResult:
    return _ridge_result(
        "paired_local_global_kmer_ridge", "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        train, test, paired_kmer_features,
    )


def nearest_neighbor(
    train: Sequence[Mapping],
    test: Sequence[Mapping],
    *,
    train_chunk_size: int = 4096,
) -> BaselineResult:
    if (
        isinstance(train_chunk_size, bool)
        or not isinstance(train_chunk_size, int)
        or train_chunk_size <= 0
    ):
        raise BaselineProtocolError("nearest-neighbor train_chunk_size must be positive")
    trains = _rows(train, target="delta")
    tests = _rows(test)
    _components(trains)
    condition_vocab = _condition_vocab(trains)
    x = _feature_matrix(trains, paired_kmer_features, condition_vocab)
    xt = _feature_matrix(tests, paired_kmer_features, condition_vocab)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    xz = (x - mean) / scale
    xzt = (xt - mean) / scale
    y = np.asarray([float(row["delta"]) for row in trains])
    predictions = []
    for value in xzt:
        minimum = np.inf
        for start in range(0, len(xz), train_chunk_size):
            distance = np.sum(
                (xz[start : start + train_chunk_size] - value) ** 2,
                axis=1,
            )
            minimum = min(minimum, float(np.min(distance)))
        tied_values: list[float] = []
        for start in range(0, len(xz), train_chunk_size):
            distance = np.sum(
                (xz[start : start + train_chunk_size] - value) ** 2,
                axis=1,
            )
            tied = np.isclose(distance, minimum, rtol=0.0, atol=1e-12)
            tied_values.extend(
                float(item)
                for item in y[start : start + train_chunk_size][tied].tolist()
            )
        if not tied_values:  # Defensive: the exact minimum must tie with itself.
            raise BaselineProtocolError("nearest-neighbor search produced no finite minimum")
        predictions.append(math.fsum(tied_values) / len(tied_values))
    return BaselineResult(
        "paired_feature_nearest_neighbor", "RUN", "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        DIRECT_LEADERBOARD, np.asarray(predictions), metadata={
            "scaler": "TRAIN_ONLY",
            "search": NEAREST_NEIGHBOR_SEARCH_POLICY,
            "train_chunk_size": train_chunk_size,
            "feature_schema": PAIRED_KMER_FEATURE_SCHEMA,
            "condition_features": "TRAIN_ONLY_ENDPOINT_CONTEXT_REGION_ONE_HOT_WITH_UNK",
            "condition_vocab": [list(values) for values in condition_vocab],
        },
    )


def _component_cell_balanced_sample_weights(
    cell_ids: Sequence[str],
    component_ids: Sequence[str],
) -> np.ndarray:
    """Give every frozen estimand cell and source component equal mass."""

    cells = list(cell_ids)
    components = list(component_ids)
    if not cells or len(cells) != len(components):
        raise BaselineProtocolError("cell/component weight inputs must align")
    if any(not isinstance(value, str) or not value for value in cells):
        raise BaselineProtocolError("training cell IDs must be non-empty strings")
    if any(not isinstance(value, str) or not value for value in components):
        raise BaselineProtocolError("source component IDs must be non-empty strings")
    cell_components: dict[str, dict[str, list[int]]] = {}
    for index, (cell, component) in enumerate(zip(cells, components, strict=True)):
        cell_components.setdefault(cell, {}).setdefault(component, []).append(index)
    weights = np.zeros(len(cells), dtype=np.float64)
    n_cells = len(cell_components)
    for _, components in sorted(cell_components.items()):
        n_components = len(components)
        for _, indices in sorted(components.items()):
            per_row = 1.0 / (n_cells * n_components * len(indices))
            weights[np.asarray(indices, dtype=np.int64)] = per_row
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise BaselineProtocolError("component/cell-balanced weights are invalid")
    return weights * (len(weights) / float(np.sum(weights)))


def anchor_only(train: Sequence[Mapping], test: Sequence[Mapping]) -> BaselineResult:
    trains = _rows(train, target="candidate_value")
    tests = _rows(test)
    _components(trains)
    for row in tests:
        value = row.get("source_value")
        if value is None or not np.isfinite(float(value)):
            raise BaselineProtocolError("anchor-only baseline requires finite test source_value")
    groups: dict[str, list[float]] = {}
    for row in trains:
        groups.setdefault(str(row["source_component_id"]), []).append(float(row["candidate_value"]))
    candidate_prior = float(np.mean([np.mean(values) for _, values in sorted(groups.items())]))
    pred = np.asarray([candidate_prior - float(row["source_value"]) for row in tests])
    return BaselineResult(
        "candidate_absolute_prior_minus_measured_anchor", "RUN",
        "MEASURED_SOURCE_ANCHOR_ASSISTED", ANCHOR_LEADERBOARD, pred,
    )


def absolute_candidate_plus_anchor(train: Sequence[Mapping], test: Sequence[Mapping]) -> BaselineResult:
    tests = _rows(test)
    for row in tests:
        value = row.get("source_value")
        if value is None or not np.isfinite(float(value)):
            raise BaselineProtocolError("absolute-candidate baseline requires finite test source_value")
    absolute = _ridge_result(
        "candidate_absolute_kmer", "MEASURED_SOURCE_ANCHOR_ASSISTED", train, tests,
        candidate_kmer_features, target="candidate_value", leaderboard=ANCHOR_LEADERBOARD,
    )
    prediction = absolute.predictions - np.asarray([float(row["source_value"]) for row in tests])
    return BaselineResult(
        "candidate_absolute_kmer_minus_measured_anchor", "RUN",
        "MEASURED_SOURCE_ANCHOR_ASSISTED", ANCHOR_LEADERBOARD, prediction,
        metadata=absolute.metadata,
    )


def sign_logistic_diagnostic(train: Sequence[Mapping], test: Sequence[Mapping]) -> BaselineResult:
    trains = _rows(train, target="delta")
    tests = _rows(test)
    _components(trains)
    keep = np.asarray([float(row["delta"]) != 0.0 for row in trains])
    if not np.any(keep):
        raise BaselineProtocolError("sign diagnostic has no non-zero training effects")
    condition_vocab = _condition_vocab(trains)
    x = _feature_matrix(trains, edit_only_features, condition_vocab)[keep]
    xt = _feature_matrix(tests, edit_only_features, condition_vocab)
    y = np.asarray([float(row["delta"]) > 0.0 for row in trains], dtype=np.float64)[keep]
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    zt = np.column_stack([np.ones(len(xt)), (xt - mean) / scale])
    if np.all(y == y[0]):
        probability = np.full(len(xt), 0.999 if y[0] else 0.001)
    else:
        weights = np.zeros(z.shape[1], dtype=np.float64)
        penalty = np.eye(z.shape[1])
        penalty[0, 0] = 0.0
        for _ in range(50):
            logits = np.clip(z @ weights, -30.0, 30.0)
            prob = 1.0 / (1.0 + np.exp(-logits))
            curvature = np.maximum(prob * (1.0 - prob), 1e-6)
            gradient = z.T @ (prob - y) + penalty @ weights
            hessian = (z.T * curvature) @ z + penalty
            step = np.linalg.solve(hessian, gradient)
            weights -= step
            if np.linalg.norm(step) < 1e-8:
                break
        probability = 1.0 / (1.0 + np.exp(-np.clip(zt @ weights, -30.0, 30.0)))
    return BaselineResult(
        "edit_only_sign_logistic", "RUN", "EDIT_ONLY", DIRECT_LEADERBOARD,
        np.asarray(probability), role="DIAGNOSTIC_ONLY_NOT_EFFECT_GATE",
        metadata={"target": "SIGN_DELTA", "cannot_substitute_regression_sign_gate": True},
    )


def fit_optional_hist_gradient_boosting(
    train: Sequence[Mapping],
    test: Sequence[Mapping],
    *,
    execution_authorized: bool = False,
    minimum_train_rows: int = 100,
    training_cell_ids: Sequence[str] | None = None,
) -> BaselineResult:
    # Some unsupported Python/platform builds fail inside the compiled
    # histogram backend rather than raising a catchable exception.  Therefore
    # availability must be established by the frozen environment audit and the
    # run must be explicitly authorized; importing it is not a harmless probe.
    if not execution_authorized:
        return BaselineResult(
            "hist_gradient_boosting", "NOT_RUN_WITH_REASON",
            "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR", DIRECT_LEADERBOARD,
            reason="optional compiled baseline requires explicit frozen-environment execution authorization",
        )
    if len(train) < int(minimum_train_rows):
        return BaselineResult(
            "hist_gradient_boosting", "NOT_RUN_WITH_REASON",
            "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR", DIRECT_LEADERBOARD,
            reason=f"fewer than frozen minimum_train_rows={int(minimum_train_rows)}",
        )
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except Exception as exc:  # pragma: no cover - depends on optional environment
        return BaselineResult(
            "hist_gradient_boosting", "NOT_RUN_WITH_REASON",
            "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR", DIRECT_LEADERBOARD,
            reason=f"optional scikit-learn unavailable: {type(exc).__name__}",
        )
    trains = _rows(train, target="delta")
    tests = _rows(test)
    components = _components(trains)
    if training_cell_ids is None:
        return BaselineResult(
            "hist_gradient_boosting", "NOT_RUN_WITH_REASON",
            "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR", DIRECT_LEADERBOARD,
            reason=(
                "HGB requires exact-bound frozen estimand-cell IDs for "
                "component/cell-balanced training weights"
            ),
        )
    condition_vocab = _condition_vocab(trains)
    x = _feature_matrix(trains, paired_kmer_features, condition_vocab)
    xt = _feature_matrix(tests, paired_kmer_features, condition_vocab)
    y = np.asarray([float(row["delta"]) for row in trains])
    sample_weight = _component_cell_balanced_sample_weights(
        training_cell_ids, components.tolist()
    )
    model = HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=42, early_stopping=False,
    )
    try:
        model.fit(x, y, sample_weight=sample_weight)
        prediction = np.asarray(model.predict(xt), dtype=np.float64)
    except Exception as exc:
        return BaselineResult(
            "hist_gradient_boosting", "NOT_RUN_WITH_REASON",
            "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR", DIRECT_LEADERBOARD,
            reason=f"optional baseline not executable: {type(exc).__name__}: {exc}",
        )
    return BaselineResult(
        "hist_gradient_boosting", "RUN", "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        DIRECT_LEADERBOARD, prediction,
        metadata={
            "random_state": 42,
            "tuning": "FROZEN_NO_TEST_SELECTION",
            "training_weighting": HGB_TRAINING_WEIGHT_POLICY,
            "feature_schema": PAIRED_KMER_FEATURE_SCHEMA,
            "condition_features": "TRAIN_ONLY_ENDPOINT_CONTEXT_REGION_ONE_HOT_WITH_UNK",
            "condition_vocab": [list(values) for values in condition_vocab],
        },
    )


def foundation_baseline_status(*, exposure_qualified: bool, feature_path: Optional[Path]) -> BaselineResult:
    if not exposure_qualified:
        reason = "foundation exposure/license qualification is not frozen"
    elif feature_path is None:
        reason = "foundation feature artifact is not provided"
    elif not Path(feature_path).is_file():
        reason = "foundation feature artifact does not exist"
    else:
        reason = "foundation adapter requires a separately frozen feature schema and is not executable here"
    return BaselineResult(
        "foundation_embedding_linear_head", "NOT_RUN_WITH_REASON",
        "FOUNDATION_EXPOSURE_STRATIFIED", "FOUNDATION_EXPOSURE_STRATIFIED",
        reason=reason,
    )
