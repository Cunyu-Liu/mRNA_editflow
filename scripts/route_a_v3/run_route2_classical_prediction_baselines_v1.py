#!/usr/bin/env python3
"""Fit frozen-budget classical Route 2 prediction baselines on Development only."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import torch
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.feature_extraction import DictVectorizer
from threadpoolctl import threadpool_limits

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_gpu_failure_evidence import cuda_device_observation, write_gpu_failure_evidence


BASES = "ACGT"
TRANSITIONS = tuple(f"{left}>{right}" for left in BASES for right in BASES if left != right)
MOTIFS = ("AATAAA", "ATTAAA", "TATTTAT", "TTTTT", "CG", "TGTA")


class BaselineError(RuntimeError):
    pass


class GPUExecutionError(BaselineError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BaselineError(message)


def _gpu_require(condition: bool, message: str) -> None:
    if not condition:
        raise GPUExecutionError(message)


def _finite_optional(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BaselineError(f"invalid manifest JSON at line {line_number}") from exc
            _require(row["pool_assignment"] == "DEVELOPMENT", "non-Development record entered training manifest")
            record_id = str(row["canonical_record_id"])
            _require(record_id not in result, f"manifest record duplicated: {record_id}")
            _require(row["split"] in {"TRAIN", "VALIDATION", "TEST"}, "unexpected Development split")
            result[record_id] = {
                "split": str(row["split"]),
                "study_unit_id": str(row["study_unit_id"]),
                "connected_source_component_id": str(row["connected_source_component_id"]),
            }
    _require(result, "Development manifest is empty")
    return result


def load_records(canonical_paths: list[Path], manifest: Mapping[str, Mapping[str, str]]) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in canonical_paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise BaselineError(f"invalid canonical JSON in {path.name}:{line_number}") from exc
                record_id = str(row["canonical_record_id"])
                if record_id not in manifest:
                    continue
                _require(record_id not in records, f"canonical record duplicated: {record_id}")
                _require(row["pool_assignment"] == "DEVELOPMENT", "Evaluation record entered baseline loader")
                source, candidate = str(row["source_sequence"]).upper(), str(row["candidate_sequence"]).upper()
                _require(len(source) == len(candidate) and not ((set(source) | set(candidate)) - set(BASES)), "invalid sequence pair")
                target = _finite_optional(row["direction_normalized_delta"])
                _require(target is not None, f"target missing: {record_id}")
                records[record_id] = {
                    "canonical_record_id": record_id,
                    "split": manifest[record_id]["split"],
                    "connected_source_component_id": manifest[record_id]["connected_source_component_id"],
                    "study_unit_id": str(row["study_unit_id"]),
                    "region": str(row["region"]),
                    "endpoint_id": str(row["endpoint_id"]),
                    "assay_id": str(row["assay_id"]),
                    "biological_context_id": str(row["biological_context_id"]),
                    "source_id": str(row["source_id"]),
                    "source_sequence": source,
                    "candidate_sequence": candidate,
                    "edit_operations": row["edit_operations"],
                    "target": target,
                    "source_endpoint_value": _finite_optional(row.get("source_endpoint_value")),
                    "candidate_endpoint_value": _finite_optional(row.get("candidate_endpoint_value")),
                }
    _require(set(records) == set(manifest), "canonical records do not exactly cover Development manifest")
    return [records[record_id] for record_id in sorted(records)]


@lru_cache(maxsize=None)
def _kmer_vector(sequence: str) -> np.ndarray:
    values: list[float] = []
    for k in (1, 2, 3):
        denominator = max(1, len(sequence) - k + 1)
        counts = Counter(sequence[index:index + k] for index in range(denominator))
        for token in map("".join, itertools.product(BASES, repeat=k)):
            values.append(counts[token] / denominator)
    return np.asarray(values, dtype=np.float32)


def _edit_parts(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    source = record["source_sequence"]
    transitions = Counter()
    positions = []
    for edit in record["edit_operations"]:
        _require(edit["type"] == "SUB", "classical baseline received non-SUB action")
        position = edit.get("position_zero_based", edit.get("position"))
        _require(isinstance(position, int) and 0 <= position < len(source), "edit position is invalid")
        ref, alt = str(edit["ref"]), str(edit["alt"])
        transitions[f"{ref}>{alt}"] += 1
        positions.append(position / max(1, len(source) - 1))
    _require(positions, "zero-edit record entered baseline")
    position_vector = np.asarray([
        float(len(positions)),
        float(min(positions)),
        float(max(positions)),
        float(np.mean(positions)),
        float(np.std(positions)),
    ], dtype=np.float32)
    transition_vector = np.asarray([float(transitions[token]) for token in TRANSITIONS], dtype=np.float32)
    return position_vector, transition_vector


def _edit_vector(record: Mapping[str, Any]) -> np.ndarray:
    position, transition = _edit_parts(record)
    return np.concatenate([position, transition])


@lru_cache(maxsize=None)
def _gc_mfe_motif_vector(sequence: str) -> np.ndarray:
    try:
        import RNA
    except ImportError as exc:
        raise BaselineError("ViennaRNA is unavailable for the GC/MFE/motif baseline") from exc
    rna = sequence.replace("T", "U")
    _structure, mfe = RNA.fold(rna)
    denominator = max(1, len(sequence))
    return np.asarray([
        (sequence.count("G") + sequence.count("C")) / denominator,
        float(mfe) / denominator,
        *[sequence.count(motif) / denominator for motif in MOTIFS],
    ], dtype=np.float32)


def _numeric_features(record: Mapping[str, Any], mode: str) -> np.ndarray:
    source = _kmer_vector(record["source_sequence"])
    candidate = _kmer_vector(record["candidate_sequence"])
    difference = candidate - source
    edit = _edit_vector(record)
    if mode == "full":
        return np.concatenate([source, candidate, difference, edit])
    if mode == "source_centered":
        return np.concatenate([source, difference, edit])
    if mode == "candidate_only":
        return candidate
    if mode == "source_only":
        return source
    if mode == "edit_only":
        return edit
    if mode == "edit_position_only":
        return _edit_parts(record)[0]
    if mode == "ref_alt_only":
        return _edit_parts(record)[1]
    if mode == "gc_mfe_motif":
        source_values = _gc_mfe_motif_vector(record["source_sequence"])
        candidate_values = _gc_mfe_motif_vector(record["candidate_sequence"])
        return np.concatenate([source_values, candidate_values, candidate_values - source_values])
    if mode == "context_only":
        return np.empty(0, dtype=np.float32)
    if mode == "sequence_absolute":
        raise BaselineError("absolute sequence features require an explicit sequence")
    raise BaselineError(f"unknown feature mode: {mode}")


def _absolute_sequence_features(sequence: str) -> np.ndarray:
    return _kmer_vector(sequence)


def _context_features(record: Mapping[str, Any]) -> dict[str, float]:
    return {
        f"study={record['study_unit_id']}": 1.0,
        f"region={record['region']}": 1.0,
        f"endpoint={record['endpoint_id']}": 1.0,
        f"assay={record['assay_id']}": 1.0,
        f"context={record['biological_context_id']}": 1.0,
    }


class FeatureEncoder:
    def __init__(self, mode: str):
        self.mode = mode
        self.context = DictVectorizer(sparse=True)

    def fit_transform(self, records: list[dict[str, Any]]) -> sparse.csr_matrix:
        numeric_values = [_numeric_features(record, self.mode) for record in records]
        numeric = sparse.csr_matrix((len(records), 0)) if not len(numeric_values[0]) else sparse.csr_matrix(np.vstack(numeric_values))
        context = self.context.fit_transform([_context_features(record) for record in records])
        return sparse.hstack([numeric, context], format="csr")

    def transform(self, records: list[dict[str, Any]]) -> sparse.csr_matrix:
        numeric_values = [_numeric_features(record, self.mode) for record in records]
        numeric = sparse.csr_matrix((len(records), 0)) if not len(numeric_values[0]) else sparse.csr_matrix(np.vstack(numeric_values))
        context = self.context.transform([_context_features(record) for record in records])
        return sparse.hstack([numeric, context], format="csr")


class AbsoluteEncoder:
    def __init__(self):
        self.context = DictVectorizer(sparse=True)

    def fit_transform(self, records: list[dict[str, Any]], roles: list[str]) -> sparse.csr_matrix:
        numeric = sparse.csr_matrix(np.vstack([
            _absolute_sequence_features(record[f"{role}_sequence"]) for record, role in zip(records, roles)
        ]))
        context_dicts = []
        for record, role in zip(records, roles):
            values = _context_features(record)
            values[f"role={role}"] = 1.0
            context_dicts.append(values)
        context = self.context.fit_transform(context_dicts)
        return sparse.hstack([numeric, context], format="csr")

    def transform_role(self, records: list[dict[str, Any]], role: str) -> sparse.csr_matrix:
        numeric = sparse.csr_matrix(np.vstack([
            _absolute_sequence_features(record[f"{role}_sequence"]) for record in records
        ]))
        context_dicts = []
        for record in records:
            values = _context_features(record)
            values[f"role={role}"] = 1.0
            context_dicts.append(values)
        return sparse.hstack([numeric, self.context.transform(context_dicts)], format="csr")


def _mae(records: list[dict[str, Any]], predictions: np.ndarray) -> float:
    targets = np.asarray([record["target"] for record in records], dtype=float)
    return float(np.mean(np.abs(predictions - targets)))


def _spearman(records: list[dict[str, Any]], predictions: np.ndarray) -> float | None:
    targets = np.asarray([record["target"] for record in records], dtype=float)
    if len(targets) < 3 or np.std(targets) == 0 or np.std(predictions) == 0:
        return None
    result = float(spearmanr(targets, predictions).statistic)
    return result if math.isfinite(result) else None


def _task_macro_spearman(records: list[dict[str, Any]], predictions: np.ndarray) -> float | None:
    _require(len(records) == len(predictions) and records, "task-macro metric inputs differ")
    indices_by_task: dict[tuple[str, str], list[int]] = {}
    for index, record in enumerate(records):
        indices_by_task.setdefault((str(record["region"]), str(record["endpoint_id"])), []).append(index)
    correlations = []
    for indices in indices_by_task.values():
        correlation = _spearman([records[index] for index in indices], predictions[indices])
        if correlation is None:
            return None
        correlations.append(correlation)
    return float(np.mean(correlations))


def _training_weights(records: list[dict[str, Any]], weighting: str) -> np.ndarray | None:
    _require(
        weighting in {"ROW", "SOURCE_GROUP_EQUAL", "STUDY_THEN_SOURCE_GROUP_EQUAL"},
        f"unknown training weighting: {weighting}",
    )
    if weighting == "ROW":
        return None
    keys = [
        (
            record["study_unit_id"], record["source_id"],
            record["biological_context_id"], record["endpoint_id"],
        )
        for record in records
    ]
    counts: dict[tuple[str, str, str, str], int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    if weighting == "SOURCE_GROUP_EQUAL":
        scale = len(records) / len(counts)
        weights = np.asarray([scale / counts[key] for key in keys], dtype=float)
    else:
        studies = sorted({key[0] for key in keys})
        group_count_by_study: dict[str, int] = {}
        for key in counts:
            group_count_by_study[key[0]] = group_count_by_study.get(key[0], 0) + 1
        scale = len(records) / len(studies)
        weights = np.asarray([
            scale / (group_count_by_study[key[0]] * counts[key])
            for key in keys
        ], dtype=float)
    _require(np.isclose(weights.mean(), 1.0), "source-group weights do not normalize")
    return weights


def require_cuda_device(
    device_text: str,
    physical_gpu_index: int,
    minimum_free_gpu_memory_bytes: int = 0,
) -> tuple[torch.device, int]:
    _gpu_require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden for physical-device provenance")
    _gpu_require(device_text.startswith("cuda:"), "classical baseline fitting requires an explicit CUDA device")
    _gpu_require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    _gpu_require(0 <= physical_gpu_index < torch.cuda.device_count(), "physical GPU index is unavailable")
    device = torch.device(device_text)
    _gpu_require(device.index == physical_gpu_index, "CUDA device index differs from declared physical GPU")
    torch.cuda.set_device(device)
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    _gpu_require(free_bytes >= minimum_free_gpu_memory_bytes, "selected GPU has insufficient free memory")
    probe = torch.ones(1, device=device)
    _gpu_require(probe.is_cuda and probe.device == device, "CUDA allocation did not remain on the declared device")
    del probe
    return device, int(free_bytes)


def _dense_cuda(matrix: sparse.spmatrix, device: torch.device) -> torch.Tensor:
    values = np.asarray(matrix.toarray(), dtype=np.float32)
    result = torch.from_numpy(values).to(device)
    _gpu_require(result.is_cuda and result.device == device, "feature tensor left the declared CUDA device")
    return result


def _scale_cuda(
    x_train: torch.Tensor, x_target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    variance = torch.mean((x_train - x_train.mean(dim=0)) ** 2, dim=0)
    scale = torch.sqrt(torch.clamp(variance, min=0.0))
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return x_train / scale, x_target / scale, scale.detach().cpu().numpy()


def _weights_cuda(
    records: list[dict[str, Any]], weighting: str, device: torch.device
) -> torch.Tensor:
    values = _training_weights(records, weighting)
    if values is None:
        return torch.ones(len(records), dtype=torch.float32, device=device)
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _ridge_cuda(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_target: torch.Tensor,
    alpha: float,
    sample_weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    _require(alpha > 0, "ridge alpha must be positive")
    ones = torch.ones((len(x_train), 1), dtype=x_train.dtype, device=x_train.device)
    design = torch.cat((x_train, ones), dim=1)
    weighted_design = design * sample_weight[:, None]
    gram = (design.T @ weighted_design).to(torch.float64)
    rhs = (design.T @ (sample_weight * y_train)).to(torch.float64)
    penalty = torch.eye(gram.shape[0], dtype=torch.float64, device=x_train.device) * alpha
    penalty[-1, -1] = 0.0
    try:
        solution = torch.linalg.solve(gram + penalty, rhs).to(x_train.dtype)
    except RuntimeError as exc:
        raise GPUExecutionError(f"CUDA ridge solve failed: {exc}") from exc
    prediction = x_target @ solution[:-1] + solution[-1]
    _gpu_require(prediction.is_cuda, "ridge prediction silently fell back to CPU")
    return prediction, {
        "coefficient": solution[:-1].detach().cpu().numpy(),
        "intercept": float(solution[-1].detach().cpu()),
        "alpha": alpha,
        "optimizer": "CUDA_WEIGHTED_NORMAL_EQUATION",
    }


def _elastic_net_cuda(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_target: torch.Tensor,
    alpha: float,
    l1_ratio: float,
    sample_weight: torch.Tensor,
    max_iter: int,
    tolerance: float,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    _require(alpha > 0 and 0 <= l1_ratio <= 1, "elastic-net penalties are invalid")
    _require(max_iter > 0 and tolerance > 0, "elastic-net optimization budget is invalid")
    generator = torch.Generator(device=x_train.device)
    generator.manual_seed(seed)
    weight_sum = sample_weight.sum()
    coefficient = torch.zeros(x_train.shape[1], dtype=x_train.dtype, device=x_train.device)
    intercept = torch.sum(sample_weight * y_train) / weight_sum

    # Power iteration estimates the smooth objective's Lipschitz constant on CUDA.
    vector = torch.randn(x_train.shape[1] + 1, generator=generator, dtype=x_train.dtype, device=x_train.device)
    vector = vector / vector.norm().clamp_min(1e-12)
    for _ in range(30):
        projected = x_train @ vector[:-1] + vector[-1]
        next_coefficient = x_train.T @ (sample_weight * projected) / weight_sum
        next_intercept = torch.sum(sample_weight * projected) / weight_sum
        next_coefficient = next_coefficient + alpha * (1.0 - l1_ratio) * vector[:-1]
        updated = torch.cat((next_coefficient, next_intercept.reshape(1)))
        vector = updated / updated.norm().clamp_min(1e-12)
    projected = x_train @ vector[:-1] + vector[-1]
    smooth_action = torch.cat((
        x_train.T @ (sample_weight * projected) / weight_sum
        + alpha * (1.0 - l1_ratio) * vector[:-1],
        (torch.sum(sample_weight * projected) / weight_sum).reshape(1),
    ))
    lipschitz = torch.dot(vector, smooth_action).clamp_min(1e-8)
    step = 1.0 / lipschitz

    momentum_coefficient = coefficient.clone()
    momentum_intercept = intercept.clone()
    acceleration = 1.0
    converged = False
    completed_iterations = 0
    for iteration in range(1, max_iter + 1):
        residual = x_train @ momentum_coefficient + momentum_intercept - y_train
        gradient_coefficient = x_train.T @ (sample_weight * residual) / weight_sum
        gradient_coefficient = gradient_coefficient + alpha * (1.0 - l1_ratio) * momentum_coefficient
        gradient_intercept = torch.sum(sample_weight * residual) / weight_sum
        proposal = momentum_coefficient - step * gradient_coefficient
        threshold = step * alpha * l1_ratio
        next_coefficient = torch.sign(proposal) * torch.clamp(torch.abs(proposal) - threshold, min=0.0)
        next_intercept = momentum_intercept - step * gradient_intercept
        maximum_change = torch.max(torch.abs(next_coefficient - coefficient))
        maximum_change = torch.maximum(maximum_change, torch.abs(next_intercept - intercept))
        next_acceleration = (1.0 + math.sqrt(1.0 + 4.0 * acceleration * acceleration)) / 2.0
        momentum = (acceleration - 1.0) / next_acceleration
        momentum_coefficient = next_coefficient + momentum * (next_coefficient - coefficient)
        momentum_intercept = next_intercept + momentum * (next_intercept - intercept)
        coefficient, intercept = next_coefficient, next_intercept
        acceleration = next_acceleration
        completed_iterations = iteration
        if float(maximum_change.detach().cpu()) <= tolerance:
            converged = True
            break
    _require(converged, f"CUDA elastic net did not converge within {max_iter} iterations")
    prediction = x_target @ coefficient + intercept
    _gpu_require(prediction.is_cuda, "elastic-net prediction silently fell back to CPU")
    return prediction, {
        "coefficient": coefficient.detach().cpu().numpy(),
        "intercept": float(intercept.detach().cpu()),
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "optimizer": "CUDA_FISTA",
        "completed_iterations": completed_iterations,
        "converged": converged,
    }


def _fit_predict_linear(
    train: list[dict[str, Any]],
    target: list[dict[str, Any]],
    feature_mode: str,
    kind: str,
    parameters: Mapping[str, Any],
    device: torch.device,
    weighting: str = "ROW",
):
    encoder = FeatureEncoder(feature_mode)
    x_train = encoder.fit_transform(train)
    x_target = encoder.transform(target)
    x_train_cuda, x_target_cuda = _dense_cuda(x_train, device), _dense_cuda(x_target, device)
    x_train_scaled, x_target_scaled, scale = _scale_cuda(x_train_cuda, x_target_cuda)
    y_train = torch.tensor([record["target"] for record in train], dtype=torch.float32, device=device)
    sample_weight = _weights_cuda(train, weighting, device)
    if kind == "ridge":
        prediction, fitted = _ridge_cuda(
            x_train_scaled, y_train, x_target_scaled, float(parameters["alpha"]), sample_weight
        )
    elif kind == "elastic_net":
        prediction, fitted = _elastic_net_cuda(
            x_train_scaled,
            y_train,
            x_target_scaled,
            float(parameters["alpha"]),
            float(parameters["l1_ratio"]),
            sample_weight,
            int(parameters.get("max_iter", 2000)),
            float(parameters.get("tolerance", 1e-5)),
            int(parameters["seed"]),
        )
    else:
        raise BaselineError(f"unknown linear kind: {kind}")
    return prediction.detach().cpu().numpy(), {
        "encoder_spec": {
            "type": "FEATURE",
            "mode": encoder.mode,
            "context": encoder.context,
        },
        "feature_scale": scale,
        "model": fitted,
        "weighting": weighting,
    }


def _fit_predict_xgboost(
    train: list[dict[str, Any]],
    target: list[dict[str, Any]],
    parameters: Mapping[str, Any],
    device: torch.device,
    weighting: str = "ROW",
):
    import xgboost as xgb
    _require(int(parameters["n_jobs"]) <= 4, "XGBoost CPU thread cap exceeded")
    encoder = FeatureEncoder("full")
    x_train = encoder.fit_transform(train)
    x_target = encoder.transform(target)
    y_train = np.asarray([record["target"] for record in train], dtype=np.float32)
    weights = _training_weights(train, weighting)
    model = xgb.XGBRegressor(
        n_estimators=int(parameters["n_estimators"]),
        max_depth=int(parameters["max_depth"]),
        learning_rate=float(parameters["learning_rate"]),
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device=str(device),
        n_jobs=int(parameters["n_jobs"]),
        random_state=int(parameters["seed"]),
    )
    # XGBoost accepts CPU-resident DMatrix staging while `device` controls both
    # histogram construction and booster prediction.  Passing torch CUDA tensors
    # would add a CuPy-only adapter dependency that is absent from this runtime.
    model.fit(x_train, y_train, sample_weight=weights)
    booster_device = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
    _gpu_require(booster_device == str(device), f"XGBoost device mismatch: {booster_device}")
    prediction = model.get_booster().predict(xgb.DMatrix(x_target))
    device_after_prediction = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
    _gpu_require(device_after_prediction == str(device), f"XGBoost prediction device mismatch: {device_after_prediction}")
    return np.asarray(prediction, dtype=float), {
        "encoder_spec": {
            "type": "FEATURE",
            "mode": encoder.mode,
            "context": encoder.context,
        },
        "model": model,
        "weighting": weighting,
        "booster_device": booster_device,
        "data_staging_execution": "CPU",
        "parameter_fit_execution": "CUDA",
        "prediction_execution": "CUDA",
    }


def _fit_predict_absolute(
    train: list[dict[str, Any]],
    target: list[dict[str, Any]],
    parameters: Mapping[str, Any],
    device: torch.device,
    weighting: str = "ROW",
):
    eligible = [
        record for record in train
        if record["source_endpoint_value"] is not None and record["candidate_endpoint_value"] is not None
    ]
    _require(len(eligible) >= int(parameters["minimum_complete_training_records"]), "absolute endpoint training set is too small")
    expanded_records = [record for record in eligible for _role in ("source", "candidate")]
    roles = [role for _record in eligible for role in ("source", "candidate")]
    targets = torch.tensor([
        record[f"{role}_endpoint_value"] for record, role in zip(expanded_records, roles)
    ], dtype=torch.float32, device=device)
    encoder = AbsoluteEncoder()
    x_train = encoder.fit_transform(expanded_records, roles)
    x_candidate = encoder.transform_role(target, "candidate")
    x_source = encoder.transform_role(target, "source")
    x_train_cuda = _dense_cuda(x_train, device)
    x_pair_cuda = _dense_cuda(sparse.vstack((x_candidate, x_source), format="csr"), device)
    x_train_scaled, x_pair_scaled, scale = _scale_cuda(x_train_cuda, x_pair_cuda)
    original_weights = _training_weights(eligible, weighting)
    if original_weights is None:
        expanded_weights = torch.ones(len(expanded_records), dtype=torch.float32, device=device)
    else:
        expanded_weights = torch.tensor(np.repeat(original_weights / 2.0, 2), dtype=torch.float32, device=device)
    pair_prediction, fitted = _ridge_cuda(
        x_train_scaled, targets, x_pair_scaled, float(parameters["alpha"]), expanded_weights
    )
    candidate_prediction, source_prediction = pair_prediction[:len(target)], pair_prediction[len(target):]
    return (candidate_prediction - source_prediction).detach().cpu().numpy(), {
        "encoder_spec": {
            "type": "ABSOLUTE",
            "context": encoder.context,
        },
        "feature_scale": scale,
        "model": fitted,
        "weighting": weighting,
    }


def _fit_predict_group_mean(
    train: list[dict[str, Any]],
    target: list[dict[str, Any]],
    fields: list[str],
    device: torch.device,
    weighting: str = "ROW",
) -> tuple[np.ndarray, dict[str, Any]]:
    values = torch.tensor([record["target"] for record in train], dtype=torch.float32, device=device)
    sample_weight = _weights_cuda(train, weighting, device)
    global_mean = float((values * sample_weight).sum().div(sample_weight.sum()).detach().cpu())
    keys = [tuple(str(record[field]) for field in fields) for record in train]
    key_to_index = {key: index for index, key in enumerate(sorted(set(keys)))}
    indices = torch.tensor([key_to_index[key] for key in keys], dtype=torch.long, device=device)
    sums = torch.zeros(len(key_to_index), dtype=torch.float32, device=device)
    counts = torch.zeros_like(sums)
    sums.scatter_add_(0, indices, values * sample_weight)
    counts.scatter_add_(0, indices, sample_weight)
    means_array = (sums / counts).detach().cpu().numpy()
    means = {key: float(means_array[index]) for key, index in key_to_index.items()}
    target_indices = [
        key_to_index.get(tuple(str(record[field]) for field in fields), -1)
        for record in target
    ]
    target_index_tensor = torch.tensor(target_indices, dtype=torch.long, device=device)
    predictions = torch.full((len(target),), global_mean, dtype=torch.float32, device=device)
    seen = target_index_tensor >= 0
    predictions[seen] = (sums / counts)[target_index_tensor[seen]]
    _gpu_require(predictions.is_cuda, "group-mean prediction silently fell back to CPU")
    return predictions.detach().cpu().numpy(), {
        "global_mean": global_mean,
        "group_fields": fields,
        "group_means": means,
        "weighting": weighting,
    }


def _fit_predict_majority_sign(
    train: list[dict[str, Any]], target: list[dict[str, Any]], device: torch.device,
    weighting: str = "ROW",
) -> tuple[np.ndarray, dict[str, Any]]:
    values = torch.tensor([record["target"] for record in train], dtype=torch.float32, device=device)
    sample_weight = _weights_cuda(train, weighting, device)
    positive = float(sample_weight[values > 0].sum().detach().cpu())
    negative = float(sample_weight[values < 0].sum().detach().cpu())
    sign = 1.0 if positive >= negative else -1.0
    absolute, order = torch.sort(torch.abs(values))
    cumulative = torch.cumsum(sample_weight[order], dim=0)
    median_index = torch.searchsorted(cumulative, sample_weight.sum() / 2.0)
    magnitude = float(absolute[median_index].detach().cpu())
    prediction = sign * magnitude
    predictions = torch.full((len(target),), prediction, dtype=torch.float32, device=device)
    _gpu_require(predictions.is_cuda, "majority-sign prediction silently fell back to CPU")
    return predictions.detach().cpu().numpy(), {
        "majority_sign": int(sign),
        "weighted_positive_mass": positive,
        "weighted_negative_mass": negative,
        "weighted_median_absolute_target": magnitude,
        "weighting": weighting,
    }


def _permute_candidates_within_source(
    records: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], int]:
    groups: dict[tuple[str, str, str, str], list[int]] = {}
    for index, record in enumerate(records):
        key = (
            record["study_unit_id"],
            record["source_id"],
            record["biological_context_id"],
            record["endpoint_id"],
        )
        groups.setdefault(key, []).append(index)
    randomizer = np.random.default_rng(seed)
    result = [dict(record) for record in records]
    changed = 0
    for indices in groups.values():
        if len(indices) < 2:
            continue
        donors = list(indices)
        randomizer.shuffle(donors)
        if donors == indices:
            donors = donors[1:] + donors[:1]
        for target_index, donor_index in zip(indices, donors):
            donor = records[donor_index]
            result[target_index]["candidate_sequence"] = donor["candidate_sequence"]
            result[target_index]["edit_operations"] = donor["edit_operations"]
            changed += int(target_index != donor_index)
    return result, changed


def _fit_predict_candidate_permutation(
    train: list[dict[str, Any]],
    target: list[dict[str, Any]],
    parameters: Mapping[str, Any],
    device: torch.device,
    weighting: str = "ROW",
):
    permuted_train, changed_train = _permute_candidates_within_source(train, int(parameters["seed"]))
    permuted_target, changed_target = _permute_candidates_within_source(target, int(parameters["seed"]) + 1)
    predictions, artifact = _fit_predict_linear(
        permuted_train, permuted_target, "full", "ridge", parameters, device, weighting
    )
    artifact["permuted_training_records"] = changed_train
    artifact["permuted_target_records"] = changed_target
    return predictions, artifact


def _parameter_trials(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    grid = spec["parameter_grid"]
    keys = sorted(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))]


def _fit_predict(
    baseline: Mapping[str, Any],
    train: list[dict[str, Any]],
    target: list[dict[str, Any]],
    parameters: Mapping[str, Any],
    device: torch.device,
):
    kind = baseline["kind"]
    weighting = str(baseline.get("weighting", "ROW"))
    if kind == "mean":
        values = torch.tensor([record["target"] for record in train], dtype=torch.float32, device=device)
        sample_weight = _weights_cuda(train, weighting, device)
        mean = float((values * sample_weight).sum().div(sample_weight.sum()).detach().cpu())
        predictions = torch.full((len(target),), mean, dtype=torch.float32, device=device)
        _gpu_require(predictions.is_cuda, "mean prediction silently fell back to CPU")
        return predictions.detach().cpu().numpy(), {"mean": mean, "weighting": weighting}
    if kind in {"ridge", "elastic_net"}:
        return _fit_predict_linear(train, target, baseline["feature_mode"], kind, parameters, device, weighting)
    if kind == "xgboost":
        return _fit_predict_xgboost(train, target, parameters, device, weighting)
    if kind == "absolute_difference_ridge":
        return _fit_predict_absolute(train, target, parameters, device, weighting)
    if kind == "group_mean":
        return _fit_predict_group_mean(
            train, target, list(baseline["group_fields"]), device, weighting
        )
    if kind == "majority_sign":
        return _fit_predict_majority_sign(train, target, device, weighting)
    if kind == "candidate_permutation_ridge":
        return _fit_predict_candidate_permutation(train, target, parameters, device, weighting)
    raise BaselineError(f"unknown baseline kind: {kind}")


def predict_from_frozen_artifact(
    baseline: Mapping[str, Any],
    artifact: Mapping[str, Any],
    target: list[dict[str, Any]],
    device: torch.device,
) -> np.ndarray:
    """Apply one already-fitted classical artifact without accessing outcomes."""
    _require(target, "frozen classical prediction target is empty")
    kind = str(baseline["kind"])
    encoder_spec = artifact.get("encoder_spec")
    if kind == "mean":
        prediction = torch.full(
            (len(target),), float(artifact["mean"]), dtype=torch.float32, device=device
        )
    elif kind in {"ridge", "elastic_net", "candidate_permutation_ridge"}:
        _require(isinstance(encoder_spec, Mapping) and encoder_spec.get("type") == "FEATURE", "frozen feature encoder is absent")
        encoder = FeatureEncoder(str(encoder_spec["mode"]))
        encoder.context = encoder_spec["context"]
        matrix = encoder.transform(target)
        values = _dense_cuda(matrix, device)
        scale = torch.as_tensor(artifact["feature_scale"], dtype=values.dtype, device=device)
        _require(scale.ndim == 1 and scale.shape[0] == values.shape[1], "frozen feature scale is incompatible")
        fitted = artifact["model"]
        coefficient = torch.as_tensor(fitted["coefficient"], dtype=values.dtype, device=device)
        _require(coefficient.shape == scale.shape, "frozen coefficient is incompatible")
        prediction = (values / scale) @ coefficient + float(fitted["intercept"])
    elif kind == "xgboost":
        import xgboost as xgb
        _require(isinstance(encoder_spec, Mapping) and encoder_spec.get("type") == "FEATURE", "frozen XGBoost encoder is absent")
        encoder = FeatureEncoder(str(encoder_spec["mode"]))
        encoder.context = encoder_spec["context"]
        matrix = encoder.transform(target)
        model = artifact["model"]
        model.set_params(device=str(device))
        booster = model.get_booster()
        booster.set_param({"device": str(device)})
        booster_device = json.loads(booster.save_config())["learner"]["generic_param"]["device"]
        _gpu_require(booster_device == str(device), f"frozen XGBoost device mismatch: {booster_device}")
        values = np.asarray(booster.predict(xgb.DMatrix(matrix)), dtype=float)
        device_after = json.loads(booster.save_config())["learner"]["generic_param"]["device"]
        _gpu_require(device_after == str(device), f"frozen XGBoost prediction device mismatch: {device_after}")
        return values
    elif kind == "absolute_difference_ridge":
        _require(isinstance(encoder_spec, Mapping) and encoder_spec.get("type") == "ABSOLUTE", "frozen absolute encoder is absent")
        encoder = AbsoluteEncoder()
        encoder.context = encoder_spec["context"]
        candidate = encoder.transform_role(target, "candidate")
        source = encoder.transform_role(target, "source")
        values = _dense_cuda(sparse.vstack((candidate, source), format="csr"), device)
        scale = torch.as_tensor(artifact["feature_scale"], dtype=values.dtype, device=device)
        _require(scale.ndim == 1 and scale.shape[0] == values.shape[1], "frozen absolute feature scale is incompatible")
        fitted = artifact["model"]
        coefficient = torch.as_tensor(fitted["coefficient"], dtype=values.dtype, device=device)
        _require(coefficient.shape == scale.shape, "frozen absolute coefficient is incompatible")
        absolute = (values / scale) @ coefficient + float(fitted["intercept"])
        prediction = absolute[:len(target)] - absolute[len(target):]
    elif kind == "group_mean":
        global_mean = float(artifact["global_mean"])
        fields = list(artifact["group_fields"])
        means = artifact["group_means"]
        prediction = torch.tensor([
            float(means.get(tuple(str(record[field]) for field in fields), global_mean))
            for record in target
        ], dtype=torch.float32, device=device)
    elif kind == "majority_sign":
        value = float(artifact["majority_sign"]) * float(artifact["median_absolute_target"])
        prediction = torch.full((len(target),), value, dtype=torch.float32, device=device)
    else:
        raise BaselineError(f"unsupported frozen classical artifact kind: {kind}")
    _gpu_require(prediction.is_cuda and prediction.device == device, "frozen classical prediction left CUDA")
    return prediction.detach().cpu().numpy()


def bind_artifact_identity(
    artifact: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        **dict(artifact),
        "artifact_baseline_id": str(baseline["baseline_id"]),
        "artifact_baseline_kind": str(baseline["kind"]),
    }


def _write_predictions(path: Path, baseline_id: str, records: list[dict[str, Any]], values: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record, prediction in zip(records, values):
            handle.write(json.dumps({
                "canonical_record_id": record["canonical_record_id"],
                "baseline_id": baseline_id,
                "predicted_direction_normalized_delta": float(prediction),
            }, sort_keys=True) + "\n")


def _completion_summary(baseline_results: Mapping[str, Mapping[str, Any]], run_mode: str) -> dict[str, Any]:
    completed = sum(
        str(result.get("status", "")).startswith("COMPLETED_")
        for result in baseline_results.values()
    )
    not_run = len(baseline_results) - completed
    _require(completed > 0, "no classical baseline completed")
    prefix = (
        "CLASSICAL_DEVELOPMENT_BASELINES"
        if run_mode == "FIXED_GROUPED_SPLIT"
        else "CLASSICAL_DEVELOPMENT_LOSO_BASELINES"
    )
    return {
        "status": prefix + ("_COMPLETED" if not_run == 0 else "_PARTIAL_WITH_NOT_RUN"),
        "completed_baseline_count": completed,
        "not_run_baseline_count": not_run,
    }


def manifest_for_result_stage(
    manifest: Mapping[str, Mapping[str, str]], run_mode: str, result_stage: str
) -> tuple[dict[str, Mapping[str, str]], int]:
    if run_mode == "FIXED_GROUPED_SPLIT":
        _require(
            result_stage in {"HPO_VALIDATION_ONLY", "FROZEN_DEVELOPMENT_TEST"},
            f"invalid result_stage for fixed split: {result_stage}",
        )
        if result_stage == "HPO_VALIDATION_ONLY":
            withheld = sum(value["split"] == "TEST" for value in manifest.values())
            _require(withheld > 0, "Development test split is empty")
            return {
                record_id: value for record_id, value in manifest.items()
                if value["split"] != "TEST"
            }, withheld
        return dict(manifest), 0
    _require(run_mode == "LOSO_FROZEN_PARAMETERS", f"unknown run mode: {run_mode}")
    _require(
        result_stage == "LOSO_FROZEN_PARAMETERS",
        f"invalid result_stage for LOSO: {result_stage}",
    )
    return dict(manifest), 0


def execute(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"output directory already exists: {output_dir}")
    _require(config["evaluation_outcomes_accessed"] is False, "classical Development baselines accessed Evaluation")
    _require(config["cpu_thread_cap"] <= 4, "classical baseline CPU cap exceeds Route 2 budget")
    cpu_thread_cap = int(config["cpu_thread_cap"])
    _require(cpu_thread_cap > 0, "classical baseline CPU cap must be positive")
    thread_controller = threadpool_limits(limits=cpu_thread_cap)
    torch.set_num_threads(cpu_thread_cap)
    device, observed_free_bytes = require_cuda_device(
        str(config["device"]),
        int(config["physical_gpu_index"]),
        int(config.get("minimum_free_gpu_memory_bytes", 0)),
    )
    cuda_provenance = cuda_device_observation(int(config["physical_gpu_index"]))
    manifest = load_manifest(Path(config["development_manifest_path"]))
    run_mode = str(config.get("run_mode", "FIXED_GROUPED_SPLIT"))
    _require(run_mode in {"FIXED_GROUPED_SPLIT", "LOSO_FROZEN_PARAMETERS"}, f"unknown run mode: {run_mode}")
    result_stage = str(config.get("result_stage", ""))
    manifest, development_test_record_count_withheld = manifest_for_result_stage(
        manifest, run_mode, result_stage
    )
    records = load_records([Path(path) for path in config["canonical_paths"]], manifest)
    loso_holdout = None
    excluded_bridge_count = 0
    if run_mode == "FIXED_GROUPED_SPLIT":
        train = [record for record in records if record["split"] == "TRAIN"]
        validation = [record for record in records if record["split"] == "VALIDATION"]
        test = [record for record in records if record["split"] == "TEST"]
        _require(train and validation, "Development train/validation split is incomplete")
        _require(
            (result_stage == "HPO_VALIDATION_ONLY" and not test and development_test_record_count_withheld > 0)
            or (result_stage == "FROZEN_DEVELOPMENT_TEST" and bool(test)),
            "Development test exposure differs from result_stage",
        )
    else:
        loso_holdout = str(config["loso_holdout_study_unit_id"])
        holdout_components = {
            value["connected_source_component_id"]
            for value in manifest.values() if value["study_unit_id"] == loso_holdout
        }
        _require(holdout_components, f"LOSO study is absent: {loso_holdout}")
        train = [
            record for record in records
            if record["study_unit_id"] != loso_holdout
            and record["connected_source_component_id"] not in holdout_components
        ]
        validation = []
        test = [record for record in records if record["study_unit_id"] == loso_holdout]
        excluded_bridge_count = sum(
            record["study_unit_id"] != loso_holdout
            and record["connected_source_component_id"] in holdout_components
            for record in records
        )
        _require(train and test, "LOSO train or holdout set is empty")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    temporary = output_dir
    serialized_config = json.dumps(dict(config), indent=2, sort_keys=True) + "\n"
    (temporary / "run_config.json").write_text(serialized_config, encoding="utf-8")
    (temporary / "config.yaml").write_text(serialized_config, encoding="utf-8")
    log_path = temporary / "train.log"
    metrics_path = temporary / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    def progress(row: Mapping[str, Any]) -> None:
        serialized = json.dumps(dict(row), sort_keys=True) + "\n"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(serialized)
        if row.get("event") in {"HPO_TRIAL_COMPLETED", "FROZEN_BASELINE_COMPLETED", "LOSO_BASELINE_COMPLETED"}:
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(serialized)

    progress({
        "event": "RUN_STARTED",
        "device": str(device),
        "physical_gpu_index": int(config["physical_gpu_index"]),
        "cuda_device_uuid": cuda_provenance["cuda_device_uuid"],
        "result_stage": result_stage,
        "run_mode": run_mode,
    })
    baseline_results: dict[str, Any] = {}
    try:
        for baseline in config["baselines"]:
            baseline_id = baseline["baseline_id"]
            torch.cuda.reset_peak_memory_stats(device)
            if run_mode == "LOSO_FROZEN_PARAMETERS":
                _require("frozen_parameters" in baseline, f"LOSO parameters are not frozen: {baseline_id}")
                selected_parameters = baseline["frozen_parameters"]
                started = time.monotonic()
                try:
                    test_predictions, artifact = _fit_predict(baseline, train, test, selected_parameters, device)
                    torch.cuda.synchronize(device)
                    baseline_dir = temporary / baseline_id
                    baseline_dir.mkdir()
                    _write_predictions(baseline_dir / "loso_predictions.jsonl", baseline_id, test, test_predictions)
                    joblib.dump(bind_artifact_identity(artifact, baseline), baseline_dir / "model.joblib")
                    result = {
                        "status": "COMPLETED_DEVELOPMENT_LOSO",
                        "holdout_study_unit_id": loso_holdout,
                        "frozen_parameters": selected_parameters,
                        "holdout_mae": _mae(test, test_predictions),
                        "holdout_spearman": _spearman(test, test_predictions),
                        "training_record_count": len(train),
                        "holdout_record_count": len(test),
                        "hpo_trial_count": 0,
                        "wall_seconds": time.monotonic() - started,
                        "evaluation_outcomes_accessed": False,
                        "execution_provenance": {
                            "parameter_fit_execution": "CUDA",
                            "prediction_execution": "CUDA",
                            "device": str(device),
                            "physical_gpu_index": int(config["physical_gpu_index"]),
                            "cpu_fallback_used": False,
                            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                        },
                    }
                    (baseline_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    baseline_results[baseline_id] = result
                    progress({"event": "LOSO_BASELINE_COMPLETED", "baseline_id": baseline_id, **result})
                except GPUExecutionError:
                    raise
                except BaselineError as exc:
                    baseline_results[baseline_id] = {
                        "status": "NOT_RUN", "reason": str(exc),
                        "holdout_study_unit_id": loso_holdout,
                    }
                continue
            if result_stage == "FROZEN_DEVELOPMENT_TEST":
                _require("frozen_parameters" in baseline, f"Development test parameters are not frozen: {baseline_id}")
                selected_parameters = baseline["frozen_parameters"]
                started = time.monotonic()
                try:
                    fit_records = train + validation
                    test_predictions, artifact = _fit_predict(
                        baseline, fit_records, test, selected_parameters, device
                    )
                    torch.cuda.synchronize(device)
                    baseline_dir = temporary / baseline_id
                    baseline_dir.mkdir()
                    _write_predictions(
                        baseline_dir / "test_predictions.jsonl", baseline_id, test, test_predictions
                    )
                    joblib.dump(bind_artifact_identity(artifact, baseline), baseline_dir / "model.joblib")
                    result = {
                        "status": "COMPLETED_FROZEN_DEVELOPMENT_TEST",
                        "frozen_parameters": selected_parameters,
                        "test_mae": _mae(test, test_predictions),
                        "test_spearman": _spearman(test, test_predictions),
                        "training_record_count": len(fit_records),
                        "test_record_count": len(test),
                        "hpo_trial_count": 0,
                        "wall_seconds": time.monotonic() - started,
                        "evaluation_outcomes_accessed": False,
                        "execution_provenance": {
                            "parameter_fit_execution": "CUDA",
                            "prediction_execution": "CUDA",
                            "device": str(device),
                            "physical_gpu_index": int(config["physical_gpu_index"]),
                            "cpu_fallback_used": False,
                            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                            **cuda_provenance,
                        },
                    }
                    (baseline_dir / "result.json").write_text(
                        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                    )
                    baseline_results[baseline_id] = result
                    progress({"event": "FROZEN_BASELINE_COMPLETED", "baseline_id": baseline_id, **result})
                except GPUExecutionError:
                    raise
                except BaselineError as exc:
                    baseline_results[baseline_id] = {"status": "NOT_RUN", "reason": str(exc)}
                continue
            trial_results = []
            for parameters in _parameter_trials(baseline):
                started = time.monotonic()
                try:
                    predictions, _artifact = _fit_predict(baseline, train, validation, parameters, device)
                    result = {
                        "parameters": parameters,
                        "validation_mae": _mae(validation, predictions),
                        "validation_task_macro_spearman": _task_macro_spearman(validation, predictions),
                        "validation_pooled_spearman": _spearman(validation, predictions),
                        "wall_seconds": time.monotonic() - started,
                        "status": "COMPLETED",
                    }
                except GPUExecutionError:
                    raise
                except BaselineError as exc:
                    result = {
                        "parameters": parameters,
                        "status": "NOT_RUN",
                        "reason": str(exc),
                        "wall_seconds": time.monotonic() - started,
                    }
                trial_results.append(result)
                progress({"event": "HPO_TRIAL_COMPLETED", "baseline_id": baseline_id, **result})
            completed_trials = [result for result in trial_results if result["status"] == "COMPLETED"]
            if not completed_trials:
                baseline_results[baseline_id] = {"status": "NOT_RUN", "trials": trial_results}
                continue
            finite_spearman_trials = [
                result for result in completed_trials
                if result["validation_task_macro_spearman"] is not None
            ]
            if finite_spearman_trials:
                selected_trial = min(
                    finite_spearman_trials,
                    key=lambda result: (
                        -result["validation_task_macro_spearman"],
                        result["validation_mae"],
                        json.dumps(result["parameters"], sort_keys=True),
                    ),
                )
                selection_metric = "DEVELOPMENT_VALIDATION_TASK_MACRO_SPEARMAN"
            else:
                selected_trial = min(
                    completed_trials,
                    key=lambda result: (
                        result["validation_mae"],
                        json.dumps(result["parameters"], sort_keys=True),
                    ),
                )
                selection_metric = "DEVELOPMENT_VALIDATION_MAE_ALL_TASK_SPEARMAN_UNDEFINED"
            selected_parameters = selected_trial["parameters"]
            validation_predictions, _validation_artifact = _fit_predict(
                baseline, train, validation, selected_parameters, device
            )
            artifact = _validation_artifact
            torch.cuda.synchronize(device)
            baseline_dir = temporary / baseline_id
            baseline_dir.mkdir()
            _write_predictions(
                baseline_dir / "validation_predictions.jsonl",
                baseline_id,
                validation,
                validation_predictions,
            )
            joblib.dump(bind_artifact_identity(artifact, baseline), baseline_dir / "model.joblib")
            result = {
                "status": "COMPLETED_DEVELOPMENT_VALIDATION_ONLY",
                "selected_parameters": selected_parameters,
                "selection_metric": selection_metric,
                "selected_validation_mae": _mae(validation, validation_predictions),
                "selected_validation_task_macro_spearman": _task_macro_spearman(validation, validation_predictions),
                "selected_validation_pooled_spearman": _spearman(validation, validation_predictions),
                "training_record_count": len(train),
                "development_test_record_count_withheld": development_test_record_count_withheld,
                "hpo_trial_count": len(trial_results),
                "trials": trial_results,
                "evaluation_outcomes_accessed": False,
                "execution_provenance": {
                    "parameter_fit_execution": "CUDA",
                    "prediction_execution": "CUDA",
                    "device": str(device),
                    "physical_gpu_index": int(config["physical_gpu_index"]),
                    "cpu_fallback_used": False,
                    "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                    **cuda_provenance,
                },
            }
            (baseline_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            baseline_results[baseline_id] = result
        completion = _completion_summary(baseline_results, run_mode)
        summary = {
            "schema_version": "route_a_v3_route2_classical_prediction_baselines.v1",
            **completion,
            "run_mode": run_mode,
            "result_stage": result_stage,
            "development_test_outcomes_evaluated": result_stage == "FROZEN_DEVELOPMENT_TEST",
            "development_test_record_count_withheld": development_test_record_count_withheld,
            "loso_holdout_study_unit_id": loso_holdout,
            "loso_excluded_connected_other_study_record_count": excluded_bridge_count,
            "development_record_counts": {"TRAIN": len(train), "VALIDATION": len(validation), "TEST": len(test)},
            "cpu_thread_cap": config["cpu_thread_cap"],
            "device": str(device),
            "physical_gpu_index": int(config["physical_gpu_index"]),
            "cpu_fallback_used": False,
            "observed_free_gpu_memory_bytes_before_run": observed_free_bytes,
            **cuda_provenance,
            "evaluation_outcomes_accessed": False,
            "baselines": baseline_results,
            "scientific_claim_status": "NOT_ESTABLISHED",
        }
        (temporary / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "final_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        progress({"event": "RUN_COMPLETED", "completed_baseline_count": completion["completed_baseline_count"]})
        return summary
    finally:
        thread_controller.restore_original_limits()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or Path(config["output_directory"])
    try:
        summary = execute(config, output_dir)
    except Exception as exc:
        write_gpu_failure_evidence(
            output_dir.with_name(output_dir.name + ".failed.json"), config, exc,
            entrypoint="run_route2_classical_prediction_baselines_v1",
            evaluation_outcomes_accessed=config.get("evaluation_outcomes_accessed"),
        )
        raise
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
