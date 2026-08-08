"""Deterministic training primitives for R3X ordinary critic candidates.

This module intentionally contains no real-data CLI.  The ordinary runner must
complete protocol activation before it may construct a model or call these
functions.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


SIGN_GATE = 0.60
SELECTION_RULE = "SIGN_GATE_THEN_SPEARMAN_THEN_NLL_THEN_EARLIEST_EPOCH"


class TrainingProtocolError(ValueError):
    """Raised when training semantics or frozen policy are violated."""


def _tensor_vector(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise TrainingProtocolError(f"{name} must be a one-dimensional tensor")
    if not torch.isfinite(value).all():
        raise TrainingProtocolError(f"{name} contains non-finite values")
    return value


def normal_sign_probability(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    *,
    temperature: float = 1.0,
    detach_variance: bool = False,
) -> torch.Tensor:
    """Gaussian ``P(delta>0) = Phi(mu/sigma/temperature)``.

    ``detach_variance`` is used only for the sign BCE term so the model cannot
    improve that term by shrinking variance while leaving the hard sign fixed.
    """

    mean = _tensor_vector(mean, "mean")
    logvar = _tensor_vector(logvar, "logvar")
    if mean.shape != logvar.shape:
        raise TrainingProtocolError("mean and logvar must align")
    if not isinstance(temperature, (int, float)) or not math.isfinite(float(temperature)) or temperature <= 0:
        raise TrainingProtocolError("temperature must be finite and positive")
    variance_term = logvar.detach() if detach_variance else logvar
    sigma = torch.exp(0.5 * torch.clamp(variance_term, min=-20.0, max=20.0))
    z = mean / (sigma * float(temperature))
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def hard_sign(mean: torch.Tensor, *, temperature: float | None = None) -> torch.Tensor:
    """Frozen hard sign.  Positive temperature calibration cannot change it."""

    mean = _tensor_vector(mean, "mean")
    if temperature is not None and (
        not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0
    ):
        raise TrainingProtocolError("temperature must be finite and positive")
    return torch.where(mean >= 0.0, torch.ones_like(mean), -torch.ones_like(mean))


def effect_loss(
    outputs: Mapping[str, torch.Tensor],
    true_delta: torch.Tensor,
    *,
    huber_weight: float = 1.0,
    nll_weight: float = 0.25,
    sign_weight: float = 0.25,
    sign_temperature: float = 1.0,
    sample_weight: torch.Tensor | None = None,
) -> dict:
    """Coherent mean/NLL/probit objective with a mean-derived hard sign."""

    if set(outputs).isdisjoint({"mean", "logvar"}) or "mean" not in outputs or "logvar" not in outputs:
        raise TrainingProtocolError("model outputs must contain mean and logvar")
    mean = _tensor_vector(outputs["mean"], "mean")
    logvar = _tensor_vector(outputs["logvar"], "logvar")
    target = _tensor_vector(true_delta, "true_delta")
    if mean.shape != target.shape or logvar.shape != target.shape:
        raise TrainingProtocolError("mean, logvar and target must align")
    weights = [float(huber_weight), float(nll_weight), float(sign_weight)]
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise TrainingProtocolError("loss weights must be finite and non-negative")
    if sample_weight is None:
        row_weight = torch.ones_like(target)
    else:
        row_weight = _tensor_vector(sample_weight, "sample_weight")
        if row_weight.shape != target.shape or torch.any(row_weight < 0.0) or torch.sum(row_weight) <= 0:
            raise TrainingProtocolError("sample_weight must align, be non-negative and have positive mass")
    row_weight = row_weight / torch.sum(row_weight)

    huber_rows = F.smooth_l1_loss(mean, target, reduction="none")
    bounded_logvar = torch.clamp(logvar, min=-20.0, max=20.0)
    nll_rows = 0.5 * (
        math.log(2.0 * math.pi)
        + bounded_logvar
        + (target - mean) ** 2 * torch.exp(-bounded_logvar)
    )
    nonzero = target != 0.0
    if torch.any(nonzero):
        probability = normal_sign_probability(
            mean[nonzero], logvar[nonzero],
            temperature=sign_temperature, detach_variance=True,
        ).clamp(1e-7, 1.0 - 1e-7)
        sign_target = (target[nonzero] > 0.0).to(mean.dtype)
        sign_rows = F.binary_cross_entropy(probability, sign_target, reduction="none")
        sign_row_weight = row_weight[nonzero]
        sign_row_weight = sign_row_weight / torch.sum(sign_row_weight)
        sign_term = torch.sum(sign_row_weight * sign_rows)
    else:
        sign_term = mean.sum() * 0.0

    huber_term = torch.sum(row_weight * huber_rows)
    nll_term = torch.sum(row_weight * nll_rows)
    loss = weights[0] * huber_term + weights[1] * nll_term + weights[2] * sign_term
    return {
        "loss": loss,
        "huber": huber_term,
        "gaussian_nll": nll_term,
        "sign_bce_probit": sign_term,
        "sign_probability": normal_sign_probability(mean, logvar, temperature=sign_temperature),
        "sign_decision": hard_sign(mean),
        "sign_decision_rule": "RAW_PREDICTED_DELTA_MEAN_GE_ZERO",
        "sign_variance_gradient_policy": "DETACHED_FOR_SIGN_BCE",
    }


def frozen_train_sign_scale(true_delta: torch.Tensor) -> float:
    """Freeze the alternative sign-link scale from train targets only."""

    target = _tensor_vector(true_delta, "true_delta")
    nonzero = torch.abs(target[target != 0.0])
    if len(nonzero) == 0:
        raise TrainingProtocolError(
            "fixed-scale sign candidate requires non-zero train effects"
        )
    scale = float(torch.median(nonzero).item())
    if not math.isfinite(scale) or scale <= 0.0:
        raise TrainingProtocolError("frozen train sign scale must be positive")
    return scale


def fixed_train_scale_sign_candidate_loss(
    outputs: Mapping[str, torch.Tensor],
    true_delta: torch.Tensor,
    *,
    frozen_train_scale: float,
    huber_weight: float = 1.0,
    nll_weight: float = 0.25,
    sign_weight: float = 0.25,
    sample_weight: torch.Tensor | None = None,
) -> dict:
    """Development-only alternative using BCEWithLogits(mu/tau_train).

    This is an explicitly inactive candidate for one pre-registered loss
    comparison.  It does not change the raw-mean hard sign or the 0.60 gate.
    """

    if (
        isinstance(frozen_train_scale, bool)
        or not isinstance(frozen_train_scale, (int, float))
        or not math.isfinite(float(frozen_train_scale))
        or float(frozen_train_scale) <= 0.0
    ):
        raise TrainingProtocolError("frozen_train_scale must be finite and positive")
    if "mean" not in outputs or "logvar" not in outputs:
        raise TrainingProtocolError("model outputs must contain mean and logvar")
    mean = _tensor_vector(outputs["mean"], "mean")
    logvar = _tensor_vector(outputs["logvar"], "logvar")
    target = _tensor_vector(true_delta, "true_delta")
    if mean.shape != target.shape or logvar.shape != target.shape:
        raise TrainingProtocolError("mean, logvar and target must align")
    weights = [float(huber_weight), float(nll_weight), float(sign_weight)]
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise TrainingProtocolError("loss weights must be finite and non-negative")
    if sample_weight is None:
        row_weight = torch.ones_like(target)
    else:
        row_weight = _tensor_vector(sample_weight, "sample_weight")
        if (
            row_weight.shape != target.shape
            or torch.any(row_weight < 0.0)
            or torch.sum(row_weight) <= 0
        ):
            raise TrainingProtocolError(
                "sample_weight must align, be non-negative and have positive mass"
            )
    row_weight = row_weight / torch.sum(row_weight)
    huber_rows = F.smooth_l1_loss(mean, target, reduction="none")
    bounded_logvar = torch.clamp(logvar, min=-20.0, max=20.0)
    nll_rows = 0.5 * (
        math.log(2.0 * math.pi)
        + bounded_logvar
        + (target - mean) ** 2 * torch.exp(-bounded_logvar)
    )
    nonzero = target != 0.0
    if torch.any(nonzero):
        logits = mean[nonzero] / float(frozen_train_scale)
        sign_target = (target[nonzero] > 0.0).to(mean.dtype)
        sign_rows = F.binary_cross_entropy_with_logits(
            logits, sign_target, reduction="none"
        )
        sign_row_weight = row_weight[nonzero]
        sign_row_weight = sign_row_weight / torch.sum(sign_row_weight)
        sign_term = torch.sum(sign_row_weight * sign_rows)
    else:
        sign_term = mean.sum() * 0.0
    huber_term = torch.sum(row_weight * huber_rows)
    nll_term = torch.sum(row_weight * nll_rows)
    loss = weights[0] * huber_term + weights[1] * nll_term + weights[2] * sign_term
    return {
        "loss": loss,
        "huber": huber_term,
        "gaussian_nll": nll_term,
        "sign_bce_fixed_train_scale_logit": sign_term,
        "sign_probability": torch.sigmoid(mean / float(frozen_train_scale)),
        "sign_decision": hard_sign(mean),
        "sign_decision_rule": "RAW_PREDICTED_DELTA_MEAN_GE_ZERO",
        "sign_link": "FIXED_TRAIN_MEDIAN_ABS_NONZERO_DELTA_LOGIT",
        "frozen_train_scale": float(frozen_train_scale),
        "candidate_status": "PROPOSED_NOT_ACTIVE_DEVELOPMENT_ONLY",
        "sign_accuracy_gate": SIGN_GATE,
    }


def grouped_pairwise_rank_loss(
    mean: torch.Tensor,
    true_delta: torch.Tensor,
    groups: Sequence[str],
    *,
    enabled: bool,
    policy_status: str,
) -> tuple[torch.Tensor, int]:
    """Pairwise logistic rank loss directly on predicted delta mean."""

    mean = _tensor_vector(mean, "mean")
    target = _tensor_vector(true_delta, "true_delta")
    if mean.shape != target.shape or len(groups) != len(mean):
        raise TrainingProtocolError("rank inputs must align")
    if not enabled:
        return mean.sum() * 0.0, 0
    if policy_status != "PUBLICATION_QUALIFIED":
        raise TrainingProtocolError(
            "rank loss requires a frozen PUBLICATION_QUALIFIED policy"
        )
    pairs = []
    for left in range(len(mean)):
        group = groups[left]
        if not isinstance(group, str) or not group:
            raise TrainingProtocolError("rank rows require non-empty frozen pool IDs")
        for right in range(left + 1, len(mean)):
            if groups[right] != group or target[left] == target[right]:
                continue
            direction = torch.sign(target[left] - target[right])
            pairs.append(F.softplus(-direction * (mean[left] - mean[right])))
    if not pairs:
        return mean.sum() * 0.0, 0
    return torch.stack(pairs).mean(), len(pairs)


def _cell(row: Mapping) -> tuple[str, str, str, str]:
    return (
        str(row.get("study") or ""),
        str(row.get("endpoint_id") or row.get("endpoint") or ""),
        str(row.get("publication_context_id") or ""),
        str(row.get("edit_count") or row.get("actual_edit_count") or ""),
    )


def balanced_epoch_indices(rows: Sequence[Mapping], *, seed: int) -> list[int]:
    """Equalize study-endpoint-context cells and components within each cell.

    Oversampling is deterministic for a seed and happens only inside the
    training partition supplied by the caller.
    """

    rows = list(rows)
    if not rows:
        raise TrainingProtocolError("balanced sampler refuses empty rows")
    grouped: dict[tuple[str, str, str, str], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        cell = _cell(row)
        if any(not value for value in cell):
            raise TrainingProtocolError("sampler rows require study, endpoint and context")
        component = row.get("source_component_id")
        if not isinstance(component, str) or not component:
            raise TrainingProtocolError("sampler rows require source_component_id")
        grouped[cell][component].append(index)
    rng = np.random.default_rng(int(seed))
    # Keep an epoch bounded by roughly the original row count.  Expanding every
    # singleton component to the largest component size can turn 100k rows into
    # billions of presentations.  Instead each cell receives the same bounded
    # quota and components are traversed round-robin; component counts differ by
    # at most one inside a cell.
    cell_target = int(math.ceil(len(rows) / len(grouped)))
    balanced: list[int] = []
    for cell in sorted(grouped):
        components = grouped[cell]
        component_names = sorted(components)
        per_component = {
            component: list(
                np.asarray(indices, dtype=np.int64)[rng.permutation(len(indices))]
            )
            for component, indices in components.items()
        }
        offsets = {component: 0 for component in component_names}
        component_cycle = component_names.copy()
        rng.shuffle(component_cycle)
        for draw in range(cell_target):
            component = component_cycle[draw % len(component_cycle)]
            candidates = per_component[component]
            offset = offsets[component]
            if offset and offset % len(candidates) == 0:
                rng.shuffle(candidates)
            balanced.append(int(candidates[offset % len(candidates)]))
            offsets[component] = offset + 1
    balanced_array = np.asarray(balanced, dtype=np.int64)
    balanced_array = balanced_array[rng.permutation(len(balanced_array))]
    return [int(value) for value in balanced_array]


def apply_condition_unknown_dropout(
    endpoint_ids: torch.Tensor,
    context_ids: torch.Tensor,
    *,
    rate: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Train the explicit UNK rows with deterministic training-only masking."""

    if endpoint_ids.ndim != 1 or context_ids.ndim != 1 or endpoint_ids.shape != context_ids.shape:
        raise TrainingProtocolError("condition dropout IDs must be aligned vectors")
    if not isinstance(rate, (int, float)) or isinstance(rate, bool) or not math.isfinite(float(rate)) or not 0.0 < float(rate) < 1.0:
        raise TrainingProtocolError("condition unknown dropout rate must be in (0,1)")
    rng = np.random.default_rng(int(seed))

    def mask_ids(values: torch.Tensor) -> tuple[torch.Tensor, int]:
        output = values.clone()
        eligible = torch.nonzero(values > 0, as_tuple=False).flatten().cpu().numpy()
        if len(eligible) == 0:
            return output, 0
        chosen = eligible[rng.random(len(eligible)) < float(rate)]
        if len(chosen) == 0:
            chosen = np.asarray([eligible[int(rng.integers(0, len(eligible)))]])
        output[torch.as_tensor(chosen, dtype=torch.long, device=values.device)] = 0
        return output, int(len(chosen))

    endpoint, n_endpoint = mask_ids(endpoint_ids)
    context, n_context = mask_ids(context_ids)
    return endpoint, context, {
        "policy": "DETERMINISTIC_TRAINING_ONLY_UNK_MASK",
        "rate": float(rate),
        "seed": int(seed),
        "n_endpoint_masked": n_endpoint,
        "n_context_masked": n_context,
    }


def _finite_metric(candidate: Mapping, key: str) -> float:
    value = candidate.get(key)
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise TrainingProtocolError(f"checkpoint candidate requires finite {key}")
    return float(value)


def select_checkpoint(
    candidates: Sequence[Mapping],
    *,
    sign_gate: float = SIGN_GATE,
) -> dict:
    """Select from dev epochs only under the frozen lexicographic rule."""

    candidates = list(candidates)
    if not candidates:
        raise TrainingProtocolError("checkpoint selection refuses empty candidates")
    if not math.isfinite(float(sign_gate)) or not 0.0 <= float(sign_gate) <= 1.0:
        raise TrainingProtocolError("sign_gate must be in [0, 1]")
    normalized = []
    for candidate in candidates:
        epoch = candidate.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise TrainingProtocolError("checkpoint candidate requires a non-negative integer epoch")
        normalized.append({
            "candidate": candidate,
            "epoch": int(epoch),
            "sign": _finite_metric(candidate, "macro_sign_accuracy"),
            "spearman": _finite_metric(candidate, "macro_delta_spearman"),
            "nll": _finite_metric(candidate, "macro_gaussian_nll"),
        })
    eligible = [item for item in normalized if item["sign"] >= float(sign_gate)]
    if eligible:
        # Once the gate is met, sign magnitude is not a tuning target.
        selected = max(eligible, key=lambda item: (item["spearman"], -item["nll"], -item["epoch"]))
        status = "DEV_SIGN_GATE_MET"
    else:
        # Recovery-only selection remains explicit HOLD and prioritizes sign.
        selected = max(normalized, key=lambda item: (item["sign"], item["spearman"], -item["nll"], -item["epoch"]))
        status = "HOLD_DEV_SIGN_GATE_NOT_MET"
    return {
        "selected_epoch": selected["epoch"],
        "selected_metrics": dict(selected["candidate"]),
        "status": status,
        "sign_gate": float(sign_gate),
        "selection_rule": SELECTION_RULE,
        "selection_data_role": "ORDINARY_DEV_ONLY",
        "test_based_selection": False,
        "best_seed_selection": False,
    }
