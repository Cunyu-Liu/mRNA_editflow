"""Shared model and metric runtime for XEditSetFlow V3 training arms."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from core.route2_base_flow_model import Route2BaseFlowModel
from core.route2_xeditsetflow_v3 import XEditSetFlowV3, set_marginal_negative_log_likelihood_v1


class XEditSetFlowRuntimeV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditSetFlowRuntimeV3Error(message)


ARM_CONFIGS_V3: dict[str, dict[str, Any]] = {
    "f1": {
        "model_kind": "V2_TWO_CONV_SET_MARGINAL_DIAGNOSTIC",
        "hidden_dim": 256,
        "depth": 2,
        "selectable": False,
    },
    "f2": {
        "model_kind": "XEDITSETFLOW_V3_HYBRID",
        "model_width": 384,
        "depth": 8,
        "heads": 8,
        "ffn_width": 1536,
        "selectable": True,
    },
    "f3": {
        "model_kind": "XEDITSETFLOW_V3_HYBRID",
        "model_width": 512,
        "depth": 12,
        "heads": 8,
        "ffn_width": 2048,
        "selectable": True,
    },
}


def build_setflow_arm_v3(
    arm: str,
    *,
    vocabs: Mapping[str, Mapping[str, int]],
    dropout: float,
) -> tuple[nn.Module, dict[str, Any]]:
    _require(arm in ARM_CONFIGS_V3, "unknown SetFlow V3 arm")
    frozen = dict(ARM_CONFIGS_V3[arm])
    if arm == "f1":
        model = Route2BaseFlowModel(
            hidden_dim=256,
            assay_count=len(vocabs["assay"]),
            context_count=len(vocabs["context"]),
            position_progress_features=True,
        )
        model_config = {
            **frozen,
            "assay_count": len(vocabs["assay"]),
            "context_count": len(vocabs["context"]),
            "position_progress_features": True,
        }
    else:
        model = XEditSetFlowV3(
            model_width=int(frozen["model_width"]),
            depth=int(frozen["depth"]),
            heads=int(frozen["heads"]),
            ffn_width=int(frozen["ffn_width"]),
            assay_count=len(vocabs["assay"]),
            context_count=len(vocabs["context"]),
            quantity_count=len(vocabs["quantity"]),
            measurement_count=len(vocabs["measurement"]),
            numerator_count=len(vocabs["numerator"]),
            denominator_count=len(vocabs["denominator"]),
            dropout=float(dropout),
        )
        model_config = {
            **frozen,
            "assay_count": len(vocabs["assay"]),
            "context_count": len(vocabs["context"]),
            "quantity_count": len(vocabs["quantity"]),
            "measurement_count": len(vocabs["measurement"]),
            "numerator_count": len(vocabs["numerator"]),
            "denominator_count": len(vocabs["denominator"]),
            "dropout": float(dropout),
            "pretrained_width": 768,
            "local_attention_window": 64,
        }
    model_config["trainable_parameter_count"] = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return model, model_config


def setflow_arm_rates_v3(
    model: nn.Module, arm: str, batch: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    if arm == "f1":
        _require(isinstance(model, Route2BaseFlowModel), "F1 model class changed")
        return model.rates(
            batch["source_tokens"],
            batch["current_tokens"],
            batch["padding_mask"],
            batch["region_ids"],
            batch["assay_ids"],
            batch["context_ids"],
            batch["remaining_budget"],
        )
    _require(isinstance(model, XEditSetFlowV3), "F2/F3 model class changed")
    return model.rates(batch)


def setflow_batch_loss_v3(
    model: nn.Module, arm: str, batch: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    rates, legal = setflow_arm_rates_v3(model, arm, batch)
    loss = set_marginal_negative_log_likelihood_v1(
        rates,
        legal,
        batch["positive_action_mask"],
        batch["structural_budget_exhausted"],
        sample_weight=batch["sample_weight"],
    )
    active_weight = batch["sample_weight"][
        ~batch["structural_budget_exhausted"]
    ].sum()
    _require(float(active_weight.item()) > 0.0, "SetFlow batch has no active loss weight")
    return loss, active_weight


def early_stop_update_v3(
    value: float,
    *,
    best: float | None,
    stale_passes: int,
    patience: int,
) -> tuple[bool, float, int, bool]:
    _require(torch.isfinite(torch.tensor(value)).item(), "validation common NLL is nonfinite")
    _require(patience >= 1 and stale_passes >= 0, "early-stop state is invalid")
    improved = best is None or value < best
    next_best = value if improved else float(best)
    next_stale = 0 if improved else stale_passes + 1
    return improved, next_best, next_stale, next_stale >= patience
