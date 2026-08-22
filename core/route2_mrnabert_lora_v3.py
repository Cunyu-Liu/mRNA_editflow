"""Frozen last-four-block LoRA policy for the XEditCritic V3 C3 arm."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class MRNABERTLoRAError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MRNABERTLoRAError(message)


class LoRALinearV3(nn.Module):
    """A frozen Linear plus a rank-limited trainable residual."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        _require(type(base) is nn.Linear, "LoRA target must be an exact Linear module")
        _require(rank > 0 and alpha > 0.0, "LoRA rank/alpha is invalid")
        _require(0.0 <= dropout < 1.0, "LoRA dropout is invalid")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.dropout = float(dropout)
        self.scaling = self.alpha / self.rank
        self.lora_a = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual_input = F.dropout(values, p=self.dropout, training=self.training)
        residual = F.linear(F.linear(residual_input, self.lora_a), self.lora_b)
        return self.base(values) + self.scaling * residual


@dataclass(frozen=True)
class MRNABERTLoRAInstallationV3:
    layer_indices: tuple[int, ...]
    module_names: tuple[str, ...]
    rank: int
    alpha: float
    dropout: float
    trainable_parameter_count: int


def install_last_four_mrnabert_lora_v3(
    model: nn.Module,
    *,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
) -> MRNABERTLoRAInstallationV3:
    """Freeze mRNABERT and adapt only QKV/O/gated-FFN/down-FFN projections."""

    _require(hasattr(model, "encoder") and hasattr(model.encoder, "layer"), "mRNABERT encoder layers are absent")
    layers = model.encoder.layer
    _require(len(layers) >= 4, "mRNABERT has fewer than four blocks")
    model.requires_grad_(False)
    layer_indices = tuple(range(len(layers) - 4, len(layers)))
    installed_names: list[str] = []
    targets = (
        ("attention.self.Wqkv", lambda layer: layer.attention.self, "Wqkv"),
        ("attention.output.dense", lambda layer: layer.attention.output, "dense"),
        ("mlp.gated_layers", lambda layer: layer.mlp, "gated_layers"),
        ("mlp.wo", lambda layer: layer.mlp, "wo"),
    )
    for layer_index in layer_indices:
        layer = layers[layer_index]
        for relative_name, parent_getter, attribute in targets:
            parent = parent_getter(layer)
            base = getattr(parent, attribute)
            _require(type(base) is nn.Linear, f"unexpected LoRA target: layer {layer_index} {relative_name}")
            setattr(
                parent,
                attribute,
                LoRALinearV3(base, rank=rank, alpha=alpha, dropout=dropout),
            )
            installed_names.append(f"encoder.layer.{layer_index}.{relative_name}")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    _require(bool(trainable), "LoRA installation produced no trainable parameters")
    _require(
        all(
            name.endswith((".lora_a", ".lora_b"))
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ),
        "a non-LoRA mRNABERT parameter remains trainable",
    )
    return MRNABERTLoRAInstallationV3(
        layer_indices=layer_indices,
        module_names=tuple(installed_names),
        rank=int(rank),
        alpha=float(alpha),
        dropout=float(dropout),
        trainable_parameter_count=sum(parameter.numel() for parameter in trainable),
    )
