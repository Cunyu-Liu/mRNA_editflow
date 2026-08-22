from __future__ import annotations

import copy

import torch
from torch import nn

from core.route2_mrnabert_lora_v3 import LoRALinearV3, install_last_four_mrnabert_lora_v3


class _AttentionSelf(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.Wqkv = nn.Linear(8, 24)


class _AttentionOutput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(8, 8)


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self = _AttentionSelf()
        self.output = _AttentionOutput()


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gated_layers = nn.Linear(8, 64)
        self.wo = nn.Linear(32, 8)


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = _Attention()
        self.mlp = _MLP()


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.ModuleList([_Layer() for _ in range(12)])


class _MRNABERT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()
        self.embedding = nn.Embedding(5, 8)


def test_installation_only_adapts_declared_last_four_projections() -> None:
    model = _MRNABERT()
    installation = install_last_four_mrnabert_lora_v3(model)
    assert installation.layer_indices == (8, 9, 10, 11)
    assert len(installation.module_names) == 16
    assert all(
        name.endswith((".lora_a", ".lora_b"))
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    for index, layer in enumerate(model.encoder.layer):
        expected = LoRALinearV3 if index >= 8 else nn.Linear
        assert type(layer.attention.self.Wqkv) is expected
        assert type(layer.attention.output.dense) is expected
        assert type(layer.mlp.gated_layers) is expected
        assert type(layer.mlp.wo) is expected
    assert not model.embedding.weight.requires_grad


def test_zero_initialized_lora_preserves_frozen_linear_output() -> None:
    torch.manual_seed(5)
    base = nn.Linear(8, 12)
    reference = copy.deepcopy(base)
    adapted = LoRALinearV3(base).eval()
    values = torch.randn(4, 8)
    assert torch.equal(adapted(values), reference(values))


def test_real_mrnabert_geometry_has_frozen_rank16_parameter_count() -> None:
    geometries = ((768, 2304), (768, 768), (768, 6144), (3072, 768))
    per_layer = sum(16 * left + right * 16 for left, right in geometries)
    assert per_layer == 245760
    assert per_layer * 4 == 983040
