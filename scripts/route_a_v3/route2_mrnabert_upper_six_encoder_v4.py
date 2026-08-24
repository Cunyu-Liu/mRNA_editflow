"""Trainable pretrained mRNABERT blocks 6–11 for XEditCritic V4."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from scripts.route_a_v3.route2_mrnabert_edit_site_encoder_v3 import (
    ATTENTION_BACKENDS,
    pytorch_sdpa_qkvpacked,
)


class MRNABERTUpperSixEncoderV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MRNABERTUpperSixEncoderV4Error(message)


def additive_attention_mask_v4(
    attention_mask: torch.Tensor, *, dtype: torch.dtype
) -> torch.Tensor:
    """Convert active-token booleans to the additive BERT attention mask."""

    _require(attention_mask.ndim == 2, "upper-six attention mask must be chunk x token")
    active = attention_mask.to(dtype=dtype)
    return (1.0 - active[:, None, None, :]) * torch.finfo(dtype).min


def forward_upper_six_layers_v4(
    layers: Sequence[nn.Module],
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    activation_checkpointing: bool = False,
) -> torch.Tensor:
    """Run exactly six pretrained upper blocks on cached bottom-six hidden."""

    _require(len(layers) == 6, "formal V4 upper encoder must contain exactly blocks 6–11")
    _require(hidden.ndim == 3 and attention_mask.shape == hidden.shape[:2], "upper-six hidden/mask geometry changed")
    extended = additive_attention_mask_v4(attention_mask, dtype=hidden.dtype)
    for layer in layers:
        def layer_forward(
            values: torch.Tensor,
            active_layer: nn.Module = layer,
        ) -> torch.Tensor:
            return active_layer(
                values,
                attention_mask=extended,
                head_mask=None,
                output_attentions=False,
            )[0]

        hidden = (
            checkpoint(
                layer_forward,
                hidden,
                use_reentrant=False,
                preserve_rng_state=True,
            )
            if activation_checkpointing and torch.is_grad_enabled()
            else layer_forward(hidden)
        )
    _require(torch.isfinite(hidden).all().item(), "upper-six hidden is nonfinite")
    return hidden


class TrainableMRNABERTUpperSixEncoderV4(nn.Module):
    """Load only pretrained blocks 6–11 as the formal trainable upper encoder."""

    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        *,
        attention_backend: str = "PYTORCH_SDPA_AUTO",
        activation_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        _require(device.type == "cuda", "formal V4 upper-six encoder requires CUDA")
        _require(torch.cuda.is_available(), "CUDA is unavailable")
        _require(model_path.is_dir(), "mRNABERT model directory is absent")
        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as exc:
            raise MRNABERTUpperSixEncoderV4Error("Transformers is unavailable") from exc
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        configuration = AutoConfig.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        model = AutoModel.from_config(
            configuration,
            trust_remote_code=True,
            add_pooling_layer=False,
        )
        backend = str(attention_backend)
        _require(backend in ATTENTION_BACKENDS, "unknown upper-six attention backend")
        modeling_module = sys.modules[model.__class__.__module__]
        modeling_module.flash_attn_qkvpacked_func = (
            None if backend == "OFFICIAL_PYTORCH_FALLBACK" else pytorch_sdpa_qkvpacked
        )
        checkpoint = torch.load(
            model_path / "pytorch_model.bin",
            map_location="cpu",
            weights_only=False,
        )
        base_state = {
            key.removeprefix("bert."): value
            for key, value in checkpoint.items()
            if key.startswith("bert.")
        }
        _require(bool(base_state), "mRNABERT base encoder weights are absent")
        model.load_state_dict(base_state, strict=True)
        del checkpoint, base_state
        _require(len(model.encoder.layer) == 12, "mRNABERT encoder depth changed")
        self.layers = nn.ModuleList(list(model.encoder.layer[6:12]))
        self.layers.requires_grad_(True)
        self.device = device
        self.attention_backend = backend
        self.activation_checkpointing = bool(activation_checkpointing)
        self.embedding_width = int(configuration.hidden_size)
        _require(self.embedding_width == 768, "mRNABERT upper-six width changed")
        self.to(device)
        _require(
            all(parameter.requires_grad for parameter in self.parameters()),
            "a formal mRNABERT upper-six parameter is frozen",
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        _require(hidden.device == self.device, "upper-six hidden is on the wrong CUDA device")
        return forward_upper_six_layers_v4(
            self.layers,
            hidden,
            attention_mask,
            activation_checkpointing=self.training and self.activation_checkpointing,
        )

    def scope_summary(self) -> dict[str, object]:
        parameter_names = [name for name, _ in self.named_parameters()]
        _require(parameter_names and all(name.startswith("layers.") for name in parameter_names), "non-upper parameter entered the formal upper-six adapter")
        return {
            "source_blocks": [6, 7, 8, 9, 10, 11],
            "embedding_layer_loaded": False,
            "bottom_six_loaded": False,
            "all_retained_parameters_trainable": True,
            "trainable_parameter_count": self.trainable_parameter_count,
            "parameter_name_count": len(parameter_names),
            "attention_backend": self.attention_backend,
            "activation_checkpointing": self.activation_checkpointing,
        }
