"""Differentiable last-four-LoRA mRNABERT edit features for Critic V3 C3."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from core.route2_mrnabert_edit_site_features_v3 import LOCAL_RADIUS
from core.route2_mrnabert_lora_v3 import (
    LoRALinearV3,
    MRNABERTLoRAInstallationV3,
    disabled_lora_residuals_v3,
    install_last_four_mrnabert_lora_v3,
)


class MRNABERTLoRAEncoderError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MRNABERTLoRAEncoderError(message)


def populate_edit_features_from_hidden_v3(
    batch: Mapping[str, Any],
    *,
    source_hidden: torch.Tensor,
    candidate_hidden: torch.Tensor,
    source_attention_mask: torch.Tensor,
    candidate_attention_mask: torch.Tensor,
    local_radius: int = LOCAL_RADIUS,
) -> dict[str, Any]:
    """Apply the same one-leading-special and radius-16 rules as the cache."""

    _require(source_hidden.shape == candidate_hidden.shape, "paired hidden geometry differs")
    _require(source_hidden.ndim == 3, "paired hidden must be batch x token x hidden")
    _require(source_attention_mask.shape == candidate_attention_mask.shape, "paired attention masks differ")
    result = dict(batch)
    active_source = source_attention_mask.to(source_hidden.dtype).unsqueeze(-1)
    active_candidate = candidate_attention_mask.to(candidate_hidden.dtype).unsqueeze(-1)
    result["source_global"] = (
        source_hidden * active_source
    ).sum(dim=1) / active_source.sum(dim=1).clamp_min(1)
    result["candidate_global"] = (
        candidate_hidden * active_candidate
    ).sum(dim=1) / active_candidate.sum(dim=1).clamp_min(1)
    batch_size, maximum_edits = batch["edit_padding_mask"].shape
    feature_lists: dict[str, list[torch.Tensor]] = {
        f"{side}_{name}": []
        for side in ("source", "candidate")
        for name in ("site", "window_mean", "window_max")
    }
    sequence_lengths = (~batch["padding_mask"]).sum(dim=1)
    for batch_index in range(batch_size):
        per_record = {key: [] for key in feature_lists}
        length = int(sequence_lengths[batch_index].item())
        for edit_index in range(maximum_edits):
            if bool(batch["edit_padding_mask"][batch_index, edit_index].item()):
                for key in per_record:
                    per_record[key].append(source_hidden[batch_index, 0] * 0.0)
                continue
            position = int(batch["edit_positions"][batch_index, edit_index].item())
            _require(0 <= position < length, "edit position is outside the online sequence")
            start = max(0, position - local_radius)
            end = min(length, position + local_radius + 1)
            for side, hidden in (
                ("source", source_hidden),
                ("candidate", candidate_hidden),
            ):
                # Token zero is CLS; nucleotide p is token p+1; SEP follows the
                # last nucleotide.  Local windows never include either special.
                site = hidden[batch_index, position + 1]
                window = hidden[batch_index, start + 1 : end + 1]
                per_record[f"{side}_site"].append(site)
                per_record[f"{side}_window_mean"].append(window.mean(dim=0))
                per_record[f"{side}_window_max"].append(window.max(dim=0).values)
        for key in feature_lists:
            feature_lists[key].append(torch.stack(per_record[key]))
    for key, records in feature_lists.items():
        result[key] = torch.stack(records)
    return result


_ANCHORED_FEATURE_NAMES = (
    "source_site", "candidate_site",
    "source_window_mean", "candidate_window_mean",
    "source_window_max", "candidate_window_max",
    "source_global", "candidate_global",
)


def anchor_online_lora_delta_to_cached_features_v3(
    cached_batch: Mapping[str, Any],
    adapted_online: Mapping[str, Any],
    zero_lora_online: Mapping[str, Any],
) -> dict[str, Any]:
    """Use the cache as base and retain only the differentiable LoRA delta."""

    result = dict(adapted_online)
    for name in _ANCHORED_FEATURE_NAMES:
        cached = cached_batch[name].to(
            device=adapted_online[name].device,
            dtype=adapted_online[name].dtype,
        )
        _require(cached.shape == adapted_online[name].shape == zero_lora_online[name].shape, f"anchored feature geometry differs: {name}")
        result[name] = cached + (
            adapted_online[name] - zero_lora_online[name].detach()
        )
    return result


class TrainableMRNABERTEditSiteEncoderV3(nn.Module):
    """Frozen mRNABERT plus the prospectively fixed last-four LoRA modules."""

    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        *,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
        local_radius: int = LOCAL_RADIUS,
    ) -> None:
        super().__init__()
        _require(device.type == "cuda", "C3 online encoder requires CUDA")
        _require(model_path.is_dir(), "mRNABERT model directory is absent")
        try:
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except ImportError as exc:
            raise MRNABERTLoRAEncoderError("Transformers is unavailable") from exc
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        config = AutoConfig.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True
        )
        model = AutoModel.from_config(
            config, trust_remote_code=True, add_pooling_layer=False
        )
        modeling_module = sys.modules[model.__class__.__module__]
        modeling_module.flash_attn_qkvpacked_func = None
        checkpoint = torch.load(
            model_path / "pytorch_model.bin", map_location="cpu", weights_only=False
        )
        base_state = {
            key.removeprefix("bert."): value
            for key, value in checkpoint.items()
            if key.startswith("bert.")
        }
        _require(bool(base_state), "mRNABERT base weights are absent")
        model.load_state_dict(base_state, strict=True)
        del checkpoint, base_state
        self.lora_installation: MRNABERTLoRAInstallationV3 = (
            install_last_four_mrnabert_lora_v3(
                model, rank=rank, alpha=alpha, dropout=dropout
            )
        )
        self.model = model.to(device)
        self.device = device
        self.local_radius = int(local_radius)
        self.embedding_width = int(model.config.hidden_size)
        _require(self.embedding_width == 768, "mRNABERT hidden width changed")
        base_tokens = tokenizer(
            "A C G T", add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        _require(len(base_tokens) == 4 and len(set(base_tokens)) == 4, "mRNABERT base token mapping changed")
        _require(tokenizer.cls_token_id is not None and tokenizer.sep_token_id is not None, "mRNABERT special tokens are absent")
        _require(tokenizer.pad_token_id is not None, "mRNABERT padding token is absent")
        self.register_buffer(
            "base_token_ids",
            torch.tensor(base_tokens, dtype=torch.long, device=device),
            persistent=True,
        )
        self.cls_token_id = int(tokenizer.cls_token_id)
        self.sep_token_id = int(tokenizer.sep_token_id)
        self.pad_token_id = int(tokenizer.pad_token_id)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def lora_parameters(self) -> list[nn.Parameter]:
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        _require(
            sum(parameter.numel() for parameter in parameters)
            == self.lora_installation.trainable_parameter_count,
            "online encoder trainable parameter count differs from LoRA installation",
        )
        return parameters

    def train(self, mode: bool = True):
        """Keep the frozen encoder deterministic; train only LoRA dropout."""

        self.training = bool(mode)
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, LoRALinearV3):
                module.train(mode)
        return self

    def _model_inputs(
        self, tokens: torch.Tensor, padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _require(tokens.ndim == padding_mask.ndim == 2, "online tokens/mask must be matrices")
        batch_size, maximum_length = tokens.shape
        _require(maximum_length <= 1000, "Development C3 sequence exceeds the frozen single-chunk geometry")
        lengths = (~padding_mask).sum(dim=1)
        input_ids = torch.full(
            (batch_size, maximum_length + 2),
            self.pad_token_id,
            dtype=torch.long,
            device=tokens.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        input_ids[:, 0] = self.cls_token_id
        attention_mask[:, 0] = 1
        safe_tokens = tokens.clamp(min=0, max=3)
        mapped = self.base_token_ids[safe_tokens]
        input_ids[:, 1 : maximum_length + 1] = torch.where(
            padding_mask,
            torch.full_like(mapped, self.pad_token_id),
            mapped,
        )
        attention_mask[:, 1 : maximum_length + 1] = (~padding_mask).long()
        for index, length in enumerate(lengths.tolist()):
            input_ids[index, int(length) + 1] = self.sep_token_id
            attention_mask[index, int(length) + 1] = 1
        token_type_ids = torch.zeros_like(input_ids)
        return input_ids, attention_mask, token_type_ids

    def _online_features(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        paired_tokens = torch.cat(
            (batch["source_tokens"], batch["candidate_tokens"]), dim=0
        )
        paired_padding = torch.cat((batch["padding_mask"], batch["padding_mask"]), dim=0)
        input_ids, attention_mask, token_type_ids = self._model_inputs(
            paired_tokens, paired_padding
        )
        hidden = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[0]
        batch_size = batch["source_tokens"].shape[0]
        source_hidden = hidden[:batch_size]
        candidate_hidden = hidden[batch_size:]
        identity = torch.all(
            batch["source_tokens"] == batch["candidate_tokens"], dim=1
        )
        candidate_hidden = torch.where(
            identity[:, None, None], source_hidden, candidate_hidden
        )
        return populate_edit_features_from_hidden_v3(
            batch,
            source_hidden=source_hidden,
            candidate_hidden=candidate_hidden,
            source_attention_mask=attention_mask[:batch_size],
            candidate_attention_mask=attention_mask[batch_size:],
            local_radius=self.local_radius,
        )

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        return self._online_features(batch)

    def forward_cache_anchored(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        _require(
            all(name in batch for name in _ANCHORED_FEATURE_NAMES),
            "cache-anchored online encoding lacks cached base features",
        )
        with torch.no_grad(), disabled_lora_residuals_v3(self.model):
            zero_lora_online = self._online_features(batch)
        adapted_online = self._online_features(batch)
        return anchor_online_lora_delta_to_cached_features_v3(
            batch, adapted_online, zero_lora_online
        )
