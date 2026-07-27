"""PairedDeltaFormer with explicit edits, context, uncertainty and ranking.

The three supported sequence backbones are deliberately exposed through one
interface: ``small`` (trainable task-specific encoder), ``frozen_foundation``
and ``partial_foundation``. The latter two require a local, provenance-recorded
foundation checkpoint for scientific runs. A deterministic adapter stub is
available only for shape/smoke tests and is marked in the artifact metadata.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from mrna_editflow.models.context_encoder import ContextEncoder
from mrna_editflow.models.edit_token_encoder import EditTokenEncoder
from mrna_editflow.models.uncertainty_head import UncertaintyHead


def _stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % (2**31 - 1)


class SmallTaskSequenceEncoder(nn.Module):
    """Small task-specific CNN/Transformer encoder returning token features."""

    status = "trainable_task_specific"
    is_real_foundation = False

    def __init__(self, hidden_dim: int = 128, layers: int = 2, max_len: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(5, hidden_dim, padding_idx=4)
        self.position = nn.Embedding(max_len, hidden_dim)
        self.local = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=4)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] > self.position.num_embeddings:
            raise ValueError("sequence length exceeds configured max_len")
        pos = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        x = self.embedding(tokens.long()) + self.position(pos)
        x = x + self.local(x.transpose(1, 2)).transpose(1, 2)
        x = self.encoder(x, src_key_padding_mask=~mask.bool())
        return self.norm(x) * mask.unsqueeze(-1).to(x.dtype)


class _FoundationStub(nn.Module):
    """Deterministic placeholder used only when ``allow_stub`` is explicit."""

    def __init__(self, hidden_dim: int, max_len: int, name: str):
        super().__init__()
        self.embedding = nn.Embedding(5, hidden_dim, padding_idx=4)
        self.position = nn.Embedding(max_len, hidden_dim)
        generator = torch.Generator().manual_seed(_stable_seed(name))
        with torch.no_grad():
            self.embedding.weight.copy_(torch.randn(self.embedding.weight.shape, generator=generator) * 0.02)
            self.position.weight.copy_(torch.randn(self.position.weight.shape, generator=generator) * 0.02)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        return (self.embedding(tokens.long()) + self.position(pos)) * mask.unsqueeze(-1).to(torch.float32)


class FoundationSequenceEncoder(nn.Module):
    """Local RNA-foundation adapter with explicit real/stub provenance."""

    is_real_foundation = False

    def __init__(
        self,
        hidden_dim: int = 128,
        max_len: int = 256,
        model_path: Optional[str] = None,
        freeze: bool = True,
        allow_stub: bool = False,
        name: str = "rna_foundation",
        unfreeze_last_n: int = 0,
    ):
        super().__init__()
        self.name = name
        self.model_path = str(model_path) if model_path else None
        self.status = "missing_local_checkpoint"
        self.foundation = None
        self.stage_backbone = None
        self.stage_model = None
        self.foundation_kind = None
        self.input_embedding = None
        self.output_proj = None
        if model_path and Path(model_path).exists():
            if Path(model_path).suffix in {".pt", ".pth", ".bin"}:
                try:
                    self._load_stage_a_checkpoint(model_path, hidden_dim)
                except Exception as exc:  # pragma: no cover - checkpoint-dependent.
                    self.status = f"stage_a_load_failed:{type(exc).__name__}"
            else:
                try:
                    from transformers import AutoModel
                    self.foundation = AutoModel.from_pretrained(model_path, local_files_only=True)
                    model_dim = int(self.foundation.config.hidden_size)
                    self.input_embedding = nn.Embedding(5, model_dim, padding_idx=4)
                    self.output_proj = nn.Linear(model_dim, hidden_dim)
                    self.is_real_foundation = True
                    self.foundation_kind = "local_huggingface_rna_model"
                    self.status = "real_local_checkpoint"
                except Exception as exc:  # pragma: no cover - depends on optional package/checkpoint.
                    self.status = f"load_failed:{type(exc).__name__}"
        if self.foundation is None and self.stage_model is None:
            if not allow_stub:
                raise RuntimeError(
                    "real RNA foundation checkpoint is required; pass a local model_path "
                    "and verify its SHA256, or use allow_stub only for smoke tests"
                )
            self.foundation = _FoundationStub(hidden_dim, max_len, name)
            self.status = "adapter_stub_smoke_only"
            self.is_real_foundation = False
        if freeze:
            self.freeze()
        elif self.is_real_foundation:
            self._configure_partial(unfreeze_last_n)

    def _load_stage_a_checkpoint(self, model_path: str, hidden_dim: int) -> None:
        """Load the repository's RNA-pretrained Stage-A trunk safely.

        Stage-A was trained on the GENCODE mRNA corpus. Its flow output heads
        are discarded; only the frozen token-level trunk is used here. This is
        explicitly tagged as an *internal mRNA-pretrained* foundation, not an
        external RNA-FM/Orthrus claim.
        """
        from mrna_editflow.core.config import BackboneConfig, ModelConfig
        from mrna_editflow.models.backbones import FrozenBackbone
        from mrna_editflow.models.mrna_editformer import MRNAEditFormer

        payload = torch.load(model_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "config" not in payload:
            raise ValueError("Stage-A checkpoint must contain a config")
        cfg = payload["config"]
        self.stage_backbone = FrozenBackbone(BackboneConfig(**cfg["backbone"]))
        self.stage_backbone.load_state_dict(payload["backbone_state"], strict=True)
        self.stage_model = MRNAEditFormer(
            ModelConfig(**cfg["model"]), backbone_dim=self.stage_backbone.out_dim,
        )
        self.stage_model.load_state_dict(payload["model_state"], strict=True)
        self.foundation_kind = "internal_stage_a_mrna_pretrained"
        self.is_real_foundation = True
        self.status = "real_internal_stage_a_mrna_pretrained"
        self.output_proj = nn.Linear(self.stage_model.dim, hidden_dim)
        self.stage_backbone.eval()
        self.stage_model.eval()

    def _configure_stage_a_partial(self, unfreeze_last_n: int) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if self.stage_model is not None and unfreeze_last_n > 0:
            for block in list(self.stage_model.blocks)[-unfreeze_last_n:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
        if self.output_proj is not None:
            for parameter in self.output_proj.parameters():
                parameter.requires_grad_(True)

    def _configure_partial(self, unfreeze_last_n: int) -> None:
        if self.stage_model is not None:
            self._configure_stage_a_partial(unfreeze_last_n)
            return
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        blocks = None
        for attr in ("encoder.layer", "transformer.layer", "layers", "block"):
            value = self.foundation
            for part in attr.split("."):
                value = getattr(value, part, None)
                if value is None:
                    break
            if value is not None:
                blocks = value
                break
        if blocks is not None and unfreeze_last_n > 0:
            for block in list(blocks)[-unfreeze_last_n:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
        if self.output_proj is not None:
            for parameter in self.output_proj.parameters():
                parameter.requires_grad_(True)
        if self.input_embedding is not None:
            for parameter in self.input_embedding.parameters():
                parameter.requires_grad_(True)

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen foundation representations must remain deterministic even when
        # the paired-delta head enters train mode.
        if all(not p.requires_grad for p in self.parameters()):
            super().train(False)
        return self

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.stage_model is not None:
            if tokens.shape[1] > self.stage_model.cfg.max_seq_len:
                raise ValueError("sequence length exceeds Stage-A foundation max_seq_len")
            from mrna_editflow.core.constants import PAD_TOKEN, PHASE_NONE
            stage_tokens = tokens.long().masked_fill(tokens.eq(4), PAD_TOKEN)
            region_ids = torch.zeros_like(stage_tokens)
            phase_ids = torch.full_like(stage_tokens, PHASE_NONE)
            padding_mask = ~mask.bool()
            time_step = torch.full((tokens.shape[0], 1), 0.5, device=tokens.device)
            output = self.stage_model.encode(
                stage_tokens, region_ids, phase_ids, time_step,
                padding_mask, self.stage_backbone,
            )
            output = self.output_proj(output)
        elif self.is_real_foundation:
            inputs = self.input_embedding(tokens.long())
            output = self.foundation(
                inputs_embeds=inputs,
                attention_mask=mask.long(),
            ).last_hidden_state
            output = self.output_proj(output)
        else:
            output = self.foundation(tokens, mask)
        return output * mask.unsqueeze(-1).to(output.dtype)


def build_sequence_encoder(
    backbone: str,
    hidden_dim: int,
    layers: int,
    max_len: int,
    foundation_path: Optional[str],
    allow_foundation_stub: bool,
    foundation_name: str,
    unfreeze_last_n: int,
) -> nn.Module:
    if backbone == "small":
        return SmallTaskSequenceEncoder(hidden_dim, layers, max_len)
    if backbone == "frozen_foundation":
        return FoundationSequenceEncoder(
            hidden_dim, max_len, foundation_path, freeze=True,
            allow_stub=allow_foundation_stub, name=foundation_name,
        )
    if backbone == "partial_foundation":
        return FoundationSequenceEncoder(
            hidden_dim, max_len, foundation_path, freeze=False,
            allow_stub=allow_foundation_stub, name=foundation_name,
            unfreeze_last_n=unfreeze_last_n,
        )
    raise ValueError("backbone must be small, frozen_foundation or partial_foundation")


class PairedDeltaFormer(nn.Module):
    """Predict source-relative delta, uncertainty, benefit and ranking signals."""

    def __init__(
        self,
        hidden_dim: int = 128,
        layers: int = 2,
        max_len: int = 256,
        backbone: str = "small",
        foundation_path: Optional[str] = None,
        allow_foundation_stub: bool = False,
        foundation_name: str = "rna_foundation",
        unfreeze_last_n: int = 1,
    ):
        super().__init__()
        if hidden_dim % 4:
            raise ValueError("hidden_dim must be divisible by 4 for the cross-attention adapter")
        self.backbone = backbone
        self.sequence_encoder = build_sequence_encoder(
            backbone, hidden_dim, layers, max_len, foundation_path,
            allow_foundation_stub, foundation_name, unfreeze_last_n,
        )
        # Siamese source/candidate encoder: source-relative comparison is the
        # invariant, and it keeps backbone capacity identical across arms.
        self.source_encoder = self.sequence_encoder
        self.candidate_encoder = self.sequence_encoder
        self.cross_candidate_to_source = nn.MultiheadAttention(hidden_dim, 4, dropout=0.1, batch_first=True)
        self.cross_source_to_candidate = nn.MultiheadAttention(hidden_dim, 4, dropout=0.1, batch_first=True)
        self.edit_encoder = EditTokenEncoder(hidden_dim=hidden_dim)
        self.context_encoder = ContextEncoder(hidden_dim=hidden_dim)
        self.source_value_proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim),
        )
        self.head = UncertaintyHead(hidden_dim)

    @staticmethod
    def _pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
        return (x * mask.unsqueeze(-1)).sum(dim=1) / denom

    def forward(
        self,
        source_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        edit_tokens: torch.Tensor,
        context_ids: torch.Tensor,
        source_value: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
        protein_embedding: Optional[torch.Tensor] = None,
        cell_embedding: Optional[torch.Tensor] = None,
        assay_embedding: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if source_mask is None:
            source_mask = torch.ones_like(source_tokens, dtype=torch.bool)
        if candidate_mask is None:
            candidate_mask = torch.ones_like(candidate_tokens, dtype=torch.bool)
        src = self.source_encoder(source_tokens, source_mask)
        cand = self.candidate_encoder(candidate_tokens, candidate_mask)
        src_cross, _ = self.cross_source_to_candidate(
            src, cand, cand, key_padding_mask=~candidate_mask.bool(), need_weights=False,
        )
        cand_cross, _ = self.cross_candidate_to_source(
            cand, src, src, key_padding_mask=~source_mask.bool(), need_weights=False,
        )
        src_pool = self._pool(src, source_mask)
        cand_pool = self._pool(cand, candidate_mask)
        cross_delta = self._pool(cand_cross, candidate_mask) - self._pool(src_cross, source_mask)
        edit = self.edit_encoder(edit_tokens)
        context = self.context_encoder(context_ids, protein_embedding, cell_embedding, assay_embedding)
        value = self.source_value_proj(source_value.float().reshape(-1, 1))
        fused = self.fusion(torch.cat([
            src_pool, cand_pool, cand_pool - src_pool, cross_delta, edit, context + value,
        ], dim=-1))
        return self.head(fused)

    @property
    def backbone_status(self) -> str:
        return str(getattr(self.sequence_encoder, "status", "unknown"))

    @property
    def foundation_kind(self) -> str:
        return str(getattr(self.sequence_encoder, "foundation_kind", "none"))

    @property
    def is_real_foundation(self) -> bool:
        return bool(getattr(self.sequence_encoder, "is_real_foundation", False))


__all__ = [
    "PairedDeltaFormer", "SmallTaskSequenceEncoder", "FoundationSequenceEncoder",
    "build_sequence_encoder",
]
