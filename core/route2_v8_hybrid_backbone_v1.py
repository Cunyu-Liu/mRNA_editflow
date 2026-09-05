"""V8 Stage 1 hybrid backbone: mRNABERT trunk + optional CNN motif stem + domain conditioning.

Pre-registered architecture decisions (docs/paper/route2_v8_stage1_prereg_v1.md):

- Arm S (pure): mRNABERT 12-layer trunk (768d, raw pretrained init) + masked-mean-pool
  + domain-conditional readout + linear head.
- Arm H (hybrid): Arm S + Optimus-style CNN motif stem. The stem consumes a
  nucleotide one-hot derived from input_ids; its output is projected to 768d and
  ADDED to the word embeddings (residual injection) before BertEmbeddings.LayerNorm.
  The alternative (concat-then-project) was rejected because it changes the encoder
  input width and would require surgery on the pinned bert_layers implementation.

- CNN stem (two Optimus-style conv blocks, ~0.5M-parameter budget):
      one-hot [B,L,4] -> Conv1d(4->96, k=8) -> GELU -> MaxPool1d(2)
      -> Conv1d(96->128, k=6) -> GELU -> nearest upsample back to L
      -> Linear(128->512) -> GELU -> Linear(512->768)     [537,056 parameters]
  Length is preserved exactly via asymmetric zero padding (TF-SAME convention:
  left (k-1)//2, right k//2). Non-nucleotide positions (CLS/SEP/PAD/UNK/N) receive
  exactly zero stem features so special-token embeddings stay untouched.

- Zero-initialised final stem projection: at initialisation the stem contributes
  exactly zero, so arm H is functionally identical to arm S at step 0. This removes
  the init confound from the S-vs-H adjudication (the stem must EARN its keep).

- Domain conditioning: a learned 768d embedding per domain added to the POOLED
  representation before the head (not to the first token). Reasons: (a) it leaves
  the pretrained encoder input distribution untouched -- Phase 0 showed raw-init
  mRNABERT is fragile (direct-FT from raw weights failed MRL/MPRAU), so the trunk
  input must stay as close to the pre-finetuning distribution as possible;
  (b) the trunk stays domain-agnostic, which is the point of a JOINT prior, with
  the domain token only selecting the readout; (c) identical code path for both
  arms and trivial zero-shot application (the domain id is always known at
  inference).

mRNABERT loading pattern (from run_route2_mrnabert_directft_arm_a_v1.py): AutoConfig
+ AutoModel.from_config + manual strip of the "bert." prefix + flash_attn_
qkvpacked_func = None (AutoModel.from_pretrained is incompatible with the custom
ALiBi stack).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

# mRNABERT vocab (vocab.txt of the pinned checkpoint): [PAD]=0 [UNK]=1 [CLS]=2
# [SEP]=3 [MASK]=4 A=5 T=6 C=7 G=8 N=9. Verify with verify_vocab_alignment().
NUCLEOTIDE_TOKEN_IDS = {"A": 5, "T": 6, "C": 7, "G": 8, "N": 9}
_ONE_HOT_CHANNELS = 4

DOMAIN_IDS = {"mrl": 0, "polya": 1, "cms": 2}
NUM_DOMAINS = 3

STEM_CONV1_CHANNELS = 96
STEM_CONV1_KERNEL = 8
STEM_CONV2_CHANNELS = 128
STEM_CONV2_KERNEL = 6
STEM_PROJECTION_HIDDEN = 512
HIDDEN_SIZE = 768


def verify_vocab_alignment(tokenizer) -> None:
    """Assert the hardcoded nucleotide token ids match the pinned tokenizer."""
    for token, expected in NUCLEOTIDE_TOKEN_IDS.items():
        actual = tokenizer.convert_tokens_to_ids(token)
        if actual != expected:
            raise ValueError(
                f"mRNABERT vocab misalignment: token {token} -> id {actual}, "
                f"expected {expected}; nucleotide_one_hot would be wrong"
            )


def nucleotide_one_hot(input_ids: torch.Tensor) -> torch.Tensor:
    """[B, L] token ids -> [B, L, 4] float one-hot (A/T/C/G channels).

    Non-nucleotide tokens (PAD/UNK/CLS/SEP/MASK/N and anything outside 5..8)
    map to the all-zero vector.
    """
    is_nt = (input_ids >= 5) & (input_ids <= 8)
    channel = (input_ids - 5).clamp(min=0, max=_ONE_HOT_CHANNELS - 1)
    one_hot = F.one_hot(channel, num_classes=_ONE_HOT_CHANNELS).to(torch.float32)
    return one_hot * is_nt.unsqueeze(-1).to(torch.float32)


def _same_pad_conv1d(x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
    """Conv1d with exact TF-SAME length preservation for even kernels."""
    kernel = conv.kernel_size[0]
    left = (kernel - 1) // 2
    right = kernel - 1 - left
    return conv(F.pad(x, (left, right)))


class CNNMotifStem(nn.Module):
    """Optimus-style CNN motif stem, nucleotide resolution in / 768d out.

    Output at position i is a function of the one-hot neighbourhood around
    nucleotide i (receptive field ~ 2*(8 + 2*6) + 1 after pool/upsample), and is
    exactly zero at non-nucleotide positions. The final projection starts at
    zero so an untrained stem is a no-op.
    """

    def __init__(self, projection_hidden: int = STEM_PROJECTION_HIDDEN, output_dim: int = HIDDEN_SIZE) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(_ONE_HOT_CHANNELS, STEM_CONV1_CHANNELS, kernel_size=STEM_CONV1_KERNEL)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(STEM_CONV1_CHANNELS, STEM_CONV2_CHANNELS, kernel_size=STEM_CONV2_KERNEL)
        self.proj1 = nn.Linear(STEM_CONV2_CHANNELS, projection_hidden)
        self.proj2 = nn.Linear(projection_hidden, output_dim)
        nn.init.zeros_(self.proj2.weight)
        nn.init.zeros_(self.proj2.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        length = input_ids.shape[1]
        is_nt = ((input_ids >= 5) & (input_ids <= 8)).to(torch.float32)  # [B, L]
        x = nucleotide_one_hot(input_ids).transpose(1, 2)  # [B, 4, L]
        x = F.gelu(_same_pad_conv1d(x, self.conv1))
        # Zero non-nucleotide responses BEFORE pooling: pooled windows over
        # PAD/CLS/SEP then contribute exactly 0, which equals the SAME-pad
        # zeros seen at the right boundary of a shorter batch -- this makes
        # stem features at real positions EXACTLY invariant to batch padding
        # length (train/eval consistency).
        x = x * is_nt.unsqueeze(1)
        x = self.pool(x)
        x = F.gelu(_same_pad_conv1d(x, self.conv2))
        if x.shape[-1] != length:
            # Explicit position-aligned unpooling (dst i <- pooled i//2, clamped):
            # F.interpolate(nearest) misaligns when the pooled length is
            # floor(L/2) with odd L (MaxPool1d drops the tail), which would make
            # stem features depend on batch padding parity.
            src_index = (torch.arange(length, device=x.device) // 2).clamp(max=x.shape[-1] - 1)
            x = x[:, :, src_index]
        x = x.transpose(1, 2)  # [B, L, 128]
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)  # [B, L, 768]
        return x * is_nt.unsqueeze(-1)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class V8JointRegressor(nn.Module):
    """Joint pre-finetuning regressor for V8 Stage 1 arms S (pure) and H (hybrid).

    forward(input_ids, attention_mask, domain_ids) -> [B] activity prediction
    (domain-standardised scale). encode_pooled(...) exposes the conditioned
    pooled representation for diagnostics / Stage-2 reuse.
    """

    def __init__(self, base_model: nn.Module, use_stem: bool, num_domains: int = NUM_DOMAINS) -> None:
        super().__init__()
        self.base = base_model
        self.use_stem = use_stem
        if use_stem:
            self.stem = CNNMotifStem(output_dim=base_model.config.hidden_size)
        self.domain_embeddings = nn.Embedding(num_domains, base_model.config.hidden_size)
        nn.init.normal_(self.domain_embeddings.weight, mean=0.0, std=0.02)
        self.head = nn.Linear(base_model.config.hidden_size, 1)

    def _sequence_output(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if not self.use_stem:
            return self.base(input_ids=input_ids, attention_mask=attention_mask)[0]
        # Hybrid path: word embeddings + stem features, then the standard
        # BertEmbeddings post-processing (token-type add + LayerNorm + dropout)
        # via inputs_embeds, then the standard encoder call (same ops as
        # BertModel.forward, which does not expose inputs_embeds itself).
        word_embeddings = self.base.embeddings.word_embeddings(input_ids)
        stem_features = self.stem(input_ids).to(word_embeddings.dtype)
        # token_type_ids passed explicitly (zeros) exactly as BertModel.forward
        # does; the buffer fallback in the pinned bert_layers would mis-index.
        embedding_output = self.base.embeddings(
            inputs_embeds=word_embeddings + stem_features,
            token_type_ids=torch.zeros_like(input_ids),
        )
        encoder_outputs = self.base.encoder(embedding_output, attention_mask, output_all_encoded_layers=False)
        return encoder_outputs[-1]

    def encode_pooled(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        hidden = self._sequence_output(input_ids, attention_mask)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled + self.domain_embeddings(domain_ids)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        pooled = self.encode_pooled(input_ids, attention_mask, domain_ids)
        return self.head(pooled).squeeze(-1)


def load_mrnabert_base(mrnabert_path: Path):
    """Load the pinned mRNABERT checkpoint exactly as the Route A reference scripts."""
    from transformers import AutoConfig, AutoModel

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model_config = AutoConfig.from_pretrained(mrnabert_path, local_files_only=True, trust_remote_code=True)
    base = AutoModel.from_config(model_config, trust_remote_code=True, add_pooling_layer=False)
    modeling_module = sys.modules[base.__class__.__module__]
    modeling_module.flash_attn_qkvpacked_func = None  # official fallback backend
    checkpoint = torch.load(mrnabert_path / "pytorch_model.bin", map_location="cpu", weights_only=False)
    base_state = {
        key.removeprefix("bert."): value
        for key, value in checkpoint.items()
        if key.startswith("bert.")
    }
    base.load_state_dict(base_state, strict=True)
    del checkpoint, base_state
    return base


def build_v8_regressor(mrnabert_path: Path, arch: str, num_domains: int = NUM_DOMAINS) -> V8JointRegressor:
    """arch in {"s", "h"}: pure mRNABERT / hybrid CNN-stem."""
    if arch not in ("s", "h"):
        raise ValueError(f"arch must be 's' or 'h', got {arch!r}")
    base = load_mrnabert_base(mrnabert_path)
    return V8JointRegressor(base, use_stem=(arch == "h"), num_domains=num_domains)


def parameter_report(model: V8JointRegressor) -> dict:
    """Trainable parameter accounting for the run manifest."""
    stem_count = model.stem.parameter_count() if model.use_stem else 0
    domain_count = model.domain_embeddings.weight.numel()
    head_count = sum(p.numel() for p in model.head.parameters())
    base_count = sum(p.numel() for p in model.base.parameters())
    return {
        "trainable_total": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "mrnabert_base": base_count,
        "cnn_stem": stem_count,
        "domain_embeddings": domain_count,
        "linear_head": head_count,
        "use_stem": model.use_stem,
    }
