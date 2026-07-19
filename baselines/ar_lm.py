"""Autoregressive language-model baseline for mRNA sequence generation.

This baseline represents the traditional left-to-right sequence generation
family. It trains on the same nucleotide records as mRNA-EditFlow but predicts
``x_i`` from ``BOS, x_0, ..., x_{i-1}``, so it has no explicit edit operator,
region grammar, or variable-length alignment state.

The default implementation is a small GRU LM for fast CPU smoke tests. A causal
Transformer path is included for ablations that need the same public API with an
attention-based decoder.

Complexity
----------
GRU mode costs ``O(B * L * hidden_dim^2 * num_layers)`` time and
``O(B * L * hidden_dim)`` memory. Transformer mode costs
``O(num_layers * B * L^2 * hidden_dim)`` time due to causal attention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mrna_editflow.core.constants import (
    BOS_TOKEN,
    ID_TO_NUC,
    PAD_TOKEN,
    V as NUC_VOCAB_SIZE,
    VOCAB_MODEL_SIZE,
)
from mrna_editflow.core.schema import MRNARecord


@dataclass
class ARLMConfig:
    """Configuration for :class:`AutoregressiveLM`.

    ``cell_type`` may be ``"gru"`` or ``"transformer"``. Smoke tests use GRU
    because it is the fastest faithful autoregressive baseline on CPU.

    Complexity: construction is ``O(parameters)``; execution follows the module
    docstring.
    """

    max_len: int = 256
    hidden_dim: int = 64
    num_layers: int = 1
    num_heads: int = 4
    ffn_mult: int = 2
    dropout: float = 0.0
    cell_type: str = "gru"
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 4
    steps: int = 20
    seed: int = 0


@dataclass
class ARLMTrainResult:
    """Return object from :func:`train_ar_lm`.

    Complexity: stores ``O(steps)`` scalar losses plus the trained model.
    """

    model: "AutoregressiveLM"
    losses: List[float]
    final_loss: float
    config: ARLMConfig


class AutoregressiveLM(nn.Module):
    """Left-to-right nucleotide LM with GRU or causal Transformer decoder.

    The model consumes ids from the shared model vocabulary
    ``A,C,G,U,BOS,PAD`` and returns logits over the four real nucleotides.

    Complexity: GRU and Transformer modes follow the module docstring.
    """

    def __init__(self, config: Optional[ARLMConfig] = None):
        super().__init__()
        self.config = config or ARLMConfig()
        if self.config.max_len <= 0:
            raise ValueError("max_len must be positive")
        if self.config.cell_type not in {"gru", "transformer"}:
            raise ValueError("cell_type must be 'gru' or 'transformer'")
        if self.config.cell_type == "transformer" and self.config.hidden_dim % self.config.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.token_emb = nn.Embedding(VOCAB_MODEL_SIZE, self.config.hidden_dim)
        self.pos_emb = nn.Embedding(self.config.max_len, self.config.hidden_dim)
        if self.config.cell_type == "gru":
            self.decoder = nn.GRU(
                input_size=self.config.hidden_dim,
                hidden_size=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.is_transformer = False
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=self.config.hidden_dim,
                nhead=self.config.num_heads,
                dim_feedforward=self.config.hidden_dim * self.config.ffn_mult,
                dropout=self.config.dropout,
                activation="gelu",
                batch_first=True,
            )
            self.decoder = nn.TransformerEncoder(layer, num_layers=self.config.num_layers)
            self.is_transformer = True
        self.norm = nn.LayerNorm(self.config.hidden_dim)
        self.output = nn.Linear(self.config.hidden_dim, NUC_VOCAB_SIZE)

    def forward(self, input_ids: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        """Return next-token logits at every prefix position.

        ``padding_mask`` follows PyTorch convention: ``True`` marks padded input
        positions. It is used by Transformer mode and ignored by GRU mode because
        targets already mask padded positions.

        Complexity: one autoregressive decoder pass.
        """
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B,L], got {tuple(input_ids.shape)}")
        batch, length = input_ids.shape
        if length > self.config.max_len:
            raise ValueError(f"input length {length} exceeds max_len {self.config.max_len}")
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        hidden = self.token_emb(input_ids.clamp(0, VOCAB_MODEL_SIZE - 1)) + self.pos_emb(positions)

        if self.is_transformer:
            causal = torch.full((length, length), float("-inf"), device=input_ids.device)
            causal = torch.triu(causal, diagonal=1)
            hidden = self.decoder(hidden, mask=causal, src_key_padding_mask=padding_mask)
        else:
            hidden, _ = self.decoder(hidden)
        return self.output(self.norm(hidden))


def make_ar_lm_batch(records: Sequence[MRNARecord], max_len: Optional[int] = None) -> Dict[str, Tensor]:
    """Build a BOS-prefix batch for left-to-right training.

    Returns ``input_ids`` with ``BOS`` at the first real position, nucleotide
    prefixes afterwards, ``targets`` over ``A,C,G,U`` with ``-100`` on padding,
    and a boolean ``padding_mask``.

    Complexity: ``O(B * L)``.
    """
    if not records:
        raise ValueError("records must be non-empty")
    inferred_len = max(len(r.seq) for r in records)
    seq_len = int(max_len or inferred_len)
    if seq_len <= 0:
        raise ValueError("max_len must be positive")

    batch = len(records)
    input_ids = torch.full((batch, seq_len), PAD_TOKEN, dtype=torch.long)
    targets = torch.full((batch, seq_len), -100, dtype=torch.long)
    padding_mask = torch.ones((batch, seq_len), dtype=torch.bool)

    for row, rec in enumerate(records):
        ids = rec.token_ids()[:seq_len]
        length = len(ids)
        if length == 0:
            continue
        input_ids[row, 0] = BOS_TOKEN
        if length > 1:
            input_ids[row, 1:length] = torch.tensor(ids[:-1], dtype=torch.long)
        targets[row, :length] = torch.tensor(ids, dtype=torch.long)
        padding_mask[row, :length] = False
    return {"input_ids": input_ids, "targets": targets, "padding_mask": padding_mask}


def ar_lm_loss(model: AutoregressiveLM, batch: Dict[str, Tensor]) -> Tensor:
    """Compute autoregressive cross-entropy for one batch.

    Complexity: dominated by one model forward pass.
    """
    logits = model(batch["input_ids"], padding_mask=batch.get("padding_mask"))
    return F.cross_entropy(
        logits.reshape(-1, NUC_VOCAB_SIZE),
        batch["targets"].reshape(-1),
        ignore_index=-100,
    )


def _step_records(records: Sequence[MRNARecord], batch_size: int, step: int) -> List[MRNARecord]:
    """Deterministic wrap-around minibatch selection. Complexity: ``O(batch)``."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    n = len(records)
    return [records[(step * batch_size + i) % n] for i in range(min(batch_size, n))]


def train_ar_lm(
    records: Sequence[MRNARecord],
    config: Optional[ARLMConfig] = None,
    device: str = "cpu",
) -> ARLMTrainResult:
    """Train an autoregressive LM baseline on ``records``.

    The helper performs real AdamW updates and returns the trained model plus a
    finite loss trace for smoke tests and ablation scripts.

    Complexity: ``O(config.steps * model_forward_backward_cost)``.
    """
    if not records:
        raise ValueError("records must be non-empty")
    cfg = config or ARLMConfig()
    torch.manual_seed(cfg.seed)
    dev = torch.device(device)
    model = AutoregressiveLM(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    losses: List[float] = []

    model.train()
    for step in range(cfg.steps):
        rec_batch = _step_records(records, cfg.batch_size, step)
        batch = make_ar_lm_batch(rec_batch, max_len=cfg.max_len)
        batch = {key: value.to(dev) for key, value in batch.items()}
        opt.zero_grad(set_to_none=True)
        loss = ar_lm_loss(model, batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite AR LM loss at step {step}: {loss.item()}")
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))

    return ARLMTrainResult(model=model, losses=losses, final_loss=losses[-1], config=cfg)


@torch.no_grad()
def sample_ar_lm(
    model: AutoregressiveLM,
    length: int,
    temperature: float = 1.0,
    greedy: bool = True,
    seed: int = 0,
) -> str:
    """Generate one sequence left-to-right.

    The returned string has exactly ``length`` nucleotides and contains no
    special tokens.

    Complexity: ``O(length * prefix_forward_cost)`` for the simple smoke sampler.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    if length > model.config.max_len:
        raise ValueError(f"length {length} exceeds model max_len {model.config.max_len}")

    dev = next(model.parameters()).device
    gen = torch.Generator(device=dev).manual_seed(seed)
    prefix: List[int] = [BOS_TOKEN]
    out: List[int] = []
    model.eval()
    for _ in range(length):
        inp = torch.tensor([prefix], dtype=torch.long, device=dev)
        padding_mask = torch.zeros_like(inp, dtype=torch.bool)
        logits = model(inp, padding_mask=padding_mask)[:, -1, :]
        if temperature <= 0.0 or greedy:
            next_id = int(logits.argmax(dim=-1).item())
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1, generator=gen).item())
        out.append(next_id)
        prefix.append(next_id)

    return "".join(ID_TO_NUC[int(i)] for i in out)


__all__ = [
    "ARLMConfig",
    "ARLMTrainResult",
    "AutoregressiveLM",
    "make_ar_lm_batch",
    "ar_lm_loss",
    "train_ar_lm",
    "sample_ar_lm",
]
