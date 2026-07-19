"""Masked discrete-diffusion baseline for fixed-canvas mRNA generation.

This module implements a real trainable baseline for the core ablation
``edit-flow variable-length edits`` versus ``fixed-length canvas denoising``.
It follows the practical BERT-style simplification of masked discrete diffusion:
sample a noise time through the mask probability, replace observed nucleotides
with a learned ``MASK`` token, then train a denoising network to recover the
original nucleotide at masked positions.

The baseline intentionally has no insert/delete state. During sampling it starts
from a fixed-length all-mask canvas and repeatedly fills/re-masks positions,
making the contrast with Edit Flow explicit: all candidates keep the requested
canvas length.

Complexity
----------
For a batch of size ``B`` and canvas length ``L``, the Transformer denoiser costs
``O(num_layers * B * L^2 * hidden_dim)`` time and ``O(B * L * hidden_dim)`` memory
outside the attention maps. The smoke training helper is ``O(steps * cost)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mrna_editflow.core.constants import (
    ID_TO_NUC,
    NUM_REGIONS,
    PAD_TOKEN,
    V as NUC_VOCAB_SIZE,
    VOCAB_MODEL_SIZE,
)
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.mrna_dataset import REGION_PAD

MASK_TOKEN: int = VOCAB_MODEL_SIZE
"""Special token id used only by the masked-diffusion baseline."""

DIFFUSION_INPUT_VOCAB_SIZE: int = VOCAB_MODEL_SIZE + 1
"""Embedding vocabulary: RNA nucleotides + BOS/PAD inherited ids + MASK."""


@dataclass
class MaskedDiffusionConfig:
    """Configuration for :class:`MaskedDiffusionBaseline`.

    ``max_len`` is the fixed canvas length. Sequences longer than the canvas are
    cropped by the smoke helper, which is acceptable for the baseline ablation
    because the method itself cannot model variable-length edits.

    Complexity: construction is ``O(1)``; model execution follows the module
    docstring.
    """

    max_len: int = 256
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    ffn_mult: int = 2
    dropout: float = 0.0
    mask_prob: float = 0.30
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 4
    steps: int = 20
    seed: int = 0


@dataclass
class MaskedDiffusionTrainResult:
    """Return object from :func:`train_masked_diffusion`.

    Attributes record the trained model and scalar losses so tests and ablation
    scripts can assert finite optimisation without inspecting private state.

    Complexity: storing the result is ``O(steps)`` for the loss trace.
    """

    model: "MaskedDiffusionBaseline"
    losses: List[float]
    final_loss: float
    config: MaskedDiffusionConfig


class MaskedDiffusionBaseline(nn.Module):
    """Denoising Transformer over a fixed nucleotide canvas.

    Inputs are token ids in ``[0, DIFFUSION_INPUT_VOCAB_SIZE)`` plus optional
    region ids aligned to the same canvas. Outputs are nucleotide logits over
    ``A,C,G,U`` only; special tokens are never generated.

    Complexity: ``O(num_layers * B * L^2 * hidden_dim)`` time and
    ``O(B * L * hidden_dim)`` activation memory for sequence length ``L``.
    """

    def __init__(self, config: Optional[MaskedDiffusionConfig] = None):
        super().__init__()
        self.config = config or MaskedDiffusionConfig()
        if self.config.max_len <= 0:
            raise ValueError("max_len must be positive")
        if self.config.hidden_dim % self.config.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.token_emb = nn.Embedding(DIFFUSION_INPUT_VOCAB_SIZE, self.config.hidden_dim)
        self.region_emb = nn.Embedding(NUM_REGIONS + 1, self.config.hidden_dim)
        self.pos_emb = nn.Embedding(self.config.max_len, self.config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.hidden_dim * self.config.ffn_mult,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.config.num_layers)
        self.norm = nn.LayerNorm(self.config.hidden_dim)
        self.output = nn.Linear(self.config.hidden_dim, NUC_VOCAB_SIZE)

    def forward(
        self,
        token_ids: Tensor,
        region_ids: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Return nucleotide logits for every canvas position.

        Parameters follow the ``nn.Transformer`` convention: ``padding_mask`` is
        ``True`` at padded positions to ignore.

        Complexity: same as :class:`MaskedDiffusionBaseline`.
        """
        if token_ids.dim() != 2:
            raise ValueError(f"token_ids must be [B,L], got {tuple(token_ids.shape)}")
        batch, length = token_ids.shape
        if length > self.config.max_len:
            raise ValueError(f"input length {length} exceeds max_len {self.config.max_len}")
        if region_ids is None:
            region_ids = torch.full_like(token_ids, REGION_PAD)
        if padding_mask is None:
            padding_mask = torch.zeros_like(token_ids, dtype=torch.bool)

        positions = torch.arange(length, device=token_ids.device).unsqueeze(0).expand(batch, -1)
        tok = token_ids.clamp(0, DIFFUSION_INPUT_VOCAB_SIZE - 1)
        reg = region_ids.clamp(0, NUM_REGIONS)
        hidden = self.token_emb(tok) + self.region_emb(reg) + self.pos_emb(positions)
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        hidden = self.norm(hidden)
        return self.output(hidden)


def make_fixed_canvas_batch(
    records: Sequence[MRNARecord],
    max_len: Optional[int] = None,
) -> Dict[str, Tensor]:
    """Convert records into a padded fixed-canvas denoising batch.

    Returns ``token_ids`` in RNA ids ``0..3``, aligned ``region_ids`` and a
    boolean ``padding_mask``. Long records are cropped to ``max_len``.

    Complexity: ``O(B * L)`` for batch size ``B`` and canvas length ``L``.
    """
    if not records:
        raise ValueError("records must be non-empty")
    inferred_len = max(len(r.seq) for r in records)
    canvas_len = int(max_len or inferred_len)
    if canvas_len <= 0:
        raise ValueError("max_len must be positive")

    batch = len(records)
    token_ids = torch.full((batch, canvas_len), PAD_TOKEN, dtype=torch.long)
    region_ids = torch.full((batch, canvas_len), REGION_PAD, dtype=torch.long)
    padding_mask = torch.ones((batch, canvas_len), dtype=torch.bool)

    for row, rec in enumerate(records):
        ids = rec.token_ids()[:canvas_len]
        regs = rec.region_ids()[:canvas_len]
        length = len(ids)
        if length == 0:
            continue
        token_ids[row, :length] = torch.tensor(ids, dtype=torch.long)
        region_ids[row, :length] = torch.tensor(regs, dtype=torch.long)
        padding_mask[row, :length] = False
    return {"token_ids": token_ids, "region_ids": region_ids, "padding_mask": padding_mask}


def corrupt_with_masks(
    token_ids: Tensor,
    padding_mask: Tensor,
    mask_prob: float,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Tensor]:
    """Apply BERT-style mask corruption for the denoising objective.

    At least one real token per non-empty sequence is masked to keep the loss
    well-defined even for tiny smoke batches.

    Complexity: ``O(B * L)``.
    """
    if not 0.0 < mask_prob <= 1.0:
        raise ValueError("mask_prob must be in (0, 1]")
    real = ~padding_mask
    rand = torch.rand(token_ids.shape, device=token_ids.device, generator=generator)
    denoise_mask = real & (rand < mask_prob)
    for row in range(token_ids.shape[0]):
        if real[row].any() and not denoise_mask[row].any():
            first_real = int(torch.nonzero(real[row], as_tuple=False)[0].item())
            denoise_mask[row, first_real] = True

    corrupted = token_ids.clone()
    corrupted[denoise_mask] = MASK_TOKEN
    corrupted[padding_mask] = PAD_TOKEN
    return {
        "corrupted_token_ids": corrupted,
        "targets": token_ids.clamp(0, NUC_VOCAB_SIZE - 1),
        "denoise_mask": denoise_mask,
    }


def masked_diffusion_loss(
    model: MaskedDiffusionBaseline,
    batch: Dict[str, Tensor],
    mask_prob: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Compute masked denoising cross-entropy for one batch.

    The loss is evaluated only on corrupted real positions, matching the
    denoising score-matching surrogate used by masked discrete diffusion.

    Complexity: dominated by one model forward pass.
    """
    prob = float(model.config.mask_prob if mask_prob is None else mask_prob)
    corrupted = corrupt_with_masks(batch["token_ids"], batch["padding_mask"], prob, generator)
    logits = model(
        corrupted["corrupted_token_ids"],
        region_ids=batch.get("region_ids"),
        padding_mask=batch.get("padding_mask"),
    )
    mask = corrupted["denoise_mask"]
    if not bool(mask.any()):
        raise RuntimeError("masked_diffusion_loss received no masked real positions")
    return F.cross_entropy(logits[mask], corrupted["targets"][mask])


def _step_records(records: Sequence[MRNARecord], batch_size: int, step: int) -> List[MRNARecord]:
    """Deterministic wrap-around minibatch selection. Complexity: ``O(batch)``."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    n = len(records)
    return [records[(step * batch_size + i) % n] for i in range(min(batch_size, n))]


def train_masked_diffusion(
    records: Sequence[MRNARecord],
    config: Optional[MaskedDiffusionConfig] = None,
    device: str = "cpu",
) -> MaskedDiffusionTrainResult:
    """Train a small masked-diffusion baseline on ``records``.

    This helper is intentionally minimal but fully differentiable: it builds the
    denoising model, runs AdamW updates, and returns the trained module with its
    finite loss trace.

    Complexity: ``O(config.steps * model_forward_backward_cost)``.
    """
    if not records:
        raise ValueError("records must be non-empty")
    cfg = config or MaskedDiffusionConfig()
    torch.manual_seed(cfg.seed)
    dev = torch.device(device)
    model = MaskedDiffusionBaseline(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    generator = torch.Generator(device=dev).manual_seed(cfg.seed + 17)
    losses: List[float] = []

    model.train()
    for step in range(cfg.steps):
        rec_batch = _step_records(records, cfg.batch_size, step)
        batch = make_fixed_canvas_batch(rec_batch, max_len=cfg.max_len)
        batch = {key: value.to(dev) for key, value in batch.items()}
        opt.zero_grad(set_to_none=True)
        loss = masked_diffusion_loss(model, batch, generator=generator)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite masked diffusion loss at step {step}: {loss.item()}")
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))

    return MaskedDiffusionTrainResult(
        model=model,
        losses=losses,
        final_loss=losses[-1],
        config=cfg,
    )


@torch.no_grad()
def sample_masked_diffusion(
    model: MaskedDiffusionBaseline,
    length: int,
    region_ids: Optional[Sequence[int]] = None,
    denoise_steps: int = 4,
    temperature: float = 1.0,
    greedy: bool = True,
    seed: int = 0,
) -> str:
    """Generate one fixed-length mRNA-like sequence from an all-mask canvas.

    The returned string always has exactly ``length`` nucleotides. ``region_ids``
    may be supplied to condition on a desired UTR/CDS/UTR partition; otherwise
    an unknown-region sentinel is used.

    Complexity: ``O(denoise_steps * model_forward_cost)``.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    if length > model.config.max_len:
        raise ValueError(f"length {length} exceeds model max_len {model.config.max_len}")
    if denoise_steps <= 0:
        raise ValueError("denoise_steps must be positive")

    dev = next(model.parameters()).device
    gen = torch.Generator(device=dev).manual_seed(seed)
    tokens = torch.full((1, length), MASK_TOKEN, dtype=torch.long, device=dev)
    if region_ids is None:
        reg = torch.full((1, length), REGION_PAD, dtype=torch.long, device=dev)
    else:
        if len(region_ids) != length:
            raise ValueError("region_ids length must equal requested sample length")
        reg = torch.tensor([list(region_ids)], dtype=torch.long, device=dev)
    padding_mask = torch.zeros((1, length), dtype=torch.bool, device=dev)

    model.eval()
    for step in range(denoise_steps):
        logits = model(tokens, region_ids=reg, padding_mask=padding_mask)
        if temperature <= 0.0 or greedy:
            proposed = logits.argmax(dim=-1)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            proposed = torch.multinomial(
                probs.reshape(-1, NUC_VOCAB_SIZE),
                num_samples=1,
                generator=gen,
            ).reshape(1, length)

        if step == denoise_steps - 1:
            tokens = proposed
        else:
            keep_prob = float(step + 1) / float(denoise_steps)
            remask = torch.rand((1, length), device=dev, generator=gen) > keep_prob
            tokens = torch.where(remask, torch.full_like(proposed, MASK_TOKEN), proposed)

    ids = tokens.squeeze(0).detach().cpu().tolist()
    return "".join(ID_TO_NUC[int(i)] for i in ids)


__all__ = [
    "MASK_TOKEN",
    "DIFFUSION_INPUT_VOCAB_SIZE",
    "MaskedDiffusionConfig",
    "MaskedDiffusionTrainResult",
    "MaskedDiffusionBaseline",
    "make_fixed_canvas_batch",
    "corrupt_with_masks",
    "masked_diffusion_loss",
    "train_masked_diffusion",
    "sample_masked_diffusion",
]
