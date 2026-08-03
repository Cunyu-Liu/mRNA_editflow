"""GPU-only frozen UTR-LM source/current fusion for the MK0 rate field.

The public forward interface accepts only the inference-visible ``EditState``
and external time.  It cannot receive an alignment, target sequence or
remaining-target-edit feature.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .state_action import enumerate_legal_actions
from .types import ALPHABET, ActionType, AtomicAction, EditState, TokenOrigin


def require_neural_cuda(device: torch.device | str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "MK0 neural forward/backward is GPU-only; CUDA absence or CPU fallback is fatal"
        )
    return resolved


def load_official_utrlm(
    snapshot_dir: str | Path,
    *,
    device: torch.device | str,
    from_scratch: bool = False,
    seed: int = 20260802,
):
    """Load the frozen official FM0 bare encoder or same-architecture control."""

    resolved = require_neural_cuda(device)
    snapshot = str(Path(snapshot_dir).resolve())
    from multimolecule import RnaTokenizer, UtrLmConfig, UtrLmModel

    tokenizer = RnaTokenizer.from_pretrained(snapshot, local_files_only=True)
    if from_scratch:
        torch.manual_seed(seed)
        config = UtrLmConfig.from_pretrained(snapshot, local_files_only=True)
        model = UtrLmModel(config)
    else:
        model = UtrLmModel.from_pretrained(snapshot, local_files_only=True)
    model = model.to(device=resolved, dtype=torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def _stable_unit_interval(text: str) -> float:
    value = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    return value / float(16**16 - 1)


class FoundationFusionRateField(nn.Module):
    """Frozen-foundation fusion plus a small trainable absolute-rate head."""

    inference_signature_fields = (
        "source",
        "current",
        "M_run",
        "region",
        "context",
        "target_condition",
        "time",
        "remaining_budget",
        "h_run",
    )

    def __init__(
        self,
        foundation: nn.Module,
        tokenizer,
        *,
        device: torch.device | str,
        min_length: int,
        max_length: int,
        hidden_size: Optional[int] = None,
        train_foundation: bool = False,
        cache_current_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.device = require_neural_cuda(device)
        self.train_foundation = bool(train_foundation)
        self.cache_current_embeddings = bool(cache_current_embeddings)
        if self.train_foundation and self.cache_current_embeddings:
            raise ValueError(
                "train_foundation and embedding caching are incompatible: "
                "caching would retain an optimizer graph across updates"
            )
        self.foundation = foundation.to(self.device)
        for parameter in self.foundation.parameters():
            parameter.requires_grad_(self.train_foundation)
        self.foundation.train(self.train_foundation)
        self.tokenizer = tokenizer
        self.min_length = min_length
        self.max_length = max_length
        self.hidden_size = int(
            hidden_size
            or getattr(getattr(foundation, "config", None), "hidden_size", 128)
        )
        # 3 pooled H vectors + current/action-local H + source-aligned/action-
        # local H + 8 state scalars + 9 action scalars.  Keeping the two local
        # vectors explicit is essential: a global mean alone cannot represent
        # the current token/gap or its source-aligned identity at the proposed
        # action coordinate.
        self.rate_head = nn.Sequential(
            nn.Linear(5 * self.hidden_size + 17, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        ).to(self.device, dtype=torch.float32)
        # A caller may replace this with a registered low-rank residual adapter.
        # Keeping the identity in the production module makes the representation
        # path explicit without changing the frozen-foundation default.
        self.feature_adapter: nn.Module = nn.Identity()
        self._source_cache: dict[str, torch.Tensor] = {}
        self.source_encode_calls = 0
        self.current_encode_calls = 0

    def _encode_tokens(self, sequence: str, *, source_cache: bool) -> torch.Tensor:
        cache_key = hashlib.sha256(sequence.encode("ascii")).hexdigest()
        adapter_trainable = any(
            parameter.requires_grad for parameter in self.feature_adapter.parameters()
        )
        cache_allowed = (
            (source_cache or self.cache_current_embeddings)
            and not self.train_foundation
            and not adapter_trainable
        )
        if cache_allowed and cache_key in self._source_cache:
            return self._source_cache[cache_key]
        batch = self.tokenizer([sequence], padding=True, return_tensors="pt")
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with (nullcontext() if self.train_foundation else torch.no_grad()):
            hidden = self.foundation(**batch).last_hidden_state
        # Frozen FM0 tokenizer contract: <cls>, nucleotides, <eos>.
        if hidden.shape[1] < len(sequence) + 2:
            raise RuntimeError("UTR-LM tokenization is not nucleotide aligned")
        tokens = hidden[0, 1 : 1 + len(sequence), :]
        if tokens.shape != (len(sequence), self.hidden_size):
            raise RuntimeError("unexpected UTR-LM token embedding shape")
        if not self.train_foundation:
            tokens = tokens.detach()
        tokens = self.feature_adapter(tokens)
        if cache_allowed:
            self.source_encode_calls += 1
            self._source_cache[cache_key] = tokens
        else:
            if source_cache:
                self.source_encode_calls += 1
            else:
                self.current_encode_calls += 1
        return tokens

    def clear_embedding_cache(self) -> None:
        """Clear frozen-representation cache at an auditable run boundary."""

        self._source_cache.clear()

    def train(self, mode: bool = True):  # type: ignore[override]
        """Keep a frozen foundation in eval mode even when heads train."""

        super().train(mode)
        self.foundation.train(mode and self.train_foundation)
        return self

    def representation_mode(self) -> str:
        adapter_trainable = any(
            parameter.requires_grad for parameter in self.feature_adapter.parameters()
        )
        if self.train_foundation:
            return "from_scratch_foundation"
        if adapter_trainable:
            return "low_rank_residual_adapter"
        if self.cache_current_embeddings:
            return "frozen_foundation_cached_embeddings"
        return "frozen_foundation_dynamic_current"

    def _encoded_state(
        self, state: EditState, time: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not 0.0 <= time < 1.0:
            raise ValueError("rate field time must be in [0,1)")
        source_tokens = self._encode_tokens(state.source, source_cache=True)
        # Deliberately recomputed on every call.  A state update can never reuse
        # a source-only pooled cache as the current representation.
        current_tokens = self._encode_tokens(state.current, source_cache=False)
        aligned = []
        zero = torch.zeros(self.hidden_size, device=self.device, dtype=torch.float32)
        for ref in state.mapping.tokens:
            if ref.origin == TokenOrigin.SOURCE:
                aligned.append(source_tokens[int(ref.source_index)])
            else:
                aligned.append(zero)
        aligned_tensor = torch.stack(aligned)
        context_payload = json.dumps(
            dict(state.context),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        scalar = torch.tensor(
            [
                time,
                state.remaining_budget / max(1, state.initial_budget),
                state.history.executed / max(1, state.initial_budget),
                state.history.ins / max(1, state.initial_budget),
                state.history.sub / max(1, state.initial_budget),
                state.history.delete / max(1, state.initial_budget),
                0.0 if state.region == "5UTR" else 1.0,
                0.5
                * (
                    _stable_unit_interval(context_payload)
                    + _stable_unit_interval(state.target_condition)
                ),
            ],
            device=self.device,
            dtype=torch.float32,
        )
        shared = torch.cat(
            (
                source_tokens.mean(dim=0),
                current_tokens.mean(dim=0),
                aligned_tensor.mean(dim=0),
                scalar,
            )
        )
        return source_tokens, current_tokens, aligned_tensor, shared

    def _gap_representation(self, tokens: torch.Tensor, gap: int) -> torch.Tensor:
        if gap < 0 or gap > tokens.shape[0]:
            raise ValueError("gap is outside the current sequence")
        neighbours: list[torch.Tensor] = []
        if gap > 0:
            neighbours.append(tokens[gap - 1])
        if gap < tokens.shape[0]:
            neighbours.append(tokens[gap])
        if not neighbours:
            return torch.zeros(
                self.hidden_size, device=self.device, dtype=torch.float32
            )
        return torch.stack(neighbours).mean(dim=0)

    def _local_representations(
        self,
        state: EditState,
        action: AtomicAction,
        current_tokens: torch.Tensor,
        aligned_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather action-position current/gap and source-aligned features."""

        if action.kind == ActionType.INS:
            gap = int(action.position)
            return (
                self._gap_representation(current_tokens, gap),
                self._gap_representation(aligned_tokens, gap),
            )
        if action.kind in {ActionType.SUB, ActionType.DEL}:
            position = int(action.position)
            if position >= len(state.current):
                raise ValueError("action token position is outside current state")
            return current_tokens[position], aligned_tokens[position]
        if action.kind == ActionType.STOP:
            return current_tokens.mean(dim=0), aligned_tokens.mean(dim=0)
        raise ValueError(f"unsupported action type: {action.kind}")

    def _action_features(self, state: EditState, action: AtomicAction) -> torch.Tensor:
        kind = torch.zeros(4, device=self.device, dtype=torch.float32)
        kind[list(ActionType).index(action.kind)] = 1.0
        position = (
            0.0
            if action.position is None
            else action.position / max(1, len(state.current))
        )
        token = torch.zeros(4, device=self.device, dtype=torch.float32)
        if action.token is not None:
            token[ALPHABET.index(action.token)] = 1.0
        return torch.cat((kind, torch.tensor([position], device=self.device), token))

    def forward(
        self,
        state: EditState,
        time: float,
        actions: Optional[Sequence[AtomicAction]] = None,
    ) -> dict[AtomicAction, torch.Tensor]:
        if next(self.rate_head.parameters()).device.type != "cuda":
            raise RuntimeError("rate head silently left CUDA")
        legal = set(
            enumerate_legal_actions(
                state,
                min_length=self.min_length,
                max_length=self.max_length,
                include_stop=True,
            )
        )
        selected = (
            tuple(actions)
            if actions is not None
            else tuple(sorted(legal, key=lambda a: a.key))
        )
        if any(action not in legal for action in selected):
            raise ValueError("requested action is hard-masked")
        _, current_tokens, aligned_tokens, shared = self._encoded_state(state, time)
        result: dict[AtomicAction, torch.Tensor] = {}
        for action in selected:
            current_local, source_aligned_local = self._local_representations(
                state, action, current_tokens, aligned_tokens
            )
            raw = self.rate_head(
                torch.cat(
                    (
                        shared,
                        current_local,
                        source_aligned_local,
                        self._action_features(state, action),
                    )
                )
            )
            rate = F.softplus(raw.squeeze())
            if rate.device.type != "cuda" or not bool(torch.isfinite(rate)):
                raise FloatingPointError("invalid neural action rate or CPU fallback")
            result[action] = rate
        return result

    def action_representation_audit(
        self, state: EditState, time: float, action: AtomicAction
    ) -> dict[str, torch.Tensor]:
        """Expose the exact production gathers for a gate-specific audit.

        This invokes the same encoder and helper used by ``forward`` and
        returns detached CUDA tensors; it is not a second implementation.
        """

        legal = set(
            enumerate_legal_actions(
                state,
                min_length=self.min_length,
                max_length=self.max_length,
                include_stop=True,
            )
        )
        if action not in legal:
            raise ValueError("cannot audit a hard-masked action")
        _, current_tokens, aligned_tokens, _ = self._encoded_state(state, time)
        current_local, source_aligned_local = self._local_representations(
            state, action, current_tokens, aligned_tokens
        )
        return {
            "current_local": current_local.detach().clone(),
            "source_aligned_local": source_aligned_local.detach().clone(),
        }


def full_reencode_equivalence(
    rate_field: FoundationFusionRateField,
    state: EditState,
    time: float,
) -> bool:
    """Reference-mode equivalence: incremental optimization is disabled."""

    first = rate_field(state, time)
    second = rate_field(state, time)
    return all(torch.equal(first[action], second[action]) for action in first)


class OfficialPaperRateAdapter:
    """Auditable paper-reference route backed only by the official FM0 model.

    The paper sampler still receives ordinary numeric rates, but every rate
    evaluation crosses this adapter and is classified from the instantiated
    foundation class/module.  Unknown, dummy, placeholder, or project-local
    encoders are rejected before the first sampler step.
    """

    OFFICIAL_CLASS = "UtrLmModel"
    OFFICIAL_MODULE_PREFIX = "multimolecule"

    def __init__(self, rate_field: FoundationFusionRateField) -> None:
        if not isinstance(rate_field, FoundationFusionRateField):
            raise TypeError("paper mode requires FoundationFusionRateField")
        foundation_type = type(rate_field.foundation)
        self.foundation_class = foundation_type.__name__
        self.foundation_module = foundation_type.__module__
        if (
            self.foundation_class != self.OFFICIAL_CLASS
            or not self.foundation_module.startswith(self.OFFICIAL_MODULE_PREFIX)
        ):
            raise RuntimeError(
                "paper mode requires the official frozen UtrLmModel; "
                "placeholder/project-local foundations are forbidden"
            )
        self.rate_field = rate_field
        self.telemetry: dict[str, Any] = {
            "paper_rate_calls": 0,
            "official_foundation_forward_calls": 0,
            "placeholder_foundation_forward_calls": 0,
            "foundation_class": self.foundation_class,
            "foundation_module": self.foundation_module,
        }

    def __call__(self, state: EditState, time: float) -> dict[AtomicAction, float]:
        before = (
            self.rate_field.source_encode_calls + self.rate_field.current_encode_calls
        )
        tensor_rates = self.rate_field(state, time)
        after = (
            self.rate_field.source_encode_calls + self.rate_field.current_encode_calls
        )
        forward_delta = after - before
        if forward_delta <= 0:
            raise RuntimeError(
                "paper-mode rate call observed no official foundation forward"
            )
        self.telemetry["paper_rate_calls"] += 1
        self.telemetry["official_foundation_forward_calls"] += forward_delta
        return {
            action: float(rate.detach().to(device="cpu", dtype=torch.float64).item())
            for action, rate in tensor_rates.items()
        }
