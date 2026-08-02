"""Trainable EF0 rate field built on the frozen MK0 foundation fusion.

The public model interface is intentionally inference-state only:

    rate(a | x_current, x_source, region, context, target, budget, time)

Training-only target alignments and remaining target edits are not accepted by
this module.  The implementation subclasses the frozen MK0 fusion so source
encoding, dynamic-current re-encoding, source/current mapping and CUDA guards
remain the single production path.  EF0 adds explicit operation heads and
region adapters on top of that path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from ..mk0.foundation_fusion import (
    FoundationFusionRateField,
    _stable_unit_interval,
    require_neural_cuda,
)
from ..mk0.state_action import enumerate_legal_actions
from ..mk0.types import ActionType, AtomicAction, EditState


@dataclass(frozen=True)
class EF0ModelConfig:
    """Frozen engineering defaults for an EF0 tiny/formal model instance."""

    min_length: int = 1
    max_length: int = 256
    hidden_head_width: int = 128

    def __post_init__(self) -> None:
        if self.min_length < 1 or self.max_length < self.min_length:
            raise ValueError("invalid EF0 length bounds")
        if self.hidden_head_width < 4:
            raise ValueError("EF0 head width is too small")


class TrueUTREditFlowRateField(FoundationFusionRateField):
    """Source-conditioned, region-aware, non-negative action-rate field.

    ``FoundationFusionRateField`` supplies frozen official UTR-LM features for
    the source, the dynamically changing current sequence, and the source
    mapping at every rate evaluation.  EF0 replaces its single action-aware
    head with four explicit operation heads and a learned two-region adapter.
    Hard legality is still applied by MK0 before an action is returned.
    """

    _EXPLICIT_CONDITION_WIDTH = 10

    def __init__(
        self,
        foundation: nn.Module,
        tokenizer,
        *,
        device: torch.device | str,
        config: EF0ModelConfig | None = None,
        hidden_size: Optional[int] = None,
    ) -> None:
        self.config = config or EF0ModelConfig()
        super().__init__(
            foundation,
            tokenizer,
            device=device,
            min_length=self.config.min_length,
            max_length=self.config.max_length,
            hidden_size=hidden_size,
        )
        # The parent head is retained only for MK0 API compatibility.  EF0
        # uses explicit operation heads below; there is no hidden fallback to
        # one undifferentiated scorer.
        self.rate_head = nn.Identity()
        input_width = (
            5 * self.hidden_size
            + 17
            + self._EXPLICIT_CONDITION_WIDTH
        )
        self.action_heads = nn.ModuleDict(
            {
                action_type.value: nn.Sequential(
                    nn.Linear(input_width, self.config.hidden_head_width),
                    nn.SiLU(),
                    nn.Linear(self.config.hidden_head_width, 1),
                )
                for action_type in ActionType
            }
        )
        # Rows are 5UTR and 3UTR; columns are INS/SUB/DEL/STOP.  The positive
        # gate keeps rates non-negative without post-hoc clipping.
        self.region_adapter = nn.Parameter(
            torch.zeros(2, len(ActionType), device=self.device, dtype=torch.float32)
        )
        self._runtime_forward_calls = 0
        self.to(self.device, dtype=torch.float32)

    @staticmethod
    def _context_feature(value: object) -> float:
        if value is None:
            value = "unspecified"
        return _stable_unit_interval(str(value))

    def _explicit_condition_features(self, state: EditState) -> torch.Tensor:
        context = dict(state.context)
        region = (1.0, 0.0) if state.region == "5UTR" else (0.0, 1.0)
        target = tuple(
            1.0 if state.target_condition == name else 0.0
            for name in ("increase", "decrease", "maintain", "interval")
        )
        context_features = tuple(
            self._context_feature(context.get(name))
            for name in ("assay", "cell_or_tissue", "endpoint", "batch")
        )
        values = region + target + context_features
        if len(values) != self._EXPLICIT_CONDITION_WIDTH:
            raise AssertionError("EF0 condition feature width drifted")
        return torch.tensor(values, device=self.device, dtype=torch.float32)

    @staticmethod
    def _region_index(state: EditState) -> int:
        return 0 if state.region == "5UTR" else 1

    def _region_gate(self, state: EditState, action: AtomicAction) -> torch.Tensor:
        # 0.25 < gate < 1.75, so the adapter cannot create a negative or NaN
        # rate and never needs a silent rate clip.
        raw = self.region_adapter[
            self._region_index(state), list(ActionType).index(action.kind)
        ]
        return 0.25 + 1.5 * torch.sigmoid(raw)

    def forward(
        self,
        state: EditState,
        time: float,
        actions: Optional[Sequence[AtomicAction]] = None,
    ) -> dict[AtomicAction, torch.Tensor]:
        require_neural_cuda(self.device)
        if next(self.action_heads.parameters()).device.type != "cuda":
            raise RuntimeError("EF0 action heads silently left CUDA")
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
            else tuple(sorted(legal, key=lambda action: action.key))
        )
        if any(action not in legal for action in selected):
            raise ValueError("requested action is hard-masked by EF0")
        _, current_tokens, aligned_tokens, shared = self._encoded_state(state, time)
        condition = self._explicit_condition_features(state)
        result: dict[AtomicAction, torch.Tensor] = {}
        for action in selected:
            current_local, source_aligned_local = self._local_representations(
                state, action, current_tokens, aligned_tokens
            )
            features = torch.cat(
                (
                    shared,
                    current_local,
                    source_aligned_local,
                    self._action_features(state, action),
                    condition,
                )
            )
            raw = self.action_heads[action.kind.value](features).squeeze()
            rate = F.softplus(raw) * self._region_gate(state, action)
            if rate.device.type != "cuda" or not bool(torch.isfinite(rate)):
                raise FloatingPointError("EF0 produced invalid CUDA action rate")
            if float(rate.detach().cpu()) < 0.0:
                raise FloatingPointError("EF0 produced a negative action rate")
            result[action] = rate
        self._runtime_forward_calls += 1
        return result

    def rate_fn(
        self, state: EditState, time: float
    ) -> dict[AtomicAction, float]:
        """Convert the CUDA rate field to the MK0 sampler scalar interface."""

        with torch.no_grad():
            tensors = self.forward(state, time)
        result = {
            action: float(rate.detach().to(device="cpu", dtype=torch.float64).item())
            for action, rate in tensors.items()
        }
        if any(not math.isfinite(value) or value < 0.0 for value in result.values()):
            raise FloatingPointError("EF0 numeric rate conversion failed")
        return result

    def runtime_device_audit(self) -> dict[str, object]:
        trainable = [parameter for parameter in self.parameters() if parameter.requires_grad]
        frozen = [parameter for parameter in self.parameters() if not parameter.requires_grad]
        return {
            "cuda_available": bool(torch.cuda.is_available()),
            "model_device": str(self.device),
            "trainable_parameter_count": int(sum(parameter.numel() for parameter in trainable)),
            "frozen_parameter_count": int(sum(parameter.numel() for parameter in frozen)),
            "trainable_parameters_cuda": all(parameter.device.type == "cuda" for parameter in trainable),
            "frozen_parameters_cuda": all(parameter.device.type == "cuda" for parameter in frozen),
            "foundation_requires_grad_count": int(
                sum(parameter.requires_grad for parameter in self.foundation.parameters())
            ),
            "runtime_forward_calls": self._runtime_forward_calls,
        }


class TrueUTREditFlow(nn.Module):
    """Small public wrapper exposing the required EF0 model interface."""

    def __init__(self, rate_field: TrueUTREditFlowRateField) -> None:
        super().__init__()
        self.rate_field = rate_field

    @property
    def inference_signature_fields(self) -> tuple[str, ...]:
        return self.rate_field.inference_signature_fields

    def forward(
        self,
        state: EditState,
        time: float,
        actions: Optional[Sequence[AtomicAction]] = None,
    ) -> dict[AtomicAction, torch.Tensor]:
        return self.rate_field(state, time, actions=actions)

    def rate_fn(self, state: EditState, time: float) -> dict[AtomicAction, float]:
        return self.rate_field.rate_fn(state, time)

    def runtime_device_audit(self) -> dict[str, object]:
        return self.rate_field.runtime_device_audit()
