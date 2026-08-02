"""EF0 generation wrapper around the frozen MK0 constrained sampler."""

from __future__ import annotations

from dataclasses import dataclass

from ..mk0.samplers import (
    RemainingHazardCertificate,
    SamplerResult,
    constrained_single_event_first_order,
)
from ..mk0.types import EditState
from .model import TrueUTREditFlow


@dataclass(frozen=True)
class EF0SamplerConfig:
    step_size: float = 0.0078125
    stability_hazard: float = 0.05
    min_length: int = 1
    max_length: int = 256
    horizon: float = 1.0
    remaining_hazard_zero_atol: float = 1.0e-10
    remaining_hazard_convergence_atol: float = 1.0e-10

    def __post_init__(self) -> None:
        if self.step_size <= 0.0 or self.stability_hazard <= 0.0:
            raise ValueError("EF0 sampler step and stability values must be positive")
        if self.min_length < 1 or self.max_length < self.min_length:
            raise ValueError("invalid EF0 sampler length bounds")
        if not 0.0 < self.horizon <= 1.0:
            raise ValueError("EF0 sampler horizon must be in (0,1]")


def generate_candidates(
    flow: TrueUTREditFlow,
    initial_state: EditState,
    *,
    config: EF0SamplerConfig,
    seed: int,
    remaining_hazard_verifier=None,
) -> SamplerResult:
    """Generate with MK0's hard-constrained single-event approximation.

    Each rate callback is a real CUDA forward through the source/current
    foundation fusion.  The sampler itself remains the frozen first-order
    approximation and is never described as exact Gillespie.
    """

    return constrained_single_event_first_order(
        initial_state,
        flow.rate_fn,
        step_size=config.step_size,
        stability_hazard=config.stability_hazard,
        min_length=config.min_length,
        max_length=config.max_length,
        seed=seed,
        horizon=config.horizon,
        remaining_hazard_verifier=remaining_hazard_verifier,
        remaining_hazard_zero_atol=config.remaining_hazard_zero_atol,
        remaining_hazard_convergence_atol=config.remaining_hazard_convergence_atol,
    )
