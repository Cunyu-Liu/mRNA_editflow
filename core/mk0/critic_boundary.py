"""Fail-closed role separation between generator, critic and final evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .samplers import RateFunction, SamplerResult, constrained_single_event_first_order
from .types import EditState


@dataclass(frozen=True)
class RoleAudit:
    critic_present: bool
    final_evaluator_queries: int
    guidance_queries: int

    @property
    def pass_no_final_evaluator_guidance(self) -> bool:
        return self.final_evaluator_queries == 0


def base_generation_without_critic(
    initial_state: EditState,
    rate_fn: RateFunction,
    **sampler_kwargs,
) -> tuple[SamplerResult, RoleAudit]:
    forbidden = {
        "critic",
        "evaluator",
        "final_evaluator",
        "guidance",
        "reward",
        "reranker",
        "selector",
        "score_fn",
    }
    prohibited = sorted(forbidden.intersection(sampler_kwargs))
    if prohibited:
        raise PermissionError(
            "base MK0 generation forbids role-bearing arguments: "
            + ", ".join(prohibited)
        )
    result = constrained_single_event_first_order(
        initial_state, rate_fn, **sampler_kwargs
    )
    return result, RoleAudit(False, final_evaluator_queries=0, guidance_queries=0)


def reject_final_evaluator_as_guidance(
    final_evaluator: Optional[Callable[..., float]], *, as_guidance: bool
) -> None:
    if final_evaluator is not None and as_guidance:
        raise PermissionError("E_final may not guide, select or train the generator")
