"""EF0 true UTR Edit Flow engineering implementation.

This package is deliberately downstream of the frozen ``core.mk0`` kernel.
It adds the trainable region-aware rate field and a thin generation API while
leaving the MK0 state/action, legality and sampler semantics unchanged.
"""

from .model import EF0ModelConfig, TrueUTREditFlow, TrueUTREditFlowRateField
from .sampler import EF0SamplerConfig, generate_candidates

__all__ = [
    "EF0ModelConfig",
    "TrueUTREditFlow",
    "TrueUTREditFlowRateField",
    "EF0SamplerConfig",
    "generate_candidates",
]
