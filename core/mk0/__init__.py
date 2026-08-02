"""MK0-v1 mathematical kernel for UTR Edit Flow.

This package freezes E0 mathematical and engineering semantics only.  Its
samplers are first-order approximations; none is an exact Gillespie sampler.
Training-only alignment variables are deliberately absent from every rate
field interface.
"""

from .types import (
    ALPHABET,
    ActionType,
    AtomicAction,
    EditHistory,
    EditState,
    Phase,
    RuntimeMapping,
    TerminationReason,
    TokenRef,
)
from .target_kernel import (
    TargetKernelRejected,
    TargetPathLedgerEntry,
    TargetTransition,
    TargetTransitionOracle,
    build_target_transition_oracle,
)
from .bregman import edit_flow_loss

__all__ = [
    "ALPHABET",
    "ActionType",
    "AtomicAction",
    "EditHistory",
    "EditState",
    "Phase",
    "RuntimeMapping",
    "TerminationReason",
    "TokenRef",
    "TargetKernelRejected",
    "TargetPathLedgerEntry",
    "TargetTransition",
    "TargetTransitionOracle",
    "build_target_transition_oracle",
    "edit_flow_loss",
]
