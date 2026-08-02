"""Fail-closed boundary between training-only coupling data and rate inputs.

The paired target/coupling auxiliary is required to construct transition-level
training weights, but it is never part of the inference-visible state passed to
the neural rate field.  This module makes that separation explicit and gives
the M05 audit a real pipeline boundary to exercise.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from .types import EditState


@dataclass(frozen=True)
class EditFlowTrainingExample:
    inference_state: EditState
    training_auxiliary: Any

    def __post_init__(self) -> None:
        if not isinstance(self.inference_state, EditState):
            raise TypeError("training example requires a canonical EditState")


def rate_input_state(example: EditFlowTrainingExample) -> EditState:
    """Return the sole object the rate network may receive."""

    if not isinstance(example, EditFlowTrainingExample):
        raise TypeError("rate input must cross the EditFlowTrainingExample boundary")
    return example.inference_state


def canonical_rate_input_bytes(example: EditFlowTrainingExample) -> bytes:
    """Canonical bytes used to prove auxiliary replacement non-interference."""

    return (
        json.dumps(
            rate_input_state(example).inference_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
