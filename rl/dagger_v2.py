"""Small DAgger data contract; expert labels are legal-action indices."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class DAggerExample:
    state_key: tuple
    legal_action_keys: tuple[tuple, ...]
    expert_action_index: int
    iteration: int


class DAggerBuffer:
    def __init__(self) -> None:
        self.examples: List[DAggerExample] = []

    def add(self, example: DAggerExample) -> None:
        if not (0 <= example.expert_action_index < len(example.legal_action_keys)):
            raise ValueError("expert action must be in legal action set")
        self.examples.append(example)

    def __len__(self) -> int:
        return len(self.examples)
