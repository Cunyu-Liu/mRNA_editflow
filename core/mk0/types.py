"""Immutable state and action types for the MK0 extended-state CTMC."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Mapping, Optional, Tuple

ALPHABET: Tuple[str, ...] = ("A", "C", "G", "U")
INFERENCE_CONTEXT_KEYS: Tuple[str, ...] = (
    "assay",
    "cell_or_tissue",
    "endpoint",
    "batch",
)
TARGET_DIRECTIONS: Tuple[str, ...] = (
    "increase",
    "decrease",
    "maintain",
    "interval",
)


class ActionType(str, Enum):
    INS = "INS"
    SUB = "SUB"
    DEL = "DEL"
    STOP = "STOP"


class Phase(str, Enum):
    ACTIVE = "ACTIVE"
    HALTED = "HALTED"


class TerminationReason(str, Enum):
    LEARNED_STOP = "LEARNED_STOP"
    FORCED_BUDGET = "FORCED_BUDGET"
    FORCED_NO_LEGAL_EDIT_ACTION = "FORCED_NO_LEGAL_EDIT_ACTION"
    FORCED_ZERO_REMAINING_INTEGRATED_HAZARD = "FORCED_ZERO_REMAINING_INTEGRATED_HAZARD"
    FORCED_TIME_HORIZON = "FORCED_TIME_HORIZON"
    FAILED_NUMERICAL = "FAILED_NUMERICAL"


class TokenOrigin(str, Enum):
    SOURCE = "SOURCE"
    INSERTED = "INSERTED"


@dataclass(frozen=True)
class AtomicAction:
    """One current-coordinate edit or STOP.

    ``position`` is a zero-based gap for INS and a zero-based token index for
    SUB/DEL.  STOP has position and token set to ``None``.
    """

    kind: ActionType
    position: Optional[int] = None
    token: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ActionType):
            raise ValueError("action kind must be an ActionType")
        if self.kind == ActionType.STOP:
            if self.position is not None or self.token is not None:
                raise ValueError("STOP must not carry position or token")
            return
        if not isinstance(self.position, int) or self.position < 0:
            raise ValueError("edit action position must be a non-negative integer")
        if self.kind in (ActionType.INS, ActionType.SUB):
            if self.token not in ALPHABET:
                raise ValueError("INS/SUB token must be one of A,C,G,U")
        elif self.token is not None:
            raise ValueError("DEL must not carry a token")

    @property
    def key(self) -> str:
        if self.kind == ActionType.STOP:
            return "STOP"
        suffix = f":{self.token}" if self.token is not None else ""
        return f"{self.kind.value}:{self.position}{suffix}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "position": self.position,
            "token": self.token,
        }


@dataclass(frozen=True)
class TokenRef:
    origin: TokenOrigin
    stable_id: str
    source_index: Optional[int]
    protected: bool = False

    def __post_init__(self) -> None:
        if self.origin == TokenOrigin.SOURCE and self.source_index is None:
            raise ValueError("source token requires source_index")
        if self.origin == TokenOrigin.INSERTED and self.source_index is not None:
            raise ValueError("inserted token cannot have source_index")


@dataclass(frozen=True)
class RuntimeMapping:
    """Inference-visible source/current mapping.

    It stores only source identities, inserted-event identities and protection
    flags.  No target or target-alignment field exists by construction.
    """

    tokens: Tuple[TokenRef, ...]
    gap_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.gap_ids) != len(self.tokens) + 1:
            raise ValueError("mapping must expose exactly L+1 stable gaps")

    @staticmethod
    def _gap_ids(tokens: Tuple[TokenRef, ...]) -> Tuple[str, ...]:
        ids = ("LEFT",) + tuple(token.stable_id for token in tokens) + ("RIGHT",)
        return tuple(f"gap:{ids[i]}|{ids[i + 1]}" for i in range(len(tokens) + 1))

    @classmethod
    def from_source(
        cls, source: str, protected_indices: Tuple[int, ...] = ()
    ) -> "RuntimeMapping":
        protected = set(protected_indices)
        tokens = tuple(
            TokenRef(TokenOrigin.SOURCE, f"src:{i}", i, i in protected)
            for i in range(len(source))
        )
        return cls(tokens=tokens, gap_ids=cls._gap_ids(tokens))

    @classmethod
    def rebuild(cls, tokens: Tuple[TokenRef, ...]) -> "RuntimeMapping":
        return cls(tokens=tokens, gap_ids=cls._gap_ids(tokens))


@dataclass(frozen=True)
class EditHistory:
    executed: int = 0
    ins: int = 0
    sub: int = 0
    delete: int = 0
    action_keys: Tuple[str, ...] = ()

    def append(self, action: AtomicAction) -> "EditHistory":
        if action.kind == ActionType.STOP:
            return self
        return EditHistory(
            executed=self.executed + 1,
            ins=self.ins + int(action.kind == ActionType.INS),
            sub=self.sub + int(action.kind == ActionType.SUB),
            delete=self.delete + int(action.kind == ActionType.DEL),
            action_keys=self.action_keys + (action.key,),
        )


def _frozen_context(
    value: Optional[Mapping[str, Any]],
) -> Tuple[Tuple[str, Optional[str]], ...]:
    """Validate and freeze the entire inference-visible context.

    This is a security/scientific boundary, not a convenience serializer.  An
    unknown key could otherwise carry a target sequence, alignment-derived
    feature, or remaining-target-edit count into the rate network and then be
    silently omitted from the public state artifact.  We therefore reject
    unknown keys at construction time and materialize all schema keys.
    """

    supplied = {} if value is None else dict(value)
    if any(not isinstance(key, str) for key in supplied):
        raise ValueError("context keys must be strings")
    unknown = set(supplied) - set(INFERENCE_CONTEXT_KEYS)
    if unknown:
        raise ValueError(
            "context contains non-inference-visible keys: " + ", ".join(sorted(unknown))
        )
    normalized: dict[str, Optional[str]] = {
        "assay": "unspecified",
        "cell_or_tissue": "unspecified",
        "endpoint": "unspecified",
        "batch": None,
    }
    normalized.update(supplied)
    for key in ("assay", "cell_or_tissue", "endpoint"):
        if not isinstance(normalized[key], str) or not normalized[key]:
            raise ValueError(f"context.{key} must be a non-empty string")
    batch = normalized["batch"]
    if batch is not None and (not isinstance(batch, str) or not batch):
        raise ValueError("context.batch must be null or a non-empty string")
    return tuple((key, normalized[key]) for key in INFERENCE_CONTEXT_KEYS)


@dataclass(frozen=True)
class EditState:
    source: str
    current: str
    mapping: RuntimeMapping
    region: str
    context: Tuple[Tuple[str, Optional[str]], ...]
    target_condition: str
    initial_budget: int
    remaining_budget: int
    history: EditHistory = field(default_factory=EditHistory)
    phase: Phase = Phase.ACTIVE
    termination_reason: Optional[TerminationReason] = None

    def __post_init__(self) -> None:
        if any(token not in ALPHABET for token in self.source + self.current):
            raise ValueError("source/current must use the RNA alphabet A,C,G,U")
        if len(self.mapping.tokens) != len(self.current):
            raise ValueError("mapping token count must equal current length")
        if self.region not in ("5UTR", "3UTR"):
            raise ValueError("region must be 5UTR or 3UTR")
        # Direct dataclass construction must enforce the same boundary as
        # ``initial``; callers cannot bypass it by fabricating frozen pairs.
        if self.context != _frozen_context(dict(self.context)):
            raise ValueError("context is not in canonical inference-visible form")
        if self.target_condition not in TARGET_DIRECTIONS:
            raise ValueError("target_condition must be an inference-visible direction")
        if (
            self.initial_budget < 0
            or not 0 <= self.remaining_budget <= self.initial_budget
        ):
            raise ValueError("invalid edit budget")
        if self.remaining_budget != self.initial_budget - self.history.executed:
            raise ValueError("remaining budget must equal B minus executed edits")
        if self.phase == Phase.ACTIVE and self.termination_reason is not None:
            raise ValueError("ACTIVE state cannot have a termination reason")
        if self.phase == Phase.HALTED and self.termination_reason is None:
            raise ValueError("HALTED state requires a termination reason")

    @classmethod
    def initial(
        cls,
        source: str,
        *,
        region: str = "5UTR",
        context: Optional[Mapping[str, Any]] = None,
        target_condition: str = "increase",
        budget: int = 4,
        protected_indices: Tuple[int, ...] = (),
    ) -> "EditState":
        return cls(
            source=source,
            current=source,
            mapping=RuntimeMapping.from_source(source, protected_indices),
            region=region,
            context=_frozen_context(context),
            target_condition=target_condition,
            initial_budget=budget,
            remaining_budget=budget,
        )

    def inference_dict(self) -> dict[str, Any]:
        """Canonical inference-visible payload; deliberately excludes target data."""

        return {
            "source": self.source,
            "current": self.current,
            "mapping": {
                "tokens": [
                    {
                        "origin": token.origin.value,
                        "stable_id": token.stable_id,
                        "source_index": token.source_index,
                        "protected": token.protected,
                    }
                    for token in self.mapping.tokens
                ],
                "gap_ids": list(self.mapping.gap_ids),
            },
            "region": self.region,
            "context": dict(self.context),
            "target_condition": self.target_condition,
            "initial_budget": self.initial_budget,
            "remaining_budget": self.remaining_budget,
            "history": asdict(self.history),
            "phase": self.phase.value,
            "termination_reason": (
                self.termination_reason.value if self.termination_reason else None
            ),
        }

    @property
    def state_hash(self) -> str:
        payload = json.dumps(
            self.inference_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class AppliedTransition:
    before: EditState
    action: AtomicAction
    after: EditState

    def __post_init__(self) -> None:
        if self.before.phase != Phase.ACTIVE:
            raise ValueError("transition must start from ACTIVE state")
