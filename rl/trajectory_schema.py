"""Serializable offline DAgger trajectories and potential-reward labels.

These schemas deliberately record *what the policy visited* separately from
the Oracle labels added afterwards.  The rollout code never receives an Oracle;
the Oracle is used only while relabelling saved states offline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class TrajectoryAction:
    """One constrained decoder action, including the explicit STOP action."""

    op: str
    pos: Optional[int]
    nt: str = ""
    log_score: float = 0.0


@dataclass(frozen=True)
class TrajectoryState:
    """A full state snapshot after zero or more policy actions."""

    transcript_id: str
    sequence: str
    five_utr: str
    cds: str
    three_utr: str
    step_index: int
    species: str = "human"

    def record_dict(self) -> dict[str, object]:
        return {
            "transcript_id": self.transcript_id,
            "five_utr": self.five_utr,
            "cds": self.cds,
            "three_utr": self.three_utr,
            "species": self.species,
            "metadata": {"dagger_step_index": self.step_index},
        }


@dataclass
class OfflineTrajectory:
    """A policy-only rollout with enough provenance to reproduce relabelling."""

    source_transcript_id: str
    task_id: str
    policy_checkpoint_sha256: str
    states: list[TrajectoryState]
    action_history: list[TrajectoryAction] = field(default_factory=list)
    terminated: bool = False
    termination_reason: str = "edit_budget"
    cycle_rejections: int = 0
    rollout_temperature: float = 1.0
    rollout_top_k: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "source_transcript_id": self.source_transcript_id,
            "task_id": self.task_id,
            "policy_checkpoint_sha256": self.policy_checkpoint_sha256,
            "states": [asdict(state) for state in self.states],
            "action_history": [asdict(action) for action in self.action_history],
            "terminated": bool(self.terminated),
            "termination_reason": self.termination_reason,
            "cycle_rejections": int(self.cycle_rejections),
            "rollout_temperature": float(self.rollout_temperature),
            "rollout_top_k": int(self.rollout_top_k),
            # A rollout has no Oracle action guidance by contract.
            "oracle_used_for_action_selection": False,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "OfflineTrajectory":
        raw_states = value.get("states", [])
        raw_actions = value.get("action_history", [])
        if not isinstance(raw_states, Sequence) or isinstance(raw_states, (str, bytes)):
            raise ValueError("trajectory states must be a sequence")
        if not isinstance(raw_actions, Sequence) or isinstance(raw_actions, (str, bytes)):
            raise ValueError("trajectory action_history must be a sequence")
        states = [TrajectoryState(**dict(item)) for item in raw_states if isinstance(item, Mapping)]
        actions = [TrajectoryAction(**dict(item)) for item in raw_actions if isinstance(item, Mapping)]
        if not states:
            raise ValueError("trajectory must contain its initial state")
        policy_hash = str(value.get("policy_checkpoint_sha256", ""))
        if not policy_hash:
            raise ValueError("trajectory is missing policy_checkpoint_sha256")
        return cls(
            source_transcript_id=str(value["source_transcript_id"]),
            task_id=str(value.get("task_id", "T5")).upper(),
            policy_checkpoint_sha256=policy_hash,
            states=states,
            action_history=actions,
            terminated=bool(value.get("terminated", False)),
            termination_reason=str(value.get("termination_reason", "edit_budget")),
            cycle_rejections=int(value.get("cycle_rejections", 0)),
            rollout_temperature=float(value.get("rollout_temperature", 1.0)),
            rollout_top_k=int(value.get("rollout_top_k", 1)),
        )


@dataclass(frozen=True)
class TrajectoryTeacherRow:
    """A one-step offline teacher label rooted at a visited rollout state."""

    source_transcript_id: str
    state_transcript_id: str
    state_sequence: str
    step_index: int
    policy_checkpoint_sha256: str
    action_history: tuple[TrajectoryAction, ...]
    candidate_action: Mapping[str, object]
    reward_vector: Mapping[str, object]
    raw_delta: Mapping[str, float]
    raw_absolute_level: Mapping[str, float]
    normalized_within_group: Mapping[str, float]
    source_properties: Mapping[str, float]
    state_properties: Mapping[str, float]
    candidate_properties: Mapping[str, float]
    potential_before: float
    potential_after: float
    potential_reward: float
    terminated: bool
    teacher_payload: Mapping[str, object]
    replay_source: str = "rollout"

    def to_dict(self) -> dict[str, object]:
        """Emit a ranker-compatible row augmented with trajectory provenance."""
        row = dict(self.teacher_payload)
        row.update(
            {
                "transcript_id": self.state_transcript_id,
                "source_transcript_id": self.source_transcript_id,
                "state_sequence": self.state_sequence,
                "step_index": int(self.step_index),
                "policy_checkpoint_sha256": self.policy_checkpoint_sha256,
                "action_history": [asdict(action) for action in self.action_history],
                "candidate_action": dict(self.candidate_action),
                "reward_vector": dict(self.reward_vector),
                "raw_delta": {str(k): float(v) for k, v in self.raw_delta.items()},
                "raw_delta_from_source": {str(k): float(v) for k, v in self.raw_delta.items()},
                "raw_absolute_level": {str(k): float(v) for k, v in self.raw_absolute_level.items()},
                "normalized_within_group": {str(k): float(v) for k, v in self.normalized_within_group.items()},
                "source_properties": {str(k): float(v) for k, v in self.source_properties.items()},
                "state_properties": {str(k): float(v) for k, v in self.state_properties.items()},
                "candidate_properties": {str(k): float(v) for k, v in self.candidate_properties.items()},
                "potential_before": float(self.potential_before),
                "potential_after": float(self.potential_after),
                "potential_reward": float(self.potential_reward),
                "terminated": bool(self.terminated),
                "replay_source": self.replay_source,
                "oracle_used_for_action_selection": False,
            }
        )
        return row


def potential(properties: Mapping[str, float], *, objective: str = "te") -> float:
    """Return the declared scalar potential used only for transition labels."""
    if objective not in properties:
        raise ValueError(f"potential property {objective!r} is missing")
    return float(properties[objective])


def potential_reward(before: Mapping[str, float], after: Mapping[str, float], *, objective: str = "te") -> float:
    """Potential-based offline reward ``Phi(after) - Phi(before)``."""
    return potential(after, objective=objective) - potential(before, objective=objective)


def telescoping_error(properties: Sequence[Mapping[str, float]], *, objective: str = "te") -> float:
    """Return ``sum_t r_t - (Phi(final)-Phi(initial))`` for a state path."""
    if len(properties) < 2:
        return 0.0
    rewards = sum(
        potential_reward(before, after, objective=objective)
        for before, after in zip(properties, properties[1:])
    )
    return rewards - (potential(properties[-1], objective=objective) - potential(properties[0], objective=objective))


def validate_telescoping(
    properties: Sequence[Mapping[str, float]], *, objective: str = "te", atol: float = 1e-8
) -> bool:
    """Check potential telescoping with an explicit numerical tolerance."""
    return abs(telescoping_error(properties, objective=objective)) <= float(atol)


__all__ = [
    "TrajectoryAction", "TrajectoryState", "OfflineTrajectory", "TrajectoryTeacherRow",
    "potential", "potential_reward", "telescoping_error", "validate_telescoping",
]
