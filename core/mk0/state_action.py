"""Hard legality, deterministic extended-state updates, undo and replay."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from .types import (
    ALPHABET,
    ActionType,
    AppliedTransition,
    AtomicAction,
    EditState,
    Phase,
    RuntimeMapping,
    TerminationReason,
    TokenOrigin,
    TokenRef,
)


class IllegalAction(ValueError):
    pass


def _protected_gap(state: EditState, gap: int) -> bool:
    left = gap > 0 and state.mapping.tokens[gap - 1].protected
    right = gap < len(state.current) and state.mapping.tokens[gap].protected
    return left and right


def is_legal(
    state: EditState,
    action: AtomicAction,
    *,
    min_length: int,
    max_length: int,
) -> bool:
    if state.phase != Phase.ACTIVE:
        return False
    if action.kind == ActionType.STOP:
        return True
    if state.remaining_budget <= 0:
        return False
    pos = int(action.position)  # non-None by AtomicAction validation
    if action.kind == ActionType.INS:
        return (
            len(state.current) < max_length
            and pos <= len(state.current)
            and not _protected_gap(state, pos)
        )
    if pos >= len(state.current):
        return False
    if state.mapping.tokens[pos].protected:
        return False
    if action.kind == ActionType.SUB:
        return action.token in ALPHABET and action.token != state.current[pos]
    if action.kind == ActionType.DEL:
        return len(state.current) > min_length
    return False


def enumerate_legal_actions(
    state: EditState,
    *,
    min_length: int,
    max_length: int,
    include_stop: bool = True,
) -> tuple[AtomicAction, ...]:
    if state.phase != Phase.ACTIVE:
        return ()
    candidates: list[AtomicAction] = []
    for gap in range(len(state.current) + 1):
        for token in ALPHABET:
            candidates.append(AtomicAction(ActionType.INS, gap, token))
    for pos, old_token in enumerate(state.current):
        for token in ALPHABET:
            if token != old_token:
                candidates.append(AtomicAction(ActionType.SUB, pos, token))
        candidates.append(AtomicAction(ActionType.DEL, pos))
    if include_stop:
        candidates.append(AtomicAction(ActionType.STOP))
    return tuple(
        action
        for action in candidates
        if is_legal(state, action, min_length=min_length, max_length=max_length)
    )


def apply_action(
    state: EditState,
    action: AtomicAction,
    *,
    min_length: int,
    max_length: int,
) -> AppliedTransition:
    """Apply an action exactly at fixed external time.

    The external CTMC time is intentionally not stored or modified here.
    """

    if not is_legal(state, action, min_length=min_length, max_length=max_length):
        raise IllegalAction(f"hard-masked action: {action.key}")
    if action.kind == ActionType.STOP:
        after = replace(
            state,
            phase=Phase.HALTED,
            termination_reason=TerminationReason.LEARNED_STOP,
        )
        return AppliedTransition(state, action, after)

    pos = int(action.position)
    sequence = list(state.current)
    refs = list(state.mapping.tokens)
    next_event_id = state.history.executed + 1
    if action.kind == ActionType.INS:
        sequence.insert(pos, str(action.token))
        refs.insert(
            pos,
            TokenRef(
                TokenOrigin.INSERTED,
                f"ins:{next_event_id}",
                source_index=None,
                protected=False,
            ),
        )
    elif action.kind == ActionType.SUB:
        sequence[pos] = str(action.token)
    elif action.kind == ActionType.DEL:
        del sequence[pos]
        del refs[pos]
    else:  # pragma: no cover - exhaustive enum guard
        raise IllegalAction(f"unsupported action: {action.kind}")
    history = state.history.append(action)
    after = replace(
        state,
        current="".join(sequence),
        mapping=RuntimeMapping.rebuild(tuple(refs)),
        remaining_budget=state.remaining_budget - 1,
        history=history,
    )
    return AppliedTransition(state, action, after)


def undo_transition(transition: AppliedTransition) -> EditState:
    """Exact inverse using the audited pre-action extended state snapshot."""

    return transition.before


def replay_actions(
    initial: EditState,
    actions: Sequence[AtomicAction],
    *,
    min_length: int,
    max_length: int,
) -> tuple[EditState, tuple[AppliedTransition, ...]]:
    current = initial
    records: list[AppliedTransition] = []
    for action in actions:
        record = apply_action(
            current, action, min_length=min_length, max_length=max_length
        )
        records.append(record)
        current = record.after
    return current, tuple(records)


def force_terminate(state: EditState, reason: TerminationReason) -> EditState:
    if reason == TerminationReason.LEARNED_STOP:
        raise ValueError("LEARNED_STOP must be produced by the STOP action")
    if state.phase != Phase.ACTIVE:
        raise ValueError("cannot force-terminate an already halted state")
    return replace(state, phase=Phase.HALTED, termination_reason=reason)


def action_mask(
    state: EditState,
    actions: Iterable[AtomicAction],
    *,
    min_length: int,
    max_length: int,
) -> tuple[bool, ...]:
    return tuple(
        is_legal(state, action, min_length=min_length, max_length=max_length)
        for action in actions
    )


def termination_to_schema_record(
    *,
    reason: TerminationReason,
    external_time: float,
    state_hash_before: str,
    state_hash_after: str,
    instantaneous_edit_hazard: float = 0.0,
    instantaneous_stop_hazard: float = 0.0,
    remaining_integrated_total_hazard: float | None = None,
    diagnostic: str | None = None,
) -> dict[str, Any]:
    learned = reason == TerminationReason.LEARNED_STOP
    forced = reason in {
        TerminationReason.FORCED_BUDGET,
        TerminationReason.FORCED_NO_LEGAL_EDIT_ACTION,
        TerminationReason.FORCED_ZERO_REMAINING_INTEGRATED_HAZARD,
        TerminationReason.FORCED_TIME_HORIZON,
    }
    record: dict[str, Any] = {
        "schema_version": "termination_event_v1",
        "reason": reason.value,
        "external_time": external_time,
        "state_hash_before": state_hash_before,
        "state_hash_after": state_hash_after,
        "learned_stop": learned,
        "forced_termination": forced,
        "inside_ctmc": learned,
        "consumes_edit_budget": False,
        "instantaneous_edit_hazard": instantaneous_edit_hazard,
        "instantaneous_stop_hazard": instantaneous_stop_hazard,
        "remaining_integrated_total_hazard": remaining_integrated_total_hazard,
        "diagnostic": diagnostic,
    }
    if reason == TerminationReason.FORCED_ZERO_REMAINING_INTEGRATED_HAZARD:
        record["remaining_integrated_total_hazard"] = 0.0
    if reason == TerminationReason.FAILED_NUMERICAL and not diagnostic:
        raise ValueError("FAILED_NUMERICAL schema record requires diagnostic evidence")
    return record


def state_to_schema_record(
    state: EditState,
    *,
    source_id: str,
    external_time: float,
    parent_state_hash: str | None = None,
    termination: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize the internal dataclass to the frozen public JSON schema."""

    context = dict(state.context)
    expected_context = {"assay", "cell_or_tissue", "endpoint", "batch"}
    if set(context) != expected_context:
        # Never make an illicit runtime feature disappear from the audit
        # artifact by filtering it at serialization time.
        raise ValueError("internal context differs from frozen inference schema")
    source_to_current: list[int | None] = [None] * len(state.source)
    origins: list[dict[str, Any]] = []
    protected_indices: list[int] = []
    for current_index, ref in enumerate(state.mapping.tokens):
        if ref.protected:
            protected_indices.append(current_index)
        if ref.origin == TokenOrigin.SOURCE:
            source_index = int(ref.source_index)
            source_to_current[source_index] = current_index
            origins.append(
                {
                    "origin": "SOURCE",
                    "source_token_id": source_index,
                    "protected": ref.protected,
                }
            )
        else:
            origins.append(
                {
                    "origin": "INSERTED",
                    "inserted_event_id": ref.stable_id,
                    "protected": False,
                }
            )
    history_payload = json.dumps(
        list(state.history.action_keys), separators=(",", ":")
    ).encode("utf-8")
    target_direction = (
        state.target_condition
        if state.target_condition in {"increase", "decrease", "maintain", "interval"}
        else "interval"
    )
    if state.phase == Phase.HALTED and termination is None:
        termination = termination_to_schema_record(
            reason=state.termination_reason,  # type: ignore[arg-type]
            external_time=external_time,
            state_hash_before=parent_state_hash or state.state_hash,
            state_hash_after=state.state_hash,
        )
    return {
        "schema_version": "edit_state_v1",
        "source_id": source_id,
        "x_src": state.source,
        "x_current": state.current,
        "external_time": external_time,
        "time_direction": "source_at_0_to_target_at_1",
        "region": state.region,
        "context": {
            "assay": context["assay"],
            "cell_or_tissue": context["cell_or_tissue"],
            "endpoint": context["endpoint"],
            "batch": context["batch"],
        },
        "target_condition": {
            "direction": target_direction,
            "kind": "direction" if target_direction != "interval" else "interval",
            "value": state.target_condition,
        },
        "initial_budget": state.initial_budget,
        "remaining_budget": state.remaining_budget,
        "executed_action_count": state.history.executed,
        "mapping_run": {
            "token_origins": origins,
            "source_to_current": source_to_current,
            "gap_ids": list(state.mapping.gap_ids),
            "protected_current_indices": protected_indices,
        },
        "history_run": {
            "ins_count": state.history.ins,
            "sub_count": state.history.sub,
            "del_count": state.history.delete,
            "ordered_action_ids": list(state.history.action_keys),
            "history_hash": hashlib.sha256(history_payload).hexdigest(),
        },
        "run_state": state.phase.value,
        "termination": termination,
        "parent_state_hash": parent_state_hash,
        "state_hash": state.state_hash,
    }


def action_to_schema_record(
    transition: AppliedTransition,
    *,
    action_id: str,
    external_time: float,
) -> dict[str, Any]:
    action = transition.action
    record: dict[str, Any] = {
        "schema_version": "edit_action_v1",
        "action_id": action_id,
        "action_type": action.kind.value,
        "external_time": external_time,
        "coordinate_system": "pre_action_current_state_zero_based",
        "budget_cost": 0 if action.kind == ActionType.STOP else 1,
        "pre_state_hash": transition.before.state_hash,
        "post_state_hash": transition.after.state_hash,
        "legal_under_pre_state_mask": True,
    }
    if action.kind == ActionType.INS:
        record["gap_index"] = action.position
        record["nucleotide"] = action.token
        record["inserted_event_id"] = f"ins:{transition.after.history.executed}"
    elif action.kind in (ActionType.SUB, ActionType.DEL):
        record["token_index"] = action.position
        if action.kind == ActionType.SUB:
            record["nucleotide"] = action.token
        ref = transition.before.mapping.tokens[int(action.position)]
        record["source_token_id"] = ref.source_index
        record["inserted_event_id"] = (
            ref.stable_id if ref.origin == TokenOrigin.INSERTED else None
        )
    return record


_SCHEMA_NAMES = {
    "state": "edit_state_v1.schema.json",
    "action": "edit_action_v1.schema.json",
    "trajectory": "edit_trajectory_v1.schema.json",
    "termination": "termination_event_v1.schema.json",
    "coupling": "coupling_manifest_v1.schema.json",
}


@lru_cache(maxsize=1)
def _frozen_schema_validators() -> dict[str, Any]:
    """Load the frozen Draft 2020-12 schemas into an offline registry.

    Formal evidence must be checked by the actual schema engine.  The manual
    checks below are retained only for invariants JSON Schema cannot express.
    Missing ``jsonschema``/``referencing`` is fatal rather than a reason to
    silently downgrade validation.
    """

    try:
        from jsonschema import Draft202012Validator
        from referencing import Registry, Resource
    except ImportError as error:  # pragma: no cover - deployment guard
        raise RuntimeError(
            "MK0 schema validation requires jsonschema Draft 2020-12 support"
        ) from error
    schema_dir = Path(__file__).resolve().parents[2] / "schemas"
    loaded: dict[str, dict[str, Any]] = {}
    registry = Registry()
    for schema_path in sorted(schema_dir.glob("*_v1.schema.json")):
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        loaded[schema_path.name] = schema
        resource = Resource.from_contents(schema)
        registry = registry.with_resource(schema["$id"], resource)
        registry = registry.with_resource(schema_path.resolve().as_uri(), resource)
    missing = set(_SCHEMA_NAMES.values()) - set(loaded)
    if missing:
        raise RuntimeError(f"missing frozen MK0 schemas: {sorted(missing)}")
    return {
        kind: Draft202012Validator(loaded[name], registry=registry)
        for kind, name in _SCHEMA_NAMES.items()
    }


def _validate_frozen_schema(record: dict[str, Any], kind: str) -> None:
    try:
        validator = _frozen_schema_validators()[kind]
    except KeyError as error:
        raise ValueError(f"unsupported schema-facing record kind: {kind}") from error
    errors = sorted(
        validator.iter_errors(record),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if errors:
        first = errors[0]
        location = "$" + "".join(f"[{part!r}]" for part in first.absolute_path)
        raise ValueError(
            f"{kind} Draft 2020-12 validation failed at {location}: {first.message}"
        )


def validate_schema_facing_record(record: dict[str, Any], kind: str) -> None:
    """Validate a real record against schema plus cross-field invariants."""

    _validate_frozen_schema(record, kind)

    if kind == "state":
        allowed = {
            "schema_version",
            "source_id",
            "x_src",
            "x_current",
            "external_time",
            "time_direction",
            "region",
            "context",
            "target_condition",
            "initial_budget",
            "remaining_budget",
            "executed_action_count",
            "mapping_run",
            "history_run",
            "run_state",
            "termination",
            "parent_state_hash",
            "state_hash",
        }
        required = {
            "schema_version",
            "source_id",
            "x_src",
            "x_current",
            "external_time",
            "region",
            "context",
            "target_condition",
            "initial_budget",
            "remaining_budget",
            "executed_action_count",
            "mapping_run",
            "history_run",
            "run_state",
            "state_hash",
        }
        if (
            not required <= record.keys()
            or not set(record) <= allowed
            or record["schema_version"] != "edit_state_v1"
        ):
            raise ValueError("invalid edit_state_v1 runtime record")
        if not isinstance(record["source_id"], str) or not record["source_id"]:
            raise ValueError("state source_id is required")
        if (
            not isinstance(record["external_time"], (int, float))
            or not math.isfinite(record["external_time"])
            or not 0.0 <= record["external_time"] <= 1.0
        ):
            raise ValueError("state external_time is outside [0,1]")
        if (
            record.get("time_direction", "source_at_0_to_target_at_1")
            != "source_at_0_to_target_at_1"
        ):
            raise ValueError("state time direction is reversed")
        if record["region"] not in {"5UTR", "3UTR"}:
            raise ValueError("invalid UTR region")
        if set(record["context"]) != {"assay", "cell_or_tissue", "endpoint", "batch"}:
            raise ValueError("state context properties differ from schema")
        if set(record["target_condition"]) != {"direction", "kind", "value"}:
            raise ValueError("state target_condition properties differ from schema")
        if (
            not isinstance(record["x_src"], str)
            or not isinstance(record["x_current"], str)
            or not record["x_src"]
            or not record["x_current"]
            or any(
                token not in ALPHABET for token in record["x_src"] + record["x_current"]
            )
        ):
            raise ValueError("state record contains non-RNA token")
        if any(
            not isinstance(record[key], int) or record[key] < 0
            for key in ("initial_budget", "remaining_budget", "executed_action_count")
        ):
            raise ValueError("state budget fields must be non-negative integers")
        if any(
            not isinstance(record["context"][key], str) or not record["context"][key]
            for key in ("assay", "cell_or_tissue", "endpoint")
        ):
            raise ValueError("state context required values must be non-empty strings")
        condition = record["target_condition"]
        if condition["direction"] not in {
            "increase",
            "decrease",
            "maintain",
            "interval",
        }:
            raise ValueError("invalid target direction")
        if condition["kind"] not in {"direction", "interval", "quantile"}:
            raise ValueError("invalid target condition kind")
        if not isinstance(condition["value"], (int, float, str, list)):
            raise ValueError("invalid target condition value")
        if len(record["mapping_run"]["token_origins"]) != len(record["x_current"]):
            raise ValueError("state record mapping length mismatch")
        source_to_current = record["mapping_run"]["source_to_current"]
        if len(source_to_current) != len(record["x_src"]):
            raise ValueError("source_to_current length must equal source length")
        observed_source_positions: dict[int, int] = {}
        for current_index, origin in enumerate(record["mapping_run"]["token_origins"]):
            if origin["origin"] == "SOURCE":
                source_index = origin["source_token_id"]
                if (
                    source_index >= len(record["x_src"])
                    or source_index in observed_source_positions
                ):
                    raise ValueError(
                        "source token identity is duplicated or out of range"
                    )
                observed_source_positions[source_index] = current_index
            elif origin.get("protected") is not False:
                raise ValueError("inserted token cannot be protected")
        expected_source_to_current = [
            observed_source_positions.get(index)
            for index in range(len(record["x_src"]))
        ]
        if source_to_current != expected_source_to_current:
            raise ValueError("source_to_current disagrees with token origins")
        if len(record["mapping_run"]["gap_ids"]) != len(record["x_current"]) + 1:
            raise ValueError("state record gap length mismatch")
        if len(set(record["mapping_run"]["gap_ids"])) != len(
            record["mapping_run"]["gap_ids"]
        ):
            raise ValueError("state gap IDs must be unique")
        if (
            record["remaining_budget"]
            != record["initial_budget"] - record["executed_action_count"]
        ):
            raise ValueError("state record budget invariant failed")
        history = record["history_run"]
        if (
            history["ins_count"] + history["sub_count"] + history["del_count"]
            != record["executed_action_count"]
        ):
            raise ValueError("history counters disagree with executed_action_count")
        if len(history["ordered_action_ids"]) != record["executed_action_count"]:
            raise ValueError("history action IDs disagree with executed_action_count")
        expected_history_hash = hashlib.sha256(
            json.dumps(history["ordered_action_ids"], separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        if history["history_hash"] != expected_history_hash:
            raise ValueError("history_hash does not bind ordered_action_ids")
        protected_from_origins = [
            index
            for index, origin in enumerate(record["mapping_run"]["token_origins"])
            if origin["protected"]
        ]
        if record["mapping_run"]["protected_current_indices"] != protected_from_origins:
            raise ValueError("protected index ledger disagrees with token origins")
        if not re.fullmatch(r"[0-9a-f]{64}", record["state_hash"]):
            raise ValueError("state_hash is not SHA-256")
        if record.get("parent_state_hash") is not None and not re.fullmatch(
            r"[0-9a-f]{64}", record["parent_state_hash"]
        ):
            raise ValueError("parent_state_hash is not SHA-256")
        if record["run_state"] == "ACTIVE" and record.get("termination") is not None:
            raise ValueError("ACTIVE state cannot carry termination")
        if record["run_state"] == "HALTED":
            if not isinstance(record.get("termination"), dict):
                raise ValueError("HALTED state requires termination record")
            if record["termination"].get("reason") == "FAILED_NUMERICAL":
                raise ValueError("FAILED_NUMERICAL is outside the CTMC state")
        elif record["run_state"] != "ACTIVE":
            raise ValueError("invalid run_state")
    elif kind == "action":
        allowed = {
            "schema_version",
            "action_id",
            "action_type",
            "external_time",
            "coordinate_system",
            "gap_index",
            "token_index",
            "nucleotide",
            "budget_cost",
            "pre_state_hash",
            "post_state_hash",
            "legal_under_pre_state_mask",
            "source_token_id",
            "inserted_event_id",
        }
        required = {
            "schema_version",
            "action_id",
            "action_type",
            "external_time",
            "coordinate_system",
            "budget_cost",
            "pre_state_hash",
            "post_state_hash",
            "legal_under_pre_state_mask",
        }
        if (
            not required <= record.keys()
            or not set(record) <= allowed
            or record["schema_version"] != "edit_action_v1"
        ):
            raise ValueError("invalid edit_action_v1 runtime record")
        if record["action_type"] not in {"INS", "SUB", "DEL", "STOP"}:
            raise ValueError("unknown atomic action type")
        if (
            not isinstance(record["external_time"], (int, float))
            or not math.isfinite(record["external_time"])
            or not 0.0 <= record["external_time"] <= 1.0
        ):
            raise ValueError("action external_time is outside [0,1]")
        if record["coordinate_system"] != "pre_action_current_state_zero_based":
            raise ValueError("action coordinate system drift")
        if record["legal_under_pre_state_mask"] is not True:
            raise ValueError("illegal action cannot be serialized as accepted")
        if any(
            not re.fullmatch(r"[0-9a-f]{64}", record[key])
            for key in ("pre_state_hash", "post_state_hash")
        ):
            raise ValueError("action state hash is not SHA-256")
        expected_cost = 0 if record["action_type"] == "STOP" else 1
        if record["budget_cost"] != expected_cost:
            raise ValueError("action budget cost differs from frozen kernel")
        if (
            record["action_type"] == "INS"
            and not {"gap_index", "nucleotide"} <= record.keys()
        ):
            raise ValueError("INS schema fields missing")
        if record["action_type"] in {"SUB", "DEL"} and "token_index" not in record:
            raise ValueError("token action index missing")
        if record["action_type"] == "SUB" and "nucleotide" not in record:
            raise ValueError("SUB nucleotide missing")
        if record["action_type"] == "STOP" and any(
            key in record
            for key in (
                "gap_index",
                "token_index",
                "nucleotide",
                "source_token_id",
                "inserted_event_id",
            )
        ):
            raise ValueError("STOP record carries edit coordinate")
        if "gap_index" in record and (
            not isinstance(record["gap_index"], int) or record["gap_index"] < 0
        ):
            raise ValueError("gap_index must be a non-negative integer")
        if "token_index" in record and (
            not isinstance(record["token_index"], int) or record["token_index"] < 0
        ):
            raise ValueError("token_index must be a non-negative integer")
        if "nucleotide" in record and record["nucleotide"] not in ALPHABET:
            raise ValueError("action nucleotide is outside A,C,G,U")
    elif kind == "termination":
        allowed = {
            "schema_version",
            "reason",
            "external_time",
            "state_hash_before",
            "state_hash_after",
            "learned_stop",
            "forced_termination",
            "inside_ctmc",
            "consumes_edit_budget",
            "instantaneous_edit_hazard",
            "instantaneous_stop_hazard",
            "remaining_integrated_total_hazard",
            "diagnostic",
        }
        required = {
            "schema_version",
            "reason",
            "external_time",
            "state_hash_before",
            "state_hash_after",
            "learned_stop",
            "forced_termination",
            "inside_ctmc",
            "consumes_edit_budget",
        }
        if not required <= set(record) or not set(record) <= allowed:
            raise ValueError("invalid termination_event_v1 properties")
        if record["schema_version"] != "termination_event_v1":
            raise ValueError("termination schema version drift")
        if (
            not isinstance(record["external_time"], (int, float))
            or not math.isfinite(record["external_time"])
            or not 0.0 <= record["external_time"] <= 1.0
        ):
            raise ValueError("termination external_time is outside [0,1]")
        if any(
            not isinstance(record[key], str)
            or not re.fullmatch(r"[0-9a-f]{64}", record[key])
            for key in ("state_hash_before", "state_hash_after")
        ):
            raise ValueError("termination state hash is not SHA-256")
        for key in ("instantaneous_edit_hazard", "instantaneous_stop_hazard"):
            if key in record and (
                not isinstance(record[key], (int, float))
                or not math.isfinite(record[key])
                or record[key] < 0.0
            ):
                raise ValueError("termination hazard must be finite and non-negative")
        remaining = record.get("remaining_integrated_total_hazard")
        if remaining is not None and (
            not isinstance(remaining, (int, float))
            or not math.isfinite(remaining)
            or remaining < 0.0
        ):
            raise ValueError(
                "remaining integrated hazard must be finite and non-negative"
            )
        try:
            reason = TerminationReason(record["reason"])
        except ValueError as error:
            raise ValueError("unknown termination reason") from error
        learned = reason == TerminationReason.LEARNED_STOP
        forced = reason.value.startswith("FORCED_")
        if (
            record["learned_stop"] is not learned
            or record["forced_termination"] is not forced
            or record["inside_ctmc"] is not learned
            or record["consumes_edit_budget"] is not False
        ):
            raise ValueError("learned/forced/numerical termination semantics conflated")
        if reason == TerminationReason.FAILED_NUMERICAL and not record.get(
            "diagnostic"
        ):
            raise ValueError("FAILED_NUMERICAL requires diagnostic")
        if (
            reason == TerminationReason.FORCED_ZERO_REMAINING_INTEGRATED_HAZARD
            and record.get("remaining_integrated_total_hazard") != 0.0
        ):
            raise ValueError("zero remaining integrated hazard is not verified")
    elif kind == "coupling":
        required = {
            "schema_version",
            "source_id",
            "target_id",
            "coupling_type",
            "alignment_algorithm_version",
            "alignment_cost",
            "tie_break_rule",
            "alignment_hash",
            "path_is_observed",
            "path_semantics",
            "z_src",
            "z_tar",
            "source_reconstruction",
            "target_reconstruction",
            "joint_path",
            "schedule",
            "target_auxiliary_isolation",
            "rejected_path_ledger",
        }
        allowed = required | {"switch_clocks", "manifest_hash"}
        if not required <= set(record) or not set(record) <= allowed:
            raise ValueError("invalid coupling_manifest_v1 properties")
        if record["schema_version"] != "coupling_manifest_v1":
            raise ValueError("coupling schema version drift")
        if (
            record["path_is_observed"] is not False
            or record["path_semantics"] != "latent_algorithmic"
        ):
            raise ValueError("constructed alignment falsely marked observed")
        if record["joint_path"] != "independent_switch_clock_product":
            raise ValueError("joint product path drift")
        if record["coupling_type"] not in {
            "canonical_optimal",
            "sampled_optimal_sensitivity",
            "equivalent_edit_order_sensitivity",
        }:
            raise ValueError("unknown coupling type")
        if (
            not isinstance(record["alignment_cost"], int)
            or record["alignment_cost"] < 0
        ):
            raise ValueError("alignment cost must be a non-negative integer")
        if (
            not isinstance(record["z_src"], list)
            or not isinstance(record["z_tar"], list)
            or len(record["z_src"]) != len(record["z_tar"])
            or not record["z_src"]
            or any(
                token not in {*ALPHABET, "EPSILON"}
                for token in record["z_src"] + record["z_tar"]
            )
        ):
            raise ValueError("invalid augmented alignment token arrays")
        if (
            "".join(token for token in record["z_src"] if token != "EPSILON")
            != record["source_reconstruction"]
        ):
            raise ValueError("source alignment reconstruction failed")
        if (
            "".join(token for token in record["z_tar"] if token != "EPSILON")
            != record["target_reconstruction"]
        ):
            raise ValueError("target alignment reconstruction failed")
        schedule = record["schedule"]
        if set(schedule) != {"name", "kappa", "kappa_derivative", "rho", "time_eps"}:
            raise ValueError("coupling schedule properties differ from schema")
        if (
            schedule["name"] not in {"cubic", "linear"}
            or schedule["rho"] != "kappa_derivative/(1-kappa)"
        ):
            raise ValueError("coupling schedule is not frozen")
        if (
            not isinstance(schedule["time_eps"], (int, float))
            or not 0.0 < schedule["time_eps"] <= 0.1
        ):
            raise ValueError("coupling time_eps outside schema range")
        isolation = record["target_auxiliary_isolation"]
        if (
            isolation.get("z_aux_training_only") is not True
            or isolation.get("rate_network_receives_z_aux") is not False
        ):
            raise ValueError("target auxiliary leakage")
        if not re.fullmatch(r"[0-9a-f]{64}", record["alignment_hash"]):
            raise ValueError("alignment hash is not SHA-256")
    elif kind == "trajectory":
        required = {
            "schema_version",
            "trajectory_id",
            "source_id",
            "seed",
            "sampler",
            "sampler_semantics",
            "exact_gillespie",
            "time_direction",
            "sampler_config",
            "remaining_hazard_certificate",
            "initial_state",
            "steps",
            "final_state",
            "run_status",
            "termination",
            "replay",
        }
        if set(record) != required or record["schema_version"] != "edit_trajectory_v1":
            raise ValueError("invalid edit_trajectory_v1 properties")
        if (
            record["sampler"]
            not in {
                "paper_first_order_parallel",
                "constrained_single_event_first_order",
            }
            or record["exact_gillespie"] is not False
        ):
            raise ValueError("trajectory sampler semantics are unsupported")
        expected_semantics = {
            "paper_first_order_parallel": "fixed_grid_parallel_first_order_approximation",
            "constrained_single_event_first_order": "endpoint_single_event_frozen_rate_approximation",
        }[record["sampler"]]
        if record["sampler_semantics"] != expected_semantics:
            raise ValueError("trajectory sampler semantics mismatch")
        if record["time_direction"] != "source_at_0_to_target_at_1":
            raise ValueError("trajectory time direction is reversed")
        if record["run_status"] not in {"COMPLETED", "FAILED_NUMERICAL"}:
            raise ValueError("unknown trajectory run status")
        sampler_config = record["sampler_config"]
        if sampler_config["max_length"] < sampler_config["min_length"]:
            raise ValueError("trajectory sampler length bounds are reversed")
        if record["sampler"] == "paper_first_order_parallel":
            if sampler_config["stability_hazard"] is not None:
                raise ValueError(
                    "paper sampler cannot carry an adaptive stability hazard"
                )
        elif sampler_config["stability_hazard"] is None:
            raise ValueError("constrained sampler requires a stability hazard")
        validate_schema_facing_record(record["initial_state"], "state")
        validate_schema_facing_record(record["final_state"], "state")
        validate_schema_facing_record(record["termination"], "termination")
        if (
            record["initial_state"]["source_id"] != record["source_id"]
            or record["final_state"]["source_id"] != record["source_id"]
        ):
            raise ValueError("trajectory source_id differs from nested states")
        if (
            record["steps"]
            and record["initial_state"]["state_hash"]
            != record["steps"][0]["state_hash_before"]
        ):
            raise ValueError("first trajectory step is not bound to initial state")
        if (
            record["termination"]["state_hash_after"]
            != record["final_state"]["state_hash"]
        ):
            raise ValueError("termination does not bind the final state")
        if record["final_state"].get("termination") != record["termination"]:
            raise ValueError("nested and top-level termination records differ")
        if (
            record["run_status"] == "COMPLETED"
            and record["final_state"]["run_state"] != "HALTED"
        ):
            raise ValueError("completed trajectory final state is not HALTED")

        certificate = record["remaining_hazard_certificate"]
        zero_reason = (
            record["termination"]["reason"] == "FORCED_ZERO_REMAINING_INTEGRATED_HAZARD"
        )
        if zero_reason:
            if certificate is None:
                raise ValueError(
                    "zero-integral termination lacks numerical certificate"
                )
            if (
                certificate["integral"] > certificate["zero_atol"]
                or certificate["lower_order_integral"] > certificate["zero_atol"]
                or certificate["disagreement"] > certificate["convergence_atol"]
                or certificate["higher_order"] <= certificate["lower_order"]
            ):
                raise ValueError("remaining-hazard certificate does not verify zero")
        elif certificate is not None:
            raise ValueError(
                "non-zero-integral termination carries a stale certificate"
            )

        replay = record["replay"]
        if set(replay) != {
            "status",
            "replayed_step_count",
            "state_hash_match_fraction",
        }:
            raise ValueError("trajectory replay properties differ from schema")
        if (
            replay["status"] not in {"PASS", "FAIL"}
            or not isinstance(replay["replayed_step_count"], int)
            or replay["replayed_step_count"] < 0
        ):
            raise ValueError("invalid trajectory replay status/count")
        if (
            not isinstance(replay["state_hash_match_fraction"], (int, float))
            or not math.isfinite(replay["state_hash_match_fraction"])
            or not 0.0 <= replay["state_hash_match_fraction"] <= 1.0
        ):
            raise ValueError("invalid trajectory replay match fraction")
        if replay["status"] == "PASS" and (
            replay["replayed_step_count"] != len(record["steps"])
            or replay["state_hash_match_fraction"] != 1.0
        ):
            raise ValueError("PASS replay does not cover every step exactly")

        previous_time = 0.0
        previous_hash = record["initial_state"]["state_hash"]
        constrained_outcomes = {"NO_EVENT", "INS", "SUB", "DEL", "STOP"}
        paper_outcomes = {
            "NO_EVENT",
            "PARALLEL_EVENTS_APPLIED",
            "INVALID_JOINT_PROPOSAL_REPORTED",
        }
        for index, step in enumerate(record["steps"]):
            if step["step_index"] != index:
                raise ValueError("trajectory step indices are not contiguous")
            if not math.isclose(
                step["t_start"], previous_time, abs_tol=1.0e-12, rel_tol=1.0e-12
            ):
                raise ValueError("trajectory step times are not contiguous")
            if step["t_end"] <= step["t_start"] or not math.isclose(
                step["t_end"] - step["t_start"],
                step["substep_h"],
                abs_tol=1.0e-12,
                rel_tol=1.0e-12,
            ):
                raise ValueError("trajectory substep duration is inconsistent")
            if step["t_end"] > sampler_config["horizon"] + 1.0e-12:
                raise ValueError("trajectory step exceeds sampler horizon")
            if step["state_hash_before"] != previous_hash:
                raise ValueError("trajectory state-hash chain is broken")
            if step["rate_recomputed_after_step"] is not True:
                raise ValueError("trajectory omitted rate recomputation")

            if record["sampler"] == "constrained_single_event_first_order":
                if step["outcome"] not in constrained_outcomes:
                    raise ValueError("constrained trajectory carries paper outcome")
                if step["parallel_trials"] or step["parallel_actions"]:
                    raise ValueError("constrained step carries parallel-event fields")
                if step["outcome"] == "NO_EVENT":
                    if (
                        step["selected_action"] is not None
                        or step["action_uniform"] is not None
                    ):
                        raise ValueError("NO_EVENT step carries a selected action")
                    if step["state_hash_before"] != step["state_hash_after"]:
                        raise ValueError("NO_EVENT step changed the extended state")
                else:
                    selected = step["selected_action"]
                    if selected is None or selected["action_type"] != step["outcome"]:
                        raise ValueError("selected action disagrees with step outcome")
                    if (
                        selected["pre_state_hash"] != step["state_hash_before"]
                        or selected["post_state_hash"] != step["state_hash_after"]
                    ):
                        raise ValueError(
                            "selected action does not bind step state hashes"
                        )
            else:
                if step["outcome"] not in paper_outcomes:
                    raise ValueError("paper trajectory carries constrained outcome")
                if (
                    step["selected_action"] is not None
                    or step["event_uniform"] is not None
                    or step["action_uniform"] is not None
                ):
                    raise ValueError("paper step carries single-event draw fields")
                if len(step["parallel_trials"]) == 0 and step["total_hazard"] > 0.0:
                    raise ValueError("paper step omitted per-action Bernoulli trials")
                proposed_keys = {
                    trial["action_key"]
                    for trial in step["parallel_trials"]
                    if trial["proposed"]
                }
                if len(step["parallel_actions"]) != len(proposed_keys):
                    raise ValueError(
                        "paper proposed-action ledger cardinality mismatch"
                    )
                if step["outcome"] == "NO_EVENT" and step["parallel_actions"]:
                    raise ValueError("paper NO_EVENT step carries proposed actions")
                if (
                    step["outcome"] == "INVALID_JOINT_PROPOSAL_REPORTED"
                    and step["state_hash_before"] != step["state_hash_after"]
                ):
                    raise ValueError("invalid joint proposal changed the state")

            previous_time = step["t_end"]
            previous_hash = step["state_hash_after"]

        reason = record["termination"]["reason"]
        if reason == "LEARNED_STOP":
            if not record["steps"] or record["steps"][-1]["outcome"] not in {
                "STOP",
                "PARALLEL_EVENTS_APPLIED",
            }:
                raise ValueError("learned STOP lacks a terminal sampled STOP step")
            if previous_hash != record["final_state"]["state_hash"]:
                raise ValueError("learned STOP final state differs from last step")
            if (
                record["termination"]["state_hash_before"]
                != record["steps"][-1]["state_hash_before"]
            ):
                raise ValueError("learned STOP termination pre-state is inconsistent")
        elif reason.startswith("FORCED_"):
            if record["termination"]["state_hash_before"] != previous_hash:
                raise ValueError("forced termination pre-state differs from step chain")
    else:
        raise ValueError(f"unknown schema-facing record kind: {kind}")
