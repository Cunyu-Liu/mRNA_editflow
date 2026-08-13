#!/usr/bin/env python3
"""Run the Route-A-v3 A6 CPU-only exact absorbing-DAG fixture.

This module contains no learned parameters, training path, GPU path, ordinary
row data, private data, or sealed-data access.  Production first proves the
clean pushed repository authority and the exact active goal/config bytes.  It
then computes and validates every synthetic result in memory.  Only a complete
PASS is published, by one atomic directory rename, as exactly three public
aggregate files.

The exact object is a time-homogeneous CTMC on a finite source-anchored edit
DAG.  Algorithmic time remains an explicit continuous state field.  Rates and
terminal tilt are frozen time-invariant, so the harmonic extension is also
time-invariant; no general time-inhomogeneous exactness is asserted.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


GOAL_SHA256 = "cbac4c3dcba8f1b8df95d8edad52d19e3c126d1c865d0cc423537c754cc90982"
ACTIVE_CONFIG_SHA256 = "c908ac57b7c9667398f616a0ccf7101b41451b80bf169e768131844d3b63a678"
FROZEN_BASE_COMMIT = "db297787b3cd9f74908a1ae726cb64b19a9161fb"
UNKNOWN = "UNKNOWN_NOT_ASSERTED"
CONFIG_REPO_PATH = "configs/route_a_v3_a6_cpu_exact_absorbing_dag_v1.json"
SCRIPT_REPO_PATH = "scripts/route_a_v3/run_a6_cpu_exact_absorbing_dag.py"
TEST_REPO_PATH = "tests/route_a_v3/test_a6_cpu_exact_absorbing_dag.py"
EXACT_IMPLEMENTATION_PATHS = (CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH)
BINDING_KEYS = (
    "status",
    "implementation_commit",
    "implementation_script_sha256",
    "implementation_test_sha256",
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PRODUCTION_CONFIG_PATH = Path(__file__).resolve().parents[2] / CONFIG_REPO_PATH
OUTPUT_NAMES = (
    "A6_CPU_EXACT_ABSORBING_DAG_REPORT.json",
    "RUN_MANIFEST.json",
    "EVENT_LOG.jsonl",
)
TERMINAL_CAUSES = (
    "EXPLICIT_STOP",
    "BUDGET_EXHAUSTED",
    "NO_LEGAL_ACTION",
    "NUMERICAL_FAILURE",
)
EDIT = "SOURCE_BASE_TO_ALT_BASE"
STOP = "STOP"


class A6ExactError(RuntimeError):
    """Base error for static, numerical, authority, or publication failure."""


class ConfigError(A6ExactError):
    """The frozen fixture configuration is invalid."""


class StateError(A6ExactError):
    """A state violates the source-anchored DAG semantics."""


class NumericalFailure(A6ExactError):
    """A rate, probability, potential, or comparison is invalid."""

    terminal_cause = "NUMERICAL_FAILURE"


class AuthorityError(A6ExactError):
    """Production is not the exact clean pushed active authority."""


class PublicationError(A6ExactError):
    """The exclusive atomic three-file publication cannot be completed."""


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def json_line(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ConfigError(f"non-finite JSON constant: {value}")


def load_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"invalid JSON: {label}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"JSON root must be an object: {label}")
    return value


def load_config(path: Path = PRODUCTION_CONFIG_PATH) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read config: {path}") from exc
    config = load_json_bytes(payload, label=str(path))
    validate_static_config(config)
    return config


def _expect(observed: Any, expected: Any, *, label: str) -> None:
    if observed != expected:
        raise ConfigError(f"{label} differs: expected {expected!r}, observed {observed!r}")


def _positive_finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ConfigError(f"{label} must be finite and strictly positive")
    return result


def _binding_mode(config: Mapping[str, Any]) -> str:
    binding = config.get("implementation_binding")
    if not isinstance(binding, Mapping) or set(binding) != set(BINDING_KEYS):
        raise ConfigError("implementation_binding must contain exactly four frozen scalars")
    values = [binding[key] for key in BINDING_KEYS]
    if all(value == UNKNOWN for value in values):
        return "UNKNOWN"
    if any(value == UNKNOWN for value in values):
        raise ConfigError("implementation_binding is partially known")
    if binding["status"] != "BOUND":
        raise ConfigError("implementation_binding.status must be BOUND or UNKNOWN_NOT_ASSERTED")
    if not isinstance(binding["implementation_commit"], str) or not HEX40.fullmatch(
        binding["implementation_commit"]
    ):
        raise ConfigError("implementation commit is not a lowercase 40-hex SHA")
    for key in ("implementation_script_sha256", "implementation_test_sha256"):
        if not isinstance(binding[key], str) or not HEX64.fullmatch(binding[key]):
            raise ConfigError(f"{key} is not a lowercase 64-hex SHA-256")
    return "BOUND"


def candidate_i_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize exactly the four I-to-B scalars back to candidate-I UNKNOWN."""

    normalized = copy.deepcopy(dict(config))
    binding = normalized.get("implementation_binding")
    if not isinstance(binding, dict) or set(binding) != set(BINDING_KEYS):
        raise ConfigError("implementation_binding must contain exactly four frozen scalars")
    for key in BINDING_KEYS:
        binding[key] = UNKNOWN
    return normalized


def validate_static_config(config: Mapping[str, Any]) -> None:
    """Validate the small frozen protocol without reading Git or external data."""

    _expect(
        config.get("schema_version"),
        "route_a_v3_a6_cpu_exact_absorbing_dag.v1",
        label="schema version",
    )
    _expect(config.get("contract_id"), "mrna_xeditflow_route_a_v3", label="contract")
    _expect(config.get("phase_id"), "A6", label="phase")
    _expect(
        config.get("run_scope"),
        "DEVELOPMENT_ONLY_CPU_EXACT_ABSORPTION_FIXTURE",
        label="run scope",
    )
    _expect(
        config.get("production_python"),
        "/home/cunyuliu/miniconda3/envs/editflow/bin/python",
        label="production Python",
    )
    _binding_mode(config)
    authority = config["authority"]
    _expect(authority["goal_sha256"], GOAL_SHA256, label="goal SHA")
    _expect(authority["active_config_sha256"], ACTIVE_CONFIG_SHA256, label="active config SHA")
    _expect(authority["active_contract_count_required"], 1, label="active contract count")
    _expect(authority["authority_uniqueness_required"], True, label="authority uniqueness")
    _expect(authority["required_remote"], "origin", label="required remote")
    _expect(authority["frozen_base_commit"], FROZEN_BASE_COMMIT, label="frozen base commit")
    if not authority["goal_critical_literals"] or not authority["active_config_critical_literals"]:
        raise ConfigError("critical literal sets must be non-empty")

    _expect(config["clock_semantics"], "CONTINUOUS_ALGORITHMIC_TIME", label="clock")
    _expect(config["rate_time_dependence"], "NONE", label="rate time dependence")
    _expect(
        config["terminal_tilt_time_dependence"],
        "NONE",
        label="terminal tilt time dependence",
    )
    _expect(
        config["general_time_inhomogeneous_exactness"],
        "NOT_RUN",
        label="general time-inhomogeneous exactness",
    )

    graph = config["graph_contract"]
    _expect(graph["state_type"], "SOURCE_ANCHORED_ACYCLIC_SPARSE_EDIT_DAG", label="state type")
    _expect(graph["action_types"], [EDIT, STOP], label="action types")
    _expect(graph["repeated_position_edit_allowed"], False, label="position re-edit")
    _expect(graph["revert_to_source_allowed"], False, label="revert")
    _expect(graph["hard_legality_before_rates"], True, label="legality ordering")
    _expect(graph["raw_alias_aggregation_key"], "FULL_NEXT_EXTENDED_STATE", label="alias key")
    _expect(graph["terminal_causes"], list(TERMINAL_CAUSES), label="terminal causes")
    _expect(
        graph["terminal_precedence"],
        ["NUMERICAL_FAILURE", "BUDGET_EXHAUSTED", "NO_LEGAL_ACTION", "EXPLICIT_STOP"],
        label="terminal precedence",
    )
    alphabet = tuple(graph["alphabet"])
    if alphabet != ("A", "C", "G", "U"):
        raise ConfigError("alphabet must be the frozen RNA alphabet")

    rates = config["rate_parameters"]
    for key in ("support_floor", "edit_rate_scale", "stop_rate"):
        _positive_finite(rates[key], label=f"rate_parameters.{key}")
    for group in (
        "source_base_multipliers",
        "alt_base_multipliers",
        "assay_multipliers",
        "context_multipliers",
    ):
        if not rates[group]:
            raise ConfigError(f"{group} must not be empty")
        for key, value in rates[group].items():
            _positive_finite(value, label=f"{group}.{key}")
    for action_type in (EDIT, STOP):
        aliases = rates["raw_alias_shares"][action_type]
        ids = [item["alias_id"] for item in aliases]
        if len(ids) < 2 or len(ids) != len(set(ids)):
            raise ConfigError(f"{action_type} requires at least two unique raw aliases")
        shares = [_positive_finite(item["share"], label=f"alias {item['alias_id']}") for item in aliases]
        if not math.isclose(sum(shares), 1.0, rel_tol=0.0, abs_tol=1e-15):
            raise ConfigError(f"{action_type} alias shares must sum to one")

    tilt = config["terminal_tilt"]
    _expect(tilt["strictly_positive"], True, label="strict terminal tilt")
    _positive_finite(tilt["offset"], label="terminal tilt offset")
    if set(tilt["terminal_cause_bonus"]) != set(TERMINAL_CAUSES):
        raise ConfigError("terminal tilt cause keys differ")
    for group in ("alt_base_bonus", "terminal_cause_bonus"):
        for key, value in tilt[group].items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ConfigError(f"{group}.{key} must be finite and nonnegative")

    tolerances = config["numerical_tolerances"]
    for key, value in tolerances.items():
        _positive_finite(value, label=f"numerical_tolerances.{key}")
    _expect(tolerances["true_rate_relative_error_max"], 1e-5, label="rate tolerance")
    _expect(tolerances["terminal_distribution_tv_max"], 1e-12, label="terminal TV tolerance")
    _expect(tolerances["path_product_relative_error_max"], 1e-12, label="path tolerance")

    fixed = config["fixed_cases"]
    combinations = {(len(item["source_sequence"]), item["budget"]) for item in fixed}
    if combinations != {(length, budget) for length in (2, 3) for budget in (0, 1, 2)}:
        raise ConfigError("fixed cases must cover L=2/3 and budgets 0/1/2 exactly")
    if len(fixed) != 6 or len({item["case_id"] for item in fixed}) != 6:
        raise ConfigError("exactly six uniquely named fixed cases are required")
    for item in fixed:
        if any(base not in alphabet for base in item["source_sequence"]):
            raise ConfigError(f"invalid fixed source: {item['case_id']}")
        if item["assay"] not in rates["assay_multipliers"]:
            raise ConfigError(f"unknown assay: {item['case_id']}")
        if item["context"] not in rates["context_multipliers"]:
            raise ConfigError(f"unknown context: {item['case_id']}")
        if not isinstance(item["budget"], int) or isinstance(item["budget"], bool) or item["budget"] < 0:
            raise ConfigError(f"invalid budget: {item['case_id']}")
        if not isinstance(item["algorithmic_time"], (int, float)) or not math.isfinite(item["algorithmic_time"]):
            raise ConfigError(f"invalid algorithmic time: {item['case_id']}")

    status = config["status_contract"]
    expected_status = {
        "run_status_on_success": "PASS",
        "a6_evidence_status": "IN_PROGRESS",
        "exact_guidance_toy_graph_evidence_status": "PASS",
        "exact_guidance_toy_graph_result": "DEVELOPMENT_CPU_EXACT_FIXTURE_PASS",
        "flow_base_legal_ctmc_evidence_status": "NOT_RUN",
        "l3_evidence_status": "IN_PROGRESS",
        "l3_claim_status": "NOT_ESTABLISHED",
        "learned_potential_approximation_error": "NOT_RUN",
        "a7_unlock": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "gpu_work_allowed": False,
        "private_payload_access_allowed": False,
        "sealed_contact_allowed": False,
    }
    for key, expected in expected_status.items():
        _expect(status[key], expected, label=f"status_contract.{key}")
    publication = config["publication"]
    _expect(publication["output_names"], list(OUTPUT_NAMES), label="output names")
    for key in ("exclusive_new_directory_required", "atomic_directory_publish_required", "public_aggregate_only"):
        _expect(publication[key], True, label=f"publication.{key}")
    for key in ("row_data_allowed", "trajectory_data_allowed", "private_or_sealed_data_allowed"):
        _expect(publication[key], False, label=f"publication.{key}")


@dataclass(frozen=True)
class State:
    source_sequence: str
    current_sequence: str
    source_relative_edit_set: tuple[tuple[int, str], ...]
    remaining_budget: int
    assay: str
    context: str
    algorithmic_time: float
    terminal_cause: str | None = None

    @property
    def net_edit_count(self) -> int:
        return len(self.source_relative_edit_set)


@dataclass(frozen=True)
class RawAction:
    action_type: str
    alias_id: str
    position: int | None = None
    source_base: str | None = None
    alt_base: str | None = None


@dataclass(frozen=True)
class RawTransition:
    action: RawAction
    next_state: State
    rate: float


@dataclass(frozen=True)
class CanonicalTransition:
    action_type: str
    next_state: State
    rate: float
    raw_alias_ids: tuple[str, ...]
    raw_rates: tuple[float, ...]


@dataclass(frozen=True)
class Graph:
    root: State
    states: frozenset[State]
    outgoing: Mapping[State, tuple[CanonicalTransition, ...]]


@dataclass(frozen=True)
class GeneratorRow:
    off_diagonal: tuple[tuple[State, float], ...]
    diagonal: float
    total_exit_rate: float


def validate_state(state: State, config: Mapping[str, Any]) -> None:
    alphabet = set(config["graph_contract"]["alphabet"])
    if not state.source_sequence or len(state.source_sequence) != len(state.current_sequence):
        raise StateError("source/current length mismatch")
    if any(base not in alphabet for base in state.source_sequence + state.current_sequence):
        raise StateError("state sequence is outside the RNA alphabet")
    edits = state.source_relative_edit_set
    if edits != tuple(sorted(edits)) or len({position for position, _ in edits}) != len(edits):
        raise StateError("source-relative edit set must be sorted with unique positions")
    reconstructed = list(state.source_sequence)
    for position, alt in edits:
        if not isinstance(position, int) or position < 0 or position >= len(reconstructed):
            raise StateError("edit position is invalid")
        if alt not in alphabet or alt == state.source_sequence[position]:
            raise StateError("edit must be source-base to a distinct RNA alt")
        reconstructed[position] = alt
    if "".join(reconstructed) != state.current_sequence:
        raise StateError("current sequence is not exactly source plus the edit set")
    if not isinstance(state.remaining_budget, int) or isinstance(state.remaining_budget, bool) or state.remaining_budget < 0:
        raise StateError("remaining budget must be a nonnegative integer")
    if not isinstance(state.algorithmic_time, (int, float)) or not math.isfinite(state.algorithmic_time):
        raise StateError("algorithmic time must be finite")
    if state.terminal_cause is not None and state.terminal_cause not in TERMINAL_CAUSES:
        raise StateError("unknown terminal cause")


def _edited_positions(state: State) -> set[int]:
    return {position for position, _ in state.source_relative_edit_set}


def legal_edit_specs(state: State, config: Mapping[str, Any]) -> list[tuple[int, str, str]]:
    if state.terminal_cause is not None or state.remaining_budget == 0:
        return []
    edited = _edited_positions(state)
    alphabet = config["graph_contract"]["alphabet"]
    return [
        (position, source_base, alt)
        for position, source_base in enumerate(state.source_sequence)
        if position not in edited
        for alt in alphabet
        if alt != source_base
    ]


def structural_terminal_cause(
    state: State,
    config: Mapping[str, Any],
    *,
    numerical_failure: bool = False,
) -> str | None:
    """Apply the frozen precedence before STOP is made available."""

    validate_state(state, config)
    if numerical_failure:
        return "NUMERICAL_FAILURE"
    if state.remaining_budget == 0:
        return "BUDGET_EXHAUSTED"
    if not legal_edit_specs(replace(state, terminal_cause=None), config):
        return "NO_LEGAL_ACTION"
    return None


def with_structural_terminal(state: State, config: Mapping[str, Any]) -> State:
    cause = structural_terminal_cause(replace(state, terminal_cause=None), config)
    return replace(state, terminal_cause=cause)


def initial_state(case: Mapping[str, Any], config: Mapping[str, Any]) -> State:
    state = State(
        source_sequence=case["source_sequence"],
        current_sequence=case["source_sequence"],
        source_relative_edit_set=(),
        remaining_budget=case["budget"],
        assay=case["assay"],
        context=case["context"],
        algorithmic_time=float(case["algorithmic_time"]),
    )
    return with_structural_terminal(state, config)


def is_action_legal(state: State, action: RawAction, config: Mapping[str, Any]) -> bool:
    """Pure hard legality; callers must invoke this before evaluating a rate."""

    try:
        validate_state(state, config)
    except StateError:
        return False
    if state.terminal_cause is not None or state.remaining_budget == 0:
        return False
    if action.action_type == STOP:
        return bool(legal_edit_specs(state, config))
    if action.action_type != EDIT or action.position is None:
        return False
    if action.position < 0 or action.position >= len(state.source_sequence):
        return False
    if action.position in _edited_positions(state):
        return False
    source_base = state.source_sequence[action.position]
    return (
        action.source_base == source_base
        and action.alt_base in config["graph_contract"]["alphabet"]
        and action.alt_base != source_base
    )


def raw_actions(state: State, config: Mapping[str, Any]) -> list[RawAction]:
    specs = legal_edit_specs(state, config)
    if not specs:
        return []
    aliases = config["rate_parameters"]["raw_alias_shares"]
    actions = [
        RawAction(EDIT, alias["alias_id"], position, source_base, alt)
        for position, source_base, alt in specs
        for alias in aliases[EDIT]
    ]
    actions.extend(RawAction(STOP, alias["alias_id"]) for alias in aliases[STOP])
    return actions


def _alias_share(action: RawAction, config: Mapping[str, Any]) -> float:
    for item in config["rate_parameters"]["raw_alias_shares"][action.action_type]:
        if item["alias_id"] == action.alias_id:
            return float(item["share"])
    raise NumericalFailure(f"unknown raw alias: {action.alias_id}")


def _canonical_base_rate(state: State, action: RawAction, config: Mapping[str, Any]) -> float:
    rates = config["rate_parameters"]
    assay = float(rates["assay_multipliers"][state.assay])
    context = float(rates["context_multipliers"][state.context])
    if action.action_type == STOP:
        return float(rates["support_floor"]) + float(rates["stop_rate"]) * assay * context
    assert action.position is not None and action.source_base is not None and action.alt_base is not None
    return float(rates["support_floor"]) + (
        float(rates["edit_rate_scale"])
        * float(action.position + 1)
        * float(rates["source_base_multipliers"][action.source_base])
        * float(rates["alt_base_multipliers"][action.alt_base])
        * assay
        * context
    )


def raw_action_rate(state: State, action: RawAction, config: Mapping[str, Any]) -> float:
    """Return zero for a prohibited action without evaluating its rate formula."""

    if not is_action_legal(state, action, config):
        return 0.0
    rate = _canonical_base_rate(state, action, config) * _alias_share(action, config)
    if not math.isfinite(rate) or rate <= 0.0:
        raise NumericalFailure("legal raw action rate is not finite and strictly positive")
    return rate


def transition_state(state: State, action: RawAction, config: Mapping[str, Any]) -> State:
    if not is_action_legal(state, action, config):
        raise StateError("cannot apply an illegal action")
    if action.action_type == STOP:
        return replace(state, terminal_cause="EXPLICIT_STOP")
    assert action.position is not None and action.alt_base is not None
    current = list(state.current_sequence)
    current[action.position] = action.alt_base
    edits = tuple(sorted((*state.source_relative_edit_set, (action.position, action.alt_base))))
    child = State(
        source_sequence=state.source_sequence,
        current_sequence="".join(current),
        source_relative_edit_set=edits,
        remaining_budget=state.remaining_budget - 1,
        assay=state.assay,
        context=state.context,
        algorithmic_time=state.algorithmic_time,
    )
    return with_structural_terminal(child, config)


def aggregate_raw_transitions(raw: Iterable[RawTransition]) -> tuple[CanonicalTransition, ...]:
    """Aggregate only aliases with the same complete next extended state."""

    groups: dict[State, list[RawTransition]] = defaultdict(list)
    for transition in raw:
        groups[transition.next_state].append(transition)
    canonical: list[CanonicalTransition] = []
    for next_state, members in groups.items():
        rate = math.fsum(item.rate for item in members)
        if not math.isfinite(rate) or rate <= 0.0:
            raise NumericalFailure("canonical rate is not finite and strictly positive")
        action_types = {item.action.action_type for item in members}
        action_type = next(iter(action_types)) if len(action_types) == 1 else "OBSERVABLE_TRANSITION_ALIAS"
        canonical.append(
            CanonicalTransition(
                action_type=action_type,
                next_state=next_state,
                rate=rate,
                raw_alias_ids=tuple(sorted(item.action.alias_id for item in members)),
                raw_rates=tuple(item.rate for item in members),
            )
        )
    return tuple(sorted(canonical, key=lambda item: state_sort_key(item.next_state)))


def canonical_transitions(state: State, config: Mapping[str, Any]) -> tuple[CanonicalTransition, ...]:
    if state.terminal_cause is not None:
        return ()
    transitions = []
    for action in raw_actions(state, config):
        rate = raw_action_rate(state, action, config)
        if rate <= 0.0:
            raise NumericalFailure("a generated legal action lost positive support")
        transitions.append(RawTransition(action, transition_state(state, action, config), rate))
    if not transitions:
        raise StateError("active state has no STOP/edit transition")
    return aggregate_raw_transitions(transitions)


def state_sort_key(state: State) -> tuple[Any, ...]:
    return (
        state.net_edit_count,
        state.terminal_cause is not None,
        state.current_sequence,
        state.remaining_budget,
        state.assay,
        state.context,
        state.algorithmic_time,
        state.terminal_cause or "",
    )


def build_graph(root: State, config: Mapping[str, Any]) -> Graph:
    root = with_structural_terminal(root, config) if root.terminal_cause is None else root
    queue = deque([root])
    states: set[State] = set()
    outgoing: dict[State, tuple[CanonicalTransition, ...]] = {}
    while queue:
        state = queue.popleft()
        if state in states:
            continue
        validate_state(state, config)
        states.add(state)
        if state.terminal_cause is not None:
            continue
        transitions = canonical_transitions(state, config)
        outgoing[state] = transitions
        for transition in transitions:
            child = transition.next_state
            if child.terminal_cause != "EXPLICIT_STOP" and child.net_edit_count != state.net_edit_count + 1:
                raise StateError("non-STOP edge does not strictly increase edit-set cardinality")
            if child.terminal_cause == "EXPLICIT_STOP" and child.net_edit_count != state.net_edit_count:
                raise StateError("STOP changed the edit set")
            queue.append(child)
    return Graph(root, frozenset(states), outgoing)


def _jump_probabilities(transitions: Sequence[CanonicalTransition]) -> tuple[tuple[CanonicalTransition, float], ...]:
    total = math.fsum(item.rate for item in transitions)
    if not math.isfinite(total) or total <= 0.0:
        raise NumericalFailure("total exit rate is invalid")
    probabilities = tuple((item, item.rate / total) for item in transitions)
    if not math.isclose(math.fsum(probability for _, probability in probabilities), 1.0, rel_tol=0.0, abs_tol=1e-14):
        raise NumericalFailure("embedded jump probabilities do not sum to one")
    return probabilities


def terminal_distribution_dp(graph: Graph, *, rates: Mapping[State, GeneratorRow] | None = None) -> dict[State, float]:
    mass: dict[State, float] = defaultdict(float)
    mass[graph.root] = 1.0
    for state in sorted(graph.outgoing, key=state_sort_key):
        state_mass = mass[state]
        if rates is None:
            jumps = _jump_probabilities(graph.outgoing[state])
            for transition, probability in jumps:
                mass[transition.next_state] += state_mass * probability
        else:
            row = rates[state]
            for child, rate in row.off_diagonal:
                mass[child] += state_mass * rate / row.total_exit_rate
    terminal = {state: mass[state] for state in graph.states if state.terminal_cause is not None and mass[state] != 0.0}
    if not math.isclose(math.fsum(terminal.values()), 1.0, rel_tol=0.0, abs_tol=1e-13):
        raise NumericalFailure("DP terminal distribution does not sum to one")
    return terminal


def enumerate_complete_paths(
    graph: Graph,
    start: State,
    *,
    rates: Mapping[State, GeneratorRow] | None = None,
) -> Iterator[tuple[State, float, tuple[tuple[State, State], ...]]]:
    """Independent exhaustive recursion; deliberately no DP memoization."""

    if start.terminal_cause is not None:
        yield start, 1.0, ()
        return
    if rates is None:
        next_items = tuple(
            (transition.next_state, probability)
            for transition, probability in _jump_probabilities(graph.outgoing[start])
        )
    else:
        row = rates[start]
        next_items = tuple((child, rate / row.total_exit_rate) for child, rate in row.off_diagonal)
    for child, probability in next_items:
        for terminal, suffix_probability, suffix_edges in enumerate_complete_paths(graph, child, rates=rates):
            yield terminal, probability * suffix_probability, ((start, child), *suffix_edges)


def terminal_distribution_enumeration(
    graph: Graph,
    *,
    rates: Mapping[State, GeneratorRow] | None = None,
) -> tuple[dict[State, float], int]:
    terminal: dict[State, float] = defaultdict(float)
    count = 0
    for state, probability, _ in enumerate_complete_paths(graph, graph.root, rates=rates):
        terminal[state] += probability
        count += 1
    if not math.isclose(math.fsum(terminal.values()), 1.0, rel_tol=0.0, abs_tol=1e-13):
        raise NumericalFailure("enumerated terminal distribution does not sum to one")
    return dict(terminal), count


def terminal_weight(state: State, config: Mapping[str, Any], *, unit_tilt: bool = False) -> float:
    if state.terminal_cause is None:
        raise NumericalFailure("terminal w is defined only on absorbing states")
    if unit_tilt:
        return 1.0
    tilt = config["terminal_tilt"]
    weight = float(tilt["offset"]) + float(tilt["net_edit_bonus"]) * state.net_edit_count
    weight += math.fsum(float(tilt["alt_base_bonus"][alt]) for _, alt in state.source_relative_edit_set)
    weight += float(tilt["terminal_cause_bonus"][state.terminal_cause])
    if not math.isfinite(weight) or weight <= 0.0:
        raise NumericalFailure("terminal tilt w is not finite and strictly positive")
    return weight


def harmonic_extension_dp(graph: Graph, config: Mapping[str, Any], *, unit_tilt: bool = False) -> dict[State, float]:
    """Compute scalar h from terminal w by reverse topological jump DP."""

    h = {
        state: terminal_weight(state, config, unit_tilt=unit_tilt)
        for state in graph.states
        if state.terminal_cause is not None
    }
    for state in sorted(graph.outgoing, key=state_sort_key, reverse=True):
        if unit_tilt:
            h[state] = 1.0
        else:
            h[state] = math.fsum(probability * h[transition.next_state] for transition, probability in _jump_probabilities(graph.outgoing[state]))
        if not math.isfinite(h[state]) or h[state] <= 0.0:
            raise NumericalFailure("harmonic extension h is not finite and strictly positive")
    return h


def harmonic_extension_enumeration(
    graph: Graph,
    config: Mapping[str, Any],
    *,
    unit_tilt: bool = False,
) -> dict[State, float]:
    """Compute h independently by enumerating every complete suffix path."""

    h: dict[State, float] = {}
    for state in graph.states:
        value = math.fsum(
            probability * terminal_weight(terminal, config, unit_tilt=unit_tilt)
            for terminal, probability, _ in enumerate_complete_paths(graph, state)
        )
        if not math.isfinite(value) or value <= 0.0:
            raise NumericalFailure("enumerated harmonic extension is invalid")
        h[state] = value
    return h


def generator(graph: Graph, h: Mapping[State, float] | None = None) -> dict[State, GeneratorRow]:
    rows: dict[State, GeneratorRow] = {}
    for state, transitions in graph.outgoing.items():
        off_diagonal = []
        for transition in transitions:
            rate = transition.rate
            if h is not None:
                rate *= h[transition.next_state] / h[state]
            if not math.isfinite(rate) or rate <= 0.0:
                raise NumericalFailure("generator off-diagonal rate is invalid")
            off_diagonal.append((transition.next_state, rate))
        total = math.fsum(rate for _, rate in off_diagonal)
        rows[state] = GeneratorRow(tuple(off_diagonal), -total, total)
    return rows


def _relative_error(observed: float, expected: float) -> float:
    if not math.isfinite(observed) or not math.isfinite(expected) or expected == 0.0:
        raise NumericalFailure("relative-error operands are invalid")
    return abs(observed - expected) / abs(expected)


def _tv(left: Mapping[State, float], right: Mapping[State, float]) -> float:
    return 0.5 * math.fsum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in set(left) | set(right))


def _rate_map(row: GeneratorRow) -> dict[State, float]:
    return dict(row.off_diagonal)


def _time_free_state_key(state: State) -> tuple[Any, ...]:
    return (
        state.source_sequence,
        state.current_sequence,
        state.source_relative_edit_set,
        state.remaining_budget,
        state.assay,
        state.context,
        state.terminal_cause,
    )


def _fixture_terminal_causes(config: Mapping[str, Any]) -> dict[str, str]:
    common = {"assay": "TOY_ASSAY_ALPHA", "context": "TOY_CONTEXT_LEFT", "algorithmic_time": 0.2}
    observed: dict[str, str] = {}
    for expected, fixture in config["terminal_fixture_states"].items():
        state = State(
            source_sequence=fixture["source_sequence"],
            current_sequence=fixture["current_sequence"],
            source_relative_edit_set=tuple(tuple(item) for item in fixture["source_relative_edit_set"]),
            remaining_budget=fixture["remaining_budget"],
            **common,
        )
        if fixture.get("numerical_failure"):
            cause = structural_terminal_cause(state, config, numerical_failure=True)
        elif expected == "EXPLICIT_STOP":
            stop_action = next(action for action in raw_actions(state, config) if action.action_type == STOP)
            cause = transition_state(state, stop_action, config).terminal_cause
        else:
            cause = structural_terminal_cause(state, config)
        if cause != expected:
            raise NumericalFailure(f"terminal fixture {expected} produced {cause}")
        observed[expected] = cause
    if set(observed.values()) != set(TERMINAL_CAUSES):
        raise NumericalFailure("terminal causes are not four distinct values")
    return observed


def evaluate_case(case: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    graph = build_graph(initial_state(case, config), config)
    base_dp = terminal_distribution_dp(graph)
    base_enum, path_count = terminal_distribution_enumeration(graph)
    h_dp = harmonic_extension_dp(graph, config)
    h_enum = harmonic_extension_enumeration(graph, config)
    base_generator = generator(graph)
    guided_dp_generator = generator(graph, h_dp)
    guided_enum_generator = generator(graph, h_enum)
    guided_dp = terminal_distribution_dp(graph, rates=guided_dp_generator)
    guided_enum, guided_path_count = terminal_distribution_enumeration(graph, rates=guided_dp_generator)
    if guided_path_count != path_count:
        raise NumericalFailure("base/guided path counts differ")
    normalizer = h_dp[graph.root]
    tilted = {
        terminal: probability * terminal_weight(terminal, config) / normalizer
        for terminal, probability in base_dp.items()
    }
    if not math.isclose(math.fsum(tilted.values()), 1.0, rel_tol=0.0, abs_tol=1e-13):
        raise NumericalFailure("tilted base terminal law does not sum to one")

    harmonic_error = max((_relative_error(h_dp[state], h_enum[state]) for state in graph.states), default=0.0)
    true_rate_error = 0.0
    total_exit_error = 0.0
    diagonal_error = 0.0
    min_support = math.inf
    alias_error = 0.0
    alias_counts: list[int] = []
    for state, transitions in graph.outgoing.items():
        dp_rates = _rate_map(guided_dp_generator[state])
        enum_rates = _rate_map(guided_enum_generator[state])
        base_row = base_generator[state]
        guided_row = guided_dp_generator[state]
        true_rate_error = max(
            true_rate_error,
            *(_relative_error(dp_rates[child], enum_rates[child]) for child in dp_rates),
        )
        total_exit_error = max(total_exit_error, _relative_error(guided_row.total_exit_rate, base_row.total_exit_rate))
        diagonal_error = max(diagonal_error, abs(guided_row.diagonal + math.fsum(rate for _, rate in guided_row.off_diagonal)))
        for transition in transitions:
            min_support = min(min_support, transition.rate)
            alias_error = max(alias_error, abs(transition.rate - math.fsum(transition.raw_rates)))
            alias_counts.append(len(transition.raw_alias_ids))

    unit_h = harmonic_extension_dp(graph, config, unit_tilt=True)
    unit_generator = generator(graph, unit_h)
    base_recovery_error = 0.0
    for state, row in base_generator.items():
        base_rates = _rate_map(row)
        unit_rates = _rate_map(unit_generator[state])
        base_recovery_error = max(
            base_recovery_error,
            *(_relative_error(unit_rates[child], base_rates[child]) for child in base_rates),
        )

    path_product_error = 0.0
    for terminal, _, edges in enumerate_complete_paths(graph, graph.root):
        product = 1.0
        for parent, child in edges:
            product *= _rate_map(guided_dp_generator[parent])[child] / _rate_map(base_generator[parent])[child]
        expected = terminal_weight(terminal, config) / h_dp[graph.root]
        path_product_error = max(path_product_error, _relative_error(product, expected))

    time_invariance_error = 0.0
    for state in graph.outgoing:
        shifted = replace(state, algorithmic_time=state.algorithmic_time + 0.9375)
        original = {
            _time_free_state_key(item.next_state): item.rate for item in canonical_transitions(state, config)
        }
        shifted_rates = {
            _time_free_state_key(item.next_state): item.rate for item in canonical_transitions(shifted, config)
        }
        if set(original) != set(shifted_rates):
            raise NumericalFailure("time shift changed canonical transition support")
        time_invariance_error = max(
            time_invariance_error,
            *(abs(original[key] - shifted_rates[key]) for key in original),
        )

    budget = int(case["budget"])
    budget_violations = sum(state.net_edit_count > budget for state in graph.states)
    event_count_violations = sum(
        state.net_edit_count
        != sum(left != right for left, right in zip(state.source_sequence, state.current_sequence))
        for state in graph.states
    )
    support_ok = (not graph.outgoing) or (min_support > 0.0 and min(alias_counts) >= 2)
    metrics = {
        "base_dp_vs_enumeration_tv": _tv(base_dp, base_enum),
        "harmonic_extension_relative_error": harmonic_error,
        "true_per_rate_relative_error": true_rate_error,
        "guided_terminal_tv_vs_tilted_base": max(_tv(guided_dp, tilted), _tv(guided_enum, tilted)),
        "path_product_relative_error": path_product_error,
        "w_equals_one_base_recovery_relative_error": base_recovery_error,
        "guided_vs_base_total_exit_rate_relative_error": total_exit_error,
        "guided_generator_diagonal_absolute_error": diagonal_error,
        "alias_aggregation_absolute_error": alias_error,
        "time_invariance_absolute_error": time_invariance_error,
    }
    tolerances = config["numerical_tolerances"]
    checks = {
        "base_dp_vs_enumeration": metrics["base_dp_vs_enumeration_tv"] <= tolerances["base_dp_vs_enumeration_tv_max"],
        "harmonic_extension": metrics["harmonic_extension_relative_error"] <= tolerances["harmonic_extension_relative_error_max"],
        "true_per_rate": metrics["true_per_rate_relative_error"] <= tolerances["true_rate_relative_error_max"],
        "terminal_tv": metrics["guided_terminal_tv_vs_tilted_base"] <= tolerances["terminal_distribution_tv_max"],
        "path_product": metrics["path_product_relative_error"] <= tolerances["path_product_relative_error_max"],
        "base_recovery": metrics["w_equals_one_base_recovery_relative_error"] <= tolerances["base_recovery_relative_error_max"],
        "total_exit_rate": metrics["guided_vs_base_total_exit_rate_relative_error"] <= tolerances["total_exit_rate_relative_error_max"],
        "generator_diagonal": diagonal_error <= tolerances["total_exit_rate_relative_error_max"],
        "alias_aggregation": alias_error <= tolerances["alias_aggregation_absolute_error_max"],
        "time_invariance": time_invariance_error == 0.0,
        "positive_support": support_ok,
        "budget": budget_violations == 0,
        "event_count_equals_net_edits": event_count_violations == 0,
        "absorbing_dag": all(
            child.terminal_cause is not None or child.net_edit_count == state.net_edit_count + 1
            for state, transitions in graph.outgoing.items()
            for child in (item.next_state for item in transitions)
        ),
    }
    if not all(checks.values()) or not all(math.isfinite(value) for value in metrics.values()):
        failed = sorted(key for key, passed in checks.items() if not passed)
        raise NumericalFailure(f"case {case['case_id']} failed exact checks: {failed}")
    terminal_counts = {cause: sum(state.terminal_cause == cause for state in graph.states) for cause in TERMINAL_CAUSES}
    return {
        "case_id": case["case_id"],
        "sequence_length": len(case["source_sequence"]),
        "budget": budget,
        "state_count": len(graph.states),
        "active_state_count": len(graph.outgoing),
        "terminal_state_count": len(graph.states) - len(graph.outgoing),
        "complete_path_count": path_count,
        "terminal_cause_state_counts": terminal_counts,
        "budget_violation_count": budget_violations,
        "event_count_violation_count": event_count_violations,
        "minimum_canonical_positive_support": None if math.isinf(min_support) else min_support,
        "metrics": metrics,
        "checks": {key: "PASS" for key in checks},
        "status": "PASS",
    }


def run_exact_suite(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_static_config(config)
    terminal_fixtures = _fixture_terminal_causes(config)
    cases = [evaluate_case(case, config) for case in config["fixed_cases"]]
    metric_names = cases[0]["metrics"]
    aggregate_maxima = {
        name: max(case["metrics"][name] for case in cases) for name in metric_names
    }
    total_states = sum(case["state_count"] for case in cases)
    total_paths = sum(case["complete_path_count"] for case in cases)
    return {
        "status": "PASS",
        "fixed_case_count": len(cases),
        "total_state_count": total_states,
        "total_complete_path_count": total_paths,
        "terminal_fixture_causes": terminal_fixtures,
        "aggregate_maximum_metrics": aggregate_maxima,
        "cases": cases,
        "fixture_results": {
            "SOURCE_ANCHOR_AND_NO_REEDIT_OR_REVERT": "PASS",
            "HARD_LEGALITY_BEFORE_RATES": "PASS",
            "FOUR_DISTINCT_TERMINAL_CAUSES": "PASS",
            "POSITIVE_SUPPORT": "PASS",
            "STOP_AND_BUDGET": "PASS",
            "TRANSITION_ALIAS_AGGREGATION_BY_FULL_NEXT_EXTENDED_STATE": "PASS",
            "FINITE_ABSORBING_DAG": "PASS",
            "INDEPENDENT_EXHAUSTIVE_PATH_ENUMERATION": "PASS",
            "STRICTLY_POSITIVE_TERMINAL_W_AND_SCALAR_H_V": "PASS",
            "TRUE_PER_RATE_DOOB_COMPARISON": "PASS",
            "GUIDED_AND_BASE_TOTAL_EXIT_RATES_EQUAL": "PASS",
            "W_EQUALS_ONE_BASE_RECOVERY": "PASS",
            "CONTINUOUS_ALGORITHMIC_TIME_TIME_HOMOGENEOUS_FIXTURE": "PASS",
        },
    }


def _git_bytes(repo: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", os.fspath(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        error = completed.stderr.decode("utf-8", errors="replace").strip()
        raise AuthorityError(f"git {' '.join(args)} failed: {error}")
    return completed.stdout


def _git_text(repo: Path, *args: str) -> str:
    return _git_bytes(repo, *args).decode("utf-8").strip()


def _commit_parent(repo: Path, commit: str) -> str:
    return _git_text(repo, "rev-parse", f"{commit}^")


def _changed_paths(repo: Path, commit: str) -> list[str]:
    return sorted(
        line
        for line in _git_text(
            repo,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            commit,
        ).splitlines()
        if line
    )


def _git_config(repo: Path, commit: str) -> dict[str, Any]:
    return load_json_bytes(
        _git_bytes(repo, "show", f"{commit}:{CONFIG_REPO_PATH}"),
        label=f"{commit}:{CONFIG_REPO_PATH}",
    )


def _validate_active_contract(repo: Path, config: Mapping[str, Any]) -> None:
    authority = config["authority"]
    goal_path = authority["goal_path"]
    active_path = authority["active_config_path"]
    try:
        goal_payload = (repo / goal_path).read_bytes()
        active_payload = (repo / active_path).read_bytes()
    except OSError as exc:
        raise AuthorityError("active goal or config is unavailable") from exc
    if sha256(goal_payload) != GOAL_SHA256 or sha256(active_payload) != ACTIVE_CONFIG_SHA256:
        raise AuthorityError("active goal or config SHA differs")
    try:
        goal_text = goal_payload.decode("utf-8")
        active_text = active_payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AuthorityError("active goal or config is not UTF-8") from exc
    for literal in authority["goal_critical_literals"]:
        if literal not in goal_text:
            raise AuthorityError(f"goal critical literal is absent: {literal}")
    for literal in authority["active_config_critical_literals"]:
        if literal not in active_text:
            raise AuthorityError(f"active config critical literal is absent: {literal}")
    if active_text.count(f"sha256: {GOAL_SHA256}") != 1 or active_text.count(
        f"repository_path: {goal_path}"
    ) != 1:
        raise AuthorityError("active config does not bind one unique goal path/SHA")
    tracked_goals = _git_text(repo, "ls-files", "docs/goals").splitlines()
    matches = [path for path in tracked_goals if sha256((repo / path).read_bytes()) == GOAL_SHA256]
    if matches != [goal_path]:
        raise AuthorityError("the contract SHA is not unique at the frozen goal path")


def validate_production_authority(
    config: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
) -> dict[str, str]:
    """Prove exact base -> exact3 I -> config-only B and active authority."""

    validate_static_config(config)
    if _binding_mode(config) != "BOUND":
        raise AuthorityError("production implementation binding is not BOUND")
    binding = config["implementation_binding"]
    implementation = binding["implementation_commit"]
    repo = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    top = Path(_git_text(repo, "rev-parse", "--show-toplevel")).resolve()
    if top != repo.resolve():
        raise AuthorityError("script is not under the production Git root")
    if _git_text(repo, "status", "--porcelain"):
        raise AuthorityError("production worktree is not clean")
    branch = _git_text(repo, "branch", "--show-current")
    if not branch:
        raise AuthorityError("production HEAD is detached")
    head = _git_text(repo, "rev-parse", "HEAD")
    upstream = _git_text(repo, "rev-parse", "@{u}")
    remote = config["authority"]["required_remote"]
    origin = _git_text(repo, "rev-parse", f"refs/remotes/{remote}/{branch}")
    upstream_ref = _git_text(repo, "rev-parse", "--symbolic-full-name", "@{u}")
    if upstream_ref != f"refs/remotes/{remote}/{branch}" or head != upstream or head != origin:
        raise AuthorityError("production HEAD, upstream, and origin tracking ref are not identical")

    if _commit_parent(repo, head) != implementation:
        raise AuthorityError("binding B is not the direct child of implementation I")
    if _commit_parent(repo, implementation) != FROZEN_BASE_COMMIT:
        raise AuthorityError("implementation I is not the direct child of the frozen base")
    if _changed_paths(repo, implementation) != sorted(EXACT_IMPLEMENTATION_PATHS):
        raise AuthorityError("implementation I changed-path set is not exact3")
    if _changed_paths(repo, head) != [CONFIG_REPO_PATH]:
        raise AuthorityError("binding B changed-path set is not config-only")

    i_config = _git_config(repo, implementation)
    b_config = _git_config(repo, head)
    validate_static_config(i_config)
    validate_static_config(b_config)
    if _binding_mode(i_config) != "UNKNOWN":
        raise AuthorityError("implementation I config does not leave exactly four UNKNOWN scalars")
    if b_config != dict(config):
        raise AuthorityError("runtime config does not equal binding B config")
    if candidate_i_config(b_config) != i_config:
        raise AuthorityError("binding B config differs from I outside the exact four scalars")
    if b_config["implementation_binding"]["implementation_commit"] != implementation:
        raise AuthorityError("binding B does not name implementation I")

    try:
        worktree_config = load_json_bytes(
            (repo / CONFIG_REPO_PATH).read_bytes(), label=f"worktree:{CONFIG_REPO_PATH}"
        )
    except OSError as exc:
        raise AuthorityError("worktree binding config is unavailable") from exc
    if worktree_config != b_config:
        raise AuthorityError("worktree config does not equal binding B config")

    for path, hash_key in (
        (SCRIPT_REPO_PATH, "implementation_script_sha256"),
        (TEST_REPO_PATH, "implementation_test_sha256"),
    ):
        expected_sha = binding[hash_key]
        try:
            worktree_payload = (repo / path).read_bytes()
        except OSError as exc:
            raise AuthorityError(f"worktree implementation leaf is unavailable: {path}") from exc
        payloads = (
            _git_bytes(repo, "show", f"{implementation}:{path}"),
            _git_bytes(repo, "show", f"{head}:{path}"),
            worktree_payload,
        )
        if any(sha256(payload) != expected_sha for payload in payloads):
            raise AuthorityError(f"I/B/worktree blob differs from bound SHA: {path}")

    _validate_active_contract(repo, config)
    return {
        "head": head,
        "binding_commit": head,
        "implementation_commit": implementation,
        "branch": branch,
        "upstream_ref": upstream_ref,
        "implementation_script_sha256": binding["implementation_script_sha256"],
        "implementation_test_sha256": binding["implementation_test_sha256"],
    }


def _parse_recorded_at(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ConfigError("recorded_at must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ConfigError("recorded_at must include a timezone offset")
    return value


def build_public_payloads(
    config: Mapping[str, Any],
    suite: Mapping[str, Any],
    *,
    recorded_at: str,
    authority: Mapping[str, str] | None,
) -> dict[str, bytes]:
    recorded_at = _parse_recorded_at(recorded_at)
    status = config["status_contract"]
    report = {
        "schema_version": "route_a_v3_a6_cpu_exact_absorbing_dag_report.v1",
        "protocol_id": config["protocol_id"],
        "recorded_at": recorded_at,
        "run_scope": config["run_scope"],
        "production_python": config["production_python"],
        "run_status": "PASS",
        "phase_state": {"phase_id": "A6", "evidence_status": "IN_PROGRESS"},
        "task_states": {
            "EXACT_GUIDANCE_TOY_GRAPH": {
                "evidence_status": status["exact_guidance_toy_graph_evidence_status"],
                "result": status["exact_guidance_toy_graph_result"],
                "scope": status["exact_guidance_toy_graph_scope"],
            },
            "FLOW_BASE_LEGAL_CTMC": {"evidence_status": "NOT_RUN"},
        },
        "claim_state": {
            "claim_id": "L3_LEGAL_POTENTIAL_CONSISTENT_XEDITFLOW",
            "evidence_status": "IN_PROGRESS",
            "claim_status": "NOT_ESTABLISHED",
        },
        "time_scope": {
            "clock_semantics": config["clock_semantics"],
            "rate_time_dependence": config["rate_time_dependence"],
            "terminal_tilt_time_dependence": config["terminal_tilt_time_dependence"],
            "general_time_inhomogeneous_exactness": "NOT_RUN",
            "dp_time_key_rule": config["time_quotient_rule"],
        },
        "mathematical_scope": {
            "terminal_w": "STRICTLY_POSITIVE_TERMINAL_TILT_ONLY",
            "harmonic_h": "EMBEDDED_JUMP_HARMONIC_EXTENSION_OF_W",
            "potential_v": "LOG_H",
            "guided_offdiagonal_rate": "Q_H_EQUALS_Q_TIMES_H_NEXT_OVER_H_CURRENT",
            "guided_diagonal": "NEGATIVE_SUM_OF_GUIDED_OFFDIAGONAL_RATES",
            "guided_base_total_exit_rates_equal": True,
        },
        "algorithms": config["algorithms"],
        "numerical_tolerances": config["numerical_tolerances"],
        "exact_suite": suite,
        "boundaries": {
            "learned_potential_approximation_error": "NOT_RUN",
            "ordinary_flow": "NOT_RUN",
            "a7_unlock": False,
            "training_allowed": False,
            "model_selection_allowed": False,
            "gpu_work_allowed": False,
            "private_payload_access_allowed": False,
            "sealed_contact_allowed": False,
        },
    }
    manifest = {
        "schema_version": "route_a_v3_a6_cpu_exact_absorbing_dag_run_manifest.v1",
        "protocol_id": config["protocol_id"],
        "recorded_at": recorded_at,
        "run_scope": config["run_scope"],
        "production_python": config["production_python"],
        "run_status": "PASS",
        "cpu_only": True,
        "learned_parameter_count": 0,
        "training_run_count": 0,
        "gpu_work_count": 0,
        "ordinary_row_read_count": 0,
        "private_payload_read_count": 0,
        "sealed_contact_count": 0,
        "authority": dict(authority) if authority is not None else {"mode": "SYNTHETIC_TEST_NO_GIT"},
        "output_count": 3,
        "outputs": [
            {"name": OUTPUT_NAMES[0], "artifact_type": "PUBLIC_AGGREGATE_REPORT"},
            {"name": OUTPUT_NAMES[1], "artifact_type": "PUBLIC_AGGREGATE_RUN_MANIFEST"},
            {"name": OUTPUT_NAMES[2], "artifact_type": "PUBLIC_AGGREGATE_EVENT_LOG"},
        ],
        "phase_evidence_status": "IN_PROGRESS",
        "exact_guidance_toy_graph_result": "DEVELOPMENT_CPU_EXACT_FIXTURE_PASS",
        "flow_base_legal_ctmc_evidence_status": "NOT_RUN",
        "l3_claim_status": "NOT_ESTABLISHED",
        "a7_unlock": False,
    }
    event = {
        "event_id": "A6-CPU-EXACT-001",
        "at": recorded_at,
        "event": "A6_CPU_EXACT_ABSORBING_DAG_FIXTURE_COMPLETED",
        "run_status": "PASS",
        "a6_evidence_status": "IN_PROGRESS",
        "exact_guidance_toy_graph_result": "DEVELOPMENT_CPU_EXACT_FIXTURE_PASS",
        "exact_guidance_toy_graph_scope": "SYNTHETIC_TIME_HOMOGENEOUS_CPU_EXACT",
        "flow_base_legal_ctmc_evidence_status": "NOT_RUN",
        "l3_evidence_status": "IN_PROGRESS",
        "l3_claim_status": "NOT_ESTABLISHED",
        "general_time_inhomogeneous_exactness": "NOT_RUN",
        "learned_potential_approximation_error": "NOT_RUN",
        "a7_unlock": False,
        "training_allowed": False,
        "gpu_work_allowed": False,
        "private_payload_access_allowed": False,
        "sealed_contact_allowed": False,
        "fixed_case_count": suite["fixed_case_count"],
        "aggregate_maximum_metrics": suite["aggregate_maximum_metrics"],
    }
    payloads = {
        OUTPUT_NAMES[0]: json_bytes(report),
        OUTPUT_NAMES[1]: json_bytes(manifest),
        OUTPUT_NAMES[2]: json_line(event),
    }
    for name, payload in payloads.items():
        load_json_bytes(payload.rstrip(b"\n") if name.endswith(".jsonl") else payload, label=name)
    return payloads


def _validate_output_path(output_directory: Path, config: Mapping[str, Any], *, production: bool) -> Path:
    output = output_directory.resolve(strict=False)
    if not output.is_absolute():
        raise PublicationError("output directory must be absolute")
    if production:
        expected = Path(config["publication"]["output_directory"]).resolve(strict=False)
        allowed = Path(config["publication"]["allowed_output_root"]).resolve(strict=False)
        if output != expected or output.parent != allowed or not str(output).startswith("/mnt/"):
            raise PublicationError("production output is not the exclusive frozen ordinary /mnt directory")
    if output.exists():
        raise PublicationError("exclusive output directory already exists")
    return output


def publish_exact_three(output_directory: Path, payloads: Mapping[str, bytes]) -> None:
    if set(payloads) != set(OUTPUT_NAMES) or len(payloads) != 3:
        raise PublicationError("publication payload is not exactly the frozen three files")
    parent = output_directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_directory.name}.tmp.", dir=parent))
    try:
        for name in OUTPUT_NAMES:
            path = temporary / name
            with path.open("xb") as stream:
                stream.write(payloads[name])
                stream.flush()
                os.fsync(stream.fileno())
        if sorted(path.name for path in temporary.iterdir()) != sorted(OUTPUT_NAMES):
            raise PublicationError("temporary publication is not exactly three files")
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.rename(temporary, output_directory)
        parent_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception:
        if temporary.exists():
            for path in temporary.iterdir():
                if path.is_file():
                    path.unlink()
            temporary.rmdir()
        raise


def execute(
    *,
    output_directory: Path,
    recorded_at: str,
    production: bool = True,
    config_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate authority/numerics first, then atomically publish exactly three."""

    config = dict(config_override) if config_override is not None else load_config()
    validate_static_config(config)
    authority = validate_production_authority(config) if production else None
    output = _validate_output_path(output_directory, config, production=production)
    suite = run_exact_suite(config)
    payloads = build_public_payloads(config, suite, recorded_at=recorded_at, authority=authority)
    publish_exact_three(output, payloads)
    return {
        "status": "PASS",
        "output_directory": str(output),
        "output_names": list(OUTPUT_NAMES),
        "a6_evidence_status": "IN_PROGRESS",
        "exact_guidance_toy_graph_result": "DEVELOPMENT_CPU_EXACT_FIXTURE_PASS",
        "flow_base_legal_ctmc_evidence_status": "NOT_RUN",
        "l3_claim_status": "NOT_ESTABLISHED",
        "a7_unlock": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="must equal the frozen production output directory",
    )
    parser.add_argument(
        "--recorded-at",
        default=None,
        help="timezone-aware ISO-8601 time; defaults to the local current time",
    )
    args = parser.parse_args(argv)
    config = load_config()
    output = args.output_directory or Path(config["publication"]["output_directory"])
    recorded_at = args.recorded_at or datetime.now().astimezone().isoformat(timespec="seconds")
    result = execute(output_directory=output, recorded_at=recorded_at, config_override=config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
