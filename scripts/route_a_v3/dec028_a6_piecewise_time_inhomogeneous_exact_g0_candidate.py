#!/usr/bin/env python3
"""Bounded DEC028 A6 time-inhomogeneous synthetic exact-reference candidate.

This module models a finite source-anchored edit DAG as a continuous
algorithmic-time CTMC.  Rates are piecewise constant over a finite schedule
and deliberately change the relative edit/STOP probabilities between segments.
Uniformization supplies the primary reference together with an explicit
Poisson-tail bound; independently refined RK4 is a numerical cross-check.
The model is synthetic, nonlearned, CPU-only, and intentionally does not claim
physical kinetics, arbitrary time-varying-rate exactness, A6 PASS, L3, A7, or
any biological result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs/route_a_v3_dec028_a6_piecewise_time_inhomogeneous_exact_g0_candidate_v1.json"
)
BEFORE_DECISION_IDS = (
    "V3-DEC-017",
    "V3-DEC-018",
    "V3-DEC-019",
    "V3-DEC-020",
    "V3-DEC-021",
    "V3-DEC-022",
    "V3-DEC-023",
    "V3-DEC-024",
    "V3-DEC-027",
)
FROZEN_COUNTS = {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547}
EDIT = "EDIT"
STOP = "STOP"
TERMINAL_CAUSES = ("BUDGET_EXHAUSTED", "EXPLICIT_STOP")


class CandidateError(RuntimeError):
    """A static boundary, synthetic CTMC, or numerical reference invariant failed."""


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CandidateError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise CandidateError(f"non-finite JSON constant: {value}")


def load_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateError(f"invalid JSON for {label}") from exc
    if not isinstance(value, dict):
        raise CandidateError(f"JSON root is not an object for {label}")
    return value


def _expect(observed: Any, expected: Any, *, label: str) -> None:
    if observed != expected:
        raise CandidateError(f"{label} differs: expected {expected!r}, observed {observed!r}")


def _positive(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise CandidateError(f"{label} is not finite positive")
    return result


def _nonnegative(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise CandidateError(f"{label} is not finite nonnegative")
    return result


def _expect_keys(value: Any, expected: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CandidateError(f"{label} key set differs")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _expect(
        set(config),
        {
            "schema_version",
            "protocol_id",
            "authority_status",
            "static_authority",
            "synthetic_fixture_generator",
            "transition_contract",
            "time_inhomogeneous_reference_contract",
            "forbidden_operations",
            "runtime_truth",
        },
        label="config root keys",
    )
    _expect(
        config["schema_version"],
        "route_a_v3_dec028_a6_piecewise_time_inhomogeneous_exact_g0_candidate.v1",
        label="schema version",
    )
    _expect(
        config["protocol_id"],
        "ROUTE_A_V3_DEC028_A6_PIECEWISE_TIME_INHOMOGENEOUS_EXACT_G0_CANDIDATE_V1",
        label="protocol ID",
    )
    _expect(
        config["authority_status"],
        "NON_AUTHORITATIVE_G0_NONLEARNED_SYNTHETIC_TIME_INHOMOGENEOUS_PREPARATION",
        label="authority status",
    )

    authority = _expect_keys(
        config["static_authority"],
        {
            "root_config_path",
            "root_config_sha256",
            "dec028_amendment_path",
            "dec028_amendment_sha256",
            "effective_active_amendment_decision_ids",
            "pending_successor_amendment_decision_ids",
            "current_qualified_counts",
            "scientific_claim_status",
        },
        label="static authority",
    )
    _expect(authority["root_config_path"], "configs/route_a_v3.yaml", label="root config path")
    _expect(
        authority["root_config_sha256"],
        "1f11e6a84ed394aecc5ef7a5626b7a07b2a877a4aa8c2a4c67a3d79e9771aca8",
        label="root config hash",
    )
    _expect(
        authority["dec028_amendment_path"],
        "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec028.yaml",
        label="amendment path",
    )
    _expect(
        authority["dec028_amendment_sha256"],
        "bd0e845daca76a75998b3bca3b8d2b93a9011a0cfb1ec8b40acd7ef133fed3c8",
        label="amendment hash",
    )
    _expect(
        tuple(authority["effective_active_amendment_decision_ids"]),
        BEFORE_DECISION_IDS,
        label="effective decisions",
    )
    _expect(authority["pending_successor_amendment_decision_ids"], ["V3-DEC-028"], label="pending decision")
    _expect(authority["current_qualified_counts"], FROZEN_COUNTS, label="qualified counts")
    _expect(authority["scientific_claim_status"], "NOT_ESTABLISHED", label="scientific claim")

    fixtures = _expect_keys(
        config["synthetic_fixture_generator"],
        {
            "source_lengths",
            "budgets",
            "deterministic_variants",
            "expected_fixture_count",
            "alphabet",
            "source_anchored",
            "repeated_position_edit_allowed",
            "revert_to_source_allowed",
        },
        label="fixture generator",
    )
    _expect(fixtures["source_lengths"], [2, 3, 4, 5], label="source lengths")
    _expect(fixtures["budgets"], [1, 3, 5], label="budgets")
    _expect(fixtures["deterministic_variants"], 8, label="variant count")
    _expect(fixtures["expected_fixture_count"], 96, label="fixture count")
    _expect(fixtures["alphabet"], ["A", "C"], label="synthetic alphabet")
    for key, expected in (
        ("source_anchored", True),
        ("repeated_position_edit_allowed", False),
        ("revert_to_source_allowed", False),
    ):
        _expect(fixtures[key], expected, label=f"fixture generator {key}")

    transition = _expect_keys(
        config["transition_contract"],
        {
            "support_floor",
            "edit_rate_scale",
            "stop_rate",
            "raw_aliases_per_canonical_transition",
            "raw_alias_shares",
            "aggregate_aliases_by",
            "stop_is_competing_positive_rate",
            "terminal_causes",
            "budget_invariant",
        },
        label="transition contract",
    )
    for key in ("support_floor", "edit_rate_scale", "stop_rate"):
        _positive(transition[key], label=key)
    _expect(transition["raw_aliases_per_canonical_transition"], 2, label="raw alias count")
    _expect(transition["raw_alias_shares"], [0.4, 0.6], label="raw alias shares")
    _expect(
        transition["aggregate_aliases_by"],
        "FULL_NEXT_EXTENDED_STATE_BEFORE_NORMALIZATION",
        label="alias aggregation",
    )
    _expect(transition["stop_is_competing_positive_rate"], True, label="STOP support")
    _expect(transition["terminal_causes"], list(TERMINAL_CAUSES), label="terminal causes")
    _expect(
        transition["budget_invariant"],
        "remaining_budget_plus_net_edit_count_is_constant",
        label="budget invariant",
    )

    reference = _expect_keys(
        config["time_inhomogeneous_reference_contract"],
        {
            "clock_semantics",
            "finite_schedule_type",
            "finite_segments",
            "post_schedule_tail",
            "primary_reference",
            "independent_crosscheck",
            "uniformization_tail_probability_max",
            "independent_terminal_distribution_tv_max",
            "reference_scope",
            "contract_wide_general_time_inhomogeneous_exactness",
            "physical_kinetics_claim",
        },
        label="time-inhomogeneous reference contract",
    )
    _expect(
        reference["clock_semantics"],
        "CONTINUOUS_ALGORITHMIC_TIME_NOT_PHYSICAL_KINETICS",
        label="clock semantics",
    )
    _expect(
        reference["finite_schedule_type"],
        "THREE_PIECEWISE_CONSTANT_NONCOMMON_RATE_SEGMENTS",
        label="finite schedule type",
    )
    _expect(
        reference["primary_reference"],
        "PIECEWISE_CONSTANT_CTMC_UNIFORMIZATION_WITH_RECORDED_POISSON_TAIL_BOUND",
        label="primary reference",
    )
    _expect(
        reference["independent_crosscheck"],
        "KOLMOGOROV_FORWARD_RK4_STEP_REFINEMENT_PLUS_COMPLETE_TERMINAL_PATH_ENUMERATION_VERSUS_TOPOLOGICAL_TAIL_DP",
        label="independent reference",
    )
    _expect(
        reference["contract_wide_general_time_inhomogeneous_exactness"],
        "NOT_ESTABLISHED_NOT_CLAIMED",
        label="contract-wide time scope",
    )
    _expect(reference["physical_kinetics_claim"], "NOT_CLAIMED", label="physical claim")
    _positive(reference["uniformization_tail_probability_max"], label="uniformization tail")
    _positive(reference["independent_terminal_distribution_tv_max"], label="independent TV")
    segments = reference["finite_segments"]
    if not isinstance(segments, list) or len(segments) != 3:
        raise CandidateError("finite schedule must have exactly three segments")
    previous_end = 0.0
    multiplier_pairs: set[tuple[float, float]] = set()
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping) or set(segment) != {"start", "end", "edit_multiplier", "stop_multiplier"}:
            raise CandidateError("finite segment shape differs")
        start = _nonnegative(segment["start"], label=f"segment {index} start")
        end = _positive(segment["end"], label=f"segment {index} end")
        if not math.isclose(start, previous_end, abs_tol=0.0):
            raise CandidateError("finite segments are not contiguous")
        if end <= start:
            raise CandidateError("finite segment has nonpositive duration")
        edit = _positive(segment["edit_multiplier"], label=f"segment {index} edit multiplier")
        stop = _positive(segment["stop_multiplier"], label=f"segment {index} stop multiplier")
        if math.isclose(edit, stop, rel_tol=0.0, abs_tol=1e-15):
            raise CandidateError("finite segment reduces time dependence to a common scale")
        multiplier_pairs.add((edit, stop))
        previous_end = end
    if len(multiplier_pairs) < 2 or not math.isclose(previous_end, 1.0, abs_tol=1e-15):
        raise CandidateError("finite schedule does not supply the frozen nontrivial horizon")
    tail = _expect_keys(reference["post_schedule_tail"], {"edit_multiplier", "stop_multiplier"}, label="tail schedule")
    if math.isclose(
        _positive(tail["edit_multiplier"], label="tail edit multiplier"),
        _positive(tail["stop_multiplier"], label="tail stop multiplier"),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise CandidateError("tail reduces edit and STOP to a common scale")

    expected_forbidden = {
        "PROJECT_ROW_OR_SEQUENCE_READ",
        "REAL_TRAJECTORY_OR_MEMBERSHIP_INPUT",
        "CUDA_OR_GPU_TOUCH",
        "TORCH_MODEL_OR_DATALOADER",
        "OPTIMIZER_OR_PARAMETER_UPDATE",
        "CHECKPOINT_READ_OR_WRITE",
        "RUNTIME_OUTPUT_WRITE",
        "A6_PASS_OR_L3_CLAIM",
        "A7_UNLOCK",
        "SEALED_ACCESS",
    }
    if not isinstance(config["forbidden_operations"], list) or set(config["forbidden_operations"]) != expected_forbidden:
        raise CandidateError("forbidden-operation closure differs")
    truth = _expect_keys(
        config["runtime_truth"],
        {
            "project_rows_read",
            "sequences_read",
            "real_trajectories_read",
            "cuda_probe_calls",
            "gpu_runs",
            "model_constructions",
            "optimizer_steps",
            "checkpoint_reads",
            "checkpoint_writes",
            "runtime_output_files_written",
            "a6_pass_asserted",
            "l3_claim_established",
            "a7_unlocked",
            "contract_wide_general_time_inhomogeneous_exactness_established",
        },
        label="runtime truth",
    )
    for key, value in truth.items():
        if key.endswith("asserted") or key.endswith("established") or key == "a7_unlocked":
            _expect(value, False, label=f"runtime truth {key}")
        else:
            _expect(value, 0, label=f"runtime truth {key}")


def load_config(path: Path = PRODUCTION_CONFIG_PATH) -> dict[str, Any]:
    try:
        config = load_json(path.read_bytes(), label="DEC028 A6 piecewise time-inhomogeneous candidate")
    except OSError as exc:
        raise CandidateError(f"cannot read candidate config: {path}") from exc
    validate_config(config)
    if path.resolve() == PRODUCTION_CONFIG_PATH.resolve():
        authority = config["static_authority"]
        for relative, expected in (
            (authority["root_config_path"], authority["root_config_sha256"]),
            (authority["dec028_amendment_path"], authority["dec028_amendment_sha256"]),
        ):
            try:
                actual = sha256((REPOSITORY_ROOT / relative).read_bytes())
            except OSError as exc:
                raise CandidateError(f"cannot read bound static authority: {relative}") from exc
            if actual != expected:
                raise CandidateError("bound current-authority byte identity differs")
    return config


@dataclass(frozen=True)
class State:
    source: str
    current: str
    edited_positions: tuple[int, ...]
    remaining_budget: int
    terminal_cause: str | None = None

    @property
    def net_edit_count(self) -> int:
        return len(self.edited_positions)


@dataclass(frozen=True)
class RawAction:
    action_type: str
    alias_id: str
    position: int | None = None


@dataclass(frozen=True)
class Transition:
    next_state: State
    rate: float
    raw_alias_ids: tuple[str, ...]
    raw_rate_sum: float


@dataclass(frozen=True)
class Case:
    case_id: str
    source: str
    budget: int
    variant: int


@dataclass(frozen=True)
class Graph:
    root: State
    states: tuple[State, ...]
    state_index: Mapping[State, int]


@dataclass(frozen=True)
class Generator:
    graph: Graph
    outgoing: tuple[tuple[tuple[int, float], ...], ...]
    total_exit_rates: tuple[float, ...]


@dataclass(frozen=True)
class UniformizationResult:
    probabilities: tuple[float, ...]
    poisson_tail_bound: float
    retained_terms: int


@dataclass(frozen=True)
class CaseOutcome:
    certified_terminal_distribution: Mapping[State, float]
    rk4_terminal_distribution: Mapping[State, float]
    uniformization_error_bound: float
    terminal_mass_shortfall: float
    rk4_refinement_steps: int
    terminal_distribution_tv: float
    state_count: int
    maximum_exit_rate: float


def _alternate(base: str) -> str:
    if base == "A":
        return "C"
    if base == "C":
        return "A"
    raise CandidateError("synthetic source alphabet is not binary A/C")


def state_key(state: State) -> tuple[Any, ...]:
    return (state.source, state.current, state.edited_positions, state.remaining_budget, state.terminal_cause or "")


def validate_state(state: State, config: Mapping[str, Any]) -> None:
    if not state.source or len(state.source) != len(state.current):
        raise CandidateError("state source/current shape is invalid")
    if any(base not in {"A", "C"} for base in state.source + state.current):
        raise CandidateError("state base is outside synthetic alphabet")
    if state.edited_positions != tuple(sorted(state.edited_positions)) or len(set(state.edited_positions)) != len(state.edited_positions):
        raise CandidateError("edited position set is invalid")
    if state.remaining_budget < 0:
        raise CandidateError("remaining budget is negative")
    edited = set(state.edited_positions)
    if any(position < 0 or position >= len(state.source) for position in edited):
        raise CandidateError("edited position is out of bounds")
    for position, (source_base, current_base) in enumerate(zip(state.source, state.current)):
        if position in edited:
            if current_base != _alternate(source_base):
                raise CandidateError("edited base is not source-relative legal")
        elif current_base != source_base:
            raise CandidateError("current sequence is not source-anchored")
    if state.terminal_cause not in {None, *TERMINAL_CAUSES}:
        raise CandidateError("terminal cause is invalid")
    if state.terminal_cause == "BUDGET_EXHAUSTED" and state.remaining_budget != 0:
        raise CandidateError("budget-exhausted state has nonzero budget")


def initial_state(case: Case) -> State:
    return State(case.source, case.source, (), case.budget)


def _structural_terminal(state: State) -> State:
    if state.terminal_cause is not None:
        return state
    if state.remaining_budget == 0:
        return State(state.source, state.current, state.edited_positions, state.remaining_budget, "BUDGET_EXHAUSTED")
    return state


def legal_raw_actions(state: State, config: Mapping[str, Any]) -> tuple[RawAction, ...]:
    validate_state(state, config)
    state = _structural_terminal(state)
    if state.terminal_cause is not None:
        return ()
    aliases = ("ALIAS_PRIMARY", "ALIAS_SECONDARY")
    actions: list[RawAction] = []
    edited = set(state.edited_positions)
    for position in range(len(state.source)):
        if position not in edited:
            actions.extend(RawAction(EDIT, alias, position) for alias in aliases)
    actions.extend(RawAction(STOP, alias) for alias in aliases)
    return tuple(actions)


def apply_raw_action(state: State, action: RawAction, config: Mapping[str, Any]) -> State:
    validate_state(state, config)
    state = _structural_terminal(state)
    if state.terminal_cause is not None:
        raise CandidateError("terminal state has no legal action")
    if action.action_type == STOP:
        if action.alias_id not in {"ALIAS_PRIMARY", "ALIAS_SECONDARY"} or action.position is not None:
            raise CandidateError("raw STOP action is illegal")
        return State(state.source, state.current, state.edited_positions, state.remaining_budget, "EXPLICIT_STOP")
    if (
        action.action_type != EDIT
        or action.alias_id not in {"ALIAS_PRIMARY", "ALIAS_SECONDARY"}
        or action.position is None
        or action.position in state.edited_positions
        or action.position < 0
        or action.position >= len(state.source)
    ):
        raise CandidateError("raw edit action is illegal")
    current = list(state.current)
    current[action.position] = _alternate(state.source[action.position])
    next_state = State(
        state.source,
        "".join(current),
        tuple(sorted((*state.edited_positions, action.position))),
        state.remaining_budget - 1,
    )
    return _structural_terminal(next_state)


def _schedule_multiplier(schedule: Mapping[str, Any], action_type: str) -> float:
    key = "edit_multiplier" if action_type == EDIT else "stop_multiplier"
    return _positive(schedule[key], label=f"schedule {key}")


def _raw_rate(state: State, action: RawAction, case: Case, config: Mapping[str, Any], schedule: Mapping[str, Any]) -> float:
    transition = config["transition_contract"]
    support = float(transition["support_floor"])
    if action.action_type == EDIT:
        if action.position is None:
            raise CandidateError("edit action lacks a position")
        canonical = support + float(transition["edit_rate_scale"]) * (1.0 + 0.1 * ((case.variant + action.position) % 8))
    elif action.action_type == STOP:
        canonical = support + float(transition["stop_rate"]) * (1.0 + 0.05 * case.variant)
    else:
        raise CandidateError("raw action type is invalid")
    canonical *= _schedule_multiplier(schedule, action.action_type)
    shares = {"ALIAS_PRIMARY": 0.4, "ALIAS_SECONDARY": 0.6}
    try:
        rate = canonical * shares[action.alias_id]
    except KeyError as exc:
        raise CandidateError("raw alias is invalid") from exc
    if not math.isfinite(rate) or rate <= 0.0:
        raise CandidateError("legal raw action lost positive support")
    return rate


def canonical_transitions(state: State, case: Case, config: Mapping[str, Any], schedule: Mapping[str, Any]) -> tuple[Transition, ...]:
    state = _structural_terminal(state)
    if state.terminal_cause is not None:
        return ()
    grouped: dict[State, list[tuple[RawAction, float]]] = {}
    for action in legal_raw_actions(state, config):
        next_state = apply_raw_action(state, action, config)
        grouped.setdefault(next_state, []).append((action, _raw_rate(state, action, case, config, schedule)))
    transitions = tuple(
        Transition(
            next_state=next_state,
            rate=math.fsum(rate for _, rate in members),
            raw_alias_ids=tuple(sorted(action.alias_id for action, _ in members)),
            raw_rate_sum=math.fsum(rate for _, rate in members),
        )
        for next_state, members in sorted(grouped.items(), key=lambda item: state_key(item[0]))
    )
    if not transitions or not any(item.next_state.terminal_cause == "EXPLICIT_STOP" for item in transitions):
        raise CandidateError("active state lacks competing positive STOP transition")
    for transition in transitions:
        if transition.raw_alias_ids != ("ALIAS_PRIMARY", "ALIAS_SECONDARY"):
            raise CandidateError("canonical transition does not aggregate exactly two aliases")
        if not math.isclose(transition.rate, transition.raw_rate_sum, rel_tol=0.0, abs_tol=1e-15):
            raise CandidateError("canonical rate differs from raw alias sum")
    return transitions


def generate_cases(config: Mapping[str, Any]) -> tuple[Case, ...]:
    fixtures = config["synthetic_fixture_generator"]
    cases: list[Case] = []
    for length in fixtures["source_lengths"]:
        for budget in fixtures["budgets"]:
            for variant in range(fixtures["deterministic_variants"]):
                source = "".join("A" if (position * 3 + variant + length) % 2 == 0 else "C" for position in range(length))
                cases.append(Case(f"L{length}_B{budget}_V{variant}", source, budget, variant))
    if len(cases) != fixtures["expected_fixture_count"] or len({case.case_id for case in cases}) != len(cases):
        raise CandidateError("synthetic fixture closure differs")
    return tuple(cases)


def build_graph(case: Case, config: Mapping[str, Any]) -> Graph:
    root = _structural_terminal(initial_state(case))
    stack = [root]
    states: set[State] = set()
    structural_schedule = config["time_inhomogeneous_reference_contract"]["post_schedule_tail"]
    while stack:
        state = stack.pop()
        if state in states:
            continue
        validate_state(state, config)
        states.add(state)
        for transition in canonical_transitions(state, case, config, structural_schedule):
            stack.append(transition.next_state)
    ordered = tuple(sorted(states, key=state_key))
    if root not in states or not any(state.terminal_cause for state in ordered):
        raise CandidateError("structural graph did not reach a terminal state")
    return Graph(root=root, states=ordered, state_index={state: index for index, state in enumerate(ordered)})


def build_generator(graph: Graph, case: Case, config: Mapping[str, Any], schedule: Mapping[str, Any]) -> Generator:
    outgoing: list[tuple[tuple[int, float], ...]] = []
    exits: list[float] = []
    for state in graph.states:
        transitions = canonical_transitions(state, case, config, schedule)
        row = tuple((graph.state_index[item.next_state], item.rate) for item in transitions)
        total = math.fsum(rate for _, rate in row)
        if state.terminal_cause is not None:
            if row or total != 0.0:
                raise CandidateError("terminal state is not absorbing")
        elif not math.isfinite(total) or total <= 0.0:
            raise CandidateError("active state has invalid total exit rate")
        outgoing.append(row)
        exits.append(total)
    return Generator(graph, tuple(outgoing), tuple(exits))


def _probability_mass(vector: Sequence[float]) -> float:
    return math.fsum(vector)


def _validate_probability_vector(vector: Sequence[float], *, label: str, allow_subprobability: bool) -> None:
    if any(not math.isfinite(value) or value < -1e-13 for value in vector):
        raise CandidateError(f"{label} has invalid probability")
    mass = _probability_mass(vector)
    if mass > 1.0 + 1e-12 or (not allow_subprobability and not math.isclose(mass, 1.0, rel_tol=0.0, abs_tol=1e-10)):
        raise CandidateError(f"{label} probability mass is invalid")


def _uniformized_step(vector: Sequence[float], generator: Generator, uniformization_rate: float) -> tuple[float, ...]:
    out = [0.0] * len(vector)
    for index, mass in enumerate(vector):
        if mass == 0.0:
            continue
        exit_rate = generator.total_exit_rates[index]
        stay = 1.0 - exit_rate / uniformization_rate
        if stay < -1e-14:
            raise CandidateError("uniformization rate is below exit rate")
        out[index] += mass * max(0.0, stay)
        for successor, rate in generator.outgoing[index]:
            out[successor] += mass * rate / uniformization_rate
    _validate_probability_vector(out, label="uniformized step", allow_subprobability=True)
    return tuple(out)


def propagate_uniformization(
    probabilities: Sequence[float],
    generator: Generator,
    duration: float,
    tail_limit: float,
) -> UniformizationResult:
    _validate_probability_vector(probabilities, label="uniformization input", allow_subprobability=True)
    if duration <= 0.0:
        raise CandidateError("uniformization duration is nonpositive")
    uniformization_rate = max(generator.total_exit_rates)
    if not math.isfinite(uniformization_rate) or uniformization_rate <= 0.0:
        raise CandidateError("uniformization rate is invalid")
    mean = uniformization_rate * duration
    if mean > 20.0:
        raise CandidateError("synthetic schedule exceeds bounded uniformization regime")
    term = tuple(float(value) for value in probabilities)
    weight = math.exp(-mean)
    result = [weight * value for value in term]
    cumulative_weight = weight
    retained_terms = 0
    while max(0.0, 1.0 - cumulative_weight) > tail_limit:
        retained_terms += 1
        if retained_terms > 4096:
            raise CandidateError("uniformization did not reach its tail bound")
        term = _uniformized_step(term, generator, uniformization_rate)
        weight *= mean / retained_terms
        if not math.isfinite(weight) or weight < 0.0:
            raise CandidateError("uniformization Poisson weight is invalid")
        for index, value in enumerate(term):
            result[index] += weight * value
        cumulative_weight += weight
    tail_bound = max(0.0, 1.0 - cumulative_weight)
    _validate_probability_vector(result, label="uniformization result", allow_subprobability=True)
    return UniformizationResult(tuple(result), tail_bound, retained_terms)


def _generator_derivative(probabilities: Sequence[float], generator: Generator) -> tuple[float, ...]:
    _validate_probability_vector(probabilities, label="RK4 input", allow_subprobability=False)
    derivative = [0.0] * len(probabilities)
    for index, mass in enumerate(probabilities):
        if mass == 0.0:
            continue
        derivative[index] -= mass * generator.total_exit_rates[index]
        for successor, rate in generator.outgoing[index]:
            derivative[successor] += mass * rate
    if not math.isclose(math.fsum(derivative), 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise CandidateError("Kolmogorov derivative does not conserve probability")
    return tuple(derivative)


def _rk4_step(
    probabilities: Sequence[float],
    generator: Generator,
    step_size: float,
) -> tuple[float, ...]:
    def shifted(base: Sequence[float], derivative: Sequence[float], factor: float) -> tuple[float, ...]:
        return tuple(value + factor * delta for value, delta in zip(base, derivative))

    k1 = _generator_derivative(probabilities, generator)
    k2 = _generator_derivative(shifted(probabilities, k1, step_size / 2.0), generator)
    k3 = _generator_derivative(shifted(probabilities, k2, step_size / 2.0), generator)
    k4 = _generator_derivative(shifted(probabilities, k3, step_size), generator)
    result = tuple(
        value + step_size * (a + 2.0 * b + 2.0 * c + d) / 6.0
        for value, a, b, c, d in zip(probabilities, k1, k2, k3, k4)
    )
    _validate_probability_vector(result, label="RK4 step", allow_subprobability=False)
    return result


def _rk4_integrate(
    probabilities: Sequence[float],
    generator: Generator,
    duration: float,
    steps: int,
) -> tuple[float, ...]:
    result = tuple(probabilities)
    step_size = duration / steps
    for _ in range(steps):
        result = _rk4_step(result, generator, step_size)
    return result


def propagate_rk4_refined(
    probabilities: Sequence[float],
    generator: Generator,
    duration: float,
    tolerance: float,
) -> tuple[tuple[float, ...], int]:
    previous = _rk4_integrate(probabilities, generator, duration, 4)
    steps = 8
    while steps <= 2048:
        current = _rk4_integrate(probabilities, generator, duration, steps)
        difference = math.fsum(abs(left - right) for left, right in zip(previous, current))
        if difference <= tolerance:
            return current, steps
        previous = current
        steps *= 2
    raise CandidateError("RK4 refinement did not converge within the bounded synthetic regime")


def enumerate_tail_terminal_distribution(
    state: State,
    case: Case,
    config: Mapping[str, Any],
    tail_schedule: Mapping[str, Any],
) -> Mapping[State, float]:
    result: defaultdict[State, float] = defaultdict(float)

    def walk(current: State, weight: float) -> None:
        current = _structural_terminal(current)
        if current.terminal_cause is not None:
            result[current] += weight
            return
        transitions = canonical_transitions(current, case, config, tail_schedule)
        total = math.fsum(transition.rate for transition in transitions)
        if total <= 0.0 or not math.isfinite(total):
            raise CandidateError("tail path enumeration encountered invalid exit rate")
        for transition in transitions:
            walk(transition.next_state, weight * transition.rate / total)

    walk(state, 1.0)
    if not math.isclose(math.fsum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise CandidateError("complete tail path enumeration lost probability mass")
    return dict(result)


def tail_absorption_table(graph: Graph, case: Case, config: Mapping[str, Any]) -> Mapping[State, Mapping[State, float]]:
    """Compute every conditional tail absorption distribution once by DAG order."""

    tail_schedule = config["time_inhomogeneous_reference_contract"]["post_schedule_tail"]
    table: dict[State, Mapping[State, float]] = {}
    ordered = sorted(
        graph.states,
        key=lambda state: (state.net_edit_count, 1 if state.terminal_cause is not None else 0, state_key(state)),
        reverse=True,
    )
    for state in ordered:
        if state.terminal_cause is not None:
            table[state] = {state: 1.0}
            continue
        result: defaultdict[State, float] = defaultdict(float)
        transitions = canonical_transitions(state, case, config, tail_schedule)
        total = math.fsum(transition.rate for transition in transitions)
        if total <= 0.0 or not math.isfinite(total):
            raise CandidateError("tail DP encountered invalid exit rate")
        for transition in transitions:
            child = table.get(transition.next_state)
            if child is None:
                raise CandidateError("tail graph is not acyclic in reverse topological order")
            for terminal, probability in child.items():
                result[terminal] += transition.rate / total * probability
        if not math.isclose(math.fsum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise CandidateError("topological tail DP lost probability mass")
        table[state] = dict(result)
    root_enumerated = enumerate_tail_terminal_distribution(graph.root, case, config, tail_schedule)
    if total_variation(table[graph.root], root_enumerated) > 1e-12:
        raise CandidateError("topological tail DP differs from complete root-path enumeration")
    return table


def absorb_homogeneous_tail(
    probabilities: Sequence[float],
    graph: Graph,
    case: Case,
    config: Mapping[str, Any],
    table: Mapping[State, Mapping[State, float]] | None = None,
) -> Mapping[State, float]:
    _validate_probability_vector(probabilities, label="tail absorption input", allow_subprobability=True)
    if table is None:
        table = tail_absorption_table(graph, case, config)
    result: defaultdict[State, float] = defaultdict(float)
    for state, mass in zip(graph.states, probabilities):
        if mass == 0.0:
            continue
        if state.terminal_cause is not None:
            result[state] += mass
            continue
        for terminal, conditional_probability in table[state].items():
            result[terminal] += mass * conditional_probability
    _validate_probability_vector(tuple(result.values()), label="tail absorption result", allow_subprobability=True)
    return dict(result)


def total_variation(left: Mapping[State, float], right: Mapping[State, float]) -> float:
    return 0.5 * math.fsum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in set(left) | set(right))


def normalized_transition_probabilities(transitions: Sequence[Transition]) -> Mapping[State, float]:
    total = math.fsum(item.rate for item in transitions)
    if total <= 0.0:
        raise CandidateError("cannot normalize empty transition collection")
    return {item.next_state: item.rate / total for item in transitions}


def run_case(case: Case, config: Mapping[str, Any]) -> CaseOutcome:
    graph = build_graph(case, config)
    root_index = graph.state_index[graph.root]
    certified = [0.0] * len(graph.states)
    certified[root_index] = 1.0
    rk4 = tuple(certified)
    accumulated_uniformization_bound = 0.0
    max_exit_rate = 0.0
    maximum_rk4_steps = 0
    reference = config["time_inhomogeneous_reference_contract"]
    tail_limit = float(reference["uniformization_tail_probability_max"])
    rk4_tolerance = float(reference["independent_terminal_distribution_tv_max"]) / 20.0
    for segment in reference["finite_segments"]:
        duration = float(segment["end"]) - float(segment["start"])
        generator = build_generator(graph, case, config, segment)
        max_exit_rate = max(max_exit_rate, max(generator.total_exit_rates))
        uniformized = propagate_uniformization(certified, generator, duration, tail_limit)
        certified = list(uniformized.probabilities)
        accumulated_uniformization_bound += uniformized.poisson_tail_bound
        rk4, steps = propagate_rk4_refined(rk4, generator, duration, rk4_tolerance)
        maximum_rk4_steps = max(maximum_rk4_steps, steps)
    tail_table = tail_absorption_table(graph, case, config)
    certified_terminal = absorb_homogeneous_tail(certified, graph, case, config, tail_table)
    rk4_terminal = absorb_homogeneous_tail(rk4, graph, case, config, tail_table)
    shortfall = max(0.0, 1.0 - math.fsum(certified_terminal.values()))
    if shortfall > accumulated_uniformization_bound + 1e-12:
        raise CandidateError("terminal mass shortfall exceeds recorded uniformization bound")
    comparison_tv = total_variation(certified_terminal, rk4_terminal)
    if comparison_tv > float(reference["independent_terminal_distribution_tv_max"]):
        raise CandidateError("certified and independently refined terminal distributions disagree")
    return CaseOutcome(
        certified_terminal_distribution=certified_terminal,
        rk4_terminal_distribution=rk4_terminal,
        uniformization_error_bound=accumulated_uniformization_bound,
        terminal_mass_shortfall=shortfall,
        rk4_refinement_steps=maximum_rk4_steps,
        terminal_distribution_tv=comparison_tv,
        state_count=len(graph.states),
        maximum_exit_rate=max_exit_rate,
    )


def validate_all_cases(config: Mapping[str, Any]) -> Mapping[str, Any]:
    cases = generate_cases(config)
    outcomes = [run_case(case, config) for case in cases]
    reference = config["time_inhomogeneous_reference_contract"]
    return {
        "status": "G0_NONLEARNED_SYNTHETIC_PIECEWISE_TIME_INHOMOGENEOUS_REFERENCE_PASS_NOT_A6_PASS",
        "fixture_count": len(cases),
        "source_lengths": config["synthetic_fixture_generator"]["source_lengths"],
        "budgets": config["synthetic_fixture_generator"]["budgets"],
        "time_scope": reference["reference_scope"],
        "contract_wide_general_time_inhomogeneous_exactness": reference[
            "contract_wide_general_time_inhomogeneous_exactness"
        ],
        "physical_kinetics_claim": reference["physical_kinetics_claim"],
        "max_uniformization_error_bound": max(outcome.uniformization_error_bound for outcome in outcomes),
        "max_terminal_mass_shortfall": max(outcome.terminal_mass_shortfall for outcome in outcomes),
        "max_independent_terminal_distribution_tv": max(outcome.terminal_distribution_tv for outcome in outcomes),
        "max_rk4_refinement_steps": max(outcome.rk4_refinement_steps for outcome in outcomes),
        "max_graph_state_count": max(outcome.state_count for outcome in outcomes),
        "max_exit_rate": max(outcome.maximum_exit_rate for outcome in outcomes),
        "a6_evidence_status": "IN_PROGRESS",
        "l3_claim_status": "NOT_ESTABLISHED",
        "runtime_truth": config["runtime_truth"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true", help="run the in-memory synthetic reference checks")
    parser.parse_args(argv)
    config = load_config()
    print(json.dumps(validate_all_cases(config), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
