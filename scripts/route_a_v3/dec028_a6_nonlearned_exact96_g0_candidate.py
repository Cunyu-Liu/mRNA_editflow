#!/usr/bin/env python3
"""Non-authoritative DEC028 A6 exact-reference candidate for 96 synthetic DAGs.

The candidate is deliberately restricted to time-homogeneous synthetic jump
chains. It compares a topological DAG dynamic program with complete-path
enumeration, but does not claim general time-inhomogeneous exactness or perform
any data, CUDA, model, checkpoint, parameter-update, or runtime I/O.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_CONFIG_PATH = REPOSITORY_ROOT / "configs/route_a_v3_dec028_a6_nonlearned_exact96_g0_candidate_v1.json"
BEFORE_DECISION_IDS = (
    "V3-DEC-017", "V3-DEC-018", "V3-DEC-019", "V3-DEC-020", "V3-DEC-021",
    "V3-DEC-022", "V3-DEC-023", "V3-DEC-024", "V3-DEC-027",
)
FROZEN_COUNTS = {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547}
EDIT = "EDIT"
STOP = "STOP"


class CandidateError(RuntimeError):
    """The static candidate or a synthetic exact-reference invariant failed."""


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
    states: frozenset[State]
    outgoing: Mapping[State, tuple[Transition, ...]]
    initial_budget: int


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CandidateError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, CandidateError) as exc:
        raise CandidateError(f"{label} is not a unique-key JSON object") from exc
    if not isinstance(value, dict):
        raise CandidateError(f"{label} must be a JSON object")
    return value


def _expect(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected or type(actual) is not type(expected):
        raise CandidateError(f"{label} differs from the frozen candidate")


def _expect_keys(value: Any, keys: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise CandidateError(f"{label} key closure differs")
    return value


def _sha256(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise CandidateError("static authority hash is invalid")
    return value


def _positive(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise CandidateError(f"{label} must be finite and positive")
    return float(value)


def validate_config(config: Mapping[str, Any]) -> None:
    _expect(config.get("schema_version"), "route_a_v3_dec028_a6_nonlearned_exact96_g0_candidate.v1", label="schema")
    _expect(config.get("candidate_id"), "ROUTE_A_V3_DEC028_A6_NONLEARNED_EXACT96_G0_CANDIDATE_V1", label="candidate id")
    _expect(config.get("document_status"), "DRAFT_FOR_DISTINCT_REVIEW_NOT_ACTIVE_PROTOCOL", label="document status")
    _expect(config.get("authority_status"), "NON_AUTHORITATIVE_G0_NONLEARNED_SYNTHETIC_PREPARATION", label="authority status")
    _expect(config.get("activation_state"), "INACTIVE_NO_PRODUCTION_PUBLISHER", label="activation")
    authority = _expect_keys(
        config.get("static_authority"),
        {"root_config_path", "root_config_sha256", "dec028_amendment_path", "dec028_amendment_sha256", "effective_active_amendment_decision_ids", "pending_successor_decision_id", "current_qualified_counts", "scientific_claim_status"},
        label="static authority",
    )
    _expect(authority["root_config_path"], "configs/route_a_v3.yaml", label="root path")
    _expect(authority["dec028_amendment_path"], "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec028.yaml", label="amendment path")
    _sha256(authority["root_config_sha256"])
    _sha256(authority["dec028_amendment_sha256"])
    _expect(tuple(authority["effective_active_amendment_decision_ids"]), BEFORE_DECISION_IDS, label="active decision ids")
    _expect(authority["pending_successor_decision_id"], "V3-DEC-028", label="pending successor")
    _expect(authority["current_qualified_counts"], FROZEN_COUNTS, label="counts")
    _expect(authority["scientific_claim_status"], "NOT_ESTABLISHED", label="claim")

    fixtures = _expect_keys(
        config.get("synthetic_fixture_generator"),
        {"source_lengths", "budgets", "deterministic_variants_per_length_budget", "expected_fixture_count", "alphabet", "source_anchored", "repeated_position_edit_allowed", "revert_to_source_allowed"},
        label="fixture generator",
    )
    _expect(fixtures["source_lengths"], [2, 3, 4, 5], label="source lengths")
    _expect(fixtures["budgets"], [1, 3, 5], label="budgets")
    _expect(fixtures["deterministic_variants_per_length_budget"], 8, label="variants")
    _expect(fixtures["expected_fixture_count"], 96, label="fixture count")
    _expect(fixtures["alphabet"], ["A", "C"], label="alphabet")
    _expect(fixtures["source_anchored"], True, label="source anchoring")
    _expect(fixtures["repeated_position_edit_allowed"], False, label="repeated edit")
    _expect(fixtures["revert_to_source_allowed"], False, label="revert")
    if len(fixtures["source_lengths"]) * len(fixtures["budgets"]) * fixtures["deterministic_variants_per_length_budget"] != fixtures["expected_fixture_count"]:
        raise CandidateError("fixture generator cardinality is not 96")

    transition = _expect_keys(
        config.get("transition_contract"),
        {"hard_legality_before_rate_construction", "support_floor", "edit_rate_scale", "stop_rate", "stop_is_competing_positive_rate", "raw_aliases_per_canonical_transition", "raw_alias_shares", "aggregate_aliases_by", "terminal_causes", "budget_invariant"},
        label="transition contract",
    )
    _expect(transition["hard_legality_before_rate_construction"], True, label="hard legality")
    for key in ("support_floor", "edit_rate_scale", "stop_rate"):
        _positive(transition[key], label=key)
    _expect(transition["stop_is_competing_positive_rate"], True, label="stop rate")
    _expect(transition["raw_aliases_per_canonical_transition"], 2, label="raw alias count")
    _expect(transition["raw_alias_shares"], [0.4, 0.6], label="raw alias shares")
    _expect(transition["aggregate_aliases_by"], "FULL_NEXT_EXTENDED_STATE_BEFORE_NORMALIZATION", label="alias aggregation")
    _expect(transition["terminal_causes"], ["BUDGET_EXHAUSTED", "EXPLICIT_STOP"], label="terminal causes")
    _expect(transition["budget_invariant"], "remaining_budget_plus_net_edit_count_is_constant", label="budget invariant")

    reference = _expect_keys(
        config.get("reference_contract"),
        {"primary_reference", "independent_reference", "absolute_error_max", "time_scope", "general_time_inhomogeneous_exactness"},
        label="reference contract",
    )
    _expect(reference["primary_reference"], "TOPOLOGICAL_DAG_DYNAMIC_PROGRAMMING_TERMINAL_JUMP_DISTRIBUTION", label="primary reference")
    _expect(reference["independent_reference"], "COMPLETE_PATH_ENUMERATION_TERMINAL_JUMP_DISTRIBUTION", label="independent reference")
    _expect(reference["absolute_error_max"], 1e-12, label="reference tolerance")
    _expect(reference["time_scope"], "SYNTHETIC_TIME_HOMOGENEOUS_JUMP_CHAIN_EXACT_REFERENCE_ONLY", label="time scope")
    _expect(reference["general_time_inhomogeneous_exactness"], "NOT_ESTABLISHED_NOT_CLAIMED", label="general time state")

    forbidden = config.get("forbidden_operations")
    expected_forbidden = {"PROJECT_ROW_OR_SEQUENCE_READ", "REAL_TRAJECTORY_OR_MEMBERSHIP_INPUT", "CUDA_OR_GPU_TOUCH", "TORCH_MODEL_OR_DATALOADER", "OPTIMIZER_OR_PARAMETER_UPDATE", "CHECKPOINT_READ_OR_WRITE", "RUNTIME_OUTPUT_WRITE", "A6_PASS_OR_L3_CLAIM", "A7_UNLOCK", "SEALED_ACCESS"}
    if not isinstance(forbidden, list) or set(forbidden) != expected_forbidden:
        raise CandidateError("forbidden operation closure differs")
    truth = _expect_keys(
        config.get("runtime_truth"),
        {"project_rows_read", "sequences_read", "real_trajectories_read", "cuda_probe_calls", "gpu_runs", "model_constructions", "optimizer_steps", "checkpoint_reads", "checkpoint_writes", "runtime_output_files_written", "a6_pass_asserted", "l3_claim_established", "a7_unlocked", "general_time_inhomogeneous_exactness_established"},
        label="runtime truth",
    )
    for key, value in truth.items():
        if key.endswith("asserted") or key.endswith("established") or key == "a7_unlocked":
            _expect(value, False, label=f"runtime truth {key}")
        else:
            _expect(value, 0, label=f"runtime truth {key}")


def load_config(path: Path = PRODUCTION_CONFIG_PATH) -> dict[str, Any]:
    try:
        config = load_json(path.read_bytes(), label="DEC028 A6 exact96 candidate")
    except OSError as exc:
        raise CandidateError(f"cannot read exact96 candidate config: {path}") from exc
    validate_config(config)
    if path.resolve() == PRODUCTION_CONFIG_PATH.resolve():
        authority = config["static_authority"]
        for relative, expected in ((authority["root_config_path"], authority["root_config_sha256"]), (authority["dec028_amendment_path"], authority["dec028_amendment_sha256"])):
            try:
                actual = sha256((REPOSITORY_ROOT / relative).read_bytes())
            except OSError as exc:
                raise CandidateError(f"cannot read bound static authority: {relative}") from exc
            if actual != expected:
                raise CandidateError("bound current-authority byte identity differs")
    return config


def _alternate(base: str) -> str:
    if base == "A":
        return "C"
    if base == "C":
        return "A"
    raise CandidateError("source alphabet is not binary A/C")


def validate_state(state: State, config: Mapping[str, Any]) -> None:
    if state.terminal_cause not in {None, "BUDGET_EXHAUSTED", "EXPLICIT_STOP"}:
        raise CandidateError("terminal cause is invalid")
    if len(state.source) != len(state.current) or not state.source:
        raise CandidateError("state source/current shape is invalid")
    if tuple(sorted(state.edited_positions)) != state.edited_positions or len(set(state.edited_positions)) != len(state.edited_positions):
        raise CandidateError("edited position set is invalid")
    if state.remaining_budget < 0:
        raise CandidateError("remaining budget is negative")
    edited = set(state.edited_positions)
    for position, (source_base, current_base) in enumerate(zip(state.source, state.current)):
        if position in edited:
            if current_base != _alternate(source_base):
                raise CandidateError("edited base is not the one legal source-relative alternative")
        elif current_base != source_base:
            raise CandidateError("current sequence is not source-anchored")
    if state.terminal_cause == "BUDGET_EXHAUSTED" and state.remaining_budget != 0:
        raise CandidateError("budget-exhausted terminal has nonzero remaining budget")


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
        return State(state.source, state.current, state.edited_positions, state.remaining_budget, "EXPLICIT_STOP")
    if action.action_type != EDIT or action.position is None or action.position in state.edited_positions or action.position < 0 or action.position >= len(state.source):
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


def _raw_rate(state: State, action: RawAction, case: Case, config: Mapping[str, Any]) -> float:
    transition = config["transition_contract"]
    floor = float(transition["support_floor"])
    if action.action_type == STOP:
        canonical = floor + float(transition["stop_rate"]) * (1.0 + 0.05 * case.variant)
    else:
        canonical = floor + float(transition["edit_rate_scale"]) * (1.0 + 0.1 * ((case.variant + action.position) % 8))
    shares = {"ALIAS_PRIMARY": 0.4, "ALIAS_SECONDARY": 0.6}
    return canonical * shares[action.alias_id]


def aggregate_transitions(state: State, case: Case, config: Mapping[str, Any]) -> tuple[Transition, ...]:
    state = _structural_terminal(state)
    if state.terminal_cause is not None:
        return ()
    grouped: dict[State, list[tuple[RawAction, float]]] = {}
    for action in legal_raw_actions(state, config):
        next_state = apply_raw_action(state, action, config)
        rate = _raw_rate(state, action, case, config)
        if rate <= 0 or not math.isfinite(rate):
            raise CandidateError("legal raw action lost positive support")
        grouped.setdefault(next_state, []).append((action, rate))
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
        raise CandidateError("active state lacks competing STOP transition")
    for transition in transitions:
        if transition.rate <= float(config["transition_contract"]["support_floor"]):
            raise CandidateError("canonical transition does not retain support floor")
        if not math.isclose(transition.rate, transition.raw_rate_sum, rel_tol=0.0, abs_tol=1e-15):
            raise CandidateError("alias aggregation differs from raw rate sum")
        if len(transition.raw_alias_ids) != 2:
            raise CandidateError("canonical transition does not aggregate exactly two aliases")
    return transitions


def state_key(state: State) -> tuple[Any, ...]:
    return (state.net_edit_count, state.remaining_budget, state.source, state.current, state.edited_positions, state.terminal_cause or "")


def build_graph(case: Case, config: Mapping[str, Any]) -> Graph:
    root = initial_state(case)
    validate_state(root, config)
    pending = [root]
    states = {root}
    outgoing: dict[State, tuple[Transition, ...]] = {}
    while pending:
        state = pending.pop()
        state = _structural_terminal(state)
        states.add(state)
        if state.terminal_cause is not None:
            outgoing[state] = ()
            continue
        transitions = aggregate_transitions(state, case, config)
        outgoing[state] = transitions
        for transition in transitions:
            child = transition.next_state
            if child.terminal_cause is None and child.net_edit_count != state.net_edit_count + 1:
                raise CandidateError("nonterminal edit edge is not acyclic")
            if child.terminal_cause == "EXPLICIT_STOP" and child.net_edit_count != state.net_edit_count:
                raise CandidateError("STOP changed the edit set")
            if child not in states:
                states.add(child)
                pending.append(child)
    return Graph(root=root, states=frozenset(states), outgoing=outgoing, initial_budget=case.budget)


def _jump_probabilities(transitions: tuple[Transition, ...]) -> tuple[tuple[Transition, float], ...]:
    total = math.fsum(item.rate for item in transitions)
    if total <= 0 or not math.isfinite(total):
        raise CandidateError("active state has invalid exit rate")
    return tuple((item, item.rate / total) for item in transitions)


def terminal_distribution_dp(graph: Graph) -> dict[State, float]:
    mass: dict[State, float] = {graph.root: 1.0}
    active = sorted((state for state in graph.states if state.terminal_cause is None), key=state_key)
    for state in active:
        probability = mass.get(state, 0.0)
        if probability == 0.0:
            continue
        for transition, jump_probability in _jump_probabilities(graph.outgoing[state]):
            mass[transition.next_state] = mass.get(transition.next_state, 0.0) + probability * jump_probability
    terminal = {state: value for state, value in mass.items() if state.terminal_cause is not None and value != 0.0}
    if not terminal or not math.isclose(math.fsum(terminal.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise CandidateError("DP terminal distribution does not conserve mass")
    return terminal


def _enumerate_paths(graph: Graph, state: State) -> dict[State, float]:
    if state.terminal_cause is not None:
        return {state: 1.0}
    result: dict[State, float] = {}
    for transition, probability in _jump_probabilities(graph.outgoing[state]):
        for terminal, suffix in _enumerate_paths(graph, transition.next_state).items():
            result[terminal] = result.get(terminal, 0.0) + probability * suffix
    return result


def terminal_distribution_enumeration(graph: Graph) -> dict[State, float]:
    result = _enumerate_paths(graph, graph.root)
    if not result or not math.isclose(math.fsum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise CandidateError("enumerated terminal distribution does not conserve mass")
    return result


def generate_cases(config: Mapping[str, Any]) -> tuple[Case, ...]:
    fixtures = config["synthetic_fixture_generator"]
    cases: list[Case] = []
    for length in fixtures["source_lengths"]:
        for budget in fixtures["budgets"]:
            for variant in range(fixtures["deterministic_variants_per_length_budget"]):
                source = "".join("C" if (variant >> (position % 3)) & 1 else "A" for position in range(length))
                cases.append(Case(f"L{length}_B{budget}_V{variant}", source, budget, variant))
    if len(cases) != fixtures["expected_fixture_count"] or len({case.case_id for case in cases}) != len(cases):
        raise CandidateError("generated fixture closure is not exact96")
    return tuple(cases)


def validate_graph_invariants(graph: Graph, config: Mapping[str, Any]) -> dict[str, int]:
    support_violations = 0
    alias_violations = 0
    budget_violations = 0
    for state in graph.states:
        validate_state(state, config)
        if state.net_edit_count + state.remaining_budget != graph.initial_budget:
            budget_violations += 1
        for transition in graph.outgoing[state]:
            if transition.rate <= float(config["transition_contract"]["support_floor"]):
                support_violations += 1
            if not math.isclose(transition.rate, transition.raw_rate_sum, rel_tol=0.0, abs_tol=1e-15):
                alias_violations += 1
    if support_violations or alias_violations or budget_violations:
        raise CandidateError("graph invariant violation")
    return {"state_count": len(graph.states), "support_violations": 0, "alias_violations": 0, "budget_violations": 0}


def run_exact96_suite(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    max_error = 0.0
    state_count = 0
    terminal_cause_counts = {"BUDGET_EXHAUSTED": 0, "EXPLICIT_STOP": 0}
    for case in generate_cases(config):
        graph = build_graph(case, config)
        invariants = validate_graph_invariants(graph, config)
        state_count += invariants["state_count"]
        dp = terminal_distribution_dp(graph)
        enumeration = terminal_distribution_enumeration(graph)
        for terminal in set(dp) | set(enumeration):
            max_error = max(max_error, abs(dp.get(terminal, 0.0) - enumeration.get(terminal, 0.0)))
            terminal_cause_counts[terminal.terminal_cause or "EXPLICIT_STOP"] += 1
    if max_error > float(config["reference_contract"]["absolute_error_max"]):
        raise CandidateError("DP and independent enumeration differ beyond tolerance")
    return {
        "status": "G0_NONLEARNED_SYNTHETIC_EXACT96_REFERENCE_PASS_NOT_A6_PASS",
        "fixture_count": len(generate_cases(config)),
        "budgets": list(config["synthetic_fixture_generator"]["budgets"]),
        "source_lengths": list(config["synthetic_fixture_generator"]["source_lengths"]),
        "aggregate_state_count": state_count,
        "max_terminal_distribution_absolute_error": max_error,
        "terminal_cause_state_counts": terminal_cause_counts,
        "time_scope": config["reference_contract"]["time_scope"],
        "general_time_inhomogeneous_exactness": config["reference_contract"]["general_time_inhomogeneous_exactness"],
        "a6_evidence_status": "IN_PROGRESS",
        "l3_claim_status": "NOT_ESTABLISHED",
        "runtime_truth": copy.deepcopy(config["runtime_truth"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true", help="run the in-memory synthetic exact96 suite")
    args = parser.parse_args(argv)
    if not args.validate_only:
        parser.error("only --validate-only is available for this non-authoritative candidate")
    try:
        report = run_exact96_suite(load_config())
    except CandidateError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
