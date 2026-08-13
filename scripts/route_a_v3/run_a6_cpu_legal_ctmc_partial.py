#!/usr/bin/env python3
"""Run the scoped CPU-only A6 single-event Gillespie/replay partial."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence


CONFIG_REPO_PATH = "configs/route_a_v3_a6_cpu_legal_ctmc_partial_v1.json"
SCRIPT_REPO_PATH = "scripts/route_a_v3/run_a6_cpu_legal_ctmc_partial.py"
TEST_REPO_PATH = "tests/route_a_v3/test_a6_cpu_legal_ctmc_partial.py"
EXACT_IMPLEMENTATION_PATHS = (CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH)
PRODUCTION_CONFIG_PATH = Path(__file__).resolve().parents[2] / CONFIG_REPO_PATH
UNKNOWN = "UNKNOWN_NOT_ASSERTED"
FROZEN_BASE_COMMIT = "a28bf3c67cf538a8754fde8505635b2ef2c3d68b"
GOAL_SHA256 = "cbac4c3dcba8f1b8df95d8edad52d19e3c126d1c865d0cc423537c754cc90982"
ACTIVE_CONFIG_SHA256 = "c908ac57b7c9667398f616a0ccf7101b41451b80bf169e768131844d3b63a678"
OUTPUT_NAMES = (
    "A6_CPU_LEGAL_CTMC_PARTIAL_REPORT.json",
    "RUN_MANIFEST.json",
    "EVENT_LOG.jsonl",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class A6CTMCError(RuntimeError):
    pass


class ConfigError(A6CTMCError):
    pass


class AuthorityError(A6CTMCError):
    pass


class NumericalError(A6CTMCError):
    pass


class ReplayError(A6CTMCError):
    pass


class PublicationError(A6CTMCError):
    pass


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def json_line(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ConfigError(f"non-finite JSON number: {value}")


def load_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"invalid JSON: {label}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"JSON root is not an object: {label}")
    return value


def load_config(path: Path = PRODUCTION_CONFIG_PATH) -> dict[str, Any]:
    try:
        return load_json_bytes(path.read_bytes(), label=os.fspath(path))
    except OSError as exc:
        raise ConfigError(f"cannot read config: {path}") from exc


def _expect(observed: Any, expected: Any, *, label: str) -> None:
    if observed != expected:
        raise ConfigError(f"{label} differs: {observed!r} != {expected!r}")


def _binding_mode(config: Mapping[str, Any]) -> str:
    binding = config.get("implementation_binding")
    keys = {"status", "implementation_commit", "implementation_script_sha256", "implementation_test_sha256"}
    if not isinstance(binding, Mapping) or set(binding) != keys:
        raise ConfigError("implementation_binding must contain exactly four scalars")
    values = [binding[key] for key in keys]
    if all(value == UNKNOWN for value in values):
        return "UNKNOWN"
    if any(value == UNKNOWN for value in values):
        raise ConfigError("implementation_binding is partially known")
    if binding["status"] != "BOUND":
        raise ConfigError("bound implementation status is not BOUND")
    if not COMMIT_RE.fullmatch(str(binding["implementation_commit"])):
        raise ConfigError("implementation commit is not a full Git SHA")
    for key in ("implementation_script_sha256", "implementation_test_sha256"):
        if not SHA256_RE.fullmatch(str(binding[key])):
            raise ConfigError(f"{key} is not SHA-256")
    return "BOUND"


def candidate_i_config(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(config))
    normalized["implementation_binding"] = {
        "status": UNKNOWN,
        "implementation_commit": UNKNOWN,
        "implementation_script_sha256": UNKNOWN,
        "implementation_test_sha256": UNKNOWN,
    }
    return normalized


def validate_static_config(config: Mapping[str, Any]) -> None:
    root_keys = {
        "schema_version",
        "protocol_id",
        "contract_id",
        "phase_id",
        "run_scope",
        "production_python",
        "implementation_binding",
        "authority",
        "dependency_contract",
        "clock_contract",
        "sampler_contract",
        "acceptance",
        "status_contract",
        "publication",
    }
    if set(config) != root_keys:
        raise ConfigError("config root key set differs")
    _expect(config["schema_version"], "route_a_v3_a6_cpu_legal_ctmc_partial.v1", label="schema")
    _expect(config["protocol_id"], "ROUTE_A_V3_A6_CPU_LEGAL_CTMC_PARTIAL_V1", label="protocol")
    _expect(config["contract_id"], "mrna_xeditflow_route_a_v3", label="contract")
    _expect(config["phase_id"], "A6", label="phase")
    _expect(config["run_scope"], "SYNTHETIC_NONLEARNED_CPU_GILLESPIE_BASE_RECOVERY", label="scope")
    _expect(config["production_python"], "/home/cunyuliu/miniconda3/envs/editflow/bin/python", label="python")
    _binding_mode(config)

    authority = config["authority"]
    _expect(authority["goal_sha256"], GOAL_SHA256, label="goal SHA")
    _expect(authority["active_config_sha256"], ACTIVE_CONFIG_SHA256, label="active config SHA")
    _expect(authority["frozen_base_commit"], FROZEN_BASE_COMMIT, label="frozen base")
    _expect(authority["required_remote"], "origin", label="remote")
    dependencies = authority["dependency_leaves"]
    expected_dependencies = {
        "configs/route_a_v3_a6_cpu_exact_absorbing_dag_v1.json": "84e9a3f21ac6293faa167eb08eb40e8886bfe43daaa374b2c7613fbc9baecab8",
        "scripts/route_a_v3/run_a6_cpu_exact_absorbing_dag.py": "4cc0e10784c218cf4fa18ede1280cb84e1b7daf2553db4d82e0ee88f71e0f7c8",
    }
    _expect(dependencies, expected_dependencies, label="dependency leaves")

    dependency = config["dependency_contract"]
    _expect(dependency["kernel_config_path"], next(iter(expected_dependencies)), label="kernel config path")
    _expect(
        dependency["kernel_script_path"],
        "scripts/route_a_v3/run_a6_cpu_exact_absorbing_dag.py",
        label="kernel script path",
    )
    _expect(dependency["kernel_protocol_id"], "ROUTE_A_V3_A6_CPU_EXACT_ABSORBING_DAG_V1", label="kernel")
    _expect(dependency["kernel_case_id"], "L2_B2", label="kernel case")

    clock = config["clock_contract"]
    _expect(clock["clock_semantics"], "CONTINUOUS_ALGORITHMIC_TIME", label="clock")
    _expect(clock["rate_time_dependence"], "NONE", label="rate time dependence")
    _expect(clock["terminal_tilt_time_dependence"], "NONE", label="tilt time dependence")
    _expect(clock["holding_time_law"], "EXPONENTIAL_WITH_CURRENT_TOTAL_EXIT_RATE", label="holding law")
    _expect(clock["general_time_inhomogeneous_exactness"], "NOT_RUN", label="general time scope")

    sampler = config["sampler_contract"]
    _expect(sampler["algorithm"], "SINGLE_EVENT_GILLESPIE_DIRECT_METHOD", label="algorithm")
    _expect(
        sampler["raw_alias_aggregation_order"],
        "BEFORE_EXIT_RATE_NORMALIZATION_AND_SAMPLING",
        label="alias order",
    )
    for key in ("stop_competes_as_positive_rate_jump", "source_relative_no_reedit_no_revert"):
        _expect(sampler[key], True, label=key)
    if sampler["trajectory_count"] < 10000 or sampler["replay_trajectory_count"] < 1:
        raise ConfigError("sampling denominators are too small")
    _expect(sampler["maximum_jumps"], 3, label="maximum jumps")

    acceptance = config["acceptance"]
    if not 0.0 < float(acceptance["terminal_distribution_tv_max"]) <= 0.05:
        raise ConfigError("terminal TV threshold is outside frozen range")
    if not 0.0 < float(acceptance["initial_holding_time_mean_relative_error_max"]) <= 0.05:
        raise ConfigError("holding-time threshold is outside frozen range")
    for key, value in acceptance.items():
        if key.endswith("_count_max") and value != 0:
            raise ConfigError(f"{key} must remain zero")

    status = config["status_contract"]
    expected_status = {
        "run_status_on_success": "PASS",
        "a6_evidence_status": "IN_PROGRESS",
        "a6_phase_complete": False,
        "a6_pass_asserted": False,
        "flow_base_legal_ctmc_evidence_status": "IN_PROGRESS",
        "flow_base_legal_ctmc_result": "DEVELOPMENT_CPU_NONLEARNED_GILLESPIE_REPLAY_PARTIAL_PASS",
        "flow_base_legal_ctmc_scope": "SYNTHETIC_NONLEARNED_CPU_GILLESPIE_BASE_RECOVERY",
        "formal_flow_base_task_pass_asserted": False,
        "exact_guidance_toy_graph_evidence_status": "PASS",
        "l3_evidence_status": "IN_PROGRESS",
        "l3_claim_status": "NOT_ESTABLISHED",
        "learned_potential_approximation_error": "NOT_RUN",
        "ordinary_data_evidence": "NOT_RUN",
        "a7_evidence_status": "NOT_RUN",
        "a7_unlock": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "gpu_work_allowed": False,
        "private_payload_access_allowed": False,
        "sealed_contact_allowed": False,
    }
    _expect(status, expected_status, label="status contract")

    publication = config["publication"]
    _expect(publication["output_names"], list(OUTPUT_NAMES), label="output names")
    for key in ("exclusive_new_directory_required", "atomic_directory_publish_required", "public_aggregate_only"):
        _expect(publication[key], True, label=f"publication.{key}")
    for key in ("row_data_allowed", "trajectory_data_allowed", "ordinary_private_or_sealed_data_allowed"):
        _expect(publication[key], False, label=f"publication.{key}")


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("route_a_v3_a6_frozen_exact_kernel", path)
    if spec is None or spec.loader is None:
        raise ConfigError("cannot load frozen exact kernel")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_kernel(repo_root: Path, config: Mapping[str, Any]) -> tuple[ModuleType, dict[str, Any]]:
    dependency = config["dependency_contract"]
    hashes = config["authority"]["dependency_leaves"]
    config_path = repo_root / dependency["kernel_config_path"]
    script_path = repo_root / dependency["kernel_script_path"]
    for path in (config_path, script_path):
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ConfigError(f"frozen dependency is unavailable: {path}") from exc
        relative = path.relative_to(repo_root).as_posix()
        if sha256(payload) != hashes[relative]:
            raise ConfigError(f"frozen dependency SHA differs: {relative}")
    kernel = _load_module(script_path)
    kernel_config = kernel.load_config(config_path)
    kernel.validate_static_config(kernel_config)
    _expect(kernel_config["protocol_id"], dependency["kernel_protocol_id"], label="loaded kernel protocol")
    return kernel, kernel_config


def _open_unit(rng: random.Random) -> float:
    return (rng.getrandbits(53) + 1) / ((1 << 53) + 1)


def _state_key(state: Any, *, include_time: bool) -> tuple[Any, ...]:
    parts: tuple[Any, ...] = (
        state.source_sequence,
        state.current_sequence,
        tuple(state.source_relative_edit_set),
        state.remaining_budget,
        state.assay,
        state.context,
    )
    if include_time:
        parts += (state.algorithmic_time,)
    return parts + (state.terminal_cause,)


def _terminal_key(state: Any) -> tuple[Any, ...]:
    return _state_key(state, include_time=False)


def _canonical_alias_pairs(kernel: ModuleType, state: Any, selected: Any, kernel_config: Mapping[str, Any]) -> tuple[tuple[str, float], ...]:
    pairs: list[tuple[str, float]] = []
    for action in kernel.raw_actions(state, kernel_config):
        rate = kernel.raw_action_rate(state, action, kernel_config)
        child = kernel.transition_state(state, action, kernel_config)
        if child == selected.next_state:
            pairs.append((action.alias_id, rate))
    return tuple(sorted(pairs))


@dataclass(frozen=True)
class TraceEvent:
    before: Any
    after: Any
    survival_uniform: float
    jump_uniform: float
    holding_time: float
    total_exit_rate: float
    selected_rate: float
    selected_action_type: str
    alias_pairs: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class Trajectory:
    initial: Any
    final: Any
    events: tuple[TraceEvent, ...]


def _choose_transition(transitions: Sequence[Any], total: float, jump_uniform: float) -> Any:
    if not 0.0 < jump_uniform < 1.0:
        raise NumericalError("jump uniform is not strictly inside (0,1)")
    threshold = jump_uniform * total
    cumulative = 0.0
    for transition in transitions:
        cumulative += transition.rate
        if threshold < cumulative:
            return transition
    return transitions[-1]


def gillespie_step(
    kernel: ModuleType,
    kernel_config: Mapping[str, Any],
    state: Any,
    *,
    survival_uniform: float,
    jump_uniform: float,
) -> TraceEvent:
    kernel.validate_state(state, kernel_config)
    if state.terminal_cause is not None:
        raise NumericalError("cannot step an absorbing state")
    if not 0.0 < survival_uniform < 1.0:
        raise NumericalError("survival uniform is not strictly inside (0,1)")
    transitions = kernel.canonical_transitions(state, kernel_config)
    total = math.fsum(item.rate for item in transitions)
    if not math.isfinite(total) or total <= 0.0:
        raise NumericalError("active state total exit rate is not finite and positive")
    selected = _choose_transition(transitions, total, jump_uniform)
    holding = -math.log(survival_uniform) / total
    if not math.isfinite(holding) or holding <= 0.0:
        raise NumericalError("holding time is not finite and positive")
    after = replace(selected.next_state, algorithmic_time=state.algorithmic_time + holding)
    kernel.validate_state(after, kernel_config)
    pairs = _canonical_alias_pairs(kernel, state, selected, kernel_config)
    if not pairs or not math.isclose(math.fsum(rate for _, rate in pairs), selected.rate, rel_tol=1e-14, abs_tol=1e-14):
        raise NumericalError("selected canonical rate does not equal paired raw-alias sum")
    return TraceEvent(
        before=state,
        after=after,
        survival_uniform=survival_uniform,
        jump_uniform=jump_uniform,
        holding_time=holding,
        total_exit_rate=total,
        selected_rate=selected.rate,
        selected_action_type=selected.action_type,
        alias_pairs=pairs,
    )


def sample_trajectory(
    kernel: ModuleType,
    kernel_config: Mapping[str, Any],
    initial: Any,
    rng: random.Random,
    *,
    maximum_jumps: int,
) -> Trajectory:
    state = initial
    events: list[TraceEvent] = []
    while state.terminal_cause is None:
        if len(events) >= maximum_jumps:
            raise NumericalError("trajectory exceeded the frozen maximum jump count")
        event = gillespie_step(
            kernel,
            kernel_config,
            state,
            survival_uniform=_open_unit(rng),
            jump_uniform=_open_unit(rng),
        )
        events.append(event)
        state = event.after
    return Trajectory(initial=initial, final=state, events=tuple(events))


def replay_trajectory(
    kernel: ModuleType,
    kernel_config: Mapping[str, Any],
    trajectory: Trajectory,
) -> Trajectory:
    state = trajectory.initial
    replayed: list[TraceEvent] = []
    for expected in trajectory.events:
        observed = gillespie_step(
            kernel,
            kernel_config,
            state,
            survival_uniform=expected.survival_uniform,
            jump_uniform=expected.jump_uniform,
        )
        if observed != expected:
            raise ReplayError("deterministic trajectory replay differs")
        replayed.append(observed)
        state = observed.after
    if state != trajectory.final or state.terminal_cause is None:
        raise ReplayError("trajectory replay did not reach the same terminal state")
    return Trajectory(trajectory.initial, state, tuple(replayed))


def _tv(left: Mapping[Any, float], right: Mapping[Any, float]) -> float:
    return 0.5 * math.fsum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in set(left) | set(right))


def _relative_error(observed: float, expected: float) -> float:
    if expected == 0.0:
        return abs(observed)
    return abs(observed - expected) / abs(expected)


def _trajectory_violation_counts(kernel: ModuleType, kernel_config: Mapping[str, Any], trajectory: Trajectory) -> Counter[str]:
    counts: Counter[str] = Counter()
    initial_budget = trajectory.initial.remaining_budget + trajectory.initial.net_edit_count
    previous_positions = {position for position, _ in trajectory.initial.source_relative_edit_set}
    for event in trajectory.events:
        try:
            kernel.validate_state(event.before, kernel_config)
            kernel.validate_state(event.after, kernel_config)
        except Exception:
            counts["source_reconstruction"] += 1
        if event.after.algorithmic_time <= event.before.algorithmic_time:
            counts["time_monotonicity"] += 1
        if event.after.remaining_budget + event.after.net_edit_count != initial_budget:
            counts["budget"] += 1
        after_positions = {position for position, _ in event.after.source_relative_edit_set}
        if not previous_positions.issubset(after_positions) or len(after_positions - previous_positions) > 1:
            counts["reedit_or_revert"] += 1
        if event.selected_action_type == kernel.STOP:
            if after_positions != previous_positions or event.after.remaining_budget != event.before.remaining_budget:
                counts["legality"] += 1
        else:
            if len(after_positions - previous_positions) != 1 or event.after.remaining_budget != event.before.remaining_budget - 1:
                counts["legality"] += 1
        if not math.isclose(math.fsum(rate for _, rate in event.alias_pairs), event.selected_rate, rel_tol=1e-14, abs_tol=1e-14):
            counts["alias_aggregation"] += 1
        previous_positions = after_positions
    return counts


def run_sampling_suite(config: Mapping[str, Any], *, repo_root: Path) -> dict[str, Any]:
    validate_static_config(config)
    kernel, kernel_config = load_kernel(repo_root, config)
    case_id = config["dependency_contract"]["kernel_case_id"]
    case = next(item for item in kernel_config["fixed_cases"] if item["case_id"] == case_id)
    initial = kernel.initial_state(case, kernel_config)
    if initial.terminal_cause is not None:
        raise NumericalError("frozen sampler root is not transient")

    graph = kernel.build_graph(initial, kernel_config)
    oracle_raw = kernel.terminal_distribution_dp(graph)
    oracle: Counter[Any] = Counter()
    for state, probability in oracle_raw.items():
        oracle[_terminal_key(state)] += probability
    oracle_total = math.fsum(oracle.values())
    if not math.isclose(oracle_total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise NumericalError("DP terminal oracle is not normalized")

    sampler = config["sampler_contract"]
    count = int(sampler["trajectory_count"])
    replay_count = int(sampler["replay_trajectory_count"])
    maximum_jumps = int(sampler["maximum_jumps"])
    rng = random.Random(int(sampler["random_seed"]))
    terminals: Counter[Any] = Counter()
    causes: Counter[str] = Counter()
    violations: Counter[str] = Counter()
    first_holding_times: list[float] = []
    jump_counts: Counter[int] = Counter()
    edit_counts: Counter[int] = Counter()
    total_jumps = 0
    total_source_relative_edits = 0
    replay_mismatches = 0
    started = time.perf_counter()
    for index in range(count):
        trajectory = sample_trajectory(kernel, kernel_config, initial, rng, maximum_jumps=maximum_jumps)
        terminals[_terminal_key(trajectory.final)] += 1
        causes[trajectory.final.terminal_cause] += 1
        jump_counts[len(trajectory.events)] += 1
        edit_counts[trajectory.final.net_edit_count] += 1
        total_jumps += len(trajectory.events)
        total_source_relative_edits += trajectory.final.net_edit_count
        first_holding_times.append(trajectory.events[0].holding_time)
        violations.update(_trajectory_violation_counts(kernel, kernel_config, trajectory))
        if index < replay_count:
            try:
                replay_trajectory(kernel, kernel_config, trajectory)
            except ReplayError:
                replay_mismatches += 1
    wall_clock_seconds = time.perf_counter() - started

    empirical = {key: value / count for key, value in terminals.items()}
    terminal_tv = _tv(empirical, oracle)
    initial_total = math.fsum(item.rate for item in kernel.canonical_transitions(initial, kernel_config))
    expected_holding_mean = 1.0 / initial_total
    observed_holding_mean = math.fsum(first_holding_times) / len(first_holding_times)
    holding_error = _relative_error(observed_holding_mean, expected_holding_mean)
    acceptance = config["acceptance"]
    metrics = {
        "terminal_distribution_tv": terminal_tv,
        "initial_total_exit_rate": initial_total,
        "expected_initial_holding_time_mean": expected_holding_mean,
        "observed_initial_holding_time_mean": observed_holding_mean,
        "initial_holding_time_mean_relative_error": holding_error,
        "replay_mismatch_count": replay_mismatches,
        "legality_violation_count": violations["legality"],
        "source_reconstruction_violation_count": violations["source_reconstruction"],
        "budget_violation_count": violations["budget"],
        "reedit_or_revert_violation_count": violations["reedit_or_revert"],
        "algorithmic_time_monotonicity_violation_count": violations["time_monotonicity"],
        "alias_aggregation_violation_count": violations["alias_aggregation"],
        "numerical_failure_count": 0,
    }
    failures = []
    if terminal_tv > float(acceptance["terminal_distribution_tv_max"]):
        failures.append("TERMINAL_DISTRIBUTION_TV")
    if holding_error > float(acceptance["initial_holding_time_mean_relative_error_max"]):
        failures.append("INITIAL_HOLDING_TIME_MEAN")
    count_mapping = {
        "replay_mismatch_count": "replay_mismatch_count_max",
        "legality_violation_count": "legality_violation_count_max",
        "source_reconstruction_violation_count": "source_reconstruction_violation_count_max",
        "budget_violation_count": "budget_violation_count_max",
        "reedit_or_revert_violation_count": "reedit_or_revert_violation_count_max",
        "algorithmic_time_monotonicity_violation_count": "algorithmic_time_monotonicity_violation_count_max",
        "alias_aggregation_violation_count": "alias_aggregation_violation_count_max",
        "numerical_failure_count": "numerical_failure_count_max",
    }
    for metric, threshold in count_mapping.items():
        if metrics[metric] > int(acceptance[threshold]):
            failures.append(metric.upper())
    if failures:
        raise NumericalError("sampling acceptance failed: " + ",".join(failures))

    return {
        "status": "PASS",
        "kernel_case_id": case_id,
        "trajectory_count": count,
        "replay_trajectory_count": replay_count,
        "total_jump_count": total_jumps,
        "total_source_relative_edit_count": total_source_relative_edits,
        "terminal_cause_counts": {cause: causes[cause] for cause in kernel.TERMINAL_CAUSES},
        "jump_count_histogram": {str(key): jump_counts[key] for key in sorted(jump_counts)},
        "source_relative_edit_count_histogram": {str(key): edit_counts[key] for key in sorted(edit_counts)},
        "metrics": metrics,
        "acceptance": dict(acceptance),
        "checks": {
            "SINGLE_EVENT_GILLESPIE_DIRECT_METHOD": "PASS",
            "EXPONENTIAL_HOLDING_TIME_BASE_RECOVERY": "PASS",
            "DETERMINISTIC_TRAJECTORY_REPLAY": "PASS",
            "TERMINAL_LAW_VS_EXACT_DP": "PASS",
            "HARD_LEGALITY_BEFORE_RATE_EVALUATION": "PASS",
            "SOURCE_RELATIVE_NO_REEDIT_NO_REVERT": "PASS",
            "STOP_AS_COMPETING_POSITIVE_RATE_JUMP": "PASS",
            "FULL_NEXT_STATE_ALIAS_AGGREGATION": "PASS",
            "BUDGET_AND_TIME_INVARIANTS": "PASS",
        },
        "compute_ledger": {
            "device": "CPU",
            "learned_parameter_count": 0,
            "parameter_update_count": 0,
            "trajectory_count": count,
            "total_jump_count": total_jumps,
            "total_source_relative_edit_count": total_source_relative_edits,
            "wall_clock_seconds": wall_clock_seconds,
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
        raise AuthorityError(completed.stderr.decode("utf-8", errors="replace").strip())
    return completed.stdout


def _git_text(repo: Path, *args: str) -> str:
    return _git_bytes(repo, *args).decode("utf-8").strip()


def _changed_paths(repo: Path, commit: str) -> list[str]:
    return sorted(
        line
        for line in _git_text(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", commit).splitlines()
        if line
    )


def _git_config(repo: Path, commit: str) -> dict[str, Any]:
    return load_json_bytes(_git_bytes(repo, "show", f"{commit}:{CONFIG_REPO_PATH}"), label=f"{commit}:config")


def _validate_active_authority(repo: Path, config: Mapping[str, Any]) -> None:
    authority = config["authority"]
    for path_key, sha_key in (("goal_path", "goal_sha256"), ("active_config_path", "active_config_sha256")):
        path = repo / authority[path_key]
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise AuthorityError(f"authority leaf unavailable: {path}") from exc
        if sha256(payload) != authority[sha_key]:
            raise AuthorityError(f"authority leaf drift: {authority[path_key]}")
        text = payload.decode("utf-8")
        literals = authority["goal_critical_literals" if path_key == "goal_path" else "active_config_critical_literals"]
        if any(literal not in text for literal in literals):
            raise AuthorityError(f"authority critical literal drift: {authority[path_key]}")


def validate_production_authority(config: Mapping[str, Any], *, repo_root: Path | None = None) -> dict[str, str]:
    validate_static_config(config)
    if _binding_mode(config) != "BOUND":
        raise AuthorityError("production implementation binding is not BOUND")
    repo = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    if Path(_git_text(repo, "rev-parse", "--show-toplevel")).resolve() != repo:
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
    if head != upstream or head != origin or upstream_ref != f"refs/remotes/{remote}/{branch}":
        raise AuthorityError("production HEAD, upstream, and origin tracking ref differ")

    implementation = config["implementation_binding"]["implementation_commit"]
    if _git_text(repo, "rev-parse", f"{head}^") != implementation:
        raise AuthorityError("binding B is not the direct child of implementation I")
    if _git_text(repo, "rev-parse", f"{implementation}^") != FROZEN_BASE_COMMIT:
        raise AuthorityError("implementation I is not the direct child of the frozen base")
    if _changed_paths(repo, implementation) != sorted(EXACT_IMPLEMENTATION_PATHS):
        raise AuthorityError("implementation I changed-path set is not exact3")
    if _changed_paths(repo, head) != [CONFIG_REPO_PATH]:
        raise AuthorityError("binding B changed-path set is not config-only")

    i_config = _git_config(repo, implementation)
    b_config = _git_config(repo, head)
    validate_static_config(i_config)
    validate_static_config(b_config)
    if _binding_mode(i_config) != "UNKNOWN" or candidate_i_config(b_config) != i_config:
        raise AuthorityError("I/B config lifecycle differs outside four binding scalars")
    if b_config != dict(config) or b_config["implementation_binding"]["implementation_commit"] != implementation:
        raise AuthorityError("worktree binding config does not name its implementation I")
    if load_config(repo / CONFIG_REPO_PATH) != b_config:
        raise AuthorityError("worktree config differs from binding B")

    for path, key in (
        (SCRIPT_REPO_PATH, "implementation_script_sha256"),
        (TEST_REPO_PATH, "implementation_test_sha256"),
    ):
        expected = config["implementation_binding"][key]
        payloads = (
            _git_bytes(repo, "show", f"{implementation}:{path}"),
            _git_bytes(repo, "show", f"{head}:{path}"),
            (repo / path).read_bytes(),
        )
        if any(sha256(payload) != expected for payload in payloads):
            raise AuthorityError(f"bound implementation leaf drift: {path}")

    for path, expected in config["authority"]["dependency_leaves"].items():
        payloads = (
            _git_bytes(repo, "show", f"{FROZEN_BASE_COMMIT}:{path}"),
            _git_bytes(repo, "show", f"{implementation}:{path}"),
            _git_bytes(repo, "show", f"{head}:{path}"),
            (repo / path).read_bytes(),
        )
        if any(sha256(payload) != expected for payload in payloads):
            raise AuthorityError(f"frozen kernel dependency drift: {path}")
    _validate_active_authority(repo, config)
    return {
        "head": head,
        "binding_commit": head,
        "implementation_commit": implementation,
        "branch": branch,
        "upstream_ref": upstream_ref,
    }


def _recorded_at(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ConfigError("recorded_at is not ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ConfigError("recorded_at lacks timezone")
    return value


def build_public_payloads(
    config: Mapping[str, Any],
    suite: Mapping[str, Any],
    *,
    recorded_at: str,
    authority: Mapping[str, str] | None,
) -> dict[str, bytes]:
    recorded_at = _recorded_at(recorded_at)
    status = config["status_contract"]
    boundaries = {
        "development_only": True,
        "a6_phase_complete": False,
        "a6_pass_asserted": False,
        "formal_flow_base_task_pass_asserted": False,
        "l3_claim_status": "NOT_ESTABLISHED",
        "general_time_inhomogeneous_exactness": "NOT_RUN",
        "learned_potential_approximation_error": "NOT_RUN",
        "ordinary_data_evidence": "NOT_RUN",
        "a7_evidence_status": "NOT_RUN",
        "a7_unlock": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "gpu_work_allowed": False,
        "ordinary_row_read_count": 0,
        "private_payload_read_count": 0,
        "sealed_contact_count": 0,
    }
    report = {
        "schema_version": "route_a_v3_a6_cpu_legal_ctmc_partial_report.v1",
        "protocol_id": config["protocol_id"],
        "recorded_at": recorded_at,
        "run_scope": config["run_scope"],
        "run_status": "PASS",
        "phase_state": {"phase_id": "A6", "evidence_status": "IN_PROGRESS", "phase_complete": False},
        "task_states": {
            "EXACT_GUIDANCE_TOY_GRAPH": {"evidence_status": status["exact_guidance_toy_graph_evidence_status"]},
            "FLOW_BASE_LEGAL_CTMC": {
                "evidence_status": status["flow_base_legal_ctmc_evidence_status"],
                "result": status["flow_base_legal_ctmc_result"],
                "scope": status["flow_base_legal_ctmc_scope"],
            },
        },
        "claim_state": {
            "claim_id": "L3_LEGAL_POTENTIAL_CONSISTENT_XEDITFLOW",
            "evidence_status": "IN_PROGRESS",
            "claim_status": "NOT_ESTABLISHED",
        },
        "sampler_contract": dict(config["sampler_contract"]),
        "clock_contract": dict(config["clock_contract"]),
        "sampling_suite": dict(suite),
        "boundaries": boundaries,
    }
    manifest = {
        "schema_version": "route_a_v3_a6_cpu_legal_ctmc_partial_run_manifest.v1",
        "protocol_id": config["protocol_id"],
        "recorded_at": recorded_at,
        "run_status": "PASS",
        "authority": dict(authority) if authority is not None else {"mode": "SYNTHETIC_TEST_NO_GIT"},
        "cpu_only": True,
        "learned_parameter_count": 0,
        "parameter_update_count": 0,
        "output_count": 3,
        "outputs": [
            {"name": OUTPUT_NAMES[0], "artifact_type": "PUBLIC_AGGREGATE_REPORT"},
            {"name": OUTPUT_NAMES[1], "artifact_type": "PUBLIC_AGGREGATE_RUN_MANIFEST"},
            {"name": OUTPUT_NAMES[2], "artifact_type": "PUBLIC_AGGREGATE_EVENT_LOG"},
        ],
        "flow_base_legal_ctmc_evidence_status": "IN_PROGRESS",
        "l3_claim_status": "NOT_ESTABLISHED",
        "boundaries": boundaries,
    }
    event = {
        "event_id": "A6-CPU-LEGAL-CTMC-001",
        "at": recorded_at,
        "event": "A6_CPU_NONLEARNED_GILLESPIE_REPLAY_PARTIAL_COMPLETED",
        "run_status": "PASS",
        "a6_evidence_status": "IN_PROGRESS",
        "flow_base_legal_ctmc_evidence_status": "IN_PROGRESS",
        "flow_base_legal_ctmc_result": status["flow_base_legal_ctmc_result"],
        "l3_claim_status": "NOT_ESTABLISHED",
        "a7_unlock": False,
        "training_allowed": False,
        "gpu_work_allowed": False,
        "trajectory_count": suite["trajectory_count"],
        "aggregate_metrics": suite["metrics"],
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
            raise PublicationError("production output path differs from the frozen exclusive target")
    if output.exists():
        raise PublicationError("exclusive output directory already exists")
    return output


def publish_exact_three(output_directory: Path, payloads: Mapping[str, bytes]) -> None:
    if set(payloads) != set(OUTPUT_NAMES) or len(payloads) != 3:
        raise PublicationError("publication payload is not exact3")
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
            raise PublicationError("temporary publication is not exact3")
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
    repo_root: Path | None = None,
) -> dict[str, Any]:
    config = dict(config_override) if config_override is not None else load_config()
    validate_static_config(config)
    if production and _binding_mode(config) != "BOUND":
        raise AuthorityError("production implementation binding is not BOUND")
    repo = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    authority = validate_production_authority(config, repo_root=repo) if production else None
    output = _validate_output_path(output_directory, config, production=production)
    suite = run_sampling_suite(config, repo_root=repo)
    payloads = build_public_payloads(config, suite, recorded_at=recorded_at, authority=authority)
    publish_exact_three(output, payloads)
    return {
        "status": "PASS",
        "output_directory": str(output),
        "output_names": list(OUTPUT_NAMES),
        "a6_evidence_status": "IN_PROGRESS",
        "flow_base_legal_ctmc_evidence_status": "IN_PROGRESS",
        "l3_claim_status": "NOT_ESTABLISHED",
        "a7_unlock": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--recorded-at")
    args = parser.parse_args(argv)
    config = load_config()
    output = args.output_directory or Path(config["publication"]["output_directory"])
    recorded_at = args.recorded_at or datetime.now().astimezone().isoformat(timespec="seconds")
    result = execute(output_directory=output, recorded_at=recorded_at, production=True)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
