#!/usr/bin/env python3
"""Run the DEC028 SS6 CPU-only synthetic exact-reference suite."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/route_a_v3_dec028_a6_ss6_nonlearned_exact_reference_v1.json"
STOP = (-1,)


class ContractError(RuntimeError):
    pass


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["protocol_id"] != "ROUTE_A_V3_DEC028_A6_SS6_NONLEARNED_EXACT_REFERENCE_V1":
        raise ContractError("protocol identity differs")
    graph = value["graph_contract"]
    if graph["graph_count"] != 96 or graph["budgets"] != [1, 3, 5] or graph["graphs_per_budget"] != 32:
        raise ContractError("96-graph budget geometry differs")
    if graph["editable_positions"] != 5 or graph["alias_actions_per_edit"] != 2:
        raise ContractError("DAG or alias geometry differs")
    if value["time_contract"]["semantics"] != "ALGORITHMIC_TIME_ONLY_NOT_PHYSICAL_KINETICS":
        raise ContractError("algorithmic-time boundary differs")
    if value["future_learned_contract"]["learned_execution_authorized"] is not False:
        raise ContractError("learned execution was enabled")
    if any(value["locks"][key] != 0 for key in ("project_data_row_reads", "torch_imports", "model_constructions", "optimizer_constructions", "cuda_touches", "parameter_updates", "checkpoint_writes")):
        raise ContractError("nonlearned zero-touch boundary differs")
    return value


def states(position_count: int, budget: int) -> list[tuple[int, ...]]:
    return [tuple((mask >> position) & 1 for position in range(position_count)) for mask in range(1 << position_count) if bin(mask).count("1") <= budget]


def raw_actions(state: tuple[int, ...], budget: int, graph_index: int, segment: int, support_floor: float):
    if state == STOP:
        return [(STOP, 0.0, "STOP_ABSORB")]
    actions = []
    edits = sum(state)
    if edits < budget:
        for position, value in enumerate(state):
            if value:
                continue
            target = list(state); target[position] = 1; target = tuple(target)
            for alias in range(2):
                rate = support_floor + 0.025 * (1 + ((graph_index * 7 + position * 3 + alias * 5 + segment * 11) % 17))
                actions.append((target, rate, f"EDIT_{position}_ALIAS_{alias}"))
    stop_rate = support_floor + 0.02 * (1 + ((graph_index * 13 + edits * 5 + segment * 7) % 11))
    actions.append((STOP, stop_rate, "STOP"))
    return actions


def aggregated_transitions(state, budget, graph_index, segment, support_floor):
    rates: dict[tuple[int, ...], float] = {}
    for target, rate, _label in raw_actions(state, budget, graph_index, segment, support_floor):
        rates[target] = rates.get(target, 0.0) + rate
    return rates


def uniformization_rate(all_states, budget, graph_index, segment, support_floor):
    return max(sum(aggregated_transitions(state, budget, graph_index, segment, support_floor).values()) for state in all_states if state != STOP)


def poisson_weights(mean: float, tolerance: float = 1e-15) -> list[float]:
    values = [math.exp(-mean)]
    cumulative = values[0]
    order = 0
    while 1.0 - cumulative > tolerance:
        order += 1
        values.append(values[-1] * mean / order)
        cumulative += values[-1]
        if order > 256:
            raise ContractError("uniformization Poisson tail did not close")
    values[-1] += 1.0 - cumulative
    return values


def dp_segment(distribution, all_states, budget, graph_index, segment, duration, support_floor):
    uniform = uniformization_rate(all_states, budget, graph_index, segment, support_floor)
    weights = poisson_weights(uniform * duration)
    current = dict(distribution)
    result = {state: weights[0] * current.get(state, 0.0) for state in all_states}
    for weight in weights[1:]:
        following = {state: 0.0 for state in all_states}
        for state, mass in current.items():
            if state == STOP:
                following[STOP] += mass
                continue
            transitions = aggregated_transitions(state, budget, graph_index, segment, support_floor)
            total = sum(transitions.values())
            following[state] += mass * (1.0 - total / uniform)
            for target, rate in transitions.items():
                following[target] += mass * rate / uniform
        current = following
        for state, mass in current.items():
            result[state] += weight * mass
    return result


def enumeration_segment(distribution, all_states, budget, graph_index, segment, duration, support_floor):
    uniform = uniformization_rate(all_states, budget, graph_index, segment, support_floor)
    weights = poisson_weights(uniform * duration)

    @lru_cache(maxsize=None)
    def terminal_after(state: tuple[int, ...], steps: int):
        if steps == 0 or state == STOP:
            return {state: 1.0}
        raw = raw_actions(state, budget, graph_index, segment, support_floor)
        total = sum(rate for _target, rate, _label in raw)
        choices = [(state, 1.0 - total / uniform)] + [(target, rate / uniform) for target, rate, _label in raw]
        result: dict[tuple[int, ...], float] = {}
        for target, probability in choices:
            for terminal, conditional in terminal_after(target, steps - 1).items():
                result[terminal] = result.get(terminal, 0.0) + probability * conditional
        return result

    result = {state: 0.0 for state in all_states}
    for initial, initial_mass in distribution.items():
        for steps, weight in enumerate(weights):
            for terminal, probability in terminal_after(initial, steps).items():
                result[terminal] += initial_mass * weight * probability
    return result


def total_variation(left, right):
    return 0.5 * sum(abs(left.get(state, 0.0) - right.get(state, 0.0)) for state in set(left) | set(right))


def run_suite(config: Mapping[str, Any]) -> dict[str, Any]:
    graph = config["graph_contract"]; reference = config["reference_contract"]
    durations = config["time_contract"]["piecewise_constant_segment_durations"]
    support_floor = float(graph["support_floor"])
    max_tv = 0.0; max_mass_error = 0.0; max_recovery_error = 0.0
    checked = 0; alias_merges = 0; illegal_edges = 0
    for budget in graph["budgets"]:
        all_states = states(graph["editable_positions"], budget) + [STOP]
        source = (0,) * graph["editable_positions"]
        for local_index in range(graph["graphs_per_budget"]):
            graph_index = graph["graphs_per_budget"] * graph["budgets"].index(budget) + local_index
            dp = {state: 0.0 for state in all_states}; dp[source] = 1.0
            enumeration = dict(dp)
            for segment, duration in enumerate(durations):
                for state in all_states:
                    if state == STOP:
                        continue
                    raw = raw_actions(state, budget, graph_index, segment, support_floor)
                    aggregated = aggregated_transitions(state, budget, graph_index, segment, support_floor)
                    alias_merges += len(raw) - len(aggregated)
                    for target, rate, _label in raw:
                        if rate < support_floor or (target != STOP and (sum(target) != sum(state) + 1 or sum(target) > budget)):
                            illegal_edges += 1
                        if target != STOP:
                            h_source = 1.0 + 0.1 * sum(state)
                            h_target = 1.0 + 0.1 * sum(target)
                            tilted = rate * h_target / h_source
                            recovered = tilted * h_source / h_target
                            max_recovery_error = max(max_recovery_error, abs(recovered - rate))
                dp = dp_segment(dp, all_states, budget, graph_index, segment, duration, support_floor)
                enumeration = enumeration_segment(enumeration, all_states, budget, graph_index, segment, duration, support_floor)
            max_tv = max(max_tv, total_variation(dp, enumeration))
            max_mass_error = max(max_mass_error, abs(sum(dp.values()) - 1.0), abs(sum(enumeration.values()) - 1.0))
            checked += 1
    passed = max_tv <= reference["terminal_total_variation_maximum"] and max_mass_error <= reference["mass_error_maximum"] and max_recovery_error <= reference["base_rate_recovery_error_maximum"] and illegal_edges == 0 and checked == graph["graph_count"]
    return {
        "protocol_id": config["protocol_id"],
        "status": "PASS_SS6_NONLEARNED_ENGINEERING_REFERENCE" if passed else "FAIL_SS6_NONLEARNED_ENGINEERING_REFERENCE",
        "graph_count": checked,
        "budgets": graph["budgets"],
        "graphs_per_budget": graph["graphs_per_budget"],
        "time_scope": "GENERAL_FINITE_PIECEWISE_CONSTANT_ALGORITHMIC_TIME_SCHEDULE",
        "arbitrary_continuous_time_claimed": False,
        "maximum_terminal_total_variation": max_tv,
        "maximum_mass_error": max_mass_error,
        "maximum_base_rate_recovery_error": max_recovery_error,
        "alias_merge_count": alias_merges,
        "illegal_edge_count": illegal_edges,
        "stop_absorbing": True,
        "support_floor": support_floor,
        "future_learned_execution_authorized": False,
        "critic_lcb_manifest_available": False,
        "project_data_row_reads": 0,
        "torch_imports": 0,
        "model_constructions": 0,
        "optimizer_constructions": 0,
        "cuda_touches": 0,
        "parameter_updates": 0,
        "checkpoint_writes": 0,
        "a7_allowed": False,
        "scientific_claim_status": "NOT_ESTABLISHED"
    }


def publish(report: Mapping[str, Any], output_dir: Path, filename: str) -> None:
    if output_dir.exists():
        raise ContractError("output directory already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp.", dir=output_dir.parent))
    try:
        (temporary / filename).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.rename(temporary, output_dir)
    except Exception:
        if temporary.exists():
            for child in temporary.iterdir(): child.unlink()
            temporary.rmdir()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    report = run_suite(config)
    if not report["status"].startswith("PASS"):
        raise ContractError(report["status"])
    publish(report, args.output_dir, config["output_filename"])
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
