"""Exact order-invariant terminal probabilities and closed-neighborhood metrics."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from core.route2_legal_xeditflow import (
    STOP,
    FlowState,
    LegalAction,
    RateFunction,
    initial_state,
    jump_distribution,
)


class ClosedNeighborhoodV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ClosedNeighborhoodV3Error(message)


def source_relative_substitutions_v3(
    source_sequence: str, candidate_sequence: str
) -> tuple[tuple[int, str], ...]:
    source = source_sequence.upper().replace("T", "U")
    candidate = candidate_sequence.upper().replace("T", "U")
    _require(len(source) == len(candidate) and bool(source), "closed candidate length differs")
    _require(set(source) <= set("ACGU") and set(candidate) <= set("ACGU"), "closed candidate alphabet differs")
    edits = tuple(
        (index, candidate_base)
        for index, (source_base, candidate_base) in enumerate(zip(source, candidate))
        if source_base != candidate_base
    )
    _require(len(edits) <= 5, "closed candidate exceeds the five-edit enumeration ceiling")
    return edits


def exact_order_invariant_terminal_probability_v3(
    source_sequence: str,
    candidate_sequence: str,
    *,
    edit_budget: int,
    assay_id: str,
    context_id: str,
    rate_function: RateFunction,
    support_floor: float = 1e-8,
) -> dict[str, Any]:
    edits = source_relative_substitutions_v3(source_sequence, candidate_sequence)
    _require(edit_budget in {1, 3, 5}, "closed benchmark edit budget is not 1/3/5")
    _require(len(edits) <= edit_budget, "closed candidate exceeds its edit budget")
    root = initial_state(
        source_sequence.upper().replace("T", "U"),
        budget=edit_budget,
        assay_id=assay_id,
        context_id=context_id,
    )
    edge_cache: dict[FlowState, dict[LegalAction, tuple[FlowState, float]]] = {}

    def edges(state: FlowState) -> dict[LegalAction, tuple[FlowState, float]]:
        if state not in edge_cache:
            edge_cache[state] = {
                action: (child, probability)
                for action, child, probability in jump_distribution(
                    state, rate_function, support_floor=support_floor
                )
            }
        return edge_cache[state]

    path_probabilities = []
    terminal_causes = set()
    for permutation in itertools.permutations(edits):
        state = root
        probability = 1.0
        for position, alt_base in permutation:
            _require(state.terminal_cause is None, "closed edit path terminated before its target")
            action = LegalAction("SUB", position, alt_base)
            _require(action in edges(state), "closed target edit is not legal")
            state, edge_probability = edges(state)[action]
            probability *= edge_probability
        if state.terminal_cause is None:
            stop = LegalAction(STOP)
            _require(stop in edges(state), "closed target lacks a legal STOP")
            state, stop_probability = edges(state)[stop]
            probability *= stop_probability
        _require(
            state.current_sequence == candidate_sequence.upper().replace("T", "U"),
            "closed path did not reach its terminal candidate",
        )
        _require(state.terminal_cause is not None, "closed target is not terminal")
        terminal_causes.add(state.terminal_cause)
        path_probabilities.append(probability)
    _require(path_probabilities, "closed candidate produced no edit permutation")
    probability = math.fsum(path_probabilities)
    _require(math.isfinite(probability) and 0.0 <= probability <= 1.0, "closed terminal probability is invalid")
    return {
        "schema_version": "ClosedNeighborhoodResultV1",
        "edit_count": len(edits),
        "edit_budget": edit_budget,
        "permutation_path_count": math.factorial(len(edits)),
        "terminal_probability": probability,
        "terminal_causes": sorted(terminal_causes),
        "unique_scored_state_count": len(edge_cache),
        "all_edit_permutations_enumerated": True,
        "maximum_supported_edit_count": 5,
    }


def _descending_tie_blocks(scores: np.ndarray) -> list[np.ndarray]:
    ordered = np.argsort(-scores, kind="mergesort")
    blocks = []
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and scores[ordered[end]] == scores[ordered[start]]:
            end += 1
        blocks.append(ordered[start:end])
        start = end
    return blocks


def _tie_aware_dcg(gains: np.ndarray, scores: np.ndarray) -> float:
    rank = 0
    result = 0.0
    for block in _descending_tie_blocks(scores):
        block_gain = float(np.mean(gains[block]))
        for offset in range(len(block)):
            result += block_gain / math.log2(rank + offset + 2.0)
        rank += len(block)
    return result


def closed_neighborhood_metrics_v1(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        source_key = str(row["source_key"])
        probability = float(row["terminal_probability"])
        outcome = float(row["measured_direction_normalized_delta"])
        _require(
            math.isfinite(probability) and probability >= 0.0,
            f"closed terminal probability is invalid: {source_key}",
        )
        _require(math.isfinite(outcome), f"closed measured outcome is invalid: {source_key}")
        by_source[source_key].append(row)
    _require(bool(by_source), "closed neighborhood is empty")
    per_source = {}
    for source_key in sorted(by_source):
        members = by_source[source_key]
        sequences = [str(row["candidate_sequence"]) for row in members]
        _require(
            len(sequences) == len(set(sequences)),
            f"closed measured candidate is duplicated: {source_key}",
        )
        if len(members) < 2:
            per_source[source_key] = {
                "status": "UNDEFINED_FEWER_THAN_TWO_LEGAL_MEASURED_CANDIDATES",
                "measured_candidate_count": len(members),
                "ndcg": None,
                "normalized_regret": None,
                "top_1_recall": None,
            }
            continue
        outcomes = np.asarray(
            [float(row["measured_direction_normalized_delta"]) for row in members],
            dtype=float,
        )
        probabilities = np.asarray(
            [float(row["terminal_probability"]) for row in members], dtype=float
        )
        gains = outcomes - float(np.min(outcomes))
        ideal_dcg = _tie_aware_dcg(gains, outcomes)
        ndcg = None if ideal_dcg == 0.0 else _tie_aware_dcg(gains, probabilities) / ideal_dcg
        predicted_top = np.flatnonzero(probabilities == float(np.max(probabilities)))
        true_top = set(np.flatnonzero(outcomes == float(np.max(outcomes))).tolist())
        top_1_recall = sum(int(index in true_top) for index in predicted_top) / len(predicted_top)
        selected_outcome = float(np.mean(outcomes[predicted_top]))
        span = float(np.max(outcomes) - np.min(outcomes))
        regret = 0.0 if span == 0.0 else (float(np.max(outcomes)) - selected_outcome) / span
        per_source[source_key] = {
            "status": (
                "DEFINED"
                if ndcg is not None
                else "UNDEFINED_ZERO_MEASURED_GAIN"
            ),
            "measured_candidate_count": len(members),
            "terminal_probability_sum_over_measured_candidates": float(
                np.sum(probabilities)
            ),
            "ndcg": None if ndcg is None else float(ndcg),
            "normalized_regret": regret,
            "top_1_recall": float(top_1_recall),
            "selected_measured_outcome": selected_outcome,
        }
    defined = [row for row in per_source.values() if row["status"] == "DEFINED"]
    regret_defined = [
        row for row in per_source.values() if row["normalized_regret"] is not None
    ]
    top_defined = [
        row for row in per_source.values() if row["top_1_recall"] is not None
    ]
    return {
        "schema_version": "ClosedNeighborhoodResultV1",
        "analysis_unit": "SOURCE",
        "source_count": len(per_source),
        "defined_source_count": len(defined),
        "undefined_source_count": len(per_source) - len(defined),
        "source_macro_ndcg": (
            None if not defined else float(np.mean([row["ndcg"] for row in defined]))
        ),
        "source_macro_normalized_regret": (
            None
            if not regret_defined
            else float(np.mean([row["normalized_regret"] for row in regret_defined]))
        ),
        "source_macro_top_1_recall": (
            None
            if not top_defined
            else float(np.mean([row["top_1_recall"] for row in top_defined]))
        ),
        "undefined_sources_are_not_filled_with_zero": True,
        "per_source": per_source,
    }
