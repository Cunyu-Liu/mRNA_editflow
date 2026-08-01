"""Correlation and redundancy diagnostics for vector reward teacher artifacts."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from mrna_editflow.rl.reward_vector import RewardVector


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    lx, rx = _mean(left), _mean(right)
    numerator = sum((a - lx) * (b - rx) for a, b in zip(left, right))
    denom_left = sum((a - lx) ** 2 for a in left) ** 0.5
    denom_right = sum((b - rx) ** 2 for b in right) ** 0.5
    return numerator / (denom_left * denom_right) if denom_left and denom_right else 0.0


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        for index in order[start:end]:
            result[index] = rank
        start = end
    return result


def reward_correlation_report(
    vectors: Sequence[RewardVector], *, redundancy_threshold: float = 0.95
) -> dict[str, object]:
    columns: dict[str, list[float]] = defaultdict(list)
    for vector in vectors:
        for name, value in vector.raw_delta_from_source.items():
            if vector.validity.get(name, True):
                columns[name].append(float(value))
    names = sorted(columns)
    constants = [name for name in names if len(set(columns[name])) <= 1]
    pairs = []
    groups: list[set[str]] = []
    for left_pos, left in enumerate(names):
        for right in names[left_pos + 1:]:
            if len(columns[left]) != len(columns[right]):
                continue
            pearson = _pearson(columns[left], columns[right])
            spearman = _pearson(_ranks(columns[left]), _ranks(columns[right]))
            rank_agreement = _mean([
                1.0 if a == b else 0.0
                for a, b in zip(_ranks(columns[left]), _ranks(columns[right]))
            ])
            redundant = abs(pearson) >= redundancy_threshold and abs(spearman) >= redundancy_threshold
            pairs.append({"left": left, "right": right, "pearson": pearson, "spearman": spearman, "rank_agreement": rank_agreement, "mutual_redundancy_warning": redundant})
            if redundant:
                matching = [group for group in groups if left in group or right in group]
                merged = {left, right}
                for group in matching:
                    merged.update(group)
                    groups.remove(group)
                groups.append(merged)
    return {
        "n_vectors": len(vectors), "redundancy_threshold": redundancy_threshold,
        "constant_objective_warning": constants,
        "pairs": pairs,
        "redundancy_groups": [sorted(group) for group in groups],
    }


def write_reward_correlation_report(report: Mapping[str, object], json_path: str, markdown_path: str) -> None:
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(dict(report), fh, indent=2, sort_keys=True)
    warnings = report.get("constant_objective_warning", [])
    groups = report.get("redundancy_groups", [])
    lines = ["# Reward correlation report", "", f"Vectors: {report.get('n_vectors', 0)}", "", "## Warnings", "", f"- Constant objectives: {warnings}", f"- Redundancy groups: {groups}", "", "## Pairwise statistics", "", "| left | right | Pearson | Spearman | rank agreement | redundant |", "|---|---|---:|---:|---:|---:|"]
    for pair in report.get("pairs", []):
        if isinstance(pair, Mapping):
            lines.append(f"| {pair['left']} | {pair['right']} | {float(pair['pearson']):.4f} | {float(pair['spearman']):.4f} | {float(pair['rank_agreement']):.4f} | {bool(pair['mutual_redundancy_warning'])} |")
    with open(markdown_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


__all__ = ["reward_correlation_report", "write_reward_correlation_report"]
