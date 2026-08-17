#!/usr/bin/env python3
"""Build a connected-component inner split from frozen Development TRAIN only."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


SPLITS = ("TRAIN", "VALIDATION", "TEST")


class TrainInnerSplitError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainInnerSplitError(message)


def validate_config(config: Mapping[str, Any]) -> None:
    _require(
        config.get("schema_version") == "route_a_v3_route2_train_inner_split.v1",
        "unexpected config schema",
    )
    _require(
        config.get("scientific_role")
        == "TRAIN_ONLY_GROUPED_MODEL_SELECTION_WITHOUT_DEVELOPMENT_VALIDATION",
        "inner split scientific role changed",
    )
    _require(config.get("source_split") == "TRAIN", "inner split source is not TRAIN")
    for key in (
        "development_validation_outcomes_accessed",
        "development_test_outcomes_accessed",
        "evaluation_outcomes_accessed",
    ):
        _require(config.get(key) is False, f"forbidden outcome access enabled: {key}")
    source_manifest = Path(str(config.get("source_development_manifest", "")))
    _require(source_manifest.is_absolute(), "source manifest path is not absolute")
    expected = config.get("expected_parent_record_counts", {})
    _require(set(expected) == set(SPLITS), "expected parent split counts are incomplete")
    _require(all(isinstance(expected[key], int) and expected[key] >= 0 for key in SPLITS), "invalid expected parent count")
    policy = config.get("split_policy", {})
    _require(policy.get("unit") == "CONNECTED_SOURCE_COMPONENT", "inner split unit changed")
    ratios = policy.get("ratios", {})
    _require(set(ratios) == set(SPLITS), "inner split ratios are incomplete")
    _require(all(isinstance(ratios[key], (int, float)) and ratios[key] > 0 for key in SPLITS), "inner split ratio is nonpositive")
    _require(math.isclose(sum(float(ratios[key]) for key in SPLITS), 1.0), "inner split ratios do not sum to one")
    _require(isinstance(policy.get("seed"), int) and not isinstance(policy.get("seed"), bool), "invalid inner split seed")
    output = config.get("output", {})
    _require(output.get("overwrite_allowed") is False, "inner split overwrite was enabled")
    _require(
        str(output.get("directory", "")).startswith(
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"
        ),
        "inner split output leaves Route 2 root",
    )
    _require(bool(output.get("manifest_filename")), "inner manifest filename is empty")
    _require(bool(output.get("summary_filename")), "inner summary filename is empty")


def read_parent_manifest(
    path: Path,
    expected_counts: Mapping[str, int],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    _require(path.is_file(), f"source Development manifest is absent: {path}")
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrainInnerSplitError(f"invalid parent manifest JSON at line {line_number}") from exc
        _require(row.get("pool_assignment") == "DEVELOPMENT", "non-Development row entered inner split")
        split = str(row.get("split"))
        _require(split in SPLITS, f"unknown parent split: {split}")
        record_id = str(row.get("canonical_record_id", ""))
        _require(record_id and record_id not in seen_ids, "parent manifest record id is empty or duplicated")
        seen_ids.add(record_id)
        counts[split] += 1
        if split != "TRAIN":
            continue
        _require(bool(row.get("connected_source_component_id")), "TRAIN row lacks connected component")
        stratum = row.get("stratum")
        _require(isinstance(stratum, list) and len(stratum) == 3, "TRAIN row has invalid stratum")
        selected.append(row)
    _require(dict(counts) == {key: expected_counts[key] for key in SPLITS}, "parent split counts changed")
    _require(bool(selected), "parent TRAIN split is empty")
    return selected, counts


def task_key(row: Mapping[str, Any]) -> str:
    stratum = row["stratum"]
    return f"{stratum[1]}|{stratum[2]}"


def assign_components(
    rows: list[dict[str, Any]],
    ratios: Mapping[str, float],
    seed: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    component_record_counts: Counter[str] = Counter()
    component_tasks: dict[str, Counter[str]] = defaultdict(Counter)
    total_tasks: Counter[str] = Counter()
    for row in rows:
        component = str(row["connected_source_component_id"])
        task = task_key(row)
        component_record_counts[component] += 1
        component_tasks[component][task] += 1
        total_tasks[task] += 1

    randomizer = random.Random(seed)
    components = list(component_record_counts)
    randomizer.shuffle(components)
    components.sort(key=component_record_counts.__getitem__, reverse=True)
    assigned_records: Counter[str] = Counter()
    assigned_tasks: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    assignment: dict[str, str] = {}
    for component in components:
        scores: dict[str, float] = {}
        for split in SPLITS:
            overall_deficit = ratios[split] * len(rows) - assigned_records[split]
            task_deficits = [
                ratios[split] * total_tasks[task] - assigned_tasks[split][task]
                for task in component_tasks[component]
            ]
            scores[split] = overall_deficit + sum(task_deficits) / len(task_deficits)
        chosen = max(SPLITS, key=lambda split: (scores[split], -SPLITS.index(split)))
        assignment[component] = chosen
        assigned_records[chosen] += component_record_counts[component]
        assigned_tasks[chosen].update(component_tasks[component])

    _require(all(assigned_records[split] > 0 for split in SPLITS), "inner split produced an empty partition")
    for task in total_tasks:
        _require(
            all(assigned_tasks[split][task] > 0 for split in SPLITS),
            f"task is absent from an inner partition: {task}",
        )
    component_counts = Counter(assignment.values())
    multitask_components = sum(len(tasks) > 1 for tasks in component_tasks.values())
    return assignment, {
        "record_counts": dict(assigned_records),
        "component_counts": dict(component_counts),
        "task_record_counts": {
            task: {split: assigned_tasks[split][task] for split in SPLITS}
            for task in sorted(total_tasks)
        },
        "task_count": len(total_tasks),
        "connected_component_count": len(component_record_counts),
        "multitask_connected_component_count": multitask_components,
    }


def execute(config: Mapping[str, Any], output_dir: Path | None = None) -> dict[str, Any]:
    validate_config(config)
    output = config["output"]
    destination = output_dir or Path(output["directory"])
    _require(not destination.exists(), f"inner split output already exists: {destination}")
    rows, parent_counts = read_parent_manifest(
        Path(config["source_development_manifest"]),
        config["expected_parent_record_counts"],
    )
    assignment, audit = assign_components(
        rows,
        config["split_policy"]["ratios"],
        int(config["split_policy"]["seed"]),
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        manifest_path = temporary / output["manifest_filename"]
        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                payload = dict(row)
                payload["parent_split"] = "TRAIN"
                payload["split"] = assignment[str(row["connected_source_component_id"])]
                payload["inner_split_id"] = str(config["inner_split_id"])
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        summary = {
            "schema_version": "route_a_v3_route2_train_inner_split_summary.v1",
            "status": "ROUTE2_TRAIN_ONLY_GROUPED_INNER_SPLIT_MATERIALIZED",
            "scientific_role": config["scientific_role"],
            "inner_split_id": config["inner_split_id"],
            "source_development_manifest": config["source_development_manifest"],
            "source_split": "TRAIN",
            "parent_record_counts": {key: parent_counts[key] for key in SPLITS},
            "excluded_parent_validation_record_count": parent_counts["VALIDATION"],
            "excluded_parent_test_record_count": parent_counts["TEST"],
            "component_overlap_across_inner_splits": 0,
            "development_validation_outcomes_accessed": False,
            "development_test_outcomes_accessed": False,
            "evaluation_outcomes_accessed": False,
            **audit,
            "scientific_claim_status": "NOT_ESTABLISHED",
        }
        (temporary / output["summary_filename"]).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "run_config.json").write_text(
            json.dumps(dict(config), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    summary = execute(config, args.output_dir)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
