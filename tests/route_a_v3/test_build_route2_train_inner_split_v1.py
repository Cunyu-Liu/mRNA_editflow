from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/build_route2_train_inner_split_v1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("train_inner_split_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(record_id: str, component: str, task: str, split: str = "TRAIN") -> dict:
    region, endpoint = task.split("|", maxsplit=1)
    return {
        "canonical_record_id": record_id,
        "study_unit_id": f"study-{task}",
        "pool_assignment": "DEVELOPMENT",
        "split": split,
        "connected_source_component_id": component,
        "source_group_key": component,
        "stratum": [f"study-{task}", region, endpoint],
    }


def _config(manifest: Path, counts: dict[str, int]) -> dict:
    return {
        "schema_version": "route_a_v3_route2_train_inner_split.v1",
        "scientific_role": "TRAIN_ONLY_GROUPED_MODEL_SELECTION_WITHOUT_DEVELOPMENT_VALIDATION",
        "inner_split_id": "SYNTHETIC_INNER_V1",
        "source_development_manifest": str(manifest),
        "source_split": "TRAIN",
        "expected_parent_record_counts": counts,
        "split_policy": {
            "unit": "CONNECTED_SOURCE_COMPONENT",
            "ratios": {"TRAIN": 0.7, "VALIDATION": 0.15, "TEST": 0.15},
            "seed": 17,
        },
        "development_validation_outcomes_accessed": False,
        "development_test_outcomes_accessed": False,
        "evaluation_outcomes_accessed": False,
        "output": {
            "directory": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/test-inner",
            "manifest_filename": "development_inner_manifest.jsonl",
            "summary_filename": "summary.json",
            "overwrite_allowed": False,
        },
    }


def test_inner_split_uses_parent_train_only_and_keeps_components_whole(tmp_path: Path) -> None:
    module = _load_module()
    rows = []
    tasks = ("3UTR|TASK_A", "5UTR|TASK_B")
    for task in tasks:
        for index in range(12):
            rows.append(_row(f"{task}-{index}", f"{task}-component-{index}", task))
    rows.extend([
        _row("shared-a", "shared-component", tasks[0]),
        _row("shared-b", "shared-component", tasks[1]),
        _row("parent-validation", "withheld-validation", tasks[0], "VALIDATION"),
        _row("parent-test", "withheld-test", tasks[1], "TEST"),
    ])
    manifest = tmp_path / "parent.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    config = _config(manifest, {"TRAIN": 26, "VALIDATION": 1, "TEST": 1})
    output = tmp_path / "output"
    summary = module.execute(config, output)
    materialized = [
        json.loads(line)
        for line in (output / "development_inner_manifest.jsonl").read_text().splitlines()
    ]

    assert len(materialized) == 26
    assert {row["canonical_record_id"] for row in materialized}.isdisjoint(
        {"parent-validation", "parent-test"}
    )
    component_splits = defaultdict(set)
    task_splits = defaultdict(set)
    for row in materialized:
        component_splits[row["connected_source_component_id"]].add(row["split"])
        task_splits[f"{row['stratum'][1]}|{row['stratum'][2]}"].add(row["split"])
        assert row["parent_split"] == "TRAIN"
    assert all(len(splits) == 1 for splits in component_splits.values())
    assert all(splits == set(module.SPLITS) for splits in task_splits.values())
    assert summary["record_counts"] == {"TRAIN": 18, "VALIDATION": 4, "TEST": 4}
    assert summary["connected_component_count"] == 25
    assert summary["multitask_connected_component_count"] == 1
    assert summary["component_overlap_across_inner_splits"] == 0
    assert summary["excluded_parent_validation_record_count"] == 1
    assert summary["excluded_parent_test_record_count"] == 1
    assert summary["development_validation_outcomes_accessed"] is False
    assert summary["development_test_outcomes_accessed"] is False
    assert summary["evaluation_outcomes_accessed"] is False


def test_inner_split_rejects_any_outcome_access(tmp_path: Path) -> None:
    module = _load_module()
    config = _config(
        tmp_path / "unused.jsonl",
        {"TRAIN": 1, "VALIDATION": 0, "TEST": 0},
    )
    config["development_validation_outcomes_accessed"] = True
    with pytest.raises(module.TrainInnerSplitError):
        module.validate_config(config)
