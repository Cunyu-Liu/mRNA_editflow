from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/route_a_v3/build_route2_classical_strongest_loso_configs_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("route2_classical_loso_config_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_builder_freezes_only_selected_classical_winners() -> None:
    module = _load()
    baselines = []
    for baseline_id in module.FROZEN_PARAMETERS:
        baselines.append({
            "baseline_id": baseline_id,
            "kind": "group_mean" if baseline_id == "context_only_mean" else "ridge",
            "parameter_grid": {"unused": [1]},
        })
    base = {
        "evaluation_outcomes_accessed": False,
        "cpu_thread_cap": 4,
        "minimum_free_gpu_memory_bytes": 8,
        "development_manifest_path": "/manifest.jsonl",
        "canonical_paths": ["/canonical.jsonl"],
        "baselines": baselines,
    }
    config = module.build_config(base, "GSE269595", 2)
    assert config["run_mode"] == config["result_stage"] == "LOSO_FROZEN_PARAMETERS"
    assert config["device"] == "cuda:2"
    assert config["physical_gpu_index"] == 2
    assert config["evaluation_outcomes_accessed"] is False
    assert {row["baseline_id"] for row in config["baselines"]} == {
        f"classical_{baseline_id}" for baseline_id in module.FROZEN_PARAMETERS
    }
    assert all("parameter_grid" not in row for row in config["baselines"])
    assert {row["baseline_id"]: row["frozen_parameters"] for row in config["baselines"]} == {
        f"classical_{baseline_id}": parameters
        for baseline_id, parameters in module.FROZEN_PARAMETERS.items()
    }
