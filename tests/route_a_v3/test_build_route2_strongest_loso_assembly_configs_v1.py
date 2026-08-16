from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/route_a_v3/build_route2_strongest_loso_assembly_configs_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("route2_loso_assembly_config_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_task_specific_loso_prediction_specs_are_minimal_and_seed_aligned() -> None:
    module = _load()
    assert {row["baseline_id"] for row in module.prediction_specs("GSE149487", 20260816)} == {
        "classical_edit_position_only_ridge",
        "classical_ref_alt_only_ridge",
    }
    gse269 = module.prediction_specs("GSE269595", 20260818)
    assert gse269 == [{
        "baseline_id": "neural_main_siamese_cnn",
        "prediction_path": (
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/runs/development_loso_baselines/"
            "neural_main_siamese_gse269595_seed20260818_gpu2_v1/test_predictions.jsonl"
        ),
    }]


def test_builder_remains_development_only() -> None:
    module = _load()
    base = {
        "evaluation_outcomes_accessed": False,
        "requested_split": "VALIDATION",
        "development_manifest_path": "/development.jsonl",
        "canonical_paths": ["/canonical.jsonl"],
        "strongest_selection_path": "/strongest.json",
    }
    config = module.build_config(base, "GSE114002", 20260817)
    assert config["requested_split"] == "LOSO::GSE114002"
    assert config["evaluation_outcomes_accessed"] is False
    assert config["baseline_predictions"][0]["baseline_id"] == "external_framepool"
