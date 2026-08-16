from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/route_a_v3_route2_delta_capacity_profiles_v1.json"
MODEL = ROOT / "core/route2_delta_predictor.py"


def test_capacity_profiles_match_actual_full_context_parameter_geometry() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("route2_capacity_model_test", MODEL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    geometry = config["development_geometry"]["vocabulary_sizes_including_unknown"]
    for profile in config["profiles"]:
        model = module.Route2DeltaPredictor(
            hidden_dim=profile["hidden_dim"],
            depth=profile["depth"],
            **geometry,
        )
        assert sum(parameter.numel() for parameter in model.parameters()) == profile["actual_parameter_count"]


def test_profiles_are_near_named_budgets_and_do_not_claim_science() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["development_geometry"]["evaluation_records_read"] == 0
    assert config["scientific_claim_status"] == "NOT_ESTABLISHED"
    for profile in config["profiles"]:
        relative_error = abs(profile["actual_parameter_count"] - profile["target_parameter_count"]) / profile["target_parameter_count"]
        # Integer hidden widths make the small profile slightly less granular;
        # all three remain within 0.2% of their named capacity budget.
        assert relative_error < 0.002


def test_neural_baselines_are_in_matched_parameter_bands() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("route2_baseline_capacity_model_test", MODEL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    geometry = config["development_geometry"]["vocabulary_sizes_including_unknown"]
    profiles = config["architecture_controlled_neural_baseline_profiles"]
    tolerance = profiles["target_band_relative_tolerance"]
    for band in ("MEDIUM_0_5M", "MAIN_2M"):
        assert {row["model_kind"] for row in profiles[band]} == module.Route2NeuralBaseline.MODES
        for row in profiles[band]:
            model = module.Route2NeuralBaseline(
                mode=row["model_kind"], hidden_dim=row["hidden_dim"], depth=row["depth"],
                max_length=config["development_geometry"]["sequence_length"]["configured_maximum"],
                **geometry,
            )
            observed = sum(parameter.numel() for parameter in model.parameters())
            assert observed == row["actual_parameter_count"]
            assert abs(observed - row["target_parameter_count"]) / row["target_parameter_count"] <= tolerance
