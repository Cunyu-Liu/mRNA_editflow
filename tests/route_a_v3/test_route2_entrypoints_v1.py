from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINTS = (
    "adjudicate_route2_independent_generation_evaluator_v1.py",
    "build_route2_generation_baseline_selection_input_v2.py",
    "compare_route2_exhaustive_small_space_reference_v1.py",
    "predict_route2_frozen_classical_v1.py",
    "predict_route2_frozen_neural_v1.py",
    "run_route2_aparent_baseline_v1.py",
    "run_route2_base_flow_g0_validation_v1.py",
    "run_route2_classical_prediction_baselines_v1.py",
    "run_route2_external_prediction_baselines_v1.py",
    "run_route2_matched_generation_suite_v1.py",
    "run_route2_search_generation_baselines_v1.py",
    "run_route2_utrlm_baseline_v1.py",
    "score_route2_generation_independent_evaluator_v1.py",
    "train_route2_base_flow_g0_v1.py",
    "train_route2_delta_predictor_v1.py",
)


@pytest.mark.parametrize("filename", ENTRYPOINTS)
def test_direct_script_entrypoint_resolves_repository_packages(filename: str) -> None:
    path = ROOT / "scripts/route_a_v3" / filename
    completed = subprocess.run(
        [sys.executable, str(path), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()
