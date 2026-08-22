from __future__ import annotations

from scripts.route_a_v3.adjudicate_route2_xeditflow_final_v3 import METHODS
from scripts.route_a_v3.compose_route2_xeditflow_final_comparison_manifest_v3 import (
    compose_final_comparison_manifest_v3,
)


def test_final_manifest_composes_only_exact_three_seed_rows() -> None:
    rows = [
        {
            "base_flow_training_seed": seed,
            "methods": {method: f"/{seed}/{method}.json" for method in METHODS},
            "paired_bootstrap_path": f"/{seed}/bootstrap.json",
        }
        for seed in (20260904, 20260905, 20260906)
    ]
    result = compose_final_comparison_manifest_v3(
        rows,
        {
            "status": "XEDITFLOW_V3_GUIDANCE_SCREEN_FROZEN",
            "selected_kappa": 0.5,
            "selected_temperature": 1.0,
            "selected_beta_max": 2.0,
        },
    )
    assert result["status"] == "XEDITFLOW_V3_FINAL_COMPARISON_RESULTS_COMPLETE"
    assert [row["base_flow_training_seed"] for row in result["seeds"]] == [20260904, 20260905, 20260906]
    assert result["additional_seed_authorized"] is False
