from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/route_a_v3_route2_xeditcritic_v6_screen_v1.json"


def _load() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_v6_screen_has_exactly_four_runs_with_distinct_lambda_weights() -> None:
    config = _load()
    runs = config["required_screen_runs"]
    assert [run["run_id"] for run in runs] == [
        "v6_full",
        "v6_h3_lambda_0_5",
        "v6_h3_lambda_0_75",
        "v6_h3_lambda_1_0",
    ]
    weights = {run["run_id"]: run["lambda_pairwise_weight"] for run in runs}
    assert weights == {
        "v6_full": 1.0,
        "v6_h3_lambda_0_5": 0.5,
        "v6_h3_lambda_0_75": 0.75,
        "v6_h3_lambda_1_0": 1.0,
    }


def test_v6_screen_lambda_ablation_is_distinct_from_full() -> None:
    """The H3 lambda ablation arms must be genuinely distinct from v6_full,
    otherwise the W2-a ablation is meaningless (regression guard)."""
    config = _load()
    weights = {
        run["run_id"]: run["lambda_pairwise_weight"]
        for run in config["required_screen_runs"]
    }
    assert weights["v6_h3_lambda_0_5"] != weights["v6_full"]
    assert weights["v6_h3_lambda_0_75"] != weights["v6_full"]
    assert weights["v6_h3_lambda_0_5"] < weights["v6_h3_lambda_0_75"] < weights["v6_full"]


def test_v6_screen_keeps_global_lambda_default_and_pair_mean_settings() -> None:
    config = _load()
    training = config["training"]
    # global default remains as fallback for arms that do not specify a per-run lambda
    assert training["lambda_pairwise_weight"] == 1.0
    assert training["pair_mean_targets"] is True
    assert training["per_task_rank_gaussian"] is True
    assert training["per_pass_validation"] is True
    assert training["checkpoint_selection"] == "FINAL_PASS_8_FIXED_NO_VALIDATION_PEAK_RESELECTION"
