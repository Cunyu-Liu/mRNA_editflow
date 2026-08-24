from __future__ import annotations

import copy
import json
from pathlib import Path

from core.route2_xeditcritic_gate_v4 import (
    CONFIRMATION_SEEDS_V4,
    LOSO_STUDIES_V4,
    adjudicate_critic_loso_v4,
    adjudicate_critic_readiness_v4,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = (
    ROOT / "configs/route_a_v3_route2_xeditcritic_v4_posttest_protocol_v1.json"
)


def _loso_payloads() -> dict[int, dict[str, object]]:
    return {
        seed: {
            "status": "XEDITCRITIC_V4_PAIRED_LOSO_COMPLETE",
            "held_out_study_count": 7,
            "model_study_macro_spearman": 0.35,
            "baseline_study_macro_spearman": 0.20,
            "fold_margins": {study: 0.15 for study in LOSO_STUDIES_V4},
            "development_test_outcomes_accessed_during_loso": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
        for seed in CONFIRMATION_SEEDS_V4
    }


def _readiness_inputs():
    three = {
        "status": "XEDITCRITIC_V4_THREE_SEED_PASS",
        "development_test_authorized": True,
        "atomic_development_test_only": True,
        "required_seeds": list(CONFIRMATION_SEEDS_V4),
    }
    frozen = {
        "status": "XEDITCRITIC_V4_FROZEN_TEST_PASS",
        "all_development_refit_authorized": True,
    }
    refit = {
        "status": "XEDITCRITIC_V4_ALL_DEVELOPMENT_REFIT_COMPLETE",
        "required_seeds": list(CONFIRMATION_SEEDS_V4),
        "completed_refit_count": 3,
        "refit_pass_count": 8,
        "development_test_outcomes_accessed_during_refit": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    loso = adjudicate_critic_loso_v4(_loso_payloads())
    return three, frozen, refit, loso


def test_v4_posttest_protocol_freezes_refit_loso_and_protected_boundaries() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    screen = json.loads(
        (
            ROOT / "configs/route_a_v3_route2_xeditcritic_v4_screen_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["required_seeds"] == list(CONFIRMATION_SEEDS_V4)
    assert protocol["all_development_refit"]["passes"] == 8
    assert protocol["all_development_refit"]["job_count"] == 3
    assert protocol["test_preserving_loso"]["held_out_studies"] == list(
        LOSO_STUDIES_V4
    )
    assert protocol["test_preserving_loso"]["paired_job_count"] == 42
    assert protocol["test_preserving_loso"]["unknown_held_out_study_scale"] == 1.0
    assert protocol["development_test_outcomes_accessed_during_refit_or_loso"] is False
    assert protocol["new_final_evaluation_outcomes_accessed"] is False
    assert protocol["additional_seed_authorized"] is False
    assert protocol["posttest_authorization_receipt_path"].endswith(
        "/posttest_authorization_receipt.json"
    )
    assert "atomic_frozen_test_path" not in protocol
    assert protocol["formal_preflight_path"] == screen["preflight_output"]


def test_v4_loso_gate_requires_all_three_strict_seed_results() -> None:
    result = adjudicate_critic_loso_v4(_loso_payloads())
    assert result["status"] == "XEDITCRITIC_V4_LOSO_PASS"
    assert result["median_study_macro_spearman"] == 0.35
    assert result["guidance_readiness_authorized"] is True

    payloads = _loso_payloads()
    payloads[20260910]["model_study_macro_spearman"] = 0.24
    payloads[20260910]["fold_margins"]["GSE269595"] = -0.01
    result = adjudicate_critic_loso_v4(payloads)
    assert result["status"] == "XEDITCRITIC_V4_LOSO_NO_GO"
    assert result["guidance_readiness_authorized"] is False


def test_v4_readiness_requires_test_refit_and_loso_without_new_reads() -> None:
    inputs = _readiness_inputs()
    result = adjudicate_critic_readiness_v4(*inputs)
    assert result["status"] == "CRITIC_V4_READY_FOR_GUIDANCE"
    assert result["guidance_authorized"] is True
    assert result["development_test_access_event_count"] == 1
    assert result["general_test_projection_persisted"] is False
    assert result["new_final_evaluation_authorized"] is False

    failed = list(copy.deepcopy(inputs))
    failed[3]["status"] = "XEDITCRITIC_V4_LOSO_NO_GO"
    failed[3]["guidance_readiness_authorized"] = False
    result = adjudicate_critic_readiness_v4(*failed)
    assert result["status"] == "CRITIC_V4_NOT_READY_FOR_GUIDANCE"
    assert result["guidance_authorized"] is False
