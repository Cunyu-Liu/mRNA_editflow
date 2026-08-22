from __future__ import annotations

from scripts.route_a_v3.adjudicate_route2_xeditcritic_v3_readiness import compose_readiness_v3


def _payloads():
    three = {
        "status": "XEDITCRITIC_V3_THREE_SEED_PASS",
        "development_test_authorized": True,
    }
    atomic = {
        "status": "ATOMIC_FROZEN_DEVELOPMENT_TEST_TERMINAL",
        "frozen_test_gate": {
            "status": "XEDITCRITIC_V3_FROZEN_TEST_PASS",
            "all_development_refit_authorized": True,
        },
    }
    refit = {
        "status": "XEDITCRITIC_V3_ALL_DEVELOPMENT_REFIT_COMPLETE",
        "required_seeds": [20260831, 20260901, 20260902],
        "completed_refit_count": 3,
        "development_test_outcomes_accessed_during_refit": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    loso = {
        "status": "XEDITCRITIC_V3_LOSO_TERMINAL",
        "loso_gate": {
            "status": "XEDITCRITIC_V3_LOSO_PASS",
            "guidance_readiness_authorized": True,
        },
    }
    return three, atomic, refit, loso


def test_readiness_requires_all_four_predecessors_and_keeps_evaluation_closed() -> None:
    result = compose_readiness_v3(*_payloads())
    assert result["status"] == "CRITIC_READY_FOR_GUIDANCE"
    assert result["guidance_authorized"] is True
    assert result["new_final_evaluation_authorized"] is False
    assert result["development_test_access_event_count"] == 1
    assert result["general_test_projection_persisted"] is False


def test_readiness_stays_blocked_when_loso_gate_fails() -> None:
    payloads = list(_payloads())
    payloads[3]["loso_gate"]["status"] = "XEDITCRITIC_V3_LOSO_NO_GO"
    payloads[3]["loso_gate"]["guidance_readiness_authorized"] = False
    result = compose_readiness_v3(*payloads)
    assert result["status"] == "CRITIC_NOT_READY_FOR_GUIDANCE"
    assert result["guidance_authorized"] is False
