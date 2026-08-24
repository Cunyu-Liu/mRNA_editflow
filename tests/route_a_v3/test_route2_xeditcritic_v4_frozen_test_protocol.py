from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_v4_frozen_test_protocol_is_single_atomic_and_keeps_projections_ephemeral() -> None:
    protocol = json.loads(
        (
            ROOT
            / "configs/route_a_v3_route2_xeditcritic_v4_frozen_test_protocol_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["required_confirmation_seeds"] == [
        20260908,
        20260909,
        20260910,
    ]
    assert protocol["single_atomic_access_authorized_only_after_three_seed_pass"] is True
    assert protocol["ephemeral_test_rows_only"] is True
    assert protocol["general_test_projection_persisted"] is False
    assert protocol["test_bottom_six_cache_persisted"] is False
    assert protocol["new_final_evaluation_outcomes_accessed"] is False
