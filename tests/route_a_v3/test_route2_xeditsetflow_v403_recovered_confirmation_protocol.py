from __future__ import annotations

import json
from pathlib import Path

from scripts.route_a_v3.authorize_route2_xeditsetflow_v403_recovered_confirmation import (
    SCIENCE_PROTOCOL_KEYS,
    TRAINING_HEAD,
    VALIDATION_HEAD,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = json.loads(
    (
        ROOT
        / "configs/route_a_v3_route2_xeditsetflow_v4_confirmation_protocol_v1.json"
    ).read_text(encoding="utf-8")
)
DERIVED = json.loads(
    (
        ROOT
        / "configs/route_a_v3_route2_xeditsetflow_v403_recovered_confirmation_protocol_v1.json"
    ).read_text(encoding="utf-8")
)


def test_recovery_derived_protocol_changes_no_scientific_control() -> None:
    assert all(DERIVED[key] == BASE[key] for key in SCIENCE_PROTOCOL_KEYS)
    assert DERIVED["required_seeds"] == [20260912, 20260913, 20260914]
    assert DERIVED["training_policy"] == BASE["training_policy"]
    assert DERIVED["checkpoint_gate"] == BASE["checkpoint_gate"]
    assert DERIVED["paired_bootstrap"] == BASE["paired_bootstrap"]


def test_recovery_derived_protocol_binds_dual_head_and_recovered_gate() -> None:
    provenance = DERIVED["validation_recovery_provenance"]
    assert provenance["training_git_head"] == TRAINING_HEAD
    assert provenance["validation_git_head"] == VALIDATION_HEAD
    assert TRAINING_HEAD != VALIDATION_HEAD
    assert DERIVED["screen_gate_path"] == provenance["recovered_screen_gate_path"]
    assert "v403_validation_recovery_37c590" in DERIVED["screen_gate_path"]
    assert provenance["parameter_update_count"] == 0
    assert provenance["scientific_thresholds_changed"] is False


def test_posttraining_paths_remain_bound_to_recovered_sources() -> None:
    provenance = DERIVED["validation_recovery_provenance"]
    binding = DERIVED["posttraining_binding"]
    assert binding["recovery_config_path"] == provenance["recovery_config_path"]
    assert (
        binding["recovered_screen_gate_path"]
        == provenance["recovered_screen_gate_path"]
    )
    assert "confirmation_v403_recovered" in binding["config_manifest_path"]
    assert "{runner_git_head}" in binding[
        "confirmation_authorization_output_template"
    ]


def test_protocol_never_authorizes_protected_outcomes() -> None:
    assert DERIVED["development_test_authorized"] is False
    assert DERIVED["guidance_authorized"] is False
    assert DERIVED["development_test_outcome_reads"] == 0
    assert DERIVED["new_final_evaluation_outcome_reads"] == 0
    provenance = DERIVED["validation_recovery_provenance"]
    assert provenance["development_test_outcome_reads"] == 0
    assert provenance["new_final_evaluation_outcome_reads"] == 0
