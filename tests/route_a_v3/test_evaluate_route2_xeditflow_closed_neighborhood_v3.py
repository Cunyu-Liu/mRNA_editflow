from __future__ import annotations

import pytest

from scripts.route_a_v3.evaluate_route2_xeditflow_closed_neighborhood_v3 import validate_closed_run_config_v3


def _config():
    return {
        "schema_version": "route_a_v3_route2_xeditflow_closed_neighborhood_config.v1",
        "pool_assignment": "DEVELOPMENT",
        "split": "VALIDATION",
        "maximum_enumerated_edits": 5,
        "maximum_permutation_paths": 120,
        "enumeration": "ALL_EDIT_PERMUTATIONS_EXACT_SUM",
        "analysis_unit": "SOURCE",
        "undefined_source_policy": "EXCLUDE_NOT_ZERO_FILL",
        "beta_max": 1.0,
        "potential_kind": "SOFT_VALUE",
        "method_id": "full_soft_value_smc",
        "value_checkpoint_path": "/mnt/value.pt",
    }


def test_closed_runner_freezes_validation_exact_enumeration_and_source_unit() -> None:
    validate_closed_run_config_v3(_config())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pool_assignment", "EVALUATION", "pool"),
        ("split", "TEST", "split"),
        ("maximum_enumerated_edits", 6, "edit ceiling"),
        ("maximum_permutation_paths", 720, "permutation ceiling"),
        ("enumeration", "SINGLE_ORDER", "enumeration"),
        ("analysis_unit", "ROW", "analysis unit"),
        ("undefined_source_policy", "ZERO_FILL", "undefined-source"),
        ("potential_kind", "FREE_ACTION_RATIO", "potential kind"),
    ],
)
def test_closed_runner_rejects_benchmark_drift(field, value, message) -> None:
    config = _config()
    config[field] = value
    with pytest.raises(Exception, match=message):
        validate_closed_run_config_v3(config)


def test_closed_runner_accepts_unguided_zero_potential() -> None:
    config = _config()
    config.update(
        {
            "potential_kind": "ZERO",
            "method_id": "unguided_setflow",
        }
    )
    validate_closed_run_config_v3(config)


def test_closed_runner_rejects_method_potential_mismatch() -> None:
    config = _config()
    config.update({"potential_kind": "ZERO", "method_id": "simple_rate_guidance"})
    with pytest.raises(Exception, match="method and potential"):
        validate_closed_run_config_v3(config)
