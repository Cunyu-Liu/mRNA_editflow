from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.r3x.metrics import (
    MetricProtocolError,
    cell_metrics,
    component_balanced_cell_metrics,
    legacy_signed_ratio_enrichment,
    macro_metrics,
    normalized_selection_gain,
    within_pool_rank_metrics,
)
from scripts.r3x.statistics import (
    holm_adjust,
    paired_component_bootstrap,
    paired_component_permutation,
    paired_primary_sign_bootstrap,
    paired_primary_sign_permutation,
    summarize_seeds,
)


def test_cell_metrics_perfect_and_zero_policy() -> None:
    out = cell_metrics(
        np.array([-2.0, 0.0, 1.0, 3.0]),
        np.array([-1.0, 99.0, 2.0, 4.0]),
        logvar=np.zeros(4),
    )
    assert out["sign_accuracy"] == 1.0
    assert out["n_sign"] == 3
    assert out["delta_spearman"] == pytest.approx(0.4)
    assert out["rmse"] == pytest.approx(math.sqrt((1 + 99**2 + 1 + 1) / 4))
    assert out["mae"] == pytest.approx(25.5)
    assert math.isfinite(out["gaussian_nll"])
    assert out["zero_policy"] == "EXCLUDE_TRUE_ZERO"


def test_cell_metrics_rejects_nonfinite_and_shape_mismatch() -> None:
    with pytest.raises(MetricProtocolError):
        cell_metrics([1.0], [float("nan")])
    with pytest.raises(MetricProtocolError):
        cell_metrics([1.0, 2.0], [1.0])


def test_all_zero_sign_is_hold_not_pass() -> None:
    out = cell_metrics([0.0, 0.0], [1.0, -1.0])
    assert out["sign_accuracy"] is None
    assert out["sign_status"] == "HOLD_NO_NONZERO_TRUE_EFFECTS"


def test_zero_prediction_uses_same_ge_zero_rule_as_training() -> None:
    out = cell_metrics([1.0, -1.0], [0.0, 0.0])
    assert out["sign_accuracy"] == pytest.approx(0.5)


def test_normalized_selection_gain_is_monotone_on_all_negative_values() -> None:
    y = np.array([-5.0, -4.0, -3.0, -2.0, -1.0])
    perfect = normalized_selection_gain(y, y, fraction=0.4)
    reverse = normalized_selection_gain(y, -y, fraction=0.4)
    assert perfect["status"] == "DIAGNOSTIC_ONLY_NO_FROZEN_GATE"
    assert perfect["value"] == pytest.approx(1.0)
    assert reverse["value"] < perfect["value"]


def test_normalized_selection_gain_holds_on_zero_headroom() -> None:
    out = normalized_selection_gain([1.0, 1.0, 1.0], [3.0, 2.0, 1.0])
    assert out["value"] is None
    assert out["status"] == "HOLD_NO_STABLE_ORACLE_HEADROOM"


def test_selection_gain_cutoff_ties_are_row_order_invariant() -> None:
    y = np.asarray([10.0, 1.0, -3.0, 2.0, 4.0])
    pred = np.zeros_like(y)
    first = normalized_selection_gain(y, pred, fraction=0.4)
    permutation = np.asarray([4, 2, 0, 3, 1])
    changed = normalized_selection_gain(y[permutation], pred[permutation], fraction=0.4)
    assert first == changed
    assert first["tie_policy"] == "EXPECTED_OUTCOME_OVER_CUTOFF_TIE_BLOCK"


def test_legacy_signed_ratio_is_prohibited() -> None:
    with pytest.raises(MetricProtocolError, match="PROHIBITED"):
        legacy_signed_ratio_enrichment([-3.0, -2.0], [-2.0, -3.0])


def test_macro_is_cell_balanced_not_row_weighted() -> None:
    cells = {
        "large": {"n": 1000, "sign_accuracy": 1.0, "delta_spearman": 0.8,
                  "rmse": 1.0, "mae": 0.8, "gaussian_nll": 1.2},
        "small": {"n": 2, "sign_accuracy": 0.0, "delta_spearman": -0.2,
                  "rmse": 3.0, "mae": 2.8, "gaussian_nll": 4.2},
    }
    out = macro_metrics(cells)
    assert out["macro_sign_accuracy"] == pytest.approx(0.5)
    assert out["macro_delta_spearman"] == pytest.approx(0.3)
    assert out["n_cells"] == 2


def test_macro_never_silently_drops_an_undefined_primary_cell() -> None:
    cells = {
        "easy": {"n": 10, "sign_accuracy": 0.9, "delta_spearman": 0.8,
                 "rmse": 1.0, "mae": 0.8, "gaussian_nll": 1.2},
        "hard": {"n": 10, "sign_accuracy": 0.5, "delta_spearman": None,
                 "rmse": 2.0, "mae": 1.8, "gaussian_nll": 2.2},
    }
    out = macro_metrics(cells)
    assert out["macro_delta_spearman"] is None
    assert out["primary_macro_status"] == "HOLD_PRIMARY_CELL_UNDEFINED"
    assert out["metric_cell_coverage"]["delta_spearman"]["undefined_cell_ids"] == ["hard"]


def test_holm_adjust_is_order_invariant_and_monotone() -> None:
    a = holm_adjust({"x": 0.01, "y": 0.04, "z": 0.03})
    b = holm_adjust({"z": 0.03, "x": 0.01, "y": 0.04})
    assert a == b
    assert a["x"] == pytest.approx(0.03)
    assert a["z"] == pytest.approx(0.06)
    assert a["y"] == pytest.approx(0.06)


def test_component_bootstrap_is_deterministic_and_clustered() -> None:
    y = np.array([1.0, 1.0, -1.0, -1.0, 2.0, 2.0])
    a = np.array([1.0, 1.0, -1.0, -1.0, 2.0, 2.0])
    b = -a
    comp = np.array(["a", "a", "b", "b", "c", "c"])
    one = paired_component_bootstrap(y, a, b, comp, metric="sign_accuracy",
                                     n_resamples=200, seed=23)
    two = paired_component_bootstrap(y, a, b, comp, metric="sign_accuracy",
                                     n_resamples=200, seed=23)
    assert one == two
    assert one["estimate_a_minus_b"] == pytest.approx(1.0)
    assert one["resampling_unit"] == "source_component"
    assert one["n_components"] == 3


def test_component_permutation_is_deterministic() -> None:
    y = np.array([1.0, -1.0, 2.0, -2.0])
    a = y.copy()
    b = -y
    comp = np.array(["a", "b", "c", "d"])
    one = paired_component_permutation(y, a, b, comp, metric="mae",
                                       n_permutations=500, seed=42)
    two = paired_component_permutation(y, a, b, comp, metric="mae",
                                       n_permutations=500, seed=42)
    assert one == two
    assert one["estimate_a_minus_b"] > 0.0
    assert one["orientation"] == "LARGER_UTILITY_IS_BETTER"
    assert one["gate_eligibility"] == "DEVELOPMENT_ROW_MICRO_DIAGNOSTIC_ONLY"
    assert 0.0 < one["p_value_two_sided"] <= 1.0


def test_exact_component_permutation_uses_exact_not_plus_one_probability() -> None:
    y = np.asarray([1.0, -1.0])
    a = y.copy()
    b = -y
    out = paired_component_permutation(
        y, a, b, ["a", "b"], metric="sign_accuracy", n_permutations=10
    )
    assert out["mode"] == "EXACT"
    assert out["n_permutations"] == 4
    assert out["p_value_two_sided"] == pytest.approx(0.5)


def test_seed_summary_does_not_claim_biological_replication() -> None:
    out = summarize_seeds({11: 0.4, 23: 0.6, 42: 0.5})
    assert out["mean"] == pytest.approx(0.5)
    assert out["range"] == pytest.approx([0.4, 0.6])
    assert out["inferential_unit"] == "NOT_BIOLOGICAL_REPLICATES"


def _paired_row(index: int, component: str, *, cell: str = "ctx") -> dict:
    return {
        "r3_input_row_sha256": f"{index + 1:064x}",
        "record_id": f"row_{index}",
        "study": "study",
        "estimand_id": "estimand",
        "context_id_v2": cell,
        "actual_edit_count": 1,
        "source_component_id": component,
        "delta": 1.0,
    }


def test_primary_paired_sign_is_component_then_cell_balanced_not_row_micro() -> None:
    rows = [_paired_row(index, "large") for index in range(100)]
    rows += [_paired_row(100, "small_a"), _paired_row(101, "small_b")]
    # Candidate wins every row of one large component but loses both small
    # components.  Row-micro says +0.96; the frozen component estimand says -1/3.
    candidate = np.asarray([1.0] * 100 + [-1.0, -1.0])
    baseline = -candidate
    out = paired_primary_sign_permutation(
        rows, candidate, baseline,
        n_permutations=10000, min_components_per_cell=2,
    )
    assert out["observed_oriented_improvement"] == pytest.approx(-1.0 / 3.0)
    assert out["permutation_unit"] == "GLOBAL_SOURCE_RELATION_COMPONENT"


def test_component_balanced_sign_reweights_after_excluding_true_zeros() -> None:
    # The large component contributes one informative sign row plus 99 zeros;
    # the small component contributes one informative sign row.  Each of the
    # two informative components must still receive half of the sign weight.
    y = np.asarray([1.0] + [0.0] * 99 + [1.0])
    pred = np.asarray([1.0] + [99.0] * 99 + [-1.0])
    components = ["large"] * 100 + ["small"]
    out = component_balanced_cell_metrics(y, pred, components)
    assert out["sign_accuracy"] == pytest.approx(0.5)


def test_rank_macro_weights_components_not_number_of_pools() -> None:
    y = [1.0, 2.0, 3.0] * 3
    pred = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0]
    pools = ["a1"] * 3 + ["a2"] * 3 + ["b1"] * 3
    components = ["component_a"] * 6 + ["component_b"] * 3
    out = within_pool_rank_metrics(y, pred, pools, components)
    assert out["component_spearman"] == pytest.approx({
        "component_a": 1.0,
        "component_b": -1.0,
    })
    assert out["spearman"] == pytest.approx(0.0)
    assert out["aggregation"].startswith("WITHIN_POOL_THEN_EQUAL_POOL_WITHIN_COMPONENT")


def test_structurally_registered_pool_is_not_dropped_for_label_variation() -> None:
    out = within_pool_rank_metrics(
        [1.0, 1.0, 1.0], [1.0, 2.0, 3.0],
        ["registered"] * 3, ["component"] * 3,
    )
    assert out["n_structurally_eligible_pools"] == 1
    assert out["spearman"] is None
    assert out["status"] == "HOLD_REGISTERED_POOL_UNDEFINED"
    assert out["pools"]["registered"]["status"] == "HOLD_REGISTERED_POOL_NO_TRUE_VARIATION"


def test_primary_paired_bootstrap_preserves_registered_cell_denominator() -> None:
    rows = []
    candidate = []
    baseline = []
    index = 0
    for cell in ("ctx_a", "ctx_b"):
        for component_index in range(3):
            rows.append(_paired_row(index, f"{cell}_c{component_index}", cell=cell))
            candidate.append(1.0)
            baseline.append(-1.0)
            index += 1
    first = paired_primary_sign_bootstrap(
        rows, candidate, baseline,
        n_resamples=50, seed=23, min_components_per_cell=2,
    )
    second = paired_primary_sign_bootstrap(
        rows, candidate, baseline,
        n_resamples=50, seed=23, min_components_per_cell=2,
    )
    assert first == second
    assert first["observed_oriented_improvement"] == pytest.approx(1.0)
    assert first["confidence_interval_oriented_improvement"] == pytest.approx([1.0, 1.0])
    assert first["registered_cell_denominator_fixed"] is True


def test_primary_paired_statistics_fail_closed_on_underpowered_cell() -> None:
    rows = [_paired_row(0, "only_a"), _paired_row(1, "only_b")]
    with pytest.raises(MetricProtocolError, match="frozen minimum"):
        paired_primary_sign_bootstrap(
            rows, [1.0, 1.0], [-1.0, -1.0],
            n_resamples=10, min_components_per_cell=3,
        )


def test_primary_component_swap_is_synchronized_across_cells() -> None:
    rows = [
        _paired_row(0, "shared_a", cell="ctx_a"),
        _paired_row(1, "shared_a", cell="ctx_b"),
        _paired_row(2, "shared_b", cell="ctx_a"),
        _paired_row(3, "shared_b", cell="ctx_b"),
    ]
    out = paired_primary_sign_permutation(
        rows, [1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0],
        min_components_per_cell=2,
    )
    assert out["mode"] == "EXACT"
    assert out["n_global_components"] == 2
    assert out["n_permutations"] == 4
    assert out["component_swap_is_synchronized_across_cells"] is True
