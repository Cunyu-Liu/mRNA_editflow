from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.r3x.baselines import (
    NEAREST_NEIGHBOR_SEARCH_POLICY,
    _StandardizedRidge,
    _component_cell_balanced_sample_weights,
    BaselineProtocolError,
    absolute_candidate_plus_anchor,
    anchor_only,
    candidate_kmer_ridge,
    edit_only_ridge,
    fit_optional_hist_gradient_boosting,
    foundation_baseline_status,
    group_balanced_prior,
    nearest_neighbor,
    paired_kmer_ridge,
    sign_logistic_diagnostic,
    zero_predictor,
)
from scripts.r3x.features import (
    _mean_window_kmer_frequencies,
    edit_only_features,
    kmer_frequencies,
    paired_kmer_features,
)


def row(i: int, *, delta: float, component: str, source_value: float = 1.0) -> dict:
    source = "ACGUACGU"
    pos = i % len(source)
    alt = {"A": "C", "C": "G", "G": "U", "U": "A"}[source[pos]]
    candidate = source[:pos] + alt + source[pos + 1 :]
    return {
        "record_id": f"r{i}",
        "source_sequence": source,
        "candidate_sequence": candidate,
        "edit_list": [{"op": "SUB", "pos": pos, "ref": source[pos], "alt": alt}],
        "actual_edit_count": 1,
        "delta": delta,
        "source_value": source_value,
        "candidate_value": source_value + delta,
        "source_component_id": component,
        "endpoint_id": "ep",
        "context_id_v2": "ctx",
        "region": "5UTR",
    }


def train_rows() -> list[dict]:
    return [
        row(0, delta=-2.0, component="a"),
        row(1, delta=-1.0, component="a"),
        row(2, delta=1.0, component="b"),
        row(3, delta=2.0, component="c"),
        row(4, delta=3.0, component="c"),
        row(5, delta=4.0, component="d"),
    ]


def test_zero_and_component_balanced_prior() -> None:
    train = train_rows()
    test = [row(8, delta=99.0, component="z")]
    assert zero_predictor(train, test).predictions.tolist() == [0.0]
    prior = group_balanced_prior(train, test)
    component_means = [-1.5, 1.0, 2.5, 4.0]
    assert prior.predictions[0] == pytest.approx(np.mean(component_means))
    assert prior.information_set == "NO_ANCHOR_ZERO_PRIOR"


def test_predictions_never_read_test_delta() -> None:
    train = train_rows()
    test = [row(8, delta=9999.0, component="z"), row(9, delta=-9999.0, component="y")]
    changed = copy.deepcopy(test)
    for r in changed:
        r["delta"] *= -123.0
        r["candidate_value"] *= 77.0
    for fn in (zero_predictor, group_balanced_prior, edit_only_ridge,
               candidate_kmer_ridge, paired_kmer_ridge, nearest_neighbor):
        a = fn(train, test).predictions
        b = fn(train, changed).predictions
        assert np.allclose(a, b), fn.__name__


def test_edit_and_pair_features_have_stable_dimensions() -> None:
    r = train_rows()[0]
    assert edit_only_features(r).ndim == 1
    assert paired_kmer_features(r, max_k=2).shape == paired_kmer_features(r, max_k=2).shape
    broken = copy.deepcopy(r)
    broken["edit_list"][0]["ref"] = "U"
    # EDIT_ONLY consumes only the frozen edit declaration; exact replay is a
    # separate row-qualification invariant, not a hidden sequence feature.
    assert np.all(np.isfinite(edit_only_features(broken)))
    with pytest.raises(ValueError, match="replay"):
        paired_kmer_features(broken)


def test_local_kmers_do_not_cross_between_distant_edit_windows() -> None:
    separated = _mean_window_kmer_frequencies(("A", "C"), max_k=2)
    concatenated = kmer_frequencies("AC", max_k=2)
    # The 2-mer block follows two summary fields and four 1-mer frequencies.
    assert np.count_nonzero(separated[6:]) == 0
    assert np.count_nonzero(concatenated[6:]) == 1
    assert np.array_equal(
        separated,
        _mean_window_kmer_frequencies(("C", "A"), max_k=2),
    )


def test_features_accept_canonical_r3_preparation_field_names() -> None:
    canonical = {
        "normalized_source_sequence": "AAAA",
        "normalized_candidate_sequence": "CAAA",
        "edit_list": [{"op": "SUB", "pos": 0, "source_nt": "A", "target_nt": "C"}],
        "edit_count": 1,
    }
    assert np.all(np.isfinite(edit_only_features(canonical)))
    assert np.all(np.isfinite(paired_kmer_features(canonical, max_k=2)))


def test_edit_only_features_are_invariant_to_nonallowlisted_sequences():
    short = train_rows()[0]
    long = copy.deepcopy(short)
    long["source_sequence"] = "A" * 400
    long["candidate_sequence"] = "C" + "A" * 399
    long["edit_list"] = [{"op": "SUB", "pos": 0, "ref": "A", "alt": "C"}]
    assert np.array_equal(edit_only_features(short), edit_only_features(long))


def test_ridge_baselines_are_deterministic_and_finite() -> None:
    train = train_rows()
    test = [row(8, delta=0.0, component="z"), row(9, delta=0.0, component="y")]
    for fn in (edit_only_ridge, candidate_kmer_ridge, paired_kmer_ridge):
        a = fn(train, test)
        b = fn(train, test)
        assert a.status == "RUN"
        assert np.all(np.isfinite(a.predictions))
        assert np.allclose(a.predictions, b.predictions)


def test_ridge_uses_the_smaller_feature_system_at_large_n() -> None:
    rng = np.random.default_rng(19)
    features = rng.normal(size=(2048, 12))
    target = rng.normal(size=2048)
    model = _StandardizedRidge(1.0).fit(features, target)
    assert model.solver_ == "PRIMAL_SMALLER_FEATURE_SYSTEM"
    assert model.square_system_dimension_ == 12
    assert model.square_system_dimension_ != len(features)
    assert np.all(np.isfinite(model.predict(features[:7])))


def test_ridge_uses_dual_only_when_sample_system_is_smaller() -> None:
    rng = np.random.default_rng(23)
    features = rng.normal(size=(7, 31))
    target = rng.normal(size=7)
    model = _StandardizedRidge(1.0).fit(features, target)
    assert model.solver_ == "DUAL_SMALLER_SAMPLE_SYSTEM"
    assert model.square_system_dimension_ == 7


def test_nearest_neighbor_exact_predictions_are_chunk_size_invariant() -> None:
    train = train_rows()
    test = [row(8, delta=0.0, component="z"), row(9, delta=0.0, component="y")]
    single = nearest_neighbor(train, test, train_chunk_size=1)
    blocked = nearest_neighbor(train, test, train_chunk_size=4)
    assert np.array_equal(single.predictions, blocked.predictions)
    assert single.metadata["search"] == NEAREST_NEIGHBOR_SEARCH_POLICY
    with pytest.raises(BaselineProtocolError, match="chunk_size"):
        nearest_neighbor(train, test, train_chunk_size=0)


def test_nearest_neighbor_tie_mean_is_bitwise_chunk_invariant_under_cancellation() -> None:
    values = [1e16, 1.0, -1e16, 1.0]
    train = []
    for index, value in enumerate(values):
        item = row(0, delta=value, component=f"component_{index}")
        item["record_id"] = f"tie_{index}"
        train.append(item)
    test = [row(0, delta=0.0, component="held")]
    predictions = [
        nearest_neighbor(train, test, train_chunk_size=size).predictions[0]
        for size in (1, 2, 4)
    ]
    assert predictions[0] == predictions[1] == predictions[2]
    assert predictions[0] == pytest.approx(0.5)


def test_component_cell_balanced_weights_equalize_cells_and_components() -> None:
    cells = ["study_a:estimand:context:1"] * 9 + [
        "study_b:estimand:context:1"
    ]
    components = [f"a_{index}" for index in range(9)] + ["b_0"]
    weights = _component_cell_balanced_sample_weights(cells, components)
    assert np.sum(weights[:9]) == pytest.approx(np.sum(weights[9:]))
    assert np.sum(weights) == pytest.approx(len(cells))
    with pytest.raises(BaselineProtocolError, match="align"):
        _component_cell_balanced_sample_weights(cells, components[:-1])


def test_baseline_rejects_conflicting_condition_aliases() -> None:
    train = train_rows()
    train[0]["publication_context_id"] = "different"
    with pytest.raises(BaselineProtocolError, match="context aliases conflict"):
        paired_kmer_ridge(train, [row(8, delta=0.0, component="z")])


def test_direct_and_anchor_assisted_live_on_separate_information_sets() -> None:
    train = train_rows()
    test = [row(8, delta=0.0, component="z", source_value=10.0)]
    direct = candidate_kmer_ridge(train, test)
    anch = absolute_candidate_plus_anchor(train, test)
    only = anchor_only(train, test)
    assert direct.information_set == "CANDIDATE_ONLY"
    assert anch.information_set == "MEASURED_SOURCE_ANCHOR_ASSISTED"
    assert only.information_set == "MEASURED_SOURCE_ANCHOR_ASSISTED"
    assert anch.leaderboard != direct.leaderboard


def test_anchor_assisted_refuses_missing_measured_anchor() -> None:
    train = train_rows()
    test = [row(8, delta=0.0, component="z")]
    test[0]["source_value"] = None
    with pytest.raises(BaselineProtocolError, match="source_value"):
        absolute_candidate_plus_anchor(train, test)


def test_sign_logistic_is_diagnostic_not_gate_substitute() -> None:
    out = sign_logistic_diagnostic(train_rows(), [row(8, delta=0.0, component="z")])
    assert out.role == "DIAGNOSTIC_ONLY_NOT_EFFECT_GATE"
    assert out.information_set == "EDIT_ONLY"
    assert out.predictions.shape == (1,)


def test_optional_baselines_are_explicit_about_not_run() -> None:
    hist = fit_optional_hist_gradient_boosting(train_rows(), [row(8, delta=0.0, component="z")])
    assert hist.status in {"RUN", "NOT_RUN_WITH_REASON"}
    unbound_weights = fit_optional_hist_gradient_boosting(
        train_rows(),
        [row(8, delta=0.0, component="z")],
        execution_authorized=True,
        minimum_train_rows=1,
    )
    assert unbound_weights.status == "NOT_RUN_WITH_REASON"
    assert "estimand-cell" in unbound_weights.reason
    fm = foundation_baseline_status(exposure_qualified=False, feature_path=None)
    assert fm.status == "NOT_RUN_WITH_REASON"
    assert "exposure" in fm.reason.lower()


def test_baseline_rejects_study_as_feature_or_missing_component() -> None:
    train = train_rows()
    train[0]["source_component_id"] = ""
    with pytest.raises(BaselineProtocolError, match="source_component"):
        group_balanced_prior(train, [row(8, delta=0.0, component="z")])
