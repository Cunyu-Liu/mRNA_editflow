from __future__ import annotations

import pytest

from core.route2_xeditcritic_structure_features_v5 import (
    LOCAL_PAIRING_WINDOW_V5,
    STRUCTURE_FEATURE_WIDTH_V5,
    gc_content,
    structure_differential_features_v5,
    structure_feature_matrix_v5,
)

try:
    import RNA  # noqa: F401

    VIENNARNA_AVAILABLE = True
except ImportError:  # pragma: no cover - server always has ViennaRNA 2.7.2
    VIENNARNA_AVAILABLE = False


def _structured_pair() -> tuple[str, str]:
    # A source whose MFE structure pairs around the middle, and a candidate
    # whose substitution breaks that pairing.
    source = "GGGAAACCCGGGAAACC"
    candidate = "GGGAAACCCGGGAAACA"
    return source, candidate


def test_gc_content_basic_and_sanitization():
    assert gc_content("GGAACC") == pytest.approx(2 / 3)
    assert gc_content("ggaatt") == pytest.approx(1 / 3)
    assert gc_content("GGAA TT") == pytest.approx(1 / 3)
    with pytest.raises(Exception, match="non-ACGU"):
        gc_content("GGN")


@pytest.mark.skipif(not VIENNARNA_AVAILABLE, reason="ViennaRNA absent")
def test_feature_vector_width_and_finiteness():
    source, candidate = _structured_pair()
    features = structure_differential_features_v5(source, candidate)
    vector = features.to_vector()
    assert len(vector) == STRUCTURE_FEATURE_WIDTH_V5
    assert all(value == value for value in vector)  # no NaN
    assert all(abs(value) != float("inf") for value in vector)
    assert features.edit_count == 1.0
    assert features.delta_mfe != 0.0 or features.delta_ensemble_energy != 0.0


@pytest.mark.skipif(not VIENNARNA_AVAILABLE, reason="ViennaRNA absent")
def test_identical_pair_has_zero_deltas():
    source, _ = _structured_pair()
    features = structure_differential_features_v5(source, source)
    assert features.delta_mfe == 0.0
    assert features.delta_ensemble_energy == 0.0
    assert features.edit_site_pairing_delta_mean == 0.0
    assert features.delta_gc == 0.0
    assert features.edit_count == 0.0


@pytest.mark.skipif(not VIENNARNA_AVAILABLE, reason="ViennaRNA absent")
def gc_pairing_break_detects_structure_change():
    source, candidate = _structured_pair()
    features = structure_differential_features_v5(source, candidate)
    # Breaking a G-C pair at the tail must change pairing somewhere.
    assert features.local_pairing_delta_max > 0.0
    assert features.local_pairing_delta_mean >= features.edit_site_pairing_delta_mean * 0.5


@pytest.mark.skipif(not VIENNARNA_AVAILABLE, reason="ViennaRNA absent")
def test_matrix_rows_align_with_pair_count_and_width():
    source, candidate = _structured_pair()
    matrix = structure_feature_matrix_v5(
        [(source, candidate), (candidate, source), (source, source)]
    )
    assert len(matrix) == 3
    assert all(len(row) == STRUCTURE_FEATURE_WIDTH_V5 for row in matrix)
    reversed_features = matrix[1]
    forward_features = matrix[0]
    assert reversed_features[2] == pytest.approx(-forward_features[2])


def test_length_mismatch_and_invalid_bases_rejected():
    source, candidate = _structured_pair()
    with pytest.raises(Exception, match="equal-length"):
        structure_differential_features_v5(source, candidate + "A")
    with pytest.raises(Exception, match="non-ACGU"):
        structure_differential_features_v5(source, candidate.replace("A", "N"))


def test_local_window_constant_is_frozen():
    assert LOCAL_PAIRING_WINDOW_V5 == 4
