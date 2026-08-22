from __future__ import annotations

from core.route2_xeditsetflow_runtime_v3 import build_setflow_arm_v3, early_stop_update_v3


def _vocabs():
    return {
        "assay": {"__UNK__": 0, "a": 1},
        "context": {"__UNK__": 0, "c": 1},
        "quantity": {"__UNK__": 0, "q": 1},
        "measurement": {"__UNK__": 0, "m": 1},
        "numerator": {"__UNK__": 0, "n": 1},
        "denominator": {"__UNK__": 0, "d": 1},
    }


def test_f1_is_original_small_trunk_and_not_selectable() -> None:
    model, config = build_setflow_arm_v3("f1", vocabs=_vocabs(), dropout=0.1)
    assert config["hidden_dim"] == 256 and config["depth"] == 2
    assert config["selectable"] is False
    assert 700_000 <= config["trainable_parameter_count"] <= 1_200_000
    assert model.position_progress_features is True


def test_f2_f3_exact_frozen_geometry_and_capacity_order() -> None:
    _f2, f2 = build_setflow_arm_v3("f2", vocabs=_vocabs(), dropout=0.1)
    _f3, f3 = build_setflow_arm_v3("f3", vocabs=_vocabs(), dropout=0.1)
    assert (f2["model_width"], f2["depth"], f2["ffn_width"]) == (384, 8, 1536)
    assert (f3["model_width"], f3["depth"], f3["ffn_width"]) == (512, 12, 2048)
    assert 15_000_000 <= f2["trainable_parameter_count"] < f3["trainable_parameter_count"] <= 46_000_000


def test_early_stopping_is_strict_common_nll_patience_two() -> None:
    improved, best, stale, stop = early_stop_update_v3(2.0, best=None, stale_passes=0, patience=2)
    assert (improved, best, stale, stop) == (True, 2.0, 0, False)
    improved, best, stale, stop = early_stop_update_v3(2.1, best=best, stale_passes=stale, patience=2)
    assert (improved, best, stale, stop) == (False, 2.0, 1, False)
    improved, best, stale, stop = early_stop_update_v3(2.0, best=best, stale_passes=stale, patience=2)
    assert (improved, best, stale, stop) == (False, 2.0, 2, True)
