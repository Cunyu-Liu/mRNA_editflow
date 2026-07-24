"""Tests for position-aware oracle models (SeqDiffModel, SeqCNNModel).

Verifies that the new models can learn position-specific edit effects,
breaking the constant-predictor degeneracy of the old global-feature models.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO.parent))

from core.p3_02_delta_oracle import (
    MODEL_REGISTRY,
    SeqDiffModel,
    SeqCNNModel,
    SeqLinearModel,
    extract_features,
    batch_extract_features,
    cross_fit_predict,
    CrossFitConfig,
    DeltaRecord,
)


def _make_synthetic_dataset(
    n_sources=20, seq_len=40, n_edits_per_source=10, seed=0
):
    """Create a dataset where delta depends on edit position.

    Positions 0-9: positive delta (+0.05 to +0.01, decreasing)
    Positions 10-19: near-zero delta
    Positions 20-39: negative delta (-0.01 to -0.05)
    """
    rng = np.random.RandomState(seed)
    NUCS = "ACGU"
    records = []
    for s in range(n_sources):
        src_seq = "".join(rng.choice(list(NUCS)) for _ in range(seq_len))
        for _ in range(n_edits_per_source):
            pos = rng.randint(0, seq_len)
            ref = src_seq[pos]
            alt = rng.choice([c for c in NUCS if c != ref])
            cand_seq = src_seq[:pos] + alt + src_seq[pos+1:]
            # Position-dependent delta
            if pos < 10:
                delta = 0.05 * (1.0 - pos / 10.0) + rng.randn() * 0.01
            elif pos < 20:
                delta = rng.randn() * 0.005
            else:
                delta = -0.01 * (pos - 19) / 20.0 + rng.randn() * 0.01
            records.append(DeltaRecord(
                record_id=f"rec_{s}_{len(records)}",
                source_id=f"src_{s}",
                source_sequence=src_seq,
                candidate_sequence=cand_seq,
                delta=float(delta),
                edit_count=1,
                edit_type="single_sub",
                edited_region="five_utr",
                source_value=None,
                candidate_value=None,
                value_std=None,
                confidence="measured",
                edit_list=[{"pos": pos, "ref": ref, "alt": alt, "region": "five_utr"}],
                split_role="train" if s < 16 else "val",
                family_cluster_id=f"fam_{s % 4}",
            ))
    return records


class TestModelRegistry:
    def test_seq_diff_registered(self):
        assert "seq_diff" in MODEL_REGISTRY
        assert MODEL_REGISTRY["seq_diff"] is SeqDiffModel

    def test_seq_cnn_registered(self):
        assert "seq_cnn" in MODEL_REGISTRY
        assert MODEL_REGISTRY["seq_cnn"] is SeqCNNModel


class TestSeqDiffModel:
    def test_fit_predict_shape(self):
        records = _make_synthetic_dataset(n_sources=10, n_edits_per_source=5)
        feats = batch_extract_features(records, max_seq_len=50)
        labels = feats["delta"]
        model = SeqDiffModel(hidden_dim=32, n_epochs=50, seed=42)
        model.fit(feats, labels)
        preds = model.predict_delta(feats)
        assert preds.shape == (len(records),)

    def test_learns_position_specific_signal(self):
        """The model must produce different predictions for edits at
        different positions (position sensitivity > 1e-3)."""
        records = _make_synthetic_dataset(n_sources=20, n_edits_per_source=15, seed=1)
        feats = batch_extract_features(records, max_seq_len=50)
        labels = feats["delta"]
        model = SeqDiffModel(hidden_dim=64, n_epochs=100, seed=42)
        model.fit(feats, labels)
        # Predict on all single-edit candidates from one source
        src_seq = records[0].source_sequence
        src_oh = extract_features(src_seq, src_seq, [], 50)["source_onehot"]
        preds_at_pos = []
        for pos in range(len(src_seq)):
            ref = src_seq[pos]
            alt = "G" if ref != "G" else "A"
            cand = src_seq[:pos] + alt + src_seq[pos+1:]
            f = extract_features(src_seq, cand, [{"pos": pos, "ref": ref, "alt": alt, "region": "five_utr"}], 50)
            batch = {k: v[np.newaxis] for k, v in f.items()}
            pred = float(model.predict_delta(batch)[0])
            preds_at_pos.append(pred)
        preds_arr = np.array(preds_at_pos)
        pos_sensitivity = float(np.std(preds_arr))
        assert pos_sensitivity > 1e-3, (
            f"Position sensitivity {pos_sensitivity:.6f} < 1e-3; "
            "model is still a constant predictor"
        )


class TestSeqCNNModel:
    def test_fit_predict_shape(self):
        records = _make_synthetic_dataset(n_sources=10, n_edits_per_source=5)
        feats = batch_extract_features(records, max_seq_len=50)
        labels = feats["delta"]
        model = SeqCNNModel(n_filters=16, kernel_size=3, n_epochs=50, seed=42)
        model.fit(feats, labels)
        preds = model.predict_delta(feats)
        assert preds.shape == (len(records),)

    def test_learns_position_specific_signal(self):
        records = _make_synthetic_dataset(n_sources=20, n_edits_per_source=15, seed=1)
        feats = batch_extract_features(records, max_seq_len=50)
        labels = feats["delta"]
        model = SeqCNNModel(n_filters=32, kernel_size=5, n_epochs=100, seed=42)
        model.fit(feats, labels)
        src_seq = records[0].source_sequence
        preds_at_pos = []
        for pos in range(len(src_seq)):
            ref = src_seq[pos]
            alt = "G" if ref != "G" else "A"
            cand = src_seq[:pos] + alt + src_seq[pos+1:]
            f = extract_features(src_seq, cand, [{"pos": pos, "ref": ref, "alt": alt, "region": "five_utr"}], 50)
            batch = {k: v[np.newaxis] for k, v in f.items()}
            pred = float(model.predict_delta(batch)[0])
            preds_at_pos.append(pred)
        preds_arr = np.array(preds_at_pos)
        pos_sensitivity = float(np.std(preds_arr))
        assert pos_sensitivity > 1e-3, (
            f"Position sensitivity {pos_sensitivity:.6f} < 1e-3; "
            "model is still a constant predictor"
        )


class TestSeqLinearModel:
    def test_fit_predict_shape(self):
        records = _make_synthetic_dataset(n_sources=10, n_edits_per_source=5)
        feats = batch_extract_features(records, max_seq_len=50)
        labels = feats["delta"]
        model = SeqLinearModel(seed=42)
        model.fit(feats, labels)
        preds = model.predict_delta(feats)
        assert preds.shape == (len(records),)

    def test_learns_position_specific_signal(self):
        records = _make_synthetic_dataset(n_sources=20, n_edits_per_source=15, seed=1)
        feats = batch_extract_features(records, max_seq_len=50)
        labels = feats["delta"]
        model = SeqLinearModel(seed=42)
        model.fit(feats, labels)
        src_seq = records[0].source_sequence
        preds_at_pos = []
        for pos in range(len(src_seq)):
            ref = src_seq[pos]
            alt = "G" if ref != "G" else "A"
            cand = src_seq[:pos] + alt + src_seq[pos+1:]
            f = extract_features(src_seq, cand, [{"pos": pos, "ref": ref, "alt": alt, "region": "five_utr"}], 50)
            batch = {k: v[np.newaxis] for k, v in f.items()}
            pred = float(model.predict_delta(batch)[0])
            preds_at_pos.append(pred)
        preds_arr = np.array(preds_at_pos)
        pos_sensitivity = float(np.std(preds_arr))
        assert pos_sensitivity > 1e-3, (
            f"Position sensitivity {pos_sensitivity:.6f} < 1e-3; "
            "model is still a constant predictor"
        )


class TestCrossFitIntegration:
    def test_cross_fit_with_seq_diff(self):
        """SeqDiffModel works with cross_fit_predict."""
        records = _make_synthetic_dataset(n_sources=15, n_edits_per_source=10)
        feats = batch_extract_features(records, max_seq_len=50)
        labels = feats["delta"]
        n = len(labels)
        rng = np.random.RandomState(0)
        idx = np.arange(n)
        rng.shuffle(idx)
        fold_size = n // 3
        folds = [idx[:fold_size], idx[fold_size:2*fold_size], idx[2*fold_size:]]
        config = CrossFitConfig(n_folds=3, hidden_dim=32, n_epochs=30, seed=42)
        oof, models = cross_fit_predict(
            SeqDiffModel, feats, labels, folds, config
        )
        assert oof.shape == (n,)
        assert len(models) == 3
        assert not np.any(np.isnan(oof))

    def test_cross_fit_with_seq_cnn(self):
        """SeqCNNModel works with cross_fit_predict."""
        records = _make_synthetic_dataset(n_sources=15, n_edits_per_source=10)
        feats = batch_extract_features(records, max_seq_len=50)
        labels = feats["delta"]
        n = len(labels)
        rng = np.random.RandomState(0)
        idx = np.arange(n)
        rng.shuffle(idx)
        fold_size = n // 3
        folds = [idx[:fold_size], idx[fold_size:2*fold_size], idx[2*fold_size:]]
        config = CrossFitConfig(n_folds=3, hidden_dim=32, n_epochs=30, seed=42)
        oof, models = cross_fit_predict(
            SeqCNNModel, feats, labels, folds, config
        )
        assert oof.shape == (n,)
        assert len(models) == 3
        assert not np.any(np.isnan(oof))
