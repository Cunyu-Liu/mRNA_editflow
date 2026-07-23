"""Unit tests for P3-02 Local-Delta Oracle module.

Tests cover:
- Data loading and DeltaRecord schema
- Feature extraction (sequence, edit, one-hot)
- 4 model architectures (absolute, difference, siamese, edit-conditioned)
- Local-delta metrics (sign accuracy, top-k enrichment, etc.)
- Cross-fitted ensemble
- Region sensitivity perturbation
- Headroom search (exact, greedy, beam, SA, MCTS)
- GO/NO-GO gate evaluation
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.p3_02_delta_oracle import (
    DeltaRecord, load_benchmark_tier, load_benchmark,
    extract_features, batch_extract_features,
    AbsoluteModel, DifferenceModel, SiameseModel, EditConditionedModel,
    MODEL_REGISTRY,
    compute_all_metrics, sign_accuracy, top_k_enrichment,
    beneficial_edit_precision, delta_spearman, delta_pearson,
    pairwise_ranking_auc, calibration_error,
    CrossFitConfig, cross_fit_predict, build_oracle_ensemble,
    run_region_sensitivity, analyze_sensitivity_checks,
    run_headroom_search, analyze_headroom,
    exact_one_edit_enumeration, greedy_search, beam_search,
    simulated_annealing, mcts_search, oracle_guided_search,
    evaluate_go_gate,
    NUC_VOCAB, START_CODON,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(idx: int, src: str = "", cand: str = "", delta: float = 0.0,
                 split: str = "train", edits: list = None) -> DeltaRecord:
    if not src:
        src = "ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUAC"
    if not cand:
        cand = src
    if edits is None:
        edits = []
    return DeltaRecord(
        record_id=f"test_{idx}",
        source_id=f"src_{idx // 5}",
        source_sequence=src,
        candidate_sequence=cand,
        edit_list=edits,
        edit_count=len(edits),
        edited_region="five_utr",
        delta=delta,
        source_value=5.0,
        candidate_value=5.0 + delta,
        value_std=0.3,
        confidence="measured",
        split_role=split,
        family_cluster_id=f"fam_{idx // 10}",
        edit_type="measured_single" if edits else "wild_type_anchor",
    )


def _make_synthetic_records(n: int = 100, seed: int = 42) -> list:
    """Generate synthetic records with position-dependent delta."""
    import random
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    records = []
    for i in range(n):
        src = "".join(rng.choice(list(NUC_VOCAB)) for _ in range(50))
        pos = rng.randint(0, len(src) - 1)
        old = src[pos]
        new_nt = rng.choice([c for c in NUC_VOCAB if c != old])
        cand = src[:pos] + new_nt + src[pos + 1:]
        edits = [{"pos": pos, "ref": old, "alt": new_nt, "region": "five_utr"}]
        # Delta depends on position (early = more effect) and GC change
        pos_effect = 1.0 - pos / 50.0
        gc_change = 1.0 if new_nt in "GC" and old in "AU" else -0.5
        delta = pos_effect * gc_change * 0.3 + np_rng.randn() * 0.1
        split = "train" if i < 70 else ("val" if i < 85 else "test")
        records.append(_make_record(i, src, cand, delta, split, edits))
    return records


# ---------------------------------------------------------------------------
# Test: Data Loading
# ---------------------------------------------------------------------------

class TestDeltaRecord:
    def test_record_creation(self):
        rec = _make_record(0)
        assert rec.record_id == "test_0"
        assert rec.confidence == "measured"
        assert rec.split_role == "train"

    def test_load_benchmark_tier(self):
        """Test loading from a temporary JSONL file."""
        rec = _make_record(0, delta=0.5, edits=[{"pos": 0, "ref": "A", "alt": "G", "region": "five_utr"}])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({
                "record_id": rec.record_id,
                "source_id": rec.source_id,
                "source_sequence": rec.source_sequence,
                "candidate_sequence": rec.candidate_sequence,
                "edit_list": rec.edit_list,
                "edit_count": rec.edit_count,
                "edited_region": rec.edited_region,
                "delta": rec.delta,
                "measured_or_proxy_source_value": rec.source_value,
                "measured_or_proxy_candidate_value": rec.candidate_value,
                "value_std": rec.value_std,
                "confidence": rec.confidence,
                "split_role": rec.split_role,
                "family_cluster_id": rec.family_cluster_id,
                "edit_type": rec.edit_type,
            }) + "\n")
            # Add an unlabeled record (should be skipped)
            f.write(json.dumps({
                "record_id": "skip_me",
                "source_id": "src",
                "source_sequence": "ACGU",
                "candidate_sequence": "ACGU",
                "edit_list": [],
                "edit_count": 0,
                "edited_region": "five_utr",
                "delta": None,
                "measured_or_proxy_source_value": None,
                "measured_or_proxy_candidate_value": None,
                "value_std": None,
                "confidence": "unlabeled",
                "split_role": "train",
                "family_cluster_id": "fam",
                "edit_type": "wild_type_anchor",
            }) + "\n")
            f.flush()
            loaded = load_benchmark_tier(f.name)
            os.unlink(f.name)

        assert len(loaded) == 1
        assert loaded[0].delta == 0.5


# ---------------------------------------------------------------------------
# Test: Feature Extraction
# ---------------------------------------------------------------------------

class TestFeatureExtraction:
    def test_extract_features_shapes(self):
        feats = extract_features("ACGUACGUAC", "ACGUACGUAG",
                                 [{"pos": 9, "ref": "C", "alt": "G", "region": "five_utr"}])
        assert feats["source_onehot"].shape == (100, 4)
        assert feats["candidate_onehot"].shape == (100, 4)
        assert feats["source_feat"].shape[0] >= 20  # at least 20 features
        assert feats["candidate_feat"].shape[0] >= 20
        assert feats["diff_feat"].shape[0] >= 20
        assert feats["edit_feat"].shape == (12,)

    def test_diff_features_nonzero_when_different(self):
        feats = extract_features("AAAA", "GGGG",
                                 [{"pos": 0, "ref": "A", "alt": "G", "region": "five_utr"}])
        assert np.any(feats["diff_feat"] != 0)

    def test_diff_features_zero_when_identical(self):
        feats = extract_features("ACGU", "ACGU", [])
        assert np.allclose(feats["diff_feat"], 0.0)

    def test_batch_extract(self):
        records = _make_synthetic_records(20)
        batch = batch_extract_features(records)
        assert batch["source_feat"].shape[0] == 20
        assert batch["delta"].shape == (20,)


# ---------------------------------------------------------------------------
# Test: Model Architectures
# ---------------------------------------------------------------------------

class TestModels:
    def test_absolute_model(self):
        records = _make_synthetic_records(100)
        features = batch_extract_features(records)
        model = AbsoluteModel(hidden_dim=32, n_epochs=50, seed=42)
        model.fit(features, features["source_value"] + features["delta"])
        pred = model.predict_delta(features)
        assert pred.shape == (100,)
        assert np.all(np.isfinite(pred))

    def test_difference_model(self):
        records = _make_synthetic_records(100)
        features = batch_extract_features(records)
        model = DifferenceModel(hidden_dim=32, n_epochs=50, seed=42)
        model.fit(features, features["delta"])
        pred = model.predict_delta(features)
        assert pred.shape == (100,)
        assert np.all(np.isfinite(pred))

    def test_siamese_model(self):
        records = _make_synthetic_records(100)
        features = batch_extract_features(records)
        model = SiameseModel(hidden_dim=32, n_epochs=50, seed=42)
        model.fit(features, features["delta"])
        pred = model.predict_delta(features)
        assert pred.shape == (100,)
        assert np.all(np.isfinite(pred))

    def test_edit_conditioned_model(self):
        records = _make_synthetic_records(100)
        features = batch_extract_features(records)
        model = EditConditionedModel(hidden_dim=32, n_epochs=50, seed=42)
        model.fit(features, features["delta"])
        pred = model.predict_delta(features)
        assert pred.shape == (100,)
        assert np.all(np.isfinite(pred))

    def test_model_registry(self):
        assert "absolute" in MODEL_REGISTRY
        assert "difference" in MODEL_REGISTRY
        assert "siamese" in MODEL_REGISTRY
        assert "edit_conditioned" in MODEL_REGISTRY


# ---------------------------------------------------------------------------
# Test: Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_sign_accuracy_perfect(self):
        pred = np.array([1.0, -1.0, 2.0, -0.5])
        true = np.array([0.5, -0.3, 1.0, -0.2])
        assert sign_accuracy(pred, true) == 1.0

    def test_sign_accuracy_random(self):
        np.random.seed(42)
        pred = np.random.randn(1000)
        true = np.random.randn(1000)
        sa = sign_accuracy(pred, true)
        assert 0.4 < sa < 0.6  # random ~0.5

    def test_sign_accuracy_zero_true_delta(self):
        """Zero true deltas should be excluded."""
        pred = np.array([1.0, -1.0, 0.5])
        true = np.array([0.0, 0.0, 0.3])
        sa = sign_accuracy(pred, true)
        assert sa == 1.0  # only the non-zero true delta

    def test_top_k_enrichment(self):
        # Perfect ranking: all top-k are beneficial
        pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
        true = np.array([5.0, 4.0, 3.0, 2.0, 1.0, -1.0, -2.0, -3.0, -4.0, -5.0])
        enrichment = top_k_enrichment(pred, true, k=0.2)
        assert enrichment > 1.0  # top-k enriched

    def test_top_k_enrichment_no_signal(self):
        np.random.seed(42)
        pred = np.random.randn(100)
        true = np.random.randn(100)
        enrichment = top_k_enrichment(pred, true, k=0.1)
        assert 0.5 < enrichment < 2.0  # near 1.0 (no enrichment)

    def test_beneficial_edit_precision(self):
        pred = np.array([1.0, 1.0, 1.0, -1.0])
        true = np.array([0.5, 0.3, -0.1, 0.2])
        # pred>0: indices 0,1,2; true>0: indices 0,1 → precision = 2/3
        prec = beneficial_edit_precision(pred, true)
        assert abs(prec - 2.0 / 3.0) < 1e-6

    def test_delta_pearson(self):
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        true = np.array([1.0, 2.0, 3.0, 4.0])
        assert abs(delta_pearson(pred, true) - 1.0) < 1e-6

    def test_delta_spearman(self):
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        true = np.array([1.0, 2.0, 3.0, 4.0])
        assert abs(delta_spearman(pred, true) - 1.0) < 1e-6

    def test_pairwise_ranking_auc(self):
        pred = np.array([4.0, 3.0, 2.0, 1.0])
        true = np.array([4.0, 3.0, 2.0, 1.0])
        auc = pairwise_ranking_auc(pred, true, n_pairs=100, seed=42)
        assert auc > 0.9  # near perfect

    def test_calibration_error(self):
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        true = np.array([1.0, 2.0, 3.0, 4.0])
        ce = calibration_error(pred, true)
        assert ce < 0.01  # perfectly calibrated

    def test_compute_all_metrics(self):
        pred = np.array([1.0, -0.5, 2.0, -1.0])
        true = np.array([0.8, -0.3, 1.5, -0.8])
        src_vals = np.array([5.0, 5.0, 5.0, 5.0])
        metrics = compute_all_metrics(pred, true, src_vals)
        assert "sign_accuracy" in metrics
        assert "top_k_enrichment_10pct" in metrics
        assert "beneficial_edit_precision" in metrics
        assert "delta_spearman" in metrics
        assert "source_normalized_rmse" in metrics
        assert metrics["sign_accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Test: Cross-Fitted Ensemble
# ---------------------------------------------------------------------------

class TestCrossFitEnsemble:
    def test_cross_fit_predict(self):
        records = _make_synthetic_records(100)
        features = batch_extract_features(records)
        config = CrossFitConfig(n_folds=3, n_epochs=30, hidden_dim=32)
        # Create simple folds
        n = len(records)
        fold_size = n // 3
        folds = [np.arange(i * fold_size, (i + 1) * fold_size) for i in range(3)]

        oof, models = cross_fit_predict(DifferenceModel, features, features["delta"], folds, config)
        assert oof.shape == (n,)
        assert len(models) == 3
        assert np.all(np.isfinite(oof))

    def test_build_oracle_ensemble(self):
        records = _make_synthetic_records(100)
        features = batch_extract_features(records)
        config = CrossFitConfig(n_folds=3, n_epochs=30, hidden_dim=32)
        n = len(records)
        fold_size = n // 3
        folds = [np.arange(i * fold_size, (i + 1) * fold_size) for i in range(3)]

        result = build_oracle_ensemble(
            features, features["delta"], folds, config,
            model_names=("difference", "siamese"),
        )
        assert "ensemble_pred" in result
        assert "ensemble_uncertainty" in result
        assert "disagreement" in result
        assert result["ensemble_pred"].shape == (n,)
        assert result["ensemble_uncertainty"].shape == (n,)
        assert len(result["per_model_oof"]) == 2


# ---------------------------------------------------------------------------
# Test: Region Sensitivity
# ---------------------------------------------------------------------------

class TestRegionSensitivity:
    def test_run_region_sensitivity(self):
        records = _make_synthetic_records(50)
        features = batch_extract_features(records)
        config = CrossFitConfig(n_folds=3, n_epochs=20, hidden_dim=16)

        # Train a simple model
        model = DifferenceModel(hidden_dim=16, n_epochs=20, seed=42)
        model.fit(features, features["delta"])

        source_seqs = [r.source_sequence for r in records[:20]]
        results = run_region_sensitivity(model, source_seqs, config, n_samples=10, seed=42)

        assert "five_utr_single_sub" in results
        assert "matched_random" in results
        assert results["five_utr_single_sub"].n_samples > 0

    def test_analyze_sensitivity_checks(self):
        from core.p3_02_delta_oracle import PerturbationResult
        results = {
            "five_utr_single_sub": PerturbationResult(
                "five_utr_single_sub", 0.1, 0.05, 20,
                position_sensitivity={"0": 0.1, "10": 0.05, "20": 0.01},
            ),
            "matched_random": PerturbationResult("matched_random", 0.09, 0.04, 20),
            "start_proximal_cds": PerturbationResult("start_proximal_cds", 0.02, 0.01, 15),
            "late_cds": PerturbationResult("late_cds", 0.01, 0.01, 15),
        }
        checks = analyze_sensitivity_checks(results)
        assert "position_sensitive" in checks
        assert "gc_only_risk" in checks
        assert "length_only_risk" in checks


# ---------------------------------------------------------------------------
# Test: Headroom Search
# ---------------------------------------------------------------------------

class TestHeadroomSearch:
    def test_exact_one_edit(self):
        records = _make_synthetic_records(50)
        features = batch_extract_features(records)
        config = CrossFitConfig(n_folds=3, n_epochs=20, hidden_dim=16)

        model = DifferenceModel(hidden_dim=16, n_epochs=20, seed=42)
        model.fit(features, features["delta"])

        src = records[0].source_sequence
        result = exact_one_edit_enumeration(src, model, config, "five_utr", max_positions=50)
        assert result.search_method == "exact_one_edit"
        assert result.n_evaluated > 0
        assert result.best_delta >= 0.0  # at least as good as no edit

    def test_greedy_search(self):
        records = _make_synthetic_records(50)
        features = batch_extract_features(records)
        config = CrossFitConfig(n_folds=3, n_epochs=20, hidden_dim=16)

        model = DifferenceModel(hidden_dim=16, n_epochs=20, seed=42)
        model.fit(features, features["delta"])

        src = records[0].source_sequence
        result = greedy_search(src, model, config, max_edits=3, editable_region="five_utr")
        assert result.search_method == "greedy"
        assert result.best_delta >= 0.0

    def test_beam_search(self):
        records = _make_synthetic_records(50)
        features = batch_extract_features(records)
        config = CrossFitConfig(n_folds=3, n_epochs=20, hidden_dim=16)

        model = DifferenceModel(hidden_dim=16, n_epochs=20, seed=42)
        model.fit(features, features["delta"])

        src = records[0].source_sequence
        result = beam_search(src, model, config, max_edits=2, beam_width=3, editable_region="five_utr")
        assert result.search_method == "beam_search"
        assert result.n_evaluated > 0

    def test_simulated_annealing(self):
        records = _make_synthetic_records(50)
        features = batch_extract_features(records)
        config = CrossFitConfig(n_folds=3, n_epochs=20, hidden_dim=16)

        model = DifferenceModel(hidden_dim=16, n_epochs=20, seed=42)
        model.fit(features, features["delta"])

        src = records[0].source_sequence
        result = simulated_annealing(src, model, config, n_iterations=50, editable_region="five_utr")
        assert result.search_method == "simulated_annealing"
        assert result.n_evaluated > 0

    def test_mcts_search(self):
        records = _make_synthetic_records(50)
        features = batch_extract_features(records)
        config = CrossFitConfig(n_folds=3, n_epochs=20, hidden_dim=16)

        model = DifferenceModel(hidden_dim=16, n_epochs=20, seed=42)
        model.fit(features, features["delta"])

        src = records[0].source_sequence
        result = mcts_search(src, model, config, n_simulations=30, max_depth=2, editable_region="five_utr")
        assert result.search_method == "mcts"
        assert result.n_evaluated > 0

    def test_run_headroom_search(self):
        records = _make_synthetic_records(30)
        features = batch_extract_features(records)
        config = CrossFitConfig(n_folds=3, n_epochs=20, hidden_dim=16)

        model = DifferenceModel(hidden_dim=16, n_epochs=20, seed=42)
        model.fit(features, features["delta"])

        source_seqs = [r.source_sequence for r in records[:5]]
        results = run_headroom_search(
            model, source_seqs, config,
            methods=("exact_one_edit", "greedy"),
            n_sources=5,
        )
        assert "exact_one_edit" in results
        assert "greedy" in results
        assert len(results["exact_one_edit"]) == 5

    def test_analyze_headroom(self):
        from core.p3_02_delta_oracle import HeadroomResult
        results = {
            "exact_one_edit": [
                HeadroomResult("s0", 0, 0.3, "ACGU", [{"pos": 1, "ref": "A", "alt": "G", "region": "five_utr"}], "exact_one_edit", 10),
                HeadroomResult("s1", 0, -0.1, "ACGU", [], "exact_one_edit", 10),
            ],
            "greedy": [
                HeadroomResult("s0", 0, 0.5, "ACGU", [{"pos": 1, "ref": "A", "alt": "G", "region": "five_utr"}], "greedy", 20),
                HeadroomResult("s1", 0, 0.2, "ACGU", [{"pos": 0, "ref": "A", "alt": "G", "region": "five_utr"}], "greedy", 20),
            ],
        }
        analysis = analyze_headroom(results)
        assert "exact_one_edit" in analysis
        assert "greedy" in analysis
        assert "positive_fraction" in analysis["greedy"]


# ---------------------------------------------------------------------------
# Test: GO/NO-GO Gate
# ---------------------------------------------------------------------------

class TestGoNoGoGate:
    def test_go_verdict(self):
        metrics = {"sign_accuracy": 0.7, "top_k_enrichment_10pct": 1.5}
        sensitivity = {"gc_only_risk": False, "length_only_risk": False}
        headroom = {"greedy": {"positive_fraction": 0.5, "mean_best_delta": 0.3}}
        gate = evaluate_go_gate(metrics, sensitivity, headroom)
        assert gate["verdict"] in ("GO", "PARTIAL")
        assert gate["n_criteria_pass"] >= 3

    def test_no_go_verdict(self):
        metrics = {"sign_accuracy": 0.45, "top_k_enrichment_10pct": 0.8}
        sensitivity = {"gc_only_risk": True, "length_only_risk": True}
        headroom = {"greedy": {"positive_fraction": 0.05, "mean_best_delta": 0.0}}
        gate = evaluate_go_gate(metrics, sensitivity, headroom)
        assert gate["verdict"] == "NO_GO"
        assert gate["n_criteria_pass"] <= 2

    def test_partial_verdict(self):
        metrics = {"sign_accuracy": 0.56, "top_k_enrichment_10pct": 1.1}
        sensitivity = {"gc_only_risk": True, "length_only_risk": False}
        headroom = {"greedy": {"positive_fraction": 0.15, "mean_best_delta": 0.1}}
        gate = evaluate_go_gate(metrics, sensitivity, headroom)
        assert gate["verdict"] in ("PARTIAL", "NO_GO")


# ---------------------------------------------------------------------------
# Test: End-to-End Smoke Test
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_full_pipeline_smoke(self):
        """Smoke test: run the full pipeline with tiny synthetic data."""
        records = _make_synthetic_records(50)
        features = batch_extract_features(records)
        config = CrossFitConfig(n_folds=3, n_epochs=20, hidden_dim=16)

        n = len(records)
        fold_size = n // 3
        folds = [np.arange(i * fold_size, (i + 1) * fold_size) for i in range(3)]

        # Build ensemble
        result = build_oracle_ensemble(
            features, features["delta"], folds, config,
            model_names=("difference", "siamese"),
        )

        # Compute metrics
        metrics = compute_all_metrics(result["ensemble_pred"], features["delta"])
        assert "sign_accuracy" in metrics

        # Sensitivity
        source_seqs = [r.source_sequence for r in records[:5]]
        sensitivity = run_region_sensitivity(
            type("P", (), {"predict_delta": lambda self, f: result["ensemble_pred"][:f["source_feat"].shape[0]]})(),
            source_seqs, config, n_samples=5, seed=42,
        )
        assert "five_utr_single_sub" in sensitivity

        # Headroom
        headroom = run_headroom_search(
            type("P", (), {"predict_delta": lambda self, f: result["ensemble_pred"][:f["source_feat"].shape[0]]})(),
            source_seqs, config,
            methods=("exact_one_edit",),
            n_sources=3,
        )
        assert "exact_one_edit" in headroom

        # Gate
        sensitivity_checks = analyze_sensitivity_checks(sensitivity)
        headroom_analysis = analyze_headroom(headroom)
        gate = evaluate_go_gate(metrics, sensitivity_checks, headroom_analysis)
        assert gate["verdict"] in ("GO", "PARTIAL", "NO_GO")
