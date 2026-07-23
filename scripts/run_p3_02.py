#!/usr/bin/env python
"""P3-02: Local-Delta Oracle and optimization-headroom gate — main driver.

Executes the full P3-02 pipeline:
1. Load P3-01 benchmark data (measured + proxy tiers)
2. Extract features for all records
3. Train 4 model architectures with cross-fitting
4. Compute local-delta metrics
5. Build cross-fitted oracle ensemble
6. Run region sensitivity perturbation tests
7. Run optimization headroom search
8. Evaluate GO/PARTIAL/NO-GO gate
9. Save all results and generate docs

Usage:
    python scripts/run_p3_02.py --benchmark-dir data/p3/benchmark \
        --output-dir results/p3_02 --doc-dir docs --ckpt-dir checkpoints/p3_delta_oracles

    # Quick smoke test with synthetic data:
    python scripts/run_p3_02.py --smoke-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.p3_02_delta_oracle import (
    DeltaRecord, load_benchmark, load_benchmark_tier,
    batch_extract_features, extract_features,
    AbsoluteModel, DifferenceModel, SiameseModel, EditConditionedModel,
    MODEL_REGISTRY, compute_all_metrics,
    CrossFitConfig, cross_fit_predict, build_oracle_ensemble,
    run_region_sensitivity, analyze_sensitivity_checks,
    run_headroom_search, analyze_headroom,
    evaluate_go_gate,
    NUC_VOCAB, START_CODON,
)
import random


def generate_synthetic_data(n: int = 500, seed: int = 42) -> List[DeltaRecord]:
    """Generate synthetic data for smoke testing."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    records: List[DeltaRecord] = []
    for i in range(n):
        # Generate a random 50nt 5'UTR sequence
        src = "".join(rng.choice(list(NUC_VOCAB)) for _ in range(50))
        # Apply 0-2 random substitutions
        n_edits = rng.choice([0, 1, 1, 2])
        cand = src
        edits: List[Dict[str, Any]] = []
        for _ in range(n_edits):
            pos = rng.randint(0, len(cand) - 1)
            old = cand[pos]
            new_nt = rng.choice([c for c in NUC_VOCAB if c != old])
            cand = cand[:pos] + new_nt + cand[pos + 1:]
            edits.append({"pos": pos, "ref": old, "alt": new_nt, "region": "five_utr"})

        # Synthetic delta: depends on position and GC change
        delta = 0.0
        for ed in edits:
            # Position near start has more effect
            pos_effect = 1.0 - ed["pos"] / 50.0
            # GC-increasing edits are beneficial
            gc_change = 1.0 if ed["alt"] in "GC" and ed["ref"] in "AU" else -0.5
            delta += pos_effect * gc_change * 0.3 + np_rng.randn() * 0.1
        src_val = 5.0 + np_rng.randn() * 0.5

        split = "train" if i < int(n * 0.7) else ("val" if i < int(n * 0.85) else "test")
        records.append(DeltaRecord(
            record_id=f"syn_{i:04d}",
            source_id=f"syn_src_{i // 5}",
            source_sequence=src,
            candidate_sequence=cand,
            edit_list=edits,
            edit_count=len(edits),
            edited_region="five_utr",
            delta=float(delta),
            source_value=float(src_val),
            candidate_value=float(src_val + delta),
            value_std=0.3,
            confidence="measured",
            split_role=split,
            family_cluster_id=f"fam_{i // 10}",
            edit_type="measured_single" if n_edits == 1 else "wild_type_anchor",
        ))
    return records


def make_k_fold_indices(
    records: Sequence[DeltaRecord],
    n_folds: int = 5,
    seed: int = 42,
) -> List[np.ndarray]:
    """Create K-fold indices respecting family clusters."""
    # Group by source_id to avoid leakage
    from collections import defaultdict
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        groups[rec.source_id].append(i)

    group_ids = list(groups.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(group_ids)

    fold_size = len(group_ids) // n_folds
    folds: List[np.ndarray] = []
    for k in range(n_folds):
        start = k * fold_size
        end = (k + 1) * fold_size if k < n_folds - 1 else len(group_ids)
        val_groups = group_ids[start:end]
        val_indices = []
        for gid in val_groups:
            val_indices.extend(groups[gid])
        folds.append(np.array(sorted(val_indices)))
    return folds


def run_p3_02(
    benchmark_dir: Optional[str] = None,
    output_dir: str = "results/p3_02",
    doc_dir: str = "docs",
    ckpt_dir: str = "checkpoints/p3_delta_oracles",
    smoke_test: bool = False,
    n_folds: int = 5,
    n_epochs: int = 100,
    hidden_dim: int = 64,
    n_sensitivity_samples: int = 100,
    n_headroom_sources: int = 50,
    max_proxy: int = 10000,
    seed: int = 42,
):
    """Run the full P3-02 pipeline."""
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(doc_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    config = CrossFitConfig(
        n_folds=n_folds,
        seed=seed,
        hidden_dim=hidden_dim,
        lr=1e-3,
        n_epochs=n_epochs,
        max_seq_len=100,
    )

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("P3-02: Local-Delta Oracle and Optimization-Headroom Gate")
    print("=" * 60)

    if smoke_test:
        print("\n[SMOKE TEST] Using synthetic data")
        all_records = generate_synthetic_data(n=500, seed=seed)
        measured_records = [r for r in all_records if r.split_role in ("train", "val", "test")]
        proxy_records = []
    else:
        print(f"\n[1] Loading benchmark data from {benchmark_dir}")
        tiers = load_benchmark(benchmark_dir, tiers=("measured", "proxy"))
        measured_records = tiers.get("measured", [])
        proxy_records = tiers.get("proxy", [])
        print(f"  Measured: {len(measured_records)} records")
        print(f"  Proxy: {len(proxy_records)} records")

    # Split measured into train/val/test
    train_recs = [r for r in measured_records if r.split_role == "train"]
    val_recs = [r for r in measured_records if r.split_role == "val"]
    test_recs = [r for r in measured_records if r.split_role == "test"]
    ood_recs = [r for r in measured_records if r.split_role == "ood"]
    print(f"  Split: train={len(train_recs)}, val={len(val_recs)}, test={len(test_recs)}, ood={len(ood_recs)}")

    # ------------------------------------------------------------------
    # Step 2: Extract features
    # ------------------------------------------------------------------
    print("\n[2] Extracting features")
    # Use measured for training, proxy as additional training data if available
    train_all = train_recs + val_recs  # train+val for cross-fitting
    if proxy_records and not smoke_test:
        # Subsample proxy to keep training time reasonable
        rng = np.random.RandomState(seed)
        n_proxy_sample = min(len(proxy_records), max_proxy)
        proxy_sample_idx = rng.choice(len(proxy_records), n_proxy_sample, replace=False)
        proxy_sample = [proxy_records[i] for i in proxy_sample_idx]
        train_all = train_all + proxy_sample
        print(f"  Added {n_proxy_sample} proxy records to training set")

    train_features = batch_extract_features(train_all, config.max_seq_len)
    test_features = batch_extract_features(test_recs, config.max_seq_len)
    ood_features = batch_extract_features(ood_recs, config.max_seq_len) if ood_recs else None
    train_labels = train_features["delta"]
    print(f"  Train features: {train_features['source_feat'].shape}")
    print(f"  Test features: {test_features['source_feat'].shape}")

    # ------------------------------------------------------------------
    # Step 3: Cross-fitted training of 4 model architectures
    # ------------------------------------------------------------------
    print("\n[3] Cross-fitted training of 4 model architectures")
    fold_indices = make_k_fold_indices(train_all, n_folds=n_folds, seed=seed)

    ensemble_result = build_oracle_ensemble(
        train_features, train_labels, fold_indices, config,
        model_names=("absolute", "difference", "siamese", "edit_conditioned"),
    )

    for name in ensemble_result["model_names"]:
        oof = ensemble_result["per_model_oof"][name]
        m = compute_all_metrics(oof, train_labels, train_features.get("source_value"))
        print(f"  {name}: sign_acc={m['sign_accuracy']:.4f}, "
              f"top_k={m['top_k_enrichment_10pct']:.4f}, "
              f"spearman={m['delta_spearman']:.4f}")

    # ------------------------------------------------------------------
    # Step 4: Evaluate on test set
    # ------------------------------------------------------------------
    print("\n[4] Evaluating on test set")
    test_metrics: Dict[str, Dict[str, float]] = {}
    test_predictions: Dict[str, np.ndarray] = {}

    for name in ensemble_result["model_names"]:
        # Use fold-0 model as representative (or average across folds)
        preds_list = []
        for fold_id, model in ensemble_result["per_model_models"][name].items():
            pred = model.predict_delta(test_features)
            preds_list.append(pred)
        test_pred = np.mean(preds_list, axis=0)
        test_predictions[name] = test_pred
        m = compute_all_metrics(test_pred, test_features["delta"], test_features.get("source_value"))
        test_metrics[name] = m
        print(f"  {name}: sign_acc={m['sign_accuracy']:.4f}, "
              f"top_k={m['top_k_enrichment_10pct']:.4f}, "
              f"spearman={m['delta_spearman']:.4f}, "
              f"beneficial_prec={m['beneficial_edit_precision']:.4f}")

    # Ensemble test predictions
    ensemble_test_pred = np.mean(list(test_predictions.values()), axis=0)
    ensemble_test_metrics = compute_all_metrics(
        ensemble_test_pred, test_features["delta"], test_features.get("source_value")
    )
    print(f"  ENSEMBLE: sign_acc={ensemble_test_metrics['sign_accuracy']:.4f}, "
          f"top_k={ensemble_test_metrics['top_k_enrichment_10pct']:.4f}, "
          f"spearman={ensemble_test_metrics['delta_spearman']:.4f}")

    # Evaluate on OOD if available
    ood_metrics = None
    if ood_features is not None and len(ood_recs) > 0:
        print("\n[4b] Evaluating on OOD set")
        ood_predictions: Dict[str, np.ndarray] = {}
        for name in ensemble_result["model_names"]:
            preds_list = []
            for fold_id, model in ensemble_result["per_model_models"][name].items():
                pred = model.predict_delta(ood_features)
                preds_list.append(pred)
            ood_predictions[name] = np.mean(preds_list, axis=0)
        ensemble_ood_pred = np.mean(list(ood_predictions.values()), axis=0)
        ood_metrics = compute_all_metrics(
            ensemble_ood_pred, ood_features["delta"], ood_features.get("source_value")
        )
        print(f"  ENSEMBLE OOD: sign_acc={ood_metrics['sign_accuracy']:.4f}, "
              f"top_k={ood_metrics['top_k_enrichment_10pct']:.4f}")

    # ------------------------------------------------------------------
    # Step 5: Region sensitivity tests
    # ------------------------------------------------------------------
    print("\n[5] Region sensitivity perturbation tests")
    # Use a simple ensemble wrapper as the predictor
    class EnsemblePredictor:
        def __init__(self, models_dict, config):
            self.models_dict = models_dict
            self.config = config

        def predict_delta(self, features):
            preds = []
            for name, fold_models in self.models_dict.items():
                fold_preds = []
                for fold_id, model in fold_models.items():
                    fold_preds.append(model.predict_delta(features))
                preds.append(np.mean(fold_preds, axis=0))
            return np.mean(preds, axis=0)

    predictor = EnsemblePredictor(ensemble_result["per_model_models"], config)
    source_seqs = [r.source_sequence for r in test_recs[:n_sensitivity_samples]]
    sensitivity_results = run_region_sensitivity(
        predictor, source_seqs, config,
        n_samples=n_sensitivity_samples, seed=seed,
    )
    sensitivity_checks = analyze_sensitivity_checks(sensitivity_results)

    for ptype, result in sensitivity_results.items():
        print(f"  {ptype}: mean_delta={result.mean_pred_delta:.4f}, "
              f"std={result.std_pred_delta:.4f}, n={result.n_samples}")

    print(f"  Position sensitive: {sensitivity_checks.get('position_sensitive', False)}")
    print(f"  GC-only risk: {sensitivity_checks.get('gc_only_risk', True)}")
    print(f"  Length-only risk: {sensitivity_checks.get('length_only_risk', True)}")

    # ------------------------------------------------------------------
    # Step 6: Optimization headroom search
    # ------------------------------------------------------------------
    print("\n[6] Optimization headroom search")
    headroom_results = run_headroom_search(
        predictor, source_seqs, config,
        methods=("exact_one_edit", "greedy", "beam_search", "simulated_annealing", "mcts"),
        n_sources=n_headroom_sources,
        editable_region="five_utr",
    )
    headroom_analysis = analyze_headroom(headroom_results)

    for method, analysis in headroom_analysis.items():
        if isinstance(analysis, dict) and "mean_best_delta" in analysis:
            print(f"  {method}: mean_delta={analysis['mean_best_delta']:.4f}, "
                  f"positive_frac={analysis['positive_fraction']:.4f}, "
                  f"n={analysis['n_sources']}")

    # ------------------------------------------------------------------
    # Step 7: GO/NO-GO gate
    # ------------------------------------------------------------------
    print("\n[7] GO/PARTIAL/NO-GO Gate Evaluation")
    gate = evaluate_go_gate(
        ensemble_test_metrics, sensitivity_checks, headroom_analysis,
    )
    print(f"  Verdict: {gate['verdict']}")
    print(f"  Criteria passed: {gate['n_criteria_pass']}/{gate['n_criteria_total']}")
    for r in gate["rationale"]:
        print(f"  - {r}")

    # ------------------------------------------------------------------
    # Step 8: Save results
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n[8] Saving results (elapsed: {elapsed:.1f}s)")

    # Save results JSON
    results_json = {
        "phase": "P3-02",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "smoke_test": smoke_test,
        "data_summary": {
            "n_measured": len(measured_records),
            "n_proxy": len(proxy_records) if not smoke_test else 0,
            "n_train": len(train_recs),
            "n_val": len(val_recs),
            "n_test": len(test_recs),
            "n_ood": len(ood_recs),
        },
        "model_architectures": list(ensemble_result["model_names"]),
        "per_model_test_metrics": test_metrics,
        "ensemble_test_metrics": ensemble_test_metrics,
        "ood_metrics": ood_metrics,
        "sensitivity_checks": sensitivity_checks,
        "headroom_analysis": headroom_analysis,
        "gate": gate,
    }

    results_path = os.path.join(doc_dir, "p3_02_delta_oracle_results.json")
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"  Saved: {results_path}")

    # Save raw predictions
    pred_path = os.path.join(output_dir, "test_predictions.npz")
    np.savez(pred_path, **{f"pred_{k}": v for k, v in test_predictions.items()},
             true_delta=test_features["delta"])
    print(f"  Saved: {pred_path}")

    # Save ensemble model (fold-0 as representative)
    for name in ensemble_result["model_names"]:
        model = ensemble_result["per_model_models"][name][0]
        if hasattr(model, "_weights"):
            ckpt_path = os.path.join(ckpt_dir, f"{name}_fold0.npz")
            np.savez(ckpt_path, **model._weights)
    print(f"  Saved checkpoints to: {ckpt_dir}")

    return results_json


def main():
    parser = argparse.ArgumentParser(description="P3-02: Local-Delta Oracle")
    parser.add_argument("--benchmark-dir", default="data/p3/benchmark",
                        help="Path to P3-01 benchmark JSONL directory")
    parser.add_argument("--output-dir", default="results/p3_02")
    parser.add_argument("--doc-dir", default="docs")
    parser.add_argument("--ckpt-dir", default="checkpoints/p3_delta_oracles")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run with synthetic data for quick testing")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-sensitivity-samples", type=int, default=100)
    parser.add_argument("--n-headroom-sources", type=int, default=50)
    parser.add_argument("--max-proxy", type=int, default=10000,
                        help="Max proxy records to sample for training")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_p3_02(
        benchmark_dir=args.benchmark_dir,
        output_dir=args.output_dir,
        doc_dir=args.doc_dir,
        ckpt_dir=args.ckpt_dir,
        smoke_test=args.smoke_test,
        n_folds=args.n_folds,
        n_epochs=args.n_epochs,
        hidden_dim=args.hidden_dim,
        n_sensitivity_samples=args.n_sensitivity_samples,
        n_headroom_sources=args.n_headroom_sources,
        max_proxy=args.max_proxy,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
