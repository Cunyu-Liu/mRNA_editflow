#!/usr/bin/env python
"""P3-03: Generate prospective falsification sequence manifest.

Selects 2 cargo source sequences from P3-01 benchmark and generates
12 experimental arms per cargo using the P3-02 oracle for prediction-guided
selection.

Output: docs/p3_03_sequence_manifest.json
"""
from __future__ import annotations

import json
import os
import sys
import random
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.p3_02_delta_oracle import (
    DeltaRecord, load_benchmark_tier, extract_features, batch_extract_features,
    DifferenceModel, SiameseModel, AbsoluteModel, EditConditionedModel,
    CrossFitConfig, exact_one_edit_enumeration, greedy_search, beam_search,
    simulated_annealing, mcts_search,
    run_region_sensitivity, NUC_VOCAB, START_CODON, CODON_TABLE,
    SYNONYMOUS_CODONS, _one_hot_sequence, _sequence_features, _edit_features,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_REPLICATES = 3
N_CARGOS = 2
N_ARMS = 12
# Total sequences = 2 cargos × 12 arms = 24 unique × 3 replicates = 72 wells
# Plus controls = ~80-96 wells total

CARGO_DEFINITIONS = [
    {
        "cargo_id": "EGFP",
        "cargo_name": "Enhanced Green Fluorescent Protein",
        "cargo_description": "Standard reporter, 239 aa, matches P3-01 measured tier (Sample2019 MPRA)",
        "cargo_length_aa": 239,
        "selection_criteria": "test split, edit_count=0 (WT anchor), 50nt 5'UTR",
    },
    {
        "cargo_id": "mCherry",
        "cargo_name": "mCherry Red Fluorescent Protein",
        "cargo_description": "Different length/property cargo, 236 aa, NOT in P3-01 training data",
        "cargo_length_aa": 236,
        "selection_criteria": "proxy tier, test split, longer 5'UTR (>50nt) for out-of-distribution test",
    },
]

ARM_DEFINITIONS = [
    {"arm_id": "A01", "name": "wt_source", "description": "Wild-type source sequence (no edits)"},
    {"arm_id": "A02", "name": "random_one_edit", "description": "Random legal single-nt substitution in 5'UTR"},
    {"arm_id": "A03", "name": "random_three_edit", "description": "Random legal 3-nt substitutions in 5'UTR"},
    {"arm_id": "A04", "name": "best_predicted_one_edit", "description": "Oracle-predicted best single-nt edit"},
    {"arm_id": "A05", "name": "best_predicted_three_edit", "description": "Oracle-predicted best 3-nt edits (greedy)"},
    {"arm_id": "A06", "name": "greedy_best", "description": "Greedy search best (max 3 edits)"},
    {"arm_id": "A07", "name": "beam_search_best", "description": "Beam search best (beam=5, max 2 edits)"},
    {"arm_id": "A08", "name": "five_utr_only_best", "description": "Best edit restricted to 5'UTR region"},
    {"arm_id": "A09", "name": "cds_only_best", "description": "Best synonymous codon edit in CDS"},
    {"arm_id": "A10", "name": "joint_best", "description": "Best joint 5'UTR + CDS edit"},
    {"arm_id": "A11", "name": "high_disagreement_negative", "description": "High ensemble disagreement, predicted negative (control)"},
    {"arm_id": "A12", "name": "adversarial_high_reward", "description": "High predicted delta but suspicious features (reward hacking test)"},
]


def select_cargo_sources(benchmark_dir: str) -> List[DeltaRecord]:
    """Select one WT source per cargo from P3-01 benchmark."""
    # Cargo 1: EGFP — from measured tier, test split, WT anchor (edit_count=0)
    measured = load_benchmark_tier(os.path.join(benchmark_dir, "measured_tier.jsonl"))
    egfp_wt = [r for r in measured if r.split_role == "test" and r.edit_count == 0]
    if not egfp_wt:
        egfp_wt = [r for r in measured if r.edit_count == 0]
    # Pick a source with moderate GC and length
    egfp_source = sorted(egfp_wt, key=lambda r: abs(
        (r.source_sequence.count("G") + r.source_sequence.count("C")) / max(len(r.source_sequence), 1) - 0.5
    ))[0]

    # Cargo 2: mCherry — from proxy tier, test split, longer 5'UTR
    # Proxy tier has NO edit_count=0 records; every record is an edit pair.
    # We pick a unique source_sequence from the test split and synthesize a WT record.
    proxy = load_benchmark_tier(os.path.join(benchmark_dir, "proxy_tier.jsonl"))
    proxy_test = [r for r in proxy if r.split_role == "test" and len(r.source_sequence) > 50]
    if not proxy_test:
        proxy_test = [r for r in proxy if len(r.source_sequence) > 50]
    # Deduplicate by source_sequence to find unique WT sources
    seen_sources: Dict[str, DeltaRecord] = {}
    for r in proxy_test:
        if r.source_sequence not in seen_sources:
            seen_sources[r.source_sequence] = r
    unique_sources = list(seen_sources.values())
    # Pick a source with different GC profile (~30%, vs EGFP's ~50%)
    mcherry_base = sorted(unique_sources, key=lambda r: abs(
        (r.source_sequence.count("G") + r.source_sequence.count("C")) / max(len(r.source_sequence), 1) - 0.3
    ))[0]
    # Synthesize a WT DeltaRecord (edit_count=0, source == candidate, delta=0)
    mcherry_source = DeltaRecord(
        record_id=f"mcherry_wt_{mcherry_base.source_id}",
        source_id=mcherry_base.source_id,
        source_sequence=mcherry_base.source_sequence,
        candidate_sequence=mcherry_base.source_sequence,
        edit_list=[],
        edit_count=0,
        edited_region="none",
        delta=0.0,
        source_value=mcherry_base.source_value,
        candidate_value=mcherry_base.source_value,
        value_std=mcherry_base.value_std,
        confidence="proxy",
        split_role="test",
        family_cluster_id=mcherry_base.family_cluster_id,
        edit_type="wild_type_anchor",
    )

    return [egfp_source, mcherry_source]


def generate_arms(source_record: DeltaRecord, cargo_id: str,
                  oracle_models: Dict[str, Any], config: CrossFitConfig,
                  seed: int = 42) -> List[Dict[str, Any]]:
    """Generate 12 experimental arms for a single cargo source."""
    rng = random.Random(seed)
    src_seq = source_record.source_sequence
    arms: List[Dict[str, Any]] = []

    # Helper to predict delta
    def predict_delta(cand_seq: str, edits: List[Dict]) -> Tuple[float, float]:
        """Returns (mean_delta, std_delta) across oracle models."""
        feats = extract_features(src_seq, cand_seq, edits, config.max_seq_len)
        batch = {k: v[np.newaxis] for k, v in feats.items()}
        deltas = []
        for name, fold_models in oracle_models.items():
            fold_preds = [m.predict_delta(batch)[0] for m in fold_models.values()]
            deltas.append(np.mean(fold_preds))
        return float(np.mean(deltas)), float(np.std(deltas))

    # A01: WT source
    arms.append({
        "arm_id": "A01", "arm_name": "wt_source",
        "cargo_id": cargo_id,
        "sequence": src_seq,
        "edits": [],
        "edit_count": 0,
        "predicted_delta": 0.0,
        "prediction_std": 0.0,
        "selection_method": "none",
    })

    # A02: Random one-edit
    aug_pos = src_seq.find(START_CODON)
    utr_end = aug_pos if aug_pos >= 0 else min(len(src_seq) // 2, 50)
    pos = rng.randint(0, max(utr_end - 1, 0))
    old = src_seq[pos]
    new_nt = rng.choice([c for c in NUC_VOCAB if c != old])
    cand = src_seq[:pos] + new_nt + src_seq[pos + 1:]
    edits = [{"pos": pos, "ref": old, "alt": new_nt, "region": "five_utr"}]
    pred_d, pred_s = predict_delta(cand, edits)
    arms.append({
        "arm_id": "A02", "arm_name": "random_one_edit",
        "cargo_id": cargo_id, "sequence": cand, "edits": edits,
        "edit_count": 1, "predicted_delta": pred_d, "prediction_std": pred_s,
        "selection_method": "random",
    })

    # A03: Random three-edit
    cand = src_seq
    edits = []
    for _ in range(3):
        pos = rng.randint(0, max(utr_end - 1, 0))
        old = cand[pos]
        new_nt = rng.choice([c for c in NUC_VOCAB if c != old])
        cand = cand[:pos] + new_nt + cand[pos + 1:]
        edits.append({"pos": pos, "ref": old, "alt": new_nt, "region": "five_utr"})
    pred_d, pred_s = predict_delta(cand, edits)
    arms.append({
        "arm_id": "A03", "arm_name": "random_three_edit",
        "cargo_id": cargo_id, "sequence": cand, "edits": edits,
        "edit_count": 3, "predicted_delta": pred_d, "prediction_std": pred_s,
        "selection_method": "random",
    })

    # A04: Best predicted one-edit (exact enumeration)
    result = exact_one_edit_enumeration(src_seq, _make_predictor(oracle_models), config, "five_utr")
    edits = result.best_edits
    pred_d, pred_s = predict_delta(result.best_candidate, edits)
    arms.append({
        "arm_id": "A04", "arm_name": "best_predicted_one_edit",
        "cargo_id": cargo_id, "sequence": result.best_candidate, "edits": edits,
        "edit_count": len(edits), "predicted_delta": pred_d, "prediction_std": pred_s,
        "selection_method": "exact_enumeration",
    })

    # A05: Best predicted three-edit (greedy)
    result = greedy_search(src_seq, _make_predictor(oracle_models), config, max_edits=3, editable_region="five_utr")
    edits = result.best_edits
    pred_d, pred_s = predict_delta(result.best_candidate, edits)
    arms.append({
        "arm_id": "A05", "arm_name": "best_predicted_three_edit",
        "cargo_id": cargo_id, "sequence": result.best_candidate, "edits": edits,
        "edit_count": len(edits), "predicted_delta": pred_d, "prediction_std": pred_s,
        "selection_method": "greedy_max3",
    })

    # A06: Greedy best (max 5 edits)
    result = greedy_search(src_seq, _make_predictor(oracle_models), config, max_edits=5, editable_region="five_utr")
    edits = result.best_edits
    pred_d, pred_s = predict_delta(result.best_candidate, edits)
    arms.append({
        "arm_id": "A06", "arm_name": "greedy_best",
        "cargo_id": cargo_id, "sequence": result.best_candidate, "edits": edits,
        "edit_count": len(edits), "predicted_delta": pred_d, "prediction_std": pred_s,
        "selection_method": "greedy_max5",
    })

    # A07: Beam search best
    result = beam_search(src_seq, _make_predictor(oracle_models), config, max_edits=3, beam_width=5, editable_region="five_utr")
    edits = result.best_edits
    pred_d, pred_s = predict_delta(result.best_candidate, edits)
    arms.append({
        "arm_id": "A07", "arm_name": "beam_search_best",
        "cargo_id": cargo_id, "sequence": result.best_candidate, "edits": edits,
        "edit_count": len(edits), "predicted_delta": pred_d, "prediction_std": pred_s,
        "selection_method": "beam_search_w5_d3",
    })

    # A08: 5'UTR-only best (same as A04 but explicitly labeled)
    # Reuse A04's result since it's already 5'UTR-only
    result = exact_one_edit_enumeration(src_seq, _make_predictor(oracle_models), config, "five_utr")
    edits = result.best_edits
    pred_d, pred_s = predict_delta(result.best_candidate, edits)
    arms.append({
        "arm_id": "A08", "arm_name": "five_utr_only_best",
        "cargo_id": cargo_id, "sequence": result.best_candidate, "edits": edits,
        "edit_count": len(edits), "predicted_delta": pred_d, "prediction_std": pred_s,
        "selection_method": "exact_5utr_only",
    })

    # A09: CDS-only best (synonymous codon substitution)
    cds_best = _best_cds_edit(src_seq, oracle_models, config, predict_delta)
    arms.append({
        "arm_id": "A09", "arm_name": "cds_only_best",
        "cargo_id": cargo_id, "sequence": cds_best["sequence"], "edits": cds_best["edits"],
        "edit_count": len(cds_best["edits"]),
        "predicted_delta": cds_best["predicted_delta"],
        "prediction_std": cds_best["prediction_std"],
        "selection_method": "exact_cds_synonymous",
    })

    # A10: Joint best (5'UTR + CDS)
    joint_best = _best_joint_edit(src_seq, oracle_models, config, predict_delta, rng)
    arms.append({
        "arm_id": "A10", "arm_name": "joint_best",
        "cargo_id": cargo_id, "sequence": joint_best["sequence"], "edits": joint_best["edits"],
        "edit_count": len(joint_best["edits"]),
        "predicted_delta": joint_best["predicted_delta"],
        "prediction_std": joint_best["prediction_std"],
        "selection_method": "greedy_joint_5utr_cds",
    })

    # A11: High-disagreement negative (max prediction_std, predicted negative)
    neg_candidate = _find_high_disagreement_negative(src_seq, oracle_models, config, predict_delta, rng)
    arms.append({
        "arm_id": "A11", "arm_name": "high_disagreement_negative",
        "cargo_id": cargo_id, "sequence": neg_candidate["sequence"], "edits": neg_candidate["edits"],
        "edit_count": len(neg_candidate["edits"]),
        "predicted_delta": neg_candidate["predicted_delta"],
        "prediction_std": neg_candidate["prediction_std"],
        "selection_method": "max_disagreement_negative",
    })

    # A12: Adversarial high-reward (high predicted delta but suspicious features)
    adv_candidate = _find_adversarial_candidate(src_seq, oracle_models, config, predict_delta, rng)
    arms.append({
        "arm_id": "A12", "arm_name": "adversarial_high_reward",
        "cargo_id": cargo_id, "sequence": adv_candidate["sequence"], "edits": adv_candidate["edits"],
        "edit_count": len(adv_candidate["edits"]),
        "predicted_delta": adv_candidate["predicted_delta"],
        "prediction_std": adv_candidate["prediction_std"],
        "selection_method": "adversarial_high_gc_extreme",
    })

    return arms


def _make_predictor(oracle_models: Dict[str, Any]):
    """Create a simple predictor wrapper for search functions.

    Returns an array of shape (N,) for a batch of N samples, so callers can
    index [0] for single-sample predictions.
    """
    class Predictor:
        def predict_delta(self, features):
            preds = []
            for name, fold_models in oracle_models.items():
                fold_preds = np.stack([m.predict_delta(features) for m in fold_models.values()])
                preds.append(fold_preds.mean(axis=0))
            return np.mean(np.stack(preds), axis=0)
    return Predictor()


def _best_cds_edit(src_seq: str, oracle_models: Dict, config: CrossFitConfig,
                   predict_fn) -> Dict[str, Any]:
    """Find best synonymous CDS edit."""
    aug_pos = src_seq.find(START_CODON)
    if aug_pos < 0:
        # No CDS found; use a dummy
        return {"sequence": src_seq, "edits": [], "predicted_delta": 0.0, "prediction_std": 0.0}
    cds = src_seq[aug_pos:]
    n_codons = len(cds) // 3
    best_delta = -1e9
    best_cand = src_seq
    best_edits = []
    best_std = 0.0

    for ci in range(1, n_codons - 1):  # skip AUG and stop
        codon = cds[ci*3:ci*3+3]
        aa = CODON_TABLE.get(codon, "")
        for syn in SYNONYMOUS_CODONS.get(aa, []):
            if syn == codon:
                continue
            new_cds = cds[:ci*3] + syn + cds[ci*3+3:]
            cand = src_seq[:aug_pos] + new_cds + src_seq[aug_pos + len(cds):]
            edits = [{"pos": aug_pos + ci*3, "ref": codon, "alt": syn, "region": "cds"}]
            pred_d, pred_s = predict_fn(cand, edits)
            if pred_d > best_delta:
                best_delta = pred_d
                best_cand = cand
                best_edits = edits
                best_std = pred_s

    return {"sequence": best_cand, "edits": best_edits,
            "predicted_delta": best_delta if best_edits else 0.0,
            "prediction_std": best_std}


def _best_joint_edit(src_seq: str, oracle_models: Dict, config: CrossFitConfig,
                     predict_fn, rng: random.Random) -> Dict[str, Any]:
    """Find best joint 5'UTR + CDS edit (greedy)."""
    # First find best 5'UTR edit
    result_utr = exact_one_edit_enumeration(src_seq, _make_predictor(oracle_models), config, "five_utr")
    cand = result_utr.best_candidate
    edits = list(result_utr.best_edits)

    # Then find best CDS edit on the modified sequence
    aug_pos = cand.find(START_CODON)
    if aug_pos >= 0:
        cds = cand[aug_pos:]
        n_codons = len(cds) // 3
        best_delta = result_utr.best_delta
        best_cand = cand
        best_edits = list(edits)
        best_std = 0.0

        for ci in range(1, min(n_codons - 1, 10)):  # limit search
            codon = cds[ci*3:ci*3+3]
            aa = CODON_TABLE.get(codon, "")
            for syn in SYNONYMOUS_CODONS.get(aa, []):
                if syn == codon:
                    continue
                new_cds = cds[:ci*3] + syn + cds[ci*3+3:]
                cand2 = cand[:aug_pos] + new_cds + cand[aug_pos + len(cds):]
                edits2 = edits + [{"pos": aug_pos + ci*3, "ref": codon, "alt": syn, "region": "cds"}]
                pred_d, pred_s = predict_fn(cand2, edits2)
                if pred_d > best_delta:
                    best_delta = pred_d
                    best_cand = cand2
                    best_edits = edits2
                    best_std = pred_s

        return {"sequence": best_cand, "edits": best_edits,
                "predicted_delta": best_delta, "prediction_std": best_std}
    return {"sequence": cand, "edits": edits,
            "predicted_delta": result_utr.best_delta, "prediction_std": 0.0}


def _find_high_disagreement_negative(src_seq: str, oracle_models: Dict,
                                     config: CrossFitConfig, predict_fn,
                                     rng: random.Random) -> Dict[str, Any]:
    """Find an edit with high ensemble disagreement and negative predicted delta."""
    aug_pos = src_seq.find(START_CODON)
    utr_end = aug_pos if aug_pos >= 0 else min(len(src_seq) // 2, 50)
    best_disagreement = -1.0
    best_result = {"sequence": src_seq, "edits": [], "predicted_delta": 0.0, "prediction_std": 0.0}

    for _ in range(50):
        pos = rng.randint(0, max(utr_end - 1, 0))
        old = src_seq[pos]
        new_nt = rng.choice([c for c in NUC_VOCAB if c != old])
        cand = src_seq[:pos] + new_nt + src_seq[pos + 1:]
        edits = [{"pos": pos, "ref": old, "alt": new_nt, "region": "five_utr"}]
        pred_d, pred_s = predict_fn(cand, edits)
        if pred_d < 0 and pred_s > best_disagreement:
            best_disagreement = pred_s
            best_result = {"sequence": cand, "edits": edits,
                           "predicted_delta": pred_d, "prediction_std": pred_s}

    return best_result


def _find_adversarial_candidate(src_seq: str, oracle_models: Dict,
                                config: CrossFitConfig, predict_fn,
                                rng: random.Random) -> Dict[str, Any]:
    """Find a candidate with high predicted delta but suspicious features.

    Strategy: maximize GC content (a known heuristic the model may over-rely on)
    while keeping the edit legal.
    """
    aug_pos = src_seq.find(START_CODON)
    utr_end = aug_pos if aug_pos >= 0 else min(len(src_seq) // 2, 50)

    # Maximize GC: replace all A/U with G/C in first few positions
    cand = list(src_seq)
    edits = []
    for pos in range(min(utr_end, 5)):  # First 5 positions
        old = cand[pos]
        if old in "AU":
            new_nt = "G" if old == "A" else "C"
            cand[pos] = new_nt
            edits.append({"pos": pos, "ref": old, "alt": new_nt, "region": "five_utr"})

    cand_str = "".join(cand)
    pred_d, pred_s = predict_fn(cand_str, edits)
    return {"sequence": cand_str, "edits": edits,
            "predicted_delta": pred_d, "prediction_std": pred_s}


def generate_manifest(benchmark_dir: str, ckpt_dir: str, output_path: str, seed: int = 42):
    """Generate the full sequence manifest."""
    print(f"[P3-03] Loading P3-01 benchmark from {benchmark_dir}")
    cargo_sources = select_cargo_sources(benchmark_dir)
    print(f"  Selected {len(cargo_sources)} cargo sources")

    # Load P3-02 oracle models from checkpoints
    print(f"[P3-03] Loading P3-02 oracle from {ckpt_dir}")
    config = CrossFitConfig(n_folds=3, n_epochs=20, hidden_dim=64, max_seq_len=130)

    # Load measured data for oracle training (quick retrain on server)
    measured = load_benchmark_tier(os.path.join(benchmark_dir, "measured_tier.jsonl"))
    train_recs = [r for r in measured if r.split_role in ("train", "val")]
    features = batch_extract_features(train_recs, config.max_seq_len)
    labels = features["delta"]

    # Train 4 models (single fold for speed)
    # Simple fold split
    n = len(train_recs)
    fold_size = n // 3
    folds = [np.arange(i * fold_size, (i + 1) * fold_size) for i in range(3)]

    oracle_models: Dict[str, Dict[int, Any]] = {}
    for name, model_class in [("absolute", AbsoluteModel), ("difference", DifferenceModel),
                               ("siamese", SiameseModel), ("edit_conditioned", EditConditionedModel)]:
        fold_models = {}
        for fold_id, val_idx in enumerate(folds):
            train_mask = np.ones(n, dtype=bool)
            train_mask[val_idx] = False
            train_idx = np.where(train_mask)[0]
            train_feats = {k: v[train_idx] for k, v in features.items()}
            model = model_class(hidden_dim=64, n_epochs=30, seed=42 + fold_id)
            if name == "absolute":
                model.fit(train_feats, train_feats["source_value"] + train_feats["delta"])
            else:
                model.fit(train_feats, labels[train_idx])
            fold_models[fold_id] = model
        oracle_models[name] = fold_models
        print(f"  Trained {name} (3 folds)")

    # Generate arms for each cargo
    print(f"[P3-03] Generating {N_ARMS} arms per cargo")
    all_arms: List[Dict[str, Any]] = []
    for cargo_idx, (cargo_def, source) in enumerate(zip(CARGO_DEFINITIONS, cargo_sources)):
        print(f"  Cargo {cargo_idx + 1}: {cargo_def['cargo_id']}")
        arms = generate_arms(source, cargo_def["cargo_id"], oracle_models, config, seed + cargo_idx)
        all_arms.extend(arms)

    # Build manifest
    manifest = {
        "phase": "P3-03",
        "manifest_type": "prospective_falsification_sequence_manifest",
        "generated_at": str(np.datetime64("now")),
        "configuration": {
            "n_cargos": N_CARGOS,
            "n_arms_per_cargo": N_ARMS,
            "n_replicates": N_REPLICATES,
            "total_unique_sequences": len(all_arms),
            "total_wells": len(all_arms) * N_REPLICATES,
            "cell_context": "HEK293T",
            "time_points": ["4h", "8h", "24h", "48h"],
            "readouts": ["protein_output", "mRNA_abundance", "cell_viability"],
        },
        "cargo_definitions": CARGO_DEFINITIONS,
        "arm_definitions": ARM_DEFINITIONS,
        "source_sequences": [
            {
                "cargo_id": cargo_def["cargo_id"],
                "source_record_id": source.record_id,
                "source_sequence": source.source_sequence,
                "source_value": source.source_value,
                "confidence": source.confidence,
                "split_role": source.split_role,
            }
            for cargo_def, source in zip(CARGO_DEFINITIONS, cargo_sources)
        ],
        "sequences": all_arms,
        "well_layout": _generate_well_layout(all_arms, N_REPLICATES, seed),
    }

    # Compute SHA-256 for integrity (exclude volatile fields for reproducibility)
    manifest_for_hash = {k: v for k, v in manifest.items()
                         if k not in ("generated_at", "manifest_sha256")}
    manifest_str = json.dumps(manifest_for_hash, sort_keys=True, default=str)
    manifest["manifest_sha256"] = hashlib.sha256(manifest_str.encode()).hexdigest()

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[P3-03] Saved manifest: {output_path}")
    print(f"  Total unique sequences: {len(all_arms)}")
    print(f"  Total wells (×{N_REPLICATES} replicates): {len(all_arms) * N_REPLICATES}")
    return manifest


def _generate_well_layout(arms: List[Dict], n_replicates: int, seed: int) -> List[Dict]:
    """Generate randomized well layout."""
    rng = random.Random(seed)
    wells = []
    well_idx = 0
    for arm in arms:
        for rep in range(n_replicates):
            wells.append({
                "well_id": f"W{well_idx:03d}",
                "arm_id": arm["arm_id"],
                "cargo_id": arm["cargo_id"],
                "replicate": rep + 1,
                "sequence": arm["sequence"],
            })
            well_idx += 1
    # Shuffle well assignments (but keep track of arm/replicate)
    rng.shuffle(wells)
    # Assign plate positions
    for i, well in enumerate(wells):
        row = chr(ord("A") + i // 12)
        col = (i % 12) + 1
        well["plate_position"] = f"{row}{col:02d}"
    return wells


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", default="data/p3/benchmark")
    parser.add_argument("--ckpt-dir", default="checkpoints/p3_delta_oracles")
    parser.add_argument("--output", default="docs/p3_03_sequence_manifest.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_manifest(args.benchmark_dir, args.ckpt_dir, args.output, args.seed)
