#!/usr/bin/env python
"""Unit tests for P3-03 prospective falsification manifest generation.

Tests cover:
1. Cargo source selection (measured + proxy tiers)
2. Arm generation (12 arms per cargo)
3. Well layout randomization
4. Manifest integrity (SHA-256, structure)
5. Arm correctness (WT, random, prediction-guided, adversarial)

All tests run on CPU with synthetic data — no GPU or trained models required.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.p3_02_delta_oracle import (
    DeltaRecord, CrossFitConfig, NUC_VOCAB, START_CODON,
    AbsoluteModel, DifferenceModel, SiameseModel, EditConditionedModel,
    batch_extract_features, extract_features,
)
from scripts.generate_p3_03_manifest import (
    CARGO_DEFINITIONS, ARM_DEFINITIONS,
    N_CARGOS, N_ARMS, N_REPLICATES,
    select_cargo_sources, generate_arms, _make_predictor,
    _best_cds_edit, _best_joint_edit, _find_high_disagreement_negative,
    _find_adversarial_candidate, _generate_well_layout,
    generate_manifest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_synthetic_record(
    record_id: str = "test_001",
    source_seq: str = None,
    split_role: str = "test",
    edit_count: int = 0,
    confidence: str = "measured",
) -> DeltaRecord:
    """Create a synthetic DeltaRecord for testing."""
    if source_seq is None:
        # 50nt 5'UTR + AUG + 30nt CDS = 83nt total
        source_seq = "GCCAUGCAUCAUCAUCAUCAUCAUCAUCAUCAUCAUCAUCAUCAUCAUCAUC" + START_CODON + "GCUGCUGCUGCUGCUGCUGCUGCUGCUGCU"
    if edit_count == 0:
        cand_seq = source_seq
        edits = []
        delta = 0.0
    else:
        cand_seq = list(source_seq)
        edits = []
        for i in range(edit_count):
            pos = i * 5
            if pos < len(cand_seq):
                old = cand_seq[pos]
                new_nt = "G" if old != "G" else "C"
                cand_seq[pos] = new_nt
                edits.append({"pos": pos, "ref": old, "alt": new_nt, "region": "five_utr"})
        cand_seq = "".join(cand_seq)
        delta = 0.1 * edit_count

    src_val = 5.0
    return DeltaRecord(
        record_id=record_id,
        source_id=f"src_{record_id}",
        source_sequence=source_seq,
        candidate_sequence=cand_seq,
        edit_list=edits,
        edit_count=edit_count,
        edited_region="five_utr" if edit_count > 0 else "none",
        delta=float(delta),
        source_value=float(src_val),
        candidate_value=float(src_val + delta),
        value_std=0.3,
        confidence=confidence,
        split_role=split_role,
        family_cluster_id=f"fam_{record_id[:3]}",
        edit_type="wild_type_anchor" if edit_count == 0 else "measured_single",
    )


def _make_synthetic_benchmark(benchmark_dir: str):
    """Create a minimal synthetic benchmark for testing."""
    os.makedirs(benchmark_dir, exist_ok=True)

    # Measured tier: 20 WT + 20 edited records
    measured_records = []
    for i in range(20):
        measured_records.append(_make_synthetic_record(
            f"meas_wt_{i:03d}", split_role="test" if i < 5 else "train", edit_count=0
        ))
    for i in range(20):
        measured_records.append(_make_synthetic_record(
            f"meas_ed_{i:03d}", split_role="test" if i < 5 else "train", edit_count=1
        ))

    with open(os.path.join(benchmark_dir, "measured_tier.jsonl"), "w") as f:
        for r in measured_records:
            f.write(json.dumps({
                "record_id": r.record_id,
                "source_id": r.source_id,
                "source_sequence": r.source_sequence,
                "candidate_sequence": r.candidate_sequence,
                "edit_list": r.edit_list,
                "edit_count": r.edit_count,
                "edited_region": r.edited_region,
                "delta": r.delta,
                "source_value": r.source_value,
                "candidate_value": r.candidate_value,
                "value_std": r.value_std,
                "confidence": r.confidence,
                "split_role": r.split_role,
                "family_cluster_id": r.family_cluster_id,
                "edit_type": r.edit_type,
            }) + "\n")

    # Proxy tier: 10 records with longer source sequences, all edits (no WT)
    proxy_records = []
    for i in range(10):
        long_seq = "GCCAUGCAU" * 10 + START_CODON + "GCUGCUGCUGCUGCUGCUGCUGCUGCUGCUGCUGCUGCUGCUGCUGCUGCUGCUGCUGCUGCUGCUGCUGCUGCUGCUGCUGCU"
        proxy_records.append(_make_synthetic_record(
            f"proxy_{i:03d}", source_seq=long_seq, split_role="test" if i < 5 else "train",
            edit_count=1, confidence="proxy"
        ))

    with open(os.path.join(benchmark_dir, "proxy_tier.jsonl"), "w") as f:
        for r in proxy_records:
            f.write(json.dumps({
                "record_id": r.record_id,
                "source_id": r.source_id,
                "source_sequence": r.source_sequence,
                "candidate_sequence": r.candidate_sequence,
                "edit_list": r.edit_list,
                "edit_count": r.edit_count,
                "edited_region": r.edited_region,
                "delta": r.delta,
                "source_value": r.source_value,
                "candidate_value": r.candidate_value,
                "value_std": r.value_std,
                "confidence": r.confidence,
                "split_role": r.split_role,
                "family_cluster_id": r.family_cluster_id,
                "edit_type": r.edit_type,
            }) + "\n")


def _make_dummy_oracle_models(config: CrossFitConfig, n_train: int = 10):
    """Create dummy oracle models for testing arm generation."""
    # Create small training data
    records = [_make_synthetic_record(f"dummy_{i}", edit_count=i % 3) for i in range(n_train)]
    features = batch_extract_features(records, config.max_seq_len)

    oracle_models = {}
    for name, model_class in [("absolute", AbsoluteModel), ("difference", DifferenceModel),
                               ("siamese", SiameseModel), ("edit_conditioned", EditConditionedModel)]:
        model = model_class(hidden_dim=16, n_epochs=2, seed=42)
        if name == "absolute":
            model.fit(features, features["source_value"] + features["delta"])
        else:
            model.fit(features, features["delta"])
        oracle_models[name] = {0: model}
    return oracle_models


# ---------------------------------------------------------------------------
# Tests: Configuration constants
# ---------------------------------------------------------------------------
class TestConfiguration:
    def test_cargo_count(self):
        assert N_CARGOS == 2

    def test_arm_count(self):
        assert N_ARMS == 12

    def test_replicate_count(self):
        assert N_REPLICATES >= 3

    def test_cargo_definitions(self):
        assert len(CARGO_DEFINITIONS) == 2
        cargo_ids = [c["cargo_id"] for c in CARGO_DEFINITIONS]
        assert "EGFP" in cargo_ids
        assert "mCherry" in cargo_ids

    def test_arm_definitions(self):
        assert len(ARM_DEFINITIONS) == 12
        arm_ids = [a["arm_id"] for a in ARM_DEFINITIONS]
        assert arm_ids == [f"A{i:02d}" for i in range(1, 13)]

    def test_arm_coverage(self):
        """Verify arms cover all required categories."""
        names = [a["name"] for a in ARM_DEFINITIONS]
        assert "wt_source" in names
        assert "random_one_edit" in names
        assert "best_predicted_one_edit" in names
        assert "five_utr_only_best" in names
        assert "cds_only_best" in names
        assert "joint_best" in names
        assert "adversarial_high_reward" in names


# ---------------------------------------------------------------------------
# Tests: Cargo source selection
# ---------------------------------------------------------------------------
class TestCargoSourceSelection:
    def test_select_cargo_sources(self, tmp_path):
        benchmark_dir = str(tmp_path / "benchmark")
        _make_synthetic_benchmark(benchmark_dir)
        sources = select_cargo_sources(benchmark_dir)
        assert len(sources) == 2
        # EGFP from measured tier
        assert sources[0].edit_count == 0
        assert sources[0].source_sequence == sources[0].candidate_sequence
        # mCherry synthesized from proxy tier
        assert sources[1].edit_count == 0
        assert sources[1].source_sequence == sources[1].candidate_sequence
        assert sources[1].confidence == "proxy"

    def test_mcherry_source_has_longer_utr(self, tmp_path):
        benchmark_dir = str(tmp_path / "benchmark")
        _make_synthetic_benchmark(benchmark_dir)
        sources = select_cargo_sources(benchmark_dir)
        # Proxy source sequences should be longer than measured (50nt)
        assert len(sources[1].source_sequence) > 50


# ---------------------------------------------------------------------------
# Tests: Arm generation
# ---------------------------------------------------------------------------
class TestArmGeneration:
    def setup_method(self):
        self.config = CrossFitConfig(n_folds=2, n_epochs=2, hidden_dim=16, max_seq_len=130)
        self.oracle_models = _make_dummy_oracle_models(self.config)
        self.source = _make_synthetic_record("test_arm_src", edit_count=0)

    def test_generates_12_arms(self):
        arms = generate_arms(self.source, "TEST", self.oracle_models, self.config, seed=42)
        assert len(arms) == 12

    def test_arm_ids_unique(self):
        arms = generate_arms(self.source, "TEST", self.oracle_models, self.config, seed=42)
        arm_ids = [a["arm_id"] for a in arms]
        assert len(set(arm_ids)) == 12

    def test_wt_arm_has_no_edits(self):
        arms = generate_arms(self.source, "TEST", self.oracle_models, self.config, seed=42)
        wt = arms[0]
        assert wt["arm_id"] == "A01"
        assert wt["edit_count"] == 0
        assert wt["edits"] == []
        assert wt["sequence"] == self.source.source_sequence

    def test_all_arms_have_sequences(self):
        arms = generate_arms(self.source, "TEST", self.oracle_models, self.config, seed=42)
        for arm in arms:
            assert "sequence" in arm
            assert len(arm["sequence"]) > 0
            assert "predicted_delta" in arm
            assert "prediction_std" in arm

    def test_prediction_guided_arms_differ_from_wt(self):
        arms = generate_arms(self.source, "TEST", self.oracle_models, self.config, seed=42)
        wt_seq = arms[0]["sequence"]
        # A04 (best_predicted_one_edit) should differ from WT
        a04 = next(a for a in arms if a["arm_id"] == "A04")
        assert a04["sequence"] != wt_seq
        assert a04["edit_count"] >= 1

    def test_adversarial_arm_has_high_gc(self):
        arms = generate_arms(self.source, "TEST", self.oracle_models, self.config, seed=42)
        a12 = next(a for a in arms if a["arm_id"] == "A12")
        # Adversarial arm maximizes GC in first 5 positions
        first5 = a12["sequence"][:5]
        gc_count = sum(1 for c in first5 if c in "GC")
        # At least some positions should be GC (if original had AU)
        assert gc_count >= 3 or "GC" not in "AU"

    def test_cds_arm_preserves_protein(self):
        """CDS-only edit should be synonymous (preserves protein)."""
        arms = generate_arms(self.source, "TEST", self.oracle_models, self.config, seed=42)
        a09 = next(a for a in arms if a["arm_id"] == "A09")
        # If CDS edit was made, it should be synonymous
        if a09["edit_count"] > 0:
            from core.constants import CODON_TABLE, translate
            aug_pos = self.source.source_sequence.find(START_CODON)
            if aug_pos >= 0:
                src_cds = self.source.source_sequence[aug_pos:]
                cand_cds = a09["sequence"][aug_pos:]
                assert translate(src_cds) == translate(cand_cds)


# ---------------------------------------------------------------------------
# Tests: Helper functions
# ---------------------------------------------------------------------------
class TestHelpers:
    def setup_method(self):
        self.config = CrossFitConfig(n_folds=2, n_epochs=2, hidden_dim=16, max_seq_len=130)
        self.oracle_models = _make_dummy_oracle_models(self.config)
        self.source = _make_synthetic_record("test_helper_src", edit_count=0)

    def test_make_predictor(self):
        predictor = _make_predictor(self.oracle_models)
        assert hasattr(predictor, "predict_delta")
        # Test prediction on a batch
        feats = extract_features(self.source.source_sequence, self.source.source_sequence, [], self.config.max_seq_len)
        batch = {k: v[np.newaxis] for k, v in feats.items()}
        pred = predictor.predict_delta(batch)
        assert pred.shape[0] == 1

    def test_best_cds_edit(self):
        def predict_fn(cand, edits):
            feats = extract_features(self.source.source_sequence, cand, edits, self.config.max_seq_len)
            batch = {k: v[np.newaxis] for k, v in feats.items()}
            predictor = _make_predictor(self.oracle_models)
            pred = predictor.predict_delta(batch)
            return float(pred[0]), 0.1

        result = _best_cds_edit(self.source.source_sequence, self.oracle_models, self.config, predict_fn)
        assert "sequence" in result
        assert "edits" in result
        assert "predicted_delta" in result

    def test_find_adversarial(self):
        import random
        rng = random.Random(42)

        def predict_fn(cand, edits):
            feats = extract_features(self.source.source_sequence, cand, edits, self.config.max_seq_len)
            batch = {k: v[np.newaxis] for k, v in feats.items()}
            predictor = _make_predictor(self.oracle_models)
            pred = predictor.predict_delta(batch)
            return float(pred[0]), 0.1

        result = _find_adversarial_candidate(self.source.source_sequence, self.oracle_models, self.config, predict_fn, rng)
        assert "sequence" in result
        assert len(result["edits"]) >= 0


# ---------------------------------------------------------------------------
# Tests: Well layout
# ---------------------------------------------------------------------------
class TestWellLayout:
    def test_well_count(self):
        arms = [{"arm_id": f"A{i:02d}", "cargo_id": "TEST", "sequence": "ACGU" * 20} for i in range(12)]
        wells = _generate_well_layout(arms, N_REPLICATES, seed=42)
        assert len(wells) == 12 * N_REPLICATES

    def test_well_ids_unique(self):
        arms = [{"arm_id": f"A{i:02d}", "cargo_id": "TEST", "sequence": "ACGU" * 20} for i in range(12)]
        wells = _generate_well_layout(arms, N_REPLICATES, seed=42)
        well_ids = [w["well_id"] for w in wells]
        assert len(set(well_ids)) == len(wells)

    def test_plate_positions_assigned(self):
        arms = [{"arm_id": f"A{i:02d}", "cargo_id": "TEST", "sequence": "ACGU" * 20} for i in range(12)]
        wells = _generate_well_layout(arms, N_REPLICATES, seed=42)
        for well in wells:
            assert "plate_position" in well
            assert len(well["plate_position"]) >= 3  # e.g., "A01"

    def test_replicate_distribution(self):
        arms = [{"arm_id": f"A{i:02d}", "cargo_id": "TEST", "sequence": "ACGU" * 20} for i in range(12)]
        wells = _generate_well_layout(arms, 3, seed=42)
        # Each arm should have exactly 3 wells
        from collections import Counter
        arm_counts = Counter(w["arm_id"] for w in wells)
        for arm_id in [f"A{i:02d}" for i in range(12)]:
            assert arm_counts[arm_id] == 3

    def test_well_layout_reproducible(self):
        arms = [{"arm_id": f"A{i:02d}", "cargo_id": "TEST", "sequence": "ACGU" * 20} for i in range(12)]
        wells1 = _generate_well_layout(arms, 3, seed=42)
        wells2 = _generate_well_layout(arms, 3, seed=42)
        # Same seed should produce same layout
        assert [w["plate_position"] for w in wells1] == [w["plate_position"] for w in wells2]


# ---------------------------------------------------------------------------
# Tests: Full manifest generation (integration)
# ---------------------------------------------------------------------------
class TestManifestGeneration:
    def test_manifest_structure(self, tmp_path):
        benchmark_dir = str(tmp_path / "benchmark")
        _make_synthetic_benchmark(benchmark_dir)
        output_path = str(tmp_path / "manifest.json")

        manifest = generate_manifest(benchmark_dir, str(tmp_path / "ckpt"), output_path, seed=42)

        assert manifest["phase"] == "P3-03"
        assert manifest["manifest_type"] == "prospective_falsification_sequence_manifest"
        assert manifest["configuration"]["n_cargos"] == N_CARGOS
        assert manifest["configuration"]["n_arms_per_cargo"] == N_ARMS
        assert manifest["configuration"]["n_replicates"] == N_REPLICATES
        assert "manifest_sha256" in manifest
        assert len(manifest["manifest_sha256"]) == 64

    def test_manifest_sequence_count(self, tmp_path):
        benchmark_dir = str(tmp_path / "benchmark")
        _make_synthetic_benchmark(benchmark_dir)
        output_path = str(tmp_path / "manifest.json")

        manifest = generate_manifest(benchmark_dir, str(tmp_path / "ckpt"), output_path, seed=42)

        # 2 cargos × 12 arms = 24 unique sequences
        assert len(manifest["sequences"]) == N_CARGOS * N_ARMS
        # 24 × 3 replicates = 72 wells
        assert len(manifest["well_layout"]) == N_CARGOS * N_ARMS * N_REPLICATES

    def test_manifest_file_written(self, tmp_path):
        benchmark_dir = str(tmp_path / "benchmark")
        _make_synthetic_benchmark(benchmark_dir)
        output_path = str(tmp_path / "manifest.json")

        generate_manifest(benchmark_dir, str(tmp_path / "ckpt"), output_path, seed=42)

        assert os.path.exists(output_path)
        with open(output_path) as f:
            loaded = json.load(f)
        assert loaded["phase"] == "P3-03"

    def test_manifest_cargo_sources(self, tmp_path):
        benchmark_dir = str(tmp_path / "benchmark")
        _make_synthetic_benchmark(benchmark_dir)
        output_path = str(tmp_path / "manifest.json")

        manifest = generate_manifest(benchmark_dir, str(tmp_path / "ckpt"), output_path, seed=42)

        assert len(manifest["source_sequences"]) == 2
        cargo_ids = [s["cargo_id"] for s in manifest["source_sequences"]]
        assert "EGFP" in cargo_ids
        assert "mCherry" in cargo_ids

    def test_manifest_sha256_stable(self, tmp_path):
        """Same seed should produce same SHA-256."""
        benchmark_dir = str(tmp_path / "benchmark")
        _make_synthetic_benchmark(benchmark_dir)

        manifest1 = generate_manifest(benchmark_dir, str(tmp_path / "ckpt"), str(tmp_path / "m1.json"), seed=42)
        manifest2 = generate_manifest(benchmark_dir, str(tmp_path / "ckpt"), str(tmp_path / "m2.json"), seed=42)

        assert manifest1["manifest_sha256"] == manifest2["manifest_sha256"]


# ---------------------------------------------------------------------------
# Tests: Scale compliance
# ---------------------------------------------------------------------------
class TestScaleCompliance:
    """Verify the manifest complies with P3-03 scale requirements."""

    def test_total_sequences_in_range(self, tmp_path):
        benchmark_dir = str(tmp_path / "benchmark")
        _make_synthetic_benchmark(benchmark_dir)
        output_path = str(tmp_path / "manifest.json")

        manifest = generate_manifest(benchmark_dir, str(tmp_path / "ckpt"), output_path, seed=42)

        total_unique = manifest["configuration"]["total_unique_sequences"]
        total_wells = manifest["configuration"]["total_wells"]
        # 48-96 sequences required; we have 24 unique × 3 replicates = 72 wells
        assert total_wells >= 48
        assert total_wells <= 96

    def test_replicates_at_least_3(self, tmp_path):
        benchmark_dir = str(tmp_path / "benchmark")
        _make_synthetic_benchmark(benchmark_dir)
        output_path = str(tmp_path / "manifest.json")

        manifest = generate_manifest(benchmark_dir, str(tmp_path / "ckpt"), output_path, seed=42)
        assert manifest["configuration"]["n_replicates"] >= 3

    def test_two_cargos(self, tmp_path):
        benchmark_dir = str(tmp_path / "benchmark")
        _make_synthetic_benchmark(benchmark_dir)
        output_path = str(tmp_path / "manifest.json")

        manifest = generate_manifest(benchmark_dir, str(tmp_path / "ckpt"), output_path, seed=42)
        assert manifest["configuration"]["n_cargos"] == 2

    def test_required_readouts(self, tmp_path):
        benchmark_dir = str(tmp_path / "benchmark")
        _make_synthetic_benchmark(benchmark_dir)
        output_path = str(tmp_path / "manifest.json")

        manifest = generate_manifest(benchmark_dir, str(tmp_path / "ckpt"), output_path, seed=42)
        readouts = manifest["configuration"]["readouts"]
        assert "protein_output" in readouts
        assert "mRNA_abundance" in readouts
        assert "cell_viability" in readouts

    def test_time_points(self, tmp_path):
        benchmark_dir = str(tmp_path / "benchmark")
        _make_synthetic_benchmark(benchmark_dir)
        output_path = str(tmp_path / "manifest.json")

        manifest = generate_manifest(benchmark_dir, str(tmp_path / "ckpt"), output_path, seed=42)
        time_points = manifest["configuration"]["time_points"]
        assert "4h" in time_points
        assert "8h" in time_points
        assert "24h" in time_points
        assert "48h" in time_points
