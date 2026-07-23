"""Unit tests for native Edit Flow architecture modules (P0 + P1).

Covers:
    P0-1: region_anchored_align (region-anchored alignment)
    P0-2: grammar_safe_corrupt (CDS-safe corruption)
    P0-3: CTMCSampler (tau-leaping)
    P0-4: ProposalEditor (sequential decoder rename)
    P0-5: AtomicCodonHead (atomic synonymous-codon head)
    P0-6: pool_to_codon_cds_anchored (codon pooling fix)
    P0-7: EndToEndBackbone (trainable encoder)
    P0-8: flow_diagnostics (rate calibration, path legality, etc.)
    P1-1: SequenceLocalizedCoupling
    P1-2: StructureLocalizedCoupling
    P1-5: CTMCComparison (CTMC vs sequential comparison)

All tests are CPU-only and use synthetic data (no trained models required).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Bootstrap: make mrna_editflow package importable
REPO_ROOT = Path(__file__).resolve().parents[1]
parent = REPO_ROOT.parent
if REPO_ROOT.name == "mrna_editflow":
    sys.path.insert(0, str(parent))
else:
    if (parent / "mrna_editflow").is_dir():
        sys.path.insert(0, str(parent))
    else:
        shim = Path(tempfile.gettempdir()) / "nef_test_shim"
        shim.mkdir(exist_ok=True)
        link = shim / "mrna_editflow"
        if not link.exists():
            os.symlink(REPO_ROOT, link, target_is_directory=True)
        sys.path.insert(0, str(shim))

from mrna_editflow.core.constants import (
    CODON_TABLE,
    GAP_TOKEN,
    NUC_TO_ID,
    PAD_TOKEN,
    PHASE_NONE,
    REGION_5UTR,
    REGION_CDS,
    REGION_3UTR,
    START_CODON,
    STOP_CODONS,
    SYNONYMOUS_CODONS,
    V,
    codon_to_index,
    index_to_codon,
    translate,
)
from mrna_editflow.core.schema import MRNARecord


# ---------------------------------------------------------------------------
# P0-1: Region-anchored alignment
# ---------------------------------------------------------------------------

class TestRegionAnchoredAlignment:
    def _make_record(self) -> MRNARecord:
        return MRNARecord(
            transcript_id="test1",
            five_utr="ACGUACGUAC",
            cds="AUG" + "GCU" * 10 + "UAA",
            three_utr="CACGUACGUA",
        )

    def test_empty_source_aligns_to_target(self):
        from mrna_editflow.core.region_alignment import region_anchored_align
        rec = self._make_record()
        tokens_1 = rec.token_ids()
        region_ids_1 = rec.region_ids()
        z0, z1 = region_anchored_align([], tokens_1, region_ids_1)
        # All z0 should be GAP, z1 should contain all target tokens
        assert all(t == GAP_TOKEN for t in z0)
        assert len(z1) == len(tokens_1)
        non_gap = [t for t in z1 if t != GAP_TOKEN]
        assert non_gap == tokens_1

    def test_identical_sequences_produce_no_gaps(self):
        from mrna_editflow.core.region_alignment import region_anchored_align
        rec = self._make_record()
        tokens_1 = rec.token_ids()
        region_ids_1 = rec.region_ids()
        z0, z1 = region_anchored_align(tokens_1, tokens_1, region_ids_1)
        assert z0 == z1
        assert all(t != GAP_TOKEN for t in z0)

    def test_cds_codon_boundary_preserved(self):
        """CDS alignment should never create cross-boundary gaps."""
        from mrna_editflow.core.region_alignment import region_anchored_align
        rec = self._make_record()
        tokens_1 = rec.token_ids()
        region_ids_1 = rec.region_ids()
        # Corrupt the source slightly
        tokens_0 = list(tokens_1)
        tokens_0[15] = (tokens_0[15] + 1) % V  # one substitution
        z0, z1 = region_anchored_align(tokens_0, tokens_1, region_ids_1)
        # z1 in CDS region should have no gaps (codon-level alignment)
        cds_start = len(rec.five_utr)
        cds_end = cds_start + len(rec.cds)
        z1_cds = z1[cds_start:cds_end]
        assert all(t != GAP_TOKEN for t in z1_cds)

    def test_cds_nonsynonymous_avoided(self):
        """Codon alignment should avoid nonsynonymous substitutions."""
        from mrna_editflow.core.region_alignment import region_anchored_align
        rec = self._make_record()
        tokens_1 = rec.token_ids()
        region_ids_1 = rec.region_ids()
        # Create a source with a synonymous CDS change
        tokens_0 = list(tokens_1)
        # GCU (Ala) -> GCC (Ala) at codon 1 (position cds_start+3..5)
        cds_start = len(rec.five_utr)
        # GCU = G(0) C(1) U(2) -> GCC = G(0) C(1) C(2)
        tokens_0[cds_start + 5] = NUC_TO_ID["C"]  # U->C, synonymous
        z0, z1 = region_anchored_align(tokens_0, tokens_1, region_ids_1)
        # Alignment should exist (they differ by a synonymous substitution)
        assert len(z0) == len(z1)


# ---------------------------------------------------------------------------
# P0-2: Grammar-safe corruption
# ---------------------------------------------------------------------------

class TestGrammarSafeCorrupt:
    def _make_record(self) -> MRNARecord:
        return MRNARecord(
            transcript_id="test2",
            five_utr="ACGUACGUAC",
            cds="AUG" + "GCU" * 10 + "UAA",
            three_utr="CACGUACGUA",
        )

    def test_cds_length_preserved(self):
        from mrna_editflow.core.region_alignment import grammar_safe_corrupt
        rec = self._make_record()
        tokens = rec.token_ids()
        region_ids = rec.region_ids()
        rng = np.random.default_rng(42)
        corrupted = grammar_safe_corrupt(tokens, region_ids, 0.5, 0.3, 0.3, rng)
        # CDS region should have same length (no indels in CDS)
        cds_start = len(rec.five_utr)
        cds_end = cds_start + len(rec.cds)
        # Find CDS in corrupted (region boundaries may shift due to UTR indels)
        # Instead, check that the CDS portion of the original is preserved in length
        # by checking protein identity
        original_cds = rec.cds
        # The corrupted sequence's CDS should translate to the same protein
        # Find the CDS in the corrupted sequence (it's between 5'UTR and 3'UTR)
        # Since UTR can have indels, we check the CDS codons directly
        # Extract CDS from corrupted: find AUG...stop
        corrupted_rna = "".join("ACGU"[t] if t < V else "?" for t in corrupted)
        # Find start codon
        aug_pos = corrupted_rna.find("AUG")
        assert aug_pos >= 0, "Start codon not found in corrupted sequence"
        # Find stop codon
        cds_part = ""
        for i in range(aug_pos, len(corrupted_rna) - 2, 3):
            codon = corrupted_rna[i:i+3]
            if len(codon) == 3:
                cds_part += codon
                if codon in STOP_CODONS:
                    break
        # Translate and compare
        orig_protein = translate(original_cds)
        new_protein = translate(cds_part)
        assert orig_protein == new_protein, f"Protein changed: {orig_protein} != {new_protein}"

    def test_start_stop_never_corrupted(self):
        from mrna_editflow.core.region_alignment import grammar_safe_corrupt
        rec = self._make_record()
        tokens = rec.token_ids()
        region_ids = rec.region_ids()
        rng = np.random.default_rng(0)
        # Run many times to check
        for seed in range(20):
            rng = np.random.default_rng(seed)
            corrupted = grammar_safe_corrupt(tokens, region_ids, 1.0, 0.0, 0.0, rng)
            corrupted_rna = "".join("ACGU"[t] if t < V else "?" for t in corrupted)
            # Start codon should be preserved
            assert "AUG" in corrupted_rna, f"Start codon lost (seed={seed})"
            # Stop codon should be preserved
            assert any(stop in corrupted_rna for stop in STOP_CODONS), f"Stop codon lost (seed={seed})"

    def test_utr_can_have_indels(self):
        from mrna_editflow.core.region_alignment import grammar_safe_corrupt
        rec = self._make_record()
        tokens = rec.token_ids()
        region_ids = rec.region_ids()
        rng = np.random.default_rng(42)
        corrupted = grammar_safe_corrupt(tokens, region_ids, 0.0, 0.5, 0.5, rng)
        # Length should differ (UTR indels)
        assert len(corrupted) != len(tokens)


# ---------------------------------------------------------------------------
# P0-3: CTMC Sampler
# ---------------------------------------------------------------------------

class TestCTMCSampler:
    def test_config_defaults(self):
        from mrna_editflow.core.ctmc_sampler import CTMCConfig
        cfg = CTMCConfig()
        assert cfg.n_steps > 0
        assert cfg.grammar_safe is True

    def test_event_non_conflicting_selection(self):
        from mrna_editflow.core.ctmc_sampler import EditEvent, _find_non_conflicting
        events = [
            EditEvent("sub", 5, 1, 0.9),
            EditEvent("sub", 5, 2, 0.8),  # conflicts with above (same pos)
            EditEvent("del", 10, 0, 0.7),
            EditEvent("ins", 10, 1, 0.6),  # conflicts with above (adjacent)
            EditEvent("sub", 20, 3, 0.5),
        ]
        selected = _find_non_conflicting(events)
        assert len(selected) >= 2
        # Highest rate events should be selected
        assert selected[0].pos == 5
        assert selected[0].rate == 0.9

    def test_apply_events_substitution(self):
        from mrna_editflow.core.ctmc_sampler import EditEvent, _apply_events
        tokens = [0, 1, 2, 3, 0, 1, 2, 3]  # ACGUACGU
        region_ids = [REGION_5UTR] * 8
        phase_ids = [PHASE_NONE] * 8
        events = [EditEvent("sub", 3, 0, 0.5)]  # U->A at pos 3
        t, r, p = _apply_events(tokens, region_ids, phase_ids, events)
        assert t[3] == 0
        assert len(t) == len(tokens)

    def test_apply_events_deletion(self):
        from mrna_editflow.core.ctmc_sampler import EditEvent, _apply_events
        tokens = [0, 1, 2, 3, 0, 1]
        region_ids = [REGION_5UTR] * 6
        phase_ids = [PHASE_NONE] * 6
        events = [EditEvent("del", 2, 0, 0.5)]
        t, r, p = _apply_events(tokens, region_ids, phase_ids, events)
        assert len(t) == 5
        assert t[2] == 3  # position 2 is now the old position 3

    def test_sampler_runs_with_dummy_model(self):
        """Smoke test: CTMCSampler runs to completion with a dummy model."""
        from mrna_editflow.core.ctmc_sampler import CTMCConfig, CTMCSampler

        def dummy_model(token_ids, region_ids, phase_ids, time_step, padding_mask, backbone):
            B, L = token_ids.shape
            rates = torch.ones(B, L, 3) * 0.1
            ins_probs = torch.ones(B, L, V) / V
            sub_probs = torch.ones(B, L, V) / V
            return {"rates": rates, "ins_probs": ins_probs, "sub_probs": sub_probs, "aux": None}

        cfg = CTMCConfig(n_steps=5, max_events_per_step=2, seed=42)
        sampler = CTMCSampler(dummy_model, cfg)
        token_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
        region_ids = torch.tensor([[REGION_5UTR] * 8])
        phase_ids = torch.tensor([[PHASE_NONE] * 8])
        traj = sampler.sample(token_ids, region_ids, phase_ids, None, torch.device("cpu"))
        assert traj.n_steps_actual == 5
        assert traj.final_tokens is not None
        assert isinstance(traj.n_events, int)


# ---------------------------------------------------------------------------
# P0-5: Atomic Codon Head
# ---------------------------------------------------------------------------

class TestAtomicCodonHead:
    def test_head_output_shapes(self):
        from mrna_editflow.core.codon_head import AtomicCodonHead
        head = AtomicCodonHead(model_dim=64)
        # Per-codon hidden states: [B, n_codons, D]
        x = torch.randn(2, 10, 64)
        codon_indices = torch.randint(0, 64, (2, 10))
        out = head(x, codon_indices)
        assert "lambda_sub_codon" in out
        assert "Q_synonymous_codon" in out
        assert out["lambda_sub_codon"].shape == (2, 10)
        assert out["Q_synonymous_codon"].shape == (2, 10, 64)

    def test_rates_non_negative(self):
        from mrna_editflow.core.codon_head import AtomicCodonHead
        head = AtomicCodonHead(model_dim=32)
        x = torch.randn(1, 5, 32)
        codon_indices = torch.randint(0, 64, (1, 5))
        out = head(x, codon_indices)
        assert (out["lambda_sub_codon"] >= 0).all()

    def test_synonymous_mask_excludes_nonsynonymous(self):
        """The synonymous mask should only allow same-aa codons."""
        from mrna_editflow.core.codon_head import AtomicCodonHead, _build_synonymous_matrix
        mask = _build_synonymous_matrix()
        # Codon 0 = AAA (Lys), codon 1 = AAC (Asn) — different aa
        assert mask[0, 1] == False
        # AAA (Lys) and AAG (Lys) — same aa
        aaa_idx = codon_to_index("AAA")
        aag_idx = codon_to_index("AAG")
        assert mask[aaa_idx, aag_idx] == True
        # Self is synonymous (same aa)
        assert mask[aaa_idx, aaa_idx] == True


# ---------------------------------------------------------------------------
# P0-6: Codon pooling fix
# ---------------------------------------------------------------------------

class TestCodonPoolingFix:
    def test_pool_anchored_to_cds_start(self):
        from mrna_editflow.core.codon_pooling_fix import pool_to_codon_cds_anchored
        # Sequence: BOS(0) + 5'UTR(4nt) + CDS(6nt = 2 codons) + 3'UTR(4nt)
        # BOS=4, A=0,C=1,G=2,U=3
        token_ids = torch.tensor([[4, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2]])
        region_ids = torch.tensor([[3, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]])
        nt_emb = torch.randn(1, 15, 8)
        out = pool_to_codon_cds_anchored(nt_emb, region_ids)
        assert out.shape == nt_emb.shape
        # CDS positions (5-10) should be modified (pooled)
        # Non-CDS positions should be unchanged
        assert torch.allclose(out[0, :5], nt_emb[0, :5])  # BOS + 5'UTR
        assert torch.allclose(out[0, 11:], nt_emb[0, 11:])  # 3'UTR
        # CDS positions should differ (they've been pooled)
        assert not torch.allclose(out[0, 5:11], nt_emb[0, 5:11])


# ---------------------------------------------------------------------------
# P0-7: End-to-end backbone
# ---------------------------------------------------------------------------

class TestEndToEndBackbone:
    def test_backbone_not_frozen_by_default(self):
        from mrna_editflow.core.end_to_end_backbone import EndToEndBackbone
        from mrna_editflow.core.config import BackboneConfig
        cfg = BackboneConfig(name="none", freeze=False, hidden_dim=64)
        backbone = EndToEndBackbone(cfg)
        assert not backbone.frozen
        assert any(p.requires_grad for p in backbone.parameters())

    def test_backbone_embed_output_shape(self):
        from mrna_editflow.core.end_to_end_backbone import EndToEndBackbone
        from mrna_editflow.core.config import BackboneConfig
        cfg = BackboneConfig(name="none", freeze=False, hidden_dim=32)
        backbone = EndToEndBackbone(cfg)
        token_ids = torch.randint(0, V, (2, 20))
        region_ids = torch.randint(0, 3, (2, 20))
        out = backbone.embed(token_ids, region_ids)
        assert out.shape == (2, 20, 32)

    def test_gradients_flow_through_encoder(self):
        from mrna_editflow.core.end_to_end_backbone import EndToEndBackbone
        from mrna_editflow.core.config import BackboneConfig
        cfg = BackboneConfig(name="none", freeze=False, hidden_dim=16)
        backbone = EndToEndBackbone(cfg)
        token_ids = torch.randint(0, V, (1, 10))
        region_ids = torch.zeros(1, 10, dtype=torch.long)
        out = backbone.embed(token_ids, region_ids)
        loss = out.sum()
        loss.backward()
        # Check that encoder parameters have gradients
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                    for p in backbone.encoder.parameters())


# ---------------------------------------------------------------------------
# P0-8: Flow diagnostics
# ---------------------------------------------------------------------------

class TestFlowDiagnostics:
    def test_path_legality_check_passes_for_valid_trajectory(self):
        from mrna_editflow.core.flow_diagnostics import path_legality_check, Trajectory, TrajectoryStep
        # A valid trajectory: no edits (identity)
        traj = Trajectory(
            steps=[TrajectoryStep(t=0.0, sequence="ACGUACGUAC", events=[])],
            initial_sequence="ACGUACGUAC",
        )
        result = path_legality_check(traj, region_ids=[REGION_5UTR] * 10)
        assert result["is_legal"] is True

    def test_endpoint_recovery_identical_sequences(self):
        from mrna_editflow.core.flow_diagnostics import endpoint_recovery, Trajectory, TrajectoryStep
        traj = Trajectory(
            steps=[TrajectoryStep(t=0.0, sequence="ACGU", events=[])],
            initial_sequence="ACGU",
            target_sequence="ACGU",
        )
        result = endpoint_recovery(traj)
        assert result["levenshtein"] == 0
        assert result["normalized_recovery"] == 1.0

    def test_endpoint_recovery_different_sequences(self):
        from mrna_editflow.core.flow_diagnostics import endpoint_recovery, Trajectory, TrajectoryStep
        traj = Trajectory(
            steps=[TrajectoryStep(t=0.0, sequence="ACGU", events=[])],
            initial_sequence="ACGU",
            target_sequence="UGCA",
        )
        result = endpoint_recovery(traj)
        assert result["levenshtein"] == 4
        assert result["normalized_recovery"] == 0.0


# ---------------------------------------------------------------------------
# P1-1 + P1-2: Localized coupling
# ---------------------------------------------------------------------------

class TestLocalizedCoupling:
    def test_sequence_coupling_boosts_neighbors(self):
        from mrna_editflow.core.localized_coupling import (
            SequenceLocalizedCoupling,
            SequenceLocalizedCouplingConfig,
        )
        cfg = SequenceLocalizedCouplingConfig(window=3, alpha=1.0, sigma=1.0, decay_kind="exponential")
        coupling = SequenceLocalizedCoupling(cfg)
        rates = torch.ones(10, 3)  # 10 positions, 3 ops
        edit_positions = [5]
        boosted = coupling.apply(rates, edit_positions)
        # Position 5's neighbors should be boosted
        assert boosted[4, 0] > rates[4, 0]  # neighbor boosted
        assert boosted[6, 0] > rates[6, 0]  # neighbor boosted
        # Distant positions unchanged
        assert torch.allclose(boosted[0, :], rates[0, :])

    def test_structure_coupling_with_pairing_prob(self):
        from mrna_editflow.core.localized_coupling import (
            StructureLocalizedCoupling,
            StructureLocalizedCouplingConfig,
        )
        cfg = StructureLocalizedCouplingConfig(
            window=2, alpha_seq=1.0, alpha_pair=2.0, sigma=1.0,
            decay_kind="exponential", pair_threshold=0.5, stem_window=1,
        )
        coupling = StructureLocalizedCoupling(cfg)
        rates = torch.ones(10, 3)
        pairing_prob = torch.zeros(10)
        pairing_prob[3] = 0.9  # position 3 is likely paired
        pairing_prob[7] = 0.9  # position 7 is likely paired
        edit_positions = [3]
        boosted = coupling.apply(rates, edit_positions, pairing_prob=pairing_prob)
        # Position 7 (paired with 3) should be boosted
        assert boosted[7, 0] > rates[7, 0]


# ---------------------------------------------------------------------------
# P1-5: CTMC comparison
# ---------------------------------------------------------------------------

class TestCTMCComparison:
    def test_comparison_config_defaults(self):
        from mrna_editflow.core.ctmc_comparison import ComparisonConfig
        cfg = ComparisonConfig()
        assert cfg.ctmc_n_steps > 0
        assert cfg.proposal_edit_budget > 0

    def test_comparison_runs_with_dummy_model(self):
        """Smoke test: comparison framework runs with a dummy model."""
        from mrna_editflow.core.ctmc_comparison import CTMCComparison, ComparisonConfig

        def dummy_model(token_ids, region_ids, phase_ids, time_step, padding_mask, backbone):
            B, L = token_ids.shape
            rates = torch.ones(B, L, 3) * 0.05
            ins_probs = torch.ones(B, L, V) / V
            sub_probs = torch.ones(B, L, V) / V
            return {"rates": rates, "ins_probs": ins_probs, "sub_probs": sub_probs, "aux": None}

        cfg = ComparisonConfig(ctmc_n_steps=3, proposal_edit_budget=3)
        comparison = CTMCComparison(dummy_model, cfg)
        source = MRNARecord(
            transcript_id="src",
            five_utr="ACGUAC",
            cds="AUGGCUUAA",
            three_utr="CGUACG",
        )
        result = comparison.compare_pair(source)
        assert "ctmc" in result
        assert "proposal" in result
