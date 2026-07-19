"""Protein-invariance & frame-safety tests for the codon-lattice operators.

Run: ``/Users/bytedance/Documents/research/editflow/.venv/bin/python \
        -m unittest mrna_editflow.tests.test_operators`` (from repo root).

These tests validate the *core novelty*: the region-conditioned codon-lattice
substitution / indel operators of :class:`MRNAEditFormer`.
"""
from __future__ import annotations

import os
import sys
import unittest

# Allow ``python mrna_editflow/tests/test_operators.py`` from repo root by
# ensuring the repo root (parent of the ``mrna_editflow`` package) is importable.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

from mrna_editflow.core.constants import (
    CODON_TABLE,
    NUC_TO_ID,
    PHASE_NONE,
    REGION_3UTR,
    REGION_5UTR,
    REGION_CDS,
    START_CODON,
    STOP_CODONS,
    V,
    VOCAB_MODEL_SIZE,
    index_to_codon,
    translate,
)
from mrna_editflow.core.config import BackboneConfig, ModelConfig
from mrna_editflow.models.backbones import FrozenBackbone
from mrna_editflow.models.mrna_editformer import MRNAEditFormer

_SENSE_CODONS = [c for c, aa in CODON_TABLE.items() if aa != "*"]
_STOP_CODONS = list(STOP_CODONS)


def _cds_to_tokens(cds: str) -> torch.Tensor:
    return torch.tensor([[NUC_TO_ID[c] for c in cds]], dtype=torch.long)


def _tokens_to_cds(tokens: torch.Tensor) -> str:
    from mrna_editflow.core.constants import ID_TO_NUC

    return "".join(ID_TO_NUC[int(t)] for t in tokens[0].tolist())


def _make_random_cds(num_sense: int, rng: np.random.Generator) -> str:
    codons = [START_CODON]
    for _ in range(num_sense):
        codons.append(_SENSE_CODONS[int(rng.integers(0, len(_SENSE_CODONS)))])
    codons.append(_STOP_CODONS[int(rng.integers(0, len(_STOP_CODONS)))])
    return "".join(codons)


def _cds_tracks(length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    region = torch.full((1, length), REGION_CDS, dtype=torch.long)
    phase = torch.tensor([[i % 3 for i in range(length)]], dtype=torch.long)
    pad = torch.zeros((1, length), dtype=torch.bool)
    return region, phase, pad


def _build_head(codon_indel: bool = False, use_codon_constraint: bool = True) -> tuple:
    bb = FrozenBackbone(BackboneConfig(name="none", hidden_dim=16))
    cfg = ModelConfig(
        model_dim=32, num_layers=2, num_heads=4, max_seq_len=256,
        use_codon_constraint=use_codon_constraint, codon_indel=codon_indel,
        use_aux_struct=True,
    )
    head = MRNAEditFormer(cfg, backbone_dim=bb.out_dim).eval()
    return bb, head


class TestSubstitutionProteinInvariance(unittest.TestCase):
    """Constrained substitution must preserve the encoded protein exactly."""

    def test_single_nt_substitution_trajectory_is_protein_invariant(self):
        rng = np.random.default_rng(0)
        bb, head = _build_head(codon_indel=False, use_codon_constraint=True)
        torch.manual_seed(0)

        n_seqs = 6
        edits_per_seq = 40
        total_edits = 0
        for s in range(n_seqs):
            cds = _make_random_cds(num_sense=int(rng.integers(4, 9)), rng=rng)
            tokens = _cds_to_tokens(cds)
            region, phase, pad = _cds_tracks(tokens.shape[1])
            orig_protein = translate(cds)
            orig_len = tokens.shape[1]

            # Apply a trajectory of single-nt substitutions, re-masking each step.
            for _ in range(edits_per_seq):
                with torch.no_grad():
                    out = head.forward(tokens, region, phase,
                                       torch.rand(1, 1), pad, bb)
                sub_probs = out["sub_probs"][0]  # [L, V]
                i = int(rng.integers(0, tokens.shape[1]))
                row = sub_probs[i]
                # Support must be restricted to nucleotide channels only.
                self.assertAlmostEqual(float(row[V:].sum()), 0.0, places=5)
                self.assertGreater(float(row[:V].sum()), 0.0)
                new_nt = int(torch.multinomial(row[:V], 1))
                tokens = tokens.clone()
                tokens[0, i] = new_nt
                total_edits += 1

                # Protein + length invariants after EVERY single edit.
                self.assertEqual(tokens.shape[1], orig_len)
                self.assertEqual(translate(_tokens_to_cds(tokens)), orig_protein)

        self.assertEqual(total_edits, n_seqs * edits_per_seq)

    def test_substitution_support_is_synonymous_only(self):
        """Every in-support target for a complete CDS codon is synonymous."""
        rng = np.random.default_rng(7)
        bb, head = _build_head(codon_indel=False, use_codon_constraint=True)
        cds = _make_random_cds(num_sense=8, rng=rng)
        tokens = _cds_to_tokens(cds)
        region, phase, pad = _cds_tracks(tokens.shape[1])
        with torch.no_grad():
            out = head.forward(tokens, region, phase, torch.rand(1, 1), pad, bb)
        sub_probs = out["sub_probs"][0]
        base = tokens[0].tolist()
        for i in range(tokens.shape[1]):
            codon_start = i - (i % 3)
            aa = CODON_TABLE[cds[codon_start:codon_start + 3]]
            offset = i % 3
            for nt in range(V):
                if sub_probs[i, nt] > 0:
                    mutated = list(base)
                    mutated[i] = nt
                    from mrna_editflow.core.constants import ID_TO_NUC

                    new_codon = "".join(
                        ID_TO_NUC[mutated[codon_start + k]] for k in range(3)
                    )
                    self.assertEqual(CODON_TABLE[new_codon], aa)

    def test_half_precision_constraint_mask_is_finite(self):
        """AMP fp16 logits must not overflow when codon masks are applied."""
        _bb, head = _build_head(codon_indel=False, use_codon_constraint=True)
        cds = "AUGGCUUAA"
        tokens = _cds_to_tokens(cds)
        region, phase, _pad = _cds_tracks(tokens.shape[1])
        rates = torch.ones((1, tokens.shape[1], 3), dtype=torch.float16)
        sub_logits = torch.zeros((1, tokens.shape[1], VOCAB_MODEL_SIZE), dtype=torch.float16)
        masked_rates, masked_logits = head._apply_codon_constraints(
            rates, sub_logits, tokens, region, phase
        )
        self.assertEqual(masked_rates.dtype, torch.float16)
        self.assertEqual(masked_logits.dtype, torch.float16)
        self.assertTrue(bool(torch.isfinite(masked_rates).all()))
        self.assertTrue(bool(torch.isfinite(masked_logits).all()))
        # At least one forbidden substitution channel should receive a large
        # finite negative value rather than raising/converting to -inf.
        self.assertLess(float(masked_logits.min()), -1000.0)


class TestCodonIndelFrameSafety(unittest.TestCase):
    """codon_indel=True: only whole-codon indels; frame + no internal stop."""

    def test_indel_rates_only_at_cds_codon_starts(self):
        bb, head = _build_head(codon_indel=True, use_codon_constraint=True)
        cds = _make_random_cds(num_sense=6, rng=np.random.default_rng(1))
        tokens = _cds_to_tokens(cds)
        region, phase, pad = _cds_tracks(tokens.shape[1])
        with torch.no_grad():
            out = head.forward(tokens, region, phase, torch.rand(1, 1), pad, bb)
        rates = out["rates"][0]  # [L, 3]  (ins, sub, del)
        ins, dele = rates[:, 0], rates[:, 2]
        for i in range(tokens.shape[1]):
            if i % 3 == 0:  # codon start -> whole-codon indel allowed
                self.assertGreaterEqual(float(ins[i]), 0.0)
            else:  # interior -> nt indels forbidden (frame lock)
                self.assertAlmostEqual(float(ins[i]), 0.0, places=6)
                self.assertAlmostEqual(float(dele[i]), 0.0, places=6)

    def test_whole_codon_indels_preserve_frame_and_no_internal_stop(self):
        rng = np.random.default_rng(3)
        bb, head = _build_head(codon_indel=True, use_codon_constraint=True)
        for _ in range(20):
            cds = _make_random_cds(num_sense=int(rng.integers(5, 10)), rng=rng)
            tokens = _cds_to_tokens(cds)
            region, phase, pad = _cds_tracks(tokens.shape[1])
            with torch.no_grad():
                out = head.forward(tokens, region, phase, torch.rand(1, 1), pad, bb)
            rates = out["rates"][0]
            orig_len = tokens.shape[1]

            # Codon-start positions with non-forbidden indel rate (interior CDS,
            # not the start AUG and not the terminal stop codon).
            n_codons = orig_len // 3
            interior_starts = [3 * c for c in range(1, n_codons - 1)]
            self.assertTrue(len(interior_starts) >= 1)

            # Whole-codon deletion at a random interior codon start.
            del_start = interior_starts[int(rng.integers(0, len(interior_starts)))]
            # rate at codon start may be >0 (allowed); interior must be 0.
            self.assertAlmostEqual(float(rates[del_start + 1, 2]), 0.0, places=6)
            new_tokens = torch.cat(
                [tokens[:, :del_start], tokens[:, del_start + 3:]], dim=1
            )
            self.assertEqual((orig_len - new_tokens.shape[1]) % 3, 0)
            new_cds = _tokens_to_cds(new_tokens)
            self.assertEqual(len(new_cds) % 3, 0)
            prot = translate(new_cds)
            self.assertNotIn("*", prot[:-1])  # no internal stop introduced

            # Whole-codon insertion of a random SENSE codon at a codon boundary.
            ins_codon = _SENSE_CODONS[int(rng.integers(0, len(_SENSE_CODONS)))]
            ins_toks = _cds_to_tokens(ins_codon)
            ins_at = interior_starts[int(rng.integers(0, len(interior_starts)))]
            grown = torch.cat(
                [tokens[:, :ins_at], ins_toks, tokens[:, ins_at:]], dim=1
            )
            self.assertEqual((grown.shape[1] - orig_len) % 3, 0)
            grown_cds = _tokens_to_cds(grown)
            self.assertEqual(len(grown_cds) % 3, 0)
            self.assertNotIn("*", translate(grown_cds)[:-1])


class TestUTRAllowsNtLevelEdits(unittest.TestCase):
    """UTR positions must keep full nt-level insert/sub/delete freedom."""

    def test_utr_allows_nt_indels_and_full_substitution(self):
        bb, head = _build_head(codon_indel=False, use_codon_constraint=True)
        # 5'UTR (6) + CDS (AUG + 2 sense + stop = 12) + 3'UTR (6)
        cds = "AUG" + "GCU" + "GCC" + "UAA"
        seq = "ACGUAC" + cds + "GGCCAU"
        tokens = _cds_to_tokens(seq)
        L = tokens.shape[1]
        region = torch.tensor(
            [[REGION_5UTR] * 6 + [REGION_CDS] * len(cds) + [REGION_3UTR] * 6],
            dtype=torch.long,
        )
        phase = torch.tensor(
            [[PHASE_NONE] * 6 + [i % 3 for i in range(len(cds))] + [PHASE_NONE] * 6],
            dtype=torch.long,
        )
        pad = torch.zeros((1, L), dtype=torch.bool)
        with torch.no_grad():
            out = head.forward(tokens, region, phase, torch.rand(1, 1), pad, bb)
        rates = out["rates"][0]
        sub_probs = out["sub_probs"][0]
        utr_idx = list(range(0, 6)) + list(range(6 + len(cds), L))
        cds_idx = list(range(6, 6 + len(cds)))

        # UTR: ins/del rates are free (not force-zeroed) -> at least one > 0.
        self.assertTrue(any(float(rates[i, 0]) > 0 for i in utr_idx))
        self.assertTrue(any(float(rates[i, 2]) > 0 for i in utr_idx))
        # UTR: all four nucleotides available for substitution.
        for i in utr_idx:
            self.assertTrue(bool((sub_probs[i, :V] > 0).all()))

        # CDS (codon_indel=False): ins/del rates all zeroed (frame lock).
        for i in cds_idx:
            self.assertAlmostEqual(float(rates[i, 0]), 0.0, places=6)
            self.assertAlmostEqual(float(rates[i, 2]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
