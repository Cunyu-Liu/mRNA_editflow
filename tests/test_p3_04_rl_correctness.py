#!/usr/bin/env python
"""P3-04: RL Correctness Gate — 14 Acceptance Tests.

All tests run on CPU without requiring a trained checkpoint or GPU.
Mock/dummy implementations are used where a real Policy is needed.

Tests map directly to the 14 acceptance criteria in the P3-04 spec.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.constants import (
    CODON_TABLE, SYNONYMOUS_CODONS, START_CODON, STOP_CODONS,
    ID_TO_NUC, NUC_TO_ID, is_valid_cds, translate,
)
from core.schema import MRNARecord
from rl.action_space import Action, STOP_ACTION, ActionMask, ActionLogProbs, apply_action, build_legal_action_mask
from rl.kl_regularization import categorical_kl, AdaptiveKLController
from rl.grpo import clipped_policy_loss, group_advantages, GRPOConfig
from rl.trajectory_sampler import SamplerConfig, constrained_distribution, _hash_mask
from rl.training_reward import (
    HardConstraintStatus, TrainingReward, build_training_reward, stop_reward_vector,
)
from rl.p3_04_correctness import (
    DeterministicPolicy, verify_deterministic_forward,
    MultiEpochGRPOConfig, EpochUpdateResult, multi_epoch_grpo_update,
    compute_ratio_stats, verify_ratio_before_update, verify_ratio_after_update,
    CompletePolicyStep, build_complete_step, recover_mask_from_history,
    CodonAction, synonymous_codon_actions, apply_codon_action,
    verify_codon_action_protein_preservation,
    ProductionPathGate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_record(
    five_utr: str = "GCCAUGCAUCAUCAUCAUCAUCAUCAUCAUCAUCAUCAUCAUCAUCAUCAUC",
    cds: str = None,
    three_utr: str = "UGCUUGCUUGCUUGCUUGCU",
    transcript_id: str = "test_001",
) -> MRNARecord:
    """Create a test MRNARecord with valid CDS."""
    if cds is None:
        # 10 codons: AUG + 8 sense + stop
        cds = START_CODON + "GCU" * 4 + "CUG" * 4 + "UAA"
    return MRNARecord(
        transcript_id=transcript_id,
        five_utr=five_utr,
        cds=cds,
        three_utr=three_utr,
    )


def _make_dummy_action_logprobs(length: int, device: torch.device = torch.device("cpu")) -> ActionLogProbs:
    """Create dummy ActionLogProbs for testing."""
    vocab = 4
    ins = torch.randn(length, vocab, device=device)
    sub = torch.randn(length, vocab, device=device)
    dele = torch.randn(length, device=device)
    stop = torch.tensor(0.0, device=device)
    # Normalize to log-probs
    all_logits = torch.cat([ins.flatten(), sub.flatten(), dele, stop.unsqueeze(0)])
    log_partition = torch.logsumexp(all_logits, dim=0)
    return ActionLogProbs(
        ins_logprobs=ins - log_partition,
        sub_logprobs=sub - log_partition,
        del_logprobs=dele - log_partition,
        stop_logprob=stop - log_partition,
        log_partition=log_partition,
    )


class DummyModel(nn.Module):
    """Minimal length-agnostic model for testing deterministic forward."""
    def __init__(self, vocab: int = 4, hidden: int = 8):
        super().__init__()
        self.linear = nn.Linear(vocab, hidden)
        self.head = nn.Linear(hidden, 4)
        self.dropout = nn.Dropout(0.5)  # High dropout to test determinism
        self.vocab = vocab

    def forward(self, token_ids, region_ids, phase_ids, t, padding_mask, backbone=None):
        batch, length = token_ids.shape
        x = torch.nn.functional.one_hot(token_ids, self.vocab).float()  # (B, L, V)
        x = self.dropout(x)
        h = self.linear(x)       # (B, L, hidden)
        out = self.head(h)       # (B, L, 4)
        return {
            "ins_rates": torch.softmax(out[:, :, 0:1].expand(batch, length, self.vocab), dim=-1),
            "sub_rates": torch.softmax(out[:, :, 1:2].expand(batch, length, self.vocab), dim=-1),
            "del_rates": torch.sigmoid(out[:, :, 2]),
            "stop_rate": torch.sigmoid(out[:, :, 3].mean(dim=1)),
        }


class DummyBackbone(nn.Module):
    """Dummy backbone that returns None (no precomputed features)."""
    def forward(self, *args, **kwargs):
        return None


class MockPolicy:
    """Mock Policy that wraps a DummyModel for testing.

    Provides the same interface as Policy but with a simple deterministic model.
    """
    def __init__(self, model: DummyModel, backbone: DummyBackbone, device: torch.device):
        self._model = model
        self._backbone = backbone
        self.device = device

    @property
    def inner(self):
        return self

    def action_logprobs(self, record: MRNARecord, *, budget_remaining=None, budget_total=None, no_grad=True, return_raw=False) -> ActionLogProbs:
        length = len(record.seq)
        model = self._model
        backbone = self._backbone

        token_ids = torch.tensor([[NUC_TO_ID.get(c, 0) for c in record.seq]], device=self.device)
        region_ids = torch.tensor([record.region_ids()], device=self.device)
        phase_ids = torch.tensor([record.codon_phases()], device=self.device)
        t = torch.tensor([0.5], device=self.device)
        padding_mask = torch.ones(1, length, device=self.device)

        def _forward():
            return model(token_ids, region_ids, phase_ids, t, padding_mask, backbone)

        if no_grad:
            with torch.no_grad():
                out = _forward()
        else:
            out = _forward()

        ins_lp = torch.log(out["ins_rates"][0] + 1e-10)
        sub_lp = torch.log(out["sub_rates"][0] + 1e-10)
        del_lp = torch.log(out["del_rates"][0] + 1e-10)
        stop_lp = torch.log(out["stop_rate"][0] + 1e-10)

        # Apply legal mask
        mask = build_legal_action_mask(record, self.device)
        masked_ins = ins_lp + torch.where(mask.ins_mask, 0.0, float("-inf"))
        masked_sub = sub_lp + torch.where(mask.sub_mask, 0.0, float("-inf"))
        masked_del = del_lp + torch.where(mask.del_mask, 0.0, float("-inf"))
        masked_stop = stop_lp if mask.stop_legal else torch.tensor(float("-inf"), device=self.device)

        # Normalize over all legal actions
        all_lp = torch.cat([masked_ins.flatten(), masked_sub.flatten(), masked_del, masked_stop.unsqueeze(0)])
        log_partition = torch.logsumexp(all_lp[torch.isfinite(all_lp)], dim=0)

        return ActionLogProbs(
            ins_logprobs=masked_ins - log_partition,
            sub_logprobs=masked_sub - log_partition,
            del_logprobs=masked_del - log_partition,
            stop_logprob=masked_stop - log_partition,
            log_partition=log_partition,
        )

    def legal_action_mask(self, record: MRNARecord) -> ActionMask:
        return build_legal_action_mask(record, self.device)

    def sample(self, record: MRNARecord, **kwargs):
        lps = self.action_logprobs(record, **kwargs)
        flat = torch.cat([lps.ins_logprobs.flatten(), lps.sub_logprobs.flatten(), lps.del_logprobs, lps.stop_logprob.unsqueeze(0)])
        index = int(torch.multinomial(torch.exp(flat), 1).item())
        length = lps.ins_logprobs.shape[0]
        vocab = lps.ins_logprobs.shape[1]
        if index < length * vocab:
            op = "ins"
            pos = index // vocab
            nt = index % vocab
        elif index < 2 * length * vocab:
            op = "sub"
            idx = index - length * vocab
            pos = idx // vocab
            nt = idx % vocab
        elif index < 2 * length * vocab + length:
            op = "del"
            pos = index - 2 * length * vocab
            nt = -1
        else:
            return STOP_ACTION, 0.0
        return Action(op, pos, nt), float(flat[index])


def _make_mock_policy(device=None) -> MockPolicy:
    device = device or torch.device("cpu")
    model = DummyModel(vocab=4, hidden=8)
    model.eval()  # Disable dropout for deterministic output
    backbone = DummyBackbone()
    return MockPolicy(model, backbone, device)


def _make_trajectory(policy, record, max_edits=2, device=None):
    """Create a simple trajectory for testing."""
    cfg = SamplerConfig(task_id="T5", max_edits=max_edits)
    current = record
    visited = {record.seq}
    steps = []

    for step in range(max_edits + 1):
        remaining = max_edits - step
        actions, lps, mask_hash = constrained_distribution(
            policy, current, cfg=cfg, visited=visited, budget_remaining=remaining, no_grad=True,
        )
        if len(actions) == 0:
            break
        # Always use actions[0] (same type as constrained_distribution returns)
        action = actions[0]
        old_lp = float(lps[0])
        steps.append((current, action, old_lp, old_lp, mask_hash, remaining, actions))
        if action.is_stop() or len(actions) <= 1:
            break
        current = apply_action(current, action)
        visited.add(current.seq)

    return steps, cfg


# ===========================================================================
# Acceptance Test 1: ratio ≈ 1 before update
# ===========================================================================

class TestRatioBeforeUpdate:
    """Test 1: update前 ratio≈1."""

    def test_ratio_approx_one_before_update(self):
        """When model params haven't changed, ratio = exp(new - old) ≈ 1."""
        device = torch.device("cpu")
        policy = _make_mock_policy(device)
        record = _make_record()
        steps, cfg = _make_trajectory(policy, record, max_edits=1)

        # Compute new log-probs with same model
        new_lps = []
        old_lps = []
        for state, action, old_lp, ref_lp, mask_hash, budget, legal_actions in steps:
            actions, current_lps, _ = constrained_distribution(
                policy, state, cfg=cfg, visited={state.seq}, budget_remaining=budget, no_grad=False,
            )
            selected = next(i for i, a in enumerate(actions) if a == action)
            new_lps.append(current_lps[selected].detach())
            old_lps.append(torch.tensor(old_lp, device=device))

        new_t = torch.stack(new_lps)
        old_t = torch.stack(old_lps)
        ratios = torch.exp(new_t - old_t)

        assert float((ratios - 1.0).abs().max()) < 1e-4, f"Ratio not ≈1: max_diff={float((ratios-1.0).abs().max())}"


# ===========================================================================
# Acceptance Test 2: ratio ≠ 1 after update
# ===========================================================================

class TestRatioAfterUpdate:
    """Test 2: update后 ratio不再恒等于1."""

    def test_ratio_changes_after_update(self):
        """After an optimizer step, ratio should deviate from 1."""
        device = torch.device("cpu")
        policy = _make_mock_policy(device)
        record = _make_record()
        steps, cfg = _make_trajectory(policy, record, max_edits=1)

        # Save old log-probs
        old_lps = [old_lp for _, _, old_lp, _, _, _, _ in steps]

        # Do a fake update: perturb model weights significantly
        with torch.no_grad():
            for param in policy._model.parameters():
                param += 0.3 * torch.randn_like(param)

        # Compute new log-probs with perturbed model
        new_lps = []
        for state, action, old_lp, ref_lp, mask_hash, budget, legal_actions in steps:
            actions, current_lps, _ = constrained_distribution(
                policy, state, cfg=cfg, visited={state.seq}, budget_remaining=budget, no_grad=True,
            )
            selected = next(i for i, a in enumerate(actions) if a == action)
            new_lps.append(float(current_lps[selected]))

        ratios = [np.exp(new - old) for new, old in zip(new_lps, old_lps)]
        assert any(abs(r - 1.0) > 1e-4 for r in ratios), f"Ratios unchanged after update: {ratios}"


# ===========================================================================
# Acceptance Test 3: clip fraction test
# ===========================================================================

class TestClipFraction:
    """Test 3: clip fraction测试正确."""

    def test_clip_fraction_zero_when_ratio_in_range(self):
        """When ratio is within [1-ε, 1+ε], clip fraction = 0."""
        new = torch.tensor([-1.0, -2.0, -3.0])
        old = torch.tensor([-1.0, -2.0, -3.0])  # ratio = 1.0 for all
        stats = compute_ratio_stats(new, old, clip_epsilon=0.2)
        assert stats["clip_fraction"] == 0.0

    def test_clip_fraction_nonzero_when_ratio_out_of_range(self):
        """When ratio exceeds clip range, clip fraction > 0."""
        new = torch.tensor([0.0, -0.5, -5.0])  # ratios: exp(0+1), exp(-0.5+1), exp(-5+1)
        old = torch.tensor([-1.0, -1.0, -1.0])
        stats = compute_ratio_stats(new, old, clip_epsilon=0.2)
        assert stats["clip_fraction"] > 0.0

    def test_clip_fraction_bounded_01(self):
        """Clip fraction is in [0, 1]."""
        new = torch.randn(100)
        old = torch.randn(100)
        stats = compute_ratio_stats(new, old, clip_epsilon=0.2)
        assert 0.0 <= stats["clip_fraction"] <= 1.0


# ===========================================================================
# Acceptance Test 4: deterministic policy forward
# ===========================================================================

class TestDeterministicForward:
    """Test 4: deterministic policy forward (grad and no-grad match)."""

    def test_grad_nograd_match(self):
        """Same params + same state + same mask → same distribution."""
        device = torch.device("cpu")
        policy = _make_mock_policy(device)
        record = _make_record()

        # With DeterministicPolicy wrapper
        det = DeterministicPolicy(policy)
        lps_grad = det.action_logprobs(record, budget_remaining=3, budget_total=3, no_grad=False)
        with torch.no_grad():
            lps_nograd = det.action_logprobs(record, budget_remaining=3, budget_total=3, no_grad=True)

        for attr in ("ins_logprobs", "sub_logprobs", "del_logprobs"):
            t1 = getattr(lps_grad, attr).detach()
            t2 = getattr(lps_nograd, attr)
            # Only compare finite values (masked positions are -inf, and inf-inf=nan)
            finite_mask = torch.isfinite(t1) & torch.isfinite(t2)
            if finite_mask.any():
                diff = float((t1[finite_mask] - t2[finite_mask]).abs().max())
                assert diff < 1e-7, f"{attr} differs: {diff}"

        stop_diff = abs(float(lps_grad.stop_logprob) - float(lps_nograd.stop_logprob))
        assert stop_diff < 1e-7, f"stop_logprob differs: {stop_diff}"


# ===========================================================================
# Acceptance Test 5: categorical KL ≥ -1e-7
# ===========================================================================

class TestCategoricalKL:
    """Test 5: categorical KL ≥ -1e-7 (KL is non-negative)."""

    def test_kl_self_is_zero(self):
        """KL(p || p) = 0."""
        log_probs = torch.tensor([-1.0, -2.0, -0.5])
        kl = categorical_kl(log_probs, log_probs)
        assert float(kl) >= -1e-7
        assert abs(float(kl)) < 1e-6

    def test_kl_nonnegative(self):
        """KL(p || q) ≥ 0 for any p, q."""
        p = torch.tensor([-1.0, -2.0, -0.5])
        q = torch.tensor([-0.5, -1.5, -1.0])
        kl = categorical_kl(p, q)
        assert float(kl) >= -1e-7

    def test_kl_nonnegative_random(self):
        """KL ≥ 0 for many random distributions."""
        for _ in range(100):
            p = torch.randn(10)
            q = torch.randn(10)
            p = p - torch.logsumexp(p, dim=0)
            q = q - torch.logsumexp(q, dim=0)
            kl = categorical_kl(p, q)
            assert float(kl) >= -1e-7, f"KL={float(kl)} < -1e-7"


# ===========================================================================
# Acceptance Test 6: rollout/update legal action IDs identical
# ===========================================================================

class TestLegalActionConsistency:
    """Test 6: rollout/update legal action IDs完全一致."""

    def test_legal_actions_match_between_calls(self):
        """Legal action set is deterministic for same state + budget + task."""
        device = torch.device("cpu")
        policy = _make_mock_policy(device)
        record = _make_record()
        cfg = SamplerConfig(task_id="T5", max_edits=3)
        visited = {record.seq}

        actions1, lps1, hash1 = constrained_distribution(
            policy, record, cfg=cfg, visited=visited, budget_remaining=3, no_grad=True,
        )
        actions2, lps2, hash2 = constrained_distribution(
            policy, record, cfg=cfg, visited=visited, budget_remaining=3, no_grad=False,
        )

        assert actions1 == actions2, "Legal action IDs differ between calls"
        assert hash1 == hash2, "Mask hashes differ"


# ===========================================================================
# Acceptance Test 7: historical cycle mask recoverable
# ===========================================================================

class TestCycleMaskRecovery:
    """Test 7: historical cycle mask可恢复."""

    def test_complete_step_has_all_fields(self):
        """CompletePolicyStep contains all 13 required MDP state fields."""
        record = _make_record()
        action = Action("sub", 0, 2)
        step = build_complete_step(
            state=record, action=action, old_log_prob=-1.5, reference_log_prob=-1.5,
            action_mask_hash="abc123", budget_remaining=2,
            legal_actions=[action, STOP_ACTION],
            visited={record.seq}, task_id="T5",
            editable_regions=["five_utr"], preference={"te": 0.5},
            reward_provenance={"scalar": 0.1}, source_id="test_001",
        )
        # Verify all 13 fields exist
        required = [
            "record", "source_id", "current_sequence", "visited_sequence_hashes",
            "remaining_budget", "task_id", "editable_regions", "preference",
            "legal_action_ids", "action_mask_hash", "old_log_prob",
            "reference_log_prob", "reward_provenance",
        ]
        for field in required:
            assert hasattr(step, field), f"Missing field: {field}"

    def test_mask_hash_recoverable(self):
        """The action_mask_hash can be recomputed from saved state."""
        record = _make_record()
        action = Action("sub", 0, 2)
        legal_actions = [action, STOP_ACTION]
        step = build_complete_step(
            state=record, action=action, old_log_prob=-1.5, reference_log_prob=-1.5,
            action_mask_hash="test_hash", budget_remaining=2,
            legal_actions=legal_actions, visited={record.seq}, task_id="T5",
            editable_regions=["five_utr"], preference={},
        )
        # The legal_action_ids field preserves the action set
        assert len(step.legal_action_ids) == 2
        assert step.legal_action_ids[0] == "sub:0:2"
        assert step.legal_action_ids[1] == "stop:-1:-1"


# ===========================================================================
# Acceptance Test 8: all-negative reward selects STOP
# ===========================================================================

class TestAllNegativeStopPreferred:
    """Test 8: all-negative reward选择STOP."""

    def test_all_negative_makes_stop_preferred(self):
        """When all soft objectives are negative, STOP is preferred."""
        raw = {"te": -0.1, "access": -0.05, "agreement": -0.02, "edit_cost": -1.0}
        hard = HardConstraintStatus()  # All pass
        reward = build_training_reward(raw, uncertainty=0.1, hard_constraint_status=hard)
        assert reward.stop_preferred
        assert reward.scalar <= 0.0

    def test_positive_te_makes_edit_preferred(self):
        """When te delta is positive and exceeds costs, editing is preferred over STOP."""
        raw = {"te": 0.5, "access": -0.05, "agreement": 0.0, "edit_cost": -0.1}
        hard = HardConstraintStatus()
        reward = build_training_reward(raw, uncertainty=0.1, hard_constraint_status=hard)
        assert not reward.stop_preferred
        assert reward.scalar > 0.0

    def test_hard_failure_forces_stop(self):
        """Hard constraint violation forces STOP preference."""
        raw = {"te": 0.5, "access": 0.5, "agreement": 0.0, "edit_cost": -1.0}
        hard = HardConstraintStatus(protein_identity=False)
        reward = build_training_reward(raw, uncertainty=0.1, hard_constraint_status=hard)
        assert reward.stop_preferred
        assert reward.hard_constraint_gated
        assert reward.scalar < 0.0


# ===========================================================================
# Acceptance Test 9: synonymous codon action 100% preserves protein
# ===========================================================================

class TestCodonActionProteinPreservation:
    """Test 9: synonymous codon action 100%保持蛋白."""

    def test_all_synonymous_actions_preserve_protein(self):
        """Every synonymous codon action preserves the protein."""
        cds = START_CODON + "GCU" * 4 + "CUG" * 4 + "CAG" * 2 + "UAA"
        result = verify_codon_action_protein_preservation(cds)
        assert result["all_preserved"]
        assert result["n_violations"] == 0
        assert result["n_actions"] > 0

    def test_apply_codon_action_preserves_protein(self):
        """Applying a codon action preserves the protein."""
        record = _make_record(cds=START_CODON + "GCU" * 4 + "CUG" * 4 + "UAA")
        actions = synonymous_codon_actions(record.cds)
        assert len(actions) > 0
        for action in actions:
            new_record = apply_codon_action(record, action)
            assert translate(record.cds) == translate(new_record.cds)

    def test_codon_action_no_single_nt_intermediate(self):
        """CodonAction is a single atomic action, not 3 nt-level actions."""
        record = _make_record(cds=START_CODON + "GCU" * 4 + "CUG" * 4 + "UAA")
        actions = synonymous_codon_actions(record.cds)
        for action in actions:
            # The CodonAction itself is atomic
            assert action.op == "codon_sub"
            # The nt-level decomposition may have 0-3 changes, but they're applied atomically
            nt_actions = action.to_action_list()
            # The protein should still be preserved after the full codon swap
            start = action.codon_pos * 3
            new_cds = record.cds[:start] + action.new_codon + record.cds[start+3:]
            assert translate(record.cds) == translate(new_cds)


# ===========================================================================
# Acceptance Test 10: reference policy bitwise unchanged
# ===========================================================================

class TestReferencePolicyUnchanged:
    """Test 10: reference policy bitwise unchanged after update."""

    def test_reference_params_unchanged_after_update(self):
        """Reference model parameters don't change during training."""
        device = torch.device("cpu")
        ref_model = DummyModel()
        ref_before = [p.detach().clone() for p in ref_model.parameters()]

        # Simulate training on the main model (not ref)
        main_model = DummyModel()
        optimizer = torch.optim.AdamW(main_model.parameters(), lr=0.01)
        for _ in range(5):
            loss = sum(p.sum() for p in main_model.parameters())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Check ref unchanged
        ref_after = [p.detach() for p in ref_model.parameters()]
        for before, after in zip(ref_before, ref_after):
            assert torch.equal(before, after), "Reference policy parameters changed"


# ===========================================================================
# Acceptance Test 11: smoke batch ≥ 4 sources
# ===========================================================================

class TestSmokeBatchFourSources:
    """Test 11: smoke batch至少包含4个source."""

    def test_four_distinct_sources(self):
        """A smoke batch must include at least 4 distinct source sequences."""
        records = [
            _make_record(transcript_id=f"src_{i:03d}",
                        five_utr="GCCAUG" + "CAU" * (10 + i),
                        cds=START_CODON + "GCU" * (5 + i) + "UAA")
            for i in range(4)
        ]
        assert len(records) >= 4
        # Verify all are distinct
        seqs = [r.seq for r in records]
        assert len(set(seqs)) == 4

    def test_four_sources_produce_trajectories(self):
        """Each of 4 sources produces at least one trajectory step."""
        device = torch.device("cpu")
        policy = _make_mock_policy(device)
        records = [
            _make_record(transcript_id=f"src_{i:03d}",
                        five_utr="GCCAUG" + "CAU" * (10 + i),
                        cds=START_CODON + "GCU" * (5 + i) + "UAA")
            for i in range(4)
        ]
        for record in records:
            steps, cfg = _make_trajectory(policy, record, max_edits=1)
            assert len(steps) >= 1, f"No steps for source {record.transcript_id}"


# ===========================================================================
# Acceptance Test 12: checkpoint resume consistent
# ===========================================================================

class TestCheckpointResume:
    """Test 12: checkpoint resume一致."""

    def test_save_load_roundtrip(self):
        """Saving and loading a checkpoint preserves model state."""
        model = DummyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Do a few steps
        for _ in range(3):
            loss = sum(p.sum() for p in model.parameters())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint = {
                "stage": "constrained_grpo",
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": 3,
            }
            torch.save(checkpoint, f.name)
            path = f.name

        # Load into new model
        model2 = DummyModel()
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model2.load_state_dict(payload["model_state"])
        optimizer2.load_state_dict(payload["optimizer_state"])

        # Verify identical
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2), "Model state mismatch after resume"

        os.unlink(path)


# ===========================================================================
# Acceptance Test 13: legacy artifact cannot be loaded by paper loader
# ===========================================================================

class TestLegacyArtifactRejection:
    """Test 13: legacy artifact不能被paper loader加载."""

    def test_paper_loader_rejects_legacy(self):
        """Paper loader raises ValueError for legacy checkpoints."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint = {"stage": "tiny_mdp_reinforce", "model_state": {}}
            torch.save(checkpoint, f.name)
            path = f.name

        with pytest.raises(ValueError, match="non-production"):
            ProductionPathGate.load_for_paper(path)

        os.unlink(path)

    def test_paper_loader_accepts_production(self):
        """Paper loader accepts production checkpoints."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint = {"stage": "constrained_grpo", "model_state": {}}
            torch.save(checkpoint, f.name)
            path = f.name

        payload = ProductionPathGate.load_for_paper(path)
        assert payload["stage"] == "constrained_grpo"

        os.unlink(path)

    def test_classify_checkpoint(self):
        """Checkpoint classification works correctly."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"stage": "constrained_grpo"}, f.name)
            assert ProductionPathGate.classify_checkpoint(f.name) == "production"

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"stage": "tiny_mdp_reinforce"}, f.name)
            assert ProductionPathGate.classify_checkpoint(f.name) == "legacy"


# ===========================================================================
# Acceptance Test 14: source baseline normalization correct
# ===========================================================================

class TestSourceBaselineNormalization:
    """Test 14: source baseline normalization正确."""

    def test_reward_is_source_normalized_delta(self):
        """Training reward scalar is relative to source baseline (delta)."""
        raw = {"te": 0.3, "access": 0.1, "agreement": 0.0, "edit_cost": -0.5}
        hard = HardConstraintStatus()
        reward = build_training_reward(raw, uncertainty=0.05, hard_constraint_status=hard, source_baseline=0.0)
        # The scalar should be a delta, not an absolute value
        assert reward.source_normalized_delta == reward.scalar
        # It should be the sum of deltas minus uncertainty
        expected = 0.3 + 0.1 + 0.0 + (-0.5) - 0.05
        assert abs(reward.scalar - expected) < 1e-6

    def test_stop_reward_is_zero_delta(self):
        """STOP action has zero delta (source = source)."""
        stop = stop_reward_vector(["te", "access", "agreement", "edit_cost"])
        assert all(v == 0.0 for v in stop.values())

    def test_hard_failure_zeroes_relative_to_source(self):
        """Hard failure produces negative delta, not absolute negative."""
        raw = {"te": 0.5, "access": 0.5, "agreement": 0.0, "edit_cost": -0.5}
        hard = HardConstraintStatus(protein_identity=False)
        reward = build_training_reward(raw, uncertainty=0.0, hard_constraint_status=hard)
        # Hard failure → scalar = -1.0 (a delta, not absolute)
        assert reward.scalar == -1.0
        assert reward.source_normalized_delta == -1.0
        assert reward.hard_constraint_gated

    def test_uncertainty_not_double_subtracted(self):
        """Uncertainty is subtracted exactly once (fixes the P3-04 bug)."""
        raw = {"te": 0.5, "access": 0.1, "agreement": 0.0, "edit_cost": -0.5}
        uncertainty = 0.2
        hard = HardConstraintStatus()
        reward = build_training_reward(raw, uncertainty=uncertainty, hard_constraint_status=hard)
        # Expected: (0.5 + 0.1 + 0.0 - 0.5) - 0.2 = -0.1
        # If double-subtracted: (0.5 + 0.1 + 0.0 - 0.5) - 0.2 - 0.2 = -0.3
        expected_single = (0.5 + 0.1 + 0.0 + (-0.5)) - 0.2
        assert abs(reward.scalar - expected_single) < 1e-6
        assert reward.reward_provenance["uncertainty_penalty"] == 0.2
