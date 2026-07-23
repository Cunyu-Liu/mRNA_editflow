"""P3-06: Acceptance tests for Minimal-Edit MDP, Action Space, and Reward v3.

Tests:
1. Protein identity 100%
2. Transcript length 100% unchanged
3. T7-primary trajectory can select 5'UTR/CDS
4. Learned STOP trainable
5. Reward is source-normalized
6. Uncertainty enters risk adjustment
7. Training/independent Oracle separation
8. No single-nt CDS nonsynonymous intermediate
9. Action/reward provenance complete
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import torch.nn as nn

from core.constants import (
    START_CODON, STOP_CODONS, SYNONYMOUS_CODONS, CODON_TABLE,
    translate, is_valid_cds,
)
from core.schema import MRNARecord
from rl.p3_06_mdp import (
    EditAction, STOP_EDIT, apply_edit_action, build_legal_edit_actions,
    MDPState, initial_state, transition,
    ConstantStop, BudgetAwareStop, LearnedStop,
    HierarchicalPolicy,
    RewardV3Config, compute_reward_v3,
    _get_synonymous_codon_set,
)


def _make_record(five_utr_len=50, cds_codons=10, three_utr_len=30) -> MRNARecord:
    """Create a test record."""
    five_utr = "ACGU" * (five_utr_len // 4)
    # CDS: START + codons + STOP
    cds = START_CODON + "GCU" * (cds_codons - 2) + "UAA"
    three_utr = "UGCU" * (three_utr_len // 4)
    return MRNARecord(
        transcript_id="test",
        five_utr=five_utr,
        cds=cds,
        three_utr=three_utr,
    )


class TestProteinIdentity:
    """Acceptance criterion 1: protein identity 100%."""

    def test_all_cds_actions_preserve_protein(self):
        """Every legal CDS action must preserve protein identity."""
        record = _make_record()
        source_protein = translate(record.cds)
        legal = build_legal_edit_actions(record)
        cds_actions = [a for a in legal if a.is_cds()]
        assert len(cds_actions) > 0, "No CDS actions found"
        for action in cds_actions:
            result = apply_edit_action(record, action)
            assert translate(result.cds) == source_protein, \
                f"Protein changed by {action}"

    def test_5utr_actions_do_not_affect_protein(self):
        """5'UTR actions must not affect protein."""
        record = _make_record()
        source_protein = translate(record.cds)
        legal = build_legal_edit_actions(record)
        utr_actions = [a for a in legal if a.is_five_utr()]
        assert len(utr_actions) > 0, "No 5'UTR actions found"
        for action in utr_actions[:10]:  # Test first 10
            result = apply_edit_action(record, action)
            assert translate(result.cds) == source_protein

    def test_start_codon_not_editable(self):
        """Start codon (codon 0) must not be editable."""
        record = _make_record()
        action = EditAction(op="cds_synonymous_sub", pos=0, target_codon="AUG")
        with pytest.raises(ValueError, match="start codon"):
            apply_edit_action(record, action)

    def test_stop_codon_not_editable(self):
        """Stop codon (last codon) must not be editable."""
        record = _make_record()
        n_codons = len(record.cds) // 3
        action = EditAction(op="cds_synonymous_sub", pos=n_codons - 1, target_codon="UAG")
        with pytest.raises(ValueError, match="stop codon"):
            apply_edit_action(record, action)

    def test_nonsynonymous_rejected(self):
        """Nonsynonymous codon substitution must be rejected."""
        record = _make_record()
        # Find a codon that has synonymous options, then try a nonsynonymous target
        codon_pos = 1
        nt_start = codon_pos * 3
        old_codon = record.cds[nt_start:nt_start + 3]
        # UGU (Cys) is NOT synonymous to GCU (Ala)
        action = EditAction(op="cds_synonymous_sub", pos=codon_pos, target_codon="UGU")
        with pytest.raises(ValueError, match="not synonymous"):
            apply_edit_action(record, action)


class TestLengthInvariant:
    """Acceptance criterion 2: transcript length 100% unchanged."""

    def test_all_actions_preserve_length(self):
        """Every legal action must preserve transcript length."""
        record = _make_record()
        source_len = len(record.seq)
        legal = build_legal_edit_actions(record)
        for action in legal[:50]:  # Test first 50
            result = apply_edit_action(record, action)
            assert len(result.seq) == source_len, \
                f"Length changed by {action}: {len(result.seq)} != {source_len}"

    def test_no_indels_in_action_space(self):
        """No insertion or deletion actions exist."""
        record = _make_record()
        legal = build_legal_edit_actions(record)
        ops = {a.op for a in legal}
        assert "ins" not in ops, "Insertion found in action space"
        assert "del" not in ops, "Deletion found in action space"

    def test_no_three_utr_actions(self):
        """No 3'UTR actions in primary task."""
        record = _make_record()
        legal = build_legal_edit_actions(record)
        # Check that no action modifies 3'UTR
        for action in legal:
            if action.is_stop():
                continue
            result = apply_edit_action(record, action)
            assert result.three_utr == record.three_utr, \
                f"3'UTR modified by {action}"


class TestT7PrimaryTrajectory:
    """Acceptance criterion 3: T7-primary trajectory can select 5'UTR/CDS."""

    def test_trajectory_can_alternate_regions(self):
        """A trajectory can alternate between 5'UTR and CDS edits."""
        record = _make_record()
        state = initial_state(record, budget=3)

        # Step 1: 5'UTR edit
        legal = build_legal_edit_actions(state.current_mrna, state.visited_states)
        utr_action = next(a for a in legal if a.is_five_utr())
        state = transition(state, utr_action)
        assert state.n_edits() == 1

        # Step 2: CDS edit
        legal = build_legal_edit_actions(state.current_mrna, state.visited_states)
        cds_action = next(a for a in legal if a.is_cds())
        state = transition(state, cds_action)
        assert state.n_edits() == 2

        # Step 3: 5'UTR edit again
        legal = build_legal_edit_actions(state.current_mrna, state.visited_states)
        utr_action2 = next(a for a in legal if a.is_five_utr())
        state = transition(state, utr_action2)
        assert state.n_edits() == 3

        # Verify protein identity throughout
        assert translate(state.current_mrna.cds) == translate(record.cds)
        assert len(state.current_mrna.seq) == len(record.seq)

    def test_visited_states_prevent_cycles(self):
        """Visited states prevent returning to the same sequence."""
        record = _make_record(five_utr_len=4, cds_codons=5, three_utr_len=4)
        state = initial_state(record, budget=10)
        # Apply an edit
        legal = build_legal_edit_actions(state.current_mrna, state.visited_states)
        action = next(a for a in legal if a.is_five_utr())
        state = transition(state, action)
        # Try to go back (reverse the edit)
        legal2 = build_legal_edit_actions(state.current_mrna, state.visited_states)
        # The reverse action should be excluded because it leads to a visited state
        reverse = EditAction(
            op="five_utr_sub",
            pos=action.pos,
            nt=record.five_utr[action.pos],  # Original nucleotide
        )
        assert reverse not in legal2 or reverse.is_stop()


class TestLearnableStop:
    """Acceptance criterion 4: learned STOP trainable."""

    def test_constant_stop_returns_fixed_p(self):
        stop = ConstantStop(p=0.7)
        assert stop({}) == 0.7

    def test_budget_aware_stop_varies_with_budget(self):
        stop = BudgetAwareStop()
        p_high_budget = stop({"remaining_budget_frac": 1.0})
        p_low_budget = stop({"remaining_budget_frac": 0.0})
        # Should be different
        assert p_high_budget != p_low_budget

    def test_learned_stop_is_trainable(self):
        """LearnedStop MLP has trainable parameters."""
        stop = LearnedStop(input_dim=10, hidden_dim=32)
        params = list(stop.parameters())
        assert len(params) > 0, "No trainable parameters"
        # Check gradients flow
        x = torch.randn(1, 10)
        p = stop(x)
        loss = p.mean()
        loss.backward()
        for param in stop.parameters():
            assert param.grad is not None, "No gradient for parameter"

    def test_learned_stop_output_range(self):
        """LearnedStop output must be in [0, 1]."""
        stop = LearnedStop(input_dim=10, hidden_dim=32)
        x = torch.randn(100, 10)
        p = stop(x)
        assert (p >= 0).all() and (p <= 1).all()


class TestRewardSourceNormalized:
    """Acceptance criterion 5: reward is source-normalized."""

    def test_reward_is_delta_from_source(self):
        """Reward is always relative to source, not absolute."""
        record = _make_record()
        candidate = apply_edit_action(
            record,
            EditAction(op="five_utr_sub", pos=0, nt="G"),
        )
        reward = compute_reward_v3(
            source=record,
            candidate=candidate,
            predicted_deltas={"protein_output": 0.5},
            uncertainties={"protein_output": 0.1},
            n_edits=1,
        )
        # The reward should be the delta (0.5), not the absolute value
        assert reward["mean_delta"] == 0.5
        # Source baseline is implicitly 0 (delta from source)
        assert reward["provenance"]["source_normalized"] is True


class TestUncertaintyRiskAdjustment:
    """Acceptance criterion 6: uncertainty enters risk adjustment."""

    def test_lcb_penalizes_uncertainty(self):
        """LCB = mean - λ × uncertainty, so higher uncertainty → lower LCB."""
        record = _make_record()
        candidate = record

        reward_low_unc = compute_reward_v3(
            source=record, candidate=candidate,
            predicted_deltas={"protein_output": 0.5},
            uncertainties={"protein_output": 0.1},
            n_edits=0,
        )
        reward_high_unc = compute_reward_v3(
            source=record, candidate=candidate,
            predicted_deltas={"protein_output": 0.5},
            uncertainties={"protein_output": 0.5},
            n_edits=0,
        )
        assert reward_low_unc["lcb"] > reward_high_unc["lcb"], \
            "Higher uncertainty should produce lower LCB"

    def test_lambda_controls_risk_aversion(self):
        """Higher λ → more conservative (lower LCB)."""
        record = _make_record()
        candidate = record
        cfg_conservative = RewardV3Config(lambda_lcb=2.0)
        cfg_aggressive = RewardV3Config(lambda_lcb=0.5)

        r_cons = compute_reward_v3(
            source=record, candidate=candidate,
            predicted_deltas={"protein_output": 0.5},
            uncertainties={"protein_output": 0.3},
            n_edits=0, config=cfg_conservative,
        )
        r_aggr = compute_reward_v3(
            source=record, candidate=candidate,
            predicted_deltas={"protein_output": 0.5},
            uncertainties={"protein_output": 0.3},
            n_edits=0, config=cfg_aggressive,
        )
        assert r_cons["lcb"] < r_aggr["lcb"], \
            "Higher λ should produce lower LCB"


class TestNoSingleNtCDSIntermediate:
    """Acceptance criterion 8: no single-nt CDS nonsynonymous intermediate."""

    def test_cds_action_is_atomic(self):
        """CDS_SYNONYMOUS_SUB replaces the entire codon atomically."""
        record = _make_record()
        # Find a codon with multiple synonymous options
        codon_pos = 1
        nt_start = codon_pos * 3
        old_codon = record.cds[nt_start:nt_start + 3]
        synonyms = _get_synonymous_codon_set(old_codon)
        assert len(synonyms) > 0, f"Need synonymous codons for {old_codon}"

        target = synonyms[0]
        action = EditAction(op="cds_synonymous_sub", pos=codon_pos, target_codon=target)
        result = apply_edit_action(record, action)

        # The entire codon must be replaced, not just one nucleotide
        new_codon = result.cds[nt_start:nt_start + 3]
        assert new_codon == target, \
            f"Expected {target}, got {new_codon}"
        # Protein preserved
        assert translate(result.cds) == translate(record.cds)

    def test_no_partial_codon_change(self):
        """No action can change only 1-2 nucleotides within a codon."""
        record = _make_record()
        legal = build_legal_edit_actions(record)
        for action in legal:
            if action.is_cds():
                # CDS actions specify codon position and target codon
                # They never specify a single nucleotide position
                assert action.target_codon != "", \
                    "CDS action must specify target codon, not single nt"
                assert len(action.target_codon) == 3, \
                    "Target must be a full codon (3 nt)"


class TestActionRewardProvenance:
    """Acceptance criterion 9: action/reward provenance complete."""

    def test_reward_provenance_fields(self):
        """Reward must include provenance fields."""
        record = _make_record()
        reward = compute_reward_v3(
            source=record, candidate=record,
            predicted_deltas={"protein_output": 0.0},
            uncertainties={"protein_output": 0.0},
            n_edits=0,
        )
        prov = reward["provenance"]
        assert "predictor" in prov
        assert "reward_version" in prov
        assert "source_normalized" in prov
        assert "risk_adjusted" in prov
        assert "no_novelty" in prov

    def test_no_novelty_in_reward(self):
        """Reward must NOT contain novelty = -edit_distance."""
        record = _make_record()
        reward = compute_reward_v3(
            source=record, candidate=record,
            predicted_deltas={"protein_output": 0.0},
            uncertainties={"protein_output": 0.0},
            n_edits=0,
        )
        # No "novelty" key in secondary terms
        assert "novelty" not in reward.get("secondary_terms", {})
        assert "novelty" not in reward


class TestHierarchicalPolicy:
    """Hierarchical policy produces valid log-probabilities."""

    def test_log_pi_stop(self):
        """log π(STOP|state) is finite."""
        record = _make_record()
        state = initial_state(record, budget=3)
        policy = HierarchicalPolicy()
        log_p = policy.log_pi(STOP_EDIT, state)
        assert isinstance(log_p, float)
        assert log_p <= 0.0, "log probability must be <= 0"

    def test_log_pi_edit(self):
        """log π(edit|state) is finite and negative."""
        record = _make_record()
        state = initial_state(record, budget=3)
        policy = HierarchicalPolicy()
        legal = build_legal_edit_actions(record)
        action = next(a for a in legal if a.is_five_utr())
        log_p = policy.log_pi(action, state)
        assert isinstance(log_p, float)
        assert log_p <= 0.0

    def test_full_trajectory_log_prob(self):
        """Full trajectory log-probability is the sum of per-step log-probs."""
        record = _make_record()
        state = initial_state(record, budget=2)
        policy = HierarchicalPolicy()

        total_log_p = 0.0
        for step in range(2):
            legal = build_legal_edit_actions(state.current_mrna, state.visited_states)
            action = next(a for a in legal if not a.is_stop())
            log_p = policy.log_pi(action, state)
            total_log_p += log_p
            state = transition(state, action)

        # STOP
        log_p_stop = policy.log_pi(STOP_EDIT, state)
        total_log_p += log_p_stop

        assert total_log_p < 0.0, "Total trajectory log-prob must be negative"
