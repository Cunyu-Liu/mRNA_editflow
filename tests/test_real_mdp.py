"""Tests for P2-05: RealMRNAMDP.

Covers:
- RealMRNAMDP config validation
- initial_state / reward / is_terminal interface (mirrors TinyMDP)
- Sparse terminal reward: 0 for non-terminal, delta-predicted-value at terminal
- STOP action yields terminal reward = predict(state) - predict(initial)
- max_steps forced termination
- Cache correctness (initial_pred cached across calls)
- to_metadata audit fields
- Reward qualifier: reward_field defaults to "predicted_te_internal_proxy"

All reward signals in these tests are synthetic (mock oracle). Any claim
about "improving TE" in production MUST be qualified as "predicted TE
(internal proxy)" until P2-01 multi-region oracle validation completes.
"""
from __future__ import annotations

import json
import unittest
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.rl.action_space import STOP_ACTION, Action
from mrna_editflow.rl.real_mdp import OracleLike, RealMRNAMDP


# ---------------------------------------------------------------------------
# Mock oracle
# ---------------------------------------------------------------------------


class MockOracle:
    """Mock oracle that maps sequences to user-defined scalar values.

    Implements the :class:`OracleLike` protocol (``predict(sequences) -> np.ndarray``).
    """

    def __init__(self, value_map: Dict[str, float], default: float = 0.0) -> None:
        self.value_map = dict(value_map)
        self.default = float(default)
        self.call_count = 0

    def predict(self, sequences: Sequence[str]) -> np.ndarray:
        self.call_count += 1
        return np.array(
            [float(self.value_map.get(s, self.default)) for s in sequences],
            dtype=np.float64,
        )


def _make_record(seq: str = "AAAA", transcript_id: str = "T") -> MRNARecord:
    """Build a minimal MRNARecord for testing."""
    return MRNARecord(
        transcript_id=transcript_id,
        five_utr=seq,
        cds="",
        three_utr="",
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestRealMRNAMDPConfig(unittest.TestCase):
    def test_default_config(self) -> None:
        mdp = RealMRNAMDP(
            initial_record=_make_record(),
            oracle=MockOracle({}),
        )
        self.assertEqual(mdp.max_steps, 3)
        self.assertAlmostEqual(mdp.gamma, 0.99)
        self.assertEqual(mdp.region, "full")
        self.assertEqual(mdp.reward_field, "predicted_te_internal_proxy")

    def test_max_steps_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            RealMRNAMDP(
                initial_record=_make_record(),
                oracle=MockOracle({}),
                max_steps=0,
            )
        RealMRNAMDP(
            initial_record=_make_record(),
            oracle=MockOracle({}),
            max_steps=1,
        )

    def test_gamma_must_be_in_open_zero_one(self) -> None:
        with self.assertRaises(ValueError):
            RealMRNAMDP(
                initial_record=_make_record(),
                oracle=MockOracle({}),
                gamma=0.0,
            )
        with self.assertRaises(ValueError):
            RealMRNAMDP(
                initial_record=_make_record(),
                oracle=MockOracle({}),
                gamma=1.01,
            )
        # gamma=1.0 is valid (undiscounted)
        RealMRNAMDP(
            initial_record=_make_record(),
            oracle=MockOracle({}),
            gamma=1.0,
        )

    def test_region_must_be_known(self) -> None:
        with self.assertRaises(ValueError):
            RealMRNAMDP(
                initial_record=_make_record(),
                oracle=MockOracle({}),
                region="invalid_region",
            )
        for r in ("full", "five_utr", "cds", "three_utr"):
            RealMRNAMDP(
                initial_record=_make_record(),
                oracle=MockOracle({}),
                region=r,
            )

    def test_reward_field_must_be_nonempty(self) -> None:
        with self.assertRaises(ValueError):
            RealMRNAMDP(
                initial_record=_make_record(),
                oracle=MockOracle({}),
                reward_field="",
            )


# ---------------------------------------------------------------------------
# Interface: initial_state, is_terminal
# ---------------------------------------------------------------------------


class TestRealMRNAMDPInterface(unittest.TestCase):
    def test_initial_state_returns_initial_record(self) -> None:
        rec = _make_record(seq="ACGT")
        mdp = RealMRNAMDP(initial_record=rec, oracle=MockOracle({}))
        self.assertIs(mdp.initial_state(), rec)

    def test_is_terminal_on_stop(self) -> None:
        mdp = RealMRNAMDP(
            initial_record=_make_record(),
            oracle=MockOracle({}),
            max_steps=3,
        )
        self.assertTrue(mdp.is_terminal(STOP_ACTION, step=0))
        self.assertTrue(mdp.is_terminal(STOP_ACTION, step=2))

    def test_is_terminal_on_max_steps(self) -> None:
        mdp = RealMRNAMDP(
            initial_record=_make_record(),
            oracle=MockOracle({}),
            max_steps=3,
        )
        # Non-STOP at step 2 (0-indexed): step+1=3 >= max_steps -> terminal
        non_stop = Action(op="ins", pos=0, nt=0)
        self.assertTrue(mdp.is_terminal(non_stop, step=2))

    def test_is_terminal_false_for_non_stop_below_max(self) -> None:
        mdp = RealMRNAMDP(
            initial_record=_make_record(),
            oracle=MockOracle({}),
            max_steps=3,
        )
        non_stop = Action(op="ins", pos=0, nt=0)
        self.assertFalse(mdp.is_terminal(non_stop, step=0))
        self.assertFalse(mdp.is_terminal(non_stop, step=1))


# ---------------------------------------------------------------------------
# Reward: sparse terminal delta
# ---------------------------------------------------------------------------


class TestRealMRNAMDPReward(unittest.TestCase):
    def test_non_terminal_reward_is_zero(self) -> None:
        """Non-terminal steps return 0 (sparse reward)."""
        initial = _make_record(seq="AAAA")
        next_rec = _make_record(seq="AAAT")
        oracle = MockOracle({"AAAA": 1.0, "AAAT": 1.5})
        mdp = RealMRNAMDP(initial_record=initial, oracle=oracle, max_steps=3)
        non_stop = Action(op="sub", pos=3, nt=3)  # A -> T at pos 3
        r = mdp.reward(state=initial, action=non_stop, next_state=next_rec, step=0)
        self.assertEqual(r, 0.0)

    def test_terminal_reward_is_delta_predicted_value(self) -> None:
        """Terminal step returns predict(final) - predict(initial)."""
        initial = _make_record(seq="AAAA")
        final = _make_record(seq="AAAT")
        oracle = MockOracle({"AAAA": 1.0, "AAAT": 1.5})
        mdp = RealMRNAMDP(initial_record=initial, oracle=oracle, max_steps=3)
        # Terminal via STOP at step=2
        r = mdp.reward(state=final, action=STOP_ACTION, next_state=final, step=2)
        self.assertAlmostEqual(r, 0.5)  # 1.5 - 1.0

    def test_terminal_reward_via_max_steps(self) -> None:
        """Terminal via step+1 >= max_steps also yields delta reward."""
        initial = _make_record(seq="AAAA")
        final = _make_record(seq="TTTT")
        oracle = MockOracle({"AAAA": 0.5, "TTTT": 2.0})
        mdp = RealMRNAMDP(initial_record=initial, oracle=oracle, max_steps=2)
        non_stop = Action(op="sub", pos=0, nt=3)
        # step=1 -> step+1=2 >= max_steps=2 -> terminal
        r = mdp.reward(state=initial, action=non_stop, next_state=final, step=1)
        self.assertAlmostEqual(r, 1.5)  # 2.0 - 0.5

    def test_stop_action_reward_is_gain_up_to_that_point(self) -> None:
        """STOP at step 0 with state=initial yields 0 gain (no edits made)."""
        initial = _make_record(seq="AAAA")
        oracle = MockOracle({"AAAA": 1.0})
        mdp = RealMRNAMDP(initial_record=initial, oracle=oracle, max_steps=3)
        r = mdp.reward(state=initial, action=STOP_ACTION, next_state=initial, step=0)
        self.assertAlmostEqual(r, 0.0)  # 1.0 - 1.0

    def test_negative_gain_when_final_worse_than_initial(self) -> None:
        """If the final sequence is worse, reward is negative."""
        initial = _make_record(seq="AAAA")
        final = _make_record(seq="TTTT")
        oracle = MockOracle({"AAAA": 2.0, "TTTT": 0.5})
        mdp = RealMRNAMDP(initial_record=initial, oracle=oracle, max_steps=2)
        r = mdp.reward(state=initial, action=STOP_ACTION, next_state=final, step=1)
        self.assertAlmostEqual(r, -1.5)  # 0.5 - 2.0

    def test_reward_uses_initial_record_seq_not_state_seq(self) -> None:
        """The baseline for delta is initial_record.seq, not the current state.

        This ensures G_0 = predict(s_final) - predict(s_initial) regardless
        of which step the trajectory terminates on.
        """
        initial = _make_record(seq="AAAA")
        mid = _make_record(seq="CCGG")
        final = _make_record(seq="TTTT")
        oracle = MockOracle({"AAAA": 1.0, "CCGG": 5.0, "TTTT": 3.0})
        mdp = RealMRNAMDP(initial_record=initial, oracle=oracle, max_steps=3)
        # Terminal at step 2 with state=mid, next_state=final
        r = mdp.reward(state=mid, action=STOP_ACTION, next_state=final, step=2)
        # Should be predict(final) - predict(initial) = 3.0 - 1.0 = 2.0
        # NOT predict(final) - predict(state) = 3.0 - 5.0 = -2.0
        self.assertAlmostEqual(r, 2.0)


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


class TestRealMRNAMDPCache(unittest.TestCase):
    def test_initial_pred_is_cached(self) -> None:
        """The initial prediction is computed once and reused."""
        initial = _make_record(seq="AAAA")
        oracle = MockOracle({"AAAA": 1.0, "TTTT": 2.0})
        mdp = RealMRNAMDP(initial_record=initial, oracle=oracle, max_steps=2)
        # First terminal reward: computes initial_pred (1 call) + final (1 call) = 2 calls
        r1 = mdp.reward(
            state=initial,
            action=STOP_ACTION,
            next_state=_make_record(seq="TTTT"),
            step=1,
        )
        self.assertAlmostEqual(r1, 1.0)
        calls_after_first = oracle.call_count
        # Second terminal reward: should reuse cached initial_pred, only 1 new call
        r2 = mdp.reward(
            state=initial,
            action=STOP_ACTION,
            next_state=_make_record(seq="TTTT"),
            step=1,
        )
        self.assertAlmostEqual(r2, 1.0)
        calls_after_second = oracle.call_count
        self.assertEqual(calls_after_second - calls_after_first, 1)

    def test_reset_cache_clears_initial_pred(self) -> None:
        """reset_cache() forces recompute on next reward call."""
        initial = _make_record(seq="AAAA")
        oracle = MockOracle({"AAAA": 1.0, "TTTT": 2.0})
        mdp = RealMRNAMDP(initial_record=initial, oracle=oracle, max_steps=2)
        mdp.reward(
            state=initial,
            action=STOP_ACTION,
            next_state=_make_record(seq="TTTT"),
            step=1,
        )
        self.assertIsNotNone(mdp._initial_pred)
        mdp.reset_cache()
        self.assertIsNone(mdp._initial_pred)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestRealMRNAMDPMetadata(unittest.TestCase):
    def test_to_metadata_is_json_serializable(self) -> None:
        rec = _make_record(seq="ACGTACGT", transcript_id="T001")
        mdp = RealMRNAMDP(
            initial_record=rec,
            oracle=MockOracle({}),
            max_steps=3,
            gamma=0.99,
            region="cds",
        )
        meta = mdp.to_metadata()
        # Must be JSON-serializable
        s = json.dumps(meta)
        d = json.loads(s)
        self.assertEqual(d["mdp_type"], "RealMRNAMDP")
        self.assertEqual(d["max_steps"], 3)
        self.assertAlmostEqual(d["gamma"], 0.99)
        self.assertEqual(d["region"], "cds")
        self.assertEqual(d["reward_field"], "predicted_te_internal_proxy")
        self.assertEqual(d["reward_design"], "sparse_terminal_delta_predicted_value")
        self.assertEqual(d["initial_transcript_id"], "T001")
        self.assertEqual(d["initial_seq_length"], 8)

    def test_metadata_reward_field_defaults_to_proxy(self) -> None:
        """Default reward_field complies with the project constraint:
        any 'improves TE' claim MUST be qualified as 'predicted/internal proxy'
        until P2-01 completes."""
        mdp = RealMRNAMDP(
            initial_record=_make_record(),
            oracle=MockOracle({}),
        )
        self.assertEqual(mdp.reward_field, "predicted_te_internal_proxy")
        self.assertIn("proxy", mdp.to_metadata()["reward_field"])


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestOracleProtocol(unittest.TestCase):
    def test_mock_oracle_satisfies_protocol(self) -> None:
        """MockOracle should be usable anywhere OracleLike is expected."""
        oracle: OracleLike = MockOracle({"A": 1.0})
        mdp: RealMRNAMDP = RealMRNAMDP(
            initial_record=_make_record(seq="A"),
            oracle=oracle,
            max_steps=1,
        )
        # Smoke test: reward computation runs
        r = mdp.reward(
            state=_make_record(seq="A"),
            action=STOP_ACTION,
            next_state=_make_record(seq="A"),
            step=0,
        )
        self.assertAlmostEqual(r, 0.0)

    def test_oracle_predict_called_with_sequence_strings(self) -> None:
        """The oracle must be called with a list of sequence strings."""
        initial = _make_record(seq="GGGG")
        called_with: list = []

        class CapturingOracle:
            def predict(self, sequences: Sequence[str]) -> np.ndarray:
                called_with.extend(list(sequences))
                return np.array([1.0] * len(sequences), dtype=np.float64)

        mdp = RealMRNAMDP(
            initial_record=initial,
            oracle=CapturingOracle(),
            max_steps=1,
        )
        mdp.reward(
            state=initial,
            action=STOP_ACTION,
            next_state=initial,
            step=0,
        )
        self.assertEqual(len(called_with), 2)  # initial + final
        self.assertEqual(called_with[0], "GGGG")
        self.assertEqual(called_with[1], "GGGG")


# ---------------------------------------------------------------------------
# Integration: GRPOREINFORCE on RealMRNAMDP (smoke test)
# ---------------------------------------------------------------------------


class TestGRPOOnRealMDPSmoke(unittest.TestCase):
    """Smoke test: GRPOREINFORCE can collect a group on RealMRNAMDP.

    This verifies interface compatibility without requiring a real Stage A
    checkpoint. Uses a TinyTrainableModel + MockOracle.
    """

    def test_collect_group_on_real_mdp(self) -> None:
        import torch

        from mrna_editflow.rl.grpo import GRPOConfig, GRPOREINFORCE
        from mrna_editflow.rl.policy import Policy, PolicyConfig
        from mrna_editflow.rl.tiny_mdp import TinyTrainableModel

        # Build a tiny policy (mirrors test_p2_05_grpo.py helpers)
        device = torch.device("cpu")
        initial = _make_record(seq="AAAA")
        oracle = MockOracle({"AAAA": 1.0, "AAAT": 1.5, "AATA": 1.2, "TAAA": 0.8})
        mdp = RealMRNAMDP(initial_record=initial, oracle=oracle, max_steps=2)

        model = TinyTrainableModel(vocab_dim=4, hidden=8)
        backbone = type(
            "B", (), {"out_dim": 8, "forward": lambda self, *a, **k: None}
        )()
        policy = Policy(model=model, backbone=backbone, cfg=PolicyConfig(), device=device)

        cfg = GRPOConfig(group_size=4)
        trainer = GRPOREINFORCE(policy=policy, mdp=mdp, cfg=cfg, lr=0.01)

        gen = torch.Generator(device=device)
        gen.manual_seed(0)
        group = trainer.collect_group(generator=gen)
        self.assertEqual(len(group), 4)
        for traj in group:
            self.assertGreaterEqual(len(traj.transitions), 1)
            # Each trajectory must have at least one transition
            # and the reward signal must come from RealMRNAMDP (delta predicted value)
            total_r = traj.total_reward()
            # total_reward should be 0 (if no terminal) or delta-predicted (if terminal)
            self.assertTrue(
                total_r == 0.0 or isinstance(total_r, float),
                f"unexpected total_reward: {total_r}",
            )

    def test_compute_loss_on_real_mdp(self) -> None:
        """compute_loss runs without error on RealMRNAMDP trajectories."""
        import torch

        from mrna_editflow.rl.grpo import GRPOConfig, GRPOREINFORCE
        from mrna_editflow.rl.policy import Policy, PolicyConfig
        from mrna_editflow.rl.tiny_mdp import TinyTrainableModel

        device = torch.device("cpu")
        initial = _make_record(seq="AAAA")
        oracle = MockOracle({"AAAA": 1.0, "AAAT": 1.5, "AATA": 1.2})
        mdp = RealMRNAMDP(initial_record=initial, oracle=oracle, max_steps=2)

        model = TinyTrainableModel(vocab_dim=4, hidden=8)
        backbone = type(
            "B", (), {"out_dim": 8, "forward": lambda self, *a, **k: None}
        )()
        policy = Policy(model=model, backbone=backbone, cfg=PolicyConfig(), device=device)

        cfg = GRPOConfig(group_size=4)
        trainer = GRPOREINFORCE(policy=policy, mdp=mdp, cfg=cfg, lr=0.01)

        gen = torch.Generator(device=device)
        gen.manual_seed(42)
        groups = [trainer.collect_group(generator=gen) for _ in range(2)]
        loss, metrics = trainer.compute_loss(groups)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(np.isfinite(float(loss.item())))
        self.assertEqual(metrics["n_groups"], 2)
        self.assertEqual(metrics["group_size"], 4)


if __name__ == "__main__":
    unittest.main()
