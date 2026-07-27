"""P0-03: cross-role sequence collision tests.

Hard rules:
* exact candidate overlap between train and test = 0
* near-duplicate rate (Hamming <= threshold) below pre-registered cap
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.p3_02_delta_oracle import DeltaRecord
from scripts.audit_training_membership import (
    audit_exact_sequence_collision,
    audit_edit_neighborhood_collision,
    _hamming,
)


def make_rec(record_id: str, cand: str, role: str) -> DeltaRecord:
    return DeltaRecord(
        record_id=record_id,
        source_id=f"src_{record_id}",
        source_sequence="A" * len(cand),
        candidate_sequence=cand,
        edit_list=[],
        edit_count=1,
        edited_region="five_utr",
        delta=0.1,
        source_value=1.0,
        candidate_value=1.1,
        value_std=0.1,
        confidence="measured",
        split_role=role,
        family_cluster_id=f"fam_{record_id}",
        edit_type="measured_single",
    )


class TestExactSequenceCollision:
    def test_no_collision_passes(self):
        train = [make_rec("t1", "ACGUACGU", "train")]
        test = [make_rec("s1", "UGCAUGCA", "test")]
        report = audit_exact_sequence_collision(train, test)
        assert report["pass"] is True
        assert report["n_collisions"] == 0

    def test_exact_collision_fails(self):
        train = [make_rec("t1", "ACGUACGU", "train")]
        test = [make_rec("s1", "ACGUACGU", "test")]
        report = audit_exact_sequence_collision(train, test)
        assert report["pass"] is False
        assert report["n_collisions"] == 1

    def test_collision_examples_capped(self):
        train = [make_rec(f"t{i}", f"ACGU{i:04d}"[:8], "train") for i in range(30)]
        test = [make_rec(f"s{i}", f"ACGU{i:04d}"[:8], "test") for i in range(30)]
        report = audit_exact_sequence_collision(train, test)
        assert report["pass"] is False
        assert len(report["collision_examples"]) <= 20


class TestEditNeighborhoodCollision:
    def test_hamming_helper(self):
        assert _hamming("ACGU", "ACGU") == 0
        assert _hamming("ACGU", "ACGA") == 1
        assert _hamming("ACGU", "UGCA") == 4
        assert _hamming("ACGU", "ACGUA") == 5  # different lengths -> max len

    def test_distant_sequences_pass(self):
        train = [make_rec("t1", "AAAAAAAA", "train")]
        test = [make_rec("s1", "UUUUUUUU", "test"),
                make_rec("s2", "CCCCCCCC", "test")]
        report = audit_edit_neighborhood_collision(train, test, threshold=1)
        assert report["pass"] is True
        assert report["n_near_duplicate"] == 0

    def test_one_off_detected(self):
        train = [make_rec("t1", "ACGUACGU", "train")]
        test = [make_rec("s1", "ACGUACGA", "test")]  # Hamming 1
        report = audit_edit_neighborhood_collision(
            train, test, threshold=1, max_rate=0.0)
        assert report["n_near_duplicate"] == 1
        assert report["pass"] is False  # max_rate=0.0 -> any near-dup fails

    def test_rate_below_threshold_passes(self):
        train = [make_rec("t1", "ACGUACGU", "train")]
        test = [make_rec(f"s{i}", "UUUUUUUU", "test") for i in range(99)]
        test.append(make_rec("s99", "ACGUACGA", "test"))  # 1 near-dup / 100
        report = audit_edit_neighborhood_collision(
            train, test, threshold=1, max_rate=0.01)
        assert report["rate"] <= 0.01
        assert report["pass"] is True

    def test_different_lengths_not_compared(self):
        train = [make_rec("t1", "ACGU", "train")]
        test = [make_rec("s1", "ACGUACGU", "test")]
        report = audit_edit_neighborhood_collision(train, test, threshold=1)
        assert report["n_near_duplicate"] == 0

    def test_fast_path_matches_brute_force(self):
        """Pigeonhole fast path (threshold=1) must give verdicts identical to
        an independent brute-force all-pairs scan on randomized pools."""
        import random

        rng = random.Random(20260726)
        alphabet = "ACGU"

        def rand_seq(length: int) -> str:
            return "".join(rng.choice(alphabet) for _ in range(length))

        for _trial in range(5):
            length = rng.choice([7, 8, 9, 12])
            # Plant some near-duplicates so both branches are exercised.
            base = [rand_seq(length) for _ in range(60)]
            train_seqs = base + [
                s[:3] + rng.choice(alphabet) + s[4:] for s in base[:10]
            ]
            test_seqs = [rand_seq(length) for _ in range(40)] + base[:5]
            train = [make_rec(f"t{i}", s, "train") for i, s in enumerate(train_seqs)]
            test = [make_rec(f"s{i}", s, "test") for i, s in enumerate(test_seqs)]

            report = audit_edit_neighborhood_collision(train, test, threshold=1)

            brute_near = 0
            train_set = train_seqs
            for rec in test:
                if any(
                    len(rec.candidate_sequence) == len(other)
                    and sum(1 for x, y in zip(rec.candidate_sequence, other) if x != y) <= 1
                    for other in train_set
                ):
                    brute_near += 1
            assert report["n_near_duplicate"] == brute_near
            assert report["rate"] == brute_near / len(test)
