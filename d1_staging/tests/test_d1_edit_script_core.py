"""Unit tests for D1-01 edit_script_core.

D1-01 acceptance criterion: apply(edit_script, source) == candidate 100%
plus path ambiguity quantified.

Run: pytest d1_staging/tests/test_d1_edit_script_core.py -v
"""

import os
import sys
import random
import json

import pytest

# Make the d1 scripts dir importable
HERE = os.path.dirname(os.path.abspath(__file__))
D1_SCRIPTS = os.path.join(HERE, "..", "scripts", "d1")
sys.path.insert(0, D1_SCRIPTS)

from edit_script_core import (  # noqa: E402
    EditOp,
    compute_edit_script,
    apply_edit_script,
    count_optimal_alignments,
    edit_distance,
    summarize_edit_script,
    canonical_record,
    canonical_record_no_edit,
)


# ---------------------------------------------------------------------------
# EditOp dataclass
# ---------------------------------------------------------------------------


class TestEditOp:
    def test_construct_ins(self):
        op = EditOp("INS", 3, "A")
        assert op.op == "INS"
        assert op.pos == 3
        assert op.token == "A"

    def test_construct_del(self):
        op = EditOp("DEL", 2, "")
        assert op.op == "DEL"
        assert op.token == ""

    def test_frozen(self):
        op = EditOp("SUB", 0, "G")
        with pytest.raises(Exception):
            op.op = "INS"  # type: ignore

    def test_to_from_dict_roundtrip(self):
        op = EditOp("INS", 5, "T")
        d = op.to_dict()
        assert d == {"op": "INS", "pos": 5, "token": "T"}
        op2 = EditOp.from_dict(d)
        assert op == op2

    def test_from_dict_missing_token_defaults_empty(self):
        op = EditOp.from_dict({"op": "DEL", "pos": 1})
        assert op.token == ""


# ---------------------------------------------------------------------------
# compute_edit_script + apply_edit_script round-trip
# ---------------------------------------------------------------------------


class TestComputeEditScript:
    def test_identical_empty(self):
        assert compute_edit_script("", "") == []

    def test_identical_nonempty(self):
        src = "ACGTACGT"
        assert compute_edit_script(src, src) == []

    def test_empty_to_nonempty_all_insertions(self):
        ops = compute_edit_script("", "ACGT")
        assert apply_edit_script("", ops) == "ACGT"
        assert all(o.op == "INS" for o in ops)

    def test_nonempty_to_empty_all_deletions(self):
        ops = compute_edit_script("ACGT", "")
        assert apply_edit_script("ACGT", ops) == ""
        assert all(o.op == "DEL" for o in ops)
        assert len(ops) == 4

    def test_single_substitution(self):
        src = "ACGT"
        cand = "ACCT"
        ops = compute_edit_script(src, cand)
        assert apply_edit_script(src, ops) == cand
        assert len(ops) == 1
        assert ops[0].op == "SUB"
        assert ops[0].pos == 2
        assert ops[0].token == "C"

    def test_single_insertion_middle(self):
        src = "ACGT"
        cand = "ACGGT"
        ops = compute_edit_script(src, cand)
        assert apply_edit_script(src, ops) == cand
        assert len(ops) == 1
        assert ops[0].op == "INS"

    def test_single_deletion_middle(self):
        src = "ACGGT"
        cand = "ACGT"
        ops = compute_edit_script(src, cand)
        assert apply_edit_script(src, ops) == cand
        assert len(ops) == 1
        assert ops[0].op == "DEL"

    def test_mixed_edits(self):
        src = "ACGTACGT"
        cand = "AGGTACCT"
        ops = compute_edit_script(src, cand)
        assert apply_edit_script(src, ops) == cand

    def test_completely_different(self):
        src = "AAAA"
        cand = "TTTT"
        ops = compute_edit_script(src, cand)
        assert apply_edit_script(src, ops) == cand

    def test_length_change_large(self):
        src = "A" * 20
        cand = "C" * 5
        ops = compute_edit_script(src, cand)
        assert apply_edit_script(src, ops) == cand

    def test_traceback_preference_match_over_sub(self):
        # Identical char in middle should be a MATCH, not a SUB
        src = "ACG"
        cand = "TCG"
        ops = compute_edit_script(src, cand)
        assert apply_edit_script(src, ops) == cand
        # Should be exactly 1 SUB at pos 0
        assert len(ops) == 1
        assert ops[0].op == "SUB"
        assert ops[0].pos == 0


# ---------------------------------------------------------------------------
# Round-trip property tests (randomized) — this is the D1-01 acceptance core
# ---------------------------------------------------------------------------


class TestRoundTripProperty:
    """D1-01 acceptance: apply(compute(src, cand), src) == cand 100%."""

    def test_roundtrip_random_short(self):
        rng = random.Random(42)
        alphabet = "ACGT"
        for _ in range(500):
            n = rng.randint(0, 15)
            m = rng.randint(0, 15)
            src = "".join(rng.choice(alphabet) for _ in range(n))
            cand = "".join(rng.choice(alphabet) for _ in range(m))
            ops = compute_edit_script(src, cand)
            assert apply_edit_script(src, ops) == cand, (
                f"roundtrip failed: src={src!r} cand={cand!r} ops={ops}"
            )

    def test_roundtrip_random_longer(self):
        rng = random.Random(7)
        alphabet = "ACGT"
        for _ in range(200):
            n = rng.randint(0, 80)
            m = rng.randint(0, 80)
            src = "".join(rng.choice(alphabet) for _ in range(n))
            cand = "".join(rng.choice(alphabet) for _ in range(m))
            ops = compute_edit_script(src, cand)
            assert apply_edit_script(src, ops) == cand

    def test_roundtrip_small_edits(self):
        # Start from a base seq, apply small perturbations, verify recovery
        rng = random.Random(99)
        alphabet = "ACGT"
        for _ in range(300):
            n = rng.randint(5, 30)
            base = "".join(rng.choice(alphabet) for _ in range(n))
            cand = list(base)
            k = rng.randint(0, 5)
            for _ in range(k):
                if not cand:
                    cand.append(rng.choice(alphabet))
                    continue
                action = rng.choice(["sub", "ins", "del"])
                if action == "sub":
                    p = rng.randint(0, len(cand) - 1)
                    cand[p] = rng.choice(alphabet)
                elif action == "ins":
                    p = rng.randint(0, len(cand))
                    cand.insert(p, rng.choice(alphabet))
                else:
                    p = rng.randint(0, len(cand) - 1)
                    cand.pop(p)
            cand_str = "".join(cand)
            ops = compute_edit_script(base, cand_str)
            assert apply_edit_script(base, ops) == cand_str


# ---------------------------------------------------------------------------
# Minimality of edit distance
# ---------------------------------------------------------------------------


class TestMinimality:
    def test_edit_count_equals_edit_distance(self):
        cases = [
            ("", ""),
            ("A", ""),
            ("", "A"),
            ("ACGT", "ACGT"),
            ("ACGT", "TGCA"),
            ("ACGTACGT", "ACGGTACGT"),
            ("AAAA", "TTTT"),
        ]
        for src, cand in cases:
            ops = compute_edit_script(src, cand)
            d = edit_distance(src, cand)
            assert len(ops) == d, (
                f"len(ops)={len(ops)} != edit_distance={d} for "
                f"src={src!r} cand={cand!r}"
            )

    def test_minimality_random(self):
        rng = random.Random(2026)
        alphabet = "ACGT"
        for _ in range(200):
            src = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 30)))
            cand = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 30)))
            ops = compute_edit_script(src, cand)
            assert len(ops) == edit_distance(src, cand)


# ---------------------------------------------------------------------------
# count_optimal_alignments (path ambiguity)
# ---------------------------------------------------------------------------


class TestPathAmbiguity:
    def test_identical_one_path(self):
        assert count_optimal_alignments("ACGT", "ACGT") == 1

    def test_empty_to_empty_one(self):
        assert count_optimal_alignments("", "") == 1

    def test_all_inserts_one_path(self):
        # "" -> "ACGT": only one way (4 sequential insertions)
        assert count_optimal_alignments("", "ACGT") == 1

    def test_all_deletes_one_path(self):
        assert count_optimal_alignments("ACGT", "") == 1

    def test_repeated_char_ambiguity(self):
        # AAA -> AAAA: ambiguity because the extra A can be inserted at any
        # of 4 positions and all are minimal. count >= 1.
        c = count_optimal_alignments("AAA", "AAAA")
        assert c >= 1
        # Known closed form for single-char L+1 insert = L+1 paths
        assert c == 4

    def test_single_sub_one_path(self):
        assert count_optimal_alignments("ACGT", "ACCT") == 1

    def test_count_at_least_one_always(self):
        rng = random.Random(31)
        alphabet = "ACGT"
        for _ in range(200):
            src = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 15)))
            cand = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 15)))
            assert count_optimal_alignments(src, cand) >= 1


# ---------------------------------------------------------------------------
# apply_edit_script error handling
# ---------------------------------------------------------------------------


class TestApplyErrors:
    def test_del_out_of_range(self):
        with pytest.raises(IndexError):
            apply_edit_script("A", [EditOp("DEL", 5, "")])

    def test_sub_out_of_range(self):
        with pytest.raises(IndexError):
            apply_edit_script("A", [EditOp("SUB", 5, "T")])

    def test_unknown_op(self):
        with pytest.raises(ValueError):
            apply_edit_script("A", [EditOp("XYZ", 0, "")])

    def test_stop_terminates_early(self):
        # STOP should break the loop without error
        result = apply_edit_script("ACGT", [EditOp("STOP", 0, "")])
        assert result == "ACGT"


# ---------------------------------------------------------------------------
# summarize_edit_script
# ---------------------------------------------------------------------------


class TestSummarize:
    def test_counts(self):
        ops = [
            EditOp("INS", 0, "A"),
            EditOp("DEL", 1, ""),
            EditOp("SUB", 2, "G"),
            EditOp("INS", 3, "T"),
        ]
        s = summarize_edit_script(ops)
        assert s["n_ins"] == 2
        assert s["n_del"] == 1
        assert s["n_sub"] == 1
        assert s["n_total"] == 4
        assert len(s["ops_summary"]) == 4

    def test_empty(self):
        s = summarize_edit_script([])
        assert s["n_total"] == 0


# ---------------------------------------------------------------------------
# canonical_record
# ---------------------------------------------------------------------------


class TestCanonicalRecord:
    def test_basic_record(self):
        rec = canonical_record(
            record_id="r001",
            dataset="sample2019",
            accession="GSE114002",
            region="5'UTR",
            source="ACGT",
            candidate="ACCT",
            labels={"rl": 5.5},
            metadata={"family": "F1"},
        )
        assert rec["record_id"] == "r001"
        assert rec["dataset"] == "sample2019"
        assert rec["region"] == "5'UTR"
        assert rec["source_sequence"] == "ACGT"
        assert rec["candidate_sequence"] == "ACCT"
        # D1-01 acceptance: verified True
        assert rec["edit_script_verified"] is True
        assert rec["edit_distance"] == 1
        assert rec["n_sub"] == 1
        assert rec["path_ambiguity"] >= 1
        assert rec["labels"] == {"rl": 5.5}
        assert rec["metadata"] == {"family": "F1"}
        # edit_script is serializable list of dicts
        assert isinstance(rec["edit_script"], list)
        for op_dict in rec["edit_script"]:
            assert set(op_dict.keys()) == {"op", "pos", "token"}

    def test_no_edit_identical(self):
        rec = canonical_record(
            record_id="r002",
            dataset="x",
            accession="GSE1",
            region="3'UTR",
            source="ACGT",
            candidate="ACGT",
            labels={},
        )
        assert rec["edit_script_verified"] is True
        assert rec["edit_distance"] == 0
        assert rec["edit_script"] == []
        assert rec["path_ambiguity"] == 1

    def test_metadata_defaults_empty(self):
        rec = canonical_record(
            record_id="r003",
            dataset="x",
            accession="GSE1",
            region="5'UTR",
            source="A",
            candidate="T",
            labels={},
        )
        assert rec["metadata"] == {}

    def test_json_serializable(self):
        rec = canonical_record(
            record_id="r004",
            dataset="x",
            accession="GSE1",
            region="5'UTR",
            source="ACGTACGT",
            candidate="ACGTTCGT",
            labels={"rl": 2.0},
            metadata={"note": "test"},
        )
        s = json.dumps(rec)  # should not raise
        assert isinstance(s, str)
        rec2 = json.loads(s)
        assert rec2["record_id"] == "r004"

    def test_roundtrip_verified_random(self):
        rng = random.Random(555)
        alphabet = "ACGT"
        for _ in range(200):
            n = rng.randint(0, 25)
            m = rng.randint(0, 25)
            src = "".join(rng.choice(alphabet) for _ in range(n))
            cand = "".join(rng.choice(alphabet) for _ in range(m))
            rec = canonical_record(
                record_id="rnd",
                dataset="test",
                accession="GSE0",
                region="5'UTR",
                source=src,
                candidate=cand,
                labels={},
            )
            assert rec["edit_script_verified"] is True
            assert rec["edit_distance"] == edit_distance(src, cand)


class TestCanonicalRecordNoEdit:
    def test_observational_record(self):
        rec = canonical_record_no_edit(
            record_id="obs001",
            dataset="GSE207584",
            accession="GSE207584",
            region="CDS",
            sequence="ATGCATGC",
            labels={"count": 100},
            metadata={"source_file": "x.csv"},
        )
        assert rec["record_id"] == "obs001"
        assert rec["source_sequence"] is None
        assert rec["candidate_sequence"] == "ATGCATGC"
        assert rec["edit_script"] == []
        assert rec["edit_script_verified"] is True
        assert rec["edit_distance"] == 0
        assert rec["path_ambiguity"] == 1
        assert rec["metadata"]["record_type"] == "observational"
        assert rec["metadata"]["source_file"] == "x.csv"

    def test_json_serializable(self):
        rec = canonical_record_no_edit(
            record_id="obs002",
            dataset="x",
            accession="GSE1",
            region="3'UTR",
            sequence="ACGT",
            labels={},
        )
        json.dumps(rec)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
