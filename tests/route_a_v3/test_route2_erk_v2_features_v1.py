#!/usr/bin/env python3
"""Unit test: ERK v2 feature construction is the E7-a v2 probe form (structure-anchored).

Verifies block/context indexing semantics replaced the probe ctx_feats exactly:
  block = pos // 8, ctx = index(left)*16 + index(right)*4 + index(alt),
  left/right from ORIGINAL source sequence, N-flank handling identical, counts accumulate.
"""
import sys
sys.path.insert(0, "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_w0_diagnosis_20260902")

import numpy as np

from scripts.route_a_v3.run_route2_erk_v2_train_v1 import build_features

BASES = "ACGT"


def mk_row(src, edits):
    return {
        "source_sequence": src,
        "source_relative_edits": [{"position": p, "candidate_base": b} for p, b in edits],
        "candidate_sequence": "X",
        "direction_normalized_delta": 0.0,
        "source_id": "S0",
        "task_id": "MEAN_RIBOSOME_LOAD::region=0",
    }


def test_single_edit_maps_to_block_ctx():
    # src ACGTACGT (8nt): edit pos3 T->C. left=s[2]=G, right=s[4]=A, block=0
    # ci = G(2)*16 + A(0)*4 + C(1) = 33
    rows = [mk_row("ACGTACGT", [(3, "C")])]
    X, _ = build_features(rows, n_blocks=1)
    flat = X[0]
    assert flat[33] == 1.0, flat[33]
    assert flat.sum() == 1.0


def test_block_indexing():
    # src 20nt; edit at pos 9 -> block 1 (pos//8)
    src = "ACGTACGTAC" + "GTACGTACGT"
    rows = [mk_row(src, [(9, "T")])]  # s[9]='G'->'T'; left=s[8]='A'(0), right=s[10]='G'(2)
    X, _ = build_features(rows, n_blocks=3)
    ci = 0 * 16 + 2 * 4 + BASES.index("T")  # A*16 + G*4 + T = 0+8+3 = 11
    assert X[0, 1 * 64 + ci] == 1.0
    assert X[0, :64].sum() == 0.0


def test_multi_edit_accumulates_counts():
    src = "ACGTACGT" + "ACGTACGT"
    # two edits at pos 0 and pos 1, same left/right/alt combos independently mapped
    rows = [mk_row(src, [(0, "G"), (1, "A")])]
    X, _ = build_features(rows, n_blocks=2)
    assert X[0].sum() == 2.0


def test_N_flank_adoption():
    # single-nt source: pos0 A->C; left/right both N -> skip (zero features)
    rows = [mk_row("A", [(0, "C")])]
    X, skip = build_features(rows, n_blocks=1)
    assert X[0].sum() == 0.0
    assert skip == 1  # row counted as zero-feature (mirrors probe skip semantics)


def test_alt_out_of_vocab_ignored():
    src = "ACGTACGT"
    rows = [mk_row(src, [(0, "X")])]
    X, skip = build_features(rows, n_blocks=1)
    assert X[0].sum() == 0.0 and skip == 1


if __name__ == "__main__":
    test_single_edit_maps_to_block_ctx()
    test_block_indexing()
    test_multi_edit_accumulates_counts()
    test_N_flank_adoption()
    test_alt_out_of_vocab_ignored()
    print("ALL_ERK_FEATURE_TESTS_PASS (5/5)")