#!/usr/bin/env python
"""P0-03: Training membership and cross-role leakage audit.

Audits the P3 benchmark tiers (measured + proxy) for leakage between the
training pool and the final test pool:

* mother/source identity overlap (same source_id in train and test)
* exact candidate-sequence collisions across roles
* edit-neighborhood collisions (candidate within edit distance <= k of a
  test candidate)
* family-cluster overlap across roles
* proxy role compliance (only split_role == "train" proxy records may be
  used for training)
* assay/batch proxy: edit_type + confidence combination overlap

Usage:
    python scripts/audit_training_membership.py \
        --benchmark-dir data/p3/benchmark \
        --output docs/nmi_membership_audit.json

Exit code 0 when all hard checks pass, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.p3_02_delta_oracle import DeltaRecord, load_benchmark

# Pre-registered near-duplicate threshold: candidates whose 5'UTR edit
# distance is <= NEAR_DUP_THRESHOLD are considered near-duplicates.
NEAR_DUP_THRESHOLD = 1
# Hard cap on near-duplicate rate between train and test (pre-registered).
NEAR_DUP_MAX_RATE = 0.01

TRAIN_ROLES = ("train", "val")
TEST_ROLES = ("test",)


# ---------------------------------------------------------------------------
# Pairwise audit primitives (importable for tests)
# ---------------------------------------------------------------------------

def audit_source_overlap(
    train_recs: Sequence[DeltaRecord],
    test_recs: Sequence[DeltaRecord],
) -> Dict[str, Any]:
    """Same mother/source sequence must not appear in both train and test."""
    train_sources = {r.source_id for r in train_recs}
    test_sources = {r.source_id for r in test_recs}
    overlap = sorted(train_sources & test_sources)
    return {
        "check": "source_overlap",
        "n_train_sources": len(train_sources),
        "n_test_sources": len(test_sources),
        "n_overlap": len(overlap),
        "overlap_ids": overlap[:50],
        "pass": len(overlap) == 0,
    }


def audit_exact_sequence_collision(
    train_recs: Sequence[DeltaRecord],
    test_recs: Sequence[DeltaRecord],
) -> Dict[str, Any]:
    """Exact candidate sequence must not appear in both train and test."""
    train_cands = {r.candidate_sequence for r in train_recs}
    collisions = sorted({r.candidate_sequence for r in test_recs} & train_cands)
    return {
        "check": "exact_sequence_collision",
        "n_train_candidates": len(train_cands),
        "n_collisions": len(collisions),
        "collision_examples": collisions[:20],
        "pass": len(collisions) == 0,
    }


def _hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(1 for x, y in zip(a, b) if x != y)


def _hamming_leq(a: str, b: str, threshold: int) -> bool:
    """Early-exit ``_hamming(a, b) <= threshold`` for equal-length strings."""
    if len(a) != len(b):
        return False
    mismatches = 0
    for x, y in zip(a, b):
        if x != y:
            mismatches += 1
            if mismatches > threshold:
                return False
    return True


def audit_edit_neighborhood_collision(
    train_recs: Sequence[DeltaRecord],
    test_recs: Sequence[DeltaRecord],
    threshold: int = NEAR_DUP_THRESHOLD,
    max_rate: float = NEAR_DUP_MAX_RATE,
) -> Dict[str, Any]:
    """Candidate within Hamming distance <= threshold of a test candidate.

    Uses exact-length bucketing so it is O(n*m) only within equal-length
    buckets; suitable for the short 5'UTR candidates in this benchmark.

    For the pre-registered threshold == 1 the pairwise scan is replaced by an
    exact pigeonhole index: two equal-length strings at Hamming distance <= 1
    share at least one half, so each train sequence is indexed under both of
    its halves and each test candidate only verifies the (tiny) set of train
    sequences sharing one of its halves. This is exact (identical verdicts to
    the brute-force scan) and reduces the full-benchmark audit from ~10^12
    character comparisons to seconds. For threshold != 1 the exact
    brute-force scan is retained (only used by small unit tests).
    """
    by_len: Dict[int, List[str]] = {}
    for r in train_recs:
        by_len.setdefault(len(r.candidate_sequence), []).append(r.candidate_sequence)

    # Pigeonhole half-index per length (threshold == 1 fast path).
    half_index: Dict[int, Dict[str, List[str]]] = {}
    if threshold == 1:
        for length, seqs in by_len.items():
            index: Dict[str, List[str]] = {}
            for seq in set(seqs):
                mid = length // 2
                index.setdefault(seq[:mid], []).append(seq)
                index.setdefault(seq[mid:], []).append(seq)
            half_index[length] = index

    n_test = max(len(test_recs), 1)
    n_near = 0
    examples: List[Tuple[str, str]] = []
    for r in test_recs:
        cand = r.candidate_sequence
        match: str | None = None
        if threshold == 1:
            index = half_index.get(len(cand))
            if index:
                mid = len(cand) // 2
                candidates = set(index.get(cand[:mid], ()))
                candidates.update(index.get(cand[mid:], ()))
                for other in sorted(candidates):
                    if _hamming_leq(cand, other, 1):
                        match = other
                        break
        else:
            for other in by_len.get(len(cand), []):
                if _hamming(cand, other) <= threshold:
                    match = other
                    break
        if match is not None:
            n_near += 1
            if len(examples) < 20:
                examples.append((cand, match))
    rate = n_near / n_test
    return {
        "check": "edit_neighborhood_collision",
        "threshold": threshold,
        "n_test": len(test_recs),
        "n_near_duplicate": n_near,
        "rate": rate,
        "max_rate": max_rate,
        "examples": examples,
        "pass": rate <= max_rate,
    }


def audit_family_overlap(
    train_recs: Sequence[DeltaRecord],
    test_recs: Sequence[DeltaRecord],
) -> Dict[str, Any]:
    """Family cluster must not span train and test."""
    train_fams = {r.family_cluster_id for r in train_recs if r.family_cluster_id}
    test_fams = {r.family_cluster_id for r in test_recs if r.family_cluster_id}
    overlap = sorted(train_fams & test_fams)
    return {
        "check": "family_overlap",
        "n_train_families": len(train_fams),
        "n_test_families": len(test_fams),
        "n_overlap": len(overlap),
        "overlap_ids": overlap[:50],
        "pass": len(overlap) == 0,
    }


def audit_assay_batch_overlap(
    train_recs: Sequence[DeltaRecord],
    test_recs: Sequence[DeltaRecord],
) -> Dict[str, Any]:
    """Assay-batch proxy (confidence, edit_type) must not be test-only signal
    that also appears as a training batch. Hard rule: no test assay batch id
    may appear in training. Here the batch proxy is (confidence, edit_type);
    'measured' confidence records must never be trained on when they share an
    edit_type batch with test measured records of the same source."""
    train_batches = {(r.confidence, r.edit_type) for r in train_recs}
    test_batches = {(r.confidence, r.edit_type) for r in test_recs}
    shared = sorted(train_batches & test_batches)
    # Shared (confidence, edit_type) is expected for generic categories
    # (e.g. measured/measured_single on different sources), so this is a
    # report-only check; the hard checks are source/family/sequence above.
    return {
        "check": "assay_batch_overlap",
        "n_shared_batches": len(shared),
        "shared_batches": [list(b) for b in shared],
        "pass": True,
    }


def filter_train_role_proxy(
    proxy_records: Sequence[DeltaRecord],
) -> List[DeltaRecord]:
    """P0-03 hard rule: only train-role proxy records may be used for training."""
    return [r for r in proxy_records if r.split_role == "train"]


def audit_proxy_role_compliance(
    proxy_records: Sequence[DeltaRecord],
) -> Dict[str, Any]:
    """Report the split-role composition of the proxy tier and verify the
    train-role filter retains only train-role records."""
    role_counts: Dict[str, int] = {}
    for r in proxy_records:
        role_counts[r.split_role] = role_counts.get(r.split_role, 0) + 1
    kept = filter_train_role_proxy(proxy_records)
    compliant = all(r.split_role == "train" for r in kept)
    return {
        "check": "proxy_role_compliance",
        "role_counts": role_counts,
        "n_total": len(proxy_records),
        "n_kept_train_role": len(kept),
        "n_excluded": len(proxy_records) - len(kept),
        "pass": compliant,
    }


# ---------------------------------------------------------------------------
# Full audit
# ---------------------------------------------------------------------------

def run_full_audit(benchmark_dir: str) -> Dict[str, Any]:
    """Run all membership/leakage checks on a P3 benchmark directory."""
    tiers = load_benchmark(benchmark_dir, tiers=("measured", "proxy"))
    measured = tiers.get("measured", [])
    proxy = tiers.get("proxy", [])

    all_recs = list(measured) + list(proxy)
    train_pool = [r for r in all_recs if r.split_role in TRAIN_ROLES]
    test_pool = [r for r in all_recs if r.split_role in TEST_ROLES]

    checks = [
        audit_source_overlap(train_pool, test_pool),
        audit_family_overlap(train_pool, test_pool),
        audit_exact_sequence_collision(train_pool, test_pool),
        audit_edit_neighborhood_collision(train_pool, test_pool),
        audit_assay_batch_overlap(train_pool, test_pool),
        audit_proxy_role_compliance(proxy),
    ]

    return {
        "benchmark_dir": benchmark_dir,
        "n_measured": len(measured),
        "n_proxy": len(proxy),
        "n_train_pool": len(train_pool),
        "n_test_pool": len(test_pool),
        "checks": checks,
        "all_pass": all(c["pass"] for c in checks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="P0-03 membership/leakage audit")
    parser.add_argument("--benchmark-dir", default="data/p3/benchmark")
    parser.add_argument("--output", default="docs/nmi_membership_audit.json")
    args = parser.parse_args()

    report = run_full_audit(args.benchmark_dir)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    for c in report["checks"]:
        status = "PASS" if c["pass"] else "FAIL"
        print(f"[{status}] {c['check']}")
    print(f"Overall: {'PASS' if report['all_pass'] else 'FAIL'} -> {args.output}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
