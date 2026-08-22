#!/usr/bin/env python
"""B0-02: Audit split manifests for leakage and format validity.

Checks performed (acceptance: unexplained_overlap=0, reverse/path_leakage=0):
  (a) manifest_format    — each JSONL line has required fields with valid values
  (b) source_overlap     — for *_source_disjoint splits: no source_sequence
                           string appears in both train and test (exact match)
  (c) accession_overlap  — for study_disjoint: no accession in both train and test
  (d) reverse_leakage    — candidate_sequence(train) ∩ source_sequence(test) = ∅
                           (a train candidate must not be a test source, else the
                           model can reverse a memorised candidate into a test source)
  (e) path_leakage       — edit_script prefix intermediate states of train must not
                           equal the final candidate of test, and vice versa.
                           (per contract amendment v2.2: B0 path-state scope =
                           frozen D1 canonical edit_script prefixes + declared
                           intermediates)

Only splits where the check is *applicable* are audited. For example:
  - source_overlap applies to 5utr/3utr_source_disjoint only.
  - accession_overlap applies to study_disjoint (and cross_region_transfer).
  - reverse/path_leakage apply to all splits that have train and test with
    populated source/candidate sequences.

Contract: utr_editflow_contract_v2 (FROZEN)
Task: B0-02
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Make B0 schemas + D1 edit_script_core importable
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_D1_DIR = os.path.normpath(os.path.join(HERE, "..", "d1"))
if _D1_DIR not in sys.path:
    sys.path.insert(0, _D1_DIR)

from canonical_schemas import UTREditRecord  # noqa: E402
from edit_script_core import apply_edit_script  # noqa: E402
from legacy_split_guard import reject_legacy_b0_splits  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPLIT_TYPES = (
    "5utr_source_disjoint",
    "3utr_source_disjoint",
    "study_disjoint",
    "cross_region_transfer",
)

VALID_SPLITS = ("train", "val", "test")
REQUIRED_MANIFEST_FIELDS = ("record_id", "accession", "region", "split", "split_type")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_manifest(path: str) -> List[Dict[str, Any]]:
    """Load a split manifest JSONL file into a list of entry dicts."""
    entries: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path}:{lineno}: invalid JSON: {e}"
                ) from e
            entries.append(d)
    return entries


def load_paired_records_by_id(path: str) -> Dict[str, UTREditRecord]:
    """Load paired canonical records keyed by record_id."""
    out: Dict[str, UTREditRecord] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rec = UTREditRecord.from_dict(d)
            if rec.is_paired:
                out[rec.record_id] = rec
    return out


# ---------------------------------------------------------------------------
# Intermediate path-state computation
# ---------------------------------------------------------------------------

def compute_intermediate_states(rec: UTREditRecord) -> List[str]:
    """Compute all intermediate states along the edit path.

    Returns the list of states after applying each non-empty prefix of the
    edit_script to the source. The final candidate is NOT included (it is the
    endpoint, audited separately as reverse_leakage on candidate_sequence).

    For records with empty edit_script (edit_distance=0) there are no
    intermediate states, so an empty list is returned.
    """
    ops = list(rec.edit_script.ops) if rec.edit_script else []
    if not ops or rec.source_sequence is None:
        return []
    states: List[str] = []
    current = rec.source_sequence
    # Apply ops one by one; collect state after each op except the last
    # (last op yields the final candidate, audited separately). Actually we
    # collect ALL intermediate prefix states including the final candidate so
    # that path_leakage can catch any shared state; reverse_leakage already
    # covers the final candidate. To avoid double-counting, we include all
    # prefixes of length 1..len(ops).
    for i in range(1, len(ops) + 1):
        state = apply_edit_script(rec.source_sequence, ops[:i])
        if i < len(ops):
            states.append(state)
        # i == len(ops) -> final candidate; covered by reverse_leakage, skip
    return states


# ---------------------------------------------------------------------------
# Audit checks
# ---------------------------------------------------------------------------

def audit_manifest_format(
    entries: List[Dict[str, Any]],
    expected_split_type: str,
) -> Dict[str, Any]:
    """Check (a): manifest format validity.

    Each entry must:
      - be a dict
      - contain all REQUIRED_MANIFEST_FIELDS
      - have split in VALID_SPLITS
      - have split_type == expected_split_type
      - have non-empty record_id / accession / region
    """
    errors: List[str] = []
    n = len(entries)
    for idx, e in enumerate(entries):
        if not isinstance(e, dict):
            errors.append(f"entry[{idx}] is not a dict")
            continue
        for field in REQUIRED_MANIFEST_FIELDS:
            if field not in e:
                errors.append(f"entry[{idx}] missing field '{field}'")
        for field in ("record_id", "accession", "region"):
            if field in e and (not e[field] or not isinstance(e[field], str)):
                errors.append(f"entry[{idx}] field '{field}' must be non-empty str")
        if e.get("split") not in VALID_SPLITS:
            errors.append(
                f"entry[{idx}] split={e.get('split')!r} not in {VALID_SPLITS}"
            )
        if e.get("split_type") != expected_split_type:
            errors.append(
                f"entry[{idx}] split_type={e.get('split_type')!r} != {expected_split_type!r}"
            )
    return {
        "check": "manifest_format",
        "n_entries": n,
        "n_errors": len(errors),
        "errors": errors[:20],
        "pass": len(errors) == 0,
    }


def audit_source_overlap(
    entries: List[Dict[str, Any]],
    records_by_id: Dict[str, UTREditRecord],
) -> Dict[str, Any]:
    """Check (b): for source_disjoint splits, no source_sequence string in
    both train and test (exact match).

    This is the direct leakage guard for source-disjoint splits. An overlap
    means the same source UTR was placed in both train and test, which the
    mmseqs cluster split is designed to prevent.
    """
    train_sources: Set[str] = set()
    test_sources: Set[str] = set()
    missing = 0
    for e in entries:
        rid = e["record_id"]
        rec = records_by_id.get(rid)
        if rec is None or rec.source_sequence is None:
            missing += 1
            continue
        if e["split"] == "train":
            train_sources.add(rec.source_sequence)
        elif e["split"] == "test":
            test_sources.add(rec.source_sequence)
    overlap = train_sources & test_sources
    return {
        "check": "source_overlap",
        "n_train_sources": len(train_sources),
        "n_test_sources": len(test_sources),
        "n_overlap": len(overlap),
        "overlap_examples": sorted(overlap)[:5],
        "n_missing_records": missing,
        "pass": len(overlap) == 0,
    }


def audit_accession_overlap(
    entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Check (c): for study_disjoint, no accession in both train and test."""
    train_acc: Set[str] = set()
    test_acc: Set[str] = set()
    val_acc: Set[str] = set()
    for e in entries:
        acc = e["accession"]
        if e["split"] == "train":
            train_acc.add(acc)
        elif e["split"] == "test":
            test_acc.add(acc)
        elif e["split"] == "val":
            val_acc.add(acc)
    overlap = train_acc & test_acc
    return {
        "check": "accession_overlap",
        "train_accessions": sorted(train_acc),
        "test_accessions": sorted(test_acc),
        "val_accessions": sorted(val_acc),
        "n_overlap": len(overlap),
        "overlap": sorted(overlap),
        "pass": len(overlap) == 0,
    }


def audit_reverse_leakage(
    entries: List[Dict[str, Any]],
    records_by_id: Dict[str, UTREditRecord],
) -> Dict[str, Any]:
    """Check (d): candidate_sequence(train) ∩ source_sequence(test) = ∅.

    A train candidate appearing as a test source means the model can memorise
    the candidate and reverse it into the test source — a reverse-leakage
    channel. We also check source_sequence(train) ∩ candidate_sequence(test)
    for symmetry.
    """
    train_candidates: Set[str] = set()
    test_sources: Set[str] = set()
    train_sources: Set[str] = set()
    test_candidates: Set[str] = set()
    missing = 0
    for e in entries:
        rid = e["record_id"]
        rec = records_by_id.get(rid)
        if rec is None:
            missing += 1
            continue
        cand = rec.candidate_sequence
        src = rec.source_sequence
        if e["split"] == "train":
            if cand:
                train_candidates.add(cand)
            if src:
                train_sources.add(src)
        elif e["split"] == "test":
            if cand:
                test_candidates.add(cand)
            if src:
                test_sources.add(src)
    # candidate(train) -> source(test): the primary reverse channel
    rev_leak = train_candidates & test_sources
    # source(train) -> candidate(test): symmetric guard
    sym_leak = train_sources & test_candidates
    return {
        "check": "reverse_leakage",
        "n_train_candidates": len(train_candidates),
        "n_test_sources": len(test_sources),
        "n_reverse_leakage": len(rev_leak),
        "reverse_leakage_examples": sorted(rev_leak)[:5],
        "n_symmetric_leakage": len(sym_leak),
        "symmetric_leakage_examples": sorted(sym_leak)[:5],
        "n_missing_records": missing,
        "pass": len(rev_leak) == 0 and len(sym_leak) == 0,
    }


def audit_path_leakage(
    entries: List[Dict[str, Any]],
    records_by_id: Dict[str, UTREditRecord],
) -> Dict[str, Any]:
    """Check (e): edit_script prefix intermediate states must not leak across
    train/test.

    Two directions are checked:
      1. train intermediate state == test final candidate
         (train path visits the test endpoint — forward path leakage)
      2. test intermediate state == train final candidate
         (test path visits the train endpoint — reverse path leakage)

    The final candidate itself is covered by reverse_leakage; here we focus on
    intermediate prefix states per the contract amendment v2.2 (frozen D1
    canonical edit_script prefixes + declared intermediates).
    """
    train_intermediates: Set[str] = set()
    test_intermediates: Set[str] = set()
    train_candidates: Set[str] = set()
    test_candidates: Set[str] = set()
    skipped_no_source = 0
    for e in entries:
        rid = e["record_id"]
        rec = records_by_id.get(rid)
        if rec is None or rec.source_sequence is None:
            skipped_no_source += 1
            continue
        inter = compute_intermediate_states(rec)
        cand = rec.candidate_sequence
        if e["split"] == "train":
            train_intermediates.update(inter)
            if cand:
                train_candidates.add(cand)
        elif e["split"] == "test":
            test_intermediates.update(inter)
            if cand:
                test_candidates.add(cand)
    # train intermediate visits test endpoint
    fwd_leak = train_intermediates & test_candidates
    # test intermediate visits train endpoint
    rev_leak = test_intermediates & train_candidates
    return {
        "check": "path_leakage",
        "n_train_intermediates": len(train_intermediates),
        "n_test_intermediates": len(test_intermediates),
        "n_forward_path_leakage": len(fwd_leak),
        "forward_path_leakage_examples": sorted(fwd_leak)[:5],
        "n_reverse_path_leakage": len(rev_leak),
        "reverse_path_leakage_examples": sorted(rev_leak)[:5],
        "n_skipped_no_source": skipped_no_source,
        "pass": len(fwd_leak) == 0 and len(rev_leak) == 0,
    }


# ---------------------------------------------------------------------------
# Split-level audit orchestration
# ---------------------------------------------------------------------------

def audit_one_split(
    manifest_path: str,
    expected_split_type: str,
    records_by_id: Dict[str, UTREditRecord],
) -> Dict[str, Any]:
    """Run all applicable checks for one split manifest."""
    entries = load_manifest(manifest_path)

    # (a) format — always
    fmt = audit_manifest_format(entries, expected_split_type)

    # (b) source_overlap — only for *_source_disjoint
    source_overlap: Optional[Dict[str, Any]] = None
    if expected_split_type.endswith("source_disjoint"):
        source_overlap = audit_source_overlap(entries, records_by_id)

    # (c) accession_overlap — for study_disjoint and cross_region_transfer
    accession_overlap: Optional[Dict[str, Any]] = None
    if expected_split_type in ("study_disjoint", "cross_region_transfer"):
        accession_overlap = audit_accession_overlap(entries)

    # (d) reverse_leakage — all splits with train+test
    reverse = audit_reverse_leakage(entries, records_by_id)

    # (e) path_leakage — all splits with train+test
    path = audit_path_leakage(entries, records_by_id)

    # Compute SHA-256 of manifest
    with open(manifest_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    checks = {
        "manifest_format": fmt,
        "source_overlap": source_overlap,
        "accession_overlap": accession_overlap,
        "reverse_leakage": reverse,
        "path_leakage": path,
    }
    # Determine applicability and overall pass
    applicable_pass = []
    for name, result in checks.items():
        if result is None:
            continue
        applicable_pass.append(result["pass"])

    # Acceptance: unexplained_overlap=0 (source_overlap + accession_overlap
    # where applicable) AND reverse/path_leakage=0
    unexplained_overlap = 0
    if source_overlap is not None:
        unexplained_overlap += source_overlap["n_overlap"]
    if accession_overlap is not None:
        unexplained_overlap += accession_overlap["n_overlap"]
    reverse_leakage_count = reverse["n_reverse_leakage"] + reverse["n_symmetric_leakage"]
    path_leakage_count = path["n_forward_path_leakage"] + path["n_reverse_path_leakage"]

    overall_pass = (
        fmt["pass"]
        and all(applicable_pass)
        and unexplained_overlap == 0
        and reverse_leakage_count == 0
        and path_leakage_count == 0
    )

    return {
        "split_type": expected_split_type,
        "manifest_path": manifest_path,
        "sha256": sha256,
        "n_entries": len(entries),
        "checks": checks,
        "acceptance": {
            "unexplained_overlap": unexplained_overlap,
            "reverse_leakage": reverse_leakage_count,
            "path_leakage": path_leakage_count,
        },
        "pass": overall_pass,
    }


def audit_all_splits(
    splits_dir: str,
    canonical_records_path: str,
) -> Dict[str, Any]:
    """Audit all 4 split manifests."""
    splits_dir = reject_legacy_b0_splits(splits_dir)
    manifest_map = {
        "5utr_source_disjoint": "split_5utr_source_disjoint.jsonl",
        "3utr_source_disjoint": "split_3utr_source_disjoint.jsonl",
        "study_disjoint": "split_study_disjoint.jsonl",
        "cross_region_transfer": "split_cross_region_transfer.jsonl",
    }
    print("Loading canonical records for sequence lookup...")
    records_by_id = load_paired_records_by_id(canonical_records_path)
    print(f"  Loaded {len(records_by_id)} paired records")

    results: Dict[str, Any] = {}
    all_pass = True
    for split_type, filename in manifest_map.items():
        path = str(splits_dir / filename)
        if not os.path.exists(path):
            results[split_type] = {"split_type": split_type, "pass": False,
                                   "error": f"manifest not found: {path}"}
            all_pass = False
            continue
        print(f"\nAuditing {split_type} ({path})...")
        res = audit_one_split(path, split_type, records_by_id)
        results[split_type] = res
        status = "PASS" if res["pass"] else "FAIL"
        print(f"  -> {status}  unexplained_overlap={res['acceptance']['unexplained_overlap']} "
              f"reverse_leakage={res['acceptance']['reverse_leakage']} "
              f"path_leakage={res['acceptance']['path_leakage']}")

    return {
        "task": "B0-02",
        "contract": "utr_editflow_contract_v2",
        "split_audit_results": results,
        "overall_pass": all_pass,
        "acceptance_criteria": {
            "unexplained_overlap_must_be_zero": all(
                r.get("acceptance", {}).get("unexplained_overlap", 0) == 0
                for r in results.values() if "acceptance" in r
            ),
            "reverse_leakage_must_be_zero": all(
                r.get("acceptance", {}).get("reverse_leakage", 0) == 0
                for r in results.values() if "acceptance" in r
            ),
            "path_leakage_must_be_zero": all(
                r.get("acceptance", {}).get("path_leakage", 0) == 0
                for r in results.values() if "acceptance" in r
            ),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="B0-02: Audit split manifests")
    parser.add_argument(
        "--splits-dir", default="data/b0_splits",
        help="Directory containing split manifest JSONL files",
    )
    parser.add_argument(
        "--canonical-records", default="data/d1_canonical_records.jsonl",
        help="Path to canonical_records.jsonl for sequence lookup",
    )
    parser.add_argument(
        "--output", default="data/b0_02_audit_report.json",
        help="Output audit report JSON path",
    )
    args = parser.parse_args()

    report = audit_all_splits(args.splits_dir, args.canonical_records)

    # Write report
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n=== Audit report written to {out_path} ===")
    print(f"Overall pass: {report['overall_pass']}")

    # Exit non-zero on failure so CI can catch regressions
    sys.exit(0 if report["overall_pass"] else 1)


if __name__ == "__main__":
    main()
