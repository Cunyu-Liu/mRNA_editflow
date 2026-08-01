#!/usr/bin/env python
"""B0-04: Evaluation track assignment + track-role ambiguity audit.

Contract: utr_editflow_contract_v2 (FROZEN) §10.
Task: B0-04
Acceptance: track-role ambiguity = 0

Three evaluation tracks (v2 contract §10):
  closed_measured_pool:  measured source-candidate pairs, closed support
  heldout_generative:    held-out source->candidate generative likelihood/recovery
  open_legal_generation: open-support legal generation under constraints

The three tracks are evaluation MODES applied to the existing split manifests
(B0-02). Each record in a split manifest is assigned a per-track role based on
its split role (train/val/test) and track eligibility.

Track eligibility:
  closed_measured_pool:   D_C paired records (measured source + candidate)
  heldout_generative:     D_C paired records (measured candidate = recovery target)
  open_legal_generation:  any record with source_sequence

Track roles (per split, per track):
  closed_measured_pool:   {train,val,test}_pair | none
  heldout_generative:     {train,val,test}_heldout_source | none
  open_legal_generation:  {train,val,test}_gen_source | none

Exposure class (derived from the split-role prefix):
  TRAIN     (train_*)      — record is exposed to the model during training
  EVAL_VAL  (val_*)        — record used for model selection
  EVAL_TEST (test_*)       — record used for final evaluation (held out)
  NONE      (none)         — record not used in this track

Ambiguity = 0 means: within each split, every record has a single consistent
exposure class across all three tracks. A record that is TRAIN in one track and
EVAL (val/test) in another would be ambiguous (it would simultaneously be
"seen" and "unseen"). Because all three tracks inherit the same split
partition, the exposure class is consistent by construction; this module audits
that property and reports 0 violations.

Cross-split role differences (a record is train in split A but test in split B)
are BY DESIGN — each split is a separate evaluation scenario with its own
trained model — and are reported as informational, not as failures.

Outputs:
  data/b0_04_eval_track_manifest.jsonl   — per (split, record) track assignment
  data/b0_04_eval_track_audit_report.json — ambiguity audit report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Make B0 schemas importable
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from canonical_schemas import EVAL_TRACKS  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPLIT_FILES: Dict[str, str] = {
    "5utr_source_disjoint": "split_5utr_source_disjoint.jsonl",
    "3utr_source_disjoint": "split_3utr_source_disjoint.jsonl",
    "study_disjoint": "split_study_disjoint.jsonl",
    "cross_region_transfer": "split_cross_region_transfer.jsonl",
}

# Exposure classes
EXPOSURE_TRAIN = "TRAIN"
EXPOSURE_EVAL_VAL = "EVAL_VAL"
EXPOSURE_EVAL_TEST = "EVAL_TEST"
EXPOSURE_NONE = "NONE"

# Map split role -> exposure class
SPLIT_ROLE_TO_EXPOSURE: Dict[str, str] = {
    "train": EXPOSURE_TRAIN,
    "val": EXPOSURE_EVAL_VAL,
    "test": EXPOSURE_EVAL_TEST,
}

# Eval exposure classes (anything that is not TRAIN and not NONE)
EVAL_EXPOSURE_CLASSES = frozenset({EXPOSURE_EVAL_VAL, EXPOSURE_EVAL_TEST})

# Track-specific role suffixes (the split-role prefix carries the exposure class)
TRACK_ROLE_SUFFIX: Dict[str, str] = {
    "closed_measured_pool": "pair",
    "heldout_generative": "heldout_source",
    "open_legal_generation": "gen_source",
}

# Eligibility requirements per track (checked against canonical record + ledger)
TRACK_REQUIRES_PAIRED: Dict[str, bool] = {
    "closed_measured_pool": True,   # needs measured source + candidate
    "heldout_generative": True,     # needs measured candidate as recovery target
    "open_legal_generation": False,  # only needs a source sequence
}

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_canonical_records(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in load_jsonl(path):
        out[r["record_id"]] = r
    return out


def load_exposure_ledger(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return out
    for r in load_jsonl(path):
        out[r["record_id"]] = r
    return out


# ---------------------------------------------------------------------------
# Track assignment
# ---------------------------------------------------------------------------

def record_is_paired(record: Dict[str, Any]) -> bool:
    """A paired record has both source_sequence and candidate_sequence."""
    return bool(record.get("source_sequence")) and bool(
        record.get("candidate_sequence")
    )


def record_has_source(record: Dict[str, Any]) -> bool:
    return bool(record.get("source_sequence"))


def is_eligible(track: str, record: Dict[str, Any]) -> bool:
    """Return True if record satisfies the eligibility requirements for track."""
    if TRACK_REQUIRES_PAIRED[track]:
        return record_is_paired(record)
    return record_has_source(record)


def role_for_track(track: str, split_role: str, eligible: bool) -> str:
    """Return the track-specific role string.

    Examples:
      closed_measured_pool, train, True  -> "train_pair"
      heldout_generative, test, True     -> "test_heldout_source"
      open_legal_generation, val, True   -> "val_gen_source"
      *, *, False                        -> "none"
    """
    if not eligible or split_role not in SPLIT_ROLE_TO_EXPOSURE:
        return "none"
    suffix = TRACK_ROLE_SUFFIX[track]
    return f"{split_role}_{suffix}"


def exposure_class_for_role(role: str) -> str:
    """Map a track role to its exposure class."""
    if role == "none":
        return EXPOSURE_NONE
    for prefix, exposure in SPLIT_ROLE_TO_EXPOSURE.items():
        if role.startswith(prefix + "_"):
            return exposure
    return EXPOSURE_NONE


def assign_track_roles(
    split_name: str,
    split_entries: List[Dict[str, Any]],
    canonical_records: Dict[str, Dict[str, Any]],
    ledger: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Assign per-track roles for every record in a split manifest.

    Returns a list of assignment dicts (one per record):
      {
        "split": split_name,
        "record_id": ...,
        "accession": ...,
        "region": ...,
        "split_role": "train"|"val"|"test",
        "data_role": "D_C"|... (from ledger, may be None),
        "record_type": "paired"|... (from ledger, may be None),
        "exposure_class": "TRAIN"|"EVAL_VAL"|"EVAL_TEST",
        "eligible_tracks": [track, ...],
        "track_roles": {track: role, ...},
        "track_exposure_classes": {track: exposure, ...}
      }
    """
    assignments: List[Dict[str, Any]] = []
    for entry in split_entries:
        rid = entry["record_id"]
        split_role = entry["split"]
        rec = canonical_records.get(rid)
        led = ledger.get(rid, {})

        track_roles: Dict[str, str] = {}
        track_exposures: Dict[str, str] = {}
        eligible_tracks: List[str] = []
        for track in EVAL_TRACKS:
            elig = is_eligible(track, rec) if rec else False
            role = role_for_track(track, split_role, elig)
            track_roles[track] = role
            track_exposures[track] = exposure_class_for_role(role)
            if elig:
                eligible_tracks.append(track)

        # The overall exposure class is the split_role's class (consistent
        # across all eligible tracks by construction).
        overall = SPLIT_ROLE_TO_EXPOSURE.get(split_role, EXPOSURE_NONE)

        assignments.append({
            "split": split_name,
            "record_id": rid,
            "accession": entry.get("accession") or rec.get("accession") if rec else entry.get("accession"),
            "region": entry.get("region") or (rec.get("region") if rec else None),
            "split_role": split_role,
            "data_role": led.get("data_role"),
            "record_type": led.get("record_type"),
            "exposure_class": overall,
            "eligible_tracks": eligible_tracks,
            "track_roles": track_roles,
            "track_exposure_classes": track_exposures,
        })
    return assignments


# ---------------------------------------------------------------------------
# Ambiguity audit
# ---------------------------------------------------------------------------

def check_cross_track_consistency(
    assignments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """HARD GATE: within each split, no record has conflicting exposure classes
    across tracks.

    A conflict means a record is TRAIN in one track and EVAL (val/test) in
    another, or EVAL_VAL in one and EVAL_TEST in another.
    """
    violations: List[Dict[str, Any]] = []
    for a in assignments:
        classes = set(a["track_exposure_classes"].values()) - {EXPOSURE_NONE}
        has_train = EXPOSURE_TRAIN in classes
        has_eval = bool(classes & EVAL_EXPOSURE_CLASSES)
        has_val = EXPOSURE_EVAL_VAL in classes
        has_test = EXPOSURE_EVAL_TEST in classes
        conflict = False
        reasons: List[str] = []
        if has_train and has_eval:
            conflict = True
            reasons.append("TRAIN vs EVAL")
        if has_val and has_test:
            conflict = True
            reasons.append("EVAL_VAL vs EVAL_TEST")
        if conflict:
            violations.append({
                "split": a["split"],
                "record_id": a["record_id"],
                "track_exposure_classes": a["track_exposure_classes"],
                "reason": "+".join(reasons),
            })
    return {
        "name": "cross_track_exposure_consistency",
        "hard_gate": True,
        "n_violations": len(violations),
        "violations": violations[:20],
        "pass": len(violations) == 0,
    }


def check_single_role_per_track(
    assignments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """HARD GATE: each record has at most one role per track per split.

    This is structural (the assignment produces one role per track), but we
    audit it to detect any future regression that assigns multiple roles.
    """
    seen: Dict[Tuple[str, str, str], int] = defaultdict(int)
    duplicates: List[Dict[str, Any]] = []
    for a in assignments:
        for track, role in a["track_roles"].items():
            key = (a["split"], a["record_id"], track)
            seen[key] += 1
            if seen[key] > 1:
                duplicates.append({
                    "split": a["split"],
                    "record_id": a["record_id"],
                    "track": track,
                    "count": seen[key],
                })
    return {
        "name": "single_role_per_track",
        "hard_gate": True,
        "n_violations": len(duplicates),
        "violations": duplicates[:20],
        "pass": len(duplicates) == 0,
    }


def check_eligibility_correctness(
    assignments: List[Dict[str, Any]],
    canonical_records: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """HARD GATE: a record assigned a non-'none' role must be eligible for that
    track; a record not eligible must have role 'none'."""
    violations: List[Dict[str, Any]] = []
    for a in assignments:
        rec = canonical_records.get(a["record_id"])
        if rec is None:
            violations.append({
                "split": a["split"],
                "record_id": a["record_id"],
                "reason": "record missing from canonical_records",
            })
            continue
        for track in EVAL_TRACKS:
            role = a["track_roles"][track]
            elig = is_eligible(track, rec)
            if role != "none" and not elig:
                violations.append({
                    "split": a["split"],
                    "record_id": a["record_id"],
                    "track": track,
                    "role": role,
                    "reason": "assigned role but not eligible",
                })
            if role == "none" and elig and a["split_role"] in SPLIT_ROLE_TO_EXPOSURE:
                violations.append({
                    "split": a["split"],
                    "record_id": a["record_id"],
                    "track": track,
                    "role": role,
                    "reason": "eligible but assigned none",
                })
    return {
        "name": "eligibility_correctness",
        "hard_gate": True,
        "n_violations": len(violations),
        "violations": violations[:20],
        "pass": len(violations) == 0,
    }


def check_train_eval_boundary(
    assignments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """HARD GATE: within each split, no record is both TRAIN (in any track) and
    EVAL (in any track). This is a stricter restatement of cross-track
    consistency focused on the train/eval leakage boundary."""
    violations: List[Dict[str, Any]] = []
    for a in assignments:
        classes = set(a["track_exposure_classes"].values()) - {EXPOSURE_NONE}
        if EXPOSURE_TRAIN in classes and (classes & EVAL_EXPOSURE_CLASSES):
            violations.append({
                "split": a["split"],
                "record_id": a["record_id"],
                "track_exposure_classes": a["track_exposure_classes"],
            })
    return {
        "name": "train_eval_boundary",
        "hard_gate": True,
        "n_violations": len(violations),
        "violations": violations[:20],
        "pass": len(violations) == 0,
    }


def summarize_cross_split_roles(
    all_assignments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """INFORMATIONAL: how many records have different split_roles across splits.

    This is expected (different evaluation scenarios) and is NOT a failure.
    """
    rid_to_roles: Dict[str, Set[str]] = defaultdict(set)
    rid_to_splits: Dict[str, Set[str]] = defaultdict(set)
    for a in all_assignments:
        rid_to_roles[a["record_id"]].add(a["split_role"])
        rid_to_splits[a["record_id"]].add(a["split"])

    multi_role = {
        rid: sorted(roles)
        for rid, roles in rid_to_roles.items()
        if len(roles) > 1
    }
    return {
        "name": "cross_split_role_diversity",
        "hard_gate": False,
        "n_records_in_multiple_splits": sum(
            1 for s in rid_to_splits.values() if len(s) > 1
        ),
        "n_records_with_multiple_split_roles": len(multi_role),
        "examples": dict(list(multi_role.items())[:10]),
        "note": (
            "Cross-split role differences are BY DESIGN — each split is a "
            "separate evaluation scenario. Reported for transparency; not a "
            "failure."
        ),
    }


def summarize_track_counts(
    all_assignments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Count records per (split, track, role)."""
    counts: Dict[str, Counter] = defaultdict(Counter)
    for a in all_assignments:
        for track, role in a["track_roles"].items():
            if role != "none":
                counts[f"{a['split']}/{track}"][role] += 1
    return {
        "name": "track_role_counts",
        "hard_gate": False,
        "counts": {k: dict(v) for k, v in sorted(counts.items())},
    }


def run_eval_track_audit(
    splits_dir: str,
    canonical_records_path: str,
    exposure_ledger_path: str,
) -> Dict[str, Any]:
    """Run the full B0-04 eval track assignment + ambiguity audit."""
    print("Loading canonical records...")
    canonical_records = load_canonical_records(canonical_records_path)
    print(f"  {len(canonical_records)} records")

    print("Loading exposure ledger...")
    ledger = load_exposure_ledger(exposure_ledger_path)
    print(f"  {len(ledger)} ledger entries")

    all_assignments: List[Dict[str, Any]] = []
    per_split: Dict[str, Any] = {}

    for split_name, filename in SPLIT_FILES.items():
        path = os.path.join(splits_dir, filename)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping {split_name}")
            per_split[split_name] = {"error": "manifest not found", "n_records": 0}
            continue
        entries = load_jsonl(path)
        print(f"Assigning tracks for {split_name} ({len(entries)} records)...")
        assignments = assign_track_roles(
            split_name, entries, canonical_records, ledger
        )
        all_assignments.extend(assignments)
        per_split[split_name] = {
            "n_records": len(assignments),
            "by_split_role": dict(Counter(a["split_role"] for a in assignments)),
        }

    print("\nRunning ambiguity audits...")
    checks = [
        check_cross_track_consistency(all_assignments),
        check_single_role_per_track(all_assignments),
        check_eligibility_correctness(all_assignments, canonical_records),
        check_train_eval_boundary(all_assignments),
    ]
    for c in checks:
        status = "PASS" if c["pass"] else "FAIL"
        print(f"  [{status}] {c['name']}: {c['n_violations']} violations")

    informational = [
        summarize_cross_split_roles(all_assignments),
        summarize_track_counts(all_assignments),
    ]

    hard_pass = all(c["pass"] for c in checks)
    ambiguity_zero = all(
        c["n_violations"] == 0 for c in checks if c.get("hard_gate")
    )

    return {
        "task": "B0-04",
        "contract": "utr_editflow_contract_v2",
        "tracks": list(EVAL_TRACKS),
        "track_eligibility": {
            "closed_measured_pool": "D_C paired (measured source + candidate)",
            "heldout_generative": "D_C paired (measured candidate = recovery target)",
            "open_legal_generation": "any record with source_sequence",
        },
        "n_total_assignments": len(all_assignments),
        "per_split": per_split,
        "hard_gate_checks": checks,
        "informational": informational,
        "acceptance": {
            "track_role_ambiguity_must_be_zero": ambiguity_zero,
        },
        "overall_pass": hard_pass and ambiguity_zero,
        "manifest_path": "data/b0_04_eval_track_manifest.jsonl",
    }


def write_manifest(
    all_assignments: List[Dict[str, Any]], output_path: str
) -> str:
    """Write the eval track manifest as JSONL. Returns sha256."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with open(out, "w", encoding="utf-8") as f:
        for a in all_assignments:
            line = json.dumps(a, ensure_ascii=False, sort_keys=True)
            f.write(line + "\n")
            h.update(line.encode("utf-8"))
            h.update(b"\n")
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="B0-04: Eval track audit")
    parser.add_argument("--splits-dir", default="data/b0_splits")
    parser.add_argument(
        "--canonical-records", default="data/d1_canonical_records.jsonl"
    )
    parser.add_argument(
        "--exposure-ledger", default="data/data_exposure_ledger.jsonl"
    )
    parser.add_argument(
        "--manifest-output", default="data/b0_04_eval_track_manifest.jsonl"
    )
    parser.add_argument(
        "--report-output", default="data/b0_04_eval_track_audit_report.json"
    )
    args = parser.parse_args()

    report = run_eval_track_audit(
        args.splits_dir, args.canonical_records, args.exposure_ledger
    )

    # Re-run assignment to capture manifest (the audit function does not return
    # the assignments, so we reconstruct by re-loading). Simpler: refactor to
    # return assignments too. For now, re-build.
    canonical_records = load_canonical_records(args.canonical_records)
    ledger = load_exposure_ledger(args.exposure_ledger)
    all_assignments: List[Dict[str, Any]] = []
    for split_name, filename in SPLIT_FILES.items():
        path = os.path.join(args.splits_dir, filename)
        if not os.path.exists(path):
            continue
        entries = load_jsonl(path)
        all_assignments.extend(
            assign_track_roles(split_name, entries, canonical_records, ledger)
        )

    manifest_sha = write_manifest(all_assignments, args.manifest_output)
    report["manifest_sha256"] = manifest_sha
    report["manifest_n_records"] = len(all_assignments)
    print(f"\nManifest written to {args.manifest_output} ({len(all_assignments)} records)")
    print(f"  sha256={manifest_sha}")

    out = Path(args.report_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Audit report written to {out}")
    print(f"Overall pass: {report['overall_pass']}")
    sys.exit(0 if report["overall_pass"] else 1)


if __name__ == "__main__":
    main()
