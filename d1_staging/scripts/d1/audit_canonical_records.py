#!/usr/bin/env python
"""D1-01: Audit canonical records.

Verifies D1-01 acceptance criteria:
1. apply(edit_script, source) == candidate 100% for all paired records
2. path_ambiguity is quantified (>= 1) for all records
3. Records are JSON-serializable and schema-consistent
4. Reports per-dataset statistics

Usage:
    python scripts/d1/audit_canonical_records.py [--input data/d1_canonical_records.jsonl]

Contract: utr_editflow_contract_v2 (FROZEN)
Task: D1-01
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from edit_script_core import (  # noqa: E402
    EditOp,
    apply_edit_script,
    edit_distance,
    count_optimal_alignments,
)


REQUIRED_FIELDS = {
    "record_id", "dataset", "accession", "region",
    "source_sequence", "candidate_sequence",
    "edit_script", "edit_script_verified",
    "edit_distance", "n_ins", "n_del", "n_sub",
    "path_ambiguity", "labels", "metadata",
}

EXPECTED_DATASETS = {
    "GSE114002", "GSE200304", "GSE145046", "GSE207584", "GSE173083",
    "ENCSR854RUF", "GSE246381", "GSE217518", "GSE149487",
}


def load_records(path: Path) -> list:
    records = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError as e:
                print(f"  ERROR: line {lineno}: JSON decode failed: {e}")
    return records


def audit_schema(records: list) -> dict:
    """Check that all records have required fields."""
    issues = []
    for rec in records:
        missing = REQUIRED_FIELDS - set(rec.keys())
        if missing:
            issues.append(f"  {rec.get('record_id', '?')}: missing fields {missing}")
    return {"passed": len(issues) == 0, "issues": issues}


def audit_edit_script_verification(records: list) -> dict:
    """D1-01 acceptance: apply(edit_script, source) == candidate 100%."""
    n_paired = 0
    n_verified = 0
    n_failed = 0
    failures = []

    for rec in records:
        meta = rec.get("metadata", {})
        rtype = meta.get("record_type", "")
        if rtype in ("observational", "incomplete"):
            continue
        source = rec.get("source_sequence")
        candidate = rec.get("candidate_sequence")
        if source is None or candidate is None:
            continue

        n_paired += 1
        ops_list = rec.get("edit_script", [])
        try:
            ops = [EditOp.from_dict(o) for o in ops_list]
            result = apply_edit_script(source, ops)
            if result == candidate:
                n_verified += 1
                if not rec.get("edit_script_verified", False):
                    failures.append(
                        f"  {rec['record_id']}: apply OK but edit_script_verified=False"
                    )
            else:
                n_failed += 1
                failures.append(
                    f"  {rec['record_id']}: apply FAILED "
                    f"(expected {candidate[:30]!r}..., got {result[:30]!r}...)"
                )
        except Exception as e:
            n_failed += 1
            failures.append(f"  {rec['record_id']}: apply raised {e}")

    return {
        "n_paired": n_paired,
        "n_verified": n_verified,
        "n_failed": n_failed,
        "passed": n_failed == 0,
        "failures": failures[:20],  # cap output
    }


def audit_path_ambiguity(records: list) -> dict:
    """Check that path_ambiguity is quantified (>= 1) for all records."""
    n_checked = 0
    n_ok = 0
    n_missing = 0
    issues = []

    for rec in records:
        meta = rec.get("metadata", {})
        if meta.get("record_type") in ("observational", "incomplete"):
            # Observational records have path_ambiguity=1 by convention
            pa = rec.get("path_ambiguity")
            if pa is None or pa < 1:
                issues.append(f"  {rec['record_id']}: path_ambiguity should be >= 1, got {pa}")
            continue

        source = rec.get("source_sequence")
        candidate = rec.get("candidate_sequence")
        if source is None or candidate is None:
            continue

        n_checked += 1
        pa = rec.get("path_ambiguity")
        if pa is None or pa < 1:
            n_missing += 1
            issues.append(f"  {rec['record_id']}: path_ambiguity missing or < 1: {pa}")
        else:
            n_ok += 1
            # Spot-check: recompute ambiguity for a sample
            if n_checked <= 100:
                expected_pa = count_optimal_alignments(source, candidate)
                if pa != expected_pa:
                    issues.append(
                        f"  {rec['record_id']}: path_ambiguity mismatch "
                        f"(stored={pa}, recomputed={expected_pa})"
                    )

    return {
        "n_checked": n_checked,
        "n_ok": n_ok,
        "n_missing": n_missing,
        "passed": len(issues) == 0,
        "issues": issues[:20],
    }


def audit_minimality(records: list) -> dict:
    """Check that edit_distance == len(edit_script) == Levenshtein distance."""
    n_checked = 0
    issues = []

    for rec in records:
        meta = rec.get("metadata", {})
        if meta.get("record_type") in ("observational", "incomplete"):
            continue
        source = rec.get("source_sequence")
        candidate = rec.get("candidate_sequence")
        if source is None or candidate is None:
            continue

        n_checked += 1
        ops_list = rec.get("edit_script", [])
        stored_ed = rec.get("edit_distance", -1)
        actual_len = len(ops_list)
        expected_ed = edit_distance(source, candidate)

        if stored_ed != expected_ed:
            issues.append(
                f"  {rec['record_id']}: edit_distance={stored_ed} != "
                f"Levenshtein={expected_ed}"
            )
        if actual_len != expected_ed:
            issues.append(
                f"  {rec['record_id']}: len(edit_script)={actual_len} != "
                f"Levenshtein={expected_ed}"
            )

    return {
        "n_checked": n_checked,
        "passed": len(issues) == 0,
        "issues": issues[:20],
    }


def audit_dataset_coverage(records: list) -> dict:
    """Check which datasets are represented."""
    by_dataset = defaultdict(lambda: {"total": 0, "paired": 0,
                                       "observational": 0, "incomplete": 0})
    for rec in records:
        ds = rec.get("accession", "?")
        by_dataset[ds]["total"] += 1
        meta = rec.get("metadata", {})
        rtype = meta.get("record_type", "")
        if rtype == "incomplete":
            by_dataset[ds]["incomplete"] += 1
        elif rtype == "observational" or rec.get("source_sequence") is None:
            by_dataset[ds]["observational"] += 1
        else:
            by_dataset[ds]["paired"] += 1

    found = set(by_dataset.keys())
    missing = EXPECTED_DATASETS - found

    return {
        "by_dataset": dict(by_dataset),
        "found": found,
        "missing": missing,
        "passed": len(missing) == 0,
    }


def audit_json_serializable(records: list) -> dict:
    """Verify all records are JSON-serializable."""
    issues = []
    for rec in records:
        try:
            json.dumps(rec)
        except (TypeError, ValueError) as e:
            issues.append(f"  {rec.get('record_id', '?')}: {e}")
    return {"passed": len(issues) == 0, "issues": issues[:10]}


def main():
    parser = argparse.ArgumentParser(description="D1-01: Audit canonical records")
    parser.add_argument("--input", default="data/d1_canonical_records.jsonl",
                        help="Input JSONL path")
    parser.add_argument("--report", default=None,
                        help="Write audit report to this file (JSON)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        sys.exit(1)

    print(f"D1-01: Auditing canonical records from {input_path}")
    records = load_records(input_path)
    print(f"  loaded {len(records)} records\n")

    # Run all audits
    results = {}
    print("--- Schema audit ---")
    r = audit_schema(records)
    results["schema"] = r
    print(f"  passed: {r['passed']}")
    for issue in r["issues"][:5]:
        print(f"  {issue}")

    print("\n--- Edit script verification (D1-01 acceptance) ---")
    r = audit_edit_script_verification(records)
    results["edit_script_verification"] = {
        "n_paired": r["n_paired"],
        "n_verified": r["n_verified"],
        "n_failed": r["n_failed"],
        "passed": r["passed"],
    }
    print(f"  paired records:  {r['n_paired']}")
    print(f"  verified:        {r['n_verified']}")
    print(f"  failed:          {r['n_failed']}")
    print(f"  PASSED: {r['passed']}")
    for fail in r["failures"][:5]:
        print(f"  {fail}")

    print("\n--- Path ambiguity quantification ---")
    r = audit_path_ambiguity(records)
    results["path_ambiguity"] = {
        "n_checked": r["n_checked"],
        "n_ok": r["n_ok"],
        "n_missing": r["n_missing"],
        "passed": r["passed"],
    }
    print(f"  checked:  {r['n_checked']}")
    print(f"  ok:       {r['n_ok']}")
    print(f"  missing:  {r['n_missing']}")
    print(f"  PASSED: {r['passed']}")
    for issue in r["issues"][:5]:
        print(f"  {issue}")

    print("\n--- Minimality (edit_distance == Levenshtein) ---")
    r = audit_minimality(records)
    results["minimality"] = {"n_checked": r["n_checked"], "passed": r["passed"]}
    print(f"  checked:  {r['n_checked']}")
    print(f"  PASSED: {r['passed']}")
    for issue in r["issues"][:5]:
        print(f"  {issue}")

    print("\n--- Dataset coverage ---")
    r = audit_dataset_coverage(records)
    results["dataset_coverage"] = {
        "found": sorted(r["found"]),
        "missing": sorted(r["missing"]),
        "passed": r["passed"],
    }
    print(f"  datasets found:   {sorted(r['found'])}")
    print(f"  datasets missing: {sorted(r['missing'])}")
    for ds, stats in sorted(r["by_dataset"].items()):
        print(f"    {ds}: total={stats['total']}, paired={stats['paired']}, "
              f"observational={stats['observational']}, incomplete={stats['incomplete']}")

    print("\n--- JSON serializability ---")
    r = audit_json_serializable(records)
    results["json_serializable"] = r["passed"]
    print(f"  PASSED: {r['passed']}")
    for issue in r["issues"][:5]:
        print(f"  {issue}")

    # Overall verdict
    all_passed = (
        results["schema"]["passed"] and
        results["edit_script_verification"]["passed"] and
        results["path_ambiguity"]["passed"] and
        results["minimality"]["passed"] and
        results["json_serializable"]
    )
    print(f"\n{'='*60}")
    print(f"D1-01 ACCEPTANCE: {'PASS' if all_passed else 'FAIL'}")
    print(f"  - apply(edit_script, source) == candidate 100%: "
          f"{'PASS' if results['edit_script_verification']['passed'] else 'FAIL'}")
    print(f"  - path ambiguity quantified: "
          f"{'PASS' if results['path_ambiguity']['passed'] else 'FAIL'}")
    print(f"  - schema consistency: "
          f"{'PASS' if results['schema']['passed'] else 'FAIL'}")
    print(f"  - minimality: "
          f"{'PASS' if results['minimality']['passed'] else 'FAIL'}")
    print(f"  - JSON serializable: "
          f"{'PASS' if results['json_serializable'] else 'FAIL'}")
    print(f"  - dataset coverage: "
          f"{'PASS' if results['dataset_coverage']['passed'] else 'PARTIAL'}")
    print(f"{'='*60}")

    # Write report
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nAudit report written to {report_path}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
