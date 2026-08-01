#!/usr/bin/env python
"""D1-02: Audit exposure ledger.

Verifies D1-02 acceptance: exposure ledger coverage = 100%.

Checks:
1. Every canonical record has exactly one ledger entry (coverage = 100%).
2. No duplicate record_ids in the ledger.
3. Every ledger entry has required fields.
4. Per-dataset policy consistency (all records from same dataset share policy).
5. GSE246381 entries have historical_exposure_path populated.

Usage:
    python scripts/d1/audit_exposure_ledger.py \
        [--records data/d1_canonical_records.jsonl] \
        [--ledger data/data_exposure_ledger.jsonl]

Contract: utr_editflow_contract_v2 (FROZEN)
Task: D1-02
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


REQUIRED_FIELDS = {
    "record_id",
    "accession",
    "dataset",
    "region",
    "data_role",
    "evidence_grade",
    "exposure_status",
    "historically_exposed",
    "labels_allowed_for_new_training",
    "labels_allowed_for_new_hyperparameter_selection",
    "allowed_claims",
    "forbidden_claims",
    "historical_exposure_path",
    "record_type",
    "notes",
}

VALID_DATA_ROLES = {"D_A", "D_C", "D_D", "D_E"}
VALID_EVIDENCE_GRADES = {"E1", "E2", "E3", "E4", "E5"}
VALID_EXPOSURE_STATUS = {
    "unexposed",
    "historically_exposed",
    "observational_no_labels",
    "incomplete",
    "unknown",
}


def load_jsonl(path: Path) -> list:
    records = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ERROR: {path} line {lineno}: JSON decode failed: {e}",
                      file=sys.stderr)
    return records


def main():
    parser = argparse.ArgumentParser(description="D1-02: Audit exposure ledger")
    parser.add_argument(
        "--records",
        default="data/d1_canonical_records.jsonl",
        help="Canonical records JSONL",
    )
    parser.add_argument(
        "--ledger",
        default="data/data_exposure_ledger.jsonl",
        help="Exposure ledger JSONL",
    )
    parser.add_argument("--report", default=None, help="Write audit report JSON")
    args = parser.parse_args()

    records_path = Path(args.records)
    ledger_path = Path(args.ledger)

    if not records_path.exists():
        print(f"ERROR: {records_path} not found", file=sys.stderr)
        sys.exit(1)
    if not ledger_path.exists():
        print(f"ERROR: {ledger_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"D1-02: Auditing exposure ledger")
    print(f"  records: {records_path}")
    print(f"  ledger:  {ledger_path}\n")

    canonical = load_jsonl(records_path)
    ledger = load_jsonl(ledger_path)
    print(f"  canonical records: {len(canonical)}")
    print(f"  ledger entries:    {len(ledger)}")

    results = {}

    # --- Check 1: coverage = 100% ---
    canonical_ids = set(r.get("record_id", "") for r in canonical)
    ledger_ids = set(e.get("record_id", "") for e in ledger)
    missing = canonical_ids - ledger_ids
    extra = ledger_ids - canonical_ids
    coverage = (len(canonical_ids & ledger_ids) / len(canonical_ids) * 100) if canonical_ids else 0.0
    cov_passed = (len(missing) == 0 and len(extra) == 0 and coverage == 100.0)
    results["coverage"] = {
        "canonical_count": len(canonical_ids),
        "ledger_count": len(ledger_ids),
        "missing": len(missing),
        "extra": len(extra),
        "coverage_pct": round(coverage, 4),
        "passed": cov_passed,
    }
    print(f"\n--- Coverage ---")
    print(f"  canonical record_ids: {len(canonical_ids)}")
    print(f"  ledger record_ids:    {len(ledger_ids)}")
    print(f"  missing from ledger:  {len(missing)}")
    print(f"  extra in ledger:      {len(extra)}")
    print(f"  coverage: {coverage:.2f}%")
    print(f"  PASSED: {cov_passed}")

    # --- Check 2: no duplicates ---
    from collections import Counter
    id_counts = Counter(e.get("record_id", "") for e in ledger)
    duplicates = {rid: c for rid, c in id_counts.items() if c > 1}
    dup_passed = len(duplicates) == 0
    results["duplicates"] = {"count": len(duplicates), "passed": dup_passed}
    print(f"\n--- Duplicates ---")
    print(f"  duplicate record_ids: {len(duplicates)}")
    print(f"  PASSED: {dup_passed}")

    # --- Check 3: required fields ---
    field_issues = []
    for e in ledger:
        missing_fields = REQUIRED_FIELDS - set(e.keys())
        if missing_fields:
            field_issues.append(f"  {e.get('record_id', '?')}: missing {missing_fields}")
    field_passed = len(field_issues) == 0
    results["required_fields"] = {"passed": field_passed, "issues": field_issues[:10]}
    print(f"\n--- Required fields ---")
    print(f"  PASSED: {field_passed}")
    for issue in field_issues[:5]:
        print(f"  {issue}")

    # --- Check 4: per-dataset policy consistency ---
    policy_issues = []
    by_dataset = defaultdict(list)
    for e in ledger:
        by_dataset[e.get("accession", "?")].append(e)
    for acc, entries in by_dataset.items():
        if not entries:
            continue
        ref = entries[0]
        for key in ("data_role", "evidence_grade", "historically_exposed",
                     "labels_allowed_for_new_training",
                     "labels_allowed_for_new_hyperparameter_selection"):
            ref_val = ref.get(key)
            for e in entries[1:]:
                if e.get(key) != ref_val:
                    policy_issues.append(
                        f"  {acc}: {key} mismatch — "
                        f"{e['record_id']} has {e.get(key)!r} vs ref {ref_val!r}"
                    )
    policy_passed = len(policy_issues) == 0
    results["policy_consistency"] = {"passed": policy_passed, "issues": policy_issues[:10]}
    print(f"\n--- Per-dataset policy consistency ---")
    print(f"  PASSED: {policy_passed}")
    for issue in policy_issues[:5]:
        print(f"  {issue}")

    # --- Check 5: GSE246381 historical_exposure_path ---
    gse246_issues = []
    for e in ledger:
        if e.get("accession") == "GSE246381":
            if not e.get("historical_exposure_path"):
                gse246_issues.append(f"  {e['record_id']}: missing historical_exposure_path")
            if not e.get("historically_exposed"):
                gse246_issues.append(f"  {e['record_id']}: historically_exposed=False")
            if e.get("labels_allowed_for_new_training"):
                gse246_issues.append(f"  {e['record_id']}: labels_allowed_for_new_training=True")
    gse246_passed = len(gse246_issues) == 0
    results["gse246381_constraints"] = {"passed": gse246_passed, "issues": gse246_issues[:10]}
    print(f"\n--- GSE246381 constraints ---")
    print(f"  PASSED: {gse246_passed}")
    for issue in gse246_issues[:5]:
        print(f"  {issue}")

    # --- Check 6: valid enum values ---
    enum_issues = []
    for e in ledger:
        dr = e.get("data_role")
        if dr not in VALID_DATA_ROLES:
            enum_issues.append(f"  {e['record_id']}: invalid data_role={dr!r}")
        eg = e.get("evidence_grade")
        if eg not in VALID_EVIDENCE_GRADES:
            enum_issues.append(f"  {e['record_id']}: invalid evidence_grade={eg!r}")
        es = e.get("exposure_status")
        if es not in VALID_EXPOSURE_STATUS:
            enum_issues.append(f"  {e['record_id']}: invalid exposure_status={es!r}")
    enum_passed = len(enum_issues) == 0
    results["enum_validation"] = {"passed": enum_passed, "issues": enum_issues[:10]}
    print(f"\n--- Enum validation ---")
    print(f"  PASSED: {enum_passed}")

    # --- Per-dataset summary ---
    print(f"\n--- Per-dataset summary ---")
    for acc in sorted(by_dataset):
        entries = by_dataset[acc]
        ref = entries[0]
        print(f"  {acc}: {len(entries)} entries, "
              f"role={ref['data_role']}, grade={ref['evidence_grade']}, "
              f"exposure={ref['exposure_status']}, "
              f"train_labels={'Y' if ref['labels_allowed_for_new_training'] else 'N'}")

    # --- Overall verdict ---
    all_passed = (
        cov_passed and dup_passed and field_passed and
        policy_passed and gse246_passed and enum_passed
    )
    results["overall_passed"] = all_passed
    print(f"\n{'='*60}")
    print(f"D1-02 ACCEPTANCE: {'PASS' if all_passed else 'FAIL'}")
    print(f"  - coverage = 100%:          {'PASS' if cov_passed else 'FAIL'}")
    print(f"  - no duplicates:            {'PASS' if dup_passed else 'FAIL'}")
    print(f"  - required fields present:  {'PASS' if field_passed else 'FAIL'}")
    print(f"  - policy consistency:       {'PASS' if policy_passed else 'FAIL'}")
    print(f"  - GSE246381 constraints:    {'PASS' if gse246_passed else 'FAIL'}")
    print(f"  - enum validation:          {'PASS' if enum_passed else 'FAIL'}")
    print(f"{'='*60}")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nAudit report written to {report_path}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
