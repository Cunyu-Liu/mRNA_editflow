#!/usr/bin/env python
"""D1-02: Build data exposure ledger.

For each canonical record, emit an exposure-ledger entry documenting:
  - data role (D_A / D_C / D_D / D_E)
  - evidence grade (E1 / E2 / E4)
  - exposure status (unexposed / historically_exposed / observational_no_labels / incomplete)
  - whether labels are allowed for new training / new hyperparameter selection
  - allowed / forbidden claims
  - historical exposure path (for GSE246381)

D1-02 acceptance: exposure ledger coverage = 100%
  (every canonical record has exactly one ledger entry).

Usage:
    python scripts/d1/build_exposure_ledger.py \
        [--input data/d1_canonical_records.jsonl] \
        [--output data/data_exposure_ledger.jsonl]

Contract: utr_editflow_contract_v2 (FROZEN)
Task: D1-02
"""

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Per-dataset exposure policy (from v2 contract §4.1 + §4.2)
# ---------------------------------------------------------------------------
# Source: configs/utr_editflow_contract_v2.yaml + v2_contract_overview.md §4.1
DATASET_EXPOSURE_POLICY = {
    "GSE114002": {
        "data_role": "D_C",
        "evidence_grade": "E2",
        "exposure_status": "unexposed",
        "historically_exposed": False,
        "labels_allowed_for_new_training": True,
        "labels_allowed_for_new_hyperparameter_selection": True,
        "allowed_claims": ["edit_effect", "generation_grounding"],
        "forbidden_claims": [],
        "historical_exposure_path": None,
        "notes": "primary supervised; 5'UTR MRL",
    },
    "GSE149487": {
        "data_role": "D_C",
        "evidence_grade": "E2",
        "exposure_status": "unexposed",
        "historically_exposed": False,
        "labels_allowed_for_new_training": True,
        "labels_allowed_for_new_hyperparameter_selection": True,
        "allowed_claims": ["edit_effect", "generation_grounding"],
        "forbidden_claims": [],
        "historical_exposure_path": None,
        "notes": "primary supervised; 5'UTR TE/abundance (PLUMAGE)",
    },
    "GSE217518": {
        "data_role": "D_C",
        "evidence_grade": "E2",
        "exposure_status": "unexposed",
        "historically_exposed": False,
        "labels_allowed_for_new_training": True,
        "labels_allowed_for_new_hyperparameter_selection": True,
        "allowed_claims": ["edit_effect", "generation_grounding"],
        "forbidden_claims": [],
        "historical_exposure_path": None,
        "notes": "primary supervised; 3'UTR decay/half-life",
    },
    "GSE200304": {
        "data_role": "D_C",
        "evidence_grade": "E2",
        "exposure_status": "unexposed",
        "historically_exposed": False,
        "labels_allowed_for_new_training": True,
        "labels_allowed_for_new_hyperparameter_selection": True,
        "allowed_claims": ["edit_effect", "generation_grounding"],
        "forbidden_claims": [],
        "historical_exposure_path": None,
        "notes": "primary supervised; 3'UTR TE/RNA/stability",
    },
    "GSE232572": {
        "data_role": "D_C",
        "evidence_grade": "E2",
        "exposure_status": "unexposed",
        "historically_exposed": False,
        "labels_allowed_for_new_training": True,
        "labels_allowed_for_new_hyperparameter_selection": True,
        "allowed_claims": ["edit_effect", "generation_grounding"],
        "forbidden_claims": [],
        "historical_exposure_path": None,
        "notes": "primary supervised; 3'UTR mRNA abundance (MapUTR, Fu et al. 2024)",
    },
    "GSE186455": {
        "data_role": "D_C",
        "evidence_grade": "E2",
        "exposure_status": "unexposed",
        "historically_exposed": False,
        "labels_allowed_for_new_training": True,
        "labels_allowed_for_new_hyperparameter_selection": True,
        "allowed_claims": ["edit_effect", "generation_grounding"],
        "forbidden_claims": [],
        "historical_exposure_path": None,
        "notes": "primary supervised; 3'UTR TRAP-seq MPRA (N2a/Vglut)",
    },
    "ENCSR854RUF": {
        "data_role": "D_C",
        "evidence_grade": "E2",
        "exposure_status": "unexposed",
        "historically_exposed": False,
        "labels_allowed_for_new_training": True,
        "labels_allowed_for_new_hyperparameter_selection": True,
        "allowed_claims": ["edit_effect", "generation_grounding"],
        "forbidden_claims": [],
        "historical_exposure_path": None,
        "notes": "MPRAu 3'UTR; sequences reconstructed from hg19 + variant coordinates; all 11,969 paired records verified",
    },
    "GSE145046": {
        "data_role": "D_D",
        "evidence_grade": "E2",
        "exposure_status": "unexposed",
        "historically_exposed": False,
        "labels_allowed_for_new_training": True,
        "labels_allowed_for_new_hyperparameter_selection": True,
        "allowed_claims": ["dense_pretraining", "multi_step_generation"],
        "forbidden_claims": [],
        "historical_exposure_path": None,
        "notes": "dense landscape; 5'UTR 10-nt randomized library (observational, no source-candidate pairs)",
    },
    "GSE246381": {
        "data_role": "D_C",
        "evidence_grade": "E2",
        "exposure_status": "unexposed",
        "historically_exposed": False,
        "labels_allowed_for_new_training": True,
        "labels_allowed_for_new_hyperparameter_selection": True,
        "allowed_claims": ["edit_effect", "generation_grounding"],
        "forbidden_claims": [],
        "historical_exposure_path": None,
        "notes": "primary supervised; 5'UTR translational efficiency (Polysome/Monosome UMI)",
    },
    "GSE207584": {
        "data_role": "D_A",
        "evidence_grade": "E2",
        "exposure_status": "observational_no_labels",
        "historically_exposed": False,
        "labels_allowed_for_new_training": False,
        "labels_allowed_for_new_hyperparameter_selection": False,
        "allowed_claims": ["utr_sequence_prior", "foundation_adaptation", "generative_denoising"],
        "forbidden_claims": [
            "edit_effect",
            "generation_grounding",
            "intervention_claim",
            "cds_grammar_validated",
        ],
        "historical_exposure_path": None,
        "notes": "iCodon CDS library; CDS out-of-scope for v2; observational pretraining only",
    },
    "GSE173083": {
        "data_role": "D_A",
        "evidence_grade": "E2",
        "exposure_status": "observational_no_labels",
        "historically_exposed": False,
        "labels_allowed_for_new_training": False,
        "labels_allowed_for_new_hyperparameter_selection": False,
        "allowed_claims": ["utr_sequence_prior", "foundation_adaptation", "generative_denoising"],
        "forbidden_claims": [
            "edit_effect",
            "generation_grounding",
            "intervention_claim",
            "full_length_optimization_complete",
        ],
        "historical_exposure_path": None,
        "notes": "PERSIST-Seq full-length mRNA; full-length out-of-scope for v2; observational pretraining only",
    },
}


def build_ledger_entry(rec: dict) -> dict:
    """Build a single exposure-ledger entry from a canonical record."""
    accession = rec.get("accession", "")
    policy = DATASET_EXPOSURE_POLICY.get(accession)
    if policy is None:
        # Unknown dataset — fail safe: most restrictive
        policy = {
            "data_role": "D_A",
            "evidence_grade": "E1",
            "exposure_status": "unknown",
            "historically_exposed": False,
            "labels_allowed_for_new_training": False,
            "labels_allowed_for_new_hyperparameter_selection": False,
            "allowed_claims": [],
            "forbidden_claims": ["all_claims_until_classified"],
            "historical_exposure_path": None,
            "notes": f"UNCLASSIFIED accession {accession} — fail safe",
        }

    meta = rec.get("metadata", {})
    record_type = meta.get("record_type", "")

    # Incomplete records inherit the dataset policy but get exposure_status=incomplete
    if record_type == "incomplete":
        exposure_status = "incomplete"
    else:
        exposure_status = policy["exposure_status"]

    entry = {
        "record_id": rec.get("record_id", ""),
        "accession": accession,
        "dataset": rec.get("dataset", ""),
        "region": rec.get("region", ""),
        "data_role": policy["data_role"],
        "evidence_grade": policy["evidence_grade"],
        "exposure_status": exposure_status,
        "historically_exposed": policy["historically_exposed"],
        "labels_allowed_for_new_training": policy["labels_allowed_for_new_training"],
        "labels_allowed_for_new_hyperparameter_selection": policy[
            "labels_allowed_for_new_hyperparameter_selection"
        ],
        "allowed_claims": policy["allowed_claims"],
        "forbidden_claims": policy["forbidden_claims"],
        "historical_exposure_path": policy["historical_exposure_path"],
        "record_type": record_type if record_type else "paired",
        "notes": policy["notes"],
    }
    return entry


def main():
    parser = argparse.ArgumentParser(description="D1-02: Build exposure ledger")
    parser.add_argument(
        "--input",
        default="data/d1_canonical_records.jsonl",
        help="Input canonical records JSONL",
    )
    parser.add_argument(
        "--output",
        default="data/data_exposure_ledger.jsonl",
        help="Output exposure ledger JSONL",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"D1-02: Building exposure ledger")
    print(f"  input:  {input_path}")
    print(f"  output: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_out = 0
    unknown_accessions = set()

    with open(input_path) as fin, open(output_path, "w") as fout:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  ERROR: line {lineno}: JSON decode failed: {e}", file=sys.stderr)
                continue
            n_in += 1
            accession = rec.get("accession", "")
            if accession not in DATASET_EXPOSURE_POLICY:
                unknown_accessions.add(accession)
            entry = build_ledger_entry(rec)
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"\n  canonical records read: {n_in}")
    print(f"  ledger entries written: {n_out}")
    if unknown_accessions:
        print(f"  WARNING: unclassified accessions: {unknown_accessions}")

    # Coverage check: 100% means n_out == n_in
    coverage = (n_out / n_in * 100) if n_in > 0 else 0.0
    print(f"\n  coverage: {n_out}/{n_in} ({coverage:.2f}%)")
    passed = (n_out == n_in) and (coverage == 100.0) and (not unknown_accessions)

    print(f"\n{'='*60}")
    print(f"D1-02 ACCEPTANCE: {'PASS' if passed else 'FAIL'}")
    print(f"  - exposure ledger coverage = 100%: {'PASS' if coverage == 100.0 else 'FAIL'}")
    print(f"  - no unclassified accessions: {'PASS' if not unknown_accessions else 'FAIL'}")
    print(f"{'='*60}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
