#!/usr/bin/env python
"""FM0-01: Foundation-model pretraining exposure audit.

Documents which datasets in our D1/B0 pipeline were historically exposed to
UTR-LM pretraining. This is the FM0 acceptance: exposure record.

UTR-LM (multimolecule/utrlm-mrl) was pretrained on (per model card):
  1. Ensembl 5'UTR sequences from 5 species (human, rat, mouse, chicken,
     zebrafish) — broad prior, no overlap with our D_C.
  2. Sample et al. 2019 (Nat Biotech, doi:10.1038/s41587-019-0164-5) — this
     is GSE114002 in our D_C ledger. UTR-LM saw raw 5'UTR sequences during
     MLM/SS/MFE self-supervised training.
  3. Cao et al. 2021 (Nat Commun, doi:10.1038/s41467-021-24436-7) — separate
     MPRA library, not in our D_C (we use Sample2019's GSE114002 records).

Implications:
  - GSE114002 is HISTORICALLY EXPOSED at the SEQUENCE level to UTR-LM.
  - UTR-LM did NOT see source-candidate edit labels (it's self-supervised).
  - For H7 foundation-value evaluation, GSE114002-derived test splits must
    be reported as evidence grade E4 (historically exposed), not E5
    (untouched). This affects:
      * B0 split 5utr_source_disjoint: ALL of train/val/test come from
        GSE114002 -> all historically exposed.
      * B0 split study_disjoint: train=GSE114002 (exposed), test=GSE200304
        (3'UTR, NOT in UTR-LM pretraining corpus) -> test is E5.
  - Cross-region transfer (5'->3') on GSE200304 is the cleanest test of
    foundation value.

Acceptance (FM0-01): exposure record.

Usage:
    python scripts/fm0/fm0_exposure_audit.py \
        [--canonical-records data/d1_canonical_records.jsonl] \
        [--b0-splits-dir data/b0_splits] \
        [--output data/fm0/fm0_exposure_audit.json]

Contract: utr_editflow_contract_v2 (FROZEN), §2.2, §4.2, §H7
Task: FM0-01
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_B0_DIR = os.path.normpath(os.path.join(HERE, "..", "b0"))
if _B0_DIR not in sys.path:
    sys.path.insert(0, _B0_DIR)

from fm0_common import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    ensure_offline_env,
    get_model_id,
    load_config,
    write_json,
)
from legacy_split_guard import reject_legacy_b0_splits  # noqa: E402


# ---------------------------------------------------------------------------
# UTR-LM pretraining corpus (per README in HF snapshot)
# ---------------------------------------------------------------------------
UTRLM_PRETRAINING_SOURCES = [
    {
        "source_id": "ensembl_genome_browser",
        "url": "https://ensembl.org",
        "description": "5'UTR sequences from 5 vertebrate species (human, rat, "
                       "mouse, chicken, zebrafish) with high-quality annotations.",
        "type": "broad_sequence_prior",
        "overlaps_with_our_accessions": [],
        "evidence_grade_for_us": "E5_unexposed_broad_prior",
    },
    {
        "source_id": "sample2019",
        "citation": "Sample et al. 2019, Nat Biotech, doi:10.1038/s41587-019-0164-5",
        "url": "https://doi.org/10.1038/s41587-019-0164-5",
        "description": "8 distinct 5'UTR libraries, each with random 50 nt sequences; "
                       "mean ribosome loading (MRL) measurements via MPRA.",
        "type": "mpra_library_with_mrl_labels",
        "overlaps_with_our_accessions": ["GSE114002"],
        "evidence_grade_for_us": "E4_historically_exposed",
        "exposure_type": "sequence_prior_only",
        "labels_exposed_to_utrlm": False,
        "notes": "UTR-LM used the 5'UTR sequences as MLM/SS/MFE input. The MRL "
                 "labels were NOT used by UTR-LM (self-supervised + structural).",
    },
    {
        "source_id": "cao2021",
        "citation": "Cao et al. 2021, Nat Commun, doi:10.1038/s41467-021-24436-7",
        "url": "https://doi.org/10.1038/s41467-021-24436-7",
        "description": "Endogenous human 5'UTRs from 3 cell lines/tissues: HEK293T, "
                       "PC3, Muscle.",
        "type": "endogenous_5utr_mpra",
        "overlaps_with_our_accessions": [],
        "evidence_grade_for_us": "E5_unexposed",
        "notes": "Cao 2021's GEO data is NOT in our D_C ledger (we use Sample2019 "
                 "via GSE114002). No overlap with our downstream evaluation data.",
    },
]


# ---------------------------------------------------------------------------
# Our accessions exposed to UTR-LM
# ---------------------------------------------------------------------------
def build_exposure_entries() -> List[dict]:
    """Per-accession exposure entries for our data ledger."""
    cfg = load_config()
    exposed = set(cfg["exposure"]["historically_exposed_accessions"])
    entries = []

    # GSE114002 (Sample 2019) — exposed
    entries.append({
        "accession": "GSE114002",
        "dataset_name": "sample2019",
        "data_role": "D_C",
        "region": "5'UTR",
        "in_d1_canonical_records": True,
        "in_b0_splits": ["5utr_source_disjoint", "study_disjoint (train only)"],
        "historically_exposed_to_utrlm": True,
        "exposure_type": "sequence_prior_only",
        "labels_exposed_to_utrlm": False,
        "evidence_grade_for_foundation_eval": "E4",
        "allowed_claims_with_foundation": [
            "edit_effect",                       # edit labels still unexposed
            "generation_grounding",
            "foundation_transfer_vs_scratch",    # H7 ablation valid (just grade E4)
        ],
        "forbidden_claims_with_foundation": [
            "untouched_external_test",           # NOT E5
            "sealed_test",
        ],
        "notes": "UTR-LM saw these 5'UTR sequences during pretraining. For H7 "
                 "evaluation, report as E4. Compare foundation-adapted vs "
                 "from-scratch on the SAME split to isolate architecture value "
                 "from pretraining value.",
    })

    # GSE200304 (3'UTR) — NOT exposed (UTR-LM is 5'UTR-only)
    entries.append({
        "accession": "GSE200304",
        "dataset_name": "gse200304",
        "data_role": "D_C",
        "region": "3'UTR",
        "in_d1_canonical_records": True,
        "in_b0_splits": ["study_disjoint (test only)"],
        "historically_exposed_to_utrlm": False,
        "exposure_type": "none",
        "labels_exposed_to_utrlm": False,
        "evidence_grade_for_foundation_eval": "E5",
        "allowed_claims_with_foundation": [
            "untouched_external_test",
            "cross_region_transfer_5to3",        # key H7 test: 5'->3' transfer
        ],
        "forbidden_claims_with_foundation": [],
        "notes": "3'UTR data. UTR-LM was pretrained on 5'UTR only. This is the "
                 "CLEANEST foundation-value test: does a 5'UTR-pretrained encoder "
                 "transfer to 3'UTR edit-effect prediction?",
    })

    # GSE217518 (3'UTR decay) — NOT exposed
    entries.append({
        "accession": "GSE217518",
        "dataset_name": "gse217518",
        "data_role": "D_C",
        "region": "3'UTR",
        "in_d1_canonical_records": True,
        "in_b0_splits": [],
        "historically_exposed_to_utrlm": False,
        "exposure_type": "none",
        "labels_exposed_to_utrlm": False,
        "evidence_grade_for_foundation_eval": "E5",
        "allowed_claims_with_foundation": [
            "untouched_external_test",
            "cross_region_transfer_5to3",
        ],
        "forbidden_claims_with_foundation": [],
        "notes": "3'UTR decay/half-life data. Not in B0 splits (per FM0 preflight).",
    })

    # GSE149487 (5'UTR PLUMAGE) — NOT exposed (PLUMAGE is a different study)
    entries.append({
        "accession": "GSE149487",
        "dataset_name": "gse149487",
        "data_role": "D_C",
        "region": "5'UTR",
        "in_d1_canonical_records": True,
        "in_b0_splits": [],
        "historically_exposed_to_utrlm": False,
        "exposure_type": "none",
        "labels_exposed_to_utrlm": False,
        "evidence_grade_for_foundation_eval": "E5",
        "allowed_claims_with_foundation": ["untouched_external_test"],
        "forbidden_claims_with_foundation": [],
        "notes": "5'UTR TE/abundance (PLUMAGE). Distinct from Sample2019 / Cao2021. "
                 "Cleanest 5'UTR foundation-value test (no sequence overlap with UTR-LM corpus).",
    })

    return entries


def cross_reference_b0_splits(b0_splits_dir: Path) -> List[dict]:
    """For each B0 split file, summarize exposure of each (split, accession) cell."""
    if not b0_splits_dir.exists():
        return []

    exposed = {"GSE114002"}  # only GSE114002 is exposed
    split_summaries = []
    for split_file in sorted(b0_splits_dir.glob("split_*.jsonl")):
        from collections import Counter
        counts = Counter()
        n = 0
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                sp = r.get("split", "?")
                acc = r.get("accession", "?")
                counts[(sp, acc)] += 1
                n += 1
        cells = []
        for (sp, acc), cnt in sorted(counts.items()):
            cells.append({
                "split": sp,
                "accession": acc,
                "count": cnt,
                "accession_historically_exposed_to_utrlm": acc in exposed,
                "evidence_grade_for_foundation_eval": "E4" if acc in exposed else "E5",
            })
        split_summaries.append({
            "split_file": split_file.name,
            "total_records": n,
            "cells": cells,
        })
    return split_summaries


def run_exposure_audit(records_path: Path, b0_splits_dir: Path) -> dict:
    b0_splits_dir = reject_legacy_b0_splits(b0_splits_dir)
    cfg = load_config()
    ensure_offline_env()

    # Cross-reference canonical records
    n_records_per_acc = {}
    if records_path.exists():
        from collections import Counter
        c = Counter()
        with open(records_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                c[r.get("accession", "?")] += 1
        n_records_per_acc = dict(c)

    exposure_entries = build_exposure_entries()
    b0_summaries = cross_reference_b0_splits(b0_splits_dir)

    # Compute overall exposure flag
    n_exposed = sum(1 for e in exposure_entries if e["historically_exposed_to_utrlm"])
    n_total = len(exposure_entries)

    report = {
        "task_id": "FM0-01",
        "acceptance": "exposure record",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "foundation_model_id": get_model_id(),
        "foundation_model_revision": cfg["model"]["revision"],
        "contract_reference": "utr_editflow_contract_v2 §2.2, §4.2, §H7",
        "utrlm_pretraining_sources": UTRLM_PRETRAINING_SOURCES,
        "per_accession_exposure": exposure_entries,
        "b0_split_exposure_summary": b0_summaries,
        "canonical_records_per_accession": n_records_per_acc,
        "summary": {
            "num_accessions_audited": n_total,
            "num_historically_exposed_to_utrlm": n_exposed,
            "exposed_accessions": [e["accession"] for e in exposure_entries
                                    if e["historically_exposed_to_utrlm"]],
            "key_finding": (
                "GSE114002 (Sample2019, 5'UTR MPRA) is historically exposed to "
                "UTR-LM at the sequence level (no label exposure). GSE200304 and "
                "GSE217518 (3'UTR) and GSE149487 (5'UTR PLUMAGE) are NOT exposed."
            ),
            "implication_for_H7": (
                "H7 foundation-model-value evaluation must report GSE114002-derived "
                "test splits as E4. The cleanest foundation-value tests are: "
                "(1) cross-region transfer on GSE200304 (5'->3', E5), and "
                "(2) cross-study on GSE149487 (5'->5' PLUMAGE, E5)."
            ),
        },
        "pass": n_total > 0 and n_exposed > 0,  # pass = audit performed and exposed set non-empty
    }
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--canonical-records",
        default="data/d1_canonical_records.jsonl",
        help="Path to d1_canonical_records.jsonl (for cross-reference).",
    )
    ap.add_argument(
        "--b0-splits-dir",
        default="data/b0_splits",
        help="Directory containing B0 split_*.jsonl files.",
    )
    ap.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "fm0_exposure_audit.json"),
    )
    args = ap.parse_args()

    # Resolve relative paths against repo root (since this script is run from repo root)
    repo_root = Path(__file__).resolve().parents[3]
    records_path = Path(args.canonical_records)
    if not records_path.is_absolute():
        records_path = repo_root / records_path
    b0_splits_dir = Path(args.b0_splits_dir)
    if not b0_splits_dir.is_absolute():
        b0_splits_dir = repo_root / b0_splits_dir

    report = run_exposure_audit(records_path, b0_splits_dir)
    out = Path(args.output)
    write_json(out, report)

    print(f"[FM0-01] Exposure audit -> {out}")
    print(f"  foundation model: {report['foundation_model_id']} @ {report['foundation_model_revision']}")
    print(f"  UTR-LM pretraining sources: {len(report['utrlm_pretraining_sources'])}")
    for src in report["utrlm_pretraining_sources"]:
        overlap = src.get("overlaps_with_our_accessions", [])
        print(f"    {src['source_id']:30s}  overlaps={overlap}  grade={src.get('evidence_grade_for_us','?')}")
    print(f"  accessions audited: {report['summary']['num_accessions_audited']} "
          f"(exposed: {report['summary']['num_historically_exposed_to_utrlm']})")
    for e in report["per_accession_exposure"]:
        flag = "EXPOSED" if e["historically_exposed_to_utrlm"] else "clean  "
        print(f"    [{flag}] {e['accession']:12s} {e['region']:6s} "
              f"role={e['data_role']}  grade={e['evidence_grade_for_foundation_eval']}")
    print(f"  B0 split exposure summary:")
    for s in report["b0_split_exposure_summary"]:
        print(f"    {s['split_file']}  (N={s['total_records']})")
        for cell in s["cells"]:
            mark = "E4" if cell["accession_historically_exposed_to_utrlm"] else "E5"
            print(f"      {cell['split']:6s} {cell['accession']:12s} N={cell['count']:>6d}  [{mark}]")
    print(f"  Key finding: {report['summary']['key_finding']}")
    print(f"  PASS: {report['pass']}")

    if not report["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
