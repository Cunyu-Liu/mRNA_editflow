#!/usr/bin/env python
"""
frozen_reference_v2
====================
Rebuild the pigeonhole reference space to the FULL frozen corpus:
  - public canonical:  GSE200304 (3UTR 6547), GSE217518 (3UTR 2314 + 5UTR 1695)
  - private canonical: ENCSR854RUF (3UTR 79800), GSE186455 (3UTR 752),
    GSE232572 (3UTR 8068), GSE269595 (3UTR 30966), GSE114002 (5UTR 3899),
    GSE149487 (5UTR 192), GSE256185 (0 rows)
Split authority: development_manifest.jsonl (126,165 records) +
evaluation_manifest.jsonl (8,068 = GSE232572 EVALUATION_ZERO_SHOT).

Audit semantics (unchanged from v1):
  - exact match on candidate_sequence / source_sequence
  - overlap with VALIDATION/TEST splits => gate BLOCK
  - evaluation pool overlap => separate flag (headline zero-shot eligibility)
Outputs:
  - audits/candidate_datasets_v1/reference_space_v2.json
  - re-runs M1 (16.5M rows) and M6 (65,980 rows) exact audits with full space
"""
import gzip
import json
import os
import re
import sys
from datetime import datetime, timezone

ROUTE2 = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2"
CANONICAL = f"{ROUTE2}/canonical"
FROZEN = f"{ROUTE2}/manifests/route2_complete_frozen_v1"
CD_DIR = f"{ROUTE2}/external_model_assets/candidate_datasets"
AUDIT_DIR = f"{ROUTE2}/audits/candidate_datasets_v1"
UTC_NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")
AUDIT_SCHEMA = "route_a_v3_route2_candidate_dataset_pigeonhole_audit.v1"

CANONICAL_FILES = {
    "GSE200304": "canonical_records.jsonl",
    "GSE217518": "canonical_records.jsonl",
    "ENCSR854RUF": "canonical_records.private.jsonl",
    "GSE114002": "canonical_records.private.jsonl",
    "GSE149487": "canonical_records.private.jsonl",
    "GSE186455": "canonical_records.private.jsonl",
    "GSE232572": "canonical_records.private.jsonl",
    "GSE256185": "canonical_records.private.jsonl",
    "GSE269595": "canonical_records.private.jsonl",
}


def load_reference():
    split_map = {}
    eval_ids = set()
    with open(f"{FROZEN}/development_manifest.jsonl") as f:
        for line in f:
            r = json.loads(line)
            split_map[r["canonical_record_id"]] = r.get("split")
    with open(f"{FROZEN}/evaluation_manifest.jsonl") as f:
        for line in f:
            r = json.loads(line)
            eval_ids.add(r["canonical_record_id"])
    seq_split = {}
    stats = {}
    for study, fname in CANONICAL_FILES.items():
        p = f"{CANONICAL}/{study}/v1/{fname}"
        if not os.path.exists(p):
            stats[study] = "MISSING"
            continue
        n = 0
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                n += 1
                seq = r.get("candidate_sequence") or r.get("source_sequence")
                if not seq:
                    continue
                if r["canonical_record_id"] in eval_ids:
                    sp = "EVALUATION_ZERO_SHOT"
                else:
                    sp = split_map.get(r["canonical_record_id"], "UNASSIGNED")
                seq_split.setdefault(seq, set()).add(sp)
        stats[study] = n
    return seq_split, stats, split_map, eval_ids


def audit_m1(seq_split):
    per_file = []
    total = 0
    matched = 0
    matched_splits = {}
    ds_dir = f"{CD_DIR}/castillohair2024/GSE232927"
    files = [
        "GSE232927_processed_defined_end_hepg2_r1.csv.gz",
        "GSE232927_processed_defined_end_tcell_r1.csv.gz",
        "GSE232927_processed_defined_end_tcell_r2.csv.gz",
        "GSE232927_processed_random_end_hek293t_N25_r1.csv.gz",
        "GSE232927_processed_random_end_hek293t_N25_r2.csv.gz",
        "GSE232927_processed_random_end_hek293t_N50_r1.csv.gz",
    ]
    for fname in files:
        path = os.path.join(ds_dir, fname)
        fmatch = 0
        ftotal = 0
        fsplit = {}
        examples = []
        with gzip.open(path, "rt", errors="replace") as f:
            header = f.readline().rstrip("\n").split(",")
            idx = None
            for c in ("UTR", "utr"):
                if c in header:
                    idx = header.index(c)
                    break
            if idx is None:
                continue
            for line in f:
                parts = line.rstrip("\n").split(",")
                if len(parts) <= idx:
                    continue
                seq = parts[idx].strip().upper()
                ftotal += 1
                if seq in seq_split:
                    fmatch += 1
                    for sp in seq_split[seq]:
                        fsplit[sp] = fsplit.get(sp, 0) + 1
                    if len(examples) < 3:
                        examples.append(seq[:80])
        per_file.append({
            "filename": fname,
            "rows": ftotal,
            "exact_overlap_rows": fmatch,
            "overlap_rows_by_frozen_split": fsplit,
            "example_sequences_head80": examples,
        })
        total += ftotal
        matched += fmatch
        for sp, n in fsplit.items():
            matched_splits[sp] = matched_splits.get(sp, 0) + n
    return {
        "audit_mode": "exact_sequence_match_vs_full_frozen_corpus_v2",
        "reference_space": "all 9 canonical studies (public+private), candidate+source sequences",
        "total_m1_rows": total,
        "exact_overlap_rows": matched,
        "overlap_rows_by_frozen_split": matched_splits,
        "per_file": per_file,
        "gate": "BLOCK_M1_RECORDS_OVERLAPPING_VALIDATION_OR_TEST",
        "gate_triggered": bool(matched_splits.get("VALIDATION") or matched_splits.get("TEST")),
    }


def audit_m6(split_map):
    pat = re.compile(r"^Variant;([^;]+);([ACGT]+)\|([ACGT]+);")
    corpus_ids = {}
    for cri, sp in split_map.items():
        corpus_ids.setdefault(cri.split(":", 1)[0], set()).add(cri.split(":", 1)[1])
    # also GSE232572-style ids are row-keys, not genomic; only GSE200304 uses genomic ids
    gse200304_ids = corpus_ids.get("GSE200304", set())
    per_file = []
    total = 0
    hits = 0
    ds_dir = f"{CD_DIR}/plassmeyer2025"
    for fname in ("GSE246381_hek_combined_umi_counts.csv.gz", "GSE246381_vglut_combined_umi_counts.csv.gz"):
        path = os.path.join(ds_dir, fname)
        fmatch = 0
        ftotal = 0
        examples = []
        with gzip.open(path, "rt", errors="replace") as f:
            f.readline()
            for line in f:
                seqid = line.split(",", 1)[0]
                ftotal += 1
                m = pat.match(seqid)
                if not m:
                    continue
                locus = m.group(1)
                ref, alt = m.group(2), m.group(3)
                for gid in gse200304_ids:
                    if gid.startswith(locus + "_") and gid.endswith("_" + ref + "-" + alt):
                        fmatch += 1
                        if len(examples) < 3:
                            examples.append(seqid)
                        break
        per_file.append({
            "filename": fname,
            "rows": ftotal,
            "same_locus_variant_overlap_rows": fmatch,
            "examples": examples,
        })
        total += ftotal
        hits += fmatch
    return {
        "audit_mode": "variant_id_locus_overlap_vs_gse200304_candidate_ids",
        "total_m6_rows": total,
        "same_locus_overlap_rows": hits,
        "per_file": per_file,
        "expected": "ZERO (5UTR NDD variants vs 3UTR GSE200304 library design differ)",
        "gate": "INVESTIGATE_IF_ANY_OVERLAP",
        "gate_triggered": hits > 0,
    }


def main():
    seq_split, stats, split_map, eval_ids = load_reference()
    print(f"[reference v2] studies={stats} eval={len(eval_ids)} seq_keys={len(seq_split)}", flush=True)
    ref_out = {
        "schema_version": "route_a_v3_route2_candidate_dataset_reference_space.v2",
        "built_utc": UTC_NOW,
        "studies_and_record_counts": stats,
        "evaluation_record_count": len(eval_ids),
        "distinct_sequence_keys": len(seq_split),
        "frozen_reference": "route2_complete_frozen_v1",
    }
    with open(f"{AUDIT_DIR}/reference_space_v2.json", "w") as f:
        json.dump(ref_out, f, indent=2)
    a1 = audit_m1(seq_split)
    a1.update({
        "schema_version": AUDIT_SCHEMA,
        "dataset_key": "M1_CASTILLOHAIR_2024",
        "audit_completed_utc": UTC_NOW,
        "frozen_reference": "route2_complete_frozen_v1 (full corpus v2)",
        "principle": "audit_before_training; additive-only rows",
    })
    with open(f"{AUDIT_DIR}/M1_CASTILLOHAIR_2024.pigeonhole_audit.v2.json", "w") as f:
        json.dump(a1, f, indent=2)
    print(f"[audit v2] M1 overlap={a1['exact_overlap_rows']} by_split={a1['overlap_rows_by_frozen_split']}", flush=True)
    a6 = audit_m6(split_map)
    a6.update({
        "schema_version": AUDIT_SCHEMA,
        "dataset_key": "M6_PLASSMEYER_2025",
        "audit_completed_utc": UTC_NOW,
        "frozen_reference": "route2_complete_frozen_v1 (full corpus v2)",
        "principle": "audit_before_training; additive-only rows",
    })
    with open(f"{AUDIT_DIR}/M6_PLASSMEYER_2025.pigeonhole_audit.v2.json", "w") as f:
        json.dump(a6, f, indent=2)
    print(f"[audit v2] M6 overlap={a6['same_locus_overlap_rows']}", flush=True)


if __name__ == "__main__":
    main()
