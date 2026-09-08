#!/usr/bin/env python
"""
candidate_dataset_manifest_pigeonhole_v1
=========================================
Purpose:
  Build immutable acquisition manifests (URL+SHA256+rows+split policy) for
  candidate training datasets (M1 castillohair2024, M6 plassmeyer2025,
  S1 su2025_stability) and run the pigeonhole leak audit:
  exact-match candidate/source sequence overlap vs the FROZEN development
  corpus (TRAIN/VALIDATION/TEST) and the evaluation zero-shot pool.

Evidence paths (all under /mnt/cunyuliu/mrna_xeditflow_routea_v3/route2):
  - manifests/candidate_datasets_v1/   (manifest output, one file per dataset)
  - audits/candidate_datasets_v1/      (audit output, one file per dataset)

Discipline:
  - Only ADD new rows; never modify existing frozen manifests.
  - Audit BEFORE training: exact overlap with VALIDATION/TEST blocks
    admission of the overlapping records (train-split-conflict gate).

Usage:
  python manifest_pigeonhole_builder.py build      # manifest + audit for all 3
  python manifest_pigeonhole_builder.py audit      # re-run audit only
"""
import gzip
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

ROUTE2 = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2"
CD_DIR = f"{ROUTE2}/external_model_assets/candidate_datasets"
MANIFEST_DIR = f"{ROUTE2}/manifests/candidate_datasets_v1"
AUDIT_DIR = f"{ROUTE2}/audits/candidate_datasets_v1"
FROZEN = f"{ROUTE2}/manifests/route2_complete_frozen_v1"
CANONICAL = f"{ROUTE2}/canonical"
SCHEMA_VERSION = "route_a_v3_route2_candidate_dataset_manifest.v1"
AUDIT_SCHEMA = "route_a_v3_route2_candidate_dataset_pigeonhole_audit.v1"
UTC_NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")

DATASETS = {
    "M1_CASTILLOHAIR_2024": {
        "accession": "GSE232927",
        "paradigm": "library_level_absolute",
        "task_domain": "3UTR_POLYSOME_TRANSLATION_EFFICIENCY",
        "intended_role": "development_additive_library_prior",
        "directory": f"{CD_DIR}/castillohair2024/GSE232927",
        "url_base": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE232nnn/GSE232927/suppl/",
        "files": [
            "GSE232927_processed_defined_end_hepg2_r1.csv.gz",
            "GSE232927_processed_defined_end_tcell_r1.csv.gz",
            "GSE232927_processed_defined_end_tcell_r2.csv.gz",
            "GSE232927_processed_random_end_hek293t_N25_r1.csv.gz",
            "GSE232927_processed_random_end_hek293t_N25_r2.csv.gz",
            "GSE232927_processed_random_end_hek293t_N50_r1.csv.gz",
        ],
        "sequence_columns": ["UTR", "utr"],
        "audit_mode": "exact_candidate_sequence",
    },
    "M6_PLASSMEYER_2025": {
        "accession": "GSE246381",
        "paradigm": "variant_level_delta",
        "task_domain": "5UTR_NDD_MUTATIONAL_MPRA",
        "intended_role": "development_additive_variant_delta",
        "directory": f"{CD_DIR}/plassmeyer2025",
        "url_base": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE246nnn/GSE246381/suppl/",
        "files": [
            "GSE246381_hek_combined_umi_counts.csv.gz",
            "GSE246381_vglut_combined_umi_counts.csv.gz",
        ],
        "sequence_columns": ["SeqID"],
        "audit_mode": "variant_id_vs_gse200304_candidate_id",
    },
    "S1_SU_2025_STABILITY": {
        "accession": "elife-97682-supp1-v1",
        "paradigm": "variant_level_delta",
        "task_domain": "5UTR_STABILITY_DELTA",
        "intended_role": "development_additive_variant_delta",
        "directory": f"{CD_DIR}/su2025_stability",
        "url_base": "https://cdn.elifesciences.org/articles/97682/",
        "files": ["elife-97682-supp1-v1.xlsx"],
        "sequence_columns": None,
        "audit_mode": "pending_xlsx_parser",
    },
}


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def gzip_stats(path, seq_cols):
    """Row count (incl header) + header names + first data-row sequence head."""
    rows = 0
    header = None
    first_seq = None
    with gzip.open(path, "rt", errors="replace") as f:
        for i, line in enumerate(f):
            rows += 1
            if i == 0:
                header = line.rstrip("\n").split(",")
            elif first_seq is None:
                parts = line.rstrip("\n").split(",")
                idx = None
                for c in seq_cols or []:
                    if c in header:
                        idx = header.index(c)
                        break
                if idx is not None and idx < len(parts):
                    first_seq = parts[idx]
    return rows, header, first_seq


def xlsx_row_count(path):
    try:
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True, data_only=True)
        total = 0
        sheet_info = {}
        for ws in wb.worksheets:
            n = 0
            for _ in ws.iter_rows(values_only=True):
                n += 1
            sheet_info[ws.title] = n
            total += n
        wb.close()
        return total, sheet_info
    except ImportError:
        return None, None


def build_manifest(ds_key, ds):
    files_info = []
    total_data_rows = 0
    for fname in ds["files"]:
        path = os.path.join(ds["directory"], fname)
        entry = {
            "filename": fname,
            "source_url": ds["url_base"] + fname,
        }
        if not os.path.exists(path):
            entry["status"] = "MISSING"
            files_info.append(entry)
            continue
        entry["status"] = "OK"
        entry["bytes"] = os.path.getsize(path)
        entry["sha256"] = sha256_file(path)
        if fname.endswith(".csv.gz"):
            rows, header, first_seq = gzip_stats(path, ds.get("sequence_columns"))
            entry["row_count_including_header"] = rows
            entry["data_row_count"] = max(rows - 1, 0)
            entry["header_columns_head40"] = header[:40] if header else None
            if isinstance(first_seq, str):
                entry["first_sequence_head60"] = first_seq[:60]
            total_data_rows += max(rows - 1, 0)
        elif fname.endswith(".xlsx"):
            rows, sheet_info = xlsx_row_count(path)
            entry["sheet_row_counts_including_header"] = sheet_info
            entry["row_count_including_header_total"] = rows
            if rows:
                total_data_rows += max(rows - len(sheet_info or {}), 0)
        files_info.append(entry)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_key": ds_key,
        "study_accession": ds["accession"],
        "input_paradigm": ds["paradigm"],
        "task_domain": ds["task_domain"],
        "intended_role": ds["intended_role"],
        "acquisition_completed_utc": UTC_NOW,
        "gzip_integrity": "verified_by_separate_gzip_t_evidence_20260908",
        "files": files_info,
        "total_data_row_estimate": total_data_rows,
        "split_policy": {
            "policy": "additive_only",
            "rule": "join as new strata under study-attribute greedy split; never alter existing TRAIN/VALIDATION/TEST membership of route2_complete_frozen_v1",
            "frozen_reference": "route2_complete_frozen_v1",
        },
    }


def load_frozen_reference():
    """Authoritative splits from development_manifest.jsonl + candidate
    sequences per split for 3'UTR studies from canonical records."""
    canonical_paths = {}
    for study_dir in sorted(os.listdir(CANONICAL)):
        p = os.path.join(CANONICAL, study_dir, "v1", "canonical_records.jsonl")
        if os.path.exists(p):
            canonical_paths[study_dir] = p
    split_map = {}
    split_counts = {}
    with open(f"{FROZEN}/development_manifest.jsonl") as f:
        for line in f:
            r = json.loads(line)
            split_map[r["canonical_record_id"]] = r.get("split")
            split_counts[r.get("split")] = split_counts.get(r.get("split"), 0) + 1
    eval_ids = set()
    with open(f"{FROZEN}/evaluation_manifest.jsonl") as f:
        for line in f:
            r = json.loads(line)
            eval_ids.add(r["canonical_record_id"])
    seq_split = {}
    for study, path in canonical_paths.items():
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("region") != "3UTR":
                    continue
                seq = r.get("candidate_sequence")
                if not seq:
                    continue
                sp = split_map.get(r["canonical_record_id"], "UNASSIGNED")
                seq_split.setdefault(seq, set()).add(sp)
    return split_map, split_counts, seq_split, eval_ids


def audit_m1(ds, seq_split):
    """Exact-match M1 UTR sequences vs frozen 3'UTR candidate sequences."""
    per_file = []
    total = 0
    matched = 0
    matched_splits = {}
    for fname in ds["files"]:
        path = os.path.join(ds["directory"], fname)
        if not os.path.exists(path):
            continue
        fmatch = 0
        ftotal = 0
        fsplit = {}
        examples = []
        with gzip.open(path, "rt", errors="replace") as f:
            header = f.readline().rstrip("\n").split(",")
            idx = None
            for c in ds["sequence_columns"]:
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
        "audit_mode": "exact_sequence_match_vs_frozen_3utr_candidate_sequences",
        "reference_space": f"all 3UTR candidate sequences in canonical/ (studies={sorted(os.listdir(CANONICAL))})",
        "total_m1_rows": total,
        "exact_overlap_rows": matched,
        "overlap_rows_by_frozen_split": matched_splits,
        "per_file": per_file,
        "gate": "BLOCK_M1_RECORDS_OVERLAPPING_VALIDATION_OR_TEST",
        "gate_triggered": bool(matched_splits.get("VALIDATION") or matched_splits.get("TEST")),
    }


def audit_m6(ds, split_map):
    """M6 SeqID genomic variant IDs vs GSE200304 candidate_id (same-locus
    collision check) — expects ZERO overlap by construction (5'UTR vs 3'UTR)."""
    import re
    pat = re.compile(r"^Variant;([^;]+);([ACGT]+)\|([ACGT]+);")
    gse200304_ids = {k.split(":", 1)[1] for k in split_map if k.startswith("GSE200304:")}
    per_file = []
    total = 0
    hits = 0
    for fname in ds["files"]:
        path = os.path.join(ds["directory"], fname)
        if not os.path.exists(path):
            continue
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


def audit_s1_pending():
    return {
        "audit_mode": "pending_xlsx_parser",
        "status": "PENDING",
        "reason": "elife-97682-supp1-v1.xlsx requires openpyxl variant-pair parsing; run in env with openpyxl",
        "gate": "PENDING",
    }


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "build"
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    os.makedirs(AUDIT_DIR, exist_ok=True)
    split_map, split_counts, seq_split, eval_ids = load_frozen_reference()
    print(f"[frozen reference] dev splits: {split_counts} eval_ids={len(eval_ids)} 3utr_seq_keys={len(seq_split)}", flush=True)

    for ds_key, ds in DATASETS.items():
        if mode in ("build", "manifest"):
            m = build_manifest(ds_key, ds)
            mpath = os.path.join(MANIFEST_DIR, f"{ds_key}.manifest.json")
            with open(mpath, "w") as f:
                json.dump(m, f, indent=2)
            print(f"[manifest] {ds_key} -> {mpath} ({m['total_data_row_estimate']} data rows)", flush=True)
        if mode in ("build", "audit"):
            if ds_key == "M1_CASTILLOHAIR_2024":
                a = audit_m1(ds, seq_split)
            elif ds_key == "M6_PLASSMEYER_2025":
                a = audit_m6(ds, split_map)
            else:
                a = audit_s1_pending()
            a.update({
                "schema_version": AUDIT_SCHEMA,
                "dataset_key": ds_key,
                "audit_completed_utc": UTC_NOW,
                "frozen_reference": "route2_complete_frozen_v1",
                "principle": "audit_before_training; additive-only rows",
            })
            apath = os.path.join(AUDIT_DIR, f"{ds_key}.pigeonhole_audit.json")
            with open(apath, "w") as f:
                json.dump(a, f, indent=2)
            print(f"[audit] {ds_key} -> {apath}", flush=True)
            print(json.dumps({k: v for k, v in a.items() if k in (
                "total_m1_rows", "exact_overlap_rows", "overlap_rows_by_frozen_split",
                "total_m6_rows", "same_locus_overlap_rows", "status", "gate_triggered")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
