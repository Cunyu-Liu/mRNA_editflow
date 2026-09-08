#!/usr/bin/env python
"""
S1_su2025_stability_pigeonhole_audit_v1
========================================
Purpose:
  Sequence-level pigeonhole audit for S1 (Su 2025 elife-97682 Sup_T2_metaTable,
  5,072 variant records) against the frozen development corpus reference:
  - 5'UTR arm: exact source_sequence overlap vs GSE217518 5'UTR canonical records
    (1,695 records, 1,069 distinct source groups, endpoint RNA_HALF_LIFE_MINUTES)
  - 3'UTR arm: exact candidate/source sequence overlap vs all 3'UTR canonical
    records (GSE200304 + GSE217518 3'UTR arm) — including the Dao 2025 cryptic
    splicing risk arm flagged by the QC hard gate.
Key mapping note:
  S1 records do NOT carry the full 5'UTR sequences; audit proceeds at
  variant-cohort level: (GeneSymbol, Chromosome, Start, Stop, Ref, Alt)
  cannot be directly matched to GSE217518 source groups. Therefore this audit
  uses S1's own supplementary sequence columns if present, otherwise it
  records the PENDING status with an explicit reason and the exact key-space
  that a follow-up extraction must provide.
Discipline: additive-only; audit before training; never fabricate matches.
"""
import json
import os
from datetime import datetime, timezone
from openpyxl import load_workbook

ROUTE2 = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2"
CANONICAL = f"{ROUTE2}/canonical"
AUDIT_DIR = f"{ROUTE2}/audits/candidate_datasets_v1"
S1_XLSX = f"{ROUTE2}/external_model_assets/candidate_datasets/su2025_stability/elife-97682-supp1-v1.xlsx"
AUDIT_SCHEMA = "route_a_v3_route2_candidate_dataset_pigeonhole_audit.v1"
UTC_NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_frozen_3utr_seqs():
    seq_split = {}
    split_map = {}
    with open(f"{ROUTE2}/manifests/route2_complete_frozen_v1/development_manifest.jsonl") as f:
        for line in f:
            r = json.loads(line)
            split_map[r["canonical_record_id"]] = r.get("split")
    for study in ("GSE200304", "GSE217518"):
        p = f"{CANONICAL}/{study}/v1/canonical_records.jsonl"
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if r.get("region") != "3UTR":
                    continue
                for key in ("candidate_sequence", "source_sequence"):
                    seq = r.get(key)
                    if seq:
                        sp = split_map.get(r["canonical_record_id"], "UNASSIGNED")
                        seq_split.setdefault(seq, set()).add(sp)
    return seq_split


def load_s1_records():
    wb = load_workbook(S1_XLSX, read_only=True, data_only=True)
    ws = wb["Sup_T2_metaTable"]
    rows = ws.iter_rows(values_only=True)
    hdr = next(rows)
    col = {h: i for i, h in enumerate(hdr) if h is not None}
    recs = []
    for row in rows:
        if row is None:
            continue
        rec = {h: row[i] for h, i in col.items() if i < len(row)}
        if rec.get("Mutant") is None:
            continue
        recs.append(rec)
    wb.close()
    return recs, col


def main():
    # 1) does S1 carry any real sequence column? (a column is a sequence column
    #    only if it holds long nucleotide strings; UTR_Group is a category label)
    recs, col = load_s1_records()
    candidate_cols = [h for h in col if h and ("seq" in str(h).lower() or "utr" in str(h).lower() or "sequence" in str(h).lower())]
    seq_cols = []
    for h in candidate_cols:
        sample_values = [r.get(h) for r in recs[:50] if r.get(h) is not None]
        if not sample_values:
            continue
        is_sequence_col = any(
            isinstance(v, str) and len(v) >= 20 and sum(c in "ACGTN" for c in v.upper()) / len(v) > 0.9
            for v in sample_values
        )
        if is_sequence_col:
            seq_cols.append(h)
    has_sequence = bool(seq_cols)

    # 2) frozen 3'UTR reference
    seq_split_3utr = load_frozen_3utr_seqs()

    # 3) 5'UTR arm reference: GSE217518 5UTR source sequences
    seq_split_5utr = {}
    split_map = {}
    with open(f"{ROUTE2}/manifests/route2_complete_frozen_v1/development_manifest.jsonl") as f:
        for line in f:
            r = json.loads(line)
            split_map[r["canonical_record_id"]] = r.get("split")
    with open(f"{CANONICAL}/GSE217518/v1/canonical_records.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("region") != "5UTR":
                continue
            for key in ("candidate_sequence", "source_sequence"):
                seq = r.get(key)
                if seq:
                    sp = split_map.get(r["canonical_record_id"], "UNASSIGNED")
                    seq_split_5utr.setdefault(seq, set()).add(sp)

    # 4) attempt exact match if sequence columns exist
    result = {
        "schema_version": AUDIT_SCHEMA,
        "dataset_key": "S1_SU_2025_STABILITY",
        "audit_completed_utc": UTC_NOW,
        "frozen_reference": "route2_complete_frozen_v1",
        "principle": "audit_before_training; additive-only rows",
        "s1_record_count": len(recs),
        "s1_utr_group_counts": {
            "5'UTR": sum(1 for r in recs if r.get("UTR_Group") == "5'UTR"),
            "3'UTR": sum(1 for r in recs if r.get("UTR_Group") == "3'UTR"),
        },
        "s1_sequence_columns_detected": seq_cols,
        "frozen_reference_sizes": {
            "3utr_sequence_keys": len(seq_split_3utr),
            "5utr_sequence_keys": len(seq_split_5utr),
        },
    }
    if not has_sequence:
        result["audit_mode"] = "variant_cohort_key_pending_extraction"
        result["status"] = "PENDING_SEQUENCE_EXTRACTION"
        result["reason"] = (
            "Sup_T2_metaTable carries only variant coordinates (GeneSymbol/Chromosome/Start/Stop/"
            "ReferenceAllele/AlternateAllele) and GC/TA summary stats, no full UTR sequence. "
            "The eLife supplement may carry sequence tables in other sheets of the same workbook "
            "or other supplementary files of elife-97682; extraction is required to run the "
            "exact-match audit. Variant-coordinate-level cross-referencing to GSE217518 source "
            "groups (which are sequence-group ids, not genomic coordinates) is not possible "
            "without sequence extraction."
        )
        result["required_follow_up"] = [
            "fetch remaining elife-97682 supplementary files (supp2/supp3 or equivalent sequence tables)",
            "extract 5'UTR source sequences for the 2,492 5'UTR records",
            "re-run exact-match audit vs GSE217518 5UTR arm (1,695 records) and 3'UTR arm vs all frozen 3'UTR sequences",
        ]
        result["gate"] = "PENDING_DO_NOT_TRAIN_UNTIL_SEQUENCE_LEVEL_AUDIT_PASSES"
        result["gate_triggered"] = None
    else:
        # exact match against both arms
        stats_5 = {"total": 0, "matched": 0, "by_split": {}}
        stats_3 = {"total": 0, "matched": 0, "by_split": {}}
        for r in recs:
            group = r.get("UTR_Group")
            seq = None
            for c in seq_cols:
                v = r.get(c)
                if isinstance(v, str) and len(v) >= 20:
                    seq = v.strip().upper()
                    break
            if not seq:
                continue
            if group == "5'UTR":
                stats_5["total"] += 1
                if seq in seq_split_5utr:
                    stats_5["matched"] += 1
                    for sp in seq_split_5utr[seq]:
                        stats_5["by_split"][sp] = stats_5["by_split"].get(sp, 0) + 1
            else:
                stats_3["total"] += 1
                if seq in seq_split_3utr:
                    stats_3["matched"] += 1
                    for sp in seq_split_3utr[seq]:
                        stats_3["by_split"][sp] = stats_3["by_split"].get(sp, 0) + 1
        result["audit_mode"] = "exact_sequence_match_vs_frozen_5utr_and_3utr"
        result["utr5_arm"] = stats_5
        result["utr3_arm"] = stats_3
        result["gate"] = "BLOCK_S1_RECORDS_OVERLAPPING_VALIDATION_OR_TEST"
        result["gate_triggered"] = bool(
            (stats_5["by_split"].get("VALIDATION") or stats_5["by_split"].get("TEST")
             or stats_3["by_split"].get("VALIDATION") or stats_3["by_split"].get("TEST"))
        )
    out = os.path.join(AUDIT_DIR, "S1_SU_2025_STABILITY.pigeonhole_audit.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[audit] S1 -> {out}")
    print(json.dumps({k: result[k] for k in (
        "s1_record_count", "s1_utr_group_counts", "s1_sequence_columns_detected",
        "status", "gate", "gate_triggered") if k in result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
