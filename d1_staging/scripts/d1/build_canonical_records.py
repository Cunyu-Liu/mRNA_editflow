#!/usr/bin/env python
"""D1-01: Build canonical records from all datasets.

For each dataset, extract source-candidate pairs (where available), compute
verified edit scripts, and write canonical records to JSONL.

D1-01 acceptance: apply(edit_script, source) == candidate 100% +
path ambiguity quantified.

Usage:
    python scripts/d1/build_canonical_records.py [--data-root DATA/p0] \
        [--output data/d1_canonical_records.jsonl] [--dataset GSE114002]

Contract: utr_editflow_contract_v2 (FROZEN)
Task: D1-01
"""

import argparse
import gzip
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make edit_script_core importable
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from edit_script_core import (  # noqa: E402
    canonical_record,
    canonical_record_no_edit,
    apply_edit_script,
    compute_edit_script,
)

import pandas as pd  # noqa: E402


# Global cap on records per dataset (0 = no cap). Default raised to 0
# (unlimited) for D1-B0 full-data remediation; --max-records still enforces.
MAX_RECORDS_PER_DATASET = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_seq(s: str) -> str:
    """Uppercase, keep only ACGT, convert U to T."""
    if s is None:
        return ""
    s = str(s).strip().upper().replace("U", "T")
    return "".join(c for c in s if c in "ACGT")


def _safe_float(v) -> Optional[float]:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return float(v)
    except (ValueError, TypeError):
        return None


def _open_maybe_gzip(path: Path) -> io.TextIOBase:
    """Open a file that may or may not be gzipped."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _find_file(data_root: Path, accession: str, patterns: List[str]) -> Optional[Path]:
    """Find the first file matching any pattern in the accession directory.

    Searches the accession directory and its subdirectories (processed/,
    reconstructed/) recursively.
    """
    d = data_root / accession
    if not d.exists():
        return None
    for pattern in patterns:
        # Search recursively
        matches = list(d.rglob(pattern))
        if matches:
            return matches[0]
    # List directory contents for debugging
    all_files = sorted(d.rglob("*"))
    for f in all_files:
        if f.is_file():
            print(f"    [scan] {f.relative_to(d)}")
    return None


def _write_records(records: List[dict], output_path: Path):
    """Write records as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  wrote {len(records)} records to {output_path}")


# ---------------------------------------------------------------------------
# Dataset extractors
# ---------------------------------------------------------------------------

def extract_gse114002(data_root: Path) -> List[dict]:
    """GSE114002 — Sample 2019, 5'UTR, D_C.

    File: GSM3130443_designed_library.csv.gz
    Columns: mother (WT source), utr (candidate), rl (MRL label), id, library, info1, info2

    The designed library contains natural 5'UTR variants with their WT mother
    sequence, enabling source→candidate edit script computation.
    """
    accession = "GSE114002"
    print(f"\n[{accession}] Extracting canonical records...")
    fpath = _find_file(data_root, accession, [
        "GSM3130443*designed_library*",
        "*designed_library*",
        "*designed*",
    ])
    if fpath is None:
        print(f"  WARNING: No designed library file found for {accession}")
        return []

    df = pd.read_csv(fpath, compression="infer")
    print(f"  loaded {len(df)} rows, columns: {list(df.columns)}")

    # Locate key columns
    mother_col = None
    utr_col = None
    rl_col = None
    id_col = None
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("mother", "wt", "parent", "reference"):
            mother_col = c
        elif cl in ("utr", "seq", "sequence", "five_prime_utr", "5utr"):
            utr_col = c
        elif cl in ("rl", "mrl", "ribosome_load", "mean_rl"):
            rl_col = c
        elif cl in ("id", "variant_id"):
            id_col = c

    if mother_col is None or utr_col is None:
        print(f"  WARNING: missing mother ({mother_col}) or utr ({utr_col}) column")
        # Fallback: try any column containing 'mother' or 'parent'
        for c in df.columns:
            cl = c.strip().lower()
            if "mother" in cl or "parent" in cl or "wt" in cl:
                mother_col = mother_col or c
            if "utr" in cl or c.strip().lower() == "seq":
                utr_col = utr_col or c

    if mother_col is None or utr_col is None:
        print(f"  SKIP: cannot identify source/candidate columns")
        return []

    records = []
    skipped = 0
    n_identical = 0
    n_edited = 0
    max_records = MAX_RECORDS_PER_DATASET
    for idx, row in df.iterrows():
        if max_records and len(records) >= max_records:
            print(f"  capping at {max_records} records (total rows: {len(df)})")
            break
        source = _normalize_seq(str(row[mother_col]))
        candidate = _normalize_seq(str(row[utr_col]))
        if not source or not candidate:
            skipped += 1
            continue
        if source == candidate:
            n_identical += 1
        else:
            n_edited += 1
        if len(records) % 5000 == 0 and len(records) > 0:
            print(f"  progress: {len(records)} records "
                  f"({n_edited} edited, {n_identical} identical)")
        labels = {}
        if rl_col:
            v = _safe_float(row.get(rl_col))
            if v is not None:
                labels["rl"] = v
        # Ensure unique record_id by always including the row index
        id_val = str(row.get(id_col, "")).strip() if id_col else ""
        if id_val and id_val.lower() != "nan":
            rid = f"{id_val}_{idx}"
        else:
            rid = str(idx)

        metadata = {"source_file": fpath.name, "library": str(row.get("library", ""))}
        rec = canonical_record(
            record_id=f"{accession}_{rid}",
            dataset="sample2019",
            accession=accession,
            region="5'UTR",
            source=source,
            candidate=candidate,
            labels=labels,
            metadata=metadata,
        )
        records.append(rec)

    print(f"  extracted {len(records)} records (skipped {skipped})")
    return records


def extract_gse200304(data_root: Path) -> List[dict]:
    """GSE200304 — 3'UTR, D_C.

    File: GSM6030637_Twist_Oligo_Order_with_merged_ids.txt.gz
    Columns: 201bp (UTR seq), Type (WT/Mutant), ID, merged_id

    Match WT/Mutant pairs on merged_id, then join with count file for labels.
    """
    accession = "GSE200304"
    print(f"\n[{accession}] Extracting canonical records...")
    fpath = _find_file(data_root, accession, [
        "*Twist_Oligo*merged_ids*",
        "*Twist*merged*",
        "*Twist*",
    ])
    if fpath is None:
        print(f"  WARNING: No Twist oligo file found for {accession}")
        return []

    df = pd.read_csv(fpath, sep="\t", compression="infer")
    print(f"  loaded {len(df)} rows, columns: {list(df.columns)}")

    # Locate columns
    seq_col = None
    type_col = None
    id_col = None
    merged_col = None
    for c in df.columns:
        cl = c.strip().lower()
        if "201bp" in cl or cl in ("utr", "seq", "sequence"):
            seq_col = c
        elif cl in ("type", "variant_type"):
            type_col = c
        elif cl in ("id",):
            id_col = c
        elif "merged_id" in cl:
            merged_col = c

    if seq_col is None or type_col is None or merged_col is None:
        print(f"  WARNING: missing key columns: seq={seq_col}, type={type_col}, merged={merged_col}")
        return []

    # Split into WT and Mutant
    wt_df = df[df[type_col].str.strip().str.upper() == "WT"].copy()
    mut_df = df[df[type_col].str.strip().str.upper().isin(["MUTANT", "MUT"])].copy()
    print(f"  WT rows: {len(wt_df)}, Mutant rows: {len(mut_df)}")

    # Try to find count/label file
    labels_by_merged = {}
    count_file = _find_file(data_root, accession, ["*count*", "*cpm*", "*label*", "*activity*"])
    if count_file:
        try:
            cdf = pd.read_csv(count_file, sep="\t", compression="infer")
            print(f"  loaded count file: {count_file.name}, columns: {list(cdf.columns)}")
            # Count file has Barcode (merged_id) and Freq columns
            c_barcode = None
            for c in cdf.columns:
                cl = c.strip().lower()
                if cl in ("barcode", "merged_id", "id"):
                    c_barcode = c
                    break
            if c_barcode:
                for _, crow in cdf.iterrows():
                    mid = str(crow[c_barcode])
                    labels = {}
                    for c in cdf.columns:
                        if c == c_barcode:
                            continue
                        v = _safe_float(crow.get(c))
                        if v is not None:
                            labels[c] = v
                    if labels:
                        labels_by_merged[mid] = labels
        except Exception as e:
            print(f"  WARNING: could not parse count file: {e}")

    # Match WT-Mutant pairs: merged_id has format like
    # "chr2:69461620_G-C_WT" and "chr2:69461620_G-C_Mutant"
    # Strip _WT/_Mutant suffix to get the pair key
    def _pair_key(mid: str) -> str:
        for suffix in ("_WT", "_Mutant", "_MUT", "_wt", "_mutant"):
            if mid.endswith(suffix):
                return mid[: -len(suffix)]
        return mid

    wt_map = {}
    for _, wrow in wt_df.iterrows():
        key = _pair_key(str(wrow[merged_col]))
        wt_map[key] = wrow

    records = []
    skipped = 0
    for _, mrow in mut_df.iterrows():
        key = _pair_key(str(mrow[merged_col]))
        if key not in wt_map:
            skipped += 1
            continue
        wrow = wt_map[key]
        # Also try to get labels by the full mutant merged_id
        full_mid = str(mrow[merged_col])
        source = _normalize_seq(str(wrow[seq_col]))
        candidate = _normalize_seq(str(mrow[seq_col]))
        if not source or not candidate or source == candidate:
            skipped += 1
            continue
        labels = labels_by_merged.get(full_mid, {})
        rid = str(mrow.get(id_col, key)) if id_col else key
        metadata = {
            "source_file": fpath.name,
            "merged_id": full_mid,
            "pair_key": key,
            "wt_id": str(wrow.get(id_col, "")) if id_col else "",
        }
        rec = canonical_record(
            record_id=f"{accession}_{rid}",
            dataset="gse200304",
            accession=accession,
            region="3'UTR",
            source=source,
            candidate=candidate,
            labels=labels,
            metadata=metadata,
        )
        records.append(rec)

    print(f"  extracted {len(records)} records (skipped {skipped})")
    return records


def extract_gse145046(data_root: Path) -> List[dict]:
    """GSE145046 — 5'UTR, D_D (dense landscape, random library).

    Files are tab-separated with no header: seq\tcount\tnorm_count.
    First file: GSM4305122_1_read_count_Randomly_synthesized_oligos.txt.gz
    No source-candidate pairs — use canonical_record_no_edit.
    """
    accession = "GSE145046"
    print(f"\n[{accession}] Extracting observational records (D_D, no pairs)...")
    # Prefer the randomly synthesized oligos file (has sequences)
    fpath = _find_file(data_root, accession, [
        "*Randomly_synthesized*",
        "*read_count*",
        "*.txt*",
    ])
    if fpath is None:
        print(f"  WARNING: No data file found for {accession}")
        return []

    # No header, tab-separated: first column is the 10-nt sequence
    df = pd.read_csv(fpath, sep="\t", compression="infer", header=None,
                     names=["seq", "count", "norm_count"])
    print(f"  loaded {len(df)} rows from {fpath.name}")

    records = []
    skipped = 0
    max_records = MAX_RECORDS_PER_DATASET
    for idx, row in df.iterrows():
        seq = _normalize_seq(str(row["seq"]))
        if len(seq) < 10:
            skipped += 1
            continue
        labels = {}
        v = _safe_float(row.get("count"))
        if v is not None:
            labels["count"] = v
        v = _safe_float(row.get("norm_count"))
        if v is not None:
            labels["norm_count"] = v
        rec = canonical_record_no_edit(
            record_id=f"{accession}_{idx}",
            dataset="gse145046",
            accession=accession,
            region="5'UTR",
            sequence=seq,
            labels=labels,
            metadata={"source_file": fpath.name, "data_role": "D_D"},
        )
        records.append(rec)
        if max_records and len(records) >= max_records:
            print(f"  capping at {max_records} records (total rows: {len(df)})")
            break

    print(f"  extracted {len(records)} records (skipped {skipped})")
    return records


def extract_gse207584(data_root: Path) -> List[dict]:
    """GSE207584 — CDS, D_A (observational).

    iCodon synonymous CDS library. CSV has Name, Protein_id, Group, and
    decay count columns but NO sequences. Sequences are in a separate
    reference FASTA. Mark as observational without sequences.
    """
    accession = "GSE207584"
    print(f"\n[{accession}] Extracting observational records (D_A, CDS)...")
    fpath = _find_file(data_root, accession, [
        "*perfect*",
        "*imperfect*",
        "*.csv*",
    ])
    if fpath is None:
        print(f"  WARNING: No data file found for {accession}")
        return []

    df = pd.read_csv(fpath, compression="infer")
    print(f"  loaded {len(df)} rows, columns: {list(df.columns)}")

    # Try to find sequence column
    seq_col = None
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("cds", "seq", "sequence", "coding_sequence") or "seq" in cl:
            seq_col = c
            break

    records = []
    if seq_col is None:
        # No sequences in CSV — create observational records with identifiers
        print(f"  no sequence column; creating identifier-only observational records")
        for idx, row in df.iterrows():
            labels = {}
            for c in df.columns:
                v = _safe_float(row.get(c))
                if v is not None:
                    labels[str(c)] = v
            rec = canonical_record_no_edit(
                record_id=f"{accession}_{idx}",
                dataset="gse207584",
                accession=accession,
                region="CDS",
                sequence="",  # no sequence in CSV
                labels=labels,
                metadata={
                    "source_file": fpath.name,
                    "data_role": "D_A",
                    "name": str(row.get("Name", "")),
                    "protein_id": str(row.get("Protein_id", "")),
                    "group": str(row.get("Group", "")),
                    "note": "CDS out-of-scope for v2; no sequences in CSV (in reference FASTA)",
                },
            )
            records.append(rec)
    else:
        for idx, row in df.iterrows():
            seq = _normalize_seq(str(row[seq_col]))
            if len(seq) < 10:
                continue
            labels = {}
            for c in df.columns:
                if c == seq_col:
                    continue
                v = _safe_float(row.get(c))
                if v is not None:
                    labels[str(c)] = v
            rec = canonical_record_no_edit(
                record_id=f"{accession}_{idx}",
                dataset="gse207584",
                accession=accession,
                region="CDS",
                sequence=seq,
                labels=labels,
                metadata={"source_file": fpath.name, "data_role": "D_A",
                          "note": "CDS out-of-scope for v2; observational only"},
            )
            records.append(rec)

    print(f"  extracted {len(records)} records")
    return records


def _parse_rdat(path: Path) -> List[dict]:
    """Parse a .rdat file and extract ANNOTATION_DATA entries.

    Each ANNOTATION_DATA line is tab-separated with key:value fields:
      ANNOTATION_DATA:1<TAB>MAPseq:design_name:testing<TAB>...<TAB>sequence:GGAAAUUU...<TAB>signal_to_noise:medium:3.746

    Extracts: MAPseq:ID, MAPseq:design_name, sequence, signal_to_noise (quality + value).

    Returns:
        List of dicts with keys: mapseq_id, design_name, sequence,
        snr_quality, snr_value.
    """
    entries = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("ANNOTATION_DATA:"):
                continue
            parts = line.rstrip("\n").split("\t")
            entry = {
                "mapseq_id": "",
                "design_name": "",
                "sequence": "",
                "snr_quality": "",
                "snr_value": None,
            }
            for part in parts:
                if part.startswith("sequence:"):
                    entry["sequence"] = part[len("sequence:"):]
                elif part.startswith("MAPseq:ID:"):
                    entry["mapseq_id"] = part[len("MAPseq:ID:"):]
                elif part.startswith("MAPseq:design_name:"):
                    entry["design_name"] = part[len("MAPseq:design_name:"):]
                elif part.startswith("signal_to_noise:"):
                    rest = part[len("signal_to_noise:"):]
                    snr_parts = rest.split(":", 1)
                    if len(snr_parts) == 2:
                        entry["snr_quality"] = snr_parts[0]
                        try:
                            entry["snr_value"] = float(snr_parts[1])
                        except ValueError:
                            pass
                    else:
                        entry["snr_quality"] = rest
            if entry["sequence"]:
                entries.append(entry)
    return entries


def extract_gse173083(data_root: Path) -> List[dict]:
    """GSE173083 — PERSIST-seq, full-length synthetic mRNA, D_A (observational).

    Two data sources (zero sequence overlap between them):
    1. .rdat files (6 conditions, same 3030 unique EteRNA designs each):
       ANNOTATION_DATA lines with sequence:, MAPseq:ID:, signal_to_noise:.
       Dedup by MAPseq:ID; record conditions + SNR per condition.
    2. Table S1 xlsx (233 pooled full-length constructs, 1098nt):
       Rich labels (ribosome load, half-life, CAI, dG(MFE), etc.).

    Both: canonical_record_no_edit, region="full_length", data_role="D_A".
    """
    accession = "GSE173083"
    print(f"\n[{accession}] Extracting observational records (D_A, full-length)...")
    d = data_root / accession
    if not d.exists():
        print(f"  WARNING: {d} does not exist")
        return []

    records = []
    max_records = MAX_RECORDS_PER_DATASET

    # --- Source 1: .rdat files ---
    rdat_files = sorted(d.rglob("*.rdat"))
    if rdat_files:
        print(f"  found {len(rdat_files)} .rdat files")
        # Dedup by MAPseq:ID (same designs probed under 6 conditions)
        by_id: Dict[str, dict] = {}
        for rdat_path in rdat_files:
            cond = rdat_path.stem
            entries = _parse_rdat(rdat_path)
            for e in entries:
                mid = e["mapseq_id"]
                if not mid:
                    continue
                if mid not in by_id:
                    by_id[mid] = {
                        "sequence": e["sequence"],
                        "design_name": e["design_name"],
                        "conditions": {},
                    }
                if e["snr_value"] is not None:
                    by_id[mid]["conditions"][cond] = e["snr_value"]

        print(f"  {len(by_id)} unique designs across {len(rdat_files)} conditions")
        n_rdat = 0
        for mid, info in sorted(by_id.items()):
            if max_records and len(records) >= max_records:
                print(f"  capping at {max_records} records")
                break
            seq = _normalize_seq(info["sequence"])
            if len(seq) < 20:
                continue
            labels = {}
            snr_vals = list(info["conditions"].values())
            if snr_vals:
                labels["signal_to_noise_mean"] = sum(snr_vals) / len(snr_vals)
                labels["signal_to_noise_min"] = min(snr_vals)
                labels["signal_to_noise_max"] = max(snr_vals)
            rec = canonical_record_no_edit(
                record_id=f"{accession}_rdat_{mid}",
                dataset="lepplek2022_persistseq",
                accession=accession,
                region="full_length",
                sequence=seq,
                labels=labels,
                metadata={
                    "source_file": "rdat",
                    "data_role": "D_A",
                    "mapseq_id": mid,
                    "design_name": info["design_name"],
                    "conditions": info["conditions"],
                    "n_conditions": len(info["conditions"]),
                    "note": "PERSIST-seq EteRNA design; full-length synthetic RNA, observational",
                },
            )
            records.append(rec)
            n_rdat += 1
        print(f"  extracted {n_rdat} records from .rdat")

    # --- Source 2: Table S1 xlsx (233 pooled constructs) ---
    s1_path = _find_file(data_root, accession, [
        "*Table_S1*",
        "*Attributes*pooled*",
        "*.xlsx",
    ])
    if s1_path is not None:
        try:
            df = pd.read_excel(s1_path, sheet_name=0)
            print(f"  Table S1: {len(df)} rows, {len(df.columns)} columns")
            seq_col = None
            for c in df.columns:
                cl = c.strip().lower()
                if cl == "rna sequence":
                    seq_col = c
                    break
            if seq_col is None:
                for c in df.columns:
                    if "rna sequence" in c.strip().lower():
                        seq_col = c
                        break
            if seq_col is None:
                print(f"  WARNING: no 'RNA sequence' column in Table S1")
            else:
                id_col = None
                for c in df.columns:
                    if c.strip().lower() in ("sequence id", "sequence_id"):
                        id_col = c
                        break
                # Skip sub-region sequence columns from labels
                skip_cols = {seq_col, id_col}
                for c in df.columns:
                    if "sequence" in c.strip().lower() and c != seq_col:
                        skip_cols.add(c)
                n_s1 = 0
                for idx, row in df.iterrows():
                    if max_records and len(records) >= max_records:
                        print(f"  capping at {max_records} records")
                        break
                    seq = _normalize_seq(str(row[seq_col]))
                    if len(seq) < 20:
                        continue
                    labels = {}
                    for c in df.columns:
                        if c in skip_cols:
                            continue
                        v = _safe_float(row.get(c))
                        if v is not None:
                            labels[str(c)] = v
                    sid = str(row.get(id_col, idx)).strip() if id_col else str(idx)
                    rec = canonical_record_no_edit(
                        record_id=f"{accession}_s1_{sid}",
                        dataset="lepplek2022_persistseq",
                        accession=accession,
                        region="full_length",
                        sequence=seq,
                        labels=labels,
                        metadata={
                            "source_file": s1_path.name,
                            "data_role": "D_A",
                            "table": "Table_S1",
                            "sequence_id": sid,
                            "note": "PERSIST-seq pooled 233 full-length construct; observational with labels",
                        },
                    )
                    records.append(rec)
                    n_s1 += 1
                print(f"  extracted {n_s1} records from Table S1")
        except Exception as e:
            print(f"  WARNING: could not parse Table S1: {e}")

    if not records:
        print(f"  INCOMPLETE: no .rdat or Table S1 data found")
        return [{
            "record_id": f"{accession}_INCOMPLETE",
            "dataset": "lepplek2022_persistseq",
            "accession": accession,
            "region": "full_length",
            "source_sequence": None,
            "candidate_sequence": None,
            "edit_script": [],
            "edit_script_verified": True,
            "edit_distance": 0,
            "n_ins": 0, "n_del": 0, "n_sub": 0,
            "path_ambiguity": 1,
            "labels": {},
            "metadata": {
                "record_type": "incomplete",
                "data_role": "D_A",
                "note": "No .rdat or Table S1 data found",
            },
        }]

    print(f"  total extracted {len(records)} records")
    return records


def extract_encsr854ruf(data_root: Path) -> List[dict]:
    """ENCSR854RUF — 3'UTR, D_C (MPRAu processed table).

    15,266 paired 3'UTR variants with MPRA activity data across 6 cell types.
    Source xlsx has no sequences — reconstructed via
    reconstruct_encsr854ruf_sequences.py -> reconstructed_oligos.jsonl.
    """
    accession = "ENCSR854RUF"
    print(f"\n[{accession}] Extracting canonical records...")

    # Locate reconstructed_oligos.jsonl
    candidates = [
        data_root / accession / "reconstructed_oligos.jsonl",
    ]
    fpath = None
    for c in candidates:
        if c.exists():
            fpath = c
            break
    if fpath is None:
        print(f"  WARNING: reconstructed_oligos.jsonl not found in:")
        for c in candidates:
            print(f"    {c}")
        print(f"  Run reconstruct_encsr854ruf_sequences.py first.")
        return [{
            "record_id": f"{accession}_INCOMPLETE",
            "dataset": "encsr854ruf_mprau",
            "accession": accession,
            "region": "3'UTR",
            "source_sequence": None,
            "candidate_sequence": None,
            "edit_script": [],
            "edit_script_verified": True,
            "edit_distance": 0,
            "n_ins": 0, "n_del": 0, "n_sub": 0,
            "path_ambiguity": 1,
            "labels": {},
            "metadata": {
                "record_type": "incomplete",
                "data_role": "D_C",
                "note": "reconstructed_oligos.jsonl not found; run reconstruct_encsr854ruf_sequences.py",
            },
        }]

    print(f"  reading: {fpath}")
    records = []
    skipped_no_seq = 0
    skipped_identical = 0
    max_records = MAX_RECORDS_PER_DATASET
    seen_ids = set()

    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if max_records and len(records) >= max_records:
                print(f"  capping at {max_records} records")
                break
            rec = json.loads(line)
            source = _normalize_seq(rec.get("source_sequence", ""))
            candidate = _normalize_seq(rec.get("candidate_sequence", ""))
            if not source or not candidate:
                skipped_no_seq += 1
                continue

            # Truncate to window around variant for long sequences (oligos
            # are ~100bp so this rarely triggers, but keeps alignment feasible).
            MAX_UTR_LEN = 500
            if len(source) > MAX_UTR_LEN or len(candidate) > MAX_UTR_LEN:
                raw_meta_tmp = rec.get("metadata", {})
                var_pos = raw_meta_tmp.get("variant_position")
                if var_pos is not None:
                    half = MAX_UTR_LEN // 2
                    pos_0idx = max(0, int(var_pos))
                    src_start = max(0, pos_0idx - half)
                    src_end = min(len(source), pos_0idx + half)
                    delta = len(candidate) - len(source)
                    source = source[src_start:src_end]
                    candidate = candidate[src_start:src_end + delta]

            if source == candidate:
                skipped_identical += 1
                continue

            # Build labels
            raw_labels = rec.get("labels", {})
            labels = {}
            for k, v in raw_labels.items():
                fv = _safe_float(v)
                if fv is not None:
                    labels[k] = fv

            # Build metadata
            raw_meta = rec.get("metadata", {})
            metadata = {
                "data_role": "D_C",
                "gene_symbol": rec.get("gene_symbol", raw_meta.get("gene_symbol", "")),
                "variant_type": rec.get("variant_type", ""),
            }
            metadata.update(raw_meta)

            # Ensure unique record_id
            rid = rec.get("record_id", "")
            if not rid:
                rid = f"{accession}_{len(records)}"
            if rid in seen_ids:
                rid = f"{rid}_{len(records)}"
            seen_ids.add(rid)

            crec = canonical_record(
                record_id=rid,
                dataset="encsr854ruf_mprau",
                accession=accession,
                region=rec.get("region", "3'UTR"),
                source=source,
                candidate=candidate,
                labels=labels,
                metadata=metadata,
            )
            records.append(crec)

    print(f"  extracted {len(records)} records "
          f"(skipped: {skipped_no_seq} no_seq, {skipped_identical} identical)")
    return records


def extract_gse246381(data_root: Path) -> List[dict]:
    """GSE246381 — 5'UTR, D_C (paired REF/ALT variants).

    CSV with variant annotation in SeqID format. Actual UTR sequences
    reconstructed via reconstruct_gse246381_sequences.py -> reconstructed_utrs.jsonl.

    Variant annotation format: Variant;chr3:123954485;CA|C;Family=...;ENST...;REF;TTAAGCTTCA
    """
    accession = "GSE246381"
    print(f"\n[{accession}] Extracting canonical records...")

    # Locate reconstructed_utrs.jsonl
    candidates = [
        data_root / accession / "reconstructed_utrs.jsonl",
    ]
    fpath = None
    for c in candidates:
        if c.exists():
            fpath = c
            break
    if fpath is None:
        print(f"  WARNING: reconstructed_utrs.jsonl not found in:")
        for c in candidates:
            print(f"    {c}")
        print(f"  Run reconstruct_gse246381_sequences.py first.")
        return [{
            "record_id": f"{accession}_INCOMPLETE",
            "dataset": "gse246381",
            "accession": accession,
            "region": "5'UTR",
            "source_sequence": None,
            "candidate_sequence": None,
            "edit_script": [],
            "edit_script_verified": True,
            "edit_distance": 0,
            "n_ins": 0, "n_del": 0, "n_sub": 0,
            "path_ambiguity": 1,
            "labels": {},
            "metadata": {
                "record_type": "incomplete",
                "data_role": "D_C",
                "note": "reconstructed_utrs.jsonl not found; run reconstruct_gse246381_sequences.py",
            },
        }]

    print(f"  reading: {fpath}")
    records = []
    skipped_no_seq = 0
    skipped_identical = 0
    max_records = MAX_RECORDS_PER_DATASET
    seen_ids = set()

    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if max_records and len(records) >= max_records:
                print(f"  capping at {max_records} records")
                break
            rec = json.loads(line)
            source = _normalize_seq(rec.get("source_sequence", ""))
            candidate = _normalize_seq(rec.get("candidate_sequence", ""))
            if not source or not candidate:
                skipped_no_seq += 1
                continue

            # Truncate to window around variant for long UTRs.
            MAX_UTR_LEN = 500
            if len(source) > MAX_UTR_LEN or len(candidate) > MAX_UTR_LEN:
                raw_meta_tmp = rec.get("metadata", {})
                var_pos = raw_meta_tmp.get("variant_position")
                if var_pos is not None:
                    half = MAX_UTR_LEN // 2
                    pos_0idx = max(0, int(var_pos))
                    src_start = max(0, pos_0idx - half)
                    src_end = min(len(source), pos_0idx + half)
                    delta = len(candidate) - len(source)
                    source = source[src_start:src_end]
                    candidate = candidate[src_start:src_end + delta]

            if source == candidate:
                skipped_identical += 1
                continue

            # Build labels
            raw_labels = rec.get("labels", {})
            labels = {}
            for k, v in raw_labels.items():
                fv = _safe_float(v)
                if fv is not None:
                    labels[k] = fv

            # Build metadata
            raw_meta = rec.get("metadata", {})
            metadata = {
                "data_role": "D_C",
                "variant_type": rec.get("variant_type", ""),
                "enst": rec.get("enst", raw_meta.get("enst", "")),
            }
            metadata.update(raw_meta)

            # Ensure unique record_id
            rid = rec.get("record_id", "")
            if not rid:
                rid = f"{accession}_{len(records)}"
            if rid in seen_ids:
                rid = f"{rid}_{len(records)}"
            seen_ids.add(rid)

            crec = canonical_record(
                record_id=rid,
                dataset="gse246381",
                accession=accession,
                region=rec.get("region", "5'UTR"),
                source=source,
                candidate=candidate,
                labels=labels,
                metadata=metadata,
            )
            records.append(crec)

    print(f"  extracted {len(records)} records "
          f"(skipped: {skipped_no_seq} no_seq, {skipped_identical} identical)")
    return records


def extract_gse217518(data_root: Path) -> List[dict]:
    """GSE217518 — 3'UTR/5'UTR, D_C.

    Paired UTR variants (Ref->Mut) with RNA stability labels (HEK293T, SH-SY5Y).
    Source data has only HGVS c. notation + genomic coords (no sequences).
    Sequences reconstructed via NCBI E-utilities in
    reconstruct_gse217518_sequences.py -> reconstructed_variants.jsonl.
    """
    accession = "GSE217518"
    print(f"\n[{accession}] Extracting canonical records...")

    # Locate reconstructed_variants.jsonl
    candidates = [
        data_root / accession / "reconstructed_variants.jsonl",
        data_root.parent / "raw" / "gse217518_utr_stability" / "reconstructed_variants.jsonl",
    ]
    fpath = None
    for c in candidates:
        if c.exists():
            fpath = c
            break
    if fpath is None:
        print(f"  WARNING: reconstructed_variants.jsonl not found in:")
        for c in candidates:
            print(f"    {c}")
        print(f"  Run reconstruct_gse217518_sequences.py first.")
        return [{
            "record_id": f"{accession}_INCOMPLETE",
            "dataset": "gse217518",
            "accession": accession,
            "region": "3'UTR",
            "source_sequence": None,
            "candidate_sequence": None,
            "edit_script": [],
            "edit_script_verified": True,
            "edit_distance": 0,
            "n_ins": 0, "n_del": 0, "n_sub": 0,
            "path_ambiguity": 1,
            "labels": {},
            "metadata": {
                "record_type": "incomplete",
                "data_role": "D_C",
                "note": "reconstructed_variants.jsonl not found; run reconstruct_gse217518_sequences.py",
            },
        }]

    print(f"  reading: {fpath}")
    records = []
    skipped_no_seq = 0
    skipped_identical = 0
    max_records = MAX_RECORDS_PER_DATASET
    seen_ids = set()

    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if max_records and len(records) >= max_records:
                print(f"  capping at {max_records} records")
                break
            rec = json.loads(line)
            source = _normalize_seq(rec.get("source_sequence", ""))
            candidate = _normalize_seq(rec.get("candidate_sequence", ""))
            if not source or not candidate:
                skipped_no_seq += 1
                continue

            # Truncate to window around variant position for long UTRs.
            # The O(n*m) alignment in canonical_record() is infeasible for
            # sequences >~1000bp (3'UTRs can be 25k+ bp).  Use +/-250bp
            # around the variant site (matches typical MPRA oligo context).
            MAX_UTR_LEN = 500
            if len(source) > MAX_UTR_LEN or len(candidate) > MAX_UTR_LEN:
                raw_meta_tmp = rec.get("metadata", {})
                var_pos = raw_meta_tmp.get("variant_position")
                pos_type = raw_meta_tmp.get("variant_position_type", "")
                if var_pos is not None:
                    half = MAX_UTR_LEN // 2
                    # Convert HGVS position to UTR-local 0-indexed position.
                    # 5'UTR: c.-N → utr_pos = len(utr) - N  (N bases before CDS)
                    # 3'UTR: c.*N → utr_pos = N - 1        (N bases after CDS)
                    if pos_type == "5utr":
                        pos_0idx = len(source) - int(var_pos)
                    else:
                        pos_0idx = max(0, int(var_pos) - 1)
                    pos_0idx = max(0, pos_0idx)
                    src_start = max(0, pos_0idx - half)
                    src_end = min(len(source), pos_0idx + half)
                    delta = len(candidate) - len(source)
                    cand_start = src_start
                    cand_end = min(len(candidate), src_end + delta)
                    source = source[src_start:src_end]
                    candidate = candidate[cand_start:cand_end]
                else:
                    # No position info — take first MAX_UTR_LEN bp
                    source = source[:MAX_UTR_LEN]
                    candidate = candidate[:MAX_UTR_LEN]

            if source == candidate:
                skipped_identical += 1
                continue

            # Build labels (only include non-None float values)
            raw_labels = rec.get("labels", {})
            labels = {}
            for k, v in raw_labels.items():
                fv = _safe_float(v)
                if fv is not None:
                    labels[k] = fv

            # Build metadata: role info + all reconstruction metadata
            raw_meta = rec.get("metadata", {})
            metadata = {
                "data_role": "D_C",
                "gene_symbol": rec.get("gene_symbol", ""),
                "variant_name": rec.get("variant_name", ""),
                "variant_type": rec.get("variant_type", ""),
            }
            metadata.update(raw_meta)
            if "variant_position" in raw_meta:
                metadata["utr_window"] = MAX_UTR_LEN

            # Ensure unique record_id
            rid = rec.get("record_id", "")
            if not rid:
                rid = f"{accession}_{len(records)}"
            if rid in seen_ids:
                rid = f"{rid}_{len(records)}"
            seen_ids.add(rid)

            crec = canonical_record(
                record_id=rid,
                dataset="gse217518",
                accession=accession,
                region=rec.get("region", "3'UTR"),
                source=source,
                candidate=candidate,
                labels=labels,
                metadata=metadata,
            )
            records.append(crec)

    print(f"  extracted {len(records)} records "
          f"(skipped: {skipped_no_seq} no_seq, {skipped_identical} identical)")
    if records:
        regions = {}
        for r in records:
            regions[r["region"]] = regions.get(r["region"], 0) + 1
        print(f"  by region: {regions}")
    return records


# ---------------------------------------------------------------------------
# GSE149487 helpers (Lim et al. 2021, Nat Commun — 5'UTR MPRA)
# ---------------------------------------------------------------------------

def _parse_6c_description(desc: str) -> Optional[dict]:
    """Parse a Lim 6c ``description`` string into structured fields.

    Recognized formats:
      SNV:       ``{Gene}_{Ref}_{Alt}_{chr}_{start}_{end}[_potential]``
      WT:        ``{Gene}_WT_{chr}_{start}_{end}``
      haplotype: ``{Gene}_{ref|alt}_{chr}_{start}_{end}``
      NA:        ``{Gene}_NA_{chr}_{start}_{end}``  (no variant, UTR control)

    Returns dict with ``type`` key (``snv``/``wt``/``haplotype``/``na``) plus
    fields, or ``None`` if the description does not match any known format.
    """
    d = str(desc).strip()
    # SNV (with optional _potential suffix)
    m = re.match(
        r"^([^_]+)_([ACGT])_([ACGT])_(chr[^_]+)_(\d+)_(\d+)(?:_(potential))?$", d
    )
    if m:
        return {
            "type": "snv",
            "gene": m.group(1),
            "ref": m.group(2),
            "alt": m.group(3),
            "chr": m.group(4),
            "start": int(m.group(5)),
            "end": int(m.group(6)),
            "suffix": m.group(7),
        }
    # WT
    m = re.match(r"^([^_]+)_WT_(chr[^_]+)_(\d+)_(\d+)$", d)
    if m:
        return {
            "type": "wt",
            "gene": m.group(1),
            "chr": m.group(2),
            "start": int(m.group(3)),
            "end": int(m.group(4)),
        }
    # haplotype (ref/alt)
    m = re.match(r"^([^_]+)_(ref|alt)_(chr[^_]+)_(\d+)_(\d+)$", d)
    if m:
        return {
            "type": "haplotype",
            "gene": m.group(1),
            "haplo": m.group(2),
            "chr": m.group(3),
            "start": int(m.group(4)),
            "end": int(m.group(5)),
        }
    # NA (no variant)
    m = re.match(r"^([^_]+)_NA_(chr[^_]+)_(\d+)_(\d+)$", d)
    if m:
        return {
            "type": "na",
            "gene": m.group(1),
            "chr": m.group(2),
            "start": int(m.group(3)),
            "end": int(m.group(4)),
        }
    return None


def _parse_6a_coordinate(coord: str) -> Optional[dict]:
    """Parse a MOESM8 6a ``5' UTR genomic coordinate`` string.

    Recognized formats:
      Mutant: ``{Gene}_{chr}_{pos}_{Ref}_{Alt}_UTR5``
      WT:     ``{Gene}_{chr}_{pos}_WT_UTR5``

    Returns dict with ``type`` (``mut``/``wt``) and parsed fields, or ``None``.
    """
    c = str(coord).strip()
    m = re.match(
        r"^([^_]+)_(chr[^_]+)_(\d+)_([ACGT])_([ACGT])_UTR5$", c
    )
    if m:
        return {
            "type": "mut",
            "gene": m.group(1),
            "chr": m.group(2),
            "pos": int(m.group(3)),
            "ref": m.group(4),
            "alt": m.group(5),
        }
    m = re.match(r"^([^_]+)_(chr[^_]+)_(\d+)_WT_UTR5$", c)
    if m:
        return {
            "type": "wt",
            "gene": m.group(1),
            "chr": m.group(2),
            "pos": int(m.group(3)),
        }
    return None


def _aggregate_6c_cpm(df_6c: pd.DataFrame) -> Dict[str, dict]:
    """Aggregate per-barcode CPMs in the Lim 6c table to per-description means.

    The 6c table has a "wide" layout: the left half holds
    ``description, barcode, TotalRNA_rep1, DNA_rep1, ... DNA_rep3`` and the
    right half repeats ``description.1, barcode.1, TotalRNA_rep1.1, ...`` then
    adds ``polysome_rep1/2/3``. The ``.1`` columns are duplicates of the left
    side and are skipped.

    Returns ``{description: {cpm_col: mean_value, ...}}``.
    """
    cpm_cols = []
    for c in df_6c.columns:
        cl = str(c).lower()
        if cl in ("description", "barcode") or "unnamed" in cl:
            continue
        # Skip duplicate columns from the right half (barcode.1, *.1)
        if str(c).endswith(".1"):
            continue
        cpm_cols.append(c)
    grouped = df_6c.groupby("description")[cpm_cols].mean()
    return {desc: row.to_dict() for desc, row in grouped.iterrows()}


def _find_utr_seq_6a(
    lookup: Dict[Tuple[str, str], List[dict]],
    gene: str,
    chrom: str,
    start: int,
    end: int,
    ref: Optional[str] = None,
    alt: Optional[str] = None,
    want_wt: bool = False,
) -> Optional[str]:
    """Find a UTR sequence in the 6a lookup by (gene, chr, pos in [start, end]).

    Args:
        lookup: ``{(gene, chr): [{'type':'mut'/'wt', 'pos':int, ...}, ...]}``
        gene, chrom: gene symbol and chromosome
        start, end: UTR range (inclusive)
        ref, alt: required alleles for mutant lookup (ignored if ``want_wt``)
        want_wt: if True, look for a WT entry

    Returns the UTR sequence string, or ``None`` if no match. When multiple
    entries match, prefers the one whose ``pos`` is closest to the mid-range.
    """
    entries = lookup.get((gene, chrom))
    if not entries:
        return None
    matches = []
    for e in entries:
        if not (start <= e["pos"] <= end):
            continue
        if want_wt:
            if e["type"] == "wt":
                matches.append(e)
        else:
            if (
                e["type"] == "mut"
                and e.get("ref") == ref
                and e.get("alt") == alt
            ):
                matches.append(e)
    if not matches:
        return None
    mid = (start + end) // 2
    matches.sort(key=lambda e: abs(e["pos"] - mid))
    return matches[0]["seq"]


def extract_gse149487(data_root: Path) -> List[dict]:
    """GSE149487 — Lim et al. 2021 5'UTR MPRA, paired D_C (WT -> SNV mutant).

    Data sources (all on server under ``data/p0/GSE149487/``):
      - ``41467_2021_24445_MOESM8_ESM.xlsx``
        * sheet ``6a 5' UTR sequences``: WT + mutant UTR sequences (919 rows)
        * sheet ``6d transcript FDR<0.1``: 545 significant transcript pairs
        * sheet ``6e TE FDR<0.1``: 545 significant TE pairs
      - ``Lim_et_al_Supp_Tbl_6c_293T.xlsx``: per-barcode CPMs with
        ``description`` (179,791 rows, ~236 barcodes per UTR variant).

    For each SNV description in 6c that has a matching mutant sequence and a
    matching WT sequence in 6a (pos within [start, end]):
      - source_sequence = WT UTR
      - candidate_sequence = mutant UTR
      - edit_script = compute_edit_script(WT, mutant)
      - labels = per-description mean CPMs (mutant + WT, when available) +
        significance labels from 6d (transcript) and 6e (TE).

    SNV descriptions without a WT sequence become observational records
    (mutant sequence only). WT descriptions in 6c become observational
    records (WT control UTR with CPM labels).
    """
    accession = "GSE149487"
    print(f"\n[{accession}] Extracting paired D_C 5'UTR records (Lim 2021 MPRA)...")
    d = data_root / accession
    if not d.exists():
        print(f"  WARNING: {d} does not exist")
        return []

    moesm8_path = _find_file(data_root, accession, [
        "*MOESM8*",
        "*41467_2021_24445_MOESM8*",
    ])
    lim6c_path = _find_file(data_root, accession, [
        "*Lim_et_al_Supp_Tbl_6c*",
        "*Lim*6c*",
    ])

    if moesm8_path is None or lim6c_path is None:
        print(
            f"  INCOMPLETE: missing MOESM8 ({moesm8_path is None}) "
            f"or Lim 6c ({lim6c_path is None})"
        )
        return [{
            "record_id": f"{accession}_INCOMPLETE",
            "dataset": "lim2021_5utr_mpra",
            "accession": accession,
            "region": "5'UTR",
            "source_sequence": None,
            "candidate_sequence": None,
            "edit_script": [],
            "edit_script_verified": True,
            "edit_distance": 0,
            "n_ins": 0, "n_del": 0, "n_sub": 0,
            "path_ambiguity": 1,
            "labels": {},
            "metadata": {
                "record_type": "incomplete",
                "data_role": "D_C",
                "note": (
                    f"Missing MOESM8 ({moesm8_path is None}) "
                    f"or Lim 6c ({lim6c_path is None})"
                ),
            },
        }]

    # --- Load 6a (UTR sequences) and build lookup ---
    df_6a = pd.read_excel(moesm8_path, sheet_name="6a 5' UTR sequences")
    print(f"  6a: {len(df_6a)} UTR sequence rows")
    lookup_6a: Dict[Tuple[str, str], List[dict]] = {}
    n_unparsed_6a = 0
    for _, row in df_6a.iterrows():
        coord = _parse_6a_coordinate(row["5' UTR genomic coordinate"])
        if coord is None:
            n_unparsed_6a += 1
            continue
        seq = _normalize_seq(row["sequence of 5' UTR"])
        if not seq:
            continue
        entry = {
            "type": coord["type"],
            "pos": coord["pos"],
            "seq": seq,
        }
        if coord["type"] == "mut":
            entry["ref"] = coord["ref"]
            entry["alt"] = coord["alt"]
        lookup_6a.setdefault((coord["gene"], coord["chr"]), []).append(entry)
    print(
        f"  6a lookup: {len(lookup_6a)} (gene, chr) keys, "
        f"{n_unparsed_6a} unparsed rows"
    )

    # --- Load 6c (per-barcode CPMs) and aggregate per description ---
    df_6c = pd.read_excel(lim6c_path, sheet_name=0)
    print(
        f"  6c: {len(df_6c)} barcode rows, "
        f"{df_6c['description'].nunique()} unique descriptions"
    )
    cpm_by_desc = _aggregate_6c_cpm(df_6c)
    print(f"  6c aggregated: {len(cpm_by_desc)} descriptions with mean CPMs")

    # --- Load 6d (transcript FDR<0.1) and 6e (TE FDR<0.1) for significance ---
    df_6d = pd.read_excel(moesm8_path, sheet_name="6d transcript FDR<0.1")
    df_6e = pd.read_excel(moesm8_path, sheet_name="6e TE FDR<0.1")
    sig_6d: Dict[str, dict] = {}
    for _, row in df_6d.iterrows():
        sig_6d[str(row["Mutant"])] = {
            "transcript_p_value": _safe_float(row["p value"]),
            "transcript_log_fold_change": _safe_float(row["log fold change"]),
            "transcript_padj_fdr": _safe_float(row["padj fdr"]),
        }
    sig_6e: Dict[str, dict] = {}
    for _, row in df_6e.iterrows():
        sig_6e[str(row["Mutant"])] = {
            "te_p_value": _safe_float(row["p value"]),
            "te_log_fold_change": _safe_float(row["log fold change"]),
            "te_padj_fdr": _safe_float(row["padj fdr"]),
        }
    print(
        f"  6d: {len(sig_6d)} transcript-significant, "
        f"6e: {len(sig_6e)} TE-significant variants"
    )

    # --- Iterate unique descriptions in 6c ---
    records: List[dict] = []
    max_records = MAX_RECORDS_PER_DATASET
    n_paired = 0
    n_obs_mut = 0
    n_obs_wt = 0
    n_no_mut_seq = 0
    n_no_wt_seq = 0

    unique_descs = sorted(df_6c["description"].dropna().unique())
    for desc in unique_descs:
        if max_records and len(records) >= max_records:
            print(f"  capping at {max_records} records")
            break
        parsed = _parse_6c_description(desc)
        if parsed is None:
            continue
        cpm = cpm_by_desc.get(desc, {})

        if parsed["type"] == "snv":
            mut_seq = _find_utr_seq_6a(
                lookup_6a,
                parsed["gene"],
                parsed["chr"],
                parsed["start"],
                parsed["end"],
                ref=parsed["ref"],
                alt=parsed["alt"],
                want_wt=False,
            )
            if mut_seq is None:
                n_no_mut_seq += 1
                continue
            wt_seq = _find_utr_seq_6a(
                lookup_6a,
                parsed["gene"],
                parsed["chr"],
                parsed["start"],
                parsed["end"],
                want_wt=True,
            )
            # Build a safe variant key for record_id
            var_key = (
                f"{parsed['gene']}_{parsed['ref']}_{parsed['alt']}_"
                f"{parsed['chr']}_{parsed['start']}_{parsed['end']}"
            )
            if parsed.get("suffix"):
                var_key += f"_{parsed['suffix']}"

            if wt_seq is None:
                # Observational record: mutant seq only
                n_no_wt_seq += 1
                labels = {
                    f"mutant_{k}": v for k, v in cpm.items() if v is not None
                }
                if desc in sig_6d:
                    labels.update(sig_6d[desc])
                if desc in sig_6e:
                    labels.update(sig_6e[desc])
                rec = canonical_record_no_edit(
                    record_id=f"{accession}_mut_{var_key}",
                    dataset="lim2021_5utr_mpra",
                    accession=accession,
                    region="5'UTR",
                    sequence=mut_seq,
                    labels=labels,
                    metadata={
                        "source_file": (
                            f"{lim6c_path.name} + {moesm8_path.name}"
                        ),
                        "data_role": "D_C",
                        "variant_type": "snv",
                        "gene": parsed["gene"],
                        "chrom": parsed["chr"],
                        "pos_start": parsed["start"],
                        "pos_end": parsed["end"],
                        "ref": parsed["ref"],
                        "alt": parsed["alt"],
                        "suffix": parsed.get("suffix"),
                        "note": (
                            "Mutant UTR; WT seq not found in 6a, "
                            "observational with CPM labels"
                        ),
                    },
                )
                records.append(rec)
                n_obs_mut += 1
                continue

            # Paired D_C record: WT -> mutant
            labels: dict = {}
            for k, v in cpm.items():
                if v is not None:
                    labels[f"mutant_{k}"] = v
            # Look up WT description's CPMs in 6c
            wt_desc = (
                f"{parsed['gene']}_WT_{parsed['chr']}_"
                f"{parsed['start']}_{parsed['end']}"
            )
            wt_cpm = cpm_by_desc.get(wt_desc, {})
            for k, v in wt_cpm.items():
                if v is not None:
                    labels[f"wt_{k}"] = v
            if desc in sig_6d:
                labels.update(sig_6d[desc])
            if desc in sig_6e:
                labels.update(sig_6e[desc])

            rec = canonical_record(
                record_id=f"{accession}_snv_{var_key}",
                dataset="lim2021_5utr_mpra",
                accession=accession,
                region="5'UTR",
                source=wt_seq,
                candidate=mut_seq,
                labels=labels,
                metadata={
                    "source_file": (
                        f"{lim6c_path.name} + {moesm8_path.name}"
                    ),
                    "data_role": "D_C",
                    "variant_type": "snv",
                    "gene": parsed["gene"],
                    "chrom": parsed["chr"],
                    "pos_start": parsed["start"],
                    "pos_end": parsed["end"],
                    "ref": parsed["ref"],
                    "alt": parsed["alt"],
                    "suffix": parsed.get("suffix"),
                    "note": (
                        "Paired WT->mutant 5'UTR; "
                        "labels are per-barcode mean CPMs + 6d/6e significance"
                    ),
                },
            )
            records.append(rec)
            n_paired += 1

        elif parsed["type"] == "wt":
            # Observational record for WT UTR control
            wt_seq = _find_utr_seq_6a(
                lookup_6a,
                parsed["gene"],
                parsed["chr"],
                parsed["start"],
                parsed["end"],
                want_wt=True,
            )
            if wt_seq is None:
                n_no_wt_seq += 1
                continue
            labels = {k: v for k, v in cpm.items() if v is not None}
            rec = canonical_record_no_edit(
                record_id=(
                    f"{accession}_wt_{parsed['gene']}_"
                    f"{parsed['chr']}_{parsed['start']}_{parsed['end']}"
                ),
                dataset="lim2021_5utr_mpra",
                accession=accession,
                region="5'UTR",
                sequence=wt_seq,
                labels=labels,
                metadata={
                    "source_file": (
                        f"{lim6c_path.name} + {moesm8_path.name}"
                    ),
                    "data_role": "D_C",
                    "variant_type": "wt_control",
                    "gene": parsed["gene"],
                    "chrom": parsed["chr"],
                    "pos_start": parsed["start"],
                    "pos_end": parsed["end"],
                    "note": "WT UTR control; observational with CPM labels",
                },
            )
            records.append(rec)
            n_obs_wt += 1

    print(
        f"  extracted: {n_paired} paired D_C, "
        f"{n_obs_mut} mutant-only obs, {n_obs_wt} WT-control obs; "
        f"skipped: {n_no_mut_seq} no mutant seq, {n_no_wt_seq} no WT seq"
    )

    if not records:
        return [{
            "record_id": f"{accession}_INCOMPLETE",
            "dataset": "lim2021_5utr_mpra",
            "accession": accession,
            "region": "5'UTR",
            "source_sequence": None,
            "candidate_sequence": None,
            "edit_script": [],
            "edit_script_verified": True,
            "edit_distance": 0,
            "n_ins": 0, "n_del": 0, "n_sub": 0,
            "path_ambiguity": 1,
            "labels": {},
            "metadata": {
                "record_type": "incomplete",
                "data_role": "D_C",
                "note": "No records extracted (data files present but no matches)",
            },
        }]

    print(f"  total extracted {len(records)} records")
    return records


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

EXTRACTORS = {
    "GSE114002": extract_gse114002,
    "GSE200304": extract_gse200304,
    "GSE145046": extract_gse145046,
    "GSE207584": extract_gse207584,
    "GSE173083": extract_gse173083,
    "ENCSR854RUF": extract_encsr854ruf,
    "GSE246381": extract_gse246381,
    "GSE217518": extract_gse217518,
    "GSE149487": extract_gse149487,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global MAX_RECORDS_PER_DATASET
    parser = argparse.ArgumentParser(description="D1-01: Build canonical records")
    parser.add_argument("--data-root", default="data/p0",
                        help="Root directory for p0 datasets")
    parser.add_argument("--output", default="data/d1_canonical_records.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--dataset", default=None,
                        help="Process only this dataset (default: all)")
    parser.add_argument("--max-records", type=int, default=0,
                        help="Max records per dataset (0 = no cap, default: 0)")
    args = parser.parse_args()

    MAX_RECORDS_PER_DATASET = args.max_records
    data_root = Path(args.data_root)
    print(f"D1-01: Building canonical records")
    print(f"  data_root: {data_root}")
    print(f"  output: {args.output}")
    print(f"  max_records_per_dataset: {MAX_RECORDS_PER_DATASET}")

    all_records = []
    datasets_to_process = [args.dataset] if args.dataset else list(EXTRACTORS.keys())

    for ds in datasets_to_process:
        if ds not in EXTRACTORS:
            print(f"  WARNING: unknown dataset {ds}")
            continue
        try:
            records = EXTRACTORS[ds](data_root)
            all_records.extend(records)
        except Exception as e:
            print(f"  ERROR extracting {ds}: {e}")
            import traceback
            traceback.print_exc()

    # Final verification: apply(edit_script, source) == candidate for all
    # records that have edit scripts
    print(f"\n--- Final D1-01 verification ---")
    n_verified = 0
    n_failed = 0
    n_observational = 0
    n_incomplete = 0
    for rec in all_records:
        meta = rec.get("metadata", {})
        if meta.get("record_type") in ("observational", "incomplete"):
            if meta.get("record_type") == "incomplete":
                n_incomplete += 1
            else:
                n_observational += 1
            continue
        ops_list = rec.get("edit_script", [])
        source = rec.get("source_sequence")
        candidate = rec.get("candidate_sequence")
        if source is None or candidate is None:
            n_observational += 1
            continue
        # Reconstruct EditOps and verify
        from edit_script_core import EditOp
        ops = [EditOp.from_dict(o) for o in ops_list]
        if apply_edit_script(source, ops) == candidate:
            rec["edit_script_verified"] = True
            n_verified += 1
        else:
            rec["edit_script_verified"] = False
            n_failed += 1
            print(f"  FAILED: {rec['record_id']}")

    print(f"  verified edit-script records: {n_verified}")
    print(f"  failed edit-script records:   {n_failed}")
    print(f"  observational records:        {n_observational}")
    print(f"  incomplete records:           {n_incomplete}")
    print(f"  total records:                {len(all_records)}")

    if n_failed > 0:
        print(f"\n  *** D1-01 ACCEPTANCE FAILED: {n_failed} records failed verification ***")
        sys.exit(1)

    _write_records(all_records, Path(args.output))
    print(f"\nD1-01 complete: {n_verified} verified, 0 failed, "
          f"{n_observational} observational, {n_incomplete} incomplete")


if __name__ == "__main__":
    main()
