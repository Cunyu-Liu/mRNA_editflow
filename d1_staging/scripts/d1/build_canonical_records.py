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


# Global cap on records per dataset (prevents hours-long runs on 280k+ rows)
MAX_RECORDS_PER_DATASET = 20000


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
        if len(records) >= max_records:
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
        # Cap to avoid huge output
        if len(records) >= 50000:
            print(f"  capping at 50000 records (total rows: {len(df)})")
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


def extract_gse173083(data_root: Path) -> List[dict]:
    """GSE173083 — full-length mRNA, D_A (observational).

    PERSIST-seq. Downloaded GEO files are .rdat (RNA structure) only.
    Table S1 (with sequences + labels) is in the supplementary materials,
    not in the GEO download. Mark as incomplete.
    """
    accession = "GSE173083"
    print(f"\n[{accession}] Extracting observational records (D_A, full-length)...")
    # Check for xlsx/csv (Table S1) — not in GEO download
    fpath = _find_file(data_root, accession, [
        "*.xlsx", "*.xls", "*.csv*",
    ])
    if fpath is None:
        print(f"  INCOMPLETE: Only .rdat files in GEO download; Table S1 not available.")
        print(f"  GSE173083 requires supplementary Table S1 for sequence + label extraction.")
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
                "note": "Only .rdat files in GEO download; Table S1 with sequences not available",
            },
        }]

    # If xlsx/csv found, try to parse
    try:
        if fpath.suffix in (".xlsx", ".xls"):
            df = pd.read_excel(fpath)
        else:
            df = pd.read_csv(fpath, compression="infer")
    except Exception as e:
        print(f"  WARNING: could not parse {fpath.name}: {e}")
        return []

    print(f"  loaded {len(df)} rows, columns: {list(df.columns)}")
    seq_col = None
    for c in df.columns:
        cl = c.strip().lower()
        if any(p in cl for p in ["rna sequence", "sequence", "seq"]):
            seq_col = c
            break
    if seq_col is None:
        print(f"  WARNING: no sequence column found")
        return []

    records = []
    for idx, row in df.iterrows():
        seq = _normalize_seq(str(row[seq_col]))
        if len(seq) < 20:
            continue
        labels = {}
        for c in df.columns:
            if c == seq_col:
                continue
            v = _safe_float(row.get(c))
            if v is not None:
                labels[c] = v
        rec = canonical_record_no_edit(
            record_id=f"{accession}_{idx}",
            dataset="lepplek2022_persistseq",
            accession=accession,
            region="full_length",
            sequence=seq,
            labels=labels,
            metadata={"source_file": fpath.name, "data_role": "D_A",
                      "note": "full-length out-of-scope for v2; observational only"},
        )
        records.append(rec)

    print(f"  extracted {len(records)} records")
    return records


def extract_encsr854ruf(data_root: Path) -> List[dict]:
    """ENCSR854RUF — 3'UTR, D_C (MPRAu processed table).

    Multi-sheet xlsx. The "Variant MPRAu Results" sheet has variant activity
    data but NO actual UTR sequences. The "Oligo Variant Info" sheet has
    alt/ref tags. Without sequences, edit scripts cannot be computed.
    Mark as incomplete.
    """
    accession = "ENCSR854RUF"
    print(f"\n[{accession}] Extracting canonical records...")
    fpath = _find_file(data_root, accession, [
        "*.xlsx",
        "*.xls",
        "*.csv*",
        "*.tsv*",
    ])
    if fpath is None:
        print(f"  WARNING: No processed file found for {accession}")
        return []

    # Read the "Variant MPRAu Results" sheet for activity labels
    try:
        df = pd.read_excel(fpath, sheet_name="Variant MPRAu Results")
    except Exception as e:
        print(f"  WARNING: could not parse {fpath.name}: {e}")
        return []

    print(f"  loaded {len(df)} rows from 'Variant MPRAu Results' sheet")
    print(f"  columns: {list(df.columns)[:10]}...")

    # Check for sequence columns
    seq_col = None
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ("seq", "sequence", "utr", "3utr") or "sequence" in cl:
            seq_col = c
            break

    if seq_col is None:
        print(f"  INCOMPLETE: No UTR sequence column in MPRAu results.")
        print(f"  ENCSR854RUF has variant activity data but no sequences.")
        print(f"  Requires genome reconstruction for UTR sequences.")
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
                "note": "MPRAu xlsx has variant activity data but no UTR sequences; needs genome reconstruction",
                "source_file": fpath.name,
                "n_variants": len(df),
            },
        }]

    # If sequences found, proceed with full extraction
    print(f"  found sequence column: {seq_col}")
    records = []
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
            dataset="encsr854ruf_mprau",
            accession=accession,
            region="3'UTR",
            sequence=seq,
            labels=labels,
            metadata={"source_file": fpath.name},
        )
        records.append(rec)
    print(f"  extracted {len(records)} records")
    return records


def extract_gse246381(data_root: Path) -> List[dict]:
    """GSE246381 — 5'UTR, D_E (historically exposed, E4).

    CSV with variant annotation in SeqID format. Actual UTR sequences need
    genome reconstruction. For D1-01, mark as incomplete (no sequences to
    compute edit scripts).

    Variant annotation format: Variant;chr3:123954485;CA|C;Family=...;ENST...;REF;TTAAGCTTCA
    """
    accession = "GSE246381"
    print(f"\n[{accession}] D_E historically exposed — checking for sequence data...")
    fpath = _find_file(data_root, accession, [
        "*.csv*",
        "*.tsv*",
        "*.txt*",
    ])
    if fpath is None:
        print(f"  WARNING: No data file found for {accession}")
        return []

    # Check if actual UTR sequences are available
    try:
        df = pd.read_csv(fpath, compression="infer", nrows=5)
    except Exception:
        try:
            df = pd.read_csv(fpath, sep="\t", compression="infer", nrows=5)
        except Exception as e:
            print(f"  WARNING: could not parse {fpath.name}: {e}")
            return []

    print(f"  columns: {list(df.columns)}")
    # Look for actual UTR sequence column
    seq_col = None
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("utr", "seq", "sequence", "5utr", "five_prime_utr"):
            seq_col = c
            break

    if seq_col is None:
        print(f"  INCOMPLETE: No actual UTR sequence column found.")
        print(f"  GSE246381 requires genome reconstruction for UTR sequences.")
        print(f"  Skipping for D1-01 (variant annotations only).")
        # Create a metadata-only record noting the gap
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
                "data_role": "D_E",
                "evidence_grade": "E4",
                "historically_exposed": True,
                "note": "No UTR sequences in downloaded data; requires genome reconstruction",
                "source_file": fpath.name,
            },
        }]

    # If sequences are available, proceed with full extraction
    print(f"  found sequence column: {seq_col}")
    # ... (would implement full extraction if sequences available)
    # For now, mark as incomplete
    print(f"  NOTE: Full extraction not yet implemented for {accession} with sequences")
    return []


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
            if len(records) >= max_records:
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


def extract_gse149487(data_root: Path) -> List[dict]:
    """GSE149487 — 5'UTR, D_C.

    Barcode + count files only. No barcode-to-UTR mapping in downloaded data.
    """
    accession = "GSE149487"
    print(f"\n[{accession}] Checking data completeness...")
    fpath = _find_file(data_root, accession, [
        "*barcode*",
        "*count*",
        "*.csv*",
        "*.tsv*",
        "*.txt*",
    ])
    if fpath is None:
        print(f"  WARNING: No data file found for {accession}")
        return []

    print(f"  INCOMPLETE: GSE149487 has barcode+count files but no barcode-to-UTR mapping.")
    print(f"  Needs supplementary mapping. Skipping for D1-01.")
    return [{
        "record_id": f"{accession}_INCOMPLETE",
        "dataset": "gse149487",
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
            "note": "No barcode-to-UTR mapping in downloaded data; needs supplementary mapping",
            "source_file": fpath.name,
        },
    }]


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
    parser.add_argument("--max-records", type=int, default=20000,
                        help="Max records per dataset (default: 20000)")
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
