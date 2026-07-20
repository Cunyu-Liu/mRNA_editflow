#!/usr/bin/env python3
"""
Acquire Cao 2021 Nat Commun 5'UTR dataset.

Sources:
  - GitHub repo: https://github.com/zzz2010/5UTR_Optimizer
    * data/df_counts_and_len.TE_sorted.HEK_Andrev2015.with_annot.txt (14MB)
      - HEK293T Ribo-seq-derived TE for endogenous 5'UTRs with features (training data)
    * data/df_counts_and_len.TE_sorted.pc3.with_annot.txt (14MB)
      - PC3 (prostate cancer) TE for endogenous 5'UTRs with features
    * data/df_counts_and_len.TE_sorted.Muscle.with_annot.txt (1MB)
      - Muscle TE for endogenous 5'UTRs with features
    * data/final_endogenous.txt (1.8MB)
      - Final filtered endogenous 5'UTR set used for model training
    * data/gencode_v17_5utr_15bpcds.fa (16MB)
      - GENCODE v17 5'UTR sequences with 15bp CDS context
    * data/eva_5utrseq/unique.df.hek.withUTRfas.TE_sorted.txt.top1000.fasta (328KB)
      - Top 1000 high-TE 5'UTRs from HEK MPRA screen (validation positives)
    * data/eva_5utrseq/unique.df.hek.withUTRfas.TE_sorted.txt.bottom500.fasta (150KB)
      - Bottom 500 low-TE 5'UTRs from HEK MPRA screen (validation negatives)
    * data/eva_5utrseq/unique.df.pc3.withUTRfas.TE_sorted.txt.top1000.fasta (319KB)
      - Top 1000 high-TE 5'UTRs from PC3 MPRA screen
    * data/eva_5utrseq/unique.df.pc3.withUTRfas.TE_sorted.txt.bottom500.fasta (176KB)
      - Bottom 500 low-TE 5'UTRs from PC3 MPRA screen
    * data/eva_5utrseq/unique.df.muscle.withUTRfas.TE_sorted.fasta (323KB)
      - All muscle 5'UTRs from MPRA screen

  - Raw FASTQ (DEFERRED - not downloaded here): GSE176581 at NCBI GEO
    * Raw MPRA sequencing reads for the 12,000 designed 5'UTR library
    * Would require alignment + counting + normalization to derive per-UTR expression

  - Ribo-seq data (DEFERRED - not downloaded here, available at GEO):
    * GSE55195 (HEK293T) - used for TE calculation
    * GSE35469 (PC3)
    * GSE56148 (muscle)

Paper:
  Cao, Novoa, Zhang et al. "High-throughput 5' UTR engineering for enhanced protein
  production in non-viral gene therapies." Nat Commun 12, 4138 (2021).
  PMID: 34230498, DOI: 10.1038/s41467-021-24436-7

License:
  GitHub: MIT (see LICENSE file in repo)
  Article: CC BY 4.0 (Open Access)
"""

import csv
import gzip
import hashlib
import json
import os
import ssl
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- Configuration ---------------------------------------------------------

_REPO_ROOT = os.environ.get("PYTHONPATH", "").split(":")[0] or "/home/cunyuliu/mrna_editflow_goal"
DATA_ROOT = Path(_REPO_ROOT) / "mrna_editflow" / "data" / "raw" / "cao2021_5utr"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/zzz2010/5UTR_Optimizer/master/data"
GITHUB_API_BASE = "https://api.github.com/repos/zzz2010/5UTR_Optimizer/contents/data"
GEO_ACCESSION_RAW_MPRA = "GSE176581"
GEO_ACCESSIONS_RIBOSEQ = {
    "HEK293T": "GSE55195",
    "PC3": "GSE35469",
    "Muscle": "GSE56148",
}

CITATION = (
    "Cao J, Novoa EM, Zhang Z, Chen WCW, Liu D, Choi GCG, Wong ASL, Wehrspaun C, "
    "Kellis M, Lu TK. High-throughput 5' UTR engineering for enhanced protein "
    "production in non-viral gene therapies. Nat Commun. 2021;12(1):4138. "
    "PMID:34230498. DOI:10.1038/s41467-021-24436-7."
)
LICENSE_TEXT = (
    "GitHub code/data: MIT License (https://github.com/zzz2010/5UTR_Optimizer/blob/master/LICENSE). "
    "Article: CC BY 4.0 (Open Access). "
    "Raw FASTQ at NCBI GEO: NCBI public data terms. "
    "Cite: Cao et al. 2021 Nat Commun (PMID:34230498)."
)

# Files to download (path relative to data/, name in our data dir)
FILES_TO_DOWNLOAD: List[Dict[str, str]] = [
    {
        "remote_path": "df_counts_and_len.TE_sorted.HEK_Andrev2015.with_annot.txt",
        "local_name": "TE_sorted.HEK293T.Andrev2015.with_annot.txt",
        "kind": "ribo_seq_te_features",
        "cell_type": "HEK293T",
        "description": "HEK293T Ribo-seq-derived TE for endogenous 5'UTRs with extracted features (training data).",
    },
    {
        "remote_path": "df_counts_and_len.TE_sorted.pc3.with_annot.txt",
        "local_name": "TE_sorted.PC3.with_annot.txt",
        "kind": "ribo_seq_te_features",
        "cell_type": "PC3",
        "description": "PC3 (prostate cancer) Ribo-seq-derived TE for endogenous 5'UTRs with features.",
    },
    {
        "remote_path": "df_counts_and_len.TE_sorted.Muscle.with_annot.txt",
        "local_name": "TE_sorted.Muscle.with_annot.txt",
        "kind": "ribo_seq_te_features",
        "cell_type": "Muscle",
        "description": "Muscle Ribo-seq-derived TE for endogenous 5'UTRs with features.",
    },
    {
        "remote_path": "final_endogenous.txt",
        "local_name": "final_endogenous_5utr.txt",
        "kind": "filtered_endogenous_5utr",
        "cell_type": "multiple",
        "description": "Final filtered endogenous 5'UTR set used for model training.",
    },
    {
        "remote_path": "gencode_v17_5utr_15bpcds.fa",
        "local_name": "gencode_v17_5utr_15bpcds.fa",
        "kind": "utr_sequence_fasta",
        "cell_type": "multiple",
        "description": "GENCODE v17 5'UTR sequences with 15bp CDS context (start codon included).",
    },
    {
        "remote_path": "eva_5utrseq/unique.df.hek.withUTRfas.TE_sorted.txt.top1000.fasta",
        "local_name": "hek_top1000_high_TE.fasta",
        "kind": "mpra_screen_result",
        "cell_type": "HEK293T",
        "description": "Top 1000 high-TE 5'UTRs from HEK MPRA screen (validation positives).",
    },
    {
        "remote_path": "eva_5utrseq/unique.df.hek.withUTRfas.TE_sorted.txt.bottom500.fasta",
        "local_name": "hek_bottom500_low_TE.fasta",
        "kind": "mpra_screen_result",
        "cell_type": "HEK293T",
        "description": "Bottom 500 low-TE 5'UTRs from HEK MPRA screen (validation negatives).",
    },
    {
        "remote_path": "eva_5utrseq/unique.df.pc3.withUTRfas.TE_sorted.txt.top1000.fasta",
        "local_name": "pc3_top1000_high_TE.fasta",
        "kind": "mpra_screen_result",
        "cell_type": "PC3",
        "description": "Top 1000 high-TE 5'UTRs from PC3 MPRA screen.",
    },
    {
        "remote_path": "eva_5utrseq/unique.df.pc3.withUTRfas.TE_sorted.txt.bottom500.fasta",
        "local_name": "pc3_bottom500_low_TE.fasta",
        "kind": "mpra_screen_result",
        "cell_type": "PC3",
        "description": "Bottom 500 low-TE 5'UTRs from PC3 MPRA screen.",
    },
    {
        "remote_path": "eva_5utrseq/unique.df.muscle.withUTRfas.TE_sorted.fasta",
        "local_name": "muscle_all_5utr.fasta",
        "kind": "mpra_screen_result",
        "cell_type": "Muscle",
        "description": "All muscle 5'UTRs from MPRA screen (top + bottom).",
    },
]


# --- HTTP utilities --------------------------------------------------------

def _ssl_context() -> ssl.SSLContext:
    """Bypass SSL cert verification (NCBI/GitHub cert chains sometimes broken on server)."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _http_get(url: str, dest: Path, timeout: int = 120, retries: int = 3) -> None:
    """Download URL to dest with retries. Raises on final failure."""
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mrna_editflow/1.0"})
            with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status} for {url}")
                total = 0
                with open(dest, "wb") as f:
                    while True:
                        chunk = resp.read(64 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        total += len(chunk)
                if total == 0:
                    raise RuntimeError(f"Empty response for {url}")
                return
        except Exception as e:
            last_err = e
            print(f"  [retry {attempt}/{retries}] {dest.name}: {e}", flush=True)
            time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download {url} after {retries} retries: {last_err}")


def _sha256(path: Path, buf_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(buf_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _split_bucket(seq: str) -> str:
    """Deterministic 80/10/10 train/val/test split based on sequence hash."""
    h = hashlib.sha256(seq.upper().encode()).hexdigest()
    v = int(h[:8], 16) / 0xFFFFFFFF
    if v < 0.8:
        return "train"
    elif v < 0.9:
        return "val"
    return "test"


# --- Parsers ---------------------------------------------------------------

def parse_te_features_tsv(path: Path) -> Dict:
    """Parse df_counts_and_len.TE_sorted.*.with_annot.txt files.

    These contain Ribo-seq-derived TE for endogenous 5'UTRs with extracted features.
    Returns dict with header, row_count, split_stats, len_stats.
    """
    # Detect separator and headers
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
        if "\t" in first_line:
            delimiter = "\t"
        elif "," in first_line:
            delimiter = ","
        else:
            delimiter = None  # whitespace

    # Stream through file
    row_count = 0
    seq_lens: List[int] = []
    train_n = val_n = test_n = 0
    seq_col_idx: Optional[int] = None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=delimiter) if delimiter else csv.reader(f, skipinitialspace=True)
        header = next(reader, None)
        if header:
            # Look for sequence column
            for i, col in enumerate(header):
                lc = col.lower().strip()
                if lc in ("utr", "5utr", "seq", "sequence", "utr_sequence", "five_prime_utr"):
                    seq_col_idx = i
                    break
        for row in reader:
            if not row:
                continue
            row_count += 1
            if seq_col_idx is not None and seq_col_idx < len(row):
                seq = row[seq_col_idx].strip().upper()
                if seq:
                    seq_lens.append(len(seq))
                    bucket = _split_bucket(seq)
                    if bucket == "train":
                        train_n += 1
                    elif bucket == "val":
                        val_n += 1
                    else:
                        test_n += 1
            # Sample first 5 rows for schema peek
            if row_count <= 5 and "sample_rows" not in locals():
                pass
            if row_count == 1:
                sample_rows = [row]
            elif row_count <= 5:
                sample_rows.append(row)

    return {
        "header": header,
        "row_count": row_count,
        "split_stats": {"train": train_n, "val": val_n, "test": test_n, "unhashed": row_count - train_n - val_n - test_n},
        "len_stats": {
            "min": min(seq_lens) if seq_lens else None,
            "max": max(seq_lens) if seq_lens else None,
            "mean": round(sum(seq_lens) / len(seq_lens), 2) if seq_lens else None,
            "n_with_seq": len(seq_lens),
        },
        "seq_col_idx": seq_col_idx,
        "delimiter": delimiter or "whitespace",
    }


def parse_fasta(path: Path) -> Dict:
    """Parse a FASTA file, returning stats and split counts."""
    n_records = 0
    seq_lens: List[int] = []
    train_n = val_n = test_n = 0
    current_seq_parts: List[str] = []
    current_header: Optional[str] = None

    def flush():
        nonlocal n_records, train_n, val_n, test_n
        if current_header is None:
            return
        seq = "".join(current_seq_parts).upper()
        if not seq:
            return
        n_records += 1
        seq_lens.append(len(seq))
        bucket = _split_bucket(seq)
        if bucket == "train":
            train_n += 1
        elif bucket == "val":
            val_n += 1
        else:
            test_n += 1

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if line.startswith(">"):
                flush()
                current_header = line[1:]
                current_seq_parts = []
            else:
                current_seq_parts.append(line.strip())
    flush()

    return {
        "record_count": n_records,
        "split_stats": {"train": train_n, "val": val_n, "test": test_n},
        "len_stats": {
            "min": min(seq_lens) if seq_lens else None,
            "max": max(seq_lens) if seq_lens else None,
            "mean": round(sum(seq_lens) / len(seq_lens), 2) if seq_lens else None,
        },
    }


def parse_endogenous_txt(path: Path) -> Dict:
    """Parse final_endogenous.txt. Inspect format and count rows."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline().strip()
    # Try to detect format
    n_rows = 0
    n_cols = 0
    delimiter = "\t" if "\t" in first_line else ("," if "," in first_line else "whitespace")
    seq_col_idx: Optional[int] = None
    train_n = val_n = test_n = 0
    seq_lens: List[int] = []
    header = None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        if delimiter == "whitespace":
            reader = csv.reader(f, skipinitialspace=True)
        else:
            reader = csv.reader(f, delimiter=delimiter)
        header = next(reader, None)
        if header:
            n_cols = len(header)
            for i, col in enumerate(header):
                lc = col.lower().strip()
                if lc in ("utr", "5utr", "seq", "sequence", "utr_sequence"):
                    seq_col_idx = i
                    break
        for row in reader:
            if not row:
                continue
            n_rows += 1
            if seq_col_idx is not None and seq_col_idx < len(row):
                seq = row[seq_col_idx].strip().upper()
                if seq:
                    seq_lens.append(len(seq))
                    bucket = _split_bucket(seq)
                    if bucket == "train":
                        train_n += 1
                    elif bucket == "val":
                        val_n += 1
                    else:
                        test_n += 1

    return {
        "header": header,
        "n_cols": n_cols,
        "row_count": n_rows,
        "delimiter": delimiter,
        "seq_col_idx": seq_col_idx,
        "split_stats": {"train": train_n, "val": val_n, "test": test_n},
        "len_stats": {
            "min": min(seq_lens) if seq_lens else None,
            "max": max(seq_lens) if seq_lens else None,
            "mean": round(sum(seq_lens) / len(seq_lens), 2) if seq_lens else None,
            "n_with_seq": len(seq_lens),
        },
    }


# --- Main ------------------------------------------------------------------

def main() -> int:
    print(f"[Cao 2021 5'UTR data acquisition]", flush=True)
    print(f"  Target dir: {DATA_ROOT}", flush=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    manifest_entries: List[Dict] = []

    for spec in FILES_TO_DOWNLOAD:
        remote_path = spec["remote_path"]
        local_name = spec["local_name"]
        local_path = DATA_ROOT / local_name
        url = f"{GITHUB_RAW_BASE}/{remote_path}"

        print(f"\n[{local_name}]", flush=True)
        print(f"  url: {url}", flush=True)

        # Download (skip if already exists with content)
        if local_path.exists() and local_path.stat().st_size > 0:
            print(f"  [skip] exists, size={local_path.stat().st_size}", flush=True)
        else:
            try:
                _http_get(url, local_path, timeout=180, retries=3)
                print(f"  [done] downloaded, size={local_path.stat().st_size}", flush=True)
            except Exception as e:
                print(f"  [FAIL] {e}", flush=True)
                manifest_entries.append({
                    "filename": local_name,
                    "remote_path": remote_path,
                    "source_url": url,
                    "status": "failed",
                    "error": str(e),
                })
                continue

        # SHA-256
        sha = _sha256(local_path)
        print(f"  sha256: {sha}", flush=True)

        # Parse by kind
        kind = spec["kind"]
        stats: Dict = {}
        if kind == "ribo_seq_te_features":
            stats = parse_te_features_tsv(local_path)
        elif kind == "mpra_screen_result":
            stats = parse_fasta(local_path)
        elif kind == "filtered_endogenous_5utr":
            stats = parse_endogenous_txt(local_path)
        elif kind == "utr_sequence_fasta":
            stats = parse_fasta(local_path)
        stats["kind"] = kind
        stats["cell_type"] = spec["cell_type"]
        stats["description"] = spec["description"]
        print(f"  stats: {stats}", flush=True)

        manifest_entries.append({
            "filename": local_name,
            "remote_path": remote_path,
            "source_url": url,
            "sha256": sha,
            "size_bytes": local_path.stat().st_size,
            "citation": CITATION,
            "license": LICENSE_TEXT,
            "kind": kind,
            "cell_type": spec["cell_type"],
            "description": spec["description"],
            "stats": stats,
            "status": "ok",
        })

    # Write manifest
    manifest_path = DATA_ROOT / "manifest.json"
    manifest = {
        "dataset_name": "cao2021_5utr",
        "description": (
            "Cao, Novoa, Zhang et al. 2021 Nat Commun 5'UTR engineering dataset. "
            "Includes Ribo-seq-derived TE for endogenous 5'UTRs in 3 cell types "
            "(HEK293T, PC3, Muscle), filtered endogenous 5'UTR set, GENCODE v17 "
            "5'UTR FASTA, and top/bottom 5'UTRs from MPRA screens in each cell type."
        ),
        "citation": CITATION,
        "license": LICENSE_TEXT,
        "paper": {
            "title": "High-throughput 5' UTR engineering for enhanced protein production in non-viral gene therapies",
            "authors": "Cao J, Novoa EM, Zhang Z, Chen WCW, Liu D, Choi GCG, Wong ASL, Wehrspaun C, Kellis M, Lu TK",
            "journal": "Nature Communications",
            "year": 2021,
            "volume": "12",
            "pages": "4138",
            "pmid": "34230498",
            "doi": "10.1038/s41467-021-24436-7",
            "pmcid": "PMC8260622",
        },
        "source_repositories": {
            "github_code": "https://github.com/zzz2010/5UTR_Optimizer",
            "zenodo_release": "https://doi.org/10.5281/zenodo.4782661",
            "figshare_plasmids": "https://doi.org/10.6084/m9.figshare.14624472.v1",
            "geo_raw_mpra_fastq": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={GEO_ACCESSION_RAW_MPRA}",
            "geo_riboseq": {k: f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={v}" for k, v in GEO_ACCESSIONS_RIBOSEQ.items()},
        },
        "deferred_raw_data": {
            "GSE176581": "Raw MPRA FASTQ for 12,000 designed 5'UTR library. Not downloaded here; processed features are sufficient for TE predictor training.",
            "GSE55195": "HEK293T Ribo-seq FASTQ. Not downloaded; pre-computed TE features are in TE_sorted.HEK293T.Andrev2015.with_annot.txt.",
            "GSE35469": "PC3 Ribo-seq FASTQ. Not downloaded; pre-computed TE features are in TE_sorted.PC3.with_annot.txt.",
            "GSE56148": "Muscle Ribo-seq FASTQ. Not downloaded; pre-computed TE features are in TE_sorted.Muscle.with_annot.txt.",
        },
        "split_method": {
            "algorithm": "sha256(sequence.upper())[:8] -> int / 0xFFFFFFFF",
            "thresholds": {"train": "<0.8", "val": "<0.9", "test": ">=0.9"},
            "rationale": "Deterministic 80/10/10 split. Same sequence always maps to same split, enabling cross-dataset consistency.",
        },
        "files": manifest_entries,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n[manifest] {manifest_path}", flush=True)

    # Write summary
    summary_path = DATA_ROOT / "cao2021_summary.json"
    summary = {
        "dataset": "cao2021_5utr",
        "total_files": len(manifest_entries),
        "ok_files": sum(1 for e in manifest_entries if e.get("status") == "ok"),
        "failed_files": sum(1 for e in manifest_entries if e.get("status") == "failed"),
        "total_size_bytes": sum(e.get("size_bytes", 0) for e in manifest_entries),
        "by_kind": {},
        "by_cell_type": {},
    }
    for e in manifest_entries:
        k = e.get("kind", "unknown")
        c = e.get("cell_type", "unknown")
        summary["by_kind"].setdefault(k, {"count": 0, "size_bytes": 0})
        summary["by_kind"][k]["count"] += 1
        summary["by_kind"][k]["size_bytes"] += e.get("size_bytes", 0)
        summary["by_cell_type"].setdefault(c, {"count": 0, "size_bytes": 0})
        summary["by_cell_type"][c]["count"] += 1
        summary["by_cell_type"][c]["size_bytes"] += e.get("size_bytes", 0)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[summary]  {summary_path}", flush=True)
    print(f"\nDone. ok={summary['ok_files']}/{summary['total_files']}", flush=True)
    return 0 if summary["failed_files"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
