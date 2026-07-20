#!/usr/bin/env python3
"""P1-02A: Acquire Sample 2019 NBT MPRA dataset (GSE114002).

Downloads all 10 GSM processed CSV files from NCBI GEO, computes SHA-256,
parses each file to verify schema and counts, and writes a manifest with
source URL, SHA-256, license, counts, split stats, and cell/assay metadata.

Output layout:
    data/raw/sample2019_mpra/
        GSM3130435_egfp_unmod_1.csv.gz
        ...
        GSM4084997_varying_length_25to100.csv.gz
        manifest.json               <- per-file SHA-256, counts, metadata
        sample2019_summary.json     <- aggregated stats

Manifest fields (per data acquisition hard constraint):
    - source_url
    - sha256
    - license
    - record_count
    - split_stats (train/val/test counts after deterministic hashing)
    - cell_type, rna_chemistry, cds, replicate, library_kind
    - schema (column names)
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = Path(_REPO_ROOT) / "data" / "raw" / "sample2019_mpra"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/samples"

# (gsm, filename, library_kind, cds, rna_chemistry, replicate, expected_seq_len)
# expected_seq_len = None means variable / mixed lengths
SAMPLES: List[Dict] = [
    {
        "gsm": "GSM3130435",
        "filename": "GSM3130435_egfp_unmod_1.csv.gz",
        "library_kind": "random_50mer",
        "cds": "eGFP",
        "rna_chemistry": "unmodified",
        "replicate": 1,
        "expected_seq_len": 50,
    },
    {
        "gsm": "GSM3130436",
        "filename": "GSM3130436_egfp_unmod_2.csv.gz",
        "library_kind": "random_50mer",
        "cds": "eGFP",
        "rna_chemistry": "unmodified",
        "replicate": 2,
        "expected_seq_len": 50,
    },
    {
        "gsm": "GSM3130437",
        "filename": "GSM3130437_egfp_pseudo_1.csv.gz",
        "library_kind": "random_50mer",
        "cds": "eGFP",
        "rna_chemistry": "pseudouridine",
        "replicate": 1,
        "expected_seq_len": 50,
    },
    {
        "gsm": "GSM3130438",
        "filename": "GSM3130438_egfp_pseudo_2.csv.gz",
        "library_kind": "random_50mer",
        "cds": "eGFP",
        "rna_chemistry": "pseudouridine",
        "replicate": 2,
        "expected_seq_len": 50,
    },
    {
        "gsm": "GSM3130439",
        "filename": "GSM3130439_egfp_m1pseudo_1.csv.gz",
        "library_kind": "random_50mer",
        "cds": "eGFP",
        "rna_chemistry": "1-methylpseudouridine",
        "replicate": 1,
        "expected_seq_len": 50,
    },
    {
        "gsm": "GSM3130440",
        "filename": "GSM3130440_egfp_m1pseudo_2.csv.gz",
        "library_kind": "random_50mer",
        "cds": "eGFP",
        "rna_chemistry": "1-methylpseudouridine",
        "replicate": 2,
        "expected_seq_len": 50,
    },
    {
        "gsm": "GSM3130441",
        "filename": "GSM3130441_mcherry_1.csv.gz",
        "library_kind": "random_50mer",
        "cds": "mCherry",
        "rna_chemistry": "unmodified",
        "replicate": 1,
        "expected_seq_len": 50,
    },
    {
        "gsm": "GSM3130442",
        "filename": "GSM3130442_mcherry_2.csv.gz",
        "library_kind": "random_50mer",
        "cds": "mCherry",
        "rna_chemistry": "unmodified",
        "replicate": 2,
        "expected_seq_len": 50,
    },
    {
        "gsm": "GSM3130443",
        "filename": "GSM3130443_designed_library.csv.gz",
        "library_kind": "designed_human_snv",
        "cds": "eGFP",
        "rna_chemistry": "unmodified",
        "replicate": 1,
        "expected_seq_len": None,  # variable
    },
    {
        "gsm": "GSM4084997",
        "filename": "GSM4084997_varying_length_25to100.csv.gz",
        "library_kind": "random_variable_length",
        "cds": "eGFP",
        "rna_chemistry": "unmodified",
        "replicate": 1,
        "expected_seq_len": None,  # 25-100
    },
]

LICENSE = (
    "NCBI GEO public data. Submitted to GEO per NIH data sharing policy. "
    "Cite: Sample et al. 2019 Nat Biotechnol (PMID:31267113). "
    "Use permitted for research purposes; redistribution subject to GEO terms."
)
CITATION = "Sample PJ, Wang B, Seelig G. Nat Biotechnol. 2019;37(7):807-811. PMID:31267113. GSE114002."

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gsm_url(gsm: str, filename: str) -> str:
    """Build GEO FTP URL for a GSM supplementary file.

    GEO stores samples under samples/GSMxxxnnn/GSMxxxxxxx/suppl/.
    """
    # GSM3130435 -> GSM3130nnn
    prefix = gsm[:-3] + "nnn"
    return f"{GEO_BASE}/{prefix}/{gsm}/suppl/{filename}"


def _http_get(url: str, dest: Path, timeout: int = 120, retries: int = 3) -> None:
    """Download a URL to dest with retries."""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mrna_editflow/1.0"})
            import ssl
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status} for {url}")
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(1 << 16)
                        if not chunk:
                            break
                        f.write(chunk)
                tmp.replace(dest)
            return
        except Exception as e:
            if attempt == retries:
                raise
            print(f"  attempt {attempt} failed: {e}; retrying in {5*attempt}s", file=sys.stderr)
            time.sleep(5 * attempt)


def _sha256(path: Path, buf_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(buf_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _open_csv_gz(path: Path) -> csv.reader:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
        # Sniff dialect / delimiter. Sample 2019 files use comma.
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(f, dialect)
        return list(reader)


def _split_bucket(seq: str) -> str:
    """Deterministic train/val/test bucket via SHA-256 of sequence.

    80% train / 10% val / 10% test (matches Optimus100K convention).
    """
    h = hashlib.sha256(seq.upper().encode()).hexdigest()
    v = int(h[:8], 16) / 0xFFFFFFFF
    if v < 0.8:
        return "train"
    elif v < 0.9:
        return "val"
    return "test"


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def parse_csv(path: Path) -> Tuple[List[str], int, Dict[str, object]]:
    """Parse one Sample 2019 CSV.gz, return (columns, row_count, stats)."""
    rows = _open_csv_gz(path)
    if not rows:
        return [], 0, {}
    header = rows[0]
    # Normalize header names
    norm = [h.strip().lower() for h in header]
    # Locate key columns
    utr_col = None
    rl_col = None
    for i, name in enumerate(norm):
        if name in ("utr", "seq", "sequence", "five_prime_utr", "5utr"):
            utr_col = i
        elif name in ("rl", "mrl", "ribosome_load", "mean_rl"):
            rl_col = i
    data_rows = rows[1:]
    stats: Dict[str, object] = {
        "header": header,
        "header_normalized": norm,
        "utr_column": utr_col,
        "rl_column": rl_col,
        "row_count": len(data_rows),
    }
    # Quick distribution peek (no full pass to keep this fast)
    if utr_col is not None and rl_col is not None:
        lengths = []
        rl_values = []
        for r in data_rows[:5000]:
            if len(r) <= max(utr_col, rl_col):
                continue
            try:
                lengths.append(len(r[utr_col]))
                rl_values.append(float(r[rl_col]))
            except (ValueError, IndexError):
                continue
        if lengths:
            stats["peek_min_len"] = min(lengths)
            stats["peek_max_len"] = max(lengths)
            stats["peek_mean_len"] = sum(lengths) / len(lengths)
        if rl_values:
            stats["peek_min_rl"] = min(rl_values)
            stats["peek_max_rl"] = max(rl_values)
            stats["peek_mean_rl"] = sum(rl_values) / len(rl_values)
    return header, len(data_rows), stats


def process_sample(sample: Dict) -> Dict:
    """Download + process one GSM sample. Returns manifest entry."""
    gsm = sample["gsm"]
    filename = sample["filename"]
    url = _gsm_url(gsm, filename)
    dest = DATA_ROOT / filename
    print(f"[{gsm}] downloading {url}")
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  exists ({dest.stat().st_size} bytes), skipping download")
    else:
        _http_get(url, dest)
        print(f"  downloaded {dest.stat().st_size} bytes")

    sha = _sha256(dest)
    header, n_rows, stats = parse_csv(dest)
    print(f"  sha256={sha[:16]}... rows={n_rows} header={header}")

    entry = {
        "gsm": gsm,
        "geo_url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gsm}",
        "source_url": url,
        "filename": filename,
        "local_path": str(dest.relative_to(_REPO_ROOT)),
        "sha256": sha,
        "byte_size": dest.stat().st_size,
        "license": LICENSE,
        "citation": CITATION,
        "cell_type": "HEK293T",
        "library_kind": sample["library_kind"],
        "cds": sample["cds"],
        "rna_chemistry": sample["rna_chemistry"],
        "replicate": sample["replicate"],
        "expected_seq_len": sample["expected_seq_len"],
        "schema": header,
        "record_count": n_rows,
        "stats_peek": stats,
    }
    return entry


# ---------------------------------------------------------------------------
# Split computation (sequence-level deterministic hashing)
# ---------------------------------------------------------------------------

def compute_split_stats(sample: Dict) -> Dict[str, int]:
    """Pass over the file to compute deterministic train/val/test split counts."""
    gsm = sample["gsm"]
    filename = sample["filename"]
    path = DATA_ROOT / filename
    rows = _open_csv_gz(path)
    if not rows:
        return {"train": 0, "val": 0, "test": 0}
    header = [h.strip().lower() for h in rows[0]]
    utr_col = None
    for i, name in enumerate(header):
        if name in ("utr", "seq", "sequence", "five_prime_utr", "5utr"):
            utr_col = i
            break
    if utr_col is None:
        return {"train": 0, "val": 0, "test": 0, "note": "no_utr_column"}
    counts = {"train": 0, "val": 0, "test": 0}
    for r in rows[1:]:
        if len(r) <= utr_col:
            continue
        counts[_split_bucket(r[utr_col])] += 1
    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"data_root = {DATA_ROOT}")
    manifest_entries: List[Dict] = []

    for sample in SAMPLES:
        try:
            entry = process_sample(sample)
        except Exception as e:
            print(f"  ERROR for {sample['gsm']}: {e}", file=sys.stderr)
            entry = {
                "gsm": sample["gsm"],
                "source_url": _gsm_url(sample["gsm"], sample["filename"]),
                "filename": sample["filename"],
                "error": str(e),
            }
        manifest_entries.append(entry)

    # Compute split stats (one pass per file)
    print("\n=== Computing split stats (deterministic 80/10/10) ===")
    for entry, sample in zip(manifest_entries, SAMPLES):
        if "error" in entry:
            entry["split_stats"] = {"train": 0, "val": 0, "test": 0, "note": "download_failed"}
            continue
        print(f"[{entry['gsm']}] split pass ...")
        entry["split_stats"] = compute_split_stats(sample)
        print(f"  {entry['split_stats']}")

    # Write manifest
    manifest_path = DATA_ROOT / "manifest.json"
    manifest = {
        "dataset_name": "sample2019_mpra",
        "geo_accession": "GSE114002",
        "pubmed_id": "31267113",
        "citation": CITATION,
        "license": LICENSE,
        "acquisition_date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_samples": len(SAMPLES),
        "samples": manifest_entries,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\nmanifest written: {manifest_path}")

    # Summary
    total_records = sum(e.get("record_count", 0) for e in manifest_entries if "error" not in e)
    total_bytes = sum(e.get("byte_size", 0) for e in manifest_entries if "error" not in e)
    summary = {
        "dataset_name": "sample2019_mpra",
        "geo_accession": "GSE114002",
        "total_records": total_records,
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1 << 20), 2),
        "per_chemistry_breakdown": {},
        "per_library_breakdown": {},
    }
    for e in manifest_entries:
        if "error" in e:
            continue
        chem = e.get("rna_chemistry", "unknown")
        kind = e.get("library_kind", "unknown")
        summary["per_chemistry_breakdown"][chem] = (
            summary["per_chemistry_breakdown"].get(chem, 0) + e.get("record_count", 0)
        )
        summary["per_library_breakdown"][kind] = (
            summary["per_library_breakdown"].get(kind, 0) + e.get("record_count", 0)
        )
    with open(DATA_ROOT / "sample2019_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"summary written: {DATA_ROOT / 'sample2019_summary.json'}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
