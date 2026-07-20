#!/usr/bin/env python3
"""P1-03: Acquire mRNA stability / half-life datasets.

Targets:
    1. Saluki half-life (Agarwal & Kelley 2022, Genome Biology)
       - Via mRNABench Zenodo redistribution (smaller .npz files)
         https://zenodo.org/records/14708163
       - Two splits: rna_hl_human (12,900 genes), rna_hl_mouse (13,700 genes)
       - Aggregated from 66 transcriptome-wide mRNA decay datasets (39 human + 27 mouse)
       - Cell types: 10 human + 8 mouse; multiple measurement procedures (ActD, 4sU, BrU, 5EU)

    2. CodonBERT mRNA_Stability.csv (Sanofi)
       - Source: Mordstein et al. 2022 (iCodon)
         https://www.nature.com/articles/s41598-022-15526-7
       - Direct: https://github.com/Sanofi-Public/CodonBERT/blob/master/benchmarks/CodonBERT/data/fine-tune/mRNA_Stability.csv
       - Vertebrate mRNA stability dataset (codon-composition-driven)

    3. CodonBERT CoV_Vaccine_Degradation.csv (Sanofi)
       - Source: Tops et al. 2023 (OpenVaccine, Nature Machine Intelligence)
       - Direct: https://github.com/Sanofi-Public/CodonBERT/blob/master/benchmarks/CodonBERT/data/fine-tune/CoV_Vaccine_Degradation.csv
       - RNA degradation rates across multiple positions

Output layout:
    data/raw/saluki_halflife/
        rna_hl_human.npz
        rna_hl_mouse.npz
        manifest.json
        saluki_halflife_summary.json

    data/raw/codonbert_stability/
        mRNA_Stability.csv
        CoV_Vaccine_Degradation.csv
        manifest.json
        codonbert_stability_summary.json

Manifest fields (per data acquisition hard constraint):
    - source_url, sha256, license, record_count, split_stats,
      cell_type, assay_metadata
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import ssl
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

SALUKI_ROOT = Path(_REPO_ROOT) / "data" / "raw" / "saluki_halflife"
CODONBERT_ROOT = Path(_REPO_ROOT) / "data" / "raw" / "codonbert_stability"
SALUKI_ROOT.mkdir(parents=True, exist_ok=True)
CODONBERT_ROOT.mkdir(parents=True, exist_ok=True)

ZENODO_BASE = "https://zenodo.org/records/14708163/files"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/Sanofi-Public/CodonBERT/master/benchmarks/CodonBERT/data/fine-tune"

SALUKI_FILES: List[Dict] = [
    {
        "filename": "rna_hl_human.npz",
        "url": f"{ZENODO_BASE}/rna_hl_human.npz",
        "kind": "half_life_compendium",
        "species": "Homo sapiens",
        "expected_records": 12900,
        "source_experiments": 39,
        "cell_types": ["HeLa", "K562", "RPE", "HEK293", "HepG2", "MCF-7",
                       "LCLs", "A549", "GM12878", "H1-ESC", "B cell"],
        "measurement_procedures": ["ActD", "4sU", "BrU", "5EU", "alpha-Amanitin"],
        "redistribution_source": "mRNABench (Shi et al. 2025 bioRxiv, DOI:10.1101/2025.07.05.662870)",
    },
    {
        "filename": "rna_hl_mouse.npz",
        "url": f"{ZENODO_BASE}/rna_hl_mouse.npz",
        "kind": "half_life_compendium",
        "species": "Mus musculus",
        "expected_records": 13700,
        "source_experiments": 27,
        "cell_types": ["3T3", "mESC", "MEF", "mEB", "Neuro-2a",
                       "C2C12", "Dendritic cells", "M2-10B4"],
        "measurement_procedures": ["ActD", "4sU", "5EU"],
        "redistribution_source": "mRNABench (Shi et al. 2025 bioRxiv, DOI:10.1101/2025.07.05.662870)",
    },
]

CODONBERT_FILES: List[Dict] = [
    {
        "filename": "mRNA_Stability.csv",
        "url": f"{GITHUB_RAW_BASE}/mRNA_Stability.csv",
        "kind": "codon_composition_stability",
        "source_paper": "Mordstein et al. 2022 Sci Rep DOI:10.1038/s41598-022-15526-7",
        "species": "Vertebrate (multi-species)",
        "license": "Sanofi CodonBERT Artifact License (research use); underlying data from iCodon paper (Sci Rep open access)",
    },
    {
        "filename": "CoV_Vaccine_Degradation.csv",
        "url": f"{GITHUB_RAW_BASE}/CoV_Vaccine_Degradation.csv",
        "kind": "rna_degradation_rates",
        "source_paper": "Tops et al. 2023 Nat Mach Intell DOI:10.1038/s42256-022-00571-8 (OpenVaccine)",
        "species": "SARS-CoV-2 (vaccine candidates)",
        "license": "Sanofi CodonBERT Artifact License (research use); underlying data from OpenVaccine Kaggle (CC0)",
    },
]

SALUKI_LICENSE = (
    "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). "
    "Primary source: Agarwal & Kelley 2022 Genome Biology (DOI:10.1186/s13059-022-02811-x). "
    "Redistribution: mRNABench (Shi et al. 2025 bioRxiv, DOI:10.1101/2025.07.05.662870), "
    "Zenodo record 14708163. Cite both primary and redistribution sources."
)
SALUKI_CITATION = (
    "Agarwal V, Kelley DR. The genetic and biochemical determinants of mRNA "
    "degradation rates in mammals. Genome Biol. 2022;23(1):245. "
    "PMID:36419176. DOI:10.1186/s13059-022-02811-x. "
    "Zenodo: https://zenodo.org/records/6326409"
)

CODONBERT_LICENSE = (
    "Sanofi CodonBERT Software + Artifact License "
    "(https://github.com/Sanofi-Public/CodonBERT/blob/master/ARTIFACT_LICENSE.md). "
    "Research use permitted; redistribution subject to Sanofi artifact license terms. "
    "Underlying data: iCodon (Mordstein 2022, Sci Rep CC BY) and OpenVaccine (CC0)."
)
CODONBERT_CITATION = (
    "Li S, Moayedpour S, Li R, et al. CodonBERT: large language models for mRNA "
    "design and optimization. Genome Res. 2024;34(7):1027-1035. "
    "PMCID:PMC11368176. DOI:10.1101/gr.278870.123. "
    "GitHub: https://github.com/Sanofi-Public/CodonBERT"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _http_get(url: str, dest: Path, timeout: int = 180, retries: int = 3) -> None:
    """Download a URL to dest with retries; skip if file already exists with content."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] exists, size={dest.stat().st_size}")
        return
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "mrna_editflow/1.0 (P1-03 acquisition)"}
            )
            with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
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
            print(f"  attempt {attempt} failed: {e}; retrying in {5 * attempt}s",
                  file=sys.stderr)
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


def _split_bucket(seq: str) -> str:
    """Deterministic 80/10/10 split via SHA-256 of uppercased sequence."""
    h = hashlib.sha256(seq.upper().encode()).hexdigest()
    v = int(h[:8], 16) / 0xFFFFFFFF
    if v < 0.8:
        return "train"
    elif v < 0.9:
        return "val"
    return "test"


# ---------------------------------------------------------------------------
# Saluki .npz inspection
# ---------------------------------------------------------------------------

def inspect_npz(path: Path) -> Dict:
    """Inspect a Saluki .npz file and return summary stats.

    The .npz contains one-hot encoded sequences and metadata.
    Per mRNABench source code (mrna_bench/datasets/rna_hl_*.py), keys include:
      - sequence: one-hot encoded (N, 4, L)
      - cds: binary track (N, L) marking codon first positions
      - splice: binary track (N, L) marking exon 3' ends
      - target: PCA component 1 of half-life (N,)
      - gene: gene names (N,)
      - chromosome: chromosome names (N,)
    """
    try:
        import numpy as np
    except ImportError:
        return {"error": "numpy not available"}

    try:
        data = np.load(path, allow_pickle=True)
        keys = list(data.keys())
        info: Dict = {"keys": keys}
        for k in keys:
            arr = data[k]
            info[k] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            }
            # For 1-D arrays with string-like content, peek at first few items
            if arr.ndim == 1 and arr.dtype.kind in ("U", "S", "O"):
                samples = arr[:3].tolist() if len(arr) >= 3 else arr.tolist()
                info[k]["sample_values"] = samples
            elif arr.ndim == 1 and arr.dtype.kind in ("f", "i"):
                info[k]["min"] = float(np.min(arr))
                info[k]["max"] = float(np.max(arr))
                info[k]["mean"] = float(np.mean(arr))

        # Determine primary record count
        if "y" in data:
            n_records = int(data["y"].shape[0])
        elif "target" in data:
            n_records = int(data["target"].shape[0])
        elif "genes" in data:
            n_records = int(data["genes"].shape[0])
        elif "gene" in data:
            n_records = int(data["gene"].shape[0])
        else:
            # Use first array's first dim
            n_records = int(data[keys[0]].shape[0]) if data[keys[0]].ndim >= 1 else 0

        info["record_count"] = n_records

        # Compute deterministic split stats based on gene name (if available)
        # — gene name is more semantically meaningful than sequence hash for
        # transcriptome-scale data; sequence hash would still be deterministic.
        gene_key = None
        for candidate in ("genes", "gene"):
            if candidate in data:
                gene_key = candidate
                break
        if gene_key is not None:
            genes = data[gene_key]
            if genes.dtype.kind in ("U", "S", "O"):
                counts = {"train": 0, "val": 0, "test": 0}
                for g in genes:
                    counts[_split_bucket(str(g))] += 1
                info["split_stats_by_gene"] = counts
        return info
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# CodonBERT CSV parsing
# ---------------------------------------------------------------------------

def parse_csv(path: Path) -> Tuple[List[str], int, Dict]:
    """Parse a CodonBERT CSV file, return (header, row_count, stats)."""
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(f, dialect)
        for row in reader:
            rows.append(row)
    if not rows:
        return [], 0, {}
    header = rows[0]
    data_rows = rows[1:]
    norm = [h.strip().lower() for h in header]

    # Identify sequence column
    seq_col = None
    for i, name in enumerate(norm):
        if name in ("seq", "sequence", "utr", "cds", "mrna", "rna", "transcript"):
            seq_col = i
            break

    # Identify target column
    target_col = None
    for i, name in enumerate(norm):
        if name in ("target", "label", "stability", "half_life", "halflife", "rl", "mrl"):
            target_col = i
            break

    stats: Dict = {
        "header": header,
        "header_normalized": norm,
        "sequence_column": seq_col,
        "target_column": target_col,
        "row_count": len(data_rows),
    }

    # Compute deterministic split stats from sequence column (if available)
    if seq_col is not None:
        counts = {"train": 0, "val": 0, "test": 0, "skipped": 0}
        for r in data_rows:
            if len(r) <= seq_col:
                counts["skipped"] += 1
                continue
            counts[_split_bucket(r[seq_col])] += 1
        stats["split_stats"] = counts

    return header, len(data_rows), stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_saluki() -> Tuple[List[Dict], Dict]:
    """Download + inspect Saluki half-life .npz files."""
    print("\n=== Saluki half-life acquisition ===")
    entries: List[Dict] = []
    for spec in SALUKI_FILES:
        dest = SALUKI_ROOT / spec["filename"]
        print(f"\n[{spec['filename']}] downloading {spec['url']}")
        try:
            _http_get(spec["url"], dest)
            print(f"  downloaded {dest.stat().st_size} bytes")
            sha = _sha256(dest)
            print(f"  sha256={sha[:16]}...")
            inspect = inspect_npz(dest)
            print(f"  inspect: {inspect.get('record_count', 'n/a')} records, "
                  f"keys={inspect.get('keys', [])}")
            entry = {
                "filename": spec["filename"],
                "source_url": spec["url"],
                "local_path": str(dest.relative_to(_REPO_ROOT)),
                "sha256": sha,
                "byte_size": dest.stat().st_size,
                "license": SALUKI_LICENSE,
                "citation": SALUKI_CITATION,
                "kind": spec["kind"],
                "species": spec["species"],
                "expected_records": spec["expected_records"],
                "source_experiments": spec["source_experiments"],
                "cell_types": spec["cell_types"],
                "measurement_procedures": spec["measurement_procedures"],
                "redistribution_source": spec["redistribution_source"],
                "inspection": inspect,
            }
            entry["record_count"] = inspect.get("record_count", 0)
            entry["split_stats"] = inspect.get("split_stats_by_gene",
                                                {"train": 0, "val": 0, "test": 0})
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            entry = {
                "filename": spec["filename"],
                "source_url": spec["url"],
                "error": str(e),
                "split_stats": {"train": 0, "val": 0, "test": 0, "note": "download_failed"},
            }
        entries.append(entry)

    summary = {
        "dataset_name": "saluki_halflife",
        "primary_source": SALUKI_CITATION,
        "redistribution": "mRNABench Shi et al. 2025 bioRxiv DOI:10.1101/2025.07.05.662870",
        "total_files": len(SALUKI_FILES),
        "ok_files": sum(1 for e in entries if "error" not in e),
        "failed_files": sum(1 for e in entries if "error" in e),
        "total_records": sum(e.get("record_count", 0) for e in entries),
        "by_species": {
            e.get("species", "unknown"): e.get("record_count", 0)
            for e in entries if "error" not in e
        },
    }
    return entries, summary


def process_codonbert() -> Tuple[List[Dict], Dict]:
    """Download + parse CodonBERT stability CSVs."""
    print("\n=== CodonBERT stability acquisition ===")
    entries: List[Dict] = []
    for spec in CODONBERT_FILES:
        dest = CODONBERT_ROOT / spec["filename"]
        print(f"\n[{spec['filename']}] downloading {spec['url']}")
        try:
            _http_get(spec["url"], dest)
            print(f"  downloaded {dest.stat().st_size} bytes")
            sha = _sha256(dest)
            header, n_rows, stats = parse_csv(dest)
            print(f"  sha256={sha[:16]}... rows={n_rows} header={header[:8]}...")
            entry = {
                "filename": spec["filename"],
                "source_url": spec["url"],
                "local_path": str(dest.relative_to(_REPO_ROOT)),
                "sha256": sha,
                "byte_size": dest.stat().st_size,
                "license": CODONBERT_LICENSE,
                "citation": CODONBERT_CITATION,
                "kind": spec["kind"],
                "source_paper": spec["source_paper"],
                "species": spec["species"],
                "schema": header,
                "record_count": n_rows,
                "split_stats": stats.get("split_stats",
                                          {"train": 0, "val": 0, "test": 0}),
                "stats": stats,
            }
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            entry = {
                "filename": spec["filename"],
                "source_url": spec["url"],
                "error": str(e),
                "split_stats": {"train": 0, "val": 0, "test": 0, "note": "download_failed"},
            }
        entries.append(entry)

    summary = {
        "dataset_name": "codonbert_stability",
        "primary_source": CODONBERT_CITATION,
        "total_files": len(CODONBERT_FILES),
        "ok_files": sum(1 for e in entries if "error" not in e),
        "failed_files": sum(1 for e in entries if "error" in e),
        "total_records": sum(e.get("record_count", 0) for e in entries),
        "by_kind": {
            e.get("kind", "unknown"): e.get("record_count", 0)
            for e in entries if "error" not in e
        },
    }
    return entries, summary


def main() -> int:
    print(f"saluki_root = {SALUKI_ROOT}")
    print(f"codonbert_root = {CODONBERT_ROOT}")

    saluki_entries, saluki_summary = process_saluki()
    codonbert_entries, codonbert_summary = process_codonbert()

    # Write manifests
    saluki_manifest = {
        "dataset_name": "saluki_halflife",
        "primary_source": SALUKI_CITATION,
        "license": SALUKI_LICENSE,
        "acquisition_date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_files": len(SALUKI_FILES),
        "files": saluki_entries,
    }
    saluki_manifest_path = SALUKI_ROOT / "manifest.json"
    with open(saluki_manifest_path, "w") as f:
        json.dump(saluki_manifest, f, indent=2, sort_keys=True)
    print(f"\nsaluki manifest written: {saluki_manifest_path}")

    saluki_summary_path = SALUKI_ROOT / "saluki_halflife_summary.json"
    with open(saluki_summary_path, "w") as f:
        json.dump(saluki_summary, f, indent=2, sort_keys=True)
    print(f"saluki summary written: {saluki_summary_path}")

    codonbert_manifest = {
        "dataset_name": "codonbert_stability",
        "primary_source": CODONBERT_CITATION,
        "license": CODONBERT_LICENSE,
        "acquisition_date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_files": len(CODONBERT_FILES),
        "files": codonbert_entries,
    }
    codonbert_manifest_path = CODONBERT_ROOT / "manifest.json"
    with open(codonbert_manifest_path, "w") as f:
        json.dump(codonbert_manifest, f, indent=2, sort_keys=True)
    print(f"codonbert manifest written: {codonbert_manifest_path}")

    codonbert_summary_path = CODONBERT_ROOT / "codonbert_stability_summary.json"
    with open(codonbert_summary_path, "w") as f:
        json.dump(codonbert_summary, f, indent=2, sort_keys=True)
    print(f"codonbert summary written: {codonbert_summary_path}")

    # Print overall summary
    print("\n=== Overall Summary ===")
    print(json.dumps({
        "saluki_halflife": saluki_summary,
        "codonbert_stability": codonbert_summary,
    }, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
