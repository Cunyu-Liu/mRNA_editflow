#!/usr/bin/env python
"""D1-04: Reconstruct oligo sequences for GSE186455 from RAW.tar.

GSE186455 (Zhang et al.) contains ~3,894 paired 3'UTR variants with MPRA
activity data across N2a and Vglut cell types (TRAP-seq based).

Data layout:
  - GSE186455_RAW.tar: 13 .tab.gz files, each with columns:
      bc, sequence, seqName, bcCount
    seqName format: {gene}_{source}_{type} (type = ref/alt/shuf)
    Sequences are ~105nt oligos.
  - GSE186455_N2a_counts.csv.gz: RNA counts, 13 samples (SIC0214-0226)
  - GSE186455_Vglut_counts.csv.gz: RNA counts, 17 samples (TRAP from Vglut)
  - GSE186455_TRAP_DNA_counts.csv.gz: DNA counts, 13 samples

CSV columns: Unnamed:0, BC, Element, SIC01XX...
Element = seqName. Multiple barcodes per Element.

This script:
1. Extracts seqName -> sequence mapping from RAW.tar
2. Aggregates per-Element counts from CSVs (sum across barcodes)
3. Matches ref/alt pairs by (gene, source)
4. Computes MPRA activity (RNA/DNA) and log2FC(alt/ref)
5. Writes reconstructed JSONL

Output: JSONL with one record per paired variant.

Contract: utr_editflow_contract_v2 (FROZEN)
Task: D1-04
"""

import argparse
import csv
import gzip
import io
import json
import math
import os
import re
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make genome_fetcher importable for normalize_seq
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from genome_fetcher import normalize_seq  # noqa: E402


# ---------------------------------------------------------------------------
# RAW.tar: extract seqName -> sequence mapping
# ---------------------------------------------------------------------------

def extract_sequences_from_tar(tar_path: Path) -> Dict[str, str]:
    """Extract seqName -> sequence mapping from GSE186455_RAW.tar.

    Each .tab.gz file has columns: bc, sequence, seqName, bcCount.
    Multiple barcodes map to the same seqName but should have the same
    sequence. We verify this and build a unique mapping.

    Returns: {seqName: sequence}
    """
    seq_map: Dict[str, str] = {}
    conflicts = 0

    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".tab.gz"):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            with gzip.open(io.BytesIO(f.read()), "rt") as gz:
                reader = csv.DictReader(gz, delimiter="\t")
                for row in reader:
                    seq_name = row.get("seqName", "").strip()
                    sequence = row.get("sequence", "").strip().upper()
                    if not seq_name or not sequence:
                        continue
                    if seq_name in seq_map:
                        if seq_map[seq_name] != sequence:
                            conflicts += 1
                    else:
                        seq_map[seq_name] = sequence

    if conflicts:
        print(f"  WARNING: {conflicts} sequence conflicts across barcodes")
    return seq_map


# ---------------------------------------------------------------------------
# CSV count parsing
# ---------------------------------------------------------------------------

def parse_count_csv(path: Path) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """Parse a GSE186455 count CSV file.

    Aggregates per-Element counts by summing across barcodes.

    Returns: ({element: {sample_col: total_count}}, [sample_cols])
    """
    open_fn = gzip.open if path.suffix == ".gz" else open
    sample_cols: List[str] = []
    sample_indices: List[int] = []
    element_counts: Dict[str, Dict[str, float]] = {}

    with open_fn(path, "rt") as f:
        reader = csv.reader(f)
        header = next(reader)
        elem_idx = None
        for i, col in enumerate(header):
            col_stripped = col.strip()
            if col_stripped == "Element":
                elem_idx = i
            elif col_stripped not in ("", "Unnamed: 0", "BC"):
                sample_cols.append(col_stripped)
                sample_indices.append(i)

        if elem_idx is None:
            print(f"  WARNING: no Element column in {path.name}")
            return {}, []

        for row in reader:
            if len(row) <= elem_idx:
                continue
            element = row[elem_idx].strip()
            if not element:
                continue
            if element not in element_counts:
                element_counts[element] = {s: 0.0 for s in sample_cols}
            for j, col_idx in enumerate(sample_indices):
                if col_idx < len(row):
                    try:
                        val = float(row[col_idx])
                    except (ValueError, IndexError):
                        val = 0.0
                    element_counts[element][sample_cols[j]] += val

    return element_counts, sample_cols


# ---------------------------------------------------------------------------
# Pair matching and label computation
# ---------------------------------------------------------------------------

def parse_seqname(seq_name: str) -> Optional[dict]:
    """Parse a seqName like 'RBM41_ssc_ref' into components.

    Format: {gene}_{source}_{type}
    gene may contain underscores (e.g., '9-Sep', 'AC010536.1').
    type is always the last field (ref/alt/shuf).
    source is always the second-to-last field.
    """
    parts = seq_name.split("_")
    if len(parts) < 3:
        return None
    vtype = parts[-1].strip().lower()
    source = parts[-2].strip().lower()
    gene = "_".join(parts[:-2])
    if vtype not in ("ref", "alt", "shuf"):
        return None
    return {"gene": gene, "source": source, "type": vtype}


def build_pairs(
    seq_map: Dict[str, str],
    n2a_counts: Dict[str, Dict[str, float]],
    vglut_counts: Dict[str, Dict[str, float]],
    dna_counts: Dict[str, Dict[str, float]],
    n2a_samples: List[str],
    vglut_samples: List[str],
    dna_samples: List[str],
) -> List[dict]:
    """Match ref/alt pairs by (gene, source) and compute labels.

    Returns list of paired records.
    """
    # Group by (gene, source)
    grouped: Dict[Tuple[str, str], dict] = {}
    for seq_name, seq in seq_map.items():
        parsed = parse_seqname(seq_name)
        if parsed is None:
            continue
        key = (parsed["gene"], parsed["source"])
        if key not in grouped:
            grouped[key] = {}
        grouped[key][parsed["type"]] = {
            "seq_name": seq_name,
            "sequence": seq,
        }

    pairs = []
    for (gene, source), alleles in grouped.items():
        if "ref" not in alleles or "alt" not in alleles:
            continue
        ref_info = alleles["ref"]
        alt_info = alleles["alt"]

        ref_seq = normalize_seq(ref_info["sequence"])
        alt_seq = normalize_seq(alt_info["sequence"])
        if not ref_seq or not alt_seq or ref_seq == alt_seq:
            continue

        # Find variant position
        diffs = [
            (i, ref_seq[i], alt_seq[i])
            for i in range(min(len(ref_seq), len(alt_seq)))
            if ref_seq[i] != alt_seq[i]
        ]
        if len(diffs) == 0:
            continue

        var_pos, ref_base, alt_base = diffs[0]
        variant_type = "snv" if len(ref_base) == len(alt_base) == 1 else "indel"

        # Compute labels
        labels = _compute_labels(
            ref_info["seq_name"],
            alt_info["seq_name"],
            n2a_counts,
            vglut_counts,
            dna_counts,
            n2a_samples,
            vglut_samples,
            dna_samples,
        )

        rid = f"GSE186455_{source}_{gene}"
        record = {
            "record_id": rid,
            "source_sequence": ref_seq,
            "candidate_sequence": alt_seq,
            "region": "3'UTR",
            "variant_type": variant_type,
            "labels": labels,
            "metadata": {
                "gene_symbol": gene,
                "variant_source": source,
                "ref_allele": ref_base,
                "alt_allele": alt_base,
                "variant_position": var_pos,
                "insert_length": len(ref_seq),
                "ref_seqname": ref_info["seq_name"],
                "alt_seqname": alt_info["seq_name"],
                "source_file": "GSE186455_RAW.tar",
            },
        }
        pairs.append(record)

    return pairs


def _compute_labels(
    ref_seq_name: str,
    alt_seq_name: str,
    n2a_counts: Dict[str, Dict[str, float]],
    vglut_counts: Dict[str, Dict[str, float]],
    dna_counts: Dict[str, Dict[str, float]],
    n2a_samples: List[str],
    vglut_samples: List[str],
    dna_samples: List[str],
) -> dict:
    """Compute MPRA activity labels for a ref/alt pair.

    Labels:
      n2a_dna_ref, n2a_dna_alt (summed across DNA samples matched to N2a)
      n2a_rna_ref_rep{i}, n2a_rna_alt_rep{i}
      n2a_activity_ref_rep{i}, n2a_activity_alt_rep{i}
      n2a_activity_ref_mean, n2a_activity_alt_mean
      n2a_log2fc_activity

      vglut_rna_ref_rep{i}, vglut_rna_alt_rep{i}
      vglut_dna_ref, vglut_dna_alt
      vglut_activity_ref_rep{i}, vglut_activity_alt_rep{i}
      vglut_activity_ref_mean, vglut_activity_alt_mean
      vglut_log2fc_activity
    """
    labels = {}

    ref_dna = dna_counts.get(ref_seq_name, {})
    alt_dna = dna_counts.get(alt_seq_name, {})
    ref_n2a = n2a_counts.get(ref_seq_name, {})
    alt_n2a = n2a_counts.get(alt_seq_name, {})
    ref_vglut = vglut_counts.get(ref_seq_name, {})
    alt_vglut = vglut_counts.get(alt_seq_name, {})

    # N2a: aggregate DNA (sum across DNA samples), per-sample RNA
    n2a_dna_ref = sum(ref_dna.get(s, 0) for s in dna_samples)
    n2a_dna_alt = sum(alt_dna.get(s, 0) for s in dna_samples)
    labels["n2a_dna_ref"] = n2a_dna_ref
    labels["n2a_dna_alt"] = n2a_dna_alt

    n2a_act_ref_vals = []
    n2a_act_alt_vals = []
    for i, s in enumerate(n2a_samples, 1):
        rna_r = ref_n2a.get(s, 0)
        rna_a = alt_n2a.get(s, 0)
        labels[f"n2a_rna_ref_rep{i}"] = rna_r
        labels[f"n2a_rna_alt_rep{i}"] = rna_a
        if n2a_dna_ref > 0:
            act_r = rna_r / n2a_dna_ref
            labels[f"n2a_activity_ref_rep{i}"] = act_r
            n2a_act_ref_vals.append(act_r)
        if n2a_dna_alt > 0:
            act_a = rna_a / n2a_dna_alt
            labels[f"n2a_activity_alt_rep{i}"] = act_a
            n2a_act_alt_vals.append(act_a)

    if n2a_act_ref_vals:
        labels["n2a_activity_ref_mean"] = sum(n2a_act_ref_vals) / len(n2a_act_ref_vals)
    if n2a_act_alt_vals:
        labels["n2a_activity_alt_mean"] = sum(n2a_act_alt_vals) / len(n2a_act_alt_vals)
    if (
        "n2a_activity_ref_mean" in labels
        and "n2a_activity_alt_mean" in labels
        and labels["n2a_activity_ref_mean"] > 0
        and labels["n2a_activity_alt_mean"] > 0
    ):
        labels["n2a_log2fc_activity"] = math.log2(
            labels["n2a_activity_alt_mean"] / labels["n2a_activity_ref_mean"]
        )

    # Vglut: aggregate DNA, per-sample RNA
    vglut_dna_ref = sum(ref_dna.get(s, 0) for s in dna_samples)
    vglut_dna_alt = sum(alt_dna.get(s, 0) for s in dna_samples)
    labels["vglut_dna_ref"] = vglut_dna_ref
    labels["vglut_dna_alt"] = vglut_dna_alt

    vglut_act_ref_vals = []
    vglut_act_alt_vals = []
    for i, s in enumerate(vglut_samples, 1):
        rna_r = ref_vglut.get(s, 0)
        rna_a = alt_vglut.get(s, 0)
        labels[f"vglut_rna_ref_rep{i}"] = rna_r
        labels[f"vglut_rna_alt_rep{i}"] = rna_a
        if vglut_dna_ref > 0:
            act_r = rna_r / vglut_dna_ref
            labels[f"vglut_activity_ref_rep{i}"] = act_r
            vglut_act_ref_vals.append(act_r)
        if vglut_dna_alt > 0:
            act_a = rna_a / vglut_dna_alt
            labels[f"vglut_activity_alt_rep{i}"] = act_a
            vglut_act_alt_vals.append(act_a)

    if vglut_act_ref_vals:
        labels["vglut_activity_ref_mean"] = sum(vglut_act_ref_vals) / len(vglut_act_ref_vals)
    if vglut_act_alt_vals:
        labels["vglut_activity_alt_mean"] = sum(vglut_act_alt_vals) / len(vglut_act_alt_vals)
    if (
        "vglut_activity_ref_mean" in labels
        and "vglut_activity_alt_mean" in labels
        and labels["vglut_activity_ref_mean"] > 0
        and labels["vglut_activity_alt_mean"] > 0
    ):
        labels["vglut_log2fc_activity"] = math.log2(
            labels["vglut_activity_alt_mean"] / labels["vglut_activity_ref_mean"]
        )

    return labels


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct GSE186455 paired 3'UTR sequences"
    )
    parser.add_argument(
        "--data-dir",
        default="data/p0/GSE186455",
        help="Directory containing RAW.tar and CSV files",
    )
    parser.add_argument(
        "--output",
        default="data/p0/GSE186455/reconstructed_pairs.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print("=== D1-04: Reconstruct GSE186455 sequences ===")

    # 1. Extract sequences from RAW.tar
    tar_path = data_dir / "GSE186455_RAW.tar"
    if not tar_path.exists():
        print(f"ERROR: {tar_path} not found")
        sys.exit(1)
    print(f"\n[1] Extracting sequences from {tar_path.name}")
    seq_map = extract_sequences_from_tar(tar_path)
    print(f"  {len(seq_map)} unique seqName -> sequence mappings")

    # 2. Parse count CSVs
    print(f"\n[2] Parsing count CSVs")
    n2a_path = data_dir / "GSE186455_N2a_counts.csv.gz"
    vglut_path = data_dir / "GSE186455_Vglut_counts.csv.gz"
    dna_path = data_dir / "GSE186455_TRAP_DNA_counts.csv.gz"

    n2a_counts, n2a_samples = parse_count_csv(n2a_path)
    print(f"  N2a: {len(n2a_counts)} elements, {len(n2a_samples)} samples")
    vglut_counts, vglut_samples = parse_count_csv(vglut_path)
    print(f"  Vglut: {len(vglut_counts)} elements, {len(vglut_samples)} samples")
    dna_counts, dna_samples = parse_count_csv(dna_path)
    print(f"  TRAP_DNA: {len(dna_counts)} elements, {len(dna_samples)} samples")

    # 3. Build ref/alt pairs
    print(f"\n[3] Matching ref/alt pairs")
    pairs = build_pairs(
        seq_map,
        n2a_counts,
        vglut_counts,
        dna_counts,
        n2a_samples,
        vglut_samples,
        dna_samples,
    )
    print(f"  Found {len(pairs)} paired variants")

    # 4. Write output
    print(f"\n[4] Writing {len(pairs)} records to {args.output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for rec in pairs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Stats
    print(f"\n=== Statistics ===")
    print(f"  Total paired variants: {len(pairs)}")

    by_source = {}
    for rec in pairs:
        s = rec["metadata"]["variant_source"]
        by_source[s] = by_source.get(s, 0) + 1
    print(f"  By variant source: {by_source}")

    by_type = {}
    for rec in pairs:
        by_type[rec["variant_type"]] = by_type.get(rec["variant_type"], 0) + 1
    print(f"  By variant type: {by_type}")

    n_with_labels = sum(1 for r in pairs if r["labels"])
    print(f"  With expression labels: {n_with_labels}")

    n_n2a = sum(1 for r in pairs if "n2a_log2fc_activity" in r.get("labels", {}))
    print(f"  With N2a log2fc: {n_n2a}")
    n_vglut = sum(1 for r in pairs if "vglut_log2fc_activity" in r.get("labels", {}))
    print(f"  With Vglut log2fc: {n_vglut}")

    print(f"\nDone. Output: {args.output}")


if __name__ == "__main__":
    main()
