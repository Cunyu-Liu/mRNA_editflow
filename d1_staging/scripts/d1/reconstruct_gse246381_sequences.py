#!/usr/bin/env python
"""D1-03: Reconstruct 5'UTR sequences for GSE246381 from hg19 + Gencode v19 GTF.

GSE246381 contains paired 5'UTR variants (REF/ALT) with translational
efficiency data (Polysome/Monosome UMI counts) but NO actual sequences.
SeqIDs encode variant annotations:
  Variant;chr3:123954485;CA|C;Family=14179;ENST00000485727;REF;TTAAGCTTCA

This script reconstructs WT (REF) and MUT (ALT) 5'UTR sequences by:
1. Parsing CSV SeqIDs for variant annotations + UMI counts
2. Parsing Gencode v19 GTF for 5'UTR exon coordinates per ENST
3. Fetching hg19 reference for each UTR exon
4. Splicing exons (RC if strand == '-')
5. Applying the variant at the genomic position

Output: JSONL with one record per (variant, ENST) pair.

Contract: utr_editflow_contract_v2 (FROZEN)
Task: D1-03
"""

import argparse
import gzip
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Make genome_fetcher importable
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from genome_fetcher import (  # noqa: E402
    reverse_complement,
    normalize_seq,
    ensure_fa,
    create_fetcher,
)


# ---------------------------------------------------------------------------
# SeqID parsing
# ---------------------------------------------------------------------------

def parse_seqid(seqid: str) -> Optional[dict]:
    """Parse a GSE246381 SeqID string.

    Format: Variant;chr3:123954485;CA|C;Family=14179;ENST00000485727;REF;TTAAGCTTCA

    Returns dict with: chrom, pos, ref, alt, family, enst, allele, barcode
    """
    parts = str(seqid).split(";")
    if len(parts) < 7 or parts[0] != "Variant":
        return None
    try:
        # parts[1] = "chr3:123954485"
        chrom_str, pos_str = parts[1].split(":")
        pos = int(pos_str)
        # parts[2] = "CA|C"
        ref, alt = parts[2].split("|")
        # parts[3] = "Family=14179"
        family = parts[3].split("=")[-1] if "=" in parts[3] else parts[3]
        # parts[4] = "ENST00000485727"
        enst = parts[4]
        # parts[5] = "REF" or "ALT"
        allele = parts[5].upper()
        # parts[6] = "TTAAGCTTCA" (barcode)
        barcode = parts[6] if len(parts) > 6 else ""
        return {
            "chrom": chrom_str,
            "pos": pos,
            "ref": ref.upper(),
            "alt": alt.upper(),
            "family": family,
            "enst": enst,
            "allele": allele,
            "barcode": barcode,
        }
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# GTF parsing — extract 5'UTR exons per transcript
# ---------------------------------------------------------------------------

def parse_gtf_5utr(gtf_path: Path) -> Dict[str, dict]:
    """Parse Gencode GTF to extract 5'UTR exon coordinates per transcript.

    Returns: {ENST_id: {
        "chrom": str, "strand": str,
        "cds_start": int, "cds_end": int,  # 1-indexed inclusive
        "utr5_exons": [(start, end), ...],  # 1-indexed inclusive, sorted ascending
    }}
    """
    # First pass: collect UTR and CDS features per transcript
    utr_features = defaultdict(list)  # ENST -> [(chrom, start, end, strand)]
    cds_features = defaultdict(list)  # ENST -> [(chrom, start, end, strand)]
    transcript_strand = {}  # ENST -> strand

    open_fn = gzip.open if str(gtf_path).endswith(".gz") else open
    with open_fn(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, source, feature, start, end, score, strand, frame, attrs = parts
            start = int(start)
            end = int(end)

            # Extract transcript_id
            m = re.search(r'transcript_id "([^"]+)"', attrs)
            if not m:
                continue
            tid = m.group(1)
            # Use base ID (strip version)
            tid_base = tid.split(".")[0]

            if feature == "transcript":
                transcript_strand[tid_base] = strand

            if feature == "UTR":
                utr_features[tid_base].append((chrom, start, end, strand))
            elif feature == "CDS":
                cds_features[tid_base].append((chrom, start, end, strand))

    # Second pass: determine which UTRs are 5'UTR
    result = {}
    for tid, utrs in utr_features.items():
        cds_list = cds_features.get(tid, [])
        if not cds_list:
            continue  # No CDS → can't determine 5' vs 3' UTR

        strand = transcript_strand.get(tid, "+")
        chrom = utrs[0][0]

        # CDS span (1-indexed inclusive)
        cds_start = min(c[1] for c in cds_list)
        cds_end = max(c[2] for c in cds_list)

        # 5'UTR exons:
        # '+' strand: UTRs with end < cds_start (upstream of CDS)
        # '-' strand: UTRs with start > cds_end (downstream of CDS in genomic coords)
        utr5_exons = []
        for c, s, e, st in utrs:
            if strand == "+":
                if e < cds_start:
                    utr5_exons.append((s, e))
            else:
                if s > cds_end:
                    utr5_exons.append((s, e))

        if not utr5_exons:
            continue

        # Sort by genomic start ascending
        utr5_exons.sort()
        result[tid] = {
            "chrom": chrom,
            "strand": strand,
            "cds_start": cds_start,
            "cds_end": cds_end,
            "utr5_exons": utr5_exons,
        }

    return result


# ---------------------------------------------------------------------------
# Variant application on spliced 5'UTR
# ---------------------------------------------------------------------------

def build_utr5_plus_strand(
    gf, chrom: str, utr5_exons: list
) -> str:
    """Build the + strand spliced 5'UTR sequence.

    Args:
        gf
        chrom: chromosome name
        utr5_exons: list of (start, end) tuples, 1-indexed inclusive, sorted ascending

    Returns:
        + strand spliced UTR sequence (not yet RC'd for '-' strand).
    """
    parts = []
    for start, end in utr5_exons:
        # 1-indexed inclusive -> 0-indexed half-open
        seq = gf.fetch(chrom, start - 1, end)
        parts.append(seq)
    return "".join(parts)


def apply_variant_at_genomic_pos(
    gf,
    chrom: str,
    utr5_exons: list,
    var_pos: int,
    ref_allele: str,
    alt_allele: str,
) -> Optional[Tuple[str, str, int]]:
    """Apply variant at a genomic position on the spliced 5'UTR.

    The variant is applied on the + strand before splicing/RC.

    Args:
        gf
        chrom: chromosome
        utr5_exons: list of (start, end), 1-indexed inclusive, sorted ascending
        var_pos: genomic position (1-indexed) of variant start
        ref_allele: reference allele (+ strand)
        alt_allele: alternate allele (+ strand)

    Returns:
        (wt_plus_seq, mut_plus_seq, variant_offset_in_utr) or None if:
        - variant is not within any UTR exon
        - ref_allele doesn't match
    """
    ref_len = len(ref_allele)

    # Find which exon contains the variant
    exon_idx = None
    var_offset_in_exon = None
    for i, (start, end) in enumerate(utr5_exons):
        if start <= var_pos <= end:
            exon_idx = i
            var_offset_in_exon = var_pos - start  # 0-indexed within exon
            break

    if exon_idx is None:
        return None

    # Fetch the exon + strand sequence
    exon_start, exon_end = utr5_exons[exon_idx]
    exon_seq = gf.fetch(chrom, exon_start - 1, exon_end)

    # Verify ref_allele
    actual_ref = exon_seq[var_offset_in_exon : var_offset_in_exon + ref_len]
    if actual_ref.upper() != ref_allele.upper():
        # Try with adjacent exon (for variants spanning exon boundaries)
        return None

    # Apply variant on exon
    mut_exon_seq = (
        exon_seq[:var_offset_in_exon]
        + alt_allele.upper()
        + exon_seq[var_offset_in_exon + ref_len :]
    )

    # Build full + strand UTR with variant applied
    wt_parts = []
    mut_parts = []
    for i, (start, end) in enumerate(utr5_exons):
        if i == exon_idx:
            wt_parts.append(exon_seq)
            mut_parts.append(mut_exon_seq)
        else:
            seq = gf.fetch(chrom, start - 1, end)
            wt_parts.append(seq)
            mut_parts.append(seq)

    wt_plus = "".join(wt_parts)
    mut_plus = "".join(mut_parts)

    # Calculate variant offset in the full UTR
    offset_before_exon = sum(end - start + 1 for start, end in utr5_exons[:exon_idx])
    var_offset_in_utr = offset_before_exon + var_offset_in_exon

    return (wt_plus, mut_plus, var_offset_in_utr)


# ---------------------------------------------------------------------------
# UMI count aggregation
# ---------------------------------------------------------------------------

def aggregate_umi_counts(
    df: pd.DataFrame, sample_cols: list
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Aggregate UMI counts per (variant_key, ENST, allele).

    Args:
        df: DataFrame with SeqID + sample columns
        sample_cols: list of sample column names

    Returns:
        {(chrom, pos, ref, alt, enst): {"REF": {col: sum}, "ALT": {col: sum}}}
    """
    result = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    for _, row in df.iterrows():
        parsed = parse_seqid(row["SeqID"])
        if parsed is None:
            continue
        key = (
            parsed["chrom"],
            parsed["pos"],
            parsed["ref"],
            parsed["alt"],
            parsed["enst"],
        )
        allele = parsed["allele"]
        for col in sample_cols:
            val = row[col]
            if pd.notna(val) and val > 0:
                result[key][allele][col] += float(val)

    return dict(result)


def compute_te_labels(
    ref_counts: Dict[str, float],
    alt_counts: Dict[str, float],
    sample_cols: list,
    condition_map: Optional[dict] = None,
) -> dict:
    """Compute translational efficiency labels from UMI counts.

    For Vglut data with condition_map:
        condition_map = {col: ("CreON"/"CreOFF", "Input"/"DNA"/"Monosome"/"Polysome")}
        TE = Polysome / Monosome per allele per Cre status
        log2FC_TE = log2(TE_ALT / TE_REF)

    For HEK data without condition_map:
        mean_umi = mean across all samples per allele
        log2FC = log2(mean_ALT / mean_REF)

    Returns dict of label name -> value.
    """
    labels = {}

    if condition_map:
        for cre in ("CreON", "CreOFF"):
            ref_poly = sum(v for c, v in ref_counts.items()
                          if condition_map.get(c, (None, None))[0] == cre
                          and condition_map.get(c, (None, None))[1] == "Polysome")
            ref_mono = sum(v for c, v in ref_counts.items()
                          if condition_map.get(c, (None, None))[0] == cre
                          and condition_map.get(c, (None, None))[1] == "Monosome")
            alt_poly = sum(v for c, v in alt_counts.items()
                          if condition_map.get(c, (None, None))[0] == cre
                          and condition_map.get(c, (None, None))[1] == "Polysome")
            alt_mono = sum(v for c, v in alt_counts.items()
                          if condition_map.get(c, (None, None))[0] == cre
                          and condition_map.get(c, (None, None))[1] == "Monosome")

            if ref_mono > 0:
                labels[f"te_ref_{cre.lower()}"] = ref_poly / ref_mono
            if alt_mono > 0:
                labels[f"te_alt_{cre.lower()}"] = alt_poly / alt_mono
            if ref_mono > 0 and alt_mono > 0:
                te_r = ref_poly / ref_mono if ref_mono > 0 else 0
                te_a = alt_poly / alt_mono if alt_mono > 0 else 0
                if te_r > 0 and te_a > 0:
                    labels[f"log2fc_te_{cre.lower()}"] = math.log2(te_a / te_r)

            # DNA and Input as controls
            ref_dna = sum(v for c, v in ref_counts.items()
                         if condition_map.get(c, (None, None))[0] == cre
                         and condition_map.get(c, (None, None))[1] == "DNA")
            alt_dna = sum(v for c, v in alt_counts.items()
                         if condition_map.get(c, (None, None))[0] == cre
                         and condition_map.get(c, (None, None))[1] == "DNA")
            if ref_dna > 0:
                labels[f"dna_ref_{cre.lower()}"] = ref_dna
            if alt_dna > 0:
                labels[f"dna_alt_{cre.lower()}"] = alt_dna
    else:
        # HEK: simple mean across all samples
        ref_vals = [v for v in ref_counts.values() if v > 0]
        alt_vals = [v for v in alt_counts.values() if v > 0]
        if ref_vals:
            labels["mean_umi_ref"] = sum(ref_vals) / len(ref_vals)
        if alt_vals:
            labels["mean_umi_alt"] = sum(alt_vals) / len(alt_vals)
        if ref_vals and alt_vals:
            mean_r = sum(ref_vals) / len(ref_vals)
            mean_a = sum(alt_vals) / len(alt_vals)
            if mean_r > 0 and mean_a > 0:
                labels["log2fc_umi"] = math.log2(mean_a / mean_r)

    return labels


def build_vglut_condition_map(sample_cols: list) -> dict:
    """Parse Vglut sample column names into (CreStatus, Condition) tuples.

    Column format: Vglut_{Input/DNA/Monosome/Polysome}_{CreON/CreOFF}-{id}-{type}-{flag}
    """
    cond_map = {}
    for col in sample_cols:
        # Extract condition and Cre status from column name
        m = re.match(r"Vglut_(Input|DNA|Monosome|Polysome)_(Cre\w+)-", col)
        if m:
            condition = m.group(1)
            cre = m.group(2)
            cond_map[col] = (cre, condition)
    return cond_map


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct GSE246381 5'UTR sequences from GRCh38 + Gencode v44"
    )
    parser.add_argument(
        "--hek-csv",
        default="data/p0/GSE246381/GSE246381_hek_combined_umi_counts.csv.gz",
    )
    parser.add_argument(
        "--vglut-csv",
        default="data/p0/GSE246381/GSE246381_vglut_combined_umi_counts.csv.gz",
    )
    parser.add_argument(
        "--gtf",
        default="/mnt/cunyuliu/genomes/gencode.v44.annotation.gtf.gz",
        help="Path to Gencode v44 GTF (gzipped, GRCh38)",
    )
    parser.add_argument(
        "--genome",
        default="ensembl",
        help="'ensembl' for REST API, or path to hg38 genome (.2bit or .fa)",
    )
    parser.add_argument(
        "--genome-build",
        default="GRCh38",
        help="Genome build for Ensembl API: 'GRCh37' or 'GRCh38'",
    )
    parser.add_argument(
        "--twobittofa",
        default="/mnt/cunyuliu/genomes/twoBitToFa",
    )
    parser.add_argument(
        "--output",
        default="data/p0/GSE246381/reconstructed_utrs.jsonl",
    )
    args = parser.parse_args()

    print("=== D1-03: Reconstruct GSE246381 5'UTR sequences ===")

    # 1. Create genome fetcher (Ensembl API or local FASTA)
    gf = create_fetcher(args.genome, args.twobittofa, genome_build=args.genome_build)

    # 2. Parse GTF for 5'UTR exons
    print(f"\n[1] Parsing GTF: {args.gtf}")
    utr5_by_enst = parse_gtf_5utr(Path(args.gtf))
    print(f"  {len(utr5_by_enst)} transcripts with 5'UTR")

    # 3. Read CSVs and aggregate UMI counts
    print(f"\n[2] Reading HEK CSV: {args.hek_csv}")
    df_hek = pd.read_csv(args.hek_csv, compression="gzip")
    hek_sample_cols = [c for c in df_hek.columns if c != "SeqID"]
    print(f"  {len(df_hek)} rows, {len(hek_sample_cols)} sample cols")

    print(f"\n[3] Reading Vglut CSV: {args.vglut_csv}")
    df_vglut = pd.read_csv(args.vglut_csv, compression="gzip")
    vglut_sample_cols = [c for c in df_vglut.columns if c != "SeqID"]
    print(f"  {len(df_vglut)} rows, {len(vglut_sample_cols)} sample cols")

    # Build condition map for Vglut
    vglut_cond_map = build_vglut_condition_map(vglut_sample_cols)
    print(f"  Vglut condition map: {len(vglut_cond_map)} cols mapped")

    # Aggregate UMI counts
    print(f"\n[4] Aggregating UMI counts")
    hek_umi = aggregate_umi_counts(df_hek, hek_sample_cols)
    vglut_umi = aggregate_umi_counts(df_vglut, vglut_sample_cols)
    print(f"  HEK: {len(hek_umi)} unique (variant, ENST) pairs")
    print(f"  Vglut: {len(vglut_umi)} unique (variant, ENST) pairs")

    # Merge unique variant keys from both datasets
    all_keys = set(hek_umi.keys()) | set(vglut_umi.keys())
    print(f"  Combined: {len(all_keys)} unique (variant, ENST) pairs")

    # 4b. Prefetch all UTR exon sequences (batch API for Ensembl)
    prefetch_regions = []
    for key in all_keys:
        chrom, pos, ref, alt, enst = key
        enst_base = enst.split(".")[0]
        utr5_info = utr5_by_enst.get(enst_base)
        if utr5_info is None:
            continue
        for start, end in utr5_info["utr5_exons"]:
            prefetch_regions.append((utr5_info["chrom"], start - 1, end))
    gf.prefetch(prefetch_regions)

    # 4. Reconstruct sequences
    print(f"\n[5] Reconstructing sequences")
    output_records = []
    stats = {
        "ok": 0,
        "skip_no_utr5": 0,
        "skip_variant_outside_utr5": 0,
        "skip_ref_mismatch": 0,
    }

    for key in sorted(all_keys):
        chrom, pos, ref, alt, enst = key
        enst_base = enst.split(".")[0]

        # Look up 5'UTR exons
        utr5_info = utr5_by_enst.get(enst_base)
        if utr5_info is None:
            stats["skip_no_utr5"] += 1
            continue

        utr5_exons = utr5_info["utr5_exons"]
        strand = utr5_info["strand"]

        # Apply variant on + strand
        result = apply_variant_at_genomic_pos(
            gf, chrom, utr5_exons, pos, ref, alt
        )
        if result is None:
            stats["skip_variant_outside_utr5"] += 1
            continue

        wt_plus, mut_plus, var_offset = result

        # RC if strand == '-'
        if strand == "-":
            wt_seq = reverse_complement(wt_plus)
            mut_seq = reverse_complement(mut_plus)
        else:
            wt_seq = wt_plus
            mut_seq = mut_plus

        wt_seq = normalize_seq(wt_seq)
        mut_seq = normalize_seq(mut_seq)

        if not wt_seq or not mut_seq or wt_seq == mut_seq:
            stats["skip_ref_mismatch"] += 1
            continue

        # Compute labels
        labels = {}
        hek_counts = hek_umi.get(key, {})
        if hek_counts:
            hek_labels = compute_te_labels(
                hek_counts.get("REF", {}), hek_counts.get("ALT", {}),
                hek_sample_cols, condition_map=None,
            )
            for k, v in hek_labels.items():
                labels[f"hek_{k}"] = v

        vglut_counts = vglut_umi.get(key, {})
        if vglut_counts:
            vglut_labels = compute_te_labels(
                vglut_counts.get("REF", {}), vglut_counts.get("ALT", {}),
                vglut_sample_cols, condition_map=vglut_cond_map,
            )
            labels.update(vglut_labels)

        stats["ok"] += 1

        # Build variant key string
        var_key = f"{chrom}:{pos}_{ref}>{alt}"

        record = {
            "record_id": f"GSE246381_{enst_base}_{var_key}",
            "variant_name": var_key,
            "enst": enst_base,
            "source_sequence": wt_seq,
            "candidate_sequence": mut_seq,
            "region": "5'UTR",
            "variant_type": "snv" if len(ref) == len(alt) == 1 else "indel",
            "labels": labels,
            "metadata": {
                "chrom": chrom,
                "variant_pos": pos,
                "ref_allele": ref,
                "alt_allele": alt,
                "strand": strand,
                "cds_start": utr5_info["cds_start"],
                "cds_end": utr5_info["cds_end"],
                "utr5_exons": utr5_exons,
                "variant_position": var_offset,
                "utr5_length": len(wt_seq),
                "has_hek_data": bool(hek_counts),
                "has_vglut_data": bool(vglut_counts),
            },
        }
        output_records.append(record)

    # 5. Write output
    print(f"\n[6] Writing {len(output_records)} records to {args.output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Print stats
    print(f"\n=== Statistics ===")
    print(f"  Total (variant, ENST) pairs: {len(all_keys)}")
    print(f"  Successfully reconstructed: {stats['ok']}")
    print(f"  Skipped (no 5'UTR in GTF): {stats['skip_no_utr5']}")
    print(f"  Skipped (variant outside 5'UTR): {stats['skip_variant_outside_utr5']}")
    print(f"  Skipped (ref mismatch): {stats['skip_ref_mismatch']}")
    total_skip = sum(v for k, v in stats.items() if k != "ok")
    if stats["ok"] + total_skip > 0:
        print(f"  Success rate: {stats['ok']/(stats['ok']+total_skip)*100:.1f}%")

    by_type = {}
    for rec in output_records:
        by_type[rec["variant_type"]] = by_type.get(rec["variant_type"], 0) + 1
    print(f"  By variant type: {by_type}")

    print(f"\nDone. Output: {args.output}")


if __name__ == "__main__":
    main()
