#!/usr/bin/env python
"""D1-03: Reconstruct oligo sequences for GSE232572 (MapUTR) from FASTA.

GSE232572 (MapUTR, Fu et al. 2024 Nat Commun) contains ~9,343 paired 3'UTR
variants with MPRA activity data (mRNA abundance in HeLa, 3 replicates).

The FASTA files (C4Sp1/2/3.fasta.gz) contain 200nt oligos with both reference
and alternate sequences for each variant. Headers encode variant info:
  >subpool1|COSMIC|chrX:70361273|MED12|+|alternate|T|rc

Fields: subpool|source|chr:pos|gene|strand|ref_or_alt|allele|orientation

The RAW.tar contains 18 TXT files with DNA/RNA counts (3 subpools x DNA/RNA
x 3 replicates), tab-separated: gene_header<TAB>count.

This script:
1. Parses FASTA files to extract ref/alt pairs
2. Strips adapter sequences to get 165nt insert (variant at position 82)
3. Normalizes rc sequences to forward (sense) orientation
4. Parses RAW.tar TXT files for DNA/RNA counts
5. Computes mRNA activity (RNA/DNA) and log2FC(alt/ref)
6. Writes reconstructed JSONL

Adapter structure (200nt oligo):
  orig: [21nt prefix] + [165nt insert] + [14nt suffix]
  rc:   [14nt prefix] + [165nt insert_rc] + [21nt suffix]

Output: JSONL with one record per paired variant.

Contract: utr_editflow_contract_v2 (FROZEN)
Task: D1-03
"""

import argparse
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

# Make genome_fetcher importable for reverse_complement
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from genome_fetcher import reverse_complement, normalize_seq  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ORIG_PREFIX_LEN = 21
ORIG_SUFFIX_LEN = 14
RC_PREFIX_LEN = 14
RC_SUFFIX_LEN = 21
INSERT_LEN = 165  # 200 - 21 - 14 = 165, also 200 - 14 - 21 = 165
VARIANT_POS_IN_INSERT = 82  # 0-indexed, center of 165nt insert


# ---------------------------------------------------------------------------
# FASTA parsing
# ---------------------------------------------------------------------------

def parse_fasta_header(header: str) -> Optional[dict]:
    """Parse a GSE232572 FASTA header.

    Format: subpool1|COSMIC|chrX:70361273|MED12|+|alternate|T|rc

    Returns dict with keys: subpool, source, chr_pos, gene, strand,
    allele_type (reference/alternate), allele, orientation (orig/rc).
    """
    parts = header.split("|")
    if len(parts) < 8:
        return None
    orientation = parts[7].strip().lower()
    # Handle data anomaly like "orig;subpool1"
    if orientation.startswith("orig"):
        orientation = "orig"
    elif orientation.startswith("rc"):
        orientation = "rc"
    else:
        return None
    return {
        "subpool": parts[0],
        "source": parts[1],
        "chr_pos": parts[2],
        "gene": parts[3],
        "strand": parts[4],
        "allele_type": parts[5].strip().lower(),
        "allele": parts[6].strip().upper(),
        "orientation": orientation,
    }


def extract_insert(seq: str, orientation: str) -> str:
    """Strip adapter sequences and return 165nt insert in sense orientation.

    For 'orig': insert = seq[21:186] (already sense)
    For 'rc':   insert = revcomp(seq[14:179]) (convert to sense)
    """
    if orientation == "orig":
        return seq[ORIG_PREFIX_LEN : ORIG_PREFIX_LEN + INSERT_LEN]
    elif orientation == "rc":
        rc_insert = seq[RC_PREFIX_LEN : RC_PREFIX_LEN + INSERT_LEN]
        return reverse_complement(rc_insert)
    return ""


def parse_fasta_file(path: Path) -> Dict[str, dict]:
    """Parse a FASTA file, return {header: {parsed_info, seq, insert}}."""
    result = {}
    open_fn = gzip.open if path.suffix == ".gz" else open
    with open_fn(path, "rt") as f:
        header = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                header = line[1:]
            elif header:
                parsed = parse_fasta_header(header)
                if parsed is not None:
                    insert = extract_insert(line, parsed["orientation"])
                    result[header] = {
                        "parsed": parsed,
                        "seq": line,
                        "insert": insert,
                    }
                header = None
    return result


def build_pairs(seqs_by_header: Dict[str, dict]) -> List[dict]:
    """Match reference/alternate pairs by (subpool, source, chr_pos, gene, strand).

    Returns list of dicts with ref/alt info.
    """
    pairs = {}
    for header, info in seqs_by_header.items():
        p = info["parsed"]
        key = (p["subpool"], p["source"], p["chr_pos"], p["gene"], p["strand"])
        if key not in pairs:
            pairs[key] = {}
        pairs[key][p["allele_type"]] = {
            "header": header,
            "insert": info["insert"],
            "seq": info["seq"],
            "orientation": p["orientation"],
            "allele": p["allele"],
        }

    result = []
    for key, alleles in pairs.items():
        if "reference" in alleles and "alternate" in alleles:
            subpool, source, chr_pos, gene, strand = key
            result.append({
                "subpool": subpool,
                "source": source,
                "chr_pos": chr_pos,
                "gene": gene,
                "strand": strand,
                "ref": alleles["reference"],
                "alt": alleles["alternate"],
            })
    return result


# ---------------------------------------------------------------------------
# RAW.tar count parsing
# ---------------------------------------------------------------------------

def parse_raw_tar_counts(tar_path: Path) -> Dict[str, Dict[str, dict]]:
    """Parse GSE232572_RAW.tar to extract per-allele DNA/RNA counts.

    File naming: GSM{id}_C4Sp{N}{D/R}{M}.txt.gz
      N = subpool (1-3), D=DNA/R=RNA, M=replicate (1-3)

    Each file is tab-separated: gene_header<TAB>count

    Returns: {(subpool, header): {"DNA": {rep: count}, "RNA": {rep: count}}}
    """
    counts = {}
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".txt.gz"):
                continue
            # Parse filename: GSM{id}_C4Sp{N}{D/R}{M}.txt.gz
            basename = os.path.basename(member.name)
            m = re.match(r"GSM\d+_C4Sp(\d)([DR])(\d)\.txt\.gz", basename)
            if not m:
                continue
            subpool_num = m.group(1)
            dr_type = "DNA" if m.group(2) == "D" else "RNA"
            rep_num = int(m.group(3))

            f = tar.extractfile(member)
            if f is None:
                continue
            with gzip.open(io.BytesIO(f.read()), "rt") as gz:
                next(gz)  # skip header line "gene\tcount"
                for line in gz:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 2:
                        continue
                    header = parts[0]
                    try:
                        count = float(parts[1])
                    except ValueError:
                        continue
                    # Normalize subpool name: C4Sp1 -> subpool1
                    subpool_key = f"subpool{subpool_num}"
                    ck = (subpool_key, header)
                    if ck not in counts:
                        counts[ck] = {"DNA": {}, "RNA": {}}
                    counts[ck][dr_type][rep_num] = count

    return counts


def compute_labels(
    ref_header: str,
    alt_header: str,
    subpool: str,
    counts: Dict,
) -> dict:
    """Compute expression labels for a ref/alt pair.

    Labels:
      dna_ref_rep{1,2,3}, dna_alt_rep{1,2,3}
      rna_ref_rep{1,2,3}, rna_alt_rep{1,2,3}
      activity_ref_rep{1,2,3}, activity_alt_rep{1,2,3}
      activity_ref_mean, activity_alt_mean
      log2fc_activity
    """
    labels = {}
    ref_counts = counts.get((subpool, ref_header), {"DNA": {}, "RNA": {}})
    alt_counts = counts.get((subpool, alt_header), {"DNA": {}, "RNA": {}})

    activity_ref_vals = []
    activity_alt_vals = []

    for rep in (1, 2, 3):
        dna_r = ref_counts["DNA"].get(rep, 0)
        dna_a = alt_counts["DNA"].get(rep, 0)
        rna_r = ref_counts["RNA"].get(rep, 0)
        rna_a = alt_counts["RNA"].get(rep, 0)

        labels[f"dna_ref_rep{rep}"] = dna_r
        labels[f"dna_alt_rep{rep}"] = dna_a
        labels[f"rna_ref_rep{rep}"] = rna_r
        labels[f"rna_alt_rep{rep}"] = rna_a

        if dna_r > 0:
            act_r = rna_r / dna_r
            labels[f"activity_ref_rep{rep}"] = act_r
            activity_ref_vals.append(act_r)
        if dna_a > 0:
            act_a = rna_a / dna_a
            labels[f"activity_alt_rep{rep}"] = act_a
            activity_alt_vals.append(act_a)

    if activity_ref_vals:
        labels["activity_ref_mean"] = sum(activity_ref_vals) / len(activity_ref_vals)
    if activity_alt_vals:
        labels["activity_alt_mean"] = sum(activity_alt_vals) / len(activity_alt_vals)

    if (
        "activity_ref_mean" in labels
        and "activity_alt_mean" in labels
        and labels["activity_ref_mean"] > 0
        and labels["activity_alt_mean"] > 0
    ):
        labels["log2fc_activity"] = math.log2(
            labels["activity_alt_mean"] / labels["activity_ref_mean"]
        )

    return labels


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct GSE232572 (MapUTR) paired 3'UTR sequences"
    )
    parser.add_argument(
        "--data-dir",
        default="data/p0/GSE232572",
        help="Directory containing FASTA and RAW.tar files",
    )
    parser.add_argument(
        "--output",
        default="data/p0/GSE232572/reconstructed_pairs.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print("=== D1-03: Reconstruct GSE232572 (MapUTR) sequences ===")

    # 1. Parse FASTA files
    print(f"\n[1] Parsing FASTA files from {data_dir}")
    all_seqs = {}
    for sp in (1, 2, 3):
        fname = data_dir / f"GSE232572_C4Sp{sp}.fasta.gz"
        if not fname.exists():
            print(f"  WARNING: {fname} not found, skipping")
            continue
        seqs = parse_fasta_file(fname)
        print(f"  {fname.name}: {len(seqs)} sequences")
        all_seqs.update(seqs)
    print(f"  Total: {len(all_seqs)} sequences")

    # 2. Build ref/alt pairs
    print(f"\n[2] Matching ref/alt pairs")
    pairs = build_pairs(all_seqs)
    print(f"  Found {len(pairs)} paired variants")

    # 3. Parse RAW.tar for counts
    tar_path = data_dir / "GSE232572_RAW.tar"
    counts = {}
    if tar_path.exists():
        print(f"\n[3] Parsing RAW.tar counts from {tar_path}")
        counts = parse_raw_tar_counts(tar_path)
        print(f"  Loaded counts for {len(counts)} (subpool, header) entries")
    else:
        print(f"\n[3] WARNING: {tar_path} not found, no labels will be added")

    # 4. Build output records
    print(f"\n[4] Building output records")
    output_records = []
    stats = {
        "ok": 0,
        "skip_identical": 0,
        "skip_multi_diff": 0,
        "skip_no_seq": 0,
        "no_counts": 0,
    }

    for pair in pairs:
        ref_insert = normalize_seq(pair["ref"]["insert"])
        alt_insert = normalize_seq(pair["alt"]["insert"])

        if not ref_insert or not alt_insert:
            stats["skip_no_seq"] += 1
            continue

        if ref_insert == alt_insert:
            stats["skip_identical"] += 1
            continue

        # Verify single difference (at expected variant position)
        diffs = [
            (i, ref_insert[i], alt_insert[i])
            for i in range(min(len(ref_insert), len(alt_insert)))
            if ref_insert[i] != alt_insert[i]
        ]
        if len(diffs) != 1:
            # For indels or complex variants, still include but note
            if len(diffs) == 0:
                stats["skip_identical"] += 1
                continue
            # Multiple diffs — could be multi-base variant or orientation issue
            stats["skip_multi_diff"] += 1
            continue

        var_pos, ref_base, alt_base = diffs[0]
        stats["ok"] += 1

        # Compute labels
        labels = compute_labels(
            pair["ref"]["header"],
            pair["alt"]["header"],
            pair["subpool"],
            counts,
        )
        if not labels:
            stats["no_counts"] += 1

        # Parse chromosome and position
        chr_pos = pair["chr_pos"]
        chrom, pos_str = chr_pos.split(":", 1) if ":" in chr_pos else (chr_pos, "")

        # Build record_id
        var_key = f"{pair['source']}_{chr_pos.replace(':', '_')}_{pair['gene']}"
        rid = f"GSE232572_{pair['subpool']}_{var_key}"

        record = {
            "record_id": rid,
            "source_sequence": ref_insert,
            "candidate_sequence": alt_insert,
            "region": "3'UTR",
            "variant_type": "snv" if len(ref_base) == len(alt_base) == 1 else "indel",
            "labels": labels,
            "metadata": {
                "chrom": chrom,
                "variant_pos": pos_str,
                "gene_symbol": pair["gene"],
                "strand": pair["strand"],
                "variant_source": pair["source"],
                "subpool": pair["subpool"],
                "ref_allele": ref_base,
                "alt_allele": alt_base,
                "variant_position": var_pos,
                "insert_length": len(ref_insert),
                "ref_orientation": pair["ref"]["orientation"],
                "alt_orientation": pair["alt"]["orientation"],
                "ref_header_allele": pair["ref"]["allele"],
                "alt_header_allele": pair["alt"]["allele"],
                "source_file": f"GSE232572_C4Sp{pair['subpool'][-1]}.fasta.gz",
            },
        }
        output_records.append(record)

    # 5. Write output
    print(f"\n[5] Writing {len(output_records)} records to {args.output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Print stats
    print(f"\n=== Statistics ===")
    print(f"  Total paired variants: {len(pairs)}")
    print(f"  Successfully reconstructed: {stats['ok']}")
    print(f"  Skipped (identical): {stats['skip_identical']}")
    print(f"  Skipped (multi-diff): {stats['skip_multi_diff']}")
    print(f"  Skipped (no seq): {stats['skip_no_seq']}")
    print(f"  Without counts: {stats['no_counts']}")

    by_source = {}
    for rec in output_records:
        s = rec["metadata"]["variant_source"]
        by_source[s] = by_source.get(s, 0) + 1
    print(f"  By variant source: {by_source}")

    by_type = {}
    for rec in output_records:
        by_type[rec["variant_type"]] = by_type.get(rec["variant_type"], 0) + 1
    print(f"  By variant type: {by_type}")

    n_with_labels = sum(1 for r in output_records if r["labels"])
    print(f"  With expression labels: {n_with_labels}")

    print(f"\nDone. Output: {args.output}")


if __name__ == "__main__":
    main()
