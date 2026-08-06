#!/usr/bin/env python
"""D1-XX: Reconstruct paired 3'UTR variant sequences for GSE232571 (MapUTR rare/clinical).

GSE232571 (MapUTR, Fu et al. 2024 Nat Commun — rare/clinical/exAC supplement) is the
companion to GSE232572. It contains paired 3'UTR reference/alternate oligos across
three modules:
  C1 = rare_3utr_clinical (2,093 variants + controls)
  C2 = rare_3utr_exac     (5,921 variants + controls)
  C3 = RBP motif variants (ARE / SAMD4A / dPUM / hPUM / sRSM1,CDE)
with DNA and RNA MPRA counts (HEK293 and HeLa cell lines, 3 replicates each).

FASTA files (GSE232571_C{pool}Sp{subpool}.fasta.gz) contain 200nt oligos with both
reference and alternate sequences. Headers encode variant info:
  >subpool1|rare_3utr_clinical|chr11:118006763|SCN4B|-|reference|G|orig

The RAW.tar contains per-library count files (gzip members):
  DNA:  GSM*_C{pool}Sp{subpool}D{rep}.txt.gz        (rep = 1..3)
  RNA:  GSM*_HEK293_C{pool}Sp{subpool}R{rep}.txt.gz
        GSM*_HeLa_C{pool}Sp{subpool}R{rep}.txt.gz   (some members combine two reps with '-')
Each count file is 'gene_header<TAB>count' where gene_header is the FASTA header.

This script:
1. Discovers FASTA files by glob, tags each sequence with its module key (C{pool}Sp{subpool})
2. Strips adapters to get the 165nt insert (21nt prefix / 14nt suffix / rc : 14/21)
3. Builds reference/alternate pairs within each module
4. Parses RAW.tar DNA/RNA counts (HEK293 + HeLa)
5. Computes activity (RNA/DNA) and log2FC(alt/ref) per cell line
6. Writes reconstructed JSONL (one record per paired variant)

Output: data/p0/GSE232571/reconstructed_pairs.jsonl

Contract: utr_editflow_contract_v2 (FROZEN)
Task: D1-XX
"""

import argparse
import glob
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

# Make genome_fetcher importable for reverse_complement / normalize_seq
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    from genome_fetcher import reverse_complement, normalize_seq  # noqa: E402
except Exception:  # pragma: no cover - fallback minimal impl
    def reverse_complement(s: str) -> str:
        comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N", "U": "A"}
        return "".join(comp.get(c.upper(), c) for c in reversed(s))

    def normalize_seq(s: str) -> str:
        return "".join(c for c in s.upper() if c in "ACGTN")


ORIG_PREFIX_LEN = 21
ORIG_SUFFIX_LEN = 14
RC_PREFIX_LEN = 14
RC_SUFFIX_LEN = 21
INSERT_LEN = 165  # 200 - 21 - 14 = 165

REP_COUNT = 3


# ---------------------------------------------------------------------------
# FASTA parsing
# ---------------------------------------------------------------------------

def parse_fasta_header(header: str) -> Optional[dict]:
    """Parse a GSE232571 FASTA header.

    Format: subpool1|rare_3utr_clinical|chr11:118006763|SCN4B|-|reference|G|orig

    Returns dict with keys: subpool, source, chr_pos, gene, strand,
    allele_type (reference/alternate), allele, orientation (orig/rc).
    """
    parts = header.split("|")
    if len(parts) < 8:
        return None
    orientation = parts[7].strip().lower()
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
    open_fn = gzip.open if str(path).endswith(".gz") else open
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

_FILE_RE = re.compile(
    r"GSM\d+_(?P<cell>[A-Za-z0-9]*_)?C(?P<pool>\d)Sp(?P<sub>\d)"
    r"(?P<dr>[DR])(?P<rep>\d)\.txt\.gz"
)


def parse_raw_tar_counts(tar_path: Path) -> Dict[Tuple[str, str], dict]:
    """Parse GSE232571_RAW.tar to extract per-allele DNA/RNA counts.

    DNA members:  GSM*_C{pool}Sp{sub}D{rep}.txt.gz
    RNA members:  GSM*_HEK293_C{pool}Sp{sub}R{rep}.txt.gz
                  GSM*_HeLa_C{pool}Sp{sub}R{rep}.txt.gz
    Some cells combine two reps in one member (e.g. HeLa_C1Sp1R1-C2Sp3R2);
    those members cover two (module, cell, rep) slots; we only safely read the
    first tab-separated header column, so the module-key is parsed from the
    leading 'C{pool}Sp{sub}' token.

    Each file is tab-separated: gene_header<TAB>count.

    Returns: {(module_key, header): {"DNA": {cell: {rep: count}}, "RNA": {...}}}
    """
    counts = {}
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            basename = os.path.basename(member.name)
            if not basename.endswith(".txt.gz"):
                continue
            m = _FILE_RE.search(basename)
            if not m:
                continue
            pool = m.group("pool")
            sub = m.group("sub")
            module = f"C{pool}Sp{sub}"
            dr = "DNA" if m.group("dr") == "D" else "RNA"
            rep = int(m.group("rep"))
            cell = (m.group("cell") or "").rstrip("_") or "DNA"
            if dr == "DNA":
                cell = "DNA"

            f = tar.extractfile(member)
            if f is None:
                continue
            data = f.read()
            try:
                text = gzip.decompress(data).decode("utf-8", errors="replace")
            except Exception:
                continue
            lines = text.splitlines()
            if not lines:
                continue
            for line in lines:
                if "\t" not in line:
                    continue
                header, _, count_s = line.partition("\t")
                header = header.strip()
                try:
                    count = float(count_s)
                except ValueError:
                    continue
                ck = (module, header)
                if ck not in counts:
                    counts[ck] = {"DNA": {}, "RNA": {}}
                counts[ck][dr].setdefault(cell, {})[rep] = count
    return counts


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def compute_labels(
    ref_header: str,
    alt_header: str,
    module: str,
    counts: Dict,
) -> dict:
    """Compute expression labels for a ref/alt pair.

    Labels:
      dna_ref_rep{1,2,3}, dna_alt_rep{1,2,3}
      rna_{cell}_ref_rep{1,2,3}, rna_{cell}_alt_rep{1,2,3}
      activity_{cell}_ref_rep{1,2,3}, activity_{cell}_alt_rep{1,2,3}
      activity_{cell}_ref_mean, activity_{cell}_alt_mean
      log2fc_activity_{cell}
    """
    labels = {}
    ref_counts = counts.get((module, ref_header), {"DNA": {}, "RNA": {}})
    alt_counts = counts.get((module, alt_header), {"DNA": {}, "RNA": {}})

    for cell in ("HEK293", "HeLa"):
        act_r = []
        act_a = []
        for rep in (1, 2, 3):
            dna_r = ref_counts["DNA"].get("DNA", {}).get(rep, 0)
            dna_a = alt_counts["DNA"].get("DNA", {}).get(rep, 0)
            rna_r = ref_counts["RNA"].get(cell, {}).get(rep, 0)
            rna_a = alt_counts["RNA"].get(cell, {}).get(rep, 0)

            labels[f"dna_ref_rep{rep}"] = dna_r
            labels[f"dna_alt_rep{rep}"] = dna_a
            labels[f"rna_{cell}_ref_rep{rep}"] = rna_r
            labels[f"rna_{cell}_alt_rep{rep}"] = rna_a

            if dna_r > 0:
                ar = rna_r / dna_r
                labels[f"activity_{cell}_ref_rep{rep}"] = ar
                act_r.append(ar)
            if dna_a > 0:
                aa = rna_a / dna_a
                labels[f"activity_{cell}_alt_rep{rep}"] = aa
                act_a.append(aa)
        if act_r:
            labels[f"activity_{cell}_ref_mean"] = sum(act_r) / len(act_r)
        if act_a:
            labels[f"activity_{cell}_alt_mean"] = sum(act_a) / len(act_a)
        if (
            f"activity_{cell}_ref_mean" in labels
            and f"activity_{cell}_alt_mean" in labels
            and labels[f"activity_{cell}_ref_mean"] > 0
            and labels[f"activity_{cell}_alt_mean"] > 0
        ):
            labels[f"log2fc_activity_{cell}"] = math.log2(
                labels[f"activity_{cell}_alt_mean"] / labels[f"activity_{cell}_ref_mean"]
            )
    return labels


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct GSE232571 (MapUTR rare/clinical) paired 3'UTR sequences"
    )
    parser.add_argument(
        "--data-dir",
        default="data/p0/GSE232571",
        help="Directory containing FASTA and RAW.tar files",
    )
    parser.add_argument(
        "--output",
        default="data/p0/GSE232571/reconstructed_pairs.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print("=== D1-XX: Reconstruct GSE232571 (MapUTR rare/clinical) sequences ===")

    # 1. Parse FASTA files (discover by glob)
    fasta_files = sorted(glob.glob(str(data_dir / "GSE232571_C*Sp*.fasta.gz")))
    print(f"[1] Found {len(fasta_files)} FASTA files")
    all_seqs = {}
    for fp in fasta_files:
        seqs = parse_fasta_file(Path(fp))
        print(f"  {os.path.basename(fp)}: {len(seqs)} sequences")
        all_seqs.update(seqs)
    print(f"  Total: {len(all_seqs)} sequences")

    # 2. Build ref/alt pairs
    print("\n[2] Matching ref/alt pairs")
    pairs = build_pairs(all_seqs)
    print(f"  Found {len(pairs)} paired variants")

    # 3. Parse RAW.tar for counts
    tar_candidates = [
        data_dir / "GSE232571_RAW.tar",
        data_dir / "RAW.tar",
    ]
    tar_path = next((t for t in tar_candidates if t.exists()), None)
    counts = {}
    if tar_path:
        print(f"\n[3] Parsing RAW.tar counts from {tar_path}")
        counts = parse_raw_tar_counts(tar_path)
        nmod = len({k[0] for k in counts})
        print(f"  Loaded counts for {len(counts)} (module, header) entries across {nmod} modules")
    else:
        print(f"\n[3] WARNING: RAW.tar not found, no labels will be added")

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

        diffs = [
            (i, ref_insert[i], alt_insert[i])
            for i in range(min(len(ref_insert), len(alt_insert)))
            if ref_insert[i] != alt_insert[i]
        ]
        if len(diffs) != 1:
            if len(diffs) == 0:
                stats["skip_identical"] += 1
                continue
            stats["skip_multi_diff"] += 1
            continue

        var_pos, ref_base, alt_base = diffs[0]
        stats["ok"] += 1

        # Determine module key: the module is NOT in the header (subpool is always
        # 'subpool1'); it is encoded in the FASTA file name. We recover it from the
        # count file presence by scanning which module actually carries this header.
        # To keep it deterministic we search the loaded counts for any module that
        # has this exact header; if none, module is '' (no labels).
        module = ""
        for k in counts:
            if k[1] == pair["ref"]["header"]:
                module = k[0]
                break

        labels = compute_labels(
            pair["ref"]["header"],
            pair["alt"]["header"],
            module,
            counts,
        )
        if not labels:
            stats["no_counts"] += 1

        chr_pos = pair["chr_pos"]
        chrom, pos_str = chr_pos.split(":", 1) if ":" in chr_pos else (chr_pos, "")

        var_key = f"{pair['source']}_{chr_pos.replace(':', '_')}_{pair['gene']}"
        rid = f"GSE232571_{var_key}"

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
                "module": module,
                "ref_allele": ref_base,
                "alt_allele": alt_base,
                "variant_position": var_pos,
                "insert_length": len(ref_insert),
                "ref_orientation": pair["ref"]["orientation"],
                "alt_orientation": pair["alt"]["orientation"],
                "ref_header_allele": pair["ref"]["allele"],
                "alt_header_allele": pair["alt"]["allele"],
                "source_file": f"GSE232571_{module}.fasta.gz" if module else "unknown",
            },
        }
        output_records.append(record)

    # 5. Write output
    print(f"\n[5] Writing {len(output_records)} records to {args.output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n=== Statistics ===")
    print(f"  Total paired variants: {len(pairs)}")
    print(f"  Successfully reconstructed: {stats['ok']}")
    print(f"  Skipped (identical): {stats['skip_identical']}")
    print(f"  Skipped (multi-diff): {stats['skip_multi_diff']}")
    print(f"  Skipped (no seq): {stats['skip_no_seq']}")
    print(f"  Without counts: {stats['no_counts']}")

    by_source = {}
    for rec in output_records:
        by_source[rec["metadata"]["variant_source"]] = by_source.get(
            rec["metadata"]["variant_source"], 0
        ) + 1
    print(f"  By variant source: {by_source}")

    by_type = {}
    n_with_labels = 0
    for rec in output_records:
        by_type[rec["variant_type"]] = by_type.get(rec["variant_type"], 0) + 1
        if rec["labels"]:
            n_with_labels += 1
    print(f"  By variant type: {by_type}")
    print(f"  With expression labels: {n_with_labels}")

    print(f"\nDone. Output: {args.output}")


if __name__ == "__main__":
    main()