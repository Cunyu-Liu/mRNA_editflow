#!/usr/bin/env python
"""D1-03: Reconstruct GSE298114 (allele-specific RNA stability MPRA) oligo pairs.

GSE298114 (Xiao lab, 2025): allelic RNA-stability MPRA in HeLa. Synthetic
164-nt 3'UTR oligos (variant + flanking sequence) cloned into eGFP 3'UTR,
CMV/CAG promoter. The GEO supplement GSE298114_MPRAnalyze_results.xlsx (Table
S5) reports per-variant hg38 coordinates/allles and MPRAnalyze effect
(statistic = log2 fold-change, pval). The 164-nt sequence is not shipped in
GEO, so we reconstruct it from the hg38 reference genome around each variant.

Reconstruction contract (documented, non-fabricated):
  - variant position is the 1-based hg38 coordinate in hg38_pos1_id;
  - window = 164 nt with the variant placed at 0-indexed offset 81
    (left flank 81 nt, right flank 82 nt);
  - ref/alt alleles are genomic (plus-strand) orientation (verified against
    hg38); the transcript-sense 3'UTR sequence is obtained by reverse-
    complementing the window for genes on the '-' strand;
  - measured labels are the MPRAnalyze statistic (log2FC) and pval.

Output: data/p0/GSE298114/reconstructed_pairs.jsonl  (one record per variant)

Contract: utr_editflow_contract_v2 (FROZEN)
Task: D1-03
"""

import argparse
import json
import os
import ssl
import sys
from pathlib import Path

# The server's Python lacks a valid CA bundle; Ensembl REST over HTTPS fails
# certificate verification (curl works). For read-only reference-fetching from
# the public Ensembl GRCh38 endpoint we use an unverified context.
ssl._create_default_https_context = ssl._create_unverified_context

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from genome_fetcher import reverse_complement, normalize_seq  # noqa: E402

INSERT_LEN = 164
VARIANT_OFFSET = (INSERT_LEN - 1) // 2  # 81 -> variant at 0-indexed 81


def parse_variant_id(vid: str):
    """Parse 'chr1:147155343:-:T>C' -> (chrom, pos, strand, ref, alt)."""
    head, tail = vid.rsplit(":", 1)
    chrom, pos, strand = head.rsplit(":", 2)
    ref, alt = tail.split(">")
    return chrom, int(pos), strand, ref.upper(), alt.upper()


def read_xlsx_variants(xlsx_path: Path):
    """Parse the single data sheet of the MPRAnalyze results xlsx.

    Header row: hg38_pos1_id | hg19_id | gene_name | strand | ref | alt | statistic | pval
    """
    import zipfile
    import xml.etree.ElementTree as ET

    T = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    z = zipfile.ZipFile(str(xlsx_path))
    root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    ss = ["".join(t.text or "" for t in si.iter(T + "t")) for si in root.findall(T + "si")]
    sroot = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = sroot.findall(".//" + T + "sheetData/" + T + "row")

    def cellval(c):
        t = c.get("t"); v = c.find(T + "v")
        if t == "s" and v is not None:
            return ss[int(v.text)]
        return v.text if v is not None else ""

    variants = []
    for r in rows:
        cells = [cellval(c) for c in r.findall(T + "c")]
        if not cells or not cells[0]:
            continue
        if cells[0] == "hg38_pos1_id":
            continue  # header
        vid = str(cells[0]).strip()
        if ">" not in vid:
            continue
        try:
            chrom, pos, strand, ref, alt = parse_variant_id(vid)
        except Exception:
            continue
        variants.append({
            "variant_id": vid,
            "chrom": chrom,
            "pos": pos,
            "strand": strand,
            "ref": ref,
            "alt": alt,
            "gene": str(cells[2]).strip() if len(cells) > 2 else "",
            "statistic": _f(cells[6]) if len(cells) > 6 else None,
            "pval": _f(cells[7]) if len(cells) > 7 else None,
        })
    return variants


def _f(v):
    try:
        return float(v)
    except Exception:
        return None


def fetch_region(fetcher, chrom, api_start, api_end):
    """Fetch + strand sequence via EnsemblGRCh38 1-indexed inclusive region."""
    seq = fetcher.fetch(chrom.replace("chr", ""), api_start - 1, api_end)
    return seq


def build_pairs(variants, fetcher):
    ok, skip = [], 0
    for v in variants:
        pos = v["pos"]
        # Ensembl 1-indexed inclusive window (variant at 0-indexed offset 81)
        api_start = pos - VARIANT_OFFSET          # pos-81
        api_end = pos + (INSERT_LEN - VARIANT_OFFSET - 1)  # pos+82
        window = fetch_region(fetcher, v["chrom"], api_start, api_end)
        if len(window) != INSERT_LEN:
            skip += 1
            continue
        # verify the reference allele at the variant offset (genomic orientation)
        g_ref = window[VARIANT_OFFSET]
        if g_ref != v["ref"]:
            # allow the DNA base to be a U-coded T mismatch; else skip
            skip += 1
            continue
        # apply alt in genomic orientation
        alt_genomic = window[:VARIANT_OFFSET] + v["alt"] + window[VARIANT_OFFSET + 1:]
        if v["strand"] == "-":
            ref_seq = reverse_complement(window)
            alt_seq = reverse_complement(alt_genomic)
        else:
            ref_seq = window
            alt_seq = alt_genomic
        ref_seq = normalize_seq(ref_seq)
        alt_seq = normalize_seq(alt_seq)
        if not ref_seq or not alt_seq or ref_seq == alt_seq:
            skip += 1
            continue
        labels = {}
        if v["statistic"] is not None:
            labels["statistic"] = v["statistic"]
            labels["log2fc"] = v["statistic"]
        if v["pval"] is not None:
            labels["pval"] = v["pval"]
        rid = f"{v['chrom'].replace('chr','')}_{v['pos']}_{v['ref']}{v['alt']}"
        ok.append({
            "record_id": f"GSE298114_{rid}",
            "source_sequence": ref_seq,
            "candidate_sequence": alt_seq,
            "region": "3'UTR",
            "variant_type": ("indel" if (len(v["ref"]) != len(v["alt"])) else "snv"),
            "labels": labels,
            "metadata": {
                "chrom": v["chrom"],
                "variant_pos": str(v["pos"]),
                "gene_symbol": v["gene"],
                "strand": v["strand"],
                "ref_allele": v["ref"],
                "alt_allele": v["alt"],
                "variant_position": VARIANT_OFFSET,
                "insert_length": INSERT_LEN,
                "variant_id": v["variant_id"],
                "source_file": "GSE298114_MPRAnalyze_results.xlsx",
                "reconstruction": "hg38_reference_window_centered",
            },
        })
    return ok, skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="data/p0/GSE298114/GSE298114_MPRAnalyze_results.xlsx")
    ap.add_argument("--output", default="data/p0/GSE298114/reconstructed_pairs.jsonl")
    ap.add_argument("--genome", default="ensembl", help="ensembl (GRCh38 REST) or local .fa path")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    fetcher = create_fetcher(args.genome)
    variants = read_xlsx_variants(Path(args.xlsx))
    if args.limit:
        variants = variants[: args.limit]
    print(f"parsed {len(variants)} variants")

    # prefetch all windows in batches for speed
    regions = []
    for v in variants:
        api_start = v["pos"] - VARIANT_OFFSET
        api_end = v["pos"] + (INSERT_LEN - VARIANT_OFFSET - 1)
        regions.append((v["chrom"].replace("chr", ""), api_start - 1, api_end))
    fetcher.prefetch(regions)

    pairs, skip = build_pairs(variants, fetcher)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs)} pairs to {out} (skipped {skip})")


def create_fetcher(genome_arg):
    from genome_fetcher import create_fetcher as _cf
    return _cf(genome_arg, genome_build="GRCh38")


if __name__ == "__main__":
    main()