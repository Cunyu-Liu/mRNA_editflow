#!/usr/bin/env python
"""D1-03: Reconstruct GSE261709 (gastric-cancer immune-escape 3'UTR MPRA) pairs.

GSE261709 (Miliotis et al. 2024, Nat Commun): MPRA of 3'UTR cis-eQTL variants
in AGS and SNU719 gastric-cancer cell lines. Design: each 150-nt oligo contains
the reference or alternative allele flanked by 50 nt of reference transcript
sequence (101-nt variant region; 8-nt barcode; 20-nt vector adaptors). The GEO
RAW.tar ships only barcode counts; the per-variant measured effects are in the
paper's Supplementary Data 4 (41467_2024_48436_MOESM4_ESM.xlsx, sheet4):
  variant_id | ensembl_gene_id | external_gene_name | ENST | AGS_FC |
  SNU719_FC | AGS_pval | SNU719_pval | AGS_adjust | SNU719_adjust

We reconstruct the 101-nt variant region from the hg38 reference genome with
the variant centered (50-nt flanks), verify the reference allele against hg38,
and use the published AGS/SNU719 fold-changes (alt/ref activity ratio) together
with their p-values as the measured labels. This is real published measured
data combined with reference-genome sequence reconstruction (not fabricated).

Output: data/p0/GSE261709/reconstructed_pairs.jsonl

Contract: utr_editflow_contract_v2 (FROZEN)
Task: D1-03
"""

import argparse
import json
import math
import os
import re
import ssl
import sys
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from genome_fetcher import reverse_complement, normalize_seq  # noqa: E402

INSERT_LEN = 101        # 50 nt + variant + 50 nt
VARIANT_OFFSET = 50     # 0-indexed offset of the variant's first base
VID_RE = re.compile(r"^([^:]+):(\d+)-(\d+)\(([+-])\)_(?P<ref>[ACGT]+)_(?P<alt>[ACGT]*)$")


def parse_variant_id(vid: str):
    m = VID_RE.match(vid.strip())
    if not m:
        return None
    return {
        "chrom": m.group(1),
        "start": int(m.group(2)),
        "end": int(m.group(3)),
        "strand": m.group(4),
        "ref": m.group("ref"),
        "alt": m.group("alt"),
    }


def _f(v):
    try:
        return float(v)
    except Exception:
        return None


def read_data4(xlsx_path: Path):
    """Parse sheet4 (Supplementary Data 4) -> list of variant dicts (deduped)."""
    import zipfile
    import xml.etree.ElementTree as ET

    T = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    z = zipfile.ZipFile(str(xlsx_path))
    root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    ss = ["".join(t.text or "" for t in si.iter(T + "t")) for si in root.findall(T + "si")]
    sroot = ET.fromstring(z.read("xl/worksheets/sheet4.xml"))
    rows = sroot.findall(".//" + T + "sheetData/" + T + "row")

    def cellval(c):
        t = c.get("t"); v = c.find(T + "v")
        if t == "s" and v is not None:
            return ss[int(v.text)]
        return v.text if v is not None else ""

    seen = set()
    variants = []
    for r in rows:
        cells = [cellval(c) for c in r.findall(T + "c")]
        if not cells or not cells[0]:
            continue
        vid = str(cells[0]).strip()
        if vid == "variant_id":
            continue
        parsed = parse_variant_id(vid)
        if parsed is None:
            continue
        if vid in seen:
            continue  # dedup multi-ENST rows (same variant_id)
        seen.add(vid)
        variants.append({
            "variant_id": vid,
            **parsed,
            "gene": str(cells[2]).strip() if len(cells) > 2 else "",
            "ensembl_gene": str(cells[1]).strip() if len(cells) > 1 else "",
            "ags_fc": _f(cells[4]) if len(cells) > 4 else None,
            "snu719_fc": _f(cells[5]) if len(cells) > 5 else None,
            "ags_pval": _f(cells[6]) if len(cells) > 6 else None,
            "snu719_pval": _f(cells[7]) if len(cells) > 7 else None,
        })
    return variants


def build_pairs(variants, fetcher):
    ok, skip_no_verify = [], 0
    for v in variants:
        start = v["start"]
        # hg38 window [start-50, start+50] 1-indexed inclusive (101 nt),
        # variant's first base at 0-indexed offset 50.
        api_start = start - VARIANT_OFFSET            # start-50
        api_end = start + (INSERT_LEN - VARIANT_OFFSET - 1)  # start+50
        window = fetcher.fetch(v["chrom"].replace("chr", ""), api_start - 1, api_end)
        if len(window) != INSERT_LEN:
            skip_no_verify += 1
            continue
        gref = window[VARIANT_OFFSET:VARIANT_OFFSET + len(v["ref"])]
        if gref != v["ref"]:
            skip_no_verify += 1
            continue
        alt_genomic = window[:VARIANT_OFFSET] + v["alt"] + window[VARIANT_OFFSET + len(v["ref"]):]
        if v["strand"] == "-":
            ref_seq = reverse_complement(window)
            alt_seq = reverse_complement(alt_genomic)
        else:
            ref_seq = window
            alt_seq = alt_genomic
        ref_seq = normalize_seq(ref_seq)
        alt_seq = normalize_seq(alt_seq)
        if not ref_seq or not alt_seq or ref_seq == alt_seq:
            skip_no_verify += 1
            continue
        labels = {}
        if v["ags_fc"] is not None:
            labels["AGS_FC"] = v["ags_fc"]
            labels["log2fc_AGS"] = math.log2(v["ags_fc"]) if v["ags_fc"] > 0 else None
        if v["snu719_fc"] is not None:
            labels["SNU719_FC"] = v["snu719_fc"]
            labels["log2fc_SNU719"] = math.log2(v["snu719_fc"]) if v["snu719_fc"] > 0 else None
        if v["ags_pval"] is not None:
            labels["AGS_pval"] = v["ags_pval"]
        if v["snu719_pval"] is not None:
            labels["SNU719_pval"] = v["snu719_pval"]
        labels = {k: val for k, val in labels.items() if val is not None}
        rid = f"{v['chrom'].replace('chr','')}_{v['start']}_{v['ref']}_{v['alt']}"
        ok.append({
            "record_id": f"GSE261709_{rid}",
            "source_sequence": ref_seq,
            "candidate_sequence": alt_seq,
            "region": "3'UTR",
            "variant_type": ("indel" if len(v["ref"]) != len(v["alt"]) else "snv"),
            "labels": labels,
            "metadata": {
                "chrom": v["chrom"],
                "variant_pos": str(v["start"]),
                "gene_symbol": v["gene"],
                "ensembl_gene": v["ensembl_gene"],
                "strand": v["strand"],
                "ref_allele": v["ref"],
                "alt_allele": v["alt"],
                "variant_position": VARIANT_OFFSET,
                "insert_length": INSERT_LEN,
                "variant_id": v["variant_id"],
                "source_file": "41467_2024_48436_MOESM4_ESM.xlsx",
                "source_sheet": "Supplementary Data 4",
                "reconstruction": "hg38_reference_window_centered",
            },
        })
    return ok, skip_no_verify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="data/p0/GSE261709/41467_2024_48436_MOESM4_ESM.xlsx")
    ap.add_argument("--output", default="data/p0/GSE261709/reconstructed_pairs.jsonl")
    ap.add_argument("--genome", default="ensembl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from genome_fetcher import create_fetcher as _cf
    fetcher = _cf(args.genome, genome_build="GRCh38")

    variants = read_data4(Path(args.xlsx))
    if args.limit:
        variants = variants[: args.limit]
    print(f"parsed {len(variants)} unique variants")

    regions = []
    for v in variants:
        api_start = v["start"] - VARIANT_OFFSET
        api_end = v["start"] + (INSERT_LEN - VARIANT_OFFSET - 1)
        regions.append((v["chrom"].replace("chr", ""), api_start - 1, api_end))
    fetcher.prefetch(regions)

    pairs, skip = build_pairs(variants, fetcher)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs)} pairs to {out} (skipped {skip})")


if __name__ == "__main__":
    main()