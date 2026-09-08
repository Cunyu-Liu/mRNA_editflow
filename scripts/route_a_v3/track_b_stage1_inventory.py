#!/usr/bin/env python3
"""Track B (SPECS_CRITIC_V6 spec 2026-09-08 增补 N.4 Task 15): candidate variant
datasets -- stage 1: parse + variant-coordinate inventory.

S1 (Su 2025 eLife 97682 stability, 5,072-6,555 variant pairs, dual assay
SH-SY5Y/HEK293): parse elife-97682-supp1-v1.xlsx -> variant table with
chromosome/start/stop/ref/alt/UTR group + WT/mt decay values per assay.
M6 (Plassmeyer 2025 GSE246381, 15,070 REF/ALT pairs, HEK + vGlut neuron):
parse SeqID -> chr:pos:ref|alt + family/ENST/REF-ALT tag; pair rows.

Output: variant inventory JSON (chromosome needs, coordinate sanity, ref/alt
distributions) -- the prerequisite for the hg38 window extraction + R3 pigeonhole
audit in stage 2. No training use; protected reads = 0 (external data only).
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import pandas as pd

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
CAND = MNT / "external_model_assets/candidate_datasets"
S1_XLSX = CAND / "su2025_stability/elife-97682-supp1-v1.xlsx"
M6_HEK = CAND / "plassmeyer2025/GSE246381_hek_combined_umi_counts.csv.gz"
M6_VGLUT = CAND / "plassmeyer2025/GSE246381_vglut_combined_umi_counts.csv.gz"
OUT = MNT / "experiments/analysis_track_b_20260908"
SEQID_RE = re.compile(r"^Variant;chr([^:;]+):(\d+);([ACGT]+)\|([ACGT]+);Family=(\d+);([^;]+);(REF|ALT);([ACGT]+)$")


def parse_s1() -> dict:
    df = pd.read_excel(S1_XLSX, sheet_name=0)
    report = {
        "source": str(S1_XLSX),
        "n_rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
    }
    # columns per the 2026-09-08 survey: Mutant/variant_name/GeneSymbol/Chromosome/
    # Start/Stop/UTR_Group/ReferenceAllele/AlternateAllele/t05_WT_SH/t05_mt_SH/
    # pval_SH/sig_SH/t05_WT_HEK/t05_mt_HEK/pval_HEK/sig_HEK/GC*/TA*
    chrom = df["Chromosome"].astype(str)
    report["chromosomes"] = sorted(chrom.unique().tolist())
    report["utr_groups"] = df["UTR_Group"].astype(str).value_counts().to_dict() \
        if "UTR_Group" in df else {}
    # ref/alt sanity: multi-base indels?
    ref = df["ReferenceAllele"].astype(str)
    alt = df["AlternateAllele"].astype(str)
    report["ref_len_distribution"] = ref.str.len().value_counts().to_dict()
    report["alt_len_distribution"] = alt.str.len().value_counts().to_dict()
    # decay deltas
    for assay in ("SH", "HEK"):
        wt = pd.to_numeric(df.get(f"t05_WT_{assay}"), errors="coerce")
        mt = pd.to_numeric(df.get(f"t05_mt_{assay}"), errors="coerce")
        if wt is not None and mt is not None:
            delta = mt - wt
            report[f"delta_{assay}"] = {
                "n_finite": int(delta.notna().sum()),
                "mean": float(delta.mean()) if delta.notna().any() else None,
                "std": float(delta.std()) if delta.notna().any() else None,
                "n_abs_gt_0.1": int((delta.abs() > 0.1).sum()),
            }
    # start/stop span -> window size needs
    start = pd.to_numeric(df["Start"], errors="coerce")
    stop = pd.to_numeric(df["Stop"], errors="coerce")
    span = (stop - start).astype(float)
    report["span_stats"] = {"min": float(span.min()), "max": float(span.max()),
                            "mean": float(span.mean()), "median": float(span.median())}
    return report


def parse_m6(path: Path) -> dict:
    variants = {}
    n_bad = 0
    with gzip.open(path, "rt") as fh:
        header = fh.readline().strip().split(",")
        for line in fh:
            seqid = line.split(",", 1)[0]
            m = SEQID_RE.match(seqid)
            if not m:
                n_bad += 1
                continue
            chrom, pos, ref, alt, family, enst, tag, barcode = m.groups()
            key = f"chr{chrom}:{pos}:{ref}>{alt}|{family}|{enst}"
            entry = variants.setdefault(key, {"chr": f"chr{chrom}", "pos": int(pos),
                                              "ref": ref, "alt": alt, "family": family,
                                              "enst": enst, "barcode": barcode,
                                              "tags": set()})
            entry["tags"].add(tag)
    # pair completeness
    n_complete = sum(1 for v in variants.values() if v["tags"] == {"REF", "ALT"})
    chroms = sorted({v["chr"] for v in variants.values()})
    return {
        "source": str(path),
        "n_rows_total": len(variants) + n_bad,
        "n_unique_variants": len(variants),
        "n_seqid_unparsed": n_bad,
        "n_complete_ref_alt_pairs": n_complete,
        "chromosomes": chroms,
        "ref_len_distribution": {},
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": "route_a_v3_track_b_stage1_inventory.v1"}
    report["s1"] = parse_s1()
    print("[s1]", json.dumps({k: v for k, v in report["s1"].items() if k != "columns"}, default=str)[:800], flush=True)
    report["m6_hek"] = parse_m6(M6_HEK)
    print("[m6_hek]", json.dumps(report["m6_hek"])[:400], flush=True)
    report["m6_vglut"] = parse_m6(M6_VGLUT)
    print("[m6_vglut]", json.dumps(report["m6_vglut"])[:400], flush=True)

    # combined chromosome needs (for targeted hg38 download)
    need = set(report["s1"].get("chromosomes", [])) | set(report["m6_hek"]["chromosomes"]) | set(report["m6_vglut"]["chromosomes"])
    need = {c for c in need if c.startswith("chr") and (c[3:].isdigit() or c in ("chrX", "chrY", "chrM"))}
    report["hg38_chromosomes_needed"] = sorted(need)
    (OUT / "stage1_inventory.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"hg38 chromosomes needed: {sorted(need)}", flush=True)
    print(f"wrote {OUT / 'stage1_inventory.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
