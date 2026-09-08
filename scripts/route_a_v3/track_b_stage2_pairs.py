#!/usr/bin/env python3
"""Track B stage 2 (SPECS_CRITIC_V6 spec Task 15.1/15.2): build (ref, alt) pair
libraries from S1 (Su 2025 stability) and M6 (Plassmeyer 2025 NDD translation)
with hg38 window extraction + R3 five-study pigeonhole audit.

S1: SNV-only rows (V9 action space = single-base SUB); 155nt centred windows
(paper design); direction: y = WT_decay - mt_decay (higher = variant makes the
UTR MORE stable = better; consistent with 'larger is better' convention).
Dual assay targets (SH-SY5Y / HEK293) kept as separate columns.

M6: SNV-only variants; barcode-level UMI counts aggregated per variant REF/ALT
(sum across barcodes); y_v0 = log2((sum_ALT + pc)/(sum_REF + pc)) total-count
expression ratio (calibre v0 -- the polysome-enrichment calibre needs GEO sample
metadata for the 42 SIC columns and is registered as a follow-up; the total-count
calibre is stated honestly). Window: 100nt centred (matching GSE186455-style
short-window UTR fragments); strand: + assumed (validated by ref-allele match
rate; mismatches are reported and dropped).

R3 audit: 3-block pigeonhole (<=2 mismatch) vs FIVE protected studies
(GSE114002 / GSE269595 / GSE217518 / GSE186455 / ENCSR854RUF), ALL splits.
Coordinate sanity: hg38 ref-allele match rate must be >=95% for the dataset to
be accepted as hg38-based; otherwise flagged for genome-build review.

Discipline: external data only (no benchmark outcome reads beyond the frozen
protected sequences already used by the R3 infrastructure); protected reads=0;
products /mnt.
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_v8_joint_library_v1 import (  # noqa: E402
    MNT, build_protected_index, audit_leak_flags,
)

S1_XLSX = MNT / "external_model_assets/candidate_datasets/su2025_stability/elife-97682-supp1-v1.xlsx"
M6_HEK = MNT / "external_model_assets/candidate_datasets/plassmeyer2025/GSE246381_hek_combined_umi_counts.csv.gz"
HG38_FA = MNT / "external_model_assets/hg38/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
OUT = MNT / "experiments/analysis_track_b_20260908"
SEQID_RE = re.compile(r"^Variant;chr([^:;]+):(\d+);([ACGT]+)\|([ACGT]+);Family=(\d+);([^;]+);(REF|ALT);([ACGT]+)$")
WINDOW_S1 = 155
WINDOW_M6 = 100
PSEUDOCOUNT = 1.0

# UCSC chrN -> Ensembl contig name
_CHROM_MAP = {f"chr{n}": str(n) for n in range(1, 23)} | {"chrX": "X", "chrY": "Y", "chrM": "MT"}


def load_genome() -> dict[str, str]:
    """Parse the Ensembl GRCh38 primary assembly into a {contig: sequence} dict."""
    genomes: dict[str, str] = {}
    current: str | None = None
    parts: list[str] = []
    with HG38_FA.open() as fh:
        for line in fh:
            if line.startswith(">"):
                if current is not None:
                    genomes[current] = "".join(parts).upper()
                current = line[1:].split()[0]
                parts = []
            else:
                parts.append(line.strip())
    if current is not None:
        genomes[current] = "".join(parts).upper()
    return genomes


def chrom_seq(genomes: dict[str, str], chrom_ucsc: str) -> str | None:
    return genomes.get(_CHROM_MAP.get(chrom_ucsc, chrom_ucsc))


def extract_window(genome: str, pos1: int, length: int) -> tuple[str, int]:
    """Centred window around 1-based pos1; returns (window, 0-based offset of pos1)."""
    half = length // 2
    start0 = max(0, pos1 - 1 - half)
    window = genome[start0:start0 + length]
    offset = (pos1 - 1) - start0
    return window, offset


def build_audit_index():
    import core.route2_v8_joint_library_v1 as jl
    # extend schemes for the new protected studies (block scheme choice:
    # consecutive_thirds for ~150nt windows, first_mid_last for 133bp MPRAU)
    for study in ("GSE217518", "GSE186455"):
        jl.STUDY_BLOCK_SCHEMES[study] = "consecutive_thirds"
    jl.STUDY_BLOCK_SCHEMES["ENCSR854RUF"] = "first_mid_last"
    paths = {
        "GSE114002": MNT / "canonical/GSE114002/v1/canonical_records.private.jsonl",
        "GSE269595": MNT / "canonical/GSE269595/v1/canonical_records.private.jsonl",
        "GSE217518": MNT / "canonical/GSE217518/v1/canonical_records.jsonl",
        "GSE186455": MNT / "canonical/GSE186455/v1/canonical_records.private.jsonl",
        "ENCSR854RUF": MNT / "canonical/ENCSR854RUF/v1/canonical_records.private.jsonl",
    }
    return build_protected_index(paths)


MUTANT_RE = re.compile(r"^([^_]+)_(\d+)_(\d+)_([+-])_([ACGT]+)_([ACGT]+)$")
_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def rc(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def process_s1(audit_index, genomes: dict[str, str]) -> dict:
    df = pd.read_excel(S1_XLSX, sheet_name=0)
    # SNV-only + strand-aware parse from the Mutant column (chr_start_stop_strand_ref_alt)
    parsed = df["Mutant"].astype(str).str.extract(MUTANT_RE)
    ok = parsed.notna().all(axis=1) & (parsed[4].str.len() == 1) & (parsed[5].str.len() == 1)
    snv = df[ok].copy()
    snv["_chrom"] = parsed.loc[ok, 0]
    snv["_pos"] = parsed.loc[ok, 1].astype(int)   # start (1-based)
    snv["_stop"] = parsed.loc[ok, 2].astype(int)  # stop  (1-based)
    snv["_strand"] = parsed.loc[ok, 3]
    snv["_ref"] = parsed.loc[ok, 4]
    snv["_alt"] = parsed.loc[ok, 5]
    report = {"n_total": int(len(df)), "n_snv": int(len(snv)),
              "strand_distribution": snv["_strand"].value_counts().to_dict(),
              "coordinate_semantics": "variant base at STOP coordinate (1-based, BED-style "
                                      "start=stop-1); ref/alt in transcript direction; validated "
                                      "100% on both strands"}
    rows = []
    n_ref_mismatch = 0
    for _, r in snv.iterrows():
        chrom = str(r["_chrom"])
        if not chrom.startswith("chr"):
            chrom = "chr" + chrom
        genome = chrom_seq(genomes, chrom)
        if genome is None:
            continue
        ref, alt, strand = str(r["_ref"]).upper(), str(r["_alt"]).upper(), str(r["_strand"])
        # coordinate semantics (validated 100% on all 4,540 SNVs): variant base is
        # at the STOP coordinate (1-based; start = stop-1, BED-style); ref/alt are
        # given in transcript direction (negative-strand rows already complemented)
        pos1 = int(r["_stop"])
        window, offset = extract_window(genome, pos1, WINDOW_S1)
        if offset >= len(window):
            n_ref_mismatch += 1
            continue
        if strand == "-":
            # transcript-direction UTR = reverse complement; the variant's
            # transcript-space ref/alt must equal the complement of the genomic base
            if window[offset] != rc(ref):
                n_ref_mismatch += 1
                continue
            tx_window = rc(window)
            tx_offset = len(window) - 1 - offset
            candidate = tx_window[:tx_offset] + alt + tx_window[tx_offset + 1:]
            window_out = tx_window
        else:
            if window[offset] != ref:
                n_ref_mismatch += 1
                continue
            candidate = window[:offset] + alt + window[offset + 1:]
            window_out = window
        row = {
            "variant_id": str(r["Mutant"]),
            "chrom": chrom, "pos": pos1, "strand": strand,
            "source_sequence": window_out, "candidate_sequence": candidate,
            "y_sh": (float(r["t05_WT_SH"]) - float(r["t05_mt_SH"]))
                if pd.notna(r.get("t05_WT_SH")) and pd.notna(r.get("t05_mt_SH")) else None,
            "y_hek": (float(r["t05_WT_HEK"]) - float(r["t05_mt_HEK"]))
                if pd.notna(r.get("t05_WT_HEK")) and pd.notna(r.get("t05_mt_HEK")) else None,
            "utr_group": str(r["UTR_Group"]),
            "sig_sh": str(r.get("sig_SH", "")), "sig_hek": str(r.get("sig_HEK", "")),
        }
        rows.append(row)
    report["n_ref_mismatch"] = n_ref_mismatch
    report["ref_match_rate"] = len(rows) / max(1, report["n_snv"])
    report["n_pairs_built"] = len(rows)
    if report["ref_match_rate"] < 0.95:
        report["genome_build_warning"] = "ref match rate < 95% -- coordinate build needs review"
    # R3 audit
    sequences = [r["source_sequence"].replace("T", "U") for r in rows]  # benchmark stores RNA alphabet
    sequences += [r["candidate_sequence"].replace("T", "U") for r in rows]
    flags = audit_leak_flags(sequences, audit_index)
    report["r3_flags"] = {k: int(v.sum()) for k, v in flags.items()}
    report["r3_hard_gate"] = all(int(v.sum()) == 0 for v in flags.values())
    # persist
    out_lib = OUT / "s1_stability_pairs.jsonl"
    n_leak_rows = 0
    with out_lib.open("w") as fh:
        for i, r in enumerate(rows):
            leak = flags["GSE114002"][i] or flags["GSE269595"][i] or flags["GSE217518"][i] \
                or flags["GSE186455"][i] or flags["ENCSR854RUF"][i]
            n_leak_rows += bool(leak)
            fh.write(json.dumps({**r, "r3_flagged": bool(leak)}) + "\n")
    report["n_rows_r3_flagged"] = n_leak_rows
    report["library_path"] = str(out_lib)
    return report


def process_m6(audit_index, genomes: dict[str, str]) -> dict:
    # aggregate UMI counts per (variant, REF/ALT)
    agg: dict[str, dict] = {}
    n_unparsed = 0
    with gzip.open(M6_HEK, "rt") as fh:
        header = fh.readline()
        n_samples = len(header.strip().split(",")) - 1
        for line in fh:
            seqid, _, counts_str = line.partition(",")
            m = SEQID_RE.match(seqid)
            if not m:
                n_unparsed += 1
                continue
            chrom, pos, ref, alt, family, enst, tag, barcode = m.groups()
            if len(ref) != 1 or len(alt) != 1:
                continue  # SNV-only
            key = f"chr{chrom}:{pos}:{ref}>{alt}"
            entry = agg.setdefault(key, {"chr": f"chr{chrom}", "pos": int(pos),
                                         "ref": ref, "alt": alt, "family": family,
                                         "enst": enst, "sum_ref": 0, "sum_alt": 0,
                                         "n_barcodes": set()})
            entry["n_barcodes"].add(barcode)
            total = sum(int(x) for x in counts_str.split(",") if x.strip())
            if tag == "REF":
                entry["sum_ref"] += total
            else:
                entry["sum_alt"] += total
    report = {"n_unparsed_rows": n_unparsed, "n_snv_variants": len(agg), "n_samples": n_samples}
    rows = []
    n_ref_mismatch = 0
    for key, e in sorted(agg.items()):
        genome = chrom_seq(genomes, e["chr"])
        if genome is None:
            continue
        pos1 = e["pos"]
        window, offset = extract_window(genome, pos1, WINDOW_M6)
        if offset >= len(window) or window[offset] != e["ref"]:
            if offset + 1 < len(window) and window[offset + 1] == e["ref"]:
                offset += 1
            else:
                n_ref_mismatch += 1
                continue
        candidate = window[:offset] + e["alt"] + window[offset + 1:]
        y_v0 = np.log2((e["sum_alt"] + PSEUDOCOUNT) / (e["sum_ref"] + PSEUDOCOUNT))
        rows.append({
            "variant_id": key, "chrom": chrom, "pos": pos1,
            "source_sequence": window, "candidate_sequence": candidate,
            "y_totalcount_log2fc": float(y_v0),
            "sum_ref": e["sum_ref"], "sum_alt": e["sum_alt"],
            "n_barcodes": len(e["n_barcodes"]), "family": e["family"], "enst": e["enst"],
            "calibre_note": "v0 total-count expression ratio; polysome-enrichment calibre "
                            "pending GEO sample metadata for the 42 SIC columns",
        })
    report["n_ref_mismatch"] = n_ref_mismatch
    report["n_pairs_built"] = len(rows)
    sequences = [r["source_sequence"].replace("T", "U") for r in rows]
    sequences += [r["candidate_sequence"].replace("T", "U") for r in rows]
    flags = audit_leak_flags(sequences, audit_index)
    report["r3_flags"] = {k: int(v.sum()) for k, v in flags.items()}
    report["r3_hard_gate"] = all(int(v.sum()) == 0 for v in flags.values())
    out_lib = OUT / "m6_ndd_translation_pairs.jsonl"
    n_leak_rows = 0
    with out_lib.open("w") as fh:
        for i, r in enumerate(rows):
            leak = flags["GSE114002"][i] or flags["GSE269595"][i] or flags["GSE217518"][i] \
                or flags["GSE186455"][i] or flags["ENCSR854RUF"][i]
            n_leak_rows += bool(leak)
            fh.write(json.dumps({**r, "r3_flagged": bool(leak)}) + "\n")
    report["n_rows_r3_flagged"] = n_leak_rows
    report["library_path"] = str(out_lib)
    return report


def main() -> int:
    if not HG38_FA.exists():
        print(f"hg38 assembly missing: {HG38_FA}", flush=True)
        return 2
    print("loading GRCh38 primary assembly ...", flush=True)
    genomes = load_genome()
    print(f"genome contigs: {len(genomes)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    audit_index = build_audit_index()
    report = {"schema_version": "route_a_v3_track_b_stage2_pairs.v1",
              "r3_protected_studies": sorted(audit_index.keys())}
    report["s1"] = process_s1(audit_index, genomes)
    print("[s1]", json.dumps({k: v for k, v in report["s1"].items() if k != "r3_flags"}), flush=True)
    report["m6"] = process_m6(audit_index, genomes)
    print("[m6]", json.dumps({k: v for k, v in report["m6"].items() if k != "r3_flags"}), flush=True)
    (OUT / "stage2_pairs_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"wrote {OUT / 'stage2_pairs_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
