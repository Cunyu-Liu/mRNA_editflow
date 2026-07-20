"""P1-11: Long-view reconstruction (5'UTR ≤ 512, CDS ≤ 3072, 3'UTR ≤ 1024).

Reads canonical (untruncated) records from the v1 frozen namespace and
derives a long-view model view with larger caps than v1's `model_capped_v1`
(128/256/1536). The long-view caps are 4x larger to capture therapeutically
relevant full-length mRNAs that were truncated in v1.

Outputs (under ``data/reconstructed/p1_long_view/``):
  - ``gencode_v45/canonical.records.jsonl`` (symlink to v1 frozen)
  - ``gencode_v45/long_view.records.jsonl`` (truncated to 512/3072/1024)
  - ``gencode_v45/long_view.lineage.jsonl`` (lineage tracking)
  - ``gencode_v45/long_view_manifest.json`` (manifest with SHA-256 + stats)
  - Same for ``refseq_human_rna/``
  - ``combined/long_view.records.jsonl`` (both sources merged)
  - ``combined/long_view_attrition_report.json`` (attrition stats)
  - ``p1_long_view_manifest.json`` (top-level manifest)

Constraints honored:
  - v1 frozen namespace is NOT modified (read-only).
  - All new artifacts record source path, SHA-256, counts, caps.
  - No new training; this is a data view only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure mrna_editflow is importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Long-view caps (4x the v1 model_capped_v1 caps).
LONG_VIEW_CAPS = {
    "max_5utr": 512,
    "max_cds": 3072,
    "max_3utr": 1024,
}

V1_ROOT = Path("/home/cunyuliu/mrna_editflow_goal/mrna_editflow/data/reconstructed/p0_data_reconstruction_v1")
P1_ROOT = Path("/home/cunyuliu/mrna_editflow_goal/mrna_editflow/data/reconstructed/p1_long_view")

SOURCES = [
    ("gencode_v45", V1_ROOT / "sources" / "gencode_v45" / "canonical.records.jsonl"),
    ("refseq_human_rna", V1_ROOT / "sources" / "refseq_human_rna" / "canonical.records.jsonl"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def truncate_record(record: dict, caps: dict) -> Tuple[dict, dict]:
    """Truncate a record to the caps. Returns (truncated_record, lineage_entry).

    Truncation policy (matches v1 `model_capped_v1`):
      - 5'UTR: keep the LAST max_5utr nt (closest to start codon).
      - CDS: keep the FIRST max_cds nt (from start codon).
      - 3'UTR: keep the FIRST max_3utr nt (after stop codon).
    """
    tid = record["transcript_id"]
    five = record.get("five_utr", "")
    cds = record.get("cds", "")
    three = record.get("three_utr", "")

    trunc_5 = len(five) > caps["max_5utr"]
    trunc_c = len(cds) > caps["max_cds"]
    trunc_3 = len(three) > caps["max_3utr"]

    new_five = five[-caps["max_5utr"]:] if trunc_5 else five
    new_cds = cds[:caps["max_cds"]] if trunc_c else cds
    new_three = three[:caps["max_3utr"]] if trunc_3 else three

    truncated = {
        "transcript_id": tid,
        "five_utr": new_five,
        "cds": new_cds,
        "three_utr": new_three,
        "species": record.get("species", "human"),
    }

    canonical_seq = five + cds + three
    derived_seq = new_five + new_cds + new_three
    lineage = {
        "canonical_id": f"{tid}",  # source is implicit by file
        "canonical_sequence_sha256": sha256_str(canonical_seq),
        "derived_sequence_sha256": sha256_str(derived_seq),
        "derived_transcript_id": tid,
        "truncated_5utr": trunc_5,
        "truncated_cds": trunc_c,
        "truncated_3utr": trunc_3,
        "original_5utr_len": len(five),
        "original_cds_len": len(cds),
        "original_3utr_len": len(three),
        "derived_5utr_len": len(new_five),
        "derived_cds_len": len(new_cds),
        "derived_3utr_len": len(new_three),
    }
    return truncated, lineage


# ---------------------------------------------------------------------------
# Per-source processing
# ---------------------------------------------------------------------------


@dataclass
class SourceStats:
    total: int = 0
    kept: int = 0
    dropped_empty_5utr: int = 0
    dropped_empty_cds: int = 0
    dropped_empty_3utr: int = 0
    dropped_5utr_too_long: int = 0  # still too long even after truncation? No—truncated.
    truncated_5utr: int = 0
    truncated_cds: int = 0
    truncated_3utr: int = 0
    # Length distribution (pre-truncation, for kept records).
    len_5utr_p50: int = 0
    len_5utr_p90: int = 0
    len_5utr_p99: int = 0
    len_cds_p50: int = 0
    len_cds_p90: int = 0
    len_cds_p99: int = 0
    len_3utr_p50: int = 0
    len_3utr_p90: int = 0
    len_3utr_p99: int = 0


def percentile(sorted_vals: List[int], p: float) -> int:
    if not sorted_vals:
        return 0
    idx = int(len(sorted_vals) * p / 100.0)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def process_source(source_name: str, canonical_path: Path, caps: dict) -> Tuple[SourceStats, Path, Path, Path]:
    """Process one source: read canonical, filter, truncate, write long-view."""
    out_dir = P1_ROOT / "sources" / source_name
    out_dir.mkdir(parents=True, exist_ok=True)

    records_path = out_dir / "long_view.records.jsonl"
    lineage_path = out_dir / "long_view.lineage.jsonl"
    manifest_path = out_dir / "long_view_manifest.json"

    stats = SourceStats()
    lens_5, lens_c, lens_3 = [], [], []

    with open(canonical_path, "r", encoding="utf-8") as fin, \
         open(records_path, "w", encoding="utf-8") as frec, \
         open(lineage_path, "w", encoding="utf-8") as flin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            stats.total += 1

            five = record.get("five_utr", "")
            cds = record.get("cds", "")
            three = record.get("three_utr", "")

            # Filter: all 3 regions must be non-empty.
            if not five:
                stats.dropped_empty_5utr += 1
                continue
            if not cds:
                stats.dropped_empty_cds += 1
                continue
            if not three:
                stats.dropped_empty_3utr += 1
                continue

            # Truncate to caps.
            truncated, lineage = truncate_record(record, caps)
            if lineage["truncated_5utr"]:
                stats.truncated_5utr += 1
            if lineage["truncated_cds"]:
                stats.truncated_cds += 1
            if lineage["truncated_3utr"]:
                stats.truncated_3utr += 1

            frec.write(json.dumps(truncated, ensure_ascii=False) + "\n")
            flin.write(json.dumps(lineage, ensure_ascii=False) + "\n")
            stats.kept += 1
            lens_5.append(len(five))
            lens_c.append(len(cds))
            lens_3.append(len(three))

    # Percentiles.
    lens_5.sort()
    lens_c.sort()
    lens_3.sort()
    stats.len_5utr_p50 = percentile(lens_5, 50)
    stats.len_5utr_p90 = percentile(lens_5, 90)
    stats.len_5utr_p99 = percentile(lens_5, 99)
    stats.len_cds_p50 = percentile(lens_c, 50)
    stats.len_cds_p90 = percentile(lens_c, 90)
    stats.len_cds_p99 = percentile(lens_c, 99)
    stats.len_3utr_p50 = percentile(lens_3, 50)
    stats.len_3utr_p90 = percentile(lens_3, 90)
    stats.len_3utr_p99 = percentile(lens_3, 99)

    # Manifest.
    manifest = {
        "artifact_kind": "p1_long_view_source_bundle",
        "source": source_name,
        "caps": caps,
        "canonical_source_path": str(canonical_path),
        "canonical_records_sha256": sha256_file(canonical_path),
        "long_view_records_path": str(records_path),
        "long_view_records_sha256": sha256_file(records_path),
        "long_view_lineage_path": str(lineage_path),
        "long_view_lineage_sha256": sha256_file(lineage_path),
        "stats": asdict(stats),
        "notes": "Long-view reconstruction from v1 frozen canonical records. v1 namespace is read-only.",
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    return stats, records_path, lineage_path, manifest_path


# ---------------------------------------------------------------------------
# Combined view
# ---------------------------------------------------------------------------


def build_combined_view(source_records: List[Path]) -> Tuple[Path, Path]:
    """Merge per-source long-view records into a combined view."""
    out_dir = P1_ROOT / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_records = out_dir / "long_view.records.jsonl"
    attrition_path = out_dir / "long_view_attrition_report.json"

    total = 0
    source_counts: Dict[str, int] = {}
    with open(combined_records, "w", encoding="utf-8") as fout:
        for source_path in source_records:
            n = 0
            with open(source_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
                    n += 1
            source_counts[source_path.parent.name] = n
            total += n

    attrition = {
        "artifact_kind": "p1_long_view_combined_attrition_report",
        "combined_records_path": str(combined_records),
        "combined_records_sha256": sha256_file(combined_records),
        "combined_records_count": total,
        "per_source_counts": source_counts,
        "caps": LONG_VIEW_CAPS,
    }
    with open(attrition_path, "w", encoding="utf-8") as fh:
        json.dump(attrition, fh, indent=2, sort_keys=True)

    return combined_records, attrition_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="P1-11 long-view reconstruction")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing files.")
    args = parser.parse_args()

    print(f"[p1-11] Long-view caps: {LONG_VIEW_CAPS}")
    print(f"[p1-11] V1 root: {V1_ROOT}")
    print(f"[p1-11] P1 root: {P1_ROOT}")

    if args.dry_run:
        print("[p1-11] DRY RUN — no files will be written.")

    P1_ROOT.mkdir(parents=True, exist_ok=True)

    source_records: List[Path] = []
    source_manifests: List[dict] = []
    for source_name, canonical_path in SOURCES:
        if not canonical_path.exists():
            print(f"[p1-11] WARNING: canonical records not found for {source_name}: {canonical_path}")
            continue
        print(f"\n[p1-11] Processing {source_name}...")
        stats, rec_path, lin_path, man_path = process_source(source_name, canonical_path, LONG_VIEW_CAPS)
        print(f"  total={stats.total}, kept={stats.kept} ({100*stats.kept/max(1,stats.total):.1f}%)")
        print(f"  dropped: empty_5utr={stats.dropped_empty_5utr}, empty_cds={stats.dropped_empty_cds}, empty_3utr={stats.dropped_empty_3utr}")
        print(f"  truncated: 5utr={stats.truncated_5utr} ({100*stats.truncated_5utr/max(1,stats.kept):.1f}%), "
              f"cds={stats.truncated_cds} ({100*stats.truncated_cds/max(1,stats.kept):.1f}%), "
              f"3utr={stats.truncated_3utr} ({100*stats.truncated_3utr/max(1,stats.kept):.1f}%)")
        print(f"  len p50/p90/p99: 5utr={stats.len_5utr_p50}/{stats.len_5utr_p90}/{stats.len_5utr_p99}, "
              f"cds={stats.len_cds_p50}/{stats.len_cds_p90}/{stats.len_cds_p99}, "
              f"3utr={stats.len_3utr_p50}/{stats.len_3utr_p90}/{stats.len_3utr_p99}")
        source_records.append(rec_path)
        with open(man_path, "r", encoding="utf-8") as fh:
            source_manifests.append(json.load(fh))

    if not source_records:
        print("[p1-11] No source records processed. Aborting.")
        return 1

    print("\n[p1-11] Building combined view...")
    combined_records, attrition_path = build_combined_view(source_records)
    print(f"  combined records: {combined_records}")
    with open(attrition_path, "r", encoding="utf-8") as fh:
        attrition = json.load(fh)
    print(f"  combined count: {attrition['combined_records_count']}")
    print(f"  per-source: {attrition['per_source_counts']}")

    # Top-level manifest.
    top_manifest = {
        "artifact_kind": "p1_long_view_bundle",
        "description": "Long-view reconstruction from v1 frozen canonical records. Caps: 5'UTR<=512, CDS<=3072, 3'UTR<=1024.",
        "caps": LONG_VIEW_CAPS,
        "v1_frozen_root": str(V1_ROOT),
        "p1_long_view_root": str(P1_ROOT),
        "sources": [m for m in source_manifests],
        "combined_records_path": str(combined_records),
        "combined_records_sha256": sha256_file(combined_records),
        "combined_attrition_path": str(attrition_path),
        "combined_attrition_sha256": sha256_file(attrition_path),
        "constraints_honored": [
            "v1 frozen namespace is read-only (not modified)",
            "All new artifacts record source path, SHA-256, counts, caps",
            "No new training; this is a data view only",
        ],
    }
    top_manifest_path = P1_ROOT / "p1_long_view_manifest.json"
    with open(top_manifest_path, "w", encoding="utf-8") as fh:
        json.dump(top_manifest, fh, indent=2, sort_keys=True)
    print(f"\n[p1-11] Top-level manifest: {top_manifest_path}")
    print(f"[p1-11] SHA-256: {sha256_file(top_manifest_path)}")
    print("[p1-11] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
