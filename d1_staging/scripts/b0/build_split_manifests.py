#!/usr/bin/env python
"""B0-02: Build split manifests for 4 split types.

Split types:
  1. 5utr_source_disjoint  — mmseqs cluster GSE114002 source sequences,
     split clusters 80/10/10 (train/val/test). No source sequence (nor
     any sequence in the same mmseqs cluster) appears in >1 split.
  2. 3utr_source_disjoint  — same as above for pooled 3'UTR datasets
     (GSE200304 + GSE232572 + GSE186455).
  3. study_disjoint        — train = GSE114002 (5'UTR), test = all 3'UTR
     (GSE200304 + GSE232572 + GSE186455). Cross-study + cross-region.
  4. cross_region_transfer — train = GSE114002 5'UTR (80%),
     val = GSE114002 5'UTR (10%), test = all 3'UTR (100%).

B0-02 acceptance: unexplained overlap = 0, reverse/path leakage = 0.

mmseqs clustering:
  - Createdb on source sequences (one line per record, header = record_id)
  - Cluster at 0.8 identity (--min-seq-id 0.8 -c 0.8 --cov-mode 0)
  - Split clusters (not records) into train/val/test to guarantee no
    near-duplicate leakage across splits.

Contract: utr_editflow_contract_v2 (FROZEN)
Task: B0-02
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Make B0 schemas importable
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from canonical_schemas import UTREditRecord  # noqa: E402

MMSEQS = os.environ.get("MMSEQS_BIN", "/home/cunyuliu/tools/mmseqs2/bin/mmseqs")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPLIT_TYPES = (
    "5utr_source_disjoint",
    "3utr_source_disjoint",
    "study_disjoint",
    "cross_region_transfer",
)

TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10

CLUSTER_MIN_SEQ_ID = "0.8"
CLUSTER_COV = "0.8"
CLUSTER_COV_MODE = "0"

RANDOM_SEED = 20260801


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_paired_records(path: str) -> List[UTREditRecord]:
    """Load paired records from canonical_records.jsonl."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rec = UTREditRecord.from_dict(d)
            if rec.is_paired:
                records.append(rec)
    return records


def filter_records(
    records: List[UTREditRecord],
    accession: Optional[str] = None,
    region: Optional[str] = None,
) -> List[UTREditRecord]:
    """Filter records by accession and/or region."""
    out = records
    if accession:
        out = [r for r in out if r.accession == accession]
    if region:
        out = [r for r in out if r.region == region]
    return out


# ---------------------------------------------------------------------------
# mmseqs clustering
# ---------------------------------------------------------------------------

def run_mmseqs_cluster(
    records: List[UTREditRecord],
    work_dir: Path,
    seq_field: str = "source_sequence",
) -> Dict[str, str]:
    """Cluster sequences with mmseqs; return {record_id: cluster_representative}.

    Args:
        records: list of UTREditRecord
        work_dir: temp directory for mmseqs intermediate files
        seq_field: "source_sequence" or "candidate_sequence"
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Write input FASTA with index-based headers to avoid special char issues
    fasta_path = work_dir / "input.fasta"
    idx_to_rid: Dict[str, str] = {}
    with open(fasta_path, "w") as f:
        for idx, rec in enumerate(records):
            seq = getattr(rec, seq_field)
            if not seq:
                continue
            safe_id = f"seq_{idx:08d}"
            idx_to_rid[safe_id] = rec.record_id
            f.write(f">{safe_id}\n{seq}\n")

    db_path = str(work_dir / "input_db")
    cluster_db = str(work_dir / "cluster_db")
    tsv_path = str(work_dir / "cluster.tsv")

    # mmseqs createdb
    _run([MMSEQS, "createdb", str(fasta_path), db_path], work_dir)
    # mmseqs cluster
    _run([
        MMSEQS, "cluster", db_path, cluster_db, str(work_dir / "tmp"),
        "--min-seq-id", CLUSTER_MIN_SEQ_ID,
        "-c", CLUSTER_COV,
        "--cov-mode", CLUSTER_COV_MODE,
        "--threads", "4",
    ], work_dir)
    # mmseqs createtsv
    _run([
        MMSEQS, "createtsv", db_path, db_path, cluster_db, tsv_path,
    ], work_dir)

    # Parse cluster TSV: each line is "cluster_representative\tmember"
    # Map back from index-based IDs to original record_ids
    record_to_cluster: Dict[str, str] = {}
    with open(tsv_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                rep_idx, member_idx = parts[0], parts[1]
                # Use the representative's index as the cluster key
                cluster_key = idx_to_rid.get(rep_idx, rep_idx)
                member_rid = idx_to_rid.get(member_idx, member_idx)
                record_to_cluster[member_rid] = cluster_key
    return record_to_cluster


def _run(cmd: List[str], work_dir: Path) -> None:
    """Run a command, raise on failure."""
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(work_dir)
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"stdout: {result.stdout[-500:]}\n"
            f"stderr: {result.stderr[-500:]}"
        )


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_clusters(
    record_to_cluster: Dict[str, str],
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    seed: int = RANDOM_SEED,
) -> Dict[str, str]:
    """Split records into train/val/test by cluster.

    No cluster appears in more than one split.

    Returns:
        {record_id: "train" | "val" | "test"}
    """
    # Group records by cluster
    cluster_to_records: Dict[str, List[str]] = defaultdict(list)
    for rid, cluster in record_to_cluster.items():
        cluster_to_records[cluster].append(rid)

    clusters = sorted(cluster_to_records.keys())
    rng = random.Random(seed)
    rng.shuffle(clusters)

    n_clusters = len(clusters)
    n_train = int(n_clusters * train_frac)
    n_val = int(n_clusters * val_frac)

    train_clusters = set(clusters[:n_train])
    val_clusters = set(clusters[n_train:n_train + n_val])
    test_clusters = set(clusters[n_train + n_val:])

    assignments: Dict[str, str] = {}
    for cluster, rids in cluster_to_records.items():
        if cluster in train_clusters:
            split = "train"
        elif cluster in val_clusters:
            split = "val"
        else:
            split = "test"
        for rid in rids:
            assignments[rid] = split
    return assignments


def fix_reverse_leakage(
    records: List[UTREditRecord],
    assignments: Dict[str, str],
    max_iter: int = 10,
) -> Dict[str, str]:
    """Fix reverse leakage: candidate_sequence(train) ∩ source_sequence(test).

    Excludes leaking train records from all splits (split='leakage_excluded')
    so they appear in neither train nor test, avoiding cascading leaks that
    would occur if they were moved to test (source_overlap, path_leakage).
    """
    record_by_id = {r.record_id: r for r in records}
    total_excluded = 0
    for iteration in range(max_iter):
        test_sources: Set[str] = set()
        for rid, split in assignments.items():
            if split == "test":
                rec = record_by_id[rid]
                if rec.source_sequence:
                    test_sources.add(rec.source_sequence)
        leaking = [
            rid for rid, split in assignments.items()
            if split == "train"
            and record_by_id[rid].candidate_sequence
            and record_by_id[rid].candidate_sequence in test_sources
        ]
        if not leaking:
            break
        for rid in leaking:
            assignments[rid] = "leakage_excluded"
        total_excluded += len(leaking)
        print(f"  reverse_leakage fix iter {iteration+1}: excluded {len(leaking)} records")
    if total_excluded:
        print(f"  Total leakage-excluded: {total_excluded}")
    return assignments


def assign_by_accession(
    records: List[UTREditRecord],
    train_accessions: Set[str],
    val_accessions: Set[str],
    test_accessions: Set[str],
) -> Dict[str, str]:
    """Assign records to splits by accession."""
    assignments: Dict[str, str] = {}
    for rec in records:
        if rec.accession in train_accessions:
            assignments[rec.record_id] = "train"
        elif rec.accession in val_accessions:
            assignments[rec.record_id] = "val"
        elif rec.accession in test_accessions:
            assignments[rec.record_id] = "test"
        else:
            raise ValueError(
                f"Record {rec.record_id} accession {rec.accession} not in any split set"
            )
    return assignments


# ---------------------------------------------------------------------------
# Manifest writing
# ---------------------------------------------------------------------------

def write_manifest(
    split_type: str,
    records: List[UTREditRecord],
    assignments: Dict[str, str],
    output_path: str,
    notes: str = "",
) -> Dict:
    """Write a split manifest JSONL + return summary.

    Each line: {"record_id": ..., "accession": ..., "region": ...,
                 "split": "train"|"val"|"test", "split_type": split_type}

    Records with split values outside {"train","val","test"} are excluded
    from the manifest (e.g. "train_val_unused" for cross_region_transfer).
    """
    record_by_id = {r.record_id: r for r in records}
    counts = {"train": 0, "val": 0, "test": 0}
    by_accession_split: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n_excluded = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for rid, split in sorted(assignments.items()):
            if split not in ("train", "val", "test"):
                n_excluded += 1
                continue
            rec = record_by_id[rid]
            entry = {
                "record_id": rid,
                "accession": rec.accession,
                "region": rec.region,
                "split": split,
                "split_type": split_type,
            }
            f.write(json.dumps(entry) + "\n")
            counts[split] += 1
            by_accession_split[rec.accession][split] += 1

    # Compute SHA-256
    with open(output_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    summary = {
        "split_type": split_type,
        "output_path": output_path,
        "sha256": sha256,
        "n_total": counts["train"] + counts["val"] + counts["test"],
        "n_excluded": n_excluded,
        "n_train": counts["train"],
        "n_val": counts["val"],
        "n_test": counts["test"],
        "by_accession_split": {k: dict(v) for k, v in by_accession_split.items()},
        "notes": notes,
    }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_all_splits(
    canonical_records_path: str,
    output_dir: str,
    mmseqs_work_dir: str,
) -> Dict:
    """Build all 4 split manifests.

    Returns a summary dict with all split info for the audit report.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mmseqs_work = Path(mmseqs_work_dir)
    mmseqs_work.mkdir(parents=True, exist_ok=True)

    print("Loading paired records...")
    all_records = load_paired_records(canonical_records_path)
    print(f"  Total paired: {len(all_records)}")

    gse114002 = filter_records(all_records, accession="GSE114002")
    gse200304 = filter_records(all_records, accession="GSE200304")
    gse232572 = filter_records(all_records, accession="GSE232572")
    gse186455 = filter_records(all_records, accession="GSE186455")
    print(f"  GSE114002 (5'UTR): {len(gse114002)}")
    print(f"  GSE200304 (3'UTR): {len(gse200304)}")
    print(f"  GSE232572 (3'UTR): {len(gse232572)}")
    print(f"  GSE186455 (3'UTR): {len(gse186455)}")
    # Pool all 3'UTR paired datasets for 3'UTR-related splits
    all_3utr = gse200304 + gse232572 + gse186455
    print(f"  All 3'UTR pooled: {len(all_3utr)}")

    summaries = {}

    # --- 1. 5utr_source_disjoint ---
    print("\n[1/4] Building 5utr_source_disjoint split...")
    work = mmseqs_work / "5utr_source"
    r2c = run_mmseqs_cluster(gse114002, work, seq_field="source_sequence")
    n_clusters = len(set(r2c.values()))
    print(f"  mmseqs clusters: {n_clusters}")
    assignments = split_clusters(r2c)
    assignments = fix_reverse_leakage(gse114002, assignments)
    summary = write_manifest(
        "5utr_source_disjoint",
        gse114002,
        assignments,
        str(output_dir / "split_5utr_source_disjoint.jsonl"),
        notes=f"mmseqs clustering at identity={CLUSTER_MIN_SEQ_ID}, cov={CLUSTER_COV}; "
              f"{n_clusters} clusters split 80/10/10 by cluster",
    )
    summaries["5utr_source_disjoint"] = summary
    print(f"  train={summary['n_train']} val={summary['n_val']} test={summary['n_test']}")

    # --- 2. 3utr_source_disjoint ---
    # Pools all 3'UTR paired datasets (GSE200304 + GSE232572 + GSE186455)
    print("\n[2/4] Building 3utr_source_disjoint split...")
    work = mmseqs_work / "3utr_source"
    r2c = run_mmseqs_cluster(all_3utr, work, seq_field="source_sequence")
    n_clusters = len(set(r2c.values()))
    print(f"  mmseqs clusters: {n_clusters}")
    assignments = split_clusters(r2c)
    assignments = fix_reverse_leakage(all_3utr, assignments)
    summary = write_manifest(
        "3utr_source_disjoint",
        all_3utr,
        assignments,
        str(output_dir / "split_3utr_source_disjoint.jsonl"),
        notes=f"mmseqs clustering at identity={CLUSTER_MIN_SEQ_ID}, cov={CLUSTER_COV}; "
              f"{n_clusters} clusters split 80/10/10 by cluster. "
              f"Pools GSE200304+GSE232572+GSE186455 (all 3'UTR paired datasets).",
    )
    summaries["3utr_source_disjoint"] = summary
    print(f"  train={summary['n_train']} val={summary['n_val']} test={summary['n_test']}")

    # --- 3. study_disjoint ---
    # train = GSE114002 (5'UTR), test = all 3'UTR (GSE200304+GSE232572+GSE186455)
    # Cross-study AND cross-region.
    print("\n[3/4] Building study_disjoint split...")
    pooled = gse114002 + all_3utr
    assignments = assign_by_accession(
        pooled,
        train_accessions={"GSE114002"},
        val_accessions=set(),
        test_accessions={"GSE200304", "GSE232572", "GSE186455"},
    )
    summary = write_manifest(
        "study_disjoint",
        pooled,
        assignments,
        str(output_dir / "split_study_disjoint.jsonl"),
        notes="train=GSE114002 (5'UTR), test=GSE200304+GSE232572+GSE186455 (3'UTR). "
              "Cross-study AND cross-region. Val is empty (use cross_region_transfer "
              "val instead, or source_disjoint val within train study).",
    )
    summaries["study_disjoint"] = summary
    print(f"  train={summary['n_train']} val={summary['n_val']} test={summary['n_test']}")

    # --- 4. cross_region_transfer ---
    # train = GSE114002 5'UTR (80%), val = GSE114002 5'UTR (10%),
    # test = GSE200304 3'UTR (100%)
    # Use mmseqs clusters from 5utr_source_disjoint for train/val split
    print("\n[4/4] Building cross_region_transfer split...")
    r2c_5utr = run_mmseqs_cluster(
        gse114002, mmseqs_work / "5utr_crt", seq_field="source_sequence"
    )
    clusters_5utr = defaultdict(list)
    for rid, cluster in r2c_5utr.items():
        clusters_5utr[cluster].append(rid)
    cluster_keys = sorted(clusters_5utr.keys())
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(cluster_keys)
    n_clusters = len(cluster_keys)
    n_train = int(n_clusters * 0.80)  # 80% train, 10% val, 10% discarded (not used for test)
    train_clusters = set(cluster_keys[:n_train])
    val_clusters = set(cluster_keys[n_train:n_train + int(n_clusters * 0.10)])

    assignments: Dict[str, str] = {}
    for cluster, rids in clusters_5utr.items():
        split = "train" if cluster in train_clusters else ("val" if cluster in val_clusters else "train_val_unused")
        for rid in rids:
            assignments[rid] = split
    # test = all 3'UTR (GSE200304+GSE232572+GSE186455)
    for rec in all_3utr:
        assignments[rec.record_id] = "test"

    summary = write_manifest(
        "cross_region_transfer",
        gse114002 + all_3utr,
        assignments,
        str(output_dir / "split_cross_region_transfer.jsonl"),
        notes="train+val=GSE114002 (5'UTR, mmseqs cluster split 80/10), "
              "test=GSE200304+GSE232572+GSE186455 (3'UTR, 100%). "
              "Tests 5'UTR->3'UTR region transfer. "
              "train_val_unused records are GSE114002 clusters not assigned to "
              "train or val (the remaining 10%); they are excluded from this split.",
    )
    summaries["cross_region_transfer"] = summary
    print(f"  train={summary['n_train']} val={summary['n_val']} test={summary['n_test']}")

    return summaries


def main():
    parser = argparse.ArgumentParser(description="B0-02: Build split manifests")
    parser.add_argument(
        "--input", default="data/d1_canonical_records.jsonl",
        help="Path to canonical_records.jsonl",
    )
    parser.add_argument(
        "--output-dir", default="data/b0_splits",
        help="Output directory for split manifests",
    )
    parser.add_argument(
        "--mmseqs-work", default="/tmp/b0_mmseqs_work",
        help="Working directory for mmseqs intermediate files",
    )
    args = parser.parse_args()

    summaries = build_all_splits(args.input, args.output_dir, args.mmseqs_work)

    # Write summary
    summary_path = Path(args.output_dir) / "split_manifests_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\n=== Summary written to {summary_path} ===")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
