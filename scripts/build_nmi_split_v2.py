#!/usr/bin/env python3
"""Build the immutable Benchmark v2 registry from the existing P3 records.

The builder is intentionally conservative: it preserves measured/proxy/
unlabeled semantics, maps only the already frozen source-level assignments,
and leaves unsupported context/assay/family axes empty with an explicit
blocker instead of manufacturing an independent test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROLE_MAP = {"train": "train", "val": "val", "test": "test_id", "ood": "test_ood"}
FINAL_ROLES = {"test_id", "test_family", "test_context", "test_assay", "test_ood"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def choose_source_role(old_role: str) -> str:
    if old_role not in ROLE_MAP:
        raise ValueError(f"unsupported source split role {old_role!r}")
    return ROLE_MAP[old_role]


def build(input_paths: Iterable[Path], out_dir: Path) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifests").mkdir(exist_ok=True)
    (out_dir / "indices").mkdir(exist_ok=True)
    records_path = out_dir / "records.jsonl"
    index_fhs = {
        role: (out_dir / "indices" / f"{role}.txt").open("w")
        for role in ["train", "val", *sorted(FINAL_ROLES)]
    }
    counts = Counter()
    confidence = Counter()
    source_role: Dict[str, str] = {}
    family_by_role: Dict[str, set] = defaultdict(set)
    sources_by_role: Dict[str, set] = defaultdict(set)
    context_by_role: Dict[str, set] = defaultdict(set)
    assay_by_role: Dict[str, set] = defaultdict(set)
    candidate_hashes: Dict[str, set] = defaultdict(set)
    record_ids = set()
    try:
        with records_path.open("w") as out:
            for input_path in input_paths:
                with input_path.open() as fh:
                    for line_no, line in enumerate(fh, 1):
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        old_role = rec.get("split_role")
                        if old_role is None:
                            raise ValueError(f"missing split_role in {input_path}:{line_no}")
                        role = choose_source_role(str(old_role))
                        rid = str(rec.get("record_id", ""))
                        if not rid or rid in record_ids:
                            raise ValueError(f"missing or duplicate record_id at {input_path}:{line_no}")
                        record_ids.add(rid)
                        rec["benchmark_version"] = "nmi_benchmark_v2"
                        rec["v2_source_role"] = role
                        out.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
                        index_fhs[role].write(rid + "\n")
                        counts[role] += 1
                        confidence[(role, str(rec.get("confidence")))] += 1
                        sid = str(rec.get("source_id"))
                        source_role.setdefault(sid, role)
                        if source_role[sid] != role:
                            raise ValueError(f"source {sid} crosses roles: {source_role[sid]} vs {role}")
                        sources_by_role[role].add(sid)
                        family_by_role[role].add(str(rec.get("family_cluster_id")))
                        context_by_role[role].add(str(rec.get("cell_context")))
                        assay_by_role[role].add(str(rec.get("assay_type")))
                        candidate_hashes[role].add(hashlib.sha256(str(rec.get("candidate_sequence")).encode()).hexdigest())
    finally:
        for fh in index_fhs.values():
            fh.close()

    # The source registry currently supplies a genuine source-disjoint ID test
    # and a GC/length OOD role.  Other axes require independent measured data;
    # refuse to alias existing records into multiple final tests.
    blockers = {
        "test_family": "no independent family-level holdout was added by this registry build",
        "test_context": "no independent measured cell-context holdout is present",
        "test_assay": "no independent measured assay holdout is present",
    }
    for role in blockers:
        (out_dir / "indices" / f"{role}.txt").write_text("")

    manifest_paths = {}
    roles = ["train", "val", *sorted(FINAL_ROLES)]
    for role in roles:
        idx = out_dir / "indices" / f"{role}.txt"
        obj = {
            "schema_version": "nmi_benchmark_v2",
            "role": role,
            "final_test": role in FINAL_ROLES,
            "records_path": "records.jsonl",
            "index_path": f"indices/{role}.txt",
            "record_count": sum(1 for _ in idx.open()),
            "index_sha256": sha256_file(idx),
            "source_count": len(sources_by_role.get(role, set())) if role not in blockers else 0,
            "label_policy": "hidden_by_default" if role in FINAL_ROLES else "development_allowed",
            "blocker": blockers.get(role),
        }
        p = out_dir / "manifests" / f"{role}.json"
        p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
        manifest_paths[role] = str(p.relative_to(out_dir))

    registry = {
        "schema_version": "nmi_benchmark_v2",
        "source_inputs": [str(p) for p in input_paths],
        "records_path": str(records_path.relative_to(out_dir)),
        "records_sha256": sha256_file(records_path),
        "total_records": sum(counts.values()),
        "counts_by_role": dict(sorted(counts.items())),
        "confidence_by_role": {f"{r}:{c}": n for (r, c), n in sorted(confidence.items())},
        "source_counts": {r: len(v) for r, v in sorted(sources_by_role.items())},
        "family_counts": {r: len(v) for r, v in sorted(family_by_role.items())},
        "contexts_by_role": {r: sorted(v) for r, v in sorted(context_by_role.items())},
        "assays_by_role": {r: sorted(v) for r, v in sorted(assay_by_role.items())},
        "manifest_paths": manifest_paths,
        "blockers": blockers,
        "final_test_policy": "loader refuses final roles without explicit allow_final_labels",
    }
    (out_dir / "registry.json").write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    return registry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True, help="source JSONL; repeatable")
    ap.add_argument("--out-dir", default="data/nmi_benchmark_v2")
    args = ap.parse_args()
    registry = build([Path(p) for p in args.input], Path(args.out_dir))
    print(json.dumps(registry, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
