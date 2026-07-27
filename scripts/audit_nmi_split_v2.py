#!/usr/bin/env python3
"""Audit Benchmark v2 provenance, leakage and final-test access policy."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from mrna_editflow.data.nmi_benchmark_v2 import FINAL_ROLES, load_manifest


def audit(root: Path, *, allow_final_labels: bool) -> Dict:
    roles = ["train", "val", "test_id", "test_family", "test_context", "test_assay", "test_ood"]
    manifests = {
        role: load_manifest(root / "manifests" / f"{role}.json", allow_final_labels=allow_final_labels)
        for role in roles
    }
    record_to_roles: Dict[str, set] = defaultdict(set)
    for role, manifest in manifests.items():
        index_path = root / str(manifest["index_path"])
        with index_path.open() as fh:
            for line in fh:
                rid = line.strip()
                if rid:
                    record_to_roles[rid].add(role)
    source_roles: Dict[str, set] = defaultdict(set)
    family_roles: Dict[str, set] = defaultdict(set)
    seq_roles: Dict[str, set] = defaultdict(set)
    counts = Counter()
    confidence = {role: Counter() for role in roles}
    records_path = root / str(next(iter(manifests.values()))["records_path"])
    with records_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = str(rec.get("record_id"))
            for role in record_to_roles.get(rid, ()):
                counts[role] += 1
                confidence[role][str(rec.get("confidence"))] += 1
                source_roles[str(rec.get("source_id"))].add(role)
                family_roles[str(rec.get("family_cluster_id"))].add(role)
                seq_roles[hashlib.sha256(str(rec.get("candidate_sequence")).encode()).hexdigest()].add(role)
    source_cross = {k: sorted(v) for k, v in source_roles.items() if len(v) > 1}
    seq_cross = {k: sorted(v) for k, v in seq_roles.items() if len(v) > 1}
    family_cross = {
        k: sorted(v) for k, v in family_roles.items()
        if "test_family" in v and any(r in {"train", "val"} for r in v)
    }
    # OOD is a distribution-shift diagnostic; family overlap is reported but
    # does not violate the stricter test_family contract.
    ood_family_overlap = {
        k: sorted(v) for k, v in family_roles.items()
        if "test_ood" in v and any(r in {"train", "val"} for r in v)
    }
    counts = {role: int(counts[role]) for role in roles}
    confidence = {role: dict(confidence[role]) for role in roles}
    passed = not source_cross and not seq_cross and not family_cross
    report = {
        "schema_version": "nmi_benchmark_v2_audit_v1",
        "passed": passed,
        "allow_final_labels": allow_final_labels,
        "counts": counts,
        "confidence_by_role": confidence,
        "source_cross_role_count": len(source_cross),
        "candidate_cross_role_count": len(seq_cross),
        "family_train_final_overlap_count": len(family_cross),
        "ood_family_overlap_count": len(ood_family_overlap),
        "examples": {
            "source_cross_role": dict(list(source_cross.items())[:10]),
            "candidate_cross_role": dict(list(seq_cross.items())[:10]),
            "family_train_final_overlap": dict(list(family_cross.items())[:10]),
            "ood_family_overlap": dict(list(ood_family_overlap.items())[:10]),
        },
        "unsupported_axes": {
            role: counts[role] == 0
            for role in ("test_family", "test_context", "test_assay")
        },
        "claim_policy": "no SOTA/OOD-independent claim from an empty or proxy-only role",
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/nmi_benchmark_v2")
    ap.add_argument("--out", default="artifacts/nmi_benchmark_v2_audit.json")
    ap.add_argument("--allow-final-labels", action="store_true")
    args = ap.parse_args()
    report = audit(Path(args.root), allow_final_labels=args.allow_final_labels)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
