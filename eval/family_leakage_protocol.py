"""Family-disjoint split plus k-mer leakage audit protocol.

The protocol is intentionally lightweight and read-only with respect to model
training. It can run a synthetic smoke test now, and the same entry point can be
reused for official GENCODE/RefSeq records once those files exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Mapping, Optional, Sequence

from mrna_editflow.core.config import DataConfig
from mrna_editflow.data.dedup_split import family_disjoint_split
from mrna_editflow.data.download_mrna import load_records_jsonl, synthesize_corpus
from mrna_editflow.data.leakage_audit import audit_leakage


CLAIM_POLICY = (
    "Family/leakage protocol reports are data-governance evidence. Synthetic "
    "smoke runs prove only that the split and k-mer audit plumbing executes. "
    "Real leakage-free dataset claims require official cleaned records, a "
    "reference/pretraining corpus or paired corpus, persisted split files, and "
    "a leakage audit with acceptable flagged/exact-match rates."
)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_idx(path: str, indices: Sequence[int]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for idx in indices:
            fh.write(f"{int(idx)}\n")
    return path


def _split_members(records: Sequence[object], indices: Sequence[int]) -> list[object]:
    return [records[int(idx)] for idx in indices]


def _cluster_metadata(clusters: Sequence[Sequence[int]], n_records: int) -> dict[str, object]:
    """Return deterministic cluster membership metadata for provenance."""
    assignments = [-1] * n_records
    for cluster_id, members in enumerate(clusters):
        for idx in members:
            assignments[int(idx)] = int(cluster_id)
    if any(value < 0 for value in assignments):
        raise ValueError("cluster assignments do not cover all records")
    canonical = json.dumps(assignments, separators=(",", ":")).encode("utf-8")
    return {
        "assignment_digest": hashlib.sha256(canonical).hexdigest(),
        "n_assignments": len(assignments),
        "assignments": assignments,
    }


def run_family_leakage_protocol(
    *,
    records_path: Optional[str] = None,
    reference_path: Optional[str] = None,
    out_split_dir: Optional[str] = None,
    n_synthetic: int = 64,
    seed: int = 0,
    use_mmseqs: str = "never",
    kmer: int = 15,
    top_k: int = 3,
    jaccard_threshold: float = 0.80,
    containment_threshold: float = 0.95,
) -> dict[str, object]:
    """Run family split and leakage audit over real or synthetic records."""
    synthetic = records_path is None
    records = (
        synthesize_corpus(n_synthetic, seed=seed)
        if synthetic
        else load_records_jsonl(records_path)
    )
    if not records:
        raise ValueError("family/leakage protocol requires at least one record")

    cfg = DataConfig(seed=seed)
    split = family_disjoint_split(
        records,
        cfg=cfg,
        out_dir=out_split_dir,
        use_mmseqs=use_mmseqs,
        write=bool(out_split_dir),
    )
    split_paths = dict(split.paths)
    cluster_metadata = _cluster_metadata(split.clusters, len(records))
    cluster_assignment_path = None
    if out_split_dir:
        cluster_assignment_path = os.path.join(out_split_dir, "cluster_assignments.json")
        os.makedirs(os.path.dirname(os.path.abspath(cluster_assignment_path)), exist_ok=True)
        with open(cluster_assignment_path, "w", encoding="utf-8") as fh:
            json.dump(cluster_metadata["assignments"], fh, separators=(",", ":"))
    if out_split_dir and not split_paths:
        split_paths = {
            "train": _write_idx(os.path.join(out_split_dir, "train.idx"), split.train),
            "val": _write_idx(os.path.join(out_split_dir, "val.idx"), split.val),
            "test": _write_idx(os.path.join(out_split_dir, "test.idx"), split.test),
        }

    query_records = _split_members(records, split.test)
    if reference_path:
        reference_records = load_records_jsonl(reference_path)
        reference_kind = "external_reference"
    else:
        reference_records = _split_members(records, split.train)
        reference_kind = "train_split"
    leakage = audit_leakage(
        query_records,
        reference_records,
        k=kmer,
        top_k=top_k,
        jaccard_threshold=jaccard_threshold,
        containment_threshold=containment_threshold,
    )
    leak_summary = leakage.get("summary", {})
    exact_count = (
        int(leak_summary.get("exact_match_count", 0))
        if isinstance(leak_summary, Mapping)
        else 0
    )
    flagged_fraction = (
        float(leak_summary.get("flagged_fraction", 1.0))
        if isinstance(leak_summary, Mapping)
        else 1.0
    )
    split_ready = bool(
        len(split.train) + len(split.val) + len(split.test) == len(records)
        and not (set(split.train) & set(split.val))
        and not (set(split.train) & set(split.test))
        and not (set(split.val) & set(split.test))
    )
    ready_for_audit = bool(
        records_path
        and reference_path
        and split_ready
        and len(query_records) > 0
        and len(reference_records) > 0
        and exact_count == 0
    )
    return {
        "artifact_kind": "family_leakage_protocol",
        "claim_policy": CLAIM_POLICY,
        "input": {
            "records_path": records_path,
            "reference_path": reference_path,
            "source_kind": "synthetic_smoke" if synthetic else "external_records",
            "reference_kind": reference_kind,
            "n_records": len(records),
            "n_reference": len(reference_records),
        },
        "split": {
            "method": split.method,
            "n_clusters": split.n_clusters,
            "n_train": len(split.train),
            "n_val": len(split.val),
            "n_test": len(split.test),
            "cluster_disjoint": split_ready,
            "cluster_assignment_digest": cluster_metadata["assignment_digest"],
            "cluster_assignment_count": cluster_metadata["n_assignments"],
            "cluster_assignment_path": cluster_assignment_path,
            "cluster_assignment_sha256": (
                _sha256_file(cluster_assignment_path)
                if cluster_assignment_path else None
            ),
            "paths": split_paths,
        },
        "leakage": leakage,
        "summary": {
            "synthetic_smoke_only": synthetic and reference_path is None,
            "external_records_provided": records_path is not None,
            "external_reference_provided": reference_path is not None,
            "split_ready": split_ready,
            "ready_for_family_leakage_audit": ready_for_audit,
            "ready_for_family_disjoint_leakage_claim": False,
            "n_records": len(records),
            "n_reference": len(reference_records),
            "n_clusters": split.n_clusters,
            "n_train": len(split.train),
            "n_val": len(split.val),
            "n_test": len(split.test),
            "leakage_flagged_fraction": flagged_fraction,
            "leakage_exact_match_count": exact_count,
        },
        "limitations": [
            "Synthetic smoke evidence is not an official leakage audit.",
            "Cross-corpus claims require official cleaned records and reference corpus provenance.",
            "Acceptable flagged rates must be interpreted for the exact benchmark protocol.",
        ],
    }


def write_report_json(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    return path


def write_report_markdown(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    summary = report.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    split = report.get("split", {})
    if not isinstance(split, Mapping):
        split = {}
    leakage = report.get("leakage", {})
    leak_summary = leakage.get("summary", {}) if isinstance(leakage, Mapping) else {}
    if not isinstance(leak_summary, Mapping):
        leak_summary = {}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Family Leakage Protocol\n\n")
        fh.write(f"- Claim policy: {report.get('claim_policy', CLAIM_POLICY)}\n")
        fh.write(
            f"- Synthetic smoke only: `{summary.get('synthetic_smoke_only')}`; "
            f"ready for family leakage audit: "
            f"`{summary.get('ready_for_family_leakage_audit')}`; "
            f"ready for leakage-free claim: "
            f"`{summary.get('ready_for_family_disjoint_leakage_claim')}`\n"
        )
        fh.write(
            f"- Split: method=`{split.get('method')}`, clusters=`{split.get('n_clusters')}`, "
            f"train/val/test=`{split.get('n_train')}/{split.get('n_val')}/"
            f"{split.get('n_test')}`\n"
        )
        fh.write(
            f"- Leakage: flagged_fraction=`{leak_summary.get('flagged_fraction')}`, "
            f"exact_match_count=`{leak_summary.get('exact_match_count')}`\n\n"
        )
        fh.write("## Limitations\n\n")
        limitations = report.get("limitations", [])
        if isinstance(limitations, Sequence) and not isinstance(limitations, (str, bytes)):
            for item in limitations:
                fh.write(f"- {item}\n")
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", default=None)
    parser.add_argument("--reference-jsonl", default=None)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-split-dir", default=None)
    parser.add_argument("--n-synthetic", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-mmseqs", default="never", choices=("auto", "never", "force"))
    parser.add_argument("--kmer", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--jaccard-threshold", type=float, default=0.80)
    parser.add_argument("--containment-threshold", type=float, default=0.95)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    report = run_family_leakage_protocol(
        records_path=args.records_jsonl,
        reference_path=args.reference_jsonl,
        out_split_dir=args.out_split_dir,
        n_synthetic=args.n_synthetic,
        seed=args.seed,
        use_mmseqs=args.use_mmseqs,
        kmer=args.kmer,
        top_k=args.top_k,
        jaccard_threshold=args.jaccard_threshold,
        containment_threshold=args.containment_threshold,
    )
    write_report_json(report, args.out_json)
    write_report_markdown(report, args.out_md)
    print(json.dumps({"json_path": args.out_json, "markdown_path": args.out_md}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLAIM_POLICY",
    "run_family_leakage_protocol",
    "write_report_json",
    "write_report_markdown",
    "main",
]
