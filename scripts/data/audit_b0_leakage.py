#!/usr/bin/env python3
"""Audit one frozen B0 split for graph, metadata, and foundation leakage."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.utr_benchmark_v2.leakage import audit_split_manifest
from data.utr_benchmark_v2.near_neighbors import NearNeighborClusters
from data.utr_benchmark_v2.split_graph import D1_SPLIT_BINDING_FIELDS
from data.utr_benchmark_v2.split_graph import build_split_manifest
from data.utr_benchmark_v2.split_graph import canonical_split_manifest_core
from data.utr_benchmark_v2.split_graph import global_near_neighbor_clusters
from data.utr_benchmark_v2.split_graph import record_ids_sha256
from data.utr_benchmark_v2.split_graph import record_universe_sha256
from scripts.data.build_b0_splits import load_jsonl
from scripts.data.build_b0_splits import load_structural_jsonl
from scripts.data.build_b0_splits import sha256_file
from scripts.data.build_b0_splits import write_json_exclusive


@dataclass(frozen=True)
class BoundStructuralRecomputeCache:
    """One immutable, content-bound structural snapshot and exact NN index."""

    records_path: Path
    records_sha256: str
    records_bytes: int
    record_count: int
    record_ids_sha256: str
    structural_content_sha256: str
    records: Tuple[Mapping[str, Any], ...]
    near_neighbors: NearNeighborClusters


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def build_bound_structural_recompute_cache(
    records_path: Path,
) -> BoundStructuralRecomputeCache:
    """Read and cluster one stable structural file exactly once."""

    resolved = records_path.resolve()
    before = resolved.stat()
    records_sha256 = sha256_file(resolved)
    records = tuple(load_structural_jsonl(resolved))
    after_sha256 = sha256_file(resolved)
    after = resolved.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or after_sha256 != records_sha256:
        raise ValueError("structural record store changed while building cache")
    return BoundStructuralRecomputeCache(
        records_path=resolved,
        records_sha256=records_sha256,
        records_bytes=after.st_size,
        record_count=len(records),
        record_ids_sha256=record_ids_sha256(records),
        structural_content_sha256=record_universe_sha256(records),
        records=records,
        near_neighbors=global_near_neighbor_clusters(records),
    )


def load_json(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        value = yaml.safe_load(text)
    else:
        value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


def recompute_bound_leakage_report(
    records_path: Path,
    split_manifest_path: Path,
    *,
    foundation_exposure_path: Path | None = None,
    verify_canonical_store: bool = True,
    expected_d1_acceptance_path: Path | None = None,
    _cache: BoundStructuralRecomputeCache | None = None,
) -> Dict[str, Any]:
    """Recompute one leakage report from the bound structural record store.

    This is the only production report construction path used by the standalone
    auditor, the B0 artifact builder, and the B0 acceptance CLI.  Callers must
    never substitute a supplied report for this recomputation.

    ``verify_canonical_store=False`` exists solely for the B0 builder's
    pre-label selection phase.  In that mode the caller must already have
    validated the D1 acceptance/build-manifest binding; the canonical file is
    deliberately neither opened nor hashed before the selection freeze.
    """

    records_path = records_path.resolve()
    split_manifest_path = split_manifest_path.resolve()
    foundation: Optional[Dict[str, Any]] = None
    if foundation_exposure_path is not None:
        foundation_exposure_path = foundation_exposure_path.resolve()
        foundation = load_json(foundation_exposure_path)
    split_manifest = load_json(split_manifest_path)
    if not D1_SPLIT_BINDING_FIELDS <= set(split_manifest):
        raise ValueError("split manifest lacks the complete D1 provenance overlay")
    top_level_d1_binding = {
        field: split_manifest[field] for field in D1_SPLIT_BINDING_FIELDS
    }
    partitions = split_manifest.get("partitions")
    if not isinstance(partitions, list) or not partitions:
        raise ValueError("split manifest has no auditable partitions")
    for partition in partitions:
        if (
            not isinstance(partition, Mapping)
            or {field: partition.get(field) for field in D1_SPLIT_BINDING_FIELDS}
            != top_level_d1_binding
        ):
            raise ValueError(
                "partition D1 provenance overlay differs from the manifest"
            )
    split_kind = str(split_manifest.get("split_kind") or "")
    if split_kind == "study_disjoint" and split_manifest.get("folds") != partitions:
        raise ValueError("study folds differ from bound partitions")
    if (
        split_kind == "cross_region_transfer"
        and split_manifest.get("strata") != partitions
    ):
        raise ValueError("cross-region strata differ from bound partitions")

    expected_self_hashes = {
        "partitions_sha256": _stable_sha256(
            [
                {
                    "partition_id": partition.get("partition_id"),
                    "partition_sha256": partition.get("partition_sha256"),
                }
                for partition in partitions
            ]
        )
    }
    if split_kind == "study_disjoint":
        expected_self_hashes["folds_sha256"] = _stable_sha256(
            [
                {
                    "fold_id": partition.get("fold_id"),
                    "status": partition.get("status"),
                    "partition_sha256": partition.get("partition_sha256"),
                    "blocked_reasons": partition.get("blocked_reasons", []),
                }
                for partition in partitions
            ]
        )
    if split_kind == "cross_region_transfer":
        expected_self_hashes["strata_sha256"] = _stable_sha256(
            [
                {
                    "stratum_id": partition.get("stratum_id"),
                    "status": partition.get("status"),
                    "partition_sha256": partition.get("partition_sha256"),
                    "blocked_reasons": partition.get("blocked_reasons", []),
                }
                for partition in partitions
            ]
        )
    self_hash_fields = {
        "partitions_sha256",
        "folds_sha256",
        "strata_sha256",
    }
    if set(split_manifest) & self_hash_fields != set(expected_self_hashes) or any(
        split_manifest.get(field) != expected
        for field, expected in expected_self_hashes.items()
    ):
        raise ValueError("split manifest aggregate self-hash binding changed")

    for path_field, sha_field in (
        ("d1_acceptance_path", "d1_acceptance_sha256"),
        ("d1_ambiguity_report_path", "d1_ambiguity_report_sha256"),
        (
            "canonical_validation_report_path",
            "canonical_validation_report_sha256",
        ),
    ):
        bound_path = Path(str(split_manifest[path_field] or ""))
        if (
            not bound_path.is_absolute()
            or not bound_path.is_file()
            or sha256_file(bound_path) != split_manifest[sha_field]
        ):
            raise ValueError(f"split manifest {path_field} provenance is invalid")
    if expected_d1_acceptance_path is not None:
        expected_d1_acceptance_path = expected_d1_acceptance_path.resolve()
        if Path(
            str(split_manifest["d1_acceptance_path"])
        ).resolve() != expected_d1_acceptance_path or split_manifest[
            "d1_acceptance_sha256"
        ] != sha256_file(
            expected_d1_acceptance_path
        ):
            raise ValueError(
                "split manifest references a different D1 acceptance artifact"
            )

    declared_structural_path = Path(
        str(split_manifest.get("structural_records_path") or "")
    )
    if (
        not declared_structural_path.is_absolute()
        or declared_structural_path.resolve() != records_path
    ):
        raise ValueError("structural record path differs from split manifest")
    if not records_path.is_file():
        raise ValueError("structural record store is missing")
    cache = _cache or build_bound_structural_recompute_cache(records_path)
    if cache.records_path != records_path:
        raise ValueError("structural recomputation cache path differs from input")
    actual_structural_sha = sha256_file(records_path)
    actual_structural_bytes = records_path.stat().st_size
    if (
        actual_structural_sha != cache.records_sha256
        or actual_structural_bytes != cache.records_bytes
        or split_manifest.get("structural_records_sha256") != actual_structural_sha
        or split_manifest.get("structural_records_bytes") != actual_structural_bytes
    ):
        raise ValueError("structural record store differs from split manifest")
    structural_records: Sequence[Mapping[str, Any]] = cache.records
    if (
        split_manifest.get("structural_record_count") != cache.record_count
        or split_manifest.get("structural_record_ids_sha256") != cache.record_ids_sha256
        or split_manifest.get("structural_content_sha256")
        != cache.structural_content_sha256
    ):
        raise ValueError(
            "structural record count, identity, or content binding changed"
        )

    canonical_manifest = build_split_manifest(
        structural_records,
        split_kind=split_kind,
        region=split_manifest.get("region"),
        source_region=str(split_manifest.get("source_region") or "five_utr"),
        target_region=str(split_manifest.get("target_region") or "three_utr"),
        _near_neighbors=cache.near_neighbors,
    )
    canonical_core = canonical_split_manifest_core(canonical_manifest)
    supplied_core = canonical_split_manifest_core(split_manifest)
    if supplied_core != canonical_core:
        changed_top_level_fields = sorted(
            {
                *supplied_core.keys(),
                *canonical_core.keys(),
            }
            - {
                key
                for key in set(supplied_core) & set(canonical_core)
                if supplied_core[key] == canonical_core[key]
            }
        )
        raise ValueError(
            "split manifest differs from canonical structural recomputation: "
            + ", ".join(changed_top_level_fields[:20])
        )

    canonical_path = Path(str(split_manifest.get("canonical_records_path") or ""))
    if not canonical_path.is_absolute():
        raise ValueError("canonical record path must be absolute")
    if verify_canonical_store:
        if not canonical_path.is_file() or sha256_file(
            canonical_path
        ) != split_manifest.get("canonical_records_sha256"):
            raise ValueError(
                "canonical record store is missing or differs from split manifest"
            )
        canonical_records = load_jsonl(canonical_path)
        if (
            split_manifest.get("d1_phase_gate_passed") is not True
            or split_manifest.get("canonical_record_count") != len(canonical_records)
            or split_manifest.get("canonical_record_ids_sha256")
            != record_ids_sha256(canonical_records)
        ):
            raise ValueError(
                "D1 phase gate or canonical count/identity binding is invalid"
            )
    elif split_manifest.get("d1_phase_gate_passed") is not True:
        raise ValueError("D1 phase gate binding is invalid")

    report = audit_split_manifest(
        structural_records,
        split_manifest,
        foundation_exposure=foundation,
        _near_neighbors=cache.near_neighbors,
    )
    report["structural_records_path"] = str(records_path)
    report["structural_records_sha256"] = actual_structural_sha
    report["structural_records_bytes"] = actual_structural_bytes
    report["canonical_records_path"] = split_manifest.get("canonical_records_path")
    report["canonical_records_sha256"] = split_manifest.get("canonical_records_sha256")
    report["canonical_record_count"] = split_manifest.get("canonical_record_count")
    report["canonical_record_ids_sha256"] = split_manifest.get(
        "canonical_record_ids_sha256"
    )
    report["structural_record_count"] = split_manifest.get("structural_record_count")
    report["structural_record_ids_sha256"] = split_manifest.get(
        "structural_record_ids_sha256"
    )
    report["structural_content_sha256"] = split_manifest.get(
        "structural_content_sha256"
    )
    report["split_manifest_path"] = str(split_manifest_path)
    report["split_manifest_sha256"] = sha256_file(split_manifest_path)
    report["split_manifest_bytes"] = split_manifest_path.stat().st_size
    report["foundation_exposure_path"] = (
        str(foundation_exposure_path) if foundation_exposure_path is not None else None
    )
    report["foundation_exposure_sha256"] = (
        sha256_file(foundation_exposure_path)
        if foundation_exposure_path is not None
        else None
    )
    report["recomputed_from_bound_structural_records"] = True
    report["canonical_manifest_exact_recomputation"] = True
    report["canonical_manifest_core_sha256"] = _stable_sha256(canonical_core)
    auditor_path = Path(__file__).resolve()
    leakage_module_path = (
        Path(__file__).resolve().parents[2] / "data" / "utr_benchmark_v2" / "leakage.py"
    )
    report["auditor_binding"] = {
        "schema_version": "utr_b0_leakage_auditor.v2",
        "entrypoint_path": str(auditor_path),
        "entrypoint_sha256": sha256_file(auditor_path),
        "canonical_auditor_path": str(leakage_module_path.resolve()),
        "canonical_auditor_sha256": sha256_file(leakage_module_path),
    }
    return report


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--foundation-exposure", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    report = recompute_bound_leakage_report(
        args.records,
        args.split_manifest,
        foundation_exposure_path=args.foundation_exposure,
    )
    write_json_exclusive(args.output, report)
    return 0 if report["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
