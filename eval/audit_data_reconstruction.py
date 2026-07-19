"""Read-only verifier for P0 Data Reconstruction v1 artifacts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping, Optional, Sequence

from mrna_editflow.data.reconstruction import sha256_file, verify_raw_artifact, verify_source_bundle
from mrna_editflow.data.split_contract import load_and_verify_split_manifest


def _load_json(path: str | Path) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_baseline_hashes(path: Optional[str | Path]) -> dict[str, str]:
    if path is None:
        return {}
    rows: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                rows[parts[1].lstrip("*./")] = parts[0]
    return rows


def _process_evidence(pid: int) -> dict[str, object]:
    proc = Path("/proc") / str(int(pid))
    if not proc.is_dir():
        return {"pid": int(pid), "exists": False}
    command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    state = ""
    for line in (proc / "status").read_text(encoding="utf-8").splitlines():
        if line.startswith("State:"):
            state = line.split(":", 1)[1].strip()
            break
    return {"pid": int(pid), "exists": True, "state": state, "command": command}


def audit_data_reconstruction(
    *,
    project_root: str | Path,
    frozen_root: str | Path,
    split_root: str | Path,
    baseline_hashes_path: Optional[str | Path] = None,
    protected_pids: Sequence[int] = (),
) -> dict[str, object]:
    project = Path(project_root).resolve()
    frozen = Path(frozen_root).resolve()
    splits = Path(split_root).resolve()
    source_paths = {
        "gencode_v45": frozen / "sources/gencode_v45/reconstruction_manifest.json",
        "refseq_human_rna": frozen / "sources/refseq_human_rna/reconstruction_manifest.json",
    }
    sources: dict[str, object] = {}
    for name, manifest_path in source_paths.items():
        manifest = verify_source_bundle(manifest_path)
        raw_artifacts = [
            verify_raw_artifact(
                artifact["path"],
                expected_sha256=artifact["sha256"],
                expected_size_bytes=artifact["size_bytes"],
            )
            for artifact in manifest["raw"]["artifacts"]
        ]
        sources[name] = {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "raw": {
                "artifact_count": len(raw_artifacts),
                "artifacts": raw_artifacts,
                "total_size_bytes": sum(int(row["size_bytes"]) for row in raw_artifacts),
                "sha256": raw_artifacts[0]["sha256"] if len(raw_artifacts) == 1 else None,
            },
            "canonical_count": manifest["canonical"]["count"],
            "canonical_records_sha256": manifest["canonical"]["records_sha256"],
            "model_view_count": manifest["derived_views"]["model_capped_v1"]["count"],
            "model_view_records_sha256": manifest["derived_views"]["model_capped_v1"]["records_sha256"],
            "canonical_stats": manifest["canonical"]["stats"],
            "model_view_stats": manifest["derived_views"]["model_capped_v1"]["stats"],
        }

    combined_path = frozen / "combined/combined_reconstruction_manifest.json"
    combined = _load_json(combined_path)
    combined_records = combined["combined_records"]
    combined_metadata = combined["combined_metadata"]
    families = combined["families"]
    for section, path_key, sha_key in (
        (combined_records, "path", "sha256"),
        (combined_metadata, "path", "sha256"),
        (families, "assignments_path", "assignments_sha256"),
        (families, "evidence_path", "evidence_sha256"),
    ):
        if sha256_file(section[path_key]) != section[sha_key]:
            raise ValueError(f"combined artifact SHA mismatch: {path_key}")

    split_rows: dict[str, object] = {}
    for name in ("gencode_family", "refseq_family", "combined_family", "gencode_to_refseq"):
        manifest_path = splits / name / "split_manifest.json"
        contract = load_and_verify_split_manifest(str(manifest_path))
        split_rows[name] = {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "records_count": contract.records_count,
            "role_counts": {role: contract.roles[role].count for role in ("train", "val", "test")},
            "excluded_count": contract.excluded.count if contract.excluded is not None else 0,
            "family_disjoint": contract.family_disjoint,
            "near_neighbor_threshold_passed": contract.near_neighbor_threshold_passed,
            "paper_eligible": contract.paper_eligible,
            "block_reasons": list(contract.block_reasons),
        }

    baseline_expected = _read_baseline_hashes(baseline_hashes_path)
    legacy: dict[str, object] = {}
    for relative, expected in baseline_expected.items():
        path = project / relative
        actual = sha256_file(path) if path.is_file() else None
        legacy[relative] = {
            "exists": path.is_file(),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "unchanged": actual == expected,
        }
    legacy_unchanged = all(bool(row["unchanged"]) for row in legacy.values())
    legacy_gencode_records = project / "data/processed/gencode_human_transcripts.records.jsonl"
    legacy_gencode_sha = sha256_file(legacy_gencode_records)
    reconstructed_gencode_sha = sources["gencode_v45"]["model_view_records_sha256"]
    processes = [_process_evidence(pid) for pid in protected_pids]
    process_set_intact = all(bool(row.get("exists")) for row in processes)
    all_blockers = sorted(
        {
            reason
            for row in split_rows.values()
            for reason in row["block_reasons"]
        }
    )
    return {
        "schema_version": 1,
        "artifact_kind": "p0_data_reconstruction_read_only_audit",
        "project_root": str(project),
        "frozen_root": str(frozen),
        "split_root": str(splits),
        "status": "reconstruction_complete",
        "sources": sources,
        "combined": {
            "manifest_path": str(combined_path),
            "manifest_sha256": sha256_file(combined_path),
            "records": combined_records,
            "families": families,
        },
        "split_manifests": split_rows,
        "legacy_artifacts": legacy,
        "legacy_artifacts_unchanged": legacy_unchanged,
        "gencode_legacy_model_view": {
            "legacy_path": str(legacy_gencode_records),
            "legacy_sha256": legacy_gencode_sha,
            "reconstructed_model_view_sha256": reconstructed_gencode_sha,
            "byte_reproduced": legacy_gencode_sha == reconstructed_gencode_sha,
        },
        "protected_processes": processes,
        "protected_process_set_intact": process_set_intact,
        "process_control_actions_on_protected_pids": [],
        "paper_eligible": False,
        "block_reasons": all_blockers,
        "claim_policy": (
            "This goal reconstructs and freezes data evidence. It does not claim a "
            "paper-eligible split until exhaustive cross-role near-neighbour and gene-alias audits pass."
        ),
    }


def write_audit(report: Mapping[str, object], out_json: str | Path, out_md: str | Path) -> None:
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    sources = report["sources"]
    splits = report["split_manifests"]
    lines = [
        "# P0 Data Reconstruction v1 audit",
        "",
        f"- Status: `{report['status']}`",
        f"- Legacy artifacts unchanged: `{report['legacy_artifacts_unchanged']}`",
        f"- GENCODE legacy model view byte-reproduced: `{report['gencode_legacy_model_view']['byte_reproduced']}`",
        f"- Protected process set intact: `{report['protected_process_set_intact']}`",
        f"- Paper eligible: `{report['paper_eligible']}`",
        "",
        "## Sources",
        "",
        "| Source | Canonical | Model view | Raw SHA-256 |",
        "|---|---:|---:|---|",
    ]
    for name, row in sources.items():
        lines.append(
            f"| `{name}` | {row['canonical_count']} | {row['model_view_count']} | "
            f"`{row['raw']['sha256'] or str(row['raw']['artifact_count']) + ' files'}` |"
        )
    lines.extend(["", "## Frozen split contracts", "", "| Split | Train | Val | Test | Excluded | Paper eligible |", "|---|---:|---:|---:|---:|---:|"])
    for name, row in splits.items():
        counts = row["role_counts"]
        lines.append(
            f"| `{name}` | {counts['train']} | {counts['val']} | {counts['test']} | {row['excluded_count']} | {row['paper_eligible']} |"
        )
    lines.extend(["", "## Open blockers", ""])
    lines.extend(f"- `{reason}`" for reason in report["block_reasons"])
    lines.extend(["", str(report["claim_policy"])])
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--frozen-root", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--baseline-hashes", default=None)
    parser.add_argument("--protected-pids", default="")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    pids = [int(value) for value in args.protected_pids.split(",") if value.strip()]
    report = audit_data_reconstruction(
        project_root=args.project_root,
        frozen_root=args.frozen_root,
        split_root=args.split_root,
        baseline_hashes_path=args.baseline_hashes,
        protected_pids=pids,
    )
    write_audit(report, args.out_json, args.out_md)
    print(json.dumps({"status": report["status"], "out_json": args.out_json, "out_md": args.out_md}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_data_reconstruction", "write_audit", "main"]
