"""Build a SHA-256 manifest for the current external-SOTA evidence bundle."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from typing import Mapping, Optional, Sequence


CLAIM_POLICY = (
    "This manifest proves file presence and byte-level identity for the listed "
    "external-SOTA evidence. It does not convert proxy metrics, budgeted "
    "external runs, or incomplete model coverage into a SOTA claim."
)

REQUIRED_PATHS = (
    "benchmark/external_sota/input_pack_t5_head1024/summary.json",
    "benchmark/external_sota/input_pack_t5_head1024/cds_protein_inputs.jsonl",
    "benchmark/external_sota/input_pack_t5_head1024/utr5_inputs.jsonl",
    "benchmark/external_sota/real_runs_t5_head1024/LinearDesign/summary.json",
    "benchmark/external_sota/real_runs_t5_head1024/LinearDesign/cds_outputs.jsonl",
    "benchmark/external_sota/real_runs_t5_head1024/UTRGAN/summary.json",
    "benchmark/external_sota/real_runs_t5_head1024/UTRGAN/utr5_outputs.jsonl",
    "benchmark/external_sota/real_runs_t5_head1024/UTRGAN_paper10000/summary.json",
    "benchmark/external_sota/real_runs_t5_head1024/UTRGAN_paper10000/utr5_outputs.jsonl",
    "benchmark/external_sota/real_runs_t5_head1024/UTailoR/summary.json",
    "benchmark/external_sota/real_runs_t5_head1024/UTailoR/utr5_outputs.jsonl",
    "benchmark/external_sota/real_runs_t5_head1024/codonGPT/summary.json",
    "benchmark/external_sota/real_runs_t5_head1024/codonGPT/cds_outputs.jsonl",
    "benchmark/external_sota/codongpt_multiseed_head1024/summary.json",
    "external_tools/codonGPT_hf_ee7017c4/model_manifest.json",
    "benchmark/utr_local_search_head1024.json",
    "benchmark/utr_local_search_head1024.records.jsonl",
    "benchmark/utailor_strict_25_100_sources.jsonl",
    "benchmark/utailor_strict_25_100_sources.summary.json",
    "benchmark/multiseed_t5_public_head1024_region_adapter_utr5only_top64/multiseed_summary.json",
    "benchmark/multiseed_t5_public_head1024_utr_teacher_utr5only_top64/multiseed_summary.json",
    "benchmark/multiseed_t5_public_head1024_seq_full_then_utr_utr5only_top64/multiseed_summary.json",
    "benchmark/multiseed_t5_utailor_strict315_pure_utr_teacher_budget5_top64/multiseed_summary.json",
    "docs/external_sota_real_run_audit.json",
    "docs/t4_external_cds_baseline_comparison.json",
    "docs/t5_external_utr_baseline_comparison.json",
    "docs/paper_table3_external_baseline_readiness.json",
    "docs/sota_readiness_audit_head256.json",
    "docs/sota_gap_report.json",
    "README.md",
    "docs/next_steps_sota_roadmap.md",
    "docs/reproduce_full_training_eval_commands.md",
    "baselines/external_lineardesign_adapter.py",
    "baselines/external_ensembledesign_adapter.py",
    "baselines/external_codongpt_adapter.py",
    "baselines/external_utailor_adapter.py",
    "baselines/external_utailor_runner.py",
    "baselines/external_utrgan_adapter.py",
    "baselines/utr_local_search.py",
    "eval/audit_external_sota_real_runs.py",
    "eval/build_external_sota_evidence_manifest.py",
    "eval/build_codongpt_multiseed_summary.py",
    "eval/build_t4_external_cds_comparison.py",
    "eval/build_t5_external_utr_comparison.py",
    "eval/run_multiseed_benchmark.py",
    "sample.py",
    "scripts/run_external_lineardesign_head1024.sh",
    "scripts/run_external_ensembledesign_head1024.sh",
    "scripts/external_codongpt.sh",
    "scripts/run_external_codongpt_head1024.sh",
    "scripts/run_external_codongpt_multiseed_head1024.sh",
    "scripts/setup_external_codongpt.sh",
    "scripts/run_external_utrgan_paper10000.sh",
    "scripts/run_external_utailor_head1024.sh",
    "scripts/run_external_utrgan_head1024.sh",
    "scripts/run_mef_utr5only_head1024.sh",
    "scripts/run_mef_utailor_subset_budget5.sh",
    "scripts/watch_external_ensembledesign_retry.sh",
)

EVIDENCE_GLOBS = (
    "benchmark/multiseed_t5_public_head1024_region_adapter_utr5only_top64/**/seed_*/candidates.jsonl",
    "benchmark/multiseed_t5_public_head1024_region_adapter_utr5only_top64/**/seed_*/eval_summary.json",
    "benchmark/multiseed_t5_public_head1024_utr_teacher_utr5only_top64/**/seed_*/candidates.jsonl",
    "benchmark/multiseed_t5_public_head1024_utr_teacher_utr5only_top64/**/seed_*/eval_summary.json",
    "benchmark/multiseed_t5_public_head1024_seq_full_then_utr_utr5only_top64/**/seed_*/candidates.jsonl",
    "benchmark/multiseed_t5_public_head1024_seq_full_then_utr_utr5only_top64/**/seed_*/eval_summary.json",
    "benchmark/multiseed_t5_utailor_strict315_pure_utr_teacher_budget5_top64/**/seed_*/candidates.jsonl",
    "benchmark/multiseed_t5_utailor_strict315_pure_utr_teacher_budget5_top64/**/seed_*/eval_summary.json",
    "benchmark/external_sota/codongpt_multiseed_head1024/seed_*/summary.json",
    "benchmark/external_sota/codongpt_multiseed_head1024/seed_*/cds_outputs.jsonl",
)

OPTIONAL_ACTIVE_PATHS = (
    "benchmark/external_sota/real_runs_t5_head1024/EnsembleDesign.status.json",
    "benchmark/external_sota/real_runs_t5_head1024/EnsembleDesign/progress.jsonl",
    "benchmark/external_sota/real_runs_t5_head1024/EnsembleDesign/cds_outputs.jsonl",
    "benchmark/external_sota/real_runs_t5_head1024/EnsembleDesign/failures.jsonl",
    "benchmark/external_sota/real_runs_t5_head1024/EnsembleDesign/summary.json",
)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str) -> Optional[Mapping[str, object]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload if isinstance(payload, Mapping) else None


def _entry(root: str, relative_path: str, category: str) -> dict[str, object]:
    path = os.path.join(root, relative_path)
    exists = os.path.isfile(path)
    return {
        "path": relative_path,
        "category": category,
        "exists": exists,
        "size_bytes": os.path.getsize(path) if exists else None,
        "sha256": _sha256(path) if exists else None,
    }


def _required_bundle_digest(
    rows: Sequence[Mapping[str, object]],
) -> str:
    canonical_rows = [
        {
            "path": str(row.get("path") or ""),
            "sha256": row.get("sha256"),
        }
        for row in sorted(rows, key=lambda row: str(row.get("path") or ""))
    ]
    payload = json.dumps(
        canonical_rows,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_external_sota_evidence_manifest(
    project_root: str,
) -> dict[str, object]:
    root = os.path.abspath(project_root)
    required = [_entry(root, path, "required") for path in REQUIRED_PATHS]
    discovered_paths: set[str] = set()
    for pattern in EVIDENCE_GLOBS:
        for path in glob.glob(os.path.join(root, pattern), recursive=True):
            if os.path.isfile(path):
                discovered_paths.add(os.path.relpath(path, root))
    discovered = [
        _entry(root, path, "seed_level_evidence")
        for path in sorted(discovered_paths)
    ]
    active = [
        _entry(root, path, "optional_active_external_run")
        for path in OPTIONAL_ACTIVE_PATHS
    ]

    real_run = _load_json(
        os.path.join(root, "docs", "external_sota_real_run_audit.json")
    )
    real_summary = (
        real_run.get("summary", {}) if isinstance(real_run, Mapping) else {}
    )
    real_summary = (
        real_summary if isinstance(real_summary, Mapping) else {}
    )
    t5 = _load_json(
        os.path.join(root, "docs", "t5_external_utr_baseline_comparison.json")
    )
    t5_summary = t5.get("summary", {}) if isinstance(t5, Mapping) else {}
    t5_summary = t5_summary if isinstance(t5_summary, Mapping) else {}
    ensemble_status = _load_json(
        os.path.join(
            root,
            "benchmark/external_sota/real_runs_t5_head1024/"
            "EnsembleDesign.status.json",
        )
    )
    all_entries = required + discovered + active
    missing_required = [
        row["path"] for row in required if row.get("exists") is not True
    ]
    hashed = [
        row
        for row in all_entries
        if row.get("exists") is True and row.get("sha256")
    ]
    return {
        "artifact_kind": "external_sota_evidence_sha256_manifest",
        "project_root": root,
        "claim_policy": CLAIM_POLICY,
        "summary": {
            "n_required": len(required),
            "n_required_present": len(required) - len(missing_required),
            "n_seed_level_files": len(discovered),
            "n_active_external_files_present": sum(
                row.get("exists") is True for row in active
            ),
            "n_files_hashed": len(hashed),
            "total_hashed_bytes": sum(
                int(row.get("size_bytes") or 0) for row in hashed
            ),
            "missing_required": missing_required,
            "required_bundle_sha_complete": not missing_required,
            "required_bundle_digest_sha256": _required_bundle_digest(
                required
            ),
            "n_external_models_measured": real_summary.get(
                "n_models_measured"
            ),
            "n_external_models_expected": real_summary.get(
                "n_models_expected"
            ),
            "external_real_metric_table_complete": bool(
                real_summary.get("ready_for_external_real_metric_table")
            ),
            "t5_utr_descriptive_table_ready": bool(
                t5_summary.get("ready_for_t5_utr_descriptive_table")
            ),
            "t5_mef_superiority_claim_ready": bool(
                t5_summary.get("ready_for_mef_superiority_claim")
            ),
            "ensemble_design_status": (
                ensemble_status.get("status")
                if isinstance(ensemble_status, Mapping)
                else "not_present"
            ),
            "external_sota_evidence_bundle_complete": bool(
                not missing_required
                and real_summary.get(
                    "ready_for_external_real_metric_table"
                )
                is True
            ),
            "ready_for_external_sota_claim": False,
        },
        "files": all_entries,
    }


def write_json(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    return path


def write_markdown(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    summary = report.get("summary", {})
    summary = summary if isinstance(summary, Mapping) else {}
    files = report.get("files", [])
    files = files if isinstance(files, list) else []
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# External SOTA Evidence SHA-256 Manifest\n\n")
        fh.write(f"- Claim policy: {report.get('claim_policy')}\n")
        fh.write(
            f"- Required present: `{summary.get('n_required_present')}` / "
            f"`{summary.get('n_required')}`; seed-level files: "
            f"`{summary.get('n_seed_level_files')}`; files hashed: "
            f"`{summary.get('n_files_hashed')}`\n"
        )
        fh.write(
            f"- External measured: `{summary.get('n_external_models_measured')}` / "
            f"`{summary.get('n_external_models_expected')}`; evidence bundle "
            f"complete: `{summary.get('external_sota_evidence_bundle_complete')}`; "
            f"SOTA claim ready: `{summary.get('ready_for_external_sota_claim')}`\n"
        )
        fh.write(
            "- Required bundle digest SHA-256: "
            f"`{summary.get('required_bundle_digest_sha256')}`\n\n"
        )
        fh.write("| Category | Path | Exists | Bytes | SHA-256 |\n")
        fh.write("|---|---|---:|---:|---|\n")
        for row in files:
            if not isinstance(row, Mapping):
                continue
            fh.write(
                f"| {row.get('category')} | `{row.get('path')}` | "
                f"`{row.get('exists')}` | {row.get('size_bytes') or 'NA'} | "
                f"`{row.get('sha256') or 'NA'}` |\n"
            )
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument(
        "--out-json",
        default="docs/external_sota_evidence_manifest.json",
    )
    parser.add_argument(
        "--out-md",
        default="docs/external_sota_evidence_manifest.md",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    root = os.path.abspath(args.project_root)
    out_json = (
        args.out_json
        if os.path.isabs(args.out_json)
        else os.path.join(root, args.out_json)
    )
    out_md = (
        args.out_md
        if os.path.isabs(args.out_md)
        else os.path.join(root, args.out_md)
    )
    report = build_external_sota_evidence_manifest(root)
    write_json(report, out_json)
    write_markdown(report, out_md)
    print(
        json.dumps(
            {
                "json_path": out_json,
                "markdown_path": out_md,
                "summary": report["summary"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLAIM_POLICY",
    "build_external_sota_evidence_manifest",
    "write_json",
    "write_markdown",
    "main",
]
