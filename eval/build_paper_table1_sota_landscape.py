"""Build paper Table 1: external SOTA landscape and MEF alignment.

This table is intentionally a landscape/protocol artifact. It does not report
external performance metrics unless an executable adapter and measured outputs
exist. Its job is to classify methods by task family and make the remaining
alignment work explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from typing import Mapping, Optional, Sequence

from mrna_editflow.eval.artifact_contract import normalize_run_mode, paper_builder_gate, validate_report_output_namespaces, write_paper_report_sidecars

from mrna_editflow.baselines.external_models import available_external_models
from mrna_editflow.eval.sota_gap_report import SOTA_REFERENCE_ROWS


CLAIM_POLICY = (
    "Table 1 is a SOTA landscape/protocol table. It may describe published "
    "scope and reported signals, but it must not claim MEF has reproduced or "
    "outperformed an external method until official weights/executables and "
    "measured outputs are configured."
)
CATEGORY_BY_METHOD: dict[str, str] = {
    "LinearDesign": "CDS-only structure/codon optimization",
    "EnsembleDesign": "CDS-only structure/codon optimization",
    "codonGPT": "CDS/protein-conditioned generation",
    "Prot2RNA": "CDS/protein-conditioned generation",
    "UTailoR": "UTR-only optimization",
    "GEMORNA": "full-length de novo generation",
    "mRNA-GPT": "full-length de novo generation",
    "ProMORNA": "protein-conditioned full-length generation",
    "RNAGenScape": "existing-mRNA property-guided optimization",
    "mRNA-LM": "full-length foundation/representation",
    "Helix-mRNA": "full-length foundation/representation",
    "CodonFM": "codon foundation/representation",
}
DEFAULT_SURVEY_PATH = "docs/mrna_dataset_survey.md"
DEFAULT_DRY_RUN_PATH = "benchmark/external_sota/dry_run_t5_head1024/summary.json"


def _load_json(path: str) -> Optional[Mapping[str, object]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _sha256_file(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(project_root: str, rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(project_root, rel)


def _survey_audit(project_root: str, rel_path: str = DEFAULT_SURVEY_PATH) -> dict[str, object]:
    path = _path(project_root, rel_path)
    if not os.path.exists(path):
        return {
            "path": rel_path,
            "exists": False,
            "sha256": None,
            "covered_methods": [],
            "mentions_split_or_leakage": False,
            "protocol_ready": False,
        }
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    methods = [row["method"] for row in SOTA_REFERENCE_ROWS]
    covered = [method for method in methods if method in text]
    return {
        "path": rel_path,
        "exists": True,
        "sha256": _sha256_file(path),
        "covered_methods": covered,
        "mentions_split_or_leakage": any(term in text for term in ("split", "leakage", "license", "许可")),
        "protocol_ready": bool(covered) and any(term in text for term in ("split", "leakage", "license", "许可")),
    }


def _dry_run_rows(project_root: str, rel_path: str = DEFAULT_DRY_RUN_PATH) -> dict[str, Mapping[str, object]]:
    payload = _load_json(_path(project_root, rel_path))
    if payload is None:
        return {}
    rows = payload.get("rows", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return {}
    out: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if isinstance(row, Mapping) and isinstance(row.get("model_name"), str):
            out[str(row["model_name"])] = row
    return out


def _category(method: str, scope: str) -> str:
    if method in CATEGORY_BY_METHOD:
        return CATEGORY_BY_METHOD[method]
    lowered = scope.lower()
    if "5'utr" in lowered or "utr" in lowered:
        return "UTR-only optimization"
    if "full" in lowered:
        return "full-length foundation/representation"
    if "cds" in lowered or "codon" in lowered:
        return "CDS/protein-conditioned generation"
    return "other RNA/mRNA method"


def _reproduction_status(method: str, dry_row: Optional[Mapping[str, object]]) -> str:
    if dry_row is None:
        return "landscape_only_no_adapter"
    status = dry_row.get("status")
    if status == "executable_ready":
        return "executable_ready_real_metrics_still_required"
    if status == "not_configured":
        return "protocol_only_not_configured"
    return f"unknown_dry_run_status:{status}"


def _mef_alignment(method: str, category: str, mef_gap: str) -> str:
    if category == "full-length de novo generation":
        return "Need head-to-head full-length T1-T7 with TE delta and hard-constraint satisfaction."
    if category == "protein-conditioned full-length generation":
        return "Need shared held-out protein Pareto frontier for TE/half-life/objectives."
    if category == "existing-mRNA property-guided optimization":
        return "Need shared 5'UTR/property optimization trajectory and success-rate protocol."
    if category == "UTR-only optimization":
        return "Need UTR-only T5/T7 baseline while MEF preserves CDS/3'UTR constraints."
    if category == "CDS-only structure/codon optimization":
        return "Need external CDS-only CAI/MFE/wall-clock run; local codon-lattice DP is proxy only."
    if category == "CDS/protein-conditioned generation":
        return "Need CDS/protein-conditioned generation comparison with protein identity and CAI/GC metrics."
    if "foundation" in category:
        return "Need leakage-audited frozen embeddings or official checkpoints under matched trainable budget."
    return mef_gap


def _build_paper_table1_development(project_root: str) -> dict[str, object]:
    survey = _survey_audit(project_root)
    dry_rows = _dry_run_rows(project_root)
    registered = set(available_external_models())
    survey_covered = set(survey.get("covered_methods", []))
    rows = []
    for raw in SOTA_REFERENCE_ROWS:
        method = raw["method"]
        category = _category(method, raw.get("scope", ""))
        dry_row = dry_rows.get(method)
        executable_status = dry_row.get("status") if dry_row is not None else "not_in_dry_run"
        rows.append(
            {
                "method": method,
                "category": category,
                "venue_year": raw.get("venue_year", ""),
                "scope": raw.get("scope", ""),
                "reported_signal": raw.get("reported_signal", ""),
                "metric_family": raw.get("accuracy_f1", ""),
                "speed_or_scale": raw.get("speed_scale", ""),
                "citation_url": raw.get("citation_url", ""),
                "registered_external_model": method in registered,
                "dataset_survey_covered": method in survey_covered,
                "dry_run_status": executable_status,
                "reproduction_status": _reproduction_status(method, dry_row),
                "mef_alignment_action": _mef_alignment(method, category, raw.get("mef_gap", "")),
                "claim_language": "landscape_only_no_measured_external_metric",
            }
        )
    category_counts = dict(Counter(row["category"] for row in rows))
    n_executable_ready = sum(1 for row in rows if row["dry_run_status"] == "executable_ready")
    n_not_configured = sum(1 for row in rows if row["dry_run_status"] == "not_configured")
    return {
        "artifact_kind": "paper_table1_sota_landscape",
        "project_root": os.path.abspath(project_root),
        "claim_policy": CLAIM_POLICY,
        "summary": {
            "n_methods": len(rows),
            "n_categories": len(category_counts),
            "category_counts": category_counts,
            "dataset_survey_protocol_ready": survey.get("protocol_ready"),
            "n_dataset_survey_covered": sum(1 for row in rows if row["dataset_survey_covered"]),
            "n_registered_external_models": sum(1 for row in rows if row["registered_external_model"]),
            "n_dry_run_models": len(dry_rows),
            "n_executable_ready": n_executable_ready,
            "n_not_configured": n_not_configured,
            "ready_for_landscape_table": bool(rows) and bool(survey.get("exists")),
            "ready_for_external_metric_claim": False,
            "ready_for_wet_lab_claim": False,
        },
        "dataset_survey_audit": survey,
        "rows": rows,
    }


def build_paper_table1(
    project_root: str,
    run_mode: str = "development",
    artifact_paths: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    if normalize_run_mode(run_mode) == "paper":
        return paper_builder_gate("paper_table1_sota_landscape", project_root, artifact_paths, __file__)
    report = _build_paper_table1_development(project_root)
    report.update({"claim_tier": "development_only", "paper_eligible": False})
    return report


def write_report_json(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    return path


def write_report_markdown(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rows = report.get("rows", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        rows = []
    summary = report.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Paper Table 1: SOTA Landscape\n\n")
        fh.write(f"- Claim policy: {report.get('claim_policy', CLAIM_POLICY)}\n")
        fh.write(
            f"- Methods: `{summary.get('n_methods')}`; categories: `{summary.get('n_categories')}`; "
            f"dry-run executable ready: `{summary.get('n_executable_ready')}` / `{summary.get('n_dry_run_models')}`\n"
        )
        fh.write(
            f"- Dataset survey protocol ready: `{summary.get('dataset_survey_protocol_ready')}`; "
            f"external metric claim ready: `{summary.get('ready_for_external_metric_claim')}`; "
            f"wet-lab claim ready: `{summary.get('ready_for_wet_lab_claim')}`\n\n"
        )
        fh.write("| Category | Method | Venue/year | Scope | Public signal | MEF alignment action | Status |\n")
        fh.write("|---|---|---|---|---|---|---|\n")
        for row in sorted(rows, key=lambda item: (str(item.get("category")), str(item.get("method")))):
            if not isinstance(row, Mapping):
                continue
            status = (
                f"survey={row.get('dataset_survey_covered')}; "
                f"dry_run={row.get('dry_run_status')}; "
                f"claim={row.get('claim_language')}"
            )
            fh.write(
                f"| {row.get('category')} | {row.get('method')} | {row.get('venue_year')} | "
                f"{row.get('scope')} | {row.get('reported_signal')} | "
                f"{row.get('mef_alignment_action')} | {status} |\n"
            )
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--out-json", default="docs/paper_table1_sota_landscape.json")
    parser.add_argument("--out-md", default="docs/paper_table1_sota_landscape.md")
    parser.add_argument("--run-mode", choices=("development", "paper"), default="development")
    parser.add_argument("--paper-artifact", action="append", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    project_root = os.path.abspath(args.project_root)
    report = build_paper_table1(project_root, args.run_mode, args.paper_artifact)
    out_json = args.out_json if os.path.isabs(args.out_json) else os.path.join(project_root, args.out_json)
    out_md = args.out_md if os.path.isabs(args.out_md) else os.path.join(project_root, args.out_md)
    validate_report_output_namespaces(report, (out_json, out_md))
    write_report_json(report, out_json)
    write_report_markdown(report, out_md)
    write_paper_report_sidecars(report, (out_json, out_md))
    print(json.dumps({"json_path": out_json, "markdown_path": out_md}, sort_keys=True))
    if args.run_mode == "paper" and args.paper_artifact and not report["paper_eligible"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLAIM_POLICY",
    "CATEGORY_BY_METHOD",
    "build_paper_table1",
    "write_report_json",
    "write_report_markdown",
    "main",
]
