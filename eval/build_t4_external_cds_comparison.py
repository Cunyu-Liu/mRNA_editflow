"""Build the T4 CDS comparison table for MEF and external optimizers.

The table joins existing head1024 MEF/codon-lattice artifacts with audited
LinearDesign, EnsembleDesign, and codonGPT measurements. It is descriptive
until per-record paired tests and matched objectives are available.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Mapping, Optional, Sequence


CLAIM_POLICY = (
    "This table supports descriptive T4 CDS comparison only. Do not claim MEF "
    "superiority over LinearDesign, EnsembleDesign, or codonGPT until matched "
    "per-record paired tests, common objectives, and complete external outputs "
    "are available."
)


def _load(path: str) -> Optional[Mapping[str, object]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _num(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    return float(value) if isinstance(value, (int, float)) else None


def build_t4_external_cds_comparison(project_root: str) -> dict[str, object]:
    paths = {
        "mef_summary": "benchmark/protein_conditioned_cds_head1024.summary.json",
        "mef_codon_metrics": "benchmark/protein_conditioned_codon_metrics_head1024.json",
        "codon_dp": "benchmark/codon_lattice_dp_head1024.json",
        "linear_summary": "benchmark/external_sota/real_runs_t5_head1024/LinearDesign/summary.json",
        "ensemble_summary": "benchmark/external_sota/real_runs_t5_head1024/EnsembleDesign/summary.json",
        "codongpt_summary": "benchmark/external_sota/real_runs_t5_head1024/codonGPT/summary.json",
        "codongpt_multiseed": "benchmark/external_sota/codongpt_multiseed_head1024/summary.json",
        "real_run_audit": "docs/external_sota_real_run_audit.json",
    }
    payloads = {name: _load(os.path.join(project_root, rel)) for name, rel in paths.items()}
    mef = _mapping(_mapping(payloads["mef_summary"]).get("summary"))
    mef_codon = _mapping(_mapping(payloads["mef_codon_metrics"]).get("summary"))
    codon_dp = _mapping(_mapping(payloads["codon_dp"]).get("summary"))
    linear = _mapping(payloads["linear_summary"])
    ensemble = _mapping(payloads["ensemble_summary"])
    codongpt = _mapping(payloads["codongpt_summary"])
    codongpt_multiseed = _mapping(payloads["codongpt_multiseed"])
    codongpt_multiseed_status = _mapping(
        codongpt_multiseed.get("summary")
    )
    codongpt_multiseed_aggregate = _mapping(
        codongpt_multiseed.get("aggregate")
    )
    audit_rows = _mapping(payloads["real_run_audit"]).get("rows", [])
    if not isinstance(audit_rows, list):
        audit_rows = []
    linear_audit = next(
        (
            row
            for row in audit_rows
            if isinstance(row, Mapping) and row.get("model_name") == "LinearDesign"
        ),
        {},
    )
    linear_measured = bool(
        isinstance(linear_audit, Mapping)
        and linear_audit.get("status") == "measured"
        and linear_audit.get("real_metric_ready") is True
    )
    ensemble_audit = next(
        (
            row
            for row in audit_rows
            if isinstance(row, Mapping)
            and row.get("model_name") == "EnsembleDesign"
        ),
        {},
    )
    ensemble_measured = bool(
        isinstance(ensemble_audit, Mapping)
        and ensemble_audit.get("status") == "measured"
        and ensemble_audit.get("real_metric_ready") is True
    )
    codongpt_audit = next(
        (
            row
            for row in audit_rows
            if isinstance(row, Mapping)
            and row.get("model_name") == "codonGPT"
        ),
        {},
    )
    codongpt_measured = bool(
        isinstance(codongpt_audit, Mapping)
        and codongpt_audit.get("status") == "measured"
        and codongpt_audit.get("real_metric_ready") is True
    )
    codongpt_10seed_complete = bool(
        codongpt_multiseed_status.get("complete_10seed_head1024") is True
        and codongpt_multiseed_status.get("hard_constraints_exact_1") is True
    )

    def codongpt_metric(metric: str, fallback: object) -> Optional[float]:
        stats = _mapping(codongpt_multiseed_aggregate.get(metric))
        value = _num(stats.get("mean")) if codongpt_10seed_complete else None
        return value if value is not None else _num(fallback)

    def codongpt_delta(
        metric: str,
        value_metric: str,
        fallback: object,
        native: Optional[float],
    ) -> Optional[float]:
        stats = _mapping(codongpt_multiseed_aggregate.get(metric))
        delta = _num(stats.get("mean")) if codongpt_10seed_complete else None
        value = codongpt_metric(value_metric, fallback)
        if delta is not None:
            return delta
        return value - native if value is not None and native is not None else None

    native_cai = _num(mef.get("mean_native_cai"))
    native_gc = _num(mef.get("mean_native_gc"))
    rows = [
        {
            "method": "native_source",
            "status": "measured_internal_reference" if mef else "missing",
            "n": mef.get("n"),
            "mean_cai": native_cai,
            "delta_cai_vs_native": 0.0 if native_cai is not None else None,
            "mean_gc": native_gc,
            "delta_gc_vs_native": 0.0 if native_gc is not None else None,
            "mean_gc3": _num(mef_codon.get("mean_native_gc3")),
            "codon_accuracy_vs_native": 1.0,
            "codon_usage_kl_vs_native": 0.0,
            "codon_pair_kl_vs_native": 0.0,
            "protein_identity_exact_1_fraction": 1.0,
            "mean_mfe": None,
            "mean_ensemble_free_energy": None,
            "mean_wall_clock_s": None,
        },
        {
            "method": "MEF_protein_conditioned_CDS",
            "status": "measured_internal_proxy" if mef and mef_codon else "missing",
            "n": mef.get("n"),
            "mean_cai": _num(mef.get("mean_designed_cai")),
            "delta_cai_vs_native": _num(mef.get("mean_designed_vs_native_cai_delta")),
            "mean_gc": _num(mef.get("mean_designed_gc")),
            "delta_gc_vs_native": _num(mef.get("mean_designed_vs_native_gc_delta")),
            "mean_gc3": _num(mef_codon.get("mean_designed_gc3")),
            "codon_accuracy_vs_native": _num(
                mef_codon.get("mean_native_codon_recovery")
            ),
            "codon_usage_kl_vs_native": _num(
                mef_codon.get("designed_vs_native_codon_usage_kl")
            ),
            "codon_pair_kl_vs_native": _num(
                mef_codon.get("designed_vs_native_codon_pair_kl")
            ),
            "protein_identity_exact_1_fraction": _num(
                mef.get("protein_identity_eq_1_fraction")
            ),
            "mean_mfe": None,
            "mean_ensemble_free_energy": None,
            "mean_wall_clock_s": None,
        },
        {
            "method": "MEF_codon_lattice_DP_budget3",
            "status": "measured_internal_proxy" if codon_dp else "missing",
            "n": codon_dp.get("n"),
            "mean_cai": _num(codon_dp.get("mean_optimized_cai")),
            "delta_cai_vs_native": _num(codon_dp.get("mean_delta_cai")),
            "mean_gc": _num(codon_dp.get("mean_optimized_gc")),
            "delta_gc_vs_native": _num(codon_dp.get("mean_delta_gc")),
            "mean_gc3": None,
            "codon_accuracy_vs_native": None,
            "codon_usage_kl_vs_native": None,
            "codon_pair_kl_vs_native": None,
            "protein_identity_exact_1_fraction": _num(
                codon_dp.get("protein_identity_fraction")
            ),
            "mean_mfe": None,
            "mean_ensemble_free_energy": None,
            "mean_wall_clock_s": None,
        },
        {
            "method": "LinearDesign_official",
            "status": "measured_external" if linear_measured else "pending_external_run",
            "n": linear.get("n_outputs"),
            "mean_cai": _num(linear.get("mean_cai")),
            "delta_cai_vs_native": (
                _num(linear.get("mean_cai")) - native_cai
                if _num(linear.get("mean_cai")) is not None and native_cai is not None
                else None
            ),
            "mean_gc": _num(linear.get("mean_gc")),
            "delta_gc_vs_native": (
                _num(linear.get("mean_gc")) - native_gc
                if _num(linear.get("mean_gc")) is not None and native_gc is not None
                else None
            ),
            "mean_gc3": _num(linear.get("mean_gc3")),
            "codon_accuracy_vs_native": _num(
                linear.get("mean_codon_accuracy_vs_native")
            ),
            "codon_usage_kl_vs_native": _num(
                linear.get("mean_codon_usage_kl_vs_native")
            ),
            "codon_pair_kl_vs_native": _num(
                linear.get("mean_codon_pair_kl_vs_native")
            ),
            "protein_identity_exact_1_fraction": _num(
                linear.get("protein_identity_exact_1_fraction")
            ),
            "mean_mfe": _num(linear.get("mean_mfe_without_stop")),
            "mean_ensemble_free_energy": None,
            "mean_wall_clock_s": _num(linear.get("mean_wall_clock_s")),
        },
        {
            "method": "EnsembleDesign_official",
            "status": (
                "measured_external_budgeted"
                if ensemble_measured
                else "executable_ready_metrics_pending"
            ),
            "n": ensemble.get("n_outputs"),
            "mean_cai": _num(ensemble.get("mean_cai")),
            "delta_cai_vs_native": (
                _num(ensemble.get("mean_cai")) - native_cai
                if _num(ensemble.get("mean_cai")) is not None
                and native_cai is not None
                else None
            ),
            "mean_gc": _num(ensemble.get("mean_gc")),
            "delta_gc_vs_native": (
                _num(ensemble.get("mean_gc")) - native_gc
                if _num(ensemble.get("mean_gc")) is not None
                and native_gc is not None
                else None
            ),
            "mean_gc3": _num(ensemble.get("mean_gc3")),
            "codon_accuracy_vs_native": _num(
                ensemble.get("mean_codon_accuracy_vs_native")
            ),
            "codon_usage_kl_vs_native": _num(
                ensemble.get("mean_codon_usage_kl_vs_native")
            ),
            "codon_pair_kl_vs_native": _num(
                ensemble.get("mean_codon_pair_kl_vs_native")
            ),
            "protein_identity_exact_1_fraction": _num(
                ensemble.get("protein_identity_exact_1_fraction")
            ),
            "mean_mfe": None,
            "mean_ensemble_free_energy": _num(
                ensemble.get("mean_ensemble_free_energy")
            ),
            "mean_wall_clock_s": _num(
                ensemble.get("mean_wall_clock_s")
            ),
        },
        {
            "method": "codonGPT_official_HF_pretrained",
            "status": (
                "measured_external_pretrained_checkpoint_10seed"
                if codongpt_measured and codongpt_10seed_complete
                else "measured_external_pretrained_checkpoint"
                if codongpt_measured
                else "official_checkpoint_metrics_pending"
            ),
            "n": codongpt.get("n_outputs"),
            "n_seeds": (
                int(codongpt_multiseed_status.get("n_complete_seeds", 0))
                if codongpt_10seed_complete
                else 1 if codongpt_measured else 0
            ),
            "mean_cai": codongpt_metric(
                "mean_cai",
                codongpt.get("mean_cai"),
            ),
            "delta_cai_vs_native": codongpt_delta(
                "delta_cai_vs_native",
                "mean_cai",
                codongpt.get("mean_cai"),
                native_cai,
            ),
            "mean_gc": codongpt_metric(
                "mean_gc",
                codongpt.get("mean_gc"),
            ),
            "delta_gc_vs_native": codongpt_delta(
                "delta_gc_vs_native",
                "mean_gc",
                codongpt.get("mean_gc"),
                native_gc,
            ),
            "mean_gc3": codongpt_metric(
                "mean_gc3",
                codongpt.get("mean_gc3"),
            ),
            "codon_accuracy_vs_native": codongpt_metric(
                "mean_codon_accuracy_vs_native",
                codongpt.get("mean_codon_accuracy_vs_native"),
            ),
            "codon_usage_kl_vs_native": codongpt_metric(
                "mean_codon_usage_kl_vs_native",
                codongpt.get("mean_codon_usage_kl_vs_native"),
            ),
            "codon_pair_kl_vs_native": codongpt_metric(
                "mean_codon_pair_kl_vs_native",
                codongpt.get("mean_codon_pair_kl_vs_native"),
            ),
            "protein_identity_exact_1_fraction": _num(
                codongpt.get("protein_identity_exact_1_fraction")
            ),
            "mean_mfe": None,
            "mean_ensemble_free_energy": None,
            "mean_wall_clock_s": codongpt_metric(
                "mean_wall_clock_s",
                codongpt.get("mean_wall_clock_s"),
            ),
            "delta_cai_seed_ci_low": _num(
                _mapping(
                    codongpt_multiseed_aggregate.get(
                        "delta_cai_vs_native"
                    )
                ).get("low")
            ),
            "delta_cai_seed_ci_high": _num(
                _mapping(
                    codongpt_multiseed_aggregate.get(
                        "delta_cai_vs_native"
                    )
                ).get("high")
            ),
            "delta_cai_seed_paired_p_vs_native": _num(
                codongpt_multiseed_status.get(
                    "delta_cai_vs_native_paired_signflip_p"
                )
            ),
        },
    ]
    ready = bool(
        linear_measured
        and linear.get("n_inputs") == 1024
        and linear.get("n_outputs") == 1024
        and linear.get("n_failures") == 0
        and linear.get("protein_identity_exact_1_fraction") == 1.0
        and linear.get("valid_cds_fraction") == 1.0
    )
    ensemble_complete = bool(
        ensemble_measured
        and ensemble.get("n_inputs") == 1024
        and ensemble.get("n_outputs") == 1024
        and ensemble.get("n_failures") == 0
        and ensemble.get("protein_identity_exact_1_fraction") == 1.0
        and ensemble.get("valid_cds_fraction") == 1.0
    )
    codongpt_complete = bool(
        codongpt_measured
        and codongpt.get("n_inputs") == 1024
        and codongpt.get("n_outputs") == 1024
        and codongpt.get("n_failures") == 0
        and codongpt.get("protein_identity_exact_1_fraction") == 1.0
        and codongpt.get("valid_cds_fraction") == 1.0
    )
    return {
        "artifact_kind": "t4_external_cds_baseline_comparison",
        "claim_policy": CLAIM_POLICY,
        "sources": paths,
        "summary": {
            "ready_for_t4_external_cds_descriptive_table": ready,
            "linear_design_measured": linear_measured,
            "linear_design_complete_head1024": ready,
            "ensemble_design_measured": ensemble_measured,
            "ensemble_design_complete_head1024": ensemble_complete,
            "ensemble_design_metrics_pending": not ensemble_complete,
            "codongpt_measured": codongpt_measured,
            "codongpt_complete_head1024": codongpt_complete,
            "codongpt_10seed_complete_head1024": (
                codongpt_10seed_complete
            ),
            "codongpt_seed_level_inference_ready": (
                codongpt_10seed_complete
            ),
            "codongpt_rl_policy_reproduced": False,
            "both_external_optimizers_complete_head1024": bool(
                ready and ensemble_complete
            ),
            "paired_per_record_test_ready": False,
            "common_structure_metric_ready": False,
            "ready_for_mef_superiority_claim": False,
        },
        "rows": rows,
    }


def write_report_json(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    return path


def _fmt(value: object) -> str:
    return "NA" if value is None else f"{float(value):.6f}"


def write_report_markdown(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    summary = _mapping(report.get("summary"))
    rows = report.get("rows", [])
    rows = rows if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) else []
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# T4 External CDS Baseline Comparison\n\n")
        fh.write(f"- Claim policy: {report.get('claim_policy', CLAIM_POLICY)}\n")
        fh.write(
            f"- Descriptive table ready: "
            f"`{summary.get('ready_for_t4_external_cds_descriptive_table')}`; "
            f"MEF superiority claim ready: "
            f"`{summary.get('ready_for_mef_superiority_claim')}`\n\n"
        )
        fh.write("| Method | Status | n | CAI | delta CAI | GC | GC3 | Codon accuracy | Codon KL | Pair KL | Protein exact-1 | MFE | EFE | sec/seq |\n")
        fh.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            fh.write(
                f"| {row.get('method')} | `{row.get('status')}` | "
                f"{row.get('n') if row.get('n') is not None else 'NA'} | "
                f"{_fmt(row.get('mean_cai'))} | {_fmt(row.get('delta_cai_vs_native'))} | "
                f"{_fmt(row.get('mean_gc'))} | {_fmt(row.get('mean_gc3'))} | "
                f"{_fmt(row.get('codon_accuracy_vs_native'))} | "
                f"{_fmt(row.get('codon_usage_kl_vs_native'))} | "
                f"{_fmt(row.get('codon_pair_kl_vs_native'))} | "
                f"{_fmt(row.get('protein_identity_exact_1_fraction'))} | "
                f"{_fmt(row.get('mean_mfe'))} | "
                f"{_fmt(row.get('mean_ensemble_free_energy'))} | "
                f"{_fmt(row.get('mean_wall_clock_s'))} |\n"
            )
        codongpt_row = next(
            (
                row
                for row in rows
                if isinstance(row, Mapping)
                and row.get("method")
                == "codonGPT_official_HF_pretrained"
            ),
            None,
        )
        if isinstance(codongpt_row, Mapping):
            fh.write("\n## codonGPT Seed Variability\n\n")
            fh.write(
                f"- Seeds: `{codongpt_row.get('n_seeds', 0)}`; CAI delta "
                f"95% CI: [{_fmt(codongpt_row.get('delta_cai_seed_ci_low'))}, "
                f"{_fmt(codongpt_row.get('delta_cai_seed_ci_high'))}]; "
                "sign-flip p vs native: "
                f"`{_fmt(codongpt_row.get('delta_cai_seed_paired_p_vs_native'))}`.\n"
            )
            fh.write(
                "- This quantifies the public pretrained checkpoint's sampling "
                "variability. It is not the unreleased RL policy result.\n"
            )
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--out-json", default="docs/t4_external_cds_baseline_comparison.json")
    parser.add_argument("--out-md", default="docs/t4_external_cds_baseline_comparison.md")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    root = os.path.abspath(args.project_root)
    out_json = args.out_json if os.path.isabs(args.out_json) else os.path.join(root, args.out_json)
    out_md = args.out_md if os.path.isabs(args.out_md) else os.path.join(root, args.out_md)
    report = build_t4_external_cds_comparison(root)
    write_report_json(report, out_json)
    write_report_markdown(report, out_md)
    print(json.dumps({"json_path": out_json, "markdown_path": out_md, "summary": report["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLAIM_POLICY",
    "build_t4_external_cds_comparison",
    "write_report_json",
    "write_report_markdown",
    "main",
]
