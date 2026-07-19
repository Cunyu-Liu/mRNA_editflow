"""Aggregate ten codonGPT constrained-generation seeds on the T4 head1024 set."""
from __future__ import annotations

import argparse
import json
import os
from typing import Mapping, Optional, Sequence

from mrna_editflow.baselines.external_lineardesign_adapter import (
    _load_jsonl,
    _normalise_rna,
    _resolve_pack_file,
)
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.prepare_codon import gc3_fraction
from mrna_editflow.eval.metrics import (
    cai,
    codon_weights_from_reference,
    gc_fraction,
)
from mrna_editflow.eval.run_eval import (
    bootstrap_ci,
    paired_permutation_pvalue,
)


CLAIM_POLICY = (
    "This report quantifies seed variability for the official public codonGPT "
    "pretrained checkpoint under synonymous-mask sampling. It does not "
    "reproduce the paper's unreleased RL policies, establish expression gain, "
    "or support MEF superiority."
)
METRICS = (
    "mean_cai",
    "mean_gc",
    "mean_gc3",
    "mean_codon_accuracy_vs_native",
    "mean_codon_usage_kl_vs_native",
    "mean_codon_pair_kl_vs_native",
    "mean_wall_clock_s",
)


def _load_json(path: str) -> Optional[Mapping[str, object]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload if isinstance(payload, Mapping) else None


def _write_json(payload: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return path


def _seed_dir(
    project_root: str,
    multiseed_root: str,
    seed: int,
) -> str:
    if seed == 0:
        return os.path.join(
            project_root,
            "benchmark/external_sota/real_runs_t5_head1024/codonGPT",
        )
    return os.path.join(
        project_root,
        multiseed_root,
        f"seed_{seed:03d}",
    )


def build_codongpt_multiseed_summary(
    project_root: str,
    *,
    multiseed_root: str = (
        "benchmark/external_sota/codongpt_multiseed_head1024"
    ),
    expected_seeds: Sequence[int] = tuple(range(10)),
) -> dict[str, object]:
    root = os.path.abspath(project_root)
    input_pack_path = os.path.join(
        root,
        "benchmark/external_sota/input_pack_t5_head1024/summary.json",
    )
    input_pack = _load_json(input_pack_path)
    if input_pack is None:
        raise FileNotFoundError(input_pack_path)
    cds_inputs_path = _resolve_pack_file(
        input_pack_path,
        input_pack,
        "cds_protein_jsonl",
        "cds_protein_inputs.jsonl",
    )
    input_rows = _load_jsonl(cds_inputs_path)
    reference = [
        MRNARecord(
            str(row.get("transcript_id")),
            "",
            _normalise_rna(row.get("native_cds")),
            "",
        )
        for row in input_rows
    ]
    weights = codon_weights_from_reference(reference)
    native_cai = float(
        sum(cai(record.cds, weights) for record in reference)
        / max(len(reference), 1)
    )
    native_gc = float(
        sum(gc_fraction(record.cds) for record in reference)
        / max(len(reference), 1)
    )
    native_gc3 = float(
        sum(gc3_fraction(record.cds) for record in reference)
        / max(len(reference), 1)
    )

    seed_rows: list[dict[str, object]] = []
    failures: list[str] = []
    expected_ids = [str(row.get("transcript_id")) for row in input_rows]
    for seed in expected_seeds:
        run_dir = _seed_dir(root, multiseed_root, int(seed))
        summary = _load_json(os.path.join(run_dir, "summary.json"))
        outputs = _load_jsonl(os.path.join(run_dir, "cds_outputs.jsonl"))
        if summary is None:
            failures.append(f"seed_{seed}:summary_missing")
            continue
        observed_ids = [str(row.get("transcript_id")) for row in outputs]
        reasons: list[str] = []
        if summary.get("n_inputs") != len(input_rows):
            reasons.append("n_inputs_mismatch")
        if summary.get("n_outputs") != len(input_rows):
            reasons.append("n_outputs_mismatch")
        if summary.get("n_failures") != 0:
            reasons.append("nonzero_failures")
        if observed_ids != expected_ids:
            reasons.append("ordered_coverage_mismatch")
        if summary.get("valid_cds_fraction") != 1.0:
            reasons.append("valid_cds_not_exact_1")
        if summary.get("protein_identity_exact_1_fraction") != 1.0:
            reasons.append("protein_identity_not_exact_1")
        observed_seed = (
            summary.get("runtime", {}).get("seed")
            if isinstance(summary.get("runtime"), Mapping)
            else None
        )
        if observed_seed != int(seed):
            reasons.append("seed_metadata_mismatch")
        if reasons:
            failures.extend(f"seed_{seed}:{reason}" for reason in reasons)
            continue
        row: dict[str, object] = {
            "seed": int(seed),
            "run_dir": os.path.relpath(run_dir, root),
            "n_outputs": len(outputs),
            "valid_cds_fraction": 1.0,
            "protein_identity_exact_1_fraction": 1.0,
        }
        for metric in METRICS:
            value = summary.get(metric)
            row[metric] = float(value) if isinstance(value, (int, float)) else None
        row["delta_cai_vs_native"] = float(
            row["mean_cai"] - native_cai  # type: ignore[operator]
        )
        row["delta_gc_vs_native"] = float(
            row["mean_gc"] - native_gc  # type: ignore[operator]
        )
        row["delta_gc3_vs_native"] = float(
            row["mean_gc3"] - native_gc3  # type: ignore[operator]
        )
        seed_rows.append(row)

    seeds = [int(row["seed"]) for row in seed_rows]
    aggregate: dict[str, object] = {}
    for metric in (
        *METRICS,
        "delta_cai_vs_native",
        "delta_gc_vs_native",
        "delta_gc3_vs_native",
    ):
        values = [
            float(row[metric])
            for row in seed_rows
            if isinstance(row.get(metric), (int, float))
        ]
        aggregate[metric] = (
            bootstrap_ci(values, seeds=seeds, n_bootstrap=2000)
            if values
            else {"n": 0, "mean": None, "low": None, "high": None}
        )
    cai_deltas = [
        float(row["delta_cai_vs_native"])
        for row in seed_rows
    ]
    cai_p = (
        paired_permutation_pvalue(
            cai_deltas,
            [0.0] * len(cai_deltas),
            seed=0,
            n_permutations=2000,
        )
        if len(cai_deltas) == len(expected_seeds)
        else None
    )
    complete = bool(
        len(seed_rows) == len(expected_seeds)
        and seeds == [int(seed) for seed in expected_seeds]
        and not failures
    )
    return {
        "artifact_kind": "codongpt_multiseed_head1024_summary",
        "claim_policy": CLAIM_POLICY,
        "protocol": (
            "official_hf_pretrained_checkpoint_synonymous_masked_sampling"
        ),
        "protocol_fidelity_sufficient_for_paper_rl_reproduction": False,
        "input_pack": os.path.relpath(input_pack_path, root),
        "multiseed_root": multiseed_root,
        "native_reference": {
            "n": len(reference),
            "mean_cai": native_cai,
            "mean_gc": native_gc,
            "mean_gc3": native_gc3,
        },
        "summary": {
            "complete_10seed_head1024": complete,
            "n_expected_seeds": len(expected_seeds),
            "n_complete_seeds": len(seed_rows),
            "hard_constraints_exact_1": bool(
                complete
                and all(
                    row["valid_cds_fraction"] == 1.0
                    and row["protein_identity_exact_1_fraction"] == 1.0
                    for row in seed_rows
                )
            ),
            "delta_cai_vs_native_paired_signflip_p": cai_p,
            "ready_for_pretrained_checkpoint_seed_variability_claim": complete,
            "ready_for_paper_rl_claim": False,
            "ready_for_expression_claim": False,
            "ready_for_mef_superiority_claim": False,
            "failures": failures,
        },
        "aggregate": aggregate,
        "per_seed": seed_rows,
    }


def write_markdown(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    summary = report.get("summary", {})
    summary = summary if isinstance(summary, Mapping) else {}
    aggregate = report.get("aggregate", {})
    aggregate = aggregate if isinstance(aggregate, Mapping) else {}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# codonGPT 10-Seed Head1024 Summary\n\n")
        fh.write(f"- Claim policy: {report.get('claim_policy')}\n")
        fh.write(
            "- Complete: "
            f"`{summary.get('complete_10seed_head1024')}`; hard constraints "
            f"exact-1: `{summary.get('hard_constraints_exact_1')}`; paper RL "
            f"claim: `{summary.get('ready_for_paper_rl_claim')}`\n"
        )
        fh.write(
            "- CAI delta vs native sign-flip p: "
            f"`{summary.get('delta_cai_vs_native_paired_signflip_p')}`\n\n"
        )
        fh.write("| Metric | n | mean | 95% CI |\n")
        fh.write("|---|---:|---:|---|\n")
        for metric, stats in aggregate.items():
            if not isinstance(stats, Mapping):
                continue
            fh.write(
                f"| `{metric}` | {stats.get('n')} | "
                f"{stats.get('mean')} | "
                f"[{stats.get('low')}, {stats.get('high')}] |\n"
            )
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument(
        "--multiseed-root",
        default="benchmark/external_sota/codongpt_multiseed_head1024",
    )
    parser.add_argument(
        "--out-json",
        default=(
            "benchmark/external_sota/"
            "codongpt_multiseed_head1024/summary.json"
        ),
    )
    parser.add_argument(
        "--out-md",
        default=(
            "benchmark/external_sota/"
            "codongpt_multiseed_head1024/summary.md"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    root = os.path.abspath(args.project_root)
    report = build_codongpt_multiseed_summary(
        root,
        multiseed_root=args.multiseed_root,
    )
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
    _write_json(report, out_json)
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
    "build_codongpt_multiseed_summary",
    "write_markdown",
    "main",
]
