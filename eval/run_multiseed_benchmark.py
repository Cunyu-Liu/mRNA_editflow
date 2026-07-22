"""Multi-seed public benchmark runner for mRNA-EditFlow.

This script closes the paper-style loop:

``public records -> checkpoint-guided candidates -> T1-T7 evaluation -> seed aggregation``.

It intentionally reuses the canonical sampler and evaluator rather than
duplicating metrics. Each seed writes its own candidate JSONL and
``eval_summary.json``; the final artifact aggregates scalar metrics across
seeds with bootstrap confidence intervals.

Aggregation math
----------------
For a metric value ``m_s`` measured at seed ``s``, the reported point estimate is

``mean_s m_s``.

The 95% interval is a non-parametric bootstrap over seeds using
:func:`mrna_editflow.eval.run_eval.bootstrap_ci`. This is not a substitute for
per-record paired tests, but it is the correct first-order stability summary for
the multi-seed experimental protocol required by the project.

Complexity
----------
Let ``S`` be number of seeds, ``N`` records, ``E`` edit budget and ``L`` sequence
length. Generation costs ``O(S * N * E * model_forward(L))`` when a checkpoint is
used; evaluation costs the existing T1-T7 metric complexity, with pairwise
diversity bounded by ``--max-pairwise-pairs`` and optional approximate novelty
bounded by ``--max-novelty-sources``. With ``--resume``, completed seeds are
loaded in ``O(summary_file_size)`` each and only missing seeds pay generation
and evaluation cost.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.download_mrna import load_records_jsonl
from mrna_editflow.data.split_contract import (
    VerifiedSplitContract,
    load_and_verify_split_manifest,
    sha256_file,
)
from mrna_editflow.eval import metrics, run_eval
from mrna_editflow.eval.artifact_contract import (
    OracleContractError,
    build_run_metadata,
    load_and_verify_oracle_manifest,
    normalize_run_mode,
    prepare_scientific_records,
    provenance_digest,
    require_paper_cli_inputs,
    validate_output_namespace,
    verify_paper_checkpoint,
    write_provenance_sidecar,
)
from mrna_editflow.eval.oracle import LocalTranslationOracle
from mrna_editflow.sample import generate_candidate_records, write_candidates_jsonl


DEFAULT_MULTI_SEEDS = tuple(range(10))


@dataclass(frozen=True)
class MetricSpec:
    """A scalar metric to extract from a single-seed evaluation summary."""

    name: str
    path: tuple[str, ...]
    higher_is_better: bool
    description: str


METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec("legal_fraction", ("task_metrics", "T1", "legal_fraction"), True, "valid full mRNA fraction"),
    MetricSpec("mean_oracle_te", ("task_metrics", "T1", "mean_oracle_te"), True, "translation-efficiency proxy"),
    MetricSpec("delta_oracle_te_vs_source", tuple(), True, "candidate TE minus paired source TE"),
    MetricSpec("mean_oracle_mrl", ("task_metrics", "T1", "mean_oracle_mrl"), True, "mean ribosome-load proxy"),
    MetricSpec("kmer_js", ("task_metrics", "T2", "kmer_js"), False, "k-mer JS distance to sources"),
    MetricSpec("codon_usage_kl", ("task_metrics", "T2", "codon_usage_kl"), False, "codon-usage KL to sources"),
    MetricSpec("mean_novelty", ("task_metrics", "T3", "mean_novelty"), True, "mean normalized distance to nearest source"),
    MetricSpec("exact_source_match_fraction", ("task_metrics", "T3", "exact_source_match_fraction"), False, "exact training/source copies"),
    MetricSpec("mean_protein_identity", ("task_metrics", "T4", "mean_protein_identity"), True, "paired protein identity"),
    MetricSpec("within_budget_fraction", ("task_metrics", "T5", "within_budget_fraction"), True, "fraction within edit budget"),
    MetricSpec("mean_edit_distance", ("task_metrics", "T5", "mean_edit_distance"), False, "paired nucleotide edit distance"),
    MetricSpec("mean_abs_length_error", ("task_metrics", "T6", "mean_abs_length_error"), False, "absolute target-length error"),
    MetricSpec("reading_frame_intact_fraction", ("task_metrics", "T7", "reading_frame_intact_fraction"), True, "valid reading frame fraction"),
)


def _nested_get(payload: Mapping[str, object], path: Sequence[str], default: float = 0.0) -> float:
    cur: object = payload
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return float(default)
        cur = cur[key]
    try:
        value = float(cur)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _delta_oracle_te(summary: Mapping[str, object]) -> float:
    per = summary.get("per_record_metrics", {})
    if not isinstance(per, Mapping):
        return 0.0
    cand = per.get("oracle_ensemble_te", [])
    src = per.get("source_oracle_ensemble_te", [])
    if not isinstance(cand, Sequence) or not isinstance(src, Sequence) or not cand or not src:
        return 0.0
    n = min(len(cand), len(src))
    diffs = []
    for i in range(n):
        try:
            diffs.append(float(cand[i]) - float(src[i]))  # type: ignore[index]
        except (TypeError, ValueError):
            continue
    return float(np.mean(diffs)) if diffs else 0.0


def extract_scalar_metrics(summary: Mapping[str, object]) -> dict[str, float]:
    """Extract the scalar metrics used by the multi-seed paper table."""
    out: dict[str, float] = {}
    for spec in METRIC_SPECS:
        if spec.name == "delta_oracle_te_vs_source":
            out[spec.name] = _delta_oracle_te(summary)
        else:
            out[spec.name] = _nested_get(summary, spec.path)
    return out


def aggregate_seed_metrics(
    per_seed_metrics: Sequence[Mapping[str, float]],
    seeds: Sequence[int],
    n_bootstrap: int = 1000,
) -> dict[str, dict[str, float]]:
    """Aggregate scalar metrics across seeds with bootstrap CIs."""
    aggregate: dict[str, dict[str, float]] = {}
    for spec in METRIC_SPECS:
        values = [
            float(row[spec.name])
            for row in per_seed_metrics
            if spec.name in row and math.isfinite(float(row[spec.name]))
        ]
        if not values:
            aggregate[spec.name] = {
                "mean": 0.0,
                "std": 0.0,
                "low": 0.0,
                "high": 0.0,
                "n": 0,
                "higher_is_better": float(spec.higher_is_better),
            }
            continue
        ci = run_eval.bootstrap_ci(values, seeds=seeds, n_bootstrap=n_bootstrap)
        aggregate[spec.name] = {
            "mean": float(ci["mean"]),
            "std": float(np.std(np.asarray(values, dtype=float), ddof=1)) if len(values) > 1 else 0.0,
            "low": float(ci["low"]),
            "high": float(ci["high"]),
            "n": int(len(values)),
            "higher_is_better": float(spec.higher_is_better),
        }
    return aggregate


def _format_float(value: float) -> str:
    if not math.isfinite(float(value)):
        return ""
    if abs(value) >= 100:
        return f"{value:.2f}"
    if abs(value) >= 1:
        return f"{value:.4f}"
    return f"{value:.5f}"


def write_multiseed_table(
    aggregate: Mapping[str, Mapping[str, float]],
    path: str,
) -> str:
    """Write a compact Markdown table for paper/README inspection."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    desc = {spec.name: spec for spec in METRIC_SPECS}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# mRNA-EditFlow Multi-Seed Benchmark\n\n")
        fh.write(
            "> Decoder-seed intervals are development-only and are not "
            "independent-training confidence intervals.\n\n"
        )
        fh.write("| Metric | Direction | Mean | Std | 95% CI | Description |\n")
        fh.write("|---|---:|---:|---:|---:|---|\n")
        for spec in METRIC_SPECS:
            row = aggregate.get(spec.name, {})
            direction = "higher" if bool(row.get("higher_is_better", spec.higher_is_better)) else "lower"
            ci = f"[{_format_float(float(row.get('low', 0.0)))}, {_format_float(float(row.get('high', 0.0)))}]"
            fh.write(
                f"| `{spec.name}` | {direction} | "
                f"{_format_float(float(row.get('mean', 0.0)))} | "
                f"{_format_float(float(row.get('std', 0.0)))} | {ci} | "
                f"{desc[spec.name].description} |\n"
            )
    return path


def _append_progress(progress_path: str, payload: Mapping[str, object]) -> None:
    """Append one JSONL progress event for long-running benchmark recovery.

    The file is append-only so interrupted jobs keep the last completed event.
    Complexity is ``O(size(payload))`` per event.
    """
    os.makedirs(os.path.dirname(os.path.abspath(progress_path)), exist_ok=True)
    row = {"time": time.time(), **dict(payload)}
    with open(progress_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def _load_completed_seed(
    seed_dir: str,
    expected_resume_fingerprint: Optional[str] = None,
) -> Optional[dict[str, object]]:
    """Load a completed seed if candidate and eval artifacts are present.

    A seed is considered resumable only when both ``candidates.jsonl`` and
    ``eval_summary.json`` exist and the summary can be parsed. This avoids
    accepting half-written generation output as a completed evaluation. The
    extraction cost is dominated by reading the summary JSON, ``O(file_size)``.
    """
    cand_path = os.path.join(seed_dir, "candidates.jsonl")
    eval_path = os.path.join(seed_dir, "eval_summary.json")
    paper_path = os.path.join(seed_dir, "paper_table.md")
    if not (os.path.exists(cand_path) and os.path.exists(eval_path)):
        return None
    with open(eval_path, "r", encoding="utf-8") as fh:
        summary = json.load(fh)
    if not isinstance(summary, Mapping):
        return None
    metadata = summary.get("scientific_validity", {})
    actual_fingerprint = (
        metadata.get("resume_fingerprint") if isinstance(metadata, Mapping) else None
    )
    if expected_resume_fingerprint is not None and actual_fingerprint != expected_resume_fingerprint:
        raise ValueError(
            "refusing stale resume artifact: checkpoint/config/split/oracle fingerprint changed"
        )
    scalars = extract_scalar_metrics(summary)
    return {
        "candidate_path": cand_path,
        "eval_json_path": eval_path,
        "paper_table_path": paper_path,
        "metrics": scalars,
    }


def run_multiseed_benchmark(
    records: Sequence[MRNARecord],
    *,
    checkpoint_path: Optional[str],
    cascade_recall_checkpoint_path: Optional[str] = None,
    out_dir: str,
    task_id: str = "T5",
    seeds: Sequence[int] = DEFAULT_MULTI_SEEDS,
    limit: Optional[int] = None,
    edit_budget: int = 3,
    target_length_delta: int = 0,
    device: Optional[str] = None,
    proposal_top_k: int = 8,
    cascade_recall_top_k: int = 64,
    proposal_temperature: float = 1.0,
    guidance_scale: float = 0.0,
    target_te: Optional[float] = None,
    target_start_accessibility: Optional[float] = None,
    editable_regions: Optional[Sequence[str]] = None,
    n_bootstrap: int = 1000,
    max_pairwise_pairs: int = metrics.DEFAULT_MAX_PAIRWISE_DIVERSITY_PAIRS,
    max_novelty_sources: int = 0,
    resume: bool = False,
    progress_jsonl: Optional[str] = None,
    run_mode: str = "development",
    split_contract: Optional[VerifiedSplitContract] = None,
    split_role: Optional[str] = None,
    training_seed: Optional[int] = None,
    oracle_manifest: Optional[str] = None,
    oracle: Optional[object] = None,
) -> dict[str, object]:
    """Run checkpoint-guided generation and T1-T7 eval for every seed."""
    run_mode = normalize_run_mode(run_mode)
    seed_list = [int(x) for x in seeds]
    if len(seed_list) < 5:
        raise ValueError("multi-seed benchmark requires at least 5 seeds")
    validate_output_namespace(out_dir, run_mode)
    role_records, data_provenance = prepare_scientific_records(
        records,
        run_mode=run_mode,
        split_contract=split_contract,
        split_role=split_role,
        allowed_roles=("test",),
    )
    oracle_metadata = load_and_verify_oracle_manifest(oracle_manifest, run_mode=run_mode)
    checkpoint_metadata = None
    if run_mode == "paper":
        if checkpoint_path is None:
            raise ValueError("paper multiseed benchmark requires a checkpoint")
        checkpoint_metadata = verify_paper_checkpoint(checkpoint_path, data_provenance)
        if oracle is None or isinstance(oracle, LocalTranslationOracle):
            raise OracleContractError(
                "paper multiseed benchmark requires an independent oracle implementation"
            )
    if training_seed is None and isinstance(checkpoint_metadata, Mapping):
        raw_training_seed = checkpoint_metadata.get("training_seed")
        training_seed = int(raw_training_seed) if raw_training_seed is not None else None
    selected = list(role_records[: int(limit)]) if limit is not None else list(role_records)
    checkpoint_id = sha256_file(checkpoint_path) if checkpoint_path else None
    config_payload = {
        "task_id": task_id,
        "checkpoint_id": checkpoint_id,
        "cascade_recall_checkpoint_id": (
            sha256_file(cascade_recall_checkpoint_path)
            if cascade_recall_checkpoint_path else None
        ),
        "decoder_seeds": seed_list,
        "limit": limit,
        "edit_budget": edit_budget,
        "target_length_delta": target_length_delta,
        "proposal_top_k": proposal_top_k,
        "cascade_recall_top_k": cascade_recall_top_k,
        "proposal_temperature": proposal_temperature,
        "guidance_scale": guidance_scale,
        "target_te": target_te,
        "target_start_accessibility": target_start_accessibility,
        "editable_regions": list(editable_regions) if editable_regions is not None else None,
    }
    scientific_validity = build_run_metadata(
        run_mode=run_mode,
        data_provenance=data_provenance,
        config=config_payload,
        code_paths=(__file__, run_eval.__file__),
        training_seed=training_seed,
        oracle=oracle_metadata,
        upstream={
            "checkpoint_sha256": checkpoint_id,
            "cascade_recall_checkpoint_sha256": config_payload["cascade_recall_checkpoint_id"],
        },
        extra_block_reasons=(
            "decoder_seed_bootstrap_not_independent_training_replicates",
        ) if run_mode == "development" else (),
        functional_claim=True,
    )
    resume_fingerprint = provenance_digest(
        {"scientific_validity": scientific_validity, "config": config_payload}
    )
    scientific_validity["resume_fingerprint"] = resume_fingerprint
    os.makedirs(out_dir, exist_ok=True)
    progress_path = progress_jsonl or os.path.join(out_dir, "multiseed_progress.jsonl")
    per_seed_rows = []
    per_seed_metrics = []
    source_path = os.path.join(out_dir, "sources.jsonl")
    from mrna_editflow.data.download_mrna import write_records_jsonl

    write_records_jsonl(selected, source_path)
    write_provenance_sidecar(source_path, scientific_validity)
    record_units = [
        {
            "transcript_id": record.transcript_id,
            "family_id": (
                split_contract.cluster_assignments[
                    split_contract.roles[str(split_role)].indices[index]
                ]
                if split_contract is not None
                and split_role in ("train", "val", "test")
                and split_contract.cluster_assignments
                else None
            ),
            "cluster_id": (
                split_contract.cluster_assignments[
                    split_contract.roles[str(split_role)].indices[index]
                ]
                if split_contract is not None
                and split_role in ("train", "val", "test")
                and split_contract.cluster_assignments
                else None
            ),
        }
        for index, record in enumerate(selected)
    ]

    for seed in seed_list:
        seed_dir = os.path.join(out_dir, f"seed_{seed:03d}")
        os.makedirs(seed_dir, exist_ok=True)
        if resume:
            completed = _load_completed_seed(seed_dir, resume_fingerprint)
            if completed is not None:
                scalars = completed["metrics"]
                if isinstance(scalars, Mapping):
                    per_seed_metrics.append({str(k): float(v) for k, v in scalars.items()})
                per_seed_rows.append({
                    "decoder_seed": seed,
                    "training_seed": training_seed,
                    "checkpoint_id": checkpoint_id,
                    "resumed": True,
                    "record_units": record_units,
                    **completed,
                })
                _append_progress(
                    progress_path,
                    {
                        "event": "seed_resumed",
                        "decoder_seed": seed,
                        "seed_dir": seed_dir,
                        "eval_json_path": completed["eval_json_path"],
                    },
                )
                continue
        _append_progress(
            progress_path,
            {"event": "seed_start", "decoder_seed": seed, "seed_dir": seed_dir},
        )
        candidates = generate_candidate_records(
            selected,
            task_id=task_id,
            checkpoint_path=checkpoint_path,
            cascade_recall_checkpoint_path=cascade_recall_checkpoint_path,
            limit=len(selected),
            edit_budget=edit_budget,
            target_length_delta=target_length_delta,
            seed=seed,
            device=device,
            proposal_top_k=proposal_top_k,
            cascade_recall_top_k=cascade_recall_top_k,
            proposal_temperature=proposal_temperature,
            guidance_scale=guidance_scale,
            target_te=target_te,
            target_start_accessibility=target_start_accessibility,
            editable_regions=editable_regions,
        )
        cand_path = os.path.join(seed_dir, "candidates.jsonl")
        write_candidates_jsonl(candidates, cand_path)
        write_provenance_sidecar(cand_path, scientific_validity)
        _append_progress(
            progress_path,
            {
                "event": "seed_candidates_written",
                "decoder_seed": seed,
                "candidate_path": cand_path,
                "n_candidates": len(candidates),
            },
        )
        target_lengths = (
            [len(r.seq) + int(target_length_delta) for r in selected]
            if task_id.upper() == "T6"
            else None
        )
        summary = run_eval.run_evaluation(
            candidates,
            sources=selected,
            task_id="all",
            out_dir=seed_dir,
            seeds=seed_list,
            n_bootstrap=n_bootstrap,
            max_edits=edit_budget,
            target_lengths=target_lengths,
            max_pairwise_pairs=max_pairwise_pairs,
            max_novelty_sources=max_novelty_sources,
            oracle=oracle,
            run_mode=run_mode,
            checkpoint_path=checkpoint_path,
            oracle_manifest=oracle_manifest,
            training_seed=training_seed,
            decoder_seed=seed,
            verified_data_provenance=data_provenance,
            split_contract=split_contract,
            split_role=split_role,
            resume_fingerprint=resume_fingerprint,
        )
        scalars = extract_scalar_metrics(summary)
        per_seed_metrics.append(scalars)
        _append_progress(
            progress_path,
            {
                "event": "seed_evaluated",
                "decoder_seed": seed,
                "eval_json_path": summary["json_path"],
                "delta_oracle_te_vs_source": scalars.get("delta_oracle_te_vs_source", 0.0),
            },
        )
        per_seed_rows.append(
            {
                "decoder_seed": seed,
                "training_seed": training_seed,
                "checkpoint_id": checkpoint_id,
                "resumed": False,
                "candidate_path": cand_path,
                "eval_json_path": summary["json_path"],
                "paper_table_path": summary["paper_table_path"],
                "metrics": scalars,
                "record_units": record_units,
            }
        )

    aggregate = aggregate_seed_metrics(per_seed_metrics, seed_list, n_bootstrap=n_bootstrap)
    table_path = write_multiseed_table(aggregate, os.path.join(out_dir, "multiseed_table.md"))
    decoder_family = (
        "cascade"
        if cascade_recall_checkpoint_path
        else ("checkpoint_guided" if checkpoint_path else "deterministic")
    )
    effective_proposal_top_k = (
        int(cascade_recall_top_k)
        if cascade_recall_checkpoint_path
        else int(proposal_top_k)
    )
    result = {
        "config": {
            "run_mode": run_mode,
            "task_id": task_id,
            "checkpoint_path": checkpoint_path,
            "cascade_recall_checkpoint_path": cascade_recall_checkpoint_path,
            "decoder_family": decoder_family,
            "n_records": len(selected),
            "decoder_seeds": seed_list,
            "training_seed": training_seed,
            "checkpoint_id": checkpoint_id,
            "edit_budget": int(edit_budget),
            "target_length_delta": int(target_length_delta),
            "proposal_top_k": int(proposal_top_k),
            "cascade_recall_top_k": int(cascade_recall_top_k),
            "effective_proposal_top_k": effective_proposal_top_k,
            "proposal_temperature": float(proposal_temperature),
            "guidance_scale": float(guidance_scale),
            "target_te": target_te,
            "target_start_accessibility": target_start_accessibility,
            "editable_regions": (
                [str(region) for region in editable_regions]
                if editable_regions is not None
                else ["utr5", "utr3"]
            ),
            "n_bootstrap": int(n_bootstrap),
            "max_pairwise_pairs": int(max_pairwise_pairs),
            "max_novelty_sources": int(max_novelty_sources),
            "resume": bool(resume),
            "progress_jsonl": progress_path,
        },
        "source_path": source_path,
        "progress_jsonl": progress_path,
        "per_seed": per_seed_rows,
        "aggregate": aggregate,
        "table_path": table_path,
        "seed_semantics": {
            "decoder_seed_count": len(set(seed_list)),
            "training_seed_count": 1 if training_seed is not None else 0,
            "decoder_seed_is_independent_training_run": False,
            "paper_statistical_claim_eligible": False,
            "minimum_training_seeds_for_paper_claim": 3,
            "note": (
                "Bootstrap intervals over decoder seeds are development-only and "
                "are not independent-training confidence intervals."
            ),
        },
        "scientific_validity": scientific_validity,
    }
    json_path = os.path.join(out_dir, "multiseed_summary.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    _append_progress(
        progress_path,
        {
            "event": "benchmark_complete",
            "json_path": json_path,
            "table_path": table_path,
            "resume_fingerprint": resume_fingerprint,
        },
    )
    write_provenance_sidecar(json_path, scientific_validity)
    write_provenance_sidecar(table_path, scientific_validity)
    write_provenance_sidecar(progress_path, scientific_validity)
    result["json_path"] = json_path
    return result


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cascade-recall-checkpoint", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--task-id", default="T5")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_MULTI_SEEDS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--edit-budget", type=int, default=3)
    parser.add_argument("--target-length-delta", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--proposal-top-k", type=int, default=8, help="Top-k legal proposals; <=0 evaluates all")
    parser.add_argument("--cascade-recall-top-k", type=int, default=64, help="Recall-stage top-k for cascade decoding")
    parser.add_argument("--proposal-temperature", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--target-te", type=float, default=None)
    parser.add_argument("--target-start-accessibility", type=float, default=None)
    parser.add_argument(
        "--editable-regions",
        nargs="+",
        choices=("utr5", "utr3"),
        default=None,
        help="Restrict checkpoint-guided UTR edits; default enables utr5 and utr3",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument(
        "--max-pairwise-pairs",
        type=int,
        default=metrics.DEFAULT_MAX_PAIRWISE_DIVERSITY_PAIRS,
    )
    parser.add_argument(
        "--max-novelty-sources",
        type=int,
        default=0,
        help="Per-candidate source cap for approximate novelty; <=0 keeps exact novelty",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed seed_NNN/eval_summary.json artifacts instead of rerunning those seeds",
    )
    parser.add_argument(
        "--progress-jsonl",
        default=None,
        help="Append-only progress JSONL path; defaults to OUT_DIR/multiseed_progress.jsonl",
    )
    parser.add_argument("--run-mode", choices=("development", "paper"), default="development")
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--split-role", choices=("train", "val", "test"), default=None)
    parser.add_argument("--train-idx", default=None, help="path to train.idx (paper mode: required if --split-manifest absent)")
    parser.add_argument("--val-idx", default=None, help="path to val.idx (paper mode: required if --split-manifest absent)")
    parser.add_argument("--test-idx", default=None, help="path to test.idx (paper mode: required if --split-manifest absent)")
    parser.add_argument("--training-seed", type=int, default=None)
    parser.add_argument("--oracle-manifest", default=None)
    return parser.parse_args(argv)


def _enforce_exact_match_fail_closed(
    args: argparse.Namespace,
    split_contract: Optional["VerifiedSplitContract"],
    records: Sequence[MRNARecord],
) -> None:
    """P1-10: Exact-match fail-closed enforcement.

    Verifies that the records being evaluated exactly match the split contract's
    records content digest. If there is any mismatch, the script FAILS (does not
    proceed). This prevents accidental evaluation on wrong/leaked data.
    """
    if split_contract is None:
        return  # No contract to verify against
    # Compute records content digest
    from mrna_editflow.data.split_contract import records_content_digest
    computed_digest = records_content_digest(records)
    # Get the contract's records digest
    contract_digest = getattr(split_contract, "records_content_digest", None)
    if contract_digest is None:
        # Try to get from the manifest's records_sha256
        contract_digest = getattr(split_contract, "records_sha256", None)
    if contract_digest is None:
        # No digest available to verify against; warn but don't fail
        print("WARNING: split contract has no records_content_digest; cannot enforce exact-match fail-closed")
        return
    if computed_digest != contract_digest:
        raise SystemExit(
            f"EXACT-MATCH FAIL-CLOSED: records content digest mismatch!\n"
            f"  computed: {computed_digest}\n"
            f"  contract: {contract_digest}\n"
            f"  The records being evaluated do not match the split contract.\n"
            f"  Aborting to prevent evaluation on wrong/leaked data."
        )
    # Verify idx files if provided
    idx_paths = {"train": args.train_idx, "val": args.val_idx, "test": args.test_idx}
    for role, path_str in idx_paths.items():
        if path_str is None:
            continue
        from pathlib import Path
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"--{role}-idx file not found: {path}")
        with path.open("r") as fh:
            indices = [int(line.strip()) for line in fh if line.strip()]
        contract_indices = split_contract.roles[role].indices
        if len(indices) != len(contract_indices):
            raise SystemExit(
                f"EXACT-MATCH FAIL-CLOSED: --{role}-idx has {len(indices)} indices "
                f"but split contract has {len(contract_indices)} for role '{role}'"
            )
        if set(indices) != set(contract_indices):
            raise SystemExit(
                f"EXACT-MATCH FAIL-CLOSED: --{role}-idx indices do not match "
                f"split contract for role '{role}'"
            )
    print(f"OK: exact-match fail-closed passed (records digest: {computed_digest[:16]}...)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    require_paper_cli_inputs(
        run_mode=args.run_mode,
        split_manifest=args.split_manifest,
        split_role=args.split_role,
        allowed_roles=("test",),
        oracle_manifest=args.oracle_manifest,
        require_oracle=True,
    )
    # P1-10: In paper mode, require either --split-manifest OR all three idx files.
    if args.run_mode == "paper":
        idx_provided = all([args.train_idx, args.val_idx, args.test_idx])
        if not args.split_manifest and not idx_provided:
            raise SystemExit(
                "paper mode requires either --split-manifest OR "
                "(--train-idx AND --val-idx AND --test-idx); aborting."
            )
    split_contract = (
        load_and_verify_split_manifest(args.split_manifest, records_path=args.records_jsonl)
        if args.split_manifest else None
    )
    # P1-10: Exact-match fail-closed verification
    records = load_records_jsonl(args.records_jsonl)
    _enforce_exact_match_fail_closed(args, split_contract, records)
    result = run_multiseed_benchmark(
        load_records_jsonl(args.records_jsonl),
        checkpoint_path=args.checkpoint,
        cascade_recall_checkpoint_path=args.cascade_recall_checkpoint,
        out_dir=args.out_dir,
        task_id=args.task_id,
        seeds=args.seeds,
        limit=args.limit,
        edit_budget=args.edit_budget,
        target_length_delta=args.target_length_delta,
        device=args.device,
        proposal_top_k=args.proposal_top_k,
        cascade_recall_top_k=args.cascade_recall_top_k,
        proposal_temperature=args.proposal_temperature,
        guidance_scale=args.guidance_scale,
        target_te=args.target_te,
        target_start_accessibility=args.target_start_accessibility,
        editable_regions=args.editable_regions,
        n_bootstrap=args.n_bootstrap,
        max_pairwise_pairs=args.max_pairwise_pairs,
        max_novelty_sources=args.max_novelty_sources,
        resume=args.resume,
        progress_jsonl=args.progress_jsonl,
        run_mode=args.run_mode,
        split_contract=split_contract,
        split_role=args.split_role,
        training_seed=args.training_seed,
        oracle_manifest=args.oracle_manifest,
    )
    print(json.dumps({"json_path": result["json_path"], "table_path": result["table_path"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MULTI_SEEDS",
    "METRIC_SPECS",
    "extract_scalar_metrics",
    "aggregate_seed_metrics",
    "write_multiseed_table",
    "run_multiseed_benchmark",
    "main",
]
