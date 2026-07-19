"""T1-T7 offline benchmark entry point for mRNA-EditFlow."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from mrna_editflow.core.constants import is_valid_cds
from mrna_editflow.eval import metrics
from mrna_editflow.eval.oracle import LocalTranslationOracle
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.split_contract import (
    VerifiedSplitContract,
    build_split_provenance,
    load_and_verify_split_manifest,
    sha256_file,
)
from mrna_editflow.eval.artifact_contract import (
    OracleContractError,
    build_run_metadata,
    load_and_verify_oracle_manifest,
    normalize_run_mode,
    prepare_scientific_records,
    records_identity,
    require_paper_cli_inputs,
    validate_output_namespace,
    verify_provenance_compatibility,
    verify_paper_checkpoint,
    write_provenance_sidecar,
)

DEFAULT_SEEDS = [0, 1, 2, 3, 4]

TASK_LABELS = {
    "T1": "Validity and translation oracle",
    "T2": "Distribution preservation",
    "T3": "Diversity and novelty",
    "T4": "Protein identity",
    "T5": "Edit budget",
    "T6": "Length control",
    "T7": "Motif and frame integrity",
}


def _ensure_seeds(seeds: Optional[Sequence[int]]) -> list[int]:
    out = list(DEFAULT_SEEDS if seeds is None else seeds)
    if len(out) < 5:
        raise ValueError("bootstrap requires at least 5 seeds")
    return [int(x) for x in out]


def bootstrap_ci(
    values: Sequence[float],
    seeds: Optional[Sequence[int]] = None,
    n_bootstrap: int = 250,
    confidence: float = 0.95,
) -> dict:
    """Seeded non-parametric bootstrap CI for the sample mean."""
    seed_list = _ensure_seeds(seeds)
    arr = np.asarray([float(x) for x in values if math.isfinite(float(x))], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "n": 0}
    if arr.size == 1:
        value = float(arr[0])
        return {"mean": value, "low": value, "high": value, "n": 1}
    reps_per_seed = max(1, int(math.ceil(n_bootstrap / len(seed_list))))
    means = []
    for seed in seed_list:
        rng = np.random.default_rng(int(seed))
        for _ in range(reps_per_seed):
            idx = rng.integers(0, arr.size, size=arr.size)
            means.append(float(np.mean(arr[idx])))
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(np.mean(arr)),
        "low": float(np.quantile(means, alpha)),
        "high": float(np.quantile(means, 1.0 - alpha)),
        "n": int(arr.size),
    }


def paired_permutation_pvalue(
    candidate_values: Sequence[float],
    source_values: Sequence[float],
    seed: int = 0,
    n_permutations: int = 2000,
) -> float:
    """Two-sided paired sign-flip permutation test on mean differences."""
    cand = np.asarray(candidate_values, dtype=float)
    src = np.asarray(source_values, dtype=float)
    n = min(cand.size, src.size)
    if n == 0:
        return 1.0
    diff = cand[:n] - src[:n]
    diff = diff[np.isfinite(diff)]
    if diff.size == 0 or np.allclose(diff, 0.0):
        return 1.0
    obs = abs(float(np.mean(diff)))
    rng = np.random.default_rng(int(seed))
    extreme = 0
    for _ in range(int(n_permutations)):
        signs = rng.choice(np.array([-1.0, 1.0]), size=diff.size)
        stat = abs(float(np.mean(diff * signs)))
        if stat >= obs - 1e-15:
            extreme += 1
    return float((extreme + 1) / (int(n_permutations) + 1))


def load_records(path: str) -> list[dict]:
    """Load records from JSON, JSONL, CSV or TSV."""
    suffix = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as fh:
        if suffix == ".jsonl":
            return [json.loads(line) for line in fh if line.strip()]
        if suffix == ".json":
            payload = json.load(fh)
            if isinstance(payload, Mapping) and "records" in payload:
                payload = payload["records"]
            if not isinstance(payload, list):
                raise ValueError("JSON input must be a list or {'records': list}")
            return [dict(x) for x in payload]
        if suffix in (".csv", ".tsv"):
            dialect = "excel-tab" if suffix == ".tsv" else "excel"
            return [dict(row) for row in csv.DictReader(fh, dialect=dialect)]
    raise ValueError(f"unsupported record format: {path}")


def _load_target_lengths(value: Optional[str]) -> Optional[list[int]]:
    if not value:
        return None
    if os.path.exists(value):
        with open(value, "r", encoding="utf-8") as fh:
            return [int(line.strip()) for line in fh if line.strip()]
    return [int(x) for x in value.split(",") if x.strip()]


def _valid_cds_flags(records: Sequence[object]) -> list[float]:
    return [1.0 if is_valid_cds(metrics.cds_of(r)) else 0.0 for r in records]


def _kozak_scores(records: Sequence[object]) -> list[float]:
    return [float(metrics.kozak_uaug_stats([r])["mean_kozak_score"]) for r in records]


def _motif_presence(records: Sequence[object], motif_name: str) -> list[float]:
    vals = []
    for r in records:
        vals.append(1.0 if metrics.detect_motifs(r).get(motif_name, 0) > 0 else 0.0)
    return vals


def _length_abs_errors(
    candidates: Sequence[object],
    sources: Sequence[object],
    target_lengths: Optional[Sequence[int]],
) -> list[float]:
    lc = metrics.length_control_curve(
        candidates,
        target_lengths=target_lengths,
        sources=sources if sources else None,
    )
    return [float(x) for x in lc["per_record_abs_error"]]


def _per_record_arrays(
    candidates: Sequence[object],
    sources: Sequence[object],
    oracle_scores: Sequence[Mapping[str, float]],
    source_oracle_scores: Sequence[Mapping[str, float]],
    max_edits: Optional[int],
    target_lengths: Optional[Sequence[int]],
) -> dict[str, list[float]]:
    arrays = {
        "oracle_ensemble_te": [float(s["ensemble_te"]) for s in oracle_scores],
        "oracle_ensemble_mrl": [float(s["ensemble_mrl"]) for s in oracle_scores],
        "valid_cds": _valid_cds_flags(candidates),
        "kozak_score": _kozak_scores(candidates),
        "start_accessibility": [metrics.start_accessibility_proxy(r) for r in candidates],
        "mfe_proxy": [metrics.mfe_proxy(r) for r in candidates],
        "cai": [metrics.cai(metrics.cds_of(r)) for r in candidates],
        "length_abs_error": _length_abs_errors(candidates, sources, target_lengths),
        "polyA_signal_presence": _motif_presence(candidates, "polyA_signal"),
        "uAUG_presence": _motif_presence(candidates, "uAUG"),
        "strong_kozak_presence": _motif_presence(candidates, "strong_kozak"),
    }
    frame = metrics.reading_frame_metrics(candidates)
    arrays["reading_frame_intact"] = _valid_cds_flags(candidates)
    if sources:
        arrays["protein_identity"] = [
            metrics.protein_identity(c, s) for c, s in zip(candidates, sources)
        ]
        budget = metrics.edit_budget_metrics(candidates, sources, max_edits=max_edits)
        arrays["edit_distance"] = [float(x) for x in budget["per_record_edit_distance"]]
        arrays["within_budget"] = [float(x) for x in budget["per_record_within_budget"]]
        arrays["source_oracle_ensemble_te"] = [
            float(s["ensemble_te"]) for s in source_oracle_scores
        ]
    arrays["reading_frame_intact_summary"] = [
        float(frame["reading_frame_intact_fraction"])
    ] * max(1, len(candidates))
    return arrays


def _task_summary(
    all_metrics: Mapping[str, object],
    oracle_scores: Sequence[Mapping[str, float]],
) -> dict:
    oracle_te = [float(s["ensemble_te"]) for s in oracle_scores]
    oracle_mrl = [float(s["ensemble_mrl"]) for s in oracle_scores]
    summary = {
        "T1": {
            "legal_fraction": all_metrics["legality"]["legal_fraction"],  # type: ignore[index]
            "mean_oracle_te": float(np.mean(oracle_te)) if oracle_te else 0.0,
            "mean_oracle_mrl": float(np.mean(oracle_mrl)) if oracle_mrl else 0.0,
            "mean_kozak_score": all_metrics["kozak_uaug"]["mean_kozak_score"],  # type: ignore[index]
        },
        "T3": all_metrics["diversity_novelty"],
        "T6": all_metrics["length_control"],
        "T7": all_metrics["motifs"],
    }
    if "distribution" in all_metrics:
        summary["T2"] = all_metrics["distribution"]
    if "protein_identity" in all_metrics:
        summary["T4"] = all_metrics["protein_identity"]
    if "edit_budget" in all_metrics:
        summary["T5"] = all_metrics["edit_budget"]
    return summary


def _flatten_numeric(prefix: str, payload: object, out: dict[str, float]) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key in ("per_record", "per_record_edit_distance", "per_record_within_budget", "curve"):
                continue
            _flatten_numeric(f"{prefix}.{key}" if prefix else str(key), value, out)
    elif isinstance(payload, (int, float, np.floating)) and math.isfinite(float(payload)):
        out[prefix] = float(payload)


def _format_float(value: object) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(x):
        return ""
    if abs(x) >= 100:
        return f"{x:.2f}"
    if abs(x) >= 1:
        return f"{x:.3f}"
    return f"{x:.4f}"


def _write_paper_table(summary: Mapping[str, object], path: str) -> None:
    rows = []
    task_metrics = summary["task_metrics"]
    flat = {}
    _flatten_numeric("", task_metrics, flat)
    ci = summary.get("bootstrap_ci", {})
    sig = summary.get("paired_significance", {})
    for task_id in sorted(TASK_LABELS):
        task_payload = task_metrics.get(task_id, {})  # type: ignore[union-attr]
        task_flat = {}
        _flatten_numeric("", task_payload, task_flat)
        for metric_name, value in sorted(task_flat.items()):
            ci_key = _metric_to_ci_key(task_id, metric_name)
            ci_text = ""
            if ci_key in ci:
                item = ci[ci_key]
                ci_text = f"[{_format_float(item['low'])}, {_format_float(item['high'])}]"
            p_text = ""
            if task_id == "T1" and "oracle_te_vs_source" in sig:
                p_text = _format_float(sig["oracle_te_vs_source"])
            rows.append((task_id, TASK_LABELS[task_id], metric_name, value, ci_text, p_text))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# mRNA-EditFlow Benchmark Summary\n\n")
        fh.write(f"Task id: `{summary['task_id']}`\n\n")
        fh.write("| Task | Scope | Metric | Value | 95% CI | Paired p |\n")
        fh.write("|---|---|---:|---:|---:|---:|\n")
        for task_id, label, metric_name, value, ci_text, p_text in rows:
            fh.write(
                f"| {task_id} | {label} | `{metric_name}` | "
                f"{_format_float(value)} | {ci_text} | {p_text} |\n"
            )


def _metric_to_ci_key(task_id: str, metric_name: str) -> str:
    mapping = {
        ("T1", "mean_oracle_te"): "oracle_ensemble_te",
        ("T1", "mean_oracle_mrl"): "oracle_ensemble_mrl",
        ("T1", "legal_fraction"): "valid_cds",
        ("T1", "mean_kozak_score"): "kozak_score",
        ("T4", "mean_protein_identity"): "protein_identity",
        ("T5", "within_budget_fraction"): "within_budget",
        ("T5", "mean_edit_distance"): "edit_distance",
        ("T6", "mean_abs_length_error"): "length_abs_error",
        ("T7", "polyA_signal_presence_fraction"): "polyA_signal_presence",
        ("T7", "uAUG_presence_fraction"): "uAUG_presence",
        ("T7", "strong_kozak_presence_fraction"): "strong_kozak_presence",
        ("T7", "reading_frame_intact_fraction"): "reading_frame_intact",
    }
    return mapping.get((task_id, metric_name), "")


def _json_default(obj: object) -> object:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


def run_evaluation(
    candidates: Iterable[object],
    sources: Optional[Iterable[object]] = None,
    task_id: str = "all",
    out_dir: str = "benchmark/dev",
    seeds: Optional[Sequence[int]] = None,
    n_bootstrap: int = 250,
    max_edits: Optional[int] = None,
    target_lengths: Optional[Sequence[int]] = None,
    max_pairwise_pairs: int = metrics.DEFAULT_MAX_PAIRWISE_DIVERSITY_PAIRS,
    max_novelty_sources: int = 0,
    oracle: Optional[LocalTranslationOracle] = None,
    run_mode: str = "development",
    split_contract: Optional[VerifiedSplitContract] = None,
    split_role: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    oracle_manifest: Optional[str] = None,
    training_seed: Optional[int] = None,
    decoder_seed: Optional[int] = None,
    verified_data_provenance: Optional[Mapping[str, object]] = None,
    resume_fingerprint: Optional[str] = None,
) -> dict:
    """Run offline T1-T7 evaluation and write JSON + paper table artifacts."""
    run_mode = normalize_run_mode(run_mode)
    seed_list = _ensure_seeds(seeds)
    validate_output_namespace(out_dir, run_mode)
    cand = list(candidates)
    src_raw = list(sources) if sources is not None else []
    source_records = []
    for row in src_raw:
        if isinstance(row, MRNARecord):
            source_records.append(row)
        elif isinstance(row, Mapping) and all(key in row for key in ("transcript_id", "cds")):
            source_records.append(MRNARecord.from_dict(dict(row)))
        else:
            source_records = []
            break
    if verified_data_provenance is not None:
        if run_mode == "paper":
            if split_contract is None or split_role != "test":
                raise ValueError(
                    "paper programmatic evaluation requires VerifiedSplitContract and test role"
                )
            expected_provenance = build_split_provenance(split_contract, "test")
            verify_provenance_compatibility(
                expected_provenance,
                verified_data_provenance,
                require_same_role=True,
            )
        src = src_raw
        data_provenance = dict(verified_data_provenance)
    elif source_records:
        selected_sources, data_provenance = prepare_scientific_records(
            source_records,
            run_mode=run_mode,
            split_contract=split_contract,
            split_role=split_role,
            allowed_roles=("test",),
        )
        src = selected_sources
    elif run_mode == "paper":
        raise ValueError("paper evaluation requires canonical source MRNARecords")
    else:
        src = src_raw
        data_provenance = records_identity([])
    oracle_metadata = load_and_verify_oracle_manifest(oracle_manifest, run_mode=run_mode)
    if run_mode == "paper":
        if checkpoint_path is None:
            raise ValueError("paper evaluation requires --checkpoint")
        verify_paper_checkpoint(checkpoint_path, data_provenance)
        if oracle is None or isinstance(oracle, LocalTranslationOracle):
            raise OracleContractError(
                "paper evaluation requires an independent oracle implementation matching the manifest"
            )
    oracle = oracle or LocalTranslationOracle()
    scientific_validity = build_run_metadata(
        run_mode=run_mode,
        data_provenance=data_provenance,
        config={
            "task_id": task_id,
            "n_bootstrap": n_bootstrap,
            "max_edits": max_edits,
            "max_pairwise_pairs": max_pairwise_pairs,
            "max_novelty_sources": max_novelty_sources,
        },
        code_paths=(__file__,),
        training_seed=training_seed,
        decoder_seed=decoder_seed,
        oracle=oracle_metadata,
        upstream={
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path else None,
        },
        extra_block_reasons=(
            "heuristic_functional_oracle" if run_mode == "development" else "",
        ),
        functional_claim=True,
    )
    if resume_fingerprint is not None:
        scientific_validity["resume_fingerprint"] = str(resume_fingerprint)
    oracle_scores = oracle.batch_score(cand)
    source_oracle_scores = oracle.batch_score(src) if src else []
    all_metrics = metrics.compute_all_metrics(
        cand,
        sources=src if src else None,
        budgets=max_edits if max_edits is not None else None,
        target_lengths=target_lengths,
        max_pairwise_pairs=max_pairwise_pairs,
        max_novelty_sources=max_novelty_sources,
    )
    per_record = _per_record_arrays(
        cand, src, oracle_scores, source_oracle_scores, max_edits, target_lengths
    )
    ci = {
        key: bootstrap_ci(values, seeds=seed_list, n_bootstrap=n_bootstrap)
        for key, values in per_record.items()
        if values and key != "source_oracle_ensemble_te"
    }
    paired = {}
    if src and "source_oracle_ensemble_te" in per_record:
        paired["oracle_te_vs_source"] = paired_permutation_pvalue(
            per_record["oracle_ensemble_te"],
            per_record["source_oracle_ensemble_te"],
            seed=seed_list[0],
        )
    task_metrics = _task_summary(all_metrics, oracle_scores)
    selected_task_id = str(task_id).upper()
    if selected_task_id not in TASK_LABELS and selected_task_id != "ALL":
        raise ValueError(f"task_id must be one of {sorted(TASK_LABELS)} or all")
    summary = {
        "task_id": selected_task_id,
        "n_candidates": len(cand),
        "n_sources": len(src),
        "oracle_cross_validation": oracle.cross_validate_predictors(cand),
        "oracle_scores": oracle_scores,
        "metrics": all_metrics,
        "task_metrics": task_metrics,
        "per_record_metrics": per_record,
        "bootstrap_ci": ci,
        "paired_significance": paired,
        "bootstrap_seeds": seed_list,
        "eval_config": {
            "n_bootstrap": int(n_bootstrap),
            "max_edits": max_edits,
            "max_pairwise_pairs": int(max_pairwise_pairs),
            "max_novelty_sources": int(max_novelty_sources),
        },
        "scientific_validity": scientific_validity,
    }
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "eval_summary.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, default=_json_default)
    paper_table_path = os.path.join(out_dir, "paper_table.md")
    _write_paper_table(summary, paper_table_path)
    write_provenance_sidecar(json_path, scientific_validity)
    write_provenance_sidecar(paper_table_path, scientific_validity)
    summary["json_path"] = json_path
    summary["paper_table_path"] = paper_table_path
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="JSON/JSONL/CSV/TSV candidate records")
    parser.add_argument("--sources", default=None, help="Optional source records in the same format")
    parser.add_argument("--task-id", default="all", help="T1..T7 or all")
    parser.add_argument("--out-dir", default="benchmark/dev")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=250)
    parser.add_argument("--max-edits", type=int, default=None)
    parser.add_argument(
        "--max-pairwise-pairs",
        type=int,
        default=metrics.DEFAULT_MAX_PAIRWISE_DIVERSITY_PAIRS,
        help="Exact/sampled pair count cap for pairwise diversity on long mRNAs",
    )
    parser.add_argument("--run-mode", choices=("development", "paper"), default="development")
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--split-role", choices=("train", "val", "test"), default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--oracle-manifest", default=None)
    parser.add_argument(
        "--max-novelty-sources",
        type=int,
        default=0,
        help="Per-candidate source cap for approximate novelty; <=0 keeps exact novelty",
    )
    parser.add_argument(
        "--target-lengths",
        default=None,
        help="Comma-separated target lengths or a text file with one target per line",
    )
    args = parser.parse_args(argv)
    require_paper_cli_inputs(
        run_mode=args.run_mode,
        split_manifest=args.split_manifest,
        split_role=args.split_role,
        allowed_roles=("test",),
        oracle_manifest=args.oracle_manifest,
        require_oracle=True,
    )
    candidates = load_records(args.candidates)
    sources = load_records(args.sources) if args.sources else None
    split_contract = (
        load_and_verify_split_manifest(args.split_manifest, records_path=args.sources)
        if args.split_manifest else None
    )
    run_evaluation(
        candidates,
        sources=sources,
        task_id=args.task_id,
        out_dir=args.out_dir,
        seeds=args.seeds,
        n_bootstrap=args.n_bootstrap,
        max_edits=args.max_edits,
        target_lengths=_load_target_lengths(args.target_lengths),
        max_pairwise_pairs=args.max_pairwise_pairs,
        max_novelty_sources=args.max_novelty_sources,
        run_mode=args.run_mode,
        split_contract=split_contract,
        split_role=args.split_role,
        checkpoint_path=args.checkpoint,
        oracle_manifest=args.oracle_manifest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_SEEDS",
    "TASK_LABELS",
    "bootstrap_ci",
    "paired_permutation_pvalue",
    "load_records",
    "run_evaluation",
    "main",
]
