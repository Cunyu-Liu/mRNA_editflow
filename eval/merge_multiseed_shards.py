"""Merge completed seed shards into one canonical multiseed summary.

This is intended for long T6/T5 runs where independent seed shards are launched
on separate GPUs. It only consumes completed ``seed_XXX/eval_summary.json``
artifacts and reuses :mod:`mrna_editflow.eval.run_multiseed_benchmark` for metric
extraction, seed bootstrap aggregation, and table writing.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Mapping, Optional, Sequence

from mrna_editflow.eval import run_multiseed_benchmark


DEFAULT_EXPECTED_SEEDS: tuple[int, ...] = tuple(range(10))


def _load_json(path: str) -> Mapping[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _seed_dir_name(seed: int) -> str:
    return f"seed_{int(seed):03d}"


def _load_completed_seed(seed_dir: str, seed: int) -> Optional[dict[str, object]]:
    cand_path = os.path.join(seed_dir, "candidates.jsonl")
    eval_path = os.path.join(seed_dir, "eval_summary.json")
    paper_path = os.path.join(seed_dir, "paper_table.md")
    if not (os.path.exists(cand_path) and os.path.exists(eval_path)):
        return None
    summary = _load_json(eval_path)
    scalars = run_multiseed_benchmark.extract_scalar_metrics(summary)
    return {
        "seed": int(seed),
        "resumed": True,
        "shard_merged": True,
        "source_seed_dir": seed_dir,
        "candidate_path": cand_path,
        "eval_json_path": eval_path,
        "paper_table_path": paper_path,
        "metrics": scalars,
    }


def collect_completed_seed_rows(
    source_dirs: Sequence[str],
    *,
    expected_seeds: Sequence[int] = DEFAULT_EXPECTED_SEEDS,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[int]]:
    """Collect one completed row per expected seed.

    Source directory order defines priority. Duplicate completed seeds are
    recorded for auditability but do not replace the first accepted seed.
    """
    accepted: dict[int, dict[str, object]] = {}
    duplicates: list[dict[str, object]] = []
    for source_dir in source_dirs:
        for seed in expected_seeds:
            seed_i = int(seed)
            seed_dir = os.path.join(source_dir, _seed_dir_name(seed_i))
            row = _load_completed_seed(seed_dir, seed_i)
            if row is None:
                continue
            if seed_i in accepted:
                duplicates.append(
                    {
                        "seed": seed_i,
                        "accepted_eval_json_path": accepted[seed_i]["eval_json_path"],
                        "duplicate_eval_json_path": row["eval_json_path"],
                    }
                )
                continue
            accepted[seed_i] = row

    expected = [int(seed) for seed in expected_seeds]
    missing = [seed for seed in expected if seed not in accepted]
    rows = [accepted[seed] for seed in expected if seed in accepted]
    return rows, duplicates, missing


def _first_existing_source_path(source_dirs: Sequence[str], fallback: Optional[str]) -> Optional[str]:
    if fallback:
        return fallback
    for source_dir in source_dirs:
        source_path = os.path.join(source_dir, "sources.jsonl")
        if os.path.exists(source_path):
            return source_path
    return None


def _count_jsonl(path: Optional[str]) -> int:
    if not path or not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _base_config_from_summary(source_dirs: Sequence[str]) -> dict[str, object]:
    for source_dir in source_dirs:
        summary_path = os.path.join(source_dir, "multiseed_summary.json")
        if not os.path.exists(summary_path):
            continue
        summary = _load_json(summary_path)
        config = summary.get("config", {})
        if isinstance(config, Mapping):
            return dict(config)
    return {}


def _merged_config(
    *,
    source_dirs: Sequence[str],
    expected_seeds: Sequence[int],
    source_path: Optional[str],
    task_id: str,
    checkpoint_path: Optional[str],
    cascade_recall_checkpoint_path: Optional[str],
    edit_budget: int,
    target_length_delta: int,
    proposal_top_k: int,
    cascade_recall_top_k: int,
    proposal_temperature: float,
    guidance_scale: float,
    target_te: Optional[float],
    target_start_accessibility: Optional[float],
    editable_regions: Optional[Sequence[str]],
    n_bootstrap: int,
    max_pairwise_pairs: int,
    max_novelty_sources: int,
) -> dict[str, object]:
    config = _base_config_from_summary(source_dirs)
    decoder_family = (
        "cascade"
        if cascade_recall_checkpoint_path
        else ("checkpoint_guided" if checkpoint_path else "deterministic")
    )
    effective_top_k = cascade_recall_top_k if cascade_recall_checkpoint_path else proposal_top_k
    config.update(
        {
            "task_id": task_id,
            "checkpoint_path": checkpoint_path,
            "cascade_recall_checkpoint_path": cascade_recall_checkpoint_path,
            "decoder_family": decoder_family,
            "n_records": int(config.get("n_records") or _count_jsonl(source_path)),
            "seeds": [int(seed) for seed in expected_seeds],
            "edit_budget": int(edit_budget),
            "target_length_delta": int(target_length_delta),
            "proposal_top_k": int(proposal_top_k),
            "cascade_recall_top_k": int(cascade_recall_top_k),
            "effective_proposal_top_k": int(effective_top_k),
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
            "resume": True,
            "shard_merged": True,
            "merge_source_dirs": list(source_dirs),
        }
    )
    return config


def merge_multiseed_shards(
    *,
    source_dirs: Sequence[str],
    out_dir: str,
    expected_seeds: Sequence[int] = DEFAULT_EXPECTED_SEEDS,
    source_path: Optional[str] = None,
    task_id: str = "T6",
    checkpoint_path: Optional[str] = None,
    cascade_recall_checkpoint_path: Optional[str] = None,
    edit_budget: int = 30,
    target_length_delta: int = 0,
    proposal_top_k: int = 64,
    cascade_recall_top_k: int = 64,
    proposal_temperature: float = 1.0,
    guidance_scale: float = 0.0,
    target_te: Optional[float] = None,
    target_start_accessibility: Optional[float] = None,
    editable_regions: Optional[Sequence[str]] = None,
    n_bootstrap: int = 1000,
    max_pairwise_pairs: int = 64,
    max_novelty_sources: int = 0,
    require_complete: bool = True,
) -> dict[str, object]:
    """Merge seed-level artifacts and write ``multiseed_summary.json``."""
    if not source_dirs:
        raise ValueError("at least one source directory is required")
    expected = [int(seed) for seed in expected_seeds]
    rows, duplicates, missing = collect_completed_seed_rows(source_dirs, expected_seeds=expected)
    if missing and require_complete:
        raise ValueError(f"missing completed seeds: {missing}")

    os.makedirs(out_dir, exist_ok=True)
    merged_source_path = _first_existing_source_path(source_dirs, source_path)
    per_seed_metrics = [
        {str(k): float(v) for k, v in row["metrics"].items()}  # type: ignore[union-attr]
        for row in rows
    ]
    seed_list = [int(row["seed"]) for row in rows]
    aggregate = run_multiseed_benchmark.aggregate_seed_metrics(
        per_seed_metrics,
        seed_list,
        n_bootstrap=n_bootstrap,
    )
    table_path = run_multiseed_benchmark.write_multiseed_table(
        aggregate,
        os.path.join(out_dir, "multiseed_table.md"),
    )
    progress_path = os.path.join(out_dir, "multiseed_merge_progress.jsonl")
    config = _merged_config(
        source_dirs=source_dirs,
        expected_seeds=expected,
        source_path=merged_source_path,
        task_id=task_id,
        checkpoint_path=checkpoint_path,
        cascade_recall_checkpoint_path=cascade_recall_checkpoint_path,
        edit_budget=edit_budget,
        target_length_delta=target_length_delta,
        proposal_top_k=proposal_top_k,
        cascade_recall_top_k=cascade_recall_top_k,
        proposal_temperature=proposal_temperature,
        guidance_scale=guidance_scale,
        target_te=target_te,
        target_start_accessibility=target_start_accessibility,
        editable_regions=editable_regions,
        n_bootstrap=n_bootstrap,
        max_pairwise_pairs=max_pairwise_pairs,
        max_novelty_sources=max_novelty_sources,
    )
    config["progress_jsonl"] = progress_path

    result: dict[str, object] = {
        "config": config,
        "source_path": merged_source_path,
        "progress_jsonl": progress_path,
        "per_seed": rows,
        "aggregate": aggregate,
        "table_path": table_path,
        "merge_audit": {
            "source_dirs": list(source_dirs),
            "expected_seeds": expected,
            "merged_seeds": seed_list,
            "missing_seeds": missing,
            "duplicate_seeds": duplicates,
            "complete": not missing,
        },
    }
    json_path = os.path.join(out_dir, "multiseed_summary.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    with open(progress_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "merge_complete", "json_path": json_path}, sort_keys=True) + "\n")
    result["json_path"] = json_path
    return result


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expected-seeds", nargs="*", type=int, default=list(DEFAULT_EXPECTED_SEEDS))
    parser.add_argument("--source-path", default=None)
    parser.add_argument("--task-id", default="T6")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cascade-recall-checkpoint", default=None)
    parser.add_argument("--edit-budget", type=int, default=30)
    parser.add_argument("--target-length-delta", type=int, default=0)
    parser.add_argument("--proposal-top-k", type=int, default=64)
    parser.add_argument("--cascade-recall-top-k", type=int, default=64)
    parser.add_argument("--proposal-temperature", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--target-te", type=float, default=None)
    parser.add_argument("--target-start-accessibility", type=float, default=None)
    parser.add_argument(
        "--editable-regions",
        nargs="+",
        choices=("utr5", "utr3"),
        default=None,
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--max-pairwise-pairs", type=int, default=64)
    parser.add_argument("--max-novelty-sources", type=int, default=0)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    result = merge_multiseed_shards(
        source_dirs=args.source_dir,
        out_dir=args.out_dir,
        expected_seeds=args.expected_seeds,
        source_path=args.source_path,
        task_id=args.task_id,
        checkpoint_path=args.checkpoint,
        cascade_recall_checkpoint_path=args.cascade_recall_checkpoint,
        edit_budget=args.edit_budget,
        target_length_delta=args.target_length_delta,
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
        require_complete=not args.allow_incomplete,
    )
    print(json.dumps({"json_path": result["json_path"], "table_path": result["table_path"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_EXPECTED_SEEDS",
    "collect_completed_seed_rows",
    "merge_multiseed_shards",
    "main",
]
