"""Summarise proposal-ranking JSONL artifacts.

Long full-pool audits can write large candidate JSONL files before the final
summary JSON is available. This module reconstructs the same paper-facing
ranking metrics directly from the JSONL rows, making audits inspectable,
recoverable and suitable for intermediate monitoring.

For each source transcript ``x`` with candidate rows ``C(x)``, the summary uses
the stored ``model_rank`` and ``oracle_rank`` to compute

``regret(x) = TE(oracle-rank-1 candidate) - TE(model-rank-1 candidate)``.

The aggregate ``oracle_best_in_model_top_k_fraction`` is computed only over
transcripts with at least one candidate. Complexity is ``O(N)`` JSON rows plus
``O(R)`` record summaries, where ``N`` is the number of candidates and ``R`` is
the number of source transcripts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Mapping, Optional, Sequence

from mrna_editflow.eval.proposal_ranking import _aggregate_record_summaries, _as_float


def _rank_value(row: Mapping[str, object], key: str) -> int:
    try:
        value = int(row.get(key, 10**12))
    except (TypeError, ValueError):
        value = 10**12
    return value if value > 0 else 10**12


def _record_summary_from_rows(
    transcript_id: str,
    rows: Sequence[Mapping[str, object]],
    *,
    top_k: int,
) -> dict[str, object]:
    if not rows:
        return {
            "transcript_id": transcript_id,
            "n_candidates": 0,
            "source_te": 0.0,
            "model_top_te": 0.0,
            "oracle_top_te": 0.0,
            "model_regret": 0.0,
            "oracle_best_model_rank": 0,
            "model_top_oracle_rank": 0,
            "oracle_best_in_model_top_k": False,
        }
    model_top = min(rows, key=lambda row: _rank_value(row, "model_rank"))
    oracle_top = min(rows, key=lambda row: _rank_value(row, "oracle_rank"))
    k = len(rows) if int(top_k) <= 0 else max(1, int(top_k))
    model_top_te = _as_float(model_top.get("oracle_te"))
    oracle_top_te = _as_float(oracle_top.get("oracle_te"))
    oracle_best_model_rank = _rank_value(oracle_top, "model_rank")
    return {
        "transcript_id": transcript_id,
        "n_candidates": len(rows),
        "source_te": _as_float(model_top.get("source_te")),
        "model_top_te": model_top_te,
        "oracle_top_te": oracle_top_te,
        "model_regret": float(oracle_top_te - model_top_te),
        "oracle_best_model_rank": int(oracle_best_model_rank if oracle_best_model_rank < 10**12 else 0),
        "model_top_oracle_rank": int(_rank_value(model_top, "oracle_rank")),
        "oracle_best_in_model_top_k": bool(oracle_best_model_rank <= k),
    }


def summarise_proposal_jsonl(
    jsonl_path: str,
    *,
    out_json: Optional[str] = None,
    top_k: int = 32,
) -> dict[str, object]:
    """Summarise proposal-ranking candidate rows from ``jsonl_path``."""
    grouped: dict[str, list[Mapping[str, object]]] = {}
    n_rows = 0
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                continue
            tid = str(row.get("transcript_id", ""))
            if not tid:
                continue
            grouped.setdefault(tid, []).append(row)
            n_rows += 1

    per_record = [
        _record_summary_from_rows(tid, rows, top_k=top_k)
        for tid, rows in sorted(grouped.items())
    ]
    aggregate = _aggregate_record_summaries(per_record)
    aggregate["n_candidates"] = int(n_rows)
    result: dict[str, object] = {
        "config": {
            "source_jsonl": jsonl_path,
            "top_k": int(top_k),
            "n_jsonl_rows": int(n_rows),
            "complete_summary": False,
        },
        "aggregate": aggregate,
        "per_record": per_record,
    }
    if out_json is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, sort_keys=True)
    return result


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise proposal-ranking candidate JSONL")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--top-k", type=int, default=32)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    result = summarise_proposal_jsonl(args.jsonl, out_json=args.out_json, top_k=args.top_k)
    print(json.dumps({"out_json": args.out_json, "aggregate": result["aggregate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
