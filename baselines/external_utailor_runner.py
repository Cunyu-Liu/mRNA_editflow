"""Thin subprocess runner around the unmodified official UTailoR workflow."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Sequence


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-root", required=True)
    parser.add_argument("--input-fasta", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--task-id", default="mef")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    tool_root = os.path.abspath(args.tool_root)
    sys.path.insert(0, tool_root)
    os.chdir(tool_root)
    os.makedirs("utailor_app/static/output", exist_ok=True)

    from utailor_app.utailor_utils.workflow import predict_rl

    with open(args.input_fasta, "r", encoding="utf-8") as fh:
        fasta = fh.read()
    frame = predict_rl(fasta, True, args.task_id)
    records = []
    for row in frame.to_dict(orient="records"):
        normalized = {
            str(key): (
                value.item()
                if hasattr(value, "item")
                else value
            )
            for key, value in row.items()
        }
        records.append(normalized)
    payload = {
        "artifact_kind": "utailor_official_raw_output",
        "task_id": args.task_id,
        "n_records": len(records),
        "records": records,
        "official_xlsx": os.path.abspath(
            f"utailor_app/static/output/{args.task_id}_report.xlsx"
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
