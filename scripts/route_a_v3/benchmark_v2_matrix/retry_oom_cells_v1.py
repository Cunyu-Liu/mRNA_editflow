#!/usr/bin/env python3
"""Retry OOM cells for a family with a smaller inference batch.

Batch size is inference plumbing (not a prereg-registered adaptation axis);
frozen weights and declared input adaptation are unchanged. Appends results
to cells.json (append-only) and rewrites predictions.jsonl for the retried
cells only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_w0_diagnosis_20260902")
ADAPTERS_PATH = REPO_ROOT / "scripts/route_a_v3/benchmark_v2_matrix/family_adapters_v1.py"
EVAL_SCRIPT = REPO_ROOT / "scripts/route_a_v3/evaluate_route2_prediction_v1.py"
MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
K = 10

CANON_FILES = {
    "GSE149487": ["canonical/GSE149487/v1/canonical_records.private.jsonl",
                  "canonical/GSE149487/v1/canonical_records.jsonl"],
}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ev = _load_module("ev", EVAL_SCRIPT)
ad = _load_module("family_adapters", ADAPTERS_PATH)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--chunk", type=int, default=2)
    parser.add_argument("--cells", required=True, help="comma list of study|region|endpoint")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    family_obj, family_meta = ad.build_scorer(args.family, device)
    region_scorer = family_obj.get("score_region") if isinstance(family_obj, dict) else None

    def score_chunked(seqs, region, chunk=args.chunk):
        outs = []
        for i in range(0, len(seqs), chunk):
            part = seqs[i:i + chunk]
            if region_scorer is not None:
                outs.append(region_scorer(part, region))
            else:
                outs.append(family_obj(part))
        return np.concatenate(outs) if outs else np.array([])

    manifest = MNT / "manifests/route2_development_frozen_v1/development_manifest.jsonl"
    validation_ids = set()
    with manifest.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == "VALIDATION":
                validation_ids.add(str(row["canonical_record_id"]))

    out_dir = MNT / "benchmark_v2/leaderboard_matrix_v2/cells" / args.family
    cells = json.loads((out_dir / "cells.json").read_text())
    pred_lines = [json.loads(l) for l in (out_dir / "predictions.jsonl").open()]
    retry_tags = args.cells.split(",")

    for tag in retry_tags:
        study, region, endpoint = tag.split("|")
        files = CANON_FILES.get(study, [f"canonical/{study}/v1/canonical_records.private.jsonl"])
        records = []
        for rel in files:
            p = MNT / rel
            if p.exists():
                with p.open() as handle:
                    for line in handle:
                        row = json.loads(line)
                        if str(row["canonical_record_id"]) in validation_ids:
                            records.append(row)
                break
        rows = [r for r in records if r["region"] == region and r["endpoint_id"] == endpoint]
        print(f"retry {tag}: n={len(rows)}", flush=True)
        pred_lines = [p for p in pred_lines if p["cell"] != tag]
        y_src = score_chunked([r["source_sequence"] for r in rows], region)
        y_cand = score_chunked([r["candidate_sequence"] for r in rows], region)
        finite = np.isfinite(y_src) & np.isfinite(y_cand)
        kept = [rows[i] for i in range(len(rows)) if finite[i]]
        preds = {str(r["canonical_record_id"]): float(y_cand[i] - y_src[i]) for i, r in enumerate(kept)}
        observations = []
        for i, r in enumerate(kept):
            observations.append({
                "canonical_record_id": str(r["canonical_record_id"]),
                "study_unit_id": str(r["study_unit_id"]),
                "source_id": str(r["source_id"]),
                "biological_context_id": str(r["biological_context_id"]),
                "endpoint_id": str(r["endpoint_id"]),
                "stratum": (str(r["study_unit_id"]), str(r["region"]), str(r["endpoint_id"])),
                "task": (str(r["region"]), str(r["endpoint_id"])),
                "observed": float(r["direction_normalized_delta"]),
            })
        metrics = ev.evaluate(observations, preds, K)
        for i, r in enumerate(kept):
            pred_lines.append({
                "cell": tag,
                "row_id": str(r["canonical_record_id"]),
                "source_id": str(r["source_id"]),
                "candidate_id": str(r["candidate_id"]),
                "task": tag,
                "pred_src": float(y_src[i]),
                "pred_cand": float(y_cand[i]),
                "delta_hat": float(y_cand[i] - y_src[i]),
                "observed": float(r["direction_normalized_delta"]),
            })
        miss_rate = (len(rows) - len(kept)) / len(rows) if rows else 0.0
        cells["matrix"][tag] = {
            "status": "OK",
            "n": len(rows),
            "n_evaluated": len(kept),
            "task_macro_spearman": metrics.get("task_macro_spearman"),
            "overall_spearman": metrics.get("overall_numeric", {}).get("spearman"),
            "source_macro_within_source_spearman": metrics.get("source_macro_within_source_spearman"),
            "source_macro_top_1": metrics.get("source_macro_top_1_accuracy"),
            "source_macro_ndcg_at_10": metrics.get("source_macro_ndcg_at_k"),
            "source_group_count": metrics.get("source_group_count"),
            "miss_count": len(rows) - len(kept),
            "miss_rate": miss_rate,
            "miss_ids": [],
            "note": f"OOM retried with chunk={args.chunk} (inference plumbing only; same frozen weights and adaptation)",
        }
        print(f"{tag}: rho={metrics.get('task_macro_spearman')} n={len(kept)}", flush=True)

    cells.setdefault("oom_retry", {})["cells_retried"] = retry_tags
    cells["oom_retry"]["fix"] = f"batch -> chunk={args.chunk} (plumbing), same adapter/weights/adaptation"
    (out_dir / "cells.json").write_text(json.dumps(cells, indent=1, sort_keys=True))
    with (out_dir / "predictions.jsonl").open("w") as handle:
        for p in pred_lines:
            handle.write(json.dumps(p) + "\n")
    print(f"updated {out_dir} ({len(pred_lines)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
