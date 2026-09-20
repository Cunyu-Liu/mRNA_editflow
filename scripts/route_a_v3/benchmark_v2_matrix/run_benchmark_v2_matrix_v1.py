#!/usr/bin/env python3
"""Benchmark v2 Task 2.2: frozen-delta matrix rows for 5 new families.

Preregistration: docs/paper/benchmark_v2_matrix_row_prereg_v1.md
- frozen-delta zero-tuning (pred(cand) - pred(src), no calibration)
- VALIDATION only (existing 9 tasks) + 3 new eval rows M1/M6/S1 (Task 3.5)
- Task-1 evaluator (evaluate_route2_prediction_v1.evaluate, same instance)
- append-only, missing-cell structured annotation, per-cell adaptation declaration
Runner scores one family over all 12 cells and writes predictions.jsonl + cells.
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
OUT_ROOT = MNT / "benchmark_v2/leaderboard_matrix_v2/cells"

MANIFEST = MNT / "manifests/route2_development_frozen_v1/development_manifest.jsonl"
CANON_TEMPLATE = "canonical/{study}/v1/canonical_records.private.jsonl"
NEW_ROWS = {
    "M1_MRL_EVAL_ROW": MNT / "benchmark_v2/m1_mrl_eval_row/projection_rows.jsonl",
    "M6_NDD5UTR_EVAL_ROW": MNT / "benchmark_v2/m6_ndd5utr_eval_row/projection_rows.jsonl",
    "S1_STABILITY_EVAL_ROW": MNT / "benchmark_v2/s1_stability_eval_row/projection_rows.jsonl",
}
K = 10

FIVE_UTR_REGION = "5UTR"
THREE_UTR_REGION = "3UTR"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ev = _load_module("ev", EVAL_SCRIPT)
ad = _load_module("family_adapters", ADAPTERS_PATH)

EXISTING_TASKS = [
    ("GSE114002", "5UTR", "MEAN_RIBOSOME_LOAD"),
    ("GSE269595", "3UTR", "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS"),
    ("ENCSR854RUF", "3UTR", "MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE"),
    ("GSE200304", "3UTR", "TOTAL_POLYSOME_TRANSLATION_EFFICIENCY"),
    ("GSE149487", "5UTR", "te_log2_polysome_over_totalrna"),
    ("GSE149487", "5UTR", "transcript_log2_totalrna_over_dna"),
    ("GSE186455", "3UTR", "PUBLISHED_REF_VS_ALT_ACTIVITY_LMM_LOG2_FOLD_CHANGE"),
    ("GSE217518", "5UTR", "RNA_HALF_LIFE_MINUTES"),
    ("GSE217518", "3UTR", "RNA_HALF_LIFE_MINUTES"),
]


def load_task_records():
    validation_ids = {}
    with MANIFEST.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == "VALIDATION":
                rid = str(row["canonical_record_id"])
                validation_ids[rid] = (str(row["study_unit_id"]), tuple(row["stratum"]))
    tasks = {key: [] for key in EXISTING_TASKS}
    records = {}
    for study in sorted({s for s, _, _ in EXISTING_TASKS}):
        path = MNT / CANON_TEMPLATE.format(study=study)
        if not path.exists():
            path = MNT / f"canonical/{study}/v1/canonical_records.jsonl"
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                rid = str(row["canonical_record_id"])
                if rid not in validation_ids:
                    continue
                records[rid] = row
                key = (validation_ids[rid][0], str(row["region"]), str(row["endpoint_id"]))
                if key in tasks:
                    tasks[key].append(rid)
    return {k: sorted(v) for k, v in tasks.items()}, records


def load_new_row(row_key):
    rows = []
    with NEW_ROWS[row_key].open() as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


MISSING_REASONS = {}


def score_with_missing(scorer, seqs):
    """Score and record per-row missing entries (prereg S1.5).

    A row is missing when the model forward produces non-finite output.
    Tokenization here never rejects rows (adapters map unknown chars
    deterministically per official conventions); the structured miss
    accounting still runs so miss_rate is always reported.
    """
    preds = np.asarray(scorer(seqs), dtype=float)
    finite = np.isfinite(preds)
    miss_ids = [i for i, ok in enumerate(finite) if not ok]
    return preds, miss_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True,
                        choices=["lamar_utr5te", "hydrarna", "gemorna", "utr_stcnet", "utr_insight"])
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--smoke-limit", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.physical_gpu_index}")
    out_dir = OUT_ROOT / args.family
    out_dir.mkdir(parents=True, exist_ok=True)

    family_obj, family_meta = ad.build_scorer(args.family, device)
    region_scorer = family_obj.get("score_region") if isinstance(family_obj, dict) else None
    if isinstance(family_obj, dict) and region_scorer is None:
        raise RuntimeError("GEMORNA adapter must expose score_region")

    task_records, records = load_task_records()
    new_rows = {key: load_new_row(key) for key in NEW_ROWS}

    matrix = {}
    all_preds = []

    def emit_pred(cell, row_id, source_id, cand_id, y_source, y_cand, task_tag, extra=None):
        rec = {
            "cell": cell,
            "row_id": row_id,
            "source_id": source_id,
            "candidate_id": cand_id,
            "task": task_tag,
            "pred_src": float(y_source),
            "pred_cand": float(y_cand),
            "delta_hat": float(y_cand) - float(y_source),
        }
        if extra:
            rec.update(extra)
        all_preds.append(rec)

    def score_cell(cell_tag, rows, region, observed_list, src_key, cand_key, row_id_key, obs_key, group_key, family_scoring):
        """rows: list of dicts; family_scoring(seqs, region) -> preds"""
        eligible = rows
        if args.smoke_limit:
            eligible = rows[: args.smoke_limit]
        seqs_src = [r[src_key] for r in eligible]
        seqs_cand = [r[cand_key] for r in eligible]
        try:
            y_src = np.asarray(family_scoring(seqs_src, region), dtype=float)
            y_cand = np.asarray(family_scoring(seqs_cand, region), dtype=float)
        except Exception as exc:
            matrix[cell_tag] = {
                "status": "PORT_FAILURE",
                "reason": f"{type(exc).__name__}: {exc}"[:300],
            }
            return None
        if not (len(y_src) == len(eligible) and len(y_cand) == len(eligible)):
            matrix[cell_tag] = {"status": "PORT_FAILURE", "reason": "prediction length mismatch"}
            return None
        finite = np.isfinite(y_src) & np.isfinite(y_cand)
        miss_idx = [i for i, ok in enumerate(finite) if not ok]
        kept = [eligible[i] for i in range(len(eligible)) if finite[i]]
        preds = {str(r[row_id_key]): float(y_cand[i] - y_src[i]) for i, r in enumerate(kept)}
        observations = []
        for i, r in enumerate(kept):
            observed = float(r[obs_key])
            observations.append({
                "canonical_record_id": str(r[row_id_key]),
                "study_unit_id": str(r.get("study_unit", r.get("study_unit_id", ""))),
                "source_id": str(r[group_key]),
                "biological_context_id": str(r.get("biological_context", r.get("biological_context_id", ""))),
                "endpoint_id": str(r.get("endpoint_descriptor", {}).get("endpoint_id", r.get("endpoint_id", ""))),
                "stratum": (
                    str(r.get("study_unit", r.get("study_unit_id", ""))),
                    str(r.get("region", region)),
                    str(r.get("endpoint_descriptor", {}).get("endpoint_id", r.get("endpoint_id", ""))),
                ),
                "task": (str(r.get("region", region)), str(r.get("endpoint_descriptor", {}).get("endpoint_id", r.get("endpoint_id", "")))),
                "observed": observed,
            })
        obs_values = np.asarray([o["observed"] for o in observations], dtype=float)
        finite_obs = np.isfinite(obs_values)
        if not finite_obs.all():
            keep2 = [i for i, ok in enumerate(finite_obs) if ok]
            observations = [observations[i] for i in keep2]
            preds = {o["canonical_record_id"]: preds[o["canonical_record_id"]] for o in observations}
        metrics = None
        if observations:
            try:
                metrics = ev.evaluate(observations, preds, K)
            except Exception as exc:
                matrix[cell_tag] = {
                    "status": "EVALUATOR_ERROR",
                    "reason": f"{type(exc).__name__}: {exc}"[:300],
                    "n": len(eligible),
                    "n_evaluated": len(observations),
                }
                return None
        for i, r in enumerate(kept):
            if finite[i]:
                emit_pred(cell_tag, str(r[row_id_key]), str(r[group_key]),
                          str(r.get("candidate_id", r[row_id_key])), y_src[i], y_cand[i], cell_tag,
                          {"observed": float(r[obs_key])})
        miss_rate = len(miss_idx) / len(eligible) if eligible else 0.0
        cell = {
            "status": "OK",
            "n": len(eligible),
            "n_evaluated": len(observations),
            "task_macro_spearman": metrics.get("task_macro_spearman") if metrics else None,
            "overall_spearman": metrics.get("overall_numeric", {}).get("spearman") if metrics else None,
            "source_macro_within_source_spearman": metrics.get("source_macro_within_source_spearman") if metrics else None,
            "source_macro_top_1": metrics.get("source_macro_top_1_accuracy") if metrics else None,
            "source_macro_ndcg_at_10": metrics.get("source_macro_ndcg_at_k") if metrics else None,
            "source_group_count": metrics.get("source_group_count") if metrics else None,
            "miss_count": len(miss_idx),
            "miss_rate": miss_rate,
            "miss_ids": [str(eligible[i][row_id_key]) for i in miss_idx][:200],
        }
        if miss_rate > 0.15:
            cell["HIGH_MISS"] = True
        matrix[cell_tag] = cell
        print(f"[{args.family}] {cell_tag}: rho={cell['task_macro_spearman']} n={cell['n']} miss={len(miss_idx)}", flush=True)
        return metrics

    def family_scoring(seqs, region):
        if region_scorer is not None:
            return region_scorer(seqs, region)
        return family_obj(seqs)

    # ---------------- 9 existing VALIDATION tasks
    for (study, region, endpoint), ids in task_records.items():
        tag = f"{study}|{region}|{endpoint}"
        rows = [records[rid] for rid in ids]
        for r in rows:
            r.setdefault("row_id", r["canonical_record_id"])
            r.setdefault("source_sequence", r["source_sequence"])
            r.setdefault("candidate_sequence", r["candidate_sequence"])
            r.setdefault("observed", r["direction_normalized_delta"])
            r.setdefault("group_key", r["source_id"])
            r.setdefault("study_unit", r["study_unit_id"])
            r.setdefault("region", r["region"])
            r.setdefault("endpoint_descriptor", {"endpoint_id": r["endpoint_id"]})
        score_cell(
            tag, rows, region, None,
            "source_sequence", "candidate_sequence", "row_id", "direction_normalized_delta", "source_id",
            family_scoring,
        )

    # ---------------- 3 new eval rows
    for row_key, rows in new_rows.items():
        by_region = {}
        for r in rows:
            by_region.setdefault(str(r["region"]), []).append(r)
        for region, rrows in sorted(by_region.items()):
            tag = f"{row_key}|{region}"
            score_cell(
                tag, rrows, region, None,
                "source_sequence", "candidate_sequence", "row_id", "direction_normalized_delta", "source_group_id",
                family_scoring,
            )

    result = {
        "schema_version": "benchmark_v2_matrix_family_row.v2",
        "family": args.family,
        "family_meta": family_meta,
        "evaluator": str(EVAL_SCRIPT),
        "evaluator_identity": "Task-1 evaluate_route2_prediction_v1.evaluate (same instance, K=10)",
        "split": "VALIDATION only (existing 9 tasks); new rows M1/M6/S1 full",
        "protected_test_reads": 0,
        "cpu_fallback_used": False,
        "cuda_provenance": {
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU",
        },
        "matrix": matrix,
    }
    (out_dir / "cells.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    with (out_dir / "predictions.jsonl").open("w") as handle:
        for p in all_preds:
            handle.write(json.dumps(p) + "\n")
    print(f"wrote {out_dir}/cells.json and predictions.jsonl ({len(all_preds)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
