#!/usr/bin/env python3
"""Benchmark v2 Task 2.2: frozen-delta matrix rows for new model families.

Preregistration: benchmark_v2_matrix_row_prereg_v1.md (frozen-delta zero-tuning,
VALIDATION only, Task-1 evaluator, append-only rows) + port ledger
benchmark_v2_port_ledger_v1.md (license waived by user decision, journal batch 110).

This runner scores the five new families on the 9 existing VALIDATION tasks plus
the three new eval rows (M1/M6/S1) and writes predictions.jsonl + row records.

The model adapters are imported from benchmark_v2_family_adapters_v1.py (same dir).
Each adapter provides build_scorer(device) -> callable(list[str]) -> np.ndarray.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_w0_diagnosis_20260902")
ADAPTERS_PATH = REPO_ROOT / "scripts/route_a_v3/benchmark_v2_family_adapters_v1.py"
EVAL_REPO = Path("/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901")
EVAL_SCRIPT = EVAL_REPO / "scripts/route_a_v3/evaluate_route2_prediction_v1.py"
MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
OUT_ROOT = MNT / "experiments/analysis_benchmark_v2_matrix_v1"

MANIFEST = MNT / "manifests/route2_development_frozen_v1/development_manifest.jsonl"
CANON_TEMPLATE = "canonical/{study}/v1/canonical_records.private.jsonl"
NEW_ROWS = {
    "M1_MRL_EVAL_ROW": MNT / "benchmark_v2/m1_mrl_eval_row/projection_rows.jsonl",
    "M6_NDD5UTR_EVAL_ROW": MNT / "benchmark_v2/m6_ndd5utr_eval_row/projection_rows.jsonl",
    "S1_STABILITY_EVAL_ROW": MNT / "benchmark_v2/s1_stability_eval_row/projection_rows.jsonl",
}
K = 10


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ev = _load_module("ev", EVAL_SCRIPT)
ad = _load_module("family_adapters", ADAPTERS_PATH)

# the 9 existing VALIDATION tasks (study, region, endpoint) -- deduplicated strata
EXISTING_TASKS = [
    ("GSE114002", "5UTR", "MEAN_RIBOSOME_LOAD"),
    ("GSE269595", "3UTR", "POLY_A_SITE_USAGE"),
    ("ENCSR854RUF", "5UTR", "MPRAUTR_V1_ACTIVITY"),
    ("GSE200304", "3UTR", "TOTAL_POLYSOME_TRANSLATION_EFFICIENCY"),
    ("GSE149487", "5UTR", "te_log2_polysome_over_totalrna"),
    ("GSE149487", "5UTR", "transcript_log2_totalrna_over_dna"),
    ("GSE186455", "3UTR", "ENDOGENOUS_TTR_LOG2_FOLD"),
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
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                rid = str(row["canonical_record_id"])
                if rid not in validation_ids:
                    continue
                records[rid] = row
                key = (validation_ids[rid][0], tuple(row["stratum"]))
                if key in tasks:
                    tasks[key].append(rid)
    return {k: sorted(v) for k, v in tasks.items()}, records


def load_new_row(row_key):
    rows = []
    with NEW_ROWS[row_key].open() as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=["lamar_utr5te", "hydrarna", "gemorna", "utr_stcnet", "utr_insight"])
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--smoke-limit", type=int, default=0)
    args = parser.parse_args()

    import torch

    device = torch.device(f"cuda:{args.physical_gpu_index}")
    out_dir = OUT_ROOT / args.family
    out_dir.mkdir(parents=True, exist_ok=True)

    scorer, family_meta = ad.build_scorer(args.family, device)
    task_records, records = load_task_records()
    new_rows = {key: load_new_row(key) for key in NEW_ROWS}

    matrix = {}
    pred_handles = {}
    all_preds = []

    def emit_pred(row_id, source_id, cand_id, y_source, y_cand, task_tag):
        all_preds.append({
            "row_id": row_id,
            "source_id": source_id,
            "cand_id": cand_id,
            "task": task_tag,
            "y_source": float(y_source),
            "y_cand": float(y_cand),
            "delta_pred": float(y_cand) - float(y_source),
        })

    # --- 9 existing tasks (5'UTR-capable families get all 5'UTR rows + 3'UTR rows
    # are scored too, exactly as the frozen-9 classical rows did: the model is a
    # 5'UTR head but rows are scored as-is, the mismatch is declared, not hidden).
    score_seqs = scorer
    for (study, region, endpoint), ids in task_records.items():
        tag = f"{study}|{region}|{endpoint}"
        if args.smoke_limit:
            ids = ids[: args.smoke_limit]
        obs = {}
        for rid in ids:
            row = records[rid]
            obs[rid] = row
        try:
            y_src = score_seqs([records[rid]["source_sequence"] for rid in ids])
            y_cand = score_seqs([records[rid]["candidate_sequence"] for rid in ids])
        except Exception as exc:  # paradigm/port failure declared, not faked
            matrix[tag] = {"status": "PORT_FAILURE", "reason": f"{type(exc).__name__}: {exc}"[:300]}
            continue
        preds = {rid: float(y_cand[i] - y_src[i]) for i, rid in enumerate(ids)}
        observations = []
        for rid in ids:
            row = records[rid]
            observations.append({
                "canonical_record_id": rid,
                "study_unit_id": str(row["study_unit_id"]),
                "source_id": str(row["source_id"]),
                "biological_context_id": str(row["biological_context_id"]),
                "endpoint_id": str(row["endpoint_id"]),
                "stratum": (str(row["study_unit_id"]), str(row["region"]), str(row["endpoint_id"])),
                "task": (str(row["region"]), str(row["endpoint_id"])),
                "observed": float(row["direction_normalized_delta"]),
            })
        metrics = ev.evaluate(observations, preds, K)
        for i, rid in enumerate(ids):
            emit_pred(rid, records[rid]["source_id"], records[rid]["candidate_id"],
                      y_src[i], y_cand[i], tag)
        matrix[tag] = {
            "status": "OK",
            "n": len(ids),
            "task_macro_spearman": metrics.get("task_macro_spearman"),
            "top_1": metrics.get("source_macro_top_1_accuracy"),
            "ndcg_at_10": metrics.get("source_macro_ndcg_at_k"),
            "overall_spearman": metrics.get("overall_numeric", {}).get("spearman"),
        }
        print(f"[{args.family}] {tag}: rho={matrix[tag]['task_macro_spearman']} n={len(ids)}", flush=True)

    # --- three new rows
    for row_key, rows in new_rows.items():
        eligible = rows
        region_filter = family_meta.get("new_row_region_filter")
        if region_filter:
            eligible = [r for r in rows if r["region"] in region_filter]
        if args.smoke_limit:
            eligible = eligible[: args.smoke_limit]
        ids = [r["row_id"] for r in eligible]
        tag = row_key
        try:
            y_src = score_seqs([r["source_sequence"] for r in eligible])
            y_cand = score_seqs([r["candidate_sequence"] for r in eligible])
        except Exception as exc:
            matrix[tag] = {"status": "PORT_FAILURE", "reason": f"{type(exc).__name__}: {exc}"[:300]}
            continue
        preds = {r["row_id"]: float(y_cand[i] - y_src[i]) for i, r in enumerate(eligible)}
        observations = []
        for r in eligible:
            observations.append({
                "canonical_record_id": r["row_id"],
                "study_unit_id": str(r["study_unit"]),
                "source_id": str(r["source_group_id"]),
                "biological_context_id": str(r.get("biological_context", "")),
                "endpoint_id": str(r["endpoint_descriptor"]["endpoint_id"]),
                "stratum": (str(r["study_unit"]), str(r["region"]), str(r["endpoint_descriptor"]["endpoint_id"])),
                "task": (str(r["region"]), str(r["endpoint_descriptor"]["endpoint_id"])),
                "observed": float(r["direction_normalized_delta"]),
            })
        metrics = ev.evaluate(observations, preds, K)
        for i, r in enumerate(eligible):
            emit_pred(r["row_id"], r["source_group_id"], r["row_id"], y_src[i], y_cand[i], tag)
        # robust companion stats for S1 (declared in row_definition)
        robust = {}
        if row_key == "S1_STABILITY_EVAL_ROW":
            obs_arr = np.array([float(r["direction_normalized_delta"]) for r in eligible])
            pred_arr = np.array([preds[r["row_id"]] for r in eligible])
            from scipy.stats import spearmanr
            robust["overall_spearman_full"] = float(spearmanr(obs_arr, pred_arr).statistic)
            dao = [i for i, r in enumerate(eligible) if "cryptic_splice_risk_dao_qc" in (r.get("tags") or [])]
            if dao:
                sub_obs = obs_arr[dao]
                sub_pred = pred_arr[dao]
                robust["dao_flagged_subset"] = {
                    "n": len(dao),
                    "spearman": float(spearmanr(sub_obs, sub_pred).statistic),
                }
            pass_sub = [i for i, r in enumerate(eligible) if "cryptic_splice_risk_dao_qc" not in (r.get("tags") or [])]
            if pass_sub:
                robust["dao_pass_subset"] = {
                    "n": len(pass_sub),
                    "spearman": float(spearmanr(obs_arr[pass_sub], sub_pred[pass_sub] if False else pred_arr[pass_sub]).statistic),
                }
        matrix[tag] = {
            "status": "OK",
            "n": len(eligible),
            "task_macro_spearman": metrics.get("task_macro_spearman"),
            "top_1": metrics.get("source_macro_top_1_accuracy"),
            "ndcg_at_10": metrics.get("source_macro_ndcg_at_k"),
            "overall_spearman": metrics.get("overall_numeric", {}).get("spearman"),
            **robust,
        }
        print(f"[{args.family}] {tag}: rho={matrix[tag]['task_macro_spearman']} n={len(eligible)}", flush=True)

    (out_dir / "matrix_cells.json").write_text(json.dumps({
        "schema_version": "benchmark_v2_matrix_family_row.v1",
        "family": args.family,
        "family_meta": family_meta,
        "evaluator": str(EVAL_SCRIPT),
        "K": K,
        "split": "VALIDATION (existing 9 tasks) + NEW_EVAL_ROW (M1/M6/S1)",
        "cpu_fallback_used": False,
        "matrix": matrix,
    }, indent=1, sort_keys=True))

    with (out_dir / "predictions.jsonl").open("w") as handle:
        for p in all_preds:
            handle.write(json.dumps(p) + "\n")
    print(f"wrote {out_dir}/matrix_cells.json and predictions.jsonl ({len(all_preds)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
