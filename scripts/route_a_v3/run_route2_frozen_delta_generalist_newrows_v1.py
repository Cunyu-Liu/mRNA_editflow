#!/usr/bin/env python3
"""Amendment v3 (benchmark_v2_generalist_rows_amendment_v3_new_rows.md):
generalist backbones (batch-2 + batch-0/1) x three new eval rows (M1/M6/S1).

Caliber (frozen in amendment v3, 2026-10-06):
- Frozen official backbones, zero gradients; linear probe on embedding
  differences (te-family protocol, seed 20260816, AdamW lr 1e-3 wd 1e-4,
  100 epochs, source-group-weighted MSE, epoch selection on the SAME
  source records as fit - the IN_STUDY_PROBE convention: fit = same-endpoint
  task TRAIN split, no new-row label is ever touched).
- Eval surface = NEW_EVAL_ROW full rows (M1 2805 / M6 800 / S1 dual-arm
  sub-rows), observations built from projection_rows.jsonl with the exact
  evaluator field contract (canonical_record_id / study_unit_id /
  source_id / biological_context_id / endpoint_id / stratum / task /
  observed), evaluated with the frozen Task-1 evaluator (K=10).
- reporting-only: no gate, append-only outputs, no existing row modified.

Probe fit surfaces (per amendment v3 section 2):
- m1_mrl        -> fit on MRL task TRAIN  (GSE114002 5UTR, MEAN_RIBOSOME_LOAD)
- m6_ndd5utr    -> fit on MRL task TRAIN  (same-endpoint 5UTR rationale)
- s1_stability 5UTR arm -> fit on half_life_5utr task TRAIN
- s1_stability 3UTR arm -> fit on half_life_3utr task TRAIN
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

TE_PATH = REPO_ROOT / "scripts/route_a_v3/run_route2_frozen_delta_te_family_v1.py"
GEN_PATH = REPO_ROOT / "scripts/route_a_v3/run_route2_frozen_delta_generalist_v2.py"

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
NEW_ROWS_ROOT = MNT / "benchmark_v2"
EVALUATOR_PATH = HERE / "evaluate_route2_prediction_v1.py"
K = 10


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


te = _load_module("route2_te_lib_newrows", TE_PATH)
# half-life task specs are injected by the generalist runner (same-endpoint
# probe surfaces for S1 arms); mirror them here so te.TASKS is complete.
te.TASKS["half_life_5utr"] = {
    "study": "GSE217518", "region": "5UTR",
    "endpoint": "RNA_HALF_LIFE_MINUTES", "mode": "IN_STUDY_PROBE",
}
te.TASKS["half_life_3utr"] = {
    "study": "GSE217518", "region": "3UTR",
    "endpoint": "RNA_HALF_LIFE_MINUTES", "mode": "IN_STUDY_PROBE",
}
gen = _load_module("route2_gen_lib_newrows", GEN_PATH)
ev = _load_module("route2_ev_lib_newrows", EVALUATOR_PATH)

NEW_ROW_TASKS = {
    "m1_mrl": {
        "dir": "m1_mrl_eval_row", "region": "5UTR",
        "fit_task": "gse114002_mrl",
    },
    "m6_ndd5utr": {
        "dir": "m6_ndd5utr_eval_row", "region": "5UTR",
        "fit_task": "gse114002_mrl",
    },
    "s1_stability_5utr": {
        "dir": "s1_stability_eval_row", "region": "5UTR",
        "fit_task": "half_life_5utr",
    },
    "s1_stability_3utr": {
        "dir": "s1_stability_eval_row", "region": "3UTR",
        "fit_task": "half_life_3utr",
    },
}

MODEL_CHOICES = list(gen.MODEL_CHOICES)


def load_new_row(task_key: str) -> tuple:
    spec = NEW_ROW_TASKS[task_key]
    path = NEW_ROWS_ROOT / spec["dir"] / "projection_rows.jsonl"
    rows: list[te.PairRecord] = []
    meta: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row["region"]) != spec["region"]:
                continue
            source = str(row["source_sequence"]).upper()
            candidate = str(row["candidate_sequence"]).upper()
            target = float(row["direction_normalized_delta"])
            rid = str(row["row_id"])
            rows.append(
                te.PairRecord(
                    record_id=rid,
                    source_id=str(row.get("source_group_id", "")),
                    study=str(row["study_unit"]),
                    source=source,
                    candidate=candidate,
                    target=target,
                )
            )
            meta[rid] = {
                "context": str(row["biological_context"]),
                "region": str(row["region"]),
                "endpoint": str(row["endpoint_descriptor"].get("endpoint_id", "")),
            }
    assert rows, f"new row task {task_key} is empty"
    return rows, meta


def observation_of(record: te.PairRecord, context: str, region: str, endpoint: str) -> dict:
    return {
        "canonical_record_id": record.record_id,
        "study_unit_id": record.study,
        "source_id": record.source_id,
        "biological_context_id": context,
        "endpoint_id": endpoint,
        "stratum": (record.study, region, endpoint),
        "task": (region, endpoint),
        "observed": record.target,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--tasks", nargs="+", default=list(NEW_ROW_TASKS),
                        choices=list(NEW_ROW_TASKS))
    parser.add_argument("--models", nargs="+", default=["orthrus_4track"],
                        choices=MODEL_CHOICES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-limit", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - hard gate (amendment v3 section 2)")
    device = torch.device(f"cuda:{args.physical_gpu_index}")
    torch.cuda.set_device(device)

    manifest_rows = te.load_manifest_rows()

    raw = {task: load_new_row(task) for task in args.tasks}
    eval_rows = {task: v[0] for task, v in raw.items()}
    row_meta = {task: v[1] for task, v in raw.items()}
    for task, rows in eval_rows.items():
        print(f"[data] {task}: eval {len(rows)} NEW_EVAL_ROW rows", flush=True)

    fit_sets: dict[str, list[te.PairRow]] = {}
    for task in args.tasks:
        fit_task = NEW_ROW_TASKS[task]["fit_task"]
        if fit_task not in fit_sets:
            fit_records, _ = te.task_data(fit_task, manifest_rows, 0)
            fit_sets[fit_task] = fit_records
            print(f"[fit] {fit_task}: {len(fit_records)} TRAIN rows (same-endpoint probe surface)", flush=True)

    if args.smoke_limit:
        eval_rows = {t: r[: args.smoke_limit] for t, r in eval_rows.items()}

    assert not args.output_dir.exists(), f"output already exists: {args.output_dir}"
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "cuda_provenance.json").write_text(json.dumps({
        "cuda_available": True,
        "device": torch.cuda.get_device_name(device),
        "device_index": args.physical_gpu_index,
        "torch": torch.__version__,
        "amendment": "benchmark_v2_generalist_rows_amendment_v3_new_rows.md",
    }, indent=1))

    for model_key in args.models:
        all_sequences: set[str] = set()
        for task in args.tasks:
            for record in eval_rows[task]:
                all_sequences.add(record.source)
                all_sequences.add(record.candidate)
            fit_task = NEW_ROW_TASKS[task]["fit_task"]
            for record in fit_sets[fit_task]:
                all_sequences.add(record.source)
                all_sequences.add(record.candidate)
        ordered_sequences = sorted(all_sequences)
        print(f"[embed] {model_key}: {len(ordered_sequences)} unique sequences", flush=True)

        embeddings: dict[str, torch.Tensor] = {}
        stats: dict = {"model_key": model_key}
        if model_key == "rnafm":
            te.embed_rnafm(ordered_sequences, embeddings, device, stats)
        elif model_key in gen.BATCH2_ADAPTERS:
            gen.embed_batch2(ordered_sequences, embeddings, device, stats, model_key)
        else:
            gen.embed_multimolecule(ordered_sequences, embeddings, device, stats, model_key)

        for task in args.tasks:
            fit_task = NEW_ROW_TASKS[task]["fit_task"]
            fit_records = fit_sets[fit_task]
            predict, probe_meta = te.fit_probe(
                fit_records, None, embeddings, device, "SOURCE_GROUP_EQUAL"
            )
            records = eval_rows[task]
            predictions = predict(records)
            observations = [
                observation_of(
                    r,
                    row_meta[task][r.record_id]["context"],
                    row_meta[task][r.record_id]["region"],
                    row_meta[task][r.record_id]["endpoint"],
                )
                for r in records
            ]
            metrics = ev.evaluate(observations, predictions, K)
            entry = {
                "mode": "FROZEN_ENCODER_LINEAR_PROBE_DELTA__NEW_EVAL_ROW",
                "probe_mode": "SAME_ENDPOINT_TASK_TRAIN_FIT (amendment v3)",
                "fit_task": fit_task,
                "fit_record_count": len(fit_records),
                "record_count": len(records),
                "task_macro_spearman": metrics.get("task_macro_spearman"),
                "within_source": metrics.get("source_macro_within_source_spearman"),
                "top_1": metrics.get("source_macro_top_1_accuracy"),
                "ndcg_at_10": metrics.get("source_macro_ndcg_at_k"),
                "overall_spearman": metrics.get("overall_numeric", {}).get("spearman"),
                "prediction_std": float(np.asarray(list(predictions.values()), dtype=float).std()),
                "metrics_full": metrics,
                "probe": probe_meta,
                "reporting_only": True,
            }
            out_task_dir = args.output_dir / f"{task}__{model_key}"
            out_task_dir.mkdir()
            (out_task_dir / "run_detail.json").write_text(json.dumps(entry, indent=1))
            print(f"[{model_key}] {task}: done n={len(records)}", flush=True)

    print("[done] amendment v3 run complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
