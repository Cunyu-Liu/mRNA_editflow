#!/usr/bin/env python3
"""Task D6 (P0+P1): full-coverage frozen-delta evaluation of RNA-FM / UTR-LM on
every Development task lacking an external modern-LM row, consolidated into one
2-model x 8-task view (polyA / MPRAU / HALF_LIFE are the new generalist rows).

Protocol reuse (no new caliber):
- Pipeline: scripts/route_a_v3/run_route2_frozen_delta_te_family_v1.py, imported
  as a library and extended with four generalist task specs (polyA GSE269595 /
  MPRAU ENCSR854RUF / HALF_LIFE GSE217518 5'UTR + 3'UTR). All four have in-study
  TRAIN splits -> IN_STUDY_PROBE: linear probe fit on task TRAIN, epoch selection
  by task-VALIDATION source-group-weighted MSE - the HPO_VALIDATION_ONLY protocol
  behind the MRL frozen leaderboard rows (UTR-LM 0.1107 / RNA-FM 0.1369),
  port-validated end-to-end by the te-family smoke run on GSE114002
  (UTR-LM 0.1107267878538859 exact, RNA-FM |delta| 2.4e-6).
- P0 rows (GSE200304 / GSE149487 TE + RNA / GSE186455) already exist from
  analysis_frozen_delta_te_family_20260904 (same code path, cuda:5, port
  validated). This wrapper IMPORTS them (per-run dirs + result entries) instead
  of re-computing: the probe is seeded (20260816) and deterministic, so a re-run
  on another GPU reproduces identical numbers - re-running would only duplicate
  leaderboard rows (discipline: rows only added once).
- MPRAU primary judgment metric: variant pair-mean rho - the W-ladder
  adjudication caliber (run_route2_saluki_frozen_delta_mprau_v1.py): variants =
  record id before ':context:', keep >= 2 contexts, per-variant means of
  direction_normalized_delta vs predicted delta, Spearman across variants
  (n = 2,008), unpaired bootstrap CI (2,000 iters, seed 20260816).
- Evaluation: frozen Task-1 evaluator, VALIDATION only, K=10; per-task Spearman +
  top-1 + NDCG@10 (null where every source group in the stratum is
  single-candidate). Protected pool reads = 0.

Input adaptation (unchanged from the te-family pipeline, declared per model in
frozen_delta_results.json): RNA-FM T->U tokenizer, fp32 mean-pool over
non-special tokens, >1000 nt chunking policy (not triggered here: P1 task max
sequence length is 164 nt - far under the 1024-token limit, so no truncation),
length-sorted batches <=32 sequences / <=8192 tokens; UTR-LM BOS token layer-6
representation (rotary positions, no hard length limit), batches <=128
sequences / <=16384 tokens.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
TE_FAMILY_SCRIPT = REPO_ROOT / "scripts/route_a_v3/run_route2_frozen_delta_te_family_v1.py"
TE_FAMILY_OUTPUT = MNT / "experiments/analysis_frozen_delta_te_family_20260904"
DEFAULT_OUTPUT = MNT / "experiments/analysis_frozen_delta_full_coverage_20260904"
MPRAU_BOOTSTRAP_ITERATIONS = 2000
TASK_ORDER = (
    "gse200304_te",
    "gse149487_te",
    "gse149487_rna",
    "gse186455",
    "polya_gse269595",
    "mprau_encsr854ruf",
    "half_life_5utr",
    "half_life_3utr",
)

P1_TASKS = {
    "polya_gse269595": {
        "study": "GSE269595",
        "region": "3UTR",
        "endpoint": "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS",
        "mode": "IN_STUDY_PROBE",
    },
    "mprau_encsr854ruf": {
        "study": "ENCSR854RUF",
        "region": "3UTR",
        "endpoint": "MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE",
        "mode": "IN_STUDY_PROBE",
    },
    "half_life_5utr": {
        "study": "GSE217518",
        "region": "5UTR",
        "endpoint": "RNA_HALF_LIFE_MINUTES",
        "mode": "IN_STUDY_PROBE",
    },
    "half_life_3utr": {
        "study": "GSE217518",
        "region": "3UTR",
        "endpoint": "RNA_HALF_LIFE_MINUTES",
        "mode": "IN_STUDY_PROBE",
    },
}
P0_TASKS = ("gse200304_te", "gse149487_te", "gse149487_rna", "gse186455")

# References for the aligned interpretation table (leaderboard sections 1.2-1.5 caliber).
REFERENCE = {
    "critic_v5_spearman": {
        "gse200304_te": 0.0579,
        "gse149487_te": 0.1953,
        "gse149487_rna": 0.0500,
        "gse186455": 0.0639,
        "polya_gse269595": 0.8219,
        "mprau_encsr854ruf": 0.0732,  # record-level; pair-mean caliber below is primary
        "half_life_5utr": 0.0607,
        "half_life_3utr": 0.0456,
    },
    "critic_v5_mprau_pair_mean_spearman": 0.1025449348211772,
    "internal_target_global_scaled_spearman": {
        "gse200304_te": -0.0266,
        "gse149487_te": 0.1747,
        "gse149487_rna": 0.2230,
        "gse186455": -0.0052,
        "polya_gse269595": 0.7308,
        "mprau_encsr854ruf": 0.0248,
        "half_life_5utr": -0.0522,
        "half_life_3utr": -0.0029,
    },
    "external_frozen_references": {
        "polya_gse269595": {"aparent_frozen_delta_spearman": 0.7343},
        "mprau_encsr854ruf": {"saluki_frozen_weak_control_pair_mean_spearman": 0.1205},
        "half_life_5utr": {"saluki_frozen_spearman": 0.0193},
        "half_life_3utr": {"saluki_frozen_spearman": 0.0985},
    },
    "mrl_frozen_row_spearman": {
        "utrlm": 0.1107267878538859,
        "rnafm": 0.13693329073357266,
    },
    "source": (
        "TASK6_leaderboard_freeze_20260903.md sections 1.2-1.5; "
        "analysis_task1_alignment_20260902/task1_internal_controls_per_task.json; "
        "xeditcritic_v5/v5_screen_seed_20260907_.../v5_full/run_summary.json "
        "(final_validation per-task spearman); "
        "analysis_saluki_frozen_mprau_20260903/ and "
        "analysis_saluki_frozen_gse217518_20260903/ (Saluki rows); "
        "runs/development_hpo/utrlm_lr1e3_wd1e4_replay_gpu5_v1 and "
        "external_lr1e3_wd1e4_replay_gpu5_v1 (MRL frozen rows)"
    ),
}

EXTRA_CALIBER_DECLARATIONS = [
    "P1 generalist rows reuse the te-family pipeline verbatim (frozen official "
    "encoder + seeded linear probe); the probe fit uses the task's own TRAIN "
    "split with epoch selection on task VALIDATION - the exact information "
    "boundary of the MRL frozen leaderboard rows.",
    "MPRAU primary judgment metric is the variant pair-mean rho (W-ladder "
    "adjudication caliber: variants = record id before ':context:', >=2 contexts, "
    "per-variant means, Spearman across 2,008 variants; unpaired bootstrap CI, "
    "2,000 iterations, seed 20260816) - directly comparable to the V5 reference "
    "0.1025 and the Saluki weak-control row 0.1205.",
    "P0 rows are imported from analysis_frozen_delta_te_family_20260904 (same "
    "code path, cuda:5, port-validated against the MRL frozen rows) instead of "
    "being re-computed: the pipeline is seeded and deterministic, so a re-run "
    "would reproduce identical numbers and only duplicate leaderboard rows.",
    "HALF_LIFE rows are expected to land near zero (ICC split-half ~ 0, label "
    "noise dominated): the external modern-LM rows upgrade the "
    "'physically unlearnable' attribution from inference to an empirically "
    "closed loop; they do not participate in any 'good at everything' claim.",
    "Interpretation verdict rule (presentation only, predeclared): "
    "EXTERNAL_WINS / EXTERNAL_LOSES when |rho - reference| >= 0.01, else TIE; "
    "HALF_LIFE rows additionally flag |rho| < 0.1 as consistent with the "
    "physically-unlearnable claim.",
]


class FrozenDeltaError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FrozenDeltaError(message)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


te = _load_module("route2_frozen_delta_te_family", TE_FAMILY_SCRIPT)
te.TASKS.update(P1_TASKS)
ALL_TASK_CHOICES = sorted(te.TASKS)


def load_canonical_rows(study: str, record_ids) -> dict[str, dict]:
    want = set(record_ids)
    rows: dict[str, dict] = {}
    path = te.canonical_path_for(study)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            record_id = str(row["canonical_record_id"])
            if record_id in want:
                rows[record_id] = row
    _require(set(rows) == want, f"canonical coverage mismatch for {study}")
    return rows


def mprau_pair_mean(records: dict[str, dict], predictions: dict[str, float]) -> dict:
    """Variant pair-mean rho (W-ladder adjudication caliber; see module docstring)."""
    by_variant: dict[str, list[str]] = defaultdict(list)
    for record_id in predictions:
        if record_id.startswith("ENCSR854RUF:"):
            by_variant[record_id.split(":context:")[0]].append(record_id)
    targets: list[float] = []
    preds: list[float] = []
    for variant in sorted(by_variant):
        rids = by_variant[variant]
        if len(rids) >= 2:
            targets.append(
                float(np.mean([float(records[rid]["direction_normalized_delta"]) for rid in rids]))
            )
            preds.append(float(np.mean([predictions[rid] for rid in rids])))
    target_array = np.asarray(targets)
    pred_array = np.asarray(preds)
    rho = float(spearmanr(target_array, pred_array).statistic)
    rng = np.random.default_rng(te.SEED)
    n = len(target_array)
    boots = []
    for _ in range(MPRAU_BOOTSTRAP_ITERATIONS):
        idx = rng.integers(0, n, n)
        value = spearmanr(target_array[idx], pred_array[idx]).statistic
        if np.isfinite(value):
            boots.append(float(value))
    boots_array = np.asarray(boots)
    return {
        "variant_count": int(n),
        "pair_mean_spearman": rho,
        "bootstrap_ci_95": [
            float(np.percentile(boots_array, 2.5)),
            float(np.percentile(boots_array, 97.5)),
        ],
        "bootstrap_iterations": int(boots_array.shape[0]),
        "bootstrap_seed": int(te.SEED),
        "v5_reference_pair_mean_spearman": REFERENCE["critic_v5_mprau_pair_mean_spearman"],
        "caliber": (
            "variants = record id before ':context:', >=2 contexts, per-variant means "
            "(W-ladder adjudication caliber, run_route2_saluki_frozen_delta_mprau_v1.py)"
        ),
    }


def _verdict(value: float | None, reference: float | None) -> str | None:
    if value is None or reference is None:
        return None
    delta = value - reference
    if abs(delta) < 0.01:
        return "TIE"
    return "EXTERNAL_WINS" if delta > 0 else "EXTERNAL_LOSES"


def build_interpretation(results: dict) -> dict:
    table: dict[str, dict] = {}
    for task in TASK_ORDER:
        if task not in results or not results[task]:
            continue
        rows = {}
        for model in ("rnafm", "utrlm"):
            entry = results[task].get(model)
            if entry is None:
                continue
            if task == "mprau_encsr854ruf":
                metric = "variant_pair_mean_spearman"
                value = entry.get("mprau_pair_mean", {}).get("pair_mean_spearman")
                v5 = REFERENCE["critic_v5_mprau_pair_mean_spearman"]
            else:
                metric = "task_macro_spearman"
                value = entry.get("task_macro_spearman")
                v5 = REFERENCE["critic_v5_spearman"].get(task)
            global_scaled = REFERENCE["internal_target_global_scaled_spearman"].get(task)
            row = {
                "metric": metric,
                "value": value,
                "critic_v5_reference": v5,
                "delta_vs_critic_v5": (value - v5) if (value is not None and v5 is not None) else None,
                "verdict_vs_critic_v5": _verdict(value, v5),
                "internal_global_scaled_reference": global_scaled,
                "delta_vs_global_scaled": (
                    (value - global_scaled)
                    if (value is not None and global_scaled is not None)
                    else None
                ),
                "verdict_vs_global_scaled": _verdict(value, global_scaled),
                "external_frozen_references": REFERENCE["external_frozen_references"].get(task, {}),
            }
            if task in ("half_life_5utr", "half_life_3utr") and value is not None:
                row["physically_unlearnable_consistent"] = bool(abs(value) < 0.1)
            rows[model] = row
        table[task] = rows
    return table


def import_p0_runs(import_from: Path, output_dir: Path, results: dict, record_counts: dict) -> dict:
    """Import the te-family P0 rows (per-run dirs + result entries) verbatim."""
    source_summary_path = import_from / "frozen_delta_results.json"
    _require(source_summary_path.is_file(), f"import source missing: {source_summary_path}")
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    imported: dict[str, list[str]] = {}
    for task in P0_TASKS:
        if results.get(task):
            continue  # executed in this run - do not import
        _require(task in source_summary.get("results", {}), f"import source lacks task {task}")
        results[task] = {}
        for model in ("rnafm", "utrlm"):
            entry = dict(source_summary["results"][task][model])
            entry["imported_from"] = str(import_from)
            results[task][model] = entry
            record_counts[task] = entry["record_count"]
            source_run_dir = import_from / f"{task}__{model}"
            _require(source_run_dir.is_dir(), f"import source run dir missing: {source_run_dir}")
            target_run_dir = output_dir / f"{task}__{model}"
            _require(not target_run_dir.exists(), f"run dir already exists: {target_run_dir}")
            shutil.copytree(source_run_dir, target_run_dir)
        imported[task] = ["rnafm", "utrlm"]
    return {
        "source_directory": str(import_from),
        "source_schema_version": source_summary.get("schema_version"),
        "source_cuda_provenance": source_summary.get("cuda_provenance"),
        "source_mode": source_summary.get("mode"),
        "imported_tasks": imported,
        "port_validation_note": (
            "te-family smoke run reproduced the MRL frozen rows end-to-end: "
            "UTR-LM 0.1107267878538859 (exact), RNA-FM |delta| 2.4e-6 "
            "(analysis_frozen_delta_te_family_20260904_smoke/port_validation_gse114002)"
        ),
        "re_run_equivalence": (
            "probe seed 20260816, deterministic pipeline; re-running on another GPU "
            "reproduces identical numbers, so rows are imported not duplicated"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--tasks", nargs="+", default=list(P1_TASKS), choices=ALL_TASK_CHOICES)
    parser.add_argument("--models", nargs="+", default=["rnafm", "utrlm"], choices=["rnafm", "utrlm"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--smoke-limit",
        type=int,
        default=0,
        help="cap each task to N eval rows / 4N fit rows for fast chain validation",
    )
    parser.add_argument(
        "--import-from",
        type=str,
        default=str(TE_FAMILY_OUTPUT),
        help="directory of the te-family run whose P0 rows are imported ('none' disables)",
    )
    args = parser.parse_args()

    _require(
        not os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES remapping is forbidden for physical-device provenance",
    )
    _require(torch.cuda.is_available(), "CUDA unavailable - GPU required")
    device = torch.device(f"cuda:{args.physical_gpu_index}")
    _require(0 <= device.index < torch.cuda.device_count(), "physical GPU index is unavailable")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    provenance = {
        "device": str(device),
        "physical_gpu_index": args.physical_gpu_index,
        "cuda_device_name": properties.name,
        "cuda_total_memory_mb": properties.total_memory / (1024 ** 2),
        "cuda_device_uuid": str(properties.uuid),
    }

    manifest_rows = te.load_manifest_rows()
    data = {task: te.task_data(task, manifest_rows, args.smoke_limit) for task in args.tasks}
    for task in args.tasks:
        fit_records, eval_records = data[task]
        print(
            f"[data] {task}: fit {len(fit_records)} rows | eval {len(eval_records)} rows | "
            f"mode {te.TASKS[task]['mode']}",
            flush=True,
        )

    _require(not args.output_dir.exists(), f"output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    results: dict[str, dict] = {task: {} for task in args.tasks}
    model_stats: dict[str, dict] = {}
    mprau_pair_mean_summary: dict[str, dict] = {}
    for model_key in args.models:
        all_sequences: set[str] = set()
        for task in args.tasks:
            fit_records, eval_records = data[task]
            for record in fit_records + eval_records:
                all_sequences.add(record.source)
                all_sequences.add(record.candidate)
        ordered_sequences = sorted(all_sequences)
        lengths = [len(sequence) for sequence in ordered_sequences]
        stats: dict = {
            "model_path": str(te.RNAFM_MODEL_PATH if model_key == "rnafm" else te.UTRLM_CHECKPOINT),
            "unique_sequence_count": len(ordered_sequences),
            "sequence_length_min_median_max": [
                min(lengths),
                sorted(lengths)[len(lengths) // 2],
                max(lengths),
            ],
        }
        embeddings: dict[str, torch.Tensor] = {}
        print(f"[embed] {model_key}: {len(ordered_sequences)} unique sequences", flush=True)
        if model_key == "rnafm":
            te.embed_rnafm(ordered_sequences, embeddings, device, stats)
        else:
            model, alphabet = te.utrlm_lib.load_official_encoder(
                te.UTRLM_ASSET_ROOT, te.UTRLM_CHECKPOINT, device
            )
            te.embed_utrlm(ordered_sequences, embeddings, device, model, alphabet, stats)
            del model
            torch.cuda.empty_cache()
        stats["input_adaptation"] = (
            {
                "tokenizer": "multimolecule RnaTokenizer, T->U, dynamic padding",
                "pooling": "mean over non-special tokens (fp32)",
                "length_limit": "max_position_embeddings 1026 / model_max_length 1024",
                "chunk_policy": (
                    "sequences > 1000 nt split into 1000-nt chunks, "
                    "length-weighted mean of chunk embeddings "
                    "(build_route2_rnafm_feature_cache_v1 policy)"
                ),
                "batching": "length-sorted, <=32 sequences, <=8192 tokens",
                "p1_length_note": (
                    "P1 task max sequence length 164 nt - chunking/truncation never triggered"
                ),
            }
            if model_key == "rnafm"
            else {
                "checkpoint": te.UTRLM_CHECKPOINT.name,
                "official_git_revision": "b77b589bf182eb9de6a1a5024fa09d44294d94fc",
                "embedding": "BOS ([cls]) token representation, layer 6",
                "position_embeddings": "rotary - no hard length limit",
                "batching": (
                    "length-sorted, <=128 sequences, <=16384 tokens "
                    "(memory adaptation; results identical to count batching)"
                ),
                "p1_length_note": "P1 task max sequence length 164 nt - no length constraint reached",
            }
        )
        model_stats[model_key] = stats
        for task in args.tasks:
            spec = te.TASKS[task]
            fit_records, eval_records = data[task]
            if spec["mode"] == "IN_STUDY_PROBE":
                selection_records = eval_records
                weighting = "SOURCE_GROUP_EQUAL"
            else:
                selection_records = None
                weighting = "STUDY_THEN_SOURCE_GROUP_EQUAL"
            predict, probe_meta = te.fit_probe(
                fit_records, selection_records, embeddings, device, weighting
            )
            predictions = predict(eval_records)
            metrics = te.evaluate_task(task, predictions, [r.record_id for r in eval_records])
            pair_mean = None
            if task == "mprau_encsr854ruf":
                canonical_rows = load_canonical_rows(
                    spec["study"], [r.record_id for r in eval_records]
                )
                pair_mean = mprau_pair_mean(canonical_rows, predictions)
                mprau_pair_mean_summary.setdefault(task, {})[model_key] = pair_mean
                print(
                    f"[pair-mean] {task} x {model_key}: rho {pair_mean['pair_mean_spearman']:.4f} "
                    f"CI {pair_mean['bootstrap_ci_95']} | n variants {pair_mean['variant_count']}",
                    flush=True,
                )
            run_dir = args.output_dir / f"{task}__{model_key}"
            _require(not run_dir.exists(), f"run dir already exists: {run_dir}")
            run_dir.mkdir()
            with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
                for record_id in sorted(predictions):
                    handle.write(
                        json.dumps(
                            {
                                "canonical_record_id": record_id,
                                "predicted_direction_normalized_delta": predictions[record_id],
                            }
                        )
                        + "\n"
                    )
            prediction_values = np.asarray(list(predictions.values()), dtype=float)
            entry = {
                "mode": "FROZEN_ENCODER_LINEAR_PROBE_DELTA",
                "probe_mode": (
                    "IN_STUDY_PROBE_TRAIN_FIT_VALIDATION_EPOCH_SELECTION"
                    if spec["mode"] == "IN_STUDY_PROBE"
                    else "LOSO_PROBE_POOLED_TRAIN_FINAL_EPOCH"
                ),
                "stratum": f"{spec['study']}|{spec['region']}|{spec['endpoint']}",
                "record_count": len(eval_records),
                "fit_record_count": len(fit_records),
                "task_macro_spearman": metrics.get("task_macro_spearman"),
                "within_source": metrics.get("source_macro_within_source_spearman"),
                "top_1": metrics.get("source_macro_top_1_accuracy"),
                "ndcg_at_10": metrics.get("source_macro_ndcg_at_k"),
                "source_group_count": metrics.get("source_group_count"),
                "rankable_source_group_count": metrics.get("rankable_source_group_count"),
                "prediction_std": float(prediction_values.std()) if len(prediction_values) else None,
                "metrics_full": metrics,
                "probe": probe_meta,
            }
            if pair_mean is not None:
                entry["mprau_pair_mean"] = pair_mean
            if metrics.get("rankable_source_group_count") == 0:
                entry["decision_metric_note"] = (
                    "top-1/NDCG@10 undefined: every source group in this stratum is "
                    "single-candidate"
                )
            with (run_dir / "run_detail.json").open("w", encoding="utf-8") as handle:
                json.dump(entry, handle, indent=1, sort_keys=True)
            summary_entry = {
                key: entry[key]
                for key in (
                    "mode",
                    "probe_mode",
                    "record_count",
                    "fit_record_count",
                    "task_macro_spearman",
                    "within_source",
                    "top_1",
                    "ndcg_at_10",
                    "source_group_count",
                    "rankable_source_group_count",
                    "prediction_std",
                    "decision_metric_note",
                )
                if key in entry
            }
            summary_entry["selected_epoch"] = probe_meta["selected_epoch"]
            if pair_mean is not None:
                summary_entry["mprau_pair_mean"] = pair_mean
            results[task][model_key] = summary_entry
            print(
                f"[result] {task} x {model_key}: spearman "
                f"{entry['task_macro_spearman'] if entry['task_macro_spearman'] is None else round(entry['task_macro_spearman'], 4)}"
                f" | top-1 {entry['top_1'] if entry['top_1'] is None else round(entry['top_1'], 4)}"
                f" | ndcg@10 {entry['ndcg_at_10'] if entry['ndcg_at_10'] is None else round(entry['ndcg_at_10'], 4)}"
                f" | epoch {probe_meta['selected_epoch']}",
                flush=True,
            )
        del embeddings
        torch.cuda.empty_cache()

    record_counts = {
        task: results[task][next(iter(results[task]))]["record_count"]
        for task in args.tasks
        if results[task]
    }
    imported_runs = None
    if args.import_from.lower() != "none":
        imported_runs = import_p0_runs(
            Path(args.import_from), args.output_dir, results, record_counts
        )

    interpretation = build_interpretation(results)
    summary = {
        "schema_version": "route_a_v3_route2_frozen_delta_full_coverage.v1",
        "mode": "FROZEN_ENCODER_LINEAR_PROBE_DELTA",
        "smoke_limit": args.smoke_limit or None,
        "record_scope": "DEVELOPMENT_VALIDATION_ONLY",
        "protected_reads": 0,
        "k": te.K,
        "executed_tasks": list(args.tasks),
        "caliber_declarations": list(te.CALIBER_DECLARATIONS) + EXTRA_CALIBER_DECLARATIONS,
        "probe_protocol": {
            "functional_form": (
                "linear(emb(candidate) - emb(source)); the linear readout scores each "
                "sequence absolutely and delta_hat = y_cand - y_src (bias cancels)"
            ),
            "hyperparameters": {
                "epochs": te.PROBE_EPOCHS,
                "learning_rate": te.PROBE_LEARNING_RATE,
                "weight_decay": te.PROBE_WEIGHT_DECAY,
                "seed": te.SEED,
            },
            "pretrained_weights_tuned": False,
            "source": (
                "protocol of runs/development_hpo/utrlm_lr1e3_wd1e4_replay_gpu5_v1 and "
                "external_lr1e3_wd1e4_replay_gpu5_v1 (UTR-LM / RNA-FM MRL frozen rows)"
            ),
        },
        "models": model_stats,
        "record_count": record_counts,
        "reference": REFERENCE,
        "results": results,
        "mprau_pair_mean": mprau_pair_mean_summary,
        "interpretation": interpretation,
        "imported_runs": imported_runs,
        "cuda_provenance": provenance,
    }
    output_path = args.output_dir / "frozen_delta_results.json"
    _require(not output_path.exists(), f"output already exists: {output_path}")
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1, sort_keys=True)
    print(f"wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
