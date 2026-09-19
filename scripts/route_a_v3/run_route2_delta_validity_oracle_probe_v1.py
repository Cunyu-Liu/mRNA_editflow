#!/usr/bin/env python3
"""Task 1.0.2: oracle delta-h probe control arm (delta validity decisive experiment).

Preregistration: docs/paper/delta_validity_oracle_prereg_v1.md (commit a2a2919a,
written BEFORE any computation). Decision rules (priority order): (c) NO_SIGNAL
oracle_rho <= 0.10; else (a) HEAD/EQUIVALENT if oracle - frozen_delta <= +0.05;
else (b) REPRESENTATION_SIGNAL. Reference values are the archived frozen-delta
rows of the same (model family, task).

Design: identical frozen encoder + probe pipeline as the existing frozen-delta
LM rows (run_route2_frozen_delta_te_family_v1.py imported as a library), with
the probe supervision made EXPLICIT as delta form: features =
h(cand) - h(src), target = direction_normalized_delta (delta_y). Code audit
(prereg section 1.1): the archived rows' implementation already IS delta-form
regression (docstrings claiming absolute-label training are inconsistent with
the code); the two forms are mathematically equivalent for a linear readout,
so seed-20260816 must reproduce the archived rows (MRL: UTR-LM exact
0.1107267878538859, RNA-FM 0.13693731732817227; port-validation asserted).

Tasks: GSE114002 MRL (730 VALIDATION) + GSE269595 polyA (2,628 VALIDATION);
models: UTR-LM (BOS layer-6) + RNA-FM (T->U mean-pool), unchanged input
adaptation. 3 seeds: 20260816 / 20260902 / 20260919; primary readout = mean
oracle rho (per-seed values + range registered). VALIDATION-only evaluation
with the frozen Task-1 evaluator (K=10). Protected TEST reads = 0.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
TE_FAMILY_SCRIPT = REPO_ROOT / "scripts/route_a_v3/run_route2_frozen_delta_te_family_v1.py"

ORACLE_TASKS = {
    "gse114002_mrl": {
        "study": "GSE114002",
        "region": "5UTR",
        "endpoint": "MEAN_RIBOSOME_LOAD",
        "mode": "IN_STUDY_PROBE",
    },
    "polya_gse269595": {
        "study": "GSE269595",
        "region": "3UTR",
        "endpoint": "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS",
        "mode": "IN_STUDY_PROBE",
    },
}

FROZEN_DELTA_REFERENCE = {
    "gse114002_mrl": {
        "utrlm": 0.1107267878538859,
        "rnafm": 0.13693731732817227,
    },
    "polya_gse269595": {
        "utrlm": 0.7490152634646079,
        "rnafm": 0.7114084807476264,
    },
}

REFERENCE_SOURCE = {
    "gse114002_mrl": (
        "runs/development_hpo/utrlm_lr1e3_wd1e4_replay_gpu5_v1 and "
        "runs/development_hpo/external_lr1e3_wd1e4_replay_gpu5_v1 "
        "validation_evaluation.json (task_macro_spearman)"
    ),
    "polya_gse269595": (
        "experiments/analysis_frozen_delta_full_coverage_20260904/"
        "frozen_delta_results.json (task_macro_spearman)"
    ),
}

PORT_VALIDATION_REFERENCE = {
    "gse114002_mrl": {
        "utrlm": 0.1107267878538859,
        "rnafm": 0.13693731732817227,
    },
}

NO_SIGNAL_THRESHOLD = 0.10
GAIN_THRESHOLD = 0.05
SEEDS = (20260816, 20260902, 20260919)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _verdict(oracle_rho: float, frozen_delta_rho: float) -> tuple[str, dict]:
    delta = oracle_rho - frozen_delta_rho
    if oracle_rho <= NO_SIGNAL_THRESHOLD:
        branch = "NO_SIGNAL"
        rationale = (
            f"oracle rho {oracle_rho:.4f} <= {NO_SIGNAL_THRESHOLD} (internal-target "
            "control level): representation delta carries no readable delta signal"
        )
    elif delta > GAIN_THRESHOLD:
        branch = "REPRESENTATION_SIGNAL"
        rationale = (
            f"oracle rho {oracle_rho:.4f} > {NO_SIGNAL_THRESHOLD} and gain "
            f"{delta:+.4f} > +{GAIN_THRESHOLD}: representation contains delta "
            "information the archived absolute-supervision row did not realize"
        )
    else:
        branch = "HEAD/EQUIVALENT"
        rationale = (
            f"oracle rho {oracle_rho:.4f} > {NO_SIGNAL_THRESHOLD} and gain "
            f"{delta:+.4f} <= +{GAIN_THRESHOLD} (bootstrap CI half-width order): "
            "delta supervision adds no gain - bottleneck in representation or "
            "arms equivalent (expected under the prereg 1.1 audit finding)"
        )
    return branch, {
        "oracle_rho": oracle_rho,
        "frozen_delta_rho_reference": frozen_delta_rho,
        "delta_gain": delta,
        "branch": branch,
        "rationale": rationale,
    }


def fit_probe_seeded(
    te,
    fit_records,
    selection_records,
    embeddings,
    device,
    weighting,
    seed,
):
    """te.fit_probe with the seed parameterized for multi-seed runs."""
    original_seed = te.SEED
    try:
        te.SEED = seed
        predict, meta = te.fit_probe(
            fit_records, selection_records, embeddings, device, weighting
        )
    finally:
        te.SEED = original_seed
    meta = dict(meta)
    meta["probe_seed"] = seed
    return predict, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--models", nargs="+", default=["rnafm", "utrlm"], choices=["rnafm", "utrlm"])
    parser.add_argument("--output-dir", type=Path, default=MNT / "experiments/analysis_delta_validity_v1/oracle_probe")
    parser.add_argument("--smoke-limit", type=int, default=0)
    args = parser.parse_args()

    import os
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise SystemExit("CUDA_VISIBLE_DEVICES remapping is forbidden for physical-device provenance")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - GPU required")
    device = torch.device(f"cuda:{args.physical_gpu_index}")
    if not (0 <= device.index < torch.cuda.device_count()):
        raise SystemExit("physical GPU index is unavailable")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    provenance = {
        "device": str(device),
        "physical_gpu_index": args.physical_gpu_index,
        "cuda_device_name": properties.name,
        "cuda_total_memory_mb": properties.total_memory / (1024 ** 2),
        "cuda_device_uuid": str(properties.uuid),
        "cpu_fallback": False,
    }

    te = _load_module("route2_frozen_delta_te_family", TE_FAMILY_SCRIPT)
    te.TASKS.update(ORACLE_TASKS)

    manifest_rows = te.load_manifest_rows()
    data = {task: te.task_data(task, manifest_rows, args.smoke_limit) for task in ORACLE_TASKS}
    for task, (fit_records, eval_records) in data.items():
        print(
            f"[data] {task}: fit {len(fit_records)} rows | eval {len(eval_records)} rows",
            flush=True,
        )

    if args.output_dir.exists():
        raise SystemExit(f"output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    results = {task: {} for task in ORACLE_TASKS}
    model_stats = {}
    for model_key in args.models:
        all_sequences = set()
        for fit_records, eval_records in data.values():
            for record in fit_records + eval_records:
                all_sequences.add(record.source)
                all_sequences.add(record.candidate)
        ordered_sequences = sorted(all_sequences)
        lengths = [len(sequence) for sequence in ordered_sequences]
        stats = {
            "model_path": str(te.RNAFM_MODEL_PATH if model_key == "rnafm" else te.UTRLM_CHECKPOINT),
            "unique_sequence_count": len(ordered_sequences),
            "sequence_length_min_median_max": [
                min(lengths),
                sorted(lengths)[len(lengths) // 2],
                max(lengths),
            ],
        }
        embeddings = {}
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
                    "length-weighted mean of chunk embeddings"
                ),
                "batching": "length-sorted, <=32 sequences, <=8192 tokens",
            }
            if model_key == "rnafm"
            else {
                "checkpoint": te.UTRLM_CHECKPOINT.name,
                "official_git_revision": "b77b589bf182eb9de6a1a5024fa09d44294d94fc",
                "embedding": "BOS ([cls]) token representation, layer 6",
                "position_embeddings": "rotary - no hard length limit",
                "batching": "length-sorted, <=128 sequences, <=16384 tokens",
            }
        )
        model_stats[model_key] = stats

        for task in ORACLE_TASKS:
            spec = te.TASKS[task]
            fit_records, eval_records = data[task]
            weighting = "SOURCE_GROUP_EQUAL"
            seed_rows = []
            seed_metrics = {}
            for seed in SEEDS:
                predict, probe_meta = fit_probe_seeded(
                    te, fit_records, eval_records, embeddings, device, weighting, seed
                )
                predictions = predict(eval_records)
                metrics = te.evaluate_task(task, predictions, [r.record_id for r in eval_records])
                rho = metrics.get("task_macro_spearman")
                seed_metrics[seed] = {
                    "task_macro_spearman": rho,
                    "within_source": metrics.get("source_macro_within_source_spearman"),
                    "top_1": metrics.get("source_macro_top_1_accuracy"),
                    "ndcg_at_10": metrics.get("source_macro_ndcg_at_k"),
                    "probe": {
                        key: probe_meta[key]
                        for key in (
                            "probe_parameter_count",
                            "probe_epochs",
                            "probe_learning_rate",
                            "probe_weight_decay",
                            "probe_seed",
                            "probe_weighting",
                            "epoch_selection",
                            "selected_epoch",
                        )
                    },
                }
                seed_rows.append(rho)
                print(
                    f"[seed {seed}] {task} x {model_key}: rho {rho:.6f}",
                    flush=True,
                )
                if task in PORT_VALIDATION_REFERENCE and seed == 20260816 and not args.smoke_limit:
                    expected = PORT_VALIDATION_REFERENCE[task][model_key]
                    deviation = abs(rho - expected)
                    print(
                        f"[port-validation] {task} x {model_key}: {rho:.9f} vs archived "
                        f"{expected:.9f} (|delta| {deviation:.2e})",
                        flush=True,
                    )
                    if deviation > 1e-4:
                        raise SystemExit(
                            f"port-validation failed: {task} x {model_key} seed 20260816 "
                            f"rho {rho} deviates from archived row {expected} by {deviation}"
                        )
            oracle_mean = float(np.mean(seed_rows))
            oracle_range = [float(min(seed_rows)), float(max(seed_rows))]
            reference = FROZEN_DELTA_REFERENCE[task][model_key]
            branch, verdict = _verdict(oracle_mean, reference)
            entry = {
                "mode": "ORACLE_DELTA_H_PROBE",
                "supervision_form": (
                    "features = h(cand) - h(src); target = direction_normalized_delta "
                    "(delta-form supervision, explicit; linear readout makes this "
                    "equivalent to the archived rows' implementation - prereg 1.1)"
                ),
                "stratum": f"{spec['study']}|{spec['region']}|{spec['endpoint']}",
                "fit_split": "TRAIN",
                "eval_split": "VALIDATION",
                "fit_record_count": len(fit_records),
                "record_count": len(eval_records),
                "seeds": list(SEEDS),
                "oracle_rho_per_seed": {str(seed): value for seed, value in zip(SEEDS, seed_rows)},
                "oracle_rho_mean": oracle_mean,
                "oracle_rho_range": oracle_range,
                "frozen_delta_reference": {
                    "rho": reference,
                    "source": REFERENCE_SOURCE[task],
                },
                "verdict": verdict,
                "seed_metrics": {str(seed): seed_metrics[seed] for seed in SEEDS},
            }
            run_dir = args.output_dir / f"{task}__{model_key}"
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
            with (run_dir / "run_detail.json").open("w", encoding="utf-8") as handle:
                json.dump(entry, handle, indent=1, sort_keys=True)
            compact = {key: entry[key] for key in (
                "mode",
                "supervision_form",
                "stratum",
                "fit_record_count",
                "record_count",
                "seeds",
                "oracle_rho_per_seed",
                "oracle_rho_mean",
                "oracle_rho_range",
                "verdict",
            )}
            compact["frozen_delta_reference_rho"] = reference
            results[task][model_key] = compact
            print(
                f"[result] {task} x {model_key}: oracle mean {oracle_mean:.6f} "
                f"(range {oracle_range[0]:.6f}..{oracle_range[1]:.6f}) vs frozen-delta "
                f"{reference:.6f} -> {branch}",
                flush=True,
            )
        del embeddings
        torch.cuda.empty_cache()

    summary = {
        "schema_version": "route_a_v3_route2_delta_validity_oracle_probe.v1",
        "mode": "ORACLE_DELTA_H_PROBE",
        "preregistration": (
            "docs/paper/delta_validity_oracle_prereg_v1.md (commit a2a2919a, "
            "written before computation)"
        ),
        "decision_rules": {
            "priority": "(c) first, then (a)/(b)",
            "no_signal": f"oracle rho <= {NO_SIGNAL_THRESHOLD}",
            "head_equivalent": f"gain <= +{GAIN_THRESHOLD}",
            "representation_signal": f"gain > +{GAIN_THRESHOLD}",
            "threshold_rationale": (
                "0.05 ~ project bootstrap CI half-width order; "
                "0.10 ~ internal-target control level"
            ),
        },
        "record_scope": "DEVELOPMENT_TRAIN_AND_VALIDATION_ONLY",
        "protected_reads": 0,
        "k": te.K,
        "seeds": list(SEEDS),
        "supervision_form_audit_note": (
            "prereg 1.1: the archived frozen-delta rows' code (train_probe / "
            "_train_multimolecule_rnafm_probe / te.fit_probe) already regresses "
            "delta-features on delta targets; the docstring 'absolute label y' "
            "description is inconsistent with the code. Linear readout => the two "
            "supervision forms are mathematically equivalent; seed 20260816 is "
            "port-validated against the archived MRL rows (asserted, tolerance 1e-4)."
        ),
        "probe_protocol": {
            "functional_form": "linear(h(cand) - h(src)) -> direction_normalized_delta",
            "z_score": "TRAIN-fit delta-feature statistics, std clamp 1e-6",
            "hyperparameters": {
                "epochs": te.PROBE_EPOCHS,
                "learning_rate": te.PROBE_LEARNING_RATE,
                "weight_decay": te.PROBE_WEIGHT_DECAY,
            },
            "epoch_selection": "TASK_VALIDATION_SOURCE_GROUP_WEIGHTED_MSE",
            "weighting": "SOURCE_GROUP_EQUAL",
            "pretrained_weights_tuned": False,
            "column_separation": (
                "oracle rows are TRAIN-fit probe rows; zero-shot frozen rows are "
                "reported separately and are not modified"
            ),
        },
        "models": model_stats,
        "results": results,
        "cuda_provenance": provenance,
    }
    output_path = args.output_dir / "results_oracle_probe.json"
    if output_path.exists():
        raise SystemExit(f"output already exists: {output_path}")
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1, sort_keys=True)
    print(f"wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
