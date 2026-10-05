#!/usr/bin/env python3
"""Generalist-row evaluation for DeltaBench (benchmark v2 amendment v2, batches 0-1).

Adds new generalist rows (RiNALMo micro/mega/giga, ERNIE-RNA; Orthrus/CodonFM/CaLM/
AIDO.RNA/LucaOne/mRNA-LM land in batch 2 once weights exist) under the **verbatim** frozen-Δ
probe caliber of the existing RNA-FM / UTR-LM rows.

Reuse (no caliber change):
  * `run_route2_frozen_delta_te_family_v1.py` imported as a library: task_data / fit_probe /
    evaluate_task / _length_budgeted_batches / PairRecord / TASKS / constants.
  * `run_route2_frozen_delta_full_coverage_v1.py` imported for the MPRAU variant pair-mean
    caliber (W-ladder adjudication caliber) and its canonical-row loader.
New code surface (the only additions):
  * two task specs injected for polyA (GSE269595) and MPRAU (ENCSR854RUF), mode IN_STUDY_PROBE
    (same as the MRL row: probe fit on TRAIN, epoch selection on VALIDATION);
  * `embed_multimolecule()` - official backbone, RnaTokenizer with T->U, fp32 mean over
    non-special tokens, 1000-nt chunking with length-weighted mean, length-sorted batches.

Port-validation gate (amendment v2 section 4.1): `--port-validation` runs RNA-FM on
gse114002_mrl through THIS file and hard-fails unless the value matches the archived row
0.13693329073357266 within 1e-5. Nothing else may run before that gate passes.

Discipline: frozen backbones (no gradient), VALIDATION only, protected TEST reads = 0,
append-only outputs, one output directory per invocation, no threshold edits.

Usage:
  python run_route2_frozen_delta_generalist_v2.py --physical-gpu-index 3 \
      --models rinalmo_micro [--tasks gse114002_mrl gse269595_polya encsr854ruf_mprau] \
      [--port-validation] [--smoke-limit N] [--output-dir PATH]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
HF_MODELS = Path("/mnt/cunyuliu/hf_home/models")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TE_PATH = REPO_ROOT / "scripts/route_a_v3/run_route2_frozen_delta_te_family_v1.py"
FC_PATH = REPO_ROOT / "scripts/route_a_v3/run_route2_frozen_delta_full_coverage_v1.py"
te = _load_module("route2_generalist_te_lib", TE_PATH)
fc = _load_module("route2_generalist_fc_lib", FC_PATH)

# ---- task specs injected for the three headline rows (MRL already exists in te.TASKS) --------
te.TASKS["gse269595_polya"] = {
    "study": "GSE269595",
    "region": "3UTR",
    "endpoint": "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS",
    "mode": "IN_STUDY_PROBE",
}
te.TASKS["encsr854ruf_mprau"] = {
    "study": "ENCSR854RUF",
    "region": "3UTR",
    "endpoint": "MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE",
    "mode": "IN_STUDY_PROBE",
}
# batch 3: the remaining Development tasks, specs verbatim from
# run_route2_frozen_delta_full_coverage_v1.py P1_TASKS (half-life rows); the four P0 rows
# (gse200304_te / gse149487_te / gse149487_rna / gse186455) already live in te.TASKS.
te.TASKS["half_life_5utr"] = {
    "study": "GSE217518",
    "region": "5UTR",
    "endpoint": "RNA_HALF_LIFE_MINUTES",
    "mode": "IN_STUDY_PROBE",
}
te.TASKS["half_life_3utr"] = {
    "study": "GSE217518",
    "region": "3UTR",
    "endpoint": "RNA_HALF_LIFE_MINUTES",
    "mode": "IN_STUDY_PROBE",
}
BATCH3_TASKS = (
    "gse200304_te", "gse149487_te", "gse149487_rna", "gse186455",
    "half_life_5utr", "half_life_3utr",
)
DEFAULT_TASKS = ("gse114002_mrl", "gse269595_polya", "encsr854ruf_mprau")

# ---- new generalist backbones (weights already on disk; verified loadable 2026-09-28) --------
MODELS = {
    "rinalmo_micro": {
        "module": "rinalmo",
        "dir": HF_MODELS / "multimolecule--rinalmo-micro/snapshots/main",
        "hidden_size": 480,
        "expected_parameters": 33_482_412,
    },
    "rinalmo_mega": {
        "module": "rinalmo",
        "dir": HF_MODELS / "multimolecule--rinalmo-mega/snapshots/main",
        "hidden_size": 640,
        "expected_parameters": 148_045_430,
    },
    "rinalmo_giga": {
        "module": "rinalmo",
        "dir": HF_MODELS / "multimolecule--rinalmo-giga/snapshots/main",
        "hidden_size": 1280,
        "expected_parameters": 650_878_731,
    },
    "ernierna": {
        "module": "ernierna",
        "dir": HF_MODELS / "multimolecule--ernierna/snapshots/main",
        "hidden_size": 768,
        "expected_parameters": 85_669_728,
    },
}
# batch-2 bespoke adapters (7 models; built and smoke-tested 2026-10-03, journal 133).
# Their embed_fn(list[str]) -> (N, D) CUDA interface is bridged into this runner with the
# same length-sorted batching + 1000-nt chunk policy as the RNA-FM frozen row.
try:
    from batch2_generalist_adapters_v1 import REGISTRY as BATCH2_ADAPTERS
except ImportError:
    BATCH2_ADAPTERS = {}
BATCH2_CHOICES = tuple(BATCH2_ADAPTERS)
MODEL_CHOICES = ("rnafm",) + tuple(MODELS) + BATCH2_CHOICES
# batch-2 model_path is read from the adapter meta at build time (see main()).
# archived MRL frozen rows used by the port-validation gate (te.REFERENCE holds rnafm/utrlm)
MRL_ARCHIVED = {"rnafm": 0.13693329073357266}
PORT_VALIDATION_TOLERANCE = 1e-5

MULTIMOLECULE_MAX_SEQUENCES_PER_BATCH = 32
MULTIMOLECULE_BATCH_TOKEN_BUDGET = 8192
CHUNK_NUCLEOTIDES = 1000

ADAPTATION_TEMPLATE = {
    "tokenizer": "multimolecule RnaTokenizer, T->U, dynamic padding",
    "pooling": "mean over non-special tokens (fp32)",
    "chunk_policy": (
        "sequences > 1000 nt split into 1000-nt chunks, length-weighted mean of chunk "
        "embeddings (same policy as the RNA-FM frozen row)"
    ),
    "batching": "length-sorted, <=32 sequences, <=8192 tokens",
    "pooler_head": "unused (checkpoint pooler.dense.* reported MISSING by the loader; "
                   "embeddings are mean-pooled, so the pooler is never touched)",
    "pretrained_weights_tuned": False,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise te.FrozenDeltaError(message)


def embed_multimolecule(
    sequences: list[str],
    cache: dict[str, torch.Tensor],
    device,
    stats: dict,
    model_key: str,
) -> None:
    """Frozen embedding extraction for the multimolecule generalist backbones."""
    import multimolecule.models as mm_models

    spec = MODELS[model_key]
    model_dir = Path(spec["dir"])
    _require(model_dir.is_dir(), f"model directory is absent: {model_dir}")
    module = getattr(mm_models, spec["module"])
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = module.AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = module.AutoModel.from_pretrained(str(model_dir), local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    parameter_count = sum(p.numel() for p in model.parameters())
    _require(
        parameter_count == spec["expected_parameters"],
        f"{model_key} parameter count changed: {parameter_count} != {spec['expected_parameters']}",
    )
    stats["pretrained_parameter_count"] = parameter_count
    stats["hidden_size"] = int(model.config.hidden_size)
    _require(stats["hidden_size"] == spec["hidden_size"], "hidden size changed vs registry")

    missing = [sequence for sequence in sequences if sequence not in cache]
    if not missing:
        return
    chunks: list[tuple[str, str]] = []
    for sequence in sorted(set(missing)):
        for start in range(0, len(sequence), CHUNK_NUCLEOTIDES):
            chunks.append((sequence, sequence[start : start + CHUNK_NUCLEOTIDES]))
    sums: dict[str, torch.Tensor] = {}
    lengths: dict[str, int] = {}
    batch_count = 0
    with torch.no_grad():
        for batch in te._length_budgeted_batches(
            chunks, MULTIMOLECULE_MAX_SEQUENCES_PER_BATCH, MULTIMOLECULE_BATCH_TOKEN_BUDGET
        ):
            tokens = tokenizer(
                [chunk.replace("T", "U") for _sequence, chunk in batch],
                padding=True,
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            output = model(**tokens).last_hidden_state
            attention = tokens["attention_mask"].bool()
            special = torch.zeros_like(attention)
            special[:, 0] = True
            token_lengths = attention.sum(dim=1)
            special[torch.arange(len(batch), device=device), token_lengths - 1] = True
            keep = attention & ~special
            pooled = (
                (output * keep.unsqueeze(-1)).sum(dim=1)
                / keep.sum(dim=1, keepdim=True).clamp_min(1)
            )
            _require(
                pooled.is_cuda and torch.isfinite(pooled).all().item(),
                f"{model_key} embedding left CUDA or became nonfinite",
            )
            for (sequence, chunk), embedding in zip(batch, pooled):
                if len(sequence) <= CHUNK_NUCLEOTIDES:
                    cache[sequence] = embedding.detach()
                else:
                    weight = len(chunk)
                    sums[sequence] = sums.get(sequence, 0) + weight * embedding if sequence in sums else weight * embedding
                    lengths[sequence] = lengths.get(sequence, 0) + weight
            batch_count += 1
            if batch_count % 400 == 0:
                print(f"[{model_key}] embedding batches: {batch_count}", flush=True)
    for sequence, total in sums.items():
        cache[sequence] = total / lengths[sequence]
    stats["chunked_sequence_count"] = len(sums)
    del model
    torch.cuda.empty_cache()


def embed_batch2(
    sequences: list[str],
    cache: dict[str, torch.Tensor],
    device,
    stats: dict,
    model_key: str,
) -> None:
    """Bridge the batch-2 bespoke adapters into this runner.

    Applies the same chunk policy as the RNA-FM frozen row (1000-nt chunks,
    length-weighted mean) and the same length-sorted batching, then fills
    cache[sequence] with the fp32 CUDA embedding tensor.
    """
    # adapters: REGISTRY[key](device) -> (embed_fn, meta)
    embed_fn, meta = BATCH2_ADAPTERS[model_key](device)
    stats["pretrained_parameter_count"] = meta.get("pretrained_parameter_count")
    stats["input_adaptation"] = dict(meta)

    pending = [s for s in sequences if s not in cache]
    if not pending:
        return
    chunk_map: dict[str, list[str]] = {}
    for seq in pending:
        if len(seq) > CHUNK_NUCLEOTIDES:
            chunk_map[seq] = [seq[i:i + CHUNK_NUCLEOTIDES] for i in range(0, len(seq), CHUNK_NUCLEOTIDES)]
    flat = [c for chunks in chunk_map.values() for c in chunks]
    singles = [s for s in pending if s not in chunk_map]
    ordered = sorted(set(flat + singles), key=len, reverse=True)
    batch_count = 0
    i = 0
    while i < len(ordered):
        j = min(i + MULTIMOLECULE_MAX_SEQUENCES_PER_BATCH, len(ordered))
        while j > i + 1 and sum(len(s) for s in ordered[i:j]) > MULTIMOLECULE_BATCH_TOKEN_BUDGET:
            j -= 1
        batch = ordered[i:j]
        i = j
        batch_count += 1
        emb = embed_fn(batch)
        _require(
            emb.shape[0] == len(batch) and emb.is_cuda and bool(torch.isfinite(emb).all()),
            f"{model_key} embedding left CUDA or became nonfinite",
        )
        emb = emb.float()
        for sequence, embedding in zip(batch, emb):
            if sequence not in chunk_map:
                cache[sequence] = embedding.detach()
        if batch_count % 25 == 0:
            print(f"[{model_key}] embedding batches: {batch_count}", flush=True)
    # resolve chunked sequences by length-weighted mean of their cached chunk embeddings
    for seq, chunks in chunk_map.items():
        total_weight = sum(len(c) for c in chunks)
        acc = None
        for c in chunks:
            e = cache.get(c)
            if e is None:
                e = embed_fn([c])[0]
            if acc is None:
                acc = e.detach().clone() * len(c)
            else:
                acc += e.detach() * len(c)
        cache[seq] = (acc / total_weight).float()
        for c in chunks:
            cache.pop(c, None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS),
                        choices=sorted(te.TASKS))  # half-life rows injected above
    parser.add_argument("--models", nargs="+", default=["rinalmo_micro"],
                        choices=list(MODEL_CHOICES))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-limit", type=int, default=0)
    parser.add_argument("--port-validation", action="store_true",
                        help="run RNA-FM on MRL and hard-assert the archived row value")
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
    if args.port_validation:
        _require(args.models == ["rnafm"], "port validation runs rnafm only")
        _require(args.tasks == ["gse114002_mrl"], "port validation runs gse114002_mrl only")

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
    for model_key in args.models:
        all_sequences: set[str] = set()
        for task in args.tasks:
            fit_records, eval_records = data[task]
            for record in fit_records + eval_records:
                all_sequences.add(record.source)
                all_sequences.add(record.candidate)
        ordered_sequences = sorted(all_sequences)
        lengths = [len(sequence) for sequence in ordered_sequences]
        _b2_path = None
        if model_key in BATCH2_ADAPTERS:
            _b2_result = BATCH2_ADAPTERS[model_key](device)
            _b2_path = _b2_result[1].get("model_path") if isinstance(_b2_result, tuple) else None
            del _b2_result
        stats: dict = {
            "model_path": str(
                te.RNAFM_MODEL_PATH if model_key == "rnafm"
                else (_b2_path if _b2_path is not None else MODELS[model_key]["dir"])
            ),
            "unique_sequence_count": len(ordered_sequences),
            "sequence_length_min_median_max": [
                min(lengths), sorted(lengths)[len(lengths) // 2], max(lengths)
            ],
        }
        embeddings: dict[str, torch.Tensor] = {}
        print(f"[embed] {model_key}: {len(ordered_sequences)} unique sequences", flush=True)
        if model_key == "rnafm":
            te.embed_rnafm(ordered_sequences, embeddings, device, stats)
            stats["input_adaptation"] = dict(ADAPTATION_TEMPLATE)
        elif model_key in BATCH2_ADAPTERS:
            embed_batch2(ordered_sequences, embeddings, device, stats, model_key)
        else:
            embed_multimolecule(ordered_sequences, embeddings, device, stats, model_key)
            stats["input_adaptation"] = dict(ADAPTATION_TEMPLATE)
            stats["input_adaptation"]["official_source"] = {
                "rinalmo_micro": "multimolecule/rinalmo-micro (official mirror)",
                "rinalmo_mega": "multimolecule/rinalmo-mega (official mirror)",
                "rinalmo_giga": "multimolecule/rinalmo-giga (official mirror)",
                "ernierna": "multimolecule/ernierna",
            }[model_key]
        model_stats[model_key] = stats

        for task in args.tasks:
            spec = te.TASKS[task]
            fit_records, eval_records = data[task]
            selection_records = eval_records if spec["mode"] == "IN_STUDY_PROBE" else None
            weighting = "SOURCE_GROUP_EQUAL" if spec["mode"] == "IN_STUDY_PROBE" else "STUDY_THEN_SOURCE_GROUP_EQUAL"
            predict, probe_meta = te.fit_probe(
                fit_records, selection_records, embeddings, device, weighting
            )
            predictions = predict(eval_records)
            metrics = te.evaluate_task(task, predictions, [r.record_id for r in eval_records])
            entry = {
                "mode": "FROZEN_ENCODER_LINEAR_PROBE_DELTA",
                "probe_mode": "IN_STUDY_PROBE_TRAIN_FIT_VALIDATION_EPOCH_SELECTION",
                "stratum": f"{spec['study']}|{spec['region']}|{spec['endpoint']}",
                "record_count": len(eval_records),
                "fit_record_count": len(fit_records),
                "task_macro_spearman": metrics.get("task_macro_spearman"),
                "within_source": metrics.get("source_macro_within_source_spearman"),
                "top_1": metrics.get("source_macro_top_1_accuracy"),
                "ndcg_at_10": metrics.get("source_macro_ndcg_at_k"),
                "source_group_count": metrics.get("source_group_count"),
                "rankable_source_group_count": metrics.get("rankable_source_group_count"),
                "prediction_std": float(np.asarray(list(predictions.values()), dtype=float).std()),
                "metrics_full": metrics,
                "probe": probe_meta,
            }
            if task == "encsr854ruf_mprau":
                canonical_rows = fc.load_canonical_rows(spec["study"], set(predictions))
                pair_mean = fc.mprau_pair_mean(canonical_rows, predictions)
                if pair_mean.get("variant_count"):
                    entry["mprau_pair_mean"] = pair_mean
                else:
                    entry["mprau_pair_mean"] = {
                        "variant_count": 0,
                        "note": "no multi-context variants resolved - pair-mean caliber undefined",
                    }
            run_dir = args.output_dir / f"{task}__{model_key}"
            _require(not run_dir.exists(), f"run dir already exists: {run_dir}")
            run_dir.mkdir()
            with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
                for record_id in sorted(predictions):
                    handle.write(json.dumps({
                        "canonical_record_id": record_id,
                        "predicted_direction_normalized_delta": predictions[record_id],
                    }) + "\n")
            with (run_dir / "run_detail.json").open("w", encoding="utf-8") as handle:
                json.dump(entry, handle, indent=1, sort_keys=True)
            results[task][model_key] = {
                key: entry[key]
                for key in (
                    "mode", "probe_mode", "record_count", "fit_record_count",
                    "task_macro_spearman", "within_source", "top_1", "ndcg_at_10",
                    "source_group_count", "rankable_source_group_count", "prediction_std",
                )
                if key in entry
            }
            results[task][model_key]["selected_epoch"] = probe_meta["selected_epoch"]
            if "mprau_pair_mean" in entry:
                results[task][model_key]["mprau_pair_mean"] = entry["mprau_pair_mean"]
            print(
                f"[result] {task} x {model_key}: spearman {entry['task_macro_spearman']:.6f}"
                f" | top-1 {entry['top_1']} | ndcg@10 {entry['ndcg_at_10']}"
                f" | epoch {probe_meta['selected_epoch']}",
                flush=True,
            )
            if args.port_validation:
                expected = MRL_ARCHIVED[model_key]
                delta = abs(entry["task_macro_spearman"] - expected)
                print(
                    f"[port-validation] gse114002_mrl x {model_key}: {entry['task_macro_spearman']:.17g} "
                    f"vs archived {expected:.17g} (|delta| {delta:.2e})",
                    flush=True,
                )
                _require(
                    delta <= PORT_VALIDATION_TOLERANCE,
                    f"PORT VALIDATION FAILED: |delta| {delta:.3e} > {PORT_VALIDATION_TOLERANCE:g}",
                )
                print("[port-validation] PASS", flush=True)
        del embeddings
        torch.cuda.empty_cache()

    summary = {
        "schema_version": "route_a_v3_frozen_delta_generalist.v1",
        "mode": "FROZEN_ENCODER_LINEAR_PROBE_DELTA",
        "amendment": "docs/paper/benchmark_v2_generalist_rows_amendment_v2.md",
        "port_validation": bool(args.port_validation),
        "smoke_limit": args.smoke_limit or None,
        "record_scope": "DEVELOPMENT_VALIDATION_ONLY",
        "protected_reads": 0,
        "k": te.K,
        "probe_protocol": {
            "functional_form": "linear(emb(candidate) - emb(source)); delta_hat = y_cand - y_src",
            "hyperparameters": {
                "epochs": te.PROBE_EPOCHS,
                "learning_rate": te.PROBE_LEARNING_RATE,
                "weight_decay": te.PROBE_WEIGHT_DECAY,
                "seed": te.SEED,
            },
            "pretrained_weights_tuned": False,
            "source": "verbatim reuse of run_route2_frozen_delta_te_family_v1.py (RNA-FM/UTR-LM frozen rows)",
        },
        "models": model_stats,
        "reference": {
            "rnafm_mrl_archived": MRL_ARCHIVED["rnafm"],
            "internal_targets": {
                "gse114002_mrl": 0.1192,
                "gse269595_polya": 0.7308,
                "encsr854ruf_mprau": 0.0248,
            },
            "critic_v5": {
                "gse114002_mrl": 0.1354,
                "gse269595_polya": 0.8219,
                "encsr854ruf_mprau_pair_mean": 0.1025449348211772,
            },
        },
        "results": results,
        "cuda_provenance": provenance,
    }
    output_path = args.output_dir / "frozen_delta_generalist_results.json"
    _require(not output_path.exists(), f"output already exists: {output_path}")
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1, sort_keys=True)
    print(f"wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())