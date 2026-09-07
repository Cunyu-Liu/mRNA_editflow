#!/usr/bin/env python3
"""HydraRNA frozen zero-shot delta on GSE217518 (RNA half-life, 5'UTR + 3'UTR).

D6=P1 approved row (SPECS_BASELINE_LEADERBOARD 6.3.4): HydraRNA, a full-length
RNA language model (Li et al. Genome Biol 2025), as a generalist frozen row on
the stability task.

Provenance audit (R3, per the UTR-STCNet leak finding): HydraRNA is pre-trained
on non-coding + protein-coding RNA transcriptomes (general RNA LM, same class
as UTR-LM / RNA-FM). It is NOT trained on the GSE217518 MPRA variant library.
No benchmark-study overlap -> the row is eligible.

Protocol: identical to the UTR-LM / RNA-FM generalist rows on this task
(te-family IN_STUDY_PROBE): frozen embeddings + linear probe fit on task TRAIN,
epoch selection on task VALIDATION (source-group-weighted MSE), evaluate with
the frozen Task-1 evaluator K=10. Only the base LM (HydraRNA_model.pt) is the
primary row; --model v2 optionally adds the ncRNA-extended LM as a second row.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
HYDRARNA_ROOT = MNT / "external_model_assets/hydrarna"
# bundled fairseq repo root: the package lives at fairseq/fairseq/
if str(HYDRARNA_ROOT / "fairseq") not in sys.path:
    sys.path.insert(0, str(HYDRARNA_ROOT / "fairseq"))

TE_FAMILY_SCRIPT = REPO_ROOT / "scripts/route_a_v3/run_route2_frozen_delta_te_family_v1.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


te = _load_module("route2_frozen_delta_te_family", TE_FAMILY_SCRIPT)

MODEL_PATHS = {
    "base": HYDRARNA_ROOT / "weights/models/HydraRNA_model.pt",
    "v2": HYDRARNA_ROOT / "weights/models/HydraRNA_model_V2.pt",
}
DICT_DIR = HYDRARNA_ROOT / "dict"
DEFAULT_OUTPUT = MNT / "experiments/analysis_hydrarna_frozen_delta_gse217518_20260907"

# GSE217518 tasks (registered into te.TASKS by the full-coverage wrapper pattern)
te.TASKS.update({
    "half_life_5utr": {
        "study": "GSE217518", "region": "5UTR",
        "endpoint": "RNA_HALF_LIFE_MINUTES", "mode": "IN_STUDY_PROBE",
    },
    "half_life_3utr": {
        "study": "GSE217518", "region": "3UTR",
        "endpoint": "RNA_HALF_LIFE_MINUTES", "mode": "IN_STUDY_PROBE",
    },
})

MAX_SEQUENCES_PER_BATCH = 32
BATCH_TOKEN_BUDGET = 16384


def _load_hydrarna(model_path: Path, device):
    from fairseq import checkpoint_utils, data, options, tasks

    parser = options.get_generation_parser(default_task="masked_lm_span")
    args = options.parse_args_and_arch(parser, [str(DICT_DIR)])
    task = tasks.setup_task(args)
    models, _model_args = checkpoint_utils.load_model_ensemble([str(model_path)], task=task)
    model = models[0]
    model.to(device)
    model.half()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, task


def embed_hydrarna(sequences, cache, device, stats, model, task):
    stats["pretrained_parameter_count"] = sum(
        p.numel() for p in model.parameters()
    )
    missing = sorted(set(s for s in sequences if s not in cache))
    if not missing:
        return
    chunk_size = 10240
    chunks: list[tuple[str, str]] = []
    for sequence in missing:
        seq = str(sequence)
        for i in range(chunk_size, len(seq) + chunk_size, chunk_size):
            chunks.append((sequence, seq[i - chunk_size:i]))
    sums: dict[str, torch.Tensor] = {}
    lengths: dict[str, int] = {}
    batch_count = 0
    with torch.no_grad():
        for start in range(0, len(chunks), MAX_SEQUENCES_PER_BATCH):
            batch = chunks[start:start + MAX_SEQUENCES_PER_BATCH]
            samples = []
            for _key, chunk in batch:
                token_seq = "<s> " + " ".join(list(chunk))
                tokens = task.source_dictionary.encode_line(token_seq, add_if_not_exist=False)
                samples.append({"id": -1, "source": tokens, "target": tokens})
            collated = data.monolingual_dataset.collate(
                samples, pad_idx=task.source_dictionary.pad(), eos_idx=task.source_dictionary.eos()
            )
            src_tokens = collated["net_input"]["src_tokens"].to(device)
            out = model.encoder.extract_features(src_tokens=src_tokens)
            # out[0]: [B, N+2, 1024]; mean over non-special tokens
            pooled = out[0][:, 1:-1, :].mean(dim=1).float()
            if not (pooled.is_cuda and torch.isfinite(pooled).all().item()):
                raise RuntimeError("HydraRNA embedding left CUDA or became nonfinite")
            for (sequence, chunk), emb in zip(batch, pooled):
                if len(sequence) <= chunk_size:
                    cache[sequence] = emb.detach()
                else:
                    weight = len(chunk)
                    if sequence in sums:
                        sums[sequence] = sums[sequence] + weight * emb
                    else:
                        sums[sequence] = weight * emb
                    lengths[sequence] = lengths.get(sequence, 0) + weight
            batch_count += 1
            if batch_count % 20 == 0:
                print(f"[hydrarna] batches done: {batch_count}/{len(chunks)}", flush=True)
    for sequence, total in sums.items():
        cache[sequence] = total / lengths[sequence]
    stats["chunked_sequence_count"] = len(sums)
    del model
    torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--model", default="base", choices=sorted(MODEL_PATHS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke-limit", type=int, default=0, help="restrict fit/eval rows (smoke)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - GPU required")
    device = torch.device(f"cuda:{args.physical_gpu_index}")
    _require = te._require
    _require(args.output_dir.is_dir() or not args.output_dir.exists(),
             f"output exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_PATHS[args.model]
    _require(model_path.is_file(), f"HydraRNA checkpoint absent: {model_path}")

    manifest_rows = te.load_manifest_rows()
    data = {task: te.task_data(task, manifest_rows, args.smoke_limit)
            for task in ("half_life_5utr", "half_life_3utr")}

    all_sequences: set[str] = set()
    for task in data:
        fit_records, eval_records = data[task]
        for record in fit_records + eval_records:
            all_sequences.add(record.source)
            all_sequences.add(record.candidate)
    ordered = sorted(all_sequences)
    lengths = [len(s) for s in ordered]
    stats = {
        "model_path": str(model_path),
        "unique_sequence_count": len(ordered),
        "sequence_length_min_median_max": [
            min(lengths), sorted(lengths)[len(lengths) // 2], max(lengths)],
    }
    print(f"[hydrarna:{args.model}] loading {model_path.name} + embedding {len(ordered)} seqs", flush=True)
    model, task = _load_hydrarna(model_path, device)
    embeddings: dict[str, torch.Tensor] = {}
    embed_hydrarna(ordered, embeddings, device, stats, model, task)
    stats["input_adaptation"] = {
        "checkpoint": model_path.name,
        "tokenizer": "fairseq masked_lm_span dictionary, <s> + space-joined chars",
        "embedding": "mean of extract_features over non-special tokens [B,1024]",
        "length_limit": "supports up to 10K nt; chunk policy 10240",
        "batching": "<=32 sequences per collate",
        "p1_length_note": "GSE217518 max sequence length 164 nt - no chunking triggered",
        "provenance_audit": (
            "general RNA LM pre-trained on ncRNA+pcRNA transcriptomes; "
            "NOT trained on the GSE217518 MPRA variant library (R3 clean)"
        ),
    }

    results = {}
    for task_key in ("half_life_5utr", "half_life_3utr"):
        fit_records, eval_records = data[task_key]
        selection_records = eval_records  # IN_STUDY_PROBE
        predict, probe_meta = te.fit_probe(
            fit_records, selection_records, embeddings, device, "SOURCE_GROUP_EQUAL"
        )
        predictions = predict(eval_records)
        metrics = te.evaluate_task(
            task_key, predictions, [r.record_id for r in eval_records]
        )
        rec = {
            "model": f"hydrarna_{args.model}",
            "task": task_key,
            "split": "VALIDATION",
            "n_fit": len(fit_records),
            "n_eval": len(eval_records),
            "task_macro_spearman": metrics.get("task_macro_spearman"),
            "top_1": metrics.get("source_macro_top_1_accuracy"),
            "ndcg_at_10": metrics.get("source_macro_ndcg_at_k"),
            "probe": probe_meta,
            "leakage_audit": stats["input_adaptation"]["provenance_audit"],
            "cpu_fallback_used": False,
        }
        results[task_key] = rec
        print(f"{task_key}: spearman {rec['task_macro_spearman']} | top-1 {rec['top_1']} | "
              f"ndcg {rec['ndcg_at_10']}", flush=True)

    out = {
        "schema_version": "route_a_v3_route2_hydrarna_frozen_delta_gse217518.v1",
        "model": f"hydrarna_{args.model}",
        "protocol": "FROZEN_ZERO_SHOT_DELTA_IN_STUDY_PROBE (UTR-LM/RNA-FM generalist caliber)",
        "stats": stats,
        "results": results,
    }
    (args.output_dir / f"frozen_delta_results_{args.model}.json").write_text(
        json.dumps(out, indent=1, sort_keys=True))
    print("wrote", args.output_dir / f"frozen_delta_results_{args.model}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
