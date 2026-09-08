#!/usr/bin/env python3
"""Baseline Task 6.5.6 (P1-2): LAMAR UTR3DegPred frozen zero-shot delta on
GSE217518 (HALF_LIFE stability row, external alternative to Saluki).

LAMAR (Genome Biol 2025): ESM2-style 150M RNA LM; UTR3DegPred = official
fine-tuned 3'UTR->half-life regressor (BEAS-2B, CN1 natural-3'UTR tiling
library). R3 audit PASS (r3_audit.json): zero overlap between GSE217518
VALIDATION and UTR3DegPred train/val/test; UTR5TEPred source-sequence overlap
(2/1291) registered with no label leakage and its head not loaded here.

Frozen protocol (baseline leaderboard v1): official fine-tuned checkpoint,
no further training. Score source and candidate, delta = candidate - source,
evaluate against direction_normalized_delta with the frozen Task-1 evaluator
(K=10). Native-truncation clause: LAMAR consumes 3'UTR segments; GSE217518
supplies 115bp UTR windows for both 5'UTR and 3'UTR regions — reported as-is.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

LAMAR_REPO = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/lamar")
WEIGHTS = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/lamar_weights"
    "/UTR3DegPred/saving_model/mammalian_4096/bs8_lr5e-5_wr0.05_16epochs_2/checkpoint-3180/model.safetensors"
)
EVAL_REPO = "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901"
MANIFEST = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/manifests/route2_development_frozen_v1/development_manifest.jsonl"
)
CANONICAL = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/canonical/GSE217518/v1/canonical_records.jsonl"
)
OUT_DIR = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/analysis_lamar_frozen_delta_20260909"
)
MODEL_MAX_LENGTH = 200
BATCH = 64

sys.path.insert(0, str(LAMAR_REPO))
sys.path.insert(0, EVAL_REPO + "/scripts/route_a_v3")

import importlib.util

_ev_spec = importlib.util.spec_from_file_location(
    "ev", EVAL_REPO + "/scripts/route_a_v3/evaluate_route2_prediction_v1.py"
)
ev = importlib.util.module_from_spec(_ev_spec)
_ev_spec.loader.exec_module(ev)

from transformers import AutoConfig, AutoTokenizer  # noqa: E402
from safetensors.torch import load_model  # noqa: E402

# transformers >= 4.48 moved these helpers to pytorch_utils — shim for LAMAR
import transformers.modeling_utils as _mu  # noqa: E402
import transformers.pytorch_utils as _pu  # noqa: E402
for _name in ("find_pruneable_heads_and_indices", "prune_linear_layer"):
    if not hasattr(_mu, _name) and hasattr(_pu, _name):
        setattr(_mu, _name, getattr(_pu, _name))

from LAMAR.sequence_classification_patch import EsmForSequenceClassification  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.physical_gpu_index}")

    validation_ids = set()
    with MANIFEST.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["study_unit_id"] == "GSE217518" and row["split"] == "VALIDATION":
                validation_ids.add(str(row["canonical_record_id"]))
    records = {}
    with CANONICAL.open() as handle:
        for line in handle:
            row = json.loads(line)
            rid = str(row.get("canonical_record_id"))
            if rid in validation_ids:
                records[rid] = row
    if set(records) != validation_ids:
        raise SystemExit("coverage mismatch against manifest")
    ids = sorted(records)

    tokenizer = AutoTokenizer.from_pretrained(
        LAMAR_REPO / "tokenizer/single_nucleotide", model_max_length=MODEL_MAX_LENGTH
    )
    vocab_is_rna = "U" in tokenizer.get_vocab()
    def prep(seq: str) -> str:
        s = str(seq).upper()
        return s.replace("T", "U") if vocab_is_rna else s

    config = AutoConfig.from_pretrained(
        LAMAR_REPO / "config/config_150M.json",
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        num_labels=1,
        token_dropout=False,
        positional_embedding_type="rotary",
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        num_hidden_layers=12,
    )
    model = EsmForSequenceClassification(
        config, head_type="Linear", freeze=False, kernel_sizes=[2, 3, 5], ocs=32
    )
    load_model(model, filename=str(WEIGHTS), strict=False)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model loaded: {n_params/1e6:.1f}M params; tokenizer RNA-vocab={vocab_is_rna}")

    sources = [prep(records[rid]["source_sequence"]) for rid in ids]
    candidates = [prep(records[rid]["candidate_sequence"]) for rid in ids]

    def score(seqs: list[str]) -> np.ndarray:
        out = []
        with torch.no_grad():
            for start in range(0, len(seqs), BATCH):
                enc = tokenizer(
                    seqs[start : start + BATCH], add_special_tokens=True,
                    padding=True, truncation=True, max_length=MODEL_MAX_LENGTH,
                    return_tensors="pt",
                )
                logits = model(
                    input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device),
                ).logits
                out.append(logits.float().squeeze(-1).cpu().numpy())
        return np.concatenate(out)

    source_scores = score(sources)
    candidate_scores = score(candidates)
    delta = candidate_scores - source_scores
    predictions = {rid: float(delta[i]) for i, rid in enumerate(ids)}

    observations = ev.load_observations([CANONICAL], validation_ids)
    metrics = ev.evaluate(observations, predictions, 10)

    region_counts = {}
    for rid in ids:
        region_counts[records[rid]["region"]] = region_counts.get(records[rid]["region"], 0) + 1

    # per-region spearman preview
    from scipy.stats import spearmanr
    per_region = {}
    for region in sorted(region_counts):
        sub = [(delta[i], float(records[rid]["direction_normalized_delta"]))
               for i, rid in enumerate(ids) if records[rid]["region"] == region]
        if len(sub) >= 3:
            a, b = zip(*sub)
            per_region[region] = float(spearmanr(a, b).statistic)

    report = {
        "schema_version": "route_a_v3_route2_lamar_frozen_delta_gse217518.v1",
        "mode": "FROZEN_ZERO_SHOT_DELTA",
        "model": "LAMAR UTR3DegPred official fine-tuned checkpoint-3180 (BEAS-2B half-life regressor)",
        "device": torch.cuda.get_device_name(device),
        "record_count": len(ids),
        "region_counts": region_counts,
        "metrics": metrics,
        "per_region_spearman_preview": per_region,
        "r3_audit": "PASS (see r3_audit.json)",
        "note": (
            "Official fine-tuned 3'UTR half-life head; delta = candidate - source; "
            "115bp UTR windows within the 200-token budget; both regions scored "
            "(5'UTR rows included though the head was trained on 3'UTR segments — "
            "native-domain caveat, reported as-is)."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "frozen_delta_results.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    with (OUT_DIR / "predictions.jsonl").open("w") as handle:
        for i, rid in enumerate(ids):
            handle.write(json.dumps({
                "canonical_record_id": rid,
                "prediction": float(delta[i]),
                "source_score": float(source_scores[i]),
                "candidate_score": float(candidate_scores[i]),
            }) + "\n")
    print(f"overall spearman {metrics.get('task_macro_spearman')}")
    print(f"per-region spearman: {per_region}")
    print(f"wrote {OUT_DIR}/frozen_delta_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
