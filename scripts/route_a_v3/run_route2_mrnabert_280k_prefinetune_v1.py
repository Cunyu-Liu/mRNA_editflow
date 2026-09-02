#!/usr/bin/env python3
"""Route A Stage 0c+1: 280K leakage audit + mRNABERT LoRA pre-finetuning (delta scorer).

Stage 0c (leakage hard gate): load the Sample-2019 280K random-library tables
(egfp_unmod replicates), audit near-duplicates (>= 48/50 identity, 3-block
seeding) against ALL benchmark VALIDATION/TEST source+candidate sequences,
exclude matches, write the audit JSON.

Stage 1: fine-tune mRNABERT (all 12 layers) with LoRA (rank 16) + masked mean
pool + linear head on the cleaned 280K (utr -> standardized rl). This is
supervised domain-library pre-finetuning (spec wording clause; NOT
self-supervised pretraining).

Eval: frozen-delta protocol on GSE114002 VALIDATION (delta = f(candidate) -
f(source), frozen Task-1 evaluator K=10) - directly comparable to
frozen-Optimus 0.3132 / frozen-FramePool 0.2956 / W0 0.1987.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_mrnabert_lora_v3 import LoRALinearV3  # noqa: E402

EVAL_REPO = "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901"
_ev_spec = importlib.util.spec_from_file_location(
    "ev", EVAL_REPO + "/scripts/route_a_v3/evaluate_route2_prediction_v1.py"
)
ev = importlib.util.module_from_spec(_ev_spec)
sys.modules["ev"] = ev
_ev_spec.loader.exec_module(ev)

LIB_DIR = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/sample280k")
MRNABERT_PATH = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/mrnabert_a1eb7df25804d23f08646e1cb996b234d7208a40")
MANIFEST = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/manifests/route2_development_frozen_v1/development_manifest.jsonl")
CANONICAL_GSE114002 = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/canonical/GSE114002/v1/canonical_records.private.jsonl")
OUT_DIR = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_route_a/280k_prefinetune_20260903")

LORA_RANK = 16
LORA_ALPHA = 32.0
LORA_DROPOUT = 0.05
BATCH = 128
EPOCHS = 2
LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 20260903


def load_library() -> dict[str, list[float]]:
    merged: dict[str, list[float]] = defaultdict(list)
    for name in ("GSM3130435_egfp_unmod_1.csv.gz", "GSM3130436_egfp_unmod_2.csv.gz"):
        with gzip.open(LIB_DIR / name, "rt") as handle:
            header = handle.readline().strip().split(",")
            utr_index = header.index("utr")
            rl_index = header.index("rl")
            for line in handle:
                fields = line.rstrip("\n").split(",")
                utr = fields[utr_index]
                rl = float(fields[rl_index])
                merged[utr].append(rl)
    return merged


def audit_leakage(library: dict[str, list[float]]) -> tuple[set[str], dict]:
    """3-block seeding: any 50-mer within 2 mismatches shares one exact block."""
    protected_sequences = set()
    for study in ("GSE114002",):
        with CANONICAL_GSE114002.open() as handle:
            for line in handle:
                row = json.loads(line)
                protected_sequences.add(row["source_sequence"])
                protected_sequences.add(row["candidate_sequence"])

    block_index: dict[str, set[str]] = defaultdict(set)
    for sequence in protected_sequences:
        for block in (sequence[:17], sequence[17:34], sequence[34:]):
            block_index[block].add(sequence)

    flagged: set[str] = set()
    for utr in library:
        for block in (utr[:17], utr[17:34], utr[34:]):
            for candidate in block_index.get(block, ()):
                mismatches = sum(a != b for a, b in zip(utr, candidate))
                if mismatches <= 2:
                    flagged.add(utr)
                    break
            if utr in flagged:
                break

    report = {
        "schema_version": "route_a_v3_route2_280k_leakage_audit.v1",
        "library_sequence_count": len(library),
        "protected_sequence_count": len(protected_sequences),
        "protected_scope": "GSE114002 all-split source+candidate sequences",
        "near_duplicate_threshold": "<=2 mismatches on 50-mer (>=96% identity)",
        "flagged_sequence_count": len(flagged),
        "flagged_examples": sorted(flagged)[:10],
        "method": "3-block exact seeding (pigeonhole for <=2 mismatches)",
    }
    return flagged, report


def format_sequence(sequence: str) -> str:
    return " ".join(str(sequence).upper().replace("U", "T"))


def wrap_lora(model: nn.Module) -> int:
    wrapped = 0
    for layer in model.encoder.layer:
        for attr_path in (
            ("attention", "self", "Wqkv"),
            ("attention", "output", "dense"),
            ("intermediate", "dense") if hasattr(layer, "intermediate") else ("mlp", "gated_layers"),
            ("mlp", "wo"),
        ):
            try:
                parent = layer
                for step in attr_path[:-1]:
                    parent = getattr(parent, step)
                base = getattr(parent, attr_path[-1])
            except AttributeError:
                continue
            if isinstance(base, nn.Linear):
                setattr(parent, attr_path[-1], LoRALinearV3(base, rank=LORA_RANK, alpha=LORA_ALPHA, dropout=LORA_DROPOUT))
                wrapped += 1
    return wrapped


class MeanPoolRegressor(nn.Module):
    def __init__(self, base_model: nn.Module, width: int):
        super().__init__()
        self.base = base_model
        self.head = nn.Linear(width, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.base(input_ids=input_ids, attention_mask=attention_mask)[0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--skip-train", action="store_true", help="audit only")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - GPU required")
    device = torch.device(f"cuda:{args.physical_gpu_index}")
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    library = load_library()
    flagged, audit = audit_leakage(library)
    (OUT_DIR / "leakage_audit.json").write_text(json.dumps(audit, indent=1, sort_keys=True))
    print(f"library={audit['library_sequence_count']} protected={audit['protected_sequence_count']} flagged={audit['flagged_sequence_count']}")
    if args.skip_train:
        return 0

    clean = [(utr, float(np.mean(values))) for utr, values in library.items() if utr not in flagged]
    print(f"clean library: {len(clean)} sequences")

    from transformers import AutoConfig, AutoModel, AutoTokenizer
    import os as _os
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(MRNABERT_PATH, local_files_only=True)
    model_config = AutoConfig.from_pretrained(MRNABERT_PATH, local_files_only=True, trust_remote_code=True)
    base = AutoModel.from_config(model_config, trust_remote_code=True, add_pooling_layer=False)
    modeling_module = sys.modules[base.__class__.__module__]
    modeling_module.flash_attn_qkvpacked_func = None  # official fallback backend
    checkpoint = torch.load(MRNABERT_PATH / "pytorch_model.bin", map_location="cpu", weights_only=False)
    base_state = {
        key.removeprefix("bert."): value
        for key, value in checkpoint.items()
        if key.startswith("bert.")
    }
    base.load_state_dict(base_state, strict=True)
    del checkpoint, base_state
    wrapped = wrap_lora(base)
    print(f"LoRA wrapped {wrapped} Linears (rank {LORA_RANK})")
    model = MeanPoolRegressor(base, base.config.hidden_size).to(device)

    sequences = [utr for utr, _ in clean]
    targets_np = np.asarray([rl for _, rl in clean], dtype=np.float64)
    mean, std = targets_np.mean(), targets_np.std()
    targets = torch.tensor((targets_np - mean) / std, dtype=torch.float32)

    encoded = tokenizer([format_sequence(s) for s in sequences], add_special_tokens=True, padding=True, truncation=False, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in trainable):,}")
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = (len(sequences) // BATCH + 1) * EPOCHS
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(step / max(total_steps, 1), 1.0))) if step > total_steps * 0.05 else step / max(total_steps * 0.05, 1),
    )

    order = torch.randperm(len(sequences))
    model.train()
    step = 0
    for epoch in range(EPOCHS):
        losses = []
        for start in range(0, len(order), BATCH):
            idx = order[start : start + BATCH]
            batch_ids = input_ids[idx].to(device)
            batch_mask = attention_mask[idx].to(device)
            batch_targets = targets[idx].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = model(batch_ids, batch_mask)
                loss = nn.functional.mse_loss(prediction.float(), batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(float(loss))
            step += 1
            if step % 200 == 0:
                print(f"epoch {epoch + 1} step {step}: mse {np.mean(losses[-200:]):.4f} lr {scheduler.get_last_lr()[0]:.2e}", flush=True)
        order = torch.randperm(len(sequences))

    torch.save({
        "schema_version": "route_a_v3_route2_mrnabert_280k_lora.v1",
        "model_state_dict": {k: v for k, v in model.state_dict().items()},
        "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,
        "target_mean": float(mean), "target_std": float(std),
        "seed": SEED, "epochs": EPOCHS,
        "library_clean_count": len(clean),
    }, OUT_DIR / "lora_scorer_checkpoint.pt")
    print("saved LoRA scorer checkpoint")

    # Frozen-delta evaluation on GSE114002 VALIDATION.
    model.eval()
    validation_ids = set()
    with MANIFEST.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["study_unit_id"] == "GSE114002" and row["split"] == "VALIDATION":
                validation_ids.add(str(row["canonical_record_id"]))
    records = {}
    with CANONICAL_GSE114002.open() as handle:
        for line in handle:
            row = json.loads(line)
            rid = str(row.get("canonical_record_id"))
            if rid in validation_ids:
                records[rid] = row
    ids = sorted(records)

    def score_batch(sequences: list[str]) -> np.ndarray:
        encoded_batch = tokenizer([format_sequence(s) for s in sequences], add_special_tokens=True, padding=True, truncation=False, return_tensors="pt")
        values = []
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for start in range(0, len(sequences), 256):
                ids_batch = encoded_batch["input_ids"][start : start + 256].to(device)
                mask_batch = encoded_batch["attention_mask"][start : start + 256].to(device)
                values.append(model(ids_batch, mask_batch).float().cpu().numpy())
        return np.concatenate(values)

    source_scores = score_batch([records[rid]["source_sequence"] for rid in ids])
    candidate_scores = score_batch([records[rid]["candidate_sequence"] for rid in ids])
    delta = candidate_scores - source_scores
    predictions = {rid: float(delta[i]) for i, rid in enumerate(ids)}
    observations = ev.load_observations([CANONICAL_GSE114002], validation_ids)
    metrics = ev.evaluate(observations, predictions, 10)

    report = {
        "schema_version": "route_a_v3_route2_mrnabert_280k_frozen_delta.v1",
        "mode": "ROUTE_A_STEP1_FROZEN_DELTA",
        "trainable_parameter_count": sum(p.numel() for p in trainable),
        "lora_wrapped_linears": wrapped,
        "library_clean_count": len(clean),
        "metrics": {
            "task_macro_spearman": metrics.get("task_macro_spearman"),
            "top_1": metrics.get("source_macro_top_1_accuracy"),
            "ndcg_at_10": metrics.get("source_macro_ndcg_at_k"),
        },
        "reference": {
            "frozen_optimus_280k": 0.3132,
            "frozen_framepool_280k": 0.2956,
            "w0_critic_from_scratch": 0.1987,
            "critic_v5": 0.1354,
            "optimus_arch_from_scratch": 0.0984,
        },
    }
    (OUT_DIR / "frozen_delta_results.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    with (OUT_DIR / "predictions.jsonl").open("w") as handle:
        for rid in ids:
            handle.write(json.dumps({"canonical_record_id": rid, "predicted_direction_normalized_delta": predictions[rid]}) + "\n")
    print(json.dumps(report["metrics"], indent=1))
    print("wrote", OUT_DIR / "frozen_delta_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
