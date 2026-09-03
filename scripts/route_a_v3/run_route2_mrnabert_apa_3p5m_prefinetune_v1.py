#!/usr/bin/env python3
"""Route A polyA arm: mRNABERT supervised pre-finetuning on the APARENT 3' UTR
MPRA library (GSE113849, ~3.5M variant-sequence APA measurements), 2026-09-03.

Mirrors the MRL arm (run_route2_mrnabert_280k_fullft_v2.py) exactly in protocol:
full-parameter fine-tuning, lr 2e-5, 6 epochs, batch 128, bf16, cosine+5% warmup.
Target: proximal PAS usage log2 odds = log2(p/(1-p)), p = proximal_count /
total_count_vs_distal (matches GSE269595 endpoint PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS).

Leakage hard gate: 3-block pigeonhole audit vs ALL GSE269595 splits
(source+candidate sequences, <=2 mismatches per 17bp block).

Preregistration:
- PRIMARY judgment = FINAL-EPOCH-6-FIXED frozen-delta Spearman on GSE269595
  VALIDATION (K=10, same evaluator as Task 1).
- Per-epoch frozen-delta metrics are DIAGNOSTIC ONLY (no peak-picking).
- Training target z-scored on TRAIN library only.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EVAL_REPO = "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901"
_ev_spec = importlib.util.spec_from_file_location(
    "ev", EVAL_REPO + "/scripts/route_a_v3/evaluate_route2_prediction_v1.py"
)
ev = importlib.util.module_from_spec(_ev_spec)
sys.modules["ev"] = ev
_ev_spec.loader.exec_module(ev)

LIB_GZ = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/aparent_apa_3p5m/GSE113849_data_isoforms.csv.gz")
MRNABERT_PATH = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/mrnabert_a1eb7df25804d23f08646e1cb996b234d7208a40")
MANIFEST = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/manifests/route2_development_frozen_v1/development_manifest.jsonl")
CANONICAL_GSE269595 = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/canonical/GSE269595/v1/canonical_records.private.jsonl")
OUT_DIR = Path(os.environ.get(
    "APA_PREFINETUNE_OUT_DIR",
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_route_a/apa_3p5m_prefinetune_20260903"))

BATCH = 128
LR = 2e-5
WEIGHT_DECAY = 1e-4
SEED = 20260903
MIN_TOTAL_COUNT = 10
P_CLIP = 1e-4


def load_library():
    rows = []
    with gzip.open(LIB_GZ, "rt") as handle:
        header = handle.readline().strip().split(",")
        seq_i = header.index("seq")
        prox_i = header.index("proximal_count")
        tot_i = header.index("total_count_vs_distal")
        for line in handle:
            fields = line.rstrip("\n").split(",")
            total = float(fields[tot_i])
            if total < MIN_TOTAL_COUNT:
                continue
            prox = float(fields[prox_i])
            p = min(max(prox / total, P_CLIP), 1.0 - P_CLIP)
            rows.append((fields[seq_i], float(np.log2(p / (1.0 - p)))))
    return rows


def audit_leakage(sequences):
    protected = set()
    with CANONICAL_GSE269595.open() as handle:
        for line in handle:
            row = json.loads(line)
            protected.add(row["source_sequence"])
            protected.add(row["candidate_sequence"])
    block_index = defaultdict(set)

    def blocks(s):
        n = len(s)
        return (s[:17], s[n // 2 : n // 2 + 17], s[-17:])

    for sequence in protected:
        for block in blocks(sequence):
            block_index[block].add(sequence)
    flagged = set()
    for utr in sequences:
        for block in blocks(utr):
            for cand in block_index.get(block, ()):
                if sum(a != b for a, b in zip(utr, cand)) <= 2:
                    flagged.add(utr)
                    break
            if utr in flagged:
                break
    return flagged, len(protected)


def format_sequence(sequence):
    return " ".join(str(sequence).upper().replace("U", "T"))


class MeanPoolRegressor(nn.Module):
    def __init__(self, base_model, width):
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
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    epochs = args.epochs
    seed = args.seed

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - GPU required")
    device = torch.device(f"cuda:{args.physical_gpu_index}")
    torch.manual_seed(seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    library = load_library()
    print(f"library rows (count>={MIN_TOTAL_COUNT}): {len(library)}", flush=True)
    flagged, protected_n = audit_leakage([s for s, _ in library])
    print(f"protected sequences: {protected_n} | flagged: {len(flagged)}", flush=True)
    json.dump(
        {"schema_version": "route_a_v3_route2_apa_3p5m_leakage_audit.v1",
         "library_rows": len(library), "flagged": len(flagged),
         "protected_sequences": protected_n, "rule": "3BLOCK_17BP_LE2_MISMATCH"},
        open(OUT_DIR / "leakage_audit.json", "w"), indent=1)
    clean = [(s, t) for s, t in library if s not in flagged]

    from transformers import AutoConfig, AutoModel, AutoTokenizer
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(MRNABERT_PATH, local_files_only=True)
    model_config = AutoConfig.from_pretrained(MRNABERT_PATH, local_files_only=True, trust_remote_code=True)
    base = AutoModel.from_config(model_config, trust_remote_code=True, add_pooling_layer=False)
    modeling_module = sys.modules[base.__class__.__module__]
    modeling_module.flash_attn_qkvpacked_func = None
    checkpoint = torch.load(MRNABERT_PATH / "pytorch_model.bin", map_location="cpu", weights_only=False)
    base_state = {k.removeprefix("bert."): v for k, v in checkpoint.items() if k.startswith("bert.")}
    base.load_state_dict(base_state, strict=True)
    del checkpoint, base_state
    model = MeanPoolRegressor(base, base.config.hidden_size).to(device)

    sequences = [s for s, _ in clean]
    targets_np = np.asarray([t for _, t in clean], dtype=np.float64)
    mean, std = targets_np.mean(), targets_np.std()
    targets = torch.tensor((targets_np - mean) / std, dtype=torch.float32)
    print(f"library clean: {len(clean)} | target mean {mean:.3f} std {std:.3f}", flush=True)

    encoded = tokenizer([format_sequence(s) for s in sequences], add_special_tokens=True, padding=True, truncation=True, max_length=512, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    validation_ids = set()
    with MANIFEST.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["study_unit_id"] == "GSE269595" and row["split"] == "VALIDATION":
                validation_ids.add(str(row["canonical_record_id"]))
    records = {}
    with CANONICAL_GSE269595.open() as handle:
        for line in handle:
            row = json.loads(line)
            rid = str(row.get("canonical_record_id"))
            if rid in validation_ids:
                records[rid] = row
    eval_ids = sorted(records)
    observations = ev.load_observations([CANONICAL_GSE269595], validation_ids)

    def frozen_delta_metrics():
        model.eval()

        def score_batch(seqs):
            enc = tokenizer([format_sequence(s) for s in seqs], add_special_tokens=True, padding=True, truncation=True, max_length=512, return_tensors="pt")
            values = []
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for start in range(0, len(seqs), 256):
                    b_ids = enc["input_ids"][start:start + 256].to(device)
                    b_mask = enc["attention_mask"][start:start + 256].to(device)
                    values.append(model(b_ids, b_mask).float().cpu().numpy())
            return np.concatenate(values)

        source_scores = score_batch([records[rid]["source_sequence"] for rid in eval_ids])
        candidate_scores = score_batch([records[rid]["candidate_sequence"] for rid in eval_ids])
        delta = candidate_scores - source_scores
        predictions = {rid: float(delta[i]) for i, rid in enumerate(eval_ids)}
        metrics = ev.evaluate(observations, predictions, 10)
        model.train()
        return {
            "task_macro_spearman": metrics.get("task_macro_spearman"),
            "top_1": metrics.get("source_macro_top_1_accuracy"),
            "ndcg_at_10": metrics.get("source_macro_ndcg_at_k"),
        }, predictions

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters (FULL): {sum(p.numel() for p in trainable):,}", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = (len(sequences) // BATCH + 1) * epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(step / max(total_steps, 1), 1.0))) if step > total_steps * 0.05 else step / max(total_steps * 0.05, 1),
    )

    loss_log = (OUT_DIR / "training_losses.jsonl").open("w")
    epoch_metrics_file = (OUT_DIR / "epoch_frozen_delta_metrics.jsonl").open("w")
    order = torch.randperm(len(sequences))
    model.train()
    step = 0
    final_metrics, final_predictions = None, None
    for epoch in range(epochs):
        losses = []
        for start in range(0, len(order), BATCH):
            idx = order[start:start + BATCH]
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
            if step % 50 == 0:
                loss_log.write(json.dumps({"step": step, "epoch": epoch + 1, "mse": float(np.mean(losses[-50:]))}) + "\n")
                loss_log.flush()
            if step % 1000 == 0:
                print(f"epoch {epoch + 1} step {step}: mse {np.mean(losses[-200:]):.4f} lr {scheduler.get_last_lr()[0]:.2e}", flush=True)
        order = torch.randperm(len(sequences))

        torch.save({
            "schema_version": "route_a_v3_route2_mrnabert_apa_3p5m_fullft.v1",
            "model_state_dict": {k: v for k, v in model.state_dict().items()},
            "target_mean": float(mean), "target_std": float(std),
            "seed": seed, "epochs_total": epochs, "epoch": epoch + 1,
            "library_clean_count": len(clean),
        }, OUT_DIR / f"apa_epoch_{epoch + 1}.pt")
        m, preds = frozen_delta_metrics()
        rec = {"epoch": epoch + 1, "primary": epoch + 1 == epochs, **m}
        epoch_metrics_file.write(json.dumps(rec) + "\n")
        epoch_metrics_file.flush()
        print(f"== epoch {epoch + 1} frozen-delta: {json.dumps(m)}", flush=True)
        if epoch + 1 == epochs:
            final_metrics, final_predictions = m, preds
    loss_log.close()
    epoch_metrics_file.close()

    report = {
        "schema_version": "route_a_v3_route2_mrnabert_apa_3p5m_fullft.v1",
        "mode": "ROUTE_A_POLYA_STEP1_FULLFT",
        "selection_rule": "FINAL_EPOCH_%d_FIXED" % epochs,
        "trainable_parameter_count": sum(p.numel() for p in trainable),
        "library_clean_count": len(clean),
        "min_total_count": MIN_TOTAL_COUNT,
        "epochs": epochs,
        "seed": seed,
        "metrics": final_metrics,
        "reference": {
            "critic_v5": 0.8219,
            "critic_v6": 0.8273,
            "aparent_adapter_trackB": 0.7343,
            "w0_polya": 0.8142,
        },
    }
    (OUT_DIR / "frozen_delta_results.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    with (OUT_DIR / "predictions.jsonl").open("w") as handle:
        for rid in eval_ids:
            handle.write(json.dumps({"canonical_record_id": rid, "predicted_direction_normalized_delta": final_predictions[rid]}) + "\n")
    print(json.dumps(report["metrics"], indent=1), flush=True)
    print("wrote", OUT_DIR / "frozen_delta_results.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
