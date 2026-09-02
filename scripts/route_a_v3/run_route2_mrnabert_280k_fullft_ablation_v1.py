#!/usr/bin/env python3
"""Route A Step-1 full-parameter ablation arm (preregistered, necessity
analysis §3.2: "全参臂作为消融（若时间允许）").

Identical to run_route2_mrnabert_280k_prefinetune_v1.py except: no LoRA -
ALL parameters of the 12-layer mRNABERT + masked-mean-pool + linear head are
trainable (full supervised fine-tuning on the same cleaned 280K library,
same 2-epoch budget, same batch/seed). Purpose: test whether the Step-1 gap
to frozen-Optimus (0.2470 vs 0.3132) is a LoRA-capacity artifact or an
architecture/inductive-bias difference.

Full-FT learning rate 2e-5 (standard BERT full fine-tuning scale; the LoRA
arm's 1e-4 is specific to low-rank adaptation). Eval: same frozen-delta
protocol on GSE114002 VALIDATION (K=10).
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
OUT_DIR = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_route_a/280k_fullft_ablation_20260903")

BATCH = 128
EPOCHS = 2
LR = 2e-5
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
                merged[fields[utr_index]].append(float(fields[rl_index]))
    return merged


def audit_leakage(library: dict[str, list[float]]) -> set[str]:
    """3-block seeding: identical to the Step-1 hard gate (deterministic)."""
    protected_sequences = set()
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
                if sum(a != b for a, b in zip(utr, candidate)) <= 2:
                    flagged.add(utr)
                    break
            if utr in flagged:
                break
    return flagged


def format_sequence(sequence: str) -> str:
    return " ".join(str(sequence).upper().replace("U", "T"))


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
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - GPU required")
    device = torch.device(f"cuda:{args.physical_gpu_index}")
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    library = load_library()
    flagged = audit_leakage(library)
    clean = [(utr, float(np.mean(values))) for utr, values in library.items() if utr not in flagged]
    print(f"library={len(library)} flagged={len(flagged)} clean={len(clean)}")

    from transformers import AutoConfig, AutoModel, AutoTokenizer
    import os as _os
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(MRNABERT_PATH, local_files_only=True)
    model_config = AutoConfig.from_pretrained(MRNABERT_PATH, local_files_only=True, trust_remote_code=True)
    base = AutoModel.from_config(model_config, trust_remote_code=True, add_pooling_layer=False)
    modeling_module = sys.modules[base.__class__.__module__]
    modeling_module.flash_attn_qkvpacked_func = None
    checkpoint = torch.load(MRNABERT_PATH / "pytorch_model.bin", map_location="cpu", weights_only=False)
    base_state = {
        key.removeprefix("bert."): value
        for key, value in checkpoint.items()
        if key.startswith("bert.")
    }
    base.load_state_dict(base_state, strict=True)
    del checkpoint, base_state
    model = MeanPoolRegressor(base, base.config.hidden_size).to(device)

    sequences = [utr for utr, _ in clean]
    targets_np = np.asarray([rl for _, rl in clean], dtype=np.float64)
    mean, std = targets_np.mean(), targets_np.std()
    targets = torch.tensor((targets_np - mean) / std, dtype=torch.float32)

    encoded = tokenizer([format_sequence(s) for s in sequences], add_special_tokens=True, padding=True, truncation=False, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters (FULL): {sum(p.numel() for p in trainable):,}")
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
        "schema_version": "route_a_v3_route2_mrnabert_280k_fullft.v1",
        "model_state_dict": {k: v for k, v in model.state_dict().items()},
        "target_mean": float(mean), "target_std": float(std),
        "seed": SEED, "epochs": EPOCHS,
        "library_clean_count": len(clean),
    }, OUT_DIR / "fullft_checkpoint.pt")
    print("saved full-FT checkpoint")

    # Frozen-delta evaluation on GSE114002 VALIDATION (same protocol as Step 1).
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
        "schema_version": "route_a_v3_route2_mrnabert_280k_fullft_frozen_delta.v1",
        "mode": "ROUTE_A_STEP1_FULLFT_ABLATION_FROZEN_DELTA",
        "trainable_parameter_count": sum(p.numel() for p in trainable),
        "library_clean_count": len(clean),
        "metrics": {
            "task_macro_spearman": metrics.get("task_macro_spearman"),
            "top_1": metrics.get("source_macro_top_1_accuracy"),
            "ndcg_at_10": metrics.get("source_macro_ndcg_at_k"),
        },
        "reference": {
            "route_a_step1_lora": 0.2470,
            "frozen_optimus_280k": 0.3132,
            "frozen_framepool_280k": 0.2956,
            "w0_critic_from_scratch": 0.1987,
            "critic_v5": 0.1354,
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
