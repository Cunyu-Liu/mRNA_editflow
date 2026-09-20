#!/usr/bin/env python3
"""Task 3.4 M1 real-dense-data intervention arm (R6, Route A incorporation), v1 2026-09-20.

Clone of run_route2_mrnabert_280k_fullft_v2.py (Route A full-FT V2) with a SINGLE
changed surface: data side. 280K MRL library is kept verbatim (load_library +
audit_leakage identical); M1 corpus rows (GSE232927 Castillo-Hair 2024, converted
mrl_converted_20260909, rl_paper authoritative label) are appended to the TRAIN
pool: HepG2 r1 + T-cell r1/r2 defined_end files, keep=1 rows, deterministic
downsample to 300,000 rows (seed 20260920, rng.sample over sorted pool).
Leakage exclusions: all 4,848 unique sequences of benchmark_v2/m1_mrl_eval_row
(sources+candidates) plus the 4 pigeonhole-flagged TRAIN-overlap sequences.

Model / loss / optimizer / hyperparameters / schedule / eval protocol: identical
to V2 (MRNABERT + MeanPoolRegressor, AdamW lr 2e-5 wd 1e-4, batch 128, 6 epochs,
FINAL-EPOCH-FIXED, frozen-delta K=10 on GSE114002 VALIDATION 730).

Preregistration: docs/paper/m1_intervention_arm_amendment_v1.md (frozen before
launch). G1 direction gate single-seed vs 0.3158 (3-seed ensemble baseline);
G3 efficiency items (wallclock, peak memory) recorded; G4 deferred to harvest.

Heartbeat: heartbeat.json refreshed every 500 steps (status/epoch/step/UTC);
run_summary.json written at end (final metrics + config + efficiency).
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
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
M1_DIR = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/castillohair2024/mrl_converted_20260909")
M1_EVAL_ROW = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/benchmark_v2/m1_mrl_eval_row/projection_rows.jsonl")
OUT_DIR = Path(os.environ.get("M1_INTERVENTION_OUT_DIR", "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_m1_intervention/seed_20260920"))

M1_FILES = [
    "GSE232927_processed_defined_end_hepg2_r1_mrl.tsv.gz",
    "GSE232927_processed_defined_end_tcell_r1_mrl.tsv.gz",
    "GSE232927_processed_defined_end_tcell_r2_mrl.tsv.gz",
]
M1_TARGET_ROWS = 300_000
M1_DOWNSAMPLE_SEED = 20260920
PIGEONHOLE_FLAGGED = {
    "CCCACAGCTTCCCAGGCCGCGGGTGCTGATTGCCCGCCTGCCCGTGGGTC",
    "CTGCTGGCAGAGAAGCTGGAGAACTGTGATTTCAATTAAGGTATTAAGTC",
    "CTTTCAGCAGCTCTCAGGGCCTTGGGCTCATCCCGAGTCCCGGGCTCAGT",
    "CCGGTGAGGCACGGCCCTGCAGATTTTCCAGCGGATCCCCCGGTGGCCTC",
}

BATCH = 128
LR = 2e-5
WEIGHT_DECAY = 1e-4
SEED = 20260920


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_heartbeat(out_dir: Path, status: str, **fields) -> None:
    payload = {"status": status, "utc": utcnow(), "schema_version": "route_a_v3_m1_intervention_heartbeat.v1"}
    payload.update(fields)
    (out_dir / "heartbeat.json").write_text(json.dumps(payload, indent=1, sort_keys=True))


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


def load_m1_excluded() -> set[str]:
    excluded: set[str] = set()
    with M1_EVAL_ROW.open() as handle:
        for line in handle:
            row = json.loads(line)
            excluded.add(row["source_sequence"])
            excluded.add(row["candidate_sequence"])
    excluded |= PIGEONHOLE_FLAGGED
    return excluded


def load_m1_pool(excluded: set[str]) -> dict[str, float]:
    pool: dict[str, float] = {}
    for name in M1_FILES:
        with gzip.open(M1_DIR / name, "rt") as handle:
            header = handle.readline().rstrip("\n").split("\t")
            utr_index = header.index("utr")
            rl_index = header.index("rl_paper")
            keep_index = header.index("keep")
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if fields[keep_index] != "1":
                    continue
                utr = fields[utr_index]
                if utr in excluded or utr in pool:
                    continue
                pool[utr] = float(fields[rl_index])
    return pool


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
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--m1-rows", type=int, default=M1_TARGET_ROWS)
    parser.add_argument("--m1-seed", type=int, default=M1_DOWNSAMPLE_SEED)
    args = parser.parse_args()
    epochs = args.epochs
    seed = args.seed

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - GPU required")
    device = torch.device(f"cuda:{args.physical_gpu_index}")
    torch.manual_seed(seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_heartbeat(OUT_DIR, "INIT", epoch=0, step=0, note="data loading")

    library = load_library()
    flagged = audit_leakage(library)
    clean = [(utr, float(np.mean(values))) for utr, values in library.items() if utr not in flagged]

    excluded = load_m1_excluded()
    m1_pool = load_m1_pool(excluded)
    m1_keys_sorted = sorted(m1_pool)
    rng = random.Random(args.m1_seed)
    if len(m1_keys_sorted) > args.m1_rows:
        m1_keys = rng.sample(m1_keys_sorted, args.m1_rows)
    else:
        m1_keys = list(m1_keys_sorted)
    m1_rows = [(utr, m1_pool[utr]) for utr in sorted(m1_keys)]
    print(f"library={len(library)} flagged={len(flagged)} clean={len(clean)}", flush=True)
    print(f"m1_pool={len(m1_pool)} m1_target={args.m1_rows} m1_sampled={len(m1_rows)} excluded_eval_row_seqs={len(excluded)}", flush=True)

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

    combined = clean + m1_rows
    sequences = [utr for utr, _ in combined]
    targets_np = np.asarray([rl for _, rl in combined], dtype=np.float64)
    mean, std = targets_np.mean(), targets_np.std()
    targets = torch.tensor((targets_np - mean) / std, dtype=torch.float32)
    print(f"train_pool_total={len(sequences)} target_mean={mean:.4f} target_std={std:.4f}", flush=True)

    encoded = tokenizer([format_sequence(s) for s in sequences], add_special_tokens=True, padding=True, truncation=False, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

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
    eval_ids = sorted(records)
    observations = ev.load_observations([CANONICAL_GSE114002], validation_ids)

    def frozen_delta_metrics() -> dict:
        model.eval()
        def score_batch(seqs):
            enc = tokenizer([format_sequence(s) for s in seqs], add_special_tokens=True, padding=True, truncation=False, return_tensors="pt")
            values = []
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for start in range(0, len(seqs), 256):
                    b_ids = enc["input_ids"][start:start+256].to(device)
                    b_mask = enc["attention_mask"][start:start+256].to(device)
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
    write_heartbeat(OUT_DIR, "TRAINING", epoch=0, step=0, total_steps=total_steps, train_rows=len(sequences))
    order = torch.randperm(len(sequences))
    model.train()
    step = 0
    t_start = time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        losses = []
        for start in range(0, len(order), BATCH):
            idx = order[start:start+BATCH]
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
            if step % 500 == 0:
                print(f"epoch {epoch + 1} step {step}: mse {np.mean(losses[-200:]):.4f} lr {scheduler.get_last_lr()[0]:.2e}", flush=True)
                write_heartbeat(OUT_DIR, "TRAINING", epoch=epoch + 1, step=step, total_steps=total_steps,
                                recent_mse=float(np.mean(losses[-200:])), elapsed_s=round(time.time() - t_start, 1))
        order = torch.randperm(len(sequences))

        torch.save({
            "schema_version": "route_a_v3_m1_intervention_fullft_v1.v1",
            "model_state_dict": {k: v for k, v in model.state_dict().items()},
            "target_mean": float(mean), "target_std": float(std),
            "seed": seed, "epochs_total": epochs, "epoch": epoch + 1,
            "library_clean_count": len(clean), "m1_rows": len(m1_rows),
        }, OUT_DIR / f"fullft_epoch_{epoch + 1}.pt")
        m, preds = frozen_delta_metrics()
        rec = {"epoch": epoch + 1, "primary": epoch + 1 == epochs, **m}
        epoch_metrics_file.write(json.dumps(rec) + "\n")
        epoch_metrics_file.flush()
        print(f"== epoch {epoch + 1} frozen-delta: {json.dumps(m)}", flush=True)
        write_heartbeat(OUT_DIR, "EPOCH_DONE", epoch=epoch + 1, step=step, total_steps=total_steps,
                        epoch_spearman=m.get("task_macro_spearman"), elapsed_s=round(time.time() - t_start, 1))
        if epoch + 1 == epochs:
            final_predictions = preds
            final_metrics = m
    loss_log.close()
    epoch_metrics_file.close()

    wallclock_s = time.time() - t_start
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    report = {
        "schema_version": "route_a_v3_m1_intervention_fullft_v1.v1",
        "mode": "TASK_3_4_M1_INTERVENTION_ARM_ROUTE_A",
        "selection_rule": "FINAL_EPOCH_%d_FIXED" % epochs,
        "prereg": "docs/paper/m1_intervention_arm_amendment_v1.md",
        "trainable_parameter_count": sum(p.numel() for p in trainable),
        "library_clean_count": len(clean),
        "m1_files": M1_FILES,
        "m1_pool_size": len(m1_pool),
        "m1_target_rows": args.m1_rows,
        "m1_downsample_seed": args.m1_seed,
        "m1_rows": len(m1_rows),
        "m1_excluded_seqs": len(excluded),
        "train_pool_total": len(sequences),
        "epochs": epochs,
        "seed": seed,
        "metrics": final_metrics,
        "efficiency": {
            "wallclock_s": round(wallclock_s, 1),
            "wallclock_h": round(wallclock_s / 3600, 2),
            "peak_cuda_memory_gb": round(peak_mem_gb, 2),
            "steps_total": total_steps,
        },
        "reference": {
            "route_a_fullft_v2_3seed_ensemble": 0.3158289984824722,
            "route_a_fullft_v2_seed20260903_single": 0.31979237401392574,
            "frozen_optimus_280k": 0.3132,
            "g1_direction_threshold": 0.3158289984824722 + 0.01,
        },
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    (OUT_DIR / "frozen_delta_results.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    with (OUT_DIR / "predictions.jsonl").open("w") as handle:
        for rid in eval_ids:
            handle.write(json.dumps({"canonical_record_id": rid, "predicted_direction_normalized_delta": final_predictions[rid]}) + "\n")
    write_heartbeat(OUT_DIR, "DONE", epoch=epochs, step=step, total_steps=total_steps,
                    final_spearman=final_metrics.get("task_macro_spearman"),
                    wallclock_h=round(wallclock_s / 3600, 2))
    print(json.dumps(report["metrics"], indent=1), flush=True)
    print(json.dumps(report["efficiency"], indent=1), flush=True)
    print("wrote", OUT_DIR / "run_summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
