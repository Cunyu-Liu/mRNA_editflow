#!/usr/bin/env python3
"""Baseline P1-3: Saluki fine-tuned on GSE217518 RNA half-life (VALIDATION).

SPECS_BASELINE_LEADERBOARD spec 2026-09-08 增补 §V.4 P1-3 (H3 second empirical
closure): official Saluki fold-0 checkpoint as init + fine-tune on the
GSE217518 TRAIN split (absolute half-life regression on source+candidate
sequences, six-channel encoding right-padded to 12288 -- the same native
truncation clause as the frozen row). If the fine-tuned Saluki still scores
~0 on VALIDATION frozen-delta, the "physically unlearnable" claim (label ICC
~0.001-0.013) gets its second empirical leg (after the frozen modern-LM
four-row closure of batch 6.3.4b): even a supervised endogenous half-life
specialist cannot learn the variant-level delta under this label noise.

Protocol (matched to the V8-style adaptation conventions, preregistered here):
- init: model0_best.h5 fold-0 official weights (the same file the frozen row
  used); trainable = all parameters (buffers -> parameters swap).
- data: GSE217518 TRAIN records; target = direction-normalized delta is NOT
  used (that is the eval calibre); training target = absolute
  source/candidate endpoint values z-scored per study (same convention as the
  MRL matched-FT arm), which mirrors Saluki's native absolute-regression task.
- optim: Adam 1e-4 wd 1e-6, batch 8 (12288-length six-channel inputs are
  memory-heavy), 30 epochs, 10% monitor best-state, seed 20260903.
- eval: frozen-delta on VALIDATION (frozen evaluator, K=10), both regions.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EVAL_REPO = "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901"
_ev_spec = importlib.util.spec_from_file_location(
    "ev", EVAL_REPO + "/scripts/route_a_v3/evaluate_route2_prediction_v1.py")
ev = importlib.util.module_from_spec(_ev_spec)
sys.modules["ev"] = ev
_ev_spec.loader.exec_module(ev)

from core.route2_saluki_port_v1 import SalukiGRUV1, encode_saluki_six_channel_v1  # noqa: E402

MANIFEST = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/manifests/route2_development_frozen_v1/development_manifest.jsonl")
CANONICAL = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/canonical/GSE217518/v1/canonical_records.jsonl")
SALUKI_CKPT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/saluki/datasets/deeplearning/train_gru/f6_c0/train/model0_best.h5")
OUT_ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/analysis_saluki_ft_halflife_20260909")
SALUKI_FULL_LENGTH = 12288


def load_model() -> nn.Module:
    model = SalukiGRUV1(SALUKI_CKPT)
    # buffers -> trainable parameters in place (matched-FT semantics).
    # BN running stats (moving mean/var) must stay buffers: batch_norm is not
    # differentiable w.r.t. them and they are statistics, not learned weights.
    for buf_name, tensor in list(model.named_buffers()):
        if buf_name.endswith(("_mean", "_var")):
            continue
        del model._buffers[buf_name]
        model.register_parameter(buf_name, nn.Parameter(tensor.clone()))
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.physical_gpu_index}")
    torch.manual_seed(args.seed)

    splits = {}
    for split in ("TRAIN", "VALIDATION"):
        ids = set()
        with MANIFEST.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row["study_unit_id"] == "GSE217518" and row["split"] == split:
                    ids.add(str(row["canonical_record_id"]))
        splits[split] = ids
    records = {}
    with CANONICAL.open() as handle:
        for line in handle:
            row = json.loads(line)
            rid = str(row.get("canonical_record_id"))
            if rid in splits["TRAIN"] or rid in splits["VALIDATION"]:
                records[rid] = row
    train_records = [r for r in records.values() if str(r["canonical_record_id"]) in splits["TRAIN"]]
    validation_ids = sorted(splits["VALIDATION"])
    print(f"train={len(train_records)} val={len(validation_ids)}", flush=True)

    # absolute endpoint regression targets (Saluki native task convention)
    sequences, values = [], []
    for record in train_records:
        for key in ("source_sequence", "candidate_sequence"):
            sequences.append(record[key])
        values.append(float(record["source_endpoint_value"]))
        values.append(float(record["candidate_endpoint_value"]))
    values = np.asarray(values, dtype=np.float64)
    mean, std = values.mean(), values.std()
    targets = torch.tensor((values - mean) / std, dtype=torch.float32, device=device)

    # encode once (six-channel, right-padded to 12288) on CPU then move per batch
    print("encoding train set (six-channel 12288)...", flush=True)
    encoded_np = np.stack([encode_saluki_six_channel_v1(s, SALUKI_FULL_LENGTH) for s in sequences])
    encoded = torch.from_numpy(encoded_np)
    dataset_size = encoded.shape[0]
    monitor_size = max(dataset_size // 10, 1)
    g = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(dataset_size, generator=g)
    monitor_idx = permutation[:monitor_size]
    train_idx = permutation[monitor_size:]
    monitor_x = encoded[monitor_idx].to(device)
    monitor_y = targets[monitor_idx]
    print(f"encoded {dataset_size} sequences", flush=True)

    model = load_model().to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable):,}", flush=True)
    optimizer = torch.optim.Adam(trainable, lr=1e-4, weight_decay=1e-6)

    best_state, best_monitor = None, float("inf")
    for epoch in range(args.epochs):
        model.train()
        order = train_idx[torch.randperm(len(train_idx))]
        for start in range(0, len(order), args.batch):
            idx = order[start : start + args.batch]
            x = encoded[idx].to(device)
            y = targets[idx]
            prediction = model(x).squeeze(-1)
            loss = nn.functional.mse_loss(prediction, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            monitor = float(nn.functional.mse_loss(model(monitor_x).squeeze(-1), monitor_y))
        if monitor < best_monitor:
            best_monitor = monitor
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch + 1}: monitor mse {monitor:.4f} (best {best_monitor:.4f})", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # frozen-delta eval on VALIDATION, per region
    validation_records = [records[rid] for rid in validation_ids]
    preds = {}
    batch = 8
    with torch.no_grad():
        for start in range(0, len(validation_records), batch):
            chunk = validation_records[start : start + batch]
            x = torch.from_numpy(np.stack(
                [encode_saluki_six_channel_v1(r["source_sequence"], SALUKI_FULL_LENGTH) for r in chunk])).to(device)
            src = model(x).squeeze(-1).double().cpu().numpy()
            x = torch.from_numpy(np.stack(
                [encode_saluki_six_channel_v1(r["candidate_sequence"], SALUKI_FULL_LENGTH) for r in chunk])).to(device)
            cnd = model(x).squeeze(-1).double().cpu().numpy()
            for i, r in enumerate(chunk):
                preds[str(r["canonical_record_id"])] = float(cnd[i] - src[i])

    out_dir = OUT_ROOT / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.jsonl").open("w") as fh:
        for rid in validation_ids:
            fh.write(json.dumps({"canonical_record_id": rid, "prediction": preds[rid]}) + "\n")

    # per-region metrics via frozen evaluator
    regions = {}
    for region in ("5UTR", "3UTR"):
        region_ids = {rid for rid in validation_ids
                      if str(records[rid].get("region")) == region}
        if not region_ids:
            continue
        observations = ev.load_observations([CANONICAL], region_ids)
        region_preds = {rid: preds[rid] for rid in region_ids}
        m = ev.evaluate(observations, region_preds, 10)
        regions[region] = {"n": len(region_ids),
                           "task_macro_spearman": m.get("task_macro_spearman"),
                           "top_1": m.get("source_macro_top_1_accuracy")}
        print(f"[{region}] spearman={regions[region]['task_macro_spearman']}", flush=True)

    report = {
        "schema_version": "route_a_v3_route2_saluki_ft_halflife.v1",
        "mode": "MATCHED_FINE_TUNE_OFFICIAL_INIT",
        "init": str(SALUKI_CKPT), "seed": args.seed, "epochs": args.epochs,
        "train_records": len(train_records), "best_monitor_mse": best_monitor,
        "regions": regions,
        "reference": {"saluki_frozen_5utr": 0.0193, "saluki_frozen_3utr": 0.0985,
                      "label_ceiling_icc": "0.001-0.013 (physically unlearnable)"},
        "protocol": "Adam 1e-4 wd 1e-6, batch 8, 30ep, 10% monitor best-state, "
                    "absolute-endpoint z-scored regression, six-channel 12288 native padding",
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"wrote {out_dir / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
