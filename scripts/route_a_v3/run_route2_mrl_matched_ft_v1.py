#!/usr/bin/env python3
"""Baseline P1-1: MRL matched fine-tuned external arms (Optimus / FramePool).

SPECS_BASELINE_LEADERBOARD spec 2026-09-08 增补 §V.4 P1-1 (R1 symmetry closure):
official frozen weights as INIT + fine-tune on GSE114002 task TRAIN (same data,
same budget, same HPO as the preregistered from-scratch control batch 20260903:
absolute-endpoint z-scored regression, Adam lr 1e-3 wd 1e-6, 300 epochs,
batch 128, 10% monitor split with best-monitor state selection, seed 20260903).
Eval: frozen-delta on VALIDATION (frozen Task-1 evaluator, K=10) -- identical
to the frozen/from-scratch rows, so the table gains the matched-FT mode for
MRL external rows (the same closure the MPRAU row got in batch 6.3.2).

The frozen ports register weights as buffers; for fine-tuning they are
converted to parameters (same tensors, requires_grad=True).
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

_h_spec = importlib.util.spec_from_file_location(
    "harness", str(REPO_ROOT / "scripts/route_a_v3/run_route2_external_prediction_baselines_v1.py"))
harness = importlib.util.module_from_spec(_h_spec)
sys.modules["harness"] = harness
_h_spec.loader.exec_module(harness)

MANIFEST = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/manifests/route2_development_frozen_v1/development_manifest.jsonl")
CANONICAL = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/canonical/GSE114002/v1/canonical_records.private.jsonl")
WEIGHTS = {
    "optimus5prime": Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/optimus5prime/main_MRL_model.hdf5"),
    "framepool": Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/framepool/Framepool_combined_residual.h5"),
}
OUT_ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/analysis_mrl_matched_ft_20260909")


def build_model(name: str) -> nn.Module:
    """Load the frozen port and convert its buffers to trainable parameters in-place.

    The frozen ports register official weights as buffers; swapping each buffer
    for a Parameter with the same name keeps the class forward working unchanged
    while making the weights trainable (matched-FT semantics: official init).
    """
    model_cls = harness.Optimus5Prime if name == "optimus5prime" else harness.FramePool
    model = model_cls(WEIGHTS[name])
    for buf_name, tensor in list(model.named_buffers()):
        del model._buffers[buf_name]
        model.register_parameter(buf_name, nn.Parameter(tensor.clone()))
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--model", required=True, choices=("optimus5prime", "framepool"))
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.physical_gpu_index}")
    torch.manual_seed(args.seed)

    splits = {}
    for split in ("TRAIN", "VALIDATION"):
        ids = set()
        with MANIFEST.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row["study_unit_id"] == "GSE114002" and row["split"] == split:
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

    # absolute-endpoint regression targets (identical to from-scratch control)
    sequences, values = [], []
    for record in train_records:
        for key in ("source_sequence", "candidate_sequence"):
            sequences.append(record[key])
        values.append(float(record["source_endpoint_value"]))
        values.append(float(record["candidate_endpoint_value"]))
    values = np.asarray(values, dtype=np.float64)
    mean, std = values.mean(), values.std()
    targets = torch.tensor((values - mean) / std, dtype=torch.float32, device=device)
    encoded = harness.one_hot(sequences, device)
    dataset_size = encoded.shape[0]
    monitor_size = max(dataset_size // 10, 1)
    g = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(dataset_size, generator=g)
    monitor_idx = permutation[:monitor_size]
    train_idx = permutation[monitor_size:]

    model = build_model(args.model).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"[{args.model}] trainable params: {n_trainable:,}", flush=True)

    optimizer = torch.optim.Adam(trainable_params, lr=1e-3, weight_decay=1e-6)
    best_state, best_monitor, history = None, float("inf"), []
    for epoch in range(args.epochs):
        model.train()
        for start in range(0, len(train_idx), 128):
            idx = train_idx[start : start + 128]
            prediction = model(encoded[idx]).squeeze(-1)
            loss = nn.functional.mse_loss(prediction, targets[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            monitor = float(nn.functional.mse_loss(
                model(encoded[monitor_idx]).squeeze(-1), targets[monitor_idx]))
        history.append(monitor)
        if monitor < best_monitor:
            best_monitor = monitor
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 50 == 0:
            print(f"epoch {epoch + 1}: monitor mse {monitor:.4f} (best {best_monitor:.4f})", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    validation_records = [records[rid] for rid in validation_ids]
    with torch.no_grad():
        source_scores = model(harness.one_hot([r["source_sequence"] for r in validation_records], device)).double().cpu().numpy()
        candidate_scores = model(harness.one_hot([r["candidate_sequence"] for r in validation_records], device)).double().cpu().numpy()
    delta = candidate_scores - source_scores
    predictions = {rid: float(delta[i]) for i, rid in enumerate(validation_ids)}
    observations = ev.load_observations([CANONICAL], splits["VALIDATION"])
    metrics = ev.evaluate(observations, predictions, 10)

    out_dir = OUT_ROOT / f"{args.model}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.jsonl").open("w") as fh:
        for rid in validation_ids:
            fh.write(json.dumps({"canonical_record_id": rid, "prediction": predictions[rid]}) + "\n")
    report = {
        "schema_version": "route_a_v3_route2_mrl_matched_ft.v1",
        "mode": "MATCHED_FINE_TUNE_OFFICIAL_INIT",
        "model": args.model, "seed": args.seed, "epochs": args.epochs,
        "train_record_count": len(train_records), "train_sequence_count": dataset_size,
        "best_monitor_mse": best_monitor, "n_trainable": n_trainable,
        "metrics": {"task_macro_spearman": metrics.get("task_macro_spearman"),
                    "top_1": metrics.get("source_macro_top_1_accuracy"),
                    "ndcg_at_10": metrics.get("source_macro_ndcg_at_k")},
        "protocol": "matched to from-scratch control 20260903: Adam 1e-3 wd 1e-6, "
                    "batch 128, 300 epochs, 10% monitor best-state selection, "
                    "absolute-endpoint z-scored regression on TRAIN",
        "reference": {"frozen_optimus": 0.3132, "frozen_framepool": 0.2956,
                      "optimus_from_scratch_control": 0.0984,
                      "v9_1a_ensemble": 0.3172, "route_a_3seed": 0.3158},
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report["metrics"], indent=1))
    print(f"wrote {out_dir / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
