#!/usr/bin/env python3
"""Stage 0b: Optimus-architecture from-scratch control (route A discriminator).

Separates architecture from prior: identical Optimus5Prime architecture,
random initialization, trained ONLY on GSE114002 TRAIN records (2,443 source/
candidate pairs, absolute endpoint regression) - no external 280K library.

Eval: delta = f(candidate) - f(source) on VALIDATION, frozen Task-1 evaluator
(K=10), exactly the frozen-delta protocol. Comparison targets:
- frozen-Optimus (280K prior, no task training): 0.3132
- W0 (170M critic from scratch, same TRAIN): 0.1987
If from-scratch Optimus reaches ~0.31 -> architecture dominates (route A
downgraded); if ~0.15-0.20 -> prior dominates (route A GO).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch import nn

EVAL_REPO = "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901"

_ev_spec = importlib.util.spec_from_file_location(
    "ev", EVAL_REPO + "/scripts/route_a_v3/evaluate_route2_prediction_v1.py"
)
ev = importlib.util.module_from_spec(_ev_spec)
sys.modules["ev"] = ev
_ev_spec.loader.exec_module(ev)

_h_spec = importlib.util.spec_from_file_location(
    "harness", str(REPO_ROOT / "scripts/route_a_v3/run_route2_external_prediction_baselines_v1.py")
)
harness = importlib.util.module_from_spec(_h_spec)
sys.modules["harness"] = harness
_h_spec.loader.exec_module(harness)

MANIFEST = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/manifests/route2_development_frozen_v1/development_manifest.jsonl"
)
CANONICAL = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/canonical/GSE114002/v1/canonical_records.private.jsonl"
)


def optimus_from_scratch() -> nn.Module:
    """Randomly initialized twin of the frozen Optimus5Prime port."""
    model = harness.Optimus5Prime.__new__(harness.Optimus5Prime)
    nn.Module.__init__(model)
    model.conv1_weight = nn.Parameter(torch.empty(120, 4, 8))
    model.conv1_bias = nn.Parameter(torch.zeros(120))
    model.conv2_weight = nn.Parameter(torch.empty(120, 120, 8))
    model.conv2_bias = nn.Parameter(torch.zeros(120))
    model.conv3_weight = nn.Parameter(torch.empty(120, 120, 8))
    model.conv3_bias = nn.Parameter(torch.zeros(120))
    model.dense1_weight = nn.Parameter(torch.empty(40, 6000))
    model.dense1_bias = nn.Parameter(torch.zeros(40))
    model.dense2_weight = nn.Parameter(torch.empty(1, 40))
    model.dense2_bias = nn.Parameter(torch.zeros(1))
    for module, dim in (
        (model.conv1_weight, 1), (model.conv2_weight, 1), (model.conv3_weight, 1),
        (model.dense1_weight, 1), (model.dense2_weight, 1),
    ):
        nn.init.kaiming_normal_(module, mode="fan_out", nonlinearity="relu")
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=Path(
        "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/analysis_optimus_from_scratch_20260903"
    ))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - GPU required")
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

    # Absolute-endpoint regression on TRAIN source+candidate sequences.
    sequences = []
    values = []
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
    permutation = torch.randperm(dataset_size)
    monitor_idx = permutation[:monitor_size]
    train_idx = permutation[monitor_size:]

    model = optimus_from_scratch().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    best_state = None
    best_monitor = float("inf")
    history = []
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
                model(encoded[monitor_idx]).squeeze(-1), targets[monitor_idx]
            ))
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
        source_scores = model(
            harness.one_hot([r["source_sequence"] for r in validation_records], device)
        ).double().cpu().numpy()
        candidate_scores = model(
            harness.one_hot([r["candidate_sequence"] for r in validation_records], device)
        ).double().cpu().numpy()
    delta = candidate_scores - source_scores
    predictions = {rid: float(delta[i]) for i, rid in enumerate(validation_ids)}

    observations = ev.load_observations([CANONICAL], splits["VALIDATION"])
    metrics = ev.evaluate(observations, predictions, 10)

    report = {
        "schema_version": "route_a_v3_route2_optimus_from_scratch.v1",
        "mode": "FROM_SCRATCH_TASK_ONLY",
        "seed": args.seed,
        "train_record_count": len(train_records),
        "train_sequence_count": dataset_size,
        "epochs": args.epochs,
        "best_monitor_mse": best_monitor,
        "metrics": {
            "task_macro_spearman": metrics.get("task_macro_spearman"),
            "top_1": metrics.get("source_macro_top_1_accuracy"),
            "ndcg_at_10": metrics.get("source_macro_ndcg_at_k"),
        },
        "reference": {
            "frozen_optimus_280k_prior": 0.3132,
            "frozen_framepool_280k_prior": 0.2956,
            "w0_critic_from_scratch": 0.1987,
            "critic_v5_multitask": 0.1354,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "results.json").open("w") as handle:
        json.dump(report, handle, indent=1, sort_keys=True)
    print(json.dumps(report["metrics"], indent=1))
    print("wrote", args.output_dir / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
