"""P2-02: Standalone held-out evaluator for Stage A checkpoints.

Loads a stage_a_best.pt checkpoint, rebuilds the model from the embedded config,
and computes the Edit-Flow CTMC loss on a held-out val split (no gradients, no
AMP). Used to independently assess the 4 existing Stage A checkpoints that were
trained without a held-out curve.

Usage:
    python -m eval.eval_stage_a_heldout \
        --checkpoint ckpts/stage_a_full_a100_max_gencode_100k_seed0/stage_a_best.pt \
        --records-jsonl data/reconstructed/p0_data_reconstruction_v1/combined/combined_model_view.records.jsonl \
        --val-idx benchmark/dev/p0_data_reconstruction_v1/combined_family/val.idx \
        --device cuda:6 \
        --max-eval 500 \
        --batch-size 4 \
        --output docs/stage_a_heldout_eval_seed0.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

# Allow running from repo root without install
_REPO_ROOT = Path(__file__).resolve().parents[1]
# The mrna_editflow package IS the repo root (it has __init__.py), so we need
# its PARENT on sys.path for `from mrna_editflow.* import ...` to resolve.
_PACKAGE_PARENT = _REPO_ROOT.parent
for _p in (str(_PACKAGE_PARENT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mrna_editflow.core.config import MEFConfig
from mrna_editflow.core.constants import VOCAB_MODEL_SIZE
from mrna_editflow.core import mrna_flow_utils as U
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.download_mrna import load_records_jsonl
from mrna_editflow.models.backbones import FrozenBackbone
from mrna_editflow.models.mrna_editformer import MRNAEditFormer
from train_backbone import _coerce_config, _flow_batch_loss, build_stage_a_model


def _load_idx(path: str) -> List[int]:
    """Load a .idx file (one integer per line)."""
    with open(path, "r") as fh:
        return [int(line.strip()) for line in fh if line.strip()]


def _select_val_records(
    records: Sequence[MRNARecord],
    val_idx: Sequence[int],
    max_eval: Optional[int],
) -> List[MRNARecord]:
    """Pick val records by index, optionally capped to max_eval (deterministic first-N)."""
    n = len(records)
    picked: List[MRNARecord] = []
    for i in val_idx:
        if 0 <= i < n:
            picked.append(records[i])
        if max_eval is not None and len(picked) >= max_eval:
            break
    return picked


@torch.no_grad()
def evaluate_heldout(
    checkpoint_path: str,
    records: Sequence[MRNARecord],
    val_idx: Sequence[int],
    device: torch.device,
    max_eval: Optional[int] = 500,
    batch_size: int = 4,
    seed: int = 1729,
) -> Dict[str, Any]:
    """Evaluate a Stage A checkpoint on the held-out val split.

    Returns dict with: checkpoint, n_eval, loss_mean, loss_median, loss_p95,
    loss_std, loss_min, loss_max, edit_loss_mean, aux_loss_mean, elapsed_s,
    config_summary, device.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config")
    if cfg_dict is None:
        raise ValueError("checkpoint missing 'config' field")
    cfg = _coerce_config(cfg_dict)
    # Disable aux struct supervision for eval: it requires an explicit target
    # tensor that we don't have at eval time. We only care about edit_loss.
    cfg.model.use_aux_struct = False
    ckpt_step = int(ckpt.get("step", -1))
    ckpt_best_loss = float(ckpt.get("best_loss", float("nan")))

    # Build model and load state
    backbone, model = build_stage_a_model(cfg, device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:
        print(f"[warn] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}", file=sys.stderr)
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}", file=sys.stderr)
    model.eval()
    backbone.eval()

    # Select val records
    val_records = _select_val_records(records, val_idx, max_eval)
    n_eval = len(val_records)
    if n_eval == 0:
        raise ValueError("no val records selected (empty val_idx or all out of range?)")

    scheduler = U.CubicScheduler()
    losses: List[float] = []
    edit_losses: List[float] = []
    aux_losses: List[float] = []

    start = time.perf_counter()
    cursor = 0
    n_done = 0
    while n_done < n_eval:
        batch = val_records[n_done : n_done + batch_size]
        if not batch:
            break
        # Use a deterministic seed per batch for reproducibility
        bseed = seed + n_done * 1009
        loss_dict = _flow_batch_loss(
            model=model,
            backbone=backbone,
            records=batch,
            cfg=cfg,
            device=device,
            scheduler=scheduler,
            seed=bseed,
            property_bucket=None,
        )
        # loss_dict values are tensors (fp32); take item
        losses.append(float(loss_dict["loss"].detach().cpu()))
        edit_losses.append(float(loss_dict["edit_loss"].detach().cpu()))
        aux_losses.append(float(loss_dict["aux_loss"].detach().cpu()))
        n_done += len(batch)
        if n_done % (batch_size * 10) == 0:
            print(f"  eval progress: {n_done}/{n_eval}", file=sys.stderr)
    elapsed = time.perf_counter() - start

    losses_arr = np.array(losses)
    return {
        "checkpoint": checkpoint_path,
        "checkpoint_step": ckpt_step,
        "checkpoint_best_loss": ckpt_best_loss,
        "n_eval": n_eval,
        "batch_size": batch_size,
        "loss_mean": float(np.mean(losses_arr)),
        "loss_median": float(np.median(losses_arr)),
        "loss_p95": float(np.percentile(losses_arr, 95)),
        "loss_p99": float(np.percentile(losses_arr, 99)),
        "loss_std": float(np.std(losses_arr)),
        "loss_min": float(np.min(losses_arr)),
        "loss_max": float(np.max(losses_arr)),
        "edit_loss_mean": float(np.mean(edit_losses)),
        "aux_loss_mean": float(np.mean(aux_losses)),
        "elapsed_s": float(elapsed),
        "samples_per_s": float(n_eval / max(elapsed, 1e-9)),
        "device": str(device),
        "config_summary": {
            "lr": float(cfg.train.lr),
            "batch_size": int(cfg.train.batch_size),
            "grad_accum": int(cfg.train.grad_accum),
            "amp": bool(cfg.train.amp),
            "grad_clip": float(cfg.train.grad_clip),
            "model_dim": int(cfg.model.model_dim),
            "num_layers": int(cfg.model.num_layers),
        },
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P2-02: Stage A held-out evaluator")
    parser.add_argument("--checkpoint", required=True, help="path to stage_a_best.pt")
    parser.add_argument(
        "--records-jsonl",
        default="data/reconstructed/p0_data_reconstruction_v1/combined/combined_model_view.records.jsonl",
        help="records JSONL (must match the split manifest's records path)",
    )
    parser.add_argument("--val-idx", required=True, help="path to val.idx file")
    parser.add_argument("--device", default="cuda:6", help="torch device (e.g. cuda:6, cpu)")
    parser.add_argument("--max-eval", type=int, default=500, help="cap on number of val records (0 = all)")
    parser.add_argument("--batch-size", type=int, default=4, help="eval batch size")
    parser.add_argument("--seed", type=int, default=1729, help="eval seed for coupling sampling")
    parser.add_argument("--output", default=None, help="output JSON path (default: stdout only)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    device = torch.device(args.device)
    max_eval = None if args.max_eval and args.max_eval <= 0 else args.max_eval

    print(f"[info] loading records from {args.records_jsonl}", file=sys.stderr)
    records = load_records_jsonl(args.records_jsonl)
    print(f"[info] loaded {len(records)} records", file=sys.stderr)

    val_idx = _load_idx(args.val_idx)
    print(f"[info] val_idx has {len(val_idx)} indices; max_eval={max_eval}", file=sys.stderr)

    result = evaluate_heldout(
        checkpoint_path=args.checkpoint,
        records=records,
        val_idx=val_idx,
        device=device,
        max_eval=max_eval,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    out_str = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as fh:
            fh.write(out_str + "\n")
        print(f"[info] wrote {args.output}", file=sys.stderr)
    print(out_str)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
