"""P2-10 Option C: Stage A recovery with stricter fixes.

This script is derived from ``scripts/run_stage_a_recovery_p2_02.py`` with
three additional fixes (the "Option C" pivot per ``docs/p2_10_alternative_backbone.md``):

1. **LR warmup**: linear warmup over ``--warmup-steps`` (default 500) from 0 to
   ``cfg.train.lr``, then constant. P2-02 had no warmup; the first 500 steps
   saw raw full-LR updates on a freshly-initialized model, contributing to the
   early gradient instability.
2. **AMP disabled** (via config ``train.amp=false``): P2-02 used AMP with
   ``amp_init_scale=256.0``. The pre-clip grad_norm was 2000-30000, partly
   because fp16 precision amplifies the CTMC hazard term. Option C runs in
   pure fp32 for cleaner gradients.
3. **Larger effective batch** (via config ``batch_size=8, grad_accum=8`` =
   effective 64, 4x larger than P2-02's 16): reduces gradient variance and
   helps escape the saddle point that stalled P2-02 at step 2000.

The LR warmup is the only code-level change; AMP and batch size are config
changes. The warmup is implemented by modifying ``optimizer.param_groups[0]['lr']``
before each ``_run_optimizer_step`` call, so it does NOT modify
``train_backbone.py`` (which is loaded by the 4 running backbone processes).

Hard constraint compliance:
- Does NOT terminate or modify any running process.
- Uses ``--train-idx/--val-idx/--test-idx`` (paper mode) to enforce the split
  contract on the frozen ``combined_family`` split.
- All new code has unit tests (``tests/test_stage_a_recovery_p2_10_option_c.py``).
- Does NOT modify ``train_backbone.py`` or ``scripts/run_stage_a_recovery_p2_02.py``.

Usage:
    python -m scripts.run_stage_a_recovery_p2_10_option_c \\
        --config configs/stage_a_recovery_p2_10_option_c.json \\
        --records-jsonl data/reconstructed/p0_data_reconstruction_v1/combined/combined_model_view.records.jsonl \\
        --train-idx benchmark/dev/p0_data_reconstruction_v1/combined_family/train.idx \\
        --val-idx benchmark/dev/p0_data_reconstruction_v1/combined_family/val.idx \\
        --test-idx benchmark/dev/p0_data_reconstruction_v1/combined_family/test.idx \\
        --steps 10000 \\
        --seed 42 \\
        --device cuda:0 \\
        --warmup-steps 500 \\
        --run-mode paper
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

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_PARENT = _REPO_ROOT.parent
for _p in (str(_PACKAGE_PARENT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mrna_editflow.core.config import MEFConfig
from mrna_editflow.core import mrna_flow_utils as U
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.download_mrna import load_records_jsonl
from mrna_editflow.eval.artifact_contract import (
    build_run_metadata,
    normalize_run_mode,
    prepare_scientific_records,
    require_paper_cli_inputs,
    validate_output_namespace,
)
from mrna_editflow.data.split_contract import (
    VerifiedSplitContract,
    load_and_verify_split_manifest,
)

# Import building blocks from train_backbone (do NOT modify that file —
# the 4 running Stage A processes have already loaded it into memory).
from train_backbone import (
    _amp_enabled,
    _coerce_config,
    _config_dict,
    _make_grad_scaler,
    _resolve_device,
    _run_optimizer_step,
    _save_stage_a_checkpoint,
    _set_seed,
    _write_profile,
    _init_profile,
    build_stage_a_model,
)

# Reuse helpers from P2-02 recovery script (does NOT modify that file).
from scripts.run_stage_a_recovery_p2_02 import (
    _load_idx,
    _select_records_by_idx,
    _make_records_local,
    _verify_idx_files,
    run_heldout_eval,
)


# ---------------------------------------------------------------------------
# LR warmup helper (P2-10 Option C addition)
# ---------------------------------------------------------------------------
def _apply_lr_warmup(
    optimizer: torch.optim.Optimizer,
    step: int,
    warmup_steps: int,
    target_lr: float,
) -> float:
    """Apply linear LR warmup and return the current LR.

    During warmup (``step <= warmup_steps``), LR ramps linearly from 0 to
    ``target_lr``: ``lr = target_lr * step / warmup_steps``.
    After warmup, LR stays at ``target_lr``.

    This function modifies ``optimizer.param_groups[0]['lr']`` in place.
    It does NOT use a PyTorch scheduler (to avoid coupling with the
    GradScaler / optimizer step logic in ``_run_optimizer_step``).

    Complexity: O(1).
    """
    if warmup_steps <= 0:
        for group in optimizer.param_groups:
            group["lr"] = target_lr
        return target_lr
    if step <= 0:
        current_lr = 0.0
    elif step >= warmup_steps:
        current_lr = target_lr
    else:
        current_lr = target_lr * float(step) / float(warmup_steps)
    for group in optimizer.param_groups:
        group["lr"] = current_lr
    return current_lr


# ---------------------------------------------------------------------------
# Training loop with LR warmup (P2-10 Option C)
# ---------------------------------------------------------------------------
def train_stage_a_recovery_p2_10(
    config: Any,
    records: Optional[Sequence[MRNARecord]],
    steps: Optional[int],
    device: Any,
    seed: Optional[int],
    run_mode: str,
    split_contract: Optional[VerifiedSplitContract],
    split_role: Optional[str],
    train_idx_path: Optional[str] = None,
    val_idx_path: Optional[str] = None,
    test_idx_path: Optional[str] = None,
    warmup_steps: int = 500,
) -> Dict[str, object]:
    """P2-10 Option C recovery training loop.

    Same as ``train_stage_a_recovery`` from P2-02, plus:
    1. LR warmup over ``warmup_steps`` (linear 0 -> cfg.train.lr, then constant).
    2. Current LR logged to profile as ``current_lr``.
    """
    cfg = _coerce_config(config)
    run_mode = normalize_run_mode(run_mode)
    validate_output_namespace(cfg.train.save_dir, run_mode)
    validate_output_namespace(cfg.train.profile_path, run_mode)
    run_seed = int(cfg.data.seed if seed is None else seed)
    _set_seed(run_seed)
    dev = _resolve_device(device)

    # Prepare training records (split-contract aware)
    source_records = _make_records_local(records, cfg)
    train_records, data_provenance = prepare_scientific_records(
        source_records,
        run_mode=run_mode,
        split_contract=split_contract,
        split_role=split_role,
        allowed_roles=("train",),
    )
    scientific_validity = build_run_metadata(
        run_mode=run_mode,
        data_provenance=data_provenance,
        config=_config_dict(cfg),
        code_paths=(__file__,),
        training_seed=run_seed,
        extra_block_reasons=(
            "auxiliary_structural_supervision_enabled"
            if cfg.model.use_aux_struct else ""
        ),
    )

    # Build model + optimizer + scaler
    backbone, model = build_stage_a_model(cfg, dev)
    model.train()
    backbone.eval()
    target_lr = float(cfg.train.lr)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.0,  # P2-10: start at 0, warmup ramps to target_lr
        weight_decay=float(cfg.train.weight_decay),
    )
    scaler = _make_grad_scaler(
        enabled=_amp_enabled(cfg, dev),
        init_scale=float(getattr(cfg.train, "amp_init_scale", 1024.0)),
    )
    scheduler = U.CubicScheduler()
    _init_profile(cfg.train.profile_path)
    os.makedirs(cfg.train.save_dir, exist_ok=True)
    ckpt_best_path = os.path.join(cfg.train.save_dir, "stage_a_best.pt")

    # Load val records for in-training eval
    val_records: List[MRNARecord] = []
    eval_every = int(getattr(cfg.train, "eval_every", 0) or 0)
    val_max_eval = int(getattr(cfg.train, "val_max_eval", 0) or 0)
    val_idx_p = val_idx_path or getattr(cfg.train, "val_idx_path", None)
    if eval_every > 0:
        if val_idx_p and source_records:
            val_idx = _load_idx(val_idx_p)
            val_records = _select_records_by_idx(source_records, val_idx, val_max_eval)
            print(
                f"[p2-10] loaded {len(val_records)} val records for eval_every={eval_every}",
                file=sys.stderr,
            )
        else:
            print("[p2-10] eval_every>0 but no val_idx_path/records; skipping held-out eval", file=sys.stderr)
            eval_every = 0

    save_every = int(getattr(cfg.train, "save_every", 0) or 0)
    total_steps = int(steps if steps is not None else max(1, cfg.train.epochs))
    batch_size = max(1, int(cfg.train.batch_size))
    cursor = 0
    best_loss = float("inf")
    last_stats: Dict[str, object] = {}

    print(
        f"[p2-10] Option C: amp={_amp_enabled(cfg, dev)} batch_size={batch_size} "
        f"grad_accum={cfg.train.grad_accum} lr={target_lr} warmup_steps={warmup_steps} "
        f"grad_clip={cfg.train.grad_clip}",
        file=sys.stderr,
    )

    for step in range(1, total_steps + 1):
        # P2-10 Option C: apply LR warmup before each optimizer step
        current_lr = _apply_lr_warmup(optimizer, step, warmup_steps, target_lr)

        stats, cursor, batch_size = _run_optimizer_step(
            model=model,
            backbone=backbone,
            optimizer=optimizer,
            scaler=scaler,
            records=train_records,
            cfg=cfg,
            device=dev,
            scheduler=scheduler,
            batch_size=batch_size,
            cursor=cursor,
            step_seed=run_seed + step * 1009,
            property_bucket=None,
        )
        stats.update({"step": step, "stage": "A", "current_lr": current_lr})
        loss_value = float(stats["loss"])

        # P2-10: held-out eval
        if eval_every > 0 and step % eval_every == 0 and val_records:
            eval_stats = run_heldout_eval(
                model=model,
                backbone=backbone,
                val_records=val_records,
                cfg=cfg,
                device=dev,
                batch_size=max(1, batch_size),
                seed=run_seed + step,
            )
            stats.update(eval_stats)
            print(
                f"[p2-10] step={step} train_loss={loss_value:.4f} val_loss={eval_stats['val_loss_mean']:.4f} "
                f"lr={current_lr:.2e} grad_norm={stats.get('grad_norm', 0):.2f}",
                file=sys.stderr,
            )

        _write_profile(cfg.train.profile_path, stats)
        last_stats = stats

        # P2-10: best-loss checkpoint
        if loss_value < best_loss:
            best_loss = loss_value
            _save_stage_a_checkpoint(
                ckpt_best_path,
                cfg,
                backbone,
                model,
                step,
                best_loss,
                scientific_validity=scientific_validity,
            )

        # P2-10: periodic checkpoint
        if save_every > 0 and step % save_every == 0:
            periodic_path = os.path.join(cfg.train.save_dir, f"stage_a_step{step}.pt")
            _save_stage_a_checkpoint(
                periodic_path,
                cfg,
                backbone,
                model,
                step,
                best_loss,
                scientific_validity=scientific_validity,
            )

    return {
        "stage": "A",
        "config": cfg,
        "checkpoint_path": ckpt_best_path,
        "profile_path": cfg.train.profile_path,
        "best_loss": best_loss,
        "last_stats": last_stats,
        "scientific_validity": scientific_validity,
        "p2_10_option_c_additions": {
            "warmup_steps": warmup_steps,
            "target_lr": target_lr,
            "amp_enabled": _amp_enabled(cfg, dev),
            "effective_batch": int(batch_size) * int(cfg.train.grad_accum),
            "grad_clip": float(cfg.train.grad_clip),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P2-10 Option C: Stage A recovery with stricter fixes")
    parser.add_argument("--config", required=True, help="MEFConfig JSON path")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--profile-path", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-mode", choices=("development", "paper"), default="paper")
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--split-role", choices=("train", "val", "test"), default=None)
    parser.add_argument("--train-idx", default=None)
    parser.add_argument("--val-idx", default=None)
    parser.add_argument("--test-idx", default=None)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Linear LR warmup steps (default 500). 0 disables warmup.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    require_paper_cli_inputs(
        run_mode=args.run_mode,
        split_manifest=args.split_manifest,
        split_role=args.split_role,
        allowed_roles=("train",),
    )
    if args.run_mode == "paper":
        idx_provided = all([args.train_idx, args.val_idx, args.test_idx])
        if not args.split_manifest and not idx_provided:
            raise SystemExit(
                "paper mode requires either --split-manifest OR "
                "(--train-idx AND --val-idx AND --test-idx); aborting."
            )
    cfg = _coerce_config(args.config)
    if args.save_dir is not None:
        cfg.train.save_dir = args.save_dir
    if args.profile_path is not None:
        cfg.train.profile_path = args.profile_path
    records = load_records_jsonl(args.records_jsonl) if args.records_jsonl else None
    split_contract = (
        load_and_verify_split_manifest(args.split_manifest, records_path=args.records_jsonl)
        if args.split_manifest else None
    )
    if args.train_idx and args.val_idx and args.test_idx:
        _verify_idx_files(args, split_contract, records)
    result = train_stage_a_recovery_p2_10(
        cfg,
        records=records,
        steps=args.steps,
        device=args.device,
        seed=args.seed,
        run_mode=args.run_mode,
        split_contract=split_contract,
        split_role=args.split_role,
        train_idx_path=args.train_idx,
        val_idx_path=args.val_idx,
        test_idx_path=args.test_idx,
        warmup_steps=args.warmup_steps,
    )
    print(
        json.dumps(
            {
                "stage": result["stage"],
                "checkpoint_path": result["checkpoint_path"],
                "profile_path": result["profile_path"],
                "best_loss": result["best_loss"],
                "p2_10_option_c_additions": result["p2_10_option_c_additions"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
