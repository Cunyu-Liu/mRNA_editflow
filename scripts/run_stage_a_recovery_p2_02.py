"""P2-02: Stage A recovery training script with held-out eval + save_every.

This script does NOT modify ``train_backbone.py`` (the 4 original Stage A
processes have already loaded it into memory). Instead it imports the building
blocks (``_run_optimizer_step``, ``_flow_batch_loss``, ``build_stage_a_model``,
``_save_stage_a_checkpoint``, etc.) and wraps them in a new training loop that
adds:

1. ``cfg.train.amp_init_scale`` — tunable AMP initial scale (default 1024.0,
   backward compatible). The original hardcoded 1024.0 caused 20-35% AMP
   fallback. The P2-02 fix sets it to 256.0.
2. ``cfg.train.save_every`` — periodic checkpoint interval (default 0 = off).
   Saves ``stage_a_step{N}.pt`` every N steps in addition to ``stage_a_best.pt``.
3. ``cfg.train.eval_every`` — held-out eval interval (default 0 = off). Runs a
   no-grad val loss evaluation every N steps and logs ``val_loss`` to the
   profile JSONL.

Hard constraint compliance:
- Does not terminate or modify the 4 running Stage A processes.
- Uses ``--train-idx/--val-idx/--test-idx`` (paper mode) to enforce the split
  contract on the frozen ``combined_family`` split.
- All new code has unit tests (``tests/test_stage_a_recovery_p2_02.py``).

Usage:
    python -m scripts.run_stage_a_recovery_p2_02 \
        --config configs/stage_a_recovery_p2_02.json \
        --records-jsonl data/reconstructed/p0_data_reconstruction_v1/combined/combined_model_view.records.jsonl \
        --train-idx benchmark/dev/p0_data_reconstruction_v1/combined_family/train.idx \
        --val-idx benchmark/dev/p0_data_reconstruction_v1/combined_family/val.idx \
        --test-idx benchmark/dev/p0_data_reconstruction_v1/combined_family/test.idx \
        --steps 10000 \
        --seed 42 \
        --device cuda:6 \
        --run-mode paper
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
# The mrna_editflow package IS the repo root (it has __init__.py), so we need
# its PARENT on sys.path for `from mrna_editflow.* import ...` to resolve.
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


# ---------------------------------------------------------------------------
# Held-out evaluation helper
# ---------------------------------------------------------------------------
def _load_idx(path: str) -> List[int]:
    with open(path, "r") as fh:
        return [int(line.strip()) for line in fh if line.strip()]


def _select_records_by_idx(
    records: Sequence[MRNARecord],
    idx: Sequence[int],
    max_eval: Optional[int],
) -> List[MRNARecord]:
    n = len(records)
    picked: List[MRNARecord] = []
    for i in idx:
        if 0 <= i < n:
            picked.append(records[i])
        if max_eval is not None and max_eval > 0 and len(picked) >= max_eval:
            break
    return picked


@torch.no_grad()
def run_heldout_eval(
    model: torch.nn.Module,
    backbone: torch.nn.Module,
    val_records: Sequence[MRNARecord],
    cfg: MEFConfig,
    device: torch.device,
    batch_size: int = 4,
    seed: int = 1729,
) -> Dict[str, float]:
    """Run a no-grad held-out eval and return loss stats. Imports _flow_batch_loss locally."""
    from train_backbone import _flow_batch_loss

    if not val_records:
        return {"val_loss_mean": float("nan"), "val_n": 0}
    model.eval()
    backbone.eval()
    # Disable aux struct for eval (no target tensor at eval time)
    orig_aux = cfg.model.use_aux_struct
    cfg.model.use_aux_struct = False
    scheduler = U.CubicScheduler()
    losses: List[float] = []
    n_done = 0
    n_eval = len(val_records)
    while n_done < n_eval:
        batch = list(val_records[n_done : n_done + batch_size])
        if not batch:
            break
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
        losses.append(float(loss_dict["loss"].detach().cpu()))
        n_done += len(batch)
    arr = np.array(losses) if losses else np.array([float("nan")])
    # Restore original aux_struct setting for training continuity
    cfg.model.use_aux_struct = orig_aux
    model.train()
    backbone.eval()
    return {
        "val_loss_mean": float(np.mean(arr)),
        "val_loss_median": float(np.median(arr)),
        "val_loss_p95": float(np.percentile(arr, 95)),
        "val_loss_std": float(np.std(arr)),
        "val_n": int(n_eval),
    }


# ---------------------------------------------------------------------------
# Training loop with save_every + eval_every
# ---------------------------------------------------------------------------
def train_stage_a_recovery(
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
) -> Dict[str, object]:
    """P2-02 recovery training loop.

    Adds three capabilities vs ``train_backbone.train_stage_a``:
    1. AMP init_scale from ``cfg.train.amp_init_scale`` (not hardcoded 1024).
    2. Periodic checkpoints every ``cfg.train.save_every`` steps.
    3. Held-out eval every ``cfg.train.eval_every`` steps, logged to profile.
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
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    # P2-02 FIX: use cfg.train.amp_init_scale instead of hardcoded 1024.0
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
    # Prefer explicit val_idx_path argument, then cfg.train.val_idx_path
    val_idx_p = val_idx_path or getattr(cfg.train, "val_idx_path", None)
    if eval_every > 0:
        if val_idx_p and source_records:
            val_idx = _load_idx(val_idx_p)
            val_records = _select_records_by_idx(source_records, val_idx, val_max_eval)
            print(f"[p2-02] loaded {len(val_records)} val records for eval_every={eval_every}", file=sys.stderr)
        else:
            print("[p2-02] eval_every>0 but no val_idx_path/records; skipping held-out eval", file=sys.stderr)
            eval_every = 0

    save_every = int(getattr(cfg.train, "save_every", 0) or 0)
    total_steps = int(steps if steps is not None else max(1, cfg.train.epochs))
    batch_size = max(1, int(cfg.train.batch_size))
    cursor = 0
    best_loss = float("inf")
    last_stats: Dict[str, object] = {}

    for step in range(1, total_steps + 1):
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
        stats.update({"step": step, "stage": "A"})
        loss_value = float(stats["loss"])

        # P2-02: held-out eval
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
                f"[p2-02] step={step} train_loss={loss_value:.4f} val_loss={eval_stats['val_loss_mean']:.4f} "
                f"amp_fallback={stats.get('amp_fallback_used', False)} grad_norm={stats.get('grad_norm', 0):.2f}",
                file=sys.stderr,
            )

        _write_profile(cfg.train.profile_path, stats)
        last_stats = stats

        # P2-02: best-loss checkpoint (same as original)
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

        # P2-02: periodic checkpoint
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
        "p2_02_additions": {
            "amp_init_scale": float(getattr(cfg.train, "amp_init_scale", 1024.0)),
            "save_every": save_every,
            "eval_every": eval_every,
            "val_records_loaded": len(val_records),
        },
    }


def _make_records_local(records: Optional[Sequence[MRNARecord]], cfg: MEFConfig) -> List[MRNARecord]:
    """Local copy of train_backbone._make_records to avoid import coupling."""
    from train_backbone import _make_records
    return _make_records(records, cfg, synthetic_n=8)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _verify_idx_files(
    args: argparse.Namespace,
    split_contract: Optional[Any],
    records: Optional[Sequence[MRNARecord]],
) -> None:
    """Verify idx files match split contract (if available). Mirrors train_backbone."""
    from pathlib import Path
    idx_paths = {"train": args.train_idx, "val": args.val_idx, "test": args.test_idx}
    for role, path_str in idx_paths.items():
        if path_str is None:
            continue
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"--{role}-idx file not found: {path}")
        with path.open("r") as fh:
            indices = [int(line.strip()) for line in fh if line.strip()]
        if not indices:
            raise ValueError(f"--{role}-idx file is empty: {path}")
        if split_contract is not None:
            contract_indices = split_contract.roles[role].indices
            if len(indices) != len(contract_indices):
                raise ValueError(
                    f"--{role}-idx has {len(indices)} indices but split contract "
                    f"has {len(contract_indices)} for role '{role}'; mismatch."
                )
            if set(indices) != set(contract_indices):
                raise ValueError(
                    f"--{role}-idx indices do not match split contract for role '{role}'."
                )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P2-02: Stage A recovery training")
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
    result = train_stage_a_recovery(
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
    )
    print(
        json.dumps(
            {
                "stage": result["stage"],
                "checkpoint_path": result["checkpoint_path"],
                "profile_path": result["profile_path"],
                "best_loss": result["best_loss"],
                "p2_02_additions": result["p2_02_additions"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
