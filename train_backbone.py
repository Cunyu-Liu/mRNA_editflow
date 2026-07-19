"""Stage A training entry point for mRNA-EditFlow.

The Stage A objective freezes the sequence backbone and trains the MEF
generation head with the existing Edit-Flow path:

``make_hybrid_batch -> sample_cond_pt -> rm_gap_tokens_with_aux -> forward ->
edit_flow_loss``.

The module is intentionally usable both as an importable smoke-test helper and
as a CLI. It has no GPU requirement; AMP is enabled only on CUDA and is a no-op
on CPU.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from mrna_editflow.core.config import (
    BackboneConfig,
    CouplingConfig,
    DataConfig,
    MEFConfig,
    ModelConfig,
    TrainConfig,
)
from mrna_editflow.core.constants import VOCAB_MODEL_SIZE
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.core import mrna_flow_utils as U
from mrna_editflow.data.clean_mrna import clean_corpus
from mrna_editflow.data.download_mrna import load_records_jsonl, synthesize_corpus
from mrna_editflow.data.split_contract import (
    VerifiedSplitContract,
    load_and_verify_split_manifest,
)
from mrna_editflow.eval.artifact_contract import (
    build_run_metadata,
    normalize_run_mode,
    prepare_scientific_records,
    require_paper_cli_inputs,
    validate_output_namespace,
)
from mrna_editflow.models.backbones import FrozenBackbone
from mrna_editflow.models.mrna_editformer import MRNAEditFormer


def _coerce_config(config: Optional[object]) -> MEFConfig:
    if config is None:
        return MEFConfig()
    if isinstance(config, MEFConfig):
        return config
    if isinstance(config, str):
        return MEFConfig.from_json(config)
    if isinstance(config, dict):
        return MEFConfig(
            data=DataConfig(**config.get("data", {})),
            backbone=BackboneConfig(**config.get("backbone", {})),
            model=ModelConfig(**config.get("model", {})),
            coupling=CouplingConfig(**config.get("coupling", {})),
            train=TrainConfig(**config.get("train", {})),
        )
    raise TypeError(f"unsupported config type: {type(config)!r}")


def _config_dict(cfg: MEFConfig) -> Dict[str, object]:
    if is_dataclass(cfg):
        return asdict(cfg)
    return dict(cfg)  # pragma: no cover - defensive only


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed & 0x7FFFFFFF)
    torch.manual_seed(seed)


def _resolve_device(device: Optional[object]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is not None:
        return torch.device(str(device))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _amp_enabled(cfg: MEFConfig, device: torch.device) -> bool:
    return bool(cfg.train.amp and device.type == "cuda")


def _make_grad_scaler(
    enabled: bool,
    init_scale: float = 1024.0,
    growth_interval: int = 2000,
):
    """Build an AMP GradScaler tuned for long-sequence CTMC losses.

    PyTorch's default initial scale (65536) is often too aggressive for public
    full-length mRNA batches where the Edit-Flow hazard term sums over many
    aligned positions. A smaller initial scale avoids avoidable first-step
    gradient overflows while retaining dynamic loss scaling. Complexity: O(1).
    """
    kwargs = {
        "enabled": enabled,
        "init_scale": float(init_scale),
        "growth_interval": int(growth_interval),
    }
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", **kwargs)
        except TypeError:  # pragma: no cover - older torch.amp signature
            return torch.amp.GradScaler(**kwargs)
    return torch.cuda.amp.GradScaler(**kwargs)  # pragma: no cover


def _autocast_context(enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)  # pragma: no cover


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _write_profile(path: str, row: Dict[str, object]) -> None:
    _ensure_parent(path)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def _init_profile(path: str) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("")


def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "oom" in text or "mps backend out of memory" in text


def _next_smaller_batch(batch_size: int, ladder: Sequence[int]) -> Optional[int]:
    candidates = sorted({int(x) for x in ladder if int(x) > 0} | {int(batch_size)}, reverse=True)
    for size in candidates:
        if size < batch_size:
            return size
    if batch_size > 1:
        return max(1, batch_size // 2)
    return None


def _trainable_parameters(module: torch.nn.Module) -> List[torch.nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]


def _finite_gradients(params: Iterable[torch.nn.Parameter]) -> bool:
    for p in params:
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True


def _make_records(records: Optional[Sequence[MRNARecord]], cfg: MEFConfig, synthetic_n: int) -> List[MRNARecord]:
    if records is not None:
        cleaned, _stats = clean_corpus(list(records), cfg.data)
        if cleaned:
            return cleaned
        raise ValueError("no valid records after cleaning")

    raw_n = max(int(synthetic_n) * 4, int(synthetic_n), 4)
    raw = synthesize_corpus(raw_n, seed=cfg.data.seed)
    cleaned, _stats = clean_corpus(raw, cfg.data)
    if not cleaned:
        raise ValueError("synthetic corpus produced no valid records")
    cleaned.sort(key=lambda r: len(r.seq))
    return cleaned[: max(1, int(synthetic_n))]


def _select_records(records: Sequence[MRNARecord], batch_size: int, cursor: int) -> Tuple[List[MRNARecord], int]:
    if not records:
        raise ValueError("records must be non-empty")
    batch = []
    for i in range(batch_size):
        batch.append(records[(cursor + i) % len(records)])
    return batch, cursor + batch_size


def build_stage_a_model(cfg: Optional[object] = None, device: Optional[object] = None) -> Tuple[FrozenBackbone, MRNAEditFormer]:
    """Build a frozen backbone and trainable MEF generation head."""
    mef_cfg = _coerce_config(cfg)
    dev = _resolve_device(device)
    backbone = FrozenBackbone(mef_cfg.backbone).to(dev)
    backbone.freeze()
    head = MRNAEditFormer(mef_cfg.model, backbone_dim=backbone.out_dim).to(dev)
    return backbone, head


def _aux_struct_loss(
    out: Dict[str, torch.Tensor],
    x_pad_mask: torch.Tensor,
    *,
    enabled: bool,
    target: Optional[torch.Tensor] = None,
    provenance: Optional[Mapping[str, object]] = None,
) -> torch.Tensor:
    """Return a provenance-gated auxiliary structural loss.

    The head is experimental and disabled by default.  Enabling it without a
    real target artifact is an error: an implicit all-zero tensor must never be
    used as biological supervision.
    """
    reference = out.get("rates")
    if not enabled:
        if reference is not None:
            return reference.new_zeros(())
        return torch.zeros((), device=x_pad_mask.device)
    aux = out.get("aux")
    if aux is None:
        raise ValueError("auxiliary structural supervision enabled but model returned no aux head")
    if target is None:
        raise ValueError("auxiliary structural supervision requires an explicit target tensor")
    if not isinstance(provenance, Mapping):
        raise ValueError("auxiliary structural supervision requires target provenance metadata")
    required = ("artifact_sha256", "source", "target_kind")
    missing = [key for key in required if not provenance.get(key)]
    if missing:
        raise ValueError(f"auxiliary target provenance missing fields: {', '.join(missing)}")
    if target.shape != aux.shape:
        raise ValueError(
            f"auxiliary target shape {tuple(target.shape)} does not match head {tuple(aux.shape)}"
        )
    target = target.to(device=aux.device, dtype=aux.dtype)
    valid = (~x_pad_mask).unsqueeze(-1).to(aux.dtype)
    denom = valid.sum().clamp_min(1.0)
    return ((aux - target).pow(2) * valid).sum() / denom


def _flow_batch_loss(
    model: torch.nn.Module,
    backbone: Optional[FrozenBackbone],
    records: Sequence[MRNARecord],
    cfg: MEFConfig,
    device: torch.device,
    scheduler: U.CubicScheduler,
    seed: int,
    property_bucket: Optional[int] = None,
    aux_struct_target: Optional[torch.Tensor] = None,
    aux_struct_provenance: Optional[Mapping[str, object]] = None,
) -> Dict[str, torch.Tensor]:
    hb = U.make_hybrid_batch(records, cfg.coupling, device, seed=seed)
    z_t = U.sample_cond_pt(U.x2prob(hb.z0), U.x2prob(hb.z1), hb.t, scheduler)
    x_t, x_pad, z_gap, z_pad, region_x, phase_x = U.rm_gap_tokens_with_aux(
        z_t, hb.region_ids, hb.phase_ids
    )

    if property_bucket is None:
        if backbone is None:
            raise ValueError("Stage A forward requires a backbone")
        out = model.forward(x_t, region_x, phase_x, hb.t, x_pad, backbone)
    else:
        bucket = torch.full(
            (x_t.shape[0],), int(property_bucket), dtype=torch.long, device=device
        )
        out = model.forward(x_t, region_x, phase_x, hb.t, x_pad, bucket)

    losses = U.edit_flow_loss(
        out,
        z_t,
        hb.z1,
        x_pad,
        z_gap,
        z_pad,
        hb.t,
        scheduler,
        vocab_size=VOCAB_MODEL_SIZE,
    )
    aux = _aux_struct_loss(
        out,
        x_pad,
        enabled=bool(cfg.model.use_aux_struct),
        target=aux_struct_target,
        provenance=aux_struct_provenance,
    )
    total = losses["loss"]
    if cfg.model.use_aux_struct:
        total = total + cfg.model.aux_loss_weight * aux
    return {
        "loss": total,
        "edit_loss": losses["loss"],
        "aux_loss": aux,
        "loss_ins": losses["loss_ins"],
        "loss_sub": losses["loss_sub"],
        "loss_del": losses["loss_del"],
    }


def _run_optimizer_step(
    model: torch.nn.Module,
    backbone: Optional[FrozenBackbone],
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    records: Sequence[MRNARecord],
    cfg: MEFConfig,
    device: torch.device,
    scheduler: U.CubicScheduler,
    batch_size: int,
    cursor: int,
    step_seed: int,
    property_bucket: Optional[int] = None,
) -> Tuple[Dict[str, object], int, int]:
    grad_accum = max(1, int(cfg.train.grad_accum))
    max_retries = max(0, int(cfg.train.nan_retry))
    amp = _amp_enabled(cfg, device)
    current_batch = max(1, int(batch_size))
    retries = 0
    oom_reductions = 0
    amp_fallback_used = False

    attempt = 0
    while attempt <= max_retries:
        optimizer.zero_grad(set_to_none=True)
        local_cursor = cursor
        loss_sum = 0.0
        edit_sum = 0.0
        aux_sum = 0.0
        n_samples = 0
        start = time.perf_counter()
        unscale_called = False
        try:
            for micro in range(grad_accum):
                batch_records, local_cursor = _select_records(records, current_batch, local_cursor)
                with _autocast_context(amp):
                    loss_dict = _flow_batch_loss(
                        model,
                        backbone,
                        batch_records,
                        cfg,
                        device,
                        scheduler,
                        seed=step_seed + attempt * 997 + micro,
                        property_bucket=property_bucket,
                    )
                    loss = loss_dict["loss"]
                    if not torch.isfinite(loss).all():
                        raise FloatingPointError("non-finite loss")
                    scaled_loss = loss / grad_accum
                if amp:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                loss_sum += float(loss.detach().cpu())
                edit_sum += float(loss_dict["edit_loss"].detach().cpu())
                aux_sum += float(loss_dict["aux_loss"].detach().cpu())
                n_samples += len(batch_records)

            params = _trainable_parameters(model)
            if amp:
                scaler.unscale_(optimizer)
                unscale_called = True
            if cfg.train.grad_clip and cfg.train.grad_clip > 0:
                grad_norm_t = torch.nn.utils.clip_grad_norm_(params, float(cfg.train.grad_clip))
                grad_norm = float(grad_norm_t.detach().cpu())
            else:
                grad_norm = math.sqrt(
                    sum(
                        float(p.grad.detach().pow(2).sum().cpu())
                        for p in params
                        if p.grad is not None
                    )
                )
            if not math.isfinite(grad_norm) or not _finite_gradients(params):
                raise FloatingPointError("non-finite gradients")
            if amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            elapsed = max(time.perf_counter() - start, 1e-12)
            stats: Dict[str, object] = {
                "loss": loss_sum / grad_accum,
                "edit_loss": edit_sum / grad_accum,
                "aux_loss": aux_sum / grad_accum,
                "samples_per_s": n_samples / elapsed,
                "grad_norm": grad_norm,
                "finite_loss": True,
                "finite_grad": True,
                "batch_size": current_batch,
                "retries": retries,
                "oom_reductions": oom_reductions,
                "amp_enabled": amp,
                "amp_fallback_used": amp_fallback_used,
            }
            return stats, local_cursor, current_batch
        except RuntimeError as exc:
            if not _is_oom(exc):
                raise
            next_batch = _next_smaller_batch(current_batch, cfg.train.oom_batch_ladder)
            if next_batch is None or next_batch == current_batch:
                raise
            optimizer.zero_grad(set_to_none=True)
            if amp and unscale_called:
                scaler.update()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            current_batch = next_batch
            retries += 1
            oom_reductions += 1
            attempt += 1
        except FloatingPointError:
            optimizer.zero_grad(set_to_none=True)
            if amp and unscale_called:
                scaler.update()
            retries += 1
            if amp and attempt >= max_retries and not amp_fallback_used:
                # Some full-length public batches are numerically too sharp for
                # fp16 even after scaler backoff. Preserve progress by redoing
                # the same optimizer step in fp32, then let the next step try
                # AMP again with the updated scaler.
                amp = False
                amp_fallback_used = True
                attempt = 0
                continue
            if attempt >= max_retries:
                raise
            attempt += 1

    raise FloatingPointError("failed to obtain a finite optimizer step")


def _save_stage_a_checkpoint(
    path: str,
    cfg: MEFConfig,
    backbone: FrozenBackbone,
    model: MRNAEditFormer,
    step: int,
    best_loss: float,
    aux_target_provenance: Optional[Mapping[str, object]] = None,
    scientific_validity: Optional[Mapping[str, object]] = None,
) -> None:
    _ensure_parent(path)
    torch.save(
        {
            "stage": "A",
            "step": int(step),
            "best_loss": float(best_loss),
            "config": _config_dict(cfg),
            "aux_supervision": {
                "enabled": bool(cfg.model.use_aux_struct),
                "loss_weight": float(cfg.model.aux_loss_weight),
                "target_provenance": (
                    dict(aux_target_provenance) if aux_target_provenance is not None else None
                ),
            },
            "scientific_validity": dict(scientific_validity or {}),
            "backbone_state": backbone.state_dict(),
            "model_state": model.state_dict(),
        },
        path,
    )


def train_stage_a(
    config: Optional[object] = None,
    records: Optional[Sequence[MRNARecord]] = None,
    steps: Optional[int] = None,
    synthetic_n: int = 8,
    device: Optional[object] = None,
    seed: Optional[int] = None,
    run_mode: str = "development",
    split_contract: Optional[VerifiedSplitContract] = None,
    split_role: Optional[str] = None,
) -> Dict[str, object]:
    """Run Stage A training and return model/checkpoint/profile handles.

    Parameters are intentionally small and explicit so tests can run a 2-3 step
    CPU smoke train without invoking a full experiment runner.
    """
    cfg = _coerce_config(config)
    run_mode = normalize_run_mode(run_mode)
    validate_output_namespace(cfg.train.save_dir, run_mode)
    validate_output_namespace(cfg.train.profile_path, run_mode)
    run_seed = int(cfg.data.seed if seed is None else seed)
    _set_seed(run_seed)
    dev = _resolve_device(device)
    source_records = _make_records(records, cfg, synthetic_n=synthetic_n)
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

    backbone, model = build_stage_a_model(cfg, dev)
    model.train()
    backbone.eval()
    optimizer = torch.optim.AdamW(
        _trainable_parameters(model),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    scaler = _make_grad_scaler(enabled=_amp_enabled(cfg, dev))
    scheduler = U.CubicScheduler()
    _init_profile(cfg.train.profile_path)
    os.makedirs(cfg.train.save_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.train.save_dir, "stage_a_best.pt")

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
        _write_profile(cfg.train.profile_path, stats)
        last_stats = stats
        loss_value = float(stats["loss"])
        if loss_value < best_loss:
            best_loss = loss_value
            _save_stage_a_checkpoint(
                ckpt_path,
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
        "records": train_records,
        "backbone": backbone,
        "model": model,
        "checkpoint_path": ckpt_path,
        "profile_path": cfg.train.profile_path,
        "best_loss": best_loss,
        "last_stats": last_stats,
        "scientific_validity": scientific_validity,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train mRNA-EditFlow Stage A head")
    parser.add_argument("--config", default=None, help="MEFConfig JSON path")
    parser.add_argument("--steps", type=int, default=None, help="optimizer steps")
    parser.add_argument("--synthetic-n", type=int, default=8, help="synthetic records for smoke runs")
    parser.add_argument("--records-jsonl", default=None, help="cleaned MRNARecord JSONL from public_pipeline")
    parser.add_argument("--save-dir", default=None, help="override checkpoint directory")
    parser.add_argument("--profile-path", default=None, help="override profile JSONL path")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-mode", choices=("development", "paper"), default="development")
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--split-role", choices=("train", "val", "test"), default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    require_paper_cli_inputs(
        run_mode=args.run_mode,
        split_manifest=args.split_manifest,
        split_role=args.split_role,
        allowed_roles=("train",),
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
    result = train_stage_a(
        cfg,
        records=records,
        steps=args.steps,
        synthetic_n=args.synthetic_n,
        device=args.device,
        seed=args.seed,
        run_mode=args.run_mode,
        split_contract=split_contract,
        split_role=args.split_role,
    )
    print(
        json.dumps(
            {
                "stage": result["stage"],
                "checkpoint_path": result["checkpoint_path"],
                "profile_path": result["profile_path"],
                "best_loss": result["best_loss"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
