"""Stage B adapter training entry point for mRNA-EditFlow.

Stage B freezes the backbone and the pretrained Stage A generation head, then
trains only ``AdapterWrappedMEF`` adapters plus task heads for task ids T2-T7.
The training path intentionally reuses the same Edit-Flow loss as Stage A so a
CPU smoke run exercises the real model contract without external dependencies.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

import torch

from mrna_editflow.core.config import MEFConfig
from mrna_editflow.core.constants import REGION_3UTR, REGION_5UTR, REGION_CDS
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.core import mrna_flow_utils as U
from mrna_editflow.models.adapters import AdapterWrappedMEF
from mrna_editflow.models.backbones import FrozenBackbone
from mrna_editflow.models.mrna_editformer import MRNAEditFormer
from mrna_editflow.models.region_adapters import RegionSpecializedEditFormer
from mrna_editflow.data.download_mrna import load_records_jsonl
from mrna_editflow.data.split_contract import (
    VerifiedSplitContract,
    load_and_verify_split_manifest,
    sha256_file,
)
from mrna_editflow.eval.artifact_contract import (
    build_run_metadata,
    normalize_run_mode,
    prepare_scientific_records,
    require_paper_cli_inputs,
    validate_output_namespace,
    verify_paper_checkpoint,
)
from mrna_editflow.train_backbone import (
    _amp_enabled,
    _coerce_config,
    _config_dict,
    _ensure_parent,
    _init_profile,
    _make_grad_scaler,
    _make_records,
    _resolve_device,
    _run_optimizer_step,
    _set_seed,
    _trainable_parameters,
    _write_profile,
)

_TASK_BUCKETS = {f"T{i}": i - 2 for i in range(2, 8)}
_ALLOWED_TRAINABLE_PREFIXES = (
    "adapters.",
    "property_emb.",
    "property_film.",
    "task_norm.",
    "rates_out.",
    "ins_logits.",
    "sub_logits.",
    "aux_struct.",
)
_ALLOWED_REGION_TRAINABLE_PREFIXES = ("adapters.",)
_REGION_NAME_TO_ID = {
    "5utr": REGION_5UTR,
    "utr5": REGION_5UTR,
    "cds": REGION_CDS,
    "3utr": REGION_3UTR,
    "utr3": REGION_3UTR,
}


def task_id_to_bucket(task_id: str) -> int:
    tid = task_id.upper()
    if tid not in _TASK_BUCKETS:
        raise ValueError(f"Stage B supports task_id T2-T7, got {task_id!r}")
    return _TASK_BUCKETS[tid]


def _named_parameter_snapshots(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: p.detach().cpu().clone() for name, p in module.named_parameters()}


def _changed_names(before: Dict[str, torch.Tensor], module: torch.nn.Module) -> List[str]:
    changed: List[str] = []
    for name, p in module.named_parameters():
        old = before.get(name)
        if old is None:
            continue
        new = p.detach().cpu()
        if old.shape != new.shape or not torch.equal(old, new):
            changed.append(name)
    return changed


def _trainable_names(module: torch.nn.Module) -> List[str]:
    return [name for name, p in module.named_parameters() if p.requires_grad]


def assert_only_adapter_trainable(module: AdapterWrappedMEF) -> None:
    """Raise if any frozen base/backbone parameter is accidentally trainable."""
    bad = [
        name
        for name, p in module.named_parameters()
        if p.requires_grad and not name.startswith(_ALLOWED_TRAINABLE_PREFIXES)
    ]
    if bad:
        raise AssertionError(f"unexpected trainable parameters: {bad[:8]}")
    frozen_bad = [
        name
        for name, p in module.named_parameters()
        if (name.startswith("backbone.") or name.startswith("base_head.")) and p.requires_grad
    ]
    if frozen_bad:
        raise AssertionError(f"frozen modules are trainable: {frozen_bad[:8]}")


def assert_only_region_adapter_trainable(module: RegionSpecializedEditFormer) -> None:
    """Raise if any frozen base-head parameter is accidentally trainable."""
    bad = [
        name
        for name, p in module.named_parameters()
        if p.requires_grad and not name.startswith(_ALLOWED_REGION_TRAINABLE_PREFIXES)
    ]
    if bad:
        raise AssertionError(f"unexpected trainable parameters: {bad[:8]}")
    frozen_bad = [
        name for name, p in module.named_parameters()
        if name.startswith("base.") and p.requires_grad
    ]
    if frozen_bad:
        raise AssertionError(f"frozen base head is trainable: {frozen_bad[:8]}")


def _parse_region_ids(value: Optional[str]) -> Optional[list[int]]:
    """Parse comma-separated region names/ids for region-specialized adapters."""
    if value is None or value.strip() == "":
        return None
    parsed: list[int] = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item in _REGION_NAME_TO_ID:
            parsed.append(_REGION_NAME_TO_ID[item])
        else:
            try:
                parsed.append(int(item))
            except ValueError as exc:
                allowed = ", ".join(sorted(_REGION_NAME_TO_ID))
                raise ValueError(f"unknown region {raw!r}; use one of {allowed} or numeric ids") from exc
    return parsed or None


def build_stage_b_model(
    config: Optional[object] = None,
    device: Optional[object] = None,
    adapter_rank: int = 4,
    num_property_buckets: int = 8,
    base_checkpoint: Optional[str] = None,
) -> AdapterWrappedMEF:
    cfg = _coerce_config(config)
    dev = _resolve_device(device)
    backbone = FrozenBackbone(cfg.backbone).to(dev)
    backbone.freeze()
    base_head = MRNAEditFormer(cfg.model, backbone_dim=backbone.out_dim).to(dev)

    if base_checkpoint:
        try:
            ckpt = torch.load(base_checkpoint, map_location=dev, weights_only=False)
        except TypeError:  # pragma: no cover - older torch without weights_only
            ckpt = torch.load(base_checkpoint, map_location=dev)
        if "backbone_state" in ckpt:
            backbone.load_state_dict(ckpt["backbone_state"], strict=False)
        if "model_state" in ckpt:
            base_head.load_state_dict(ckpt["model_state"], strict=False)

    wrapped = AdapterWrappedMEF(
        backbone,
        base_head,
        adapter_rank=int(adapter_rank),
        num_property_buckets=int(num_property_buckets),
    ).to(dev)
    assert_only_adapter_trainable(wrapped)
    return wrapped


def build_region_stage_b_model(
    config: Optional[object] = None,
    device: Optional[object] = None,
    adapter_bottleneck: int = 32,
    base_checkpoint: Optional[str] = None,
    regions: Optional[Sequence[int]] = None,
) -> tuple[FrozenBackbone, RegionSpecializedEditFormer]:
    """Build a frozen Stage A head plus trainable per-region adapters."""
    cfg = _coerce_config(config)
    dev = _resolve_device(device)
    backbone = FrozenBackbone(cfg.backbone).to(dev)
    backbone.freeze()
    base_head = MRNAEditFormer(cfg.model, backbone_dim=backbone.out_dim).to(dev)

    if base_checkpoint:
        try:
            ckpt = torch.load(base_checkpoint, map_location=dev, weights_only=False)
        except TypeError:  # pragma: no cover - older torch without weights_only
            ckpt = torch.load(base_checkpoint, map_location=dev)
        if "backbone_state" in ckpt:
            backbone.load_state_dict(ckpt["backbone_state"], strict=False)
        if "model_state" in ckpt:
            base_head.load_state_dict(ckpt["model_state"], strict=False)

    wrapped = RegionSpecializedEditFormer(
        base_head,
        bottleneck=int(adapter_bottleneck),
        regions=regions,
        freeze_base=True,
    ).to(dev)
    assert_only_region_adapter_trainable(wrapped)
    return backbone, wrapped


def _save_stage_b_checkpoint(
    path: str,
    cfg: MEFConfig,
    model: AdapterWrappedMEF,
    task_id: str,
    step: int,
    best_loss: float,
    scientific_validity: Optional[dict[str, object]] = None,
) -> None:
    _ensure_parent(path)
    torch.save(
        {
            "stage": "B",
            "task_id": task_id,
            "step": int(step),
            "best_loss": float(best_loss),
            "config": _config_dict(cfg),
            "adapter_state": model.state_dict(),
            "trainable_names": _trainable_names(model),
            "scientific_validity": dict(scientific_validity or {}),
        },
        path,
    )


def _save_region_stage_b_checkpoint(
    path: str,
    cfg: MEFConfig,
    backbone: FrozenBackbone,
    model: RegionSpecializedEditFormer,
    task_id: str,
    step: int,
    best_loss: float,
    scientific_validity: Optional[dict[str, object]] = None,
) -> None:
    _ensure_parent(path)
    torch.save(
        {
            "stage": "B_region",
            "task_id": task_id,
            "step": int(step),
            "best_loss": float(best_loss),
            "config": _config_dict(cfg),
            "backbone_state": backbone.state_dict(),
            "model_state": model.state_dict(),
            "region_ids": list(model.region_ids),
            "adapter_bottleneck": int(model.bottleneck),
            "trainable_names": _trainable_names(model),
            "scientific_validity": dict(scientific_validity or {}),
        },
        path,
    )


def train_stage_b(
    config: Optional[object] = None,
    records: Optional[Sequence[MRNARecord]] = None,
    task_id: str = "T2",
    steps: Optional[int] = None,
    synthetic_n: int = 8,
    device: Optional[object] = None,
    seed: Optional[int] = None,
    adapter_rank: int = 4,
    base_checkpoint: Optional[str] = None,
    run_mode: str = "development",
    split_contract: Optional[VerifiedSplitContract] = None,
    split_role: Optional[str] = None,
) -> Dict[str, object]:
    """Run a small Stage B adapter training job for task ids T2-T7."""
    cfg = _coerce_config(config)
    run_mode = normalize_run_mode(run_mode)
    validate_output_namespace(cfg.train.save_dir, run_mode)
    validate_output_namespace(cfg.train.profile_path, run_mode)
    tid = task_id.upper()
    bucket = task_id_to_bucket(tid)
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
    if run_mode == "paper":
        if base_checkpoint is None:
            raise ValueError("paper Stage B requires a verified Stage A base checkpoint")
        verify_paper_checkpoint(base_checkpoint, data_provenance)
    upstream = {
        "base_checkpoint_sha256": sha256_file(base_checkpoint) if base_checkpoint else None
    }
    scientific_validity = build_run_metadata(
        run_mode=run_mode,
        data_provenance=data_provenance,
        config=_config_dict(cfg),
        code_paths=(__file__,),
        training_seed=run_seed,
        upstream=upstream,
    )

    model = build_stage_b_model(
        cfg,
        device=dev,
        adapter_rank=adapter_rank,
        num_property_buckets=max(8, bucket + 1),
        base_checkpoint=base_checkpoint,
    )
    before = _named_parameter_snapshots(model)
    trainable_before = set(_trainable_names(model))
    assert_only_adapter_trainable(model)

    model.train()
    model.backbone.eval()
    model.base_head.eval()
    optimizer = torch.optim.AdamW(
        _trainable_parameters(model),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    scaler = _make_grad_scaler(enabled=_amp_enabled(cfg, dev))
    scheduler = U.CubicScheduler()
    _init_profile(cfg.train.profile_path)
    os.makedirs(cfg.train.save_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.train.save_dir, f"stage_b_{tid.lower()}_best.pt")

    total_steps = int(steps if steps is not None else max(1, cfg.train.epochs))
    batch_size = max(1, int(cfg.train.batch_size))
    cursor = 0
    best_loss = float("inf")
    last_stats: Dict[str, object] = {}

    for step in range(1, total_steps + 1):
        # Keep frozen dropout/norm behavior fixed even after model.train().
        model.backbone.eval()
        model.base_head.eval()
        stats, cursor, batch_size = _run_optimizer_step(
            model=model,
            backbone=None,
            optimizer=optimizer,
            scaler=scaler,
            records=train_records,
            cfg=cfg,
            device=dev,
            scheduler=scheduler,
            batch_size=batch_size,
            cursor=cursor,
            step_seed=run_seed + step * 1543,
            property_bucket=bucket,
        )
        stats.update({"step": step, "stage": "B", "task_id": tid, "property_bucket": bucket})
        _write_profile(cfg.train.profile_path, stats)
        last_stats = stats
        loss_value = float(stats["loss"])
        if loss_value < best_loss:
            best_loss = loss_value
            _save_stage_b_checkpoint(
                ckpt_path, cfg, model, tid, step, best_loss,
                scientific_validity=scientific_validity,
            )

    changed = _changed_names(before, model)
    changed_trainable = [name for name in changed if name in trainable_before]
    changed_frozen = [name for name in changed if name not in trainable_before]
    assert_only_adapter_trainable(model)

    return {
        "stage": "B",
        "task_id": tid,
        "property_bucket": bucket,
        "config": cfg,
        "records": train_records,
        "model": model,
        "checkpoint_path": ckpt_path,
        "profile_path": cfg.train.profile_path,
        "best_loss": best_loss,
        "last_stats": last_stats,
        "trainable_names": sorted(trainable_before),
        "changed_trainable_names": sorted(changed_trainable),
        "changed_frozen_names": sorted(changed_frozen),
        "scientific_validity": scientific_validity,
    }


def train_region_stage_b(
    config: Optional[object] = None,
    records: Optional[Sequence[MRNARecord]] = None,
    task_id: str = "T5",
    steps: Optional[int] = None,
    synthetic_n: int = 8,
    device: Optional[object] = None,
    seed: Optional[int] = None,
    adapter_bottleneck: int = 32,
    base_checkpoint: Optional[str] = None,
    regions: Optional[Sequence[int]] = None,
    run_mode: str = "development",
    split_contract: Optional[VerifiedSplitContract] = None,
    split_role: Optional[str] = None,
) -> Dict[str, object]:
    """Run Stage B training for per-region adapters over a frozen Stage A head."""
    cfg = _coerce_config(config)
    run_mode = normalize_run_mode(run_mode)
    validate_output_namespace(cfg.train.save_dir, run_mode)
    validate_output_namespace(cfg.train.profile_path, run_mode)
    tid = task_id.upper()
    task_id_to_bucket(tid)  # validates the same T2-T7 downstream task contract.
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
    if run_mode == "paper":
        if base_checkpoint is None:
            raise ValueError("paper Stage B requires a verified Stage A base checkpoint")
        verify_paper_checkpoint(base_checkpoint, data_provenance)
    upstream = {
        "base_checkpoint_sha256": sha256_file(base_checkpoint) if base_checkpoint else None
    }
    scientific_validity = build_run_metadata(
        run_mode=run_mode,
        data_provenance=data_provenance,
        config=_config_dict(cfg),
        code_paths=(__file__,),
        training_seed=run_seed,
        upstream=upstream,
    )

    backbone, model = build_region_stage_b_model(
        cfg,
        device=dev,
        adapter_bottleneck=adapter_bottleneck,
        base_checkpoint=base_checkpoint,
        regions=regions,
    )
    before = _named_parameter_snapshots(model)
    trainable_before = set(_trainable_names(model))
    assert_only_region_adapter_trainable(model)

    model.train()
    backbone.eval()
    model.base.eval()
    optimizer = torch.optim.AdamW(
        _trainable_parameters(model),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    scaler = _make_grad_scaler(enabled=_amp_enabled(cfg, dev))
    scheduler = U.CubicScheduler()
    _init_profile(cfg.train.profile_path)
    os.makedirs(cfg.train.save_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.train.save_dir, f"stage_b_region_{tid.lower()}_best.pt")

    total_steps = int(steps if steps is not None else max(1, cfg.train.epochs))
    batch_size = max(1, int(cfg.train.batch_size))
    cursor = 0
    best_loss = float("inf")
    last_stats: Dict[str, object] = {}

    for step in range(1, total_steps + 1):
        backbone.eval()
        model.base.eval()
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
            step_seed=run_seed + step * 1777,
            property_bucket=None,
        )
        stats.update({
            "step": step,
            "stage": "B_region",
            "task_id": tid,
            "region_ids": list(model.region_ids),
        })
        _write_profile(cfg.train.profile_path, stats)
        last_stats = stats
        loss_value = float(stats["loss"])
        if loss_value < best_loss:
            best_loss = loss_value
            _save_region_stage_b_checkpoint(
                ckpt_path, cfg, backbone, model, tid, step, best_loss,
                scientific_validity=scientific_validity,
            )

    changed = _changed_names(before, model)
    changed_trainable = [name for name in changed if name in trainable_before]
    changed_frozen = [name for name in changed if name not in trainable_before]
    assert_only_region_adapter_trainable(model)

    return {
        "stage": "B_region",
        "task_id": tid,
        "config": cfg,
        "records": train_records,
        "backbone": backbone,
        "model": model,
        "checkpoint_path": ckpt_path,
        "profile_path": cfg.train.profile_path,
        "best_loss": best_loss,
        "last_stats": last_stats,
        "region_ids": list(model.region_ids),
        "trainable_names": sorted(trainable_before),
        "changed_trainable_names": sorted(changed_trainable),
        "changed_frozen_names": sorted(changed_frozen),
        "scientific_validity": scientific_validity,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train mRNA-EditFlow Stage B adapters")
    parser.add_argument("--config", default=None, help="MEFConfig JSON path")
    parser.add_argument("--task-id", default="T2", choices=sorted(_TASK_BUCKETS))
    parser.add_argument("--adapter-kind", default="block", choices=("block", "region"))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--synthetic-n", type=int, default=8)
    parser.add_argument("--records-jsonl", default=None, help="cleaned MRNARecord JSONL from public_pipeline")
    parser.add_argument("--adapter-rank", type=int, default=4)
    parser.add_argument("--adapter-bottleneck", type=int, default=32)
    parser.add_argument("--regions", default=None, help="comma-separated region names/ids for --adapter-kind region")
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--profile-path", default=None)
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
    if args.adapter_kind == "region":
        result = train_region_stage_b(
            cfg,
            records=records,
            task_id=args.task_id,
            steps=args.steps,
            synthetic_n=args.synthetic_n,
            device=args.device,
            seed=args.seed,
            adapter_bottleneck=args.adapter_bottleneck,
            base_checkpoint=args.base_checkpoint,
            regions=_parse_region_ids(args.regions),
            run_mode=args.run_mode,
            split_contract=split_contract,
            split_role=args.split_role,
        )
    else:
        result = train_stage_b(
            cfg,
            records=records,
            task_id=args.task_id,
            steps=args.steps,
            synthetic_n=args.synthetic_n,
            device=args.device,
            seed=args.seed,
            adapter_rank=args.adapter_rank,
            base_checkpoint=args.base_checkpoint,
            run_mode=args.run_mode,
            split_contract=split_contract,
            split_role=args.split_role,
        )
    print(
        json.dumps(
            {
                "stage": result["stage"],
                "task_id": result["task_id"],
                "checkpoint_path": result["checkpoint_path"],
                "profile_path": result["profile_path"],
                "best_loss": result["best_loss"],
                "changed_frozen_names": result["changed_frozen_names"],
                "region_ids": result.get("region_ids"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
