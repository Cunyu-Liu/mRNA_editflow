#!/usr/bin/env python3
"""Fair, fail-closed training/evaluation runner for Phase 2.

Training only opens ``train`` and ``val`` manifests.  Final roles are handled
by ``scripts/evaluate_phase2_oracle.py`` with an explicit freeze manifest.
The five recipes are controls, while ``pretrain_finetune_calibrate`` is the
registered two-stage route:

    measured_only, proxy_only, mixed_training,
    pretrain_finetune, pretrain_finetune_calibrate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from mrna_editflow.data.nmi_benchmark_v2 import load_manifest, manifest_sha256
from mrna_editflow.models.context_encoder import context_feature_tensors
from mrna_editflow.models.paired_delta_former import PairedDeltaFormer


NUC = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
REGION = {"five_utr": 0, "cds_first30": 1, "cds_first50": 2, "cds_remaining": 3, "joint_5utr_cds": 4}
RECIPES = (
    "measured_only", "proxy_only", "mixed_training",
    "pretrain_finetune", "pretrain_finetune_calibrate",
)


def file_sha256(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_foundation_provenance(args: argparse.Namespace, backbone: str) -> Optional[str]:
    if backbone == "small":
        return None
    if not args.foundation_path:
        if args.allow_foundation_stub:
            return None
        raise RuntimeError("foundation_path is required for a scientific foundation run")
    actual = file_sha256(args.foundation_path)
    if args.foundation_sha256 and actual != args.foundation_sha256:
        raise RuntimeError(f"foundation SHA256 mismatch: expected {args.foundation_sha256}, got {actual}")
    if not args.foundation_sha256 and not args.allow_foundation_stub:
        raise RuntimeError("foundation_sha256 is required for a scientific foundation run")
    return actual


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_seq(seq: str, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    values = [NUC.get(c, 4) for c in seq[:max_len]]
    mask = [True] * len(values)
    values += [4] * (max_len - len(values))
    mask += [False] * (max_len - len(mask))
    return torch.tensor(values, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


def _cds_start(row: dict) -> int:
    explicit = row.get("cds_start")
    if explicit is not None:
        return int(explicit)
    source = str(row.get("source_sequence", ""))
    return source.find("AUG") if "AUG" in source else len(source)


def encode_edits(row: dict, max_edits: int) -> torch.Tensor:
    cds_start = _cds_start(row)
    encoded = []
    for edit in list(row.get("edit_list", []))[:max_edits]:
        pos = int(edit.get("pos", 0))
        encoded.append([
            REGION.get(str(edit.get("region")), 0), pos,
            NUC.get(str(edit.get("ref", "A")).upper(), 0),
            NUC.get(str(edit.get("alt", "A")).upper(), 0),
            pos - cds_start,
        ])
    encoded += [[-1, -1, -1, -1, 0]] * (max_edits - len(encoded))
    return torch.tensor(encoded, dtype=torch.long)


class DeltaDataset(Dataset):
    def __init__(self, rows: Sequence[dict], max_len: int = 256, max_edits: int = 10):
        self.rows = list(rows)
        self.max_len = max_len
        self.max_edits = max_edits

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        source, source_mask = encode_seq(str(row["source_sequence"]), self.max_len)
        candidate, candidate_mask = encode_seq(str(row["candidate_sequence"]), self.max_len)
        source_value = row.get("measured_or_proxy_source_value")
        if source_value is None:
            source_value = row.get("source_value", 0.0)
        return {
            "source_tokens": source,
            "source_mask": source_mask,
            "candidate_tokens": candidate,
            "candidate_mask": candidate_mask,
            "edit_tokens": encode_edits(row, self.max_edits),
            "context_row": row,
            "source_value": torch.tensor(float(source_value or 0.0), dtype=torch.float32),
            "delta": torch.tensor(float(row["delta"]), dtype=torch.float32),
        }


def collate(batch: Sequence[dict]) -> dict:
    context = context_feature_tensors([x["context_row"] for x in batch], torch.device("cpu"))
    return {
        "source_tokens": torch.stack([x["source_tokens"] for x in batch]),
        "source_mask": torch.stack([x["source_mask"] for x in batch]),
        "candidate_tokens": torch.stack([x["candidate_tokens"] for x in batch]),
        "candidate_mask": torch.stack([x["candidate_mask"] for x in batch]),
        "edit_tokens": torch.stack([x["edit_tokens"] for x in batch]),
        **context,
        "source_value": torch.stack([x["source_value"] for x in batch]),
        "delta": torch.stack([x["delta"] for x in batch]),
    }


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def paired_delta_loss(out: dict[str, torch.Tensor], target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Huber + pairwise ranking + benefit BCE + heteroscedastic NLL + Brier."""
    mean, logvar = out["mean"], out["logvar"]
    huber = F.huber_loss(mean, target)
    nll = 0.5 * (logvar + (target - mean).pow(2) / logvar.exp()).mean()
    label = (target > 0).float()
    beneficial = F.binary_cross_entropy_with_logits(out["beneficial_logit"], label)
    pair_delta = target[:, None] - target[None, :]
    pair_mask = (pair_delta.abs() > 1e-8) & ~torch.eye(target.numel(), dtype=torch.bool, device=target.device)
    if pair_mask.any():
        signs = pair_delta[pair_mask].sign()
        score_delta = out["rank"][:, None] - out["rank"][None, :]
        ranking = F.softplus(-signs * score_delta[pair_mask]).mean()
    else:
        ranking = huber * 0.0
    brier = (torch.sigmoid(out["beneficial_logit"]) - label).pow(2).mean()
    loss = huber + 0.20 * ranking + 0.20 * beneficial + 0.20 * nll + 0.10 * brier
    return loss, {
        "huber": float(huber.detach()), "ranking": float(ranking.detach()),
        "beneficial": float(beneficial.detach()), "nll": float(nll.detach()),
        "calibration": float(brier.detach()), "total": float(loss.detach()),
    }


class DeltaCalibrator(nn.Module):
    """Measured-only post-hoc calibration for variance and benefit probability."""

    def __init__(self) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.logvar_bias = nn.Parameter(torch.zeros(()))
        self.log_temperature = nn.Parameter(torch.zeros(()))

    def calibrate(self, out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result = dict(out)
        result["logvar"] = out["logvar"] + self.log_scale + self.logvar_bias
        result["variance"] = result["logvar"].exp()
        result["beneficial_probability"] = torch.sigmoid(
            out["beneficial_logit"] / self.log_temperature.exp().clamp_min(0.05)
        )
        return result


def _iter_loader(rows: Sequence[dict], args: argparse.Namespace, shuffle: bool) -> Iterable[dict]:
    if not rows:
        return
    loader = DataLoader(
        DeltaDataset(rows, args.max_len, args.max_edits),
        batch_size=min(args.batch_size, len(rows)), shuffle=shuffle,
        collate_fn=collate, num_workers=0,
    )
    while True:
        for batch in loader:
            yield batch


def train_steps(
    model: nn.Module,
    rows: Sequence[dict],
    args: argparse.Namespace,
    device: torch.device,
    steps: int,
    seed: int,
    stage: str,
) -> dict:
    if not rows or steps <= 0:
        return {"stage": stage, "n": len(rows), "steps": 0, "status": "skipped"}
    loader = _iter_loader(rows, args, shuffle=True)
    optimizer = getattr(train_steps, "optimizer", None)
    if optimizer is None or getattr(train_steps, "optimizer_owner", None) is not model:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate, weight_decay=args.weight_decay,
        )
        train_steps.optimizer = optimizer
        train_steps.optimizer_owner = model
    model.train()
    last: dict[str, float] = {}
    for step in range(steps):
        batch = move_batch(next(loader), device)
        target = batch.pop("delta")
        optimizer.zero_grad(set_to_none=True)
        out = model(**batch)
        loss, parts = paired_delta_loss(out, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss in {stage} at step {step}: {loss}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last = parts
    return {"stage": stage, "n": len(rows), "steps": steps, **last}


@torch.no_grad()
def _predict(model: nn.Module, rows: Sequence[dict], args: argparse.Namespace, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    ys: list[float] = []
    means: list[float] = []
    probs: list[float] = []
    variances: list[float] = []
    loader = DataLoader(
        DeltaDataset(rows, args.max_len, args.max_edits), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate, num_workers=0,
    )
    calibrator = getattr(model, "_phase2_calibrator", None)
    for batch in loader:
        batch = move_batch(batch, device)
        target = batch.pop("delta")
        out = model(**batch)
        if calibrator is not None:
            out = calibrator.calibrate(out)
        ys.extend(target.cpu().tolist())
        means.extend(out["mean"].cpu().tolist())
        probs.extend(out.get("beneficial_probability", torch.sigmoid(out["beneficial_logit"])).cpu().tolist())
        variances.extend(out["variance"].cpu().tolist())
    return {"y": np.asarray(ys), "mean": np.asarray(means), "prob": np.asarray(probs), "variance": np.asarray(variances)}


def _rank(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort").astype(float)


def metrics_from_predictions(pred: dict[str, np.ndarray]) -> dict[str, float]:
    y, mean, prob, variance = pred["y"], pred["mean"], pred["prob"], pred["variance"]
    if len(y) == 0:
        return {"n": 0}
    positive = y > 0
    predicted_positive = prob >= 0.5
    top_n = max(1, int(math.ceil(len(y) * 0.10)))
    top_idx = np.argsort(-mean, kind="mergesort")[:top_n]
    base_rate = float(positive.mean())
    top_rate = float(positive[top_idx].mean())
    ece = 0.0
    for lo, hi in zip(np.linspace(0.0, 1.0, 11)[:-1], np.linspace(0.0, 1.0, 11)[1:]):
        mask = (prob >= lo) & ((prob < hi) if hi < 1 else (prob <= hi))
        if mask.any():
            ece += float(mask.mean()) * abs(float(prob[mask].mean()) - float(positive[mask].mean()))
    rank_y, rank_p = _rank(y), _rank(mean)
    spearman = float(np.corrcoef(rank_y, rank_p)[0, 1]) if len(y) > 1 and np.std(rank_y) and np.std(rank_p) else 0.0
    precision = float(positive[predicted_positive].mean()) if predicted_positive.any() else 0.0
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean((y - mean) ** 2))),
        "spearman": spearman,
        "sign_accuracy": float(np.mean((y > 0) == (mean > 0))),
        "beneficial_precision": precision,
        "top10_enrichment": float(top_rate / base_rate) if base_rate > 0 else 0.0,
        "top10_positive_rate": top_rate,
        "beneficial_base_rate": base_rate,
        "ece": float(ece),
        "mean_pred_variance": float(variance.mean()),
    }


@torch.no_grad()
def _collect_outputs(model: nn.Module, rows: Sequence[dict], args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    means, logvars, logits, targets = [], [], [], []
    loader = DataLoader(DeltaDataset(rows, args.max_len, args.max_edits), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    for batch in loader:
        batch = move_batch(batch, device)
        target = batch.pop("delta")
        out = model(**batch)
        means.append(out["mean"]); logvars.append(out["logvar"]); logits.append(out["beneficial_logit"]); targets.append(target)
    return torch.cat(means), torch.cat(logvars), torch.cat(logits), torch.cat(targets)


def fit_calibrator(model: nn.Module, rows: Sequence[dict], args: argparse.Namespace, device: torch.device) -> dict:
    if not rows:
        return {"status": "skipped", "n": 0}
    mean, logvar, logits, target = _collect_outputs(model, rows, args, device)
    calibrator = DeltaCalibrator().to(device)
    optimizer = torch.optim.Adam(calibrator.parameters(), lr=args.calibration_learning_rate)
    label = (target > 0).float()
    for _ in range(args.calibration_steps):
        optimizer.zero_grad(set_to_none=True)
        cal_logvar = logvar + calibrator.log_scale + calibrator.logvar_bias
        nll = 0.5 * (cal_logvar + (target - mean).pow(2) / cal_logvar.exp()).mean()
        prob = torch.sigmoid(logits / calibrator.log_temperature.exp().clamp_min(0.05))
        loss = nll + 0.5 * (prob - label).pow(2).mean()
        loss.backward()
        optimizer.step()
    model._phase2_calibrator = calibrator
    return {
        "status": "fit_measured_only", "n": int(len(target)),
        "log_scale": float(calibrator.log_scale.detach().cpu()),
        "logvar_bias": float(calibrator.logvar_bias.detach().cpu()),
        "log_temperature": float(calibrator.log_temperature.detach().cpu()),
    }


def _source_group_split(rows: Sequence[dict], calibration_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    fit, calibration = [], []
    for row in rows:
        key = f"{seed}|{row.get('source_id', row.get('source_sequence', ''))}".encode()
        bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
        (calibration if bucket < calibration_fraction else fit).append(row)
    return fit, calibration


def _records_snapshot(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _eligible_training_row(row: dict, confidence: str) -> bool:
    if row.get("delta") is None or row.get("confidence") != confidence:
        return False
    if confidence == "measured":
        return (
            row.get("task_kind") == "local_delta"
            and row.get("data_layer") == "C_source_matched_intervention"
            and bool(row.get("local_delta_eligible"))
        )
    if confidence == "proxy":
        return row.get("data_layer") == "B_absolute_design_library"
    return False


def _load_training_roles(root: Path) -> tuple[dict[str, list[dict]], dict]:
    """Read the canonical records store once for the train and val roles.

    The manifest indexes are loaded first and the records file is hashed while
    it is parsed.  A before/after stat check makes a concurrent rebuild fail
    closed instead of silently producing different arms from different files.
    """
    role_indexes: dict[str, set[str]] = {}
    records_path: Optional[Path] = None
    manifest_digests: dict[str, str] = {}
    for role in ("train", "val"):
        manifest_path = root / "manifests" / f"{role}.json"
        manifest = load_manifest(manifest_path)
        current_records_path = root / str(manifest["records_path"])
        if records_path is None:
            records_path = current_records_path
        elif current_records_path != records_path:
            raise RuntimeError("train and val manifests point to different records stores")
        index_path = root / str(manifest["index_path"])
        role_indexes[role] = {
            line.strip() for line in index_path.read_text().splitlines() if line.strip()
        }
        manifest_digests[role] = manifest_sha256(manifest_path)
    assert records_path is not None
    before = _records_snapshot(records_path)
    wanted = set().union(*role_indexes.values())
    rows: dict[str, list[dict]] = {"train_measured": [], "train_proxy": [], "val_measured": []}
    digest = hashlib.sha256()
    with records_path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            record_id = str(row.get("record_id"))
            if record_id not in wanted:
                continue
            if record_id in role_indexes["train"]:
                if _eligible_training_row(row, "measured"):
                    rows["train_measured"].append(row)
                if _eligible_training_row(row, "proxy"):
                    rows["train_proxy"].append(row)
            if record_id in role_indexes["val"] and _eligible_training_row(row, "measured"):
                rows["val_measured"].append(row)
    after = _records_snapshot(records_path)
    if before != after:
        raise RuntimeError(f"records store changed during training snapshot: {before} -> {after}")
    return rows, {
        "records_snapshot": after,
        "records_sha256": digest.hexdigest(),
        "manifest_sha256": manifest_digests,
        "eligible_counts": {key: len(value) for key, value in rows.items()},
    }


def _sample_rows(rows: Sequence[dict], limit: int, seed: int) -> list[dict]:
    sampled = list(rows)
    random.Random(seed).shuffle(sampled)
    return sampled[:limit] if limit > 0 else sampled


def prepare_data(root: Path, args: argparse.Namespace, seeds: Sequence[int]) -> dict:
    rows, provenance = _load_training_roles(root)
    by_seed = {}
    for seed in seeds:
        by_seed[seed] = {
            "measured": _sample_rows(rows["train_measured"], args.measured_records, seed),
            "proxy": _sample_rows(rows["train_proxy"], args.proxy_records, seed + 2000),
            "val": _sample_rows(rows["val_measured"], args.val_records, seed + 1000),
        }
    provenance["prepared_counts"] = {
        str(seed): {key: len(value) for key, value in by_seed[seed].items()}
        for seed in seeds
    }
    return {"by_seed": by_seed, "provenance": provenance}


def run_experiment(
    args: argparse.Namespace,
    seed: int,
    backbone: str,
    recipe: str,
    device: torch.device,
    prepared: dict,
    foundation_sha256: Optional[str],
) -> dict:
    seed_everything(seed)
    seed_data = prepared["by_seed"][seed]
    measured_all = seed_data["measured"]
    measured_fit, calibration_rows = _source_group_split(measured_all, args.calibration_fraction, seed)
    measured_val = seed_data["val"]
    proxy_train = seed_data["proxy"]
    if recipe in {"measured_only", "mixed_training", "pretrain_finetune", "pretrain_finetune_calibrate"} and not measured_fit:
        raise RuntimeError("no measured local-delta training records")
    if recipe in {"proxy_only", "mixed_training", "pretrain_finetune", "pretrain_finetune_calibrate"} and not proxy_train:
        raise RuntimeError("no proxy training records")

    model = PairedDeltaFormer(
        hidden_dim=args.hidden_dim, layers=args.layers, max_len=args.max_len,
        backbone=backbone, foundation_path=args.foundation_path,
        allow_foundation_stub=args.allow_foundation_stub,
        foundation_name=args.foundation_name, unfreeze_last_n=args.unfreeze_last_n,
    ).to(device)
    train_steps.optimizer = None
    train_steps.optimizer_owner = None
    history = []
    if recipe == "proxy_only":
        history.append(train_steps(model, proxy_train, args, device, args.stage_a_steps, seed, "proxy_pretraining"))
    elif recipe == "mixed_training":
        history.append(train_steps(model, proxy_train + measured_fit, args, device, args.stage_b_steps, seed, "mixed_control"))
    elif recipe == "measured_only":
        history.append(train_steps(model, measured_fit, args, device, args.stage_b_steps, seed, "measured_only"))
    else:
        history.append(train_steps(model, proxy_train, args, device, args.stage_a_steps, seed, "proxy_pretraining"))
        history.append(train_steps(model, measured_fit, args, device, args.stage_b_steps, seed, "measured_finetuning"))
    calibration = {"status": "not_requested", "n": 0}
    if recipe == "pretrain_finetune_calibrate":
        calibration = fit_calibrator(model, calibration_rows, args, device)
    metrics = metrics_from_predictions(_predict(model, measured_val, args, device))
    out_dir = Path(args.out_dir) / f"backbone={backbone}" / f"recipe={recipe}" / f"seed={seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "paired_delta_former.pt"
    torch.save({
        "model": model.state_dict(), "calibrator": getattr(getattr(model, "_phase2_calibrator", None), "state_dict", lambda: None)(),
        "seed": seed, "backbone": backbone, "recipe": recipe, "config": vars(args),
        "history": history, "calibration": calibration, "metrics": metrics,
        "backbone_status": model.backbone_status, "is_real_foundation": model.is_real_foundation,
        "foundation_kind": model.foundation_kind,
        "foundation_sha256": foundation_sha256,
        "prepared_data": prepared["provenance"],
        "final_test_used": False,
    }, checkpoint)
    (out_dir / "metrics.json").write_text(json.dumps({
        "seed": seed, "backbone": backbone, "recipe": recipe, "history": history,
        "calibration": calibration, "metrics": metrics,
        "backbone_status": model.backbone_status, "is_real_foundation": model.is_real_foundation,
        "foundation_kind": model.foundation_kind,
        "foundation_sha256": foundation_sha256,
        "final_test_used": False,
    }, indent=2, sort_keys=True) + "\n")
    return {
        "seed": seed, "backbone": backbone, "recipe": recipe,
        "metrics": metrics, "calibration": calibration,
        "backbone_status": model.backbone_status, "is_real_foundation": model.is_real_foundation,
        "foundation_kind": model.foundation_kind,
        "prepared_data": prepared["provenance"],
        "checkpoint": str(checkpoint), "final_test_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", default="data/nmi_benchmark_v2")
    parser.add_argument("--out-dir", default="artifacts/phase2_reliable_local_delta")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--backbones", default="small")
    parser.add_argument("--recipes", default="pretrain_finetune_calibrate")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--max-edits", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stage-a-steps", type=int, default=1000)
    parser.add_argument("--stage-b-steps", type=int, default=2000)
    parser.add_argument("--calibration-steps", type=int, default=200)
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=3e-4)
    parser.add_argument("--calibration-learning-rate", type=float, default=5e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--proxy-records", type=int, default=10000)
    parser.add_argument("--measured-records", type=int, default=5000)
    parser.add_argument("--val-records", type=int, default=2000)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--foundation-path", default=None)
    parser.add_argument("--foundation-sha256", default=None)
    parser.add_argument("--foundation-name", default="rna_foundation")
    parser.add_argument("--unfreeze-last-n", type=int, default=1)
    parser.add_argument("--allow-foundation-stub", action="store_true")
    args = parser.parse_args()
    if args.recipes != "all":
        requested_recipes = [x.strip() for x in args.recipes.split(",") if x.strip()]
        unknown = set(requested_recipes) - set(RECIPES)
        if unknown:
            raise ValueError(f"unknown recipe(s): {sorted(unknown)}")
    else:
        requested_recipes = list(RECIPES)
    requested_backbones = [x.strip() for x in args.backbones.split(",") if x.strip()]
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    root = Path(args.benchmark_root)
    seeds = [int(seed_text) for seed_text in args.seeds.split(",") if seed_text.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    prepared = prepare_data(root, args, seeds)
    foundation_sha256_by_backbone = {
        backbone: validate_foundation_provenance(args, backbone)
        for backbone in requested_backbones
    }
    results = []
    for backbone in requested_backbones:
        for recipe in requested_recipes:
            for seed in seeds:
                results.append(run_experiment(
                    args, seed, backbone, recipe, device, prepared,
                    foundation_sha256_by_backbone[backbone],
                ))
    final_snapshot = _records_snapshot(root / "records.jsonl")
    if final_snapshot != prepared["provenance"]["records_snapshot"]:
        raise RuntimeError(
            "records store changed during the experiment; output is invalid: "
            f"{prepared['provenance']['records_snapshot']} -> {final_snapshot}"
        )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    provenance = {
        "schema_version": "phase2_reliable_local_delta_v1",
        "benchmark_root": str(root),
        "train_manifest_sha256": prepared["provenance"]["manifest_sha256"]["train"],
        "val_manifest_sha256": prepared["provenance"]["manifest_sha256"]["val"],
        "data_snapshot": prepared["provenance"],
        "backbones": requested_backbones, "recipes": requested_recipes,
        "feature_contract": ["source_sequence", "candidate_sequence", "explicit_edit_tokens", "relative_position_to_cds_start", "cargo_or_protein_embedding", "cell_context_embedding", "assay_embedding", "source_measured_or_proxy_value"],
        "same_training_steps": {"stage_a": args.stage_a_steps, "stage_b": args.stage_b_steps, "calibration": args.calibration_steps},
        "final_test_used": False,
        "real_foundation_required_for_scientific_claim": True,
        "foundation_path": args.foundation_path,
        "foundation_sha256": foundation_sha256_by_backbone,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    (out / "summary.json").write_text(json.dumps({"provenance": provenance, "results": results}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"provenance": provenance, "results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
