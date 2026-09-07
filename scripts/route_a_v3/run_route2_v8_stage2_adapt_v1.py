#!/usr/bin/env python3
"""V8 Stage 2: benchmark adaptation from Stage 1 joint-pretrained weights.

Preregistration: docs/paper/route2_v8_stage2_prereg_v1.md (frozen 2026-09-07).
Trigger: Stage 1 full adjudication RESOLVED (2026-09-07 04:05, polyA gate
RESOLVED; S FAILS polyA non-destruction 0.1122 < 0.1883, H PASSES 0.5262).
Architecture amendment (pre-registered clause): main arm = H.

Two data modes:

- --libraries cms  (MPRAU arms): adaptation on the ENCODE CMS array library
  (85,475 rows, cell-conditioned, domain cms=2). Primary judgment = MPRAU
  VALIDATION variant pair-mean rho vs V5 0.1025 (paired bootstrap 2,000 iters,
  seed 20260816, CI must not cross zero).
- --benchmark      (multi-task arm): full-param balanced adaptation on the
  frozen benchmark TRAIN pool (all development studies; MPRAU rows keep their
  per-cell labels). Judgment = 9-task VALIDATION frozen-delta + MPRAU pair-mean
  + task-macro vs 0.167. Gates: MRL >=0.28 / polyA >=0.80 / MPRAU >0.1025 CI /
  TE >= internal target 0.1317 / macro up.

Common:

- Init: Stage 1 checkpoint (arm S or H, num_domains=3). num_domains is
  extended for benchmark mode (ids mrl/polya/cms preserved from Stage 1; new
  ids 3..8 initialised N(0, 0.02)).
- Adaptation: full-parameter or LoRA (r16 a32, qkv+o+mlp). AdamW lr 2e-5
  wd 1e-4, cosine to 10% after 5% warmup, BF16 autocast, seed 20260907.
- Cell conditioning: per-context embedding (6 ENCODE contexts) added to the
  pooled representation; MPRAU rows keep original per-cell labels (the V6
  pair-mean aggregation failure is not repeated).
- Selection: FINAL-EPOCH-FIXED (no peak-picking, H2 red line).
- Discipline: CUDA BF16 only (cpu_fallback_used=false); protected reads = 0;
  products /mnt, code /home worktree + push.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_v8_hybrid_backbone_v1 import (  # noqa: E402
    CELL_IDS,
    DOMAIN_IDS,
    NUM_DOMAINS,
    build_v8_regressor,
    parameter_report,
    verify_vocab_alignment,
)
from core.route2_v8_joint_library_v1 import (  # noqa: E402
    CMS_ARRAY_CSV,
    MNT,
    DomainBalancedSampler,
    DomainLibrary,
    audit_leak_flags,
    build_protected_index,
    format_sequence,
    load_cms_library,
    prepare_domain_library,
)
from core.route2_mrnabert_lora_v3 import LoRALinearV3  # noqa: E402

EVAL_REPO = "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901"
_ev_spec = importlib.util.spec_from_file_location(
    "ev", EVAL_REPO + "/scripts/route_a_v3/evaluate_route2_prediction_v1.py"
)
ev = importlib.util.module_from_spec(_ev_spec)
sys.modules["ev"] = ev
_ev_spec.loader.exec_module(ev)

MRNABERT_PATH = MNT / "external_model_assets/mrnabert_a1eb7df25804d23f08646e1cb996b234d7208a40"
MANIFEST = MNT / "manifests/route2_development_frozen_v1/development_manifest.jsonl"
PROJECTION_TRAIN = MNT / "projections/xedit_v3/development_train_validation_v1/train.jsonl"
CANONICAL_ROOT = MNT / "canonical"
V5_PRED_GLOB = str(MNT / "experiments/xeditcritic_v5/*/v5_full/final_validation_predictions.jsonl")
OUT_ROOT = MNT / "experiments/xeditcritic_route_a/v8_stage2_adapt_20260907"

BATCH = 128
EPOCHS = 6
LR = 2e-5
WEIGHT_DECAY = 1e-4
SEED = 20260907
EVAL_BATCH = 256
K = 10
LORA_RANK = 16
LORA_ALPHA = 32.0
LORA_DROPOUT = 0.05
BOOT_ITERS = 2000
BOOT_SEED = 20260816

# benchmark study -> (eval domain id, canonical relative dir)
BENCHMARK_DOMAINS = {
    "GSE114002": ("mrl", "GSE114002/v1/canonical_records.private.jsonl"),
    "GSE269595": ("polya", "GSE269595/v1/canonical_records.private.jsonl"),
    "ENCSR854RUF": ("mprau", "ENCSR854RUF/v1/canonical_records.private.jsonl"),
    "GSE149487": ("gse149487", "GSE149487/v1/canonical_records.private.jsonl"),
    "GSE186455": ("gse186455", "GSE186455/v1/canonical_records.private.jsonl"),
    "GSE200304": ("gse200304", "GSE200304/v1/canonical_records.jsonl"),
    "GSE217518": ("gse217518", "GSE217518/v1/canonical_records.jsonl"),
    "GSE256185": ("gse256185", "GSE256185/v1/canonical_records.private.jsonl"),
}
# CMS csv cell_context (build_cms_library_v1 CELL_LINES order) -> CELL_IDS id
CMS_CELL_TO_ID = [CELL_IDS["GM12878"], CELL_IDS["K562"], CELL_IDS["HEPG2"],
                  CELL_IDS["SKNSH"], CELL_IDS["HMEC"]]
NUM_CELLS = max(CELL_IDS.values()) + 1

# internal target for TE family (source-only global_scaled macro 0.1317)
TE_INTERNAL_TARGET = 0.1317
V5_MPRAU_REFERENCE = 0.1025


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--arch", required=True, choices=("s", "h"))
    parser.add_argument("--init-checkpoint", required=True, help="Stage 1 S/H terminal checkpoint (.pt)")
    parser.add_argument("--adapt-mode", default="full", choices=("full", "lora"))
    parser.add_argument("--libraries", default="cms", help="comma list from {cms}")
    parser.add_argument("--benchmark", action="store_true", help="benchmark TRAIN pool mode (pair-delta balanced)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--cell-conditioning", action="store_true", default=True)
    parser.add_argument("--no-cell-conditioning", dest="cell_conditioning", action="store_false")
    parser.add_argument("--max-steps", type=int, default=None, help="smoke cap")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def load_cms_domain(tokenizer) -> tuple[DomainLibrary, torch.Tensor]:
    """CMS 85,475-row library with per-row cell ids (aligned to clean rows)."""
    sequences, activities, contexts = load_cms_library(CMS_ARRAY_CSV)
    protected_index = build_protected_index()
    flags = audit_leak_flags(sequences, protected_index)
    clean = ~np.logical_or.reduce(list(flags.values())) if flags else np.ones(len(sequences), dtype=bool)
    lib = prepare_domain_library("cms", sequences, activities, flags, tokenizer)
    cells_np = np.asarray([CMS_CELL_TO_ID[int(c)] for c in contexts], dtype=np.int64)
    cell_ids = torch.as_tensor(cells_np[np.asarray(clean, dtype=bool)], dtype=torch.long)
    assert cell_ids.shape[0] == lib.n_clean, "cell ids misaligned with clean library rows"
    return lib, cell_ids


def load_benchmark_domains(tokenizer) -> dict[str, tuple[DomainLibrary, torch.Tensor]]:
    """Benchmark TRAIN pool -> per-study pair-delta DomainLibrary + cell ids.

    Each DomainLibrary stores: input_ids/candidate ids are packed side by side
    as [2N, L] (even rows = source, odd rows = candidate), targets = z-scored
    per-study delta [N]. cell_ids [N] for MPRAU (per-cell rows preserved),
    None for non-cell tasks.
    """
    rows_by_study: dict[str, list[dict]] = defaultdict(list)
    with PROJECTION_TRAIN.open() as handle:
        for line in handle:
            row = json.loads(line)
            if str(row.get("split")) != "TRAIN":
                continue
            study = str(row["study_unit_id"])
            if study in BENCHMARK_DOMAINS:
                rows_by_study[study].append(row)
    libraries: dict[str, tuple[DomainLibrary, torch.Tensor]] = {}
    for study, rows in sorted(rows_by_study.items()):
        domain, _ = BENCHMARK_DOMAINS[study]
        deltas = np.asarray([float(r["direction_normalized_delta"]) for r in rows], dtype=np.float64)
        z, mean, std = _standardize(deltas)
        sources = [format_sequence(str(r["source_sequence"])) for r in rows]
        candidates = [format_sequence(str(r["candidate_sequence"])) for r in rows]
        seqs = []
        for s, c in zip(sources, candidates):
            seqs.append(s)
            seqs.append(c)
        encoded = tokenizer(
            seqs, add_special_tokens=True, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        )
        target_mean = float(deltas.mean())
        target_std = float(deltas.std())
        if target_std <= 0.0:
            raise ValueError(f"constant delta vector for {study}")
        lib = DomainLibrary(
            domain=domain,
            domain_id=DOMAIN_IDS[domain],
            sequences=sources,
            activities_raw=deltas,
            leak_flags={},
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            targets=torch.tensor(z, dtype=torch.float32),
            target_mean=target_mean,
            target_std=target_std,
        )
        # NOTE: input_ids has 2N rows (source even, candidate odd); n_clean
        # reflects 2N, so override for the sampler.
        cell_ids = None
        if study == "ENCSR854RUF":
            cells = torch.tensor(
                [CELL_IDS.get(str(r.get("biological_context_id")), 0) for r in rows], dtype=torch.long)
            cell_ids = cells
        libraries[study] = (lib, cell_ids)
        print(f"[bench {study}] domain={domain} n_records={len(rows)} delta_mean={target_mean:.4f} delta_std={target_std:.4f} cell={cell_ids is not None}", flush=True)
    if not libraries:
        raise SystemExit(f"benchmark pool empty from {PROJECTION_TRAIN}")
    return libraries


def _standardize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(values.mean())
    std = float(values.std())
    if std <= 0.0:
        raise ValueError("cannot standardise constant vector")
    return ((values - mean) / std).astype(np.float32), mean, std


def wrap_lora(model: nn.Module) -> int:
    wrapped = 0
    for layer in model.base.encoder.layer:
        for attr_path in (
            ("attention", "self", "Wqkv"),
            ("attention", "output", "dense"),
            ("mlp", "gated_layers"),
            ("mlp", "wo"),
        ):
            parent = layer
            for step in attr_path[:-1]:
                parent = getattr(parent, step)
            base = getattr(parent, attr_path[-1])
            if isinstance(base, nn.Linear):
                setattr(parent, attr_path[-1],
                        LoRALinearV3(base, rank=LORA_RANK, alpha=LORA_ALPHA, dropout=LORA_DROPOUT))
                wrapped += 1
    if wrapped == 0:
        raise SystemExit("LoRA wrap found no target Linear modules")
    return wrapped


def load_init(model: nn.Module, init_checkpoint: Path) -> dict:
    raw = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
    state = raw["model_state_dict"]
    if raw.get("arch") not in (None, model.use_stem and "h" or "s") and str(raw.get("arch", "")) not in ("s", "h"):
        raise SystemExit(f"init checkpoint arch field unusable: {raw.get('arch')!r}")
    if raw.get("arch") is not None and (raw["arch"] == "h") != model.use_stem:
        raise SystemExit(f"init checkpoint arch={raw['arch']} mismatches model arch (use_stem={model.use_stem})")
    missing = []
    model_state = model.state_dict()
    compatible = {}
    for key, value in state.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible[key] = value
        elif key not in model_state:
            missing.append(key)
    model.load_state_dict(compatible, strict=False)
    # preserve Stage 1 domain embeddings for trained ids (0..2)
    dom = state.get("domain_embeddings.weight")
    if dom is not None:
        n = min(dom.shape[0], model.domain_embeddings.weight.shape[0])
        model.domain_embeddings.weight.data[:n] = dom[:n]
    summary = {
        "init_checkpoint": str(init_checkpoint),
        "init_arch": raw.get("arch"),
        "loaded_keys": len(compatible),
        "skipped_keys": [k for k in state if k not in compatible],
        "missing_from_ckpt": missing,
    }
    print(f"init: {json.dumps(summary, indent=1)}", flush=True)
    return summary


def _manifest_validation_ids() -> dict[str, set[str]]:
    by_study: dict[str, set[str]] = defaultdict(set)
    with MANIFEST.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == "VALIDATION":
                by_study[str(row["study_unit_id"])].add(str(row["canonical_record_id"]))
    return by_study


def _load_canonical(rel: str) -> dict[str, dict]:
    records = {}
    path = CANONICAL_ROOT / rel
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            records[str(row["canonical_record_id"])] = row
    return records


def mprau_variant_table(predictions: dict[str, float], targets: dict[str, float]) -> dict[str, tuple[float, float]]:
    by_variant: dict[str, list[str]] = defaultdict(list)
    for rid in predictions:
        by_variant[rid.split(":context:")[0]].append(rid)
    variants = {}
    for variant, rids in by_variant.items():
        if len(rids) >= 2:
            variants[variant] = (
                float(np.mean([targets[r] for r in rids])),
                float(np.mean([predictions[r] for r in rids])),
            )
    return variants


def pair_mean_rho(variants: dict[str, tuple[float, float]]) -> float:
    if not variants:
        return float("nan")
    t = np.asarray([v[0] for v in variants.values()], dtype=np.float64)
    p = np.asarray([v[1] for v in variants.values()], dtype=np.float64)
    return float(ev.numeric_metrics(t, p)["spearman"])


def paired_bootstrap_vs_reference(ref_variants, arm_variants, label: str) -> dict:
    shared = sorted(set(ref_variants) & set(arm_variants))
    if len(shared) < 50:
        return {"reference": label, "shared_variant_count": len(shared), "skipped": "too few shared variants"}
    t = np.asarray([ref_variants[v][0] for v in shared], dtype=np.float64)
    pr = np.asarray([ref_variants[v][1] for v in shared], dtype=np.float64)
    pa = np.asarray([arm_variants[v][1] for v in shared], dtype=np.float64)

    def rho(x: np.ndarray, y: np.ndarray) -> float:
        return float(ev.numeric_metrics(x, y)["spearman"])

    rng = np.random.default_rng(BOOT_SEED)
    base_r, base_a = rho(t, pr), rho(t, pa)
    deltas = []
    n = len(shared)
    for _ in range(BOOT_ITERS):
        idx = rng.integers(0, n, size=n)
        deltas.append(rho(t[idx], pa[idx]) - rho(t[idx], pr[idx]))
    deltas = np.asarray(deltas)
    ci = np.percentile(deltas, [2.5, 97.5])
    return {
        "reference": label,
        "shared_variant_count": len(shared),
        "reference_pair_mean_spearman": base_r,
        "arm_pair_mean_spearman": base_a,
        "delta_pair_mean_spearman": base_a - base_r,
        "delta_ci95": [float(ci[0]), float(ci[1])],
        "crosses_zero": bool(ci[0] < 0 < ci[1]),
        "bootstrap_iters": BOOT_ITERS,
        "bootstrap_seed": BOOT_SEED,
    }


def score_sequences(model, tokenizer, device, sequences: list[str], domain_id: int,
                    cell_ids: torch.Tensor | None = None) -> np.ndarray:
    model.eval()
    values = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for start in range(0, len(sequences), EVAL_BATCH):
            chunk = sequences[start:start + EVAL_BATCH]
            enc = tokenizer(
                [format_sequence(s) for s in chunk],
                add_special_tokens=True, padding=True, truncation=True, max_length=512, return_tensors="pt",
            )
            dom = torch.full((len(chunk),), domain_id, dtype=torch.long, device=device)
            cells = None
            if cell_ids is not None:
                cells = cell_ids[start:start + len(chunk)].to(device)
            out = model(enc["input_ids"].to(device), enc["attention_mask"].to(device), dom, cells)
            values.append(out.float().cpu().numpy())
    model.train()
    return np.concatenate(values)


def eval_mprau_validation(model, tokenizer, device, domain_id: int) -> dict:
    validation_ids = _manifest_validation_ids().get("ENCSR854RUF", set())
    records = _load_canonical(BENCHMARK_DOMAINS["ENCSR854RUF"][1])
    ids = sorted(rid for rid in validation_ids if rid in records)
    cells = torch.tensor(
        [CELL_IDS.get(str(records[rid].get("biological_context_id")), 0) for rid in ids], dtype=torch.long)
    source = score_sequences(model, tokenizer, device, [records[rid]["source_sequence"] for rid in ids], domain_id, cells)
    candidate = score_sequences(model, tokenizer, device, [records[rid]["candidate_sequence"] for rid in ids], domain_id, cells)
    delta = candidate - source
    predictions = {rid: float(delta[i]) for i, rid in enumerate(ids)}
    targets = {rid: float(records[rid]["direction_normalized_delta"]) for rid in ids}
    variants = mprau_variant_table(predictions, targets)
    rho = pair_mean_rho(variants)
    report = {
        "study": "ENCSR854RUF",
        "split": "VALIDATION",
        "n_records": len(ids),
        "n_variants": len(variants),
        "pair_mean_spearman": rho,
        "reference_v5_pair_mean": V5_MPRAU_REFERENCE,
        "domain_id_used": domain_id,
    }
    # paired bootstrap vs V5 predictions
    v5_paths = sorted(glob.glob(V5_PRED_GLOB))
    if v5_paths:
        v5_preds = {}
        with open(v5_paths[0]) as handle:
            for line in handle:
                r = json.loads(line)
                rid = str(r.get("canonical_record_id") or r.get("record_id"))
                if rid in targets:
                    v5_preds[rid] = float(r["prediction"])
        v5_variants = mprau_variant_table(v5_preds, targets)
        report["vs_v5"] = paired_bootstrap_vs_reference(v5_variants, variants, "v5_multitask")
    return report


def eval_study_frozen_delta(model, tokenizer, device, study: str, domain: str) -> dict:
    validation_ids = _manifest_validation_ids().get(study, set())
    records = _load_canonical(BENCHMARK_DOMAINS[study][1])
    ids = sorted(rid for rid in validation_ids if rid in records)
    if not ids:
        return {"domain": domain, "study": study, "split": "VALIDATION", "n_records": 0,
                "task_macro_spearman": None, "top_1": None, "ndcg_at_10": None,
                "pair_mean_spearman": None, "skipped": "no VALIDATION records in development manifest"}
    domain_id = DOMAIN_IDS[domain]
    cell_ids = None
    if study == "ENCSR854RUF":
        cell_ids = torch.tensor(
            [CELL_IDS.get(str(records[rid].get("biological_context_id")), 0) for rid in ids], dtype=torch.long)
    source = score_sequences(model, tokenizer, device, [records[rid]["source_sequence"] for rid in ids], domain_id, cell_ids)
    candidate = score_sequences(model, tokenizer, device, [records[rid]["candidate_sequence"] for rid in ids], domain_id, cell_ids)
    delta = candidate - source
    predictions = {rid: float(delta[i]) for i, rid in enumerate(ids)}
    observations = ev.load_observations([CANONICAL_ROOT / BENCHMARK_DOMAINS[study][1]], set(ids))
    metrics = ev.evaluate(observations, predictions, K)
    return {
        "domain": domain,
        "study": study,
        "split": "VALIDATION",
        "n_records": len(ids),
        "task_macro_spearman": metrics.get("task_macro_spearman"),
        "top_1": metrics.get("source_macro_top_1_accuracy"),
        "ndcg_at_10": metrics.get("source_macro_ndcg_at_k"),
        "pair_mean_spearman": None,
    }


def _task_macro(results: list[dict]) -> float:
    vals = []
    for r in results:
        v = r.get("task_macro_spearman")
        if v is None:
            v = r.get("pair_mean_spearman")  # MPRAU primary caliber stands in for its macro
        if v is not None:
            vals.append(float(v))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - GPU required")
    device = torch.device(f"cuda:{args.physical_gpu_index}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MRNABERT_PATH, local_files_only=True)
    verify_vocab_alignment(tokenizer)

    mode = "benchmark" if args.benchmark else "library_cms"
    out_dir = Path(args.out_dir) if args.out_dir else OUT_ROOT / f"{args.arch}_{mode}_{args.adapt_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode == "benchmark":
        num_domains = len(DOMAIN_IDS)  # full 9-domain geometry
    else:
        num_domains = NUM_DOMAINS
    num_cells = NUM_CELLS if args.cell_conditioning else 0
    model = build_v8_regressor(MRNABERT_PATH, args.arch, num_domains=num_domains, num_cells=num_cells).to(device)
    params = parameter_report(model)
    print(f"parameter report: {json.dumps(params)}", flush=True)
    init_summary = load_init(model, Path(args.init_checkpoint))
    print(f"device={device} mode={mode} adapt={args.adapt_mode} cell_conditioning={args.cell_conditioning}", flush=True)

    # ---- data ----
    libraries: dict[str, tuple[DomainLibrary, torch.Tensor]] = {}
    if mode == "benchmark":
        libraries = load_benchmark_domains(tokenizer)
        # use study-domain sizes for the balanced sampler (records, not 2N)
        domain_sizes = {study: int(lib.targets.shape[0]) for study, (lib, _c) in libraries.items()}
    else:
        lib, cell_ids = load_cms_domain(tokenizer)
        summary = lib.audit_summary()
        print(f"[cms] {json.dumps(summary)}", flush=True)
        libraries = {"cms": (lib, cell_ids)}
        domain_sizes = {"cms": lib.n_clean}

    if args.skip_train:
        print("skip-train: data prepared, exiting", flush=True)
        return 0

    # ---- adaptation params ----
    if args.adapt_mode == "lora":
        wrapped = wrap_lora(model)
        model.to(device)  # LoRA modules are created in-place on CPU; move to device
        print(f"LoRA wrap: {wrapped} target modules (r={LORA_RANK} a={LORA_ALPHA})", flush=True)
        for p in model.parameters():
            p.requires_grad_(False)
        for name, p in model.named_parameters():
            if "lora_" in name or "domain_embeddings" in name or "cell_embeddings" in name or name.startswith("head."):
                p.requires_grad_(True)
    params_after = parameter_report(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params after adapt mode {args.adapt_mode}: {sum(p.numel() for p in trainable):,}", flush=True)

    sampler = DomainBalancedSampler(domain_sizes=domain_sizes, batch_size=args.batch, seed=args.seed)
    planned_steps = sampler.steps_per_epoch * args.epochs
    total_steps = args.max_steps if args.max_steps is not None else planned_steps
    budget = {
        "epochs": args.epochs, "batch": args.batch, "steps_per_epoch": sampler.steps_per_epoch,
        "planned_total_steps": planned_steps, "max_steps_cap": args.max_steps,
        "effective_total_steps": total_steps, "domain_sizes": domain_sizes,
        "mode": mode, "adapt_mode": args.adapt_mode, "cell_conditioning": args.cell_conditioning,
        "num_domains": num_domains, "num_cells": num_cells,
    }
    print(f"budget: {json.dumps(budget)}", flush=True)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(step / max(total_steps, 1), 1.0)))
        if step > total_steps * 0.05 else step / max(total_steps * 0.05, 1),
    )

    loss_log = (out_dir / "training_losses.jsonl").open("w")
    epoch_domain_loss = (out_dir / "epoch_domain_loss.jsonl").open("w")
    epoch_eval = (out_dir / "epoch_eval_metrics.jsonl").open("w")

    def save_checkpoint(name: str, epoch_done: int, steps_done: int, note: str) -> None:
        torch.save({
            "schema_version": "route_a_v3_route2_v8_stage2_adapt.v1",
            "model_state_dict": {k: v for k, v in model.state_dict().items()},
            "arch": args.arch, "mode": mode, "adapt_mode": args.adapt_mode,
            "seed": args.seed, "epochs_done": epoch_done, "steps_done": steps_done, "note": note,
        }, out_dir / name)

    def run_eval(epoch_done: int, steps_done: int, primary: bool) -> dict:
        if args.skip_eval:
            return {}
        if mode == "benchmark":
            results = []
            for study, (domain, _rel) in BENCHMARK_DOMAINS.items():
                if study == "ENCSR854RUF":
                    continue  # MPRAU primary = pair-mean (evaluated separately below)
                rec = eval_study_frozen_delta(model, tokenizer, device, study, domain)
                results.append(rec)
                print(f"== eval {study} {domain}: spearman {rec.get('task_macro_spearman')} top1 {rec.get('top_1')}", flush=True)
            mprau = eval_mprau_validation(model, tokenizer, device, DOMAIN_IDS["mprau"])
            results.append({"domain": "mprau", **mprau})
            macro = _task_macro(results)
            report = {
                "epoch": epoch_done, "steps": steps_done, "primary": primary,
                "task_macro_spearman": macro, "per_task": results, "mprau": mprau,
            }
        else:
            mprau = eval_mprau_validation(model, tokenizer, device, DOMAIN_IDS["cms"])
            report = {"epoch": epoch_done, "steps": steps_done, "primary": primary, "mprau": mprau}
        epoch_eval.write(json.dumps(report) + "\n")
        epoch_eval.flush()
        print(f"== epoch {epoch_done} eval: {json.dumps({k: v for k, v in report.items() if k not in ('per_task',)})}", flush=True)
        return report

    model.train()
    step = 0
    epoch_completed = 0
    stopped_early = False
    final_eval: dict = {}
    recent: dict[str, list[float]] = {"bench": []} if mode == "benchmark" else {d: [] for d in domain_sizes}
    for epoch in range(args.epochs):
        epoch_losses: dict[str, list[float]] = {d: [] for d in domain_sizes}
        for batch in sampler.epoch_batches(epoch):
            if mode == "benchmark":
                loss = train_benchmark_batch(model, libraries, batch, device)
                domain = "bench"
            else:
                loss, per_domain = train_cms_batch(model, libraries, batch, device)
                domain = "cms"
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if mode == "benchmark":
                recent["bench"].append(float(loss))
            else:
                for d, v in per_domain.items():
                    recent[d].append(v)
            if step % 50 == 0:
                rec = {"step": step, "epoch": epoch + 1, "mse_all": float(loss),
                       "lr": scheduler.get_last_lr()[0]}
                loss_log.write(json.dumps(rec) + "\n")
                loss_log.flush()
                if step % 500 == 0:
                    print(f"epoch {epoch + 1} step {step}: mse {float(loss):.4f}", flush=True)
            if args.max_steps is not None and step >= args.max_steps:
                stopped_early = True
                break
        epoch_completed = epoch + 1
        rec = {"epoch": epoch_completed, "steps": step, "partial": stopped_early}
        epoch_domain_loss.write(json.dumps(rec) + "\n")
        epoch_domain_loss.flush()
        if stopped_early:
            save_checkpoint(f"stage2_{args.arch}_{mode}_{args.adapt_mode}_smoke{step}.pt", epoch_completed, step,
                            "smoke max-steps stop (non-terminal)")
            final_eval = run_eval(epoch_completed, step, primary=False)
            break
        save_checkpoint(f"stage2_{args.arch}_{mode}_{args.adapt_mode}_epoch{epoch_completed}.pt",
                        epoch_completed, step, "final" if epoch_completed == args.epochs else "epoch")
        final_eval = run_eval(epoch_completed, step, primary=(epoch_completed == args.epochs))

    loss_log.close()
    epoch_domain_loss.close()
    epoch_eval.close()

    report = {
        "schema_version": "route_a_v3_route2_v8_stage2_adapt.v1",
        "mode": mode.upper(), "arch": args.arch, "adapt_mode": args.adapt_mode,
        "selection_rule": "FINAL_EPOCH_FIXED" if not stopped_early else "SMOKE_MAX_STEPS (non-terminal)",
        "seed": args.seed, "params": params, "params_after_adapt": params_after,
        "init": init_summary, "budget": budget, "steps_done": step, "epochs_completed": epoch_completed,
        "smoke": bool(args.max_steps is not None), "cpu_fallback_used": False,
        "eval_final": final_eval,
    }
    (out_dir / "run_report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"wrote {out_dir / 'run_report.json'}", flush=True)
    return 0


def train_cms_batch(model, libraries, batch, device):
    lib, cell_ids = libraries["cms"]
    idx = torch.as_tensor(np.asarray(batch["cms"]))
    ids = lib.input_ids[idx]
    mask = lib.attention_mask[idx]
    targets = lib.targets[idx]
    cells = cell_ids[idx].to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction = model(ids.to(device), mask.to(device),
                           torch.full((ids.shape[0],), lib.domain_id, dtype=torch.long, device=device), cells)
        loss = F.mse_loss(prediction.float(), targets.to(device))
    return loss, {"cms": float(loss)}


def train_benchmark_batch(model, libraries, batch, device):
    total_loss = None
    for study, n_rows in batch.items():
        lib, cell_ids = libraries[study]
        idx = torch.as_tensor(np.asarray(batch[study]))
        # rows are pairs packed [2N, L]: even = source, odd = candidate
        src_ids = lib.input_ids[idx * 2]
        cnd_ids = lib.input_ids[idx * 2 + 1]
        src_mask = lib.attention_mask[idx * 2]
        cnd_mask = lib.attention_mask[idx * 2 + 1]
        targets = lib.targets[idx].to(device)
        dom = torch.full((idx.shape[0],), lib.domain_id, dtype=torch.long, device=device)
        cells = None
        if cell_ids is not None:
            cells = cell_ids[idx].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            src_pred = model(src_ids.to(device), src_mask.to(device), dom, cells)
            cnd_pred = model(cnd_ids.to(device), cnd_mask.to(device), dom, cells)
            loss = F.mse_loss((cnd_pred - src_pred).float(), targets)
        total_loss = loss if total_loss is None else total_loss + loss
    if total_loss is None:
        raise RuntimeError("empty benchmark batch")
    return total_loss


if __name__ == "__main__":
    raise SystemExit(main())
