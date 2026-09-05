#!/usr/bin/env python3
"""V8 Stage 1: joint external-library pre-finetuning, arms S (pure) / H (hybrid).

Spec: SPECS_CRITIC_V6 "V8 攻坚线" Stage 1 -- joint pre-finetuning on the union of
external large libraries (MRL 280K + polyA APA, CMS array stub reserved), three
arms at the same library / budget / seed:

- Arm S: pure mRNABERT (12L, 768d, raw pretrained init) + domain-conditional
  readout + masked-mean-pool + linear head.
- Arm H: Arm S + Optimus-style CNN motif stem (residual injection into the word
  embeddings; zero-initialised projection, see core/route2_v8_hybrid_backbone_v1).
- Arm M (control, NOT run here): Route A 280K fullft_v2 MRL-only reproduction --
  already terminal (3 seeds 0.3198 / 0.2873 / 0.3157, mean 0.3076, 3-seed
  ensemble 0.3158 on GSE114002 VALIDATION frozen-delta); cited, not re-run.
  See experiments/analysis_fullft_v2_adjudication_20260903.

Training recipe (mirrors the 280K fullft_v2 pre-finetune recipe): full-parameter
fine-tuning, MSE on per-domain z-scored activities, AdamW lr 2e-5 wd 1e-4,
batch 128, cosine to 10% after 5% warmup, BF16 autocast, seed 20260903
(aligned with Route A). Domain-balanced batches (equal domain quotas per batch).

Zero-shot evaluation (diagnostic per epoch; PRIMARY judgment = FINAL-EPOCH-FIXED):
frozen-delta f(candidate) - f(source) with f = this model under the corresponding
domain conditioning, on the benchmark VALIDATION split only (TEST untouched):
- mrl   -> GSE114002 VALIDATION (730 records), frozen Task-1 evaluator K=10.
- polya -> GSE269595 VALIDATION (2,628 records), same evaluator.

Preregistration: docs/paper/route2_v8_stage1_prereg_v1.md (judgment gates, budget
accounting, S-vs-H adjudication rule). No peak-picking (H2 red line).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_v8_hybrid_backbone_v1 import (  # noqa: E402
    DOMAIN_IDS,
    NUM_DOMAINS,
    build_v8_regressor,
    parameter_report,
    verify_vocab_alignment,
)
from core.route2_v8_joint_library_v1 import (  # noqa: E402
    CMS_ARRAY_CSV,
    MNT,
    MRL_LIB_DIR,
    POLYA_LIB_GZ,
    DomainBalancedSampler,
    DomainLibrary,
    audit_leak_flags,
    build_protected_index,
    format_sequence,
    load_cms_library,
    load_mrl_library,
    load_polya_library,
    prepare_domain_library,
    resolve_libraries,
)

EVAL_REPO = "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901"
_ev_spec = importlib.util.spec_from_file_location(
    "ev", EVAL_REPO + "/scripts/route_a_v3/evaluate_route2_prediction_v1.py"
)
ev = importlib.util.module_from_spec(_ev_spec)
sys.modules["ev"] = ev
_ev_spec.loader.exec_module(ev)

MRNABERT_PATH = MNT / "external_model_assets/mrnabert_a1eb7df25804d23f08646e1cb996b234d7208a40"
MANIFEST = MNT / "manifests/route2_development_frozen_v1/development_manifest.jsonl"
CANONICAL = {
    "mrl": MNT / "canonical/GSE114002/v1/canonical_records.private.jsonl",
    "polya": MNT / "canonical/GSE269595/v1/canonical_records.private.jsonl",
}
EVAL_STUDY = {"mrl": "GSE114002", "polya": "GSE269595"}
OUT_ROOT = MNT / "experiments/xeditcritic_route_a/v8_stage1_joint_prefinetune_20260904"

BATCH = 128
EPOCHS = 2
LR = 2e-5
WEIGHT_DECAY = 1e-4
SEED = 20260903
EVAL_BATCH = 256
K = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--arch", required=True, choices=("s", "h"), help="S = pure mRNABERT, H = CNN-stem hybrid")
    parser.add_argument("--libraries", default="mrl,polya", help="comma list from {mrl,polya,cms}")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--num-domains", type=int, default=NUM_DOMAINS)
    parser.add_argument("--max-steps", type=int, default=None, help="smoke cap: stop training after N steps")
    parser.add_argument("--out-dir", default=None, help="default: OUT_ROOT/{arch}_{libraries}")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-train", action="store_true", help="load/audit/tokenise only")
    return parser.parse_args()


def load_raw_library(domain: str):
    if domain == "mrl":
        return load_mrl_library(MRL_LIB_DIR)
    if domain == "polya":
        return load_polya_library(POLYA_LIB_GZ)
    if domain == "cms":
        # Only reachable once the CMS array CSV exists (resolve_libraries gates it).
        sequences, activities, _contexts = load_cms_library(CMS_ARRAY_CSV)
        return sequences, activities
    raise ValueError(f"no raw loader for domain {domain!r}")


def assemble_batch(libraries: dict[str, DomainLibrary], batch: dict[str, np.ndarray], device: torch.device):
    """Concatenate per-domain chunks into one padded batch (domain-block layout)."""
    max_len = max(libraries[d].input_ids.shape[1] for d in batch)
    ids_chunks, mask_chunks, target_chunks, domain_chunks, spans = [], [], [], [], []
    offset = 0
    for domain in sorted(batch, key=lambda d: DOMAIN_IDS[d]):
        lib = libraries[domain]
        idx = torch.as_tensor(np.asarray(batch[domain]))
        ids = lib.input_ids[idx]
        mask = lib.attention_mask[idx]
        pad = max_len - ids.shape[1]
        if pad > 0:
            ids = F.pad(ids, (0, pad))
            mask = F.pad(mask, (0, pad))
        ids_chunks.append(ids)
        mask_chunks.append(mask)
        target_chunks.append(lib.targets[idx])
        domain_chunks.append(torch.full((idx.shape[0],), lib.domain_id, dtype=torch.long))
        spans.append((domain, offset, offset + idx.shape[0]))
        offset += idx.shape[0]
    return (
        torch.cat(ids_chunks).to(device, non_blocking=True),
        torch.cat(mask_chunks).to(device, non_blocking=True),
        torch.cat(target_chunks).to(device, non_blocking=True),
        torch.cat(domain_chunks).to(device, non_blocking=True),
        spans,
    )


def zeroshot_frozen_delta(model, tokenizer, domain: str, device: torch.device) -> dict:
    """Frozen-delta on the domain's benchmark VALIDATION split (K=10 evaluator)."""
    study = EVAL_STUDY[domain]
    canonical = CANONICAL[domain]
    domain_id = DOMAIN_IDS[domain]
    validation_ids = set()
    with MANIFEST.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["study_unit_id"] == study and row["split"] == "VALIDATION":
                validation_ids.add(str(row["canonical_record_id"]))
    records = {}
    with canonical.open() as handle:
        for line in handle:
            row = json.loads(line)
            rid = str(row.get("canonical_record_id"))
            if rid in validation_ids:
                records[rid] = row
    ids = sorted(records)
    observations = ev.load_observations([canonical], validation_ids)

    def score(sequences: list[str]) -> np.ndarray:
        model.eval()
        values = []
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for start in range(0, len(sequences), EVAL_BATCH):
                chunk = sequences[start : start + EVAL_BATCH]
                enc = tokenizer(
                    [format_sequence(s) for s in chunk],
                    add_special_tokens=True, padding=True, truncation=True, max_length=512, return_tensors="pt",
                )
                out = model(
                    enc["input_ids"].to(device),
                    enc["attention_mask"].to(device),
                    torch.full((len(chunk),), domain_id, dtype=torch.long, device=device),
                )
                values.append(out.float().cpu().numpy())
        return np.concatenate(values)

    source = score([records[rid]["source_sequence"] for rid in ids])
    candidate = score([records[rid]["candidate_sequence"] for rid in ids])
    delta = candidate - source
    predictions = {rid: float(delta[i]) for i, rid in enumerate(ids)}
    metrics = ev.evaluate(observations, predictions, K)
    model.train()
    return {
        "domain": domain,
        "study": study,
        "split": "VALIDATION",
        "n_records": len(ids),
        "task_macro_spearman": metrics.get("task_macro_spearman"),
        "top_1": metrics.get("source_macro_top_1_accuracy"),
        "ndcg_at_10": metrics.get("source_macro_ndcg_at_k"),
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - GPU required")
    device = torch.device(f"cuda:{args.physical_gpu_index}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    requested = [name.strip() for name in args.libraries.split(",") if name.strip()]
    active, skipped = resolve_libraries(requested)
    if not active:
        raise SystemExit(f"no active libraries (requested={requested}, skipped={skipped})")
    if any(s["domain"] == "cms" for s in skipped):
        print(f"[cms] stub skipped: {skipped} (interface frozen at {CMS_ARRAY_CSV})", flush=True)

    out_dir = Path(args.out_dir) if args.out_dir else OUT_ROOT / f"{args.arch}_{'-'.join(active)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"arch={args.arch} libraries={active} out={out_dir}", flush=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MRNABERT_PATH, local_files_only=True)
    verify_vocab_alignment(tokenizer)

    protected_index = build_protected_index()
    libraries: dict[str, DomainLibrary] = {}
    for domain in active:
        sequences, activities = load_raw_library(domain)
        flags = audit_leak_flags(sequences, protected_index)
        libraries[domain] = prepare_domain_library(domain, sequences, activities, flags, tokenizer)
        summary = libraries[domain].audit_summary()
        print(f"[{domain}] {json.dumps(summary)}", flush=True)

    audit_report = {
        "schema_version": "route_a_v3_route2_v8_stage1_leakage_audit.v1",
        "protected_studies": sorted(protected_index),
        "domains": {d: libraries[d].audit_summary() for d in active},
        "skipped": skipped,
    }
    (out_dir / "leakage_audit.json").write_text(json.dumps(audit_report, indent=1, sort_keys=True))
    if args.skip_train:
        print("skip-train: audit complete, exiting")
        return 0

    model = build_v8_regressor(MRNABERT_PATH, args.arch, num_domains=args.num_domains).to(device)
    params = parameter_report(model)
    print(f"parameter report: {json.dumps(params)}", flush=True)

    sampler = DomainBalancedSampler(
        domain_sizes={d: libraries[d].n_clean for d in active},
        batch_size=args.batch,
        seed=args.seed,
    )
    planned_steps = sampler.steps_per_epoch * args.epochs
    total_steps = args.max_steps if args.max_steps is not None else planned_steps
    draws_per_epoch = sampler.domain_draws_per_epoch()
    budget = {
        "epochs": args.epochs,
        "batch": args.batch,
        "steps_per_epoch": sampler.steps_per_epoch,
        "planned_total_steps": planned_steps,
        "max_steps_cap": args.max_steps,
        "effective_total_steps": total_steps,
        "domain_library_sizes": {d: libraries[d].n_clean for d in active},
        "domain_draws_per_epoch": draws_per_epoch,
        "domain_passes_equivalent": {
            d: draws_per_epoch[d] * args.epochs / libraries[d].n_clean for d in active
        },
        "reference_m_arm": {"epochs": 6, "steps": 31752, "mrl_passes": 6.0},
    }
    print(f"budget: {json.dumps(budget)}", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(step / max(total_steps, 1), 1.0))) if step > total_steps * 0.05 else step / max(total_steps * 0.05, 1),
    )

    loss_log = (out_dir / "training_losses.jsonl").open("w")
    epoch_domain_loss = (out_dir / "epoch_domain_loss.jsonl").open("w")
    epoch_zeroshot = (out_dir / "epoch_zeroshot_metrics.jsonl").open("w")

    def save_checkpoint(name: str, epoch_done: int, steps_done: int, note: str) -> None:
        torch.save(
            {
                "schema_version": "route_a_v3_route2_v8_stage1_joint_prefinetune.v1",
                "model_state_dict": {k: v for k, v in model.state_dict().items()},
                "arch": args.arch,
                "libraries": active,
                "domain_stats": {
                    d: {"mean": libraries[d].target_mean, "std": libraries[d].target_std,
                        "n_clean": libraries[d].n_clean, "domain_id": libraries[d].domain_id}
                    for d in active
                },
                "seed": args.seed,
                "epochs_done": epoch_done,
                "steps_done": steps_done,
                "note": note,
            },
            out_dir / name,
        )

    def run_zeroshot(epoch_done: int, steps_done: int, primary: bool) -> list[dict]:
        if args.skip_eval:
            return []
        results = []
        for domain in active:
            metrics = zeroshot_frozen_delta(model, tokenizer, domain, device)
            rec = {"epoch": epoch_done, "steps": steps_done, "primary": primary, **metrics}
            epoch_zeroshot.write(json.dumps(rec) + "\n")
            epoch_zeroshot.flush()
            results.append(rec)
            print(f"== epoch {epoch_done} zeroshot[{domain}]: spearman {metrics['task_macro_spearman']:.4f} top1 {metrics['top_1']:.4f} ndcg10 {metrics['ndcg_at_10']:.4f}", flush=True)
        return results

    model.train()
    step = 0
    epoch_completed = 0
    stopped_early = False
    final_zeroshot: list[dict] = []
    final_domain_mse: dict[str, float] = {}
    recent: dict[str, list[float]] = {d: [] for d in active}
    for epoch in range(args.epochs):
        epoch_losses: dict[str, list[float]] = {d: [] for d in active}
        for batch in sampler.epoch_batches(epoch):
            ids, mask, targets, domain_ids, spans = assemble_batch(libraries, batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = model(ids, mask, domain_ids)
                per_domain_mse = {}
                for domain, start, end in spans:
                    mse = F.mse_loss(prediction[start:end].float(), targets[start:end])
                    per_domain_mse[domain] = float(mse)
                    epoch_losses[domain].append(float(mse))
                loss = F.mse_loss(prediction.float(), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            for domain, value in per_domain_mse.items():
                recent[domain].append(value)
            if step % 50 == 0:
                rec = {"step": step, "epoch": epoch + 1, "mse_all": float(loss),
                       **{f"mse_{d}": float(np.mean(recent[d][-50:])) for d in active},
                       "lr": scheduler.get_last_lr()[0]}
                loss_log.write(json.dumps(rec) + "\n")
                loss_log.flush()
                if step % 500 == 0:
                    print(f"epoch {epoch + 1} step {step}: mse {float(loss):.4f} " +
                          " ".join(f"{d}={np.mean(recent[d][-50:]):.4f}" for d in active), flush=True)
            if args.max_steps is not None and step >= args.max_steps:
                stopped_early = True
                break
        epoch_completed = epoch + 1
        final_domain_mse = {d: float(np.mean(epoch_losses[d])) for d in active}
        rec = {"epoch": epoch_completed, "steps": step, "partial": stopped_early,
               **{f"mse_{d}": final_domain_mse[d] for d in active}}
        epoch_domain_loss.write(json.dumps(rec) + "\n")
        epoch_domain_loss.flush()
        print(f"== epoch {epoch_completed} domain mse: {json.dumps(rec)}", flush=True)
        if stopped_early:
            save_checkpoint(f"stage1_{args.arch}_smoke_steps{step}.pt", epoch_completed, step,
                            "smoke max-steps stop (non-terminal)")
            final_zeroshot = run_zeroshot(epoch_completed, step, primary=False)
            break
        save_checkpoint(f"stage1_{args.arch}_epoch{epoch_completed}.pt", epoch_completed, step,
                        "final" if epoch_completed == args.epochs else "epoch")
        final_zeroshot = run_zeroshot(epoch_completed, step, primary=(epoch_completed == args.epochs))

    loss_log.close()
    epoch_domain_loss.close()
    epoch_zeroshot.close()

    report = {
        "schema_version": "route_a_v3_route2_v8_stage1_joint_prefinetune.v1",
        "mode": f"ROUTE_A_V8_STAGE1_ARCH_{args.arch.upper()}",
        "arch": args.arch,
        "libraries": active,
        "skipped_domains": skipped,
        "selection_rule": "FINAL_EPOCH_FIXED" if not stopped_early else "SMOKE_MAX_STEPS (non-terminal, diagnostics only)",
        "seed": args.seed,
        "params": params,
        "budget": budget,
        "steps_done": step,
        "epochs_completed": epoch_completed,
        "smoke": bool(args.max_steps is not None),
        "per_domain_final_train_mse": final_domain_mse,
        "zeroshot_final": final_zeroshot,
        "zeroshot_evaluated": not args.skip_eval,
    }
    (out_dir / "run_report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({k: report[k] for k in ("arch", "libraries", "steps_done", "epochs_completed")}, indent=1), flush=True)
    print("wrote", out_dir / "run_report.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
