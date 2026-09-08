#!/usr/bin/env python3
"""V9-1b data-arm training (SPECS_CRITIC_V6 spec N.4 Task 15.5; prereg
route2_v9b_data_arm_prereg_v1.md FROZEN 2026-09-09).

Two arms over the frozen holdout splits (analysis_track_b_20260908/*_holdout_manifest.json):
- arm 1 (pure-v9b): S1+M6 TRAIN_SPLIT only (specialist analogue)
- arm 2 (bench-v9b): benchmark 9 domains + S1 + M6 (unified 11-domain model; main arm)

Geometry: V9 adapter-zoo extended -- n_domains=11 (benchmark 0-8 + stability_v9b=9
+ ndd_te_v9b=10), num_cells=8 (6 ENCODE + SH=6 + HEK=7; M6 HEK~HEK293FT近似声明).
Data: S1 (155nt transcript-direction windows, dual-assay rows) / M6 (100nt,
total-count log2FC v0 calibre). z-scored per library; pair-delta MSE identical
to V9-1a. Gates D1/D3 per prereg; probe D2 observational only.

Discipline: CUDA BF16; FINAL-EPOCH-6-FIXED; protected reads=0; products /mnt.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

V8_WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V8_WT))
sys.path.insert(0, str(V8_WT / "scripts/route_a_v3"))

_spec = importlib.util.spec_from_file_location("r2", V8_WT / "scripts/route_a_v3/run_route2_v8_stage2_adapt_v1.py")
r2 = importlib.util.module_from_spec(_spec)
sys.modules["r2"] = r2
_spec.loader.exec_module(r2)

_spec9 = importlib.util.spec_from_file_location("v9run", V8_WT / "scripts/route_a_v3/run_route2_v9_adapter_zoo_v1.py")
v9run = importlib.util.module_from_spec(_spec9)
sys.modules["v9run"] = v9run
_spec9.loader.exec_module(v9run)

from core.route2_v8_joint_library_v1 import DomainBalancedSampler  # noqa: E402

MNT = r2.MNT
TRACK_B = MNT / "experiments/analysis_track_b_20260908"
S1_LIB = TRACK_B / "s1_stability_pairs.jsonl"
M6_LIB = TRACK_B / "m6_ndd_translation_pairs.jsonl"
S1_HOLDOUT = TRACK_B / "s1_holdout_manifest.json"
M6_HOLDOUT = TRACK_B / "m6_holdout_manifest.json"
OUT_ROOT = MNT / "experiments/xeditcritic_route_a/v9b_data_arm_20260909"

DOMAIN_STABILITY = 9
DOMAIN_NDD = 10
N_DOMAINS = 11
CELL_SH, CELL_HEK = 6, 7
NUM_CELLS = 8
EPOCHS, LR, WD, BATCH = 6, 2e-5, 1e-4, 128


def load_v9b_libraries() -> dict[str, tuple[list[dict], torch.Tensor, torch.Tensor]]:
    """domain -> (rows, packed input_ids [2N,L], targets z [N], cell_ids [N]) as
    a plain list-of-tuples library compatible with a custom batch sampler."""
    s1_hold = set(json.load(open(S1_HOLDOUT))["train_split_ids"])
    m6_hold = set(json.load(open(M6_HOLDOUT))["train_split_ids"])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(r2.MRNABERT_PATH, local_files_only=True)
    libs: dict[str, tuple] = {}

    s1_rows = [json.loads(l) for l in S1_LIB.open()]
    for assay, cell_id, col in (("sh", CELL_SH, "y_sh"), ("hek", CELL_HEK, "y_hek")):
        rows = [r for r in s1_rows if r["variant_id"] in s1_hold and r.get(col) is not None]
        if not rows:
            continue
        libs[f"stability_{assay}"] = _pack(tok, rows, col, DOMAIN_STABILITY, cell_id)
    m6_rows = [json.loads(l) for l in M6_LIB.open()]
    m6_rows = [r for r in m6_rows if r["variant_id"] in m6_hold]
    libs["ndd_te"] = _pack(tok, m6_rows, "y_totalcount_log2fc", DOMAIN_NDD, CELL_HEK)
    return libs


def _pack(tok, rows, y_col, domain_id, cell_id):
    seqs = []
    for r in rows:
        seqs.append(r["source_sequence"])
        seqs.append(r["candidate_sequence"])
    enc = tok(seqs, add_special_tokens=True, padding=True, truncation=True,
              max_length=512, return_tensors="pt")
    y = np.asarray([float(r[y_col]) for r in rows], dtype=np.float64)
    z = ((y - y.mean()) / (y.std() if y.std() > 0 else 1.0)).astype(np.float32)
    return {"rows": rows, "input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
            "targets": torch.tensor(z, dtype=torch.float32),
            "cell_ids": torch.full((len(rows),), cell_id, dtype=torch.long),
            "domain_id": domain_id, "n": len(rows), "y_col": y_col}


def train_batch(model, lib, idx, device):
    # pairs are packed [2N, L]: even = source, odd = candidate (V8 convention)
    src_ids = lib["input_ids"][idx * 2]
    cnd_ids = lib["input_ids"][idx * 2 + 1]
    src_mask = lib["attention_mask"][idx * 2]
    cnd_mask = lib["attention_mask"][idx * 2 + 1]
    targets = lib["targets"][idx].to(device)
    dom = torch.full((idx.shape[0],), lib["domain_id"], dtype=torch.long, device=device)
    cells = lib["cell_ids"][idx].to(device) if lib.get("cell_ids") is not None else None
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        sp = model(src_ids.to(device), src_mask.to(device), dom, cells)
        cp = model(cnd_ids.to(device), cnd_mask.to(device), dom, cells)
        loss = torch.nn.functional.mse_loss((cp - sp).float(), targets)
    return loss


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--arm", required=True, choices=("pure-v9b", "bench-v9b"))
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.physical_gpu_index}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(r2.MRNABERT_PATH, local_files_only=True)
    r2.verify_vocab_alignment(tok)

    out_dir = Path(args.out_dir) if args.out_dir else OUT_ROOT / f"{args.arm}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    r2._OUT_DIR = str(out_dir)

    # build model with extended geometry
    model = v9run.build_v9(r2.MRNABERT_PATH, num_cells=NUM_CELLS)
    # patch geometry: n_domains 11 -> rebuild domain embeddings + heads
    import torch.nn as nn
    model.domain_embeddings = nn.Embedding(N_DOMAINS, model.base.config.hidden_size)
    nn.init.normal_(model.domain_embeddings.weight, mean=0.0, std=0.02)
    model.cell_embeddings = nn.Embedding(NUM_CELLS, model.base.config.hidden_size)
    nn.init.normal_(model.cell_embeddings.weight, mean=0.0, std=0.02)
    model.heads = nn.ModuleList([nn.Linear(model.base.config.hidden_size, 1) for _ in range(N_DOMAINS)])
    init = v9run.load_v9_init(model)
    for p in model.base.parameters():
        p.requires_grad_(False)
    # patch module-global BEFORE wrap: MultiTaskLoRALinear task count must be 11
    v9run.NUM_DOMAINS = N_DOMAINS
    wrapped = model.wrap_multitask_lora()
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]

    # data
    v9b = load_v9b_libraries()
    bench = r2.load_benchmark_domains(tok) if args.arm == "bench-v9b" else {}
    # unified sampler space: domain sizes
    domain_sizes = {}
    libs = {}
    for name, lib in v9b.items():
        domain_sizes[name] = lib["n"]
        libs[name] = lib
    for study, (lib, _c) in bench.items():
        domain_sizes[study] = int(lib.targets.shape[0])
        libs[study] = {"input_ids": lib.input_ids, "attention_mask": lib.attention_mask,
                       "targets": lib.targets, "cell_ids": _c, "domain_id": lib.domain_id,
                       "n": int(lib.targets.shape[0]), "bench": True}
    sampler = DomainBalancedSampler(domain_sizes=domain_sizes, batch_size=BATCH, seed=args.seed)
    planned = sampler.steps_per_epoch * args.epochs
    total_steps = args.max_steps if args.max_steps is not None else planned
    budget = {"arm": args.arm, "seed": args.seed, "epochs": args.epochs,
              "domain_sizes": domain_sizes, "steps_per_epoch": sampler.steps_per_epoch,
              "planned_total_steps": planned, "effective_total_steps": total_steps,
              "selection_rule": "FINAL_EPOCH_FIXED", "geometry": {"n_domains": N_DOMAINS, "num_cells": NUM_CELLS}}
    print(f"budget: {json.dumps(budget)}", flush=True)

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(step / max(total_steps, 1), 1.0)))
        if step > total_steps * 0.05 else step / max(total_steps * 0.05, 1))

    loss_log = (out_dir / "training_losses.jsonl").open("w")
    epoch_eval = (out_dir / "epoch_eval_metrics.jsonl").open("w")

    def save_ckpt(name, ep, st, note):
        torch.save({"schema_version": "route_a_v3_route2_v9b_data_arm.v1",
                    "model_state_dict": {k: v for k, v in model.state_dict().items()},
                    "arm": args.arm, "seed": args.seed, "epochs_done": ep, "steps_done": st,
                    "note": note, "init": init, "budget": budget}, out_dir / name)

    def run_eval(ep, st, primary):
        if args.skip_eval:
            return {}
        model.eval()
        report = {"epoch": ep, "steps": st, "primary": primary}
        # gate D3: v9b holdout eval (S1/M6 VALIDATION_SPLIT)
        hold = {}
        for name, path, col, dom, cell in (
            ("s1_sh", S1_LIB, "y_sh", DOMAIN_STABILITY, CELL_SH),
            ("s1_hek", S1_LIB, "y_hek", DOMAIN_STABILITY, CELL_HEK),
            ("m6", M6_LIB, "y_totalcount_log2fc", DOMAIN_NDD, CELL_HEK)):
            val_ids = set(json.load(open(S1_HOLDOUT if name.startswith("s1") else M6_HOLDOUT))["validation_split_ids"])
            rows = [json.loads(l) for l in path.open()]
            rows = [r for r in rows if r["variant_id"] in val_ids and r.get(col) is not None]
            if not rows:
                continue
            from scipy.stats import spearmanr
            src = r2.score_sequences(model, tok, device, [r["source_sequence"] for r in rows], dom,
                                     torch.tensor([cell] * len(rows)))
            cnd = r2.score_sequences(model, tok, device, [r["candidate_sequence"] for r in rows], dom,
                                     torch.tensor([cell] * len(rows)))
            pred = np.array(cnd) - np.array(src)
            tgt = np.array([float(r[col]) for r in rows])
            hold[name] = {"n": len(rows), "spearman_holdout": float(spearmanr(tgt, pred).statistic)}
        report["v9b_holdout"] = hold
        # gate D1: benchmark 9-task table (bench-v9b arm only; pure arm reports too for cross-domain transfer)
        if args.arm == "bench-v9b" or True:
            results = []
            for study, (domain, _rel) in r2.BENCHMARK_DOMAINS.items():
                if study == "ENCSR854RUF":
                    continue
                rec = r2.eval_study_frozen_delta(model, tok, device, study, domain)
                results.append(rec)
            mprau = r2.eval_mprau_validation(model, tok, device, r2.DOMAIN_IDS["mprau"])
            results.append({"domain": "mprau", **mprau})
            report["task_macro_spearman"] = r2._task_macro(results)
            report["per_task"] = results
            report["mprau"] = mprau
        epoch_eval.write(json.dumps(report) + "\n")
        epoch_eval.flush()
        model.train()
        print(f"== epoch {ep} eval: {json.dumps({k: v for k, v in report.items() if k not in ('per_task',)})[:400]}", flush=True)
        return report

    model.train()
    step = 0
    stopped = False
    final_eval = {}
    for epoch in range(args.epochs):
        for batch in sampler.epoch_batches(epoch):
            total = None
            for dname, row_idx in batch.items():
                lib = libs[dname]
                idx = torch.as_tensor(np.asarray(row_idx))
                if lib.get("bench"):
                    # bench libs pack pairs [2N, L]
                    loss = train_batch(model, lib, idx, device)
                else:
                    loss = train_batch(model, lib, idx, device)
                total = loss if total is None else total + loss
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 50 == 0:
                loss_log.write(json.dumps({"step": step, "epoch": epoch + 1, "mse_all": float(total),
                                            "lr": scheduler.get_last_lr()[0]}) + "\n")
                loss_log.flush()
            if args.max_steps is not None and step >= args.max_steps:
                stopped = True
                break
        if stopped:
            save_ckpt(f"v9b_smoke{step}.pt", epoch + 1, step, "smoke")
            final_eval = run_eval(epoch + 1, step, False)
            break
        save_ckpt(f"v9b_epoch{epoch + 1}.pt", epoch + 1, step,
                  "final" if epoch + 1 == args.epochs else "epoch")
        final_eval = run_eval(epoch + 1, step, primary=(epoch + 1 == args.epochs))

    loss_log.close()
    epoch_eval.close()
    (out_dir / "run_report.json").write_text(json.dumps({
        "schema_version": "route_a_v3_route2_v9b_data_arm.v1", "arm": args.arm, "seed": args.seed,
        "selection_rule": "FINAL_EPOCH_FIXED" if not stopped else "SMOKE", "init": init,
        "budget": budget, "steps_done": step, "smoke": bool(args.max_steps),
        "cpu_fallback_used": False, "eval_final": final_eval}, indent=1, default=str))
    print(f"wrote {out_dir / 'run_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
