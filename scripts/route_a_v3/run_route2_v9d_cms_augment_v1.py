#!/usr/bin/env python3
"""Option-3 CMS-as-training-augmentation arm (user adjudicated 2026-09-09
batch 71 "再攻一轮"; amended launch design batch 76).

Hypothesis (D5 far-band readiness, batch 67): the MPRAU near-band bottleneck
is learning-side -- 0.27 adjudication needs only 793 validation pairs vs the
2,008 available; what is missing is TRAINING supervision scale. CMS (ENCODE
Cancer Mutation Survey array, 85,475 rows, cell-conditioned, same assay
family) is the only on-disk x10 same-lineage candidate.

THIS DIFFERS from the prior CMS prior-injection arms (h_cms_full / s_cms_full,
which trained on CMS ALONE and FAILED): here CMS is added ON TOP of the full
MPRAU in-domain training data of the s_mprau_in recipe. The control keeps its
data; CMS is a pure augmentation term.

Control (identical except CMS): v8_stage2_adapt_20260907/s_mprau_in/
stage2_s_benchmark_full_epoch6.pt
  - arch S, adapt full, init stage1_s_epoch2.pt (s_mrl-polya), seed 20260907
  - single domain ENCSR854RUF (55,704 rows), batch 64, 6 epochs, 5,226 steps,
    LR 2e-5 WD 1e-4, cell_conditioning (6 cells), FINAL_EPOCH_FIXED
  - reference values: single 0.1527 / 5-seed ensemble 0.1351 (pair_mean)

This arm (the ONLY deltas vs control):
  - two domains in the balanced sampler: ENCSR854RUF (55,704) + cms (~80K clean)
  - batch 128 => each domain contributes 64 rows per step, so the MPRAU
    per-step batch density equals the control's
  - epochs 12 => MPRAU domain sees 55,704 x 12 rows (>= control's x6);
    CMS sees the full library every epoch
  - training loss = MPRAU pair-delta MSE + CMS absolute z MSE (both z-scored)

PREREGISTERED GATE (committed BEFORE launch):
  - primary (D5 near band): MPRAU VALIDATION pair_mean_spearman > 0.1351
    (s_mprau_in 5-seed ensemble reference)
  - significance: paired per-variant bootstrap (2000 iters, seed 20260816)
    vs the same-seed no-CMS control (single 0.1527) over the 2,008 validation
    variants; CI of the pair-mean difference must exclude zero POSITIVE
  - honest-null clause: point > 0.1351 but CI crossing zero => "directional
    only"; point <= 0.1351 => FAIL regardless of CI

Discipline: CUDA BF16; FINAL-EPOCH-12-FIXED; protected reads=0 (CMS clean
rows only, audit_leak_flags applied inside load_cms_domain); products /mnt.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

V8_WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V8_WT))
sys.path.insert(0, str(V8_WT / "scripts/route_a_v3"))

_spec = importlib.util.spec_from_file_location(
    "r2", V8_WT / "scripts/route_a_v3/run_route2_v8_stage2_adapt_v1.py"
)
r2 = importlib.util.module_from_spec(_spec)
sys.modules["r2"] = r2
_spec.loader.exec_module(r2)

from core.route2_v8_joint_library_v1 import DomainBalancedSampler  # noqa: E402

MNT = r2.MNT
MRNABERT_PATH = r2.MRNABERT_PATH
CONTROL_CKPT = (
    MNT / "experiments/xeditcritic_route_a/v8_stage2_adapt_20260907/"
    "s_mprau_in/stage2_s_benchmark_full_epoch6.pt"
)
INIT_CKPT = (
    MNT / "experiments/xeditcritic_route_a/v8_stage1_joint_prefinetune_20260904/"
    "s_mrl-polya/stage1_s_epoch2.pt"
)
OUT_ROOT = MNT / "experiments/xeditcritic_route_a/v9d_cms_augment_20260910"

BATCH, EPOCHS, LR, WD, SEED = 128, 12, 2e-5, 1e-4, 20260907
MPRAU_STUDY = "ENCSR854RUF"


def train_step(model, libs, batch, device):
    """MPRAU pair-delta loss + CMS absolute loss (both z-scored, summed)."""
    total = None
    parts = {}
    # MPRAU benchmark domain: pairs packed [2N, L]
    lib, cell_ids = libs[MPRAU_STUDY]
    idx = torch.as_tensor(np.asarray(batch[MPRAU_STUDY]))
    src_ids = lib.input_ids[idx * 2]
    cnd_ids = lib.input_ids[idx * 2 + 1]
    src_mask = lib.attention_mask[idx * 2]
    cnd_mask = lib.attention_mask[idx * 2 + 1]
    targets = lib.targets[idx].to(device)
    dom = torch.full((idx.shape[0],), lib.domain_id, dtype=torch.long, device=device)
    cells = cell_ids[idx].to(device) if cell_ids is not None else None
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        sp = model(src_ids.to(device), src_mask.to(device), dom, cells)
        cp = model(cnd_ids.to(device), cnd_mask.to(device), dom, cells)
        mprau_loss = F.mse_loss((cp - sp).float(), targets)
    parts["mprau"] = float(mprau_loss)
    total = mprau_loss
    # CMS domain: single sequences + z-scored activity
    lib_c, cell_ids_c = libs["cms"]
    idx_c = torch.as_tensor(np.asarray(batch["cms"]))
    ids_c = lib_c.input_ids[idx_c]
    mask_c = lib_c.attention_mask[idx_c]
    targets_c = lib_c.targets[idx_c]
    cells_c = cell_ids_c[idx_c].to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred_c = model(
            ids_c.to(device),
            mask_c.to(device),
            torch.full((ids_c.shape[0],), lib_c.domain_id, dtype=torch.long, device=device),
            cells_c,
        )
        cms_loss = F.mse_loss(pred_c.float(), targets_c.to(device))
    parts["cms"] = float(cms_loss)
    total = total + cms_loss
    return total, parts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.physical_gpu_index}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MRNABERT_PATH, local_files_only=True)
    r2.verify_vocab_alignment(tokenizer)

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else OUT_ROOT / f"s_cmsaug_seed{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    r2._OUT_DIR = str(out_dir)

    # ---- model: identical to control (arch S, 9-domain geometry, 6 cells) ----
    num_domains = len(r2.DOMAIN_IDS)
    num_cells = r2.NUM_CELLS
    model = r2.build_v8_regressor(
        MRNABERT_PATH, "s", num_domains=num_domains, num_cells=num_cells
    ).to(device)
    init_summary = r2.load_init(model, INIT_CKPT)
    print(f"device={device} arch=s mode=cms_augment seed={args.seed}", flush=True)

    # ---- data: MPRAU benchmark domain (identical rows to control) + CMS ----
    bench = r2.load_benchmark_domains(tokenizer)
    libs = {MPRAU_STUDY: bench[MPRAU_STUDY]}
    cms_lib, cms_cells = r2.load_cms_domain(tokenizer)
    libs["cms"] = (cms_lib, cms_cells)
    domain_sizes = {
        MPRAU_STUDY: int(bench[MPRAU_STUDY][0].targets.shape[0]),
        "cms": int(cms_lib.n_clean),
    }
    print(f"[cms] clean rows: {cms_lib.n_clean}", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable):,}", flush=True)

    sampler = DomainBalancedSampler(
        domain_sizes=domain_sizes, batch_size=args.batch, seed=args.seed
    )
    planned = sampler.steps_per_epoch * args.epochs
    total_steps = args.max_steps if args.max_steps is not None else planned
    budget = {
        "arm": "s-cms-augment",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch": args.batch,
        "domain_sizes": domain_sizes,
        "steps_per_epoch": sampler.steps_per_epoch,
        "planned_total_steps": planned,
        "effective_total_steps": total_steps,
        "selection_rule": "FINAL_EPOCH_FIXED",
        "control_reference": {
            "checkpoint": str(CONTROL_CKPT),
            "recipe": (
                "arch S / full-FT / init s_mrl-polya stage1_s_epoch2 / single "
                "domain ENCSR854RUF 55704 rows / batch 64 / 6 epochs / 5226 steps "
                "/ LR 2e-5 WD 1e-4 / cell conditioning 6 / FINAL_EPOCH_FIXED"
            ),
            "pair_mean_single": 0.1527,
            "pair_mean_5seed_ensemble": 0.1351,
        },
        "only_deltas_vs_control": [
            "cms domain added to balanced sampler (~80K clean rows)",
            "batch 128 => per-domain 64 rows/step (MPRAU density equals control)",
            "epochs 12 => MPRAU sees 55704x12 rows (>= control x6), CMS full each epoch",
        ],
        "gate_preregistered": (
            "MPRAU VALIDATION pair_mean > 0.1351 AND paired per-variant bootstrap "
            "vs same-seed control CI excludes zero positive (2000 iters seed 20260816)"
        ),
    }
    print(f"budget: {json.dumps(budget)}", flush=True)

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.1
        + 0.9 * 0.5 * (1 + np.cos(np.pi * min(step / max(total_steps, 1), 1.0)))
        if step > total_steps * 0.05
        else step / max(total_steps * 0.05, 1),
    )

    loss_log = (out_dir / "training_losses.jsonl").open("w")
    epoch_eval = (out_dir / "epoch_eval_metrics.jsonl").open("w")

    def save_ckpt(name, ep, st, note):
        torch.save(
            {
                "schema_version": "route_a_v3_route2_v9d_cms_augment.v1",
                "model_state_dict": {k: v for k, v in model.state_dict().items()},
                "arm": "s-cms-augment",
                "seed": args.seed,
                "epochs_done": ep,
                "steps_done": st,
                "note": note,
            },
            out_dir / name,
        )

    def run_eval(ep, st, primary):
        if args.skip_eval:
            return {}
        model.eval()
        results = []
        for study, (domain, _rel) in r2.BENCHMARK_DOMAINS.items():
            if study == MPRAU_STUDY:
                continue
            rec = r2.eval_study_frozen_delta(model, tokenizer, device, study, domain)
            results.append(rec)
        mprau = r2.eval_mprau_validation(
            model, tokenizer, device, r2.DOMAIN_IDS["mprau"]
        )
        results.append({"domain": "mprau", **mprau})
        report = {
            "epoch": ep,
            "steps": st,
            "primary": primary,
            "task_macro_spearman": r2._task_macro(results),
            "per_task": results,
            "mprau": mprau,
        }
        epoch_eval.write(json.dumps(report) + "\n")
        epoch_eval.flush()
        model.train()
        print(
            f"== epoch {ep} eval: "
            + json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k not in ("per_task",)
                }
            )[:400],
            flush=True,
        )
        return report

    model.train()
    step = 0
    stopped = False
    final_eval = {}
    for epoch in range(args.epochs):
        for batch in sampler.epoch_batches(epoch):
            loss, parts = train_step(model, libs, batch, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 50 == 0:
                loss_log.write(
                    json.dumps(
                        {
                            "step": step,
                            "epoch": epoch + 1,
                            "mse_total": float(loss),
                            "mse_mprau": parts["mprau"],
                            "mse_cms": parts["cms"],
                            "lr": scheduler.get_last_lr()[0],
                        }
                    )
                    + "\n"
                )
                loss_log.flush()
            if args.max_steps is not None and step >= args.max_steps:
                stopped = True
                break
        if stopped:
            save_ckpt(f"v9d_smoke{step}.pt", epoch + 1, step, "smoke")
            final_eval = run_eval(epoch + 1, step, False)
            break
        save_ckpt(
            f"v9d_epoch{epoch + 1}.pt",
            epoch + 1,
            step,
            "final" if epoch + 1 == args.epochs else "epoch",
        )
        final_eval = run_eval(epoch + 1, step, primary=(epoch + 1 == args.epochs))

    loss_log.close()
    epoch_eval.close()
    (out_dir / "run_report.json").write_text(
        json.dumps(
            {
                "schema_version": "route_a_v3_route2_v9d_cms_augment.v1",
                "arm": "s-cms-augment",
                "seed": args.seed,
                "selection_rule": "FINAL_EPOCH_FIXED" if not stopped else "SMOKE",
                "init": init_summary,
                "budget": budget,
                "steps_done": step,
                "smoke": bool(args.max_steps),
                "cpu_fallback_used": False,
                "eval_final": final_eval,
            },
            indent=1,
            default=str,
        )
    )
    print(f"wrote {out_dir / 'run_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
