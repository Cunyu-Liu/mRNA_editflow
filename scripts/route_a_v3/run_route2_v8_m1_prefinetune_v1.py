#!/usr/bin/env python3
"""V8.5 M1 joint pre-finetune: mrl(280K) + m1(MRL Cast-Hair 3-cell) + polya.

Follows SPECS_CRITIC_V6 Stage 1 recipe and reuses its core verbatim. New domain
slot `m1` reuses domain id 0 (MRL readout family) via its own DomainLibrary with
domain_id=0; cell conditioning via M1 loader's context tokens is NOT used in
Stage-1 geometry (num_cells=0), matching the existing v8 geometry.

Init: from v8 Stage-1 winner S (stage1_s_epoch2.pt, mrl+polya). New optimizer;
2 epochs domain-balanced; frozen-delta zero-shot on unchanged GSE114002 /
GSE269595 VALIDATION each epoch (primary = final epoch fixed).

Acceptance gate (prereg): MRL task_macro_spearman >= 0.3078 - 0.02 AND polya >=
0.5262 - 0.02 at final epoch; report delta vs S-winner baseline.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

WT = Path("/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_v8_stage1_prep_20260904")
if str(WT) not in sys.path:
    sys.path.insert(0, str(WT))

from core.route2_v8_hybrid_backbone_v1 import (  # noqa: E402
    DOMAIN_IDS,
    build_v8_regressor,
    parameter_report,
    verify_vocab_alignment,
)
from core.route2_v8_joint_library_v1 import (  # noqa: E402
    MNT,
    DomainBalancedSampler,
    DomainLibrary,
    audit_leak_flags,
    build_protected_index,
    format_sequence,
    prepare_domain_library,
    standardize,
)

EVAL_REPO = "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901"
_ev_spec = importlib.util.spec_from_file_location(
    "ev", EVAL_REPO + "/scripts/route_a_v3/evaluate_route2_prediction_v1.py"
)
ev = importlib.util.module_from_spec(_ev_spec)
sys.modules["ev"] = ev
_ev_spec.loader.exec_module(ev)

BASE_STAGE1_SCRIPT = WT / "scripts/route_a_v3/run_route2_v8_stage1_joint_prefinetune_v1.py"
_b_spec = importlib.util.spec_from_file_location("v8stage1", str(BASE_STAGE1_SCRIPT))
v8stage1 = importlib.util.module_from_spec(_b_spec)
_b_spec.loader.exec_module(v8stage1)
v8stage1.DOMAIN_IDS = dict(v8stage1.DOMAIN_IDS)
v8stage1.DOMAIN_IDS["m1"] = 0
zeroshot_frozen_delta = v8stage1.zeroshot_frozen_delta
assemble_batch = v8stage1.assemble_batch

MRNABERT_PATH = MNT / "external_model_assets/mrnabert_a1eb7df25804d23f08646e1cb996b234d7208a40"
M1_DIR = MNT / "external_model_assets/castillohair2024/mrl_converted_stable"
INIT_CKPT = MNT / "experiments/xeditcritic_route_a/v8_stage1_joint_prefinetune_20260904/s_mrl-polya/stage1_s_epoch2.pt"
OUT_ROOT = MNT / "experiments/xeditcritic_route_a/v8_m1_prefinetune_20260909"

M1_FILES = [
    "GSE232927_processed_defined_end_hepg2_r1_mrl.tsv.gz",
    "GSE232927_processed_defined_end_tcell_r1_mrl.tsv.gz",
    "GSE232927_processed_defined_end_tcell_r2_mrl.tsv.gz",
    "GSE232927_processed_random_end_hek293t_N25_r1_mrl.tsv.gz",
    "GSE232927_processed_random_end_hek293t_N25_r2_mrl.tsv.gz",
    "GSE232927_processed_random_end_hek293t_N50_r1_mrl.tsv.gz",
]
M1_MIN_TOTAL = 100  # conversion already applied total>=100; keep=1 rows only

BATCH = 128
EPOCHS = 2
LR = 1e-5
WEIGHT_DECAY = 1e-4
SEED = 20260903
EVAL_BATCH = 256
K = 10


def load_m1_library(m1_dir: Path = M1_DIR, min_total: int = M1_MIN_TOTAL):
    import gzip, csv
    per_file: dict[str, dict[str, float]] = {}
    for name in M1_FILES:
        agg: dict[str, float] = {}
        with gzip.open(m1_dir / name, "rt") as fh:
            r = csv.reader(fh, delimiter="\t")
            hdr = next(r)
            idx = {h: i for i, h in enumerate(hdr)}
            for row in r:
                if row[idx["keep"]] != "1":
                    continue
                agg[row[idx["utr"]]] = float(row[idx["rl_paper"]])
        per_file[name] = agg
    merged: dict[str, float] = {}
    for name, agg in per_file.items():
        for s, v in agg.items():
            merged.setdefault(s, []).append(v) if isinstance(merged.get(s), list) else None
    # redo merge properly
    buckets: dict[str, list[float]] = {}
    for name, agg in per_file.items():
        for s, v in agg.items():
            buckets.setdefault(s, []).append(v)
    sequences = list(buckets)
    activities = np.asarray([float(np.mean(v)) for v in buckets.values()], dtype=np.float64)
    return sequences, activities, {n: len(a) for n, a in per_file.items()}


def load_polya():
    from core.route2_v8_joint_library_v1 import POLYA_LIB_GZ, load_polya_library
    return load_polya_library(POLYA_LIB_GZ)


def load_mrl_280k():
    from core.route2_v8_joint_library_v1 import MRL_LIB_DIR, load_mrl_library
    return load_mrl_library(MRL_LIB_DIR)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--physical-gpu-index", required=True, type=int)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--init-checkpoint", default=str(INIT_CKPT))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - GPU required")
    device = torch.device(f"cuda:{args.physical_gpu_index}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir) if args.out_dir else OUT_ROOT / "s_mrl-m1-polya"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out={out_dir}", flush=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MRNABERT_PATH, local_files_only=True)
    verify_vocab_alignment(tokenizer)

    protected_index = build_protected_index()

    mrl_seqs, mrl_act = load_mrl_280k()
    m1_seqs, m1_act, m1_file_stats = load_m1_library()
    polya_seqs, polya_act = load_polya()

    raw = {
        "mrl": (mrl_seqs, mrl_act),
        "m1": (m1_seqs, m1_act),
        "polya": (polya_seqs, polya_act),
    }
    # m1 reuses the mrl readout domain id (0): same 5'UTR translation readout family.
    libraries: dict[str, DomainLibrary] = {}
    import core.route2_v8_joint_library_v1 as jointlib
    _orig_domain_ids = dict(jointlib.DOMAIN_IDS)
    jointlib.DOMAIN_IDS = dict(_orig_domain_ids)
    jointlib.DOMAIN_IDS["m1"] = 0
    for domain, (sequences, activities) in raw.items():
        flags = audit_leak_flags(sequences, protected_index)
        lib = prepare_domain_library(domain, sequences, activities, flags, tokenizer)
        lib.domain_id = 0 if domain == "m1" else _orig_domain_ids[domain]
        libraries[domain] = lib
        print(f"[{domain}] {json.dumps(lib.audit_summary())}", flush=True)
    jointlib.DOMAIN_IDS = _orig_domain_ids

    audit_report = {
        "schema_version": "route_a_v3_route2_v8_m1_prefinetune.v1",
        "protected_studies": sorted(protected_index),
        "domains": {d: libraries[d].audit_summary() for d in raw},
        "m1_file_stats": m1_file_stats,
        "m1_domain_mapping": "m1 -> domain_id 0 (mrl readout family)",
        "init": str(args.init_checkpoint),
    }
    (out_dir / "leakage_audit.json").write_text(json.dumps(audit_report, indent=1, sort_keys=True))
    if args.skip_train:
        print("skip-train: audit complete, exiting")
        return 0

    model = build_v8_regressor(MRNABERT_PATH, "s", num_domains=3).to(device)
    init = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(init["model_state_dict"], strict=False)
    print(f"init from {args.init_checkpoint}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    params = parameter_report(model)
    print(f"parameter report: {json.dumps(params)}", flush=True)

    active = list(raw)
    sampler = DomainBalancedSampler(
        domain_sizes={d: libraries[d].n_clean for d in active},
        batch_size=args.batch,
        seed=args.seed,
    )
    planned = sampler.steps_per_epoch * args.epochs
    total_steps = args.max_steps if args.max_steps is not None else planned
    print(f"budget: steps/epoch={sampler.steps_per_epoch} planned={planned} total={total_steps}", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(step / max(total_steps, 1), 1.0))) if step > total_steps * 0.05 else step / max(total_steps * 0.05, 1),
    )

    loss_log = (out_dir / "training_losses.jsonl").open("w")
    epoch_domain_loss = (out_dir / "epoch_domain_loss.jsonl").open("w")
    epoch_zeroshot = (out_dir / "epoch_zeroshot_metrics.jsonl").open("w")

    def save_checkpoint(name, epoch_done, steps_done, note):
        torch.save(
            {
                "schema_version": "route_a_v3_route2_v8_m1_prefinetune.v1",
                "model_state_dict": model.state_dict(),
                "arch": "s",
                "libraries": active,
                "domain_stats": {
                    d: {"mean": libraries[d].target_mean, "std": libraries[d].target_std,
                        "n_clean": libraries[d].n_clean, "domain_id": libraries[d].domain_id}
                    for d in active
                },
                "seed": args.seed,
                "epochs_done": epoch_done,
                "steps_done": steps_done,
                "init_from": str(args.init_checkpoint),
                "note": note,
            },
            out_dir / name,
        )

    def run_zeroshot(epoch_done, steps_done, primary):
        if args.skip_eval:
            return []
        results = []
        for domain in ("mrl", "polya"):  # benchmark-backed domains only
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
    import time
    t0 = time.time()
    try:
        for epoch in range(args.epochs):
            domain_losses: dict[str, list[float]] = {d: [] for d in active}
            for batch_idx, batch in enumerate(sampler.epoch_batches(epoch)):
                input_ids, attention_mask, targets, domain_ids_t, _spans = assemble_batch(libraries, batch, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(input_ids, attention_mask, domain_ids_t)
                    loss = torch.nn.functional.mse_loss(out.float(), targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                step += 1
                dom_list = domain_ids_t.detach().cpu().tolist()
                for di, l in zip(dom_list, [float(loss.item())] * len(dom_list)):
                    for d in active:
                        if libraries[d].domain_id == di:
                            domain_losses[d].append(l)
                if step % 500 == 0:
                    loss_log.write(json.dumps({"step": step, "loss": float(loss.item()), "lr": scheduler.get_last_lr()[0]}) + "\n")
                    loss_log.flush()
                if step % 5000 == 0:
                    print(f"step {step}/{total_steps} loss {float(loss.item()):.4f} elapsed {time.time()-t0:.0f}s", flush=True)
                if args.max_steps is not None and step >= args.max_steps:
                    break
            epoch_completed = epoch + 1
            rec = {"epoch": epoch_completed, "steps": step, "mean_loss": {d: (float(np.mean(v)) if v else None) for d, v in domain_losses.items()}}
            epoch_domain_loss.write(json.dumps(rec) + "\n")
            epoch_domain_loss.flush()
            save_checkpoint(f"v8_m1_epoch{epoch_completed}.pt", epoch_completed, step, f"epoch {epoch_completed} done")
            run_zeroshot(epoch_completed, step, primary=(epoch_completed == args.epochs))
            if args.max_steps is not None and step >= args.max_steps:
                break
    finally:
        loss_log.close()
        epoch_domain_loss.close()
        epoch_zeroshot.close()

    final = [json.loads(l) for l in (out_dir / "epoch_zeroshot_metrics.jsonl").open()] if not args.skip_eval else []
    report = {
        "schema_version": "route_a_v3_route2_v8_m1_prefinetune.v1",
        "out_dir": str(out_dir),
        "init_from": str(args.init_checkpoint),
        "epochs": args.epochs,
        "steps": step,
        "domain_sizes": {d: libraries[d].n_clean for d in active},
        "m1_file_stats": m1_file_stats,
        "final_zeroshot": final,
        "prereg_gate": {"mrl_spearman": {"baseline_s_winner": 0.3078, "threshold": 0.3078 - 0.02},
                        "polya_spearman": {"baseline": 0.5262, "threshold": 0.5262 - 0.02}},
    }
    (out_dir / "run_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"status": "DONE", "out": str(out_dir)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
