#!/usr/bin/env python3
"""Option-1 exploration arm: token-dropout regularisation for the off-manifold
hypothesis (SPECS_CRITIC_V6 Task 14 fallback ladder; user adjudicated
"single-arm first" 2026-09-10 batch 74).

Target (REVISED per C2 full-891 outcome, batch 73): can input-space
regularisation stabilise/improve RANKING quality at the 891-source scale --
NOT to reproduce the calib100 peak (that pool's positive was rejected at full
scale by the two-tier prereg design).

Design: identical recipe to run_route2_v9b_data_arm_v1.py bench-v9b arm
(frozen mRNABERT trunk + multi-task LoRA + 11-domain heads + cell embeddings +
DomainBalancedSampler + pair-delta MSE, FINAL-EPOCH-6-FIXED), with ONE change:

  --token-dropout p: during TRAINING forward, each non-special token's
  encoder-output vector is zeroed independently with probability p (a
  multiplicative Bernoulli mask, applied AFTER the encoder, BEFORE pooling).
  Special tokens (CLS/SEP/PAD positions per attention_mask & token ids) are
  never dropped. At eval time no dropout (standard).

  Mechanism: forces the pooled representation to not collapse onto a few
  token positions -- the critic must spread its evidence across the sequence,
  which is exactly the robustness the mixed-pool probe measures (candidates
  differ from source at 1-3 positions only; a critic concentrated on dropped
  positions would produce near-constant potentials).

Gate (preregistered BEFORE launch; commit precedes any training):
  - primary: 891-source mixed-pool A3 probe per-source re-run with the
    token-dropout checkpoint (persrc pipeline from the committee arm, batch 72)
    -- composite constructive-routing probe value vs the no-dropout v9b bench
    reference, paired per-source bootstrap 2000 iters seed 20260816;
  - decision: CI excludes zero in the positive direction => dropout arm wins;
    crossing zero => honest null; negative => dropout hurts.

Reference checkpoint (no-dropout control): v9b_data_arm_20260909/
bench-v9b_seed20260907/v9b_epoch6.pt (batch 57, frozen reference).
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

_spec = importlib.util.spec_from_file_location(
    "r2", V8_WT / "scripts/route_a_v3/run_route2_v8_stage2_adapt_v1.py"
)
r2 = importlib.util.module_from_spec(_spec)
sys.modules["r2"] = r2
_spec.loader.exec_module(r2)

_spec9 = importlib.util.spec_from_file_location(
    "v9run", V8_WT / "scripts/route_a_v3/run_route2_v9_adapter_zoo_v1.py"
)
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
OUT_ROOT = MNT / "experiments/xeditcritic_route_a/v9c_tokendrop_20260910"

DOMAIN_STABILITY = 9
DOMAIN_NDD = 10
N_DOMAINS = 11
CELL_SH, CELL_HEK = 6, 7
NUM_CELLS = 8
EPOCHS, LR, WD, BATCH = 6, 2e-5, 1e-4, 128


def load_v9b_libraries() -> dict[str, tuple[list[dict], torch.Tensor, torch.Tensor]]:
    spec = importlib.util.spec_from_file_location(
        "v9bmod",
        V8_WT / "scripts/route_a_v3/run_route2_v9b_data_arm_v1.py",
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["v9bmod"] = m
    spec.loader.exec_module(m)
    return m.load_v9b_libraries()


class TokenDropoutHook:
    """Zeroes encoder-output token vectors with prob p during training.

    Applied as a forward hook on the frozen base model output. The caller MUST
    stash the input_ids before each model() call (HF BertModel is invoked with
    kwargs only, so positional-hook args are empty -- the smoke run with
    calls=0 caught this). Special positions (CLS 2 / SEP 3 / PAD 0) are
    preserved. Eval is protected by both the enabled flag and grad-enabled
    check.

    Output handling: HF returns a ModelOutput; we replace its
    last_hidden_state out-of-place (autograd-safe) and return the modified
    object (a non-None hook return replaces the module output). Tuple outputs
    (other call paths) are handled by rebuilding the tuple.
    """

    def __init__(self, p: float):
        self.p = float(p)
        self.enabled = False
        self.calls = 0
        self.stashed_ids: torch.Tensor | None = None

    def stash(self, input_ids: torch.Tensor) -> None:
        self.stashed_ids = input_ids

    def __call__(self, module, args, output):
        if not self.enabled or not torch.is_grad_enabled():
            return output
        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            hidden = output.last_hidden_state
        elif isinstance(output, (tuple, list)) and torch.is_tensor(output[0]):
            hidden = output[0]
        else:
            return output
        input_ids = self.stashed_ids
        if (
            input_ids is None
            or not torch.is_tensor(input_ids)
            or input_ids.shape != hidden.shape[:2]
        ):
            return output
        special = (input_ids == 0) | (input_ids == 2) | (input_ids == 3)
        keep = (
            torch.rand_like(hidden, dtype=torch.float32) >= self.p
        ).to(hidden.dtype)
        keep = keep.masked_fill(special.unsqueeze(-1).to(hidden.device), 1.0)
        masked = hidden * keep
        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            output.last_hidden_state = masked
            self.calls += 1
            return output
        out_list = list(output)
        out_list[0] = masked
        self.calls += 1
        return tuple(out_list) if isinstance(output, tuple) else out_list


def train_batch(model, lib, idx, device, hook):
    src_ids = lib["input_ids"][idx * 2]
    cnd_ids = lib["input_ids"][idx * 2 + 1]
    src_mask = lib["attention_mask"][idx * 2]
    cnd_mask = lib["attention_mask"][idx * 2 + 1]
    targets = lib["targets"][idx].to(device)
    dom = torch.full((idx.shape[0],), lib["domain_id"], dtype=torch.long, device=device)
    cells = lib["cell_ids"][idx].to(device) if lib.get("cell_ids") is not None else None
    hook.enabled = True
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        hook.stash(src_ids.to(device))
        sp = model(src_ids.to(device), src_mask.to(device), dom, cells)
        hook.stash(cnd_ids.to(device))
        cp = model(cnd_ids.to(device), cnd_mask.to(device), dom, cells)
        loss = torch.nn.functional.mse_loss((cp - sp).float(), targets)
    hook.enabled = False
    hook.stashed_ids = None
    return loss


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--token-dropout", required=True, type=float)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()
    assert 0.0 < args.token_dropout < 0.5, "token-dropout p in (0, 0.5)"

    device = torch.device(f"cuda:{args.physical_gpu_index}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(r2.MRNABERT_PATH, local_files_only=True)
    r2.verify_vocab_alignment(tok)

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else OUT_ROOT / f"bench-v9c_drop{args.token_dropout}_seed{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    r2._OUT_DIR = str(out_dir)

    model = v9run.build_v9(r2.MRNABERT_PATH, num_cells=NUM_CELLS)
    import torch.nn as nn

    model.domain_embeddings = nn.Embedding(N_DOMAINS, model.base.config.hidden_size)
    nn.init.normal_(model.domain_embeddings.weight, mean=0.0, std=0.02)
    model.cell_embeddings = nn.Embedding(NUM_CELLS, model.base.config.hidden_size)
    nn.init.normal_(model.cell_embeddings.weight, mean=0.0, std=0.02)
    model.heads = nn.ModuleList(
        [nn.Linear(model.base.config.hidden_size, 1) for _ in range(N_DOMAINS)]
    )
    init = v9run.load_v9_init(model)
    for p in model.base.parameters():
        p.requires_grad_(False)
    v9run.NUM_DOMAINS = N_DOMAINS
    wrapped = model.wrap_multitask_lora()
    model.to(device)

    hook = TokenDropoutHook(args.token_dropout)
    model.base.register_forward_hook(hook)

    trainable = [p for p in model.parameters() if p.requires_grad]

    v9b = load_v9b_libraries()
    bench = r2.load_benchmark_domains(tok)
    domain_sizes = {}
    libs = {}
    for name, lib in v9b.items():
        domain_sizes[name] = lib["n"]
        libs[name] = lib
    for study, (lib, _c) in bench.items():
        domain_sizes[study] = int(lib.targets.shape[0])
        libs[study] = {
            "input_ids": lib.input_ids,
            "attention_mask": lib.attention_mask,
            "targets": lib.targets,
            "cell_ids": _c,
            "domain_id": lib.domain_id,
            "n": int(lib.targets.shape[0]),
            "bench": True,
        }
    sampler = DomainBalancedSampler(
        domain_sizes=domain_sizes, batch_size=BATCH, seed=args.seed
    )
    planned = sampler.steps_per_epoch * args.epochs
    total_steps = args.max_steps if args.max_steps is not None else planned
    budget = {
        "arm": "bench-v9c-token-dropout",
        "token_dropout_p": args.token_dropout,
        "seed": args.seed,
        "epochs": args.epochs,
        "domain_sizes": domain_sizes,
        "steps_per_epoch": sampler.steps_per_epoch,
        "planned_total_steps": planned,
        "effective_total_steps": total_steps,
        "selection_rule": "FINAL_EPOCH_FIXED",
        "geometry": {"n_domains": N_DOMAINS, "num_cells": NUM_CELLS},
        "control_reference": (
            "v9b_data_arm_20260909/bench-v9b_seed20260907/v9b_epoch6.pt "
            "(no-dropout, identical recipe)"
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
                "schema_version": "route_a_v3_route2_v9c_token_dropout.v1",
                "model_state_dict": {k: v for k, v in model.state_dict().items()},
                "arm": "bench-v9c-token-dropout",
                "token_dropout_p": args.token_dropout,
                "seed": args.seed,
                "epochs_done": ep,
                "steps_done": st,
                "note": note,
                "init": init,
                "budget": budget,
            },
            out_dir / name,
        )

    def run_eval(ep, st, primary):
        if args.skip_eval:
            return {}
        model.eval()
        report = {"epoch": ep, "steps": st, "primary": primary}
        hold = {}
        for name, path, col, dom, cell in (
            ("s1_sh", S1_LIB, "y_sh", DOMAIN_STABILITY, CELL_SH),
            ("s1_hek", S1_LIB, "y_hek", DOMAIN_STABILITY, CELL_HEK),
            ("m6", M6_LIB, "y_totalcount_log2fc", DOMAIN_NDD, CELL_HEK),
        ):
            val_ids = set(
                json.load(
                    open(S1_HOLDOUT if name.startswith("s1") else M6_HOLDOUT)
                )["validation_split_ids"]
            )
            rows = [json.loads(l) for l in path.open()]
            rows = [r for r in rows if r["variant_id"] in val_ids and r.get(col) is not None]
            if not rows:
                continue
            from scipy.stats import spearmanr

            src = r2.score_sequences(
                model, tok, device, [r["source_sequence"] for r in rows], dom,
                torch.tensor([cell] * len(rows)),
            )
            cnd = r2.score_sequences(
                model, tok, device, [r["candidate_sequence"] for r in rows], dom,
                torch.tensor([cell] * len(rows)),
            )
            pred = np.array(cnd) - np.array(src)
            tgt = np.array([float(r[col]) for r in rows])
            hold[name] = {
                "n": len(rows),
                "spearman_holdout": float(spearmanr(tgt, pred).statistic),
            }
        report["v9b_holdout"] = hold
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
        print(
            f"== epoch {ep} eval: "
            + json.dumps({k: v for k, v in report.items() if k != "per_task"})[:400],
            flush=True,
        )
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
                loss = train_batch(model, lib, idx, device, hook)
                total = loss if total is None else total + loss
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 50 == 0:
                loss_log.write(
                    json.dumps(
                        {
                            "step": step,
                            "epoch": epoch + 1,
                            "mse_all": float(total),
                            "lr": scheduler.get_last_lr()[0],
                            "dropout_hook_calls": hook.calls,
                        }
                    )
                    + "\n"
                )
                loss_log.flush()
            if args.max_steps is not None and step >= args.max_steps:
                stopped = True
                break
        if stopped:
            save_ckpt(f"v9c_smoke{step}.pt", epoch + 1, step, "smoke")
            final_eval = run_eval(epoch + 1, step, False)
            break
        save_ckpt(
            f"v9c_epoch{epoch + 1}.pt",
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
                "schema_version": "route_a_v3_route2_v9c_token_dropout.v1",
                "arm": "bench-v9c-token-dropout",
                "token_dropout_p": args.token_dropout,
                "seed": args.seed,
                "selection_rule": "FINAL_EPOCH_FIXED" if not stopped else "SMOKE",
                "init": init,
                "budget": budget,
                "steps_done": step,
                "dropout_hook_calls": hook.calls,
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
