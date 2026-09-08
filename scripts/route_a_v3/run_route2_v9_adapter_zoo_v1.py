#!/usr/bin/env python3
"""V9-1a adapter-zoo training (SPECS_CRITIC_V6 spec 2026-09-08 增补 N.4, Task 12).

Preregistration: docs/paper/route2_v9_adapter_zoo_prereg_v1.md (FROZEN 2026-09-08).
Architecture (fixed clauses): frozen V8-S Stage 1 trunk + per-task LoRA (r16 a32)
+ shared LoRA (r32) on all 12 layers (Wqkv/attn.dense/gated_layers/wo) + per-task
linear heads + polyA CNN stem (init from Stage 1 H, trainable) + deterministic
task_id routing (domain_ids index; NO learned router). Gradient separation:
task LoRA/head see only their task (PLE-style dedicated capacity); shared LoRA
and domain/cell embeddings see all tasks.

Data/loss/eval: identical to the V8 Stage 2 benchmark arm (pair-delta MSE on
z-scored per-study targets, DomainBalancedSampler, cell conditioning with
original per-cell MPRAU labels, frozen evaluator). FINAL-EPOCH-6-FIXED.

Discipline: CUDA BF16 only (cpu_fallback_used=false); protected reads = 0;
products /mnt, code /home worktree + push.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_RUNNER = REPO_ROOT / "scripts/route_a_v3/run_route2_v8_stage2_adapt_v1.py"
_spec = importlib.util.spec_from_file_location("r2", _RUNNER)
r2 = importlib.util.module_from_spec(_spec)
sys.modules["r2"] = r2
_spec.loader.exec_module(r2)

from core.route2_v8_hybrid_backbone_v1 import (  # noqa: E402
    CELL_IDS, DOMAIN_IDS, load_mrnabert_base,
)
from core.route2_v8_joint_library_v1 import DomainBalancedSampler  # noqa: E402

MNT = r2.MNT
STAGE1_S = MNT / "experiments/xeditcritic_route_a/v8_stage1_joint_prefinetune_20260904/s_mrl-polya/stage1_s_epoch2.pt"
STAGE1_H = MNT / "experiments/xeditcritic_route_a/v8_stage1_joint_prefinetune_20260904/h_mrl-polya/stage1_h_epoch2.pt"
OUT_ROOT = MNT / "experiments/xeditcritic_route_a/v9_adapter_zoo_20260908"

EPOCHS = 6
LR = 2e-5
WEIGHT_DECAY = 1e-4
BATCH = 128
TASK_RANK, TASK_ALPHA = 16, 32.0
SHARED_RANK, SHARED_ALPHA = 32, 32.0
LORA_DROPOUT = 0.05
POLYA_DOMAIN_ID = DOMAIN_IDS["polya"]
NUM_DOMAINS = len(DOMAIN_IDS)


class MultiTaskLoRALinear(nn.Module):
    """Frozen Linear + 1 shared LoRA (r32) + n_tasks task LoRAs (r16); holder-routed."""

    def __init__(self, base: nn.Linear, n_tasks: int, holder: dict,
                 task_rank: int = TASK_RANK, shared_rank: int = SHARED_RANK,
                 alpha: float = TASK_ALPHA, dropout: float = LORA_DROPOUT):
        super().__init__()
        assert type(base) is nn.Linear
        self.base = base
        self.base.requires_grad_(False)
        self.holder = holder  # {"task": int} shared mutable routing context
        self.n_tasks = n_tasks
        self.task_rank, self.shared_rank = task_rank, shared_rank
        self.dropout = dropout
        # shared set (rank 32)
        self.shared_a = nn.Parameter(torch.empty(shared_rank, base.in_features))
        self.shared_b = nn.Parameter(torch.zeros(base.out_features, shared_rank))
        nn.init.kaiming_uniform_(self.shared_a, a=math.sqrt(5))
        # task sets (rank 16), index t-1 -> domain t-1
        self.lora_a = nn.Parameter(torch.empty(n_tasks, task_rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(n_tasks, base.out_features, task_rank))
        for s in range(n_tasks):
            nn.init.kaiming_uniform_(self.lora_a[s], a=math.sqrt(5))
        self.task_scaling = alpha / task_rank
        self.shared_scaling = SHARED_ALPHA / shared_rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        xr = F.dropout(x, p=self.dropout, training=self.training)
        residual = F.linear(F.linear(xr, self.shared_a), self.shared_b) * self.shared_scaling
        task = int(self.holder["task"])
        if 1 <= task <= self.n_tasks:
            a_t, b_t = self.lora_a[task - 1], self.lora_b[task - 1]
            residual = residual + F.linear(F.linear(xr, a_t), b_t) * self.task_scaling
        return out + residual.to(out.dtype)


class V9AdapterZooRegressor(nn.Module):
    """Frozen V8-S trunk + multi-task LoRA + per-task heads + polyA CNN stem."""

    def __init__(self, base_model: nn.Module, n_domains: int = NUM_DOMAINS, num_cells: int = 0):
        super().__init__()
        self.base = base_model
        self.holder = {"task": 0}
        self.stem = nn.Module()  # placeholder; replaced by real CNNMotifStem when polyA stem enabled
        self.use_stem = False
        self.domain_embeddings = nn.Embedding(n_domains, base_model.config.hidden_size)
        nn.init.normal_(self.domain_embeddings.weight, mean=0.0, std=0.02)
        self.num_cells = int(num_cells)
        if self.num_cells > 0:
            self.cell_embeddings = nn.Embedding(self.num_cells, base_model.config.hidden_size)
            nn.init.normal_(self.cell_embeddings.weight, mean=0.0, std=0.02)
        self.heads = nn.ModuleList([nn.Linear(base_model.config.hidden_size, 1) for _ in range(n_domains)])

    def install_stem(self, stem: nn.Module) -> None:
        self.stem = stem
        self.use_stem = True

    def wrap_multitask_lora(self) -> int:
        wrapped = 0
        for layer in self.base.encoder.layer:
            for attr_path in (("attention", "self", "Wqkv"),
                              ("attention", "output", "dense"),
                              ("mlp", "gated_layers"),
                              ("mlp", "wo")):
                parent = layer
                for step in attr_path[:-1]:
                    parent = getattr(parent, step)
                target = getattr(parent, attr_path[-1])
                if isinstance(target, nn.Linear):
                    setattr(parent, attr_path[-1],
                            MultiTaskLoRALinear(target, NUM_DOMAINS, holder=self.holder))
                    wrapped += 1
        if wrapped == 0:
            raise SystemExit("multi-task LoRA wrap found no target Linear modules")
        return wrapped

    def _sequence_output(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                         use_stem: bool) -> torch.Tensor:
        if not use_stem:
            return self.base(input_ids=input_ids, attention_mask=attention_mask)[0]
        word_embeddings = self.base.embeddings.word_embeddings(input_ids)
        stem_features = self.stem(input_ids).to(word_embeddings.dtype)
        embedding_output = self.base.embeddings(
            inputs_embeds=word_embeddings + stem_features,
            token_type_ids=torch.zeros_like(input_ids),
        )
        encoder_outputs = self.base.encoder(embedding_output, attention_mask, output_all_encoded_layers=False)
        return encoder_outputs[-1]

    def forward(self, input_ids, attention_mask, domain_ids, cell_ids=None):
        tasks = domain_ids.tolist()
        assert len(set(tasks)) == 1, "V9 adapter-zoo expects task-homogeneous batches"
        task = int(tasks[0])
        self.holder["task"] = task
        use_stem = self.use_stem and task == POLYA_DOMAIN_ID
        hidden = self._sequence_output(input_ids, attention_mask, use_stem)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = pooled + self.domain_embeddings(domain_ids)
        if cell_ids is not None:
            if self.num_cells <= 0:
                raise ValueError("cell_ids passed but cell conditioning disabled")
            pooled = pooled + self.cell_embeddings(cell_ids)
        return self.heads[task](pooled).squeeze(-1)


def build_v9(mrnabert_path: Path, num_cells: int) -> V9AdapterZooRegressor:
    base = load_mrnabert_base(mrnabert_path)
    model = V9AdapterZooRegressor(base, n_domains=NUM_DOMAINS, num_cells=num_cells)
    return model


def load_v9_init(model: V9AdapterZooRegressor) -> dict:
    raw_s = torch.load(STAGE1_S, map_location="cpu", weights_only=False)
    state_s = raw_s["model_state_dict"]
    model_state = model.state_dict()
    compatible = {}
    for key, value in state_s.items():
        target = "base." + key if ("base." + key) in model_state else key
        if target in model_state and model_state[target].shape == value.shape:
            compatible[target] = value
    # domain embeddings: stage-1 rows 0..2 preserved
    dom = state_s.get("domain_embeddings.weight")
    if dom is not None:
        n = min(dom.shape[0], model.domain_embeddings.weight.shape[0])
        model.domain_embeddings.weight.data[:n] = dom[:n]
    # cell embeddings from stage-2 geometry absent in stage 1 -> keep fresh init
    missing = [k for k in model_state if k not in compatible and not k.startswith(("lora_", "stem.", "heads."))]
    model.load_state_dict(compatible, strict=False)
    # polyA CNN stem from Stage 1 H
    stem_loaded = 0
    raw_h = torch.load(STAGE1_H, map_location="cpu", weights_only=False)
    from core.route2_v8_hybrid_backbone_v1 import CNNMotifStem
    stem = CNNMotifStem(output_dim=model.base.config.hidden_size)
    stem_state = {k.removeprefix("stem."): v for k, v in raw_h["model_state_dict"].items() if k.startswith("stem.")}
    if stem_state:
        stem.load_state_dict(stem_state, strict=True)
        model.install_stem(stem)
        stem_loaded = len(stem_state)
    return {"stage1_s_keys": len(compatible), "stage1_s_missing_after": len(missing),
            "stem_keys_loaded": stem_loaded, "init_s": str(STAGE1_S), "init_h": str(STAGE1_H)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="smoke cap")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - GPU required")
    device = torch.device(f"cuda:{args.physical_gpu_index}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(r2.MRNABERT_PATH, local_files_only=True)
    r2.verify_vocab_alignment(tokenizer)

    out_dir = Path(args.out_dir) if args.out_dir else OUT_ROOT / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    r2._OUT_DIR = str(out_dir)

    model = build_v9(r2.MRNABERT_PATH, num_cells=r2.NUM_CELLS).to(device)
    init_summary = load_v9_init(model)
    # freeze the trunk BEFORE wrapping: MultiTaskLoRALinear registers its LoRA
    # parameters under base.* -- wrapping after the freeze keeps them trainable
    for p in model.base.parameters():
        p.requires_grad_(False)
    wrapped = model.wrap_multitask_lora()
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    # sanity: trainable must be LoRA + stem + embeddings + heads only
    bad = [n for n, p in model.named_parameters()
           if p.requires_grad and not (n.endswith((".lora_a", ".lora_b", ".shared_a", ".shared_b"))
                                       or n.startswith(("stem.", "domain_embeddings.", "cell_embeddings.", "heads.")))]
    if bad:
        raise SystemExit(f"unexpected trainable params after freeze/wrap: {bad[:5]}")
    params = {
        "trainable": sum(p.numel() for p in trainable),
        "lora_modules_wrapped": wrapped,
        "task_rank": TASK_RANK, "shared_rank": SHARED_RANK,
        "n_heads": len(model.heads), "stem": model.use_stem,
    }
    print(f"init: {json.dumps(init_summary)}", flush=True)
    print(f"params: {json.dumps(params)}", flush=True)

    libraries = r2.load_benchmark_domains(tokenizer)
    domain_sizes = {study: int(lib.targets.shape[0]) for study, (lib, _c) in libraries.items()}
    sampler = DomainBalancedSampler(domain_sizes=domain_sizes, batch_size=args.batch, seed=args.seed)
    planned = sampler.steps_per_epoch * args.epochs
    total_steps = args.max_steps if args.max_steps is not None else planned
    budget = {"epochs": args.epochs, "batch": args.batch,
              "steps_per_epoch": sampler.steps_per_epoch, "planned_total_steps": planned,
              "effective_total_steps": total_steps, "domain_sizes": domain_sizes,
              "seed": args.seed, "selection_rule": "FINAL_EPOCH_FIXED"}
    print(f"budget: {json.dumps(budget)}", flush=True)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(step / max(total_steps, 1), 1.0)))
        if step > total_steps * 0.05 else step / max(total_steps * 0.05, 1),
    )

    loss_log = (out_dir / "training_losses.jsonl").open("w")
    epoch_eval = (out_dir / "epoch_eval_metrics.jsonl").open("w")

    def save_checkpoint(name: str, epoch_done: int, steps_done: int, note: str) -> None:
        torch.save({
            "schema_version": "route_a_v3_route2_v9_adapter_zoo.v1",
            "model_state_dict": {k: v for k, v in model.state_dict().items()},
            "seed": args.seed, "epochs_done": epoch_done, "steps_done": steps_done,
            "note": note, "init": init_summary,
        }, out_dir / name)

    def run_eval(epoch_done: int, steps_done: int, primary: bool) -> dict:
        if args.skip_eval:
            return {}
        model.eval()
        results = []
        for study, (domain, _rel) in r2.BENCHMARK_DOMAINS.items():
            if study == "ENCSR854RUF":
                continue
            rec = r2.eval_study_frozen_delta(model, tokenizer, device, study, domain)
            results.append(rec)
            print(f"== eval {study}: spearman {rec.get('task_macro_spearman')} top1 {rec.get('top_1')}", flush=True)
        mprau = r2.eval_mprau_validation(model, tokenizer, device, r2.DOMAIN_IDS["mprau"])
        results.append({"domain": "mprau", **mprau})
        macro = r2._task_macro(results)
        report = {"epoch": epoch_done, "steps": steps_done, "primary": primary,
                  "task_macro_spearman": macro, "per_task": results, "mprau": mprau}
        epoch_eval.write(json.dumps(report) + "\n")
        epoch_eval.flush()
        model.train()
        print(f"== epoch {epoch_done} eval: macro {macro:.4f}", flush=True)
        return report

    model.train()
    step = 0
    stopped_early = False
    final_eval: dict = {}
    for epoch in range(args.epochs):
        for batch in sampler.epoch_batches(epoch):
            loss = r2.train_benchmark_batch(model, libraries, batch, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 50 == 0:
                loss_log.write(json.dumps({"step": step, "epoch": epoch + 1,
                                            "mse_all": float(loss),
                                            "lr": scheduler.get_last_lr()[0]}) + "\n")
                loss_log.flush()
            if args.max_steps is not None and step >= args.max_steps:
                stopped_early = True
                break
        if stopped_early:
            save_checkpoint(f"v9_adapter_zoo_smoke{step}.pt", epoch + 1, step, "smoke non-terminal")
            final_eval = run_eval(epoch + 1, step, primary=False)
            break
        save_checkpoint(f"v9_adapter_zoo_epoch{epoch + 1}.pt", epoch + 1, step,
                        "final" if epoch + 1 == args.epochs else "epoch")
        final_eval = run_eval(epoch + 1, step, primary=(epoch + 1 == args.epochs))

    loss_log.close()
    epoch_eval.close()
    report = {
        "schema_version": "route_a_v3_route2_v9_adapter_zoo.v1",
        "seed": args.seed, "selection_rule": "FINAL_EPOCH_FIXED" if not stopped_early else "SMOKE",
        "init": init_summary, "params": params, "budget": budget,
        "steps_done": step, "smoke": bool(args.max_steps), "cpu_fallback_used": False,
        "eval_final": final_eval,
    }
    (out_dir / "run_report.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    print(f"wrote {out_dir / 'run_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
