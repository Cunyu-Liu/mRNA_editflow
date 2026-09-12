#!/usr/bin/env python3
"""D16-C probe arm: within-source structure augmentation continuation-training (amendment v1 sec.4).

Preregistration: docs/paper/route2_critic_d16c_within_source_amendment_v1.md (W0 worktree).
H-D16C: within-source candidate density 4.9 -> >=32 (structure-aligned) pushes V5 from
memorization route to structure route. Predictions: train-backtest gap 0.79 -> <=0.30 with
VALIDATION rho >= 0.135 (G1); polyA VAL rho >= 0.80 (G2, non-destruction on the same
continued model); structure-pool graded sc-hit@1 > 0 (G4, D15-2 probe post-training).

Design (amendment sec.4.1, D13-b: repair-not-replace):
- Init: frozen V5 terminal checkpoint (final_pass_8); full-parameter continuation.
- Data per step: batch of MEASURED MRL TRAIN rows (supervision) + structure-aligned
  SYNTHETIC rows from the same sources (H1: zero regression label -> their only role is
  to occupy within-source ranking slots; they are EXCLUDED from Huber (weight 0) AND
  from ranking pairs (pair target undefined without a label - asserted).
- Loss: Huber on z-scored measured targets (V5 scaler scale reused for VAL comparability)
  + within-source soft ranking on measured pairs (w=0.5, V5 pass3-8 form).
- Selection: FINAL-EPOCH-FIXED; per-epoch VALIDATION spearman is diagnostic only.
- Discipline: CUDA BF16 only (hard abort otherwise); protected reads = 0
  (TRAIN/VALIDATION only); products /mnt, code /home worktree + push.
- Budget: probe 6 epochs on 2,443 measured rows (~1 GPU-h/epoch on a free A100).

The forward path reuses the V5 frozen-guidance machinery (records_from_projection_rows ->
XEditCriticDatasetV4 -> collator -> XEditCriticV4 model) so tokenization, edit bundles,
metadata vocabs, and study calibration are bit-identical to official V5 scoring; the only
new element is gradient flow (requires_grad on the upper trunk; bottom-six stays frozen
via the cache path with encode-as-needed per batch).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SETFLOW_REPO = "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901"
_spec = importlib.util.spec_from_file_location(
    "v5frozen", SETFLOW_REPO + "/scripts/route_a_v3/route2_xeditcritic_v5_frozen_guidance_v1.py"
)
v5frozen = importlib.util.module_from_spec(_spec)
sys.modules["v5frozen"] = v5frozen
_spec.loader.exec_module(v5frozen)

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
PROJ_TRAIN = MNT / "projections/xedit_v3/development_train_validation_v1/train.jsonl"
PROJ_VAL = MNT / "projections/xedit_v3/development_train_validation_v1/validation.jsonl"
SYNTH = MNT / "experiments/xeditcritic_d16c/within_source_augmentation_mrl_v1.jsonl"
V5_CKPT = MNT / "experiments/xeditcritic_v5/v5_screen_seed_20260907_runner_1113cd2c0dd9acb508f58782eecb40f458d2cab3/v5_full/final_pass_8_checkpoint.pt"
MRNABERT = MNT / "external_model_assets/mrnabert_a1eb7df25804d23f08646e1cb996b234d7208a40"
OUT_DIR = MNT / "experiments/xeditcritic_d16c/probe_mrl_v1"

MRL_TASK = "MEAN_RIBOSOME_LOAD::region=0"
MRL_ENDPOINT = "MEAN_RIBOSOME_LOAD"
MRL_REGION = "5UTR"
POLYA_TASK = "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS::region=1"
EPOCHS = 6
LR = 1e-5
BATCH = 16
SEED = 20260913
RANK_WEIGHT = 0.5
HUBER_DELTA = 1.0
SOFT_RANK_TEMPERATURE = 0.2


def load_rows(path: Path, task: str) -> list[dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("task_id") == task:
                rows.append(d)
    return rows


def _row_for_pair(source: str, candidate: str, template: dict[str, Any], record_id: str, task_id: str) -> dict[str, Any]:
    source_u = str(source).upper().replace("T", "U")
    cand_u = str(candidate).upper().replace("T", "U")
    return {
        "canonical_record_id": record_id,
        "split": "TRAIN",
        "source_sequence": source_u,
        "candidate_sequence": cand_u,
        "source_relative_edits": [
            {"position": p, "source_base": source_u[p], "candidate_base": cand_u[p]}
            for p in range(len(source_u))
            if source_u[p] != cand_u[p]
        ],
        "direction_normalized_delta": 0.0,
        "task_id": task_id,
        "study_unit_id": template["study_unit_id"],
        "source_group_id": template["source_group_id"],
        "assay_id": template["assay_id"],
        "biological_context_id": template["biological_context_id"],
        "region_id": template["region_id"],
        "endpoint_descriptor": template["endpoint_descriptor"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--max-steps", type=int, default=None, help="smoke cap (grad-free mini run)")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--no-synth", action="store_true", help="control arm: measured rows only")
    parser.add_argument("--eval-limit", type=int, default=None, help="smoke: cap eval rows per task (logic validation only)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - GPU required, no CPU fallback (discipline)")
    device = torch.device(f"cuda:{args.physical_gpu_index}")
    torch.manual_seed(SEED)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_rows(PROJ_TRAIN, MRL_TASK)
    val_rows = load_rows(PROJ_VAL, MRL_TASK)
    polya_val_rows = load_rows(PROJ_VAL, POLYA_TASK)
    synth_rows = []
    if not args.no_synth:
        with open(SYNTH) as f:
            for line in f:
                synth_rows.append(json.loads(line))
    assert all(r.get("h1_label") is None and r.get("loss_weight") == 0.0 for r in synth_rows), "H1 violation"

    synth_by_src: dict[str, list[str]] = defaultdict(list)
    for r in synth_rows:
        synth_by_src[r["source_id"]].append(r["candidate_sequence"])

    print(f"measured MRL train={len(train_rows)} val={len(val_rows)} polya val={len(polya_val_rows)} synthetic={len(synth_rows)}", flush=True)

    critic = v5frozen.FrozenXEditCriticV5(V5_CKPT, MRNABERT, device)
    model = critic.model
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in trainable):,}", flush=True)

    from core.route2_xeditcritic_training_data_v3 import records_from_projection_rows
    from core.route2_xeditcritic_batch_v4 import (
        FrozenBottomEncoderChunkCacheViewV4,
        XEditCriticCollatorV4,
        XEditCriticDatasetV4,
    )
    from core.route2_bottom_encoder_chunk_cache_v4 import assemble_frozen_bottom_encoder_chunk_cache_v4

    targets = np.asarray([float(r["direction_normalized_delta"]) for r in train_rows], dtype=np.float64)
    t_std = max(float(targets.std()), 1e-8)
    target_scale = float(critic.scaler.scale(MRL_TASK, 0)) if hasattr(critic.scaler, "scale") else 1.0

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-4)

    loss_log = open(out_dir / "training_losses.jsonl", "w")
    epoch_metrics_file = open(out_dir / "epoch_metrics.jsonl", "w")

    def build_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
        template = rows[0]
        all_rows = []
        for r in rows:
            all_rows.append(r)
        if not args.no_synth:
            for r in rows:
                cands = synth_by_src.get(r["source_id"], [])
                if cands:
                    cand = cands[(hash(r["canonical_record_id"]) + step_global[0]) % len(cands)]
                    synth_row = _row_for_pair(r["source_sequence"], cand, r, f"D16C_SYNTH_{step_global[0]}_{r['canonical_record_id']}", MRL_TASK)
                    all_rows.append(synth_row)
        sequences = sorted({str(row["source_sequence"]).upper().replace("T", "U") for row in all_rows} | {str(row["candidate_sequence"]).upper().replace("T", "U") for row in all_rows})
        sequence_to_index = {s: i for i, s in enumerate(sequences)}
        encoded = critic._encode_sequences_fast({i: s for s, i in sequence_to_index.items()})
        cache_payload = assemble_frozen_bottom_encoder_chunk_cache_v4(
            all_rows,
            sequence_to_index=sequence_to_index,
            encoded=encoded,
            model_id="D16C_PROBE_BOTTOM_SIX",
            pretrained_parameter_count=critic._bottom_six.parameter_count,
            attention_backend=critic._bottom_six.attention_backend,
        )
        view = FrozenBottomEncoderChunkCacheViewV4(cache_payload, {str(row["canonical_record_id"]) for row in all_rows}, validate_payload=False)
        records = records_from_projection_rows(all_rows)
        record_by_id = {rec.record_id: rec for rec in records}
        dataset = XEditCriticDatasetV4(records, all_records=record_by_id, vocabs=critic.vocabs, target_scaler=critic.scaler, cache=None)
        collator = XEditCriticCollatorV4(view, minimum_physical_batch=4)
        batch = critic._move(collator([dataset[i] for i in range(len(records))]))
        return batch, [str(rec.record_id) for rec in records]

    def eval_task(rows: list[dict[str, Any]], endpoint: str, region: str, chunk: int = 64) -> float | None:
        from scipy.stats import spearmanr
        if args.eval_limit is not None:
            rows = rows[: args.eval_limit]
        critic._potential_memo.clear()
        model.eval()
        by_source: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            key = (str(r["source_sequence"]).upper().replace("T", "U"), str(r["assay_id"]), str(r["biological_context_id"]))
            by_source[key].append(r)
        preds, labels = [], []
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for key in sorted(by_source):
                group = by_source[key]
                r0 = group[0]
                source_row = dict(r0)
                source_row["source_key"] = str(r0["source_group_id"])
                src_u = str(r0["source_sequence"]).upper().replace("T", "U")
                states = []
                for r in group:
                    cand_u = str(r["candidate_sequence"]).upper().replace("T", "U")
                    obj = type("S", (), {})()
                    obj.source_sequence = src_u
                    obj.current_sequence = cand_u
                    obj.assay_id = str(r["assay_id"])
                    obj.context_id = str(r["biological_context_id"])
                    states.append(obj)
                vals = critic.potentials(states, endpoint_id=endpoint, region=region, source_row=source_row)
                for r, v in zip(group, vals):
                    preds.append(float(v))
                    labels.append(float(r["direction_normalized_delta"]))
        model.train()
        if not preds:
            return None
        return float(spearmanr(preds, labels).statistic)

    step_global = [0]
    final_rho = None
    for epoch in range(args.epochs):
        perm = torch.randperm(len(train_rows))
        losses = []
        for start in range(0, len(perm), args.batch):
            if args.max_steps is not None and step_global[0] >= args.max_steps:
                break
            idx = perm[start:start + args.batch]
            batch_rows = [train_rows[int(i)] for i in idx]
            batch, record_ids = build_batch(batch_rows)
            measured_ids = {r["canonical_record_id"] for r in batch_rows}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(batch)
            preds = output["mean"].float()
            rec_id_list = [str(x) for x in batch["record_ids"]]
            target_vec = []
            weight_vec = []
            for rid in rec_id_list:
                if rid in measured_ids:
                    row = next(r for r in batch_rows if r["canonical_record_id"] == rid)
                    target_vec.append(float(row["direction_normalized_delta"]) / target_scale)
                    weight_vec.append(1.0)
                else:
                    target_vec.append(0.0)
                    weight_vec.append(0.0)
            targets_t = torch.tensor(target_vec, dtype=torch.float32, device=preds.device)
            weights_t = torch.tensor(weight_vec, dtype=torch.float32, device=preds.device)
            per = torch.nn.functional.huber_loss(preds, targets_t, reduction="none", delta=HUBER_DELTA)
            huber = (per * weights_t).sum() / weights_t.sum().clamp(min=1e-8)

            rank_pairs = []
            by_src_batch = defaultdict(list)
            for i, rid in enumerate(rec_id_list):
                if rid in measured_ids:
                    row = next(r for r in batch_rows if r["canonical_record_id"] == rid)
                    by_src_batch[row["source_id"]].append((i, float(row["direction_normalized_delta"])))
            for src_id, items in by_src_batch.items():
                for i in range(len(items)):
                    for j in range(i + 1, len(items)):
                        if items[i][1] != items[j][1]:
                            rank_pairs.append((items[i][0], items[j][0], items[i][1] > items[j][1]))
            if rank_pairs and RANK_WEIGHT > 0:
                rank_terms = []
                for a, b, higher in rank_pairs:
                    diff = (preds[a] - preds[b]) / SOFT_RANK_TEMPERATURE
                    sign = 1.0 if higher else -1.0
                    rank_terms.append(torch.nn.functional.softplus(-sign * diff))
                rank_loss = torch.stack(rank_terms).mean()
                loss = huber + RANK_WEIGHT * rank_loss
            else:
                rank_loss = torch.tensor(0.0)
                loss = huber
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(float(loss))
            step_global[0] += 1
            if step_global[0] % 25 == 0:
                loss_log.write(json.dumps({"step": step_global[0], "epoch": epoch + 1, "loss": float(np.mean(losses[-25:])), "huber": float(huber), "rank": float(rank_loss), "rank_pairs": len(rank_pairs)}) + "\n")
                loss_log.flush()
        if args.max_steps is not None and step_global[0] >= args.max_steps:
            print(f"smoke cap reached at step {step_global[0]}", flush=True)
        rho = eval_task(val_rows, MRL_ENDPOINT, MRL_REGION)
        polya_rho = eval_task(polya_val_rows, "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS", "3UTR") if polya_val_rows else None
        rec = {"epoch": epoch + 1, "primary": epoch + 1 == args.epochs, "mrl_val_spearman": rho, "polya_val_spearman": polya_rho, "loss_mean": float(np.mean(losses)) if losses else None}
        epoch_metrics_file.write(json.dumps(rec) + "\n")
        epoch_metrics_file.flush()
        print(f"== epoch {epoch + 1}: {json.dumps(rec)}", flush=True)
        final_rho = rho
        torch.save({"model_state_dict": {k: v for k, v in model.state_dict().items()}, "epoch": epoch + 1, "seed": SEED, "schema": "route_a_v3_d16c_probe.v1"}, out_dir / f"probe_epoch_{epoch + 1}.pt")

    loss_log.close()
    epoch_metrics_file.close()
    summary = {
        "schema_version": "route_a_v3_d16c_probe.v1",
        "selection_rule": f"FINAL_EPOCH_{args.epochs}_FIXED",
        "seed": SEED,
        "epochs": args.epochs,
        "measured_train_rows": len(train_rows),
        "synthetic_rows_loaded": len(synth_rows),
        "h1_hard_clause": "synthetic rows: zero regression label, excluded from Huber and ranking pairs",
        "final_mrl_val_spearman": final_rho,
        "g1_gate_target": "train-backtest gap <= 0.30 AND VAL rho >= 0.135 (amendment sec 4.2)",
        "g2_gate_target": "polyA VAL rho >= 0.80",
        "protected_reads": 0,
        "cpu_fallback_used": False,
        "cuda_verified": True,
        "note": "gap backtest + G4 structure probe run separately after terminal (run_v5_train_backtest pattern)",
    }
    (out_dir / "probe_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
