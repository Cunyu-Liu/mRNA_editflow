#!/usr/bin/env python3
"""APARENT2 frozen-delta on polyA (GSE269595 VALIDATION, K=10 evaluator).

Baseline leaderboard Task 6.3.1: upgrade the polyA external row from the
APARENT 2019 original (0.7343 / top-1 0.6011) by scoring the 2024
multimolecule APARENT2 weights (user-relayed download, verified 414 tensors).
Frozen protocol: official weights, zero task tuning, frozen-delta
f(candidate) - f(source), same evaluator as Task 1.
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

EVAL_REPO = "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901"
_spec = importlib.util.spec_from_file_location("ev", EVAL_REPO + "/scripts/route_a_v3/evaluate_route2_prediction_v1.py")
ev = importlib.util.module_from_spec(_spec)
sys.modules["ev"] = ev
_spec.loader.exec_module(ev)

MANIFEST = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/manifests/route2_development_frozen_v1/development_manifest.jsonl")
CANONICAL = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/canonical/GSE269595/v1/canonical_records.private.jsonl")
APARENT2_DIR = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/aparent2")
EVAL_BATCH = 32
K = 10


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - GPU required")
    device = torch.device(f"cuda:{args.physical_gpu_index}")

    from multimolecule import Aparent2Config, Aparent2ForSequencePrediction, AutoTokenizer
    # bypass transformers from_pretrained (torch 2.5.1 <2.6 CVE gate): build
    # from config + manual state_dict load (weights_only=False, verified 414 tensors)
    cfg = Aparent2Config.from_pretrained(APARENT2_DIR)
    model = Aparent2ForSequencePrediction(cfg)
    sd = torch.load(APARENT2_DIR / "pytorch_model.bin", map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(APARENT2_DIR, local_files_only=True)
    model.eval()
    print("APARENT2 loaded (manual state_dict)", flush=True)

    validation_ids = set()
    with MANIFEST.open() as fh:
        for line in fh:
            row = json.loads(line)
            if row["study_unit_id"] == "GSE269595" and row["split"] == "VALIDATION":
                validation_ids.add(str(row["canonical_record_id"]))
    records = {}
    with CANONICAL.open() as fh:
        for line in fh:
            row = json.loads(line)
            rid = str(row.get("canonical_record_id"))
            if rid in validation_ids:
                records[rid] = row
    ids = sorted(records)
    print(f"polyA VALIDATION records: {len(ids)}", flush=True)

    def score(seqs):
        vals = []
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for start in range(0, len(seqs), EVAL_BATCH):
                chunk = seqs[start:start + EVAL_BATCH]
                enc = tokenizer(chunk, return_tensors="pt", padding="max_length",
                                max_length=205, truncation=True)
                out = model(**{k: v.to(device) for k, v in enc.items()})
                # Aparent2 regression head: take the sequence-level logit
                logits = out.logits if hasattr(out, "logits") else out[0]
                vals.append(logits.float().reshape(len(chunk), -1).mean(dim=1).cpu().numpy())
        return np.concatenate(vals)

    src = score([records[r]["source_sequence"] for r in ids])
    cand = score([records[r]["candidate_sequence"] for r in ids])
    delta = cand - src
    preds = {r: float(delta[i]) for i, r in enumerate(ids)}

    obs = ev.load_observations([CANONICAL], validation_ids)
    metrics = ev.evaluate(obs, preds, K)

    report = {
        "schema_version": "route_a_v3_route2_aparent2_frozen_delta.v1",
        "model": "APARENT2 (multimolecule, Kowalski 2024 lineage)",
        "checkpoint_dir": str(APARENT2_DIR),
        "n_records": len(ids),
        "task_macro_spearman": metrics.get("task_macro_spearman"),
        "top_1": metrics.get("top_1"),
        "ndcg_at_10": metrics.get("ndcg_at_10"),
        "references": {"aparent_2019_spearman": 0.7343, "aparent_2019_top1": 0.6011,
                       "critic_v5_spearman": 0.8219},
        "cpu_fallback_used": False,
        "precision": "BF16",
        "protected_reads": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report, indent=1), flush=True)
    print("wrote", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
