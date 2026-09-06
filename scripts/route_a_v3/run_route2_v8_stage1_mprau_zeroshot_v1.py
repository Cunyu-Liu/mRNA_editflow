#!/usr/bin/env python3
"""V8 Stage 1 zero-shot on MPRAU (ENCSR854RUF VALIDATION, pair-mean caliber).

Loads a Stage 1 joint-pretrained checkpoint (S or H, mrl+polya domains) and
scores MPRAU records with the POLYA domain conditioning (closest 3'UTR domain
available in Stage 1; domain-conditioned readout selects the direction).
Reports record-level spearman AND variant pair-mean rho (W-ladder caliber)
vs V5 (0.1025) / Saluki (0.1205) references. Diagnostic for Stage 2 prereg
(zero-shot control arm), NOT a Stage-1 gate.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_v8_hybrid_backbone_v1 import build_v8_regressor, verify_vocab_alignment  # noqa: E402
from core.route2_v8_joint_library_v1 import format_sequence  # noqa: E402

EVAL_REPO = "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901"
_spec = importlib.util.spec_from_file_location("ev", EVAL_REPO + "/scripts/route_a_v3/evaluate_route2_prediction_v1.py")
ev = importlib.util.module_from_spec(_spec)
sys.modules["ev"] = ev
_spec.loader.exec_module(ev)

MRNABERT_PATH = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/mrnabert_a1eb7df25804d23f08646e1cb996b234d7208a40")
MANIFEST = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/manifests/route2_development_frozen_v1/development_manifest.jsonl")
CANONICAL = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/canonical/ENCSR854RUF/v1/canonical_records.private.jsonl")
EVAL_BATCH = 128
K = 10
MPRAU_DOMAIN_ID = 1  # polya (3'UTR) conditioning


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--arch", required=True, choices=("s", "h"))
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - GPU required")
    device = torch.device(f"cuda:{args.physical_gpu_index}")

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_v8_regressor(MRNABERT_PATH, args.arch, num_domains=3).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MRNABERT_PATH, local_files_only=True)
    verify_vocab_alignment(tokenizer)

    validation_ids = set()
    with MANIFEST.open() as fh:
        for line in fh:
            row = json.loads(line)
            if row["study_unit_id"] == "ENCSR854RUF" and row["split"] == "VALIDATION":
                validation_ids.add(str(row["canonical_record_id"]))
    records = {}
    with CANONICAL.open() as fh:
        for line in fh:
            row = json.loads(line)
            rid = str(row.get("canonical_record_id"))
            if rid in validation_ids:
                records[rid] = row
    ids = sorted(records)
    print(f"MPRAU VALIDATION records: {len(ids)}", flush=True)

    def score(seqs):
        vals = []
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for start in range(0, len(seqs), EVAL_BATCH):
                chunk = seqs[start:start + EVAL_BATCH]
                enc = tokenizer([format_sequence(s) for s in chunk],
                                add_special_tokens=True, padding=True, truncation=True,
                                max_length=512, return_tensors="pt")
                out = model(enc["input_ids"].to(device), enc["attention_mask"].to(device),
                            torch.full((len(chunk),), MPRAU_DOMAIN_ID, dtype=torch.long, device=device))
                vals.append(out.float().cpu().numpy())
        return np.concatenate(vals)

    src = score([records[r]["source_sequence"] for r in ids])
    cand = score([records[r]["candidate_sequence"] for r in ids])
    delta = cand - src
    preds = {r: float(delta[i]) for i, r in enumerate(ids)}

    obs = ev.load_observations([CANONICAL], validation_ids)
    metrics = ev.evaluate(obs, preds, K)

    by_variant = defaultdict(list)
    for rid in preds:
        if rid.startswith("ENCSR854RUF:"):
            by_variant[rid.split(":context:")[0]].append(rid)
    targets, ppreds = [], []
    for v in sorted(by_variant):
        rids = by_variant[v]
        if len(rids) >= 2:
            targets.append(float(np.mean([records[r]["direction_normalized_delta"] for r in rids])))
            ppreds.append(float(np.mean([preds[r] for r in rids])))
    rho = float(spearmanr(targets, ppreds).statistic)

    report = {
        "schema_version": "route_a_v3_route2_v8_stage1_mprau_zeroshot.v1",
        "arch": args.arch,
        "checkpoint": str(args.checkpoint),
        "domain_conditioning": "polya(3'UTR)",
        "n_records": len(ids),
        "n_variants": len(targets),
        "record_spearman": metrics.get("task_macro_spearman"),
        "pair_mean_rho": rho,
        "references": {"critic_v5": 0.1025, "saluki": 0.1205},
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
