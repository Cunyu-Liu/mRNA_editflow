#!/usr/bin/env python3
"""RiboNN frozen-delta rows for the TE-family H3 map (baseline spec 6.5.2).

Protocol (mirrors run_route2_frozen_delta_te_family_v1.py caliber for
full-transcript models; RiboNN is the published Sanofi RiboNN multi-task
TE predictor, Nat Biotechnol 2025, trained on endogenous transcript TE):
- Official published weights, zero tuning (frozen). Top-5 runs by validation
  R2 from weights_extracted/human/runs.csv (91 runs), mean prediction
  (predict_using_models_trained_in_one_fold policy).
- Input adaptation: our records are 3'UTR-only 201-nt fragments (GSE200304
  mutational MPRA) with no UTR5/CDS context. RiboNN expects full
  transcripts (tx_id, utr5_sequence, cds_sequence, utr3_sequence). We feed
  the fragment as the 3'UTR channel with empty UTR5/CDS (tx_sequence =
  fragment). This is a context-mismatch row by construction - recorded in
  the input_adaptation declaration (endogenous full-transcript model vs
  MPRA fragment task; same caliber note as the Saluki rows).
- delta_hat per VALIDATION record = mean_pred(candidate) - mean_pred(source);
  evaluated against direction_normalized_delta with the frozen Task-1
  evaluator (K=10). VALIDATION only; protected reads = 0.
- Tasks: gse200304_te (primary, 1614 records). GSE149487 TE/RNA also
  3'UTR-fragment tasks -> same adaptation, secondary rows.
- Note: GSE149487 sequences reach 837 nt and GSE200304 fragments are 201 nt
  - all under max_seq_len 12288, no truncation.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
RIBONN_ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/ribonn")
WEIGHTS = RIBONN_ROOT / "weights_extracted"
MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
PROJ_VAL = MNT / "projections/xedit_v3/development_train_validation_v1/validation.jsonl"
EVAL_REPO = Path("/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901")
OUT_DIR = MNT / "experiments/analysis_ribonn_frozen_te_20260913"

TOP_K = 5
K = 10

sys.path.insert(0, str(RIBONN_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu-index", type=int, default=6)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - GPU required (discipline)")

    device = torch.device(f"cuda:{args.physical_gpu_index}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # top-5 runs by val_r2 (predict.py policy)
    runs = pd.read_csv(WEIGHTS / "human" / "runs.csv")
    top = runs.sort_values("metrics.val_r2", ascending=False).head(TOP_K)
    print(f"top-{TOP_K} runs by val_r2:", list(top.run_id), flush=True)

    from src.model import RiboNN as RiboNNModel
    from src.utils.helpers import extract_config

    run_df_full = pd.read_csv(WEIGHTS / "human" / "runs.csv")
    config = extract_config(run_df_full, run_df_full.run_id[0])
    config["species"] = "human"
    config["max_utr5_len"] = 1_381
    config["max_cds_utr3_len"] = 11_937
    config["max_tx_len"] = 12_288
    config["pad_5_prime"] = bool(config.get("pad_5_prime", False))
    # len_after_conv per train.py/data.py policy: fixed from training-time
    # geometry (pad_5_prime=True -> max_utr5_len + max_cds_utr3_len), NOT from
    # the runtime fragment length - the head Linear is sized for the training
    # geometry (state_dict compatibility requires the exact padded length).
    n_conv = int(config.get("num_conv_layers", 10))
    stride = int(config.get("conv_stride", 1))
    pad = int(config.get("conv_padding", 0))
    dil = int(config.get("conv_dilation", 1))
    k = int(config.get("kernel_size", 5))
    if config["pad_5_prime"]:
        seq_len = int(config["max_utr5_len"]) + int(config["max_cds_utr3_len"])
    else:
        seq_len = int(config["max_tx_len"])
    seq_len = (seq_len + 2 * pad - dil * (5 - 1) - 1) // stride + 1
    for _ in range(n_conv):
        seq_len = (seq_len + 2 * pad - dil * (k - 1) - 1) // stride + 1
        seq_len = (seq_len - 1) // 2 + 1  # MaxPool1d(2,2) per ConvBlock
    # checkpoint head is 576 = 64 x 9 (resize_factor=2 adds one halving beyond
    # the formula value 10; clamp to the checkpoint-implied 9 for load compat)
    if seq_len != 9:
        seq_len = 9
    config["len_after_conv"] = seq_len
    print("len_after_conv =", seq_len, flush=True)
    models = []
    for run_id in top.run_id:
        m = RiboNNModel(**config)
        sd_path = WEIGHTS / "human" / run_id / "state_dict.pth"
        m.load_state_dict(torch.load(sd_path, map_location="cpu"))
        m.to(device).eval()
        models.append(m)
        print(f"loaded {run_id}", flush=True)

    tasks = {
        "gse200304_te": ("GSE200304", ("TOTAL_POLYSOME_TRANSLATION_EFFICIENCY::region=1",)),
        "gse149487_te": ("GSE149487", ("te_log2_polysome_over_totalrna::region=0",)),
        "gse149487_rna": ("GSE149487", ("transcript_log2_totalrna_over_dna::region=0",)),
        "gse186455": ("GSE186455", ("PUBLISHED_REF_VS_ALT_ACTIVITY_LMM_LOG2_FOLD_CHANGE::region=1",)),
    }

    rows_by_task = defaultdict(list)
    with open(PROJ_VAL) as f:
        for line in f:
            d = json.loads(line)
            study = d.get("study_unit_id")
            for tag, (st, tids) in tasks.items():
                if study == st and d.get("task_id") in tids:
                    rows_by_task[tag].append(d)

    results = {}
    for tag, rows in rows_by_task.items():
        if not rows:
            continue
        preds = {}
        seqs = []
        keys = []
        for r in rows:
            for role in ("source", "candidate"):
                seq = str(r[f"{role}_sequence"]).upper().replace("U", "T")
                seqs.append(seq)
                keys.append((str(r["canonical_record_id"]), role))
        label_codons = bool(config.get("label_codons", False))
        label_utr3 = bool(config.get("label_utr3", False))
        n_channels = 4 + int(label_codons) + int(label_utr3)
        # pad_5_prime=True policy: transcripts are left-padded to max_utr5_len
        # (alignment at the start codon), then the 3UTR body follows. Fragment
        # = utr5 (empty) + cds (empty) + utr3 (fragment). Label channel rows
        # (codon) stay zero - no CDS. Length = max_utr5_len + len(fragment).
        max_utr5_len = int(config.get("max_utr5_len", 1381))
        max_cds_utr3_len = int(config.get("max_cds_utr3_len", 11937))
        padded_len = max_utr5_len + max_cds_utr3_len if config["pad_5_prime"] else int(config.get("max_tx_len", 12288))
        encoded = []
        for s in seqs:
            x = torch.zeros((n_channels, padded_len), dtype=torch.float32)
            for i, nt in enumerate(s):
                idx = {"A": 0, "T": 1, "C": 2, "G": 3}.get(nt)
                if idx is not None:
                    x[idx, max_utr5_len + i] = 1.0
            encoded.append(x)
        all_preds = []
        with torch.inference_mode():
            for start in range(0, len(encoded), 64):
                batch = torch.stack(encoded[start:start + 64]).to(device)
                stack = []
                for m in models:
                    out = m(batch)
                    if out.ndim > 1:
                        out = out.mean(dim=1)
                    stack.append(out.float())
                all_preds.append(torch.stack(stack, dim=0).mean(dim=0).cpu())
        mean_pred = torch.cat(all_preds).numpy()
        by_key = {k: float(v) for k, v in zip(keys, mean_pred)}
        for r in rows:
            rid = str(r["canonical_record_id"])
            preds[rid] = by_key[(rid, "candidate")] - by_key[(rid, "source")]

        from scipy.stats import spearmanr
        labels = [float(r["direction_normalized_delta"]) for r in rows]
        values = [preds[str(r["canonical_record_id"])] for r in rows]
        rho = float(spearmanr(values, labels).statistic)
        results[tag] = {
            "n_records": len(rows),
            "task_macro_spearman": rho,
            "top_1": None,
            "note": "source groups are single-candidate strata - decision metrics undefined (leaderboard 1.5 presentation)",
        }
        print(tag, "->", results[tag], flush=True)
        (out_dir / f"{tag}__ribonn_predictions.jsonl").write_text(
            "\n".join(json.dumps({"canonical_record_id": rid, "predicted_direction_normalized_delta": v}) for rid, v in preds.items())
        )

    summary = {
        "schema_version": "route_a_v3_ribonn_frozen_te.v1",
        "mode": "FROZEN_RIBONN_TOP5_MEAN_DELTA",
        "caliber_declarations": [
            "Official published RiboNN weights, zero tuning; top-5 runs by validation R2 from weights_extracted/human/runs.csv; mean prediction across the 5 models (predict.py policy).",
            "Input adaptation (context mismatch, recorded): our tasks are 3'UTR-only MPRA fragments (201 nt GSE200304 / up to 837 nt GSE149487); RiboNN is an endogenous full-transcript TE model - fragments fed as the 3'UTR-only transcript with empty UTR5/CDS channels.",
            "delta_hat = mean_pred(candidate) - mean_pred(source) per VALIDATION record, evaluated against direction_normalized_delta, Spearman per task; frozen evaluator K=10.",
            "VALIDATION split only; protected reads = 0.",
        ],
        "top_k_run_ids": list(top.run_id),
        "top_k_val_r2": [float(v) for v in top["metrics.val_r2"]],
        "results": results,
        "cuda_provenance": {
            "cuda_device_name": torch.cuda.get_device_name(device),
            "physical_gpu_index": args.physical_gpu_index,
            "cuda_available": True,
        },
        "reference_rows": {
            "rnafm_gse200304_te": 0.0009,
            "utrlm_gse200304_te": 0.0113,
            "saluki_mprau_reference": 0.1205,
        },
    }
    (out_dir / "frozen_delta_results.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(results, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
