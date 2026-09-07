#!/usr/bin/env python3
"""UTR-STCNet frozen zero-shot delta on GSE114002 (MRL, 5'UTR ribosome load).

D6=P1 approved baseline row (SPECS_BASELINE_LEADERBOARD 6.3.4): the two modern
2025 SOTA rows are UTR-STCNet (MRL) and HydraRNA (stability). This script runs
UTR-STCNet's three official checkpoints (MPRA-H / MPRA-U / MPRA-V) under the
frozen protocol: official weights, score source and candidate sequences,
delta = candidate - source, evaluate against direction_normalized_delta with
the frozen Task-1 evaluator (K=10) on GSE114002 VALIDATION only.

Calibre identical to the frozen-Optimus/FramePool MRL rows
(run_route2_frozen_delta_gse114002_v1.py) - pure frozen zero-shot, no probe,
no fine-tuning.

Known limitation (declared, not hidden): UTR-STCNet training data (MPA_H/U/V
CSVs) is not present on the server, so the 3-block pigeonhole leakage audit
cannot be executed; the row carries leakage_audit="N/A - training data absent"
exactly as declared. The frozen protocol R1 (official weights zero-tuning)
still holds.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

# UTR-STCNet repo (code) lives under external_model_assets/utr_stcnet
UTR_STCNET_ROOT = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/utr_stcnet"
if UTR_STCNET_ROOT not in sys.path:
    sys.path.insert(0, UTR_STCNET_ROOT)
from UTR.UTRFormer import utrformer_large  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EVAL_REPO = "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901"
_ev_spec = importlib.util.spec_from_file_location(
    "ev", EVAL_REPO + "/scripts/route_a_v3/evaluate_route2_prediction_v1.py"
)
ev = importlib.util.module_from_spec(_ev_spec)
sys.modules["ev"] = ev
_ev_spec.loader.exec_module(ev)

MANIFEST = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/manifests/route2_development_frozen_v1/development_manifest.jsonl"
)
CANONICAL = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/canonical/GSE114002/v1/canonical_records.private.jsonl"
)
CKPT_DIR = Path(UTR_STCNET_ROOT) / "checkpoint/UTR-STCNet_checkpoints"
MODELS = {
    "mpra_h": CKPT_DIR / "MPRA-H/UTR-H_new_RL_epoch300_batchsize256_padd120_new.pkl",
    "mpra_u": CKPT_DIR / "MPRA-U/UTR-U_new_RL_epoch300_batchsize256_padd120_new.pkl",
    "mpra_v": CKPT_DIR / "MPRA-V/UTR-V_new_RL_epoch300_batchsize256_padd120_new.pkl",
}
SEQ_MAX_LEN = 120
SEQ_MAP = {"A": [1, 0, 0, 0], "C": [0, 1, 0, 0], "G": [0, 0, 1, 0], "T": [0, 0, 0, 1]}
EVAL_BATCH = 256
K = 10


def one_hot(sequences: list[str]) -> torch.Tensor:
    """Right-aligned [B, L, 4] one-hot (A/C/G/T; N/other -> 0), like UTRDATA."""
    tokens = torch.zeros((len(sequences), SEQ_MAX_LEN, 4), dtype=torch.int64)
    for i, seq in enumerate(sequences):
        s = str(seq).upper().replace("U", "T")
        if len(s) > SEQ_MAX_LEN:
            s = s[-SEQ_MAX_LEN:]
        for j, ch in enumerate(s):
            code = SEQ_MAP.get(ch)
            if code is not None:
                tokens[i, SEQ_MAX_LEN - len(s) + j] = torch.tensor(code, dtype=torch.int64)
    return tokens


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--output-dir", type=Path, default=Path(
        "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/analysis_utr_stcnet_frozen_delta_mrl_20260907"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - GPU required")
    device = torch.device(f"cuda:{args.physical_gpu_index}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validation_ids = set()
    with MANIFEST.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["study_unit_id"] == "GSE114002" and row["split"] == "VALIDATION":
                validation_ids.add(str(row["canonical_record_id"]))
    records = {}
    with CANONICAL.open() as handle:
        for line in handle:
            row = json.loads(line)
            rid = str(row["canonical_record_id"])
            if rid in validation_ids:
                records[rid] = row
    ids = sorted(records)
    observations = ev.load_observations([CANONICAL], set(ids))
    print(f"GSE114002 VALIDATION n={len(ids)}", flush=True)

    results = {}
    for name, ckpt_path in MODELS.items():
        if not ckpt_path.is_file():
            results[name] = {"error": f"checkpoint absent: {ckpt_path}"}
            print(f"{name}: SKIP (checkpoint absent)", flush=True)
            continue
        model = utrformer_large(padding_idx=0, token_cls=5, pooling_size=SEQ_MAX_LEN)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ck = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict({k.replace("module.", ""): v for k, v in ck.items()})
        model.to(device)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())

        def score(seqs: list[str]) -> np.ndarray:
            values = []
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for start in range(0, len(seqs), EVAL_BATCH):
                    chunk = seqs[start:start + EVAL_BATCH]
                    tok = one_hot(chunk).to(device).to(torch.float32)
                    out = model(tok, train=False)
                    values.append(out[1].float().cpu().numpy().reshape(-1))
            return np.concatenate(values)

        source = score([records[r]["source_sequence"] for r in ids])
        candidate = score([records[r]["candidate_sequence"] for r in ids])
        delta = candidate - source
        predictions = {rid: float(delta[i]) for i, rid in enumerate(ids)}
        metrics = ev.evaluate(observations, predictions, K)
        rec = {
            "model": name,
            "checkpoint": str(ckpt_path),
            "official_weights": True,
            "fine_tuned": False,
            "n_params": n_params,
            "split": "VALIDATION",
            "n_records": len(ids),
            "task_macro_spearman": metrics.get("task_macro_spearman"),
            "top_1": metrics.get("source_macro_top_1_accuracy"),
            "ndcg_at_10": metrics.get("source_macro_ndcg_at_k"),
            "delta_std": float(delta.std()),
            "leakage_audit": "N/A - training data (MPA_H/U/V CSVs) absent on server",
            "cpu_fallback_used": False,
        }
        results[name] = rec
        print(f"{name}: spearman {rec['task_macro_spearman']:.4f} | top-1 {rec['top_1']} | "
              f"ndcg@10 {rec['ndcg_at_10']:.4f} | delta_std {rec['delta_std']:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

    out = {
        "schema_version": "route_a_v3_route2_utr_stcnet_frozen_delta_mrl.v1",
        "protocol": "FROZEN_ZERO_SHOT_DELTA (frozen-Optimus/FramePool MRL calibre)",
        "evaluator": "Task-1 frozen evaluator K=10",
        "selection": "official weights, no tuning",
        "results": results,
    }
    (args.output_dir / "frozen_delta_results.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print("wrote", args.output_dir / "frozen_delta_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
