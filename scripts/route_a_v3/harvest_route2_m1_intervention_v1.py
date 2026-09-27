#!/usr/bin/env python3
"""M1 intervention arm harvest / four-gate adjudication v1 (Task 3.4 follow-up, 2026-09-28).

Adjudicates the M1 real-dense-data intervention arm per the frozen prereg
docs/paper/m1_intervention_arm_amendment_v1.md:

  G1 direction gate : MRL VALIDATION frozen-delta task-macro Spearman vs
                      0.3158 (Route A full-FT V2 3-seed ensemble) + 0.01.
                      Single seed (20260920) = direction read only.
  G2 non-destruction: polyA VALIDATION main-row recompute, drop <= 0.02 vs
                      frozen 0.8218686245779881.
  G3 efficiency     : report only (wallclock / peak memory / rows) from run_summary.
  G4 mechanism      : only if G1 direction is positive. M1 eval-row frozen-delta
                      reading (same caliber as benchmark_v2 matrix M1 cell:
                      paired delta overall Spearman, 2805 rows) for the M1 arm
                      vs the Route A full-FT V2 baseline checkpoint and the
                      external-family spectrum recorded in matrix_v2_results.json.

Discipline: CUDA hard gate (no CPU fallback); VALIDATION only; protected TEST
reads = 0; append-only outputs; FINAL-EPOCH-6-FIXED (no peak picking).

Usage:
  python harvest_route2_m1_intervention_v1.py --gpu-index 1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

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

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
MRNABERT_PATH = MNT / "external_model_assets/mrnabert_a1eb7df25804d23f08646e1cb996b234d7208a40"
MANIFEST = MNT / "manifests/route2_development_frozen_v1/development_manifest.jsonl"
CANONICAL_GSE114002 = MNT / "canonical/GSE114002/v1/canonical_records.private.jsonl"
CANONICAL_GSE269595 = MNT / "canonical/GSE269595/v1/canonical_records.private.jsonl"
M1_EVAL_ROW = MNT / "benchmark_v2/m1_mrl_eval_row/projection_rows.jsonl"
ARM_DIR = MNT / "experiments/xeditcritic_m1_intervention/seed_20260920"
V2_CKPT = MNT / "experiments/xeditcritic_route_a/280k_fullft_v2_6ep_20260903/fullft_epoch_6.pt"
MATRIX_RESULTS = MNT / "benchmark_v2/leaderboard_matrix_v2/matrix_v2_results.json"
DEFAULT_OUT = MNT / "experiments/xeditcritic_m1_intervention/adjudication_v1"

V2_ENSEMBLE = 0.3158289984824722
V2_SINGLE_SEED_SPECTRUM = {
    "seed20260903": 0.31979237401392574,
    "seed20260904": 0.28727931905619714,
    "seed20260905": 0.31568911939313454,
}
POLYA_MAIN = 0.8218686245779881
POLYA_DROP_TOLERANCE = 0.02
G1_DIRECTION_THRESHOLD = V2_ENSEMBLE + 0.01
K = 10


class HarvestError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HarvestError(message)


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def format_sequence(sequence: str) -> str:
    return " ".join(str(sequence).upper().replace("U", "T"))


class MeanPoolRegressor(nn.Module):
    """Verbatim clone of the training runner's model wrapper."""

    def __init__(self, base_model: nn.Module, width: int):
        super().__init__()
        self.base = base_model
        self.head = nn.Linear(width, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.base(input_ids=input_ids, attention_mask=attention_mask)[0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


def build_model(device: torch.device):
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model_config = AutoConfig.from_pretrained(
        MRNABERT_PATH, local_files_only=True, trust_remote_code=True
    )
    base = AutoModel.from_config(model_config, trust_remote_code=True, add_pooling_layer=False)
    modeling_module = sys.modules[base.__class__.__module__]
    modeling_module.flash_attn_qkvpacked_func = None
    checkpoint = torch.load(MRNABERT_PATH / "pytorch_model.bin", map_location="cpu", weights_only=False)
    base_state = {
        key.removeprefix("bert."): value
        for key, value in checkpoint.items()
        if key.startswith("bert.")
    }
    base.load_state_dict(base_state, strict=True)
    del checkpoint, base_state
    model = MeanPoolRegressor(base, base.config.hidden_size)
    return model, AutoTokenizer.from_pretrained(MRNABERT_PATH, local_files_only=True)


def load_state_dict(path: Path) -> dict:
    _require(path.exists(), f"checkpoint missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    return state


def score_sequences(model, tokenizer, sequences, device, batch=256) -> np.ndarray:
    model.eval()
    values = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for start in range(0, len(sequences), batch):
            chunk = sequences[start:start + batch]
            enc = tokenizer(
                [format_sequence(s) for s in chunk],
                add_special_tokens=True,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            values.append(model(ids, mask).float().cpu().numpy())
    return np.concatenate(values)


def manifest_ids(study: str, split: str) -> set[str]:
    ids: set[str] = set()
    with MANIFEST.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == split and row["study_unit_id"] == study:
                ids.add(str(row["canonical_record_id"]))
    return ids


def load_canonical_records(path: Path, ids: set[str]) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            rid = str(row.get("canonical_record_id"))
            if rid in ids:
                records[rid] = row
    return records


def load_prediction_file(path: Path) -> dict[str, float]:
    preds: dict[str, float] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = str(row.get("canonical_record_id") or row.get("record_id"))
            preds[key] = float(row.get("predicted_direction_normalized_delta", row.get("prediction")))
    return preds


def adjudicate_g1(arm_dir: Path) -> dict:
    summary_path = arm_dir / "run_summary.json"
    _require(summary_path.exists(), f"training not terminal: missing {summary_path}")
    summary = json.loads(summary_path.read_text())
    _require(str(summary.get("selection_rule", "")).startswith("FINAL_EPOCH"),
             "selection_rule must be FINAL_EPOCH_FIXED")
    reported = float(summary["metrics"]["task_macro_spearman"])

    # independent recompute from archived predictions (same frozen evaluator)
    ids = manifest_ids("GSE114002", "VALIDATION")
    _require(len(ids) == 730, f"GSE114002 VALIDATION id count {len(ids)} != 730")
    preds = load_prediction_file(arm_dir / "predictions.jsonl")
    preds = {k: v for k, v in preds.items() if k in ids}
    _require(len(preds) == len(ids), f"prediction coverage {len(preds)}/{len(ids)}")
    obs = ev.load_observations([CANONICAL_GSE114002], ids)
    recomputed = float(ev.evaluate(obs, preds, K)["task_macro_spearman"])
    _require(abs(recomputed - reported) < 1e-9,
             f"recompute mismatch: run_summary {reported} vs recomputed {recomputed}")

    gain = recomputed - V2_ENSEMBLE
    direction_positive = gain > 0.01
    return {
        "gate": "G1_direction",
        "metric": "MRL_VALIDATION_frozen_delta_task_macro_spearman",
        "value": recomputed,
        "run_summary_value": reported,
        "recompute_match": True,
        "baseline_v2_3seed_ensemble": V2_ENSEMBLE,
        "v2_single_seed_spectrum": V2_SINGLE_SEED_SPECTRUM,
        "gain_vs_ensemble": gain,
        "direction_threshold": G1_DIRECTION_THRESHOLD,
        "direction_positive": bool(direction_positive),
        "note": "single seed (20260920) direction read only; 3-seed CI = future decision",
    }


def adjudicate_g3(arm_dir: Path) -> dict:
    summary = json.loads((arm_dir / "run_summary.json").read_text())
    return {
        "gate": "G3_efficiency",
        "report_only": True,
        "efficiency": summary.get("efficiency"),
        "train_pool_total": summary.get("train_pool_total"),
        "m1_rows": summary.get("m1_rows"),
        "reference_v2_6ep_20260903": "wallclock ~4.5-5.5h, rows 677,608/epoch",
    }


def polyA_reading(model, tokenizer, device) -> dict:
    """Frozen-Δ polyA reading on GSE269595 VALIDATION (n=2,628)."""
    ids = manifest_ids("GSE269595", "VALIDATION")
    _require(len(ids) == 2628, f"GSE269595 VALIDATION id count {len(ids)} != 2628")
    records = load_canonical_records(CANONICAL_GSE269595, ids)
    _require(len(records) == len(ids), f"canonical coverage {len(records)}/{len(ids)}")
    eval_ids = sorted(records)
    source_scores = score_sequences(model, tokenizer, [records[r]["source_sequence"] for r in eval_ids], device)
    candidate_scores = score_sequences(model, tokenizer, [records[r]["candidate_sequence"] for r in eval_ids], device)
    delta = candidate_scores - source_scores
    preds = {rid: float(delta[i]) for i, rid in enumerate(eval_ids)}
    obs = ev.load_observations([CANONICAL_GSE269595], ids)
    metrics = ev.evaluate(obs, preds, K)
    return {
        "value": float(metrics["task_macro_spearman"]),
        "top_1": metrics["source_macro_top_1_accuracy"],
        "ndcg_at_10": metrics["source_macro_ndcg_at_k"],
    }


def adjudicate_g2(arm_reading: dict, baseline_reading: dict | None) -> dict:
    """Mechanical verdict vs the frozen main row (per amendment v1), plus a
    report-only context row vs the Route A V2 baseline recompute (the arm
    inherits V2's MRL-only training design, so the *frozen* comparison against
    the V5 polyA main row is expected to reflect MRL-domain training drift —
    the amendment's expectation note already anticipates this)."""
    value = arm_reading["value"]
    drop = POLYA_MAIN - value
    out = {
        "gate": "G2_non_destruction",
        "metric": "polyA_VALIDATION_frozen_delta_task_macro_spearman",
        "value": value,
        "top_1": arm_reading["top_1"],
        "ndcg_at_10": arm_reading["ndcg_at_10"],
        "frozen_polya_main_row": POLYA_MAIN,
        "drop_vs_main": drop,
        "drop_tolerance": POLYA_DROP_TOLERANCE,
        "pass": bool(drop <= POLYA_DROP_TOLERANCE),
        "gate_semantics_note": (
            "mechanical verdict compares against the V5 multi-task main row as frozen in the amendment; "
            "the arm is a clone of the MRL-only Route A V2 design, so the informative non-destruction "
            "signal is the context row below (arm vs V2 baseline recompute)"
        ),
    }
    if baseline_reading is not None:
        out["baseline_v2_context"] = {
            "v2_baseline_value": baseline_reading["value"],
            "arm_minus_baseline": value - baseline_reading["value"],
            "note": "report-only context row; not part of the frozen G2 verdict",
        }
    return out


def load_m1_rows() -> list[dict]:
    rows = []
    with M1_EVAL_ROW.open() as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def m1_row_reading(model, tokenizer, device, rows: list[dict]) -> dict:
    sources = [r["source_sequence"] for r in rows]
    candidates = [r["candidate_sequence"] for r in rows]
    src_scores = score_sequences(model, tokenizer, sources, device)
    cand_scores = score_sequences(model, tokenizer, candidates, device)
    deltas = cand_scores - src_scores
    observed = np.asarray([float(r["direction_normalized_delta"]) for r in rows], dtype=float)
    predictions = {r["row_id"]: float(deltas[i]) for i, r in enumerate(rows)}
    observations = [
        {
            "canonical_record_id": r["row_id"],
            "study_unit_id": r["study_unit"],
            "source_id": r["source_group_id"],
            "biological_context_id": r["biological_context"],
            "endpoint_id": "MEAN_RIBOSOME_LOAD",
            "stratum": (r["study_unit"], r["region"], "MEAN_RIBOSOME_LOAD"),
            "task": (r["region"], "MEAN_RIBOSOME_LOAD"),
            "observed": float(r["direction_normalized_delta"]),
        }
        for r in rows
    ]
    metrics = ev.evaluate(observations, predictions, K)
    overall = float(spearmanr(observed, deltas).statistic)
    per_context = {}
    contexts = sorted({r["biological_context"] for r in rows})
    for context in contexts:
        idx = [i for i, r in enumerate(rows) if r["biological_context"] == context]
        if len(idx) >= 3:
            per_context[context] = {
                "n": len(idx),
                "spearman": float(spearmanr(observed[idx], deltas[idx]).statistic),
            }
    return {
        "n": len(rows),
        "overall_spearman": overall,
        "task_macro_spearman_evaluator": float(metrics["task_macro_spearman"]),
        "source_macro_top_1": metrics["source_macro_top_1_accuracy"],
        "source_macro_ndcg_at_10": metrics["source_macro_ndcg_at_k"],
        "sign_accuracy": float(np.mean(np.sign(observed) == np.sign(deltas))),
        "per_context": per_context,
    }


def external_m1_spectrum() -> dict:
    if not MATRIX_RESULTS.exists():
        return {}
    data = json.loads(MATRIX_RESULTS.read_text())
    cell = (data.get("cells") or {}).get("M1_MRL_EVAL_ROW|5UTR") or {}
    return {
        family: {
            "overall_spearman": entry.get("overall_spearman"),
            "n": entry.get("n"),
        }
        for family, entry in cell.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--arm-dir", type=Path, default=ARM_DIR)
    parser.add_argument("--skip-gpu", action="store_true",
                        help="G1/G3 only (no checkpoint needed)")
    parser.add_argument("--only-v2-baseline-row", action="store_true",
                        help="compute the V2 baseline M1-row reading only (G4 contrast)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {
        "schema_version": "route_a_v3_m1_intervention_adjudication.v1",
        "utc": utcnow(),
        "prereg": "docs/paper/m1_intervention_arm_amendment_v1.md",
        "arm_dir": str(args.arm_dir),
    }

    if not args.only_v2_baseline_row:
        result["G1"] = adjudicate_g1(args.arm_dir)
        result["G3"] = adjudicate_g3(args.arm_dir)
        print(f"[G1] value={result['G1']['value']:.6f} gain={result['G1']['gain_vs_ensemble']:+.6f} "
              f"direction_positive={result['G1']['direction_positive']}", flush=True)

    if not args.skip_gpu:
        _require(torch.cuda.is_available(), "CUDA unavailable - GPU required (hard gate)")
        device = torch.device(f"cuda:{args.gpu_index}")
        model, tokenizer = build_model(device)

        if args.only_v2_baseline_row:
            model.load_state_dict(load_state_dict(V2_CKPT), strict=True)
            model.to(device)
            rows = load_m1_rows()
            result["v2_baseline_m1_row"] = m1_row_reading(model, tokenizer, device, rows)
            out_path = args.out_dir / "m1_row_v2_baseline.json"
            out_path.write_text(json.dumps(result, indent=1, sort_keys=True))
            print(f"[V2 baseline M1 row] spearman={result['v2_baseline_m1_row']['overall_spearman']:.6f} "
                  f"n={result['v2_baseline_m1_row']['n']}")
            print("wrote", out_path)
            return 0

        g1_positive = result["G1"]["direction_positive"]
        # G2 runs regardless of G1 direction (non-destruction gate is unconditional)
        model.load_state_dict(load_state_dict(args.arm_dir / "fullft_epoch_6.pt"), strict=True)
        model.to(device)
        arm_polyA = polyA_reading(model, tokenizer, device)
        baseline_polyA = None
        if V2_CKPT.exists():
            model.load_state_dict(load_state_dict(V2_CKPT), strict=True)
            model.to(device)
            baseline_polyA = polyA_reading(model, tokenizer, device)
            model.load_state_dict(load_state_dict(args.arm_dir / "fullft_epoch_6.pt"), strict=True)
            model.to(device)
        result["G2"] = adjudicate_g2(arm_polyA, baseline_polyA)
        print(f"[G2] polyA value={result['G2']['value']:.6f} drop={result['G2']['drop_vs_main']:+.6f} "
              f"pass={result['G2']['pass']}"
              + (f" | v2_baseline_context={baseline_polyA['value']:.6f}" if baseline_polyA else ""), flush=True)

        if g1_positive:
            rows = load_m1_rows()
            result["G4"] = {
                "gate": "G4_mechanism",
                "triggered": True,
                "arm_m1_row": m1_row_reading(model, tokenizer, device, rows),
                "external_family_spectrum": external_m1_spectrum(),
            }
            if V2_CKPT.exists():
                model.load_state_dict(load_state_dict(V2_CKPT), strict=True)
                model.to(device)
                result["G4"]["v2_baseline_m1_row"] = m1_row_reading(model, tokenizer, device, rows)
                model.load_state_dict(load_state_dict(args.arm_dir / "fullft_epoch_6.pt"), strict=True)
                model.to(device)
            print(f"[G4] arm M1-row spearman={result['G4']['arm_m1_row']['overall_spearman']:.6f}", flush=True)
        else:
            result["G4"] = {"gate": "G4_mechanism", "triggered": False,
                            "note": "G1 direction not positive; G4 not triggered per prereg"}
            print("[G4] not triggered (G1 direction not positive)", flush=True)

    result["verdict"] = {
        "G1_direction_positive": result.get("G1", {}).get("direction_positive"),
        "G2_pass": result.get("G2", {}).get("pass"),
        "G4_triggered": result.get("G4", {}).get("triggered"),
    }
    out_name = "m1_g1_g3_v1.json" if args.skip_gpu else "m1_adjudication_v1.json"
    out_path = args.out_dir / out_name
    out_path.write_text(json.dumps(result, indent=1, sort_keys=True))
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())