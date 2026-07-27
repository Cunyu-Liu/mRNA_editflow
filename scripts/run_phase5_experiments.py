#!/usr/bin/env python3
"""Run the reproducible Phase 5 audit/report suite.

This script can score frozen labels only with an explicit flag.  It reports
what is scientifically eligible and keeps absent independent assays/cargos as
blockers instead of turning proxy or attribution-only evidence into claims.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from mrna_editflow.data.nmi_benchmark_v2 import FINAL_ROLES, iter_role_records


def metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    if len(y) == 0:
        return {"n": 0}
    ry, rp = np.argsort(np.argsort(y)), np.argsort(np.argsort(p))
    spearman = float(np.corrcoef(ry, rp)[0, 1]) if len(y) > 1 else 0.0
    beneficial = p > 0
    return {
        "n": int(len(y)), "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "spearman": spearman, "sign_accuracy": float(np.mean((y > 0) == (p > 0))),
        "beneficial_precision": float(np.mean(y[beneficial] > 0)) if np.any(beneficial) else 0.0,
        "beneficial_rate": float(np.mean(beneficial)),
    }


def gc(seq: str) -> float:
    return float(sum(c in "GC" for c in seq) / max(1, len(seq)))


def simple_predictions(train: List[dict], rows: List[dict]) -> Dict[str, np.ndarray]:
    deltas = np.asarray([float(r["delta"]) for r in train if r.get("delta") is not None], dtype=float)
    mean = float(deltas.mean()) if len(deltas) else 0.0
    pos_delta: Dict[int, List[float]] = defaultdict(list)
    gc_delta: Dict[int, List[float]] = defaultdict(list)
    for r in train:
        if r.get("delta") is None or not r.get("edit_list"): continue
        e = r["edit_list"][0]; d = float(r["delta"])
        pos_delta[int(e.get("pos", 0))].append(d)
        gc_delta[int((gc(str(r["candidate_sequence"])) - gc(str(r["source_sequence"]))) * 1000)].append(d)
    out = {"mean_predictor": np.full(len(rows), mean), "source_mean": np.full(len(rows), mean)}
    out["position_only"] = np.asarray([float(np.mean(pos_delta.get(int((r.get("edit_list") or [{"pos": 0}])[0].get("pos", 0)), [mean]))) for r in rows])
    out["gc_delta"] = np.asarray([float(np.mean(gc_delta.get(int((gc(str(r["candidate_sequence"])) - gc(str(r["source_sequence"]))) * 1000), [mean]))) for r in rows])
    return out


def mechanism_rows(rows: Iterable[dict]) -> Dict[str, Dict[str, float]]:
    values = []
    for r in rows:
        if r.get("delta") is None: continue
        seq = str(r["source_sequence"]); cand = str(r["candidate_sequence"]); e = (r.get("edit_list") or [{}])[0]
        pos = int(e.get("pos", 0)); alt = str(e.get("alt", "")); ref = str(e.get("ref", ""))
        local = {
            "delta": float(r["delta"]),
            "gc_delta": gc(cand) - gc(seq),
            "edit_position": float(pos) / max(1, len(seq)),
            "uaug_creation": float("AUG" in cand and "AUG" not in seq),
            "kozak_like_change": float((cand[max(0, pos - 6):pos + 3].count("G") - seq[max(0, pos - 6):pos + 3].count("G"))),
            "edit_is_transition": float({ref, alt} in ({"A", "G"}, {"C", "U"})),
        }
        values.append(local)
    out = {}
    for key in ("gc_delta", "edit_position", "uaug_creation", "kozak_like_change", "edit_is_transition"):
        x = np.asarray([v[key] for v in values]); y = np.asarray([v["delta"] for v in values])
        out[key] = {"n": int(len(x)), "pearson": float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0 else 0.0, "causal_status": "observational_matched_controls_required"}
    return out


def load_labeled(root: Path, role: str, *, allow_final_labels: bool, limit: int, confidence: str = "measured") -> List[dict]:
    rows: List[dict] = []
    for row in iter_role_records(root / "manifests" / f"{role}.json", allow_final_labels=allow_final_labels):
        if row.get("delta") is None or (confidence and row.get("confidence") != confidence):
            continue
        rows.append(row)
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-root", default="data/nmi_benchmark_v2")
    ap.add_argument("--out", default="artifacts/phase5/phase5_report.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--train-limit", type=int, default=20000)
    ap.add_argument("--final-limit", type=int, default=5000)
    ap.add_argument("--allow-final-labels", action="store_true")
    args = ap.parse_args()
    if not args.allow_final_labels:
        raise SystemExit("Phase 5 final evaluation requires --allow-final-labels after freeze")
    root = Path(args.benchmark_root)
    train = load_labeled(root, "train", allow_final_labels=False, limit=args.train_limit, confidence="measured")
    val = load_labeled(root, "val", allow_final_labels=False, limit=args.train_limit, confidence="measured")
    final = {}
    for role in sorted(FINAL_ROLES):
        final[role] = load_labeled(root, role, allow_final_labels=True, limit=args.limit if args.limit > 0 else args.final_limit, confidence="measured")
    baseline_report = {}
    for role, rows in final.items():
        preds = simple_predictions(train, rows)
        y = np.asarray([float(r["delta"]) for r in rows])
        baseline_report[role] = {name: metrics(y, pred) for name, pred in preds.items()}
    report = {
        "schema_version": "phase5_sota_ood_ablation_mechanism_v1",
        "final_test_used": True,
        "baseline_metrics": baseline_report,
        "formal_baselines": {
            name: {"status": "executable_adapter_present" if Path(path).exists() else "not_available", "path": path}
            for name, path in {
                "Optimus5Prime": "baselines/external_models.py", "UTailor": "baselines/external_utailor_adapter.py", "LinearDesign": "baselines/external_lineardesign_adapter.py", "UTRGAN": "baselines/external_utrgan_adapter.py", "AR": "baselines/ar_lm.py", "masked_diffusion": "baselines/masked_diffusion.py",
            }.items()
        },
        "ablation_matrix": {
            key: {"status": "protocol_registered", "requires": "five_seeds_and_paired_CI"}
            for key in ["relative_target", "source_encoder", "edit_token", "relative_position", "cargo_context", "cell_context", "uncertainty", "STOP", "region_embedding", "codon_state", "cross_region_attention", "proxy_pretraining", "measured_finetuning", "calibration"]
        },
        "ood": {role: {"n": len(rows), "status": "scored" if rows else "blocked_empty_role"} for role, rows in final.items()},
        "mechanism": mechanism_rows(final.get("test_id", [])),
        "claim_eligibility": {
            "local_delta": bool(final.get("test_id")),
            "family_transfer": bool(final.get("test_family")),
            "context_transfer": bool(final.get("test_context")),
            "assay_transfer": bool(final.get("test_assay")),
            "prospective_protein_output": False,
            "sota_headline": bool(final.get("test_family") and final.get("test_context") and final.get("test_assay")),
        },
        "policy": "proxy/internal scores and attribution-only mechanisms are not biological claims",
        "sampling": {"confidence": "measured_only", "train_labeled_limit": args.train_limit, "val_labeled_limit": args.train_limit, "final_labeled_limit": args.limit if args.limit > 0 else args.final_limit},
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
