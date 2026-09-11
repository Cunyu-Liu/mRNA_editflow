#!/usr/bin/env python3
"""V9d CMS-augment arm terminal harvest (prereg batch76, script before data).

(1) Regenerate the same-seed control predictions (v8_stage2 s_mprau_in
    epoch6 FINAL checkpoint) with the SAME eval path as the v9d runner
    (eval_mprau_validation: 2,008 VALIDATION variants, mprau domain id 3,
    cell-conditioned, CUDA BF16 autocast, EVAL_BATCH 256).
(2) Paired per-variant bootstrap arm-vs-control (2,000 iters, seed 20260816)
    over the 2,008 shared variants, reusing the v8 runner's exact
    mprau_variant_table / pair_mean_rho / paired_bootstrap_vs_reference.
(3) Write adjudication_v9d_cms_augment.json (all numbers + three-state
    verdict per the prereg honesty clauses).

Discipline: verdicts strictly per preregistered gate (batch76):
  WINS        = point > 0.1351 AND bootstrap CI excludes zero (positive)
  DIRECTIONAL = point > 0.1351 but CI crosses zero
  FAIL        = point <= 0.1351 (regardless of CI)
No peak-picking; FINAL-EPOCH-FIXED only.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

W = Path("/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_v8_stage1_prep_20260904")
REPO_ROOT = W
sys.path.insert(0, str(REPO_ROOT))

spec = importlib.util.spec_from_file_location(
    "v8runner", str(W / "scripts/route_a_v3/run_route2_v8_stage2_adapt_v1.py"))
v8 = importlib.util.module_from_spec(spec)
sys.modules["v8runner"] = v8
spec.loader.exec_module(v8)  # parses args on demand only; main() not called

from core.route2_v8_hybrid_backbone_v1 import (  # noqa: E402
    DOMAIN_IDS, NUM_CELLS, build_v8_regressor, verify_vocab_alignment,
)
from core.route2_v8_joint_library_v1 import MNT  # noqa: E402

ARM_DIR = MNT / "experiments/xeditcritic_route_a/v9d_cms_augment_20260910/s_cmsaug_seed20260907"
CTRL_DIR = MNT / "experiments/xeditcritic_route_a/v8_stage2_adapt_20260907/s_mprau_in"
CTRL_CKPT = CTRL_DIR / "stage2_s_benchmark_full_epoch6.pt"
OUT_JSON = ARM_DIR / "adjudication_v9d_cms_augment.json"

MAIN_GATE = 0.1351


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - GPU required")
    device = torch.device(f"cuda:{args.physical_gpu_index}")
    torch.manual_seed(20260816)
    np.random.seed(20260816)

    from transformers import AutoTokenizer
    mrnabert_path = MNT / "external_model_assets/mrnabert_a1eb7df25804d23f08646e1cb996b234d7208a40"
    tokenizer = AutoTokenizer.from_pretrained(mrnabert_path, local_files_only=True)
    verify_vocab_alignment(tokenizer)

    report = json.load(open(ARM_DIR / "run_report.json"))
    arm_pair_mean = report["eval_final"]["mprau"]["pair_mean_spearman"]

    # ---- control predictions: regenerate with the identical eval path ----
    control_pred_path = CTRL_DIR / "mprau_predictions.jsonl"
    if not control_pred_path.is_file():
        model = build_v8_regressor(
            mrnabert_path, "s",
            num_domains=len(DOMAIN_IDS), num_cells=NUM_CELLS,
        ).to(device)
        v8.load_init(model, CTRL_CKPT)
        ctrl_mprau = v8.eval_mprau_validation(model, tokenizer, device, DOMAIN_IDS["mprau"])
        # eval_mprau_validation writes mprau_predictions.jsonl into v8._OUT_DIR;
        # default _OUT_DIR is "" -> repo cwd. Relocate it to the control dir.
        misplaced = W / "mprau_predictions.jsonl"
        if misplaced.is_file():
            misplaced.replace(control_pred_path)
        ctrl_pair_mean = ctrl_mprau["pair_mean_spearman"]
        ctrl_regen = True
        del model
        torch.cuda.empty_cache()
    else:
        ctrl_mprau = json.load(open(CTRL_DIR / "run_report.json"))["eval_final"]["mprau"]
        ctrl_pair_mean = ctrl_mprau["pair_mean_spearman"]
        ctrl_regen = False

    # ---- variant tables from prediction files (identical construction) ----
    def load_variants(pred_path: Path) -> dict[str, tuple[float, float]]:
        preds = {}
        with pred_path.open() as fh:
            for line in fh:
                row = json.loads(line)
                preds[str(row["canonical_record_id"])] = float(row["prediction"])
        records = v8._load_canonical(v8.BENCHMARK_DOMAINS["ENCSR854RUF"][1])
        targets = {rid: float(r["direction_normalized_delta"]) for rid, r in records.items()}
        return v8.mprau_variant_table(preds, targets)

    arm_variants = load_variants(ARM_DIR / "mprau_predictions.jsonl")
    ctrl_variants = load_variants(control_pred_path)
    arm_rho_from_preds = v8.pair_mean_rho(arm_variants)
    ctrl_rho_from_preds = v8.pair_mean_rho(ctrl_variants)

    # ---- paired bootstrap arm vs control (2,000 iters, seed 20260816) ----
    boot = v8.paired_bootstrap_vs_reference(ctrl_variants, arm_variants, "v8_s_mprau_in_seed20260907")

    # ---- three-state verdict per prereg honesty clauses ----
    point = arm_pair_mean
    ci_lo, ci_hi = boot["delta_ci95"]
    ci_excludes_zero_pos = (ci_lo > 0) and (ci_hi > 0)
    if point > MAIN_GATE and ci_excludes_zero_pos:
        verdict = "WINS"
    elif point > MAIN_GATE:
        verdict = "DIRECTIONAL"
    else:
        verdict = "FAIL"

    out = {
        "schema_version": "route_a_v3_route2_v9d_cms_augment_adjudication.v1",
        "arm": "s-cms-augment",
        "seed": 20260907,
        "generated_by": "adjudicate_route2_v9d_cms_augment_v1.py",
        "prereg": "journal_batch76 (batch 2026-09-10): main gate MPRAU VALIDATION pair_mean > 0.1351 (s_mprau_in 5-seed ensemble reference) AND paired bootstrap vs same-seed control CI excludes zero positive; honesty: point>gate & CI crosses zero -> DIRECTIONAL; point<=gate -> FAIL regardless of CI",
        "main_gate": MAIN_GATE,
        "arm_pair_mean_spearman": point,
        "arm_pair_mean_spearman_from_predictions": arm_rho_from_preds,
        "control": {
            "dir": str(CTRL_DIR),
            "checkpoint": str(CTRL_CKPT),
            "pair_mean_spearman_run_report": ctrl_pair_mean,
            "pair_mean_spearman_from_predictions": ctrl_rho_from_preds,
            "predictions_regenerated_this_run": ctrl_regen,
            "regen_protocol": "v8 runner eval_mprau_validation identical path (2,008 VALIDATION variants, domain id 3, cell-conditioned, CUDA BF16, EVAL_BATCH 256, FINAL epoch6 checkpoint)",
        },
        "bootstrap_arm_vs_control": boot,
        "verdict": verdict,
        "verdict_rules": {
            "WINS": "point > 0.1351 AND CI excludes zero (positive)",
            "DIRECTIONAL": "point > 0.1351 but CI crosses zero",
            "FAIL": "point <= 0.1351 regardless of CI",
        },
    }
    with OUT_JSON.open("w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
