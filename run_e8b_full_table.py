#!/usr/bin/env python3
"""E8b-full: complete the edit-budget x model performance stratified-rho table.
All tasks x all models x edit-count strata (1/2/3/4/5/6-8/9+)."""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
PROJ = MNT / "projections/xedit_v3/development_train_validation_v1/validation.jsonl"
HPO = MNT / "runs/development_hpo"
FULL_COV = MNT / "experiments/analysis_frozen_delta_full_coverage_20260904"
OUT = MNT / "experiments/analysis_delta_failure_mechanism_e0_e1_e3_b2"

TASKS = {
    "MRL": ("MEAN_RIBOSOME_LOAD::region=0", "mrl_gse114002"),
    "MPRAU": ("MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE::region=1", "mprau_encsr854ruf"),
    "polyA": ("PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS::region=1", "polya_gse269595"),
    "REFALT": ("PUBLISHED_REF_VS_ALT_ACTIVITY_LMM_LOG2_FOLD_CHANGE::region=1", "gse186455"),
    "HL5": ("RNA_HALF_LIFE_MINUTES::region=0", "half_life_5utr"),
    "HL3": ("RNA_HALF_LIFE_MINUTES::region=1", "half_life_3utr"),
    "TE200304": ("TOTAL_POLYSOME_TRANSLATION_EFFICIENCY::region=1", "gse200304_te"),
}

COMBOS = []
for label, (task_id, dirname) in TASKS.items():
    for model in ("rnafm", "utrlm"):
        p = FULL_COV / f"{dirname}__{model}" / "predictions.jsonl"
        if p.exists():
            COMBOS.append((label, model, task_id, p, None))
COMBOS += [
    ("MRL", "optimus", TASKS["MRL"][0], HPO / "external_lr1e3_wd1e4_replay_gpu5_v1/optimus5prime/validation_predictions.jsonl", "optimus5prime"),
    ("MRL", "framepool", TASKS["MRL"][0], HPO / "external_lr1e3_wd1e4_replay_gpu5_v1/framepool/validation_predictions.jsonl", "framepool"),
    ("polyA", "aparent", TASKS["polyA"][0], HPO / "aparent_v1/validation_predictions.jsonl", "aparent_official_base_cut_window"),
    ("MPRAU", "saluki", TASKS["MPRAU"][0], MNT / "experiments/analysis_saluki_frozen_mprau_20260903/predictions.jsonl", None),
    ("HL5", "saluki", TASKS["HL5"][0], MNT / "experiments/analysis_saluki_frozen_gse217518_20260903/predictions.jsonl", None),
    ("HL3", "saluki", TASKS["HL3"][0], MNT / "experiments/analysis_saluki_frozen_gse217518_20260903/predictions.jsonl", None),
]

rows_by_task = defaultdict(list)
with open(PROJ) as f:
    for line in f:
        d = json.loads(line)
        rows_by_task[d["task_id"]].append(d)

STRATA = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 8), (9, 99)]
table = {}
print(f"{'task':8s} {'model':10s} " + " ".join(f"e{lo}" + (f"-{hi}" if hi > lo else "") for lo, hi in STRATA))
for (label, model, task_id, pred_path, bid) in COMBOS:
    rows = rows_by_task[task_id]
    idx = {r["canonical_record_id"]: r for r in rows}
    ys, yh, nec = [], [], []
    with open(pred_path) as f:
        for line in f:
            d = json.loads(line)
            if bid is not None and d.get("baseline_id") != bid:
                continue
            rid = d["canonical_record_id"]
            if rid in idx:
                ys.append(idx[rid]["direction_normalized_delta"])
                yh.append(d["predicted_direction_normalized_delta"])
                nec.append(len(idx[rid].get("source_relative_edits", [])))
    if len(ys) < 30:
        continue
    ys, yh, nec = np.array(ys), np.array(yh), np.array(nec)
    row = {}
    cells = []
    for lo, hi in STRATA:
        m = (nec >= lo) & (nec <= hi)
        if m.sum() >= 10:
            r = float(spearmanr(yh[m], ys[m]).statistic)
            row[f"e{lo}" + (f"-{hi}" if hi > lo else "")] = {"n": int(m.sum()), "rho": r}
            cells.append(f"{r:.3f}({m.sum()})")
        else:
            m2 = (nec >= lo) & (nec <= hi)
            cells.append(f"—({int(m2.sum())})" if m2.sum() > 0 else "—")
    table[f"{label}__{model}"] = row
    print(f"{label:8s} {model:10s} " + " ".join(f"{c:12s}" for c in cells))

out = {"schema": "e8b_full_budget_rho_table.v1", "date": "2026-09-12",
       "strata": [f"e{lo}" + (f"-{hi}" if hi > lo else "") for lo, hi in STRATA], "table": table}
with open(OUT / "e8b_full_budget_rho_table.json", "w") as f:
    json.dump(out, f, indent=1)
print("saved:", OUT / "e8b_full_budget_rho_table.json")
