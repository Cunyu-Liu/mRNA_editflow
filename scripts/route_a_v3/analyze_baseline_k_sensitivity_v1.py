#!/usr/bin/env python3
"""Baseline P0-4: NDCG@K sensitivity analysis (K in {3,5,10,20}).

Re-evaluates every persisted per-record prediction file of the frozen-delta
coverage batch (+ Saluki rows + critic V5 headline rows) at multiple K using
the identical frozen evaluator (ev.evaluate), isolating the K-sensitivity of
the leaderboard's decision metrics (R4 closure, SPECS_BASELINE_LEADERBOARD
spec 2026-09-08 增补 §V.4 P0-4).

Rows WITHOUT persisted per-record predictions (Optimus/FramePool MRL Stage 0a,
APARENT/APARENT2 polyA) are registered as P0-4b (GPU rescore follow-up) --
this script covers every row that has predictions.jsonl on disk.

Discipline: read-only over frozen products; VALIDATION split only; no GPU.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

W0 = Path("/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_w0_diagnosis_20260902")
TE_SCRIPT = W0 / "scripts/route_a_v3/run_route2_frozen_delta_te_family_v1.py"
_spec = importlib.util.spec_from_file_location("te", TE_SCRIPT)
te = importlib.util.module_from_spec(_spec)
sys.modules["te"] = te
_spec.loader.exec_module(te)
ev = te.ev

# P1 task specs (mirror run_route2_frozen_delta_full_coverage_v1.py lines 73-97)
P1_TASKS = {
    "polya_gse269595": {"study": "GSE269595", "region": "3UTR",
                        "endpoint": "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS", "mode": "IN_STUDY_PROBE"},
    "mprau_encsr854ruf": {"study": "ENCSR854RUF", "region": "3UTR",
                          "endpoint": "MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE", "mode": "IN_STUDY_PROBE"},
    "half_life_5utr": {"study": "GSE217518", "region": "5UTR",
                       "endpoint": "RNA_HALF_LIFE_MINUTES", "mode": "IN_STUDY_PROBE"},
    "half_life_3utr": {"study": "GSE217518", "region": "3UTR",
                       "endpoint": "RNA_HALF_LIFE_MINUTES", "mode": "IN_STUDY_PROBE"},
}
te.TASKS.update(P1_TASKS)

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
COV = MNT / "experiments/analysis_frozen_delta_full_coverage_20260904"
OUT = MNT / "experiments/analysis_baseline_k_sensitivity_20260908"
KS = (3, 5, 10, 20)

# V5 predictions: single file with all tasks; map task -> (study, record filter)
V5_PRED = sorted(MNT.glob("experiments/xeditcritic_v5/*/v5_full/final_validation_predictions.jsonl"))


def load_predictions(path: Path) -> dict[str, float]:
    preds = {}
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            rid = str(row.get("canonical_record_id") or row.get("record_id"))
            preds[rid] = float(row.get("prediction", row.get("predicted_direction_normalized_delta")))
    return preds


def eval_at_ks(task_key: str, predictions: dict[str, float]) -> dict:
    spec = te.TASKS[task_key]
    observations = ev.load_observations([te.canonical_path_for(spec["study"])], set(predictions))
    out = {}
    for k in KS:
        m = ev.evaluate(observations, predictions, k)
        out[f"K={k}"] = {
            "ndcg_at_k": m.get("source_macro_ndcg_at_k"),
            "top_1": m.get("source_macro_top_1_accuracy"),
            "task_macro_spearman": m.get("task_macro_spearman"),
        }
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": "route_a_v3_baseline_k_sensitivity.v1",
              "ks": list(KS), "rows": {},
              "gaps": ["Optimus/FramePool MRL 与 APARENT/APARENT2 polyA 的 Stage 0a/补测产物未持久化逐条预测 -> P0-4b GPU 重打分登记"]}

    # 1) frozen-delta full coverage rows (8 tasks x rnafm/utrlm)
    for run_dir in sorted(COV.glob("*__*")):
        if not run_dir.is_dir():
            continue
        task_key, model = run_dir.name.split("__", 1)
        pred_file = run_dir / "predictions.jsonl"
        if task_key not in te.TASKS or not pred_file.exists():
            continue
        report["rows"][f"{task_key}__{model}"] = eval_at_ks(task_key, load_predictions(pred_file))
        print(f"[k] {run_dir.name}: done", flush=True)

    # 2) Saluki rows (GSE217518 two regions + MPRAU)
    for task_key, path in (("half_life_5utr", MNT / "experiments/analysis_saluki_frozen_gse217518_20260903/predictions.jsonl"),):
        if path.exists():
            report["rows"][f"saluki__{task_key}"] = eval_at_ks(task_key, load_predictions(path))
    # saluki gse217518 file covers both regions: evaluate both keys on same file
    saluki_hl = MNT / "experiments/analysis_saluki_frozen_gse217518_20260903/predictions.jsonl"
    if saluki_hl.exists():
        preds = load_predictions(saluki_hl)
        for task_key in ("half_life_5utr", "half_life_3utr"):
            report["rows"][f"saluki__{task_key}"] = eval_at_ks(task_key, preds)
            print(f"[k] saluki {task_key}: done", flush=True)
    saluki_mprau = MNT / "experiments/analysis_saluki_frozen_mprau_20260903/predictions.jsonl"
    if saluki_mprau.exists():
        report["rows"]["saluki__mprau"] = eval_at_ks("mprau_encsr854ruf", load_predictions(saluki_mprau))
        print("[k] saluki mprau: done", flush=True)

    # 3) critic V5 headline rows (single predictions file covering all studies)
    def canonical_ids(path: Path) -> set[str]:
        ids = set()
        with open(path) as handle:
            for line in handle:
                ids.add(str(json.loads(line)["canonical_record_id"]))
        return ids

    if V5_PRED:
        preds_all = load_predictions(V5_PRED[0])

        def eval_v5_task(task_key: str, cpath: Path) -> None:
            spec = te.TASKS[task_key]
            ids = canonical_ids(cpath) & set(preds_all)
            if not ids:
                return
            observations = ev.load_observations([cpath], ids)
            # stratify by the task spec (region, endpoint): the V5 file spans all tasks
            wanted = (spec["region"], spec["endpoint"])
            obs_f = [o for o in observations if o["task"] == wanted]
            if not obs_f:
                return
            task_preds = {o["canonical_record_id"]: preds_all[o["canonical_record_id"]] for o in obs_f}
            out = {}
            for k in KS:
                m = ev.evaluate(obs_f, task_preds, k)
                out[f"K={k}"] = {"ndcg_at_k": m.get("source_macro_ndcg_at_k"),
                                 "top_1": m.get("source_macro_top_1_accuracy"),
                                 "task_macro_spearman": m.get("task_macro_spearman")}
            report["rows"][f"critic_v5__{task_key}"] = out
            print(f"[k] critic_v5 {task_key}: done", flush=True)

        for task_key in ("polya_gse269595", "gse200304_te", "gse149487_te", "gse149487_rna", "gse186455",
                         "half_life_5utr", "half_life_3utr", "mprau_encsr854ruf"):
            eval_v5_task(task_key, te.canonical_path_for(te.TASKS[task_key]["study"]))
        # V5 MRL row (GSE114002 not in te.TASKS -> manual path, single-endpoint study)
        mrl_canonical = MNT / "canonical/GSE114002/v1/canonical_records.private.jsonl"
        ids = canonical_ids(mrl_canonical) & set(preds_all)
        if ids:
            observations = ev.load_observations([mrl_canonical], ids)
            task_preds = {obs["canonical_record_id"]: preds_all[obs["canonical_record_id"]] for obs in observations}
            out = {}
            for k in KS:
                m = ev.evaluate(observations, task_preds, k)
                out[f"K={k}"] = {"ndcg_at_k": m.get("source_macro_ndcg_at_k"),
                                 "top_1": m.get("source_macro_top_1_accuracy"),
                                 "task_macro_spearman": m.get("task_macro_spearman")}
            report["rows"]["critic_v5__mrl_gse114002"] = out
            print("[k] critic_v5 mrl: done", flush=True)

    (OUT / "k_sensitivity.json").write_text(json.dumps(report, indent=1))
    # markdown summary: ndcg@K per row
    md = ["# NDCG@K 敏感性分析（P0-4，冻结评估器重算，VALIDATION）", "",
          "| row | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | top-1 (K 不变) | Spearman (K 不变) |",
          "|---|---|---|---|---|---|---|"]
    for name, ks in sorted(report["rows"].items()):
        md.append("| %s | %s | %s | %s | %s | %.4f | %.4f |" % (
            name,
            *[("%.4f" % ks[f"K={k}"]["ndcg_at_k"]) if ks[f"K={k}"]["ndcg_at_k"] is not None else "—" for k in KS],
            ks["K=10"]["top_1"] or 0, ks["K=10"]["task_macro_spearman"] or 0))
    md += ["", "缺口（P0-4b）：Optimus/FramePool（MRL）、APARENT/APARENT2（polyA）逐条预测未持久化，需 GPU 重打分。"]
    (OUT / "k_sensitivity.md").write_text("\n".join(md))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
