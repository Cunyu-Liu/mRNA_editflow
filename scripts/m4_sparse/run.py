"""M4 SparseEditFormer end-to-end runner.

Trains the source-relative effect model on each S4 fold (A1-NATURAL: 5U-A1 +
3U-A1; no A2 dense pretraining), evaluates on the held-out study, runs the
abs_candidate reference baseline on the same folds, and runs the top-K paired
reranker on measured candidate pools.  Honest CUDA policy: if CUDA is
unavailable, NN training/eval is recorded as not_run with a reason (never
silently run on CPU and claimed as GPU).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from scripts.m4_sparse import config as C
from scripts.m4_sparse.dataset import build_vocab
from scripts.m4_sparse.evaluate import run_fold_evaluation, macro_over_folds
from scripts.m4_sparse.train import (
    build_folds, train_fold, calibrate_sign_temperature,
    grouped_train_dev_split,
)
from scripts.m4_sparse.ordinary_io import assert_ordinary_path, load_ordinary_jsonl
from scripts.m4_sparse.checkpoint import load_checkpoint, vocab_sha256


def load_rows(path):
    return load_ordinary_jsonl(path, require_delta=True)


def select_device(cfg, override):
    import torch
    if not torch.cuda.is_available():
        return None, "CUDA not available on this host (torch.cuda.is_available()=False)"
    if override:
        idx = override.split(":")[-1]
        if idx in cfg.FORBIDDEN_DEVICES:
            return None, "requested GPU " + override + " is forbidden"
        return torch.device(override), None
    for dev in cfg.CUDA_DEVICES:
        idx = dev.split(":")[-1]
        if idx in cfg.FORBIDDEN_DEVICES:
            continue
        return torch.device(dev), None
    if "0" not in cfg.FORBIDDEN_DEVICES:
        return torch.device("cuda:0"), None
    return None, "no permitted CUDA device"


def _npavg(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _fmt(v):
    return "n/a" if v is None else ("%.4f" % v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path(C.DATASET))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/m4_sparse"))
    ap.add_argument("--benchmarks", nargs="+", default=["5U-A1", "3U-A1"])
    ap.add_argument("--target", default=None, choices=["delta", "candidate_value"])
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only-eval", action="store_true")
    args = ap.parse_args()
    assert_ordinary_path(args.out_dir, purpose="ordinary output")

    cfg = C.get_config()
    if args.target:
        cfg.TARGET = args.target
    if args.limit:
        cfg.LIMIT = args.limit

    rows = load_rows(args.dataset)
    if args.limit:
        rows = rows[:args.limit]
    print("loaded %d delta-defined records" % len(rows))

    device, err = select_device(cfg, args.gpu)
    results = {
        "seed": cfg.SEED, "split": cfg.PRIMARY_SPLIT, "target": cfg.TARGET,
        "n_records": len(rows), "benchmarks": {}, "gpu": None,
        "cuda_available": (device is not None),
    }
    if device is None:
        results["gpu_status"] = "not_run:" + err
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "results.json").write_text(json.dumps(results, indent=2))
        print("CUDA unavailable -> NN components not_run: " + err)
        return 0

    results["gpu"] = str(device)
    print("using device " + str(device))

    for benchmark in args.benchmarks:
        folds = build_folds(rows, benchmark, split=cfg.PRIMARY_SPLIT)
        fold_evals = []
        for fold in folds:
            if len(fold["train"]) == 0:
                print("[" + benchmark + "/" + fold["held_out_study"] + "] EMPTY TRAIN -> not_run")
                fold_evals.append({"held_out_study": fold["held_out_study"],
                                   "n_train": 0, "n_test": len(fold["test"]),
                                   "model": [], "abs_candidate": [],
                                   "temperature": None,
                                   "rerank": {"n_sources": 0, "n_singleton_sources": 0,
                                              "macro_ndcg_at_10": None,
                                              "macro_top_decile_recall": None}})
                continue
            ckpt = args.out_dir / ("model_%s__%s.pt" % (benchmark, fold["held_out_study"]))
            fold_vocab = build_vocab(fold["train"])
            fold_cfg = copy.deepcopy(cfg)
            fold_cfg.N_STUDIES = len(fold_vocab["study"])
            fold_cfg.N_ENDPOINTS = len(fold_vocab["endpoint"])
            fold_cfg.N_BENCHMARKS = len(fold_vocab["benchmark"])
            t0 = time.time()
            if args.only_eval and ckpt.exists():
                from scripts.m4_sparse.model import SparseEditFormer
                sd = load_checkpoint(ckpt, device, expected_vocab=fold_vocab)
                model_cfg = sd["cfg"]
                model = SparseEditFormer(model_cfg).to(device)
                model.load_state_dict(sd["state_dict"])
                model.rank_supervision = {
                    "status": sd["rank_supervision_status"],
                    "qualified_rank_pair_count": sd["qualified_rank_pair_count"],
                    "eligible_rank_group_count": sd["eligible_rank_group_count"],
                    "pair_definition": sd["rank_pair_definition"],
                    "score_definition": sd["rank_score_definition"],
                }
                model.rank_publication_qualification = sd[
                    "rank_publication_qualification"]
                model.development_scope = sd["development_scope"]
                print("[" + benchmark + "/" + fold["held_out_study"] + "] loaded checkpoint")
            else:
                model = train_fold(fold, fold_cfg, fold_vocab, device, out_path=ckpt)
                model_cfg = fold_cfg
                print("[" + benchmark + "/" + fold["held_out_study"] + "] trained in %.1fs train=%d test=%d"
                      % (time.time() - t0, len(fold["train"]), len(fold["test"])))
            T = float(getattr(model_cfg, "SIGN_TEMPERATURE", 1.0))
            fe = run_fold_evaluation(model, fold, fold_vocab, model_cfg, device)
            fe["temperature"] = T
            fe["vocab_sha256"] = vocab_sha256(fold_vocab)
            fe["development_scope"] = getattr(
                model, "development_scope", {
                    "claim_status": "DEVELOPMENT_ONLY_UNKNOWN_SCOPE"})
            fe["rank_publication_qualification"] = getattr(
                model, "rank_publication_qualification", {
                    "status": "HOLD_MISSING_OR_FAILED_PUBLICATION_QUALIFICATION"})
            fe["effect_gate_is_not_publication_unlock"] = True
            rank_meta = getattr(model, "rank_supervision", {})
            publication_meta = getattr(
                model, "rank_publication_qualification", {})
            development_scope = getattr(model, "development_scope", {})
            # R3-P0 intentionally has no active numeric publication policy.
            # Even non-zero development pairs therefore cannot make this
            # branch executable.  Retain the metadata for audit, but never
            # emit a post-hoc reranking result that looks official.
            fe["rerank"] = {
                "status": (
                    "HOLD_REQUIRES_PREREGISTERED_RANK_QUALIFICATION_POLICY"),
                "rank_supervision_status": rank_meta.get("status"),
                "qualified_rank_pair_count": int(
                    rank_meta.get("qualified_rank_pair_count", 0)),
                "publication_qualification_status": publication_meta.get(
                    "status"),
                "development_claim_status": development_scope.get(
                    "claim_status"),
                "n_sources": 0,
                "n_singleton_sources": 0,
                "macro_ndcg_at_10": None,
                "macro_top_decile_recall": None,
            }
            fold_evals.append(fe)

        bench_res = {
            "benchmark": benchmark,
            "studies": [f["held_out_study"] for f in folds],
            "folds": [{"held_out_study": f["held_out_study"], "n_train": len(f["train"]),
                       "n_test": len(f["test"])} for f in folds],
            "model": macro_over_folds(fold_evals, "model"),
            "abs_candidate": macro_over_folds(fold_evals, "abs_candidate"),
            "rerank": {
                "n_sources": sum(fe["rerank"]["n_sources"] for fe in fold_evals),
                "n_singleton_sources": sum(fe["rerank"]["n_singleton_sources"] for fe in fold_evals),
                "macro_ndcg_at_10": _npavg([fe["rerank"]["macro_ndcg_at_10"] for fe in fold_evals]),
                "macro_top_decile_recall": _npavg([fe["rerank"]["macro_top_decile_recall"] for fe in fold_evals]),
            },
            "fold_evals": fold_evals,
        }
        results["benchmarks"][benchmark] = bench_res
        print("\n=== %s ===" % benchmark)
        for blk in ("model", "abs_candidate"):
            m = bench_res[blk]
            print("%-14s spearman=%s signacc=%s top10=%s top10pct=%s"
                  % (blk, _fmt(m["macro_delta_spearman"]), _fmt(m["macro_sign_accuracy"]),
                     _fmt(m["macro_top10_enrichment"]), _fmt(m["macro_top10pct_enrichment"])))
        rk = bench_res["rerank"]
        print("rerank      ndcg@10=%s topdecile=%s sources=%d singleton=%d"
              % (_fmt(rk["macro_ndcg_at_10"]), _fmt(rk["macro_top_decile_recall"]),
                 rk["n_sources"], rk["n_singleton_sources"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print("wrote %s" % out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
